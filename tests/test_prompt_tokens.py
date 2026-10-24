"""Prompt token counter (台帳 §1-111): POST /utils/prompt-tokens and its parts.

* services/gguf_kv.read_gguf_tensor_bytes on synthetic GGUFs,
* the three engines' counting rules (services/engines/*/adapter.py
  ``count_prompt_tokens``) on tiny hand-built tokenizers,
* the adapters' copies of engine constants, read out of the engine sources
  with ``ast`` (the engine packages import torch, which the app venv lacks),
* the one-tokenizer cache (services/prompt_tokens.py),
* the endpoint: shape, the clip join (api.models.compose_clip_prompt), the
  always-200 null on failure and the base model / text encoder it follows,
* ``constraints.prompt_token_limit`` and the access-log filter.

Real tokenizers are not used here; the counts on the shipped assets were
checked against the inference paths (Docs/TOKEN_COUNTER_RESEARCH_NOTES.md).
"""

from __future__ import annotations

import ast
import logging
import os
import struct
import sys
from pathlib import Path

import pytest

from api.models import GenerateChainRequest, compose_clip_prompt
from services import engines, prompt_tokens
from services.engines.h3 import adapter as h3
from services.engines.ltx import adapter as ltx
from services.engines.ltx25 import adapter as ltx25
from services.gguf_kv import GgufParseError, read_gguf_tensor_bytes

REPO = Path(__file__).resolve().parents[1]
GGML_I8 = 24
GGML_F32 = 0


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(prompt_tokens, "_cached", None)


# --------------------------------------------------------------------------- #
# synthetic files
# --------------------------------------------------------------------------- #


def _gstr(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def write_gguf_tensors(
    path: Path,
    tensors: list[tuple[str, int, bytes]],
    *,
    alignment: int | None = None,
) -> Path:
    """A GGUF with real tensor data: (name, ggml type, bytes), 1-D each.

    Writes ``general.alignment`` (u32) when given; the data section starts at
    the end of the tensor-info table rounded up to the alignment (32 when
    absent), and every tensor's offset is aligned too, as GGUF lays them out.
    """
    align = alignment or 32
    kv = b""
    kv_count = 0
    if alignment is not None:
        kv += _gstr("general.alignment") + struct.pack("<II", 4, alignment)
        kv_count += 1
    kv += _gstr("general.architecture") + struct.pack("<I", 8) + _gstr("gemma4")
    kv_count += 1
    infos = b""
    data = b""
    for name, ggml_type, payload in tensors:
        data += b"\0" * (-len(data) % align)
        infos += _gstr(name) + struct.pack("<IQIQ", 1, len(payload), ggml_type, len(data))
        data += payload
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), kv_count) + kv + infos
    head += b"\0" * (-len(head) % align)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(head + data)
    return path


VOCAB = {"<unk>": 0, "<bos>": 1, "a": 2, "cat": 3, "runs": 4, "<": 5, ">": 6, "d": 7, "/": 8}


def _tokenizer(*, bos_template: bool):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    tok = Tokenizer(models.WordLevel(VOCAB, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["<bos>"])
    if bos_template:
        tok.post_processor = processors.TemplateProcessing(
            single="<bos> $A", special_tokens=[("<bos>", 1)]
        )
    return tok


def write_tokenizer_json(path: Path, *, bos_template: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _tokenizer(bos_template=bos_template).save(str(path))
    return path


# --------------------------------------------------------------------------- #
# GGUF tensor bytes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alignment", [None, 32, 64])
def test_reads_an_i8_tensor_at_the_aligned_offset(tmp_path, alignment):
    p = write_gguf_tensors(
        tmp_path / "te.gguf",
        [("weight", GGML_F32, b"\1" * 12), ("tokenizer_json", GGML_I8, b'{"x": 1}')],
        alignment=alignment,
    )
    assert read_gguf_tensor_bytes(p, "tokenizer_json") == b'{"x": 1}'


def test_tensor_bytes_refuses_absent_and_non_i8(tmp_path):
    p = write_gguf_tensors(tmp_path / "te.gguf", [("weight", GGML_F32, b"\1" * 8)])
    with pytest.raises(GgufParseError, match="not found"):
        read_gguf_tensor_bytes(p, "tokenizer_json")
    with pytest.raises(GgufParseError, match="only I8"):
        read_gguf_tensor_bytes(p, "weight")


def test_tensor_bytes_refuses_a_non_gguf(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(GgufParseError, match="bad magic"):
        read_gguf_tensor_bytes(p, "tokenizer_json")


# --------------------------------------------------------------------------- #
# the three counting rules
# --------------------------------------------------------------------------- #


def _no_te():
    raise AssertionError("this engine must not resolve the text encoder")


def _assets(mapping: dict[str, Path]):
    return lambda key: mapping[key]


def test_ltx23_strips_and_counts_the_bos(tmp_path):
    write_tokenizer_json(tmp_path / "gemma" / "tokenizer.json", bos_template=True)
    counts = ltx.count_prompt_tokens(
        ["  a cat runs \n", "", "   ", "cat"],
        assets=_assets({"gemma_root": tmp_path / "gemma"}),
        text_encoder=_no_te,
    )
    assert counts == [4, 1, 1, 2]


def test_ltx25_reads_the_selected_gguf_and_adds_a_missing_bos(tmp_path):
    tok_json = _tokenizer(bos_template=False).to_str().encode("utf-8")
    te = write_gguf_tensors(tmp_path / "te.gguf", [(ltx25.TOKENIZER_JSON_TENSOR, GGML_I8, tok_json)])

    def no_assets(key):
        raise AssertionError("LTX 2.5 reads no fixed asset")

    counts = ltx25.count_prompt_tokens(
        ["  a cat runs ", "", "<bos> a cat", "  <bos>"],
        assets=no_assets,
        text_encoder=lambda: te,
    )
    # A text already starting with <bos> gets none added (the pinned ltx-core).
    assert counts == [4, 1, 3, 1]


def test_ltx25_without_a_bos_token_cannot_count(tmp_path):
    from tokenizers import Tokenizer, models

    tok_json = Tokenizer(models.WordLevel({"<unk>": 0, "a": 1}, unk_token="<unk>")).to_str()
    te = write_gguf_tensors(tmp_path / "te.gguf", [("tokenizer_json", GGML_I8, tok_json.encode())])
    with pytest.raises(RuntimeError, match="<bos>"):
        ltx25.count_prompt_tokens(["a"], assets=_assets({}), text_encoder=lambda: te)


def test_h3_adds_the_seven_specials_and_neither_strips_nor_adds_tokens(tmp_path):
    write_tokenizer_json(tmp_path / "qwen" / "tokenizer.json", bos_template=True)
    counts = h3.count_prompt_tokens(
        ["<d>a cat</d>", "", "<|cutoff|>"],
        assets=_assets({"tokenizer_dir": tmp_path / "qwen"}),
        text_encoder=_no_te,
    )
    # "<d>" and "</d>" are single tokens; the template's BOS is not added.
    assert counts == [4, 0, 1]
    tok = prompt_tokens.tokenizer_file(
        tmp_path / "qwen" / "tokenizer.json", h3.EXTRA_SPECIAL_TOKENS
    )
    first = len(VOCAB)
    assert [tok.token_to_id(t) for t in h3.EXTRA_SPECIAL_TOKENS] == list(
        range(first, first + len(h3.EXTRA_SPECIAL_TOKENS))
    )


# --------------------------------------------------------------------------- #
# the adapters' copies of engine constants
# --------------------------------------------------------------------------- #


def _module_constant(relpath: str, name: str):
    tree = ast.parse((REPO / relpath).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {relpath}")


def test_h3_specials_are_the_engines():
    assert h3.EXTRA_SPECIAL_TOKENS == _module_constant(
        "engine_h3/text_encoder.py", "EXTRA_SPECIAL_TOKENS"
    )


def test_ltx25_tensor_name_is_the_engines():
    assert ltx25.TOKENIZER_JSON_TENSOR == _module_constant(
        "engine25/assets_export.py", "TOKENIZER_JSON_TENSOR_KEY"
    )


# --------------------------------------------------------------------------- #
# one cached tokenizer
# --------------------------------------------------------------------------- #


def test_the_tokenizer_is_cached_until_its_key_changes(tmp_path):
    a = write_tokenizer_json(tmp_path / "a" / "tokenizer.json")
    b = write_tokenizer_json(tmp_path / "b" / "tokenizer.json")
    first = prompt_tokens.tokenizer_file(a)
    assert prompt_tokens.tokenizer_file(a) is first
    assert prompt_tokens.tokenizer_file(a, ("<d>",)) is not first
    other = prompt_tokens.tokenizer_file(b)
    assert other is not first
    stat = a.stat()
    os.utime(a, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert prompt_tokens.tokenizer_file(a) is not first


def test_missing_tokenizers_package_says_rerun_setup(tmp_path, monkeypatch):
    p = write_tokenizer_json(tmp_path / "tokenizer.json")
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    with pytest.raises(RuntimeError, match="setup.bat"):
        prompt_tokens.tokenizer_file(p)


# --------------------------------------------------------------------------- #
# compose_clip_prompt is clip_prompt's rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shared,own,expected",
    [
        ("a lake", None, "a lake"),
        ("a lake", "", "a lake"),
        ("a lake", "  \n ", "a lake"),
        ("  a lake  ", " mist ", "a lake mist"),
        ("   ", "birds", "birds"),
        ("a lake", "line one\nline two", "a lake line one\nline two"),
        ("", None, ""),
    ],
)
def test_compose_clip_prompt_is_clip_prompts_rule(shared, own, expected):
    assert compose_clip_prompt(shared, own) == expected
    chain = GenerateChainRequest.model_construct(
        prompt=shared, clips=[type("Clip", (), {"prompt": own})()]
    )
    assert chain.clip_prompt(0) == expected


# --------------------------------------------------------------------------- #
# the endpoint
# --------------------------------------------------------------------------- #

URL = "/api/v1/utils/prompt-tokens"


def test_endpoint_shape_and_clip_join(client, monkeypatch):
    seen = []

    def fake(texts, *, assets, text_encoder):
        seen.append(list(texts))
        return [len(t) for t in texts]

    monkeypatch.setattr(ltx, "count_prompt_tokens", fake)

    r = client.post(URL, json={"prompt": "abc"})
    assert r.status_code == 200
    assert r.json() == {"prompt": 3, "negative_prompt": None, "clips": None}

    r = client.post(
        URL, json={"prompt": " abc ", "negative_prompt": "", "clips": ["", " de ", "f\ng"]}
    )
    assert r.json() == {"prompt": 5, "negative_prompt": 0, "clips": [3, 6, 7]}
    assert seen[-1] == [" abc ", "", "abc", "abc de", "abc f\ng"]

    r = client.post(URL, json={"prompt": "abc", "clips": []})
    assert r.json() == {"prompt": 3, "negative_prompt": None, "clips": []}


def test_endpoint_answers_null_and_logs_once_when_counting_fails(client, monkeypatch, caplog):
    # The test descriptor declares no gemma_root: the LTX 2.3 counter raises.
    from api import prompt_tokens as endpoint

    monkeypatch.setattr(endpoint, "_logged_failures", set())
    with caplog.at_level(logging.WARNING, logger="ltx.prompt_tokens"):
        for _ in range(3):
            r = client.post(URL, json={"prompt": "a", "negative_prompt": "b", "clips": ["c"]})
            assert r.status_code == 200
            assert r.json() == {"prompt": None, "negative_prompt": None, "clips": None}
    lines = [rec for rec in caplog.records if rec.name == "ltx.prompt_tokens"]
    assert len(lines) == 1 and "gemma_root" in lines[0].getMessage()


def test_endpoint_null_for_a_family_without_a_counter(client, monkeypatch):
    monkeypatch.delattr(ltx, "count_prompt_tokens")
    r = client.post(URL, json={"prompt": "a", "clips": ["b"]})
    assert r.json() == {"prompt": None, "negative_prompt": None, "clips": None}


def test_endpoint_requires_auth(client):
    client.app_context.runtime.api_key = "secret"
    try:
        assert client.post(URL, json={"prompt": "a"}).status_code == 401
        r = client.post(URL, json={"prompt": "a"}, headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200
    finally:
        client.app_context.runtime.api_key = None


def _activate(client, base_model: str) -> None:
    client.post("/api/v1/pipeline/unload")
    r = client.post("/api/v1/pipeline/load", json={"base_model": base_model})
    assert r.status_code == 200, r.text
    assert client.app_context.pipeline_manager.active_base_model == base_model


def test_endpoint_follows_the_active_base_model_and_text_encoder(three_family_client):
    client = three_family_client
    models = client.app_context.config.models_dir

    # LTX 2.5: the tokenizer comes from the SELECTED text-encoder GGUF.
    ltx25_te = models / "LTX25" / "TextEncoder" / "gemma-3-12b-it-Q4_K_M.gguf"
    write_gguf_tensors(
        ltx25_te,
        [("tokenizer_json", GGML_I8, _tokenizer(bos_template=False).to_str().encode())],
    )
    _activate(client, "LTX25")
    r = client.post(URL, json={"prompt": " a cat ", "clips": ["runs"]})
    assert r.json() == {"prompt": 3, "negative_prompt": None, "clips": [4]}

    # MiniMax H3: the descriptor's tokenizer folder; no BOS, no strip.
    write_tokenizer_json(models / "MiniMaxH3" / "TextEncoder" / "qwen3vl_tokenizer" / "tokenizer.json")
    _activate(client, "MiniMaxH3")
    r = client.post(URL, json={"prompt": "<d>a cat</d>", "negative_prompt": ""})
    assert r.json() == {"prompt": 4, "negative_prompt": 0, "clips": None}


# --------------------------------------------------------------------------- #
# constraints and the access log
# --------------------------------------------------------------------------- #


def test_prompt_token_limit_is_published(client):
    assert engines.request_constraints("ltx")["prompt_token_limit"] == 1024
    assert engines.request_constraints("ltx25")["prompt_token_limit"] == 1024
    assert engines.request_constraints("h3")["prompt_token_limit"] is None
    base = client.get("/api/v1/models").json()["base_models"][0]
    assert base["constraints"]["prompt_token_limit"] == 1024


@pytest.mark.parametrize(
    "args,expected",
    [
        (("127.0.0.1:5000", "POST", "/api/v1/utils/prompt-tokens", "1.1", 200), False),
        (("127.0.0.1:5000", "POST", "/api/v1/utils/prompt-tokens/", "1.1", 200), False),
        (("127.0.0.1:5000", "POST", "/api/v1/utils/prompt-tokens", "1.1", 401), True),
        (("127.0.0.1:5000", "POST", "/api/v1/utils/prompt-tokens", "1.1", 422), True),
        (("127.0.0.1:5000", "GET", "/api/v1/utils/prompt-tokens", "1.1", 200), True),
        (("127.0.0.1:5000", "POST", "/api/v1/utils/mp4-info", "1.1", 200), True),
        (("127.0.0.1:5000", "POST", 42, "1.1", 200), True),
        (("only", "three", "slots"), True),
    ],
)
def test_prompt_tokens_access_filter(args, expected):
    from main import PromptTokensAccessFilter

    record = logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=0,
        msg='%s - "%s %s HTTP/%s" %d', args=args, exc_info=None,
    )
    assert PromptTokensAccessFilter().filter(record) is expected
