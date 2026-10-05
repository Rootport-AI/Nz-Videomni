"""§1-31: ``GET /models`` names the weight class of the selected transformer
(``base_models[].transformer_weight_class``).

Only the ACTIVE base model gets a value; every other base model, a file that
cannot be classified, and a selection that does not resolve are ``null``.
``entries[]`` and the top-level ``categories`` block are not touched (their
key sets are pinned by tests/test_model_registry.py,
tests/test_models_endpoint_compat.py and tests/test_ltx25_api_guard.py).
"""

from __future__ import annotations

import pytest

from conftest import write_gguf_with_tensors
from services import weight_class

Q4_K = 12


def _q4_tensors(n: int = 6) -> dict[str, tuple[int, int]]:
    return {f"transformer_blocks.{i}.attn1.to_q.weight": (2, Q4_K) for i in range(n)}


@pytest.fixture(autouse=True)
def _fresh_cache():
    weight_class._classify_cached.cache_clear()
    yield
    weight_class._classify_cached.cache_clear()


def _base(client, base_id: str) -> dict:
    body = client.get("/api/v1/models").json()
    return next(b for b in body["base_models"] if b["id"] == base_id)


def _default_transformer(client, base_id: str):
    ctx = client.app_context
    return ctx.model_registry.resolve("transformer", "default", base_model=base_id)


def test_every_base_model_carries_the_key(two_family_client):
    body = two_family_client.get("/api/v1/models").json()
    for block in body["base_models"]:
        assert "transformer_weight_class" in block
    for category in body["categories"].values():
        for entry in category["entries"]:
            assert "transformer_weight_class" not in entry


def test_empty_stub_is_null(client):
    """The hermetic stub GGUF has no tensors -> cannot be classified -> null."""
    assert _base(client, "LTX23")["transformer_weight_class"] is None


def test_gguf_with_q4_k_tensors_is_4bit(client):
    write_gguf_with_tensors(_default_transformer(client, "LTX23"), _q4_tensors())
    assert _base(client, "LTX23")["transformer_weight_class"] == "4bit"


def test_missing_file_is_null(client):
    _default_transformer(client, "LTX23").unlink()
    assert _base(client, "LTX23")["transformer_weight_class"] is None


def test_inactive_base_model_is_null(two_family_client):
    """LTX25 is installed with a classifiable default file, but is not the
    active base model: no guess from its default file."""
    write_gguf_with_tensors(
        _default_transformer(two_family_client, "LTX25"),
        _q4_tensors(),
        **{"general.architecture": "ltxv", "model_version": "2.5.0"},
    )
    assert _base(two_family_client, "LTX23")["active"] is True
    assert _base(two_family_client, "LTX25")["transformer_weight_class"] is None
