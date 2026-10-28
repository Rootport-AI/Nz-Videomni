"""Ref2VA reference media on the CPU (台帳 §1-107, plan §3.1): engine_h3.reference,
the text encoder's reference presentation, and the video encoder's ``encode_clips``.

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_reference_media.py

What is pinned:
  * the sizes: ``match_size`` (scale down to the generation's area, sides to
    multiples of 32) and ``video_canvas`` (768 short edge, 768 x 1344 area);
  * the 17k + 5 cut, the 2-per-second Qwen pairs and their time labels
    (15 s: 362 frames -> 16 pairs labelled 0.2 ... 14.2, 15.0; 5 s: 124 -> 6);
  * ``read_reference_video`` on synthetic mp4 files: a first PTS that is not 0
    keeps its first frame; a 30 fps file is sampled at 24 fps by the last frame
    at or before each k/24; the caps (num_frames, max_sec, 15 s);
  * ``pair_patches`` of two equal frames equals the Hugging Face
    ``Qwen2VLImageProcessor`` built from the product's tokenizer folder
    (>= 65 536 pixels, so the processor does not resize);
  * ``presentation`` without references gives the same ids as the code before
    台帳 §1-107 (a copy below); with references, labels / time labels / tags /
    vision blocks are where they belong;
  * ``encode_clips`` equals ``encode`` on a tiny random encoder.
"""

from __future__ import annotations

import fractions
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from engine_h3 import reference as R  # noqa: E402
from engine_h3 import video_vae as V  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT / "models" / "MiniMaxH3" / "TextEncoder" / "qwen3vl_tokenizer"


def _tokenizer_dir() -> Path:
    if not ((TOKENIZER_DIR / "tokenizer.json").is_file() and (TOKENIZER_DIR / "preprocessor_config.json").is_file()):
        pytest.skip("no Qwen3-VL tokenizer folder present")
    return TOKENIZER_DIR


# -- sizes ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "src, gen, want",
    [
        ((1920, 1080), (1344, 768), (1344, 768)),    # same aspect, scaled down to the area
        ((1000, 1000), (1344, 768), (992, 992)),     # sqrt(1344*768/1e6) = 1.016 -> 1 (never up); 1000/32 = 31.25 -> 992
        ((1010, 1010), (1344, 768), (1024, 1024)),   # nearest multiple: 1010/32 = 31.56 -> 1024
        ((640, 480), (1344, 768), (640, 480)),       # smaller than the generation: kept
        ((4000, 1000), (1344, 768), (2016, 512)),    # scale 0.508 -> 2032 x 508 -> 2016 x 512
        ((10, 10), (1344, 768), (32, 32)),           # at least 32
    ],
)
def test_match_size(src, gen, want):
    got = R.match_size(*src, *gen)
    assert got == want
    assert all(v % 32 == 0 for v in got)


@pytest.mark.parametrize(
    "src, want",
    [
        ((1920, 1080), (1344, 768)),
        ((1080, 1920), (768, 1344)),
        ((3840, 2160), (1344, 768)),
        ((1024, 1024), (768, 768)),     # square: 768 x 768 (area under the cap)
        ((640, 360), (640, 352)),       # smaller than the canvas: its own size to multiples of 32
        ((2560, 1080), (1568, 672)),    # wide: area cap 768*1344 -> 1564 x 660 -> 1568 x 672
    ],
)
def test_video_canvas(src, want):
    got = R.video_canvas(*src)
    assert got == want
    assert got[0] * got[1] <= R.REF_VIDEO_MAX_AREA + 32 * 1400   # rounding may pass the cap by under one row of 32


@pytest.mark.parametrize("n, want", [(0, 0), (4, 0), (5, 5), (21, 5), (22, 22), (124, 124), (130, 124), (360, 345), (362, 362)])
def test_video_frame_count(n, want):
    assert R.video_frame_count(n) == want
    if want:
        assert (want - 5) % 17 == 0


# -- Qwen pairs -----------------------------------------------------------------


def _numbered_frames(n: int, h: int = 32, w: int = 32) -> np.ndarray:
    """uint8 frames whose first pixel holds the index (mod 256) and second pixel index // 256."""
    f = np.zeros((n, h, w, 3), np.uint8)
    for i in range(n):
        f[i, 0, 0, 0] = i % 256
        f[i, 0, 1, 0] = i // 256
    return f


def _index(frame: np.ndarray) -> int:
    return int(frame[0, 0, 0]) + 256 * int(frame[0, 1, 0])


def test_qwen_pairs_15_seconds():
    pairs = R.qwen_pairs(_numbered_frames(362))
    assert len(pairs) == 16
    picked = [(_index(p[0]), _index(p[1])) for p, _ in pairs]
    assert picked[0] == (0, 12) and picked[-2] == (336, 348) and picked[-1] == (360, 360)   # 31 picked, last repeated
    labels = [label for _, label in pairs]
    assert labels == [f"<{i + 0.2:.1f} seconds>" for i in range(15)] + ["<15.0 seconds>"]
    assert labels[0] == "<0.2 seconds>" and labels[1] == "<1.2 seconds>" and labels[14] == "<14.2 seconds>"
    assert all(p.shape == (2, 32, 32, 3) and p.dtype == np.uint8 for p, _ in pairs)


def test_qwen_pairs_5_seconds():
    pairs = R.qwen_pairs(_numbered_frames(124))
    assert len(pairs) == 6   # 0..120 step 12 -> 11 picked -> 12
    assert [(_index(p[0]), _index(p[1])) for p, _ in pairs][-1] == (120, 120)
    assert pairs[-1][1] == "<5.0 seconds>"


# -- reading a video ----------------------------------------------------------------


def _write_video(path: Path, *, fps: int, n: int, start_pts: int = 0, size=(64, 64)) -> None:
    w, h = size
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps, options={"qp": "0"})
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.time_base = fractions.Fraction(1, fps)
        for i in range(n):
            f = av.VideoFrame.from_ndarray(np.full((h, w, 3), (i * 6) % 250, np.uint8), format="rgb24")
            f.pts = start_pts + i
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def _decoded_means(path: Path) -> tuple[list[float], list[float]]:
    times, means = [], []
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            times.append(float(f.time))
            means.append(float(f.to_ndarray(format="rgb24").mean()))
    return times, means


def _means(frames: np.ndarray) -> list[float]:
    return [float(f.mean()) for f in frames]


def test_read_video_24fps_from_nonzero_start(tmp_path):
    p = tmp_path / "late.mp4"
    _write_video(p, fps=24, n=30, start_pts=12)     # first frame at 0.5 s
    times, means = _decoded_means(p)
    assert times[0] == pytest.approx(0.5)            # the file really starts late
    frames, info = R.read_reference_video(str(p), num_frames=1000)
    assert frames.shape == (22, 64, 64, 3) and frames.dtype == np.uint8   # 30 -> 22 (17k+5)
    assert _means(frames) == pytest.approx(means[:22], abs=0.5)           # the first frame is kept
    assert info["source_frames"] == 30 and info["used_frames"] == 22
    assert info["source_size"] == [64, 64] and info["size"] == [64, 64]


def test_read_video_30fps_takes_the_frame_on_screen(tmp_path):
    p = tmp_path / "thirty.mp4"
    _write_video(p, fps=30, n=30, start_pts=9)      # 1 s at 30 fps, first PTS 0.3 s
    _, means = _decoded_means(p)
    frames, info = R.read_reference_video(str(p), num_frames=1000)
    assert info["source_frames"] == 24 and frames.shape[0] == 22
    want = [means[(k * 30) // 24] for k in range(22)]   # last frame at or before k/24
    assert _means(frames) == pytest.approx(want, abs=0.5)


def test_read_video_caps(tmp_path):
    p = tmp_path / "long.mp4"
    _write_video(p, fps=24, n=60)
    frames, info = R.read_reference_video(str(p), num_frames=40)
    assert frames.shape[0] == 39 and info["source_frames"] == 40           # num_frames, then 17k+5
    frames, info = R.read_reference_video(str(p), num_frames=1000, max_sec=1.0)
    assert frames.shape[0] == 39 and info["source_frames"] == 39           # 24 rounded up to 17k + 5
    with pytest.raises(ValueError, match="短すぎ"):
        R.read_reference_video(str(p), num_frames=4)


@pytest.mark.parametrize("sec, want", [(15, 362), (10, 243), (5, 124), (1.0, 39), (0.1, 5), (0.5, 22)])
def test_frames_for_seconds_rounds_up_to_the_grid(sec, want):
    assert R.frames_for_seconds(sec) == want and R.video_frame_count(want) == want


def test_read_video_max_sec_15_and_5(tmp_path):
    p = tmp_path / "long15.mp4"
    _write_video(p, fps=24, n=370, size=(32, 32))
    frames, info = R.read_reference_video(str(p), num_frames=1000, max_sec=15)
    assert frames.shape[0] == 362 and info["used_frames"] == 362 and info["source_frames"] == 362
    frames, info = R.read_reference_video(str(p), num_frames=1000, max_sec=5)
    assert frames.shape[0] == 124 and info["source_frames"] == 124


def test_read_video_15_second_cap(tmp_path, monkeypatch):
    # 15 s on the 17k + 5 grid: 362 (17 x 21 + 5), not 15 x 24 = 360 (which would cut to 345).
    assert R.REF_VIDEO_MAX_FRAMES == 362 and R.video_frame_count(R.REF_VIDEO_MAX_FRAMES) == 362
    p = tmp_path / "cap.mp4"
    _write_video(p, fps=24, n=30)
    monkeypatch.setattr(R, "REF_VIDEO_MAX_FRAMES", 24)  # the 362-frame cap, scaled down to keep the file small
    frames, info = R.read_reference_video(str(p), num_frames=1000)
    assert info["source_frames"] == 24 and frames.shape[0] == 22


def test_read_video_resizes_to_canvas_and_refuses_missing(tmp_path):
    p = tmp_path / "big.mp4"
    _write_video(p, fps=24, n=6, size=(1920, 1088))
    frames, info = R.read_reference_video(str(p), num_frames=100)
    assert frames.shape == (5, 768, 1344, 3) and info["size"] == [1344, 768]
    with pytest.raises(ValueError, match="見つかりません"):
        R.read_reference_video(str(tmp_path / "none.mp4"), num_frames=100)


def test_load_reference_image(tmp_path):
    from PIL import Image

    p = tmp_path / "ref.png"
    Image.fromarray(np.random.RandomState(0).randint(0, 256, (1080, 1920, 3), np.uint8)).save(p)
    t, info = R.load_reference_image(str(p), 1344, 768)
    assert t.shape == (1, 3, 1, 768, 1344) and t.dtype == torch.float32
    assert info == {"source_size": [1920, 1080], "size": [1344, 768]}
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0
    with pytest.raises(ValueError, match="見つかりません"):
        R.load_reference_image(str(tmp_path / "none.png"), 1344, 768)


# -- Qwen-VL patches ----------------------------------------------------------------


def test_pair_patches_equals_hf_image_processor():
    d = _tokenizer_dir()
    pytest.importorskip("transformers")
    from PIL import Image
    from transformers import AutoImageProcessor

    cfg = json.loads((d / "preprocessor_config.json").read_text(encoding="utf-8"))
    assert tuple(cfg["image_mean"]) == R.QWEN_IMAGE_MEAN and tuple(cfg["image_std"]) == R.QWEN_IMAGE_STD
    processor = AutoImageProcessor.from_pretrained(str(d))
    assert processor.rescale_factor == pytest.approx(R.QWEN_RESCALE_FACTOR)
    img = np.random.RandomState(1).randint(0, 256, (256, 320, 3), np.uint8)   # 81 920 px >= 65 536
    want = processor(images=[Image.fromarray(img)], return_tensors="pt")
    patches, grid = R.pair_patches(np.stack([img, img]))
    assert grid.tolist() == want["image_grid_thw"].tolist() == [[1, 16, 20]]
    assert patches.dtype == torch.float32 and torch.equal(patches, want["pixel_values"])


def test_pair_patches_frame_order_follows_hf_video_layout():
    # Two different frames: within each patch row the channel axis is outer and the
    # time axis inner ([C, T, p, p]), as Qwen2VLVideoProcessor flattens.
    a = np.zeros((32, 32, 3), np.uint8)
    b = np.full((32, 32, 3), 255, np.uint8)
    patches, grid = R.pair_patches(np.stack([a, b]))
    assert grid.tolist() == [[1, 2, 2]] and patches.shape == (4, 3 * 2 * 16 * 16)
    row = patches[0].reshape(3, 2, 16, 16)
    assert torch.all(row[:, 0] == -1.0) and torch.all(row[:, 1] == 1.0)
    with pytest.raises(ValueError):
        R.pair_patches(np.zeros((2, 48, 32, 3), np.uint8))   # not a multiple of 32


# -- presentation ----------------------------------------------------------------------


def _old_presentation(tokenizer, prompt, image_tokens=()):
    """``engine_h3.text_encoder.presentation`` as it was before 台帳 §1-107 (copy)."""
    ids, tags = [], []
    for i, count in enumerate(image_tokens):
        label = list(tokenizer("<Picture {}>: ".format(i + 1), add_special_tokens=False)["input_ids"])
        block = [151652] + [151655] * int(count) + [151653]
        ids += label + block
        tags += [1] * len(label) + [0] * len(block)
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    ids += prompt_ids
    tags += [1] * len(prompt_ids)
    return ids, tags


@pytest.fixture(scope="module")
def tok():
    d = _tokenizer_dir()
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(d))


PROMPT = "A cat walks. <Picture 1> sings <Audio 2>."


@pytest.mark.parametrize("image_tokens", [(), (240,), (240, 100)])
def test_presentation_without_references_is_unchanged(tok, image_tokens):
    from engine_h3.text_encoder import presentation

    want = _old_presentation(tok, PROMPT, image_tokens)
    assert presentation(tok, PROMPT, image_tokens) == want
    assert presentation(tok, PROMPT, image_tokens, ref_items=None) == want
    assert presentation(tok, PROMPT, image_tokens, ref_items=()) == want


def test_presentation_with_references(tok):
    from engine_h3.text_encoder import RefPresentation, mm_token_type_ids, presentation

    def t(s):
        return list(tok(s, add_special_tokens=False)["input_ids"])

    def block(n):
        return [151652] + [151655] * n + [151653]

    items = [
        RefPresentation("picture", 1, (12,)),
        RefPresentation("picture", 3, (6,)),
        RefPresentation("video", 2, (4, 4), ("<0.2 seconds>", "<1.0 seconds>")),
        RefPresentation("audio", 2),
    ]
    ids, tags = presentation(tok, PROMPT, ref_items=items)
    parts = [
        (t("<Picture 1>: "), 1), (block(12), 0),
        (t("<Picture 3>: "), 1), (block(6), 0),
        (t("<Video 2>: "), 1), (t("<0.2 seconds>"), 1), (block(4), 0), (t("<1.0 seconds>"), 1), (block(4), 0),
        (t("<Audio 2>: "), 1),
        (t(PROMPT), 1),
    ]
    assert ids == [i for p, _ in parts for i in p]
    assert tags == [tag for p, tag in parts for _ in p]
    assert t("<Video 2>: ")[0] != 151655 and 151655 not in t("<0.2 seconds>")   # labels are plain text
    # One run of image pads per grid row, in order: HF's get_rope_index takes them that way.
    mm = mm_token_type_ids(ids)
    runs = [len(list(g)) for k, g in __import__("itertools").groupby(mm) if k == 1]
    assert runs == [12, 6, 4, 4]

    audio_only, audio_tags = presentation(tok, PROMPT, ref_items=[RefPresentation("audio", 1)])
    assert audio_only == t("<Audio 1>: ") + t(PROMPT) and set(audio_tags) == {1}

    with pytest.raises(ValueError):
        presentation(tok, PROMPT, ref_items=[RefPresentation("video", 1, (4,), ())])
    with pytest.raises(ValueError):
        presentation(tok, PROMPT, ref_items=[RefPresentation("sound", 1)])


def test_ref_vision_inputs_order_matches_the_tokens(tok):
    """``H3TextEncoder.ref_vision_inputs`` (no weights): grid rows and pixel rows follow the token order."""
    from PIL import Image

    from engine_h3.text_encoder import H3TextEncoder, RefMedia, mm_token_type_ids, presentation

    enc = H3TextEncoder("unused.safetensors", str(_tokenizer_dir()), device="cpu")
    rng = np.random.RandomState(2)
    pic_a = Image.fromarray(rng.randint(0, 256, (256, 256, 3), np.uint8))
    pic_b_arr = rng.randint(0, 256, (256, 320, 3), np.uint8)
    pic_b = torch.from_numpy(pic_b_arr).to(torch.float32).div(255.0).permute(2, 0, 1)[None, :, None]   # load_reference_image's form
    pairs = R.qwen_pairs(rng.randint(0, 256, (22, 64, 96, 3), np.uint8))   # 2 picked -> 1 pair
    refs = [
        RefMedia("picture", 1, image=pic_a),
        RefMedia("video", 1, pairs=tuple(pairs)),
        RefMedia("audio", 1),
        RefMedia("picture", 2, image=pic_b),
    ]
    items, pv, grid, report = enc.ref_vision_inputs(refs)
    assert grid.tolist() == [[1, 16, 16], [1, 4, 6], [1, 16, 20]]
    assert pv.shape == (256 + 24 + 320, 1536)
    assert [i.vision_tokens for i in items] == [(64,), (6,), (), (80,)]
    assert items[1].labels == ("<0.2 seconds>",)
    assert report == [
        {"kind": "picture", "slot": 1, "vision_tokens": 64},
        {"kind": "video", "slot": 1, "pairs": 1, "vision_tokens": 6},
        {"kind": "audio", "slot": 1, "vision_tokens": 0},
        {"kind": "picture", "slot": 2, "vision_tokens": 80},
    ]
    # The tensor picture came back exactly: its rows equal the processor's output for the array.
    alone, _, _ = enc.vision_inputs([Image.fromarray(pic_b_arr)])
    assert torch.equal(pv[280:], alone)
    assert torch.equal(pv[256:280], R.pair_patches(pairs[0][0])[0])
    ids, _ = presentation(tok, "x", ref_items=items)
    mm = mm_token_type_ids(ids)
    runs = [len(list(g)) for k, g in __import__("itertools").groupby(mm) if k == 1]
    assert runs == [int(r[1] * r[2]) // 4 for r in grid.tolist()]

    items, pv, grid, _ = enc.ref_vision_inputs([RefMedia("audio", 1)])
    assert pv is None and grid is None and items[0].kind == "audio"


def test_encode_refuses_key_frames_and_references_at_once():
    """The one refusal of key-frame images with references is ``H3TextEncoder.encode``'s
    (with references it never counts the images' vision tokens, so ``presentation``
    could not see them); it is checked before anything is loaded."""
    from PIL import Image

    from engine_h3.text_encoder import H3TextEncoder, RefMedia

    enc = H3TextEncoder("unused.safetensors", str(_tokenizer_dir()), device="cpu")
    with pytest.raises(ValueError, match="key-frame images and references cannot both"):
        enc.encode(PROMPT, [Image.new("RGB", (64, 64))], refs=[RefMedia("audio", 1)])


# -- encode_clips ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_enc():
    torch.manual_seed(0)
    enc = V.H3VideoEncoder(ch=32).eval()
    with torch.no_grad():
        for p in enc.parameters():
            p.normal_(0.0, 0.02)
    return enc


@pytest.mark.parametrize("num_frames", [5, 22, 39])
def test_encode_clips_equals_encode(tiny_enc, num_frames):
    frames = np.random.RandomState(num_frames).randint(0, 256, (num_frames, 32, 48, 3), np.uint8)
    video = torch.from_numpy(frames).to(torch.float32).div(255.0).permute(3, 0, 1, 2)[None]   # [1, 3, F, H, W]
    with torch.no_grad():
        want = tiny_enc.encode(video)
        got = tiny_enc.encode_clips(frames, device="cpu")
    assert got.shape == want.shape == (1, 24, (num_frames - 5) // 17 * 5 + 2, 2, 3)
    assert torch.equal(got, want)


def test_encode_clips_refuses_bad_input(tiny_enc):
    with pytest.raises(ValueError):
        tiny_enc.encode_clips(np.zeros((1, 32, 32, 3), np.uint8), device="cpu")
    with pytest.raises(ValueError):
        tiny_enc.encode_clips(np.zeros((5, 40, 32, 3), np.uint8), device="cpu")
    with pytest.raises(ValueError):
        tiny_enc.encode_clips(np.zeros((5, 32, 32, 3), np.float32), device="cpu")
