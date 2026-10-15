"""Alpha Gen exit (第 3 弾, 台帳 §1-86) -- POST /jobs/{id}/alpha/rgba | chroma,
GET /jobs/{id}/alpha/rgba | preview | chroma/{color} and ``JobResponse.alpha``.

Mock backend, no GPU, real ffmpeg (skipped without it on PATH; put
``tools\\ffmpeg\\bin`` first). An Alpha Gen job is run to completion the way
``tests/test_generate_alpha_api.py`` does it, from a source WITH audio, so the
job dir has ``_alpha_window.mp4`` (with sound), ``matte.mkv`` and a
``metadata.json`` carrying the ``alpha_gen`` block. The mock's matte is then
replaced by a known one -- four vertical bands 250 / 128 / 100 / 0 -- so the
alpha of the composite can be checked against the thresholds.

The pixel checks of every levels case live in ``tests/test_video_io_alpha.py``;
here one band per case is enough to show the thresholds reach ffmpeg.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from services import alpha_output_manager as manager_module
from services import alpha_outputs_record as rec
from services import video_io
from test_generate_alpha_api import (  # noqa: F401  (alpha_client is a fixture)
    FPS,
    FRAMES,
    _body,
    _occupy,
    alpha_client,
    has_ffmpeg,
)

W, H = 320, 256
BANDS = (250, 128, 100, 0)

pytestmark = has_ffmpeg


def _ffmpeg() -> str:
    return shutil.which("ffmpeg")


def _write_source_with_audio(path) -> None:
    subprocess.run(
        [
            _ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate={FPS}:duration={FRAMES / FPS + 1.0:.3f}",
            "-f", "lavfi", "-i", f"sine=f=440:duration={FRAMES / FPS + 1.0:.3f}",
            "-frames:v", str(FRAMES),
            "-c:v", "ffv1", "-pix_fmt", "yuv444p",
            "-c:a", "pcm_s16le",
            str(path),
        ],
        check=True, capture_output=True,
    )


def _write_band_matte(path) -> None:
    band = W // 4
    boxes = ",".join(
        f"drawbox=x={i * band}:y=0:w={band}:h={H}:color=0x{v:02X}{v:02X}{v:02X}:t=fill"
        for i, v in enumerate(BANDS) if v
    )
    subprocess.run(
        [
            _ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=black:size={W}x{H}:rate={FPS}:duration=1,{boxes},format=gray",
            "-frames:v", str(FRAMES),
            "-c:v", "ffv1", "-pix_fmt", "gray",
            str(path),
        ],
        check=True, capture_output=True,
    )


def _alpha_job(client) -> tuple[str, object]:
    """A completed Alpha Gen job with the band matte. Returns (job_id, job_dir)."""
    src = client.tmp_path / "src_audio.mkv"
    if not src.exists():
        _write_source_with_audio(src)
    with src.open("rb") as fh:
        r = client.post("/api/v1/upload/video", files={"file": (src.name, fh, "video/x-matroska")})
    assert r.status_code == 200, r.text
    r = client.post("/api/v1/generate/alpha", json=_body(r.json()["video_id"]))
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    job_dir = client.app_context.config.output_dir / job_id
    assert video_io.has_audio_stream(job_dir / "_alpha_window.mp4")
    _write_band_matte(job_dir / "matte.mkv")
    return job_id, job_dir


def _streams(path) -> list[dict]:
    proc = subprocess.run(
        [
            shutil.which("ffprobe"), "-v", "error",
            "-show_entries", "stream=codec_type,codec_name,profile,pix_fmt,width,height,r_frame_rate",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return json.loads(proc.stdout)["streams"]


def _band_alpha(mov) -> list[int]:
    """The alpha at the centre of each band, first frame."""
    proc = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", str(mov), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
        check=True, capture_output=True,
    )
    frame = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(H, W, 4)
    band = W // 4
    return [int(frame[H // 2, i * band + band // 2, 3]) for i in range(4)]


def _post_rgba(client, job_id, body=None):
    return client.post(f"/api/v1/jobs/{job_id}/alpha/rgba", json=body if body is not None else {})


def _post_chroma(client, job_id, color, **thresholds):
    return client.post(f"/api/v1/jobs/{job_id}/alpha/chroma", json={"color": color, **thresholds})


def _job_alpha(client, job_id):
    return client.get(f"/api/v1/jobs/{job_id}").json()["alpha"]


DEFAULT = {"transparent_threshold": 7, "opaque_threshold": 248}


# ──────────────────────────────── rgba ────────────────────────────────


def test_rgba_makes_the_mov_and_the_preview(alpha_client):
    job_id, job_dir = _alpha_job(alpha_client)
    assert _job_alpha(alpha_client, job_id) == {"rgba": None, "chroma": {}}

    r = _post_rgba(alpha_client, job_id)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "job_id": job_id, "kind": "rgba", "color": None, **DEFAULT,
        "path": f"outputs/{job_id}/rgba.mov", "reused": False,
    }

    mov = job_dir / "rgba.mov"
    preview = job_dir / "alpha_preview.mp4"
    video, audio = _streams(mov)
    assert (video["codec_name"], video["profile"]) == ("prores", "4444")
    assert video["pix_fmt"].startswith("yuva")  # ffprobe says yuva444p12le
    assert (video["width"], video["height"]) == (W, H)
    assert video["r_frame_rate"] == f"{FPS}/1"
    assert audio["codec_name"] == "pcm_s16le"
    assert video_io.frame_count(mov) == FRAMES

    pv, pa = _streams(preview)
    assert (pv["codec_name"], pv["pix_fmt"]) == ("h264", "yuv420p")
    assert (pv["width"], pv["height"]) == (640, 512)
    assert pv["r_frame_rate"] == f"{FPS}/1"
    assert pa["codec_name"] == "aac"
    assert video_io.frame_count(preview) == FRAMES

    # Default levels 7 / 248: 250 -> 255, 128 -> 128 (linear), 0 -> 0.
    alpha = _band_alpha(mov)
    assert alpha[0] == 255 and abs(alpha[1] - 128) <= 1 and alpha[3] == 0, alpha

    assert _job_alpha(alpha_client, job_id) == {"rgba": DEFAULT, "chroma": {}}
    assert not list(job_dir.glob("*.tmp*"))


def test_rgba_without_a_body_uses_the_defaults(alpha_client):
    job_id, _job_dir = _alpha_job(alpha_client)
    r = alpha_client.post(f"/api/v1/jobs/{job_id}/alpha/rgba")
    assert r.status_code == 200, r.text
    assert r.json()["transparent_threshold"] == 7
    assert r.json()["opaque_threshold"] == 248


def test_rgba_same_thresholds_is_reused_and_others_remake(alpha_client):
    job_id, job_dir = _alpha_job(alpha_client)
    mov = job_dir / "rgba.mov"
    assert _post_rgba(alpha_client, job_id).json()["reused"] is False
    first = mov.stat().st_mtime_ns

    again = _post_rgba(alpha_client, job_id, DEFAULT)
    assert again.status_code == 200, again.text
    assert again.json()["reused"] is True
    assert mov.stat().st_mtime_ns == first

    # Equal thresholds: black and white (the 128 band becomes opaque).
    binary = {"transparent_threshold": 128, "opaque_threshold": 128}
    r = _post_rgba(alpha_client, job_id, binary)
    assert r.json()["reused"] is False
    assert _band_alpha(mov) == [255, 255, 0, 0]
    assert _job_alpha(alpha_client, job_id)["rgba"] == binary

    # 0 / 255: the matte unchanged.
    r = _post_rgba(alpha_client, job_id, {"transparent_threshold": 0, "opaque_threshold": 255})
    assert r.json()["reused"] is False
    assert _band_alpha(mov) == [250, 128, 100, 0]


def test_rgba_record_without_the_file_is_remade(alpha_client):
    job_id, job_dir = _alpha_job(alpha_client)
    _post_rgba(alpha_client, job_id)
    (job_dir / "alpha_preview.mp4").unlink()
    # The record alone does not count.
    assert _job_alpha(alpha_client, job_id)["rgba"] is None

    r = _post_rgba(alpha_client, job_id)
    assert r.json()["reused"] is False
    assert (job_dir / "alpha_preview.mp4").exists()
    assert _job_alpha(alpha_client, job_id)["rgba"] == DEFAULT


# ─────────────────────────────── chroma ───────────────────────────────


def test_chroma_one_file_per_colour(alpha_client):
    job_id, job_dir = _alpha_job(alpha_client)

    r = _post_chroma(alpha_client, job_id, "green")
    assert r.status_code == 200, r.text
    assert r.json() == {
        "job_id": job_id, "kind": "chroma", "color": "green", **DEFAULT,
        "path": f"outputs/{job_id}/chroma_green.mp4", "reused": False,
    }
    binary = {"transparent_threshold": 128, "opaque_threshold": 128}
    r = _post_chroma(alpha_client, job_id, "blue", **binary)
    assert r.status_code == 200, r.text

    green = job_dir / "chroma_green.mp4"
    blue = job_dir / "chroma_blue.mp4"
    video, audio = _streams(green)
    assert (video["codec_name"], video["pix_fmt"]) == ("h264", "yuv420p")
    assert (video["width"], video["height"]) == (W, H)
    assert video["r_frame_rate"] == f"{FPS}/1"
    assert audio["codec_name"] == "aac"
    assert video_io.frame_count(green) == FRAMES
    assert blue.exists()

    assert _job_alpha(alpha_client, job_id) == {
        "rgba": None, "chroma": {"green": DEFAULT, "blue": binary},
    }

    # The recipe plus what the file is, in the comment tag.
    tag = json.loads(video_io.probe_comment_tag(blue))
    assert tag["alpha_output"] == {"kind": "chroma", "color": "blue", **binary}
    assert tag["alpha_gen"]["output_fps"] == f"{FPS}/1"


def test_chroma_reuse_and_remake_per_colour(alpha_client):
    job_id, job_dir = _alpha_job(alpha_client)
    green = job_dir / "chroma_green.mp4"
    _post_chroma(alpha_client, job_id, "green")
    _post_chroma(alpha_client, job_id, "blue")
    green_first = green.stat().st_mtime_ns
    blue_first = (job_dir / "chroma_blue.mp4").stat().st_mtime_ns

    r = _post_chroma(alpha_client, job_id, "green", **DEFAULT)
    assert r.json()["reused"] is True
    assert green.stat().st_mtime_ns == green_first

    # Same colour, other thresholds: remade; the other colour is untouched.
    r = _post_chroma(alpha_client, job_id, "green", transparent_threshold=0, opaque_threshold=255)
    assert r.json()["reused"] is False
    assert (job_dir / "chroma_blue.mp4").stat().st_mtime_ns == blue_first
    assert _job_alpha(alpha_client, job_id)["chroma"] == {
        "green": {"transparent_threshold": 0, "opaque_threshold": 255},
        "blue": DEFAULT,
    }

    # A record without its file is remade.
    green.unlink()
    assert "green" not in _job_alpha(alpha_client, job_id)["chroma"]
    r = _post_chroma(alpha_client, job_id, "green", transparent_threshold=0, opaque_threshold=255)
    assert r.json()["reused"] is False
    assert green.exists()


# ──────────────────────────────── GET ─────────────────────────────────


def test_get_routes_serve_the_files(alpha_client):
    job_id, _job_dir = _alpha_job(alpha_client)
    for path in ("alpha/rgba", "alpha/preview", "alpha/chroma/green"):
        r = alpha_client.get(f"/api/v1/jobs/{job_id}/{path}")
        assert r.status_code == 404, (path, r.text)
        assert r.json()["error"]["code"] == "ALPHA_OUTPUT_NOT_READY"

    _post_rgba(alpha_client, job_id)
    _post_chroma(alpha_client, job_id, "green")

    r = alpha_client.get(f"/api/v1/jobs/{job_id}/alpha/rgba")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/quicktime"
    assert f"{job_id}_rgba.mov" in r.headers["content-disposition"]

    r = alpha_client.get(f"/api/v1/jobs/{job_id}/alpha/preview")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert f"{job_id}_alpha_preview.mp4" in r.headers["content-disposition"]

    r = alpha_client.get(f"/api/v1/jobs/{job_id}/alpha/chroma/green")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert f"{job_id}_chroma_green.mp4" in r.headers["content-disposition"]

    # Another colour that was not made yet.
    r = alpha_client.get(f"/api/v1/jobs/{job_id}/alpha/chroma/blue")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "ALPHA_OUTPUT_NOT_READY"


def test_get_unknown_job_or_colour(alpha_client):
    r = alpha_client.get("/api/v1/jobs/no-such-job/alpha/rgba")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "JOB_NOT_FOUND"
    r = alpha_client.get("/api/v1/jobs/no-such-job/alpha/chroma/purple")
    assert r.status_code == 422


# ─────────────────────────────── refusals ─────────────────────────────


@pytest.mark.parametrize(
    "body",
    [
        {"transparent_threshold": 200, "opaque_threshold": 100},
        {"transparent_threshold": -1},
        {"opaque_threshold": 256},
    ],
)
def test_bad_thresholds_are_422(alpha_client, body):
    r = _post_rgba(alpha_client, "any-job", body)
    assert r.status_code == 422, r.text
    r = _post_chroma(alpha_client, "any-job", "green", **body)
    assert r.status_code == 422, r.text


def test_chroma_colour_is_required_and_checked(alpha_client):
    r = alpha_client.post("/api/v1/jobs/any-job/alpha/chroma", json={})
    assert r.status_code == 422
    r = _post_chroma(alpha_client, "any-job", "purple")
    assert r.status_code == 422


def test_unknown_job_is_404(alpha_client):
    r = _post_rgba(alpha_client, "no-such-job")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "JOB_NOT_FOUND"


def test_unfinished_job_is_409(alpha_client):
    job = _occupy(alpha_client)
    r = _post_rgba(alpha_client, job.job_id)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "VIDEO_NOT_READY"


def test_ordinary_job_is_422_and_has_no_alpha(client):
    r = client.post(
        "/api/v1/generate",
        json={"prompt": "a red ball", "width": 320, "height": 256, "num_frames": FRAMES},
    )
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed"
    assert job["alpha"] is None

    r = _post_rgba(client, job_id)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "JOB_NOT_ALPHA"
    r = _post_chroma(client, job_id, "green")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "JOB_NOT_ALPHA"


def test_job_without_alpha_gen_block_is_422(alpha_client):
    job_id, job_dir = _alpha_job(alpha_client)
    meta_path = job_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["alpha_gen"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    r = _post_rgba(alpha_client, job_id)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "JOB_NOT_ALPHA"


def test_missing_matte_is_404(alpha_client):
    job_id, job_dir = _alpha_job(alpha_client)
    (job_dir / "matte.mkv").unlink()
    r = _post_rgba(alpha_client, job_id)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "JOB_NOT_FOUND"
    # No matte -> not reported as an Alpha Gen job any more.
    assert _job_alpha(alpha_client, job_id) is None


def test_ffmpeg_failure_is_503_and_keeps_nothing(alpha_client, monkeypatch):
    job_id, job_dir = _alpha_job(alpha_client)
    _post_rgba(alpha_client, job_id)

    def fail(*_a, **_k):
        raise video_io.FFmpegError("boom")

    monkeypatch.setattr(video_io, "compose_alpha_rgba", fail)
    monkeypatch.setattr(video_io, "compose_alpha_chroma", fail)
    r = _post_rgba(alpha_client, job_id, {"transparent_threshold": 0, "opaque_threshold": 255})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "ALPHA_OUTPUT_FAILED"
    assert "boom" in r.json()["error"]["detail"]
    # The old record entry was dropped before the attempt: nothing promises
    # thresholds the file does not have.
    assert _job_alpha(alpha_client, job_id)["rgba"] is None

    r = _post_chroma(alpha_client, job_id, "green")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "ALPHA_OUTPUT_FAILED"
    assert not list(job_dir.glob("*.tmp*"))


def test_replace_failure_is_503(alpha_client, monkeypatch):
    """A ``<video>`` still reading the old file can make ``os.replace`` fail
    on Windows; that is a 503, not a 500."""
    job_id, job_dir = _alpha_job(alpha_client)
    real_replace = manager_module.os.replace

    def replace(src, dst):
        if str(dst).endswith(("rgba.mov", "chroma_green.mp4")):
            raise PermissionError("in use")
        return real_replace(src, dst)

    monkeypatch.setattr(manager_module.os, "replace", replace)
    r = _post_rgba(alpha_client, job_id)
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "ALPHA_OUTPUT_FAILED"
    r = _post_chroma(alpha_client, job_id, "green")
    assert r.status_code == 503
    assert _job_alpha(alpha_client, job_id) == {"rgba": None, "chroma": {}}
    assert not list(job_dir.glob("*.tmp*"))


def test_delete_job_removes_the_outputs(alpha_client):
    job_id, job_dir = _alpha_job(alpha_client)
    _post_rgba(alpha_client, job_id)
    _post_chroma(alpha_client, job_id, "green")
    assert (job_dir / rec.RECORD_FILENAME).exists()
    r = alpha_client.delete(f"/api/v1/jobs/{job_id}")
    assert r.status_code == 200
    assert not job_dir.exists()
