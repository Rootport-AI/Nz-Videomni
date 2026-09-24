"""The generation recipe inside the delivered mp4 (台帳 §3-164).

metadata.json's JSON text is written into the finished ``output.mp4`` as the
container ``comment`` tag (``services.video_io.embed_comment_tag``) and read back
with ffprobe (``probe_comment_tag``, ``POST /utils/mp4-info``). Real ffmpeg, mock
backend (no GPU). ``joined.mp4`` is covered in tests/test_join_api.py.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from services import video_io

GENERATE = {
    "prompt": "a lighthouse at night; waves = 3 # calm \\ still",
    "width": 384,
    "height": 256,
    "num_frames": 25,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "pipeline": "distilled",
}

CHAIN = {
    "prompt": "a serene mountain lake at dawn",
    "width": 384,
    "height": 256,
    "frame_rate": 24.0,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "pipeline": "distilled",
    "overlap_frames": 2,
    "overlap_strength": 0.5,
    "clips": [{"num_frames": 25}, {"num_frames": 25}],
}


def _make_mp4(path, n_frames=6, fps=24.0, size=(64, 48)):
    frames = [Image.new("RGB", size, (i * 20 % 256, 80, 140)) for i in range(n_frames)]
    return video_io.encode_frames_to_mp4(frames, path, frame_rate=fps)


def _run(client, route, payload):
    r = client.post(route, json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    job_dir = client.app_context.config.output_dir / job_id
    return job_dir / "output.mp4", job_dir / "metadata.json"


def _local(client, host="127.0.0.1"):
    """A client on the same app whose requests come from ``host``."""
    return TestClient(client.app, client=(host, 50000))


# ------------------------------------------------------------- (a) round trip


@pytest.mark.parametrize(
    "value, expected",
    [
        ("plain", "plain"),
        ("a;b#c=d\\e", "a;b#c=d\\e"),
        ("line1\r\nline2\rline3\nline4", "line1\nline2\nline3\nline4"),
        ("日本語のプロンプト、夜の灯台 🌊🎬", "日本語のプロンプト、夜の灯台 🌊🎬"),
        ('{\n  "prompt": "x; y = z # w \\\\ v"\n}', '{\n  "prompt": "x; y = z # w \\\\ v"\n}'),
    ],
)
def test_embed_round_trips_special_characters(tmp_path, value, expected):
    mp4 = _make_mp4(tmp_path / "a.mp4")
    video_io.embed_comment_tag(mp4, value)
    assert video_io.probe_comment_tag(mp4) == expected
    # Stream copy: the video is untouched, and no temporary file is left.
    assert video_io.frame_count(mp4) == 6
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.mp4"]


def test_embed_round_trips_a_large_value(tmp_path):
    mp4 = _make_mp4(tmp_path / "a.mp4")
    value = json.dumps({"prompt": "語" * 20000, "k": list(range(2000))}, ensure_ascii=False, indent=2)
    video_io.embed_comment_tag(mp4, value)
    assert video_io.probe_comment_tag(mp4) == value


def test_probe_comment_tag_is_none_without_a_tag(tmp_path):
    assert video_io.probe_comment_tag(_make_mp4(tmp_path / "a.mp4")) is None


def test_probe_comment_tag_raises_on_unreadable_file(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(video_io.FFmpegError):
        video_io.probe_comment_tag(bad)


def test_embed_failure_keeps_the_original_file(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(video_io.FFmpegError):
        video_io.embed_comment_tag(bad, "x")
    assert bad.read_bytes() == b"not a video"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bad.mp4"]


def test_save_metadata_writes_recipe_text_in_text_mode(tmp_path):
    metadata = {"prompt": "夜\n灯台", "n": 1, "nested": {"a": [1, 2]}}
    path = video_io.save_metadata(tmp_path / "metadata.json", metadata)
    expected = video_io.recipe_text(metadata).replace("\n", os.linesep).encode("utf-8")
    assert path.read_bytes() == expected


# ------------------------------------------------------ (f) external tags


def test_embed_replaces_existing_global_tags(tmp_path):
    src = _make_mp4(tmp_path / "src.mp4")
    tagged = tmp_path / "tagged.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"), "-y", "-v", "error", "-i", str(src),
            "-c", "copy", "-metadata", "comment=someone else's", "-metadata", "title=external",
            str(tagged),
        ],
        check=True,
    )
    assert video_io.probe_comment_tag(tagged) == "someone else's"

    video_io.embed_comment_tag(tagged, "ours")

    assert video_io.probe_comment_tag(tagged) == "ours"
    probe = subprocess.run(
        [shutil.which("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", str(tagged)],
        capture_output=True, text=True, check=True,
    )
    tags = {k.lower(): v for k, v in json.loads(probe.stdout)["format"].get("tags", {}).items()}
    assert "title" not in tags


# --------------------------------------------------- (b)(c)(d) generation path


def test_generate_embeds_the_metadata_json_text(client):
    output, metadata_path = _run(client, "/api/v1/generate", GENERATE)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["request"]["embed_mp4_metadata"] is True
    assert video_io.probe_comment_tag(output) == video_io.recipe_text(metadata)
    assert sorted(p.name for p in output.parent.glob("*.tmp*")) == []


def test_generate_opted_out_writes_no_tag(client):
    output, metadata_path = _run(client, "/api/v1/generate", {**GENERATE, "embed_mp4_metadata": False})
    assert video_io.probe_comment_tag(output) is None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["request"]["embed_mp4_metadata"] is False


def test_chain_embeds_the_metadata_json_text(client):
    output, metadata_path = _run(client, "/api/v1/generate/chain", CHAIN)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["request"]["embed_mp4_metadata"] is True
    assert video_io.probe_comment_tag(output) == video_io.recipe_text(metadata)


def test_chain_opted_out_writes_no_tag(client):
    output, _ = _run(client, "/api/v1/generate/chain", {**CHAIN, "embed_mp4_metadata": False})
    assert video_io.probe_comment_tag(output) is None


def test_embed_failure_does_not_fail_the_job(client, monkeypatch, caplog):
    def boom(path, comment):
        raise video_io.FFmpegError("simulated")

    monkeypatch.setattr(video_io, "embed_comment_tag", boom)
    output, metadata_path = _run(client, "/api/v1/generate", GENERATE)
    assert output.exists() and metadata_path.exists()
    assert video_io.probe_comment_tag(output) is None
    assert any("simulated" in rec.getMessage() for rec in caplog.records)


# ------------------------------------------------- (e) POST /utils/mp4-info


def test_mp4_info_returns_the_comment(client):
    output, metadata_path = _run(client, "/api/v1/generate", GENERATE)
    r = _local(client).post("/api/v1/utils/mp4-info", json={"path": str(output)})
    assert r.status_code == 200, r.text
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert r.json() == {"comment": video_io.recipe_text(metadata)}


def test_mp4_info_null_without_a_tag(client, tmp_path):
    mp4 = _make_mp4(tmp_path / "plain.mp4")
    r = _local(client, "::1").post("/api/v1/utils/mp4-info", json={"path": str(mp4)})
    assert r.status_code == 200, r.text
    assert r.json() == {"comment": None}


@pytest.mark.parametrize("name", ["missing.mp4", "a_directory"])
def test_mp4_info_404_when_not_a_file(client, tmp_path, name):
    (tmp_path / "a_directory").mkdir()
    r = _local(client).post("/api/v1/utils/mp4-info", json={"path": str(tmp_path / name)})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "MEDIA_NOT_FOUND"


def test_mp4_info_422_when_unreadable(client, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    r = _local(client).post("/api/v1/utils/mp4-info", json={"path": str(bad)})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "MEDIA_UNREADABLE"


@pytest.mark.parametrize(
    "path",
    [
        "\\\\server\\share\\a.mp4",
        "//server/share/a.mp4",
        "/\\server\\share\\a.mp4",
        "\\\\?\\UNC\\server\\share\\a.mp4",
        "//?/UNC/server/share/a.mp4",
    ],
)
def test_mp4_info_422_for_unc_paths_before_touching_the_filesystem(client, monkeypatch, path):
    from pathlib import Path

    def no_fs(self):
        raise AssertionError("a UNC path must be refused before any filesystem access")

    monkeypatch.setattr(Path, "is_file", no_fs)
    r = _local(client).post("/api/v1/utils/mp4-info", json={"path": path})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "MEDIA_UNREADABLE"


def test_mp4_info_403_for_a_non_loopback_client(client, tmp_path):
    mp4 = _make_mp4(tmp_path / "plain.mp4")
    r = _local(client, "10.0.0.5").post("/api/v1/utils/mp4-info", json={"path": str(mp4)})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "LOCAL_ONLY"
