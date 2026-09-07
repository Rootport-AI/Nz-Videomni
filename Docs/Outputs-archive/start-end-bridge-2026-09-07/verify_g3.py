"""Verify one G3 bridge run against the acceptance conditions."""
import dataclasses, hashlib, json, pathlib, shutil, subprocess, sys

sys.path.insert(0, r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni")
from chain_math import compute_chain_layout

B = pathlib.Path(r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\start-end-bridge-2026-09-07")
FFPROBE = r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\tools\ffmpeg\bin\ffprobe.exe"
tag = sys.argv[1]                                    # ltx23 | ltx25

res = json.loads((B / f"result_g3_{tag}.json").read_text(encoding="utf-8"))
mp4 = pathlib.Path(res["video_path"]["path"])
dest = B / "g3" / tag
dest.mkdir(parents=True, exist_ok=True)
shutil.copy2(mp4.parent / "metadata.json", dest / "metadata.json")

h = hashlib.md5()
with open(mp4, "rb") as fh:
    for c in iter(lambda: fh.read(1 << 20), b""):
        h.update(c)
md5 = h.hexdigest()
(dest / "md5.txt").write_text(f"{md5}  {mp4.name}\n{mp4}\n", encoding="utf-8")

m = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
es = m.get("end_source") or {}
frz = es.get("stage1_freezes") or []

lay = dataclasses.asdict(compute_chain_layout([121, 121, 121], 24.0, 3,
                                              source_context_px=73, end_context_px=8))

# ffprobe: video stream only
out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                      "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
                      "-of", "default=nokey=0:noprint_wrappers=1", str(mp4)],
                     capture_output=True, text=True)
probe = dict(l.split("=", 1) for l in out.stdout.strip().splitlines())
out_a = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=codec_type,nb_frames",
                        "-of", "default=nokey=0:noprint_wrappers=1", str(mp4)],
                       capture_output=True, text=True)
has_audio = "codec_type=audio" in out_a.stdout

nb = int(probe.get("nb_frames", -1))
seg2 = frz[-1] if frz else {}
ka_last = (es.get("ka_list") or lay["ka_list"])[-1]
n_end_a_frozen = es.get("n_end_a_frozen")

checks = [
    ("mode == 'bridge'", es.get("mode"), "bridge", es.get("mode") == "bridge"),
    ("generation_order == [0,1,2]", es.get("generation_order"), [0, 1, 2],
     es.get("generation_order") == [0, 1, 2]),
    ("stage1_order == [0,1,2]", es.get("stage1_order"), [0, 1, 2],
     es.get("stage1_order") == [0, 1, 2]),
    ("seg2.fkv == 3", seg2.get("fkv"), 3, seg2.get("fkv") == 3),
    ("seg2.ftv == n_end_v", seg2.get("ftv"), es.get("n_end_v"),
     seg2.get("ftv") == es.get("n_end_v")),
    ("seg2.fta == n_end_a_frozen", seg2.get("fta"), n_end_a_frozen,
     seg2.get("fta") == n_end_a_frozen),
    ("n_end_a_frozen == n_end_a", n_end_a_frozen, es.get("n_end_a"),
     n_end_a_frozen == es.get("n_end_a")),
    ("seg2.fka == ka_list[-1]", seg2.get("fka"), ka_last, seg2.get("fka") == ka_last),
    ("seg0.ftv == 0", (frz[0] if frz else {}).get("ftv"), 0,
     (frz[0] if frz else {}).get("ftv") == 0),
    ("seg1.fkv == 3", (frz[1] if len(frz) > 1 else {}).get("fkv"), 3,
     (frz[1] if len(frz) > 1 else {}).get("fkv") == 3),
    ("seg1.ftv == 0", (frz[1] if len(frz) > 1 else {}).get("ftv"), 0,
     (frz[1] if len(frz) > 1 else {}).get("ftv") == 0),
    ("freeze_proof.pass == true", (es.get("freeze_proof") or {}).get("pass"), True,
     (es.get("freeze_proof") or {}).get("pass") is True),
    ("nb_frames == 256", nb, 256, nb == 256),
    ("width x height == 1280x768", f"{probe.get('width')}x{probe.get('height')}",
     "1280x768", f"{probe.get('width')}x{probe.get('height')}" == "1280x768"),
    ("r_frame_rate == 24/1", probe.get("r_frame_rate"), "24/1",
     probe.get("r_frame_rate") == "24/1"),
]

report = {
    "tag": tag,
    "job_id": res.get("job_id"),
    "elapsed_sec": res.get("elapsed_sec"),
    "vram_max_mib": res.get("vram_max_mib"),
    "mp4": str(mp4), "md5": md5, "size_bytes": mp4.stat().st_size,
    "has_audio_stream": has_audio,
    "ffprobe": probe,
    "end_source_observed": {
        "mode": es.get("mode"), "generation_order": es.get("generation_order"),
        "stage1_order": es.get("stage1_order"), "n_end_v": es.get("n_end_v"),
        "n_end_a": es.get("n_end_a"), "n_end_a_frozen": n_end_a_frozen,
        "ka_list_in_metadata": es.get("ka_list"),
        "stage1_freezes": frz,
        "freeze_proof": es.get("freeze_proof"),
    },
    "chain_math_expected": {k: lay[k] for k in
                            ("ka_list", "n_end_v", "n_end_a", "trim_px", "total_px",
                             "new_frames_px", "f_total", "end_source_mode",
                             "seg_generation_order")},
    "checks": [{"name": n, "observed": o, "expected": e, "pass": bool(p)}
               for n, o, e, p in checks],
}
report["all_pass"] = all(c["pass"] for c in report["checks"])
(dest / "verify.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"=== {tag}  job={report['job_id']}  md5={md5} ===")
print(f"elapsed={report['elapsed_sec']}s  vram_max={report['vram_max_mib']}MiB  audio={has_audio}")
for c in report["checks"]:
    print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}: observed={c['observed']} expected={c['expected']}")
print("ALL PASS:", report["all_pass"])
