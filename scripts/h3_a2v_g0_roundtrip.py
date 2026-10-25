r"""Gate G0 of the H3 A2V (台帳 §1-103, plan §4): audio encoder -> existing decoder round trip on the CPU.

Reads the owner's wav with ``engine_h3.source_audio.load_source_waveform``
(PyAV, 32 kHz stereo float32), encodes it with ``H3AudioEncoder`` (fp32, CPU),
decodes the latents with the existing ``H3AudioVAE`` (fp32, CPU) and compares
the result with the resampled input:

1. the encoder read: ``LoadReport.loaded == 134``, the skipped set is
   logs_proj 2 + the decoder + the 2 statistic tensors, nothing left on meta;
2. the RMS ratio in dB;
3. the waveform correlation r, maximised over a lag search of +-800 samples,
   and the STFT magnitude correlation as the fallback metric.

Pass: RMS within +-1.5 dB AND (r >= 0.8 OR STFT-magnitude corr >= 0.95).
Writes the reconstruction, the 32 kHz reference and ``g0.json`` to
``outputs/h3_a2v_g0/``. No GPU is touched.

    .venv-engine-h3\Scripts\python.exe scripts\h3_a2v_g0_roundtrip.py [--wav PATH]
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from engine_h3 import audio_vae as A  # noqa: E402
from engine_h3 import constants as C  # noqa: E402
from engine_h3.sft_load import meta_tensor_names  # noqa: E402
from engine_h3.source_audio import load_source_waveform  # noqa: E402

DEFAULT_WAV = ROOT / "uploads" / "h3-a2v-development" / "実験の成功、楽しみにしていますね.wav"
AUDIO_VAE = ROOT / "models" / "MiniMaxH3" / "VAE" / "minimax_h3_audio_vae_fp32.safetensors"
OUT_DIR = ROOT / "outputs" / "h3_a2v_g0"
MAX_LAG = 800
RMS_TOLERANCE_DB = 1.5
MIN_WAVE_CORR = 0.8
MIN_STFT_CORR = 0.95


def _write_wav(path: Path, data: torch.Tensor, rate: int) -> None:
    pcm = (data.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).t().contiguous().numpy()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[0])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b)) / den if den > 0 else 0.0


def best_lag_correlation(ref: np.ndarray, rec: np.ndarray, max_lag: int) -> tuple[float, int]:
    """Max Pearson r over lags in [-max_lag, max_lag]; lag > 0 means ``rec`` is late by that many samples."""
    best_r, best_lag = -2.0, 0
    n = min(len(ref), len(rec))
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = ref[:n - lag], rec[lag:n]
        else:
            a, b = ref[-lag:n], rec[:n + lag]
        r = _pearson(a, b)
        if r > best_r:
            best_r, best_lag = r, lag
    return best_r, best_lag


def stft_magnitude_correlation(ref: np.ndarray, rec: np.ndarray) -> float:
    window = torch.hann_window(1024)
    mags = [torch.stft(torch.from_numpy(x), n_fft=1024, hop_length=256, window=window,
                       return_complex=True).abs().flatten().numpy() for x in (ref, rec)]
    return _pearson(mags[0].astype(np.float64), mags[1].astype(np.float64))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wav", default=str(DEFAULT_WAV))
    ap.add_argument("--vae", default=str(AUDIO_VAE))
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Whole file: keep_samples 1 (no length requirement), no read cap.
    ref, info = load_source_waveform(args.wav, keep_samples=1, read_samples=10**9)
    n = ref.shape[1]
    print(f"input: {args.wav}\n  {info}  -> {tuple(ref.shape)} ({n / C.AUDIO_SAMPLE_RATE:.3f} s)")

    t0 = time.perf_counter()
    enc = A.H3AudioEncoder.from_file(args.vae, device="cpu")
    report = enc.load_report
    still_meta = meta_tensor_names(enc)
    skipped = set(report.skipped)
    skipped_parts = {
        "logs_proj": sorted(k for k in skipped if A.file_part(k) == A.PART_UNUSED),
        "stats": sorted(k for k in skipped if A.file_part(k) == A.PART_STATS),
        "decoder": sum(A.file_part(k) == A.PART_DECODER for k in skipped),
        "other": sorted(k for k in skipped if A.file_part(k) not in (A.PART_UNUSED, A.PART_STATS, A.PART_DECODER)),
    }
    latents = enc.encode(ref)
    t_enc = time.perf_counter() - t0
    del enc
    gc.collect()
    print(f"encoder: loaded={report.loaded} skipped={len(skipped)} {skipped_parts}"
          f" meta_left={len(still_meta)}  latents={tuple(latents.shape)}  ({t_enc:.1f} s)")

    t0 = time.perf_counter()
    dec = A.H3AudioVAE.from_file(args.vae, device="cpu")
    rec = dec.decode(latents)[:, :n]
    t_dec = time.perf_counter() - t0
    del dec
    gc.collect()
    print(f"decoder: {tuple(rec.shape)}  ({t_dec:.1f} s)")

    ref_m = ref.mean(dim=0).double().numpy()
    rec_m = rec.mean(dim=0).double().numpy()
    rms_ref = float(np.sqrt(np.mean(ref_m ** 2)))
    rms_rec = float(np.sqrt(np.mean(rec_m ** 2)))
    rms_db = 20.0 * math.log10(rms_rec / rms_ref)
    r0 = _pearson(ref_m, rec_m)
    r_best, lag = best_lag_correlation(ref_m, rec_m, MAX_LAG)
    stft_corr = stft_magnitude_correlation(ref_m.astype(np.float32), rec_m.astype(np.float32))

    weights_ok = (report.loaded == A.ENCODER_TENSOR_COUNT and not still_meta
                  and len(skipped_parts["logs_proj"]) == 2 and len(skipped_parts["stats"]) == 2
                  and not skipped_parts["other"])
    metric_ok = abs(rms_db) <= RMS_TOLERANCE_DB and (r_best >= MIN_WAVE_CORR or stft_corr >= MIN_STFT_CORR)
    result = {
        "input": str(args.wav),
        "source": info,
        "samples_32k": n,
        "latents": list(latents.shape),
        "encoder_load": {
            "loaded": report.loaded,
            "skipped_total": len(skipped),
            "skipped_logs_proj": skipped_parts["logs_proj"],
            "skipped_stats": skipped_parts["stats"],
            "skipped_decoder": skipped_parts["decoder"],
            "skipped_other": skipped_parts["other"],
            "meta_left": still_meta,
        },
        "rms_ref": rms_ref,
        "rms_rec": rms_rec,
        "rms_ratio_db": rms_db,
        "wave_corr_lag0": r0,
        "wave_corr_best": r_best,
        "wave_corr_best_lag": lag,
        "stft_mag_corr": stft_corr,
        "channels_identical_in_rec": bool(torch.equal(rec[0], rec[1])),
        "seconds": {"encode_incl_load": round(t_enc, 1), "decode_incl_load": round(t_dec, 1)},
        "pass_weights": weights_ok,
        "pass_metrics": metric_ok,
        "pass": weights_ok and metric_ok,
    }
    _write_wav(OUT_DIR / "reconstructed_32k.wav", rec, C.AUDIO_SAMPLE_RATE)
    _write_wav(OUT_DIR / "reference_32k.wav", ref, C.AUDIO_SAMPLE_RATE)
    (OUT_DIR / "g0.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("G0:", "PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
