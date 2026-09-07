"""Exhaustive-ish sweep: can any REACHABLE bridge request trip an assert?

An assert is a 500; a ValueError is a 422. Only the second is acceptable for
anything a request can express. Also records the minimum audio slack, the number
the docstring/comment claim (>= 2 latents).
"""
import itertools, sys
sys.path.insert(0, "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni")
import chain_math as cm

ALL_FPS = (23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0)
WINDOWS = ("standard", "high_resolution")
CLIP_SETS = (
    [49, 49], [49, 169], [169, 169], [257, 257], [169, 177, 185],
    [161, 161, 169], [481, 481], [257] * 4, [49] * 8, [481] * 4,
    [25, 25, 169], [105, 113], [73, 89, 97], [481, 49],
)
# v2v_context_frames is bounded [25, 145] and must be 8n+1.
CTX = [25, 33, 73, 105, 137, 145]
END = [8, 16, 24, 40, 72, 104, 136]

asserts, values, ok = [], [], 0
min_slack = None
min_ctx_slack = None
for clips, fps, kv, window, ctx, end in itertools.product(
    CLIP_SETS, ALL_FPS, (1, 2, 3, 5), WINDOWS, CTX, END
):
    if any(kv >= cm.v_latent_frames(c) for c in clips):
        continue
    v_tile, v_adv = cm.resolve_stage2_window(window)
    try:
        L = cm.compute_chain_layout(
            clips, fps, kv=kv, v_tile=v_tile, v_adv=v_adv,
            source_context_px=ctx, end_context_px=end,
        )
    except AssertionError as e:
        asserts.append((clips, fps, kv, window, ctx, end, str(e)[:200]))
        continue
    except ValueError as e:
        values.append(str(e)[:60])
        continue
    assert L.end_source_mode == "bridge", L.end_source_mode
    ok += 1
    ka = L.ka_list[-1] if L.ka_list else 0
    slack = L.seg_audio[-1] - (L.n_end_a + ka)
    if min_slack is None or slack < min_slack:
        min_slack = slack
        min_slack_at = (clips, fps, kv, window, ctx, end)
    cs = L.a_total - (L.n_ctx_a + L.n_end_a)
    if min_ctx_slack is None or cs < min_ctx_slack:
        min_ctx_slack = cs
        min_ctx_at = (clips, fps, kv, window, ctx, end)

print("accepted        :", ok)
print("rejected (422)  :", len(values))
print("ASSERTS (500!)  :", len(asserts))
for a in asserts[:5]:
    print("   ", a)
print("min audio slack (n_end_a + ka_list[-1] vs seg_audio[-1]):", min_slack, min_slack_at)
print("min ctx slack   (n_ctx_a + n_end_a vs a_total)          :", min_ctx_slack, min_ctx_at)
from collections import Counter
print("rejection kinds :")
for m, n in Counter(values).most_common():
    print(f"   {n:6d}  {m}")
