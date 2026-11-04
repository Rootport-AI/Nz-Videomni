# -*- coding: utf-8 -*-
"""【当てはめ】reserve_gap.py の形（割り当ての当てはめ直し＋φ·Nv·U ≦ cap 13,999MiB）で、参照なし・K=1 の境界を F（連続値）と S で解く。
  python -I -X utf8 analysis/boundary_estimate.py > analysis/boundary_estimate_output.txt"""
MiB = 1048576; U = 10752; CAP = 13999
def rows(W, H, F, K=1, text0=30):
    gh, gw = H // 32, W // 32; lat = (F - 5) / 17 * 5 + 2
    video = gh * gw * lat; cond = K * gh * gw; audio = 2 * round(F / 24 * 40)
    text = text0 + K * (gh * gw + 8)
    return video, cond, audio, text, video + cond + audio + text
def reserved(W, H, F, phi):
    v, c, a, t, S = rows(W, H, F)
    return 2252.1 + (51102 * S + 10287 * t) / MiB + phi * v * U / MiB, S
for W, H in ((1920, 1088), (2560, 1472)):
    for phi in (0.383, 0.454):
        lo, hi = 5.0, 600.0
        for _ in range(80):
            m = (lo + hi) / 2
            if reserved(W, H, m, phi)[0] <= CAP: lo = m
            else: hi = m
        print("%d×%d φ=%.3f: 境界 F≈%.1f フレーム・S≈%.0f 行" % (W, H, phi, lo, reserved(W, H, lo, phi)[1]))
