# -*- coding: utf-8 -*-
"""Per-kind weight scan (supervisor's calculation, copied from the session scratchpad 2026-11-03 and
wrapped into functions so make_results_ref.py can import it; the arithmetic is unchanged).

cost = video rows (h3.axes.vc; cond is 0 for every point) + w x (reference rows [+ text rows the
references add]), text rows = sequence.text (OOM points: h3.axes.text) - 30 (the prompt).
Scan w = 0.00 .. 3.00 in steps of 0.01 and report the w range where max(cost of completed) <
min(cost of OOM).  Baselines (b*) and warm* are left out.  The reference-free points (r*) are
added to every kind.  This is a POST-HOC fit (RESULTS.md §4.1), not a measurement.

  set PYTHONUTF8=1 & ..\\..\\.venv\\Scripts\\python.exe kind_weights.py
"""
import glob
import json
import os

base = os.path.dirname(os.path.abspath(__file__))


def load_points():
    pts = {}
    for p in glob.glob(os.path.join(base, 'runs', '*', 'run.json')):
        d = json.load(open(p, encoding='utf-8')); lab = os.path.basename(os.path.dirname(p))
        if lab.startswith('warm') or lab.startswith('b'):
            continue
        v = d['verdict_v3dprime']; ax = d['h3']['axes']; pr = d['predict']
        seq = (d['h3'].get('sequence') or {})

        def g(k, alt):
            x = seq.get(k) if isinstance(seq, dict) else None
            return x if x is not None else (alt if alt is not None else 0)
        text = g('text', ax.get('text'))
        refv = g('ref_video', pr.get('ref_video'))
        refa = g('ref_audio', pr.get('ref_audio'))
        kind = 'none'
        if lab.startswith('i'):
            kind = 'picture'
        elif lab.startswith('v') or lab.startswith('x'):
            kind = 'video'
        elif lab.startswith('a'):
            kind = 'audio'
        pts[lab] = dict(kind=kind, ok=(v == 'comfortable'), video=ax['vc'], ref=refv + refa, text=text - 30)  # text minus prompt 30
    return pts


def feasible(sel, use_text):
    # scan w in steps of 0.01, find range where max(ok cost) < min(oom cost)
    lo = None; hi = None
    for i in range(0, 301):
        w = i / 100
        c = {l: p['video'] + w * (p['ref'] + (p['text'] if use_text else 0)) for l, p in sel.items()}
        mo = max(c[l] for l in c if sel[l]['ok']); mn = min(c[l] for l in c if not sel[l]['ok'])
        if mo < mn:
            lo = w if lo is None else lo; hi = w
    return lo, hi


def separates_at(sel, w, use_text):
    c = {l: p['video'] + w * (p['ref'] + (p['text'] if use_text else 0)) for l, p in sel.items()}
    return max(c[l] for l in c if sel[l]['ok']) < min(c[l] for l in c if not sel[l]['ok'])


def selections(pts):
    """[(kind, {label: point})] : each kind plus the reference-free points, then all points."""
    none = {l: p for l, p in pts.items() if p['kind'] == 'none'}
    out = []
    for kind in ('picture', 'video', 'audio'):
        out.append((kind, {**none, **{l: p for l, p in pts.items() if p['kind'] == kind}}))
    out.append(('all', dict(pts)))
    return out


def main():
    pts = load_points()
    for k, v in sorted(pts.items()):
        print(k, v)
    for kind, sel in selections(pts):
        if kind == 'all':
            print('ALL ref only', feasible(sel, False), ' ref+text', feasible(sel, True))
        else:
            print(kind, 'ref only w in', feasible(sel, False), ' ref+text w in', feasible(sel, True))
    # per picture consumed video rows interval (9 pictures): from p9 OK/OOM vs none band
    print('p9: OK video', pts['i1080_226_p9']['video'], 'OOM video', pts['i1080_243_p9']['video'], 'none band', pts['r1080_260']['video'], pts['r1080_277']['video'])


if __name__ == '__main__':
    main()
