"""Compress results/*.json + results/summary.<profile>.json into site/data.js for the companion page.

    python tools/export_site.py --profile full
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s11 import analysis as A, experiments as E  # noqa: E402


def r4(x):
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else float(f'{x:.4g}')


def arr(a):
    return [r4(v) for v in np.asarray(a, float).ravel()]


def logidx(n, k=90):
    """~k step indices, log-spaced, always including the first and the last."""
    idx = np.unique(np.round(np.logspace(0, math.log10(n), k)).astype(int) - 1)
    return idx[(idx >= 0) & (idx < n)]


def main(profile):
    P = E.PROFILES[profile]
    S = json.loads((ROOT / f'results/summary.{profile}.json').read_text())
    out = dict(profile=profile, summary=S)

    # bias correction in the real model
    bc = E.load('bias_correction', profile)
    g = A.group(bc, lambda r: (r['cfg']['opt']['betas'][1], r['cfg']['sched']['warmup'], r['cfg']['opt']['bias_correction']))
    out['bias'] = []
    for (b2, W, on), rs in sorted(g.items()):
        U = np.nanmean([np.nanmedian(np.array(r['ratio']['units'], float), 1) for r in rs], 0)
        idx = logidx(len(U), 70)
        V = np.nanmean([[v for _, v in r['val']] for r in rs], 0)
        out['bias'].append(dict(b2=b2, warmup=W, on=on, step=(idx + 1).tolist(), units=arr(U[idx]),
                                vstep=[s for s, _ in rs[0]['val']], val=arr(V)))

    # update-to-weight ratio
    wr = E.load('warmup_ratio', profile)
    g = A.group(wr, lambda r: (r['cfg']['opt']['lr'], r['cfg']['sched']['warmup']))
    names, kinds = wr[0]['ratio']['names'], wr[0]['ratio']['kinds']
    out['ratio'] = dict(names=names, kinds=kinds, runs=[])
    for (lr, W), rs in sorted(g.items()):
        M = np.nanmean([np.array(r['ratio']['ratio'], float) for r in rs], 0)
        Uu = np.nanmean([np.array(r['ratio']['units'], float) for r in rs], 0)
        idx = logidx(len(M), 80)
        out['ratio']['runs'].append(dict(lr=lr, warmup=W, step=(idx + 1).tolist(),
                                         M=[arr(M[i]) for i in idx], units=arr(np.nanmedian(Uu[idx][:, [k == 'hidden' for k in kinds]], 1)),
                                         final_val=r4(np.mean([r['final_val'] for r in rs]))))

    # schedules: mean val curves per run kind and LR
    sc = E.load('schedules', profile)
    g = A.group(sc, lambda r: (r['cfg']['name'].split(' ')[0], r['cfg']['opt']['lr']))
    out['sched'] = dict(S=P['sch_steps'], stop=P['sch_stop'], warmup=P['sch_warm'], D=P['sch_steps'] // 10, curves=[])
    for (k, lr), rs in sorted(g.items()):
        st = [s for s, _ in rs[0]['val']]
        V = np.nanmean([[v for _, v in r['val']] for r in rs], 0)
        br = {}
        for r in rs:
            for b in r['branches']:
                br.setdefault(f"{b['at']}-{b['len']}-{b['shape']}", []).append(b['val'])
        ema = {}
        for r in rs:
            for d, vals in r['ema'].items():
                ema.setdefault(d, []).append([v for _, v in vals])
        out['sched']['curves'].append(dict(kind=k, lr=lr, step=st, val=arr(V),
                                           branches={kk: r4(np.mean(v)) for kk, v in br.items()},
                                           ema={d: arr(np.nanmean(v, 0)) for d, v in ema.items()}))

    # LR sweep
    sw = E.load('sweep', profile)
    out['sweep'] = []
    for (prm, w), rs in sorted(A.group(sw, lambda r: (r['cfg']['model']['param'], r['cfg']['model']['width'])).items()):
        cv = A.curve(rs)
        out['sweep'].append(dict(param=prm, width=w, held_out=w in P['sw_holdout'], lr=list(cv),
                                 loss=[r4(np.nanmean(v)) if np.isfinite(v).any() else None for v in cv.values()]))

    # Muon
    def vkey(r):
        o = r['cfg']['opt']
        if o['kind'] == 'muon':
            return 'Muon'
        if r['cfg']['model']['param'] == 'mup':
            return 'AdamW muP'
        return 'AdamW β₂=0.95' if o.get('betas', [0.9, 0.999])[1] == 0.95 else 'AdamW β₂=0.999'
    mu = E.get('muon', profile) + [r for r in E.get('sweep', profile) if r['cfg']['model']['param'] == 'mup'
                                   and r['cfg']['model']['width'] in P['mu_widths']]
    out['muon'] = []
    for (v, w), rs in sorted(A.group(mu, lambda r: (vkey(r), r['cfg']['model']['width'])).items()):
        cv = A.curve(rs)
        means = {l: np.nanmean(x) if np.isfinite(x).any() else np.inf for l, x in cv.items()}
        best = min(means, key=means.get)
        bs = [r for r in rs if r['cfg']['opt']['lr'] == best]
        tl = np.mean([A.ema(r['train_loss'], 0.95) for r in bs], 0)
        idx = np.linspace(0, len(tl) - 1, 120).round().astype(int)
        out['muon'].append(dict(opt=v, width=w, lr=list(cv), loss=[r4(means[l]) if np.isfinite(means[l]) else None for l in cv],
                                best=best, step=(idx + 1).tolist(), train=arr(tl[idx]),
                                vstep=[s for s, _ in bs[0]['val']], val=arr(np.mean([[x for _, x in r['val']] for r in bs], 0))))

    js = 'window.S11 = ' + json.dumps(out, separators=(',', ':'), allow_nan=False) + ';\n'
    (ROOT / 'site').mkdir(exist_ok=True)
    (ROOT / 'site/data.js').write_text(js, encoding='utf-8')
    src = (ROOT / 'site/index.src.html').read_text(encoding='utf-8')
    (ROOT / 'site/index.html').write_text(src.replace('/*DATA*/', js.replace('</', '<\\/')), encoding='utf-8')
    print(f'site/data.js: {len(js) / 1024:.0f} KiB -> site/index.html')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='full')
    main(ap.parse_args().profile)
