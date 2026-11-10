"""Every study as a list of run configs, with results cached by config hash.

    results/<study>.<profile>.json   ->  {"runs": [record, ...]}

A run that appears in two studies (e.g. the AdamW points shared by the LR sweep and the Muon
comparison) is trained once: `pending()` skips any config whose hash is already stored anywhere.

Profiles
    full   the reported runs (Modal A100)
    quick  the same studies shrunk until a laptop CPU runs them in minutes — a smoke test, not a result
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'

PROFILES = {
    'full': dict(base=dict(n_layer=6, width=384, head_dim=64, block=256), B=32, n_val=16,
                 bc_steps=800, bc_warm=[40, 400], ratio_steps=600, ratio_warm=[0, 25, 50, 100, 200], ratio_lrs=[1e-3, 3e-3],
                 sch_steps=300, sch_stop=200, sch_warm=30, sch_lrs=[2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3], sch_seeds=[0, 1, 2],
                 sw_model=dict(n_layer=4, head_dim=64, block=256), sw_steps=600, sw_warm=30,
                 sw_widths=[256, 512, 1024], sw_holdout=[2048], sw_k=list(range(-2, 10)), sw_seeds=[0, 1],
                 mu_widths=[256, 1024], mu_k=list(range(-1, 9)), mu_seeds=[0, 1],
                 # second pass, chosen AFTER the coarse sweep: x sqrt(2) either side of each coarse minimum
                 sw_refine={('sp', 256): [3.5, 4.5], ('sp', 512): [1.5, 2.5], ('sp', 1024): [0.5, 1.5], ('sp', 2048): [0.5, 1.5],
                            ('mup', 256): [3.5, 4.5], ('mup', 512): [3.5, 4.5], ('mup', 1024): [3.5, 4.5], ('mup', 2048): [3.5, 4.5]}),
    'quick': dict(base=dict(n_layer=2, width=64, head_dim=16, block=64), B=8, n_val=4,
                  bc_steps=80, bc_warm=[10, 40], ratio_steps=80, ratio_warm=[0, 10, 20], ratio_lrs=[1e-3, 3e-3],
                  sch_steps=60, sch_stop=40, sch_warm=4, sch_lrs=[1e-3, 3e-3], sch_seeds=[0],
                  sw_model=dict(n_layer=2, head_dim=16, block=64), sw_steps=50, sw_warm=5,
                  sw_widths=[32, 64, 128], sw_holdout=[256], sw_k=list(range(2, 9)), sw_seeds=[0],
                  mu_widths=[32, 128], mu_k=list(range(3, 9)), mu_seeds=[0]),
}


def chash(cfg):
    c = {k: v for k, v in cfg.items() if k not in ('name', 'study')}
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


def _cfg(P, name, **kw):
    c = dict(name=name, seed=0, model=dict(P['base']), batch=P['B'], n_val=P['n_val'], clip=1.0)
    c.update(kw)
    return c


# ─────────────── §2  bias correction in a real model ───────────────
def bias_correction(P):
    out = []
    for b2 in (0.999, 0.99):
        for W in [0] + P['bc_warm']:
            for bc in (True, False):
                for seed in (0, 1):
                    out.append(_cfg(P, f'bc={int(bc)} warm={W} b2={b2} s{seed}', seed=seed, steps=P['bc_steps'],
                                    sched=dict(kind='const', total=P['bc_steps'], warmup=W),
                                    opt=dict(kind='hand', lr=1e-3, wd=0.1, betas=[0.9, b2], bias_correction=bc),
                                    eval_every=20, log_ratio=True))
    return out


# ─────────────── §3  update-to-weight ratio under warmup ───────────────
def warmup_ratio(P):
    out = []
    for lr in P['ratio_lrs']:
        for W in P['ratio_warm']:
            for seed in (0, 1):
                out.append(_cfg(P, f'lr={lr} warm={W} s{seed}', seed=seed, steps=P['ratio_steps'],
                                sched=dict(kind='const', total=P['ratio_steps'], warmup=W),
                                opt=dict(kind='adamw', lr=lr, wd=0.1), eval_every=25, log_ratio=True))
    return out


# ─────────────── §4  cosine vs WSD, stopped at 200 of 300 ───────────────
def schedules(P):
    S, stop, W = P['sch_steps'], P['sch_stop'], P['sch_warm']
    D = S // 10                                  # WSD decays over the last 10% of the planned run
    ev = dict(eval_every=max(5, S // 30), eval_at=[stop - D, stop, S])
    out = []
    for lr in P['sch_lrs']:
        for seed in P['sch_seeds']:
            o = dict(kind='adamw', lr=lr, wd=0.1)
            out += [
                _cfg(P, f'cosine-{S} lr={lr} s{seed}', seed=seed, steps=S, opt=o, **ev,
                     sched=dict(kind='cosine', total=S, warmup=W, floor=0.0)),
                _cfg(P, f'cosine-{stop} lr={lr} s{seed}', seed=seed, steps=stop, opt=o, **ev,
                     sched=dict(kind='cosine', total=stop, warmup=W, floor=0.0)),
                _cfg(P, f'wsd-{S} lr={lr} s{seed}', seed=seed, steps=S, opt=o, **ev,
                     sched=dict(kind='wsd', total=S, warmup=W, decay_len=D, floor=0.0),
                     branches=[dict(at=stop - D, len=D, shape='linear'),        # same compute as cosine-200
                               dict(at=stop - D, len=D, shape='sqrt'),
                               dict(at=stop, len=D, shape='linear'),            # +D extra steps
                               dict(at=stop, len=2 * D, shape='linear')]),
                _cfg(P, f'const+ema lr={lr} s{seed}', seed=seed, steps=S, opt=o, **ev,
                     sched=dict(kind='const', total=S, warmup=W), ema=[0.9, 0.95, 0.98, 0.99]),
            ]
    return out


# ─────────────── §5  LR sweep across width, SP and muP ───────────────
def sweep_lrs(ks):
    return [round(1e-4 * 2 ** k, 10) for k in ks]


def _sw(P, width, param, lr, seed, opt='adamw', betas=None):
    S = P['sw_steps']
    o = dict(kind=opt, lr=lr, wd=0.0)
    if betas:
        o['betas'] = betas
    tag = opt + (f'(b2={betas[1]})' if betas else '')
    return _cfg(P, f'{param} w={width} {tag} lr={lr:.3g} s{seed}', seed=seed, steps=S,
                model=dict(P['sw_model'], width=width, param=param, base_width=P['sw_widths'][0]),
                sched=dict(kind='cosine', total=S, warmup=P['sw_warm'], floor=0.1),
                opt=o, eval_every=S // 4)


def sweep(P):
    coarse = [_sw(P, w, prm, lr, s) for prm in ('sp', 'mup') for w in P['sw_widths'] + P['sw_holdout']
              for lr in sweep_lrs(P['sw_k']) for s in P['sw_seeds']]
    fine = [_sw(P, w, prm, lr, s) for (prm, w), ks in P.get('sw_refine', {}).items()
            for lr in sweep_lrs(ks) for s in P['sw_seeds']]
    return coarse + fine


# ─────────────── §6  AdamW vs Muon, both tuned ───────────────
def muon(P):
    variants = [('adamw', None), ('adamw', [0.9, 0.95]), ('muon', None)]     # AdamW tuned on LR only, then on LR and beta2
    return [_sw(P, w, 'sp', lr, s, opt, betas) for opt, betas in variants for w in P['mu_widths']
            for lr in sweep_lrs(P['mu_k']) for s in P['mu_seeds']]


STUDIES = dict(bias_correction=bias_correction, warmup_ratio=warmup_ratio, schedules=schedules,
               sweep=sweep, muon=muon)


def configs(study, profile='full'):
    out = []
    for c in STUDIES[study](PROFILES[profile]):
        c['study'] = study
        out.append(c)
    return out


# ─────────────── storage ───────────────
def path(study, profile):
    return RES / f'{study}.{profile}.json'


def load(study, profile='full'):
    p = path(study, profile)
    return json.loads(p.read_text())['runs'] if p.exists() else []


def index(profile):
    idx = {}
    for s in STUDIES:
        for r in load(s, profile):
            idx[chash(r['cfg'])] = r
    return idx


def pending(study, profile='full'):
    idx = index(profile)
    have = {chash(r['cfg']) for r in load(study, profile)}
    todo, reuse = [], []
    for c in configs(study, profile):
        h = chash(c)
        if h in have:
            continue
        (reuse if h in idx else todo).append(idx.get(h, c))
    if reuse:                                     # trained already for another study: copy, don't retrain
        for r in reuse:
            store(study, profile, dict(r, cfg=dict(r['cfg'], study=study)))
    return todo


def store(study, profile, rec):
    RES.mkdir(exist_ok=True)
    runs = load(study, profile)
    h = chash(rec['cfg'])
    runs = [r for r in runs if chash(r['cfg']) != h] + [rec]
    path(study, profile).write_text(json.dumps(dict(runs=runs), separators=(',', ':')))


def run_local(study, profile='quick', verbose=True):
    from .train import run
    todo = pending(study, profile)
    for i, c in enumerate(todo):
        r = run(c)
        store(study, profile, r)
        if verbose:
            print(f'[{study}] {i + 1}/{len(todo)} {c["name"]:<40} val {r["final_val"]:.4f}  {r["time_s"]:.1f}s')
    return load(study, profile)


def get(study, profile='full', allow_run=False):
    runs = load(study, profile)
    want = {chash(c) for c in configs(study, profile)}
    if allow_run and len({chash(r['cfg']) for r in runs} & want) < len(want):
        runs = run_local(study, profile)
    return [r for r in runs if chash(r['cfg']) in want]
