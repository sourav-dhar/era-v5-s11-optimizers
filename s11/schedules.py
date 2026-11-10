"""Learning-rate schedules as a multiplier f(step) of the peak rate. `step` is 0-indexed: f(step) is
the rate used for the update that *produces* weights after step+1 steps.

    const   warmup, then flat
    cosine  warmup, then peak -> floor along half a cosine, ending exactly at `total`
    wsd     warmup, stable (flat), then decay over the last `decay_frac` of `total` (or `decay_len` steps)

Decay shapes for WSD: 'linear' (1 - p) and 'sqrt' (1 - sqrt(p), Hägele et al. 2024).
"""
import math


def warm(step, W):
    return min(1.0, (step + 1) / W) if W and W > 0 else 1.0


def decay_shape(p, shape='linear'):
    p = min(max(p, 0.0), 1.0)
    return 1 - math.sqrt(p) if shape == 'sqrt' else 1 - p


def factor(step, s: dict):
    kind, W, total = s.get('kind', 'const'), s.get('warmup', 0), s.get('total', 1)
    floor = s.get('floor', 0.0)
    w = warm(step, W)
    if kind == 'const':
        return w
    if kind == 'cosine':
        if step < W:
            return w
        p = min(1.0, (step - W + 1) / max(1, total - W))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
    if kind == 'wsd':
        D = s.get('decay_len') or max(1, round(s.get('decay_frac', 0.1) * total))
        start = total - D
        if step < start:
            return w
        return floor + (1 - floor) * decay_shape((step - start + 1) / D, s.get('shape', 'linear'))
    raise ValueError(kind)


def curve(s, n=None):
    return [factor(i, s) for i in range(n or s['total'])]
