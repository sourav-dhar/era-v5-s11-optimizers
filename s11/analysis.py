"""Turning runs into numbers: minima with uncertainty, extrapolation across width, convergence steps."""
import math
from collections import defaultdict

import numpy as np

DIVERGED = 4.5          # loss assigned to a diverged run when a curve needs a value (≈ above the unigram loss)


def val_at(run, step):
    for s, v in run['val']:
        if s == step:
            return v
    return float('nan')


def group(runs, key):
    g = defaultdict(list)
    for r in runs:
        g[key(r)].append(r)
    return dict(g)


def curve(runs):
    """{lr: [final val per seed]} for one (param, width, opt) slice; diverged -> nan."""
    out = defaultdict(list)
    for r in runs:
        out[r['cfg']['opt']['lr']].append(float('nan') if r['diverged'] else r['final_val'])
    return dict(sorted(out.items()))


def fit_min(lrs, losses, k=3):
    """Parabola in log2(lr) through the observed minimum and its two neighbours -> (lr*, loss*).
    Falls back to the observed minimum if the parabola opens downwards or leaves the window."""
    x = np.log2(np.asarray(lrs, float))
    y = np.asarray(losses, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float('nan'), float('nan')
    i = int(np.argmin(y))
    lo, hi = max(0, i - k // 2), min(len(x), i + k // 2 + 1)
    xs, ys = x[lo:hi], y[lo:hi]
    if len(xs) >= 3:
        a, b, c = np.polyfit(xs, ys, 2)
        if a > 0:
            xm = -b / (2 * a)
            if xs.min() <= xm <= xs.max():
                return float(2 ** xm), float(c - b * b / (4 * a))
    return float(2 ** x[i]), float(y[i])


def boot_min(curve_by_lr, n=2000, seed=0):
    """Bootstrap the minimum: at every LR draw one of the seeds (with replacement), refit."""
    rng = np.random.default_rng(seed)
    lrs = list(curve_by_lr)
    vals = [np.asarray(curve_by_lr[l], float) for l in lrs]
    draws = []
    for _ in range(n):
        ys = [v[rng.integers(len(v))] if len(v) else np.nan for v in vals]
        draws.append(fit_min(lrs, ys)[0])
    return np.asarray(draws)


def width_law(widths, lr_draws, target=4096):
    """Fit log lr* = a + b log w to bootstrap draws (one draw per width each time).
    Returns slope draws and predicted lr*(target) draws."""
    W = np.log(np.asarray(widths, float))
    D = np.stack([np.log(d) for d in lr_draws], 1)          # (n, widths)
    slopes, preds = [], []
    for row in D:
        if not np.all(np.isfinite(row)):
            continue
        b, a = np.polyfit(W, row, 1)
        slopes.append(b)
        preds.append(math.exp(a + b * math.log(target)))
    return np.asarray(slopes), np.asarray(preds)


def ci(x, q=(5, 95)):
    x = np.asarray(x)[np.isfinite(x)]
    return tuple(float(v) for v in np.percentile(x, q)) if len(x) else (float('nan'), float('nan'))


def settle_step(a, b, rel=0.01, smooth=1):
    """First index after which |a/b - 1| stays below `rel` for the rest of the series."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        a, b = np.convolve(a, k, 'valid'), np.convolve(b, k, 'valid')
    bad = np.where(~(np.abs(a / b - 1) < rel))[0]
    if len(bad) == 0:
        return 0
    return int(bad[-1] + 1) if bad[-1] + 1 < len(a) else None


def ema(x, a=0.9):
    out, m = [], None
    for v in x:
        m = v if m is None else a * m + (1 - a) * v
        out.append(m)
    return np.array(out)


def rolling_median(x, w=25):
    x = np.asarray(x, float)
    return np.array([np.nanmedian(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])


def join_step(a, b, tol=0.25, allow=0.10, win=25):
    """First step (1-based) from which series a agrees with series b within a factor (1 + tol), comparing
    `win`-step rolling medians, allowing at most `allow` of the later steps to fall outside the band."""
    r = np.log(rolling_median(a, win) / rolling_median(b, win))
    ok = np.abs(r) < np.log(1 + tol)
    bad_after = np.cumsum((~ok)[::-1])[::-1] / np.arange(len(ok), 0, -1)
    for t in range(len(ok)):
        if ok[t] and bad_after[t] <= allow:
            return t + 1
    return None
