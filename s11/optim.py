"""Optimizers, written out so that every line can be read against the lesson.

* `adam_by_hand`   — one scalar weight, pure Python floats (fp64), every intermediate returned.
* `HandAdamW`      — the same rule for whole tensors, with bias correction switchable. Verified
                     bit-for-bit-ish against `torch.optim.AdamW` in the notebook (§1) before it is
                     trusted for the bias-correction study (§2).
* `Muon`           — momentum -> Newton–Schulz orthogonalisation -> RMS-matched step, a line-for-line
                     re-implementation of `torch.optim.Muon` (checked against it when available) so the
                     study runs on any torch version.
* `build`          — the optimizer(s) for a model + config, with muP's per-group LR multipliers.
"""
import math

import torch
from torch.optim import Optimizer


# ─────────────────────────────── one weight, by hand ───────────────────────────────
def adam_by_hand(w, grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, bias_correction=True, decoupled=True):
    """Adam / AdamW on one scalar, written exactly as the lesson writes it. Returns one dict per step."""
    m = v = 0.0
    rows = []
    for t, g in enumerate(grads, start=1):
        if wd and not decoupled:          # L2: the penalty's gradient joins g (and is divided by sqrt(v))
            g = g + wd * w
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t) if bias_correction else m
        vh = v / (1 - b2 ** t) if bias_correction else v
        adam = mh / (math.sqrt(vh) + eps)
        w_before = w
        if wd and decoupled:              # AdamW: shrink first (torch's order), nothing divides it
            w = w * (1 - lr * wd)
        w = w - lr * adam
        rows.append(dict(t=t, g=g, m=m, v=v, m_hat=mh, v_hat=vh, step=w - w_before, w=w, step_in_lr=-(w - w_before) / lr))
    return rows


def bias_factor(t, b1=0.9, b2=0.999):
    """Uncorrected step / corrected step for a steady gradient:  (1 - b1^t) / sqrt(1 - b2^t)."""
    return (1 - b1 ** t) / math.sqrt(1 - b2 ** t)


# ─────────────────────────────── AdamW, tensors, switchable ─────────────────────────
class HandAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, bias_correction=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      bias_correction=bias_correction))

    @torch.no_grad()
    def step(self):
        for gr in self.param_groups:
            b1, b2 = gr['betas']
            for p in gr['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st['step'] = 0
                    st['exp_avg'] = torch.zeros_like(p)
                    st['exp_avg_sq'] = torch.zeros_like(p)
                st['step'] += 1
                t, m, v, g = st['step'], st['exp_avg'], st['exp_avg_sq'], p.grad
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1 = 1 - b1 ** t if gr['bias_correction'] else 1.0
                bc2 = 1 - b2 ** t if gr['bias_correction'] else 1.0
                p.mul_(1 - gr['lr'] * gr['weight_decay'])
                denom = (v / bc2).sqrt_().add_(gr['eps'])
                p.addcdiv_(m / bc1, denom, value=-gr['lr'])


# ─────────────────────────────── Muon ──────────────────────────────────────────────
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(G, steps=5, eps=1e-7, coeffs=NS_COEFFS):
    """Approximately the nearest semi-orthogonal matrix to G (all singular values pushed towards 1)."""
    a, b, c = coeffs
    X = G.to(torch.bfloat16, copy=True)
    tall = G.size(0) > G.size(1)
    if tall:
        X = X.T
    X.div_(X.norm().clamp(min=eps))
    for _ in range(steps):
        A = X @ X.T
        Bm = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, Bm, X, beta=a)
    return X.T if tall else X


class Muon(Optimizer):
    """For 2-D weight matrices only. `match_rms_adamw` scales the step by 0.2*sqrt(max(A, B)) so that
    one learning rate (and one weight decay) can be shared with the AdamW-managed parameters
    (Liu et al. 2025, "Muon is Scalable for LLM Training")."""

    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for gr in self.param_groups:
            mu = gr['momentum']
            for p in gr['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st['momentum_buffer'] = torch.zeros_like(p)
                buf = st['momentum_buffer']
                buf.lerp_(p.grad, 1 - mu)
                upd = p.grad.lerp(buf, mu) if gr['nesterov'] else buf
                upd = newton_schulz(upd, gr['ns_steps'])
                A, B = p.shape
                p.mul_(1 - gr['lr'] * gr['weight_decay'])
                p.add_(upd, alpha=-gr['lr'] * 0.2 * math.sqrt(max(A, B)))


# ─────────────────────────────── build ─────────────────────────────────────────────
def build(model, o: dict, fused=False):
    """o = dict(kind='adamw'|'hand'|'muon', lr, wd, betas, eps, bias_correction). Returns a list of
    optimizers; every param group carries `lr_mult`, so a schedule sets  lr = peak * f(t) * lr_mult."""
    groups = model.param_groups(o.get('wd', 0.0))
    betas, eps = tuple(o.get('betas', (0.9, 0.999))), o.get('eps', 1e-8)
    kind = o.get('kind', 'adamw')
    for g in groups:
        g['lr'] = o['lr'] * g['lr_mult']
    if kind == 'adamw':
        return [torch.optim.AdamW(groups, betas=betas, eps=eps, **(dict(fused=True) if fused else {}))]
    if kind == 'hand':
        return [HandAdamW(groups, betas=betas, eps=eps, bias_correction=o.get('bias_correction', True))]
    if kind == 'muon':
        mat = [g for g in groups if g['kind'] == 'hidden']
        rest = [g for g in groups if g['kind'] != 'hidden']
        return [Muon(mat, momentum=o.get('momentum', 0.95)),
                torch.optim.AdamW(rest, betas=betas, eps=eps, **(dict(fused=True) if fused else {}))]
    raise ValueError(kind)
