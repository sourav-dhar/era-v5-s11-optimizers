"""The Session 10 GPT (pre-norm, learned positions, GELU MLP, untied head), parameterised two ways.

SP  (standard parameterisation) — exactly the Session 10 model: every weight N(0, 0.02), the
    residual projections additionally / sqrt(2L), one learning rate for every parameter.

muP (maximal-update parameterisation, Adam version, Yang et al. 2022 "Tensor Programs V", Table 3 / the `mup` package)
    relative to a base width (256), with width multiplier  r = width / base_width:
      * hidden matrices (qkv, attn proj, MLP up/down): init std  x 1/sqrt(r),  Adam LR  x 1/r
      * input embeddings, biases, LayerNorm:          unchanged init, unchanged LR
      * output head: logits multiplied by 1/r ("MuReadout"), unchanged LR
    Head dimension is held at 64 so the 1/d vs 1/sqrt(d) attention-scale question does not arise.
    At width == base_width the two parameterisations are *identical*, which is the sanity check.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelCfg:
    vocab: int = 65
    block: int = 256
    n_layer: int = 6
    width: int = 384
    head_dim: int = 64
    param: str = 'sp'          # 'sp' | 'mup'
    base_width: int = 256
    init_std: float = 0.02

    @property
    def r(self):
        return self.width / self.base_width if self.param == 'mup' else 1.0


class Attention(nn.Module):
    def __init__(s, c: ModelCfg):
        super().__init__()
        C = c.width
        s.nh, s.hd = C // c.head_dim, c.head_dim
        s.c_attn, s.c_proj = nn.Linear(C, 3 * C), nn.Linear(C, C)

    def forward(s, x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, s.nh, s.hd).transpose(1, 2) for t in s.c_attn(x).split(C, dim=2))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(s, c: ModelCfg):
        super().__init__()
        s.c_fc, s.c_proj = nn.Linear(c.width, 4 * c.width), nn.Linear(4 * c.width, c.width)

    def forward(s, x):
        return s.c_proj(F.gelu(s.c_fc(x)))


class Block(nn.Module):
    def __init__(s, c: ModelCfg):
        super().__init__()
        s.ln_1, s.attn, s.ln_2, s.mlp = nn.LayerNorm(c.width), Attention(c), nn.LayerNorm(c.width), MLP(c)

    def forward(s, x):
        x = x + s.attn(s.ln_1(x))
        return x + s.mlp(s.ln_2(x))


HIDDEN = ('c_attn.weight', 'c_proj.weight', 'c_fc.weight')


class GPT(nn.Module):
    def __init__(s, c: ModelCfg):
        super().__init__()
        s.cfg = c
        s.wte, s.wpe = nn.Embedding(c.vocab, c.width), nn.Embedding(c.block, c.width)
        s.h = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        s.ln_f, s.lm_head = nn.LayerNorm(c.width), nn.Linear(c.width, c.vocab, bias=False)
        s.readout_mult = 1.0 / c.r
        for n, p in s.named_parameters():
            if p.dim() < 2:
                nn.init.ones_(p) if 'ln' in n else nn.init.zeros_(p)
                continue
            std = c.init_std
            if s.kind(n) == 'hidden':
                std /= math.sqrt(c.r)
                if n.endswith('c_proj.weight'):
                    std /= math.sqrt(2 * c.n_layer)
            nn.init.normal_(p, 0.0, std)

    @staticmethod
    def kind(name):
        if name.startswith('h.') and name.endswith(HIDDEN):
            return 'hidden'
        if name.startswith(('wte', 'wpe')):
            return 'embed'
        if name.startswith('lm_head'):
            return 'head'
        return 'vector'

    def forward(s, idx, targets=None):
        B, T = idx.shape
        x = s.wte(idx) + s.wpe(torch.arange(T, device=idx.device))
        for b in s.h:
            x = b(x)
        logits = s.lm_head(s.ln_f(x)) * s.readout_mult
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))

    def param_groups(s, wd):
        """Groups carry `lr_mult` (muP's per-group LR scale) and `kind`. Vectors are never decayed."""
        groups = {}
        for n, p in s.named_parameters():
            k = s.kind(n)
            mult = 1.0 / s.cfg.r if k == 'hidden' else 1.0
            groups.setdefault(k, dict(params=[], names=[], kind=k, lr_mult=mult,
                                      weight_decay=0.0 if k == 'vector' else wd))
            groups[k]['params'].append(p)
            groups[k]['names'].append(n)
        return [groups[k] for k in ('hidden', 'embed', 'head', 'vector') if k in groups]


def short_name(n):
    """'h.3.attn.c_attn.weight' -> 'h3.qkv' — compact labels for per-layer plots."""
    n = n.replace('.weight', '')
    rep = {'attn.c_attn': 'qkv', 'attn.c_proj': 'attn.proj', 'mlp.c_fc': 'mlp.up', 'mlp.c_proj': 'mlp.down'}
    if n.startswith('h.'):
        _, i, rest = n.split('.', 2)
        return f'h{i}.{rep.get(rest, rest)}'
    return {'lm_head': 'head'}.get(n, n)


def n_params(c: ModelCfg):
    return sum(p.numel() for p in GPT(c).parameters())


__all__ = ['ModelCfg', 'GPT', 'short_name', 'n_params', 'asdict']
