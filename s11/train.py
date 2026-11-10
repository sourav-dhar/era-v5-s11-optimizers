"""One training run, described by a JSON-able config, returning a JSON-able record of what it measured.

cfg = dict(
    name, seed,
    model   = dict(n_layer, width, head_dim, block, param='sp'|'mup', base_width),
    batch, steps,
    sched   = dict(kind='const'|'cosine'|'wsd', total, warmup, floor, decay_frac | decay_len, shape),
    opt     = dict(kind='adamw'|'hand'|'muon', lr, wd, betas, eps, bias_correction),
    clip    = 1.0,
    eval_every, eval_at=[...], n_val,
    log_ratio = False,              # per-layer ||dW|| / ||W|| and the step in units of the LR, every step
    ema       = [0.95, ...],        # exponential moving averages of the weights, evaluated at eval steps
    branches  = [dict(at, len, shape)],   # WSD: from the checkpoint at step `at`, decay over `len` steps
)
"""
import copy
import math
import time
from contextlib import nullcontext

import torch

from . import optim
from .data import CharData
from .model import GPT, ModelCfg, short_name
from .schedules import decay_shape, factor

_DATA = {}


def get_data(device):
    if device not in _DATA:
        _DATA[device] = CharData(device)
    return _DATA[device]


def _r(x, sig=5):
    x = float(x)
    return x if not math.isfinite(x) else float(f'{x:.{sig}g}')


def run(cfg: dict) -> dict:
    t_start = time.time()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    data = get_data(dev)
    seed = cfg.get('seed', 0)
    torch.manual_seed(seed)
    mc = ModelCfg(vocab=data.V, **cfg['model'])
    model = GPT(mc).to(dev)
    o = cfg['opt']
    opts = optim.build(model, o, fused=(dev == 'cuda' and o.get('kind', 'adamw') != 'hand'))
    groups = lambda: [g for op in opts for g in op.param_groups]   # re-read: load_state_dict replaces the dicts
    peak, sched, S, B, T = o['lr'], cfg['sched'], cfg['steps'], cfg['batch'], mc.block
    clip = cfg.get('clip', 1.0)
    amp = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if dev == 'cuda' else nullcontext
    val = data.val_set(cfg.get('n_val', 16), B, T)
    ev = cfg.get('eval_every', 0)
    eval_at = set(cfg.get('eval_at', [])) | (set(range(ev, S + 1, ev)) if ev else set()) | {S}

    # per-tensor bookkeeping for the update-to-weight ratio
    tracked = [(n, p) for n, p in model.named_parameters() if p.dim() >= 2]
    tparams = [p for _, p in tracked]
    numel = torch.tensor([p.numel() for p in tparams], dtype=torch.float32, device=dev)
    log_ratio = cfg.get('log_ratio', False)

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = torch.zeros((), device=dev)
        for x, y in val:
            with amp():
                tot += model(x, y).float()
        model.train()
        return (tot / len(val)).item()

    @torch.no_grad()
    def eval_with(tensors):
        ps = list(model.parameters())
        backup = [p.detach().clone() for p in ps]
        for p, e in zip(ps, tensors):
            p.copy_(e)
        r = evaluate()
        for p, b in zip(ps, backup):
            p.copy_(b)
        return r

    def one_step(step, f, log=False):
        for g in groups():
            g['lr'] = peak * f * g['lr_mult']
        x, y = data.batch('train', B, T, seed, step)
        with amp():
            loss = model(x, y)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else float('inf'))
        if log:
            prev = [p.detach().clone() for p in tparams]
        for op in opts:
            op.step()
        for op in opts:
            op.zero_grad(set_to_none=True)
        rec = None
        if log:
            with torch.no_grad():
                gof = {id(p): g for g in groups() for p in g['params']}
                cur = [p.detach() for p in tparams]
                dw = torch._foreach_sub(cur, prev)
                dn = torch.stack(torch._foreach_norm(dw)).float()
                wn = torch.stack(torch._foreach_norm(prev)).float()
                lr_eff = torch.tensor([gof[id(p)]['lr'] for p in tparams], device=dev)
                wd = torch.tensor([gof[id(p)]['weight_decay'] for p in tparams], device=dev)
                # remove the decoupled-decay part, leaving the optimizer's own step, in units of its LR
                pure = torch._foreach_add(dw, torch._foreach_mul(prev, [float(l * w) for l, w in zip(lr_eff.tolist(), wd.tolist())]))
                pn = torch.stack(torch._foreach_norm(pure)).float()
                units = torch.where(lr_eff > 1e-12, pn / (lr_eff.clamp(min=1e-30) * numel.sqrt()),
                                    torch.full_like(pn, float('nan')))
                rec = (dn / wn, units)
        return loss.detach(), gn.detach(), rec

    emas = {d: [p.detach().clone().float() for p in model.parameters()] for d in cfg.get('ema', [])}
    out = dict(cfg=cfg, device=dev, gpu=torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu',
               n_params=sum(p.numel() for p in model.parameters()), diverged=False, diverged_at=None,
               val=[], ema={str(d): [] for d in emas}, branches=[])
    losses, gnorms, lrs, ratios, units = [], [], [], [], []
    if 0 in eval_at:
        out['val'].append([0, _r(evaluate())])
    branches = sorted(cfg.get('branches', []), key=lambda b: b['at'])
    t_train = 0.0

    def do_branches(at):
        for br in [b for b in branches if b['at'] == at]:
            saved_m = copy.deepcopy(model.state_dict())
            saved_o = [copy.deepcopy(op.state_dict()) for op in opts]
            f0 = factor(at - 1, sched)
            bl = []
            for k in range(br['len']):
                l, _, _ = one_step(at + k, f0 * decay_shape((k + 1) / br['len'], br.get('shape', 'linear')))
                bl.append(l)
            out['branches'].append(dict(br, val=_r(evaluate()), train_loss=[_r(v) for v in torch.stack(bl).tolist()]))
            model.load_state_dict(saved_m)
            for op, so in zip(opts, saved_o):
                op.load_state_dict(so)

    for step in range(S):
        do_branches(step)
        f = factor(step, sched)
        t0 = time.time()
        loss, gn, rec = one_step(step, f, log_ratio)
        t_train += time.time() - t0
        losses.append(loss); gnorms.append(gn); lrs.append(peak * f)
        if rec is not None:
            ratios.append(rec[0]); units.append(rec[1])
        k = step + 1
        if emas:
            with torch.no_grad():
                ps = [p.detach().float() for p in model.parameters()]
                for d, e in emas.items():
                    torch._foreach_lerp_(e, ps, 1 - min(d, (1 + k) / (10 + k)))
        if k % 10 == 0 or k == S:
            lv = loss.item()
            if not math.isfinite(lv) or (k > 30 and lv > 6.0):
                out['diverged'], out['diverged_at'] = True, k
                break
        if k in eval_at:
            out['val'].append([k, _r(evaluate())])
            for d, e in emas.items():
                out['ema'][str(d)].append([k, _r(eval_with(e))])
    if not out['diverged']:
        do_branches(S)

    if dev == 'cuda':
        torch.cuda.synchronize()
    out['train_loss'] = [_r(v) for v in torch.stack(losses).float().tolist()]
    out['grad_norm'] = [_r(v, 4) for v in torch.stack(gnorms).float().tolist()]
    out['lr'] = [_r(v, 4) for v in lrs]
    out['final_val'] = out['val'][-1][1] if (out['val'] and not out['diverged']) else float('nan')
    if ratios:
        out['ratio'] = dict(names=[short_name(n) for n, _ in tracked], kinds=[GPT.kind(n) for n, _ in tracked],
                            ratio=[[_r(v, 4) for v in row] for row in torch.stack(ratios).tolist()],
                            units=[[_r(v, 4) for v in row] for row in torch.stack(units).tolist()])
    out['time_s'] = round(time.time() - t_start, 2)
    out['tok_per_s'] = round(len(losses) * B * T / max(t_train, 1e-9))
    return out
