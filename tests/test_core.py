"""Checks that everything the studies rely on does what it claims.  Run:  python -m pytest -q tests

Each test guards a claim the write-up makes, or a bug that was actually hit while building this repo.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s11.optim import HandAdamW, Muon, adam_by_hand, bias_factor  # noqa: E402
from s11.model import GPT, ModelCfg  # noqa: E402
from s11.train import run  # noqa: E402

G5 = [0.50, 0.40, 0.60, 0.45, 0.55]


def test_lesson_table():
    """The lesson §6 table, to the digits it prints."""
    rows = adam_by_hand(1.0, G5)
    assert [round(r['w'], 6) for r in rows] == [0.999, 0.998012, 0.997018, 0.996028, 0.995031]
    assert round(rows[1]['v_hat'], 4) == 0.2050


def test_bias_factor_closed_form():
    """Without bias correction, step 1 is 3.16 eta for any gradient (lesson §6) and the error peaks at t = 12."""
    off = adam_by_hand(1.0, [0.37] * 20, bias_correction=False)
    on = adam_by_hand(1.0, [0.37] * 20, bias_correction=True)
    for t in range(20):
        assert math.isclose(off[t]['step'] / on[t]['step'], bias_factor(t + 1), rel_tol=1e-6)
    assert round(bias_factor(1), 2) == 3.16
    assert max(range(1, 200), key=bias_factor) == 12


def test_hand_adamw_matches_torch():
    torch.manual_seed(0)
    W = torch.randn(17, 9, dtype=torch.float64)
    a, b = W.clone().requires_grad_(), W.clone().requires_grad_()
    oa = HandAdamW([a], lr=1e-3, weight_decay=0.1)
    ob = torch.optim.AdamW([b], lr=1e-3, weight_decay=0.1)
    for _ in range(25):
        g = torch.randn_like(W)
        a.grad, b.grad = g.clone(), g.clone()
        oa.step(); ob.step()
    assert (a - b).abs().max().item() < 1e-14


def test_muon_matches_torch():
    if not hasattr(torch.optim, 'Muon'):
        return
    torch.manual_seed(0)
    W = torch.randn(48, 16)
    a, b = W.clone().requires_grad_(), W.clone().requires_grad_()
    oa = Muon([a], lr=0.02, weight_decay=0.1)
    ob = torch.optim.Muon([b], lr=0.02, weight_decay=0.1, adjust_lr_fn='match_rms_adamw')
    for _ in range(10):
        g = torch.randn_like(W)
        a.grad, b.grad = g.clone(), g.clone()
        oa.step(); ob.step()
    assert torch.equal(a, b)


def test_mup_equals_sp_at_base_width():
    """At the base width the two parameterisations must be the same model."""
    torch.manual_seed(0); sp = GPT(ModelCfg(n_layer=2, width=128, head_dim=32, block=32, param='sp', base_width=128))
    torch.manual_seed(0); mp = GPT(ModelCfg(n_layer=2, width=128, head_dim=32, block=32, param='mup', base_width=128))
    x = torch.randint(65, (2, 32))
    assert torch.equal(sp(x), mp(x))
    assert all(g['lr_mult'] == 1.0 for g in mp.param_groups(0.1))


TINY = dict(seed=0, model=dict(n_layer=2, width=64, head_dim=16, block=64), batch=8, n_val=2,
            opt=dict(kind='adamw', lr=3e-3, wd=0.1))


def test_branch_equals_independent_run():
    """A WSD decay branch taken from a checkpoint must equal a run planned for that length — and must not
    disturb the main run. (Regression: load_state_dict replaces param_groups; setting the LR on the stale
    dicts silently froze it at the peak.)"""
    sched = dict(kind='wsd', total=30, warmup=3, decay_len=6)
    main = run(dict(TINY, steps=30, sched=sched, branches=[dict(at=10, len=6, shape='linear'),
                                                           dict(at=10, len=6, shape='sqrt')]))
    plain = run(dict(TINY, steps=30, sched=sched))
    short = run(dict(TINY, steps=16, sched=dict(sched, total=16)))
    assert main['final_val'] == plain['final_val']
    assert main['branches'][0]['val'] == short['final_val']
    assert main['branches'][1]['val'] != main['branches'][0]['val']


def test_ratio_units_at_step_one():
    """On step 1 Adam's step is exactly eta per element (bias-corrected) or 3.16 eta (uncorrected)."""
    for bc, want in [(True, 1.0), (False, math.sqrt(1000) / 10)]:
        r = run(dict(TINY, steps=1, sched=dict(kind='const', total=1), log_ratio=True, clip=0,
                     opt=dict(kind='hand', lr=1e-3, wd=0.0, bias_correction=bc)))
        med = sorted(r['ratio']['units'][0])[len(r['ratio']['units'][0]) // 2]
        assert abs(med - want) / want < 0.02
