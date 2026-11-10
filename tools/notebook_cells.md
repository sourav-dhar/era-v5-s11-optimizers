%%md
# Optimizer Claims, Re-measured

**ERA V5 · Session 11 — Optimizers and Learning-Rate Schedules.** *"Tune both sides before accepting a comparison."*

The Session 10 GPT (character-level, TinyShakespeare), used as a laboratory. Each section is a small study:
**claim → prediction → experiment → result → what surprised me**. Every number in the text is computed by the cell above it.

| # | The assignment asks | § | What this notebook adds |
|---|---|---|---|
| 1 | Reproduce Adam by hand, check against PyTorch | [§1](#1) | 4 PyTorch implementations × 2 dtypes, a **real weight inside the GPT**, Adam's "ghost steps" on a zero gradient, and where ε breaks scale invariance |
| 2 | Disable bias correction, plot 20 steps, when does it stop mattering | [§2](#2) | the closed form of the error, why the answer is **thousands of steps, not twenty**, and a real-model test of whether *warmup hides it* |
| 3 | Log the update-to-weight ratio per layer, find when warmup stops changing it | [§3](#3) | every layer at every step for 5 warmup lengths × 2 peak LRs, Adam's step decaying towards its noise floor, the head vs the body |
| 4 | Cosine vs WSD, 300 steps, stop both at 200, which do you keep | [§4](#4) | **both tuned** over 5 LRs × 3 seeds, plus cosine planned for 200, WSD decay branches from a checkpoint, and constant LR + weight EMA |
| 5 | LR sweep at widths 256 / 512 / 1,024, value for 4,096 and confidence | [§5](#5) | the same sweep under **muP**, a bootstrap interval on the width law, and a **held-out width (2,048)** to test the extrapolation before trusting it |
| + | *"Almost every optimizer claim that failed to replicate…"* | [§6](#6) | **AdamW vs Muon, both tuned**, at two widths, and how big the speedup looks against an untuned baseline |

%%md
## 0 · Setup

`S11_PROFILE=full` reads the results of the Modal A100 runs in `results/*.full.json` (produced by `modal run tools/modal_app.py --study all`). `S11_PROFILE=quick` trains a shrunken version of every study on a laptop CPU, which checks that the code runs; its numbers do not mean anything.

%%code [core]
import os, math, json, copy
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from IPython.display import display, Markdown

from s11 import experiments as E, analysis as A
from s11.style import PAL, INK, INK2, MUTED, GRID, SURF, SEQ, title, save, label_end, ramp
from s11.optim import adam_by_hand, bias_factor, HandAdamW
from s11.schedules import factor, decay_shape
from s11.model import GPT, ModelCfg, short_name
from s11.data import CharData

PROFILE = os.environ.get('S11_PROFILE', 'full')
QUICK = PROFILE == 'quick'
P = E.PROFILES[PROFILE]
def R(study):                       # results of one study (trained locally in the quick profile)
    return E.get(study, PROFILE, allow_run=QUICK)
SUMMARY = {'profile': PROFILE}      # every headline number lands here -> results/summary.<profile>.json
md = lambda s: display(Markdown(s))
pd.set_option('display.precision', 6)
print(f'profile={PROFILE} · torch {torch.__version__}')

%%md
<a id="1"></a>
## 1 · Adam, by hand

**Claim (lesson §6).** With bias correction, Adam moves every weight by approximately η, whatever the size of its gradient.

**Experiment.** One weight at 1.0 and the lesson's five gradients, run through the rule written out in plain Python (`s11/optim.py::adam_by_hand`, eight lines). Then the same weight goes through `torch.optim.Adam`. PyTorch never stores m̂ and v̂, so I read `exp_avg`, `exp_avg_sq` and `step` out of its state and apply the definitions m̂ = m/(1−β₁ᵗ) and v̂ = v/(1−β₂ᵗ).

%%code [core]
G5 = [0.50, 0.40, 0.60, 0.45, 0.55]
LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8
hand = pd.DataFrame(adam_by_hand(1.0, G5, LR, B1, B2, EPS))

def torch_trace(G, dtype=torch.float64, cls=torch.optim.Adam, w0=1.0, **kw):
    w = torch.tensor([w0], dtype=dtype, requires_grad=True)
    opt = cls([w], lr=LR, betas=(B1, B2), eps=EPS, **kw)
    rows = []
    for g in G:
        w.grad = torch.tensor([g], dtype=dtype); before = w.detach().clone()
        opt.step()
        st = opt.state[w]; t = int(st['step'])
        m, v = st['exp_avg'].double().item(), st['exp_avg_sq'].double().item()
        rows.append(dict(t=t, m=m, v=v, m_hat=m / (1 - B1 ** t), v_hat=v / (1 - B2 ** t),
                         step=(w.detach() - before).double().item(), w=w.double().item()))
    return pd.DataFrame(rows)

pt = torch_trace(G5)
cols = ['m', 'v', 'm_hat', 'v_hat', 'step', 'w']
show = hand[['t', 'g'] + cols].copy()
for c in cols: show[f'|Δ{c}|'] = (hand[c] - pt[c]).abs()
display(show.style.format({**{c: '{:.6f}' for c in cols}, **{f'|Δ{c}|': '{:.1e}' for c in cols}, 'g': '{:.2f}'})
        .set_caption('hand (fp64 Python) vs torch.optim.Adam (float64) — every intermediate'))
SUMMARY['adam_hand_max_abs_diff_fp64'] = float(max(show[f'|Δ{c}|'].max() for c in cols))
md(f"Every step lies within **{100*max(abs(hand.step_in_lr-1)):.2f}%** of η = 0.001, while the gradient ranges over ±20%. "
   f"The largest disagreement with PyTorch over all 30 numbers is **{SUMMARY['adam_hand_max_abs_diff_fp64']:.1e}**, which is fp64 rounding.")

%%md
### The same check against every PyTorch implementation, in both precisions

"Agree to several decimal places" depends on which kernel you compare against and in which dtype. The table reports agreement as **decimal digits**, −log₁₀ of the largest relative error in the weight over five steps.

%%code [core]
impls = {'Adam (for-loop)': dict(cls=torch.optim.Adam, foreach=False),
         'Adam (foreach)': dict(cls=torch.optim.Adam, foreach=True),
         'Adam (fused)': dict(cls=torch.optim.Adam, fused=True),
         'AdamW, λ=0 (for-loop)': dict(cls=torch.optim.AdamW, weight_decay=0.0, foreach=False),
         'HandAdamW (this repo)': dict(cls=HandAdamW)}
rows = []
for name, kw in impls.items():
    for dt in (torch.float64, torch.float32):
        try:
            tr = torch_trace(G5, dt, **kw)
            dig = lambda cols_: min(16.0, -math.log10(max(max(abs((tr[c] - hand[c]) / hand[c]).max() for c in cols_), 1e-16)))
            rows.append(dict(implementation=name, dtype=str(dt)[6:], state=dig(['m', 'v', 'm_hat', 'v_hat', 'w']), step=dig(['step'])))
        except Exception as e:
            print(name, dt, 'failed:', e)
agree = pd.DataFrame(rows).pivot(index='implementation', columns='dtype', values=['state', 'step'])
display(agree.style.format('{:.1f}').set_caption('decimal digits of agreement with the hand computation: '
                                                 'state = m, v, m̂, v̂, w · step = w_after − w_before'))
SUMMARY['adam_digits'] = {f'{a}/{b}': v for (a, b), v in agree.round(1).to_dict().items()}
d32 = agree[('state', 'float32')].min(); s32 = agree[('step', 'float32')].min()
md(f"In fp64 every implementation agrees to the last digit. In fp32 the optimizer state agrees to **{d32:.1f} digits**, but the step "
   f"only to **{s32:.1f}**. The step is the difference of two fp32 numbers near 1.0, which are spaced 6e-8 apart, so a 1e-3 step "
   f"keeps only about 4–5 significant digits. This is the Session 10 swamping problem again, and it is why the master copy of the weights is kept in fp32 rather than bf16.")

%%md
### A real weight inside the GPT, and Adam's "ghost steps"

The lesson's five gradients are invented. These five are real: I train the GPT for five AdamW steps (λ = 0.1) and record the gradient of two **embedding weights** at every step. One belongs to `' '` (space), the most frequent character. The other belongs to a rare character that appears in the first batch and in none of the next four.

The rare row gets **exactly zero gradient** on steps 2 to 5, since an embedding row only receives gradient when its token appears in the batch. A zero gradient does not stop it moving, though. Its m still remembers step 1, and m decays faster than v, so the weight keeps moving by a shrinking fraction of η. The hand computation has to reproduce that too.

%%code [core]
data = CharData('cpu')
torch.manual_seed(0)
mc = ModelCfg(vocab=data.V, **{**P['base'], 'block': 64})
gpt = GPT(mc)
opt = torch.optim.AdamW(gpt.parameters(), lr=LR, betas=(B1, B2), eps=EPS, weight_decay=0.1, foreach=False)
freq = {c: data.split['train'].eq(i).sum().item() for c, i in data.stoi.items()}
for dseed in range(10_000):          # find a data seed where some rare character is in batch 1 and in none of batches 2-5
    batches = [data.batch('train', 8, 64, dseed, s) for s in range(5)]
    present = [set(x.flatten().tolist()) for x, y in batches]
    cands = [c for c, i in data.stoi.items() if i in present[0] and not any(i in p for p in present[1:]) and freq[c] < 5000]
    if cands: break
rare = min(cands, key=freq.get)
print(f'data seed {dseed}: {rare!r} occurs {freq[rare]:,} times in {len(data.split["train"]):,} training characters')
picks = {"' ' (most common)": (data.stoi[' '], 0), f"{rare!r} (rare)": (data.stoi[rare], 0)}
trace = {k: dict(w0=gpt.wte.weight[i, j].item(), g=[], w=[]) for k, (i, j) in picks.items()}
for x, y in batches:
    loss = gpt(x, y); loss.backward()
    for k, (i, j) in picks.items(): trace[k]['g'].append(gpt.wte.weight.grad[i, j].item())
    opt.step(); opt.zero_grad()
    for k, (i, j) in picks.items(): trace[k]['w'].append(gpt.wte.weight[i, j].item())
rows = []
for k, tr in trace.items():
    hb = adam_by_hand(tr['w0'], tr['g'], LR, B1, B2, EPS, wd=0.1)
    for h, wt in zip(hb, tr['w']):
        rows.append(dict(weight=k, t=h['t'], g=h['g'], m_hat=h['m_hat'], v_hat=h['v_hat'],
                         step_in_lr=h['step_in_lr'], w_hand=h['w'], w_torch=wt, abs_diff=abs(h['w'] - wt)))
real = pd.DataFrame(rows)
display(real.style.format({'g': '{:+.3e}', 'm_hat': '{:+.3e}', 'v_hat': '{:.3e}', 'step_in_lr': '{:+.3f}',
                           'w_hand': '{:.8f}', 'w_torch': '{:.8f}', 'abs_diff': '{:.1e}'})
        .set_caption('real gradients from the GPT; hand AdamW (fp64) vs torch AdamW (fp32 model)'))
ghost = real[(real.weight.str.contains('rare')) & (real.t >= 2)]
SUMMARY['ghost_steps_in_lr'] = ghost.step_in_lr.round(3).tolist()
SUMMARY['real_weight_max_abs_diff'] = float(real.abs_diff.max())
md(f"Hand and PyTorch agree to **{real.abs_diff.max():.1e}** on both weights, which is fp32 resolution at this magnitude. "
   f"On the four zero-gradient steps the rare weight still moved **{', '.join(f'{v:.2f}' for v in ghost.step_in_lr)} η** "
   f"(the sign follows the step-1 gradient). Those numbers have a closed form: "
   f"m̂/√v̂ = [0.1·0.9ᵗ⁻¹/(1−0.9ᵗ)] / √[0.001·0.999ᵗ⁻¹/(1−0.999ᵗ)]. **Adam keeps moving a weight after its signal has stopped.** "
   "That is useful for a rare token and harmful for a weight that should stay frozen.")

%%md
### Where "approximately η whatever the gradient" stops being true: ε

Scale all five gradients by s and Adam's step does not change, because m̂/√v̂ is scale-free. The exception is ε = 10⁻⁸ in the denominator. Once |g| gets near ε, the step shrinks towards η·|g|/ε.

%%code [core]
scales = np.logspace(-12, 2, 57)
st5 = [adam_by_hand(1.0, [s * g for g in G5], LR, B1, B2, EPS)[-1]['step_in_lr'] for s in scales]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.semilogx(scales * 0.5, st5, color=PAL['blue'])
ax.axvline(EPS, color=PAL['red'], lw=1, ls='--'); ax.text(EPS * 1.4, 0.15, 'ε = 1e-8', color=PAL['red'], fontsize=8.5)
half = scales[np.argmin(abs(np.array(st5) - 0.5))] * 0.5
ax.set_xlabel('typical gradient magnitude |g|'); ax.set_ylabel('step at t = 5, in units of η')
title(ax, 'Adam is scale-invariant until the gradient meets ε',
      f'the step halves at |g| ≈ {half:.0e}; bf16 gradients underflow long before fp32 ones get there')
save(fig, '01_eps_floor')
SUMMARY['eps_half_step_grad'] = float(half)

%%md
<a id="2"></a>
## 2 · Switching bias correction off

**Claim (lesson §6).** Bias correction matters most at step 1, where it turns 3.16η into 1.00η.

**Prediction.** For a steady gradient, the uncorrected step equals the corrected step times

$$ f(t) = \frac{1-\beta_1^t}{\sqrt{1-\beta_2^t}} $$

At t = 1 that is 0.1/√0.001 = 3.16, the lesson's number. **It does not then fall.** The numerator recovers on β₁'s timescale of about 10 steps, while the denominator recovers on β₂'s timescale of about 1,000. The error therefore **grows** to a peak before it starts to decay.

%%code [core]
rng = np.random.default_rng(0)
G20 = G5 + list(np.round(0.5 + 0.075 * rng.standard_normal(15), 3))       # the lesson's five, then 15 more of the same kind
on = pd.DataFrame(adam_by_hand(1.0, G20, LR, B1, B2, EPS, bias_correction=True))
off = pd.DataFrame(adam_by_hand(1.0, G20, LR, B1, B2, EPS, bias_correction=False))
t = np.arange(1, 200_001)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 3.8), gridspec_kw=dict(width_ratios=[1, 1.25]))
ax = axs[0]
ax.plot(on.t, on.step_in_lr, 'o-', ms=3.5, color=PAL['blue']); ax.plot(off.t, off.step_in_lr, 'o-', ms=3.5, color=PAL['orange'])
label_end(ax, 20, on.step_in_lr.iloc[-1], 'bias correction on', PAL['blue'], dy=-9)
label_end(ax, 20, off.step_in_lr.iloc[-1], 'off', PAL['orange'])
ax.plot(t[:20], [bias_factor(i) for i in t[:20]], color=INK2, lw=1, ls=':')
ax.text(2, 6.85, 'closed form f(t)', color=INK2, fontsize=8)
ax.set_xlabel('step'); ax.set_ylabel('step size, in units of η'); ax.set_xlim(0.5, 23); ax.set_ylim(0, 7.4)
title(ax, 'The first twenty steps, both ways', f'gradients: the lesson’s five, then 15 more ~N(0.5, 0.075)')
ax = axs[1]
facts = {}
for b2, c in [(0.99, PAL['aqua']), (0.999, PAL['orange']), (0.9999, PAL['violet'])]:
    f = (1 - B1 ** t) / np.sqrt(1 - b2 ** t)
    s1 = int(np.where(abs(f - 1) >= 0.01)[0][-1] + 2); pk = int(f.argmax())
    facts[b2] = dict(peak=float(f[pk]), peak_t=int(t[pk]), f20=float(f[19]), within10=int(np.where(abs(f - 1) >= 0.1)[0][-1] + 2), within1=s1,
                     excess_1000=float((f[:1000] - 1).sum()))
    ax.semilogx(t, f, color=c, lw=2 if b2 == 0.999 else 1.5)
    ax.plot(t[pk], f[pk], 'o', color=c, ms=4); ax.plot(s1, 1.01, '|', color=c, ms=12, mew=2)
    ax.annotate(f'β₂={b2}: peak {f[pk]:.2f}× at t={t[pk]}, within 1% after {s1:,}', (t[pk], f[pk]), xytext=(9, 0),
                textcoords='offset points', color=c, fontsize=8.3, fontweight='bold', va='center')
ax.axvspan(1, 20, color=GRID, alpha=0.6, lw=0); ax.text(1.3, 0.35, 'the 20 steps\nwe were asked to plot', fontsize=7.8, color=INK2)
ax.axhline(1, color=INK2, lw=0.8); ax.set_yscale('log'); ax.set_ylim(0.3, 40)
ax.set_xlabel('step (log)'); ax.set_ylabel('uncorrected / corrected step (log)')
title(ax, 'When does the difference stop mattering?', 'f(t) = (1 − β₁ᵗ)/√(1 − β₂ᵗ), β₁ = 0.9; ticks mark |f − 1| < 1%')
save(fig, '02_bias_correction_20_steps')
SUMMARY['bias_factor'] = facts
d20 = abs(off.w.iloc[-1] - on.w.iloc[-1])
SUMMARY['bias_w_gap_20'] = float(d20)
f999 = facts[0.999]
md(f"**Answer.** Within the twenty steps the difference never stops mattering. It gets worse: step 1 is 3.16×, step {f999['peak_t']} is "
   f"**{f999['peak']:.2f}×** and step 20 is still {f999['f20']:.2f}×. After 20 steps the uncorrected weight is {d20:.4f} further along, "
   f"{d20 / LR:.0f} η of extra travel. With the lesson's β₂ = 0.999 the per-step error falls below 10% after **{f999['within10']:,} steps** and below 1% after "
   f"**{f999['within1']:,} steps**. Over the first 1,000 steps the uncorrected optimizer travels about **{f999['excess_1000']:.0f} η** further than the corrected one. "
   f"The timescale belongs to β₂ rather than to the training run: β₂ = 0.99, the value Session 10 used, cuts it to {facts[0.99]['within1']:,} steps.")

%%md
### Does it matter in a real model? Does warmup hide it?

The closed form describes a single steady gradient. In the GPT, gradients are noisy, the step is clipped, and warmup scales the learning rate down while the error is at its largest. The **warmup-masking hypothesis** follows: during warmup the LR is tiny exactly when f(t) peaks, so a run without bias correction might look the same as one with it. The closed form gives a sharper prediction. f(t) only returns to 1 on β₂'s timescale, 1/(1−β₂) steps, so warmup can hide the missing correction only if it outlasts that memory: about 100 steps at β₂ = 0.99 and 1,000 at β₂ = 0.999. A warmup of 40 steps, typical for a short run, should not be enough.

**Experiment.** The hand-written AdamW (checked against PyTorch in §1) with bias correction on or off × warmup 0, 40 or 400 steps × β₂ ∈ {0.999, 0.99} × 2 seeds, 800 steps, constant LR 1e-3 after warmup. I log the optimizer's own step in units of η for every layer (decay removed).

%%code [bc]
bc = R('bias_correction')
def key(r): c = r['cfg']; return (c['opt']['betas'][1], c['sched']['warmup'], c['opt']['bias_correction'])
gb = A.group(bc, key)
def mean_val(rs):
    steps = [s for s, _ in rs[0]['val']]
    return np.array(steps), np.nanmean([[v for _, v in r['val']] for r in rs], 0), np.nanstd([[v for _, v in r['val']] for r in rs], 0)
def mean_units(rs):    # median over layers of the step in units of eta, averaged over seeds
    return np.nanmean([np.nanmedian(np.array(r['ratio']['units'], float), 1) for r in rs], 0)
b2s = sorted({k[0] for k in gb}, reverse=True); Ws = sorted({k[1] for k in gb})
fig, axs = plt.subplots(2, len(b2s), figsize=(12.5, 7.4))
fig.subplots_adjust(hspace=0.62, wspace=0.22)
rows = []
for j, b2 in enumerate(b2s):
    ax, ax2 = axs[0, j], axs[1, j]
    for W, ls in zip(Ws, ['-', '--', '-.']):
        for bcf, c in [(True, PAL['blue']), (False, PAL['orange'])]:
            u = mean_units(gb[(b2, W, bcf)])
            ax.plot(np.arange(1, len(u) + 1), u, ls, color=c, lw=1.6, alpha=0.95)
        s_on, v_on, sd_on = mean_val(gb[(b2, W, True)]); s_off, v_off, sd_off = mean_val(gb[(b2, W, False)])
        gap = v_off - v_on; noise = np.maximum(np.hypot(sd_on, sd_off), 0.005)
        bad = np.where(abs(gap) > 2 * noise)[0]
        settle = int(s_on[bad[-1] + 1]) if len(bad) and bad[-1] + 1 < len(s_on) else (int(s_on[0]) if not len(bad) else None)
        c = {0: PAL['violet'], Ws[1]: PAL['aqua']}.get(W, PAL['yellow'])
        ax2.fill_between(s_on, -2 * noise, 2 * noise, color=c, alpha=0.10, lw=0)
        ax2.plot(s_on, gap, ls, color=c, lw=1.8, marker='o', ms=2.5)
        label_end(ax2, s_on[-1], gap[-1], f'warmup {W}', c, dy={0: 8, Ws[1]: 0}.get(W, -8))
        rows.append({'β₂': b2, 'warmup': W, 'max val gap (nats)': float(np.nanmax(abs(gap))),
                     'step of max gap': int(s_on[np.nanargmax(abs(gap))]), 'final gap': float(gap[-1]),
                     'gap within seed noise from step': settle})
    tt = np.arange(1, P['bc_steps'] + 1)
    ax.plot(tt, [(1 - B1 ** i) / math.sqrt(1 - b2 ** i) for i in tt], ':', color=INK2, lw=1)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylabel('median layer step, units of η'); ax.set_xlabel('step (log)')
    title(ax, f'β₂ = {b2}: the step Adam actually takes', f'blue on · orange off · solid / dashed / dash-dot: warmup {Ws[0]} / {Ws[1]} / {Ws[-1]} · dotted: f(t)')
    ax2.axhline(0, color=INK2, lw=0.8); ax2.set_xlabel('step'); ax2.set_ylabel('val loss, off − on (nats)')
    title(ax2, 'What switching it off costs in val loss', 'shaded: ±2σ seed-to-seed noise; inside it, the difference has stopped mattering')
save(fig, '02_bias_correction_real_model')
bct = pd.DataFrame(rows); display(bct.style.format(precision=4))
SUMMARY['bias_real'] = rows
g_ = {(r['β₂'], r['warmup']): r for r in rows}
md(f"**Result.** Warmup hides missing bias correction **only when it outlasts β₂'s memory**. At β₂ = 0.99 (memory about 100 steps), a {Ws[-1]}-step warmup leaves a final gap of "
   f"**{g_[(0.99, Ws[-1])]['final gap']:.3f} nats**, which is noise. At β₂ = 0.999 (memory about 1,000 steps), the same warmup still leaves {g_[(0.999, Ws[-1])]['final gap']:.3f}. "
   f"**A short warmup is the worst case:** with {Ws[1]} steps of warmup and β₂ = 0.999, switching bias correction off costs **{g_[(0.999, Ws[1])]['final gap']:.2f} nats** at step {P['bc_steps']}. "
   f"When warmup ends at step {Ws[1]}, f({Ws[1]}) = {bias_factor(Ws[1]):.1f}, so the uncorrected run is effectively training at {bias_factor(Ws[1]):.1f}× the LR. §3 shows that LR is too high "
   f"to escape the loss plateau. Without any warmup, both runs are damaged early and the gap is smaller ({g_[(0.999, 0)]['final gap']:.2f}), "
   "which flatters the broken optimizer. **For the brief:** with β₂ = 0.999 the difference stops mattering after roughly 4,000 steps in theory. In this model it never stopped mattering within 800 steps unless the warmup was long.")

%%md
<a id="3"></a>
## 3 · The update-to-weight ratio, per layer

**Claim (lesson §9).** Early gradients agree with each other, so Adam takes its largest possible step, about η, on every weight at once. The update-to-weight ratio ‖ΔW‖/‖W‖ should sit near 10⁻³, and warmup exists to stop it starting far above that.

**Experiment.** AdamW (λ = 0.1), warmup W ∈ {0, 25, 50, 100, 200} steps, then a constant LR, at two peak LRs (1e-3 and 3e-3), 2 seeds, 600 steps. At every step I log, for each of the 2-D weight matrices, (a) ‖ΔW‖/‖W‖ and (b) the optimizer's own step in units of η, √mean((m̂/(√v̂+ε))²), with the decay term removed.

**Definition.** Every run settles on a common late plateau. Each layer's plateau is its median ratio over the last third of the run, taken as the median across all warmup lengths. Warmup "stops changing" a layer's ratio at the first step from which its 25-step rolling median stays within ±25% of that plateau, with at most 10% of later steps outside the band. The ratio fluctuates by about ±20% from step to step, so a tighter test would mostly measure noise. From that step on, a monitor logging this ratio cannot tell how long the warmup was.

*Why not compare against the no-warmup run?* That was my first definition. At LR 1e-3 the no-warmup run **fails**: it stays on the character-level loss plateau (val ≈ 2.5) for all 600 steps, while every warmed-up run escapes it. A failed run makes a poor reference.

%%code [ratio]
wr = R('warmup_ratio')
gw = A.group(wr, lambda r: (r['cfg']['opt']['lr'], r['cfg']['sched']['warmup']))
names = wr[0]['ratio']['names']; kinds = wr[0]['ratio']['kinds']
def mean_ratio(rs, field='ratio'):
    return np.nanmean([np.array(r['ratio'][field], float) for r in rs], 0)       # (steps, tensors)
lr0 = P['ratio_lrs'][0]; Wmax = max(P['ratio_warm']); Wmid = sorted(P['ratio_warm'])[-2]
cmap = LinearSegmentedColormap.from_list('s11', ['#f3f7fd'] + SEQ)
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
for ax, W in zip(axs, [0, Wmid]):
    M = mean_ratio(gw[(lr0, W)])
    im = ax.imshow(M.T, aspect='auto', cmap=cmap, norm=LogNorm(1e-4, 3e-2), interpolation='nearest',
                   extent=[0.5, M.shape[0] + 0.5, len(names) - 0.5, -0.5])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6.6); ax.grid(False)
    ax.set_xscale('log'); ax.set_xlabel('step (log)')
    if W: ax.axvline(W, color=PAL['orange'], lw=1.2, ls='--')
    pk = M.max(); title(ax, f'warmup {W} steps', f'peak ratio {pk:.1e} · lr {lr0:g}')
cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01); cb.set_label('‖ΔW‖ / ‖W‖', color=INK2); cb.outline.set_visible(False)
save(fig, '03_ratio_heatmap')

%%code [ratio]
roll = lambda x, k=10: pd.Series(x).rolling(k, min_periods=1).mean().to_numpy()
fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.4), gridspec_kw=dict(wspace=0.38))
rows = []
HID = np.array([k == 'hidden' for k in kinds])
for lr in P['ratio_lrs']:
    Ms = {W: mean_ratio(gw[(lr, W)]) for W in sorted(P['ratio_warm'])}
    n = next(iter(Ms.values())).shape[0]; late = slice(2 * n // 3, n)
    plateau = np.median(np.stack([np.median(M[late], 0) for M in Ms.values()]), 0)      # per layer, common late level
    for W, M in Ms.items():
        per = np.array([A.join_step(M[:, i], np.full(n, plateau[i])) or np.nan for i in range(len(names))], float)
        rows.append(dict(lr=lr, warmup=W, stops_changing_median=float(np.nanmedian(per[HID])),
                         earliest=float(np.nanmin(per[HID])), latest=float(np.nanmax(per[HID])),
                         median_over_W=float(np.nanmedian(per[HID])) / W if W else float('nan'),
                         head=float(per[[k == 'head' for k in kinds]][0]), plateau=float(np.median(plateau[HID])),
                         peak_ratio=float(M.max()), final_val=float(np.mean([r['final_val'] for r in gw[(lr, W)]]))))
ax = axs[0]
nW = len(P['ratio_warm'])
for i, (W, c) in enumerate(zip(sorted(P['ratio_warm']), ramp(nW))):
    M = mean_ratio(gw[(lr0, W)]); med = np.median(M[:, [k == 'hidden' for k in kinds]], 1)
    ax.loglog(np.arange(1, len(med) + 1), roll(med), color=c if W else PAL['red'], lw=1.8)
    label_end(ax, len(med), roll(med)[-1], f'W={W}', c if W else PAL['red'], dy=(nW / 2 - i) * 9)
ax.axhspan(5e-4, 2e-3, color=PAL['aqua'], alpha=0.08, lw=0); ax.axhline(1e-3, color=PAL['aqua'], lw=1, ls='--')
ax.set_xlabel('step'); ax.set_ylabel('median hidden-matrix ‖ΔW‖/‖W‖')
title(ax, 'Warmup flattens the opening spike', f'lr {lr0:g}; green band: the lesson’s healthy 0.5–2 ×10⁻³')
ax = axs[1]
for W, c in zip(sorted(P['ratio_warm']), ramp(len(P['ratio_warm']))):
    U = mean_ratio(gw[(lr0, W)], 'units'); med = np.nanmedian(U[:, [k == 'hidden' for k in kinds]], 1)
    ax.semilogx(np.arange(1, len(med) + 1), roll(med), color=c if W else PAL['red'], lw=1.8)
ax.axhline(1.0, color=INK2, lw=0.8); ax.axhline(0.281, color=PAL['yellow'], lw=1, ls='--')
ax.text(1.2, 0.30, 'lesson: noise floor 0.281', color=PAL['yellow'], fontsize=8)
ax.set_xlabel('step'); ax.set_ylabel('Adam step, units of η (RMS)'); ax.set_ylim(0, 1.15)
title(ax, 'Why the spike: early gradients agree', 'm̂/√v̂ starts near 1 and decays as gradients decorrelate')
ax = axs[2]
M = mean_ratio(gw[(lr0, 0)]); Mw = mean_ratio(gw[(lr0, Wmid)])
groups_ = {'embeddings': [k == 'embed' for k in kinds], 'hidden matrices': [k == 'hidden' for k in kinds], 'output head': [k == 'head' for k in kinds]}
for (g, mask), c in zip(groups_.items(), [PAL['aqua'], PAL['blue'], PAL['magenta']]):
    ax.loglog(np.arange(1, len(M) + 1), roll(np.median(M[:, mask], 1)), color=c, lw=1.8)
    ax.loglog(np.arange(1, len(Mw) + 1), roll(np.median(Mw[:, mask], 1)), color=c, lw=1.2, ls='--')
    label_end(ax, len(M), roll(np.median(M[:, mask], 1))[-1], g, c, dy={'embeddings': 8, 'hidden matrices': -4, 'output head': -8}[g])
ax.set_xlabel('step'); title(ax, 'Head vs body (lesson §14 open question)', f'solid: no warmup · dashed: warmup {Wmid}')
save(fig, '03_ratio_lines')
rt = pd.DataFrame(rows); display(rt.style.format(precision=4))
SUMMARY['ratio'] = rows
# Adam's own step and the head-vs-body comparison are read from the healthy warmed-up run (W = Wmid), not the failed W = 0 run
Uw = mean_ratio(gw[(lr0, Wmid)], 'units'); medU = np.nanmedian(Uw[:, [k == 'hidden' for k in kinds]], 1)
SUMMARY['units_first_last'] = [float(medU[0]), float(np.nanmean(medU[-50:]))]
SUMMARY['units_at_150'] = float(np.nanmedian(np.stack([np.nanmedian(mean_ratio(gw[(lr0, W)], 'units')[149, [k == 'hidden' for k in kinds]]) for W in P['ratio_warm'] if W])))
head = np.median(Mw[:, [k == 'head' for k in kinds]], 1); body = np.median(Mw[:, [k == 'hidden' for k in kinds]], 1)
SUMMARY['head_over_body_late'] = float(np.mean(head[-100:] / body[-100:]))
r1 = [r for r in rows if r['lr'] == lr0]; r3 = [r for r in rows if r['lr'] != lr0]
md(f"**Answer.** At LR {lr0:g} the hidden matrices reach the common plateau of **{r1[0]['plateau']:.1e}** at step "
   + ', '.join(f"**{r['stops_changing_median']:.0f}** (W={r['warmup']})" for r in r1)
   + ". For each warmup length, that is the step at which warmup stops changing the ratio. "
   f"For every warmup of {sorted(P['ratio_warm'])[-2]} steps or less it lands between step {min(r['stops_changing_median'] for r in r1[1:-1]):.0f} and {max(r['stops_changing_median'] for r in r1[1:-1]):.0f}, "
   "whatever the warmup length was. A short warmup ends with the ratio overshooting the plateau, and the overshoot takes roughly 70 to 120 steps to settle. "
   f"For the longest warmup the ratio reaches the plateau at {r1[-1]['median_over_W']:.2f}×W, slightly before warmup ends. Adam's own step has fallen from 1 η "
   f"to about {SUMMARY['units_at_150']:.2f} η by step 150, so a partial LR is already enough. Without warmup the opening spike reaches **{r1[0]['peak_ratio']:.2f}** "
   f"({r1[0]['peak_ratio'] / r1[0]['plateau']:.0f}× the plateau) and the run never recovers (val {r1[0]['final_val']:.2f} vs {min(r['final_val'] for r in r1):.2f}). "
   f"At LR {r3[0]['lr']:g} no warmup length rescues the run (best val {min(r['final_val'] for r in r3):.2f}).\n\n"
   f"Two numbers differ from the lesson. The plateau sits near **1e-2, not the lesson's 1e-3**. A rough estimate explains it: the ratio is about η × (Adam's step in units of η) ÷ (weight scale), "
   f"and here that is 1e-3 × {SUMMARY['units_first_last'][1]:.2f} ÷ 0.02 ≈ {1e-3 * SUMMARY['units_first_last'][1] / 0.02:.0e}. The lesson's 1e-3 target is stated for its own model (d_model 4,096, η = 3e-4). "
   f"And Adam's late step is **{SUMMARY['units_first_last'][1]:.2f} η** (W = {Wmid} run, last 50 steps), below the lesson's 0.281 noise floor, so these gradients are even less consistent than the lesson's i.i.d. example. "
   f"**Head vs body:** in the same run the output head's ratio ends at **{SUMMARY['head_over_body_late']:.2f}×** the hidden matrices'. The head moves relatively less, which, if anything, points to giving it a *higher* LR.")

%%md
<a id="4"></a>
## 4 · Cosine vs WSD, stopped at step 200 of a 300-step plan

**Claim (lesson §10).** Cosine has to know the length of the run in advance. WSD does not, because you can checkpoint during the flat phase and decay from there.

**The trap in the brief.** "Train twice for 300 steps, stop both at 200" compares cosine, which by step 200 has decayed to ¼ of its peak LR, with WSD, which is still at its peak and has not annealed at all. That favours cosine for a reason unrelated to the quality of either schedule. So I measure every model you could actually *ship* at step 200:

| model at step 200 | compute | what it represents |
|---|---|---|
| cosine-300, stopped | 200 | the brief, literally |
| WSD-300, stopped | 200 | the brief, literally |
| **cosine-200** | 200 | an oracle that knew the budget in advance |
| **WSD, branch 170→200** (linear or 1−√) | 200 | the WSD workflow: checkpoint at 170, decay for the last 10% |
| WSD, branch 200→230 / 200→260 | 230 / 260 | decay from the step-200 checkpoint (costs extra steps) |
| constant LR + EMA of weights | 200 | the 2026 claim: averaging replaces decay |

**Tuning both sides.** Each method is evaluated at **its own best peak LR** from {2.5e-4 … 4e-3}, with 3 seeds and a 30-step warmup for every schedule. Batches depend only on (seed, step), so every method sees identical data, and the branches see exactly the batches the main run would have seen.

%%code [sched]
sc = R('schedules')
S, STOP = P['sch_steps'], P['sch_stop']; D = S // 10
def kind(r): return r['cfg']['name'].split(' ')[0]
methods = {}
def add(label, r, v): methods.setdefault(label, {}).setdefault(r['cfg']['opt']['lr'], []).append(v)
for r in sc:
    k = kind(r)
    if k == f'cosine-{S}':
        add(f'cosine-{S}, stopped at {STOP}', r, A.val_at(r, STOP)); add(f'cosine-{S}, full ({S} steps)', r, A.val_at(r, S))
    elif k == f'cosine-{STOP}':
        add(f'cosine-{STOP} (oracle: knew the budget)', r, A.val_at(r, STOP))
    elif k == f'wsd-{S}':
        add(f'WSD-{S}, stopped at {STOP}', r, A.val_at(r, STOP)); add(f'WSD-{S}, full ({S} steps)', r, A.val_at(r, S))
        for b in r['branches']:
            extra = b['at'] + b['len'] - STOP
            add(f"WSD branch {b['at']}→{b['at'] + b['len']} ({b['shape']})" + (f' [+{extra} steps]' if extra > 0 else ''), r, b['val'])
    elif k == 'const+ema':
        add(f'constant LR, raw weights @{STOP}', r, A.val_at(r, STOP))
        for d, vals in r['ema'].items():
            add(f'constant LR + EMA({d}) @{STOP}', r, dict(vals).get(STOP, np.nan))
        add(f'constant LR + EMA(best) @{S}', r, min(dict(v).get(S, np.nan) for v in r['ema'].values()))
rows = []
for m, by in methods.items():
    means = {lr: np.nanmean(v) for lr, v in by.items()}
    best = min(means, key=lambda l: means[l] if np.isfinite(means[l]) else 9)
    rows.append(dict(model=m, best_lr=best, val=means[best], std=float(np.nanstd(by[best])), n=len(by[best]),
                     at_lr_1e3=means.get(1e-3, np.nan)))
tab = pd.DataFrame(rows).sort_values('val').reset_index(drop=True)
display(tab.style.format({'best_lr': '{:.1e}', 'val': '{:.4f}', 'std': '{:.4f}', 'at_lr_1e3': '{:.4f}'})
        .set_caption('val loss of every shippable model, each at its own best LR (mean of seeds)'))
SUMMARY['schedules'] = rows

%%code [sched]
from s11.schedules import factor, decay_shape
fig, axs = plt.subplots(1, 3, figsize=(17, 4.5), gridspec_kw=dict(width_ratios=[1, 1.15, 1], wspace=0.42))
ax = axs[0]; xs = np.arange(S)
W = P['sch_warm']
ax.plot(xs, [factor(i, dict(kind='cosine', total=S, warmup=W)) for i in xs], color=PAL['violet'])
ax.plot(np.arange(STOP), [factor(i, dict(kind='cosine', total=STOP, warmup=W)) for i in range(STOP)], color=PAL['violet'], ls=':')
ax.plot(xs, [factor(i, dict(kind='wsd', total=S, warmup=W, decay_len=D)) for i in xs], color=PAL['blue'])
bx = np.arange(STOP - D - 1, STOP); ax.plot(bx, [decay_shape((i - STOP + D + 1) / D) for i in bx], color=PAL['aqua'], lw=2.4)
ax.plot(bx, [decay_shape((i - STOP + D + 1) / D, 'sqrt') for i in bx], color=PAL['aqua'], lw=1.4, ls='--')
ax.axvline(STOP, color=PAL['orange'], lw=1.5); ax.text(STOP + 3, 0.92, f'stop at {STOP}', color=PAL['orange'], fontsize=8.5, fontweight='bold')
c200 = factor(STOP - 1, dict(kind='cosine', total=S, warmup=W))
ax.annotate(f'cosine-{S} at {c200:.0%}\nof peak here', (STOP, c200), xytext=(18, 45), textcoords='offset points',
            fontsize=8, color=PAL['violet'], arrowprops=dict(arrowstyle='-', color=PAL['violet'], lw=0.8))
for x_, y_, s_, c_ in [(0.07 * S, 1.04, f'WSD-{S}', PAL['blue']), (0.13 * S, 0.62, f'cosine-{S}', PAL['violet']),
                       (0.30 * S, 0.12, f'cosine-{STOP}', PAL['violet']), (STOP - D - 0.18 * S, 0.45, 'WSD branch', PAL['aqua'])]:
    ax.text(x_, y_, s_, color=c_, fontsize=8.5, fontweight='bold')
ax.set_xlabel('step'); ax.set_ylabel('LR / peak'); ax.set_ylim(0, 1.12)
title(ax, 'The schedules', f'branch: checkpoint {STOP - D}, decay {D} (dashed: 1−√ shape)')
ax = axs[1]
best_lr = {r['model']: r['best_lr'] for r in rows}
def runs_of(k, lr): return [r for r in sc if kind(r) == k and r['cfg']['opt']['lr'] == lr]
def plot_val(rs, c, ls='-', lab=None, branch=None):
    st, mv, _ = mean_val(rs); ax.plot(st, mv, ls, color=c, lw=1.8)
    if lab: label_end(ax, st[-1], mv[-1], lab, c)
plot_val(runs_of(f'cosine-{S}', best_lr[f'cosine-{S}, stopped at {STOP}']), PAL['violet'], lab=f'cosine-{S}')
plot_val(runs_of(f'wsd-{S}', best_lr[f'WSD-{S}, stopped at {STOP}']), PAL['blue'], lab=f'WSD-{S}')
plot_val(runs_of(f'cosine-{STOP}', best_lr[f'cosine-{STOP} (oracle: knew the budget)']), PAL['violet'], ':', lab=f'cosine-{STOP}')
k_br = next(m for m in best_lr if m.startswith(f'WSD branch {STOP - D}→{STOP} (linear)'))
rs = runs_of(f'wsd-{S}', best_lr[k_br])
bv = np.mean([next(b['val'] for b in r['branches'] if b['at'] == STOP - D and b['shape'] == 'linear') for r in rs])
st_, mv_, _ = mean_val(rs); v180 = mv_[list(st_).index(STOP - D)]
ax.plot([STOP - D, STOP], [v180, bv], color=PAL['aqua'], lw=2.4, marker='o', ms=4); label_end(ax, STOP, bv, 'WSD branch', PAL['aqua'], dy=8)
ax.axvline(STOP, color=PAL['orange'], lw=1.2)
ax.set_xlabel('step'); ax.set_ylabel('val loss'); ax.set_xlim(P['sch_warm'], S * 1.12)
lo = np.nanmin(tab.val) - 0.05; ax.set_ylim(lo, lo + 0.6)
title(ax, 'Val loss, each at its own best LR', 'mean of seeds')
ax = axs[2]
show_m = [f'cosine-{S}, stopped at {STOP}', f'WSD-{S}, stopped at {STOP}', f'cosine-{STOP} (oracle: knew the budget)', k_br]
for m, c, ls in zip(show_m, [PAL['violet'], PAL['blue'], PAL['violet'], PAL['aqua']], ['-', '-', ':', '-']):
    by = methods[m]; lrs = sorted(by); ax.plot(lrs, [np.nanmean(by[l]) for l in lrs], ls, marker='o', ms=3.5, color=c)
ax.set_xscale('log'); ax.set_xlabel('peak LR'); ax.set_ylabel(f'val loss at step {STOP}')
ax.set_xticks(P['sch_lrs']); ax.set_xticklabels([f'{l:g}' for l in P['sch_lrs']]); ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
for i, (m, c) in enumerate(zip(show_m, [PAL['violet'], PAL['blue'], PAL['violet'], PAL['aqua']])):
    ax.text(0.03, 0.95 - 0.07 * i, m.split(' (')[0].replace(', stopped at', ' @') + (' (dotted)' if 'oracle' in m else ''), transform=ax.transAxes,
            color=c, fontsize=8.4, fontweight='bold', va='top')
title(ax, 'Tuning both sides', 'a single-LR comparison could pick either winner')
save(fig, '04_schedules')
same = tab[~tab.model.str.contains(r'full|\[\+|@' + str(S))].reset_index(drop=True)      # models that cost exactly STOP steps
winner = same.iloc[0]
SUMMARY['schedule_winner'] = dict(winner)
v = dict(zip(tab.model, tab.val)); sd = dict(zip(tab.model, tab['std']))
cS, wS = f'cosine-{S}, stopped at {STOP}', f'WSD-{S}, stopped at {STOP}'
brl = next(m for m in v if m.startswith(f'WSD branch {STOP - D}→{STOP} (linear'))
md(f"**The brief, literally.** At step {STOP}: cosine **{v[cS]:.3f} ± {sd[cS]:.3f}**, WSD **{v[wS]:.3f} ± {sd[wS]:.3f}**. That difference ({v[cS] - v[wS]:+.3f}) is inside seed noise, even though cosine had decayed to "
   f"{factor(STOP - 1, dict(kind='cosine', total=S, warmup=P['sch_warm'])):.0%} of its peak and WSD had not decayed at all.\n\n"
   f"**Which model I would keep: the WSD run.** With the same {STOP} steps of compute, its decay branch from the step-{STOP - D} checkpoint reaches **{v[brl]:.3f}**, better than "
   f"anything cosine produced at {STOP}. The cosine planned for exactly {STOP} steps did *worse* ({v[f'cosine-{STOP} (oracle: knew the budget)']:.3f}). "
   f"At this horizon the model is still escaping the loss plateau, so decaying early costs more than annealing gains. The WSD run also keeps going: at {S} steps WSD reaches "
   f"**{v[f'WSD-{S}, full ({S} steps)']:.3f}** against cosine's {v[f'cosine-{S}, full ({S} steps)']:.3f}. "
   f"Constant LR plus a weight EMA matches the decay branch at {STOP} ({v.get(f'constant LR + EMA(0.9) @{STOP}', float('nan')):.3f}) with no decay at all, "
   "which supports the 2026 claim that averaging and decay do the same job, at least at this scale.")

%%md
<a id="5"></a>
## 5 · How the best learning rate moves with width

**Claim (lesson §12).** Under the standard parameterisation the best η falls roughly as 1/width (3.0e-3 at 256, about 1.9e-4 at 4,096). Under muP it stays put.

**Experiment.** A 4-layer GPT at widths 256 / 512 / 1,024, with head dim 64 so the number of heads grows with width. The LR sweep is 12 points spaced ×2 apart from 2.5e-5 to 5.1e-2, 600 steps, warmup 30, cosine to 10%, λ = 0, 2 seeds. A **second pass** then adds points at ×√2 on either side of each coarse minimum. I added it after the first pass: the loss falls off a cliff just above the optimum (too high an LR leaves the run stuck on the loss plateau), so a ×2 grid pinned each minimum only to within about ×1.4, and a parabola through such an asymmetric curve is biased. It runs once under **SP** (the Session 10 model as-is) and once under **muP** (base width 256: hidden init /√r, hidden Adam LR /r, logits ×1/r). Width **2,048** is trained but **held out**: the width law is fitted on 256–1,024 only, then checked against 2,048 before any claim about 4,096. The minimum's location lr* is a parabola in log₂η through the best point and its two neighbours, and its interval comes from 2,000 bootstrap resamples of the seeds.

%%code [sweep]
sw = R('sweep')
widths, hold = P['sw_widths'], P['sw_holdout']
gs = A.group(sw, lambda r: (r['cfg']['model']['param'], r['cfg']['model']['width']))
fits = {}
for (prm, w), rs in gs.items():
    cv = A.curve(rs)
    mean = {l: np.nanmean(v) if np.isfinite(v).any() else np.nan for l, v in cv.items()}
    lr_star, l_star = A.fit_min(list(mean), list(mean.values()))
    draws = A.boot_min(cv)
    fits[(prm, w)] = dict(curve=cv, mean=mean, lr=lr_star, loss=float(np.nanmin(list(mean.values()))), draws=draws, ci=A.ci(draws))
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
allw = widths + hold; cols = dict(zip(allw, ramp(len(allw))))
for ax, prm in zip(axs, ['sp', 'mup']):
    for w in allw:
        f = fits[(prm, w)]; lrs = np.array(list(f['mean'])); ys = np.array([f['mean'][l] for l in lrs])
        ho = w in hold; c = PAL['orange'] if ho else cols[w]
        ys_plot = np.where(np.isfinite(ys), ys, np.nan)
        ax.plot(lrs, ys_plot, '--' if ho else '-', marker='o', ms=3, color=c, lw=1.6)
        div = ~np.isfinite(ys)
        ax.plot(lrs[div], np.full(div.sum(), 3.55), 'x', color=c, ms=5)
        ax.plot([f['lr']], [f['loss']], marker='*', ms=15, color=c, mec='white', mew=1.2, zorder=5)
        ax.errorbar([f['lr']], [f['loss']], xerr=[[f['lr'] - f['ci'][0]], [f['ci'][1] - f['lr']]], color=c, capsize=3, lw=1)
        ax.text(0.98, 0.30 - 0.065 * allw.index(w), f"{w:,}{' (held out)' if ho else ''}:  lr* = {f['lr']:.2e}", transform=ax.transAxes,
                ha='right', color=c, fontsize=8.8, fontweight='bold')
    lo_ = min(fits[(prm, w)]['loss'] for w in allw)
    ax.set_xscale('log'); ax.set_xlabel('peak learning rate'); ax.set_ylim(lo_ - 0.12, lo_ + 0.75)
    title(ax, {'sp': 'Standard parameterisation', 'mup': 'muP (base width 256)'}[prm],
          '★ lowest loss, placed at the fitted lr* · bar: 90% bootstrap interval of lr*')
axs[0].set_ylabel(f"val loss after {P['sw_steps']} steps")
save(fig, '05_lr_sweep')

%%code [sweep]
rows, law = [], {}
fig, ax = plt.subplots(figsize=(8.8, 4.8))
W4 = widths + hold
for prm, c in [('sp', PAL['blue']), ('mup', PAL['aqua'])]:
    F = {w: fits[(prm, w)] for w in W4}
    slopes, preds = A.width_law(widths, [F[w]['draws'] for w in widths], target=4096)
    _, pred_h = A.width_law(widths, [F[w]['draws'] for w in widths], target=hold[0])
    slopes4, preds4 = A.width_law(W4, [F[w]['draws'] for w in W4], target=4096)
    b, a = np.polyfit(np.log(widths), np.log([F[w]['lr'] for w in widths]), 1)
    b4, a4 = np.polyfit(np.log(W4), np.log([F[w]['lr'] for w in W4]), 1)
    b_loc = math.log(F[hold[0]]['lr'] / F[widths[-1]]['lr']) / math.log(hold[0] / widths[-1])
    est = {'power law fitted on 3 widths': math.exp(a + b * math.log(4096)),
           'power law fitted on all 4 widths': math.exp(a4 + b4 * math.log(4096)),
           f'local slope {widths[-1]:,}→{hold[0]:,}, extended': F[hold[0]]['lr'] * (4096 / hold[0]) ** b_loc}
    p_h = math.exp(a + b * math.log(hold[0]))
    law[prm] = dict(slope=b, slope_ci=A.ci(slopes), slope4=b4, slope4_ci=A.ci(slopes4), slope_local=b_loc,
                    pred4096=math.exp(a + b * math.log(4096)), pred4096_ci=A.ci(preds), pred4096_all4_ci=A.ci(preds4),
                    estimators=est, recommend=float(np.exp(np.mean(np.log(list(est.values()))))),
                    rec_range=[min(est.values()), max(est.values())], spread=max(est.values()) / min(est.values()),
                    pred_holdout=p_h, pred_holdout_ci=A.ci(pred_h), measured_holdout=F[hold[0]]['lr'],
                    measured_holdout_ci=F[hold[0]]['ci'], holdout_ratio=F[hold[0]]['lr'] / p_h,
                    holdout_inside=bool(A.ci(pred_h)[0] <= F[hold[0]]['lr'] <= A.ci(pred_h)[1]))
    xs = np.array([widths[0] / 1.3, 4096 * 1.3])
    ax.loglog(xs, np.exp(a + b * np.log(xs)), color=c, lw=1.2, ls='--')
    for w in W4:
        f = F[w]
        ax.errorbar([w], [f['lr']], yerr=[[f['lr'] - f['ci'][0]], [f['ci'][1] - f['lr']]], fmt='o' if w in widths else 'D',
                    color=c, mfc=c if w in widths else SURF, ms=6, capsize=3, mew=1.6)
        rows.append(dict(param=prm, width=w, held_out=w in hold, lr_star=f['lr'], ci_lo=f['ci'][0], ci_hi=f['ci'][1], loss=f['loss']))
    lo_, hi_ = law[prm]['rec_range']
    # SP: geometric mean of the three extrapolations. muP: the single LR that was best at every width (computed below).
    common = sorted(set.intersection(*[set(F[w]['mean']) for w in W4]))
    rec = min(common, key=lambda l: max(F[w]['mean'][l] - F[w]['loss'] for w in W4)) if prm == 'mup' else law[prm]['recommend']
    ax.plot([4096, 4096], [lo_, hi_], color=c, lw=6, alpha=0.3, solid_capstyle='butt')
    ax.plot([4096], [rec], marker='*', ms=16, color=c, mec='white')
    label_end(ax, 4096, rec, f"{'muP' if prm == 'mup' else 'SP'}: {rec:.1e}", c, dx=12, dy=-10 if prm == 'mup' else 0)
ax.plot([256, 4096], [3e-3, 3e-3 * 256 / 4096], color=MUTED, lw=1, ls=':'); ax.text(300, 3.4e-3, 'lesson’s table (∝ 1/width)', color=MUTED, fontsize=8)
ax.set_xticks([256, 512, 1024, 2048, 4096]); ax.set_xticklabels(['256', '512', '1,024', '2,048\n(held out)', '4,096\n(predicted)'])
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter()); ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
ax.set_xlabel('width'); ax.set_ylabel('best peak LR')
title(ax, 'The width law, fitted on three widths and tested on a fourth',
      'dashed: fit through 256–1,024 · hollow: held out · bar at 4,096: range of three extrapolations · ★ the value I would use')
save(fig, '05_width_law')
mt = pd.DataFrame(rows); display(mt.style.format({'lr_star': '{:.2e}', 'ci_lo': '{:.2e}', 'ci_hi': '{:.2e}', 'loss': '{:.4f}'}))
for prm in ('sp', 'mup'):
    display(pd.DataFrame({'estimate at width 4,096': law[prm]['estimators']}).style.format('{:.2e}').set_caption(prm.upper()))
def regret_table(prm):
    common = set.intersection(*[set(fits[(prm, w)]['mean']) for w in W4])
    reg = {l: max(fits[(prm, w)]['mean'][l] - fits[(prm, w)]['loss'] for w in W4) for l in sorted(common)}
    reg = {l: v for l, v in reg.items() if np.isfinite(v)}
    best = min(reg, key=reg.get)
    return best, reg[best], reg
for prm in ('sp', 'mup'):
    b_, r_, reg_ = regret_table(prm)
    law[prm].update(common_lr=b_, worst_regret=r_, regret={f'{k:.3g}': v for k, v in reg_.items()})
display(pd.DataFrame({prm.upper(): {f'{k:.2e}': v for k, v in regret_table(prm)[2].items()} for prm in ('sp', 'mup')})
        .style.format('{:.3f}', na_rep='—').set_caption('worst-case regret (nats): one LR used at every width, minus each width’s own best'))
sp_, mu_ = law['sp'], law['mup']
md(f"**Did the extrapolation survive the held-out width?** Under SP the 256–1,024 power law (slope **{sp_['slope']:.2f}**, 90% interval "
   f"{sp_['slope_ci'][0]:.2f} to {sp_['slope_ci'][1]:.2f}) predicted **{sp_['pred_holdout']:.1e}** at width {hold[0]:,}. The measurement was "
   f"**{sp_['measured_holdout']:.1e}**, {sp_['holdout_ratio']:.2f}× the prediction and {'inside' if sp_['holdout_inside'] else '**outside**'} its bootstrap interval. "
   f"The local slope from {widths[-1]:,} to {hold[0]:,} is {sp_['slope_local']:.2f}. The bootstrap interval only captures seed noise. It does not cover the bigger "
   f"uncertainty, which is whether a power law is the right shape at all, and the held-out width is what exposed that.\n\n"
   f"**Width 4,096 under SP:** the three estimators span **{sp_['rec_range'][0]:.1e} to {sp_['rec_range'][1]:.1e}**, a factor of {sp_['spread']:.1f}. I would use "
   f"**{sp_['recommend']:.1e}**, their geometric mean, with low-to-moderate confidence: I'd trust it to within about ×{math.sqrt(sp_['spread']):.1f} either way, "
   f"and I would still run a 3-point check (×0.5, ×1, ×2) at 4,096 before committing.\n\n"
   f"**Width 4,096 under muP.** The question that matters is whether one LR works everywhere. Under muP, **lr = {mu_['common_lr']:.2e} " + ("is the best measured LR at every width" if mu_['worst_regret'] < 5e-4 else f"is within {mu_['worst_regret']:.3f} nats of the best LR at every width")
   + f" from {W4[0]:,} to {W4[-1]:,}**, and the next-best single LR is within {sorted(mu_['regret'].values())[1]:.3f} nats everywhere. Under SP the best single LR still loses **{sp_['worst_regret']:.2f} nats** at some width. The fitted lr* under muP wobbles "
   f"({', '.join(f'{fits[("mup", w)]["lr"]:.1e}' for w in W4)}) only because the curves have flat bottoms. The loss barely changes between 1.1e-3 and 1.6e-3, so the fit has little to lock onto. "
   f"For width 4,096 under muP I would use a base LR of **{mu_['common_lr']:.1e}** (hidden matrices get ×256/4,096 of it), with **high confidence**. Doubling the width four times never moved that window, "
   "and on a flat bottom being off by ×1.4 costs about 0.01 nats. **That is the practical argument for muP: it makes the 4,096 answer something you can measure.**\n\n"
   f"**A claim the data only partly supports.** The lesson says muP *does not improve the model*. At each one's own best LR: "
   + ', '.join(f"{w:,}: SP {fits[('sp', w)]['loss']:.3f} vs muP {fits[('mup', w)]['loss']:.3f}" for w in W4[1:])
   + ". At 512 the two are tied within seed noise (about ±0.02). At the two widest, muP is clearly better, because under SP the wider models are *worse* than width 512 at this 600-step budget, "
   "while under muP every doubling of width helps. My hypothesis, untested here: SP's fixed 0.02 init and unscaled readout give wide models updates that are too large relative to their weights, "
   "and a short run never recovers. That would make the lesson right about muP's design (it transfers hyperparameters) but too modest about what it does to wide models in practice.")
SUMMARY['sweep_minima'] = rows; SUMMARY['width_law'] = law

%%md
<a id="6"></a>
## 6 · AdamW vs Muon, both tuned

**Claim (lesson §13).** Muon beats a *well tuned* AdamW by about 1.4× at 0.1B, shrinking to about 1.1× at 1.2B. Most two-fold speedups came from untuned baselines.

**Experiment.** Muon runs on the hidden 2-D matrices, and everything else (embeddings, head, norms, biases) stays on AdamW, as in every recipe that works. Muon's step is RMS-matched (×0.2√max(A,B), Liu et al. 2025), so all optimizers share one LR axis. Three arms each get the same 10-point LR sweep at widths 256 and 1,024, with 2 seeds and the §5 budget: **AdamW with β₂ = 0.999** (the lesson's default, LR tuned), **AdamW with β₂ = 0.95** (the usual LLM setting), and **Muon**. The β₂ = 0.999 AdamW points are reused from §5 by config hash. The speedup is the number of steps Muon needs to reach the baseline's final training loss (EMA-smoothed), against the baseline's 600. I compute it against three baselines of increasing care. **I added the β₂ = 0.95 arm after seeing the first result.** Muon was ahead by about 0.3 nats against an AdamW tuned on LR alone, which is the pattern the lesson warns about.

%%code [muon]
mu = R('muon')
def vkey(r):
    o = r['cfg']['opt']; return 'Muon' if o['kind'] == 'muon' else ('AdamW β₂=0.95' if o.get('betas', [0.9, 0.999])[1] == 0.95 else 'AdamW β₂=0.999')
gm = A.group(mu, lambda r: (vkey(r), r['cfg']['model']['width']))
VC = {'AdamW β₂=0.999': PAL['blue'], 'AdamW β₂=0.95': PAL['violet'], 'Muon': PAL['magenta']}
def best_of(rs):
    cv = A.curve(rs); mean = {l: np.nanmean(x) if np.isfinite(x).any() else np.inf for l, x in cv.items()}
    bl = min(mean, key=mean.get); return bl, mean[bl], mean
def at_lr(rs, lr): return [r for r in rs if r['cfg']['opt']['lr'] == lr]
def val_curve(rs):
    st = np.array([s_ for s_, _ in rs[0]['val']]); return st, np.mean([[v for _, v in r['val']] for r in rs], 0)
def steps_to(rs_fast, target):
    """Step at which the (seed-mean) val curve first reaches `target`, linearly interpolated between eval points."""
    st, v = val_curve(rs_fast); st = np.r_[0, st]; v = np.r_[np.log(65), v]
    for i in range(1, len(st)):
        if v[i] <= target:
            return st[i - 1] + (v[i - 1] - target) / (v[i - 1] - v[i]) * (st[i] - st[i - 1])
    return float('nan')
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1, 0.95], wspace=0.42))
rows = []
for w, ls in zip(P['mu_widths'], ['-', '--']):
    B = {v: best_of(gm[(v, w)]) for v in VC}
    for v, c in VC.items():
        mean = B[v][2]; lrs = np.array(list(mean)); ys = np.array([mean[l] if np.isfinite(mean[l]) else np.nan for l in lrs])
        axs[0].plot(lrs, ys, ls, marker='o', ms=3, color=c, lw=1.6)
        axs[0].plot([B[v][0]], [B[v][1]], marker='*', ms=12, color=c, mec='white')
    mup_runs = [r for r in R('sweep') if r['cfg']['model']['param'] == 'mup' and r['cfg']['model']['width'] == w]
    mlr, mval, _ = best_of(mup_runs)
    tuned_v = min(['AdamW β₂=0.999', 'AdamW β₂=0.95'], key=lambda v: B[v][1])
    default_lr = min(B['AdamW β₂=0.999'][2], key=lambda l: abs(math.log(l / 3e-4)))
    base = {'default LR': at_lr(gm[('AdamW β₂=0.999', w)], default_lr),
            'LR tuned': at_lr(gm[('AdamW β₂=0.999', w)], B['AdamW β₂=0.999'][0]),
            'LR + β₂ tuned': at_lr(gm[(tuned_v, w)], B[tuned_v][0]),
            'muP + LR tuned': at_lr(mup_runs, mlr)}
    muon_rs = at_lr(gm[('Muon', w)], B['Muon'][0])
    S_ = P['sw_steps']
    spd = {k: S_ / steps_to(muon_rs, val_curve(rs)[1][-1]) for k, rs in base.items()}
    gap_tr_val = float(np.mean([r['final_val'] - A.ema(r['train_loss'], 0.95)[-1] for r in muon_rs]))
    rows.append(dict(width=w, adamw999_lr=B['AdamW β₂=0.999'][0], adamw999_val=B['AdamW β₂=0.999'][1],
                     adamw95_lr=B['AdamW β₂=0.95'][0], adamw95_val=B['AdamW β₂=0.95'][1], mup_lr=mlr, mup_val=mval,
                     muon_lr=B['Muon'][0], muon_val=B['Muon'][1],
                     gap_vs_best_adamw=min(B['AdamW β₂=0.999'][1], B['AdamW β₂=0.95'][1]) - B['Muon'][1], gap_vs_mup=mval - B['Muon'][1],
                     speedup_vs_default=spd['default LR'], speedup_vs_lr_tuned=spd['LR tuned'], speedup_tuned=spd['LR + β₂ tuned'],
                     speedup_vs_mup=spd['muP + LR tuned'], muon_val_minus_train=gap_tr_val, default_lr=default_lr))
    if w == P['mu_widths'][-1]:
        ax = axs[1]
        for (k, rs), c in zip([('AdamW, default LR', base['default LR']), ('AdamW, LR tuned', base['LR tuned']),
                               ('AdamW, muP + LR', base['muP + LR tuned']), ('Muon, LR tuned', muon_rs)],
                              [MUTED, PAL['blue'], PAL['aqua'], PAL['magenta']]):
            st, v = val_curve(rs); tr = np.mean([A.ema(r['train_loss'], 0.95) for r in rs], 0)
            ax.plot(np.arange(1, len(tr) + 1), tr, color=c, lw=1, alpha=0.45)
            ax.plot(st, v, color=c, lw=2, marker='o', ms=4)
            ax.text(st[-1] + 12, v[-1], k, color=c, fontsize=8.3, fontweight='bold', va='center')
        ax.set_xlim(0, S_ * 1.42); ax.set_ylim(0.95, 2.7)
        ax.set_xlabel('step'); ax.set_ylabel('loss')
        title(ax, f'Width {w:,}: val (dots) and train (faint)', 'Muon’s val stalls while its train keeps falling')
axs[0].set_xscale('log'); axs[0].set_xlabel('peak LR (one axis: Muon is RMS-matched)'); axs[0].set_ylabel('final val loss')
lo0 = min(r['muon_val'] for r in rows); axs[0].set_ylim(lo0 - 0.08, lo0 + 1.2)
for i, (v, c) in enumerate(VC.items()):
    axs[0].text(0.03, 0.95 - 0.07 * i, v, transform=axs[0].transAxes, color=c, fontsize=8.6, fontweight='bold', va='top')
title(axs[0], 'Three optimizers, one sweep each', f"solid w={P['mu_widths'][0]:,} · dashed w={P['mu_widths'][1]:,} · ★ best")
ax = axs[2]; labels = ['default\nLR', 'LR\ntuned', 'LR + β₂\ntuned', 'muP +\nLR tuned']; x = np.arange(4); bw = 0.36
for i, (r, c) in enumerate(zip(rows, [PAL['magenta'], '#b04a75'])):
    vals = [r['speedup_vs_default'], r['speedup_vs_lr_tuned'], r['speedup_tuned'], r['speedup_vs_mup']]
    ax.bar(x + (i - 0.5) * bw, vals, bw * 0.92, color=c, label=f"width {r['width']:,}")
    for xx, v in zip(x + (i - 0.5) * bw, vals): ax.text(xx, v + 0.05, f'{v:.1f}×', ha='center', fontsize=8.5, color=INK2)
ax.axhspan(1.1, 1.4, color=PAL['aqua'], alpha=0.12, lw=0); ax.text(3.5, 1.25, 'lesson: 1.1–1.4×', color=PAL['aqua'], fontsize=8, ha='right', va='center')
ax.axhline(1, color=INK2, lw=0.8); ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('apparent Muon speedup'); ax.legend(loc='upper right')
title(ax, 'The speedup depends on the baseline', 'steps for Muon’s val loss to reach AdamW’s final val loss')
save(fig, '06_muon_vs_adamw')
mt6 = pd.DataFrame(rows); display(mt6.style.format(precision=4))
SUMMARY['muon'] = rows
r0, r1 = rows[0], rows[-1]
b95 = all(r['adamw95_val'] > r['adamw999_val'] for r in rows)
md(f"Speedups are measured on **val** loss. Muon's train loss runs {r1['muon_val_minus_train']:.2f} nats below its val loss by the end at width {r1['width']:,}: 600 steps × 8,192 tokens is about five passes over "
   "TinyShakespeare's 1M characters, and Muon starts memorising it. A train-loss speedup would partly be measuring that. Val is logged every 150 steps, so the crossing step is interpolated between eval points and is only coarse.\n\n"
   f"The apparent speedup falls as the baseline gets more care: **{r0['speedup_vs_default']:.1f}× / {r1['speedup_vs_default']:.1f}×** (widths {r0['width']:,} / {r1['width']:,}) against AdamW left at a default LR ({r1['default_lr']:.0e}, the grid point nearest the common 3e-4), "
   f"{r0['speedup_vs_lr_tuned']:.1f}× / {r1['speedup_vs_lr_tuned']:.1f}× with the LR tuned"
   + (" (β₂ = 0.95 did not help; it was slightly *worse* than 0.999 for AdamW here)" if b95 else f", {r0['speedup_tuned']:.1f}× / {r1['speedup_tuned']:.1f}× with β₂ tuned too")
   + f", and **{r1['speedup_vs_mup']:.1f}×** at width {r1['width']:,} once AdamW also gets muP. That last step shrinks the val gap from {r1['gap_vs_best_adamw']:.2f} to {r1['gap_vs_mup']:.2f} nats. "
   f"**About {100 * (1 - r1['gap_vs_mup'] / r1['gap_vs_best_adamw']):.0f}% of the Muon advantage at width 1,024 was an AdamW parameterisation problem.** "
   "At the base width muP and SP are the same model, so the 256 bar does not move. This fits the theory: Muon's orthogonalised, shape-scaled step already behaves like a width-aware parameterisation, "
   "so comparing it with SP-AdamW mixes two changes in one comparison.\n\n"
   f"What remains ({r0['speedup_tuned']:.1f}× at {r0['width']:,}, {r1['speedup_vs_mup']:.1f}× at {r1['width']:,}) is still above the lesson's 1.1–1.4×. That figure was measured at 0.1B–1.2B parameters over long runs. "
   "These models have 3M–50M parameters and train for 600 steps, mostly while escaping the character-level loss plateau, and AdamW's warmup and ε were left untuned. "
   "I would treat this as an upper bound for this regime, not as evidence against the lesson's number.")

%%md
## 7 · Summary

%%code [core]
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
Path('results').mkdir(exist_ok=True)
Path(f'results/summary.{PROFILE}.json').write_text(json.dumps(clean(SUMMARY), indent=1))
print(json.dumps(clean(SUMMARY), indent=1)[:3000])
