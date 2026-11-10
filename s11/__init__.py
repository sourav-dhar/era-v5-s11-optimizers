"""ERA V5 · Session 11 — Optimizers and learning-rate schedules, re-measured.

Modules
    data        TinyShakespeare, one token per character, deterministic batches
    model       the Session 10 GPT, with a standard (SP) and a muP parameterisation
    optim       Adam by hand (scalar), a hand-written AdamW with switchable bias correction, Muon
    schedules   constant / cosine / WSD (+ warmup, + decay branches)
    train       one training run -> a JSON-able dict of everything it measured
    experiments every study as a list of run configs, with result caching
    analysis    minima fits, bootstrap intervals, convergence steps
    style       one visual language for every figure (shared with Session 10)
"""
