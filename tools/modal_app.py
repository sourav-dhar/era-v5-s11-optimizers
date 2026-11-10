"""Run a study's pending configs on Modal A100s, in parallel, storing each result as it lands.

    modal run tools/modal_app.py --study sweep                 # one study
    modal run tools/modal_app.py --study all                   # every study
    modal run tools/modal_app.py --study timing                # 3 representative runs, to price the rest
    modal run tools/modal_app.py --study sweep --limit 10      # only the first 10 pending configs

Results land in results/<study>.full.json (resumable: finished configs are never re-run).
"""
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.8.0', 'numpy')
         .add_local_dir(ROOT / 'data', '/root/data')
         .add_local_python_source('s11'))

app = modal.App('era-v5-s11-optimizers', image=image)


@app.function(gpu='A100-40GB', timeout=3600, max_containers=8)
def run_remote(cfg: dict) -> dict:
    from s11.train import run
    return run(cfg)


@app.local_entrypoint()
def main(study: str = 'timing', profile: str = 'full', limit: int = 0):
    from s11 import experiments as E
    if study == 'timing':
        P = E.PROFILES[profile]
        cfgs = [E.configs('schedules', profile)[0], E._sw(P, 1024, 'sp', 1e-3, 0), E._sw(P, 2048, 'sp', 1e-3, 0)]
        for c, r in zip(cfgs, run_remote.map(cfgs)):
            print(f'{c["name"]:<36} {r["time_s"]:7.1f}s total  {r["tok_per_s"]:>10,} tok/s  val {r["final_val"]:.4f}  {r["gpu"]}')
        return
    studies = list(E.STUDIES) if study == 'all' else study.split(',')
    for st in studies:
        todo = E.pending(st, profile)
        if limit:
            todo = todo[:limit]
        print(f'[{st}] {len(todo)} runs to train')
        t0, gpu_s = time.time(), 0.0
        for i, r in enumerate(run_remote.map(todo, order_outputs=False), 1):
            r['cfg']['study'] = st
            E.store(st, profile, r)
            gpu_s += r['time_s']
            print(f'[{st}] {i:3d}/{len(todo)} {r["cfg"]["name"]:<40} val {r["final_val"]:.4f} '
                  f'{"DIVERGED" if r["diverged"] else ""} {r["time_s"]:.0f}s')
        print(f'[{st}] done in {time.time() - t0:.0f}s wall, {gpu_s / 3600:.2f} GPU-hours')
