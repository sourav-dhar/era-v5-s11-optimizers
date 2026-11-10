"""TinyShakespeare at the character level, with batches that depend only on (seed, step).

Making the batch a pure function of (seed, step) is what lets a decay *branch* taken from a
checkpoint see exactly the data the main run would have seen, and lets two schedules be compared
on identical data order.
"""
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def load_text() -> str:
    path = DATA / 'tinyshakespeare.txt'
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path.read_text(encoding='utf-8')


class CharData:
    def __init__(self, device='cpu'):
        text = load_text()
        self.chars = sorted(set(text))
        self.V = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        self.device = device
        self.split = {'train': data[:n].to(device), 'val': data[n:].to(device)}

    def batch(self, split, B, T, seed, step):
        d = self.split[split]
        g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(step) * 7919 + (split == 'val'))
        ix = torch.randint(len(d) - T - 1, (B,), generator=g).to(self.device)
        idx = ix[:, None] + torch.arange(T + 1, device=self.device)[None, :]
        chunk = d[idx]
        return chunk[:, :-1], chunk[:, 1:]

    def val_set(self, n, B, T):
        """A fixed held-out set, identical for every run (seed 424242)."""
        return [self.batch('val', B, T, 424242, i) for i in range(n)]
