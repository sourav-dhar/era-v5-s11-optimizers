"""One visual language for every figure — the Session 10 palette and chrome, so the series reads as one."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parents[1] / 'figures'

PAL = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100',
           magenta='#e87ba4', green='#008300', violet='#6250d6', red='#e34948')
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e7e6e2', '#fcfcfb'
SEQ = ['#cfe0f6', '#9cc0ec', '#5f97de', '#2a78d6', '#1c5aa6', '#123d73']   # one-hue ramp (light -> dark blue)


def apply():
    mpl.rcParams.update({
        'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
        'axes.edgecolor': GRID, 'axes.linewidth': 0.8, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
        'axes.labelcolor': INK2, 'axes.labelsize': 9.5, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'xtick.major.size': 0, 'ytick.major.size': 0,
        'xtick.minor.size': 0, 'ytick.minor.size': 0,
        'text.color': INK, 'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
        'axes.titlelocation': 'left', 'axes.titlepad': 24, 'legend.frameon': False, 'legend.fontsize': 8.5,
        'lines.linewidth': 2.0, 'lines.solid_capstyle': 'round', 'figure.dpi': 110,
        'savefig.dpi': 200, 'savefig.bbox': 'tight',
        'axes.prop_cycle': mpl.cycler(color=[PAL[k] for k in ['blue', 'orange', 'aqua', 'yellow', 'magenta', 'green', 'violet', 'red']]),
    })


def title(ax, head, sub=None):
    ax.set_title(head)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.8, color=INK2, va='bottom')


def save(fig, name, show=True):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f'{name}.png')
    if show:
        plt.show()
    else:
        plt.close(fig)


def label_end(ax, x, y, text, color, dx=4, dy=0, **kw):
    """Direct label at the end of a line, instead of a legend."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points', color=color, fontsize=8.5,
                va='center', fontweight='bold', **kw)


def ramp(n):
    idx = np.linspace(1, len(SEQ) - 1, n).round().astype(int)
    return [SEQ[i] for i in idx]


apply()
