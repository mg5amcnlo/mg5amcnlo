#!/usr/bin/env python3
"""The one-ratio-pane shape figures in the user's personal style.

A second, independent rendering of what ``plot_zz_shapes.py`` draws in the MG7
paper style.  The physics, the data, the binning, the ratio-pane windows and
the discrimination numbers are shared with that module; only the drawing
differs.

Usage::

    python3 plot_zz_shapes_userstyle.py [--data DIR] [--out DIR]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import observables_shapes as OS                                  # noqa: E402
from plot_zz_shapes import Shapes, window, log_ylim           # noqa: E402

# plot_zz_shapes imports the MG7-style module, which sets the paper rcParams
# (serif / usetex) at import time; reset, because the user's style sets none.
mpl.rcParams.update(mpl.rcParamsDefault)
matplotlib.use('Agg')
assert not mpl.rcParams['text.usetex'], (
    'the user style renders without usetex; if that ever changes these figures '
    'become exposed to the Type1 minus-subsetting bug and need their own check')

C_NLO = 'C0'
C_LI = 'C3'
C_LO = 'black'
C_TOT = 'C2'
FIGSIZE = (6, 6.6)
DPI = 300
MS = 3.2


def _step(ax, edges, y, **kw):
    ax.step(edges, np.concatenate([y[:1], y]), where='pre', **kw)


def draw(d, obs, outdir):
    edges = d.edges(obs)
    x = d.centres(obs)
    ynlo, enlo = d.density('nlo', obs)
    yli, eli = d.density('li', obs)
    ylo, elo = d.density('lo', obs)
    ytot = ynlo + yli
    etot = np.hypot(enlo, eli)

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.3], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    rx = fig.add_subplot(gs[1], sharex=ax)

    # ONLY the band between NLO and NLO+LI is filled -- the gg contribution.
    # Same choice as the parent study, and for the same reason: a solid stack
    # from zero on a log axis hides both the LO curve and the band.
    ax.fill_between(edges, np.concatenate([ytot[:1], ytot]),
                    np.concatenate([ynlo[:1], ynlo]),
                    step='pre', color=C_LI, alpha=0.30, lw=0)
    _step(ax, edges, ynlo, color=C_NLO, lw=1.2,
          label='NLO: p p > z z [QCD]  (qq~, qg, gq~)')
    _step(ax, edges, ytot, color=C_TOT, lw=1.4, label='NLO + LI gg')
    _step(ax, edges, yli, color=C_LI, lw=1.1, ls='--',
          label='loop induced: g g > z z')
    _step(ax, edges, ylo, color=C_LO, lw=1.1, ls='-.', label='LO: p p > z z')
    ax.errorbar(x, ytot, yerr=etot, fmt='o', ms=MS, color=C_TOT, zorder=5)

    ax.set_ylabel(OS.LABELS_SHAPES[obs][1])
    ax.set_xlim(edges[0], edges[-1])
    ax.tick_params(labelbottom=False)
    if obs in OS.LOGY_SHAPES:
        ax.set_yscale('log')
        lim = log_ylim((ylo, yli, ytot), obs)
        if lim:
            ax.set_ylim(*lim)
    else:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.45)
    ax.legend(loc='upper right' if obs == 'm_zz_fine' else 'best', fontsize=8)
    ax.set_title('$pp \\to ZZ$, 13 TeV, $\\mu_R=\\mu_F=m_Z$, '
                 '$p_T(Z) > %g$ GeV, NNPDF2.3 LO'
                 % d.meta['pt_z_min'], fontsize=9)

    # ONE ratio pane, NLO/LO and LI/LO separately -- not (NLO+LI)/LO.  Log y:
    # the two sit an order of magnitude apart and the figure is about their
    # shapes, not their levels.
    clip = window(obs)
    for ynum, enum, col, ls, lab in ((ynlo, enlo, C_NLO, '-', 'NLO / LO'),
                                     (yli, eli, C_LI, '--', 'LI gg / LO')):
        r, re_ = OS.ratio_with_errors(ynum, enum, ylo, elo)
        rc = np.clip(r, *clip)
        _step(rx, edges, rc, color=col, lw=1.0, ls=ls, alpha=0.55, zorder=3)
        rx.errorbar(x, rc, yerr=re_, fmt='o', ms=MS, color=col, label=lab,
                    zorder=4)
    rx.axhline(1.0, color=C_LO, ls=':', lw=0.9, zorder=2)
    rx.set_yscale('log')
    rx.set_ylim(*clip)
    ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
             if clip[0] <= t <= clip[1]]
    rx.set_yticks(ticks)
    rx.set_yticklabels(['%g' % t for t in ticks])
    rx.yaxis.set_minor_formatter(plt.NullFormatter())
    rx.set_ylabel('ratio to LO', fontsize=9)
    rx.legend(loc='best', fontsize=8, ncol=2)
    rx.set_xlabel(OS.LABELS_SHAPES[obs][0])

    if obs == 'm_zz_fine':
        two_mt = 2 * d.meta['shapes_m_top']
        for pane in (ax, rx):
            pane.axvline(two_mt, color='0.45', lw=0.9, ls=(0, (4, 2)),
                         zorder=0)
        ax.annotate('$2m_t$', xy=(two_mt, 0.62),
                    xycoords=('data', 'axes fraction'),
                    xytext=(4, 0), textcoords='offset points',
                    fontsize=9, color='0.35', ha='left', va='center')

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, 'shape_' + obs)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots_userstyle'))
    args = ap.parse_args()
    d = Shapes(args.data)
    for obs in OS.SHAPE_OBS:
        print('%-22s %s' % (obs, draw(d, obs, args.out)))


if __name__ == '__main__':
    main()
