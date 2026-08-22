#!/usr/bin/env python3
"""The stacked ``NLO + gg`` physics figure in the user's personal style.

A second, independent rendering of what ``plot_zz_stack.py`` draws in the MG7
paper style.  The physics, the data and the numbers are identical; the data
loading, the observable list and the ratio statistics are shared with that
module and only the drawing differs.

Usage::

    python3 plot_zz_stack_userstyle.py [--data DIR] [--out DIR]
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

import observables_zz as OZ                                      # noqa: E402
from plot_zz_stack import Stack, STACK_OBS, LOGY, window         # noqa: E402
_LI = os.path.abspath(os.path.join(_HERE, '..', 'zz_loopinduced'))
if _LI not in sys.path:
    sys.path.insert(0, _LI)
from plot_zz_loopinduced import ratio                            # noqa: E402

# plot_zz_stack imports the MG7-style module, which sets the paper rcParams
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
FIGSIZE = (6, 7.2)
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
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    r1 = fig.add_subplot(gs[1], sharex=ax)
    r2 = fig.add_subplot(gs[2], sharex=ax)

    # ONLY the band between NLO and NLO+LI is filled -- the gg contribution.
    # Filling from zero as well would be the textbook stacked histogram and on a
    # four-decade log axis it is one solid colour that hides both the LO curve
    # and the band it is supposed to show.  Same choice as the MG7-style module.
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

    if obs in LOGY:
        ax.set_yscale('log')
    ax.set_ylabel(OZ.LABELS_ZZ[obs][1])
    ax.set_xlim(edges[0], edges[-1])
    ax.tick_params(labelbottom=False)
    if obs in LOGY:
        # bounded to four decades below the peak of the total, rather than to
        # whatever the smallest non-empty bin of the loop-induced curve happens
        # to be: that tail is a handful of events and letting it set the axis
        # squeezes everything that carries rate into the top fifth of the pane
        top = float(np.nanmax(ytot))
        ax.set_ylim(top * 1e-4, top * 25.0)
    else:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.45)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('$pp \\to ZZ$, 13 TeV, $\\mu_R=\\mu_F=m_Z$, '
                 '$p_T(Z) > %g$ GeV, NNPDF2.3 LO'
                 % d.meta['pt_z_min'], fontsize=9)

    clip = window(obs)
    for pane, (ynum, enum, col, lab) in zip(
            (r1, r2),
            ((ynlo, enlo, C_NLO, 'NLO / LO'),
             (ytot, etot, C_TOT, '(NLO+LI) / LO'))):
        r, re_ = ratio(ynum, enum, ylo, elo)
        pane.axhline(1.0, color=C_LO, ls='--', lw=0.9, zorder=2)
        _step(pane, edges, np.clip(r, *clip), color=col, lw=1.0,
              alpha=0.55, zorder=3)
        pane.errorbar(x, np.clip(r, *clip), yerr=re_, fmt='o', ms=MS,
                      color=col, zorder=4)
        pane.set_ylim(*clip)
        pane.set_ylabel(lab, fontsize=9)
    r1.tick_params(labelbottom=False)
    r2.set_xlabel(OZ.LABELS_ZZ[obs][0])

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, 'stack_' + obs)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('%-20s NLO/LO = %.3f   (NLO+LI)/LO = %.3f   (integrated)'
          % (obs, d.sigma('nlo') / d.sigma('lo'),
             (d.sigma('nlo') + d.sigma('li')) / d.sigma('lo')))
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots_userstyle'))
    args = ap.parse_args()
    d = Stack(args.data)
    for obs in STACK_OBS:
        draw(d, obs, args.out)


if __name__ == '__main__':
    main()
