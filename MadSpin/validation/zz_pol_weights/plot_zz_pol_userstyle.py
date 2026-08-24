#!/usr/bin/env python3
"""The polarisation-weight figures in the user's personal style.

A second, independent rendering of what ``plot_zz_pol.py`` draws in the MG7
paper style.  The physics, the data, the selection, the binning and the ratio
statistics are the same objects -- all of them come from ``pol_analysis`` --
and only the drawing differs.

The extra spinmode samples are overlaid on the distribution pane only, on the
same rule as the MG7 figures: drawn where the selection leaves enough events to
be a measurement, named and explained in the pane where it does not.

The conventions are the ones the sibling studies under ``MadSpin/validation/``
follow: stock rcParams (no usetex, sans serif), plain steps for the reference
and ``errorbar(fmt='o', ms=4)`` plus a faint companion step for everything
else, shaded tolerance bands behind the ratio panes, and a dashed reference
line.  The layout is the same three tiers, for the same reason: the sum pane
is the physics and gets the width and the emphasis, the four components get a
2 x 2 breakdown underneath.

Usage::

    python3 plot_zz_pol_userstyle.py [--data DIR] [--out DIR]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import pol_analysis as PA                                        # noqa: E402
from plot_zz_pol import _ratio_ylim                              # noqa: E402

# ``plot_zz_pol`` imports the MG7-style module, which sets the paper rcParams
# (serif / usetex) at import time; they are reset here because the user's style
# sets no rcParams at all.
mpl.rcParams.update(mpl.rcParamsDefault)
matplotlib.use('Agg')

# The reset also turns ``text.usetex`` back OFF, so these figures never go
# through matplotlib's usetex Type1 subsetting path and cannot be bitten by the
# minus-eating bug ``plot_zz_loopinduced._fix_type1_subset_minus`` works around.
# Do NOT "check" these PDFs with ``check_minus``: a non-usetex PDF carries
# ``/minus`` whether or not the workaround ran, so the check would pass either
# way, which is worse than no check.  The discriminating one is
# ``plot_zz_pol.py --check-minus`` on the MG7-style PDFs.
assert not mpl.rcParams['text.usetex'], (
    'the user style renders without usetex; if that ever changes these figures '
    'become exposed to the Type1 subsetting bug and need their own check')

C_REF = 'black'
COLOR = {'LL': 'C0', 'TT': 'C3', 'TL': 'C2', 'LT': 'C4', 'SUM': 'C5',
         # The extra spinmodes take the two cycle colours the components leave
         # free; the same two hues as the MG7 figures, in this style's palette.
         'onshell': 'C1', 'PA': 'C9'}
LS_EXTRA = {'onshell': (0, (6, 2)), 'PA': (0, (3, 1, 1, 1, 1, 1))}
FIGSIZE = (7.2, 10.0)
MS = 4
STEP_ALPHA = 0.55
DPI = 300

TITLE = ('p p > z z [QCD] (MC@NLO) + MadSpin, Pythia8 showered\n'
         '13 TeV, 250k events, decay z > light light (inclusive)')


def _step(ax, edges, y, **kw):
    ax.step(edges, np.concatenate([y[:1], y]), where='pre', **kw)


def draw(d, obs, outdir, extras=()):
    c = PA.Curves(d, obs)
    edges, x = c.edges, c.centres()
    xlab, ylab = PA.LABELS_TXT[obs]

    fig = plt.figure(figsize=FIGSIZE)
    # See the note in plot_zz_pol.draw: the distribution and the sum pane are a
    # stacked PAIR sharing one x axis, the 2 x 2 is a separate block with its
    # own, and the three gaps that implies cannot come from one hspace.
    gs = fig.add_gridspec(2, 1, height_ratios=[4.7, 2.5], hspace=0.13)
    top = gs[0].subgridspec(2, 1, height_ratios=[3.0, 1.7], hspace=0.06)
    sub = gs[1].subgridspec(2, 2, hspace=0.12, wspace=0.28)
    ax = fig.add_subplot(top[0])
    axs = fig.add_subplot(top[1], sharex=ax)
    small = [fig.add_subplot(sub[0, 0], sharex=ax),
             fig.add_subplot(sub[0, 1], sharex=ax),
             fig.add_subplot(sub[1, 0], sharex=ax),
             fig.add_subplot(sub[1, 1], sharex=ax)]

    # -- the distribution: the total, with the four components on top of it ---
    y, e = c.dist['full']
    _step(ax, edges, y, color=C_REF, lw=1.6, label=PA.CURVE_TXT['full'],
          zorder=6)
    ax.errorbar(x, y, yerr=e, fmt='none', ecolor=C_REF, elinewidth=1.0,
                zorder=6)
    # The other two spinmodes: full distributions like the black one, so they
    # are drawn as steps with bars and not with the components' markers.
    dropped = []
    for key, dx in extras:
        fd = PA.full_distribution(dx, obs)
        if not fd['drawable']:
            dropped.append((key, fd['n_sel']))
            continue
        _step(ax, edges, fd['y'], color=COLOR[key], lw=1.3, ls=LS_EXTRA[key],
              label=PA.EXTRA_TXT[key], zorder=5)
        ax.errorbar(x, fd['y'], yerr=fd['err'], fmt='none',
                    ecolor=COLOR[key], elinewidth=0.9, zorder=5)
    for k in PA.POL_KEYS:
        yk, ek = c.dist[k]
        shown = np.where(yk > 0, yk, np.nan) if obs in PA.LOGY else yk
        _step(ax, edges, shown, color=COLOR[k], lw=1.0, alpha=STEP_ALPHA,
              zorder=3)
        ax.errorbar(x, shown, yerr=np.where(np.isfinite(shown), ek, np.nan),
                    fmt='o', ms=MS, color=COLOR[k], label=PA.CURVE_TXT[k],
                    zorder=4)
    if obs in PA.LOGY:
        ax.set_yscale('log')
    ax.set_ylabel(ylab)
    ax.set_xlim(edges[0], edges[-1])
    lo, hi = ax.get_ylim()
    # The user style's legend has an opaque frame, so on the log pane it needs
    # enough headroom to clear the total's peak outright rather than merely to
    # sit above the autoscale top: at 40x its lower edge landed on the black
    # step.  Checked on the rendered PNG, not on the axis limits.
    ax.set_ylim(lo, hi * (400.0 if obs in PA.LOGY else 1.45))
    ax.legend(loc='upper left', fontsize=8)
    if dropped:
        # See plot_zz_pol.draw: a figure that silently drops two of its three
        # samples implies a comparison it did not make.
        who = ' and '.join('%s (N=%d)' % (k, n) for k, n in dropped)
        # Below the legend, not beside it: this style's legend has an opaque
        # frame and is wide enough to reach the middle of the pane, so the
        # upper-right corner the MG7 figure uses is not free here.
        ax.text(0.985, 0.60,
                'spinmode %s not drawn: too few events\nfor a comparison at '
                'this selection' % who,
                transform=ax.transAxes, fontsize=7.5, ha='right', va='top',
                color='0.35')
    ax.set_title(TITLE, fontsize=9)
    ax.tick_params(labelbottom=False)

    # -- the sum: the polarisation interference, on its own scale ------------
    r, er, _ = c.ratios['SUM']
    Rint, Eint = c.integrated['SUM']
    axs.axhspan(0.95, 1.05, facecolor='C0', alpha=0.10, zorder=0)
    axs.axhspan(0.98, 1.02, facecolor='C0', alpha=0.16, zorder=0)
    axs.axhline(1.0, color=C_REF, ls='--', lw=1.0, zorder=2)
    _step(axs, edges, r, color=COLOR['SUM'], lw=1.2, alpha=0.75, zorder=3)
    axs.errorbar(x, r, yerr=er, fmt='o', ms=MS + 0.5, color=COLOR['SUM'],
                 zorder=4)
    axs.set_ylim(*_ratio_ylim(r, er, 1.0))
    axs.set_ylabel(PA.RATIO_TXT['SUM'], fontsize=8)
    # The value, not its significance: the sigma-from-1 lives in numbers.txt
    # and RESULTS.md.
    axs.text(0.012, 0.05,
             'integrated: %.4f +- %.4f\nbands: +-2%%, +-5%%' % (Rint, Eint),
             transform=axs.transAxes, fontsize=7.5, ha='left', va='bottom')
    axs.text(0.988, 0.94, 'POLARISATION INTERFERENCE', transform=axs.transAxes,
             fontsize=8.5, ha='right', va='top', color=COLOR['SUM'])
    axs.set_xlabel(xlab, fontsize=10)
    for s in axs.spines.values():
        s.set_linewidth(1.6)

    # -- the breakdown -------------------------------------------------------
    for a, k in zip(small, ['LL', 'TT', 'TL', 'LT']):
        rk, ek, _ = c.ratios[k]
        Rk, Ek = c.integrated[k]
        a.axhline(Rk, color=C_REF, ls='--', lw=0.9, zorder=2,
                  label='integrated %.4f +- %.4f' % (Rk, Ek))
        _step(a, edges, rk, color=COLOR[k], lw=1.0, alpha=STEP_ALPHA, zorder=3)
        a.errorbar(x, rk, yerr=ek, fmt='o', ms=MS, color=COLOR[k], zorder=4)
        a.set_ylim(*_ratio_ylim(rk, ek, Rk))
        a.set_ylabel(PA.RATIO_TXT[k], fontsize=9)
        # See the note in plot_zz_pol.draw: no fixed corner is free in every
        # pane, so the number rides on the reference line's legend entry.
        a.legend(loc='best', fontsize=7, framealpha=0.75)
    for a in small[:2]:
        a.tick_params(labelbottom=False)
    for a in small[2:]:
        a.set_xlabel(xlab, fontsize=10)
    # The 2 x 2 panes are half the figure wide, so the default locator packs
    # the mass axis tightly enough that the labels run into one another.
    small[-1].xaxis.set_major_locator(MaxNLocator(6))

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, PA.SHORT[obs])
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    g = PA.diagnostics(d, obs)
    print('%-10s N=%5d  purity=%.3f  eff=%.3f  sum/full = %.4f +- %.4f'
          % (PA.SHORT[obs], c.n_sel, g['purity'], g['efficiency'], Rint, Eint))
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots_userstyle'))
    a = ap.parse_args()
    d = PA.Data(a.data)
    extras = PA.load_extras(a.data)
    for obs in PA.OBS:
        draw(d, obs, a.out, extras)


if __name__ == '__main__':
    main()
