#!/usr/bin/env python3
"""The loop-induced ``g g > z z`` MadSpin comparison in the user's personal
matplotlib style.

A second, independent rendering of the figures ``plot_zz_loopinduced.py`` draws
in the MG7 paper style.  It exists so the figures match the rest of the user's
own material; the physics, the data and the numbers are identical, and it shares
the data loading, the ratio statistics, the structural-emptiness test and the
numeric report with that module -- everything except the drawing.

The style conventions are the ones the sibling studies under
``MadSpin/validation/`` follow, and they are followed here unchanged: stock
rcParams (no usetex, sans serif), figsize (6, 6), a [3, 1] gridspec, a plain
step for the reference and ``errorbar(fmt='o', ms=4)`` plus a faint companion
step for everything else, a ratio panel clipped to a fixed window rather than an
autoscaled ladder, arrows for measured points that leave the window, and open
markers for exact structural zeros.

Usage::

    python3 plot_zz_loopinduced_userstyle.py [--data DIR] [--out DIR]
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

import observables as OBS                                        # noqa: E402
# The MG7-style module sets the paper rcParams (serif / usetex) at import time,
# so they are reset immediately afterwards -- the user's style sets no rcParams
# at all.  The accessors below are literally the same code on both sides of the
# comparison.
from plot_zz_loopinduced import (                                # noqa: E402
    Data, ratio, structurally_empty, MODES, REF, CURVES_PLAIN, LOGY,
    RATIO_CLIP,
)
mpl.rcParams.update(mpl.rcParamsDefault)
matplotlib.use('Agg')

# The reset above also turns ``text.usetex`` back OFF, and that has one
# consequence worth spelling out: these figures never go through matplotlib's
# usetex Type1 font-subsetting path, so the minus-eating bug that
# ``plot_zz_loopinduced._fix_type1_subset_minus`` works around cannot bite them.
#
# Do NOT "check" these PDFs with ``plot_zz_loopinduced.check_minus``.  That
# function greps the file for ``/minus``, which a non-usetex PDF carries anyway,
# so it would report True whether or not the workaround is active -- including
# with ``NO_MINUS_FIX=1``.  A check that passes either way is worse than no
# check.  The discriminating one is ``plot_zz_loopinduced.py --check-minus`` on
# the MG7-style PDFs, which ARE rendered with usetex.
assert not mpl.rcParams['text.usetex'], (
    'the user style renders without usetex; if that ever changes these figures '
    'become exposed to the Type1 subsetting bug and need their own check')


C_REF = 'black'
COLOR = {'madspin': 'C0', 'PA': 'C1', 'onshell': 'C2', 'none': 'C3'}
FIGSIZE = (6, 6)
HEIGHT_RATIOS = [3, 1]
HSPACE = 0.05
MS = 4
STEP_ALPHA = 0.55
DPI = 300


def _step(ax, edges, y, **kw):
    ax.step(edges, np.concatenate([y[:1], y]), where='pre', **kw)


def draw(d, obs, outdir):
    edges = d.edges(obs)
    x = d.centres(obs)
    order = [m for m in MODES if d.has(m, obs)]

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=HEIGHT_RATIOS, hspace=HSPACE)
    ax = fig.add_subplot(gs[0])
    rx = fig.add_subplot(gs[1], sharex=ax)

    yref, eref = d.density(REF, obs)
    _step(ax, edges, yref, color=C_REF, lw=1.2, label=CURVES_PLAIN[REF],
          zorder=5)

    for key in order:
        y, e = d.density(key, obs)
        shown = np.where(y > 0, y, np.nan) if obs in LOGY else y
        _step(ax, edges, shown, color=COLOR[key], lw=1.0, alpha=STEP_ALPHA,
              zorder=3)
        ax.errorbar(x, shown, yerr=np.where(np.isfinite(shown), e, np.nan),
                    fmt='o', ms=MS, color=COLOR[key], label=CURVES_PLAIN[key],
                    zorder=4)

    if obs in LOGY:
        ax.set_yscale('log')
    ax.set_ylabel(OBS.LABELS[obs][1])
    ax.set_xlim(edges[0], edges[-1])
    ax.tick_params(labelbottom=False)
    # Headroom BEFORE the legend: 'best' minimises overlap against the axes as
    # they are at the time it is called, so making room afterwards leaves the
    # legend sitting where the curves used to be.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * (8.0 if obs in LOGY else 1.35))
    ax.legend(loc='best', fontsize=8)
    ax.set_title('$gg \\to ZZ \\to e^{+}e^{-}\\mu^{+}\\mu^{-}$, 13 TeV, '
                 'loop induced, $\\mu_R=\\mu_F=m_Z$, BW cut $=%g\\,\\Gamma_Z$'
                 % d.meta['BW_cut'], fontsize=9)

    rx.axhspan(0.9, 1.1, facecolor='C0', alpha=0.10, zorder=0)
    rx.axhspan(0.95, 1.05, facecolor='C0', alpha=0.16, zorder=0)
    rx.axhline(1.0, color=C_REF, ls='--', lw=0.9, zorder=2)
    rx.set_ylim(*RATIO_CLIP)

    struct_of = {k: structurally_empty(d.density(k, obs)[0], k, obs) & (yref > 0)
                 for k in order}
    circled = [k for k in order if struct_of[k].any()]

    n_out = 0
    for key in order:
        y, e = d.density(key, obs)
        r, re_ = ratio(y, e, yref, eref)
        struct = struct_of[key]
        gone = struct | ~np.isfinite(r)
        rr = np.where(struct, 0.0, np.where(gone, np.nan, r))
        _step(rx, edges, rr, color=COLOR[key], lw=1.0, alpha=STEP_ALPHA,
              zorder=3)
        rx.errorbar(x, np.where(gone, np.nan, r),
                    yerr=np.where(gone, np.nan, re_), fmt='o', ms=MS,
                    color=COLOR[key], zorder=4)
        if struct.any():
            ring = circled.index(key)
            rx.plot(x[struct], np.full(struct.sum(), RATIO_CLIP[0]), 'o',
                    mfc='white', mec=COLOR[key], mew=1.2,
                    ms=MS + 1 + 2.4 * ring, clip_on=False,
                    zorder=8 + (len(circled) - ring))
        for xi, yi in zip(x, np.where(gone, np.nan, r)):
            if not np.isfinite(yi):
                continue
            if yi > RATIO_CLIP[1] or yi < RATIO_CLIP[0]:
                edge = RATIO_CLIP[1] if yi > RATIO_CLIP[1] else RATIO_CLIP[0]
                off = -0.12 if yi > RATIO_CLIP[1] else 0.12
                rx.annotate('', xy=(xi, edge),
                            xytext=(xi, edge + off * (RATIO_CLIP[1] - RATIO_CLIP[0])),
                            arrowprops=dict(arrowstyle='-|>', color=COLOR[key],
                                            lw=0.9))
                n_out += 1

    rx.set_xlabel(OBS.LABELS[obs][0])
    rx.set_ylabel('ratio', fontsize=9)
    rx.text(0.99, 0.92, 'bands: $\\pm5\\%$, $\\pm10\\%$', transform=rx.transAxes,
            ha='right', va='top', fontsize=7, color='C0')

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, obs)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    # NOT ``'none'`` for the empty case: one of the modes is literally called
    # ``none``, so "open circles: none" would be unreadable.
    print('%-12s ratio pane clipped to %s: %d point(s) drawn as arrows; '
          'open circles (exact structural zeros): %s'
          % (obs, RATIO_CLIP, n_out,
             ', '.join(circled) if circled else '(no modes)'))
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots_userstyle'))
    args = ap.parse_args()
    d = Data(args.data)
    for obs in d.meta['observables']:
        draw(d, obs, args.out)


if __name__ == '__main__':
    main()
