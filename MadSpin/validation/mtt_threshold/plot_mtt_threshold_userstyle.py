#!/usr/bin/env python3
"""The same ``m_tt`` near-threshold comparison in the user's personal
matplotlib style.

A second, independent rendering of the figure ``plot_mtt_threshold.py`` draws in
the MG7-paper style.  It exists so the figure matches the rest of the user's own
material; the physics, the data and the numbers are identical, and
``plot_mtt_threshold.py`` is not modified.

The style conventions reproduced here are the ones the MadSpin
interference-closure validation wrote out in its ``STYLE_NOTES.md`` and that
``MadSpin/validation/mt_lineshape/plot_lineshape_userstyle.py`` already follows:
stock rcParams (no usetex, sans serif), figsize (6, 6), a [3, 1] gridspec, a
plain step for the reference and ``errorbar(fmt='o', ms=4)`` plus a faint
companion step for everything else, 'Ratio' on the lower panel, and the ratio
y-limit ladder 0.99 / 0.85 / 0.75 / 0.5.

Two deliberate departures, and the reason for each:

* the main panel is log-y.  ``dsigma/dm_tt`` falls by three decades from the
  peak into the sub-threshold tail, and the tail is the whole point of this
  figure -- on a linear axis every curve below 350 GeV collapses onto the
  baseline and the figure says nothing.  The ratio panel is linear, as in the
  user's script.
* the ratio ladder is extended downwards to include 0.  ``onshell`` is exactly
  zero below ``2 m_t`` and that zero has to be *on* the panel, drawn as an open
  marker on the axis, not clipped off the bottom.

Usage::

    python3 plot_mtt_threshold_userstyle.py [--data DIR] [--out DIR]
"""

import argparse
import math
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

# Reuse the data loading, the rebinning and the statistics of the existing
# script rather than rewriting them: same accessors, same ratio error
# propagation, same reference, same agreement definition.  That module sets the
# MG7 paper rcParams (serif / usetex) at import time, so the defaults are
# restored immediately afterwards -- the user's style sets no rcParams at all.
from plot_mtt_threshold import (                     # noqa: E402
    Data, ratio, structurally_empty, MODES, REF, CURVES_PLAIN, write_numbers,
    AGREE_HI,
)
mpl.rcParams.update(mpl.rcParamsDefault)
matplotlib.use('Agg')


C_REF = 'black'
COLOR = {'madspin': 'C0', 'PA': 'C1', 'onshell': 'C2'}

FIGSIZE = (6, 6)
HEIGHT_RATIOS = [3, 1]
HSPACE = 0.05
MS = 4
STEP_ALPHA = 0.55
DPI = 300

# The user's ladder, with 0-inclusive rungs appended: ``onshell`` sits at
# exactly 0 below threshold and must not be cropped.
RATIO_LADDER = [(0.99, 1.01), (0.85, 1.15), (0.75, 1.25), (0.5, 1.5),
                (-0.08, 1.6), (-0.08, 2.2), (-0.08, 3.2), (-0.08, 4.2),
                (-0.08, 5.5)]


def choose_ratio_ylim(series):
    """Smallest limit from the ladder that holds every point +- its error."""
    lo = hi = 1.0
    for r, re in series:
        good = np.isfinite(r) & np.isfinite(re)
        if not good.any():
            continue
        lo = min(lo, float((r[good] - re[good]).min()))
        hi = max(hi, float((r[good] + re[good]).max()))
    for cand in RATIO_LADDER:
        if lo >= cand[0] and hi <= cand[1]:
            return cand
    # Past the ladder, widen upwards only.  The user's own rungs are symmetric
    # about 1, but nothing here can go below 0 -- ``onshell`` sits exactly at 0
    # and everything else is a positive cross section -- so mirroring a large
    # upward excursion downwards would spend half the panel on empty space and
    # squash the part that carries the answer.
    return (-0.1, 1.0 + math.ceil((hi - 1.0) * 10.0) / 10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots_userstyle'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = Data(args.data)
    two_mt = d.two_mt
    lo, hi = d.edges[0], d.edges[-1]

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=HEIGHT_RATIOS, hspace=HSPACE)
    ax = fig.add_subplot(gs[0])
    rx = fig.add_subplot(gs[1], sharex=ax)

    for a in (ax, rx):
        a.axvspan(lo, two_mt, facecolor='0.90', edgecolor='none', zorder=0)
        a.axvline(two_mt, color='0.35', lw=1.0, ls='--', zorder=1)

    den, dene, dcnt = d.density(REF)
    ax.step(d.edges, np.concatenate([den[:1], den]), where='pre',
            color=C_REF, lw=1.2, label=CURVES_PLAIN['truth'], zorder=5)

    for key in MODES:
        y, ye, cnt = d.density(key)
        draw = np.where(cnt > 0, y, np.nan)
        ax.step(d.edges, np.concatenate([draw[:1], draw]), where='pre',
                color=COLOR[key], lw=1.0, alpha=STEP_ALPHA, zorder=3)
        ax.errorbar(d.centres, draw, yerr=np.where(cnt > 0, ye, np.nan),
                    fmt='o', ms=MS, color=COLOR[key],
                    label=CURVES_PLAIN[key], zorder=4)

    ax.set_yscale('log')
    ax.set_ylabel(r'$d\sigma/dm_{t\bar{t}}$  [pb/GeV]')
    ax.set_xlim(lo, hi)
    ax.tick_params(labelbottom=False)
    ax.legend(loc='lower right', fontsize=8.5)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 25)
    n_on = d.meta['runs']['onshell']['nevents']
    ax.set_title(r'$pp\to t\bar{t}j$, 13 TeV, LO, $\mu_R=\mu_F=m_t$, '
                 r'BW cut $=%g\,\Gamma_t$' % d.meta.get('bwcutoff', 15.0),
                 fontsize=10)
    ax.text(0.02, 0.965,
            'below $2m_t$: no on-shell $t\\bar{t}$ pair can land here.\n'
            'onshell: 0 of %s events, exactly -- structural, not statistical.\n'
            'madspin / PA reach it only via the production reshuffle,\n'
            'which rescales the recoil jet and so moves $m_{t\\bar{t}}$.'
            % '{:,}'.format(int(n_on)).replace(',', ' '),
            transform=ax.transAxes, ha='left', va='top', fontsize=7.5,
            color='0.25')
    ax.annotate('$2m_t$', xy=(two_mt, 0.02), xycoords=('data', 'axes fraction'),
                xytext=(3, 0), textcoords='offset points', ha='left',
                va='bottom', fontsize=9, color='0.35')

    rx.axhspan(0.9, 1.1, facecolor='C0', alpha=0.10, zorder=0)
    rx.axhspan(0.95, 1.05, facecolor='C0', alpha=0.16, zorder=0)
    rx.axhline(1.0, color=C_REF, ls='--', lw=0.9, zorder=2)

    series = []
    for key in MODES:
        y, ye, cnt = d.density(key)
        r, re = ratio(y, ye, den, dene)
        # Same distinction as the MG7-style figure: a structural zero
        # (``onshell`` below 2 m_t) is drawn AS a zero with an open marker; any
        # other empty bin is a statement about the sample size and is left as a
        # gap.
        struct = structurally_empty(d, key) & (dcnt > 0)
        stat = (cnt == 0) & (dcnt > 0) & ~struct
        rr = np.where(struct, 0.0, np.where(stat, np.nan, r))
        rx.step(d.edges, np.concatenate([rr[:1], rr]), where='pre',
                color=COLOR[key], lw=1.0, alpha=STEP_ALPHA, zorder=3)
        gone = struct | stat
        rx.errorbar(d.centres, np.where(gone, np.nan, r),
                    yerr=np.where(gone, np.nan, re), fmt='o', ms=MS,
                    color=COLOR[key], zorder=4)
        if struct.any():
            rx.plot(d.centres[struct], np.zeros(struct.sum()), 'o',
                    mfc='white', mec=COLOR[key], mew=1.2, ms=MS + 1, zorder=6)
        series.append((np.where(gone, np.nan, r), np.where(gone, np.nan, re)))

    rx.set_ylim(*choose_ratio_ylim(series))
    rx.text(0.99, 0.92, 'bands: $\\pm5\\%$, $\\pm10\\%$', transform=rx.transAxes,
            ha='right', va='top', fontsize=7, color='C0')
    rx.set_ylabel('Ratio')
    rx.set_xlabel(r'$m_{t\bar{t}}$ [GeV]   '
                  r'(per-event $m$ of $(W^+b)+(W^-\bar{b})$)')
    rx.set_xlim(lo, hi)

    fig.subplots_adjust(hspace=0.1, left=0.15, right=0.97,
                        bottom=0.12, top=0.93)
    base = os.path.join(args.out, 'mtt_threshold')
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=DPI)
    plt.close(fig)
    print('wrote %s.pdf / .png' % base)

    with open(os.path.join(args.out, 'numbers.txt'), 'w') as fh:
        write_numbers(d, args.out, fh)


if __name__ == '__main__':
    main()
