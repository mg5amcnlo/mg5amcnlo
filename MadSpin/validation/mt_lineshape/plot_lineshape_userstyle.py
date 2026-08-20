#!/usr/bin/env python3
"""The same m(t) / m(tbar) lineshape comparisons in the user's personal
matplotlib style.

This is a second, independent rendering of the figures ``plot_lineshape.py``
draws in the MG7-paper style.  It exists so the figures match the rest of the
user's own material; the physics, the data and the numbers are identical, and
``plot_lineshape.py`` is not modified.

Everything is regenerated from ``data/histograms.npz`` and ``data/meta.json``.
No MadSpin run, no LHE file and no external parser are needed -- numpy and
matplotlib are the only requirements.

The style conventions reproduced here are the ones the MadSpin
interference-closure validation wrote out in its ``STYLE_NOTES.md``, read off
the user's own ``plot_hist_with_ratio`` / ``plot_hist_with_ratio_multi`` /
``plot_wb_mass`` (that directory is on the interference-closure branch, not
this one, so the conventions are restated here rather than imported):
stock rcParams (no usetex, sans serif), figsize (6, 6), a [3, 1] gridspec, a
plain step for the reference and ``errorbar(fmt='o', ms=4)`` plus a faint
companion step for everything else, 'Ratio' on the lower panel, and the ratio
y-limit ladder 0.99 / 0.85 / 0.75 / 0.5.

One deliberate departure, and the reason for it: the main panel is log-y.  A
Breit-Wigner drops three decades between the pole and the +-15 Gamma
truncation, and the tails are the whole point of this comparison -- on a linear
axis every curve outside the peak collapses onto the baseline and the figure
says nothing.  The ratio panel is linear, as in the user's script.

Usage::

    python3 plot_lineshape_userstyle.py [--data DIR] [--out DIR]
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
# propagation, same choice of reference.  That module sets the MG7 paper
# rcParams (serif / usetex) at import time, so the defaults are restored
# immediately afterwards -- the user's style sets no rcParams at all.
from plot_lineshape import (                    # noqa: E402
    Data, ratio, LABEL, REF, write_numbers,
)
mpl.rcParams.update(mpl.rcParamsDefault)
matplotlib.use('Agg')


# --- the user's style -----------------------------------------------------
C_REF = 'black'         # reference: plain step, and the dashed ratio line
CYCLE = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C8']
C_BW = 'C7'             # the Breit-Wigner standard

FIGSIZE = (6, 6)
HEIGHT_RATIOS = [3, 1]
HSPACE = 0.05
ADJUST = dict(hspace=0.1, left=0.15, right=0.97, bottom=0.12, top=0.96)
# The user's script uses ms=4.  These figures carry 70 bins in a 6-inch panel,
# where ms=4 markers touch and hide the curve underneath, so the marks are one
# point smaller.  Everything else about the mark -- 'o', the faint companion
# step, the open-marker alternation -- is unchanged.
MS = 3
STEP_ALPHA = 0.55
OPEN = dict(markerfacecolor='none', markeredgewidth=1.2)
DPI = 300

RATIO_LADDER = [(0.99, 1.01), (0.85, 1.15), (0.75, 1.25), (0.5, 1.5)]

# The data are per-event virtualities normalised to unit area, so the y label
# departs from the user's literal 'Events' and says what is drawn.
YLABEL = r'$(1/N)\ \mathrm{d}N/\mathrm{d}m$  [GeV$^{-1}$]'
XLABEL = {
    't': r'$m(t)$ [GeV]   (per-event $\sqrt{E^2-|p|^2}$ of the intermediate $t$)',
    'tbar': r'$m(\bar{t})$ [GeV]   (per-event $\sqrt{E^2-|p|^2}$ of the '
            r'intermediate $\bar{t}$)',
}


def choose_ratio_ylim(series):
    """Smallest limit from the user's ladder that holds every point +- error.

    Nothing is clipped: if the spread escapes even the widest of their limits,
    the limits are widened to fit it.  A disagreement between two unweighting
    schemes is the whole point of these figures and must not be cropped off the
    bottom of the panel.
    """
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
    half = max(1.0 - lo, hi - 1.0)
    half = math.ceil(half * 10.0) / 10.0
    return (1.0 - half, 1.0 + half)


def _panels():
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=HEIGHT_RATIOS, hspace=HSPACE)
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)
    return fig, ax_main, ax_ratio


def _marks(ax, ctr, edges, y, ye, color, label, open_markers=False):
    """The user's non-reference mark: errorbar 'o' plus a faint companion step."""
    style = dict(OPEN, markeredgecolor=color) if open_markers else {}
    ax.errorbar(ctr, y, yerr=ye, fmt='o', ms=MS, label=label, color=color,
                **style)
    ax.step(edges, np.append(y, y[-1]), where='post', color=color,
            alpha=STEP_ALPHA)


def _reference(ax, edges, y, color, label):
    """The user's reference mark: a plain step, no markers, no error bars."""
    ax.step(edges, np.append(y, y[-1]), where='post', color=color, label=label,
            linewidth=1.4)


def lineshape_figure(d, keys, tag, outbase, formats, ref=REF,
                     show_delta=False):
    edges, ctr = d.edges, d.centre
    rp, rpe = d.density(ref, tag)
    bw = d.bw()

    fig, ax_main, ax_ratio = _panels()
    ax_main.set_yscale('log')

    _reference(ax_main, edges, rp, C_REF, '%s (reference)' % LABEL[ref])
    ax_main.step(edges, np.append(bw, bw[-1]), where='post', color=C_BW,
                 linestyle='--', linewidth=1.2,
                 label=r'Breit-Wigner, $M=%.2f$, $\Gamma=%.4f$ GeV'
                       % (d.pole, d.gamma))
    others = [k for k in keys if k != ref]
    series = []
    for i, key in enumerate(others):
        color = CYCLE[i % len(CYCLE)]
        p, pe = d.density(key, tag)
        _marks(ax_main, ctr, edges, p, pe, color, LABEL[key],
               open_markers=(i % 2 == 1))
        r, re = ratio(p, pe, rp, rpe)
        series.append((r, re, color, i % 2 == 1))
    if show_delta:
        # onshell and none put every event at the pole mass; a histogram of a
        # delta would be a one-bin spike whose height is set by the binning, so
        # the line says what they are instead of drawing a misleading curve.
        # ymax keeps the line out of the legend box it would otherwise cross
        ax_main.axvline(d.pole, linestyle=':', color='grey', linewidth=1.0,
                        ymax=0.62,
                        label=r'onshell ($\times$3), none: $\delta(m-M)$')

    ylim = choose_ratio_ylim([(r, re) for r, re, _c, _o in series])
    for r, re, color, opened in series:
        good = np.isfinite(r)
        style = dict(OPEN, markeredgecolor=color) if opened else {}
        ax_ratio.errorbar(ctr[good], r[good], yerr=re[good], fmt='o', ms=MS,
                          color=color, zorder=2, **style)
    rb, _ = ratio(bw, np.zeros_like(bw), rp, rpe)
    ax_ratio.step(edges, np.append(rb, rb[-1]), where='post', color=C_BW,
                  linestyle='--', linewidth=1.2, zorder=1)

    ax_main.set_ylabel(YLABEL)
    lo, hi = ax_main.get_ylim()
    ax_main.set_ylim(lo, hi * 10 ** (1.10 + 0.075 * len(ax_main.get_lines())))
    ax_main.legend(fontsize=7.5, borderpad=0.3, labelspacing=0.25,
                   handletextpad=0.4, loc='upper left')
    ax_main.tick_params(labelbottom=False)

    ax_ratio.axhline(1, linestyle='--', color=C_REF, zorder=1)
    ax_ratio.set_xlabel(XLABEL[tag])
    ax_ratio.set_ylabel('Ratio')
    ax_ratio.set_ylim(*ylim)

    fig.subplots_adjust(**ADJUST)
    written = []
    for ext in formats:
        path = '%s.%s' % (outbase, ext)
        plt.savefig(path, bbox_inches='tight', dpi=DPI)
        written.append(path)
    plt.close(fig)
    return written, ylim


# Same families as the MG7-style script, so the two sets of figures are
# one-to-one; imported rather than duplicated would be better, but the family
# list there carries LaTeX titles this style cannot use.
FAMILIES = [
    ('all', 't', 'us_lineshape_mt_all',
     ['madspin_joint', 'madspin_sequential', 'madspin_seqglobal',
      'PA_joint', 'PA_sequential', 'PA_seqglobal', 'PA_seqwithmass',
      'madspin_v1_joint']),
    ('all', 'tbar', 'us_lineshape_mtbar_all',
     ['madspin_joint', 'madspin_sequential', 'madspin_seqglobal',
      'PA_joint', 'PA_sequential', 'PA_seqglobal', 'PA_seqwithmass',
      'madspin_v1_joint']),
    ('madspin', 't', 'us_lineshape_mt_madspin',
     ['madspin_joint', 'madspin_sequential', 'madspin_seqglobal',
      'madspin_joint_rep']),
    ('PA', 't', 'us_lineshape_mt_PA',
     ['PA_joint', 'PA_sequential', 'PA_seqglobal', 'PA_seqwithmass',
      'PA_joint_rep']),
    ('nojac', 't', 'us_lineshape_mt_nojac',
     ['PA_joint', 'PAnojac_joint', 'PAnojac_sequential', 'PAnojac_seqglobal',
      'PAnojac_seqwithmass']),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots_userstyle'))
    ap.add_argument('--formats', default='pdf,png')
    args = ap.parse_args()

    formats = [f.strip() for f in args.formats.split(',') if f.strip()]
    os.makedirs(args.out, exist_ok=True)

    d = Data(args.data)
    produced = []
    lines = []
    for fam, tag, name, keys in FAMILIES:
        keys = [k for k in keys if d.has(k, tag)]
        ref = REF if REF in keys else keys[0]
        files, ylim = lineshape_figure(d, keys, tag,
                                       os.path.join(args.out, name), formats,
                                       ref=ref, show_delta=(fam == 'all'))
        produced += files
        lines.append('%-28s ratio ylim %s' % (name, ylim))

    log = os.path.join(args.out, 'us_ratio_limits.txt')
    with open(log, 'w') as fh:
        fh.write("ratio y limits chosen per figure (smallest of the user's\n"
                 "ladder 0.99/0.85/0.75/0.5 that contains every point +- its\n"
                 "sqrt(sumw2) error; nothing is clipped)\n\n"
                 "What is histogrammed on every figure here is a PER-EVENT\n"
                 "quantity: the invariant mass sqrt(E^2-|p|^2) of the status-2\n"
                 "resonance in the decayed LHE, i.e. the virtuality MadSpin\n"
                 "assigned to that top in that event.  It is not a fitted\n"
                 "width and not a pole parameter; the mean and rms of it are\n"
                 "single numbers per cell and live in lineshape_numbers.txt.\n\n")
        fh.write('\n'.join(lines) + '\n')
    produced.append(log)
    # the numbers are style-independent; write them here too so this script is
    # self-sufficient if it is the only one run
    write_numbers(d, args.out)
    produced.append(os.path.join(args.out, 'lineshape_numbers.txt'))

    for p in produced:
        print(p)
    print('\n%d files in %s' % (len(produced), args.out))


if __name__ == '__main__':
    main()
