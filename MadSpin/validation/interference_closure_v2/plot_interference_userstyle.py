#!/usr/bin/env python3
"""Interference-closure plots in the user's personal matplotlib style.

This is a second, independent rendering of the same comparisons that
``plot_interference.py`` draws in the MG7-paper style.  It exists so the
figures match the rest of the user's own material; the physics, the data and
the numbers are identical.  ``plot_interference.py`` is not modified.

Everything is regenerated from the committed ``data/histograms.npz`` and
``data/meta.json``.  No MadSpin run, no LHE file and no external parser are
needed -- numpy and matplotlib are the only requirements.

The style conventions reproduced here are taken from the user's own
``plot_hist_with_ratio`` / ``plot_hist_with_ratio_multi`` / ``plot_wb_mass``;
they are written out in ``STYLE_NOTES.md`` next to this file.

Usage::

    python3 plot_interference_userstyle.py [--data DIR] [--out DIR]
"""

import os
import sys
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Reuse the data loading and the statistics of the existing script rather than
# rewriting them: same accessors, same chi2, same ratio error propagation.
# That module sets the MG7 paper rcParams (serif / usetex) at import time, so
# the defaults are restored immediately afterwards -- the user's script sets no
# rcParams at all and therefore draws with stock matplotlib.
_saved_rc = mpl.rcParams.copy()
from plot_interference import (            # noqa: E402
    Data, ratio, chi2, OBS, DIAG, INTER, BLOCK_LABEL,
)
mpl.rcParams.update(mpl.rcParamsDefault)
matplotlib.use('Agg')
del _saved_rc

ALL9 = DIAG + INTER


# --- the user's style -----------------------------------------------------
# Colours: their sample dicts use the stock C0/C1/C2/C3 cycle, with the
# reference drawn in black as a plain step.
C_UNPOL = 'black'      # reference: step only, and the dashed ratio line
C_SUM4 = 'C0'          # the four diagonal blocks alone
C_SUM9 = 'C3'          # all nine blocks
C_INT = 'C2'           # the five interference blocks summed
C_BLOCKS = ['C4', 'C5', 'C6', 'C8']    # the four diagonal blocks individually

FIGSIZE = (6, 6)                       # their plot_hist_with_ratio*
HEIGHT_RATIOS = [3, 1]
HSPACE = 0.05                          # at gridspec time
ADJUST = dict(hspace=0.1, left=0.15, right=0.97, bottom=0.12, top=0.96)
MS = 4                                 # errorbar marker size
STEP_ALPHA = 0.55                      # the faint companion step
OPEN = dict(markerfacecolor='none', markeredgewidth=1.2)
DPI = 300

# Their ratio-limit vocabulary, smallest first.  The default in the function
# signature is (0.99, 1.01), but every real call site in their script picks one
# of these three.  We pick the smallest that actually contains the data.
RATIO_LADDER = [(0.99, 1.01), (0.85, 1.15), (0.75, 1.25), (0.5, 1.5)]

# The data are cross sections in picobarns, not event counts, so the y label
# departs from their literal 'Events'.
YLABEL = r'$\mathrm{d}\sigma$ per bin [pb]'


def choose_ratio_ylim(series):
    """Smallest limit from the user's ladder that holds every point +- error.

    ``series`` is a list of (ratio, ratio_error) arrays.  Nothing is clipped:
    if the deviation escapes even the widest of their limits, the limits are
    widened to fit it.  The 4-block ratio is the whole point of these plots.
    """
    lo = 1.0
    hi = 1.0
    for r, re in series:
        good = np.isfinite(r) & np.isfinite(re)
        if not good.any():
            continue
        lo = min(lo, float((r[good] - re[good]).min()))
        hi = max(hi, float((r[good] + re[good]).max()))
    for cand in RATIO_LADDER:
        if lo >= cand[0] and hi <= cand[1]:
            return cand
    # Wider than anything they use -- C_nn is: keep going rather than cut the
    # effect off, rounded up to a clean 0.1 so the axis still reads like theirs.
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
    """Their non-reference mark: errorbar 'o' plus a faint companion step."""
    style = dict(OPEN, markeredgecolor=color) if open_markers else {}
    ax.errorbar(ctr, y, yerr=ye, fmt='o', ms=MS, label=label, color=color,
                **style)
    ax.step(edges, np.append(y, y[-1]), where='post', color=color,
            alpha=STEP_ALPHA)


def _reference(ax, edges, y, color, label):
    """Their reference mark: a plain step, no markers, no error bars."""
    ax.step(edges, np.append(y, y[-1]), where='post', color=color, label=label)


def _finish(fig, ax_main, ax_ratio, xlabel, ylim, outbase, formats,
            headroom=1.18):
    ax_main.set_ylabel(YLABEL)
    # The user's script just calls .legend() (loc='best').  Keep that, but open
    # some space at the top so 'best' lands on empty canvas instead of on the
    # distribution -- these histograms are peaked and fill their frame.
    lo, hi = ax_main.get_ylim()
    ax_main.set_ylim(lo, lo + (hi - lo) * headroom)
    ax_main.legend(fontsize=9, borderpad=0.3, labelspacing=0.3,
                   handletextpad=0.4)
    ax_main.tick_params(labelbottom=False)

    ax_ratio.axhline(1, linestyle='--', color=C_UNPOL, zorder=1)
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_ylabel('Ratio')
    ax_ratio.set_ylim(*ylim)

    fig.subplots_adjust(**ADJUST)
    written = []
    for ext in formats:
        path = '%s.%s' % (outbase, ext)
        plt.savefig(path, bbox_inches='tight', dpi=DPI)
        written.append(path)
    plt.close(fig)
    return written


# --- the two figure families ----------------------------------------------
def closure_figure(d, key, label, out, formats):
    """Unpolarised reference vs the 4-block and the 9-block sum."""
    edges = d.bins(key)
    ctr = 0.5 * (edges[1:] + edges[:-1])

    u, ue = d.h(['unpol'], key)
    s4, s4e = d.h(DIAG, key)
    s9, s9e = d.h(ALL9, key)

    c4, ndf = chi2(s4, s4e, u, ue)
    c9, _ = chi2(s9, s9e, u, ue)

    r4, r4e = ratio(s4, s4e, u, ue)
    r9, r9e = ratio(s9, s9e, u, ue)
    ylim = choose_ratio_ylim([(r4, r4e), (r9, r9e)])

    fig, ax_main, ax_ratio = _panels()
    _reference(ax_main, edges, u, C_UNPOL, 'unpolarised (reference)')
    _marks(ax_main, ctr, edges, s4, s4e, C_SUM4,
           r'4 diagonal blocks  ($\chi^2 = %.1f$, %d bins)' % (c4, ndf))
    _marks(ax_main, ctr, edges, s9, s9e, C_SUM9,
           r'all 9 blocks  ($\chi^2 = %.1f$, %d bins)' % (c9, ndf),
           open_markers=True)

    for (r, re, color, opened) in ((r4, r4e, C_SUM4, False),
                                   (r9, r9e, C_SUM9, True)):
        good = np.isfinite(r)
        style = dict(OPEN, markeredgecolor=color) if opened else {}
        ax_ratio.errorbar(ctr[good], r[good], yerr=re[good], fmt='o', ms=MS,
                          color=color, zorder=2, **style)

    return _finish(fig, ax_main, ax_ratio, label, ylim, out, formats), ylim


def blocks_figure(d, key, label, out, formats):
    """The individual diagonal blocks and the interference sum they miss."""
    edges = d.bins(key)
    ctr = 0.5 * (edges[1:] + edges[:-1])

    u, ue = d.h(['unpol'], key)
    s4, s4e = d.h(DIAG, key)
    s9, s9e = d.h(ALL9, key)
    si, sie = d.h(INTER, key)

    c4, ndf = chi2(s4, s4e, u, ue)
    c9, _ = chi2(s9, s9e, u, ue)

    r4, r4e = ratio(s4, s4e, u, ue)
    r9, r9e = ratio(s9, s9e, u, ue)
    ylim = choose_ratio_ylim([(r4, r4e), (r9, r9e)])

    fig, ax_main, ax_ratio = _panels()
    _reference(ax_main, edges, u, C_UNPOL, 'unpolarised (reference)')
    for tag, color in zip(DIAG, C_BLOCKS):
        y, _ye = d.h([tag], key)
        ax_main.step(edges, np.append(y, y[-1]), where='post', color=color,
                     alpha=STEP_ALPHA, linewidth=1.0, label=BLOCK_LABEL[tag])
    _marks(ax_main, ctr, edges, si, sie, C_INT,
           'sum of the 5 interference blocks')
    _marks(ax_main, ctr, edges, s4, s4e, C_SUM4, '4 diagonal blocks')
    _marks(ax_main, ctr, edges, s9, s9e, C_SUM9, 'all 9 blocks',
           open_markers=True)
    ax_main.axhline(0, linestyle=':', color='grey', linewidth=0.8, zorder=0)

    for (r, re, color, opened, lab) in (
            (r4, r4e, C_SUM4, False, r'4 blocks  ($\chi^2=%.1f$)' % c4),
            (r9, r9e, C_SUM9, True, r'9 blocks  ($\chi^2=%.1f$)' % c9)):
        good = np.isfinite(r)
        style = dict(OPEN, markeredgecolor=color) if opened else {}
        ax_ratio.errorbar(ctr[good], r[good], yerr=re[good], fmt='o', ms=MS,
                          color=color, label=lab, zorder=2, **style)

    # seven legend entries here, so more room is needed than on the closure plot
    return _finish(fig, ax_main, ax_ratio, label, ylim, out, formats,
                   headroom=1.45), ylim


BLOCKS_FOR = ['cnn', 'cos_phi', 'dphi_lab', 'pt_t', 'm_tt']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots_userstyle'))
    ap.add_argument('--formats', default='png,pdf')
    args = ap.parse_args()

    formats = [f.strip() for f in args.formats.split(',') if f.strip()]
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    d = Data(args.data)
    produced = []
    lines = []
    for key, label, _kind in OBS:
        base = os.path.join(args.out, 'us_closure_%s' % key)
        files, ylim = closure_figure(d, key, label, base, formats)
        produced += files
        lines.append('us_closure_%-9s ratio ylim %s' % (key, ylim))
        if key in BLOCKS_FOR:
            base = os.path.join(args.out, 'us_blocks_%s' % key)
            files, ylim = blocks_figure(d, key, label, base, formats)
            produced += files
            lines.append('us_blocks_%-10s ratio ylim %s' % (key, ylim))

    log = os.path.join(args.out, 'us_ratio_limits.txt')
    with open(log, 'w') as fh:
        fh.write('ratio y limits chosen per figure (smallest of the user\'s\n'
                 'ladder 0.99/0.85/0.75/0.5 that contains every point +- its\n'
                 'sqrt(sumw2) error; nothing is clipped)\n\n')
        fh.write('\n'.join(lines) + '\n')
    produced.append(log)

    for p in produced:
        print(p)
    print('\n%d files in %s' % (len(produced), args.out))


if __name__ == '__main__':
    main()
