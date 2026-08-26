#!/usr/bin/env python3
"""The ``2 -> 2`` ``m_tt`` near-threshold comparison in the user's personal
matplotlib style.

A second, independent rendering of the figure ``plot_mtt_threshold.py`` draws in
the MG7-paper style.  It exists so the figure matches the rest of the user's own
material; the physics, the data and the numbers are identical, and it shares the
data loading, the rebinning, the ratio statistics, the structural-emptiness test
and the numeric report with that module -- everything except the drawing.

The style conventions are the ones
``MadSpin/validation/mtt_threshold/plot_mtt_threshold_userstyle.py`` follows for
``p p > t t~ j``, and they are followed here unchanged: stock rcParams (no
usetex, sans serif), figsize (6, 6), a [3, 1] gridspec, a plain step for the
reference and ``errorbar(fmt='o', ms=4)`` plus a faint companion step for
everything else, log-y on the main panel (the tail spans three decades and is
the whole point of the figure), a ratio panel clipped to a fixed +-20 % window
rather than an autoscaled ladder, arrows for measured points that leave the
window, and open markers for exact structural zeros.

One drawing detail differs from the ``t t~ j`` rendering, and it is forced by
the result rather than chosen.  There, one mode -- ``onshell`` -- is
structurally zero below ``2 m_t`` and one open circle per bin says so.  Here all
four modes are, and they sit at exactly the same place, which IS the result.
Four markers cannot be nudged apart inside a 1 GeV bin (about 4 pt of axis), so
they are drawn concentric, smallest on top, and coincidence is what the reader
sees.

Usage::

    python3 plot_mtt_threshold_userstyle.py [--data DIR] [--out DIR]
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

# This directory's module, NOT the t t~ j one: it is the one that has already
# re-pointed the shared figure code at ``p p > t t~`` and that carries this
# study's numeric report.  It in turn imports the t t~ j module for everything
# that is process-independent, so the accessors below are literally the same
# code on both sides of the comparison.  It sets the MG7 paper rcParams
# (serif / usetex) at import time, so the defaults are restored immediately
# afterwards -- the user's style sets no rcParams at all.
from plot_mtt_threshold import (                     # noqa: E402
    Data, ratio, structurally_empty, MODES, REF, CURVES_PLAIN, write_numbers,
    RATIO_CLIP, offscale_arrows, PROC_TEX,
)
mpl.rcParams.update(mpl.rcParamsDefault)
matplotlib.use('Agg')

# The reset above also turns ``text.usetex`` back OFF, and that has one
# consequence worth spelling out: this figure never goes through matplotlib's
# usetex Type1 font-subsetting path, so the minus-eating bug that
# ``plot_mtt_threshold._fix_type1_subset_minus`` works around cannot bite it.
#
# Do NOT "check" this PDF with ``plot_mtt_threshold.check_minus``. That function
# greps the file for ``/minus``, which a non-usetex PDF carries anyway, so it
# reports True whether or not the workaround is active -- with ``NO_MINUS_FIX=1``
# included. A check that passes either way is worse than no check. The
# discriminating one is ``plot_mtt_threshold.py --check-minus`` on the MG7-style
# PDF, which IS rendered with usetex: True with the fix, False without it.
assert not mpl.rcParams['text.usetex'], (
    'the user style renders without usetex; if that ever changes this figure '
    'becomes exposed to the Type1 subsetting bug and needs its own check')


C_REF = 'black'
COLOR = {'madspin': 'C0', 'PA': 'C1', 'onshell': 'C2', 'madspin_v1': 'C3'}

FIGSIZE = (6, 6)
HEIGHT_RATIOS = [3, 1]
HSPACE = 0.05
MS = 4
STEP_ALPHA = 0.55
DPI = 300


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

    # Shape comparison: every curve divided by its OWN total cross section
    # (the sample total over the full m_tt range, not the plotted window), so
    # the rate difference between the truth and MadSpin cancels.  Same accessor
    # as the MG7-style figure, so the two cannot drift apart.
    den, dene, dcnt = d.shape(REF)
    ax.step(d.edges, np.concatenate([den[:1], den]), where='pre',
            color=C_REF, lw=1.2, label=CURVES_PLAIN['truth'], zorder=5)

    for key in MODES:
        y, ye, cnt = d.shape(key)
        draw = np.where(cnt > 0, y, np.nan)
        ax.step(d.edges, np.concatenate([draw[:1], draw]), where='pre',
                color=COLOR[key], lw=1.0, alpha=STEP_ALPHA, zorder=3)
        ax.errorbar(d.centres, draw, yerr=np.where(cnt > 0, ye, np.nan),
                    fmt='o', ms=MS, color=COLOR[key],
                    label=CURVES_PLAIN[key], zorder=4)

    ax.set_yscale('log')
    ax.set_ylabel(r'$(1/\sigma)\,d\sigma/dm_{t\bar{t}}$  [1/GeV]')
    ax.set_xlim(lo, hi)
    ax.tick_params(labelbottom=False)
    ax.legend(loc='lower right', fontsize=8.5)
    # No prose in the pane.  What the shaded region means and why no mode
    # enters it is in numbers.txt and RESULTS.md, where it carries its counts;
    # the title keeps the setup, which cannot be read off the curves.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 3.5)
    ax.set_title(r'$%s$, 13 TeV, LO, $\mu_R=\mu_F=m_t$, '
                 r'BW cut $=%g\,\Gamma_t$'
                 % (PROC_TEX.replace(r'\bar t', r'\bar{t}'),
                    d.meta.get('bwcutoff', 15.0)),
                 fontsize=10)
    ax.annotate('$2m_t$', xy=(two_mt, 0.02), xycoords=('data', 'axes fraction'),
                xytext=(3, 0), textcoords='offset points', ha='left',
                va='bottom', fontsize=9, color='0.35')

    rx.axhspan(0.9, 1.1, facecolor='C0', alpha=0.10, zorder=0)
    rx.axhspan(0.95, 1.05, facecolor='C0', alpha=0.16, zorder=0)
    rx.axhline(1.0, color=C_REF, ls='--', lw=0.9, zorder=2)

    # Fixed +-20 % window, set before anything is drawn so the step lines that
    # run outside it cannot drag the autoscale along.
    rx.set_ylim(*RATIO_CLIP)

    # Which modes are structurally zero below threshold, in drawing order.
    # Concentric rings, largest first: for this process it is all four of them
    # and they coincide exactly, which is the finding.
    struct_of = {key: structurally_empty(d, key) & (dcnt > 0) for key in MODES}
    circled = [key for key in MODES if struct_of[key].any()]

    n_out = 0
    for slot, key in enumerate(MODES):
        # Both sides already self-normalised, so this pane compares SHAPES and
        # sits on 1.  Absolute statements are in numbers.txt and RESULTS.md.
        y, ye, cnt = d.shape(key)
        r, re = ratio(y, ye, den, dene)
        # Same distinction as the MG7-style figure: a structural zero keeps its
        # open marker and gets NO arrow -- it is exactly 0, not a point that ran
        # off the pane.  Any other empty bin is a statement about the sample
        # size and is left as a gap.  A measured ratio outside the window gets
        # an arrow at the boundary it left through.
        struct = struct_of[key]
        stat = (cnt == 0) & (dcnt > 0) & ~struct
        rr = np.where(struct, 0.0, np.where(stat, np.nan, r))
        rx.step(d.edges, np.concatenate([rr[:1], rr]), where='pre',
                color=COLOR[key], lw=1.0, alpha=STEP_ALPHA, zorder=3)
        gone = struct | stat
        rx.errorbar(d.centres, np.where(gone, np.nan, r),
                    yerr=np.where(gone, np.nan, re), fmt='o', ms=MS,
                    color=COLOR[key], zorder=4)
        if struct.any():
            ring = circled.index(key)
            rx.plot(d.centres[struct],
                    np.full(struct.sum(), RATIO_CLIP[0]), 'o',
                    mfc='white', mec=COLOR[key], mew=1.2,
                    ms=MS + 1 + 2.4 * ring,
                    clip_on=False, zorder=8 + (len(circled) - ring))
        nb, na = offscale_arrows(rx, d.centres, np.where(gone, np.nan, r),
                                 COLOR[key], dx=d.widths, slot=slot,
                                 nslot=len(MODES), lw=0.9, scale=8)
        n_out += nb + na

    print('ratio pane clipped to %s: %d point(s) outside it, each drawn as an '
          'arrow at the boundary it left through' % (RATIO_CLIP, n_out))
    print('open circles (exact structural zeros): %s'
          % (', '.join(circled) if circled else 'none'))
    rx.text(0.99, 0.92, 'bands: $\\pm5\\%$, $\\pm10\\%$', transform=rx.transAxes,
            ha='right', va='top', fontsize=7, color='C0')
    # Kept on purpose: a key to two MARKS, not commentary.  The axis label can
    # say the pane is clipped but not that an open circle is an exact
    # structural zero while an arrow is a measured point that left the window.
    # Bottom right is the corner no curve reaches, so it covers no point.
    rx.text(0.99, 0.04,
            'arrow: point outside the pane\n'
            '$\\circ$: exactly 0 (structural)',
            transform=rx.transAxes, ha='right', va='bottom', fontsize=7,
            color='0.30', linespacing=1.3)
    rx.set_ylabel('Shape ratio', fontsize=9)
    # The variable and its unit, nothing else.
    rx.set_xlabel(r'$m_{t\bar{t}}$ [GeV]')
    rx.set_xlim(lo, hi)

    fig.subplots_adjust(hspace=0.1, left=0.15, right=0.97,
                        bottom=0.12, top=0.93)
    base = os.path.join(args.out, 'mtt_threshold')
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=DPI)
    plt.close(fig)
    print('wrote %s.pdf / .png  (usetex=False, so the Type1 minus bug does not '
          'apply to this rendering)' % base)

    with open(os.path.join(args.out, 'numbers.txt'), 'w') as fh:
        write_numbers(d, args.out, fh)


if __name__ == '__main__':
    main()
