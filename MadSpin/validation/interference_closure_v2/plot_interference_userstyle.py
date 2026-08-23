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
# rewriting them: same accessors, same ratio error propagation.  The chi2 the
# closure test is judged on lives in RESULTS.md and plots/closure_numbers.txt;
# it is deliberately not written onto any figure, in either style.
# That module sets the MG7 paper rcParams (serif / usetex) at import time, so
# the defaults are restored immediately afterwards -- the user's script sets no
# rcParams at all and therefore draws with stock matplotlib.
# ``OBS`` also carries the axis labels, so both styles share one wording and
# one convention: what is drawn is always a per-event quantity, and where that
# quantity's mean is a spin-correlation coefficient the label says so with
# '(mean -> C_ij)' instead of naming the curve after the coefficient.
_saved_rc = mpl.rcParams.copy()
from plot_interference import (            # noqa: E402
    Data, ratio, OBS, DIAG, INTER, BLOCK_LABEL,
    signed_stack, stack_closure_check, STACK_NOTE,
    STACKED_CLOSURE_FOR, STACKED_BLOCKS_FOR,
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
# line style encodes the polarisation of the TOP, i.e. the first block index
BLOCK_LS = {'pp': 'solid', 'pm': 'solid', 'mp': 'dashdot', 'mm': 'dashdot'}

# One stock-cycle colour per block for the stacked figures.  The four diagonal
# ones keep the colours they already have above, and (I,I) keeps C2, the
# colour the interference sum is drawn in everywhere else in this script, so
# nothing changes meaning between the stacked and the unstacked rendering.
US_BLOCK_COLOR = {'pp': 'C4', 'pm': 'C5', 'mp': 'C6', 'mm': 'C8',
                  'i_dp': 'C1', 'i_dm': 'C7', 'dp_i': 'C9', 'dm_i': 'C0',
                  'ii': 'C2'}

# Observables whose closure figure also shows the four diagonal blocks.
AUGMENT_WITH_DIAG_BLOCKS = ('cos_k_p',)

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


def _panels(height_ratios=None):
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=height_ratios or HEIGHT_RATIOS,
                          hspace=HSPACE)
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
            headroom=1.18, legend_kw=None):
    ax_main.set_ylabel(YLABEL)
    # The user's script just calls .legend() (loc='best').  Keep that, but open
    # some space at the top so 'best' lands on empty canvas instead of on the
    # distribution -- these histograms are peaked and fill their frame.
    # A stacked figure fills its frame completely, so 'best' has nowhere good
    # to go and those call sites pass an explicit loc/ncol instead.
    lo, hi = ax_main.get_ylim()
    ax_main.set_ylim(lo, lo + (hi - lo) * headroom)
    ax_main.legend(**dict(dict(fontsize=9, borderpad=0.3, labelspacing=0.3,
                               handletextpad=0.4), **(legend_kw or {})))
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
def closure_figure(d, key, label, out, formats, show_diag_blocks=False):
    """Unpolarised reference vs the 4-block and the 9-block sum.

    With ``show_diag_blocks`` the upper panel also carries the four diagonal
    blocks one by one, each at its own cross section: nothing is rescaled, so
    the four add up bin by bin to the '4 diagonal blocks' series, which is the
    sum the ratio panel tests.  Line style encodes the polarisation of the top,
    the first block index: solid for D+, dash-dot for D-.
    """
    edges = d.bins(key)
    ctr = 0.5 * (edges[1:] + edges[:-1])

    u, ue = d.h(['unpol'], key)
    s4, s4e = d.h(DIAG, key)
    s9, s9e = d.h(ALL9, key)

    r4, r4e = ratio(s4, s4e, u, ue)
    r9, r9e = ratio(s9, s9e, u, ue)
    ylim = choose_ratio_ylim([(r4, r4e), (r9, r9e)])

    fig, ax_main, ax_ratio = _panels()
    _reference(ax_main, edges, u, C_UNPOL, 'unpolarised (reference)')
    if show_diag_blocks:
        for tag, color in zip(DIAG, C_BLOCKS):
            y, _ye = d.h([tag], key)
            ax_main.step(edges, np.append(y, y[-1]), where='post',
                         color=color, alpha=STEP_ALPHA, linewidth=1.2,
                         linestyle=BLOCK_LS[tag], zorder=1,
                         label=BLOCK_LABEL[tag])
    _marks(ax_main, ctr, edges, s4, s4e, C_SUM4, '4 diagonal blocks')
    _marks(ax_main, ctr, edges, s9, s9e, C_SUM9, 'all 9 blocks',
           open_markers=True)

    for (r, re, color, opened) in ((r4, r4e, C_SUM4, False),
                                   (r9, r9e, C_SUM9, True)):
        good = np.isfinite(r)
        style = dict(OPEN, markeredgecolor=color) if opened else {}
        ax_ratio.errorbar(ctr[good], r[good], yerr=re[good], fmt='o', ms=MS,
                          color=color, zorder=2, **style)

    return _finish(fig, ax_main, ax_ratio, label, ylim, out, formats,
                   headroom=1.55 if show_diag_blocks else 1.18), ylim


def blocks_figure(d, key, label, out, formats):
    """The individual diagonal blocks and the interference sum they miss."""
    edges = d.bins(key)
    ctr = 0.5 * (edges[1:] + edges[:-1])

    u, ue = d.h(['unpol'], key)
    s4, s4e = d.h(DIAG, key)
    s9, s9e = d.h(ALL9, key)
    si, sie = d.h(INTER, key)

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

    for (r, re, color, opened) in ((r4, r4e, C_SUM4, False),
                                   (r9, r9e, C_SUM9, True)):
        good = np.isfinite(r)
        style = dict(OPEN, markeredgecolor=color) if opened else {}
        ax_ratio.errorbar(ctr[good], r[good], yerr=re[good], fmt='o', ms=MS,
                          color=color, zorder=2, **style)

    # seven legend entries here, so more room is needed than on the closure plot
    return _finish(fig, ax_main, ax_ratio, label, ylim, out, formats,
                   headroom=1.45), ylim


# --- the stacked variants -------------------------------------------------
# Same sign convention as the MG7-style script, and the same helper does the
# arithmetic, so the two renderings cannot drift apart: in each bin positive
# contributions are stacked up from zero and negative ones down from zero, and
# the total is the explicit line, not the envelope.  See ``signed_stack``.
#
# The user's own script has no stacked plot to copy a convention from -- none
# of their observables goes negative -- so the marks here are the closest
# thing it does have: filled regions in the stock colour cycle, the reference
# still a plain black step, and the ratio panel untouched.
def _stack_bands(ax, edges, comps, colors, labels):
    bands, pos_top, neg_bot, net = signed_stack(comps)
    for (lo, hi), color, lab in zip(bands, colors, labels):
        ax.fill_between(edges, np.append(lo, lo[-1]), np.append(hi, hi[-1]),
                        step='post', facecolor=color, edgecolor='white',
                        linewidth=0.35, label=lab, zorder=1)
    return pos_top, neg_bot, net


def _stack_note(ax):
    ax.set_title(STACK_NOTE, fontsize=7, color='0.35', loc='left', pad=4)


def stacked_closure_figure(d, key, label, out, formats):
    """The 'all 9 blocks' total, drawn as the stack of what makes it up."""
    edges = d.bins(key)
    ctr = 0.5 * (edges[1:] + edges[:-1])

    u, ue = d.h(['unpol'], key)
    s4, s4e = d.h(DIAG, key)
    s9, s9e = d.h(ALL9, key)
    si, _sie = d.h(INTER, key)

    if key in AUGMENT_WITH_DIAG_BLOCKS:
        comps = [d.h([t], key)[0] for t in DIAG] + [si]
        colors = list(C_BLOCKS) + [C_INT]
        labels = [BLOCK_LABEL[t] for t in DIAG] + ['5 interference blocks']
    else:
        comps = [s4, si]
        colors = [C_SUM4, C_INT]
        labels = ['4 diagonal blocks', '5 interference blocks']

    r4, r4e = ratio(s4, s4e, u, ue)
    r9, r9e = ratio(s9, s9e, u, ue)
    ylim = choose_ratio_ylim([(r4, r4e), (r9, r9e)])

    fig, ax_main, ax_ratio = _panels()
    pos_top, neg_bot, net = _stack_bands(ax_main, edges, comps, colors, labels)
    ax_main.axhline(0, linestyle=':', color='grey', linewidth=0.8, zorder=0)
    ax_main.step(edges, np.append(net, net[-1]), where='post', color=C_SUM9,
                 linewidth=1.4, zorder=3, label='net: all 9 blocks')
    _reference(ax_main, edges, u, C_UNPOL, 'unpolarised (reference)')
    ax_main.set_ylim(min(float(neg_bot.min()) * 1.3, 0.0),
                     float(pos_top.max()) * 1.02)
    _stack_note(ax_main)

    for (r, re, color, opened) in ((r4, r4e, C_SUM4, False),
                                   (r9, r9e, C_SUM9, True)):
        good = np.isfinite(r)
        style = dict(OPEN, markeredgecolor=color) if opened else {}
        ax_ratio.errorbar(ctr[good], r[good], yerr=re[good], fmt='o', ms=MS,
                          color=color, zorder=2, **style)

    many = len(comps) > 2
    files = _finish(fig, ax_main, ax_ratio, label, ylim, out, formats,
                    headroom=1.42 if many else 1.30,
                    legend_kw=dict(loc='upper center', ncol=3 if many else 2,
                                   fontsize=8 if many else 9,
                                   columnspacing=1.1))
    return files, stack_closure_check(comps, s9)


def stacked_blocks_figure(d, key, label, out, formats):
    """Nine blocks stacked above, the five interference ones stacked below.

    The lower panel replaces the ratio panel here: the interference blocks are
    three orders of magnitude below the diagonal ones in integrated rate, and
    their own scale is the only one on which the sign convention is visible.
    """
    edges = d.bins(key)
    u, _ue = d.h(['unpol'], key)
    s9, _s9e = d.h(ALL9, key)
    si, _sie = d.h(INTER, key)

    comps9 = [d.h([t], key)[0] for t in ALL9]
    compsI = [d.h([t], key)[0] for t in INTER]

    fig, ax_main, ax_low = _panels(height_ratios=[2.2, 1.7])

    pos9, neg9, net9 = _stack_bands(
        ax_main, edges, comps9, [US_BLOCK_COLOR[t] for t in ALL9],
        [BLOCK_LABEL[t] for t in ALL9])
    ax_main.axhline(0, linestyle=':', color='grey', linewidth=0.8, zorder=0)
    ax_main.step(edges, np.append(net9, net9[-1]), where='post', color=C_SUM9,
                 linewidth=1.4, zorder=3, label='net: all 9')
    _reference(ax_main, edges, u, C_UNPOL, 'unpolarised (reference)')
    ax_main.set_ylim(min(float(neg9.min()) * 1.3, 0.0),
                     float(pos9.max()) * 1.02)
    ax_main.set_ylabel(YLABEL)
    _stack_note(ax_main)

    posI, negI, netI = _stack_bands(
        ax_low, edges, compsI, [US_BLOCK_COLOR[t] for t in INTER],
        [BLOCK_LABEL[t] for t in INTER])
    ax_low.axhline(0, linestyle=':', color='grey', linewidth=0.8, zorder=0)
    ax_low.step(edges, np.append(netI, netI[-1]), where='post', color=C_UNPOL,
                linewidth=1.2, zorder=3)
    span = max(float(posI.max()) - float(negI.min()), 1e-9)
    ax_low.set_ylim(float(negI.min()) - 0.12 * span,
                    float(posI.max()) + 0.12 * span)
    ax_low.set_ylabel('interference [pb]')
    ax_low.set_xlabel(label)

    # _finish is for the ratio-panel layout; this figure has none, so the
    # saving and the top-panel legend are done here with the same conventions
    lo, hi = ax_main.get_ylim()
    ax_main.set_ylim(lo, lo + (hi - lo) * 1.55)
    ax_main.legend(fontsize=7, borderpad=0.3, labelspacing=0.3,
                   handletextpad=0.4, ncol=3)
    ax_main.tick_params(labelbottom=False)
    fig.subplots_adjust(**ADJUST)
    written = []
    for ext in formats:
        path = '%s.%s' % (out, ext)
        plt.savefig(path, bbox_inches='tight', dpi=DPI)
        written.append(path)
    plt.close(fig)
    return written, (stack_closure_check(comps9, s9),
                     stack_closure_check(compsI, si))


def stacked_main(d, out, formats):
    """The stacked variants, written next to the unstacked ones."""
    lab = dict((k, l) for k, l, _ in OBS)
    produced = []
    rows = []
    for key in STACKED_CLOSURE_FOR:
        base = os.path.join(out, 'us_closure_%s_stacked' % key)
        files, chk = stacked_closure_figure(d, key, lab[key], base, formats)
        produced += files
        rows.append(('us_closure_%s_stacked' % key, chk))
    for key in STACKED_BLOCKS_FOR:
        base = os.path.join(out, 'us_blocks_%s_stacked' % key)
        files, (c9, ci) = stacked_blocks_figure(d, key, lab[key], base,
                                                formats)
        produced += files
        rows.append(('us_blocks_%s_stacked (9)' % key, c9))
        rows.append(('us_blocks_%s_stacked (5 I)' % key, ci))

    log = os.path.join(out, 'us_stacked_numbers.txt')
    with open(log, 'w') as fh:
        fh.write(
            'Stacked renderings, user style.  Same data, same binning\n'
            'and the same arithmetic as the MG7-style stacked figures\n'
            'in ../plots: both call signed_stack() in\n'
            'plot_interference.py, so the two renderings cannot\n'
            'disagree.\n\n'
            'SIGN CONVENTION (also written on every stacked figure):\n'
            '  in each bin, contributions with POSITIVE content are stacked\n'
            '  upward from zero and contributions with NEGATIVE content are\n'
            '  stacked downward from zero, so a band below the axis is a\n'
            '  subtraction and reads as one.  The top of the stack is\n'
            '  therefore the sum of the POSITIVE contributions only and is\n'
            '  NOT the total.  The total is the algebraic sum of the two\n'
            '  piles and is drawn as an explicit line, labelled "net: ...".\n'
            '  Read the line, not the envelope.\n\n'
            'It is needed because the four diagonal blocks are positive in\n'
            'every bin of every observable here while the five interference\n'
            'blocks are not: (I,I) is negative over about half the range of\n'
            'C_nn and of dphi_lab; that sign change is the whole effect.\n\n'
            'Closure: the net of the stacked bands against the total curve\n'
            'drawn on the same panel, bin by bin.  Same addition regrouped,\n'
            'so the residual is floating point only (eps = 2.2e-16).\n\n')
        fh.write('  %-34s %13s %13s\n'
                 % ('figure / panel', 'max|net-tot|', 'relative'))
        for name, (dev, rel) in rows:
            fh.write('  %-34s %13.3e %13.3e\n' % (name, dev, rel))
    produced.append(log)
    return produced


BLOCKS_FOR = ['cnn', 'cos_phi', 'dphi_lab', 'pt_t', 'm_tt']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots_userstyle'))
    ap.add_argument('--formats', default='png,pdf')
    ap.add_argument('--stacked-only', action='store_true',
                    help='write only the stacked variants, leaving the '
                         'existing figures untouched')
    ap.add_argument('--no-stacked', action='store_true',
                    help='skip the stacked variants')
    args = ap.parse_args()

    formats = [f.strip() for f in args.formats.split(',') if f.strip()]
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    d = Data(args.data)
    if args.stacked_only:
        for p in stacked_main(d, args.out, formats):
            print(p)
        return

    produced = []
    lines = []
    for key, label, _kind in OBS:
        base = os.path.join(args.out, 'us_closure_%s' % key)
        if key in AUGMENT_WITH_DIAG_BLOCKS:
            # the figure the write-up shows carries the four diagonal blocks;
            # the earlier, plain rendering is kept next to it under *_plain
            files, _ = closure_figure(d, key, label, base + '_plain', formats)
            produced += files
            files, ylim = closure_figure(d, key, label, base, formats,
                                         show_diag_blocks=True)
        else:
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
                 'sqrt(sumw2) error; nothing is clipped)\n\n'
                 'ckk, cnn and crr are the PER-EVENT products\n'
                 'c_ij = cos(theta^i_l+) cos(theta^j_l-); every figure here is\n'
                 'a histogram of that product.  The spin-correlation\n'
                 'coefficient C_ij is its MEAN, one number per sample, which is\n'
                 'why the axis labels read "mean -> C_ij" and not "C_ij".  A\n'
                 'block with C_ij = 0 still has a cross section and a non-zero\n'
                 'histogram; what vanishes for it is the first moment.\n\n')
        fh.write('\n'.join(lines) + '\n')
    produced.append(log)

    if not args.no_stacked:
        produced += stacked_main(d, args.out, formats)

    for p in produced:
        print(p)
    print('\n%d files in %s' % (len(produced), args.out))


if __name__ == '__main__':
    main()
