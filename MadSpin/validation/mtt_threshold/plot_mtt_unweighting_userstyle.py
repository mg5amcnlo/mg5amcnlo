#!/usr/bin/env python3
"""The unweighting-scheme comparison in the user's personal matplotlib style.

A second, independent rendering of the two figures ``plot_mtt_unweighting.py``
draws in the MG7-paper style.  Same data, same numbers, same clipping, same
ratio denominator and the same error treatment -- only the styling differs, and
neither of the first study's scripts is modified.

The lower panel divides by the row's ``joint`` cell, not by the truth, and the
truth is not drawn in it.  ``joint`` builds no ``Z_k`` table and
``sequential_global_retry`` cancels ``Z_k`` identically, so those two agreeing
is the null hypothesis and this panel puts it on the line.

The errors drawn are **per-curve statistics**: the band around 1 is ``joint``'s
own, and each coloured curve carries its own, both from
:meth:`UData.own_shape_err` -- the delta-method error of a self-normalised
histogram, since the bin is a subset of the sigma it is divided by.  This file
reimplements neither.

They are **not** the error on the difference between a curve and ``joint``.
The cells decay the same production events, so they fluctuate together; band
and bar in quadrature discards that and comes out too large.
:meth:`UData.paired_ratio` is the correct error on the difference and is what
``numbers.txt`` quotes for the ``sequential`` versus
``sequential_global_retry`` comparison this figure exists for.

The style conventions are the ones ``plot_mtt_threshold_userstyle.py`` already
follows: stock rcParams (no usetex, sans serif), figsize (6, 6), a [3, 1]
gridspec, a plain step for the reference and ``errorbar(fmt='o', ms=4)`` plus a
faint companion step for everything else, 'Ratio' on the lower panel.  The main
panel is log-y for the same reason as there -- the tail is the point of the
figure and a linear axis flattens it.

The ratio ladder is not used here.  The brief for this figure fixes the ratio
pane at **+-20 %**, so the ladder's job (pick the smallest rung that holds every
point) is done for it; points outside are drawn on the boundary as triangles,
never silently cut, and their unclipped values are in ``numbers.txt``.

Usage::

    python3 plot_mtt_unweighting_userstyle.py [--data DIR] [--out DIR]
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

from plot_mtt_unweighting import (                      # noqa: E402
    UData, REF, RCLIP_LO, RCLIP_HI, SCHEME_PLAIN, ROW_TITLE, write_numbers,
)
from run_mtt_unweighting import CELL_SCHEME             # noqa: E402

# ``plot_mtt_threshold`` sets the MG7 paper rcParams (serif / usetex) at import
# time; the user's style sets no rcParams at all, so restore the defaults.
mpl.rcParams.update(mpl.rcParamsDefault)
matplotlib.use('Agg')

C_REF = 'black'
COLOR = {'joint': 'C0', 'sequential': 'C1',
         'sequential_global_retry': 'C2', 'sequential_with_mass': 'C3'}

FIGSIZE = (6, 6)
HEIGHT_RATIOS = [3, 1]
HSPACE = 0.05
MS = 4
STEP_ALPHA = 0.55
DPI = 300


def make_figure(d, row, out):
    keys = d.cells(row)
    if not keys:
        return None
    two_mt = d.two_mt
    lo, hi = d.edges[0], d.edges[-1]

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=HEIGHT_RATIOS, hspace=HSPACE)
    ax = fig.add_subplot(gs[0])
    rx = fig.add_subplot(gs[1], sharex=ax)

    for a in (ax, rx):
        a.axvspan(lo, two_mt, facecolor='0.90', edgecolor='none', zorder=0)
        a.axvline(two_mt, color='0.35', lw=1.0, ls='--', zorder=1)

    # Shape comparison, matching the MG7-style figure: each curve divided by
    # its own total cross section over the full m_tt range.
    den, _dene, _dcnt = d.shape(REF)
    ax.step(d.edges, np.concatenate([den[:1], den]), where='pre',
            color=C_REF, lw=1.2, zorder=5,
            label='truth: pp -> tt~ j, t -> W+ b (off shell)')

    for key in keys:
        scheme = CELL_SCHEME[key]
        y, ye, cnt = d.shape(key)
        draw = np.where(cnt > 0, y, np.nan)
        ax.step(d.edges, np.concatenate([draw[:1], draw]), where='pre',
                color=COLOR[scheme], lw=1.0, alpha=STEP_ALPHA, zorder=3)
        ax.errorbar(d.centres, draw, yerr=np.where(cnt > 0, ye, np.nan),
                    fmt='o', ms=MS, color=COLOR[scheme], zorder=4,
                    label='unweighting = ' + SCHEME_PLAIN[scheme])

    ax.set_yscale('log')
    ax.set_ylabel(r'$(1/\sigma)\,d\sigma/dm_{t\bar{t}}$  [1/GeV]')
    ax.set_xlim(lo, hi)
    ax.tick_params(labelbottom=False)
    ax.legend(loc='lower right', fontsize=8)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 3.5)
    # No prose in the pane, and no sample size either.  Which schemes read the
    # tabulated Z_k, what the sub-threshold region means, and how many events
    # each cell holds are all in numbers.txt and RESULTS.md, where they carry
    # their errors and their measured sensitivity.  The title keeps only the
    # setup, which cannot be read off the curves.
    ax.set_title(r'$pp\to t\bar{t}j$, 13 TeV, LO, $\mu_R=\mu_F=m_t$, '
                 r'BW cut $=%g\,\Gamma_t$ -- %s'
                 % (d.meta.get('bwcutoff', 15.0), ROW_TITLE[row][1]),
                 fontsize=9)
    ax.annotate('$2m_t$', xy=(two_mt, 0.02), xycoords=('data', 'axes fraction'),
                xytext=(3, 0), textcoords='offset points', ha='left',
                va='bottom', fontsize=9, color='0.35')

    # The unity line is ``joint`` itself, drawn in joint's own colour: a black
    # dashed line here would invite it to be read as the truth, which is the
    # one curve deliberately absent from this panel.
    #
    # The band is joint's OWN statistical error, bin by bin -- not a fixed
    # reference rule.  It is the denominator's error, so it belongs on the line
    # and not on the coloured bars.  It is not an agreement band: adding it in
    # quadrature to a bar throws away the correlation between two cells that
    # decayed the same production events, and overstates the difference's
    # error.  numbers.txt has the paired error, which is the one to quote.
    ref = d.ref_of(row)
    yref, _, cref = d.shape(ref)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_ref = np.where((cref > 0) & (yref > 0),
                           d.own_shape_err(ref) / yref, np.nan)
    rx.fill_between(d.edges, np.concatenate([(1 - rel_ref)[:1], 1 - rel_ref]),
                    np.concatenate([(1 + rel_ref)[:1], 1 + rel_ref]),
                    step='pre', facecolor='C0', alpha=0.20, edgecolor='none',
                    zorder=1)

    n_off = 0
    for key in [k for k in keys if k != ref]:
        scheme = CELL_SCHEME[key]
        # Each curve's OWN statistical error, matching the band's definition.
        r, re = d.own_ratio_err(row, key)
        inside = np.isfinite(r) & (r >= RCLIP_LO) & (r <= RCLIP_HI)
        above = np.isfinite(r) & (r > RCLIP_HI)
        below = np.isfinite(r) & (r < RCLIP_LO)
        drawn = np.where(np.isfinite(r), np.clip(r, RCLIP_LO, RCLIP_HI), np.nan)
        rx.step(d.edges, np.concatenate([drawn[:1], drawn]), where='pre',
                color=COLOR[scheme], lw=1.0, alpha=STEP_ALPHA, zorder=3)
        rx.errorbar(d.centres, np.where(inside, r, np.nan),
                    yerr=np.where(inside, re, np.nan), fmt='o', ms=MS,
                    color=COLOR[scheme], zorder=4)
        # Off-scale points are drawn ON the boundary as triangles pointing the
        # way they went.  Nothing is dropped and nothing is silently clipped.
        if above.any():
            rx.plot(d.centres[above], np.full(above.sum(), RCLIP_HI), '^',
                    color=COLOR[scheme], ms=MS + 2, clip_on=False, zorder=6)
        if below.any():
            rx.plot(d.centres[below], np.full(below.sum(), RCLIP_LO), 'v',
                    color=COLOR[scheme], ms=MS + 2, clip_on=False, zorder=6)
        n_off += int(above.sum()) + int(below.sum())

    # The reference, drawn LAST and as a plain step -- which is this style's
    # convention for the reference of a panel, exactly as the truth is drawn in
    # the panel above.  Markers on it would be a row of dots on a line that is
    # 1 by definition, and would sit on top of the points that are being
    # compared with it.
    rx.step(d.edges, np.ones(len(d.edges)), where='pre',
            color=COLOR[CELL_SCHEME[ref]], lw=1.2, zorder=5)

    rx.set_ylim(RCLIP_LO, RCLIP_HI)
    rx.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    rx.text(0.99, 0.92, 'band: stat. error of joint',
            transform=rx.transAxes, ha='right', va='top', fontsize=7,
            color='C0')
    rx.set_ylabel('Shape ratio to joint', fontsize=9)
    # The variable and its unit, nothing else.
    rx.set_xlabel(r'$m_{t\bar{t}}$ [GeV]')
    rx.set_xlim(lo, hi)
    # No footnote under the figure.  The arrows already say where every
    # off-scale point went, and numbers.txt carries each one's value and error;
    # the bottom margin is the x-label's now.
    fig.subplots_adjust(hspace=0.1, left=0.15, right=0.97,
                        bottom=0.10, top=0.93)
    base = os.path.join(out, 'mtt_unweighting_%s' % row)
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=DPI)
    plt.close(fig)
    return base, n_off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out',
                    default=os.path.join(_HERE, 'plots_unweighting_userstyle'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = UData(args.data)
    for row in ('PA', 'madspin'):
        got = make_figure(d, row, args.out)
        if got is None:
            print('%s: no cells on disk, skipped' % row)
        else:
            base, n_off = got
            print('wrote %s.pdf / .png   (%d point%s off the +-20 %% pane, '
                  'drawn on the boundary)'
                  % (base, n_off, '' if n_off == 1 else 's'))

    with open(os.path.join(args.out, 'numbers.txt'), 'w') as fh:
        write_numbers(d, args.out, fh)


if __name__ == '__main__':
    main()
