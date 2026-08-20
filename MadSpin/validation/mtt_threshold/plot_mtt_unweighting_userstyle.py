#!/usr/bin/env python3
"""The unweighting-scheme comparison in the user's personal matplotlib style.

A second, independent rendering of the two figures ``plot_mtt_unweighting.py``
draws in the MG7-paper style.  Same data, same numbers, same clipping; only the
styling differs, and neither of the first study's scripts is modified.

The style conventions are the ones ``plot_mtt_threshold_userstyle.py`` already
follows: stock rcParams (no usetex, sans serif), figsize (6, 6), a [3, 1]
gridspec, a plain step for the reference and ``errorbar(fmt='o', ms=4)`` plus a
faint companion step for everything else, 'Ratio' on the lower panel.  The main
panel is log-y for the same reason as there -- the tail is the point of the
figure and a linear axis flattens it.

The ratio ladder is not used here.  The brief for this figure fixes the ratio
pane at **+-20 %**, so the ladder's job (pick the smallest rung that holds every
point) is done for it; points outside are drawn on the boundary as triangles and
counted in the caption, never silently cut.

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

from plot_mtt_threshold import ratio                    # noqa: E402
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

    den, dene, dcnt = d.density(REF)
    ax.step(d.edges, np.concatenate([den[:1], den]), where='pre',
            color=C_REF, lw=1.2, zorder=5,
            label='truth: pp -> tt~ j, t -> W+ b (off shell)')

    for key in keys:
        scheme = CELL_SCHEME[key]
        y, ye, cnt = d.density(key)
        draw = np.where(cnt > 0, y, np.nan)
        ax.step(d.edges, np.concatenate([draw[:1], draw]), where='pre',
                color=COLOR[scheme], lw=1.0, alpha=STEP_ALPHA, zorder=3)
        ax.errorbar(d.centres, draw, yerr=np.where(cnt > 0, ye, np.nan),
                    fmt='o', ms=MS, color=COLOR[scheme], zorder=4,
                    label='unweighting = ' + SCHEME_PLAIN[scheme])

    ax.set_yscale('log')
    ax.set_ylabel(r'$d\sigma/dm_{t\bar{t}}$  [pb/GeV]')
    ax.set_xlim(lo, hi)
    ax.tick_params(labelbottom=False)
    ax.legend(loc='lower right', fontsize=8)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 30)
    n_cell = int(d.meta['runs'][keys[0]]['nevents'])
    ax.set_title(r'$pp\to t\bar{t}j$, 13 TeV, LO, $\mu_R=\mu_F=m_t$, '
                 r'BW cut $=%g\,\Gamma_t$ -- %s'
                 % (d.meta.get('bwcutoff', 15.0), ROW_TITLE[row][1]),
                 fontsize=10)
    ax.text(0.02, 0.965,
            'the accept/reject scheme is the only thing that changes between\n'
            'the coloured curves: same %s production events, same seed.\n'
            'joint and sequential_global_retry do not read the tabulated\n'
            '$Z_k$; sequential does, and its residual bias is exactly '
            r'$\hat{Z}/Z$.'
            % '{:,}'.format(n_cell).replace(',', ' '),
            transform=ax.transAxes, ha='left', va='top', fontsize=7.5,
            color='0.25')
    ax.annotate('$2m_t$', xy=(two_mt, 0.02), xycoords=('data', 'axes fraction'),
                xytext=(3, 0), textcoords='offset points', ha='left',
                va='bottom', fontsize=9, color='0.35')

    rx.axhspan(0.95, 1.05, facecolor='C0', alpha=0.16, zorder=0)
    rx.axhspan(0.9, 1.1, facecolor='C0', alpha=0.10, zorder=0)
    rx.axhline(1.0, color=C_REF, ls='--', lw=0.9, zorder=2)

    n_off = 0
    for key in keys:
        scheme = CELL_SCHEME[key]
        y, ye, cnt = d.density(key)
        r, re = ratio(y, ye, den, dene)
        r = np.where((cnt == 0) | (dcnt == 0), np.nan, r)
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

    rx.set_ylim(RCLIP_LO, RCLIP_HI)
    rx.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    rx.text(0.99, 0.92, 'bands: $\\pm5\\%$, $\\pm10\\%$',
            transform=rx.transAxes, ha='right', va='top', fontsize=7,
            color='C0')
    rx.set_ylabel('Ratio (clipped $\\pm20\\%$)')
    rx.set_xlabel(r'$m_{t\bar{t}}$ [GeV]   '
                  r'(per-event $m$ of $(W^+b)+(W^-\bar{b})$)')
    rx.set_xlim(lo, hi)
    # Under the figure, not inside the pane: the pane is exactly where the
    # off-scale points are.
    fig.text(0.5, 0.012,
             '%d point%s lie outside +-20%% and are drawn as triangles on the '
             'boundary, pointing the way they went;\nunclipped values in '
             'numbers.txt.' % (n_off, '' if n_off == 1 else 's'),
             ha='center', va='bottom', fontsize=7, color='0.35')

    fig.subplots_adjust(hspace=0.1, left=0.15, right=0.97,
                        bottom=0.165, top=0.91)
    base = os.path.join(out, 'mtt_unweighting_%s' % row)
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=DPI)
    plt.close(fig)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out',
                    default=os.path.join(_HERE, 'plots_unweighting_userstyle'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = UData(args.data)
    for row in ('PA', 'madspin'):
        base = make_figure(d, row, args.out)
        if base is None:
            print('%s: no cells on disk, skipped' % row)
        else:
            print('wrote %s.pdf / .png' % base)

    with open(os.path.join(args.out, 'numbers.txt'), 'w') as fh:
        write_numbers(d, args.out, fh)


if __name__ == '__main__':
    main()
