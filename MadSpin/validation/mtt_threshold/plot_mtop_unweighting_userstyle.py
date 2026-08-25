#!/usr/bin/env python3
"""The top-virtuality comparison in the user's personal matplotlib style.

A second, independent rendering of the two figures ``plot_mtop_unweighting.py``
draws in the MG7-paper style.  Same data, same binning, same normalisation,
same ratio denominator, same clip and the same error treatment -- only the
styling differs, and none of the ``m_tt`` scripts is modified.

Everything that is *specific to this observable* -- why the virtuality is the
direct ``Z_k`` question, why both tops share one histogram, why there is no
``truth`` curve, where ``sum w^2`` comes from, why the binning is a
Breit-Wigner grid rather than ``zone_edges()``, and why the paired error and
the quadrature error coincide here when they do not on ``m_tt`` -- is in
``plot_mtop_unweighting`` and in ``numbers.txt``.  This file reimplements none
of it; it imports :class:`MData` and draws.

Two style points differ from the ``m_tt`` user-style figure, both forced by the
observable rather than chosen:

* there is **no black reference curve** in the upper panel.  On the ``m_tt``
  figure that is the truth; ``truth_mtop_*`` was never harvested and the
  decayed LHE files it would come from are gone, so every curve here is a
  scheme and none is drawn as a reference;
* the pane is clipped at **+-10 %**, not ``+-20 %``.  Same criterion, different
  observable; see the module docstring of ``plot_mtop_unweighting``.

Usage::

    python3 plot_mtop_unweighting_userstyle.py [--data DIR] [--out DIR]
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

from plot_mtt_unweighting import SCHEME_PLAIN, ROW_TITLE   # noqa: E402
from plot_mtop_unweighting import (                         # noqa: E402
    MData, RCLIP_LO, RCLIP_HI, append_numbers,
)
from run_mtt_unweighting import CELL_SCHEME                 # noqa: E402

# ``plot_mtt_threshold`` sets the MG7 paper rcParams (serif / usetex) at import
# time; the user's style sets no rcParams at all, so restore the defaults.
mpl.rcParams.update(mpl.rcParamsDefault)
matplotlib.use('Agg')

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
    lo, hi = d.medges[0], d.medges[-1]
    pole = d.mt_pole

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=HEIGHT_RATIOS, hspace=HSPACE)
    ax = fig.add_subplot(gs[0])
    rx = fig.add_subplot(gs[1], sharex=ax)

    for a in (ax, rx):
        a.axvline(pole, color='0.35', lw=1.0, ls='--', zorder=1)

    for key in keys:
        scheme = CELL_SCHEME[key]
        y, cnt = d.mtop_shape(key)
        ye = d.mtop_own_err(key)
        draw = np.where(cnt > 0, y, np.nan)
        ax.step(d.medges, np.concatenate([draw[:1], draw]), where='pre',
                color=COLOR[scheme], lw=1.0, alpha=STEP_ALPHA, zorder=3)
        ax.errorbar(d.mcentres, draw, yerr=np.where(cnt > 0, ye, np.nan),
                    fmt='o', ms=MS, color=COLOR[scheme], zorder=4,
                    label='unweighting = ' + SCHEME_PLAIN[scheme])

    ax.set_yscale('log')
    ax.set_ylabel(r'$(1/N)\,dN/dm(Wb)$  [1/GeV]')
    ax.set_xlim(lo, hi)
    ax.tick_params(labelbottom=False)
    ax.legend(loc='lower right', fontsize=8)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 3.5)
    ax.set_title(r'$pp\to t\bar{t}j$, 13 TeV, LO, $\mu_R=\mu_F=m_t$, '
                 r'BW cut $=%g\,\Gamma_t$ -- %s'
                 % (d.meta.get('bwcutoff', 15.0), ROW_TITLE[row][1]),
                 fontsize=9)
    # Above the peak rather than at the foot of the rule: the pole is mid-axis
    # here and the foot of a pane with a lower-right legend is where the
    # legend is.  Same rule, same label, same grey as the m_tt figure.
    ax.annotate('$m_t$', xy=(pole, 0.90), xycoords=('data', 'axes fraction'),
                xytext=(3, 0), textcoords='offset points', ha='left',
                va='top', fontsize=9, color='0.35')

    ref = d.ref_of(row)
    yref, cref = d.mtop_shape(ref)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_ref = np.where((cref > 0) & (yref > 0),
                           d.mtop_own_err(ref) / yref, np.nan)
    rx.fill_between(d.medges, np.concatenate([(1 - rel_ref)[:1], 1 - rel_ref]),
                    np.concatenate([(1 + rel_ref)[:1], 1 + rel_ref]),
                    step='pre', facecolor='C0', alpha=0.20, edgecolor='none',
                    zorder=1)

    n_off = 0
    for key in [k for k in keys if k != ref]:
        scheme = CELL_SCHEME[key]
        # Each curve's OWN statistical error, matching the band's definition.
        r, own, _diff = d.mtop_ratio(row, key)
        inside = np.isfinite(r) & (r >= RCLIP_LO) & (r <= RCLIP_HI)
        above = np.isfinite(r) & (r > RCLIP_HI)
        below = np.isfinite(r) & (r < RCLIP_LO)
        drawn = np.where(np.isfinite(r), np.clip(r, RCLIP_LO, RCLIP_HI), np.nan)
        rx.step(d.medges, np.concatenate([drawn[:1], drawn]), where='pre',
                color=COLOR[scheme], lw=1.0, alpha=STEP_ALPHA, zorder=3)
        rx.errorbar(d.mcentres, np.where(inside, r, np.nan),
                    yerr=np.where(inside, own, np.nan), fmt='o', ms=MS,
                    color=COLOR[scheme], zorder=4)
        # Off-scale points are drawn ON the boundary as triangles pointing the
        # way they went.  Nothing is dropped and nothing is silently clipped.
        if above.any():
            rx.plot(d.mcentres[above], np.full(above.sum(), RCLIP_HI), '^',
                    color=COLOR[scheme], ms=MS + 2, clip_on=False, zorder=6)
        if below.any():
            rx.plot(d.mcentres[below], np.full(below.sum(), RCLIP_LO), 'v',
                    color=COLOR[scheme], ms=MS + 2, clip_on=False, zorder=6)
        n_off += int(above.sum()) + int(below.sum())

    rx.step(d.medges, np.ones(len(d.medges)), where='pre',
            color=COLOR[CELL_SCHEME[ref]], lw=1.2, zorder=5)

    rx.set_ylim(RCLIP_LO, RCLIP_HI)
    rx.set_yticks([0.90, 0.95, 1.00, 1.05, 1.10])
    # Bottom right rather than top right: this pane is +-10 % and the madspin
    # row's curves reach the top of it.
    rx.text(0.99, 0.06, 'band: stat. error of joint',
            transform=rx.transAxes, ha='right', va='bottom', fontsize=7,
            color='C0')
    rx.set_ylabel('Shape ratio to joint', fontsize=9)
    rx.set_xlabel(r'$m(Wb)$ [GeV]')
    rx.set_xlim(lo, hi)
    fig.subplots_adjust(hspace=0.1, left=0.15, right=0.97,
                        bottom=0.10, top=0.93)
    base = os.path.join(out, 'mtop_unweighting_%s' % row)
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

    d = MData(args.data)
    for row in ('PA', 'madspin'):
        got = make_figure(d, row, args.out)
        if got is None:
            print('%s: no cells on disk, skipped' % row)
        else:
            base, n_off = got
            print('wrote %s.pdf / .png   (%d point%s off the +-%g %% pane, '
                  'drawn on the boundary)'
                  % (base, n_off, '' if n_off == 1 else 's',
                     100 * (RCLIP_HI - 1)))
    append_numbers(d, os.path.join(args.out, 'numbers.txt'))


if __name__ == '__main__':
    main()
