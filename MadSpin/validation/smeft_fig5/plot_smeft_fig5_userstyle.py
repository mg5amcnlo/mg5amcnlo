#!/usr/bin/env python3
"""The same four variations of Fig. 5 in the user's personal matplotlib style.

``plot_smeft_fig5.py`` draws them in the MG7 paper's style; this is a second,
independent rendering so the figure also matches the rest of the user's own
material.  The physics, the data, the binning, the normalisation and the
numbers are identical -- everything is imported from that module, which is not
modified.

The style conventions are the ones the MadSpin validation series has been using
(``mtt_threshold/plot_mtt_threshold_userstyle.py``, ``mt_lineshape``): stock
rcParams (no usetex, sans serif), ``figsize = (6, 6)``, a ``[3, 1]`` gridspec,
a faint companion step plus ``errorbar(fmt='o', ms=4)`` markers, ``'Ratio'`` on
the lower panel, and the ratio y-limit ladder 0.99 / 0.85 / 0.75 / 0.5.

One deliberate departure: the marker *fill* carries the spinmode.  Filled = spin
correlations kept (``onshell``), open = dropped (``none``), with the colour
still naming the sample.  The user's own scripts distinguish curves by colour
alone, which cannot survive six curves that come in three matched pairs -- and
the pairing is exactly what this figure is about.

Usage::

    python3 plot_smeft_fig5_userstyle.py [--data DIR] [--out DIR] [--nbins N]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Reuse the data loading, rebinning, normalisation checks and error propagation
# rather than rewriting them.  That module sets the MG7 rcParams (serif/usetex)
# at import time, so the defaults are restored immediately afterwards -- the
# user's style sets no rcParams at all.
from plot_smeft_fig5 import (                        # noqa: E402
    Data, VARIATIONS, SAMPLE_PLAIN, MODE_PLAIN, check_normalisation,
    write_curves, write_numbers,
)
mpl.rcParams.update(mpl.rcParamsDefault)
matplotlib.use('Agg')


COLOR = {'eft_int': 'C0', 'sm_lo': 'black', 'sm_nlo': 'C3'}
FILLED = {'onshell': True, 'none': False}

FIGSIZE = (6, 6)
HEIGHT_RATIOS = [3, 1]
HSPACE = 0.05
MS = 4
STEP_ALPHA = 0.55
DPI = 300

RATIO_LADDER = [(0.99, 1.01), (0.85, 1.15), (0.75, 1.25), (0.5, 1.5),
                (0.4, 1.6), (0.25, 1.75)]


def choose_ratio_ylim(series):
    """Smallest rung of the ladder that holds every point plus its error."""
    lo = hi = 1.0
    for r, e in series:
        good = np.isfinite(r) & np.isfinite(e)
        if not good.any():
            continue
        lo = min(lo, float((r[good] - e[good]).min()))
        hi = max(hi, float((r[good] + e[good]).max()))
    for cand in RATIO_LADDER:
        if lo >= cand[0] and hi <= cand[1]:
            return cand
    half = max(1.0 - lo, hi - 1.0)
    return (1.0 - 1.1 * half, 1.0 + 1.1 * half)


def _pi_ticks(ax):
    ax.xaxis.set_major_locator(MultipleLocator(np.pi / 4))
    # The locator can offer ticks just outside the axes; those must render as
    # something rather than raise, so anything off the 0..pi ladder is blank.
    names = {0: '0', 1: r'$\pi/4$', 2: r'$\pi/2$', 3: r'$3\pi/4$', 4: r'$\pi$'}
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _p: names.get(int(round(x / (np.pi / 4))), '')))


def make_figure(d, tag, out):
    spec = VARIATIONS[tag]
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=HEIGHT_RATIOS, hspace=HSPACE)
    ax = fig.add_subplot(gs[0])
    rx = fig.add_subplot(gs[1], sharex=ax)

    for sample, mode in spec['curves']:
        y, ye = d.shape(sample, mode)
        c = COLOR[sample]
        ax.step(d.edges, np.concatenate([y[:1], y]), where='pre',
                color=c, lw=1.0, alpha=STEP_ALPHA, zorder=3)
        ax.errorbar(d.centres, y, yerr=ye, fmt='o', ms=MS, color=c,
                    mfc=(c if FILLED[mode] else 'white'), mew=1.1,
                    label='%s, %s' % (SAMPLE_PLAIN[sample], MODE_PLAIN[mode]),
                    zorder=4)

    ax.set_ylabel(r'$(1/\sigma)\,d\sigma/d\Delta\phi(e^-e^+)$  [1/rad]')
    ax.set_xlim(0.0, np.pi)
    ax.tick_params(labelbottom=False)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.55 * (ymax - ymin))
    ax.legend(loc='upper left', fontsize=8.5, ncol=1)
    ax.set_title(r'$pp\to t\bar{t}$, 13 TeV, $\mu_R=\mu_F=173$ GeV, no cuts; '
                 'all curves normalised to unit area', fontsize=9)
    if any(s == 'eft_int' for s, _ in spec['curves']):
        ax.text(0.985, 0.965,
                'interference at $c_{tG}=-1$, $\\Lambda=1$ TeV,\n'
                'i.e. $-2\\,\\mathrm{Re}(\\mathcal{M}^*_{SM}\\mathcal{M}_{tG})$',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                color=COLOR['eft_int'])

    rx.axhspan(0.9, 1.1, facecolor='C0', alpha=0.10, zorder=0)
    rx.axhline(1.0, color='black', ls='--', lw=0.9, zorder=2)

    series = []
    for sample in spec['ratios']:
        r, e = d.spin_ratio(sample)
        c = COLOR[sample]
        rx.step(d.edges, np.concatenate([r[:1], r]), where='pre',
                color=c, lw=1.0, alpha=STEP_ALPHA, zorder=3)
        rx.errorbar(d.centres, r, yerr=e, fmt='o', ms=MS, color=c, zorder=4)
        series.append((r, e))
    rx.set_ylim(*choose_ratio_ylim(series))
    rx.set_ylabel('Ratio')
    rx.set_xlabel(r'$\Delta\phi(e^-e^+)$ [rad]')
    rx.set_xlim(0.0, np.pi)
    _pi_ticks(rx)
    rx.text(0.015, 0.06, 'onshell / none, each sample against itself',
            transform=rx.transAxes, ha='left', va='bottom', fontsize=7,
            color='0.3')
    rx.text(0.985, 0.94, 'band: $\\pm10\\%$', transform=rx.transAxes,
            ha='right', va='top', fontsize=7, color='C0')

    fig.subplots_adjust(hspace=0.1, left=0.145, right=0.97,
                        bottom=0.11, top=0.93)
    base = os.path.join(out, 'smeft_fig5_%s' % tag)
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=DPI)
    plt.close(fig)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots_userstyle'))
    ap.add_argument('--nbins', type=int, default=20)
    ap.add_argument('--only', default='')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = Data(args.data, nbins=args.nbins)
    rows = check_normalisation(d, fh=open(os.devnull, 'w'))
    bad = [name for name, ok, _ in rows if not ok]
    if bad:
        raise SystemExit('normalisation check failed: %s' % '; '.join(bad))

    tags = [t.strip().upper() for t in args.only.split(',') if t.strip()] \
        or list(VARIATIONS)
    for tag in tags:
        base = make_figure(d, tag, args.out)
        print('wrote %s.pdf / .png' % base)

    write_curves(d, os.path.join(args.out, 'smeft_fig5_curves.npz'))
    with open(os.path.join(args.out, 'numbers.txt'), 'w') as fh:
        write_numbers(d, args.out, fh)
    print('wrote %s' % os.path.join(args.out, 'numbers.txt'))


if __name__ == '__main__':
    main()
