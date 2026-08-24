#!/usr/bin/env python3
"""The polarisation-weight figures in the user's personal style.

A second, independent rendering of what ``plot_zz_pol.py`` draws in the MG7
paper style.  The physics, the data, the selection, the binning and the ratio
statistics are the same objects -- all of them come from ``pol_analysis`` --
and only the drawing differs.

The extra spinmode samples are overlaid on the distribution pane only, on the
same rule as the MG7 figures: drawn where the selection leaves enough events to
be a measurement.  With the exclusive ``decay z > e+ e-`` / ``decay z > mu+
mu-`` card both observables clear that threshold for all three modes.

As on the MG7 figures, **nothing but the axis labels, the tick labels and the
legend is written here** -- no title, no integrated values, no per-pane numbers
and no note about a dropped mode.  The two shaded bands behind the sum pane are
+-2 % and +-5 % about 1; they are graphics, they stay, and what they are is
stated in ``numbers.txt`` and RESULTS.md rather than on the canvas.

The conventions are the ones the sibling studies under ``MadSpin/validation/``
follow: stock rcParams (no usetex, sans serif), plain steps for the reference
and ``errorbar(fmt='o', ms=4)`` plus a faint companion step for everything
else, shaded tolerance bands behind the ratio panes, and a dashed reference
line.  The layout is the same three tiers, for the same reason: the sum pane
is the physics and gets the width and the emphasis, the four components get a
2 x 2 breakdown underneath.

The two variants of ``plot_zz_pol.py`` are rendered here too, into
``plots_userstyle/variant_A_madspin_only/`` and
``plots_userstyle/variant_B_shape_ratio/``, alongside the original figures and
never over them.  A -- the same three tiers without the ``onshell`` and ``PA``
curves.  B -- the distribution and then a single pane carrying the
self-normalised shape ratio of both extra modes to ``madspin``.  Which sigma
normalises each curve, and why, is ``pol_analysis.SHAPE_NORM``.

Usage::

    python3 plot_zz_pol_userstyle.py [--data DIR] [--out DIR] [--no-variants]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import pol_analysis as PA                                        # noqa: E402
from plot_zz_pol import _ratio_ylim, _shape_ylim                 # noqa: E402

# ``plot_zz_pol`` imports the MG7-style module, which sets the paper rcParams
# (serif / usetex) at import time; they are reset here because the user's style
# sets no rcParams at all.
mpl.rcParams.update(mpl.rcParamsDefault)
matplotlib.use('Agg')

# The reset also turns ``text.usetex`` back OFF, so these figures never go
# through matplotlib's usetex Type1 subsetting path and cannot be bitten by the
# minus-eating bug ``plot_zz_loopinduced._fix_type1_subset_minus`` works around.
# Do NOT "check" these PDFs with ``check_minus``: a non-usetex PDF carries
# ``/minus`` whether or not the workaround ran, so the check would pass either
# way, which is worse than no check.  The discriminating one is
# ``plot_zz_pol.py --check-minus`` on the MG7-style PDFs.
assert not mpl.rcParams['text.usetex'], (
    'the user style renders without usetex; if that ever changes these figures '
    'become exposed to the Type1 subsetting bug and need their own check')

C_REF = 'black'
COLOR = {'LL': 'C0', 'TT': 'C3', 'TL': 'C2', 'LT': 'C4', 'SUM': 'C5',
         # The extra spinmodes take the two cycle colours the components leave
         # free; the same two hues as the MG7 figures, in this style's palette.
         'onshell': 'C1', 'PA': 'C9'}
LS_EXTRA = {'onshell': (0, (6, 2)), 'PA': (0, (3, 1, 1, 1, 1, 1))}
FIGSIZE = (7.2, 10.0)
MS = 4
STEP_ALPHA = 0.55
DPI = 300



# The multiplier on the distribution pane's autoscale top, per figure variant
# and per y scale; what sets it is the number of rows in the opaque legend box.
# Ten rows on the full figure and on variant B, six on variant A because the
# two extra spinmodes are not there.  Checked on the rendered PNG.
# Variant B's distribution pane is shorter than the full figure's while
# carrying the same ten-row box, so it needs more of the two.
HEADROOM = {None: (4000.0, 1.45), 'A': (400.0, 1.35), 'B': (4000.0, 1.95)}


def _step(ax, edges, y, **kw):
    ax.step(edges, np.concatenate([y[:1], y]), where='pre', **kw)


def _distribution(ax, c, obs, extras, variant=None):
    """The distribution pane, shared by the full figure and both variants.

    Returns the spinmodes the selection left under ``PA.MIN_SEL_TO_DRAW``;
    that they were dropped is recorded and deliberately NOT written on the
    canvas, exactly as on the MG7 figures.
    """
    edges, x = c.edges, c.centres()
    ylab = PA.LABELS_TXT[obs][1]
    y, e = c.dist['full']
    _step(ax, edges, y, color=C_REF, lw=1.6, label=PA.CURVE_TXT['full'],
          zorder=6)
    ax.errorbar(x, y, yerr=e, fmt='none', ecolor=C_REF, elinewidth=1.0,
                zorder=6)
    # The other two spinmodes: full distributions like the black one, so they
    # are drawn as steps with bars and not with the components' markers.
    dropped = []
    for key, dx in extras:
        fd = PA.full_distribution(dx, obs)
        if not fd['drawable']:
            dropped.append((key, fd['n_sel']))
            continue
        _step(ax, edges, fd['y'], color=COLOR[key], lw=1.3, ls=LS_EXTRA[key],
              label=PA.EXTRA_TXT[key], zorder=5)
        ax.errorbar(x, fd['y'], yerr=fd['err'], fmt='none',
                    ecolor=COLOR[key], elinewidth=0.9, zorder=5)
    for k in PA.POL_KEYS:
        yk, ek = c.dist[k]
        shown = np.where(yk > 0, yk, np.nan) if obs in PA.LOGY else yk
        _step(ax, edges, shown, color=COLOR[k], lw=1.0, alpha=STEP_ALPHA,
              zorder=3)
        ax.errorbar(x, shown, yerr=np.where(np.isfinite(shown), ek, np.nan),
                    fmt='o', ms=MS, color=COLOR[k], label=PA.CURVE_TXT[k],
                    zorder=4)
    if obs in PA.LOGY:
        ax.set_yscale('log')
    ax.set_ylabel(ylab)
    ax.set_xlim(edges[0], edges[-1])
    lo, hi = ax.get_ylim()
    # The user style's legend has an opaque frame and ten rows, so on the log
    # pane it needs enough headroom to clear the total's peak outright rather
    # than merely to sit above the autoscale top.  The exclusive samples
    # resolve the Z_0 Z_0 tail down to 1e-8, stretching the pane to six
    # decades, and at the earlier 400x the box still covered the black step;
    # 4000x clears it.  Checked on the rendered PNG, not on the axis limits.
    head = HEADROOM[variant]
    ax.set_ylim(lo, hi * (head[0] if obs in PA.LOGY else head[1]))
    ax.legend(loc='upper left', fontsize=8)
    ax.tick_params(labelbottom=False)
    return dropped


def draw_shape(d, obs, outdir, extras=()):
    """Variant B in this style: the distribution, then ONE shape-ratio pane.

    The physics, the normalisation choice and the error treatment are
    ``pol_analysis``'s and are the same objects the MG7 variant draws; see
    ``plot_zz_pol.draw_shape``.  Only the rendering differs.  The two shaded
    tolerance bands behind the pane are +-1 % and +-3 % about 1 -- tighter than
    the sum pane's +-2 % / +-5 %, because a shape ratio between two spinmodes
    lives at a few percent -- and, like every other graphic in this style, what
    they are is stated in numbers.txt and not on the canvas.
    """
    c = PA.Curves(d, obs)
    edges, x = c.edges, c.centres()
    xlab = PA.LABELS_TXT[obs][0]

    fig = plt.figure(figsize=(FIGSIZE[0], 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.9], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    _distribution(ax, c, obs, extras, variant='B')

    ref = PA.shape_density(d, obs)
    axr.axhspan(0.97, 1.03, facecolor='C0', alpha=0.10, zorder=0)
    axr.axhspan(0.99, 1.01, facecolor='C0', alpha=0.16, zorder=0)
    axr.axhline(1.0, color=C_REF, ls='--', lw=1.0, zorder=2)
    series = []
    for key, dx in extras:
        if not PA.full_distribution(dx, obs)['drawable']:
            continue
        cmp = PA.compare_shape(ref, PA.shape_density(dx, obs))
        r, er = cmp['ratio'], cmp['ratio_err']
        series.append((r, er))
        _step(axr, edges, r, color=COLOR[key], lw=1.2, ls=LS_EXTRA[key],
              alpha=0.75, zorder=3)
        axr.errorbar(x, r, yerr=er, fmt='o', ms=MS + 0.5, color=COLOR[key],
                     label=PA.SHAPE_CURVE_TXT[key], zorder=4)
    if series:
        axr.set_ylim(*_shape_ylim(series))
    # A few-percent window gets two default ticks; ask for five so that the
    # scale of the excursion can be read off the pane.
    axr.yaxis.set_major_locator(MaxNLocator(5))
    axr.set_ylabel(PA.SHAPE_RATIO_TXT_2L, fontsize=7.5)
    axr.legend(loc='upper left', fontsize=8, ncol=2)
    axr.set_xlabel(xlab, fontsize=10)
    for s in axr.spines.values():
        s.set_linewidth(1.6)

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, PA.SHORT[obs])
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('%-10s variant B  N=%5d  norm=%s' % (PA.SHORT[obs], c.n_sel,
                                               PA.SHAPE_NORM))
    return base


def draw(d, obs, outdir, extras=(), variant=None):
    c = PA.Curves(d, obs)
    edges, x = c.edges, c.centres()
    xlab, ylab = PA.LABELS_TXT[obs]

    fig = plt.figure(figsize=FIGSIZE)
    # See the note in plot_zz_pol.draw: the distribution and the sum pane are a
    # stacked PAIR sharing one x axis, the 2 x 2 is a separate block with its
    # own, and the three gaps that implies cannot come from one hspace.
    gs = fig.add_gridspec(2, 1, height_ratios=[4.7, 2.5], hspace=0.13)
    top = gs[0].subgridspec(2, 1, height_ratios=[3.0, 1.7], hspace=0.06)
    sub = gs[1].subgridspec(2, 2, hspace=0.12, wspace=0.28)
    ax = fig.add_subplot(top[0])
    axs = fig.add_subplot(top[1], sharex=ax)
    small = [fig.add_subplot(sub[0, 0], sharex=ax),
             fig.add_subplot(sub[0, 1], sharex=ax),
             fig.add_subplot(sub[1, 0], sharex=ax),
             fig.add_subplot(sub[1, 1], sharex=ax)]

    # -- the distribution: the total, with the four components on top of it ---
    # Shared with variant B, and with variant A, which passes no extras.
    _distribution(ax, c, obs, extras, variant=variant)

    # -- the sum: the polarisation interference, on its own scale ------------
    r, er, _ = c.ratios['SUM']
    Rint, Eint = c.integrated['SUM']
    axs.axhspan(0.95, 1.05, facecolor='C0', alpha=0.10, zorder=0)
    axs.axhspan(0.98, 1.02, facecolor='C0', alpha=0.16, zorder=0)
    axs.axhline(1.0, color=C_REF, ls='--', lw=1.0, zorder=2)
    _step(axs, edges, r, color=COLOR['SUM'], lw=1.2, alpha=0.75, zorder=3)
    axs.errorbar(x, r, yerr=er, fmt='o', ms=MS + 0.5, color=COLOR['SUM'],
                 zorder=4)
    axs.set_ylim(*_ratio_ylim(r, er, 1.0))
    axs.set_ylabel(PA.RATIO_TXT['SUM'], fontsize=8)
    # Neither the integrated value, nor what the two bands are, nor the words
    # "polarisation interference" are written here any more.  All three are in
    # numbers.txt and RESULTS.md.
    axs.set_xlabel(xlab, fontsize=10)
    for s in axs.spines.values():
        s.set_linewidth(1.6)

    # -- the breakdown -------------------------------------------------------
    for a, k in zip(small, ['LL', 'TT', 'TL', 'LT']):
        rk, ek, _ = c.ratios[k]
        Rk = c.integrated[k][0]
        # The dashed line is that pane's integrated value; its number is in
        # numbers.txt, not beside it.
        a.axhline(Rk, color=C_REF, ls='--', lw=0.9, zorder=2)
        _step(a, edges, rk, color=COLOR[k], lw=1.0, alpha=STEP_ALPHA, zorder=3)
        a.errorbar(x, rk, yerr=ek, fmt='o', ms=MS, color=COLOR[k], zorder=4)
        a.set_ylim(*_ratio_ylim(rk, ek, Rk))
        a.set_ylabel(PA.RATIO_TXT[k], fontsize=9)
    for a in small[:2]:
        a.tick_params(labelbottom=False)
    for a in small[2:]:
        a.set_xlabel(xlab, fontsize=10)
    # The 2 x 2 panes are half the figure wide, so the default locator packs
    # the mass axis tightly enough that the labels run into one another.
    small[-1].xaxis.set_major_locator(MaxNLocator(6))

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, PA.SHORT[obs])
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    g = PA.diagnostics(d, obs)
    print('%-10s %-10s N=%5d  purity=%.3f  eff=%.3f  sum/full = %.4f +- %.4f'
          % (PA.SHORT[obs], 'variant ' + variant if variant else 'full',
             c.n_sel, g['purity'], g['efficiency'], Rint, Eint))
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots_userstyle'))
    ap.add_argument('--no-variants', action='store_true',
                    help='draw only the original three-tier figures')
    a = ap.parse_args()
    d = PA.Data(a.data)
    extras = PA.load_extras(a.data)
    for obs in PA.OBS:
        draw(d, obs, a.out, extras)
        if a.no_variants:
            continue
        # The two variants, ALONGSIDE the originals and never over them.
        draw(d, obs, os.path.join(a.out, PA.VARIANTS['A']['dir']),
             extras=(), variant='A')
        draw_shape(d, obs, os.path.join(a.out, PA.VARIANTS['B']['dir']),
                   extras)


if __name__ == '__main__':
    main()
