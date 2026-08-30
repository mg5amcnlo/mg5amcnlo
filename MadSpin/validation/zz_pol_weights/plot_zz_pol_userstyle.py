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
import collections
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
from plot_zz_pol import (_ratio_ylim, _shape_ylim,               # noqa: E402
                         _ratio6_ylim)

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
         'onshell': 'C1', 'PA': 'C9',
         # madspin_v1, on variant B only; the first cycle colour neither a
         # component nor another spinmode has taken, matching the MG7 choice.
         'madspin_v1': 'C6',
         # LO, on variant B and on the K-factor figure; the next free cycle
         # colour, again the same hue the MG7 script gives it.  C7 (grey) is
         # left alone on purpose: it reads as de-emphasis and this curve is not.
         'LO': 'C8',
         # The K-factor figure draws the unpolarised component too, and there
         # it is a curve among five rather than THE reference, so it needs an
         # entry in this dict as well as C_REF.
         'full': C_REF}
LS_EXTRA = {'onshell': (0, (6, 2)), 'PA': (0, (3, 1, 1, 1, 1, 1)),
            'madspin_v1': (0, (2, 1.5)), 'LO': (0, (7, 1.5, 1.5, 1.5))}

# The K-factor figure's two curves per panel are the same component at two
# ORDERS: same colour, told apart by the line.  See plot_zz_pol.LS_ORDER.
LS_ORDER = {'NLO': 'solid', 'LO': (0, (5, 2))}
# Its five K curves, one per component, need a line each on the shared panel.
LS_COMP = {'full': 'solid', 'LL': (0, (5, 2)), 'LT': (0, (1, 1.4)),
           'TL': (0, (6, 1.6, 1.4, 1.6)), 'TT': (0, (3, 1, 1, 1, 1, 1))}
FIGSIZE = (7.2, 10.0)
MS = 4
STEP_ALPHA = 0.55
DPI = 300



# The multiplier on the distribution pane's autoscale top, per figure variant
# and per y scale; what sets it is the number of rows in the opaque legend box.
# Ten rows on the full figure, TWELVE on variant B now that madspin_v1 and LO
# are a fourth and fifth total there, six on variant A because the extra
# spinmodes are not there.  Checked on the rendered PNG.
# Variant B's distribution pane is shorter than the full figure's while
# carrying the larger box, so it needs more of the two.
HEADROOM = {None: (4000.0, 1.45), 'A': (400.0, 1.35), 'B': (120000.0, 2.45)}


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

    ``extras`` is the three-curve variant-B list -- ``onshell``, ``PA`` and
    ``madspin_v1`` -- exactly as in the MG7 script.  ``madspin_v1`` is on this
    variant only.
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
    axr.legend(loc='upper left', fontsize=8, ncol=3)
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


# The multiplier on each K-factor panel's autoscale top.  Two legend rows on
# panels 1-5 and two on panel 6, but this style's legend has an OPAQUE frame
# and larger markers, so it needs more than the MG7 one does.  Checked on the
# rendered PNGs.
KF_HEADROOM = {True: 15.0, False: 1.42}
KF_HEADROOM_K = 1.42


def draw_kfactor(nlo, lo, obs, outdir):
    """The six-panel K-factor figure in the user style.

    The physics, the binning, the selection and every error bar are
    ``pol_analysis``'s and are the same objects ``plot_zz_pol.draw_kfactor``
    draws; only the rendering differs.  See that docstring for what the figure
    is, why it is ABSOLUTE where variant B is normalised, and which error goes
    on which curve.

    No tolerance bands here, unlike the ratio panes of the other figures in
    this style: those bands are drawn about 1 because the quantity is a ratio
    that ought to BE 1, and a K-factor ought not to be.  Putting a +-2 % band
    around 1 on panel 6 would suggest a null hypothesis nobody holds.
    """
    edges = PA.BINS[obs]
    x = 0.5 * (edges[:-1] + edges[1:])
    xlab, ylab = PA.LABELS_TXT[obs]
    name = PA.KF_CURVE_TXT
    logy = obs in PA.LOGY

    fig = plt.figure(figsize=(9.4, 10.4))
    gs = fig.add_gridspec(3, 2, hspace=0.07, wspace=0.30)
    axes = []
    for r in range(3):
        for cc in range(2):
            axes.append(fig.add_subplot(gs[r, cc],
                                        sharex=axes[cc] if r else None))

    ks = {}
    for i, key in enumerate(PA.KF_PANE_ORDER):
        ax = axes[i]
        kk = PA.kfactor(nlo, lo, obs, key)
        ks[key] = kk
        for tag, h in (('NLO', kk['nlo']), ('LO', kk['lo'])):
            y, e = np.asarray(h['y'], float), np.asarray(h['err'], float)
            shown = np.where(y > 0, y, np.nan) if logy else y
            _step(ax, edges, shown, color=COLOR[key], lw=1.4,
                  ls=LS_ORDER[tag], alpha=1.0 if tag == 'NLO' else STEP_ALPHA,
                  zorder=4 if tag == 'NLO' else 3)
            ax.errorbar(x, shown, yerr=np.where(np.isfinite(shown), e, np.nan),
                        fmt='o' if tag == 'NLO' else 's', ms=MS - 0.5,
                        mfc='none' if tag == 'LO' else None,
                        color=COLOR[key], label='%s, %s' % (name[key], tag),
                        zorder=5 if tag == 'NLO' else 4)
        if logy:
            ax.set_yscale('log')
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_xlim(edges[0], edges[-1])
        lo_y, hi_y = ax.get_ylim()
        ax.set_ylim(lo_y, hi_y * KF_HEADROOM[logy])
        ax.legend(loc='upper left', fontsize=8)
        ax.tick_params(labelsize=8.5)
        if not logy:
            ax.yaxis.set_major_locator(MaxNLocator(5))

    axk = axes[5]
    for key in PA.KF_PANE_ORDER:
        kk = ks[key]
        k, e = np.asarray(kk['k'], float), np.asarray(kk['err'], float)
        _step(axk, edges, k, color=COLOR[key], lw=1.2, ls=LS_COMP[key],
              alpha=0.8, zorder=3)
        axk.errorbar(x, k, yerr=e, fmt='o', ms=MS, color=COLOR[key],
                     label=name[key], zorder=4)
    axk.set_ylabel(PA.KFACTOR_TXT, fontsize=9)
    lo_y, hi_y = axk.get_ylim()
    axk.set_ylim(lo_y, lo_y + (hi_y - lo_y) * KF_HEADROOM_K)
    axk.legend(loc='upper left', fontsize=8, ncol=3)
    axk.yaxis.set_major_locator(MaxNLocator(6))
    axk.tick_params(labelsize=8.5)
    for sp in axk.spines.values():
        sp.set_linewidth(1.6)

    for i, ax in enumerate(axes):
        if i < 4:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(xlab, fontsize=10)

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, PA.SHORT[obs])
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    ki = PA.integrated_kfactors(nlo, lo, obs)
    print('%-10s K-factor  ' % PA.SHORT[obs]
          + '  '.join('%s=%.3f' % (PA.KF_CURVE_TXT[r['key']].replace(' ', ''),
                                   r['K']) for r in ki))
    return base


# --------------------------------------------------------------------------
# The two six-panel ratio figures in the user style.  Same panels, same
# objects, same conventions as plot_zz_pol.draw_ratio6 -- see that docstring
# for what each panel is, which error goes on which curve, and why the K panel
# carries no reference line.  Only the rendering differs.
#
# The +-2 % / +-5 % tolerance bands of this style's other ratio panes are NOT
# here.  Those are drawn about 1 on a quantity that ought to BE 1; on these
# panels only the sum pane is such a quantity, and putting a shaded tolerance
# behind a shaded SCALE band would make the two unreadable against each other
# on the one pane that carries both.  The sum pane keeps its dashed rule at 1
# and loses the two tolerance bands, and this is the reason.
BAND_ALPHA = 0.20
BAND_ALPHA_K = 0.14
KF6_HEADROOM_K = 0.40

# The seventh pane.  See plot_zz_pol._ratio6_top and the block of comments at
# pol_analysis.RATIO6_TOP_LOG_DECADES / RATIO6_TOP_BAND_KEYS for what is on it,
# why its y scale is measured rather than taken from PA.LOGY, why the scale
# band is on the unpolarised pair alone, and why the line weight steps down
# through the components.  Only the rendering differs here.
BAND_ALPHA_TOP = 0.24
# This style's legend has an OPAQUE frame and two rows of five, so it needs
# more headroom over the log pane than the MG7 one does.  Checked on the
# rendered PNGs.
TOP_HEADROOM = (26.0, 1.75)
TOP_LW_BASE = 1.9
TOP_LW_STEP = 0.22
TOP_LW_NLO = 0.30


def _band(ax, edges, lo, hi, color, alpha, zorder=1):
    """A scale band on the same step grid ``_step`` draws on.

    ``_step`` plots ``where='pre'`` after repeating the first value, so cell
    ``i`` spans ``[edges[i], edges[i+1]]``; ``fill_between(step='post')`` with
    ``x = edges`` and the LAST value repeated spans the same cells.  The two
    conventions meet in the middle and the band lands under its own curve
    rather than half a bin off it.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    ax.fill_between(edges, np.concatenate([lo, lo[-1:]]),
                    np.concatenate([hi, hi[-1:]]), step='post',
                    facecolor=color, alpha=alpha, lw=0.0, zorder=zorder)


# -- the one text-size knob the three shared painters take --------------------
# ``scale`` multiplies EVERY font size a pane writes and defaults to 1.0.  The
# argument, and the quarter-point rounding, are in plot_zz_pol._ts and are not
# repeated here.
def _ts(base, scale):
    """One font size, scaled and rounded to the quarter point."""
    return round(base * scale * 4.0) / 4.0


def _ratio6_top(ax, nlo, lo, obs, with_band=False, scale=1.0, legend_kw=None):
    """The seventh pane in the user style: dsigma/dx, both orders, five each.

    Steps and error bars, and NOT this style's usual ``errorbar(fmt='o')``
    markers: ten curves would put ten markers in every bin, which on the
    twelve-bin Delta phi pane is a hundred and twenty overlapping dots and no
    curve left to read.  The markers stay on the six panes below, which carry
    two curves each and have room for them.
    """
    edges = PA.BINS[obs]
    x = 0.5 * (edges[:-1] + edges[1:])
    ylab = PA.LABELS_TXT[obs][1]
    name = PA.KF_CURVE_TXT
    oname = PA.RATIO6_ORDER_TXT
    logy = PA.ratio6_top_logy(nlo, lo, obs)
    cur = PA.ratio6_top_curves(nlo, lo, obs, with_band=with_band)

    for i, key in enumerate(PA.KF_PANE_ORDER):
        for tag in PA.RATIO6_TOP_ORDERS:
            h = cur[tag][key]
            y, e = np.asarray(h['y'], float), np.asarray(h['err'], float)
            shown = np.where(y > 0, y, np.nan) if logy else y
            # LO over NLO, and later components over earlier ones; see the
            # MG7 script.
            z = 10 * (i + 1) + (6 if tag == 'LO' else 4)
            lw = (TOP_LW_BASE - TOP_LW_STEP * i
                  + (TOP_LW_NLO if tag == 'NLO' else 0.0))
            blo, bhi = h.get('lo'), h.get('hi')
            if with_band and blo is not None:
                _band(ax, edges, blo, bhi, COLOR[key], BAND_ALPHA_TOP,
                      zorder=z - 2)
            _step(ax, edges, shown, color=COLOR[key], lw=lw, ls=LS_ORDER[tag],
                  zorder=z, label='%s, %s' % (name[key], oname[tag]))
            ax.errorbar(x, shown, yerr=np.where(np.isfinite(shown), e, np.nan),
                        fmt='none', ecolor=COLOR[key], elinewidth=0.9,
                        alpha=0.9 if tag == 'NLO' else 0.6, zorder=z)
    if logy:
        ax.set_yscale('log')
    ax.set_ylabel(ylab, fontsize=_ts(9, scale))
    ax.set_xlim(edges[0], edges[-1])
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y * TOP_HEADROOM[0 if logy else 1])
    # Ten entries in five columns; ``legend_kw`` trims the SPACING and never
    # the structure.  See plot_zz_pol._ratio6_top.
    lkw = dict(fontsize=_ts(8, scale), columnspacing=1.2, handlelength=2.4)
    lkw.update(legend_kw or {})
    ax.legend(loc='upper left', ncol=5, **lkw)
    ax.tick_params(labelsize=_ts(8.5, scale))
    for sp in ax.spines.values():
        sp.set_linewidth(1.6)


# -- the two ratio painters, shared by the seven-pane and three-pane figures --
# Lifted out of ``draw_ratio6`` unchanged; the argument for one painter rather
# than two copies is in plot_zz_pol and is not repeated here.  Unlike the MG7
# pair these DO set ``xlim``, because this style's panes always did.


def _ratio6_ratio_pane(ax, cur, pane, anchor_line, edges, with_band=False,
                       scale=1.0):
    """One ratio pane: the named ratio at both orders, with its anchor line."""
    x = 0.5 * (edges[:-1] + edges[1:])
    rlab = PA.RATIO_TXT
    oname = PA.RATIO6_ORDER_TXT
    col = COLOR[pane] if pane != 'SUM' else COLOR['SUM']
    series = []
    for tag in PA.RATIO6_ORDERS:
        h = cur[pane][tag]
        r, e = np.asarray(h['r'], float), np.asarray(h['err'], float)
        blo = h.get('lo') if with_band else None
        bhi = h.get('hi') if with_band else None
        if with_band:
            _band(ax, edges, blo, bhi, col, BAND_ALPHA,
                  zorder=2 if tag == 'NLO' else 1)
        _step(ax, edges, r, color=col, lw=1.3, ls=LS_ORDER[tag],
              alpha=1.0 if tag == 'NLO' else STEP_ALPHA,
              zorder=4 if tag == 'NLO' else 3)
        ax.errorbar(x, r, yerr=e, fmt='o' if tag == 'NLO' else 's',
                    ms=MS - 0.5, mfc='none' if tag == 'LO' else None,
                    color=col, label=oname[tag],
                    zorder=6 if tag == 'NLO' else 5)
        series.append((r, e, blo, bhi))
    if anchor_line is not None:
        ax.axhline(anchor_line, color=C_REF, lw=1.0,
                   ls=':' if pane == 'SUM' else '--', zorder=2)
    ax.set_ylim(*_ratio6_ylim(series, anchor_line, head=0.24))
    # The compact sum name, and this painter is shared with the seven-pane A6
    # figure, which therefore takes it too; see
    # ``pol_analysis.RATIO_SUM_COMPACT``.  It is the one string this style
    # sets as mathtext rather than plain text, and that is argued there.
    ax.set_ylabel(PA.RATIO_SUM_COMPACT if pane == 'SUM' else rlab[pane],
                  fontsize=_ts(9, scale))
    ax.set_xlim(edges[0], edges[-1])
    ax.tick_params(labelsize=_ts(8.5, scale))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.legend(loc='best', fontsize=_ts(8, scale), ncol=2)


def _ratio6_k_pane(axk, cur, edges, with_band=False, scale=1.0,
                   legend_kw=None):
    """The K-factor pane: five curves, no reference line."""
    x = 0.5 * (edges[:-1] + edges[1:])
    kseries = []
    for key in PA.KF_PANE_ORDER:
        kk = cur['K'][key]
        k, e = np.asarray(kk['k'], float), np.asarray(kk['err'], float)
        blo = kk.get('lo') if with_band else None
        bhi = kk.get('hi') if with_band else None
        if with_band:
            _band(axk, edges, blo, bhi, COLOR[key], BAND_ALPHA_K,
                  zorder=2 if key == 'full' else 1)
        _step(axk, edges, k, color=COLOR[key], lw=1.2, ls=LS_COMP[key],
              alpha=0.85, zorder=3)
        axk.errorbar(x, k, yerr=e, fmt='o', ms=MS, color=COLOR[key],
                     label=PA.KF_CURVE_TXT[key], zorder=4)
        kseries.append((k, e, blo, bhi))
    axk.set_ylim(*_ratio6_ylim(kseries, None, head=KF6_HEADROOM_K))
    axk.set_ylabel(PA.KFACTOR_TXT, fontsize=_ts(9, scale))
    axk.set_xlim(edges[0], edges[-1])
    klw = dict(fontsize=_ts(8, scale))
    klw.update(legend_kw or {})
    axk.legend(loc='upper left', ncol=3, **klw)
    axk.yaxis.set_major_locator(MaxNLocator(6))
    axk.tick_params(labelsize=_ts(8.5, scale))
    for sp in axk.spines.values():
        sp.set_linewidth(1.6)


# -- the fixed canvas the two ratio6 observables share ------------------------
# The same problem the MG7 script's R6_AXES block sets out, and the same fix;
# the argument for it is written there and is not repeated here.  Measured on
# this style: M(e+ mu+) came out 2461 px wide and Delta phi(e+ e-) 2412, so the
# two were 49 px apart -- worse than the MG7 pair's 26, because this style
# writes its axis names as plain text and its tick labels larger.
#
# The axes block, 7.285 x 9.8945 in, is exactly what figsize (9.4, 12.85) and
# matplotlib's default subplot margins already gave it, so nothing inside the
# frame moves: height_ratios, hspace and wspace are untouched and span the same
# block.  Margins are the worst case over both observables and both variants,
# label extent against axes extent on the rendered figure -- left 0.653 in,
# right 0.120, bottom 0.415, top 0.000 -- each with ~0.12 in of white on top,
# which is about what the tight crop's pad was adding.  As on the MG7 figure
# the left margin is set by Z0Z0 / full on M(e+ mu+) and its 0.000 / 0.025 /
# 0.050 ticks, not by the wide pane (0.573 in) and not by the long rotated
# sum name (0.630); measured, not assumed.
R6_AXES = (7.285, 9.8945)             # the axes block, w x h in inches
# THE LEFT MARGIN HAD TO MOVE, 0.77 -> 0.92, and unlike the MG7 figure's it
# had to: with the sum pane's name written as a sum this figure needs 0.793 in
# on the left and 0.77 CLIPPED IT.  matplotlib's mathtext stacks the summation
# limits under the sign where LaTeX sets them beside it, which makes the
# rotated name 0.31 in wide against MG7's 0.17.  0.793 + 0.12, rounded up.
# Which is where the three-pane figure's left margin lands too, for the same
# reason and off the same measurement.  See pol_analysis.RATIO_SUM_COMPACT.
R6_MARGIN = (0.92, 0.24, 0.54, 0.11)  # left, right, bottom, top, in inches
R6_FIG = (R6_AXES[0] + R6_MARGIN[0] + R6_MARGIN[1],
          R6_AXES[1] + R6_MARGIN[2] + R6_MARGIN[3])
R6_BOX = dict(left=R6_MARGIN[0] / R6_FIG[0],
              right=1.0 - R6_MARGIN[1] / R6_FIG[0],
              bottom=R6_MARGIN[2] / R6_FIG[1],
              top=1.0 - R6_MARGIN[3] / R6_FIG[1])


def draw_ratio6(nlo, lo, obs, outdir, with_band=False):
    """Variant A as a distribution pane over a 3 x 2 of ratios, user style."""
    edges = PA.BINS[obs]
    xlab = PA.LABELS_TXT[obs][0]
    cur = PA.ratio6_curves(nlo, lo, obs, with_band=with_band)

    fig = plt.figure(figsize=R6_FIG)
    # Two vertical rhythms, so two gridspecs: see plot_zz_pol.draw_ratio6.  The
    # wide pane on top carries its own x tick labels and its own axis name and
    # is set off from the 3 x 2 by the outer gap.
    #
    # The share-out is the MG7 figure's, kept in step with it on purpose: the
    # wide pane takes 4.3 of the outer 12.85 where it took 2.9 of 13.3, and the
    # grid's rows stop being equal -- 3.55 for the sum-consistency and
    # K-factor row, 2.5 for each of the two fraction rows, which are flat
    # enough to read at that height.  The figure shrinks, 13.6 in to 12.85 in.
    # Height in inches, pane then row 1 then rows 2-3: 2.19 -> 3.15,
    # 2.50 -> 2.49, 2.50 -> 1.75.  Those ratio units were picked to read as
    # inches of the figsize they were set against, 4.3 + 8.55 = 12.85; since
    # R6_AXES the canvas is declared separately and no longer carries that
    # number, so only the ratios and the inch heights survive, and both are
    # unchanged -- the block they divide is still 9.8945 in tall.
    # The top row keeps MORE than its MG7 counterpart's 3.3 for one reason:
    # this style writes the sum-consistency axis name as plain text with
    # spaces, (Z0Z0 + ZTZT + ZTZ0 + Z0ZT) / full, which set rotated is longer
    # than the mathtext MG7 sets, and at 3.15 it came within 0.07 in of the
    # frame.  Measured, label extent against axes extent, not guessed.
    # The outer hspace is a fraction of the MEAN row height, so it moves with
    # the ratios: 0.10 of the new mean buys back the gap 0.09 of the old one
    # gave, which is what keeps the pane's axis name off the grid.
    outer = fig.add_gridspec(2, 1, height_ratios=[4.3, 8.55], hspace=0.10,
                             **R6_BOX)
    gs = outer[1].subgridspec(3, 2, height_ratios=[3.55, 2.5, 2.5],
                              hspace=0.07, wspace=0.30)
    axtop = fig.add_subplot(outer[0])
    axes = []
    for r in range(3):
        for cc in range(2):
            axes.append(fig.add_subplot(gs[r, cc],
                                        sharex=axes[cc] if r else None))

    _ratio6_top(axtop, nlo, lo, obs, with_band=with_band)
    axtop.set_xlabel(xlab, fontsize=10)

    _ratio6_ratio_pane(axes[0], cur, 'SUM', 1.0, edges, with_band=with_band)
    for sp in axes[0].spines.values():
        sp.set_linewidth(1.6)

    _ratio6_k_pane(axes[1], cur, edges, with_band=with_band)

    for ax, pane in zip(axes[2:], PA.RATIO6_FRACTIONS):
        _ratio6_ratio_pane(ax, cur, pane, cur[pane]['NLO']['integrated'],
                           edges, with_band=with_band)

    for i, ax in enumerate(axes):
        if i < 4:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(xlab, fontsize=10)

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, PA.SHORT[obs])
    # NO tight crop: the canvas is already the right size, and the same size
    # for both observables.  See R6_BOX.
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=DPI)
    plt.close(fig)
    print('%-10s 7-pane%-10s sum/full NLO %.4f  LO %.4f'
          % (PA.SHORT[obs], ' + band' if with_band else '',
             cur['SUM']['NLO']['integrated'], cur['SUM']['LO']['integrated']))
    return base


# -- the three-pane variation's canvas ----------------------------------------
# The reduced figure, user style.  What it is and why is at
# pol_analysis.RATIO2_DIR; the geometry argument is at plot_zz_pol.R3_ROWS and
# is not repeated.  Only what this style does DIFFERENTLY is here.
#
# The block keeps this style's seven-pane width, 7.285 in, so the two figures
# read at one scale.  The distribution pane takes the same 4.62 in the MG7
# figure gives it -- the two styles are held in step wherever they can be.
#
# THE RATIO PANES ARE TALLER THAN THE MG7 FIGURE'S, 2.49 against 2.31, for the
# one reason the seven-pane figure's top row is taller: this style writes the
# sum-consistency name as PLAIN TEXT WITH SPACES, (Z0Z0 + ZTZT + ZTZ0 + Z0ZT) /
# full, which set rotated is 2.08 in against the mathtext MG7's 1.90.  The pane
# is sized to that string, not to its content, and its width buys nothing --
# which is why the extra width this variation gives the ratio panes does not
# let either style shorten them.
R3_ROWS = (4.62, 2.49, 2.49)  # distribution pane, sum, K -- inches of axes
# The gap in INCHES, 0.140, which is what this style's seven-pane grid puts
# between its own rows; hspace is solved for it because it is a fraction of the
# mean row height and these rows are not those.  Nothing but frame is in either
# gap: the x axis is written once, at the bottom.
R3_GAP = 0.140
R3_AXES = (7.285, sum(R3_ROWS) + (len(R3_ROWS) - 1) * R3_GAP)
# Measured on this figure, worst case over both observables and both variants,
# plus 0.12 in rounded up to the hundredth -- not carried over from R6_MARGIN,
# whose left was set by a pane this figure does not draw.  Here the left is the
# sum pane's rotated name at 0.793 in, ahead of the distribution pane's 0.573
# and the K pane's 0.478.
#
# THE LEFT MARGIN MOVED FURTHER THAN THE MG7 FIGURE'S, 0.75 -> 0.92 against
# 0.69 -> 0.74, and the reason is the same one that lets this style keep its
# row heights where MG7 has to grow them.  matplotlib's mathtext sets the
# summation limits UNDER the sign; LaTeX's inline math sets them beside it.
# Stacked, the name is short along the pane and WIDE across it -- 0.31 in
# against MG7's 0.17 -- and rotated, across is the left margin.  The trade is
# 0.17 in of page width for a pane that never has to grow.  The other three
# margins are unchanged.
#
# figsize and the gridspec box come from ``_r3_geom``, here as for every
# variation; see plot_zz_pol.
R3_MARGIN = (0.92, 0.24, 0.54, 0.11)  # left, right, bottom, top, in inches


# -- the text-size variations -------------------------------------------------
# What they are and why they are a spread rather than one number is at
# plot_zz_pol.R3_TEXT; the scales themselves are shared, from
# ``pol_analysis.RATIO2_TEXT``, so the two styles cannot end up offering
# different sizes.  Only the ROWS and MARGINS below are this style's own, and
# they are its own for this style's usual reason: its type is DejaVu Sans at
# stock matplotlib metrics, not the MG7 script's LaTeX, so the same string at
# the same size is not the same number of inches.
R3Text = collections.namedtuple('R3Text', 'suffix scale rows gap margin '
                                          'top_legend k_legend')
R3_TEXT_BASE = R3Text('', 1.00, R3_ROWS, R3_GAP, R3_MARGIN, {}, {})
# THE ROWS NEVER MOVE IN THIS STYLE, and that is the reverse of how the two
# styles used to stand.  The explicit name was LONGER here than in MG7 -- 2.08
# in of plain text with spaces against 1.895 in of mathtext -- which is the
# whole reason these ratio panes are 2.49 in where MG7's are 2.31.  Set as a
# sum it is SHORTER here: matplotlib's mathtext puts the summation limits
# UNDER the sign where LaTeX's inline math puts them beside it, so the string
# is two lines deep and correspondingly short along the pane -- 1.03 in at
# 1.00x against MG7's 1.322, and 1.66 in at 1.60x, which a 2.49 in pane still
# clears by 0.415.  MG7 has to grow its panes at 1.60x and this style does
# not.
#
# WHAT IT PAYS FOR THAT is width, and the left margin is where this style's
# geometry diverges hardest.  Stacked limits make the rotated name 0.31 in
# wide against MG7's 0.17, and DejaVu Sans is a wider face than Computer
# Modern to begin with, so the sum pane needs 0.79 in on the left at 1.00x
# where MG7 needs 0.61 -- and 1.13 against 0.87 at 1.60x.  The margins below
# are measured, not scaled.
R3_TEXT = [
    # 1.15x.  Nothing gives: legend 6.78 in in a 7.285 in pane, rows and
    # spacing untouched, the sum name clearing 0.655 in per end.
    R3Text('_text115', 1.15, (4.62, 2.49, 2.49), 0.140,
           (1.00, 0.26, 0.57, 0.11), {}, {}),
    # 1.30x.  The legend's white gives, the same trim the MG7 script takes at
    # this step: 7.82 in cut to 7.08 in a 7.285 in pane.  Five columns, ten
    # entries, unchanged.
    R3Text('_text130', 1.30, (4.62, 2.49, 2.49), 0.140,
           (1.09, 0.27, 0.61, 0.11),
           dict(columnspacing=0.8, handlelength=2.0, handletextpad=0.5), {}),
    # 1.45x.  Trimmed harder than MG7 needs at the same scale -- 0.45 em
    # columns against 0.55 -- because DejaVu Sans sets the same ten entries
    # wider: at MG7's trim this legend is still 0.04 in over the frame.  With
    # it, 6.95 in in a 7.285 in pane.
    R3Text('_text145', 1.45, (4.62, 2.49, 2.49), 0.140,
           (1.17, 0.29, 0.65, 0.11),
           dict(columnspacing=0.45, handlelength=1.6, handletextpad=0.35),
           {}),
    # 1.60x.  THE LEGEND'S TYPE STOPS AT 11.5 pt, the size 1.45x gives it, for
    # the reason argued at plot_zz_pol.R3_TEXT: the alternative is fewer than
    # five columns, which would put two components in one column and break the
    # pairing the pane is read by.  The rows do NOT grow here, unlike MG7's --
    # see above.
    R3Text('_text160', 1.60, (4.62, 2.49, 2.49), 0.140,
           (1.26, 0.30, 0.69, 0.11),
           dict(fontsize=11.5, columnspacing=0.45, handlelength=1.6,
                handletextpad=0.35), {}),
]


# The suffixes and scales are pol_analysis's, not this module's.  An assertion
# and not a comment: two styles quietly offering different sizes is exactly the
# failure the shared list exists to prevent, and it would show up as a figure
# missing from a comparison rather than as an error.
assert [(t.suffix, t.scale) for t in R3_TEXT] == PA.RATIO2_TEXT, \
    'R3_TEXT disagrees with pol_analysis.RATIO2_TEXT'


def _r3_geom(t):
    """``(figsize, hspace, gridspec box)`` for one text-size variation.

    The gap is held in INCHES and the fraction solved for; see
    plot_zz_pol._r3_geom.
    """
    axes = (R3_AXES[0], sum(t.rows) + (len(t.rows) - 1) * t.gap)
    fig = (axes[0] + t.margin[0] + t.margin[1],
           axes[1] + t.margin[2] + t.margin[3])
    box = dict(left=t.margin[0] / fig[0],
               right=1.0 - t.margin[1] / fig[0],
               bottom=t.margin[2] / fig[1],
               top=1.0 - t.margin[3] / fig[1])
    return fig, t.gap / (sum(t.rows) / len(t.rows)), box


def draw_ratio3(nlo, lo, obs, outdir, with_band=False, text=R3_TEXT_BASE):
    """The reduced variation: distribution pane, sum, K-factor, user style.

    Three full-width panes sharing ONE x axis, written once at the bottom.
    See plot_zz_pol.draw_ratio3 and pol_analysis.RATIO2_DIR.
    """
    edges = PA.BINS[obs]
    xlab = PA.LABELS_TXT[obs][0]
    cur = PA.ratio6_curves(nlo, lo, obs, with_band=with_band)

    figsize, hspace, box = _r3_geom(text)
    fig = plt.figure(figsize=figsize)
    # ONE gridspec where the seven-pane figure needs two: one rhythm, three
    # full-width panes over one x axis, and the ratios are the inches.
    gs = fig.add_gridspec(3, 1, height_ratios=list(text.rows),
                          hspace=hspace, **box)
    axtop = fig.add_subplot(gs[0])
    axs = fig.add_subplot(gs[1], sharex=axtop)
    axk = fig.add_subplot(gs[2], sharex=axtop)

    _ratio6_top(axtop, nlo, lo, obs, with_band=with_band, scale=text.scale,
                legend_kw=text.top_legend)
    _ratio6_ratio_pane(axs, cur, 'SUM', 1.0, edges, with_band=with_band,
                       scale=text.scale)
    for sp in axs.spines.values():
        sp.set_linewidth(1.6)
    _ratio6_k_pane(axk, cur, edges, with_band=with_band, scale=text.scale,
                   legend_kw=text.k_legend)

    for ax in (axtop, axs):
        ax.tick_params(labelbottom=False)
    axk.set_xlabel(xlab, fontsize=_ts(10, text.scale))

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, PA.SHORT[obs])
    # NO tight crop: the canvas is already the right size, and the same size
    # for both observables.  See R3_MARGIN and ``_r3_geom``.
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=DPI)
    plt.close(fig)
    print('%-10s 3-pane%-10s sum/full NLO %.4f  LO %.4f'
          % (PA.SHORT[obs], ' + band' if with_band else '',
             cur['SUM']['NLO']['integrated'], cur['SUM']['LO']['integrated']))
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
    # madspin_v1 is variant B's alone; see pol_analysis.VARIANT_B_EXTRA_SAMPLES.
    b_extras = extras + PA.load_variant_b_extras(a.data)
    for obs in PA.OBS:
        draw(d, obs, a.out, extras)
        if a.no_variants:
            continue
        # The two variants, ALONGSIDE the originals and never over them.
        draw(d, obs, os.path.join(a.out, PA.VARIANTS['A']['dir']),
             extras=(), variant='A')
        draw_shape(d, obs, os.path.join(a.out, PA.VARIANTS['B']['dir']),
                   b_extras)
    # The K-factor figure, in its own subdirectory beside the two variants.
    # Skipped, and said to be skipped, when the LO .npz is not there.
    lo = PA.load_kfactor_partner(a.data)
    if lo is None:
        print('LO sample not in %s: K-factor figure not drawn' % a.data)
    elif not a.no_variants:
        for obs in PA.OBS:
            draw_kfactor(d, lo, obs, os.path.join(a.out, PA.KFACTOR_DIR))
            # The two six-panel ratio figures, in their own subdirectories.
            draw_ratio6(d, lo, obs, os.path.join(a.out, PA.RATIO6_DIR))
            # The reduced three-pane variation; see pol_analysis.RATIO2_DIR.
            draw_ratio3(d, lo, obs, os.path.join(a.out, PA.RATIO2_DIR))
            # ... and one directory per larger text size, alongside it.
            for t in R3_TEXT:
                draw_ratio3(d, lo, obs,
                            os.path.join(a.out, PA.RATIO2_DIR + t.suffix),
                            text=t)
        if d.has_scale and lo.has_scale:
            for obs in PA.OBS:
                draw_ratio6(d, lo, obs,
                            os.path.join(a.out, PA.RATIO6_SCALE_DIR),
                            with_band=True)
                draw_ratio3(d, lo, obs,
                            os.path.join(a.out, PA.RATIO2_SCALE_DIR),
                            with_band=True)
                for t in R3_TEXT:
                    draw_ratio3(d, lo, obs,
                                os.path.join(a.out, PA.RATIO2_SCALE_DIR
                                             + t.suffix),
                                with_band=True, text=t)
        else:
            print('no MUR*_MUF* columns in %s: the scale figure is not drawn'
                  % a.data)


if __name__ == '__main__':
    main()
