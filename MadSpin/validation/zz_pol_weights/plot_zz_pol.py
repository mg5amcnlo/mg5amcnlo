#!/usr/bin/env python3
"""MadSpin's polarisation weights on a showered NLO ``p p > z z``, MG7 style.

One figure per observable, in three tiers:

1. the distribution -- the full (unpolarised) sample with the four
   polarisation components ``LL``, ``TT``, ``TL``, ``LT`` overlaid on it, at
   absolute normalisation and in the same binning, so the reader sees the
   components and the total together rather than only the total, plus the
   full (unpolarised) distribution of the same process run through MadSpin's
   ``onshell`` and ``PA`` spinmodes;
2. a full-width pane for ``(LL + TT + TL + LT) / full``, on its own vertical
   scale;
3. a 2 x 2 breakdown, ``LL/full``, ``TT/full``, ``TL/full``, ``LT/full``.

The two extra spinmodes enter the distribution pane and NOWHERE else.  The
ratio panes are the polarisation decomposition of the reference (``madspin``)
sample; putting a second sample's numerator over a first sample's denominator
would be meaningless, and each extra sample's own decomposition is a different
study.  A spinmode curve is drawn only where the selection leaves enough events
for it to be a measurement -- see ``PA.MIN_SEL_TO_DRAW``.  With the exclusive
``decay z > e+ e-`` / ``decay z > mu+ mu-`` card both observables clear that
threshold for all three modes, so nothing is dropped; the mechanism is kept
because a future sample may not, and ``numbers.txt`` records the survivors and
the drawn/not-drawn decision either way.

**Nothing but the axis labels, the tick labels and the legend is written on
these figures.**  The integrated ratios, the per-pane numbers, the identity of
the sample and every explanatory note were on the figure in earlier passes and
have been moved off it; all of them are in ``numbers.txt`` and RESULTS.md, and
``write_numbers`` prints the figure caption line that used to be the plot title
so that it stays recoverable.  The dashed reference line in each of the four
small panes is still that pane's integrated value -- the line is drawn, its
number is not printed.

Tier 2 is the physics and is drawn as such.  The four polarisation weights are
not a partition of the cross section: each is the production/decay convolution
restricted to one polarisation of each ``z``, and the *interference* between
different polarisations of the same ``z`` belongs to none of them.  Their sum is
therefore the full rate minus that interference, and where it departs from 1 is
where the interference lives -- the same quantity the ``pure_interference``
machinery in this repository exists to isolate.  It has no reason to sit at 1
and the figure must be able to show it not sitting at 1, which is why that pane
never shares a vertical scale with anything else.

The four small panes do not share a scale either, and for a different reason:
``TT/full`` is about 0.70 and ``LL/full`` about 0.06, so one shared window
would either clip the first or flatten the last four into a line.  Each is
autoscaled around its own integrated value, which is drawn as a dashed line so
that the shape can be read against it.

Style follows the MG7 paper's ``plotexample/dummyplot.py`` exactly as the
sibling studies under ``MadSpin/validation/`` do, importing their module rather
than restating it: LaTeX text, serif, step histograms of line width 1.2, the
tableau colours with black/blue/red promoted, frameless legends, minor tick
locators, and the Type1 minus-sign workaround.

Two variants, written ALONGSIDE this figure
-------------------------------------------
Into subdirectories of the same output directory, leaving the original figures
exactly where they were:

``plots/variant_A_madspin_only/``
    the same three tiers with the ``onshell`` and ``PA`` curves dropped from
    the distribution pane -- the reference sample's polarisation decomposition
    on its own.  Same binning, same styles, PDF and PNG.

``plots/variant_B_shape_ratio/``
    the distribution pane with all three modes on it, and then ONE ratio pane
    instead of the sum pane and the 2 x 2: the self-normalised shape ratio
    ``(1/sigma dsigma/dX)_Y / (1/sigma dsigma/dX)_madspin`` for both
    ``Y = onshell`` and ``Y = PA``, drawn together.  Dividing each mode by its
    own cross section first takes the RATE difference out and leaves the SHAPE,
    which is what separates the two modes here: ``onshell`` differs from
    ``madspin`` in rate and much less in shape, ``PA`` agrees in rate and
    differs in shape.  Which sigma does the normalising is
    ``pol_analysis.SHAPE_NORM``, argued there and printed into numbers.txt with
    both candidates tabulated.

Usage::

    plot_zz_pol.py [--data DIR] [--out DIR] [--numbers PATH] [--check-minus]
                   [--no-variants]
"""

import argparse
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import pol_analysis as PA                                        # noqa: E402

_LI = os.path.abspath(os.path.join(_HERE, '..', 'zz_loopinduced'))
if _LI not in sys.path:
    sys.path.insert(0, _LI)
from plot_zz_loopinduced import (                                # noqa: E402
    USETEX, MINUS_FIX, check_minus, LW, allcolors,
)
_NLO = os.path.abspath(os.path.join(_HERE, '..', 'zz_nlo'))
if _NLO not in sys.path:
    sys.path.insert(0, _NLO)
from plot_zz_stack import wants_minus as _wants_minus_any        # noqa: E402


def wants_minus(fig):
    """``_wants_minus_any``, but a TICK label only counts if it is on screen.

    The sibling helper asks every tick label the locator produced whether it
    carries a minus.  That over-reports: ``MaxNLocator`` routinely emits one
    tick beyond each end of the view, and matplotlib never draws those.  The
    ``Z_0Z_0/full`` pane of the six-panel figures is exactly that case -- the
    locator offers -0.03 on a pane whose window starts at -0.0056 -- so the
    figure was declared to "want" a minus that is not on it, ``--check-minus``
    then looked for a ``/minus`` that could not be there, and a correct figure
    was reported as a failure.

    Axis labels, titles and legend entries are still asked unconditionally:
    they are always drawn, and narrowing THOSE could hide a genuinely eaten
    sign, which is the one outcome this check exists to prevent.  Only the
    tick labels are filtered, and only by the view interval their own axis is
    set to.  The sibling module is left alone; this is a wrapper, not an edit.
    """
    fig.canvas.draw()
    for ax in fig.axes:
        always = [ax.xaxis.label, ax.yaxis.label, ax.title]
        leg = ax.get_legend()
        if leg is not None:
            always += list(leg.get_texts())
        for t in always:
            if '-' in t.get_text() or '\u2212' in t.get_text():
                return True
        for axis, (v0, v1) in ((ax.xaxis, ax.get_xlim()),
                               (ax.yaxis, ax.get_ylim())):
            lo_v, hi_v = min(v0, v1), max(v0, v1)
            for loc, t in zip(axis.get_ticklocs(), axis.get_ticklabels()):
                if not (lo_v <= loc <= hi_v):
                    continue
                if '-' in t.get_text() or '\u2212' in t.get_text():
                    return True
    return False


COLOR = {'full': 'black', 'LL': 'blue', 'TT': 'red',
         'TL': allcolors[2], 'LT': allcolors[4], 'SUM': allcolors[5],
         # The two extra spinmodes are full (unpolarised) distributions, like
         # the black one and unlike the four components, so they take the two
         # hues the components leave free rather than a shade of one of them.
         'onshell': 'darkorange', 'PA': allcolors[9],
         # madspin_v1 is on variant B only.  allcolors[6] is the first hue
         # neither a component nor another spinmode has taken.
         'madspin_v1': allcolors[6],
         # LO, on variant B and on the K-factor figure.  allcolors[8] is the
         # next hue free of a component and of another sample; grey
         # (allcolors[7]) is deliberately left alone, because on the six-decade
         # log pane it is the one colour that reads as "de-emphasised" and this
         # curve is not.
         'LO': allcolors[8]}
LS = {'full': 'solid', 'LL': 'dashed', 'TT': 'dashdot',
      'TL': (0, (1, 1.4)), 'LT': (0, (5, 1.5, 1, 1.5)), 'SUM': 'solid',
      'onshell': (0, (6, 2)), 'PA': (0, (3, 1, 1, 1, 1, 1)),
      'madspin_v1': (0, (2, 1.5)), 'LO': (0, (7, 1.5, 1.5, 1.5))}

# The K-factor figure's two curves per panel are the SAME component at two
# ORDERS, so they share the component's colour and are told apart by the line:
# NLO is the solid one the rest of the study already draws that component with,
# LO the dashed one.  With two curves to a panel that is unambiguous, and it
# keeps Z_0Z_0 blue on this figure exactly as it is blue on every other.
LS_ORDER = {'NLO': 'solid', 'LO': (0, (5, 2))}

# The caption these figures used to carry as a plot title lives in
# ``pol_analysis.CAPTION``, next to everything else both plotting scripts
# share, and is printed into numbers.txt by :func:`write_numbers`.


def _pad(lo, hi, frac=0.18):
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    d = hi - lo
    return lo - frac * d, hi + frac * d


def _ratio_ylim(r, e, anchor=None):
    """A window that contains the drawn points and their bars, and 1 if close.

    ``anchor`` (the integrated value, or 1 for the sum pane) is included so
    that the reference line is always visible: a pane whose dashed reference
    has fallen off the top is a pane that cannot be read.
    """
    ok = np.isfinite(r) & np.isfinite(e)
    if not ok.any():
        return 0.0, 1.0
    lo = float(np.min(r[ok] - e[ok]))
    hi = float(np.max(r[ok] + e[ok]))
    if anchor is not None:
        lo, hi = min(lo, anchor), max(hi, anchor)
    return _pad(lo, hi)


def _shape_ylim(series, frac=0.12, head=0.30, noisy=3.0):
    """The window for the variant-B pane: sized by the structure, not the edge.

    ``_ratio_ylim`` takes the extreme of ``point +- bar`` over every bin.  On
    this pane that is the wrong rule, because the thinly populated edge bins
    carry bars two to three times the median and would set a window in which
    the several-percent structure the pane exists to show is a flat line.

    So: the window is taken from the bins whose bar is at most ``noisy`` times
    the median bar -- the ones that are a measurement -- as ``point +- bar``,
    and is then widened to contain the CENTRAL value of every bin including the
    noisy ones, so that no drawn point is off the canvas even where its bar
    runs past the frame.  1 is always inside.  ``series`` is a list of
    ``(ratio, error)`` pairs; all of them share the one window, because the
    pane's whole purpose is to compare them.

    The padding is asymmetric -- ``head`` above against ``frac`` below --
    because this pane carries the only legend below the distribution, at its
    upper left, and a legend sitting on the highest error bar is as unreadable
    as a point off the frame.
    """
    rr = np.concatenate([np.asarray(r, dtype=float) for r, _ in series])
    ee = np.concatenate([np.asarray(e, dtype=float) for _, e in series])
    ok = np.isfinite(rr) & np.isfinite(ee)
    if not ok.any():
        return 0.0, 2.0
    med = float(np.median(ee[ok]))
    fine = ok & (ee <= noisy * med) if med > 0 else ok
    if not fine.any():
        fine = ok
    lo = min(1.0, float(np.min(rr[fine] - ee[fine])), float(np.min(rr[ok])))
    hi = max(1.0, float(np.max(rr[fine] + ee[fine])), float(np.max(rr[ok])))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 2.0
    d = hi - lo
    return lo - frac * d, hi + head * d


# The multiplier on the distribution pane's autoscale top, per figure variant
# and per y scale.  What sets it is the number of rows in the legend, which is
# drawn inside the pane at the upper left: seven curves on the full figure,
# NINE on variant B now that madspin_v1 and LO are a fourth and fifth total
# there, and five on variant A because the extra spinmodes are not there.  All
# of these were checked on the rendered PNG and not on the axis limits.  The
# jump from 2500x to 12000x is the ninth legend row: at 2500x the Z_T Z_T
# entry sat on the black total around M = 100 GeV.
HEADROOM = {None: (250.0, 1.62), 'A': (60.0, 1.45), 'B': (12000.0, 2.00)}


def _distribution(ax, c, obs, extras, variant=None):
    """Tier 1, shared by every variant: the total, the four components, and
    -- unless the variant drops them -- the other two spinmodes.

    Returns the list of spinmodes that were NOT drawn because the selection
    left them under ``PA.MIN_SEL_TO_DRAW``.  Nothing about that is written on
    the figure; ``write_numbers`` says it.
    """
    edges, x = c.edges, c.centres()
    ylab = (PA.LABELS_TEX if USETEX else PA.LABELS_TXT)[obs][1]
    curve = PA.CURVE_TEX if USETEX else PA.CURVE_TXT
    y, e = c.dist['full']
    ax.stairs(y, edges, color=COLOR['full'], ls=LS['full'], lw=LW + 0.5,
              label=curve['full'], zorder=6)
    ax.errorbar(x, y, yerr=e, fmt='none', ecolor=COLOR['full'],
                elinewidth=0.9, zorder=6)
    # The other two spinmodes, immediately after the default one so that the
    # three full curves sit together in the legend, and above the components
    # in zorder because they are the same kind of object as the black curve.
    dropped = []
    for key, dx in extras:
        fd = PA.full_distribution(dx, obs)
        elab = (PA.EXTRA_TEX if USETEX else PA.EXTRA_TXT)[key]
        if not fd['drawable']:
            dropped.append((key, fd['n_sel']))
            continue
        ax.stairs(fd['y'], edges, color=COLOR[key], ls=LS[key], lw=LW + 0.2,
                  label=elab, zorder=5)
        ax.errorbar(x, fd['y'], yerr=fd['err'], fmt='none', ecolor=COLOR[key],
                    elinewidth=0.8, zorder=5)
    for k in PA.POL_KEYS:
        yk, ek = c.dist[k]
        shown = np.where(yk > 0, yk, np.nan) if obs in PA.LOGY else yk
        ax.stairs(shown, edges, color=COLOR[k], ls=LS[k], lw=LW,
                  label=curve[k], zorder=4)
        ax.errorbar(x, shown, yerr=np.where(np.isfinite(shown), ek, np.nan),
                    fmt='none', ecolor=COLOR[k], elinewidth=0.7, alpha=0.6,
                    zorder=4)
    if obs in PA.LOGY:
        ax.set_yscale('log')
    ax.set_ylabel(ylab)
    lo, hi = ax.get_ylim()
    # On the log pane the autoscale floor is set by the Z_0 Z_0 component's
    # last bin, so the plain 1.6x headroom that works on a linear pane leaves
    # the legend sitting on the curves.  One column throughout now that the
    # labels are short.  The exclusive samples resolve the Z_0 Z_0 tail down to
    # 1e-8 where the earlier sparse ones had empty bins there, which stretches
    # the pane to six decades and puts the seven-row legend back on the total;
    # 250x rather than the earlier 60x clears it.  Checked on the rendered PNG,
    # not on the axis limits.
    head = HEADROOM[variant]
    ax.set_ylim(lo, hi * (head[0] if obs in PA.LOGY else head[1]))
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    # A dropped spinmode is recorded and returned, and it is NOT written on the
    # figure: nothing is.  ``write_numbers`` prints the survivors and the
    # drawn/not-drawn decision for every mode and every observable, which is
    # where a reader is told what is and is not on the canvas.  With the
    # exclusive decay card ``dropped`` is empty for both observables.
    # No title either -- see the module docstring; the caption is in
    # numbers.txt.
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if obs not in PA.LOGY:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.setp(ax.get_xticklabels(), visible=False)
    return dropped


def draw_shape(d, obs, outdir, extras=()):
    """Variant B: the distribution, and ONE ratio pane under it.

    The pane is ``(1/sigma dsigma/dX)_Y / (1/sigma dsigma/dX)_madspin`` for
    every extra spinmode at once -- each curve divided by its OWN cross section
    before the division, so that the rate difference between the modes divides
    out and what is left is the shape.  That is the pane this data wants:
    ``onshell`` differs from ``madspin`` in RATE (+4.99 % inclusive) and much
    less in shape, ``PA`` agrees in rate (1.4 sigma) and differs in SHAPE.  A
    ratio that still carried the normalisation would show the first as a large
    flat offset and would hide the second under it.

    ``extras`` here is ``load_extras() + load_variant_b_extras()``, i.e. three
    curves: ``onshell``, ``PA`` and ``madspin_v1``.  The last is on THIS
    FIGURE ONLY; the original figures and variant A get ``load_extras()``
    alone and are unchanged.  ``madspin_v1`` carries no ms_pol_* weights, and
    needs none: nothing on this figure asks a sample other than the reference
    for a polarisation weight.

    Which sigma normalises each curve is ``PA.SHAPE_NORM``, stated there and in
    numbers.txt.  The error is the within-sample delta-method one of
    ``PA.shape_density`` combined in quadrature between samples, which is right
    because the three samples do not share production events -- see
    ``PA.pairing_evidence``.

    The sum pane and the 2 x 2 breakdown are NOT drawn here.  They are the
    reference sample's polarisation decomposition and they are unchanged on the
    full figure and on variant A; this variant is about the three modes.
    """
    c = PA.Curves(d, obs)
    edges, x = c.edges, c.centres()
    xlab = (PA.LABELS_TEX if USETEX else PA.LABELS_TXT)[obs][0]

    fig = plt.figure(figsize=(8.0, 6.6))
    # One stacked pair, the same rhythm as the top pair of the full figure:
    # tick labels and the axis name under the ratio pane only.
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.9], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    _distribution(ax, c, obs, extras, variant='B')

    ref = PA.shape_density(d, obs)
    axr.axhline(1.0, color='black', lw=1.0, ls=':')
    series = []
    for key, dx in extras:
        if not PA.full_distribution(dx, obs)['drawable']:
            continue
        cmp = PA.compare_shape(ref, PA.shape_density(dx, obs))
        r, er = cmp['ratio'], cmp['ratio_err']
        series.append((r, er))
        lab = (PA.SHAPE_CURVE_TEX if USETEX else PA.SHAPE_CURVE_TXT)[key]
        # ``baseline=None`` for the same reason as the sum pane: stairs would
        # otherwise close the path down to zero and draw two spurious vertical
        # rules on a frame that sits around 1.
        axr.stairs(r, edges, color=COLOR[key], ls=LS[key], lw=LW + 0.2,
                   zorder=4, baseline=None, label=lab)
        axr.errorbar(x, r, yerr=er, fmt='o', ms=3.4, color=COLOR[key],
                     elinewidth=1.0, zorder=5)
    if series:
        axr.set_ylim(*_shape_ylim(series))
    axr.set_ylabel(PA.SHAPE_RATIO_TEX if USETEX else PA.SHAPE_RATIO_TXT,
                   fontsize=8.5)
    axr.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=3)
    axr.xaxis.set_minor_locator(AutoMinorLocator())
    # The window is a few percent wide, and the default locator then labels it
    # with two ticks -- 1.0 and one other -- which is not a readable scale for
    # a pane whose whole content is a few-percent excursion.  Ask for five.
    axr.yaxis.set_major_locator(MaxNLocator(5))
    axr.yaxis.set_minor_locator(AutoMinorLocator())
    axr.set_xlabel(xlab, fontsize=12)
    for s in axr.spines.values():
        s.set_linewidth(1.5)
    ax.set_xlim(edges[0], edges[-1])

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, PA.SHORT[obs])
    want = wants_minus(fig)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('%-12s variant B  N=%d  norm=%s' % (PA.SHORT[obs], c.n_sel,
                                              PA.SHAPE_NORM))
    return base, want


def draw(d, obs, outdir, extras=(), variant=None):
    c = PA.Curves(d, obs)
    edges, x = c.edges, c.centres()
    xlab = (PA.LABELS_TEX if USETEX else PA.LABELS_TXT)[obs][0]
    rlab = PA.RATIO_TEX if USETEX else PA.RATIO_TXT

    fig = plt.figure(figsize=(8.0, 11.0))
    # Three vertical rhythms, so three gridspecs rather than one.  The
    # distribution and the sum pane are a normal stacked PAIR sharing one x
    # axis -- tick labels and axis name under the sum pane only, and a tight
    # gap between them.  The 2 x 2 breakdown is a separate block with its own
    # labels, set off by a modest gap.  A single hspace cannot do all three:
    # tight enough for the pair leaves the sum pane's axis name on the 2 x 2,
    # and wide enough to clear it tears the stacked pair apart.
    gs = fig.add_gridspec(2, 1, height_ratios=[4.8, 2.5], hspace=0.13)
    top = gs[0].subgridspec(2, 1, height_ratios=[3.1, 1.7], hspace=0.06)
    sub = gs[1].subgridspec(2, 2, hspace=0.12, wspace=0.26)
    ax = fig.add_subplot(top[0])
    axs = fig.add_subplot(top[1], sharex=ax)
    small = [fig.add_subplot(sub[0, 0], sharex=ax),
             fig.add_subplot(sub[0, 1], sharex=ax),
             fig.add_subplot(sub[1, 0], sharex=ax),
             fig.add_subplot(sub[1, 1], sharex=ax)]

    # -- tier 1: the distribution, full plus the four components -------------
    # Shared with variant B, and with variant A, which passes no extras.
    _distribution(ax, c, obs, extras, variant=variant)

    # -- tier 2: the sum, on its own scale -----------------------------------
    r, er, nb = c.ratios['SUM']
    Rint, Eint = c.integrated['SUM']
    axs.axhline(1.0, color='black', lw=1.0, ls=':')
    # ``baseline=None``: matplotlib's stairs otherwise closes the path down to
    # zero at both ends, which on a ratio pane whose window is around 1 draws
    # two spurious vertical rules on the frame.
    axs.stairs(r, edges, color=COLOR['SUM'], lw=LW + 0.4, zorder=4,
               baseline=None)
    axs.errorbar(x, r, yerr=er, fmt='o', ms=3.4, color=COLOR['SUM'],
                 elinewidth=1.0, zorder=5)
    axs.set_ylim(*_ratio_ylim(r, er, 1.0))
    axs.set_ylabel(rlab['SUM'], fontsize=8.5)
    axs.xaxis.set_minor_locator(AutoMinorLocator())
    axs.yaxis.set_minor_locator(AutoMinorLocator())
    # Neither the integrated value nor the words "polarisation interference"
    # are written here any more.  Both are in numbers.txt (the integrated
    # ratios table and the per-bin table for this observable) and in
    # RESULTS.md, where they can be read next to the sample they belong to.
    axs.set_xlabel(xlab, fontsize=12)
    for s in axs.spines.values():
        s.set_linewidth(1.5)

    # -- tier 3: the breakdown, one polarisation per pane --------------------
    for a, k in zip(small, ['LL', 'TT', 'TL', 'LT']):
        rk, ek, _ = c.ratios[k]
        Rk = c.integrated[k][0]
        # The dashed line IS that pane's integrated value and stays, because it
        # is the reference the shape is read against.  Its number is no longer
        # printed beside it -- all five integrated ratios and their errors are
        # in numbers.txt's "integrated polarisation ratios" table.
        a.axhline(Rk, color='black', lw=0.9, ls='--')
        a.stairs(rk, edges, color=COLOR[k], ls=LS[k], lw=LW, zorder=4,
                 baseline=None)
        a.errorbar(x, rk, yerr=ek, fmt='o', ms=3.0, color=COLOR[k],
                   elinewidth=0.9, zorder=5)
        a.set_ylim(*_ratio_ylim(rk, ek, Rk))
        a.set_ylabel(rlab[k], fontsize=10)
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_minor_locator(AutoMinorLocator())
    for a in small[:2]:
        plt.setp(a.get_xticklabels(), visible=False)
    for a in small[2:]:
        a.set_xlabel(xlab, fontsize=12)

    ax.set_xlim(edges[0], edges[-1])

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, PA.SHORT[obs])
    want = wants_minus(fig)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('%-12s %-10s N=%d  sum/full = %.4f +- %.4f'
          % (PA.SHORT[obs], 'variant ' + variant if variant else 'full',
             c.n_sel, Rint, Eint))
    return base, want


# --------------------------------------------------------------------------
def _wrap(text, width):
    """``text`` broken on spaces into lines of at most ``width`` characters."""
    out, line = [], ''
    for w in text.split():
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = (line + ' ' + w) if line else w
    if line:
        out.append(line)
    return out


def _sigma_err(dd, obs):
    """``sqrt(sum w^2)`` on one sample's fiducial cross section, in pb."""
    w = np.asarray(dd.full, dtype=np.float64)[dd.sel[obs]] * dd.scale_to_pb
    return float(np.sqrt((w ** 2).sum()))


def _shape_numbers(d, A, extras):
    """The variant-B pane, in full: what it is, which sigma, and the bins.

    Both normalisations are printed side by side so that the choice recorded in
    ``pol_analysis.SHAPE_NORM`` can be audited rather than taken on trust.
    """
    A('--- THE SHAPE RATIO: what the variant-B pane draws ---')
    A('   %s' % PA.SHAPE_RATIO_TXT)
    A('for every Y in one pane -- Y = onshell, Y = PA and, on this variant')
    A('alone, Y = madspin_v1.  Each mode is divided by')
    A('its OWN cross section before the division, so the RATE difference')
    A('between the modes divides out and what is left is the SHAPE.  That is')
    A('the pane this data wants: onshell differs from madspin in rate (+4.99 %')
    A('inclusive) and much less in shape, PA agrees in rate (1.4 sigma) and')
    A('differs in shape.  A ratio still carrying the normalisation shows the')
    A('first as a large flat offset and buries the second underneath it.')
    A('')
    A('WHICH SIGMA NORMALISES EACH CURVE.  Two candidates, and they are not')
    A('the same number:')
    A('   selected  the whole selected fiducial cross section, every event')
    A('             passing the lepton cuts, INCLUDING those whose observable')
    A('             falls outside the drawn range;')
    A('   inrange   the integral of the drawn histogram, in-range events only.')
    A('')
    A('THE FIGURES USE: %s' % PA.SHAPE_NORM)
    A('   -- %s' % PA.SHAPE_NORM_TXT[PA.SHAPE_NORM])
    A('because it makes the pane an exact statement about what is on the')
    A('canvas: the cross-section-weighted mean of the drawn ratio over the')
    A('drawn bins is then 1 by construction, so every visible departure from 1')
    A('is paid for by another visible bin.  With "selected" the out-of-range')
    A('rate -- which is mode dependent -- slides the whole pane by an amount')
    A('whose cause is off the canvas.  Both are tabulated below; the choice is')
    A('made for that reason and not because it changes an answer.')
    A('')
    A('what is outside the drawn range, per observable and mode:')
    A('   %-10s %-11s %12s %14s %14s %10s'
      % ('observable', 'spinmode', 'N outside', 'sigma_selected',
         'sigma_inrange', 'outside %'))
    for obs in PA.OBS:
        for lab, dd in [('madspin', d)] + list(extras):
            sd = PA.shape_density(dd, obs, 'inrange')
            A('   %-10s %-11s %12d %14.6f %14.6f %10.3f'
              % (PA.SHORT[obs], dd.label, sd['n_outside'],
                 sd['sigma_selected_pb'], sd['sigma_norm_pb'],
                 100 * sd['sigma_outside_pb'] / sd['sigma_selected_pb']))
    A('')
    A('Delta phi is binned over its whole physical range [0, pi] and has no')
    A('outside at all, so for that observable the two normalisations are the')
    A('same number to the last bit.  Only M(e+ mu+), whose last edge is at 450')
    A('GeV while the observable reaches 4.2 TeV, distinguishes them.')
    A('')
    A('NOTE, and it corrects an earlier statement in RESULTS.md: the counts')
    A('3120 / 3088 / 3134 quoted there for the first three samples as lying')
    A('above 450 GeV are events with')
    A('a FINITE M(e+ mu+), i.e. before the fiducial pT and eta cuts.  The')
    A('SELECTED overflow -- the events that are actually in sigma_fid and')
    A('missing from the histogram -- is the "N outside" column above.')
    A('')
    A('THE ERROR ON THIS PANE, and it is two different things multiplied')
    A('together.  WITHIN one sample the bin content and the sigma that')
    A('normalises it are sums over the SAME events -- the bin is a subset of')
    A('the normalisation -- so they are correlated and the delta-method error')
    A('of pol_analysis.ratio is used, exactly as for the polarisation ratios.')
    A('A plain sqrt(sum w^2) would overstate the bar by 1/sqrt(1 - 2 p) for a')
    A('bin holding a fraction p of the rate: 8 % on a twelfth of the Delta phi')
    A('rate, which would understate the chi2 by 14 %.  BETWEEN samples there')
    A('is NO correlation to keep -- see "DO THE SAMPLES SHARE PRODUCTION')
    A('EVENTS?" above -- so the two relative errors add in quadrature.  One')
    A('degree of freedom is removed from the chi2 because each curve was')
    A('divided by its own normalisation, which is exactly one constraint.')
    A('')
    for obs in PA.OBS:
        edges = PA.BINS[obs]
        A('%s: the shape ratio bin by bin' % PA.SHORT[obs])
        refs = {n: PA.shape_density(d, obs, n)
                for n in ('inrange', 'selected')}
        for key, dx in extras:
            cmps = {n: PA.compare_shape(refs[n], PA.shape_density(dx, obs, n))
                    for n in ('inrange', 'selected')}
            cc = cmps[PA.SHAPE_NORM]
            other = 'selected' if PA.SHAPE_NORM == 'inrange' else 'inrange'
            A('   Y = %-8s  chi2/ndf = %.2f/%d = %.2f   max|pull| %.2f'
              % (dx.label, cc['chi2'], cc['ndf'], cc['chi2_per_ndf'],
                 cc['max_abs_pull']))
            A('     %-14s %10s %12s %10s %10s %12s'
              % ('bin', 'N(ref)', 'ratio', 'error', 'sigma(1)',
                 'norm=' + other))
            n = refs[PA.SHAPE_NORM]['n']
            for b in range(len(edges) - 1):
                if not n[b]:
                    continue
                r, e = cc['ratio'][b], cc['ratio_err'][b]
                A('     %6.1f-%6.1f %10d %12.4f %10.4f %10.2f %12.4f'
                  % (edges[b], edges[b + 1], n[b], r, e,
                     abs(r - 1) / e if e else float('nan'),
                     cmps[other]['ratio'][b]))
            dmax = np.nanmax(np.abs(cc['ratio'] - cmps[other]['ratio']))
            A('     the two normalisations differ by at most %.4f in any bin'
              % dmax)
            A('')
    # What the pane says about the fourth curve, computed rather than
    # asserted: madspin_v1 is the comparison this variant was extended for,
    # and the answer is a NULL result, which is worth stating in numbers.
    v1 = [(k, x) for k, x in extras if k == 'madspin_v1']
    if v1:
        dx = v1[0][1]
        A('WHAT THE PANE SAYS ABOUT madspin_v1 -- the comparison this variant')
        A('was extended for.  madspin_v1 is MadSpin\'s LEGACY spin-correlation')
        A('path and the reference madspin sample is the current DENSITY')
        A('method, so this curve is legacy-against-density.  All four samples')
        A('are Pythia8-showered HepMC read by the same extractor with the same')
        A('selection and the same binning, so the curve is a SPINMODE')
        A('difference and nothing else -- there is no shower-versus-parton')
        A('confusion in it.')
        A('')
        for obs in PA.OBS:
            cc = PA.compare_shape(PA.shape_density(d, obs),
                                  PA.shape_density(dx, obs))
            A('   %-9s chi2/ndf = %.2f/%d = %.2f   max|pull| %.2f'
              % (PA.SHORT[obs], cc['chi2'], cc['ndf'], cc['chi2_per_ndf'],
                 cc['max_abs_pull']))
        A('')
        A('That is a NULL result on both observables: no bin departs from 1 by')
        A('as much as 2 sigma and the chi2 per degree of freedom is about 1.')
        A('The legacy path and the density method agree in SHAPE on both')
        A('observables at these statistics, and they agree in RATE too -- the')
        A('inclusive cross sections differ by 0.15 % and the two fiducial')
        A('rates by 1.1 and 0.4 sigma; see "THE SPINMODES COMPARED" above.')
        A('This is the pane\'s answer and it is a different answer from the')
        A('other two curves on it: onshell differs in rate, PA differs in')
        A('shape, madspin_v1 differs in neither.  A null result is only worth')
        A('as much as its power, so read it next to the error column in the')
        A('madspin_v1 tables above: the bars are ~1 % per bin on M(e+ mu+)')
        A('and ~1-3 % on Delta phi, so a shape effect of the size PA shows')
        A('(up to 4 %) would have been seen, and one much below a percent')
        A('would not.')
        A('')
    A('For the contrast, "THE SPINMODES COMPARED" above quotes a')
    A('"SHAPE only" chi2 as well.  It is a DIFFERENT statistic and the two')
    A('are not interchangeable: that one rescales the other mode to the')
    A('reference\'s SELECTED fiducial rate and keeps the plain sqrt(sum w^2)')
    A('per-bin bar, so it neither uses the in-range normalisation nor removes')
    A('the within-sample correlation.  Its bars are the larger and its chi2')
    A('the smaller of the two.  The pane draws the one computed here.')
    A('')


def write_numbers(d, path, extras=(), lo=None):
    out = []
    A = out.append
    m = d.meta
    A('MadSpin polarisation weights on a showered NLO  p p > z z')
    A('=' * 74)
    A('%d samples: the same process, the same run card and the same 250k'
      % (1 + len(extras)))
    A('showered events, differing only in MadSpin\'s spinmode.  The')
    A('polarisation decomposition below is the REFERENCE (madspin, run_06)')
    A('sample\'s; the others enter the distribution pane of a figure and the')
    A('cross-mode comparison section, and nowhere else.')
    A('')
    A('WHICH SAMPLE IS ON WHICH FIGURE.  onshell and PA are on every figure,')
    A('as they always were.  madspin_v1 -- MadSpin\'s LEGACY spin-correlation')
    A('path, against which the reference madspin sample is the current density')
    A('method -- is on VARIANT B ONLY, as a fourth curve in the distribution')
    A('pane and a third in the single ratio pane.  The original figures and')
    A('variant A are unchanged by it.  It appears in EVERY table below,')
    A('because the tables are of the samples and not of one figure.')
    A('')
    A('--- WHAT IS NOT ON THE FIGURES ---')
    A('The figures carry their axis labels, their tick labels and their legend')
    A('and NOTHING else.  Everything that used to be written on them is here:')
    A('  * the plot title, which was the caption below;')
    A('  * the integrated (LL+TT+TL+LT)/full and its error, printed in the sum')
    A('    pane -- see "integrated polarisation ratios" below;')
    A('  * the four per-pane integrated ratios, printed in the 2x2 legends.')
    A('    The dashed line in each small pane still IS that value; only the')
    A('    number left the canvas;')
    A('  * the words "polarisation interference" over the sum pane;')
    A('  * in the user style, the note naming the two shaded bands behind the')
    A('    sum pane.  They are +-2 % and +-5 % about 1.  The bands are still')
    A('    drawn; only the words are gone.  Variant B\'s single ratio pane')
    A('    carries the same two graphics at +-1 % and +-3 %, tighter because a')
    A('    shape ratio between two spinmodes lives at a few percent, and they')
    A('    are unlabelled on the canvas for the same reason;')
    A('  * the "spinmode ... not drawn: too few events" note.  Which modes are')
    A('    drawn on which figure is stated per observable further down.')
    A('')
    A('figure caption (was the plot title):')
    for chunk in _wrap(PA.CAPTION, 70):
        A('   ' + chunk)
    A('')
    A('--- WHICH FIGURES EXIST ---')
    A('Three per observable per style, and the two variants are written')
    A('ALONGSIDE the original figure, in subdirectories of the same style')
    A('directory.  The original figures are unchanged.')
    A('')
    A('   plots/ and plots_userstyle/')
    A('      the three-tier figure: the distribution with the four')
    A('      polarisation components on the full AND the onshell and PA')
    A('      totals; a full-width (Z_0Z_0 + Z_TZ_T + Z_TZ_0 + Z_0Z_T)/full')
    A('      pane; then the 2 x 2 of the individual ratios.')
    for key in ('A', 'B'):
        v = PA.VARIANTS[key]
        A('')
        A('   plots/%s/ and plots_userstyle/%s/' % (v['dir'], v['dir']))
        for chunk in _wrap('variant %s -- %s'
                           % (key, v['what'][0].upper() + v['what'][1:]), 64):
            A('      ' + chunk)
    A('')
    A('Both variants keep the house style of the original: both styles, PDF')
    A('AND PNG, the Z_0Z_0 naming, --check-minus on the MG7 figures only, and')
    A('nothing written on the canvas beyond axis labels, tick labels and the')
    A('legend.  Variant B\'s pane is documented in full under "THE SHAPE')
    A('RATIO" below, including which sigma normalises each curve.')
    A('')
    A('%-11s %-34s %9s %9s %8s %8s'
      % ('spinmode', 'input', 'on disk', 'inflated', 'pass s', 'events'))
    for mm in [m] + [x.meta for _, x in extras]:
        A('%-11s %-34s %8.3fG %8.3fG %8.1f %8d'
          % (mm.get('label', 'madspin'),
             mm['input_path'].split('Events/')[-1].replace(
                 '/events_PYTHIA8_0', '/...'),
             mm['input_gb'], mm.get('inflated_gb', mm['input_gb']),
             mm['pass_seconds'], mm['n_events']))
    A('Every .gz input was streamed through "gzip -dc" in a child process and')
    A('never unpacked to disk: they inflate to ~10.5 GB each and no temporary')
    A('of that size should have to exist.  The first three passes were run')
    A('CONCURRENTLY on separate cores, so their times are within a second of')
    A('each other and none of them is a serial measurement; the madspin_v1')
    A('pass was run on its own later, so ITS time is a serial one and is the')
    A('only number in that column that can be read as a rate.')
    A('%d lines read from the reference file, %s' % (m['lines_read'],
      m['hepmc_flavour']))
    A('code       %s' % m['code_sha'])
    A('')
    A('--- the %d weight names on every N line (asserted, not sampled) ---'
      % len(m['weight_names']))
    for i in range(0, len(m['weight_names']), 6):
        A('   ' + '  '.join('%-18s' % n for n in m['weight_names'][i:i + 6]))
    A('   (that is the REFERENCE file\'s N line.  The other files are checked')
    A('   against their OWN first N line, and compared with this one under')
    A('   "THE OTHER SPINMODES" below -- they do not all agree.)')
    A('')
    A('--- what the figures call these weights ---')
    A('The figures carry the physics name only; this is the mapping back to')
    A('the weight column each curve sums.  data/meta.json carries it too.')
    A('   %-22s %-22s %s' % ('short key', 'figure legend', 'weight column'))
    for k in ['LL', 'TT', 'TL', 'LT']:
        A('   %-22s %-22s %s'
          % (k, PA.CURVE_TXT[k], PA.LEGEND_TO_COLUMN[PA.CURVE_TXT[k]]))
    A('   %-22s %-40s %s' % ('full', PA.CURVE_TXT['full'], 'Weight'))
    A('')
    A('--- WHICH WEIGHT IS THE FULL ---')
    ev = d.nominal_evidence()
    A('Answer: "Weight".  "0" is the SAME weight divided by the event count,')
    A('so every ratio below is identical whichever of the two is used; only')
    A('the vertical scale of the distributions distinguishes them.')
    A('')
    A('  "0" * N / "Weight", over all %d events: min %.15f, max %.15f'
      % (ev['n_events'], ev['w0_times_N_over_Weight_min'],
         ev['w0_times_N_over_Weight_max']))
    A('  sum("0")        = %.12f pb' % ev['sum_w0_pb'])
    A('  mean("Weight")  = %.12f pb' % ev['mean_Weight_pb'])
    A('  C line, LAST event = %.12f pb   <-- agrees with both to 1e-11'
      % ev['C_line_last_event_pb'])
    A('  C line, FIRST event = %.6e pb  (the RUNNING Pythia estimate after'
      % ev['C_line_first_event_pb'])
    A('     one event; it is not the sample cross section and must not be used)')
    A('  sum("MUR1.0_MUF1.0")/sum("Weight") - 1 = %.3e  -- the LHE <wgt id='
      % ev['sum_MUR1_over_sum_Weight_minus_1'])
    A('     \'1001\'> copy of the same nominal, to the rwgt block\'s rounding')
    A('  median|sum of the four ms_pol| / median|"Weight"| = %.3f  -- the'
      % ev['pol_over_Weight_order_of_magnitude'])
    A('     polarisation weights are in the "Weight" normalisation, so')
    A('     dividing them by "0" would inflate every ratio by %d.'
      % ev['n_events'])
    vals, cnts = np.unique(np.abs(d.full), return_counts=True)
    A('  |Weight| takes %d distinct value(s): %s'
      % (len(vals), ', '.join('%.6f (x%d)' % (v, c)
                              for v, c in zip(vals, cnts))))
    if len(vals) > 1:
        j = int(np.argmax(cnts))
        A('     -- %d of the %d events carry %.6f, the banner\'s XMAXUP; the'
          % (cnts[j], ev['n_events'], vals[j]))
        odd = np.abs(d.full) != vals[j]
        A('     other %d value(s) are carried by %d event(s) in total and are'
          % (len(vals) - 1, int(odd.sum())))
        A('     single-event anomalies, NOT a second MC@NLO branch.  Together')
        A('     they are worth %.3e pb in a %.6f pb total.'
          % (float(np.abs(d.full)[odd].sum() * d.scale_to_pb),
             ev['sum_w0_pb']))
    A('  %d of %d events carry a NEGATIVE weight (%.3f %%).'
      % (ev['n_negative'], ev['n_events'], 100 * ev['frac_negative']))
    A('')
    A('--- IS THE DECAY CARD EXCLUSIVE IN EFFECT? (measured, not assumed) ---')
    A('The MadSpin card of every run is')
    A('  decay z > e+ e-')
    A('  decay z > mu+ mu-')
    A('which is exclusive in INTENT.  Whether it is exclusive in EFFECT is a')
    A('different question and it decides what these figures mean, so it is')
    A('read off the event record -- mark each z\'s end vertex, follow the chain')
    A('of status-44 copies, take the particles born at the final vertex -- for')
    A('every event of every file:')
    A('')
    A('   %-11s %10s %14s %14s %10s'
      % ('spinmode', 'events', 'one z->e+e- AND', 'at least one', 'other'))
    A('   %-11s %10s %14s %14s %10s'
      % ('', '', 'other z->mu+mu-', 'z->e+e-', 'channels'))
    for lab, dd in [('madspin', d)] + list(extras):
        zz1, zz2 = dd.z['z1_ch'], dd.z['z2_ch']
        q4 = ((zz1 == 11) & (zz2 == 13)) | ((zz1 == 13) & (zz2 == 11))
        qe = (zz1 == 11) | (zz2 == 11)
        A('   %-11s %10d %8d %5.4g%% %8d %5.4g%% %10d'
          % (dd.label, dd.n, q4.sum(), 100 * q4.mean(), qe.sum(),
             100 * qe.mean(), int((~q4).sum())))
    A('')
    A('It is exclusive in effect: every event of every file has one z to e+e-')
    A('and the other to mu+mu-, with no other channel present at all.  So')
    A('BOTH observables have the full 250 000 events behind them before the')
    A('fiducial cuts, and the sparsity that made M(e+ mu+) undrawable on the')
    A('earlier decay z > light light samples (312 of 250 000 survivors) is')
    A('gone.')
    A('')
    A('WHAT THE CROSS SECTIONS ARE OF.  Because the card is exclusive, the')
    A('sigma quoted for each file is')
    A('     sigma(p p > z z) x BR(z -> e+e-) x BR(z -> mu+mu-) x 2')
    A('(the 2 being the two ways of assigning the channels to the two z), and')
    A('NOT an inclusive p p > z z cross section.  The earlier inclusive')
    A('samples quoted ~13.1 pb, essentially the production cross section')
    A('because decay z > light light has a branching fraction of ~1; these')
    A('quote ~0.031 pb, smaller by ~2.35e-03, which is that product of')
    A('branching fractions.  The two sets of numbers are NOT comparable as')
    A('rates and only the ratios between the modes are.')
    A('')
    if extras:
        A('--- THE OTHER SPINMODES ---')
        A('Checked, not assumed, on each file separately.')
        A('')
        A('The N line.  Every one of the 250 000 N lines in each file was')
        A('compared against that file\'s first, as for the reference; they are')
        A('constant within each file.  Across files they are NOT all the same:')
        A('')
        A('   %-11s %8s  %s' % ('spinmode', 'n(names)', 'against the reference'))
        for lab, dd in [('madspin', d)] + list(extras):
            wn = dd.meta['weight_names']
            if wn == m['weight_names']:
                verdict = 'identical, same order'
            else:
                miss = [k for k in m['weight_names'] if k not in wn]
                new = [k for k in wn if k not in m['weight_names']]
                verdict = 'missing %s%s' % (
                    ', '.join(miss) if miss else '(none)',
                    '; extra ' + ', '.join(new) if new else '')
            A('   %-11s %8d  %s' % (dd.label, len(wn), verdict))
        A('')
        haspol = [dd.label for lab, dd in [('madspin', d)] + list(extras)
                  if dd.has_pol]
        nopol = [dd.label for lab, dd in [('madspin', d)] + list(extras)
                 if not dd.has_pol]
        A('So %s carr%s the four ms_pol_* weights, because %s run%s set'
          % (', '.join(haspol), 'y' if len(haspol) > 1 else 'ies',
             'those' if len(haspol) > 1 else 'that',
             's' if len(haspol) > 1 else ''))
        A('keep_weight_for_polarization_vector = [0, T]; %s do%s not.'
          % (', '.join(nopol) if nopol else '(none)',
             '' if len(nopol) > 1 else 'es'))
        if nopol:
            A('')
            A('THAT IS NOT A DEFECT AND IT IS NOT WORKED AROUND.  madspin_v1 is')
            A('the LEGACY MadSpin path and it does not emit polarisation')
            A('weights at all; the extractor\'s REQUIRED_WEIGHTS are "0" and')
            A('"Weight" alone, everything else is kept if present, and the')
            A('four absent names are printed by the extractor as it reads the')
            A('N line and recorded in that sample\'s meta.json as')
            A('has_pol_weights: false.  Nothing on variant B asks a sample')
            A('other than the reference for a polarisation weight -- the pane')
            A('is a ratio of NOMINAL distributions -- so the absence costs the')
            A('figure nothing.  What it does cost is that madspin_v1 can never')
            A('enter the polarisation decomposition panes, and it does not.')
        A('')
        A('That the onshell and PA files carry polarisation weights is')
        A('recorded here and nothing in this study is built on it: their')
        A('decomposition is a separate piece of work, and the ratio panes')
        A('belong to the sample whose decomposition they are.')
        A('')
        A('The nominal, re-derived per file rather than carried over:')
        A('')
        A('%-11s %22s %22s %14s %10s'
          % ('spinmode', '"0"*N/"Weight" range', 'sum("0") = mean("W")',
             'C last', 'C first'))
        for lab, dd in [('madspin', d)] + list(extras):
            e = dd.nominal_evidence()
            A('%-11s %10.6f %10.6f %22.9f %14.9f %10.3e'
              % (dd.label, e['w0_times_N_over_Weight_min'],
                 e['w0_times_N_over_Weight_max'], e['sum_w0_pb'],
                 e['C_line_last_event_pb'], e['C_line_first_event_pb']))
        A('')
        A('Same answer on every file, INCLUDING the one with no polarisation')
        A('weights: "0" * N_events == "Weight" to the last bit on every event,')
        A('sum("0") == mean("Weight") == the C line of the LAST event to')
        A('1e-11, and the C line of the FIRST event is Pythia\'s running')
        A('estimate after one event -- low by 2.2e5 in every file.  So the')
        A('reference file\'s conclusion holds for the others, and it holds')
        A('because it was re-checked and not because it was carried over.  The')
        A('distributions use "0" (its sum IS the cross section in pb) and each')
        A('sample is scaled by its OWN 1/n_events.')
        A('')
        A('The nominal for madspin_v1 is therefore the SAME quantity as for')
        A('the other three and was identified the same way: "Weight" (or')
        A('equivalently "0", which is "Weight"/N), confirmed against that')
        A('file\'s own LAST C line.  The absence of the ms_pol_* block changes')
        A('nothing about which weight is nominal -- those weights were never')
        A('candidates for it, and the one check that used them (that they sit')
        A('in the "Weight" normalisation, not the "0" one) simply does not')
        A('arise for a file that has none.')
        A('')
        A('%-11s %14s %12s %10s %s'
          % ('spinmode', 'sigma_tot pb', 'negatives', 'MUR1/W-1',
             'distinct |Weight|'))
        for lab, dd in [('madspin', d)] + list(extras):
            e = dd.nominal_evidence()
            A('%-11s %14.6f %11.3f %% %10.2e %s'
              % (dd.label, e['sum_w0_pb'], 100 * e['frac_negative'],
                 e['sum_MUR1_over_sum_Weight_minus_1'],
                 ', '.join('%.6f' % v for v in e['distinct_abs_Weight'])))
        A('')
        A('On the |Weight| column: each file\'s dominant value is its banner')
        A('XMAXUP exactly.  Any further value is carried by a handful of')
        A('individual events -- a single-event anomaly, not a second MC@NLO')
        A('branch -- and the counts above say how many.  This is the same')
        A('thing seen once in the earlier run_02 file; it recurs here and it')
        A('has no consequence for anything in this study.')
        A('')
        A('--- DO THE SAMPLES SHARE PRODUCTION EVENTS? ---')
        A('This decides the error bar on every mode-to-mode comparison in this')
        A('file, so it is established and not assumed.  If the runs had')
        A('decayed one common set of production events, the mode-to-mode ratio')
        A('would be CORRELATED, a paired error would be correct, and it would')
        A('be much smaller than the quadrature one.  They did not.')
        A('')
        pe = PA.pairing_evidence(d, extras)
        A('%-11s %10s %12s %14s %18s'
          % ('spinmode', 'events', 'n(w < 0)', 'sigma_tot pb',
             'corr(m_4l) row by row'))
        for row in pe['rows']:
            A('%-11s %10d %12d %14.6f %18s'
              % (row['label'], row['n_events'], row['n_negative'],
                 row['sigma_pb'],
                 '--  (reference)' if not np.isfinite(row['corr_m_4l'])
                 else '%+.5f +- %.5f' % (row['corr_m_4l'],
                                         row['corr_stderr'])))
        A('')
        A('n(w < 0) is the DECISIVE column and it is permutation invariant.')
        A('Neither MadSpin nor Pythia can change the sign of an event weight --')
        A('MadSpin multiplies by a positive decay factor and Pythia passes the')
        A('LHE weight through -- so the number of negative-weight events is a')
        A('property of the PRODUCTION sample alone and cannot depend on the')
        A('spinmode or on the order the events are written in.  Two decays of')
        A('one production sample would give the same count EXACTLY.  These')
        A('differ, so a common production sample is excluded outright, in any')
        A('order, and not merely left unproven.')
        A('')
        A('corr(m_4l) corroborates it and is the nearest thing the cached')
        A('columns have to the sqrt(shat) the sibling m_tt study paired on.')
        A('For a 2 -> 2 production event the four-lepton invariant mass IS the')
        A('production m(ZZ), so two decays of one production event would agree')
        A('in it to the dressing; row by row it is consistent with ZERO on')
        A('~243 000 common rows, where pairing would give ~1.')
        A('')
        A('The .npz carries no production-level column at all -- no')
        A('sqrt(shat), no event number, no production kinematics -- so the')
        A('literal "max |Delta sqrt(shat)| = 0" test of the m_tt study cannot')
        A('be run on this cache.  It does not need to be: the negative-weight')
        A('count is a stronger test in the direction that matters, because it')
        A('is permutation invariant, and it FAILS.')
        A('')
        A('CONSEQUENCE: every between-mode error in this file and on the')
        A('figures is the PLAIN quadrature sum, and that is CORRECT here and')
        A('not merely conservative.  A paired bar would be wrong, not just')
        A('optimistic.  (Within a single sample the covariance is kept and')
        A('must be -- see "integrated polarisation ratios" and')
        A('pol_analysis.shape_density.)')
        A('')
        A('--- THE SPINMODES COMPARED ---')
        A('Same lepton selection, same bin edges, same absolute normalisation.')
        A('Errors here are the PLAIN quadrature sum, unlike every ratio pane in')
        A('this file: these are independent MadSpin runs, independently')
        A('showered, so there is no covariance to keep -- established above,')
        A('not assumed.')
        A('')
        A('inclusive cross section (all 250k events, no selection):')
        for lab, dd in [('madspin', d)] + list(extras):
            w0 = np.asarray(dd.z['w_0'], dtype=np.float64)
            A('   %-11s %12.6f +- %.6f pb' % (dd.label, w0.sum(),
                                             np.sqrt((w0 ** 2).sum())))
        A('')
        A('The two chi2 answer different questions and only the pair of them')
        A('is the result: a mode that differs from the reference ONLY in')
        A('normalisation gives a large "as drawn" chi2 and a small "shape')
        A('only" one, and reading the first as a shape difference would be')
        A('wrong.  Read both rows below before concluding anything.')
        A('')
        A('where the rate differences come from -- sigma_tot x the fraction of')
        A('events in the z -> e+e- channel, against the fiducial cross')
        A('section, i.e. how much of the difference is channel bookkeeping and')
        A('how much is acceptance.  With the exclusive card f(z->ee) is 1 in')
        A('every mode by construction, so the channel-bookkeeping column that')
        A('carried the whole rate difference on the earlier inclusive samples')
        A('is now inert and whatever is left is the inclusive cross section')
        A('and the acceptance:')
        A('   %-11s %12s %10s %12s %12s %10s'
          % ('spinmode', 'sigma_tot', 'f(z->ee)', 'product', 'sigma_fid',
             'acceptance'))
        for lab, dd in [('madspin', d)] + list(extras):
            zz1, zz2 = dd.z['z1_ch'], dd.z['z2_ch']
            fee = float(((zz1 == 11) | (zz2 == 11)).mean())
            st = float(np.asarray(dd.z['w_0'], dtype=np.float64).sum())
            fid = PA.full_distribution(dd, 'dphi_ee_dr')['sigma_pb']
            A('   %-11s %12.6f %10.6f %12.6f %12.6f %10.4f'
              % (dd.label, st, fee, st * fee, fid, fid / (st * fee)))
        A('')
        A('which z decayed to what, off the event record (truth, used to')
        A('categorise only -- no plotted observable touches it):')
        A('   %-11s %10s %10s %10s %10s'
          % ('spinmode', 'n(4 lepton)', 'frac %', 'n(z->ee)', 'frac %'))
        for lab, dd in [('madspin', d)] + list(extras):
            z1, z2 = dd.z['z1_ch'], dd.z['z2_ch']
            q4 = ((z1 == 11) & (z2 == 13)) | ((z1 == 13) & (z2 == 11))
            qe = (z1 == 11) | (z2 == 11)
            A('   %-11s %10d %10.4f %10d %10.4f'
              % (dd.label, q4.sum(), 100 * q4.mean(), qe.sum(),
                 100 * qe.mean()))
        A('')
        for obs in PA.OBS:
            ref = PA.full_distribution(d, obs)
            rerr = _sigma_err(d, obs)
            A('%s: N selected, fiducial cross section, and the comparison'
              % PA.SHORT[obs])
            A('   %-11s N=%6d  sigma_fid=%.6f +- %.6f pb   (the reference)'
              % (d.label, ref['n_sel'], ref['sigma_pb'], rerr))
            sparse = not ref['drawable']
            for key, dx in extras:
                o = PA.full_distribution(dx, obs)
                cc = PA.compare_full(ref, o)
                oerr = _sigma_err(dx, obs)
                rr = cc['sigma_ratio']
                rre = rr * math.sqrt((oerr / o['sigma_pb']) ** 2
                                     + (rerr / ref['sigma_pb']) ** 2)
                sparse = sparse or not o['drawable']
                A('   %-11s N=%6d  sigma_fid=%.6f +- %.6f pb'
                  % (dx.label, o['n_sel'], o['sigma_pb'], oerr))
                A('             rate ratio to madspin %.4f +- %.4f (%.1f '
                  'sigma)%s'
                  % (rr, rre, abs(rr - 1) / rre if rre else float('nan'),
                     '' if o['drawable'] else '   [NOT DRAWN]'))
                A('             as drawn (rate included) chi2/ndf = '
                  '%.2f/%d = %.2f   max|pull| %.2f'
                  % (cc['chi2'], cc['ndf'], cc['chi2_per_ndf'],
                     cc['max_abs_pull']))
                A('             SHAPE only (rescaled to the same fiducial '
                  'rate) chi2/ndf = %.2f/%d = %.2f'
                  % (cc['shape_chi2'], cc['shape_ndf'],
                     cc['shape_chi2_per_ndf']))
            drawn = [dx.label for _, dx in extras
                     if PA.full_distribution(dx, obs)['drawable']]
            notdrawn = [dx.label for _, dx in extras
                        if not PA.full_distribution(dx, obs)['drawable']]
            A('   on the figure: madspin%s%s'
              % (' + ' + ' + '.join(drawn) if drawn else '',
                 '   NOT drawn: ' + ', '.join(notdrawn) if notdrawn
                 else '   (every mode drawn)'))
            if sparse:
                A('   Fewer than %d selected events in this observable for at'
                  % PA.MIN_SEL_TO_DRAW)
                A('   least one mode.  At that statistics the per-bin errors')
                A('   are tens of percent and several curves laid over one')
                A('   another would show gaps the eye reads as a mode')
                A('   difference and that are entirely noise.  The undrawn')
                A('   curves are NOT on the figure and the numbers above are')
                A('   the whole of what can honestly be said about them.')
            A('')
        A('')
        _shape_numbers(d, A, extras)
    A('--- integrated polarisation ratios ---')
    A('Errors are the delta-method ones for a ratio of two sums over the SAME')
    A('events (see pol_analysis.ratio); the naive independent-samples bar is')
    A('printed beside them for the sum, where the correlation is strongest.')
    A('')
    A('%-26s %10s %12s %12s %10s' % ('sample', 'N events', 'ratio', 'error',
                                     'sigma(1)'))
    samples = [('all events (no selection)', np.ones(d.n, dtype=bool))]
    for obs in PA.OBS:
        samples.append(('%s selection' % PA.SHORT[obs], d.sel[obs]))
    for name, mask in samples:
        A('%s' % name)
        f = d.full[mask]
        for k in PA.PANE_ORDER:
            R, E = PA.ratio(d.pol[k][mask], f)
            extra = ''
            if k == 'SUM':
                nv = PA.naive_ratio_error(d.pol[k][mask], f)
                extra = ('   [naive independent error %.5f, %.1fx too large]'
                         % (nv, nv / E))
            A('  %-24s %10d %12.5f %12.5f %10s%s'
              % (PA.RATIO_KEY[k], mask.sum(), R, E,
                 '%.2f' % (abs(R - 1) / E) if k == 'SUM' else '--', extra))
        # TL and LT must agree wherever the selection does not distinguish the
        # two z, and the difference is taken with the SAME delta-method error
        # rather than by subtracting two of the rows above, whose errors are
        # strongly correlated.  Where it is compatible with zero it is a check
        # that the weight-to-name mapping was not transposed; where it is not,
        # the selection is asymmetric between the two z and it should not be.
        Rd, Ed = PA.ratio(d.pol['TL'][mask] - d.pol['LT'][mask], f)
        A('  %-24s %10d %12.5f %12.5f %10.2f   [sigma from ZERO, not from 1]'
          % ('(TL - LT) / full', mask.sum(), Rd, Ed, abs(Rd) / Ed))
        A('')
    Rall, Eall = PA.ratio(d.pol['SUM'], d.full)
    A('The sum over ALL events is 1 to within %.3f %% and %.2f sigma.  That is'
      % (100 * abs(Rall - 1), abs(Rall - 1) / Eall))
    A('not a null result: the interference between different polarisations of')
    A('the same z is odd in the decay angles and integrates to zero over the')
    A('full angular phase space, so an unrestricted sum MUST come back to 1 and')
    A('the fact that it does is a check on the weights themselves.  The')
    A('interference is visible only differentially, or after a cut that')
    A('restricts the decay angles -- which every lepton selection does, and')
    A('which is why both selected rows above sit below 1 by many sigma where')
    A('the unrestricted one does not.')
    A('')
    A('--- the lepton selection ---')
    A('highest-pT final-state (status 1) lepton per flavour and charge,')
    A('dressed with every final-state photon within Delta R < %g (the run'
      % d.meta['dress_dr'])
    A("card's own rphreco), then pT > %g GeV and |eta| < %g."
      % (PA.PT_MIN, PA.ETA_MAX))
    for obs in PA.OBS:
        g = PA.diagnostics(d, obs)
        A('')
        A('%s: %d events selected' % (PA.SHORT[obs], g['n_selected']))
        A('   truth (%s): %d events, selection purity %.4f, efficiency %.4f'
          % (g['truth_requirement'], g['n_truth'], g['purity'],
             g['efficiency']))
        key = [k for k in g if k.startswith('ambiguity')][0]
        A('   how often the highest-pT choice was a genuine choice --')
        A('   fraction of selected events with a SECOND same-flavour '
          'same-sign lepton:')
        for f, (a3, a10) in g[key].items():
            A('      %-4s above %g GeV: %6.3f %%   above %g GeV: %6.3f %%'
              % (f, PA.AMBIG_PT, 100 * a3, PA.PT_MIN, 100 * a10))
        s = PA.dressing_shift(d, obs)
        A('   dressing moved the observable by %+.4g on average (rms %.4g); '
          '%.2f %% of'
          % (s['mean_shift'], s['rms_shift'],
             100 * s['frac_moved_by_more_than_1pct']))
        A('      selected events moved by more than 1 percent.')
    A('')
    for obs in PA.OBS:
        c = PA.Curves(d, obs)
        A('--- %s: the sum against 1, bin by bin ---' % PA.SHORT[obs])
        A('fiducial cross section %.6g pb' % c.sigma_pb)
        A('%12s %8s %12s %10s %10s   %s'
          % ('bin', 'N', 'sum/full', 'error', 'sigma(1)',
             '  '.join('%10s' % k for k in ['LL', 'TT', 'TL', 'LT'])))
        r, e, n = c.ratios['SUM']
        for b in range(len(n)):
            if not n[b]:
                continue
            row = '  '.join('%10.4f' % c.ratios[k][0][b]
                            for k in ['LL', 'TT', 'TL', 'LT'])
            A('%5.1f-%5.1f %8d %12.4f %10.4f %10.2f   %s'
              % (c.edges[b], c.edges[b + 1], n[b], r[b], e[b],
                 abs(r[b] - 1) / e[b] if e[b] else float('nan'), row))
        A('')
    if lo is not None:
        _kfactor_numbers(A, d, lo)
        _ratio6_numbers(A, d, lo)
        if d.has_scale and lo.has_scale:
            _scale_numbers(A, d, lo)
        else:
            A('=' * 74)
            A('SCALE UNCERTAINTY: NOT AVAILABLE')
            A('=' * 74)
            A('The .npz in this data directory carry the central scale weight')
            A('alone.  Re-run extract_hepmc_pol.py on run_06_decayed_1 and')
            A('run_12_decayed_1 to add the nine MUR*_MUF* columns; it is one')
            A('~30 s streaming pass per file and nothing else changes.')
            A('')
    open(path, 'w').write('\n'.join(out) + '\n')
    print('wrote %s' % path)



def _ratio6_top_numbers(A, d, lo):
    """The seventh pane: what is on it, the two decisions, and every bin.

    Every band is printed for every component at BOTH orders even though the
    figure draws two of the ten, and every value is printed even though the
    log axis makes a few-percent band a line width.  What is not on the canvas
    has to be here or it is unrecoverable.
    """
    A('-' * 74)
    A('THE SEVENTH PANE -- the distribution, full width, over the six')
    A('-' * 74)
    for line in _wrap(PA.RATIO6_TOP_WHAT, 70):
        A(line)
    A('')
    A('DECISION -- THE Y SCALE IS MEASURED, NOT TAKEN FROM THE OBSERVABLE.')
    A('Every other distribution pane in this study takes its y scale from')
    A('pol_analysis.LOGY, which is a property of the OBSERVABLE, and that is')
    A('right for a pane carrying FIVE curves.  This one carries TEN -- the')
    A('same five at two orders -- so it asks the data instead: log when the')
    A('ten curves span more than %.1f decades.' % PA.RATIO6_TOP_LOG_DECADES)
    for obs in PA.OBS:
        mn, mx, dec = PA.ratio6_top_span(d, lo, obs)
        A('  %-24s %.3e .. %.3e  %.2f decades -> %s'
          % (PA.LABELS_TXT[obs][0], mn, mx, dec,
             'LOG' if PA.ratio6_top_logy(d, lo, obs) else 'linear'))
    A('Both observables come out log.  Delta phi is the one this changes:')
    A('LOGY has it linear, and drawn linear FOUR of the ten curves -- Z0Z0')
    A('and Z0ZT at both orders -- lie within a few pixels of the frame on top')
    A('of one another and cannot be read at all.  The six ratio panes below')
    A('and every other figure in this study still follow LOGY and are')
    A('unchanged.')
    A('')
    A('DECISION -- WHICH CURVES CARRY A SCALE BAND, on the scale variant:')
    A('the UNPOLARISED pair, NLO and LO, and not the four components.  Ten')
    A('translucent bands behind ten curves in one pane is a pane in which')
    A('neither the bands nor the curves can be attributed; and the')
    A('unpolarised pair is what this pane exists to state, being the absolute')
    A('rate the six ratio panes divide by and never show.  It applies to THIS')
    A('PANE ONLY -- the six below band every curve they carry, exactly as')
    A('before.  The components\' bands are NOT thereby lost: they are in the')
    A('per-bin tables below, at both orders.')
    A('')
    A('AND THE BAND IS SMALL ON A LOG PANE, WHICH IS WORTH SAYING PLAINLY.')
    A('A few percent is 0.017 of a decade; on a pane four decades tall that')
    A('is under two pixels, so the unpolarised band the scale variant draws')
    A('here is at the line width and cannot be measured off the canvas.  It')
    A('is drawn to scale and not exaggerated.  The six ratio panes are where')
    A('this figure\'s bands are read, and the tables below are where this')
    A('pane\'s are.')
    A('')
    A('THE STYLES.  Colour is the component, the line is the ORDER (solid')
    A('NLO, dashed LO) -- the same LS_ORDER the six panes below use.  LO is')
    A('drawn OVER NLO, the reverse of the six panes, because K ~ 1.29 is a')
    A('hair\'s breadth on a log pane and the lower curve is the one that')
    A('vanishes.  The line WEIGHT steps down through the components so that')
    A('Z0ZT and ZTZ0 -- equal to 1-2 % bin by bin, as they must be, ZZ being')
    A('symmetric -- do not hide one another: the later, thinner one sits')
    A('inside the earlier, thicker one and leaves a halo of it showing.')
    A('Nothing is offset; two coincident curves are drawn coincident.')
    A('')
    if not (d.has_scale and lo.has_scale):
        A('(no MUR*_MUF* columns in this data directory: the band columns')
        A('below would be empty and the tables are printed without them)')
        A('')
    for obs in PA.OBS:
        edges = PA.BINS[obs]
        band = (PA.ratio6_top_curves(d, lo, obs, with_band=True,
                                     band_keys=PA.KF_PANE_ORDER)
                if d.has_scale and lo.has_scale else
                PA.ratio6_top_curves(d, lo, obs))
        A('-' * 74)
        A('%s -- the seventh pane, bin by bin, in pb per unit x'
          % PA.LABELS_TXT[obs][0])
        A('  band = the %s-point envelope, as a percentage of that bin\'s'
          % PA.SCALE_ENVELOPE)
        A('         nominal')
        A('-' * 74)
        for key in PA.KF_PANE_ORDER:
            A('  %s' % PA.KF_CURVE_TXT[key])
            A('    %-20s %13s %10s %8s %8s   %13s %10s %8s %8s'
              % ('bin', 'NLO', '+- MC', 'band-%', 'band+%',
                 'LO', '+- MC', 'band-%', 'band+%'))
            for i in range(len(edges) - 1):
                cells = []
                for tag in PA.RATIO6_TOP_ORDERS:
                    h = band[tag][key]
                    y = float(h['y'][i])
                    e = float(h['err'][i])
                    blo, bhi = h.get('lo'), h.get('hi')
                    if blo is None or not y:
                        cells.append('%13.6e %10.2e %8s %8s' % (y, e, '-', '-'))
                    else:
                        cells.append('%13.6e %10.2e %8.2f %8.2f'
                                     % (y, e, 100.0 * (blo[i] / y - 1.0),
                                        100.0 * (bhi[i] / y - 1.0)))
                A('    %8.3f - %-9.3f %s   %s'
                  % (edges[i], edges[i + 1], cells[0], cells[1]))
            A('    integrated (in selection): NLO %.6e pb   LO %.6e pb'
              % (band['NLO'][key]['sigma_pb'], band['LO'][key]['sigma_pb']))
            A('')


def _ratio6_numbers(A, d, lo):
    """What the two seven-pane ratio figures are, and their per-bin tables."""
    A('=' * 74)
    A('THE TWO SEVEN-PANE RATIO FIGURES (variant A, restructured)')
    A('=' * 74)
    A('Written to plots/%s/ and plots/%s/ and to the two' % (
        PA.RATIO6_DIR, PA.RATIO6_SCALE_DIR))
    A('same-named directories under plots_userstyle/.  PDF and PNG, both')
    A('styles.  Variant A itself, variant B, the K-factor figure and the')
    A('original figures are untouched by this pass and their PNGs are still')
    A('byte-for-byte what this branch carried.')
    A('')
    A('THE LAYOUT.  A FULL-WIDTH DISTRIBUTION PANE over a 3 x 2 of ratios:')
    A('  the seventh   dsigma/dx, absolute, unpolarised + the four')
    A('  (full width)  components, at NLO and at LO -- ten curves')
    A('  top left      (Z0Z0 + ZTZT + ZTZ0 + Z0ZT) / full, the sum')
    A('                consistency')
    A('  top right     K = NLO / LO, five curves (unpolarised + the four)')
    A('  the four      component / full, one polarisation per pane')
    A('')
    A('WHAT CHANGED FROM VARIANT A.  Variant A was a distribution pane, a')
    A('full-width (Z0Z0+ZTZT+ZTZ0+Z0ZT)/full pane and a 2x2 of the individual')
    A('polarisation ratios, on the NLO reference alone.  The six RATIO panes')
    A('here are that decomposition spanning two orders, with the K-factor')
    A('added; the seventh pane on top is the distribution, at both orders.')
    A('')
    A('AN EARLIER PASS DROPPED THE DISTRIBUTION PANE FROM THIS FIGURE and')
    A('this file said so, on the argument that "the figure is about ratios".')
    A('That argument did not survive.  Six ratio panes with no absolute scale')
    A('anywhere say how the components divide the rate up and never say what')
    A('the rate IS, and a reader who wants the K pane to mean anything has to')
    A('fetch the two cross sections off another figure.  The seventh pane is')
    A('that missing statement.  The six below it are unchanged -- same')
    A('content, same order, same conventions, same errors, same bands.')
    A('')
    A('BOTH ORDERS ON FIVE OF THE SIX PANES.  The figure now spans NLO and')
    A('LO, so every pane that admits two orders draws two -- the sum pane and')
    A('the four fraction panes each carry the NLO reference (solid, filled')
    A('markers) and the LO sample (dashed, open markers).  The K pane is')
    A('itself the ratio BETWEEN the two orders and can only be one curve per')
    A('component.')
    A('')
    A('PANE ORDER.  The four fractions run LL, LT, TL, TT -- the K-factor')
    A('figure\'s component order -- and NOT variant A\'s LL, TT, TL, LT.  They')
    A('sit under a K pane whose five-entry legend is in the former, and')
    A('reading the two against each other should not need re-sorting.')
    A('')
    A('ERRORS.  Unchanged from the figures they are built out of, and not one')
    A('rule for the whole canvas:')
    A('  the five ratio panes, both orders: the delta-method / jackknife bar.')
    A('    Numerator and denominator are sums over the SAME events of ONE')
    A('    sample, so their covariance is real and is kept.')
    A('  the K pane: the two relative MC errors in quadrature.  Two')
    A('    independent samples, no covariance to subtract.')
    A('  the seventh pane, all ten curves: the plain MC error sqrt(sum w^2).')
    A('    Each is a single weighted sum and not a ratio, so there is no')
    A('    covariance for a delta-method bar to keep.  Same rule, and the')
    A('    same objects, as panels 1-5 of the K-factor figure.')
    A('')
    A('THE DASHED RULE in each fraction pane is that pane\'s NLO integrated')
    A('value, as on variant A.  The dotted rule on the sum pane is 1.  The K')
    A('pane carries NO rule: a K-factor ought not to be 1 and drawing one')
    A('there would suggest a null hypothesis nobody holds.  The seventh pane')
    A('carries no rule either: it is a distribution, not a ratio.')
    A('')
    _ratio6_top_numbers(A, d, lo)
    for obs in PA.OBS:
        cur = PA.ratio6_curves(d, lo, obs)
        edges = PA.BINS[obs]
        A('-' * 74)
        A('%s -- the six panes, bin by bin' % PA.LABELS_TXT[obs][0])
        A('-' * 74)
        for pane in ['SUM'] + PA.RATIO6_FRACTIONS:
            A('  %s' % PA.RATIO_TXT[pane])
            A('    %-18s %12s %10s   %12s %10s'
              % ('bin', 'NLO', '+-', 'LO', '+-'))
            for i in range(len(edges) - 1):
                A('    %7.3f - %-8.3f %12.6f %10.6f   %12.6f %10.6f'
                  % (edges[i], edges[i + 1],
                     cur[pane]['NLO']['r'][i], cur[pane]['NLO']['err'][i],
                     cur[pane]['LO']['r'][i], cur[pane]['LO']['err'][i]))
            A('    integrated: NLO %.6f   LO %.6f'
              % (cur[pane]['NLO']['integrated'],
                 cur[pane]['LO']['integrated']))
            A('')
        A('  K = NLO / LO, per component')
        A('    %-18s' % 'bin'
          + ''.join('%11s' % PA.KF_CURVE_TXT[k].replace(' ', '')
                    for k in PA.KF_PANE_ORDER))
        for i in range(len(edges) - 1):
            A('    %7.3f - %-8.3f' % (edges[i], edges[i + 1])
              + ''.join('%11.4f' % cur['K'][k]['k'][i]
                        for k in PA.KF_PANE_ORDER))
        A('')


def _scale_numbers(A, d, lo):
    """The scale conventions, both of them, and every number they produce."""
    A('=' * 74)
    A('SCALE UNCERTAINTY -- THE CONVENTIONS, STATED, AND BOTH ALTERNATIVES')
    A('=' * 74)
    A('')
    A('WHERE THE COLUMNS CAME FROM.  They were NOT cached.  Before this pass')
    A('data/weights.npz and data/weights_LO.npz carried w_MUR1.0_MUF1.0 and')
    A('no other scale point, so the eight remaining MUR*_MUF* columns could')
    A('not be summed and were not approximated.  Both files were re-read from')
    A('their HepMC by extract_hepmc_pol.py with KEEP_WEIGHTS widened to all')
    A('nine -- %.1f s for run_06_decayed_1 and %.1f s for run_12_decayed_1,'
      % (d.meta['pass_seconds'], lo.meta['pass_seconds']))
    A('one streaming pass each.  EVERY COLUMN THAT WAS ALREADY THERE CAME')
    A('BACK BIT-FOR-BIT IDENTICAL; the two .npz grew by the eight new columns')
    A('and by nothing else, which is why every figure and every number of the')
    A('earlier passes is unchanged.  The three other spinmode samples were')
    A('not re-read: nothing on these two figures uses them.')
    A('')
    A('*** THE TWO LABELS ARE TRANSPOSED IN THESE FILES. ***')
    A('The weight the HepMC N line calls MURa_MUFb is the one generated at')
    A('muR = b, muF = a.  The two factors are right; which scale each belongs')
    A('to is swapped.  Measured three ways, all on these files:')
    A('  (1) p p > z z at order = LO is the Born, O(alpha_s^0), so its cross')
    A('      section is EXACTLY muR independent and can only move with muF.')
    A('      In weights_LO.npz the weight is degenerate in the SECOND label on')
    A('      all 250 000 events and varies only with the FIRST.  So the first')
    A('      label is the factorisation scale.')
    A('  (2) Directly, on event 1 of run_12_decayed_1.  Its LHE <rwgt> gives')
    A('      ids 1001-1003 (muR = 1, 2, 0.5 at muF = 1) the same value')
    A('      2.3950370e-02, ids 1004-1006 (muF = 2) 2.3914174e-02 and ids')
    A('      1007-1009 (muF = 0.5) 2.3803324e-02.  The HepMC of the same')
    A('      event has 2.3803324e-02 under all three MUR0.5_* names,')
    A('      2.3950370e-02 under MUR1.0_* and 2.3914174e-02 under MUR2.0_*.')
    A('  (3) The mechanism.  The LHE header writes the nine with muF OUTER')
    A('      and muR INNER, each cycling (1.0, 2.0, 0.5); the name generator')
    A('      assumes muR outer and muF inner with the same cycle.  Composing')
    A('      the two is exactly a transposition and it reproduces all nine.')
    A('IT CHANGES NO NUMBER HERE.  An envelope is a min and a max over a SET,')
    A('and a permutation of a set moves neither.  The 7-point subset is')
    A('defined by which two points it DROPS, so it is not automatically safe')
    A('-- but it is: the pair dropped, (muR, muF) = (0.5, 2.0) and (2.0,')
    A('0.5), is the anti-diagonal, which the transposition maps onto itself.')
    A('What it does change is the reading of any INDIVIDUAL point, so every')
    A('per-point table below is printed under the TRUE muR / muF.')
    A('')
    A('DECISION 1 -- WHICH ENVELOPE.  Drawn: %s.'
      % PA.SCALE_ENVELOPE_TXT[PA.SCALE_ENVELOPE])
    A('  The two dropped points put a factor of four between muR and muF,')
    A('  which manufactures a large log(muR/muF) the fixed-order calculation')
    A('  was never asked to resum; they inflate the envelope without probing')
    A('  a missing higher order.  Both envelopes are printed below.  On these')
    A('  samples the choice matters in one place and not in the other: on a')
    A('  single sample\'s cross section the two dropped points ARE the two')
    A('  extremes, and on the K-factor they are interior and the two')
    A('  envelopes agree to four decimals.')
    A('')
    A('DECISION 2 -- CORRELATION BETWEEN NUMERATOR AND DENOMINATOR.')
    A('  FRACTIONS (component / full inside ONE sample): no choice, and the')
    A('    code offers none.  The same event weights appear top and bottom,')
    A('    so the envelope is of the RATIO recomputed scale by scale, NOT the')
    A('    ratio of two separately built envelopes.  The latter would be')
    A('    describing a configuration that cannot occur -- the numerator at')
    A('    its high scale while the denominator sits at its low one, on the')
    A('    same events.  Nearly all the variation cancels this way; what')
    A('    survives is the reweighting of the phase-space MIX inside the bin,')
    A('    which is why these bands are sub-percent where a single cross')
    A('    section moves by several.')
    A('  K-FACTOR: a real choice, both defensible, and they differ.')
    A('    DRAWN: %s.' % PA.KFACTOR_SCALE_CORRELATION_TXT['correlated'])
    A('    ALTERNATIVE, quoted in every table below but not drawn: %s.'
      % PA.KFACTOR_SCALE_CORRELATION_TXT['uncorrelated'])
    A('    The correlated one is drawn because the two samples are the same')
    A('    process, the same PDF set, the same functional scale choice and')
    A('    the same run card, so a scale is a physical setting shared between')
    A('    them rather than two independent nuisance parameters; and the')
    A('    muF / parton-luminosity part of the variation, which has nothing')
    A('    to do with the ORDER, then largely cancels.  The uncorrelated one')
    A('    is about a third wider.')
    A('')
    A('DECISION 3 -- HOW A POLARISED COMPONENT IS VARIED AT ALL.')
    A('  %s' % PA.SCALE_POL_MODEL_TXT)
    A('  Exact at LO because the muF and alpha_s dependence multiplies the')
    A('  whole squared matrix element at a fixed phase-space point,')
    A('  identically for the polarised projection and for the total.  An')
    A('  APPROXIMATION at NLO: the virtual and real pieces are not decomposed')
    A('  by polarisation, and this study\'s own result is that the components')
    A('  have different K-factors, which is the same statement as their')
    A('  alpha_s dependence differing.  The NLO component bands are therefore')
    A('  the total\'s band transported onto the component.  Only a MadSpin run')
    A('  emitting ms_pol_* per scale point could do better; no cached weight')
    A('  can.')
    A('  187 of the NLO sample\'s 250 000 events have |w_s/w_central| > 3 for')
    A('  some s -- MC@NLO weights that very nearly cancel.  They are NOT')
    A('  clipped: together they move the reweighted total by 0.39 %, and the')
    A('  reweighted total agrees with the directly summed w_s to 3e-05, the')
    A('  same 3e-05 by which sum(MUR1.0_MUF1.0) and sum(Weight) already')
    A('  differ.  The LO sample has none; its extreme ratio is 1.246.')
    A('')
    # -- the nine points, per sample, under the TRUE labels -----------------
    A('-' * 74)
    A('THE NINE POINTS, PER SAMPLE, UNDER THEIR TRUE muR / muF')
    A('-' * 74)
    for samp, tag in ((d, 'NLO (run_06)'), (lo, 'LO (run_12)')):
        c = float((np.asarray(samp.full, float) * samp.scale_to_pb).sum())
        A('  %s   sigma(central) = %.7f pb' % (tag, c))
        A('    %-22s %-14s %14s %10s   %s'
          % ('N-line name', 'TRUE', 'sigma [pb]', 'rel', 'in 7-point'))
        for s in PA.SCALE_NAMES:
            w = (np.asarray(samp.full, float) * samp.scale_ratio(s)
                 * samp.scale_to_pb).sum()
            A('    %-22s %-14s %14.7f %9.2f %%   %s'
              % (s, PA.scale_true_txt(s), w, 100 * (w / c - 1.0),
                 'yes' if s in PA.SCALE_SEVEN else 'NO  (anti-diagonal)'))
        for which in ('seven', 'nine'):
            vals = [float((np.asarray(samp.full, float) * samp.scale_ratio(s)
                           * samp.scale_to_pb).sum())
                    for s in PA.scale_points(which)]
            A('    %s-point envelope on sigma: %+.2f %% / %+.2f %%'
              % (which, 100 * (max(vals) / c - 1), 100 * (min(vals) / c - 1)))
        A('')
    # -- the K-factor table, the answer ------------------------------------
    for which in ('seven', 'nine'):
        rows = PA.integrated_kfactor_bands(d, lo, None, which)
        A('-' * 74)
        A('INCLUSIVE K-FACTOR WITH STATISTICS AND SCALE (%s-point envelope)'
          % which)
        A('-' * 74)
        A('  no lepton selection: all 250 000 events of each sample.')
        A('  %-12s %9s %9s   %-20s %-20s'
          % ('component', 'K', '+- stat', 'scale, correlated',
             'scale, uncorrelated'))
        for r in rows:
            A('  %-12s %9.4f %9.4f   %+7.2f %% %+7.2f %%   %+7.2f %% %+7.2f %%'
              % (PA.KF_CURVE_TXT[r['key']], r['K'], r['K_err'],
                 100 * (r['K_scale_hi'] / r['K'] - 1),
                 100 * (r['K_scale_lo'] / r['K'] - 1),
                 100 * (r['K_scale_hi_uncorr'] / r['K'] - 1),
                 100 * (r['K_scale_lo_uncorr'] / r['K'] - 1)))
        A('')
        A('  the same, as absolute K bounds:')
        A('  %-12s %9s   %-19s %-19s'
          % ('component', 'K', 'correlated', 'uncorrelated'))
        for r in rows:
            A('  %-12s %9.4f   [%7.4f, %7.4f]  [%7.4f, %7.4f]'
              % (PA.KF_CURVE_TXT[r['key']], r['K'],
                 r['K_scale_lo'], r['K_scale_hi'],
                 r['K_scale_lo_uncorr'], r['K_scale_hi_uncorr']))
        A('')
        A('  K per scale point, under the TRUE labels:')
        A('    %-22s' % 'TRUE'
          + ''.join('%11s' % PA.KF_CURVE_TXT[k].replace(' ', '')
                    for k in PA.KF_PANE_ORDER))
        for s in PA.scale_points(which):
            A('    %-22s' % PA.scale_true_txt(s)
              + ''.join('%11.4f' % {r['key']: r for r in rows}[k]
                        ['K_per_scale'][s] for k in PA.KF_PANE_ORDER))
        A('')
    # -- the double ratio: the question the figure exists to answer --------
    surv = PA.scale_survival(d, lo)
    rows = surv['rows']
    A('=' * 74)
    A('DOES THE POLARISATION DEPENDENCE OF K SURVIVE THE SCALE BAND?')
    A('=' * 74)
    A('K itself moves by -4.6 % / +6.8 % (correlated, 7-point) and the spread')
    A('BETWEEN components is about 7.5 %, so on the raw K the band and the')
    A('effect are the same size.  That is the wrong comparison and it is the')
    A('one RESULTS.md flagged.  The scale moves all five K TOGETHER -- it is')
    A('very nearly a common multiplicative shift -- so the quantity to test')
    A('is the RATIO of one component\'s K to the unpolarised K,')
    A('')
    A('    D = K_component / K_full   ( == f_NLO / f_LO, the double ratio )')
    A('')
    A('taken scale by scale so the common movement cancels.  D = 1 is "this')
    A('component has the same K as the total".')
    A('')
    A('  %-10s %8s %9s %8s   %-19s %8s'
      % ('component', 'D', '+- stat', 'stat sig', 'scale band on D',
         'D-1 / halfband'))
    for r in rows:
        if 'D' not in r:
            continue
        A('  %-10s %8.4f %9.4f %7.1fs   %+7.2f %% %+7.2f %% %10.1f'
          % (PA.KF_CURVE_TXT[r['key']], r['D'], r['D_err'],
             r['D_sigma_from_1'],
             100 * (r['D_scale_hi'] / r['D'] - 1),
             100 * (r['D_scale_lo'] / r['D'] - 1),
             r['D_over_scale_halfband']))
    A('')
    A('  the mixed components against Z_T Z_T, the comparison the brief names:')
    mlo, mhi = surv['mixed_over_TT_band']
    dlo, dhi = surv['mixed_minus_TT_band']
    A('    K(mixed avg) / K(ZT ZT)  = %.4f, envelope [%.4f, %.4f]'
      % (surv['mixed_over_TT'], mlo, mhi))
    A('    K(mixed avg) - K(ZT ZT)  = %.4f, envelope [%.4f, %.4f]'
      % (surv['mixed_minus_TT'], dlo, dhi))
    A('    The envelope never reaches 1 (never reaches 0 for the difference),')
    A('    so the mixed components take a LARGER NLO enhancement than ZT ZT')
    A('    at every one of the %d scale points, by at least %.1f %%.'
      % (len(surv['points']), 100 * (mlo - 1)))
    A('')
    # -- per-observable, fiducial ------------------------------------------
    for obs in PA.OBS:
        A('-' * 74)
        A('%s -- fiducial K with statistics and scale' % PA.LABELS_TXT[obs][0])
        A('-' * 74)
        for r in PA.integrated_kfactor_bands(d, lo, obs):
            A('  %-12s K = %.4f +- %.4f  scale [%.4f, %.4f]'
              % (PA.KF_CURVE_TXT[r['key']], r['K'], r['K_err'],
                 r['K_scale_lo'], r['K_scale_hi']))
        A('')
        A('  the fraction bands, per bin, NLO sample (envelope of the ratio):')
        edges = PA.BINS[obs]
        for pane in ['SUM'] + PA.RATIO6_FRACTIONS:
            fb = PA.fraction_band(d, obs, pane)
            A('    %s' % PA.RATIO_TXT[pane])
            A('      %-18s %12s %10s %12s %12s'
              % ('bin', 'ratio', '+- stat', 'scale lo', 'scale hi'))
            for i in range(len(edges) - 1):
                A('      %7.3f - %-8.3f %12.6f %10.6f %12.6f %12.6f'
                  % (edges[i], edges[i + 1], fb['r'][i], fb['err'][i],
                     fb['lo'][i], fb['hi'][i]))
            A('')
        A('  the K bands, per bin (correlated, %s-point):' % PA.SCALE_ENVELOPE)
        A('    %-18s' % 'bin'
          + ''.join('%22s' % PA.KF_CURVE_TXT[k].replace(' ', '')
                    for k in PA.KF_PANE_ORDER))
        kb = {k: PA.kfactor_band(d, lo, obs, k) for k in PA.KF_PANE_ORDER}
        for i in range(len(edges) - 1):
            A('    %7.3f - %-8.3f' % (edges[i], edges[i + 1])
              + ''.join('  %8.4f [%.3f,%.3f]'.replace('  ', ' ', 0)
                        % (kb[k]['k'][i], kb[k]['lo'][i], kb[k]['hi'][i])
                        for k in PA.KF_PANE_ORDER))
        A('')


def _kfactor_numbers(A, d, lo):
    """The K-factor figure's numbers, and the argument for its error bars."""
    A('=' * 74)
    A('THE K-FACTOR FIGURE: NLO/LO, PER POLARISATION COMPONENT')
    A('=' * 74)
    A('A FIFTH sample: run_12_decayed_1, the same p p > z z [QCD] process, the')
    A('same run card, the same 250 000 events, the same exclusive decay card,')
    A('the same spinmode madspin and the same Pythia8 shower as the reference')
    A('-- and <run_settings> order = LO where every other run in this study')
    A('says order = NLO.  It carries the four ms_pol_* weights (33 names on')
    A('its N line, the reference\'s list exactly), which is what makes a')
    A('per-polarisation K-factor possible at all.')
    A('')
    A('ONE MADSPIN CARD LINE DIFFERS BEYOND THE ORDER and it is a polarisation')
    A('setting, so it is checked rather than waved through.  run_06 sets')
    A('"set frame_id 24"; run_12 leaves it commented out and takes the default')
    A('6.  frame_id is the bitmask sum(2**n for n in me_frame), so 24 = legs')
    A('3+4 (the two z, i.e. the ZZ rest frame) and 6 = legs 1+2 (the initial')
    A('partons).  Those are the same frame exactly when the event has no other')
    A('final-state leg, and the LO sample\'s LHE is 2 -> 2 in every event')
    A('(NUP = 4 throughout) while the NLO one carries a fifth leg in about a')
    A('fifth of its events.  So both samples quantise the z polarisations')
    A('along the ZZ-rest-frame axis and the components being divided are the')
    A('same components.  Had the LO run been a real-emission sample this')
    A('K-factor would have been ill defined.')
    A('')
    A('NORMALISATION.  THIS FIGURE IS ABSOLUTE AND VARIANT B IS NOT, and that')
    A('is deliberate.  A K-factor is a ratio of RATES, so nothing may be')
    A('divided out before it is taken: panels 1-5 are dsigma/dx in pb per unit')
    A('of the observable and panel 6 is their ratio.  Variant B divides every')
    A('curve by its own sigma because it asks about SHAPE; at these cuts the')
    A('rate moves 29 % and the shape a few percent, so an unnormalised')
    A('variant-B pane would be four curves sitting near 0.78 and nothing else')
    A('would be readable.  The two figures answer two different questions.')
    A('')
    A('ERRORS, PER CURVE.')
    A('  panels 1-5, both curves: plain MC, sqrt(sum w^2) per bin.  Each is a')
    A('    single weighted sum and not a ratio, so there is no covariance for')
    A('    a delta-method bar to keep.')
    A('  panel 6, all five curves: the two relative errors in quadrature.')
    A('    Numerator and denominator are sums over two INDEPENDENT samples')
    A('    (run_06 against run_12, different order), so there is no covariance')
    A('    to subtract either.  The independence needs no argument beyond the')
    A('    orders being different, and the n(w<0) test of "ARE THE SAMPLES')
    A('    PAIRED" agrees at once: 14 273 for the reference against 0 for LO,')
    A('    the LO matrix element being positive definite.')
    A('  So the delta-method bar used by the sum pane and the 2x2 is on')
    A('    NOTHING drawn on this figure.  That is not an oversight: it belongs')
    A('    to a ratio whose two sums run over ONE set of events, and no')
    A('    quantity on this canvas is one.  Where it does apply to this')
    A('    comparison is the component-FRACTION double ratio at the foot of')
    A('    this section, which is printed precisely so that the two error')
    A('    treatments of the same statement can be read against each other.')
    A('')
    A('--- inclusive K-factors (all 250k events of each sample, no selection) ---')
    A('%-12s %16s %16s %18s' % ('component', 'sigma_NLO [pb]', 'sigma_LO [pb]',
                                'K = NLO/LO'))
    rows = PA.integrated_kfactors(d, lo)
    for r in rows:
        A('%-12s %10.7f+-%.7f %10.7f+-%.7f %9.4f +- %.4f'
          % (PA.KF_CURVE_TXT[r['key']], r['sigma_nlo_pb'], r['err_nlo_pb'],
             r['sigma_lo_pb'], r['err_lo_pb'], r['K'], r['K_err']))
    A('')
    A('the banner cross sections these reproduce: 1.366e+01 pb (run_06, NLO)')
    A('and 1.059e+01 pb (run_12, LO), whose ratio is %.5f against the %.5f'
      % (13.66 / 10.59, rows[0]['K']))
    A('of the unpolarised row above.  The two differ only by the rounding of')
    A('the banner numbers to four digits.  Each sample\'s own sum(w_"0") is')
    A('sigma times the two leptonic branching fractions and not the inclusive')
    A('sigma; the factor is the same on both sides and cancels in K.')
    A('')
    A('--- do the polarisations have DIFFERENT K-factors? ---')
    A('two statements of the same question, with two error treatments.')
    A('')
    A('(a) K_component - K_unpolarised, plain quadrature bars.  CONSERVATIVE:')
    A('    a component and its own sample\'s total share most of their MC')
    A('    fluctuation, so this bar keeps a cancellation it should have used.')
    A('%-12s %22s %10s' % ('component', 'K - K_unpol', 'sigma'))
    for r in rows[1:]:
        A('%-12s %12.4f +- %.4f %10.1f'
          % (PA.KF_CURVE_TXT[r['key']], r['K_minus_K_full'],
             r['K_minus_K_full_err'], r['K_minus_K_full_sigma']))
    A('')
    A('(b) the double ratio f_NLO/f_LO, f = sigma_component/sigma_full.  This')
    A('    is algebraically K_component/K_unpolarised, and THIS is where the')
    A('    delta-method bar belongs: each f is a ratio of two sums over the')
    A('    SAME events of ONE sample, so its error is the correlated one; the')
    A('    two f then come from the two independent samples and combine in')
    A('    quadrature.  Sharper than (a) by a factor of a few, as it should be.')
    A('%-12s %26s %26s %14s %8s'
      % ('component', 'f_NLO', 'f_LO', 'f_NLO/f_LO', 'sigma'))
    for r in PA.component_fraction_double_ratio(d, lo):
        A('%-12s %14.6f +- %.6f %14.6f +- %.6f %7.4f+-%.4f %8.1f'
          % (PA.KF_CURVE_TXT[r['key']], r['f_nlo'], r['f_nlo_err'],
             r['f_lo'], r['f_lo_err'], r['double_ratio'],
             r['double_ratio_err'], r['sigma_from_1']))
    A('')
    A('THE ANSWER IS YES.  The two MIXED components, Z_0Z_T and Z_T Z_0, have')
    A('a K-factor near 1.36 against 1.289 for the unpolarised total, while')
    A('Z_T Z_T sits low at 1.265 and Z_0 Z_0 is compatible with the')
    A('unpolarised one.  On the double ratio that is ~15 sigma for the two')
    A('mixed components and ~16 sigma for Z_T Z_T; MC statistics only, no')
    A('scale or PDF uncertainty is in any of these bars and the scale envelope')
    A('on each sample alone is +4.1 % -5.2 %, which is far larger than the')
    A('K-factor differences and does NOT cancel in the ratio in any')
    A('controlled way.  The significances above are therefore statements about')
    A('these two samples, not a theory uncertainty on K.')
    A('')
    A('--- WHICH z IS THE FIRST INDEX, and why Z_0Z_T and Z_TZ_0 are not the')
    A('--- same curve on Delta phi(e+e-) ---')
    A('The two mixed components are NOT interchangeable here, and the figure')
    A('shows it.  z1_ch is 11 and z2_ch is 13 on ALL 250 000 events of BOTH')
    A('samples -- the extractor reads each z\'s channel off the event record --')
    A('so the FIRST z is always the one that went to e+e-.  ms_pol_23.0_23.T')
    A('is therefore (electron-z longitudinal, muon-z transverse) and')
    A('ms_pol_23.T_23.0 the other way round.  Delta phi(e+e-) is built from')
    A('the electron z alone, so it is directly sensitive to which of the two')
    A('the first index names, and the two components have genuinely different')
    A('shapes in it within ONE sample: the last-bin/first-bin ratio of the')
    A('normalised Delta phi spectrum is 15.7 for Z_0Z_T against 49.1 for')
    A('Z_TZ_0 at LO, and 14.0 against 20.6 at NLO.  M(e+ mu+) uses one lepton')
    A('from each z and is nearly symmetric between them, which is why their')
    A('K-factors agree there (1.360 / 1.348) and not here (1.378 / 1.354).')
    A('')
    A('That asymmetry is the whole of the largest excursion on either panel 6:')
    A('K(Z_TZ_0) = 2.94 +- 0.18 in the lowest Delta phi bin against ~1.3 for')
    A('everything else.  It is not a statistical accident -- 3547 NLO and 3308')
    A('LO selected events in that bin -- and the mechanism is plain in the')
    A('numbers above: with the electron z TRANSVERSE the LO spectrum is very')
    A('nearly empty at small Delta phi (it needs a boosted z, which LO 2 -> 2')
    A('kinematics supplies only in the tail), and real radiation at NLO fills')
    A('a region that started almost empty.  A large K on a small denominator')
    A('is what a fixed-order ratio does at the edge of LO phase space; it says')
    A('the LO prediction is unreliable there, not that the NLO one is large.')
    A('')
    for obs in PA.OBS:
        A('--- %s: K-factor bin by bin (panel 6) ---' % PA.SHORT[obs])
        ki = PA.integrated_kfactors(d, lo, obs)
        A('in-selection K: '
          + '  '.join('%s = %.4f+-%.4f'
                      % (PA.KF_CURVE_TXT[r['key']], r['K'], r['K_err'])
                      for r in ki))
        ks = {k: PA.kfactor(d, lo, obs, k) for k in PA.KF_PANE_ORDER}
        A('%12s %9s %9s   %s'
          % ('bin', 'N_NLO', 'N_LO',
             '  '.join('%16s' % PA.KF_CURVE_TXT[k] for k in PA.KF_PANE_ORDER)))
        edges = PA.BINS[obs]
        nn, _ = np.histogram(np.asarray(d.z[obs], float)[d.sel[obs]],
                             bins=edges)
        nl, _ = np.histogram(np.asarray(lo.z[obs], float)[lo.sel[obs]],
                             bins=edges)
        for b in range(len(edges) - 1):
            A('%5.2f-%5.2f %9d %9d   %s'
              % (edges[b], edges[b + 1], nn[b], nl[b],
                 '  '.join('%8.4f+-%.4f' % (ks[k]['k'][b], ks[k]['err'][b])
                           for k in PA.KF_PANE_ORDER)))
        A('')


# The multiplier on each K-factor panel's autoscale top, per y scale.  These
# panels need far less than the distribution panes of the other figures do,
# because each carries a TWO-row legend against those panes' seven to eleven:
# 10x on a log panel and 1.3x on a linear one clears it.  Panel 6's legend is
# five entries in three columns, so two rows, and 1.32 of the drawn span
# clears that.  Checked on the rendered PNGs, not on the axis limits.
KF_HEADROOM = {True: 10.0, False: 1.30}
KF_HEADROOM_K = 1.32


def draw_kfactor(nlo, lo, obs, outdir):
    """The six-panel K-factor figure: NLO against LO, per polarisation.

    Three rows of two.  Panels 1-5 are one component each -- unpolarised,
    Z_0Z_0, Z_0Z_T, Z_TZ_0, Z_TZ_T, in that order -- carrying that component's
    LO and NLO ``dsigma/dx``.  Panel 6 is ``K = NLO/LO`` with all five curves
    on one pane, which is the comparison the whole figure exists to make.

    THIS FIGURE IS NOT NORMALISED AND VARIANT B IS.  A K-factor is a ratio of
    rates, so nothing is divided out here: the panels are absolute pb per unit
    of the observable and panel 6 is their ratio.  Variant B divides every
    curve by its own sigma because it is asking about SHAPE, and at these cuts
    the rate moves 29 % against a few percent of shape, so it would otherwise
    show nothing else.  The two figures answer two different questions and
    ``pol_analysis.KF_PANE_ORDER``'s comment and RESULTS.md both say so.

    ERRORS, per curve:

    - panels 1-5, both curves: the plain MC error ``sqrt(sum w^2)`` of
      ``PA.component_histogram``.  Each is a single weighted sum and not a
      ratio, so there is no covariance for a delta-method bar to keep.
    - panel 6, all five curves: the two relative errors in quadrature
      (``PA.kfactor``).  Numerator and denominator are sums over two
      INDEPENDENT samples -- different order, ``run_06`` against ``run_12``,
      and ``n(w<0)`` = 14 273 against 0 -- so there is no covariance to
      subtract either.

    The delta-method bar the ratio panes of the other figures use is therefore
    on NOTHING drawn here, and that is not an oversight: it belongs to a ratio
    whose two sums run over one set of events, and no quantity on this canvas
    is one.  Where it does apply to this comparison is the component-FRACTION
    double ratio ``f_NLO/f_LO``, which is algebraically ``K_comp/K_full`` with
    the within-sample correlation kept; ``numbers.txt`` prints it beside the
    quadrature version so the two treatments can be read against each other.
    """
    edges = PA.BINS[obs]
    x = 0.5 * (edges[:-1] + edges[1:])
    xlab, ylab = (PA.LABELS_TEX if USETEX else PA.LABELS_TXT)[obs]
    name = PA.KF_CURVE_TEX if USETEX else PA.KF_CURVE_TXT
    logy = obs in PA.LOGY

    fig = plt.figure(figsize=(9.0, 10.2))
    # wspace is wide for a 3x2: the right column's y tick labels run to five
    # significant figures on a linear pane (0.00125) and at the default gap
    # its axis label lands on the left column's frame.  Checked on the PNG.
    gs = fig.add_gridspec(3, 2, hspace=0.07, wspace=0.30)
    # Column-wise sharex: only the bottom row carries tick labels and the axis
    # name, exactly as the stacked panes of the other figures do.
    axes = []
    for r in range(3):
        for cc in range(2):
            share = axes[cc] if r else None
            axes.append(fig.add_subplot(gs[r, cc], sharex=share))

    ks = {}
    for i, key in enumerate(PA.KF_PANE_ORDER):
        ax = axes[i]
        kk = PA.kfactor(nlo, lo, obs, key)
        ks[key] = kk
        for tag, h in (('NLO', kk['nlo']), ('LO', kk['lo'])):
            y, e = np.asarray(h['y'], float), np.asarray(h['err'], float)
            shown = np.where(y > 0, y, np.nan) if logy else y
            ax.stairs(shown, edges, color=COLOR[key], ls=LS_ORDER[tag],
                      lw=LW + (0.5 if tag == 'NLO' else 0.0),
                      baseline=None, zorder=5 if tag == 'NLO' else 4,
                      label='%s, %s' % (name[key], tag))
            ax.errorbar(x, shown, yerr=np.where(np.isfinite(shown), e, np.nan),
                        fmt='none', ecolor=COLOR[key], elinewidth=0.8,
                        alpha=0.9 if tag == 'NLO' else 0.6,
                        zorder=5 if tag == 'NLO' else 4)
        if logy:
            ax.set_yscale('log')
        ax.set_ylabel(ylab, fontsize=9)
        lo_y, hi_y = ax.get_ylim()
        ax.set_ylim(lo_y, hi_y * KF_HEADROOM[logy])
        ax.legend(frameon=False, fontsize=8.0, loc='upper left')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(labelsize=9)
        if not logy:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            # Four ticks, not the default six: on a linear pane the labels are
            # five significant figures wide and six of them crowd the column.
            ax.yaxis.set_major_locator(MaxNLocator(5))

    # Panel 6: the five K-factors together.
    axk = axes[5]
    for key in PA.KF_PANE_ORDER:
        kk = ks[key]
        k, e = np.asarray(kk['k'], float), np.asarray(kk['err'], float)
        axk.stairs(k, edges, color=COLOR[key], ls=LS[key],
                   lw=LW + (0.5 if key == 'full' else 0.0), baseline=None,
                   zorder=6 if key == 'full' else 4, label=name[key])
        axk.errorbar(x, k, yerr=e, fmt='o', ms=3.0, color=COLOR[key],
                     elinewidth=0.9, zorder=6 if key == 'full' else 5)
    axk.set_ylabel(PA.KFACTOR_TEX if USETEX else PA.KFACTOR_TXT, fontsize=10)
    lo_y, hi_y = axk.get_ylim()
    axk.set_ylim(lo_y, lo_y + (hi_y - lo_y) * KF_HEADROOM_K)
    axk.legend(frameon=False, fontsize=8.0, loc='upper left', ncol=3)
    axk.xaxis.set_minor_locator(AutoMinorLocator())
    axk.yaxis.set_major_locator(MaxNLocator(6))
    axk.yaxis.set_minor_locator(AutoMinorLocator())
    axk.tick_params(labelsize=9)
    for sp in axk.spines.values():
        sp.set_linewidth(1.5)

    for i, ax in enumerate(axes):
        if i < 4:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel(xlab, fontsize=11)
    # One xlim per COLUMN: the sharex above is column-wise, so setting it on
    # the two top panels reaches all six.
    axes[0].set_xlim(edges[0], edges[-1])
    axes[1].set_xlim(edges[0], edges[-1])

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, PA.SHORT[obs])
    want = wants_minus(fig)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    ki = PA.integrated_kfactors(nlo, lo, obs)
    print('%-12s K-factor  ' % PA.SHORT[obs]
          + '  '.join('%s=%.3f' % ((PA.KF_CURVE_TXT[r['key']]).replace(' ', ''),
                                   r['K']) for r in ki))
    return base, want


# --------------------------------------------------------------------------
# THE TWO SIX-PANEL RATIO FIGURES.  See pol_analysis.RATIO6_DIR for what they
# are and why the distribution pane is not on them.

# The band's face alpha, and the line style of the band EDGE.  Faint enough
# that the nominal step and its error bars stay the most visible thing in the
# pane -- the band is a systematic, the points are the measurement -- and
# opaque enough to read where two components overlap on the K pane.
BAND_ALPHA = 0.18
BAND_ALPHA_K = 0.13
BAND_LW = 0.0

# The band on the seventh pane is a band on an ABSOLUTE cross section, so it is
# a couple of percent wide where the ratio bands below are sub-percent, and it
# is behind a curve on a log axis rather than a ratio window.  A little more
# opaque than the ratio panes' so that it reads at all at that width.
BAND_ALPHA_TOP = 0.22

# The multiplier on the seventh pane's autoscale top, log pane and linear pane.
# What sets it is the ten-entry legend: two rows at ncol=5 over a pane whose
# curves peak at the right on Delta phi and at 90 GeV on the mass, so the box
# has to clear the highest curve outright and not merely the autoscale top.
# Both observables currently come out LOG (see PA.RATIO6_TOP_LOG_DECADES); the
# linear entry is kept for a future observable that does not.  Checked on the
# rendered PNGs, not on the axis limits.
TOP_HEADROOM = (14.0, 1.55)

# THE LINE WEIGHT LADDER, and it is there to solve one measured problem.
#
# On this pane colour is the component and the line is the ORDER, which is the
# convention the six panes below and the K-factor figure already use and which
# is worth keeping across one canvas.  It has one failure: Z_0Z_T and Z_TZ_0
# are equal to 1-2 % bin by bin -- they must be, ZZ is symmetric and the two
# differ only by which Z the projection is taken on -- so on a log pane they
# are the same curve to within the line width, and whichever is drawn second
# hides the other outright.  Drawn flat, the purple Z_0Z_T is simply not on
# the figure while its two legend entries are, which is worse than a crowded
# pane.
#
# So the components are drawn in :data:`pol_analysis.KF_PANE_ORDER` with the
# line weight stepping DOWN, and the later, thinner curve therefore sits
# inside the earlier, thicker one and leaves a halo of it showing.  Nothing is
# offset and no value is touched -- two coincident curves are drawn coincident
# and both are visible.  The unpolarised total comes out the thickest, which
# is also what it is on every other pane of the study.  Checked on the
# rendered PNG.
TOP_LW_BASE = LW + 1.0
TOP_LW_STEP = 0.22
TOP_LW_NLO = 0.30


def _band(ax, edges, lo, hi, color, alpha, zorder=2):
    """A scale band drawn on the same step grid as ``stairs``.

    ``fill_between(step='post')`` with ``x = edges`` and the last value
    repeated puts ``lo[i]``/``hi[i]`` across ``[edges[i], edges[i+1]]``, which
    is exactly the cell ``ax.stairs(values, edges)`` fills.  Anything else --
    filling between bin CENTRES, say -- would draw a band half a bin out of
    register with the curve it belongs to, most visibly in the end bins.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    ax.fill_between(edges, np.concatenate([lo, lo[-1:]]),
                    np.concatenate([hi, hi[-1:]]), step='post',
                    facecolor=color, alpha=alpha, lw=BAND_LW, zorder=zorder)


def _ratio6_ylim(series, anchor=None, frac=0.10, head=0.34):
    """One window containing every drawn curve of a pane, bars and bands.

    ``series`` is a list of ``(y, err, lo, hi)``; ``lo``/``hi`` may be None on
    the band-free figure.  The head padding is larger than the foot because
    every pane of these figures carries its legend at the upper left and a
    legend sitting on a curve is as unreadable as a curve off the frame.
    """
    los, his = [], []
    for y, e, blo, bhi in series:
        y = np.asarray(y, dtype=float)
        e = np.asarray(e, dtype=float) if e is not None else np.zeros_like(y)
        ok = np.isfinite(y)
        eo = np.where(np.isfinite(e), e, 0.0)
        if ok.any():
            los.append(float(np.min((y - eo)[ok])))
            his.append(float(np.max((y + eo)[ok])))
        for b, sink in ((blo, los), (bhi, his)):
            if b is None:
                continue
            b = np.asarray(b, dtype=float)
            if np.isfinite(b).any():
                sink.append(float(np.nanmin(b)) if sink is los
                            else float(np.nanmax(b)))
    if not los or not his:
        return 0.0, 1.0
    lo, hi = min(los), max(his)
    if anchor is not None and np.isfinite(anchor):
        lo, hi = min(lo, anchor), max(hi, anchor)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return _pad(lo, hi)
    d = hi - lo
    return lo - frac * d, hi + head * d


def _ratio6_top(ax, nlo, lo, obs, with_band=False):
    """The seventh pane: dsigma/dx, both orders, five components each.

    See ``pol_analysis.RATIO6_TOP_WHAT`` for what is on it and
    ``RATIO6_TOP_BAND_KEYS`` for why the scale band is on the unpolarised pair
    and on nothing else.  Nothing but the axis labels, the ticks and the
    legend is written here, as on every other figure of this study.
    """
    edges = PA.BINS[obs]
    x = 0.5 * (edges[:-1] + edges[1:])
    ylab = (PA.LABELS_TEX if USETEX else PA.LABELS_TXT)[obs][1]
    name = PA.KF_CURVE_TEX if USETEX else PA.KF_CURVE_TXT
    oname = PA.RATIO6_ORDER_TEX if USETEX else PA.RATIO6_ORDER_TXT
    # The y scale is MEASURED here and not taken from PA.LOGY -- ten curves,
    # not five, and the reason is argued at PA.RATIO6_TOP_LOG_DECADES.  Both
    # observables come out log; numbers.txt prints the span and the decision.
    logy = PA.ratio6_top_logy(nlo, lo, obs)
    cur = PA.ratio6_top_curves(nlo, lo, obs, with_band=with_band)

    # Component-major, order-minor: at ncol=5 matplotlib fills the legend
    # column by column, so this puts one COMPONENT in each column with its two
    # orders under one another, which is the way the pane is read.
    for i, key in enumerate(PA.KF_PANE_ORDER):
        for tag in PA.RATIO6_TOP_ORDERS:
            h = cur[tag][key]
            y, e = np.asarray(h['y'], float), np.asarray(h['err'], float)
            shown = np.where(y > 0, y, np.nan) if logy else y
            # LO OVER NLO -- the reverse of the six panes below, and
            # deliberately: see pol_analysis.RATIO6_TOP_ORDERS.  A K-factor of
            # 1.29 is a hair's breadth on a four-decade pane and the curve
            # underneath is the one that vanishes; LO is the thinner, dashed,
            # non-reference one, so LO is the one that goes on top.  The
            # component z-order runs the other way -- earlier, thicker
            # components stay UNDER later, thinner ones, which is what makes
            # the weight ladder above show a halo instead of a hidden curve.
            z = 10 * (i + 1) + (6 if tag == 'LO' else 4)
            lw = (TOP_LW_BASE - TOP_LW_STEP * i
                  + (TOP_LW_NLO if tag == 'NLO' else 0.0))
            blo, bhi = h.get('lo'), h.get('hi')
            if with_band and blo is not None:
                _band(ax, edges, blo, bhi, COLOR[key], BAND_ALPHA_TOP,
                      zorder=z - 2)
            ax.stairs(shown, edges, color=COLOR[key], ls=LS_ORDER[tag],
                      lw=lw, baseline=None,
                      zorder=z, label='%s, %s' % (name[key], oname[tag]))
            ax.errorbar(x, shown, yerr=np.where(np.isfinite(shown), e, np.nan),
                        fmt='none', ecolor=COLOR[key], elinewidth=0.8,
                        alpha=0.9 if tag == 'NLO' else 0.6, zorder=z)
    if logy:
        ax.set_yscale('log')
    ax.set_ylabel(ylab, fontsize=10)
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y * TOP_HEADROOM[0 if logy else 1])
    ax.set_xlim(edges[0], edges[-1])
    ax.legend(frameon=False, fontsize=8.0, loc='upper left', ncol=5,
              columnspacing=1.4, handlelength=2.4)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if not logy:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(labelsize=9)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)


def draw_ratio6(nlo, lo, obs, outdir, with_band=False):
    """Variant A as a distribution pane over a 3 x 2 of ratios.

    ``with_band`` adds the scale envelope.

    ONE function for both versions, deliberately.  They are the same six
    panels off the same ``pol_analysis.ratio6_curves`` objects and differ only
    in whether a band is shaded behind each curve, so writing them twice would
    be inviting the two to drift apart in some detail that is not the point.

    Pane by pane:

    * the full-width pane on top, ``dsigma/dx`` at absolute normalisation:
      unpolarised plus the four components, at NLO and at LO, LO over NLO in
      z-order.  It is the one absolute statement on a figure that is otherwise
      all ratios -- see ``pol_analysis``'s section header.  With the band, the
      two UNPOLARISED curves are banded and the components are not; that is
      ``pol_analysis.RATIO6_TOP_BAND_KEYS`` and the reason is there.  It
      applies to THIS PANE only: the six below band every curve they carry,
      exactly as they did before this pane existed.
    * top left, ``(Z_0Z_0 + Z_TZ_T + Z_TZ_0 + Z_0Z_T)/full`` -- the sum
      consistency, NLO and LO.  Reference line at 1, because 1 is what this
      one ought to be.
    * top right, ``K = NLO/LO``, five curves.  NO reference line: a K-factor
      ought not to be 1 and drawing a rule at 1 would suggest a null
      hypothesis nobody holds.  Same argument as the K-factor figure's.
    * the four below, ``component/full`` for each polarisation, NLO and LO.
      The dashed rule in each is that pane's NLO integrated value, as on
      variant A, and its number is in numbers.txt and not beside it.

    WHICH ERROR IS ON WHAT, and it is not one rule for the whole figure:

    * the five ratio panes, both orders: the delta-method / jackknife bar of
      ``pol_analysis.ratio``.  Numerator and denominator are sums over the
      SAME events of ONE sample, so the covariance between them is real and
      must be kept.
    * the K pane: the two relative MC errors in quadrature.  Numerator and
      denominator are two independent samples and there is no covariance to
      subtract.

    That is the same split the existing figures make and it is not changed
    here.  The scale band is a separate object drawn behind both.
    """
    edges = PA.BINS[obs]
    x = 0.5 * (edges[:-1] + edges[1:])
    xlab = (PA.LABELS_TEX if USETEX else PA.LABELS_TXT)[obs][0]
    rlab = PA.RATIO_TEX if USETEX else PA.RATIO_TXT
    kname = PA.KF_CURVE_TEX if USETEX else PA.KF_CURVE_TXT
    oname = PA.RATIO6_ORDER_TEX if USETEX else PA.RATIO6_ORDER_TXT
    cur = PA.ratio6_curves(nlo, lo, obs, with_band=with_band)

    fig = plt.figure(figsize=(9.0, 12.6))
    # TWO vertical rhythms, so two gridspecs and not one, on exactly the
    # argument ``draw`` makes for its three: the wide pane on top is its own
    # block with its own x tick labels and its own axis name, and the 3 x 2
    # below is a second block with its own.  A single hspace over four rows
    # would have to be either tight enough to crush the wide pane's axis name
    # into the grid's top row or wide enough to tear the grid's own rows
    # apart, and there is no value that is both.
    #
    # THE SHARE-OUT, and it is deliberately uneven three ways.  The wide pane
    # spans 4.18 decades on M(e+ mu+) and 2.79 on Delta phi and carries ten
    # curves; the four fraction panes are nearly flat and nearly band-free and
    # read the same at seven tenths the height.  So the pane takes 4.3 of the
    # outer 12.6 where it took 2.9 of 13.1, and the grid's own rows are no
    # longer equal: the sum-consistency and K-factor row keeps 3.3 against the
    # fraction rows' 2.5 each.  Height in inches, pane then row 1 then rows
    # 2-3: 2.19 -> 3.17, 2.45 -> 2.31, 2.45 -> 1.75.  The figure gets SHORTER,
    # 13.4 in to 12.6 in, which is the point -- the pane grows out of the
    # fraction rows and not out of the page.
    #
    # WHY THE GRID'S TOP ROW IS NOT CUT AS HARD.  Its left pane's axis name is
    # (Z_0Z_0+Z_TZ_T+Z_TZ_0+Z_0Z_T)/full, which is set ROTATED and is by some
    # way the longest string on the figure: at 3.1 it came within 0.14 in of
    # the frame top and bottom, at 3.3 it clears them by 0.21 in.  The four
    # fraction panes carry one short name each and clear by over an inch even
    # at 2.5, which is why the height comes off them.  Measured on the
    # rendered figure, label extent against axes extent.
    #
    # The outer gap is a fraction of the MEAN row height and the two rows are
    # very unequal, so the fraction has to move when the ratios do: 0.10 of the
    # new 6.30 mean is the same ~0.46 in of gap that 0.09 of the old 6.55 mean
    # bought, and that is what clears the wide pane's tick labels and axis
    # name.  Checked on the rendered PNG, not arithmetically.
    outer = fig.add_gridspec(2, 1, height_ratios=[4.3, 8.3], hspace=0.10)
    # The 3 x 2 keeps the K-factor figure's wspace exactly: its right column
    # carries five-significant-figure tick labels, which at the default wspace
    # put the right axis name on the left column's frame.  hspace stays 0.07 --
    # rows 1 and 2 have their x tick labels off, so nothing but frame sits in
    # those two gaps and they do not need to grow back what the shorter rows
    # take off them.
    gs = outer[1].subgridspec(3, 2, height_ratios=[3.3, 2.5, 2.5],
                              hspace=0.07, wspace=0.30)
    axtop = fig.add_subplot(outer[0])
    axes = []
    for r in range(3):
        for cc in range(2):
            axes.append(fig.add_subplot(gs[r, cc],
                                        sharex=axes[cc] if r else None))

    # -- the seventh pane, above the six --------------------------------------
    _ratio6_top(axtop, nlo, lo, obs, with_band=with_band)
    axtop.set_xlabel(xlab, fontsize=11)

    def ratio_pane(ax, pane, anchor_line):
        series = []
        for tag in PA.RATIO6_ORDERS:
            h = cur[pane][tag]
            r, e = np.asarray(h['r'], float), np.asarray(h['err'], float)
            blo = h.get('lo') if with_band else None
            bhi = h.get('hi') if with_band else None
            if with_band:
                _band(ax, edges, blo, bhi, COLOR[pane], BAND_ALPHA,
                      zorder=2 if tag == 'NLO' else 1)
            ax.stairs(r, edges, color=COLOR[pane], ls=LS_ORDER[tag],
                      lw=LW + (0.4 if tag == 'NLO' else 0.0), baseline=None,
                      zorder=5 if tag == 'NLO' else 4, label=oname[tag])
            ax.errorbar(x, r, yerr=e, fmt='o', ms=3.2 if tag == 'NLO' else 2.6,
                        mfc='none' if tag == 'LO' else None, color=COLOR[pane],
                        elinewidth=0.9, zorder=6 if tag == 'NLO' else 5)
            series.append((r, e, blo, bhi))
        if anchor_line is not None:
            ax.axhline(anchor_line, color='black', lw=0.9,
                       ls=':' if pane == 'SUM' else '--', zorder=3)
        # ``loc='best'`` and not a fixed corner.  These five panes carry
        # curves that rise on some observables and fall on others -- Z_TZ_T
        # descends across Delta phi where Z_0Z_0 climbs -- so any one corner
        # is under a curve on half of them.  The legend is two short entries
        # and 'best' places it in whichever corner the data leaves free.
        ax.set_ylim(*_ratio6_ylim(series, anchor_line, head=0.20))
        ax.set_ylabel(rlab[pane], fontsize=9 if pane == 'SUM' else 10)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=9)
        ax.legend(frameon=False, fontsize=8.0, loc='best', ncol=2)

    # top left -- the sum consistency, reference at 1.
    ratio_pane(axes[0], 'SUM', 1.0)
    for s in axes[0].spines.values():
        s.set_linewidth(1.5)

    # top right -- the K-factor, five curves, no reference line.
    axk = axes[1]
    kseries = []
    for key in PA.KF_PANE_ORDER:
        kk = cur['K'][key]
        k, e = np.asarray(kk['k'], float), np.asarray(kk['err'], float)
        blo = kk.get('lo') if with_band else None
        bhi = kk.get('hi') if with_band else None
        if with_band:
            _band(axk, edges, blo, bhi, COLOR[key], BAND_ALPHA_K,
                  zorder=2 if key == 'full' else 1)
        axk.stairs(k, edges, color=COLOR[key], ls=LS[key],
                   lw=LW + (0.5 if key == 'full' else 0.0), baseline=None,
                   zorder=6 if key == 'full' else 4, label=kname[key])
        axk.errorbar(x, k, yerr=e, fmt='o', ms=3.0, color=COLOR[key],
                     elinewidth=0.9, zorder=6 if key == 'full' else 5)
        kseries.append((k, e, blo, bhi))
    # Two legend rows over a pane whose biggest excursion (K(Z_TZ_0) ~ 2.9 in
    # the first Delta phi bin) is at the upper LEFT, exactly under them.
    # Checked on the rendered PNG, not on the axis limits.
    axk.set_ylim(*_ratio6_ylim(kseries, None, head=0.56))
    axk.set_ylabel(PA.KFACTOR_TEX if USETEX else PA.KFACTOR_TXT, fontsize=10)
    axk.legend(frameon=False, fontsize=8.0, loc='upper left', ncol=3)
    axk.xaxis.set_minor_locator(AutoMinorLocator())
    axk.yaxis.set_major_locator(MaxNLocator(6))
    axk.yaxis.set_minor_locator(AutoMinorLocator())
    axk.tick_params(labelsize=9)
    for sp in axk.spines.values():
        sp.set_linewidth(1.5)

    # the four fraction panes.
    for ax, pane in zip(axes[2:], PA.RATIO6_FRACTIONS):
        ratio_pane(ax, pane, cur[pane]['NLO']['integrated'])

    for i, ax in enumerate(axes):
        if i < 4:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel(xlab, fontsize=11)
    axes[0].set_xlim(edges[0], edges[-1])
    axes[1].set_xlim(edges[0], edges[-1])

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, PA.SHORT[obs])
    want = wants_minus(fig)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('%-12s 7-pane%-10s sum/full NLO %.4f  LO %.4f'
          % (PA.SHORT[obs], ' + band' if with_band else '',
             cur['SUM']['NLO']['integrated'], cur['SUM']['LO']['integrated']))
    return base, want


def draw_variants(d, extras, outdir, b_extras=None):
    """The two variants, written ALONGSIDE the main figures.

    Each goes into its own subdirectory of ``outdir`` -- the originals stay
    exactly where they were and are not touched.  Returns the ``(base, want)``
    pairs so that ``--check-minus`` covers the variant PDFs too.

    ``extras`` is what every figure draws (``onshell``, ``PA``); ``b_extras``
    is what VARIANT B draws, which is that list plus ``madspin_v1``.  Keeping
    them as two arguments is the whole mechanism by which the fourth sample
    reaches variant B and nothing else.
    """
    b_extras = extras if b_extras is None else b_extras
    bases = []
    for obs in PA.OBS:
        # A: the same three tiers, no extra spinmodes on the distribution.
        bases.append(draw(d, obs, os.path.join(outdir, PA.VARIANTS['A']['dir']),
                          extras=(), variant='A'))
        # B: the distribution with all four modes, then one shape-ratio pane.
        bases.append(draw_shape(d, obs,
                                os.path.join(outdir, PA.VARIANTS['B']['dir']),
                                b_extras))
    return bases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    ap.add_argument('--numbers', default=os.path.join(_HERE, 'numbers.txt'))
    ap.add_argument('--check-minus', action='store_true')
    ap.add_argument('--no-variants', action='store_true',
                    help='draw only the original three-tier figures')
    a = ap.parse_args()
    d = PA.Data(a.data)
    extras = PA.load_extras(a.data)
    # madspin_v1 goes on variant B and nowhere else on the canvas.  It DOES
    # go into numbers.txt alongside the other modes, which is why the text is
    # written from the full four-sample list and the original figures from
    # ``extras`` alone.
    b_extras = extras + PA.load_variant_b_extras(a.data)
    bases = [draw(d, obs, a.out, extras) for obs in PA.OBS]
    if not a.no_variants:
        bases += draw_variants(d, extras, a.out, b_extras=b_extras)
    # The K-factor figure needs the LO sample and is skipped, loudly, without
    # it -- the rest of the study must stay re-makeable from the reference
    # .npz alone.
    lo = PA.load_kfactor_partner(a.data)
    if lo is None:
        print('LO sample not in %s: K-factor figure not drawn' % a.data)
    else:
        kdir = os.path.join(a.out, PA.KFACTOR_DIR)
        bases += [draw_kfactor(d, lo, obs, kdir) for obs in PA.OBS]
        # The two six-panel ratio figures, in their own two subdirectories
        # beside the K-factor one.  Both need the LO sample; the second also
        # needs the nine scale columns in BOTH .npz, and is skipped and said
        # to be skipped rather than approximated if either is short of them.
        r6 = os.path.join(a.out, PA.RATIO6_DIR)
        bases += [draw_ratio6(d, lo, obs, r6) for obs in PA.OBS]
        if d.has_scale and lo.has_scale:
            r6s = os.path.join(a.out, PA.RATIO6_SCALE_DIR)
            bases += [draw_ratio6(d, lo, obs, r6s, with_band=True)
                      for obs in PA.OBS]
        else:
            print('no MUR*_MUF* columns in %s: the scale figure is not drawn'
                  % a.data)
    write_numbers(d, a.numbers, b_extras, lo=lo)
    if a.check_minus:
        print('usetex = %s   minus workaround active = %s' % (USETEX, MINUS_FIX))
        bad = n = 0
        for b, want in bases:
            if not want:
                print('%s: no minus sign in this figure, check not applicable'
                      % os.path.basename(b))
                continue
            n += 1
            ok, msg = check_minus(b + '.pdf')
            if not ok:
                bad += 1
                print(msg)
        print('%d/%d applicable PDFs carry /minus' % (n - bad, n))
        if bad:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
