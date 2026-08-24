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
from plot_zz_stack import wants_minus                            # noqa: E402

COLOR = {'full': 'black', 'LL': 'blue', 'TT': 'red',
         'TL': allcolors[2], 'LT': allcolors[4], 'SUM': allcolors[5],
         # The two extra spinmodes are full (unpolarised) distributions, like
         # the black one and unlike the four components, so they take the two
         # hues the components leave free rather than a shade of one of them.
         'onshell': 'darkorange', 'PA': allcolors[9]}
LS = {'full': 'solid', 'LL': 'dashed', 'TT': 'dashdot',
      'TL': (0, (1, 1.4)), 'LT': (0, (5, 1.5, 1, 1.5)), 'SUM': 'solid',
      'onshell': (0, (6, 2)), 'PA': (0, (3, 1, 1, 1, 1, 1))}

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
# drawn inside the pane at the upper left: seven curves on the full figure and
# on variant B, five on variant A because the two extra spinmodes are not
# there.  All of these were checked on the rendered PNG and not on the axis
# limits.
HEADROOM = {None: (250.0, 1.62), 'A': (60.0, 1.45), 'B': (250.0, 1.62)}


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
    both extra spinmodes at once -- each curve divided by its OWN cross section
    before the division, so that the rate difference between the modes divides
    out and what is left is the shape.  That is the pane this data wants:
    ``onshell`` differs from ``madspin`` in RATE (+4.99 % inclusive) and much
    less in shape, ``PA`` agrees in rate (1.4 sigma) and differs in SHAPE.  A
    ratio that still carried the normalisation would show the first as a large
    flat offset and would hide the second under it.

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
    axr.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=2)
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
    A('for Y = onshell and Y = PA, both in one pane.  Each mode is divided by')
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
    A('   %-10s %-9s %12s %14s %14s %10s'
      % ('observable', 'spinmode', 'N outside', 'sigma_selected',
         'sigma_inrange', 'outside %'))
    for obs in PA.OBS:
        for lab, dd in [('madspin', d)] + list(extras):
            sd = PA.shape_density(dd, obs, 'inrange')
            A('   %-10s %-9s %12d %14.6f %14.6f %10.3f'
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
    A('3120 / 3088 / 3134 quoted there as lying above 450 GeV are events with')
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
    A('is NO correlation to keep -- see "DO THE THREE SAMPLES SHARE PRODUCTION')
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
    A('For the contrast, "THE THREE SPINMODES COMPARED" above quotes a')
    A('"SHAPE only" chi2 as well.  It is a DIFFERENT statistic and the two')
    A('are not interchangeable: that one rescales the other mode to the')
    A('reference\'s SELECTED fiducial rate and keeps the plain sqrt(sum w^2)')
    A('per-bin bar, so it neither uses the in-range normalisation nor removes')
    A('the within-sample correlation.  Its bars are the larger and its chi2')
    A('the smaller of the two.  The pane draws the one computed here.')
    A('')


def write_numbers(d, path, extras=()):
    out = []
    A = out.append
    m = d.meta
    A('MadSpin polarisation weights on a showered NLO  p p > z z')
    A('=' * 74)
    A('Three samples: the same process, the same run card and the same 250k')
    A('showered events, differing only in MadSpin\'s spinmode.  The')
    A('polarisation decomposition below is the REFERENCE (madspin, run_06)')
    A('sample\'s; the other two enter the distribution pane of each figure and')
    A('the cross-mode comparison section, and nowhere else.')
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
    A('%-9s %-34s %9s %9s %8s %8s'
      % ('spinmode', 'input', 'on disk', 'inflated', 'pass s', 'events'))
    for mm in [m] + [x.meta for _, x in extras]:
        A('%-9s %-34s %8.3fG %8.3fG %8.1f %8d'
          % (mm.get('label', 'madspin'),
             mm['input_path'].split('Events/')[-1].replace(
                 '/events_PYTHIA8_0', '/...'),
             mm['input_gb'], mm.get('inflated_gb', mm['input_gb']),
             mm['pass_seconds'], mm['n_events']))
    A('All three .gz inputs were streamed through "gzip -dc" in a child')
    A('process and never unpacked to disk: they inflate to ~10.6 GB each and')
    A('no temporary of that size should have to exist.  The three passes were')
    A('run concurrently on separate cores, so the times above are within a')
    A('second of each other and none of them is a serial measurement.')
    A('%d lines read from the reference file, %s' % (m['lines_read'],
      m['hepmc_flavour']))
    A('code       %s' % m['code_sha'])
    A('')
    A('--- the %d weight names on every N line (asserted, not sampled) ---'
      % len(m['weight_names']))
    for i in range(0, len(m['weight_names']), 6):
        A('   ' + '  '.join('%-18s' % n for n in m['weight_names'][i:i + 6]))
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
    A('The MadSpin card of all three runs is')
    A('  decay z > e+ e-')
    A('  decay z > mu+ mu-')
    A('which is exclusive in INTENT.  Whether it is exclusive in EFFECT is a')
    A('different question and it decides what these figures mean, so it is')
    A('read off the event record -- mark each z\'s end vertex, follow the chain')
    A('of status-44 copies, take the particles born at the final vertex -- for')
    A('every event of every file:')
    A('')
    A('   %-9s %10s %14s %14s %10s'
      % ('spinmode', 'events', 'one z->e+e- AND', 'at least one', 'other'))
    A('   %-9s %10s %14s %14s %10s'
      % ('', '', 'other z->mu+mu-', 'z->e+e-', 'channels'))
    for lab, dd in [('madspin', d)] + list(extras):
        zz1, zz2 = dd.z['z1_ch'], dd.z['z2_ch']
        q4 = ((zz1 == 11) & (zz2 == 13)) | ((zz1 == 13) & (zz2 == 11))
        qe = (zz1 == 11) | (zz2 == 11)
        A('   %-9s %10d %8d %5.4g%% %8d %5.4g%% %10d'
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
    A('rates and only the ratios between the three modes are.')
    A('')
    if extras:
        A('--- THE OTHER TWO SPINMODES ---')
        A('Checked, not assumed, on each file separately.')
        A('')
        same = all(x.meta['weight_names'] == m['weight_names']
                   for _, x in extras)
        A('The N line.  Every one of the 250 000 N lines in each file was')
        A('compared against that file\'s first, as for the reference; they are')
        A('constant within each file.  Across files: the three name')
        A('%s'
          % ('the SAME %d weights, in the SAME order' % len(m['weight_names'])
             if same else 'DIFFERENT weight sets -- see each meta.json'))
        A('-- including the four ms_pol_* ones, because all three runs set')
        A('keep_weight_for_polarization_vector = [0, T].  That the onshell and')
        A('PA files carry polarisation weights is recorded here and nothing in')
        A('this study is built on it: their decomposition is a separate piece')
        A('of work, and the ratio panes belong to the sample whose')
        A('decomposition they are.')
        A('')
        A('The nominal, re-derived per file rather than carried over:')
        A('')
        A('%-9s %22s %22s %14s %10s'
          % ('spinmode', '"0"*N/"Weight" range', 'sum("0") = mean("W")',
             'C last', 'C first'))
        for lab, dd in [('madspin', d)] + list(extras):
            e = dd.nominal_evidence()
            A('%-9s %10.6f %10.6f %22.9f %14.9f %10.3e'
              % (dd.label, e['w0_times_N_over_Weight_min'],
                 e['w0_times_N_over_Weight_max'], e['sum_w0_pb'],
                 e['C_line_last_event_pb'], e['C_line_first_event_pb']))
        A('')
        A('Same answer on all three: "0" * N_events == "Weight" to the last')
        A('bit on every event, sum("0") == mean("Weight") == the C line of the')
        A('LAST event to 1e-11, and the C line of the FIRST event is Pythia\'s')
        A('running estimate after one event -- low by 2.2e5 in every file.  So')
        A('the reference file\'s conclusion holds for the other two, and it')
        A('holds because it was re-checked and not because it was carried')
        A('over.  The distributions use "0" (its sum IS the cross section in')
        A('pb) and each sample is scaled by its OWN 1/n_events.')
        A('')
        A('%-9s %14s %12s %10s %s'
          % ('spinmode', 'sigma_tot pb', 'negatives', 'MUR1/W-1',
             'distinct |Weight|'))
        for lab, dd in [('madspin', d)] + list(extras):
            e = dd.nominal_evidence()
            A('%-9s %14.6f %11.3f %% %10.2e %s'
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
        A('--- DO THE THREE SAMPLES SHARE PRODUCTION EVENTS? ---')
        A('This decides the error bar on every mode-to-mode comparison in this')
        A('file, so it is established and not assumed.  If the three runs had')
        A('decayed one common set of production events, the mode-to-mode ratio')
        A('would be CORRELATED, a paired error would be correct, and it would')
        A('be much smaller than the quadrature one.  They did not.')
        A('')
        pe = PA.pairing_evidence(d, extras)
        A('%-9s %10s %12s %14s %18s'
          % ('spinmode', 'events', 'n(w < 0)', 'sigma_tot pb',
             'corr(m_4l) row by row'))
        for row in pe['rows']:
            A('%-9s %10d %12d %14.6f %18s'
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
        A('spinmode or on the order the events are written in.  Three decays of')
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
        A('--- THE THREE SPINMODES COMPARED ---')
        A('Same lepton selection, same bin edges, same absolute normalisation.')
        A('Errors here are the PLAIN quadrature sum, unlike every ratio pane in')
        A('this file: these are three independent MadSpin runs, independently')
        A('showered, so there is no covariance to keep -- established above,')
        A('not assumed.')
        A('')
        A('inclusive cross section (all 250k events, no selection):')
        for lab, dd in [('madspin', d)] + list(extras):
            w0 = np.asarray(dd.z['w_0'], dtype=np.float64)
            A('   %-9s %12.6f +- %.6f pb' % (dd.label, w0.sum(),
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
        A('   %-9s %12s %10s %12s %12s %10s'
          % ('spinmode', 'sigma_tot', 'f(z->ee)', 'product', 'sigma_fid',
             'acceptance'))
        for lab, dd in [('madspin', d)] + list(extras):
            zz1, zz2 = dd.z['z1_ch'], dd.z['z2_ch']
            fee = float(((zz1 == 11) | (zz2 == 11)).mean())
            st = float(np.asarray(dd.z['w_0'], dtype=np.float64).sum())
            fid = PA.full_distribution(dd, 'dphi_ee_dr')['sigma_pb']
            A('   %-9s %12.6f %10.6f %12.6f %12.6f %10.4f'
              % (dd.label, st, fee, st * fee, fid, fid / (st * fee)))
        A('')
        A('which z decayed to what, off the event record (truth, used to')
        A('categorise only -- no plotted observable touches it):')
        A('   %-9s %10s %10s %10s %10s'
          % ('spinmode', 'n(4 lepton)', 'frac %', 'n(z->ee)', 'frac %'))
        for lab, dd in [('madspin', d)] + list(extras):
            z1, z2 = dd.z['z1_ch'], dd.z['z2_ch']
            q4 = ((z1 == 11) & (z2 == 13)) | ((z1 == 13) & (z2 == 11))
            qe = (z1 == 11) | (z2 == 11)
            A('   %-9s %10d %10.4f %10d %10.4f'
              % (dd.label, q4.sum(), 100 * q4.mean(), qe.sum(),
                 100 * qe.mean()))
        A('')
        for obs in PA.OBS:
            ref = PA.full_distribution(d, obs)
            rerr = _sigma_err(d, obs)
            A('%s: N selected, fiducial cross section, and the comparison'
              % PA.SHORT[obs])
            A('   %-9s N=%6d  sigma_fid=%.6f +- %.6f pb   (the reference)'
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
                A('   %-9s N=%6d  sigma_fid=%.6f +- %.6f pb'
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
                 else '   (all three modes drawn)'))
            if sparse:
                A('   Fewer than %d selected events in this observable for at'
                  % PA.MIN_SEL_TO_DRAW)
                A('   least one mode.  At that statistics the per-bin errors')
                A('   are tens of percent and three curves laid over one')
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
    open(path, 'w').write('\n'.join(out) + '\n')
    print('wrote %s' % path)


def draw_variants(d, extras, outdir):
    """The two variants, written ALONGSIDE the main figures.

    Each goes into its own subdirectory of ``outdir`` -- the originals stay
    exactly where they were and are not touched.  Returns the ``(base, want)``
    pairs so that ``--check-minus`` covers the variant PDFs too.
    """
    bases = []
    for obs in PA.OBS:
        # A: the same three tiers, no extra spinmodes on the distribution.
        bases.append(draw(d, obs, os.path.join(outdir, PA.VARIANTS['A']['dir']),
                          extras=(), variant='A'))
        # B: the distribution with all three modes, then one shape-ratio pane.
        bases.append(draw_shape(d, obs,
                                os.path.join(outdir, PA.VARIANTS['B']['dir']),
                                extras))
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
    bases = [draw(d, obs, a.out, extras) for obs in PA.OBS]
    if not a.no_variants:
        bases += draw_variants(d, extras, a.out)
    write_numbers(d, a.numbers, extras)
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
