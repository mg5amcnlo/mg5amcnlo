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
ratio panes are the polarisation decomposition of the default (``madspin``)
sample; putting a second sample's numerator over a first sample's denominator
would be meaningless, and each extra sample's own decomposition is a different
study.  A spinmode curve is drawn only where the selection leaves enough events
for it to be a measurement -- see ``PA.MIN_SEL_TO_DRAW``; where it does not, the
pane says so in as many words instead of carrying three noise realisations that
the eye would read as a mode difference.

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

Usage::

    plot_zz_pol.py [--data DIR] [--out DIR] [--numbers PATH] [--check-minus]
"""

import argparse
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

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

TITLE_TEX = (r'$pp \to ZZ$ [QCD] (MC@NLO) $+$ MadSpin, showered; '
             r'13 TeV, 250k events')
TITLE_TXT = 'p p > z z [QCD] (MC@NLO) + MadSpin, showered; 13 TeV, 250k events'


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


def draw(d, obs, outdir, extras=()):
    c = PA.Curves(d, obs)
    edges, x = c.edges, c.centres()
    xlab, ylab = (PA.LABELS_TEX if USETEX else PA.LABELS_TXT)[obs]
    curve = PA.CURVE_TEX if USETEX else PA.CURVE_TXT
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
    # last bin, four decades under the total, so the plain 1.6x headroom that
    # works on a linear pane leaves the legend sitting on the curves.  One
    # column throughout now that the labels are short.
    ax.set_ylim(lo, hi * (60.0 if obs in PA.LOGY else 1.62))
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    if dropped:
        # Said on the figure, not only in numbers.txt.  A reader who sees two
        # of three samples must be told that the other two exist and why they
        # are absent, or the figure implies a comparison was never made.
        who = ' and '.join('%s (N=%d)' % (k, n) for k, n in dropped)
        msg = ('spinmode %s not drawn: too few events\nfor a comparison at '
               'this selection' % who)
        # Right, and below the legend rather than beside it: the legend's
        # longest entry now names a spinmode and reaches past the middle of
        # the pane.  Low enough to clear it, high enough to stay off the
        # curves -- on the log pane this note only ever appears on, the floor
        # is where the Z_0Z_0 component runs.  Checked on the rendered PNG.
        ax.text(0.985, 0.78, msg, transform=ax.transAxes, fontsize=8,
                ha='right', va='top', color='0.35')
    ax.set_title(TITLE_TEX if USETEX else TITLE_TXT, fontsize=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if obs not in PA.LOGY:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.setp(ax.get_xticklabels(), visible=False)

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
    # The value, not its significance: the sigma-from-1 lives in numbers.txt
    # and RESULTS.md, where it can be read next to the sample it belongs to.
    note = ((r'integrated: $%.4f \pm %.4f$' % (Rint, Eint)) if USETEX else
            ('integrated: %.4f +- %.4f' % (Rint, Eint)))
    axs.text(0.015, 0.06, note, transform=axs.transAxes, fontsize=8.5,
             ha='left', va='bottom')
    axs.text(0.985, 0.94,
             'polarisation interference' if not USETEX
             else r'\textbf{polarisation interference}',
             transform=axs.transAxes, fontsize=9, ha='right', va='top',
             color=COLOR['SUM'])
    axs.set_xlabel(xlab, fontsize=12)
    for s in axs.spines.values():
        s.set_linewidth(1.5)

    # -- tier 3: the breakdown, one polarisation per pane --------------------
    for a, k in zip(small, ['LL', 'TT', 'TL', 'LT']):
        rk, ek, _ = c.ratios[k]
        Rk, Ek = c.integrated[k]
        lab = (r'integrated $%.4f$' % Rk) if USETEX else 'integrated %.4f' % Rk
        a.axhline(Rk, color='black', lw=0.9, ls='--', label=lab)
        a.stairs(rk, edges, color=COLOR[k], ls=LS[k], lw=LW, zorder=4,
                 baseline=None)
        a.errorbar(x, rk, yerr=ek, fmt='o', ms=3.0, color=COLOR[k],
                   elinewidth=0.9, zorder=5)
        a.set_ylim(*_ratio_ylim(rk, ek, Rk))
        a.set_ylabel(rlab[k], fontsize=10)
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_minor_locator(AutoMinorLocator())
        # A legend rather than a fixed-position text: these panes have no
        # common shape -- LL rises across Delta phi and falls across the mass,
        # TT does the opposite -- so any one corner is under the data in some
        # of them.  ``loc='best'`` picks the empty one per pane.
        a.legend(loc='best', frameon=False, fontsize=8)
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
    print('%-12s N=%d  sum/full = %.4f +- %.4f' % (PA.SHORT[obs], c.n_sel,
                                                   Rint, Eint))
    return base, want


# --------------------------------------------------------------------------
def _sigma_err(dd, obs):
    """``sqrt(sum w^2)`` on one sample's fiducial cross section, in pb."""
    w = np.asarray(dd.full, dtype=np.float64)[dd.sel[obs]] * dd.scale_to_pb
    return float(np.sqrt((w ** 2).sum()))


def write_numbers(d, path, extras=()):
    out = []
    A = out.append
    m = d.meta
    A('MadSpin polarisation weights on a showered NLO  p p > z z')
    A('=' * 74)
    A('Three samples: the same process, the same run card and the same 250k')
    A('showered events, differing only in MadSpin\'s spinmode.  The')
    A('polarisation decomposition below is the DEFAULT (madspin) sample\'s;')
    A('the other two enter the distribution pane of each figure and the')
    A('cross-mode comparison section, and nowhere else.')
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
    A('The two .gz inputs were streamed through "gzip -dc" in a child process,')
    A('never unpacked to disk; the pass cost the same as the plain file\'s.')
    A('%d lines read from the reference file, %s' % (m['lines_read'],
      m['hepmc_flavour']))
    A('code       %s' % m['code_sha'])
    A('')
    A('--- the 33 weight names on every N line (asserted, not sampled) ---')
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
    A('     -- 249 999 of them are the banner XWGTUP and the second value is a')
    A('     SINGLE event, not a second MC@NLO branch; see the spinmode section')
    A('     below, where the other two files each carry exactly one value.')
    A('  %d of %d events carry a NEGATIVE weight (%.3f %%).'
      % (ev['n_negative'], ev['n_events'], 100 * ev['frac_negative']))
    A('')
    A('--- THE SAMPLE IS NOT p p > z z > e+e- mu+mu- ---')
    A('The MadSpin card is  decay z > light light  with')
    A('  define light = 1 2 3 4 5 -1 -2 -3 -4 -5 11 12 13 14 15 16 -11 ... -16')
    A('i.e. every fermion but the top: an INCLUSIVE Z decay.  Read off the')
    A('event record, of the 250 000 events')
    z1, z2 = d.z['z1_ch'], d.z['z2_ch']
    t4 = ((z1 == 11) & (z2 == 13)) | ((z1 == 13) & (z2 == 11))
    tee = (z1 == 11) | (z2 == 11)
    A('  %6d (%.4f %%) have one z -> e+e- AND the other z -> mu+mu-'
      % (t4.sum(), 100 * t4.mean()))
    A('  %6d (%.3f %%) have at least one z -> e+e-'
      % (tee.sum(), 100 * tee.mean()))
    A('M(e+ mu+) therefore exists for two events in a thousand, and its figure')
    A('is statistics limited by the decay card and by nothing else.')
    A('')
    if extras:
        A('--- THE OTHER TWO SPINMODES ---')
        A('Checked, not assumed, on each file separately.')
        A('')
        A('The N line.  Every one of the 250 000 N lines in each file was')
        A('compared against that file\'s first, as for the reference; they are')
        A('constant within each file.  All three files name the SAME 33')
        A('weights, in the same order -- including the four ms_pol_* ones,')
        A('because all three runs set keep_weight_for_polarization_vector')
        A('= [0, T].  That the onshell and PA files carry polarisation weights')
        A('is recorded here and nothing in this study is built on it: their')
        A('decomposition is a separate piece of work, and the ratio panes')
        A('belong to the sample whose decomposition they are.')
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
        A('Note on the reference file\'s two |Weight| values, which the first')
        A('pass reported as "two values (MC@NLO)": 249 999 of its 250 000')
        A('events carry 14.727665 -- the banner XWGTUP exactly -- and ONE')
        A('event (index 185 691) carries 15.625049.  It is a single-event')
        A('anomaly worth 6.2e-05 pb, not a second MC@NLO branch.  The onshell')
        A('and PA files each carry exactly ONE |Weight| value, equal to their')
        A('own banner XWGTUP.')
        A('')
        A('--- THE THREE SPINMODES COMPARED ---')
        A('Same lepton selection, same bin edges, same absolute normalisation.')
        A('Errors here are the PLAIN quadrature sum, unlike every ratio pane in')
        A('this file: these are three independent MadSpin runs, independently')
        A('showered, so there is no covariance to keep.')
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
        A('wrong.  That is exactly what happens for PA below.')
        A('')
        A('where the rate differences come from -- sigma_tot x the fraction of')
        A('events in the z -> e+e- channel, against the fiducial cross')
        A('section, i.e. how much of the difference is channel bookkeeping and')
        A('how much is acceptance:')
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
            if sparse:
                A('   Fewer than %d selected events in this observable, for'
                  % PA.MIN_SEL_TO_DRAW)
                A('   every mode.  The per-bin errors are 20-50 %, the two')
                A('   chi2 are BELOW their number of degrees of freedom, and')
                A('   the rate ratios are 1 to within their errors.  This')
                A('   observable cannot compare the spinmodes at this')
                A('   statistics; the extra curves are NOT drawn on its figure')
                A('   and the numbers above are the whole of what can honestly')
                A('   be said about it.')
            A('')
        A('')
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
        A('')
    A('The sum over ALL events is 1 to within 0.03 % and 0.2 sigma.  That is')
    A('not a null result: the interference between different polarisations of')
    A('the same z is odd in the decay angles and integrates to zero over the')
    A('full angular phase space, so an unrestricted sum MUST come back to 1 and')
    A('the fact that it does is a check on the weights themselves.  The')
    A('interference is visible only differentially, or after a cut that')
    A('restricts the decay angles -- which every lepton selection does.')
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    ap.add_argument('--numbers', default=os.path.join(_HERE, 'numbers.txt'))
    ap.add_argument('--check-minus', action='store_true')
    a = ap.parse_args()
    d = PA.Data(a.data)
    extras = PA.load_extras(a.data)
    bases = [draw(d, obs, a.out, extras) for obs in PA.OBS]
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
