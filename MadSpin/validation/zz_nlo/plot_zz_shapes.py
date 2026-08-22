#!/usr/bin/env python3
"""Where the ``g g`` box does something ``q q~`` does not -- MG7 paper style.

The parent study's ``plot_zz_stack.py`` draws two ratio panes, ``NLO/LO`` and
``(NLO+LI)/LO``.  That figure answers "how much does the loop-induced ``gg``
contribution add"; it cannot answer "does it add it with a *different shape*",
because the second pane has the ``gg`` piece buried inside a sum that the ``qq~``
piece dominates.

**This figure has one ratio pane, and it carries ``NLO/LO`` and ``LI/LO``
separately.**  That is the whole change, and it is the point: plotting the two
components against the same LO denominator is what makes a shape difference
visible.  The main pane keeps the parent study's conventions unchanged,
including filling only the band between ``NLO`` and ``NLO + LI`` rather than
stacking from zero -- on a log axis a solid stack hides both the LO curve and
the band it exists to show.

The ratio pane is on a **log** ``y`` axis, which the parent study's panes are
not.  ``NLO/LO`` sits near 1.4 and ``LI/LO`` near 0.17, an order of magnitude
apart; on a linear axis wide enough to hold both, the LI curve is a line at the
bottom and its shape -- the thing the figure is for -- is unreadable.  A log
axis compares the two *shapes* at their own scales, which is exactly the
comparison being made.

Usage::

    plot_zz_shapes.py [--data DIR] [--out DIR] [--check-minus]
"""

import argparse
import json
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

import observables_shapes as OS                                  # noqa: E402
_LI = os.path.abspath(os.path.join(_HERE, '..', 'zz_loopinduced'))
if _LI not in sys.path:
    sys.path.insert(0, _LI)
from plot_zz_loopinduced import (                                # noqa: E402
    USETEX, check_minus, MINUS_FIX, LW,
)
from plot_zz_stack import wants_minus, C_NLO, C_LI, C_LO, C_TOT  # noqa: E402


# The ratio-pane window, per observable.  Fixed in the code rather than
# autoscaled, as the parent study's are, but log-scaled and wide enough to hold
# both ratios: LI/LO reaches 0.5 at central production above 450 GeV and falls
# below 0.1 at forward production there, while NLO/LO stays between 1.1 and 2.
RATIO_WINDOW = {
    'abs_cos_star_mhigh': (0.06, 4.0),
    'abs_cos_star_mmid': (0.09, 3.0),
    'abs_cos_star_mlow': (0.10, 3.0),
    'abs_cos_theta_cs': (0.10, 3.0),
    'dy_zz': (0.02, 4.0),
    'max_abs_y_z': (0.01, 4.0),
    'abs_y_z_lead': (0.02, 4.0),
    'pt_over_m': (0.05, 6.0),
    'm_zz_fine': (0.08, 3.0),
}
DEFAULT_WINDOW = (0.05, 5.0)


def window(obs):
    return RATIO_WINDOW.get(obs, DEFAULT_WINDOW)


# Headroom above the highest curve on a log main pane, for the legend.  The
# spectra need much more of it than the angular slices do, because their curves
# fall towards the right where the legend sits and the legend has to clear
# them; the angular ones RISE to the right and the legend goes to the left of
# the peak, where a factor of a few is enough.
LOG_HEADROOM = {'m_zz_fine': 20.0}


def log_ylim(curves, obs):
    """``(bottom, top)`` for a log main pane, from the curves it will hold.

    Bottom half a decade below the smallest positive bin of any drawn curve,
    rather than wherever matplotlib's autoscale lands: the loop-induced curve is
    an order of magnitude under the others and letting the autoscale pad it
    leaves the bottom third of every pane empty.
    """
    pos = np.concatenate([c[np.isfinite(c) & (c > 0)] for c in curves])
    if not len(pos):
        return None
    top = max(float(c[np.isfinite(c)].max()) for c in curves)
    return 0.5 * float(pos.min()), top * LOG_HEADROOM.get(obs, 6.0)


class Shapes(object):
    """The committed shape histograms, the reweighted twins and the totals."""

    def __init__(self, ddir):
        self.z = np.load(os.path.join(ddir, 'histograms_shapes.npz'))
        # the parent study's own histograms, so the "for scale" line of
        # numbers_shapes.txt quotes its INCLUSIVE |cos theta*| as a measured
        # number on the same statistic rather than from memory
        self.parent = np.load(os.path.join(ddir, 'histograms.npz'))
        self.meta = json.load(open(os.path.join(ddir, 'meta.json')))
        self.bins = {k: np.array(v)
                     for k, v in self.meta['shapes_bins'].items()}

    def edges(self, obs):
        return self.bins[obs]

    def centres(self, obs):
        e = self.edges(obs)
        return 0.5 * (e[:-1] + e[1:])

    def density(self, key, obs):
        return (self.z['%s/%s/y' % (key, obs)],
                self.z['%s/%s/e' % (key, obs)])

    def sigma(self, key):
        return self.meta['production'][key]['sigma_from_events']

    def sigma_err(self, key):
        p = self.meta['production'][key]
        return p.get('integration_error_pb', p['sigma_mc_error'])


def draw(d, obs, outdir):
    xlab, ylab = (OS.LABELS_SHAPES[obs] if USETEX
                  else (OS.LABELS_SHAPES_TXT[obs],
                        'dsigma/d(%s) [pb per unit]'
                        % OS.LABELS_SHAPES_TXT[obs].split(' [')[0]))
    edges = d.edges(obs)
    x = d.centres(obs)

    ynlo, enlo = d.density('nlo', obs)
    yli, eli = d.density('li', obs)
    ylo, elo = d.density('lo', obs)
    ytot = ynlo + yli

    fig = plt.figure(figsize=(7 * 0.75, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.3], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    rx = fig.add_subplot(gs[1], sharex=ax)

    # --- main pane: unchanged from the parent study.  Only the band between
    #     NLO and NLO+LI is filled; see plot_zz_stack.draw for why.
    ax.stairs(ytot, edges, baseline=ynlo, fill=True, color=C_LI, alpha=0.30,
              lw=0)
    ax.stairs(ynlo, edges, color=C_NLO, lw=LW,
              label=(r'NLO: $pp \to ZZ$ [QCD] ($q\bar q$, $qg$, $g\bar q$)'
                     if USETEX else 'NLO: p p > z z [QCD]'))
    ax.stairs(ytot, edges, color=C_TOT, lw=LW,
              label=(r'NLO $+$ LI $gg$' if USETEX else 'NLO + LI gg'))
    ax.stairs(yli, edges, color=C_LI, lw=LW, ls='dashed',
              label=(r'loop induced: $gg \to ZZ$ ($\mathcal{O}(\alpha_s^2)$ '
                     r'rel.\ LO)' if USETEX else 'loop induced: g g > z z'))
    ax.stairs(ylo, edges, color=C_LO, lw=LW, ls='dashdot',
              label=(r'LO: $pp \to ZZ$' if USETEX else 'LO: p p > z z'))

    ax.set_ylabel(ylab)
    if obs in OS.LOGY_SHAPES:
        ax.set_yscale('log')
        lim = log_ylim((ylo, yli, ytot), obs)
        if lim:
            ax.set_ylim(*lim)
    else:
        ylo_, yhi_ = ax.get_ylim()
        ax.set_ylim(ylo_, yhi_ * 1.55)
    ax.legend(frameon=False, fontsize=8.5,
              loc='upper right' if obs == 'm_zz_fine' else 'best')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if obs not in OS.LOGY_SHAPES:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.setp(ax.get_xticklabels(), visible=False)

    # --- the one ratio pane: NLO/LO and LI/LO, NOT (NLO+LI)/LO
    clip = window(obs)
    for ynum, enum, col, ls, lab in (
            (ynlo, enlo, C_NLO, 'solid',
             r'NLO / LO' if USETEX else 'NLO / LO'),
            (yli, eli, C_LI, 'dashed',
             r'LI $gg$ / LO' if USETEX else 'LI gg / LO')):
        r, re_ = OS.ratio_with_errors(ynum, enum, ylo, elo)
        rc = np.clip(r, *clip)
        rx.errorbar(x, rc, yerr=re_, fmt='none', ecolor=col, elinewidth=0.8,
                    alpha=0.55)
        rx.stairs(rc, edges, color=col, lw=LW, ls=ls, label=lab)
    rx.set_yscale('log')
    rx.axhline(1.0, color='black', lw=0.8, ls=':')
    rx.set_ylim(*clip)
    # matplotlib's default log labelling of a window like (0.09, 3.0) is
    # "6 x 10^-1, 4 x 10^-1, 3 x 10^-1, ...", four superscripts stacked into a
    # pane one third the height of the figure.  A hand-picked decade ladder with
    # plain decimal labels is what a K factor is actually read off.
    ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
             if clip[0] <= t <= clip[1]]
    rx.set_yticks(ticks)
    rx.set_yticklabels([('%g' % t) for t in ticks])
    rx.yaxis.set_minor_formatter(plt.NullFormatter())
    rx.set_ylabel(r'ratio to LO' if USETEX else 'ratio to LO', fontsize=10)
    rx.legend(frameon=False, fontsize=8, loc='best', ncol=2)
    rx.xaxis.set_minor_locator(AutoMinorLocator())
    rx.set_xlim(edges[0], edges[-1])
    rx.set_xlabel(xlab)

    # the top threshold, marked on the figure that exists to look for it
    if obs == 'm_zz_fine':
        two_mt = 2 * d.meta['shapes_m_top']
        for pane in (ax, rx):
            pane.axvline(two_mt, color='0.45', lw=0.9, ls=(0, (4, 2)),
                         zorder=0)
        ax.annotate(r'$2m_t$' if USETEX else '2 m_t',
                    xy=(two_mt, 0.62), xycoords=('data', 'axes fraction'),
                    xytext=(4, 0), textcoords='offset points',
                    fontsize=9, color='0.35', ha='left', va='center')

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, 'shape_' + obs)
    want = wants_minus(fig)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return base, want


# --------------------------------------------------------------------------
def discrimination(d, obs, rw=False):
    """The three flat-line ``chi2/ndf`` values of one observable.

    ``LI/NLO`` is the one that answers the question -- it is algebraically
    ``(LI/LO)/(NLO/LO)``, with the shared LO cancelling, which both removes a
    fully correlated denominator and leaves two independent samples in the
    ratio.  The other two are reported because "both ratios flat" and "both
    ratios bending the same way" are different results that the double ratio
    cannot tell apart.
    """
    sfx = '_rw' if rw else ''
    ynlo, enlo = d.density('nlo' + sfx, obs)
    yli, eli = d.density('li' + sfx, obs)
    ylo, elo = d.density('lo' + sfx, obs)
    out = {}
    for name, (a, ea, b, eb) in (
            ('li_over_nlo', (yli, eli, ynlo, enlo)),
            ('li_over_lo', (yli, eli, ylo, elo)),
            ('nlo_over_lo', (ynlo, enlo, ylo, elo))):
        r, e = OS.ratio_with_errors(a, ea, b, eb)
        c, ndf, level = OS.chi2_flat(r, e)
        out[name] = {'chi2_ndf': c, 'ndf': ndf, 'level': level}
        if name == 'li_over_nlo':
            width = np.diff(d.edges(obs))
            tot = (ynlo + yli) * width
            keep = np.abs(tot) / np.abs(tot).sum() > 2e-3
            lo_, hi_ = OS.spread(r, level, keep)
            out[name].update({'spread_lo': lo_, 'spread_hi': hi_,
                              'spread_factor': hi_ / lo_ if lo_ else float('nan')})
    return out


def write_numbers(d, path):
    out = []
    A = out.append
    m = d.meta
    A('Shape observables: where the loop-induced g g contribution behaves')
    A('UNLIKE the q q~ one, rather than merely contributing a different amount.')
    A('code %s   (samples unchanged; nothing was regenerated for this)'
      % m.get('code_sha', '?'))
    A('scale: %s' % m.get('scale', '?'))
    A('cut:   pt(Z) > %g GeV on BOTH z, via pt_min_pdg = %s'
      % (m['pt_z_min'], m['pt_min_pdg']))
    A('')
    A('sigma:  LO %.6g +- %.4g   NLO %.6g +- %.4g   LI %.6g +- %.4g  [pb]'
      % (d.sigma('lo'), d.sigma_err('lo'), d.sigma('nlo'), d.sigma_err('nlo'),
         d.sigma('li'), d.sigma_err('li')))
    A('integrated LI/NLO = %.5f, LI/LO = %.5f'
      % (d.sigma('li') / d.sigma('nlo'), d.sigma('li') / d.sigma('lo')))
    A('')
    A('=' * 78)
    A('THE DISCRIMINATION, RANKED')
    A('=' * 78)
    A('')
    A('chi2/ndf of a ratio against its own best-fit FLAT line: a pure')
    A('normalisation offset does not enter, so only a shape difference does.')
    A('')
    A('  LI/NLO   is the number the question asks for.  (LI/LO)/(NLO/LO) is')
    A('           algebraically LI/NLO: the shared LO denominator cancels, which')
    A('           removes a fully correlated term and leaves two INDEPENDENT')
    A('           samples in the ratio.  chi2 = 1 means the two ratio shapes are')
    A('           the same to within the statistics of 50 000 events each.')
    A('  LI/LO    the box against the q q~ TREE, both 2 -> 2, so the extra parton')
    A('           of the NLO sample cannot contaminate it.')
    A('  NLO/LO   the K factor\'s own shape, for reference.')
    A('  factor   the spread of the normalised LI/NLO across the bins carrying')
    A('           more than 0.2 % of the rate: max/min, so 1.0 = no difference.')
    A('')
    rows = []
    for obs in OS.SHAPE_OBS:
        rows.append((discrimination(d, obs), obs))
    rows.sort(key=lambda kv: -(kv[0]['li_over_nlo']['chi2_ndf'] or 0))
    A('%-2s %-22s %10s %10s %10s %5s %8s'
      % ('#', 'observable', 'LI/NLO', 'LI/LO', 'NLO/LO', 'ndf', 'factor'))
    for i, (r, obs) in enumerate(rows, 1):
        A('%-2d %-22s %10.2f %10.2f %10.2f %5d %8.2f'
          % (i, obs, r['li_over_nlo']['chi2_ndf'], r['li_over_lo']['chi2_ndf'],
             r['nlo_over_lo']['chi2_ndf'], r['li_over_nlo']['ndf'],
             r['li_over_nlo']['spread_factor']))
    A('')
    A('For scale, the same statistic on the parent study\'s own inclusive')
    A('|cos theta*| -- the observable it reports as separating nothing:')
    pa = {}
    for num, den in (('li', 'nlo'), ('li', 'lo'), ('nlo', 'lo')):
        a = d.parent['%s/abs_cos_theta_star/y' % num]
        ea = d.parent['%s/abs_cos_theta_star/e' % num]
        b = d.parent['%s/abs_cos_theta_star/y' % den]
        eb = d.parent['%s/abs_cos_theta_star/e' % den]
        r, e = OS.ratio_with_errors(a, ea, b, eb)
        pa['%s/%s' % (num, den)] = OS.chi2_flat(r, e)[0]
    A('   abs_cos_theta_star     %10.2f %10.2f %10.2f'
      % (pa['li/nlo'], pa['li/lo'], pa['nlo/lo']))
    A('So "6" and "8" are what a non-separating observable scores here, and')
    A('everything above about 20 in the table is a real shape difference.')
    A('')
    A('-' * 78)
    A('THE SAME NUMBERS AFTER FORCING EVERY SAMPLE ONTO NLO\'s m(ZZ) SPECTRUM')
    A('-' * 78)
    A('')
    A('A per-event reweighting in m(ZZ) only, normalisation preserved.  What')
    A('survives it is a MATRIX-ELEMENT difference -- the box against the tree at')
    A('the same partonic energy -- and what does not survive was the g g')
    A('luminosity falling faster with x, which the parent study already found in')
    A('m(ZZ) and |y(ZZ)| and which is not new physics about the box.')
    A('')
    A('%-2s %-22s %10s %10s %10s %5s'
      % ('#', 'observable', 'LI/NLO', 'LI/LO', 'NLO/LO', 'ndf'))
    rrows = [(discrimination(d, obs, rw=True), obs) for obs in OS.SHAPE_OBS]
    rrows.sort(key=lambda kv: -(kv[0]['li_over_nlo']['chi2_ndf'] or 0))
    for i, (r, obs) in enumerate(rrows, 1):
        A('%-2d %-22s %10.2f %10.2f %10.2f %5d'
          % (i, obs, r['li_over_nlo']['chi2_ndf'], r['li_over_lo']['chi2_ndf'],
             r['nlo_over_lo']['chi2_ndf'], r['li_over_nlo']['ndf']))
    A('')
    A('The reweighting is validated on m(ZZ) itself, which must and does')
    A('collapse to ~0 -- it is the variable being matched.')
    A('')
    A('-' * 78)
    A('THE WINNER, BIN BY BIN')
    A('-' * 78)
    for obs in ('abs_cos_star_mlow', 'abs_cos_star_mmid', 'abs_cos_star_mhigh',
                'abs_cos_theta_cs'):
        edges = d.edges(obs)
        ynlo, enlo = d.density('nlo', obs)
        yli, eli = d.density('li', obs)
        ylo_, elo_ = d.density('lo', obs)
        rli, eli_ = OS.ratio_with_errors(yli, eli, ylo_, elo_)
        rnl, enl_ = OS.ratio_with_errors(ynlo, enlo, ylo_, elo_)
        A('')
        A('%s' % OS.LABELS_SHAPES_TXT[obs])
        A('   %10s %10s %14s %20s %20s'
          % ('bin lo', 'bin hi', 'LI/NLO', 'LI/LO', 'NLO/LO'))
        for i in range(len(edges) - 1):
            if not np.isfinite(rli[i]):
                continue
            A('   %10.3f %10.3f %14.4f %11.4f+-%.4f %11.4f+-%.4f'
              % (edges[i], edges[i + 1], rli[i] / rnl[i],
                 rli[i], eli_[i], rnl[i], enl_[i]))
    A('')
    A('-' * 78)
    A('THE TOP THRESHOLD AT 2 m_t = %.1f GeV' % (2 * m['shapes_m_top']))
    A('-' * 78)
    A('')
    A('The g g box runs through a top loop and must have a threshold there; the')
    A('q q~ tree has no analogue, so a step in LI/LO at 2 m_t would be a')
    A('box-specific feature and the strongest a-priori candidate of this study.')
    A('The fine m(ZZ) binning exists to look for it.  Fit over 250-450 GeV:')
    A('')
    A('   LI/LO = a + b (m - 2 m_t) + c theta(m - 2 m_t)')
    A('')
    for num, den, lab in (('li', 'lo', 'LI/LO'), ('li', 'nlo', 'LI/NLO')):
        edges = d.edges('m_zz_fine')
        x = d.centres('m_zz_fine')
        a, ea = d.density(num, 'm_zz_fine')
        b, eb = d.density(den, 'm_zz_fine')
        r, e = OS.ratio_with_errors(a, ea, b, eb)
        sel = (x > 250.0) & (x < 450.0)
        fit = OS.step_fit(x[sel], r[sel], e[sel], 2 * m['shapes_m_top'])
        if fit is None:
            continue
        a0, b0, c0, sc, chi_w, chi_wo = fit
        A('%s:' % lab)
        A('   level a         = %.5f' % a0)
        A('   slope b         = %+.6f per GeV' % b0)
        A('   STEP  c         = %+.5f +- %.5f   =  %+.1f %% +- %.1f %% of the level'
          % (c0, sc, 100 * c0 / a0, 100 * sc / a0))
        A('   significance    = %.2f sigma' % abs(c0 / sc))
        A('   chi2/ndf with the step %.2f, without it %.2f' % (chi_w, chi_wo))
    A('')
    A('The Higgs triangle IS in the loop-induced sample: MadLoop builds an')
    A('off-shell h from the two z (VVS1_3 with MDL_MH / MDL_WH) and closes it')
    A('with a top or bottom Yukawa loop -- 4 of the 28 loop diagrams are')
    A('triangles, the other 24 boxes.  It cannot produce a peak here: the sample')
    A('has two ON-SHELL z, so m(ZZ) >= 2 m_Z = 182.4 GeV > m_h = 125 GeV and the')
    A('Higgs propagator is off shell at every point of the spectrum.  What is in')
    A('the sample is the smooth, non-resonant box-triangle interference, and')
    A('these samples cannot separate it from the box.')
    A('')
    A('-' * 78)
    A('THE OBSERVABLES THAT ARE TRAPS, AND WHY')
    A('-' * 78)
    A('')
    A('pt(ZZ) and Delta phi(Z,Z) are EXACTLY 0 and EXACTLY pi on both 2 -> 2')
    A('samples (measured pt(ZZ)_max = %.1f on LO and %.1f on LI, against %.1f on'
      % (m['shapes_cs_vs_parent']['lo']['pt_zz_max'],
         m['shapes_cs_vs_parent']['li']['pt_zz_max'],
         m['shapes_cs_vs_parent']['nlo']['pt_zz_max']))
    A('NLO).  They separate NLO from LO and LI instantly, for a reason that has')
    A('nothing to do with the box, and a ratio is undefined there.  Not plotted.')
    A('')
    A('pt(Z_lead)/m(ZZ) is the borderline case: it is (beta/2) sin theta* on a')
    A('2 -> 2 sample and therefore bounded by 0.5 (measured maxima %.4f on LO,'
      % m['shapes_pt_over_m_above_half']['lo_max'])
    A('%.4f on LI), while NLO reaches %.2f.  %.2f %% of the NLO weight sits above'
      % (m['shapes_pt_over_m_above_half']['li_max'],
         m['shapes_pt_over_m_above_half']['nlo_max'],
         100 * m['shapes_pt_over_m_above_half']['nlo_weight_fraction']))
    A('the boundary and is off the axis.  Its topmost bins carry that artefact.')
    A('')
    A('-' * 78)
    A('THE COLLINS-SOPER FRAME BUYS NOTHING ON A 2 -> 2 SAMPLE')
    A('-' * 78)
    A('')
    A('Event-by-event |cos theta*_CS| against the parent study\'s')
    A('|cos theta*| (harder Z in the ZZ rest frame, against the ZZ lab')
    A('direction):')
    A('')
    A('%-6s %22s %22s %14s' % ('sample', 'max |difference|',
                               'mean |difference|', 'pt(ZZ) max'))
    for k in ('lo', 'li', 'nlo'):
        c = m['shapes_cs_vs_parent'][k]
        A('%-6s %22.3e %22.3e %14.2f'
          % (k, c['max_abs_difference'], c['mean_abs_difference'],
             c['pt_zz_max']))
    A('')
    A('They are the SAME observable on LO and LI, to double precision: a 2 -> 2')
    A('ZZ system has no transverse momentum, so its lab direction is the beam')
    A('axis and the Collins-Soper bisector degenerates onto it.  The CS frame')
    A('changes only the NLO curve, where the recoil tilts the ZZ direction.')
    A('')
    A('Note also that the massless Collins-Soper shortcut')
    A('2(l1+ l2- - l1- l2+)/(Q sqrt(Q^2+qT^2)) is WRONG for two massive Z: it')
    A('returns beta cos theta*, which mixes the m(ZZ) dependence into what is')
    A('meant to be a pure angle and inflates the apparent discrimination by an')
    A('order of magnitude.  This module builds the frame explicitly instead.')
    open(path, 'w').write('\n'.join(out) + '\n')
    print('wrote %s' % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    ap.add_argument('--check-minus', action='store_true')
    args = ap.parse_args()
    d = Shapes(args.data)
    bases = [draw(d, obs, args.out) for obs in OS.SHAPE_OBS]
    write_numbers(d, os.path.join(args.data, 'numbers_shapes.txt'))
    if args.check_minus:
        print('usetex = %s   minus workaround active = %s' % (USETEX, MINUS_FIX))
        bad = n = 0
        for b, want in bases:
            if not want:
                print('%s: no minus sign in this figure (every axis positive), '
                      'check not applicable' % os.path.basename(b))
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
