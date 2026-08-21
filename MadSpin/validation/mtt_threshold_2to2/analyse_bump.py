#!/usr/bin/env python3
"""What the bump just above ``2 m_t`` in the ``2 -> 2`` figure is, and whether
it compensates the empty sub-threshold region.

``plots/mtt_threshold.pdf`` draws ``(1/sigma) dsigma/dm_tt`` for
``p p > t t~`` around ``2 m_t``.  Every MadSpin spinmode is exactly zero below
threshold (``RESULTS.md`` section 1), and just above it the shape ratio to the
truth rises to about ``1.12`` before falling back to unity.  This module answers
two questions about that, from the committed histograms only -- it generates
nothing:

1. **What is the bump?**  Not a MadSpin excess: it is the *truth* being depleted
   there.  The truth's ``m_tt`` is the on-shell production spectrum convolved
   with the two top virtualities, and that convolution carries rate out of the
   steeply-rising, concave turn-on and deposits it below threshold and in the
   first GeV above it.  MadSpin's ``m_tt`` is the unsmeared production spectrum
   (it equals ``sqrt(shat)`` at this multiplicity, so the reshuffle holds it
   fixed).  The test is constructive: :func:`smeared_shift` and
   :func:`smeared_beta` smear the production histogram with the sampled
   Breit-Wigner and are compared with the truth.

2. **Does the bump compensate the missing sub-threshold rate?**  That has one
   answer in the figure's normalisation and a different one in pb, and both are
   printed.  The figure is self-normalised, so the excess above threshold is
   forced to equal the truth's sub-threshold *fraction* over the full ``m_tt``
   range; the measured content is that it is all accumulated within
   ``2 m_t + 24 GeV`` and flat from there to the top of the histogram.  In
   absolute pb it does not compensate anything: the truth's total is 4.24 %
   below MadSpin's because of the Breit-Wigner truncation (``RESULTS.md``
   section 3), a global rate loss that has nothing to do with the threshold.

Outputs, next to the existing ones and never on top of them:

* ``plots/mtt_bump.pdf`` / ``.png``            MG7-paper style
* ``plots_userstyle/mtt_bump.pdf`` / ``.png``  user style
* ``plots*/bump_numbers.txt``                  the full numeric report

Usage::

    python3 analyse_bump.py [--data DIR] [--out DIR] [--out-user DIR]
"""

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# This study's figure module.  Importing it sets the MG7-paper rcParams
# (serif / usetex if latex is available), which is what the first rendering
# below wants; the user-style rendering resets them, exactly as
# ``plot_mtt_threshold_userstyle.py`` does.
from plot_mtt_threshold import (                     # noqa: E402
    P, Data, MODES, REF, CURVES_PLAIN, RATIO_CLIP, offscale_arrows, PROC_TEX,
    preserves_mtt, check_minus, USETEX,
)
import matplotlib                                    # noqa: E402
matplotlib.use('Agg')
import matplotlib as mpl                             # noqa: E402
import matplotlib.pyplot as plt                      # noqa: E402


# --------------------------------------------------------------------------
# Model parameters.  All read off the param card recorded in data/meta.json;
# nothing here is tuned to the truth.
# --------------------------------------------------------------------------
MODEL_LABEL = 'production, BW-smeared'

# The binning.  The parent figure's zones, extended by one 10 GeV zone so the
# cumulative excess can be seen to SATURATE rather than merely to pass through
# the sub-threshold rate.  420 GeV is where the parent figure stops asking its
# agreement question; 520 GeV is the top of the committed histogram.
BUMP_ZONES = list(P.ZONES) + [(420.0, 520.0, 10.0)]

# Where the near-threshold excess is integrated.  Chosen from the data, not in
# advance: the cumulative curve is flat above it (see the disjoint-window table
# in the report), and it is the first plot-bin edge at which the running total
# has reached the sub-threshold rate.
EXCESS_HI = 370.0

# The model is a leading-order description of the smearing and its normalisation
# against the truth is only good to about a percent, so it is drawn and quoted
# only where the parent figure asks its questions.
MODEL_HI = 420.0


def bump_edges():
    out = [BUMP_ZONES[0][0]]
    for lo, hi, w in BUMP_ZONES:
        for k in range(1, int(round((hi - lo) / w)) + 1):
            out.append(lo + k * w)
    return np.array(out)


# --------------------------------------------------------------------------
# The Breit-Wigner the truth actually used.
# --------------------------------------------------------------------------
def _lam(a, b, c):
    return a * a + b * b + c * c - 2.0 * (a * b + b * c + c * a)


def partial_width(m, mw, mb):
    """LO ``Gamma(t -> W b)`` at top virtuality ``m``, up to a constant.

    Only the ``m`` dependence matters here: it enters the off-shell density as
    the numerator ``m Gamma(m) / (m_t Gamma_t)``, which is the factor that turns
    a bare propagator into the properly normalised off-shell lineshape.  Over
    the truncation window it runs from 0.60 to 1.51, so it is not a detail.
    """
    lam = np.clip(_lam(m * m, mw * mw, mb * mb), 0.0, None)
    p = np.sqrt(lam) / (2.0 * m)
    num = ((m * m - mb * mb) ** 2 + mw * mw * (m * m + mb * mb) - 2.0 * mw ** 4)
    return p * num / (mw * mw) / (m * m)


def lineshape(m, mt, gt, mw, mb, numerator=True):
    """Unnormalised ``d(rate)/dm`` for one off-shell top.

    Fixed-width relativistic Breit-Wigner, which is what ``myamp.f`` uses, times
    the decay numerator when ``numerator`` is set.
    """
    p = 2.0 * m / ((m * m - mt * mt) ** 2 + (mt * gt) ** 2)
    if numerator:
        p = p * (m * partial_width(m, mw, mb)) / (mt * partial_width(mt, mw, mb))
    return p


def pair_kernel(mt, gt, mw, mb, cutoff, dx, numerator=True):
    """Distribution of ``delta = m1 + m2 - 2 m_t`` on the fine grid."""
    m = np.linspace(mt - cutoff * gt, mt + cutoff * gt, 40001)
    p = lineshape(m, mt, gt, mw, mb, numerator)
    p /= p.sum()
    edges = np.arange(-cutoff * gt - dx / 2.0, cutoff * gt + dx, dx)
    h, _ = np.histogram(m - mt, bins=edges, weights=p)
    h /= h.sum()
    k = np.convolve(h, h)                       # two independent tops
    if len(k) % 2 == 0:                         # np.convolve(..., 'same') wants
        k = np.append(k, 0.0)                   # an odd kernel to stay centred
    return k / k.sum()


# --------------------------------------------------------------------------
# Two smearing models.  They approximate the same physics differently, so
# agreement between them is the check that neither approximation is carrying
# the result.
# --------------------------------------------------------------------------
def smeared_shift(prod, kernel):
    """Model A: shift the threshold.

    ``dsigma/dM (M; m1, m2) = dsigma_prod/dM (M - [m1 + m2 - 2 m_t])``.  Exact in
    the limit where the only thing the virtualities do is move where the
    two-body phase space opens, which near threshold is what dominates: with
    ``m1 = m2 = m`` the velocity factor obeys ``beta(M) = beta_0(M - delta)`` up
    to ``O(delta / m_t)``.  Needs nothing but the production histogram.
    """
    return np.convolve(prod, kernel, mode='same')


def smeared_beta(prod, centres, mt, weights, masses, two_mt,
                 fit_lo=346.5, fit_hi=460.0, deg=4):
    """Model B: rescale the velocity factor, with an extrapolated luminosity.

    Writes ``dsigma_prod/dM = Phi(M) beta_0(M)`` with
    ``beta_0 = sqrt(1 - 4 m_t^2 / M^2)``, fits the smooth ``Phi`` (parton
    luminosity times matrix element, which has no threshold structure) over
    ``fit_lo..fit_hi`` and extrapolates it below ``2 m_t``, then rebuilds
    ``Phi(M) <beta(M; m1, m2)>`` with the exact off-shell velocity factor.  The
    velocity factor is exact where model A approximates it; the luminosity is
    extrapolated where model A does not need to be.
    """
    b0 = np.where(centres > two_mt,
                  np.sqrt(np.clip(1.0 - 4.0 * mt * mt / centres ** 2, 0.0, None)),
                  np.nan)
    ok = (centres > fit_lo) & (centres < fit_hi)
    coef = np.polyfit(centres[ok], prod[ok] / b0[ok], deg)
    phi = np.polyval(coef, centres)
    m2 = centres ** 2
    acc = np.zeros_like(centres)
    for wi, m1 in zip(weights, masses):
        lam = _lam(m2[:, None], m1 * m1, masses[None, :] ** 2)
        beta = np.where(lam > 0.0, np.sqrt(np.clip(lam, 0.0, None)) / m2[:, None],
                        0.0)
        acc += wi * (beta * weights[None, :]).sum(axis=1)
    return phi * acc


# --------------------------------------------------------------------------
# The measurement.
# --------------------------------------------------------------------------
class Bump(object):
    """Everything the report and the figure need, computed once."""

    def __init__(self, d):
        self.d = d
        z, self.fine = d.z, d.fine
        self.dx = float(self.fine[1] - self.fine[0])
        self.c = 0.5 * (self.fine[:-1] + self.fine[1:])
        self.two_mt = d.two_mt
        self.mt = float(d.meta['param_card_masses']['MT'])
        self.gt = float(d.meta['param_card_masses']['WT'])
        self.mw = 80.41851
        self.mb = float(d.meta['param_card_masses']['MB'])
        self.cutoff = float(d.meta.get('bwcutoff', 15.0))

        self.above = self.c >= self.two_mt
        self.below = ~self.above
        self.xs = self.fine[1:][self.above]

        # The sub-threshold rate the modes miss, as a rate and as a fraction.
        n, s = d.nevents(REF), d.sigma(REF)
        self.sub_pb = z['%s_sumw' % REF][self.below].sum() / n
        self.sub_pb_err = math.sqrt(z['%s_sumw2' % REF][self.below].sum()) / n
        self.f_below = self.sub_pb / s
        self.f_below_err = self.sub_pb_err / s

        self.prod = z['production_sumw'] / d.nevents('production') / self.dx
        self.truth = z['%s_sumw' % REF] / n / self.dx

        # The two models, and the mass grid model B integrates over.
        kern = pair_kernel(self.mt, self.gt, self.mw, self.mb, self.cutoff,
                           self.dx)
        self.model_A = smeared_shift(self.prod, kern)
        masses = np.linspace(self.mt - self.cutoff * self.gt,
                             self.mt + self.cutoff * self.gt, 160)
        w = lineshape(masses, self.mt, self.gt, self.mw, self.mb)
        w = w / w.sum()
        self.model_B = smeared_beta(self.prod, self.c, self.mt, w, masses,
                                    self.two_mt)
        # Model B is built from a fit, so it carries no absolute normalisation
        # of its own.  Anchor it on the production sample over the window the
        # parent figure calls flat; model A inherits the production's rate by
        # construction and is left alone.
        anchor = (self.c >= 380.0) & (self.c < 420.0)
        self.model_B *= self.prod[anchor].sum() / self.model_B[anchor].sum()
        self.model_sigma = d.sigma('production')

    # -- shape (the figure's normalisation) --------------------------------
    def shape(self, key):
        d = self.d
        return (self.d.z['%s_sumw' % key] / d.nevents(key) / self.dx
                / d.sigma(key))

    def model_shape(self, which='B'):
        m = self.model_B if which == 'B' else self.model_A
        return m / self.model_sigma

    # -- cumulative excess above threshold ---------------------------------
    def cumulative(self, key):
        """(x, excess, error) of ``F_key(x) - F_truth(x)`` from ``2 m_t`` up.

        Both sides are self-normalised, so this is the figure's normalisation
        and the excess is in units of each sample's own total cross section.
        The central value uses the event WEIGHTS, i.e. exactly what the figure
        draws -- which is the only way the overweight events of section 4 can
        show up at all, since the raw counts are identical bin for bin across
        all four modes (``RESULTS.md`` section 0).

        The error is binomial: the LHE carries ``event_norm = average``, so
        every event has the same weight bar the overweight excess (at most 70
        event-equivalents in 1 000 000), and a CDF of an equal-weight sample has
        variance ``p (1 - p) / N``.  Using the counts for the error and the
        weights for the value is the right pairing here for the same reason.
        Treating the bins as independent Poisson counts instead would ignore the
        anti-correlation that makes the far end of the curve much better
        determined than a bin-by-bin sum suggests.
        """
        z, d = self.d.z, self.d
        fm = np.cumsum(z['%s_sumw' % key][self.above]) / d.nevents(key) / d.sigma(key)
        ft = np.cumsum(z['%s_sumw' % REF][self.above]) / d.nevents(REF) / d.sigma(REF)
        pm = np.cumsum(z['%s_cnt' % key][self.above]) / d.nevents(key)
        pt = np.cumsum(z['%s_cnt' % REF][self.above]) / d.nevents(REF)
        err = np.sqrt(pm * (1.0 - pm) / d.nevents(key)
                      + pt * (1.0 - pt) / d.nevents(REF))
        return self.xs, fm - ft, err

    def cumulative_model(self, which='B'):
        m = self.model_B if which == 'B' else self.model_A
        cm = np.cumsum(m[self.above]) * self.dx / self.model_sigma
        ct = np.cumsum(self.truth[self.above]) * self.dx / self.d.sigma(REF)
        return self.xs, cm - ct

    def window(self, key, lo, hi):
        """(share of the sample's own sigma in [lo, hi), binomial error, pb).

        Same pairing as :meth:`cumulative`: the value from the weights, the
        error from the counts.
        """
        m = (self.c >= lo) & (self.c < hi)
        d = self.d
        n = d.nevents(key)
        pb = d.z['%s_sumw' % key][m].sum() / n
        k = d.z['%s_cnt' % key][m].sum() / n
        return pb / d.sigma(key), math.sqrt(k * (1.0 - k) / n), pb

    def window_excess(self, key, lo, hi):
        """(shape excess over the truth, its error, absolute excess in pb)."""
        fm, em, pbm = self.window(key, lo, hi)
        ft, et, pbt = self.window(REF, lo, hi)
        return fm - ft, math.hypot(em, et), pbm - pbt

    # -- rebinning, for the drawn ratio ------------------------------------
    def rebin(self, vec_density, edges):
        """Mean density of ``vec_density`` (per GeV) in each of ``edges``."""
        out = np.empty(len(edges) - 1)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = (self.c >= lo) & (self.c < hi)
            out[i] = vec_density[m].mean()
        return out

    def rebin_counts(self, key, edges):
        out = np.empty(len(edges) - 1, dtype=float)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = (self.c >= lo) & (self.c < hi)
            out[i] = self.d.z['%s_cnt' % key][m].sum()
        return out

    def shape_binned(self, key, edges):
        """((1/sigma) dsigma/dm on ``edges``, its error, the raw counts)."""
        d = self.d
        n, s = d.nevents(key), d.sigma(key)
        w = np.diff(edges)
        sw = np.array([d.z['%s_sumw' % key][(self.c >= lo) & (self.c < hi)].sum()
                       for lo, hi in zip(edges[:-1], edges[1:])])
        sw2 = np.array([d.z['%s_sumw2' % key][(self.c >= lo) & (self.c < hi)].sum()
                        for lo, hi in zip(edges[:-1], edges[1:])])
        return sw / n / w / s, np.sqrt(sw2) / n / w / s, self.rebin_counts(key, edges)


def ratio(num, nume, den, dene):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(den > 0, num / den, np.nan)
        e = np.abs(r) * np.sqrt(np.where(num > 0, (nume / num) ** 2, 0.0)
                                + np.where(den > 0, (dene / den) ** 2, 0.0))
    return r, e


# --------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------
WINDOWS = [(346.0, 347.0), (347.0, 352.0), (352.0, 356.0), (356.0, 362.0),
           (362.0, 370.0), (370.0, 380.0), (380.0, 420.0), (420.0, 470.0),
           (470.0, 520.0)]

SCAN = [347, 348, 350, 352, 354, 356, 358, 360, 362, 365, 370, 375, 380,
        400, 420, 450, 480, 520]


def write_numbers(b, fh=sys.stdout):
    d = b.d
    p = lambda *a: print(*a, file=fh)
    edges = bump_edges()
    two_mt = b.two_mt

    p('=' * 78)
    p('The bump just above 2 m_t in the 2 -> 2 m_tt figure')
    p('=' * 78)
    p('code            : %s (%s)' % (d.meta.get('code_sha', '?')[:12],
                                     d.meta.get('code_branch', '?')))
    p('reference figure: plots/mtt_threshold.pdf, plots_userstyle/mtt_threshold.pdf')
    p('2 m_t           : %.4f GeV' % two_mt)
    p('bwcutoff        : %g widths   (m_t = %g, Gamma_t = %g)'
      % (b.cutoff, b.mt, b.gt))
    p('')
    p('THE QUESTION.  The figure draws (1/sigma) dsigma/dm_tt.  Every spinmode')
    p('is EXACTLY zero below 2 m_t and the shape ratio to the truth rises to')
    p('about 1.12 just above it.  Is that bump a normalisation artefact of the')
    p('self-normalisation, and does it compensate the empty region below?')
    p('')

    # ---------------------------------------------------------------- 1
    p('-- 1. the two normalisations, and why the answer differs between them --')
    p('')
    p('   sub-threshold rate the modes miss (truth, m_tt < 2 m_t):')
    p('      %.4f +- %.4f pb   = %.5f%% +- %.5f%% of the truth total (%.4f pb)'
      % (b.sub_pb, b.sub_pb_err, 100 * b.f_below, 100 * b.f_below_err,
         d.sigma(REF)))
    p('')
    p('   SELF-NORMALISED (what the figure draws).  Each curve is divided by')
    p('   its own total sigma, so each integrates to 1 over the full m_tt')
    p('   range.  A mode with zero sub-threshold rate therefore has its')
    p('   above-threshold curve scaled up by 1/(1 - f_below) relative to the')
    p('   truth SHAPE:')
    p('      1/(1 - %.7f) - 1 = %+.5f%%   -- a FLAT rescaling, every bin'
      % (b.f_below, 100 * (1.0 / (1.0 - b.f_below) - 1.0)))
    p('   The visible bump is nothing like that size.  Largest drawn shape')
    p('   ratio above threshold, and the flat artefact for comparison:')
    y_t, ye_t, cnt_t = b.shape_binned(REF, edges)
    worst = None
    for key in MODES:
        y, ye, cnt = b.shape_binned(key, edges)
        r, re = ratio(y, ye, y_t, ye_t)
        m = (edges[:-1] >= two_mt) & np.isfinite(r)
        i = int(np.nanargmax(np.where(m, r, -np.inf)))
        p('      %-11s max shape ratio %.3f +- %.3f in %g-%g GeV  =  %+.1f%%, '
          'i.e. %.0f x the flat artefact'
          % (key, r[i], re[i], edges[i], edges[i + 1], 100 * (r[i] - 1),
             (r[i] - 1) / (1.0 / (1.0 - b.f_below) - 1.0)))
        if worst is None:
            worst = (key, r[i], edges[i])
    p('   => the bump is a REDISTRIBUTION of rate, not the self-normalisation.')
    p('')
    p('   ABSOLUTE pb.  The two sides do NOT have the same total: the truth')
    p('   loses 4.24% to the Breit-Wigner truncation (RESULTS.md section 3),')
    p('   which is a global rate loss and has nothing to do with the threshold.')
    for key in MODES:
        p('      %-11s sigma %.4f pb vs truth %.4f pb;  above 2 m_t the mode '
          'exceeds the truth by %+.3f pb = %.1f x the %.4f pb it misses below'
          % (key, d.sigma(key), d.sigma(REF),
             d.sigma(key) - (d.sigma(REF) - b.sub_pb),
             (d.sigma(key) - (d.sigma(REF) - b.sub_pb)) / b.sub_pb, b.sub_pb))
    p('   => in pb the bump compensates NOTHING.  The rate question can only')
    p('      be asked after the totals are matched, which is the third')
    p('      normalisation below.')
    p('')

    # ---------------------------------------------------------------- 2
    p('-- 2. where the excess actually sits ---------------------------------')
    p('')
    p('   plots/numbers.txt records shape ratios of 0.842 / 0.900 / 0.812 /')
    p('   0.812 in the FIRST bin above threshold, 346-347 GeV: the modes are')
    p('   BELOW the truth there.  So the excess is not "MadSpin piles up where')
    p('   the truth spills below".  Disjoint windows, so these add and their')
    p('   errors are independent (fraction of each sample; binomial errors):')
    p('')
    p('   window [GeV]      mode        truth       excess              sigma')
    for key in ['onshell']:
        for lo, hi in WINDOWS:
            pm = b.window(key, lo, hi)[0]
            pt = b.window(REF, lo, hi)[0]
            dd, ee, _ = b.window_excess(key, lo, hi)
            p('   [%3g,%3g)     %8.5f%%   %8.5f%%   %+8.5f%% +- %.5f%%   %+5.1f'
              % (lo, hi, 100 * pm, 100 * pt, 100 * dd, 100 * ee,
                 dd / ee if ee else 0.0))
    p('   (onshell, which took no overweights; the four modes agree to '
      '0.007% of sigma -- see section 4.)')
    p('')
    p('   The first bin is a 5 sigma DEFICIT and everything from 347 GeV up is')
    p('   an excess.  The running total crosses zero, on the raw 0.25 GeV')
    p('   grid, at:')
    for key in MODES:
        x, exc, _ = b.cumulative(key)
        i = np.flatnonzero((exc[:-1] < 0) & (exc[1:] >= 0))
        p('      %-11s %s' % (key, ('%.2f GeV' % x[i[-1]]) if len(i)
                              else 'never'))
    p('   The three distinct answers are the overweights of section 4, which')
    p('   all sit inside 1 GeV of threshold; onshell and madspin_v1 (none) are')
    p('   the clean ones.')
    p('')

    # ---------------------------------------------------------------- 3
    p('-- 3. does the excess integrate to the sub-threshold rate? -----------')
    p('')
    p('   Cumulative shape excess F_mode(x) - F_truth(x) from 2 m_t upwards,')
    p('   in units of each sample\'s own total sigma.  Errors are binomial on')
    p('   the CDF, not a bin-by-bin Poisson sum.')
    p('')
    x, exc, err = b.cumulative('onshell')
    p('   x [GeV]      onshell             vs f_below     madspin     PA      '
      ' madspin_v1')
    others = {k: b.cumulative(k)[1] for k in MODES}
    for X in SCAN:
        i = int(np.argmin(np.abs(x - X)))
        p('   %3d       %+.5f%% +- %.5f%%   %+5.1f sigma   %+.5f%% %+.5f%% '
          '%+.5f%%'
          % (X, 100 * exc[i], 100 * err[i],
             (exc[i] - b.f_below) / err[i] if err[i] else 0.0,
             100 * others['madspin'][i], 100 * others['PA'][i],
             100 * others['madspin_v1'][i]))
    p('   (f_below = %.5f%% +- %.5f%%.  madspin_v1 is onshell to the last '
      'digit: same sample, same weights.)' % (100 * b.f_below,
                                              100 * b.f_below_err))
    p('')
    p('   THE NUMBER.  Integrated over 2 m_t .. %g GeV, against the truth\'s'
      % EXCESS_HI)
    p('   sub-threshold rate, in all three normalisations:')
    p('')
    for key in MODES:
        dd, ee, pb = b.window_excess(key, two_mt, EXCESS_HI)
        p('      %-11s SELF-NORMALISED  %+.5f%% +- %.5f%%  vs f_below '
          '%.5f%% +- %.5f%%   ratio %.3f +- %.3f'
          % (key, 100 * dd, 100 * ee, 100 * b.f_below, 100 * b.f_below_err,
             dd / b.f_below, ee / b.f_below))
        p('      %-11s RATE-MATCHED     %+.4f +- %.4f pb  vs %.4f +- %.4f pb '
          '  (mode scaled by sigma_truth/sigma_mode = %.5f)'
          % ('', dd * d.sigma(REF), ee * d.sigma(REF), b.sub_pb, b.sub_pb_err,
             d.sigma(REF) / d.sigma(key)))
        p('      %-11s ABSOLUTE         %+.4f pb  vs %.4f pb   -- does not '
          'balance and cannot: the totals differ by 1 - truth/mode = %.2f%% '
          '(= %+.3f pb over the whole spectrum, of which %+.3f pb sits above '
          '2 m_t)'
          % ('', pb, b.sub_pb, 100 * (1.0 - d.sigma(REF) / d.sigma(key)),
             d.sigma(key) - d.sigma(REF),
             d.sigma(key) - (d.sigma(REF) - b.sub_pb)))
        p('')
    p('   THE SYSTEMATIC on that number, and it is not small.  Dividing each')
    p('   side by its own TOTAL sigma is the figure\'s choice, and it assumes')
    p('   the truth\'s 4.24% truncation loss is flat in m_tt.  It is not quite:')
    p('   the parent study measures truth/mode = 0.9576 on the totals against')
    p('   0.9517 +- 0.0025 on the 380-420 GeV anchor, a 0.6% difference, and')
    p('   that same 0.6% is visible above as the +0.105% +- 0.041% the')
    p('   cumulative picks up over [380, 420).  Renormalising each mode on the')
    p('   380-420 GeV anchor instead of its total moves the answer to:')
    p('')
    anchor = (380.0, 420.0)
    for key in MODES:
        fa = b.window(key, *anchor)[0] / b.window(REF, *anchor)[0]
        fm = b.window(key, two_mt, EXCESS_HI)[0] / fa
        ft = b.window(REF, two_mt, EXCESS_HI)[0]
        _, ee, _ = b.window_excess(key, two_mt, EXCESS_HI)
        p('      %-11s anchor-normalised  %+.5f%% +- %.5f%%   = %.2f x f_below'
          '   (total-normalised: %.2f x)'
          % (key, 100 * (fm - ft), 100 * ee, (fm - ft) / b.f_below,
             b.window_excess(key, two_mt, EXCESS_HI)[0] / b.f_below))
    p('')
    p('   So the excess is 0.75 to 1.0 times the sub-threshold rate depending')
    p('   on which of the two defensible normalisations is used, against a')
    p('   statistical error of 0.15.  The normalisation choice, not the')
    p('   statistics, is what limits how sharply the compensation can be')
    p('   stated -- and it is a broad-band effect that the parent study')
    p('   already records, not anything to do with the threshold.')
    p('')
    p('   Above %g GeV nothing further accumulates:' % EXCESS_HI)
    for key in ['onshell']:
        dd, ee, _ = b.window_excess(key, EXCESS_HI, 520.0)
        p('      %-11s [%g, 520) GeV  %+.5f%% +- %.5f%%   (%.1f sigma from '
          'zero) -- flat' % (key, EXCESS_HI, 100 * dd, 100 * ee,
                             dd / ee if ee else 0.0))
    p('')
    p('   CAVEAT, stated because it is what makes the self-normalised answer')
    p('   partly an identity.  Over the FULL m_tt range the self-normalised')
    p('   excess above threshold is forced to equal f_below exactly, because')
    p('   both curves integrate to 1 and the mode has nothing below.  The')
    p('   MEASURED content is not that the total comes out right -- it is')
    p('   WHERE it comes out: all of it inside 2 m_t + %g GeV, over a window'
      % (EXCESS_HI - two_mt))
    p('   holding %.2f%% of the cross section, and flat above.'
      % (100 * b.window(REF, two_mt, EXCESS_HI)[0]))
    p('   The histogram itself stops at 520 GeV and holds %.1f%% of sigma, so'
      % (100 * b.window(REF, 290.0, 520.0)[0]))
    p('   the identity is not available to the numbers above as a shortcut.')
    p('')

    # ---------------------------------------------------------------- 4
    p('-- 4. do the overweight events contaminate this? ---------------------')
    p('')
    p('   RESULTS.md section 2: the overweight excess is 21.11 event-')
    p('   equivalents for madspin (joint) and 69.53 for PA (sequential), all')
    p('   of it within ~1 GeV of 2 m_t; onshell and madspin_v1 took none and')
    p('   are the production sample weight for weight.  Measured against')
    p('   onshell, in the same units as section 3:')
    p('')
    dd, ee, _ = b.window_excess('onshell', two_mt, EXCESS_HI)
    worst_rel = 0.0
    _, exo, _ = b.cumulative('onshell')
    for key in ['madspin', 'PA']:
        x, exc, _ = b.cumulative(key)
        i = int(np.argmin(np.abs(x - EXCESS_HI)))
        shift = exc[i] - exo[i]
        worst_rel = max(worst_rel, abs(shift) / ee)
        p('      %-11s cumulative excess at %g GeV sits %+.5f%% of sigma above '
          'onshell\'s' % (key, EXCESS_HI, 100 * shift))
        p('      %-11s   = %.1f%% of the measured excess, %.0f%% of its '
          'statistical error' % ('', 100 * shift / dd, 100 * shift / ee))
    p('      to compare: the excess being measured is %.5f%% +- %.5f%%'
      % (100 * dd, 100 * ee))
    p('   => the overweights move the answer by at most %.0f%% of its own'
      % (100 * worst_rel))
    p('      statistical error, so they do NOT contaminate it.  The clean')
    p('      modes are onshell and madspin_v1 (zero overweights), and every')
    p('      headline number above is quoted from onshell for that reason.')
    p('   NB the overweights ARE the whole of the visible spread between the')
    p('   four modes in the 346-347 GeV bin (shape ratios 0.812 / 0.842 /')
    p('   0.900), which is why that bin is not used to define anything here,')
    p('   and why the integration window starts at 2 m_t and not inside it.')
    p('')

    # ---------------------------------------------------------------- 5
    p('-- 5. the mechanism, tested and not asserted -------------------------')
    p('')
    p('   HYPOTHESIS.  The truth\'s m_tt is the on-shell production spectrum')
    p('   smeared by the two top virtualities: at virtualities m1, m2 the pair')
    p('   threshold is m1 + m2, not 2 m_t.  The production spectrum is zero')
    p('   below 2 m_t and rises like sqrt(M - 2 m_t) above it -- CONCAVE -- so')
    p('   smearing it (i) fills the region below threshold, (ii) RAISES the')
    p('   first GeV above it, where the sharp spectrum is still near zero, and')
    p('   (iii) LOWERS the concave rise that follows.  MadSpin\'s m_tt is the')
    p('   unsmeared spectrum (m_tt = sqrt(shat) at this multiplicity, held')
    p('   fixed by the reshuffle), so mode/truth < 1 in the first bin and > 1')
    p('   over the following ~10 GeV.  That is exactly the measured pattern.')
    p('')
    p('   TEST.  Smear the committed production histogram with the same')
    p('   Breit-Wigner the truth used -- fixed-width relativistic, truncated at')
    p('   %g Gamma_t, times the decay numerator m Gamma(m)/(m_t Gamma_t) -- and'
      % b.cutoff)
    p('   compare with the truth.  Two independent approximations:')
    p('      A  threshold shift : M -> M + (m1 + m2 - 2 m_t)')
    p('      B  velocity factor: dsigma_prod/dM / beta_0(M) fitted and')
    p('         extrapolated below 2 m_t, times the exact beta(M; m1, m2)')
    p('   Neither is tuned to the truth.  Model B carries no absolute')
    p('   normalisation of its own and is anchored on the production sample')
    p('   over 380-420 GeV.')
    p('')
    p('   bin [GeV]     truth shape    A/truth   B/truth   MadSpin/truth')
    yA = b.rebin(b.model_A, edges) / b.model_sigma
    yB = b.rebin(b.model_B, edges) / b.model_sigma
    ymode, _, _ = b.shape_binned('onshell', edges)
    ZONES_DEV = [('deep tail   316-336', 316.0, 336.0),
                 ('below 2m_t  336-346', 336.0, two_mt),
                 ('the bump    346-356', two_mt, 356.0),
                 ('recovery    356-380', 356.0, 380.0),
                 ('flat        380-420', 380.0, MODEL_HI)]
    dev = {z[0]: [0.0, 0.0, 0.0] for z in ZONES_DEV}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if hi > MODEL_HI:
            break
        t = y_t[i]
        if t <= 0:
            continue
        rm = ymode[i] / t if ymode[i] > 0 else float('nan')
        p('   %4g-%4g     %10.3e   %8.3f  %8.3f   %s'
          % (lo, hi, t, yA[i] / t, yB[i] / t,
             ('%8.3f' % rm) if np.isfinite(rm) and rm > 0 else '   0 exact'))
        for name, zlo, zhi in ZONES_DEV:
            if not (zlo <= lo and hi <= zhi):
                continue
            dev[name][0] = max(dev[name][0], abs(yA[i] / t - 1.0))
            dev[name][1] = max(dev[name][1], abs(yB[i] / t - 1.0))
            if lo >= two_mt:
                dev[name][2] = max(dev[name][2], abs(rm - 1.0))
    p('')
    p('   largest |ratio - 1|, by region:')
    p('      region [GeV]          model A   model B   MadSpin')
    for name, zlo, zhi in ZONES_DEV:
        a_, b_, m_ = dev[name]
        p('      %-20s %6.1f%%   %6.1f%%   %s'
          % (name, 100 * a_, 100 * b_,
             ('%6.1f%%' % (100 * m_)) if m_ else '  0 exact (no support)'))
    p('   The bump row is the answer: over 346-356 GeV the smeared production')
    p('   sample tracks the truth to %.0f%% (model B) where MadSpin -- which is'
      % (100 * dev['the bump    346-356'][1]))
    p('   the SAME production sample, unsmeared -- is off by %.0f%%.  The deep'
      % (100 * dev['the bump    346-356'][2]))
    p('   tail is where the model\'s own approximations break down; it holds')
    p('   0.001% of sigma (32 truth events in the deepest drawn bin) and no')
    p('   claim here rests on it.')
    devM = dev['the bump    346-356'][2]
    p('')
    p('   the sub-threshold rate, which is the thing MadSpin misses entirely:')
    subA = b.model_A[b.below].sum() * b.dx / b.model_sigma
    subB = b.model_B[b.below].sum() * b.dx / b.model_sigma
    p('      truth    %.5f%% of sigma' % (100 * b.f_below))
    p('      model A  %.5f%%   = %.0f%% of the truth\'s'
      % (100 * subA, 100 * subA / b.f_below))
    p('      model B  %.5f%%   = %.0f%% of the truth\'s'
      % (100 * subB, 100 * subB / b.f_below))
    p('      MadSpin  0 exactly = 0%')
    p('')
    p('   and the cumulative excess of section 3, recomputed with the model in')
    p('   place of MadSpin -- if the smearing IS the mechanism, the model must')
    p('   not have the excess:')
    xA, cA = b.cumulative_model('A')
    xB, cB = b.cumulative_model('B')
    _, cM, eM = b.cumulative('onshell')
    for X in (352, 356, 362, 370, 380, 420):
        i = int(np.argmin(np.abs(xA - X)))
        p('      x = %3d GeV   MadSpin %+.5f%% +- %.5f%%   model A %+.5f%%   '
          'model B %+.5f%%'
          % (X, 100 * cM[i], 100 * eM[i], 100 * cA[i], 100 * cB[i]))
    p('')
    p('   VERDICT.  The smeared production sample reproduces the truth\'s')
    p('   turn-on to a few percent bin by bin where MadSpin is off by %.0f%%,'
      % (100 * devM))
    p('   and recovers most of the sub-threshold rate where MadSpin has none.')
    p('   The residual of the model is a roughly uniform ~1% normalisation')
    p('   offset against the truth over the whole drawn window, not a failure')
    p('   of the turn-on shape -- so the smearing accounts for the bump and')
    p('   slightly overshoots it.  The model is a leading-order description')
    p('   (the matrix element is taken to depend on the virtualities only')
    p('   through the phase space) and is NOT a substitute for the truth.')
    p('')

    # ---------------------------------------------------------------- 6
    p('-- 6. what this does not settle -------------------------------------')
    p('')
    p('   * The model\'s ~1% normalisation offset is not resolved into its')
    p('     parts.  Candidates: the 4.24% Breit-Wigner truncation is not')
    p('     perfectly flat in m_tt (the parent study measures truth/mode as')
    p('     0.9576 on the total against 0.9517 +- 0.0025 on the 380-420 GeV')
    p('     anchor, a 0.6% difference), and the model\'s neglect of the')
    p('     virtuality dependence of the matrix element.  Separating them')
    p('     needs a truth run at a second bwcutoff, which does not exist.')
    p('   * Only bwcutoff = 15 was run, so how far below threshold the truth')
    p('     reaches -- and hence the size of everything measured here -- is a')
    p('     parameter, not a prediction.')
    p('   * Single- and non-resonant W b W b~ diagrams are absent from BOTH')
    p('     sides (the truth is an MG5 decay chain), so nothing here bounds')
    p('     their contribution near threshold.')
    p('   * One seed per mode.  The overweight numbers of section 4 are one')
    p('     draw; the rest does not depend on them.')


# --------------------------------------------------------------------------
# The figure.
# --------------------------------------------------------------------------
MG7_COLOR = {'madspin': 'blue', 'PA': 'red', 'onshell': 'tab:green',
             'madspin_v1': 'tab:purple'}
USER_COLOR = {'madspin': 'C0', 'PA': 'C1', 'onshell': 'C2', 'madspin_v1': 'C3'}
MODEL_COLOR = '0.45'


def draw(b, out, style):
    """Two panes, split in x.

    The cumulative excess does everything it is going to do inside 25 GeV of
    threshold and then has 150 GeV of flat tail to be checked against, and one
    linear x-axis cannot show both: the turn-on collapses into a quarter of the
    frame.  So the x-axis is broken at 380 GeV, the left column carrying the
    turn-on and the right the tail, with the left given the larger share of the
    width.  Both columns share their y-axis, so the two halves are directly
    comparable and nothing is rescaled across the break.

    Deliberately no prose inside the panes.  The marks that need a key -- the
    open circle for an exact structural zero, the arrow for a point that left
    the clipped ratio window -- are the parent figure's and are keyed in
    ``bump_numbers.txt`` and ``BUMP.md``.
    """
    d = b.d
    edges = bump_edges()
    edges = edges[edges >= 336.0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    two_mt = b.two_mt
    split = 380.0
    xlim = [(edges[0], split), (split, edges[-1])]
    colour = MG7_COLOR if style == 'mg7' else USER_COLOR
    figsize = (8.0, 6.6) if style == 'mg7' else (7.0, 5.8)
    ms = 5 if style == 'mg7' else 4

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[3, 2],
                          hspace=0.06, wspace=0.035)
    axes = [fig.add_subplot(gs[0, 0])]
    axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
    rxs = [fig.add_subplot(gs[1, 0], sharex=axes[0])]
    rxs.append(fig.add_subplot(gs[1, 1], sharex=axes[1], sharey=rxs[0]))

    for col in (0, 1):
        for a in (axes[col], rxs[col]):
            a.set_xlim(*xlim[col])
            if col == 0:
                a.axvspan(xlim[0][0], two_mt, facecolor='0.90',
                          edgecolor='none', zorder=0)
                a.axvline(two_mt, color='0.35', lw=1.0, ls='--', zorder=1)
        axes[col].tick_params(labelbottom=False)
        if col == 1:
            axes[col].tick_params(labelleft=False)
            rxs[col].tick_params(labelleft=False)

    # ---- top: the cumulative excess --------------------------------------
    # The dashed line is the truth's sub-threshold rate.  If the bump is the
    # rate the truth moved across threshold, the curve has to climb to it.
    for col, ax in enumerate(axes):
        ax.axhline(0.0, color='0.35', lw=0.8, zorder=1)
        ax.axhspan(100 * (b.f_below - b.f_below_err),
                   100 * (b.f_below + b.f_below_err),
                   facecolor='black', alpha=0.18, zorder=1)
        ax.axhline(100 * b.f_below, color='black', lw=1.2, ls='--', zorder=2,
                   label=(r'truth $\sigma(m_{t\bar{t}} < 2m_t)\,/\,\sigma$'
                          if col == 0 else None))
        for key in MODES:
            x, exc, err = b.cumulative(key)
            m = (x >= xlim[col][0]) & (x <= xlim[col][1])
            ax.plot(x[m], 100 * exc[m], color=colour[key], lw=1.3, zorder=4,
                    label=CURVES_PLAIN[key] if col == 0 else None)
            if key == 'onshell':
                ax.fill_between(x[m], 100 * (exc - err)[m],
                                100 * (exc + err)[m], color=colour[key],
                                alpha=0.16, lw=0, zorder=3)
        xB, cB = b.cumulative_model('B')
        m = (xB >= xlim[col][0]) & (xB <= min(xlim[col][1], MODEL_HI))
        if m.any():
            ax.plot(xB[m], 100 * cB[m], color=MODEL_COLOR, lw=1.8, ls=':',
                    zorder=5, label=MODEL_LABEL if col == 0 else None)
        ax.set_ylim(-0.09, 0.46)

    axes[0].set_ylabel(r'$\int_{2m_t}^{m_{t\bar t}}'
                       r'\left[\,\mathrm{mode}-\mathrm{truth}\,\right]$'
                       '\n'
                       r'$[\%\ \mathrm{of}\ \sigma]$',
                       fontsize=11 if style == 'mg7' else 10)
    # Upper left is the one corner of either pane that no curve reaches.
    axes[0].legend(loc='upper left', fontsize=7.5 if style == 'mg7' else 8,
                   framealpha=0.92)
    fig.suptitle(r'$%s$, 13 TeV, LO, $\mu_R=\mu_F=m_t$, '
                 r'BW cut $=%g\,\Gamma_t$'
                 % (PROC_TEX.replace(r'\bar t', r'\bar{t}'),
                    d.meta.get('bwcutoff', 15.0)),
                 fontsize=12 if style == 'mg7' else 11, y=0.965)

    # ---- bottom: the drawn shape ratio, clipped --------------------------
    den, dene, dcnt = b.shape_binned(REF, edges)
    yB = b.rebin(b.model_B, edges) / b.model_sigma
    rB = np.where(den > 0, yB / den, np.nan)
    n_out = 0
    for col, rx in enumerate(rxs):
        rx.axhspan(0.9, 1.1, facecolor='C0', alpha=0.10, zorder=0)
        rx.axhspan(0.95, 1.05, facecolor='C0', alpha=0.16, zorder=0)
        rx.axhline(1.0, color='black', ls='--', lw=0.9, zorder=2)
        rx.set_ylim(*RATIO_CLIP)
        inpane = (centres >= xlim[col][0]) & (centres <= xlim[col][1])
        for slot, key in enumerate(MODES):
            y, ye, cnt = b.shape_binned(key, edges)
            r, re = ratio(y, ye, den, dene)
            # Structural zeros on THIS binning.  Same rule as the parent
            # figure's ``structurally_empty``, re-derived because that helper is
            # tied to the parent's edges: a mode that provably returns every
            # event's m_tt unchanged inherits the on-shell production sample's
            # hard boundary at 2 m_t, so its emptiness below it is exact and not
            # a statement about N.
            struct = ((centres < two_mt) & (dcnt > 0) if preserves_mtt(d, key)
                      else np.zeros(len(centres), dtype=bool))
            assert not (struct & (cnt > 0)).any(), (
                '%s has events below 2 m_t -- the structural-zero mark is '
                'false' % key)
            gone = struct | ((cnt == 0) & (dcnt > 0))
            show = inpane & ~gone
            rx.errorbar(centres[show], r[show], yerr=re[show], fmt='o', ms=ms,
                        color=colour[key], zorder=4)
            ring = struct & inpane
            if ring.any():
                rx.plot(centres[ring], np.full(ring.sum(), RATIO_CLIP[0]), 'o',
                        mfc='white', mec=colour[key], mew=1.2,
                        ms=ms + 1 + 2.4 * MODES.index(key), clip_on=False,
                        zorder=8 + (len(MODES) - MODES.index(key)))
            nb, na = offscale_arrows(rx, centres[show], r[show], colour[key],
                                     dx=widths[show], slot=slot,
                                     nslot=len(MODES), lw=0.9, scale=8)
            n_out += nb + na
        keep = inpane & (edges[1:] <= MODEL_HI)
        if keep.any():
            k = np.flatnonzero(keep)
            step_edges = np.concatenate([[edges[k[0]]], edges[k + 1]])
            rx.step(step_edges, np.concatenate([rB[k][:1], rB[k]]),
                    where='pre', color=MODEL_COLOR, lw=1.8, ls=':', zorder=6)
            nb, na = offscale_arrows(rx, centres[keep], rB[keep], MODEL_COLOR,
                                     dx=widths[keep], lw=0.9, scale=8)
            n_out += nb + na
        rx.set_xlabel(r'$m_{t\bar t}$ [GeV]' if col == 1 else ' ')

    rxs[0].set_ylabel('Shape ratio\n(clipped to $\\pm20\\%$)',
                      fontsize=9 if style == 'mg7' else 8.5)

    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.105, top=0.905)
    base = os.path.join(out, 'mtt_bump')
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=300)
    plt.close(fig)
    print('wrote %s.pdf / .png  (%d point(s) outside the clipped pane, each '
          'drawn as an arrow at the boundary it left through)' % (base, n_out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    ap.add_argument('--out-user', default=os.path.join(_HERE, 'plots_userstyle'))
    ap.add_argument('--no-check-minus', action='store_true')
    args = ap.parse_args()

    d = Data(args.data)
    b = Bump(d)

    # MG7 style first: importing plot_mtt_threshold has already set its
    # rcParams.  Then reset, exactly as plot_mtt_threshold_userstyle.py does.
    os.makedirs(args.out, exist_ok=True)
    draw(b, args.out, 'mg7')
    # The MG7 rendering goes through matplotlib's usetex Type1 subsetting, which
    # has silently eaten every minus sign in this project twice; this figure
    # carries one in its y-axis label.  Same guard the parent figure uses, and
    # it is discriminating here too: NO_MINUS_FIX=1 makes it report False.
    if USETEX and not args.no_check_minus:
        ok, why = check_minus(os.path.join(args.out, 'mtt_bump.pdf'))
        print('minus check: %s -- %s' % (ok, why))
        assert ok, why
    mpl.rcParams.update(mpl.rcParamsDefault)
    matplotlib.use('Agg')
    os.makedirs(args.out_user, exist_ok=True)
    draw(b, args.out_user, 'user')

    write_numbers(b)
    for out in (args.out, args.out_user):
        with open(os.path.join(out, 'bump_numbers.txt'), 'w') as fh:
            write_numbers(b, fh)
        print('wrote %s' % os.path.join(out, 'bump_numbers.txt'))


if __name__ == '__main__':
    main()
