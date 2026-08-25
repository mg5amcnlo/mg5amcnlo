#!/usr/bin/env python3
"""Everything the figures and ``numbers.txt`` are computed from, in one place.

Runs entirely off ``data/weights.npz`` and ``data/meta.json`` as written by
``extract_hepmc_pol.py``.  The 14 GB HepMC file is read exactly once, by that
script; nothing here ever touches it.

Three things this module owns and the plotting scripts do not re-decide.

Which weight is the full
------------------------
``"Weight"``.  The evidence, all of it checked in :func:`nominal_evidence` and
printed into ``numbers.txt``:

* ``"0"`` and ``"Weight"`` are the *same* weight in two normalisations --
  ``"0" * N_events == "Weight"`` holds to the last bit for every one of the
  250 000 events.  ``sum("0") == mean("Weight")`` is the cross section, and it
  reproduces the ``C`` line of the last event to twelve digits.  Every ratio in
  this study is therefore identical whichever of the two is used, and the
  question "which is nominal" has no numerical consequence for the ratios --
  only for the vertical scale of the distributions.
* The four ``ms_pol_*`` weights are in the ``"Weight"`` normalisation: they are
  of order 10, not of order 1e-5.  Dividing them by ``"0"`` would inflate every
  ratio by a factor of 250 000.
* ``"MUR1.0_MUF1.0"`` is the LHE ``<wgt id='1001'>`` copy of the same nominal,
  agreeing with ``"Weight"`` to 4e-6 in the sum -- the rounding of the LHE
  ``rwgt`` block.  It is a re-computation of the nominal, not an independent
  candidate.

So: ratios use ``"Weight"``; the absolute ``dsigma/dx`` of the distributions
uses ``"0"``, whose sum over events *is* the cross section in pb.

The ratio errors
----------------
Numerator and denominator are sums over the *same events*, and the polarisation
weight of an event is very strongly correlated with its nominal weight -- for
the sum of the four it is nearly proportional to it.  Treating them as
independent samples overstates the error by a factor of two on the sum pane and
turns a 5-sigma shape into a 2-sigma one.

:func:`ratio` uses the linearised (delta-method) error for a ratio of two sums
over a common sample::

    R = N / D,   N = sum_i n_i,  D = sum_i d_i
    dR = (dN - R dD) / D
    var(R) = sum_i (n_i - R d_i)^2 / D^2

which is algebraically the jackknife over events, and is exactly the naive
independent-samples formula minus the covariance term.  When ``n_i = c d_i``
event by event it correctly returns zero error, which the naive formula does
not.

The selection
-------------
See :data:`SEL` and RESULTS.md.  The MadSpin card of these three samples is
EXCLUSIVE -- ``decay z > e+ e-`` and ``decay z > mu+ mu-`` -- and the event
record says it is exclusive in effect as well as in intent: all 250 000 events
of all three files have one ``z`` to ``e+e-`` and the other to ``mu+mu-``.  The
four-lepton final state is therefore every event, not the 0.23 % of the earlier
inclusive ``decay z > light light`` samples, and both observables have real
statistics.  The absolute normalisation means something different for it: these
cross sections carry the two leptonic branching fractions, the earlier ones
carried an essentially inclusive one.

The three spinmodes
-------------------
Three HepMC files, the same process and the same 250 000 events' worth of
showering, differing only in MadSpin's ``spinmode``: ``madspin`` (run_06, the
reference), ``onshell`` (run_08) and ``PA`` (run_07).  Each has its own ``.npz``
and its own ``meta.json``, each re-derives its own nominal from its own ``"0"``
/ ``"Weight"`` / ``C`` line, and :func:`full_distribution` puts the three on the
same binning, the same lepton selection and the same absolute ``dsigma/dx``
scale so that they are comparable in rate and not only in shape.

The polarisation decomposition remains the *reference* (``madspin``, run_06)
sample's, and the extra samples enter one place only: the distribution pane of
each figure.  All three files happen to carry the four ``ms_pol_*`` weights
(they are all ``keep_weight_for_polarization_vector = [0, T]`` runs), but
nothing here is built on that -- decomposing the other two modes is a different
study, and the ratio panes belong to the sample whose decomposition they are.

The two figure variants
-----------------------
Beside the three-tier figure the two plotting scripts also draw, into
subdirectories of their own output directories and without touching it, the two
variants registered in :data:`VARIANTS`:

``A``
    the same three tiers with the ``onshell`` and ``PA`` curves dropped, i.e.
    the reference sample's polarisation decomposition on its own.

``B``
    the distribution pane, and then ONE ratio pane instead of the sum pane and
    the 2 x 2: the SHAPE ratio of each extra mode to the reference, each curve
    divided by its own cross section first so that the rate difference divides
    out.  See :data:`SHAPE_NORM` for which cross section that is,
    :func:`shape_density` for the within-sample error and
    :func:`pairing_evidence` for why the between-sample one is a plain
    quadrature sum.
"""

import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

POL_KEYS = ['LL', 'LT', 'TL', 'TT']
# The order the ratio panes draw the four in, as asked: the sum leads, then
# LL, TT, TL, LT.
PANE_ORDER = ['SUM', 'LL', 'TT', 'TL', 'LT']

# The lepton selection.  pT > 10 GeV and |eta| < 2.5 is the standard
# four-lepton fiducial and it is the one used here; the reasons it is not
# looser, given how thin the four-lepton sample already is, are in RESULTS.md.
# It buys a truth-matched purity of 90 % (four lepton) and 96 % (e+e-);
# dropping the eta cut would nearly double the four-lepton statistics but the
# resulting "leptons" reach |eta| = 8 and correspond to no measurement.
PT_MIN = 10.0
ETA_MAX = 2.5
# "a few GeV": the threshold at which a second same-flavour same-sign lepton is
# counted as making the highest-pT choice a genuine choice.
AMBIG_PT = 3.0

N_EVENTS_NORM = 250000     # asserted against the file, see ``load``


class Data(object):
    """One sample's ``.npz`` plus the selections and the weight columns.

    ``npz`` / ``meta`` name the pair to read, so that the three spinmode
    samples can live side by side in the same ``data/`` directory.  The
    polarisation block is optional: :attr:`has_pol` says whether this file's
    ``N`` line carried the four ``ms_pol_*`` weights, and everything that
    consumes :attr:`pol` must ask first.
    """

    def __init__(self, ddir=None, npz='weights.npz', meta='meta.json'):
        ddir = ddir or os.path.join(_HERE, 'data')
        self.ddir = ddir
        self.z = dict(np.load(os.path.join(ddir, npz)))
        self.meta = json.load(open(os.path.join(ddir, meta)))
        self.label = self.meta.get('label', 'madspin')
        self.n = len(self.z['w_Weight'])
        assert self.n == self.meta['n_events']

        self.full = self.z['w_Weight']
        # ``"0"`` is ``"Weight" / n_events``: its sum over the sample IS the
        # cross section in pb, so it is the right column for an absolute
        # dsigma/dx.  The polarisation weights live in the ``"Weight"``
        # normalisation and are put on the same footing by the same factor.
        # Re-derived per sample rather than carried over -- see
        # :meth:`nominal_evidence`, which every sample is put through.
        self.scale_to_pb = 1.0 / self.n
        pm = self.meta['pol_map']
        self.has_pol = all('w_' + pm[k] in self.z for k in POL_KEYS)
        self.pol = {}
        if self.has_pol:
            self.pol = {k: self.z['w_' + pm[k]].astype(np.float64)
                        for k in POL_KEYS}
            self.pol['SUM'] = sum(self.pol[k] for k in POL_KEYS)

        self.sel = {k: v(self) for k, v in SEL.items()}

    # -- the pieces of the "which weight is nominal" argument ----------------
    def nominal_evidence(self):
        w0, w = self.z['w_0'], self.z['w_Weight']
        mu = self.z.get('w_MUR1.0_MUF1.0')
        r = w0 * self.n / w
        return {
            'label': self.label,
            'n_events': self.n,
            'w0_times_N_over_Weight_min': float(r.min()),
            'w0_times_N_over_Weight_max': float(r.max()),
            'sum_w0_pb': float(w0.sum()),
            'mean_Weight_pb': float(w.mean()),
            'C_line_last_event_pb': self.meta['C_line_last_event'][0],
            'C_line_first_event_pb': self.meta['C_line_first_event'][0],
            'sum_MUR1_over_sum_Weight_minus_1':
                float(mu.sum() / w.sum() - 1.0) if mu is not None
                else float('nan'),
            'n_negative': int((w < 0).sum()),
            'frac_negative': float((w < 0).mean()),
            'distinct_abs_Weight': [float(x) for x in np.unique(np.abs(w))],
            'has_pol_weights': self.has_pol,
            'pol_over_Weight_order_of_magnitude':
                float(np.median(np.abs(self.pol['SUM'])) / np.median(np.abs(w)))
                if self.has_pol else float('nan'),
        }


def _lep_ok(d, name):
    """pT and |eta| cuts on one DRESSED lepton, False where it is absent."""
    pt = d.z['pt_%s_dr' % name]
    eta = d.z['eta_%s' % name]
    with np.errstate(invalid='ignore'):
        return np.nan_to_num(pt, nan=-1.0) > PT_MIN, np.abs(
            np.nan_to_num(eta, nan=99.0)) < ETA_MAX


def _sel(d, names, obs):
    ok = np.isfinite(d.z[obs])
    for nm in names:
        a, b = _lep_ok(d, nm)
        ok &= a & b
    return ok


SEL = {
    # M(e+ mu+) needs one lepton from each Z, so all four are required.
    'm_epmup_dr': lambda d: _sel(d, ('ep', 'em', 'mup', 'mum'), 'm_epmup_dr'),
    # Delta phi(e+ e-) needs only the Z that went to electrons.
    'dphi_ee_dr': lambda d: _sel(d, ('ep', 'em'), 'dphi_ee_dr'),
}

# The two observables, their binning, their labels and their vertical scale.
OBS = ['m_epmup_dr', 'dphi_ee_dr']

BINS = {
    # These seven edges were chosen for the earlier inclusive samples, where
    # only 312 events survived the four-lepton fiducial and anything finer
    # would have been a plot of its own noise.  They are KEPT unchanged for the
    # exclusive samples, which leave ~118 000 -- not because the statistics
    # still demand them but so that the two passes' figures and per-bin tables
    # are read against the same edges.  Finer bins are now affordable and are
    # noted in RESULTS.md as something this pass did not do.
    'm_epmup_dr': np.array([0., 45., 70., 95., 125., 175., 260., 450.]),
    # 12 uniform bins over the full [0, pi] range of the observable.
    'dphi_ee_dr': np.linspace(0.0, math.pi, 13),
}

LABELS_TEX = {
    'm_epmup_dr': (r'$M(e^{+}\mu^{+})$ [GeV]',
                   r'$\mathrm{d}\sigma/\mathrm{d}M$ [pb/GeV]'),
    'dphi_ee_dr': (r'$\Delta\phi(e^{+}e^{-})$ [rad]',
                   r'$\mathrm{d}\sigma/\mathrm{d}\Delta\phi$ [pb/rad]'),
}
LABELS_TXT = {
    'm_epmup_dr': ('M(e+ mu+) [GeV]', 'dsigma/dM [pb/GeV]'),
    'dphi_ee_dr': ('Delta phi(e+ e-) [rad]', 'dsigma/dDeltaphi [pb/rad]'),
}
SHORT = {'m_epmup_dr': 'm_epmup', 'dphi_ee_dr': 'dphi_ee'}
LOGY = {'m_epmup_dr'}

# The nominal curve is named for what it is -- the unpolarised total -- and not
# for the weight column it was summed from.  Which column that is is a fact
# about the file and belongs in numbers.txt, not on the figure.
# The nominal curve is named for what it is -- the unpolarised total -- and not
# for the weight column it was summed from.  Which column that is is a fact
# about the file and belongs in numbers.txt, not on the figure; the same goes
# for the four polarisation curves, which carry only their physics name here.
# numbers.txt prints the name-to-column mapping so nothing is unrecoverable.
# The two extra spinmode samples.  Each is a separate MadSpin run of the same
# process showered the same way, cached in its own .npz beside the reference's
# so that the figures stay re-makeable without going near 14 GB of HepMC again.
EXTRA_SAMPLES = [
    ('onshell', 'weights_onshell.npz', 'meta_onshell.json'),
    ('PA', 'weights_PA.npz', 'meta_PA.json'),
]

# A FOURTH sample, and it goes on VARIANT B ONLY -- not on the original
# figures and not on variant A, both of which are left exactly as they were.
#
# ``madspin_v1`` is MadSpin's LEGACY spin-correlation path, against which the
# reference ``madspin`` sample is the current density method.  It is the same
# process, the same run card, the same 250 000 events, the same exclusive
# decay card AND -- the thing that makes the comparison mean anything -- the
# same Pythia8 shower as the other three, from ``run_10_decayed_1``.  All four
# are therefore showered HepMC2 read by the same extractor with the same
# selection, so a difference between the curves is a SPINMODE difference and
# not a difference of event-record level.
#
# It carries NO ms_pol_* weights: its N line names 29 weights where the other
# three name 33.  That costs nothing here, because the variant-B pane needs
# only the nominal, and it is why this sample can never enter the polarisation
# decomposition panes.  ``Data.has_pol`` is False for it and every consumer of
# ``Data.pol`` already asks first.
# A FIFTH sample, and the only one of the five that is not NLO: ``run_12``
# is ``<run_settings> order = LO`` where every other run in the study says
# ``order = NLO``.  Everything else about it matches the reference: the same
# ``p p > z z [QCD]`` process, the same run card, the same 250 000 events, the
# same exclusive ``decay z > e+ e-`` / ``decay z > mu+ mu-`` card, the same
# ``spinmode madspin`` and the same Pythia8 shower.  It therefore DOES carry
# the four ms_pol_* weights -- 33 names on its N line, exactly the reference's
# list -- which is what makes the K-factor figure possible at all.
#
# It goes on variant B, as a FOURTH ``Y`` curve, and on the K-factor figure.
# The original figures and variant A are untouched by it, on the same rule
# that kept ``madspin_v1`` off them.
#
# WHAT ITS VARIANT-B CURVE MEANS IS NOT WHAT THE OTHER THREE MEAN, and the
# figure cannot say so because nothing is written on it, so it is said here,
# in numbers.txt and in RESULTS.md.  The pane is
# ``(1/sigma dsigma/dX)_Y / (1/sigma dsigma/dX)_madspin``.  For ``onshell``,
# ``PA`` and ``madspin_v1`` the numerator and the denominator are the same
# ORDER and differ in SPINMODE, so the curve is a spinmode effect.  For ``LO``
# they are the same SPINMODE and differ in ORDER, so that curve is an order
# effect.  The pane's DEFINITION is unchanged and no curve is computed
# differently -- ``LO`` is simply another ``Y`` -- but a reader who takes the
# fourth curve for a fourth spinmode would be reading the wrong physics off
# it.  Note also that the pane divides each curve by its own sigma, so the
# thing an order change is most visible in -- the RATE, the K-factor -- is
# exactly what this pane removes.  That is what the K-factor figure is for.
VARIANT_B_EXTRA_SAMPLES = [
    ('madspin_v1', 'weights_madspin_v1.npz', 'meta_madspin_v1.json'),
    ('LO', 'weights_LO.npz', 'meta_LO.json'),
]

# The LO sample again, on its own, for the K-factor figure.  Same entry as the
# variant-B one; kept separate because that figure loads exactly one partner
# for the reference and must not silently acquire whatever else variant B
# grows.
KFACTOR_SAMPLE = ('LO', 'weights_LO.npz', 'meta_LO.json')

# The caption these figures used to carry as a plot title.  Nothing is written
# on them now beyond the axis labels, the tick labels and the legend, so the
# caption lives here and is printed into numbers.txt by
# ``plot_zz_pol.write_numbers``.
CAPTION = ('p p > z z [QCD] (MC@NLO) + MadSpin, Pythia8 showered; 13 TeV, '
           '250k events per spinmode; MadSpin card decay z > e+ e- and '
           'decay z > mu+ mu- (exclusive: every event has one z -> e+e- and '
           'one z -> mu+mu-).  Cross sections are therefore sigma(p p > z z) '
           'times the two leptonic branching fractions, not an inclusive '
           'sigma.')

# Below this many selected events a curve is not drawn.  It is not a
# statistics-free number: with the earlier inclusive ``decay z > light light``
# card the four-lepton observable kept ~0.13 % of the events, and at 300 events
# over 7 bins the per-bin error is already 15-20 %.  What the threshold buys is
# the refusal to draw a *second* and *third* such curve on top of the first and
# let the eye read the gaps between three noise realisations as a mode
# difference.  ``numbers.txt`` prints the survivors and the chi2 between the
# modes either way, so nothing is hidden by not drawing it.
#
# With the EXCLUSIVE card of the present three samples both observables clear
# it for every mode by a factor of ~60 (118 000 and 164 000 survivors against
# 2 000), so nothing is dropped and all three modes are on both figures.  The
# threshold is kept, unchanged, for the next sample that does not.
MIN_SEL_TO_DRAW = 2000

CURVE_TEX = {
    'full': r'full (unpolarised), \texttt{spinmode = madspin}',
    'LL': r'$Z_{0}Z_{0}$',
    'TT': r'$Z_{T}Z_{T}$',
    'TL': r'$Z_{T}Z_{0}$',
    'LT': r'$Z_{0}Z_{T}$',
    'SUM': r'$Z_{0}Z_{0} + Z_{T}Z_{T} + Z_{T}Z_{0} + Z_{0}Z_{T}$',
}
CURVE_TXT = {'full': 'full (unpolarised), spinmode = madspin',
             'LL': 'Z0 Z0', 'TT': 'ZT ZT', 'TL': 'ZT Z0', 'LT': 'Z0 ZT',
             'SUM': 'Z0Z0 + ZTZT + ZTZ0 + Z0ZT'}

# What each figure name is, in terms of the weight column it sums.  Printed
# into numbers.txt by plot_zz_pol.write_numbers.
LEGEND_TO_COLUMN = {'Z0 Z0': 'ms_pol_23.0_23.0', 'ZT ZT': 'ms_pol_23.T_23.T',
                    'ZT Z0': 'ms_pol_23.T_23.0', 'Z0 ZT': 'ms_pol_23.0_23.T',
                    CURVE_TXT['full']: 'Weight'}

# The two overlaid samples are the same 'Weight' column of a different file, so
# they are named for the run that produced them and not for a weight.
EXTRA_TEX = {'onshell': r'full, \texttt{spinmode = onshell}',
             'PA': r'full, \texttt{spinmode = PA}',
             'madspin_v1': r'full, \texttt{spinmode = madspin\_v1}',
             # The LO curve is named for the thing that makes it different,
             # which is the ORDER and not the spinmode: its spinmode is
             # ``madspin``, the same as the black reference's.  Labelling it
             # ``spinmode = LO`` like its three neighbours would be a false
             # statement on a figure that carries no other text to correct it.
             'LO': r'full (unpolarised), LO'}
EXTRA_TXT = {'onshell': 'full, spinmode = onshell',
             'PA': 'full, spinmode = PA',
             'madspin_v1': 'full, spinmode = madspin_v1',
             'LO': 'full (unpolarised), LO'}

RATIO_TEX = {
    'SUM': r'$(Z_{0}Z_{0}{+}Z_{T}Z_{T}{+}Z_{T}Z_{0}{+}Z_{0}Z_{T})/\mathrm{full}$',
    'LL': r'$Z_{0}Z_{0}/\mathrm{full}$',
    'TT': r'$Z_{T}Z_{T}/\mathrm{full}$',
    'TL': r'$Z_{T}Z_{0}/\mathrm{full}$',
    'LT': r'$Z_{0}Z_{T}/\mathrm{full}$'}
RATIO_TXT = {'SUM': '(Z0Z0 + ZTZT + ZTZ0 + Z0ZT) / full',
             'LL': 'Z0Z0 / full', 'TT': 'ZTZT / full',
             'TL': 'ZTZ0 / full', 'LT': 'Z0ZT / full'}

# numbers.txt keeps the short internal keys instead: its rows are read next to
# the weight-name list and the mapping table, where LL/TT/TL/LT is the shorter
# and less ambiguous handle.  Only the FIGURES use the Z_0 Z_T spelling.
RATIO_KEY = {'SUM': '(LL+TT+TL+LT) / full', 'LL': 'LL / full',
             'TT': 'TT / full', 'TL': 'TL / full', 'LT': 'LT / full'}


# --------------------------------------------------------------------------
# The three figure variants.  ``full`` is the original three-tier figure and is
# what both plotting scripts still draw by default, into their original output
# directories; the two variants are written ALONGSIDE it, into a subdirectory
# of the same style directory, and nothing about the original figures changes.
VARIANTS = {
    # tier 1 only, and only the reference sample on it.
    'A': {'dir': 'variant_A_madspin_only',
          'extras_on_distribution': False,
          'panes': ['SUM', '2x2'],
          'what': 'the same three-tier figure with the onshell and PA curves '
                  'dropped from the distribution pane'},
    # tier 1 with all three modes, then ONE ratio pane: the shape ratio.
    'B': {'dir': 'variant_B_shape_ratio',
          'extras_on_distribution': True,
          'panes': ['SHAPE'],
          'what': 'the distribution pane, then a single ratio pane carrying '
                  'the self-normalised shape ratio of onshell, PA, '
                  'madspin_v1 and LO to madspin.  The sum pane and the 2 x 2 '
                  'are not drawn.  madspin_v1 and LO are on THIS VARIANT '
                  'ONLY -- the original figures and variant A carry the same '
                  'three modes they always did.  The pane definition is '
                  'unchanged by the LO curve, but that curve is an ORDER '
                  'difference at fixed spinmode where the other three are '
                  'SPINMODE differences at fixed order, and the pane divides '
                  'out exactly the rate change the K-factor figure is about'},
}

# --------------------------------------------------------------------------
# WHICH SIGMA NORMALISES EACH CURVE IN THE VARIANT-B PANE.
#
# ``(1/sigma) dsigma/dx`` needs a sigma and there are two candidates, which are
# not the same number:
#
#   'selected'  the whole selected fiducial cross section, every event that
#               passes the lepton cuts, INCLUDING those whose observable falls
#               outside the drawn range;
#   'inrange'   the integral of the drawn histogram, i.e. only the events
#               inside [edges[0], edges[-1]].
#
# They differ only for ``M(e+ mu+)``, whose last edge is at 450 GeV while the
# observable reaches 4.2 TeV.  ``Delta phi`` is binned over its entire physical
# range [0, pi] and has no outside at all, so for that observable the two are
# the same number to the last bit.
#
# THE CHOICE MADE HERE IS 'inrange', and the reason is that it makes the pane
# an exact statement about what is on the canvas.  With 'inrange' the
# cross-section-weighted mean of the drawn ratio over the drawn bins is 1 by
# construction, so every visible departure from 1 is paid for by another
# visible bin and the reader can account for the whole pane without being told
# about events that are not on it.  With 'selected' the out-of-range rate --
# which is mode dependent, 0.686 % of the selected sigma for madspin against
# 0.730 % for onshell and 0.720 % for PA -- slides the whole pane by a small
# amount whose cause is off the canvas.
#
# It is a small effect and ``numbers.txt`` prints BOTH normalisations bin by
# bin so that the choice can be audited: the largest difference between them in
# any bin of either observable is 0.09 %, about a tenth of the smallest error
# bar on the pane.  The choice is made for the reason above, not because it
# changes an answer.
SHAPE_NORM = 'inrange'
SHAPE_NORM_TXT = {
    'inrange': 'the integral of the drawn histogram (in-range events only)',
    'selected': 'the whole selected fiducial cross section, '
                'including out-of-range events',
}

SHAPE_RATIO_TEX = (r'$\left[\frac{1}{\sigma}\frac{\mathrm{d}\sigma}'
                   r'{\mathrm{d}X}\right]_{Y}\Big/\left[\frac{1}{\sigma}'
                   r'\frac{\mathrm{d}\sigma}{\mathrm{d}X}\right]'
                   r'_{\mathrm{madspin}}$')
SHAPE_RATIO_TXT = '(1/sigma dsigma/dX)_Y / (1/sigma dsigma/dX)_madspin'
# The same label broken over two lines.  The sans-serif rendering spells the
# quantity out where the TeX one sets it as a fraction, and on one line it is
# wider than the pane is tall, so it would run off the bottom of the figure.
SHAPE_RATIO_TXT_2L = ('(1/sigma dsigma/dX)_Y\n'
                      '/ (1/sigma dsigma/dX)_madspin')
SHAPE_CURVE_TEX = {'onshell': r'$Y = $ \texttt{onshell}',
                   'PA': r'$Y = $ \texttt{PA}',
                   'madspin_v1': r'$Y = $ \texttt{madspin\_v1}',
                   'LO': r'$Y = $ LO'}
SHAPE_CURVE_TXT = {'onshell': 'Y = onshell', 'PA': 'Y = PA',
                   'madspin_v1': 'Y = madspin_v1', 'LO': 'Y = LO'}


def shape_density(d, obs, norm=SHAPE_NORM):
    """``(1/sigma) dsigma/dx`` per bin for one sample, with its own error.

    ``norm`` picks the sigma, ``'inrange'`` or ``'selected'``; see
    :data:`SHAPE_NORM` for which one the figures use and why.

    The error is the delta-method one of :func:`ratio`, not a plain
    ``sqrt(sum w^2)``.  The bin content and the normalising sigma are sums over
    the SAME events -- the bin is a subset of the normalisation -- so they are
    correlated, and the part of a bin's fluctuation that is common to the
    normalisation does not move the normalised fraction at all.  Treating them
    as independent overstates the bar by ``1/sqrt(1 - 2 p_b)`` for a bin
    holding a fraction ``p_b`` of the rate, which is 8 % on a twelfth of the
    ``Delta phi`` rate and would understate the chi2 by 14 %.

    This is a WITHIN-sample correlation and is unrelated to the between-sample
    one; see :func:`pairing_evidence`, which establishes that there is no
    between-sample correlation to keep.
    """
    edges = BINS[obs]
    sel = d.sel[obs]
    x = np.asarray(d.z[obs], dtype=np.float64)[sel]
    w = np.asarray(d.full, dtype=np.float64)[sel] * d.scale_to_pb
    if norm == 'inrange':
        den = w * ((x >= edges[0]) & (x <= edges[-1]))
    elif norm == 'selected':
        den = w
    else:
        raise ValueError('unknown shape normalisation %r' % (norm,))
    nb = len(edges) - 1
    width = np.diff(edges)
    idx = np.digitize(x, edges) - 1
    f = np.full(nb, np.nan)
    e = np.full(nb, np.nan)
    n = np.zeros(nb, dtype=int)
    for b in range(nb):
        # np.histogram puts x == edges[-1] in the last bin; np.digitize does
        # not.  Match np.histogram so that this pane and the distribution pane
        # above it bin the same events.
        m = (idx == b) if b < nb - 1 else ((idx == b) | (x == edges[-1]))
        n[b] = int(m.sum())
        R, E = ratio(w * m, den)
        f[b], e[b] = R / width[b], E / width[b]
    return {'label': d.label, 'f': f, 'err': e, 'n': n, 'norm': norm,
            'sigma_norm_pb': float(den.sum()),
            'sigma_selected_pb': float(w.sum()),
            'n_outside': int(len(x) - int(((x >= edges[0])
                                           & (x <= edges[-1])).sum())),
            'sigma_outside_pb': float(w.sum() - den.sum())
            if norm == 'inrange' else
            float(w.sum() - (w * ((x >= edges[0])
                                  & (x <= edges[-1]))).sum())}


def compare_shape(ref, other):
    """``other / ref`` of two :func:`shape_density` results, and its chi2.

    The two samples are independent -- see :func:`pairing_evidence` -- so the
    two relative errors add in quadrature and there is no covariance term to
    subtract.  Were the two samples a common set of production events decayed
    twice, this bar would be wrong and too large; they are not.

    One degree of freedom is removed from the chi2 because each curve was
    divided by its own normalisation, which is exactly one constraint.
    """
    fr, er = np.asarray(ref['f']), np.asarray(ref['err'])
    fo, eo = np.asarray(other['f']), np.asarray(other['err'])
    ok = np.isfinite(fr) & np.isfinite(fo) & (fr > 0) & (fo > 0)
    r = np.full(len(fr), np.nan)
    rel = np.full(len(fr), np.nan)
    r[ok] = fo[ok] / fr[ok]
    rel[ok] = np.sqrt((eo[ok] / fo[ok]) ** 2 + (er[ok] / fr[ok]) ** 2)
    e = r * rel
    good = ok & np.isfinite(e) & (e > 0)
    chi2 = float(np.sum((r[good] - 1.0) ** 2 / e[good] ** 2))
    ndf = max(int(good.sum()) - 1, 0)
    return {'label': other['label'], 'ratio': r, 'ratio_err': e, 'chi2': chi2,
            'ndf': ndf, 'chi2_per_ndf': chi2 / ndf if ndf else float('nan'),
            'max_abs_pull': float(np.max(np.abs(r[good] - 1.0) / e[good]))
            if good.any() else float('nan')}


def pairing_evidence(d, extras):
    """Do the three samples decay a COMMON set of production events?

    If they did, the mode-to-mode ratio would be correlated and a paired error
    would be both correct and much smaller than the quadrature one.  So it has
    to be established rather than assumed, and it is established here from the
    cached ``.npz`` alone.

    Two tests, and they answer at two different strengths.

    ``n_negative``
        the decisive one.  Neither MadSpin nor Pythia can change the SIGN of an
        event weight -- MadSpin multiplies by a positive decay/branching factor
        and Pythia passes the LHE weight through -- so the number of
        negative-weight events is a property of the PRODUCTION sample alone,
        and it is invariant under any reordering of the file.  Three decays of
        one production sample must give the same count exactly.  They do not:
        14 273 / 13 962 / 14 099.  A common production sample is therefore
        excluded outright, in any order, and not merely unproven.

    ``m_4l correlation``
        the corroborating one, and the closest thing the cached columns have to
        the ``sqrt(shat)`` the sibling ``m_tt`` study paired on.  For a
        ``2 -> 2`` production event the four-lepton invariant mass IS the
        production ``m(ZZ)``, so two decays of one production event would agree
        in it to the dressing.  Row by row the correlation is 4e-04 on 243 000
        common rows, against a standard error of 2e-03: zero, where pairing
        would give ~1.

    The cached ``.npz`` carries no production-level column at all -- no
    ``sqrt(shat)``, no event number, no production kinematics -- so the literal
    ``max |Delta sqrt(shat)| = 0`` test of the ``m_tt`` study cannot be run
    here.  It does not need to be: the negative-weight count is a strictly
    stronger test in the direction that matters, because it is permutation
    invariant and it FAILS.
    """
    out = {'ref': d.label, 'rows': [], 'paired': False}
    ref_neg = int((d.full < 0).sum())
    for lab, dx in [(d.label, d)] + list(extras):
        w = np.asarray(dx.full, dtype=np.float64)
        row = {'label': dx.label, 'n_events': dx.n,
               'n_negative': int((w < 0).sum()),
               'frac_negative': float((w < 0).mean()),
               'abs_weight_values': [float(v)
                                     for v in np.unique(np.abs(w))],
               'sigma_pb': float(np.asarray(dx.z['w_0'],
                                            dtype=np.float64).sum())}
        if dx is d:
            row['corr_m_4l'] = float('nan')
            row['n_common_rows'] = dx.n
            row['same_n_negative'] = True
        else:
            a = np.asarray(d.z['m_4l'], dtype=np.float64)
            b = np.asarray(dx.z['m_4l'], dtype=np.float64)
            k = min(len(a), len(b))
            a, b = a[:k], b[:k]
            ok = np.isfinite(a) & np.isfinite(b)
            row['n_common_rows'] = int(ok.sum())
            row['corr_m_4l'] = float(np.corrcoef(a[ok], b[ok])[0, 1]) \
                if ok.sum() > 2 else float('nan')
            row['corr_stderr'] = 1.0 / math.sqrt(max(int(ok.sum()), 1))
            row['same_n_negative'] = row['n_negative'] == ref_neg
        out['rows'].append(row)
    out['paired'] = all(r['same_n_negative'] for r in out['rows'])
    return out


# --------------------------------------------------------------------------
def ratio(num, den):
    """``(R, sigma_R)`` for two weight arrays summed over the SAME events.

    See the module docstring: the error is the delta-method / jackknife one,
    ``var(R) = sum_i (n_i - R d_i)^2 / D^2``, which keeps the covariance
    between numerator and denominator instead of throwing it away.
    """
    D = den.sum()
    if D == 0 or len(den) == 0:
        return float('nan'), float('nan')
    R = num.sum() / D
    return float(R), float(math.sqrt(np.sum((num - R * den) ** 2)) / abs(D))


def naive_ratio_error(num, den):
    """What a wrong, independent-samples error bar would have been.

    Kept only so that ``numbers.txt`` can print the factor by which the correct
    treatment shrinks the bar; never used to draw anything.
    """
    N, D = num.sum(), den.sum()
    if N == 0 or D == 0:
        return float('nan')
    return float(abs(N / D) * math.sqrt(np.sum(num ** 2) / N ** 2
                                        + np.sum(den ** 2) / D ** 2))


def histogram(x, w, edges):
    """``(dsigma/dx, error)`` in pb per unit of ``x``."""
    y, _ = np.histogram(x, bins=edges, weights=w)
    y2, _ = np.histogram(x, bins=edges, weights=w * w)
    width = np.diff(edges)
    return y / width, np.sqrt(y2) / width


def binned_ratio(x, num, den, edges):
    """Per-bin ``num/den`` with the correlated error, plus the bin populations."""
    idx = np.digitize(x, edges) - 1
    nb = len(edges) - 1
    r = np.full(nb, np.nan)
    e = np.full(nb, np.nan)
    n = np.zeros(nb, dtype=int)
    for b in range(nb):
        m = idx == b
        n[b] = m.sum()
        if n[b]:
            r[b], e[b] = ratio(num[m], den[m])
    return r, e, n


class Curves(object):
    """One observable's histograms and ratio panes, computed once."""

    def __init__(self, d, obs):
        self.obs = obs
        self.edges = BINS[obs]
        self.x = np.asarray(d.z[obs], dtype=np.float64)
        self.sel = d.sel[obs]
        xs = self.x[self.sel]
        self.xs = xs
        self.n_sel = int(self.sel.sum())
        f = d.full[self.sel]
        s = d.scale_to_pb
        self.dist = {'full': histogram(xs, f * s, self.edges)}
        for k in POL_KEYS:
            self.dist[k] = histogram(xs, d.pol[k][self.sel] * s, self.edges)
        self.ratios = {}
        for k in PANE_ORDER:
            self.ratios[k] = binned_ratio(xs, d.pol[k][self.sel], f, self.edges)
        self.integrated = {k: ratio(d.pol[k][self.sel], f) for k in PANE_ORDER}
        self.sigma_pb = float((f * s).sum())

    def centres(self):
        return 0.5 * (self.edges[:-1] + self.edges[1:])


def _load_listed(samples, ddir):
    """Those of ``samples`` whose ``.npz`` and meta are actually on disk.

    Missing quietly rather than loudly: the reference figures are a complete
    piece of work on their own and must still be re-makeable from
    ``weights.npz`` alone, which is what someone who only ran the first
    extraction has.
    """
    ddir = ddir or os.path.join(_HERE, 'data')
    out = []
    for key, npz, meta in samples:
        if os.path.exists(os.path.join(ddir, npz)) and \
                os.path.exists(os.path.join(ddir, meta)):
            out.append((key, Data(ddir, npz, meta)))
    return out


def load_extras(ddir=None):
    """The extra spinmode samples drawn on EVERY figure, in figure order.

    ``onshell`` and ``PA``.  Deliberately NOT ``madspin_v1``: that one is
    variant B's alone -- see :func:`load_variant_b_extras` -- so that this
    function keeps returning exactly what it returned before it existed and
    the original figures and variant A are untouched by its arrival.
    """
    return _load_listed(EXTRA_SAMPLES, ddir)


def load_variant_b_extras(ddir=None):
    """The samples that go on VARIANT B ONLY, in figure order.

    Currently ``madspin_v1``.  Variant B draws
    ``load_extras() + load_variant_b_extras()``; every other figure draws
    ``load_extras()`` alone.
    """
    return _load_listed(VARIANT_B_EXTRA_SAMPLES, ddir)


def full_distribution(d, obs):
    """One sample's full (unpolarised) ``dsigma/dx``, on the shared binning.

    Same lepton selection, same bin edges and the same absolute normalisation
    (``"Weight" / n_events``, whose sum over the sample is the cross section in
    pb) as :class:`Curves` builds for the reference, so that the three samples
    are comparable in RATE and not only in shape.  Each sample's own
    ``n_events`` is used, not the reference's.
    """
    sel = d.sel[obs]
    x = np.asarray(d.z[obs], dtype=np.float64)[sel]
    w = np.asarray(d.full, dtype=np.float64)[sel] * d.scale_to_pb
    y, e = histogram(x, w, BINS[obs])
    return {'label': d.label, 'n_sel': int(sel.sum()), 'y': y, 'err': e,
            'sigma_pb': float(w.sum()),
            'drawable': int(sel.sum()) >= MIN_SEL_TO_DRAW}


def compare_full(ref, other):
    """``other / ref`` bin by bin, and the chi2 between the two.

    Note the error treatment, which is the OPPOSITE of :func:`ratio`'s and for
    a good reason.  The polarisation ratios divide two weights of the SAME
    events and must keep the covariance.  These are two different files -- two
    independent MadSpin runs, independently showered -- so the plain
    quadrature sum is the correct error here and there is no covariance to
    keep.

    Two chi2 come back, and reporting only the first would misread the result.
    ``chi2`` compares the two distributions as drawn, rate included.
    ``shape_chi2`` rescales ``other`` to ``ref``'s fiducial rate first and so
    tests the SHAPE alone, on one fewer degree of freedom.  A mode that differs
    only in normalisation gives a large ``chi2`` and a small ``shape_chi2``,
    and calling that a shape difference would be wrong.
    """
    y1, e1 = np.asarray(ref['y']), np.asarray(ref['err'])
    y2, e2 = np.asarray(other['y']), np.asarray(other['err'])
    ok = np.isfinite(y1) & np.isfinite(y2) & (y1 > 0) & (y2 > 0)
    r = np.full(len(y1), np.nan)
    er = np.full(len(y1), np.nan)
    r[ok] = y2[ok] / y1[ok]
    er[ok] = r[ok] * np.sqrt((e2[ok] / y2[ok]) ** 2 + (e1[ok] / y1[ok]) ** 2)
    d2 = e1 ** 2 + e2 ** 2
    good = ok & np.isfinite(d2) & (d2 > 0)
    chi2 = float(np.sum((y2[good] - y1[good]) ** 2 / d2[good]))
    ndf = int(good.sum())
    s1, s2 = ref['sigma_pb'], other['sigma_pb']
    f = s1 / s2 if s2 else float('nan')
    y2s, e2s = y2 * f, e2 * f
    d2s = e1 ** 2 + e2s ** 2
    gs = ok & np.isfinite(d2s) & (d2s > 0)
    shape_chi2 = float(np.sum((y2s[gs] - y1[gs]) ** 2 / d2s[gs]))
    shape_ndf = max(int(gs.sum()) - 1, 0)
    return {'ratio': r, 'ratio_err': er, 'chi2': chi2, 'ndf': ndf,
            'shape_ratio': np.where(ok, y2s / y1, np.nan),
            'shape_chi2': shape_chi2, 'shape_ndf': shape_ndf,
            'shape_chi2_per_ndf': shape_chi2 / shape_ndf if shape_ndf
            else float('nan'),
            'chi2_per_ndf': chi2 / ndf if ndf else float('nan'),
            'sigma_ratio': s2 / s1 if s1 else float('nan'),
            'max_abs_pull': float(np.max(np.abs(y2[good] - y1[good])
                                         / np.sqrt(d2[good]))) if ndf
            else float('nan')}


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# THE K-FACTOR FIGURE: NLO/LO, per polarisation component.
#
# This is the one figure in the study that is about the RATE.  Everything else
# here either divides the rate out (variant B's shape pane, which normalises
# each curve by its own sigma) or divides one component of a sample by that
# same sample's total (the sum pane and the 2 x 2).  A K-factor is a ratio of
# CROSS SECTIONS, so nothing may be normalised away before it is taken:
# ``K = sigma_NLO / sigma_LO`` bin by bin, with both sides in pb.
#
# THE TWO FIGURES THEREFORE USE DIFFERENT NORMALISATIONS AND THIS IS NOT AN
# INCONSISTENCY.  Variant B answers "does the SHAPE move", and to see that it
# has to remove the rate, which is the largest and least interesting
# difference between an LO and an NLO sample -- at these cuts the rate moves by
# 29 % and the shape by a few percent, so an un-normalised variant-B pane would
# be four curves sitting at 0.78 and nothing else would be legible.  This
# figure answers "does the RATE move, and does it move by the same factor for
# every polarisation", and normalising would delete the entire answer.
#
# The panel order is the one the figure was asked for and is NOT
# :data:`PANE_ORDER`: unpolarised first, then Z_0Z_0, Z_0Z_T, Z_TZ_0, Z_TZ_T,
# and the K-factor panel sixth.
KF_PANE_ORDER = ['full', 'LL', 'LT', 'TL', 'TT']

KFACTOR_TEX = r'$K = \mathrm{NLO}/\mathrm{LO}$'
KFACTOR_TXT = 'K = NLO / LO'
KF_ORDER_TEX = {'NLO': 'NLO', 'LO': 'LO'}

# Where the K-factor figure is written, inside each style's plot directory,
# beside plots/ and plots/variant_A_madspin_only/ and
# plots/variant_B_shape_ratio/.  Nothing already there is touched.
KFACTOR_DIR = 'kfactor_LO_NLO'

# The y-axis label of panels 1-5.  It is the ABSOLUTE dsigma/dx that
# ``LABELS_TEX[obs][1]`` already names, and it is reused verbatim so that the
# panels and the distribution pane of the other figures are read on the same
# axis.  Panel 6 is dimensionless and takes KFACTOR_TEX / KFACTOR_TXT.
KF_CURVE_TEX = {'full': r'unpolarised', 'LL': r'$Z_{0}Z_{0}$',
                'LT': r'$Z_{0}Z_{T}$', 'TL': r'$Z_{T}Z_{0}$',
                'TT': r'$Z_{T}Z_{T}$'}
KF_CURVE_TXT = {'full': 'unpolarised', 'LL': 'Z0 Z0', 'LT': 'Z0 ZT',
                'TL': 'ZT Z0', 'TT': 'ZT ZT'}


def load_kfactor_partner(ddir=None):
    """The LO sample, or ``None`` when its ``.npz`` is not on disk.

    ``None`` rather than an exception, on the same rule as :func:`_load_listed`:
    someone who has run only the reference extraction must still be able to
    remake every figure that does not need this one.  The K-factor figure is
    then skipped and said to be skipped, not faked.
    """
    got = _load_listed([KFACTOR_SAMPLE], ddir)
    return got[0][1] if got else None


def component_histogram(d, obs, key):
    """One sample's ABSOLUTE ``dsigma/dx`` for one component, in pb per unit x.

    ``key`` is ``'full'`` or one of :data:`POL_KEYS`.  The weight column is put
    in pb by the sample's own ``scale_to_pb`` -- its own ``n_events``, not the
    reference's -- so that two samples of different size would still be
    comparable in rate.  (Here both are 250 000.)

    THE ERROR IS THE PLAIN MC ONE, ``sqrt(sum w^2)``, and that is correct here
    where it would be wrong on the ratio panes.  The delta-method bar of
    :func:`ratio` exists to keep the covariance between a numerator and a
    denominator summed over the SAME events; this function returns a single
    sum, not a ratio, so there is no covariance to keep.
    """
    sel = d.sel[obs]
    x = np.asarray(d.z[obs], dtype=np.float64)[sel]
    w = np.asarray(d.full if key == 'full' else d.pol[key],
                   dtype=np.float64)[sel] * d.scale_to_pb
    y, e = histogram(x, w, BINS[obs])
    return {'label': d.label, 'key': key, 'n_sel': int(sel.sum()),
            'y': y, 'err': e, 'sigma_pb': float(w.sum()),
            'drawable': int(sel.sum()) >= MIN_SEL_TO_DRAW}


def kfactor(nlo, lo, obs, key):
    """``(NLO/LO, error)`` bin by bin for one component, plus both histograms.

    WHICH ERROR, AND WHY IT IS THE PLAIN QUADRATURE ONE.  The numerator is a
    sum over the NLO sample and the denominator a sum over the LO one, and
    those are two different generations -- different order, different run
    (``run_06`` against ``run_12``), and no event in one is an event in the
    other.  :func:`pairing_evidence` is not even needed to see it, though it
    agrees: the LO sample has ``n(w<0) = 0`` against the reference's 14 273.
    So the two relative errors add in quadrature and there is no covariance
    term to subtract.

    The delta-method bar of :func:`ratio` does NOT apply to this quantity and
    is deliberately not used.  It applies to a ratio whose two sums run over
    one set of events -- a component against its OWN sample's total, which is
    what the sum pane and the 2 x 2 draw, and what
    :func:`component_fraction_double_ratio` below uses on each side before
    dividing.  Using it here would be claiming a cancellation between two
    independent samples that does not exist, and would make every bar on the
    K-factor panel too small.

    The bin widths cancel in the ratio and are left in only so that the two
    returned histograms are the same ``dsigma/dx`` the panels above draw.
    """
    a = component_histogram(nlo, obs, key)
    b = component_histogram(lo, obs, key)
    ya, ea = np.asarray(a['y'], float), np.asarray(a['err'], float)
    yb, eb = np.asarray(b['y'], float), np.asarray(b['err'], float)
    ok = np.isfinite(ya) & np.isfinite(yb) & (ya > 0) & (yb > 0)
    k = np.full(len(ya), np.nan)
    e = np.full(len(ya), np.nan)
    k[ok] = ya[ok] / yb[ok]
    e[ok] = k[ok] * np.sqrt((ea[ok] / ya[ok]) ** 2 + (eb[ok] / yb[ok]) ** 2)
    return {'key': key, 'k': k, 'err': e, 'nlo': a, 'lo': b,
            'k_integrated': (a['sigma_pb'] / b['sigma_pb']
                             if b['sigma_pb'] else float('nan'))}


def integrated_kfactors(nlo, lo, obs=None):
    """The K-factor of every component as ONE number, with its error.

    ``obs=None`` gives the inclusive one: every event of both samples, no
    lepton selection at all, which is the cleanest statement of the physics
    question the figure exists to answer.  Passing an observable restricts both
    sides to that observable's fiducial selection, which is what the panels
    actually show.

    Same error argument as :func:`kfactor`: two independent samples, so the two
    relative MC errors go in quadrature.
    """
    rows = []
    for key in KF_PANE_ORDER:
        def one(d):
            w = np.asarray(d.full if key == 'full' else d.pol[key],
                           dtype=np.float64)
            if obs is not None:
                w = w[d.sel[obs]]
            w = w * d.scale_to_pb
            return float(w.sum()), float(math.sqrt(float(np.sum(w * w))))
        a, ea = one(nlo)
        b, eb = one(lo)
        K = a / b if b else float('nan')
        eK = (abs(K) * math.sqrt((ea / a) ** 2 + (eb / b) ** 2)
              if a and b else float('nan'))
        rows.append({'key': key, 'sigma_nlo_pb': a, 'err_nlo_pb': ea,
                     'sigma_lo_pb': b, 'err_lo_pb': eb, 'K': K, 'K_err': eK})
    full = rows[0]
    for r in rows:
        # How far this component's K sits from the unpolarised one.  Quoted
        # with a plain quadrature bar, which OVERSTATES it -- a component and
        # its own sample's total are strongly correlated, so most of the
        # common MC fluctuation cancels in the difference and the true
        # significance is larger, not smaller.  The sharp version of the same
        # statement is the double ratio below, which handles that correlation
        # properly on each side.  This column is the conservative one.
        d = r['K'] - full['K']
        sd = math.sqrt(r['K_err'] ** 2 + full['K_err'] ** 2)
        r['K_minus_K_full'] = d
        r['K_minus_K_full_err'] = sd
        r['K_minus_K_full_sigma'] = abs(d) / sd if sd else float('nan')
    return rows


def component_fraction_double_ratio(nlo, lo, obs=None):
    """``f_NLO / f_LO`` where ``f = sigma_component / sigma_full`` per sample.

    THIS is where the delta-method bar belongs, and it is the sharpest form of
    the question "do the polarisations have different K-factors".  A component
    fraction ``f`` is a ratio of two sums over the SAME events of ONE sample,
    so its error is :func:`ratio`'s correlated one -- the part of the
    fluctuation common to the component and to the total does not move the
    fraction and must not be counted.  The two fractions then come from the two
    independent samples, so THEY combine in quadrature.

    The double ratio is algebraically ``K_component / K_full``, so it is the
    same statement as the ``K_minus_K_full`` column of
    :func:`integrated_kfactors` -- but with the within-sample correlation kept
    on each side instead of thrown away, which is why its bars are two to four
    times smaller and its significances correspondingly larger.  Both are
    printed into numbers.txt so the difference between the two treatments is
    on the record rather than a choice made silently.
    """
    rows = []
    for key in POL_KEYS:
        def frac(d):
            num = np.asarray(d.pol[key], dtype=np.float64)
            den = np.asarray(d.full, dtype=np.float64)
            if obs is not None:
                num, den = num[d.sel[obs]], den[d.sel[obs]]
            return ratio(num, den)
        f1, e1 = frac(nlo)
        f2, e2 = frac(lo)
        D = f1 / f2 if f2 else float('nan')
        eD = (abs(D) * math.sqrt((e1 / f1) ** 2 + (e2 / f2) ** 2)
              if f1 and f2 else float('nan'))
        rows.append({'key': key, 'f_nlo': f1, 'f_nlo_err': e1,
                     'f_lo': f2, 'f_lo_err': e2, 'double_ratio': D,
                     'double_ratio_err': eD,
                     'sigma_from_1': abs(D - 1.0) / eD if eD else float('nan')})
    return rows


def diagnostics(d, obs):
    """How honest the selection is, measured rather than asserted.

    Two separate things, both of which the brief asks to be quoted rather than
    assumed:

    ``ambiguity``
        how often the highest-pT choice was a genuine CHOICE -- the fraction of
        selected events carrying a second same-flavour same-sign lepton above
        :data:`AMBIG_PT`, per flavour and charge, and above the selection's own
        pT threshold.  A second lepton at 4 GeV next to a selected one at 45
        GeV is a choice that was never close; one above 10 GeV is.

    ``purity`` / ``efficiency``
        against the ``z`` decay channels read out of the event record.  Purity
        is the fraction of selected events whose ``z`` really took the channel
        the observable assumes; efficiency is the fraction of those events the
        selection keeps.  On the earlier INCLUSIVE ``z > light light`` samples
        purity was a real question -- an event could have an ``e+`` and a
        ``mu+`` without either ``z`` decaying to them, via ``Z -> tau tau`` or
        semileptonic ``b``.  On the present EXCLUSIVE samples every event is in
        the channel, so purity is 1 by construction and only the efficiency
        carries information.  It is still computed from the record rather than
        asserted, which is how the exclusivity itself is established.
    """
    z, sel = d.z, d.sel[obs]
    n = int(sel.sum())
    flav = ('ep', 'em', 'mup', 'mum') if obs == 'm_epmup_dr' else ('ep', 'em')
    amb = {}
    for f in flav:
        sub = z['sub_pt_%s' % f][sel]
        amb[f] = (float((sub > AMBIG_PT).mean()), float((sub > PT_MIN).mean()))
    z1, z2 = z['z1_ch'], z['z2_ch']
    if obs == 'm_epmup_dr':
        truth = ((z1 == 11) & (z2 == 13)) | ((z1 == 13) & (z2 == 11))
        what = 'one z -> e+e- AND the other z -> mu+mu-'
    else:
        truth = (z1 == 11) | (z2 == 11)
        what = 'at least one z -> e+e-'
    return {
        'n_selected': n,
        'ambiguity_above_%g_and_above_%g_GeV' % (AMBIG_PT, PT_MIN): amb,
        'truth_requirement': what,
        'n_truth': int(truth.sum()),
        'frac_truth_of_all': float(truth.mean()),
        'purity': float((sel & truth).sum() / n) if n else float('nan'),
        'efficiency': float((sel & truth).sum() / truth.sum()),
    }


def dressing_shift(d, obs):
    """How much the photon dressing moved the observable, and of what size.

    Reported rather than hidden: the study dresses, and a reader is entitled to
    know whether that decision mattered.
    """
    bare = SHORT[obs]
    sel = d.sel[obs]
    a = np.asarray(d.z[bare], dtype=np.float64)[sel]
    b = np.asarray(d.z[obs], dtype=np.float64)[sel]
    ok = np.isfinite(a) & np.isfinite(b)
    dd = b[ok] - a[ok]
    return {'mean_shift': float(dd.mean()), 'rms_shift': float(dd.std()),
            'frac_moved_by_more_than_1pct':
                float((np.abs(dd) > 0.01 * np.abs(a[ok])).mean())}
