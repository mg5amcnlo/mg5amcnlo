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
             'PA': r'full, \texttt{spinmode = PA}'}
EXTRA_TXT = {'onshell': 'full, spinmode = onshell',
             'PA': 'full, spinmode = PA'}

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


def load_extras(ddir=None):
    """The extra spinmode samples that are actually cached, in figure order.

    Missing quietly rather than loudly: the reference figures are a complete
    piece of work on their own and must still be re-makeable from
    ``weights.npz`` alone, which is what someone who only ran the first
    extraction has.
    """
    ddir = ddir or os.path.join(_HERE, 'data')
    out = []
    for key, npz, meta in EXTRA_SAMPLES:
        if os.path.exists(os.path.join(ddir, npz)) and \
                os.path.exists(os.path.join(ddir, meta)):
            out.append((key, Data(ddir, npz, meta)))
    return out


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
