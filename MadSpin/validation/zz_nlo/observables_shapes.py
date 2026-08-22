#!/usr/bin/env python3
"""The *shape* observables: where the ``g g`` box does something ``q q~`` does not.

The stacked figure of ``plot_zz_stack.py`` answers "how much does the
loop-induced ``g g -> Z Z`` contribution add, and where".  This module answers a
different question: **is there an observable in which ``LI / LO`` and
``NLO / LO`` differ in SHAPE**, i.e. one in which the box-mediated ``g g``
process behaves unlike the ``q q~`` tree, rather than merely contributing a
different amount.

Nothing here is regenerated.  The observables are read off the same four
samples the parent study already wrote, through the same reader
(``observables_zz.read_lhe_zz``) and the same self-testing boost
(``../zz_loopinduced/observables.boost_to_rest``).

Two facts fix the design and are worth stating before the list.

**The LO denominator cancels out of the discrimination.**  ``(LI/LO)/(NLO/LO)``
is algebraically ``LI/NLO``, so the number that measures "do the two ratio
shapes differ" is the ``chi2`` of ``LI/NLO`` against a flat line -- and its
errors are clean, because ``LI`` and ``NLO`` are independent samples while the
shared ``LO`` would have been fully correlated between the two ratios.  Both
per-ratio ``chi2`` values are reported as well, because "both ratios are flat"
and "both bend the same way" are different results and the double ratio alone
cannot tell them apart.

**Anything sensitive to recoil is a trap.**  ``pt(ZZ)`` and ``Delta phi(Z,Z)``
are exactly ``0`` and exactly ``pi`` for every LO and every loop-induced event
-- both are ``2 -> 2`` -- so they separate NLO from the other two immediately
and for a reason that has nothing to do with the box, and a ratio is not even
defined there.  The parent study already records them and keeps them out of the
figures; the same rule applies here.  ``pt(Z_lead)/m(ZZ)`` is the borderline
case: it is bounded by ``beta/2 < 0.5`` on a ``2 -> 2`` sample and is not
bounded at NLO, so its topmost bin carries a piece of that same artefact.  It
is kept, plotted only over ``[0, 0.5]``, and flagged.

The list
--------

``m_zz_fine``
    ``m(ZZ)``, finely binned through ``2 m_t = 346 GeV``.  The ``g g`` box runs
    through a top loop and has a threshold there; the ``q q~`` tree has no
    analogue.  This is the strongest a-priori candidate.  It does not show a
    step -- see RESULTS.md for the measured limit -- and the fine binning is
    what makes that a limit rather than an absence of looking.

``abs_cos_theta_cs``
    ``|cos theta*|`` of the harder ``Z`` in the **Collins-Soper** frame: the
    polar axis in the ``ZZ`` rest frame is the bisector of the first beam and
    the reversed second beam, both boosted into that frame.

    On a ``2 -> 2`` sample this is **identical** to the ``abs_cos_theta_star``
    the parent study already uses -- the ``ZZ`` system has no transverse
    momentum, so its lab direction *is* the beam axis and the bisector
    degenerates onto it.  Measured, not argued: the two agree to ``2.2e-16``
    event by event on both the LO and the loop-induced sample.  They differ
    only on the NLO one, where the recoil tilts the ``ZZ`` direction away from
    the beam (mean ``|difference| = 0.035``).  So the Collins-Soper frame buys
    nothing for the LO-vs-LI comparison and is here to say so, and to give the
    NLO curve a definition that does not degrade when a jet is present.

    Note the standard massless Collins-Soper shortcut
    ``2(l1+ l2- - l1- l2+)/(Q sqrt(Q^2+qT^2))`` must **not** be used here: with
    massive decay products it returns ``beta cos theta*``, not
    ``cos theta*``, which silently mixes the ``m(ZZ)`` dependence into what is
    meant to be a pure angle.  The frame is therefore built explicitly.

``abs_cos_star_mlow`` / ``abs_cos_star_mmid`` / ``abs_cos_star_mhigh``
    the same ``|cos theta*|``, in three slices of ``m(ZZ)``:
    ``< 300``, ``300-450``, ``>= 450 GeV``.  **This is the answer.**
    Inclusively, ``|cos theta*|`` is famously flat here -- the parent study
    reports ``LI/NLO`` running from 13.3 % to 14.3 % across the whole range and
    concludes it separates nothing.  That conclusion is right about the
    inclusive distribution and wrong about the physics: the inclusive
    flatness is a *cancellation* between a forward rise at low ``m(ZZ)`` and a
    central rise at high ``m(ZZ)``.  Slicing shows both.

``dy_zz``
    ``|y(Z1) - y(Z2)|``.  Boost invariant, so unlike ``|y(ZZ)|`` it carries no
    information about how the initial-state momentum fractions were shared: it
    is the production angle in longitudinal-boost-invariant clothing.  For a
    ``2 -> 2`` event it is a function of ``m(ZZ)`` and ``cos theta*`` alone.

``abs_y_z_lead`` / ``max_abs_y_z``
    the rapidity of the harder ``Z``, and the larger of the two ``|y(Z)|``.
    These have the **largest raw discrimination of anything in the list**, and
    they are the least interesting: they are the ``|y(ZZ)|`` effect the parent
    study already found -- the gluon luminosity is more central than the quark
    one -- read off a single ``Z`` instead of the pair.  They are here so that
    the ranking is honest about which variables win and why.

``pt_over_m``
    ``pt`` of the harder ``Z`` divided by ``m(ZZ)``.  Dimensionless, so the
    overall rate and the ``m(ZZ)`` spectrum both factor out of the *scale*; for
    a ``2 -> 2`` event it is exactly ``(beta/2) sin theta*``.  See the boundary
    caveat above.

Separating a matrix-element effect from a luminosity one
-------------------------------------------------------
:func:`mzz_reweight` returns per-event weights that force a sample's ``m(ZZ)``
spectrum onto a target one.  Reweighting LO and LI onto the NLO ``m(ZZ)``
spectrum and re-measuring the discrimination answers "is what is left still
there once the two samples are made to sit at the same partonic energy" --
which is the difference between "the box is a different matrix element" and
"the gluon PDF falls faster".  It is a diagnostic, reported in
``numbers_shapes.txt``, and it is deliberately NOT what the figures draw: the
figures show the samples as generated.

The four-lepton twins
---------------------
Every ``Z``-level observable above is also defined on the decayed samples, as a
function of the reconstructed pairs ``(e+ e-)`` and ``(mu+ mu-)`` and of
nothing else.  **The reconstruction is exact**, in a specific and checkable
sense: the two ``Z`` are flavour tagged -- one decays to electrons and one to
muons -- so there is no combinatorial ambiguity, and every observable in the
list is a function of the two pair four-momenta only, which are the two ``Z``
four-momenta by four-momentum conservation.  The one thing that is *not*
inherited is the mass: a reconstructed pair is off shell where a produced ``Z``
was not, so ``m_4l`` is not ``m(ZZ)`` event by event and ``beta`` computed from
it is not the produced ``beta``.  That is a real difference between the two
sides of the study, not an approximation inside either.
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import observables_zz as OZ                                      # noqa: E402

mass = OZ.mass
pt = OZ.pt
boost_to_rest = OZ.boost_to_rest
histogram = OZ.histogram
read_lhe_zz = OZ.read_lhe_zz
M_Z = OZ.M_Z

# The beam energy the samples were generated with, needed to build the
# Collins-Soper axis.  Read out of meta.json by the harvester rather than
# trusted from here; this is only the default.
EBEAM = 6500.0

# The top-quark threshold the fine m(ZZ) binning is there to resolve.  The value
# is the one in the param card the samples were generated with (SM default
# MT = 173 GeV), so the threshold sits at 346 GeV.
M_TOP = 173.0
TWO_MT = 2 * M_TOP

# The m(ZZ) slice boundaries of the three sliced angular observables.
M_SLICES = (300.0, 450.0)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def rapidity(v):
    num = v[:, 0] + v[:, 3]
    den = v[:, 0] - v[:, 3]
    return 0.5 * np.log(np.clip(num, 1e-12, None) / np.clip(den, 1e-12, None))


def _unit(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.where(n > 0, n, 1.0)


def cos_theta_cs(a, b, ebeam=EBEAM):
    """``cos theta*`` of ``a`` in the Collins-Soper frame of the ``(a, b)`` system.

    Built explicitly rather than through the massless shortcut: the two beams
    are boosted into the ``(a+b)`` rest frame, the polar axis is the bisector
    of the first beam and the *reversed* second beam, and the angle is measured
    against it.  Signed by the rapidity of the pair, which is the usual
    stand-in for the quark direction; every use here takes the modulus, so the
    sign convention only has to be consistent, not right.
    """
    par = a + b
    n = len(par)
    b1 = np.tile(np.array([ebeam, 0.0, 0.0, ebeam]), (n, 1))
    b2 = np.tile(np.array([ebeam, 0.0, 0.0, -ebeam]), (n, 1))
    d1 = _unit(boost_to_rest(b1, par)[:, 1:4])
    d2 = _unit(boost_to_rest(b2, par)[:, 1:4])
    axis = _unit(d1 - d2)
    pa = _unit(boost_to_rest(a, par)[:, 1:4])
    cs = np.clip(np.sum(axis * pa, axis=1), -1.0, 1.0)
    return cs * np.where(rapidity(par) >= 0.0, 1.0, -1.0)


def mzz_reweight(m, w, target_m, target_w, edges):
    """Per-event weights that force ``m``'s spectrum onto ``target_m``'s.

    Normalised so the total weight is unchanged, so only the *shape* moves.
    Bins the target does not populate get zero, which is why ``edges`` has to
    reach beyond the samples: a bin that only one side reaches would otherwise
    quietly delete events instead of reweighting them.
    """
    src, _ = np.histogram(m, bins=edges, weights=w)
    tgt, _ = np.histogram(target_m, bins=edges, weights=target_w)
    src = src / w.sum()
    tgt = tgt / target_w.sum()
    fac = np.where(src != 0.0, tgt / np.where(src != 0.0, src, 1.0), 0.0)
    idx = np.clip(np.digitize(m, edges) - 1, 0, len(edges) - 2)
    return w * fac[idx]


REWEIGHT_EDGES = np.concatenate([np.arange(180.0, 400.0, 10.0),
                                 np.arange(400.0, 600.0, 25.0),
                                 np.array([600.0, 700.0, 800.0, 1000.0,
                                           1400.0, 14000.0])])


# --------------------------------------------------------------------------
# the production-level observables
# --------------------------------------------------------------------------
def compute_shapes(z1, z2, ebeam=EBEAM):
    """Dict of arrays, from the two ``z`` four-vectors of a production event."""
    zz = z1 + z2
    p1, p2 = pt(z1), pt(z2)
    harder = (p1 >= p2)
    hard = np.where(harder[:, None], z1, z2)
    soft = np.where(harder[:, None], z2, z1)
    m = mass(zz)
    y1, y2 = rapidity(z1), rapidity(z2)
    acs = np.abs(cos_theta_cs(hard, soft, ebeam))
    lead = np.where(harder, p1, p2)

    out = {
        'm_zz_fine': m,
        'abs_cos_theta_cs': acs,
        'dy_zz': np.abs(y1 - y2),
        'abs_y_z_lead': np.abs(np.where(harder, y1, y2)),
        'max_abs_y_z': np.maximum(np.abs(y1), np.abs(y2)),
        'pt_over_m': lead / np.clip(m, 1e-9, None),
    }
    lo, hi = M_SLICES
    out['abs_cos_star_mlow'] = np.where(m < lo, acs, np.nan)
    out['abs_cos_star_mmid'] = np.where((m >= lo) & (m < hi), acs, np.nan)
    out['abs_cos_star_mhigh'] = np.where(m >= hi, acs, np.nan)
    return out


_COS = np.linspace(0.0, 1.0, 11)
_COS_COARSE = np.linspace(0.0, 1.0, 9)

BINS_SHAPES = {
    # 6 GeV through the whole 2 m_t region, and STOPPING at 604 GeV.  This is
    # not the parent study's inclusive m(ZZ) spectrum in finer bins -- that one
    # already exists and runs to 1.4 TeV -- it is a magnifying glass on the top
    # threshold, and carrying the four-decade tail alongside would compress the
    # region the observable exists to resolve into a fifth of the axis.  The
    # first edge is the on-shell threshold 2 m_Z = 182.376 GeV, which all three
    # production samples respect exactly.
    'm_zz_fine': np.concatenate([np.array([182.376, 190.0]),
                                 np.arange(196.0, 460.0, 6.0),
                                 np.arange(460.0, 616.0, 12.0)]),
    'abs_cos_theta_cs': _COS,
    'abs_cos_star_mlow': _COS,
    'abs_cos_star_mmid': _COS,
    'abs_cos_star_mhigh': _COS_COARSE,
    'dy_zz': np.concatenate([np.linspace(0.0, 3.0, 16),
                             np.array([3.4, 3.8, 4.4, 5.2, 6.5])]),
    'abs_y_z_lead': np.linspace(0.0, 4.0, 21),
    'max_abs_y_z': np.linspace(0.0, 4.5, 19),
    # bounded at 0.5 on a 2 -> 2 sample; the NLO events above that (3.7 % of
    # its weight) fall outside the axis on purpose.  See the module docstring.
    'pt_over_m': np.linspace(0.0, 0.5, 21),
}

# Drawn on a log y axis in the MAIN pane.  The two high-mass angular slices are
# in here and the low-mass one is not, for a reason that is the physics of the
# figure rather than taste: the forward peak of the q q~ tree sharpens with
# m(ZZ), so above 450 GeV the LO curve spans two decades across the pane and a
# linear axis shows one bin and nine flat lines.  Below 300 GeV it spans a
# factor of three and linear is the honest rendering.
LOGY_SHAPES = {'m_zz_fine', 'abs_cos_star_mmid', 'abs_cos_star_mhigh'}

LABELS_SHAPES = {
    'm_zz_fine': (r'$m(ZZ)$ [GeV]  (fine, through $2m_t$)',
                  r'$\mathrm{d}\sigma/\mathrm{d}m(ZZ)$ [pb/GeV]'),
    'abs_cos_theta_cs': (r'$|\cos\theta^{*}_{\mathrm{CS}}|$ of the harder $Z$',
                         r'$\mathrm{d}\sigma/\mathrm{d}|\cos\theta^{*}_{\mathrm{CS}}|$ [pb]'),
    'abs_cos_star_mlow': (r'$|\cos\theta^{*}_{\mathrm{CS}}|$, $m(ZZ) < 300$ GeV',
                          r'$\mathrm{d}\sigma/\mathrm{d}|\cos\theta^{*}_{\mathrm{CS}}|$ [pb]'),
    'abs_cos_star_mmid': (r'$|\cos\theta^{*}_{\mathrm{CS}}|$, $300 \leq m(ZZ) < 450$ GeV',
                          r'$\mathrm{d}\sigma/\mathrm{d}|\cos\theta^{*}_{\mathrm{CS}}|$ [pb]'),
    'abs_cos_star_mhigh': (r'$|\cos\theta^{*}_{\mathrm{CS}}|$, $m(ZZ) \geq 450$ GeV',
                           r'$\mathrm{d}\sigma/\mathrm{d}|\cos\theta^{*}_{\mathrm{CS}}|$ [pb]'),
    'dy_zz': (r'$|\Delta y(Z,Z)|$',
              r'$\mathrm{d}\sigma/\mathrm{d}|\Delta y|$ [pb]'),
    'abs_y_z_lead': (r'$|y|$ of the harder $Z$',
                     r'$\mathrm{d}\sigma/\mathrm{d}|y(Z_{\mathrm{lead}})|$ [pb]'),
    'max_abs_y_z': (r'$\max(|y(Z_1)|, |y(Z_2)|)$',
                    r'$\mathrm{d}\sigma/\mathrm{d}\max|y(Z)|$ [pb]'),
    'pt_over_m': (r'$p_{T}(Z_{\mathrm{lead}}) / m(ZZ)$',
                  r'$\mathrm{d}\sigma/\mathrm{d}(p_{T}/m)$ [pb]'),
}

LABELS_SHAPES_TXT = {
    'm_zz_fine': 'm(ZZ) [GeV], fine binning through 2 m_t',
    'abs_cos_theta_cs': '|cos(theta*)| of the harder Z, Collins-Soper frame',
    'abs_cos_star_mlow': '|cos(theta*_CS)|, m(ZZ) < 300 GeV',
    'abs_cos_star_mmid': '|cos(theta*_CS)|, 300 <= m(ZZ) < 450 GeV',
    'abs_cos_star_mhigh': '|cos(theta*_CS)|, m(ZZ) >= 450 GeV',
    'dy_zz': '|Delta y(Z, Z)|',
    'abs_y_z_lead': '|y| of the harder Z',
    'max_abs_y_z': 'max(|y(Z1)|, |y(Z2)|)',
    'pt_over_m': 'pt(Z_lead) / m(ZZ)',
}

# The order the ranking table and the figure loop use.
SHAPE_OBS = ('abs_cos_star_mhigh', 'abs_cos_star_mmid', 'abs_cos_star_mlow',
             'abs_cos_theta_cs', 'dy_zz', 'max_abs_y_z', 'abs_y_z_lead',
             'pt_over_m', 'm_zz_fine')


# --------------------------------------------------------------------------
# the four-lepton twins
# --------------------------------------------------------------------------
def compute_shapes_4l(p, ebeam=EBEAM):
    """The same observables, from the four leptons of a decayed event.

    ``p`` is the dict returned by ``observables.read_lhe``.  The two ``Z`` are
    the reconstructed ``(e+ e-)`` and ``(mu+ mu-)`` pairs; see the module
    docstring for why that reconstruction is exact.
    """
    ep, em, mup, mum = p[-11], p[11], p[-13], p[13]
    zee = ep + em
    zmm = mup + mum
    four = zee + zmm
    p1, p2 = pt(zee), pt(zmm)
    harder = (p1 >= p2)
    hard = np.where(harder[:, None], zee, zmm)
    soft = np.where(harder[:, None], zmm, zee)
    m = mass(four)
    y1, y2 = rapidity(zee), rapidity(zmm)
    acs = np.abs(cos_theta_cs(hard, soft, ebeam))

    out = {
        'm_4l_fine': m,
        'abs_cos_theta_cs_4l': acs,
        'dy_pairs': np.abs(y1 - y2),
        'max_abs_y_pair': np.maximum(np.abs(y1), np.abs(y2)),
        'pt_over_m_4l': np.where(harder, p1, p2) / np.clip(m, 1e-9, None),
        'min_m_ll': np.minimum(mass(zee), mass(zmm)),
    }
    lo, hi = M_SLICES
    out['abs_cos_star_4l_mlow'] = np.where(m < lo, acs, np.nan)
    out['abs_cos_star_4l_mhigh'] = np.where(m >= hi, acs, np.nan)
    return out


BINS_SHAPES_4L = {
    # An edge sits EXACTLY on 2 m_Z = 182.376 GeV, and the grid below it is
    # built downwards from there in 4 GeV steps.  That is not cosmetic: the
    # sub-threshold rate is the discriminating quantity, it rises steeply
    # towards the threshold, and a grid whose bin straddles 2 m_Z puts part of
    # the region on the wrong side of the count.  With the edge in place the
    # integral of the drawn curve below threshold reproduces the per-event
    # fraction in meta.json to five digits; with a straddling edge it came out
    # a factor of 1.5 to 2 low.
    #
    # The grid reaches 106.4 GeV, under the kinematic floor 2 M_LO = 109.13 GeV
    # that the 15-width Breit-Wigner cut imposes on the two pairs, so nothing
    # falls off the left; the parent study's m_4l grid starts at 150 GeV and its
    # figure therefore shows only part of the effect (its quoted fractions are
    # per-event counts and are unaffected).  Above the threshold it coarsens and
    # stops at 500: the high tail is the parent figure's job and every mode
    # agrees with the truth there to within statistics.
    'm_4l_fine': np.concatenate([2 * OZ.M_Z - 4.0 * np.arange(19, -1, -1),
                                 np.arange(190.0, 320.0, 8.0),
                                 np.arange(320.0, 520.0, 20.0)]),
    'abs_cos_theta_cs_4l': _COS,
    'abs_cos_star_4l_mlow': _COS,
    'abs_cos_star_4l_mhigh': _COS_COARSE,
    'dy_pairs': np.concatenate([np.linspace(0.0, 3.0, 16),
                                np.array([3.4, 3.8, 4.4, 5.2, 6.5])]),
    'max_abs_y_pair': np.linspace(0.0, 4.5, 19),
    'pt_over_m_4l': np.linspace(0.0, 0.6, 25),
    # 75 bins, ODD on purpose.  m_Z is exactly the midpoint of the BW window
    # (M_LO and M_HI are m_Z -+ 15 Gamma_Z), so an even number of bins puts it
    # on an edge and splits the delta function of an on-shell mode across two
    # bins -- which then looks like a two-bin distribution with a shape, and a
    # flat-line chi2 computed on it returns a large number that is pure binning
    # noise.  An odd count puts m_Z mid-bin, where a delta function belongs.
    # This is also the parent study's m_ee / m_mumu binning.
    'min_m_ll': np.linspace(OZ.LEP.M_LO, OZ.LEP.M_HI, 76),
}

LOGY_SHAPES_4L = {'m_4l_fine', 'min_m_ll'}

LABELS_SHAPES_4L = {
    'm_4l_fine': (r'$m_{4\ell}$ [GeV]  (fine, through $2m_Z$)',
                  r'$\mathrm{d}\sigma/\mathrm{d}m_{4\ell}$ [pb/GeV]'),
    'abs_cos_theta_cs_4l': (r'$|\cos\theta^{*}_{\mathrm{CS}}|$ of the harder pair',
                            r'$\mathrm{d}\sigma/\mathrm{d}|\cos\theta^{*}_{\mathrm{CS}}|$ [pb]'),
    'abs_cos_star_4l_mlow': (r'$|\cos\theta^{*}_{\mathrm{CS}}|$, $m_{4\ell} < 300$ GeV',
                             r'$\mathrm{d}\sigma/\mathrm{d}|\cos\theta^{*}_{\mathrm{CS}}|$ [pb]'),
    'abs_cos_star_4l_mhigh': (r'$|\cos\theta^{*}_{\mathrm{CS}}|$, $m_{4\ell} \geq 450$ GeV',
                              r'$\mathrm{d}\sigma/\mathrm{d}|\cos\theta^{*}_{\mathrm{CS}}|$ [pb]'),
    'dy_pairs': (r'$|\Delta y(e^{+}e^{-}, \mu^{+}\mu^{-})|$',
                 r'$\mathrm{d}\sigma/\mathrm{d}|\Delta y|$ [pb]'),
    'max_abs_y_pair': (r'$\max(|y(e^{+}e^{-})|, |y(\mu^{+}\mu^{-})|)$',
                       r'$\mathrm{d}\sigma/\mathrm{d}\max|y|$ [pb]'),
    'pt_over_m_4l': (r'$p_{T}$ of the harder pair $/ m_{4\ell}$',
                     r'$\mathrm{d}\sigma/\mathrm{d}(p_{T}/m_{4\ell})$ [pb]'),
    'min_m_ll': (r'$\min\left(m(e^{+}e^{-}), m(\mu^{+}\mu^{-})\right)$ [GeV]',
                 r'$\mathrm{d}\sigma/\mathrm{d}m_{\min}$ [pb/GeV]'),
}

LABELS_SHAPES_4L_TXT = {
    'm_4l_fine': 'm_4l [GeV], fine binning through 2 m_Z',
    'abs_cos_theta_cs_4l': '|cos(theta*_CS)| of the harder lepton pair',
    'abs_cos_star_4l_mlow': '|cos(theta*_CS)|, m_4l < 300 GeV',
    'abs_cos_star_4l_mhigh': '|cos(theta*_CS)|, m_4l >= 450 GeV',
    'dy_pairs': '|Delta y(e+e-, mu+mu-)|',
    'max_abs_y_pair': 'max(|y(e+e-)|, |y(mu+mu-)|)',
    'pt_over_m_4l': 'pt of the harder pair / m_4l',
    'min_m_ll': 'min(m(e+e-), m(mu+mu-)) [GeV]',
}

SHAPE_OBS_4L = ('m_4l_fine', 'min_m_ll', 'abs_cos_theta_cs_4l',
                'abs_cos_star_4l_mlow', 'abs_cos_star_4l_mhigh',
                'dy_pairs', 'max_abs_y_pair', 'pt_over_m_4l')

# The four-lepton observables that are a delta function for a mode drawing no
# virtuality.  ``min_m_ll`` is the new one: ``onshell`` / ``none`` /
# ``onshell_v1`` put both pair masses exactly at ``m_Z``, so the minimum of the
# two is exactly ``m_Z`` as well and every other bin of the window is empty by
# construction rather than by sample size.
PAIR_MASS_OBS_4L = ('min_m_ll',)


# --------------------------------------------------------------------------
# the discrimination statistic
# --------------------------------------------------------------------------
def ratio_with_errors(num, nerr, den, derr):
    """``num/den`` and its error, NaN where either side is empty."""
    ok = (den > 0) & (num != 0)
    r = np.full_like(num, np.nan, dtype=float)
    e = np.full_like(num, np.nan, dtype=float)
    r[ok] = num[ok] / den[ok]
    e[ok] = np.abs(r[ok]) * np.sqrt((nerr[ok] / num[ok]) ** 2
                                    + (derr[ok] / den[ok]) ** 2)
    return r, e


def chi2_flat(r, e):
    """``(chi2/ndf, ndf, best-fit level)`` of a ratio against a flat line.

    A pure normalisation offset does not enter -- the level is fitted -- so
    only a genuine shape difference is measured.  Same statistic the parent
    study uses for the spinmode comparison, so the two sets of numbers are on
    the same scale.
    """
    ok = np.isfinite(r) & np.isfinite(e) & (e > 0)
    if ok.sum() < 2:
        return None, 0, float('nan')
    w = 1.0 / e[ok] ** 2
    mu = float(np.sum(w * r[ok]) / np.sum(w))
    return (float(np.sum(w * (r[ok] - mu) ** 2) / (ok.sum() - 1)),
            int(ok.sum() - 1), mu)


def spread(r, level, keep=None):
    """``(min, max)`` of ``r/level`` over the bins that carry rate.

    The ratio of the two is the "factor across the range" quoted in the
    ranking: 1.0 means the two normalised ratio shapes are identical.
    """
    v = r / level
    if keep is not None:
        v = np.where(keep, v, np.nan)
    v = v[np.isfinite(v)]
    if not len(v):
        return float('nan'), float('nan')
    return float(v.min()), float(v.max())


def step_fit(x, r, e, x0):
    """Least squares ``r = a + b (x - x0) + c theta(x - x0)``.

    The top-threshold test: ``c`` is the size of a step at ``x0 = 2 m_t`` on
    top of whatever smooth slope the ratio already has, and its error is the
    sensitivity of these samples to one.  Returns
    ``(a, b, c, sigma_c, chi2_ndf_with, chi2_ndf_without)``.
    """
    ok = np.isfinite(r) & np.isfinite(e) & (e > 0)
    xx, rr, ee = x[ok], r[ok], e[ok]
    if len(xx) < 5:
        return None
    wm = np.diag(1.0 / ee ** 2)
    a3 = np.column_stack([np.ones(len(xx)), xx - x0,
                          (xx > x0).astype(float)])
    c3 = np.linalg.inv(a3.T @ wm @ a3)
    b3 = c3 @ (a3.T @ wm @ rr)
    res3 = rr - a3 @ b3
    a2 = np.column_stack([np.ones(len(xx)), xx - x0])
    c2 = np.linalg.inv(a2.T @ wm @ a2)
    b2 = c2 @ (a2.T @ wm @ rr)
    res2 = rr - a2 @ b2
    return (float(b3[0]), float(b3[1]), float(b3[2]),
            float(math.sqrt(c3[2, 2])),
            float((res3 ** 2 / ee ** 2).sum() / (len(xx) - 3)),
            float((res2 ** 2 / ee ** 2).sum() / (len(xx) - 2)))
