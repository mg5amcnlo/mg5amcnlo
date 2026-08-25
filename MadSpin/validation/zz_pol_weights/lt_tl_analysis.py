#!/usr/bin/env python3
"""Is the ``Z_0Z_T`` / ``Z_TZ_0`` K-factor split physics, or an artefact?

The K-factor panel of ``plots*/kfactor_LO_NLO/dphi_ee.*`` shows
``K(Z_TZ_0) = 2.94`` in the lowest ``Delta phi(e+e-)`` bin against
``K(Z_0Z_T) = 1.47``, while the two components' INTEGRATED K-factors agree to
0.0007 on 0.0063 bars.  This script decides whether that split is real, using
the one discriminator that separates the two possibilities, and it reads
nothing but the two cached ``.npz``.

THE DISCRIMINATOR.  ``ZZ`` is symmetric under exchanging the two ``Z``, and
with ``decay z > e+ e-`` listed FIRST in the MadSpin card the first weight
index is the electron-``Z`` (:func:`mapping_test` verifies that from the
kinematics rather than from the card).  So:

* ``M(e+ mu+)`` takes one lepton from each ``Z`` and is EXACTLY invariant under
  ``e <-> mu``, as is its four-lepton selection.  That exchange maps
  ``Z_0Z_T`` onto ``Z_TZ_0``, so on this observable the two components must
  agree bin by bin.  A disagreement here would be a bug.
* ``Delta phi(e+e-)`` is built from the electron-``Z`` alone and is blind to
  the muon-``Z``.  It is not invariant under the exchange, so the two
  components are under no obligation to agree, and a difference here can be
  physics.

Agreement on the first plus disagreement on the second is the signature of real
physics.  Disagreement on BOTH would point at the weight-index-to-``Z``-identity
mapping.  :func:`symmetry_test` measures both.

THE THREE TESTS, and what each rules out:

1. :func:`mapping_test` -- ``<cos^2 theta*>`` of the ``e+`` in the ``e+e-``
   rest frame and of the ``mu+`` in the ``mu+mu-`` rest frame, weighted by each
   polarisation column.  A longitudinal ``Z`` decays as ``sin^2 theta*``
   (``<cos^2> = 1/5``), a transverse one as ``1 + cos^2 theta*``
   (``<cos^2> = 2/5``).  This says which physical ``Z`` each weight index
   refers to, measured, and it does not consult the card.
2. :func:`symmetry_test` -- ``TL/LT`` bin by bin on both observables, in both
   samples, with the delta-method error (the two sums run over the SAME events,
   so the covariance must be kept), and ``K(TL)/K(LT)`` bin by bin, with the
   two independent samples' errors in quadrature.
3. :func:`mirror_test` -- the sharpest one.  ``Delta phi(mu+mu-)`` is the
   ``e <-> mu`` image of ``Delta phi(e+e-)``, so ``(TL/LT)`` measured on it
   must equal ``1/(TL/LT)`` measured on ``Delta phi(e+e-)``, bin by bin, across
   the whole factor-of-three range it spans.  Nothing in the weights enforces
   that; if it holds, the split is the physics of which ``Z`` is longitudinal
   and not a mislabelled column.

``Delta phi(mu+mu-)`` is not a cached column.  It is reconstructed from
``m_mumu`` and the muon ``pT`` / ``eta``, which determine it exactly for
massless leptons -- ``m^2 = 2 pT1 pT2 (cosh(d eta) - cos(d phi))``.  The same
identity run backwards on the ELECTRONS, where ``Delta phi`` IS cached,
reproduces ``m_ee`` to 3e-4 GeV, which is :func:`_closure` and is printed.

Usage::

    python3 lt_tl_analysis.py [--data DIR] [--out DIR] [--userstyle-out DIR]
                              [--no-figures] [--check-minus]

Writes ``plots/lt_tl/`` and ``plots_userstyle/lt_tl/`` (PDF and PNG, both
observables' panes on one canvas) and the numbers to stdout.  It touches no
existing figure and no existing module.
"""

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import pol_analysis as PA                                        # noqa: E402

OUT_DIR = 'lt_tl'

# The two ideal moments the mapping test is read against.  For Z -> l+ l- the
# helicity-0 angular distribution is ``3/4 sin^2 theta*`` and the sum over the
# two transverse helicities is ``3/8 (1 + cos^2 theta*)``, giving
# <cos^2> = 1/5 and 2/5.  The MEASURED values are diluted towards each other by
# the fiducial cuts, by the ZZ-frame quantisation axis (``frame_id 24``) not
# being the lab Z direction used here, and by the T column summing the two
# transverse helicities coherently.  The test is the CONTRAST and the 2 x 2
# pattern, not the absolute value.
IDEAL = {'L': 0.2, 'T': 0.4}

COS_EDGES = np.linspace(-1.0, 1.0, 21)


# --------------------------------------------------------------------------
# kinematics rebuilt from the cached columns
# --------------------------------------------------------------------------
def _dphi_mumu(d):
    """``Delta phi(mu+ mu-)``, exact for massless leptons, from ``m_mumu``.

    ``m^2 = 2 pT1 pT2 (cosh(d eta) - cos(d phi))`` inverted.  Only ``|d phi|``
    comes back, which is all any of the observables here use.
    """
    pt1 = np.asarray(d.z['pt_mup'], float)
    pt2 = np.asarray(d.z['pt_mum'], float)
    e1 = np.asarray(d.z['eta_mup'], float)
    e2 = np.asarray(d.z['eta_mum'], float)
    m = np.asarray(d.z['m_mumu'], float)
    with np.errstate(invalid='ignore', divide='ignore'):
        c = np.cosh(e1 - e2) - m * m / (2.0 * pt1 * pt2)
    return np.arccos(np.clip(c, -1.0, 1.0))


def _closure(d):
    """``max |m(ee) rebuilt - m_ee cached|`` -- the identity, run forwards."""
    pt1 = np.asarray(d.z['pt_ep'], float)
    pt2 = np.asarray(d.z['pt_em'], float)
    e1 = np.asarray(d.z['eta_ep'], float)
    e2 = np.asarray(d.z['eta_em'], float)
    dp = np.asarray(d.z['dphi_ee'], float)
    m2 = 2.0 * pt1 * pt2 * (np.cosh(e1 - e2) - np.cos(dp))
    m = np.sqrt(np.maximum(m2, 0.0))
    return float(np.nanmax(np.abs(m - np.asarray(d.z['m_ee'], float))))


def _costhetastar(pt1, eta1, pt2, eta2, dphi):
    """``cos theta*`` of particle 1 in the ``(1+2)`` rest frame.

    Measured against the ``(1+2)`` direction in the lab -- the helicity frame.
    Both particles massless.  Only the RELATIVE azimuth enters, so the absent
    absolute azimuth of the cached columns costs nothing.
    """
    p1 = np.stack([pt1 * np.cosh(eta1), pt1,
                   np.zeros_like(pt1), pt1 * np.sinh(eta1)])
    p2 = np.stack([pt2 * np.cosh(eta2), pt2 * np.cos(dphi),
                   pt2 * np.sin(dphi), pt2 * np.sinh(eta2)])
    P = p1 + p2
    b = P[1:] / P[0]
    b2 = (b ** 2).sum(0)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-15))
    bp = (b * p1[1:]).sum(0)
    p1s = p1[1:] + b * (((g - 1.0) / np.maximum(b2, 1e-30)) * bp - g * p1[0])
    n = P[1:] / np.sqrt(np.maximum((P[1:] ** 2).sum(0), 1e-30))
    return ((p1s * n).sum(0)
            / np.sqrt(np.maximum((p1s ** 2).sum(0), 1e-30)))


def cos_star(d):
    """``(electron-side, muon-side)`` ``cos theta*`` for every event."""
    g = lambda k: np.asarray(d.z[k], float)                      # noqa: E731
    ce = _costhetastar(g('pt_ep'), g('eta_ep'), g('pt_em'), g('eta_em'),
                       g('dphi_ee'))
    cm = _costhetastar(g('pt_mup'), g('eta_mup'), g('pt_mum'), g('eta_mum'),
                       _dphi_mumu(d))
    return ce, cm


def sel_mumu(d):
    """The ``Delta phi(mu+mu-)`` selection: the two MUONS, same cuts."""
    ok = np.isfinite(_dphi_mumu(d))
    for nm in ('mup', 'mum'):
        a, b = PA._lep_ok(d, nm)
        ok &= a & b
    return ok


def sel_4l(d):
    """All four leptons, which is what ``M(e+ mu+)`` already requires."""
    return d.sel['m_epmup_dr']


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def wmean_err(w, v):
    """``sum(w v)/sum(w)`` with the delta-method bar -- same events, so the
    covariance between numerator and denominator is kept exactly as
    :func:`pol_analysis.ratio` does it."""
    D = w.sum()
    if D == 0:
        return float('nan'), float('nan')
    R = float(np.sum(w * v) / D)
    return R, float(math.sqrt(np.sum((w * v - R * w) ** 2)) / abs(D))


def binned_delta_ratio(x, num, den, edges):
    """Per-bin ``num/den`` with the delta-method error and the populations.

    ``pol_analysis.binned_ratio`` in everything but the return shape; kept
    local so this script imports no private helper it would then pin.
    """
    idx = np.digitize(x, edges) - 1
    nb = len(edges) - 1
    r = np.full(nb, np.nan)
    e = np.full(nb, np.nan)
    n = np.zeros(nb, dtype=int)
    neff_d = np.zeros(nb, dtype=int)
    neff_n = np.zeros(nb, dtype=int)
    for b in range(nb):
        m = idx == b
        n[b] = int(m.sum())
        D = den[m].sum()
        if n[b] and D != 0:
            R = num[m].sum() / D
            r[b] = R
            e[b] = math.sqrt(np.sum((num[m] - R * den[m]) ** 2)) / abs(D)
            # The EFFECTIVE population of a weighted sum, (sum w)^2 / sum w^2.
            # A bin's raw event count overstates what a polarisation column can
            # support there: the column is a fraction of each event's weight
            # and is far from uniform across the events in the bin.
            for arr, out in ((den[m], neff_d), (num[m], neff_n)):
                s2 = np.sum(arr ** 2)
                out[b] = int(round(arr.sum() ** 2 / s2)) if s2 > 0 else 0
    return {'r': r, 'err': e, 'n': n, 'neff': neff_d, 'neff_num': neff_n}


def chi2_vs(r, e, ref):
    ok = np.isfinite(r) & np.isfinite(e) & (e > 0)
    return float(np.sum(((r[ok] - ref) / e[ok]) ** 2)), int(ok.sum())


def best_constant(r, e):
    ok = np.isfinite(r) & np.isfinite(e) & (e > 0)
    w = 1.0 / e[ok] ** 2
    c = float(np.sum(w * r[ok]) / np.sum(w))
    return c, float(1.0 / math.sqrt(np.sum(w)))


# --------------------------------------------------------------------------
# the three tests
# --------------------------------------------------------------------------
def mapping_test(d):
    """Which physical ``Z`` does each weight INDEX refer to?  Measured.

    Returns ``<cos^2 theta*>`` on the electron side and on the muon side under
    each polarisation column.  If the first index is the electron-``Z``, then
    under ``LT`` the electron side must be the LOW (longitudinal) one and the
    muon side the HIGH (transverse) one, and under ``TL`` the reverse; ``LL``
    must be low on both and ``TT`` high on both.  A transposed mapping would
    show the ``LT`` and ``TL`` rows swapped.
    """
    ce, cm = cos_star(d)
    sel = sel_4l(d) & np.isfinite(ce) & np.isfinite(cm)
    out = {'label': d.label, 'n_sel': int(sel.sum()), 'rows': [],
           'shape': {}}
    for key in ['full'] + PA.POL_KEYS:
        w = np.asarray(d.full if key == 'full' else d.pol[key],
                       float)[sel]
        me, ee = wmean_err(w, ce[sel] ** 2)
        mm, em = wmean_err(w, cm[sel] ** 2)
        out['rows'].append({'key': key, 'e': me, 'e_err': ee,
                            'mu': mm, 'mu_err': em})
        he, _ = np.histogram(ce[sel], bins=COS_EDGES, weights=w)
        hm, _ = np.histogram(cm[sel], bins=COS_EDGES, weights=w)
        out['shape'][key] = {'e': he / max(he.sum(), 1e-300) / np.diff(COS_EDGES),
                             'mu': hm / max(hm.sum(), 1e-300) / np.diff(COS_EDGES)}
    return out


def symmetry_test(nlo, lo, obs):
    """``TL/LT`` per bin in each sample, and ``K(TL)/K(LT)`` per bin.

    ERRORS.  ``TL/LT`` within one sample divides two weight columns of the SAME
    events, so it takes the delta-method bar.  ``K(TL)/K(LT)`` is a ratio
    across ``run_06`` and ``run_12`` -- two independent generations, no shared
    event -- so it takes the two within-sample bars in quadrature, which is the
    same rule ``pol_analysis.kfactor`` follows for its own numerator and
    denominator.
    """
    edges = PA.BINS[obs]
    out = {'obs': obs, 'edges': edges, 'within': {}}
    for d, tag in ((nlo, 'NLO'), (lo, 'LO')):
        sel = d.sel[obs]
        x = np.asarray(d.z[obs], float)[sel]
        num = np.asarray(d.pol['TL'], float)[sel]
        den = np.asarray(d.pol['LT'], float)[sel]
        w = binned_delta_ratio(x, num, den, edges)
        w['chi2'], w['ndf'] = chi2_vs(w['r'], w['err'], 1.0)
        w['sigma_TL'] = float(num.sum() * d.scale_to_pb)
        w['sigma_LT'] = float(den.sum() * d.scale_to_pb)
        R, E = PA.ratio(num, den)
        w['integrated'], w['integrated_err'] = R, E
        out['within'][tag] = w
    a, b = out['within']['NLO'], out['within']['LO']
    with np.errstate(invalid='ignore', divide='ignore'):
        dr = a['r'] / b['r']
        de = np.abs(dr) * np.sqrt((a['err'] / a['r']) ** 2
                                  + (b['err'] / b['r']) ** 2)
    out['double'] = {'r': dr, 'err': de}
    out['double']['chi2'], out['double']['ndf'] = chi2_vs(dr, de, 1.0)
    c, ce_ = best_constant(dr, de)
    out['double']['const'], out['double']['const_err'] = c, ce_
    out['double']['chi2_const'] = chi2_vs(dr, de, c)[0]
    # K of each component separately, which is what the K panel draws.
    out['kLT'] = PA.kfactor(nlo, lo, obs, 'LT')
    out['kTL'] = PA.kfactor(nlo, lo, obs, 'TL')
    out['kfull'] = PA.kfactor(nlo, lo, obs, 'full')
    return out


def mirror_test(d):
    """``(TL/LT)`` on ``Delta phi(mu+mu-)`` against ``1/(TL/LT)`` on
    ``Delta phi(e+e-)``, bin by bin.

    The ``e <-> mu`` image of the electron measurement.  Nothing in the weight
    columns enforces this; it can only come out right if the first index really
    is the electron-``Z`` and the split really is the physics of which ``Z`` is
    longitudinal.  The two sides are NOT the same event set -- each keeps the
    two leptons its own observable is built from -- so the comparison is a
    physics one and not an algebraic identity.
    """
    edges = PA.BINS['dphi_ee_dr']
    se = d.sel['dphi_ee_dr']
    ee = binned_delta_ratio(np.asarray(d.z['dphi_ee_dr'], float)[se],
                            np.asarray(d.pol['TL'], float)[se],
                            np.asarray(d.pol['LT'], float)[se], edges)
    sm = sel_mumu(d)
    mm = binned_delta_ratio(_dphi_mumu(d)[sm],
                            np.asarray(d.pol['TL'], float)[sm],
                            np.asarray(d.pol['LT'], float)[sm], edges)
    with np.errstate(invalid='ignore', divide='ignore'):
        pred = 1.0 / ee['r']
        pred_err = ee['err'] / ee['r'] ** 2
        tot = np.sqrt(mm['err'] ** 2 + pred_err ** 2)
        pull = (mm['r'] - pred) / tot
    ok = np.isfinite(pull)
    return {'label': d.label, 'edges': edges, 'ee': ee, 'mumu': mm,
            'pred': pred, 'pred_err': pred_err,
            'chi2': float(np.sum(pull[ok] ** 2)), 'ndf': int(ok.sum()),
            'n_ee': int(se.sum()), 'n_mumu': int(sm.sum())}


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------
def report(nlo, lo, stream=sys.stdout):
    P = lambda s='': stream.write(s + '\n')                       # noqa: E731
    P('=' * 78)
    P('LT / TL : is the K-factor split physics or an artefact?')
    P('=' * 78)
    P('closure of the massless-lepton identity on the ELECTRONS, where')
    P('Delta phi IS cached:  max |m(ee) rebuilt - m_ee| = %.2e GeV (NLO), '
      '%.2e GeV (LO)' % (_closure(nlo), _closure(lo)))
    P('so the same identity is trusted to give Delta phi(mu+mu-) from m_mumu.')
    P()

    P('-' * 78)
    P('TEST 1  the weight-index-to-Z mapping, measured from the kinematics')
    P('-' * 78)
    P('<cos^2 theta*> of the l+ in its parent rest frame, per weight column.')
    P('ideal: longitudinal Z -> %.3f,  transverse Z -> %.3f.  The measured'
      % (IDEAL['L'], IDEAL['T']))
    P('values are diluted towards each other (fiducial cuts, lab-Z helicity')
    P('axis rather than the ZZ-frame one, T summing both transverse')
    P('helicities); the test is the CONTRAST and the 2x2 pattern.')
    P()
    for d in (nlo, lo):
        m = mapping_test(d)
        P('%s  (%d events in the four-lepton fiducial)' % (m['label'], m['n_sel']))
        P('  %-6s %-22s %-22s %s' % ('column', 'electron side', 'muon side',
                                     'reading'))
        for r in m['rows']:
            if r['key'] == 'full':
                rd = '-'
            else:
                rd = '%s electron, %s muon' % (
                    'LONG' if r['e'] < 0.5 * (m['rows'][1]['e'] + m['rows'][4]['e'])
                    else 'TRANS',
                    'LONG' if r['mu'] < 0.5 * (m['rows'][1]['mu'] + m['rows'][4]['mu'])
                    else 'TRANS')
            P('  %-6s %.4f +- %.4f       %.4f +- %.4f       %s'
              % (r['key'], r['e'], r['e_err'], r['mu'], r['mu_err'], rd))
        P()

    P('-' * 78)
    P('TEST 2  the discriminator: TL/LT on the symmetric and the asymmetric')
    P('        observable')
    P('-' * 78)
    res = {}
    for obs in PA.OBS:
        s = symmetry_test(nlo, lo, obs)
        res[obs] = s
        e = s['edges']
        sym = 'M(e+ mu+)' if obs == 'm_epmup_dr' else 'Delta phi(e+e-)'
        P()
        P('%s   [%s under e <-> mu]'
          % (sym, 'INVARIANT' if obs == 'm_epmup_dr' else 'NOT invariant'))
        P('%-15s %7s %7s %7s | %-20s %-20s | %-20s'
          % ('bin', 'N_LO', 'NeffLT', 'NeffTL', 'TL/LT  (LO)', 'TL/LT  (NLO)',
             'K(TL)/K(LT)'))
        for b in range(len(e) - 1):
            lo_w, nl_w = s['within']['LO'], s['within']['NLO']
            dr, de = s['double']['r'][b], s['double']['err'][b]
            P('%6.2f - %6.2f %7d %7d %7d | %6.3f +- %5.3f %4.1fs | '
              '%6.3f +- %5.3f %4.1fs | %6.3f +- %5.3f %4.1fs'
              % (e[b], e[b + 1], lo_w['n'][b], lo_w['neff'][b],
                 lo_w['neff_num'][b],
                 lo_w['r'][b], lo_w['err'][b],
                 abs(lo_w['r'][b] - 1) / lo_w['err'][b],
                 nl_w['r'][b], nl_w['err'][b],
                 abs(nl_w['r'][b] - 1) / nl_w['err'][b],
                 dr, de, abs(dr - 1) / de))
        P('  chi2 against TL/LT = 1 :  LO %.1f/%d   NLO %.1f/%d'
          % (s['within']['LO']['chi2'], s['within']['LO']['ndf'],
             s['within']['NLO']['chi2'], s['within']['NLO']['ndf']))
        P('  chi2 of K(TL)/K(LT) against 1        : %.1f/%d'
          % (s['double']['chi2'], s['double']['ndf']))
        P('  ... and against its own best constant: %.1f/%d  (constant '
          '%.4f +- %.4f)'
          % (s['double']['chi2_const'], s['double']['ndf'] - 1,
             s['double']['const'], s['double']['const_err']))
        P('  integrated in this fiducial: TL/LT = %.4f +- %.4f (LO), '
          '%.4f +- %.4f (NLO)'
          % (s['within']['LO']['integrated'], s['within']['LO']['integrated_err'],
             s['within']['NLO']['integrated'], s['within']['NLO']['integrated_err']))
        P('  K(Z_0Z_T) = %.4f   K(Z_TZ_0) = %.4f   K(unpol) = %.4f'
          % (res[obs]['kLT']['k_integrated'], res[obs]['kTL']['k_integrated'],
             res[obs]['kfull']['k_integrated']))

    P()
    P('-' * 78)
    P('TEST 3  the mirror: Delta phi(mu+mu-) is the e <-> mu image of')
    P('        Delta phi(e+e-), so (TL/LT) on it must be 1/(TL/LT) on that one')
    P('-' * 78)
    for d in (nlo, lo):
        m = mirror_test(d)
        P()
        P('%s   (%d events pass the e+e- selection, %d the mu+mu- one)'
          % (m['label'], m['n_ee'], m['n_mumu']))
        P('%-15s %-14s %-14s %-22s %s'
          % ('bin', '(TL/LT)|ee', 'predicted', 'measured on mu+mu-', 'pull'))
        e = m['edges']
        for b in range(len(e) - 1):
            tot = math.sqrt(m['mumu']['err'][b] ** 2 + m['pred_err'][b] ** 2)
            P('%6.2f - %6.2f %8.4f       %8.4f       %8.4f +- %6.4f      %+5.1f'
              % (e[b], e[b + 1], m['ee']['r'][b], m['pred'][b],
                 m['mumu']['r'][b], m['mumu']['err'][b],
                 (m['mumu']['r'][b] - m['pred'][b]) / tot if tot else 0.0))
        P('  chi2 of measured against predicted: %.1f / %d'
          % (m['chi2'], m['ndf']))
    P()

    P('-' * 78)
    P('RECONCILIATION  the largest bin-by-bin departure of Z_TZ_0 from Z_0Z_T')
    P('        in the ABSOLUTE dsigma/dx the seventh (distribution) pane draws')
    P('-' * 78)
    P('%-14s %-6s %8s %8s %10s' % ('observable', 'order', 'min', 'max',
                                   'max |dev|'))
    for obs in PA.OBS:
        for d, tag in ((lo, 'LO'), (nlo, 'NLO')):
            a = np.asarray(PA.component_histogram(d, obs, 'LT')['y'], float)
            b = np.asarray(PA.component_histogram(d, obs, 'TL')['y'], float)
            with np.errstate(invalid='ignore', divide='ignore'):
                r = b / a
            P('%-14s %-6s %8.3f %8.3f %9.1f %%'
              % (PA.SHORT[obs], tag, np.nanmin(r), np.nanmax(r),
                 100 * np.nanmax(np.abs(r - 1))))
    P('"Z_0Z_T and Z_TZ_0 are equal to 1-2 % bin by bin" is a statement about')
    P('the M(e+ mu+) pane, where it holds.  It does NOT hold on the')
    P('Delta phi(e+e-) pane, and the reason given for it -- "they must be, ZZ')
    P('is symmetric" -- is only an argument for an observable that respects')
    P('the exchange, which Delta phi(e+e-) does not.')
    P()
    return res


# --------------------------------------------------------------------------
# figures -- one canvas, 3 x 2, in each of the two house styles
# --------------------------------------------------------------------------
def _panes(nlo, lo):
    """Everything the figure draws, computed once and shared by both styles."""
    mp = {d.label: mapping_test(d) for d in (nlo, lo)}
    sy = {obs: symmetry_test(nlo, lo, obs) for obs in PA.OBS}
    return mp, sy


def _obs_title(obs):
    return ('M(e+ mu+): symmetric under e <-> mu'
            if obs == 'm_epmup_dr'
            else 'Delta phi(e+e-): electron Z only')


def _obs_title_tex(obs):
    return (r'$M(e^{+}\mu^{+})$ --- symmetric under $e\leftrightarrow\mu$'
            if obs == 'm_epmup_dr'
            else r'$\Delta\phi(e^{+}e^{-})$ --- electron $Z$ only')


def draw(nlo, lo, outdir, style='mg7'):
    """The one figure.  ``style`` is ``'mg7'`` or ``'user'``.

    Eight panes, two columns.  In rows 2 and 3 the LEFT column is the
    exchange-symmetric observable and the RIGHT the asymmetric one, and both
    rows are the same quantity on both, so the figure is read across.

    Row 1 -- the mapping test.  It is a property of the WEIGHTS and not of an
    observable, so here the two columns are the two sides of the event: the
    electron ``Z`` on the left and the muon ``Z`` on the right.
    Row 2 -- ``TL/LT`` within each sample.
    Row 3 -- ``K(TL)/K(LT)``, which is the row the question was asked about.
    Row 4 -- the mirror test, one column per SAMPLE (LO left, NLO right),
    because it compares two observables rather than splitting on one.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if style == 'mg7':
        from plot_zz_pol import COLOR, LS_ORDER, LW, wants_minus
        tex = __import__('plot_zz_pol').USETEX
        figsize, ms, alpha = (9.0, 13.6), 3.0, 1.0
    else:
        import plot_zz_pol_userstyle as US
        COLOR, LS_ORDER, LW = US.COLOR, US.LS_ORDER, 1.4
        tex, wants_minus = False, None
        figsize, ms, alpha = (9.4, 13.8), US.MS, US.STEP_ALPHA
    plt.rcdefaults() if style == 'user' else None
    if style == 'user':
        import matplotlib as mpl
        mpl.rcParams.update(mpl.rcParamsDefault)

    mp, sy = _panes(nlo, lo)
    fig = plt.figure(figsize=figsize)
    # hspace is wide because every row carries its own x axis name AND the
    # row below it carries a title; at the 0.07 the stacked figures use, the
    # two collide.  Checked on the rendered PNG.
    gs = fig.add_gridspec(4, 2, hspace=0.34, wspace=0.26)
    ax = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(4)]

    # ---- row 1: the mapping, measured -----------------------------------
    for c, side in enumerate(('e', 'mu')):
        a = ax[0][c]
        for key in ('LT', 'TL'):
            y = mp[nlo.label]['shape'][key][side]
            nm = (PA.KF_CURVE_TEX if tex else PA.KF_CURVE_TXT)[key]
            a.stairs(y, COS_EDGES, color=COLOR[key], lw=LW + 0.4,
                     baseline=None, label=nm)
        a.set_xlabel((r'$\cos\theta^{*}(e^{+})$ in the $e^{+}e^{-}$ frame'
                      if tex else 'cos theta*(e+) in the e+e- frame')
                     if side == 'e' else
                     (r'$\cos\theta^{*}(\mu^{+})$ in the $\mu^{+}\mu^{-}$ frame'
                      if tex else 'cos theta*(mu+) in the mu+mu- frame'),
                     fontsize=9.5)
        a.set_ylabel((r'$1/\sigma\,\mathrm{d}\sigma/\mathrm{d}\cos\theta^{*}$'
                      if tex else 'normalised'), fontsize=9)
        rows = {r['key']: r for r in mp[nlo.label]['rows']}
        k = 'e' if side == 'e' else 'mu'
        a.set_title(('%s: <cos2> = %.3f (Z0ZT), %.3f (ZTZ0)'
                     % ('electron side' if side == 'e' else 'muon side',
                        rows['LT'][k], rows['TL'][k]))
                    if not tex else
                    (r'%s: $\langle\cos^{2}\theta^{*}\rangle = %.3f$ '
                     r'($Z_{0}Z_{T}$), $%.3f$ ($Z_{T}Z_{0}$)'
                     % ('electron side' if side == 'e' else 'muon side',
                        rows['LT'][k], rows['TL'][k])), fontsize=8.5)
        a.legend(frameon=(style == 'user'), fontsize=8, loc='lower center',
                 ncol=2)
        a.set_xlim(-1, 1)
        y0, y1 = a.get_ylim()
        a.set_ylim(y0, y0 + (y1 - y0) * 1.34)
        a.tick_params(labelsize=8.5)

    # ---- rows 2 and 3 ---------------------------------------------------
    for c, obs in enumerate(PA.OBS):
        s = sy[obs]
        edges = np.asarray(s['edges'], float)
        x = 0.5 * (edges[:-1] + edges[1:])
        xlab = (PA.LABELS_TEX if tex else PA.LABELS_TXT)[obs][0]

        a = ax[1][c]
        for tag, col in (('NLO', 'k'), ('LO', COLOR['LO'])):
            w = s['within'][tag]
            a.stairs(w['r'], edges, color=col, ls=LS_ORDER[tag],
                     lw=LW + (0.4 if tag == 'NLO' else 0.0), baseline=None,
                     alpha=1.0 if tag == 'NLO' else alpha, zorder=4)
            a.errorbar(x, w['r'], yerr=w['err'],
                       fmt='o' if tag == 'NLO' else 's', ms=ms,
                       mfc='none' if tag == 'LO' else None, color=col,
                       elinewidth=0.9, zorder=5,
                       label='%s  (chi2 %.0f/%d)' % (tag, w['chi2'], w['ndf'])
                       if not tex else
                       r'%s ($\chi^{2} = %.0f/%d$)' % (tag, w['chi2'], w['ndf']))
        a.axhline(1.0, color='grey', ls=':', lw=1.0, zorder=1)
        a.set_ylabel((r'$Z_{T}Z_{0}\,/\,Z_{0}Z_{T}$' if tex
                      else 'ZT Z0 / Z0 ZT'), fontsize=10)

        a = ax[2][c]
        dr, de = s['double']['r'], s['double']['err']
        a.stairs(dr, edges, color=COLOR['TL'], lw=LW + 0.4, baseline=None,
                 zorder=4)
        a.errorbar(x, dr, yerr=de, fmt='o', ms=ms, color=COLOR['TL'],
                   elinewidth=0.9, zorder=5,
                   label=('K(ZTZ0)/K(Z0ZT)  (chi2 %.0f/%d)'
                          % (s['double']['chi2'], s['double']['ndf'])) if not tex
                   else (r'$K(Z_{T}Z_{0})/K(Z_{0}Z_{T})$ '
                         r'($\chi^{2} = %.0f/%d$)'
                         % (s['double']['chi2'], s['double']['ndf'])))
        a.axhline(1.0, color='grey', ls=':', lw=1.0, zorder=1)
        a.set_ylabel((r'$K(Z_{T}Z_{0})\,/\,K(Z_{0}Z_{T})$' if tex
                      else 'K(ZT Z0) / K(Z0 ZT)'), fontsize=10)

        for r in (1, 2):
            b = ax[r][c]
            b.set_xlim(edges[0], edges[-1])
            b.set_xlabel(xlab, fontsize=10)
            b.legend(frameon=(style == 'user'), fontsize=7.5, loc='upper left')
            b.tick_params(labelsize=8.5)
            y0, y1 = b.get_ylim()
            b.set_ylim(y0, y0 + (y1 - y0) * 1.30)
        ax[1][c].set_title(_obs_title_tex(obs) if tex else _obs_title(obs),
                           fontsize=9)

    # ---- row 4: the mirror test, one column per sample -------------------
    for c, d in enumerate((lo, nlo)):
        m = mirror_test(d)
        edges = np.asarray(m['edges'], float)
        x = 0.5 * (edges[:-1] + edges[1:])
        a = ax[3][c]
        a.stairs(m['pred'], edges, color='grey', lw=LW + 1.6, baseline=None,
                 alpha=0.45, zorder=2,
                 label=(r'$1/(Z_{T}Z_{0}/Z_{0}Z_{T})$ on $\Delta\phi(e^{+}e^{-})$'
                        if tex else '1/(ZTZ0/Z0ZT) on Delta phi(e+e-)'))
        a.errorbar(x, m['mumu']['r'], yerr=m['mumu']['err'], fmt='o', ms=ms,
                   color=COLOR['LL'], elinewidth=0.9, zorder=5,
                   label=(r'$Z_{T}Z_{0}/Z_{0}Z_{T}$ on $\Delta\phi(\mu^{+}\mu^{-})$'
                          if tex else 'ZTZ0/Z0ZT on Delta phi(mu+mu-)'))
        a.axhline(1.0, color='grey', ls=':', lw=1.0, zorder=1)
        a.set_xlim(edges[0], edges[-1])
        a.set_xlabel((r'$\Delta\phi(\ell^{+}\ell^{-})$ [rad]' if tex
                      else 'Delta phi(l+ l-) [rad]'), fontsize=10)
        a.set_ylabel((r'$Z_{T}Z_{0}\,/\,Z_{0}Z_{T}$' if tex
                      else 'ZT Z0 / Z0 ZT'), fontsize=10)
        a.set_title((r'mirror test, %s: $\chi^{2} = %.0f/%d$'
                     if tex else 'mirror test, %s: chi2 = %.0f/%d')
                    % (d.label, m['chi2'], m['ndf']), fontsize=9)
        a.legend(frameon=(style == 'user'), fontsize=7.5, loc='upper right')
        a.tick_params(labelsize=8.5)
        y0, y1 = a.get_ylim()
        a.set_ylim(y0, y0 + (y1 - y0) * 1.22)

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, 'lt_tl')
    want = wants_minus(fig) if wants_minus is not None else None
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote %s.pdf / .png' % base)
    return base, want


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    ap.add_argument('--userstyle-out',
                    default=os.path.join(_HERE, 'plots_userstyle'))
    ap.add_argument('--no-figures', action='store_true')
    ap.add_argument('--check-minus', action='store_true',
                    help='the MG7-style PDF only; the user style renders '
                         'without usetex and cannot be bitten by the bug')
    a = ap.parse_args()

    nlo = PA.Data(a.data)
    lo = PA.load_kfactor_partner(a.data)
    if lo is None:
        raise SystemExit('the LO sample is not in %s; this script needs both'
                         % a.data)
    report(nlo, lo)
    if a.no_figures:
        return
    # The MG7 style sets usetex rcParams at import; the user style resets
    # them.  Drawing MG7 FIRST and user second is the order the two existing
    # scripts are run in and the only one in which both come out right.
    base, want = draw(nlo, lo, os.path.join(a.out, OUT_DIR), style='mg7')
    draw(nlo, lo, os.path.join(a.userstyle_out, OUT_DIR), style='user')
    if a.check_minus:
        from plot_zz_pol import USETEX, MINUS_FIX
        from plot_zz_loopinduced import check_minus
        print('usetex = %s   minus workaround active = %s'
              % (USETEX, MINUS_FIX))
        if not want:
            print('%s: no minus sign in this figure, check not applicable'
                  % os.path.basename(base))
            return
        ok, msg = check_minus(base + '.pdf')
        print(msg if not ok else '1/1 applicable PDFs carry /minus')
        if not ok:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
