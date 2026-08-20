#!/usr/bin/env python3
"""Did the per-event mass-stage bound change the written lineshape?

The bound in question is the one ``interface_madspin._mass_stage_bound``
computes per production event for the ``PA`` sequential schemes, in place of the
run-wide ``maxwgts[0]`` extrapolated from the probe.  The safety case for
replacing it is one line of algebra: the mass stage accepts a trial with
probability ``min(1, w/C)`` and redraws otherwise, so the accepted density is

    q_e(m) . min(1, w(m)/C)   propto   q_e(m) . w(m)     for any C >= max w,

i.e. **the bound cancels out of the sample and only sets the cost**.  This
script measures whether that is what happened, on the raw histograms of two
campaigns run off the same production events with the same seeds -- one on the
code before the bound, one after.

READ THE TWO CAMPAIGNS HONESTLY.  The baseline in ``data/`` was taken on
``a311d2e64``, which predates BOTH the per-event bound and the overweight safety
net that landed with it.  So "before vs after" here spans two changes, not one,
and they are separable: the bound changes WHICH trials are accepted (entry
counts move), the safety net changes only the WEIGHT of an event whose trial
weight exceeded its bound (entry counts do not move, one bin does).  Section 1
below reports the two questions separately for exactly that reason.

It also compares both against ``decay_output = weighted``, which makes no
accept/reject at all (one decay configuration per production event, kept, with
``w propto W``).  Histogrammed with its event weights that is the target density
itself, measured with no bound anywhere in the way, so it is the reference that
*cannot* move when a bound changes -- unlike either unweighted campaign.

Three tests per comparison, all on the raw fine grid or the plot grid, none of
them re-deriving anything MadSpin did not write:

  chi2/dof   on the plot bins, two unit-area densities, dof = nbins - 1
             (the same estimator ``plot_lineshape.chi2`` uses, so the numbers
             are directly comparable with the baseline campaign's RESULTS.md
             and with its two replica rows, which are the measured noise floor).
  KS p       two-sample Kolmogorov-Smirnov on the FINE grid, with the effective
             sample sizes (sum w)^2 / sum w^2.  Distribution-free and sensitive
             to a coherent shift that a per-bin chi2 dilutes.
  mean shift the first moment of the per-event virtuality, with its MC error.

and each of them again in slices of the top's velocity in the ``t tbar`` rest
frame, because that is where the bound placement differs most: the per-event
bound is evaluated at the low corner of the Breit-Wigner windows and bounds the
RAMBO reshuffling jacobian, whose ``lambda^(1/2)`` ratio is what carries the
``1/beta`` behaviour near threshold.

Usage::

    python3 compare_bound.py --before data --after data_after_bound \\
                             --out plots_bound --out-userstyle plots_bound_userstyle

Needs only numpy and matplotlib and the two committed data directories; no
MadSpin run, no LHE file, no repository parser.
"""

from __future__ import division

import argparse
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# --------------------------------------------------------------------------
# Statistics.  scipy is not a dependency of this repository, so the two p-values
# are implemented here; both are textbook and both are checked against known
# values by ``--selftest``.
# --------------------------------------------------------------------------
def _gammq(a, x):
    """Regularised upper incomplete gamma Q(a, x), series + continued fraction
    (Numerical Recipes 6.2).  ``chi2`` survival function = Q(dof/2, chi2/2)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError('gammq domain')
    if x == 0.0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:                                     # series for P(a, x)
        ap, total, delta = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - gln)
    tiny = 1e-300                                       # continued fraction
    b, c, d = x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_pvalue(chi2, dof):
    if dof <= 0:
        return float('nan')
    return _gammq(0.5 * dof, 0.5 * chi2)


def _ks_q(lam):
    """Q_KS(lambda) = 2 sum_{j>=1} (-1)^(j-1) exp(-2 j^2 lambda^2)."""
    if lam < 0.0:
        raise ValueError('ks domain')
    if lam < 1e-6:
        return 1.0
    total = 0.0
    for j in range(1, 200):
        term = 2.0 * (-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam)
        total += term
        if abs(term) < 1e-14 * max(abs(total), 1e-30):
            break
    return min(max(total, 0.0), 1.0)


def ks_binned(w1, w2_1, w2, w2_2):
    """Two-sample KS on two BINNED weighted samples sharing a grid.

    ``D`` is the largest gap between the two cumulative distributions, taken at
    the bin edges -- with 360 fine bins over the support that is a slight
    under-estimate of the continuous D, which makes the p-value mildly
    conservative (higher), i.e. it errs towards "consistent" and is therefore
    not the statistic to lean on for a *positive* claim of agreement on its own.
    The effective sample size of a weighted histogram is
    ``(sum w)^2 / sum w^2``, which reduces to the entry count for unit weights.

    Returns (D, n_eff_1, n_eff_2, p).
    """
    s1, s2 = w1.sum(), w2.sum()
    if s1 <= 0 or s2 <= 0:
        return float('nan'), 0.0, 0.0, float('nan')
    c1, c2 = np.cumsum(w1) / s1, np.cumsum(w2) / s2
    d = float(np.abs(c1 - c2).max())
    n1 = s1 * s1 / w2_1.sum() if w2_1.sum() > 0 else 0.0
    n2 = s2 * s2 / w2_2.sum() if w2_2.sum() > 0 else 0.0
    if n1 <= 0 or n2 <= 0:
        return d, n1, n2, float('nan')
    ne = n1 * n2 / (n1 + n2)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    return d, n1, n2, _ks_q(lam)


def selftest():
    """Known values for the two p-value implementations."""
    # chi2 survival: median of chi2_k is ~ k(1-2/(9k))^3; Q at the mean-ish
    # points below are standard table entries.
    checks = [(3.841, 1, 0.05), (5.991, 2, 0.05), (18.307, 10, 0.05),
              (124.342, 100, 0.05), (0.004, 1, 0.95), (3.940, 10, 0.95)]
    for chi2, dof, want in checks:
        got = chi2_pvalue(chi2, dof)
        assert abs(got - want) < 5e-3, (chi2, dof, got, want)
    # KS: Q_KS(0) = 1, Q_KS(1.36) ~ 0.05, Q_KS(1.63) ~ 0.01
    assert abs(_ks_q(0.0) - 1.0) < 1e-12
    assert abs(_ks_q(1.3581) - 0.05) < 1e-3, _ks_q(1.3581)
    assert abs(_ks_q(1.6276) - 0.01) < 1e-3, _ks_q(1.6276)
    print('selftest: chi2 and KS p-values reproduce their table values')


# --------------------------------------------------------------------------
# Merging the two campaigns into one dataset the existing plotting code can eat.
# --------------------------------------------------------------------------
BEFORE, AFTER = 'before__', 'after__'


def merge(before_dir, after_dir, dest):
    """Write a merged histograms.npz + meta.json with prefixed run keys.

    The two campaigns MUST share the binning, the pole and the width -- they are
    the same production sample re-decayed -- and that is asserted rather than
    assumed, because a silent mismatch would make every ratio below meaningless.
    """
    zb = np.load(os.path.join(before_dir, 'histograms.npz'))
    za = np.load(os.path.join(after_dir, 'histograms.npz'))
    mb = json.load(open(os.path.join(before_dir, 'meta.json')))
    ma = json.load(open(os.path.join(after_dir, 'meta.json')))
    assert np.array_equal(zb['bins'], za['bins']), 'binning differs'
    for field in ('pole', 'width', 'bw_cut', 'nevents_requested', 'seed',
                  'nb_core'):
        assert mb[field] == ma[field], '%s differs: %r vs %r' % (
            field, mb[field], ma[field])

    payload = {'bins': zb['bins']}
    if 'beta_edges' in za.files:
        payload['beta_edges'] = za['beta_edges']
    for prefix, z in ((BEFORE, zb), (AFTER, za)):
        for key in z.files:
            if key in ('bins', 'beta_edges'):
                continue
            head, rest = key.split('__', 1)
            payload['%s__%s%s' % (head, prefix, rest)] = z[key]
    os.makedirs(dest, exist_ok=True)
    np.savez_compressed(os.path.join(dest, 'histograms.npz'), **payload)

    meta = dict(ma)
    meta['runs'] = {}
    for prefix, m in ((BEFORE, mb), (AFTER, ma)):
        for key, run in m['runs'].items():
            meta['runs'][prefix + key] = run
    meta['before_code_sha'] = mb['code_sha']
    meta['after_code_sha'] = ma['code_sha']
    meta['before_cross_in'] = mb['cross_in']
    meta['after_cross_in'] = ma['cross_in']
    with open(os.path.join(dest, 'meta.json'), 'w') as fp:
        json.dump(meta, fp, indent=2, sort_keys=False)
    return mb, ma


# --------------------------------------------------------------------------
# What is compared.
#
# Only four cells CAN move.  ``_mass_stage_bound`` is reached under
#   probe is None and maxwgts and upfront and draw_mass and not offshell
# with ``draw_mass = (spinmode == 'PA' and nb_prod_final > 1)`` and
# ``upfront = mode not in ('joint', 'sequential_with_mass')``.  So it is PA (and
# PA with density_keep_jacobian = False), sequential and
# sequential_global_retry, and nothing else: madspin/full keep the global bound
# by decision, onshell samples no virtuality at all, and PA/joint and
# PA/sequential_with_mass have no up-front mass stage to bound.
#
# Every other cell is therefore expected to be BIT-IDENTICAL before and after,
# which is also the check that the production sample was reproduced exactly --
# so it is verified here bin by bin rather than assumed.
# --------------------------------------------------------------------------
MOVERS = ['PA_sequential', 'PA_seqglobal',
          'PAnojac_sequential', 'PAnojac_seqglobal']
UNCHANGED = ['madspin_joint', 'madspin_sequential', 'madspin_seqglobal',
             'PA_joint', 'PA_seqwithmass',
             'PAnojac_joint', 'PAnojac_seqwithmass',
             'onshell_joint', 'onshell_sequential', 'onshell_seqglobal',
             'madspin_joint_rep', 'PA_joint_rep']
# The reference each family is measured against, all from the AFTER campaign.
#
# ``PA`` and ``madspin`` get ``decay_output = weighted``: no accept/reject, so no
# bound, so the target density itself.
#
# ``PAnojac`` does not, and the reason is worth stating rather than hiding.
# ``decay_output = weighted`` DOES honour ``density_keep_jacobian`` (the weighted
# branch shares the ``test = wgt*jac`` line with the joint accept/reject), so a
# ``PAnojac_weighted`` run would be the right reference -- it was simply not run
# in this campaign.  What is used instead is the family's own ``joint`` cell,
# which the bound cannot reach either (``joint`` is not an up-front scheme, so
# ``_mass_stage_bound`` is never called for it) but which is still an unweighted
# sample with a bound of its own at the joint stage.  A weaker reference, and
# labelled as such wherever it is quoted.
WEIGHTED = {'PA': 'PA_weighted', 'PAnojac': 'PAnojac_joint',
            'madspin': 'madspin_weighted'}
# Which of the above are true no-accept/reject references
TRUE_WEIGHTED = {'PA_weighted', 'madspin_weighted'}

FAMILY_OF = {'PA_sequential': 'PA', 'PA_seqglobal': 'PA',
             'PA_joint': 'PA', 'PA_seqwithmass': 'PA',
             'PAnojac_sequential': 'PAnojac', 'PAnojac_seqglobal': 'PAnojac',
             'PAnojac_joint': 'PAnojac', 'PAnojac_seqwithmass': 'PAnojac',
             'madspin_joint': 'madspin', 'madspin_sequential': 'madspin',
             'madspin_seqglobal': 'madspin',
             'madspin_joint_rep': 'madspin', 'PA_joint_rep': 'PA'}

PRETTY = {'PA_joint': 'PA / joint',
          'PA_sequential': 'PA / sequential',
          'PA_seqglobal': 'PA / seq. global retry',
          'PA_seqwithmass': 'PA / seq. with mass',
          'PAnojac_joint': 'PA no-jac / joint',
          'PAnojac_sequential': 'PA no-jac / sequential',
          'PAnojac_seqglobal': 'PA no-jac / seq. global retry',
          'PAnojac_seqwithmass': 'PA no-jac / seq. with mass',
          'madspin_joint': 'madspin / joint',
          'madspin_sequential': 'madspin / sequential',
          'madspin_seqglobal': 'madspin / seq. global retry',
          'madspin_joint_rep': 'madspin / joint (replica seed)',
          'PA_joint_rep': 'PA / joint (replica seed)',
          'onshell_joint': 'onshell / joint',
          'onshell_sequential': 'onshell / sequential',
          'onshell_seqglobal': 'onshell / seq. global retry',
          'PA_weighted': 'PA / decay_output = weighted',
          'madspin_weighted': 'madspin / decay_output = weighted'}


def ref_note(wref):
    return ('decay_output = weighted, no accept/reject'
            if wref in TRUE_WEIGHTED else
            '%s -- NOT a weighted reference, see WEIGHTED in this file'
            % PRETTY.get(wref, wref))


# --------------------------------------------------------------------------
def moments_from_meta(meta, key, tag):
    """(mean, error) of the per-event virtuality, from the raw moments the run
    recorded -- not from the histogram, so the binning plays no part."""
    mom = meta['runs'][key]['%s_moments' % tag]
    s0, s1, s2, sq0 = mom['s0'], mom['s1'], mom['s2'], mom['sq0']
    mean = s1 / s0
    var = max(s2 / s0 - mean * mean, 0.0)
    neff = s0 * s0 / sq0 if sq0 > 0 else 0.0
    return mean, math.sqrt(var / neff) if neff > 0 else float('nan')


class Slice(object):
    """A view of one Data restricted to one beta band (or all of them).

    ``plot_lineshape.Data`` exposes everything through ``raw()``; overriding
    that alone gives the density, the errors, the chi2 and the figures in the
    slice for free, with no copy of any of that logic.
    """

    def __init__(self, data, band=None):
        self._d = data
        self._band = band

    def __getattr__(self, name):
        return getattr(self._d, name)

    def has(self, key, tag='t'):
        if self._band is None:
            return self._d.has(key, tag)
        return ('sumwb__%s__%s' % (key, tag)) in self._d.z

    def raw(self, key, tag='t'):
        if self._band is None:
            return self._d.raw(key, tag)
        from plot_lineshape import _apply
        g = self._d.groups
        b = self._band
        return (_apply(self._d.z['sumwb__%s__%s' % (key, tag)][b], g),
                _apply(self._d.z['sumw2b__%s__%s' % (key, tag)][b], g),
                _apply(self._d.z['nb__%s__%s' % (key, tag)][b], g))

    def density(self, key, tag='t'):
        w, w2, _n = self.raw(key, tag)
        norm = w.sum()
        if norm <= 0:
            nan = np.full_like(w, np.nan, dtype=float)
            return nan, nan
        return w / (norm * self._d.width), np.sqrt(w2) / (norm * self._d.width)


def compare(view, fine_z, a, b, tag, band=None):
    """chi2/dof + p, KS D + p, for two cells of one (possibly sliced) view."""
    from plot_lineshape import chi2 as chi2_of
    pa, pae = view.density(a, tag)
    pb, pbe = view.density(b, tag)
    c, dof = chi2_of(pa, pae, pb, pbe)
    if band is None:
        wa, w2a = fine_z['sumw__%s__%s' % (a, tag)], fine_z['sumw2__%s__%s' % (a, tag)]
        wb, w2b = fine_z['sumw__%s__%s' % (b, tag)], fine_z['sumw2__%s__%s' % (b, tag)]
    else:
        wa = fine_z['sumwb__%s__%s' % (a, tag)][band]
        w2a = fine_z['sumw2b__%s__%s' % (a, tag)][band]
        wb = fine_z['sumwb__%s__%s' % (b, tag)][band]
        w2b = fine_z['sumw2b__%s__%s' % (b, tag)][band]
    d, n1, n2, p = ks_binned(wa, w2a, wb, w2b)
    return dict(chi2=c, dof=dof, chi2_p=chi2_pvalue(c, dof),
                ks_d=d, ks_p=p, neff_a=n1, neff_b=n2)


def identical(z, a, b):
    """How close two cells are on the raw fine grid, both resonances.

    Two separate questions, and they have different answers here:

      * are the same events in the same bins?  That is the ENTRY count
        ``n__...``, and it is what a change of accept/reject bound would move:
        a different bound is a different trial sequence.
      * do they carry the same weights?  That is ``sumw__...``.  The overweight
        safety net (PR #375, which sits between the two campaigns compared here)
        writes ``max(1, w/C)`` instead of 1 on a trial whose weight exceeded its
        bound, so a cell that drew *exactly* the same events can still differ in
        ``sumw`` -- by the carried excess, and only in the bins those events
        landed in.

    Returns a dict, or None when the cell is not in both campaigns.
    """
    out = {'entries': True, 'weights': True, 'bins': 0, 'dsumw': 0.0}
    for tag in ('t', 'tbar'):
        for field, flag in (('n', 'entries'), ('sumw', 'weights')):
            ka = '%s__%s__%s' % (field, a, tag)
            kb = '%s__%s__%s' % (field, b, tag)
            if ka not in z.files or kb not in z.files:
                return None
            if not np.array_equal(z[ka], z[kb]):
                out[flag] = False
                if field == 'sumw':
                    diff = z[ka] - z[kb]
                    out['bins'] = max(out['bins'],
                                      int((np.abs(diff) > 0).sum()))
                    out['dsumw'] = max(out['dsumw'], abs(float(diff.sum())))
    return out


def verdict(same):
    if same is None:
        return 'not in both campaigns'
    if same['entries'] and same['weights']:
        return 'identical'
    if same['entries']:
        return ('same events, %d bin(s) differ in weight by %.4g total'
                % (same['bins'], same['dsumw']))
    return '*** DIFFERENT EVENTS ***'


# --------------------------------------------------------------------------
def render(specs, out_mg7, out_user):
    """Draw every figure, MG7 style first and the user's style second.

    The order is not cosmetic.  ``plot_lineshape`` sets serif/usetex rcParams at
    import time and ``plot_lineshape_userstyle`` resets rcParams to the
    matplotlib defaults at ITS import time -- which is right when each is run as
    its own process, as they are meant to be, but means that interleaving the
    two renderings in one process silently strips the MG7 style off every figure
    drawn after the first user-style one.  So: import and finish one, then the
    other.
    """
    import plot_lineshape as P
    for _view, keys, ref, _tag, _name, _title in specs:
        for key in list(keys) + [ref]:
            P.LABEL.setdefault(key, key)
            P.COLOR.setdefault(key, 'black')
            P.LS.setdefault(key, 'solid')
    written = []
    os.makedirs(out_mg7, exist_ok=True)
    for view, keys, ref, tag, name, title in specs:
        written += P.lineshape_figure(view, list(keys), tag, out_mg7, name,
                                      title=title, ref=ref, ncol=1)
    import plot_lineshape_userstyle as U           # resets rcParams: last
    os.makedirs(out_user, exist_ok=True)
    for view, keys, ref, tag, name, _title in specs:
        paths, _ylim = U.lineshape_figure(view, list(keys) + [ref], tag,
                                          os.path.join(out_user, 'us_' + name),
                                          ('pdf', 'png'), ref=ref)
        written += paths
    return written


def register_styles():
    """Colour / linestyle for the prefixed keys, in the MG7 script's dicts.

    One colour per CELL and one linestyle per CAMPAIGN, so a before/after pair
    is the same colour and the eye compares the right two curves; the weighted
    reference is black.
    """
    import plot_lineshape as P
    cell_colour = {'PA_sequential': 'tab:orange', 'PA_seqglobal': 'tab:brown',
                   'PA_joint': 'red', 'PA_seqwithmass': 'tab:pink',
                   'PAnojac_sequential': 'tab:orange',
                   'PAnojac_seqglobal': 'tab:brown',
                   'PAnojac_joint': 'red', 'PAnojac_seqwithmass': 'tab:pink',
                   'madspin_joint': 'blue', 'madspin_sequential': 'tab:green',
                   'madspin_seqglobal': 'tab:purple',
                   'madspin_joint_rep': 'gray', 'PA_joint_rep': 'gray',
                   'PA_weighted': 'black', 'madspin_weighted': 'black'}
    for cell, colour in cell_colour.items():
        for prefix, ls, what in ((BEFORE, 'dashed', 'before'),
                                 (AFTER, 'solid', 'after')):
            key = prefix + cell
            P.COLOR[key] = colour
            P.LS[key] = ls
            P.LABEL[key] = '%s, %s' % (PRETTY.get(cell, cell), what)
    for cell in ('PA_weighted', 'madspin_weighted'):
        P.LABEL[AFTER + cell] = PRETTY[cell]
        P.LS[AFTER + cell] = 'solid'


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--after', default=os.path.join(_HERE, 'data_after_bound'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots_bound'))
    ap.add_argument('--out-userstyle',
                    default=os.path.join(_HERE, 'plots_bound_userstyle'))
    ap.add_argument('--merged', default=None,
                    help='keep the merged npz/meta here instead of a temp dir')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return

    tmp = args.merged or tempfile.mkdtemp(prefix='ms_bound_merge_')
    mb, ma = merge(args.before, args.after, tmp)

    import plot_lineshape as P
    register_styles()
    d = P.Data(tmp)
    z = d.z
    full = Slice(d, None)
    beta_edges = z['beta_edges'] if 'beta_edges' in z.files else None
    nband = (len(beta_edges) - 1) if beta_edges is not None else 0

    L = []
    def say(fmt='', *a):
        L.append(fmt % a if a else fmt)

    say('The per-event mass-stage bound: does the written lineshape move?')
    say('=' * 74)
    say('before campaign: %s   code %s' % (args.before, mb['code_sha'][:12]))
    say('after  campaign: %s   code %s' % (args.after, ma['code_sha'][:12]))
    say('%d production events, madevent+MadSpin seed %d, nb_core %d, shared'
        % (ma['nevents_requested'], ma['seed'], ma['nb_core']))
    say('production sigma: before %.6g pb, after %.6g pb'
        % (mb['cross_in'], ma['cross_in']))
    say('pole M = %.6f GeV, width Gamma = %.6f GeV, support |m-M| < %g Gamma'
        % (ma['pole'], ma['width'], ma['bw_cut']))
    say()

    # ---- 1. which cells are even allowed to move -------------------------
    say('1. Cells whose ACCEPTED EVENTS cannot be touched by the bound, checked')
    say('   bin by bin on the raw fine grid.  _mass_stage_bound is reached only')
    say('   under spinmode = PA, an up-front scheme (not joint, not')
    say('   sequential_with_mass) and not offshell -- so every cell below must')
    say('   draw and accept exactly the same events before and after, and a')
    say('   difference in the ENTRY counts here would be a reproducibility')
    say('   failure (a different production sample) and not a lineshape result.')
    say('   Their WEIGHTS may still differ: the two campaigns are separated by')
    say('   the overweight safety net as well as by the bound, and that writes')
    say('   max(1, w/C) instead of 1 on an event whose trial weight exceeded')
    say('   its bound.  The two questions are reported separately.')
    n_ok = n_carry = n_bad = n_missing = 0
    for cell in UNCHANGED:
        same = identical(z, BEFORE + cell, AFTER + cell)
        if same is None:
            n_missing += 1
        elif same['entries'] and same['weights']:
            n_ok += 1
        elif same['entries']:
            n_carry += 1
        else:
            n_bad += 1
        say('   %-32s %s' % (PRETTY.get(cell, cell), verdict(same)))
    say('   -> %d bit-identical, %d same events with carried weights, '
        '%d different events, %d not run in both'
        % (n_ok, n_carry, n_bad, n_missing))
    say()

    say('2. Cells the bound DOES touch (PA / PA no-jac, sequential and')
    say('   sequential_global_retry).  These are expected to differ event by')
    say('   event -- a different bound is a different accept/reject sequence --')
    say('   and to agree in distribution, which is what the rest measures.')
    for cell in MOVERS:
        say('   %-32s %s' % (PRETTY.get(cell, cell),
                             verdict(identical(z, BEFORE + cell, AFTER + cell))))
    say()

    # ---- 3. the lineshape numbers ---------------------------------------
    say('3. Lineshape, per cell and per resonance.')
    say('   "after vs before"   the direct question: did the written sample')
    say('                       move when the bound moved?')
    say('   "vs weighted"       against decay_output = weighted, which makes no')
    say('                       accept/reject at all and is therefore the target')
    say('                       density with no bound in it.  Quoted for BOTH')
    say('                       campaigns, so the two can be read against each')
    say('                       other and against the reference itself.')
    for fam in sorted(set(WEIGHTED)):
        say('                       %-9s -> %s' % (fam, ref_note(WEIGHTED[fam])))
    say('   chi2 is on the %d plot bins, dof = nbins - 1 (both densities are'
        % len(d.centre))
    say('   unit-area, one shared linear constraint).  KS is two-sample on the')
    say('   360-bin fine grid with effective sample sizes (sum w)^2/sum w^2.')
    say('   Both statistics ERR TOWARDS AGREEMENT here and neither is evidence')
    say('   on its own: every cell is decayed off the SAME production events, so')
    say('   two cells are positively correlated and an independent-errors chi2')
    say('   is conservative (several baseline rows sit below their dof for that')
    say('   reason), while the KS D is taken on a 360-bin grid and between')
    say('   correlated samples, both of which push its p-value up.  That is why')
    say('   the reference is a MEASURED floor and not a nominal 1.0.')
    say('   Read every chi2/dof against the baseline campaign\'s measured noise')
    say('   floor -- two replicas of ONE scheme at a second MadSpin seed gave')
    say('   139.2/138 and 135.7/138 combined over the two resonances, i.e.')
    say('   1.009 and 0.983 per dof.  Nothing at or below that is a signal.')
    say()
    rows = []
    for cell in MOVERS + [c for c in UNCHANGED if FAMILY_OF.get(c)]:
        fam = FAMILY_OF.get(cell)
        wref = WEIGHTED.get(fam)
        for tag in ('t', 'tbar'):
            if not (d.has(BEFORE + cell, tag) and d.has(AFTER + cell, tag)):
                continue
            ab = compare(full, z, AFTER + cell, BEFORE + cell, tag)
            ma_mean, ma_err = moments_from_meta(d.meta, AFTER + cell, tag)
            mb_mean, mb_err = moments_from_meta(d.meta, BEFORE + cell, tag)
            row = dict(cell=cell, tag=tag, ab=ab,
                       dmean=1000.0 * (ma_mean - mb_mean),
                       dmean_err=1000.0 * math.hypot(ma_err, mb_err))
            for who, prefix in (('after', AFTER), ('before', BEFORE)):
                if wref and wref != cell and d.has(AFTER + wref, tag):
                    row[who + '_w'] = compare(full, z, prefix + cell,
                                              AFTER + wref, tag)
                    wm, we = moments_from_meta(d.meta, AFTER + wref, tag)
                    cm, ce = moments_from_meta(d.meta, prefix + cell, tag)
                    row[who + '_wmean'] = 1000.0 * (cm - wm)
                    row[who + '_wmean_err'] = 1000.0 * math.hypot(ce, we)
            rows.append(row)
    hdr = ('   %-30s %-5s | %11s %7s | %11s %7s | %11s %7s | %14s'
           % ('cell', 'res', 'after/before', 'KS p', 'after/wgt', 'KS p',
              'before/wgt', 'KS p', 'd<m> [MeV]'))
    say(hdr)
    say('   ' + '-' * (len(hdr) - 3))
    for r in rows:
        def cell_of(k):
            if k not in r:
                return '%11s %7s' % ('--', '--')
            return '%7.1f/%-3d %7.3f' % (r[k]['chi2'], r[k]['dof'], r[k]['ks_p'])
        say('   %-30s %-5s | %s | %s | %s | %6.1f +- %-5.1f'
            % (PRETTY.get(r['cell'], r['cell']),
               'm(t)' if r['tag'] == 't' else 'm(tb)',
               '%7.1f/%-3d %7.3f' % (r['ab']['chi2'], r['ab']['dof'],
                                     r['ab']['ks_p']),
               cell_of('after_w'), cell_of('before_w'),
               r['dmean'], r['dmean_err']))
    say()
    say('   d<m> is after minus before, in MeV, on the per-event virtuality.')
    say('   A real bias moves m(t) and m(tbar) the SAME way: they are two')
    say('   independent virtuality draws in one event, so opposite signs are a')
    say('   fluctuation.  The baseline campaign found single-resonance shifts')
    say('   reaching 4.1 sigma that reverse sign between the two; both columns')
    say('   are therefore quoted and neither is read on its own.')
    say()
    say('   The same rows combined over the two resonances, which is the form')
    say('   the baseline campaign quotes and the form the replica floor is in.')
    say('   "combined chi2" is m(t) + m(tbar) on 138 dof, so 1 sigma is')
    say('   sqrt(2*138) = 16.6; the replicas sat at 139.2 and 135.7.')
    say('   The last two columns are the decisive ones: how far each campaign')
    say('   sits from the SAME unbounded reference.  A bound that biased the')
    say('   sample would show as the after column being further away.')
    say('   %-30s %11s %11s %11s %13s %13s'
        % ('cell', 'a/b chi2', 'a/w chi2', 'b/w chi2', 'a-b <m> MeV',
           'a-w, b-w <m>'))
    for r in rows:
        if r['tag'] != 't':
            continue
        mate = next((x for x in rows
                     if x['cell'] == r['cell'] and x['tag'] == 'tbar'), None)
        if mate is None:
            continue
        def comb(key):
            if key not in r or key not in mate:
                return '%11s' % '--'
            return '%7.1f/138' % (r[key]['chi2'] + mate[key]['chi2'])
        dm = 0.5 * (r['dmean'] + mate['dmean'])
        dme = 0.5 * math.hypot(r['dmean_err'], mate['dmean_err'])
        if 'after_wmean' in r and 'after_wmean' in mate:
            aw = 0.5 * (r['after_wmean'] + mate['after_wmean'])
            bw = 0.5 * (r['before_wmean'] + mate['before_wmean'])
            tail = '%+6.1f / %+6.1f' % (aw, bw)
        else:
            tail = '%13s' % '--'
        say('   %-30s %s %s %s %+7.1f+-%-4.1f %s'
            % (PRETTY.get(r['cell'], r['cell']), comb('ab'), comb('after_w'),
               comb('before_w'), dm, dme, tail))
    say()

    # ---- 4. threshold slices --------------------------------------------
    if nband:
        say('4. Sliced by beta, the top velocity in the t tbar rest frame.')
        say('   The per-event bound is evaluated at the LOW corner of the')
        say('   Breit-Wigner windows and bounds the RAMBO reshuffling jacobian,')
        say('   a ratio of lambda^(1/2) factors -- so the low-beta (near')
        say('   threshold) slice is where the bound and the weight it bounds are')
        say('   furthest apart, and where a bound that failed to dominate would')
        say('   show first.  Slices exist in the after campaign only: the')
        say('   baseline predates the slicing, so the comparison here is against')
        say('   decay_output = weighted, which is the stronger reference anyway.')
        for b in range(nband):
            say('   beta in [%.2f, %.2f]' % (beta_edges[b], beta_edges[b + 1]))
            view = Slice(d, b)
            for cell in MOVERS + ['PA_joint', 'madspin_sequential']:
                fam = FAMILY_OF.get(cell)
                wref = WEIGHTED.get(fam)
                if not wref or not view.has(AFTER + cell, 't'):
                    continue
                out = []
                for tag in ('t', 'tbar'):
                    c = compare(view, z, AFTER + cell, AFTER + wref, tag, band=b)
                    out.append('%s %6.1f/%-3d p=%.3f'
                               % ('m(t) ' if tag == 't' else 'm(tb)',
                                  c['chi2'], c['dof'], c['chi2_p']))
                say('     %-30s %s' % (PRETTY.get(cell, cell), ' | '.join(out)))
        say()
        say('   share of the sample per slice (weight fraction):')
        for cell in ('PA_sequential', 'PA_weighted', 'madspin_sequential'):
            key = AFTER + cell
            if key not in d.meta['runs'] or 'beta_slices' not in d.meta['runs'][key]:
                continue
            sl = d.meta['runs'][key]['beta_slices']
            tot = sum(sl['sumw']) or 1.0
            say('     %-30s %s   (%d events with no beta)'
                % (PRETTY.get(cell, cell),
                   '  '.join('%.3f' % (x / tot) for x in sl['sumw']),
                   sl['events_without_beta']))
        say()

    # ---- 5. the overweight counters -------------------------------------
    say('5. Overweight counters (the PR-#375 safety net), per cell.')
    say('   A bound that dominates its weight can never be exceeded, so a')
    say('   per-event bound must read 0.  A NON-zero counter together with a')
    say('   lineshape shift is the signature of a bound that does not dominate;')
    say('   a non-zero counter alone means the excess was carried on the event')
    say('   weight instead of being clipped, which is exact but is still a')
    say('   weighted sample.')
    say('   %-32s %10s %10s' % ('cell', 'before', 'after'))
    for cell in MOVERS + UNCHANGED:
        vb = d.meta['runs'].get(BEFORE + cell, {}).get('overflows')
        va = d.meta['runs'].get(AFTER + cell, {}).get('overflows')
        if vb is None and va is None:
            continue
        say('   %-32s %10s %10s' % (PRETTY.get(cell, cell),
                                    '--' if vb is None else vb,
                                    '--' if va is None else va))
    say('   NOTE: the baseline campaign parsed this counter with a regex that')
    say('   matched the pre-safety-net wording ("... exceeded their per-particle')
    say('   maximum").  The line now reads "... exceeded their stage maximum",')
    say('   so a BEFORE column of 0 here can mean either zero or unparsed; the')
    say('   baseline\'s own RESULTS.md, read off the logs, is the authority for')
    say('   the before numbers (madspin/sequential 2, madspin/seq-global 2,')
    say('   PA/seq-global 3, everything else 0 out of 200 000).')
    say()

    # ---- 6. figures ------------------------------------------------------
    say('6. Figures')
    fam_figs = [
        ('PA', 'PA: the per-event mass bound, before and after',
         ['PA_sequential', 'PA_seqglobal', 'PA_joint'], WEIGHTED['PA']),
        ('PAnojac', 'PA, density_keep_jacobian = False: before and after',
         ['PAnojac_sequential', 'PAnojac_seqglobal'],
         WEIGHTED['PAnojac']),
        ('madspin', 'madspin: the global bound is kept, and nothing moves',
         ['madspin_sequential', 'madspin_seqglobal', 'madspin_joint'],
         WEIGHTED['madspin']),
    ]
    specs = []
    for fam, title, cells, wref in fam_figs:
        keys = []
        for cell in cells:
            for prefix in (BEFORE, AFTER):
                if d.has(prefix + cell, 't') and prefix + cell != AFTER + wref:
                    keys.append(prefix + cell)
        ref = AFTER + wref
        if not d.has(ref, 't') or not keys:
            continue
        for tag in ('t', 'tbar'):
            specs.append((full, keys, ref, tag,
                          'bound_m%s_%s' % ('t' if tag == 't' else 'tbar', fam),
                          title))
    if nband:
        for b in range(nband):
            view = Slice(d, b)
            keys = [AFTER + c for c in ('PA_sequential', 'PA_seqglobal',
                                        'PA_joint')
                    if view.has(AFTER + c, 't')]
            ref = AFTER + 'PA_weighted'
            if not view.has(ref, 't') or not keys:
                continue
            title = ('PA against the unbounded weighted reference, '
                     r'$\beta \in [%.2f, %.2f]$'
                     % (beta_edges[b], beta_edges[b + 1]))
            for tag in ('t', 'tbar'):
                specs.append((view, keys, ref, tag,
                              'bound_m%s_beta%d'
                              % ('t' if tag == 't' else 'tbar', b), title))
    written = render(specs, args.out, args.out_userstyle)
    for path in written:
        say('   %s' % os.path.relpath(path, _HERE))
    say()

    text = '\n'.join(L) + '\n'
    for target in (args.out, args.out_userstyle):
        os.makedirs(target, exist_ok=True)
    with open(os.path.join(args.out, 'bound_numbers.txt'), 'w') as fp:
        fp.write(text)
    sys.stdout.write(text)
    if not args.merged:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
