#!/usr/bin/env python3
"""What correcting ``A_e`` would change, per mode: the ``sqrt(shat)`` spectrum,
the ``m(t)`` / ``m(tbar)`` lineshapes, the cross section, and the weight
distribution it introduces.

This is exact arithmetic on the runs that exist, not a second Monte Carlo.  The
correction multiplies the written event by ``A_e/<A>`` and changes **no random
decision** -- verified on 3000 events: the decayed LHE of a
``mass_normalisation`` run is byte-identical to the baseline's except in the
weight field, and the weights it carries reproduce the offline quadrature to
2.6e-8 (the LHE's own weight precision).  So reweighting a baseline run's events
by ``A_e/<A>`` *is* the corrected run, and every number below is computed that
way.

The lineshape errors are quoted twice on purpose:

* the **paired** error ``sd(f)_bin/sqrt(N_bin)``, which is the right one for
  "does the correction move this bin", since the two histograms are the same
  events;
* the shift in the mean in MeV, which is directly comparable to the
  ``MadSpin/validation/mt_lineshape/RESULTS.md`` table -- same process, same
  200 000 production events, same observable -- where the *measured* replica
  noise floor is +-10.1 MeV per resonance and +-7.1 MeV combined, and where
  single-resonance shifts of 4.1 sigma that reverse sign between ``t`` and
  ``tbar`` were seen and shown to be fluctuations.  Only a same-sign move in
  both is evidence of a real shift.

Usage::

    python3 doc/madspin_ae_normalisation/analyse_impact.py \
        --ae data/ae_pa.npz --lhe <events_decayed.lhe.gz> --label PA \
        --out data/impact_pa.json
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import gzip
import json
import math

import numpy as np


SQRTS_EDGES = [346, 347, 348, 350, 352, 355, 360, 370, 380, 400, 450, 500,
               600, 800, 1200, 1e9]


def harvest(path, edges):
    """``m(t)``, ``m(tbar)`` and ``sqrt(shat)`` per decayed event, in file
    order.  ``sqrt(shat)`` is the invariant mass of the two status-2
    resonances, which the reshuffling preserves, so it is the production
    event's."""
    opener = gzip.open if path.endswith('.gz') else open
    mt, mtb, sh, wg = [], [], [], []
    with opener(path, 'rt') as fh:
        in_ev = need = False
        acc = None
        cur = {}
        for line in fh:
            if line.startswith('<event'):
                in_ev, need = True, True
                acc = [0.0, 0.0, 0.0, 0.0]
                cur = {}
                continue
            if not in_ev:
                continue
            if line.startswith('</event'):
                in_ev = False
                if 6 in cur and -6 in cur:
                    mt.append(cur[6])
                    mtb.append(cur[-6])
                    m2 = acc[3] ** 2 - acc[0] ** 2 - acc[1] ** 2 - acc[2] ** 2
                    sh.append(math.sqrt(max(m2, 0.0)))
                continue
            f = line.split()
            if need:
                wg.append(float(f[2]))
                need = False
                continue
            if len(f) < 11 or f[1] != '2':
                continue
            pid = int(f[0])
            if pid not in (6, -6) or pid in cur:
                continue
            px, py, pz, en = (float(f[6]), float(f[7]), float(f[8]),
                              float(f[9]))
            m2 = en * en - px * px - py * py - pz * pz
            cur[pid] = math.sqrt(max(m2, 0.0))
            acc[0] += px
            acc[1] += py
            acc[2] += pz
            acc[3] += en
    return (np.asarray(mt), np.asarray(mtb), np.asarray(sh), np.asarray(wg))


def slice_table(sqrts, f, edges):
    rows = []
    n = len(sqrts)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (sqrts >= lo) & (sqrts < hi)
        k = int(sel.sum())
        if not k:
            continue
        fs = f[sel]
        rows.append(dict(lo=lo, hi=None if hi > 1e8 else hi, n=k,
                         frac=k / n, mean=float(fs.mean()),
                         err=float(fs.std() / math.sqrt(k)),
                         lo_f=float(fs.min()), hi_f=float(fs.max())))
    return rows


def _group(fine, groups):
    """Sum a fine-grid array into whole groups of fine bins."""
    out, i = [], 0
    for g in groups:
        out.append(fine[i:i + g].sum())
        i += g
    assert i == len(fine), (i, len(fine))
    return np.asarray(out)


def coarse_groups(per_width=12):
    """The 70-bin scheme ``MadSpin/validation/mt_lineshape/plot_lineshape.py``
    uses, so a chi2 here is on the same bins as the replica noise floor it is
    compared against: Gamma/6 in the core (|m-M| < 3.5 Gamma), Gamma/2 on the
    shoulders (to 9 Gamma), 2 Gamma in the far tails (to 15 Gamma).  Whole
    numbers of Gamma/12 fine bins, so no edge moves.
    """
    tail = [24] * 3            # 9..15 Gamma, 2 Gamma each
    shoulder = [6] * 11        # 3.5..9 Gamma, Gamma/2 each
    core = [2] * 42            # -3.5..3.5 Gamma, Gamma/6 each
    return tail + shoulder + core + shoulder[::-1] + tail[::-1]


def lineshape(m, f, pole, width, bw_cut=15.0, per_width=12):
    """``<f> - 1`` per bin, on the fine grid and on the 70-bin plotting one.

    Two chi2 are reported and they answer different questions:

    ``chi2_paired``  the two histograms are the SAME events, so the error on
                     ``<f>_bin - 1`` is ``sd(f)_bin/sqrt(N_bin)`` and vanishes
                     when the correction is the identity.  This is the
                     sensitive test: does the correction move this bin at all.
    ``chi2_indep``   what a comparison of two independent 200 000-event runs
                     would report, ``sum (S1-N)^2/(S2+N)``.  This is the one to
                     put against the measured replica floor of 139.2/138 and
                     135.7/138 -- i.e. would anyone notice.
    """
    nb = int(round(2 * bw_cut * per_width))
    edges = np.linspace(pole - bw_cut * width, pole + bw_cut * width, nb + 1)
    idx = np.clip(np.searchsorted(edges, m, side='right') - 1, 0, nb - 1)
    n = np.bincount(idx, minlength=nb).astype(float)
    s1 = np.bincount(idx, weights=f, minlength=nb)
    s2 = np.bincount(idx, weights=f * f, minlength=nb)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = s1 / n
        var = s2 / n - mean * mean
        err = np.sqrt(np.maximum(var, 0.0) / n)
    good = n > 0
    t = np.where(good & (err > 0), (mean - 1.0) / np.where(err > 0, err, 1), 0.0)

    g = coarse_groups(per_width)
    cn, cs1, cs2 = _group(n, g), _group(s1, g), _group(s2, g)
    cgood = cn > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        cmean = cs1 / cn
        cvar = cs2 / cn - cmean * cmean
        cerr = np.sqrt(np.maximum(cvar, 0.0) / cn)
    ct = np.where(cgood & (cerr > 0), (cmean - 1.0)
                  / np.where(cerr > 0, cerr, 1), 0.0)
    ci = np.where(cgood, (cs1 - cn) ** 2 / np.where(cgood, cs2 + cn, 1), 0.0)
    return dict(edges=edges, n=n, mean=mean, err=err,
                chi2=float((t[good] ** 2).sum()), ndof=int(good.sum()),
                coarse_n=cn, coarse_mean=cmean, coarse_err=cerr,
                chi2_paired_coarse=float((ct[cgood] ** 2).sum()),
                chi2_indep_coarse=float(ci[cgood].sum()),
                ndof_coarse=int(cgood.sum()) - 1)


def moment_shift(m, f):
    """``<m>`` before and after, and the shift with its PAIRED error.

    ``Delta = sum f_e (m_e - <m>_0) / sum f_e``; its error is the sample
    standard error of ``(f_e - 1)(m_e - <m>_0)``, i.e. it vanishes when the
    correction is the identity, which an unpaired error would not.
    """
    n = len(m)
    m0 = m.mean()
    m1 = float((f * m).sum() / f.sum())
    d = (f - 1.0) * (m - m0)
    err = float(d.std() / math.sqrt(n) / f.mean())
    return m0, m1, (m1 - m0), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ae', required=True, help='npz from measure_ae.py')
    ap.add_argument('--lhe', default=None, help='the baseline decayed LHE')
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--ae-override', default=None,
                    help='npz with an "A" that replaces the quadrature one '
                         '(used for the offshell mode, where A_e carries the '
                         'measured Tr(rho)/|M|^2 factor)')
    args = ap.parse_args()

    d = np.load(args.ae)
    sqrts, A = d['sqrts'], d['A']
    pole, width, bw_cut = d['meta'][0], d['meta'][1], d['meta'][2]
    if args.ae_override:
        A = np.load(args.ae_override)['A']
        assert len(A) == len(sqrts)
    f = A / A.mean()

    out = dict(label=args.label, n=len(A),
               A_mean=float(A.mean()), A_sd=float(A.std()),
               A_min=float(A.min()), A_max=float(A.max()),
               A_rel_sd=float(A.std() / A.mean()),
               weight_quantiles={str(q): float(np.percentile(f, q))
                                 for q in (0.1, 1, 5, 50, 95, 99, 99.9)},
               frac_beyond_1pct=float((np.abs(f - 1) > 0.01).mean()),
               frac_beyond_5pct=float((np.abs(f - 1) > 0.05).mean()),
               mean_abs_dev=float(np.abs(f - 1).mean()),
               # effective statistics after the reweighting
               neff_over_n=float(f.sum() ** 2 / (len(f) * (f ** 2).sum())),
               # a probe-based <A> from k events would carry this error, and it
               # would go straight onto the cross section
               ref_err_80=float(A.std() / A.mean() / math.sqrt(80)),
               ref_err_2000=float(A.std() / A.mean() / math.sqrt(2000)),
               sqrts_slices=slice_table(sqrts, f, SQRTS_EDGES))

    print('=== %s   %d events' % (args.label, len(A)))
    print('A_e   mean %.6f  rel sd %.4f  range [%.4f, %.4f]'
          % (A.mean(), A.std() / A.mean(), A.min(), A.max()))
    print('weight A_e/<A>: |f-1|>1%% in %.2f%% of events, >5%% in %.2f%%, '
          'mean |f-1| = %.3f%%, N_eff/N = %.5f'
          % (100 * out['frac_beyond_1pct'], 100 * out['frac_beyond_5pct'],
             100 * out['mean_abs_dev'], out['neff_over_n']))
    print('\nsqrt(shat) spectrum: relative change of the decayed sample')
    print('  %-14s %8s %8s   %s' % ('slice [GeV]', 'events', '% of', 'change'))
    for r in out['sqrts_slices']:
        print('  %6.0f - %-6s %8d %7.3f%%   %+7.2f%% +- %.2f%%   [%.3f, %.3f]'
              % (r['lo'], '%.0f' % r['hi'] if r['hi'] else 'inf', r['n'],
                 100 * r['frac'], 100 * (r['mean'] - 1), 100 * r['err'],
                 r['lo_f'], r['hi_f']))

    if args.lhe:
        mt, mtb, sh, wg = harvest(args.lhe, None)
        print('\ndecayed LHE: %d events, sum(w) = %.6g' % (len(mt), wg.sum()))
        assert len(mt) == len(A), (len(mt), len(A))
        # the decayed sqrt(shat) must be the production one, event by event:
        # the reshuffling preserves it. This is what pins the alignment.
        rel = np.abs(sh / sqrts - 1)
        print('alignment: max |sqrt(shat)_decayed/sqrt(shat)_prod - 1| = %.3g'
              % rel.max())
        out['alignment_max_rel'] = float(rel.max())
        out['sigma_baseline'] = float(wg.mean())
        out['sigma_corrected'] = float((wg * f).mean())
        out['sigma_rel_change'] = float((wg * f).mean() / wg.mean() - 1)
        print('cross section: mean weight %.6f -> %.6f  (%+.3g relative)'
              % (wg.mean(), (wg * f).mean(), out['sigma_rel_change']))
        for tag, m in (('t', mt), ('tbar', mtb)):
            ls = lineshape(m, f, pole, width, bw_cut)
            m0, m1, dm, derr = moment_shift(m, f)
            out['%s_mean_before' % tag] = m0
            out['%s_mean_after' % tag] = m1
            out['%s_mean_shift_MeV' % tag] = 1000 * dm
            out['%s_mean_shift_err_MeV' % tag] = 1000 * derr
            out['%s_chi2_paired' % tag] = ls['chi2']
            out['%s_ndof' % tag] = ls['ndof']
            out['%s_chi2_paired_70bin' % tag] = ls['chi2_paired_coarse']
            out['%s_chi2_indep_70bin' % tag] = ls['chi2_indep_coarse']
            out['%s_ndof_70bin' % tag] = ls['ndof_coarse']
            out['%s_rms_before' % tag] = float(m.std())
            out['%s_rms_after' % tag] = float(
                math.sqrt((f * (m - m1) ** 2).sum() / f.sum()))
            print('m(%s): <m> %.6f -> %.6f  shift %+.2f +- %.2f MeV (paired)'
                  % (tag, m0, m1, 1000 * dm, 1000 * derr))
            print('       rms %.5f -> %.5f;  70-bin chi2: paired %.1f/%d, '
                  'independent-error %.3f/%d'
                  % (out['%s_rms_before' % tag], out['%s_rms_after' % tag],
                     ls['chi2_paired_coarse'], ls['ndof_coarse'],
                     ls['chi2_indep_coarse'], ls['ndof_coarse']))
            g = coarse_groups()
            centres = []
            i = 0
            for gg in g:
                centres.append(0.5 * (ls['edges'][i] + ls['edges'][i + gg]))
                i += gg
            b = int(np.argmax(np.abs(ls['coarse_mean'] - 1)))
            print('       largest 70-bin move: m = %.2f GeV  %+.3f%% +- %.3f%%'
                  ' (%d events)'
                  % (centres[b], 100 * (ls['coarse_mean'][b] - 1),
                     100 * ls['coarse_err'][b], int(ls['coarse_n'][b])))

    if args.out:
        with open(args.out, 'w') as fp:
            json.dump(out, fp, indent=2, sort_keys=True)
        print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
