#!/usr/bin/env python3
"""Offshell only: does ``R = Tr(rho_off)/|M_prod|^2_on`` compensate the
truncated-window effect near threshold, or compound it?

``madspin``/``full`` carry a third factor in the mass-stage weight that ``PA``
does not.  It was measured at ``1.00000 +- 0.0119`` *averaged over a whole
sample* -- but that average is dominated by the 80 % of events above 400 GeV,
and the whole of ``A_e``'s variation lives in the last few GeV above threshold,
where it had never been looked at.

The measurement here needs no new probe.  A MadSpin run with
``mass_normalisation`` on writes ``A_e/<A>`` onto every event weight, so
dividing a ``mass_normalisation_draws = k`` run by its own baseline gives
``A_e`` (with ``R``) per event; ``measure_ae.py --mode madspin`` gives the same
integral with ``R`` set to 1, exactly.  Their ratio is

    <R>_w  =  Int q_e w R / Int q_e w

i.e. ``R`` averaged over the mass sets the stage actually weights -- which is
the only average of ``R`` that matters here.

Usage::

    python3 doc/madspin_ae_normalisation/ratio_R.py \
        --baseline <b_madspin>/events_decayed.lhe.gz \
        --corrected <p_madspin>/events_decayed.lhe.gz \
        --ae data/ae_madspin_R1.npz --ref 0.956832 \
        --out data/ratio_R.json
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


def weights(path):
    out = []
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as fh:
        need = False
        for line in fh:
            if line.startswith('<event'):
                need = True
                continue
            if need:
                out.append(float(line.split()[2]))
                need = False
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--corrected', required=True)
    ap.add_argument('--ae', required=True)
    ap.add_argument('--ref', type=float, required=True)
    ap.add_argument('--draws', type=int, default=24)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    wb, wc = weights(args.baseline), weights(args.corrected)
    assert len(wb) == len(wc), (len(wb), len(wc))
    A_mc = wc / wb * args.ref            # A_e WITH R, per event, by k draws
    d = np.load(args.ae)
    A_r1, sqrts, rel = d['A'], d['sqrts'], d['rel_sd']
    assert len(A_r1) == len(A_mc)

    ratio = A_mc / A_r1
    # per-event noise on the ratio: the MC estimator's, since A_r1 is exact
    noise = rel / math.sqrt(args.draws)
    print('A_e (R=1, exact)     mean %.6f  rel sd %.4f' %
          (A_r1.mean(), A_r1.std() / A_r1.mean()))
    print('A_e (with R, k=%d MC) mean %.6f  rel sd %.4f  '
          '(of which MC noise %.4f)'
          % (args.draws, A_mc.mean(), A_mc.std() / A_mc.mean(),
             math.sqrt((noise ** 2).mean())))
    print('<R>_w over the whole sample: %.6f +- %.6f'
          % (ratio.mean(), ratio.std() / math.sqrt(len(ratio))))
    print('\n<R>_w sliced in sqrt(shat) -- the question the average could not '
          'answer:')
    print('  %-14s %8s   %-22s %-10s %s'
          % ('slice [GeV]', 'events', '<R>_w', 'A_e(R=1)/plateau',
             'A_e(R)/plateau'))
    rows = []
    plateau_r1 = A_r1[sqrts > 800].mean()
    plateau_mc = A_mc[sqrts > 800].mean()
    for lo, hi in zip(SQRTS_EDGES[:-1], SQRTS_EDGES[1:]):
        sel = (sqrts >= lo) & (sqrts < hi)
        k = int(sel.sum())
        if not k:
            continue
        r = ratio[sel]
        err = float(np.sqrt((noise[sel] ** 2).sum()) / k)
        rows.append(dict(lo=lo, hi=None if hi > 1e8 else hi, n=k,
                         R=float(r.mean()), R_err=err,
                         a_r1=float(A_r1[sel].mean() / plateau_r1),
                         a_mc=float(A_mc[sel].mean() / plateau_mc)))
        print('  %6.0f - %-6s %8d   %.5f +- %.5f      %7.4f     %7.4f'
              % (lo, '%.0f' % hi if hi < 1e8 else 'inf', k, r.mean(), err,
                 rows[-1]['a_r1'], rows[-1]['a_mc']))

    out = dict(draws=args.draws, ref=args.ref,
               A_r1_mean=float(A_r1.mean()), A_mc_mean=float(A_mc.mean()),
               R_mean=float(ratio.mean()),
               R_err=float(ratio.std() / math.sqrt(len(ratio))),
               A_mc_rel_sd=float(A_mc.std() / A_mc.mean()),
               mc_noise=float(math.sqrt((noise ** 2).mean())),
               slices=rows)
    if args.out:
        with open(args.out, 'w') as fp:
            json.dump(out, fp, indent=2, sort_keys=True)
        print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
