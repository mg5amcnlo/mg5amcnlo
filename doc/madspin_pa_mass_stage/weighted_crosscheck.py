#!/usr/bin/env python3
"""The per-event normalisation, measured from a *shipped* MadSpin path.

``per_event_weight.py`` measures ``A_e = <w | production event>`` by replaying
the mass draw outside MadSpin.  This measures the same thing from inside it,
with no instrumentation at all, using ``set decay_output = weighted``
(``doc/madspin_sequential_plan.md`` section 13.18): that mode draws **one**
decay configuration per production event and writes it with

    w = w_prod * BR * W / c

so it carries ``W`` instead of normalising it away.  Under PA with
``density_keep_jacobian`` the probe applies the reshuffling jacobian to ``W``
(``_joint_maxwgt_range``), so ``<W | e>`` is exactly the ``A_e`` of the mass
stage up to a decay-side constant.

Binning the written weights in ``sqrt(shat)`` therefore gives the profile of
``A_e/<A>`` -- and the *unweighted* run, which writes one unit-weight event per
production event, gives a flat 1 by construction.  The difference between the
two curves is the bias, measured end to end.

Usage::

    export PATH="$HOME/.pyenv/versions/mg-3.12/bin:$PATH"
    python3 doc/madspin_pa_mass_stage/weighted_crosscheck.py \
        --weighted <events_decayed.lhe.gz from decay_output=weighted> \
        --out <dir>
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import json
import math
import os
import sys

import numpy as np

pjoin = os.path.join

_here = os.path.dirname(os.path.realpath(__file__))
_root = os.path.split(os.path.split(_here)[0])[0]
if _root not in sys.path:
    sys.path.insert(0, _root)
if _here not in sys.path:
    sys.path.insert(0, _here)

import madgraph.various.lhe_parser as lhe_parser   # noqa: E402
from jacobian_analytic import analytic_A, MT       # noqa: E402


def read(path, nmax=None):
    lhe = lhe_parser.EventFile(path)
    sqrts, wgt = [], []
    for i, event in enumerate(lhe):
        if nmax and i >= nmax:
            break
        tot = None
        for p in event:
            if int(p.status) != 1:
                continue
            v = lhe_parser.FourMomentum(p)
            tot = v if tot is None else tot + v
        sqrts.append(math.sqrt(max(0.0, tot.mass_sqr)))
        wgt.append(float(event.wgt))
    try:
        lhe.close()
    except Exception:
        pass
    return np.array(sqrts), np.array(wgt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weighted', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--ztables', default=None)
    parser.add_argument('--nmax', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sqrts, wgt = read(args.weighted, args.nmax or None)
    print('%d weighted events, mean weight %.6g' % (len(wgt), wgt.mean()))

    tabs = None
    if args.ztables:
        with open(args.ztables) as fp:
            zt = json.load(fp)['z_tables']
        keys = sorted(zt, key=lambda k: -int(k.split('_')[0]))
        tabs = (zt[keys[0]], zt[keys[1]])
        norm = float(np.mean([analytic_A(s, tabs) for s in sqrts[:4000]]))

    edges = [346., 346.5, 347, 348, 349, 350, 352, 355, 358, 362, 366, 370,
             375, 380, 390, 400, 425, 450, 500, 600, 800, 1200, 1e9]
    mean = wgt.mean()
    rows = []
    print('\n%-14s %8s %9s %9s %9s' % ('sqrt(shat)', 'events', 'frac %',
                                       '<w>/<w>_all', 'analytic'))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (sqrts >= lo) & (sqrts < hi)
        k = int(sel.sum())
        if k < 5:
            continue
        r = float(wgt[sel].mean() / mean)
        err = float(wgt[sel].std() / math.sqrt(k) / mean)
        ana = None
        if tabs:
            mids = sqrts[sel]
            ana = float(np.mean([analytic_A(s, tabs) for s in
                                 mids[:min(k, 400)]]) / norm)
        rows.append({'lo': lo, 'hi': None if hi > 1e8 else hi, 'events': k,
                     'fraction': k / len(wgt), 'ratio': r, 'error': err,
                     'analytic': ana})
        print('%6.1f-%-7s %8d %8.2f%% %6.4f+-%.4f %9s'
              % (lo, hi if hi < 1e8 else 'inf', k, 100 * k / len(wgt), r, err,
                 '%.4f' % ana if ana else '-'))

    with open(pjoin(args.out, 'weighted_crosscheck.json'), 'w') as fp:
        json.dump({'n': int(len(wgt)), 'mean_weight': float(mean),
                   'by_sqrts': rows}, fp, indent=2, sort_keys=True)
    np.savez_compressed(pjoin(args.out, 'weighted_crosscheck.npz'),
                        sqrts=sqrts.astype(np.float32),
                        wgt=wgt.astype(np.float64))
    print('\nwrote %s' % pjoin(args.out, 'weighted_crosscheck.json'))


if __name__ == '__main__':
    main()
