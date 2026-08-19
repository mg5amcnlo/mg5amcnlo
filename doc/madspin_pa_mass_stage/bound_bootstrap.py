#!/usr/bin/env python3
"""How stable is ``maxwgts[0]`` -- the mass stage's bound -- for PA?

``eps_m`` is ``C / <w>`` with ``C`` the bound, so if ``C`` is set by a rare
excursion of the reshuffling jacobian then ``eps_m`` is not a property of the
weight at all: it is a property of how deep into the tail the max-weight scan
happened to look.  This script measures that directly, using the *shipped*
functions -- ``MadSpinInterface._draw_mass_value``,
``MadSpinInterface._production_jacobian_for`` and
``MadSpinInterface._combine_maxwgt`` are called on a shim that carries nothing
but the banner and the two options they read, so no logic is re-implemented
here.

For a given ``(Nevents_for_max_weight, nb_sigma)`` it replays the scan many
times on independently drawn probe events and reports the spread of the bound
and of ``eps_m = C/<w>``.  The pairs to use are the ones MadSpin derives from
the run card (``interface_madspin.py``, ``do_import``)::

    N_weight = max(75, int(3 * nevents**(1/3)))
    nb_sigma = max(4.5, log(nevents, 7.7))

which for 2 000 / 20 000 / 100 000 requested events gives (75, 4.50),
(81, 4.85) and (139, 5.64).

Usage::

    export PATH="$HOME/.pyenv/versions/mg-3.12/bin:$PATH"
    python3 doc/madspin_pa_mass_stage/bound_bootstrap.py \
        --events <production .lhe.gz> --out <dir> --replicas 40
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

pjoin = os.path.join

_here = os.path.dirname(os.path.realpath(__file__))
_root = os.path.split(os.path.split(_here)[0])[0]
if _root not in sys.path:
    sys.path.insert(0, _root)

import madgraph.various.lhe_parser as lhe_parser  # noqa: E402
from MadSpin.interface_madspin import MadSpinInterface  # noqa: E402


class Shim(object):
    """The two shipped methods' whole dependency surface: a banner (pole mass
    and width) and ``options['BW_cut']`` / ``options['nb_sigma']``."""

    def __init__(self, banner, bw_cut, nb_sigma):
        self.banner = banner
        self.options = {'BW_cut': bw_cut, 'nb_sigma': nb_sigma}

    def _raise_degenerate_weight(self, msg):
        raise RuntimeError(msg)


def load_events(path, nmax):
    lhe = lhe_parser.EventFile(path)
    banner = lhe.get_banner()
    events, pdgs = [], None
    for i, event in enumerate(lhe):
        if i >= nmax:
            break
        finals = [p for p in event if int(p.status) == 1]
        if pdgs is None:
            pdgs = [p.pid for p in finals]
        events.append(event)
    try:
        lhe.close()
    except Exception:
        pass
    return events, banner, pdgs


def draw_chain(shim, event, slot_to_index, pdgs):
    """One mass set for one production event: the shipped ``_draw_mass_value``
    per slot (budget-chained, exactly as ``_upfront_production`` does it) and
    the shipped ``_production_jacobian_for`` on the result.

    Returns ``(w_raw, jac_prod)`` with ``w_raw = jac_prod * prod(jac_bw)`` --
    the mass-set weight of PA up to ``Z_hat``, which is 1 to a few per mil
    (measured: see summary.json ``zhat_product``).  ``None`` for a mass set the
    production cannot be reshuffled onto.
    """
    budget = event.sqrts
    slot_masses, jac_bw = {}, 1.0
    for slot, pdg in enumerate(pdgs):
        mass, info, jbw = MadSpinInterface._draw_mass_value(shim, pdg, budget)
        slot_masses[slot] = (mass, info)
        jac_bw *= jbw
        budget -= mass
    jac = MadSpinInterface._production_jacobian_for(event, slot_to_index,
                                                    slot_masses)
    if jac in (0, -1):
        return None
    return jac * jac_bw, jac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--pool', type=int, default=4000,
                        help='production events read into memory')
    parser.add_argument('--replicas', type=int, default=40)
    parser.add_argument('--ps-point', type=int, default=500,
                        help='max_weight_ps_point (MadSpin default 500)')
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)

    events, banner, pdgs = load_events(args.events, args.pool)
    bw_cut = float(banner.get_detail('run_card', 'bwcutoff'))
    slot_to_index = list(range(len(pdgs)))
    print('%d production events, finals %s, BW_cut %s'
          % (len(events), pdgs, bw_cut), flush=True)

    # (label, Nevents_for_max_weight, nb_sigma) exactly as do_import derives
    # them from the run card's nevents
    settings = []
    for nevents in (2000, 20000, 100000, 1000000):
        settings.append(('nevents=%d' % nevents,
                         max(75, int(3 * nevents ** (1 / 3))),
                         max(4.5, math.log(nevents, 7.7))))

    shim0 = Shim(banner, bw_cut, 4.5)

    # One big pooled sample of the weight, for <w> and for the tail quantiles.
    t0 = time.time()
    pooled_w, pooled_jac, pooled_sqrts = [], [], []
    n_infeasible = 0
    for event in events:
        for _ in range(20):
            out = draw_chain(shim0, event, slot_to_index, pdgs)
            if out is None:
                n_infeasible += 1
                continue
            pooled_w.append(out[0])
            pooled_jac.append(out[1])
            pooled_sqrts.append(event.sqrts)
    pooled_w = np.array(pooled_w)
    pooled_jac = np.array(pooled_jac)
    pooled_sqrts = np.array(pooled_sqrts)
    print('pooled sample: %d mass sets in %.0f s, <w> = %.5f, <J> = %.5f'
          % (len(pooled_w), time.time() - t0, pooled_w.mean(),
             pooled_jac.mean()), flush=True)

    result = {
        'n_pooled': int(len(pooled_w)),
        'n_infeasible': n_infeasible,
        'mean_w': float(pooled_w.mean()),
        'mean_jac': float(pooled_jac.mean()),
        'jac_quantiles': {str(q): float(np.percentile(pooled_jac, q))
                          for q in (50, 90, 99, 99.9, 99.99)},
        'jac_max': float(pooled_jac.max()),
        'ps_point': args.ps_point,
        'replicas': {},
    }

    for label, n_probe, nb_sigma in settings:
        shim = Shim(banner, bw_cut, nb_sigma)
        bounds, epss = [], []
        t0 = time.time()
        for _ in range(args.replicas):
            per_event = []
            for event in random.sample(events, min(n_probe, len(events))):
                best = 0.0
                for _ in range(args.ps_point):
                    out = draw_chain(shim, event, slot_to_index, pdgs)
                    if out is not None:
                        best = max(best, out[0])
                per_event.append(best)
            bound = MadSpinInterface._combine_maxwgt(shim, per_event)
            bounds.append(bound)
            epss.append(bound / pooled_w.mean())
        bounds = np.array(bounds)
        epss = np.array(epss)
        result['replicas'][label] = {
            'Nevents_for_max_weight': n_probe,
            'nb_sigma': nb_sigma,
            'bound_mean': float(bounds.mean()),
            'bound_sd': float(bounds.std()),
            'bound_min': float(bounds.min()),
            'bound_max': float(bounds.max()),
            'eps_m_mean': float(epss.mean()),
            'eps_m_sd': float(epss.std()),
            'eps_m_min': float(epss.min()),
            'eps_m_max': float(epss.max()),
            'bounds': [float(b) for b in bounds],
        }
        print('%-16s N=%3d sigma=%.2f  C = %.2f +- %.2f  [%.2f, %.2f]  '
              'eps_m = %.2f +- %.2f  (%.0f s)'
              % (label, n_probe, nb_sigma, bounds.mean(), bounds.std(),
                 bounds.min(), bounds.max(), epss.mean(), epss.std(),
                 time.time() - t0), flush=True)

    with open(pjoin(args.out, 'bound_bootstrap.json'), 'w') as fp:
        json.dump(result, fp, indent=2, sort_keys=True)
    np.savez_compressed(pjoin(args.out, 'bound_bootstrap_pool.npz'),
                        w=pooled_w.astype(np.float32),
                        jac=pooled_jac.astype(np.float32),
                        sqrts=pooled_sqrts.astype(np.float32))
    print('wrote %s' % pjoin(args.out, 'bound_bootstrap.json'))


if __name__ == '__main__':
    main()
