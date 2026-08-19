#!/usr/bin/env python3
"""Is ``<w>`` -- the mean of the PA mass-set weight -- the *same constant for
every production event*, or does it vary event to event?

That is the question the whole bound design hangs on.  MadSpin's mass stage is
redraw-until-accept *within* one production event: a rejected mass set is
redrawn against the same production, and the loop only exits on an acceptance.
Such a loop is unbiased only when the conditional normalisation

    A_e = E_{q_e}[ w ]        w = J . prod_k jac_BW_k . prod_k Zhat_k

is the same number for every production event ``e``.  If it is not, the redraw
divides it out and the *relative* population of production events comes out
wrong -- exactly the mechanism of ``doc/madspin_sequential_plan.md`` 13.7b,
where ``<W>`` being a per-event constant is what made redraw-until-accept legal.

The earlier probe (``findings.md``) measured ``<w> = 0.955`` *globally*, pooled
over every trial of a 100 000 event run.  A global mean says nothing about
per-event constancy, and the run's own trials cannot answer it either: the
redraw stops at the first acceptance, so an event contributes ~3.8 trials whose
*last* one is accepted -- a stopped, not an i.i.d., sample.

So this measures it directly and cleanly.  For each production event of a real
``p p > t t~`` sample it draws ``N`` **free** mass sets -- no accept/reject, no
stopping rule -- through the *shipped* functions

    MadSpinInterface._draw_mass_value          (the Breit-Wigner draw + jac_BW)
    MadSpinInterface._production_jacobian_for  (the RAMBO reshuffling jacobian J)
    MadSpinInterface._zhat                     (the tabulated Zhat)

on a shim carrying nothing but the banner, ``BW_cut`` and, optionally, the
``z_tables`` a real MadSpin PA run left in its ``ms_dir`` cache.  Nothing is
re-implemented.

Two per-event means are reported, because two different questions want two
different denominators:

``A_e``   = sum(w)/N with an **infeasible** mass set counted as ``w = 0``.
            An infeasible set (``_production_jacobian_for`` returns 0/-1) makes
            ``_upfront_production`` return None and the chain restart *without*
            reaching the accept/reject, so it is invisible to MadSpin's own
            counters -- but it is part of the event's normalisation, since the
            physical target for event ``e`` is the feasible region only.
            **This is the quantity redraw-until-accept normalises away**, so it
            is the one the bias question is about.
``A_e^f`` = sum(w)/n_feasible, the mean of what the accept/reject actually
            tests.  This is the one that sets the acceptance rate, ``C/A_e^f``.

Per event it also stores ``max(w)``, which is what a *per-event bound* (design
option B) would have to be built from.

Usage::

    export PATH="$HOME/.pyenv/versions/mg-3.12/bin:$PATH"
    python3 doc/madspin_pa_mass_stage/per_event_weight.py \
        --events <production .lhe.gz> --out <dir> \
        --ztables <ms_dir>/max_wgt_sequential_pa --pool 4000
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import collections
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

import madgraph.various.lhe_parser as lhe_parser        # noqa: E402
from MadSpin.interface_madspin import MadSpinInterface  # noqa: E402


class Shim(object):
    """The whole dependency surface of the three shipped methods: a banner
    (pole mass and width), ``options['BW_cut']``, and ``_z_tables``."""

    def __init__(self, banner, bw_cut, z_tables=None):
        self.banner = banner
        self.options = {'BW_cut': bw_cut, 'nb_sigma': 4.5}
        self._z_tables = z_tables or {}

    def _raise_degenerate_weight(self, msg):
        raise RuntimeError(msg)


def load_events(path, nmax):
    lhe = lhe_parser.EventFile(path)
    banner = lhe.get_banner()
    events, pdgs = [], None
    for i, event in enumerate(lhe):
        if i >= nmax:
            break
        if pdgs is None:
            pdgs = [p.pid for p in event if int(p.status) == 1]
        events.append(event)
    try:
        lhe.close()
    except Exception:
        pass
    return events, banner, pdgs


def draw_one(shim, event, slot_to_index, pdgs, zkeys):
    """One mass set for one production event, exactly as ``_upfront_production``
    builds it: the shipped per-slot Breit-Wigner draw with the budget chained
    down, the shipped reshuffling jacobian on the resulting set, and the shipped
    ``Zhat``.

    Returns ``(w, w_nozhat, jac)``, or ``None`` for a mass set the production
    cannot be reshuffled onto (jac 0 or -1 -> ``_upfront_production`` returns
    None and MadSpin restarts the whole set, uncounted).
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
    zhat = 1.0
    for slot in slot_masses:
        zhat *= MadSpinInterface._zhat(shim, zkeys[slot], slot_masses[slot][0])
    w_nz = jac * jac_bw
    return w_nz * zhat, w_nz, jac


def draws_for(sqrts, base, threshold):
    """More draws where the weight has structure.  The reshuffling jacobian is
    flat to a per mil above ~450 GeV (findings.md section 4), so a few hundred
    draws pin ``A_e`` there to better than 0.1 %; near the ``t t~`` threshold the
    per-event spread is large and needs an order of magnitude more."""
    if sqrts < 380:
        return base * 20
    if sqrts < 450:
        return base * 5
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--ztables', default=None,
                        help='a MadSpin max_wgt_sequential_pa cache; without it '
                             'Zhat is 1 (which is what MadSpin itself uses '
                             'during the probe)')
    parser.add_argument('--pool', type=int, default=4000,
                        help='production events measured, in file order')
    parser.add_argument('--draws', type=int, default=400,
                        help='free mass sets per production event, away from '
                             'threshold (scaled up near it)')
    parser.add_argument('--seed', type=int, default=20260819)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)

    z_tables = None
    if args.ztables:
        with open(args.ztables) as fp:
            z_tables = json.load(fp)['z_tables']
        print('Zhat tables: %s' % sorted(z_tables), flush=True)

    events, banner, pdgs = load_events(args.events, args.pool)
    bw_cut = float(banner.get_detail('run_card', 'bwcutoff'))
    slot_to_index = list(range(len(pdgs)))
    zkeys = MadSpinInterface._z_slot_keys(
        [p for p in events[0] if int(p.status) == 1], slot_to_index)
    print('%d production events, finals %s, keys %s, BW_cut %s'
          % (len(events), pdgs, zkeys, bw_cut), flush=True)
    if z_tables is not None and set(zkeys) - set(z_tables):
        raise SystemExit('z_tables %s do not cover the slots %s'
                         % (sorted(z_tables), zkeys))

    shim = Shim(banner, bw_cut, z_tables)

    cols = collections.defaultdict(list)
    t0 = time.time()
    for i, event in enumerate(events):
        n = draws_for(event.sqrts, args.draws, 380)
        s = s2 = snz = sj = 0.0
        wmax = nzmax = jmax = 0.0
        nfeas = 0
        for _ in range(n):
            out = draw_one(shim, event, slot_to_index, pdgs, zkeys)
            if out is None:
                continue
            w, wnz, jac = out
            nfeas += 1
            s += w
            s2 += w * w
            snz += wnz
            sj += jac
            if w > wmax:
                wmax = w
            if wnz > nzmax:
                nzmax = wnz
            if jac > jmax:
                jmax = jac
        cols['sqrts'].append(event.sqrts)
        cols['n'].append(n)
        cols['nfeas'].append(nfeas)
        cols['sum_w'].append(s)
        cols['sum_w2'].append(s2)
        cols['sum_wnz'].append(snz)
        cols['sum_jac'].append(sj)
        cols['max_w'].append(wmax)
        cols['max_wnz'].append(nzmax)
        cols['max_jac'].append(jmax)
        if (i + 1) % 500 == 0:
            print('  %5d / %d events, %.0f s' % (i + 1, len(events),
                                                 time.time() - t0), flush=True)

    arr = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}
    n = arr['n']
    # A_e: infeasible mass sets are part of the normalisation and count as w = 0
    arr['A'] = arr['sum_w'] / n
    arr['A_nozhat'] = arr['sum_wnz'] / n
    # A_e^f: the mean of what the accept/reject tests, which sets acceptance
    arr['Af'] = arr['sum_w'] / np.maximum(arr['nfeas'], 1)
    # statistical error on A_e (over the N free draws, infeasible = 0)
    var = arr['sum_w2'] / n - arr['A'] ** 2
    arr['A_err'] = np.sqrt(np.maximum(var, 0.0) / n)
    arr['feas'] = arr['nfeas'] / n

    total_draws = float(n.sum())
    pooled = float(arr['sum_w'].sum() / total_draws)
    # event-to-event spread, and how much of it is real rather than MC noise
    spread = float(arr['A'].std())
    noise = float(np.sqrt((arr['A_err'] ** 2).mean()))
    true_var = max(spread ** 2 - noise ** 2, 0.0)
    summary = {
        'n_events': int(len(n)),
        'total_draws': total_draws,
        'pooled_mean_w': pooled,
        'A_mean': float(arr['A'].mean()),
        'A_sd': spread,
        'A_sd_stat_only': noise,
        'A_sd_true': math.sqrt(true_var),
        'A_rel_sd_true': math.sqrt(true_var) / float(arr['A'].mean()),
        'A_min': float(arr['A'].min()),
        'A_max': float(arr['A'].max()),
        'A_quantiles': {str(q): float(np.percentile(arr['A'], q))
                        for q in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)},
        'Af_mean': float(arr['Af'].mean()),
        'Af_sd': float(arr['Af'].std()),
        'feas_min': float(arr['feas'].min()),
        'feas_mean': float(arr['feas'].mean()),
        'max_w_global': float(arr['max_w'].max()),
        'zhat': bool(z_tables),
        'draws_base': args.draws,
    }

    # profile against sqrt(shat) -- the variable that drives the tail
    edges = [346, 350, 355, 360, 370, 380, 400, 450, 500, 600, 800, 1200, 1e9]
    prof = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (arr['sqrts'] >= lo) & (arr['sqrts'] < hi)
        if not sel.any():
            continue
        prof.append({
            'lo': lo, 'hi': hi if hi < 1e8 else None,
            'n_events': int(sel.sum()),
            'fraction': float(sel.sum()) / len(n),
            'A': float(arr['sum_w'][sel].sum() / n[sel].sum()),
            'A_sd': float(arr['A'][sel].std()),
            'Af': float(arr['sum_w'][sel].sum() / arr['nfeas'][sel].sum()),
            'feas': float(arr['nfeas'][sel].sum() / n[sel].sum()),
            'mean_jac': float(arr['sum_jac'][sel].sum() / arr['nfeas'][sel].sum()),
            'max_w': float(arr['max_w'][sel].max()),
            'median_max_w': float(np.median(arr['max_w'][sel])),
        })
    summary['by_sqrts'] = prof

    # what a per-event bound would cost (option B) against the global one
    for sigma in (1.1,):
        Ce = sigma * arr['max_w']
        summary['per_event_bound'] = {
            'safety': sigma,
            'eps_median': float(np.median(Ce / arr['Af'])),
            'eps_mean': float((Ce / arr['Af']).mean()),
            'eps_trials_weighted': float(Ce.sum() / arr['Af'].sum()),
            'eps_p95': float(np.percentile(Ce / arr['Af'], 95)),
            'eps_max': float((Ce / arr['Af']).max()),
        }

    np.savez_compressed(pjoin(args.out, 'per_event.npz'),
                        **{k: v.astype(np.float64) for k, v in arr.items()})
    with open(pjoin(args.out, 'per_event.json'), 'w') as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    print('\n%d events, %.3g draws, %.0f s' % (len(n), total_draws,
                                               time.time() - t0))
    print('pooled <w> = %.5f   (findings.md, global: 0.9551)' % pooled)
    print('A_e: mean %.5f  sd %.5f  (stat-only %.5f -> true sd %.5f, %.2f %%)'
          % (arr['A'].mean(), spread, noise, math.sqrt(true_var),
             100 * summary['A_rel_sd_true']))
    print('A_e range [%.4f, %.4f]' % (arr['A'].min(), arr['A'].max()))
    print('wrote %s' % pjoin(args.out, 'per_event.json'))


if __name__ == '__main__':
    main()
