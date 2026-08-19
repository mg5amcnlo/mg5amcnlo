#!/usr/bin/env python3
"""Is the PA reshuffling jacobian analytic?  For a 2 -> 2 production, yes --
exactly, and that is what makes a per-event bound (and a per-event
normalisation) free rather than expensive.

``Event.mass_shuffle`` implements RAMBO eq. 4.2/4.9: every spatial momentum in
the production CM is scaled by a common ``chi`` fixed by
``sum_i sqrt(m_i'^2 + chi^2 |p_i|^2) = sqrt(shat)``, and

    J = chi^(3n-3) . prod_i (E_i/E_i') . (sum_i |p_i|^2/E_i)
                                       / (sum_i |p_i'|^2/E_i')

For ``n = 2`` the two momenta are back to back with the same modulus, so
``chi = |p'|/|p|`` and every factor collapses:

    J = chi^3 . (E1 E2)/(E1' E2') . [|p|^2 sqrt(s)/(E1 E2)]
                                  / [|p'|^2 sqrt(s)/(E1' E2')]
      = chi^3 . |p|^2/|p'|^2 = chi
      = |p'| / |p|
      = lambda^(1/2)(s, m1'^2, m2'^2) / lambda^(1/2)(s, m1^2, m2^2)

i.e. the 2-body phase-space volume ratio, and nothing else.  That is the
``1/beta_t`` divergence in closed form: on shell at threshold ``|p| -> 0``.

Consequences this script checks numerically:

1. ``J = |p'|/|p|`` for 2 -> 2, to machine precision, for random ``sqrt(shat)``,
   random mass sets and random orientations.
2. For ``n >= 3`` it is **not** a function of the mass set alone -- the same
   masses at the same ``sqrt(shat)`` give different ``J`` for different momentum
   configurations, so nothing analytic is available there.
3. For 2 -> 2 the per-event maximum over the Breit-Wigner window,

       J_max(e) = |p'|(sqrt(s), m_lo, m_lo) / |p|(sqrt(s), m_pole, m_pole)

   is exact and costs two square roots -- no probing, no Monte Carlo.  And the
   per-event normalisation ``A_e = E_q[w]`` is a two-dimensional quadrature over
   the same window.  Both are compared against the Monte Carlo measurement of
   ``per_event_weight.py`` by ``analyse_per_event.py``.

Usage::

    export PATH="$HOME/.pyenv/versions/mg-3.12/bin:$PATH"
    python3 doc/madspin_pa_mass_stage/jacobian_analytic.py [--json out.json]
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import json
import math
import os
import random
import sys

_here = os.path.dirname(os.path.realpath(__file__))
_root = os.path.split(os.path.split(_here)[0])[0]
if _root not in sys.path:
    sys.path.insert(0, _root)

from madgraph.various.lhe_parser import Event, FourMomentum  # noqa: E402

MT = 173.0
WT = 1.4915


def pmag(sqrts, m1, m2):
    """|p| of a two-body final state: lambda^(1/2)(s,m1^2,m2^2)/(2 sqrt(s))."""
    s = sqrts * sqrts
    lam = (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)
    return math.sqrt(max(0.0, lam)) / (2 * sqrts)


def two_body(sqrts, m, orientation):
    """A back-to-back pair of mass ``m`` at ``sqrt(s)``, pointing anywhere."""
    p = pmag(sqrts, m, m)
    ct, ph = orientation
    st = math.sqrt(max(0.0, 1 - ct * ct))
    v = (p * st * math.cos(ph), p * st * math.sin(ph), p * ct)
    e = math.sqrt(p * p + m * m)
    return [FourMomentum(e, *v), FourMomentum(e, -v[0], -v[1], -v[2])], p


def check_two_to_two(ntrial=2000, seed=7):
    random.seed(seed)
    worst, worst_case, done = 0.0, None, 0
    for _ in range(ntrial):
        sqrts = math.exp(random.uniform(math.log(347.0), math.log(4000.0)))
        m1 = random.uniform(MT - 15 * WT, MT + 15 * WT)
        m2 = random.uniform(MT - 15 * WT, MT + 15 * WT)
        if m1 + m2 >= sqrts - 1e-3:
            continue
        mom, p = two_body(sqrts, MT, (random.uniform(-1, 1),
                                      random.uniform(0, 2 * math.pi)))
        if p <= 0:
            continue
        _, jac = Event.mass_shuffle(mom, sqrts, [m1, m2])
        analytic = pmag(sqrts, m1, m2) / p
        if analytic <= 0:
            continue
        done += 1
        dev = abs(jac / analytic - 1)
        if dev > worst:
            worst, worst_case = dev, (sqrts, m1, m2, jac, analytic)
    return {'trials': done, 'max_rel_dev': worst,
            'worst_case': worst_case}


def check_three_body(seed=101, nconfig=8, sqrts=600.0,
                     masses=(MT, MT, 0.0), new_masses=(160.0, 190.0, 0.0)):
    """Same masses, same sqrt(shat), different momentum configurations."""
    random.seed(seed)
    out = []
    while len(out) < nconfig:
        legs = []
        for m in masses:
            ct = random.uniform(-1, 1)
            st = math.sqrt(1 - ct * ct)
            ph = random.uniform(0, 2 * math.pi)
            pm = random.uniform(20, 220)
            legs.append((pm * st * math.cos(ph), pm * st * math.sin(ph), pm * ct))
        tot = [sum(v[i] for v in legs) / len(legs) for i in range(3)]
        legs = [(v[0] - tot[0], v[1] - tot[1], v[2] - tot[2]) for v in legs]
        mom = [FourMomentum(math.sqrt(m * m + sum(c * c for c in v)), *v)
               for m, v in zip(masses, legs)]
        e_tot = sum(p.E for p in mom)
        if e_tot >= sqrts:
            continue
        # bring the configuration to exactly sqrt(shat) at the ORIGINAL masses
        mom, _ = Event.mass_shuffle(mom, e_tot, list(masses), new_sqrts=sqrts)
        probe = [FourMomentum(p.E, p.px, p.py, p.pz) for p in mom]
        _, jac = Event.mass_shuffle(probe, sqrts, list(new_masses))
        out.append({'sum_p': sum(math.sqrt(p.px ** 2 + p.py ** 2 + p.pz ** 2)
                                 for p in mom),
                    'jac': jac})
    js = [o['jac'] for o in out]
    return {'sqrts': sqrts, 'masses': list(masses),
            'new_masses': list(new_masses), 'configs': out,
            'min': min(js), 'max': max(js), 'spread': max(js) / min(js)}


def analytic_bound(sqrts, pole=MT, width=WT, bw_cut=15):
    """The exact per-event maximum of J over the Breit-Wigner window, 2 -> 2.

    J = |p'|/|p| grows monotonically as the masses go down, so the maximum sits
    at the bottom of the window for both -- capped by the budget, which for the
    lower edge never binds.
    """
    m_lo = pole - bw_cut * width
    denom = pmag(sqrts, pole, pole)
    if denom <= 0:
        return float('inf')
    return pmag(sqrts, m_lo, m_lo) / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default=None)
    args = parser.parse_args()

    two = check_two_to_two()
    print('2 -> 2:  J vs |p\'|/|p| over %d random (sqrt(shat), m1, m2, angles)'
          % two['trials'])
    print('         max relative deviation %.3g' % two['max_rel_dev'])

    three = check_three_body()
    print('\n3 body:  same masses %s -> %s at sqrt(shat) = %.0f, %d random '
          'configurations' % (three['masses'], three['new_masses'],
                              three['sqrts'], len(three['configs'])))
    for cfg in three['configs']:
        print('         sum|p| = %8.2f   J = %.6f' % (cfg['sum_p'], cfg['jac']))
    print('         J spans %.4f - %.4f (a factor %.3f) at fixed masses'
          % (three['min'], three['max'], three['spread']))

    print('\n2 -> 2 analytic per-event bound J_max(e), t t~, BW_cut 15:')
    rows = []
    for sqrts in (346.5, 348, 350, 355, 360, 370, 380, 400, 450, 500, 700, 1000):
        rows.append((sqrts, analytic_bound(sqrts)))
        print('         sqrt(shat) = %7.1f GeV   J_max = %9.3f'
              % (sqrts, rows[-1][1]))

    if args.json:
        with open(args.json, 'w') as fp:
            json.dump({'two_to_two': two, 'three_body': three,
                       'analytic_bound': rows}, fp, indent=2, sort_keys=True)
        print('\nwrote %s' % args.json)


if __name__ == '__main__':
    main()
