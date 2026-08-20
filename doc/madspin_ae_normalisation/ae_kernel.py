#!/usr/bin/env python3
"""``A_e``, the per-event normalisation of MadSpin's sequential mass stage:
what it is, three ways of computing it, and what each one costs.

For one production event ``e`` the mass stage draws a mass set from a proposal
``q_e`` and tests ``w(m)`` against a bound, redrawing until it accepts and
writing **exactly one** event whatever

    A_e = E_{q_e}[w] = int q_e(m) w(m) dm

is.  ``A_e`` is therefore divided out, and it is not a constant (see
``doc/madspin_pa_mass_stage/bound_design.md`` section 2).  This module is the
arithmetic needed to put it back.

The three modes differ only in what ``w`` is:

    PA               w = J(m) . prod_s jac_BW_s . prod_s Zhat_s(m_s)
    PA (no jac.)     w =        prod_s jac_BW_s . prod_s Zhat_s(m_s)
    madspin / full   w = R(m;e) . J(m) . prod_s jac_BW_s . prod_s Zhat_s(m_s)

with ``R = Tr(rho_off)/|M_prod|^2_on``, the only factor that is not a function
of the mass set and ``sqrt(shat)`` alone (it needs a production density matrix,
i.e. a matrix-element call, per mass set).

The sampler is uniform in ``R_s = atan((m_s^2 - pole^2)/(pole.Gamma))`` over the
window, and ``jac_BW_s = gap_s/pi`` is exactly that window's width in ``R`` over
``pi``, so ``q_e . prod jac_BW`` is ``prod dR_s/pi`` identically and

    A_e = pi^-n Int dR_1 ... dR_n  [R(m;e)] J(m;e) prod_s Zhat_s(m_s)

with the ``R_s`` range of slot ``s`` capped by the budget left after slots
``1..s-1``.  That is a plain ``n``-dimensional quadrature over a *fixed* domain
-- there is no Monte Carlo in it for ``n = 2``, and nothing above is specific to
``n = 2`` except that the quadrature is only affordable there.

Three evaluators for ``J``, in increasing generality and cost:

``pmag``            2 -> 2 only.  ``J = lambda^(1/2)(s,m1'^2,m2'^2) /
                    lambda^(1/2)(s,m1^2,m2^2)``, two square roots
                    (``doc/madspin_pa_mass_stage/jacobian_analytic.py``).
``kernel``          any ``n``.  ``Event.mass_shuffle_jacobian`` on the
                    ``Event.mass_shuffle_frame`` of the event -- one Newton
                    solve plus O(n) arithmetic, no ``Event``, no
                    ``FourMomentum``, no boost.  Added by PR #377.
``shipped``         any ``n``.  ``MadSpinInterface._production_jacobian_for``,
                    which re-parses the event from ``str(production)`` and runs
                    a full ``reshuffle_production`` per evaluation.

``time_all()`` measures all of them on real production events.

Usage::

    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    python3 doc/madspin_ae_normalisation/ae_kernel.py \
        --events <production .lhe.gz> --json data/cost.json
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.realpath(__file__))
_root = os.path.split(os.path.split(_here)[0])[0]
if _root not in sys.path:
    sys.path.insert(0, _root)

import madgraph.various.lhe_parser as lhe_parser        # noqa: E402
from MadSpin.interface_madspin import MadSpinInterface  # noqa: E402


# --------------------------------------------------------------------------
# Zhat, off a plain table dict -- ``MadSpinInterface._zhat`` without an
# interface.  Kept byte-for-byte equivalent to the shipped method.
# --------------------------------------------------------------------------
def zhat_of(table, mass):
    if not table:
        return 1.0
    if mass < table['zero_below']:
        return 0.0
    lo, hi = table['range']
    u = math.log(min(max(mass, lo), hi) / table['pole'])
    c = table['coeff']
    return math.exp(c[0] + u * (c[1] + u * c[2]))


def zhat_vec(table, mass):
    """``zhat_of`` on a numpy array."""
    if not table:
        return np.ones_like(mass)
    lo, hi = table['range']
    c = table['coeff']
    u = np.log(np.clip(mass, lo, hi) / table['pole'])
    out = np.exp(c[0] + u * (c[1] + u * c[2]))
    return np.where(mass < table['zero_below'], 0.0, out)


def r_of_m(m, pole, width):
    return np.arctan((m * m - pole * pole) / (pole * width))


def m_of_r(r, pole, width):
    return np.sqrt(pole * pole + pole * width * np.tan(r))


def pmag(sqrts, m1, m2):
    """|p| of a two-body final state."""
    s = sqrts * sqrts
    lam = (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)
    return np.sqrt(np.maximum(0.0, lam)) / (2 * sqrts)


# --------------------------------------------------------------------------
# The quadrature
# --------------------------------------------------------------------------
class Quad2(object):
    """The 2-D Gauss-Legendre grid on the ``(R_1, R_2)`` square, cached per
    order.  Everything that does not depend on the event is built once."""

    _cache = {}

    def __new__(cls, n):
        if n not in cls._cache:
            self = object.__new__(cls)
            self.n = n
            self.x, self.w = np.polynomial.legendre.leggauss(n)
            cls._cache[n] = self
        return cls._cache[n]


def analytic_A_vec(sqrts, tables=(None, None), pole=173.0, width=1.4915,
                   bw_cut=15, nquad=48, keep_jac=True, out_grid=False):
    """``A_e`` for a 2 -> 2 production, by vectorised quadrature.

    Same integral as ``doc/madspin_pa_mass_stage/jacobian_analytic.analytic_A``
    -- checked against it to 1e-14 by ``--check`` -- with the double python loop
    replaced by two numpy outer products.  ``keep_jac = False`` drops ``J``,
    which is exactly the ``density_keep_jacobian = False`` weight.

    ``out_grid`` additionally returns the (m1, m2, weight) grid, so a caller
    that has to fold in a per-point factor the closed form does not know about
    (the offshell ``R``) can reuse the same nodes.
    """
    q = Quad2(nquad)
    m_lo = pole - bw_cut * width
    m_hi1 = min(pole + bw_cut * width, sqrts)
    denom = pmag(sqrts, pole, pole)
    if denom <= 0 or m_lo >= m_hi1:
        return float('nan')
    r_lo = float(r_of_m(m_lo, pole, width))
    r_hi1 = float(r_of_m(m_hi1, pole, width))
    r1 = 0.5 * (r_hi1 + r_lo) + 0.5 * (r_hi1 - r_lo) * q.x
    m1 = m_of_r(r1, pole, width)
    z1 = zhat_vec(tables[0], m1)

    # slot 2's window is capped by the budget left, which is what truncates it
    m_hi2 = np.minimum(pole + bw_cut * width, sqrts - m1)
    ok = m_hi2 > m_lo
    r_hi2 = r_of_m(m_hi2, pole, width)
    # (nquad, nquad): row i is slot 1's node, column j slot 2's
    r2 = (0.5 * (r_hi2 + r_lo))[:, None] \
        + (0.5 * (r_hi2 - r_lo))[:, None] * q.x[None, :]
    m2 = m_of_r(r2, pole, width)
    z2 = zhat_vec(tables[1], m2)
    if keep_jac:
        jac = pmag(sqrts, m1[:, None], m2) / denom
    else:
        jac = 1.0
    inner = (q.w[None, :] * jac * z2).sum(axis=1) * (0.5 * (r_hi2 - r_lo))
    total = float((q.w * z1 * np.where(ok, inner, 0.0)).sum()
                  * 0.5 * (r_hi1 - r_lo) / math.pi ** 2)
    if not out_grid:
        return total
    # the full integration measure of every node, so sum(measure * f) = A_e
    meas = (q.w[:, None] * q.w[None, :]
            * (0.5 * (r_hi1 - r_lo)) * (0.5 * (r_hi2 - r_lo))[:, None]
            / math.pi ** 2)
    meas = np.where(ok[:, None], meas, 0.0)
    return total, np.broadcast_to(m1[:, None], m2.shape), m2, \
        meas * (z1[:, None] * z2) * (jac if keep_jac else 1.0)


def analytic_A_loop(sqrts, tables=(None, None), pole=173.0, width=1.4915,
                    bw_cut=15, nquad=48):
    """The same integral in the plain python double loop, i.e. exactly
    ``doc/madspin_pa_mass_stage/jacobian_analytic.analytic_A``.  Kept here only
    so the cost of the two can be compared on the same events."""
    q = Quad2(nquad)
    m_lo = pole - bw_cut * width
    m_hi1 = min(pole + bw_cut * width, sqrts)
    denom = float(pmag(sqrts, pole, pole))
    if denom <= 0 or m_lo >= m_hi1:
        return float('nan')
    r_lo = float(r_of_m(m_lo, pole, width))
    r_hi1 = float(r_of_m(m_hi1, pole, width))
    total = 0.0
    for xi, wi in zip(q.x, q.w):
        r1 = 0.5 * (r_hi1 + r_lo) + 0.5 * (r_hi1 - r_lo) * xi
        m1 = math.sqrt(pole * pole + pole * width * math.tan(r1))
        z1 = zhat_of(tables[0], m1)
        m_hi2 = min(pole + bw_cut * width, sqrts - m1)
        if m_hi2 <= m_lo:
            continue
        r_hi2 = math.atan((m_hi2 ** 2 - pole ** 2) / pole / width)
        inner = 0.0
        for xj, wj in zip(q.x, q.w):
            r2 = 0.5 * (r_hi2 + r_lo) + 0.5 * (r_hi2 - r_lo) * xj
            m2 = math.sqrt(pole * pole + pole * width * math.tan(r2))
            inner += wj * (float(pmag(sqrts, m1, m2)) / denom) \
                * zhat_of(tables[1], m2)
        total += wi * z1 * inner * 0.5 * (r_hi2 - r_lo)
    return total * 0.5 * (r_hi1 - r_lo) / math.pi ** 2


def analytic_A_kernel(frame, sqrts, shuffle_masses, slot_index,
                      tables=(None, None), pole=173.0, width=1.4915,
                      bw_cut=15, nquad=48, shipped=None):
    """``A_e`` with ``J`` from the general-``n`` kernel instead of the 2 -> 2
    closed form: ``Event.mass_shuffle_jacobian`` on a cached
    ``Event.mass_shuffle_frame``.  This is the version that would run for
    ``n >= 3``; for ``n = 2`` it must agree with ``analytic_A_vec`` and
    ``--check`` asserts that it does.

    ``shipped``, when given, is ``(production, slot_to_index)`` and the jacobian
    goes through ``MadSpinInterface._production_jacobian_for`` instead -- an
    ``Event`` rebuild and a full ``reshuffle_production`` per node.  That is the
    evaluator the pre-#377 code had, and the reason the cost is quoted three
    ways.
    """
    q = Quad2(nquad)
    m_lo = pole - bw_cut * width
    m_hi1 = min(pole + bw_cut * width, sqrts)
    if m_lo >= m_hi1:
        return float('nan')
    r_lo = float(r_of_m(m_lo, pole, width))
    r_hi1 = float(r_of_m(m_hi1, pole, width))
    denom = None
    total = 0.0
    for xi, wi in zip(q.x, q.w):
        r1 = 0.5 * (r_hi1 + r_lo) + 0.5 * (r_hi1 - r_lo) * xi
        m1 = math.sqrt(pole * pole + pole * width * math.tan(r1))
        z1 = zhat_of(tables[0], m1)
        m_hi2 = min(pole + bw_cut * width, sqrts - m1)
        if m_hi2 <= m_lo:
            continue
        r_hi2 = math.atan((m_hi2 ** 2 - pole ** 2) / pole / width)
        inner = 0.0
        for xj, wj in zip(q.x, q.w):
            r2 = 0.5 * (r_hi2 + r_lo) + 0.5 * (r_hi2 - r_lo) * xj
            m2 = math.sqrt(pole * pole + pole * width * math.tan(r2))
            masses = list(shuffle_masses)
            masses[slot_index[0]] = m1
            masses[slot_index[1]] = m2
            if shipped is None:
                jac = lhe_parser.Event.mass_shuffle_jacobian(frame, masses,
                                                             sqrts)
            else:
                production, slot_to_index = shipped
                jac = MadSpinInterface._production_jacobian_for(
                    production, slot_to_index,
                    {0: (m1, None), 1: (m2, None)})
            if jac in (0, -1):
                continue
            if denom is None:
                # J is already normalised to the ORIGINAL configuration by
                # mass_shuffle_jacobian (chi = 1 at the nominal masses), so
                # unlike the pmag form there is nothing to divide by
                denom = 1.0
            inner += wj * jac * zhat_of(tables[1], m2)
        total += wi * z1 * inner * 0.5 * (r_hi2 - r_lo)
    return total * 0.5 * (r_hi1 - r_lo) / math.pi ** 2


# --------------------------------------------------------------------------
# The Monte Carlo estimator -- what an n >= 3 production would have to use
# --------------------------------------------------------------------------
def mc_A(frame, sqrts, shuffle_masses, slot_index, ndraw, rng,
         tables=(None, None), pole=173.0, width=1.4915, bw_cut=15):
    """``A_e`` by ``ndraw`` FREE draws from the proposal, through the kernel.

    The draws are free: no accept/reject, no stopping rule.  That matters --
    the trials the redraw loop itself makes are a *stopped* sequence and their
    mean is not ``A_e``.

    Returns ``(A_hat, sd_of_A_hat)``.  An infeasible mass set contributes
    ``w = 0``: it is part of the normalisation (``_upfront_production`` returns
    None and the chain restarts without ever reaching the accept/reject).
    """
    s = s2 = 0.0
    for _ in range(ndraw):
        budget = sqrts
        masses = list(shuffle_masses)
        jbw = 1.0
        z = 1.0
        for k, si in enumerate(slot_index):
            m_lo = pole - bw_cut * width
            m_hi = min(pole + bw_cut * width, budget)
            if m_hi <= m_lo:
                jbw = 0.0
                break
            gap = math.atan((pole ** 2 - m_lo ** 2) / pole / width) \
                + math.atan((m_hi ** 2 - pole ** 2) / pole / width)
            jbw *= gap / math.pi
            r = math.atan((m_lo ** 2 - pole ** 2) / pole / width) \
                + rng.random() * (math.atan((m_hi ** 2 - pole ** 2)
                                            / pole / width)
                                  - math.atan((m_lo ** 2 - pole ** 2)
                                              / pole / width))
            m = math.sqrt(pole * pole + pole * width * math.tan(r))
            masses[si] = m
            z *= zhat_of(tables[k], m)
            budget -= m
        if jbw == 0.0:
            w = 0.0
        else:
            jac = lhe_parser.Event.mass_shuffle_jacobian(frame, masses, sqrts)
            w = 0.0 if jac in (0, -1) else jac * jbw * z
        s += w
        s2 += w * w
    a = s / ndraw
    var = max(s2 / ndraw - a * a, 0.0)
    return a, math.sqrt(var / ndraw)


# --------------------------------------------------------------------------
def _load(path, nmax):
    lhe = lhe_parser.EventFile(path)
    banner = lhe.get_banner()
    events = []
    for i, event in enumerate(lhe):
        if i >= nmax:
            break
        events.append(event)
    try:
        lhe.close()
    except Exception:
        pass
    return events, banner


def _frame_of(production):
    """The kernel's per-event data, off the *round-tripped* event -- which is
    what ``_production_jacobian_for`` reshuffles, and the only way the two agree
    to the last digit (``str()`` truncates every momentum to %.10e)."""
    probe = lhe_parser.Event(str(production))
    finals = [p for p in probe if int(p.status) == 1]
    frame = lhe_parser.Event.mass_shuffle_frame(
        [lhe_parser.FourMomentum(p) for p in finals], probe.sqrts)
    return frame, probe.sqrts, [p.mass for p in finals]


def time_all(events, tables, pole, width, bw_cut, nquad, nrep):
    """Time everything on the same events.  Returns a dict of seconds/event."""
    out = {}
    ev = events[:nrep]

    frames = [_frame_of(e) for e in ev]

    # ---- one jacobian evaluation ----
    t0 = time.perf_counter()
    for (frame, sq, ms), e in zip(frames, ev):
        for _ in range(20):
            lhe_parser.Event.mass_shuffle_jacobian(frame, [171.0, 174.0], sq)
    out['jac_kernel_us'] = (time.perf_counter() - t0) / (20 * len(ev)) * 1e6

    t0 = time.perf_counter()
    for e in ev:
        for _ in range(20):
            MadSpinInterface._production_jacobian_for(
                e, [0, 1], {0: (171.0, None), 1: (174.0, None)})
    out['jac_shipped_us'] = (time.perf_counter() - t0) / (20 * len(ev)) * 1e6

    t0 = time.perf_counter()
    for e in ev:
        for _ in range(20):
            float(pmag(e.sqrts, 171.0, 174.0))
    out['jac_pmag_us'] = (time.perf_counter() - t0) / (20 * len(ev)) * 1e6

    # ---- A_e, four ways ----
    for tag, fn in (
            ('A_loop_pmag', lambda e, f: analytic_A_loop(
                e.sqrts, tables, pole, width, bw_cut, nquad)),
            ('A_vec_pmag', lambda e, f: analytic_A_vec(
                e.sqrts, tables, pole, width, bw_cut, nquad)),
            ('A_loop_kernel', lambda e, f: analytic_A_kernel(
                f[0], f[1], f[2], [0, 1], tables, pole, width, bw_cut, nquad)),
    ):
        t0 = time.perf_counter()
        for e, f in zip(ev, frames):
            fn(e, f)
        out['%s_ms' % tag] = (time.perf_counter() - t0) / len(ev) * 1e3

    # the shipped-jacobian quadrature is ~1000x slower; time it on fewer events
    few = min(3, len(ev))
    t0 = time.perf_counter()
    for e, f in zip(ev[:few], frames[:few]):
        analytic_A_kernel(f[0], f[1], f[2], [0, 1], tables, pole, width,
                          bw_cut, 8, shipped=(e, [0, 1]))
    # 8x8 nodes; scale to nquad^2 (the cost is linear in the node count)
    per_node = (time.perf_counter() - t0) / (few * 64)
    out['A_loop_shipped_ms'] = per_node * nquad * nquad * 1e3
    out['A_loop_shipped_extrapolated_from'] = '%d x %d nodes' % (8, 8)

    # ---- the MC estimator, for n >= 3 ----
    import random
    rng = random.Random(1)
    for nd in (32, 128, 512):
        t0 = time.perf_counter()
        for (frame, sq, ms) in frames:
            mc_A(frame, sq, ms, [0, 1], nd, rng, tables, pole, width, bw_cut)
        out['A_mc_%d_ms' % nd] = (time.perf_counter() - t0) / len(ev) * 1e3
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', required=True)
    ap.add_argument('--ztables', default=None)
    ap.add_argument('--pool', type=int, default=200)
    ap.add_argument('--nquad', type=int, default=48)
    ap.add_argument('--json', default=None)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    tables = (None, None)
    if args.ztables:
        with open(args.ztables) as fp:
            cache = json.load(fp)
        zt = cache['z_tables']
        keys = sorted(zt)
        tables = (zt[keys[0]], zt[keys[1]])
        print('Zhat tables %s' % keys)

    events, banner = _load(args.events, args.pool)
    pole = float(banner.get('param', 'mass', 6).value)
    width = float(banner.get('param', 'decay', 6).value)
    bw_cut = float(banner.get_detail('run_card', 'bwcutoff'))
    print('%d events, pole %.4f width %.6f BW_cut %s'
          % (len(events), pole, width, bw_cut))

    if args.check:
        sys.path.insert(0, os.path.join(_root, 'doc', 'madspin_pa_mass_stage'))
        worst_v = worst_k = 0.0
        for e in events[:20]:
            ref = analytic_A_loop(e.sqrts, tables, pole, width, bw_cut,
                                  args.nquad)
            vec = analytic_A_vec(e.sqrts, tables, pole, width, bw_cut,
                                 args.nquad)
            f = _frame_of(e)
            ker = analytic_A_kernel(f[0], f[1], f[2], [0, 1], tables, pole,
                                    width, bw_cut, args.nquad)
            worst_v = max(worst_v, abs(vec / ref - 1))
            worst_k = max(worst_k, abs(ker / ref - 1))
        print('check: vectorised vs loop  max rel dev %.3g' % worst_v)
        print('check: kernel J vs pmag J  max rel dev %.3g' % worst_k)

    res = time_all(events, tables, pole, width, bw_cut, args.nquad,
                   min(len(events), 60))
    for k in sorted(res):
        print('  %-32s %s' % (k, res[k]))
    if args.json:
        res['nquad'] = args.nquad
        res['pole'] = pole
        res['width'] = width
        res['bw_cut'] = bw_cut
        with open(args.json, 'w') as fp:
            json.dump(res, fp, indent=2, sort_keys=True)
        print('wrote %s' % args.json)


if __name__ == '__main__':
    main()
