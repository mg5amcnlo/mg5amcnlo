#!/usr/bin/env python3
"""Measure ``A_e`` -- the mass stage's per-event normalisation -- over a whole
production sample, for each of the three modes that have one, and turn it into
the reweighting the correction would apply.

Three modes, three weights (``doc/madspin_ae_normalisation/ae_kernel.py`` has
the derivation of the integral):

    pa        w = J . prod jac_BW . prod Zhat            (PA, the default)
    panojac   w =     prod jac_BW . prod Zhat            (density_keep_jacobian
                                                          = False)
    madspin   w = R . J . prod jac_BW . prod Zhat        (offshell)

``R = Tr(rho_off)/|M_prod|^2_on`` is the only factor with no closed form.  This
script computes the ``R = 1`` part of the offshell ``A_e`` by the same exact
quadrature -- with the *offshell* Zhat tables, which are much steeper than the
PA ones -- and leaves ``R`` to ``ratio_R.py``, which reads it back out of a
MadSpin run that carried it.

Everything else here is exact: there is no Monte Carlo in this file.

Beside ``A_e`` it computes ``E_q[w^2]`` on the same grid, which gives the
*within-event* spread of ``w`` -- the number that says how many free draws an
``n >= 3`` production would need to estimate ``A_e`` to a given precision.

    E_q[w]   = pi^-2  Int dR1 dR2                J Z1 Z2
    E_q[w^2] = pi^-4  Int dR1 dR2 gap1 gap2(m1)  J^2 Z1^2 Z2^2

(the proposal is uniform in each ``R_s`` over a window of width ``gap_s``, and
``jac_BW_s = gap_s/pi``; ``gap_2`` depends on ``m_1`` through the budget, which
is the truncation that makes ``A_e`` vary at all).

Usage::

    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    python3 doc/madspin_ae_normalisation/measure_ae.py \
        --events <production .lhe.gz> \
        --mode pa --ztables <ms_dir>/max_wgt_sequential_pa \
        --out doc/madspin_ae_normalisation/data/ae_pa.npz
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import gzip
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


def read_production(path, pdgs=(6, -6), nmax=None):
    """``(sqrt(shat), m1, m2)`` per production event, straight off the text.

    ``sqrt(shat)`` is the invariant mass of the final state, which is what
    ``Event.sqrts`` returns and what the reshuffling preserves; ``m1, m2`` are
    the two resonances' on-shell masses as the file carries them (the
    reshuffling jacobian is normalised to *those*, not to the pole).
    """
    opener = gzip.open if path.endswith('.gz') else open
    out_s, out_m = [], []
    with opener(path, 'rt') as fh:
        in_ev = need_head = False
        acc = None
        want = list(pdgs)
        with_m = None
        for line in fh:
            if line.startswith('<event'):
                in_ev, need_head = True, True
                acc = [0.0, 0.0, 0.0, 0.0]
                with_m = {}
                continue
            if not in_ev:
                continue
            if line.startswith('</event'):
                in_ev = False
                if len(with_m) == len(want):
                    m2 = (acc[3] ** 2 - acc[0] ** 2 - acc[1] ** 2
                          - acc[2] ** 2)
                    out_s.append(math.sqrt(max(m2, 0.0)))
                    out_m.append([with_m[p] for p in want])
                    if nmax and len(out_s) >= nmax:
                        break
                continue
            f = line.split()
            if need_head:
                need_head = False
                continue
            if len(f) < 11 or f[1] != '1':
                continue
            acc[0] += float(f[6])
            acc[1] += float(f[7])
            acc[2] += float(f[8])
            acc[3] += float(f[9])
            pid = int(f[0])
            if pid in want and pid not in with_m:
                with_m[pid] = float(f[10])
    return np.asarray(out_s), np.asarray(out_m)


def load_tables(path, keys=('6_0', '-6_0')):
    if not path:
        return (None, None), None
    with open(path) as fp:
        cache = json.load(fp)
    zt = cache['z_tables']
    return tuple(zt.get(k) for k in keys), cache.get('maxwgts')


def zhat_vec(table, mass):
    if not table:
        return np.ones_like(mass)
    lo, hi = table['range']
    c = table['coeff']
    u = np.log(np.clip(mass, lo, hi) / table['pole'])
    return np.where(mass < table['zero_below'], 0.0,
                    np.exp(c[0] + u * (c[1] + u * c[2])))


def pmag(sqrts, m1, m2):
    s = sqrts * sqrts
    lam = (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)
    return np.sqrt(np.maximum(0.0, lam)) / (2 * sqrts)


def ae_batch(sqrts, m_orig, tables, pole, width, bw_cut, nquad, keep_jac):
    """``(A_e, E_q[w^2])`` for a whole array of production events at once.

    The grid is the same ``(R_1, R_2)`` square for every event; only the
    ``R_2`` cap moves with ``sqrt(shat) - m_1``, so the whole sample is one
    three-index numpy expression (event, node1, node2).  Chunked so the
    intermediate stays a sensible size.
    """
    x, wq = np.polynomial.legendre.leggauss(nquad)
    lo = pole - bw_cut * width
    A = np.empty_like(sqrts)
    A2 = np.empty_like(sqrts)
    r_lo = math.atan((lo ** 2 - pole ** 2) / pole / width)
    chunk = max(1, int(4e6 / (nquad * nquad)))
    for i0 in range(0, len(sqrts), chunk):
        s = sqrts[i0:i0 + chunk]
        mo = m_orig[i0:i0 + chunk]
        hi1 = np.minimum(pole + bw_cut * width, s)
        r_hi1 = np.arctan((np.maximum(hi1, lo) ** 2 - pole ** 2) / pole / width)
        # (nev, nq)
        r1 = (0.5 * (r_hi1 + r_lo))[:, None] \
            + (0.5 * (r_hi1 - r_lo))[:, None] * x[None, :]
        m1 = np.sqrt(pole ** 2 + pole * width * np.tan(r1))
        z1 = zhat_vec(tables[0], m1)
        gap1 = (r_hi1 - r_lo)[:, None] * np.ones_like(m1)

        hi2 = np.minimum(pole + bw_cut * width, s[:, None] - m1)
        ok2 = hi2 > lo
        r_hi2 = np.arctan((np.maximum(hi2, lo) ** 2 - pole ** 2) / pole / width)
        # (nev, nq, nq)
        r2 = (0.5 * (r_hi2 + r_lo))[:, :, None] \
            + (0.5 * (r_hi2 - r_lo))[:, :, None] * x[None, None, :]
        m2 = np.sqrt(pole ** 2 + pole * width * np.tan(r2))
        z2 = zhat_vec(tables[1], m2)
        gap2 = (r_hi2 - r_lo)[:, :, None]

        if keep_jac:
            denom = pmag(s, mo[:, 0], mo[:, 1])
            jac = pmag(s[:, None, None], m1[:, :, None], m2) \
                / denom[:, None, None]
        else:
            jac = np.ones_like(m2)

        half1 = (0.5 * (r_hi1 - r_lo))[:, None]
        half2 = (0.5 * (r_hi2 - r_lo))[:, :, None]
        w2d = wq[None, None, :] * half2
        inner = (w2d * jac * z2).sum(axis=2)
        A[i0:i0 + chunk] = ((wq[None, :] * half1 * z1
                             * np.where(ok2, inner, 0.0)).sum(axis=1)
                            / math.pi ** 2)
        inner2 = (w2d * gap2 * (jac * z2) ** 2).sum(axis=2)
        A2[i0:i0 + chunk] = ((wq[None, :] * half1 * gap1 * z1 ** 2
                              * np.where(ok2, inner2, 0.0)).sum(axis=1)
                             / math.pi ** 4)
    return A, A2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', required=True)
    ap.add_argument('--mode', required=True,
                    choices=['pa', 'panojac', 'madspin'])
    ap.add_argument('--ztables', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--nquad', type=int, default=48)
    ap.add_argument('--nmax', type=int, default=None)
    ap.add_argument('--pole', type=float, default=None)
    ap.add_argument('--width', type=float, default=None)
    ap.add_argument('--bw-cut', type=float, default=15.0)
    args = ap.parse_args()

    pole, width = args.pole, args.width
    if pole is None or width is None:
        import madgraph.various.lhe_parser as lhe_parser
        banner = lhe_parser.EventFile(args.events).get_banner()
        pole = float(banner.get('param', 'mass', 6).value)
        width = float(banner.get('param', 'decay', 6).value)
    t0 = time.time()
    sqrts, morig = read_production(args.events, nmax=args.nmax)
    print('%d events read in %.0f s, sqrt(shat) in [%.3f, %.0f]'
          % (len(sqrts), time.time() - t0, sqrts.min(), sqrts.max()),
          flush=True)

    tables, maxwgts = load_tables(args.ztables)
    print('mode %s  pole %.4f width %.6f  Zhat %s  maxwgts %s'
          % (args.mode, pole, width,
             'yes' if tables[0] else 'NO', maxwgts))

    keep_jac = args.mode in ('pa', 'madspin')
    t0 = time.time()
    A, A2 = ae_batch(sqrts, morig, tables, pole, width, args.bw_cut,
                     args.nquad, keep_jac)
    dt = time.time() - t0
    print('quadrature: %.1f s for %d events (%.1f us/event, %d x %d nodes)'
          % (dt, len(A), dt / len(A) * 1e6, args.nquad, args.nquad))

    # within-event relative spread of w: what a Monte Carlo estimator of A_e
    # would have to average down
    rel_sd = np.sqrt(np.maximum(A2 - A * A, 0.0)) / A
    print('A_e   mean %.6f  sd %.6f (%.3f %%)  range [%.4f, %.4f]'
          % (A.mean(), A.std(), 100 * A.std() / A.mean(), A.min(), A.max()))
    print('within-event sd(w)/A_e: median %.3f  p95 %.3f  max %.3f'
          % (np.median(rel_sd), np.percentile(rel_sd, 95), rel_sd.max()))
    np.savez_compressed(args.out, sqrts=sqrts, m_orig=morig, A=A, A2=A2,
                        rel_sd=rel_sd,
                        meta=np.array([pole, width, args.bw_cut, args.nquad,
                                       1.0 if keep_jac else 0.0]))
    print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
