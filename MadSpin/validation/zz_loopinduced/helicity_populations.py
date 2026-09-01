#!/usr/bin/env python3
"""The helicity populations ``P(lambda_1, lambda_2)``, read straight off the LHE.

Why this file exists
--------------------
``spin_definitions.tex`` and ``SPIN_COEFFICIENTS.md`` quote

    C_kk(g g -> Z Z, box)       = +0.380 +- 0.072
    C_kk(q qbar -> Z Z, cont.)  = -0.645 +- 0.080

and the paper explains the opposite signs by a statement about *helicity
populations*: that the ``q qbar`` channel favours ``Z`` pairs of OPPOSITE
helicity and the loop-induced ``g g`` channel pairs of EQUAL helicity.

Since ``C_kk = P(++) + P(--) - P(+-) - P(-+)`` is the definition
(Eq. (ckk) of ``spin_definitions.tex``, each ``S_k`` on its own ``Z``'s
helicity axis), re-measuring ``C_kk`` from the decay angles tests the sign
and nothing else: the population statement is true by construction once the
sign is.  To test the population statement itself one needs the populations,
independently of the decay angles.

They are in the LHE.  ``unwgt.f`` calls ``get_helicities`` and writes the
per-event helicity into column 13, ``SPINUP``, of every particle line, and
``matrix1.f`` picks that helicity by importance sampling,

    SUMHEL = SUMHEL + DABS(TS(I)) / ANS   ;   IF (RHEL .LT. SUMHEL) HEL_SELECTED = I

i.e. with probability ``|M_i|^2 / sum_j |M_j|^2``.  On an unweighted file the
frequency of a helicity configuration is therefore its physical fraction, and
no decay, no analysing power and no ``eta_l`` dilution stands between the file
and the number.  ``TS`` is a squared amplitude in both samples here -- tree
level on the ``q qbar`` side, ``|M_loop|^2`` on the ``g g`` one -- so the
``DABS`` is inert and the sampling is a genuine probability.

The frame is the right one for free.  MadEvent evaluates the matrix element in
the partonic centre-of-mass frame and boosts to the lab only afterwards (the
``Boost momentum to lab frame`` block of ``unwgt.f`` runs after
``get_helicities``), and both samples are ``2 -> 2`` at the hard level, so the
partonic CM frame *is* the ``Z Z`` rest frame and ``SPINUP`` is the helicity
along each ``Z``'s own direction there.  That is exactly the axis of
Eq. (ckk).

The bonus is the initial state.  ``SPINUP`` is written for the incoming
partons too, so the same file carries

* the ``g g`` total angular momentum along the beam, ``J_z = lambda_1 -
  lambda_2`` for gluon 1 along ``+z`` and gluon 2 along ``-z``, which is the
  ``J_z = 0`` clause of the paper sentence; and
* the ``q qbar`` helicity pair, which is the chirality-conservation clause.

Samples
-------
``qq``  ``p p > z z`` at LO, 200 000 events, the undecayed production sample of
        the T131 diagram study.  At LO ``p p > z z`` is ``q qbar`` only --
        ``g g -> Z Z`` is loop induced and absent from the tree-level process
        -- so this sample is the ``q qbar`` mechanism with no admixture.
``gg``  ``g g > z z [noborn=QCD]``, 200 000 events, the undecayed production
        sample of the T118 loop-induced study.

Usage::

    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    python3 helicity_populations.py
"""

import gzip
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

SAMPLES = {
    'qq': os.path.expanduser(
        '~/Documents/madspin_validation_samples/t131_qq_diagrams/'
        'work/lo_zz/Events/prod/unweighted_events.lhe.gz'),
    'gg': os.path.expanduser(
        '~/Documents/madspin_validation_samples/t118_zz_loopinduced/'
        'ggzz/Events/prod/unweighted_events.lhe.gz'),
}

LABEL = {
    'qq': 'q qbar -> Z Z   (LO, p p > z z)',
    'gg': 'g g -> Z Z      (loop induced, box)',
}


# --------------------------------------------------------------------------
# the reader
# --------------------------------------------------------------------------
def read_spinup(path):
    """Columns of the undecayed LHE: incoming and outgoing ids, pz and SPINUP.

    Returns a dict of arrays, one row per event.  A particle line of the LHE is

        IDUP ISTUP MOTH1 MOTH2 ICOL1 ICOL2 px py pz E m VTIMUP SPINUP
          1     2     3     4     5     6    7  8  9 10 11  12    13

    so ``SPINUP`` is ``fields[12]`` and ``ISTUP`` is ``fields[1]``.
    """
    opener = gzip.open if path.endswith('.gz') else open
    ids_in, hel_in, pz_in = [], [], []
    hel_z, pz_z, e_z, px_z, py_z = [], [], [], [], []
    wgt = []
    in_event = False
    header = False
    buf = []
    with opener(path, 'rt') as fh:
        for line in fh:
            s = line.strip()
            if s.startswith('<event'):
                in_event = True
                header = True
                buf = []
                continue
            if not in_event:
                continue
            if s.startswith('</event>'):
                in_event = False
                inc = [p for p in buf if p[1] == -1]
                zs = [p for p in buf if p[0] == 23 and p[1] == 1]
                if len(inc) != 2 or len(zs) != 2:
                    raise RuntimeError('unexpected event topology in %s' % path)
                # order the incoming by pz so "1" is always the +z beam
                inc.sort(key=lambda p: -p[4])
                ids_in.append([inc[0][0], inc[1][0]])
                hel_in.append([inc[0][5], inc[1][5]])
                pz_in.append([inc[0][4], inc[1][4]])
                hel_z.append([zs[0][5], zs[1][5]])
                px_z.append([zs[0][2], zs[1][2]])
                py_z.append([zs[0][3], zs[1][3]])
                pz_z.append([zs[0][4], zs[1][4]])
                e_z.append([zs[0][6], zs[1][6]])
                continue
            if not s or s.startswith('<') or s.startswith('#'):
                continue
            f = s.split()
            if header:
                # NUP IDPRUP XWGTUP SCALUP AQEDUP AQCDUP
                if len(f) >= 6:
                    wgt.append(float(f[2]))
                    header = False
                continue
            if len(f) < 13:
                continue
            buf.append((int(f[0]), int(f[1]), float(f[6]), float(f[7]),
                        float(f[8]), int(round(float(f[12]))), float(f[9])))
    return {
        'w': np.array(wgt),
        'ids_in': np.array(ids_in),
        'hel_in': np.array(hel_in),
        'pz_in': np.array(pz_in),
        'hel_z': np.array(hel_z),
        'p_z': np.stack([np.array(e_z), np.array(px_z),
                         np.array(py_z), np.array(pz_z)], axis=-1),
    }


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------
def wfrac(mask, w):
    """Weighted fraction of ``mask`` and its error, valid for signed weights.

    ``f = sum(w[mask]) / sum(w)``; the bar is the error of the weighted mean of
    the indicator, ``sqrt(sum w^2 (x - f)^2) / |sum w|``, which reduces to the
    binomial ``sqrt(f (1-f) / N)`` on unit weights.
    """
    x = mask.astype(float)
    sw = w.sum()
    f = float(np.dot(w, x) / sw)
    e = float(np.sqrt(np.dot(w ** 2, (x - f) ** 2)) / abs(sw))
    return f, e


def wmean(x, w):
    sw = w.sum()
    m = float(np.dot(w, x) / sw)
    e = float(np.sqrt(np.dot(w ** 2, (x - m) ** 2)) / abs(sw))
    return m, e


def populations(hel_z, w):
    """The 3 x 3 joint helicity population matrix and the four TT entries."""
    l1, l2 = hel_z[:, 0], hel_z[:, 1]
    out = {}
    for a in (1, 0, -1):
        for b in (1, 0, -1):
            f, e = wfrac((l1 == a) & (l2 == b), w)
            out['P(%+d,%+d)' % (a, b)] = (f, e)
    # the four transverse entries, and the two physically meaningful sums
    out['P_eq']  = wfrac(((l1 == 1) & (l2 == 1)) | ((l1 == -1) & (l2 == -1)), w)
    out['P_opp'] = wfrac(((l1 == 1) & (l2 == -1)) | ((l1 == -1) & (l2 == 1)), w)
    out['f_TT']  = wfrac((l1 != 0) & (l2 != 0), w)
    out['f_00']  = wfrac((l1 == 0) & (l2 == 0), w)
    out['f_0_1'] = wfrac(l1 == 0, w)
    out['f_0_2'] = wfrac(l2 == 0, w)
    out['f_0']   = wmean(0.5 * ((l1 == 0).astype(float)
                                + (l2 == 0).astype(float)), w)
    # C_kk = <S_k(1) S_k(2)>, each S on its own Z's helicity axis: exactly the
    # product of the two SPINUP values, event by event.  No eta_l anywhere.
    out['C_kk'] = wmean((l1 * l2).astype(float), w)
    return out


def initial_state(d):
    """``2 J_z`` of the initial state along ``+z``, in units of one half.

    ``hel_in`` is ordered so row 0 is the parton with the larger ``pz``, i.e.
    the one travelling along ``+z``.  Its spin projection on ``+z`` is its
    helicity; the other travels along ``-z`` so its projection is MINUS its
    helicity.  Hence ``J_z = lambda_1 - lambda_2``.

    A caution on units.  MadGraph's ``NHEL``, which is what lands in
    ``SPINUP``, is ``+-1`` for a fermion and means helicity ``+-1/2``; for a
    vector it is ``0, +-1`` and means the helicity itself.  So on the ``g g``
    sample ``lambda_1 - lambda_2`` already is ``J_z``, while on the ``q qbar``
    one it is ``2 J_z``.  This function returns the ``NHEL`` difference and the
    caller divides: ``twice`` is 1 for gluons and 2 for quarks.  Getting this
    wrong turns the ``q qbar`` ``|J_z| = 1`` into a spurious ``|J_z| = 2``.
    """
    l1, l2 = d['hel_in'][:, 0], d['hel_in'][:, 1]
    fermionic = np.all(np.abs(d['ids_in']) <= 6)
    return l1 - l2, (2 if fermionic else 1)


# --------------------------------------------------------------------------
# The angular extraction this has to reproduce.  These are the published rows
# of data/numbers.txt (g g) and data/numbers_lo_diagrams.txt (q qbar), taken on
# the MadSpin decay of *these very production files* -- t118 ms_madspin is
# MadSpin run on ggzz/Events/prod/unweighted_events.lhe.gz and t131 ms_madspin
# is MadSpin run on lo_zz/Events/prod/unweighted_events.lhe.gz -- so this is a
# same-sample closure and not a comparison of two generations.
ANGULAR = {
    'qq': {'label': 't131 zz_madspin (MadSpin decay of THIS lo_zz file)',
           'f_0': (0.1725, 0.0024), 'f_TT': (0.7120, 0.0066),
           'C_kk': (-0.7214, 0.0682)},
    'gg': {'label': 't118 madspin (MadSpin decay of THIS ggzz file)',
           'f_0': (0.0673, 0.0025), 'f_TT': (0.9166, 0.0071),
           'C_kk': (+0.4599, 0.0723)},
}


def report(tag, d, fh):
    w = d['w']
    n = len(w)
    hel_z = d['hel_z']
    p = populations(hel_z, w)
    jz_nhel, twice = initial_state(d)

    def line(s=''):
        fh.write(s + '\n')

    line('=' * 78)
    line('%s   -- %d events' % (LABEL[tag], n))
    line('=' * 78)
    line('  weights: min %.6g  max %.6g  negative %d'
         % (w.min(), w.max(), int((w < 0).sum())))
    unseen = sorted(set(np.unique(hel_z).tolist()))
    line('  SPINUP values seen on the two z: %s' % unseen)
    if unseen == [9] or unseen == [9.0]:
        line('  *** SPINUP is 9 = UNKNOWN.  This file carries no helicity. ***')
        return None
    line('')
    line('  --- the joint helicity population matrix P(lambda_1, lambda_2) ---')
    line('  rows: lambda of the FIRST z on the LHE line; columns: the second.')
    line('  %-8s %-22s %-22s %-22s' % ('', 'l2 = +1', 'l2 =  0', 'l2 = -1'))
    for a in (1, 0, -1):
        cells = []
        for b in (1, 0, -1):
            f, e = p['P(%+d,%+d)' % (a, b)]
            cells.append('%.5f +- %.5f' % (f, e))
        line('  l1 =%+2d  %-22s %-22s %-22s' % (a, cells[0], cells[1], cells[2]))
    line('')
    line('  --- the statement under test ---')
    for k, txt in (('P_eq',  'P(++) + P(--)   EQUAL   helicities'),
                   ('P_opp', 'P(+-) + P(-+)   OPPOSITE helicities')):
        f, e = p[k]
        line('    %-40s %.5f +- %.5f' % (txt, f, e))
    fe, ee = p['P_eq']
    fo, eo = p['P_opp']
    diff = fe - fo
    ediff = np.sqrt(ee ** 2 + eo ** 2)
    line('    %-40s %+.5f +- %.5f   (%+.1f sigma)'
         % ('difference  EQUAL - OPPOSITE', diff, ediff, diff / ediff))
    line('    which of the two dominates: %s'
         % ('EQUAL' if diff > 0 else 'OPPOSITE'))
    line('')
    line('  --- the coefficients, from SPINUP alone ---')
    for k, txt in (('f_0',  'f_0   (per-event average of the two z)'),
                   ('f_00', 'f_00  (both longitudinal)'),
                   ('f_TT', 'f_TT  (both transverse)'),
                   ('C_kk', 'C_kk  = <lambda_1 lambda_2>')):
        f, e = p[k]
        line('    %-40s %+.5f +- %.5f' % (txt, f, e))
    f00, _ = p['f_00']
    f01, _ = p['f_0_1']
    f02, _ = p['f_0_2']
    line('    %-40s %+.5f' % ('f_00 - f_0(1) f_0(2)', f00 - f01 * f02))
    line('')
    line('  --- closure against the decay-angle extraction ---')
    line('  %s' % ANGULAR[tag]['label'])
    line('  If the two disagree, either the helicity sampling is not a')
    line('  probability or SPINUP is not on the Z Z rest-frame axis.')
    line('    %-8s %-22s %-22s %-16s' % ('', 'SPINUP (this file)',
                                         'decay angles', 'difference'))
    for k in ('f_0', 'f_TT', 'C_kk'):
        a, ea = p[k]
        b, eb = ANGULAR[tag][k]
        dd = a - b
        ed = np.sqrt(ea ** 2 + eb ** 2)
        line('    %-8s %+.5f +- %.5f     %+.5f +- %.5f     %+.1f sigma'
             % (k, a, ea, b, eb, dd / ed))
    line('')
    line('  --- the initial state, also from SPINUP ---')
    ids = d['ids_in']
    line('    incoming pdg pairs seen: %s'
         % sorted(set(map(tuple, np.abs(ids).tolist())))[:8])
    line('    NHEL is %s, so J_z = (NHEL_1 - NHEL_2) / %d'
         % ('+-1 meaning helicity +-1/2 (fermions)' if twice == 2
            else 'the helicity itself (vectors)', twice))
    jz2 = jz_nhel  # in units of 1/twice
    for v in sorted(set(jz2.tolist())):
        f, e = wfrac(jz2 == v, w)
        line('    J_z = %+.1f along the beam : %.5f +- %.5f'
             % (v / float(twice), f, e))
    f0jz, e0jz = wfrac(jz2 == 0, w)
    line('    J_z = 0 in total          : %.5f +- %.5f' % (f0jz, e0jz))
    line('')
    line('  --- Z helicities CONDITIONAL on the initial J_z ---')
    for v in sorted(set(np.abs(jz2).tolist())):
        m = np.abs(jz2) == v
        if m.sum() < 50:
            continue
        sub = populations(hel_z[m], w[m])
        fe2, ee2 = sub['P_eq']
        fo2, eo2 = sub['P_opp']
        ftt2, _ = sub['f_TT']
        ck2, ek2 = sub['C_kk']
        line('    |J_z| = %.1f  (%.4f of the rate)'
             % (v / float(twice), wfrac(m, w)[0]))
        line('        P_eq  %.5f +- %.5f    P_opp %.5f +- %.5f'
             % (fe2, ee2, fo2, eo2))
        line('        f_TT  %.5f              C_kk  %+.5f +- %.5f'
             % (ftt2, ck2, ek2))
    line('')
    return p


def main():
    out_txt = os.path.join(_HERE, 'data', 'numbers_helicity_populations.txt')
    out_json = os.path.join(_HERE, 'data', 'numbers_helicity_populations.json')
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    blob = {}
    with open(out_txt, 'w') as fh:
        fh.write(__doc__.split('Usage')[0].strip() + '\n\n')
        for tag in ('qq', 'gg'):
            path = SAMPLES[tag]
            if not os.path.exists(path):
                fh.write('MISSING: %s\n' % path)
                continue
            sys.stderr.write('reading %s ...\n' % path)
            d = read_spinup(path)
            p = report(tag, d, fh)
            if p is not None:
                blob[tag] = {k: list(v) for k, v in p.items()}
                blob[tag]['N'] = int(len(d['w']))
                blob[tag]['lhe'] = path
    with open(out_json, 'w') as fh:
        json.dump(blob, fh, indent=1)
    sys.stderr.write('wrote %s\n' % out_txt)
    print(open(out_txt).read())


if __name__ == '__main__':
    main()
