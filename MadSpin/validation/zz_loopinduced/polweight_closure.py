#!/usr/bin/env python3
"""Close the angular extraction of ``f_0`` against MadSpin's polarised-ME weights.

It closes: the eight entries of the table this prints agree with 1/5 and 2/5 to
1.4 sigma at worst on 250 000 events, and the two extractions of ``f_0`` agree
to 1.4 sigma.  It did NOT close when this script first ran, and the reason was
here rather than in MadSpin -- see ``--mixed`` and
POLWEIGHT_CLOSURE_DIAGNOSIS.md.

This is the cross-check of section 9 of ``SPIN_COEFFICIENTS.md``, and it is the
only thing in this directory that does NOT run on this study's samples: the
``g g > z z`` runs carry no ``ms_pol_*`` weights, so the check is done on a
``p p > z z`` sample that does, and it is a check of the *method* rather than of
these events.

What it does, in one streaming pass over a MadSpin-decayed Les Houches file that
was produced with ``set keep_weight_for_polarization_vector [0, T]``:

  * measures ``f_0 = 2 - 5 <cos^2 theta>`` and ``f_00`` from the decay angles,
    using this study's own :mod:`observables` definitions;
  * measures the same quantities from the four ``ms_pol_*`` weights, as
    ``(LL+LT)/full`` and ``LL/full``;
  * reweights the sample by each polarised weight in turn and prints
    ``<cos^2 theta>`` for it, which has to be exactly 1/5 for a longitudinal Z
    and 2/5 for a transverse one if the two definitions live on the same axis.

Use a **Born** sample. MadSpin quantises on the ``me_frame`` axis (run-card
``me_frame``, i.e. the rest frame of legs 1 and 2 by default); on a ``2 -> 2``
production that frame is the ``ZZ`` rest frame and coincides with the helicity
axis ``observables.compute`` uses, so any residual is a real disagreement and
not a frame mismatch -- **provided the boosts into the pair rest frame are
composed through that frame and not taken from the lab**, which is exactly the
mistake ``--mixed`` reproduces. The script boosts to the initial-state CM by default so
that it is also usable on events with extra radiation, and ``--frame 4l`` asks
for the four-lepton frame instead; on a Born sample the two agree bit for bit,
which is itself worth checking.

    python3 polweight_closure.py .../run_12_decayed_1/events.lhe.gz

The sample this file's numbers were taken on was
``PROCNLO_loop_sm_7/Events/run_12_decayed_1``: ``p p > z z [QCD]`` at
``order = LO``, ``spinmode = madspin``, ``BW_cut = 15``, exclusive
``decay z > e+ e-`` / ``decay z > mu+ mu-``, 250 000 events, no cuts.
"""

import argparse
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import observables as OBS                                        # noqa: E402


# the four combinations a ``[0, T]`` list produces for two z, in slot order;
# slot 1 is the z that appears first in the production record
POL = ['ms_pol_23:0_23:0', 'ms_pol_23:0_23:T',
       'ms_pol_23:T_23:0', 'ms_pol_23:T_23:T']
SHORT = ['LL', 'LT', 'TL', 'TT']
# <cos^2 theta_i> that each combination MUST give: 1/5 longitudinal, 2/5 transverse
PREDICTED = {'LL': (0.2, 0.2), 'LT': (0.2, 0.4), 'TL': (0.4, 0.2), 'TT': (0.4, 0.4)}


def read(path, want=(-11, 11, -13, 13), maxev=-1):
    """``(weights, {pdg: (N,4)}, {label: (N,)}, initial-state sum (N,4))``."""
    w_nom, wpol, init, nup = [], [], [], []
    cols = {p: [] for p in want}
    opener = gzip.open if path.endswith('.gz') else open
    nread = 0
    with opener(path, 'rt', errors='replace') as fh:
        inev = False
        for line in fh:
            if not inev:
                inev = line.startswith('<event')
                if inev:
                    cur, ini, pol, w, n = {}, [], {}, None, None
                continue
            s = line.strip()
            if s.startswith('</event>'):
                inev = False
                nread += 1
                if len(cur) == len(want) and len(pol) == len(POL) and w is not None:
                    w_nom.append(w)
                    wpol.append([pol[k] for k in POL])
                    nup.append(n)
                    init.append([sum(x) for x in zip(*ini)])
                    for p in want:
                        cols[p].append(cur[p])
                if maxev > 0 and nread >= maxev:
                    break
                continue
            if s.startswith("<wgt id='ms_pol"):
                i = s.index("'") + 1
                j = s.index("'", i)
                if s[i:j] in POL:
                    pol[s[i:j]] = float(s[s.index('>', j) + 1:s.rindex('<')])
                continue
            if not s or s.startswith('<') or s.startswith('#'):
                continue
            f = s.split()
            if w is None:
                if len(f) >= 6:
                    n, w = int(f[0]), float(f[2])
                continue
            if len(f) < 13:
                continue
            pdg, st = int(f[0]), int(f[1])
            v = (float(f[9]), float(f[6]), float(f[7]), float(f[8]))
            if st == -1:
                ini.append(v)
            elif st == 1 and pdg in want:
                cur[pdg] = v
    return (np.array(w_nom), {p: np.array(cols[p]) for p in want},
            {k: np.array(wpol)[:, i] for i, k in enumerate(SHORT)},
            np.array(init), np.array(nup))


def angles(p, ref, mixed=False):
    """``(cos theta1, cos theta2)`` with the analysing frame taken in ``ref``.

    The boosts are SEQUENTIAL: lab -> ``ref`` -> the pair's rest frame.  Taking
    the axis in ``ref`` while boosting the lepton into the pair rest frame
    straight from the lab -- ``mixed=True``, which is what this script and
    ``observables.compute`` did until the closure was diagnosed -- composes two
    non-collinear boosts, and the Wigner rotation of that composition tilts the
    analysing frame off the axis by a median 8 degrees on this sample.  That
    tilt, and nothing else, is the '5 % leakage' this script used to report.
    ``mixed=True`` is kept so the failure can be reproduced on demand.
    """
    ep, em, mup, mum = p[-11], p[11], p[-13], p[13]
    z1, z2 = ep + em, mup + mum
    z1r, z2r = OBS.boost_to_rest(z1, ref), OBS.boost_to_rest(z2, ref)
    d1 = OBS._unit(z1r[:, 1:4])
    d2 = OBS._unit(z2r[:, 1:4])
    if mixed:
        e1 = OBS._unit(OBS.boost_to_rest(ep, z1)[:, 1:4])
        e2 = OBS._unit(OBS.boost_to_rest(mup, z2)[:, 1:4])
    else:
        e1 = OBS._unit(OBS.boost_to_rest(OBS.boost_to_rest(ep, ref), z1r)[:, 1:4])
        e2 = OBS._unit(OBS.boost_to_rest(OBS.boost_to_rest(mup, ref), z2r)[:, 1:4])
    return (np.clip(np.sum(d1 * e1, axis=1), -1.0, 1.0),
            np.clip(np.sum(d2 * e2, axis=1), -1.0, 1.0))


def wmean(v, w):
    s = w.sum()
    mu = float((w * v).sum() / s)
    var = float((w * (v - mu) ** 2).sum() / s)
    return mu, float(np.sqrt(abs(var) / (s ** 2 / (w ** 2).sum())))


def jackknife(v, w, nblock=20):
    """Error of ``<v>_w`` from 20 delete-one blocks -- the honest bar when the
    reweighting has a tail, which ``LL`` does."""
    idx = np.arange(len(w)) % nblock
    vals = np.array([float((w[idx != b] * v[idx != b]).sum() / w[idx != b].sum())
                     for b in range(nblock)])
    return float(np.sqrt((nblock - 1) / nblock * np.sum((vals - vals.mean()) ** 2)))


def ratio(num, den):
    """``(sum num / sum den, error)``, delta method over the SAME events."""
    a, b = num.sum(), den.sum()
    r = a / b
    err = abs(r) * np.sqrt((num ** 2).sum() / a ** 2 + (den ** 2).sum() / b ** 2
                           - 2 * (num * den).sum() / (a * b))
    return r, float(err)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('lhe')
    ap.add_argument('--frame', choices=('me', '4l'), default='me',
                    help="'me' (default) = rest frame of the initial state, which "
                         "is what MadSpin quantises on; '4l' = four-lepton frame")
    ap.add_argument('--max-events', type=int, default=-1)
    ap.add_argument('--mixed', action='store_true',
                    help='reproduce the historical failure: take the axis in the '
                         'analysing frame but boost the lepton into the pair rest '
                         'frame straight from the lab. That composition carries a '
                         'Wigner rotation and is what made this test look like a '
                         '5 %% leakage; see POLWEIGHT_CLOSURE_DIAGNOSIS.md.')
    args = ap.parse_args()

    w, p, pol, init, nup = read(args.lhe, maxev=args.max_events)
    four = p[-11] + p[11] + p[-13] + p[13]
    c1, c2 = angles(p, init if args.frame == 'me' else four, mixed=args.mixed)

    print('%d events, NUP: %s' % (len(w), dict(zip(*np.unique(nup, return_counts=True)))))
    print('analysing axis taken in the %s frame'
          % ('initial-state CM (me_frame)' if args.frame == 'me' else 'four-lepton'))
    print('pt(4l) max = %.3g GeV  (0 means the production is 2 -> 2 and the two '
          'frames coincide)' % np.abs(np.hypot(four[:, 1], four[:, 2])).max())
    print()

    X, Y = OBS.POL_0(c1), OBS.POL_0(c2)
    print('--- from the decay angles ---')
    for name, v in (('f_0 (slot 1)', X), ('f_0 (slot 2)', Y), ('f_00', X * Y),
                    ('f_TT', (1 - X) * (1 - Y)), ('<cos th1 cos th2>', c1 * c2)):
        print('  %-18s %+.6f +- %.6f' % ((name,) + wmean(v, w)))
    print('  %-18s %+.6f +- %.6f'
          % (('C_kk',) + OBS.c_kk_from_moment(*wmean(c1 * c2, w))))
    print()

    print('--- from the polarised matrix-element weights ---')
    for name, v in (('LL', pol['LL']), ('LT', pol['LT']), ('TL', pol['TL']),
                    ('TT', pol['TT']),
                    ('f_0 (slot 1) = LL+LT', pol['LL'] + pol['LT']),
                    ('f_0 (slot 2) = LL+TL', pol['LL'] + pol['TL']),
                    ('sum of the four', sum(pol.values()))):
        print('  %-22s %+.6f +- %.6f' % ((name,) + ratio(v, w)))
    print()

    print('--- the test: <cos^2 theta> under each polarised weight ---')
    print('  %-6s %-24s %-24s' % ('', 'slot 1 (measured/must be)',
                                  'slot 2 (measured/must be)'))
    for k in SHORT:
        a, b = PREDICTED[k]
        m1, m2 = wmean(c1 ** 2, pol[k])[0], wmean(c2 ** 2, pol[k])[0]
        j1, j2 = jackknife(c1 ** 2, pol[k]), jackknife(c2 ** 2, pol[k])
        print('  %-6s %.4f +- %.4f / %.1f      %.4f +- %.4f / %.1f'
              % (k, m1, j1, a, m2, j2, b))
    worst = 0.0
    for k in SHORT:
        for c, target in ((c1, PREDICTED[k][0]), (c2, PREDICTED[k][1])):
            worst = max(worst, abs(wmean(c ** 2, pol[k])[0] - target)
                        / max(jackknife(c ** 2, pol[k]), 1e-12))
    print('  worst deviation over the eight entries: %.1f sigma' % worst)
    f0a = 2 - 5 * wmean(c1 ** 2, w)[0]
    e0a = 5 * wmean(c1 ** 2, w)[1]
    f0m, f0me = ratio(pol['LL'] + pol['LT'], w)
    print('  f_0 (slot 1): angular %.5f +- %.5f   polarised ME %.5f +- %.5f'
          '   -> %.1f sigma' % (f0a, e0a, f0m, f0me,
                                (f0a - f0m) / np.hypot(e0a, f0me)))
    if args.mixed:
        print('\n  (--mixed) the deviations above are the Wigner rotation of a '
              'non-sequential\n  boost composition, not a property of the '
              'weights. Drop --mixed to close.')


if __name__ == '__main__':
    main()
