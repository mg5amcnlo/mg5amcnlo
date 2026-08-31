#!/usr/bin/env python3
"""Every number of QQ_F0_DEFICIT.md that comes out of the EXISTING NLO samples.

Nothing here generates anything.  It reads the two independently seeded
200 000-event q qbar samples (event columns written by ``qq_coefficients.py``)
and the g g loop-induced sample, and produces:

  1. the pt(l+l-) spectrum near the 1 GeV threshold and the bound on the
     "cut acts on the z on one side and on the pair on the other" asymmetry
  2. f_0 after reweighting each MadSpin sample to the truth's distribution in
     m_4l, pt(4l), m_ee x m_mumu, ... -- i.e. the deficit at fixed kinematics
  3. the pooled |cos theta| shape
  4. the split by extra-parton multiplicity and by MC@NLO weight sign
  5. what the g g block would have excluded, for an absolute and for a
     proportional effect

pt(4l), |y_4l| and the parton multiplicity are not in the stored columns, so
they are recomputed here from the LHE files of sample 2; pass --no-lhe to skip
the parts that need them.

    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    python3 nlo_breakdown.py
"""
import argparse
import glob
import gzip
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
import observables as O                                          # noqa: E402

pjoin = os.path.join
PT_MIN = 1.0
MODES = ('madspin', 'onshell', 'PA')
S1 = os.path.expanduser('~/Documents/git_workspace/zz_spin_events/'
                        'qq_ppzz_nlo_madspin')
T130 = os.path.expanduser('~/Documents/madspin_validation_samples/'
                          't130_qq_seed_check')
S2 = pjoin(T130, 'event_columns')
GG = os.path.expanduser('~/Documents/madspin_validation_samples/'
                        't118_zz_loopinduced/event_columns')


def wmean(x, w):
    sw = w.sum()
    m = float(np.dot(w, x) / sw)
    e = float(np.sqrt(np.dot(w ** 2, (x - m) ** 2)) / abs(sw))
    return m, e


def f0(o, m=None):
    x = 0.5 * (o['pol0_1'] + o['pol0_2'])
    w = o['w']
    if m is not None:
        x, w = x[m], w[m]
    return wmean(x, w)


def select(obs, w, extra=None):
    sel = ((obs['m_ee'] > O.M_LO) & (obs['m_ee'] < O.M_HI)
           & (obs['m_mumu'] > O.M_LO) & (obs['m_mumu'] < O.M_HI)
           & (obs['pt_ee'] > PT_MIN) & (obs['pt_mumu'] > PT_MIN))
    out = {k: v[sel] for k, v in obs.items()}
    out['w'] = w[sel]
    if extra:
        out.update({k: v[sel] for k, v in extra.items()})
    return out


def read_npz(dirs, key, extra=None):
    obs, ws = None, []
    for d in dirs:
        z = np.load(pjoin(d, 'events_%s.npz' % key))
        cur = {k: z[k] for k in z.files if k != 'w'}
        ws.append(z['w'])
        obs = cur if obs is None else {k: np.concatenate([obs[k], cur[k]])
                                       for k in obs}
    return select(obs, np.concatenate(ws), extra)


# --------------------------------------------------------------------------
# pt(4l), |y_4l| and the parton multiplicity, straight off the LHE
# --------------------------------------------------------------------------
def scan_lhe(path):
    """(w, p4 of the four leptons, n extra partons) per event."""
    w, p4, nj = [], [], []
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt', errors='replace') as fh:
        inev = False
        wgt = tot = None
        n = found = 0
        for line in fh:
            s = line.strip()
            if s.startswith('<event'):
                inev, wgt, tot, n, found = True, None, np.zeros(4), 0, 0
                continue
            if not inev:
                continue
            if s.startswith('</event>'):
                inev = False
                if found == 4:
                    w.append(wgt)
                    p4.append(tot.copy())
                    nj.append(n)
                continue
            if not s or s.startswith(('<', '#')):
                continue
            f = s.split()
            if wgt is None:
                if len(f) >= 6:
                    try:
                        wgt = float(f[2])
                    except ValueError:
                        wgt = None
                continue
            if len(f) < 13:
                continue
            try:
                pdg, st = int(f[0]), int(f[1])
            except ValueError:
                continue
            if st != 1:
                continue
            if pdg in (-11, 11, -13, 13):
                tot += np.array([float(f[9]), float(f[6]), float(f[7]),
                                 float(f[8])])
                found += 1
            elif abs(pdg) <= 5 or pdg == 21:
                n += 1
    return (np.array(w), np.array(p4), np.array(nj))


def extras(paths):
    W, P, J = [], [], []
    for p in paths:
        a, b, c = scan_lhe(p)
        W.append(a)
        P.append(b)
        J.append(c)
    W, P, J = np.concatenate(W), np.concatenate(P), np.concatenate(J)
    E, pz = P[:, 0], P[:, 3]
    y = 0.5 * np.log(np.clip((E + pz) / np.clip(E - pz, 1e-12, None),
                             1e-12, None))
    return W, {'pt4l': np.hypot(P[:, 1], P[:, 2]), 'njet': J, 'y4l': y}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=pjoin(_HERE, '..', 'data'))
    ap.add_argument('--no-lhe', action='store_true')
    args = ap.parse_args()

    L = []
    A = L.append
    A('The q qbar f_0 deficit, broken down on the EXISTING NLO samples')
    A('nothing here is regenerated; see QQ_F0_DEFICIT.md')
    A('')

    # ---- combined (both seeds), the headline ----------------------------
    comb = {k: read_npz([S1, S2], k)
            for k in ('truth',) + MODES + ('none',)}
    A('--- combined, 400 000 events per row (both seeds) ---')
    tf = f0(comb['truth'])
    A('  truth      f_0 = %+.5f +- %.5f' % tf)
    for m in MODES + ('none',):
        a = f0(comb[m])
        e = np.hypot(a[1], tf[1])
        A('  %-9s f_0 = %+.5f +- %.5f   D = %+.5f +- %.5f   %+.2f sigma'
          % (m, a[0], a[1], a[0] - tf[0], e, (a[0] - tf[0]) / e))
    A('')

    # ---- the pt threshold asymmetry -------------------------------------
    s2 = {k: read_npz([S2], k) for k in ('truth',) + MODES}
    A('--- pt(l+ l-) near the 1 GeV threshold (sample 2), weight fraction ---')
    edges = [1, 2, 3, 5, 8, 12, 20, 40, np.inf]
    A('  %-9s %s' % ('', ' '.join('%7.0f' % e for e in edges[:-1])))
    for t in ('truth',) + MODES:
        o = s2[t]
        r = [o['w'][(o['pt_ee'] >= edges[i]) & (o['pt_ee'] < edges[i + 1])
                    ].sum() / o['w'].sum() for i in range(len(edges) - 1)]
        A('  %-9s %s' % (t, ' '.join('%7.5f' % x for x in r)))
    A('')
    A('  the hole the analysis re-cut cannot fill: pt(l+l-) < 2 GeV on either')
    A('  pair, where the production side lost events to pt_min_pdg = {23: 1}')
    A('  acting on the z BEFORE the reshuffle')
    for t in ('truth', 'madspin'):
        o = s2[t]
        m = (o['pt_ee'] < 2) | (o['pt_mumu'] < 2)
        A('    %-9s weight fraction %.5f   f_0 there %+.4f'
          % (t, o['w'][m].sum() / o['w'].sum(), f0(o, m)[0]))
    o, om = s2['truth'], s2['madspin']
    dw = (o['w'][(o['pt_ee'] < 2) | (o['pt_mumu'] < 2)].sum() / o['w'].sum()
          - om['w'][(om['pt_ee'] < 2) | (om['pt_mumu'] < 2)].sum()
          / om['w'].sum())
    A('    missing weight %.5f; even at f_0 = 1 there it is worth %.6f on f_0'
      % (dw, dw * (1.0 - tf[0])))
    A('')

    # ---- the |cos theta| shape ------------------------------------------
    A('--- pooled |cos theta| (both pairs), ratio to the truth, sample 2 ---')
    be = np.linspace(0, 1, 11)

    def pooled(o):
        c = np.concatenate([np.abs(o['cos_theta1']), np.abs(o['cos_theta2'])])
        w = np.concatenate([o['w'], o['w']])
        h, _ = np.histogram(c, bins=be, weights=w)
        return h / w.sum()
    ht = pooled(s2['truth'])
    A('  %-9s %s' % ('bin from', ' '.join('%7.1f' % x for x in be[:-1])))
    for t in MODES:
        A('  %-9s %s' % (t, ' '.join('%+7.4f' % x
                                     for x in pooled(s2[t]) / ht - 1)))
    A('')

    # ---- reweighting ----------------------------------------------------
    if not args.no_lhe:
        srcs = {'truth': sorted(glob.glob(pjoin(T130, 'events',
                                                'qq_pp4l_nlo', '*.lhe.gz')))}
        for m in MODES:
            srcs[m] = [pjoin(T130, 'work', 'ms_%s' % m,
                             'events_decayed.lhe.gz')]
        s2x = {}
        for t, ps in srcs.items():
            w, ex = extras(ps)
            z = np.load(pjoin(S2, 'events_%s.npz' % t))
            obs = {k: z[k] for k in z.files if k != 'w'}
            if not np.allclose(z['w'], w):
                raise RuntimeError('LHE and stored columns disagree on %s' % t)
            s2x[t] = select(obs, w, ex)

        A('--- f_0 of each mode after reweighting to the TRUTH distribution ---')
        A('    (sample 2; the deficit at fixed production kinematics)')
        trx = s2x['truth']
        tfx = f0(trx)

        def rw(fs, edges_, tag):
            Ht, _ = np.histogramdd([f(trx) for f in fs], bins=edges_,
                                   weights=trx['w'])
            row = '  %-30s' % tag
            for t in MODES:
                o = s2x[t]
                Hm, _ = np.histogramdd([f(o) for f in fs], bins=edges_,
                                       weights=o['w'])
                idx = tuple(np.clip(np.digitize(f(o), e) - 1, 0, len(e) - 2)
                            for f, e in zip(fs, edges_))
                den = Hm[idx]
                ok = np.abs(den) > 1e-12
                r = np.where(ok, Ht[idx] / np.where(ok, den, 1.0), 0.0)
                a = wmean(0.5 * (o['pol0_1'] + o['pol0_2']), o['w'] * r)
                e = np.hypot(a[1], tfx[1])
                row += '  %+.5f (%+.2f)' % (a[0] - tfx[0], (a[0] - tfx[0]) / e)
            A(row)

        A('  %-30s  %s' % ('reweighted in',
                           '  '.join('%-16s' % m for m in MODES)))
        rw([lambda o: o['m_4l']], [np.array([-1e9, 1e9])],
           'nothing (one bin)')
        rw([lambda o: o['m_4l']],
           [np.concatenate([[0], np.linspace(180, 600, 43), [1e9]])],
           'm_4l, 44 bins')
        rw([lambda o: o['pt4l']],
           [np.concatenate([[-1, 1], np.linspace(5, 200, 40), [1e9]])],
           'pt(4l), 42 bins')
        E6 = np.linspace(O.M_LO, O.M_HI, 7)
        rw([lambda o: o['m_ee'], lambda o: o['m_mumu']], [E6, E6],
           'm_ee x m_mumu, 6 x 6')
        E4 = np.linspace(O.M_LO, O.M_HI, 5)
        rw([lambda o: o['m_ee'], lambda o: o['m_mumu'], lambda o: o['m_4l']],
           [E4, E4, np.concatenate([[0], np.linspace(180, 500, 9), [1e9]])],
           'm_ee x m_mumu x m_4l, 4x4x10')
        rw([lambda o: o['pt4l'], lambda o: o['m_4l']],
           [np.array([-1, 1, 30, 80, 1e9]),
            np.concatenate([[0], np.linspace(180, 600, 15), [1e9]])],
           'pt(4l) x m_4l, 4 x 16')
        rw([lambda o: np.abs(o['y4l'])],
           [np.concatenate([np.linspace(0, 3, 13), [99]])], '|y_4l|, 13 bins')
        A('')

        A('--- split by extra-parton multiplicity (sample 2) ---')
        for nj in (0, 1):
            mt = trx['njet'] == nj
            a = f0(trx, mt)
            A('  njet = %d   truth f_0 = %+.5f +- %.5f   weight fraction %.4f'
              % (nj, a[0], a[1], trx['w'][mt].sum() / trx['w'].sum()))
            for t in MODES:
                o = s2x[t]
                m = o['njet'] == nj
                b = f0(o, m)
                e = np.hypot(a[1], b[1])
                A('              %-9s %+.5f +- %.5f  wfrac %.4f  D = %+.5f'
                  '  %+.2f sigma'
                  % (t, b[0], b[1], o['w'][m].sum() / o['w'].sum(),
                     b[0] - a[0], (b[0] - a[0]) / e))
        A('')

    # ---- the MC@NLO weight sign -----------------------------------------
    A('--- split by the sign of the MC@NLO weight (combined, 400 000) ---')
    tr, ms = comb['truth'], comb['madspin']
    for lab, f in (('w > 0', lambda w: w > 0), ('w < 0', lambda w: w < 0)):
        a = f0(tr, f(tr['w']))
        b = f0(ms, f(ms['w']))
        e = np.hypot(a[1], b[1])
        A('  %-6s truth %+.5f +- %.5f   madspin %+.5f +- %.5f'
          '   D = %+.5f +- %.5f  %+.2f sigma'
          % (lab, a[0], a[1], b[0], b[1], b[0] - a[0], e, (b[0] - a[0]) / e))
        A('         share of sum|w|: truth %.4f   madspin %.4f'
          % (np.abs(tr['w'][f(tr['w'])]).sum() / np.abs(tr['w']).sum(),
             np.abs(ms['w'][f(ms['w'])]).sum() / np.abs(ms['w']).sum()))
    A('  the two subsets are the counter-events of two DIFFERENT processes')
    A('  (p p > 4l and p p > z z), so only the signed sums have to agree.')
    A('')

    # ---- what the g g block excludes ------------------------------------
    if os.path.isdir(GG):
        g = {k: read_npz([GG], k) for k in ('truth', 'madspin')}
        a, b = f0(g['truth']), f0(g['madspin'])
        e = np.hypot(a[1], b[1])
        A('--- what the g g block actually excludes ---')
        A('  g g truth f_0 = %+.5f, madspin %+.5f, D = %+.5f +- %.5f'
          '  %+.2f sigma' % (a[0], b[0], b[0] - a[0], e, (b[0] - a[0]) / e))
        dqq = f0(comb['madspin'])[0] - tf[0]
        A('  an ABSOLUTE effect of the q qbar size (%+.5f) would show at'
          ' %.2f sigma' % (dqq, abs(dqq) / e))
        prop = dqq / tf[0] * a[0]
        A('  a PROPORTIONAL one (%.1f %% of f_0) would be %+.5f, i.e. %.2f'
          ' sigma' % (100 * dqq / tf[0], prop, abs(prop) / e))
        A('')

    txt = '\n'.join(L) + '\n'
    print(txt)
    os.makedirs(args.out, exist_ok=True)
    open(pjoin(args.out, 'numbers_qq_breakdown.txt'), 'w').write(txt)


if __name__ == '__main__':
    main()
