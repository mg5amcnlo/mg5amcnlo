#!/usr/bin/env python3
"""The NLO truth with and without its four singly-resonant Born diagrams.

Combines the study's two independently seeded 200 000-event samples for the
full truth and for each MadSpin mode (400 000 each) and puts the new
doubly-resonant-only NLO truth of ``run_nlo_zzonly.py`` beside them, on the
study's own selection re-imposed offline on all of them alike.

    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    python3 harvest_nlo_zzonly.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
import observables as O                                          # noqa: E402

pjoin = os.path.join
PT_MIN = 1.0
S1 = os.path.expanduser('~/Documents/git_workspace/zz_spin_events/'
                        'qq_ppzz_nlo_madspin')
S2 = os.path.expanduser('~/Documents/madspin_validation_samples/'
                        't130_qq_seed_check/event_columns')


def wmean(x, w):
    sw = w.sum()
    m = float(np.dot(w, x) / sw)
    e = float(np.sqrt(np.dot(w ** 2, (x - m) ** 2)) / abs(sw))
    return m, e


def cut(obs, w):
    sel = ((obs['m_ee'] > O.M_LO) & (obs['m_ee'] < O.M_HI)
           & (obs['m_mumu'] > O.M_LO) & (obs['m_mumu'] < O.M_HI)
           & (obs['pt_ee'] > PT_MIN) & (obs['pt_mumu'] > PT_MIN))
    out = {k: v[sel] for k, v in obs.items()}
    out['w'] = w[sel]
    return out


def from_npz(paths, key):
    obs, ws = None, []
    for p in paths:
        z = np.load(pjoin(p, 'events_%s.npz' % key))
        d = {k: z[k] for k in z.files if k != 'w'}
        ws.append(z['w'])
        obs = d if obs is None else {k: np.concatenate([obs[k], d[k]])
                                     for k in obs}
    return cut(obs, np.concatenate(ws))


def from_lhe(paths):
    ws, obs = [], None
    for p in paths:
        w, mom = O.read_lhe(p)
        d = O.compute(mom)
        ws.append(w)
        obs = d if obs is None else {k: np.concatenate([obs[k], d[k]])
                                     for k in obs}
    return cut(obs, np.concatenate(ws))


def numbers(o):
    w = o['w']
    f0 = 0.5 * (o['pol0_1'] + o['pol0_2'])
    d = {'N': int(len(w)),
         'N_eff': float(w.sum() ** 2 / (w ** 2).sum()),
         'sigma_pb': float(w.sum() / len(w)),
         'f_0 (both)': list(wmean(f0, w)),
         'f_00': list(wmean(o['pol00'], w)),
         'f_TT': list(wmean(o['polTT'], w)),
         'cos1cos2': list(wmean(o['cos1cos2'], w))}
    d['C_kk'] = list(O.c_kk_from_moment(*d['cos1cos2']))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zzonly', default=os.path.expanduser(
        '~/Documents/madspin_validation_samples/t131_qq_diagrams/nlo/'
        'pp4l_zzonly_nlo/Events'))
    ap.add_argument('--out', default=pjoin(_HERE, '..', 'data'))
    args = ap.parse_args()

    S = {}
    for key in ('truth', 'madspin', 'onshell', 'PA', 'none'):
        S[key] = from_npz([S1, S2], key)
    zz = sorted(glob.glob(pjoin(args.zzonly, 'z*', 'events.lhe.gz')))
    if zz:
        S['truth_zzonly'] = from_lhe(zz)
    res = {k: numbers(v) for k, v in S.items()}

    L = []
    A = L.append
    A('The NLO q qbar truth with and without its singly-resonant Born diagrams')
    A('')
    A('  truth         p p > e+ e- mu+ mu- / a [QCD]                    6 Born')
    A('                                                                 diagrams'
      ' per subprocess')
    A('  truth_zzonly  p p > e+ e- mu+ mu- / a e+ e- mu+ mu- [QCD]      2, the'
      ' doubly-resonant ones')
    A('  madspin/...   p p > z z [QCD] + MadSpin')
    A('')
    A('truth and the MadSpin rows are the study\'s two independently seeded')
    A('200 000-event samples combined; %d blocks of the new NLO run.' % len(zz))
    A('Same run card, same zz_equivalent_cuts_nlo.f, same window re-imposed')
    A('offline on every row.')
    A('')
    A('%-14s %9s %11s %13s' % ('sample', 'N', 'N_eff', 'sigma [pb]'))
    for t, r in res.items():
        A('%-14s %9d %11.0f %13.6f' % (t, r['N'], r['N_eff'], r['sigma_pb']))
    A('')
    names = ['f_0 (both)', 'f_00', 'f_TT', 'C_kk']
    A('%-14s %s' % ('sample', ' '.join('%-21s' % n for n in names)))
    for t, r in res.items():
        A('%-14s %s' % (t, ' '.join('%-21s' % ('%+.4f+-%.4f' % tuple(r[n]))
                                    for n in names)))
    A('')
    A('--- the differences ---')

    def diff(a, b_, note):
        if a not in res or b_ not in res:
            return
        for n in names:
            x, ex = res[a][n]
            y, ey = res[b_][n]
            e = np.hypot(ex, ey)
            A('  %-30s %-11s %+.5f +- %.5f   %+.2f sigma'
              % ('%s - %s' % (a, b_), n, x - y, e, (x - y) / e))
        A('    %s' % note)
        A('')

    diff('truth', 'truth_zzonly',
         'the four singly-resonant Born diagrams at NLO, and nothing else')
    diff('madspin', 'truth', 'the finding under investigation, both samples')
    diff('madspin', 'truth_zzonly',
         'what the finding becomes once the truth is stripped of the '
         'diagrams MadSpin cannot carry')
    for m in ('onshell', 'PA'):
        diff(m, 'truth_zzonly', '')

    txt = '\n'.join(L) + '\n'
    print(txt)
    os.makedirs(args.out, exist_ok=True)
    open(pjoin(args.out, 'numbers_nlo_zzonly.txt'), 'w').write(txt)
    json.dump(res, open(pjoin(args.out, 'numbers_nlo_zzonly.json'), 'w'),
              indent=1)


if __name__ == '__main__':
    main()
