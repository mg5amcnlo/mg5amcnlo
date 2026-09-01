#!/usr/bin/env python3
"""f_0 of the three LO samples of ``run_lo_diagrams.py``, on ONE offline cut.

Reads the LHE files, computes the study's own observables with the study's own
``observables.py``, applies the study's own window and pt cut as an analysis
filter to every sample alike, and prints the two differences that matter:

    f_0(full4l) - f_0(zz4l)        the four singly-resonant Born diagrams
    f_0(zz4l)  - f_0(zz+MadSpin)   everything else in the chain, at LO

    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    python3 harvest_lo_diagrams.py
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
import observables as O                                          # noqa: E402

pjoin = os.path.join
PT_MIN = 1.0


def wmean(x, w):
    sw = w.sum()
    m = float(np.dot(w, x) / sw)
    e = float(np.sqrt(np.dot(w ** 2, (x - m) ** 2)) / abs(sw))
    return m, e


def load(path):
    w, mom = O.read_lhe(path)
    obs = O.compute(mom)
    sel = ((obs['m_ee'] > O.M_LO) & (obs['m_ee'] < O.M_HI)
           & (obs['m_mumu'] > O.M_LO) & (obs['m_mumu'] < O.M_HI)
           & (obs['pt_ee'] > PT_MIN) & (obs['pt_mumu'] > PT_MIN))
    out = {k: v[sel] for k, v in obs.items()}
    out['w'] = w[sel]
    out['_n_all'] = len(w)
    out['_n_sel'] = int(sel.sum())
    out['_sumw_all'] = float(w.sum())
    return out


def numbers(o):
    w = o['w']
    f0 = 0.5 * (o['pol0_1'] + o['pol0_2'])
    f01, _ = wmean(o['pol0_1'], w)
    f02, _ = wmean(o['pol0_2'], w)
    d = {'N_all': o['_n_all'], 'N': o['_n_sel'],
         'N_eff': float(w.sum() ** 2 / (w ** 2).sum()),
         'sigma_pb_all': o['_sumw_all'] / o['_n_all'],
         'sigma_pb_sel': float(w.sum() / o['_n_all']),
         'f_0 (e+ e-)': [f01, wmean(o['pol0_1'], w)[1]],
         'f_0 (mu+ mu-)': [f02, wmean(o['pol0_2'], w)[1]],
         'f_0 (both)': list(wmean(f0, w)),
         'f_00': list(wmean(o['pol00'], w)),
         # the rank-2 correlation, per event so the bar keeps the covariance
         # of the two sides -- the fourth column of Table 1
         'f_00 - f_0 f_0': list(wmean(o['pol00'] - f01 * f02, w)),
         'f_TT': list(wmean(o['polTT'], w)),
         'cos1cos2': list(wmean(o['cos1cos2'], w)),
         # the 40 GeV same-flavour floor of the generator-level run card is
         # inert if the CROSS pairs reach below it; measured, not assumed
         'frac m(e+ mu-) < 40 GeV': float((o['m_epmum'] < 40).mean()),
         'min m_ee': float(o['m_ee'].min()),
         'max m_ee': float(o['m_ee'].max()),
         'min pt_ee': float(o['pt_ee'].min())}
    d['C_kk'] = list(O.c_kk_from_moment(*d['cos1cos2']))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--basedir', default=os.path.expanduser(
        '~/Documents/madspin_validation_samples/t131_qq_diagrams/work'))
    ap.add_argument('--out', default=pjoin(_HERE, '..', 'data'))
    args = ap.parse_args()
    b = args.basedir
    paths = {
        'full4l': pjoin(b, 'lo_full4l', 'Events', 'prod',
                        'unweighted_events.lhe.gz'),
        'zz4l': pjoin(b, 'lo_zz4l', 'Events', 'prod',
                      'unweighted_events.lhe.gz'),
        'zz_madspin': pjoin(b, 'ms_madspin', 'events_decayed.lhe.gz'),
        'zz_none': pjoin(b, 'ms_none', 'events_decayed.lhe.gz'),
    }
    res, cols = {}, {}
    for tag, p in paths.items():
        if not os.path.exists(p):
            print('missing %s: %s' % (tag, p))
            continue
        o = load(p)
        cols[tag] = o
        res[tag] = numbers(o)

    L = []
    A = L.append
    A('LO diagram-content test of the q qbar f_0 deficit')
    A('')
    A('  full4l      p p > e+ e- mu+ mu- / a                 (6 Born diagrams)')
    A('  zz4l        p p > e+ e- mu+ mu- / a e+ e- mu+ mu-   (2, doubly res.)')
    A('  zz_madspin  p p > z z + MadSpin spinmode = madspin')
    A('  zz_none     p p > z z + MadSpin spinmode = none')
    A('')
    A('one offline selection on all four: |m(l+l-) - %.4f| < %.4f GeV and'
      % (O.M_Z, O.BW_CUT * O.W_Z))
    A('pt(l+l-) > %.1f GeV on both reconstructed pairs.  NEITHER cut is'
      % PT_MIN)
    A('applied at the generator level on any of them, so neither can see a z')
    A('on one side and a lepton pair on the other -- which is the asymmetry the')
    A('NLO comparison carries.  The one generator-level threshold there is, a')
    A('40 GeV floor on the same-flavour pairs, is measured inert below.')
    A('')
    A('%-12s %9s %9s %11s %13s' % ('sample', 'N_all', 'N_sel', 'N_eff',
                                   'sigma [pb]'))
    for t, r in res.items():
        A('%-12s %9d %9d %11.0f %13.6f'
          % (t, r['N_all'], r['N'], r['N_eff'], r['sigma_pb_sel']))
    A('')
    names = ['f_0 (both)', 'f_00', 'f_00 - f_0 f_0', 'f_TT', 'C_kk']
    A('the five columns of Table 1 of spin_definitions.tex')
    A('%-12s %s' % ('sample', ' '.join('%-20s' % n for n in names)))
    for t, r in res.items():
        A('%-12s %s' % (t, ' '.join('%-20s' % ('%+.4f+-%.4f' % tuple(r[n]))
                                    for n in names)))
    A('')
    A('per-Z: %s' % '  '.join(
        '%s %+.4f+-%.4f' % ((n,) + tuple(res['full4l'][n]))
        for n in ('f_0 (e+ e-)', 'f_0 (mu+ mu-)') if 'full4l' in res))
    A('')
    A('--- the selection actually reached, sample by sample ---')
    A('  the window is %.4f .. %.4f GeV and pt(l+l-) > %.1f GeV'
      % (O.M_LO, O.M_HI, PT_MIN))
    A('  the run card carries mxx_min_pdg = {11: 40, 13: 40} with')
    A('  mxx_only_part_antipart, i.e. a 40 GeV floor on the SAME-FLAVOUR pairs')
    A('  only.  It sits below the window edge and is therefore inert; the')
    A('  cross-pair fraction below 40 GeV shows it did not reach them either.')
    A('  %-12s %10s %10s %10s %22s'
      % ('sample', 'min m_ee', 'max m_ee', 'min pt_ee', 'frac m(e+mu-) < 40'))
    for t, r in res.items():
        A('  %-12s %10.4f %10.4f %10.4f %22.5f'
          % (t, r['min m_ee'], r['max m_ee'], r['min pt_ee'],
             r['frac m(e+ mu-) < 40 GeV']))
    A('')
    A('--- the differences ---')

    def diff(a, b_, note):
        if a not in res or b_ not in res:
            return
        for n in ('f_0 (both)', 'f_00 - f_0 f_0', 'f_TT', 'C_kk'):
            x, ex = res[a][n]
            y, ey = res[b_][n]
            e = np.hypot(ex, ey)
            A('  %-24s %-12s %+.5f +- %.5f   %+.2f sigma'
              % ('%s - %s' % (a, b_), n, x - y, e, (x - y) / e))
        A('    %s' % note)
        A('')

    diff('full4l', 'zz4l',
         'the four singly-resonant Born diagrams, and nothing else')
    diff('full4l', 'zz_madspin',
         'the LO twin of the NLO comparison under test')
    diff('zz4l', 'zz_madspin',
         'what is left once the extra diagrams are taken out of the truth')

    txt = '\n'.join(L) + '\n'
    print(txt)
    os.makedirs(args.out, exist_ok=True)
    open(pjoin(args.out, 'numbers_lo_diagrams.txt'), 'w').write(txt)
    json.dump(res, open(pjoin(args.out, 'numbers_lo_diagrams.json'), 'w'),
              indent=1)
    # The per-event columns are ~260 MB and belong with the events, not in the
    # repository: they go to the durable sample tree beside the LHE files.
    colsdir = pjoin(b, '..', 'event_columns')
    os.makedirs(colsdir, exist_ok=True)
    np.savez_compressed(
        pjoin(colsdir, 'lo_diagrams_columns.npz'),
        **{'%s__%s' % (t, k): v for t, o in cols.items()
           for k, v in o.items() if not k.startswith('_')})
    print('per-event columns: %s' % pjoin(colsdir, 'lo_diagrams_columns.npz'))


if __name__ == '__main__':
    main()
