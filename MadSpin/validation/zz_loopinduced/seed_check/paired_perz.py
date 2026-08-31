#!/usr/bin/env python3
"""The two per-Z longitudinal fractions of one sample, compared PAIRWISE.

The table quotes ``f_0 (e+ e-)`` and ``f_0 (mu+ mu-)`` with the bar each would
have as a stand-alone measurement.  Reading a difference between them off those
two bars in quadrature is wrong: they are the two Z of the SAME events, so the
difference has its own, smaller (or larger) variance, and only the per-event
difference ``d = pol0_1 - pol0_2`` measures it.  This is the same paired
construction T129 used for Delta phi(e+ e-) against Delta phi(mu+ mu-).

Both Z are equivalent by construction -- ``g g > z z`` and ``q qbar > z z`` are
symmetric under exchanging the two identical bosons, and MadSpin's positional
rule only decides which one is called (e+ e-) -- so E[d] = 0 is a null test.
"""
import glob
import os
import sys

import numpy as np

OBS = ('/Users/omattelaer/Documents/git_workspace/madgraph5/.claude/'
       'worktrees/qqseed/MadSpin/validation/zz_loopinduced')
sys.path.insert(0, OBS)
import observables as O                                          # noqa: E402

PT_MIN = 1.0

BLOCKS = [
    ('g g box', os.path.expanduser(
        '~/Documents/madspin_validation_samples/t118_zz_loopinduced/'
        'event_columns')),
    ('q qbar continuum, sample 1', os.path.expanduser(
        '~/Documents/git_workspace/zz_spin_events/qq_ppzz_nlo_madspin')),
    ('q qbar continuum, sample 2', os.path.expanduser(
        '~/Documents/madspin_validation_samples/t130_qq_seed_check/'
        'event_columns')),
]
ORDER = ('truth', 'madspin', 'PA', 'onshell', 'none')


def wmean(x, w):
    sw = w.sum()
    m = float(np.dot(w, x) / sw)
    e = float(np.sqrt(np.dot(w ** 2, (x - m) ** 2)) / abs(sw))
    return m, e


def one(path, recut):
    d = np.load(path)
    w = d['w']
    if recut:
        sel = ((d['m_ee'] > O.M_LO) & (d['m_ee'] < O.M_HI)
               & (d['m_mumu'] > O.M_LO) & (d['m_mumu'] < O.M_HI)
               & (d['pt_ee'] > PT_MIN) & (d['pt_mumu'] > PT_MIN))
    else:
        sel = np.ones(len(w), dtype=bool)
    w = w[sel]
    a, b = d['pol0_1'][sel], d['pol0_2'][sel]
    m1, e1 = wmean(a, w)
    m2, e2 = wmean(b, w)
    dm, de = wmean(a - b, w)                       # the paired estimator
    naive = (e1 ** 2 + e2 ** 2) ** 0.5             # the wrong yardstick
    # the correlation the pairing exploits (or fails to)
    mu = np.dot(w, a) / w.sum(), np.dot(w, b) / w.sum()
    cov = float(np.dot(w ** 2, (a - mu[0]) * (b - mu[1])))
    va = float(np.dot(w ** 2, (a - mu[0]) ** 2))
    vb = float(np.dot(w ** 2, (b - mu[1]) ** 2))
    rho = cov / (va * vb) ** 0.5
    return dict(n=int(sel.sum()), n_cut=int((~sel).sum()),
                m1=m1, e1=e1, m2=m2, e2=e2, d=dm, ed=de,
                naive=naive, rho=rho)


def main():
    recut = '--norecut' not in sys.argv
    print('paired per-Z null test:  d = f_0(e+ e-) - f_0(mu+ mu-),')
    print('the weighted mean of the PER-EVENT difference.')
    print('re-cut applied: %s   (window %.3f .. %.3f GeV, pt > %.1f GeV)'
          % ('yes' if recut else 'no', O.M_LO, O.M_HI, PT_MIN))
    print()
    for label, base in BLOCKS:
        print('--- %s   (%s)' % (label, base))
        print('%-9s %8s  %-19s %-19s  %-21s %-9s %6s %6s'
              % ('sample', 'N', 'f_0(e+e-)', 'f_0(mu+mu-)',
                 'paired d', 'naive bar', 'ratio', 'pull'))
        for tag in ORDER:
            p = os.path.join(base, 'events_%s.npz' % tag)
            if not os.path.exists(p):
                cands = glob.glob(os.path.join(base, '*%s*.npz' % tag))
                if not cands:
                    print('%-9s  (missing)' % tag)
                    continue
                p = cands[0]
            r = one(p, recut)
            print('%-9s %8d  %+.4f +- %.4f  %+.4f +- %.4f  '
                  '%+.4f +- %.4f  %.4f  %5.2f  %+5.2f'
                  % (tag, r['n'], r['m1'], r['e1'], r['m2'], r['e2'],
                     r['d'], r['ed'], r['naive'], r['ed'] / r['naive'],
                     r['d'] / r['ed']))
        print()


if __name__ == '__main__':
    main()
