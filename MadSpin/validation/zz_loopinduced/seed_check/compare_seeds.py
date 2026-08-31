#!/usr/bin/env python3
"""Sample 1 against sample 2: does the q qbar f_0 pull survive a reseed?

Reads the two ``numbers_qq*.json`` written by ``qq_coefficients.py`` -- the same
code on both samples -- and forms, for every mode and every coefficient, the
difference against that sample's OWN truth:

    D = mode - truth,   sigma_D = sqrt(sigma_mode^2 + sigma_truth^2)

Both sides of both samples are independently seeded, so D1 and D2 are
independent and the combination is the inverse-variance mean.  Had the truth
been reused, D1 and D2 would have shared the truth's noise -- which is about
half the variance of D here -- and combining them as independent would have
inflated the answer.  That is why the truth was reseeded too.
"""
import json
import sys

NAMES = ['f_0 (e+ e-)', 'f_0 (mu+ mu-)', 'f_0 (both)', 'f_00',
         'f_00 - f_0 f_0', 'f_TT', 'C_kk']
MODES = ['madspin', 'onshell', 'PA', 'none']


def load(p):
    return json.load(open(p))['runs']


def diff(r, t, n):
    d = r[n][0] - t[n][0]
    s = (r[n][1] ** 2 + t[n][1] ** 2) ** 0.5
    return d, s


def main(p1, p2):
    A, B = load(p1), load(p2)
    ta, tb = A['truth'], B['truth']

    print('=' * 100)
    print('the two truths, side by side  (they are independent samples)')
    print('%-16s %-22s %-22s %-22s %6s' % ('coefficient', 'sample 1 truth',
                                           'sample 2 truth', 'difference', 'z'))
    for n in NAMES:
        d = tb[n][0] - ta[n][0]
        s = (tb[n][1] ** 2 + ta[n][1] ** 2) ** 0.5
        print('%-16s %+.4f +- %.4f       %+.4f +- %.4f       %+.4f +- %.4f  %+6.2f'
              % (n, ta[n][0], ta[n][1], tb[n][0], tb[n][1], d, s, d / s))
    print()

    for mode in MODES:
        if mode not in A or mode not in B:
            continue
        print('=' * 100)
        print('mode %s   (pull = mode - truth, in sigma; negative = mode low)'
              % mode)
        print('%-16s %-24s %-24s %-24s %7s %7s'
              % ('coefficient', 'sample 1  D +- s  (z)', 'sample 2  D +- s  (z)',
                 'combined  D +- s', 'z comb', 'chi2/1'))
        for n in NAMES:
            d1, s1 = diff(A[mode], ta, n)
            d2, s2 = diff(B[mode], tb, n)
            w1, w2 = 1 / s1 ** 2, 1 / s2 ** 2
            dc = (w1 * d1 + w2 * d2) / (w1 + w2)
            sc = (1 / (w1 + w2)) ** 0.5
            # are the two samples consistent with each other?
            chi = (d1 - d2) ** 2 / (s1 ** 2 + s2 ** 2)
            print('%-16s %+.4f+-%.4f (%+5.2f)  %+.4f+-%.4f (%+5.2f)  '
                  '%+.4f+-%.4f       %+6.2f  %6.2f'
                  % (n, d1, s1, d1 / s1, d2, s2, d2 / s2, dc, sc,
                     dc / sc, chi))
        print()

    print('=' * 100)
    print('the raw rows')
    for tag, lab in (('truth', 'truth'),) + tuple((m, m) for m in MODES):
        for src, name in ((A, 'sample 1'), (B, 'sample 2')):
            if tag not in src:
                continue
            r = src[tag]
            print('%-8s %-9s %s' % (lab, name, ' '.join(
                '%-21s' % ('%+.4f+-%.4f' % tuple(r[n])) for n in NAMES)))
    print('  columns: ' + '  '.join(NAMES))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
