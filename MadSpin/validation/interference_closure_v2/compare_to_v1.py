#!/usr/bin/env python3
"""Cross-check the new card spelling of a mixed block against the old braced one.

    compare_to_v1.py <v2_data_dir> <v1_data_dir>

The four mixed blocks are the only ones whose *recipe* changed.  In the first
run of this test the diagonal factor of `(I,D+)` came from a production brace on
the other leg -- the sample was `p p > t t~{+}` with `pure_interference t = + -`
-- because the card could not spell a diagonal entry.  Here it comes from the
card, `set pure_interference t~ = + +`, on an unpolarised `p p > t t~`.

Both are the same block of the same density matrix and both are quoted in pb,
so their histograms must agree bin by bin.  They do not share a single event, a
production process or a normalisation scheme (the first run reconstructed
`max_weight / c` by hand; this one reads `sum w / N_file` straight off the
file), so agreement here tests the new card spelling AND the two normalisations
against each other.

`(I,I)` has no v1 counterpart to compare against -- v1 obtained it by
subtraction, which is exactly what this rework removes -- so it is compared to
`x_t - (I,D+) - (I,D-)` of the old run, the subtraction route itself.
"""

import math
import os
import sys

import numpy as np

# v2 tag -> (v1 positive tags, v1 negative tags)
PAIRS = [
    ('i_dp', (['i_tbp'], []),  '(I,D+)   card t~=+ +   vs   brace t~{+}'),
    ('i_dm', (['i_tbm'], []),  '(I,D-)   card t~=- -   vs   brace t~{-}'),
    ('dp_i', (['i_tp'], []),   '(D+,I)   card t =+ +   vs   brace t{+}'),
    ('dm_i', (['i_tm'], []),   '(D-,I)   card t =- -   vs   brace t{-}'),
    ('ii',   (['x_t'], ['i_tbp', 'i_tbm']),
     '(I,I)    named directly   vs   v1 subtraction x_t-(I,D+)-(I,D-)'),
]

KEYS = ['cos_k_p', 'cos_k_m', 'ckk', 'cnn', 'crr', 'cos_phi', 'cos_n_p',
        'dphi_lab', 'pt_t', 'm_tt']


def chi2(a, ae, b, be):
    d = a - b
    s2 = ae ** 2 + be ** 2
    good = (s2 > 0) & np.isfinite(d)
    return float((d[good] ** 2 / s2[good]).sum()), int(good.sum())


def main():
    v2 = np.load(os.path.join(sys.argv[1], 'histograms.npz'))
    v1 = np.load(os.path.join(sys.argv[2], 'histograms.npz'))

    print('block-by-block agreement between the two runs, per-bin chi2 / 20 bins')
    print('(both sides in pb, no free parameter anywhere)')
    print()
    print('  %-62s %s' % ('block', '  '.join('%9s' % k for k in KEYS)))
    for tag, (plus, minus), label in PAIRS:
        row = []
        for key in KEYS:
            a = v2['sumw__%s__%s' % (tag, key)]
            ae = np.sqrt(v2['sumw2__%s__%s' % (tag, key)])
            b = (sum(v1['sumw__%s__%s' % (t, key)] for t in plus)
                 - sum(v1['sumw__%s__%s' % (t, key)] for t in minus))
            be = np.sqrt(sum(v1['sumw2__%s__%s' % (t, key)]
                             for t in list(plus) + list(minus)))
            c, n = chi2(a, ae, b, be)
            row.append('%9.1f' % c)
        print('  %-62s %s' % (label, '  '.join(row)))
    print()
    print('  (20 bins each; a value around 20 is agreement)')

    # and the integral of each block, which must be zero on both sides
    print()
    print('  %-10s %26s %26s' % ('block', 'v2 integral [pb]', 'v1 integral [pb]'))
    for tag, (plus, minus), _label in PAIRS:
        m2 = v2['mom__%s__%s' % (tag, 'cos_k_p')]
        s2, e2 = m2[0], math.sqrt(m2[2])
        s1 = sum(v1['mom__%s__%s' % (t, 'cos_k_p')][0] for t in plus) \
            - sum(v1['mom__%s__%s' % (t, 'cos_k_p')][0] for t in minus)
        e1 = math.sqrt(sum(v1['mom__%s__%s' % (t, 'cos_k_p')][2]
                           for t in list(plus) + list(minus)))
        print('  %-10s %+14.6f +- %-9.6f %+14.6f +- %-9.6f'
              % (tag, s2, e2, s1, e1))


if __name__ == '__main__':
    main()
