#!/usr/bin/env python3
"""Verify the nine-block decomposition against the code, not against the prose.

Run from the repository root:   python3 .../check_blocks.py

Claim under test.  For one decaying particle the per-particle restriction can
take exactly three inequivalent, transposition-closed forms on a two-state
helicity basis {-1,+1}:

    D-  = {-1}          (symmetric)     -> the single entry (-1,-1)
    D+  = {+1}          (symmetric)     -> the single entry (+1,+1)
    I   = ({+1},{-1})   (cross)         -> the pair (+1,-1), (-1,+1)

``_restriction_row_mask`` builds the mask as a *product of per-index
conditions*, so with two decaying particles the 3 x 3 = 9 products of these
forms must (a) be pairwise disjoint, (b) cover all 16 (bra, ket) entries of the
joint 4 x 4 density matrix exactly once, and (c) have their nine contractions
add up to the unrestricted one.  That is the whole basis of the closure test.

It also checks the two coarser masks the closure test actually generates,
because a card cannot carry two ``pure_interference`` entries at once (see
README.md):

    (I  x  anything)  =  (I,D+) + (I,D-) + (I,I)
    (anything x  I )  =  (D+,I) + (D-,I) + (I,I)
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import MadSpin.decay as madspin

FERMION = [1, -1]


def packed(seed, n=2):
    """A hermitian matrix in the packed upper-triangular Fortran storage."""
    rng = np.random.default_rng(seed)
    arr = (rng.normal(size=n * (n + 1) // 2)
           + 1j * rng.normal(size=n * (n + 1) // 2)).astype('complex64')
    for i in range(n):
        arr[i * (2 * n - i + 1) // 2] = abs(arr[i * (2 * n - i + 1) // 2])
    return arr


def one(seed):
    return madspin.DensityMatrix(packed(seed), 1, FERMION, len(FERMION))


def two(seed_a, seed_b):
    return one(seed_a).tensor_product(one(seed_b))


DP = (1,)
DM = (-1,)
INT = ((1,), (-1,))
FORMS = [('D+', DP), ('D-', DM), ('I', INT)]


def main():
    prod = two(11, 12)
    dec = two(21, 22)
    labels = [tuple(int(v) for v in l) for l in prod.helicities]
    n_entries = len(labels)
    print('joint basis: %d entries (expect 16)' % n_entries)
    assert n_entries == 16

    full = complex(prod.scalar_multiplication(dec))

    seen = {}
    total = 0j
    print()
    print('  %-10s %8s   %s' % ('block', 'entries', 'contraction'))
    for (na, a), (nb, b) in itertools.product(FORMS, FORMS):
        spec = madspin.DensityMatrix.normalize_hel_restriction([a, b])
        p = two(11, 12)
        p.set_hel_restriction([a, b])
        mask = p._restriction_row_mask(p.hel_restriction)
        kept = [labels[i] for i in np.flatnonzero(mask)]
        for lab in kept:
            assert lab not in seen, 'entry %s in two blocks: %s and (%s,%s)' % (
                lab, seen.get(lab), na, nb)
            seen[lab] = (na, nb)
        val = complex(p.scalar_multiplication(dec))
        total += val
        # every block is closed under the global transposition, so its own
        # contraction is real
        assert abs(val.imag) < 1e-4 * max(1.0, abs(val)), \
            'block (%s,%s) is complex: %s' % (na, nb, val)
        print('  (%-3s,%-3s) %8d   %+12.6f %+12.6fj  spec=%s'
              % (na, nb, len(kept), val.real, val.imag, spec))

    print()
    print('entries covered  : %d / %d' % (len(seen), n_entries))
    assert len(seen) == n_entries, 'the nine blocks do not cover the matrix'
    print('sum of the blocks: %+.8f' % total.real)
    print('unrestricted     : %+.8f' % full.real)
    rel = abs(total - full) / abs(full)
    print('relative difference: %.3e' % rel)
    assert rel < 1e-5

    # ---- the two coarser masks the closure test really uses ---------------
    print()
    for name, spec, parts in (
            ('(I x anything)', [INT, None], [(INT, DP), (INT, DM), (INT, INT)]),
            ('(anything x I)', [None, INT], [(DP, INT), (DM, INT), (INT, INT)])):
        p = two(11, 12)
        p.set_hel_restriction(spec)
        val = complex(p.scalar_multiplication(dec))
        acc = 0j
        for a, b in parts:
            q = two(11, 12)
            q.set_hel_restriction([a, b])
            acc += complex(q.scalar_multiplication(dec))
        print('  %-15s = %+.8f   sum of its 3 blocks = %+.8f  (rel %.2e)'
              % (name, val.real, acc.real, abs(val - acc) / abs(val)))
        assert abs(val - acc) / abs(val) < 1e-5

    # ---- and that the diagonal-only sum is NOT the full one ---------------
    diag = 0j
    for a, b in itertools.product([DP, DM], [DP, DM]):
        q = two(11, 12)
        q.set_hel_restriction([a, b])
        diag += complex(q.scalar_multiplication(dec))
    print()
    print('  4 diagonal blocks only : %+.8f' % diag.real)
    print('  missing (interference) : %+.8f  (%.1f%% of the total)'
          % ((full - diag).real, 100 * abs((full - diag) / full)))
    print()
    print('OK: 3 forms per particle, 9 disjoint blocks, they tile the matrix '
          'and sum to the full contraction.')


if __name__ == '__main__':
    main()
