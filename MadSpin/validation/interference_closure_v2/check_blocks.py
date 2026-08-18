#!/usr/bin/env python3
"""Verify the nine-block decomposition, and establish which blocks a *card* can
name on the reworked pure-interference interface.

Run from the repository root:   python3 .../check_blocks.py

Part A -- the decomposition (unchanged from the first run of this test,
``../interference_closure/check_blocks.py``).  For one decaying particle the
per-particle restriction can take exactly three inequivalent,
transposition-closed forms on a two-state helicity basis {-1,+1}:

    D-  = {-1}          (symmetric)     -> the single entry (-1,-1)
    D+  = {+1}          (symmetric)     -> the single entry (+1,+1)
    I   = ({+1},{-1})   (cross)         -> the pair (+1,-1), (-1,+1)

``_restriction_row_mask`` builds the mask as a *product of per-index
conditions*, so with two decaying particles the 3 x 3 = 9 products of these
forms must (a) be pairwise disjoint, (b) cover all 16 (bra, ket) entries of the
joint 4 x 4 density matrix exactly once, and (c) have their nine contractions
add up to the unrestricted one.

Part B -- what the card can express.  This is what changed since the first run.
``pure_interference`` used to require the two sides of an entry to be DISJOINT,
so only ``I`` was spellable and every diagonal factor of a mixed block had to
come from a production brace on the other leg; the ``;`` spelling of a
multi-particle request was silently truncated, so ``(I,I)`` was not reachable at
all and had to be obtained by subtraction.  Now:

  * two IDENTICAL sides name the diagonal block ``D_S`` of that particle
    (``normalize_hel_restriction`` collapses ``(S, S)`` back to ``S``);
  * repeated ``set`` lines accumulate, so a card can name both legs;
  * a PARTIAL overlap is refused;
  * the ``;`` spelling raises instead of truncating;
  * ``_validate_pure_interference`` still requires at least one particle to
    carry a genuine (disjoint) interference pair.

Part B drives the real ``MadSpinInterface`` -- ``exec_cmd`` through the same
``precmd`` a card line goes through -- for all nine blocks and prints, for each,
whether the card names it and whether the restriction it produces is the
intended one.
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import MadSpin.decay as madspin
from MadSpin import interface_madspin

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
# how each form is spelled on the two sides of a card entry
CARD = {'D+': ('+', '+'), 'D-': ('-', '-'), 'I': ('+', '-')}


def part_a():
    prod = two(11, 12)
    dec = two(21, 22)
    labels = [tuple(int(v) for v in l) for l in prod.helicities]
    n_entries = len(labels)
    print('joint basis: %d entries (expect 16)' % n_entries)
    assert n_entries == 16

    full = complex(prod.scalar_multiplication(dec))

    seen = {}
    total = 0j
    blocks = {}
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
        blocks[(na, nb)] = spec
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

    diag = 0j
    for a, b in itertools.product([DP, DM], [DP, DM]):
        q = two(11, 12)
        q.set_hel_restriction([a, b])
        diag += complex(q.scalar_multiplication(dec))
    print()
    print('  4 diagonal blocks only : %+.8f' % diag.real)
    print('  missing (interference) : %+.8f  (%.1f%% of the total)'
          % ((full - diag).real, 100 * abs((full - diag) / full)))
    return blocks


def _card(lines):
    """Feed card lines to a real MadSpinInterface; return it, or the exception."""
    ms = interface_madspin.MadSpinInterface()
    try:
        for line in lines:
            ms.exec_cmd(line, precmd=True)
    except Exception as error:          # InvalidCmd and friends
        return None, error
    return ms, None


def part_b(blocks):
    """Which of the nine blocks can a card name, and does it name them right?"""
    print()
    print('=' * 78)
    print('Part B: what a *card* can name (real MadSpinInterface, via precmd)')
    print('=' * 78)
    print()
    print('  %-10s  %-46s  %s' % ('block', 'card lines', 'result'))
    reachable = {}
    for (na, a), (nb, b) in itertools.product(FORMS, FORMS):
        # pdg spellings: a bare MadSpinInterface carries no model, so 't' /
        # 't~' cannot be resolved to a pdg here (a real card, run after the
        # model is loaded, takes either spelling).
        lines = ['set pure_interference 6  = %s %s' % CARD[na],
                 'set pure_interference -6 = %s %s' % CARD[nb]]
        ms, error = _card(lines)
        spelled = '; '.join(l.split('pure_interference')[1].strip()
                            for l in lines)
        if ms is None:
            reachable[(na, nb)] = False
            print('  (%-3s,%-3s)  %-46s  REFUSED by do_set/precmd' % (na, nb, spelled))
            continue
        # parse, then validate exactly as a launch would
        parsed = ms._pure_interference()
        try:
            ms.options['spinmode'] = 'onshell'
            ms.list_branches = {}
            ms._validate_pure_interference()
        except Exception as error:
            reachable[(na, nb)] = False
            reason = str(error).split('.')[0]
            print('  (%-3s,%-3s)  %-46s  REFUSED: %s' % (na, nb, spelled, reason[:60]))
            continue
        # the parsed spec has to be the block we meant: run it through the same
        # normalisation the density matrix applies and compare to part A
        spec = madspin.DensityMatrix.normalize_hel_restriction(
            [parsed[6], parsed[-6]])
        ok = spec == blocks[(na, nb)]
        reachable[(na, nb)] = True
        print('  (%-3s,%-3s)  %-46s  named directly, spec %s %s'
              % (na, nb, spelled, spec, 'OK' if ok else '*** MISMATCH ***'))
        assert ok, 'card spec %s != block spec %s' % (spec, blocks[(na, nb)])

    n_named = sum(reachable.values())
    print()
    print('  blocks a card names directly : %d / 9' % n_named)
    print('  blocks needing a production brace: %s'
          % ', '.join('(%s,%s)' % k for k, v in sorted(reachable.items()) if not v))

    # the five interference blocks -- at least one I -- must all be reachable
    for key, ok in reachable.items():
        expect = ('I' in key)
        assert ok == expect, \
            'block %s: reachable=%s but "has an I index"=%s' % (key, ok, expect)
    print('  => exactly the 5 blocks with an I index are card-nameable, and the')
    print('     4 diagonal-diagonal ones are not (they are not interference at')
    print('     all: _validate_pure_interference requires one disjoint pair).')

    # ---- the refusals the reworked interface promises ---------------------
    print()
    print('  refusals:')
    ms, error = _card(['set pure_interference 6 = + - ; -6 = + -'])
    print('    %-42s %s' % ("';' spelling",
                            'raises' if ms is None else '*** ACCEPTED ***'))
    assert ms is None
    # partial overlap: {+,-} vs {-} share -1 without being equal
    ms3 = interface_madspin.MadSpinInterface()
    ms3.exec_cmd('set pure_interference 6  = +,- -', precmd=True)
    try:
        ms3._pure_interference()
        overlap_refused = False
    except Exception:
        overlap_refused = True
    print('    %-42s %s' % ('partial overlap ({+,-} vs {-})',
                            'raises' if overlap_refused else '*** ACCEPTED ***'))
    assert overlap_refused

    # accumulation
    ms = interface_madspin.MadSpinInterface()
    ms.exec_cmd('set pure_interference 6  = + -', precmd=True)
    ms.exec_cmd('set pure_interference -6 = + -', precmd=True)
    print('    %-42s %r' % ('two set lines accumulate to',
                            ms.options['pure_interference']))
    assert ms.options['pure_interference'] == '6 = + - ; -6 = + -'
    return reachable


def main():
    blocks = part_a()
    part_b(blocks)
    print()
    print('OK: 3 forms per particle, 9 disjoint blocks tiling the matrix; the 5')
    print('    blocks carrying an I index are named directly from the card --')
    print('    including (I,I), which the first run had to obtain by subtraction')
    print('    -- and the 4 diagonal-diagonal ones come from production braces.')


if __name__ == '__main__':
    main()
