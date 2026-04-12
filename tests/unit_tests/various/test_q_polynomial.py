################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Unit tests for madgraph/various/q_polynomial.py"""

from __future__ import absolute_import
import unittest

import madgraph.various.q_polynomial as q_polynomial

Polynomial = q_polynomial.Polynomial
Polynomial_naive_ordering = q_polynomial.Polynomial_naive_ordering
PolynomialError = q_polynomial.PolynomialError
get_number_of_coefs_for_rank = q_polynomial.get_number_of_coefs_for_rank


class TestGetNumberOfCoefsForRank(unittest.TestCase):
    """Tests for the helper function get_number_of_coefs_for_rank."""

    def test_rank_zero(self):
        """Rank-0 polynomial has exactly 1 coefficient (the scalar)."""
        self.assertEqual(get_number_of_coefs_for_rank(0), 1)

    def test_rank_one(self):
        """Rank-1 polynomial in 4 dimensions: 1 + 4 = 5 coefficients."""
        self.assertEqual(get_number_of_coefs_for_rank(1), 5)

    def test_rank_two(self):
        """Rank-2 symmetric tensor in 4-d: 5 + 10 = 15 coefficients."""
        self.assertEqual(get_number_of_coefs_for_rank(2), 15)

    def test_rank_three(self):
        """Rank-3: 15 + 20 = 35 coefficients."""
        self.assertEqual(get_number_of_coefs_for_rank(3), 35)

    def test_rank_four(self):
        """Rank-4: 35 + 35 = 70 coefficients."""
        self.assertEqual(get_number_of_coefs_for_rank(4), 70)

    def test_cumulative_growth(self):
        """Each rank adds strictly more coefficients than the previous one."""
        prev = get_number_of_coefs_for_rank(0)
        for r in range(1, 6):
            curr = get_number_of_coefs_for_rank(r)
            self.assertGreater(curr, prev)
            prev = curr


class TestPolynomialNaiveOrdering(unittest.TestCase):
    """Tests for Polynomial_naive_ordering."""

    def test_rank_zero_coef_list_length(self):
        """Rank-0 has exactly 1 entry."""
        pno = Polynomial_naive_ordering(0)
        self.assertEqual(len(pno.coef_list), 1)
        self.assertEqual(list(pno.coef_list[0]), [])

    def test_rank_one_coef_list_length(self):
        """Rank-1 has 5 entries."""
        pno = Polynomial_naive_ordering(1)
        self.assertEqual(len(pno.coef_list), 5)

    def test_rank_two_coef_list_length(self):
        """Rank-2 has 15 entries."""
        pno = Polynomial_naive_ordering(2)
        self.assertEqual(len(pno.coef_list), 15)

    def test_negative_rank_raises(self):
        """Negative rank must raise AssertionError."""
        with self.assertRaises(AssertionError):
            Polynomial_naive_ordering(-1)

    def test_rank_one_indices_cover_all_dimensions(self):
        """Rank-1 coefficients should include each of the 4 dimensions once."""
        pno = Polynomial_naive_ordering(1)
        rank1_coefs = [list(c) for c in pno.coef_list if len(c) == 1]
        self.assertEqual(sorted(rank1_coefs), [[0], [1], [2], [3]])

    def test_get_coef_position_roundtrip(self):
        """get_coef_position then coef_list lookup returns the original coef."""
        pno = Polynomial_naive_ordering(2)
        for i, coef in enumerate(pno.coef_list):
            pos = pno.get_coef_position(list(coef))
            self.assertEqual(pos, i)

    def test_get_coef_position_permutation_invariant(self):
        """Permuting the index list should not change the position."""
        pno = Polynomial_naive_ordering(3)
        # [0,1,2] sorted == [0,1,2] — all permutations map to same position
        pos_012 = pno.get_coef_position([0, 1, 2])
        pos_201 = pno.get_coef_position([2, 0, 1])
        pos_120 = pno.get_coef_position([1, 2, 0])
        self.assertEqual(pos_012, pos_201)
        self.assertEqual(pos_012, pos_120)

    def test_get_coef_at_position_valid(self):
        """get_coef_at_position returns a list."""
        pno = Polynomial_naive_ordering(1)
        result = pno.get_coef_at_position(0)
        self.assertIsInstance(result, list)

    def test_get_coef_position_invalid_index_raises(self):
        """Asking for an index beyond 3 should raise PolynomialError."""
        pno = Polynomial_naive_ordering(1)
        with self.assertRaises(PolynomialError):
            pno.get_coef_position([4])


class TestPolynomial(unittest.TestCase):
    """Tests for Polynomial (canonical ordering)."""

    def test_rank_zero_coef_list_length(self):
        """Rank-0 polynomial has 1 coefficient."""
        p = Polynomial(0)
        self.assertEqual(len(p.coef_list), 1)

    def test_rank_one_coef_list_length(self):
        """Rank-1 polynomial has 5 coefficients."""
        p = Polynomial(1)
        self.assertEqual(len(p.coef_list), 5)

    def test_rank_two_coef_list_length(self):
        """Rank-2 polynomial has 15 coefficients."""
        p = Polynomial(2)
        self.assertEqual(len(p.coef_list), 15)

    def test_rank_three_coef_list_length(self):
        """Rank-3 polynomial has 35 coefficients."""
        p = Polynomial(3)
        self.assertEqual(len(p.coef_list), 35)

    def test_negative_rank_raises(self):
        """Negative rank must raise AssertionError."""
        with self.assertRaises(AssertionError):
            Polynomial(-1)

    def test_scalar_coefficient_at_position_zero(self):
        """Position 0 always holds the scalar (empty index list)."""
        for rank in range(4):
            p = Polynomial(rank)
            self.assertEqual(p.get_coef_at_position(0), [])

    def test_get_coef_position_empty_list_is_zero(self):
        """Empty index list always maps to position 0."""
        for rank in range(4):
            p = Polynomial(rank)
            self.assertEqual(p.get_coef_position([]), 0)

    def test_get_coef_position_rank1_dimensions(self):
        """Rank-1 components occupy positions 1-4."""
        p = Polynomial(1)
        positions = {p.get_coef_position([d]) for d in range(4)}
        self.assertEqual(positions, {1, 2, 3, 4})

    def test_roundtrip_rank2(self):
        """get_coef_at_position(get_coef_position(c)) == sorted(c) for rank 2."""
        p = Polynomial(2)
        for i in range(len(p.coef_list)):
            coef = p.get_coef_at_position(i)
            pos = p.get_coef_position(coef)
            self.assertEqual(pos, i,
                msg="Round-trip failed at position %d: coef=%s, back_pos=%d" % (i, coef, pos))

    def test_roundtrip_rank3(self):
        """Round-trip invariant holds for rank 3."""
        p = Polynomial(3)
        for i in range(len(p.coef_list)):
            coef = p.get_coef_at_position(i)
            pos = p.get_coef_position(coef)
            self.assertEqual(pos, i)

    def test_permutation_invariance(self):
        """All permutations of the same index list map to the same position."""
        p = Polynomial(3)
        # [0,1,2] — 6 permutations
        permutations = [
            [0, 1, 2], [0, 2, 1], [1, 0, 2],
            [1, 2, 0], [2, 0, 1], [2, 1, 0],
        ]
        reference = p.get_coef_position([0, 1, 2])
        for perm in permutations:
            self.assertEqual(p.get_coef_position(perm), reference,
                msg="Permutation %s gave different position" % perm)

    def test_repeated_index_permutation(self):
        """Permutations of [0,0,1] all map to the same position."""
        p = Polynomial(3)
        pos_001 = p.get_coef_position([0, 0, 1])
        pos_010 = p.get_coef_position([0, 1, 0])
        pos_100 = p.get_coef_position([1, 0, 0])
        self.assertEqual(pos_001, pos_010)
        self.assertEqual(pos_001, pos_100)

    def test_distinct_coefs_have_distinct_positions(self):
        """All coefficients at different index tuples have unique positions."""
        p = Polynomial(2)
        positions = [p.get_coef_position(list(c)) for c in p.coef_list]
        self.assertEqual(len(positions), len(set(positions)),
            msg="Duplicate positions found: %s" % positions)

    def test_no_none_in_coef_list(self):
        """After initialization, every slot in coef_list is populated."""
        for rank in range(4):
            p = Polynomial(rank)
            for i, coef in enumerate(p.coef_list):
                self.assertIsNotNone(coef,
                    msg="coef_list[%d] is None for rank %d" % (i, rank))

    def test_coef_list_entries_are_sorted(self):
        """Every entry in coef_list (after init) should be sorted."""
        for rank in range(4):
            p = Polynomial(rank)
            for coef in p.coef_list:
                lst = list(coef)
                self.assertEqual(lst, sorted(lst),
                    msg="Unsorted coef entry %s for rank %d" % (lst, rank))

    def test_naive_and_canonical_agree_on_positions(self):
        """Both orderings agree on the total number of coefficients per rank."""
        for rank in range(4):
            p = Polynomial(rank)
            pno = Polynomial_naive_ordering(rank)
            self.assertEqual(len(p.coef_list), len(pno.coef_list))


class TestPolynomialRoutines(unittest.TestCase):
    """Basic smoke tests for PolynomialRoutines and FortranPolynomialRoutines."""

    def test_polynomial_routines_init(self):
        """PolynomialRoutines can be instantiated for standard ranks."""
        pr = q_polynomial.PolynomialRoutines(3)
        self.assertEqual(pr.max_rank, 3)
        self.assertEqual(pr.updater_max_rank, 3)

    def test_polynomial_routines_updater_rank(self):
        """updater_max_rank can be set lower than max_rank."""
        pr = q_polynomial.PolynomialRoutines(4, updater_max_rank=2)
        self.assertEqual(pr.updater_max_rank, 2)

    def test_polynomial_routines_updater_rank_too_high(self):
        """Setting updater_max_rank above max_rank raises PolynomialError."""
        with self.assertRaises(PolynomialError):
            q_polynomial.PolynomialRoutines(2, updater_max_rank=3)

    def test_polynomial_routines_negative_rank(self):
        """Negative max_rank raises PolynomialError."""
        with self.assertRaises(PolynomialError):
            q_polynomial.PolynomialRoutines(-1)

    def test_fortran_routines_constant_module(self):
        """write_polynomial_constant_module returns a non-empty Fortran string."""
        fpr = q_polynomial.FortranPolynomialRoutines(2)
        output = fpr.write_polynomial_constant_module()
        self.assertIsInstance(output, str)
        self.assertIn('POLYNOMIAL_CONSTANTS', output)
        self.assertIn('NCOEF_R', output)

    def test_fortran_routines_coef_format_double(self):
        """Default coef_format is complex*16 with correct zero strings."""
        fpr = q_polynomial.FortranPolynomialRoutines(1)
        self.assertEqual(fpr.rzero, '0.0d0')
        self.assertEqual(fpr.czero, '(0.0d0,0.0d0)')

    def test_fortran_routines_coef_format_quad(self):
        """complex*32 format produces quad-precision zero strings."""
        fpr = q_polynomial.FortranPolynomialRoutines(1, coef_format='complex*32')
        self.assertEqual(fpr.rzero, '0.0e0_16')

    def test_fortran_routines_coef_format_single(self):
        """Other coef_format falls back to single-precision zero strings."""
        fpr = q_polynomial.FortranPolynomialRoutines(1, coef_format='complex*8')
        self.assertEqual(fpr.rzero, '0.0e0')


if __name__ == '__main__':
    unittest.main()
