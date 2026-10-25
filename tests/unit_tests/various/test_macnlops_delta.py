"""Finite-measure checks of conditional MC@NLO-Delta postprocessing.

These tests use exact rational weights, not showers or sampled events.  They
assemble source H records and signed first-S records independently, including
source auxiliary variables and owner mixtures.  They do not test numerical
Sudakov tables or assert that a source Delta equals the shower Sudakov.
"""

from fractions import Fraction as F
import unittest


def conditional_h(real, subtraction, delta):
    return sum(p * (r - d) for r, d, p in zip(real, subtraction, delta))


def bounded_acceptances(hard, kernel):
    negative = max(-hard, F(0))
    cancelled = min(negative, kernel)
    keep_h = F(1) - cancelled / negative if negative else F(1)
    keep_s = F(1) - cancelled / kernel if kernel else F(1)
    return keep_h, keep_s


class TestMacNLOPSDeltaMeasure(unittest.TestCase):
    real = (F(2), F(6))
    subtraction = (F(6), F(1))
    delta_states = ((F(1), F(1, 5)), (F(1, 4), F(1)))

    def test_delta_keeps_the_compensating_s_source_term(self):
        # Independent real-H, MC-H, real-S and MC-S source records.
        for delta in self.delta_states:
            hard = conditional_h(self.real, self.subtraction, delta)
            soft = sum((1-p)*r+p*d for r, d, p in
                       zip(self.real, self.subtraction, delta))
            self.assertEqual(hard + soft, F(8))
            self.assertNotEqual(hard + sum(self.subtraction), F(8))

    def test_negative_part_is_taken_before_auxiliary_marginalization(self):
        hard = [conditional_h(self.real, self.subtraction, delta)
                for delta in self.delta_states]
        self.assertEqual(hard, [F(-3), F(4)])
        mean_h = sum(hard) / 2
        actual_negative_records = sum(max(-h, F(0)) for h in hard) / 2
        self.assertEqual(mean_h, F(1, 2))
        self.assertEqual(max(-mean_h, F(0)), 0)
        self.assertEqual(actual_negative_records, F(3, 2))

    def test_original_veto_cancels_a_common_physical_sudakov(self):
        # The ideal Nason--Salam kernel is also the hard subtraction.
        # Delta_PS = Delta_source and Bbar/B = 1 here, not in general.
        real, kernel, delta = F(3, 5), F(1), F(1, 5)
        hard = delta * (real - kernel)
        first_s = delta * kernel
        gain = -hard
        current_veto = 1 - bounded_acceptances(hard, kernel)[1]
        original_veto = (kernel - real) / kernel
        self.assertEqual(current_veto, F(2, 25))
        self.assertEqual(original_veto, F(2, 5))
        self.assertEqual(gain - first_s * original_veto, 0)
        self.assertEqual(gain - first_s * current_veto, F(8, 125))

    def test_undamped_bounded_veto_also_needs_a_different_h_residual(self):
        # G completion can make the undamped negative H larger than K.
        h0, kernel, delta = F(2), F(1), F(1, 4)
        source_h = -delta * h0
        first_s = delta * kernel
        veto = min(h0 / kernel, F(1))
        correct_keep_h = max(1 - kernel / h0, F(0))
        current_keep_h = bounded_acceptances(source_h, kernel)[0]
        self.assertEqual(-source_h * (1 - correct_keep_h) - first_s * veto, 0)
        self.assertEqual(-source_h * (1 - current_keep_h) - first_s * veto, F(1, 4))

    def test_different_history_sudakovs_obstruct_global_r_over_k(self):
        kernel = sum(self.subtraction)
        real = sum(self.real)
        hard = conditional_h(self.real, self.subtraction, self.delta_states[0])
        self.assertEqual((real, kernel, hard), (F(8), F(7), F(-3)))
        self.assertEqual(max(1 - real / kernel, F(0)), 0)
        self.assertEqual(max(-hard, F(0)), 3)

    def test_replacing_completed_subtraction_by_kernel_can_change_nlo(self):
        # Independent source and shower densities at leading order. The
        # coefficient of epsilon is nonzero even with Delta=1; it is not
        # a second-order difference attributable to Sudakov expansion.
        for epsilon in (F(1), F(1, 10), F(1, 100)):
            real, completed_d, kernel = (epsilon * value
                                        for value in (F(3, 5), F(4, 5), F(1)))
            hard = real - completed_d
            keep_h, keep_s = bounded_acceptances(hard, kernel)
            gain = -hard * (1 - keep_h)
            original_veto = max(1 - real / kernel, F(0))
            self.assertEqual(gain - kernel * (1 - keep_s), 0)
            self.assertEqual((gain - kernel * original_veto) / epsilon, F(-1, 5))

    def test_independent_auxiliary_thinning_closes_each_leading_bin(self):
        # q bins have (K_g, positive eligible S, negative eligible S,
        # signed protected S).  Protected S is never vetoed.
        bins = ((F(2), F(3), F(-1), F(5)),
                (F(4), F(6), F(-2), F(-1)),
                (F(0), F(0), F(0), F(4)))
        hard = [conditional_h(self.real, self.subtraction, delta)
                for delta in self.delta_states]
        for kernel, positive_s, negative_s, protected_s in bins:
            self.assertEqual(positive_s + negative_s, kernel)
            kept_h = sum(h * bounded_acceptances(h, kernel)[0]
                         for h in hard) / 2
            # A NEW source auxiliary is independently integrated for S.
            s_acceptance = sum(bounded_acceptances(h, kernel)[1]
                               for h in hard) / 2
            retained = (kept_h + (positive_s + negative_s)*s_acceptance
                        + protected_s)
            baseline = sum(hard)/2 + positive_s + negative_s + protected_s
            self.assertEqual(retained, baseline)
            if kernel == 0:
                self.assertEqual(kept_h, sum(hard)/2)
            elif kernel == 2:
                # The negative source record has magnitude3 but only2
                # can be cancelled in that auxiliary stratum.
                self.assertEqual(kept_h, F(3, 2))

    def test_cap_must_precede_auxiliary_average(self):
        negatives = (F(0), F(4))
        kernel = F(1)
        conditional_cancellation = sum(min(h, kernel) for h in negatives)/2
        incorrectly_capped_mean = min(sum(negatives)/2, kernel)
        self.assertEqual(conditional_cancellation, F(1, 2))
        self.assertEqual(incorrectly_capped_mean, F(1))

    def test_actual_supporting_start_is_not_a_new_source_auxiliary(self):
        # Source H uses equally likely low/high auxiliary draws.  The
        # actual S endpoint can only have come from its high start, so
        # reusing that observed draw gives a different conditional law.
        source_h = (F(4), F(-3))
        kernel = F(2)
        h_gain = sum(-h*(1-bounded_acceptances(h, kernel)[0])
                     for h in source_h if h < 0)/2
        independent_veto = sum(1-bounded_acceptances(h, kernel)[1]
                               for h in source_h)/2
        reused_supporting_veto = 1-bounded_acceptances(source_h[1], kernel)[1]
        self.assertEqual(h_gain, F(1))
        self.assertEqual(kernel*independent_veto, h_gain)
        self.assertEqual(kernel*reused_supporting_veto, F(2))

    def test_source_owner_mixture_cannot_be_replaced_by_one_owner(self):
        # The two owners have different auxiliary laws at the same q.
        # Outer source partitions are1/4 and3/4; each owner has H=-3/+4.
        owners = ((F(1, 4), (F(1, 2), F(1, 2))),
                  (F(3, 4), (F(1, 10), F(9, 10))))
        hard, kernel = (F(-3), F(4)), F(2)
        gain = sum(owner_weight*sum(prob*(-h)*(1-bounded_acceptances(h, kernel)[0])
                                   for prob, h in zip(law, hard) if h < 0)
                   for owner_weight, law in owners)
        veto = sum(owner_weight*sum(prob*(1-bounded_acceptances(h, kernel)[1])
                                   for prob, h in zip(law, hard))
                   for owner_weight, law in owners)
        one_owner_veto = sum(prob*(1-bounded_acceptances(h, kernel)[1])
                             for prob, h in zip(owners[0][1], hard))
        self.assertEqual(gain, F(2, 5))
        self.assertEqual(kernel*veto, gain)
        self.assertEqual(kernel*one_owner_veto, F(1))

    def test_first_point_closure_does_not_identify_full_continuations(self):
        cancelled, kernel = F(1), F(2)
        first_s_density = kernel
        self.assertEqual(cancelled-first_s_density*cancelled/kernel, 0)
        h_continuation, s_continuation = F(3, 4), F(1, 4)
        full_shift = (cancelled*h_continuation
                      - first_s_density*cancelled/kernel*s_continuation)
        self.assertEqual(full_shift, F(1, 2))


if __name__ == '__main__':
    unittest.main()
