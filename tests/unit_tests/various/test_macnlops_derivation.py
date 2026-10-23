"""Independent stream-measure checks of the physical-history source gate.

No production rejection helper is imported: the checks assemble the source
records, one-step shower and hard records separately.  A normalized forced
competitor can have a nonperturbative survival probability on an ungated
history, which makes a common veto on all histories fail already at NLO.
"""
import math
import random
import unittest


def source_records(real, kernel, gates):
    """Return explicit real-H, real-S, MC-H, MC-S records per history."""
    return [(p*r, (1-p)*r, -p*k, p*k)
            for r, k, p in zip(real, kernel, gates)]


def resolved_streams(alpha, real, rates, gates, forced_survival,
                     sudakov_coefficients, born_corrections,
                     gate_observed_history=True):
    records = source_records([alpha*r for r in real],
                             [alpha*k for k in rates], gates)
    hard = sum(rh+mh for rh, rs, mh, ms in records)
    denominator = sum(ms for rh, rs, mh, ms in records)
    negative = max(-hard, 0.)
    shower = []
    for k, p, forced, sudakov, correction in zip(
            rates, gates, forced_survival, sudakov_coefficients,
            born_corrections):
        # Above every forced threshold, survival has an ordinary
        # perturbative expansion.  Below it, a normalized competing
        # conversion can remain with fixed probability as alpha -> 0.
        survival = math.exp(-alpha*sudakov) if p else forced
        shower.append(alpha*k*survival*(1+alpha*correction))
    signed_baseline = sum(shower)+hard
    accepted = max(hard, 0.)
    for value, p in zip(shower, gates):
        actual_gate = p if gate_observed_history else 1.
        acceptance = 1.-actual_gate*negative/denominator
        accepted += value*acceptance
    return signed_baseline, accepted


def expanded_prediction(alpha, real, kernel, gates, born_measurements,
                        real_measurements):
    """Assemble S+H from independent finite radiation bins at first order."""
    total_s, total_h, nlo = 0., 0., sum(born_measurements)
    for r_values, k_values, p_values, observable_born in zip(
            real, kernel, gates, born_measurements):
        records = source_records(r_values, k_values, p_values)
        # Unit Born densities and V=0 suffice for this finite-bin check.
        # D is read from both S records, rather than inferred from H.
        born_s = 1.+alpha*sum(rs+ms for rh, rs, mh, ms in records)
        total_s += born_s*observable_born
        for r, k, (rh, rs, mh, ms), observable_real in zip(
                r_values, k_values, records, real_measurements):
            total_s += alpha*k*(observable_real-observable_born)
            total_h += alpha*(rh+mh)*observable_real
            nlo += alpha*r*observable_real
    return total_s+total_h, nlo


class TestMacNLOPSDerivation(unittest.TestCase):
    def test_complete_hard_sign_has_a_positive_gated_envelope(self):
        rng = random.Random(710234)
        negative_examples = 0
        for unused in range(2000):
            real = [rng.uniform(0., 2.) for history in range(5)]
            kernel = [rng.uniform(0., 2.) for history in range(5)]
            gates = [0., 1., rng.randrange(2), 0., 1.]
            records = source_records(real, kernel, gates)
            # Each radiation bin's transfer is exactly paired: H+S=R.
            for expected, record in zip(real, records):
                self.assertAlmostEqual(sum(record), expected, places=14)
            hard = sum(rh+mh for rh, rs, mh, ms in records)
            denominator = sum(ms for rh, rs, mh, ms in records)
            negative = max(-hard, 0.)
            negative_examples += negative > 0.
            self.assertLessEqual(negative, denominator+1.e-14)
            for gate in gates:
                acceptance = 1.-gate*negative/denominator
                self.assertGreaterEqual(acceptance, -1.e-14)
                self.assertLessEqual(acceptance, 1.)
            # At leading order the observed-history veto removes Hminus.
            removed = sum(k*p*negative/denominator
                          for k, p in zip(kernel, gates))
            self.assertAlmostEqual(removed, negative, places=14)
        self.assertGreater(negative_examples, 100)

    def test_observed_history_gate_avoids_a_real_order_competition_error(self):
        arguments = dict(real=(.9, .1, .2), rates=(1.4, .5, .4),
                         gates=(0., 1., 1.),
                         forced_survival=(.23, 1., 1.),
                         sudakov_coefficients=(.7, .8, .4),
                         born_corrections=(.1, -.3, .2))
        correct_errors, common_errors = [], []
        for alpha in (.004, .002, .001):
            baseline, accepted = resolved_streams(alpha, **arguments)
            unused, common = resolved_streams(
                alpha, **arguments, gate_observed_history=False)
            correct_errors.append(accepted-baseline)
            common_errors.append(common-baseline)
        # Removing negative H adds O(alpha); only the gated shower
        # removal cancels it despite the ungated forced competitor.
        for index in range(2):
            self.assertAlmostEqual(correct_errors[index]/
                                   correct_errors[index+1], 4., delta=.006)
            self.assertAlmostEqual(common_errors[index]/
                                   common_errors[index+1], 2., delta=.015)
        self.assertGreater(abs(common_errors[-1]),
                           100*abs(correct_errors[-1]))

    def test_cutoff_residual_and_subthreshold_jet_veto(self):
        # Two overlapping inverse Born histories.  The middle bin lies
        # below the second history's physical matching scale only.
        real = ((.6, .5, .3), (.2, .4, .7))
        kernel = ((.9, .4, .4), (.5, .8, .6))
        gates = ((0., 1., 1.), (0., 0., 1.))
        alpha = .03
        # Above all matching scales, an observable zero on the Born
        # and changed real region has the exact NLO coefficient.
        prediction, nlo = expanded_prediction(
            alpha, real, kernel, gates, (0., 0.), (0., 0., 1.))
        self.assertAlmostEqual(prediction, nlo, places=14)
        # A veto between the first and second radiation bins is infrared
        # safe, but it resolves part of the altered region.  Its mismatch
        # is -alpha*(K-R) from history two's middle bin, at NLO itself.
        prediction, nlo = expanded_prediction(
            alpha, real, kernel, gates, (1., 1.), (1., 0., 0.))
        expected = -alpha*(kernel[1][1]-real[1][1])
        self.assertNotEqual(expected, 0.)
        self.assertAlmostEqual(prediction-nlo, expected, places=14)
        half_prediction, half_nlo = expanded_prediction(
            alpha/2, real, kernel, gates, (1., 1.), (1., 0., 0.))
        self.assertAlmostEqual((prediction-nlo)/
                               (half_prediction-half_nlo), 2., places=12)


if __name__ == '__main__':
    unittest.main()
