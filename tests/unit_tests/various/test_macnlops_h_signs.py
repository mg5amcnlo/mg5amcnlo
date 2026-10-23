"""Source signs must refer to complete H, before any native shower runs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from madgraph.various.macnlops import Densities, MatchingError
from tests.unit_tests.various.test_macnlops import HEADER, event

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('source_h_sign_audit',
    ROOT / 'docs/audits/check_macnlops_h_signs.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
SUPPORT = dict(beam_xmax=[.99, .99], shower_cutoffs=[.5, .6])


class Evaluator:
    def __init__(self, table):
        self.table, self.requests = table, []

    def configure(self, *configuration):
        self.configuration = configuration

    def evaluate(self, point, sequence):
        self.requests.append((sequence, point.weight))
        return self.table[sequence]


class TestSourceHSigns(unittest.TestCase):
    def run_audit(self, folder, blocks, evaluator, **options):
        source = folder / 'source.lhe'
        original = HEADER + ''.join(blocks) + '</LesHouchesEvents>\n'
        source.write_text(original)
        report = AUDIT.audit_source(source, evaluator, SUPPORT,
                                    folder / 'audit', **options)
        self.assertEqual(source.read_text(), original)
        return report

    def test_checks_both_H_signs_and_bound_without_selecting_or_rescaling(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            evaluator = Evaluator({2: Densities(2, 1, 1),
                                   3: Densities(1, 2, 1)})
            report = self.run_audit(folder,
                [event('S', -2), event('H', 3), event('H', -4),
                 event('H', 0), event('S', 5)], evaluator, expected_events=5)
            self.assertEqual(report['status'], 'passed')
            self.assertTrue(report['complete_source'])
            self.assertEqual(evaluator.requests, [(2, 3), (3, -4)])
            self.assertEqual(report['original_weight_sums']['source'], 2)
            self.assertEqual(report['original_weight_sums']['H'], -1)
            self.assertEqual(report['counts']['zero_weight_H'], 1)
            self.assertEqual(report['maxima']['negative_H_over_K'], 1)

    def test_rejects_observed_partial_H_group_sign_after_complete_sum(self):
        # Actual TT event233: one local group negative, complete physical H positive.
        density = Densities(305343345.1382738, 287954881.2659519,
                            287954881.2659519)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            block = event('H', -.013833534, 233)
            report = self.run_audit(folder, [event('S'), block],
                                    Evaluator({2: density}))
            self.assertEqual(report['status'], 'failed')
            failure = json.loads((folder / 'audit/first_failure.json').read_text())
            self.assertEqual(failure['sequence'], 2)
            self.assertEqual(failure['source_block'], block)
            self.assertEqual(failure['densities']['real'], density.real)
            self.assertGreater(failure['global_H'], 17.e6)
            self.assertIn('source_H_sign_disagrees_with_global_R_minus_D', failure['errors'])
            self.assertIn(block, (folder / 'audit/first_failure.lhe').read_text())
            self.assertIn('-2.075030100000e-02', failure['source_block'])

    def test_positive_H_wrong_sign_zero_density_and_negative_bound_fail_closed(self):
        cases = [(2, Densities(1, 2, 1), 'source_H_sign_disagrees'),
                 (-2, Densities(0, 1.5, 1), 'negative_H_exceeds_shower_bound'),
                 (2, Densities(1, 1, 1), 'zero_global_H')]
        for weight, density, reason in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temporary:
                report = self.run_audit(Path(temporary), [event('H', weight)],
                                        Evaluator({1: density}))
                self.assertEqual(report['status'], 'failed')
                self.assertTrue(any(reason in item for item in report['first_failure']['errors']))

    def test_diagnostic_prefix_cannot_pass_a_full_source_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self.run_audit(Path(temporary), [event('S'), event('H')],
                Evaluator({}), max_events=1)
            self.assertEqual(report['status'], 'incomplete')
            self.assertFalse(report['complete_source'])

    def test_evaluator_failure_saves_original_point_and_cannot_certify_rest(self):
        class FailedEvaluator(Evaluator):
            def evaluate(self, point, sequence):
                raise MatchingError('Unavailable real-point map')
        with tempfile.TemporaryDirectory() as temporary:
            block = event('H', -2)
            report = self.run_audit(Path(temporary), [block, event('H', 3)],
                                    FailedEvaluator({}))
            self.assertEqual(report['status'], 'failed')
            self.assertFalse(report['complete_source'])
            self.assertEqual(report['counts']['source_events'], 1)
            self.assertEqual(report['first_failure']['source_block'], block)
            self.assertEqual(report['first_failure']['errors'], ['native_evaluator_error'])


if __name__ == '__main__':
    unittest.main()
