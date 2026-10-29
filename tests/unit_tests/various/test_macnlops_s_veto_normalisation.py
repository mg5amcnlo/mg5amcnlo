"""S-veto normalisation beta*K_sh: channel integrals, couplings and decisions."""
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from madgraph.various import macnlops as mc
from tests.unit_tests.various.test_macnlops import HEADER, TableEvaluator, event


class TestScaledAcceptance(unittest.TestCase):
    def test_only_the_s_denominator_is_rescaled(self):
        density = mc.Densities(0., 1., 4.)
        self.assertEqual(density.acceptance(), .75)
        self.assertAlmostEqual(density.acceptance(kernel_scale=2.), 1. - 1. / 8.)
        # A small denominator saturates the veto instead of exceeding one.
        self.assertEqual(density.acceptance(kernel_scale=.2), 0.)
        # The H removal of min(u,K) does not depend on the S normalisation.
        self.assertEqual(density.hard_acceptance(), 0.)
        self.assertEqual(mc.Densities(0., 2., 1.).hard_acceptance(signed_residuals=True), .5)
        for scale in (0., -1., math.nan, math.inf):
            with self.subTest(scale=scale), self.assertRaises(mc.MatchingError):
                density.acceptance(kernel_scale=scale)

    def test_transfers_needs_negative_hard_density_and_shower_support(self):
        self.assertTrue(mc.Densities(0., 1., 4.).transfers())
        self.assertFalse(mc.Densities(1., 1., 4.).transfers())
        self.assertFalse(mc.Densities(0., 1., 0.).transfers())
        self.assertFalse(mc.Densities(0., 1., -1.).transfers())

    def test_kernel_scale_uses_both_couplings_and_beta(self):
        density = mc.Densities(0., 1., 4., kernel_alpha_s=.1)
        self.assertAlmostEqual(mc.shower_kernel_scale(1.25, dict(alpha_s_emission=.3), density),
                               3.75)
        with self.assertRaisesRegex(mc.MatchingError, 'couplings'):
            mc.shower_kernel_scale(1.25, {}, density)
        with self.assertRaisesRegex(mc.MatchingError, 'couplings'):
            mc.shower_kernel_scale(1.25, dict(alpha_s_emission=.3), mc.Densities(0., 1., 4.))
        for beta, alpha in ((0., .3), (1., -.3), (math.nan, .3), (1., True)):
            with self.subTest(beta=beta, alpha=alpha), \
                    self.assertRaisesRegex(mc.MatchingError, 'normalisation'):
                mc.shower_kernel_scale(beta, dict(alpha_s_emission=alpha), density)


class TestChannelNormalisation(unittest.TestCase):
    RES_1 = (' Final result [ABS]:   459.64900652255659       +/-  0.88463798894517753\n'
             ' Final result:   418.52834722694541       +/-   1.0261713952620646\n'
             ' Final result [S]:   360.00000000000000       +/-   1.0000000000000000\n'
             ' Final result [Born]:   300.00000000000000       +/-  0.50000000000000000\n')

    def test_split_job_reads_its_channel_through_res_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            channel, job = Path(tmp) / 'GF2.0', Path(tmp) / 'GF2.0_3'
            channel.mkdir(), job.mkdir()
            (channel / 'res_1').write_text(self.RES_1)
            (job / 'res_1').symlink_to('../GF2.0/res_1')
            result = mc.channel_normalisation(job / 'events.lhe')
        self.assertEqual((result['sigma_S'], result['sigma_B']), (360., 300.))
        self.assertEqual((result['sigma_S_error'], result['sigma_B_error']), (1., .5))
        self.assertAlmostEqual(result['beta'], 1.2)

    def test_missing_or_nonpositive_channel_integrals_are_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with self.assertRaisesRegex(mc.MatchingError, 'Cannot read'):
                mc.channel_normalisation(folder / 'events.lhe')
            (folder / 'res_1').write_text(''.join(self.RES_1.splitlines(True)[:2]))
            with self.assertRaisesRegex(mc.MatchingError, 'lacks the channel S and Born'):
                mc.channel_normalisation(folder / 'events.lhe')
            (folder / 'res_1').write_text(self.RES_1.replace('300.000', '-300.000'))
            with self.assertRaisesRegex(mc.MatchingError, 'Nonpositive'):
                mc.channel_normalisation(folder / 'events.lhe')


class FakeWorker:
    def __init__(self, lines):
        self.stdin, self.stdout = io.StringIO(), iter(lines)


class TestKernelCouplingRecord(unittest.TestCase):
    def query(self, lines, request_id=7):
        evaluator = object.__new__(mc.NativeEvaluator)
        directory = Path('P0_uux_emep')
        evaluator.workers = {directory: (FakeWorker(lines), io.StringIO())}
        return evaluator._query(directory, 'REQUEST\n', request_id)

    def test_coupling_precedes_and_annotates_its_density(self):
        density = self.query(['native log line\n',
                              'MACNLOPS_KERNEL_COUPLING_V1 7 1.25e-01\n',
                              'MACNLOPS_DENSITY_V1 7 0 1.0 2.0 4.0\n'])
        self.assertEqual(density, mc.Densities(1., 2., 4., kernel_alpha_s=.125))
        # Evaluators predating the record still return their densities.
        self.assertIsNone(self.query(['MACNLOPS_DENSITY_V1 7 0 1.0 2.0 4.0\n']).kernel_alpha_s)
        with self.assertRaisesRegex(mc.MatchingError, 'kernel coupling'):
            self.query(['MACNLOPS_KERNEL_COUPLING_V1 8 1.25e-01\n',
                        'MACNLOPS_DENSITY_V1 7 0 1.0 2.0 4.0\n'])


class TestNormalisedVetoDecisions(unittest.TestCase):
    def run_veto(self, folder, densities, beta=1.25):
        blocks = [event('S', 2, 1), event('S', 2, 2), event('H', -2, 3)]
        records = [dict(emitted=True, alpha_s_emission=.3), dict(emitted=True, alpha_s_emission=.3),
                   dict(emitted=False, alpha_s_emission=0.)]
        proposals = folder / 'proposals.lhe'
        proposals.write_text(HEADER + ''.join(blocks) + '</LesHouchesEvents>\n')
        metadata = folder / 'shower.jsonl'
        metadata.write_text(''.join(json.dumps(dict(record, sequence=i, event_kind=kind)) + '\n'
                                    for i, (kind, record) in enumerate(zip('SSH', records), 1)))
        output = folder / 'accepted.lhe.gz'
        report = mc.reject_proposals(proposals, metadata, output, TableEvaluator(densities),
                                     matching_policy=mc.UNDAMPED_DELTA_POLICY, beta=beta)
        decisions = [json.loads(line) for line in
                     output.with_name(output.name + '.decisions.jsonl').read_text().splitlines()]
        return report, decisions

    def test_s_veto_uses_beta_and_the_shower_coupling(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, decisions = self.run_veto(Path(tmp), {
                1: mc.Densities(0., 1., 4., kernel_alpha_s=.1),
                # No transfer: neither coupling is needed.
                2: mc.Densities(1., .5, 4.),
                3: mc.Densities(0., 1., 4., kernel_alpha_s=.1)})
        self.assertEqual(report['s_veto_beta'], 1.25)
        self.assertEqual(report['veto_prescription'], 'component-undamped-delta-v2')
        self.assertAlmostEqual(decisions[0]['kernel_scale'], 3.75)
        self.assertAlmostEqual(decisions[0]['acceptance'], 1. - .25 / 3.75)
        self.assertEqual(decisions[0]['densities']['kernel_alpha_s'], .1)
        self.assertIsNone(decisions[1]['kernel_scale'])
        self.assertEqual(decisions[1]['acceptance'], 1.)
        self.assertIsNone(decisions[2]['kernel_scale'])
        self.assertEqual(decisions[2]['acceptance'], 0.)

    def test_transferring_s_event_needs_the_kernel_coupling(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(mc.MatchingError, 'couplings'):
                self.run_veto(Path(tmp), {1: mc.Densities(0., 1., 4.),
                                          2: mc.Densities(1., .5, 4.),
                                          3: mc.Densities(0., 1., 4.)})
            self.assertFalse((Path(tmp) / 'accepted.lhe.gz').exists())

    def test_without_beta_the_veto_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, decisions = self.run_veto(Path(tmp), {
                1: mc.Densities(0., 1., 4.), 2: mc.Densities(1., .5, 4.),
                3: mc.Densities(0., 1., 4.)}, beta=None)
        self.assertIsNone(report['s_veto_beta'])
        self.assertEqual(decisions[0]['acceptance'], .75)
        self.assertIsNone(decisions[0]['kernel_scale'])


if __name__ == '__main__':
    unittest.main()
