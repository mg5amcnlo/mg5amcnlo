"""Delta source identity and stochastic replay; no builds or showers."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from madgraph.various import macnlops as mc
from tests.unit_tests.various.test_macnlops import HEADER, event, hard_annotation


REGISTRY = 'a' * 64


def packet(registry=REGISTRY, owner=1, rows='1 1 1 0.25\n1 1 2 0.75'):
    return ('<macnlops_delta_context version="1" owner="%d" insertion="0.3" '
            'registry="%s">\n%s\n</macnlops_delta_context>\n' % (owner, registry, rows))


def delta_event(kind='H', registry=REGISTRY, context=True):
    annotation = hard_annotation(kind).replace('source="ordinary"', 'source="delta"')
    return event(kind, weight=-2).replace('</event>', annotation +
        (packet(registry) if context else '') + '</event>')


class TestDeltaProtocol(unittest.TestCase):
    def evaluator(self, root):
        (root / 'Cards').mkdir()
        (root / 'Cards/run_card.dat').write_text('True = mcatnlo_delta\n')
        hashes = []
        for index in (1, 2):
            directory = root / 'SubProcesses' / ('P%d' % index)
            directory.mkdir(parents=True)
            # The extra alias is deliberately first: it must not become the
            # source owner merely because it has the same external flavours.
            (directory / 'leshouche_info.dat').write_text(
                'I 7 1 2 -2 11 -11\nI 1 1 2 -2 11 -11\n')
            process = ['same amplitude restrictions', [
                [pdg, leg >= 2, [], False, None] for leg, pdg in enumerate((2, -2, 11, -11))]]
            registry = dict(outer=dict(context=index, orders=['QCD'], amp_orders=[[2]],
                sectors=[dict(sector=1, processes=[process])]),
                tables=[dict(sector=1, histories=[dict(context=3, sector=2,
                    permutation=[1, 2, 3, 4])])], contexts=[{'context': 3, 'native': 'shared'}],
                flavour_complete=True)
            fingerprint = hashlib.sha256(json.dumps(registry, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest()
            (directory / 'mc_histories.json').write_text(json.dumps(dict(
                schema='macnlops-history-registry-v1', sha256=fingerprint, registry=registry)))
            hashes.append(fingerprint)
        with patch.object(mc, 'hard_model_masses', return_value={}):
            return mc.NativeEvaluator(root, root, auxiliary_seed=83), hashes

    def test_construction_keeps_normal_delta_and_legacy_sources_distinct(self):
        self.assertEqual(mc.source_construction({'mcatnlo_delta': '.true.'}), mc.DELTA_SOURCE)
        self.assertEqual(mc.source_construction({'mcatnlo_delta': False}), mc.ORDINARY_SOURCE)
        with self.assertRaisesRegex(mc.MatchingError, 'resolved_source=False'):
            mc.source_construction(dict(mcatnlo_delta=True, macnlops_resolved_source=True))
        with self.assertRaisesRegex(mc.MatchingError, 'Invalid mcatnlo_delta'):
            mc.source_construction({'mcatnlo_delta': 'typo'})

    def test_run_card_accepts_delta_and_rejects_the_legacy_combination(self):
        from madgraph.various.banner import InvalidRunCard, RunCardNLO
        card = RunCardNLO()
        for key, value in dict(macnlops=True, mcatnlo_delta=True, parton_shower='PYTHIA8',
                               event_norm='sum', pdlabel='lhapdf', lhaid=[260000],
                               reweight_scale=[False], reweight_pdf=[False]).items():
            card.set(key, value, user=True)
        card.check_validity()
        self.assertEqual(mc.source_construction(card), mc.DELTA_SOURCE)
        card.set('macnlops_resolved_source', True, user=True)
        with self.assertRaisesRegex(InvalidRunCard, 'resolved_source=False'):
            card.check_validity()

    def test_run_card_requires_hard_momenta_only_for_macnlops(self):
        from madgraph.various.banner import InvalidRunCard, RunCardNLO
        for delta in (False, True):
            with self.subTest(delta=delta):
                card = RunCardNLO()
                for key, value in dict(macnlops=True, mcatnlo_delta=delta,
                        store_hard_momenta=False, parton_shower='PYTHIA8',
                        event_norm='sum', pdlabel='lhapdf', lhaid=[260000],
                        reweight_scale=[False], reweight_pdf=[False]).items():
                    card.set(key, value, user=True)
                with self.assertRaisesRegex(InvalidRunCard, 'store_hard_momenta=True'):
                    card.check_validity()
                self.assertFalse(card['store_hard_momenta'])
                card.set('store_hard_momenta', True, user=True)
                card.check_validity()
                self.assertEqual(mc.source_construction(card),
                                 mc.DELTA_SOURCE if delta else mc.ORDINARY_SOURCE)
                card.set('macnlops', False, user=True)
                card.set('store_hard_momenta', False, user=True)
                card.check_validity()

    def test_context_round_trip_and_fail_closed_validation(self):
        parsed = mc.Event(delta_event())
        self.assertEqual(parsed.delta.owner, 1)
        self.assertEqual(parsed.delta.probabilities, ((1, 1, 1, .25), (1, 1, 2, .75)))
        invalid = [packet(rows='1 1 1 1.001'), packet(rows='1 1 1 nan'),
                   packet(rows='1 1 1 .2\n1 1 1 .3'), packet(rows='0 1 1 .2'),
                   packet(registry='abc'), packet().replace('0.3', 'nan'),
                   packet().replace('version="1"', 'version="2"'),
                   packet().replace('owner="1"', 'owner="2"'),
                   packet() + packet(), packet().replace('</macnlops_delta_context>', '')]
        base = delta_event(context=False)
        for bad in invalid:
            with self.subTest(packet=bad), self.assertRaises(mc.MatchingError):
                mc.Event(base.replace('</event>', bad + '</event>'))
        with self.assertRaisesRegex(mc.MatchingError, 'Delta hard-event'):
            mc.Event(delta_event().replace('source="delta"', 'source="ordinary"'))

    def test_original_h_requires_context_but_original_s_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.lhe'
            path.write_text(HEADER + delta_event() + delta_event('S', context=False) +
                            '</LesHouchesEvents>\n')
            result = mc.hard_source_contract(path, mc.DELTA_SOURCE)
            self.assertEqual(result['counts'], {'H': 1, 'S': 1})
            path.write_text(HEADER + delta_event(context=False) + '</LesHouchesEvents>\n')
            with self.assertRaisesRegex(mc.MatchingError, 'conditional Delta replay'):
                mc.hard_source_contract(path, mc.DELTA_SOURCE)

    def test_h_replay_routes_to_exact_registry_and_keeps_the_full_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluator, fingerprints = self.evaluator(Path(tmp))
            directory, request = evaluator._request(mc.Event(delta_event(registry=fingerprints[1])), 7)
            self.assertEqual(directory.name, 'P2')
            self.assertTrue(request.startswith('MACNLOPS_POINT_V2 7 1 4\n'))
            self.assertIn('DELTA_REPLAY 2 2.99999999999999989e-01 ' + fingerprints[1], request)
            self.assertIn('1 1 2 7.50000000000000000e-01\nEND_POINT\n', request)
            with self.assertRaisesRegex(mc.MatchingError, 'exact Delta replay registry'):
                evaluator._request(mc.Event(delta_event()), 8)
            with self.assertRaisesRegex(mc.MatchingError, 'conditional Delta replay'):
                evaluator._request(mc.Event(delta_event(context=False)), 8)

    def test_s_samples_fresh_reproducible_auxiliary_independently_of_source_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluator, _ = self.evaluator(Path(tmp))
            source = delta_event('S', context=False)
            parsed = mc.Event(source)
            directory, request = evaluator._request(parsed, 9)
            self.assertEqual(directory.name, 'P1')
            self.assertIn('MACNLOPS_POINT_V2 9 1 4\n', request)
            self.assertIn('DELTA_SAMPLE ', request)
            self.assertNotIn('DELTA_REPLAY', request)
            changed = mc.Event(source.replace('100 0.007297', '400 0.007297').replace(
                '</event>', '<scales scalup_1_2="400"/>\n</event>'))
            self.assertEqual(evaluator._request(changed, 9)[1], request)
            self.assertEqual(evaluator._request(parsed, 9)[1], request)
            self.assertEqual(evaluator._request(mc.Event(delta_event('S')), 9)[1], request)
            self.assertNotEqual(evaluator._request(parsed, 10)[1], request)
            evaluator.auxiliary_seed += 1
            self.assertNotEqual(evaluator._request(parsed, 9)[1], request)

    def test_changed_registry_content_is_rejected_before_a_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.evaluator(root)
            path = root / 'SubProcesses/P1/mc_histories.json'
            record = json.loads(path.read_text())
            record['registry']['tables'][0]['sector'] = 2
            path.write_text(json.dumps(record))
            with patch.object(mc, 'hard_model_masses', return_value={}), \
                    self.assertRaisesRegex(mc.MatchingError, 'Invalid Delta history registry'):
                mc.NativeEvaluator(root, root)

    def test_different_joint_owner_laws_require_an_explicit_source_mixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluator, _ = self.evaluator(Path(tmp))
            table = evaluator.registries[Path(tmp) / 'SubProcesses/P2']['registry']['tables'][0]
            table['histories'].append(dict(context=3, sector=3, permutation=[1, 2, 3, 4]))
            with self.assertRaisesRegex(mc.MatchingError, 'joint auxiliary laws'):
                evaluator._request(mc.Event(delta_event('S', context=False)), 9)

    def test_owner_certificate_composes_the_full_external_permutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluator, _ = self.evaluator(Path(tmp))
            directory = Path(tmp) / 'SubProcesses/P2'
            registry = evaluator.registries[directory]['registry']
            process = registry['outer']['sectors'][0]['processes'][0]
            process[1][2:] = process[1][2:][::-1]
            evaluator.routes = [(path, owner, (2, -2, -11, 11) if path == directory else labels)
                                for path, owner, labels in evaluator.routes]
            history = registry['tables'][0]['histories'][0]
            history['permutation'] = [1, 2, 4, 3]
            evaluator._request(mc.Event(delta_event('S', context=False)), 9)
            self.assertEqual(len(evaluator.delta_law_certificates), 1)
            evaluator.delta_law_certificates.clear()
            history['permutation'] = [1, 2, 3, 4]
            with self.assertRaisesRegex(mc.MatchingError, 'joint auxiliary laws'):
                evaluator._request(mc.Event(delta_event('S', context=False)), 9)

    def test_equal_histogram_means_do_not_justify_duplicate_independent_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluator, _ = self.evaluator(Path(tmp))
            table = evaluator.registries[Path(tmp) / 'SubProcesses/P1']['registry']['tables'][0]
            table['histories'] *= 2
            with self.assertRaisesRegex(mc.MatchingError, 'duplicate native draws'):
                evaluator._request(mc.Event(delta_event('S', context=False)), 9)

    def test_delta_probe_passes_the_native_scale_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'probe.json'
            def probe(command, **kwargs):
                self.assertIn('--mcatnlo-delta', command)
                output.write_text(json.dumps({'schema': 'macnlops-shower-config-v1'}))
            with patch.object(mc.subprocess, 'run', side_effect=probe):
                mc.probe_shower(root / 'driver', output, [2212, 2212], [6500, 6500],
                    'LHAPDF6:set/0', .118, 2, root / 'settings', {}, root / 'probe.log',
                    mcatnlo_delta=True)

    def test_randomized_collection_preserves_the_signed_delta_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'channel.lhe', root / 'events.lhe.gz'
            blocks = [delta_event(), delta_event('S', context=False)]
            source.write_text(HEADER + ''.join(blocks) + '</LesHouchesEvents>\n')
            report = dict(seed=12, counts=dict(input_H=1, input_S=1,
                                               accepted_H=1, accepted_S=1))
            mc.collect_matched_events([source], [report], output, 31)
            with mc.LHEStream(output) as stream:
                retained = list(stream)
            self.assertCountEqual(retained, blocks)
            self.assertEqual([mc.Event(block).delta for block in retained
                              if mc.Event(block).kind == 'H'], [mc.Event(blocks[0]).delta])


if __name__ == '__main__':
    unittest.main()
