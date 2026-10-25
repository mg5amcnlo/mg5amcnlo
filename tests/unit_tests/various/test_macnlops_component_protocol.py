"""Sampled fixed-flow source identity and the Delta-free local veto path.

Uses saved branching records and native protocol boundaries, with no event
generation. The separate compiled tests check the physical density formula.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from madgraph.various import macnlops as mc
from tests.unit_tests.various.test_macnlops import HEADER, event, hard_annotation
from tests.unit_tests.various.test_macnlops_components import PREFIXES


def packet(registry='a' * 64, sector=1, flow=1, source_factor=.2, insertion=.3):
    return ('<macnlops_component version="1" registry="%s">\n'
            '%d 2 7 %d\n.5 %.17g %.17g 3\n</macnlops_component>\n' %
            (registry, sector, flow, insertion, source_factor))


def annotated(kind='H', source='delta', registry='a' * 64):
    rows = hard_annotation(kind, partition='sampled').splitlines()
    rows[0] = rows[0].replace('source="ordinary"', 'source="%s"' % source)
    metadata = rows[1].split()
    metadata[7:10] = ['1', '0', '1']
    rows[1] = ' '.join(metadata)
    component = packet(registry, source_factor=1. if kind == 'S' else .2,
                       insertion=0. if kind == 'S' else .3)
    return event(kind, weight=-2).replace('</event>', '\n'.join(rows) + '\n' +
                                          component + '</event>')


def lhe_rows(particles):
    result = []
    for p in particles:
        e, px, py, pz = p['p']
        result.append('%d %d 0 0 %d %d %.17e %.17e %.17e %.17e %.17e 0 9' %
                      (p['id'], p['status'], *p['colors'], px, py, pz, e, p.get('m', 0.)))
    return '\n'.join(result) + '\n'


class TestComponentProtocol(unittest.TestCase):
    def evaluator(self, root, delta=True, case=None):
        (root / 'Cards').mkdir()
        (root / 'Cards/run_card.dat').write_text('%s = mcatnlo_delta\n' % delta)
        registry = deepcopy(case['registry'] if case else dict(outer={}))
        registry.update(tables=[], contexts=[], flavour_complete=True)
        directory = root / 'SubProcesses' / registry['outer'].get('directory', 'P0')
        directory.mkdir(parents=True)
        routes = [(1, (2, -2, 11, -11))]
        if case:
            routes = [(s['sector'], tuple(leg[0] for leg in process[1]))
                      for s in registry['outer']['sectors'] for process in s['processes']]
        (directory / 'leshouche_info.dat').write_text(''.join(
            'I %d 1 %s\n' % (sector, ' '.join(map(str, pdgs))) for sector, pdgs in routes))
        digest = hashlib.sha256(json.dumps(registry, sort_keys=True,
                                          separators=(',', ':')).encode()).hexdigest()
        (directory / 'mc_histories.json').write_text(json.dumps(dict(
            schema='macnlops-history-registry-v1', sha256=digest, registry=registry)))
        with patch.object(mc, 'hard_model_masses', return_value={}):
            evaluator = mc.NativeEvaluator(root, root, auxiliary_seed=73, partition='sampled')
        evaluator.configure((1., 1.), (.5, .5))
        return evaluator, digest

    def test_qualified_packet_and_hard_colour_topology(self):
        parsed = mc.Event(annotated())
        self.assertEqual((parsed.component.sector, parsed.component.config,
                          parsed.component.graph, parsed.component.flow), (1, 2, 7, 1))
        self.assertEqual(parsed.component.flow_probability, .5)
        self.assertEqual(parsed.hard.colours, ((501, 0), (0, 501), (0, 0), (0, 0)))
        self.assertIsNone(parsed.delta)
        self.assertEqual(parsed.component, mc.Component.from_block(parsed.block))

    def test_invalid_or_incomplete_component_cannot_select_a_native_density(self):
        base = annotated()
        controls = [
            base.replace(packet(), packet() + packet()),
            base.replace('</macnlops_component>', ''),
            base.replace('1 2 7 1\n', '1 0 7 1\n'),
            base.replace('1 2 7 1\n', '2 2 7 1\n'),
            base.replace('1 2 7 1\n', '1 2 7 2\n'),
            base.replace('.5 ', 'nan '), base.replace('.5 ', '0 '),
            base.replace('registry="' + 'a' * 64, 'registry="ABC'),
            base.replace('1 0 1 0 1 1 1\n', '1 0 1 0 0 1 1\n'),
            annotated('S').replace('1 2 7 1\n.5 0 1 3', '1 2 7 1\n.5 0 .2 3'),
        ]
        for block in controls:
            with self.subTest(block=block), self.assertRaises(mc.MatchingError):
                mc.Event(block)

    def test_source_contract_requires_component_and_separates_old_global_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source.lhe'
            for name, construction in [('ordinary', mc.ORDINARY_SOURCE), ('delta', mc.DELTA_SOURCE)]:
                blocks = [annotated(kind, name) for kind in ('S', 'H')]
                source.write_text(HEADER + ''.join(blocks) + '</LesHouchesEvents>\n')
                contract = mc.hard_source_contract(source, construction)
                self.assertEqual(contract['partition'], 'sampled')
                self.assertEqual(contract['counts'], {'S': 1, 'H': 1})
                source.write_text(HEADER + re.sub(r'<macnlops_component.*?</macnlops_component>',
                    '', blocks[1], flags=re.S) + '</LesHouchesEvents>\n')
                with self.assertRaisesRegex(mc.MatchingError, 'conditional component'):
                    mc.hard_source_contract(source, construction)
            legacy = event('S').replace('</event>', hard_annotation('S') + '</event>')
            source.write_text(HEADER + annotated('H', 'ordinary') + legacy + '</LesHouchesEvents>\n')
            with self.assertRaisesRegex(mc.MatchingError, 'Mixed sampled'):
                mc.hard_source_contract(source, mc.ORDINARY_SOURCE)

    def test_h_request_uses_exact_source_basis_and_undamped_delta_without_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            evaluator, digest = self.evaluator(Path(temp))
            parsed = mc.Event(annotated(registry=digest))
            directory, request, audit = evaluator._component_requests(parsed, 19)[0]
            self.assertEqual(directory.name, 'P0')
            self.assertTrue(request.startswith('MACNLOPS_POINT_V3 19 1 4\n'))
            self.assertIn('COMPONENT_LOCAL_V1 2 7 1 1 3.00000000000000000e+00\n', request)
            self.assertNotIn('DELTA_', request)
            self.assertEqual(audit['prescription'], 'undamped-delta')
            changed = mc.Event(parsed.block.replace('2.00000000000000011e-01', '.9'))
            # Source Delta is provenance only: neither its value nor flow
            # importance-sampling probability enters the physical local ratio.
            changed.component = mc.Component(digest, 1, 2, 7, 1, .9, .8, .9, 3.)
            self.assertEqual(evaluator._component_requests(changed, 19)[0][1], request)
            parsed.component = mc.Component('b' * 64, 1, 2, 7, 1, .5, .3, .2, 3.)
            with self.assertRaisesRegex(mc.MatchingError, 'unique native Born provider'):
                evaluator._component_requests(parsed, 19)

    def test_ordinary_local_mode_keeps_the_native_bogus_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            evaluator, digest = self.evaluator(Path(temp), delta=False)
            parsed = mc.Event(annotated(source='ordinary', registry=digest))
            request = evaluator._component_requests(parsed, 7)[0][1]
            self.assertIn('COMPONENT_LOCAL_V1 2 7 1 0 ', request)
            self.assertNotIn('DELTA_', request)

    def test_saved_s_branch_overrides_integration_owner_and_carries_actual_born(self):
        for case in PREFIXES:
            with self.subTest(process=case['label']), tempfile.TemporaryDirectory() as temp:
                evaluator, digest = self.evaluator(Path(temp), case=case)
                record = case['record']
                source_rows, endpoint_rows = map(lhe_rows,
                    (record['source_particles'], record['particles']))
                born_count, real_count = len(record['source_particles']), len(record['particles'])
                annotation = ('<macnlops_hard_event version="1" frame="lab" source="delta" '
                    'partition="sampled">\n%d 1 99 99 1 1 0 1 0 1 0 1 1 1\n%s'
                    '</macnlops_hard_event>\n' % (born_count, source_rows))
                block = ('<event>\n%d 1 2 100 .007 .118\n%s#aMCatNLO 1 5 1\n%s%s'
                    '<macnlops_endpoint version="1" origin="shower" stage="first"/>\n'
                    '</event>\n' % (real_count, endpoint_rows, annotation,
                                    packet(digest, sector=99, insertion=0., source_factor=1.)))
                parsed = mc.Event(block)
                parsed.shower_record = record
                _, request, audit = evaluator._component_requests(parsed, 5)[0]
                self.assertEqual(audit['sector'], case['expected_sector'])
                self.assertIn('COMPONENT_BORN_V1 %d\n' % born_count, request)
                self.assertNotIn('DELTA_', request)
                self.assertEqual(request.count('END_POINT'), 1)
                self.assertEqual(len(request.splitlines()), real_count + born_count + 4)

    def test_orientation_mark_uses_physical_kernels_and_never_zero_kernel(self):
        with tempfile.TemporaryDirectory() as temp:
            evaluator, digest = self.evaluator(Path(temp))
            parsed = mc.Event(annotated(registry=digest))
            candidates = [(Path(temp), 'first', {'sector': 1}),
                          (Path(temp), 'second', {'sector': 2})]
            values = [mc.Densities(1., 2., 0.), mc.Densities(1., 4., 2.)]
            with patch.object(evaluator, '_component_requests', return_value=candidates), \
                    patch.object(evaluator, '_query', side_effect=values), \
                    patch.object(mc.hashlib, 'sha256') as digest_mock:
                digest_mock.return_value.digest.return_value = b'\xff' * 32
                self.assertEqual(evaluator._evaluate_component(parsed, 1), values[1])
            self.assertEqual(parsed.matching_component['marking_probability'], 1.)
            counts = [0, 0]
            values = [mc.Densities(1., 2., 1.), mc.Densities(1., 4., 3.)]
            with patch.object(evaluator, '_component_requests', return_value=candidates), \
                    patch.object(evaluator, '_query', side_effect=values * 1000):
                for sequence in range(1000):
                    evaluator._evaluate_component(parsed, sequence)
                    counts[parsed.matching_component['sector'] - 1] += 1
            self.assertLess(abs(counts[0] - 250), 50)

    def test_delta_free_density_retains_the_required_signed_h_residual(self):
        # H_source=-Delta*u. Only its acceptance ratio is Delta-free.
        for delta in (.01, .4, 1.):
            density = mc.Densities(2., 5., 2.)
            self.assertEqual(density.acceptance(signed_residuals=True), 0.)
            retained = -delta * 3. * density.hard_acceptance(signed_residuals=True)
            self.assertAlmostEqual(retained, -delta * (3. - 2.))
            with self.assertRaisesRegex(mc.MatchingError, 'exceeds shower density'):
                density.acceptance()


if __name__ == '__main__':
    unittest.main()
