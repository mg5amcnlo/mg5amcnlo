"""Bind sampled proposal sidecars and collection labels to the actual LHE."""
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import tempfile
import unittest

from madgraph.various import macnlops as mc
from tests.unit_tests.various.test_macnlops import HEADER, hard_annotation
from tests.unit_tests.various.test_macnlops_component_protocol import annotated
from tests.unit_tests.various.test_macnlops_continuation import endpoint


def qualified_endpoint(state='emitted'):
    kind = 'H' if state == 'unshowered' else 'S'
    source = annotated(kind, source='ordinary')
    hard = re.search(r'<macnlops_hard_event\b.*?</macnlops_hard_event>', source, re.S)[0]
    component = re.search(r'<macnlops_component\b.*?</macnlops_component>', source, re.S)[0]
    block = re.sub(r'<macnlops_hard_event\b.*?</macnlops_hard_event>', hard,
                   endpoint(state), flags=re.S)
    return block.replace('</event>', component + '\n</event>')


def native_record(event):
    def particles(rows, masses, colours):
        return [dict(id=pdg, status=status, colors=list(colour), p=list(point), mass=mass)
                for (pdg, status, point), mass, colour in zip(rows, masses, colours)]
    names = ('nup', 'kind', 'owner', 'native_sector', 'native_provider', 'native_context',
             'native_history', 'born_flow', 'mc_kl_sum', 'fixed_flow', 'fks_sum',
             'flavour_coalesced', 'max_bcol', 'group_size')
    partition = dict(zip(names, event.hard.metadata))
    partition.update(version=1, frame='lab', source=event.hard.source,
                     partition=event.hard.partition)
    hard = particles(event.hard.particles, event.hard.masses, event.hard.colours)
    continuation = event.continuation
    emitted = continuation.state == 'emitted'
    return dict(sequence=1, event_kind=event.kind, emitted=emitted,
        source_weight=event.weight, source_partition=partition, beam_xmax=[1., 1.],
        particles=particles(event.particles, event.particle_masses, event.particle_colours),
        source_hard_particles=hard, source_particles=deepcopy(hard),
        pt_evol=event.scale if emitted else 0., continuation_state=continuation.state,
        continuation_scalup=event.scale, source_scalup=event.source_scale,
        shower_cutoffs=[continuation.cutoff_isr, continuation.cutoff_fsr],
        continuation_leg_scales=[list(item) for item in sorted(event.continuation_leg_scales.items())],
        continuation_dipole_scales=[list(pair) + [scale] for pair, scale in
                                   sorted(event.continuation_dipole_scales.items())])


class TestSampledProposalBinding(unittest.TestCase):
    def test_exact_records_and_one_ulp_roundtrip_are_accepted(self):
        for state in ('emitted', 'exhausted', 'unshowered'):
            event = mc.Event(qualified_endpoint(state))
            record = native_record(event)
            event.check_proposal_metadata(record, True)
            record['particles'][0]['p'][0] = math.nextafter(record['particles'][0]['p'][0], math.inf)
            event.check_proposal_metadata(record, True)

    def test_binding_applies_before_emission_and_boundary_selection(self):
        for state in ('emitted', 'exhausted'):
            event = mc.Event(qualified_endpoint(state))
            record = native_record(event)
            record['source_particles'][0]['id'] = 5
            # The source list also controls the heavy-flavour protection gate.
            with self.assertRaisesRegex(mc.MatchingError, 'showered source point'):
                event.check_proposal_metadata(record, True)

    def test_missing_or_stale_sidecar_cannot_reuse_constant_weights(self):
        event = mc.Event(qualified_endpoint())
        record = native_record(event)
        for key in ('particles', 'source_hard_particles', 'source_particles',
                    'source_partition', 'source_weight', 'beam_xmax', 'continuation_leg_scales'):
            stale = deepcopy(record); del stale[key]
            with self.subTest(missing=key), self.assertRaises(mc.MatchingError):
                event.check_proposal_metadata(stale, True)
        for key in ('particles', 'source_hard_particles', 'source_particles'):
            for field, value in (('id', 1), ('status', 1), ('colors', [502, 0]),
                                 ('mass', .33), ('p', [101., 0., 0., 100.])):
                stale = deepcopy(record); stale[key][0][field] = value
                with self.subTest(packet=key, field=field), self.assertRaises(mc.MatchingError):
                    event.check_proposal_metadata(stale, True)
        stale = deepcopy(record); stale['source_partition']['group_size'] += 1
        with self.assertRaisesRegex(mc.MatchingError, 'source partition'):
            event.check_proposal_metadata(stale, True)

    def test_s_certificate_and_resolved_source_cannot_be_fabricated(self):
        block = annotated('S')
        for position in (11, 13):
            rows = block.splitlines()
            row = next(i + 1 for i, text in enumerate(rows) if '<macnlops_hard_event ' in text)
            fields = rows[row].split(); fields[position] = '0'; rows[row] = ' '.join(fields)
            with self.assertRaises(mc.MatchingError):
                mc.Event('\n'.join(rows))
        with self.assertRaisesRegex(mc.MatchingError, 'resolved source'):
            mc.Event(annotated('S', source='resolved'))
        with self.assertRaisesRegex(mc.MatchingError, 'resolved source'):
            mc.matching_prescription(mc.RESOLVED_SOURCE, 'sampled')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'Cards').mkdir()
            (root / 'Cards/run_card.dat').write_text('True = macnlops\nTrue = macnlops_resolved_source\n')
            with self.assertRaisesRegex(mc.MatchingError, 'resolved source'):
                mc.NativeEvaluator(root, root, partition='sampled')


class TestCollectionBodyPolicy(unittest.TestCase):
    def collect(self, folder, block, policy):
        header = HEADER.replace('</header>', '<macnlops normalization="sum" stage="first-emission"%s>'
            '</macnlops></header>' % ('' if policy is None else ' prescription="%s"' % policy))
        source = folder / 'input.lhe'; source.write_text(header + block + '</LesHouchesEvents>\n')
        event = mc.Event(block)
        report = dict(counts=dict(input_H=1, input_S=0, accepted_H=1, accepted_S=0),
                      veto_prescription=policy, weight_sums=dict(accepted=event.weight))
        return mc.collect_matched_events([source], [report], folder / 'output.lhe', 5)

    def test_sampled_policy_cannot_be_missing_or_relabel_global(self):
        for policy in (None, 'global-source-v1', 'component-source-v1'):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex(mc.MatchingError, 'body/header'):
                    self.collect(Path(folder), annotated('H'), policy)
                self.assertFalse((Path(folder) / 'output.lhe').exists())

    def test_new_policy_requires_qualified_body_annotation(self):
        block = re.sub(r'<macnlops_component\b.*?</macnlops_component>', '', annotated('H'), flags=re.S)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(mc.MatchingError, 'qualified component'):
                self.collect(Path(folder), block, 'component-undamped-delta-v1')
        block = re.sub(r'<macnlops_hard_event\b.*?</macnlops_hard_event>', '', block, flags=re.S)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(mc.MatchingError, 'source provenance'):
                self.collect(Path(folder), block, 'component-undamped-delta-v1')

    def test_truthful_sampled_and_legacy_global_are_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = self.collect(Path(folder), annotated('H'), 'component-undamped-delta-v1')
            self.assertEqual(summary['veto_prescription'], 'component-undamped-delta-v1')
        with tempfile.TemporaryDirectory() as folder:
            block = re.sub(r'<macnlops_component\b.*?</macnlops_component>', '', annotated('H'), flags=re.S)
            block = re.sub(r'<macnlops_hard_event\b.*?</macnlops_hard_event>',
                           hard_annotation('H'), block, flags=re.S)
            self.assertIsNone(self.collect(Path(folder), block, None)['veto_prescription'])


class TestSavedNativeSmokeBinding(unittest.TestCase):
    def test_all_saved_genuine_sampled_records_are_bound_read_only(self):
        root = Path(__file__).resolve().parents[3] / 'docs/audits/sampled_source_smoke_20261025'
        samples = [(root / 'DY_uubar', 'smoke_ordinary'),
                   (root / 'DY_uubar_delta', 'smoke_delta'),
                   (root / 'DY_uubar_delta', 'smoke_delta128')]
        if not all((process / 'Events' / name / 'macnlops/events/proposals.lhe').is_file()
                   for process, name in samples):
            self.skipTest('local genuine source smoke files are optional audit artifacts')
        count = 0
        for process, name in samples:
            job = process / 'Events' / name / 'macnlops/events'
            records = [json.loads(line) for line in (job / 'shower.jsonl').read_text().splitlines()]
            with mc.LHEStream(job / 'proposals.lhe') as stream:
                blocks = list(stream)
            self.assertEqual(len(blocks), len(records))
            for block, record in zip(blocks, records):
                mc.Event(block).check_proposal_metadata(record, True)
                count += 1
        self.assertEqual(count, 256)


if __name__ == '__main__':
    unittest.main()
