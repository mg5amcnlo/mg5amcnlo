"""MAcNLOPS cancellation, support, event identity and absolute weights."""

import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from madgraph.various.macnlops import (Densities, Event, LHEStream,
                                      MatchingError, reject_proposals,
                                      check_manifest, write_source_manifest,
                                      matching_scale, run_generated_sample,
                                      source_construction, generation_profile,
                                      ORDINARY_SOURCE, RESOLVED_SOURCE,
                                      check_hard_mass_coordinates,
                                      hard_source_contract, hard_model_masses)


HEADER = '''<LesHouchesEvents version="3.0">
<header><MGRunCard>sum = event_norm</MGRunCard></header>
<init>
2212 2212 6500 6500 0 0 260000 260000 -3 1
1 0 1 1
</init>
'''


def event(kind='S', weight=2, label=1):
    return '''<event id='%s'>
4 1 %.12e 100 0.007297 0.118
2 -1 0 0 501 0 0 0 100 100 0 0 9
-2 -1 0 0 0 501 0 0 -100 100 0 0 9
11 1 1 2 0 0 100 0 0 100 0 0 9
-11 1 1 2 0 0 -100 0 0 100 0 0 9
#aMCatNLO %d 5 1 1 2 0 0 0 5 0 0 0 0 0 0
<rwgt><wgt id='1001'> %.12e </wgt></rwgt>
</event>
''' % (label, weight, 1 if kind == 'S' else 2, 1.5 * weight)


def hard_annotation(kind='H', partition='global_sum'):
    # FKS_SUM=0 is allowed: the complete H sum is made by the native
    # history/coalescing construction before sign selection.
    return '''<macnlops_hard_event version="1" frame="lab" source="ordinary" partition="%s">
4 %d 1 1 1 1 1 0 1 0 0 1 1 1
2 -1 0 0 501 0 0 0 100 100 0 0 9
-2 -1 0 0 0 501 0 0 -100 100 0 0 9
11 1 1 2 0 0 100 0 0 100 0 0 9
-11 1 1 2 0 0 -100 0 0 100 0 0 9
</macnlops_hard_event>
''' % (partition, 1 if kind == 'S' else 2)


class TestHardAnnotation(unittest.TestCase):
    def test_signed_born_flow_is_retained_as_source_provenance(self):
        rows = hard_annotation().splitlines()
        fields = rows[1].split()
        fields[7] = '-1'
        rows[1] = ' '.join(fields)
        parsed = Event(event('H').replace('</event>', '\n'.join(rows) + '\n</event>'))
        self.assertEqual(parsed.hard.metadata[7], -1)

    def test_four_flavour_bottom_mass_is_retained_in_the_hard_point(self):
        def bottom_pair(block, mass):
            momentum = math.sqrt(100.**2 - mass**2)
            return block.replace('11 1 1 2 0 0 100 0 0 100 0 0 9',
                '5 1 1 2 0 0 %.17g 0 0 100 %.17g 0 9' % (momentum, mass)).replace(
                '-11 1 1 2 0 0 -100 0 0 100 0 0 9',
                '-5 1 1 2 0 0 %.17g 0 0 100 %.17g 0 9' % (-momentum, mass))
        body = bottom_pair(event('H'), 4.8)
        block = body.replace('</event>', bottom_pair(hard_annotation(), 4.7) + '</event>')
        parsed = Event(block)
        check_hard_mass_coordinates(parsed, {5: 4.7})
        with self.assertRaisesRegex(MatchingError, 'hard model mass'):
            check_hard_mass_coordinates(parsed, {})
        self.assertEqual(parsed.particle_masses[2:], [4.8, 4.8])
        self.assertEqual(parsed.hard.masses[2:], (4.7, 4.7))
        self.assertEqual(parsed.block, block)

    def test_source_contract_checks_both_streams_before_any_veto(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.lhe'
            blocks = [event(kind).replace('</event>', hard_annotation(kind) + '</event>')
                      for kind in ('H', 'S')]
            source.write_text(HEADER + ''.join(blocks) + '</LesHouchesEvents>\n')
            contract = hard_source_contract(source, ORDINARY_SOURCE)
            self.assertEqual(contract['counts'], {'H': 1, 'S': 1})
            source.write_text(HEADER + blocks[0] + blocks[1].replace(
                'partition="global_sum"', 'partition="sampled"') + '</LesHouchesEvents>\n')
            with self.assertRaisesRegex(MatchingError, 'event 2.*sampled H partition'):
                hard_source_contract(source, ORDINARY_SOURCE)

    def test_hard_h_coordinates_override_only_the_mass_shuffled_body(self):
        body = event('H').replace('0 0 -100 100 0 0 9',
                                  '0 0 -99.999 100 .33 0 9')
        block = body.replace('</event>', hard_annotation() + '</event>')
        parsed = Event(block)
        self.assertEqual(parsed.block, block)
        self.assertEqual(parsed.particle_masses[1], .33)
        self.assertEqual(parsed.particles[1][2][3], -99.999)
        self.assertEqual(parsed.external()[1][1][3], -100.)
        check_hard_mass_coordinates(parsed, {})
        self.assertEqual(parsed.hard.partition, 'global_sum')

    def test_s_emission_uses_real_endpoint_and_keeps_born_as_provenance(self):
        endpoint = event('S').replace('100 0 0 100 0 0 9',
                                     '90 0 0 90 0 0 9')
        endpoint = endpoint.replace('</event>', hard_annotation('S') +
            '<macnlops_endpoint version="1" origin="shower" stage="first"/>\n</event>')
        parsed = Event(endpoint)
        self.assertEqual(parsed.external()[2][1][0], 90.)
        self.assertEqual(parsed.hard.particles[2][2][0], 100.)
        self.assertEqual(parsed.endpoint_origin, 'shower')

    def test_invalid_contracts_fail_without_modifying_the_event(self):
        annotation = hard_annotation()
        invalid = [annotation + annotation,
                   annotation.replace('</macnlops_hard_event>', ''),
                   annotation.replace('version="1"', 'version="2"'),
                   annotation.replace('frame="lab"', 'frame="lab" frame="lab"'),
                   annotation.replace('4 2 1 1 1 1 1 0 1 0 0 1 1 1',
                                      '4 1 1 1 1 1 1 0 1 0 0 1 1 1'),
                   annotation.replace('4 2 1 1 1 1 1 0 1 0 0 1 1 1',
                                      '4 2 1 1 1 1 1 0 1 1 0 1 1 1'),
                   annotation.replace('-2 -1 0 0', '-1 -1 0 0'),
                   annotation.replace('100 100 0 0 9', '100 nan 0 0 9')]
        for extra in invalid:
            with self.subTest(extra=extra), self.assertRaises(MatchingError):
                Event(event('H').replace('</event>', extra + '</event>'))

    def test_malformed_endpoint_marker_cannot_be_treated_as_a_source(self):
        for marker in ['<macnlops_endpoint version="1"', '</macnlops_endpoint>',
                       '<macnlops_endpoint version="1" origin="shower" stage="full"/>']:
            block = event('S').replace('</event>', hard_annotation('S') + marker + '\n</event>')
            with self.subTest(marker=marker), self.assertRaisesRegex(MatchingError, 'endpoint'):
                Event(block)

    def test_s_source_flavours_must_match_the_saved_hard_point(self):
        block = event('S').replace('</event>', hard_annotation('S').replace(
            '-2 -1 0 0', '-1 -1 0 0') + '</event>')
        with self.assertRaisesRegex(MatchingError, 'flavours disagree'):
            Event(block)


class TestMatchingAlgebra(unittest.TestCase):
    def test_ordinary_wbb_bound_failure_retains_the_required_signed_residual(self):
        # Actual canonical-point ordinary G/P diagnostic, not a rescaled
        # artificial kernel chosen to make the probability fit.
        density = Densities(3.0056558293616222e10,
                            8.068707181129524e10,
                            4.5793903749948944e10)
        with self.assertRaisesRegex(MatchingError, 'exceeds'):
            density.acceptance()
        self.assertEqual(density.acceptance(signed_residuals=True), 0.)
        residual = density.hard_acceptance(signed_residuals=True)
        self.assertGreater(residual, 0.)
        negative_hard = density.subtraction - density.real
        self.assertAlmostEqual(negative_hard * (1-residual) / density.shower, 1.)

    def test_original_formula_only_when_subtraction_equals_shower(self):
        self.assertAlmostEqual(Densities(.8, 1, 1).acceptance(), .8)
        self.assertEqual(Densities(2, 1, 1).acceptance(), 1)
        self.assertAlmostEqual(Densities(.8, .9, 1).acceptance(), .9)
        self.assertNotAlmostEqual(Densities(.8, .9, 1).acceptance(), .8 / .9)

    def test_nlo_cancellation_with_different_shower_and_subtraction(self):
        # A one-bin real contribution. K and H scale as alpha_s, while
        # barB/B and Delta start at one. The surviving difference is quadratic.
        differences = []
        for alpha in (.04, .02, .01):
            real, subtraction, shower = (alpha * v for v in (.8, .9, 1))
            factor = (1 + .4 * alpha) * (1 - .7 * alpha)
            ordinary = factor * shower + real - subtraction
            transformed = factor * shower * Densities(real, subtraction, shower).acceptance()
            differences.append(transformed - ordinary)
        for i in (0, 1):
            self.assertAlmostEqual(differences[i] / differences[i + 1], 4, delta=.08)

    def test_take_sign_only_after_history_sum(self):
        # Opposite-signed native remainders at one physical real point.
        real = .4 + .6
        subtraction = .8 + .1
        self.assertEqual(Densities(real, subtraction, 1).acceptance(), 1)
        self.assertGreater(max(.8 - .4, 0) + max(.1 - .6, 0), 0)

    def test_signed_residual_compensates_a_nonvetoable_h_density(self):
        for real, subtraction, shower in ((0, 2, 1), (0, 2, 0), (.8, .9, 1)):
            density = Densities(real, subtraction, shower)
            negative_hard = subtraction - real
            factor = 1.03
            ordinary = factor * shower - negative_hard
            transformed = (factor * shower * density.acceptance(signed_residuals=True) -
                           negative_hard * density.hard_acceptance(signed_residuals=True))
            cancellation = min(negative_hard, shower)
            self.assertAlmostEqual(transformed - ordinary, cancellation * (1 - factor))
        self.assertEqual(Densities(0, 2, 0).hard_acceptance(signed_residuals=True), 1)

    def test_support_and_envelope_violations_are_errors(self):
        for density in (Densities(0, 1, 0), Densities(.1, 2, 1),
                        Densities(-1, 1, 1), Densities(1, 1, -1),
                        Densities(1, float('nan'), 1)):
            with self.subTest(density=density), self.assertRaises(MatchingError):
                density.acceptance()
        self.assertEqual(Densities(0, 0, 0).acceptance(), 1)
        self.assertEqual(Densities(0, 1.0000000000000004, 1).acceptance(), 0)
        with self.assertRaises(MatchingError):
            Densities(0, 1.000000001, 1).acceptance()

    def test_signed_pdf_luminosities_leave_nonpositive_shower_measure_unchanged(self):
        # The saved DY Delta event has a negative high-x anti-strange PDF.
        dy = Densities(-180137.16691678748, -186101.79956101454,
                       -186101.79956101452)
        self.assertEqual(dy.acceptance(signed_residuals=True), 1.)
        for density in (Densities(-3., -2., -1.), Densities(1., 3., -2.)):
            self.assertEqual(density.acceptance(signed_residuals=True), 1.)
            self.assertEqual(density.hard_acceptance(signed_residuals=True), 1.)
        for density in (dy, Densities(-3., -2., -1.)):
            with self.assertRaises(MatchingError):
                density.acceptance()

    def test_signed_pdf_transfer_is_bounded_by_positive_signed_shower_support(self):
        for real, subtraction, shower in ((-1., 1., 1.), (-1., 0., 0.),
                                         (-3., -2., -1.), (1., 3., -2.)):
            with self.subTest(densities=(real, subtraction, shower)):
                density = Densities(real, subtraction, shower)
                negative_hard = subtraction - real
                transfer = min(negative_hard, max(shower, 0.))
                s_accept = density.acceptance(signed_residuals=True)
                h_accept = density.hard_acceptance(signed_residuals=True)
                self.assertAlmostEqual(shower * (1. - s_accept), transfer)
                self.assertAlmostEqual(negative_hard * (1. - h_accept), transfer)
                self.assertGreaterEqual(s_accept, 0.)
                self.assertLessEqual(s_accept, 1.)
                self.assertGreaterEqual(h_accept, 0.)
                self.assertLessEqual(h_accept, 1.)
        for density in (Densities(float('nan'), 1., -1.),
                        Densities(-1., 1., float('-inf')),
                        Densities(-1.e308, 1.e308, 1.)):
            with self.assertRaises(MatchingError):
                density.acceptance(signed_residuals=True)


class TableEvaluator:
    def __init__(self, densities):
        self.densities = densities
        self.requests = []

    def evaluate(self, event, sequence):
        self.requests.append(sequence)
        return self.densities[sequence]

    def configure(self, *args):
        pass


class TestRejection(unittest.TestCase):
    def test_unmodified_shower_mass_coordinates_are_not_used_as_hard_coordinates(self):
        block = event()
        check_hard_mass_coordinates(Event(block), {})
        changed = block.replace('0 0 -100 100 0 0 9',
                                '0 0 -100 100 .33 0 9')
        incoming = Event(changed)
        with self.assertRaisesRegex(MatchingError, 'inverse write-time transport'):
            check_hard_mass_coordinates(incoming, {})
        self.assertEqual(incoming.block, changed)
        electron = Event(block.replace('100 0 0 100 0 0 9',
                                       '100 0 0 100 .000511 0 9'))
        with self.assertRaisesRegex(MatchingError, 'hard model mass'):
            check_hard_mass_coordinates(electron, {})
        check_hard_mass_coordinates(electron, {11: .000511})

    def inputs(self, folder, events, records):
        proposals = folder / 'proposals.lhe'
        proposals.write_text(HEADER + ''.join(events) + '</LesHouchesEvents>\n')
        metadata = folder / 'shower.jsonl'
        metadata.write_text(''.join(json.dumps(dict(sequence=i, event_kind=kind,
                                                   emitted=emitted)) + '\n'
                                    for i, (kind, emitted) in enumerate(records, 1)))
        return proposals, metadata

    def test_no_emission_signed_s_and_h_and_whole_event_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            blocks = [event('S', -2, 1), event('S', 2, 2),
                      event('H', -2, 3), event('H', 2, 4)]
            proposals, metadata = self.inputs(folder, blocks,
                [('S', False), ('S', True), ('H', False), ('H', False)])
            evaluator = TableEvaluator({2: Densities(0, 1, 1),
                                        3: Densities(0, 1, 1)})
            output = folder / 'accepted.lhe.gz'
            report = reject_proposals(proposals, metadata, output, evaluator)
            with LHEStream(output) as stream:
                accepted = list(stream)
            self.assertEqual(accepted, [blocks[0], blocks[3]])
            self.assertEqual(evaluator.requests, [2, 3])
            self.assertEqual(report['weight_sums']['accepted'], 0)
            self.assertEqual(report['counts']['accepted_S'], 1)
            self.assertEqual(report['counts']['rejected_S'], 1)
            self.assertEqual(report['counts']['rejected_H'], 1)
            # Original alternate weights remain associated with their event.
            self.assertIn('-3.000000000000e+00', accepted[0])

    def test_rejection_does_not_renormalize_retained_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            proposals, metadata = self.inputs(folder,
                [event('S', 2, 1), event('S', 2, 2)], [('S', False), ('S', True)])
            output = folder / 'accepted.lhe.gz'
            reject_proposals(proposals, metadata, output,
                             TableEvaluator({2: Densities(0, 1, 1)}))
            with LHEStream(output) as stream:
                self.assertEqual([Event(block).weight for block in stream], [2])

    def test_signed_h_residual_is_sampled_with_original_negative_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            proposals, metadata = self.inputs(folder,
                [event('H', -2, 1), event('S', 2, 2)], [('H', False), ('S', True)])
            output = folder / 'accepted.lhe.gz'
            reject_proposals(proposals, metadata, output,
                TableEvaluator({1: Densities(0, 2, 1), 2: Densities(0, 2, 1)}),
                seed=1, signed_residuals=True)
            with LHEStream(output) as stream:
                self.assertEqual([Event(block).weight for block in stream], [-2])

    def test_invalid_density_never_publishes_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            proposals, metadata = self.inputs(folder, [event(), event()], [('S', False), ('S', True)])
            output = folder / 'accepted.lhe.gz'
            with self.assertRaises(MatchingError):
                reject_proposals(proposals, metadata, output,
                                 TableEvaluator({2: Densities(0, 2, 1)}))
            self.assertFalse(output.exists())
            self.assertEqual(sorted(p.name for p in folder.iterdir()),
                             ['proposals.lhe', 'shower.jsonl'])

    def test_shower_proposals_cannot_be_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            proposals, metadata = self.inputs(folder, [event(), event()], [('S', False)])
            output = folder / 'accepted.lhe.gz'
            with self.assertRaisesRegex(MatchingError, 'Missing'):
                reject_proposals(proposals, metadata, output, TableEvaluator({}))
            self.assertFalse(output.exists())

    def test_actual_born_history_gates_threshold_emission(self):
        profile = dict(shower_cutoffs=[.5, .6], beam_xmax=[.99, .99],
                       heavy_mass_floors={'4': 1.2, '5': 4.})
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            proposals, metadata = self.inputs(folder, [event(), event()],
                                              [('S', True), ('S', True)])
            records = [json.loads(line) for line in metadata.read_text().splitlines()]
            # Same real event, different actual Born histories. Only the light
            # history participates in the shower-density veto at pT=3 GeV.
            for record, pdg in zip(records, (5, 2)):
                record.update(source_particles=[dict(id=pdg, status=-1),
                                                 dict(id=-pdg, status=-1)],
                              pt_evol=3., beam_xmax=profile['beam_xmax'],
                              shower_cutoffs=profile['shower_cutoffs'])
            metadata.write_text(''.join(json.dumps(r) + '\n' for r in records))
            evaluator = TableEvaluator({2: Densities(0, 1, 1)})
            report = reject_proposals(proposals, metadata, folder / 'out.lhe',
                                      evaluator, profile=profile)
            self.assertEqual(evaluator.requests, [2])
            self.assertEqual(report['counts']['below_matching_boundary_S'], 1)
            self.assertEqual(report['counts']['accepted_S'], 1)
            self.assertEqual(report['counts']['rejected_S'], 1)

    def test_invalid_boundary_and_profile_changes_fail_closed(self):
        profile = dict(shower_cutoffs=[.5, .6], beam_xmax=[.99, .99],
                       heavy_mass_floors={'4': 1.2, '5': 4.})
        with self.assertRaisesRegex(MatchingError, 'incoming'):
            matching_scale(dict(source_particles=[], pt_evol=1), profile)
        record = dict(source_particles=[dict(id=2, status=-1)] * 2,
                      pt_evol=float('nan'))
        with self.assertRaisesRegex(MatchingError, 'evolution'):
            matching_scale(record, profile)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            proposals, metadata = self.inputs(folder, [event()], [('S', True)])
            record = json.loads(metadata.read_text())
            record.update(beam_xmax=[1., 1.], shower_cutoffs=[.5, .6])
            metadata.write_text(json.dumps(record) + '\n')
            with self.assertRaisesRegex(MatchingError, 'generation profile'):
                reject_proposals(proposals, metadata, folder / 'out.lhe',
                                 TableEvaluator({}), profile=profile)
            self.assertFalse((folder / 'out.lhe').exists())


class TestSourceState(unittest.TestCase):
    def test_dependent_model_masses_and_distinct_exported_models(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as tmp:
            for name, bottom in [('five', 0.), ('four', 4.7)]:
                process = Path(tmp) / name
                (process / 'Cards').mkdir(parents=True)
                (process / 'bin/internal').mkdir(parents=True)
                (process / 'bin/internal/ufomodel').symlink_to(root / 'models/loop_sm', target_is_directory=True)
                text = (root / 'models/loop_sm/restrict_default.dat').read_text()
                text = text.replace('4.700000e+00', '%.6e' % bottom)
                text = text.replace('Block MASS', 'Block MASS\n   24 8.041900e+01 # inactive rounded dependent MW')
                card = process / 'Cards/param_card.dat'
                card.write_text(text)
                masses = hard_model_masses(process)
                self.assertEqual(masses[5], bottom)
                self.assertAlmostEqual(masses[24], 80.41900244575616, places=11)
                self.assertNotEqual(masses[24], 80.419)
                self.assertEqual(card.read_text(), text)

    def test_source_construction_accepts_the_actual_generation_run_card(self):
        from madgraph.various.banner import RunCardNLO
        card = RunCardNLO()
        self.assertTrue(card['store_hard_momenta'])
        self.assertEqual(source_construction(card), ORDINARY_SOURCE)
        card.set('macnlops_resolved_source', True, user=True)
        self.assertEqual(source_construction(card), RESOLVED_SOURCE)

    def test_ordinary_source_is_the_default_and_legacy_source_is_explicit(self):
        self.assertEqual(source_construction({}), ORDINARY_SOURCE)
        for value in (False, 'false', '.false.', 'f'):
            self.assertEqual(source_construction(dict(macnlops_resolved_source=value)),
                             ORDINARY_SOURCE)
        for value in (True, 'true', '.true.', 't'):
            self.assertEqual(source_construction(dict(macnlops_resolved_source=value)),
                             RESOLVED_SOURCE)
        with self.assertRaises(MatchingError):
            source_construction(dict(macnlops_resolved_source='typo'))

    def test_a_saved_profile_cannot_reinterpret_an_existing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'Cards').mkdir()
            card = work / 'Cards/run_card.dat'
            profile = work / 'Cards/macnlops_profile.json'
            card.write_text('true = macnlops\nfalse = macnlops_resolved_source\n')
            profile.write_text(json.dumps(dict(schema='macnlops-generation-profile-v1',
                                               construction=ORDINARY_SOURCE)))
            self.assertEqual(generation_profile(work)['construction'], ORDINARY_SOURCE)
            card.write_text('true = macnlops\ntrue = macnlops_resolved_source\n')
            with self.assertRaisesRegex(MatchingError, 'constructions differ'):
                generation_profile(work)
            profile.write_text(json.dumps(dict(schema='macnlops-generation-profile-v1',
                                               construction=RESOLVED_SOURCE)))
            self.assertEqual(generation_profile(work)['construction'], RESOLVED_SOURCE)

    def test_normal_launch_dispatches_only_matching_nonempty_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'SubProcesses').mkdir()
            events = work / 'Events/test'
            events.mkdir(parents=True)
            collected = events / 'events.lhe.gz'
            with gzip.open(collected, 'wt') as stream:
                stream.write(HEADER + event() + '</LesHouchesEvents>\n')
            channel = work / 'SubProcesses/channel.lhe'
            channel.write_text(HEADER + event() + '</LesHouchesEvents>\n')
            (work / 'SubProcesses/nevents_unweighted').write_text(
                'channel.lhe 1\nempty.lhe 0\n')
            write_source_manifest(work, 'test', [collected, channel])
            with patch('madgraph.various.macnlops.main') as run:
                output = run_generated_sample(work, collected.with_suffix(''), '/pythia', 42)
                args = run.call_args.args[0]
                self.assertEqual(args[-1], str(channel))
                self.assertEqual(output, events / 'macnlops/events.lhe.gz')
            # A stale raw list must never select a different run's channels.
            write_source_manifest(work, 'other', [channel])
            with patch('madgraph.various.macnlops.main') as run:
                run_generated_sample(work, collected, '/pythia', 42)
                self.assertEqual(run.call_args.args[0][-1], str(collected))

    def test_matching_profile_and_source_kernel_are_generation_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'Cards').mkdir()
            (work / 'SubProcesses').mkdir()
            profile = work / 'Cards/macnlops_profile.json'
            profile.write_text('{"construction":"resolved-convex-v1"}')
            helper = work / 'SubProcesses/mc_pythia8_kernel.f90'
            helper.write_text('source kernel')
            path = work / 'events.lhe'
            path.write_text(HEADER + event() + '</LesHouchesEvents>\n')
            write_source_manifest(work, 'test', [path])
            record = check_manifest(path, work)
            self.assertIn('Cards/macnlops_profile.json', record['sources'])
            self.assertIn('SubProcesses/mc_pythia8_kernel.f90', record['sources'])
            helper.write_text('different support')
            with self.assertRaisesRegex(MatchingError, 'changed'):
                check_manifest(path, work)

    def test_manifest_checks_decompressed_events_and_original_subtraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'Cards').mkdir()
            (work / 'SubProcesses').mkdir()
            card = work / 'Cards/run_card.dat'
            card.write_text('true = macnlops\n')
            subtraction = work / 'SubProcesses/fks_singular.f'
            subtraction.write_text('original subtraction')
            path = work / 'events.lhe.gz'
            with gzip.open(path, 'wt') as stream:
                stream.write(HEADER + event() + '</LesHouchesEvents>\n')
            write_source_manifest(work, 'test', [path])
            check_manifest(path, work)
            plain = work / 'events.lhe'
            with gzip.open(path, 'rt') as stream:
                plain.write_text(stream.read())
            check_manifest(plain, work)
            subtraction.write_text('changed subtraction')
            with self.assertRaisesRegex(MatchingError, 'changed'):
                check_manifest(path, work)


if __name__ == '__main__':
    unittest.main()
