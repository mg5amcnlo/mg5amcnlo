"""Ordinary/Delta multistep orchestration with tiny native-boundary fixtures.

The native executables return prescribed endpoints and densities, not simulated
physics. Profile/source checks, evaluator routing/protocol, rejection, ordinary
event collection and continuation dispatch all execute their production code.
No process export, native compilation, PDFs or Pythia installation are needed.
"""
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from madgraph.various import macnlops as mc
from madgraph.various.banner import RunCardNLO
from tests.unit_tests.various.test_macnlops import HEADER, event, hard_annotation
from tests.unit_tests.various.test_macnlops_continuation import endpoint
from tests.unit_tests.various.test_macnlops_delta_protocol import packet


PRODUCER = '''import json, re, sys
from pathlib import Path
root = Path(__file__).parent
fixture = json.loads((root / 'native.json').read_text())
args = sys.argv[1:]
with (root / 'producer_calls.jsonl').open('a') as log:
    log.write(json.dumps(args) + '\\n')
def option(name):
    return Path(args[args.index(name) + 1])
if '--probe-config' in args:
    option('--probe-config').write_text(json.dumps(fixture['support']))
else:
    source = option('--input').read_text()
    labels = re.findall(r"<event id='(\\d+)'>", source)
    if any(label in fixture['fail_labels'] for label in labels):
        sys.exit(17)
    header = source.split('<event ', 1)[0]
    blocks, rows = [], []
    for sequence, label in enumerate(labels, 1):
        record = fixture['events'][label]
        blocks.append(record['endpoint'])
        rows.append(dict(record['metadata'], sequence=sequence))
    option('--output').write_text(header + ''.join(blocks) + '</LesHouchesEvents>\\n')
    option('--metadata').write_text(''.join(json.dumps(row) + '\\n' for row in rows))
    effective = dict(fixture['effective_settings'])
    for family in ('SpaceShower', 'TimeShower'):
        effective[family + ':alphaSvalue'] = args[args.index('--alpha-s-mz') + 1]
        effective[family + ':alphaSorder'] = args[args.index('--alpha-s-order') + 1]
    effective['PDF:pSet'] = args[args.index('--pdf') + 1]
    Path(str(option('--metadata')) + '.settings.cmnd').write_text(
        option('--settings').read_text() + ''.join('%s = %s\\n' % item for item in effective.items()))
'''


EVALUATOR = '''import json, sys
from pathlib import Path
log = Path(__file__).parent / 'requests.jsonl'
# R, R-H, K: residual H; removable H; vetoable S; surviving S.
densities = {2: (1, 5, 1), 3: (1, 2, 2), 4: (1, 2, 4), 5: (1, 1.5, 4)}
request = []
for line in sys.stdin:
    request.append(line)
    if line.startswith('MACNLOPS_CONFIG_'):
        with log.open('a') as stream:
            stream.write(json.dumps(''.join(request)) + '\\n')
        request = []
    elif line.strip() == 'END_POINT':
        with log.open('a') as stream:
            stream.write(json.dumps(''.join(request)) + '\\n')
        sequence = int(request[0].split()[1])
        print('MACNLOPS_DENSITY_V1', sequence, 0, *densities[sequence], flush=True)
        request = []
'''


class TestDeltaWorkflow(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='macnlops-workflow-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.process = self.root / 'process'
        self.cards = self.process / 'Cards'
        self.cards.mkdir(parents=True)
        self.context = self.process / 'SubProcesses/P1'
        self.context.mkdir(parents=True)
        (self.context / 'Makefile').write_text('macnlops_evaluator:\n\t@true\n')
        self.executable(self.context / 'macnlops_evaluator', EVALUATOR)
        self.driver = self.root / 'native_fixture'
        self.executable(self.driver, PRODUCER)
        self.pythia = self.root / 'pythia'
        (self.pythia / 'lib').mkdir(parents=True)
        (self.pythia / 'lib/libpythia8.so').write_text('fixture library identity\n')
        pdf = self.root / 'pdfs/fixture'
        pdf.mkdir(parents=True)
        (pdf.parent / 'pdfsets.index').write_text('260000 fixture\n')
        (pdf / 'fixture.info').write_text(
            'NumMembers: 1\nNumFlavors: 5\nAlphaS_MZ: 0.118\nAlphaS_OrderQCD: 1\n')
        (pdf / 'fixture_0000.dat').write_text('fixture PDF identity; never evaluated\n')
        (self.cards / 'param_card.dat').write_text('Block MASS\n  23 91.188\n')
        (self.process / 'SubProcesses/FKS_params.dat').write_text(
            'MCExplicitKLSum = .true.\nMCSubtractionAtFixedFlow = .false.\n')
        self.install_registry()
        self.native = dict(support=dict(schema='macnlops-shower-config-v1',
            n_quark_in=5, n_gluon_to_quark=5, flavour_count=5,
            alpha_s_nfmax=5, particle_masses={'1': 0., '4': 0., '5': 0., '21': 0.},
            qcd_only_off_switches={'PartonLevel:MPI': False, 'PartonLevel:Remnants': False},
            shower_cutoffs=[.5, .5], heavy_mass_floors={'4': 1.5, '5': 4.8},
            beam_xmax=[1., 1.], beam_ids=[2212, 2212],
            beam_energies_lab=[6500., 6500.]), events={}, fail_labels=[],
            effective_settings={'SpaceShower:nQuarkIn': '5', 'TimeShower:nGluonToQuark': '5',
                                'StandardModel:alphaSnfmax': '5', 'SpaceShower:pT0Ref': '0.'})
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, LHAPDF_DATA_PATH=str(pdf.parent)))
        self.stack.enter_context(patch.object(mc, 'hard_model_masses', return_value={}))
        self.build = self.stack.enter_context(patch.object(mc, 'build_shower', return_value=self.driver))
        # Keep tiny fixture collection independent of optional parallel gzip.
        self.stack.enter_context(patch.object(mc.collect_events.shutil, 'which', return_value=None))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    @staticmethod
    def executable(path, program):
        path.write_text('#!%s\n%s' % (sys.executable, program))
        path.chmod(0o755)

    def install_registry(self):
        sectors, tables, routes = [], [], []
        for owner, ids in ((1, (2, -2, 11, -11)), (2, (2, -2, 11, -11, 21))):
            process = ['fixture amplitude', [[pdg, i >= 2, [], False, None]
                                           for i, pdg in enumerate(ids)]]
            sectors.append(dict(sector=owner, processes=[process]))
            tables.append(dict(sector=owner, histories=[dict(context=1, sector=owner,
                permutation=list(range(1, len(ids) + 1)))]))
            routes.append('I %d 1 %s\n' % (owner, ' '.join(map(str, ids))))
        registry = dict(outer=dict(context=1, orders=['QCD'], amp_orders=[[2]], sectors=sectors),
                        tables=tables, contexts=[{'context': 1}], flavour_complete=True)
        self.fingerprint = hashlib.sha256(json.dumps(registry, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        (self.context / 'mc_histories.json').write_text(json.dumps(dict(
            schema='macnlops-history-registry-v1', sha256=self.fingerprint, registry=registry)))
        (self.context / 'leshouche_info.dat').write_text(''.join(routes))

    def prepare(self, delta=True):
        self.delta = delta
        self.write_native_fixture()
        values = dict(macnlops=True, mcatnlo_delta=delta, macnlops_resolved_source=False,
            store_hard_momenta=True, parton_shower='PYTHIA8', event_norm='sum',
            pdlabel='lhapdf', lhaid=260000, lpp1=1, lpp2=1, ebeam1=6500., ebeam2=6500.,
            macnlops_flavour_count=5, reweight_scale=False, reweight_pdf=False, nevents=12)
        card = RunCardNLO()
        for key, value in values.items():
            card.set(key, [value] if key in ('lhaid', 'reweight_scale', 'reweight_pdf')
                     else value, user=True)
        card.check_validity()
        profile = mc.prepare_generation(self.process, card, self.pythia)
        self.assertEqual(profile['construction'], mc.DELTA_SOURCE if delta else mc.ORDINARY_SOURCE)
        self.assertEqual(card['macnlops_isr_cutoff'], .5)
        self.assertEqual(card['macnlops_beam_xmax1'], 1.)
        run_card = ''.join('%s = %s\n' % (value, key) for key, value in values.items())
        (self.cards / 'run_card.dat').write_text(run_card)
        self.header = HEADER.replace('<MGRunCard>sum = event_norm</MGRunCard>',
            '<MGRunCard>\n%s</MGRunCard>\n<slha>\n%s</slha>' % (
                run_card, (self.cards / 'param_card.dat').read_text()))
        self.inputs, self.source_blocks = [], {}
        for channel in (1, 2):
            blocks = []
            for sequence, state in enumerate(('unshowered', 'unshowered', 'unshowered',
                                             'emitted', 'emitted', 'exhausted'), 1):
                label = str(channel * 100 + sequence)
                kind = 'H' if state == 'unshowered' else 'S'
                weight = 3 if sequence == 1 else -2 if kind == 'H' else 2
                annotation = hard_annotation(kind)
                if delta:
                    annotation = annotation.replace('source="ordinary"', 'source="delta"')
                context = packet(self.fingerprint) if delta and kind == 'H' else ''
                original_scales = ('<scales scalup_1_2="100" scalup_2_1="100"/>\n'
                                   if delta else '')
                source = event(kind, weight, label).replace('</event>',
                    annotation + context + original_scales + '</event>')
                final = endpoint(state, delta).replace("id='1'", "id='%s'" % label)
                if weight == 3:
                    final = final.replace('%.12e' % -2, '%.12e' % weight)
                    final = final.replace('%.12e' % -3, '%.12e' % (1.5 * weight))
                final = final.replace('</event>', context + '</event>')
                self.native['events'][label] = dict(endpoint=final, metadata=dict(
                    event_kind=kind, emitted=state == 'emitted', source_weight=weight,
                    pt_evol=10. if state == 'emitted' else 0.,
                    beam_xmax=[1., 1.], shower_cutoffs=[.5, .5],
                    source_particles=[{'id': 2, 'status': -1}, {'id': -2, 'status': -1}]))
                self.source_blocks[label] = source
                blocks.append(source)
            path = self.context / ('channel_%d/events.lhe' % channel)
            path.parent.mkdir()
            path.write_text(self.header + ''.join(blocks) + '</LesHouchesEvents>\n')
            self.inputs.append(path)
        self.source = self.process / 'Events/run_01/events.lhe.gz'
        self.source.parent.mkdir(parents=True)
        self.write_sources()
        (self.process / 'SubProcesses/nevents_unweighted').write_text(''.join(
            '%s 6\n' % path.relative_to(self.process / 'SubProcesses') for path in self.inputs))
        self.write_native_fixture()

    def write_sources(self):
        with mc.open_lhe(self.source, 'wt') as stream:
            stream.write(self.header)
            for path in self.inputs:
                with mc.LHEStream(path) as channel:
                    stream.write(''.join(channel))
            stream.write('</LesHouchesEvents>\n')
        mc.write_source_manifest(self.process, 'run_01', [self.source] + self.inputs)

    def write_native_fixture(self):
        (self.root / 'native.json').write_text(json.dumps(self.native))

    def calls(self):
        return [json.loads(row) for row in (self.root / 'producer_calls.jsonl').read_text().splitlines()]

    def run_workflow(self):
        # Production dispatcher converts this to per-channel seeds 13 and 14.
        return mc.run_generated_sample(self.process, self.source, self.pythia, seed=12)

    def assert_completed_workflow(self):
        output = self.run_workflow()
        self.assertEqual(output, self.source.parent / 'macnlops/events.lhe.gz')
        with mc.LHEStream(output) as stream:
            blocks = list(stream)
            self.assertTrue(mc.matched_lhe_header(stream.header))
            card = mc.run_card_values(re.search(r'<MGRunCard>(.*?)</MGRunCard>',
                                               stream.header, re.S)[1])
        expected_labels = [str(channel * 100 + sequence) for channel in (1, 2)
                           for sequence in (1, 2, 5, 6)]
        self.assertCountEqual(blocks, [self.native['events'][label]['endpoint']
                                      for label in expected_labels])
        self.assertNotEqual(blocks, [self.native['events'][label]['endpoint']
                                    for label in expected_labels])
        self.assertEqual(card['nevents'], '8')
        self.assertEqual(card['macnlops'], 'true')
        collection = json.loads((output.parent / 'collection.json').read_text())
        self.assertEqual(collection['matching_seeds'], [13, 14])
        self.assertEqual(collection['shuffle_seed'], 13)
        self.assertEqual(collection['counts']['input_H'] + collection['counts']['input_S'], 12)
        self.assertEqual(collection['sumw'], 10.)
        parsed = [mc.Event(block) for block in blocks]
        self.assertEqual(sum(item.weight for item in parsed), 10.)
        self.assertEqual(sum(item.weight < 0 for item in parsed), 2)
        for item in parsed:
            label = re.search(r"<event id='(\d+)'>", item.block)[1]
            original = mc.Event(self.source_blocks[label])
            self.assertEqual(item.weight, original.weight)
            self.assertEqual(item.hard, original.hard)
            self.assertEqual(item.delta, original.delta)
            self.assertEqual(re.search(r'<rwgt>.*?</rwgt>', item.block, re.S)[0],
                             re.search(r'<rwgt>.*?</rwgt>', original.block, re.S)[0])
            self.assertEqual(item.source_scale, 100.)
            self.assertEqual(item.scale, {'emitted': 10., 'exhausted': .25,
                                         'unshowered': 100.}[item.continuation.state])
            if self.delta and item.kind == 'H':
                self.assertEqual(item.delta.registry, self.fingerprint)
                self.assertEqual(item.delta.probabilities, ((1, 1, 1, .25), (1, 1, 2, .75)))
            if self.delta:
                self.assertIn('<macnlops_source_scales ', item.block)
        reports = sorted(output.parent.glob('*/report.json'))
        self.assertEqual(len(reports), 2)
        for path in reports:
            report = json.loads(path.read_text())
            self.assertTrue(report['signed_residuals'])
            self.assertEqual(report['counts']['rejected_H'], 1)
            self.assertEqual(report['counts']['rejected_S'], 1)
            self.assertEqual(report['counts']['negative_accepted_H'], 1)
            decisions = [json.loads(line) for line in Path(report['decisions']).read_text().splitlines()]
            self.assertEqual([row['accepted'] for row in decisions], [True, True, False, False, True, True])
            self.assertEqual(decisions[1]['acceptance'], .75)
            if self.delta:
                self.assertEqual(len(report['delta_auxiliary']['owner_law_certificates']), 1)
        requests = [json.loads(line) for line in (self.context / 'requests.jsonl').read_text().splitlines()]
        points = [request for request in requests if request.startswith('MACNLOPS_POINT')]
        self.assertEqual(len(points), 8)
        if self.delta:
            self.assertEqual(sum('DELTA_REPLAY' in request for request in points), 4)
            self.assertEqual(sum('DELTA_SAMPLE' in request for request in points), 4)
            self.assertTrue(all(request.startswith('MACNLOPS_POINT_V2 ') for request in points))
        else:
            self.assertTrue(all('DELTA_' not in request for request in points))
            self.assertTrue(all(request.startswith('MACNLOPS_POINT_V1 ') for request in points))
        self.assertTrue(all(('--mcatnlo-delta' in call) == self.delta for call in self.calls()))
        self.assertEqual(sum('--input' in call for call in self.calls()), 2)
        self.assert_continuation_dispatch(output)

    def assert_continuation_dispatch(self, output):
        from madgraph.interface import amcatnlo_run_interface as interface
        class ReachedNormalShower(Exception):
            pass
        fake_banner = SimpleNamespace(run_card={'nevents': 1000}, get_detail=Mock(
            side_effect=lambda section, key: 'PYTHIA8' if key == 'parton_shower' else 8))
        command = SimpleNamespace(me_dir=str(self.process),
            options={'pythia8_path': str(self.pythia)}, get_randinit_seed=lambda: 12,
            shower_card={'nsplit_jobs': 3, 'nevents': -1},
            banner_to_mcatnlo=Mock(side_effect=ReachedNormalShower))
        with patch.object(mc, 'run_generated_sample') as matching, \
                patch.object(interface.misc, 'gunzip'), \
                patch.object(interface.banner_mod, 'Banner', return_value=fake_banner), \
                self.assertRaises(ReachedNormalShower):
            interface.aMCatNLOCmd.run_mcatnlo(command, str(output), {})
        matching.assert_not_called()
        command.banner_to_mcatnlo.assert_called_once_with(str(output))
        self.assertEqual(command.macnlops_continuation['events'], 8)
        self.assertEqual(fake_banner.run_card['nevents'], 8)
        self.assertEqual(command.shower_card['nsplit_jobs'], 1)
        self.assertEqual(command.shower_card['pdfcode'], 1)
        self.assertEqual(command.macnlops_continuation['shower_profile']['pdf'], 'LHAPDF6:fixture/0')

    def test_delta_generation_matching_collection_and_normal_shower_dispatch(self):
        self.prepare(delta=True)
        self.assert_completed_workflow()

    def test_ordinary_source_uses_same_multistep_workflow_without_delta_replay(self):
        self.prepare(delta=False)
        self.assert_completed_workflow()

    def assert_failed_channel(self, error):
        with self.assertRaisesRegex(mc.MatchingError, '1 of 2 source files failed'):
            self.run_workflow()
        output = self.source.parent / 'macnlops'
        self.assertFalse((output / 'events.lhe.gz').exists())
        self.assertFalse((output / 'collection.json').exists())
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual([item['status'] for item in summary], ['complete', 'failed'])
        self.assertTrue(Path(summary[0]['output']).is_file())
        self.assertIn(error, summary[1]['error'])
        failures = list(output.glob('*/failure.json'))
        self.assertEqual(len(failures), 1)
        self.assertFalse((failures[0].parent / 'events.lhe.gz').exists())

    def test_missing_h_replay_packet_prevents_channel_processing_and_collection(self):
        self.prepare()
        path = self.inputs[1]
        path.write_text(re.sub(r'<macnlops_delta_context\b.*?</macnlops_delta_context>\n',
                               '', path.read_text(), count=1, flags=re.S))
        self.write_sources()
        self.assert_failed_channel('conditional Delta replay context')
        self.assertEqual(sum('--input' in call for call in self.calls()), 1)

    def test_wrong_source_mode_is_rejected_before_channel_processing(self):
        self.prepare()
        path = self.inputs[1]
        # Replace one S source, which has no H replay packet of its own.
        block = self.source_blocks['204']
        path.write_text(path.read_text().replace(block, block.replace('source="delta"', 'source="ordinary"')))
        self.write_sources()
        self.assert_failed_channel('different counterterm construction')
        self.assertEqual(sum('--input' in call for call in self.calls()), 1)

    def test_native_failure_after_successful_channel_never_publishes_combined_sample(self):
        self.prepare()
        self.native['fail_labels'] = ['201']
        self.write_native_fixture()
        self.assert_failed_channel('exit status 17')
        self.assertEqual(sum('--input' in call for call in self.calls()), 2)


if __name__ == '__main__':
    unittest.main()
