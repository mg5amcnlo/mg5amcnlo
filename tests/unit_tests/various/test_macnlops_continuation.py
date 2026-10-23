"""First-emission LHE hand-off contracts; no native build or shower."""
from collections import Counter
import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from madgraph.various import macnlops as mc
from tests.unit_tests.various.test_macnlops import HEADER, event, hard_annotation


def endpoint(state='emitted', delta=False):
    kind = 'H' if state == 'unshowered' else 'S'
    scale = {'emitted': 10., 'exhausted': .25, 'unshowered': 100.}[state]
    block = event(kind, weight=-2 if kind == 'H' else 2)
    block = block.replace('2.000000000000e+00 100 ', '2.000000000000e+00 %.17g ' % scale)
    if state == 'emitted':
        block = block.replace('\n4 1 ', '\n5 1 ')
        momentum = math.sqrt(8000.)
        block = block.replace('11 1 1 2 0 0 100 0 0 100 0 0 9',
                              '11 1 1 2 0 0 -10 %.17g 0 90 0 0 9' % momentum)
        block = block.replace('-11 1 1 2 0 0 -100 0 0 100 0 0 9',
                              '-11 1 1 2 0 0 -10 %.17g 0 90 0 0 9' % -momentum)
        block = block.replace('#aMCatNLO', '21 1 1 2 501 502 20 0 0 20 0 0 9\n#aMCatNLO')
    annotation = hard_annotation(kind)
    if delta:
        annotation = annotation.replace('source="ordinary"', 'source="delta"')
    nlegs = 5 if state == 'emitted' else 4
    scales = ' '.join('scalup_%d="%.17g"' % (index, scale if index != 1 else .7 * scale)
                      for index in range(1, nlegs + 1))
    if state == 'emitted':
        scales += ' scalup_5_1="3.5" scalup_5_2="10"'
    original = ('<macnlops_source_scales scalup_1_2="100" scalup_2_1="100">'
                '</macnlops_source_scales>\n') if delta else ''
    marker = ('<macnlops_continuation version="1" state="%s" scale="%.17g" '
              'source_scalup="100" cutoff_isr="0.5" cutoff_fsr="0.5"/>\n') % (state, scale)
    origin = 'shower' if state == 'emitted' else 'hard_annotation'
    extra = (annotation + '<macnlops_endpoint version="1" origin="%s" stage="first"/>\n' % origin +
             marker + original + '<scales %s>\n</scales>\n' % scales)
    return block.replace('</event>', extra + '</event>')


class KeepEverything:
    configuration = None

    def configure(self, bounds, cutoffs):
        self.configuration = bounds, cutoffs

    def evaluate(self, event, sequence):
        return mc.Densities(1., 3., 0.) if event.kind == 'H' else mc.Densities(2., 1., 1.)


class TestContinuation(unittest.TestCase):
    def fixture(self, root, delta=False):
        blocks = [endpoint(state, delta) for state in ('emitted', 'exhausted', 'unshowered')]
        source = root / 'proposals.lhe'
        # Both the run-card event target and source flag are operational here:
        # the accepted file must acquire its own count and continuation stage.
        header = HEADER.replace('sum = event_norm',
            'sum = event_norm\nTrue = macnlops\n1000 = nevents')
        source.write_text(header + ''.join(blocks) + '</LesHouchesEvents>\n')
        metadata = root / 'metadata.jsonl'
        import json
        rows = [dict(sequence=index, event_kind=mc.Event(block).kind,
                     emitted=index == 1, pt_evol=10. if index == 1 else 0.)
                for index, block in enumerate(blocks, 1)]
        metadata.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        output = root / 'accepted.lhe.gz'
        report = mc.reject_proposals(source, metadata, output, KeepEverything(),
                                     signed_residuals=True, require_continuation=True)
        return output, report, blocks

    def test_current_starts_and_source_provenance_are_distinct_in_both_modes(self):
        for delta in (False, True):
            for state in ('emitted', 'exhausted', 'unshowered'):
                with self.subTest(delta=delta, state=state):
                    parsed = mc.Event(endpoint(state, delta))
                    self.assertEqual(parsed.continuation.state, state)
                    self.assertEqual(parsed.source_scale, 100.)
                    self.assertEqual(parsed.scale, dict(emitted=10., exhausted=.25,
                                                       unshowered=100.)[state])

    def test_bad_active_caps_or_state_cannot_silently_restart(self):
        base = endpoint()
        bad = [base.replace('scalup_5_1="3.5"', 'scalup_5_1="100"'),
               base.replace('scalup_1="7"', 'scalup_1="0"'),
               base.replace('scalup_1="7"', 'scalup_1="nan"'),
               base.replace('scalup_1="7"', ''),
               base.replace('scalup_5_1="3.5"', 'scalup_8_1="3.5"'),
               base.replace('scalup_5_1="3.5"', 'scalup_5_5="3.5"'),
               base.replace('scalup_5_1="3.5"', 'scalup_5_1="3.5" scalup_5_1="3.5"'),
               base.replace('state="emitted"', 'state="unshowered"'),
               base.replace('scale="10"', 'scale="11"'),
               base.replace('source_scalup="100"', 'source_scalup="inf"'),
               base.replace('cutoff_isr="0.5"', 'cutoff_isr="0"'),
               endpoint('exhausted').replace('cutoff_isr="0.5"', 'cutoff_isr="0.2"'),
               base.replace('<scales ', '<scales/><scales ')]
        for document in bad:
            with self.subTest(document=document), self.assertRaises(mc.MatchingError):
                mc.Event(document)

    def test_unshowered_delta_zero_pair_retains_stock_ignore_zero_semantics(self):
        block = endpoint('unshowered', delta=True).replace(
            '<scales ', '<scales scalup_1_2="0" scalup_2_1="80" ')
        parsed = mc.Event(block)
        self.assertEqual(parsed.continuation_dipole_scales[(1, 2)], 0.)
        self.assertEqual(parsed.scale, 100.)
        # A zero production anchor itself would trigger the stock fallback.
        with self.assertRaises(mc.MatchingError):
            mc.Event(block.replace('scalup_1="70"', 'scalup_1="0"'))

    def test_veto_and_randomized_collection_preserve_all_active_starts(self):
        for delta in (False, True):
            with self.subTest(delta=delta), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                accepted, report, expected = self.fixture(root, delta)
                with mc.LHEStream(accepted) as stream:
                    self.assertTrue(mc.matched_lhe_header(stream.header))
                    self.assertEqual(mc.run_card_values(stream.header.split('<MGRunCard>')[1].split(
                        '</MGRunCard>')[0])['nevents'], '3')
                    self.assertEqual(list(stream), expected)
                self.assertEqual(report['counts']['accepted_S'], 2)
                self.assertEqual(report['counts']['accepted_H'], 1)
                collected = root / 'collected/events.lhe.gz'
                mc.collect_matched_events([accepted], [report], collected, 61)
                with mc.LHEStream(collected) as stream:
                    self.assertTrue(mc.matched_lhe_header(stream.header))
                    self.assertEqual(Counter(stream), Counter(expected))
                contract = mc.continuation_contract(collected)
                self.assertEqual(contract['events'], 3)
                self.assertEqual(contract['shower_cutoffs'], [.5, .5])
                self.assertEqual(contract['counts']['exhausted'], 1)

    def test_old_endpoint_files_are_not_claimed_as_corrected_continuations(self):
        self.assertFalse(mc.matched_lhe_header(HEADER))
        with self.assertRaisesRegex(mc.MatchingError, 'older matched LHE'):
            mc.matched_lhe_header(HEADER.replace('</header>',
                '<macnlops normalization="sum" stage="first-emission">x</macnlops></header>'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal, metadata, output = (root / name for name in ('p.lhe', 'm.jsonl', 'out.lhe'))
            proposal.write_text(HEADER + event('S') + '</LesHouchesEvents>\n')
            metadata.write_text('{"sequence":1,"event_kind":"S","emitted":false}\n')
            with self.assertRaisesRegex(mc.MatchingError, 'lacks shower-continuation scales'):
                mc.reject_proposals(proposal, metadata, output, KeepEverything(), require_continuation=True)
            self.assertFalse(output.exists())

    def test_normal_shower_dispatch_does_not_repeat_first_emission_or_veto(self):
        from madgraph.interface import amcatnlo_run_interface as interface
        class StageTwoEntered(Exception):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted, _, _ = self.fixture(root)
            fake = SimpleNamespace(me_dir=str(root), options={'pythia8_path': '/unused'},
                                   get_randinit_seed=lambda: 11)
            with patch.object(interface.macnlops, 'run_generated_sample') as first_step, \
                    patch.object(interface.misc, 'gunzip'), \
                    patch.object(interface.banner_mod, 'Banner', side_effect=StageTwoEntered), \
                    self.assertRaises(StageTwoEntered):
                interface.aMCatNLOCmd.run_mcatnlo(fake, str(accepted), {})
            first_step.assert_not_called()
            self.assertEqual(fake.macnlops_continuation['events'], 3)

    def test_nested_collected_sample_keeps_the_source_run_identifier(self):
        from madgraph.interface import amcatnlo_run_interface as interface
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted, _, _ = self.fixture(root)
            directory = root / 'Events/run_01/macnlops'
            directory.mkdir(parents=True)
            (directory / 'events.lhe.gz').write_bytes(accepted.read_bytes())
            for name in ('run_01/macnlops', './run_01/macnlops', str(directory)):
                with self.subTest(path=name):
                    fake = SimpleNamespace(me_dir=str(root), InvalidCmd=ValueError,
                                           set_run_name=Mock())
                    args = [name]
                    interface.aMCatNLOCmd.check_shower(fake, args, {'force': False})
                    fake.set_run_name.assert_called_once_with('run_01', level='shower')
                    self.assertEqual(Path(args[0]), directory)

    def test_normal_consumer_settings_read_current_caps_with_local_recoil(self):
        template = Path(__file__).resolve().parents[3] / (
            'Template/NLO/MCatNLO/Scripts/MCatNLO_MadFKS_PYTHIA8.Script')
        text = template.read_text()
        block = text[text.index('if [ "${MACNLOPS_CONTINUATION:-OFF}"'):text.index(
            'cp -f $thisdir/$ifile Pythia8.cmd')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'xmldoc').mkdir()
            parameter = root / 'xmldoc/BeamParameters.xml'
            parameter.write_text('<flag name="Beams:setDipoleShowerStartingScalesFromLHEF"/>')
            script, output = root / 'settings.sh', root / 'pythia.cmnd'
            script.write_text(block)
            environment = dict(os.environ, MACNLOPS_CONTINUATION='ON', thisdir=str(root),
                               ifile=str(output), MACNLOPS_ISR_CUTOFF='0.5', MACNLOPS_FSR_CUTOFF='0.7')
            output.write_text('TimeShower:globalRecoil = on\nHadronLevel:all = on\n')
            result = subprocess.run(['bash', str(script)], env=environment,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            settings = dict(line.split('=', 1) for line in output.read_text().splitlines()
                            if '=' in line and not line.startswith('!'))
            settings = {key.strip(): value.strip() for key, value in settings.items()}
            for key in ('Beams:setProductionScalesFromLHEF',
                        'Beams:setDipoleShowerStartingScalesFromLHEF', 'Main:spareFlag1'):
                self.assertEqual(settings[key], 'on')
            for key in ('TimeShower:globalRecoil', 'HadronLevel:all', 'PartonLevel:MPI'):
                self.assertEqual(settings[key], 'off')
            self.assertEqual(settings['SpaceShower:pTmin'], '0.5')
            self.assertEqual(settings['TimeShower:pTmin'], '0.7')
            self.assertEqual(settings['LesHouches:setQuarkMass'], '0')
            previous = output.read_bytes()
            environment['MACNLOPS_CONTINUATION'] = 'OFF'
            self.assertEqual(subprocess.run(['bash', str(script)], env=environment,
                                           capture_output=True).returncode, 0)
            self.assertEqual(output.read_bytes(), previous)
            environment['MACNLOPS_CONTINUATION'] = 'ON'
            parameter.unlink()
            self.assertNotEqual(subprocess.run(['bash', str(script)], env=environment,
                                              capture_output=True).returncode, 0)

    def test_failed_continuation_cannot_publish_a_partial_shower_file(self):
        template = Path(__file__).resolve().parents[3] / 'Template/NLO/MCatNLO/shower_template.sh'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / 'shower.sh'
            script.write_text(template.read_text() % dict(ld_library_path='LD_LIBRARY_PATH', extralibs=''))
            executable = root / 'Pythia8.exe'
            executable.write_text('#!/usr/bin/env bash\nexit 1\n')
            executable.chmod(0o755)
            partial = root / 'Pythia8.hep'
            partial.write_text('partial event output')
            result = subprocess.run(['bash', str(script), 'PYTHIA8', 'HEP', 'run_01'],
                                    cwd=root, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(partial.exists())
            self.assertFalse((root / 'events.hepmc.gz').exists())


if __name__ == '__main__':
    unittest.main()
