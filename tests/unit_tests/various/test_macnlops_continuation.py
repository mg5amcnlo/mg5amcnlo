"""First-emission LHE hand-off contracts; no native build or shower."""
from collections import Counter
import copy
import math
import os
from pathlib import Path
import subprocess
import tempfile
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from madgraph.various import macnlops as mc
from tests.unit_tests.various.test_macnlops import HEADER, event, hard_annotation


def shower_profile(delta=False):
    return dict(schema='macnlops-generation-profile-v1', shower_backend='stock-pythia8',
        construction=mc.DELTA_SOURCE if delta else mc.ORDINARY_SOURCE,
        alpha_s_mz=.120179, alpha_s_order=2, pdf='LHAPDF6:fixture/0',
        lhapdf_library_sha256=None,
        pdf_info_sha256='a' * 64, pdf_member_sha256='b' * 64,
        shower_libraries={'libpythia8.so': 'c' * 64},
        support=dict(alpha_s_nfmax=5, n_quark_in=4, n_gluon_to_quark=4,
            shower_cutoffs=[.5, .5], particle_masses={'1': 0., '4': 0., '5': 4.7, '21': 0.},
            qcd_only_off_switches={'PartonLevel:MPI': False, 'PartonLevel:Remnants': False}),
        effective_shower_settings={
            'SpaceShower:alphaSvalue': '0.12018', 'TimeShower:alphaSvalue': '0.12018',
            'SpaceShower:alphaSorder': '2', 'TimeShower:alphaSorder': '2',
            'SpaceShower:nQuarkIn': '4', 'TimeShower:nGluonToQuark': '4',
            'StandardModel:alphaSnfmax': '5', 'SpaceShower:pT0Ref': '0.',
            'PDF:pSet': 'LHAPDF6:fixture/0', 'TimeShower:globalRecoil': 'on'})


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
        if delta:
            blocks[-1] = blocks[-1].replace('source_scalup="100"', 'source_scalup="-1"')
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
                                     signed_residuals=True, require_continuation=True,
                                     shower_profile=shower_profile(delta))
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

    def test_delta_h_source_sentinel_is_inert_provenance(self):
        block = endpoint('unshowered', delta=True).replace(
            'source_scalup="100"', 'source_scalup="-1"')
        parsed = mc.Event(block)
        self.assertEqual(parsed.source_scale, -1.)
        self.assertEqual(parsed.scale, 100.)
        self.assertTrue(all(value > 0 for value in parsed.continuation_leg_scales.values()))
        parsed.check_proposal_metadata(dict(event_kind='H', emitted=False,
            source_scalup=-1., continuation_scalup=100.), require_continuation=True)
        invalid = [block.replace('source="delta"', 'source="ordinary"'),
                   block.replace('source_scalup="-1"', 'source_scalup="-2"'),
                   block.replace('scale="100" source_scalup', 'scale="-1" source_scalup')]
        invalid.extend(endpoint(state, delta=True).replace(
            'source_scalup="100"', 'source_scalup="-1"') for state in ('emitted', 'exhausted'))
        for document in invalid:
            with self.subTest(document=document), self.assertRaises(mc.MatchingError):
                mc.Event(document)

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
                self.assertEqual(contract['shower_profile'], shower_profile(delta))

    def test_portable_profile_preserves_exact_coupling_masses_and_flavours(self):
        profile = shower_profile()
        header = mc.with_shower_profile(HEADER, profile)
        self.assertEqual(mc.read_shower_profile(header), profile)
        settings = dict(line.split(' = ', 1) for line in mc.continuation_physics(profile).splitlines()
                        if not line.startswith('!'))
        self.assertEqual(float(settings['SpaceShower:alphaSvalue']), .120179)
        self.assertEqual(float(settings['TimeShower:alphaSvalue']), .120179)
        self.assertEqual(settings['SpaceShower:alphaSorder'], '2')
        self.assertEqual(settings['SpaceShower:nQuarkIn'], '4')
        self.assertEqual(settings['TimeShower:nGluonToQuark'], '4')
        self.assertEqual(float(settings['5:m0']), 4.7)
        self.assertNotIn('SpaceShower:pT0Ref', settings)
        with self.assertRaisesRegex(mc.MatchingError, 'fingerprint'):
            mc.read_shower_profile(header.replace('0.120179', '0.118'))
        with self.assertRaisesRegex(mc.MatchingError, 'duplicate'):
            mc.read_shower_profile(header + header)
        for key, value in (('SpaceShower:pT0Ref', '.5'), ('SpaceShower:nQuarkIn', '5'),
                           ('TimeShower:alphaSorder', '1'), ('PDF:pSet', '13')):
            invalid = copy.deepcopy(profile)
            invalid['effective_shower_settings'][key] = value
            with self.subTest(key=key), self.assertRaises(mc.MatchingError):
                mc.with_shower_profile(HEADER, invalid)

    def test_profile_identity_is_required_across_collection_and_for_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted, report, blocks = self.fixture(root)
            with mc.LHEStream(accepted) as stream:
                header = re.sub(r'<macnlops_shower_profile\b.*?</macnlops_shower_profile>',
                                '', stream.header, flags=re.S)
            legacy = root / 'legacy.lhe'
            legacy.write_text(header + ''.join(blocks) + '</LesHouchesEvents>\n')
            with self.assertRaisesRegex(mc.MatchingError, 'lacks its measured shower profile'):
                mc.continuation_contract(legacy)
            with self.assertRaisesRegex(mc.MatchingError, 'different first-emission shower profiles'):
                mc.collect_matched_events([accepted, legacy], [report, report], root / 'mixed.lhe', 3)
            changed = shower_profile()
            changed['support']['particle_masses']['5'] = 4.8
            legacy.write_text(mc.with_shower_profile(header, changed) + ''.join(blocks) +
                              '</LesHouchesEvents>\n')
            with self.assertRaisesRegex(mc.MatchingError, 'different first-emission shower profiles'):
                mc.collect_matched_events([accepted, legacy], [report, report], root / 'mixed.lhe', 3)

    def test_native_snapshot_keeps_physics_without_paths_or_random_seeds(self):
        profile = shower_profile()
        settings = profile.pop('effective_shower_settings')
        profile['native_build'] = {'path': '/old/machine/native.build.json', 'sha256': 'a' * 64}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'effective.cmnd'
            path.write_text('! Complete native settings dump\nRandom:seed = 73\n'
                'Beams:LHEF = /old/machine/input.lhe\n' +
                ''.join('%s = %s\n' % item for item in settings.items()))
            recorded = mc.recorded_shower_profile(profile, path)
        self.assertNotIn('native_build', recorded)
        self.assertEqual(recorded['effective_shower_settings'], settings)
        self.assertEqual(recorded['alpha_s_mz'], .120179)

    def test_continuation_checks_original_pdf_and_libraries_without_reading_current_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted, _, _ = self.fixture(root)
            # Resolve the source PDF from the accepted banner, even when the
            # process Cards and generation profile have subsequently changed.
            with mc.open_lhe(accepted, 'rt') as stream:
                text = stream.read().replace('sum = event_norm',
                    'sum = event_norm\nlhapdf = pdlabel\n260000 = lhaid')
            with mc.open_lhe(accepted, 'wt') as stream:
                stream.write(text)
            info = root / 'fixture.info'
            info.write_text('original PDF metadata')
            member = root / 'fixture_0000.dat'
            member.write_text('original PDF member')
            backend = root / 'native/libLHAPDF.so'
            backend.parent.mkdir()
            backend.write_text('original native LHAPDF library')
            profile = shower_profile()
            profile['pdf_info_sha256'] = mc.sha256(info)
            profile['pdf_member_sha256'] = mc.sha256(member)
            profile['lhapdf_library_sha256'] = mc.sha256(backend)
            contract = dict(shower_profile=profile)
            (root / 'Cards').mkdir()
            (root / 'Cards/run_card.dat').write_text('1 = lhaid\ninternal = pdlabel\n')
            (root / 'Cards/macnlops_profile.json').write_text('{}')
            output = root / 'physics.cmnd'
            def original_pdf(process, card, with_info=False):
                self.assertEqual(card['lhaid'], '260000')
                self.assertEqual(card['pdlabel'], 'lhapdf')
                return profile['pdf'], info
            with patch.object(mc, 'pdf_name', side_effect=original_pdf), \
                    patch.object(mc, 'shower_library_hashes', return_value=profile['shower_libraries']), \
                    patch.object(mc, 'shower_environment', return_value={'LD_LIBRARY_PATH': str(backend.parent)}):
                environment = mc.write_continuation_settings(root, '/pythia', accepted, contract, output)
                self.assertEqual(environment, {'LD_LIBRARY_PATH': str(backend.parent)})
                actual = re.search(r'^SpaceShower:alphaSvalue = (.*)$', output.read_text(), re.M)
                self.assertEqual(float(actual[1]), .120179)
                backend.write_text('different native LHAPDF library')
                with self.assertRaisesRegex(mc.MatchingError, 'Continuation LHAPDF library differs'):
                    mc.write_continuation_settings(root, '/pythia', accepted, contract, output)
                backend.write_text('original native LHAPDF library')
                member.write_text('changed PDF member')
                with self.assertRaisesRegex(mc.MatchingError, 'Continuation PDF differs'):
                    mc.write_continuation_settings(root, '/pythia', accepted, contract, output)
            with patch.object(mc, 'shower_library_hashes', return_value={'libpythia8.so': 'd' * 64}), \
                    self.assertRaisesRegex(mc.MatchingError, 'Continuation Pythia libraries differ'):
                mc.write_continuation_settings(root, '/pythia', accepted, contract, output)

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
            physics = root / 'physics.cmnd'
            physics.write_text(mc.continuation_physics(shower_profile()))
            environment = dict(os.environ, MACNLOPS_CONTINUATION='ON', thisdir=str(root),
                               ifile=str(output), MACNLOPS_ISR_CUTOFF='0.5', MACNLOPS_FSR_CUTOFF='0.7',
                               MACNLOPS_PHYSICS_FILE=physics.name)
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
            self.assertEqual(float(settings['SpaceShower:alphaSvalue']), .120179)
            self.assertEqual(settings['TimeShower:alphaSorder'], '2')
            self.assertEqual(settings['SpaceShower:nQuarkIn'], '4')
            self.assertEqual(settings['TimeShower:nGluonToQuark'], '4')
            self.assertEqual(float(settings['5:m0']), 4.7)
            self.assertEqual(settings['PartonLevel:Remnants'], 'on')
            previous = output.read_bytes()
            environment['MACNLOPS_CONTINUATION'] = 'OFF'
            self.assertEqual(subprocess.run(['bash', str(script)], env=environment,
                                           capture_output=True).returncode, 0)
            self.assertEqual(output.read_bytes(), previous)
            environment['MACNLOPS_CONTINUATION'] = 'ON'
            physics.unlink()
            self.assertNotEqual(subprocess.run(['bash', str(script)], env=environment,
                                              capture_output=True).returncode, 0)
            physics.write_text(mc.continuation_physics(shower_profile()))
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
