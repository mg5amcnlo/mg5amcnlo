"""Compiled first-emission shower checks (opt in with PYTHIA8=/prefix).

These exercise Pythia's accepted first branching, source-event accounting,
the Sudakov no-emission outcome, and exact H pass-through. They do not
certify the separate analytic shower-density evaluator.
"""

import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.various.test_macnlops_hard_presentation import (
    active_scales, assert_handoff, lhe_particles)


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'Template/NLO/MCatNLO/srcPythia8/MacNLOPS.cc'


def event(kind, scale, weight=1.0, sequence=0):
    return """<event>
4 1 {weight} {scale} 0.0072973525693 0.118
2 -1 0 0 501 0 0 0 100 100 0 0 9
-2 -1 0 0 0 501 0 0 -100 100 0 0 9
11 1 1 2 0 0 100 0 0 100 0 0 9
-11 1 1 2 0 0 -100 0 0 100 0 0 9
#aMCatNLO {kind} 3 1 1 2 {scale} 0 -9 4 0 0 0 0 0 0
<rwgt><wgt id="nominal">{weight}</wgt></rwgt>
# preserved-source-{sequence}
</event>
""".format(kind=kind, scale=scale, weight=weight, sequence=sequence)


def blocks(path):
    return re.findall(r'<event(?:\s[^>]*)?>.*?</event>',
                      path.read_text(), re.S)


class TestMacNLOPSShower(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix = os.environ.get('PYTHIA8') or os.environ.get('PYTHIA8LOCATION')
        if not prefix:
            raise unittest.SkipTest('Set PYTHIA8 to compile Pythia integration checks')
        cls.prefix = Path(prefix).resolve()
        compiler = shutil.which(os.environ.get('CXX', 'g++'))
        if not compiler:
            raise unittest.SkipTest('A C++ compiler is required')
        if not (cls.prefix / 'include/Pythia8/Pythia.h').is_file():
            raise unittest.SkipTest('PYTHIA8 does not contain Pythia headers')
        cls.tmp = tempfile.TemporaryDirectory(prefix='macnlops_shower_')
        cls.work = Path(cls.tmp.name)
        cls.exe = cls.work / 'MacNLOPS.exe'
        command = [compiler, '-std=c++11', '-O1',
                   '-I' + str(cls.prefix / 'include'), str(SOURCE),
                   '-o', str(cls.exe), '-L' + str(cls.prefix / 'lib'),
                   '-Wl,-rpath,' + str(cls.prefix / 'lib'), '-lpythia8']
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            cls.tmp.cleanup()
            raise RuntimeError('One-step driver compilation failed:\n' +
                               compiled.stdout + compiled.stderr)
        cls.input = cls.work / 'input.lhe'
        cls.source_events = [event(1, 100., sequence=i) for i in range(12)]
        cls.source_events.insert(3, event(1, 0.01, sequence=12))
        cls.source_events.insert(7, event(2, 100., weight=-1., sequence=13))
        cls.input.write_text("""<LesHouchesEvents version="3.0">
<header><generator>MAcNLOPS first-emission integration fixture</generator></header>
<init>
2212 2212 6500 6500 0 0 0 0 -4 1
1 0 1 1
</init>
""" + ''.join(cls.source_events) + '</LesHouchesEvents>\n')
        cls.env = os.environ.copy()
        cls.pdf = cls.env.get('MACNLOPS_TEST_PDF', 'LHAPDF6:MSTW2008nlo90cl/0')
        cls.output, cls.metadata = cls.run_driver('first')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def run_driver(cls, label):
        output = cls.work / (label + '.lhe')
        metadata = cls.work / (label + '.jsonl')
        command = [str(cls.exe), '--input', str(cls.input),
                   '--output', str(output), '--metadata', str(metadata),
                   '--pdf', cls.pdf, '--seed', '7349']
        result = subprocess.run(command, capture_output=True, text=True,
                                env=cls.env)
        if result.returncode:
            raise RuntimeError('One-step shower failed:\n' +
                               result.stdout + result.stderr)
        return output, metadata

    def test_source_accounting_and_h_pass_through(self):
        records = [json.loads(line) for line in self.metadata.read_text().splitlines()]
        output_events = blocks(self.output)
        self.assertEqual(len(records), len(self.source_events))
        self.assertEqual(len(output_events), len(self.source_events))
        self.assertEqual([r['sequence'] for r in records],
                         list(range(1, len(records) + 1)))
        hard_index = next(i for i, r in enumerate(records)
                          if r['event_kind'] == 'H')
        self.assertEqual(lhe_particles(output_events[hard_index]),
                         lhe_particles(self.source_events[hard_index]))
        assert_handoff(self, self.source_events[hard_index], output_events[hard_index], 'unshowered')
        self.assertFalse(records[hard_index]['emitted'])
        self.assertEqual(records[hard_index]['source_weight'], -1.)
        self.assertTrue(self.output.read_text().rstrip().endswith(
                        '</LesHouchesEvents>'))

    def test_no_emission_is_preserved(self):
        records = [json.loads(line) for line in self.metadata.read_text().splitlines()]
        index = next(i for i, r in enumerate(records) if r['source_scalup'] < 0.1)
        self.assertEqual(records[index]['event_kind'], 'S')
        self.assertFalse(records[index]['emitted'])
        self.assertEqual(records[index]['emission_type'], 'none')
        self.assertEqual(lhe_particles(blocks(self.output)[index]),
                         lhe_particles(self.source_events[index]))
        marker = assert_handoff(self, self.source_events[index], blocks(self.output)[index], 'exhausted')
        self.assertEqual(float(marker['scale']), records[index]['continuation_scalup'])
        self.assertEqual(records[index]['continuation_cap_source'], 'effective-shower-cutoffs')

    def test_one_emission_kinematics_and_metadata(self):
        records = [json.loads(line) for line in self.metadata.read_text().splitlines()]
        emitted = [r for r in records if r['emitted']]
        self.assertTrue(emitted)
        for record in emitted:
            self.assertEqual(record['event_kind'], 'S')
            self.assertEqual(record['emission_type'], 'ISR')
            self.assertGreater(record['pt_evol'], 0.)
            self.assertLessEqual(record['pt_evol'], record['source_scalup'])
            endpoint = blocks(self.output)[record['sequence']-1]
            marker = assert_handoff(self, self.source_events[record['sequence']-1], endpoint, 'emitted')
            self.assertEqual(float(marker['scale']), record['pt_evol'])
            self.assertEqual(record['continuation_cap_source'], 'stock-pythia-8.318-public-dipole-list')
            for cap in active_scales(endpoint).values():
                self.assertLessEqual(cap, record['pt_evol'])
            particles = record['particles']
            self.assertEqual(len(particles), len(record['source_particles']) + 1)
            self.assertEqual(sum(p['status'] == -1 for p in particles), 2)
            self.assertEqual(len(record['hard_pythia_indices']), len(particles))
            for component in range(4):
                imbalance = sum(p['p'][component] *
                                (-1 if p['status'] == -1 else 1)
                                for p in particles)
                self.assertLess(abs(imbalance), 1.e-7)
            for particle in particles:
                energy, px, py, pz = particle['p']
                self.assertAlmostEqual(energy**2-px**2-py**2-pz**2,
                                       particle['mass']**2, delta=1.e-6)
            source_extra = self.source_events[record['sequence'] - 1].split(
                           '#aMCatNLO', 1)[1]
            self.assertEqual(blocks(self.output)[record['sequence']-1].split(
                             '#aMCatNLO', 1)[1], source_extra.rstrip())

    def test_fixed_seed_reproducibility(self):
        output, metadata = self.run_driver('repeat')
        self.assertEqual(self.output.read_bytes(), output.read_bytes())
        self.assertEqual(self.metadata.read_bytes(), metadata.read_bytes())

    def test_fsr_competes_with_isr(self):
        # A known hard-model point, rather than ordinary MC-mass-shuffled
        # LHE records whose pre-shuffle coordinates cannot be recovered.
        import math
        momentum = math.sqrt(250.**2 - 173.**2)
        source = event(1, 200.).replace('100 100 0 0 9', '250 250 0 0 9')
        source = source.replace('-100 100 0 0 9', '-250 250 0 0 9')
        source = source.replace('-2 -1 0 0 0 501', '-2 -1 0 0 0 502')
        source = source.replace('11 1 1 2 0 0 100 0 0 100 0 0 9',
            '6 1 1 2 501 0 %.17g 0 0 250 173 0 9' % momentum)
        source = source.replace('-11 1 1 2 0 0 -100 0 0 100 0 0 9',
            '-6 1 1 2 0 502 %.17g 0 0 250 173 0 9' % -momentum)
        fixture = self.work / 'ttbar.lhe'
        header = self.input.read_text().split('<event>', 1)[0]
        fixture.write_text(header + source * 100 + '</LesHouchesEvents>\n')
        settings = self.work / 'ttbar.cmnd'
        settings.write_text('6:m0 = 173.\n')
        output, metadata = self.work / 'ttbar_out.lhe', self.work / 'ttbar.jsonl'
        result = subprocess.run([str(self.exe), '--input', str(fixture),
            '--output', str(output), '--metadata', str(metadata),
            '--pdf', self.pdf, '--seed', '1', '--settings', str(settings)], capture_output=True, text=True,
            env=self.env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [json.loads(line) for line in metadata.read_text().splitlines()]
        self.assertEqual(len(records), 100)
        self.assertEqual({r['emission_type'] for r in records if r['emitted']},
                         {'ISR', 'FSR'})
        for record in records:
            if not record['emitted']:
                continue
            history = {p['index']: p for p in record['pythia_event']}
            branch = record['branch_indices']
            self.assertIn(branch['radiator_before'], history)
            self.assertIn(branch['radiator_after'], history)
            self.assertIn(branch['emitted'], record['hard_pythia_indices'])

    def test_incompatible_polarization_setting_fails(self):
        settings = self.work / 'incompatible.cmnd'
        settings.write_text('SpaceShower:phiPolAsymHard = on\n')
        result = subprocess.run([str(self.exe), '--input', str(self.input),
            '--output', str(self.work / 'invalid.lhe'),
            '--metadata', str(self.work / 'invalid.jsonl'),
            '--settings', str(settings), '--pdf', self.pdf],
            capture_output=True, text=True, env=self.env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Incompatible scalar shower setting: '
                      'SpaceShower:phiPolAsymHard', result.stderr)

    def test_incompatible_resonance_qed_setting_fails(self):
        # Pythia separately enables QED radiation from undecayed resonances,
        # including top quarks, through QEDshowerByOther.
        settings = self.work / 'incompatible_resonance_qed.cmnd'
        settings.write_text('TimeShower:QEDshowerByOther = on\n')
        result = subprocess.run([str(self.exe), '--input', str(self.input),
            '--output', str(self.work / 'invalid_resonance_qed.lhe'),
            '--metadata', str(self.work / 'invalid_resonance_qed.jsonl'),
            '--settings', str(settings), '--pdf', self.pdf],
            capture_output=True, text=True, env=self.env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Incompatible scalar shower setting: '
                      'TimeShower:QEDshowerByOther', result.stderr)

    def test_incompatible_qcd_support_and_recoil_settings_fail(self):
        for setting, value in [('SpaceShower:QCDshower', 'off'),
                               ('TimeShower:QCDshower', 'off'),
                               ('TimeShower:allowBeamRecoil', 'off'),
                               ('TimeShower:nMaxGlobalRecoil', '1'),
                               ('TimeShower:alphaSmax', '0.5'),
                               ('Tune:pp', '14')]:
            with self.subTest(setting=setting):
                settings = self.work / 'incompatible_qcd.cmnd'
                settings.write_text('%s = %s\n' % (setting, value))
                result = subprocess.run([str(self.exe), '--input', str(self.input),
                    '--output', str(self.work / 'invalid_qcd.lhe'),
                    '--metadata', str(self.work / 'invalid_qcd.jsonl'),
                    '--settings', str(settings), '--pdf', self.pdf],
                    capture_output=True, text=True, env=self.env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Incompatible scalar shower setting: ' + setting,
                              result.stderr)

    def test_nested_decay_trees_and_effective_cutoffs(self):
        # ISR recoils against H -> Z Z -> e e mu mu. Pythia postpones the
        # supplied decay tree until after production radiation; the one-step
        # endpoint must restore it before evaluating the external real point.
        fixture = self.work / 'nested.lhe'
        header = self.input.read_text().split('<event>', 1)[0]
        source = '''<event>
9 1 1 100 0.0072973525693 0.120179
2 -1 0 0 501 0 0 0 100 100 0 0 9
-2 -1 0 0 0 501 0 0 -100 100 0 0 9
25 2 1 2 0 0 0 0 0 200 200 0 9
23 2 3 3 0 0 80 0 0 100 60 0 9
23 2 3 3 0 0 -80 0 0 100 60 0 9
11 1 4 4 0 0 40 30 0 50 0 0 9
-11 1 4 4 0 0 40 -30 0 50 0 0 9
13 1 5 5 0 0 -40 30 0 50 0 0 9
-13 1 5 5 0 0 -40 -30 0 50 0 0 9
</event>
'''
        fixture.write_text(header + source * 6 + '</LesHouchesEvents>\n')
        settings = self.work / 'nested.cmnd'
        settings.write_text('# Physical model and infrared support only\n\n'
                            'SpaceShower:pTmin = 0.5 ! ISR boundary\n'
                            'TimeShower:pTmin = 0.5 # FSR boundary\n'
                            '11:m0 = 0.\n13:m0 = 0.\n')
        output, metadata = self.work / 'nested_out.lhe', self.work / 'nested.jsonl'
        result = subprocess.run([str(self.exe), '--input', str(fixture),
            '--output', str(output), '--metadata', str(metadata),
            '--pdf', self.pdf, '--seed', '1', '--event-kind', 'S',
            '--alpha-s-mz', '0.120179', '--alpha-s-order', '2',
            '--settings', str(settings)], capture_output=True, text=True,
            env=self.env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [json.loads(line) for line in metadata.read_text().splitlines()]
        emitted = [r for r in records if r['emitted']]
        self.assertTrue(emitted)
        for record in records:
            self.assertAlmostEqual(record['shower_cutoffs'][0], 0.5, places=14)
            self.assertAlmostEqual(record['shower_cutoffs'][1],
                                   0.5867333369950678, places=13)
            self.assertTrue(all(0.99 < x < 1. for x in record['beam_xmax']))
        for record in emitted:
            particles = record['particles']
            self.assertEqual(len(particles), 10)
            self.assertEqual(sum(p['status'] == 2 for p in particles), 3)
            self.assertEqual(sorted(p['id'] for p in particles
                                    if p['status'] == 1 and abs(p['id']) > 5
                                    and p['id'] != 21), [-13, -11, 11, 13])
            self.assertEqual(sum(p['status'] == 1 and
                                 (abs(p['id']) <= 5 or p['id'] == 21)
                                 for p in particles), 1)
            for index, parent in enumerate(particles, 1):
                if parent['status'] != 2:
                    continue
                children = [p for p in particles if p['mothers'] == [index, index]]
                self.assertEqual(len(children), 2)
                for component in range(4):
                    self.assertAlmostEqual(sum(p['p'][component] for p in children),
                                           parent['p'][component], delta=1.e-8)
            external = [p for p in particles if p['status'] != 2]
            for component in range(4):
                self.assertAlmostEqual(sum(p['p'][component] *
                    (-1 if p['status'] == -1 else 1) for p in external),
                    0., delta=1.e-8)

    def test_stock_threshold_conversion_keeps_native_mass(self):
        # A massless hard b entering the native forced threshold window still
        # creates its massive sister; removing our patches must not project it.
        fixture = self.work / 'threshold.lhe'
        header = self.input.read_text().split('<event>', 1)[0]
        source = event(1, 5.)
        source = source.replace('2 -1 0 0 501', '5 -1 0 0 501')
        source = source.replace('-2 -1 0 0 0 501', '-5 -1 0 0 0 501')
        fixture.write_text(header + source * 16 + '</LesHouchesEvents>\n')
        output = self.work / 'threshold.lhe.out'
        metadata = self.work / 'threshold.jsonl'
        result = subprocess.run([str(self.exe), '--input', str(fixture),
            '--output', str(output), '--metadata', str(metadata),
            '--pdf', self.pdf, '--seed', '7349', '--massless-light-quarks',
            '--alpha-s-mz', '0.118', '--alpha-s-order', '1'],
            capture_output=True, text=True, env=self.env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in metadata.read_text().splitlines()]
        self.assertEqual(len(rows), 16)
        self.assertEqual(len(blocks(output)), 16)
        for row in rows:
            self.assertTrue(row['emitted'])
            self.assertEqual(row['emission_type'], 'ISR')
            self.assertTrue(any(abs(p['id']) == 5 and p['status'] == 1
                                and p['mass'] == 4. for p in row['particles']))
            self.assertEqual(row['trial_headroom'], 1)
            self.assertEqual(row['alpha_s_nfmax'], 5)

    def test_removed_trial_headroom_option_is_rejected(self):
        result = subprocess.run([str(self.exe), '--trial-headroom', '100'],
                                capture_output=True, text=True, env=self.env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unknown option: --trial-headroom', result.stderr)

    def test_logged_stock_warning_does_not_filter_or_retry_events(self):
        # Inject only a logger message into a private driver copy. It must
        # change neither the native RNG trajectory nor input accounting.
        source = SOURCE.read_text()
        call = 'const bool success = pythia.next();'
        self.assertEqual(source.count(call), 1)
        injected = self.work / 'warning_driver.cc'
        injected.write_text(source.replace(call, call + '\n' +
            'const_cast<Logger&>(pythia.logger).WARNING_MSG('
            '"weight above unity (injected warning control)");'))
        executable = self.work / 'warning_driver.exe'
        built = subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++11', '-O1',
            '-I' + str(self.prefix / 'include'), str(injected), '-o', str(executable),
            '-L' + str(self.prefix / 'lib'), '-Wl,-rpath,' + str(self.prefix / 'lib'),
            '-lpythia8'], capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        output, metadata = self.work / 'warning.lhe', self.work / 'warning.jsonl'
        result = subprocess.run([str(executable), '--input', str(self.input),
            '--output', str(output), '--metadata', str(metadata), '--pdf', self.pdf,
            '--seed', '7349'], capture_output=True, text=True, env=self.env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(output.read_bytes(), self.output.read_bytes())
        rows = [json.loads(line) for line in metadata.read_text().splitlines()]
        self.assertEqual(len(rows), len(self.source_events))
        for row in rows:
            injected_counts = [item['count'] for item in row['native_trial_warning_counts']
                               if 'injected warning control' in item['message']]
            self.assertEqual(injected_counts, [1] if row['event_kind'] == 'S' else [])

    def test_configuration_probe_matches_event_profile(self):
        probe = self.work / 'profile.json'
        result = subprocess.run([str(self.exe), '--probe-config', str(probe),
            '--beam-id-a', '2212', '--beam-id-b', '2212',
            '--beam-energy-a', '6500', '--beam-energy-b', '6500',
            '--pdf', self.pdf, '--massless-light-quarks'],
            capture_output=True, text=True, env=self.env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        configuration = json.loads(probe.read_text())
        event_configuration = json.loads(self.metadata.read_text().splitlines()[0])
        self.assertEqual(configuration['beam_xmax'], event_configuration['beam_xmax'])
        self.assertEqual(configuration['shower_cutoffs'], event_configuration['shower_cutoffs'])
        self.assertNotIn('headroom_patch_version', configuration)
        self.assertNotIn('four_flavour_patch_version', configuration)
        self.assertEqual(configuration['trial_headroom'], 1)
        self.assertEqual(configuration['shower_backend'], 'stock-pythia8')
        for key in ['flavour_count', 'n_quark_in', 'n_gluon_to_quark', 'alpha_s_nfmax']:
            self.assertEqual(configuration[key], 5)
            self.assertEqual(event_configuration[key], 5)
        self.assertEqual(configuration['heavy_mass_floors'], {'4': 1.2, '5': 4.})
        self.assertEqual(configuration['particle_masses']['4'], 0.)
        self.assertEqual(configuration['particle_masses']['5'], 0.)

    def test_unsupported_flavour_count_fails(self):
        result = subprocess.run([str(self.exe), '--input', str(self.input),
            '--output', str(self.work / 'invalid_flavour.lhe'),
            '--metadata', str(self.work / 'invalid_flavour.jsonl'),
            '--flavour-count', '6'], capture_output=True, text=True, env=self.env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--flavour-count must be 4 or 5', result.stderr)


if __name__ == '__main__':
    unittest.main()
