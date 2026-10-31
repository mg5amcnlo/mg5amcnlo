"""Fixed e+e- beam profiles and genuine no-PDF stock-shower handling."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from madgraph.various import macnlops as mc
from madgraph.various.banner import RunCardNLO
from tests.unit_tests.various.test_macnlops_continuation import shower_profile

ROOT = Path(__file__).resolve().parents[3]


class TestFixedLeptonProfiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='macnlops-fixed-beams-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'Cards').mkdir()
        self.table = self.root/'SubProcesses/P0/leshouche_info.dat'
        self.table.parent.mkdir(parents=True)
        self.table.write_text('I 1 1 -11 11 2 -2 21\nI 2 1 -11 11 1 -1 21\n')
        self.card = dict(lpp1=0, lpp2=0, pdlabel='nn23nlo', lhaid=[331700])

    def test_ordered_fixed_beams_come_from_generated_flavours(self):
        self.assertEqual(mc.source_beam_ids(self.root, self.card), [-11, 11])
        self.table.write_text('I 1 1 11 -11 2 -2 21\n')
        self.assertEqual(mc.source_beam_ids(self.root, self.card), [11, -11])
        self.assertEqual(mc.source_beam_ids(self.root, dict(lpp1=-1, lpp2=1)), [-2212, 2212])

    def test_mixed_ambiguous_or_partonic_fixed_beams_are_rejected(self):
        with self.assertRaises(mc.MatchingError):
            mc.source_beam_ids(self.root, dict(lpp1=0, lpp2=1))
        self.table.write_text('I 1 1 11 -11 2 -2\nI 2 1 -11 11 2 -2\n')
        with self.assertRaisesRegex(mc.MatchingError, 'unique ordered'):
            mc.source_beam_ids(self.root, self.card)
        self.table.write_text('I 1 1 2 -2 21 21\n')
        with self.assertRaisesRegex(mc.MatchingError, 'fixed e'):
            mc.source_beam_ids(self.root, self.card)

    def test_fixed_beams_do_not_resolve_or_fingerprint_an_inactive_pdf(self):
        with patch.object(mc, 'pdf_name', side_effect=AssertionError('unused PDF lookup')):
            self.assertEqual(mc.shower_pdf(self.root, self.card), ('none', None))

    def test_generation_profiles_fixed_beams_without_pdf_metadata(self):
        card = RunCardNLO()
        for key, value in dict(lpp1=0, lpp2=0, ebeam1=45.594, ebeam2=45.594,
                               mcatnlo_delta=True, macnlops_flavour_count=5).items():
            card.set(key, value, user=False)
        (self.root/'SubProcesses/FKS_params.dat').write_text('#DeltaCommonScale\n.true.\n')
        support = dict(n_quark_in=5, n_gluon_to_quark=5, shower_cutoffs=[.5, .5],
                       heavy_mass_floors={'4': 1.2, '5': 4.}, beam_xmax=[1., 1.])
        with patch.object(mc, 'hard_model_masses', return_value={}), \
                patch.object(mc, 'build_shower', return_value=Path('/driver')) as build, \
                patch.object(mc, 'probe_shower', return_value=support) as probe, \
                patch.object(mc, 'shower_environment', return_value={}), \
                patch.object(mc, 'shower_library_hashes', return_value={'libpythia8.so': 'a'*64}), \
                patch.object(mc, 'pdf_name', side_effect=AssertionError('unused PDF lookup')):
            profile = mc.prepare_generation(self.root, card, '/pythia')
        self.assertEqual(profile['construction'], mc.NATIVE_IR_DELTA_SOURCE)
        self.assertEqual(profile['pdf'], 'none')
        self.assertIsNone(profile['pdf_info_sha256'])
        self.assertIsNone(profile['pdf_member_sha256'])
        self.assertEqual(probe.call_args.args[2], [-11, 11])
        self.assertEqual(build.call_args.kwargs['pdf'], 'none')
        self.assertEqual(card['macnlops_beam_xmax1'], 1.)

    def test_continuation_requires_fixed_beams_and_disabled_lepton_pdf(self):
        profile = shower_profile(delta=True)
        profile.update(pdf='none', pdf_info_sha256=None, pdf_member_sha256=None)
        profile['support']['beam_ids'] = [-11, 11]
        profile['effective_shower_settings'].update({'PDF:pSet': '13', 'PDF:lepton': 'off'})
        self.assertIn('PDF:lepton = off', mc.continuation_physics(profile))
        for key, value in [('beams', [2212, 2212]), ('lepton_pdf', 'on'), ('pdf_hash', 'a'*64)]:
            bad = copy.deepcopy(profile)
            if key == 'beams': bad['support']['beam_ids'] = value
            elif key == 'lepton_pdf': bad['effective_shower_settings']['PDF:lepton'] = value
            else: bad['pdf_info_sha256'] = value
            with self.assertRaises(mc.MatchingError): mc.continuation_physics(bad)


class TestStockFixedLeptonShower(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix = os.environ.get('PYTHIA8') or os.environ.get('PYTHIA8LOCATION')
        if not prefix or not shutil.which('g++'):
            raise unittest.SkipTest('Set PYTHIA8 to test stock fixed-lepton shower')
        cls.prefix = Path(prefix)
        cls.tmp = tempfile.TemporaryDirectory(prefix='macnlops-ee-native-')
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.work = Path(cls.tmp.name)
        cls.exe = cls.work/'MacNLOPS'
        subprocess.run(['g++', '-O2', '-std=c++17', '-I', str(cls.prefix/'include'),
            str(ROOT/'Template/NLO/MCatNLO/srcPythia8/MacNLOPS.cc'),
            '-L', str(cls.prefix/'lib'), '-Wl,-rpath,'+str(cls.prefix/'lib'),
            '-lpythia8', '-ldl', '-o', str(cls.exe)], check=True, capture_output=True, text=True)
        cls.environment = dict(os.environ, PYTHIA8DATA=str(cls.prefix/'share/Pythia8/xmldoc'))
        cls.settings = cls.work/'physics.cmnd'
        cls.settings.write_text('SpaceShower:pTmin = .5\nTimeShower:pTmin = .5\n')

    def test_native_fixed_beam_probe_and_hadron_mislabel_rejection(self):
        output = self.work/'probe.json'
        command = [str(self.exe), '--probe-config', str(output), '--beam-id-a', '-11',
            '--beam-id-b', '11', '--beam-energy-a', '45.594', '--beam-energy-b', '45.594',
            '--settings', str(self.settings), '--pdf', 'none', '--alpha-s-mz', '.1365',
            '--alpha-s-order', '1', '--mcatnlo-delta']
        result = subprocess.run(command, env=self.environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        probe = json.loads(output.read_text())
        self.assertEqual(probe['beam_ids'], [-11, 11])
        self.assertEqual(probe['beam_xmax'], [1, 1])
        self.assertEqual(probe['beam_masses'], [0, 0])
        self.assertIn('PDF:lepton = off', Path(str(output)+'.settings.cmnd').read_text())
        command[command.index('--beam-id-a')+1] = '2212'
        command[command.index('--beam-id-b')+1] = '2212'
        result = subprocess.run(command, env=self.environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no-PDF profile requires fixed e+e-', result.stderr)

    def test_first_emissions_are_fsr_without_lepton_structure_functions(self):
        source = self.work/'source.lhe'
        header = '''<LesHouchesEvents version="3.0">
<header></header>
<init>
-11 11 45.594 45.594 0 0 0 0 -4 1
1e-9 0 1e-9 1
</init>
'''
        event = '''<event>
4 1 1.0 45.594 0.007297 0.118
-11 -1 0 0 0 0 0 0 45.594 45.594 0 0 9
11 -1 0 0 0 0 0 0 -45.594 45.594 0 0 9
2 1 1 2 501 0 45.594 0 0 45.594 0 0 9
-2 1 1 2 0 501 -45.594 0 0 45.594 0 0 9
#aMCatNLO 1
<scales scalup_3_4="45.594" scalup_4_3="45.594"></scales>
</event>
'''
        source.write_text(header+event*100+'</LesHouchesEvents>\n')
        output, metadata = self.work/'events.lhe', self.work/'events.jsonl'
        result = subprocess.run([str(self.exe), '--input', str(source), '--output', str(output),
            '--metadata', str(metadata), '--settings', str(self.settings), '--pdf', 'none',
            '--alpha-s-mz', '.1365', '--alpha-s-order', '1', '--mcatnlo-delta'],
            env=self.environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        rows = [json.loads(line) for line in metadata.read_text().splitlines()]
        self.assertEqual(len(rows), 100)
        emitted = [row for row in rows if row['emitted']]
        self.assertTrue(emitted)
        self.assertTrue(all(row['emission_type']=='FSR' for row in emitted))
        self.assertTrue(all(row['beam_xmax']==[1, 1] and row['nominal_shower_weight']==1 for row in rows))
