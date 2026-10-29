"""FxFx-Delta uses ordinary FxFx tags; standalone Delta retains dipole scales."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestFxFxDeltaLHE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_fxfx_delta_lhe_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.work = Path(cls.tempdir.name)
        (cls.work / 'run.inc').write_text(
            '      integer ickkw\n'
            '      logical mcatnlo_delta,do_rwgt_scale,do_rwgt_pdf\n'
            '      logical store_rwgt_info\n'
            '      common/test_run/ickkw,mcatnlo_delta,do_rwgt_scale,\n'
            '     & do_rwgt_pdf,store_rwgt_info\n')
        shutil.copyfile(TEMPLATE / 'unlops.inc', cls.work / 'unlops.inc')
        names = ('write_lhef_event', 'read_lhef_event', 'read_lhef_event_catch',
                 'read_lhef_delta_scales', 'read_rwgt_line')
        (cls.work / 'routines.f').write_text('\n'.join(
            fortran_routine(TEMPLATE / 'handling_lhe_events.f', name)
            for name in names))
        cls.exe = cls.work / 'check'
        result = subprocess.run([
            shutil.which('gfortran'), '-O0', '-g', '-std=legacy',
            '-ffixed-line-length-none', '-fcheck=all',
            '-ffpe-trap=invalid,zero,overflow', '-I', str(cls.work),
            str(ROOT / 'Template/NLO/Source/extra_weights.f'),
            str(cls.work / 'routines.f'),
            str(ROOT / 'tests/input_files/check_fxfx_delta_lhe.f90'),
            '-o', str(cls.exe)], cwd=cls.work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def roundtrip(self, mode, comments=True, reweight=False):
        result = subprocess.run([str(self.exe), mode, str(int(comments)),
                                 str(int(reweight))],
                                cwd=self.work, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for filename in ('event.lhe', 'roundtrip.lhe'):
            event = ET.fromstring((self.work / filename).read_text())
            scales = event.findall('scales')
            self.assertEqual(len(scales), 1)
            attrs = scales[0].attrib
            if mode in ('fxfx', 'combined'):
                self.assertEqual(float(attrs['pt_clust_3']), 30.)
                self.assertEqual(float(attrs['pt_clust_12']), 29.999)
                self.assertTrue(all(name.startswith('pt_clust_') for name in attrs))
            else:
                self.assertEqual(float(attrs['scalup_10_11']), 20.11)
                self.assertEqual(float(attrs['scalup_11_10']), 21.1)
                self.assertNotIn('scalup_1_12', attrs)
            self.assertEqual(len(event.findall('rwgt')), int(reweight))

    def test_combined_uses_ordinary_fxfx_format_in_both_readers(self):
        for comments, reweight in ((False, False), (True, False), (True, True)):
            with self.subTest(comments=comments, reweight=reweight):
                self.roundtrip('fxfx', comments, reweight)
                ordinary = (self.work / 'roundtrip.lhe').read_text()
                self.roundtrip('combined', comments, reweight)
                self.assertEqual((self.work / 'roundtrip.lhe').read_text(), ordinary)

    def test_standalone_delta_retains_directed_scales(self):
        for comments, reweight in ((False, False), (True, False), (True, True)):
            with self.subTest(comments=comments, reweight=reweight):
                self.roundtrip('delta', comments, reweight)

    def test_python_matching_scale_edits_keep_fxfx_values(self):
        from madgraph.various.lhe_parser import Event

        self.roundtrip('combined')
        event = Event((self.work / 'event.lhe').read_text())
        self.assertEqual(event.parse_matching_scale()[2], 30.)
        event.matched_scale_data[2] = 28.
        attrs = ET.fromstring(str(event)).find('scales').attrib
        self.assertEqual(float(attrs['pt_clust_3']), 28.)
        self.assertEqual(float(attrs['pt_clust_12']), 29.999)

    def test_python_matching_parser_preserves_standalone_dipole_scales(self):
        from madgraph.various.lhe_parser import Event

        self.roundtrip('delta')
        event = Event((self.work / 'event.lhe').read_text())
        event.parse_matching_scale()
        attrs = ET.fromstring(str(event)).find('scales').attrib
        self.assertEqual(float(attrs['scalup_10_11']), 20.11)
        self.assertEqual(float(attrs['scalup_11_10']), 21.1)
