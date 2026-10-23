"""Roundtrip FxFx tags and Delta dipole scales through the production LHE I/O."""

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

    def roundtrip(self, mode, comments=True, reweight=False, legacy=False):
        result = subprocess.run([str(self.exe), mode, str(int(comments)),
                                 str(int(reweight)), str(int(legacy))],
                                cwd=self.work, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for filename in ('event.lhe', 'roundtrip.lhe'):
            if legacy and filename == 'event.lhe':
                continue
            event = ET.fromstring((self.work / filename).read_text())
            scales = event.findall('scales')
            self.assertEqual(len(scales), 1)
            attrs = scales[0].attrib
            if mode == 'fxfx':
                self.assertEqual(float(attrs['pt_clust_3']), 30.)
                self.assertEqual(float(attrs['pt_clust_12']), 29.999)
            if mode == 'combined':
                self.assertEqual(attrs['fxfxdelta'], '1')
                self.assertEqual(attrs['fxfxmatch3'], '1')
                self.assertEqual(attrs['fxfxmatch12'], '0')
                self.assertFalse(any(name.startswith('pt_clust_') for name in attrs))
            if mode in ('delta', 'combined'):
                self.assertEqual(float(attrs['scalup_10_11']), 20.11)
                self.assertEqual(float(attrs['scalup_11_10']), 21.1)
                self.assertNotIn('scalup_1_12', attrs)
            self.assertEqual(len(event.findall('rwgt')), int(reweight))

    def test_combined_scales_survive_both_readers_and_rewriting(self):
        for comments, reweight in ((False, False), (True, False), (True, True)):
            with self.subTest(comments=comments, reweight=reweight):
                self.roundtrip('combined', comments, reweight)

    def test_standalone_delta_and_fxfx_formats_roundtrip(self):
        for mode in ('delta', 'fxfx'):
            for comments in (False, True):
                with self.subTest(mode=mode, comments=comments):
                    self.roundtrip(mode, comments)

    def test_legacy_separate_tags_are_merged_on_rewrite(self):
        self.roundtrip('combined', legacy=True)

    def test_python_matching_scale_edits_preserve_dipole_attributes(self):
        import copy
        from madgraph.various.lhe_parser import Event

        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                self.roundtrip('combined', legacy=legacy)
                path = self.work / ('legacy.lhe' if legacy else 'event.lhe')
                event = Event(path.read_text())
                scales = event.parse_matching_scale()
                self.assertEqual(scales[2], 1.)
                self.assertEqual(scales[11], 0.)
                event = copy.deepcopy(event)
                # Exercise the same per-particle representation used by decay
                # insertion: reordering flags must preserve their meaning.
                event.matched_scale_data[2], event.matched_scale_data[11] = 0., 1.
                output = ET.fromstring(str(event))
                self.assertEqual(len(output.findall('scales')), 1)
                attrs = output.find('scales').attrib
                self.assertEqual(attrs['fxfxdelta'], '1')
                self.assertEqual(attrs['fxfxmatch3'], '0')
                self.assertEqual(attrs['fxfxmatch12'], '1')
                self.assertFalse(any(name.startswith('pt_clust_') for name in attrs))
                self.assertEqual(float(attrs['scalup_10_11']), 20.11)
                self.assertEqual(float(attrs['scalup_11_10']), 21.1)

    def test_python_flags_supersede_legacy_matching_scales(self):
        from madgraph.various.lhe_parser import Event

        self.roundtrip('combined')
        text = (self.work / 'event.lhe').read_text()
        event = Event(text.replace('fxfxmatch3="1"',
                                   'fxfxmatch3="1" pt_clust_3="30"'))
        self.assertEqual(event.parse_matching_scale()[2], 1.)
        attrs = ET.fromstring(str(event)).find('scales').attrib
        self.assertEqual(attrs['fxfxmatch3'], '1')
        self.assertNotIn('pt_clust_3', attrs)

    def test_python_ordinary_fxfx_retains_scale_values(self):
        from madgraph.various.lhe_parser import Event

        self.roundtrip('fxfx')
        event = Event((self.work / 'event.lhe').read_text())
        self.assertEqual(event.parse_matching_scale()[2], 30.)
        event.matched_scale_data[2] = 28.
        attrs = ET.fromstring(str(event)).find('scales').attrib
        self.assertEqual(float(attrs['pt_clust_3']), 28.)
        self.assertNotIn('fxfxdelta', attrs)

    def test_python_numeric_delta_marker_is_validated(self):
        from madgraph.various.lhe_parser import Event

        self.roundtrip('combined')
        text = (self.work / 'event.lhe').read_text()
        for marker in ('1.0', '1e0'):
            event = Event(text.replace('fxfxdelta="1"', 'fxfxdelta="%s"' % marker))
            self.assertEqual(event.parse_matching_scale()[2], 1.)
            self.assertEqual(event.matched_scale_format, 'fxfxmatch')
        for marker in ('0', '2', 'nan', 'invalid'):
            event = Event(text.replace('fxfxdelta="1"', 'fxfxdelta="%s"' % marker))
            with self.assertRaises(ValueError):
                event.parse_matching_scale()

    def test_python_resonance_decay_preserves_original_matching_labels(self):
        from madgraph.various.lhe_parser import Event

        event = Event('''<event>
5 1 1 30 0.01 0.1
1 -1 0 0 501 0 0 0 100 100 0 0 9
-1 -1 0 0 0 501 0 0 -100 100 0 0 9
23 1 1 2 0 0 0 0 0 80 80 0 9
21 1 1 2 502 503 60 0 0 60 0 0 9
21 1 1 2 503 502 -60 0 0 60 0 0 9
<scales fxfxdelta="1" fxfxmatch3="0" fxfxmatch4="1" fxfxmatch5="0"
 scalup_1_2="30" scalup_2_1="25" scalup_4_5="20" scalup_5_4="15"></scales>
</event>
''')
        decay = Event('''<event>
3 1 1 30 0.01 0.1
23 -1 0 0 0 0 0 0 0 80 80 0 9
11 1 1 1 0 0 0 0 40 40 0 0 9
-11 1 1 1 0 0 0 0 -40 40 0 0 9
</event>
''')
        event.add_decay_to_particle(2, decay)
        self.assertEqual(len(event.matched_scale_data), len(event))
        self.assertEqual(event[2].status, 2)
        attrs = ET.fromstring(str(event)).find('scales').attrib
        self.assertNotIn('fxfxmatch3', attrs)
        self.assertEqual(attrs['fxfxmatch4'], '1')
        self.assertEqual(attrs['fxfxmatch5'], '0')
        self.assertEqual(attrs['fxfxmatch6'], '0')
        self.assertEqual(attrs['fxfxmatch7'], '0')
        self.assertEqual(attrs['scalup_4_5'], '20')
        self.assertEqual(attrs['scalup_5_4'], '15')
