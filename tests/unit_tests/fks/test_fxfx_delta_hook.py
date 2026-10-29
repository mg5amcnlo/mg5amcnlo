"""Check scalar FxFx matching with real Pythia; set PYTHIA8 to its installation."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT


@unittest.skipUnless(os.environ.get('PYTHIA8') and shutil.which('g++'),
                     'requires g++ and a Pythia installation in PYTHIA8')
class TestFxFxDeltaHook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pythia = Path(os.environ['PYTHIA8']).resolve()
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_fxfx_pythia_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.work = Path(cls.tempdir.name)
        cls.exe = cls.work / 'check_matching'
        cls.env = dict(os.environ, PYTHIA8DATA=str(pythia / 'share/Pythia8/xmldoc'))
        result = subprocess.run([
            shutil.which('g++'), '-O2', '-std=c++11', '-I' + str(pythia / 'include'),
            str(ROOT / 'tests/input_files/check_fxfx_delta_hook.cc'),
            '-L' + str(pythia / 'lib'), '-Wl,-rpath,' + str(pythia / 'lib'),
            '-lpythia8', '-ldl', '-o', str(cls.exe)],
            capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def check_event(self, order):
        for matched in (False, True):
            with self.subTest(order=order, matched=matched):
                # A physical u u~ -> Z g point, with optional Z -> e+ e-.
                rows = [
                    '2 -1 0 0 501 0 0 0 100 100 0 0 9',
                    '-2 -1 0 0 0 502 0 0 -100 100 0 0 9',
                    '23 %d 1 2 0 0 -79.2120540156 0 0 120.7879459844 91.1876 0 9'
                    % (1 if order == 'stable' else 2)]
                gluon = '21 1 1 2 501 502 79.2120540156 0 0 79.2120540156 0 0 9'
                daughters = [
                    '-11 1 3 3 0 0 -39.6060270078 45.5938 0 60.3939729922 0 0 9',
                    '11 1 3 3 0 0 -39.6060270078 -45.5938 0 60.3939729922 0 0 9']
                if order == 'daughters_first':
                    rows.extend(daughters + [gluon])
                    gluon_label, daughter_labels = 6, (4, 5)
                else:
                    rows.append(gluon)
                    gluon_label = 4
                    daughter_labels = () if order == 'stable' else (5, 6)
                    if order != 'stable':
                        rows.extend(daughters)
                attrs = ['pt_clust_3="60"',
                         'pt_clust_%d="%s"' % (gluon_label, '60' if matched else '59.999')]
                # Non-QCD particles have need_matching=0, so the production
                # writer leaves their tags at SCALUP. Pythia ignores them
                # when selecting matching partons.
                attrs.extend('pt_clust_%d="60"' % i for i in daughter_labels)
                path = self.work / 'event.lhe'
                path.write_text('''<LesHouchesEvents version="3.0">
<header></header>
<init>
2212 2212 6500 6500 0 0 0 0 3 1
1 0 1 1
</init>
<event npNLO="1" npLO="-1">
%d 1 1 60 0.007297 0.118
%s
<scales %s></scales>
</event>
</LesHouchesEvents>
''' % (len(rows), '\n'.join(rows), ' '.join(attrs)))
                result = subprocess.run([str(self.exe), str(path), str(int(matched))],
                    cwd=self.work, env=self.env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('PASS stock FxFx matching', result.stdout)

    def test_stable_resonance_control(self):
        self.check_event('stable')

    def test_predecayed_daughters_before_production_gluon(self):
        self.check_event('daughters_first')

    def test_predecayed_production_gluon_before_daughters(self):
        self.check_event('gluon_first')
