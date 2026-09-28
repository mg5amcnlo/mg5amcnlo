"""Herwig angular-shower scale reconstruction without Herwig or ThePEG."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestHerwig7Scales(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_herwig7_scales_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        work = Path(cls.tempdir.name)
        cls.executable = work / 'check_herwig7_scales'
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-std=f2003', '-fcheck=all',
            str(ROOT / 'Template/NLO/SubProcesses/herwig7_scales.f90'),
            str(ROOT / 'tests/input_files/check_herwig7_scales.f90'),
            '-o', str(cls.executable)], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def check_case(self, name):
        result = subprocess.run([str(self.executable), name],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS ' + name, result.stdout)

    def test_massless_topologies_and_separate_angular_and_pt_limits(self):
        self.check_case('massless')

    def test_massive_pt_envelopes_and_boost_invariance(self):
        self.check_case('massive')

    def test_threshold_and_collinear_connections(self):
        self.check_case('threshold')

    def test_invalid_inputs(self):
        self.check_case('invalid')
