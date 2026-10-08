"""Numerical history-map failures are recoverable; broken label maps are not."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMCMomentumPermutation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='mg5_mc_permutation_')
        cls.addClassCleanup(cls.tmp.cleanup)
        work = Path(cls.tmp.name)
        (work / 'nexternal.inc').write_text(
            '      integer nexternal,nincoming\n'
            '      parameter(nexternal=6,nincoming=2)\n')
        cls.executable = work / 'check_permutation'
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
            '-ffixed-line-length-none', '-ffunction-sections', '-fdata-sections',
            '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
            '-I', str(work), str(TEMPLATE / 'genps_fks_helpers.f'),
            str(ROOT / 'tests/input_files/check_mc_momentum_permutation.f90'),
            '-o', str(cls.executable)], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def check_mode(self, mode, diagnostic=None):
        result = subprocess.run([str(self.executable), mode], capture_output=True, text=True)
        output = result.stdout + result.stderr
        if diagnostic:
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn(diagnostic, output)
        else:
            self.assertEqual(result.returncode, 0, output)
            self.assertIn('PASS '+mode, output)

    def test_valid_massless_and_massive_permutations(self):
        self.check_mode('valid')

    def test_mass_mismatch_rejects_then_recovers(self):
        self.check_mode('mass')

    def test_w2j_sumkl_crash_point_rejects_without_stopping(self):
        self.check_mode('w2j_sumkl')

    def test_nonfinite_momenta_reject(self):
        self.check_mode('nonfinite')

    def test_callers_without_status_keep_strict_check(self):
        self.check_mode('strict', 'changes a leg mass')

    def test_out_of_range_map_remains_fatal(self):
        self.check_mode('range', 'Out-of-range MC momentum permutation')

    def test_nonbijective_map_remains_fatal(self):
        self.check_mode('duplicate', 'Non-bijective MC momentum permutation')

    def test_incoming_swap_remains_fatal(self):
        self.check_mode('incoming', 'exchanges an incoming leg')
