"""Numerical regressions for the existing SUSY MC subtraction support."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMCKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_mc_kernels_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        work = Path(cls.tempdir.name)
        (work / 'coupl.inc').write_text(
            '      double precision g\n'
            '      double complex gal(2)\n'
            '      common/test_couplings/g,gal\n')
        routines = [fortran_routine(TEMPLATE / filename, name)
                    for filename, name in (
                        ('cluster.f', 'set_particle_type'),
                        ('cluster.f', 'get_clustering_type'),
                        ('fks_singular.f', 'AP_reduced'),
                        ('fks_singular.f', 'AP_reduced_SUSY'))]
        (work / 'kernels.f').write_text('\n'.join(routines))
        cls.executable = work / 'check_kernels'
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-ffixed-line-length-none',
            '-fcheck=all', '-I', str(work), str(work / 'kernels.f'),
            str(ROOT / 'tests/input_files/check_mc_kernels.f90'),
            '-o', str(cls.executable)], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def check_kernel(self, mode):
        result = subprocess.run([str(self.executable), mode],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS ' + mode, result.stdout)

    def test_massive_octet_clustering(self):
        self.check_kernel('clustering')

    def test_susy_splitting_kernels(self):
        self.check_kernel('susy')
