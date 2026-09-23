"""Numerical checks of MINT's Born-spreading restart."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestBornSpreading(unittest.TestCase):
    def test_restart_preserves_integral_normalization(self):
        """The first restarted iteration must average its sampled weights."""
        with tempfile.TemporaryDirectory(prefix='mg5_born_spread_') as tmp:
            work = Path(tmp)
            (work / 'params.f90').write_text(
                'module FKSParams\n'
                'logical :: use_poly_virtual=.false.\n'
                'double precision :: virt_fraction=1d0,min_virt_fraction=0.01d0\n'
                'end module\n')
            # This analytic integrand needs neither plots nor FKS sampling.
            unused = ('accum', 'hwu_accum_iter', 'hwu_add_points', 'initplot',
                      'empty_mc_integer', 'regrid_mc_integer', 'reset_mc_grid',
                      'deallocate_born_spread_lines')
            polyfit = ('add_point_polyfit', 'do_polyfit', 'get_polyfit',
                       'init_polyfit', 'restore_polyfit', 'save_polyfit')
            (work / 'stubs.f90').write_text(
                ''.join('subroutine %s\nend subroutine\n' % name
                        for name in unused) +
                ''.join('subroutine %s\nstop 99\nend subroutine\n' % name
                        for name in polyfit))
            executable = work / 'check_restart'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all', '-fno-automatic',
                str(work / 'params.f90'), str(TEMPLATE / 'mint_module.f90'),
                str(work / 'stubs.f90'),
                str(ROOT / 'tests/input_files/check_born_spreading_restart.f90'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for channels, enabled in ((1, 0), (1, 1), (2, 1)):
                with self.subTest(channels=channels, enabled=enabled):
                    result = subprocess.run(
                        [str(executable), str(channels), str(enabled)],
                        cwd=work, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0,
                                     result.stdout + result.stderr)
                    self.assertIn('PASS MINT restart normalization', result.stdout)
