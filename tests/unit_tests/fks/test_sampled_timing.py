"""Sample complete integration points without changing physics or timer nesting."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestSampledTiming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_sampled_timing_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        work = Path(cls.tempdir.name)
        # A controlled clock makes exclusive intervals and disabled-clock
        # behavior testable without timing noise or assumptions about hardware.
        source = (TEMPLATE.parent / 'Source/timing.f').read_text()
        (work / 'timing.f').write_text(source.replace('call cpu_time(', 'call test_cpu_time('))
        (work / 'clock.f90').write_text(
            'module test_clock\nreal(8) :: clock=0d0,read_cost=0d0\ninteger :: reads=0\nend module\n'
            'subroutine test_cpu_time(now)\nuse test_clock\nreal(8) :: now\n'
            'reads=reads+1\nclock=clock+read_cost\nnow=clock\nend subroutine\n')
        # Retain the production MINT point boundary. This fixture uses one
        # evaluation per point and checks both MC closing calls and FO calls.
        mint = (TEMPLATE / 'mint_module.f90').read_text()
        routine = re.search(r'^  subroutine compute_integrand\b.*?^  end subroutine compute_integrand',
                            mint, re.M | re.S).group()
        (work / 'mint.f90').write_text(
            'module mint_timing_fixture\nimplicit none\n'
            'integer,parameter :: ndimmax=2,nintegrals=2\n'
            'integer :: ndim=2,imode=0,ifold(2)=1\n'
            'logical :: fixed_order=.false.\nreal(8) :: f(2)\ncontains\n' + routine +
            '\nsubroutine nextlexi(folds,indices,iret)\n'
            'integer :: folds(2),indices(2),iret\niret=1\nend subroutine\n'
            'subroutine get_random_x_next_fold(x,vol,kfold)\n'
            'real(8) :: x(2),vol\ninteger :: kfold(2)\nstop 99\nend subroutine\n'
            'end module\n')
        cls.executable = work / 'check'
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
            '-ffixed-line-length-132', '-fno-automatic', '-fbacktrace',
            '-I', str(TEMPLATE), str(work / 'clock.f90'), str(work / 'timing.f'),
            str(work / 'mint.f90'), str(ROOT / 'tests/input_files/check_sampled_timing.f'),
            '-o', str(cls.executable)], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def run_mode(self, mode):
        return subprocess.run([str(self.executable), mode], capture_output=True, text=True, timeout=20)

    def check_mode(self, mode):
        result = self.run_mode(mode)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS sampled timing ' + mode, result.stdout)
        return result.stdout

    def test_one_percent_sampling_does_not_change_random_numbers(self):
        self.check_mode('sampling')

    def test_nested_categories_are_exclusive_and_same_category_is_cheap(self):
        self.check_mode('exclusive')

    def test_unsampled_jobs_report_only_the_exact_total(self):
        output = self.check_mode('empty')
        self.assertIn('No sampled integration points', output)
        self.assertNotIn('Time spent in Born', output)
        self.assertIn('42.500', output)

    def test_estimates_keep_their_sampling_weight_and_total_stays_exact(self):
        output = self.check_mode('estimate')
        times = dict((name, float(value.replace('D', 'E'))) for name, value in re.findall(
            r'Time spent in\s+(\w+)\s*:\s*([-+\d.EeDd]+)', output))
        self.assertEqual(times['Born_matrix_elements'], 200.0)
        self.assertEqual(times['Total'], 42.5)
        self.assertIn('exclusive estimates', output)

    def test_clock_cost_is_removed_from_the_physics_estimate(self):
        output = self.check_mode('clock_cost')
        born = re.search(r'Time spent in Born_matrix_elements\s*:\s*([-+\d.EeDd]+)', output)
        self.assertEqual(float(born[1].replace('D', 'E')), 200.0)
        self.assertIn('clock cost subtracted', output)

    def test_mint_sampling_covers_the_complete_point(self):
        self.check_mode('mint')

    def test_bad_nesting_is_detected(self):
        for mode, message in [('mismatch', 'Mismatched timing category'),
                              ('unclosed', 'Unclosed timing category')]:
            with self.subTest(mode=mode):
                result = self.run_mode(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout + result.stderr)
