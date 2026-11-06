"""Check the variance of the iteration estimator that MINT actually computes."""

import math
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_mint_channels import SUPPORT


ROOT = pathlib.Path(__file__).resolve().parents[3]


def combine(means, errors):
    """Independent linear-coefficient oracle for the legacy central estimator."""
    weights = [1.]
    weight_error = errors[0]
    for error in errors[1:]:
        previous = error / (weight_error + error)
        current = weight_error / (weight_error + error)
        weights = [previous * weight for weight in weights] + [current]
        weight_error = 1. / math.sqrt(1. / weight_error**2 + 1. / error**2)
    return (sum(weight * mean for weight, mean in zip(weights, means)),
            math.sqrt(sum((weight * error)**2
                          for weight, error in zip(weights, errors))),
            weight_error, weights)


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TestMintUncertainty(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='mg5-mint-errors-')
        cls.path = pathlib.Path(cls.directory.name)
        source = (ROOT / 'Template/NLO/SubProcesses/mint_module.f90').read_text()
        (cls.path / 'mint_module.f90').write_text(source.replace('private', 'public'))
        (cls.path / 'support.f90').write_text(SUPPORT)
        (cls.path / 'check.f90').write_text(PROGRAM)
        result = subprocess.run(
            ['gfortran', '-O0', '-g', '-fcheck=all', '-ffpe-trap=zero,invalid',
             '-ffree-line-length-none',
             '-o', 'check', 'support.f90',
             str(ROOT / 'Template/NLO/SubProcesses/virtual_surrogate.f90'),
             'mint_module.f90', 'check.f90'],
            cwd=cls.path, capture_output=True, text=True)
        if result.returncode:
            cls.directory.cleanup()
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_case(self, case):
        result = subprocess.run([str(self.path / 'check'), case], cwd=self.path,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return {parts[1]: list(map(float, parts[2:]))
                for line in result.stdout.splitlines()
                if (parts := line.split()) and parts[0] == 'CHECK'}

    def assert_combined(self, result, means, errors):
        expected_mean, expected_error, harmonic, weights = combine(means, errors)
        self.assertAlmostEqual(result['TOTAL'][0], expected_mean, places=13)
        self.assertAlmostEqual(result['TOTAL'][1], expected_error, places=13)
        self.assertAlmostEqual(result['NORM'][0], harmonic, places=13)
        return weights

    def test_two_equal_errors_are_unchanged(self):
        result = self.run_case('equal2')
        self.assert_combined(result, [5., 9.], [2., 2.])
        self.assertAlmostEqual(result['TOTAL'][0], 7.)
        self.assertAlmostEqual(result['TOTAL'][1], math.sqrt(2.))

    def test_unequal_errors_follow_actual_mean_weights(self):
        result = self.run_case('unequal')
        self.assert_combined(result, [5., 9., -2., 4., 11.], [1., 4., 2., 3., 1.5])
        # The legacy harmonic number selects weights, but is not their error.
        self.assertGreater(result['TOTAL'][1], result['NORM'][0])
        # Histograms must receive the old harmonic selector, not the new error.
        self.assertAlmostEqual(result['HWU'][0], 1.5)
        self.assertAlmostEqual(result['HWU'][1], combine([5., 9., -2., 4.],
                                                       [1., 4., 2., 3.])[2])

    def test_equal_error_recursion_retains_late_iteration_weights(self):
        result = self.run_case('equal5')
        self.assert_combined(result, [5., 9., -2., 4., 11.], [2.] * 5)
        self.assertAlmostEqual(result['TOTAL'][1] / result['NORM'][0],
                               1.0769226076780012)

    def test_reported_variance_matches_exhaustive_independent_noise(self):
        # Enumerate every independent +/-sigma outcome for five iterations.
        # The population variance is exact and does not reuse the combiner.
        mean, population_variance, reported_variance = self.run_case('ensemble')['ENSEMBLE']
        self.assertAlmostEqual(mean, 3.)
        self.assertAlmostEqual(population_variance, reported_variance, places=13)

    def test_channels_use_total_weights_and_retain_zero_error_history(self):
        result = self.run_case('unequal')
        weights = combine([5., 9., -2., 4., 11.], [1., 4., 2., 3., 1.5])[3]
        channel_means = ([0., 3., 0., 1., -2.], [5., 6., -2., 3., 13.])
        channel_errors = ([0., 3.2, 0., 2.4, 1.2], [1., 3.6, 2., 2.7, 1.35])
        for key, means, errors in zip(('CHANNEL1', 'CHANNEL2'), channel_means,
                                      channel_errors):
            self.assertAlmostEqual(result[key][0], sum(w * x for w, x in zip(weights, means)))
            self.assertAlmostEqual(result[key][1], math.sqrt(sum((w * e)**2
                                                                 for w, e in zip(weights, errors))))
        self.assertAlmostEqual(result['TOTAL'][0], result['CHANNEL1'][0] + result['CHANNEL2'][0])
        self.assertGreater(result['CHANNEL1'][1], 0.)
        self.assertTrue(math.isfinite(result['CHANNEL1_CHI2'][0]))
        # A total second moment contains cross-channel covariance already.
        self.assertNotAlmostEqual(result['TOTAL'][1]**2,
                                  result['CHANNEL1'][1]**2 + result['CHANNEL2'][1]**2)

    def test_mc_and_automatic_callers_use_same_variance_rule(self):
        for mode in ('mc', 'adaptive'):
            with self.subTest(mode=mode):
                self.assert_combined(self.run_case(mode), [5., 9., -2., 4., 11.],
                                     [1., 4., 2., 3., 1.5])

    def test_last_three_window_and_signed_zero_or_negative_first_mean(self):
        for mode, first in (('unequal', -2.), ('lastzero', 0.), ('lastnegative', -7.)):
            with self.subTest(mode=mode):
                result = self.run_case(mode)
                mean, error, _, _ = combine([first, 4., 11.], [2., 3., 1.5])
                self.assertAlmostEqual(result['LAST3'][0], mean)
                self.assertAlmostEqual(result['LAST3'][1], error)

    def test_last_three_skips_optional_zero_error_entry(self):
        result = self.run_case('lastskip')
        mean, error, _, _ = combine([4., 11.], [3., 1.5])
        self.assertAlmostEqual(result['LAST3'][0], mean)
        self.assertAlmostEqual(result['LAST3'][1], error)

    def test_reset_clears_weight_state_and_accuracy_uses_propagated_error(self):
        result = self.run_case('accuracy')
        self.assertEqual(result['ACCURACY'], [0., 1.])
        self.assertEqual(result['RESET'], [0., 0., 0.])
        self.assertEqual(result['RESTART'], [3., 2., 2.])


PROGRAM = r'''
program check_mint_errors
use mint_module
implicit none
character(len=16) :: mode
integer :: j,count,pattern
logical :: done1,done2
double precision :: means(5),errors(5),first_channel(5),e1,e2,total,total_squared,noise
call get_command_argument(1,mode)
fixed_order=.true.
nlo_ps=.false.
nchans=2
double_events=.false.
if(mode.eq.'mc') fixed_order=.false.
if(mode.eq.'adaptive') double_events=.true.
if(mode.eq.'ensemble') then
 nchans=1
 errors=[1d0,4d0,2d0,3d0,1.5d0]
 total=0d0
 total_squared=0d0
 do pattern=0,31
  call setup_common
  do j=1,5
   nit=j
   noise=errors(j)
   if(btest(pattern,j-1)) noise=-noise
   vtot=0d0
   etot=0d0
   vtot(1:2,0)=3d0+noise
   vtot(1:2,1)=vtot(1:2,0)
   etot(1:2,0:1)=errors(j)
   call combine_iterations
  enddo
  total=total+ans(2,0)
  total_squared=total_squared+ans(2,0)**2
 enddo
 write(*,*) 'CHECK ENSEMBLE',total/32d0,total_squared/32d0-(total/32d0)**2,unc(2,0)**2
 stop
endif
call setup_common
count=5
means=[5d0,9d0,-2d0,4d0,11d0]
errors=[1d0,4d0,2d0,3d0,1.5d0]
first_channel=[0d0,3d0,0d0,1d0,-2d0]
if(mode.eq.'equal2') then
 count=2
 errors=2d0
elseif(mode.eq.'equal5') then
 errors=2d0
elseif(mode.eq.'lastzero') then
 means(3)=0d0
elseif(mode.eq.'lastnegative') then
 means(3)=-7d0
endif
do j=1,count
 nit=j
 vtot=0d0
 etot=0d0
 vtot(2,0)=means(j)
 vtot(2,1)=first_channel(j)
 vtot(2,2)=means(j)-first_channel(j)
 vtot(1,0:2)=abs(vtot(2,0:2))
 etot(1:2,0)=errors(j)
 e1=.8d0*errors(j)
 e2=.9d0*errors(j)
 if(j.eq.1.or.j.eq.3) then
  e1=0d0
  e2=errors(j)
 endif
 etot(1:2,1)=e1
 etot(1:2,2)=e2
 if(mode.eq.'lastskip'.and.j.eq.3) etot(2,0)=0d0
 call combine_iterations
 call combine_final_three_iterations
enddo
write(*,*) 'CHECK TOTAL',ans(2,0),unc(2,0)
write(*,*) 'CHECK CHANNEL1',ans(2,1),unc(2,1)
write(*,*) 'CHECK CHANNEL2',ans(2,2),unc(2,2)
write(*,*) 'CHECK CHANNEL1_CHI2',chi2(2,1)
write(*,*) 'CHECK NORM',iteration_weight_unc(2)
write(*,*) 'CHECK HWU',HwU_values
write(*,*) 'CHECK LAST3',ans_l3(2),unc_l3(2)
if(mode.eq.'accuracy') then
 chi2=0d0
 min_it=0
 ! Disable the optional final-three stopping alternative for this check.
 ans_l3(1)=1d0
 unc_l3(1)=1d99
 accuracy=(iteration_weight_unc(1)+unc(1,0))/(2d0*ans(1,0))
 call check_desired_accuracy(done1)
 accuracy=2d0*unc(1,0)/ans(1,0)
 call check_desired_accuracy(done2)
 write(*,*) 'CHECK ACCURACY',merge(1,0,done1),merge(1,0,done2)
 call setup_common
 write(*,*) 'CHECK RESET',iteration_weight_unc(2),ans(2,0),unc(2,0)
 nit=1
 vtot(1:2,0)=3d0
 etot(1:2,0)=2d0
 call combine_iterations
 write(*,*) 'CHECK RESTART',ans(2,0),unc(2,0),iteration_weight_unc(2)
endif
end program
'''
