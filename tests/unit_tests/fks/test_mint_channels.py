"""Exercise MINT channel allocation and integration with controlled integrands."""

import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TestMintChannels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='mg5-mint-channels-')
        cls.path = pathlib.Path(cls.directory.name)
        source = (ROOT / 'Template/NLO/SubProcesses/mint_module.f90').read_text()
        # Expose state to the test driver without changing the production API.
        source = source.replace('private', 'public')
        (cls.path / 'mint_module.f90').write_text(source)
        (cls.path / 'support.f90').write_text(SUPPORT)
        (cls.path / 'check.f90').write_text(PROGRAM)
        result = subprocess.run(
            ['gfortran', '-O0', '-g', '-fcheck=all', '-ffree-line-length-none',
             '-o', 'check', 'support.f90', 'mint_module.f90', 'check.f90'],
            cwd=cls.path, capture_output=True, text=True)
        if result.returncode:
            cls.directory.cleanup()
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_case(self, case, directory=None):
        result = subprocess.run([str(self.path / 'check'), case],
                                cwd=directory or self.path, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [line.split()[1:] for line in result.stdout.splitlines()
                if line.startswith(' CHECK ')]

    def test_rms_allocation_and_unbiased_channel_compensation(self):
        probability, integral = self.run_case('rms')
        probability = list(map(float, probability))
        # The two relevant channels have equal integrals, but RMS weights 1:10.
        self.assertAlmostEqual(probability[0], .95 / 11 + .05 / 3)
        self.assertAlmostEqual(probability[1], .95 * 10 / 11 + .05 / 3)
        # A tiny channel with a 100% pilot error gets only its exploration share.
        self.assertAlmostEqual(probability[2], .05 / 3, delta=1.e-9)
        mean, error = map(float, integral)
        self.assertLess(abs(mean - (2 + 1.e-9)), 5 * error)

    def test_unvisited_channel_keeps_its_previous_moment(self):
        moment, calls = self.run_case('history')
        self.assertEqual(list(map(float, moment)), [4., 9.])
        self.assertEqual(list(map(int, calls)), [17, 5])

    def test_first_automatic_iteration_still_probes_each_channel(self):
        self.run_case('pilot')

    def test_mc_channel_selection_keeps_legacy_probabilities(self):
        counts, = self.run_case('mc')
        self.assertAlmostEqual(int(counts[0]) / sum(map(int, counts)), .875, delta=.006)

    def test_zero_moments_give_uniform_exploration(self):
        probability, = self.run_case('zero')
        self.assertEqual(list(map(float, probability)), [.5, .5])

    def test_automatic_pilot_updates_both_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.run_case('automatic', directory)
            mean, error, second = map(float, result)
            self.assertLess(abs(mean - 3.), 5 * error)
            self.assertLess(abs(second - 1.5), .1)

    def test_restart_recovers_a_channel_with_zero_integral(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as directory:
                self.run_case('seed', directory)
                grid = pathlib.Path(directory) / 'mint_grids'
                if legacy:
                    grid.write_text(''.join(' '.join(line.split()[:4]) + '\n'
                                            if line.split()[0] == 'SPE' else line
                                            for line in grid.read_text().splitlines(True)))
                result, moments = self.run_case('recover', directory)
                mean, error, recovered = map(float, result)
                # Two channels each integrate to 1.5, with split multiplier .25.
                self.assertLess(abs(mean - .75), 5 * error)
                self.assertGreater(recovered, .3)
                # Stored moments exclude both channel probability and wgt_mult.
                self.assertTrue(all(2.1 < float(moment) < 2.6 for moment in moments))
                fields = [line.split() for line in grid.read_text().splitlines()
                          if line.split()[0] == 'SPE']
                self.assertTrue(all(len(row) == 6 and int(row[5]) > 0 for row in fields))


SUPPORT = '''
module FKSParams
logical :: use_poly_virtual=.false.
double precision :: virt_fraction=1d0,min_virt_fraction=.005d0
end module

double precision function ran2()
implicit none
call random_number(ran2)
end function
'''

# Plotting, virtual-fit and FKS-integer hooks are irrelevant to these integrands.
for _name in ('accum', 'HwU_accum_iter', 'HwU_add_points', 'initplot',
              'regrid_MC_integer', 'empty_MC_integer', 'reset_MC_grid',
              'init_polyfit', 'get_polyfit', 'add_point_polyfit', 'do_polyfit',
              'save_polyfit', 'restore_polyfit'):
    SUPPORT += '\nsubroutine %s\nend subroutine\n' % _name


PROGRAM = '''
module integrand_state
logical :: seed_only=.false.
end module

program check_mint
use mint_module
use integrand_state
implicit none
character(len=16) :: mode
integer :: i,j,nseed,counts(3)
integer, allocatable :: seed(:)
double precision :: estimate,total,total2,weight,point,error,u
double precision, external :: integrand
call get_command_argument(1,mode)
call random_seed(size=nseed)
allocate(seed(nseed))
seed=12345
call random_seed(put=seed)
fixed_order=.true.
nlo_ps=.false.
nchans=2
iconfigs(1:3)=[1,2,3]
imode=0
nit=2
double_events=.false.
select case(mode)
case('rms')
 nchans=3
 ! The tiny channel has mean 1e-9 and RMS 1e-8: about 100% error
 ! after a pilot with 100 samples and one nonzero weight of 1e-7.
 channel_moment(1:3)=[1d0,100d0,1d-16]
 call set_channel_probabilities
 write(*,*) 'CHECK ',channel_probability(1:3)
 total=0d0
 total2=0d0
 do i=1,200000
  call get_channel
  point=1d0
  if(ichan.eq.2) then
   call random_number(u)
   point=0d0
   if(u.lt..01d0) point=100d0
  elseif(ichan.eq.3) then
   call random_number(u)
   point=0d0
   if(u.lt..01d0) point=1d-7
  endif
  weight=point/vol_chan
  total=total+weight
  total2=total2+weight**2
 enddo
 estimate=total/200000d0
 error=sqrt((total2/200000d0-estimate**2)/200000d0)
 write(*,*) 'CHECK ',estimate,error
case('history')
 channel_moment(1:2)=[4d0,100d0]
 channel_moment_calls(1:2)=[17,17]
 channel_calls(1:2)=[0,5]
 channel_squares(1:2)=[0d0,45d0]
 call update_channel_sampling
 write(*,*) 'CHECK ',channel_moment(1:2)
 write(*,*) 'CHECK ',channel_moment_calls(1:2)
case('pilot')
 nit=1
 double_events=.true.
 channel_moment(1:2)=[0d0,100d0]
 call set_channel_probabilities
 do j=1,2
  ans_chan=0d0
  ans_chan(0)=1d0
  ans_chan(j)=1d0
  do i=1,100
   call get_channel
   if(ichan.ne.j.or.vol_chan.ne.1d0) stop 1
  enddo
 enddo
case('mc')
 fixed_order=.false.
 channel_moment(1:2)=[1d0,100d0]
 call set_channel_probabilities
 ans_chan(0:2)=[8d0,7d0,1d0]
 counts=0
 do i=1,100000
  call get_channel
  counts(ichan)=counts(ichan)+1
  if(abs(vol_chan-ans_chan(ichan)/8d0).gt.1d-15) stop 2
 enddo
 write(*,*) 'CHECK ',counts(1:2)
case('zero')
 channel_moment=0d0
 call set_channel_probabilities
 write(*,*) 'CHECK ',channel_probability(1:2)
case('seed','recover','automatic')
 seed_only=mode.eq.'seed'
 imode=0
 wgt_mult=.5d0
 if(mode.eq.'recover') then
  imode=-1
  wgt_mult=.25d0
 endif
 ndim=4
 ncalls0=20000
 itmax=4
 accuracy=0d0
 n_ord_virt=0
 only_virt=.false.
 ifold=1
 ifold_energy=1
 ifold_yij=1
 ifold_phi=1
 min_virt_fraction_mint=.005d0
 if(mode.eq.'automatic') then
  wgt_mult=1d0
  ncalls0=-1
  accuracy=.01d0
  itmax=6
 endif
 call mint(integrand)
 write(*,*) 'CHECK ',ans(2,0),unc(2,0),ans(2,2)
 write(*,*) 'CHECK ',channel_moment(1:2)
end select
end program

double precision function integrand(x,vol,ifl,f)
use mint_module, only: ndimmax,nintegrals,ichan,pass_cuts_check
use integrand_state
implicit none
integer :: ifl
double precision :: x(ndimmax),vol,f(nintegrals),point
point=1d0+x(1)
if(seed_only.and.ichan.eq.2) point=0d0
f=0d0
f(1)=point*vol
f(2)=f(1)
pass_cuts_check=.true.
integrand=f(2)
end function
'''
