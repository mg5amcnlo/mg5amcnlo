"""Compile the production hard-weight and damping routines against controlled MEs."""

import pathlib
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_real_channels_phase_space import fortran_routine


ROOT = pathlib.Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TestRealChannelsWeights(unittest.TestCase):
    """Independent weights expose ownership, coupling and normalization mistakes."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='mg5-real-weights-')
        cls.path = pathlib.Path(cls.directory.name)
        includes = {
            'nexternal.inc': 'integer nexternal,nincoming\nparameter(nexternal=5,nincoming=2)',
            'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=3)',
            'run.inc': 'double precision real_channels_x0\ncommon /c_real_channels_x0/ real_channels_x0',
            'coupl.inc': 'double precision g,testmass(5)\ncommon /test_coupl/ g,testmass',
            'pmass.inc': 'pmass=testmass',
            'fks_info.inc': '',
            'orders.inc': '''integer nsplitorders,amp_split_size,qcd_pos
parameter(nsplitorders=2,amp_split_size=2,qcd_pos=1)
double precision amp_split(amp_split_size)
common /to_amp_split/ amp_split''',
            # Channels 1 and 3 share ME1; channel 2 and sector 2 belong to ME2.
            # The huge ME2 diagram weight must never dilute ME1's partition.
            'real_channels.inc': '''integer n_real_channels
parameter(n_real_channels=3)
integer real_channel_me(3),real_channel_fks(3)
integer real_channel_config(3),real_channel_diagram(3),real_sector_me(3)
data real_channel_me /1,2,1/
data real_channel_fks /1,2,1/
data real_channel_config /1,1,2/
data real_channel_diagram /1,3,2/
data real_sector_me /1,2,1/''',
            'real_matrix_weights.inc': '''double precision real_amp2(3)
common /c_real_diagram_weights/ real_amp2
logical get_real_diagram_weights
common /c_get_real_diagram_weights/ get_real_diagram_weights''',
        }
        for name, content in includes.items():
            (cls.path / name).write_text(''.join('      ' + line + '\n'
                                               for line in content.splitlines()))
        driver = (ROOT / 'Template/NLO/SubProcesses/driver_mintFO.f').read_text()
        singular = (ROOT / 'Template/NLO/SubProcesses/fks_singular.f').read_text()
        kin = (ROOT / 'Template/NLO/Source/kin_functions.f').read_text()
        production = fortran_routine(driver, 'compute_hard_real_channels')
        production += fortran_routine(singular, 'real_channel_soft_fraction')
        production += fortran_routine(singular, 'real_channel_invariant')
        production += fortran_routine(kin, 'dot')
        (cls.path / 'production.f').write_text(production)
        (cls.path / 'support.f90').write_text(SUPPORT)
        (cls.path / 'check.f90').write_text(PROGRAM)
        command = ['gfortran', '-O0', '-g', '-fcheck=all',
                   '-ffixed-line-length-none', '-ffree-line-length-none',
                   '-fallow-argument-mismatch', '-o', 'check',
                   'support.f90', 'production.f', 'check.f90']
        result = subprocess.run(command, cwd=cls.path, capture_output=True, text=True)
        if result.returncode:
            cls.directory.cleanup()
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_case(self, *arguments):
        result = subprocess.run([str(self.path / 'check')] + list(map(str, arguments)),
                                cwd=self.path, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
        return rows

    def weights(self, channel, sample='normal', theta=1.e-12):
        rows = self.run_case('weights', channel, sample, theta)
        summary = next(row for row in rows if row[0] == 'summary')
        records = [([int(value) for value in row[1:7]],
                    [float(value) for value in row[7:]])
                   for row in rows if row[0] == 'record']
        soft = {int(row[1]): float(row[2]) for row in rows if row[0] == 'soft'}
        return [int(value) for value in summary[1:]], records, soft

    def test_owned_sectors_and_cached_coupling_records(self):
        summary, records, _ = self.weights(1)
        self.assertEqual(summary, [1, 4, 0, 1])  # one raw ME call, four records, flag reset
        self.assertEqual([item[0] for item in records],
                         [[1, 1, 2, 3, 1, 32], [1, 1, 4, 5, 2, 54],
                          [1, 3, 2, 3, 1, 32], [1, 3, 4, 5, 2, 54]])
        # raw/g**QCD = 2 or 3; jac=7, vegas=11, partition=1/4, hard=4/5.
        for (tags, weights), expected in zip(records, [12.32, 18.48, 27.72, 41.58]):
            self.assertAlmostEqual(weights[0], expected, places=11)
            self.assertEqual(weights[1:], [0.0, 0.0])
        summary, records, _ = self.weights(2)
        self.assertEqual(summary[0:2], [1, 2])
        self.assertEqual([record[0][1] for record in records], [2, 2])

    def test_real_diagram_partition_normalizes_within_one_me(self):
        _, first, _ = self.weights(1)
        _, second, _ = self.weights(3)
        for (_, left), (_, right) in zip(first, second):
            self.assertAlmostEqual(right[0], 3 * left[0], places=11)
        totals = [sum(records[index][1][0] for records in (first, second)
                      for index in (amp, amp + 2)) for amp in (0, 1)]
        self.assertAlmostEqual(totals[0], 160.16, places=11)
        self.assertAlmostEqual(totals[1], 240.24, places=11)

    def test_zero_diagram_sum_has_uniform_nonzero_partition(self):
        _, first, _ = self.weights(1, 'zero_diagrams')
        _, second, _ = self.weights(3, 'zero_diagrams')
        self.assertEqual(len(first), 4)
        for (_, left), (_, right), expected in zip(first, second, [24.64, 36.96, 55.44, 83.16]):
            self.assertAlmostEqual(left[0], expected, places=11)
            self.assertAlmostEqual(right[0], expected, places=11)

    def test_zero_selected_diagram_does_not_create_weight_records(self):
        summary, records, _ = self.weights(1, 'one_zero')
        self.assertEqual(summary[0], 1)
        self.assertEqual(records, [])
        _, records, _ = self.weights(3, 'one_zero')
        self.assertAlmostEqual(sum(weights[0] for tags, weights in records), 400.4, places=10)

    def test_soft_plus_hard_reconstructs_full_local_real_weight(self):
        _, first, soft = self.weights(1)
        _, second, _ = self.weights(3)
        for amp, raw_over_coupling in enumerate((2., 3.)):
            hard = sum(records[index][1][0] for records in (first, second)
                       for index in (amp, amp + 2))
            nonhard = raw_over_coupling * 7 * 11 * (2 * .2 * soft[1] + 3 * .3 * soft[3])
            full = raw_over_coupling * 7 * 11 * (2 * .2 + 3 * .3)
            self.assertAlmostEqual(hard + nonhard, full, places=11)

    def test_tiny_collinear_invariant_has_no_legacy_dot_cutoff(self):
        for theta in (1.e-4, 1.e-12, 1.e-25):
            values = self.run_case('invariant', theta, .1, 'massless')[0]
            invariant, old_dot, soft = map(float, values[1:])
            self.assertGreater(invariant, 0.)
            self.assertEqual(old_dot, 0.)
            self.assertAlmostEqual(invariant / (5 * theta**2), 1., delta=1.e-7)
            summary, records, _ = self.weights(1, 'tiny', theta)
            self.assertEqual(summary[0:2], [1, 4])
            first_sector_weight = records[0][1][0]
            self.assertGreater(first_sector_weight, 0.)
            self.assertAlmostEqual(first_sector_weight / (15.4 * theta**4), 1., delta=1.e-7)

    def test_exact_endpoints_cuts_and_invalid_maps_skip_matrix_element(self):
        # foreign_endpoint is singular in a sector owned by the other ME.
        for sample in ('endpoint', 'foreign_endpoint', 'cut', 'invalid_map'):
            summary, records, _ = self.weights(1, sample)
            self.assertEqual(summary[0:2], [0, 0])
            self.assertEqual(records, [])

    def test_massive_and_rest_frame_invariants(self):
        for scenario, expected in (('massive', 29.), ('rest', 52.)):
            values = self.run_case('invariant', 0., .1, scenario)[0]
            invariant, old_dot, soft = map(float, values[1:])
            self.assertAlmostEqual(invariant, expected, places=13)
            self.assertAlmostEqual(old_dot, expected, places=13)
            x = 2 * expected / 100.
            self.assertAlmostEqual(soft, 1 / (1 + (x / .1)**2), places=13)

    def test_disabled_damping_is_exactly_one(self):
        for x0 in (-1., 0., 1., 2.):
            values = self.run_case('invariant', .5, x0, 'massless')[0]
            self.assertEqual(float(values[3]), 1.)


SUPPORT = r'''
module extra_weights
integer, parameter :: cpower_pos=2
integer :: QCD_power,orders_tag,amp_pos
double precision :: wgtcpower
end module

module mint_module
logical :: pass_cuts_check=.false.
end module

module test_state
integer :: me_calls=0,nrecords=0,sector_now=0
integer :: tags(6,20)=0
double precision :: recorded(3,20)=0d0,theta=1d-12
character(len=32) :: sample='normal'
integer i_fks,j_fks
common /fks_indices/ i_fks,j_fks
double precision fkssymmetryfactor,fkssymmetryfactorBorn,fkssymmetryfactorDeg
integer ngluons,nquarks(-6:6),nphotons
common /numberofparticles/ fkssymmetryfactor,fkssymmetryfactorBorn, &
                          fkssymmetryfactorDeg,ngluons,nquarks,nphotons
double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
common /parton_cms_stuff/ ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
contains
subroutine point(p)
double precision p(0:3,5)
p(:,1)=[5d0,0d0,0d0,5d0]
p(:,2)=[5d0,0d0,0d0,-5d0]
p(:,3)=[3d0,-3d0,0d0,0d0]
p(:,4)=[3d0,0d0,3d0,0d0]
p(:,5)=[2d0,2d0,0d0,0d0]
if(sample=='tiny') p(:,5)=[2d0,2d0*theta,0d0,2d0]
if(sample=='endpoint') p(:,5)=[2d0,0d0,0d0,2d0]
if(sample=='foreign_endpoint') p(:,4)=p(:,3)
end subroutine
end module

subroutine initialize_weight_analysis
use test_state
nrecords=0
end subroutine

subroutine update_fks_dir(sector)
use test_state
implicit none
include 'orders.inc'
integer sector,fks_j_from_i(5,0:5),particle_type(5),pdg_type(5)
common /c_fks_inc/ fks_j_from_i,particle_type,pdg_type
sector_now=sector
i_fks=5
j_fks=1
fkssymmetryfactor=2d0
if(sector==2) then
 i_fks=4
 j_fks=3
 fkssymmetryfactor=100d0
elseif(sector==3) then
 j_fks=2
 fkssymmetryfactor=3d0
endif
! Deliberately invalidate the shared buffer: the hard routine must retain
! the raw split amplitudes across sector changes.
amp_split=-999d0
fks_j_from_i=0
fks_j_from_i(5,0)=2
fks_j_from_i(5,1)=1
fks_j_from_i(5,2)=2
fks_j_from_i(4,0)=1
fks_j_from_i(4,1)=3
end subroutine

subroutine generate_momenta_real_channels(ndim,config,fks,x,p,jac)
use test_state
implicit none
integer ndim,config,fks
double precision x(99),p(0:3,5),jac
call point(p)
jac=7d0
if(sample=='invalid_map') jac=-1d0
end subroutine

subroutine set_cms_stuff(dummy)
use test_state
integer dummy
sqrtshat=10d0
shat=100d0
end subroutine

logical function passcuts(p,rwgt)
use test_state
double precision p(0:3,5),rwgt
rwgt=1d0
passcuts=sample/='cut'
end function

subroutine set_alphaS(p)
double precision p(0:3,5)
end subroutine

subroutine smatrix_real(p,total)
use test_state
implicit none
include 'orders.inc'
include 'real_matrix_weights.inc'
double precision p(0:3,5),total
if(.not.get_real_diagram_weights) stop 71
me_calls=me_calls+1
amp_split=[8d0,48d0]
total=sum(amp_split)
real_amp2=[1d0,3d0,10000d0]
if(sample=='zero_diagrams') real_amp2=0d0
if(sample=='one_zero') real_amp2(1)=0d0
end subroutine

double precision function fks_Sij(p,i,j,xi,y)
use test_state
integer i,j
double precision p(0:3,5),xi,y
if(i/=i_fks.or.j/=j_fks) stop 72
if(abs(y)>1d0.or.xi<=0d0) stop 73
if(sector_now==1) fks_Sij=.2d0
if(sector_now==2) fks_Sij=.9d0
if(sector_now==3) fks_Sij=.3d0
end function

subroutine amp_split_pos_to_orders(iamp,orders)
integer iamp,orders(2)
orders=[2*iamp,2*iamp+1]
end subroutine

integer function get_orders_tag(orders)
integer orders(2)
get_orders_tag=orders(1)+10*orders(2)
end function

subroutine add_wgt(kind,orders,w1,w2,w3)
use extra_weights
use test_state
integer kind,orders(2)
double precision w1,w2,w3
if(orders(1)/=QCD_power.or.dble(orders(2))/=wgtcpower) stop 74
nrecords=nrecords+1
tags(:,nrecords)=[kind,sector_now,QCD_power,int(wgtcpower),amp_pos,orders_tag]
recorded(:,nrecords)=[w1,w2,w3]
end subroutine
'''


PROGRAM = r'''
program check_real_weights
use test_state
use mint_module
implicit none
include 'run.inc'
include 'coupl.inc'
include 'real_matrix_weights.inc'
integer id,i
double precision x(99),p(0:3,5),value,old,soft
double precision real_channel_invariant,real_channel_soft_fraction,dot
external real_channel_invariant,real_channel_soft_fraction,dot
character(len=32) action,arg
call get_command_argument(1,action)
g=2d0
testmass=0d0
real_channels_x0=.1d0
get_real_diagram_weights=.false.
shat=100d0
sqrtshat=10d0
x=.5d0
if(action=='weights') then
 call get_command_argument(2,arg)
 read(arg,*) id
 call get_command_argument(3,sample)
 call get_command_argument(4,arg)
 read(arg,*) theta
 call compute_hard_real_channels(5,id,x,p,11d0)
 write(*,*) 'summary',me_calls,nrecords,merge(1,0,get_real_diagram_weights), &
                       merge(1,0,pass_cuts_check)
 do i=1,nrecords
  write(*,*) 'record',tags(:,i),recorded(:,i)
 enddo
 do i=1,3,2
  call update_fks_dir(i)
  write(*,*) 'soft',i,real_channel_soft_fraction(p)
 enddo
elseif(action=='invariant') then
 call get_command_argument(2,arg)
 read(arg,*) theta
 call get_command_argument(3,arg)
 read(arg,*) real_channels_x0
 call get_command_argument(4,arg)
 sample='tiny'
 call point(p)
 i_fks=5
 j_fks=1
 if(arg=='massive') then
  p(:,1)=[5d0,0d0,0d0,3d0]
  p(:,5)=[13d0,0d0,0d0,12d0]
  testmass(1)=4d0
  testmass(5)=5d0
 elseif(arg=='rest') then
  p(:,1)=[4d0,0d0,0d0,0d0]
  p(:,5)=[13d0,0d0,0d0,12d0]
  testmass(1)=4d0
  testmass(5)=5d0
 endif
 value=real_channel_invariant(p,i_fks,j_fks)
 old=dot(p(:,i_fks),p(:,j_fks))
 soft=real_channel_soft_fraction(p)
 write(*,*) 'invariant',value,old,soft
else
 stop 75
endif
end program
'''
