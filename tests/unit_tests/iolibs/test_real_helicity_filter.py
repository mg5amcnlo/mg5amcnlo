"""Numerical checks of the generated real-emission helicity loop."""

import copy
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from madgraph.iolibs import born_support as support
from madgraph.core.base_objects import ModelVariable, ParamCardVariable
from models import import_ufo


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestRealHelicityFilter(unittest.TestCase):
    def test_accidental_equalities_and_model_changes(self):
        self.run_driver(4, 2, DRIVER)

    def test_split_order_cancellation_does_not_prune_helicity(self):
        self.run_driver(2, 3, MIXED_DRIVER)

    def test_coherent_strong_scaling_preserves_learned_mask(self):
        self.run_driver(4, 2, SCALING_DRIVER, SCALING_MEMBERS, SCALING_SIGNATURE)

    def test_generated_helicity_state_comparator(self):
        self.run_driver(4, 2, COMPARATOR_DRIVER, SCALING_MEMBERS,
                        SCALING_SIGNATURE, include_real=False)

    def test_matrix_scaling_certifies_each_squared_order(self):
        model = import_ufo.import_model('sm')
        external = dict(mothers=[], coupling=[], interaction_id=0)

        class Matrix:
            def __init__(self, couplings, groups):
                self.couplings, self.groups = couplings, groups

            def get_used_couplings(self, output):
                return set(self.couplings)

            def get_used_lorentz(self):
                return []

            def get_all_amplitudes(self):
                return [dict(number=i, mothers=[external], coupling=[name],
                             interaction_id=1)
                        for i, name in enumerate(self.couplings, 1)]

            def get_split_orders_mapping(self):
                return [], self.groups

        cases = [
            (['GC_10', 'GC_12'], [((1,), (1,)), ((2,), (2,))], True),
            # One amplitude order cannot hide distinct G powers.
            (['GC_10', 'GC_12'], [((1,), (1, 2))], False),
            # Individually homogeneous amplitude orders can still collide
            # in a squared-order slot with different total G degrees.
            (['GC_10', 'GC_10', 'GC_12'],
             [((0,), (1,)), ((1,), (2,)), ((2,), (3,))], False),
            (['UNKNOWN'], [((1,), (1,))], False),
        ]
        for couplings, groups, expected in cases:
            with self.subTest(couplings=couplings, groups=groups):
                metadata = support.helicity_matrix_metadata(Matrix(couplings, groups), model)
                self.assertEqual(metadata['homogeneous'], expected)

    def test_prefixed_strong_input_is_not_an_exact_mask_key(self):
        model = dict(parameters={('external',): [
            ParamCardVariable('mdl_aS', .118, 'SMINPUTS', [3]),
            ParamCardVariable('mass', 10., 'MASS', [6]),
            ParamCardVariable('MU_R', 91., 'LOOP', [1]),
        ]})
        members = [('g', 'real', '1'), ('mass', 'real', '2'),
                   ('mdl_as', 'real', '3'), ('mu_r', 'real', '4'),
                   ('gc_10', 'complex', '1')]
        records = [dict(directory='P0', helicity_matrices=[
            dict(homogeneous=True, couplings={'gc_10': 1})])]
        with tempfile.TemporaryDirectory(prefix='mg5_helicity_signature_') as tmp:
            root = Path(tmp)
            (root/'Source/DHELAS').mkdir(parents=True)
            sub = root/'SubProcesses/P0'
            sub.mkdir(parents=True)
            (sub/'born.f').write_text(
                "      subroutine born\n      include 'coupl.inc'\n"
                "      call vertex(gc_10,mass)\n      end\n")
            signature = support.helicity_state_signature(root, records, model, members)
        self.assertIsNotNone(signature)
        self.assertEqual(signature['real'], [2])
        self.assertEqual(signature['couplings'], [('complex', 1, 1)])

    def test_sm_and_heft_coupling_degrees(self):
        # Legacy HEFT assigns QCD order zero to GH, despite GH being
        # proportional to G**2. Keep this fixture local: CI need not download
        # the optional HEFT model to verify its dependency structure.
        heft = dict(parameters={
            ('external',): [ParamCardVariable('aS', .118, 'SMINPUTS', [3]),
                            ParamCardVariable('vev', 246., 'TEST', [1]),
                            ParamCardVariable('mass', 125., 'TEST', [2])],
            ('aS',): [ModelVariable('sqrt_aS', 'cmath.sqrt(aS)', 'real'),
                      ModelVariable('G', '2*sqrt_aS*cmath.sqrt(cmath.pi)', 'real'),
                      ModelVariable('G_squared', 'G**2', 'real'),
                      ModelVariable('GH', '-G_squared*(1+mass**2)/(12*cmath.pi**2*vev)', 'real')],
        }, couplings={('aS',): [
            ModelVariable('GC_13', '-complex(0,1)*GH', 'complex'),
            ModelVariable('GC_14', '-G*GH', 'complex'),
            ModelVariable('GC_15', 'complex(0,1)*G_squared*GH', 'complex'),
            ModelVariable('MIXED', 'G+G_squared', 'complex'),
            ModelVariable('RUNNING', 'G*cmath.log(G)', 'complex'),
        ]})
        prefixed_heft = copy.deepcopy(heft)
        for group in ('parameters', 'couplings'):
            for values in prefixed_heft[group].values():
                for variable in values:
                    if variable.name == 'aS':
                        variable.name = 'mdl_aS'
                    if hasattr(variable, 'expr'):
                        variable.expr = re.sub(r'\baS\b', 'mdl_aS', variable.expr)
        cases = [
            ('sm', import_ufo.import_model('sm'),
             {'gc_10': 1, 'gc_11': 1, 'gc_12': 2}),
            ('heft', heft, {'gc_13': 2, 'gc_14': 3, 'gc_15': 4,
                            'mixed': None, 'running': None}),
            ('prefixed_heft', prefixed_heft,
             {'gc_13': 2, 'gc_14': 3, 'gc_15': 4,
              'mixed': None, 'running': None}),
        ]
        for model_name, model, couplings in cases:
            degrees = {name.lower(): degree for name, degree in
                       support.helicity_coupling_degrees(model).items()}
            for name, degree in couplings.items():
                with self.subTest(model=model_name, coupling=name):
                    self.assertEqual(degrees[name], degree)

    def run_driver(self, ncomb, nsqamps, driver, members=None, signature=None,
                   include_real=True):
        template = (ROOT/'madgraph/iolibs/template_files/'
                    'realmatrix_splitorders_fks.inc').read_text()
        start = template.index('      SUBROUTINE SMATRIX%(proc_prefix)s_SPLITORDERS')
        end = template.index('      SUBROUTINE MATRIX_%(proc_prefix)s', start)
        routine = template[start:end] % dict(
            proc_prefix='1', info_lines='', process_lines='', ncomb=ncomb,
            nSqAmpSplitOrders=nsqamps,
            helicity_lines='      DATA NHEL /%s/' % ','.join(map(str, range(1, ncomb+1))),
            den_factor_line='      PARAMETER (IDEN=2)')
        lines = support.optimize_real_helicities(list(support.statements(routine)))
        with tempfile.TemporaryDirectory(prefix='mg5_real_helicities_') as tmp:
            work = Path(tmp)
            (work/'nexternal.inc').write_text(
                '      INTEGER NEXTERNAL\n      PARAMETER(NEXTERNAL=1)\n')
            support.write_fortran(work/'real.f', lines)
            module = work/'mc_born_types.f90'
            shutil.copyfile(ROOT/'Template/NLO/Source/BornSupport/mc_born_types.f90',
                            module)
            if signature is not None:
                support.write_helicity_state_comparator(module, members, signature)
            (work/'check.f90').write_text(driver)
            sources = [str(work/'real.f')] if include_real else []
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all',
                '-ffixed-line-length-none', '-I'+str(work),
                str(module), *sources, str(work/'check.f90'), '-o', 'check'],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            result = subprocess.run([str(work/'check')], cwd=work,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('PASS real helicity filtering', result.stdout)


DRIVER = '''\
program check
  implicit none
  real(8) width,p(0:3,1)
  complex(8) coupling
  common/test_model/coupling,width
  p=0d0
  width=0d0
  coupling=(1d0,0d0)
  ! Helicity 1 and 2 agree here in every split order, but nowhere else.
  p(0,1)=1d0
  call check_point(p,4)
  p(0,1)=2d0
  call check_point(p,2)
  call check_point(p,2)
  ! Real model parameters can activate helicities that were exactly zero.
  width=2d0
  call check_point(p,4)
  call check_point(p,3)
  ! Complex model changes must relearn the mask too, including zero -> nonzero.
  coupling=(0d0,0d0)
  call check_point(p,4)
  call check_point(p,1)
  coupling=(0.7d0,0.2d0)
  call check_point(p,4)
  call check_point(p,3)
  width=0d0
  call check_point(p,4)
  call check_point(p,2)
  print *, 'PASS real helicity filtering'
end program

subroutine check_point(p,expected_calls)
  use,intrinsic::ieee_arithmetic
  implicit none
  real(8) p(0:3,1),ans(0:2),reference(0:2),term(0:2)
  integer expected_calls,calls,observed_calls,i
  common/test_calls/calls
  calls=0
  call smatrix1_splitorders(p,ans)
  observed_calls=calls
  ! The oracle evaluates all helicities directly, with no learned mask.
  reference=0d0
  do i=1,4
    call matrix_1(p,[i],term)
    reference=reference+term/2d0
  enddo
  if (.not.all(ieee_is_finite(ans))) stop 1
  if (any(abs(ans-reference)>1d-13*max(1d0,abs(reference)))) then
    print *, 'Incorrect helicity sum',ans,reference
    stop 2
  endif
  if (observed_calls/=expected_calls) then
    print *, 'Incorrect helicity work',observed_calls,expected_calls
    stop 3
  endif
end subroutine

subroutine matrix_1(p,nhel,t)
  implicit none
  real(8) p(0:3,1),t(0:2),width,norm
  complex(8) coupling
  integer nhel(1),calls
  common/test_model/coupling,width
  common/test_calls/calls
  calls=calls+1
  norm=real(coupling*conjg(coupling),8)
  t=0d0
  select case(nhel(1))
  case(1)
    t(1)=norm*(p(0,1)+1d0)**2
    t(2)=norm*(p(0,1)+2d0)**2
  case(2)
    t(1)=norm*(3d0-p(0,1))**2
    t(2)=norm*(4d0-p(0,1))**2
  case(3)
    t(1)=width**2
  end select
  t(0)=sum(t(1:2))
end subroutine

subroutine mc_capture_model_state(state)
  use mc_born_types
  implicit none
  type(BornModelState) state
  real(8) width
  complex(8) coupling
  common/test_model/coupling,width
  call born_resize_model_state(state,1,1)
  state%real_values=[width]
  state%complex_values=[coupling]
end subroutine
'''


MIXED_DRIVER = '''\
program check
  implicit none
  real(8) p(0:3,1),ans(0:3),reference(0:3)
  integer calls
  common/test_calls/calls
  p=0d0
  p(0,1)=1d0
  calls=0
  call smatrix1_splitorders(p,ans)
  if (calls/=2) stop 1
  reference=[0d0,1d0,-2d0,1d0]/2d0
  if (any(abs(ans-reference)>1d-13)) stop 2
  ! The first helicity has nonzero split orders despite a vanishing sum.
  ! A second momentum removes the cancellation without changing the model.
  p(0,1)=2d0
  calls=0
  call smatrix1_splitorders(p,ans)
  reference=[1d0,1d0,-4d0,4d0]/2d0
  if (any(abs(ans-reference)>1d-13)) then
    print *, 'Lost helicity through split-order cancellation',ans,reference
    stop 3
  endif
  if (calls/=1) stop 4
  print *, 'PASS real helicity filtering'
end program

subroutine matrix_1(p,nhel,t)
  implicit none
  real(8) p(0:3,1),t(0:3)
  integer nhel(1),calls
  common/test_calls/calls
  calls=calls+1
  t=0d0
  if (nhel(1)==1) then
    t(1:3)=[1d0,-2d0*p(0,1),p(0,1)**2]
    t(0)=sum(t(1:3))
  endif
end subroutine

subroutine mc_capture_model_state(state)
  use mc_born_types
  implicit none
  type(BornModelState) state
  call born_resize_model_state(state,1,1)
  state%real_values=[1d0]
  state%complex_values=[(1d0,0d0)]
end subroutine
'''


SCALING_MEMBERS = [
    ('g', 'real', '1'), ('mass', 'real', '2'), ('mu_r', 'real', '3'),
    ('as', 'real', '4'), ('g_squared', 'real', '5'),
    ('gc1', 'complex', '1'), ('gc2', 'complex', '2'),
    ('weak', 'complex', '3'), ('unused_ct', 'complex', '4'),
]
SCALING_SIGNATURE = dict(g=1, real=[2], complex=[3],
                         couplings=[('complex', 1, 1), ('complex', 2, 2)])


SCALING_DRIVER = '''\
program check
  use mc_born_types
  implicit none
  real(8) g,mass,mu_r,p(0:3,1)
  complex(8) gc1,gc2
  type(BornModelState) previous,current
  common/test_model/gc1,gc2,g,mass,mu_r
  p=0d0
  p(0,1)=1d0
  g=1d0
  gc1=(1d0,0d0)
  gc2=(0d0,1d0)
  mass=0d0
  mu_r=91d0
  call check_point(p,4)
  call check_point(p,1)
  call mc_capture_model_state(previous)
  ! Coherent running changes amplitudes but preserves the learned mask.
  g=2d0
  gc1=2d0*gc1
  gc2=4d0*gc2
  mu_r=180d0
  call mc_capture_model_state(current)
  if (born_model_state_equal(previous,current)) stop 10
  if (.not.born_helicity_state_equal(previous,current)) stop 11
  call check_point(p,1)
  ! Relative coupling changes can activate a previously vanishing helicity.
  gc1=gc1+1d0
  call check_point(p,4)
  call check_point(p,2)
  mass=10d0
  call check_point(p,4)
  call check_point(p,3)
  ! A zero strong coupling needs a new mask on entry and on exit.
  g=0d0
  gc1=(0d0,0d0)
  gc2=(0d0,0d0)
  call check_point(p,4)
  call check_point(p,0)
  g=2d0
  gc1=(2d0,0d0)
  gc2=(0d0,4d0)
  call check_point(p,4)
  call check_point(p,2)
  print *, 'PASS real helicity filtering'
end program

subroutine matrix_1(p,nhel,t)
  implicit none
  real(8) p(0:3,1),t(0:2),g,mass,mu_r
  complex(8) gc1,gc2
  integer nhel(1),calls
  common/test_model/gc1,gc2,g,mass,mu_r
  common/test_calls/calls
  calls=calls+1
  t=0d0
  select case(nhel(1))
  case(1)
    t(1)=abs(gc1)**2*(1d0+p(0,1))
    t(2)=abs(gc2)**2
  case(2)
    t(1)=mass**2*abs(gc1)**2
  case(3)
    t(2)=abs(gc1**2+(0d0,1d0)*gc2)**2
  end select
  t(0)=sum(t(1:2))
end subroutine

subroutine mc_capture_model_state(state)
  use mc_born_types
  implicit none
  type(BornModelState) state
  real(8) g,mass,mu_r
  complex(8) gc1,gc2
  common/test_model/gc1,gc2,g,mass,mu_r
  call born_resize_model_state(state,5,4)
  state%real_values=[g,mass,mu_r,g**2/(4d0*acos(-1d0)),g**2]
  state%complex_values=[gc1,gc2,(0.5d0,0d0),cmplx(mu_r,0d0,8)]
end subroutine
''' + DRIVER[DRIVER.index('subroutine check_point'):DRIVER.index('subroutine matrix_1')]


COMPARATOR_DRIVER = '''\
program check
  use mc_born_types
  use,intrinsic::ieee_arithmetic
  implicit none
  type(BornModelState) a,b
  call born_resize_model_state(a,5,4)
  a%real_values=[1d0,10d0,91d0,0.1d0,1d0]
  a%complex_values=[(1d0,0d0),(0d0,1d0),(0.5d0,0d0),(999d0,0d0)]
  b=a
  if (.not.born_helicity_state_equal(a,b)) stop 1
  b%real_values=[2d0,10d0,180d0,0.4d0,4d0]
  b%complex_values=[(2d0,0d0),(0d0,4d0),(0.5d0,0d0),(-22d0,3d0)]
  if (.not.born_helicity_state_equal(a,b)) stop 2
  if (born_model_state_equal(a,b)) stop 3
  b%complex_values(1)=b%complex_values(1)*(0d0,1d0)
  if (born_helicity_state_equal(a,b)) stop 4
  b=a
  b%real_values(2)=11d0
  if (born_helicity_state_equal(a,b)) stop 5
  b=a
  b%complex_values(3)=1d0
  if (born_helicity_state_equal(a,b)) stop 6
  b=a
  b%real_values(1)=0d0
  b%complex_values(1:2)=0d0
  if (born_helicity_state_equal(a,b)) stop 7
  if (born_helicity_state_equal(b,a)) stop 8
  b=a
  b%complex_values(1)=0d0
  if (born_helicity_state_equal(a,b)) stop 9
  if (born_helicity_state_equal(b,a)) stop 10
  a%complex_values(1)=0d0
  b=a
  b%complex_values(1)=1d-200
  if (born_helicity_state_equal(a,b)) stop 11
  b=a
  b%complex_values(2)=cmplx(ieee_value(1d0,ieee_quiet_nan),0d0,8)
  if (born_helicity_state_equal(a,b)) stop 12
  b=a
  b%real_values(1)=ieee_value(1d0,ieee_positive_inf)
  if (born_helicity_state_equal(a,b)) stop 13
  b=a
  b%real_values(1)=-1d0
  if (born_helicity_state_equal(a,b)) stop 14
  ! Finite G can still produce an overflowing/underflowing power. Such
  ! normalization must not hide a relative change of the degree-two coupling.
  a%real_values(1)=1d200
  a%complex_values(1)=1d200
  a%complex_values(2)=(0d0,1d100)
  b=a
  b%real_values(1)=2d200
  b%complex_values(1)=2d200
  b%complex_values(2)=(0d0,2d100)
  if (born_helicity_state_equal(a,b)) stop 15
  a%real_values(1)=1d-200
  a%complex_values(1)=1d-200
  a%complex_values(2)=(0d0,1d-300)
  b=a
  b%real_values(1)=2d-200
  b%complex_values(1)=2d-200
  b%complex_values(2)=(0d0,2d-300)
  if (born_helicity_state_equal(a,b)) stop 16
  print *, 'PASS real helicity filtering'
end program
'''
