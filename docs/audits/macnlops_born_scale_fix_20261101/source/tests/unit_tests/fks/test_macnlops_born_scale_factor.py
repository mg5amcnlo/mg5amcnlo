"""Native Born-reference ratios with controlled scale-dependent amplitudes."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMacnlopsBornScaleFactor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='macnlops_born_scale_')
        cls.addClassCleanup(cls.tmp.cleanup)
        work = Path(cls.tmp.name)
        (work / 'genps.inc').write_text('')
        run = '''      double precision scale,muR2_current,muF12_current,
     $     muF22_current,QES2_current,q2fact(2),xbk(2),ebeam(2)
      integer lpp(2)
      common/test_run/scale,muR2_current,muF12_current,
     $     muF22_current,QES2_current,q2fact,xbk,ebeam,lpp
'''
        (work / 'run.inc').write_text(run)
        (work / 'orders.inc').write_text('''      integer,parameter :: amp_split_size=2,nsplitorders=2,
     $     qcd_pos=1
''')
        modules = '''module fks_phase_space_data
  implicit none
  real(8) :: xi_i_fks_ev=.2d0,y_ij_fks_ev=.1d0
  real(8) :: shat=17d0,sqrtshat=19d0,ybst_til_tocm=23d0,ybst_til_tolab=29d0
end module
module extra_weights
  integer,parameter :: cpower_pos=2
end module
module fixture_state
  use, intrinsic :: ieee_arithmetic
  implicit none
  real(8) :: scale,muR2_current,muF12_current,muF22_current,QES2_current
  real(8) :: q2fact(2),xbk(2),ebeam(2)
  integer :: lpp(2)
  common/test_run/scale,muR2_current,muF12_current,muF22_current,QES2_current,q2fact,xbk,ebeam,lpp
  real(8) :: model_coupling=1d0,born_cache=1d0,pdf_scales(4)=0d0
  integer :: pdf_calls=0,born_calls=0
  character(32) :: mode
end module
subroutine set_alphaS(p)
  use fixture_state
  use fks_phase_space_data,only:ybst_til_tolab
  implicit none
  real(8) :: p(0:3,5),mu
  mu=sum(abs(p(1,3:5)))
  if(mu<100d0)then
    if(mode=='ee')then
      if(abs(ybst_til_tolab)>1d-14)stop 22
    else if(mode=='rapidity')then
      if(abs(ybst_til_tolab+.34657359027997265d0)>1d-14)stop 23
    else
      if(abs(ybst_til_tolab-.34657359027997265d0)>1d-14)stop 24
    endif
  endif
  if(mode=='rapidity')mu=mu*exp(ybst_til_tolab)
  if(mode=='fixed')mu=120d0
  scale=mu;muR2_current=mu**2;QES2_current=(2*mu)**2
  muF12_current=(1.5d0*mu)**2;muF22_current=(5d0*mu)**2
  q2fact=[muF12_current,muF22_current]
  model_coupling=120d0/mu
end subroutine
subroutine set_cms_stuff(index)
  use fks_phase_space_data
  use fixture_state,only:xbk
  implicit none
  integer index
  if(index/=-100)stop 10
  shat=40000d0;sqrtshat=200d0;ybst_til_tocm=0d0;ybst_til_tolab=0d0
  xbk=[.1d0,.2d0]
end subroutine
subroutine amp_split_pos_to_orders(index,orders)
  use fixture_state,only:mode
  implicit none
  integer index,orders(2)
  orders=[2*index,0]
  if(mode=='mixed_cpower')orders(2)=index
end subroutine
real(8) function rwgt_muR_dep_fac(mu,reference,power)
  implicit none
  real(8) mu,reference,power
  if(mu/=reference)stop 11
  rwgt_muR_dep_fac=(mu/120d0)**power
end function
real(8) function pdg2pdf(beam,parton,side,x,mu)
  use fixture_state
  implicit none
  integer beam,parton,side
  real(8) x,mu
  if(abs(beam)/=1.or.x<=0d0.or.x>=1d0)stop 12
  if(side==1.and.parton/=0)stop 13
  if(side==2.and.parton/=-2)stop 14
  pdf_calls=pdf_calls+1;pdf_scales(pdf_calls)=mu
  pdg2pdf=(side+1d0)*mu**side
  if(mu<100d0.and.side==1)then
    if(mode=='zero_pdf')pdg2pdf=0d0
    if(mode=='negative_pdf')pdg2pdf=-1d0
    if(mode=='nan_pdf')pdg2pdf=ieee_value(1d0,ieee_quiet_nan)
  endif
end function
'''
        (work / 'fixture.f90').write_text(modules)
        helper = fortran_routine(TEMPLATE / 'montecarlocounter.f',
                                 'macnlops_born_scale_factor')
        stub = '''      subroutine get_mbar(p,xi,y,pborn,leg,flow,order,w,s)
      use fixture_state, only: model_coupling,born_cache,born_calls,
     $     mode
      use, intrinsic :: ieee_arithmetic
      implicit none
      double precision p(0:3,5),xi,y,pborn(0:3,4),w(2),s(2),g
      integer leg,flow,order
      if(leg.ne.3.or.flow.ne.2.or.order.ne.1)stop 15
      g=model_coupling
      born_cache=g
      born_calls=born_calls+1
      w=0d0
      s=0d0
      select case(trim(mode))
      case('power0','fixed','ee','zero_pdf','negative_pdf','nan_pdf')
         w(1)=3d0
      case('power1','rapidity')
         w(1)=g**2
      case('power2')
         w(1)=g**4
      case('effective')
         w(1)=g**6
      case('mixed')
         w(1)=(1d0+g*g)**2
      case('mixed_cpower')
         w(1)=1d0+g*g
         w(2)=2d0*g**4
      case('zero_born','negative_born','nan_born')
         w(1)=1d0
         if(g.gt.1.5d0)then
            if(mode.eq.'zero_born')w(1)=0d0
            if(mode.eq.'negative_born')w(1)=-1d0
            if(mode.eq.'nan_born')
     $           w(1)=ieee_value(1d0,ieee_quiet_nan)
         endif
      case default
         stop 16
      end select
      end subroutine get_mbar
'''
        (work / 'helper.f').write_text(
            '      module native_helper\n'
            '      implicit none\n'
            '      integer,parameter :: nexternal=5,nincoming=2,ileg=3\n'
            '      contains\n' + helper + '\n' + stub +
            '      end module native_helper\n')
        driver = '''program check
  use fixture_state
  use fks_phase_space_data
  use native_helper,only:macnlops_born_scale_factor
  implicit none
  real(8) :: p(0:3,5),pb(0:3,4),ratio,pdf,initial(13),after(13)
  integer :: status,i_fks,j_fks
  common/fks_indices/i_fks,j_fks
  call get_command_argument(1,mode)
  p=0d0;pb=0d0;p(0,1:2)=100d0;pb(0,1:2)=100d0
  p(3,1)=100d0;p(3,2)=-100d0;pb(3,1:2)=p(3,1:2)
  p(0,3:5)=[100d0,50d0,50d0];p(1,3:5)=[20d0,40d0,-60d0]
  pb(0,3:4)=100d0;pb(1,3:4)=[30d0,-30d0]
  scale=31d0;muR2_current=37d0;muF12_current=41d0
  muF22_current=43d0;QES2_current=47d0;q2fact=[53d0,59d0]
  xbk=[.31d0,.37d0];ebeam=1000d0;lpp=[1,-1]
  if(mode=='ee')lpp=0
  if(mode=='rapidity')ebeam=[2000d0,500d0]
  initial=[shat,sqrtshat,ybst_til_tocm,ybst_til_tolab,scale,muR2_current,muF12_current, &
           muF22_current,QES2_current,q2fact,xbk]
  i_fks=5;j_fks=3
  if(mode=='ee')then
    call macnlops_born_scale_factor(p,pb,[11,-11],[1d0+1d-10,1d0-1d-10],2,ratio,pdf,status)
  else
    call macnlops_born_scale_factor(p,pb,[21,-2],[.1d0,.2d0],2,ratio,pdf,status)
  endif
  after=[shat,sqrtshat,ybst_til_tocm,ybst_til_tolab,scale,muR2_current,muF12_current, &
         muF22_current,QES2_current,q2fact,xbk]
  if(any(initial/=after))stop 17
  if(model_coupling/=1d0.or.born_cache/=1d0)stop 18
  if(mode=='ee')then
    if(pdf_calls/=0)stop 19
  else
    if(pdf_calls/=4.or.any(abs(pdf_scales-[90d0,90d0,180d0,180d0])>1d-12))then
      if(mode=='fixed')then
        if(any(pdf_scales/=180d0))stop 21
      else if(mode=='rapidity')then
        if(any(abs(pdf_scales-[90d0/sqrt(2d0),90d0/sqrt(2d0),180d0,180d0])>1d-12))stop 25
      else
        stop 20
      endif
    endif
  endif
  write(*,*)status,ratio,pdf,born_calls,pdf_calls
end program
'''
        (work / 'check.f90').write_text(driver)
        cls.exe = work / 'check'
        result = subprocess.run([
            shutil.which('gfortran'), '-O1', '-ffixed-line-length-none',
            '-fcheck=all', '-I', str(work), str(work / 'fixture.f90'),
            str(work / 'helper.f'), str(work / 'check.f90'),
            '-o', str(cls.exe)], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def evaluate(self, mode):
        result = subprocess.run([str(self.exe), mode], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        status, coupling, pdf, calls, pdf_calls = result.stdout.split()
        return int(status), float(coupling), float(pdf), int(calls), int(pdf_calls)

    def test_born_qcd_powers_including_zero(self):
        for mode, expected in [('power0', 1.), ('power1', 4.), ('power2', 16.)]:
            with self.subTest(mode=mode):
                status, coupling, pdf, calls, _ = self.evaluate(mode)
                self.assertEqual(status, 0)
                self.assertAlmostEqual(coupling, expected)
                self.assertAlmostEqual(pdf, .125)
                self.assertEqual(calls, 2)

    def test_effective_and_mixed_model_couplings_use_native_amplitudes(self):
        for mode, expected in [('effective', 64.), ('mixed', 6.25)]:
            with self.subTest(mode=mode):
                self.assertAlmostEqual(self.evaluate(mode)[1], expected)

    def test_each_split_order_retains_custom_running_factor(self):
        self.assertAlmostEqual(self.evaluate('mixed_cpower')[1], 10.5 / 4.)

    def test_fixed_scales_are_unity_and_state_is_restored(self):
        self.assertEqual(self.evaluate('fixed')[:3], (0, 1., 1.))

    def test_bare_lepton_beams_do_not_query_pdfs(self):
        self.assertEqual(self.evaluate('ee'), (0, 1., 1., 2, 0))

    def test_custom_scale_uses_canonical_born_lab_boost(self):
        result = self.evaluate('rapidity')
        self.assertEqual(result[0], 0)
        self.assertAlmostEqual(result[1], 8.)
        self.assertAlmostEqual(result[2], 1. / (16. * 2. ** .5))

    def test_asymmetric_requested_scales_use_actual_native_common_scale(self):
        # Fixture checks all four PDF calls: Born muF1=90, real muF1=180;
        # the requested muF2 values 300 and 600 must never be queried.
        self.assertAlmostEqual(self.evaluate('power1')[2], .125)

    def test_nonpositive_support_returns_zero_without_floors(self):
        for mode in ('zero_born', 'negative_born', 'zero_pdf', 'negative_pdf'):
            with self.subTest(mode=mode):
                status, coupling, pdf, _, _ = self.evaluate(mode)
                self.assertEqual(status, 0)
                self.assertEqual(coupling * pdf, 0.)

    def test_nonfinite_support_fails_and_restores_real_state(self):
        for mode in ('nan_born', 'nan_pdf'):
            with self.subTest(mode=mode):
                self.assertNotEqual(self.evaluate(mode)[0], 0)


if __name__ == '__main__':
    unittest.main()
