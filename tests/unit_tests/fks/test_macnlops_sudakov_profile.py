"""Native source/start-scale marginal with an independently integrable table.

The production quadrature and connection selector are compiled unchanged.
For the fixture S(q,m)=q/m and constant PDFs, the source probability is
min(2.5/Q,1)*min(7.5/Q,1), making the source and conditional shower measures
independently checkable without exporting a matrix element.
"""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import fortran_routine

ROOT=Path(__file__).resolve().parents[3]
TEMPLATE=ROOT/'Template/NLO/SubProcesses'

STUBS='''
      module process_module
      logical valid_dipole_n(3,3,1)
      double precision mass_n(3)
      end module
      module fks_phase_space_data
      double precision xbjrk_cnt(2,0:0)
      end module
      module scale_module
      integer born_flow_picked
      double precision shower_scale_nbody_min(3,3),
     $ shower_scale_nbody_max(3,3)
      end module
      module FKSParams
      logical,parameter :: delta_backward_isr=.false.
      logical :: delta_common_scale=.false.
      double precision,parameter :: delta_backward_alphas=.1365d0
      integer,parameter :: delta_backward_alphas_order=1
      end module
      module weight_lines
      integer,parameter :: macnlops_sudakov_max=512
      integer macnlops_sudakov_nquarks
      double precision macnlops_sudakov_quarks(4,64)
      integer macnlops_sudakov_ngluons
      double precision macnlops_sudakov_gluons(5,64)
      end module
'''

HELPERS='''
      subroutine delta_source_flavours(ids)
      integer ids(4)
      ids=[2,-2,25,21]
      end subroutine
      subroutine delta_native_ir_probability(p,pb,ids,start,pt,prob,n,t)
      double precision p(0:3,4),pb(0:3,3),start(3,3),pt,prob,t(9),q
      integer ids(4),n
      q=start(1,2)
      prob=1d0
      if(q.gt.7.5d0)prob=(7.5d0/q)**2
c A resolved charm event must use the SAME representative source law.
      if(any(ids.ne.[2,-2,25,21]))prob=1d0
      n=0
      t=0d0
      end subroutine

      subroutine get_delta_stopping_scales(p,ids,col,start,upper,
     $ stop,mass,dead)
      double precision p(0:3,4),start(3,3),upper,stop(3,3),mass(3,3)
      integer ids(4),col(2,3)
      logical*1 dead(3,3)
      stop=-1d0
      mass=-1d0
      dead=.true.
      stop(1,2)=2.5d0
      stop(2,1)=7.5d0
      mass(1,2)=125d0
      mass(2,1)=125d0
      dead(1,2)=.false.
      dead(2,1)=.false.
      end subroutine
      subroutine find_color_connectors(flow,i,n,partners)
      integer flow,i,n,partners(2)
      n=1
      partners=1
      end subroutine
      subroutine delta_alphas_lambdas(alpha,order,lambda)
      integer order
      double precision alpha,lambda(3:5)
      lambda=1d0
      end subroutine
      double precision function delta_backward_leg(pdg,flavours,
     $ beam,x,connections,lambda,order)
      integer pdg,flavours(:),beam,order
      double precision x,lambda(3:5)
      type(delta_connections) connections
      delta_backward_leg=-1d0
      end function
'''

EXTERNAL='''
      double precision function pysudakov(q,m,id,typ,mc)
      double precision q,m,mc(21),lo,hi,ml,mh
      integer id,typ
      common/cstxmbds/lo,hi,ml,mh
      lo=1d0
      hi=7000d0
      ml=1d0
      mh=7000d0
      pysudakov=q/m
      end
      double precision function pdg2pdf(ih,id,beam,x,q)
      integer ih,id,beam
      double precision x,q
      pdg2pdf=1d0
      end
      program main
      use matching_test
      use process_module
      use scale_module
      use fks_phase_space_data
      use FKSParams
      implicit none
      include 'run.inc'
      double precision p(0:3,4),pb(0:3,3),pt,source,
     $ q(512),w(512),fsr(512),cdf,mc(21)
      integer n,status,ids(4)
      character*16 mode
      call get_command_argument(1,mode)
      delta_common_scale=index(trim(mode),'native').eq.1
      lpp=1
      valid_dipole_n=.false.
      valid_dipole_n(1,2,1)=.true.
      valid_dipole_n(2,1,1)=.true.
      mass_n=[0d0,0d0,125d0]
      born_flow_picked=1
      shower_scale_nbody_min=-1d0
      shower_scale_nbody_max=-1d0
      where(valid_dipole_n(:,:,1))
         shower_scale_nbody_min=3d0
         shower_scale_nbody_max=20d0
      endwhere
      xbjrk_cnt=.01d0
      pb(:,1)=[62.5d0,0d0,0d0,62.5d0]
      pb(:,2)=[62.5d0,0d0,0d0,-62.5d0]
      pb(:,3)=[125d0,0d0,0d0,0d0]
      p=0d0
      ids=[2,-2,25,21]
      if(trim(mode).eq.'native_charm')ids=[4,-4,25,21]
      mc=0d0
      do
         read(*,*,end=10)pt
         call macnlops_delta_profile(p,pb,ids,pt,source,n,q,w,fsr,
     $        cdf,status)
         write(*,'(i5,1x,i5,5(1x,es25.16))')status,n,source,cdf,
     $        sum(w(1:n)),maxval(abs(fsr-1d0)),
     $        pysudakov_safe(.5d0,125d0,1,2,mc)
      enddo
 10   continue
      end
'''


@unittest.skipUnless(shutil.which('gfortran'),'requires gfortran')
class TestSudakovMarginal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='macnlops_sudakov_marginal_')
        cls.addClassCleanup(cls.tmp.cleanup)
        w=Path(cls.tmp.name)
        (w/'genps.inc').write_text('      integer,parameter :: maxproc=1,iproc_born=1\n')
        (w/'born_leshouche.inc').write_text('''      data idup/2,-2,25/
      data mothup/6*0/
      data icolup/501,0,0,501,0,0/
''')
        source=TEMPLATE/'montecarlocounter.f'
        dtype=re.search(r'^      type :: delta_connections\n.*?^      end type delta_connections',source.read_text(),re.M|re.S)[0]
        names=['macnlops_delta_profile','macnlops_gauss_rule',
               'get_delta_connections','delta_leg_probability','gl_safe','pysudakov_safe',
               'get_to_zero','emscafun','setSudType','get_parton_id']
        program=STUBS+'      module matching_test\n      implicit none\n'+dtype+'''
      integer,parameter :: nexternal=4,nincoming=2,max_bcol=1
      logical :: isspecial(1)=.false.
      contains
'''+''.join(fortran_routine(source,n) for n in names)+HELPERS+'      end module\n'+EXTERNAL
        (w/'check.f').write_text(program)
        cls.exe=w/'check'
        result=subprocess.run(['gfortran','-O2','-fcheck=all','-ffixed-line-length-none',
            '-I'+str(w),'-I'+str(TEMPLATE),str(TEMPLATE/'mcatnlo_delta_scales.f90'),
            str(w/'check.f'),'-o',str(cls.exe)],cwd=w,capture_output=True,text=True)
        if result.returncode:raise AssertionError(result.stdout+result.stderr)

    @staticmethod
    def reference():
        def f(q):
            u=(q-3.)/17.
            density=2.*u*(1.-u)/(17.*(u*u+(1.-u)**2)**2)
            return density*min(2.5/q,1.)*min(7.5/q,1.)
        value=0.
        for a,b in [(3.,7.5),(7.5,20.)]:
            n=4000;h=(b-a)/n
            value+=h/3.*(f(a)+f(b)+sum((4 if k%2 else 2)*f(a+k*h) for k in range(1,n)))
        return value

    def test_original_source_average_and_conditional_start_cdf(self):
        result=subprocess.run([str(self.exe)],input='0.6\n5\n10\n21\n',capture_output=True,text=True,check=True)
        for pt,line in zip([.6,5.,10.,21.],result.stdout.splitlines()):
            status,n,source,cdf,weight,fsr_table_error,source_cut=map(float,line.split())
            self.assertEqual(status,0)
            self.assertAlmostEqual(source,self.reference(),delta=2e-10)
            u=min(max((pt-3.)/17.,0.),1.)
            self.assertAlmostEqual(cdf,1.-u*u/(u*u+(1.-u)**2),delta=2e-15)
            self.assertAlmostEqual(weight,1. if pt<20. else 0.,delta=2e-10)
            self.assertEqual(fsr_table_error,0.)
            self.assertEqual(source_cut,0.)
            self.assertEqual(n>0,pt<20.)


    def test_native_source_average_includes_unsupported_start_nodes(self):
        def integrand(q):
            u=(q-3.)/17.
            rho=2.*u*(1.-u)/(17.*(u*u+(1.-u)**2)**2)
            return rho*(1. if q<=7.5 else (7.5/q)**2)
        expected=0.
        for a,b in [(3.,7.5),(7.5,20.)]:
            n=4000; h=(b-a)/n
            expected+=h/3.*(integrand(a)+integrand(b)+sum(
                (4 if k%2 else 2)*integrand(a+k*h) for k in range(1,n)))
        result=subprocess.run([str(self.exe),'native'],input='5\n10\n',
                              capture_output=True,text=True,check=True)
        for line in result.stdout.splitlines():
            status,n,source,cdf,weight,_,_=map(float,line.split())
            self.assertEqual(status,0)
            self.assertAlmostEqual(source,expected,delta=2e-10)
            self.assertAlmostEqual(weight,1.,delta=2e-10)


    def test_grouped_source_average_uses_representative_flavours(self):
        reference=subprocess.run([str(self.exe),'native'],input='5\n10\n',
                                 capture_output=True,text=True,check=True)
        resolved=subprocess.run([str(self.exe),'native_charm'],input='5\n10\n',
                                capture_output=True,text=True,check=True)
        self.assertEqual(resolved.stdout,reference.stdout)


@unittest.skipUnless(shutil.which('gfortran'),'requires gfortran')
class TestGlobalGluonProfile(unittest.TestCase):
    def test_global_recoiler_two_caps_source_measure_and_c_api(self):
        """A W+g Born separates its two local caps from global recoil."""
        stubs=STUBS.replace('valid_dipole_n(3,3,1)',
                            'valid_dipole_n(4,4,1)').replace(
            'mass_n(3)','mass_n(4)').replace('(3,3)', '(4,4)')
        stubs=stubs.replace('      integer macnlops_sudakov_ngluons', '''
      logical macnlops_sudakov_enabled,macnlops_sudakov_ready
      double precision macnlops_sudakov_requested_pt
      integer macnlops_sudakov_ngluons''')
        helpers='''
      subroutine delta_source_flavours(ids)
      integer ids(5)
      ids=[2,-1,24,21,21]
      end subroutine
      subroutine delta_native_ir_probability(p,pb,ids,start,pt,prob,n,t)
      double precision p(0:3,5),pb(0:3,4),start(4,4),pt,prob,t(16)
      integer ids(5),n
      prob=1d0-start(1,4)/200d0
      n=0
      t=0d0
      end subroutine
      subroutine get_delta_stopping_scales(p,ids,col,start,upper,
     $ stop,mass,dead)
      double precision p(0:3,5),start(4,4),upper,stop(4,4),mass(4,4)
      integer ids(5),col(2,4)
      logical*1 dead(4,4)
      stop=-1d0
      mass=125d0
      dead=.true.
      stop(1,4)=7.5d0
      stop(2,4)=12.5d0
      stop(4,1)=7.5d0
      stop(4,2)=12.5d0
      where(stop.gt.0d0)dead=.false.
      end subroutine
      subroutine find_color_connectors(flow,i,n,partners)
      integer flow,i,n,partners(2)
      if(i.eq.4)then
         n=2
         partners=[1,2]
      else
         n=1
         partners=4
      endif
      end subroutine
      subroutine delta_alphas_lambdas(alpha,order,lambda)
      integer order
      double precision alpha,lambda(3:5)
      lambda=1d0
      end subroutine
      double precision function delta_backward_leg(pdg,flavours,
     $ beam,x,connections,lambda,order)
      integer pdg,flavours(:),beam,order
      double precision x,lambda(3:5)
      type(delta_connections) connections
      delta_backward_leg=-1d0
      end function
'''
        external=EXTERNAL[:EXTERNAL.index('      program main')]+'''
      program main
      use matching_test
      use process_module
      use scale_module
      use fks_phase_space_data
      use FKSParams
      use weight_lines
      implicit none
      include 'run.inc'
      integer idup(4,1),mothup(2,4,1),icolup(2,4,1)
      common/native_born/idup,mothup,icolup
      double precision p(0:3,5),pb(0:3,4),source,cdf,q(512),w(512),
     $ fsr(512),qsave(512),wsave(512),saved_source,saved_cdf,
     $ packed(5,64),expected(5),energy,u,gate_fraction,reference
      integer n,nsave,status,ids(5),ng
      delta_common_scale=.true.
      idup(:,1)=[2,-1,24,21]
      mothup=0
      icolup(:,:,1)=reshape([501,0,0,502,0,0,501,502],[2,4])
      lpp=1
      macnlops_charm_mass=1.2d0
      macnlops_bottom_mass=4d0
      valid_dipole_n=.false.
      valid_dipole_n(1,4,1)=.true.
      valid_dipole_n(2,4,1)=.true.
      valid_dipole_n(4,1,1)=.true.
      valid_dipole_n(4,2,1)=.true.
      mass_n=[0d0,0d0,80d0,0d0]
      born_flow_picked=1
      shower_scale_nbody_min=-1d0
      shower_scale_nbody_max=-1d0
      where(valid_dipole_n(:,:,1))
         shower_scale_nbody_min=3d0
         shower_scale_nbody_max=100d0
      endwhere
      xbjrk_cnt=.01d0
      energy=sqrt(80d0**2+30d0**2)+30d0
      pb(:,1)=[energy/2d0,0d0,0d0,energy/2d0]
      pb(:,2)=[energy/2d0,0d0,0d0,-energy/2d0]
      pb(:,3)=[energy-30d0,-24d0,0d0,-18d0]
      pb(:,4)=[30d0,24d0,0d0,18d0]
      p=0d0
      ids=[2,-1,24,21,21]
      call macnlops_delta_profile(p,pb,ids,2d0,source,n,q,w,fsr,
     $     cdf,status)
      if(status.ne.0)stop 1
      call macnlops_c_gluon_profile(ng,packed)
      expected=[0d0,80d0,energy,sqrt(energy*12d0)/2d0,
     $     sqrt(energy*48d0)/2d0]
      if(ng.ne.1.or.any(abs(packed(:,1)-expected).gt.1d-12))stop 2
      if(any(packed(:,2:64).ne.0d0))stop 3
      if(abs(source-.7425d0).gt.2d-10)stop 4
      if(cdf.ne.1d0.or.abs(sum(w(1:n))-1d0).gt.2d-10)stop 5
      if(any(fsr.ne.1d0))stop 6
      nsave=n
      qsave=q
      wsave=w
      saved_source=source
      saved_cdf=cdf
c Changing only shower-radiator metadata must preserve the source and
c its starting-scale measure bit for bit, including the unused nodes.
      idup(4,1)=2
      call macnlops_delta_profile(p,pb,ids,2d0,source,n,q,w,fsr,
     $     cdf,status)
      if(status.ne.0.or.n.ne.nsave)stop 7
      if(any(q.ne.qsave).or.any(w.ne.wsave))stop 8
      if(source.ne.saved_source.or.cdf.ne.saved_cdf)stop 9
      call macnlops_c_gluon_profile(ng,packed)
      if(ng.ne.0.or.any(packed.ne.0d0))stop 10
      if(macnlops_sudakov_nquarks.ne.1)stop 11
      if(any(abs(macnlops_sudakov_quarks(:,1)-expected(1:4))
     $     .gt.1d-12))stop 12
c Between the two colour-end caps only the second line can evolve.
      idup(4,1)=21
      call macnlops_delta_profile(p,pb,ids,30d0,source,n,q,w,fsr,
     $     cdf,status)
      if(status.ne.0.or.any(q(1:n).le.30d0))stop 13
      u=(30d0-3d0)/97d0
      if(abs(cdf-(1d0-u*u/(u*u+(1d0-u)**2))).gt.1d-14)stop 14
      if(abs(source-.7425d0).gt.2d-10)stop 15
      call macnlops_c_gluon_profile(ng,packed)
      if(ng.ne.1.or.packed(4,1).ge.30d0.or.
     $     packed(5,1).le.30d0)stop 16
      call macnlops_c_sudakov_request(0,0d0)
      call macnlops_c_gluon_profile(ng,packed)
      if(ng.ne.0.or.any(packed.ne.0d0))stop 17
c A failed/new profile must not expose gluons from the previous event.
      call macnlops_delta_profile(p,pb,ids,0d0,source,n,q,w,fsr,
     $     cdf,status)
      call macnlops_c_gluon_profile(ng,packed)
      if(status.eq.0.or.ng.ne.0.or.any(packed.ne.0d0))stop 18
c A physical soft W+g Born can radiate below the native bottom slab,
c but Q>4 makes its first slab kinematically empty. Refine only the
c shower measure; the source average still uses its original grid.
      energy=sqrt(80d0**2+7d0**2)+7d0
      pb(:,1)=[energy/2d0,0d0,0d0,energy/2d0]
      pb(:,2)=[energy/2d0,0d0,0d0,-energy/2d0]
      pb(:,3)=[energy-7d0,-5.6d0,0d0,-4.2d0]
      pb(:,4)=[7d0,5.6d0,0d0,4.2d0]
      where(valid_dipole_n(:,:,1))
         shower_scale_nbody_min=1d0
         shower_scale_nbody_max=10d0
      endwhere
      call macnlops_delta_profile(p,pb,ids,1d0,source,n,q,w,fsr,
     $     cdf,status)
      if(status.ne.0.or.abs(source-.9725d0).gt.2d-8)stop 19
      call macnlops_c_gluon_profile(ng,packed)
      if(ng.ne.1.or.minval(packed(4:5,1)).le.4d0.or.
     $     (packed(3,1)-packed(2,1))/2d0.ge.4d0)stop 20
      saved_source=source
      saved_cdf=cdf
      nsave=n
      u=(4d0-1d0)/9d0
      reference=u*u/(u*u+(1d0-u)**2)
      gate_fraction=sum(w(1:n),mask=q(1:n).lt.4d0)
      if(abs(gate_fraction-reference).gt.2d-10)stop 21
      if(any(fsr.ne.1d0).or.abs(sum(w(1:n))-1d0).gt.2d-10)
     $     stop 22
c Removing the gluon disables only the PS refinement. Its original
c source mean and exact starting CDF remain bit-identical, whereas
c the old unsplit PS grid does not integrate the Q=4 gate accurately.
      idup(4,1)=2
      call macnlops_delta_profile(p,pb,ids,1d0,source,n,q,w,fsr,
     $     cdf,status)
      if(status.ne.0.or.source.ne.saved_source.or.cdf.ne.saved_cdf)
     $     stop 23
      if(n.ge.nsave)stop 24
      gate_fraction=sum(w(1:n),mask=q(1:n).lt.4d0)
      if(abs(gate_fraction-reference).lt.1d-4)stop 25
c Invalid measured flavour floors must fail closed only when used.
      macnlops_bottom_mass=0d0
      call macnlops_delta_profile(p,pb,ids,1d0,source,n,q,w,fsr,
     $     cdf,status)
      if(status.ne.0)stop 26
      idup(4,1)=21
      call macnlops_delta_profile(p,pb,ids,1d0,source,n,q,w,fsr,
     $     cdf,status)
      if(status.eq.0)stop 27
      write(*,*)'global gluon profile passed'
      end
'''
        with tempfile.TemporaryDirectory(prefix='macnlops-gluon-profile-') as temporary:
            work=Path(temporary)
            (work/'genps.inc').write_text(
                '      integer,parameter :: maxproc=1,iproc_born=1\n')
            (work/'born_leshouche.inc').write_text(
                '      common/native_born/idup,mothup,icolup\n')
            source=TEMPLATE/'montecarlocounter.f'
            dtype=re.search(r'^      type :: delta_connections\n.*?^      end type delta_connections',
                            source.read_text(),re.M|re.S)[0]
            names=['macnlops_delta_profile','macnlops_gauss_rule',
                   'get_delta_connections','delta_leg_probability','gl_safe',
                   'pysudakov_safe','get_to_zero','emscafun','setSudType','get_parton_id']
            code=stubs+'      module matching_test\n      implicit none\n'+dtype+'''
      integer,parameter :: nexternal=5,nincoming=2,max_bcol=1
      logical :: isspecial(1)=.false.
      contains
'''+''.join(fortran_routine(source,name) for name in names)+helpers
            code+=''.join(fortran_routine(TEMPLATE/'macnlops_capi.f',name)
                          for name in ['macnlops_c_gluon_profile',
                                       'macnlops_c_sudakov_request'])
            code+='      end module\n'+external
            (work/'check.f').write_text(code)
            exe=work/'check'
            build=subprocess.run(['gfortran','-O2','-fcheck=all',
                '-ffixed-line-length-none','-I'+str(work),'-I'+str(TEMPLATE),
                str(TEMPLATE/'mcatnlo_delta_scales.f90'),str(work/'check.f'),
                '-o',str(exe)],cwd=work,capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stdout+build.stderr)
            run=subprocess.run([str(exe)],cwd=work,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('global gluon profile passed',run.stdout)


if __name__=='__main__':unittest.main()
