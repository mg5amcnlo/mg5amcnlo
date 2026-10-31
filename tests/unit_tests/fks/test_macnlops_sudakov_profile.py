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
      double precision function sumdot(a,b,s)
      double precision a(0:3),b(0:3),s
      sumdot=(a(0)+s*b(0))**2-sum((a(1:3)+s*b(1:3))**2)
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
     $        sum(w(1:n)),macnlops_fsr_sudakov(.5d0,125d0,1,2,mc),
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
        names=['macnlops_delta_profile','macnlops_gauss_rule','macnlops_fsr_sudakov',
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
            status,n,source,cdf,weight,physical_cut,source_cut=map(float,line.split())
            self.assertEqual(status,0)
            self.assertAlmostEqual(source,self.reference(),delta=2e-10)
            u=min(max((pt-3.)/17.,0.),1.)
            self.assertAlmostEqual(cdf,1.-u*u/(u*u+(1.-u)**2),delta=2e-15)
            self.assertAlmostEqual(weight,1. if pt<20. else 0.,delta=2e-10)
            self.assertAlmostEqual(physical_cut,.5/125.,delta=2e-15)
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


if __name__=='__main__':unittest.main()
