"""Persistent NLO state must survive automatic local storage."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from madgraph.iolibs.file_writers import FortranWriter
from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestAutomaticStorage(unittest.TestCase):
    def compile_and_run(self, work, *sources):
        executable = work / 'check'
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-g', '-frecursive',
            '-finit-real=snan', '-finit-integer=-13579', '-finit-character=63',
            '-ffpe-trap=invalid,zero,overflow', '-fcheck=all',
            '-ffixed-line-length-132', '-I', str(work),
            *map(str, sources), '-o', str(executable)],
            cwd=work, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = subprocess.run([str(executable)], cwd=work,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def includes(self, work, sources):
        for name, source in sources.items():
            path = work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join('      ' + line + '\n'
                                    for line in source.splitlines()))

    def test_cteq3_reuses_coefficients_when_only_x_changes(self):
        with tempfile.TemporaryDirectory(prefix='mg5_pdf_storage_') as tmp:
            work = Path(tmp)
            (work / 'driver.f90').write_text('''program check
implicit none
double precision ctq3df,cold,warm,unused
integer iset,flavour,i,irt
! Changing Q forces a fresh parametrization; changing x reuses it.
do iset=1,3
  do flavour=-5,2
    do i=1,3
      unused=ctq3df(iset,flavour,0.1d0,90d0,irt)
      cold=ctq3df(iset,flavour,0.1d0*i,100d0,irt)
      unused=ctq3df(iset,flavour,0.4d0,100d0,irt)
      warm=ctq3df(iset,flavour,0.1d0*i,100d0,irt)
      if(cold.ne.warm)error stop 'PDF coefficients were not retained'
    enddo
  enddo
enddo
end program
''')
            self.compile_and_run(work, ROOT / 'Template/NLO/Source/PDF/cteq3.f',
                                 work / 'driver.f90')

    def test_virtual_buffers_and_dimensions_survive_repeated_calls(self):
        with tempfile.TemporaryDirectory(prefix='mg5_virtual_storage_') as tmp:
            work = Path(tmp) / 'SubProcesses/P0'
            work.mkdir(parents=True)
            self.includes(work, {
                'nexternal.inc': 'integer nexternal,nincoming\nparameter(nexternal=4,nincoming=2)',
                'coupl.inc': 'double precision g,mu_r\ncommon/test_couplings/g,mu_r',
                '../../Source/MODEL/input.inc': '',
                'born_nhel.inc': 'integer max_bhel\nparameter(max_bhel=1)',
                'a0Gmuconv.inc': '',
                'pmass.inc': 'pmass=0d0',
                'orders.inc': 'integer nsplitorders,amp_split_size,amp_split_size_born,qed_pos\n'
                              'parameter(nsplitorders=1,amp_split_size=2,amp_split_size_born=1,qed_pos=1)\n'
                              'integer nlo_orders(1)\nparameter(nlo_orders=[0])\n'
                              "character(3) ordernames(1)\nparameter(ordernames=['QCD'])\n"
                              'double precision amp_split(2)\ncommon/test_amp_split/amp_split',
            })
            (work / 'params.f90').write_text('''module FKSParams
implicit none
double precision :: IRPoleCheckThreshold=1d-5,PrecisionVirtualAtRunTime=1d-3
integer :: NHelForMCoverHels=-1
end module
''')
            (work / 'virtual.f').write_text(fortran_routine(TEMPLATE / 'BinothLHA.f', 'BinothLHA'))
            (work / 'driver.f90').write_text('''program check
implicit none
double precision p(0:3,3),born,virt,g,mu_r,qes2
common/test_couplings/g,mu_r
common/coupl_es/qes2
logical cs_run,force_polecheck,polecheck_passed,fillh
common/to_cs_run/cs_run
common/to_polecheck/force_polecheck,polecheck_passed
double precision volh
integer mc_hel,ihel,i
common/mc_int2/volh,mc_hel,ihel,fillh
g=1d0
qes2=100d0
cs_run=.true.
force_polecheck=.false.
mc_hel=0
born=1d0
do i=1,5
  p=dble(i)
  call BinothLHA(p,born,virt)
  if(virt.ne.3d0*i)error stop 'Lost virtual buffers or dimensions'
enddo
end program
subroutine get_nsqso_loop(n)
integer n
n=2
end subroutine
subroutine get_answer_dimension(n)
integer n
n=2
end subroutine
subroutine sloopmatrix_thres(p,answer,tolerance,accuracy,code)
double precision p(0:3,3),answer(0:3,0:2),tolerance,accuracy(0:2)
integer code
answer=0d0
answer(1,1)=p(0,1)
answer(1,2)=2d0*p(0,1)
accuracy=0d0
code=200
end subroutine
integer function getordpowfromindex_ml5(j,i)
integer j,i
getordpowfromindex_ml5=0
end function
integer function orders_to_amp_split_pos(orders)
integer orders(1)
orders_to_amp_split_pos=1
end function
! The test exits before pole checks and does not sample helicities.
subroutine update_as_param()
end subroutine
subroutine set_forbid_hel_doublecheck(flag)
logical flag
end subroutine
subroutine force_stability_check(flag)
logical flag
end subroutine
subroutine collier_compute_uv_poles(flag)
logical flag
end subroutine
subroutine collier_compute_ir_poles(flag)
logical flag
end subroutine
subroutine pickhelicitymc(p,goodhel,hel,ihel,volh)
error stop 'unexpected helicity sampling'
end subroutine
subroutine sloopmatrixhel_thres(p,hel,answer,tolerance,accuracy,code)
error stop 'unexpected helicity evaluation'
end subroutine
subroutine getpoles(p,qes2,double,single,prefactor)
error stop 'unexpected pole check'
end subroutine
subroutine amp_split_pos_to_orders(i,orders)
error stop 'unexpected order conversion'
end subroutine
integer function get_n_tagged_photons()
get_n_tagged_photons=0
end function
double precision function get_virtual_a0Gmu_conv(i,j,k,born)
error stop 'unexpected conversion'
end function
''')
            self.compile_and_run(work, work / 'params.f90', work / 'virtual.f',
                                 work / 'driver.f90')

    def test_histogram_layout_and_entry_state_survive_calls(self):
        with tempfile.TemporaryDirectory(prefix='mg5_histogram_storage_') as tmp:
            work = Path(tmp)
            source = ROOT / 'Template/NLO/FixedOrderAnalysis/dbook.f'
            (work / 'histograms.f').write_text(''.join(
                fortran_routine(source, name) for name in ('MULTITOP4', 'mclear')))
            shutil.copy2(source.with_suffix('.inc'), work / 'dbook.inc')
            (work / 'driver.f').write_text('''      program check
      implicit none
      include 'dbook.inc'
      character(70) filename
      nbin(1:2)=1
      hist(1,1)=2d0
      hist(2,1)=0.1d0
      xhis(1,1)=0.5d0
      hint(1)=2d0
      ient(1)=1
      title(1)='test histogram'
      book(1)='YES'
      book(2)='NO'
      uscore(1)=0d0
      oscore(1)=0d0
      hmin(1)=0d0
      hmax(1)=1d0
      open(99,file='plots.top',status='replace')
      call multitop4(1,2,2,1,'x','y','LIN')
      call mtfill(1,2,2,1,'x','y','LIN')
      call multitop4(1,2,2,1,'x','y','LIN')
      hist(1,1)=0d0
      hist(2,1)=0d0
      call multitop4(1,2,2,1,'x','y','LIN')
      close(99)
      filename='first.dat'
      call addfil(filename)
      filename='second.dat'
      call addfil(filename)
      call mclear
      call mclear
      end program
      subroutine sumfil(files,nfil)
      implicit none
      integer nfil
      character(70) files(20)
      if(nfil.eq.0)return
      if(nfil.ne.2)error stop 'lost file count'
      if(files(1).ne.'first.dat'.or.files(2).ne.'second.dat')
     $     error stop 'lost file list'
      end subroutine
''')
            self.compile_and_run(work, work / 'histograms.f', work / 'driver.f')
            plots = (work / 'plots.top').read_text()
            self.assertNotIn('nan', plots.lower())
            self.assertEqual(plots.count('HIST SOLID'), 3)
            self.assertEqual(plots.count('HIST DASHES'), 1)
            windows = [line for line in plots.splitlines() if 'SET WINDOW X' in line]
            self.assertEqual(len(windows), 3)
            self.assertNotEqual(windows[0], windows[1])

    def test_ew_helicity_list_survives_first_call(self):
        with tempfile.TemporaryDirectory(prefix='mg5_ew_hel_storage_') as tmp:
            work = Path(tmp)
            self.includes(work, {
                'nexternal.inc': 'integer nexternal\nparameter(nexternal=4)',
                'born_nhel.inc': '',
                'orders.inc': 'integer amp_split_size\nparameter(amp_split_size=1)',
                'ewsudakov_haslo.inc': 'logical has_lo1,has_lo2\n'
                                      'parameter(has_lo1=.true.,has_lo2=.false.)\n'
                                      'integer lo1_pos,lo2_pos\nparameter(lo1_pos=1,lo2_pos=1)',
            })
            template = (ROOT / 'madgraph/iolibs/template_files/ewsudakov_wrapper.inc').read_text()
            template = template.split('integer function sdk_get_ncomb()')[0]
            source = template % dict(
                ncomb=2, helicity_lines='data nhel /1,1,1,-1,-1,-1/',
                process_declarations='', born_cache_declarations='',
                iflist='iflist=1', reset_born_cache='',
                calls_to_me='amp_split_ewsud_lsc=amp_split_ewsud_lsc+amp_split_born_onehel')
            writer = FortranWriter(str(work / 'wrapper.f'))
            writer.writelines(source)
            writer.close()
            (work / 'driver.f90').write_text('''program check
implicit none
logical sud_mc_hel,sud_filter_hel
common/to_mc_hel/sud_mc_hel
common/to_filter_hel/sud_filter_hel
integer selected
common/to_ewsud_helselect/selected
double precision p(0:3,3)
complex(8) weight(1)
common/to_amp_ewsud_lsc/weight
sud_mc_hel=.true.
sud_filter_hel=.true.
selected=0
p=1d0
call sudakov_wrapper(p)
if(weight(1).ne.2d0)error stop 'initial helicity sum'
call sudakov_wrapper(p)
if(weight(1).ne.2d0)error stop 'sampled helicity'
end program
subroutine sdk_get_invariants(p,iflist,invariants)
double precision p(0:3,3),invariants(3,3)
integer iflist(3)
invariants=1d0
end subroutine
subroutine sborn_onehel(p,helicities,ihel,answer)
double precision p(0:3,3)
integer helicities(3),ihel
complex(8) answer,weight(1),lo2(1)
common/to_amp_split_ewsud/weight
common/to_amp_split_ewsud_lo2/lo2
answer=0d0
if(ihel.eq.2)answer=2d0
weight=answer
lo2=0d0
end subroutine
subroutine pickhelicitymc(p,goodhel,hel,pick,volh)
double precision p(0:3,3),volh
integer goodhel(2),hel(0:2),pick
if(hel(0).ne.1.or.hel(1).ne.2)error stop 'Lost EW helicity list'
if(goodhel(1).ne.1)error stop 'Lost EW helicity weight'
pick=1
volh=1d0
end subroutine
''')
            self.compile_and_run(work, work / 'wrapper.f', work / 'driver.f90')

    def test_resonance_without_identical_daughter_stays_in_bounds(self):
        with tempfile.TemporaryDirectory(prefix='mg5_resonance_bounds_') as tmp:
            work = Path(tmp)
            self.includes(work, {
                'nexternal.inc': 'integer nexternal\nparameter(nexternal=4)',
                'run.inc': 'double precision bwcutoff\nparameter(bwcutoff=10d0)',
            })
            (work / 'resonance.f').write_text(
                fortran_routine(TEMPLATE / 'cluster.f', 'IsBreitWigner') +
                fortran_routine(ROOT / 'Template/NLO/Source/kin_functions.f', 'dot'))
            (work / 'driver.f90').write_text('''program check
implicit none
integer tree(2,-2:-1),pdg(4),sprop(-1:-1),resonances(2,0:1)
double precision p(0:3,4),width(-1:-1),mass(-1:-1)
p=0d0
p(0,:)=50d0
p(3,1)=50d0
p(3,2)=-50d0
p(1,3)=50d0
p(1,4)=-50d0
tree(:, -1)=[3,4]
pdg=[2,-2,11,-11]
sprop=23
width=1d0
mass=100d0
call IsBreitWigner(4,1,p,tree,width,mass,pdg,sprop,resonances)
if(resonances(1,0).ne.1)error stop 'missing resonance'
if(any(resonances(:,1).ne.[12,23]))error stop 'resonance metadata'
end program
''')
            self.compile_and_run(work, work / 'resonance.f', work / 'driver.f90')

    def test_polynomial_fits_with_all_degree_selection_modes(self):
        with tempfile.TemporaryDirectory(prefix='mg5_polyfit_storage_') as tmp:
            work = Path(tmp)
            (work / 'driver.f90').write_text('''program check
implicit none
integer,parameter :: n=12,maxdeg=8
double precision x(n),y(n),w(n),r(n),a(3*n+3*maxdeg+3),eps,fit,yp(1)
double precision tolerances(3)
integer i,mode,degree,ierr
tolerances=[-0.01d0,0d0,1d-12]
do i=1,n
  x(i)=dble(i-6)/8d0
  y(i)=1d0+2d0*x(i)+0.5d0*x(i)**2
enddo
do mode=1,3
  w=1d0
  eps=tolerances(mode)
  call polfit(n,x,y,w,maxdeg,degree,eps,r,ierr,a)
  if(ierr.ne.1)error stop 'polynomial fit failed'
  do i=1,n
    call pvalue(degree,0,x(i),fit,yp,a)
    if(abs(fit-y(i)).gt.1d-12)error stop 'polynomial prediction'
  enddo
enddo
end program
''')
            self.compile_and_run(work, TEMPLATE / 'polfit.f', work / 'driver.f90')
