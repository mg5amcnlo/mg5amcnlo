"""Regressions for invalid returns and Born colour-table bounds."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestWarningFixes(unittest.TestCase):
    def compile(self, work, *sources):
        executable = work / 'check'
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-Wall', '-fcheck=all',
            '-ffixed-line-length-132', '-I', str(work),
            *map(str, sources), '-o', str(executable)],
            cwd=work, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return executable

    def test_born_colour_copy_uses_born_flow_count(self):
        with tempfile.TemporaryDirectory(prefix='mg5_born_flow_bounds_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=4,nincoming=2)',
                'born_nhel.inc': 'integer max_bcol\nparameter(max_bcol=1)',
                'genps.inc': 'integer maxproc,maxflow\nparameter(maxproc=1,maxflow=2)',
                'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=1)',
                'fks_info.inc': 'integer pdg_type_d(1,4),fks_i_d(1),fks_j_d(1)\n'
                                'data pdg_type_d/2,-2,22,-21/\n'
                                'data fks_i_d/4/,fks_j_d/3/',
                'leshouche_decl.inc': 'integer maxproc_used,maxflow_used\n'
                                      'parameter(maxproc_used=1,maxflow_used=2)\n'
                                      'integer idup_d(1,4,1),mothup_d(1,2,4,1)\n'
                                      'integer icolup_d(1,2,4,2),niprocs_d(1)',
                'born_leshouche.inc': 'data idup/2,-2,22/\n'
                                      'data mothup/0,0,0,0,1,2/\n'
                                      'data icolup/501,0,0,501,0,0/',
            }
            for name, source in includes.items():
                (work / name).write_text(''.join('      ' + line + '\n'
                                               for line in source.splitlines()))
            (work / 'chooser.f').write_text(fortran_routine(
                TEMPLATE / 'chooser_functions.f', 'read_leshouche_info'))
            (work / 'driver.f90').write_text('''program check
implicit none
integer ids(1,4,1),mothers(1,2,4,1),colours(1,2,4,2),nprocs(1)
colours=-123
call read_leshouche_info(ids,mothers,colours,nprocs)
if(any(ids(1,:,1).ne.[2,-2,22,-21]).or.nprocs(1).ne.1)error stop 'Born IDs'
if(any(colours(1,:,1:3,1).ne.reshape([501,0,0,501,0,0],[2,3])))error stop 'Born colours'
if(any(colours(:,:,:,2).ne.0))error stop 'unused colour flow'
end program
''')
            executable = self.compile(work, work / 'chooser.f', work / 'driver.f90')
            result = subprocess.run([str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_sumpow_rejects_zero_exponent(self):
        with tempfile.TemporaryDirectory(prefix='mg5_sumpow_') as tmp:
            work = Path(tmp)
            source = (ROOT / 'Template/NLO/FixedOrderAnalysis/dbook.f').read_text()
            routine = re.search(r'^      REAL\*8 FUNCTION SUMPOW\b.*?^      END\s*$',
                                source, re.MULTILINE | re.DOTALL)
            (work / 'sumpow.f').write_text(routine[0] + '\n')
            (work / 'driver.f90').write_text('''program check
implicit none
double precision sumpow,answer
integer exponent
character(16) arg
call get_command_argument(1,arg)
read(arg,*)exponent
answer=sumpow(3d0,4d0,exponent)
if(exponent.eq.1.and.answer.ne.7d0)error stop 'linear sum'
if(exponent.eq.2.and.answer.ne.5d0)error stop 'quadrature sum'
write(*,*)'PASS SUMPOW'
end program
subroutine mwarn(routine)
character(*) routine
write(*,*)routine
end subroutine
''')
            executable = self.compile(work, work / 'sumpow.f', work / 'driver.f90')
            for exponent in (1, 2, 0):
                result = subprocess.run([str(executable), str(exponent)], cwd=work,
                                        capture_output=True, text=True)
                if exponent:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('IPOW=0 not allowed', result.stdout)
                    self.assertNotIn('PASS SUMPOW', result.stdout)

    def test_isr_shower_scale_rejects_unknown_shower(self):
        with tempfile.TemporaryDirectory(prefix='mg5_shower_scale_return_') as tmp:
            work = Path(tmp)
            routines = '\n'.join(fortran_routine(TEMPLATE / 'montecarlocounter.f', name)
                                 for name in ('qMC_ileg1', 'qMC_ileg2'))
            (work / 'scales.f').write_text('''      module scales
      implicit none
      character(10) shower_mc_mod
      double precision shat_n1,xtk,xuk
      parameter(shat_n1=100d0,xtk=-1d0,xuk=-2d0)
      contains
''' + routines + '\n      end module\n')
            (work / 'driver.f90').write_text('''program check
use scales
implicit none
integer leg
double precision result
character(16) arg
call get_command_argument(1,arg)
read(arg,*)leg
call get_command_argument(2,shower_mc_mod)
if(leg.eq.1)then
  result=qMC_ileg1(.5d0,.2d0)
  if(abs(result-sqrt(.5d0)).gt.1d-14)error stop 'left shower scale'
else
  result=qMC_ileg2(.5d0,.2d0)
  if(abs(result-1d0).gt.1d-14)error stop 'right shower scale'
endif
end program
''')
            executable = self.compile(work, work / 'scales.f', work / 'driver.f90')
            for leg in (1, 2):
                for shower in ('PYTHIA8', 'UNKNOWN'):
                    result = subprocess.run([str(executable), str(leg), shower], cwd=work,
                                            capture_output=True, text=True)
                    if shower == 'PYTHIA8':
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('Unsupported shower', result.stdout)

    def test_legacy_sorting_preserves_order_and_permutations(self):
        with tempfile.TemporaryDirectory(prefix='mg5_cut_sorting_') as tmp:
            work = Path(tmp)
            (work / 'sort.f').write_text('\n'.join(
                fortran_routine(TEMPLATE / 'cuts.f', name)
                for name in ('SORTZV', 'SORTTF', 'SORTTI', 'SORTTC', 'ICMPCH')))
            (work / 'driver.f90').write_text('''program check
implicit none
real values(7)
integer integers(7),indices(7),direction,i,j,icmpch
values=[.5,-1.,4.,4.,-8.25,0.,1.e-10]
integers=[3,-1,0,-8,3,2,huge(1)]
do direction=0,1
  call sortzv(values,indices,7,1,direction,0)
  do i=1,7
    if(count(indices.eq.i).ne.1)error stop 'real index permutation'
    if(i.eq.7)cycle
    if(direction.eq.0.and.values(indices(i)).gt.values(indices(i+1)))error stop 'ascending real sort'
    if(direction.eq.1.and.values(indices(i)).lt.values(indices(i+1)))error stop 'descending real sort'
  enddo
enddo
indices=[(i,i=1,7)]
call sortti(integers,indices,7)
do i=1,6
  if(integers(indices(i)).gt.integers(indices(i+1)))error stop 'integer sort'
enddo
indices=[(i,i=1,7)]
call sorttc(integers,indices,7)
do i=1,6
  if(icmpch(integers(indices(i)),integers(indices(i+1))).gt.0)error stop 'character sort'
enddo
do i=1,7
  do j=1,7
    if(icmpch(integers(i),integers(j)).ne.-icmpch(integers(j),integers(i)))error stop 'comparison symmetry'
  enddo
enddo
end program
''')
            executable = self.compile(work, work / 'sort.f', work / 'driver.f90')
            result = subprocess.run([str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_isolation_distance_depends_on_spatial_momenta(self):
        with tempfile.TemporaryDirectory(prefix='mg5_isolation_distance_') as tmp:
            work = Path(tmp)
            (work / 'geometry.f').write_text('\n'.join(
                fortran_routine(TEMPLATE / 'cuts.f', name) for name in
                ('iso_getdrv40', 'iso_getdr', 'iso_getpseudorap', 'iso_getdelphi')))
            (work / 'driver.f90').write_text('''program check
implicit none
double precision p(0:3),q(0:3),expected,iso_getdrv40
p=[99d0,1d0,0d0,1d0]
q=[123d0,0d0,2d0,-2d0]
expected=sqrt((2d0*asinh(1d0))**2+(acos(-1d0)/2d0)**2)
if(abs(iso_getdrv40(p,q)-expected).gt.1d-12)error stop 'isolation distance'
p(0)=-1d7
q(0)=1d-8
if(abs(iso_getdrv40(p,q)-expected).gt.1d-12)error stop 'energy changed pseudorapidity'
end program
''')
            executable = self.compile(work, work / 'geometry.f', work / 'driver.f90')
            result = subprocess.run([str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
