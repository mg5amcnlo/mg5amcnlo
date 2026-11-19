"""Check the stable-top analysis with real HwU and anti-kt clustering."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from madgraph.various.histograms import HwUList


ROOT = Path(__file__).resolve().parents[3]
MCATNLO = ROOT / 'Template/NLO/MCatNLO'


@unittest.skipUnless(shutil.which('gfortran') and shutil.which('g++'),
                     'requires Fortran and C++ compilers')
class TestSingleTopRecoilAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(prefix='mg5_tj_analysis_')
        cls.addClassCleanup(cls.work.cleanup)
        work = Path(cls.work.name)
        # This process-dependent include is normally supplied at export time.
        # PineAPPL is disabled in this shower analysis.
        (work / 'pineappl_maxproc.inc').write_text(
            '      integer mxpdflumi,max_nproc\n'
            '      parameter(mxpdflumi=1,max_nproc=1)\n')
        wrapper = (MCATNLO / 'srcCommon/myfastjetfortran.cc').read_text()
        wrapper = wrapper.replace('#include "fastjet/ClusterSequence.hh"',
                                  '#include "fjcore.hh"')
        wrapper = wrapper.replace('namespace fj = fastjet;',
                                  'namespace fj = fjcore;')
        (work / 'wrapper.cc').write_text(wrapper)
        includes = ['-I', str(work), '-I', str(MCATNLO / 'include'),
                    '-I', str(MCATNLO / 'srcCommon')]
        for source, target in ((work / 'wrapper.cc', 'wrapper.o'),
                (MCATNLO / 'srcCommon/fjcore.cc', 'fjcore.o')):
            subprocess.run(['g++', '-c', '-O0', '-w', *includes,
                            str(source), '-o', target], cwd=work,
                           check=True, capture_output=True)
        (work / 'check.f90').write_text(r'''
program check
  use st_recoil_settings
  use HwU_variables
  implicit none
  integer, parameter :: nmax=100000
  integer :: nevhep,nhep,isthep(nmax),idhep(nmax)
  integer :: jmohep(2,nmax),jdahep(2,nmax),i
  double precision :: phep(5,nmax),vhep(4,nmax),evwgt,w(2)
  common /hepevt/ nevhep,nhep,isthep,idhep,jmohep,jdahep,phep,vhep
  common /cevwgt/ evwgt
  logical :: pineappl
  common /for_pineappl/ pineappl
  character(80) :: labels(2)
  character(20) :: mode

  call get_command_argument(1,mode)
  pineappl=.false.
  labels(1)='central value'
  labels(2)='variation'
  call pyabeg(2,labels)
  isthep=1
  jmohep=0
  jdahep=0
  vhep=0d0
  nevhep=12345 ! Deliberately different from the number of analysis calls.
  nhep=3
  phep=0d0
  idhep(1:3)=[6,2,21]
  call momentum(1,80d0,-1d0,st_pi,173d0)
  call momentum(2,103d0,3.1d0,0d0,0d0)
  call momentum(3,25d0,0d0,1.5d0,0d0)
  if (trim(mode)=='split') then
     ! A collinear split of j1 must recombine to the unsplit jet.
     nhep=4
     idhep(4)=21
     call momentum(2,51.5d0,3.1d0,0d0,0d0)
     call momentum(4,51.5d0,3.1d0,0d0,0d0)
  endif
  if (trim(mode)=='long_record') then
     ! A global-recoil history can put the final top beyond entry 4000.
     nhep=5003
     idhep(5001:5003)=idhep(1:3)
     phep(:,5001:5003)=phep(:,1:3)
     isthep(1:5000)=2
     idhep(1:5000)=21
  endif
  evwgt=2d0
  w=[9d0,13.5d0] ! Test that relative weights are normalized with EVWGT.
  call pyanal(2,w)
  call expect(sum(histy(1,5,1:nbin(5))),2d0,'jet multiplicity rate')
  call expect(sum(histy(1,5,3:3)),2d0,'exactly two jets excluding top')
  call expect(histy(1,61,1),2d0,'wide-gap selection')
  ! Reverse the beams: the oriented Zeppenfeld coordinate must not change.
  phep(3,1:nhep)=-phep(3,1:nhep)
  evwgt=-0.5d0
  w=[-4d0,-8d0]
  call pyanal(2,w)
  call expect(sum(histy(1,41,1:nbin(41))),1.5d0,'oriented zstar rate')
  call expect(histy(1,41,18),1.5d0,'same zstar bin for both beam orientations')
  call expect(sum(histy(1,43,26:50)),1.5d0,'positive distance inside gap')
  call expect(histy(1,51,1),0d0,'gap veto fails at Q0=21')
  call expect(histy(1,51,4),1.5d0,'gap veto passes at Q0=27')
  call expect(minval(histy(1,52,1:40)),1.5d0,'veto reference')
  call expect(histy2(51,4),4.25d0,'veto squared signed weights')
  call expect(histy(2,21,1),2d0,'variation on selected events')
  ! No accepted jet: include the event in inclusive plots and normalization.
  nhep=1
  isthep(1)=1
  idhep(1)=6
  evwgt=0.25d0
  w=[1d0,2d0]
  call pyanal(2,w)
  call expect(histy(1,1,1),1.75d0,'inclusive signed weights')
  call expect(histy(2,1,1),2.5d0,'inclusive variation weights')
  call expect(histy2(1,1),4.3125d0,'inclusive sum of squared weights')
  call expect(histy(1,21,1),1.5d0,'fiducial rate excludes zero jets')
  if (st_nevents/=3) stop 1
  call pyaend(1d0)
  call expect(histy_acc(1,1,1),1.75d0/3d0,'normalization counts cut failures')
  call expect(histy_err(1,1),sqrt(4.3125d0)/3d0,'signed-weight error')
contains
  subroutine momentum(k,pt,y,phi,m)
    integer, intent(in) :: k
    double precision, intent(in) :: pt,y,phi,m
    double precision :: mt
    mt=sqrt(pt*pt+m*m)
    phep(:,k)=[pt*cos(phi),pt*sin(phi),mt*sinh(y),mt*cosh(y),m]
  end subroutine
  subroutine expect(actual,expected,label)
    double precision, intent(in) :: actual,expected
    character(*), intent(in) :: label
    if (abs(actual-expected)>1d-10) then
      print *,label,actual,expected
      stop 1
    endif
  end subroutine
end program

! The real interface configures HepMC; this standalone Fortran check
! verifies that the analysis requests its declared common-block capacity.
subroutine hepmc_set_max_number_entries(entries)
  implicit none
  integer, intent(in) :: entries
  if(entries/=100000)stop 1
end subroutine
''')
        cls.exe = work / 'check'
        # Keep module compilation in dependency order, just as the MG5 driver.
        sources = [MCATNLO / 'srcCommon/HwU.f',
                   MCATNLO / 'PY8Analyzer/py8an_HwU_pp_tj_recoil.f',
                   MCATNLO / 'srcCommon/pineappl_interface_dummy.f',
                   work / 'check.f90']
        result = subprocess.run(
            ['gfortran', '-ffixed-line-length-132', '-fcheck=all',
             *includes, *map(str, sources), 'wrapper.o', 'fjcore.o',
             '-lstdc++', '-o', str(cls.exe)], cwd=work, capture_output=True,
            text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def run_analysis(self, mode):
        result = subprocess.run([str(self.exe), mode], cwd=self.work.name,
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return HwUList(str(Path(self.work.name) / 'MADatNLO.HwU'),
                       raw_labels=True)

    def test_selections_signed_weights_and_normalization(self):
        histograms = self.run_analysis('unsplit')
        self.assertEqual(len(histograms), 75)

    def test_collinear_split_does_not_change_jet_observables(self):
        unsplit = self.run_analysis('unsplit')
        split = self.run_analysis('split')
        for left, right in zip(unsplit, split):
            self.assertEqual((left.title, left.type), (right.title, right.type))
            for a, b in zip(left.bins, right.bins):
                self.assertEqual(a.boundaries, b.boundaries)
                self.assertEqual(a.wgts, b.wgts)

    def test_long_shower_history_preserves_final_top_and_histograms(self):
        reference = self.run_analysis('unsplit')
        long_record = self.run_analysis('long_record')
        for left, right in zip(reference, long_record):
            self.assertEqual((left.title, left.type), (right.title, right.type))
            for a, b in zip(left.bins, right.bins):
                self.assertEqual(a.boundaries, b.boundaries)
                self.assertEqual(a.wgts, b.wgts)
