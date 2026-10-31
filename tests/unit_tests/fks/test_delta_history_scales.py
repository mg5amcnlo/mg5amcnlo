"""Supported singular reconstruction scales for native-history Delta damping."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestDeltaHistoryScales(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='delta-history-scales-')
        cls.addClassCleanup(cls.tmp.cleanup)
        work = Path(cls.tmp.name)
        source = work/'probe.f90'
        source.write_text('''program probe
  use mcatnlo_delta_scales
  implicit none
  real(dp) :: pr(0:3),pe(0:3),pk(0:3),t,energy,costheta,check_t,check_z
  integer :: ir,ie,ib,fsr,status,check_status
  logical :: supported
  character(len=64) :: arg
  call get_command_argument(1,arg)
  read(arg,*)ir
  call get_command_argument(2,arg)
  read(arg,*)ie
  call get_command_argument(3,arg)
  read(arg,*)ib
  call get_command_argument(4,arg)
  read(arg,*)fsr
  call get_command_argument(5,arg)
  read(arg,*)energy
  call get_command_argument(6,arg)
  read(arg,*)costheta
  if(fsr==0)then
    pr=[500._dp,0._dp,0._dp,500._dp]
    pe=[energy,-sqrt(max(0._dp,1._dp-costheta**2))*energy,0._dp,costheta*energy]
    pk=[500._dp,0._dp,0._dp,-500._dp]
  else
    pr=[40._dp,40._dp,0._dp,0._dp]
    pe=[30._dp,0._dp,30._dp,0._dp]
    if(fsr==2)pe=[30._dp,30._dp*costheta,30._dp*sqrt(max(0._dp,1._dp-costheta**2)),0._dp]
    pk=[30._dp,-30._dp,0._dp,0._dp]
  endif
  call delta_history_scale(pr,pe,pk,ir,ie,ib,fsr/=0,.true., &
                             0._dp,0._dp,t,supported,status)
  call pythia_pt_lund(pr,pe,pk,ir,ie,fsr/=0,.true.,0._dp,0._dp, &
                      check_t,check_status,z_out=check_z)
  write(*,*)status,t,merge(1,0,supported),check_t,check_z,check_status
end program
''')
        cls.exe = work/'probe'
        subprocess.run([shutil.which('gfortran'), '-O2', '-std=f2008',
                        '-fcheck=all', '-ffpe-trap=invalid,zero,overflow',
                        str(ROOT/'Template/NLO/SubProcesses/mcatnlo_delta_scales.f90'),
                        str(source), '-o', str(cls.exe)], cwd=work,
                       check=True, capture_output=True, text=True)

    def probe(self, rad, emitted, born, fsr=False, energy=50, costheta=-.8):
        result = subprocess.run([str(self.exe),str(rad),str(emitted),str(born),
                                 str(int(fsr)),str(energy),str(costheta)], check=True,
                                capture_output=True, text=True)
        status,t,supported,check_t,z,check_status = map(float,result.stdout.split())
        self.assertEqual(status,0)
        self.assertEqual(check_status,0)
        self.assertEqual(t,check_t)
        self.assertIn(supported,(0,1))
        return t,z,bool(supported)

    def test_native_lund_scale_and_kinematic_support(self):
        t,z,supported = self.probe(1,21,1)
        self.assertTrue(supported)
        self.assertAlmostEqual(z,0.9,places=14)
        self.assertAlmostEqual(t*t,9000,places=9)
        t,z,supported = self.probe(1,21,1,fsr=True)
        self.assertTrue(supported)
        self.assertAlmostEqual(z,12/19,places=14)
        self.assertAlmostEqual(t*t,201600/361,places=10)

    def test_flavour_preserving_quark_and_gluon(self):
        for fsr in (False,True):
            for ids in [(1,21,1),(-1,21,-1),(21,21,21)]:
                self.assertTrue(self.probe(*ids,fsr=fsr)[2])

    def test_isr_crossing(self):
        for ids in [(21,2,-2),(21,-2,2),(2,2,21),(-2,-2,21)]:
            self.assertTrue(self.probe(*ids)[2])

    def test_fsr_pair_creation_and_reordered_quark(self):
        for ids in [(2,-2,21),(-2,2,21),(21,2,2)]:
            self.assertTrue(self.probe(*ids,fsr=True)[2])

    def test_flavour_incompatible_candidates_are_unsupported(self):
        cases=[(1,21,2),(21,21,1),(2,1,21),(21,2,2),(2,-2,21),(22,21,22)]
        for ids in cases:
            self.assertFalse(self.probe(*ids)[2])
        for ids in [(2,21,1),(2,2,21),(21,2,-2),(21,1,2)]:
            self.assertFalse(self.probe(*ids,fsr=True)[2])

    def test_soft_limit_and_invalid_kinematic_support(self):
        for rad,born in [(1,1),(21,21)]:
            self.assertTrue(self.probe(rad,21,born,energy=.001)[2])
        # A negative reconstructed Born mass is a dead sentinel, not an IR zero.
        t,_,supported=self.probe(1,21,1,energy=600)
        self.assertEqual(t,1e-5)
        self.assertFalse(supported)
        # Exact soft emission has z=1; source damping handles this boundary.
        self.assertFalse(self.probe(1,21,1,energy=0)[2])

    def test_deep_collinear_sentinel_retains_singular_history(self):
        for cosine in (1-1e-12,1.0):
            t,z,supported=self.probe(1,21,1,energy=100,costheta=cosine)
            self.assertEqual(t,1e-6)
            self.assertAlmostEqual(z,.8,places=14)
            self.assertTrue(supported)
            t,z,supported=self.probe(1,21,1,fsr=2,costheta=cosine)
            self.assertEqual(t,1e-6)
            self.assertAlmostEqual(z,4/7,places=11)
            self.assertTrue(supported)
