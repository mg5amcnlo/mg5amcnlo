"""Production support and shared PYTHIA photon-conversion bounds.

Compare the scalar support to independently evaluated shower inequalities;
these tests do not constitute an end-to-end NLO QED shower validation.
"""

from pathlib import Path
import math
import random
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine

SCRIPT = ROOT / 'Template/NLO/MCatNLO/Scripts/MCatNLO_MadFKS_PYTHIA8.Script'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestPythia8QEDSupport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_qed_support_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        work = Path(cls.tempdir.name)
        (work / 'orders.inc').write_text(
            '      integer nsplitorders\n      parameter(nsplitorders=1)\n')
        (work / 'support.f').write_text(fortran_routine(
            TEMPLATE / 'montecarlocounter.f', 'py8_global_fsr_support'))
        (work / 'driver.f90').write_text("""program check
use FKSParams
implicit none
real*8 z,t,s,m2,mr2,q2,mmax
integer conversion,ios
logical py8_global_fsr_support,accepted
character(1024) card
call get_command_argument(1,card)
if (len_trim(card)>0) then
  call FKSParamReader(trim(card),.false.,.true.)
  write(*,*) Pythia8MMaxGamma
else
  do
    read(*,*,iostat=ios) z,t,s,m2,mr2,q2,conversion,mmax
    if (ios/=0) exit
    accepted=py8_global_fsr_support(z,t,s,m2,mr2,q2,conversion==1,mmax)
    write(*,*) merge(1,0,accepted)
  enddo
endif
end program
""")
        cls.executable = work / 'check'
        result = subprocess.run([
            shutil.which('gfortran'), '-O0', '-fcheck=all',
            '-ffpe-trap=invalid,zero,overflow', '-ffixed-line-length-none',
            '-I', str(work), str(TEMPLATE / 'FKSParams.f90'),
            'support.f', 'driver.f90', '-o', str(cls.executable)],
            cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)

    def support(self, points):
        data = ''.join(' '.join(map(str,point))+'\n' for point in points)
        result = subprocess.run([str(self.executable)], input=data,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return list(map(int,result.stdout.split()))

    def test_conversion_bound_and_singular_limit(self):
        # t = z(1-z) m_pair^2; all points have abundant global phase space.
        points = [(0.5, q2/4, 1e6, 0, 0, q2, c, 10)
                  for c in (0,1) for q2 in (1e-30,99.999,100,100.001)]
        self.assertEqual(self.support(points), [1,1,1,1,1,1,0,0])
        # Changing the shared bound must change only photon conversion.
        self.assertEqual(self.support([(0.5,25,1e6,0,0,100,1,11)]), [1])

    def test_global_mass_and_energy_boundaries(self):
        # Corrected global dipole mass is (1000-900)^2=10000.
        # Last two cases would trigger negative square roots in the old code.
        points = [(0.5,2499,1e6,0,810000,9996,0,10),
                  (0.5,2501,1e6,0,810000,10004,0,10),
                  (0.1,50,1e6,0,810000,4000,0,10),
                  (0.5,1,1e6,0,1e6,4,0,10),
                  (0.5,1,1e6,0,-1,4,0,10)]
        self.assertEqual(self.support(points), [1,0,0,0,0])

    def test_massless_recoil_roundoff_retains_collinear_support(self):
        # Values reproduced by a massless lepton recoil in generated
        # e+e- -> mu+mu- [QED] endpoint checks at shat=1e6 GeV^2.
        points = [(0.7,1e-10,1e6,0,mr2,1e-10/(0.7*0.3),0,10)
                  for mr2 in (-1.3e-10,-2.9e-11,0,2.9e-11,-1e-5)]
        self.assertEqual(self.support(points), [1,1,1,1,0])

    def test_matches_independent_pythia_trial_constraints(self):
        rng = random.Random(93812)
        points, expected = [], []
        for _ in range(250):
            z = rng.uniform(0.01,0.99)
            m, mr = rng.uniform(0,150), rng.uniform(0,750)
            mdip2 = (1000-mr)**2-m*m
            t = rng.uniform(0.00001,0.35)*mdip2
            q2 = m*m+t/(z*(1-z))
            # Literal trial constraints, evaluated independently with roots.
            if 4*t > mdip2:
                accepted = False
            else:
                zmin = 0.5-math.sqrt(0.25-t/mdip2)
                accepted = zmin <= z <= 1-zmin and (
                    q2*1e6 <= z*(1-z)*(1e6+q2-mr*mr)**2)
            points.append((z,t,1e6,m*m,mr*mr,q2,0,10))
            expected.append(int(accepted))
        self.assertEqual(self.support(points), expected)

    def test_fortran_and_shower_script_read_same_bound(self):
        work = Path(self.tempdir.name)
        for value in ('7.5d0', '2.3D1', '.5', '10.', '10.d0', '5000'):
            card = work / 'bound.dat'
            card.write_text('#Pythia8MMaxGamma\n'+value+'\n')
            f = subprocess.run([str(self.executable), str(card)],
                               capture_output=True, text=True)
            shell = subprocess.run(['bash','-c',
                'source "$1"; pythia8_mmaxgamma "$2"', '_', str(SCRIPT), str(card)],
                capture_output=True, text=True)
            self.assertEqual(f.returncode,0,f.stdout+f.stderr)
            self.assertEqual(shell.returncode,0,shell.stdout+shell.stderr)
            self.assertEqual(float(f.stdout),float(shell.stdout))
        for value in ('0', '-1', '5001', 'garbage'):
            card.write_text('#Pythia8MMaxGamma\n'+value+'\n')
            shell = subprocess.run(['bash','-c',
                'source "$1"; pythia8_mmaxgamma "$2"', '_', str(SCRIPT), str(card)],
                capture_output=True,text=True)
            self.assertNotEqual(shell.returncode,0,value)
