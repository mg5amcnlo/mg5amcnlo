"""Accepted FSR support beyond the global inverse map's AP coefficient."""
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TestPythiaFSRSupport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='macnlops-fsr-')
        cls.addClassCleanup(cls.directory.cleanup)
        cls.work = Path(cls.directory.name)
        driver = cls.work/'driver.f90'
        driver.write_text('''program test
  use mc_pythia8_kernel
  implicit none
  integer :: parent,first,second,nf,status,ios
  double precision :: z,t,mp,mf,ms,cut,w
  do
    read(*,*,iostat=ios) parent,first,second,z,t,mp,mf,ms,cut,nf
    if(ios/=0) exit
    call py8_fsr_ordinary_weight(parent,first,second,z,t,mp,mf,ms,cut,nf,w,status)
    write(*,'(I2,1X,ES25.17)') status,w
  enddo
end program test
''')
        subprocess.run([
            'gfortran', '-std=f2008', '-O2', '-fcheck=all',
            '-ffpe-trap=invalid,zero,overflow',
            str(ROOT/'Template/NLO/SubProcesses/mc_pythia8_kernel.f90'),
            str(driver), '-o', str(cls.work/'driver')], cwd=cls.work,
            check=True, capture_output=True, text=True)

    def evaluate(self, parent, first, second, z=.6, t=100., mp=0.,
                 mf=0., ms=0., cut=.5867333369950678, nf=5):
        values = (parent, first, second, z, t, mp, mf, ms, cut, nf)
        output = subprocess.run([str(self.work/'driver')],
            input=' '.join(map(str, values))+'\n', text=True,
            capture_output=True, check=True).stdout.split()
        return int(output[0]), float(output[1])

    def test_massless_channels_and_physical_gluon_cut(self):
        for parent, first, second in ((21,21,21),(2,2,21),(-2,-2,21),
                                      (2,21,2),(-2,21,-2),(21,2,-2)):
            self.assertEqual(self.evaluate(parent,first,second), (0,1.))
            self.assertEqual(self.evaluate(parent,first,second,t=.25), (0,0.))
        self.assertEqual(self.evaluate(2,-2,21)[0], 2)
        self.assertEqual(self.evaluate(21,2,2)[0], 2)

    def test_massive_quark_cut_after_daughter_rescaling(self):
        # Locate the physical boundary independently in the branch's
        # virtual-pair mass.  Before rescaling t=z(1-z)(Mpair^2-mq^2);
        # afterwards pTorig^2=t*(1-mq^2/Mpair^2)^2.  This boundary can
        # lie far above the nominal evolution cutoff for a top quark.
        cut = .5867333369950678
        for flavour, mass in ((4,1.2),(5,4.),(6,173.)):
            for z in (.1,.5,.9):
                low, high = mass*mass, mass*mass+1.e6
                for unused in range(100):
                    pair2 = (low+high)/2
                    evolution = z*(1-z)*(pair2-mass*mass)
                    physical = evolution*(1-mass*mass/pair2)**2
                    if physical < cut*cut:
                        low = pair2
                    else:
                        high = pair2
                boundary_t = z*(1-z)*(high-mass*mass)
                self.assertGreater(boundary_t, cut*cut)
                for factor, expected in ((.999,0.),(1.001,1.)):
                    self.assertEqual(self.evaluate(flavour,flavour,21,
                        z=z,t=factor*boundary_t,mp=mass,mf=mass,cut=cut),
                        (0,expected))
                    self.assertEqual(self.evaluate(-flavour,21,-flavour,
                        z=1-z,t=factor*boundary_t,mp=mass,ms=mass,cut=cut),
                        (0,expected))

    def test_massless_pair_species_follow_configured_flavour_draw(self):
        # Internal c/b masses set alpha-s evolution ranges; with m0=0
        # they do not forbid massless c/b pair production below 1.2/4 GeV.
        for flavour in (1,2,3,4,5):
            self.assertEqual(self.evaluate(21,flavour,-flavour,t=1.), (0,1.))
            self.assertEqual(self.evaluate(21,-flavour,flavour,t=1.), (0,1.))
        self.assertEqual(self.evaluate(21,5,-5,nf=4), (0,0.))
        self.assertEqual(self.evaluate(21,1,-1,nf=0), (0,0.))
        self.assertEqual(self.evaluate(21,6,-6,nf=5), (0,0.))

    def test_massive_pair_requires_its_own_inverse_map(self):
        self.assertEqual(self.evaluate(21,5,-5,mf=4.,ms=4.)[0], 4)
        # Species support is an exact zero even if the unused massive
        # map would otherwise be unsupported (e.g. top pair production).
        self.assertEqual(self.evaluate(21,6,-6,mf=173.,ms=173.), (0,0.))

    def test_invalid_inputs_fail_before_evaluating_a_probability(self):
        self.assertEqual(self.evaluate(21,21,21,z=0.)[0], 1)
        self.assertEqual(self.evaluate(21,21,21,z=math.nan)[0], 1)
        self.assertEqual(self.evaluate(21,21,21,nf=6)[0], 1)


if __name__ == '__main__':
    unittest.main()
