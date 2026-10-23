"""Accepted ISR coefficients and support, distinct from MC subtraction D.

References are the physical AP coefficients and global-II kinematics of
SimpleSpaceShower 8.318.  The module deliberately reports the forced heavy
threshold conversion and missing massive-map information as unsupported.
"""
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TestPythiaISRScalar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='macnlops-isr-')
        cls.work = Path(cls.directory.name)
        driver = cls.work / 'driver.f90'
        driver.write_text('''program test
  use mc_pythia8_kernel
  implicit none
  integer :: d,m,s,status,ios,imap
  double precision :: z,t,shat,x,xmax,mc,mb,ms,w
  do
    read(*,*,iostat=ios) d,m,s,z,t,shat,x,xmax,mc,mb,ms,imap
    if(ios/=0) exit
    call py8_isr_ordinary_weight(d,m,s,z,t,shat,x,xmax,mc,mb,ms,imap/=0,w,status)
    write(*,'(I2,1X,ES25.17)') status,w
  enddo
end program test
''')
        subprocess.run([
            'gfortran', '-std=f2008', '-O2', '-fcheck=all',
            '-ffpe-trap=invalid,zero,overflow',
            str(ROOT/'Template/NLO/SubProcesses/mc_pythia8_kernel.f90'),
            str(driver), '-o', str(cls.work/'driver')],
            cwd=cls.work, check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def evaluate(self, daughter, mother, sister, z=.6, t=100., shat=1e6,
                 x=.1, xmax=1., mc=1.5, mb=4.8, ms=0., massive_map=False):
        values = (daughter, mother, sister, z, t, shat, x, xmax, mc, mb, ms,
                  int(massive_map))
        output = subprocess.run([str(self.work/'driver')],
            input=' '.join(map(str, values))+'\n', text=True,
            capture_output=True, check=True).stdout.split()
        return int(output[0]), float(output[1])

    def test_light_channels_and_charge_conjugates(self):
        for channel in ((21,21,21),(2,2,21),(-2,-2,21),
                        (2,21,-2),(-2,21,2),(21,2,2),(21,-2,-2)):
            self.assertEqual(self.evaluate(*channel), (0,1.))
        self.assertEqual(self.evaluate(2,-2,21)[0], 2)
        self.assertEqual(self.evaluate(2,21,2)[0], 2)

    def test_heavy_quark_coefficients_against_absolute_ap_functions(self):
        cf,tr = 4/3, .5
        for flavour,mass in ((4,1.5),(5,4.8)):
            for sign in (-1,1):
                for z in (.2,.55,.8):
                    t=200.
                    status, ratio = self.evaluate(sign*flavour,sign*flavour,21,z=z,t=t)
                    massless=cf*(1+z*z)/(1-z)
                    physical=massless-cf*2*z*(1-z)*mass*mass/t
                    self.assertEqual(status,0)
                    self.assertAlmostEqual(ratio*massless,physical,places=13)
                    status, ratio = self.evaluate(sign*flavour,21,-sign*flavour,z=z,t=t)
                    massless=tr*(z*z+(1-z)**2)
                    physical=massless+tr*2*z*(1-z)*mass*mass/t
                    self.assertEqual(status,0)
                    self.assertAlmostEqual(ratio*massless,physical,places=13)

    def test_zero_particle_masses_do_not_remove_pythia_heavy_floors(self):
        status, ratio = self.evaluate(5,5,21,mc=0.,mb=0.)
        self.assertEqual(status,0)
        expected=1-2*.6*.4**2*4.**2/(100*(1+.6**2))
        self.assertAlmostEqual(ratio,expected,places=15)
        self.assertNotEqual(ratio,1.)

    def test_threshold_is_not_misrepresented_as_an_absent_kernel(self):
        self.assertEqual(self.evaluate(4,21,-4,t=3.)[0],3)
        self.assertEqual(self.evaluate(5,21,-5,t=45.)[0],3)
        self.assertEqual(self.evaluate(5,5,21,t=2*4.8**2)[0],0)

    def test_heavy_gluon_emission_is_absent_below_conversion_window(self):
        # The threshold algorithm selects g->QQbar before the ordinary
        # channel choice. It never emits Q->Qg in this interval.
        for sign in (-1,1):
            for flavour,t in ((4,3.),(5,45.),(5,3.04**2)):
                self.assertEqual(self.evaluate(sign*flavour,sign*flavour,21,t=t),(0,0.))
        self.assertEqual(self.evaluate(5,5,21,t=2*4.8**2)[0],0)

    def test_massive_sister_requires_the_corresponding_measure(self):
        self.assertEqual(self.evaluate(2,21,-2,ms=.33)[0],4)
        self.assertEqual(self.evaluate(2,21,-2,ms=.33,massive_map=True), (0,1.))

    def test_global_ii_support_and_additional_heavy_pair_bound(self):
        z,shat=.6,1000.
        # pT(real)^2=0 at the massless global-II kinematic boundary.
        boundary=shat*(1-z)**2/z
        self.assertEqual(self.evaluate(2,2,21,z=z,t=boundary*1.001,shat=shat),(0,0.))
        self.assertEqual(self.evaluate(2,2,21,z=z,t=boundary*.999,shat=shat),(0,1.))
        # Ordinary kinematics is allowed, but the required earlier QQbar
        # pair has insufficient invariant mass for a heavy mother.
        self.assertEqual(self.evaluate(21,5,5,z=.6,t=100.,shat=1000.,ms=4.8,
                                       massive_map=True),(0,0.))
        self.assertEqual(self.evaluate(21,5,5,z=.6,t=100.,shat=1000.,ms=0.),(0,1.))

    def test_heavy_x_bound_and_invalid_input(self):
        mass,shat=4.8,1000.
        r=mass/math.sqrt(shat)
        zmax=(1-r)/(1+r*(1-r))
        self.assertEqual(self.evaluate(5,5,21,z=.98*zmax,t=50.,shat=shat,
                                      x=.91*zmax),(0,0.))
        self.assertEqual(self.evaluate(2,2,21,x=.7,z=.6),(0,0.))
        self.assertEqual(self.evaluate(5,5,21,shat=4.,t=100.),(0,0.))
        for extra in ({'z':0.},{'z':1.},{'t':0.},{'shat':0.},{'ms':-1.}):
            self.assertEqual(self.evaluate(2,2,21,**extra)[0],1)


if __name__ == '__main__':
    unittest.main()
