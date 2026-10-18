"""Check FxFx finite terms against independent forward splitting integrals."""

import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


CF, CA, NF = 4 / 3, 3, 5
K = CA * (67 / 18 - math.pi**2 / 6) - NF * 5 / 9


def simpson(function, low, high, intervals=1024):
    step = (high - low) / intervals
    total = function(low) + function(high)
    total += sum((4 if i % 2 else 2) * function(low + i * step)
                 for i in range(1, intervals))
    return total * step / 3


def integrated_splitting(ratio, kind):
    """Integrate z*P(z), using u=-log(1-z) to resolve the soft pole."""
    epsilon = 2 * ratio / (math.sqrt(4 + ratio**2) + ratio)

    def integrand(u):
        z = -math.expm1(-u)
        one_minus_z = math.exp(-u)
        if kind == 1:
            # z*(P_qq + P_gq), including the Jacobian dz/du = 1-z.
            return CF * (z * (1 + z*z)
                         + (1 + one_minus_z**2) * one_minus_z)
        gg = 2 * CA * (z*z + one_minus_z**2 + z*z*one_minus_z**2)
        qq = NF * z * (z*z + one_minus_z**2) * one_minus_z
        return gg + qq

    return simpson(integrand, 0, -math.log(epsilon))


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestFxFxSudakov(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_fxfx_sudakov_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.work = Path(cls.tempdir.name)
        (cls.work / 'orders.inc').write_text(
            '      integer nsplitorders\n      parameter(nsplitorders=1)\n')
        (cls.work / 'cuts.inc').write_text(
            '      integer maxjetflavor\n      parameter(maxjetflavor=5)\n')
        names = ('Gamma', 'sudakov_exp', 'expanded_sudakov_exp', 'QCDsudakov')
        (cls.work / 'kernel.f').write_text('\n'.join(
            fortran_routine(TEMPLATE / 'cluster.f', name) for name in names))
        (cls.work / 'driver.f90').write_text('''program check
use FKSParams
implicit none
real*8 q,hard,m,alpha,g1,g2,e,e1,rate,Q1,mass,sudakov_exp,expanded_sudakov_exp,Gamma
real*8 qe,qe1,masses(1)
integer kind,finite,itype,mode,types(0:1)
external Gamma
character(1024) card
common /fixture_alpha/ alpha
common /to_sud_exp/ Q1,mass,itype,mode
call get_command_argument(1,card)
if (len_trim(card)>0) then
  call FKSParamReader(trim(card),.false.,.true.)
  write(*,*) FxFxPythia8FiniteTerms
  stop
endif
! Keep one process alive to exercise cached values across scheme changes.
do
  read(*,*,end=100) finite,alpha,q,hard,kind,m
  FxFxPythia8FiniteTerms=finite/=0
  Q1=hard
  mass=m
  itype=kind
  mode=1
  g1=Gamma(q)
  mode=2
  g2=Gamma(q)
  e=sudakov_exp(q,hard,kind,m)
  e1=expanded_sudakov_exp(q,hard,kind,m)
  types=[1,kind]
  masses=m
  qe=0d0
  qe1=0d0
  call QCDsudakov(q,hard,q,1,types,masses,qe,qe1,.false.)
  write(*,'(6es25.16)') g1,g2,e,e1,qe,qe1
enddo
100 continue
end program

double precision function alphas(q)
implicit none
real*8 q,alpha
common /fixture_alpha/ alpha
alphas=alpha
end function

double precision function get_mass_from_id(i)
implicit none
integer i
! Decouple the massive sixth flavour to isolate the massless kernels.
get_mass_from_id=0d0
if(i==6) get_mass_from_id=1d30
end function
''')
        cls.exe = cls.work / 'check'
        result = subprocess.run([
            shutil.which('gfortran'), '-O0', '-std=legacy', '-fcheck=all',
            '-ffixed-line-length-none', '-ffpe-trap=invalid,zero,overflow',
            '-I', str(cls.work), str(TEMPLATE / 'FKSParams.f90'),
            'kernel.f', 'driver.f90', str(ROOT / 'Template/NLO/Source/dgauss.f'),
            '-o', str(cls.exe)], cwd=cls.work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def evaluate(self, points):
        data = ''.join(' '.join(map(str, point)) + '\n' for point in points)
        result = subprocess.run([str(self.exe)], input=data,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [list(map(float, line.split())) for line in result.stdout.splitlines()]

    def test_matches_full_splitting_integrals(self):
        points = [(1, .12, r*80, 80, kind, 0)
                  for kind in (1, 3) for r in (.001, .05, .25, .5, .999)]
        for point, actual in zip(points, self.evaluate(points)):
            _, alpha, q, hard, kind, _ = point
            expected = -integrated_splitting(q/hard, kind) / (math.pi*q)
            self.assertAlmostEqual(actual[0], expected, delta=abs(expected)*2e-9)
            colour = CF if kind == 1 else CA
            a2 = -colour*alpha**2*K*math.log(hard/q)/(math.pi**2*q)
            self.assertAlmostEqual(actual[1], alpha*expected+a2,
                                   delta=abs(expected)*2e-9)

    def test_preserves_nll_coefficients_and_legacy_option(self):
        points = [(finite, .12, r*80, 80, kind, 0)
                  for kind in (1, 3) for r in (1e-7, .25, .8)
                  for finite in (0, 1)]
        results = self.evaluate(points)
        for i in range(0, len(points), 2):
            _, alpha, q, hard, kind, _ = points[i]
            colour = CF if kind == 1 else CA
            bracket = math.log(hard/q) - (.75 if kind == 1 else 11/12)
            legacy = -2*(colour*bracket + (NF/6 if kind == 3 else 0))/(math.pi*q)
            self.assertAlmostEqual(results[i][0], legacy, delta=abs(legacy)*1e-12)
            if q/hard < 1e-6:
                # The difference in the integrand vanishes as k/Q, so A1
                # and B1 agree; mode 2 has the same A2 correction in both.
                difference = abs((results[i+1][0]-legacy)*math.pi*q/2)
                self.assertLess(difference, 10*q/hard)
                for result in results[i:i+2]:
                    a2 = -colour*alpha**2*K*math.log(hard/q)/(math.pi**2*q)
                    self.assertAlmostEqual(result[1]-alpha*result[0], a2,
                                           delta=abs(a2)*1e-11)

    def test_first_order_and_nlo_compensation(self):
        # A separate process per coupling avoids the existing fixed-coupling
        # assumption of the scale caches; generation never changes alpha_s.
        for kind in (1, 3):
            errors = []
            q, hard = 21.5, 80.419
            expected_e1 = simpson(lambda t: -integrated_splitting(
                math.exp(t)/hard, kind)/math.pi, math.log(q), math.log(hard), 128)
            for alpha in (.001, .0005):
                g1, g2, e, e1, qe, qe1 = self.evaluate(
                    [(1, alpha, q, hard, kind, 0)])[0]
                self.assertAlmostEqual(e1, expected_e1, delta=2e-9)
                self.assertAlmostEqual(qe, e, places=14)
                self.assertAlmostEqual(qe1, e1, places=14)
                colour = CF if kind == 1 else CA
                a2_integral = -colour*alpha**2*K*math.log(hard/q)**2/(2*math.pi**2)
                self.assertAlmostEqual(e, alpha*e1+a2_integral, delta=1e-11)
                errors.append(abs(math.exp(2*e)*(1-2*alpha*e1)-1))
            self.assertAlmostEqual(errors[0]/errors[1], 4, delta=.02)

    def test_scheme_is_part_of_both_cache_keys(self):
        for kind in (1, 3):
            points = [(finite, .12, 20, 80, kind, 0) for finite in (0, 1, 0, 1)]
            old, new, old_cached, new_cached = self.evaluate(points)
            self.assertEqual(old, old_cached)
            self.assertEqual(new, new_cached)
            self.assertLess(new[2], old[2])
            self.assertLess(new[3], old[3])

    def test_massive_kernels_are_unchanged(self):
        points = [(finite, .12, 20, 200, kind, 100)
                  for kind in (2, 4, 5) for finite in (0, 1)]
        results = self.evaluate(points)
        for old, new in zip(results[::2], results[1::2]):
            self.assertEqual(old, new)

    def test_card_default_and_explicit_selection(self):
        for text, expected in [('', 'F'), ('#FxFxPythia8FiniteTerms\n.true.\n', 'T'),
                               ('#FxFxPythia8FiniteTerms\n.false.\n', 'F')]:
            card = self.work / 'card.dat'
            card.write_text(text)
            result = subprocess.run([str(self.exe), str(card)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.strip(), expected)

    def test_massless_rates_and_unit_endpoint(self):
        points = [(1, .12, r*80, 80, kind, 0)
                  for kind in (1, 3) for r in (.001, .25, .9, 1)]
        for point, actual in zip(points, self.evaluate(points)):
            self.assertTrue(all(value <= 0 for value in actual))
            if point[2] == point[3]:
                self.assertTrue(all(value == 0 for value in actual))
