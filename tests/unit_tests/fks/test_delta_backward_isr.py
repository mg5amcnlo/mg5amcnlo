"""MC@NLO-Delta backward-ISR prototype: PYTHIA8 alpha_s and the quark leg probability."""

import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import fortran_routine

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'
ROUTINES = ('delta_backward_leg', 'delta_group_xf', 'delta_backward_omz',
            'delta_alphas_lambdas', 'delta_shower_alphas', 'delta_alphas_nf',
            'delta_alphas_lambda', 'get_parton_id')
# PYTHIA 8.318 AlphaStrong (CMW off, nfmax 5), including the freezing below
# 1.33 (second order) or 1.07 (first order) Lambda_3. Second order with 0.118
# as in the NNPDF4.0 NLO profile; first order with 0.1365, the archived table's.
PYTHIA_ALPHAS = {
    (.118, 2): ((0.4006386434, 4.352238399), (1.20029065, 0.4354249345),
                (1.602942725, 0.3475180847), (3.004981386, 0.2559199742),
                (5.201260188, 0.2110617612), (12.02283687, 0.1708827951),
                (91.08104981, 0.1180209114), (1502.293315, 0.08313901628)),
    (.1365, 1): ((0.3, 10.318439941464703), (1.0, 0.62114254126870971),
                 (1.51, 0.45464154807199286), (3.0, 0.32153954080968622),
                 (4.81, 0.26767035945864026), (10.0, 0.21603034203324506),
                 (91.188, 0.1365), (500.0, 0.10635602188389676))}
CF, TR, MC, MB = 4. / 3., .5, 1.5, 4.8

# Toy x*f(x, mu), rising with scale at small x. Flavour 4 starts at 2 GeV, so a
# grouped (2, 4) leg also changes its gluon-feed multiplicity there.
TOY_PDF = '''
      double precision function pdg2pdf(ih,ipdg,ibeam,x,xmu)
      implicit none
      integer ih,ipdg,ibeam
      double precision x,xmu,l
      l=log(xmu)
      if (ipdg.eq.0) then
         pdg2pdf=2d0*x**(-0.1d0-0.02d0*l)*(1d0-x)**(5d0+0.3d0*l)
      elseif (abs(ipdg).eq.4) then
         pdg2pdf=0d0
         if (xmu.gt.2d0) pdg2pdf=0.1d0*(1d0-4d0/xmu**2)*x**(-0.1d0)*
     $        (1d0-x)**6
      else
         pdg2pdf=x**(0.1d0-0.02d0*l)*(1d0-x)**(3d0+0.3d0*l)
      endif
      pdg2pdf=pdg2pdf/x
      end
'''

DRIVER = '''
      program check
      use backward_test
      implicit none
      character(1) mode
      integer n,flav(3),beam,order
      double precision x,alpha,lambda(3:5)
      type(delta_connections) leg
      leg%count=1
      leg%partner(1)=2
      leg%sudakov_type(1)=1
      do
         read(*,*,end=10) mode
         if (mode.eq.'c') then
            read(*,*) alpha,order
            call delta_alphas_lambdas(alpha,order,lambda)
         elseif (mode.eq.'a') then
            read(*,*) x
            write(*,'(es24.16)') delta_shower_alphas(x,lambda,order)
         else
            read(*,*) n,flav(1:n),beam,x,leg%stop(1),leg%start(1),
     $           leg%mass(1)
            write(*,'(es24.16)') delta_backward_leg(flav(1),flav(1:n),
     $           beam,x,leg,lambda,order)
         endif
      enddo
 10   continue
      end
'''


def toy_xf(pid, x, mu):
    l = math.log(mu)
    if pid == 21:
        return 2. * x**(-.1 - .02 * l) * (1. - x)**(5. + .3 * l)
    if abs(pid) == 4:
        return .1 * (1. - 4. / mu**2) * x**(-.1) * (1. - x)**6 if mu > 2. else 0.
    return x**(.1 - .02 * l) * (1. - x)**(3. + .3 * l)


def pythia_alphas(scale, alpha_mz=.118, order=2):
    b = {nf: (33. - 2. * nf, 6. * (153. - 19. * nf) / (33. - 2. * nf)**2 if order == 2 else 0.)
         for nf in (3, 4, 5)}

    def value(nf, q, lam):
        l = 2. * math.log(q / lam)
        return 12. * math.pi / (b[nf][0] * l) * (1. - b[nf][1] * math.log(l) / l)

    def solve(nf, q, target, lam):
        for _ in range(10):
            l = 2. * math.log(q / lam)
            lam = q * math.exp(-6. * math.pi / (b[nf][0] * target / (1. - b[nf][1] * math.log(l) / l)))
        return lam

    lam5 = solve(5, 91.188, alpha_mz, 91.188 * math.exp(-6. * math.pi / (23. * alpha_mz)))
    lam4 = solve(4, MB, value(5, MB, lam5), lam5)
    lam3 = solve(3, MC, value(4, MC, lam4), lam4)
    q = max(scale, (1.33 if order == 2 else 1.07) * lam3)
    return value(5, q, lam5) if q > MB else value(4, q, lam4) if q > MC else value(3, q, lam3)


GL8 = ((-.9602898564975363, .1012285362903763), (-.7966664774136267, .2223810344533745),
       (-.5255324099163290, .3137066458778873), (-.1834346424956498, .3626837833783620),
       (.1834346424956498, .3626837833783620), (.5255324099163290, .3137066458778873),
       (.7966664774136267, .2223810344533745), (.9602898564975363, .1012285362903763))


def integrate(f, edges, panels):
    """Composite 8-point Gauss-Legendre over consecutive edges."""
    total = 0.
    for a, b in zip(edges[:-1], edges[1:]):
        h = (b - a) / panels
        for k in range(panels):
            lo = a + k * h
            total += sum(.5 * h * w * f(lo + .5 * h * (u + 1.)) for u, w in GL8)
    return total


def reference_leg(flavs, x, q, start, mass, alpha_mz=.118, order=2):
    """F*R*exp(C) on fine grids, split at the coupling and toy-flavour thresholds."""
    a = lambda t: pythia_alphas(math.sqrt(t), alpha_mz, order) / (2. * math.pi)
    omz = lambda t: .5 * t / mass**2 * (math.sqrt(1. + 4. * mass**2 / t) - 1.)
    lo, hi = 2. * math.log(q), 2. * math.log(start)
    edges = [lo] + [2. * math.log(c) for c in (MC, 2., MB) if lo < 2. * math.log(c) < hi] + [hi]

    def forward(u):
        o = omz(math.exp(u))
        return a(math.exp(u)) * CF * (-2. * math.log(o) - 1.5 * (1. - o)**2)

    def boundary(u):
        t = math.exp(u)
        pt, o0 = math.sqrt(t), omz(t)
        den = sum(toy_xf(f, x, pt) for f in flavs)
        nact = sum(1 for f in flavs if toy_xf(f, x, pt) > 0.)
        ol = min(1. - x, o0)
        val = CF * (1. - o0) * o0
        if o0 > 1. - x:
            val -= CF * (2. * math.log(o0 / ol) - (o0 - ol) - .5 * ((1. - ol)**2 - (1. - o0)**2))

        def inner(o):
            z = 1. - o
            y = x / z
            fq = sum(toy_xf(f, y, pt) for f in flavs)
            return CF * (1. + z * z) / o * (fq / den - 1.) + nact * TR * (z * z + o * o) * toy_xf(21, y, pt) / den
        return a(t) * (val + integrate(inner, [0., ol], 16))

    ratio = sum(toy_xf(f, x, q) for f in flavs) / sum(toy_xf(f, x, start) for f in flavs)
    return math.exp(integrate(boundary, edges, 12) - integrate(forward, edges, 12)) * ratio


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestDeltaBackwardISR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_delta_backward_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        work = Path(cls.tempdir.name)
        source = (TEMPLATE / 'montecarlocounter.f').read_text()
        # The production thresholds and connection type, without the rest of
        # the module's state.
        declarations = re.search(r'^c     PYTHIA8.s default AlphaStrong.*?'
                                 r'^      end type delta_connections\n', source,
                                 re.M | re.S).group(0)
        (work / 'check.f').write_text(
            '      module backward_test\n      implicit none\n'
            '      integer, parameter :: nincoming=2\n' + declarations
            + '      contains\n' + ''.join(fortran_routine(TEMPLATE / 'montecarlocounter.f', name)
                                     for name in ROUTINES)
            + '      end module backward_test\n' + TOY_PDF + DRIVER)
        cls.executable = work / 'check'
        result = subprocess.run([shutil.which('gfortran'), '-O2', '-fcheck=all',
                                 '-ffixed-line-length-none', str(work / 'check.f'),
                                 '-o', str(cls.executable)], cwd=work,
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def run_lines(self, lines):
        result = subprocess.run([str(self.executable)], input=''.join(lines),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [float(v) for v in result.stdout.split()]

    def leg(self, flavs, x, q, start, mass, coupling=(.118, 2)):
        return self.run_lines(['c\n%r %d\n' % coupling,
                               'b\n%d %s 1 %.17g %.17g %.17g %.17g\n' % (
                                   len(flavs), ' '.join(map(str, flavs)), x, q, start, mass)])[0]

    def test_alpha_s_reproduces_pythia(self):
        for coupling, references in PYTHIA_ALPHAS.items():
            values = self.run_lines(['c\n%r %d\n' % coupling] +
                                    ['a\n%.17g\n' % scale for scale, _ in references])
            for (scale, expected), value in zip(references, values):
                with self.subTest(coupling=coupling, scale=scale):
                    self.assertAlmostEqual(value / expected, 1., delta=2e-9)

    def test_fixed_rules_reproduce_the_identity(self):
        # The smooth toy keeps the rules within 1e-4, also across the coupling
        # and toy-flavour thresholds, from small to large x.
        for flavs, x, q, start, mass in (
                ((2,), .007, 20., 60., 91.19), ((-1,), .02, 10., 45., 91.19),
                ((2, 4), .003, 5.5, 80., 120.), ((2, 4), .007, 3., 50., 91.19),
                ((-2, -4), .4, 1.6, 30., 300.), ((1,), .9, 8., 40., 1000.)):
            with self.subTest(flavs=flavs, x=x, q=q):
                value = self.leg(flavs, x, q, start, mass)
                # The toy PDF can exceed a probability at large x; the leg is bounded.
                expected = min(reference_leg(flavs, x, q, start, mass), 1.)
                self.assertAlmostEqual(value / expected, 1., delta=2e-4)
                self.assertGreater(value, 0.)

    def test_first_order_running(self):
        for flavs, x, q, start, mass in (((2,), .007, 20., 60., 91.19),
                                         ((2, 4), .007, 3., 50., 91.19)):
            with self.subTest(flavs=flavs, q=q):
                value = self.leg(flavs, x, q, start, mass, coupling=(.1365, 1))
                self.assertAlmostEqual(
                    value / reference_leg(flavs, x, q, start, mass, .1365, 1), 1., delta=2e-4)
                # The larger one-loop coupling suppresses emission-free evolution.
                self.assertLess(value, self.leg(flavs, x, q, start, mass))

    def test_no_evolution_and_table_fallbacks(self):
        self.assertEqual(self.leg((2,), .01, 30., 30., 91.19), 1.)
        # The lowest ln(t) node lies at 1.44 GeV, where this flavour has no PDF.
        self.assertEqual(self.leg((4,), .01, 1.2, 6., 91.19), -1.)
        # Gluons, and quarks below the table's 1.01 GeV continuation, keep the table.
        self.assertEqual(self.leg((21,), .01, 10., 30., 91.19), -1.)
        self.assertEqual(self.leg((2,), .01, 1., 30., 91.19), -1.)


if __name__ == '__main__':
    unittest.main()
