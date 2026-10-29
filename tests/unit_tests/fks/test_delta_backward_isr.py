"""MC@NLO-Delta backward-ISR prototype: PYTHIA8 alpha_s and incoming quark and gluon legs."""

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
ROUTINES = ('delta_backward_leg', 'delta_line_exponents', 'delta_backward_ratios',
            'delta_group_xf', 'delta_quark_xf', 'delta_active_quarks', 'delta_backward_omz',
            'delta_alphas_lambdas', 'delta_shower_alphas', 'delta_alphas_nf',
            'delta_alphas_lambda', 'gl_safe', 'get_parton_id')
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
CA, CF, TR, MC, MB = 3., 4. / 3., .5, 1.5, 4.8
QUARKS = tuple(q for q in range(-5, 6) if q)

# Toy x*f(x, mu), rising with scale at small x. Flavour 4 starts at 2 GeV, so a
# grouped (2, 4) leg changes its gluon-feed multiplicity there, and a gluon has
# five active quarks only above 2 GeV.
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
      integer n,flav(3),beam,order,k
      double precision x,alpha,lambda(3:5)
      type(delta_connections) leg
      leg%sudakov_type=1
      do
         read(*,*,end=10) mode
         if (mode.eq.'c') then
            read(*,*) alpha,order
            call delta_alphas_lambdas(alpha,order,lambda)
         elseif (mode.eq.'a') then
            read(*,*) x
            write(*,'(es24.16)') delta_shower_alphas(x,lambda,order)
         else
c     Each colour line: stopping and starting scale, mass and partner.
            read(*,*) n,flav(1:n),beam,x,leg%count
            do k=1,leg%count
               read(*,*) leg%stop(k),leg%start(k),leg%mass(k),
     $              leg%partner(k)
            enddo
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


def reference_lines(flavs, x, lines, alpha_mz=.118, order=2):
    """F*R*exp(C) on fine grids for (stop, start, mass) colour lines, combined as
    eq.3.34 (delta_leg_probability): forward factors from every line's start to
    each stopping line's scale, weighted by the lines' emission exponents."""
    gluon = flavs[0] == 21
    a = lambda t: pythia_alphas(math.sqrt(t), alpha_mz, order) / (2. * math.pi)
    nfmax = sum(1 for q in range(1, 6) if toy_xf(q, x, max(l[1] for l in lines)) > 0.)

    def nf(t):
        return min(nfmax, 3 if t <= MC**2 else 4 if t <= MB**2 else 5)

    def zmax(t, mass):
        return 1. - .5 * t / mass**2 * (math.sqrt(1. + 4. * mass**2 / t) - 1.)

    def grid(lo, hi):
        return [lo] + [2. * math.log(c) for c in (MC, 2., MB) if lo < 2. * math.log(c) < hi] + [hi]

    def forward(lo, hi, mass):
        """Signed exponent of one colour line from lo up to hi."""
        def rate(u):
            t = math.exp(u)
            z = zmax(t, mass)
            if gluon:  # half of the full gluon kernel per line
                v = .5 * (2. * CA * (-math.log(1. - z) - z**2 + z**3 / 3. - z**4 / 4.)
                          + 2. * nf(t) * TR * (z**2 / 2. - 2. * z**3 / 3. + z**4 / 2.))
            else:
                v = CF * (-2. * math.log(1. - z) - 1.5 * z * z)
            return a(t) * v
        a0, b0 = sorted((2. * math.log(lo), 2. * math.log(hi)))
        value = integrate(rate, grid(a0, b0), 12)
        return value if hi >= lo else -value

    def boundary(stop, start, mass):
        def rate(u):
            t = math.exp(u)
            pt, z0 = math.sqrt(t), zmax(t, mass)
            zl = max(x, z0)
            if gluon:
                den = toy_xf(21, x, pt)
                val = (2. * CA * (-math.log(z0) - 2. * (1. - z0) + 1.5 * (1. - z0**2)
                                  - 2. / 3. * (1. - z0**3) + .25 * (1. - z0**4))
                       - 2. * nf(t) * TR * (1. / 3. - z0**2 / 2. + 2. * z0**3 / 3. - z0**4 / 2.))
                pgg = lambda z: 2. * CA * (1. - z + z * z)**2 / (z * (1. - z))
                if z0 < x:
                    val -= integrate(pgg, [z0, x], 16)

                def inner(z):
                    y = x / z
                    return (pgg(z) * (toy_xf(21, y, pt) / den - 1.)
                            + CF * (1. + (1. - z)**2) / z * sum(toy_xf(f, y, pt) for f in QUARKS) / den)
            else:
                den = sum(toy_xf(f, x, pt) for f in flavs)
                nact = sum(1 for f in flavs if toy_xf(f, x, pt) > 0.)
                val = CF * z0 * (1. - z0)
                if z0 < x:
                    val -= integrate(lambda z: CF * (1. + z * z) / (1. - z), [z0, x], 16)

                def inner(z):
                    y = x / z
                    fq = sum(toy_xf(f, y, pt) for f in flavs)
                    return (CF * (1. + z * z) / (1. - z) * (fq / den - 1.)
                            + nact * TR * (z * z + (1. - z)**2) * toy_xf(21, y, pt) / den)
            return a(t) * (val + integrate(inner, [zl, 1.], 16))
        return integrate(rate, grid(2. * math.log(stop), 2. * math.log(start)), 12)

    def gl(exponent):  # gl_safe(exp(-exponent), 1)
        return 0. if exponent <= 0. else 1e8 if exponent >= -math.log(1e-20) else 2. * exponent

    xfg = lambda mu: sum(toy_xf(f, x, mu) for f in flavs)
    weights = [1.] if len(lines) == 1 else [gl(forward(q, s, m)) for q, s, m in lines]
    total = sum(weights)
    return sum(w / total * xfg(q) / xfg(s) * math.exp(boundary(q, s, m) - sum(
        forward(q, s2, m2) for _, s2, m2 in lines)) for w, (q, s, m) in zip(weights, lines))


def reference_leg(flavs, x, q, start, mass, alpha_mz=.118, order=2):
    """One colour line from start down to q."""
    return reference_lines(flavs, x, [(q, start, mass)], alpha_mz, order)


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

    def lines(self, flavs, x, lines, coupling=(.118, 2), beam=1):
        """Leg probability for colour lines (stop, start, mass[, partner])."""
        rows = ''.join('%.17g %.17g %.17g %d\n' % (tuple(l[:3]) + (l[3] if len(l) > 3 else 2,))
                       for l in lines)
        return self.run_lines(['c\n%r %d\n' % coupling,
                               'b\n%d %s %d %.17g %d\n' % (len(flavs), ' '.join(map(str, flavs)), beam,
                                                         x, len(lines)) + rows])[0]

    def leg(self, flavs, x, q, start, mass, coupling=(.118, 2)):
        return self.lines(flavs, x, [(q, start, mass)], coupling)

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

    def test_gluon_lines(self):
        # One line takes half the gluon exponent; two equal lines the full one,
        # and unequal stopping scales are weighted as in the table's eq.3.34.
        for x, lines in ((.01, [(10., 60., 125.)]), (.03, [(5., 200., 400.)] * 2),
                         (.0015, [(2.5, 80., 150.)] * 2), (.25, [(4., 250., 1000.)] * 2),
                         (.6, [(3., 40., 7000.)] * 2), (.02, [(3., 60., 125.), (12., 60., 125.)]),
                         (.05, [(8., 90., 300.), (8., 40., 300.)])):
            with self.subTest(x=x, lines=lines):
                value = self.lines((21,), x, lines)
                self.assertAlmostEqual(value / min(reference_lines((21,), x, lines), 1.), 1.,
                                       delta=2e-4)
                self.assertGreater(value, 0.)
        one = self.lines((21,), .03, [(5., 200., 400.)])
        self.assertLess(self.lines((21,), .03, [(5., 200., 400.)] * 2), one)

    def test_if_connections_use_the_ii_dipole(self):
        # PYTHIA's ISR recoils against the other beam: the partner type is irrelevant.
        for flavs, lines in (((2,), [(7., 70., 160.)]), ((21,), [(7., 70., 160.)] * 2)):
            with self.subTest(flavs=flavs):
                ii = self.lines(flavs, .02, [l + (2,) for l in lines])
                self.assertEqual(self.lines(flavs, .02, [l + (3,) for l in lines]), ii)
                self.assertNotEqual(ii, -1.)

    def test_first_order_running(self):
        for flavs, x, q, start, mass in (((2,), .007, 20., 60., 91.19),
                                         ((2, 4), .007, 3., 50., 91.19),
                                         ((21,), .02, 5., 80., 200.)):
            with self.subTest(flavs=flavs, q=q):
                value = self.leg(flavs, x, q, start, mass, coupling=(.1365, 1))
                self.assertAlmostEqual(
                    value / reference_leg(flavs, x, q, start, mass, .1365, 1), 1., delta=2e-4)
                # The larger one-loop coupling suppresses emission-free evolution.
                self.assertLess(value, self.leg(flavs, x, q, start, mass))

    def test_no_evolution_and_table_fallbacks(self):
        self.assertEqual(self.leg((2,), .01, 30., 30., 91.19), 1.)
        self.assertEqual(self.lines((21,), .01, [(30., 30., 91.19)] * 2), 1.)
        # The lowest ln(t) node lies at 1.44 GeV, where this flavour has no PDF.
        self.assertEqual(self.leg((4,), .01, 1.2, 6., 91.19), -1.)
        # A vanishing starting PDF, lines below the table's 1.01 GeV continuation,
        # groups mixing quarks and gluons and non-hadron beams keep the table.
        self.assertEqual(self.leg((4,), .01, 1.5, 1.8, 91.19), -1.)
        self.assertEqual(self.leg((2,), .01, 1., 30., 91.19), -1.)
        self.assertEqual(self.lines((21,), .01, [(10., 30., 91.19), (1., 30., 91.19)]), -1.)
        self.assertEqual(self.leg((21, 2), .01, 10., 30., 91.19), -1.)
        for beam in (0, 2, -2):
            self.assertEqual(self.lines((2,), .01, [(10., 30., 91.19)], beam=beam), -1.)
        self.assertNotEqual(self.lines((2,), .01, [(10., 30., 91.19)], beam=-1), -1.)


if __name__ == '__main__':
    unittest.main()
