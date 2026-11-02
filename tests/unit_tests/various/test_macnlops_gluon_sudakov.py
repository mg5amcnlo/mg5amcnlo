"""Global-gluon rate checks independent of the production quadrature.

The massless-recoiler reference is a closed primitive. Massive-recoiler
values come from independent integration of the stock 8.318 inequalities,
colour-end kernels and recoil weight, including native coupling thresholds.
"""
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
LAMBDAS = (.3249942945718149, .28756663468675603, .22513027856717657)


@unittest.skipUnless(shutil.which('g++'), 'requires g++')
class TestGlobalGluonRate(unittest.TestCase):
    source_path = ROOT/'Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc'

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='macnlops_gluon_rate_')
        cls.addClassCleanup(cls.tmp.cleanup)
        work = Path(cls.tmp.name)
        source = cls.source_path.read_text()
        rule = source[source.index('struct GaussRule {'):
                      source.index('// Exact massless-II backward')]
        rate = source[source.index('double gluonQuadrature('):
                      source.index('double quarkFsrRate(')]
        marginal = source[source.index('double showerMarginal('):
                          source.index('double sudakovAcceptance(')]
        (work/'rate.cc').write_text('''
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
struct Coupling {
  double Lambda3()const{return .3249942945718149;}
  double Lambda4()const{return .28756663468675603;}
  double Lambda5()const{return .22513027856717657;}
};
struct Hook {double cutoffFSR=.5; Coupling fsrAlphaS;};
struct Engine {
  struct Data {double charm=0.,bottom=0.;
    double m0(int id)const{return id==4?charm:id==5?bottom:0.;}
  } particleData;
  int mode(const char*)const{return 1;}
};
struct Context {Hook* hook;Engine* engine;int flavours=5;};
struct SudakovProfile {
  int ngluons=1,count=1;double pt=.5,cdf=1.;
  std::array<double,320> gluons{};
  std::array<double,512> q{},w{},fsr{};
};
double backwardIntegral(const Context&,const SudakovProfile&,double,double){return 0.;}
''' + rule + rate + marginal + '''
int main(int argc,char**) {
  Hook hook;Engine engine;Context c;c.hook=&hook;c.engine=&engine;
  SudakovProfile p;double q;
  while(std::cin>>p.gluons[1]>>p.gluons[2]>>p.gluons[3]>>p.gluons[4]
        >>q>>c.flavours>>engine.particleData.charm>>engine.particleData.bottom
        >>p.ngluons) {
    double result;
    if(argc>1){p.pt=q;std::cin>>p.q[0];p.w[0]=p.fsr[0]=1.;result=showerMarginal(c,p);}
    else result=gluonFsrRate(c,p,2.*std::log(q));
    std::cout<<std::setprecision(17)<<result<<'\\n';
  }
}
''')
        cls.exe = work/'rate'
        subprocess.run(['g++', '-std=c++11', '-O2', str(work/'rate.cc'),
                        '-o', str(cls.exe)], check=True)

    def rates(self, points, marginal=False):
        result = subprocess.run([str(self.exe)]+(['marginal'] if marginal else []), input=''.join(
            ' '.join(map(str, p))+'\n' for p in points), text=True,
            capture_output=True, check=True)
        return list(map(float, result.stdout.split()))

    @staticmethod
    def massless_reference(M, caps, q, flavours):
        ends = sum(q < cap for cap in caps)
        if q < .5 or q >= M/2 or not ends:
            return 0.
        nf = 5 if q > 4 else 4 if q > 1.2 else 3
        alpha_over_2pi = 1/((11-2*nf/3)/2*math.log(q*q/LAMBDAS[nf-3]**2))
        z = q/M

        def primitive(x):
            return 1.5*(-2*math.log1p(-x)-x-x*x/2-x**3/3) + \
                flavours/4*(x-x*x+2*x**3/3)
        return ends*alpha_over_2pi*(primitive(1-z)-primitive(z))

    def test_closed_massless_limit_and_each_colour_end(self):
        points = [(0., M, a, b, q, nf, 0., 0., 1)
                  for M in (20., 125., 13000.)
                  for a, b in ((10., 30.), (30., 10.), (30., 30.))
                  for q in (.49, .51, 1.19, 1.21, 3.99, 4.01, 4.8, 9.9, 10.1, 29., 31.)
                  for nf in (0, 4, 5)]
        for point, actual in zip(points, self.rates(points)):
            mr, M, a, b, q, nf, _, _, _ = point
            with self.subTest(point=point):
                self.assertAlmostEqual(actual, self.massless_reference(M, (a, b), q, nf),
                                       delta=3e-9)

    def test_massive_global_recoiler_and_native_running_thresholds(self):
        # Independent high-precision integration, alpha_s(MZ)=.1365 at one loop.
        points = [(.51, 5.952467640413093),
                  (1.199999988, 1.614283200657210),
                  (1.200000012, 1.594111738810938),
                  (3.99999996, .529204789365399),
                  (4.00000004, .526288909760065),
                  (4.8, .446221722952770),
                  (9.9999999, .195769813689650),
                  (10.0000001, .0978849039986084),
                  (20., .011930385608426),
                  (22.500000225, 0.)]
        actual = self.rates([(80., 125., 10., 30., q, 5, 0., 0., 1)
                             for q, _ in points])
        for (q, expected), value in zip(points, actual):
            with self.subTest(q=q):
                self.assertAlmostEqual(value, expected, delta=3e-9)

    def test_captured_wjet_roundoff_failure_and_neighbouring_scales(self):
        # The old x1/x2 subtraction exhausted depth18 on the first point.
        # Independent 90-digit values integrate x=|2z-1|, after analytically
        # adding the two colour-end orientations (not the logit Gauss rule).
        # See docs/audits/check_gluon_quadrature_reference_20261101.py.
        scales = [(16.307745317636243, .4074932789486993229054889338601),
                  (16.307729009890924, .4074934653761065083427673521248),
                  (16.30776162538156, .4074930925215657358759251455104)]
        points = [(80.419002445775163, 2679.8203470176318,
                   28.056336658409535, 0., q, 5, 0., 0., 1)
                  for q, _ in scales]
        for (q, expected), actual in zip(scales, self.rates(points)):
            with self.subTest(q=q):
                self.assertAlmostEqual(actual, expected, delta=3e-11)

    def test_recoil_boundary_layers_ceiling_and_native_floors(self):
        rec, M = 80.419002445775163, 2679.8203470176318
        # Both small-r and close-ceiling references use the physical native
        # support. The 1.3/13000 point activates a denominator floor: removing
        # that floor shifts its reference by 1.2131e-9, above this tolerance.
        values = [
            ((rec, 5000., 6000., 0., .51, 5, 0., 0., 1),
             6.3251069420624623074974877326581),
            ((rec, 13000., 14000., 0., .51, 5, 0., 0., 1),
             7.0327478693096797276285396969606),
            ((rec, 13000., 14000., 0., 16.3, 5, 0., 0., 1),
             .5521655197412242842969635338552),
            ((rec, M, M, 0., 1299.699372585256, 5, 0., 0., 1),
             .0000022726672614090989828666973954332),
            ((rec, M, M, 0., 1299.7006721559583, 5, 0., 0., 1),
             .000000022721039879033922645365694825719),
            ((1.3, 13000., 14000., 0., .51, 5, 0., 0., 1),
             7.0329200767076626474912285655079),
            ((80., 125., 30., 0., .6, 5, 2., 6., 1),
             2.0404252426893745116491050314789)]
        for (point, expected), actual in zip(values, self.rates([p for p, _ in values])):
            with self.subTest(point=point):
                self.assertAlmostEqual(actual, expected, delta=3e-11)

    def test_pair_mass_threshold_does_not_become_an_evolution_pt_cut(self):
        def point(M, nf):
            return (M-2., M, 10., 30., .6, nf, 2., 6., 1)
        # At M=12 the virtual gluon cannot make a charm pair. At M=125 it
        # can, even though the evolution pT is below the charm mass.
        no_charm, blocked = self.rates([point(12., 3), point(12., 4)])
        self.assertAlmostEqual(no_charm, blocked, delta=1e-12)
        light, with_charm = self.rates([(80., 125., 10., 30., .6, nf, 2., 6., 1)
                                      for nf in (3, 4)])
        self.assertGreater(with_charm, light)

    def test_absent_gluons_and_dead_global_phase_space(self):
        points = [(80., 125., 10., 30., 5., 5, 0., 0., 0),
                  (125., 125., 10., 30., 5., 5, 0., 0., 1),
                  (80., 125., 0., 0., 5., 5, 0., 0., 1)]
        self.assertEqual(self.rates(points), [0., 0., 0.])

    def test_native_starting_slab_gate_depends_on_q_and_each_end_cap(self):
        M = 91.188
        rec = math.sqrt(M*M-2*M*7.)

        def point(Q, caps=(12., 12.)):
            return (rec, M, *caps, .75, 5, 0., 0., 1, Q)

        # This is a physical soft-gluon Durham-three-jet Born state. The
        # global ceiling is below m_b. Stock QCD evolution aborts an end
        # starting above m_b before it can descend into lower slabs.
        low, high, at_b, above_b, one_cap, two_caps = self.rates([
            point(3.), point(8.), point(4.), point(4.000001),
            point(8., (3., 12.)), point(8., (3., 3.))], marginal=True)
        self.assertLess(low, .95)
        self.assertEqual(high, 1.)
        self.assertLess(at_b, 1.)
        self.assertEqual(above_b, 1.)
        self.assertAlmostEqual(one_cap, math.sqrt(low), delta=2e-8)
        self.assertAlmostEqual(two_caps, low, delta=2e-8)

        # The analogous charm-slab boundary must use the same native rule.
        below_c, above_c = self.rates([
            (10., 12., 10., 30., .51, 5, 0., 0., 1, Q)
            for Q in (1.2, 1.200001)], marginal=True)
        self.assertLess(below_c, 1.)
        self.assertEqual(above_c, 1.)

    def test_gluon_marginal_matches_independent_massless_integral(self):
        # Integrate the closed-form z primitive independently in log(pT),
        # splitting only at native coupling and colour-end thresholds.
        M, caps, stop, Q = 125., (10., 30.), .75, 20.
        exponent = 0.
        for a, b in zip((stop, 1.2, 4., 10.), (1.2, 4., 10., Q)):
            lo, hi = math.log(a), math.log(b)
            n = 2000
            h = (hi-lo)/n
            def f(x):
                # Endpoint values are taken from inside this smooth panel.
                q = math.exp(min(max(x, lo+1e-12), hi-1e-12))
                return 2*self.massless_reference(M, caps, q, 5)
            exponent += h/3*(f(lo)+f(hi)+sum(
                (4 if j%2 else 2)*f(lo+j*h) for j in range(1, n)))
        actual = self.rates([(0., M, *caps, stop, 5, 0., 0., 1, Q)], marginal=True)[0]
        self.assertAlmostEqual(actual, math.exp(-exponent), delta=2e-8)


if __name__ == '__main__':
    unittest.main()
