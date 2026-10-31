"""Check the matching denominator against the shower's kinematic inequalities.

The reference integrates Pqq numerically in z and tests the untransformed
virtual-mass and mass-reduced-cutoff conditions independently of the
production analytic roots and antiderivative.
"""
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('g++'), 'requires g++')
class TestMassiveShowerRate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='macnlops_sudakov_kernel_')
        cls.addClassCleanup(cls.tmp.cleanup)
        work = Path(cls.tmp.name)
        source = (ROOT/'Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc').read_text()
        rate = source[source.index('double quarkFsrRate('):source.index('double backwardIntegral(')]
        (work/'rate.cc').write_text('''
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
struct Coupling {double alphaS(double) const {return 1.;}};
struct Hook {double cutoffFSR=.5; Coupling fsrAlphaS;};
struct Context {Hook* hook;};
struct SudakovProfile {int nquarks=1; std::array<double,256> quarks;};
''' + rate + '''
int main() {
  Hook hook; Context c{&hook}; SudakovProfile p; double q;
  while (std::cin >> p.quarks[0] >> p.quarks[1] >> p.quarks[2]
                  >> p.quarks[3] >> q)
    std::cout << std::setprecision(17) << quarkFsrRate(c,p,2.*std::log(q)) << '\\n';
}
''')
        cls.exe = work/'rate'
        subprocess.run(['g++','-std=c++11','-O2',str(work/'rate.cc'),'-o',str(cls.exe)],check=True)

    @staticmethod
    def reference(mr, mk, M, ceiling, q):
        if q >= ceiling:
            return 0.
        t = q*q
        corrected = (M-mk)**2-mr*mr
        if corrected <= 0:
            return 0.

        def allowed(v):
            z = 1/(1+math.exp(-v))
            s = z*(1-z)
            virtual = mr*mr+t/s
            return (s > t/corrected and
                    virtual*M*M < s*(M*M+virtual-mk*mk)**2 and
                    q*(1-mr*mr/virtual) > .5)

        # On z<=1/2 the accepted interval has at most two boundaries.
        # Locate them using the original inequalities, then bisect them.
        edges = [-40.]
        previous = edges[0]
        for k in range(1,2001):
            current = -40.+k*.02
            if allowed(current) != allowed(previous):
                a,b = previous,current
                for _ in range(55):
                    mid = (a+b)/2
                    if allowed(mid) == allowed(a):
                        a = mid
                    else:
                        b = mid
                edges.append((a+b)/2)
            previous = current
        edges.append(0.)

        def f(v):
            z = 1/(1+math.exp(-v))
            # Pqq(z)+Pqq(1-z), times dz/dlogit(z).
            return (4/3)*((1+z*z)*z+(1+(1-z)**2)*(1-z))/(2*math.pi)

        total = 0.
        for a,b in zip(edges[:-1],edges[1:]):
            if not allowed((a+b)/2):
                continue
            n = 2000
            h = (b-a)/n
            total += h/3*(f(a)+f(b)+sum(
                (4 if k%2 else 2)*f(a+k*h) for k in range(1,n)))
        return total

    def test_massive_support_and_reduced_transverse_momentum_cutoff(self):
        points = [(mr,mk,M,ceiling,q)
                  for mr,mk,M,ceiling in [(173.,173.,500.,150.),
                      (173.,173.,900.,200.),(0.,0.,125.,50.),
                      (4.8,0.,125.,50.),(173.,173.,347.,10.)]
                  for q in [.51,1.,3.,7.,10.,30.,80.,201.]]
        result = subprocess.run([str(self.exe)],input=''.join(
            ' '.join(map(str,p))+'\n' for p in points),
            text=True,capture_output=True,check=True)
        for p,value in zip(points,map(float,result.stdout.splitlines())):
            with self.subTest(point=p):
                self.assertAlmostEqual(value,self.reference(*p),delta=2e-10)


if __name__ == '__main__':
    unittest.main()
