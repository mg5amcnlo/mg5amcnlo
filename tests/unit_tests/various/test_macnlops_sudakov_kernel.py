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



@unittest.skipUnless(shutil.which('g++'), 'requires g++')
class TestSudakovTransferSupport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='macnlops_sudakov_transfer_')
        cls.addClassCleanup(cls.tmp.cleanup)
        work = Path(cls.tmp.name)
        source = (ROOT/'Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc').read_text()
        function = source[source.index('double sudakovAcceptance('):
                          source.index('// macnlops.matching_scale:')]
        (work/'transfer.cc').write_text('''
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
struct Density {double r,d,k; struct {double source;} sudakov;};
''' + function + '''
int main() {
  Density x; double scale,ps; int hard;
  while (std::cin>>x.r>>x.d>>x.k>>x.sudakov.source>>scale>>ps>>hard) {
    try {std::cout<<std::setprecision(17)<<sudakovAcceptance(x,scale,ps,hard)<<"\\n";}
    catch(const std::runtime_error&) {std::cout<<"invalid\\n";}
  }
}
''')
        cls.exe=work/'transfer'
        subprocess.run(['g++','-std=c++11','-O2',str(work/'transfer.cc'),
                        '-o',str(cls.exe)],check=True)

    def test_zero_shower_support_retains_the_source_hard_remainder(self):
        # The million-event DY run exposed a negative H contribution with
        # positive source suppression but zero resolved shower-PDF density.
        points=[(1.249686376664813e12,1.258672154209456e12,
                 1.258672154209456e12,1.,0.,.028761,1),
                (2.,5.,4.,.7,1.,0.,1)]
        out=subprocess.run([str(self.exe)],input=''.join(
            ' '.join(map(str,p))+'\n' for p in points),
            text=True,capture_output=True,check=True)
        self.assertEqual(list(map(float,out.stdout.split())),[1.,1.])

    def test_finite_transfer_remains_bounded_by_both_densities(self):
        points=[(2.,5.,4.,a,s,p,h) for a in [.01,.7,1.]
                for s in [1e-14,.3,2.] for p in [.01,.8,1.] for h in [0,1]]
        out=subprocess.run([str(self.exe)],input=''.join(
            ' '.join(map(str,p))+'\n' for p in points),
            text=True,capture_output=True,check=True)
        for point,keep in zip(points,map(float,out.stdout.split())):
            r,d,k,a,scale,ps,hard=point
            F=scale*k*ps
            transfer=min(min(d-r,k)*a,F)
            denominator=(d-r)*a if hard else F
            self.assertAlmostEqual(keep,1.-transfer/denominator,delta=2e-14)

    def test_negative_shower_density_still_fails(self):
        out=subprocess.run([str(self.exe)],input='2 5 4 1 -1 .5 1\n',
                           text=True,capture_output=True,check=True)
        self.assertEqual(out.stdout.strip(),'invalid')


if __name__ == '__main__':
    unittest.main()
