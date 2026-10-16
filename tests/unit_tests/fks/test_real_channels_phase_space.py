"""Compile and integrate the real-tree maps without generating a process."""

import math
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]


def fortran_routine(source, name):
    start = re.search(r"^      (?:subroutine|(?:double precision )?function) "
                      + name + r"\(", source, re.I | re.M)
    if start is None:
        raise AssertionError("Missing Fortran routine " + name)
    end = re.search(r"^      end\s*$", source[start.start():], re.I | re.M)
    return source[start.start():start.start() + end.end()] + "\n"


@unittest.skipUnless(shutil.which("gfortran"), "gfortran is required")
class TestRealChannelsPhaseSpace(unittest.TestCase):
    """Known phase-space volumes detect flux and (2*pi) normalization errors."""

    def integrate(self, nexternal, nincoming, tree, masses=None, pdf=False,
                  propagators=None):
        source = (ROOT / "Template/NLO/SubProcesses/genps_fks.f").read_text()
        routines = ["generate_momenta_real_channels", "generate_inv_mass_sch",
                    "generate_si", "generate_t_channel_branchings",
                    "fill_born_momenta", "generate_tau_BW", "generate_tau",
                    "generate_y", "trans_x", "lambda", "yminmax", "gentcms"]
        selected = "\n".join(fortran_routine(source, name) for name in routines)
        kin = (ROOT / "Template/NLO/Source/kin_functions.f").read_text()
        selected += fortran_routine(kin, "dot")
        singular = (ROOT / "Template/NLO/SubProcesses/fks_singular.f").read_text()
        selected += fortran_routine(singular, "phspncheck_nocms")
        selected += fortran_routine(singular, "xlen4")
        tree_lines = "\n".join("iforest(:,%d,1,1) = [%d,%d]" %
                               (-i, left, right)
                               for i, (left, right) in enumerate(tree, 1))
        mass_values = masses or [0.] * nexternal
        masses_line = ",".join(("%.16e" % m).replace("e", "d")
                               for m in mass_values)
        prop_lines = "\n".join(
            "props_mass(%d,1,1)=%sd0\nprops_width(%d,1,1)=%sd0" %
            (index, mass, index, width)
            for index, (mass, width) in (propagators or {}).items())
        # The beam-integrated massless three-body volume divided by flux is
        # constant, so its exact dx1 dx2 integral equals the fixed-energy one.
        program = """program check_real_map
use mint_module
implicit none
include 'genps.inc'
include 'nexternal.inc'
include 'nFKSconfigs.inc'
include 'run.inc'
double precision props_mass(-nexternal:0,lmaxconfigs,0:fks_configs)
double precision props_width(-nexternal:0,lmaxconfigs,0:fks_configs)
integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
integer mapconfig(0:lmaxconfigs,0:fks_configs)
common/c_configurations/props_mass,props_width,iforest,sprop,tprid,mapconfig
double precision emass(nexternal)
common/to_mass/emass
double precision x(99),p(0:3,nexternal),wgt,total,total2,delta(0:3),dot
double precision shat,mean,error
integer i,j,ndim,nseed
integer, allocatable :: seed(:)
external dot
call random_seed(size=nseed)
allocate(seed(nseed))
seed=12345
call random_seed(put=seed)
props_mass=0d0
props_width=0d0
PROPS
iforest=0
sprop=0
tprid=0
mapconfig=0
mapconfig(0,1)=1
mapconfig(1,1)=1
TREE
emass=[MASSES]
ebeam=50d0
lpp=PDF
ndim=3*(nexternal-nincoming)-4
if(PDF.ne.0) ndim=ndim+2
total=0d0
total2=0d0
do i=1,12000
 call random_number(x)
 call generate_momenta_real_channels(ndim,1,1,x,p,wgt)
 if(wgt.le.0d0.or.wgt.ne.wgt) stop 2
 delta=sum(p(:,1:nincoming),dim=2)-sum(p(:,nincoming+1:nexternal),dim=2)
 if(maxval(abs(delta)).gt.1d-7) stop 3
 shat=dot(sum(p(:,1:nincoming),dim=2),sum(p(:,1:nincoming),dim=2))
 do j=1,nexternal
  if(abs(dot(p(:,j),p(:,j))-emass(j)**2).gt.1d-7*shat) stop 4
 enddo
 total=total+wgt
 total2=total2+wgt*wgt
enddo
mean=total/12000d0
error=sqrt(max(0d0,total2/12000d0-mean**2)/12000d0)
write(*,*) mean,error
end program
""".replace("TREE", tree_lines).replace("MASSES", masses_line)
        program = program.replace("PROPS", prop_lines)
        program = program.replace("PDF", "1" if pdf else "0")
        support = """      subroutine schan_order(n,order)
      implicit none
      include 'nexternal.inc'
      integer n,order(-nexternal:0),i
      do i=1,n
         order(-i)=-i
      enddo
      end
      subroutine generate_ee_tau_y
      stop 99
      end
"""
        with tempfile.TemporaryDirectory(prefix="mg5-real-map-") as tmp:
            tmp = pathlib.Path(tmp)
            (tmp / "genps.inc").write_text(
                "      integer max_branch,max_particles,lmaxconfigs\n"
                "      parameter(max_branch=10,max_particles=10,lmaxconfigs=1)\n")
            (tmp / "nexternal.inc").write_text(
                "      integer nexternal,nincoming\n"
                "      parameter(nexternal=%d,nincoming=%d)\n" %
                (nexternal, nincoming))
            (tmp / "nFKSconfigs.inc").write_text(
                "      integer fks_configs\n      parameter(fks_configs=1)\n")
            (tmp / "run.inc").write_text(
                "      integer lpp(2)\n      double precision ebeam(2)\n"
                "      common/to_test_beams/ebeam,lpp\n")
            (tmp / "mint.f90").write_text(
                "module mint_module\nlogical :: new_point=.true.\nend module\n")
            (tmp / "maps.f").write_text(selected + support)
            (tmp / "check.f90").write_text(program)
            command = ["gfortran", "-O0", "-g", "-fcheck=all",
                       "-ffixed-line-length-none", "-ffree-line-length-none",
                       "-fallow-argument-mismatch", "-o", "check", "mint.f90",
                       "maps.f", "check.f90"]
            command += [str(ROOT / "HELAS" / name)
                        for name in ("mom2cx.F", "boostx.F", "rotxxx.F")]
            compiled = subprocess.run(command, cwd=tmp, capture_output=True,
                                      text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(tmp / "check")], cwd=tmp,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return tuple(float(x) for x in result.stdout.split()[-2:])

    def assert_volume(self, result, expected):
        mean, error = result
        self.assertLess(abs(mean - expected), max(5 * error, expected * 1.e-8))

    def test_two_body_s_and_t_channels(self):
        for tree in ([(3, 4)], [(1, 3), (-1, 4)]):
            self.assert_volume(self.integrate(4, 2, tree),
                               1. / (16 * math.pi * 10000))
            self.assert_volume(self.integrate(4, 2, tree, [0., 0., 10., 10.]),
                               math.sqrt(.96) / (16 * math.pi * 10000))

    def test_three_body_s_t_and_mixed_channels(self):
        for tree in ([(3, 4), (-1, 5)],
                     [(1, 3), (-1, 4), (-2, 5)],
                     [(3, 4), (1, 5), (-2, -1)]):
            self.assert_volume(self.integrate(5, 2, tree),
                               1. / (512 * math.pi**3))

    def test_massless_beam_mapping_has_full_support(self):
        self.assert_volume(self.integrate(5, 2, [(3, 4), (-1, 5)], pdf=True),
                           1. / (512 * math.pi**3))

    def test_four_body_nested_s_channels(self):
        self.assert_volume(self.integrate(6, 2,
                                          [(3, 4), (5, 6), (-1, -2)]),
                           10000. / (49152 * math.pi**5))

    def test_real_resonance_mapping(self):
        self.assert_volume(self.integrate(5, 2, [(3, 4), (-1, 5)],
                                          propagators={-1: (40., 5.)}),
                           1. / (512 * math.pi**3))

    def test_massive_three_body_threshold(self):
        # Independent factorization dPhi3=dq2/(2*pi)*dPhi2*dPhi2.
        shat, m1, m2, m3 = 10000., 10., 15., 20.
        low, high = (m1 + m2)**2, (math.sqrt(shat) - m3)**2
        nsteps = 20000
        step = (high - low) / nsteps

        def kallen(a, b, c):
            return a*a + b*b + c*c - 2*(a*b + a*c + b*c)

        integral = sum(math.sqrt(kallen(shat, low + (i + .5)*step, m3*m3)
                                  * kallen(low + (i + .5)*step, m1*m1, m2*m2))
                       / (low + (i + .5)*step) for i in range(nsteps)) * step
        expected = integral / (256 * math.pi**3 * shat**2)
        for tree in ([(3, 4), (-1, 5)], [(1, 3), (-1, 4), (-2, 5)]):
            self.assert_volume(self.integrate(5, 2, tree,
                                              [0., 0., m1, m2, m3]), expected)

    def test_decay_normalization(self):
        self.assert_volume(self.integrate(4, 1, [(2, 3), (-1, 4)],
                                          [100., 0., 0., 0.]),
                           100. / (512 * math.pi**3))
