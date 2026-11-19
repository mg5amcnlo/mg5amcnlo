"""Resolved soft-quark energies must remain positive in FSR counterterms."""
from decimal import Decimal, localcontext
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestFSRCollinearStability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="fks_collinear_energy_")
        cls.work = Path(cls.tmp.name)
        (cls.work / "nexternal.inc").write_text(
            "      integer nexternal,nincoming\n"
            "      parameter(nexternal=5,nincoming=2)\n")
        (cls.work / "nFKSconfigs.inc").write_text(
            "      integer fks_configs\n      parameter(fks_configs=1)\n")
        (cls.work / "fks_info.inc").write_text("")
        (cls.work / "coupl.inc").write_text(
            "      double precision g\n      double complex gal(1)\n"
            "      common/test_coupling/g,gal\n")
        (cls.work / "orders.inc").write_text(
            "      integer amp_split_size,nsplitorders,qcd_pos,qed_pos\n"
            "      parameter(amp_split_size=1,nsplitorders=2,qcd_pos=1,qed_pos=2)\n"
            "      double precision amp_split(amp_split_size)\n"
            "      double complex amp_split_cnt(amp_split_size,2,nsplitorders)\n"
            "      common/test_amplitudes/amp_split,amp_split_cnt\n")
        (cls.work / "geometry.f").write_text(
            "      module fks_phase_space_helpers\n      contains\n" +
            fortran_routine(TEMPLATE / "genps_fks_helpers.f", "dot") +
            "      end module\n")
        (cls.work / "collinear.f").write_text("\n".join(
            fortran_routine(TEMPLATE / "fks_singular.f", name) for name in
            ("sborncol_fsr", "AP_reduced", "Qterms_reduced_timelike")))
        (cls.work / "driver.f90").write_text("""
program check_collinear
  use fks_phase_space_data
  implicit none
  include 'nexternal.inc'
  include 'coupl.inc'
  include 'orders.inc'
  real(8) :: p(0:3,nexternal),wgt,ch_i,ch_j,ch_m,iden_comp,iden_comp_FKS(1)
  integer :: i_fks,j_fks,i_type,j_type,m_type,j_pdg,iextra_cnt,isplitorder_born,isplitorder_cnt
  integer :: count,mode,i
  logical :: split_type(nsplitorders),fixed_order,nlo_ps
  common/fks_indices/i_fks,j_fks
  common/cparticle_types/ch_i,ch_j,ch_m,i_type,j_type,m_type,j_pdg
  common/c_extra_cnt/iextra_cnt,isplitorder_born,isplitorder_cnt
  common/c_split_type/split_type
  common/c_iden_comp/iden_comp,iden_comp_FKS
  common/c_fnlo_nlops/fixed_order,nlo_ps
  g=1d0
  gal=(1d0,0d0)
  i_fks=5
  j_fks=4
  i_type=8
  j_type=3
  m_type=3
  j_pdg=1
  ch_i=0d0
  ch_j=0d0
  ch_m=0d0
  iextra_cnt=0
  split_type=(/.true.,.false./)
  iden_comp=1d0
  fixed_order=.false.
  nlo_ps=.true.
  p_born=0d0
  p_born(0,1)=100d0
  read(*,*) count
  do i=1,count
    read(*,*) mode,shat,resonance_mass2
    resonance_recoil=mode.eq.1
    read(*,*) resonance_momentum
    p=0d0
    read(*,*) p(:,i_fks)
    read(*,*) p(:,j_fks)
    call sborncol_fsr(p,.1d0,1d0,wgt)
    write(*,'(2ES26.17E3)') wgt,amp_split(1)
  enddo
end program

subroutine sborn_native(p,wgt)
  implicit none
  include 'orders.inc'
  real(8) :: p(0:3,4),wgt
  complex(8) :: ans_cnt(2,nsplitorders)
  common/c_born_cnt/ans_cnt
  wgt=1d0
  ans_cnt=(0d0,0d0)
  ans_cnt(1,1)=(1d0,0d0)
  amp_split_cnt=(0d0,0d0)
  amp_split_cnt(1,1,1)=(1d0,0d0)
end subroutine

subroutine extra_cnt_native()
  stop 99
end subroutine
subroutine getaziangles()
  stop 99
end subroutine
subroutine IXXXSO()
  stop 99
end subroutine
subroutine OXXXSO()
  stop 99
end subroutine
""")
        cls.executable = cls.work / "check"
        build = subprocess.run([
            shutil.which("gfortran"), "-O2", "-fcheck=all",
            "-ffixed-line-length-none", "-ffpe-trap=invalid,zero,overflow",
            "-I", str(cls.work), str(TEMPLATE / "fks_phase_space_data.f"),
            "geometry.f", "collinear.f", "driver.f90", "-o", str(cls.executable)],
            cwd=cls.work, capture_output=True, text=True)
        if build.returncode:
            raise RuntimeError(build.stdout + build.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_cases(self, cases):
        lines = [str(len(cases))]
        for mode, shat, mass2, q, emitted, quark in cases:
            lines.append("{} {} {}".format(mode, shat, mass2))
            lines.extend(" ".join(format(x, ".17g") for x in v)
                         for v in (q, emitted, quark))
        result = subprocess.run([str(self.executable)], input="\n".join(lines)+"\n",
                                capture_output=True, text=True, check=True)
        rows = [[float(x) for x in row.split()] for row in result.stdout.splitlines()]
        self.assertEqual(len(rows), len(cases))
        self.assertTrue(all(math.isfinite(x) for row in rows for x in row))
        return rows

    def test_captured_boosted_dipole_keeps_small_positive_products(self):
        q = [9086.530343011271, -.005072698972037142, -.005642633573320355,
             -9086.530285905545]
        emitted = [.5041068216226863, -.005072692245484154,
                   -.005642626091016767, -.5040497159722336]
        quark = [6.684615308500982e-7, -6.726547878164632e-9,
                 -7.482297904745574e-9, -6.683858069802579e-7]
        mass2 = 1.0377282701938384
        cases = [(1, 30000., mass2, q, emitted, [x*scale for x in quark])
                 for scale in (1., 1.e-6, 1.e-12)]
        for case, row in zip(cases, self.run_cases(cases)):
            with localcontext() as ctx:
                ctx.prec = 70
                vectors = [[Decimal.from_float(x) for x in v]
                           for v in (q, emitted, case[-1])]
                qd, kd, jd = vectors
                qi = qd[0]*kd[0]-sum(x*y for x,y in zip(qd[1:],kd[1:]))
                qj = qd[0]*jd[0]-sum(x*y for x,y in zip(qd[1:],jd[1:]))
                self.assertGreater(qj, 0.)
                self.assertLess(qj, Decimal('1e-6'))
                z = qj/(qi+qj)
                reference = float(Decimal(4)/3*(1+z*z)/(z*Decimal.from_float(mass2)/4))
            for value in row:
                self.assertAlmostEqual(value/reference, 1., delta=3.e-11)

    def test_soft_quark_fraction_is_not_subtracted_from_one(self):
        q = [1., 0., 0., 0.]
        emitted = [1., 0., 0., 1.]
        cases = [(0, 10000., 1., q, emitted, [energy, 0., 0., energy])
                 for energy in (1.e-18, 1.e-30, 1.e-100)]
        for case, row in zip(cases, self.run_cases(cases)):
            energy = case[-1][0]
            self.assertEqual(1.-1./(1.+energy), 0.)
            z = energy/(1.+energy)
            reference = (4./3.)*(1.+z*z)/(z*10000./4.)
            for value in row:
                self.assertAlmostEqual(value/reference, 1., delta=5.e-15)

    def test_ordinary_kernel_is_unchanged(self):
        cases = [(resonance, 10000., 10000., [100.,0.,0.,0.],
                  [30.,0.,0.,30.], [70.,0.,0.,70.]) for resonance in (0,1)]
        reference = (4./3.)*(1.+.7**2)/(.7*10000./4.)
        for row in self.run_cases(cases):
            for value in row:
                self.assertAlmostEqual(value/reference, 1., delta=5.e-15)


if __name__ == "__main__":
    unittest.main()
