"""Numerical regressions for the MC/ME limit test and convergence checker.

Compile the actual test routines with controlled amplitudes and phase-space
weights. No generated process, PDFs or loop libraries are needed.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import fortran_routine


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "Template/NLO/SubProcesses"


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestSoftColLimits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix="mg5_soft_col_limits_")
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.work = Path(cls.tempdir.name)
        (cls.work / "nexternal.inc").write_text(
            "      integer nexternal,nincoming\n"
            "      parameter (nexternal=5,nincoming=2)\n")
        (cls.work / "orders.inc").write_text(
            "      integer amp_split_size\n"
            "      parameter (amp_split_size=3)\n"
            "      double precision amp_split(amp_split_size)\n"
            "      common /test_amp_split/amp_split\n")
        (cls.work / "nFKSconfigs.inc").write_text(
            "      integer fks_integrated\n      parameter(fks_integrated=6)\n")
        (cls.work / "mint_module.f90").write_text(
            "module mint_module\n"
            "  integer :: ndim=5, iconfig=1\n"
            "end module mint_module\n")
        routines = [fortran_routine(TEMPLATE / "fks_singular.f", name)
                    for name in ("checkres", "xprintout")]
        routines += [fortran_routine(TEMPLATE / "test_soft_col_limits.f", name)
                     for name in ("compute_towards_limit", "compute_in_the_limit",
                                  "check_limits", "read_input_file")]
        (cls.work / "limits.f").write_text("\n".join(routines))
        cls.executable = cls.work / "check_limits"
        command = [shutil.which("gfortran"), "-O2", "-std=legacy",
                   "-ffixed-line-length-none", "-fcheck=bounds", "-I", str(cls.work),
                   str(TEMPLATE / "fks_phase_space_data.f"),
                   str(cls.work / "mint_module.f90"),
                   str(ROOT / "tests/input_files/check_soft_col_limits.f90"),
                   str(cls.work / "limits.f"),
                   "-o", str(cls.executable)]
        result = subprocess.run(command, cwd=cls.work, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(result.stdout)

    def check_limits(self, case):
        result = subprocess.run([str(self.executable), case], cwd=self.work,
                                text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("PASS " + case, result.stdout)

    def test_roundoff_plateau(self):
        self.check_limits("roundoff")

    def test_convergence_and_finite_offsets(self):
        self.check_limits("convergence")

    def test_missing_mc_counterterms_fail(self):
        self.check_limits("missing_mc")

    def test_native_sector_and_order_normalization(self):
        self.check_limits("native_sector")

    def test_fixed_order_normalization(self):
        self.check_limits("fixed_order")

    def diagnostic_input(self, xi, y):
        return subprocess.run([str(self.executable), "read_input"], cwd=self.work,
                              input=f"1\n1 -0.1\n-1 -0.1\n{xi} {y}\n20 20\n0\n",
                              text=True, capture_output=True)

    def test_fixed_soft_collinear_overlap_input(self):
        result = self.diagnostic_input("0.0001", "0.9999")
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn("PASS read_input", result.stdout)

    def test_invalid_fixed_input(self):
        for xi, y, message in (("0", "0.5", "Fixed xi"),
                               ("0.5", "1", "Fixed y"),
                               ("NaN", "0.5", "Fixed xi")):
            with self.subTest(xi=xi, y=y):
                result = self.diagnostic_input(xi, y)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout)
