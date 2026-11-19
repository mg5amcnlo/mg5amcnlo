"""Small-invariant and unit-rescaling checks for local subtraction scales."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import fortran_routine


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "Template/NLO/SubProcesses"


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestResonanceSubtractionScales(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix="mg5_small_local_scales_")
        cls.addClassCleanup(cls.scratch.cleanup)
        work = Path(cls.scratch.name)
        # Compile the actual helper product inside its module, the actual
        # subtraction routine, and the actual clipped legacy external product.
        source = ("      module fks_phase_space_helpers\n      contains\n"
                  + fortran_routine(TEMPLATE / "genps_fks_helpers.f", "dot")
                  + "      end module fks_phase_space_helpers\n"
                  + fortran_routine(TEMPLATE / "resonance_recoil.f",
                                    "resonance_subtraction_scales")
                  + fortran_routine(TEMPLATE.parent / "Source/kin_functions.f", "dot"))
        (work / "scales.f").write_text(source)
        cls.executable = work / "check_scales"
        result = subprocess.run(
            [shutil.which("gfortran"), "-O2", "-std=legacy",
             "-ffixed-line-length-none", "-fcheck=all", str(work / "scales.f"),
             str(ROOT / "tests/input_files/check_resonance_subtraction_scales.f90"),
             "-o", str(cls.executable)], cwd=work, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(result.stdout)

    def check_mode(self, mode):
        result = subprocess.run([str(self.executable), mode], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("PASS " + mode, result.stdout)

    def test_exact_C_LOC_generation_failure_point(self):
        self.check_mode("point")

    def test_scale_ratios_are_independent_of_momentum_units(self):
        self.check_mode("units")

    def test_invalid_recoil_is_still_rejected(self):
        self.check_mode("invalid")
