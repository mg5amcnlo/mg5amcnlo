"""Numerical regressions for native MC support and G replacements.

Compile the production support, damping, kinematics and history routines.
Controlled colour connections isolate the G policy from matrix elements;
the final-state dipole boundary uses the actual support check throughout.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import fortran_routine


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "Template/NLO/SubProcesses"
COUNTER = TEMPLATE / "montecarlocounter.f"


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestMCDeadZones(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix="mg5_mc_dead_zones_")
        cls.addClassCleanup(cls.tempdir.cleanup)
        work = Path(cls.tempdir.name)
        (work / "nexternal.inc").write_text(
            "      integer nexternal,nincoming\n"
            "      parameter (nexternal=6,nincoming=2)\n")
        (work / "orders.inc").write_text(
            "      integer amp_split_size\n"
            "      parameter (amp_split_size=1)\n")
        (work / "fks_info.inc").write_text("")
        shutil.copyfile(TEMPLATE / "fks_powers.inc", work / "fks_powers.inc")
        routines = [fortran_routine(COUNTER, name) for name in
                    ("compute_MCsubtraction_kl", "compute_damping_weight",
                     "emscafun", "get_dead_zone", "get_angle")]
        routines.append(fortran_routine(
            ROOT / "Template/NLO/Source/kin_functions.f", "dot"))
        (work / "counter.f").write_text("\n".join(routines))
        cls.executable = work / "check_dead_zones"
        command = [shutil.which("gfortran"), "-O2", "-std=legacy",
                   "-ffixed-line-length-none", "-fcheck=bounds",
                   "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
                   "-I", str(work), str(TEMPLATE / "process_module.f90"),
                   str(TEMPLATE / "kinematics_module.f90"),
                   str(TEMPLATE / "scale_module.f90"), str(work / "counter.f"),
                   str(ROOT / "tests/input_files/check_mc_dead_zones.f90"),
                   "-o", str(cls.executable)]
        result = subprocess.run(command, cwd=work, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(result.stdout)

    def check_case(self, name):
        result = subprocess.run([str(self.executable), name], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("PASS " + name, result.stdout)

    def test_massless_partner_roundoff(self):
        self.check_case("massless")

    def test_massive_partner_boundary(self):
        self.check_case("massive")

    def test_soft_wide_angle_transition(self):
        self.check_case("soft_transition")

    def test_shower_scale_endpoint(self):
        self.check_case("scale_endpoint")

    def test_two_colour_connections(self):
        self.check_case("two_connections")

    def test_singular_limits_and_disabled_g(self):
        self.check_case("limits")
