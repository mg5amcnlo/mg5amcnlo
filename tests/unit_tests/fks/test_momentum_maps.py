"""Numerical regressions for the Fortran maps used by MC H histories.

Compile the production routines, without generating matrix elements or requiring
PDF/loop libraries. The driver tests inversion against independently generated
momenta and checks the lab boost when the massive map has no counterevent.
"""

from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "Template/NLO/SubProcesses"


def fortran_routine(path, name):
    """Select a standalone fixed-form routine and its original dependencies."""
    source = path.read_text()
    start = re.search(
        r"^      (?:subroutine|(?:double precision )?function) " + name + r"\b",
        source, re.MULTILINE | re.IGNORECASE)
    if start is None:
        raise ValueError("Missing Fortran routine: " + name)
    end = re.search(r"^      end[ \t]*$", source[start.start():],
                    re.MULTILINE | re.IGNORECASE)
    if end is None:
        raise ValueError("Missing end of Fortran routine: " + name)
    return source[start.start():start.start() + end.end()] + "\n"


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestMomentumMaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix="mg5_momentum_maps_")
        cls.addClassCleanup(cls.tempdir.cleanup)
        work = Path(cls.tempdir.name)
        # These maps only need the multiplicity and array bounds from export.
        (work / "nexternal.inc").write_text(
            "      integer nexternal,nincoming\n"
            "      parameter (nexternal=5,nincoming=2)\n")
        (work / "genps.inc").write_text(
            "      integer max_branch,max_particles\n"
            "      parameter (max_branch=8,max_particles=8)\n")
        shutil.copyfile(TEMPLATE / "fks_powers.inc", work / "fks_powers.inc")
        (work / "native_context.f90").write_text(
            "module mc_native_context\n"
            "logical :: native_mapping=.false.\nend module\n")
        routines = []
        for name in ("generate_momenta_massive_final",
                     "generate_momenta_massless_final",
                     "generate_momenta_massive_final_inverse",
                     "generate_momenta_massless_final_inverse",
                     "native_fsr_angle",
                     "lambda", "yminmax", "gentcms", "gentcms_inverse",
                     "rotxxx_inv",
                     "fill_FKS_commons", "getangles", "get_recoil"):
            routines.append(fortran_routine(TEMPLATE / "genps_fks.f", name))
        routines.append(fortran_routine(TEMPLATE / "fks_singular.f", "rotate_invar"))
        for name in ("dot", "rho", "threedot"):
            routines.append(fortran_routine(
                ROOT / "Template/NLO/Source/kin_functions.f", name))
        (work / "maps.f").write_text("\n".join(routines))
        cls.executable = work / "check_maps"
        command = [shutil.which("gfortran"), "-O2", "-std=legacy",
                   "-ffixed-line-length-none", "-ffunction-sections",
                   "-fdata-sections",
                   "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                   "-I", str(work),
                   str(TEMPLATE / "process_module.f90"),
                   str(TEMPLATE / "kinematics_module.f90"),
                   str(work / "native_context.f90"),
                   str(work / "maps.f"), str(TEMPLATE / "boostwdir2.f"),
                   str(ROOT / "HELAS/boostx.F"), str(ROOT / "HELAS/rotxxx.F"),
                   str(ROOT / "tests/input_files/check_momentum_maps.f90"),
                   "-o", str(cls.executable)]
        result = subprocess.run(command, cwd=work, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(result.stdout)

    def check_map(self, name):
        result = subprocess.run([str(self.executable), name], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("PASS " + name, result.stdout)

    def test_asymmetric_beam_boost(self):
        self.check_map("boost")

    def test_t_channel_inverse_with_soft_massless_recoil(self):
        self.check_map("born_threshold")

    def test_massless_final_inverse(self):
        self.check_map("massless")

    def test_massive_final_inverse_both_solutions(self):
        self.check_map("massive")

    def test_native_maps_below_outer_sampling_cutoffs(self):
        self.check_map("native_massless")
        self.check_map("native_massive")

    def test_massive_history_with_stationary_recoil(self):
        self.check_map("massive_recoil")

    def test_massive_history_near_branch_boundary(self):
        self.check_map("massive_branch")

    def test_massive_history_with_soft_massless_recoil(self):
        self.check_map("massless_recoil")
        self.check_map("massless_recoil2")
