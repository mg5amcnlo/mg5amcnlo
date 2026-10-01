"""Hadronic measure and subtraction checks for final-initial recoil."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import fortran_routine


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "Template/NLO/SubProcesses"


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestInitialRecoil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix="mg5_initial_recoil_")
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.executable = cls.build(5)
        cls.minimal_executable = cls.build(4)

    @classmethod
    def build(cls, nexternal):
        work = Path(cls.tempdir.name) / str(nexternal)
        work.mkdir()
        for name in ("resonance_recoil.inc", "fks_powers.inc", "timing_variables.inc"):
            shutil.copyfile(TEMPLATE / name, work / name)
        (work / "nexternal.inc").write_text(
            "      integer nexternal,nincoming\n"
            f"      parameter (nexternal={nexternal},nincoming=2)\n")
        (work / "genps.inc").write_text(
            "      integer max_branch,max_particles\n"
            "      parameter (max_branch=10,max_particles=10)\n")
        (work / "pmass.inc").write_text("      common/to_mass/pmass\n")
        (work / "run.inc").write_text(
            "      double precision ebeam(2)\n"
            "      integer lpp(2)\n"
            "      common/test_run/ebeam,lpp\n")
        (work / "coupl.inc").write_text("")
        (work / "native_context.f90").write_text(
            "module mc_native_context\n"
            "logical :: native_mapping=.false.\nend module\n")
        routines = []
        for name in ("generate_momenta_massless_final", "generate_momenta_massive_final",
                     "generate_momenta_massless_final_inverse", "generate_momenta_massive_final_inverse",
                     "native_fsr_angle", "getangles", "get_recoil",
                     "generate_FKS_kinematics", "generate_native_momenta", "invert_fks_radiation",
                     "generate_momenta_initial", "generate_momenta_initial_inverse",
                     "compute_flux", "fill_FKS_commons", "lambda", "yminmax", "gentcms"):
            routines.append(fortran_routine(TEMPLATE / "genps_fks.f", name))
        for name in ("rotate_invar", "trp_rotate_invar", "compute_prefactors_n1body",
                     "phspncheck_nocms", "xlen4", "xmom_compare", "xmcompare", "xprintout"):
            routines.append(fortran_routine(TEMPLATE / "fks_singular.f", name))
        for name in ("dot", "rho", "threedot"):
            routines.append(fortran_routine(ROOT / "Template/NLO/Source/kin_functions.f", name))
        (work / "maps.f").write_text("\n".join(routines))
        executable = work / "check_initial_recoil"
        result = subprocess.run([
            shutil.which("gfortran"), "-O2", "-std=legacy", "-fcheck=all",
            "-ffixed-line-length-none", "-ffunction-sections", "-fdata-sections",
            "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
            "-fno-automatic", "-I", str(work),
            str(work / "native_context.f90"), str(TEMPLATE / "process_module.f90"),
            str(TEMPLATE / "kinematics_module.f90"), str(work / "maps.f"),
            str(TEMPLATE / "resonance_recoil.f"),
            str(TEMPLATE / "initial_recoil.f"), str(TEMPLATE / "boostwdir2.f"),
            str(ROOT / "HELAS/boostx.F"), str(ROOT / "HELAS/rotxxx.F"),
            str(ROOT / "tests/input_files/check_initial_recoil.f90"),
            "-o", str(executable)], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return executable

    def check_map(self, mode):
        result = subprocess.run([str(self.executable), mode], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS " + mode, result.stdout)

    def test_locality_masses_and_inverse_for_both_beams(self):
        self.check_map("inverse")

    def test_soft_collinear_counterevents_and_measure(self):
        self.check_map("limits")

    def test_full_hadronic_phase_space_volume(self):
        self.check_map("volume")

    def test_integrated_and_endpoint_frame_conversion(self):
        self.check_map("endpoints")

    def test_invalid_recoilers_and_beam_fractions(self):
        self.check_map("invalid")

    def test_production_flux_pdfs_counterevents_and_asymmetric_inverse(self):
        self.check_map("production")

    def test_single_massive_born_final_state_without_auxiliary_slot(self):
        result = subprocess.run([str(self.minimal_executable), "minimal"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS minimal", result.stdout)

    def test_small_and_nearly_exhausted_born_beam_fraction(self):
        self.check_map("corners")
