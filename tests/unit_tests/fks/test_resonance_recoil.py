"""Invariant and measure checks for the local resonance radiation maps."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import fortran_routine, fks_test_module


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "Template/NLO/SubProcesses"


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestResonanceRecoil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix="mg5_resonance_recoil_")
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.executables = {}
        for nincoming in (1, 2):
            work = Path(cls.tempdir.name) / str(nincoming)
            work.mkdir()
            shutil.copy(TEMPLATE / "fks_powers.inc", work)
            (work / "coupl.inc").write_text("")
            (work / "pmass.inc").write_text("      common/to_mass/pmass\n")
            (work / "run.inc").write_text(
                "      double precision ebeam(2)\n"
                "      integer lpp(2)\n"
                "      common/test_run/ebeam,lpp\n")
            (work / "nexternal.inc").write_text(
                "      integer nexternal,nincoming\n"
                f"      parameter (nexternal={nincoming + 5},nincoming={nincoming})\n")
            (work / "orders.inc").write_text(
                "      integer nsplitorders\n      parameter(nsplitorders=1)\n")
            (work / "genps.inc").write_text(
                "      integer max_branch,max_particles\n"
                "      parameter (max_branch=10,max_particles=10)\n")
            (work / "native_context.f90").write_text(
                "module mc_native_context\n"
                "logical :: native_mapping=.false.\nend module\n")
            (work / "fks_phase_space.f").write_text(fks_test_module())
            routines = []
            for name in ("rotate_invar", "trp_rotate_invar", "phspncheck_nocms",
                         "xlen4", "xmom_compare", "xmcompare", "xprintout"):
                routines.append(fortran_routine(TEMPLATE / "fks_singular.f", name))
            for name in ("dot", "rho", "threedot"):
                routines.append(fortran_routine(
                    ROOT / "Template/NLO/Source/kin_functions.f", name))
            (work / "maps.f").write_text("\n".join(routines))
            executable = work / "check_resonance_recoil"
            command = [shutil.which("gfortran"), "-O2", "-std=legacy",
                       "-ffixed-line-length-none", "-fcheck=all",
                       "-ffunction-sections", "-fdata-sections",
                       "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                       "-fno-automatic", "-I", str(work),
                       str(work / "native_context.f90"),
                       str(TEMPLATE / "fks_phase_space_data.f"),
                       str(TEMPLATE / "genps_fks_helpers.f"),
                       str(TEMPLATE / "FKSParams.f90"),
                       str(TEMPLATE / "genps_fks_radiation.f"),
                       str(work / "fks_phase_space.f"), str(work / "maps.f"),
                       str(TEMPLATE / "resonance_recoil.f"),
                       str(TEMPLATE / "initial_recoil.f"),
                       str(TEMPLATE / "boostwdir2.f"),
                       str(ROOT / "HELAS/boostx.F"),
                       str(ROOT / "HELAS/rotxxx.F"),
                       str(ROOT / "tests/input_files/check_resonance_recoil.f90"),
                       "-o", str(executable)]
            result = subprocess.run(command, cwd=work, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(result.stdout)
            cls.executables[nincoming] = executable

    def check_map(self, mode):
        for nincoming, executable in self.executables.items():
            with self.subTest(nincoming=nincoming):
                result = subprocess.run([str(executable), mode], text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("PASS " + mode, result.stdout)

    def test_locality_shells_and_resonance_mass(self):
        self.check_map("invariants")

    def test_inverse_and_massive_branches(self):
        self.check_map("inverse")

    def test_soft_and_collinear_counterevents(self):
        self.check_map("limits")

    def test_collinear_spin_phase_after_boost(self):
        self.check_map("spin")

    def test_decay_phase_space_volume(self):
        self.check_map("volume")

    def test_invalid_subsystems_are_rejected(self):
        self.check_map("invalid")

    def test_soft_mismatch_and_collinear_reference_scales(self):
        self.check_map("soft_scales")

    def test_production_flux_counterevents_and_native_projection(self):
        self.check_map("production")


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestResonancePartition(unittest.TestCase):
    def test_real_partition_and_native_mc_group_with_mixed_orders(self):
        with tempfile.TemporaryDirectory(prefix="mg5_resonance_partition_") as tmp:
            work = Path(tmp)
            includes = {
                "nexternal.inc": "integer nexternal,nincoming\nparameter(nexternal=5,nincoming=2)",
                "genps.inc": "integer max_branch,lmaxconfigs,ngraphs,ncolor\n"
                             "parameter(max_branch=3,lmaxconfigs=3,ngraphs=3,ncolor=1)",
                "nFKSconfigs.inc": "integer fks_configs\nparameter(fks_configs=1)",
                "run.inc": "",
                "born_conf.inc": "integer mapconfig(0:3),iforest(2,-3:-1,3),sprop(-3:-1,3),tprid(-3:-1,3)\n"
                                 "logical forcebw(-3:-1,3)\n"
                                 "common/MC_NATIVE_BORN_TOPOLOGY/mapconfig,iforest,sprop,tprid,forcebw",
            }
            for name, source in includes.items():
                (work / name).write_text("".join("      " + line + "\n" for line in source.splitlines()))
            shutil.copyfile(TEMPLATE / "timing_variables.inc", work / "timing_variables.inc")
            routines = [fortran_routine(TEMPLATE / filename, name) for filename, name in (
                ("fks_singular.f", "include_multichannel_enhance"),
                ("mc_native_runtime.f", "mc_outer_channel_weight"),
                ("resonance_histories.f", "native_recoil_weight"))]
            (work / "partitions.f").write_text("\n".join(routines))
            executable = work / "check_partitions"
            result = subprocess.run([
                shutil.which("gfortran"), "-O2", "-std=legacy", "-fcheck=all",
                "-ffixed-line-length-none", "-I", str(work),
                str(TEMPLATE / "fks_phase_space_data.f"),
                str(ROOT / "tests/input_files/check_resonance_partition.f90"),
                str(work / "partitions.f"), "-o", str(executable)],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS resonance partitions and mixed Born orders", result.stdout)
