"""Numerical regressions for the shared Fortran FKS momentum maps.

Compile the production routines, without generating matrix elements or requiring
PDF/loop libraries. The driver tests inversion against independently generated
momenta, finite ISR subtraction integrals, and the lab boost when the massive
map has no counterevent.
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
    """Select a fixed-form routine, including contained FKS procedures."""
    paths = ([path.parent / source for source in (
                  "genps_fks.f", "genps_fks_sampling.f",
                  "genps_fks_radiation.f", "genps_fks_helpers.f")]
             if path.name == "genps_fks.f" else [path])
    pattern = re.compile(
        r"^      (?:subroutine|(?:(?:double precision|double complex|logical|integer) )?function) "
        + re.escape(name) + r"\b", re.MULTILINE | re.IGNORECASE)
    matches = []
    for candidate in paths:
        source = candidate.read_text()
        matches.extend((candidate, source, start) for start in pattern.finditer(source))
    if not matches:
        raise ValueError("Missing Fortran routine: " + name)
    if len(matches) != 1:
        raise ValueError("Ambiguous Fortran routine {} in {}".format(
            name, ", ".join(str(candidate) for candidate, _, _ in matches)))
    _, source, start = matches[0]
    end = re.search(r"^      end(?:[ \t]+(?:subroutine|function)"
                    r"(?:[ \t]+" + re.escape(name) + r")?)?[ \t]*$",
                    source[start.start():],
                    re.MULTILINE | re.IGNORECASE)
    if end is None:
        raise ValueError("Missing end of Fortran routine: " + name)
    return source[start.start():start.start() + end.end()] + "\n"


FKS_MAIN_ROUTINES = (
    "generate_native_momenta", "invert_fks_radiation",
    "generate_fks_radiation", "reject_fks_phase_space",
    "record_fks_phase_space", "capture_fks_phase_space",
    "generate_FKS_kinematics", "reset_fks_kinematics",
    "compute_flux", "fill_fks_point_data", "boost_born_momenta_noevpr")


def fks_test_module(names=FKS_MAIN_ROUTINES, template=TEMPLATE):
    """Retain production types and data imports around selected high-level routines."""
    source = (template / "genps_fks.f").read_text()
    header, _ = re.split(r"^      contains[ \t]*$", source, maxsplit=1,
                         flags=re.MULTILINE | re.IGNORECASE)
    # Born sampling needs generated topology tables. These radiation fixtures
    # use actual helper/radiation modules and expose selected orchestration
    # internals without copying their interfaces or host declarations.
    header = re.sub(r"^      (?:use fks_born_sampling\b|private\b|public\b)"
                    r"[^\n]*(?:\n     [^ 0\s][^\n]*)*\n?", "", header,
                    flags=re.MULTILINE | re.IGNORECASE)
    return (header + "      contains\n" + "\n".join(
        fortran_routine(template / "genps_fks.f", name) for name in names)
        + "      end module fks_phase_space\n")


MC_KINEMATICS_ROUTINES = (
    "prepare_mc_kinematics", "fill_father_and_ileg", "fill_ileg",
    "get_momenta_emitter_recoiler", "fill_invariants_ileg1",
    "fill_invariants_ileg2", "fill_invariants_ileg3", "fill_invariants_ileg4",
    "check_invariants_ileg12", "check_invariants_ileg3", "check_invariants_ileg4",
    "get_qMC", "qMC_ileg1", "qMC_ileg2", "qMC_ileg3", "qMC_ileg4",
    "py8_massive_fsr_fractions", "get_zeta", "compute_gfun", "gfunction",
    "mc_shower_scale_mass")


def mc_counterterm_test_module(names, template=TEMPLATE, external_functions=(),
                              include_kinematics=True):
    """Keep the production MC module's state around selected fixture routines.

    Fixtures expose internals for direct calls and may replace omitted
    functions with controlled external stubs, declared explicitly here.
    Shower fixtures retain their production kinematics helpers; pure flow
    fixtures can omit these helpers and their unrelated module dependencies.
    """
    names = tuple(names)
    if include_kinematics:
        names = tuple(dict.fromkeys(names + MC_KINEMATICS_ROUTINES))
    source = (template / "montecarlocounter.f").read_text()
    header, _ = re.split(r"^      contains[ \t]*$", source, maxsplit=1,
                         flags=re.MULTILINE | re.IGNORECASE)
    header = re.sub(r"^      (?:private\b|public\b)"
                    r"[^\n]*(?:\n     [^ 0\s][^\n]*)*\n?", "", header,
                    flags=re.MULTILINE | re.IGNORECASE)
    if not include_kinematics:
        header = re.sub(r"^      use (?:process_module|fks_phase_space_helpers)\b"
                        r"[^\n]*(?:\n     [^ 0\s][^\n]*)*\n?", "", header,
                        flags=re.MULTILINE | re.IGNORECASE)
    header += "".join("      {}, external :: {}\n".format(kind, name)
                      for kind, name in external_functions)
    return (header + "      contains\n" + "\n".join(
        fortran_routine(template / "montecarlocounter.f", name) for name in names)
        + "      end module mc_counterterms\n")


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
        (work / "orders.inc").write_text(
            "      integer nsplitorders\n      parameter(nsplitorders=1)\n")
        (work / "genps.inc").write_text(
            "      integer max_branch,max_particles\n"
            "      parameter (max_branch=8,max_particles=8)\n")
        shutil.copyfile(TEMPLATE / "fks_powers.inc", work / "fks_powers.inc")
        shutil.copyfile(TEMPLATE / "timing_variables.inc", work / "timing_variables.inc")
        (work / "native_context.f90").write_text(
            "module mc_native_context\n"
            "logical :: native_mapping=.false.\nend module\n")
        (work / "run.inc").write_text(
            "      double precision ebeam(2),xbk(2)\n"
            "      integer lpp(2)\n"
            "      common/test_run/ebeam,xbk,lpp\n")
        (work / "coupl.inc").write_text("")
        (work / "pmass.inc").write_text("      common/to_mass/pmass\n")
        (work / "fks_phase_space.f").write_text(fks_test_module())
        routines = []
        for name in ("rotate_invar", "trp_rotate_invar", "phspncheck_nocms", "xlen4",
                     "xmom_compare", "xmcompare", "xprintout",
                     "compute_prefactors_n1body", "set_cms_stuff"):
            routines.append(fortran_routine(TEMPLATE / "fks_singular.f", name))
        for name in ("dot", "rho", "threedot"):
            routines.append(fortran_routine(
                ROOT / "Template/NLO/Source/kin_functions.f", name))
        (work / "maps.f").write_text("\n".join(routines))
        cls.executable = work / "check_maps"
        command = [shutil.which("gfortran"), "-O2", "-std=legacy",
                   "-ffixed-line-length-none", "-ffunction-sections",
                   "-fdata-sections", "-fcheck=all", "-fno-automatic",
                   "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                   "-I", str(work),
                   str(TEMPLATE / "fks_phase_space_data.f"),
                   str(TEMPLATE / "genps_fks_helpers.f"),
                   str(work / "native_context.f90"),
                   str(TEMPLATE / "FKSParams.f90"),
                   str(TEMPLATE / "mcatnlo_delta_scales.f90"),
                   str(TEMPLATE / "genps_fks_radiation.f"),
                   str(work / "fks_phase_space.f"),
                   str(work / "maps.f"), str(TEMPLATE / "boostwdir2.f"),
                   str(TEMPLATE / "resonance_recoil.f"),
                   str(TEMPLATE / "initial_recoil.f"),
                   str(ROOT / "HELAS/boostx.F"), str(ROOT / "HELAS/rotxxx.F"),
                   str(ROOT / "tests/input_files/check_native_projection.f90"),
                   str(ROOT / "tests/input_files/check_isr_mapping.f90"),
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

    def test_native_projection_and_shared_state(self):
        self.check_map("native_projection")

    def test_native_massless_projection_with_soft_recoil(self):
        self.check_map("ee_soft_recoil")

    def test_dijet_native_isr_projection_preserves_mass_shell(self):
        self.check_map("dijet_isr_boundary")

    def test_onshell_isr_inverse_boost_at_large_rapidity(self):
        self.check_map("isr_onshell_boost")

    def test_wjet_native_projection_near_radiation_boundary(self):
        self.check_map("wjet_boundary")

    def test_single_top_native_massless_recoil(self):
        self.check_map("singletop_recoil")

    def test_initial_state_recoil_and_endpoints(self):
        self.check_map("isr")

    def test_symmetric_initial_state_recoil_inverse_measure_and_endpoints(self):
        self.check_map("isr_symmetric")

    def test_automatic_fixed_order_mapping_preserves_asymmetric_default(self):
        self.check_map("isr_automatic")

    def test_initial_state_fks_finite_integrals(self):
        self.check_map("isr_fks")

    def test_symmetric_initial_state_fks_finite_integrals(self):
        self.check_map("isr_symmetric_fks")

    def test_symmetric_initial_state_restricted_and_empty_domains(self):
        self.check_map("isr_symmetric_bounds")

    def test_asymmetric_beam_boost(self):
        self.check_map("boost")

    def test_t_channel_bounds_with_soft_massless_recoil(self):
        self.check_map("born_threshold")

    def test_massless_final_inverse(self):
        self.check_map("massless")

    def test_outer_massless_map_with_soft_daughter(self):
        self.check_map("soft_daughter")

    def test_finite_soft_momenta_keep_their_own_directions(self):
        self.check_map("soft_direction")

    def test_wjet_native_history_with_soft_gluon(self):
        self.check_map("wjet_soft_history")

    def test_outer_massive_map_with_soft_sister(self):
        self.check_map("outer_angles")

    def test_outer_massive_map_with_small_recoil(self):
        self.check_map("outer_massive_recoil")
        self.check_map("outer_massless_recoil")
        self.check_map("outer_massless_recoil2")

    def test_massless_inverse_with_soft_recoil_in_both_maps(self):
        self.check_map("soft_recoil_inverse")

    def test_massive_soft_counterevents(self):
        self.check_map("soft_counterevent")

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
