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
            routines.extend(spin_frame_routines())
            (work / "maps.f").write_text("\n".join(routines))
            (work / "spin_check.f90").write_text(SPIN_CHECK)
            executable = work / "check_resonance_recoil"
            command = [shutil.which("gfortran"), str(TEMPLATE.parent / 'Source/timing.f'), "-O2", "-std=legacy",
                       "-ffixed-line-length-none", "-fcheck=all",
                       "-ffunction-sections", "-fdata-sections",
                       "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                       "-I", str(work), "-I", str(TEMPLATE),
                       str(work / "native_context.f90"),
                       str(TEMPLATE / "fks_phase_space_data.f"),
                       str(TEMPLATE / "genps_fks_helpers.f"),
                       str(TEMPLATE / "FKSParams.f90"),
                       str(TEMPLATE / "genps_fks_radiation.f"),
                       str(work / "fks_phase_space.f"), str(work / "maps.f"),
                       str(work / "spin_check.f90"),
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
                shutil.which("gfortran"), str(TEMPLATE.parent / 'Source/timing.f'), "-O2", "-std=legacy", "-fcheck=all",
                "-ffixed-line-length-none", "-I", str(work), "-I", str(TEMPLATE),
                str(TEMPLATE / "fks_phase_space_data.f"),
                str(ROOT / "tests/input_files/check_resonance_partition.f90"),
                str(work / "partitions.f"), "-o", str(executable)],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS resonance partitions and mixed Born orders", result.stdout)


def spin_frame_routines():
    """Keep the real frame/phase consumers; only generated masks are supplied."""
    return [fortran_routine(TEMPLATE / "boost_to_frame.f", name) for name in
            ("azifact_me_frame", "azifact_mc_frame", "azifact_from_kperp",
             "get_me_frame_boost", "me_frame_apply", "snap_mother_at_rest")] + [
        fortran_routine(TEMPLATE / "fks_singular.f", "getaziangles"),
        fortran_routine(ROOT / "Template/Common/Source/impose_frame_rest.f",
                        "get_me_frame_boosts")]


SPIN_CHECK = """subroutine check_recoil_spin(pborn,imother,pi,pj,transported,check_pair)
use fks_phase_space_data, only: xij_kperp,xij_aor
use fks_phase_space, only: fks_phase_space_point,capture_fks_phase_space,restore_fks_phase_space
use fks_phase_space_helpers, only: getangles
implicit none
include 'nexternal.inc'
real(8) :: pborn(0:3,nexternal-1),pi(0:3),pj(0:3),transported(0:3)
real(8) :: boosts(0:3,2),mother(0:3),kt(0:3),pib(0:3),pjb(0:3),pair(0:3),rot(0:3)
real(8) :: th,cth,sth,ph,cph,sph
complex(8) :: expected,actual,phase_save
integer :: imother,ids(nexternal-1)
common/test_spin_frame_mask/ids
logical :: check_pair,trivial,boosted
type(fks_phase_space_point) :: saved
! The time component is essential: a second boost mixes it back into space.
if(maxval(abs(xij_kperp-transported)).gt.2d-8)error stop 'local transverse vector not transported'
if(abs(xij_kperp(0)*pborn(0,imother)-sum(xij_kperp(1:3)*pborn(1:3,imother))).gt.2d-7) &
    error stop 'spin vector not orthogonal to Born mother'
if(abs(xij_kperp(0)**2-sum(xij_kperp(1:3)**2)+1d0).gt.2d-8)error stop 'spin vector norm'
phase_save=xij_aor
call capture_fks_phase_space(saved)
if(any(saved%spin_transverse.ne.xij_kperp))error stop 'spin vector capture'
xij_kperp=13d0
xij_aor=(0d0,0d0)
call restore_fks_phase_space(saved)
if(any(xij_kperp.ne.saved%spin_transverse))error stop 'spin vector restore'
if(xij_aor.ne.phase_save)error stop 'spin phase restore'
! Select a nontrivial timelike subsystem, using the same two-step boost
! that the Born and its collinear spin contraction actually consume.
ids=0
ids(1)=1
ids(imother)=1
call get_me_frame_boost(pborn,nexternal-1,ids,boosts,trivial)
if(trivial)error stop 'spin fixture requested a trivial me_frame'
call me_frame_apply(pborn(:,imother),boosts,mother)
call me_frame_apply(transported,boosts,kt)
call azifact_from_kperp(mother,kt,expected)
call azifact_me_frame(pborn,imother,pi,pj,1d0,actual,boosted)
if(.not.boosted.or.abs(actual-expected).gt.1d-10)error stop 'local recoil me_frame phase'
call getaziangles(mother,cph,sph)
call azifact_mc_frame(pborn,imother,pi,pj,1d0,boosts,actual,cph,sph)
if(abs(actual-expected*dcmplx(cph,sph)**2).gt.1d-10)error stop 'local recoil MC frame phase'
if(check_pair)then
  ! The azimuth of a finite nearly collinear physical pair is an
  ! independent limit; no saved transverse vector enters this reference.
  call me_frame_apply(pi,boosts,pib)
  call me_frame_apply(pj,boosts,pjb)
  pair=pib+pjb
  call getangles(pair,th,cth,sth,ph,cph,sph)
  call trp_rotate_invar(pib,rot,cth,sth,cph,sph)
  actual=-dcmplx(rot(1),rot(2))**2/(rot(1)**2+rot(2)**2)
  if(abs(actual-expected).gt.3d-3)then
    write(*,*)'physical pair and saved vector',actual,expected,abs(actual-expected)
    error stop 'local recoil physical collinear phase'
  endif
endif
end subroutine

subroutine get_frame_mask_born(mask)
implicit none
include 'nexternal.inc'
integer :: mask(nexternal-1),ids(nexternal-1)
common/test_spin_frame_mask/ids
mask=ids
end subroutine

subroutine azifact_from_spinors(pi,pj,phase)
! This fixture exercises exactly the collinear consumers; the finite
! physical-pair reference is reconstructed geometrically above.
implicit none
real(8) :: pi(0:3),pj(0:3)
complex(8) :: phase
error stop 'unexpected finite-angle spinor branch'
end subroutine
"""
