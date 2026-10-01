"""Card and explicit recoil-selection checks using the production routines."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "Template/NLO/SubProcesses"


DRIVER = """
program check_recoil_selection
  use FKSParams
  implicit none
  include 'nexternal.inc'
  logical :: local, members(nexternal), mask(nexternal), defaults(nexternal), pass
  double precision :: momentum(0:3), mass2, pmass(nexternal)
  common /c_resonance_recoil/ momentum, mass2, local, members
  integer :: beam, ifks, jfks
  common /c_initial_recoil/ beam
  common /fks_indices/ ifks, jfks
  common /to_mass/ pmass
  integer :: lpp(2)
  double precision :: ebeam(2), xbk(2), q2fact(2)
  common /to_collider/ ebeam, xbk, q2fact, lpp
  logical :: fixed_order, nlo_ps
  common /c_fnlo_nlops/ fixed_order, nlo_ps
  character(32) :: mode

  call get_command_argument(1, mode)
  ifks = nexternal
  jfks = 3
  pmass = 0d0
  lpp = 1
  fixed_order = .true.
  nlo_ps = .false.
  defaults = .false.
  mask = .false.

  select case (trim(mode))
  case ('selection')
    call select_fks_recoil(defaults, .true.)
    if (local.or.beam.ne.0) error stop 'default global recoil'
    FKSFinalRecoil = 1
    call select_fks_recoil(defaults, .true.)
    if (.not.local.or.beam.ne.1.or.any(members)) error stop 'beam one'
    FKSFinalRecoil = 2
    call select_fks_recoil(defaults, .true.)
    if (.not.local.or.beam.ne.2.or.any(members)) error stop 'beam two'
    mask(4:5) = .true.
    call set_fks_recoilers(mask, pass)
    if (.not.pass) error stop 'explicit final system rejected'
    call select_fks_recoil(defaults, .true.)
    if (.not.local.or.beam.ne.0) error stop 'explicit final override'
    mask(ifks) = .true.
    mask(jfks) = .true.
    if (any(members.neqv.mask)) error stop 'subsystem members'
    call set_fks_recoilers(mask, pass)
    if (pass) error stop 'accepted emission and emitter as recoilers'
    call select_fks_recoil(defaults, .true.)
    if (any(members.neqv.mask)) error stop 'invalid request mutated selection'
    call clear_fks_recoilers()
    call select_fks_recoil(defaults, .true.)
    if (beam.ne.2) error stop 'clear did not restore card policy'
    mask = .false.
    mask(1) = .true.
    call set_fks_recoilers(mask, pass)
    if (.not.pass) error stop 'explicit initial recoiler rejected'
    call select_fks_recoil(defaults, .true.)
    if (beam.ne.1) error stop 'explicit initial override'
    call select_fks_recoil(defaults, .false.)
    if (local.or.beam.ne.0.or.any(members)) error stop 'disabled selection'
    jfks = 1
    call select_fks_recoil(defaults, .true.)
    if (local.or.beam.ne.0.or.any(members)) error stop 'initial emitter'
    jfks = 3
    call clear_fks_recoilers()
    FKSFinalRecoil = 0
    defaults(3:6) = .true.
    call select_fks_recoil(defaults, .true.)
    if (.not.local.or.beam.ne.0) error stop 'resonance default'
    if (any(defaults.neqv.members)) error stop 'resonance mask'
    call select_fks_recoil(members, .true.)
    if (any(defaults.neqv.members)) error stop 'aliased default mask'

  case ('invalid')
    call set_fks_recoilers(mask, pass)
    if (pass) error stop 'accepted empty recoil'
    mask(1) = .true.
    mask(4) = .true.
    call set_fks_recoilers(mask, pass)
    if (pass) error stop 'accepted mixed recoil'
    mask(4) = .false.
    mask(2) = .true.
    call set_fks_recoilers(mask, pass)
    if (pass) error stop 'accepted two beam recoilers'
    mask(2) = .false.
    lpp(1) = 0
    call set_fks_recoilers(mask, pass)
    if (pass) error stop 'accepted fixed beam'
    lpp(1) = 3
    call set_fks_recoilers(mask, pass)
    if (pass) error stop 'accepted dressed lepton beam'
    lpp(1) = 1
    pmass(1) = 1d0
    call set_fks_recoilers(mask, pass)
    if (pass) error stop 'accepted massive incoming recoiler'
    pmass(1) = 0d0
    mask = .false.
    mask(jfks) = .true.
    call set_fks_recoilers(mask, pass)
    if (pass) error stop 'accepted emitter as recoiler'
    mask = .false.
    mask(ifks) = .true.
    call set_fks_recoilers(mask, pass)
    if (pass) error stop 'accepted emission as recoiler'
    mask = .false.
    mask(4) = .true.
    jfks = 1
    call set_fks_recoilers(mask, pass)
    if (pass) error stop 'accepted initial emitter'
    jfks = 3
    ifks = nexternal + 1
    call set_fks_recoilers(mask, pass)
    if (pass) error stop 'accepted invalid emission label'

  case ('decay')
    mask(1) = .true.
    call set_fks_recoilers(mask, pass)
    if (pass) error stop 'accepted decay initial recoil'
    mask = .false.
    mask(4:5) = .true.
    call set_fks_recoilers(mask, pass)
    if (.not.pass) error stop 'rejected decay final recoil'
    call select_fks_recoil(defaults, .true.)
    if (.not.local.or.beam.ne.0) error stop 'decay final selection'

  case ('card')
    if (FKSFinalRecoil.ne.0) error stop 'declaration default'
    call FKSParamReader('beam_card.dat', .false., .true.)
    if (FKSFinalRecoil.ne.2) error stop 'read beam card'
    call FKSParamReader('old_card.dat', .false., .true.)
    if (FKSFinalRecoil.ne.0) error stop 'old card default'
    FKSFinalRecoil = 1
    call DefaultFKSParam()
    if (FKSFinalRecoil.ne.0) error stop 'reset default'

  case ('invalid_card')
    call FKSParamReader('invalid_card.dat', .false., .true.)
    error stop 'accepted invalid recoil parameter'

  case ('mcatnlo')
    FKSFinalRecoil = 1
    fixed_order = .false.
    nlo_ps = .true.
    call select_fks_recoil(defaults, .true.)
    error stop 'accepted initial recoil with MC@NLO'

  case default
    error stop 'unknown test mode'
  end select
  write(*,*) 'PASS ', trim(mode)
end program
"""


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestRecoilSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix="mg5_recoil_selection_")
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.workdirs = {}
        for nincoming in (1, 2):
            work = Path(cls.tempdir.name) / str(nincoming)
            work.mkdir()
            (work / "nexternal.inc").write_text(
                "      integer nexternal,nincoming\n"
                f"      parameter(nexternal=6,nincoming={nincoming})\n")
            (work / "orders.inc").write_text(
                "      integer nsplitorders\n      parameter(nsplitorders=1)\n")
            (work / "driver.f90").write_text(DRIVER)
            (work / "beam_card.dat").write_text("#FKSFinalRecoil\n2\n")
            (work / "old_card.dat").write_text("#UsePolyVirtual\n.False.\n")
            (work / "invalid_card.dat").write_text("#FKSFinalRecoil\n3\n")
            result = subprocess.run([
                shutil.which("gfortran"), "-O2", "-std=legacy",
                "-ffixed-line-length-none", "-fcheck=all", "-I", str(work),
                str(TEMPLATE / "FKSParams.f90"),
                str(TEMPLATE / "recoil_selection.f"),
                str(work / "driver.f90"), "-o", str(work / "check")],
                cwd=work, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            cls.workdirs[nincoming] = work

    def run_mode(self, mode, nincoming=2):
        work = self.workdirs[nincoming]
        return subprocess.run([str(work / "check"), mode], cwd=work,
                              capture_output=True, text=True)

    def check_mode(self, mode, nincoming=2):
        result = self.run_mode(mode, nincoming)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS " + mode, result.stdout)

    def test_card_policy_explicit_overrides_and_aliasing(self):
        self.check_mode("selection")

    def test_invalid_recoilers_and_beam_configurations(self):
        self.check_mode("invalid")

    def test_decay_only_allows_final_recoilers(self):
        self.check_mode("decay", nincoming=1)

    def test_card_reader_and_backwards_compatible_default(self):
        self.check_mode("card")

    def test_invalid_card_parameter_is_rejected(self):
        result = self.run_mode("invalid_card")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FKSFinalRecoil must be 0, 1 or 2.", result.stdout)

    def test_initial_recoil_is_rejected_for_mcatnlo(self):
        result = self.run_mode("mcatnlo")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MC@NLO counterterms require their own map", result.stdout)
