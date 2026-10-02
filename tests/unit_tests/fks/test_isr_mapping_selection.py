"""ISR card parsing and shower compatibility through production FKSParams."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import TEMPLATE


DRIVER = """
program check_isr_mapping_selection
  use FKSParams
  implicit none
  character(32) :: mode,shower,arg
  integer :: choice,mapping
  logical :: fixed_order,nlo_ps

  call get_command_argument(1,mode)
  select case(trim(mode))
  case('card')
    if(FKSISRMapping.ne.0)error stop 'declaration default'
    call FKSParamReader('symmetric.dat',.false.,.true.)
    if(FKSISRMapping.ne.1)error stop 'read symmetric card'
    call FKSParamReader('asymmetric.dat',.false.,.true.)
    if(FKSISRMapping.ne.2)error stop 'read asymmetric card'
    call FKSParamReader('old.dat',.false.,.true.)
    if(FKSISRMapping.ne.0)error stop 'old card default'
    FKSISRMapping=2
    call DefaultFKSParam()
    if(FKSISRMapping.ne.0)error stop 'reset default'
    call FKSParamReader('automatic.dat',.true.,.true.)
    if(FKSISRMapping.ne.0)error stop 'read automatic card'
    write(*,*) 'PASS card'
  case('invalid_card')
    call get_command_argument(2,arg)
    call FKSParamReader(trim(arg),.false.,.true.)
    error stop 'accepted invalid ISR parameter'
  case('policy')
    call get_command_argument(2,arg)
    read(arg,*)choice
    FKSISRMapping=choice
    call get_command_argument(3,arg)
    read(arg,*)fixed_order
    call get_command_argument(4,arg)
    read(arg,*)nlo_ps
    call get_command_argument(5,shower)
    mapping=get_fks_isr_mapping(fixed_order,nlo_ps,trim(shower))
    write(*,'(a,i0)')'MAPPING ',mapping
  case default
    error stop 'unknown mode'
  end select
end program
"""


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestISRMappingSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix="mg5_isr_selection_")
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.work = work = Path(cls.tempdir.name)
        (work / "orders.inc").write_text(
            "      integer nsplitorders\n      parameter(nsplitorders=1)\n")
        (work / "driver.f90").write_text(DRIVER)
        for name, choice in (("automatic", 0), ("symmetric", 1),
                             ("asymmetric", 2), ("negative", -1), ("large", 3)):
            (work / (name + ".dat")).write_text("#FKSISRMapping\n%d\n" % choice)
        (work / "old.dat").write_text("#UsePolyVirtual\n.False.\n")
        result = subprocess.run([
            shutil.which("gfortran"), "-O2", "-fcheck=all", "-I", str(work),
            str(TEMPLATE / "FKSParams.f90"), str(work / "driver.f90"),
            "-o", str(work / "check")], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def run_mode(self, *args):
        return subprocess.run([str(self.work / "check"), *map(str, args)],
                              cwd=self.work, capture_output=True, text=True)

    def assert_mapping(self, choice, fixed_order, nlo_ps, shower, expected):
        result = self.run_mode("policy", choice, fixed_order, nlo_ps, shower)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("MAPPING %d" % expected, result.stdout)

    def test_card_reader_default_reset_and_print(self):
        result = self.run_mode("card")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS card", result.stdout)
        self.assertIn("FKSISRMapping", result.stdout)

    def test_invalid_card_values(self):
        for card in ("negative.dat", "large.dat"):
            with self.subTest(card=card):
                result = self.run_mode("invalid_card", card)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("FKSISRMapping", result.stdout + result.stderr)
                self.assertNotIn("accepted invalid", result.stdout + result.stderr)

    def test_fixed_order_default_and_both_explicit_choices(self):
        # A run card can retain a shower name in a fixed-order launch.
        for shower in ("PYTHIA8", "HERWIG7", "HERWIG6", "PYTHIA6Q", "PYTHIA6PT", ""):
            for choice, expected in ((0, 2), (1, 1), (2, 2)):
                with self.subTest(shower=shower, choice=choice):
                    self.assert_mapping(choice, True, False, shower, expected)

    def test_matching_preserves_asymmetric_default(self):
        for shower in ("PYTHIA8", "HERWIG7", "HERWIG6", "PYTHIA6Q", "PYTHIA6PT"):
            for choice in (0, 2):
                with self.subTest(shower=shower, choice=choice):
                    self.assert_mapping(choice, False, True, shower, 2)

    def test_diagnostic_default_preserves_current_map(self):
        self.assert_mapping(0, False, False, "PYTHIA8", 2)
        self.assert_mapping(1, False, False, "PYTHIA8", 1)

    def test_incompatible_shower_mapping_is_rejected(self):
        for shower in ("PYTHIA8", "HERWIG7", "HERWIG6", "PYTHIA6Q", "PYTHIA6PT"):
            for fixed_order in (False, True):
                with self.subTest(shower=shower, fixed_order=fixed_order):
                    result = self.run_mode("policy", 1, fixed_order, True, shower)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("FKSISRMapping", result.stdout + result.stderr)
                    self.assertIn(shower, result.stdout + result.stderr)
                    self.assertNotIn("MAPPING 1", result.stdout)

    def test_invalid_direct_selector_values(self):
        for choice in (-1, 3):
            with self.subTest(choice=choice):
                result = self.run_mode("policy", choice, True, False, "PYTHIA8")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("FKSISRMapping", result.stdout + result.stderr)
