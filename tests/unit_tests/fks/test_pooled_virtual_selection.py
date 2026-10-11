"""Parse pooled virtual options through the production FKS card reader."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from itertools import permutations


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "Template/NLO/SubProcesses"


DRIVER = """
program check_pooled_virtual_selection
  use FKSParams
  implicit none
  character(512) :: card
  logical :: expected_poly,expected_pooled
  call DefaultFKSParam()
  if(.not.use_poly_virtual.or.use_pooled_virtual)error stop 'legacy defaults'
  call get_command_argument(1,card)
  call FKSParamReader(trim(card),.true.,.true.)
  call get_command_argument(2,card)
  read(card,*)expected_poly
  call get_command_argument(3,card)
  read(card,*)expected_pooled
  if(use_poly_virtual.neqv.expected_poly)error stop 'polynomial selection'
  if(use_pooled_virtual.neqv.expected_pooled)error stop 'pooled selection'
  call FKSParamReader('empty.dat',.false.,.true.)
  if(.not.use_poly_virtual.or.use_pooled_virtual)error stop 'reset defaults'
  write(*,*) 'PASS'
end program
"""


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestPooledVirtualSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix="mg5_pooled_selection_")
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.work = work = Path(cls.tempdir.name)
        (work / "orders.inc").write_text(
            "      integer nsplitorders\n      parameter(nsplitorders=1)\n")
        (work / "driver.f90").write_text(DRIVER)
        (work / "empty.dat").write_text("")
        result = subprocess.run([
            shutil.which("gfortran"), "-O2", "-fcheck=all", "-I", str(work),
            str(TEMPLATE / "FKSParams.f90"), str(work / "driver.f90"),
            "-o", str(work / "check")], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def check_card(self, card, poly=True, pooled=False):
        result = subprocess.run([
            str(self.work / "check"), str(card), str(poly), str(pooled)], cwd=self.work,
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)
        self.assertIn("UsePooledVirtual", result.stdout)

    def test_old_cards_retain_both_existing_modes(self):
        self.check_card(self.work / "empty.dat")
        for poly in (False, True):
            with self.subTest(poly=poly):
                card = self.work / "old.dat"
                card.write_text("#UsePolyVirtual\n.%s.\n" % poly)
                self.check_card(card, poly=poly)

    def test_pooled_precedence_is_independent_of_card_order(self):
        options = ["#UsePooledVirtual\n.True.\n", "#UsePolyVirtual\n.True.\n"]
        for ordering in permutations(options):
            card = self.work / 'pooled.dat'
            card.write_text(''.join(ordering))
            self.check_card(card, pooled=True, poly=False)

    def test_shipped_card_enables_guarded_pooled_mode(self):
        self.check_card(ROOT / "Template/NLO/Cards/FKS_params.dat", poly=False, pooled=True)

    def test_disabled_pooled_preserves_legacy_selection(self):
        for poly in (False, True):
            card = self.work / 'no_pool.dat'
            card.write_text('#UsePooledVirtual\n.False.\n#UsePolyVirtual\n.%s.\n' % poly)
            self.check_card(card, poly=poly)
