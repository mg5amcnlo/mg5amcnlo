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
  integer :: expected_surrogate
  call DefaultFKSParam()
  if(.not.use_poly_virtual.or.use_pooled_virtual)error stop 'legacy defaults'
  if(VirtTrickSurrogate/=2)error stop 'surrogate default'
  call get_command_argument(1,card)
  call FKSParamReader(trim(card),.true.,.true.)
  call get_command_argument(2,card)
  read(card,*)expected_poly
  call get_command_argument(3,card)
  read(card,*)expected_pooled
  if(use_poly_virtual.neqv.expected_poly)error stop 'polynomial selection'
  if(use_pooled_virtual.neqv.expected_pooled)error stop 'pooled selection'
  call get_command_argument(4,card)
  read(card,*)expected_surrogate
  if(VirtTrickSurrogate/=expected_surrogate)error stop 'surrogate selection'
  call FKSParamReader('empty.dat',.false.,.true.)
  if(.not.use_poly_virtual.or.use_pooled_virtual)error stop 'reset defaults'
  if(VirtTrickSurrogate/=2)error stop 'reset surrogate default'
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

    def check_card(self, card, poly=True, pooled=False, surrogate=None):
        if surrogate is None:
            surrogate = 2
        result = subprocess.run([
            str(self.work / "check"), str(card), str(poly), str(pooled), str(surrogate)], cwd=self.work,
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
        options = ["#UsePooledVirtual\n.True.\n", "#UsePolyVirtual\n.True.\n",
                   "#VirtTrickSurrogate\n2\n"]
        for ordering in permutations(options):
            card = self.work / 'pooled.dat'
            card.write_text(''.join(ordering))
            self.check_card(card, pooled=True, poly=False)

    def test_shipped_card_enables_per_channel_surrogate(self):
        self.check_card(ROOT / "Template/NLO/Cards/FKS_params.dat", poly=False, pooled=False)

    def test_surrogate_modes_preserve_polynomial_fallback(self):
        for surrogate in (0, 1, 2):
            card = self.work / 'surrogate.dat'
            card.write_text('#VirtTrickSurrogate\n%d\n' % surrogate)
            self.check_card(card, surrogate=surrogate)

    def test_invalid_surrogate_mode_is_rejected(self):
        for surrogate in (-1, 3):
            card = self.work / 'invalid_surrogate.dat'
            card.write_text('#VirtTrickSurrogate\n%d\n' % surrogate)
            result = subprocess.run([str(self.work / 'check'), str(card)],
                                    cwd=self.work, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('VirtTrickSurrogate must be 0, 1 or 2', result.stdout + result.stderr)

    def test_disabled_pooled_preserves_legacy_selection(self):
        for poly in (False, True):
            card = self.work / 'no_pool.dat'
            card.write_text('#UsePooledVirtual\n.False.\n#UsePolyVirtual\n.%s.\n' % poly)
            self.check_card(card, poly=poly)
