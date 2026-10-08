"""Exercise the NLO+PS backend switch through the production card reader."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import TEMPLATE


DRIVER = """
program check_nlops_integrator_selection
  use FKSParams
  implicit none
  character(512) :: mode,filename

  call get_command_argument(1,mode)
  select case(trim(mode))
  case('defaults')
    if(NLOPSIntegrator.ne.0)error stop 'declaration default'
    call FKSParamReader('amplicol.dat',.false.,.true.)
    if(NLOPSIntegrator.ne.1)error stop 'read AmpliCol card'
    call FKSParamReader('old.dat',.false.,.true.)
    if(NLOPSIntegrator.ne.0)error stop 'old card default'
    NLOPSIntegrator=1
    call DefaultFKSParam()
    if(NLOPSIntegrator.ne.0)error stop 'reset default'
    write(*,*) 'PASS defaults'
  case('card')
    call get_command_argument(2,filename)
    call FKSParamReader(trim(filename),.true.,.true.)
    write(*,'(a,i0)') 'INTEGRATOR ',NLOPSIntegrator
  case('invalid')
    call get_command_argument(2,filename)
    call FKSParamReader(trim(filename),.false.,.true.)
    error stop 'accepted invalid integrator'
  case default
    error stop 'unknown mode'
  end select
end program
"""


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestNLOPSIntegratorSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_nlops_integrator_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.work = work = Path(cls.tempdir.name)
        (work / 'orders.inc').write_text(
            '      integer nsplitorders\n      parameter(nsplitorders=2)\n')
        (work / 'driver.f90').write_text(DRIVER)
        for name, choice in (('mint', 0), ('amplicol', 1),
                             ('negative', -1), ('large', 2)):
            (work / (name + '.dat')).write_text('#NLOPSIntegrator\n%d\n' % choice)
        (work / 'old.dat').write_text('#UsePolyVirtual\n.False.\n')
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-fcheck=all', '-I', str(work),
            str(TEMPLATE / 'FKSParams.f90'), str(work / 'driver.f90'),
            '-o', str(work / 'check')], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def run_mode(self, *args):
        return subprocess.run([str(self.work / 'check'), *map(str, args)],
                              cwd=self.work, capture_output=True, text=True)

    def test_defaults_and_old_cards_keep_mint(self):
        result = self.run_mode('defaults')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS defaults', result.stdout)

    def test_supported_choices_are_read_and_printed(self):
        for card, choice in (('mint.dat', 0), ('amplicol.dat', 1)):
            with self.subTest(card=card):
                result = self.run_mode('card', card)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('INTEGRATOR %d' % choice, result.stdout)
                self.assertIn(' > NLOPSIntegrator', result.stdout)

    def test_template_card_keeps_mint(self):
        result = self.run_mode('card', TEMPLATE.parent / 'Cards/FKS_params.dat')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('INTEGRATOR 0', result.stdout)

    def test_invalid_choices_are_rejected(self):
        for card in ('negative.dat', 'large.dat'):
            with self.subTest(card=card):
                result = self.run_mode('invalid', card)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('NLOPSIntegrator must be 0 (MINT) or 1 (AmpliCol)',
                              result.stdout + result.stderr)
                self.assertNotIn('accepted invalid integrator',
                                 result.stdout + result.stderr)
