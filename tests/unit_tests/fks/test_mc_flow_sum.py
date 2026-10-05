"""Born-flow summation and its FKS card switch, without matrix-element export."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMCFlowCard(unittest.TestCase):
    def test_default_override_and_old_card_reset(self):
        with tempfile.TemporaryDirectory(prefix='mg5_mc_flow_card_') as tmp:
            work = Path(tmp)
            (work / 'orders.inc').write_text(
                'integer nsplitorders\nparameter(nsplitorders=1)\n')
            (work / 'summed.dat').write_text(
                '#MCSubtractionAtFixedFlow\n.false.\n')
            (work / 'old.dat').write_text('#MCExplicitKLSum\n.true.\n')
            (work / 'check.f90').write_text('''program check_card
  use FKSParams
  implicit none
  character(1024) :: shipped_card
  call get_command_argument(1,shipped_card)
  if (.not.MCSubtractionAtFixedFlow) error stop 'declaration default'
  call DefaultFKSParam()
  if (.not.MCSubtractionAtFixedFlow) error stop 'routine default'
  call FKSParamReader('summed.dat',.true.,.true.)
  if (MCSubtractionAtFixedFlow) error stop 'false override'
  call FKSParamReader('old.dat',.false.,.true.)
  if (.not.MCSubtractionAtFixedFlow) error stop 'old card reset'
  call FKSParamReader('summed.dat',.false.,.true.)
  if (MCSubtractionAtFixedFlow) error stop 'repeated false override'
  call FKSParamReader(trim(shipped_card),.false.,.true.)
  if (.not.MCSubtractionAtFixedFlow) error stop 'shipped default'
  MCSubtractionAtFixedFlow=.false.
  call DefaultFKSParam()
  if (.not.MCSubtractionAtFixedFlow) error stop 'default reset'
  print *, 'PASS Born-flow card'
end program
''')
            executable = work / 'check'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all',
                '-I', str(work), str(TEMPLATE / 'FKSParams.f90'),
                str(work / 'check.f90'), '-o', str(executable)],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([
                str(executable), str(TEMPLATE.parent / 'Cards/FKS_params.dat')],
                cwd=work, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS Born-flow card', result.stdout)
            self.assertIn('MCSubtractionAtFixedFlow', result.stdout)


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMCFlowSum(unittest.TestCase):
    def test_native_sum_and_event_owner_restoration(self):
        with tempfile.TemporaryDirectory(prefix='mg5_mc_flow_sum_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=5,nincoming=2)',
                'born_nhel.inc': 'integer max_bcol\nparameter(max_bcol=3)',
                'orders.inc': 'integer nsplitorders\nparameter(nsplitorders=1)',
                'run.inc': 'integer ickkw\ncommon/test_run/ickkw',
            }
            for name, source in includes.items():
                (work / name).write_text(''.join('      ' + line + '\n'
                                               for line in source.splitlines()))
            (work / 'weights.f').write_text('\n'.join([
                fortran_routine(TEMPLATE / 'driver_mintMC.f',
                                'compute_NLOPS_flow_weights'),
                fortran_routine(TEMPLATE / 'fks_singular.f',
                                'include_born_flow_weight'),
                fortran_routine(TEMPLATE / 'fks_singular.f',
                                'get_born_flow_weights'),
            ]))
            executable = work / 'check'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
                '-ffixed-line-length-none', '-I', str(work),
                str(TEMPLATE / 'FKSParams.f90'),
                str(ROOT / 'tests/input_files/check_mc_flow_sum.f90'),
                str(work / 'weights.f'), '-o', str(executable)],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], cwd=work,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS Born-flow sum, sampling, compensation and owner restoration',
                          result.stdout)
