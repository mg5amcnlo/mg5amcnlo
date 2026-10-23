"""Check the combined FxFx/Delta integrand with independent analytic weights."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestFxFxDeltaWeights(unittest.TestCase):
    def test_real_transfer_and_native_counterterms(self):
        with tempfile.TemporaryDirectory(prefix='mg5_fxfx_delta_weights_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=5,nincoming=2)',
                'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=1)',
                'coupl.inc': 'double precision g\ncommon/test_couplings/g',
                'orders.inc': 'integer nsplitorders,amp_split_size,qcd_pos,cpower_pos\n'
                              'parameter(nsplitorders=1,amp_split_size=1,qcd_pos=1,cpower_pos=0)\n'
                              'double precision amp_split(1)\ncommon/test_amplitudes/amp_split',
                'pmass.inc': 'pmass=0d0',
                'run.inc': 'integer ickkw,ktscheme\ncommon/test_run/ickkw,ktscheme',
            }
            for name, source in includes.items():
                (work / name).write_text(''.join(
                    '      ' + line + '\n' for line in source.splitlines()))
            for name in ('timing_variables.inc', 'fks_powers.inc'):
                shutil.copyfile(TEMPLATE / name, work / name)
            routines = ('compute_native_NLOPS_weights', 'compute_MC_subt_term',
                        'compute_real_emission', 'compute_soft_counter_term',
                        'compute_collinear_counter_term',
                        'compute_soft_collinear_counter_term', 'set_FxFx_scale')
            (work / 'weights.f').write_text('\n'.join(
                fortran_routine(TEMPLATE / 'fks_singular.f', name) for name in routines))
            executable = work / 'check_fxfx_delta_weights'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
                '-ffixed-line-length-none', '-I', str(work),
                str(TEMPLATE.parent / 'Source/timing.f'),
                str(ROOT / 'tests/input_files/check_fxfx_delta_weights.f90'),
                str(work / 'weights.f'), '-o', str(executable)],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS combined FxFx Delta weights', result.stdout)
