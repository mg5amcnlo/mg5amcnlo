"""Fixed-order FxFx counterevents must keep their underlying Born scales."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestFxFxFixedOrder(unittest.TestCase):
    def test_counterevent_and_real_scales(self):
        with tempfile.TemporaryDirectory(prefix='mg5_fxfx_fixed_order_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=7,nincoming=2)',
                'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=1)',
                'run.inc': 'integer ickkw,ktscheme\n'
                           'logical do_rwgt_scale,do_rwgt_pdf\n'
                           'common/test_run/ickkw,ktscheme,do_rwgt_scale,do_rwgt_pdf',
                'orders.inc': 'integer amp_split_size,maxchannels\n'
                              'parameter(amp_split_size=1,maxchannels=1)',
                'fks_info.inc': '',
                'has_ewsudakov.inc': '',
            }
            for name, source in includes.items():
                (work / name).write_text(''.join(
                    '      ' + line + '\n' for line in source.splitlines()))
            shutil.copyfile(TEMPLATE / 'timing_variables.inc',
                            work / 'timing_variables.inc')
            (work / 'driver.f').write_text(fortran_routine(
                TEMPLATE / 'driver_mintFO.f', 'sigint'))
            (work / 'fxfx.f').write_text(fortran_routine(
                TEMPLATE / 'fks_singular.f', 'set_FxFx_scale'))
            # These operations do not participate in selecting the scales.
            # The fixture supplies Born/real process metadata and records the
            # scales received by every contribution, including counterevents
            # that pass collinear cuts while their Born point fails cuts.
            noops = ('update_fks_dir', 'compute_prefactors_nbody',
                     'compute_prefactors_n1body', 'include_multichannel_enhance',
                     'compute_ewsudakov', 'init_process_module_n1body_wrapper',
                     'include_veto_multiplier', 'include_PDF_and_alphas',
                     'include_bias_wgt', 'reweight_scale', 'reweight_scale_NNLL',
                     'reweight_pdf', 'fill_pineappl_weights', 'fill_MC_integer',
                     'fill_plots')
            (work / 'noops.f90').write_text('\n'.join(
                'subroutine ' + name + '\nend subroutine' for name in noops))
            executable = work / 'check_fixed_order'
            result = subprocess.run([
                shutil.which('gfortran'), str(TEMPLATE.parent / 'Source/timing.f'), '-O2', '-std=legacy', '-fcheck=all',
                '-ffixed-line-length-none', '-I', str(work),
                str(ROOT / 'tests/input_files/check_fxfx_fixed_order.f90'),
                str(work / 'driver.f'), str(work / 'fxfx.f'),
                str(work / 'noops.f90'), '-o', str(executable)],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS fixed-order FxFx counterevent scales', result.stdout)
