"""Delayed exact loops preserve native orders, PDF records and fold context."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestDeferredVirtual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='mg5_deferred_virtual_')
        cls.addClassCleanup(cls.tmp.cleanup)
        root = Path(cls.tmp.name)
        work = root / 'SubProcesses' / 'P0'
        work.mkdir(parents=True)
        model = root / 'Source' / 'MODEL'
        model.mkdir(parents=True)
        (model / 'input.inc').write_text(
            '      double precision mu_r\n      common/test_mu_r/mu_r\n')
        includes = {
            'nexternal.inc': 'integer nexternal\nparameter(nexternal=5)',
            'orders.inc': 'integer amp_split_size\nparameter(amp_split_size=3)',
            'coupl.inc': 'double precision g\ncommon/test_g/g',
            'run.inc': 'double precision scale\ncommon/test_scale/scale',
            'q_es.inc': 'double precision QES2\ncommon/coupl_es/QES2',
        }
        for name, contents in includes.items():
            (work / name).write_text(''.join('      '+line+'\n'
                                           for line in contents.splitlines()))
        shutil.copyfile(TEMPLATE / 'timing_variables.inc', work / 'timing_variables.inc')
        # Extract the actual implementation rather than duplicating its
        # conversion logic in a test-only helper.
        routine = fortran_routine(TEMPLATE / 'fks_singular.f',
                                  'deferred_virtual_context')
        (work / 'callback.f').write_text(routine)
        cls.executable = work / 'check_deferred'
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-fcheck=all', '-fno-automatic',
            '-std=legacy', '-ffixed-line-length-none', '-I', str(work),
            str(TEMPLATE.parent / 'Source/timing.f'),
            str(TEMPLATE / 'weight_lines.f'),
            str(ROOT / 'tests/input_files/check_deferred_virtual.f90'),
            str(work / 'callback.f'), '-o', str(cls.executable)],
            cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def check_mode(self, mode):
        result = subprocess.run([str(self.executable), mode],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS '+mode, result.stdout)

    def test_mixed_orders_rescale_raw_central_and_flavour_weights(self):
        self.check_mode('mixed_orders')

    def test_zero_signed_born_sum_preserves_nonzero_order_residual(self):
        self.check_mode('zero_signed_born')

    def test_selected_helicity_subtraction_differs_from_saved_mean(self):
        self.check_mode('conditional_selected')

    def test_veto_avoids_loop_and_clears_integration_records(self):
        self.check_mode('veto')

    def test_inactive_callback_keeps_records_and_context(self):
        self.check_mode('inactive')
