"""Exact cancellation and weighted observations in the pooled virt-trick."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestPooledVirtualWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='mg5_pooled_weights_')
        cls.addClassCleanup(cls.tmp.cleanup)
        work = Path(cls.tmp.name)
        includes = {
            'nexternal.inc': 'integer nexternal,nincoming\n'
                             'parameter(nexternal=5,nincoming=2)',
            'genps.inc': 'integer maxproc\nparameter(maxproc=2)',
            'run.inc': 'double precision xbk(2),q2fact(2)\n'
                       'integer lpp(2),flavour_bias(2)\n'
                       'common/test_pdf_run/xbk,q2fact,lpp,flavour_bias',
            'orders.inc': 'integer nsplitorders,amp_split_size,qcd_pos,qed_pos\n'
                          'parameter(nsplitorders=2,amp_split_size=3,qcd_pos=1,qed_pos=2)',
            'coupl.inc': '',
        }
        for name, text in includes.items():
            (work / name).write_text(''.join('      '+line+'\n' for line in text.splitlines()))
        for name in ('timing_variables.inc', 'q_es.inc'):
            shutil.copyfile(TEMPLATE / name, work / name)

        source = (TEMPLATE / 'fks_singular.f').read_text()
        # Compile the actual virtual stage, with matrix elements supplied by
        # the fixture. Other parts of bornsoftvirtual require generated
        # process-specific integrated subtraction and loop-library code.
        virtual = source.split(' 548  continue\n', 1)[1].split('c eq.(MadFKS.C.13)', 1)[0]
        tracking = source.split('      virt_wgt_mint(0)=virt_wgt_mint(0)+virt_wgt*f_nb', 1)[1]
        tracking = ('      virt_wgt_mint(0)=virt_wgt_mint(0)+virt_wgt*f_nb' +
                    tracking.split('C This is the counterterm for the 6f->5f scheme change', 1)[0])
        wrappers = (
            '      subroutine exercise_virtual_stage\n'
            '      use pooled_weight_fixture\n'
            '      implicit none\n' + virtual + '      end\n' +
            '      subroutine exercise_virtual_tracking\n'
            '      use pooled_weight_fixture\n'
            '      use extra_weights\n'
            '      implicit none\n' + tracking + '      end\n')
        routines = ''.join(fortran_routine(TEMPLATE / 'fks_singular.f', name)
                           for name in ('find_virtual_mint_carrier',
                                        'include_PDF_and_alphas', 'include_bias_wgt',
                                        'recompute_xlum_for_wgt_mint'))
        (work / 'routines.f').write_text(wrappers + routines)
        cls.executable = work / 'check_weights'
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-fcheck=all', '-fno-automatic',
            '-std=legacy', '-ffixed-line-length-none', '-I', str(work),
            str(TEMPLATE / 'weight_lines.f'),
            str(ROOT / 'Template/NLO/Source/extra_weights.f'),
            str(ROOT / 'tests/input_files/pooled_virtual_weight_context.f90'),
            str(work / 'routines.f'),
            str(ROOT / 'tests/input_files/check_pooled_virtual_weights.f90'),
            '-o', str(cls.executable)], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def check_mode(self, mode):
        result = subprocess.run([str(self.executable), mode], capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS '+mode, result.stdout)

    def test_pooled_exact_cancellation_and_frozen_fold_ratio(self):
        self.check_mode('pooled_cancellation')

    def test_conditional_mean_and_selected_subtraction_reconstruct_sample(self):
        self.check_mode('conditional_actual_subtraction')

    def test_pooled_subsampling_keeps_exact_mean(self):
        self.check_mode('pooled_subsampling')

    def test_pooled_zero_residual_keeps_pdf_and_bias_normalization(self):
        self.check_mode('pooled_normalization')

    def test_pooled_tail_uses_actual_probability_in_exact_correction(self):
        self.check_mode('pooled_tail_probability')

    def test_deferred_zero_carrier_falls_back_to_exact_virtual(self):
        self.check_mode('deferred_zero_carrier')

    def test_deferred_veto_xsec_retains_immediate_virtual(self):
        self.check_mode('deferred_veto_xsec')
