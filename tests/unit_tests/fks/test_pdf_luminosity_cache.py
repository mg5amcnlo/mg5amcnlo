"""Reuse central luminosities without losing native flavour or PDF state."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestPDFLuminosityCache(unittest.TestCase):
    def test_exact_keys_common_state_and_flavour_growth(self):
        with tempfile.TemporaryDirectory(prefix='mg5_pdf_luminosity_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=5,nincoming=2)',
                'genps.inc': 'integer maxproc\nparameter(maxproc=3)',
                'run.inc': 'double precision xbk(2),q2fact(2)\n'
                           'integer lpp(2),flavour_bias(2)\n'
                           'common/test_pdf_run/xbk,q2fact,lpp,flavour_bias',
                'orders.inc': 'integer nsplitorders,amp_split_size,qcd_pos\n'
                              'parameter(nsplitorders=1,amp_split_size=1,qcd_pos=1)',
                'coupl.inc': '',
            }
            for name, text in includes.items():
                (work / name).write_text(''.join('      '+s+'\n' for s in text.splitlines()))
            for name in ('timing_variables.inc', 'q_es.inc'):
                shutil.copyfile(TEMPLATE / name, work / name)
            (work / 'modules.f90').write_text(
                'module mint_module\n'
                'double precision :: virt_wgt_mint(0:1)=0d0,born_wgt_mint(0:1)=0d0\n'
                'end module\n'
                'module FKSParams\nlogical :: separate_flavour_configs=.false.\nend module\n')
            (work / 'weights.f').write_text('\n'.join(
                fortran_routine(TEMPLATE / 'fks_singular.f', name)
                for name in ('include_PDF_and_alphas', 'separate_flavour_config')))
            executable = work / 'check_pdf_cache'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
                '-ffixed-line-length-none', '-I', str(work),
                str(work / 'modules.f90'), str(TEMPLATE / 'weight_lines.f'),
                str(ROOT / 'Template/NLO/Source/extra_weights.f'),
                str(work / 'weights.f'),
                str(ROOT / 'tests/input_files/check_pdf_luminosity_cache.f90'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS PDF luminosity cache', result.stdout)
