"""Production S/H scale plumbing for leptons, W bosons and photon conversion."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestQEDShowerScales(unittest.TestCase):
    def test_colourless_born_and_real_connections(self):
        with tempfile.TemporaryDirectory(prefix='mg5_qed_scales_') as tmp:
            work = Path(tmp)
            # The full geometry helper module also declares export dimensions.
            (work / 'nexternal.inc').write_text(
                '      integer nexternal,nincoming\n'
                '      parameter(nexternal=5,nincoming=2)\n')
            shutil.copyfile(TEMPLATE / 'fks_powers.inc', work / 'fks_powers.inc')
            executable = work / 'check'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-std=legacy',
                '-ffixed-line-length-none', '-fcheck=all', '-fno-automatic',
                '-ffpe-trap=invalid,zero,overflow', '-I', str(work),
                '-ffunction-sections', '-fdata-sections',
                '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
                str(TEMPLATE / 'qed_shower_support.f90'),
                str(TEMPLATE / 'process_module.f90'),
                str(TEMPLATE / 'fks_phase_space_data.f'),
                str(TEMPLATE / 'genps_fks_helpers.f'),
                str(TEMPLATE / 'mcatnlo_delta_scales.f90'),
                str(TEMPLATE / 'herwig7_scales.f90'),
                str(TEMPLATE / 'scale_module.f90'),
                str(ROOT / 'tests/input_files/check_qed_shower_scales.f90'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS QED S/H scales', result.stdout)
