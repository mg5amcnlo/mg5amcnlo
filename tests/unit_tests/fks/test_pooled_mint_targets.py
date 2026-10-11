"""The pooled control must not alter targets trained by ordinary MINT grids."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestPooledMintTargets(unittest.TestCase):
    def test_original_marginal_targets_and_sampled_zero_residuals(self):
        with tempfile.TemporaryDirectory(prefix='mg5_pooled_mint_') as tmp:
            work = Path(tmp)
            production = (TEMPLATE / 'mint_module.f90').read_text()
            routine = re.search(
                r'^  subroutine add_point_to_grids\(x\).*?^  end subroutine add_point_to_grids',
                production, re.M | re.S).group(0)
            (work / 'pooled_mint_routine.inc').write_text(routine+'\n')
            executable = work / 'check_pooled_mint'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fno-automatic', '-fcheck=all',
                '-ffpe-trap=invalid,zero,overflow', '-I', str(work),
                str(ROOT / 'tests/input_files/check_pooled_mint.f90'),
                '-o', str(executable)], cwd=work, capture_output=True,
                text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], cwd=work,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS pooled MINT', result.stdout)
