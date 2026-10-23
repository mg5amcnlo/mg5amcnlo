"""FxFx labels must survive Delta's independent particle production scales."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT


@unittest.skipUnless(shutil.which('g++'), 'requires g++')
class TestFxFxDeltaHook(unittest.TestCase):
    def test_matching_flags_do_not_change_shower_scales(self):
        with tempfile.TemporaryDirectory(prefix='mg5_fxfx_delta_hook_') as tmp:
            executable = Path(tmp) / 'check_hook'
            result = subprocess.run([
                shutil.which('g++'), '-std=c++11', '-Wall', '-Wextra', '-pedantic',
                '-I', str(ROOT / 'Template/NLO/MCatNLO/include'),
                str(ROOT / 'tests/input_files/check_fxfx_delta_hook.cc'),
                '-o', str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS FxFx Delta matching flags', result.stdout)
