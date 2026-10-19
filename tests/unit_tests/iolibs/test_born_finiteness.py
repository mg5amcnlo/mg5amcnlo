"""Born guards retain NaN/Inf rejection without changing floating-point state."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestBornFiniteness(unittest.TestCase):
    def check_finiteness(self, standard=False):
        with tempfile.TemporaryDirectory(prefix='mg5_born_finite_') as tmp:
            work = Path(tmp)
            source = (ROOT / 'Template/NLO/Source/BornSupport/mc_born_types.f90').read_text()
            if standard:
                # Exercise the portable fallback even on a binary64 host.
                source = source.replace('if (binary64) then', 'if (.false.) then')
            (work / 'mc_born_types.f90').write_text(source)
            flags = [] if standard else ['-ffpe-trap=invalid,zero,overflow']
            result = subprocess.run([
                'gfortran', '-O', '-fno-automatic', '-fcheck=all', *flags,
                'mc_born_types.f90',
                str(ROOT / 'tests/input_files/check_born_finiteness.f90'),
                '-o', 'check'], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([
                str(work / 'check'), 'standard' if standard else 'binary64'],
                cwd=work, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS Born finiteness', result.stdout)

    def test_binary64_classification_preserves_flags_and_rounding(self):
        self.check_finiteness()

    def test_portable_fallback(self):
        self.check_finiteness(standard=True)
