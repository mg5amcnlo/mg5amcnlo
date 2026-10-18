"""Check SudGen's nominal probabilities with a real Pythia installation.

Set PYTHIA8 to an MC@NLO-Delta-compatible Pythia installation to run these
numerical integration checks. They are skipped when Pythia is unavailable.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(os.environ.get('PYTHIA8') and shutil.which('g++'),
                     'requires g++ and a Pythia installation in PYTHIA8')
class TestSudGen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pythia = Path(os.environ['PYTHIA8']).resolve()
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_sudgen_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        executable = Path(cls.tempdir.name)/'check_sudgen'
        result = subprocess.run([
            shutil.which('g++'), '-O2', '-std=c++11',
            '-I'+str(pythia/'include'),
            str(ROOT/'tests/input_files/check_sudgen.cc'),
            str(ROOT/'vendor/SudGen/pythia_interface_for_sudgen.cc'),
            '-L'+str(pythia/'lib'), '-Wl,-rpath,'+str(pythia/'lib'),
            '-lpythia8', '-ldl', '-o', str(executable)],
            capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
        result = subprocess.run([str(executable)], cwd=cls.tempdir.name,
            env=dict(os.environ, PYTHIA8DATA=str(pythia/'share/Pythia8/xmldoc')),
            capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
        cls.probabilities = {}
        cls.variations = None
        cls.explicit_x = None
        for line in result.stdout.splitlines():
            if line.startswith('VARIATIONS '):
                cls.variations = tuple(map(int, line.split()[1:]))
            elif line.startswith('PROBABILITY '):
                _, kind, stop, probability = line.split()
                cls.probabilities[int(kind), round(float(stop), 6)] = float(probability)
            elif line.startswith('EXPLICIT_X '):
                cls.explicit_x = float(line.split()[1])

    def test_nominal_initial_and_final_state_probabilities(self):
        self.assertEqual(self.variations, (0, 0, 0))
        for kind in (1, 2):
            with self.subTest(dipole_type=kind):
                low = self.probabilities[kind, 20.]
                high = self.probabilities[kind, 23.]
                # Variations bias these short-interval probabilities strongly.
                self.assertGreater(low, .90 if kind == 1 else .80)
                self.assertLess(low, high)
                self.assertLess(high, 1.)
                self.assertEqual(self.probabilities[kind, 80.419002], 1.)

    def test_both_fortran_entry_points_agree(self):
        self.assertIsNotNone(self.explicit_x)
        self.assertEqual(self.explicit_x, self.probabilities[1, 20.])


if __name__ == '__main__':
    unittest.main()
