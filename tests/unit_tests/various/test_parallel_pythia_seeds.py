"""Check seed isolation in the actual split-shower shell wrapper."""

from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


class TestParallelPythiaSeeds(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix='mg5_py8_seeds_')
        self.addCleanup(self.work.cleanup)
        self.directory = Path(self.work.name)
        template = (ROOT / 'Template/NLO/MCatNLO/shower_template.sh').read_text()
        (self.directory / 'shower.sh').write_text(template % {
            'ld_library_path': 'LD_LIBRARY_PATH', 'extralibs': ''})
        executable = self.directory / 'Pythia8.exe'
        executable.write_text('#!/usr/bin/env bash\n'
                              'cp "$1" captured.cmd\n'
                              'touch test.HwU\n')
        executable.chmod(0o755)

    def run_shower(self, seed, worker=None, prefix=''):
        command = prefix + 'Random:setSeed = on\nRandom:seed = %d ! seed\n' % seed
        (self.directory / 'Pythia8.cmd').write_text(command)
        arguments = ['bash', 'shower.sh', 'PYTHIA8', 'HWU', 'run_01']
        if worker is not None:
            (self.directory / ('events_%d.lhe' % worker)).write_text('events\n')
            arguments.append(str(worker))
            output = self.directory / ('run_%d' % worker) / 'captured.cmd'
        else:
            (self.directory / 'events.lhe').write_text('events\n')
            output = self.directory / 'captured.cmd'
        result = subprocess.run(arguments, cwd=self.directory,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.directory / 'Pythia8.cmd').read_text(), command)
        return output.read_text()

    def effective_seed(self, command):
        return int(re.findall(r'(?im)^\s*Random:seed\s*=\s*([+-]?\d+)',
                              command)[-1])

    def test_fixed_seeds_are_distinct_and_reproducible(self):
        self.assertEqual(self.effective_seed(self.run_shower(860101, 1)), 860101)
        self.assertEqual(self.effective_seed(self.run_shower(860101, 2)), 860102)
        self.assertEqual(self.effective_seed(self.run_shower(860101, 64)), 860164)

    def test_seed_wraps_within_pythia_range(self):
        self.assertEqual(self.effective_seed(self.run_shower(900000000, 2)), 1)

    def test_serial_seed_is_unchanged(self):
        self.assertEqual(self.effective_seed(self.run_shower(860101)), 860101)

    def test_automatic_seed_is_unchanged(self):
        self.assertEqual(self.effective_seed(self.run_shower(0, 2)), 0)

    def test_only_last_seed_assignment_is_adjusted(self):
        command = self.run_shower(860101, 2, 'random:seed = 7\n')
        self.assertIn('random:seed = 7\n', command)
        self.assertEqual(self.effective_seed(command), 860102)


if __name__ == '__main__':
    unittest.main()
