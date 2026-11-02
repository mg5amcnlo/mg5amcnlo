"""Exercise the test-manager CLI's process status, including optional skips."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


class TestTestManagerExitStatus(unittest.TestCase):
    def test_cli_exit_status_matches_unittest_result(self):
        # Input fixtures are excluded from ordinary test discovery. Run this
        # package explicitly so its deliberate failure/error stay isolated.
        with tempfile.TemporaryDirectory(prefix='manager_cases_',
                                         dir=ROOT / 'tests/input_files') as tmp:
            fixture = Path(tmp)
            (fixture / '__init__.py').write_text('')
            (fixture / 'cases.py').write_text('''import unittest

class Cases(unittest.TestCase):
    def test_pass(self):
        pass

    @unittest.skip('optional dependency absent')
    def test_skip(self):
        pass

    def test_fail(self):
        self.fail('deliberate fixture failure')

    def test_error(self):
        raise RuntimeError('deliberate fixture error')

    @unittest.expectedFailure
    def test_expected_failure(self):
        self.fail('expected fixture failure')

    @unittest.expectedFailure
    def test_unexpected_success(self):
        pass
''')
            # Execute the real CLI in a child process. Replace desktop
            # notification delivery while retaining both CLI result paths.
            launch = '''import runpy, sys
import madgraph.various.misc as misc
misc.system_notify = lambda *args: print('FIXTURE NOTIFICATION')
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
            cases = [
                (['test_pass'], 0, False),
                (['test_skip'], 0, True),
                (['test_pass', 'test_skip'], 0, True),
                (['test_fail'], 1, False),
                (['test_error'], 1, False),
                (['test_expected_failure'], 0, False),
                (['test_unexpected_success'], 1, False),
            ]
            for notify in (False, True):
                for names, expected, skipped in cases:
                    with self.subTest(tests=names, notification=notify):
                        result = subprocess.run([
                            sys.executable, '-c', launch,
                            str(ROOT / 'tests/test_manager.py'), *names,
                            '-p', str(fixture.relative_to(ROOT)), '-t0',
                            '-N', '1e-20' if notify else '-1', '--nocaffeinate',
                        ], cwd=ROOT, capture_output=True, text=True, timeout=30)
                        output = result.stdout + result.stderr
                        self.assertEqual(result.returncode, expected, output)
                        if skipped:
                            self.assertIn('OK (skipped=1)', output)
                        self.assertEqual('FIXTURE NOTIFICATION' in output, notify,
                                         output)
