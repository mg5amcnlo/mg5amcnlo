"""Multicore job failures must reach the caller even with queued work."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class TestMultiCoreFailure(unittest.TestCase):
    def check_failure(self, executable):
        # Bound the test itself: the regression leaves wait() asleep forever.
        script = textwrap.dedent('''\
            import sys
            import threading
            from madgraph.various.cluster import MultiCore

            pool = MultiCore(1)
            ready = threading.Event()
            def fail():
                ready.wait()
                raise RuntimeError('worker regression sentinel')
            def queued():
                raise AssertionError('queued job ran after failure')

            if int(sys.argv[1]):
                pool.submit(sys.executable, ['-c',
                    'import time, sys; time.sleep(0.2); sys.exit(17)'])
                expected = 'non zero status: 17'
            else:
                pool.submit(fail)
                expected = 'worker regression sentinel'
            pool.submit(queued)
            ready.set()
            try:
                pool.wait(None, lambda *args: None,
                          update_first=lambda *args: None)
            except Exception as error:
                assert expected in str(error), str(error)
            else:
                raise AssertionError('worker failure was not reported')
            print('PASS')
        ''')
        result = subprocess.run([sys.executable, '-c', script, str(int(executable))],
                                cwd=Path(__file__).resolve().parents[3],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS', result.stdout)

    def test_callable_failure_with_queued_work(self):
        self.check_failure(False)

    def test_executable_failure_with_queued_work(self):
        self.check_failure(True)
