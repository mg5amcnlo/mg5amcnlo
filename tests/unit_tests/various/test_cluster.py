################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Unit tests for madgraph.various.cluster"""

import threading
from pathlib import Path
import subprocess
import sys
import textwrap

import madgraph.various.cluster as cluster
import tests.unit_tests as unittest


class TestMultiCore(unittest.TestCase):
    """Test the MultiCore cluster"""

    def test_failure_with_queued_jobs(self):
        """a failing job stops wait(), even with jobs still in the queue"""

        mycluster = cluster.MultiCore(nb_core=1)
        mycluster.submit('/bin/sh', ['-c', 'exit 1'])
        for i in range(2):
            mycluster.submit('/bin/sh', ['-c', 'sleep 30'])

        raised = []
        def wait():
            try:
                mycluster.wait('.', lambda idle, run, done: None)
            except Exception as error:
                raised.append(error)

        # wait() used to wait forever for the queued jobs, which the stopped
        # workers never start: do not let that block the test suite
        thread = threading.Thread(target=wait, daemon=True)
        thread.start()
        thread.join(60)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(raised), 1)
        self.assertIn('non zero status', str(raised[0]))


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
