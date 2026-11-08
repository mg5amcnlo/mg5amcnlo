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


class TestMultiCoreReuse(unittest.TestCase):
    def check_reuse(self, failure):
        # Keep a regression in worker shutdown or queue accounting bounded.
        script = textwrap.dedent('''\
            import pathlib
            import sys
            import tempfile
            import threading
            from madgraph.various.cluster import MultiCore

            failure, keep, cores = sys.argv[1], bool(int(sys.argv[2])), int(sys.argv[3])
            pool = MultiCore(cores, keep_thread=keep)
            gates = [threading.Event() for _ in range(cores)]
            started = [threading.Event() for _ in range(cores)]
            forbidden = []
            completed = []
            simultaneous = threading.Barrier(cores + 1)

            def occupy(index):
                started[index].set()
                assert gates[index].wait(5), 'worker release timed out'

            def fail():
                if failure == 'dual':
                    simultaneous.wait(5)
                if failure == 'system_exit':
                    raise SystemExit('reuse failure sentinel')
                if failure == 'keyboard_interrupt':
                    raise KeyboardInterrupt('reuse failure sentinel')
                raise RuntimeError('reuse failure sentinel')

            def clean_batch():
                assert pool.fail_msg is None, 'previous failure survived cleanup'
                assert pool.nb_done == 0, 'previous completion count survived cleanup'
                assert not pool.id_to_packet, 'previous packet mappings survived cleanup'
                for name in ('queue', 'done', 'submitted', 'pids', 'done_pid_queue'):
                    q = getattr(pool, name)
                    assert q.empty(), name + ' retained previous work'
                    assert q.unfinished_tasks == 0, name + ' retained unfinished tasks'

            # Occupy every worker before submitting the failure and cancelled
            # work. With two cores the second callable stays active until the
            # failure occurs, exercising cleanup of a still-running worker.
            for index in range(cores):
                pool.submit(occupy, [index])
            for event in started:
                assert event.wait(5), 'worker did not start'
            workers = list(pool.demons)
            old_queue = pool.queue
            with tempfile.TemporaryDirectory() as temp:
                if failure in ('callable', 'dual', 'system_exit', 'keyboard_interrupt'):
                    for _ in range(cores if failure == 'dual' else 1):
                        pool.submit(fail)
                    expected = 'reuse failure sentinel'
                elif failure == 'executable':
                    pool.submit(sys.executable, ['-c', 'import sys; sys.exit(17)'])
                    expected = 'non zero status: 17'
                else:
                    pool.submit(str(pathlib.Path(temp) / 'missing-executable'))
                    expected = 'missing-executable'
                for index in range(3):
                    pool.submit(forbidden.append, [index])

                if failure == 'dual':
                    # Both workers must be executing their failing callable
                    # before either failure can cancel the other queued work.
                    for gate in gates:
                        gate.set()
                    simultaneous.wait(5)
                else:
                    gates[0].set()
                    assert pool.stoprequest.wait(5), 'failure was not detected'
                    # A parent may still submit work before it calls wait().
                    # That must not clear the worker failure or restart work.
                    pool.submit(forbidden.append, ['late submission'])
                    assert pool.stoprequest.is_set(), 'submission cleared failure'
                    for gate in gates[1:]:
                        gate.set()
                try:
                    pool.wait(None, lambda *args: None)
                except BaseException as error:
                    assert expected in str(error), str(error)
                    if failure == 'system_exit':
                        assert isinstance(error, SystemExit), type(error)
                    elif failure == 'keyboard_interrupt':
                        assert isinstance(error, KeyboardInterrupt), type(error)
                else:
                    raise AssertionError('original failure was not reported')
                assert all(not worker.is_alive() for worker in workers), 'failed workers still running'
                assert old_queue.unfinished_tasks == 0, 'aborted queue was not drained'
                assert not forbidden, 'cancelled work ran after failure'
                clean_batch()
                # Existing launchers call remove() again in their error handler.
                pool.remove()

                # The same object must support subsequent successful batches,
                # including the keep-thread path across a second wait().
                for batch in range(3):
                    if batch == 2:
                        pool.remove()  # stopping an idle successful pool is safe too
                    for index in range(3):
                        pool.submit(completed.append, [3 * batch + index])
                    pool.wait(None, lambda *args: None)
                    assert sorted(completed) == list(range(3 * (batch + 1))), completed
                    assert not forbidden, 'cancelled work leaked into a new batch'
                    clean_batch()
                pool.remove()
            print('PASS failure and pool reuse')
        ''')
        for keep_thread in (False, True):
            for cores in ((2,) if failure == 'dual' else (1, 2)):
                with self.subTest(failure=failure, keep_thread=keep_thread, cores=cores):
                    result = subprocess.run(
                        [sys.executable, '-c', script, failure,
                         str(int(keep_thread)), str(cores)],
                        cwd=Path(__file__).resolve().parents[3],
                        capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('PASS failure and pool reuse', result.stdout)

    def test_callable_failure_then_pool_reuse(self):
        self.check_reuse('callable')

    def test_executable_failure_then_pool_reuse(self):
        self.check_reuse('executable')

    def test_launch_failure_then_pool_reuse(self):
        self.check_reuse('launch')

    def test_simultaneous_failures_then_pool_reuse(self):
        self.check_reuse('dual')

    def test_system_exit_then_pool_reuse(self):
        self.check_reuse('system_exit')

    def test_keyboard_interrupt_then_pool_reuse(self):
        self.check_reuse('keyboard_interrupt')

    def test_packet_failure_or_cancellation_then_pool_reuse(self):
        script = textwrap.dedent('''\
            import pathlib
            import sys
            import tempfile
            import threading
            from madgraph.various.cluster import MultiCore, Packet

            fail, keep = bool(int(sys.argv[1])), bool(int(sys.argv[2]))
            pool = MultiCore(1, keep_thread=keep)
            gate = threading.Event()
            started = threading.Event()
            callbacks = []
            packet = Packet('reuse packet', callbacks.append, ['completed'])

            def occupy():
                started.set()
                assert gate.wait(5), 'worker release timed out'

            with tempfile.TemporaryDirectory() as temp:
                forbidden = pathlib.Path(temp) / 'cancelled-job-ran'
                pool.submit(occupy)
                assert started.wait(5), 'worker did not start'
                workers = list(pool.demons)
                old_queue = pool.queue
                pool.cluster_submit(sys.executable,
                    ['-c', 'import sys; sys.exit(19)' if fail else 'pass'],
                    packet_member=packet)
                pool.cluster_submit(sys.executable,
                    ['-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).touch()',
                     str(forbidden)], packet_member=packet)
                assert packet.queue.unfinished_tasks == 2
                if not fail:
                    pool.remove()
                gate.set()
                try:
                    pool.wait(None, lambda *args: None)
                except Exception as error:
                    assert fail and 'non zero status: 19' in str(error), str(error)
                else:
                    assert not fail, 'packet failure was not reported'
                assert all(not worker.is_alive() for worker in workers)
                assert old_queue.unfinished_tasks == 0
                assert packet.queue.empty()
                assert packet.queue.unfinished_tasks == 0
                assert not callbacks, 'cancelled packet callback ran'
                assert not forbidden.exists(), 'cancelled packet job ran'
                assert not pool.id_to_packet

                # Reuse the same Packet object through the public submission
                # API: stale members would prevent its callback from running.
                for index in range(2):
                    pool.cluster_submit(sys.executable,
                        ['-c', 'pass', str(index)], packet_member=packet)
                pool.wait(None, lambda *args: None)
                assert callbacks == ['completed'], callbacks
                assert packet.queue.empty()
                assert packet.queue.unfinished_tasks == 0
                assert pool.queue.unfinished_tasks == 0
                assert not pool.id_to_packet
                pool.remove()
                pool.wait(None, lambda *args: None)
            print('PASS packet cleanup and reuse')
        ''')
        for failure in (False, True):
            for keep_thread in (False, True):
                with self.subTest(failure=failure, keep_thread=keep_thread):
                    result = subprocess.run(
                        [sys.executable, '-c', script,
                         str(int(failure)), str(int(keep_thread))],
                        cwd=Path(__file__).resolve().parents[3],
                        capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('PASS packet cleanup and reuse', result.stdout)
