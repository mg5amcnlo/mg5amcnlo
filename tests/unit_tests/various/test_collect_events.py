################################################################################
#
# Copyright (c) 2012 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Test the collection/shuffling of events from several LHE files."""

from __future__ import absolute_import
import collections
import gzip
import os
import resource
import shutil
import tempfile
import threading
import time
import traceback
import unittest

import madgraph.various.collect_events as collect_events

pjoin = os.path.join


def event_multiset(text):
    """The multiset of <event> bodies found in an LHE text."""
    return collections.Counter(chunk.split('</event>')[0]
                               for chunk in text.split('<event>')[1:])


class TestCollectEvents(unittest.TestCase):
    """Check that events are collected verbatim and without leaking
    file descriptors."""

    nb_files = 40
    nb_events = 25
    tight_limit = 128

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='collect_events_test_')
        self.inputs = []
        for idx in range(self.nb_files):
            path = pjoin(self.tmpdir, 'in%03d.lhe' % idx)
            events = ''.join(
                '<event>\n 5 %d 0.1 91.2 0.0078 0.118\n body %d %d\n</event>\n'
                % (idx, idx, iev) for iev in range(self.nb_events))
            with open(path, 'w') as fsock:
                fsock.write('<LesHouchesEvents version="3.0">\n'
                            '<header>\n<MGVersion>3.7.3</MGVersion>\n'
                            '<MGRunCard>\n  0 = iseed\n</MGRunCard>\n</header>\n'
                            '<init>\n 2212 2212 6.5e3 6.5e3\n</init>\n'
                            + events + '</LesHouchesEvents>\n')
            self.inputs.append(path)

        self.banner = pjoin(self.tmpdir, 'banner.txt')
        with open(self.banner, 'w') as fsock:
            fsock.write('<LesHouchesEvents version="3.0">\n'
                        '<header>\n<MGVersion>3.7.3</MGVersion>\n'
                        '<MGRunCard>\n  0 = iseed\n</MGRunCard>\n'
                        '<slha>BLOCK MASS</slha>\n</header>\n'
                        '</LesHouchesEvents>\n')

        self.expected = collections.Counter()
        for path in self.inputs:
            with open(path) as fsock:
                self.expected.update(event_multiset(fsock.read()))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def collect(self, output, workers=1, mode='memory', seed=42):
        collect_events.collect_events(
            output=pjoin(self.tmpdir, output),
            header_template=self.banner,
            template_header_keep_tags=['MGVersion', 'MGRunCard', 'slha'],
            input_files=self.inputs,
            seed=seed,
            subset=None,
            workers=workers,
            mode=mode,
            prefer_pigz=False,
            gzip_level=6,
            verbose=False)
        path = pjoin(self.tmpdir, output)
        if output.endswith('.gz'):
            with gzip.open(path, 'rt') as fsock:
                return fsock.read()
        with open(path) as fsock:
            return fsock.read()

    def test_events_are_conserved(self):
        """no event is lost or duplicated, whatever the writing strategy"""
        for name, kwargs in [('single.lhe', {'workers': 1}),
                             ('threaded.lhe', {'workers': 18}),
                             ('zipped.lhe.gz', {'workers': 18}),
                             ('external.lhe', {'mode': 'external'})]:
            text = self.collect(name, **kwargs)
            self.assertEqual(event_multiset(text), self.expected)
            self.assertIn('<init>', text)
            self.assertIn('<MGVersion>', text)
            self.assertTrue(text.rstrip().endswith('</LesHouchesEvents>'))

    def test_worker_count_does_not_change_output(self):
        """the shuffle is set by the seed only, not by the thread count"""
        self.assertEqual(self.collect('w1.lhe', workers=1),
                         self.collect('w18.lhe', workers=18))

    def test_low_file_descriptor_limit(self):
        """many workers over many files must not exhaust RLIMIT_NOFILE.

        macOS defaults to a soft limit of 256, which a per-worker cache of
        open input files exceeds as soon as the machine has enough cores.
        The check runs in a forked child with both the soft and the hard
        limit lowered, so the writer has to stay within budget rather than
        raise the soft limit out of the way. Lowering the hard limit is
        irreversible, hence the fork.
        """
        if not hasattr(os, 'fork'):
            raise unittest.SkipTest('no fork available on this platform')
        hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
        if hard != resource.RLIM_INFINITY and hard < self.tight_limit:
            raise unittest.SkipTest('inherited hard limit is already below %d'
                                    % self.tight_limit)

        pid = os.fork()
        if pid == 0:
            status = 1
            try:
                resource.setrlimit(
                    resource.RLIMIT_NOFILE, (self.tight_limit, self.tight_limit))
                text = self.collect('tight.lhe', workers=18)
                status = 0 if event_multiset(text) == self.expected else 2
            except BaseException:
                traceback.print_exc()
            finally:
                os._exit(status)

        status = os.waitpid(pid, 0)[1]
        self.assertTrue(os.WIFEXITED(status),
                        'event writer died on a low descriptor limit')
        self.assertEqual(os.WEXITSTATUS(status), 0,
                         'event writer failed on a low descriptor limit')

    def test_pool_respects_its_cap(self):
        """concurrent readers must not push the pool past max_open.

        Evicting only unpinned entries is not enough on its own: if every
        pooled entry is pinned by a concurrent read, opening anyway would
        make the cap advisory and let the pool grow with the worker count.
        """
        cap, nb_readers = 4, 32
        pool = collect_events.FDPool(self.inputs, max_open=cap)
        peak = [0]
        peak_lock = threading.Lock()
        original = collect_events._pread_exact

        def slow_pread(fd, offset, size):
            time.sleep(0.02)          # hold the pin long enough to overlap
            with peak_lock:
                peak[0] = max(peak[0], len(pool._open))
            return original(fd, offset, size)

        start = threading.Barrier(nb_readers)

        def reader(idx):
            start.wait()
            pool.read(idx % len(self.inputs), 0, 32)

        collect_events._pread_exact = slow_pread
        try:
            threads = [threading.Thread(target=reader, args=(i,))
                       for i in range(nb_readers)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=60)
                self.assertFalse(thread.is_alive(), 'FDPool deadlocked')
        finally:
            collect_events._pread_exact = original
            pool.close()

        self.assertLessEqual(peak[0], cap)

    def test_external_merge_is_bounded(self):
        """many shuffle runs must not exhaust the descriptor limit.

        The final k-way merge opens every run it is handed, so a run
        capacity of 5 records (200 runs for this input) overruns a limit of
        128 on exactly the large jobs this path exists for.
        """
        if not hasattr(os, 'fork'):
            raise unittest.SkipTest('no fork available on this platform')
        hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
        if hard != resource.RLIM_INFINITY and hard < self.tight_limit:
            raise unittest.SkipTest('inherited hard limit is already below %d'
                                    % self.tight_limit)

        pid = os.fork()
        if pid == 0:
            status = 1
            try:
                resource.setrlimit(
                    resource.RLIMIT_NOFILE, (self.tight_limit, self.tight_limit))
                collect_events.collect_events(
                    output=pjoin(self.tmpdir, 'runs.lhe'),
                    header_template=self.banner,
                    template_header_keep_tags=['MGVersion'],
                    input_files=self.inputs, seed=42, subset=None, workers=1,
                    mode='external', external_run_capacity=5,
                    prefer_pigz=False, gzip_level=6, verbose=False)
                with open(pjoin(self.tmpdir, 'runs.lhe')) as fsock:
                    same = event_multiset(fsock.read()) == self.expected
                status = 0 if same else 2
            except BaseException:
                traceback.print_exc()
            finally:
                os._exit(status)

        status = os.waitpid(pid, 0)[1]
        self.assertTrue(os.WIFEXITED(status),
                        'external merge died on a low descriptor limit')
        self.assertEqual(os.WEXITSTATUS(status), 0,
                         'external merge failed on a low descriptor limit')

    def test_merge_fan_in_does_not_change_output(self):
        """merging runs in several passes keeps the shuffled order"""
        outputs = []
        original = collect_events._reduce_run_paths
        try:
            for fan_in in (4, 8, 10 ** 6):
                collect_events._reduce_run_paths = (
                    lambda paths, tmp, fan_in=None, verbose=False, _f=fan_in:
                    original(paths, tmp, fan_in=_f, verbose=False))
                collect_events.collect_events(
                    output=pjoin(self.tmpdir, 'fan%d.lhe' % fan_in),
                    header_template=self.banner,
                    template_header_keep_tags=['MGVersion'],
                    input_files=self.inputs, seed=7, subset=None, workers=1,
                    mode='external', external_run_capacity=20,
                    prefer_pigz=False, gzip_level=6, verbose=False)
                with open(pjoin(self.tmpdir, 'fan%d.lhe' % fan_in)) as fsock:
                    outputs.append(fsock.read())
        finally:
            collect_events._reduce_run_paths = original

        for text in outputs[1:]:
            self.assertEqual(text, outputs[0])
        self.assertEqual(event_multiset(outputs[0]), self.expected)

    def test_short_read_is_reported(self):
        """a truncated read must fail loudly, not write a partial event"""
        path = pjoin(self.tmpdir, 'in000.lhe')
        size = os.path.getsize(path)
        pool = collect_events.FDPool([path], max_open=2)
        try:
            self.assertRaises(RuntimeError, pool.read, 0, size - 4, size + 64)
        finally:
            pool.close()
