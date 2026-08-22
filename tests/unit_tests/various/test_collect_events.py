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

        pid = os.fork()
        if pid == 0:
            status = 1
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
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
