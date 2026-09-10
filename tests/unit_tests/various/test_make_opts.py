################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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
"""Concurrency tests for Source/make_opts.

``Source/make_opts`` is written by several code paths (madgraph/__init__, 
MadGraphCmd.__init__, set_fortran_compiler / set_cpp_compiler) and read by many
more: it is copied verbatim into every new output directory and then parsed by
``make``. ``MG5DIR/Template/LO/Source/make_opts`` in particular is shared by
every MG5aMC process on the machine, and MadSpin forks a worker per core, each
of which can be exporting a decay matrix element at the same moment.

If any of those writers truncates the file in place (which ``open(..,'w')`` and
therefore ``shutil.copy`` do), a concurrent reader can copy out an empty
make_opts. Nothing complains: an empty make_opts still *parses*, so make simply
keeps its builtins -- $(FC)=f77, no $(libext), no -ffixed-line-length-132 -- and
the build dies far away, as 'Missing exponent in real number' at column 72 of
aloha_functions.f. These tests pin the two properties that prevent that: every
write is atomic, and a make_opts that has lost its body is refused loudly.
"""

from __future__ import absolute_import

import itertools
import os
import shutil
import tempfile
import threading
import unittest

import madgraph.various.misc as misc
import madgraph.interface.common_run_interface as common_run_interface
from madgraph import MG5DIR, MadGraph5Error

pjoin = os.path.join

MIN_READS = 50   # reads that must happen *while* writes are in flight


def race(write, path, is_valid, nb_write=400):
    """Run ``write`` in one thread and read ``path`` in another, and report
    every read whose content ``is_valid`` rejects.

    The two threads meet at a barrier so neither can run to completion before
    the other starts, and the writer keeps going until the reader has managed
    MIN_READS reads: without that the writer could finish every iteration
    before the reader is first scheduled, and the test would pass having
    observed nothing at all. Returns (bad, reads).
    """
    bad = []
    reads = [0]
    start = threading.Barrier(2)
    stop = threading.Event()

    def writer():
        start.wait()
        try:
            for _ in range(nb_write):
                write()
            # Keep writing until the reader has really seen the file. Bounded,
            # so a reader that died on an exception cannot hang the suite.
            for _ in range(10 * nb_write):
                if reads[0] >= MIN_READS or not reader_thread.is_alive():
                    break
                write()
        finally:
            stop.set()

    def reader():
        start.wait()
        while not stop.is_set():
            reads[0] += 1
            try:
                with open(path) as fsock:
                    content = fsock.read()
            except (IOError, OSError):
                bad.append('missing')
                continue
            if not is_valid(content):
                bad.append(len(content))

    reader_thread = threading.Thread(target=reader)
    threads = [threading.Thread(target=writer), reader_thread]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if reads[0] < MIN_READS:
        raise AssertionError('the reader only managed %d reads: the race was '
                             'never exercised' % reads[0])
    return bad, reads[0]


class TestAtomicWrite(unittest.TestCase):
    """misc.atomic_write / misc.atomic_copy"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.source = pjoin(MG5DIR, 'Template', 'LO', 'Source', '.make_opts')
        with open(self.source) as fsock:
            self.reference = fsock.read()

    def test_atomic_write_replaces_content(self):
        path = pjoin(self.tmpdir, 'make_opts')
        misc.atomic_write(path, 'first\n')
        misc.atomic_write(path, 'second\n')
        with open(path) as fsock:
            self.assertEqual(fsock.read(), 'second\n')

    def test_atomic_write_leaves_no_temporary_behind(self):
        path = pjoin(self.tmpdir, 'make_opts')
        misc.atomic_write(path, 'x\n')
        self.assertEqual(os.listdir(self.tmpdir), ['make_opts'])

    def test_atomic_write_is_readable(self):
        """mkstemp creates 0600 files; make_opts has to stay world readable."""
        path = pjoin(self.tmpdir, 'make_opts')
        misc.atomic_write(path, 'x\n')
        self.assertTrue(os.stat(path).st_mode & 0o044)

    def test_atomic_copy_accepts_a_directory_destination(self):
        """same signature as shutil.copy, which is what it replaces"""
        dest = pjoin(self.tmpdir, 'Source')
        os.mkdir(dest)
        misc.atomic_copy(self.source, dest)
        with open(pjoin(dest, '.make_opts')) as fsock:
            self.assertEqual(fsock.read(), self.reference)

    def test_concurrent_reader_never_sees_a_partial_file(self):
        """The regression itself.

        With shutil.copy this fails immediately -- the reader observes hundreds
        of zero-length make_opts, which is exactly what gets copied into
        madspin_me/Source and then compiled against.
        """
        dest = pjoin(self.tmpdir, 'make_opts')
        misc.atomic_copy(self.source, dest)

        def check(content):
            return content == self.reference

        bad, reads = race(lambda: misc.atomic_copy(self.source, dest),
                          dest, check)
        self.assertEqual(bad, [],
            'reader saw %d truncated/missing make_opts in %d reads'
            % (len(bad), reads))


class TestUpdateMakeOptsFull(unittest.TestCase):
    """CommonRunCmd.update_make_opts_full"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.source = pjoin(MG5DIR, 'Template', 'LO', 'Source', '.make_opts')
        self.path = pjoin(self.tmpdir, 'make_opts')
        shutil.copy(self.source, self.path)
        self.update = common_run_interface.CommonRunCmd.update_make_opts_full

    def content(self):
        with open(self.path) as fsock:
            return fsock.read()

    def test_updates_the_variable_and_keeps_the_body(self):
        self.update(self.path, {'DEFAULT_F_COMPILER': 'gfortran-14'})
        content = self.content()
        self.assertIn('DEFAULT_F_COMPILER=gfortran-14', content)
        # the part make actually needs, which lives after the marker
        self.assertIn('FC=$(DEFAULT_F_COMPILER)', content)
        self.assertIn('libext=a', content)
        self.assertIn('-ffixed-line-length-132', content)

    def test_refuses_an_empty_make_opts(self):
        """A truncated file used to be read as 'no body', and the body-less
        result was then written back -- making a transient race permanent."""
        open(self.path, 'w').close()
        self.assertRaises(MadGraph5Error, self.update,
                          self.path, {'DEFAULT_F_COMPILER': 'gfortran'})

    def test_refuses_a_make_opts_truncated_inside_the_variable_block(self):
        with open(self.path, 'w') as fsock:
            fsock.write('DEFAULT_F2PY_COMPILER=f2py\nDEFAULT_F_COMPI')
        self.assertRaises(MadGraph5Error, self.update,
                          self.path, {'DEFAULT_F_COMPILER': 'gfortran'})

    def test_refuses_a_make_opts_truncated_just_after_the_marker(self):
        """The variables and the marker survive, the whole body is gone.

        The file still parses, so nothing downstream complains -- make just
        keeps its builtin $(FC) and an undefined $(libext). Rewriting it here
        would make that permanent.
        """
        with open(self.path, 'w') as fsock:
            fsock.write('DEFAULT_F2PY_COMPILER=f2py\n'
                        'DEFAULT_F_COMPILER=gfortran\n'
                        '#end_of_make_opts_variables\n')
        self.assertRaises(MadGraph5Error, self.update,
                          self.path, {'DEFAULT_F_COMPILER': 'gfortran-14'})

    def test_refuses_a_make_opts_whose_body_lost_the_compiler_definitions(self):
        """Body present but cut short, before FC=$(DEFAULT_F_COMPILER)."""
        with open(self.path) as fsock:
            head = fsock.read().split('FC=$(DEFAULT_F_COMPILER)')[0]
        with open(self.path, 'w') as fsock:
            fsock.write(head)
        self.assertRaises(MadGraph5Error, self.update,
                          self.path, {'DEFAULT_F_COMPILER': 'gfortran-14'})

    def test_refuses_a_make_opts_without_default_f_compiler(self):
        """Without it make keeps its builtin $(FC), i.e. f77."""
        with open(self.path, 'w') as fsock:
            fsock.write('STDLIB=-lstdc++\n#end_of_make_opts_variables\n\n'
                        'FC=$(DEFAULT_F_COMPILER)\nlibext=a\n')
        self.assertRaises(MadGraph5Error, self.update,
                          self.path, {'STDLIB': '-lc++'})

    def test_a_body_less_make_opts_is_never_written(self):
        """Belt and braces: whatever happens, the file that is left on disk
        still carries the definitions the compilation depends on."""
        try:
            open(self.path, 'w').close()
            self.update(self.path, {'DEFAULT_F_COMPILER': 'gfortran'})
        except MadGraph5Error:
            pass
        content = self.content()
        self.assertTrue(not content.strip() or 'FC=$(DEFAULT_F_COMPILER)' in content,
                        'update_make_opts_full persisted a make_opts with no body')

    def test_write_is_atomic(self):
        """A reader racing update_make_opts_full always sees a usable file."""
        counter = itertools.count()

        def write():
            self.update(self.path, {'DEFAULT_F_COMPILER':
                                    'gfortran%d' % (next(counter) % 2)})

        def check(content):
            return ('FC=$(DEFAULT_F_COMPILER)' in content
                    and 'libext=a' in content)

        bad, reads = race(write, self.path, check)
        self.assertEqual(bad, [],
            'reader saw %d make_opts without the compiler definitions in %d reads'
            % (len(bad), reads))
