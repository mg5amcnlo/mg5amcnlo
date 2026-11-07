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
"""CommonRunCmd.copy_lep_densities puts the chosen dressed-lepton density into
Source/PDF before Source is compiled. Source/makefile builds libpdf.a from the
PDF *directory*, whose mtime does not move when an existing file is
overwritten, so switching pdlabel between two launches of one process
directory used to keep the previous density linked: a fcce365ll run after a
fcce240ll one gave 1.324 pb instead of 1.195 pb for e- e+ > mu- mu+."""

from __future__ import absolute_import

import os
import shutil
import tempfile
import unittest

import madgraph.interface.common_run_interface as common_run

pjoin = os.path.join


class _Cmd(object):
    """copy_lep_densities needs no state"""

    copy_lep_densities = common_run.CommonRunCmd.copy_lep_densities


class TestCopyLepDensities(unittest.TestCase):

    DENSITY_FILES = ('eepdf.f', 'gridpdfaux.f')

    def setUp(self):
        self.me_dir = tempfile.mkdtemp(prefix='copy_lep_densities_')
        self.source = pjoin(self.me_dir, 'Source')
        self.pdf = pjoin(self.source, 'PDF')
        self.lib = pjoin(self.me_dir, 'lib', 'libpdf.a')
        for name in ('setA', 'setB'):
            os.makedirs(pjoin(self.pdf, 'lep_densities', name))
            for filename in self.DENSITY_FILES:
                self.write(pjoin(self.pdf, 'lep_densities', name, filename),
                           '      ! %s %s\n' % (name, filename))
        os.makedirs(os.path.dirname(self.lib))
        self.cmd = _Cmd()

    def tearDown(self):
        shutil.rmtree(self.me_dir)

    @staticmethod
    def write(path, text):
        with open(path, 'w') as stream:
            stream.write(text)

    def compile(self):
        """stand in for 'make': the objects and the library of what is there"""
        for filename in self.DENSITY_FILES:
            self.write(pjoin(self.pdf, filename[:-2] + '.o'), 'object')
        self.write(self.lib, 'library')

    def read(self, filename):
        with open(pjoin(self.pdf, filename)) as stream:
            return stream.read()

    def test_switching_density_invalidates_the_build(self):
        self.cmd.copy_lep_densities('setA', self.source)
        self.compile()

        self.cmd.copy_lep_densities('setB', self.source)

        for filename in self.DENSITY_FILES:
            self.assertIn('setB', self.read(filename))
            self.assertFalse(os.path.exists(pjoin(self.pdf, filename[:-2] + '.o')),
                             filename)
        # without this, Source/makefile never even enters PDF
        self.assertFalse(os.path.exists(self.lib))

    def test_same_density_keeps_the_build(self):
        self.cmd.copy_lep_densities('setA', self.source)
        self.compile()
        mtimes = dict((path, os.path.getmtime(path)) for path in
                      [self.lib] + [pjoin(self.pdf, f) for f in self.DENSITY_FILES])

        self.cmd.copy_lep_densities('setA', self.source)

        for filename in self.DENSITY_FILES:
            self.assertTrue(os.path.exists(pjoin(self.pdf, filename[:-2] + '.o')))
        for path, mtime in mtimes.items():
            self.assertEqual(os.path.getmtime(path), mtime, path)

    def test_first_copy_without_a_build(self):
        self.cmd.copy_lep_densities('setA', self.source)
        for filename in self.DENSITY_FILES:
            self.assertIn('setA', self.read(filename))

    def test_unknown_density(self):
        with self.assertRaises(Exception):
            self.cmd.copy_lep_densities('nosuchset', self.source)


if __name__ == '__main__':
    unittest.main()
