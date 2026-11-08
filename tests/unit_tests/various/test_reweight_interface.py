################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
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
"""Unit tests for the reweight interface helpers."""
from __future__ import absolute_import

import os
import unittest

import madgraph.interface.reweight_interface as rwgt_interface


class TestF2pyLibraryName(unittest.TestCase):
    """A library already loaded in the process cannot be replaced: the
    dynamic loader resolves a new library of the same name (from any
    directory) to the loaded one, so a rebuild gets a new name."""

    def setUp(self):
        self.saved = set(rwgt_interface.loaded_f2py_names)
        rwgt_interface.loaded_f2py_names.clear()

    def tearDown(self):
        rwgt_interface.loaded_f2py_names.clear()
        rwgt_interface.loaded_f2py_names.update(self.saved)

    def test_suffix_only_for_a_name_already_loaded(self):
        suffix = rwgt_interface.ReweightInterface.get_f2py_suffix
        sdir = os.path.join('a', 'rwgt', 'rw_me', 'SubProcesses')
        other = os.path.join('b', 'rwgt', 'rw_me', 'SubProcesses') + os.sep
        self.assertEqual(suffix(sdir), '')
        rwgt_interface.loaded_f2py_names.add('rw_me')
        # same name from another directory: the loader would reuse it
        self.assertEqual(suffix(other), '_r2')
        rwgt_interface.loaded_f2py_names.add('rw_me_r2')
        self.assertEqual(suffix(sdir), '_r3')
        # the virtual output has its own name
        virt = os.path.join('a', 'rwgt', 'rw_mevirt', 'SubProcesses')
        self.assertEqual(suffix(virt), '')


if __name__ == '__main__':
    unittest.main()
