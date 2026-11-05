##############################################################################
#
# Copyright (c) 2010 The MadGraph Development team and Contributors
#
# This file is a part of the MadGraph 5 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph license which should accompany this 
# distribution.
#
# For more information, please visit: http://madgraph.phys.ucl.ac.be
#
################################################################################
""" Basic test of the command interface """

from __future__ import absolute_import
import unittest
import madgraph
import madgraph.interface.master_interface as mgcmd
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.interface.madevent_interface as mecmd
import madgraph.interface.common_run_interface as runcmd
import madgraph.iolibs.files as files
import madgraph.various.misc as misc
import madgraph.madevent.combine_runs as combine_runs
import io
import os
import readline
import tempfile

class TestCombineRuns(unittest.TestCase):
    """ check if the ValidCmd works correctly """
    

    def test_copy_events_no_blank_line(self):
        """copy_events must not leave a blank line after the event header.

        The header is re-emitted with a rewritten weight; splitting the
        original line with a maxsplit leaves the trailing newline on the
        last field, so the rewritten line ends with two. Pythia's LHEF
        reader calls such an event corrupt and refuses the whole file,
        which makes every multi-channel sample unshowerable."""

        event = (
            "<event>\n"
            " 4 1 +1.3649000e+06 0.2333376E+02  0.7546771E-02  0.1686156E+00\n"
            "        11   -1    0    0    0    0  0.0E+00  0.0E+00"
            "  0.1E+03  0.1E+03  0.0E+00 0. -1.\n"
            "       -11   -1    0    0    0    0  0.0E+00  0.0E+00"
            " -0.1E+03  0.1E+03  0.0E+00 0.  1.\n"
            "</event>\n")

        fdi, inpath = tempfile.mkstemp(suffix='.lhe')
        with os.fdopen(fdi, 'w') as fsock:
            fsock.write(event)
        try:
            out = io.StringIO()
            combiner = combine_runs.CombineRuns.__new__(combine_runs.CombineRuns)
            # max_wgt of 0 keeps every event, so the output is deterministic
            nb = combiner.copy_events(out, inpath, 1.0, 0.0)
        finally:
            os.remove(inpath)

        self.assertEqual(nb, 1)
        lines = out.getvalue().split('\n')
        self.assertEqual(lines[0], '<event>')
        self.assertNotEqual(lines[2].strip(), '',
                            'blank line after the event header:\n%s'
                            % out.getvalue())
        self.assertTrue(lines[2].lstrip().startswith('11'))

    def test_get_fortran_str(self):
        fct = combine_runs.CombineRuns.get_fortran_str
        self.assertEqual(fct(1.0),'0.1000000E+01')
        self.assertEqual(fct(1.123456789123456789e-88),'0.1123457E-87')
        self.assertEqual(fct(1.123456789123456789e-128),'0.1123457E-127')

    
