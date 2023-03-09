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

import unittest
import madgraph
import madgraph.interface.master_interface as mgcmd
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.interface.madevent_interface as mecmd
import madgraph.interface.common_run_interface as runcmd
import madgraph.iolibs.files as files
import madgraph.various.misc as misc
import madgraph.madevent.combine_runs as combine_runs
import os
import readline

class TestCombineRuns(unittest.TestCase):
    """ check if the ValidCmd works correctly """
    

    def test_get_fortran_str(self):
        fct = combine_runs.CombineRuns.get_fortran_str
        self.assertEqual(fct(1.0),'0.1000000E+01')
        self.assertEqual(fct(1.123456789123456789e-88),'0.1123457E-87')
        self.assertEqual(fct(1.123456789123456789e-128),'0.1123457E-127')

    
