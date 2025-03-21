##############################################################################
#
# Copyright (c) 2010 The MadGraph5_aMC@NLO Development team and Contributors
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
from __future__ import absolute_import
from cmd import Cmd
from six.moves import zip
""" Basic test of the command interface """

import unittest
import madgraph
import madgraph.interface.master_interface as mgcmd
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.interface.amcatnlo_interface as amcatnlocmd
import os
import madgraph.fks.fks_helas_objects as fks_helas
import copy
import madgraph.iolibs.save_load_object as save_load_object
import madgraph
import madgraph.various.misc as misc
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.dirname(root_path)
# root_path is ./tests
pjoin = os.path.join

class TestAMCatNLOEWTagPh(unittest.TestCase):
    """ a suite of extra tests for the ew stuff """
    
    def setUp(self):
        self.interface = mgcmd.MasterCmd()

    def test_generate_fks_tagph(self):
        """check that the generate command works as expected.
        In particular the correct number of real-emission processes
        and FKS partners"""
        cmd_list = [
            'u u~ > a a [real=QED]',
            'u u~ > !a! !a! [real=QED]',
            'u u~ > !2a!  [real=QED]',
            'u u~ > 2!a! [real=QED]'] 

        nrealproc_list = [9, 3, 3, 3]
        fks_j_list = [[1,2,3], [1,2], [1,2], [1,2]] # 4 is not there in the first set for symmetry


        for cmd, nrealproc, fks_j in \
                zip(cmd_list, nrealproc_list, fks_j_list):
            self.interface.do_generate(cmd)

            fksprocess = self.interface._fks_multi_proc['born_processes'][0]

            self.assertEqual(len(fksprocess.real_amps), nrealproc)
            self.assertEqual(set([r.fks_infos[0]['j'] for r in fksprocess.real_amps]), set(fks_j))
            
    def test_invalid_syntax_tag(self):

        cmd = "u u~ > !a! !a!"
        self.assertRaises(madgraph.InvalidCmd,  self.interface.do_generate, cmd)

        #cmd = "u u~ > !z! a [real=QED]"
        #self.assertRaises(madgraph.InvalidCmd,  self.interface.do_generate, cmd)

        #cmd = "u u~ > !t! t~ a [real=QED]"
        #self.assertRaises(madgraph.InvalidCmd,  self.interface.do_generate, cmd)

        #cmd = "u u~ > !a! a [real=QCD]" # is this valid ? I guess NOT [note this is QCD]
        #self.assertRaises(madgraph.InvalidCmd,  self.interface.do_generate, cmd)

        
