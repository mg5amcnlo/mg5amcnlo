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
from cmd import Cmd
""" Basic test of the command interface """

import unittest
import madgraph
import madgraph.interface.master_interface as mgcmd
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.interface.amcatnlo_interface as amcatnlocmd
import os


root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.dirname(root_path)
# root_path is ./tests
pjoin = os.path.join

class TestAMCatNLOCmd(unittest.TestCase):
    """ check if the ValidCmd works correctly """
    
    interface = mgcmd.MasterCmd()

    def test_generate(self):
        """check that the generate command works as expected"""
        cmd_list = [
            'u u~ > d d~ QED=0 QCD=2 [real=QCD]',
            'u u~ > d d~ QED=0 QCD=2 [real=QED]',
            'u u~ > d d~ QED=0 QCD=2 [real=QED QCD]',
            'u u~ > d d~  [real=QCD]',
            'u u~ > d d~  [real=QED]',
            'u u~ > d d~  [real=QED QCD]',
            'u u~ > d d~ QCD=2 QED=2 [real=QCD]',
            'u u~ > d d~ QCD=2 QED=2 [real=QED]',
            'u u~ > d d~ QCD=2 QED=2 [real=QED QCD]']
        # number of expected born diagrams
        nborndiag_list = [1, 4, 4, 1, 4, 4, 4, 4, 4]
        # number of expected real emission processes
        nrealproc_list = [3, 3, 6, 3, 3, 6, 3, 3, 6]

        for cmd, nborndiag, nrealproc in zip(cmd_list, nborndiag_list, nrealproc_list):
            self.interface.do_generate(cmd)
            fksprocess = self.interface._fks_multi_proc['born_processes'][0]
            # all these processes should only have 1 born
            self.assertEqual(len(fksprocess.born_amp_list), 1)
            self.assertEqual(len(fksprocess.born_amp_list[0]['diagrams']), nborndiag)
            self.assertEqual(len(fksprocess.real_amps), nrealproc)

