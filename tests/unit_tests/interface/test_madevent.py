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
import madgraph.interface.madevent_interface as mecmd
import os


root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.dirname(root_path)
# root_path is ./tests
pjoin = os.path.join

class TestMadEventCmd(unittest.TestCase):
    """ check if the ValidCmd works correctly """
    
    def test_card_type_recognition(self):
        """Check that the different card are recognize correctly"""

        detect = mecmd.MadEventCmd.detect_card_type

        # run_card
        card_dir= pjoin(root_path,'..','Template/LO', 'Cards')
        self.assertEqual(detect(pjoin(card_dir, 'run_card.dat')),
                         'run_card.dat')
        self.assertEqual(detect(pjoin(root_path, 'input_files','run_card_matching.dat')),
                         'run_card.dat')

        # PYTHIA_CARD
        self.assertEqual(detect(pjoin(card_dir, 'pythia_card_default.dat')),
                         'pythia_card.dat')

        # PARAM_CARD
        self.assertEqual(detect(pjoin(card_dir, 'param_card.dat')),
                         'param_card.dat')
        self.assertEqual(detect(pjoin(root_path, 'input_files','sps1a_param_card.dat')),
                         'param_card.dat')
        self.assertEqual(detect(pjoin(root_path, 'input_files','restrict_sm.dat')),
                         'param_card.dat')

        card_dir= pjoin(root_path,'..','Template/Common', 'Cards')

        # PLOT_CARD
        self.assertEqual(detect(pjoin(card_dir, 'plot_card.dat')),
                         'plot_card.dat')

        # Delphes
        self.assertEqual(detect(pjoin(card_dir, 'delphes_card_CMS.dat')),
                         'delphes_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'delphes_card_default.dat')),
                         'delphes_card.dat')
        # PGS
        self.assertEqual(detect(pjoin(card_dir, 'pgs_card_ATLAS.dat')),
                         'pgs_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'pgs_card_CMS.dat')),
                         'pgs_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'pgs_card_LHC.dat')),
                         'pgs_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'pgs_card_TEV.dat')),
                         'pgs_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'pgs_card_default.dat')),
                         'pgs_card.dat')
        
        # Reweight
        card_dir= pjoin(root_path,'..','Template','Common', 'Cards')
        self.assertEqual(detect(pjoin(card_dir, 'reweight_card_default.dat')),
                         'reweight_card.dat')
        
        #MadSpin card are tested in their specific routine. (in fact acceptance test)
        card_dir= pjoin(root_path,'..','Template', 'Common', 'Cards')
        self.assertEqual(detect(pjoin(card_dir, 'madspin_card_default.dat')),
                         'madspin_card.dat') 

        card_dir= pjoin(root_path,'..','Template', 'NLO', 'Cards')
        # NLO Card
        self.assertEqual(detect(pjoin(card_dir, 'run_card.dat')),
                         'run_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'shower_card.dat')),
                         'shower_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'FO_analyse_card.dat')),
                         'FO_analyse_card.dat')
        
    def test_help_category(self):
        """Check that no help category are introduced by mistake.
           If this test failes, this is due to a un-expected ':' in a command of
           the cmd interface.
        """
        cmd = mecmd.MadEventCmdShell
        category = set()
        valid_command = [c for c in dir(cmd) if c.startswith('do_')]
        
        for command in valid_command:
            obj = getattr(cmd,command)
            if obj.__doc__ and ':' in obj.__doc__:
                category.add(obj.__doc__.split(':',1)[0])
                
        target = set(['Main Commands','Advanced commands', 'Require MG5 directory', 'Not in help'])
        self.assertEqual(target, category)
