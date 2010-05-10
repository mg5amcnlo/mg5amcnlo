################################################################################
#
# Copyright (c) 2009 The MadGraph Development team and Contributors
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

import unittest
import os
import shutil

import madgraph.interface.cmd_interface as Cmd
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]


#===============================================================================
# TestCmd
#===============================================================================
class TestCmdShell(unittest.TestCase):
    """The TestCase class for the test the FeynmanLine"""

    def setUp(self):
        """ basic building of the class to test """
        
        self.cmd = Cmd.MadGraphCmdShell()
        try:
            self.out_dir = os.path.join(Cmd.MGME_dir, 'AUTO_TEST_MG5')
        except:
            self.out_dir = ''
            
        _input_path = os.path.join(_file_path , 
                                        '../input_files/v4_sm_particles.dat')
        self.do('import model_v4 %s' % _input_path)
        _input_path = os.path.join(_file_path , 
                                        '../input_files/v4_sm_interactions.dat')
        self.do('import model_v4 %s' % _input_path)        
        
    def tearDown(self):
        """ basic destruction after have run """
        
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
    
    def do(self, line):
        """ exec a line in the cmd under test """        
        self.cmd.exec_cmd(line)
    
    def test_generate(self):
        """command 'generate' works"""
        
        self.do('generate e+ e- > e+ e-')
        self.assertTrue(self.cmd._curr_amps)
        
    def test_addprocess(self):
        """ command 'add process' works"""
        
        self.do('generate e+ e- > e+ e-')
        self.do('add process e+ e- > e+ e- z')
        self.assertTrue(self.cmd._curr_amps)
        self.assertEqual(len(self.cmd._curr_amps), 2)
        
        
    def test_define(self):
        """command 'define' works"""
        self.do('define P u u~')
        self.do('generate e+ e- > P P')
        self.assertTrue(self.cmd._curr_amps)
        
    def test_setup(self):
        """ command 'setup' works"""
        self.do('import model_v4 sm')
        self.do('generate e+ e- > e+ e-')
        self.do('setup madevent_v4 %s' % self.out_dir)
        self.assertTrue(os.path.exists(self.out_dir))
        
        
        

