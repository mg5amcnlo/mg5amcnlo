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
import subprocess
import unittest
import os
import re
import shutil
import sys
import logging

logger = logging.getLogger('test_cmd')

import tests.unit_tests.iolibs.test_file_writers as test_file_writers

import madgraph.interface.cmd_interface as MGCmd
import madgraph.interface.madevent_interface as MECmd
import madgraph.interface.launch_ext_program as launch_ext
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd

#===============================================================================
# TestCmd
#===============================================================================
class TestMECmdShell(unittest.TestCase):
    """this treats all the command not related to MG_ME"""
    
    def generate(self, process, model):
        """Create a process"""

        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception, error:
            print error
            pass

        interface = MGCmd.MadGraphCmdShell()
        interface.onecmd('import model %s' % model)
        interface.onecmd('generate %s' % process)
        interface.onecmd('output madevent /tmp/MGPROCESS/ -f')
        self.cmd_line = MECmd.MadEventCmdShell(me_dir= '/tmp/MGPROCESS')


    @staticmethod
    def join_path(*path):
        """join path and treat spaces"""     
        combine = os.path.join(*path)
        return combine.replace(' ','\ ')        
    
    def do(self, line):
        """ exec a line in the cmd under test """        
        self.cmd_line.exec_cmd(line)
        
    def test_creating_matched_plot(self):
        """test that the creation of matched plot works"""
        
        self.generate('p p > W+ j', 'sm')
        shutil.copy(os.path.join(_file_path, 'input_files', 'run_card_matching.dat'),
                    '/tmp/MGPROCESS/Cards/run_card.dat')
        shutil.copy('/tmp/MGPROCESS/Cards/pythia_card_default.dat',
                    '/tmp/MGPROCESS/Cards/pythia_card.dat')
        self.do('generate_events')
        self.do('generate_events')
        self.do('pythia run_01')

