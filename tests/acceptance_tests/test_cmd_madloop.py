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
from __future__ import division
import subprocess
import unittest
import os
import re
import shutil
import copy
import sys
import logging
import time

logger = logging.getLogger('test_cmd')

import tests.unit_tests.iolibs.test_file_writers as test_file_writers

import madgraph.interface.master_interface as MGCmd
import madgraph.interface.amcatnlo_run_interface as NLOCmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.various.misc as misc

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd

pjoin = os.path.join
path = os.path

#===============================================================================
# TestCmdLoop
#===============================================================================
class TestCmdLoop(unittest.TestCase):
    """this treats all the command not related to MG_ME"""
    
    def setUp(self):
        """ Initialize the test """
        self.interface = MGCmd.MasterCmd()
        # Below the key is the name of the logger and the value is a tuple with
        # first the handlers and second the level.
        self.logger_saved_info = {}
    
    def generate(self, process, model):
        """Create a process"""
        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception, error:
            pass

        self.interface.onecmd('import model %s' % model)
        if isinstance(process, str):
            interface.onecmd('generate %s' % process)
        else:
            for p in process:
                self.interface.onecmd('add process %s' % p)
        self.interface.onecmd('output /tmp/MGPROCESS -f')      
    
    def do(self, line):
        """ exec a line in the interface """        
        self.interface.onecmd(line)
    
    def setup_logFile_for_logger(self,full_logname,restore=False,level='DEBUG'):
        """ Setup the logger by redirecting them all to logfiles in tmp """
        
        logs = full_logname.split('.')
        lognames = [ '.'.join(logs[:(len(logs)-i)]) for i in\
                                            range(len(full_logname.split('.')))]
        for logname in lognames:
            try:
                os.remove('/tmp/%s.log'%logname)
            except Exception, error:
                pass
            my_logger = logging.getLogger(logname)
            if restore:
                my_logger.setLevel(self.logger_saved_info[logname][1])
            else:
                self.logger_saved_info[logname] = (copy.copy(my_logger.handlers),\
                                                                my_logger.level)
                my_logger.setLevel(level)
            allHandlers = copy.copy(my_logger.handlers)
            for h in allHandlers:
                my_logger.removeHandler(h)
            if restore:
                for h in self.logger_saved_info[logname][0]:
                    my_logger.addHandler(h)
            else:
                hdlr = logging.FileHandler('/tmp/%s.log'%logname)
                my_logger.addHandler(hdlr)
        if not restore:
            for logname in lognames:      
                my_logger.info('Log of %s'%logname)

    def test_ML_launch_gg_ddx(self):
        """test that the output works fine for g g > d d~ [virt=QCD]"""

        self.setup_logFile_for_logger('cmdprint.ext_program')
        cmd = os.getcwd()
        self.generate(['g g > d d~ [virt=QCD]'], 'loop_sm')
        self.assertEqual(cmd, os.getcwd())
        self.do('launch -f')
        
        # Test that the result of the run is present.
        self.assertTrue(path.isfile('/tmp/MGPROCESS/SubProcesses/P0_gg_ddx/result.dat'))
        # Test that the Helicity filter is correctly configured.
        self.assertTrue(path.isfile('/tmp/MGPROCESS/SubProcesses/P0_gg_ddx/HelFilter.dat'))
        # Test that the cmdprint log file is there
        self.assertTrue(path.isfile('/tmp/cmdprint.ext_program.log'))
        # Test that it contains the expected output. 
        # Of course I could setup a detailed regexpr to make sure the values
        # in the output are not NaN or so, but it is not really the idea of these
        # acceptance tests.
        self.assertTrue('Results for process gg > ddx' in \
                                   open('/tmp/cmdprint.ext_program.log').read())
        self.setup_logFile_for_logger('cmdprint.ext_program',restore=True)      
    
    def test_ML_check_brs_gd_gd(self):
        """test that the brs check works fine on g d > g d"""
        
        self.setup_logFile_for_logger('madgraph.export_v4')
        cmd = os.getcwd()
        self.do('import model loop_sm')
        self.do('check brs -reuse g d > g d [virt=QCD]')
        self.assertEqual(cmd, os.getcwd())
        self.assertTrue(path.isfile('/tmp/madgraph.export_v4.log'))
        res = open('/tmp/madgraph.export_v4.log').read()
        self.assertTrue('Process [QCD]' in res)
        self.assertTrue('Summary: 1/1 passed, 0/1 failed' in res)
        self.assertTrue('BRS' in res)
        self.assertTrue('Passed' in res)
        self.setup_logFile_for_logger('madgraph.export_v4',restore=True)

    def test_ML_check_full_epem_ttx(self):
        """ Test that check full e+ e- > t t~ works fine """
        
        self.setup_logFile_for_logger('madgraph.export_v4')
        cmd = os.getcwd()
        self.do('import model loop_sm')
        self.do('check full -reuse e+ e- > t t~ [virt=QCD]')
        self.assertEqual(cmd, os.getcwd())
        self.assertTrue(path.isfile('/tmp/madgraph.export_v4.log'))
        res = open('/tmp/madgraph.export_v4.log').read()
        # Needs the loop_sm feynman model to successfully run the gauge check.
        # self.assertTrue('Gauge results' in res)
        self.assertTrue('Lorentz invariance results' in res)
        self.assertTrue('Process permutation results:' in res)
        self.assertTrue('Summary: 1/1 passed, 0/1 failed' in res)
        self.assertTrue(res.count('Passed')==3)
        self.assertTrue('1/1 passed' in res) 
        self.setup_logFile_for_logger('madgraph.export_v4',restore=True)

    def test_ML_check_timing_epem_ttx(self):
        """ Test that check timing e+ e- > t t~ works fine """
        
        self.setup_logFile_for_logger('madgraph.export_v4')
        cmd = os.getcwd()
        self.do('import model loop_sm')
        self.do('check timing -reuse e+ e- > t t~ [virt=QCD]')
        self.assertEqual(cmd, os.getcwd())
        self.assertTrue(path.isdir(pjoin(MG5DIR,'TMP_CHECK_epem_ttx')))
        self.assertTrue(path.isfile(pjoin(MG5DIR,'TMP_CHECK_epem_ttx',\
                                        'SubProcesses/P0_epem_ttx/result.dat')))        
        self.assertTrue(path.isfile('/tmp/madgraph.export_v4.log'))
        res = open('/tmp/madgraph.export_v4.log').read()
        self.assertTrue('Generation time total' in res)
        self.assertTrue('Executable size' in res)
        self.assertTrue(not 'NA' in res)
        try:
            shutil.rmtree('TMP_CHECK_epem_ttx')
        except Exception, error:
            pass
        self.setup_logFile_for_logger('madgraph.export_v4',restore=True)

    def test_ML_check_profile_epem_ttx(self):
        """ Test that check profile e+ e- > t t~ works fine """

        self.setup_logFile_for_logger('madgraph.export_v4')
        cmd = os.getcwd()
        self.do('import model loop_sm')
        self.do('check profile -reuse e+ e- > t t~ [virt=QCD]')
        self.assertEqual(cmd, os.getcwd())
        self.assertTrue(path.isdir(pjoin(MG5DIR,'TMP_CHECK_epem_ttx')))
        self.assertTrue(path.isfile(pjoin(MG5DIR,'TMP_CHECK_epem_ttx',\
                                        'SubProcesses/P0_epem_ttx/result.dat')))        
        self.assertTrue(path.isfile('/tmp/madgraph.export_v4.log'))
        res = open('/tmp/madgraph.export_v4.log').read()        
        self.assertTrue('Generation time total' in res)
        self.assertTrue('Executable size' in res)
        self.assertTrue('Double precision results' in res)
        self.assertTrue('Number of Exceptional PS points' in res)
        self.assertTrue(res.count('NA')<=3)
        try:
            shutil.rmtree('TMP_CHECK_epem_ttx')
        except Exception, error:
            pass
        self.setup_logFile_for_logger('madgraph.export_v4',restore=True)