################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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
import tests.IOTests as IOTests

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
    logger_saved_info = {}
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
    
    @classmethod
    def setup_logFile_for_logger(cls,full_logname,restore=False,level=logging.DEBUG):
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
            hdlr = logging.FileHandler('/tmp/%s.log'%logname)            
            if restore:
                my_logger.removeHandler(cls.logger_saved_info[logname][0])
                my_logger.setLevel(cls.logger_saved_info[logname][1])
                for i, h in enumerate(my_logger.handlers):
                    h.setLevel(cls.logger_saved_info[logname][2][i])
            else:
                # I assume below that the orders of the handlers in my_logger.handlers
                # remains the same after having added/removed the FileHandler
                cls.logger_saved_info[logname] = [hdlr,my_logger.level,\
                                                [h.level for h in my_logger.handlers]]
                for h in my_logger.handlers:
                    # This not elegant, but the only way I could find to mute this handlers
                    h.setLevel(logging.CRITICAL+1)
                my_logger.addHandler(hdlr)
                my_logger.setLevel(level)

        if not restore:
            for logname in lognames:
                logging.getLogger(logname).debug('Log of %s'%logname)
    
    def notest_ML_launch_gg_ddx(self):
        """test that the output works fine for g g > d d~ [virt=QCD]"""

        self.setup_logFile_for_logger('cmdprint.ext_program')
        try:
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
        except:
            self.setup_logFile_for_logger('cmdprint.ext_program',restore=True)      
            raise
        self.setup_logFile_for_logger('cmdprint.ext_program',restore=True)

    def test_ML_check_brs_gd_gd(self):
        """test that the brs check works fine on g d > g d"""
        
        self.setup_logFile_for_logger('madgraph.check_cmd')
        try:
            cmd = os.getcwd()
            self.do('import model loop_sm')
            self.do('check brs -reuse g d > g d [virt=QCD]')
            self.assertTrue(path.isfile(pjoin(MG5DIR,'TMP_CHECK',\
                                               'SubProcesses/P0_gd_gd/result.dat')))
            shutil.rmtree(pjoin(MG5DIR,'TMP_CHECK'))
            self.assertEqual(cmd, os.getcwd())
            self.assertTrue(path.isfile('/tmp/madgraph.check_cmd.log'))
            res = open('/tmp/madgraph.check_cmd.log').read()
            self.assertTrue('Process [virt=QCD]' in res)
            self.assertTrue('Summary: 1/1 passed, 0/1 failed' in res)
            self.assertTrue('BRS' in res)
            self.assertTrue('Passed' in res)
        except:
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)      
            raise
        self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)

    def test_ML_check_full_epem_ttx(self):
        """ Test that check full e+ e- > t t~ works fine """
        
        self.setup_logFile_for_logger('madgraph.check_cmd')
        try:
            cmd = os.getcwd()
            self.do('import model loop_sm')
            self.do('check full -reuse e+ e- > t t~ [virt=QCD]')
            self.assertEqual(cmd, os.getcwd())
            self.assertTrue(path.isfile(pjoin(MG5DIR,'TMP_CHECK',\
                                            'SubProcesses/P0_epem_ttx/result.dat')))
            shutil.rmtree(pjoin(MG5DIR,'TMP_CHECK'))
            self.assertTrue(path.isfile('/tmp/madgraph.check_cmd.log'))
            res = open('/tmp/madgraph.check_cmd.log').read()
            # Needs the loop_sm feynman model to successfully run the gauge check.
            # self.assertTrue('Gauge results' in res)
            self.assertTrue('Lorentz invariance results' in res)
            self.assertTrue('Process permutation results:' in res)
            self.assertTrue('Gauge results' in res)
            self.assertTrue('Summary: passed' in res)
            self.assertTrue('Passed' in res)
            self.assertTrue('Failed' not in res)
            self.assertTrue('1/1 failed' not in res)
            self.assertTrue('1/1 passed' in res)
        except:
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
            raise
        self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)

    def test_ML_check_timing_epem_ttx(self):
        """ Test that check timing e+ e- > t t~ works fine """
        
        self.setup_logFile_for_logger('madgraph.check_cmd')
        try:
            cmd = os.getcwd()
            self.do('import model loop_sm')
            if path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')):
                shutil.rmtree(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx'))
            # Make sure it works for an initial run
            self.do('check timing -reuse e+ e- > t t~ [virt=QCD]')
            self.assertEqual(cmd, os.getcwd())
            self.assertTrue(path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')))
            self.assertTrue(path.isfile(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx',\
                                            'SubProcesses/P0_epem_ttx/result.dat')))
            self.assertTrue(path.isfile('/tmp/madgraph.check_cmd.log'))
            res = open('/tmp/madgraph.check_cmd.log').read()
            self.assertTrue('Generation time total' in res)
            self.assertTrue('Executable size' in res)
            self.assertTrue(not 'NA' in res)
            
            # Now for a Reuse-run
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
            self.setup_logFile_for_logger('madgraph.check_cmd')
            self.do('check timing -reuse e+ e- > t t~ [virt=QCD]')
            self.assertEqual(cmd, os.getcwd())
            self.assertTrue(path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')))
            self.assertTrue(path.isfile(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx',\
                                            'SubProcesses/P0_epem_ttx/result.dat')))
            shutil.rmtree(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx'))
            self.assertTrue(path.isfile('/tmp/madgraph.check_cmd.log'))
            res = open('/tmp/madgraph.check_cmd.log').read()
            self.assertTrue('Generation time total' in res)
            self.assertTrue('Executable size' in res)
            self.assertTrue(res.count('NA')<=8)
        except:
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
            if path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')):
                shutil.rmtree(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx'))
            raise
        self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)

    def test_ML_check_profile_epem_ttx(self):
        """ Test that check profile e+ e- > t t~ works fine """

        self.setup_logFile_for_logger('madgraph.check_cmd')
        try:
            cmd = os.getcwd()
            self.do('import model loop_sm')
            if path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')):
                shutil.rmtree(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx'))
            
            # Make sure it works for an initial run
            self.do('check profile -reuse e+ e- > t t~ [virt=QCD]')
            self.assertEqual(cmd, os.getcwd())
            self.assertTrue(path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')))
            self.assertTrue(path.isfile(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx',\
                                            'SubProcesses/P0_epem_ttx/result.dat')))
            self.assertTrue(path.isfile('/tmp/madgraph.check_cmd.log'))
            res = open('/tmp/madgraph.check_cmd.log').read()
            self.assertTrue('Generation time total' in res)
            self.assertTrue('Executable size' in res)
            self.assertTrue('Tool (DoublePrec for CT)' in res)
            self.assertTrue('Number of Unstable PS points' in res)
            self.assertTrue(res.count('NA')<=3)

            # Now for a Reuse-run
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
            self.setup_logFile_for_logger('madgraph.check_cmd')
            self.do('check profile -reuse e+ e- > t t~ [virt=QCD]')
            self.assertEqual(cmd, os.getcwd())
            self.assertTrue(path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')))
            self.assertTrue(path.isfile(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx',\
                                            'SubProcesses/P0_epem_ttx/result.dat')))
            shutil.rmtree(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx'))
            self.assertTrue(path.isfile('/tmp/madgraph.check_cmd.log'))
            res = open('/tmp/madgraph.check_cmd.log').read()
            self.assertTrue('Generation time total' in res)
            self.assertTrue('Executable size' in res)
            self.assertTrue('Tool (DoublePrec for CT)' in res)
            self.assertTrue('Number of Unstable PS points' in res)
            self.assertTrue(res.count('NA')<=11)
        except:
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
            if path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')):
                shutil.rmtree(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx'))
            raise
        self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)

#===============================================================================
# IOTestMadLoopOutputFromInterface
#===============================================================================
class IOTestMadLoopOutputFromInterface(IOTests.IOTestManager):
    """Test MadLoop outputs when generated directly from the interface."""

    @IOTests.createIOTest(groupName='MadLoop_output_from_the_interface')
    def testIO_TIR_output(self):
        """ target: [ggttx_IOTest/SubProcesses/(.*)\.f]
        """
        interface = MGCmd.MasterCmd()

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        # Make sure the potential TIR are set uniformly across users
        if interface.options['golem'] in [None,'auto']:
            interface.run_cmd('install Golem95')
        interface.options['pjfry']=None
        
        run_cmd('generate g g > t t~ [virt=QCD]')
        interface.onecmd('output %s -f' % str(pjoin(self.IOpath,'ggttx_IOTest')))
