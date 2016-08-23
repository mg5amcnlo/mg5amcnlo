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

"""Unit test library for the export_FKS format routines"""

import StringIO
import copy
import fractions
import os 
import sys
import tempfile
import glob
import shutil

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.various.misc as misc
import madgraph.iolibs.files as files
import tests.IOTests as IOTests
import madgraph.interface.master_interface as MGCmd

import madgraph.fks.fks_common as fks_common
from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir,
                                'input_files')

#===============================================================================
# IOExportFKSTest
#===============================================================================
class IOExportFKSTest(IOTests.IOTestManager):
    """Test class for the export fks module"""

#     def setUp(self):
#         """ Initialize the test """
# 
#         self.interface = MGCmd.MasterCmd()
#         self.interface.no_notification()
#         # Below the key is the name of the logger and the value is a tuple with
#         # first the handlers and second the level.
#         self.logger_saved_info = {}
# 
#         # Select the Tensor Integral to include in the test
#         misc.deactivate_dependence('pjfry', cmd = self.interface, log='stdout')
#         misc.deactivate_dependence('samurai', cmd = self.interface, log='stdout')        
#         misc.activate_dependence('golem', cmd = self.interface, log='stdout')
#         misc.activate_dependence('ninja', cmd = self.interface, log='stdout',MG5dir=MG5DIR)


    def generate(self, process, model, multiparticles=[]):
        """Create a process"""

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        interface.no_notification()
        # Select the Tensor Integral to include in the test
        misc.deactivate_dependence('pjfry', cmd = interface, log='stdout')
        misc.deactivate_dependence('samurai', cmd = interface, log='stdout')        
        misc.activate_dependence('golem', cmd = interface, log='stdout')
        misc.activate_dependence('ninja', cmd = interface, log='stdout',MG5dir=MG5DIR)        
        
        run_cmd('import model %s' % model)
        for multi in multiparticles:
            run_cmd('define %s' % multi)
        if isinstance(process, str):
            run_cmd('generate %s' % process)
        else:
            for p in process:
                run_cmd('add process %s' % p)

        files.rm(self.IOpath)
        run_cmd('output %s -f' % self.IOpath)


    @IOTests.createIOTest()
    def testIO_test_pptt_fksreal(self):
        """ target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > t t~ [real=QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_ppw_fksall(self):
        """ target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > w+ [QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_tdecay_fksreal(self):
        """ target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['t > j j b [real=QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_pptt_fks_loonly(self):
        """ target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > t t~ [LOonly=QCD]'], 'sm')
        
