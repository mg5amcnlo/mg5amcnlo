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

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir,
                                'input_files')

#===============================================================================
# IOExportFKSTest
#===============================================================================
class IOExportFKSTest(IOTests.IOTestManager):
    """Test class for the export fks module"""

    def generate(self, process, model, multiparticles=[]):
        """Create a process"""

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        
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
        
