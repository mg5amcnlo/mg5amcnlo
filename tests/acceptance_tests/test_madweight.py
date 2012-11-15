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
import sys
import logging
import time

logger = logging.getLogger('test_cmd')

import tests.unit_tests.iolibs.test_file_writers as test_file_writers

import madgraph.interface.master_interface as MGCmd
import madgraph.interface.madevent_interface as MECmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.various.misc as misc

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd

pjoin = os.path.join


#===============================================================================
# TestCmd
#===============================================================================
class Testmadweight(unittest.TestCase):
    """ check if the ValidCmd works correctly """

    def generate(self, process, model):
        """Create a process"""

        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception, error:
            pass

        interface = MGCmd.MasterCmd()
        interface.onecmd('import model %s' % model)
        if isinstance(process, str):
            interface.onecmd('generate %s' % process)
        else:
            for p in process:
                interface.onecmd('add process %s' % p)
        interface.onecmd('output madweight /tmp/MGPROCESS/ -f')




    def test_zh(self):
        """test output madweight for one specific process"""

        cmd = os.getcwd()
        self.generate('p p > Z h , Z > mu+ mu- , h > b b~ ' , 'sm')
        misc.sprint(cmd, level=50)
        # test that each file in P0_qq_zh_z_ll_h_bbx has been correctly written
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/matrix1.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/matrix2.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/auto_dsig1.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/auto_dsig2.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/auto_dsig.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/gen_ps.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/configs.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/coupl.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/driver.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/initialization.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/leshouche.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/madweight_param.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/makefile'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/nexternal.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/mirrorprocs.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/permutation.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/phasespace.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/pmass.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/props.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/run.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_ll_h_bbx/setscales.f'))


        # test that all libraries have been compiled

        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libblocks.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libcernlib.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libdhelas.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libgeneric.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libmodel.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libpdf.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libtools.a'))
    
    def test_tt_semi(self):
        """test output madweight for one specific process"""

        cmd = os.getcwd()
        self.generate('p p > t t~ , t > e+ ve b , ( t~ > W- b~ , W- > j j )' , 'sm')
        misc.sprint(cmd, level=50)
        # test that each file in P0_qq_zh_z_ll_h_bbx has been correctly written
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_gg_ttx_t_lvlb_tx_wmbx_wm_qq'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_ttx_t_lvlb_tx_wmbx_wm_qq'))                        
                        
        # test that all libraries have been compiled

        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libblocks.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libcernlib.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libdhelas.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libgeneric.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libmodel.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libpdf.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libtools.a'))
        
    
