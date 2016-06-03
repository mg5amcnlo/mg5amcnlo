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
        interface.no_notification()
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
        # test that each file in P0_qq_zh_z_ll_h_bbx has been correctly written
        
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/matrix1.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/matrix2.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/auto_dsig1.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/auto_dsig2.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/auto_dsig.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/gen_ps.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/configs.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/coupl.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/driver.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/initialization.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/leshouche.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/madweight_param.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/makefile'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/nexternal.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/mirrorprocs.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/phasespace.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/pmass.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/props.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/run.inc'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_qq_zh_z_mupmum_h_bbx/setscales.f'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Cards/run_card_default.dat'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Cards/MadWeight_card_default.dat'))

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
        # test that each file in P0_qq_zh_z_ll_h_bbx has been correctly written
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/SubProcesses/P0_gg_ttx_t_epveb_tx_wmbx_wm_qq'))
                        
        # test that all libraries have been compiled

        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libblocks.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libcernlib.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libdhelas.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libgeneric.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libmodel.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libpdf.a'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/lib/libtools.a'))

class TestMadWeight(unittest.TestCase):
    """A couple of points in order to ensure the MW is working fine."""

    def get_result(self, text):
        solution = {}
        for line in text.split('\n'):
            line = line.strip().split('#')[0]
            split = line.split()
            if not len(split) in [4,5]:
                continue
            if len(split) ==4:
                event_nb, card_nb, weight, error = map(float, split)
                tf_set = 1.
            else:
                event_nb, card_nb, tf_set, weight, error = map(float, split)
            
            solution[(event_nb,card_nb,tf_set)] = (weight,error)
        return solution
            
    def test_mw_wproduction(self):
        """checking that the weight for p p > w+ > e+ ve is working"""

        try:
            shutil.rmtree(pjoin(MG5DIR,'TEST_MW_W_prod'))
        except Exception, error:
            pass
        
        cmd = """set automatic_html_opening False --no-save
                 set cluster_temp_path /tmp --no-save
                 generate p p > w+, w+ > e+ ve
                 output madweight TEST_MW_W_prod -f
                 launch
                 change_tf all_delta
                 ./tests/input_files/mw_wprod.lhco.gz
                 set madweight_card mw_parameter 13 175
                 set nb_exp_events 4
                 set log_level weight
                 set nb_event_by_node 1
                 set ebeam1 7000
                 set ebeam2 7000
                 set pdlabel cteq6l1
                 """
        open('/tmp/mg5_cmd','w').write(cmd)
        
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull
        
        subprocess.call([pjoin(MG5DIR,'bin','mg5'), 
                         '/tmp/mg5_cmd'],
                         cwd=pjoin(MG5DIR),
                        stdout=stdout, stderr=stderr)

        data = open(pjoin(MG5DIR, 'TEST_MW_W_prod', 'Events', 'fermi', 'weights.out')).read() 

        solution = self.get_result(data)
        expected = """# Weight (un-normalize) for each card/event
# format: LHCO_event_number card_id value integration_error
0 1 1.52322508477e-08 3.69373736836e-11
1 1 6.2231722171e-09 2.95094501214e-11
2 1 1.8932900739e-08 6.23556414283e-11
3 1 1.86550627721e-08 4.37562400224e-11
"""
        expected = self.get_result(expected)

        for key, (value,error) in expected.items():
            assert key in solution
            value2, error2 = solution[key]
            
            self.assertTrue(abs(value-value2) < 5* abs(error+error2),'%s != %s' % (value, value2))
            self.assertTrue(abs(value-value2)/abs(value+value2) < 0.01)
            self.assertTrue(abs(error2)/abs(value2) < 0.02)
            
        try:
            shutil.rmtree(pjoin(MG5DIR,'TEST_MW_W_prod'))
        except Exception, error:
            pass

    def test_mw_wjjproduction(self):
        """checking that the weight for p p > w+ jj,w+ > e+ ve is working"""

        try:
            shutil.rmtree(pjoin(MG5DIR,'TEST_MW_W2J_prod'))
        except Exception, error:
            pass
        
        cmd = """set automatic_html_opening False --no-save
                 set cluster_temp_path /tmp --no-save
                 generate p p > w+ j j, w+ > e+ ve
                 output madweight TEST_MW_W2J_prod
                 launch
                 change_tf all_delta
                 ./tests/input_files/mw_wjjprod.lhco.gz
                 set madweight_card mw_parameter 13 175
                 set nb_exp_events 4
                 set log_level debug
                 set nb_event_by_node 1
                 set ebeam1 7000
                 set ebeam2 7000
                 set pdlabel cteq6l1
                 """
        open('/tmp/mg5_cmd','w').write(cmd)
        
        devnull =open(os.devnull,'w')
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull
        subprocess.call([pjoin(MG5DIR,'bin','mg5'), 
                         '/tmp/mg5_cmd'],
                         cwd=pjoin(MG5DIR),
                        stdout=stdout, stderr=stderr)

        data = open(pjoin(MG5DIR, 'TEST_MW_W2J_prod', 'Events', 'fermi', 'weights.out')).read() 

        solution = self.get_result(data)
        expected = """# Weight (un-normalize) for each card/event
# format: LHCO_event_number card_id value integration_error
0 1 1.41756248942e-17 3.69590396941e-20
1 1 2.40167262714e-15 5.71647567991e-18
2 1 1.48907038945e-18 1.84546397304e-21
3 1 3.79640435481e-16 1.72128108188e-18
"""
        expected = self.get_result(expected)

        for key, (value,error) in expected.items():
            assert key in solution
            value2, error2 = solution[key]
            
            self.assertTrue(abs(value-value2) < 5* abs(error+error2), '%s != %s' % (value, value2))
            self.assertTrue(abs(value-value2)/abs(value+value2) < 0.01)
            self.assertTrue(abs(error2)/abs(value2) < 0.02)          

        try:
            shutil.rmtree(pjoin(MG5DIR,'TEST_MW_W2J_prod'))
        except Exception, error:
            pass
        
        import glob
        if glob.glob('/tmp/run*'):
            self.assertFalse(True, '''Path /tmp/run???? found on filesystem. 
            This might indicates that the cluster cleaning are commented (in submit2). 
            Please check''')
        
        
