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
"""A set of objects to allow for easy comparisons of results from various ME
generators (e.g., MG v5 against v4, ...) and output nice reports in different
formats (txt, tex, ...).
"""

import datetime
import glob
import itertools
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import unittest

pjoin = os.path.join
# Get the grand parent directory (mg5 root) of the module real path 
# (tests/acceptance_tests) and add it to the current PYTHONPATH to allow
# for easy import of MG5 tools

_file_path = os.path.dirname(os.path.realpath(__file__))

import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.save_load_object as save_load_object
import madgraph.interface.master_interface as cmd_interface

import madgraph.various.misc as misc

from madgraph import MadGraph5Error, MG5DIR
import me_comparator

class TestMadWeight(unittest.TestCase):
    """A couple of points in order to ensure the MW is working fine."""
    
    
    def get_result(self, text):
        solution = {}
        for line in text.split('\n'):
            line = line.strip().split('#')[0]
            split = line.split()
            if not len(split) == 4:
                continue
            event_nb, card_nb, weight, error = map(float, split)
            
            solution[(event_nb,card_nb)] = (weight,error)
        return solution
            
            
    def test_short_mw_wproduction(self):
        """checking that the weight for p p > w+ > e+ ve is working"""

        try:
            shutil.rmtree(pjoin(MG5DIR,'TEST_MW_W_prod'))
        except Exception, error:
            pass
        
        cmd = """set automatic_html_opening False --no-save
                 set cluster_temp_path /tmp --no-save
                 generate p p > w+, w+ > e+ ve
                 output madweight TEST_MW_W_prod
                 launch
                 change_tf all_delta
                 ./tests/input_files/mw_wprod.lhco.gz
                 set nb_exp_events 4
                 set log_level weight
                 set nb_event_by_node 1
                 """
        open('/tmp/mg5_cmd','w').write(cmd)
        
        devnull =open(os.devnull,'w')
        subprocess.call([pjoin(MG5DIR,'bin','mg5'), 
                         '/tmp/mg5_cmd'],
                         cwd=pjoin(MG5DIR),
                        stdout=devnull, stderr=devnull)

        data = open(pjoin(MG5DIR, 'TEST_MW_W_prod', 'Events', 'fermi', 'weights.out')).read() 
        try:
            shutil.rmtree(pjoin(MG5DIR,'TEST_MW_W2J_prod'))
        except Exception, error:
            pass

        solution = self.get_result(data)
        expected = """# Weight (un-normalize) for each card/event
# format: LHCO_event_number card_id value integration_error
0 1 1.52322508477e-08 3.69373736836e-11
0 2 1.52322508477e-08 3.69373736836e-11
1 1 6.2231722171e-09 2.95094501214e-11
1 2 6.2231722171e-09 2.95094501214e-11
2 1 1.8932900739e-08 6.23556414283e-11
2 2 1.8932900739e-08 6.23556414283e-11
3 1 1.86550627721e-08 4.37562400224e-11
3 2 1.86550627721e-08 4.37562400224e-11
"""
        expected = self.get_result(expected)

        for key, (value,error) in expected.items():
            assert key in solution
            value2, error2 = solution[key]
            
            self.assertTrue(abs(value-value2) < 5* abs(error+error2))
            self.assertTrue(abs(value-value2)/abs(value+value2) < 0.01)
            self.assertTrue(abs(error2)/abs(value2) < 0.02)

    def test_short_mw_wjjproduction(self):
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
                 set nb_exp_events 4
                 set log_level weight
                 set nb_event_by_node 1
                 """
        open('/tmp/mg5_cmd','w').write(cmd)
        
        devnull =open(os.devnull,'w')
        subprocess.call([pjoin(MG5DIR,'bin','mg5'), 
                         '/tmp/mg5_cmd'],
                         cwd=pjoin(MG5DIR),
                        stdout=devnull, stderr=devnull)

        data = open(pjoin(MG5DIR, 'TEST_MW_W2J_prod', 'Events', 'fermi', 'weights.out')).read() 
        try:
            shutil.rmtree(pjoin(MG5DIR,'TEST_MW_W2J_prod'))
        except Exception, error:
            pass

        solution = self.get_result(data)
        expected = """# Weight (un-normalize) for each card/event
# format: LHCO_event_number card_id value integration_error
0 1 1.41756248942e-17 3.69590396941e-20
0 2 1.41756248942e-17 3.69590396941e-20
1 1 2.40167262714e-15 5.71647567991e-18
1 2 2.40167262714e-15 5.71647567991e-18
2 1 1.48907038945e-18 1.84546397304e-21
2 2 1.48907038945e-18 1.84546397304e-21
3 1 3.79640435481e-16 1.72128108188e-18
3 2 3.79640435481e-16 1.72128108188e-18
"""
        expected = self.get_result(expected)

        for key, (value,error) in expected.items():
            assert key in solution
            value2, error2 = solution[key]
            
            self.assertTrue(abs(value-value2) < 5* abs(error+error2))
            self.assertTrue(abs(value-value2)/abs(value+value2) < 0.01)
            self.assertTrue(abs(error2)/abs(value2) < 0.02)            
            
                
        


