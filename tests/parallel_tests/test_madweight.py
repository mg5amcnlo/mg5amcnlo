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
            
            
    def test_short_mw_tt_semi(self):
        """checking that the weight for p p > t t~ semilept is working"""

        try:
            shutil.rmtree(pjoin(MG5DIR,'TEST_MW_TT_prod'))
        except Exception, error:
            pass
        
        cmd = """set automatic_html_opening False --no-save
                 set cluster_temp_path /tmp --no-save
                 generate p p > t t~, (t > w+ b, w+ > l+ vl), (t~ > w- b~, w- > j j)
                 output madweight TEST_MW_TT_prod
                 launch
                 change_tf dbl_gauss_pt_jet
                 ./tests/input_files/mw_ttprod.lhco.gz
                 set nb_exp_events 1
                 set log_level debug
                 set nb_event_by_node 1
                 """
        open('/tmp/mg5_cmd','w').write(cmd)
        
        devnull =open(os.devnull,'w')
        start = time.time()
        print 'this mw test is expected to take 3 min on two core. (MBP retina 2012) current time: %02dh%02d' % (time.localtime().tm_hour, time.localtime().tm_min) 
        subprocess.call([pjoin(MG5DIR,'bin','mg5'), 
                         '/tmp/mg5_cmd'],
                         cwd=pjoin(MG5DIR),
                        stdout=devnull, stderr=devnull)
        run_time =  time.time() - start
        print 'tt~ semi takes %smin %is' % (run_time//60, run_time % 60)
        data = open(pjoin(MG5DIR, 'TEST_MW_TT_prod', 'Events', 'fermi', 'weights.out')).read()


        solution = self.get_result(data)
        expected = """# Weight (un-normalize) for each card/event
# format: LHCO_event_number card_id value integration_error
2 1 2.10407296143e-23 2.23193727155e-25 
2 2 1.10783659513e-23 1.45410120216e-25
"""
        expected = self.get_result(expected)
        for key, (value,error) in expected.items():
            assert key in solution
            value2, error2 = solution[key]
            
            self.assertTrue(abs(value-value2) < 5* abs(error+error2))
            self.assertTrue(abs(value-value2)/abs(value+value2) < 2*abs(value/error))
            self.assertTrue(abs(error2)/abs(value2) < 0.02)
        try:
            shutil.rmtree(pjoin(MG5DIR,'TEST_MW_TT_prod'))
        except Exception, error:
            pass
  