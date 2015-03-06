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
            if not len(split) in [4,5]:
                continue
            if len(split) ==4:
                event_nb, card_nb, weight, error = map(float, split)
                tf_set = 1.
            else:
                event_nb, card_nb, tf_set, weight, error = map(float, split)
            
            solution[(event_nb,card_nb,tf_set)] = (weight,error)
        return solution

    def test_short_mw_tt_full_lept(self):
        """checking that the weight for p p > t t~ fulllept is working"""

        try:
            shutil.rmtree(pjoin(MG5DIR,'TEST_MW_TT_prod'))
        except Exception, error:
            pass
        
        cmd = """set automatic_html_opening False --no-save
                 set cluster_temp_path /tmp --no-save
                 generate p p > t t~, (t > w+ b, w+ > e+ ve), (t~ > w- b~, w- > e- ve~)
                 output madweight TEST_MW_TT_prod_full
                 launch
                 change_tf dbl_gauss_pt_jet
                 ./tests/input_files/mw_ttprod.lhco.gz
                 set nb_exp_events 2
                 set log_level debug
                 set nb_event_by_node 1
                 set mw_perm montecarlo F
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

        start = time.time()
        print 'this mw test is expected to take 30s on two core. (MBP retina 2012) current time: %02dh%02d' % (time.localtime().tm_hour, time.localtime().tm_min) 
        subprocess.call([pjoin(MG5DIR,'bin','mg5'), 
                         '/tmp/mg5_cmd'],
                         cwd=pjoin(MG5DIR),
                        stdout=stdout, stderr=stderr)
        run_time =  time.time() - start
        print 'tt~ full takes %smin %is' % (run_time//60, run_time % 60)
        data = open(pjoin(MG5DIR, 'TEST_MW_TT_prod_full', 'Events', 'fermi', 'weights.out')).read()

        solution = self.get_result(data)
        expected = """# Weight (un-normalize) for each card/event
# format: LHCO_event_number card_id value integration_error
24 1 7.13353274182e-22 3.48595541497e-24 
24 2 4.48106063562e-22 2.23501639194e-24 
28 1 1.22526200347e-23 5.8955444892e-26 
28 2 5.53271960779e-23 2.59251688524e-25 
"""   
        expected = self.get_result(expected)
        for key, (value,error) in expected.items():
            assert key in solution
            value2, error2 = solution[key]
            self.assertTrue(abs(value-value2) < 5* abs(error+error2))
            self.assertTrue(abs(value-value2)/abs(value+value2) < 2*abs(value/error))
            self.assertTrue(abs(error2)/abs(value2) < 0.02)
        #try:
        #    shutil.rmtree(pjoin(MG5DIR,'TEST_MW_TT_prod_full'))
        #except Exception, error:
        #    pass
            
    def test_short_mw_tt_semi(self):
        """checking that the weight for p p > t t~ semilept is working"""

        try:
            shutil.rmtree(pjoin(MG5DIR,'TEST_MW_TT_prod'))
        except Exception, error:
            pass
        
        cmd = """set automatic_html_opening False --no-save
                 set cluster_temp_path /tmp --no-save
                 generate p p > t t~, (t > w+ b, w+ > l+ vl), (t~ > w- b~, w- > j j)
                 output madweight TEST_MW_TT_prod -f
                 launch
                 change_tf dbl_gauss_pt_jet
                 ./tests/input_files/mw_ttprod.lhco.gz
                 set nb_exp_events 1
                 set log_level debug
                 set nb_event_by_node 1
                 set mw_run pretrained T
                 set mw_perm montecarlo T
                 set mw_run MW_int_points 1000
                 set mw_run MW_int_refine 8000
                 set mw_run  use_sobol T
                 set mw_gen force_nwa 2
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
        
        start = time.time()
        print 'this mw test is expected to take 2 min on two core. (MBP retina 2012) current time: %02dh%02d' % (time.localtime().tm_hour, time.localtime().tm_min) 
        subprocess.call([pjoin(MG5DIR,'bin','mg5'), 
                         '/tmp/mg5_cmd'],
                         cwd=pjoin(MG5DIR),
                        stdout=stdout, stderr=stderr)
        run_time =  time.time() - start
        print 'tt~ semi takes %smin %is' % (run_time//60, run_time % 60)
        data = open(pjoin(MG5DIR, 'TEST_MW_TT_prod', 'Events', 'fermi', 'weights.out')).read()


        solution = self.get_result(data)
        expected = """# Weight (un-normalize) for each card/event
# format: LHCO_event_number card_id value integration_error
2 1 1.06068538348e-23 6.39167252183e-26
2 2 5.59862383052e-24 3.76145999572e-26
9 1 6.92859060639e-25 6.04804891841e-27
9 2 6.79399430333e-25 7.38824630883e-27
"""
        expected = self.get_result(expected)
        
        for key, (value,error) in expected.items():
            assert key in solution
            value2, error2 = solution[key]
            self.assertTrue(abs(value-value2) < 5* abs(error+error2))
            self.assertTrue(abs(value-value2)/abs(value+value2) < 2*abs(value/error))
            self.assertTrue(abs(error2)/abs(value2) < 0.02)
        #try:
        #    shutil.rmtree(pjoin(MG5DIR,'TEST_MW_TT_prod'))
        #except Exception, error:
        #    pass


    def test_short_mw_wa_refine(self):
        """checking that the weight for p p > w a, w > l- is working"""

        try:
            shutil.rmtree(pjoin(MG5DIR,'TEST_MW_WA_prod'))
        except Exception, error:
            pass
        
        cmd = """set automatic_html_opening False --no-save
                 set cluster_temp_path /tmp --no-save
                 generate p p > w- a , w- > e- ve~
                 output madweight TEST_MW_WA_prod -f
                 launch
                 change_tf dbl_gauss_pt_jet
                 ./tests/input_files/mw_wa_prod.lhco
                 set nb_exp_events 1
                 set log_level debug
                 set nb_event_by_node 1
                 set mw_parameter 12 23
                 set mw_parameter 13 80 90
                 set mw_run MW_int_points 100
                 set mw_run MW_int_refine 100
                 launch -i
                 refine 0.01
                 set mw_run MW_int_points 1000
                 set mw_run MW_int_refine 10000
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
        
        start = time.time()
        print 'this mw test is expected to take 15s on two core. (MBP retina 2012) current time: %02dh%02d' % (time.localtime().tm_hour, time.localtime().tm_min) 
        subprocess.call([pjoin(MG5DIR,'bin','mg5'), 
                         '/tmp/mg5_cmd'],
                         cwd=pjoin(MG5DIR),
                        stdout=stdout, stderr=stderr)
        run_time =  time.time() - start
        print 'wa takes %smin %is' % (run_time//60, run_time % 60)
        data = open(pjoin(MG5DIR, 'TEST_MW_WA_prod', 'Events', 'fermi', 'weights.out')).read()


        solution = self.get_result(data)
        expected = """# Weight (un-normalize) for each card/event
# format: LHCO_event_number card_id value integration_error
# Weight (un-normalize) for each card/event
# format: LHCO_event_number card_id value integration_error
0 1 2.68641824739e-14 1.75587340837e-17
0 2 1.10047493409e-13 4.9103491463e-16
"""
        expected = self.get_result(expected)
        for key, (value,error) in expected.items():
            assert key in solution
            value2, error2 = solution[key]
            
            self.assertTrue(abs(value-value2) < 5* abs(error+error2))
            self.assertTrue(abs(value-value2)/abs(value+value2) < 2*abs(value/error))
            self.assertTrue(abs(error2)/abs(value2) < 0.02)
        try:
            shutil.rmtree(pjoin(MG5DIR,'TEST_MW_WA_prod'))
        except Exception, error:
            pass
  