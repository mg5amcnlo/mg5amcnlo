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
from tests.parallel_tests import me_comparator

"""Parallel tests comparing processes between MG4 and MG5.
"""

import logging
import os

import me_comparator
import unittest

# Get full logging info
logging.basicConfig(level=logging.INFO)
_file_path = os.path.dirname(os.path.realpath(__file__))
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR

class TestParallelMG4MG5(unittest.TestCase):

    mg4runner = me_comparator.MG4Runner
    mg5runner = me_comparator.MG5Runner

    def setUp(self):
        """Set up paths"""
        # specify the position of different codes
        self.mg4_path = MG4DIR
        self.mg5_path = MG5DIR
        
    def test_mg4_mg5_minitest(self):
        """Test a minimal list of sm 2->2 processes, mainly to test the test"""
        # Create a list of processes to check automatically
        my_proc_list = me_comparator.create_proc_list(\
            ['u'],
            initial=2, final=2)

        # Store list of non-zero processes and results in file
        pickle_file = "mg4_sm_minitest.pkl"
        self.compare_MG4_MG5(my_proc_list,
                             orders = {'QED':2, 'QCD':2},
                             filename = "sm_mini.log",
                             pickle_file = pickle_file)

    def test_mg4_mg5_sm_22(self):
        """Test a semi-complete list of sm 2->2 processes"""
        # Create a list of processes to check automatically
        my_proc_list = me_comparator.create_proc_list(\
            ['w+', 'w-', 'a', 'z', 'h', 'g', 'u', 'u~', 'd', 'd~',
            'b', 'b~', 't', 't~', 'ta+', 'ta-', 'vt', 'vt~'],
            initial=2, final=2)

        # Store list of non-zero processes and results in file
        pickle_file = "mg4_sm_22.pkl"
        self.compare_MG4_MG5(my_proc_list,
                             orders = {'QED':2, 'QCD':2},
                             filename = "sm_22.log",
                             pickle_file = pickle_file)


    def test_mg4_mg5_sm_23(self):
        """Test a semi-complete list of sm 2->3 processes"""
        # Create a list of processes to check automatically
        my_proc_list = me_comparator.create_proc_list(\
            ['w+', 'w-','a', 'z', 'h', 'g', 'u', 'u~', 'd', 'd~',
             'b', 'b~', 't', 't~', 'ta+', 'ta-', 'vt', 'vt~'],
            initial=2, final=3)

        # Store list of non-zero processes and results in file
        pickle_file = "mg4_sm_23.pkl"
        self.compare_MG4_MG5(my_proc_list,
                             orders = {'QED':3, 'QCD':3},
                             filename = "sm_23.log",
                             pickle_file = pickle_file)


    def test_mg4_mg5_mssm_22(self):
        """Test a semi-complete list of mssm 2->2 processes"""
        # Create a list of processes to check automatically
        sm_parts = ['w+', 'w-', 'a', 'z', 'h1', 'h+', 'h-', 'g', 'u', 'u~',
            'd', 'd~', 'b', 'b~', 't', 't~', 'ta+', 'ta-', 'vt', 'vt~']
        mssm_parts = ['dl', 'dl~', 'dr', 'dr~', 'ul', 'ul~', 'ur', 'ur~', 'b1',
                      'b1~', 'b2', 'b2~', 't1', 't1~', 'ta1-', 'ta1+', 'ta2-',
                      'ta2+', 'svt', 'svt~', 'x1-', 'x1+', 'x2-', 'x2+',
                      'go', 'n1']
        # Generate 2 -> 2 processes, with MSSM particles in pairs in
        # final state
        my_proc_list = me_comparator.create_proc_list_enhanced(\
            sm_parts, mssm_parts)

        # Store list of non-zero processes and results in file
        pickle_file = "mg4_mssm_22.pkl"
        self.compare_MG4_MG5(my_proc_list,
                             orders = {'QED':2, 'QCD':2},
                             model = "mssm",
                             energy = 2000,
                             filename = "mssm_22.log",
                             pickle_file = pickle_file)

    def test_mg4_mg5_mssm_23(self):
        """Test a semi-complete list of mssm 2->3 processes"""
        # Create a list of processes to check automatically
        sm_parts = ['w+', 'w-', 'a', 'z', 'h1', 'h+', 'h-', 'g', 'u', 'u~',
            'd', 'd~', 'b', 'b~', 't', 't~', 'ta+', 'ta-', 'vt', 'vt~']
        mssm_parts = ['dl', 'dl~', 'dr', 'dr~', 'ul', 'ul~', 'ur', 'ur~', 'b1',
                      'b1~', 'b2', 'b2~', 't1', 't1~', 'ta1-', 'ta1+', 'ta2-',
                      'ta2+', 'svt', 'svt~', 'x1-', 'x1+', 'x2-', 'x2+',
                      'go', 'n1']
        # Generate 2 -> 2+1 processes, with MSSM particles in pairs in
        # final state
        my_proc_list = me_comparator.create_proc_list_enhanced(\
            sm_parts, mssm_parts, sm_parts)

        # Store list of non-zero processes and results in file
        pickle_file = "mg4_mssm_23.pkl"
        self.compare_MG4_MG5(my_proc_list,
                             orders = {'QED':3, 'QCD':3},
                             model = "mssm",
                             energy = 2000,
                             filename = "mssm_23.log",
                             pickle_file = pickle_file)

    def test_mg4_mg5_heft_23(self):
        """Test a heft 2->3 processes"""
        # Create a list of processes to check automatically
        sm_parts = ['g', 'a', 'w+', 'w-']
        heft_parts = ['h']
        # Generate 2 -> 1+2 processes, with one Higgs in final state
        # final state
        my_proc_list = me_comparator.create_proc_list_enhanced(\
            sm_parts, sm_parts, heft_parts)

        # Store list of non-zero processes and results in file
        pickle_file = "mg4_heft_23.pkl"
        self.compare_MG4_MG5(my_proc_list,
                             orders = {'QED':2, 'QCD':0, 'HIG':1, 'HIW':1},
                             model = "heft",
                             energy = 500,
                             filename = "heft_23.log",
                             pickle_file = pickle_file)


    def compare_MG4_MG5(self, my_proc_list = [], orders = {}, model = 'sm',
                        energy = 500, filename = "", pickle_file = "",
                        tolerance = 1e-06):
        """Run comparison between MG4 and MG5 for the list of processes"""

        # Create a MERunner object for MG4
        my_mg4 = self.mg4runner()
        my_mg4.setup(self.mg4_path)

        # Create a MERunner object for MG5
        my_mg5 = self.mg5runner()
        my_mg5.setup(self.mg5_path, self.mg4_path)

        # Create and setup a comparator
        my_comp = me_comparator.MEComparator()
        my_comp.set_me_runners(my_mg4, my_mg5)

        # Run the actual comparison
        my_comp.run_comparison(my_proc_list,
                               model, orders, energy)

        # Print the output
        if filename:
            my_comp.output_result(filename=filename)

        # Store output to a pickle file in the input_files directory
        if pickle_file:
            me_comparator.PickleRunner.store_comparison(\
                os.path.join(_pickle_path, pickle_file),
                my_comp.get_non_zero_processes(),
                my_comp.me_runners[0].model,
                my_comp.me_runners[0].name,
                my_comp.me_runners[0].orders,
                my_comp.me_runners[0].energy)

        # Assert that all process comparisons passed the tolerance cut
        my_comp.assert_processes(self, tolerance)
            
        # Do some cleanup
        my_comp.cleanup()

class TestParallelMG4MG5_UFO(TestParallelMG4MG5):
    
    mg5runner = me_comparator.MG5_UFO_Runner
    
    def test_mg4_mg5_minitest_ufo(self):
        """Test a minimal list of sm 2->2 processes, mainly to test the test"""
        # Create a list of processes to check automatically
        my_proc_list = me_comparator.create_proc_list(\
            ['a'],
            initial=2, final=2)

        # Store list of non-zero processes and results in file
        pickle_file = "mg4_sm_ufo_minitest.pkl"
        self.compare_MG4_MG5(my_proc_list,
                             orders = {'QED':2, 'QCD':2},
                             filename = "sm_ufo_mini.log",
                             pickle_file = pickle_file)
