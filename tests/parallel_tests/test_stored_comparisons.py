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

"""Parallel tests comparing sm processes between MG4 and MG5.
"""

import logging
import os

import me_comparator
import unittest
from madgraph import MG4DIR, MG5DIR

# Get full logging info
logging.basicConfig(level=logging.INFO)
_file_path = os.path.dirname(os.path.realpath(__file__))
_pickle_path =os.path.join(_file_path, 'input_files')

class TestParallelPickle(unittest.TestCase):
    """Test MG5 against stored comparisons."""

    def setUp(self):
        """Set up paths and load comparisons"""
        # specify the position of different codes
        self.mg4_path = MG4DIR
        self.mg5_path = MG5DIR
        
    def test_all_stored_comparisons(self):
        """Test MG5 against all stored comparisons, one after the other."""

        comparisons = me_comparator.PickleRunner.find_comparisons(_pickle_path,
                                                                  model = "")
        for stored_runner in comparisons:

            # Create a MERunner object for MG5
            my_mg5 = me_comparator.MG5Runner()
            my_mg5.setup(self.mg5_path, self.mg4_path)

            # Create and setup a comparator
            my_comp = me_comparator.MEComparator()
            my_comp.set_me_runners(stored_runner, my_mg5)

            # Run the actual comparison
            my_comp.run_comparison(stored_runner.proc_list,
                                   stored_runner.model,
                                   stored_runner.orders,
                                   stored_runner.energy)

            my_comp.assert_processes(self)

            # Do some cleanup
            my_comp.cleanup()

