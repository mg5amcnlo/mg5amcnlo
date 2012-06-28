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
"""Unit test Library for the objects in decay module."""
from __future__ import division

import copy
import os
import sys
import time

import tests.unit_tests as unittest
import madgraph.core.base_objects as base_objects
import models.import_ufo as import_ufo
import models.model_reader as model_reader

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

#===============================================================================
# TestModelReader
#===============================================================================
class TestModelReader(unittest.TestCase):
    """Test class for the ModelReader object"""

    sm_path = import_ufo.find_ufo_path('sm')
    base_model = import_ufo.import_model(sm_path)

    def setUp(self):
        """Set up decay model"""
        #Read the full SM
        self.model_reader = model_reader.ModelReader(self.base_model)

    def test_set_parameters_and_couplings(self):
        """Test reading a param card"""
        param_path = os.path.join(_file_path, '../input_files/param_card_sm.dat')
        self.model_reader.set_parameters_and_couplings(os.path.join(param_path))

        for param in sum([self.base_model.get('parameters')[key] for key \
                              in self.base_model.get('parameters')], []):
            value = param.value
            self.assertTrue(isinstance(value, (complex, float, int)))
            self.assertTrue(isinstance(self.model_reader.get('parameter_dict')[\
                param.name], (complex, float, int)))

            
        for coupl in sum([self.base_model.get('couplings')[key] for key \
                              in self.base_model.get('couplings')], []):
            value = coupl.value
            self.assertTrue(isinstance(value, complex))     

            self.assertTrue(isinstance(self.model_reader.get('coupling_dict')[\
                coupl.name], complex))
            

if __name__ == '__main__':
    unittest.unittest.main()
