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
import madgraph.iolibs.import_ufo as import_ufo
import models.model_reader as model_reader

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

#===============================================================================
# TestModelReader
#===============================================================================
class TestModelReader(unittest.TestCase):
    """Test class for the DecayModel object"""

    base_model = import_ufo.import_model('sm')

    def setUp(self):
        """Set up decay model"""
        #Full SM DecayModel
        self.model_reader = model_reader.ModelReader(self.base_model)

    def test_read_param_card(self):
        """Test reading a param card"""
        param_path = os.path.join(_file_path, '../input_files/param_card_sm.dat')
        self.model_reader.read_param_card(os.path.join(param_path))

        for param in sum([self.base_model.get('parameters')[key] for key \
                              in self.base_model.get('parameters')], []):
            print param.name, ": ",param.value
            value = param.value
            self.assertTrue(isinstance(value, int) or \
                            isinstance(value, float) or \
                            isinstance(value, complex),
                            "value is %s and type %s" % (value, type(value))) 
            
        for coupl in sum([self.base_model.get('couplings')[key] for key \
                              in self.base_model.get('couplings')], []):
            value = coupl.value
            print coupl.name, ": ",coupl.value
            self.assertTrue(isinstance(value, complex))     

if __name__ == '__main__':
    unittest.unittest.main()
