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
"""Unit test Library for importing and restricting model"""
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
# TestRestrictModel
#===============================================================================
class TestRestrictModel(unittest.TestCase):
    """Test class for the RestrictModel object"""

    sm_path = import_ufo.find_ufo_path('sm')
    base_model = import_ufo.import_full_model(sm_path)

    def setUp(self):
        """Set up decay model"""
        #Read the full SM
        model = copy.deepcopy(self.base_model)
        self.model = import_ufo.RestrictModel(model)
        self.restrict_file = os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'restrict_sm.dat')
        self.model.set_parameters_and_couplings(self.restrict_file)
         
        
    def test_detect_zero_parameters(self):
        """ check that detect zero parameters works"""        
        
        expected = set(['cabi', 'conjg__CKM13', 'conjg__CKM12', 'CKM21', 'conjg__CKM31', 'CKM23', 'WT', 'conjg__CKM32', 'ymc', 'ymb', 'Me', 'CKM32', 'CKM31', 'ym', 'CKM13', 'CKM12', 'yc', 'yb', 'ye', 'conjg__CKM21', 'conjg__CKM23', 'ys', 'MD', 'MC', 'MB', 'MM', 'yup', 'ydo', 'MU', 'MS', 'sin__cabi'])
        result = set(self.model.detect_zero_parameters())

        self.assertEqual(expected, result)
        
    def test_detect_identical_parameters(self):
        """ check that we detect correctly identical parameter """
        
        expected=set([('MZ','MH')])
        result = self.model.detect_identical_parameters()
        result = [tuple([obj.name for obj in obj_list]) for obj_list in result]
        
        self.assertEqual(expected, set(result))
        
    def test_merge_identical_parameters(self):
        """check that we treat correctly the identical parameters"""
        
        parameters = self.model.detect_identical_parameters()
        self.model.merge_identical_parameters(parameters[0])
        
        
        #check that both MZ and MH are not anymore in the external_parameter
        keeped = parameters[0][0].name
        removed = parameters[0][1].name
        for dep,data in self.model['parameters'].items():
            if dep == ('external'):
                for param in data:
                    self.assertNotEqual(param.name, removed)
            elif dep == ():
                found=0      
                for param in data:
                    if removed == param.name:
                        found += 1
                        self.assertEqual(param.expr, keeped)
                self.assertEqual(found, 1)

        
    def test_detect_zero_couplings(self):
        """ check that detect zero couplings works"""
        

        expected = set(['GC_16', 'GC_15', 'GC_14', 'GC_46', 'GC_20', 'GC_19', 'GC_18', 'GC_39', 'GC_38', 'GC_50', 'GC_48', 'GC_42', 'GC_44', 'GC_45', 'GC_49', 'GC_35', 'GC_34', 'GC_37', 'GC_36'])
        result = set(self.model.detect_zero_couplings())
        
        self.assertEqual(expected, result)        
        
        
    def test_remove_couplings(self):
        """ check that the detection of irrelevant interactions works """
        
        # first test case where they are all deleted
        # check that we have the valid model
        input = self.model['interactions'][3]  # four gluon
        input2 = self.model['interactions'][28] # b b~ h
        self.assertTrue('GC_6' in input['couplings'].values())
        self.assertTrue('GC_34' in input2['couplings'].values())
        found_6 = 0
        found_34 = 0
        for dep,data in self.model['couplings'].items():
            for param in data:
                if param.name == 'GC_6': found_6 +=1
                elif param.name == 'GC_34': found_34 +=1
        self.assertTrue(found_6>0)
        self.assertTrue(found_34>0)
        
        # make the real test
        result = self.model.remove_couplings(['GC_34','GC_6'])
        self.assertFalse(input in self.model['interactions'])
        self.assertFalse(input2 in self.model['interactions'])
        
        for dep,data in self.model['couplings'].items():
            for param in data:
                self.assertFalse(param.name in  ['GC_6', 'GC_34'])
        
        # Now test case where some of them are deleted and some not
        input = self.model['interactions'][29]  # d d~ Z
        input2 = self.model['interactions'][59] # e+ e- Z
        self.assertTrue('GC_24' in input['couplings'].values())
        self.assertTrue('GC_22' in input['couplings'].values())
        self.assertTrue('GC_22' in input2['couplings'].values())
        self.assertTrue('GC_25' in input2['couplings'].values())
        result = self.model.remove_couplings(['GC_22','GC_25'])
        input = self.model['interactions'][29]
        self.assertTrue('GC_24' in input['couplings'].values())
        self.assertFalse('GC_22' in input['couplings'].values())
        self.assertFalse('GC_25' in input2['couplings'].values())
        self.assertFalse('GC_22' in input2['couplings'].values())

    def test_put_parameters_to_zero(self):
        """check that we remove parameters correctly"""
        
        part_t = self.model.get_particle(6)
        # Check that we remove a mass correctly
        self.assertEqual(part_t['mass'], 'MT')
        self.model.put_parameters_to_zero(['MT'])
        self.assertEqual(part_t['mass'], 'ZERO')
        for dep,data in self.model['parameters'].items():
            for param in data:
                self.assertNotEqual(param.name, 'MT')
        
        for particle in self.model['particles']:
            self.assertNotEqual(particle['mass'], 'MT')
                    
        for pdg, particle in self.model['particle_dict'].items():
            self.assertNotEqual(particle['mass'], 'MT')
        
        # Check that we remove a width correctly
        self.assertEqual(part_t['width'], 'WT')
        self.model.put_parameters_to_zero(['WT'])
        self.assertEqual(part_t['width'], 'ZERO')
        for dep,data in self.model['parameters'].items():
            for param in data:
                self.assertNotEqual(param.name, 'WT')

        for pdg, particle in self.model['particle_dict'].items():
            self.assertNotEqual(particle['width'], 'WT')        
        # Check that we can remove correctly other external parameter
        self.model.put_parameters_to_zero(['ymb','yb'])
        for dep,data in self.model['parameters'].items():
            for param in data:
                self.assertFalse(param.name in  ['ymb'])
                if param.name == 'yb':
                    param.expr == 'ZERO'
                        
    def test_restrict_from_a_param_card(self):
        """ check the full restriction chain in one case b b~ h """
        
        interaction = self.model['interactions'][28]
        #check sanity of the checks
        assert [p['pdg_code'] for p in interaction['particles']] == [5, 5, 25], \
             'Initial model not valid for this test => update the test'
        assert interaction['couplings'] == {(0,0): 'GC_34'}
        
        self.model.restrict_model(self.restrict_file)
        
        # check remove interactions
        self.assertFalse(interaction in self.model['interactions'])
        
        # check remove parameters
        for dep,data in self.model['parameters'].items():
            for param in data:
                self.assertFalse(param.name in  ['yb','ymb','MB','WT'])

        # check remove couplings
        for dep,data in self.model['couplings'].items():
            for param in data:
                self.assertFalse(param.name in  ['GC_34'])

        # check masses
        part_b = self.model.get_particle(5)
        part_t = self.model.get_particle(6)
        self.assertEqual(part_b['mass'], 'ZERO')
        self.assertEqual(part_t['width'], 'ZERO')
                
        # check identical masses
        keeped, rejected = None, None 
        for param in self.model['parameters'][('external',)]:
            if param.name == 'MH':
                self.assertEqual(keeped, None)
                keeped, rejected = 'MH','MZ'
            elif param.name == 'MZ':
                self.assertEqual(keeped, None)
                keeped, rejected = 'MZ','MH'
                
        self.assertNotEqual(keeped, None)
        
        found = 0
        for param in self.model['parameters'][()]:
            self.assertNotEqual(param.name, keeped)
            if param.name == rejected:
                found +=1
        self.assertEqual(found, 1)
                
        