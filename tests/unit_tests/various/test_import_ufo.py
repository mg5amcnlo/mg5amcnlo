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
import madgraph.iolibs.export_v4 as export_v4

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
         
        
    def test_detect_special_parameters(self):
        """ check that detect zero parameters works"""        
        
        expected = set(['cabi', 'conjg__CKM13', 'conjg__CKM12', 'CKM21', 'conjg__CKM31', 'CKM23', 'WT', 'WTA', 'conjg__CKM32', 'ymc', 'ymb', 'Me', 'CKM32', 'CKM31', 'ym', 'CKM13', 'CKM12', 'yc', 'yb', 'ye', 'conjg__CKM21', 'conjg__CKM23', 'ys', 'MD', 'MC', 'MB', 'MM', 'yup', 'ydo', 'MU', 'MS', 'sin__cabi'])
        zero, one = self.model.detect_special_parameters()
        result = set(zero)
        self.assertEqual(expected, result)
        
        expected = set(['CKM33', 'conjg__CKM11', 'conjg__CKM33', 'CKM22', 'CKM11', 'cos__cabi', 'conjg__CKM22'])
        result = set(one)
        self.assertEqual(expected, result)

        
        
    def test_detect_identical_parameters(self):
        """ check that we detect correctly identical parameter """
        
        expected=set([('MZ','MH'), ('WZ','WH')])
        result = self.model.detect_identical_parameters()
        result = [tuple([obj.name for obj in obj_list]) for obj_list in result]
        
        self.assertEqual(expected, set(result))
        
    def test_merge_identical_parameters(self):
        """check that we treat correctly the identical parameters"""
        
        parameters = self.model.detect_identical_parameters()
        self.model.merge_iden_parameters(parameters[0])
        self.model.merge_iden_parameters(parameters[1])
        
        
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
        
        # checked that the mass (and the width) of those particles identical
        self.assertEqual(self.model['particle_dict'][23]['mass'],
                         self.model['particle_dict'][25]['mass'])
        self.assertEqual(self.model['particle_dict'][23]['width'],
                         self.model['particle_dict'][25]['width'])
        

        
    def test_detect_zero_iden_couplings(self):
        """ check that detect zero couplings works"""
        
        zero, iden = self.model.detect_identical_couplings()
        
        # check what is the zero coupling
        expected = set(['GC_86', 'GC_75', 'GC_74', 'GC_77', 'GC_76', 'GC_71', 'GC_70', 'GC_73', 'GC_72', 'GC_31', 'GC_30', 'GC_32', 'GC_79', 'GC_78', 'GC_88', 'GC_87', 'GC_89', 'GC_105', 'GC_126', 'GC_127', 'GC_124', 'GC_125', 'GC_123', 'GC_120', 'GC_121', 'GC_108', 'GC_109', 'GC_82', 'GC_83', 'GC_84', 'GC_85', 'GC_128', 'GC_129', 'GC_99', 'GC_106', 'GC_111', 'GC_98', 'GC_80', 'GC_28', 'GC_115', 'GC_81', 'GC_119', 'GC_26', 'GC_27', 'GC_68', 'GC_69', 'GC_114', 'GC_107', 'GC_113', 'GC_112', 'GC_135', 'GC_118', 'GC_131', 'GC_130', 'GC_133', 'GC_132', 'GC_117', 'GC_116', 'GC_95', 'GC_94', 'GC_93', 'GC_92', 'GC_91', 'GC_90'])
        result = set(zero)
        for name in result:
            self.assertEqual(self.model['coupling_dict'][name], 0)
        
        self.assertEqual(expected, result)        
        
        # check what are the identical coupling
        expected = [['GC_33', 'GC_29', 'GC_24', 'GC_25', 'GC_122', 'GC_134', 'GC_110'], ['GC_19', 'GC_66'], ['GC_18', 'GC_65'], ['GC_37', 'GC_12'], ['GC_49', 'GC_4'], ['GC_46', 'GC_62']]
        expected.sort()
        iden.sort()
        self.assertEqual(expected, iden)

    def test_locate_couplings(self):
        """ check the creation of the coupling to vertex dict """
        
        target = ['GC_33', 'GC_29', 'GC_24', 'GC_25', 'GC_122', 'GC_134', 'GC_110']
        sol = {'GC_33':[41], 
               'GC_29':[37],
               'GC_24':[63, 64, 65, 66, 67, 68], 
               'GC_25':[33], 
               'GC_122':[46], 
               'GC_134':[50], 
               'GC_110':[42]}
        
        self.model.locate_coupling()
        for coup in target:
            self.assertTrue(coup in self.model.coupling_pos)
            self.assertEqual(sol[coup], [v['id'] for v in self.model.coupling_pos[coup]])

  
    def test_merge_iden_couplings(self):
        """ check that the merged couplings are treated correctly:
             suppression and replacement in the vertex """
        
        self.model.locate_coupling()
        zero, iden = self.model.detect_identical_couplings()
        
        # Check that All the code/model is the one intended for this test
        target = ['GC_33', 'GC_29', 'GC_24', 'GC_25', 'GC_122', 'GC_134', 'GC_110']
        assert target in iden, 'test not up-to-date'
        vertex_id = [33,37,41,42,46,50,63,64,65,66,67,68]
        

        for id in vertex_id:
            is_in_target = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                if coup in target:
                    is_in_target = True
            assert is_in_target == True, 'test not up-to-date'
        
        # check now that everything is fine        
        self.model.merge_iden_couplings(target)
        for id in vertex_id:
            has_33 = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                self.assertFalse(coup in target[1:])
                if coup == 'GC_33':
                    has_33 = True
            self.assertTrue(has_33, True)

    def test_remove_couplings(self):
        """ check that the detection of irrelevant interactions works """
        
        # first test case where they are all deleted
        # check that we have the valid model
        input = self.model['interactions'][3]  # four gluon
        input2 = self.model['interactions'][28] # b b~ h
        self.assertTrue('GC_11' in input['couplings'].values())
        self.assertTrue('GC_68' in input2['couplings'].values())
        found_6 = 0
        found_34 = 0
        for dep,data in self.model['couplings'].items():
            for param in data:
                if param.name == 'GC_11': found_6 +=1
                elif param.name == 'GC_68': found_34 +=1
        self.assertTrue(found_6>0)
        self.assertTrue(found_34>0)
        
        # make the real test
        result = self.model.remove_couplings(['GC_68','GC_11'])
        
        for dep,data in self.model['couplings'].items():
            for param in data:
                self.assertFalse(param.name in  ['GC_11', 'GC_68'])

             
    def test_remove_interactions(self):
        """ check that the detection of irrelevant interactions works """
        
        
        # first test case where they are all deleted
        # check that we have the valid model
        input = self.model['interactions'][3]  # four gluon
        input2 = self.model['interactions'][28] # b b~ h
        self.assertTrue('GC_11' in input['couplings'].values())
        self.assertTrue('GC_68' in input2['couplings'].values())
        found_6 = 0
        found_34 = 0
        for dep,data in self.model['couplings'].items():
            for param in data:
                if param.name == 'GC_11': found_6 +=1
                elif param.name == 'GC_68': found_34 +=1
        self.assertTrue(found_6>0)
        self.assertTrue(found_34>0)
        
        # make the real test
        self.model.locate_coupling()
        result = self.model.remove_interactions(['GC_68','GC_11'])
        self.assertFalse(input in self.model['interactions'])
        self.assertFalse(input2 in self.model['interactions'])
        
    
        # Now test case where some of them are deleted and some not
        input = self.model['interactions'][29]  # d d~ Z
        input2 = self.model['interactions'][59] # e+ e- Z
        self.assertTrue('GC_47' in input['couplings'].values())
        self.assertTrue('GC_34' in input['couplings'].values())
        self.assertTrue('GC_34' in input2['couplings'].values())
        self.assertTrue('GC_48' in input2['couplings'].values())
        result = self.model.remove_interactions(['GC_34','GC_48'])
        input = self.model['interactions'][29]
        self.assertTrue('GC_47' in input['couplings'].values())
        self.assertFalse('GC_34' in input['couplings'].values())
        self.assertFalse('GC_34' in input2['couplings'].values())
        self.assertFalse('GC_48' in input2['couplings'].values())

    def test_put_parameters_to_zero(self):
        """check that we remove parameters correctly"""
        
        part_t = self.model.get_particle(6)
        # Check that we remove a mass correctly
        self.assertEqual(part_t['mass'], 'MT')
        self.model.fix_parameter_values(['MT'],[])
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
        self.model.fix_parameter_values(['WT'],[])
        self.assertEqual(part_t['width'], 'ZERO')
        for dep,data in self.model['parameters'].items():
            for param in data:
                self.assertNotEqual(param.name, 'WT')

        for pdg, particle in self.model['particle_dict'].items():
            self.assertNotEqual(particle['width'], 'WT')       
             
        # Check that we can remove correctly other external parameter
        self.model.fix_parameter_values(['ymb','yb'],[])
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
        assert interaction['couplings'] == {(0,0): 'GC_68'}
        
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
                self.assertFalse(param.name in  ['GC_68'])

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
       
class TestBenchmarkModel(unittest.TestCase):
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
        
        
    def test_use_as_benchmark(self):
        """check that the value inside the restrict card overwritte the default
        parameter such that this option can be use for benchmark point"""
        
        params_ext = self.model['parameters'][('external',)]
        value = {}
        [value.__setitem__(data.name, data.value) for data in params_ext] 
        self.model.restrict_model(self.restrict_file)
        #use the UFO -> MG4 converter class
        
        params_ext = self.model['parameters'][('external',)]
        value2 = {}
        [value2.__setitem__(data.name, data.value) for data in params_ext] 
        
        self.assertNotEqual(value['WW'], value2['WW'])
                
    def test_model_name(self):
        """ test that the model name is correctly set """
        self.assertEqual(self.base_model["name"], "sm")
        model = import_ufo.import_model('sm-full') 
        self.assertEqual(model["name"], "sm-full")
        model = import_ufo.import_model('sm-no_b_mass') 
        self.assertEqual(model["name"], "sm-no_b_mass")        

