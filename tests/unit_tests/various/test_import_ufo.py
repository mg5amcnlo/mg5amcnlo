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
# TestImportUFO
#===============================================================================
class TestImportUFO(unittest.TestCase):
    """Test class for the RestrictModel object"""

    def setUp(self):
        """Set up decay model"""
        #Read the full SM
        sm_path = import_ufo.find_ufo_path('heft')
        self.base_model = import_ufo.import_full_model(sm_path)

    def test_coupling_hierarchy(self):
        """Test that the coupling_hierarchy is set"""
        self.assertEqual(self.base_model.get('order_hierarchy'),
                         {'QCD': 1, 'QED': 2, 'HIG':2, 'HIW': 2})
         
    def test_expansion_order(self):
        """Test that the expansion_order is set"""
        self.assertEqual(self.base_model.get('expansion_order'),
                         {'QCD': 99, 'QED': 99, 'HIG':1, 'HIW': 1})

class TestNFlav(unittest.TestCase):
    """Test class for the get_nflav function"""

    def test_get_nflav_sm(self):
        """Tests the get_nflav_function for the full SM.
        Here b and c quark are massive"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_full_model(sm_path)
        self.assertEqual(model.get_nflav(), 3)

    def test_get_nflav_sm_nobmass(self):
        """Tests the get_nflav_function for the SM, with the no-b-mass restriction"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_model(sm_path + '-no_b_mass')
        self.assertEqual(model.get_nflav(), 5)


    def test_get_nflav_sm_nomasses(self):
        """Tests the get_nflav_function for the SM, with the no_masses restriction"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_model(sm_path + '-no_masses')
        self.assertEqual(model.get_nflav(), 5)

#===============================================================================
# TestRestrictModel
#===============================================================================
class TestRestrictModel(unittest.TestCase):
    """Test class for the RestrictModel object"""

    def setUp(self):
        """Set up decay model"""
        #Read the full SM
        sm_path = import_ufo.find_ufo_path('sm')
        self.base_model = import_ufo.import_full_model(sm_path)

        model = copy.deepcopy(self.base_model)
        self.model = import_ufo.RestrictModel(model)
        self.restrict_file = os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'restrict_sm.dat')
        self.model.set_parameters_and_couplings(self.restrict_file)
         
        
    def test_detect_special_parameters(self):
        """ check that detect zero parameters works"""        
        
        expected = set(['etaWS', 'conjg__CKM13', 'conjg__CKM12', 'conjg__CKM32', 'conjg__CKM31', 'CKM23', 'WT', 'lamWS', 'WTau', 'AWS', 'ymc', 'ymb', 'yme', 'ymm', 'Me', 'MB', 'CKM32', 'CKM31', 'ym', 'CKM13', 'CKM12', 'lamWS__exp__2', 'lamWS__exp__3', 'yc', 'yb', 'ye', 'conjg__CKM21', 'CKM21', 'conjg__CKM23', 'MC', 'MM', 'rhoWS'])
        zero, one = self.model.detect_special_parameters()
        result = set(zero)
        self.assertEqual(expected, result)
        
        expected = set(['CKM33', 'conjg__CKM11', 'conjg__CKM33', 'CKM22', 'CKM11', 'conjg__CKM22'])
        result = set(one)
        self.assertEqual(expected, result)

        
        
    def test_detect_identical_parameters(self):
        """ check that we detect correctly identical parameter """
        
        expected=set([('MZ','MH'), ('WZ','WH')])
        result = self.model.detect_identical_parameters()
        result = [tuple([obj[0].name for obj in obj_list]) for obj_list in result]
        
        self.assertEqual(expected, set(result))
        
    def test_merge_identical_parameters(self):
        """check that we treat correctly the identical parameters"""
        
        parameters = self.model.detect_identical_parameters()
        self.model.merge_iden_parameters(parameters[0])
        self.model.merge_iden_parameters(parameters[1])
        
        
        #check that both MZ and MH are not anymore in the external_parameter
        keeped = '1*%s' % parameters[0][0][0].name
        removed = parameters[0][1][0].name
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
        expected = set(['GC_100', 'GC_108', 'GC_109', 'GC_31', 'GC_30', 'GC_32', 'GC_28', 'GC_26', 'GC_27', 'GC_99', 'GC_98', 'GC_97', 'GC_96', 'GC_75', 'GC_74', 'GC_77', 'GC_76', 'GC_71', 'GC_70', 'GC_73', 'GC_72', 'GC_79', 'GC_78', 'GC_88', 'GC_89', 'GC_102', 'GC_103', 'GC_104', 'GC_105', 'GC_106', 'GC_107', 'GC_80', 'GC_81', 'GC_82', 'GC_83', 'GC_84', 'GC_85', 'GC_68', 'GC_69', 'GC_111'])
        result = set(zero)
        for name in result:
            self.assertEqual(self.model['coupling_dict'][name], 0)
        
        self.assertEqual(expected, result)        
        
        # check what are the identical coupling
        expected = [['GC_101', 'GC_33', 'GC_29', 'GC_24', 'GC_25', 'GC_95', 'GC_110'], ['GC_19', 'GC_66'], ['GC_18', 'GC_65'], ['GC_37', 'GC_12'], ['GC_49', 'GC_4'], ['GC_46', 'GC_62']]
        expected.sort()
        iden.sort()
        self.assertEqual(expected, iden)

    def test_locate_couplings(self):
        """ check the creation of the coupling to vertex dict """
        
        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [5, 5, 25]:
                input_bbh = candidate
                coupling_bbh = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [23, 23, 25, 25]:
                input_zzhh = candidate
                coupling_zzhh = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [11, 12, 24]:
                input_wen = candidate
                coupling_wen = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [22, 24, 24]:
                input_aww = candidate
                coupling_aww = candidate['couplings'][(0,0)]            
        
        
        target = [coupling_bbh, coupling_zzhh, coupling_wen, coupling_aww]
        sol = {coupling_bbh: [input_bbh['id']],
               coupling_zzhh: [input_zzhh['id']],
               coupling_wen: [43, 44, 45, 66, 67, 68],
               coupling_aww: [input_aww['id']]}
        # b b~ h // z z h h //w- e+ ve // a w+ w-
        
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
        target = ['GC_101', 'GC_33', 'GC_29', 'GC_24', 'GC_25', 'GC_95', 'GC_110']

        assert target in iden, 'test not up-to-date'
        check_content = [['d', 'u', 'w+'], ['s', 'c', 'w+'], ['b', 't', 'w+'], ['u', 'd', 'w+'], ['c', 's', 'w+'], ['t', 'b', 'w+'], ['e-', 've', 'w+'], ['m-', 'vm', 'w+'], ['tt-', 'vt', 'w+'], ['ve', 'e-', 'w+'], ['vm', 'm-', 'w+'], ['vt', 'tt-', 'w+']]
        content =  [[p.get('name') for p in v.get('particles')] \
               for v in self.model.get('interactions') \
               if any([c in target for c in v['couplings'].values()])]
        [a.sort() for a in check_content+content]
        check_content.sort()
        content.sort()
        assert check_content == content, 'test not up-to-date'      

        vertex_id = [v.get('id') \
               for v in self.model.get('interactions') \
               if any([c in target for c in v['couplings'].values()])]


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
                if coup == 'GC_101':
                    has_101 = True
            self.assertTrue(has_101, True)

    def test_remove_couplings(self):
        """ check that the detection of irrelevant interactions works """
        
        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [5, 5, 25]:
                input_bbh = candidate
                coupling_bbh = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [21, 21, 21, 21]:
                input_4g = candidate
                coupling_4g = candidate['couplings'][(0,0)]
        
        found_bbh = 0
        found_4g = 0
        for dep,data in self.model['couplings'].items():
            for param in data:
                if param.name == coupling_bbh: found_bbh +=1
                elif param.name == coupling_4g: found_4g +=1
        self.assertTrue(found_bbh>0)
        self.assertTrue(found_4g>0)
        
        # make the real test
        result = self.model.remove_couplings([coupling_bbh,coupling_4g])
        
        for dep,data in self.model['couplings'].items():
            for param in data:
                self.assertFalse(param.name in  [coupling_bbh, coupling_4g])

             
    def test_remove_interactions(self):
        """ check that the detection of irrelevant interactions works """
        
        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [5, 5, 25]:
                input_bbh = candidate
                coupling_bbh = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [21, 21, 21, 21]:
                input_4g = candidate
                coupling_4g = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [1, 1, 23]:
                input_ddz = candidate
                coupling_ddz_1 = candidate['couplings'][(0,0)]
                coupling_ddz_2 = candidate['couplings'][(0,1)]
            if [p['pdg_code'] for p in candidate['particles']] == [11, 11, 23]:
                input_eez = candidate
                coupling_eez_1 = candidate['couplings'][(0,0)]            
                coupling_eez_2 = candidate['couplings'][(0,1)]
        
        #security                                      
        found_4g = 0  
        found_bbh = 0 
        for dep,data in self.model['couplings'].items():
            for param in data:
                if param.name == coupling_4g: found_4g +=1
                elif param.name == coupling_bbh: found_bbh +=1
        self.assertTrue(found_bbh>0)
        self.assertTrue(found_4g>0)
        
        # make the real test
        self.model.locate_coupling()
        result = self.model.remove_interactions([coupling_bbh, coupling_4g])
        self.assertFalse(input_bbh in self.model['interactions'])
        self.assertFalse(input_4g in self.model['interactions'])
        
    
        # Now test case where some of them are deleted and some not
        if coupling_ddz_1 != coupling_eez_1:
            coupling_eez_1, coupling_eez_2 = coupling_eez_2, coupling_eez_1
        assert coupling_ddz_1 == coupling_eez_1
        
        result = self.model.remove_interactions([coupling_ddz_1, coupling_ddz_2])
        self.assertTrue(coupling_eez_2 in input_eez['couplings'].values())
        self.assertFalse(coupling_eez_1 in input_eez['couplings'].values())
        self.assertFalse(coupling_ddz_1 in input_ddz['couplings'].values())
        self.assertFalse(coupling_ddz_2 in input_ddz['couplings'].values())

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
        
        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [5, 5, 25]:
                interaction = candidate
                coupling = interaction['couplings'][(0,0)]
    

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
                self.assertFalse(param.name in  [coupling])

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

    def setUp(self):
        """Set up decay model"""
        #Read the full SM
        sm_path = import_ufo.find_ufo_path('sm')
        self.base_model = import_ufo.import_full_model(sm_path)
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

