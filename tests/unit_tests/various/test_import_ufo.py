################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
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
import models as ufomodels

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

class TestImportUFONoSideEffect(unittest.TestCase):
    """Test class for the the possible side effect on a UFO model loaded when
       converting it to a MG5 model"""

    def test_ImportUFONoSideEffectLO(self):
        """Checks that there are no side effects of the import of the LO UFO sm"""       
        ufo_model = ufomodels.load_model(import_ufo.find_ufo_path('sm'),False)
        original_all_particles = copy.copy(ufo_model.all_particles)
        original_all_vertices = copy.copy(ufo_model.all_vertices)
        original_all_couplings = copy.copy(ufo_model.all_couplings)
        original_all_lorentz = copy.copy(ufo_model.all_lorentz)
        original_all_parameters = copy.copy(ufo_model.all_parameters)
        original_all_orders = copy.copy(ufo_model.all_orders)
        original_all_functions = copy.copy(ufo_model.all_functions)

        ufo2mg5_converter = import_ufo.UFOMG5Converter(ufo_model)
        model = ufo2mg5_converter.load_model()
        # It is important to run import_ufo.OrganizeModelExpression(ufo_model).main() 
        # since this reverts some of the changes done in load_model()
        # There *is* side effects in-between, namely the expression of the CTcouplings
        # which contained CTparameters have been substituted to dictionaries.
        parameters, couplings = import_ufo.OrganizeModelExpression(ufo_model).main()        

        self.assertEqual(original_all_particles,ufo_model.all_particles)        
        self.assertEqual(original_all_vertices,ufo_model.all_vertices)
        self.assertEqual(original_all_couplings,ufo_model.all_couplings)
        self.assertEqual(original_all_lorentz,ufo_model.all_lorentz)
        self.assertEqual(original_all_parameters,ufo_model.all_parameters)
        self.assertEqual(original_all_orders,ufo_model.all_orders)
        self.assertEqual(original_all_functions,ufo_model.all_functions)
        
    def test_ImportUFONoSideEffectNLO(self):
        """Checks that there are no side effects of the import of the NLO UFO sm"""
        ufo_model = ufomodels.load_model(import_ufo.find_ufo_path('loop_sm'),False)
        original_all_particles = copy.copy(ufo_model.all_particles)
        original_all_vertices = copy.copy(ufo_model.all_vertices)
        original_all_couplings = copy.copy(ufo_model.all_couplings)
        original_all_lorentz = copy.copy(ufo_model.all_lorentz)
        original_all_parameters = copy.copy(ufo_model.all_parameters)
        original_all_orders = copy.copy(ufo_model.all_orders)
        original_all_functions = copy.copy(ufo_model.all_functions)
        original_all_CTvertices = copy.copy(ufo_model.all_CTvertices)
        original_all_CTparameters = copy.copy(ufo_model.all_CTparameters)


        ufo2mg5_converter = import_ufo.UFOMG5Converter(ufo_model)
        model = ufo2mg5_converter.load_model()
        # It is important to run import_ufo.OrganizeModelExpression(ufo_model).main() 
        # since this reverts some of the changes done in load_model()
        # There *is* side effects in-between, namely the expression of the CTcouplings
        # which contained CTparameters have been substituted to dictionaries.
        parameters, couplings = import_ufo.OrganizeModelExpression(ufo_model).main()        

        self.assertEqual(original_all_particles,ufo_model.all_particles)
        self.assertEqual(original_all_vertices,ufo_model.all_vertices)
#        self.assertEqual(original_all_couplings,ufo_model.all_couplings)
        self.assertEqual(original_all_lorentz,ufo_model.all_lorentz)
        self.assertEqual(original_all_parameters,ufo_model.all_parameters)
        self.assertEqual(original_all_orders,ufo_model.all_orders)
        self.assertEqual(original_all_functions,ufo_model.all_functions)
        self.assertEqual(original_all_CTvertices,ufo_model.all_CTvertices)
        self.assertEqual(original_all_CTparameters,ufo_model.all_CTparameters)

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
        
        expected = set(['I3x32', 'etaWS', 'conjg__CKM3x2', 'CKM1x2', 'WT', 'I1x32', 'I1x33', 'I1x31', 'I2x32', 'CKM3x1', 'I2x13', 'I2x12', 'I3x23', 'I3x22', 'I3x21', 'conjg__CKM2x1', 'lamWS', 'conjg__CKM2x3', 'I2x23', 'AWS', 'CKM1x3', 'conjg__CKM3x1', 'I4x23', 'ymc', 'ymb', 'yme', 'CKM3x2', 'CKM2x3', 'CKM2x1', 'ymm', 'conjg__CKM1x3', 'Me', 'ym', 'I2x22', 'WTau', 'lamWS__exp__2', 'lamWS__exp__3', 'yc', 'yb', 'ye', 'MC', 'MB', 'MM', 'conjg__CKM1x2', 'I3x31', 'rhoWS', 'I4x33', 'I4x13'])
        zero, one = self.model.detect_special_parameters()
        result = set(zero)
        self.assertEqual(len(result), len(expected))

        self.assertEqual(expected, result)
        
        expected = set(['conjg__CKM3x3', 'conjg__CKM2x2', 'CKM1x1', 'CKM2x2', 'CKM3x3', 'conjg__CKM1x1'])
        result = set(one)
        self.assertEqual(expected, result)

        
        
    def test_detect_identical_parameters(self):
        """ check that we detect correctly identical parameter """
        
        expected=set([('MZ','MH')])
        result = self.model.detect_identical_parameters()
        result = [tuple([obj[0].name for obj in obj_list]) for obj_list in result]
        
        self.assertEqual(expected, set(result))
        
    def test_merge_identical_parameters(self):
        """check that we treat correctly the identical parameters"""
        
        parameters = self.model.detect_identical_parameters()
        self.model.merge_iden_parameters(parameters[0])
        
        
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
        self.assertNotEqual(self.model['particle_dict'][23]['width'],
                         self.model['particle_dict'][25]['width'])
        

        
    def test_detect_zero_iden_couplings(self):
        """ check that detect zero couplings works"""
        
        zero, iden = self.model.detect_identical_couplings()
        
        # check what is the zero coupling
        expected = set(['GC_17', 'GC_16', 'GC_15', 'GC_14', 'GC_13', 'GC_19', 'GC_18', 'GC_22', 'GC_30', 'GC_20', 'GC_89', 'GC_88', 'GC_101', 'GC_102', 'GC_103', 'GC_42', 'GC_106', 'GC_107', 'GC_82', 'GC_43', 'GC_84', 'GC_85', 'GC_86', 'GC_105', 'GC_28', 'GC_29', 'GC_48', 'GC_44', 'GC_23', 'GC_46', 'GC_47', 'GC_26', 'GC_24', 'GC_25', 'GC_83', 'GC_87', 'GC_93', 'GC_92', 'GC_91', 'GC_90'])
        result = set(zero)
        self.assertEqual(len(expected), len(result))
        for name in result:
            self.assertEqual(self.model['coupling_dict'][name], 0)
        
        self.assertEqual(expected, result)        
        
        # check what are the identical coupling
        expected = [['GC_100', 'GC_108', 'GC_49', 'GC_45', 'GC_40', 'GC_41', 'GC_104']]
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
        target = [i for i in iden if len(i)==7][0] 
        GC = target[0]
        
        check_content = [['d', 'u', 'w+'], ['s', 'c', 'w+'], ['b', 't', 'w+'], ['u', 'd', 'w+'], ['c', 's', 'w+'], ['t', 'b', 'w+'], ['e-', 've', 'w+'], ['m-', 'vm', 'w+'], ['tt-', 'vt', 'w+'], ['ve', 'e-', 'w+'], ['vm', 'm-', 'w+'], ['vt', 'tt-', 'w+']]
        content =  [[p.get('name') for p in v.get('particles')] \
               for v in self.model.get('interactions') \
               if any([c in target for c in v['couplings'].values()])]

        self.assertEqual(len(check_content),len(content))#, 'test not up-to-date'      

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
            has_GC = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                self.assertFalse(coup in target[1:])
                if coup == GC:
                    has_GC = True
            self.assertTrue(has_GC, True)

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

