################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""Testing modules for FKS_process class"""

from __future__ import absolute_import
import sys
import os
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import tests.unit_tests as unittest
import madgraph.various.misc as misc
import madgraph
import madgraph.fks.fks_base as fks_base
import madgraph.fks.fks_common as fks_common
import madgraph.core.base_objects as MG
import madgraph.core.color_algebra as color
import madgraph.core.diagram_generation as diagram_generation
import madgraph.various.misc as misc
import madgraph.models.import_ufo as import_ufo
import copy
import array
class TestFKSProcess(unittest.TestCase):
    """a class to test FKS Processes"""

    def setUp(self):
        # these build the fks amplitudes by hand, so they have to turn the
        # four gluon merging off themselves, as FKSMultiProcess does
        self.merge_quartic = madgraph.merge_quartic_vertices
        madgraph.merge_quartic_vertices = False

    def tearDown(self):
        madgraph.merge_quartic_vertices = self.merge_quartic

    # the model, import the SM but remove 2nd and 3rd gen quarks
    remove_list = [3,4,5,6,-3,-4,-5,-6]
    mymodel = import_ufo.import_model('sm', options={'apply_flavor_grouping':False})
    for p in mymodel['particles'][:]:
        if p.get_pdg_code() in remove_list:
            mymodel['particles'].remove(p)
    for ii in mymodel['interactions'][:]:
        for p in ii['particles']:
            if p.get_pdg_code() in remove_list:
                mymodel['interactions'].remove(ii)
                break

    myleglist = MG.LegList()
    # PROCESS: u g > u g 
    mylegs = [{
        'id': 2,
        'number': 1,
        'state': False}, 
        { 
        'id': 21,
        'number': 2,
        'state': False},
        {
        'id': 2,
        'number': 3,
        'state': True},
        {
        'id': 21,
        'number': 4,
        'state': True}]

    for i in mylegs:
        myleglist.append(MG.Leg(i))

    myleglist2 = MG.LegList()
    # PROCESS: d d~ > u u~
    mylegs2 = [{ 
        'id': 1,
        'number': 1,
        'state': False}, 
        { 
        'id': -1,
        'number': 2,
        'state': False},
        {
        'id': 2,
        'number': 3,
        'state': True},
        {
        'id': -2,
        'number': 4,
        'state': True}]

    for i in mylegs2:
        myleglist2.append(MG.Leg(i))
        
        myleglist3 = MG.LegList()
    # PROCESS: d d~ > a a
    mylegs3 = [{ 
        'id': 1,
        'number': 1,
        'state': False}, 
        { 
        'id': -1,
        'number': 2,
        'state': False},
        {
        'id': 22,
        'number': 3,
        'state': True},
        {
        'id': 22,
        'number': 4,
        'state': True}]

    for i in mylegs3:
        myleglist3.append(MG.Leg(i))
    
    # PROCESS: u g > u g 
    dict_qcd = {'legs' : myleglist, 
                'born_sq_orders':{'QCD':4, 'QED':0},
                'squared_orders':{'QCD':6, 'QED':0},
                'split_orders':['QCD', 'QED'],
                'sqorders_types':{'QED':'=', 'QCD':'='},
                'model': mymodel,
                'id': 1,
                'required_s_channels':[],
                'forbidden_s_channels':[],
                'forbidden_particles':[],
                'is_decay_chain': False,
                'perturbation_couplings':['QCD'],
                'decay_chains': MG.ProcessList(),
                'overall_orders': {}}
    
    dict_qed = {'legs' : myleglist, 
                'born_sq_orders':{'QCD':4, 'QED':0},
                'squared_orders':{'QCD':4, 'QED':2},
                'split_orders':['QCD', 'QED'],
                'sqorders_types':{'QED':'=', 'QCD':'='},
                'model': mymodel,
                'id': 1,
                'required_s_channels':[],
                'forbidden_s_channels':[],
                'forbidden_particles':[],
                'is_decay_chain': False,
                'perturbation_couplings':['QED'],
                'decay_chains': MG.ProcessList(),
                'overall_orders': {}}

    # PROCESS: d d~ > u u~
    dict2_qcd = {'legs' : myleglist2, 
                 #'born_sq_orders':{'QCD':4, 'QED':0, 'WEIGHTED':4},
                 #'squared_orders':{'QCD':6, 'QED':0, 'WEIGHTED':6},
                 'born_sq_orders':{'QCD':4, 'QED':0},
                 'squared_orders':{'QCD':6, 'QED':0},
                 'split_orders':['QCD', 'QED'],
                 'sqorders_types':{'QED':'=', 'QCD':'='},
                 'model': mymodel,
                 'id': 1,
                 'required_s_channels':[],
                 'forbidden_s_channels':[],
                 'forbidden_particles':[],
                 'is_decay_chain': False,
                 'perturbation_couplings':['QCD'],
                 'decay_chains': MG.ProcessList(),
                 'overall_orders': {}}
    
    dict2_qed = {'legs' : myleglist2, 
                 #'born_sq_orders':{'QCD':4, 'QED':0, 'WEIGHTED':4},
                 #'squared_orders':{'QCD':4, 'QED':2, 'WEIGHTED':8},
                 'born_sq_orders':{'QCD':4, 'QED':0},
                 'squared_orders':{'QCD':4, 'QED':2},
                 'split_orders':['QCD', 'QED'],
                 'sqorders_types':{'QED':'=', 'QCD':'='},
                 'model': mymodel,
                 'id': 1,
                 'required_s_channels':[],
                 'forbidden_s_channels':[],
                 'forbidden_particles':[],
                 'is_decay_chain': False,
                 'perturbation_couplings':['QED'],
                 'decay_chains': MG.ProcessList(),
                 'overall_orders': {}}
    
    # PROCESS: d d~ > a a
    dict3_qcd = {'legs' : myleglist3, 
                 #'born_sq_orders':{'QCD':0, 'QED':4, 'WEIGHTED':8},
                 #'squared_orders':{'QCD':2, 'QED':4, 'WEIGHTED':10},
                 'born_sq_orders':{'QCD':0, 'QED':4},
                 'squared_orders':{'QCD':2, 'QED':4},
                 'split_orders':['QCD', 'QED'],
                 'sqorders_types':{'QED':'=', 'QCD':'='},
                 'model': mymodel,
                 'id': 1,
                 'required_s_channels':[],
                 'forbidden_s_channels':[],
                 'forbidden_particles':[],
                 'is_decay_chain': False,
                 'perturbation_couplings':['QCD'],
                 'decay_chains': MG.ProcessList(),
                 'overall_orders': {}}
    
    dict3_qed = {'legs' : myleglist3, 
                 #'born_sq_orders':{'QCD':0, 'QED':4, 'WEIGHTED':8},
                 #'squared_orders':{'QCD':0, 'QED':6, 'WEIGHTED':12},
                 'born_sq_orders':{'QCD':0, 'QED':4},
                 'squared_orders':{'QCD':0, 'QED':6},
                 'split_orders':['QCD', 'QED'],
                 'sqorders_types':{'QED':'=', 'QCD':'='},
                 'model': mymodel,
                 'id': 1,
                 'required_s_channels':[],
                 'forbidden_s_channels':[],
                 'forbidden_particles':[],
                 'is_decay_chain': False,
                 'perturbation_couplings':['QED'],
                 'decay_chains': MG.ProcessList(),
                 'overall_orders': {}}
    
    myproc = MG.Process(dict_qcd)
    myproc2 = MG.Process(dict2_qcd)
    myprocaa= MG.Process(dict3_qcd)
    myproc_qed = MG.Process(dict_qed)
    myproc2_qed = MG.Process(dict2_qed)
    myprocaa_qed = MG.Process(dict3_qed)
    
    
    def test_FKSMultiProcess(self):
        """tests the correct initialization of a FKSMultiProcess. In particular
        checks that the correct number of borns is found"""
        
        p = [1, 21]

        my_multi_leg = MG.MultiLeg({'ids': p, 'state': True});

        # Define the multiprocess
        my_multi_leglist = MG.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * 4])
        
        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)
        my_process_definition = MG.ProcessDefinition({\
                        'legs': my_multi_leglist,
                        'born_sq_orders': {'QCD':4, 'QED':0},
                        'squared_orders': {'QCD':6, 'QED':0},
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])
        my_process_definition_qed = MG.ProcessDefinition({\
                        'legs': my_multi_leglist,
                        'born_sq_orders': {'QCD':4, 'QED':0},
                        'squared_orders': {'QCD':4, 'QED':2},
                        'perturbation_couplings': ['QED'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions_qed = MG.ProcessDefinitionList(\
            [my_process_definition_qed])

        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions':my_process_definitions})
        my_multi_process_qed = fks_base.FKSMultiProcess(\
                {'process_definitions':my_process_definitions_qed})
        
        self.assertEqual(len(my_multi_process.get('born_processes')),4)
        self.assertEqual(len(my_multi_process_qed.get('born_processes')),4)
#        misc.sprint('Skipping ISR/FSR check')
#        self.assertEqual(my_multi_process.get('has_isr'),True)
#        self.assertEqual(my_multi_process.get('has_fsr'),True)
#        self.assertEqual(my_multi_process_qed.get('has_isr'),True)
#        self.assertEqual(my_multi_process_qed.get('has_fsr'),True)
        #check the total numbers of reals:
        #
        # QCD process
        # - 56 possible splittings (including also splittings involving photons
        # - 40 ij configurations (after throwing away the configurations
        #      with photons and those not to be integrated
        # - 30 different real amplitudes
        totreals = 0
        for born in my_multi_process.get('born_processes'):
            for reals in born.reals:
                totreals += len(reals)
        self.assertEqual(totreals, 56)
        totrealamps = 0
        totrealinfo = 0
        for born in my_multi_process.get('born_processes'):
            totrealamps += len(born.real_amps)
            for real in born.real_amps:
                totrealinfo += len(real.fks_infos)
        self.assertEqual(totrealamps, 30)
        self.assertEqual(totrealinfo, 40)
        # QED process
        # - 56 possible splittings (including also splittings involving photons
        # - 28 ij configurations (after throwing away the configurations
        #      with photons and those not to be integrated
        # - 22 different real amplitudes
        totreals = 0
        for born in my_multi_process_qed.get('born_processes'):
            for reals in born.reals:
                totreals += len(reals)
        self.assertEqual(totreals, 56)
        totrealamps = 0
        totrealinfo = 0
        for born in my_multi_process_qed.get('born_processes'):
            totrealamps += len(born.real_amps)
            for real in born.real_amps:
                totrealinfo += len(real.fks_infos)
        self.assertEqual(totrealamps, 22)
        self.assertEqual(totrealinfo, 28)


    def test_FKSMultiProcess_no_fsr(self):
        """tests the correct initialization of a FKSMultiProcess. In particular
        checks the setting for has_isr/fsr"""
        
        p = [1, -1]
        a = [22]
        g = [21]
        my_multi_leg_p = MG.MultiLeg({'ids': p, 'state': True});
        my_multi_leg_a = MG.MultiLeg({'ids': a, 'state': True});
        my_multi_leg_g = MG.MultiLeg({'ids':g,'state':True})

        # Define the multiprocess
        my_multi_leglist = MG.MultiLegList([copy.copy(leg) for leg in [my_multi_leg_p] * 2] + \
                                            [copy.copy(leg) for leg in [my_multi_leg_a] * 2])
        my_multi_leglist_qed = MG.MultiLegList([copy.copy(leg) for leg in [my_multi_leg_p] * 2] + \
                                            [copy.copy(leg) for leg in [my_multi_leg_g] * 2])
        
        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)
        my_process_definition = MG.ProcessDefinition({\
                        'legs': my_multi_leglist,
                        'born_sq_orders': {'QCD':0, 'QED':4},
                        'squared_orders': {'QCD':2, 'QED':4},
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])
        my_multi_leglist_qed[0].set('state', False)
        my_multi_leglist_qed[1].set('state', False)
        my_process_definition_qed = MG.ProcessDefinition({\
                        'legs': my_multi_leglist_qed,
                        'born_sq_orders': {'QCD':4, 'QED':0},
                        'squared_orders': {'QCD':4, 'QED':2},
                        'perturbation_couplings': ['QED'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions_qed = MG.ProcessDefinitionList(\
            [my_process_definition_qed])

        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions':my_process_definitions})
#        misc.sprint('Skipping ISR/FSR check')
#        self.assertEqual(my_multi_process.get('has_isr'),True)
#        self.assertEqual(my_multi_process.get('has_fsr'),False)
        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions':my_process_definitions_qed})
#        misc.sprint('Skipping ISR/FSR check')
#        self.assertEqual(my_multi_process.get('has_isr'),True)
#        self.assertEqual(my_multi_process.get('has_fsr'),False)
        


    def test_FKSMultiProcess_no_isr(self):
        """tests the correct initialization of a FKSMultiProcess. In particular
        checks the setting for has_isr/fsr"""
        p = [1, -1]
        a = [22]
        g = [21]

        my_multi_leg_p = MG.MultiLeg({'ids': p, 'state': True});
        my_multi_leg_a = MG.MultiLeg({'ids': a, 'state': True});
        my_multi_leg_g = MG.MultiLeg({'ids': g, 'state': True});

        # Define the multiprocess
        my_multi_leglist = MG.MultiLegList([copy.copy(leg) for leg in [my_multi_leg_a] * 2] + \
                                            [copy.copy(leg) for leg in [my_multi_leg_p] * 2])
        my_multi_leglist_qed = MG.MultiLegList([copy.copy(leg) for leg in [my_multi_leg_g] * 2] + \
                                            [copy.copy(leg) for leg in [my_multi_leg_p] * 2])
        
        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)
        my_multi_leglist_qed[0].set('state', False)
        my_multi_leglist_qed[1].set('state', False)
        my_process_definition = MG.ProcessDefinition({\
                        'legs': my_multi_leglist,
                        'born_sq_orders': {'QCD':0, 'QED':4},
                        'squared_orders': {'QCD':2, 'QED':4},
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definition_qed = MG.ProcessDefinition({\
                        'legs': my_multi_leglist_qed,
                        'born_sq_orders': {'QCD':4, 'QED':0},
                        'squared_orders': {'QCD':4, 'QED':2},
                        'perturbation_couplings': ['QED'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])
        my_process_definitions_qed = MG.ProcessDefinitionList(\
            [my_process_definition_qed])

        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions':my_process_definitions})
#        misc.sprint('Skipping ISR/FSR check')
#        self.assertEqual(my_multi_process.get('has_isr'),False)
#        self.assertEqual(my_multi_process.get('has_fsr'),True)
        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions':my_process_definitions_qed})
#        misc.sprint('Skipping ISR/FSR check')
#        self.assertEqual(my_multi_process.get('has_isr'),False)
#        self.assertEqual(my_multi_process.get('has_fsr'),True)


    def test_FKSMultiProcess_add(self):
        """tests the correct initialization of a FKSMultiProcess and the add function. In particular
        checks the setting for has_isr/fsr"""
        p = [1, -1]
        a = [22]

        my_multi_leg_p = MG.MultiLeg({'ids': p, 'state': True});
        my_multi_leg_a = MG.MultiLeg({'ids': a, 'state': True});

        # Define the first multiprocess
        my_multi_leglist = MG.MultiLegList([copy.copy(leg) for leg in [my_multi_leg_a] * 2] + \
                                            [copy.copy(leg) for leg in [my_multi_leg_p] * 2])
        
        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)
        my_process_definition = MG.ProcessDefinition({\
                        'legs': my_multi_leglist,
                        'born_sq_orders': {'QCD':0, 'QED':4},
                        'squared_orders': {'QCD':2, 'QED':4},
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])
        my_process_definition_qed = MG.ProcessDefinition({\
                        'legs': my_multi_leglist,
                        'born_sq_orders': {'QCD':0, 'QED':4},
                        'squared_orders': {'QCD':0, 'QED':6},
                        'perturbation_couplings': ['QED'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions_qed = MG.ProcessDefinitionList(\
            [my_process_definition_qed])

        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions':my_process_definitions})
        nborn = len(my_multi_process['born_processes'])
        my_multi_process_qed = fks_base.FKSMultiProcess(\
                {'process_definitions':my_process_definitions_qed})
        nborn_qed = len(my_multi_process_qed['born_processes'])

        # Define the second multiprocess
        my_multi_leglist1 = MG.MultiLegList([copy.copy(leg) for leg in [my_multi_leg_p] * 2] + \
                                            [copy.copy(leg) for leg in [my_multi_leg_a] * 2])
        
        my_multi_leglist1[0].set('state', False)
        my_multi_leglist1[1].set('state', False)
        my_process_definition1 = MG.ProcessDefinition({\
                        'legs': my_multi_leglist1,
                        'born_sq_orders': {'QCD':0, 'QED':4},
                        'squared_orders': {'QCD':2, 'QED':4},
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions1 = MG.ProcessDefinitionList(\
            [my_process_definition1])
        my_process_definition1_qed = MG.ProcessDefinition({\
                        'legs': my_multi_leglist1,
                        'perturbation_couplings': ['QED'],
                        'born_sq_orders': {'QCD':0, 'QED':4},
                        'squared_orders': {'QCD':0, 'QED':6},
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions1_qed = MG.ProcessDefinitionList(\
            [my_process_definition1_qed])

        my_multi_process1 = fks_base.FKSMultiProcess(\
                {'process_definitions':my_process_definitions1})
        nborn1 = len(my_multi_process1['born_processes'])
        my_multi_process1_qed = fks_base.FKSMultiProcess(\
                {'process_definitions':my_process_definitions1_qed})
        nborn1_qed = len(my_multi_process1_qed['born_processes'])

        my_multi_process.add(my_multi_process1)
        my_multi_process_qed.add(my_multi_process1_qed)
        # mixing QED and QCD ?
        
        self.assertEqual(nborn + nborn1, len(my_multi_process['born_processes']))
#        misc.sprint('Skipping ISR/FSR check')
#        self.assertEqual(my_multi_process.get('has_isr'),True)
#        self.assertEqual(my_multi_process.get('has_fsr'),True)
        self.assertEqual(nborn_qed + nborn1_qed, len(my_multi_process_qed['born_processes']))
#        misc.sprint('Skipping ISR/FSR check')
#        self.assertEqual(my_multi_process_qed.get('has_isr'),True)
#        self.assertEqual(my_multi_process_qed.get('has_fsr'),True)


    def test_FKSProcess_gggg(self):
        """tests that for g g > g g all the relevant splittings are there"""
        glu = MG.Leg({'id': 21, 'state':True})
        leglist = MG.LegList([MG.Leg({'id': 21, 'state':False}),
                              MG.Leg({'id': 21, 'state':False}),
                              MG.Leg({'id': 21, 'state':True}),
                              MG.Leg({'id': 21, 'state':True})])
        
        dict = {'legs' : leglist, 
                'born_sq_orders':{'QCD':4, 'QED':0},
                'squared_orders':{'QCD':6, 'QED':0},
                'split_orders':['QED','QCD'],
                'sqorders_types':{'QED':'=','QCD':'='},
                'model': self.mymodel,
                'id': 1,
                'required_s_channels':[],
                'forbidden_s_channels':[],
                'forbidden_particles':[],
                'is_decay_chain': False,
                'perturbation_couplings':['QCD'],
                'decay_chains': MG.ProcessList(),
                'overall_orders': {}}

        myfks = fks_base.FKSProcess(MG.Process(dict))

#        misc.sprint('fix rb_links')
#        target_fks_infos = [ \
#                # real config 1: g g > g g g
#                [{'i':5, 'j':1, 'ij':1, 'ij_id':1, 'need_color_links':True,
#                    'rb_links':[{'born_conf': 0, 'real_conf': 11},
#                                {'born_conf': 1, 'real_conf': 10},
#                                {'born_conf': 2, 'real_conf': 9}]},
#                 {'i':5, 'j':2, 'ij':2, 'ij_id':2, 'need_color_links':True,
#                     'rb_links':[{'born_conf': 0, 'real_conf': 14},
#                                 {'born_conf': 1, 'real_conf': 4},
#                                 {'born_conf': 2, 'real_conf': 7}]},
#                 {'i':5, 'j':4, 'ij':4, 'ij_id':4, 'need_color_links':True,
#                     'rb_links':[{'born_conf': 0, 'real_conf': 2},
#                                 {'born_conf': 1, 'real_conf': 5},
#                                 {'born_conf': 2, 'real_conf': 12}]}],
#                # real config 2: u g > u g g
#                [{'i':3, 'j':1, 'ij':1, 'ij_id':1, 'need_color_links':False,
#                    'rb_links':[{'born_conf': 0, 'real_conf': 5},
#                                {'born_conf': 1, 'real_conf': 4},
#                                {'born_conf': 2, 'real_conf': 3}]}],
#                # real config 3: ux g > ux g g
#                [{'i':3, 'j':1, 'ij':1, 'ij_id':1, 'need_color_links':False,
#                    'rb_links':[{'born_conf': 0, 'real_conf': 5},
#                                {'born_conf': 1, 'real_conf': 4},
#                                {'born_conf': 2, 'real_conf': 3}]}],
#                # real config 4: d g > d g g
#                [{'i':3, 'j':1, 'ij':1, 'ij_id':1, 'need_color_links':False,
#                    'rb_links':[{'born_conf': 0, 'real_conf': 5},
#                                {'born_conf': 1, 'real_conf': 4},
#                                {'born_conf': 2, 'real_conf': 3}]}],
#                # real config 5: dx g > dx g g
#                [{'i':3, 'j':1, 'ij':1, 'ij_id':1, 'need_color_links':False,
#                    'rb_links':[{'born_conf': 0, 'real_conf': 5},
#                                {'born_conf': 1, 'real_conf': 4},
#                                {'born_conf': 2, 'real_conf': 3}]}],
#                # real config 6: g u > u g g
#                [{'i':3, 'j':2, 'ij':2, 'ij_id':2, 'need_color_links':False,
#                    'rb_links':[{'born_conf': 0, 'real_conf': 12},
#                                {'born_conf': 1, 'real_conf': 6},
#                                {'born_conf': 2, 'real_conf': 9}]}],
#                # real config 7: g ux > ux g g
#                [{'i':3, 'j':2, 'ij':2, 'ij_id':2, 'need_color_links':False,
#                    'rb_links':[{'born_conf': 0, 'real_conf': 12},
#                                {'born_conf': 1, 'real_conf': 6},
#                                {'born_conf': 2, 'real_conf': 9}]}],
#                # real config 8: g d > d g g
#                [{'i':3, 'j':2, 'ij':2, 'ij_id':2, 'need_color_links':False,
#                    'rb_links':[{'born_conf': 0, 'real_conf': 12},
#                                {'born_conf': 1, 'real_conf': 6},
#                                {'born_conf': 2, 'real_conf': 9}]}],
#                # real config 9: g dx > dx g g
#                [{'i':3, 'j':2, 'ij':2, 'ij_id':2, 'need_color_links':False,
#                    'rb_links':[{'born_conf': 0, 'real_conf': 12},
#                                {'born_conf': 1, 'real_conf': 6},
#                                {'born_conf': 2, 'real_conf': 9}]}],
#                # real config 10: g g > u ux g
#                [{'i':4, 'j':3, 'ij':3, 'ij_id':3, 'need_color_links':False,
#                    'rb_links':[{'born_conf': 0, 'real_conf': 0},
#                                {'born_conf': 1, 'real_conf': 14},
#                                {'born_conf': 2, 'real_conf': 11}]}],
#                # real config 11: g g > d dx g
#                [{'i':4, 'j':3, 'ij':3, 'ij_id':3, 'need_color_links':False,
#                    'rb_links':[{'born_conf': 0, 'real_conf': 0},
#                                {'born_conf': 1, 'real_conf': 14},
#                                {'born_conf': 2, 'real_conf': 11}]}]]

        target_fks_infos = [ \
                # real config 1: g g > g g g
                [{'i':5, 'j':1, 'ij':1, 'ij_id':21, 'need_color_links':True,
                  'need_charge_links':False, 'splitting_type':['QCD'],
                  'extra_cnt_index':-1, 'underlying_born':[[21,21,21,21]],
                    'rb_links':[{'born_conf': 0, 'real_conf': 11},
                                {'born_conf': 1, 'real_conf': 10},
                                {'born_conf': 2, 'real_conf': 9}],
                    },
                 {'i':5, 'j':2, 'ij':2, 'ij_id':21, 'need_color_links':True,
                  'need_charge_links':False, 'splitting_type':['QCD'],
                  'extra_cnt_index':-1, 'underlying_born':[[21,21,21,21]],
                     'rb_links':[{'born_conf': 0, 'real_conf': 14},
                                 {'born_conf': 1, 'real_conf': 4},
                                 {'born_conf': 2, 'real_conf': 7}],
                     },
                 {'i':5, 'j':4, 'ij':4, 'ij_id':21, 'need_color_links':True,
                  'need_charge_links':False, 'splitting_type':['QCD'],
                  'extra_cnt_index':-1, 'underlying_born':[[21,21,21,21]],
                     'rb_links':[{'born_conf': 0, 'real_conf': 2},
                                 {'born_conf': 1, 'real_conf': 5},
                                 {'born_conf': 2, 'real_conf': 12}],
                     }],
                # real config 2: u g > u g g
                [{'i':3, 'j':1, 'ij':1, 'ij_id':21, 'need_color_links':False,
                  'need_charge_links':False, 'splitting_type':['QCD'],
                  'extra_cnt_index':-1, 'underlying_born':[[21,21,21,21]],
                    'rb_links':[{'born_conf': 0, 'real_conf': 5},
                                {'born_conf': 1, 'real_conf': 4},
                                {'born_conf': 2, 'real_conf': 3}],
                    }],
                # real config 3: ux g > ux g g
                [{'i':3, 'j':1, 'ij':1, 'ij_id':21, 'need_color_links':False,
                  'need_charge_links':False, 'splitting_type':['QCD'],
                  'extra_cnt_index':-1, 'underlying_born':[[21,21,21,21]],
                    'rb_links':[{'born_conf': 0, 'real_conf': 5},
                                {'born_conf': 1, 'real_conf': 4},
                                {'born_conf': 2, 'real_conf': 3}],
                    }],
                # real config 4: d g > d g g
                [{'i':3, 'j':1, 'ij':1, 'ij_id':21, 'need_color_links':False,
                  'need_charge_links':False, 'splitting_type':['QCD'],
                  'extra_cnt_index':-1, 'underlying_born':[[21,21,21,21]],
                    'rb_links':[{'born_conf': 0, 'real_conf': 5},
                                {'born_conf': 1, 'real_conf': 4},
                                {'born_conf': 2, 'real_conf': 3}],
                    }],
                # real config 5: dx g > dx g g
                [{'i':3, 'j':1, 'ij':1, 'ij_id':21, 'need_color_links':False,
                  'need_charge_links':False, 'splitting_type':['QCD'],
                  'extra_cnt_index':-1, 'underlying_born':[[21,21,21,21]],
                    'rb_links':[{'born_conf': 0, 'real_conf': 5},
                                {'born_conf': 1, 'real_conf': 4},
                                {'born_conf': 2, 'real_conf': 3}],
                    }],
                # real config 6: g u > u g g
                [{'i':3, 'j':2, 'ij':2, 'ij_id':21, 'need_color_links':False,
                  'need_charge_links':False, 'splitting_type':['QCD'],
                  'extra_cnt_index':-1, 'underlying_born':[[21,21,21,21]],
                    'rb_links':[{'born_conf': 0, 'real_conf': 12},
                                {'born_conf': 1, 'real_conf': 6},
                                {'born_conf': 2, 'real_conf': 9}],
                    }],
                # real config 7: g ux > ux g g
                [{'i':3, 'j':2, 'ij':2, 'ij_id':21, 'need_color_links':False,
                  'need_charge_links':False, 'splitting_type':['QCD'],
                  'extra_cnt_index':-1, 'underlying_born':[[21,21,21,21]],
                    'rb_links':[{'born_conf': 0, 'real_conf': 12},
                                {'born_conf': 1, 'real_conf': 6},
                                {'born_conf': 2, 'real_conf': 9}],
                    }],
                # real config 8: g d > d g g
                [{'i':3, 'j':2, 'ij':2, 'ij_id':21, 'need_color_links':False,
                  'need_charge_links':False, 'splitting_type':['QCD'],
                  'extra_cnt_index':-1, 'underlying_born':[[21,21,21,21]],
                    'rb_links':[{'born_conf': 0, 'real_conf': 12},
                                {'born_conf': 1, 'real_conf': 6},
                                {'born_conf': 2, 'real_conf': 9}],
                    }],
                # real config 9: g dx > dx g g
                [{'i':3, 'j':2, 'ij':2, 'ij_id':21, 'need_color_links':False,
                  'need_charge_links':False, 'splitting_type':['QCD'],
                  'extra_cnt_index':-1, 'underlying_born':[[21,21,21,21]],
                    'rb_links':[{'born_conf': 0, 'real_conf': 12},
                                {'born_conf': 1, 'real_conf': 6},
                                {'born_conf': 2, 'real_conf': 9}],
                    }],
                # real config 10: g g > u ux g
                [{'i':4, 'j':3, 'ij':3, 'ij_id':21, 'need_color_links':False,
                  'need_charge_links':False, 'splitting_type':['QCD'],
                  'extra_cnt_index':-1, 'underlying_born':[[21,21,21,21]],
                    'rb_links':[{'born_conf': 0, 'real_conf': 0},
                                {'born_conf': 1, 'real_conf': 14},
                                {'born_conf': 2, 'real_conf': 11}],
                    }],
                # real config 11: g g > d dx g
                [{'i':4, 'j':3, 'ij':3, 'ij_id':21, 'need_color_links':False,
                  'need_charge_links':False, 'splitting_type':['QCD'],
                  'extra_cnt_index':-1, 'underlying_born':[[21,21,21,21]],
                    'rb_links':[{'born_conf': 0, 'real_conf': 0},
                                {'born_conf': 1, 'real_conf': 14},
                                {'born_conf': 2, 'real_conf': 11}]
                    }]]

        myfks.generate_reals([],[])
        self.assertEqual(len(myfks.real_amps),11)
        for real in myfks.real_amps:
            self.assertIn(real.fks_infos, target_fks_infos)


    def test_FKSProcess_aguux_qed(self):
        """tests that for a g > u u~ all the relevant QED splittings are there"""
        glu = MG.Leg({'id': 22, 'state':True})
        leglist = MG.LegList([MG.Leg({'id': 22, 'state':False}),
                              MG.Leg({'id': 21, 'state':False}),
                              MG.Leg({'id': 2, 'state':True}),
                              MG.Leg({'id': -2, 'state':True})])

        dict = {'legs' : leglist, 
                'born_sq_orders': {'QCD':2, 'QED':2},
                'squared_orders': {'QCD':2, 'QED':4},
                'sqorders_types': {'QED':'=', 'QCD':'='},
                'model': self.mymodel,
                'id': 1,
                'required_s_channels':[],
                'forbidden_s_channels':[],
                'forbidden_particles':[],
                'is_decay_chain': False,
                'perturbation_couplings':['QED'],
                'decay_chains': MG.ProcessList(),
                'overall_orders': {}}

        myfks = fks_base.FKSProcess(MG.Process(dict))
        target_real_pdgs = [\
                [-1,21,2,-2,-1],
                [1,21,2,-2,1],
                [-2,21,2,-2,-2],
                [2,21,2,2,-2],
                [22,-1,2,-2,-1],
                [22,1,2,1,-2],
                [22,-2,2,-2,-2],
                [22,2,2,2,-2],
                [22,21,2,-2,22]]
        target_fks_infos = [ \
                # real config 1: d~ g > u u~ d~
                [{'i':5, 'j':1, 'ij':1, 'ij_id':22, 'need_color_links':False,
                 'need_charge_links':False, 'underlying_born':[[22,21,2,-2]],
                 'splitting_type':['QED'], 'extra_cnt_index': -1}],
                # real config 2: d g > u u~ d
                [{'i':5, 'j':1, 'ij':1, 'ij_id':22, 'need_color_links':False,
                 'need_charge_links':False, 'underlying_born':[[22,21,2,-2]],
                 'splitting_type':['QED'], 'extra_cnt_index': -1}],
                # real config 3: u~ g > u u~ u~
                [{'i':5, 'j':1, 'ij':1, 'ij_id':22, 'need_color_links':False,
                 'need_charge_links':False, 'underlying_born':[[22,21,2,-2]],
                 'splitting_type':['QED'], 'extra_cnt_index': -1}],
                # real config 4: u g > u u u~
                [{'i':4, 'j':1, 'ij':1, 'ij_id':22, 'need_color_links':False,
                 'need_charge_links':False, 'underlying_born':[[22,21,2,-2]],
                 'splitting_type':['QED'], 'extra_cnt_index': -1}],
                # real config 5: a d~ > u u~ d~
                [{'i':5, 'j':2, 'ij':2, 'ij_id':21, 'need_color_links':False,
                 'need_charge_links':False, 'underlying_born':[[22,21,2,-2]],
                 'splitting_type':['QCD'], 'extra_cnt_index': -1}],
                # real config 6: a d > u u~ d
                [{'i':4, 'j':2, 'ij':2, 'ij_id':21, 'need_color_links':False,
                 'need_charge_links':False, 'underlying_born':[[22,21,2,-2]],
                 'splitting_type':['QCD'], 'extra_cnt_index': -1}],
                # real config 7: a u~ > u u~ u~
                [{'i':5, 'j':2, 'ij':2, 'ij_id':21, 'need_color_links':False,
                 'need_charge_links':False, 'underlying_born':[[22,21,2,-2]],
                 'splitting_type':['QCD'], 'extra_cnt_index': -1}],
                # real config 8: a u > u u~ u
                [{'i':4, 'j':2, 'ij':2, 'ij_id':21, 'need_color_links':False,
                 'need_charge_links':False, 'underlying_born':[[22,21,2,-2]],
                 'splitting_type':['QCD'], 'extra_cnt_index': -1}],
                # real config 9: a g > u u~ a
                [{'i':5, 'j':3, 'ij':3, 'ij_id':2, 'need_color_links':False,
                 'need_charge_links':True, 'underlying_born':[[22,21,2,-2]],
                 'splitting_type':['QED'], 'extra_cnt_index': -1},
                 {'i':5, 'j':4, 'ij':4, 'ij_id':-2, 'need_color_links':False,
                 'need_charge_links':True, 'underlying_born':[[22,21,2,-2]],
                 'splitting_type':['QED'], 'extra_cnt_index': -1}],
                    ]
#        misc.sprint('fix rb_links')

        myfks.generate_reals([],[])
        self.assertEqual(len(myfks.real_amps),9)
        for real, fks_info, pdgs in zip(myfks.real_amps, target_fks_infos, target_real_pdgs):
            self.assertEqual([l['id'] for l in real.process['legs']], pdgs)
            self.assertEqual(real.fks_infos, fks_info)


    def test_FKSRealProcess_init(self):
        """tests the correct initialization of the FKSRealProcess class. 
        In particular checks that
        --fks_info
        --amplitude (also the generate_real_amplitude function is tested)
        are set to the correct values"""
        #u g > u g
        fksproc = fks_base.FKSProcess(self.myproc)
        #u g > g u
        fksproc_qed = fks_base.FKSProcess(self.myproc_qed)
        #take the third real of the first leg for this process 2j 21 >2 21 21i
        leglist = fksproc.reals[0][2]['leglist']
        leglist_qed = fksproc_qed.reals[0][0]['leglist']

        realproc = fks_base.FKSRealProcess(fksproc.born_amp['process'], leglist, 1, 2,\
                                           [[2,21,2,21]], ['QCD'],\
                                           perturbed_orders = ['QCD'])
        # 2j 21 > 2 21 22i
        #leglist_qed = fksproc_qed.reals[0][0]['leglist']
        realproc_qed = fks_base.FKSRealProcess(fksproc_qed.born_amp['process'],leglist_qed,1,2,\
                                           [[2,21,21,2]], ['QED'],\
                                            perturbed_orders = ['QED'])
        self.assertEqual(realproc.fks_infos, [{'i' : 5,
                                               'j' : 1,
                                               'ij' : 1,
                                               'ij_id' : 2,
                                               'splitting_type': ['QCD'],
                                               'underlying_born':[[2,21,2,21]],
                                               'extra_cnt_index': -1,
                                               'need_color_links': True,
                                               'need_charge_links': False}])
        self.assertEqual(realproc_qed.fks_infos,[{'i':5,
                                                  'j':1,
                                                  'ij':1,
                                                  'ij_id':2,
                                                  'splitting_type': ['QED'],
                                                  'underlying_born':[[2,21,21,2]],
                                                  'extra_cnt_index': -1,
                                                  'need_color_links':False,
                                                  'need_charge_links':True}])

        sorted_legs = fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel )
        
        sorted_legs_qed = fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :22,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel )        

        sorted_real_proc = MG.Process({'legs':sorted_legs, 'model':self.mymodel,
            'orders':{'QCD':3, 'QED':0, 'WEIGHTED': 3}, 'id':1})
        sorted_real_proc_qed = MG.Process({'legs':sorted_legs_qed, 'model':self.mymodel,
            'orders':{'QCD':2, 'QED':1, 'WEIGHTED': 4}, 'id':1})
## an empty amplitude is generated so far...
        self.assertEqual(diagram_generation.Amplitude(), realproc.amplitude)
        self.assertEqual(diagram_generation.Amplitude(), realproc_qed.amplitude)

## now generate the amplitude
        realproc.generate_real_amplitude()
        realproc_qed.generate_real_amplitude()

        self.assertEqual(sorted_real_proc, realproc.amplitude.get('process'))
        self.assertEqual(realproc.amplitude['process']['legs_with_decays'], MG.LegList())
        self.assertEqual(sorted_real_proc_qed, realproc_qed.amplitude.get('process'))
        self.assertEqual(realproc_qed.amplitude['process']['legs_with_decays'], MG.LegList())
        amp = diagram_generation.Amplitude(sorted_real_proc)
        amp_qed = diagram_generation.Amplitude(sorted_real_proc_qed)
        self.assertEqual(amp,realproc.amplitude)
        self.assertEqual(array.array('i',[2,21,2,21,21]), realproc.pdgs)
        self.assertEqual([3,8,3,8,8], realproc.colors)
        self.assertEqual([0.,0.,0.,0.,0.], realproc.charges) # charges are set to 0 if only QCD is being perturbed
#        self.assertEqual(amp_qed,realproc_qed.amplitude)
        self.assertEqual(array.array('i',[2,21,2,21,22]), realproc_qed.pdgs)
        self.assertEqual([3,8,3,8,1],realproc_qed.colors)
        self.assertEqual([2./3.,0.,2./3.,0.,0.], realproc_qed.charges)
 ##       self.assertEqual(realproc.permutation, [1,2,4,5,3])


    def test_find_fks_j_from_i(self):
        """tests that the find_fks_j_from_i function of a FKSRealProcess returns the
        correct result"""
        #u g > u g
        fksproc = fks_base.FKSProcess(self.myproc)
        #u g > g u
        fksproc_qed = fks_base.FKSProcess(self.myproc_qed)

        #take the first real for this process 2j 21 >2 21 21i
        leglist = fksproc.reals[0][2]['leglist']
        # 2j 21 > 21 2 22i for QED process
        leglist_qed = fksproc_qed.reals[0][0]['leglist']
        

        # safety check (pick the right real)
        self.assertEqual([l['id'] for l in leglist], [2,21,2,21,21])

        
        
        
        # safety check (pick the right real)
        self.assertEqual([l['id'] for l in leglist_qed], [2,21,2,21,22])
        realproc = fks_base.FKSRealProcess(fksproc.born_amp['process'], leglist, 1,0,\
                                           [[2,21,2,21]], ['QCD'],\
                                           perturbed_orders = ['QCD'])
        realproc_qed = fks_base.FKSRealProcess(fksproc_qed.born_amp['process'],leglist_qed,1,0,\
                                           [2,21,21,2], ['QED'],\
                                            perturbed_orders = ['QED'])
        target_full = {1:[], 2:[], 3:[1,2], 4:[1,2,3,5], 5:[1,2,3,4] }
        target_full_qed = {1:[],2:[],3:[1],4:[],5:[1,3]}
        borns = [[2,21,2,21], [21,21,21,21], [2,-2,21,21]]
        borns_qed = [[22,21,22,21],[2,21,2,21]]
        self.assertEqual(target_full, realproc.find_fks_j_from_i(borns))
        self.assertEqual(target_full_qed,realproc_qed.find_fks_j_from_i(borns_qed))
        #now the fks_j from_i corresponding only to u g > u g born
        target_born = {1:[], 2:[], 3:[], 4:[1,2,3,5], 5:[1,2,3,4]}
        borns = [[2,21,2,21]]
        self.assertEqual(target_born, realproc.find_fks_j_from_i(borns))
        #now the fks_j from_i corresponding only to u g > u g born
        target_born_qed = {1:[], 2:[], 3:[], 4:[], 5:[1,3]}
        borns_qed = [[2,21,2,21]]
        self.assertEqual(target_born_qed, realproc_qed.find_fks_j_from_i(borns_qed))


    def test_fks_real_process_get_leg_i_j(self):
        """test the correct output of the FKSRealProcess.get_leg_i/j() function"""
        #u g > u g
        fksproc = fks_base.FKSProcess(self.myproc)
        #take the first real for this process 2j 21 > 2 21 21i
        leglist = fksproc.reals[0][0]['leglist']
        realproc = fks_base.FKSRealProcess(fksproc.born_amp['process'], leglist,1,0,
                                           [[2,21,2,21]], ['QCD'], ['QCD'])
        self.assertEqual(realproc.get_leg_i(), leglist[4])
        self.assertEqual(realproc.get_leg_j(), leglist[0])
    

    def test_generate_reals_no_combine(self):
        """tests the generate_reals function, if all the needed lists
        -- amplitudes
        -- real amps
        have the correct number of elements
        checks also the find_reals_to_integrate, find_real_nbodyonly functions
        that are called by generate_reals"""
        
        #process u g > u g 
        fksproc = fks_base.FKSProcess(self.myproc)
        fksproc.generate_reals([],[],False)
        
        #there should be 11 real processes for this born
        self.assertEqual(len(fksproc.real_amps), 11)


    def test_generate_reals_combine(self):
        """tests the generate_reals function, if all the needed lists
        -- amplitudes
        -- real amps
        have the correct number of elements. 
        Check also that real emissions with the same m.e. are combined together"""
        
        #process u g > u g 
        fksproc = fks_base.FKSProcess(self.myproc)
        fksproc.generate_reals([],[])
        
        #there should be 8 real processes for this born
        self.assertEqual(len(fksproc.real_amps), 8)
        # the process u g > u g g should correspond to 4 possible fks_confs:
        amp_ugugg = [amp for amp in fksproc.real_amps \
                if amp.pdgs == array.array('i', [2, 21, 2, 21, 21])]
        self.assertEqual(len(amp_ugugg), 1)
        self.assertEqual(len(amp_ugugg[0].fks_infos), 4)
        self.assertEqual(amp_ugugg[0].fks_infos,
                [{'i':5, 'j':1, 'ij':1, 'ij_id':2, 'need_color_links':True,
                  'need_charge_links':False, 'splitting_type':['QCD'], 'underlying_born':[[2,21,2,21]],
                  'extra_cnt_index':-1,
                  'rb_links':[{'born_conf': 0, 'real_conf': 11},
                                {'born_conf': 1, 'real_conf': 10},
                                {'born_conf': 2, 'real_conf': 9}]},
                 {'i':5, 'j':2, 'ij':2, 'ij_id':21, 'need_color_links':True,
                  'need_charge_links':False, 'splitting_type':['QCD'], 'underlying_born':[[2,21,2,21]],
                  'extra_cnt_index':-1,
                  'rb_links':[{'born_conf': 0, 'real_conf': 14},
                                 {'born_conf': 1, 'real_conf': 4},
                                 {'born_conf': 2, 'real_conf': 7}]},
                 {'i':5, 'j':3, 'ij':3, 'ij_id':2, 'need_color_links':True,
                  'need_charge_links':False, 'splitting_type':['QCD'], 'underlying_born':[[2,21,2,21]],
                  'extra_cnt_index':-1,
                  'rb_links':[{'born_conf': 0, 'real_conf': 1},
                                {'born_conf': 1, 'real_conf': 13},
                                {'born_conf': 2, 'real_conf': 8}]},
                 {'i':5, 'j':4, 'ij':4, 'ij_id':21, 'need_color_links':True,
                  'need_charge_links':False, 'splitting_type':['QCD'], 'underlying_born':[[2,21,2,21]],
                  'extra_cnt_index':-1,
                  'rb_links':[{'born_conf': 0, 'real_conf': 2},
                                 {'born_conf': 1, 'real_conf': 5},
                                 {'born_conf': 2, 'real_conf': 12}]}
                     ])

        
    def test_find_reals(self):
        """tests if all the real processes are found for a given born"""
        #process is u g > u g
        fksproc = fks_base.FKSProcess(self.myproc)
        # for the QED the born is u g > g u instead of u g > u g
        fksproc_qed = fks_base.FKSProcess(self.myproc_qed)

        target = []
        # the splittings are the same for the two processes
        # generate_reals will take care of selecting those which
        # satisfy the order constraints
        
        #leg 1 can split as u g > u g g or  g g > u u~ g 
        #and u g > u g a or  a g > u u~ g (will be removed by diagrams/order check))
        target.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :22,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel ),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :22,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                         fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel ),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :21,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                         fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ], self.mymodel)]
                                        )
        #leg 2 can split as u d > d u g OR u d~ > d~ u g OR
        #                   u u > u u g OR u u~ > u u~ g OR
        #                   u g > u g g 
        
        target.append([fks_common.to_fks_legs([#ug>ugg
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                       fks_common.to_fks_legs([#ud>dug
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ],self.mymodel ),
                    fks_common.to_fks_legs([#ud~>d~ug
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :1,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([#uu>uug
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([#uu~>uu~g
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ], self.mymodel)
                    ])
        #leg 3 can split as u g > u g g or u g > u g a
        target.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :22,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                        fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel)
                                        ]
                                        )
        #leg 4 can split as u g > u g g or u g > u u u~ or u g > u d d~ 
        target.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                        fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :1,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                        fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel)
                                        ]
                                        )
        self.assertEqual(len(fksproc.reals),len(target))
        self.assertEqual([len(real) for real in fksproc.reals],\
                         [len(real) for real in target])
        self.assertEqual(len(fksproc_qed.reals),len(target))
        self.assertEqual([len(real) for real in fksproc.reals],\
                         [len(real) for real in target])                
        for i in range(len(fksproc.reals)):
            for j in range(len(fksproc.reals[i])):
                if True:
                    for k in range(len(fksproc.reals[i])):
                        if fksproc.reals[i][j]['leglist'] ==  target[i][k]:
                            break
                    else:
                        self.assertTrue(False)
                else:    
                    self.assertEqual(fksproc.reals[i][j]['leglist'], target[i][j])
        for i in range(len(fksproc_qed.reals)):
            for j in range(len(fksproc_qed.reals[i])):
                if True:
                    for k in range(len(fksproc_qed.reals[i])):
                        if fksproc_qed.reals[i][j]['leglist'] ==  target[i][k]:
                            break
                    else:
                        self.assertTrue(False)
                else:
                    self.assertEqual(fksproc_qed.reals[i][j]['leglist'], target[i][j]) 
        
        #process is d d~ > u u~
        fksproc2 = fks_base.FKSProcess(self.myproc2)
        target2 = []
        #leg 1 can split as d d~ > u u~ g or  g d~ > d~ u u~ 
        target2.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :22,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :22,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :21,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                                        ])
        
        #leg 2 can split as d d~ > u u~ g or  d g > d u u~ 
        target2.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :22,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :22,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' : 1,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 1,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_common.FKSLeg( 
                                         {'id' : -2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ], self.mymodel),
                                        ])
        
        #leg 3 can split as d d~ > u u~ g  
        target2.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :22,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel)
                                        ])

        #leg 4 can split as d d~ > u u~ g  
        target2.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg( 
                                         {'id' :22,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                                        ])

        for i in range(len(fksproc2.reals)):
            for j in range(len(fksproc2.reals[i])):
                self.assertEqual(fksproc2.reals[i][j]['leglist'], target2[i][j])
    
        #d d~ > a a
        fksproc3 = fks_base.FKSProcess(self.myprocaa)
        target3 = []
        #leg 1 can split as d d~ > g a a or  g d~ > d~ a a 
        target3.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :22,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :22,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :21,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                                        ])
        #leg 2 can split as d d~ > g a a  or  d g > d a a 
        target3.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :22,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :22,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 1,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 1,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                                        ])        
        # leg 3 and 4 can split in lep/lep, u u~ / d d~
        target3.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :11,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-11,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :13,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-13,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :1,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-1,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                                        ])        
        # leg 3 and 4 can split in lep/lep, u u~ / d d~
        target3.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :11,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-11,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :13,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-13,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 1,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-1,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                                        ])        

        for i in range(len(fksproc3.reals)):
            for j in range(len(fksproc3.reals[i])):
                self.assertEqual(fksproc3.reals[i][j]['leglist'], target3[i][j])


    def test_sort_fks_proc(self):
        """tests that two FKSProcesses with different legs order in the
        input process/amplitude are returned as equal. check also that
        born_proc has 'legs_with_decay' = madgraph.base_objects.LegList()"""
        model = import_ufo.import_model('sm')

# sorted leglist for e+ e- > u u~ g
        myleglist_s = MG.LegList()
        myleglist_s.append(MG.Leg({'id':-11, 'state':False}))
        myleglist_s.append(MG.Leg({'id':11, 'state':False}))
        myleglist_s.append(MG.Leg({'id':2, 'state':True}))
        myleglist_s.append(MG.Leg({'id':-2, 'state':True}))
        myleglist_s.append(MG.Leg({'id':21, 'state':True}))

# unsorted leglist: e+ e- > u g u~
        myleglist_u = MG.LegList()
        myleglist_u.append(MG.Leg({'id':-11, 'state':False}))
        myleglist_u.append(MG.Leg({'id':11, 'state':False}))
        myleglist_u.append(MG.Leg({'id':2, 'state':True}))
        myleglist_u.append(MG.Leg({'id':21, 'state':True}))
        myleglist_u.append(MG.Leg({'id':-2, 'state':True}))

# define (un)sorted processes:
        proc_s = MG.Process({'model':model, 'legs':myleglist_s,\
                             'orders':{'QED':2, 'QCD':1}})
        proc_u = MG.Process({'model':model, 'legs':myleglist_u,\
                             'orders':{'QED':2, 'QCD':1}})
        proc_s_qed = MG.Process({'model':model, 'legs':myleglist_s,\
                             'orders':{'QED':2, 'QCD':1},\
                             "perturbation_couplings":['QED']})
        proc_u_qed = MG.Process({'model':model, 'legs':myleglist_u,\
                             'orders':{'QED':2, 'QCD':1},\
                             "perturbation_couplings":['QED']})
# define (un)sorted amplitudes:
        amp_s = diagram_generation.Amplitude(proc_s)
        amp_u = diagram_generation.Amplitude(proc_u)
        amp_s_qed = diagram_generation.Amplitude(proc_s_qed)
        amp_u_qed = diagram_generation.Amplitude(proc_u_qed)

        fks_p_s = fks_base.FKSProcess(proc_s)
        fks_p_u = fks_base.FKSProcess(proc_u)
        fks_p_s_qed = fks_base.FKSProcess(proc_s_qed)
        fks_p_u_qed = fks_base.FKSProcess(proc_u_qed)
        self.assertEqual(fks_p_s.born_amp, fks_p_u.born_amp)
        self.assertEqual(fks_p_s_qed.born_amp,fks_p_u_qed.born_amp)

        fks_a_s = fks_base.FKSProcess(amp_s)
        fks_a_u = fks_base.FKSProcess(amp_u)
        fks_a_s_qed = fks_base.FKSProcess(amp_s_qed)
        fks_a_u_qed = fks_base.FKSProcess(amp_u_qed)

        self.assertEqual(fks_a_s.born_amp, fks_a_u.born_amp)
        self.assertEqual(fks_a_s_qed.born_amp,fks_a_u_qed.born_amp)

        self.assertEqual(fks_a_s.born_amp['process']['legs_with_decays'], MG.LegList())
        self.assertEqual(fks_a_u.born_amp['process']['legs_with_decays'], MG.LegList())
        self.assertEqual(fks_a_s_qed.born_amp['process']['legs_with_decays'], MG.LegList())
        self.assertEqual(fks_a_u_qed.born_amp['process']['legs_with_decays'], MG.LegList())


    def test_real_keeps_born_polarization(self):
        """The real emission has to fix the same helicity as the born for the
        leg it splits: in the singular limit the real's leg j IS the born's
        leg ij, and the two are subtracted against each other. A polarized
        born must therefore give a polarized real.

        Regression test: split_leg used to build the daughters from a bare
        dict, so they took the Leg default polarization = [] and insert_legs
        then replaced the polarized born leg by an unpolarized one. That was
        invisible while a polarized coloured particle was refused outright (a
        colour singlet never splits in QCD, so it stayed a spectator that the
        deepcopy preserved), and it became reachable when massive coloured
        polarized particles were allowed.

        e+ e- > t{+} t~ is the process that shows it: there is no initial-state
        QCD splitting, so the only real comes from the (anti)top splitting --
        the path that dropped the polarization. p p > t{+} t~ hides the bug,
        because find_reals enumerates the ISR splittings first, that real keeps
        the polarization as a spectator, and the FSR ones are merged onto it on
        PDGs alone.
        """
        # flavour grouping off: with it on, the light quarks are merged and a
        # process written with a bare pdg has no diagram
        model = import_ufo.import_model(
            'sm', options={'apply_flavor_grouping': False})

        leglist = MG.LegList()
        leglist.append(MG.Leg({'id': -11, 'state': False}))
        leglist.append(MG.Leg({'id': 11, 'state': False}))
        leglist.append(MG.Leg({'id': 6, 'state': True, 'polarization': [1]}))
        leglist.append(MG.Leg({'id': -6, 'state': True}))
        proc = MG.Process({'model': model, 'legs': leglist,
                           'orders': {'QED': 2},
                           'squared_orders': {'QCD': 2, 'QED': 4},
                           'sqorders_types': {'QCD': '<=', 'QED': '<='},
                           'perturbation_couplings': ['QCD'],
                           'NLO_mode': 'real'})

        # init_lep_split as the interface sets it for a leptonic initial state
        fksproc = fks_base.FKSProcess(proc, init_lep_split=True)
        fksproc.generate_reals([], [])

        self.assertTrue(len(fksproc.real_amps) > 0)
        for real in fksproc.real_amps:
            self.assertEqual([(l['id'], list(l['polarization']))
                              for l in real.process['legs']],
                             [(-11, []), (11, []), (6, [1]),
                              (-6, []), (21, [])])
            # ... and the key that decides whether two reals may share one
            # amplitude sees the polarization, not only the PDGs
            self.assertEqual(real.pdgs_pols[1],
                             ((), (), (1,), (), ()))


    def test_split_leg_polarization_transfer(self):
        """split_leg carries the mother's polarization onto the daughter j,
        and refuses rather than dropping it when j is not the same particle as
        the mother (the massless case, which the interface guard refuses at
        generation time)."""
        model = import_ufo.import_model(
            'sm', options={'apply_flavor_grouping': False})

        # t -> t g : leg j is the top, it inherits; the gluon does not
        top = fks_common.to_fks_leg(MG.Leg({'id': 6, 'number': 3,
                                            'state': True,
                                            'polarization': [1]}), model)
        splittings = fks_common.find_splittings(top, model, {}, 'QCD')
        self.assertTrue(len(splittings) > 0)
        for split in splittings:
            self.assertEqual([l['id'] for l in split], [6, 21])
            self.assertEqual(list(split[0]['polarization']), [1])
            self.assertEqual(list(split[1]['polarization']), [])
        # not aliased onto the mother's list
        self.assertFalse(any(split[0]['polarization'] is top['polarization']
                             for split in splittings))

        # an unpolarized mother is untouched
        top_unpol = fks_common.to_fks_leg(MG.Leg({'id': 6, 'number': 3,
                                                  'state': True}), model)
        for split in fks_common.find_splittings(top_unpol, model, {}, 'QCD'):
            self.assertEqual([list(l['polarization']) for l in split],
                             [[], []])

        # g -> g g and g -> q q~ have nothing to carry the polarization to,
        # so they raise instead of silently dropping it
        gluon = fks_common.to_fks_leg(MG.Leg({'id': 21, 'number': 3,
                                              'state': True,
                                              'polarization': [1]}), model)
        self.assertRaises(fks_common.FKSProcessError,
                          fks_common.find_splittings, gluon, model, {}, 'QCD')
        # ... but that one raises on the SECOND condition (both daughters have
        # the mother's identity): find_splittings reaches the ggg vertex
        # first, so it never exercises the identity-change condition. Two
        # cases that do:
        #
        # (1) g -> q q~ asked for directly. Leg j is the quark, which is not
        #     the mother.
        # (tests.unit_tests.TestCase.assertRaises is not usable as a context
        #  manager -- it swallows the return value -- so check the message by
        #  hand)
        try:
            fks_common.split_leg(gluon,
                                 [model.get_particle(1),
                                  model.get_particle(-1)], model)
            self.fail('g -> q q~ must not silently drop the polarization')
        except fks_common.FKSProcessError as error:
            self.assertIn('the FKS emitter j is a different particle',
                          str(error))

        # (1b) g -> g g asked for directly. Both daughters ARE the mother, so
        #      the first condition passes and the SECOND one raises. Asking
        #      through find_splittings would not pin this raise on its own:
        #      the gluon also has g -> q q~, which raises on the first
        #      condition, so the assertRaises above holds either way.
        try:
            fks_common.split_leg(gluon,
                                 [model.get_particle(21),
                                  model.get_particle(21)], model)
            self.fail('g -> g g must not silently pick a daughter')
        except fks_common.FKSProcessError as error:
            self.assertIn('both daughters have the identity', str(error))

        # (2) a polarized quark in the INITIAL state. Its two QCD splittings
        #     are [j=u, i=g] -- perfectly well defined -- and the backward
        #     [j=g, i=u~], in which the polarized quark is an internal line.
        #     split_leg builds both and carries the polarization onto both, so
        #     the bad one raises and the whole leg is refused: this is why the
        #     interface guard refuses a coloured initial state wholesale
        #     rather than channel by channel.
        u_init = fks_common.to_fks_leg(MG.Leg({'id': 2, 'number': 1,
                                               'state': False,
                                               'polarization': [1]}), model)
        try:
            fks_common.find_splittings(u_init, model, {}, 'QCD')
            self.fail('a polarized initial-state quark must not split '
                      'silently')
        except fks_common.FKSProcessError as error:
            self.assertIn('the FKS emitter j is a different particle',
                          str(error))
        # the well-defined channel really is there when the mother carries no
        # polarization -- so the refusal above is a whole-leg one, not the
        # absence of a good splitting
        u_unpol = fks_common.to_fks_leg(MG.Leg({'id': 2, 'number': 1,
                                                'state': False}), model)
        self.assertEqual(
            [[l['id'] for l in s]
             for s in fks_common.find_splittings(u_unpol, model, {}, 'QCD')],
            [[2, 21], [21, -2]])
