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

"""Testing modules for FKS_process class"""

import sys
import os
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import tests.unit_tests as unittest
import madgraph.fks.fks_real as fks_real
import madgraph.fks.fks_born as fks_born
import madgraph.fks.fks_real_helas_objects as fks_real_helas
import madgraph.fks.fks_born_helas_objects as fks_born_helas
import madgraph.core.base_objects as MG
import madgraph.core.color_algebra as color
import madgraph.core.diagram_generation as diagram_generation
import models.import_ufo as import_ufo
import copy
import string



class TestGenerateLoopFKSFromBorn(unittest.TestCase):
    """a class to test the generation of the virtual amps for a FKSMultiProcessFromReals"""

    def setUp(self):
        if not hasattr(self, 'mymodel'):
            TestGenerateLoopFKSFromBorn.mymodel = import_ufo.import_model('loop_sm')
    

    def test_generate_virtuals_single_processB(self):
        """checks that the virtuals are correctly generated for a single process"""

        myleglist = MG.MultiLegList()
        
        # test process is u u~ > u u~  
        myleglist.append(MG.MultiLeg({'ids':[2], 'state':False}))
        myleglist.append(MG.MultiLeg({'ids':[-2], 'state':False}))
        myleglist.append(MG.MultiLeg({'ids':[2], 'state':True}))
        myleglist.append(MG.MultiLeg({'ids':[-2], 'state':True}))
    
        myproc1 = MG.ProcessDefinition({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders': {'QED': 0},
                                           'perturbation_couplings':['QCD'],
                                           'NLO_mode': 'real'})

        myproc2 = MG.ProcessDefinition({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders': {'QED': 0},
                                           'perturbation_couplings':['QCD'],
                                           'NLO_mode': 'all'})
        
        my_process_definitions1 = MG.ProcessDefinitionList([myproc1])
        my_process_definitions2 = MG.ProcessDefinitionList([myproc2])
        
        # without virtuals
        myfksmulti1 = fks_born.FKSMultiProcessFromBorn(\
                {'process_definitions': my_process_definitions1})
        # with wirtuals
        myfksmulti2 = fks_born.FKSMultiProcessFromBorn(\
                {'process_definitions': my_process_definitions2})

        self.assertEqual(myfksmulti1['born_processes'][0].virt_amp, None)
        #there should be 4 virt_amps
        self.assertNotEqual(myfksmulti2['born_processes'][0].virt_amp, None)
        self.assertEqual([l['id'] for l in \
                        myfksmulti2['born_processes'][0].virt_amp.get('process').get('legs')], \
                         [2,-2,2,-2])


    def test_generate_virtuals_helas_matrix_elementB(self):
        """checks that the virtuals are correctly generated for a FKShelasMatrixElement"""

        myleglist = MG.MultiLegList()
        
        # test process is u u~ > u u~  
        myleglist.append(MG.MultiLeg({'ids':[2], 'state':False}))
        myleglist.append(MG.MultiLeg({'ids':[-2], 'state':False}))
        myleglist.append(MG.MultiLeg({'ids':[2], 'state':True}))
        myleglist.append(MG.MultiLeg({'ids':[-2], 'state':True}))
    
        myproc = MG.ProcessDefinition({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders': {'QED': 0},
                                           'perturbation_couplings':['QCD'],
                                           'NLO_mode': 'all'})
        
        my_process_definitions = MG.ProcessDefinitionList([myproc])
        
        myfksmulti = fks_born.FKSMultiProcessFromBorn(\
                {'process_definitions': my_process_definitions})

        myfksmulti.generate_virtuals()
        myfksme = fks_born_helas.FKSHelasMultiProcessFromBorn(myfksmulti)
        self.assertNotEqual(myfksme['matrix_elements'][0].virt_matrix_element, None)


class TestGenerateLoopFKSFromReals(unittest.TestCase):
    """a class to test the generation of the virtual amps for a FKSMultiProcessFromReals"""

    def setUp(self):
        if not hasattr(self, 'mymodel'):
            TestGenerateLoopFKSFromReals.mymodel = import_ufo.import_model('loop_sm')
    

    def test_generate_virtuals_single_process_from_procdefR(self):
        """checks that the virtuals are correctly generated for a single process, 
        using the process attribute nlo_mode"""

        myleglist = MG.MultiLegList()
        
        # test process is u u~ > u u~ g 
        myleglist.append(MG.MultiLeg({'ids':[2], 'state':False}))
        myleglist.append(MG.MultiLeg({'ids':[-2], 'state':False}))
        myleglist.append(MG.MultiLeg({'ids':[2], 'state':True}))
        myleglist.append(MG.MultiLeg({'ids':[-2], 'state':True}))
        myleglist.append(MG.MultiLeg({'ids':[21], 'state':True}))
    
        myproc1 = MG.ProcessDefinition({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders': {'QED': 0},
                                           'perturbation_couplings':['QCD'],
                                           'NLO_mode': 'real'})

        myproc2 = MG.ProcessDefinition({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders': {'QED': 0},
                                           'perturbation_couplings':['QCD'],
                                           'NLO_mode': 'all'})
        
        my_process_definitions1 = MG.ProcessDefinitionList([myproc1])
        my_process_definitions2 = MG.ProcessDefinitionList([myproc2])
        
        myfksmulti1 = fks_real.FKSMultiProcessFromReals(\
                {'process_definitions': my_process_definitions1})
        myfksmulti2 = fks_real.FKSMultiProcessFromReals(\
                {'process_definitions': my_process_definitions2})

        self.assertEqual(myfksmulti1['real_processes'][0].virt_amp, None)
      #  myfksmulti.generate_virtuals()
        #there should be 4 virt_amps
        self.assertNotEqual(myfksmulti2['real_processes'][0].virt_amp, None)
        self.assertEqual([l['id'] for l in \
                        myfksmulti2['real_processes'][0].virt_amp.get('process').get('legs')], \
                         [2,-2,2,-2])


    def test_generate_virtuals_single_processR(self):
        """checks that the virtuals are correctly generated for a single process"""

        myleglist = MG.MultiLegList()
        
        # test process is u u~ > u u~ g 
        myleglist.append(MG.MultiLeg({'ids':[2], 'state':False}))
        myleglist.append(MG.MultiLeg({'ids':[-2], 'state':False}))
        myleglist.append(MG.MultiLeg({'ids':[2], 'state':True}))
        myleglist.append(MG.MultiLeg({'ids':[-2], 'state':True}))
        myleglist.append(MG.MultiLeg({'ids':[21], 'state':True}))
    
        myproc = MG.ProcessDefinition({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders': {'QED': 0},
                                           'perturbation_couplings':['QCD'],
                                           'NLO_mode': 'all'})
        
        my_process_definitions = MG.ProcessDefinitionList([myproc])
        
        myfksmulti = fks_real.FKSMultiProcessFromReals(\
                {'process_definitions': my_process_definitions})

        #there should be 4 virt_amps
        self.assertNotEqual(myfksmulti['real_processes'][0].virt_amp, None)
        self.assertEqual([l['id'] for l in \
                        myfksmulti['real_processes'][0].virt_amp.get('process').get('legs')], \
                         [2,-2,2,-2])


    def test_generate_virtuals_helas_matrix_elementR(self):
        """checks that the virtuals are correctly generated for a single process"""

        myleglist = MG.MultiLegList()
        
        # test process is u u~ > u u~ g 
        myleglist.append(MG.MultiLeg({'ids':[2], 'state':False}))
        myleglist.append(MG.MultiLeg({'ids':[-2], 'state':False}))
        myleglist.append(MG.MultiLeg({'ids':[2], 'state':True}))
        myleglist.append(MG.MultiLeg({'ids':[-2], 'state':True}))
        myleglist.append(MG.MultiLeg({'ids':[21], 'state':True}))
    
        myproc = MG.ProcessDefinition({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders': {'QED': 0},
                                           'perturbation_couplings':['QCD'],
                                           'NLO_mode': 'all'})
        
        my_process_definitions = MG.ProcessDefinitionList([myproc])
        
        myfksmulti = fks_real.FKSMultiProcessFromReals(\
                {'process_definitions': my_process_definitions})

        myfksmulti.generate_virtuals()
        myfksme = fks_real_helas.FKSHelasMultiProcessFromReals(myfksmulti)
        self.assertNotEqual(myfksme['matrix_elements'][0].virt_matrix_element, None)


