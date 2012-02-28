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
import madgraph.core.base_objects as MG
import madgraph.core.color_algebra as color
import madgraph.core.diagram_generation as diagram_generation
import models.import_ufo as import_ufo
import copy
import string

class TestGenerateLoopFKSFromReal(unittest.TestCase):
    """a class to test the generation of the virtual amps for a FKSMultiProcessFromReals"""

    def setUp(self):
        self.mymodel = import_ufo.import_model('loop_sm')
    

    def test_generate_virtuals_single_processR(self):
        """checks that the virtuals are correctly generated for a single process"""

        myleglist = MG.MultiLegList()
        
        # test process is u g > u g g 
        myleglist.append(MG.MultiLeg({'ids':[2], 'state':False}))
        myleglist.append(MG.MultiLeg({'ids':[-2], 'state':False}))
        myleglist.append(MG.MultiLeg({'ids':[2], 'state':True}))
        myleglist.append(MG.MultiLeg({'ids':[-2], 'state':True}))
        myleglist.append(MG.MultiLeg({'ids':[21], 'state':True}))
    
        myproc = MG.ProcessDefinition({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders': {'QED': 0},
                                           'perturbation_couplings':['QCD']})
        
        my_process_definitions = MG.ProcessDefinitionList([myproc])
        
        myfksmulti = fks_real.FKSMultiProcessFromReals(\
                {'process_definitions': my_process_definitions})

        self.assertEqual(myfksmulti['real_processes'][0].virt_amp, None)
        myfksmulti.generate_virtuals()
        #there should be 4 virt_amps
        self.assertNotEqual(myfksmulti['real_processes'][0].virt_amp, None)
        self.assertEqual([l['id'] for l in \
                        myfksmulti['real_processes'][0].virt_amp.get('process').get('legs')], \
                         [2,-2,2,-2])


    def test_generate_virtuals_multi_processR(self):
        """checks that the virtuals are correctly generated for a multi process"""

        p = [1, -1, 21]

        my_multi_leg = MG.MultiLeg({'ids': p, 'state': True});

        # Define the multiprocess
        my_multi_leglist = MG.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * 5])
        
        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)
        myproc = MG.ProcessDefinition({'legs':my_multi_leglist,
                                        'model':self.mymodel,
                                        'orders': {'QED': 0},
                                        'perturbation_couplings':['QCD']})

        my_process_definitions = MG.ProcessDefinitionList([myproc])
        
        myfksmulti = fks_real.FKSMultiProcessFromReals(\
                {'process_definitions': my_process_definitions})

        myfksmulti.generate_virtuals()
        for real in myfksmulti['real_processes']:
            pdgs = [l.get('id') for l in real.leglist] 
            # if the real process has soft singularities
            if 21 in pdgs[3:]:
                self.assertEqual([l['id'] for l in \
                    real.virt_amp.get('process').get('legs')], \
                    pdgs[:-1])
            else:
                self.assertEqual(real.virt_amp, None)
