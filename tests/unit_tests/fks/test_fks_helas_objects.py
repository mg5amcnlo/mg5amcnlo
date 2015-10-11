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

"""Testing modules for fks_helas_objects module"""

import sys
import os
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import tests.unit_tests as unittest
import madgraph.fks.fks_base as fks_base
import madgraph.fks.fks_common as fks_common
import madgraph.fks.fks_helas_objects as fks_helas
import madgraph.core.base_objects as MG
import madgraph.core.helas_objects as helas_objects
import madgraph.core.color_algebra as color
import madgraph.core.color_amp as color_amp
import madgraph.core.diagram_generation as diagram_generation
import copy
import array
import models.import_ufo as import_ufo

class testFKSHelasObjects(unittest.TestCase):
    """a class to test the module FKSHelasObjects"""

    def setUp(self):
        if not hasattr(self, 'mymodel') or \
           not hasattr(self, 'myleglist3') or \
           not hasattr(self, 'myproc1') or \
           not hasattr(self, 'myproc3'):

            myleglist1 = MG.LegList()
            # PROCESS: u g > u g 
            mylegs = [{'id': 2, 'number': 1, 'state': False},
                      {'id': 21, 'number': 2, 'state': False},
                      {'id': 2, 'number': 3, 'state': True},
                      {'id': 21, 'number': 4, 'state': True}]
            for i in mylegs:
                myleglist1.append(MG.Leg(i))
                
            myleglist3 = MG.LegList()
            # PROCESS: d d~ > u u~
            mylegs = [{'id': 1, 'number': 1, 'state': False},
                      {'id': -1, 'number': 2, 'state': False},
                      {'id': 2, 'number': 3, 'state': True},
                      {'id': -2, 'number': 4, 'state': True}]
            for i in mylegs:
                myleglist3.append(MG.Leg(i))

            mymodel = import_ufo.import_model('sm')

            dict1 = {'legs' : myleglist1, 'orders':{'QCD':10, 'QED':0},
                               'model': mymodel,
                               'id': 1,
                               'required_s_channels':[],
                               'forbidden_s_channels':[],
                               'forbidden_particles':[],
                               'is_decay_chain': False,
                               'orders': {'QED': 0, 'WEIGHTED':2},
                               'perturbation_couplings' : ['QCD'],
                               'decay_chains': MG.ProcessList(),
                               'overall_orders': {}}
            
            dict1_qed = {'legs' : myleglist1, 'orders':{'QCD':10, 'QED':0},
                               'model': mymodel,
                               'id': 1,
                               'required_s_channels':[],
                               'forbidden_s_channels':[],
                               'forbidden_particles':[],
                               'is_decay_chain': False,
                               'orders': {'QED': 0, 'WEIGHTED':2},
                               'perturbation_couplings' : ['QED'],
                               'decay_chains': MG.ProcessList(),
                               'overall_orders': {}}

            dict3 = {'legs' : myleglist3, 'orders':{'QCD':10, 'QED':0},
                               'model': mymodel,
                               'id': 1,
                               'required_s_channels':[],
                               'forbidden_s_channels':[],
                               'forbidden_particles':[],
                               'is_decay_chain': False,
                               'orders': {'QED': 0, 'WEIGHTED':2 },
                               'perturbation_couplings' : ['QCD'],
                               'decay_chains': MG.ProcessList(),
                               'overall_orders': {}}
            
            dict3_qed = {'legs' : myleglist3, 'orders':{'QCD':10, 'QED':0},
                               'model': mymodel,
                               'id': 1,
                               'required_s_channels':[],
                               'forbidden_s_channels':[],
                               'forbidden_particles':[],
                               'is_decay_chain': False,
                               'orders': {'QED': 0, 'WEIGHTED':2 },
                               'perturbation_couplings' : ['QED'],
                               'decay_chains': MG.ProcessList(),
                               'overall_orders': {}}
            
            testFKSHelasObjects.mymodel = mymodel
            testFKSHelasObjects.myleglist3 = myleglist3
            testFKSHelasObjects.myproc1 = MG.Process(dict1)
            testFKSHelasObjects.myproc3 = MG.Process(dict3)
            testFKSHelasObjects.myproc1_qed = MG.Process(dict1_qed)
            testFKSHelasObjects.myproc3_qed = MG.Process(dict3_qed)


    def test_fks_helas_multi_process_ppz(self):
        """tests the correct recycling of color infos for MEs with the same 
        color flow (e.g. uu~>z and dd~>z)
        """
        p= [21, 1, 2, 3, 4, -1, -2, -3, -4]
        z_leg= MG.MultiLeg({'ids':[23], 'state': True})
        p_leg = MG.MultiLeg({'ids': p, 'state': False});

        # Define the multiprocess
        my_multi_leglist = MG.MultiLegList([copy.copy(leg) for leg in [p_leg] * 2] \
                    + MG.MultiLegList([z_leg]))

        my_process_definition = MG.ProcessDefinition({ \
                        'orders': {'QED':1},
                        'legs': my_multi_leglist,
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])

        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions': my_process_definitions})
        my_helas_mp = fks_helas.FKSHelasMultiProcess(my_multi_process, gen_color = True)

        self.assertEqual(my_helas_mp['has_isr'], True)
        self.assertEqual(my_helas_mp['has_fsr'], False)

        for me in my_helas_mp['matrix_elements']:
            self.assertEqual(len(me.born_matrix_element['color_basis']), 1)
            self.assertEqual(me.get_nexternal_ninitial(), (4,2))


    def test_fks_helas_multi_process_ppz_loonly(self):
        """tests the LOonly NLO mode. In particular test that no
        reals are generated and that the get_nexternal_ninitial funciton
        returns the values as if the reals were generated.
        """
        p= [21, 1, 2, 3, 4, -1, -2, -3, -4]
        z_leg= MG.MultiLeg({'ids':[23], 'state': True})
        p_leg = MG.MultiLeg({'ids': p, 'state': False});

        # Define the multiprocess
        my_multi_leglist = MG.MultiLegList([copy.copy(leg) for leg in [p_leg] * 2] \
                    + MG.MultiLegList([z_leg]))

        my_process_definition = MG.ProcessDefinition({ \
                        'orders': {'QED':1},
                        'legs': my_multi_leglist,
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'LOonly',
                        'model': self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])

        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions': my_process_definitions})
        my_helas_mp = fks_helas.FKSHelasMultiProcess(my_multi_process, gen_color = True)

        for me in my_helas_mp['matrix_elements']:
            #
            self.assertEqual(len(me.real_processes), 0)
            self.assertEqual(me.get_nexternal_ninitial(), (4,2))


    def test_fks_ppzz_in_RS(self):
        """"""

        p = [21, 1, 2, 3, -1, -2, -3 ]
        z_leg = MG.MultiLeg({'ids':[23], 'state': True})
        p_leg = MG.MultiLeg({'ids': p, 'state': False});
        my_multi_leglist = MG.MultiLegList([copy.copy(leg) for leg in [p_leg] * 2] \
                    + MG.MultiLegList([z_leg, z_leg]))
        mymodel = import_ufo.import_model('RS')
        my_process_definition = MG.ProcessDefinition({ \
                        'orders': {'WEIGHTED': 4},
                        'legs': my_multi_leglist,
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])

        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions': my_process_definitions})
        for born in my_multi_process['born_processes']:
            born_pdg_list = [l['id'] for l in born.born_proc['legs']]
            if born_pdg_list[0] == 21:
            # gg initiated
                self.assertEqual(len(born.born_amp['diagrams']), 1)
                for amp in born.real_amps:
                    if amp.pdgs[0] != 21 or amp.pdgs[1] != 21:
                        self.assertEqual(len(amp.amplitude['diagrams']), 12)
                    else:
                        self.assertEqual(len(amp.amplitude['diagrams']), 4)
            else:
            # qq initiated
                self.assertEqual(len(born.born_amp['diagrams']), 4)
                for amp in born.real_amps:
                    self.assertEqual(len(amp.amplitude['diagrams']), 12)

        my_helas_mp = fks_helas.FKSHelasMultiProcess(my_multi_process, gen_color = False)
        for born in my_helas_mp['matrix_elements']:
            born_pdg_list = [l['id'] for l in born.born_matrix_element['base_amplitude']['process']['legs']]
            if born_pdg_list[0] == 21:
            # gg initiated
                self.assertEqual(len(born.born_matrix_element['diagrams']), 1)
                for real in born.real_processes:
                    pdgs = [l['id'] for l in real.matrix_element['base_amplitude']['process']['legs']]
                    if pdgs[0] != 21 or pdgs[1] != 21:
                        self.assertEqual(len(real.matrix_element['diagrams']), 12)
                    else:
                        self.assertEqual(len(real.matrix_element['diagrams']), 4)
            else:
            # qq initiated
                self.assertEqual(len(born.born_matrix_element['diagrams']), 4)
                for real in born.real_processes:
                    self.assertEqual(len(real.matrix_element['diagrams']), 12)



    def test_fks_helas_multi_process_ppwj(self):
        """tests the correct initialization of a FKSHelasMultiProcess, 
        given an FKSMultiProcess. This also checks that, when combining 
        2 FKSHelasProcess using the add_process function, the real
        emissions are combined consistently.
        The p p > w+ j process is studied
        """
        p= [21, 1, 2, 3, 4, -1, -2, -3, -4]
        w_leg= MG.MultiLeg({'ids':[24], 'state': True})
        j_leg= MG.MultiLeg({'ids':p, 'state': True})
        p_leg = MG.MultiLeg({'ids': p, 'state': False});

        # Define the multiprocess
        my_multi_leglist = MG.MultiLegList([copy.copy(leg) for leg in [p_leg] * 2] \
                    + MG.MultiLegList([w_leg, j_leg]))
        
        my_process_definition = MG.ProcessDefinition({ \
                        'orders': {'WEIGHTED': 3},
                        'legs': my_multi_leglist,
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])

        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions': my_process_definitions})
        my_helas_mp = fks_helas.FKSHelasMultiProcess(my_multi_process, gen_color = False)

        #there are 6  borns 
        self.assertEqual(len(my_helas_mp.get('matrix_elements')),6)
        #born processes are initiated by : gu, gdx, ug, udx, dxg, dxu
        n_real_processes = [8,8,8,6,8,6]
        real_subprocesses = \
                [     [ #these are for gu -initiated born
                    [ [21,2,24,1,21], [21,4,24,3,21] ], #subrpocs for real 1
                    [ [-1,2,24,1,-1], [-3,4,24,3,-3] ], #subrpocs for real 2
                    [ [1,2,24,1,1], [3,4,24,3,3] ],     #subrpocs for real 3
                    [ [-3,2,24,1,-3], [-4,2,24,1,-4], [-1,4,24,3,-1], [-2,4,24,3,-2] ], #subrpocs for real 4
                    [ [3,2,24,1,3], [4,2,24,1,4], [1,4,24,3,1], [2,4,24,3,2] ], #subrpocs for real 5
                    [ [-2,2,24,1,-2], [-4,4,24,3,-4] ], #subrpocs for real 6
                    [ [2,2,24,1,2], [4,4,24,3,4] ],     #subrpocs for real 7
                    [ [21,21,24,1,-2], [21,21,24,3,-4] ]#subrpocs for real 8
                    ],[ #these are for gdx-initiated born
                    [ [21,-1,24,-2,21], [21,-3,24,-4,21] ], #subrpocs for real 1
                    [ [-1,-1,24,-2,-1], [-3,-3,24,-4,-3] ], #subrpocs for real 2
                    [ [1,-1,24,1,-2], [3,-3,24,3,-4] ],     #subrpocs for real 3
                    [ [-3,-1,24,-2,-3], [-4,-1,24,-2,-4], [-1,-3,24,-4,-1], [-2,-3,24,-4,-2] ], #subrpocs for real 4
                    [ [3,-1,24,3,-2], [4,-1,24,4,-2], [1,-3,24,1,-4], [2,-3,24,2,-4] ], #subrpocs for real 5
                    [ [-2,-1,24,-2,-2], [-4,-3,24,-4,-4] ], #subrpocs for real 6
                    [ [2,-1,24,2,-2], [4,-3,24,4,-4] ],     #subrpocs for real 7
                    [ [21,21,24,1,-2], [21,21,24,3,-4] ]#subrpocs for real 8
                    ],[ #these are for ug -initiated born
                    [ [2,21,24,1,21], [4,21,24,3,21] ], #subrpocs for real 1
                    [ [21,21,24,1,-2], [21,21,24,3,-4] ],#subrpocs for real 2
                    [ [2,-1,24,1,-1], [4,-3,24,3,-3] ], #subrpocs for real 3
                    [ [2,1,24,1,1], [4,3,24,3,3] ],     #subrpocs for real 4
                    [ [2,-3,24,1,-3], [2,-4,24,1,-4], [4,-1,24,3,-1], [4,-2,24,3,-2] ], #subrpocs for real 5
                    [ [2,3,24,1,3], [2,4,24,1,4], [4,1,24,3,1], [4,2,24,3,2] ], #subrpocs for real 6
                    [ [2,-2,24,1,-2], [4,-4,24,3,-4] ], #subrpocs for real 7
                    [ [2,2,24,1,2], [4,4,24,3,4] ]     #subrpocs for real 8
                    ],[ #these are for udx-initiated born
                    [ [2,-1,24,21,21], [4,-3,24,21,21] ], #subrpocs for real 1
                    [ [21,-1,24,-2,21], [21,-3,24,-4,21] ],#subrpocs for real 2
                    [ [2,21,24,1,21], [4,21,24,3,21] ], #subrpocs for real 3
                    [ [2,-1,24,1,-1], [4,-3,24,3,-3] ],     #subrpocs for real 4
                    [ [2,-1,24,3,-3], [2,-1,24,4,-4], [4,-3,24,1,-1], [4,-3,24,2,-2] ], #subrpocs for real 5
                    [ [2,-1,24,2,-2], [4,-3,24,4,-4]] #subrpocs for real 6
                    ],[ #these are for dxg-initiated born
                    [ [-1,21,24,-2,21], [-3,21,24,-4,21] ], #subrpocs for real 1
                    [ [21,21,24,1,-2], [21,21,24,3,-4] ],#subrpocs for real 2
                    [ [-1,-1,24,-2,-1], [-3,-3,24,-4,-3] ], #subrpocs for real 3
                    [ [-1,1,24,1,-2], [-3,3,24,3,-4] ],     #subrpocs for real 4
                    [ [-1,-3,24,-2,-3], [-1,-4,24,-2,-4], [-3,-1,24,-4,-1], [-3,-2,24,-4,-2] ], #subrpocs for real 5
                    [ [-1,3,24,3,-2], [-1,4,24,4,-2], [-3,1,24,1,-4], [-3,2,24,2,-4] ], #subrpocs for real 6
                    [ [-1,-2,24,-2,-2], [-3,-4,24,-4,-4] ], #subrpocs for real 7
                    [ [-1,2,24,2,-2], [-3,4,24,4,-4] ]     #subrpocs for real 8
                    ],[ #these are for dxu-initiated born
                    [ [-1,2,24,21,21], [-3,4,24,21,21] ], #subrpocs for real 1
                    [ [21,2,24,1,21], [21,4,24,3,21] ],#subrpocs for real 2
                    [ [-1,21,24,-2,21], [-3,21,24,-4,21] ], #subrpocs for real 3
                    [ [-1,2,24,1,-1], [-3,4,24,3,-3] ],     #subrpocs for real 4
                    [ [-1,2,24,3,-3], [-1,2,24,4,-4], [-3,4,24,1,-1], [-3,4,24,2,-2] ], #subrpocs for real 5
                    [ [-1,2,24,2,-2], [-3,4,24,4,-4]] #subrpocs for real 6
                ]]
                  
        #each born correspond to 2 partonic processes
        for i, me in enumerate(my_helas_mp.get('matrix_elements')):
            # gu and gc
            self.assertEqual(len(me.get('processes')),2)
            self.assertEqual(len(me.real_processes), n_real_processes[i])
            for j,real in enumerate(me.real_processes):
                pdgs = [ [leg['id'] for leg in proc['legs'] ] for proc in real.matrix_element['processes']]
                self.assertEqual(real_subprocesses[i][j], pdgs)


    def test_fks_helas_multi_process_pptt(self):
        """tests the correct initialization of a FKSHelasMultiProcess, 
        given an FKSMultiProcess. This test also checks that each real
        process corresponds to the correct number of FKS configurations.
        The p p > t t~ process is studied
        """
        p= [21, 1, 2, 3, 4, -1, -2, -3, -4]
        t= MG.MultiLeg({'ids':[6], 'state': True})
        tx= MG.MultiLeg({'ids':[-6], 'state': True})

        p_leg = MG.MultiLeg({'ids': p, 'state': False});

        # Define the multiprocess
        my_multi_leglist = MG.MultiLegList([copy.copy(leg) for leg in [p_leg] * 2] \
                    + MG.MultiLegList([t, tx]))
        
        my_process_definition = MG.ProcessDefinition({ \
                        'orders': {'WEIGHTED': 2},
                        'legs': my_multi_leglist,
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])

        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions': my_process_definitions})
        my_helas_mp = fks_helas.FKSHelasMultiProcess(my_multi_process, False)
        
        #there are 3 (gg uux uxu initiated) borns 
        self.assertEqual(len(my_helas_mp.get('matrix_elements')),3)
        # and 25 real matrix elements
        self.assertEqual(len(my_helas_mp.get('real_matrix_elements')), 25)
        # the first me is gg tt, with 5 different real emissions
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[0].real_processes), 5)
        # the first real emission corresponds to gg ttxg, with 4 different configs
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[0].real_processes[0].matrix_element['processes']), 1)
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[0].real_processes[0].fks_infos), 4)
        # for the 2nd to the 5th real emissions, corresponding to the q g > t tx g and crossings
        # there is only one config per processes, and the 4 quark flavours should be combined together
        for real in my_helas_mp.get('matrix_elements')[0].real_processes[1:]:
            self.assertEqual(len(real.matrix_element['processes']), 4)
            self.assertEqual(len(real.fks_infos), 1)

        # the 2nd me is uux tt, with 3 different real emissions
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[1].real_processes), 3)
        # the first real emission corresponds to qqx ttxg, with 4 different configs
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[1].real_processes[0].matrix_element['processes']), 4)
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[1].real_processes[0].fks_infos), 4)
        # the 2nd and 3rd real emission corresponds to qg ttxq (and gqx...), with 1 config
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[1].real_processes[1].matrix_element['processes']), 4)
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[1].real_processes[1].fks_infos), 1)
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[1].real_processes[2].matrix_element['processes']), 4)
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[1].real_processes[2].fks_infos), 1)

        # the 3rd me is uxu tt, with 3 different real emissions
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[2].real_processes), 3)
        # the first real emission corresponds to qxq ttxg, with 4 different configs
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[2].real_processes[0].matrix_element['processes']), 4)
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[2].real_processes[0].fks_infos), 4)
        # the 2nd and 3rd real emission corresponds to qxg ttxqx (and gq...), with 1 config
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[2].real_processes[1].matrix_element['processes']), 4)
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[2].real_processes[1].fks_infos), 1)
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[2].real_processes[2].matrix_element['processes']), 4)
        self.assertEqual(len(my_helas_mp.get('matrix_elements')[2].real_processes[2].fks_infos), 1)
        
    
    def test_fks_helas_real_process_init(self):
        """tests the correct initialization of an FKSHelasRealProcess, from a 
        FKSRealProc. The process uu~>dd~ is used as born.
        For the real we use dd~>uu~(j) g(i) and dd~>uu~(j)a(i).
        We test The correct initialization of:
        --i/j fks
        --permutation
        --matrix element
        """         
        #dd~> uu~
        fks3 = fks_base.FKSProcess(self.myproc3)
        fks3_qed = fks_base.FKSProcess(self.myproc3_qed)
        fksleglist = copy.copy(fks_common.to_fks_legs(self.myleglist3,
                                                      self.mymodel))
        fksleglist_qed = copy.copy(fksleglist)
        amplist = []
        amp_id_list = []
        me_list=[]
        me_id_list=[]
        amplist_qed = []
        amp_id_list_qed = []
        me_list_qed = []
        me_id_list_qed = []
        
        fksleglist.append(fks_common.to_fks_leg(MG.Leg({'id' : 21,
                                                 'state' : True,
                                                 'number' : 5,
                                                 'from_group' : True}),
                                                 self.mymodel))
        fksleglist_qed.append(fks_common.to_fks_leg(MG.Leg({'id' : 22,
                                                 'state' : True,
                                                 'number' : 5,
                                                 'from_group' : True}),
                                                 self.mymodel))
        
        fksleglist[0]['fks']='n'
        fksleglist[1]['fks']='n'
        fksleglist[2]['fks']='n'
        fksleglist[3]['fks']='j'
        fksleglist[4]['fks']='i'
        fksleglist_qed[0]['fks']='n'
        fksleglist_qed[1]['fks']='n'
        fksleglist_qed[2]['fks']='n'
        fksleglist_qed[3]['fks']='j'
        fksleglist_qed[4]['fks']='i'
        
        real_proc = fks_base.FKSRealProcess(fks3.born_proc, fksleglist, 4, 0,\
                                            perturbed_orders = ['QCD'])
        real_proc.generate_real_amplitude()
        helas_real_proc = fks_helas.FKSHelasRealProcess(real_proc, me_list, me_id_list)
        real_proc_qed = fks_base.FKSRealProcess(fks3_qed.born_proc, fksleglist_qed, 4, 0,\
                                            perturbed_orders = ['QED'])
        real_proc_qed.generate_real_amplitude()
        helas_real_proc_qed = fks_helas.FKSHelasRealProcess(real_proc_qed, me_list_qed, me_id_list_qed)
        self.assertEqual(helas_real_proc.fks_infos,
                [{'i':5, 'j':4, 'ij':4, 'ij_glu':0, 'need_color_links': True}])
        self.assertEqual(helas_real_proc_qed.fks_infos,
                [{'i':5, 'j':4, 'ij':4, 'ij_glu':0, 'need_color_links': True}])
        target_me = helas_objects.HelasMatrixElement(real_proc.amplitude)
        target_me_qed = helas_objects.HelasMatrixElement(real_proc_qed.amplitude)
        self.assertEqual(helas_real_proc.matrix_element, target_me)
        self.assertEqual(helas_real_proc.matrix_element.get('color_matrix'), 
                         target_me.get('color_matrix'))
        self.assertEqual(helas_real_proc_qed.matrix_element, target_me_qed)
        self.assertEqual(helas_real_proc_qed.matrix_element.get('color_matrix'), 
                         target_me_qed.get('color_matrix'))
        
        
    def test_fks_helas_process_init(self):
        """tests the correct initialization of a FKSHelasProcess object.
        in particular checks:
        -- born ME
        -- list of FKSHelasRealProcesses
        -- color links
        -- fks_infos
        """
        #ug> ug
        fks1 = fks_base.FKSProcess(self.myproc1)
        #ug>gu
        fks1_qed = fks_base.FKSProcess(self.myproc1_qed)
        #dd~> uu~
        fks3 = fks_base.FKSProcess(self.myproc3)
        
        pdg_list1 = []
        real_amp_list1 = diagram_generation.AmplitudeList()
        pdg_list1_qed = []
        real_amp_list1_qed = diagram_generation.AmplitudeList()
        pdg_list3 = [] 
        real_amp_list3 = diagram_generation.AmplitudeList()
        fks1.generate_reals(pdg_list1, real_amp_list1)
        fks1_qed.generate_reals(pdg_list1_qed, real_amp_list1_qed)
        fks3.generate_reals(pdg_list3, real_amp_list3)

        me_list=[]
        me_id_list=[]
        me_list_qed=[]
        me_id_list_qed=[]
        me_list3=[]
        me_id_list3=[]
        res_me_list=[]
        res_me_id_list=[]

        helas_born_proc = fks_helas.FKSHelasProcess(
                                    fks1, me_list, me_id_list)
        helas_born_proc_qed = fks_helas.FKSHelasProcess(
                                    fks1_qed, me_list_qed, me_id_list_qed)
        helas_born_proc3 = fks_helas.FKSHelasProcess(
                                    fks3, me_list3, me_id_list3)
        
        self.assertEqual(helas_born_proc.born_matrix_element,
                          helas_objects.HelasMatrixElement(
                                    fks1.born_amp))
        self.assertEqual(helas_born_proc_qed.born_matrix_element,
                          helas_objects.HelasMatrixElement(
                                    fks1_qed.born_amp))
        res_reals = []
        for real in fks1.real_amps:
            res_reals.append(fks_helas.FKSHelasRealProcess(
                                real,res_me_list, res_me_id_list))
            # the process u g > u g g corresponds to 4 fks configs
            # it is testing fks_base.FKSProcess.combine_real_amplitudes
            if real.pdgs == array.array('i', [2,21,2,21,21]):
                self.assertEqual(len(real.fks_infos), 4)
            else:
            # any other process has only 1 fks config
                self.assertEqual(len(real.fks_infos), 1)

        self.assertEqual(me_list, res_me_list)
        self.assertEqual(me_id_list, res_me_id_list)
        self.assertEqual(8, len(helas_born_proc.real_processes))
        self.assertNotEqual(helas_born_proc.born_matrix_element,
                            helas_born_proc3.born_matrix_element)
        self.assertNotEqual(helas_born_proc.born_matrix_element,\
                            helas_born_proc_qed.born_matrix_element)
        for a,b in zip(helas_born_proc3.real_processes, 
                    helas_born_proc.real_processes):
            self.assertNotEqual(a,b)

    def test_set_color_links(self):
        """tests that the set_color_links of a FKSHelasProcess
        returns the correct list of color_links."""
        #ug> ug
        fks1 = fks_base.FKSProcess(self.myproc1)
        #ug>gu
        fks1_qed = fks_base.FKSProcess(self.myproc1_qed)
        me_list=[]
        me_list_qed = []
        me_id_list=[]
        me_id_list_qed = []
        pdg_list1 = []
        pdg_list1_qed = []
        real_amp_list1 = diagram_generation.AmplitudeList()
        real_amp_list1_qed = diagram_generation.AmplitudeList()
        fks1.generate_reals(pdg_list1, real_amp_list1)
        fks1_qed.generate_reals(pdg_list1_qed,real_amp_list1_qed)
        helas_born_proc = fks_helas.FKSHelasProcess(
                                    fks1, me_list, me_id_list)
        helas_born_proc_qed = fks_helas.FKSHelasProcess(\
                                    fks1_qed,me_list_qed,me_id_list_qed)
        helas_born_proc.set_color_links()
        helas_born_proc_qed.set_color_links()
        legpair = [link['link'] for link in helas_born_proc.color_links]
        legpair_qed = [link['link'] for link in helas_born_proc_qed.color_links]
        tar_legpair = [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
        tar_legpair_qed = [[1,4]]
        self.assertEqual(legpair,tar_legpair)
        self.assertEqual(legpair_qed,tar_legpair_qed)
        
    def test_get_fks_info_list(self):
        """tests that the get_fks_info_list of a FKSHelasProcess 
        returns the correct list of configurations/fks_configs"""
        
        #ug> ug
        fks1 = fks_base.FKSProcess(self.myproc1)
        me_list=[]
        me_id_list=[]
        pdg_list1 = []
        real_amp_list1 = diagram_generation.AmplitudeList()
        fks1.generate_reals(pdg_list1, real_amp_list1)
        helas_born_proc = fks_helas.FKSHelasProcess(
                                    fks1, me_list, me_id_list)
        goal = \
            [
             {'n_me' : 1, 'pdgs':[2,21,2,21,21], \
                 'fks_info': {'i':5, 'j':1, 'ij':1, 'ij_glu':0, 'need_color_links':True,
                              'rb_links': [{'born_conf': 0, 'real_conf': 11},
                                           {'born_conf': 1, 'real_conf': 10},
                                           {'born_conf': 2, 'real_conf': 9}]}},
             {'n_me' : 1, 'pdgs':[2,21,2,21,21], \
                 'fks_info': {'i':5, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':True,
                              'rb_links': [{'born_conf': 0, 'real_conf': 14},
                                           {'born_conf': 1, 'real_conf': 4},
                                           {'born_conf': 2, 'real_conf': 7}]}},
             {'n_me' : 1, 'pdgs':[2,21,2,21,21], \
                 'fks_info': {'i':5, 'j':3, 'ij':3, 'ij_glu':0, 'need_color_links':True,
                              'rb_links': [{'born_conf': 0, 'real_conf': 1},
                                           {'born_conf': 1, 'real_conf': 13},
                                           {'born_conf': 2, 'real_conf': 8}]}},
             {'n_me' : 1, 'pdgs':[2,21,2,21,21], \
                 'fks_info': {'i':5, 'j':4, 'ij':4, 'ij_glu':4, 'need_color_links':True,
                              'rb_links': [{'born_conf': 0, 'real_conf': 2},
                                           {'born_conf': 1, 'real_conf': 5},
                                           {'born_conf': 2, 'real_conf': 12}]}},
             {'n_me' : 2, 'pdgs':[21,21,2,-2,21], \
                 'fks_info': {'i':4, 'j':1, 'ij':1, 'ij_glu':0, 'need_color_links':False,
                              'rb_links': [{'born_conf': 0, 'real_conf': 8},
                                           {'born_conf': 1, 'real_conf': 7},
                                           {'born_conf': 2, 'real_conf': 6}]}},
             {'n_me' : 3, 'pdgs':[2,-1,2,-1,21], \
                 'fks_info': {'i':4, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':False,
                              'rb_links': [{'born_conf': 0, 'real_conf': 4},
                                           {'born_conf': 1, 'real_conf': 0},
                                           {'born_conf': 2, 'real_conf': 3}]}},
             {'n_me' : 4, 'pdgs':[2,1,2,1,21], \
                 'fks_info': {'i':4, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':False,
                              'rb_links': [{'born_conf': 0, 'real_conf': 4},
                                           {'born_conf': 1, 'real_conf': 0},
                                           {'born_conf': 2, 'real_conf': 3}]}},
             {'n_me' : 5, 'pdgs':[2,-2,2,-2,21], \
                 'fks_info': {'i':4, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':False,
                              'rb_links': [{'born_conf': 0, 'real_conf': 8},
                                           {'born_conf': 1, 'real_conf': 3},
                                           {'born_conf': 2, 'real_conf': 6}]}},
             {'n_me' : 6, 'pdgs':[2,2,2,2,21], \
                 'fks_info': {'i':4, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':False,
                              'rb_links': [{'born_conf': 0, 'real_conf': 9},
                                           {'born_conf': 1, 'real_conf': 0},
                                           {'born_conf': 2, 'real_conf': 7}]}},
             {'n_me' : 7, 'pdgs':[2,21,2,1,-1], \
                 'fks_info': {'i':5, 'j':4, 'ij':4, 'ij_glu':4, 'need_color_links':False,
                              'rb_links': [{'born_conf': 0, 'real_conf': 0},
                                           {'born_conf': 1, 'real_conf': 3},
                                           {'born_conf': 2, 'real_conf': 4}]}},
             {'n_me' : 8, 'pdgs':[2,21,2,2,-2], \
                 'fks_info': {'i':5, 'j':4, 'ij':4, 'ij_glu':4, 'need_color_links':False,
                              'rb_links': [{'born_conf': 0, 'real_conf': 1},
                                           {'born_conf': 1, 'real_conf': 4},
                                           {'born_conf': 2, 'real_conf': 8}]}},
             ]
        for a, b in zip(goal, helas_born_proc.get_fks_info_list()):
            self.assertEqual(a,b)

        self.assertEqual(goal, helas_born_proc.get_fks_info_list())

        
