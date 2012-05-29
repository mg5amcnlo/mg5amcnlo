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

"""Testing modules for fks_born_helas_objects module"""

import sys
import os
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import tests.unit_tests as unittest
import madgraph.fks.fks_born as fks_born
import madgraph.fks.fks_common as fks_common
import madgraph.fks.fks_born_helas_objects as fks_born_helas
import madgraph.core.base_objects as MG
import madgraph.core.helas_objects as helas_objects
import madgraph.core.color_algebra as color
import madgraph.core.color_amp as color_amp
import madgraph.core.diagram_generation as diagram_generation
import copy
import array
import models.import_ufo as import_ufo

class testFKSBornHelasObjects(unittest.TestCase):
    """a class to test the module FKSBornHelasObjects"""
    myleglist1 = MG.LegList()
    # PROCESS: u g > u g 
    mylegs = [{ \
    'id': 2,\
    'number': 1,\
    'state': False,\
 #   'from_group': True \
}, \
{ \
    'id': 21,\
    'number': 2,\
    'state': False,\
    #'from_group': True\
},\
{\
    'id': 2,\
    'number': 3,\
    'state': True,\
  #  'from_group': True\
},\
{\
    'id': 21,\
    'number': 4,\
    'state': True,\
   # 'from_group': True\
}
]

    for i in mylegs:
        myleglist1.append(MG.Leg(i))
        
    myleglist2 = MG.LegList()
    # PROCESS: d g > d g 
    mylegs = [{ \
    'id': 1,\
    'number': 1,\
    'state': False,\
 #   'from_group': True \
}, \
{ \
    'id': 21,\
    'number': 2,\
    'state': False,\
    #'from_group': True\
},\
{\
    'id': 1,\
    'number': 3,\
    'state': True,\
  #  'from_group': True\
},\
{\
    'id': 21,\
    'number': 4,\
    'state': True,\
   # 'from_group': True\
}
]

    for i in mylegs:
        myleglist2.append(MG.Leg(i))

    myleglist3 = MG.LegList()
    # PROCESS: d d~ > u u~
    mylegs = [{ \
    'id': 1,\
    'number': 1,\
    'state': False,\
 #   'from_group': True \
}, \
{ \
    'id': -1,\
    'number': 2,\
    'state': False,\
    #'from_group': True\
},\
{\
    'id': 2,\
    'number': 3,\
    'state': True,\
  #  'from_group': True\
},\
{\
    'id': -2,\
    'number': 4,\
    'state': True,\
   # 'from_group': True\
}
]
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
                       'perturbation_couplings' : ['QCD'],
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}


    dict2 = {'legs' : myleglist2, 'orders':{'QCD':10, 'QED':0},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'perturbation_couplings' : ['QCD'],
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}

    dict3 = {'legs' : myleglist3, 'orders':{'QCD':10, 'QED':0},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'perturbation_couplings' : ['QCD'],
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}
    
    myproc1 = MG.Process(dict1)
    myproc1.set('orders', {'QED':0})
    myproc2 = MG.Process(dict2)
    myproc2.set('orders', {'QED':0})
    myproc3 = MG.Process(dict3)
    myproc3.set('orders', {'QED':0})


    def test_fks_helas_multi_process_from_born(self):
        """tests the correct initialization of a FKSHelasMultiProcess, 
        given an FKSMultiProcess"""
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

        my_multi_process = fks_born.FKSMultiProcessFromBorn(\
                {'process_definitions': my_process_definitions})
        my_helas_mp = fks_born_helas.FKSHelasMultiProcessFromBorn(my_multi_process, False)
        
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
        For the real we use uu~>dd~(j) g(i).
        We test The correct initialization of:
        --i/j fks
        --permutation
        --matrix element
        """         
        #ug> ug
        fks1 = fks_born.FKSProcessFromBorn(self.myproc1)
        #dg> dg
        fks2 = fks_born.FKSProcessFromBorn(self.myproc2)
        #uu~> dd~
        fks3 = fks_born.FKSProcessFromBorn(self.myproc3)
 
        fksleglist = copy.copy(fks_common.to_fks_legs(self.myleglist3,
                                                      self.mymodel))
        amplist = []
        amp_id_list = []
        me_list=[]
        me_id_list=[]
        
        fksleglist.append(fks_common.to_fks_leg(MG.Leg({'id' : 21,
                                                 'state' : True,
                                                 'number' : 5,
                                                 'from_group' : True}),
                                                 self.mymodel))
        fksleglist[0]['fks']='n'
        fksleglist[1]['fks']='n'
        fksleglist[2]['fks']='n'
        fksleglist[3]['fks']='j'
        fksleglist[4]['fks']='i'
        
        real_proc = fks_born.FKSRealProcess(fks3.born_proc, fksleglist, 4, 0)
        real_proc.generate_real_amplitude()
        helas_real_proc = fks_born_helas.FKSHelasRealProcess(real_proc, me_list, me_id_list)
        self.assertEqual(helas_real_proc.fks_infos,
                [{'i':5, 'j':4, 'ij':4, 'ij_glu':0, 'need_color_links': True}])
        target_me = helas_objects.HelasMatrixElement(real_proc.amplitude)
        self.assertEqual(helas_real_proc.matrix_element, target_me)
        self.assertEqual(helas_real_proc.matrix_element.get('color_matrix'), 
                         target_me.get('color_matrix'))
        
        
    def test_fks_helas_process_from_born_init(self):
        """tests the correct initialization of a FKSHelasProcessFromBorn object.
        in particular checks:
        -- born ME
        -- list of FKSHelasRealProcesses
        -- color links
        we use the born process ug > ug"""
        
        #ug> ug
        fks1 = fks_born.FKSProcessFromBorn(self.myproc1)
        #dg> dg
        fks2 = fks_born.FKSProcessFromBorn(self.myproc2)
        #uu~> dd~
        fks3 = fks_born.FKSProcessFromBorn(self.myproc3)
        
        pdg_list1 = []
        real_amp_list1 = diagram_generation.AmplitudeList()
        pdg_list3 = [] 
        real_amp_list3 = diagram_generation.AmplitudeList()
        fks1.generate_reals(pdg_list1, real_amp_list1)
        fks3.generate_reals(pdg_list3, real_amp_list3)

        me_list=[]
        me_id_list=[]
        me_list3=[]
        me_id_list3=[]
        res_me_list=[]
        res_me_id_list=[]


        helas_born_proc = fks_born_helas.FKSHelasProcessFromBorn(
                                    fks1, me_list, me_id_list)
        helas_born_proc3 = fks_born_helas.FKSHelasProcessFromBorn(
                                    fks3, me_list3, me_id_list3)
        
        self.assertEqual(helas_born_proc.born_matrix_element,
                          helas_objects.HelasMatrixElement(
                                    fks1.born_amp))
        res_reals = []
        for real in fks1.real_amps:
            res_reals.append(fks_born_helas.FKSHelasRealProcess(
                                real,res_me_list, res_me_id_list))
            # the process u g > u g g corresponds to 4 fks configs
            if real.pdgs == array.array('i', [2,21,2,21,21]):
                self.assertEqual(len(real.fks_infos), 4)
            else:
            # any other process has only 1 fks config
                self.assertEqual(len(real.fks_infos), 1)

        self.assertEqual(me_list, res_me_list)
        self.assertEqual(me_id_list, res_me_id_list)
        self.assertEqual(8, len(helas_born_proc.real_processes))
#        for a,b in zip(res_reals, helas_born_proc.real_processes):
#            self.assertEqual(a,b)
        #more (in)sanity checks
        self.assertNotEqual(helas_born_proc.born_matrix_element,
                            helas_born_proc3.born_matrix_element)
        for a,b in zip(helas_born_proc3.real_processes, 
                    helas_born_proc.real_processes):
            self.assertNotEqual(a,b)


    def test_get_fks_info_list(self):
        """tests that the get_fks_info_list of a FKSHelasProcessFromBorn 
        returns the correct list of configurations/fks_configs"""
        
        #ug> ug
        fks1 = fks_born.FKSProcessFromBorn(self.myproc1)
        me_list=[]
        me_id_list=[]
        pdg_list1 = []
        real_amp_list1 = diagram_generation.AmplitudeList()
        fks1.generate_reals(pdg_list1, real_amp_list1)
        helas_born_proc = fks_born_helas.FKSHelasProcessFromBorn(
                                    fks1, me_list, me_id_list)
        goal = \
            [
             {'n_me' : 1, 'pdgs':[2,21,2,21,21], \
                 'fks_info': {'i':5, 'j':1, 'ij':1, 'ij_glu':0, 'need_color_links':True}},
             {'n_me' : 1, 'pdgs':[2,21,2,21,21], \
                 'fks_info': {'i':5, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':True}},
             {'n_me' : 1, 'pdgs':[2,21,2,21,21], \
                 'fks_info': {'i':5, 'j':3, 'ij':3, 'ij_glu':0, 'need_color_links':True}},
             {'n_me' : 1, 'pdgs':[2,21,2,21,21], \
                 'fks_info': {'i':5, 'j':4, 'ij':4, 'ij_glu':4, 'need_color_links':True}},
             {'n_me' : 2, 'pdgs':[21,21,-2,2,21], \
                 'fks_info': {'i':3, 'j':1, 'ij':1, 'ij_glu':0, 'need_color_links':False}},
             {'n_me' : 3, 'pdgs':[2,-1,-1,2,21], \
                 'fks_info': {'i':3, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':False}},
             {'n_me' : 4, 'pdgs':[2,1,1,2,21], \
                 'fks_info': {'i':3, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':False}},
             {'n_me' : 5, 'pdgs':[2,-2,-2,2,21], \
                 'fks_info': {'i':3, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':False}},
             {'n_me' : 6, 'pdgs':[2,2,2,2,21], \
                 'fks_info': {'i':4, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':False}},
             {'n_me' : 7, 'pdgs':[2,21,2,1,-1], \
                 'fks_info': {'i':5, 'j':4, 'ij':4, 'ij_glu':4, 'need_color_links':False}},
             {'n_me' : 8, 'pdgs':[2,21,2,2,-2], \
                 'fks_info': {'i':5, 'j':4, 'ij':4, 'ij_glu':4, 'need_color_links':False}},
             ]
        for a, b in zip(goal, helas_born_proc.get_fks_info_list()):
            self.assertEqual(a,b)

        self.assertEqual(goal, helas_born_proc.get_fks_info_list())

        
    def test_fks_helas_process_from_born_add_process(self):
        """test the correct implementation of the add process function, which
        should add both the born and the reals.
        We consider the process ug>ug and dg>dg, which 11 reals each"""
        me_list=[]
        me_id_list=[]
        res_me_list=[]
        res_me_id_list=[]
        
        #ug> ug
        fks1 = fks_born.FKSProcessFromBorn(self.myproc1)
        #dg> dg
        fks2 = fks_born.FKSProcessFromBorn(self.myproc2)
        #uu~> dd~
        fks3 = fks_born.FKSProcessFromBorn(self.myproc3)

        pdg_list1 = []
        real_amp_list1 = diagram_generation.AmplitudeList()
        fks1.generate_reals(pdg_list1, real_amp_list1)
        pdg_list2 = []
        real_amp_list2 = diagram_generation.AmplitudeList()
        fks2.generate_reals(pdg_list2, real_amp_list2)

        helas_born_proc1 = copy.deepcopy(fks_born_helas.FKSHelasProcessFromBorn(
                                    fks1, me_list, me_id_list))
        helas_born_proc2 = copy.deepcopy(fks_born_helas.FKSHelasProcessFromBorn(
                                    fks2, me_list, me_id_list))
        
        #check the correct adding of the reals
        born_1 =copy.deepcopy(helas_born_proc1.born_matrix_element.get('processes'))
        born_2 =copy.deepcopy(helas_born_proc2.born_matrix_element.get('processes'))
        
        
        reals_1= []
        for me in helas_born_proc1.real_processes:
            reals_1.append(copy.deepcopy(me.matrix_element.get('processes')))
        reals_2= []
        for me in helas_born_proc2.real_processes:
            reals_2.append(copy.deepcopy(me.matrix_element.get('processes')))
        self.assertEqual(len(reals_1),8)
        self.assertEqual(len(reals_2),8)
        helas_born_proc1.add_process(helas_born_proc2)
        
        born_1.extend(born_2)
        self.assertEqual(len(born_1), 
            len(helas_born_proc1.born_matrix_element.get('processes')))
        self.assertEqual(born_1, 
            helas_born_proc1.born_matrix_element.get('processes'))
      
        
        for me, r1, r2 in  zip(helas_born_proc1.real_processes, reals_1, reals_2):
            r1.extend(r2)
            self.assertEqual(len(me.matrix_element.get('processes')),
                             len(r1)
                             )
            
