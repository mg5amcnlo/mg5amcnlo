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

    
    mypartlist = MG.ParticleList()
    mypartlistbad = MG.ParticleList()
    myinterlist = MG.InteractionList()
    myinterlistbad = MG.InteractionList()
    mypartlist.append(MG.Particle({'name':'u',
                  'antiname':'u~',
                  'spin':2,
                  'color':3,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'u',
                  'antitexname':'\\overline{u}',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':2,
                  'propagating':True,
                  #'is_part': True,
                  'self_antipart':False}))
    mypartlist.append(MG.Particle({'name':'d',
                  'antiname':'d~',
                  'spin':2,
                  'color':3,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'d',
                  'antitexname':'\\overline{d}',
                  'line':'straight',
                  'charge':-1. / 3.,
                  'pdg_code':1,
                  #'is_part': True,
                  'propagating':True,
                  'self_antipart':False}))
    mypartlist.append(MG.Particle({'name':'g',
                      'antiname':'g',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':21,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

    mypartlist.append(MG.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\gamma',
                      'antitexname':'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
    
    mypartlist.append(MG.Particle({'name':'t',
                  'antiname':'t~',
                  'spin':2,
                  'color':3,
                  'mass':'tmass',
                  'width':'twidth',
                  'texname':'t',
                  'antitexname':'\\overline{t}',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':6,
                  'propagating':True,
                  #'is_part': True,
                  'self_antipart':False}))
        
    antiu = MG.Particle({'name':'u',
                  'antiname':'u~',
                  'spin':2,
                  'color': 3,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'u',
                  'antitexname':'\\overline{u}',
                  'line':'straight',
                  'charge':  2. / 3.,
                  'pdg_code': 2,
                  'propagating':True,
                  'is_part':False,
                  'self_antipart':False})
  #  mypartlist.append(antiu)

    
    antid = MG.Particle({'name':'d',
                  'antiname':'d~',
                  'spin':2,
                  'color':3,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'d',
                  'antitexname':'\\overline{d}',
                  'line':'straight',
                  'charge':-1. / 3.,
                  'pdg_code':1,
                  'is_part': False,
                  'propagating':True,
                  'self_antipart':False})
    
    antit = MG.Particle({'name':'t',
                  'antiname':'t~',
                  'spin':2,
                  'color':3,
                  'mass':'tmass',
                  'width':'twidth',
                  'texname':'t',
                  'antitexname':'\\overline{t}',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':6,
                  'propagating':True,
                  'is_part': False,
                  'self_antipart':False})
    
        
    myinterlist.append(MG.Interaction({\
                      'id':1,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[1], \
                                             antid, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))    
    
    myinterlist.append(MG.Interaction({\
                      'id':2,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[0], \
                                             antiu, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2,0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

    myinterlist.append(MG.Interaction({\
                      'id':5,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[4], \
                                             antit, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    
    myinterlist.append(MG.Interaction({\
                      'id':3,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[2]] *3 \
                                             ),
                      'color': [color.ColorString([color.f(0, 1, 2)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    
    myinterlist.append(MG.Interaction({\
                      'id':4,\
                      'particles': MG.ParticleList([mypartlist[1], \
                                             antid, \
                                             mypartlist[3]]
                                             ),
                      'color': [color.ColorString([color.T(0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'ADD'},
                      'orders':{'QED':1}}))
    
    mymodel = MG.Model()
    mymodel.set('particles', mypartlist)
    mymodel.set('interactions', myinterlist)

    dict1 = {'legs' : myleglist1, 'orders':{'QCD':10, 'QED':0},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}

    dict2 = {'legs' : myleglist2, 'orders':{'QCD':10, 'QED':0},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}

    dict3 = {'legs' : myleglist3, 'orders':{'QCD':10, 'QED':0},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}
    
    myproc1 = MG.Process(dict1)
    myproc1.set('orders', {'QED':0})
    myproc2 = MG.Process(dict2)
    myproc2.set('orders', {'QED':0})
    myproc3 = MG.Process(dict3)
    myproc3.set('orders', {'QED':0})


    def test_fks_helas_multi_process(self):
        """tests the correct initialization of a FKSHelasMultiProcess, 
        given an FKSMultiProcess"""
        p = [1,2, 21]

        my_multi_leg = MG.MultiLeg({'ids': p, 'state': True});

        # Define the multiprocess
        my_multi_leglist = MG.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * 4])
        
        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)
        my_process_definition = MG.ProcessDefinition({'legs':my_multi_leglist,
                                                    'model':self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])

        amps = diagram_generation.MultiProcess(
                {'process_definitions':my_process_definitions}).get('amplitudes')
        my_multi_process = fks_born.FKSMultiProcessFromBorn(amps)
        my_helas_mp = fks_born_helas.FKSHelasMultiProcessFromBorn(my_multi_process, False)
        
        #there should be 6 independent born_matrix_elements 
        #for me in my_helas_mp.get('matrix_elements'):
        #    print "--"
        #    for proc in me.born_matrix_element.get('processes'):
        #        print proc.nice_string()
        self.assertEqual(len(my_helas_mp.get('matrix_elements')),6)
        
        
        
    
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
        
        real_proc = fks_born.FKSRealProcess(fks3.born_proc, fksleglist,\
                                            4,0,amplist, amp_id_list)
        helas_real_proc = fks_born_helas.FKSHelasRealProcess(real_proc, me_list, me_id_list)
        self.assertEqual(helas_real_proc.i_fks, 5)
        self.assertEqual(helas_real_proc.j_fks, 4)
        self.assertEqual(helas_real_proc.ij, 4)
        self.assertEqual(helas_real_proc.ijglu, 0)
##        self.assertEqual(len(helas_real_proc.permutation), 5)
##        self.assertEqual(helas_real_proc.permutation, [1, 2, 4, 3, 5 ])
##        self.assertEqual(helas_real_proc.permutation, real_proc.permutation)
        #self.assertEqual(len(me_list), 1)
        #self.assertEqual(len(me_id_list), 1)
        self.assertEqual(helas_real_proc.matrix_element, 
                        helas_objects.HelasMatrixElement(real_proc.amplitude))
        #self.assertEqual(me_list[0],
        #                helas_objects.HelasMatrixElement(real_proc.amplitude))
##        self.assertEqual(me_id_list[0], array.array('i',[1 ,-1, -2,2,21]))
        
        
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
        
        me_list=[]
        me_id_list=[]
        me_list3=[]
        me_id_list3=[]
        res_me_list=[]
        res_me_id_list=[]
        amp_list=[]
        amp_id_list=[]
        amp_list3=[]
        amp_id_list3=[]
        fks1.generate_reals(amp_list, amp_id_list)
        fks3.generate_reals(amp_list3, amp_id_list3)


        helas_born_proc = fks_born_helas.FKSHelasProcessFromBorn(
                                    fks1, me_list, me_id_list)
        helas_born_proc3 = fks_born_helas.FKSHelasProcessFromBorn(
                                    fks3, me_list3, me_id_list3)
        
        self.assertEqual(helas_born_proc.born_matrix_element,
                          helas_objects.HelasMatrixElement(
                                    fks1.born_amp)
                          )
        res_reals = []
        for real in fks1.real_amps:
            res_reals.append(fks_born_helas.FKSHelasRealProcess(
                                real,res_me_list, res_me_id_list))
        self.assertEqual(me_list, res_me_list)
        self.assertEqual(me_id_list, res_me_id_list)
        self.assertEqual(11, len(helas_born_proc.real_processes))
#        for a,b in zip(res_reals, helas_born_proc.real_processes):
#            self.assertEqual(a,b)
        #more (in)sanity checks
        self.assertNotEqual(helas_born_proc.born_matrix_element,
                            helas_born_proc3.born_matrix_element)
        for a,b in zip(helas_born_proc3.real_processes, 
                    helas_born_proc.real_processes):
            self.assertNotEqual(a,b)
        
    def test_fks_helas_process_from_born_add_process(self):
        """test the correct implementation of the add process function, which
        should add both the born and the reals.
        We consider the process ug>ug and dg>dg, which 11 reals each"""
        me_list=[]
        me_id_list=[]
        res_me_list=[]
        res_me_id_list=[]
        amp_list=[]
        amp_id_list=[]
        
        #ug> ug
        fks1 = fks_born.FKSProcessFromBorn(self.myproc1)
        #dg> dg
        fks2 = fks_born.FKSProcessFromBorn(self.myproc2)
        #uu~> dd~
        fks3 = fks_born.FKSProcessFromBorn(self.myproc3)
        
        fks1.generate_reals(amp_list, amp_id_list)
        fks2.generate_reals(amp_list, amp_id_list)
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
        self.assertEqual(len(reals_1),11)
        self.assertEqual(len(reals_2),11)
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
            
