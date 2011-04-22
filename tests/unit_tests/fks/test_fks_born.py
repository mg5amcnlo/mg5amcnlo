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

"""Testing modules for FKS_process_born class"""

import sys
import os
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import tests.unit_tests as unittest
import madgraph.fks.fks_born as fks_born
import madgraph.core.base_objects as MG
import madgraph.core.color_algebra as color
import madgraph.core.diagram_generation as diagram_generation
import copy
import array




class TestFKSProcess(unittest.TestCase):
    """a class to test FKS Processes"""
    myleglist = MG.LegList()
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
        myleglist.append(MG.Leg(i))

    myleglist2 = MG.LegList()
    # PROCESS: d d~ > u u~
    mylegs2 = [{ \
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
    for i in mylegs2:
        myleglist2.append(MG.Leg(i))
        
        myleglist3 = MG.LegList()
    # PROCESS: d d~ > a a~
    mylegs3 = [{ \
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
    'id': 22,\
    'number': 3,\
    'state': True,\
  #  'from_group': True\
},\
{\
    'id': 22,\
    'number': 4,\
    'state': True,\
   # 'from_group': True\
}
]
    for i in mylegs3:
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
    myinterlist.append(MG.Interaction({\
                      'id':6,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[0], \
                                             antiu, \
                                             mypartlist[3]]),
                      'color': [color.ColorString([color.T(0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'AUU'},
                      'orders':{'QED':1}}))
    
    expected_qcd_inter = MG.InteractionList()
        
    expected_qcd_inter.append(MG.Interaction({\
                      'id':1,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[1], \
                                             antid, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    
    expected_qcd_inter.append(MG.Interaction({\
                      'id':2,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[0], \
                                             antiu, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2,0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    
    expected_qcd_inter.append(MG.Interaction({\
                      'id':3,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[2]] *3 \
                                             ),
                      'color': [color.ColorString([color.f(0, 1, 2)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    expected_qcd_inter.append(MG.Interaction({\
                      'id':5,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[4], \
                                             antit, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    
    expected_qcd_inter.sort()

    mymodel = MG.Model()
    mymodel.set('particles', mypartlist)
    mymodel.set('interactions', myinterlist)
 #   mymodel.set('couplings', {(0, 0):'GQQ'})
 #   mymodel.set('lorentz', ['L1'])

    #mymodel.set('couplings',{(0, 0):'GQQ'} )

    
    dict = {'legs' : myleglist, 'orders':{'QCD':10, 'QED':0},
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
    dict3 = {'legs' : myleglist3, 'orders':{'QCD':0, 'QED':2},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}
    
    
    myproc = MG.Process(dict)

    myproc2 = MG.Process(dict2)
    
    myprocaa= MG.Process(dict3)
    

    
    
    def test_FKSMultiProcess(self):
        """tests the correct initializiation of a FKSMultiProcess. In particular
        checks that the correct number of borns is found"""
        
        p = [1, 21]

        my_multi_leg = MG.MultiLeg({'ids': p, 'state': True});

        # Define the multiprocess
        my_multi_leglist = MG.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * 4])
        
        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)
        my_process_definition = MG.ProcessDefinition({'legs':my_multi_leglist,
                                                    'model':self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])

        my_multi_process = fks_born.FKSMultiProcess(\
            {'process_definitions':my_process_definitions})
        
        self.assertEqual(len(my_multi_process.get('born_processes')),4)
        #check the total numbers of reals 11 11 6 16
        totreals = 0
        for born in my_multi_process.get('born_processes'):
            for reals in born.reals:
                totreals += len(reals)
        self.assertEqual(totreals, 44)
    
    def test_find_splittings(self):
        """tests if the correct splittings are found by the find_splitting function"""
        bfks = fks_born.FKSProcessFromBorn(self.myproc)
        leg_list = []
        res_list = []
        
        #INITIAL STATE SPLITTINGS
        # u to u>g u or g>u~u
        leg_list.append( MG.Leg({'id' : 2, 
                                 'state' : False, 
                                 'number' : 5}))
        res_list.append(sorted([sorted(bfks.to_fks_legs(
                        [fks_born.FKSLeg({'id' : 2, 
                                 'state' : False,
                                 'fks' : 'j'}),
                        fks_born.FKSLeg({'id' : 21, 
                                 'state' : True,
                                 'fks' : 'i'})]) ),
                        sorted(bfks.to_fks_legs(
                        [fks_born.FKSLeg({'id' : -2, 
                                 'state' : True,
                                 'fks' : 'i'}),
                        fks_born.FKSLeg({'id' : 21, 
                                 'state' : False,
                                 'fks' : 'j'})]) ),
                                 ]))

        #FINAL STATE SPLITTINGS
        #u to ug
        leg_list.append( MG.Leg({'id' : 2, 
                                 'state' : True, 
                                 'number' : 5}))
        res_list.append([sorted(bfks.to_fks_legs(
                        [fks_born.FKSLeg({'id' : 2, 
                                 'state' : True,
                                 'fks' : 'j'}),
                        fks_born.FKSLeg({'id' : 21, 
                                 'state' : True,
                                 'fks' : 'i'})]) )])
        #d to dg
        leg_list.append( MG.Leg({'id' : 1, 
                                 'state' : True, 
                                 'number' : 5}))
        res_list.append([sorted(bfks.to_fks_legs(
                        [fks_born.FKSLeg({'id' : 1, 
                                 'state' : True,
                                 'fks' : 'j'}),
                        fks_born.FKSLeg({'id' : 21, 
                                 'state' : True,
                                 'fks' : 'i'})]) )])
        #t to tg
        leg_list.append( MG.Leg({'id' : 6, 
                                 'state' : True, 
                                 'number' : 5}))
        res_list.append([sorted(bfks.to_fks_legs(
                        [fks_born.FKSLeg({'id' : 6, 
                                 'state' : True,
                                 'fks' : 'j'}),
                        fks_born.FKSLeg({'id' : 21, 
                                 'state' : True,
                                 'fks' : 'i'})]) )])

        #u~ to ug
        leg_list.append( MG.Leg({'id' : -2, 
                                 'state' : True, 
                                 'number' : 5}))
        res_list.append([sorted(bfks.to_fks_legs(
                        [fks_born.FKSLeg({'id' : -2, 
                                 'state' : True,
                                 'fks' : 'j'}),
                        fks_born.FKSLeg({'id' : 21, 
                                 'state' : True,
                                 'fks' : 'i'})]) )])
        #d~ to dg
        leg_list.append( MG.Leg({'id' : -1, 
                                 'state' : True, 
                                 'number' : 5}))
        res_list.append([sorted(bfks.to_fks_legs(
                        [fks_born.FKSLeg({'id' : -1, 
                                 'state' : True,
                                 'fks' : 'j'}),
                        fks_born.FKSLeg({'id' : 21, 
                                 'state' : True,
                                 'fks' : 'i'})]) )])
        #t~ to tg
        leg_list.append( MG.Leg({'id' : -6, 
                                 'state' : True, 
                                 'number' : 5}))
        res_list.append([sorted(bfks.to_fks_legs(
                        [fks_born.FKSLeg({'id' : -6, 
                                 'state' : True,
                                 'fks' : 'j'}),
                        fks_born.FKSLeg({'id' : 21, 
                                 'state' : True,
                                 'fks' : 'i'})]) )])

        #g > uu~ or dd~
        leg_list.append( MG.Leg({'id' : 21, 
                                 'state' : True, 
                                 'number' : 5}))
        res_list.append(sorted([
                        sorted(bfks.to_fks_legs(
                        [fks_born.FKSLeg({'id' : 2, 
                                 'state' : True,
                                 'fks' : 'j'}),
                        fks_born.FKSLeg({'id' : -2, 
                                 'state' : True,
                                 'fks' : 'i'})])), 
                        sorted(bfks.to_fks_legs(
                        [fks_born.FKSLeg({'id' : 1, 
                                 'state' : True,
                                 'fks' : 'j'}),
                        fks_born.FKSLeg({'id' : -1, 
                                 'state' : True,
                                 'fks' : 'i'})])),
                        sorted(bfks.to_fks_legs(
                        [fks_born.FKSLeg({'id' : 21, 
                                 'state' : True,
                                 'fks' : 'i'}),
                        fks_born.FKSLeg({'id' : 21, 
                                 'state' : True,
                                 'fks' : 'j'})])) ]))

        for i in range(len(leg_list)):
            leg = leg_list[i]
            res = res_list[i]
            res.sort            
            self.assertEqual(sorted(res), sorted(bfks.find_splittings(leg)) )   
    
    def test_FKSRealProcess_init(self):
        """tests the correct initialization of the FKSRealProcess class. 
        In particular checks that
        --i /j fks
        --amplitude
        --leg_permutation <<REMOVED
        are set to the correct values"""
        amplist = []
        amp_id_list = []
        #u g > u g
        fksproc = fks_born.FKSProcessFromBorn(self.myproc)
        #take the first real for this process 2j 21 > 21i 2 21
        leglist = fksproc.reals[0][0]
        realproc = fks_born.FKSRealProcess(fksproc.born_proc, leglist, amplist, amp_id_list)

        self.assertEqual(realproc.i_fks, 3)
        self.assertEqual(realproc.j_fks, 1)
        #something should be appended to amplist and amp_id_list
        self.assertEqual(len(amplist),1 )
        self.assertEqual(len(amp_id_list),1 )
        
##        self.assertEqual(amp_id_list[0], array.array('i',[2,21,2,21,21]))
        sorted_legs = fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg( 
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ] )
        
        sorted_real_proc= copy.copy(self.myproc)
        sorted_real_proc['legs']=  sorted_legs
        sorted_real_proc.set('orders', {'QCD' :11, 'QED' :0 })
        amp = diagram_generation.Amplitude(sorted_real_proc)
        self.assertEqual(amplist[0],amp)
 ##       self.assertEqual(realproc.permutation, [1,2,4,5,3])

    
    def test_generate_reals(self):
        """tests the generate reals function, if all the needed lists
        -- amplitudes
        -- id_list
        -- real amps
        have the correct number of elements"""
        
        amplist = []
        amp_id_list = []
        #process u g > u g 
        fksproc = fks_born.FKSProcessFromBorn(self.myproc)
        fksproc.generate_reals(amplist, amp_id_list)
        
        #there should be 11 real processes for this born
        self.assertEqual(len(fksproc.real_amps), 11)
        #first check that amp list and amp id list have the same length
        self.assertEqual(len(amplist), len(amp_id_list))
##        #there should be 8 amps
##        self.assertEqual(len(amplist), 8)

        #there should be 11 amps
        self.assertEqual(len(amplist), 11)

        
    def test_legs_to_color_link_string(self):
        """tests for some pair of legs that the color string and the index 
        replacements from leg_to_color_link_funcion are correctly assigned"""
        legs1 =[]
        legs2 =[]
        replacs = []
        strings = []
        
        #two different final state legs, triplets
        legs1.append(fks_born.FKSLeg(
                    {'state' : True,
                     'number' : 4,
                     'id' : 2,
                     'color' : 3,
                     'massless' : True,
                     'from_group' : True,
                     'spin' : 2
                     }               
                     ))
        
        legs2.append(fks_born.FKSLeg(
                    {'state' : True,
                     'number' : 5,
                     'id' : 2,
                     'color' : 3,
                     'massless' : True,
                     'from_group' : True,
                     'spin' : 2
                     }               
                     ))
        replacs.append([[4, -3001],[5, -3002]])
        strings.append(color.ColorString([
                    color.T(-6000, 4, -3001),
                    color.T(-6000, 5, -3002)
                                        ]))
        
        #two different final state legs, triplet+ antitriplet
        legs1.append(fks_born.FKSLeg(
                    {'state' : True,
                     'number' : 4,
                     'id' : 2,
                     'color' : 3,
                     'massless' : True,
                     'from_group' : True,
                     'spin' : 2
                     }               
                     ))
        
        legs2.append(fks_born.FKSLeg(
                    {'state' : True,
                     'number' : 5,
                     'id' : -2,
                     'color' : -3,
                     'massless' : True,
                     'from_group' : True,
                     'spin' : 2
                     }               
                     ))
        replacs.append([[4, -3001],[5, -3002]])
        strings.append(color.ColorString([
                    color.T(-6000, 4, -3001),
                    color.T(-6000, -3002, 5)
                                        ]))
        
        #initial and final state legs, triplets
        legs1.append(fks_born.FKSLeg(
                    {'state' : False,
                     'number' : 4,
                     'id' : 2,
                     'color' : 3,
                     'massless' : True,
                     'from_group' : True,
                     'spin' : 2
                     }               
                     ))
        
        legs2.append(fks_born.FKSLeg(
                    {'state' : True,
                     'number' : 5,
                     'id' : 2,
                     'color' : 3,
                     'massless' : True,
                     'from_group' : True,
                     'spin' : 2
                     }               
                     ))
        replacs.append([[4, -3001],[5, -3002]])
        strings.append(color.ColorString([
                    color.T(-6000, -3001, 4),
                    color.T(-6000, 5, -3002)
                                        ]))
        
        #same leg,  triplet
        legs1.append(fks_born.FKSLeg(
                    {'state' : True,
                     'number' : 4,
                     'id' : 6,
                     'color' : 3,
                     'massless' : True,
                     'from_group' : True,
                     'spin' : 2
                     }               
                     ))
        
        legs2.append(fks_born.FKSLeg(
                    {'state' : True,
                     'number' : 4,
                     'id' : 6,
                     'color' : 3,
                     'massless' : True,
                     'from_group' : True,
                     'spin' : 2
                     }               
                     ))
        replacs.append([[4, -3001]])
        strings.append(color.ColorString([
                    color.T(-6000, 4,-3002),
                    color.T(-6000, -3002, -3001)
                                        ]))
        
        #same leg,  antitriplet
        legs1.append(fks_born.FKSLeg(
                    {'state' : True,
                     'number' : 4,
                     'id' : -6,
                     'color' : -3,
                     'massless' : True,
                     'from_group' : True,
                     'spin' : 2
                     }               
                     ))
        
        legs2.append(fks_born.FKSLeg(
                    {'state' : True,
                     'number' : 4,
                     'id' : -6,
                     'color' : -3,
                     'massless' : True,
                     'from_group' : True,
                     'spin' : 2
                     }               
                     ))
        replacs.append([[4, -3001]])
        strings.append(color.ColorString([
                    color.T(-6000, -3002, 4),
                    color.T(-6000, -3001, -3002)
                                        ]))
        
        for i in range(len(legs1)):
            leg1 = legs1[i]
            leg2 = legs2[i]
            replac = replacs[i]
            string = strings[i]
            self.assertEqual(fks_born.legs_to_color_link_string(leg1, leg2),
                             {'string' : string, 'replacements': replac})
        
        
    def test_find_color_links(self): 
        """tests if all the correct color links are found for a given born process"""
        myleglist = MG.LegList()
        # PROCESS: u u~ > g t t~ a 
        mylegs = [{ \
        'id': 2,\
        'number': 1,\
        'state': False,\
     #   'from_group': True \
    }, \
    { \
        'id': -2,\
        'number': 2,\
        'state': False,\
        #'from_group': True\
    },\
    {\
        'id': 21,\
        'number': 3,\
        'state': True,\
      #  'from_group': True\
    },\
    {\
        'id': 6,\
        'number': 4,\
        'state': True,\
       # 'from_group': True\
    },\
    {\
        'id': -6,\
        'number': 5,\
        'state': True,\
       # 'from_group': True\
    },\
    {\
        'id': 22,\
        'number': 6,\
        'state': True,\
       # 'from_group': True\
    }
    ]

        for i in mylegs:
            myleglist.append(MG.Leg(i))   
        proc = fks_born.FKSProcessFromBorn(MG.Process({
                       'legs' : myleglist, 
                       'orders':{'QCD':10, 'QED':10},
                       'model': self.mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}
                       }))
        fkslegs = proc.to_fks_legs(myleglist)
        links = []
        links.append([fkslegs[0], fkslegs[1]])
        links.append([fkslegs[0], fkslegs[2]])
        links.append([fkslegs[0], fkslegs[3]])
        links.append([fkslegs[0], fkslegs[4]])
        
        links.append([fkslegs[1], fkslegs[0]])
        links.append([fkslegs[1], fkslegs[2]])
        links.append([fkslegs[1], fkslegs[3]])
        links.append([fkslegs[1], fkslegs[4]])
        
        links.append([fkslegs[2], fkslegs[0]])
        links.append([fkslegs[2], fkslegs[1]])
        links.append([fkslegs[2], fkslegs[3]])
        links.append([fkslegs[2], fkslegs[4]])
        
        links.append([fkslegs[3], fkslegs[0]])
        links.append([fkslegs[3], fkslegs[1]])
        links.append([fkslegs[3], fkslegs[2]])
        links.append([fkslegs[3], fkslegs[3]])
        links.append([fkslegs[3], fkslegs[4]])
        
        links.append([fkslegs[4], fkslegs[0]])
        links.append([fkslegs[4], fkslegs[1]])
        links.append([fkslegs[4], fkslegs[2]])
        links.append([fkslegs[4], fkslegs[3]])
        links.append([fkslegs[4], fkslegs[4]])
        
        self.assertEqual(len(links), len(proc.color_links))
        for i in range(len(links)):
            self.assertEqual(links[i], proc.color_links[i]['legs'])

        
        
        
    def test_find_reals(self):
        """tests if all the real processes are found for a given born"""
        #process is u g > u g
        fksproc = fks_born.FKSProcessFromBorn(self.myproc)
        target = []
        
        #leg 1 can split as u g > u g g or  g g > u u~ g 
        target.append( [fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ] ),
                    fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :21,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                         fks_born.FKSLeg(
                                         {'id' :-2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ])]
                                        )
        
        #leg 2 can split as u d > d u g OR u d~ > d~ u g OR
        #                   u u > u u g OR u u~ > u u~ g OR
        #                   u g > u g g 
        
        target.append([fksproc.to_fks_legs([#ug>ugg
                                        fks_born.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ]),
                       fksproc.to_fks_legs([#ud>dug
                                        fks_born.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' :1,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ] ),
                    fksproc.to_fks_legs([#ud~>d~ug
                                        fks_born.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' :-1,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ]),
                    fksproc.to_fks_legs([#uu>uug
                                        fks_born.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ] ),
                    fksproc.to_fks_legs([#uu~>uu~g
                                        fks_born.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :-2,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' :-2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ])
                    ])
        
        #leg 3 can split as u g > u g g
        target.append( [fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ] )
                                        ]
                                        )
        #leg 4 can split as u g > u g g or u g > u u u~ or u g > u d d~ 
        target.append( [fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ] ),
                        fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg( 
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :1,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' :-1,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ] ),
                        fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg( 
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ] )
                                        
                                        ]
                                        )

                       
        for i in range(len(fksproc.reals)):
            for j in range(len(fksproc.reals[i])):
                self.assertEqual(fksproc.reals[i][j], target[i][j]) 
        
        #process is d d~ > u u~
        fksproc2 = fks_born.FKSProcessFromBorn(self.myproc2)
        target2 = []
        #leg 1 can split as d d~ > u u~ g or  g d~ > d~ u u~ 
        target2.append( [fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ] ),
                    fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :21,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :-1,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg( 
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ])]
                                        )
        
        #leg 2 can split as d d~ > u u~ g or  d g > d u u~ 
        target2.append( [fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ] ),
                    fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' : 1,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg( 
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ])]
                                        )
        
        #leg 3 can split as d d~ > u u~ g  
        target2.append( [fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ] )])

        #leg 4 can split as d d~ > u u~ g  
        target2.append( [fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ] )])




        for i in range(len(fksproc2.reals)):
            self.assertEqual(fksproc2.reals[i], target2[i]) 
            
    
        #d d~ > a a
        fksproc3 = fks_born.FKSProcessFromBorn(self.myprocaa)
        target3 = []
        #leg 1 can split as d d~ > g a a or  g d~ > d~ a a 
        target3.append( [fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' : 22,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ] ),
                    fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :21,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :-1,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg( 
                                         {'id' : 22,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ])]
                                        )
        #leg 2 can split as d d~ > g a a  or  d g > d a a 
        target3.append( [fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg( 
                                         {'id' :21,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' :22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' : 22,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ] ),
                    fksproc.to_fks_legs([
                                        fks_born.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_born.FKSLeg(
                                         {'id' : 1,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_born.FKSLeg(
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_born.FKSLeg( 
                                         {'id' : 22,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ])]
                                        )        


        for real, res in zip(fksproc3.reals, target3):
            self.assertEqual(real, res) 
  
        

    
    def test_to_fks_leg_s(self):
        """tests if color and massless and spin entries of a fks leg are correctly set"""
        bfks = fks_born.FKSProcessFromBorn(self.myproc)
        leg_list = []
        res_list = []
        leg_list.append( MG.Leg({'id' : 21, 
                                 'state' : True, 
                                 'number' : 5}))
        res_list.append( fks_born.FKSLeg({'id' : 21, 
                                 'state' : True, 
                                 'number' : 5,
                                 'massless' : True,
                                 'color' : 8,
                                 'spin' : 3})) 
        leg_list.append( MG.Leg({'id' : 6, 
                                 'state' : True, 
                                 'number' : 5}))
        res_list.append( fks_born.FKSLeg({'id' : 6, 
                                 'state' : True, 
                                 'number' : 5,
                                 'massless' : False,
                                 'color' : 3,
                                 'spin' : 2}))  

        self.assertEqual(bfks.to_fks_legs(leg_list), res_list)
        
        for leg in leg_list:
            res = res_list[leg_list.index(leg)]
            self.assertEqual(bfks.to_fks_leg(leg), res)

        
    
    def test_split_leg(self):
        """tests the correct splitting of a leg into two partons"""
        leg_list = []
        parts_list = []
        res_list = []
        bfks = fks_born.FKSProcessFromBorn(self.myproc)
        leg_list.append( fks_born.FKSLeg({'id' : 21, 
                                 'state' : True, 
                                 'number' : 5}))
        parts_list.append([MG.Particle({'name':'u',
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
                  'is_part': True,
                  'self_antipart':False}),
                  MG.Particle({'name':'d',
                  'antiname':'d~',
                  'spin':2,
                  'color':3,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'u',
                  'antitexname':'\\overline{u}',
                  'line':'straight',
                  'charge':-1. / 3.,
                  'pdg_code':1,
                  'propagating':True,
                  'is_part': False,
                  'self_antipart':False})
                           ])
        res_list.append([[fks_born.FKSLeg({'id' : 2, 
                                 'state' : True,
                                 'color' : 3,
                                 'spin' : 2,
                                 'massless' : True,
                                 'fks' : 'j'}),
                        fks_born.FKSLeg({'id' : -1, 
                                 'state' : True,
                                 'color' : -3,
                                 'spin' : 2,
                                 'massless' : True,
                                 'fks' : 'i'})]]
                        )
        
        leg_list.append( fks_born.FKSLeg({'id' : 21, 
                                 'state' : False, 
                                 'number' : 5}))
        parts_list.append([MG.Particle({'name':'u',
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
                  'is_part': True,
                  'self_antipart':False}),
                  MG.Particle({'name':'d',
                  'antiname':'d~',
                  'spin':2,
                  'color':3,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'u',
                  'antitexname':'\\overline{u}',
                  'line':'straight',
                  'charge':-1. / 3.,
                  'pdg_code':1,
                  'propagating':True,
                  'is_part': False,
                  'self_antipart':False})
                           ])
        res_list.append([
                        [fks_born.FKSLeg({'id' : 2, 
                                 'state' : False,
                                 'color' : 3,
                                 'spin' : 2,
                                 'massless' : True,
                                 'fks' : 'j'}),
                        fks_born.FKSLeg({'id' : 1, 
                                 'state' : True,
                                 'color' : 3,
                                 'spin' : 2,
                                 'massless' : True,
                                 'fks' : 'i'})],
                        [fks_born.FKSLeg({'id' : -1, 
                                 'state' : False,
                                 'color' : -3,
                                 'spin' : 2,
                                 'massless' : True,
                                 'fks' : 'j'}),
                        fks_born.FKSLeg({'id' : -2, 
                                 'state' : True,
                                 'color' : -3,
                                 'spin' : 2,
                                 'massless' : True,
                                 'fks' : 'i'})]
                        ]
                        )
        
        leg_list.append( fks_born.FKSLeg({'id' : 21, 
                                 'state' : False, 
                                 'number' : 5}))
        parts_list.append([MG.Particle({'name':'g',
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
                      'self_antipart':True}),
                      MG.Particle({'name':'g',
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
                      'self_antipart':True})
                      ])
        res_list.append([
                        [fks_born.FKSLeg({'id' : 21, 
                                 'state' : False,
                                 'color' : 8,
                                 'spin' : 3,
                                 'massless' : True,
                                 'fks' : 'j'}),
                        fks_born.FKSLeg({'id' : 21, 
                                 'state' : True,
                                 'color' : 8,
                                 'spin' : 3,
                                 'massless' : True,
                                 'fks' : 'i'})]])
        
        for i in range(len(leg_list)):
            leg = leg_list[i]
            parts = parts_list[i]
            res = res_list[i]
            self.assertEqual(sorted(res), bfks.split_leg(leg,parts) ) 
        
    
    def test_find_qcd_interactions(self):
        """tests the correct implementation of the find_qcd_interactions function"""
        bfks = fks_born.FKSProcessFromBorn(self.myproc)
        self.assertEqual(self.expected_qcd_inter, bfks.find_qcd_interactions(self.mymodel))


        
    
    
    def test_find_brem_particles(self):
        """tests the correct implementation of the find_soft_particles function"""
        bfks = fks_born.FKSProcessFromBorn(self.myproc)
        expected_res = {1 :MG.Particle({'name':'d',
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
                  'is_part': True,
                  'propagating':True,
                  'self_antipart':False}),
                  -1 :MG.Particle({'name':'d',
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
                  'self_antipart':False}),
                  2 :MG.Particle({'name':'u',
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
                  'is_part':True,
                  'self_antipart':False}),
                  -2 :MG.Particle({'name':'u',
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
                  'self_antipart':False}),
                  21 :MG.Particle({'name':'g',
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
                      'self_antipart':True})
            }
        self.assertEqual(expected_res, bfks.find_brem_particles(self.mymodel))
    
    
    def test_find_qcd_particles(self):
        """tests the correct implementation of the find_qcd_particles function"""
        bfks = fks_born.FKSProcessFromBorn(self.myproc)
        expected_res = {1 :MG.Particle({'name':'d',
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
                  'is_part': True,
                  'propagating':True,
                  'self_antipart':False}),
                  -1 :MG.Particle({'name':'d',
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
                  'self_antipart':False}),
                  2 :MG.Particle({'name':'u',
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
                  'is_part':True,
                  'self_antipart':False}),
                  -2 :MG.Particle({'name':'u',
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
                  'self_antipart':False}),
                  6 :MG.Particle({'name':'t',
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
                  'is_part': True,
                  'self_antipart':False}),
                  -6 :MG.Particle({'name':'t',
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
                  'self_antipart':False}),
                  21 :MG.Particle({'name':'g',
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
                      'self_antipart':True})
            }
        self.assertEqual(expected_res, bfks.find_qcd_particles(self.mymodel))
    

    