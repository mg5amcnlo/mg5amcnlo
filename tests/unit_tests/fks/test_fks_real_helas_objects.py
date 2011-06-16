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

"""Testing modules for FKS_real_helas_objects module"""

import sys
import os
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import tests.unit_tests as unittest
import madgraph.fks.fks_common as fks_common
import madgraph.fks.fks_real as fks
import madgraph.fks.fks_real_helas_objects as fks_helas
import madgraph.core.base_objects as MG
import madgraph.core.helas_objects as helas_objects
import madgraph.core.color_algebra as color
import madgraph.core.color_amp as color_amp
import madgraph.core.diagram_generation as diagram_generation
import copy
import array

class testFKSRealHelasObjects(unittest.TestCase):
    """a class to test the module FKSRealHelasObjects"""
    
    myleglist = MG.LegList()
    # PROCESS: u u~ > d d~ g g 
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
    'id': 1,\
    'number': 3,\
    'state': True,\
  #  'from_group': True\
},\
{\
    'id': -1,\
    'number': 4,\
    'state': True,\
   # 'from_group': True\
},\
{\
    'id': 21,\
    'number': 5,\
    'state': True,\
   # 'from_group': True\
},\
{\
    'id': 21,\
    'number': 6,\
    'state': True,\
   # 'from_group': True\
}
]

    for i in mylegs:
        myleglist.append(MG.Leg(i))

    myleglist2 = MG.LegList()
    # PROCESS: u u~ > u u~ g  
    mylegs2 = [{ \
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
},\
{\
    'id': 21,\
    'number': 5,\
    'state': True,\
   # 'from_group': True\
}
]
    for i in mylegs2:
        myleglist2.append(MG.Leg(i))

    
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
    

    mymodel = MG.Model()
    mymodel.set('particles', mypartlist)
    mymodel.set('interactions', myinterlist)
 #   mymodel.set('couplings', {(0, 0):'GQQ'})
 #   mymodel.set('lorentz', ['L1'])

    #mymodel.set('couplings',{(0, 0):'GQQ'} )

    
    dict = {'legs' : fks_common.to_fks_legs(myleglist, mymodel), 'orders':{'QCD':10, 'QED':0},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}

    dict2 = {'legs' : fks_common.to_fks_legs(myleglist2, mymodel), 'orders':{'QCD':10, 'QED':0},
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
    
    fks1 = fks.FKSProcessFromReals(myproc)
    fks2 = fks.FKSProcessFromReals(myproc2)
    
    
    def test_fks_helas_process_from_reals(self):
        """tests the correct initialization of a FKSHelasProcessFromReals.
        We test the correct initalization of:
        --real emission me
        --fks_born_processes_list
        --fks.inc_string (to be the same as for the FKSProcessFromReals)
        we test the process u u~ > d d~ g g"""
        helasfks1 = fks_helas.FKSHelasProcessFromReals(self.fks1)
        self.assertEqual(helasfks1.real_matrix_element,
                        helas_objects.HelasMatrixElement(self.fks1.real_amp))
        self.assertEqual(len(helasfks1.born_processes), 6)
        self.assertEqual(helasfks1.fks_inc_string, self.fks1.get_fks_inc_string())
        
    
    def test_fks_helas_born_process_init(self):
        """tests the correct initialization of a FKSHelasBornProcess, from a 
        FKSBornProc.
        We test the correct initialization of:
        --i/j fks
        --ijglu
        --matrix element
        --color link
        --is_nbody_only
        --is_to_integrate
        """         
        model = self.mymodel
        born1 = fks.FKSBornProcess(self.myproc, 
                                        fks_common.to_fks_leg(MG.Leg({\
                                        'id': -1,\
                                        'number': 4,\
                                        'state': True,\
                                       # 'from_group': True\
                                    }), model),
                                        fks_common.to_fks_leg(MG.Leg({\
                                        'id': 1,\
                                        'number': 3,\
                                        'state': True,\
                                       # 'from_group': True\
                                    }), model),
                                        fks_common.to_fks_leg(MG.Leg({\
                                        'id': 21,\
                                        'number': 3,\
                                        'state': True,\
                                       # 'from_group': True\
                                    }), model) )
        
        helas_born1 = fks_helas.FKSHelasBornProcess(born1)
        me = helas_objects.HelasMatrixElement(born1.amplitude)
        
        self.assertEqual(helas_born1.i_fks, 4)
        self.assertEqual(helas_born1.j_fks, 3)
        self.assertEqual(helas_born1.ijglu, 3)
        self.assertEqual(helas_born1.matrix_element, me)
        self.assertEqual(helas_born1.color_links, [])
        self.assertEqual(helas_born1.is_nbody_only, False)
        self.assertEqual(helas_born1.is_to_integrate, True)
        
    