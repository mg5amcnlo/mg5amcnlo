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

"""Testing modules for FKS_process class"""

import sys
import os
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import tests.unit_tests as unittest
import madgraph.fks.fks_base as fks_base
import madgraph.fks.fks_common as fks_common
import madgraph.core.base_objects as MG
import madgraph.core.color_algebra as color
import madgraph.core.diagram_generation as diagram_generation
import models.import_ufo as import_ufo
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
}, \
{ \
    'id': 21,\
    'number': 2,\
    'state': False,\
},\
{\
    'id': 2,\
    'number': 3,\
    'state': True,\
},\
{\
    'id': 21,\
    'number': 4,\
    'state': True,\
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
}, \
{ \
    'id': -1,\
    'number': 2,\
    'state': False,\
},\
{\
    'id': 2,\
    'number': 3,\
    'state': True,\
},\
{\
    'id': -2,\
    'number': 4,\
    'state': True,\
}
]
    for i in mylegs2:
        myleglist2.append(MG.Leg(i))
        
        myleglist3 = MG.LegList()
    # PROCESS: d d~ > a a
    mylegs3 = [{ \
    'id': 1,\
    'number': 1,\
    'state': False,\
}, \
{ \
    'id': -1,\
    'number': 2,\
    'state': False,\
},\
{\
    'id': 22,\
    'number': 3,\
    'state': True,\
},\
{\
    'id': 22,\
    'number': 4,\
    'state': True,\
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
                  'self_antipart':False}))
    mypartlist.append(MG.Particle({'name':'e-',
                    'antiname':'e+',
                    'spin':2,
                    'color':1,
                    'mass':'zero',
                    'width':'zero',
                    'texname':'e-',
                    'antitexname':'e+',
                    'line':'straight',
                    'charge':-1.,
                    'pdg_code':11,
                    'is_part':True,
                    'propagating':True,
                    'self_antipart':False}))
            
    mypartlist.append(MG.Particle({'name':'ve',
                    'antiname':'ve~',
                    'spin':2,
                    'color':1,
                    'mass':'zero',
                    'width':'zero',
                    'texname':'ve',
                    'antitexname':'ve~',
                    'line':'straight',
                    'charge':0.,
                    'pdg_code':12,
                    'is_part':True,
                    'propagating':True,
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
    
    ep = MG.Particle({'name':'e-',
                    'antiname':'e+',
                    'spin':2,
                    'color': 1,
                    'mass':'zero',
                    'width':'zero',
                    'texname':'e-',
                    'antitexname':'e+',
                    'line':'straight',
                    'charge':  -1.,
                    'pdg_code': 11,
                    'propagating':True,
                    'is_part':False,
                    'self_antipart':False})
            
    antive = MG.Particle({'name':'ve',
                    'antiname':'ve~',
                    'spin':2,
                    'color': 1,
                    'mass':'zero',
                    'width':'zero',
                    'texname':'ve',
                    'antitexname':'ve~',
                    'line':'straight',
                    'charge':  0.,
                    'pdg_code': 12,
                    'propagating':True,
                    'is_part':False,
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
    
    myinterlist.append(MG.Interaction({\
                              'id':7,\
                              'particles': MG.ParticleList(\
                                                    [mypartlist[5], \
                                                     ep, \
                                                     mypartlist[3]]),
                              'color': [color.ColorString([])],
                              'lorentz':['L1'],
                              'couplings':{(0, 0):'Aee'},
                              'orders':{'QED':1}}))
            
    myinterlist.append(MG.Interaction({\
                              'id':8,\
                              'particles': MG.ParticleList(\
                                                    [mypartlist[4], \
                                                     antit, \
                                                     mypartlist[3]]),
                              'color': [color.ColorString([color.T(0,1)])],
                              'lorentz':['L1'],
                              'couplings':{(0, 0):'ATT'},
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

    expected_qed_inter = MG.InteractionList()
            
    expected_qed_inter.append(MG.Interaction({\
                              'id':4,\
                              'particles': MG.ParticleList([mypartlist[1], \
                                                     antid, \
                                                     mypartlist[3]]
                                                     ),
                              'color': [color.ColorString([color.T(0,1)])],
                              'lorentz':['L1'],
                              'couplings':{(0, 0):'ADD'},
                              'orders':{'QED':1}}))

    expected_qed_inter.append(MG.Interaction({\
                              'id':6,\
                              'particles': MG.ParticleList(\
                                                    [mypartlist[0], \
                                                     antiu, \
                                                     mypartlist[3]]),
                              'color': [color.ColorString([color.T(0,1)])],
                              'lorentz':['L1'],
                              'couplings':{(0, 0):'AUU'},
                              'orders':{'QED':1}}))

    expected_qed_inter.append(MG.Interaction({\
                              'id':7,\
                              'particles': MG.ParticleList(\
                                                    [mypartlist[5], \
                                                     ep, \
                                                     mypartlist[3]]),
                              'color': [color.ColorString([])],
                              'lorentz':['L1'],
                              'couplings':{(0, 0):'Aee'},
                              'orders':{'QED':1}}))
            
    expected_qed_inter.append(MG.Interaction({\
                              'id':8,\
                              'particles': MG.ParticleList(\
                                                    [mypartlist[4], \
                                                     antit, \
                                                     mypartlist[3]]),
                              'color': [color.ColorString([color.T(0,1)])],
                              'lorentz':['L1'],
                              'couplings':{(0, 0):'ATT'},
                              'orders':{'QED':1}}))
            
    expected_qed_inter.sort()
        
    mymodel = MG.Model()
    mymodel.set('particles', mypartlist)
    mymodel.set('interactions', myinterlist)
    
    dict_qcd = {'legs' : myleglist, 'orders':{'QCD':2, 'QED':0, 'WEIGHTED':2},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'perturbation_couplings':['QCD'],
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}
    
    dict_qed = {'legs' : myleglist, 'orders':{'QCD':2, 'QED':0, 'WEIGHTED':2},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'perturbation_couplings':['QED'],
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}

    dict2_qcd = {'legs' : myleglist2, 'orders':{'QCD':2, 'QED':0, 'WEIGHTED':2},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'perturbation_couplings':['QCD'],
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}
    
    dict2_qed = {'legs' : myleglist2, 'orders':{'QCD':2, 'QED':0, 'WEIGHTED':2},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'perturbation_couplings':['QED'],
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}
    
    dict3_qcd = {'legs' : myleglist3, 'orders':{'QCD':0, 'QED':2, 'WEIGHTED':4},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'perturbation_couplings':['QCD'],
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}
    
    dict3_qed = {'legs' : myleglist3, 'orders':{'QCD':0, 'QED':2, 'WEIGHTED':4},
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
        """tests the correct initializiation of a FKSMultiProcess. In particular
        checks that the correct number of borns is found"""
        
        p = [1, 21]

        my_multi_leg = MG.MultiLeg({'ids': p, 'state': True});

        # Define the multiprocess
        my_multi_leglist = MG.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * 4])
        
        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)
        my_process_definition = MG.ProcessDefinition({\
                        'legs': my_multi_leglist,
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])
        my_process_definition_qed = MG.ProcessDefinition({\
                        'legs': my_multi_leglist,
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
        self.assertEqual(my_multi_process.get('has_isr'),True)
        self.assertEqual(my_multi_process.get('has_fsr'),True)
        self.assertEqual(my_multi_process_qed.get('has_isr'),True)
        self.assertEqual(my_multi_process_qed.get('has_fsr'),True)
        #check the total numbers of reals 11 11 6 16
        totreals = 0
        for born in my_multi_process.get('born_processes'):
            for reals in born.reals:
                totreals += len(reals)
        self.assertEqual(totreals, 44)
        # check the total numbers of reals 6 3 3 0
        totreals = 0
        for born in my_multi_process_qed.get('born_processes'):
            for reals in born.reals:
                totreals += len(reals)
        self.assertEqual(totreals, 12)


    def test_FKSMultiProcess_no_fsr(self):
        """tests the correct initializiation of a FKSMultiProcess. In particular
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
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])
        my_multi_leglist_qed[0].set('state', False)
        my_multi_leglist_qed[1].set('state', False)
        my_process_definition_qed = MG.ProcessDefinition({\
                        'legs': my_multi_leglist_qed,
                        'perturbation_couplings': ['QED'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions_qed = MG.ProcessDefinitionList(\
            [my_process_definition_qed])

        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions':my_process_definitions})
        self.assertEqual(my_multi_process.get('has_isr'),True)
        self.assertEqual(my_multi_process.get('has_fsr'),False)
        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions':my_process_definitions_qed})
        self.assertEqual(my_multi_process.get('has_isr'),True)
        self.assertEqual(my_multi_process.get('has_fsr'),False)
        


    def test_FKSMultiProcess_no_isr(self):
        """tests the correct initializiation of a FKSMultiProcess. In particular
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
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definition_qed = MG.ProcessDefinition({\
                        'legs': my_multi_leglist_qed,
                        'perturbation_couplings': ['QED'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])
        my_process_definitions_qed = MG.ProcessDefinitionList(\
            [my_process_definition_qed])

        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions':my_process_definitions})
        self.assertEqual(my_multi_process.get('has_isr'),False)
        self.assertEqual(my_multi_process.get('has_fsr'),True)
        my_multi_process = fks_base.FKSMultiProcess(\
                {'process_definitions':my_process_definitions_qed})
        self.assertEqual(my_multi_process.get('has_isr'),False)
        self.assertEqual(my_multi_process.get('has_fsr'),True)


    def test_FKSMultiProcess_add(self):
        """tests the correct initializiation of a FKSMultiProcess and the add funciton. In particular
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
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])
        my_process_definition_qed = MG.ProcessDefinition({\
                        'legs': my_multi_leglist,
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
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions1 = MG.ProcessDefinitionList(\
            [my_process_definition1])
        my_process_definition1_qed = MG.ProcessDefinition({\
                        'legs': my_multi_leglist1,
                        'perturbation_couplings': ['QED'],
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
        self.assertEqual(my_multi_process.get('has_isr'),True)
        self.assertEqual(my_multi_process.get('has_fsr'),True)
        self.assertEqual(nborn_qed + nborn1_qed, len(my_multi_process_qed['born_processes']))
        self.assertEqual(my_multi_process_qed.get('has_isr'),True)
        self.assertEqual(my_multi_process_qed.get('has_fsr'),True)


    def test_FKSProcess_gggg(self):
        """tests that for g g > g g all the relevant splittings are there"""
        glu = MG.Leg({'id': 21, 'state':True})
        leglist = MG.LegList([MG.Leg({'id': 21, 'state':False}),
                              MG.Leg({'id': 21, 'state':False}),
                              MG.Leg({'id': 21, 'state':True}),
                              MG.Leg({'id': 21, 'state':True})])
        

        dict = {'legs' : leglist, 'orders':{'QCD':2, 'QED':0, 'WEIGHTED':2},
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

        target_fks_infos = [ \
                # real config 1: g g > g g g
                [{'i':5, 'j':1, 'ij':1, 'ij_glu':1, 'need_color_links':True,
                    'rb_links':[{'born_conf': 0, 'real_conf': 11},
                                {'born_conf': 1, 'real_conf': 10},
                                {'born_conf': 2, 'real_conf': 9}]},
                 {'i':5, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':True,
                     'rb_links':[{'born_conf': 0, 'real_conf': 14},
                                 {'born_conf': 1, 'real_conf': 4},
                                 {'born_conf': 2, 'real_conf': 7}]},
                 {'i':5, 'j':4, 'ij':4, 'ij_glu':4, 'need_color_links':True,
                     'rb_links':[{'born_conf': 0, 'real_conf': 2},
                                 {'born_conf': 1, 'real_conf': 5},
                                 {'born_conf': 2, 'real_conf': 12}]}],
                # real config 2: u g > u g g
                [{'i':3, 'j':1, 'ij':1, 'ij_glu':1, 'need_color_links':False,
                    'rb_links':[{'born_conf': 0, 'real_conf': 5},
                                {'born_conf': 1, 'real_conf': 4},
                                {'born_conf': 2, 'real_conf': 3}]}],
                # real config 3: ux g > ux g g
                [{'i':3, 'j':1, 'ij':1, 'ij_glu':1, 'need_color_links':False,
                    'rb_links':[{'born_conf': 0, 'real_conf': 5},
                                {'born_conf': 1, 'real_conf': 4},
                                {'born_conf': 2, 'real_conf': 3}]}],
                # real config 4: d g > d g g
                [{'i':3, 'j':1, 'ij':1, 'ij_glu':1, 'need_color_links':False,
                    'rb_links':[{'born_conf': 0, 'real_conf': 5},
                                {'born_conf': 1, 'real_conf': 4},
                                {'born_conf': 2, 'real_conf': 3}]}],
                # real config 5: dx g > dx g g
                [{'i':3, 'j':1, 'ij':1, 'ij_glu':1, 'need_color_links':False,
                    'rb_links':[{'born_conf': 0, 'real_conf': 5},
                                {'born_conf': 1, 'real_conf': 4},
                                {'born_conf': 2, 'real_conf': 3}]}],
                # real config 6: g u > u g g
                [{'i':3, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':False,
                    'rb_links':[{'born_conf': 0, 'real_conf': 12},
                                {'born_conf': 1, 'real_conf': 6},
                                {'born_conf': 2, 'real_conf': 9}]}],
                # real config 7: g ux > ux g g
                [{'i':3, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':False,
                    'rb_links':[{'born_conf': 0, 'real_conf': 12},
                                {'born_conf': 1, 'real_conf': 6},
                                {'born_conf': 2, 'real_conf': 9}]}],
                # real config 8: g d > d g g
                [{'i':3, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':False,
                    'rb_links':[{'born_conf': 0, 'real_conf': 12},
                                {'born_conf': 1, 'real_conf': 6},
                                {'born_conf': 2, 'real_conf': 9}]}],
                # real config 9: g dx > dx g g
                [{'i':3, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':False,
                    'rb_links':[{'born_conf': 0, 'real_conf': 12},
                                {'born_conf': 1, 'real_conf': 6},
                                {'born_conf': 2, 'real_conf': 9}]}],
                # real config 10: g g > u ux g
                [{'i':4, 'j':3, 'ij':3, 'ij_glu':3, 'need_color_links':False,
                    'rb_links':[{'born_conf': 0, 'real_conf': 0},
                                {'born_conf': 1, 'real_conf': 14},
                                {'born_conf': 2, 'real_conf': 11}]}],
                # real config 11: g g > d dx g
                [{'i':4, 'j':3, 'ij':3, 'ij_glu':3, 'need_color_links':False,
                    'rb_links':[{'born_conf': 0, 'real_conf': 0},
                                {'born_conf': 1, 'real_conf': 14},
                                {'born_conf': 2, 'real_conf': 11}]}]]

        myfks.generate_reals([],[])
        self.assertEqual(len(myfks.real_amps),11)
        for real, fks_info in zip(myfks.real_amps, target_fks_infos):
            self.assertEqual(real.fks_infos, fks_info)

    def test_FKSProcess_aguux_qed(self):
        """tests that for a g > u u~ all the relevant QED splittings are there"""
        glu = MG.Leg({'id': 22, 'state':True})
        leglist = MG.LegList([MG.Leg({'id': 22, 'state':False}),
                              MG.Leg({'id': 21, 'state':False}),
                              MG.Leg({'id': 2, 'state':True}),
                              MG.Leg({'id': -2, 'state':True})])
        

        dict = {'legs' : leglist, 'orders':{'QCD':1, 'QED':1},
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
        target_fks_infos = [ \
                # real config 1: e- g > u u~ e-
                [{'i':5, 'j':1, 'ij':1, 'ij_glu':1, 'need_color_links':False,
                    'rb_links': [{'real_conf': 1, 'born_conf': 0}, {'real_conf': 0, 'born_conf': 1}]}],
                # real config 2: e+ g > u u~ e+
                [{'i':5, 'j':1, 'ij':1, 'ij_glu':1, 'need_color_links':False,
                    'rb_links': [{'real_conf': 1, 'born_conf': 0}, {'real_conf': 0, 'born_conf': 1}]}],
                # real config 3: d g > u u~ d
                [{'i':5, 'j':1, 'ij':1, 'ij_glu':1, 'need_color_links':False,
                    'rb_links': [{'real_conf': 2, 'born_conf': 0}, {'real_conf': 1, 'born_conf': 1}]}],
                # real config 4: d~ g > u u~ d~
                [{'i':5, 'j':1, 'ij':1, 'ij_glu':1, 'need_color_links':False,
                    'rb_links': [{'real_conf': 2, 'born_conf': 0}, {'real_conf': 1, 'born_conf': 1}]}],
                # real config 5: u g > u u u~
                [{'i':4, 'j':1, 'ij':1, 'ij_glu':1, 'need_color_links':False,
                    'rb_links': [{'real_conf': 5, 'born_conf': 0}, {'real_conf': 4, 'born_conf': 1}]}],
                # real config 6: u~ g > u u~ u~
                [{'i':5, 'j':1, 'ij':1, 'ij_glu':1, 'need_color_links':False,
                    'rb_links': [{'real_conf': 5, 'born_conf': 0}, {'real_conf': 4, 'born_conf': 1}]}],
                # real config 7: a g > u u~ a
                [{'i':5, 'j':3, 'ij':3, 'ij_glu':0, 'need_color_links':True,
                    'rb_links': [{'real_conf': 5, 'born_conf': 0}, {'real_conf': 3, 'born_conf': 1}]},
                 {'i':5, 'j':4, 'ij':4, 'ij_glu':0, 'need_color_links':True,
                    'rb_links': [{'real_conf': 1, 'born_conf': 0}, {'real_conf': 4, 'born_conf': 1}]}]]

        myfks.generate_reals([],[])
        self.assertEqual(len(myfks.real_amps),7)
        for real, fks_info in zip(myfks.real_amps, target_fks_infos):
            self.assertEqual(real.fks_infos, fks_info)


    def test_FKSRealProcess_init(self):
        """tests the correct initialization of the FKSRealProcess class. 
        In particular checks that
        --fks_info
        --amplitude (also the generate_real_amplitude function is tested)
        --leg_permutation <<REMOVED
        are set to the correct values"""
        #u g > u g
        fksproc = fks_base.FKSProcess(self.myproc)
        #u g > g u
        fksproc_qed = fks_base.FKSProcess(self.myproc_qed)
        #take the first real for this process 2j 21 >2 21 21i
        leglist = fksproc.reals[0][0]
        realproc = fks_base.FKSRealProcess(fksproc.born_proc, leglist, 1,0,\
                                           perturbed_orders = ['QCD'])
        # 2j 21 > 21 2 22i
        leglist_qed = fksproc_qed.reals[0][0]
        realproc_qed = fks_base.FKSRealProcess(fksproc_qed.born_proc,leglist_qed,1,0,\
                                               perturbed_orders = ['QED'])
        self.assertEqual(realproc.fks_infos, [{'i' : 5,
                                               'j' : 1,
                                               'ij' : 1,
                                               'ij_glu' : 0,
                                               'need_color_links': True}])
        self.assertEqual(realproc_qed.fks_infos,[{'i':5,
                                                  'j':1,
                                                  'ij':1,
                                                  'ij_glu':0,
                                                  'need_color_links':True}])

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
                                         {'id' :21,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :2,
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
## an emplty amplitude is generted so far...
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
        self.assertEqual([0.,0.,0.,0.,0.],realproc.charges) # charge is irrelevant in QCD
        self.assertEqual(amp_qed,realproc_qed.amplitude)
        self.assertEqual(array.array('i',[2,21,21,2,22]), realproc_qed.pdgs)
        self.assertEqual([3,8,8,3,1],realproc_qed.colors)
        self.assertEqual([2./3.,0.,0.,2./3.,0.], realproc_qed.charges)
 ##       self.assertEqual(realproc.permutation, [1,2,4,5,3])


    def test_find_fks_j_from_i(self):
        """tests that the find_fks_j_from_i function of a FKSRealProcess returns the
        correct result"""
        #u g > u g
        fksproc = fks_base.FKSProcess(self.myproc)
        #u g > g u
        fksproc_qed = fks_base.FKSProcess(self.myproc_qed)
        #take the first real for this process 2j 21 >2 21 21i
        leglist = fksproc.reals[0][0]
        # 2j 21 > 21 2 22i
        leglist_qed = fksproc_qed.reals[0][0]
        realproc = fks_base.FKSRealProcess(fksproc.born_proc, leglist, 1,0,\
                                           perturbed_orders = ['QCD'])
        realproc_qed = fks_base.FKSRealProcess(fksproc_qed.born_proc,leglist_qed,1,0,\
                                               perturbed_orders = ['QED'])
        target_full = {1:[], 2:[], 3:[1,2], 4:[1,2,3,5], 5:[1,2,3,4] }
        target_full_qed = {1:[],2:[],3:[],4:[1],5:[1,4]}
        borns = [[2,21,2,21], [21,21,21,21], [2,-2,21,21]]
        borns_qed = [[22,21,21,22],[2,21,21,2]]
        self.assertEqual(target_full, realproc.find_fks_j_from_i(borns))
        self.assertEqual(target_full_qed,realproc_qed.find_fks_j_from_i(borns_qed))
        #now the fks_j from_i corresponding onluy to u g > u g born
        target_born = {1:[], 2:[], 3:[], 4:[1,2,3,5], 5:[1,2,3,4]}
        borns = [[2,21,2,21]]
        self.assertEqual(target_born, realproc.find_fks_j_from_i(borns))
        #now the fks_j from_i corresponding onluy to u g > g u born
        target_born_qed = {1:[], 2:[], 3:[], 4:[], 5:[1,4]}
        borns_qed = [[2,21,21,2]]
        self.assertEqual(target_born_qed, realproc_qed.find_fks_j_from_i(borns_qed))


    def test_fks_real_process_get_leg_i_j(self):
        """test the correct output of the FKSRealProcess.get_leg_i/j() function"""
        #u g > u g
        fksproc = fks_base.FKSProcess(self.myproc)
        #take the first real for this process 2j 21 > 2 21 21i
        leglist = fksproc.reals[0][0]
        realproc = fks_base.FKSRealProcess(fksproc.born_proc, leglist,1,0)
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
                [{'i':5, 'j':1, 'ij':1, 'ij_glu':0, 'need_color_links':True,
                    'rb_links':[{'born_conf': 0, 'real_conf': 11},
                                {'born_conf': 1, 'real_conf': 10},
                                {'born_conf': 2, 'real_conf': 9}]},
                 {'i':5, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':True,
                     'rb_links':[{'born_conf': 0, 'real_conf': 14},
                                 {'born_conf': 1, 'real_conf': 4},
                                 {'born_conf': 2, 'real_conf': 7}]},
                 {'i':5, 'j':3, 'ij':3, 'ij_glu':0, 'need_color_links':True,
                     'rb_links':[{'born_conf': 0, 'real_conf': 1},
                                {'born_conf': 1, 'real_conf': 13},
                                {'born_conf': 2, 'real_conf': 8}]},
                 {'i':5, 'j':4, 'ij':4, 'ij_glu':4, 'need_color_links':True,
                     'rb_links':[{'born_conf': 0, 'real_conf': 2},
                                 {'born_conf': 1, 'real_conf': 5},
                                 {'born_conf': 2, 'real_conf': 12}]}])

        
    def test_find_reals(self):
        """tests if all the real processes are found for a given born"""
        #process is u g > u g
        fksproc = fks_base.FKSProcess(self.myproc)
        target = []
        # for the QED the born is u g > g u instead of u g > u g
        fksproc_qed = fks_base.FKSProcess(self.myproc_qed)
        target_qed = []
        
        #leg 1 can split as u g > u g g or  g g > u u~ g 
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
        #leg 1 can split as u g > g u a or  a g > g u u~ 
        target_qed.append( [fks_common.to_fks_legs([
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
                                         {'id' :21,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :2,
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
                                         {'id' :21,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
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
                                        ],self.mymodel ),
                    fks_common.to_fks_legs([#ud~>d~ug
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
                                        ], self.mymodel),
                    fks_common.to_fks_legs([#uu>uug
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
                                        ], self.mymodel),
                    fks_common.to_fks_legs([#uu~>uu~g
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
                                        ], self.mymodel)
                    ])
        target_qed.append([])
        #leg 3 can split as u g > u g g
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
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel)
                                        ]
                                        )
        target_qed.append([])
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
        #leg 3 can split as u g > g u a
        target_qed.append( [fks_common.to_fks_legs([
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
                                         {'id' :21,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg( 
                                         {'id' :22,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel)
                                        ]
                                        )
        self.assertEqual(len(fksproc.reals),len(target))
        self.assertEqual([len(real) for real in fksproc.reals],\
                         [len(real) for real in target])
        self.assertEqual(len(fksproc_qed.reals),len(target_qed))
        self.assertEqual([len(real) for real in fksproc_qed.reals],\
                         [len(real) for real in target_qed])                
        for i in range(len(fksproc.reals)):
            for j in range(len(fksproc.reals[i])):
                self.assertEqual(fksproc.reals[i][j], target[i][j])
        for i in range(len(fksproc_qed.reals)):
            for j in range(len(fksproc_qed.reals[i])):
                self.assertEqual(fksproc_qed.reals[i][j], target_qed[i][j]) 
        
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
                                        ], self.mymodel)]
                                        )
        
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
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ], self.mymodel)]
                                        )
        
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
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel)])

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
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel)])

        for i in range(len(fksproc2.reals)):
            self.assertEqual(fksproc2.reals[i], target2[i])       
    
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
                                        ], self.mymodel)]
                                        )
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
                                        ], self.mymodel)]
                                        )        

        for real, res in zip(fksproc3.reals, target3):
            self.assertEqual(real, res) 


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
        self.assertEqual(fks_p_s.born_proc, fks_p_u.born_proc)
        self.assertEqual(fks_p_s.born_amp, fks_p_u.born_amp)
        self.assertEqual(fks_p_s_qed.born_proc, fks_p_u_qed.born_proc)
        self.assertEqual(fks_p_s_qed.born_amp,fks_p_u_qed.born_amp)
        self.assertEqual(fks_p_s.born_proc,fks_p_s_qed.born_proc)

        fks_a_s = fks_base.FKSProcess(amp_s)
        fks_a_u = fks_base.FKSProcess(amp_u)
        fks_a_s_qed = fks_base.FKSProcess(amp_s_qed)
        fks_a_u_qed = fks_base.FKSProcess(amp_u_qed)

        self.assertEqual(fks_a_s.born_proc, fks_a_u.born_proc)
        self.assertEqual(fks_a_s.born_amp, fks_a_u.born_amp)
        self.assertEqual(fks_a_s_qed.born_proc,fks_a_u_qed.born_proc)
        self.assertEqual(fks_a_s_qed.born_amp,fks_a_u_qed.born_amp)
        self.assertEqual(fks_a_s.born_proc,fks_a_s_qed.born_proc)

        self.assertEqual(fks_a_s.born_proc['legs_with_decays'], MG.LegList())
        self.assertEqual(fks_a_u.born_proc['legs_with_decays'], MG.LegList())
        self.assertEqual(fks_a_s_qed.born_proc['legs_with_decays'], MG.LegList())
        self.assertEqual(fks_a_u_qed.born_proc['legs_with_decays'], MG.LegList())
