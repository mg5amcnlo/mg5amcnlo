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

"""Testing modules for fks_common functions and classes"""

import sys
import os
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import tests.unit_tests as unittest
import madgraph.fks.fks_common as fks_common
import madgraph.core.base_objects as MG
import madgraph.core.color_algebra as color
import madgraph.core.color_amp as color_amp
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import copy
import array
import fractions

class TestFKSCommon(unittest.TestCase):
    """ a class to test FKS common functions and classes"""
    
    ##define a model:
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

    model = MG.Model()
    model.set('particles', mypartlist)
    model.set('interactions', myinterlist)
    
    def test_combine_ij(self):
        """tests if legs i/j are correctly combined into leg ij"""
        legs_i = []
        legs_j = []
        legs_ij = []
        #i,j final u~ u pair
        legs_i.append(MG.Leg({'id' : -2, 'state' : True, 'number': 3}))
        legs_j.append(MG.Leg({'id' : 2, 'state' : True, 'number': 4}))
        legs_ij.append([MG.Leg({'id' : 21, 'state' : True , 'number' : 3})
                        ])
        #i,j final u u~ pair ->NO COMBINATION
        legs_i.append(MG.Leg({'id' : 2, 'state' : True, 'number': 3}))
        legs_j.append(MG.Leg({'id' : -2, 'state' : True, 'number': 4}))
        legs_ij.append([])
        #i,j final u d~ pair ->NO COMBINATION
        legs_i.append(MG.Leg({'id' : 2, 'state' : True, 'number': 3}))
        legs_j.append(MG.Leg({'id' : -1, 'state' : True, 'number': 4}))
        legs_ij.append([])
        #i,j initial/final u quark
        legs_i.append(MG.Leg({'id' : 2, 'state' : True, 'number': 3}))
        legs_j.append(MG.Leg({'id' : 2, 'state' : False, 'number': 1}))
        legs_ij.append([MG.Leg({'id' : 21, 'state' : False , 'number' : 1})
                        ])
        #i,j initial/final u~ quark
        legs_i.append(MG.Leg({'id' : -2, 'state' : True, 'number': 3}))
        legs_j.append(MG.Leg({'id' : -2, 'state' : False, 'number': 1}))
        legs_ij.append([MG.Leg({'id' : 21, 'state' : False , 'number' : 1})
                        ])
        #i,j initial/final u quark, i initial ->NO COMBINATION
        legs_i.append(MG.Leg({'id' : 2, 'state' : False, 'number': 3}))
        legs_j.append(MG.Leg({'id' : 2, 'state' : True, 'number': 1}))
        legs_ij.append([])
        #i, j final glu /quark
        legs_i.append(MG.Leg({'id' : 21, 'state' : True, 'number': 3}))
        legs_j.append(MG.Leg({'id' : 2, 'state' : True, 'number': 4}))
        legs_ij.append([MG.Leg({'id' : 2, 'state' : True , 'number' : 3})
                        ])
        #i, j final glu /anti quark
        legs_i.append(MG.Leg({'id' : 21, 'state' : True, 'number': 3}))
        legs_j.append(MG.Leg({'id' : -2, 'state' : True, 'number': 4}))
        legs_ij.append([MG.Leg({'id' : -2, 'state' : True , 'number' : 3})
                        ])
        #i, j final glu /initial quark
        legs_i.append(MG.Leg({'id' : 21, 'state' : True, 'number': 3}))
        legs_j.append(MG.Leg({'id' : 2, 'state' : False, 'number': 1}))
        legs_ij.append([MG.Leg({'id' : 2, 'state' : False , 'number' : 1})
                        ])
        #i, j final glu /initial anti quark
        legs_i.append(MG.Leg({'id' : 21, 'state' : True, 'number': 3}))
        legs_j.append(MG.Leg({'id' : -2, 'state' : False, 'number': 1}))
        legs_ij.append([MG.Leg({'id' : -2, 'state' : False , 'number' : 1})
                        ])
        #i, j final: j glu, i quark -> NO COMBINATION
        legs_i.append(MG.Leg({'id' : 2, 'state' : True, 'number': 3}))
        legs_j.append(MG.Leg({'id' : 21, 'state' : True, 'number': 4}))
        legs_ij.append([])
        # i, j final final gluons 
        legs_i.append(MG.Leg({'id' : 21, 'state' : True, 'number': 3}))
        legs_j.append(MG.Leg({'id' : 21, 'state' : True, 'number': 4}))
        legs_ij.append([MG.Leg({'id' : 21, 'state' : True , 'number' : 3})
                        ])
        # i, j final final/initial gluons 
        legs_i.append(MG.Leg({'id' : 21, 'state' : True, 'number': 3}))
        legs_j.append(MG.Leg({'id' : 21, 'state' : False, 'number': 1}))
        legs_ij.append([MG.Leg({'id' : 21, 'state' : False , 'number' : 1})
                        ])
        # i, j initial gluon/final quark 
        legs_i.append(MG.Leg({'id' : 2, 'state' : True, 'number': 3}))
        legs_j.append(MG.Leg({'id' : 21, 'state' : False, 'number': 1}))
        legs_ij.append([MG.Leg({'id' : -2, 'state' : False , 'number' : 1})
                        ])
        
        dict = {}
 
        for i, j, ij in zip(legs_i, legs_j, legs_ij):
            self.assertEqual(fks_common.combine_ij(
                                    fks_common.to_fks_leg(i, self.model), 
                                    fks_common.to_fks_leg(j, self.model), 
                                    self.model, dict, 'QCD'), 
                                    fks_common.to_fks_legs(ij, self.model))
        


    def test_find_pert_particles_interactions(self):
        """test if interactions, particles and massless particles corresponding
        to the perturbative expansion are correctly extracted from the model"""
        
        dict = fks_common.find_pert_particles_interactions(self.model, 'QCD')
        res_int = self.expected_qcd_inter
        res_part = [-6,-2,-1,1,2,6,21]
        res_soft = [-2,-1,1,2,21]
        self.assertEqual(dict['pert_particles'], res_part)
        self.assertEqual(dict['soft_particles'], res_soft)
        self.assertEqual(dict['interactions'], res_int)
    
    
    def test_to_fks_leg_s(self):
        """tests if color, massless and spin entries of a fks leg/leglist
         are correctly set"""
        leg_list = MG.LegList()
        res_list = fks_common.FKSLegList()
        leg_list.append( MG.Leg({'id' : 21, 
                                     'state' : True, 
                                     'number' : 5}))
        res_list.append( fks_common.FKSLeg({'id' : 21, 
                                     'state' : True, 
                                     'number' : 5,
                                     'massless' : True,
                                     'color' : 8,
                                     'spin' : 3})) 
        leg_list.append( MG.Leg({'id' : 6, 
                                     'state' : True, 
                                     'number' : 5}))
        res_list.append( fks_common.FKSLeg({'id' : 6, 
                                     'state' : True, 
                                     'number' : 5,
                                     'massless' : False,
                                     'color' : 3,
                                     'spin' : 2}))  
        leg_list.append( MG.Leg({'id' : -1, 
                                     'state' : True, 
                                     'number' : 5}))
        res_list.append( fks_common.FKSLeg({'id' : -1, 
                                     'state' : True, 
                                     'number' : 5,
                                     'massless' : True,
                                     'color' : -3,
                                     'spin' : 2}))  
        leg_list.append( MG.Leg({'id' : -1, 
                                     'state' : False, 
                                     'number' : 5}))
        res_list.append( fks_common.FKSLeg({'id' : -1, 
                                     'state' : False, 
                                     'number' : 5,
                                     'massless' : True,
                                     'color' : -3,
                                     'spin' : 2})) 
    
    
        self.assertEqual(fks_common.to_fks_legs(leg_list, self.model), res_list)
            
        for leg, res in zip(leg_list, res_list):
            self.assertEqual(fks_common.to_fks_leg(leg, self.model), res)                                                                                            

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

        fkslegs = fks_common.to_fks_legs(myleglist, self.model)
        color_links = fks_common.find_color_links(fkslegs)

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
        
        self.assertEqual(len(links), len(color_links))
        for l1, l2 in zip (links, color_links):
            self.assertEqual(l1,l2['legs'])
            
    def test_insert_color_links(self):
        """given a list of color links, tests if the insert color link works, ie 
        if a list of dictionaries is returned. Each dict has the following entries:
        --link, list of number of linked legs
        --link_basis the linked color basis
        --link_matrix the color matrix created from the original basis"""
        #test the process u u~ > d d~, link u and d
        leglist = MG.LegList([
                    MG.Leg({'id':2, 'state':False, 'number':1}),
                    MG.Leg({'id':-2, 'state':False, 'number':2}),
                    MG.Leg({'id':1, 'state':True, 'number':3}),
                    MG.Leg({'id':-1, 'state':True, 'number':4}),
                    ])
        myproc = MG.Process({'legs' : leglist,
                       'orders':{'QCD':10, 'QED':0},
                       'model': self.model,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}})
        helas = helas_objects.HelasMatrixElement(
                        diagram_generation.Amplitude(myproc))
        
        basis_orig = copy.deepcopy(helas['color_basis'])
        
        links = fks_common.find_color_links(
                                    fks_common.to_fks_legs(leglist, self.model))
        dicts = fks_common.insert_color_links(
                    helas['color_basis'],
                    helas['color_basis'].create_color_dict_list(helas['base_amplitude']),
                    links)
        #color_string for link 1-3 (dicts[1])
        linkstring = color.ColorString([
                        color.T(-1000,2,-3001), color.T(-1000,-3002,4),
                        color.T(-6000,-3001,1), color.T(-6000,3,-3002)])
        linkstring.coeff = linkstring.coeff * (-1)
        
        linkdicts = [{(0,0) : linkstring}]
        
        link_basis = color_amp.ColorBasis()
        for i, dict in enumerate(linkdicts):
            link_basis.update_color_basis(dict, i)
        
        matrix = color_amp.ColorMatrix(basis_orig, link_basis)
        
        self.assertEqual(len(dicts), 12)
        self.assertEqual(link_basis, dicts[1]['link_basis'])
        self.assertEqual(matrix, dicts[1]['link_matrix'])
        

    def test_legs_to_color_link_string(self):
        """tests if, given two fks legs, the color link between them is correctly 
        computed, i.e. if string and replacements are what they are expected to be"""
        pairs = []
        strings = []
        replacs = []
        ###MASSIVE
        #FINAL t with itself
        pairs.append([MG.Leg({'id' : 6, 'state' : True, 'number' : 5 }),
                      MG.Leg({'id' : 6, 'state' : True, 'number' : 5 })])
        replacs.append([[5, -3001]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000,-6000, 5, -3001)],
                       coeff = fractions.Fraction(1,2)))
        #FINAL anti-t with itself
        pairs.append([MG.Leg({'id' : -6, 'state' : True, 'number' : 5 }),
                      MG.Leg({'id' : -6, 'state' : True, 'number' : 5 })])
        replacs.append([[5, -3001]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000,-6000, -3001, 5)],
                       coeff = fractions.Fraction(1,2)))
        ####INITIAL-INITIAL
        #INITIAL state quark with INITIAL state quark
        pairs.append([MG.Leg({'id' : 1, 'state' : False, 'number' : 1 }),
                      MG.Leg({'id' : 2, 'state' : False, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, -3001, 1),color.T(-6000, -3002, 2)],
                       coeff = fractions.Fraction(1,1)))
        #INITIAL state quark with INITIAL state anti-quark
        pairs.append([MG.Leg({'id' : 1, 'state' : False, 'number' : 1 }),
                      MG.Leg({'id' : -2, 'state' : False, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, -3001, 1),color.T(-6000, 2,-3002)],
                       coeff = fractions.Fraction(-1,1)))
        #INITIAL state anti-quark with INITIAL state quark
        pairs.append([MG.Leg({'id' : -1, 'state' : False, 'number' : 1 }),
                      MG.Leg({'id' : 2, 'state' : False, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, 1, -3001),color.T(-6000, -3002, 2)],
                       coeff = fractions.Fraction(-1,1)))
        #INITIAL state anti-quark with INITIAL state anti-quark
        pairs.append([MG.Leg({'id' : -1, 'state' : False, 'number' : 1 }),
                      MG.Leg({'id' : -2, 'state' : False, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, 1,-3001),color.T(-6000, 2,-3002)],
                       coeff = fractions.Fraction(1,1)))
        #INITIAL state quark with INITIAL state gluon
        pairs.append([MG.Leg({'id' : 1, 'state' : False, 'number' : 1 }),
                      MG.Leg({'id' : 21, 'state' : False, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, -3001,1),color.f(-3002,-6000, 2)],
                       coeff = fractions.Fraction(1,1),
                       is_imaginary = True))
        #INITIAL state anti-quark with INITIAL state gluon
        pairs.append([MG.Leg({'id' : -1, 'state' : False, 'number' : 1 }),
                      MG.Leg({'id' : 21, 'state' : False, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, 1, -3001),color.f(-3002,-6000, 2)],
                       coeff = fractions.Fraction(-1,1),
                       is_imaginary = True))
        #INITIAL state gluon with INITIAL state gluon
        pairs.append([MG.Leg({'id' : 21, 'state' : False, 'number' : 1 }),
                      MG.Leg({'id' : 21, 'state' : False, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.f(-3001,-6000, 1),color.f(-3002,-6000, 2)],
                       coeff = fractions.Fraction(-1,1),
                       is_imaginary = False))
        ####FINAL-FINIAL
        #FINAL state quark with FINIAL state quark
        pairs.append([MG.Leg({'id' : 1, 'state' : True, 'number' : 1 }),
                      MG.Leg({'id' : 2, 'state' : True, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, 1,-3001),color.T(-6000, 2,-3002)],
                       coeff = fractions.Fraction(1,1)))
        #FINAL state quark with FINAL state anti-quark
        pairs.append([MG.Leg({'id' : 1, 'state' : True, 'number' : 1 }),
                      MG.Leg({'id' : -2, 'state' : True, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, 1,-3001),color.T(-6000,-3002,2)],
                       coeff = fractions.Fraction(-1,1)))
        #FINAL state anti-quark with FINAL state quark
        pairs.append([MG.Leg({'id' : -1, 'state' : True, 'number' : 1 }),
                      MG.Leg({'id' : 2, 'state' : True, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, -3001,1),color.T(-6000, 2,-3002)],
                       coeff = fractions.Fraction(-1,1)))
        #FINAL state anti-quark with FINAL state anti-quark
        pairs.append([MG.Leg({'id' : -1, 'state' : True, 'number' : 1 }),
                      MG.Leg({'id' : -2, 'state' : True, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, -3001,1),color.T(-6000,-3002,2)],
                       coeff = fractions.Fraction(1,1)))
        #FINAL state quark with FINAL state gluon
        pairs.append([MG.Leg({'id' : 1, 'state' : True, 'number' : 1 }),
                      MG.Leg({'id' : 21, 'state' : True, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, 1,-3001),color.f(-3002,-6000, 2)],
                       coeff = fractions.Fraction(-1,1),
                       is_imaginary = True))
        #FINAL state anti-quark with FINAL state gluon
        pairs.append([MG.Leg({'id' : -1, 'state' : True, 'number' : 1 }),
                      MG.Leg({'id' : 21, 'state' : True, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, -3001,1),color.f(-3002,-6000, 2)],
                       coeff = fractions.Fraction(1,1),
                       is_imaginary = True))
        #FINAL state gluon with FINAL state gluon
        pairs.append([MG.Leg({'id' : 21, 'state' : True, 'number' : 1 }),
                      MG.Leg({'id' : 21, 'state' : True, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.f(-3001,-6000, 1),color.f(-3002,-6000, 2)],
                       coeff = fractions.Fraction(-1,1),
                       is_imaginary = False))
        ###INITIAL-FINAL
        #INITIAL state quark with FINAL state quark
        pairs.append([MG.Leg({'id' : 1, 'state' : False, 'number' : 1 }),
                      MG.Leg({'id' : 2, 'state' : True, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, -3001, 1),color.T(-6000, 2,-3002)],
                       coeff = fractions.Fraction(-1,1)))
        #INITIAL state quark with FINAL state anti-quark
        pairs.append([MG.Leg({'id' : 1, 'state' : False, 'number' : 1 }),
                      MG.Leg({'id' : -2, 'state' : True, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, -3001, 1),color.T(-6000, -3002,2)],
                       coeff = fractions.Fraction(1,1)))
        #INITIAL state anti-quark with FINAL state quark
        pairs.append([MG.Leg({'id' : -1, 'state' : False, 'number' : 1 }),
                      MG.Leg({'id' : 2, 'state' : True, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, 1, -3001),color.T(-6000, 2,-3002)],
                       coeff = fractions.Fraction(1,1)))
        #INITIAL state anti-quark with FINAL state anti-quark
        pairs.append([MG.Leg({'id' : -1, 'state' : False, 'number' : 1 }),
                      MG.Leg({'id' : -2, 'state' : True, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, 1,-3001),color.T(-6000, -3002,2)],
                       coeff = fractions.Fraction(-1,1)))
        #INITIAL state quark with FINAL state gluon
        pairs.append([MG.Leg({'id' : 1, 'state' : False, 'number' : 1 }),
                      MG.Leg({'id' : 21, 'state' : True, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, -3001,1),color.f(-3002,-6000, 2)],
                       coeff = fractions.Fraction(1,1),
                       is_imaginary = True))
        #INITIAL state anti-quark with FINAL state gluon
        pairs.append([MG.Leg({'id' : -1, 'state' : False, 'number' : 1 }),
                      MG.Leg({'id' : 21, 'state' : True, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, 1, -3001),color.f(-3002,-6000, 2)],
                       coeff = fractions.Fraction(-1,1),
                       is_imaginary = True))
        #FINAL state quark with INITIAL state gluon
        pairs.append([MG.Leg({'id' : 1, 'state' : True, 'number' : 1 }),
                      MG.Leg({'id' : 21, 'state' : False, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, 1,-3001),color.f(-3002,-6000, 2)],
                       coeff = fractions.Fraction(-1,1),
                       is_imaginary = True))
        #FINAL state anti-quark with INITIAL state gluon
        pairs.append([MG.Leg({'id' : -1, 'state' : True, 'number' : 1 }),
                      MG.Leg({'id' : 21, 'state' : False, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.T(-6000, -3001,1),color.f(-3002,-6000, 2)],
                       coeff = fractions.Fraction(1,1),
                       is_imaginary = True))
        #FINAL state gluon with INITIAL state gluon
        pairs.append([MG.Leg({'id' : 21, 'state' : True, 'number' : 1 }),
                      MG.Leg({'id' : 21, 'state' : False, 'number' : 2 })])
        replacs.append([[1, -3001], [2, -3002]])
        strings.append(color.ColorString(init_list = [
                       color.f(-3001,-6000, 1),color.f(-3002,-6000, 2)],
                       coeff = fractions.Fraction(-1,1),
                       is_imaginary = False))
        
        for pair, string, replac in zip(pairs, strings, replacs):
            dict = fks_common.legs_to_color_link_string(\
                fks_common.to_fks_leg(pair[0],self.model), 
                fks_common.to_fks_leg(pair[1],self.model)
                                                    )
            self.assertEqual(string, dict['string'])
            self.assertEqual(replac, dict['replacements'])
            