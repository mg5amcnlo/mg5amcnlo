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

"""Unit test library for the various properties of objects in 
   loop_diagram_generaiton"""

import copy
import itertools
import logging
import math


import tests.unit_tests as unittest

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
from madgraph import MadGraph5Error

#===============================================================================
# LoopDiagramGeneration Test
#===============================================================================
class LoopDiagramGenerationTest(unittest.TestCase):
    """Test class for all functions related to the Loop diagram generation"""

    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    myloopmodel = loop_base_objects.LoopModel()
    myprocess = base_objects.Process()

    ref_dict_to0 = {}
    ref_dict_to1 = {}

    myamplitude = diagram_generation.Amplitude()

    mypertorders=['QCD','QED']

    def setUp(self):
        """Setup the NLO model"""
        
        # A gluon
        self.mypartlist.append(base_objects.Particle({'name':'g',
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
                      'perturbation':['QCD',],
                      'self_antipart':True}))
        
        # A quark U and its antiparticle
        self.mypartlist.append(base_objects.Particle({'name':'u',
                      'antiname':'u~',
                      'spin':2,
                      'color':3,
                      'mass':'umass',
                      'width':'zero',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':2,
                      'propagating':True,
                      'is_part':True,
                      'perturbation':['QCD','QED'],
                      'self_antipart':False}))
        antiu = copy.copy(self.mypartlist[1])
        antiu.set('is_part', False)

        # A quark D and its antiparticle
        self.mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':3,
                      'mass':'dmass',
                      'width':'zero',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'perturbation':['QCD','QED'],           
                      'self_antipart':False}))
        antid = copy.copy(self.mypartlist[2])
        antid.set('is_part', False)

        # A photon
        self.mypartlist.append(base_objects.Particle({'name':'a',
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
                      'perturbation':['QED',],
                      'self_antipart':True}))

        # A electron and positron
        self.mypartlist.append(base_objects.Particle({'name':'e-',
                      'antiname':'e+',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'e^-',
                      'antitexname':'e^+',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':11,
                      'propagating':True,
                      'perturbation':['QED',],
                      'is_part':True,
                      'self_antipart':False}))
        antie = copy.copy(self.mypartlist[4])
        antie.set('is_part', False)

        # First set up the base interactions.

        # 3 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G^2'},
                      'orders':{'QCD':2}}))

        # Gluon and photon couplings to quarks
        self.myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of e to gamma

        self.myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[4], \
                                             antie, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Then set up the R2 interactions proportional to those existing in the
        # tree-level model.

        # 3 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':3},
                      'type':['R2',()]}))

        # 4 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G^2'},
                      'orders':{'QCD':4},
                      'type':['R2',()]}))

        # Gluon and photon couplings to quarks
        self.myinterlist.append(base_objects.Interaction({
                      'id': 10,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':3},
                      'type':['R2',()]}))
            
        self.myinterlist.append(base_objects.Interaction({
                      'id': 11,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1, 'QED':2},
                      'type':['R2',()]}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 12,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1, 'QCD':2},
                      'type':['R2',()]}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 13,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':3},
                      'type':['R2',()]}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 14,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':3},
                      'type':['R2',()]}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 15,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1, 'QED':2},
                      'type':['R2',()]}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 16,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1, 'QCD':2},
                      'type':['R2',()]}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 17,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':3},
                      'type':['R2',()]}))

        # Coupling of e to gamma

        self.myinterlist.append(base_objects.Interaction({
                      'id': 18,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[4], \
                                             antie, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':3},
                      'type':['R2',()]}))

        # R2 interactions not proportional to the base interactions

        # Two point interactions

        # The gluon
        self.myinterlist.append(base_objects.Interaction({
                      'id': 19,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 2),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':2},
                      'type':['R2',()]}))

        # The photon
        self.myinterlist.append(base_objects.Interaction({
                      'id': 20,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[3]] * 2),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QED':2},
                      'type':['R2',()]}))

        # The electron
        self.myinterlist.append(base_objects.Interaction({
                      'id': 21,
                      'particles': base_objects.ParticleList([\
                                            self.mypartlist[4], \
                                             antie]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QED':2},
                      'type':['R2',()]}))

        # The up quark, R2QED
        self.myinterlist.append(base_objects.Interaction({
                      'id': 22,
                      'particles': base_objects.ParticleList([\
                                            self.mypartlist[2], \
                                             antid]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QED':2},
                      'type':['R2',()]}))

        # The up quark, R2QCD
        self.myinterlist.append(base_objects.Interaction({
                      'id': 23,
                      'particles': base_objects.ParticleList([\
                                            self.mypartlist[2], \
                                             antid]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':2},
                      'type':['R2',()]}))

        # The down quark, R2QED
        self.myinterlist.append(base_objects.Interaction({
                      'id': 24,
                      'particles': base_objects.ParticleList([\
                                            self.mypartlist[1], \
                                             antiu]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QED':2},
                      'type':['R2',()]}))

        # The down quark, R2QCD
        self.myinterlist.append(base_objects.Interaction({
                      'id': 25,
                      'particles': base_objects.ParticleList([\
                                            self.mypartlist[1], \
                                             antid]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':2},
                      'type':['R2',()]}))

        # The R2 three and four point interactions not proportional to the
        # base interaction

        # 3 photons
        self.myinterlist.append(base_objects.Interaction({
                      'id': 26,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[3]] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QED':3},
                      'type':['R2',()]}))

        # 2 photon and 1 gluons
        self.myinterlist.append(base_objects.Interaction({
                      'id': 27,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[3],\
                                            self.mypartlist[3],\
                                            self.mypartlist[0],]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QED':2, 'QCD':1},
                      'type':['R2',()]}))

        # 1 photon and 2 gluons
        self.myinterlist.append(base_objects.Interaction({
                      'id': 28,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[3],\
                                            self.mypartlist[0],\
                                            self.mypartlist[0],]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QED':1, 'QCD':2},
                      'type':['R2',()]}))

        # 4 photons
        self.myinterlist.append(base_objects.Interaction({
                      'id': 29,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[3]] * 4),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QED':4},
                      'type':['R2',()]}))

        # 3 photons and 1 gluon
        self.myinterlist.append(base_objects.Interaction({
                      'id': 30,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[3],\
                                            self.mypartlist[3],\
                                            self.mypartlist[3],\
                                            self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QED':3,'QCD':1},
                      'type':['R2',()]}))

        # 2 photons and 2 gluons
        self.myinterlist.append(base_objects.Interaction({
                      'id': 31,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[3],\
                                            self.mypartlist[3],\
                                            self.mypartlist[0],\
                                            self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QED':2,'QCD':2},
                      'type':['R2',()]}))

        # 1 photon and 3 gluons
        self.myinterlist.append(base_objects.Interaction({
                      'id': 32,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[3],\
                                            self.mypartlist[0],\
                                            self.mypartlist[0],\
                                            self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QED':1,'QCD':3},
                      'type':['R2',()]})) 

        # Finally the UV interactions Counter-Terms

        # 3 gluon vertex CT
        self.myinterlist.append(base_objects.Interaction({
                      'id': 33,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':3},
                      'type':['UV',()]}))

        # 4 gluon vertex CT
        self.myinterlist.append(base_objects.Interaction({
                      'id': 34,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G^2'},
                      'orders':{'QCD':4},
                      'type':['UV',()]}))

        # Gluon and photon couplings to quarks CT
        self.myinterlist.append(base_objects.Interaction({
                      'id': 35,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':3},
                      'type':['UV',()]}))
        
        # this is the CT for the renormalization of the QED corrections to alpha_QCD
        self.myinterlist.append(base_objects.Interaction({
                      'id': 36,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1, 'QED':2},
                      'type':['UV',()]}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 37,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1, 'QCD':2},
                      'type':['UV',()]}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 38,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':3},
                      'type':['UV',()]}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 39,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':3},
                      'type':['UV',()]}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 40,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1, 'QED':2},
                      'type':['UV',()]}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 41,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1, 'QCD':2},
                      'type':['UV',()]}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 42,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':3},
                      'type':['UV',()]}))
        
        # alpha_QED to electron CT

        self.myinterlist.append(base_objects.Interaction({
                      'id': 43,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[4], \
                                             antie, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':3},
                      'type':['UV',()]}))
          
        # Finally the mass renormalization of the up and down quark dotted with
        # a mass for the occasion

        # The up quark, UVQED
        self.myinterlist.append(base_objects.Interaction({
                      'id': 44,
                      'particles': base_objects.ParticleList([\
                                            self.mypartlist[2], \
                                             antid]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QED':2},
                      'type':['UV',()]}))

        # The up quark, UVQCD
        self.myinterlist.append(base_objects.Interaction({
                      'id': 45,
                      'particles': base_objects.ParticleList([\
                                            self.mypartlist[2], \
                                             antid]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':2},
                      'type':['UV',()]}))

        # The down quark, UVQED
        self.myinterlist.append(base_objects.Interaction({
                      'id': 46,
                      'particles': base_objects.ParticleList([\
                                            self.mypartlist[1], \
                                             antiu]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QED':2},
                      'type':['UV',()]}))

        # The down quark, UVQCD
        self.myinterlist.append(base_objects.Interaction({
                      'id': 47,
                      'particles': base_objects.ParticleList([\
                                            self.mypartlist[1], \
                                             antid]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':2},
                      'type':['UV',()]}))

        
        self.myloopmodel.set('particles', self.mypartlist)
        self.myloopmodel.set('couplings', ['QCD','QED'])        
        self.myloopmodel.set('interactions', self.myinterlist)
        self.myloopmodel.set('perturbation_couplings', self.mypertorders)
        self.myloopmodel.set('order_hierarchy', {'QCD':1,'QED':2})

        self.ref_dict_to0 = self.myinterlist.generate_ref_dict(self.mypertorders)[0]
        self.ref_dict_to1 = self.myinterlist.generate_ref_dict(self.mypertorders)[1]

    def test_NLOAmplitude(self):
        """test different features of the NLOAmplitude class"""
        ampNLOlist=[]
        ampdefaultlist=[]

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))
        dummyproc = base_objects.Process({'legs':myleglist,
                                          'model':self.myloopmodel})

        ampdefaultlist.append(diagram_generation.Amplitude())
        ampdefaultlist.append(diagram_generation.Amplitude(dummyproc))
        ampdefaultlist.append(diagram_generation.Amplitude({'process':dummyproc}))        
        ampdefaultlist.append(diagram_generation.DecayChainAmplitude(dummyproc,False))

        dummyproc.set("perturbation_couplings",self.mypertorders)
        ampNLOlist.append(loop_diagram_generation.LoopAmplitude({'process':dummyproc}))                
        ampNLOlist.append(loop_diagram_generation.LoopAmplitude())        
        ampNLOlist.append(loop_diagram_generation.LoopAmplitude(dummyproc))        

        # Test the __new__ constructor of NLOAmplitude
        for ampdefault in ampdefaultlist:
            self.assertTrue(isinstance(ampdefault,diagram_generation.Amplitude))
            self.assertFalse(isinstance(ampdefault,loop_diagram_generation.LoopAmplitude))            
        for ampNLO in ampNLOlist:
            self.assertTrue(isinstance(ampNLO,loop_diagram_generation.LoopAmplitude))

        # Now test for the usage of getter/setter of diagrams.
        ampNLO=loop_diagram_generation.LoopAmplitude(dummyproc)
        mydiaglist=base_objects.DiagramList([loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':1}),\
                                             loop_base_objects.LoopDiagram({'type':2}),\
                                             loop_base_objects.LoopDiagram({'type':3}),\
                                             loop_base_objects.LoopDiagram({'type':4})])        
        ampNLO.set('diagrams',mydiaglist)
        self.assertEqual(len(ampNLO.get('diagrams')),10)
        self.assertEqual(len(ampNLO.get('born_diagrams')),6)
        self.assertEqual(len(ampNLO.get('loop_diagrams')),4)        
        mydiaglist=base_objects.DiagramList([loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':0})])
        ampNLO.set('born_diagrams',mydiaglist)
        self.assertEqual(len(ampNLO.get('born_diagrams')),3)        

    def test_diagram_generation_epem_ddx(self):
        """Test the number of loop diagrams generated for e+e->dd~ (s channel)
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.myloopmodel,
                                       'perturbation_couplings':self.mypertorders})

    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myproc)
        myloopamplitude.generate_diagrams()

        #self.assertEqual(len(myloopamplitude.get('loop_diagrams')), 2)
        self.assertEqual(len(myloopamplitude.get('born_diagrams')), 1)

    def test_diagram_generation_uux_ddx(self):
        """Test the number of loop diagrams generated for e+e->dd~ (s channel)
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.myloopmodel,
                                       'orders':{'QED':5},
                                       'squared_orders': {},
                                       'perturbation_couplings':['QCD',]})

        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myproc)
        myloopamplitude.generate_diagrams()

        #self.assertEqual(len(myloopamplitude.get('loop_diagrams')), 2)
        self.assertEqual(len(myloopamplitude.get('born_diagrams')), 2)
