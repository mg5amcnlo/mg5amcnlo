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
import os


import tests.unit_tests as unittest


import madgraph.core.drawing as draw_lib
import madgraph.iolibs.drawing_eps as draw
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
           with different choices for the perturbation couplings and squared orders.
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

        ordersChoices=[({},['QCD'],{},1),\
                       ({},['QED'],{},7),\
                       ({},['QCD','QED'],{},10),\
                       ({},['QED','QCD'],{'QED':-1},3),\
                       ({},['QED','QCD'],{'QCD':-1},7)]

        for (bornOrders,pert,sqOrders,nDiagGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nDiagGoal)

            ### This is to plot the diagrams obtained
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/Users/Spooner/Documents/PhD/MG5/NLO', 'diagramsVall_' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(myloopamplitude['loop_diagrams'],
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude='',
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)

            ### This is to display some informations
            #mydiag1=myloopamplitude.get('loop_diagrams')[0]
            #mydiag2=myloopamplitude.get('loop_diagrams')[5]      
            #print "I got tag for diag 1=",mydiag1['canonical_tag']
            #print "I got tag for diag 2=",mydiag2['canonical_tag']
            #print "I got vertices for diag 1=",mydiag1['vertices']
            #print "I got vertices for diag 2=",mydiag2['vertices']
            #print "mydiag=",str(mydiag)
            #mydiag1.tag(trial,5,6,self.myloopmodel)
            #print "I got tag=",mydiag['tag']
            #print "I got struct[0]=\n",myloopamplitude['structure_repository'][0].nice_string()
            #print "I got struct[2]=\n",myloopamplitude['structure_repository'][2].nice_string()   
            #print "I got struct[3]=\n",myloopamplitude['structure_repository'][3].nice_string()

    def test_diagram_generation_gg_ng(self):
        """Test the number of loop diagrams generated for gg>ng. n being in [1,2,3]
        """
        
        # For quick test 
        nGluons = [(1,8),(2,81)]
        # For a longer one
        #nGluons += [(3,905),(4,11850)]

        for (n, nDiagGoal) in nGluons:
            myleglist=base_objects.LegList([base_objects.Leg({'id':21,
                                              'number':num,
                                              'loop_line':False}) \
                                              for num in range(1, (n+3))])
            myleglist[0].set('state',False)
            myleglist[1].set('state',False)        

            myproc=base_objects.Process({'legs':myleglist,
                                       'model':self.myloopmodel,
                                       'orders':{},
                                       'squared_orders': {},
                                       'perturbation_couplings':['QCD']})

            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')), nDiagGoal)


    def test_diagram_generation_uux_ddx(self):
        """Test the number of loop diagrams generated for uu~>dd~ for different choices
           of orders.
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

        ordersChoices=[({},['QCD','QED'],{},21),\
                       ({},['QCD','QED',],{'QED':-99},28),\
                       ({},['QCD'],{},9),\
                       ({},['QED'],{},2),\
                       ({'QED':0},['QCD'],{},9),\
                       ({'QCD':0},['QED'],{},7),\
                       ({},['QCD','QED'],{'QED':-1},9),\
                       ({},['QCD','QED'],{'QCD':-1},7),\
                       # These last two are of no physics interest
                       # It is just for the sake of the test.
                       ({'QED':0},['QCD','QED'],{},21),\
                       ({'QCD':0},['QCD','QED'],{},19)]

        for (bornOrders,pert,sqOrders,nDiagGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nDiagGoal)

    def test_diagram_generation_ddx_ddx(self):
        """Test the number of loop diagrams generated for uu~>dd~ for different choices
           of orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))

        ordersChoices=[({},['QCD','QED'],{},42),\
                       ({},['QCD','QED',],{'QED':-99},56),\
                       ({},['QED'],{},4),\
                       ({},['QCD'],{},18),\
                       ({'QED':0},['QCD'],{},18),\
                       ({'QCD':0},['QED'],{},14),\
                       ({},['QCD','QED'],{'QED':-1},18),\
                       ({},['QCD','QED'],{'QCD':-1},14)]
        for (bornOrders,pert,sqOrders,nDiagGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nDiagGoal)

#===============================================================================
# LoopDiagramFDStruct Test
#===============================================================================
class LoopDiagramFDStructTest(unittest.TestCase):
    """Test class for the tagging functions of LoopDiagram and FDStructure classes"""

    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    mymodel = base_objects.Model()
    myproc = base_objects.Process()
    myloopdiag = loop_base_objects.LoopDiagram()

    def setUp(self):
        """ Setup a toy-model with gluon and down-quark only """

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
                      'self_antipart':True}))

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
                      'self_antipart':False}))
        antid = copy.copy(self.mypartlist[1])
        antid.set('is_part', False)

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

        # Gluon coupling to the down-quark
        self.myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.mymodel.set('particles', self.mypartlist)
        self.mymodel.set('interactions', self.myinterlist)
        self.myproc.set('model',self.mymodel)

    def test_gg_5gglgl_bubble_tag(self):
        """ Test the gg>ggggg g*g* tagging of a bubble"""

        # Five gluon legs with two initial states
        myleglist = base_objects.LegList([base_objects.Leg({'id':21,
                                              'number':num,
                                              'loop_line':False}) \
                                              for num in range(1, 10)])
        myleglist[7].set('loop_line', True)
        myleglist[8].set('loop_line', True)
        l1=myleglist[0]
        l2=myleglist[1]
        l3=myleglist[2]
        l4=myleglist[3]
        l5=myleglist[4]
        l6=myleglist[5]
        l7=myleglist[6]
        l8=myleglist[7]
        l9=myleglist[8]

        self.myproc.set('legs',myleglist)

        l67 = base_objects.Leg({'id':21,'number':6,'loop_line':False})
        l56 = base_objects.Leg({'id':21,'number':5,'loop_line':False})
        l235 = base_objects.Leg({'id':21,'number':2,'loop_line':False}) 
        l24 = base_objects.Leg({'id':21,'number':2,'loop_line':False})
        l28 = base_objects.Leg({'id':21,'number':2,'loop_line':True})
        l19 = base_objects.Leg({'id':21,'number':1,'loop_line':True})

        vx19 = base_objects.Vertex({'legs':base_objects.LegList([l1, l9, l19]), 'id': 1})
        vx67 = base_objects.Vertex({'legs':base_objects.LegList([l6, l7, l67]), 'id': 1})
        vx56 = base_objects.Vertex({'legs':base_objects.LegList([l5, l67, l56]), 'id': 1})
        vx235 = base_objects.Vertex({'legs':base_objects.LegList([l2, l3, l56, l235]), 'id': 2})
        vx24 = base_objects.Vertex({'legs':base_objects.LegList([l4, l235, l24]), 'id': 1})
        vx28 = base_objects.Vertex({'legs':base_objects.LegList([l235, l8, l28]), 'id': 1})
        vx0 = base_objects.Vertex({'legs':base_objects.LegList([l19, l28]), 'id': 0})

        myVertexList=base_objects.VertexList([vx19,vx67,vx56,vx235,vx24,vx28,vx0])

        myBubbleDiag=loop_base_objects.LoopDiagram({'vertices':myVertexList,'type':21})

        myStructRep=loop_base_objects.FDStructureList()
        myStruct=loop_base_objects.FDStructure()

        goal_canonicalStruct=(((2, 3, 4, 5, 6, 7), 1), ((2, 3, 5, 6, 7), 2), ((5, 6, 7), 1), ((6, 7), 1))
        canonicalStruct=myBubbleDiag.construct_FDStructure(5, 0, 2, myStruct)
        self.assertEqual(canonicalStruct, goal_canonicalStruct)
        
        goal_vxList=base_objects.VertexList([vx67,vx56,vx235,vx24])
        myStruct.set('canonical',canonicalStruct)
        myStruct.generate_vertices(self.myproc)
        self.assertEqual(myStruct['vertices'],goal_vxList)

        goal_tag=[[21, [0], 1], [21, [1], 1]]
        vx28_tag=base_objects.Vertex({'legs':base_objects.LegList([l235, l8, l28]), 'id': 1})
        vx129_tag=base_objects.Vertex({'legs':base_objects.LegList([l1, l28, l9]), 'id': 1})
        goal_vertices=base_objects.VertexList([vx28_tag,vx129_tag])
        myBubbleDiag.tag(myStructRep,8,9,self.myproc)
        self.assertEqual(myBubbleDiag.get('canonical_tag'), goal_tag)
        self.assertEqual(myBubbleDiag.get('vertices'), goal_vertices)

    def test_gg_4gdldxl_penta_tag(self):
        """ Test the gg>gggg d*dx* tagging of a quark pentagon"""

        # Five gluon legs with two initial states
        myleglist = base_objects.LegList([base_objects.Leg({'id':21,
                                              'number':num,
                                              'loop_line':False}) \
                                              for num in range(1, 7)])
        myleglist.append(base_objects.Leg({'id':1,'number':7,'loop_line':True}))
        myleglist.append(base_objects.Leg({'id':-1,'number':8,'loop_line':True}))                         
        l1=myleglist[0]
        l2=myleglist[1]
        l3=myleglist[2]
        l4=myleglist[3]
        l5=myleglist[4]
        l6=myleglist[5]
        l7=myleglist[6]
        l8=myleglist[7]

        self.myproc.set('legs',myleglist)

        # One way of constructing this diagram, with a three-point amplitude
        l17 = base_objects.Leg({'id':1,'number':1,'loop_line':True})
        l12 = base_objects.Leg({'id':1,'number':1,'loop_line':True})
        l68 = base_objects.Leg({'id':-1,'number':6,'loop_line':True}) 
        l56 = base_objects.Leg({'id':-1,'number':5,'loop_line':True})
        l34 = base_objects.Leg({'id':21,'number':3,'loop_line':False})

        vx17 = base_objects.Vertex({'legs':base_objects.LegList([l1, l7, l17]), 'id': 3})
        vx12 = base_objects.Vertex({'legs':base_objects.LegList([l17, l2, l12]), 'id': 3})
        vx68 = base_objects.Vertex({'legs':base_objects.LegList([l6, l8, l68]), 'id': 3})
        vx56 = base_objects.Vertex({'legs':base_objects.LegList([l5, l68, l56]), 'id': 3})
        vx34 = base_objects.Vertex({'legs':base_objects.LegList([l3, l4, l34]), 'id': 1})
        vx135 = base_objects.Vertex({'legs':base_objects.LegList([l12, l56, l34]), 'id': 3})

        myVertexList1=base_objects.VertexList([vx17,vx12,vx68,vx56,vx34,vx135])

        myPentaDiag1=loop_base_objects.LoopDiagram({'vertices':myVertexList1,'type':1})

        myStructRep=loop_base_objects.FDStructureList()
        myStruct=loop_base_objects.FDStructure()
        
        goal_tag=[[1, [0], 3], [1, [1], 3], [1, [2], 3], [1, [3], 3], [1, [4], 3]]
        myPentaDiag1.tag(myStructRep,7,8,self.myproc)
        self.assertEqual(myPentaDiag1.get('canonical_tag'), goal_tag)

        vx17_tag=base_objects.Vertex({'legs':base_objects.LegList([l1, l7, l17]), 'id': 3})
        vx12_tag=base_objects.Vertex({'legs':base_objects.LegList([l2, l17, l12]), 'id': 3})
        vx13_tag=base_objects.Vertex({'legs':base_objects.LegList([l34, l12, l17]), 'id': 3})
        vx15_tag=base_objects.Vertex({'legs':base_objects.LegList([l5, l17, l17]), 'id': 3})
        vx168_tag=base_objects.Vertex({'legs':base_objects.LegList([l6, l17, l8]), 'id': 3})        
        goal_vertices=base_objects.VertexList([vx17_tag,vx12_tag,vx13_tag,vx15_tag,vx168_tag])
        self.assertEqual(myPentaDiag1.get('vertices'), goal_vertices)
