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
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))


import tests.unit_tests as unittest


import madgraph.core.drawing as draw_lib
import madgraph.iolibs.drawing_eps as draw
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.iolibs.save_load_object as save_load_object
from madgraph import MadGraph5Error

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')

#===============================================================================
# Function to load a toy hardcoded Loop Model
#===============================================================================

def loadLoopModel():
    """Setup the NLO model"""
    
    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    myloopmodel = loop_base_objects.LoopModel()

    # A gluon
    mypartlist.append(base_objects.Particle({'name':'g',
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
    mypartlist.append(base_objects.Particle({'name':'u',
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
    antiu = copy.copy(mypartlist[1])
    antiu.set('is_part', False)

    # A quark D and its antiparticle
    mypartlist.append(base_objects.Particle({'name':'d',
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
    antid = copy.copy(mypartlist[2])
    antid.set('is_part', False)

    # A photon
    mypartlist.append(base_objects.Particle({'name':'a',
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
    mypartlist.append(base_objects.Particle({'name':'e-',
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
    antie = copy.copy(mypartlist[4])
    antie.set('is_part', False)

    # First set up the base interactions.

    # 3 gluon vertex
    myinterlist.append(base_objects.Interaction({
                  'id': 1,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 3),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':1}}))

    # 4 gluon vertex
    myinterlist.append(base_objects.Interaction({
                  'id': 2,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 4),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G^2'},
                  'orders':{'QCD':2}}))

    # Gluon and photon couplings to quarks
    myinterlist.append(base_objects.Interaction({
                  'id': 3,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':1}}))

    myinterlist.append(base_objects.Interaction({
                  'id': 4,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[3]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':1}}))

    myinterlist.append(base_objects.Interaction({
                  'id': 5,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':1}}))

    myinterlist.append(base_objects.Interaction({
                  'id': 6,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[3]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':1}}))

    # Coupling of e to gamma

    myinterlist.append(base_objects.Interaction({
                  'id': 7,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[4], \
                                         antie, \
                                         mypartlist[3]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':1}}))

    # Then set up the R2 interactions proportional to those existing in the
    # tree-level model.

    # 3 gluon vertex
    myinterlist.append(base_objects.Interaction({
                  'id': 8,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 3),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':3},
                  'type':['R2',()]}))

    # 4 gluon vertex
    myinterlist.append(base_objects.Interaction({
                  'id': 9,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 4),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G^2'},
                  'orders':{'QCD':4},
                  'type':['R2',()]}))

    # Gluon and photon couplings to quarks
    myinterlist.append(base_objects.Interaction({
                  'id': 10,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':3},
                  'type':['R2',()]}))
        
    myinterlist.append(base_objects.Interaction({
                  'id': 11,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':1, 'QED':2},
                  'type':['R2',()]}))

    myinterlist.append(base_objects.Interaction({
                  'id': 12,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[3]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':1, 'QCD':2},
                  'type':['R2',()]}))

    myinterlist.append(base_objects.Interaction({
                  'id': 13,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[3]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':3},
                  'type':['R2',()]}))

    myinterlist.append(base_objects.Interaction({
                  'id': 14,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':3},
                  'type':['R2',()]}))

    myinterlist.append(base_objects.Interaction({
                  'id': 15,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':1, 'QED':2},
                  'type':['R2',()]}))

    myinterlist.append(base_objects.Interaction({
                  'id': 16,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[3]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':1, 'QCD':2},
                  'type':['R2',()]}))

    myinterlist.append(base_objects.Interaction({
                  'id': 17,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[3]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':3},
                  'type':['R2',()]}))

    # Coupling of e to gamma

    myinterlist.append(base_objects.Interaction({
                  'id': 18,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[4], \
                                         antie, \
                                         mypartlist[3]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':3},
                  'type':['R2',()]}))

    # R2 interactions not proportional to the base interactions

    # Two point interactions

    # The gluon
    myinterlist.append(base_objects.Interaction({
                  'id': 19,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 2),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':2},
                  'type':['R2',()]}))

    # The photon
    myinterlist.append(base_objects.Interaction({
                  'id': 20,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3]] * 2),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2},
                  'type':['R2',()]}))

    # The electron
    myinterlist.append(base_objects.Interaction({
                  'id': 21,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[4], \
                                         antie]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2},
                  'type':['R2',()]}))

    # The up quark, R2QED
    myinterlist.append(base_objects.Interaction({
                  'id': 22,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[2], \
                                         antid]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2},
                  'type':['R2',()]}))

    # The up quark, R2QCD
    myinterlist.append(base_objects.Interaction({
                  'id': 23,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[2], \
                                         antid]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':2},
                  'type':['R2',()]}))

    # The down quark, R2QED
    myinterlist.append(base_objects.Interaction({
                  'id': 24,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[1], \
                                         antiu]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2},
                  'type':['R2',()]}))

    # The down quark, R2QCD
    myinterlist.append(base_objects.Interaction({
                  'id': 25,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[1], \
                                         antid]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':2},
                  'type':['R2',()]}))

    # The R2 three and four point interactions not proportional to the
    # base interaction

    # 3 photons
    myinterlist.append(base_objects.Interaction({
                  'id': 26,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3]] * 3),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':3},
                  'type':['R2',()]}))

    # 2 photon and 1 gluons
    myinterlist.append(base_objects.Interaction({
                  'id': 27,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3],\
                                        mypartlist[3],\
                                        mypartlist[0],]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2, 'QCD':1},
                  'type':['R2',()]}))

    # 1 photon and 2 gluons
    myinterlist.append(base_objects.Interaction({
                  'id': 28,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3],\
                                        mypartlist[0],\
                                        mypartlist[0],]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':1, 'QCD':2},
                  'type':['R2',()]}))

    # 4 photons
    myinterlist.append(base_objects.Interaction({
                  'id': 29,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3]] * 4),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':4},
                  'type':['R2',()]}))

    # 3 photons and 1 gluon
    myinterlist.append(base_objects.Interaction({
                  'id': 30,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3],\
                                        mypartlist[3],\
                                        mypartlist[3],\
                                        mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':3,'QCD':1},
                  'type':['R2',()]}))

    # 2 photons and 2 gluons
    myinterlist.append(base_objects.Interaction({
                  'id': 31,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3],\
                                        mypartlist[3],\
                                        mypartlist[0],\
                                        mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2,'QCD':2},
                  'type':['R2',()]}))

    # 1 photon and 3 gluons
    myinterlist.append(base_objects.Interaction({
                  'id': 32,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3],\
                                        mypartlist[0],\
                                        mypartlist[0],\
                                        mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':1,'QCD':3},
                  'type':['R2',()]})) 

    # Finally the UV interactions Counter-Terms

    # 3 gluon vertex CT
    myinterlist.append(base_objects.Interaction({
                  'id': 33,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 3),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':3},
                  'type':['UV1eps',()]}))

    # 4 gluon vertex CT
    myinterlist.append(base_objects.Interaction({
                  'id': 34,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 4),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G^2'},
                  'orders':{'QCD':4},
                  'type':['UV1eps',()]}))

    # Gluon and photon couplings to quarks CT
    myinterlist.append(base_objects.Interaction({
                  'id': 35,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':3},
                  'type':['UV1eps',()]}))
    
    # this is the CT for the renormalization of the QED corrections to alpha_QCD
    myinterlist.append(base_objects.Interaction({
                  'id': 36,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':1, 'QED':2},
                  'type':['UV1eps',()]}))

    myinterlist.append(base_objects.Interaction({
                  'id': 37,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[3]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':1, 'QCD':2},
                  'type':['UV1eps',()]}))

    myinterlist.append(base_objects.Interaction({
                  'id': 38,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[3]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':3},
                  'type':['UV1eps',()]}))

    myinterlist.append(base_objects.Interaction({
                  'id': 39,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':3},
                  'type':['UV1eps',()]}))

    myinterlist.append(base_objects.Interaction({
                  'id': 40,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':1, 'QED':2},
                  'type':['UV1eps',()]}))

    myinterlist.append(base_objects.Interaction({
                  'id': 41,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[3]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':1, 'QCD':2},
                  'type':['UV1eps',()]}))

    myinterlist.append(base_objects.Interaction({
                  'id': 42,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[3]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':3},
                  'type':['UV1eps',()]}))
    
    # alpha_QED to electron CT

    myinterlist.append(base_objects.Interaction({
                  'id': 43,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[4], \
                                         antie, \
                                         mypartlist[3]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':3},
                  'type':['UV1eps',()]}))
      
    # Finally the mass renormalization of the up and down quark dotted with
    # a mass for the occasion

    # The up quark, UVQED
    myinterlist.append(base_objects.Interaction({
                  'id': 44,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[2], \
                                         antid]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2},
                  'type':['UV1eps',()]}))

    # The up quark, UVQCD
    myinterlist.append(base_objects.Interaction({
                  'id': 45,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[2], \
                                         antid]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':2},
                  'type':['UV1eps',()]}))

    # The down quark, UVQED
    myinterlist.append(base_objects.Interaction({
                  'id': 46,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[1], \
                                         antiu]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2},
                  'type':['UV1eps',()]}))

    # The down quark, UVQCD
    myinterlist.append(base_objects.Interaction({
                  'id': 47,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[1], \
                                         antid]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':2},
                  'type':['UV1eps',()]}))

    
    myloopmodel.set('particles', mypartlist)
    myloopmodel.set('couplings', ['QCD','QED'])        
    myloopmodel.set('interactions', myinterlist)
    myloopmodel.set('perturbation_couplings', ['QCD','QED'])
    myloopmodel.set('order_hierarchy', {'QCD':1,'QED':2})

    return myloopmodel

    # Save this model so that it can be loaded by other loop tests
    # save_load_object.save_to_file(os.path.join(_input_file_path, 'test_toyLoopModel.pkl'),self.myloopmodel)

#===============================================================================
# LoopDiagramGeneration Test
#===============================================================================

class LoopDiagramGenerationTest(unittest.TestCase):
    """Test class for all functions related to the Loop diagram generation"""

    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    myloopmodel = loop_base_objects.LoopModel()
    
    ref_dict_to0 = {}
    ref_dict_to1 = {}

    myamplitude = diagram_generation.Amplitude()

    def setUp(self):
        """Load different objects for the tests."""
        
        self.myloopmodel = loadLoopModel()
        self.mypartlist = self.myloopmodel['particles']
        self.myinterlist = self.myloopmodel['interactions']
        self.ref_dict_to0 = self.myinterlist.generate_ref_dict(['QCD','QED'])[0]
        self.ref_dict_to1 = self.myinterlist.generate_ref_dict(['QCD','QED'])[1]
        
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

        dummyproc.set("perturbation_couplings",['QCD','QED'
                                                ])
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

    def test_diagram_generation_uux_ga(self):
        """Test the number of loop diagrams generated for uu~>g gamma (s channel)
           with different choices for the perturbation couplings and squared orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        ordersChoices=[({},['QCD','QED'],{},37,12,10),
                       ({},['QCD'],{},15,5,4)]
        for (bornOrders,pert,sqOrders,nDiagGoal, nR2, nUV) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nDiagGoal)
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(sumR2,nR2)
            self.assertEqual(sumUV,nUV)                

            ### This is to plot the diagrams obtained
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/Users/Spooner/Documents/PhD/MG5/NLO', 'diagramsVall_' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(myloopamplitude['loop_diagrams'],
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)

    def test_diagram_generation_gg_ng(self):
        """Test the number of loop diagrams generated for gg>ng. n being in [1,2,3]
        """
        
        # For quick test 
        nGluons = [(1,8,1,1),(2,81,7,10)]
        # For a longer one
        # nGluons += [(3,905,65,105),(4,11850,755,1290)]

        for (n, nDiagGoal, nUVGoal, nR2Goal) in nGluons:
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
            sum=0
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sum+=len(diag.get('CT_vertices'))
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')), nDiagGoal)
            self.assertEqual(sumR2, nR2Goal)
            self.assertEqual(sumUV, nUVGoal)            
            self.assertEqual(sum, nUVGoal+nR2Goal)

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

        ordersChoices=[({},['QCD','QED'],{},21,7,6),\
                       ({},['QCD','QED',],{'QED':-99},28,10,8),\
                       ({},['QCD'],{},9,3,2),\
                       ({},['QED'],{},2,2,2),\
                       ({'QED':0},['QCD'],{},9,3,2),\
                       ({'QCD':0},['QED'],{},7,3,2),\
                       ({},['QCD','QED'],{'QED':-1},9,3,2),\
                       ({},['QCD','QED'],{'QCD':-1},7,3,2),\
                       ({},['QCD','QED'],{'QED':-2},21,7,6),\
                       ({},['QCD','QED'],{'QED':-3},28,10,8),\
                       # These last two are of no physics interest
                       # It is just for the sake of the test.
                       ({'QED':0},['QCD','QED'],{},21,7,6),\
                       ({'QCD':0},['QCD','QED'],{},19,7,6)]
        
        for (bornOrders,pert,sqOrders,nDiagGoal,nR2Goal,nUVGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')), nDiagGoal)
            self.assertEqual(sumR2, nR2Goal)
            self.assertEqual(sumUV, nUVGoal)            

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

        ordersChoices=[({},['QCD','QED'],{},42,14,12),\
                       ({},['QCD','QED',],{'QED':-99},56,20,16),\
                       ({},['QED'],{},4,4,4),\
                       ({},['QCD'],{},18,6,4),\
                       ({'QED':0},['QCD'],{},18,6,4),\
                       ({'QCD':0},['QED'],{},14,6,4),\
                       ({},['QCD','QED'],{'QED':-1},18,6,4),\
                       ({},['QCD','QED'],{'QCD':-1},14,6,4)]
        
        for (bornOrders,pert,sqOrders,nDiagGoal,nR2Goal,nUVGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nDiagGoal)
            self.assertEqual(sumR2, nR2Goal)
            self.assertEqual(sumUV, nUVGoal)  
            
    def test_CT_vertices_generation_gg_gg(self):
        """ test that the Counter Term vertices are correctly
            generated by adding some new CT interactions to the model and
            comparing how many CT vertices are generated on the 
            process gg_gg for different R2 specifications. """
            
        newLoopModel=copy.deepcopy(self.myloopmodel)
        newInteractionList=base_objects.InteractionList()
        for inter in newLoopModel['interactions']:
            if inter['type'][0]=='base':
                newInteractionList.append(inter)
        
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
                
        newInteractionList.append(base_objects.Interaction({
                      'id': 666,
                      # a dd~d~ R2
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]]*4),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':4},
                      # We don't specify the loop content here
                      'type':['R2',()]}))
        
        myproc = base_objects.Process({'legs':myleglist,
                                       'model':newLoopModel,
                                       'orders':{},
                                       'perturbation_couplings':['QCD'],
                                       'squared_orders':{'WEIGHTED':99}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myproc)
        myloopamplitude.generate_diagrams()
        
        CTChoice=[((),{'QCD':4},1),
                  ((21,21,21,21),{'QCD':4},1),
                  ((1,1,1,1),{'QCD':4},1),
                  ((2,2,2,2),{'QCD':4},1),
                  ((21,21,21,21),{'QCD':4,'QED':1},0),
                  ((1,1,2,2),{'QCD':4},0)]
        
        for (parts,orders,nCTGoal) in CTChoice:
            newInteractionList[-1]['type'][1]=parts
            newInteractionList[-1]['orders']=orders            
            newLoopModel.set('interactions',newInteractionList)
            myloopamplitude['process']['model']=newLoopModel
            for diag in myloopamplitude.get('loop_diagrams'):
                diag['CT_vertices']=base_objects.VertexList()
            myloopamplitude.setCT_vertices()            
            sumR2=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(newLoopModel,'R2'))
            self.assertEqual(sumR2, nCTGoal)

    def test_CT_vertices_generation_ddx_ddx(self):
        """ test that the Counter Term vertices are correctly
            generated by adding some new CT interactions to the model and
            comparing how many CT vertices are generated on the 
            process ddx_ddx for different R2 specifications. """
            
        newLoopModel=copy.deepcopy(self.myloopmodel)
        newInteractionList=base_objects.InteractionList()
        for inter in newLoopModel['interactions']:
            if inter['type'][0]=='base':
                newInteractionList.append(inter)
        
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))
                
        antid=copy.copy(self.mypartlist[2])
        antid.set('is_part',False)
        newInteractionList.append(base_objects.Interaction({
                      'id': 666,
                      # a dd~d~ R2
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2],
                                             antid,
                                             self.mypartlist[2],
                                             antid,]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':4},
                      # We don't specify the loop content here
                      'type':['R2',()]}))
        
        myproc = base_objects.Process({'legs':myleglist,
                                       'model':newLoopModel,
                                       'orders':{},
                                       'perturbation_couplings':['QCD','QED'],
                                       'squared_orders':{'WEIGHTED':99}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myproc)
        myloopamplitude.generate_diagrams()
        
        CTChoice=[((),{'QCD':4},1),
                  ((1,21,1,21),{'QCD':4},1),
                  ((1,22,1,22),{'QED':4},1),
                  ((22,22,1,1),{'QED':4},1),
                  ((1,21,1,21),{'QED':4},0),
                  ((22,22,1,1),{'QCD':4},0)]
        
        for (parts,orders,nCTGoal) in CTChoice:
            newInteractionList[-1]['type'][1]=parts
            newInteractionList[-1]['orders']=orders            
            newLoopModel.set('interactions',newInteractionList)
            myloopamplitude['process']['model']=newLoopModel
            for diag in myloopamplitude.get('loop_diagrams'):
                diag['CT_vertices']=base_objects.VertexList()
            myloopamplitude.setCT_vertices()            
            sumR2=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(newLoopModel,'R2'))
            self.assertEqual(sumR2, nCTGoal)
            
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
