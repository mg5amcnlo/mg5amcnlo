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

"""Unit test library to check that Loop UFO models are correctly imported """

import copy
import itertools
import logging
import math
import os
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import tests.unit_tests.loop.test_loop_diagram_generation as looptest
import madgraph.core.base_objects as base_objects
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.iolibs.save_load_object as save_load_object
import models.import_ufo as models
from madgraph import MadGraph5Error

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')

#===============================================================================
# Function to load a toy hardcoded Loop QCD Model with d, dx, u, ux, and g.
#===============================================================================

def loadLoopToyModel():
    """Setup the Loop Toy QCD model with d,dx,u,ux and the gluon"""
    
    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    myloopmodel = loop_base_objects.LoopModel()

    # A gluon
    mypartlist.append(base_objects.Particle({'name':'g',
                  'antiname':'g',
                  'spin':3,
                  'color':8,
                  'mass':'ZERO',
                  'width':'ZERO',
                  'texname':'G',
                  'antitexname':'G',
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
                  'mass':'MU',
                  'width':'ZERO',
                  'texname':'u',
                  'antitexname':'u',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':2,
                  'propagating':True,
                  'is_part':True,
                  'perturbation':['QCD'],
                  'self_antipart':False}))
    antiu = copy.copy(mypartlist[1])
    antiu.set('is_part', False)

    # A quark D and its antiparticle
    mypartlist.append(base_objects.Particle({'name':'d',
                  'antiname':'d~',
                  'spin':2,
                  'color':3,
                  'mass':'MD',
                  'width':'ZERO',
                  'texname':'d',
                  'antitexname':'d',
                  'line':'straight',
                  'charge':-1. / 3.,
                  'pdg_code':1,
                  'propagating':True,
                  'is_part':True,
                  'perturbation':['QCD'],           
                  'self_antipart':False}))
    antid = copy.copy(mypartlist[2])
    antid.set('is_part', False)
    
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

    # Gluon coupling to u quark
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

    # Gluon coupling to d quark
    myinterlist.append(base_objects.Interaction({
                  'id': 4,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':1}}))

    # Then set up the R2 interactions proportional to those existing in the
    # tree-level model.
    
    # ggg R2
    myinterlist.append(base_objects.Interaction({
                  'id': 5,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 3),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':3},
                  'type':['R2',()]}))

    # gggg R2
    myinterlist.append(base_objects.Interaction({
                  'id': 6,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 4),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G^2'},
                  'orders':{'QCD':4},
                  'type':['R2',()]}))

    # guu R2
    myinterlist.append(base_objects.Interaction({
                  'id': 7,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':3},
                  'type':['R2',()]}))
    # gdd R2
    myinterlist.append(base_objects.Interaction({
                  'id': 8,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':3},
                  'type':['R2',()]}))

    # R2 interactions not proportional to the base interactions

    # Two point interactions

    # gg R2
    myinterlist.append(base_objects.Interaction({
                  'id': 9,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 2),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':2},
                  'type':['R2',()]}))

    # uu~ R2
    myinterlist.append(base_objects.Interaction({
                  'id': 10,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[2], \
                                         antid]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':2},
                  'type':['R2',()]}))

    # dd~ R2
    myinterlist.append(base_objects.Interaction({
                  'id': 11,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[1], \
                                         antid]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':2},
                  'type':['R2',()]}))


    # Finally the UV interactions Counter-Terms

    # ggg UV
    myinterlist.append(base_objects.Interaction({
                  'id': 12,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 3),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':3},
                  'type':['UV',()]}))

    # gggg UV
    myinterlist.append(base_objects.Interaction({
                  'id': 13,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 4),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G^2'},
                  'orders':{'QCD':4},
                  'type':['UV',()]}))

    # guu~ UV
    myinterlist.append(base_objects.Interaction({
                  'id': 14,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':3},
                  'type':['UV',()]}))
    
    # gdd~ UV
    myinterlist.append(base_objects.Interaction({
                  'id': 15,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[0]]),
                  'color': [],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':3},
                  'type':['UV',()]}))

   # We take the u and d-quark as massless so that there is no
   # UV 2-point interactions for them.

    myloopmodel.set('particles', mypartlist)
    myloopmodel.set('couplings', ['QCD'])        
    myloopmodel.set('interactions', myinterlist)
    myloopmodel.set('perturbation_couplings', ['QCD'])
    myloopmodel.set('order_hierarchy', {'QCD':1})

    return myloopmodel

#===============================================================================
# LoopUFOImport Test
#===============================================================================

class LoopUFOImportTest(unittest.TestCase):
    """Test class to check that the import of the loop UFO model is correct."""
    
    hardcoded_loopmodel = loop_base_objects.LoopModel()
    imported_loopmodel = loop_base_objects.LoopModel()
    
    def setUp(self):
        """load the hardcoded NLO toy model to compare against the imported 
            one"""
        
        self.hardcoded_loopmodel = loadLoopToyModel() 
        # Make sure to move the pickle first in order not to load it
        if os.path.exists(os.path.join(\
            _input_file_path,'loop_ToyModel','model.pkl')):
            os.system("mv -f "+str(os.path.join(\
                _input_file_path,'loop_ToyModel','model.pkl'))+" "+\
                str(os.path.join(_input_file_path,'loop_ToyModel',\
                'model_old.pkl')))
        self.imported_loopmodel = models.import_full_model(os.path.join(\
            _input_file_path,'loop_ToyModel'))
        self.imported_loopmodel.actualize_dictionaries()
        self.hardcoded_loopmodel.actualize_dictionaries()
    
    def test_loadingLoopToyModel(self):
        """ Several test on the correctness of the model imported.
        Initially the idea was to compare against the hardcoded model but it
        is too tidious to copy it down. So only a few characteristics are tested
        """
        
        self.assertEqual(self.imported_loopmodel['perturbation_couplings'],\
                         ['QCD',])
        self.assertEqual(len(self.imported_loopmodel.get('interactions')),21)
        self.assertEqual(self.imported_loopmodel.get('name'),'loop_ToyModel')
        self.assertEqual(self.imported_loopmodel.get('order_hierarchy'),\
                         {'QCD':1})
        self.assertEqual(self.imported_loopmodel.get('coupling_orders'),\
                         set(['QCD']))
        # The up quark
        for key in self.hardcoded_loopmodel['particles'][1].keys():
            self.assertEqual(self.imported_loopmodel['particles'][0][key],\
                         self.hardcoded_loopmodel['particles'][1][key])
        # The down quark
        for key in self.hardcoded_loopmodel['particles'][2].keys():
            self.assertEqual(self.imported_loopmodel['particles'][1][key],\
                         self.hardcoded_loopmodel['particles'][2][key])
        # The gluon
        for key in self.hardcoded_loopmodel['particles'][0].keys():
            self.assertEqual(self.imported_loopmodel['particles'][2][key],\
                         self.hardcoded_loopmodel['particles'][0][key])
        self.assertEqual(len(self.imported_loopmodel['ref_dict_to0']),\
                         len(self.hardcoded_loopmodel['ref_dict_to0']))
        self.assertEqual(len(self.imported_loopmodel['ref_dict_to1']),\
                         len(self.hardcoded_loopmodel['ref_dict_to1']))
                