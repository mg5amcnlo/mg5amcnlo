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
import madgraph.various.misc as misc
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
                  #'texname':'G',
                  #'antitexname':'G',
                  'line':'curly',
                  'charge':0.,
                  'pdg_code':21,
                  #'propagating':True,
                  'propagator':0,                                             
                  'is_part':True,
                  'counterterm':{('QCD', ((6,),)): {0: 'UVWfct_G_1', -1: 'UVWfct_G_1_1eps'}, ('QCD', ((5,),)): {0: 'UVWfct_G_0', -1: 'UVWfct_G_0_1eps'}},
                  'self_antipart':True}))
    
    # A quark U and its antiparticle
    mypartlist.append(base_objects.Particle({'name':'u',
                  'antiname':'u~',
                  'spin':2,
                  'color':3,
                  'mass':'ZERO',
                  'width':'ZERO',
                  #'texname':'u',
                  #'antitexname':'u',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':2,
                  #'propagating':True,
                  'propagator':'',                                             
                  'is_part':True,
                  'self_antipart':False}))
    antiu = copy.copy(mypartlist[1])
    antiu.set('is_part', False)

    # A quark D and its antiparticle
    mypartlist.append(base_objects.Particle({'name':'d',
                  'antiname':'d~',
                  'spin':2,
                  'color':3,
                  'mass':'ZERO',
                  'width':'ZERO',
                  #'texname':'d',
                  #'antitexname':'d',
                  'line':'straight',
                  'charge':-1. / 3.,
                  'pdg_code':1,
                  #'propagating':True,
                  'propagator':'',
                  'is_part':True,
                  'self_antipart':False}))
    antid = copy.copy(mypartlist[2])
    antid.set('is_part', False)

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
#        self.imported_loopmodel = models.import_full_model(os.path.join(\
#            _input_file_path,'loop_ToyModel'))
        self.imported_loopmodel = models.import_full_model(os.path.join(\
            _input_file_path,'LoopSMTest'))
        self.imported_loopmodel.actualize_dictionaries()
        self.hardcoded_loopmodel.actualize_dictionaries()
    
    def test_loadingLoopToyModel(self):
        """ Several test on the correctness of the model imported.
        Initially the idea was to compare against the hardcoded model but it
        is too tidious to copy it down. So only a few characteristics are tested
        """
        
        self.assertEqual(self.imported_loopmodel['perturbation_couplings'],\
                         ['QCD',])
        self.assertEqual(len(self.imported_loopmodel.get('interactions')),205)
        self.assertEqual(len(self.imported_loopmodel.get('interactions').\
                             get_type('base')),71)
        self.assertEqual(len(self.imported_loopmodel.get('interactions').\
                             get_R2()),78)
        self.assertEqual(len(self.imported_loopmodel.get('interactions').\
                             get_UV()),56)
        self.assertEqual(len(self.imported_loopmodel.get('interactions').\
                             get_UVmass()),4)
        self.assertEqual(self.imported_loopmodel.get('name'),'LoopSMTest')
        self.assertEqual(self.imported_loopmodel.get('order_hierarchy'),\
                         {'QCD':1,'QED':2})
        self.assertEqual(self.imported_loopmodel.get('coupling_orders'),\
                         set(['QCD','QED']))
        # The up quark
        for key in self.hardcoded_loopmodel['particles'][2].keys():
            self.assertEqual(self.imported_loopmodel['particles'][0][key],\
                         self.hardcoded_loopmodel['particles'][2][key])
        # The down quark
        for key in self.hardcoded_loopmodel['particles'][1].keys():
            self.assertEqual(self.imported_loopmodel['particles'][1][key],\
                         self.hardcoded_loopmodel['particles'][1][key])
        # The gluon
        for key in self.hardcoded_loopmodel['particles'][0].keys():
            self.assertEqual(self.imported_loopmodel['particles'][6][key],\
                         self.hardcoded_loopmodel['particles'][0][key])
        
