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
   loop_helas_objects.py"""

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
import madgraph.core.drawing as draw_lib
import madgraph.iolibs.drawing_eps as draw
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.loop.loop_helas_objects as loop_helas_objects
import madgraph.iolibs.save_load_object as save_load_object
from madgraph import MadGraph5Error

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')

#===============================================================================
# LoopDiagramGeneration Test
#===============================================================================
class LoopHelasMatrixElementTest(unittest.TestCase):
    """Test class for all functions related to the LoopHelasMatrixElement"""
    
    myloopmodel = loop_base_objects.LoopModel()
    
    def setUp(self):
        """load the NLO toy model"""
        
        self.myloopmodel = looptest.loadLoopModel() 

    def test_helas_diagrams_epemddx(self):
        """Test the generation of the helas diagrams for the process e+e->dd~
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
                                        'orders':{},
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myproc)
        myloopamplitude.generate_diagrams()
        
        ### First let's try the born diagram which should be exactly as in a
        ### HelasMatrixElement. For that we put only one born diagrams in the
        ### amplitude diagrams.
        myloopamplitude.set('diagrams', \
          base_objects.DiagramList([myloopamplitude['born_diagrams'][0]]))
        myloopME=loop_helas_objects.LoopHelasMatrixElement(myloopamplitude)
        # myloopME=loop_helas_objects.LoopHelasMatrixElement(myloopamplitude)
