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
import logging
import math
import os
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import tests.unit_tests.loop.test_loop_diagram_generation as looptest
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import madgraph.core.color_amp as color_amp
import madgraph.loop.loop_color_amp as loop_color_amp
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.loop.loop_helas_objects as loop_helas_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.loop.loop_exporters as loop_exporters
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.save_load_object as save_load_object
import madgraph.iolibs.helas_call_writers as helas_call_writers
import models.import_ufo as models
from madgraph import MadGraph5Error

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')
_mgme_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                os.path.pardir)
_loop_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                os.path.pardir, 'loop_material')
_proc_file_path = os.path.join(_file_path, 'test_proc')


#===============================================================================
# LoopExporterTest Test
#===============================================================================
class LoopExporterTest(unittest.TestCase):
    """Test class for all functions related to the Loop exporters."""
    
    myloopmodel = loop_base_objects.LoopModel()
    fortran_model= helas_call_writers.FortranUFOHelasCallWriter()
    loopExporter = loop_exporters.LoopProcessExporterFortranSA(\
                                  _mgme_file_path, _proc_file_path,
                                  False, _loop_file_path)
    def setUp(self):
        """load the NLO toy model"""
        
        self.myloopmodel = models.import_full_model(os.path.join(\
            _input_file_path,'smQCDNLO'))
        self.fortran_model = helas_call_writers.FortranUFOHelasCallWriter(\
                                                            self.myloopmodel)
        
    def check_output_sanity(self, loopME):
        """ Test different characteristics of the output of the 
        LoopMatrixElement given in argument to check the correct behavior
        of the loop exporter"""
        
        self.loopExporter.copy_v4template()
        self.assertTrue(os.path.isdir(os.path.join(_proc_file_path,'Source',\
                                              'CutTools')))
        self.loopExporter.generate_loop_subprocess(\
                            loopME, self.fortran_model)
        wanted_lorentz = loopME.get_used_lorentz()
        wanted_couplings = list(set(sum(loopME.get_used_couplings(),[])))
        self.loopExporter.convert_model_to_mg4(self.myloopmodel,
                                           wanted_lorentz,
                                           wanted_couplings)
        self.loopExporter.finalize_v4_directory( \
                        helas_objects.HelasMatrixElementList([loopME,]),
                        ["Generation from test_loop_exporters.py",],
                        False,False)
        
    def test_LoopProcessExporterFortranSA_ddx_uux(self):
        """Test the StandAlone output for different processes.
        """
        
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':True}))
                
        myloopproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        myloopME=loop_helas_objects.LoopHelasMatrixElement(myloopamplitude)
        self.check_output_sanity(myloopME)

    def test_LoopProcessExporterFortranSA_ddx_ddx(self):
        """Test the StandAlone output for different processes.
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
                        
        myloopproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        myloopME=loop_helas_objects.LoopHelasMatrixElement(myloopamplitude)
        self.check_output_sanity(myloopME)

    def test_LoopProcessExporterFortranSA_ddx_ddxddx(self):
        """Test the StandAlone output for different processes.
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
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))
                        
        myloopproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        myloopME=loop_helas_objects.LoopHelasMatrixElement(myloopamplitude)
        self.check_output_sanity(myloopME)
        
    def notest_LoopProcessExporterFortranSA_gg_gg(self):
        """Test the StandAlone output for different processes.
        """
        
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
                
        myloopproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        myloopME=loop_helas_objects.LoopHelasMatrixElement(myloopamplitude)
        self.check_output_sanity(myloopME)
