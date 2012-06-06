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
import shutil

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
import aloha
import tests.unit_tests.various.test_aloha as test_aloha
import aloha.create_aloha as create_aloha

from madgraph import MadGraph5Error

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')
_mgme_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                os.path.pardir)
_loop_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                os.path.pardir, 'Template','loop_material')
_cuttools_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                os.path.pardir, 'vendor','CutTools')
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
                                  False, False, True, _loop_file_path,
                                  _cuttools_file_path)
    def setUp(self):
        """load the NLO toy model"""
        self.loadModel("UFO")
        
    def loadModel(self, mode="UFO"):
        """ Either loads the model from a UFO model in the input files
        (if mode is set to "UFO") or use the equivalent hardcoded one."""
        # Import it from the stone-graved smQCDNLO model in the input_files of
        # the test.
        if mode=="UFO":
            self.myloopmodel = models.import_full_model(os.path.join(\
                                                _input_file_path,'LoopSMTest'))
            self.fortran_model = helas_call_writers.FortranUFOHelasCallWriter(\
                                                            self.myloopmodel)
            return

    @test_aloha.set_global(loop=True, unitary=False, mp=True, cms=False) 
    def check_output_sanity(self, loopME):
        """ Test different characteristics of the output of the 
        LoopMatrixElement given in argument to check the correct behavior
        of the loop exporter"""

        # Cleaning last process directory
        if os.path.exists(_proc_file_path):
            shutil.rmtree(_proc_file_path)
        
        self.loopExporter.copy_v4template(self.myloopmodel.get('name'))
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
        proc_name='P'+loopME.get('processes')[0].shell_string()
        files=['mpmodule.mod','libcts.a']
        for file in files:
            self.assertTrue(os.path.exists(os.path.join(_proc_file_path\
                                              ,'lib',file)))
        files=['param_card.dat']
        for file in files:
            self.assertTrue(os.path.exists(os.path.join(_proc_file_path\
                                              ,'Cards',file)))
        files=['DHELAS','MODEL','coupl.inc','make_opts','makefile']
        for file in files:
            self.assertTrue(os.path.exists(os.path.join(_proc_file_path\
                                              ,'Source',file)))
        files=[proc_name,'coupl.inc','cts_mprec.h','makefile',\
               'check_sa.f','cts_mpc.h']
        for file in files:
            self.assertTrue(os.path.exists(os.path.join(_proc_file_path\
                                              ,'SubProcesses',file)))    
        
        files=['CT_interface.f','check_sa.f','cts_mprec.h','loop_num.f',
               'nexternal.inc','born_matrix.f','coupl.inc',
               'makefile','ngraphs.inc','born_matrix.ps',
               'cts_mpc.h','loop_matrix.f','mpmodule.mod','pmass.inc']
#        files.append('loop_matrix.ps')
        for file in files:
            self.assertTrue(os.path.exists(os.path.join(_proc_file_path\
                             ,'SubProcesses',proc_name,file)))
    
    def test_aloha_loop_HELAS_subroutines(self):
        """ Test that Aloha correctly processes loop HELAS subroutines. """
        
        aloha_model = create_aloha.AbstractALOHAModel(self.myloopmodel.get('name'))
        target_lorentz=[(('R2_QQ_1',), (), 0), (('FFV1',), ('L',), 1)]
        target_lorentz+=[(('FFV1',), ('L',), 0), (('VVVV3',), ('L',), 0)]
        target_lorentz+=[(('R2_GG_1', 'R2_GG_2'), (), 0), (('R2_GG_1', 'R2_GG_3'), (), 0)]
        target_lorentz+=[(('FFV1',), ('L',), 3), (('VVVV3',), ('L',), 1), (('VVVV4',), ('L',), 0)]
        target_lorentz+=[(('VVV1',), (), 0), (('GHGHG',), ('L',), 0), (('VVV1',), (), 1)]
        target_lorentz+=[(('GHGHG',), ('L',), 1),(('VVVV1',), ('L',), 0), (('VVVV1',), ('L',), 1)]
        target_lorentz+=[(('VVV1',), ('L',), 1), (('VVV1',), ('L',), 0),(('VVV1',), ('L',), 0)]
        target_lorentz+=[(('FFV1',), (), 2), (('FFV1',), (), 3),(('FFV1',), (), 0)]
        target_lorentz+=[(('FFV1',), (), 1), (('VVVV4',), ('L',), 1), (('R2_GG_1',), (), 0)]
        aloha_model.compute_subset(target_lorentz)
        
        for list_l_name, tag, outgoing in target_lorentz:
            entry_tuple=(list_l_name[0]+''.join(tag),outgoing)
            self.assertTrue(entry_tuple in aloha_model.keys())
            AbstractRoutine=aloha_model[entry_tuple]
            self.assertEqual(list(tag),AbstractRoutine.tag)
        
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
                                        'forbidden_particles':[],
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
        
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        myloopME=loop_helas_objects.LoopHelasMatrixElement(myloopamplitude)
        self.check_output_sanity(myloopME)
        
        subproc_file_path=os.path.join(_proc_file_path,'Subprocesses',\
                                       'P0_ddx_uux')
        
        #        shutil.rmtree(_proc_file_path)

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

    def test_LoopProcessExporterFortranSA_ddx_ttx(self):
        """Test the StandAlone output for different processes.
        """
        
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-6,
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

    def test_LoopProcessExporterFortranSA_gg_ttx(self):
        """Test the StandAlone output for different processes.
        """
        
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-6,
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

    def notest_LoopProcessExporterFortranSA_gg_gddx(self):
        """Test the StandAlone output for different processes.
        """
        
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
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

    def test_LoopProcessExporterFortranSA_ddx_gg(self):
        """Test the StandAlone output for different processes.
        """
        
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
                        
        myloopproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'forbidden_particles':[],
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        myloopME=loop_helas_objects.LoopHelasMatrixElement(myloopamplitude)
        self.check_output_sanity(myloopME)
        # Further Check that the right ALOHA subroutines are created
        HELAS_files=['aloha_file.inc', 'aloha_functions.f',
                     'FFV1_0.f', 'FFV1_1.f', 'FFV1_2.f', 'FFV1_3.f', 'FFV1L_1.f',
                     'FFV1L_2.f', 'FFV1L_3.f', 'GHGHGL_1.f', 'GHGHGL_2.f', 'makefile',
                     'MP_FFV1_0.f', 'MP_FFV1_1.f', 'MP_FFV1_2.f', 'MP_FFV1_3.f',
                     'MP_FFV1L_1.f', 'MP_FFV1L_2.f', 'MP_FFV1L_3.f', 'MP_GHGHGL_1.f',
                     'MP_GHGHGL_2.f', 'MP_R2_GG_1_0.f', 'MP_R2_GG_1_R2_GG_2_0.f',
                     'MP_R2_GG_1_R2_GG_3_0.f', 'MP_R2_GG_2_0.f', 'MP_R2_GG_3_0.f',
                     'MP_R2_QQ_1_0.f', 'MP_VVV1_0.f', 'MP_VVV1_1.f', 'MP_VVV1L_1.f',
                     'MP_VVVV1L_1.f', 'MP_VVVV3L_1.f', 'MP_VVVV4L_1.f', 'R2_GG_1_0.f',
                     'R2_GG_1_R2_GG_2_0.f', 'R2_GG_1_R2_GG_3_0.f',
                     'R2_GG_2_0.f', 'R2_GG_3_0.f', 'R2_QQ_1_0.f', 'VVV1_0.f',
                     'VVV1_1.f', 'VVV1L_1.f', 'VVVV1L_1.f', 'VVVV3L_1.f', 'VVVV4L_1.f']
        for hFile in HELAS_files:
            self.assertTrue(os.path.exists(os.path.join(_proc_file_path\
                                              ,'Source','DHELAS',hFile)))        
        
    def notest_LoopProcessExporterFortranSA_dg_dg(self):
        """Test the StandAlone output for different processes.
        """
        
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
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

    def notest_LoopProcessExporterFortranSA_ddx_uuxddx(self):
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

    def notest_LoopProcessExporterFortranSA_ddx_ddxddx(self):
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
        
    def test_LoopProcessExporterFortranSA_gg_gg(self):
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
                                        'forbidden_particles':[3,4,5,6],
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
        
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        myloopME=loop_helas_objects.LoopHelasMatrixElement(myloopamplitude)
        self.check_output_sanity(myloopME)

    def notest_LoopProcessExporterFortranSA_gg_ggg(self):
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
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))                     
        myloopproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'forbidden_particles':[3,4,5,6],
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        myloopME=loop_helas_objects.LoopHelasMatrixElement(myloopamplitude)
        self.check_output_sanity(myloopME)
