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
import re
import glob
import tarfile
import datetime

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
import models.import_ufo as import_ufo
import madgraph.various.misc as misc
import madgraph.various.process_checks as process_checks
import aloha
import tests.unit_tests.various.test_aloha as test_aloha
import aloha.create_aloha as create_aloha
from tests.unit_tests.various.test_aloha import set_global
import tests.unit_tests.iolibs.test_file_writers as test_file_writers

from madgraph.iolibs.files import cp, ln, mv
from madgraph import MadGraph5Error

pjoin = os.path.join
path = os.path

_file_path = os.path.dirname(os.path.realpath(__file__))

_input_file_path = os.path.abspath(os.path.join(_file_path, \
                                  os.path.pardir, os.path.pardir,'input_files'))
_mgme_file_path = os.path.abspath(os.path.join(_file_path, *([os.path.pardir]*3)))
_loop_file_path = os.path.join(_mgme_file_path,'Template','loop_material')
_cuttools_file_path = os.path.join(_mgme_file_path, 'vendor','CutTools')
_proc_file_path = os.path.join(_mgme_file_path, 'UNITTEST_proc')

# The folder below contains the files to which the exporter outputs are contained.
# You can regenerate them by running the __main__ of this module if you believe
# that one of them is outdated and want to automatically update it.
_hc_comparison_files = os.path.join(_input_file_path,'LoopExporterTestComparison')
_hc_comparison_tarball = os.path.join(_input_file_path,'LoopExporterTest.tar.bz2')
_hc_comparison_modif_log = os.path.join(_input_file_path,'RefFilesModifs.log')

#===============================================================================
# LoopExporterTest Test
#===============================================================================
class LoopExporterTest(unittest.TestCase):
    """Test class for all functions related to the Loop exporters."""

    def setUp(self):
        """load the models and exporters if necessary."""
        if not hasattr(self, 'mymodel') or \
           not hasattr(self, 'fortran_model') or \
           not hasattr(self, 'loopExporter') or \
           not hasattr(self, 'loopOptimizedExporter'):
            self.mymodel = import_ufo.import_model('loop_sm')
            self.fortran_model = helas_call_writers.FortranUFOHelasCallWriter(\
                                                               self.myloopmodel)
            self.loopExporter = loop_exporters.LoopProcessExporterFortranSA(\
                                  _mgme_file_path, _proc_file_path,
                                  {'clean':False, 'complex_mass':False, 
                                   'export_format':'madloop','mp':True,
                                   'loop_dir':_loop_file_path,
                                   'cuttools_dir':_cuttools_file_path,
                                   'fortran_compiler':'gfortran'})
    
            self.loopOptimizedExporter = loop_exporters.\
                                  LoopProcessOptimizedExporterFortranSA(\
                                  _mgme_file_path, _proc_file_path,
                                  {'clean':False, 'complex_mass':False, 
                                   'export_format':'madloop','mp':True,
                                   'loop_dir':_loop_file_path,
                                   'cuttools_dir':_cuttools_file_path,
                                   'fortran_compiler':'gfortran'})

    @test_aloha.set_global(loop=True, unitary=False, mp=True, cms=False) 
    def check_output_sanity(self, loopME, chosenLoopExporter=None):
        """ Test different characteristics of the output of the 
        LoopMatrixElement given in argument to check the correct behavior
        of the loop exporter"""

        if chosenLoopExporter==None:
            exporter=self.loopExporter
        else:
            exporter=chosenLoopExporter

        # Cleaning last process directory
        if os.path.exists(_proc_file_path):
            shutil.rmtree(_proc_file_path)
        
        exporter.copy_v4template(self.myloopmodel.get('name'))
        exporter.generate_loop_subprocess(\
                            loopME, self.fortran_model)
        wanted_lorentz = loopME.get_used_lorentz()
        wanted_couplings = list(set(sum(loopME.get_used_couplings(),[])))
        exporter.convert_model_to_mg4(self.myloopmodel,
                                           wanted_lorentz,
                                           wanted_couplings)
        exporter.finalize_v4_directory( \
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
                     ,'SubProcesses',file)),'File %s not created.'%file)    
        
        files=['CT_interface.f','check_sa.f','cts_mprec.h','loop_num.f',
               'nexternal.inc','born_matrix.f','coupl.inc',
               'makefile','ngraphs.inc','born_matrix.ps',
               'cts_mpc.h','loop_matrix.f','mpmodule.mod','pmass.inc']
        files.append('loop_matrix.ps')
        for file in files:
            self.assertTrue(os.path.exists(os.path.join(_proc_file_path\
                             ,'SubProcesses',proc_name,file)))

        # Make sure it initializes fine
        n_points = process_checks.LoopMatrixElementTimer.run_initialization( \
               run_dir = os.path.join(_proc_file_path,'SubProcesses',proc_name))
        self.assertTrue(not n_points is None)
    
    @test_aloha.set_global()
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
        myloopME=loop_helas_objects.LoopHelasMatrixElement(myloopamplitude,
                                                          optimized_output=True)
        self.check_output_sanity(myloopME,self.loopOptimizedExporter)

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
        myloopME=loop_helas_objects.LoopHelasMatrixElement(myloopamplitude,
                                                          optimized_output=True)
        self.check_output_sanity(myloopME,self.loopOptimizedExporter)

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
                        ,'Source','DHELAS',hFile)), 'file %s not found in %s' % 
                                                       (hFile, _proc_file_path))        
        
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

#===============================================================================
# IOExportMadLoopTest
#===============================================================================
class IOExportMadLoopTest(unittest.TestCase,
                     test_file_writers.CheckFileCreate):
    """Test class for the loop exporter modules. It uses hardcoded output 
    for the comparisons."""
    
    # To control the verbosity of the test
    verbose = False
    # To filter files which are checked, edit the tag ['ALL'] by the 
    # chosen filenames.
    # These filenames should be the path relative to the
    # position SubProcess/<P0_proc_name>/ in the output. Notice that you can
    # use the parent directory keyword ".." and instead of the filename you
    # can exploit the syntax [regexp] (brackets not part of the regexp)
    # Ex. ['../../Source/DHELAS/[.+\.(inc|f)]']
    filesChecked_filter = ['ALL']
    # To filter what tests you want to use, edit the tag ['ALL'] by the
    # list of test folders and names you want in.
    # You can prepend '-' to the folder or test name to veto it instead of
    # selecting it. Typically, ['-longTest'] considers all tests but the
    # longTest one (synthax not available for filenames)
    testFolders_filter = ['ALL']
    testNames_filter = ['ALL']  

    def applyFilter(self, testName=None, folderName=None):
        """ Returns True if the selected filter does not excluded the testName
        and folderName given in argument. Specify None to disregard the filters
        corresponding to this category."""
        
        if testName is None and folderName is None:
            return True
        
        if not testName is None:
            chosen = [f for f in self.testNames_filter if not f.startswith('-')]
            veto = [f[1:] for f in self.testNames_filter if f.startswith('-')]
            if testName in veto:
                return False
            if len(chosen)>0 and chosen!=['ALL'] and not testName in chosen:
                return False

        if not folderName is None:
            chosen = [f for f in self.testFolders_filter if not f.startswith('-')]
            veto = [f[1:] for f in self.testFolders_filter if f.startswith('-')]
            if folderName in veto:
                return False
            if len(chosen)>0 and chosen!=['ALL'] and not folderName in chosen:
                return False
        
        return True

    def toFileName(self, file_path):
        """ transforms a file specification like ../../Source/MODEL/myfile to
        %..%..%Source%MODEL%myfile """
        fpath = copy.copy(file_path)
        if not isinstance(fpath, str):
            fpath=str(fpath)
        if '/' not in fpath:
            return fpath
        
        return '%'+'%'.join(file_path.split('/'))
        
    def toFilePath(self, file_name):
        """ transforms a file name specification like %..%..%Source%MODEL%myfile
        to ../../Source/MODEL/myfile"""
        
        if not file_name.startswith('%'):
            return file_name
        
        return pjoin(file_name[1:].split('%'))

    def setUp(self):
        """load the models and exporters if necessary. Very similar to the one
        of LoopExporterTest, but I want to keep them separate."""

        # Extract the tarball for hardcoded comparison if necessary
        if not path.isdir(_hc_comparison_files):
            if path.isfile(_hc_comparison_tarball):
                tar = tarfile.open(_hc_comparison_tarball,mode='r:bz2')
                tar.extractall(path.dirname(_hc_comparison_files))
                tar.close()
            else:
                os.makedirs(_hc_comparison_files)
            
        if not hasattr(self, 'models') or \
           not hasattr(self, 'fortran_models') or \
           not hasattr(self, 'loop_exporters') or \
           not hasattr(self, 'my_procs'):
            self.models = { \
                'loop_sm' : import_ufo.import_model('loop_sm') 
                          }
            self.fortran_models = {
                'fortran_model' : helas_call_writers.FortranUFOHelasCallWriter(\
                                                         self.models['loop_sm']) 
                                  }
            
            self.loop_exporters = {
                'default' : loop_exporters.LoopProcessExporterFortranSA(\
                                  _mgme_file_path, _proc_file_path,
                                  {'clean':False, 'complex_mass':False, 
                                   'export_format':'madloop','mp':True,
                                   'loop_dir':_loop_file_path,
                                   'cuttools_dir':_cuttools_file_path,
                                   'fortran_compiler':'gfortran'}),
                'optimized' : loop_exporters.\
                                  LoopProcessOptimizedExporterFortranSA(\
                                  _mgme_file_path, _proc_file_path,
                                  {'clean':False, 'complex_mass':False, 
                                   'export_format':'madloop','mp':True,
                                   'loop_dir':_loop_file_path,
                                   'cuttools_dir':_cuttools_file_path,
                                   'fortran_compiler':'gfortran'})
                                  }
            
            # Notice that here the file should be the path relative to the
            # position SubProcess/<P0_proc_name>/ in the output.
            # You are allowed to use the parent directory specification ..
            # You can use the synthax [regexp] instead of a specific filename.
            # This includes only the files in this directory matching it.
            # Typically '../../Source/DHELAS/[.+\.(inc|f)]' matches any file
            # with the extension .f or .inc.
            # Notice that the squared brackets are not part of the reg expr. 
            
            # The version below is for the hardcoded version
            proc_files = ['CT_interface.f',
                            'ColorDenomFactors.dat',
                            'ColorNumFactors.dat',
                            'HelConfigs.dat',
                            'improve_ps.f',
                            'loop_matrix.f',
                            'loop_num.f',
                            'pmass.inc',
                            'nexternal.inc']
            # It's best to replace it with the regexp lookup
            proc_files = ['[.+\.(f|dat|inc)]']
            
            # For the DHELAS and model folder, only the regexp make sense
            model_files = ['../../Source/MODEL/[.+\.(f|inc)]']            
            helas_files = ['../../Source/DHELAS/[.+\.(f|inc)]']            
            
            # Of course feel free to use whatever pleases you for your own
            # specific tests which might target one specific file only.
            
            # Now we create the various processes for which we want to perform
            # the comparison versus old outputs.
            # Each class of test is stored in a dictionary with entries of 
            # the format:
            # {(folder_name, test_name) : 
            #    [LoopAmplitude, exporter, fortran_model, files_to_test]}
            # The format above is typically useful because we don't aim at
            # testing all processes for all exporters and all model, but we 
            # choose certain combinations which spans most possibilities.
            # Notice that the process and model can anyway be recovered from the 
            # LoopAmplitude object, so I did not bother to put it here.
            
            self.my_tests = {}
            
            # g g > t t~
            testName = 'gg_ttx'
            if self.applyFilter(testName=testName, folderName=None):
                myleglist = base_objects.LegList()
                myleglist.append(base_objects.Leg({'id':21, 'state':False}))
                myleglist.append(base_objects.Leg({'id':21, 'state':False}))
                myleglist.append(base_objects.Leg({'id':6, 'state':True}))
                myleglist.append(base_objects.Leg({'id':-6, 'state':True}))
        
                myproc = base_objects.Process({'legs': myleglist,
                                     'model': self.models['loop_sm'],
                                     'orders':{'QCD': 2, 'QED': 0},
                                     'perturbation_couplings': ['QCD'],
                                     'NLO_mode': 'virt'})
                
                myloopamp = loop_diagram_generation.LoopAmplitude(myproc)
                for exporter in ['default','optimized']:
                    testFolder = 'SM_virtQCD_%s'%exporter
                    if self.applyFilter(testName=None, folderName=testFolder): 
                        # For this first process, I check everything in both
                        # output modes
                        self.my_tests[(testFolder,testName)] = \
                            [loop_helas_objects.LoopHelasMatrixElement(myloopamp),
                             self.loop_exporters[exporter],
                             self.fortran_models['fortran_model'],
                             proc_files+helas_files+model_files]
            
            # d d~ > t t~
            testName = 'ddx_ttx'
            if self.applyFilter(testName=testName, folderName=None):
                myleglist = base_objects.LegList()
                myleglist.append(base_objects.Leg({'id':1, 'state':False}))
                myleglist.append(base_objects.Leg({'id':-1, 'state':False}))
                myleglist.append(base_objects.Leg({'id':6, 'state':True}))
                myleglist.append(base_objects.Leg({'id':-6, 'state':True}))
        
                myproc = base_objects.Process({'legs': myleglist,
                                     'model': self.models['loop_sm'],
                                     'orders':{'QCD': 2, 'QED': 0},
                                     'perturbation_couplings': ['QCD'],
                                     'NLO_mode': 'virt'})
                
                myloopamp = loop_diagram_generation.LoopAmplitude(myproc)
                for exporter in ['default','optimized']:
                    testFolder = 'SM_virtQCD_%s'%exporter
                    if self.applyFilter(testName=None, folderName=testFolder):
                        # In this case, the model and helas files would be the 
                        # same, so I only check the proc files
                        self.my_tests[(testFolder,testName)] = \
                            [loop_helas_objects.LoopHelasMatrixElement(myloopamp),
                             self.loop_exporters[exporter],
                             self.fortran_models['fortran_model'],
                             proc_files]

            # d u~ > mu- vmx g
            testName = 'dux_mumvmxg'
            if self.applyFilter(testName=testName, folderName=None):
                myleglist = base_objects.LegList()
                myleglist.append(base_objects.Leg({'id':1, 'state':False}))
                myleglist.append(base_objects.Leg({'id':-2, 'state':False}))
                myleglist.append(base_objects.Leg({'id':13, 'state':True}))
                myleglist.append(base_objects.Leg({'id':-14, 'state':True}))
                myleglist.append(base_objects.Leg({'id':21, 'state':True}))
        
                myproc = base_objects.Process({'legs': myleglist,
                                     'model': self.models['loop_sm'],
                                     'orders':{'QCD': 1, 'QED': 2},
                                     'perturbation_couplings': ['QCD'],
                                     'NLO_mode': 'virt'})
                
                myloopamp = loop_diagram_generation.LoopAmplitude(myproc)
                for exporter in ['default','optimized']:
                    testFolder = 'SM_virtQCD_%s'%exporter
                    if self.applyFilter(testName=None, folderName=testFolder):
                        # In this case, the model and helas files are mostly
                        # new, so I add them.
                        self.my_tests[(testFolder,testName)] = \
                            [loop_helas_objects.LoopHelasMatrixElement(myloopamp),
                             self.loop_exporters[exporter],
                             self.fortran_models['fortran_model'],
                             proc_files+helas_files+model_files]
            
            # Single top (rather long but really includes everything)
            # g g > w- t b~
            testName = 'gg_wmtbx'
            if self.applyFilter(testName=testName, folderName=None):
                myleglist = base_objects.LegList()
                myleglist.append(base_objects.Leg({'id':21, 'state':False}))
                myleglist.append(base_objects.Leg({'id':21, 'state':False}))
                myleglist.append(base_objects.Leg({'id':-24, 'state':True}))
                myleglist.append(base_objects.Leg({'id':6, 'state':True}))
                myleglist.append(base_objects.Leg({'id':-5, 'state':True}))
        
                myproc = base_objects.Process({'legs': myleglist,
                                     'model': self.models['loop_sm'],
                                     'orders':{'QCD': 2, 'QED': 1},
                                     'perturbation_couplings': ['QCD'],
                                     'NLO_mode': 'virt'})
                
                myloopamp = loop_diagram_generation.LoopAmplitude(myproc)
                for exporter in ['default','optimized']:
                    testFolder = 'SM_virtQCD_%s'%exporter
                    if self.applyFilter(testName=None, folderName=testFolder):
                        # In this case, the model and helas files are mostly
                        # new, so I add them.
                        self.my_tests[(testFolder,testName)] = \
                            [loop_helas_objects.LoopHelasMatrixElement(myloopamp),
                             self.loop_exporters[exporter],
                             self.fortran_models['fortran_model'],
                             proc_files+helas_files+model_files]
            
            # And the loop induced g g > h h for good measure
            testName = 'gg_hh'
            testFolder = 'SM_virtQCD_LoopInduced'
            if self.applyFilter(testName=testName, folderName=testFolder):
                myleglist = base_objects.LegList()
                myleglist.append(base_objects.Leg({'id':21, 'state':False}))
                myleglist.append(base_objects.Leg({'id':21, 'state':False}))
                myleglist.append(base_objects.Leg({'id':25, 'state':True}))
                myleglist.append(base_objects.Leg({'id':25, 'state':True}))
        
                myproc = base_objects.Process({'legs': myleglist,
                                     'model': self.models['loop_sm'],
                                     'orders':{'QCD': 1, 'QED': 2},
                                     'perturbation_couplings': ['QCD'],
                                     'NLO_mode': 'virt'})
                
                myloopamp = loop_diagram_generation.LoopAmplitude(myproc)
                # In this case, the model and helas files are mostly
                # new, so I add them.
                self.my_tests[(testFolder,testName)] = \
                    [loop_helas_objects.LoopHelasMatrixElement(myloopamp),
                     self.loop_exporters['default'],
                     self.fortran_models['fortran_model'],
                     proc_files+helas_files+model_files]

    def test_files(self, create_files = False):
        """ Compare the files of the chosen tests against the hardcoded ones
            stored in tests/input_files/LoopExporterTestComparison. If you see
            an error in the comparison and you are sure that the newest output
            is correct (i.e. you understand that this modification is meant to
            be so). Then feel free to automatically regenerate this file with
            the newest version by doing 
            
                ./test_loop_exporters folderName/testName/fileName
                
            If create_files is True (meant to be used by __main__ only) then
            it will create/update/remove the files instead of testing them.
        """
        # In create_files = True mode, we keep track of the modification to 
        # provide summary information
        modifications={'updated':[],'created':[], 'removed':[]}
        
        if self.verbose: print "\n== Operational mode : file %s ==\n"%\
                                     ('CREATION' if create_files else 'TESTING')
        for (folder_name, test_name),\
             [loop_me, exporter, fortran_model, files] in self.my_tests.items():
            if self.verbose: print "Processing %s in %s"%(test_name,folder_name)
            model = loop_me.get('processes')[0].get('model')
            if os.path.isdir(_proc_file_path):
                shutil.rmtree(_proc_file_path)
            exporter.copy_v4template(model.get('name'))
            exporter.generate_loop_subprocess(loop_me, fortran_model)
            wanted_lorentz = loop_me.get_used_lorentz()
            wanted_couplings = list(set(sum(loop_me.get_used_couplings(),[])))
            exporter.convert_model_to_mg4(model,wanted_lorentz,wanted_couplings)
            
            proc_name='P'+loop_me.get('processes')[0].shell_string()
            files_path = pjoin(_proc_file_path,'SubProcesses',proc_name)

            # First create the list of files to check as the user might be using
            # regular expressions.
            filesToCheck=[]
            for fname in files:
                if fname.endswith(']'):
                    split=fname[:-1].split('[')
                    # folder without the final /
                    folder=split[0][:-1]
                    search = re.compile('['.join(split[1:]))
                    # In filesToCheck, we must remove the files_path/ prepended
                    filesToCheck += [ f[(len(str(files_path))+1):]
                           for f in glob.glob(pjoin(files_path,folder,'*')) if \
                               (not search.match(path.basename(f)) is None and \
                                      not path.isdir(f) and not path.islink(f))]
                else:
                    filesToCheck.append(fname)
                    
            if create_files:
                # Remove files which are no longer used for comparison
                activeFiles = [self.toFileName(f) for f in filesToCheck]
                for file in glob.glob(pjoin(_hc_comparison_files,folder_name,\
                                                                test_name,'*')):
                    # Ignore the .BackUp files and directories
                    if path.basename(file).endswith('.BackUp') or\
                                                               path.isdir(file):
                        continue
                    if path.basename(file) not in activeFiles:
                        os.remove(file)
                        if self.verbose: print "    > [ REMOVED ] %s"\
                                                            %path.basename(file)
                        modifications['removed'].append(
                                            '/'.join(str(file).split('/')[-3:]))

                    
            # Make sure it is not filtered out by the user-filter
            if self.filesChecked_filter!=['ALL']:
                new_filesToCheck = []
                for file in filesToCheck:
                    # Try if it matches any filter
                    for filter in self.filesChecked_filter:
                        # A regular expression
                        if filter.endswith(']'):
                            split=filter[:-1].split('[')
                            # folder without the final /
                            folder=split[0][:-1]
                            if folder!=path.dirname(pjoin(file)):
                                continue
                            search = re.compile('['.join(split[1:]))
                            if not search.match(path.basename(file)) is None:
                                new_filesToCheck.append(file)
                                break    
                        # Just the exact filename
                        elif filter==file:
                            new_filesToCheck.append(file)
                            break
                filesToCheck = new_filesToCheck
            
            # Now we can scan them and process them one at a time
            for fname in filesToCheck:
                file_path = path.abspath(pjoin(files_path,fname))
                self.assertTrue(path.isfile(file_path),
                                            'File %s not found.'%str(file_path))
                comparison_path = pjoin(_hc_comparison_files,\
                                    folder_name,test_name,self.toFileName(fname))
                if not create_files:
                    if not os.path.isfile(comparison_path):
                        raise MadGraph5Error, 'The file %s'%str(comparison_path)+\
                                                              ' does not exist.'
                    goal = open(comparison_path).read()%misc.get_pkg_info()
                    if not self.verbose:
                        self.assertFileContains(open(file_path), goal)
                    else:
                        try:
                            self.assertFileContains(open(file_path), goal)
                        except AssertionError:
                            print "    > %s differs from the reference."%fname
                            
                else:                        
                    if not path.isdir(pjoin(_hc_comparison_files,folder_name)):
                        os.makedirs(pjoin(_hc_comparison_files,folder_name))
                    if not path.isdir(pjoin(_hc_comparison_files,folder_name,
                                                                    test_name)):
                        os.makedirs(pjoin(_hc_comparison_files,folder_name,
                                                                    test_name))
                    # Transform the package information to make it a template
                    file = open(file_path,'r')
                    target=file.read()
                    target = target.replace('MadGraph 5 v. %(version)s, %(date)s'\
                                                           %misc.get_pkg_info(),
                                          'MadGraph 5 v. %(version)s, %(date)s')
                    file.close()
                    if os.path.isfile(comparison_path):
                        file = open(comparison_path,'r')
                        existing = file.read()
                        file.close()
                        if existing == target:
                            # The following print statement is a bit of a flood
                            # if self.verbose: print "    > [ IDENTICAL ] %s"%fname
                            continue
                        else:
                            if self.verbose: print "    > [ UPDATED ] %s"%fname
                            modifications['updated'].append(
                                      '/'.join(comparison_path.split('/')[-3:]))
                        # Copying the existing reference as a backup
                        back_up_path = pjoin(_hc_comparison_files,folder_name,\
                                     test_name,self.toFileName(fname)+'.BackUp')
                        if os.path.isfile(back_up_path):
                            os.remove(back_up_path)
                        mv(comparison_path,back_up_path)
                    else:
                        if self.verbose: print "    > [ CREATED ] %s"%fname
                        modifications['created'].append(
                                      '/'.join(comparison_path.split('/')[-3:]))
                    file = open(comparison_path,'w')
                    file.write(target)
                    file.close()

            # Clean the process created
            if os.path.isdir(_proc_file_path):
                shutil.rmtree(_proc_file_path)

        # Monitor the modifications when in creation files mode by returning the
        # modifications dictionary.
        if create_files:
            return modifications
        else:
            return 'test_over'

    # This is just so that this method can be instantiated from anywhere as it 
    # is a child of 
    def runTest(self,*args,**opts):
        return super(self,IOExportMadLoopTest).runTest(*args,**opts)

if __name__ == '__main__':
    help = """ 
    To ease the actualization of these tests when some of the hardcoded
    comparison file must be updated, you can simply run this module in
    standalone. When provided with no argument, it will update everything.
    Otherwise run it with these possible arguments (all optional) :

           --folders="folder1&folder2&folder3&etc..."
           --testNames="testName1&testName2&testName3&etc..."
           --filePaths="filePath1&filePath2&filePath3&etc..."
    
    or directly like this:
    
            python test_loop_exporter "folders/testNames/filePaths"
    
    Notice that the filePath use a file path relative to
    the position SubProcess/<P0_proc_name>/ in the output.
    You are allowed to use the parent directory specification ".."
    You can use the synthax [regexp] instead of a specific filename.
    This includes only the files in this directory matching it.
    Typically '../../Source/DHELAS/[.+\.(inc|f)]' matches any file in DHELAS
    with extension .inc or .f
    Also, you can prepend '-' to the folder or test name to veto it instead of
    selecting it. Typically, ['-longTest'] considers all tests but the
    'longTest' one (synthax not available for filenames).
    
    Finally, you can launch a test only from here too. Same synthax as above,
    but add the argument --test somewhere.
    """
   
    launcher = IOExportMadLoopTest()
    launcher.filesChecked_filter = ['ALL']
    launcher.testFolders_filter = ['ALL']
    launcher.testNames_filter = ['ALL']    
    launcher.verbose = True
    UpdateFiles=True
    
    for arg in sys.argv[1:]:
        if arg.startswith('--folders='):
            launcher.testFolders_filter = arg[10:].split('&')
        elif arg.startswith('--testNames='):
            launcher.testNames_filter = arg[12:].split('&')   
        elif arg.startswith('--filePaths='):
            launcher.filesChecked_filter = arg[12:].split('&')
        elif arg == '--test':
            UpdateFiles = False
        elif not arg.startswith('--') and '/' in arg:
            launcher.testFolders_filter = arg.split('/')[0].split('&')
            launcher.testNames_filter = arg.split('/')[1].split('&')
            launcher.filesChecked_filter = '/'.join(arg.split('/')[2:]).split('&')
        else:
            print help
            sys.exit(0)
    
    print "INFO:: Using folders %s"%str(launcher.testFolders_filter)    
    print "INFO:: Using test names %s"%str(launcher.testNames_filter)         
    print "INFO:: Using file paths %s"%str(launcher.filesChecked_filter)      
    launcher.setUp()
    modifications = launcher.test_files(create_files=UpdateFiles)
    if modifications == 'test_over':
        print "Loop exporters test successfully finished."
        sys.exit(0)        
    elif not isinstance(modifications,dict):
        print "Error during the files update."
        sys.exit(0)

    if sum(len(v) for v in modifications.values())>0:
        # Display the modifications
        text = " \nModifications performed on %s at %s in"%(\
                         str(datetime.date.today()),misc.format_timer(0.0)[14:])
        text += '\n   MadGraph 5 v. %(version)s, %(date)s\n'%misc.get_pkg_info()
        for key in modifications.keys():
            if len(modifications[key])==0:
                continue
            text += "The following reference files have been %s :"%key
            text += '\n'+'\n'.join(["   %s"%mod for mod in modifications[key]])
            text += '\n'
        log = open(_hc_comparison_modif_log,mode='a')
        log.write(text)
        log.close()
        print text
        # Update the tarball, while removing the .backups.
        def noBackUps(tarinfo):
            if tarinfo.name.endswith('.BackUp'):
                return None
            else:
                return tarinfo
        tar = tarfile.open(_hc_comparison_tarball, "w:bz2")
        tar.add(_hc_comparison_files, \
                  arcname=path.basename(_hc_comparison_files), filter=noBackUps)
        tar.close()
        print "INFO:: tarball %s updated"%str(_hc_comparison_tarball)
    else:
        print "\nAll files already identical to the reference ones."+\
              " No update necessary."


