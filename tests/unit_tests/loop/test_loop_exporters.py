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

"""Unit test library for the various properties of objects in 
   loop_helas_objects.py"""

import copy
import math
import os
import sys
import shutil
import tarfile
import datetime
import glob

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.interface.master_interface as MGCmd

import tests.unit_tests.loop.test_loop_diagram_generation as looptest
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import madgraph.core.color_amp as color_amp
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
import madgraph.iolibs.file_writers as writers
import tests.IOTests as IOTests
import aloha

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

#===============================================================================
# IOExportMadLoopUTest
#===============================================================================
class IOExportMadLoopUnitTest(IOTests.IOTestManager):
    """Test class for the loop exporter modules. It uses hardcoded output 
    for the comparisons."""

    loop_exporters = {}

    # A helper function to add more easily IOTests for several exporters.
    def addIOTestsForProcess(self,testName,testFolder,particles_ids,exporters,orders,
                             files_to_check=IOTests.IOTest.all_files,
                             perturbation_couplings=['QCD'],
                             NLO_mode='virt',
                             model=None,
                             fortran_model=None):
        """ Simply adds a test for the process defined and all the exporters
        specified."""
        
        if model==None:
            model = self.models['loop_sm']
        if fortran_model==None:
            fortran_model = self.fortran_models['fortran_model']
        
        needed = False

        if not isinstance(exporters,list):
            if self.need(testFolder,testName):
                needed = True    
        elif any(self.need('%s_%s'%(testFolder,exporter) ,testName) for \
                                                         exporter in exporters):
            needed = True
            
        if not needed:
            return
        
        myleglist = base_objects.LegList()
        for i, pid in enumerate(particles_ids):
            myleglist.append(base_objects.Leg({'id':pid, 
                                           'state':False if i<2 else True}))
        myproc = base_objects.Process({'legs': myleglist,
                        'model': model,
                        'orders': orders,
                        'perturbation_couplings': perturbation_couplings,
                        'NLO_mode': NLO_mode})
        
        if isinstance(exporters, dict):
            # Several exporters given in a dictionary
            test_list = [('%s_%s'%(testFolder,exp),exp) for exp in exporters]            
        elif not isinstance(exporters,list) :
            # Exporter directly given
            test_list = [(testFolder,exporters)]
        else:
            test_list = [('%s_%s'%(testFolder,exp),exp) for exp in exporters] 

        for (folderName, exporter) in test_list:
            if self.need(folderName,testName):
                self.addIOTest(folderName,testName, IOTests.IOTest(\
                  test_instance = self,
                  procdef=myproc,
                  exporter=exporter,
                  helasModel=fortran_model,
                  testedFiles=files_to_check,
                  outputPath=_proc_file_path))
    
    def get_exporter_withName(self, exporter_name):
        """ Returns on demand the exporter of given nickname """


        if exporter_name == 'default':
            self.loop_exporters[exporter_name] = loop_exporters.\
                              LoopProcessExporterFortranSA(
                              _mgme_file_path, _proc_file_path,
                              {'clean':False, 'complex_mass':False, 
                               'export_format':'madloop','mp':True,
                               'loop_dir':_loop_file_path,
                               'cuttools_dir':_cuttools_file_path,
                               'fortran_compiler':'gfortran',
                               'output_dependencies':'external',
                               'SubProc_prefix': 'P',
                               'compute_color_flows': False})
        elif exporter_name == 'optimized':
            self.loop_exporters[exporter_name] = loop_exporters.\
                              LoopProcessOptimizedExporterFortranSA(\
                              _mgme_file_path, _proc_file_path,
                              {'clean':False, 'complex_mass':False, 
                               'export_format':'madloop','mp':True,
                               'loop_dir':_loop_file_path,
                               'cuttools_dir':_cuttools_file_path,
                               'fortran_compiler':'gfortran',
                               'output_dependencies':'external',
                               'SubProc_prefix': 'P',
                               'compute_color_flows': False})
        else:
            raise MadGraph5Error, 'Exporter with nickname '+\
                                          '%s not implemented'%exporter_name
        return self.loop_exporters[exporter_name]

    def load_IOTestsUnit(self):
        """load the models and exporters if necessary."""
            
        if not hasattr(self, 'models') or \
           not hasattr(self, 'fortran_models') or \
           not hasattr(self, 'loop_exporters'):\
           
            self.models = { \
                'loop_sm' : import_ufo.import_model('loop_sm') 
                          }
            self.fortran_models = {
                'fortran_model' : helas_call_writers.FortranUFOHelasCallWriter(\
                                                         self.models['loop_sm']) 
                                  }

            # g g > t t~
            self.addIOTestsForProcess( testName = 'gg_ttx',
                                       testFolder = 'short_ML_SMQCD',
                                       particles_ids = [21,21,6,-6],
                                       exporters = ['default','optimized'],
                                       orders = {'QCD':2,'QED':0} )

            # d d > t t~ (only the proc files for this one)
            self.addIOTestsForProcess( testName = 'ddx_ttx',
                                       testFolder = 'short_ML_SMQCD',
                                       particles_ids = [1,-1,6,-6],
                                       exporters = ['default','optimized'],
                                       orders = {'QCD':2,'QED':0},
                                       files_to_check=IOTests.IOTest.proc_files)

            # And the loop induced g g > h h for good measure 
            # Use only one exporter only here
            self.addIOTestsForProcess( testName = 'gg_hh',
                                       testFolder = 'short_ML_SMQCD_LoopInduced',
                                       particles_ids = [21,21,25,25],
                                       exporters = 'default',
                                       orders = {'QCD': 2, 'QED': 2} )

    def testIO_ProcOutputIOTests(self, load_only=False):
      """ Run the iotests """
      
      self.load_IOTestsUnit()      
      if not load_only:
          # Set it to True if you want info during the regular test_manager.py runs
          self.runIOTests(verbose=False)

#===============================================================================
# IOExportMadLoopUTest
#===============================================================================
class IOTestMadLoopSquaredOrdersExport(IOTests.IOTestManager):
    """Test class for the writing of loop_matrix.f in the presence of squared 
    order constraints and differentiation of different "split orders" 
    combinations."""
    
    @IOTests.set_global(unitary=False)
    def setUp(self):
       """Loading the different writers, exporters and model used for these
       IOTests"""
       if not hasattr(self, 'model'):
           self.model=import_ufo.import_model('loop_qcd_qed_sm-full')
           
       if not hasattr(self, 'exporter'):
           self.exporter = loop_exporters.\
                                  LoopProcessOptimizedExporterFortranSA(\
                                  _mgme_file_path, _proc_file_path,
                                  {'clean':False, 'complex_mass':False, 
                                   'export_format':'madloop','mp':True,
                                   'loop_dir':_loop_file_path,
                                   'cuttools_dir':_cuttools_file_path,
                                   'fortran_compiler':'gfortran',
                                   'output_dependencies':'external',
                                   'SubProc_prefix': '',
                                   'compute_color_flows': False})

    @IOTests.set_global(unitary=False)
    @IOTests.createIOTest(groupName='LoopSquaredOrder_IOTest')
    def testIO_Loop_sqso_uux_ddx(self):
        """ target: [loop_matrix(.*)\.f]
                    [write_mp_compute_loop_coefs(.*)\.f]
                    [mp_helas_calls(.*)\.f]
                    [helas_calls(.*)\.f]
                    [loop_CT_calls(.*)\.f]
        """
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':2,'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,'state':False}))
        myleglist.append(base_objects.Leg({'id':1,'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,'state':True}))

        fortran_model=\
          helas_call_writers.FortranUFOHelasCallWriterOptimized(self.model,False)
        
        SO_tests = [({},['QCD','QED'],{},{},['QCD','QED'],'QCDQEDpert_default')
                    ,({},['QCD'],{},{},['QCD'],'QCDpert_default')
                    ,({},['QED'],{},{},['QED'],'QEDpert_default')            
                    ,({},['QCD','QED'],{'QCD':4},{'QCD':'=='},['QCD','QED'],
                                                        'QCDQEDpert_QCDsq_eq_4')
                    ,({},['QCD','QED'],{'QED':4},{'QCD':'<='},['QCD','QED'],
                                                        'QCDQEDpert_QEDsq_le_4')
                    ,({},['QCD','QED'],{'QCD':4},{'QCD':'>'},['QCD','QED'],
                                                        'QCDQEDpert_QCDsq_gt_4')
                    ,({'QED':2},['QCD','QED'],{'QCD':0,'QED':2},
                      {'QCD':'>','QED':'>'},['QCD','QED'],
                                   'QCDQEDpert_QCDsq_gt_0_QEDAmpAndQEDsq_gt_2')
                    ,({'QED':2},['QCD','QED'],{'WEIGHTED':10,'QED':2},
                      {'WEIGHTED':'<=','QED':'>'},['WEIGHTED','QCD','QED'],
                                  'QCDQEDpert_WGTsq_le_10_QEDAmpAndQEDsq_gt_2')]

        for orders, pert_orders, sq_orders , sq_orders_type, split_orders, name \
                                                                    in SO_tests:
            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.model,
                                           'orders': orders,
                                           'squared_orders': sq_orders,
                                           'perturbation_couplings':pert_orders,
                                           'sqorders_types':sq_orders_type,
                                           'split_orders':split_orders})
                
            myloopamp = loop_diagram_generation.LoopAmplitude(myproc)
            matrix_element=loop_helas_objects.LoopHelasMatrixElement(\
                                                myloopamp,optimized_output=True)
            
            # It is enough here to generate and check the filer loop_matrix.f 
            # only here. For that we must initialize the general replacement 
            # dictionary first (The four functions below are normally directly
            # called from the write_matrix_element function in the exporter
            # [but we don't call it here because we only want the file 
            # loop_matrix.f]).
            matrix_element.rep_dict = self.exporter.\
                                   generate_general_replace_dict(matrix_element)
            
            # and for the same reason also force the computation of the analytical
            # information in the Helas loop diagrams.
            matrix_element.compute_all_analytic_information(
                                      self.exporter.get_aloha_model(self.model))
            
            # Finally the entries specific to the optimized output
            self.exporter.set_optimized_output_specific_replace_dict_entries(\
                                                                 matrix_element)
        
            # We can then finally write out 'loop_matrix.f'
            with misc.chdir(self.IOpath):
                writer = writers.FortranWriter(\
                                     pjoin(self.IOpath,'loop_matrix_%s.f'%name))
                self.exporter.write_loopmatrix(writer,matrix_element,
                                     fortran_model, write_auxiliary_files=False)
                writer = writers.FortranWriter(\
                     pjoin(self.IOpath,'write_mp_compute_loop_coefs_%s.f'%name))                    
                self.exporter.write_mp_compute_loop_coefs(writer, 
                                                  matrix_element, fortran_model)
            for file in glob.glob(pjoin(self.IOpath,'mp_helas_calls_*.f'))+\
                        glob.glob(pjoin(self.IOpath,'helas_calls_*.f'))+\
                        glob.glob(pjoin(self.IOpath,'loop_CT_calls*.f')):
                base_name = '.'.join(file.split('.')[:-1])+'_%s.'+file.split('.')[-1]
                shutil.move(file,base_name%name)