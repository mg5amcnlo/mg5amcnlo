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

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir))

import tests.unit_tests as unittest

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
import tests.IOTests as IOTests
import tests.unit_tests.loop.test_loop_exporters as test_loop_exporters

from madgraph.iolibs.files import cp, ln, mv
from madgraph import MadGraph5Error

pjoin = os.path.join
path = os.path

_file_path = os.path.dirname(os.path.realpath(__file__))

_input_file_path = os.path.abspath(os.path.join(_file_path, \
                                                  os.path.pardir,'input_files'))
_mgme_file_path = os.path.abspath(os.path.join(_file_path, *([os.path.pardir]*2)))
_loop_file_path = os.path.join(_mgme_file_path,'Template','loop_material')
_cuttools_file_path = os.path.join(_mgme_file_path, 'vendor','CutTools')
_proc_file_path = os.path.join(_mgme_file_path, 'UNITTEST_proc')

#===============================================================================
# IOExportMadLoopUTest
#===============================================================================
class IOExportMadLoopAcceptanceTest(test_loop_exporters.IOExportMadLoopUnitTest):
    """Test class for the loop exporter modules. It uses hardcoded output 
    for the comparisons."""

    def testIO_ProcOutputIOTests(self, load_only=False):
      """ Run the iotests """
      
      self.load_IOTestsAcceptance()      
      if not load_only:
          # Set it to True if you want info during the regular test_manager.py runs
          self.runIOTests(verbose=False)

    def load_IOTestsAcceptance(self):
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
            
            self.loop_exporters = {
                'default' : loop_exporters.LoopProcessExporterFortranSA(\
                                  _mgme_file_path, _proc_file_path,
                                  {'clean':False, 'complex_mass':False, 
                                   'export_format':'madloop','mp':True,
                                   'loop_dir':_loop_file_path,
                                   'cuttools_dir':_cuttools_file_path,
                                   'fortran_compiler':'gfortran',
                                   'output_dependencies':'external',
                                   'SubProc_prefix': '',
                                   'compute_color_flows': False}),
                'optimized' : loop_exporters.\
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
                                  }

            # d u~ > mu- vmx g
            self.addIOTestsForProcess( testName = 'dux_mumvmxg',
                                       testFolder = 'long_ML_SMQCD',
                                       particles_ids = [1,-2,13,-14,21],
                                       exporters = ['default','optimized'],
                                       orders = {'QCD': 1, 'QED': 2} )

            # g g > w- t b~ Single top (long but really includes everything)
            self.addIOTestsForProcess( testName = 'gg_wmtbx',
                                       testFolder = 'long_ML_SMQCD',
                                       particles_ids = [21,21,-24,6,-5],
                                       exporters = ['default','optimized'],
                                       orders = {'QCD': 2, 'QED': 1} )

