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
"""Unit tests for four-fermion models."""
from __future__ import division

import copy
import logging
import math
import os
import sys
import time

import tests.unit_tests as unittest
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.various.process_checks as process_checks
import madgraph.various.diagram_symmetry as diagram_symmetry
import models.import_ufo as import_ufo
import models.model_reader as model_reader

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

# Special logger
logger = logging.getLogger('tests.unit_tests')

#===============================================================================
# Models4FermionTest
#===============================================================================
class Models4FermionTest(unittest.TestCase):
    """Base test class for comparing 4-fermion models to resolved models"""

    def uu_to_ttng_test(self, nglue = 0):
        """Test the process u u > t t g for 4fermion models"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':6}))
        myleglist.append(base_objects.Leg({'id':6}))
        myleglist.extend([base_objects.Leg({'id':21}) for i in range(nglue)])

        values = {}
        p = None
        for model in 'scalar', '4ferm':

            base_model = eval('self.base_model_%s' % model)
            full_model = eval('self.full_model_%s' % model)
            myproc = base_objects.Process({'legs':myleglist,
                                           'model':base_model})

            evaluator = process_checks.MatrixElementEvaluator(base_model,
                                                                  reuse = False)
            evaluator.full_model = full_model
            
            if not p:
                p, w_rambo = evaluator.get_momenta(myproc)

            amplitude = diagram_generation.Amplitude(myproc)

            matrix_element = helas_objects.HelasMatrixElement(amplitude)

            stored_quantities = {}

            values[model] = evaluator.evaluate_matrix_element(matrix_element,
                                                              p)[0]

            
        self.assertAlmostEqual(values['scalar'], values['4ferm'], 3)

#===============================================================================
# TestSchannelModels
#===============================================================================
class TestSchannelModels(Models4FermionTest):
    """Test class for the s-channel type 4-fermion model"""

    def setUp(self):
        self.base_model_scalar = import_ufo.import_model('sextet_diquarks')
        self.full_model_scalar = \
                               model_reader.ModelReader(self.base_model_scalar)
        self.full_model_scalar.set_parameters_and_couplings()
        self.full_model_scalar.get('parameter_dict')['mdl_MSIX'] = 1.e5
        
        self.base_model_4ferm = import_ufo.import_model('uutt_sch_4fermion')
        self.full_model_4ferm = \
                               model_reader.ModelReader(self.base_model_4ferm)
        self.full_model_4ferm.set_parameters_and_couplings()
    
    def test_uu_to_tt_sch(self):
        """Test the process u u > t t between s-channel and 4fermion vertex"""
        self.uu_to_ttng_test(0)

    def test_uu_to_ttg_sch(self):
        """Test the process u u > t t g between s-channel and 4fermion vertex"""
        self.uu_to_ttng_test(1)


    def test_find_symmetry_uu_tt(self):
        """Test the find_symmetry function"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':6,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model_4ferm})

        myamplitude = diagram_generation.Amplitude(myproc)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        symmetry, perms, ident_perms = diagram_symmetry.find_symmetry(matrix_element)

        self.assertEqual(symmetry, [1])

        # Check that the momentum assignments work
        process = matrix_element.get('processes')[0]

        evaluator = process_checks.MatrixElementEvaluator(self.base_model_4ferm,
                                                          auth_skipping = True,
                                                          reuse = True)
        
        p, w_rambo = evaluator.get_momenta(process)
        me_value, amp2_org = evaluator.evaluate_matrix_element(\
                                          matrix_element, p)

        for isym, (sym, perm) in enumerate(zip(symmetry, perms)):
            new_p = [p[i] for i in perm]
            if sym >= 0:
                continue
            me_value, amp2 = evaluator.evaluate_matrix_element(matrix_element,
                                                               new_p)
            self.assertAlmostEqual(amp2[isym], amp2_org[-sym-1])
        
    def test_find_symmetry_uu_tt_with_subprocess_group(self):
        """Test the find_symmetry function for subprocess groups"""

        procs = [[2,2,6,6]]
        amplitudes = diagram_generation.AmplitudeList()

        for proc in procs:
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)
            my_leglist[1].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':self.base_model_4ferm})
            my_amplitude = diagram_generation.Amplitude(my_process)
            amplitudes.append(my_amplitude)

        subproc_group = \
                  group_subprocs.SubProcessGroup.group_amplitudes(amplitudes, "madevent")[0]

        symmetry, perms, ident_perms = diagram_symmetry.find_symmetry(\
                                                                  subproc_group)

        self.assertEqual(len([s for s in symmetry if s > 0]), 1)

        self.assertEqual(symmetry,
                         [1])

        return
        
#===============================================================================
# TestTchannelModels
#===============================================================================
class TestTchannelModels(Models4FermionTest):
    """Test class for the t-channel type 4-fermion model"""

    #def setUp(self):
    #    self.base_model_scalar = import_ufo.import_model('uutt_tch_scalar')
    #    self.full_model_scalar = \
    #                           model_reader.ModelReader(self.base_model_scalar)
    #    self.full_model_scalar.set_parameters_and_couplings()
    #    
    #    self.base_model_4ferm = import_ufo.import_model('uutt_tch_4fermion')
    #    self.full_model_4ferm = \
    #                           model_reader.ModelReader(self.base_model_4ferm)
    #    self.full_model_4ferm.set_parameters_and_couplings()
    
    def test_uu_to_tt_tch(self):
        """Test the process u u > t t between t-channel and 4fermion vertex"""
        logger.info('test_uu_to_tt_tch bypassed')
        #self.uu_to_ttng_test(0)

    def test_uu_to_ttg_tch(self):
        """Test the process u u > t t g between t-channel and 4fermion vertex"""
        logger.info('test_uu_to_ttg_tch bypassed')
    #    self.uu_to_ttng_test(1)
        
