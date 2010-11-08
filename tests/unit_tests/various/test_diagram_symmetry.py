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
"""Unit test Library for the objects in decay module."""
from __future__ import division

import math
import copy
import os
import sys
import time

import tests.unit_tests as unittest
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.various.diagram_symmetry as diagram_symmetry
import madgraph.various.process_checks as process_checks
import models.import_ufo as import_ufo
import models.model_reader as model_reader

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

#===============================================================================
# TestModelReader
#===============================================================================
class TestDiagramSymmetry(unittest.TestCase):
    """Test class for the MatrixElementChecker and get_momenta"""


    def setUp(self):
        self.base_model = import_ufo.import_model('sm')
    
    def test_find_symmetry_epem_aaa(self):
        """Test the find_symmetry function"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        myamplitude = diagram_generation.Amplitude(myproc)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        symmetry, perms, ident_perms = \
                                diagram_symmetry.find_symmetry(matrix_element)

        self.assertEqual(symmetry, [6,-1,-1,-1,-1,-1])

        # Check that the momentum assignments work
        process = matrix_element.get('processes')[0]
        full_model = model_reader.ModelReader(self.base_model)
        full_model.set_parameters_and_couplings()
        stored_quantities = {}
        helas_writer = helas_call_writers.PythonUFOHelasCallWriter(\
                                                               self.base_model)

        
        p, w_rambo = process_checks.get_momenta(process, full_model)
        me_value, amp2_org = process_checks.evaluate_matrix_element(\
                                          matrix_element,stored_quantities,
                                          helas_writer, full_model, p)

        for isym, (sym, perm) in enumerate(zip(symmetry, perms)):
            new_p = [p[i] for i in perm]
            stored_quantities['matrix_elements'] = []
            me_value, amp2 = process_checks.evaluate_matrix_element(\
                                              matrix_element,stored_quantities,
                                              helas_writer, full_model, new_p)
            if sym >= 0:
                continue
            self.assertAlmostEqual(amp2[isym], amp2_org[-sym-1])
        

    def test_find_symmetry_udbar_ggg(self):
        """Test the find_symmetry function"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                           'state':False}))
        #myleglist.append(base_objects.Leg({'id':24,
        #                                   'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        myamplitude = diagram_generation.Amplitude(myproc)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        symmetry, perms, ident_perms = diagram_symmetry.find_symmetry(matrix_element)

        self.assertEqual(len([s for s in symmetry if s > 0]), 4)

        self.assertEqual(symmetry,
                         [3, -1, -1, 0, 6, -5, 3, -5, -5, -7, -5, -5, -7,
                          3 , -14, -14])

        # Check that the momentum assignments work
        process = matrix_element.get('processes')[0]
        full_model = model_reader.ModelReader(self.base_model)
        full_model.set_parameters_and_couplings()
        stored_quantities = {}
        helas_writer = helas_call_writers.PythonUFOHelasCallWriter(\
                                                               self.base_model)

        
        p, w_rambo = process_checks.get_momenta(process, full_model)
        me_value, amp2_org = process_checks.evaluate_matrix_element(\
                                          matrix_element,stored_quantities,
                                          helas_writer, full_model, p,
                                          reuse = True)

        for isym, (sym, perm) in enumerate(zip(symmetry, perms)):
            new_p = [p[i] for i in perm]
            me_value, amp2 = process_checks.evaluate_matrix_element(\
                                              matrix_element,stored_quantities,
                                              helas_writer, full_model, new_p,
                                              reuse = True)
            if sym >= 0:
                continue
            self.assertAlmostEqual(amp2[isym], amp2_org[-sym-1])
        
