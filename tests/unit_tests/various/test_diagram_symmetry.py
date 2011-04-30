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
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.various.diagram_symmetry as diagram_symmetry
import madgraph.various.process_checks as process_checks
import models.import_ufo as import_ufo
import models.model_reader as model_reader

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

#===============================================================================
# TestModelReader
#===============================================================================
class TestDiagramSymmetry(unittest.TestCase):
    """Test class for the DiagramSymmetry class"""


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

        symmetry, perms, ident_perms = diagram_symmetry.find_symmetry(matrix_element)

        self.assertEqual(symmetry, [6,-1,-1,-1,-1,-1])

        # Check that the momentum assignments work
        process = matrix_element.get('processes')[0]

        evaluator = process_checks.MatrixElementEvaluator(self.base_model,
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
        
    def test_find_symmetry_qq_qqg_with_subprocess_group(self):
        """Test the find_symmetry function for subprocess groups"""

        procs = [[2,-2,2,-2,21], [2,2,2,2,21]]
        amplitudes = diagram_generation.AmplitudeList()

        for proc in procs:
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)
            my_leglist[1].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':self.base_model})
            my_amplitude = diagram_generation.Amplitude(my_process)
            amplitudes.append(my_amplitude)

        subproc_group = \
                  group_subprocs.SubProcessGroup.group_amplitudes(amplitudes)[0]

        symmetry, perms, ident_perms = diagram_symmetry.find_symmetry(\
                                                subproc_group)

        self.assertEqual(len([s for s in symmetry if s > 0]), 26)

        self.assertEqual(symmetry,
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -34, -35, -36, 1, 1, 1, -37, -38, -39, 1, 1, 1, -8, -9, -10, -11, -12, -13, -14, 1, 1, 1, 1, 1, 1])

        # Check that the momentum assignments work
        matrix_element = \
                     subproc_group.get('matrix_elements')[1]
        process = matrix_element.get('processes')[0]

        evaluator = process_checks.MatrixElementEvaluator(self.base_model,
                                                          auth_skipping = True,
                                                          reuse = True)
        p, w_rambo = evaluator.get_momenta(process)
        me_value, amp2_org = evaluator.evaluate_matrix_element(\
                                                        matrix_element, p)

        for isym, (sym, perm) in enumerate(zip(symmetry, perms)):
            new_p = [p[i] for i in perm]
            if sym >= 0:
                continue
            iamp = subproc_group.get('diagram_maps')[1].index(isym+1)
            isymamp = subproc_group.get('diagram_maps')[1].index(-sym)
            me_value, amp2 = evaluator.evaluate_matrix_element(\
                                              matrix_element, new_p)
            self.assertAlmostEqual(amp2[iamp], amp2_org[isymamp])
        
    def test_rotate_momenta(self):
        """Test that matrix element and amp2 identical for rotated momenta"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        myamp = diagram_generation.Amplitude(myproc)

        matrix_element = helas_objects.HelasMatrixElement(myamp)

        evaluator = process_checks.MatrixElementEvaluator(self.base_model,
                                                          auth_skipping = True,
                                                          reuse = True)
        p, w_rambo = evaluator.get_momenta(myproc)

        me_val, amp2 = evaluator.evaluate_matrix_element(\
                                          matrix_element,p)
        # Rotate momenta around x axis
        for mom in p:
            mom[2] = -mom[2]
            mom[3] = -mom[3]

        new_me_val, new_amp2 = evaluator.evaluate_matrix_element(\
                                          matrix_element, p)

        self.assertAlmostEqual(me_val, new_me_val, 12)

        for amp, new_amp in zip(amp2, new_amp2):
            self.assertAlmostEqual(amp, new_amp, 12)
            
        
#===============================================================================
# TestModelReader
#===============================================================================
class TestDiagramTag(unittest.TestCase):
    """Test class for the DiagramTag class"""


    def setUp(self):
        self.base_model = import_ufo.import_model('sm')
    
    def test_diagram_tag_gg_ggg(self):
        """Test the find_symmetry function"""

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

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        myamplitude = diagram_generation.Amplitude(myproc)

        tags = []
        permutations = []
        diagram_classes = []
        for idiag, diagram in enumerate(myamplitude.get('diagrams')):
            tag = diagram_symmetry.DiagramTag(diagram)
            try:
                ind = tags.index(tag)
            except:
                diagram_classes.append([idiag + 1])
                permutations.append([tag.get_external_numbers()])
                tags.append(tag)
            else:
                diagram_classes[ind].append(idiag + 1)
                permutations[ind].append(tag.get_external_numbers())

        permutations = [[diagram_symmetry.DiagramTag.reorder_permutation(p, perms[0])\
                         for p in perms] for perms in permutations]        

        goal_classes =  [[1, 2, 3],
                         [4],
                         [5, 6, 9, 10, 13, 14],
                         [7, 11, 15],
                         [8, 12, 16],
                         [17, 18, 19],
                         [20, 21, 22],
                         [23, 24, 25]]
        goal_perms =  [[[0, 1, 2, 3, 4], [0, 1, 2, 4, 3], [0, 1, 4, 2, 3]],
                       [[0, 1, 2, 3, 4]],
                       [[0, 1, 2, 3, 4], [0, 1, 2, 4, 3], [0, 1, 3, 2, 4],
                        [0, 1, 4, 2, 3], [0, 1, 3, 4, 2], [0, 1, 4, 3, 2]],
                       [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 1, 3, 4, 2]],
                       [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 1, 3, 4, 2]],
                       [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 1, 3, 4, 2]],
                       [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 1, 3, 4, 2]],
                       [[0, 1, 2, 3, 4], [0, 1, 2, 4, 3], [0, 1, 4, 2, 3]]]

        for i in range(len(diagram_classes)):
            self.assertEqual(diagram_classes[i], goal_classes[i])
            self.assertEqual(permutations[i], goal_perms[i])

    def test_diagram_tag_uu_uug(self):
        """Test the find_symmetry function"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        myamplitude = diagram_generation.Amplitude(myproc)

        tags = []
        permutations = []
        diagram_classes = []
        for idiag, diagram in enumerate(myamplitude.get('diagrams')):
            tag = diagram_symmetry.DiagramTag(diagram)
            try:
                ind = tags.index(tag)
            except:
                diagram_classes.append([idiag + 1])
                permutations.append([tag.get_external_numbers()])
                tags.append(tag)
            else:
                diagram_classes[ind].append(idiag + 1)
                permutations[ind].append(tag.get_external_numbers())

        permutations = [[diagram_symmetry.DiagramTag.reorder_permutation(p, perms[0])\
                         for p in perms] for perms in permutations]        

        goal_classes = [[1, 8], [2, 9], [3, 10], [4, 11], [5, 12], [6, 13],
                        [7, 14], [15, 18], [16, 19], [17, 20], [21, 24],
                        [22, 25], [23, 26]]

        goal_perms = [[[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]]]

        for i in range(len(diagram_classes)):
            self.assertEqual(diagram_classes[i], goal_classes[i])
            self.assertEqual(permutations[i], goal_perms[i])

