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
                  group_subprocs.SubProcessGroup.group_amplitudes(amplitudes,"madevent")[0]

        symmetry, perms, ident_perms = diagram_symmetry.find_symmetry(\
                                                subproc_group)

        self.assertEqual(len([s for s in symmetry if s > 0]), 23)

        self.assertEqual(symmetry,
                         [1, 1, 1, 1, -2, -3, -4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, -8, -9, -10, -11, -12, -13,
                          -14, -15, -16, -17, -21, -22, -23])

        return

        # The test below doesn't apply with the new way of determining
        # config symmetry for subprocess groups, since we don't demand
        # that symmetric diagrams have identical particles.

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

    def test_find_symmetry_gg_tt_fullylept(self):
        """Test the find_symmetry function for subprocess groups"""

        procs = [[21,21, 6, -6]]
        decayt = [[6,5,-11,12],[6, 5,-13, 14]]
        decaytx = [[-6,-5,11,-12],[-6, -5, 13,-14]]
        amplitudes = diagram_generation.AmplitudeList()
        decay_amps = diagram_generation.DecayChainAmplitudeList()

        proc = procs[0]
        my_leglist = base_objects.LegList([\
                    base_objects.Leg({'id': id, 'state': True}) for id in proc])
        my_leglist[0].set('state', False)
        my_leglist[1].set('state', False)
        my_process = base_objects.Process({'legs':my_leglist,
                                                       'model':self.base_model})
        my_amplitude = diagram_generation.Amplitude(my_process)
        amplitudes.append(my_amplitude)
        
        for dect in decayt:
            my_top_decaylegs = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in dect])
            my_top_decaylegs[0].set('state', False)
            my_decayt_proc = base_objects.Process({'legs':my_top_decaylegs,
                                                  'model':self.base_model,
                                                  'is_decay_chain': True})
            my_decayt = diagram_generation.DecayChainAmplitude(my_decayt_proc)
            decay_amps.append(my_decayt)
            
        for dectx in decaytx:
            # Define the multiprocess
            my_topx_decaylegs = base_objects.LegList([\
               base_objects.Leg({'id': id, 'state': True}) for id in dectx])
            my_topx_decaylegs[0].set('state', False)
            my_decaytx_proc = base_objects.Process({'legs':my_topx_decaylegs,
                                              'model':self.base_model,
                                              'is_decay_chain': True}) 
            
            my_decaytx = diagram_generation.DecayChainAmplitude(my_decaytx_proc)
            decay_amps.append(my_decaytx)                   
                
                

        amplitudes = diagram_generation.DecayChainAmplitudeList([\
                diagram_generation.DecayChainAmplitude({\
                        'amplitudes': amplitudes,
                        'decay_chains': decay_amps})])

        subproc_groups = \
                  group_subprocs.DecayChainSubProcessGroup.group_amplitudes(\
                         amplitudes).generate_helas_decay_chain_subproc_groups()
        self.assertEqual(len(subproc_groups), 1)

        subproc_group = subproc_groups[0]
        self.assertEqual(len(subproc_group.get('matrix_elements')), 1)

        symmetry, perms, ident_perms = diagram_symmetry.find_symmetry(\
                                                subproc_group)
        
        sol_perms = [range(8), range(8), [0,1,5,6,7,2,3,4]]  

        self.assertEqual(len([s for s in symmetry if s > 0]), 2)
        self.assertEqual(symmetry, [1, 1, -2])

        self.assertEqual(perms, sol_perms)
        
        
        
    def test_find_symmetry_decay_chain_with_subprocess_group(self):
        """Test the find_symmetry function for subprocess groups"""

        procs = [[2,-1,24,21,21], [-3,4,24,21,21]]
        decays = [[24,-11,12],[24,-13,14]]
        amplitudes = diagram_generation.AmplitudeList()
        decay_amps = diagram_generation.DecayChainAmplitudeList()

        for proc, decay in zip(procs, decays):
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)
            my_leglist[1].set('state', False)

            my_decaylegs = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in decay])

            my_decaylegs[0].set('state', False)
            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':self.base_model})
            my_decay_proc = base_objects.Process({'legs':my_decaylegs,
                                                  'model':self.base_model,
                                                  'is_decay_chain': True})
            my_amplitude = diagram_generation.Amplitude(my_process)
            my_decay = diagram_generation.DecayChainAmplitude(my_decay_proc)
            amplitudes.append(my_amplitude)
            decay_amps.append(my_decay)

        amplitudes = diagram_generation.DecayChainAmplitudeList([\
                diagram_generation.DecayChainAmplitude({\
                        'amplitudes': amplitudes,
                        'decay_chains': decay_amps})])

        subproc_groups = \
                  group_subprocs.DecayChainSubProcessGroup.group_amplitudes(\
                         amplitudes,"madevent").generate_helas_decay_chain_subproc_groups()
        self.assertEqual(len(subproc_groups), 1)

        subproc_group = subproc_groups[0]
        self.assertEqual(len(subproc_group.get('matrix_elements')), 2)

        symmetry, perms, ident_perms = diagram_symmetry.find_symmetry(\
                                                subproc_group)

        self.assertEqual(len([s for s in symmetry if s > 0]), 5)

        self.assertEqual(symmetry,
                         [1, -1, 1, 1, 1, -4, -5, 1])

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

        self.assertAlmostEqual(me_val, new_me_val, 10)

        for amp, new_amp in zip(amp2, new_amp2):
            self.assertAlmostEqual(amp, new_amp, 10)
            
        
