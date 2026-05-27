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

from __future__ import absolute_import
import math
import copy
import os
import sys
import time

import tests.unit_tests as unittest
import madgraph.core.base_objects as base_objects
import madgraph.various.process_checks as process_checks
import madgraph.various.misc as misc
import models.import_ufo as import_ufo
import models.model_reader as model_reader
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

#===============================================================================
# TestModelReader
#===============================================================================
class TestMatrixElementChecker(unittest.TestCase):
    """Test class for the MatrixElementChecker and get_momenta"""


    def setUp(self):
        
        self.base_model = import_ufo.import_model('sm')
        #sm_path = import_ufo.find_ufo_path('sm')
        #self.base_model = import_ufo.import_model(sm_path)
    
    def test_get_momenta(self):
        """Test the get_momenta function"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':11,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True,
                                           'number': 4}))
        myleglist.append(base_objects.Leg({'id':23,
                                           'state':True,
                                           'number': 5}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        evaluator = process_checks.MatrixElementEvaluator(self.base_model)
        full_model = evaluator.full_model
        p, w_rambo = evaluator.get_momenta(myproc)

        # Check massless external momenta
        for mom in p[:-1]:
            mass = mom[0]**2-(mom[1]**2+mom[2]**2+mom[3]**2)
            self.assertAlmostEqual(mass, 0., 8)

        mom = p[-1]
        mass = math.sqrt(mom[0]**2-(mom[1]**2+mom[2]**2+mom[3]**2))
        self.assertAlmostEqual(mass,
                               full_model.get('parameter_dict')['mdl_MZ'],
                               8)

        # Check momentum balance
        outgoing = [0]*4
        incoming = [0]*4
        for i in range(4):
            incoming[i] = sum([mom[i] for mom in p[:2]])
            outgoing[i] = sum([mom[i] for mom in p[2:]])
            self.assertAlmostEqual(incoming[i], outgoing[i], 8)

        # Check non-zero final state momenta
        for mom in p[2:]:
            for i in range(4):
                self.assertGreater(abs(mom[i]), 0.)

    def test_comparison_for_process(self):
        """Test check process for e+ e- > a Z"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':23,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})
        process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
        comparison = process_checks.check_processes(myproc)[0][0]

        self.assertEqual(len(comparison['values']), 8)
        self.assertGreater(comparison['values'][0], 0)
        self.assertTrue(comparison['passed'])

        comparison = process_checks.check_gauge(myproc)
        
        #check number of helicities/jamp
        nb_hel = []
        nb_jamp = [] 
        for one_comp in comparison:
            nb_hel.append(len(one_comp['value']['jamp']))
            nb_jamp.append(len(one_comp['value']['jamp'][0]))
        self.assertEqual(nb_hel, [24])
        self.assertEqual(nb_jamp, [1])
        
        nb_fail = process_checks.output_gauge(comparison, output='fail')
        self.assertEqual(nb_fail, 0)
        
        comparison = process_checks.check_lorentz(myproc)
        #check number of helicities/jamp
        nb_hel = []
        nb_jamp = [] 
        for one_comp in comparison:
            nb_hel.append(len(one_comp['results'][0]['jamp']))
            nb_jamp.append(len(one_comp['results'][0]['jamp'][0]))
        self.assertEqual(nb_hel, [24])
        self.assertEqual(nb_jamp, [1])
        
        nb_fail = process_checks.output_lorentz_inv(comparison, output='fail')
        self.assertEqual(0, nb_fail)        
        
    def test_comparison_for_multiprocess(self):
        """Test check process for multiprocess"""

        myleglist = base_objects.MultiLegList()

        p = [1,2,-1,-2]

        myleglist.append(base_objects.MultiLeg({'ids':p,
                                           'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':p,
                                           'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':p}))
        myleglist.append(base_objects.MultiLeg({'ids':p}))

        myproc = base_objects.ProcessDefinition({'legs':myleglist,
                                                 'model':self.base_model,
                                                 'orders':{'QED':0}})
        process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
        comparisons, used_aloha = process_checks.check_processes(myproc)
        goal_value_len = [8, 2]

        for i, comparison in enumerate(comparisons):
            self.assertEqual(len(comparison['values']), goal_value_len[i])
            self.assertTrue(comparison['passed'])
        
        comparisons = process_checks.check_lorentz(myproc)
        nb_fail = process_checks.output_lorentz_inv(comparisons, 
                                                    output='fail')
        self.assertEqual(0, nb_fail)
        
        #check number of helicities/jamp
        nb_hel = []
        nb_jamp = [] 
        for one_comp in comparisons:
            if one_comp['results'] != 'pass':
                nb_hel.append(len(one_comp['results'][0]['jamp']))
                nb_jamp.append(len(one_comp['results'][0]['jamp'][0]))
        self.assertEqual(nb_hel, [16, 16])
        self.assertEqual(nb_jamp, [2, 2])
        
        for i, comparison in enumerate(comparisons):
            if i == 2:
                self.assertEqual(comparison['results'],'pass')
                continue
            else:
                nb_fail = process_checks.output_lorentz_inv([comparison], 
                                                            output='fail')
                self.assertEqual(0, nb_fail)

    def test_fd_python_value_matches_unitary_fixed_point(self):
        """FD Python evaluation should stay close to Unitary at a fixed point."""

        import madgraph.interface.master_interface as interface
        import madgraph.core.helas_objects as helas_objects

        process_line = 'u u > w+ w- u u QCD=0'
        p = [
            [500.0, 0.0, 0.0, 500.0],
            [500.0, 0.0, 0.0, -500.0],
            [401.01469431414546, 32.49828279011273, -244.71617101372763, 305.6197414460834],
            [260.789910425083, 5.193462736487908, -64.91667713260222, -239.38048040727654],
            [35.78415077155445, 20.545542318439836, 24.517065174688458, 16.04056272400125],
            [302.41124448921704, -58.237287845040406, 285.11578297164135, -82.27982376280812]
        ]

        values = {}
        for gauge in ['unitary', 'FD']:
            cmd = interface.MasterCmd()
            cmd.no_notification()
            cmd.exec_cmd('set gauge %s' % gauge)
            cmd.exec_cmd('import model sm')
            cmd.exec_cmd('generate %s' % process_line)
            me = helas_objects.HelasMatrixElement(cmd._curr_amps[0])
            evaluator = process_checks.MatrixElementEvaluator(cmd._curr_model, cmd=cmd, reuse=False)
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[particle.get('width')] = 0.
            values[gauge], _ = evaluator.evaluate_matrix_element(me, p=p)

        relative_difference = abs(values['unitary'] - values['FD']) / \
            max(abs(values['unitary']), abs(values['FD']))
        self.assertLess(relative_difference, 5e-2)

    def test_failed_process(self):
        """Test that check process fails for wrong color-Lorentz."""

        # Change 4g interaction so color and lorentz don't agree
        id = [int.get('id') for int in self.base_model.get('interactions')
               if [p['pdg_code'] for p in int['particles']] == [21,21,21,21]][0]
        gggg = self.base_model.get_interaction(id)
        assert [p['pdg_code'] for p in gggg['particles']] == [21,21,21,21]
        gggg.set('lorentz', ['VVVV1', 'VVVV4', 'VVVV3'])

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})
        process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
        comparison = process_checks.check_processes(myproc)[0][0]

        self.assertFalse(comparison['passed'])

        comparison = process_checks.check_processes(myproc, quick = True)[0][0]

        self.assertFalse(comparison['passed'])
        
        comparison = process_checks.check_gauge(myproc)
        nb_fail = process_checks.output_gauge(comparison, output='fail')
        self.assertNotEqual(nb_fail, 0)
        
        comparison = process_checks.check_lorentz(myproc)
        nb_fail = process_checks.output_lorentz_inv(comparison, output='fail')
        self.assertNotEqual(0, nb_fail)
        #self.assertNotAlmostEqual(max(comparison[0][1]), min(comparison[0][1]))

#===============================================================================
# TestLorentzInvariance
#===============================================================================
class TestLorentzInvariance(unittest.TestCase):
    """Test class for the Lorentz Invariance and boost_momenta"""
    
    def setUp(self):
        sm_path = import_ufo.find_ufo_path('MSSM_SLHA2')
        self.base_model = import_ufo.import_model(sm_path)
        
    def test_boost_momenta(self):
        """check if the momenta are boosted correctly by checking invariant mass
        """    
        
        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':11,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True,
                                           'number': 4}))
        myleglist.append(base_objects.Leg({'id':23,
                                           'state':True,
                                           'number': 5}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        evaluator = process_checks.MatrixElementEvaluator(self.base_model)
        p, w_rambo = evaluator.get_momenta(myproc)

        def invariant_mass(p1, p2):
            #helping function to compute invariant mass
            return p1[0] * p2[0] - p1[1] * p2[1] - p1[2] * p2[2] -p1[3] * p2[3]

        # Compute invariant mass on the initial set of impulsion
        invariant_mass_result = []
        for p1 in p:
            for p2 in p: 
                m12 = invariant_mass(p1, p2)
                if abs(m12) < 1e-8:
                    m12=0
                invariant_mass_result.append(m12)
        
        # Compute invariant mass on a x direction boost
        invariant_mass_boost=[]
        p_boost = process_checks.boost_momenta(p)
        for p1 in p_boost:
            for p2 in p_boost: 
                m12 = invariant_mass(p1, p2)
                invariant_mass_boost.append(m12)  
                     
        for i in range(len(invariant_mass_boost)):
            self.assertAlmostEqual(invariant_mass_boost[i], 
                                   invariant_mass_result[i])

        # Compute invariant mass on a y direction boost
        invariant_mass_boost=[]
        p_boost = process_checks.boost_momenta(p, boost_direction=2)
        for p1 in p_boost:
            for p2 in p_boost: 
                m12 = invariant_mass(p1, p2)
                invariant_mass_boost.append(m12)  
                     
        for i in range(len(invariant_mass_boost)):
            self.assertAlmostEqual(invariant_mass_boost[i], 
                                   invariant_mass_result[i])
    
        # Compute invariant mass on a z direction boost
        invariant_mass_boost=[]
        p_boost = process_checks.boost_momenta(p, boost_direction=3, beta=0.8)
        for p1 in p:
            for p2 in p: 
                m12 = invariant_mass(p1, p2)
                invariant_mass_boost.append(m12)  
                     
        for i in range(len(invariant_mass_boost)):
            self.assertAlmostEqual(invariant_mass_boost[i], 
                                   invariant_mass_result[i])    

    def test_boost_momenta_gluino(self):
        """check if the momenta are boosted correctly by checking invariant mass
        in the case of massive final state
        """    
        
        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':1000021,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':1000021,
                                           'state':True,
                                           'number': 4}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        evaluator = process_checks.MatrixElementEvaluator(self.base_model)
        p, w_rambo = evaluator.get_momenta(myproc)

        def invariant_mass(p1, p2):
            #helping function to compute invariant mass
            return p1[0] * p2[0] - p1[1] * p2[1] - p1[2] * p2[2] -p1[3] * p2[3]

        # Compute invariant mass on the initial set of impulsion
        invariant_mass_result = []
        for p1 in p:
            for p2 in p: 
                m12 = invariant_mass(p1, p2)
                if abs(m12) < 1e-8:
                    m12=0
                invariant_mass_result.append(m12)
                self.assertGreaterEqual(m12, 0)
        
        # Compute invariant mass on a x direction boost
        invariant_mass_boost=[]
        p_boost = process_checks.boost_momenta(p)
        for p1 in p_boost:
            for p2 in p_boost: 
                m12 = invariant_mass(p1, p2)
                invariant_mass_boost.append(m12)  
                     
        for i in range(len(invariant_mass_boost)):
            self.assertAlmostEqual(invariant_mass_boost[i], 
                                   invariant_mass_result[i])
        
        # Compute invariant mass on a y direction boost
        invariant_mass_boost=[]
        p_boost = process_checks.boost_momenta(p, boost_direction=2)
        for p1 in p_boost:
            for p2 in p_boost: 
                m12 = invariant_mass(p1, p2)
                invariant_mass_boost.append(m12)  
                     
        for i in range(len(invariant_mass_boost)):
            self.assertAlmostEqual(invariant_mass_boost[i], 
                                   invariant_mass_result[i])
    
        # Compute invariant mass on a z direction boost
        invariant_mass_boost=[]
        p_boost = process_checks.boost_momenta(p, boost_direction=3, beta=0.8)
        for p1 in p_boost:
            for p2 in p_boost: 
                m12 = invariant_mass(p1, p2)
                invariant_mass_boost.append(m12)  
                     
        for i in range(len(invariant_mass_boost)):
            self.assertAlmostEqual(invariant_mass_boost[i], 
                                   invariant_mass_result[i])    



if __name__ == '__main__':
    unittest.unittest.main()
