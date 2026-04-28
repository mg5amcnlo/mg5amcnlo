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
import re
import subprocess
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
        
        self.base_model = import_ufo.import_model('sm', options={'apply_flavor_grouping':False})
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



class TestFlavorCheck(unittest.TestCase):
    """Test the flavor-grouping check (check_flavor / output_flavor)."""

    def setUp(self):
        self.merged_model = import_ufo.import_model(
            'sm', options={'apply_flavor_grouping': True})

    def _make_proc_def(self, is_ids, fs_ids):
        """Helper: build a ProcessDefinition with multi-legs."""
        legs = base_objects.MultiLegList(
            [base_objects.MultiLeg({'ids': list(is_ids), 'state': False})] * 1 +
            [base_objects.MultiLeg({'ids': list(is_ids), 'state': False})] * 0
        )
        # Build proper multi-leg list
        all_legs = base_objects.MultiLegList()
        for ids, state in [(is_id_list, False) for is_id_list in
                           ([list(i) if isinstance(i, (list, tuple)) else [i]
                             for i in is_ids])]:
            all_legs.append(base_objects.MultiLeg({'ids': ids, 'state': state}))
        for ids, state in [(list(i) if isinstance(i, (list, tuple)) else [i], True)
                           for i in fs_ids]:
            all_legs.append(base_objects.MultiLeg({'ids': ids, 'state': state}))
        return base_objects.ProcessDefinition({
            'legs': all_legs,
            'model': self.merged_model,
        })

    def test_check_flavor_qqbar_to_tt(self):
        """check_flavor for _quark _anti_quark > t t~ should pass all flavors."""
        # _quark has PDG 81 in the merged model
        q_id = 81
        t_id = 6
        proc_def = base_objects.ProcessDefinition({
            'legs': base_objects.MultiLegList([
                base_objects.MultiLeg({'ids': [q_id],  'state': False}),
                base_objects.MultiLeg({'ids': [-q_id], 'state': False}),
                base_objects.MultiLeg({'ids': [t_id],  'state': True}),
                base_objects.MultiLeg({'ids': [-t_id], 'state': True}),
            ]),
            'model': self.merged_model,
        })

        result = process_checks.check_flavor(proc_def, cmd=process_checks.FakeInterface())
        # Should have at least one entry (one per unmerged flavor)
        self.assertTrue(len(result) > 0,
                        "check_flavor returned no results for _quark _anti_quark > t t~")

        # Every result should pass (merged agrees with unmerged)
        fail_count = process_checks.output_flavor(result, output='fail')
        self.assertEqual(fail_count, 0,
                         "check_flavor: some flavor subprocesses disagree.\n" +
                         process_checks.output_flavor(result))

    def test_check_flavor_qqbar_to_zg(self):
        """check_flavor for _quark _anti_quark > z g (mixed merged+non-merged FS).

        This process has non-M gluon vertices mixed with M-tagged Z vertices.
        Flavor must be propagated through the non-M gluon off-shell wavefunction
        so the subsequent M-tagged Z amplitude receives the correct flavor.
        """
        q_id = 81
        z_id = 23
        g_id = 21
        proc_def = base_objects.ProcessDefinition({
            'legs': base_objects.MultiLegList([
                base_objects.MultiLeg({'ids': [q_id],  'state': False}),
                base_objects.MultiLeg({'ids': [-q_id], 'state': False}),
                base_objects.MultiLeg({'ids': [z_id],  'state': True}),
                base_objects.MultiLeg({'ids': [g_id],  'state': True}),
            ]),
            'model': self.merged_model,
        })

        result = process_checks.check_flavor(proc_def, cmd=process_checks.FakeInterface())
        self.assertTrue(len(result) > 0,
                        "check_flavor returned no results for _quark _anti_quark > z g")

        fail_count = process_checks.output_flavor(result, output='fail')
        self.assertEqual(fail_count, 0,
                         "check_flavor: some flavor subprocesses disagree.\n" +
                         process_checks.output_flavor(result))


    def test_check_language_proclist_merged_expansion(self):
        """check_language with a merged-particle ProcessDefinition.

        For ``_quark _anti_quark > t t~``, the SA binaries internally loop over
        all individual quark flavors.  check_language must therefore return one
        result entry per flavor combination (u u~ > t t~, d d~ > t t~, …)
        when at least one compiled backend is available, or exactly one
        placeholder entry when no backend can run.

        Each returned entry must:
        * carry the original merged-id Process object so callers can trace back
          which merged process it came from,
        * have a ``process_label`` string that includes '>' and the individual
          flavor particle names.
        * have Fortran and C++ values that agree within 1e-4 relative when both
          backends ran successfully.
        """
        q_id = 81
        t_id = 6
        proc_def = base_objects.ProcessDefinition({
            'legs': base_objects.MultiLegList([
                base_objects.MultiLeg({'ids': [q_id],  'state': False}),
                base_objects.MultiLeg({'ids': [-q_id], 'state': False}),
                base_objects.MultiLeg({'ids': [t_id],  'state': True}),
                base_objects.MultiLeg({'ids': [-t_id], 'state': True}),
            ]),
            'model': self.merged_model,
        })

        results = process_checks.check_language(
            proc_def,
            param_card=None,
            options=None,
            cmd=process_checks.FakeInterface(),
        )

        self.assertGreater(len(results), 0,
                           "Expected at least one result for merged process")

        for entry in results:
            # The original Process object must use the merged-particle ids.
            proc = entry['process']
            legs = proc.get('legs')
            is_legs = [l for l in legs if not l.get('state')]
            for leg in is_legs:
                self.assertEqual(abs(leg.get('id')), q_id,
                                 "IS leg should use merged id %d" % q_id)

            # Every entry must carry a process_label with a '>' separator.
            label = entry.get('process_label', '')
            self.assertIn('>', label,
                          "process_label should contain '>' separator: %r" % label)

            # When both backends ran they must agree within 1e-4 relative.
            vf  = entry['value_fortran']
            vc  = entry['value_cpp']
            if vf is not None and vc is not None:
                me_f   = vf['m2']
                me_cpp = vc['m2']
                ref = abs(me_f) if me_f != 0 else abs(me_cpp)
                if ref > 0:
                    rel = abs(me_f - me_cpp) / ref
                    self.assertLess(rel, 1e-4,
                                    "Fortran/C++ disagree for %s: F=%g C++=%g "
                                    "rel=%g" % (label, me_f, me_cpp, rel))

    def test_output_flavor_summary(self):
        """output_flavor returns a string with summary line."""
        q_id = 81
        t_id = 6
        proc_def = base_objects.ProcessDefinition({
            'legs': base_objects.MultiLegList([
                base_objects.MultiLeg({'ids': [q_id],  'state': False}),
                base_objects.MultiLeg({'ids': [-q_id], 'state': False}),
                base_objects.MultiLeg({'ids': [t_id],  'state': True}),
                base_objects.MultiLeg({'ids': [-t_id], 'state': True}),
            ]),
            'model': self.merged_model,
        })
        result = process_checks.check_flavor(proc_def, cmd=process_checks.FakeInterface())
        text = process_checks.output_flavor(result)
        self.assertIn('Summary:', text)
        self.assertIn('passed', text)



#===============================================================================
# TestMultiLanguageComparison
#===============================================================================
class TestMultiLanguageComparison(unittest.TestCase):
    """Compare Python, Fortran SA, and C++ SA matrix elements for the same
    phase-space point.  The strategy is:
      1. Generate and run the Fortran (or C++) standalone check binary.
      2. Parse the phase-space point and |M|^2 value printed by the binary.
      3. Evaluate the Python matrix element with those *same* momenta.
      4. Assert that both results agree to within a relative tolerance.
    Tests are skipped automatically if gfortran / g++ is not found.
    """

    @classmethod
    def setUpClass(cls):
        """Import heavy dependencies once and decide which back-ends exist."""
        import shutil as _shutil
        import madgraph.iolibs.export_v4 as _ev4
        import madgraph.iolibs.export_cpp as _ecpp
        import madgraph.iolibs.helas_call_writers as _hcw
        import madgraph.various.misc as _misc
        cls._ev4 = _ev4
        cls._ecpp = _ecpp
        cls._hcw = _hcw
        cls._misc = _misc
        cls._shutil = _shutil

        cls.has_fortran = bool(_misc.which('gfortran'))
        cls.has_cpp = bool(_misc.which('g++'))

        cls.model = import_ufo.import_model(
            'sm', options={'apply_flavor_grouping': False})

        # Build the HelasMatrixElement for e+ e- > a a once
        legs = base_objects.LegList([
            base_objects.Leg({'id': -11, 'state': False, 'number': 1}),
            base_objects.Leg({'id':  11, 'state': False, 'number': 2}),
            base_objects.Leg({'id':  22, 'state': True,  'number': 3}),
            base_objects.Leg({'id':  22, 'state': True,  'number': 4}),
        ])
        import madgraph.core.diagram_generation as _dg
        import madgraph.core.helas_objects as _ho
        proc = base_objects.Process({
            'legs': legs, 'model': cls.model,
            'orders': {}, 'forbidden_particles': [],
            'forbidden_onsh_s_channels': [],
            'forbidden_s_channels': [],
            'perturbation_couplings': [],
        })
        amplitude = _dg.Amplitude(proc)
        cls.matrix_element = _ho.HelasMatrixElement(amplitude, gen_color=True)

        # Ensure Template/LO/Source/make_opts exists (created on first MG5 run)
        import madgraph
        MG5DIR = madgraph.MG5DIR
        make_opts = os.path.join(MG5DIR, 'Template', 'LO', 'Source', 'make_opts')
        make_opts_src = os.path.join(MG5DIR, 'Template', 'LO', 'Source', '.make_opts')
        if not os.path.exists(make_opts) and os.path.exists(make_opts_src):
            _shutil.copy(make_opts_src, make_opts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    _me_re = re.compile(
        r"\sMatrix\selement\s=\s*(?P<val>-?\d*\.\d*(E[+-]?\d*)?)"
        r"\s*GeV\^\s*(?P<pow>-?\d+)", re.IGNORECASE | re.VERBOSE)
    _mom_re = re.compile(
        r"""\s*\d+\s+(?P<p0>-?\d*\.\d*E[+-]?\d*)\s+
             (?P<p1>-?\d*\.\d*E[+-]?\d*)\s+
             (?P<p2>-?\d*\.\d*E[+-]?\d*)\s+
             (?P<p3>-?\d*\.\d*E[+-]?\d*)""",
        re.IGNORECASE | re.VERBOSE)
    # Matches the "PDG  2  -2  6  -6" line written per flavor by Fortran SA.
    _pdg_re = re.compile(r'^\s*PDG\s+(.+)$', re.IGNORECASE)

    def _parse_sa_output(self, text, target_pdgs=None):
        """Return (me_value, [[E,px,py,pz], ...]) from check binary output.

        If *target_pdgs* is given the matrix-element value for the flavor
        whose preceding "PDG …" line matches those PDG codes is returned.
        If no block matches (e.g. because the process uses merged-particle
        legs whose PDG codes are flavor indices in the Fortran output), falls
        back to the last seen matrix-element value.
        If *target_pdgs* is not given the last value is returned (legacy).
        """
        momenta = []
        me_val = None
        last_val = None
        current_pdgs = None
        for line in text.split('\n'):
            mp = self._pdg_re.match(line)
            if mp:
                try:
                    current_pdgs = [int(x) for x in mp.group(1).split()]
                except ValueError:
                    current_pdgs = None
            m = self._me_re.match(line)
            if m:
                val = float(m.group('val'))
                last_val = val
                if target_pdgs is not None and current_pdgs is not None:
                    if current_pdgs == list(target_pdgs):
                        me_val = val
                else:
                    me_val = val
                current_pdgs = None
            m2 = self._mom_re.match(line)
            if m2:
                momenta.append([float(x) for x in m2.groups()[:4]])
        # Fallback: target was specified but no PDG block matched.
        if target_pdgs is not None and me_val is None:
            me_val = last_val
        return me_val, momenta

    def _run_check_binary(self, check_dir):
        """Compile and run ./check in *check_dir*, return stdout."""
        devnull = open(os.devnull, 'w')
        ret = subprocess.call(['make', 'check'], cwd=check_dir,
                              stdout=devnull, stderr=devnull)
        devnull.close()
        if ret != 0:
            self.skipTest('make check failed in %s' % check_dir)
        return subprocess.check_output(
            './check', cwd=check_dir, stderr=subprocess.STDOUT).decode()

    def _first_P_dir(self, sa_root):
        """Return the first P-directory found under sa_root/SubProcesses."""
        sub = os.path.join(sa_root, 'SubProcesses')
        for d in sorted(os.listdir(sub)):
            if d.startswith('P') and os.path.isdir(os.path.join(sub, d)):
                return os.path.join(sub, d)
        return None

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_parse_sa_output_flavor_selection(self):
        """_parse_sa_output correctly selects or falls back for target_pdgs.

        When the Fortran SA iterates over multiple flavors (e.g. u, d, b for a
        merged _quark leg) it prints one "PDG …" + "Matrix element = …" pair per
        flavor.  The function should:
        1. Return the ME for the block whose PDG line matches target_pdgs.
        2. Fall back to the last ME value when target_pdgs is given but no block
           matches.  This handles merged-particle processes where Fortran stores
           flavor *indices* (1, -1, …) rather than actual PDG codes.
        3. Return the last ME when target_pdgs is not given (legacy behaviour).
        """
        # Simulated Fortran SA output with three flavors:
        #   u (PDG 2 -2 6 -6)  → ME = 0.6177
        #   d (PDG 1 -1 6 -6)  → ME = 0.6178 (same coupling; differs slightly)
        #   b (PDG 5 -5 6 -6)  → ME = 0.5894 (b mass ≈ 4.7 GeV gives ~5% diff)
        fake_output = (
            "\n Phase space point:\n"
            "  1   5.0000000E+02  0.0000000E+00  0.0000000E+00  4.9000000E+02\n"
            "  2   5.0000000E+02  0.0000000E+00  0.0000000E+00 -4.9000000E+02\n"
            "  3   6.0000000E+02  1.0000000E+02  2.0000000E+02 -3.0000000E+02\n"
            "  4   4.0000000E+02 -1.0000000E+02 -2.0000000E+02  3.0000000E+02\n"
            " PDG           2          -2           6          -6\n"
            " Matrix element =  0.617700E+00 GeV^  -4\n"
            " -----------------------------------------------------------------------------\n"
            " PDG           1          -1           6          -6\n"
            " Matrix element =  0.617800E+00 GeV^  -4\n"
            " -----------------------------------------------------------------------------\n"
            " PDG           5          -5           6          -6\n"
            " Matrix element =  0.589400E+00 GeV^  -4\n"
            " -----------------------------------------------------------------------------\n"
        )

        # ── no target: backward-compat, returns last flavor (b quark) ──────
        me, p = self._parse_sa_output(fake_output)
        self.assertAlmostEqual(me, 0.5894, places=7,
                               msg='Without target_pdgs should return last ME')
        self.assertEqual(len(p), 4,
                         'Should parse 4 momentum vectors')

        # ── matching target: u quark → 0.6177 ───────────────────────────────
        me_u, _ = self._parse_sa_output(fake_output, target_pdgs=[2, -2, 6, -6])
        self.assertAlmostEqual(me_u, 0.6177, places=7,
                               msg='target_pdgs=[2,-2,6,-6] should return u ME')

        # ── matching target: d quark → 0.6178 ───────────────────────────────
        me_d, _ = self._parse_sa_output(fake_output, target_pdgs=[1, -1, 6, -6])
        self.assertAlmostEqual(me_d, 0.6178, places=7,
                               msg='target_pdgs=[1,-1,6,-6] should return d ME')

        # ── matching target: b quark → 0.5894 ───────────────────────────────
        me_b, _ = self._parse_sa_output(fake_output, target_pdgs=[5, -5, 6, -6])
        self.assertAlmostEqual(me_b, 0.5894, places=7,
                               msg='target_pdgs=[5,-5,6,-6] should return b ME')

        # ── sanity: b quark ME differs >1% from u/d (mass effect) ───────────
        rel_ub = abs(me_u - me_b) / abs(me_u)
        self.assertGreater(rel_ub, 0.01,
                           'u and b quark MEs should differ by >1%% due to b mass')

        # ── unmatched target: Fortran uses flavor indices for merged legs ────
        # In this scenario Fortran outputs PDG lines like "PDG  1  -1  6  -6"
        # (flavor *indices*) while proc.get('legs') contains the merged-particle
        # PDG code (e.g. 2000).  No block matches, so the function falls back
        # to the *last* ME value (0.5894 here) instead of returning None.
        me_merged, _ = self._parse_sa_output(fake_output,
                                             target_pdgs=[2000, -2000, 6, -6])
        self.assertAlmostEqual(me_merged, 0.5894, places=7,
                               msg='Unmatched target_pdgs should fall back to '
                                   'last ME, not return None')


    def test_python_vs_fortran_epem_aa(self):
        """e+ e- > a a: Python and Fortran SA must agree within 1e-4 rel."""
        if not self.has_fortran:
            self.skipTest('gfortran not available')

        import tempfile
        ev4 = self._ev4
        hcw = self._hcw
        misc = self._misc
        model = self.model
        me = self.matrix_element

        wl = misc.make_unique(me.get_used_lorentz())
        wc = misc.make_unique([c for l in me.get_used_couplings() for c in l])

        parent = tempfile.mkdtemp(prefix='mg5_test_f_')
        sa_dir = os.path.join(parent, 'sa_f')
        try:
            opt = {'sa_symmetry': False, 'export_format': 'standalone',
                   'mp': False, 'v5_model': True,
                   'output_options': {'noeps': 'True'}}
            exporter = ev4.ProcessExporterFortranSA(sa_dir, opt)
            fmodel = hcw.FortranUFOHelasCallWriter(model)
            exporter.copy_template(model)
            exporter.generate_subprocess_directory(me, fmodel, 0)
            exporter.convert_model(model, wl, wc)
            exporter.finalize({'matrix_elements': [me]}, '',
                              {'fortran_compiler': 'gfortran',
                               'cpp_compiler': 'g++',
                               'f2py_compiler': 'f2py',
                               'output_dependencies': 'external'},
                              ['nojpeg'])

            check_dir = self._first_P_dir(sa_dir)
            if check_dir is None:
                self.skipTest('No subprocess directory created')

            output = self._run_check_binary(check_dir)
            me_f, p_f = self._parse_sa_output(output)

            self.assertIsNotNone(me_f, 'Could not parse Fortran ME value')
            self.assertGreater(len(p_f), 0, 'Could not parse Fortran momenta')

            # Evaluate Python with the *same* phase-space point
            process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
            evaluator = process_checks.MatrixElementEvaluator(model)
            me_py, _ = evaluator.evaluate_matrix_element(me, p=p_f)

            self.assertGreater(abs(me_f), 0.,
                               'Fortran ME is zero – unexpected')
            rel_diff = abs(me_f - me_py) / abs(me_f)
            self.assertLess(rel_diff, 1e-4,
                            'Python and Fortran SA disagree: '
                            'Fortran=%g  Python=%g  rel_diff=%g'
                            % (me_f, me_py, rel_diff))
        finally:
            self._shutil.rmtree(parent, ignore_errors=True)

    def test_python_vs_cpp_epem_aa(self):
        """e+ e- > a a: Python and C++ SA must agree within 1e-4 rel."""
        if not self.has_cpp:
            self.skipTest('g++ not available')

        import tempfile
        ecpp = self._ecpp
        hcw = self._hcw
        misc = self._misc
        model = self.model
        me = self.matrix_element

        wl = misc.make_unique(me.get_used_lorentz())
        wc = misc.make_unique([c for l in me.get_used_couplings() for c in l])

        parent = tempfile.mkdtemp(prefix='mg5_test_cpp_')
        sa_dir = os.path.join(parent, 'sa_cpp')
        try:
            opt = {'sa_symmetry': False, 'export_format': 'standalone_cpp',
                   'mp': False, 'v5_model': True, 'cpp_compiler': 'g++'}
            exporter = ecpp.ProcessExporterCPP(sa_dir, opt)
            cpp_model = hcw.CPPUFOHelasCallWriter(model)
            exporter.copy_template(model)
            exporter.generate_subprocess_directory(me, cpp_model, 0)
            exporter.convert_model(model, wl, wc)
            exporter.finalize({'matrix_elements': [me]}, '',
                              {'fortran_compiler': 'gfortran',
                               'cpp_compiler': 'g++',
                               'f2py_compiler': 'f2py',
                               'output_dependencies': 'external'},
                              ['nojpeg'])

            check_dir = self._first_P_dir(sa_dir)
            if check_dir is None:
                self.skipTest('No subprocess directory created')

            output = self._run_check_binary(check_dir)
            me_cpp, p_cpp = self._parse_sa_output(output)

            self.assertIsNotNone(me_cpp, 'Could not parse C++ ME value')
            self.assertGreater(len(p_cpp), 0, 'Could not parse C++ momenta')

            # Evaluate Python with the *same* phase-space point
            process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
            evaluator = process_checks.MatrixElementEvaluator(model)
            me_py, _ = evaluator.evaluate_matrix_element(me, p=p_cpp)

            self.assertGreater(abs(me_cpp), 0.,
                               'C++ ME is zero – unexpected')
            rel_diff = abs(me_cpp - me_py) / abs(me_cpp)
            self.assertLess(rel_diff, 1e-4,
                            'Python and C++ SA disagree: '
                            'C++=%g  Python=%g  rel_diff=%g'
                            % (me_cpp, me_py, rel_diff))
        finally:
            self._shutil.rmtree(parent, ignore_errors=True)

    def test_check_language_function_epem_aa(self):
        """check_language() must return Passed for e+ e- > a a when at least
        one compiled backend (gfortran or g++) is available.

        The new check_language compares Fortran SA vs C++ SA only (no Python).
        Both backends use the same fixed energy so RAMBO produces identical
        momenta and the matrix elements can be compared directly.
        """
        model = self.model

        # Build a single Process (not a ProcessDefinition) so check_language
        # takes the non-ProcessDefinition branch.
        import madgraph.core.base_objects as _bo
        legs = _bo.LegList([
            _bo.Leg({'id': -11, 'state': False, 'number': 1}),
            _bo.Leg({'id':  11, 'state': False, 'number': 2}),
            _bo.Leg({'id':  22, 'state': True,  'number': 3}),
            _bo.Leg({'id':  22, 'state': True,  'number': 4}),
        ])
        proc = _bo.Process({
            'legs': legs, 'model': model,
            'orders': {}, 'forbidden_particles': [],
            'forbidden_onsh_s_channels': [],
            'forbidden_s_channels': [],
            'perturbation_couplings': [],
        })

        results = process_checks.check_language(proc)
        self.assertEqual(len(results), 1)

        entry = results[0]
        # New API: only Fortran and C++ backends (no Python).
        self.assertIn('value_fortran', entry)
        self.assertIn('value_cpp', entry)
        self.assertNotIn('value_python', entry)

        if not (self.has_fortran or self.has_cpp):
            self.skipTest('No compiled backend available')

        me_f   = entry['value_fortran']['m2'] if entry['value_fortran'] else None
        me_cpp = entry['value_cpp']['m2']      if entry['value_cpp']     else None

        # If both backends ran they must agree within 1e-4 relative.
        if me_f is not None and me_cpp is not None:
            ref = abs(me_f)
            rel = abs(me_f - me_cpp) / ref if ref > 0 else 0.
            self.assertLess(rel, 1e-4,
                            'Fortran/C++ disagree: F=%g C++=%g rel=%g'
                            % (me_f, me_cpp, rel))

        # output_language must at least contain a Summary line.
        text = process_checks.output_language(results)
        self.assertIn('Summary', text)

    def test_check_language_gg_ttx_cpp_ps_point(self):
        """g g > t t~: Fortran SA and C++ SA must agree within 1e-4 rel.

        Both backends receive the same energy argument so RAMBO generates
        identical momenta; the matrix-element values should therefore agree
        to machine precision (modulo compiler differences).
        """
        if not (self.has_fortran and self.has_cpp):
            self.skipTest('Both gfortran and g++ are required for this test')

        import madgraph.core.base_objects as _bo

        model = self.model
        legs = _bo.LegList([
            _bo.Leg({'id': 21, 'state': False, 'number': 1}),
            _bo.Leg({'id': 21, 'state': False, 'number': 2}),
            _bo.Leg({'id':  6, 'state': True,  'number': 3}),
            _bo.Leg({'id': -6, 'state': True,  'number': 4}),
        ])
        proc = _bo.Process({
            'legs': legs, 'model': model,
            'orders': {}, 'forbidden_particles': [],
            'forbidden_onsh_s_channels': [],
            'forbidden_s_channels': [],
            'perturbation_couplings': [],
        })

        results = process_checks.check_language(proc)
        self.assertEqual(len(results), 1, 'Expected exactly one subprocess result')

        entry = results[0]
        # New API: no Python key.
        self.assertNotIn('value_python', entry)
        self.assertIn('value_fortran', entry)
        self.assertIn('value_cpp', entry)

        if entry['value_fortran'] is None:
            self.skipTest('Fortran SA evaluation failed')
        if entry['value_cpp'] is None:
            self.skipTest('C++ SA evaluation failed (no g++ or compile error)')

        me_f   = entry['value_fortran']['m2']
        me_cpp = entry['value_cpp']['m2']

        self.assertGreater(abs(me_cpp), 0., 'C++ ME is zero – unexpected')
        rel_diff = abs(me_f - me_cpp) / abs(me_f)
        self.assertLess(rel_diff, 1e-4,
                        'Fortran and C++ SA disagree for g g > t t~: '
                        'F=%g  C++=%g  rel_diff=%g'
                        % (me_f, me_cpp, rel_diff))

        # output_language must also report Passed
        text = process_checks.output_language(results)
        self.assertIn('Passed', text,
                      'output_language reported failure for g g > t t~:\n' + text)


if __name__ == '__main__':
    unittest.unittest.main()
