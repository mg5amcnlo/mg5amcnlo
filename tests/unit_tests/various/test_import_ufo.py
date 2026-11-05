################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Unit test Library for importing and restricting model"""
from __future__ import division

from __future__ import absolute_import
import copy
import os
import sys
import time
from types import SimpleNamespace
from unittest import mock

import tests.unit_tests as unittest

import madgraph.interface.master_interface as Cmd
import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.models.import_ufo as import_ufo
import madgraph.models.model_reader as model_reader
import madgraph.iolibs.export_v4 as export_v4
import madgraph.models as ufomodels
import madgraph.various.misc as misc

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]


#===============================================================================
# TestImportUFO
#===============================================================================
class TestOrganizeModelExpression(unittest.TestCase):
    """A model's EW scheme and LOOP inputs must not affect later imports."""

    def setUp(self):
        defaults = mock.patch.object(import_ufo.OrganizeModelExpression,
                                    'track_dependant', ['aS', 'aEWM1', 'MU_R'])
        defaults.start()
        self.addCleanup(defaults.stop)

    def organize(self, gmu=False, loop_scale=False):
        def external(name, block, code):
            return SimpleNamespace(name=name, nature='external', type='real',
                                   value=1., lhablock=block, lhacode=[code])

        parameters = [external('Gf', 'SMINPUTS', 2),
                      external('custom_scale', 'LOOP' if loop_scale else 'INPUTS', 1)]
        if not gmu:
            parameters.append(external('aEWM1', 'SMINPUTS', 1))
        parameters += [SimpleNamespace(name=name, nature='internal', type='real',
                                       value=expression)
                       for name, expression in (('mass', '2*Gf'),
                                                ('scale', '2*custom_scale'))]
        model = SimpleNamespace(all_parameters=parameters, all_couplings=[])
        organizer = import_ufo.OrganizeModelExpression(model)
        organizer.main()
        return organizer

    def test_gmu_dependencies_do_not_leak_between_models(self):
        before = self.organize()
        gmu = self.organize(gmu=True)
        after = self.organize()
        self.assertEqual(before.all_expr['mass'].depend, ())
        self.assertEqual(gmu.all_expr['mass'].depend, ('Gf',))
        self.assertEqual(after.all_expr['mass'].depend, ())
        self.assertNotIn('Gf', import_ufo.OrganizeModelExpression.track_dependant)

    def test_loop_scale_dependencies_do_not_leak_between_models(self):
        before = self.organize()
        running = self.organize(loop_scale=True)
        after = self.organize()
        self.assertEqual(before.all_expr['scale'].depend, ())
        self.assertEqual(running.all_expr['scale'].depend, ('custom_scale',))
        self.assertEqual(after.all_expr['scale'].depend, ())
        self.assertNotIn('custom_scale',
                         import_ufo.OrganizeModelExpression.track_dependant)


class TestImportUFO(unittest.TestCase):
    """Test class for the RestrictModel object"""

    def setUp(self):
        """Set up decay model"""
        #Read the full SM
        sm_path = import_ufo.find_ufo_path('heft')
        self.base_model = import_ufo.import_full_model(sm_path)

    def test_coupling_hierarchy(self):
        """Test that the coupling_hierarchy is set"""
        self.assertEqual(self.base_model.get('order_hierarchy'),
                         {'QCD': 1, 'QED': 2, 'HIG':2, 'HIW': 2})
         
    def test_expansion_order(self):
        """Test that the expansion_order is set"""
        self.assertEqual(self.base_model.get('expansion_order'),
                         {'QCD': 99, 'QED': 99, 'HIG':1, 'HIW': 1})
        

    def test_get_symmetric_lorentz(self):

        import madgraph.models as ufomodels
        ufo_model = ufomodels.load_model(import_ufo.find_ufo_path('sm'), decay=False)
        ufo2mg5_converter = import_ufo.UFOMG5Converter(ufo_model)    
        model = ufo2mg5_converter.load_model()

        #sm = import_ufo.load_model(import_ufo.find_ufo_path('sm'), False)
        #obj = import_ufo.UFOMG5Converter(sm)
        #obj.load_model()
        old_lor = ufo2mg5_converter.get_symmetric_lorentz('VSS1', {}, change_number=False)
        self.assertEqual(old_lor.name, 'VSS1')
        self.assertEqual(old_lor.structure, 'P(1,2) - P(1,3)')        
        new_lor = ufo2mg5_converter.get_symmetric_lorentz('VSS1', {0: 1, 1: 2, 2: 0}, change_number=True)
        self.assertEqual(new_lor.name, 'SVS2')
        self.assertEqual(new_lor.structure, 'P(2,3) - P(2,1)')
        new_lor = ufo2mg5_converter.get_symmetric_lorentz('VSS1', {1: 2, 2: 1}, change_number=True)
        self.assertEqual(new_lor.name, 'VSS2')
        self.assertEqual(new_lor.structure, 'P(1,3) - P(1,2)')


        old_lor = ufo2mg5_converter.get_symmetric_lorentz('VVSS1', {}, change_number=False)
        self.assertEqual(old_lor.name, 'VVSS1')
        self.assertEqual(old_lor.structure, 'Metric(1,2)')     

        new_lor = ufo2mg5_converter.get_symmetric_lorentz('VVSS1', {0: 1, 1: 0, 2: 3, 3:2}, change_number=True)
        self.assertEqual(new_lor.name, 'VVSS2')
        self.assertEqual(new_lor.structure, 'Metric(2,1)')

        # here they are no need of a new lorentz
        new_lor = ufo2mg5_converter.get_symmetric_lorentz('VVSS1', {2: 3, 3:2}, change_number=True)
        self.assertEqual(new_lor.name, 'VVSS1')
        self.assertEqual(new_lor.structure, 'Metric(1,2)')

        # here flip Scalar and Vector
        # the exact index is not checked: the UFO module is global to the
        # process, so an equivalent SSVV lorentz can already exist (and be
        # returned) if another test did convert the sm model before this one
        new_lor = ufo2mg5_converter.get_symmetric_lorentz('VVSS1', {0: 3, 1:2,2: 1, 3:0}, change_number=True)
        self.assertRegex(new_lor.name, r'^SSVV\d+$')
        self.assertEqual(new_lor.structure, 'Metric(4,3)')

    def test_get_symmetric_color(self):
        """ """

        fct = import_ufo.UFOMG5Converter.get_symmetric_color

        output = fct(' 1 T(2,1,0)', {})
        self.assertEqual(output, ' 1 T(2,1,0)')

        output = fct(' 1 T(2,1,0)', {0:1, 1:0})
        self.assertEqual(output, ' 1 T(2,0,1)')

        output = fct(' 1 T(2,1,0)', {0:1, 1:2, 2:0})
        self.assertEqual(output, ' 1 T(0,2,1)')

        output = fct(' 1', {0:1, 1:2, 2:0})
        self.assertEqual(output, ' 1') 

    def test_reshape_FFV_coeff(self):
        """ test the possibility to reshape FFV vertex"""

        import madgraph.models as ufomodels
        ufo_model = ufomodels.load_model(import_ufo.find_ufo_path('sm'), decay=False)
        ufo2mg5_converter = import_ufo.UFOMG5Converter(ufo_model, FFV=False)    
        model = ufo2mg5_converter.load_model()

        fct = import_ufo.UFOMG5Converter.reshape_FFV_coeff

        def find_interaction(model, l1, l2=None):
            """find the interaction with the given lorentz structure""" 
            for interaction in model.get('interactions'):
                names = [l for l in interaction['lorentz']]
                if l1 in names:
                    if l2 is None and len(interaction['lorentz']) == 1:
                        return interaction
                    if l2 in names:
                        return interaction
            raise Exception('No interaction found')

        # check that only FFV are reshaped
        FFS = find_interaction(model, 'FFS4')
        output = fct(model, FFS)
        self.assertEqual(output, None)

        # check that Gamma(3,2,1) are not reshaped
        FFV1 = find_interaction(model, 'FFV1')
        output = fct(model, FFV1)
        self.assertEqual(output, None)

        Zdd = find_interaction(model, 'FFV2', 'FFV3')
        #assert Zdd['couplings'][(0,0)] == 'GC_40'
        #assert Zdd['couplings'][(0,1)] == 'GC_53'
        output = fct(model, Zdd)
        self.assertEqual(output, [(0,1, 0, 0), (-2,1,0,0)])

        Zuu = find_interaction(model, 'FFV2', 'FFV5')
        output = fct(model, Zuu)
        self.assertEqual(output, [(0,1,0,0), (4,1,0,0)]) 

        Zee = find_interaction(model, 'FFV2', 'FFV4') 
        output = fct(model, Zee)    
        self.assertEqual(output, [(0,1,0,0), (2,1,0,0)])

    def test_reshape_FFV_coeff_gamma5_and_vector(self):
        """test that Gamma5(1,-1)*Gamma(3,2,-1) and Gamma(3,2,1) are handled correctly"""

        import madgraph.models as ufomodels
        path = os.path.join(_file_path, '..', 'input_files', 'DM_pion')
        ufo_model = ufomodels.load_model(path, decay=False)
        ufo2mg5_converter = import_ufo.UFOMG5Converter(ufo_model, FFV=False)
        model = ufo2mg5_converter.load_model()

        fct = import_ufo.UFOMG5Converter.reshape_FFV_coeff

        def find_interaction(model, l1, l2=None):
            """find the interaction with the given lorentz structure"""
            for interaction in model.get('interactions'):
                names = [l for l in interaction['lorentz']]
                if l1 in names:
                    if l2 is None and len(interaction['lorentz']) == 1:
                        return interaction
                    if l2 is not None and l2 in names:
                        return interaction
            raise Exception('No interaction found with %s and %s' % (l1, l2))

        # Verify that DM_pion has the expected structures
        # FFV1 = Gamma(3,2,1), FFV2 = Gamma5(-1,1)*Gamma(3,2,-1)
        lor1 = model.get_lorentz('FFV1')
        lor2 = model.get_lorentz('FFV2')
        self.assertEqual(lor1.get('structure'), 'Gamma(3,2,1)')
        self.assertEqual(lor2.get('structure'), 'Gamma5(-1,1)*Gamma(3,2,-1)')

        # Test: interaction with Gamma(3,2,1) and Gamma5(-1,1)*Gamma(3,2,-1)
        # Gamma(3,2,1) -> (R=1, L=1), Gamma5*Gamma -> (R=1, L=-1)
        inter = find_interaction(model, 'FFV1', 'FFV2')
        output = fct(model, inter)
        self.assertEqual(output, [(1, 1, 0, 0), (1, -1, 0, 0)])

    def test_reshape_FFV_coeff_unknown_structure_ignored(self):
        """test that unrecognized FFV Lorentz structures are ignored (return None)
        instead of raising an exception, so the interaction is handled later
        in flavor merging (e.g. tensor operators from TopEffTh)"""

        import madgraph.core.base_objects as base_objects

        # Create a minimal mock model with an unrecognized FFV Lorentz structure
        # (e.g. a tensor/derivative coupling like P(3,1)*Gamma(-1,2,1))
        class MockLorentz:
            def __init__(self, name, spins, structure):
                self._d = {'name': name, 'spins': spins, 'structure': structure}
            def get(self, key):
                return self._d[key]

        class MockModel:
            def __init__(self, lors):
                self._d = {l.get('name'): l for l in lors}
            def get_lorentz(self, name):
                return self._d[name]

        fct = import_ufo.UFOMG5Converter.reshape_FFV_coeff

        # Test with one recognized (ProjM) and one unrecognized (tensor) FFV structure
        lor_known = MockLorentz('FFV_L', [2, 2, 3], 'Gamma(3,2,-1)*ProjM(-1,1)')
        lor_tensor = MockLorentz('FFV_T', [2, 2, 3], 'P(3,1)*Gamma(-1,2,1)')
        mock_model = MockModel([lor_known, lor_tensor])
        inter = base_objects.Interaction({
            'id': 1,
            'lorentz': ['FFV_L', 'FFV_T'],
            'couplings': {(0, 0): 'GC_1', (0, 1): 'GC_2'},
            'orders': {},
            'color': [],
            'particles': base_objects.ParticleList(),
        })
        output = fct(mock_model, inter)
        self.assertIsNone(output,
            "Expected None for unknown FFV Lorentz structure, got %s" % repr(output))

        # Test with an FFV structure using a sum that contains an unknown term
        lor_sum = MockLorentz('FFV_SUM', [2, 2, 3],
                              'Gamma(3,2,1) + P(-1,3)*P(3,1)*Gamma(-1,2,1)')
        lor_projm = MockLorentz('FFV_L2', [2, 2, 3], 'Gamma(3,2,-1)*ProjM(-1,1)')
        mock_model2 = MockModel([lor_sum, lor_projm])
        inter2 = base_objects.Interaction({
            'id': 2,
            'lorentz': ['FFV_SUM', 'FFV_L2'],
            'couplings': {(0, 0): 'GC_1', (0, 1): 'GC_2'},
            'orders': {},
            'color': [],
            'particles': base_objects.ParticleList(),
        })
        output2 = fct(mock_model2, inter2)
        self.assertIsNone(output2,
            "Expected None when a sum contains an unknown term, got %s" % repr(output2))







    def test_lorentz_info_cache_refresh(self):
        """the name -> lorentz cache used by the coupling merging must follow the
        model when new structures are added to it after the cache was built.
        In FD gauge load_model optimises once, then merge_all_goldstone_with_vector
        invents structures (SVS5, VSV2, ...) and optimises again: a frozen cache
        made that second pass raise KeyError (SMEFTatNLO, 2HDMtII_NLO, IDM_NLO)."""

        ufo_model = ufomodels.load_model(import_ufo.find_ufo_path('sm'), decay=False)
        converter = import_ufo.UFOMG5Converter(ufo_model)
        converter.load_model()

        converter.refresh_lorentz_info()
        self.assertIn('VSS1', converter.lorentz_info)

        # a structure created after the cache was built
        name = 'SVSTESTREFRESH'
        new_lor = converter.add_lorentz(name, [1, 3, 1], 'P(2,1) - P(2,3)')
        self.assertNotIn(name, converter.lorentz_info)
        self.assertIs(converter.get_lorentz_info(name), new_lor)
        self.assertEqual(converter.get_lorentz_info(name).get('spins'), [1, 3, 1])

        # a name the model really does not know about is reported as such
        self.assertIsNone(converter.get_lorentz_info('NOSUCHLORENTZ'))

    def test_optimise_iden_coup_lorentz_added_after_cache(self):
        """optimise_iden_coup merges two structures sharing a coupling even when
        one of them was added to the model after the cache was built."""

        ufo_model = ufomodels.load_model(import_ufo.find_ufo_path('sm'), decay=False)
        converter = import_ufo.UFOMG5Converter(ufo_model)
        converter.load_model()

        converter.add_lorentz('SVSTESTEARLY', [1, 3, 1], 'P(2,3)')
        converter.refresh_lorentz_info()
        converter.add_lorentz('SVSTESTLATE', [1, 3, 1], 'P(2,1)')
        # the cache is now stale exactly as FD gauge leaves it
        self.assertNotIn('SVSTESTLATE', converter.lorentz_info)

        inter = base_objects.Interaction({
            'id': 1,
            'lorentz': ['SVSTESTEARLY', 'SVSTESTLATE'],
            'couplings': {(0, 0): 'GC_1', (0, 1): 'GC_1'},
            'orders': {'QED': 1},
            'color': [],
            'particles': base_objects.ParticleList(),
            })
        converter.optimise_iden_coup(inter)

        # the two structures are replaced by their sum
        self.assertEqual(len(inter.get('couplings')), 1)
        merged = inter.get('lorentz')[list(inter.get('couplings'))[0][1]]
        self.assertEqual(converter.get_lorentz_info(merged).get('structure'),
                         'P(2,3) + P(2,1)')
        self.assertEqual(converter.get_lorentz_info(merged).get('spins'), [1, 3, 1])

    def test_optimise_iden_coup_unknown_lorentz_is_skipped(self):
        """a lorentz name the model does not define at all must leave the
        interaction untouched instead of raising."""

        ufo_model = ufomodels.load_model(import_ufo.find_ufo_path('sm'), decay=False)
        converter = import_ufo.UFOMG5Converter(ufo_model)
        converter.load_model()

        inter = base_objects.Interaction({
            'id': 1,
            'lorentz': ['VSS1', 'NOSUCHLORENTZ'],
            'couplings': {(0, 0): 'GC_1', (0, 1): 'GC_1'},
            'orders': {'QED': 1},
            'color': [],
            'particles': base_objects.ParticleList(),
            })
        converter.optimise_iden_coup(inter)

        self.assertEqual(inter.get('lorentz'), ['VSS1', 'NOSUCHLORENTZ'])
        self.assertEqual(inter.get('couplings'), {(0, 0): 'GC_1', (0, 1): 'GC_1'})


    def test_goldstone_merge_keeps_coupling_orders_apart(self):
        """A goldstone vertex must not be absorbed by a vector vertex that has
        different coupling orders: the merged coupling silently inherits the
        host's orders.  SMEFTatNLO lost its QED=2 'a a G- G+' coupling into an
        NP=2 'a a W+ G-' vertex that way, so 'a a > w+ w- NP=0' came out 4% off
        in FD gauge.  The orders were only checked when several candidate
        vertices existed."""

        def particle(pdg, name, is_part=True):
            return base_objects.Particle({'name': name, 'antiname': name,
                                          'pdg_code': abs(pdg), 'spin': 3,
                                          'is_part': is_part,
                                          'self_antipart': pdg == 22})

        a = particle(22, 'a')
        wp = particle(24, 'w+')
        wm = particle(24, 'w-', is_part=False)
        gm = particle(251, 'g-', is_part=False)

        def interaction(iid, parts, orders, couplings):
            return base_objects.Interaction({
                'id': iid,
                'particles': base_objects.ParticleList(parts),
                'lorentz': ['VVSS1'],
                'color': [color.ColorString()],
                'couplings': couplings,
                'orders': orders,
                })

        # the goldstone vertex is pure QED, the only candidate host is NP=2
        gold = interaction(1, [a, a, wp, gm], {'QED': 2}, {(0, 0): 'GC_6'})
        host = interaction(2, [a, a, wp, wm], {'NP': 2, 'QED': 2}, {})

        fct = import_ufo.UFOMG5Converter.update_vertex_for_goldstone
        # the guard returns before `self` is ever needed
        to_be_done = fct(None, [host], gold, gm, wm)

        self.assertTrue(to_be_done,
                        'the caller must build a standalone vertex instead')
        self.assertEqual(host.get('couplings'), {},
                         'the QED=2 coupling leaked into the NP=2 vertex')
        self.assertEqual(host.get('lorentz'), ['VVSS1'])

        # same orders: the merge does go ahead
        host_ok = interaction(3, [a, a, wp, wm], {'QED': 2}, {})
        self.assertFalse(fct(None, [host_ok], gold, gm, wm))
        self.assertEqual(host_ok.get('couplings'), {(0, 0): 'GC_6'})


    def test_goldstone_merge_is_atomic(self):
        """A goldstone vertex that cannot be absorbed whole must leave the host
        exactly as it found it.  Copying part of the couplings and *then*
        telling the caller to build a standalone vertex counted what had
        already been copied twice."""

        def particle(pdg, name, is_part=True):
            return base_objects.Particle({'name': name, 'antiname': name,
                                          'pdg_code': abs(pdg), 'spin': 3,
                                          'is_part': is_part,
                                          'self_antipart': pdg == 22})

        a, wp = particle(22, 'a'), particle(24, 'w+')
        wm, gm = particle(24, 'w-', False), particle(251, 'g-', False)

        def interaction(iid, parts, lorentz, couplings):
            return base_objects.Interaction({
                'id': iid, 'particles': base_objects.ParticleList(parts),
                'lorentz': lorentz, 'color': [color.ColorString()],
                'couplings': couplings, 'orders': {'QED': 2}})

        # the host already uses the slot the second structure would land in
        gold = interaction(1, [a, a, wp, gm], ['VVSS1', 'VVSS2'],
                           {(0, 0): 'GC_1', (0, 1): 'GC_2'})
        host = interaction(2, [a, a, wp, wm], ['VVSS1', 'VVSS2'],
                           {(0, 1): 'GC_9'})

        fct = import_ufo.UFOMG5Converter.update_vertex_for_goldstone
        self.assertTrue(fct(None, [host], gold, gm, wm),
                        'the caller must build a standalone vertex instead')
        self.assertEqual(host.get('couplings'), {(0, 1): 'GC_9'},
                         'the host was mutated even though the merge was refused')
        self.assertEqual(host.get('lorentz'), ['VVSS1', 'VVSS2'])

    def test_collapse_duplicate_lorentz(self):
        """Permuting the legs of a vertex with identical particles can send two
        of its structures onto the same one.  Two entries for one structure is
        not something the goldstone merge can use: collapse them and add the
        couplings up."""

        ufo_model = ufomodels.load_model(import_ufo.find_ufo_path('sm'), decay=False)
        converter = import_ufo.UFOMG5Converter(ufo_model)
        converter.load_model()

        inter = base_objects.Interaction({
            'id': 1, 'particles': base_objects.ParticleList(),
            'lorentz': ['VVSS1', 'VVSS2', 'VVSS1'],
            'color': [color.ColorString()],
            'couplings': {(0, 0): 'GC_1', (0, 1): 'GC_2', (0, 2): 'GC_3'},
            'orders': {'QED': 2}})
        converter.collapse_duplicate_lorentz(inter)

        self.assertEqual(inter.get('lorentz'), ['VVSS1', 'VVSS2'])
        self.assertEqual(sorted(inter.get('couplings')), [(0, 0), (0, 1)])
        self.assertEqual(inter.get('couplings')[(0, 1)], 'GC_2')
        # the two entries for VVSS1 became one carrying their sum
        summed = inter.get('couplings')[(0, 0)]
        self.assertNotIn(summed, ('GC_1', 'GC_3'))
        value = [c.value for c in converter.additional_couplings
                 if c.name == summed][0]
        expr = {c.name: c.value for c in ufo_model.all_couplings}
        self.assertEqual(value, '(%s)+(%s)' % (expr['GC_1'], expr['GC_3']))

        # a vertex with no repeat is left alone
        clean = base_objects.Interaction({
            'id': 2, 'particles': base_objects.ParticleList(),
            'lorentz': ['VVSS1', 'VVSS2'], 'color': [color.ColorString()],
            'couplings': {(0, 0): 'GC_1'}, 'orders': {'QED': 2}})
        converter.collapse_duplicate_lorentz(clean)
        self.assertEqual(clean.get('lorentz'), ['VVSS1', 'VVSS2'])
        self.assertEqual(clean.get('couplings'), {(0, 0): 'GC_1'})


    def test_parse_fermion_structure(self):
        """The chiral decomposition the goldstone phase is read from has to cope
        with however a model chose to write its two-fermion structures."""

        fct = import_ufo.parse_fermion_structure

        self.assertEqual(fct('Gamma(3,2,-1)*ProjM(-1,1)'), {'L': 1})
        self.assertEqual(fct('Gamma(3,2,1)'), {'L': 1, 'R': 1})
        self.assertEqual(fct('ProjM(2,1) - ProjP(2,1)'), {'M': 1, 'P': -1})
        # a model may write the neutral current as a single structure
        self.assertEqual(
            fct('Gamma(3,2,-1)*ProjM(-1,1) + 4*Gamma(3,2,-1)*ProjP(-1,1)'),
            {'L': 1, 'R': 4})
        # ... and the pseudoscalar coupling as a gamma5
        self.assertEqual(fct('Gamma5(2,1)'), {'P': 1, 'M': -1})
        # anything else is refused rather than guessed at
        self.assertIsNone(fct('P(3,1)*Gamma(-1,2,1)'))
        self.assertIsNone(fct('Gamma(3,2,1) + Sigma(1,2,3,4)'))
        self.assertIsNone(fct('Metric(1,2)'))

    @staticmethod
    def feynman_gauge_sm():
        """The sm converted with its goldstones still in place, which is what
        the phase is read off."""

        import aloha
        keep = aloha.unitary_gauge
        aloha.unitary_gauge = 0      # Feynman: the goldstones survive
        try:
            ufo_model = ufomodels.load_model(import_ufo.find_ufo_path('sm'),
                                             decay=False)
            converter = import_ufo.UFOMG5Converter(ufo_model)
            converter.load_model()
        finally:
            aloha.unitary_gauge = keep
        return converter

    def goldstone_pairs(self, converter):
        out = []
        for particle in converter.particles:
            if particle.get('type') != 'goldstone':
                continue
            vector = [p for p in converter.particles
                      if p.get('mass') == particle.get('mass') and p.get('spin') == 3]
            self.assertEqual(len(vector), 1)
            out.append((particle, vector[0]))
        return out

    def test_goldstone_phase_of_the_sm_is_trivial(self):
        """The SM UFO *is* the convention FD gauge assumes, so both of its
        goldstones must measure exactly one -- anything else would mean the
        measurement rotates a model that is already right."""

        converter = self.feynman_gauge_sm()
        pairs = self.goldstone_pairs(converter)
        self.assertEqual(sorted(g.get('name') for g, v in pairs), ['g+', 'g0'])
        for goldstone, vector in pairs:
            phase = converter.measure_goldstone_phase(goldstone, vector)
            self.assertIsNotNone(phase, 'no phase read for %s' % goldstone.get('name'))
            self.assertAlmostEqual(abs(phase - 1), 0, places=9,
                                   msg='%s measured %s' % (goldstone.get('name'), phase))

    def test_goldstone_phase_detects_a_rotated_convention(self):
        """Give the SM's charged goldstone another phase convention and the
        measurement has to find it.  This is what 2HDMtII_NLO (i) and
        SMEFTatNLO (-1 on the neutral one) look like."""

        converter = self.feynman_gauge_sm()
        charged = [(g, v) for g, v in self.goldstone_pairs(converter) if v.get('charge')]
        self.assertEqual(len(charged), 1)
        goldstone, vector = charged[0]

        # rotate every vertex holding the goldstone, the antiparticle one by the
        # conjugate so that the model stays hermitian
        for inter in converter.interactions:
            legs = [p.get_pdg_code() for p in inter.get('particles')
                    if abs(p.get_pdg_code()) == abs(goldstone.get_pdg_code())]
            if len(legs) != 1:
                continue
            factor = 1j if legs[0] > 0 else -1j
            inter.set('couplings', dict(
                (key, converter.rotate_coupling(name, factor))
                for key, name in inter.get('couplings').items()))
        for cache in ('_coupling_expr', '_coupling_values'):
            if hasattr(converter, cache):
                delattr(converter, cache)

        phase = converter.measure_goldstone_phase(goldstone, vector)
        self.assertIsNotNone(phase)
        self.assertAlmostEqual(abs(phase - (-1j)), 0, places=9,
                               msg='measured %s, expected -1j' % phase)

    def test_rotate_coupling(self):
        """A rotated coupling is a new coupling of the model, reused between
        the vertices that need the same rotation, and the identity is a no-op."""

        ufo_model = ufomodels.load_model(import_ufo.find_ufo_path('sm'), decay=False)
        converter = import_ufo.UFOMG5Converter(ufo_model)
        converter.load_model()

        self.assertEqual(converter.rotate_coupling('GC_1', 1), 'GC_1')
        self.assertEqual(converter.rotate_coupling('-GC_1', 1.0000000001), '-GC_1')

        rotated = converter.rotate_coupling('GC_1', 1j)
        self.assertNotEqual(rotated, 'GC_1')
        self.assertEqual(converter.rotate_coupling('GC_1', 1j), rotated)
        self.assertEqual(converter.rotate_coupling('-GC_1', 1j), '-' + rotated)

        expr = dict((c.name, c.value) for c in converter.additional_couplings)
        base = dict((c.name, c.value) for c in ufo_model.all_couplings)
        self.assertEqual(expr[rotated], '(complex(0,1))*(%s)' % base['GC_1'])
        # a different rotation is a different coupling
        self.assertNotEqual(converter.rotate_coupling('GC_1', -1), rotated)


    def test_goldstone_mass_mismatches(self):
        """A goldstone coupling carries a mass, and it has to be the one the
        particle propagates with.  heft ships ymb=4.2 against MB=4.7 and
        EWdim6NLO leaks a dim-6 shift into lam; neither is visible in unitary
        gauge, and both cost a few per mil in Feynman and FD."""

        def particle(name, pdg, spin, mass, charge=0., is_part=True):
            return base_objects.Particle({
                'name': name, 'antiname': name, 'pdg_code': abs(pdg),
                'spin': spin, 'mass': mass, 'is_part': is_part,
                'charge': charge, 'self_antipart': False})

        class Lorentz(object):
            def __init__(self, structure, spins):
                self._d = {'structure': structure, 'spins': spins}
            def get(self, key):
                return self._d[key]

        lorentz = {'FFV': Lorentz('Gamma(3,2,-1)*ProjM(-1,1)', [2, 2, 3]),
                   'FFVR': Lorentz('Gamma(3,2,-1)*ProjP(-1,1)', [2, 2, 3]),
                   'FFS': Lorentz('ProjM(2,1) - ProjP(2,1)', [2, 2, 1])}

        z = particle('z', 23, 3, 'MZ', charge=0.)
        b = particle('b', 5, 2, 'MB', charge=-1. / 3)
        bbar = particle('b~', 5, 2, 'MB', charge=1. / 3, is_part=False)

        # the merged vertex FD builds: the current and the goldstone coupling
        # of the same fermion pair, side by side
        inter = base_objects.Interaction({
            'id': 1, 'particles': base_objects.ParticleList([bbar, b, z]),
            'lorentz': ['FFV', 'FFVR', 'FFS'], 'color': [color.ColorString()],
            'couplings': {(0, 0): 'CL', (0, 1): 'CR', (0, 2): 'CS'},
            'orders': {'QED': 1}})

        class Model(object):
            def __init__(self, couplings):
                self.couplings = couplings
            def get(self, key):
                return {'coupling_dict': self.couplings,
                        'parameter_dict': {'MZ': 91.188, 'MB': 4.7},
                        'particles': [z, b, bbar],
                        'interactions': [inter]}[key]
            def get_lorentz(self, name):
                return lorentz[name]

        # gauge invariance puts -i m/M (cL - cR) on ProjM - ProjP
        axial = 0.37035403723587573
        couplings = {'CL': -0.31548078172104344j, 'CR': 0.054873255514832284j}

        couplings['CS'] = -axial * 4.7 / 91.188
        self.assertEqual(import_ufo.goldstone_mass_mismatches(Model(couplings)), [],
                         'a consistent model must be left alone')

        # now build the coupling with 4.2, the way heft does
        couplings['CS'] = -axial * 4.2 / 91.188
        found = import_ufo.goldstone_mass_mismatches(Model(couplings))
        self.assertEqual(len(found), 1)
        name, implied, actual, parameter, source = found[0]
        self.assertAlmostEqual(implied, 4.2, places=6)
        self.assertAlmostEqual(actual, 4.7, places=6)
        self.assertEqual(parameter, 'MB')
        self.assertEqual(source, 'z')


class TestImportUFO_fromcmd(unittest.TestCase):

    def test_import_from_cmd(self):
        """check that a model that defines "j" as a particle is correctly handle"""

        self.cmd = Cmd.MasterCmd()
        self.cmd.exec_cmd("import model sm") # important to trigger the bug
        self.assertIn("j", self.cmd._multiparticles)

        path = os.path.join(_file_path, '..', 'input_files', '231_Model_UFO')
        self.cmd.exec_cmd("import model %s" % path, postcmd=True, precmd=True)

        self.assertNotIn("j", self.cmd._multiparticles) 

    def test_fd_gauge_import(self):
        """check that the import of a model with FD gauge does not crash"""

        self.cmd = Cmd.MasterCmd()
        self.cmd.exec_cmd("import model sm") 
        self.cmd.exec_cmd("set gauge FD")


        qqz = [i for i  in self.cmd._curr_model.get('interactions') \
               if [p.get_pdg_code() for p in i.get('particles')] == [-81,81,23]]

        nb_lor = [0,0,0,0]
        for coup in qqz[0].get('couplings').keys():
            nb_lor[coup[1]] += 1

        self.assertEqual(nb_lor, [1,1,0,0])
        ttz = [i for i  in self.cmd._curr_model.get('interactions') \
               if [p.get_pdg_code() for p in i.get('particles')] == [-6,6,23]]

        # Pre-optimization in FD gauge converts Z-tbar-t from [FFV2,FFV5] to
        # [FFV6,FFV2,FFS3,FFS1] before goldstone merging, so FFS2 (goldstone)
        # is appended at index 4 rather than index 2.
        nb_lor = [0,0,0,0,0]
        for coup in ttz[0].get('couplings').keys():
            nb_lor[coup[1]] += 1

        self.assertEqual(nb_lor, [1,1,0,0,1])

    def test_fd_gauge_interaction_ids_stay_unique(self):
        """interaction ids are the key model.get_interaction() is looked up by,
        so they must stay unique.  In FD gauge merge_all_goldstone_with_vector
        shrinks the interaction list, and ids derived from its length were then
        handed out twice to the counterterm interactions of an NLO model --
        diagram generation ended up on the wrong vertex."""

        self.cmd = Cmd.MasterCmd()
        self.cmd.exec_cmd("set gauge FD")
        self.cmd.exec_cmd("import model loop_sm")

        interactions = self.cmd._curr_model.get('interactions')
        ids = [inter.get('id') for inter in interactions]
        self.assertEqual(len(ids), len(set(ids)),
                         'duplicated interaction ids in FD gauge')
        for inter in interactions:
            self.assertIs(self.cmd._curr_model.get_interaction(inter.get('id')),
                          inter)

        


class TestNFlav(unittest.TestCase):
    """Test class for the get_nflav function"""

    def test_get_nflav_sm(self):
        """Tests the get_nflav_function for the full SM.
        Here b and c quark are massive"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_full_model(sm_path)
        self.assertEqual(model.get_nflav(), 3)

    def test_get_nflav_sm_nobmass(self):
        """Tests the get_nflav_function for the SM, with the no-b-mass restriction"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_model(sm_path + '-no_b_mass')
        self.assertEqual(model.get_nflav(), 5)

    def test_get_nflav_sm_nomasses(self):
        """Tests the get_nflav_function for the SM, with the no_masses restriction"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_model(sm_path + '-no_masses')
        self.assertEqual(model.get_nflav(), 5)

class TestGetQuarkPDG(unittest.TestCase):
    """Test class for the get_nflav function"""

    def test_get_quark_pdgs_sm(self):
        """Tests the get_quark_pdg_function for the full SM.
        Here b and c quark are massive"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_full_model(sm_path)
        self.assertEqual(model.get_quark_pdgs(), [-3, -2, -1, 1, 2, 3])

    def test_get_quark_pdgs_sm_nobmass(self):
        """Tests the get_quark_pdg_function for the SM, with the no-b-mass restriction"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_model(sm_path + '-no_b_mass')
        self.assertEqual(model.get_quark_pdgs(), [-81,-5, -4, -3, -2, -1, 1, 2, 3, 4, 5,81])

    def test_get_quark_pdgs_sm_nomasses(self):
        """Tests the get_quark_pdg_function for the SM, with the no_masses restriction"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_model(sm_path + '-no_masses')
        self.assertEqual(model.get_quark_pdgs(), [-81,-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 81])

class TestNLeps(unittest.TestCase):
    """Test class for the get_nflav function"""

    def test_get_nleps_sm(self):
        """Tests the get_nleps_function for the full SM.
        Here all leptons have a mass"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_full_model(sm_path)
        self.assertEqual(model.get_nleps(), 0)

    def test_get_nleps_sm_nobmass(self):
        """Tests the get_nleps_function for the SM, with the no-b-mass restriction
        here the electron and muon are massless"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_model(sm_path + '-no_b_mass')
        self.assertEqual(model.get_nleps(), 2)

    def test_get_nleps_sm_nomasses(self):
        """Tests the get_nleps_function for the SM, with the no_masses restriction
        here the three leptons are massless"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_model(sm_path + '-no_masses')
        self.assertEqual(model.get_nleps(), 3)

class TestGetLuarkPDG(unittest.TestCase):
    """Test class for the get_nflav function"""

    def test_get_lepton_pdgs_sm(self):
        """Tests the get_lepton_pdg_function for the full SM.
        Here all leptons have a mass"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_full_model(sm_path)
        self.assertEqual(model.get_lepton_pdgs(), [])

    def test_get_lepton_pdgs_sm_nobmass(self):
        """Tests the get_lepton_pdg_function for the SM, with the no-b-mass restriction
        here the electron and muon are massless"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_model(sm_path + '-no_b_mass')
        self.assertEqual(model.get_lepton_pdgs(), [-82,-13, -11, 11, 13,82])

    def test_get_lepton_pdgs_sm_nomasses(self):
        """Tests the get_lepton_pdg_function for the SM, with the no_masses restriction
        here the three leptons are massless"""
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_model(sm_path + '-no_masses')
        self.assertEqual(model.get_lepton_pdgs(), [-82,-15, -13, -11, 11, 13, 15,82])

class TestImportUFONoSideEffect(unittest.TestCase):
    """Test class for the possible side effect on a UFO model loaded when
       converting it to a MG5 model"""

    def test_ImportUFONoSideEffectLO(self):
        """Checks that there are no side effects of the import of the LO UFO sm"""       
        ufo_model = ufomodels.load_model(import_ufo.find_ufo_path('sm'),False)
        original_all_particles = copy.copy(ufo_model.all_particles)
        original_all_vertices = copy.copy(ufo_model.all_vertices)
        original_all_couplings = copy.copy(ufo_model.all_couplings)
        original_all_lorentz = copy.copy(ufo_model.all_lorentz)
        original_all_parameters = copy.copy(ufo_model.all_parameters)
        original_all_orders = copy.copy(ufo_model.all_orders)
        original_all_functions = copy.copy(ufo_model.all_functions)

        ufo2mg5_converter = import_ufo.UFOMG5Converter(ufo_model)
        model = ufo2mg5_converter.load_model()
        # It is important to run import_ufo.OrganizeModelExpression(ufo_model).main() 
        # since this reverts some of the changes done in load_model()
        # There *is* side effects in-between, namely the expression of the CTcouplings
        # which contained CTparameters have been substituted to dictionaries.
        parameters, couplings = import_ufo.OrganizeModelExpression(ufo_model).main()        

        self.assertEqual(original_all_particles,ufo_model.all_particles)        
        self.assertEqual(original_all_vertices,ufo_model.all_vertices)
        self.assertEqual(original_all_couplings,ufo_model.all_couplings)
        self.assertEqual(original_all_lorentz,ufo_model.all_lorentz)
        self.assertEqual(original_all_parameters,ufo_model.all_parameters)
        self.assertEqual(original_all_orders,ufo_model.all_orders)
        self.assertEqual(original_all_functions,ufo_model.all_functions)

    def test_ImportUFOcheckgoldstone(self):
        """Check goldstone is correct in NLO UFO"""
        ufo_model = ufomodels.load_model(import_ufo.find_ufo_path('loop_qcd_qed_sm'),False)
        original_all_particles = copy.copy(ufo_model.all_particles)
        for part in original_all_particles:
            if part.name.lower() in ['g0','g+']:
                if hasattr(part,"goldstoneboson") and part.goldstoneboson:
                    pass
                elif hasattr(part,"GoldstoneBoson") and part.GoldstoneBoson:
                    pass
                elif hasattr(part,"goldstone") and part.goldstone:
                    pass
                else:
                    raise import_ufo.UFOImportError("Goldstone %s has no goldstone attribute set in loop_qcd_qed_sm"%part.name)
                    
        
    def test_ImportUFONoSideEffectNLO(self):
        """Checks that there are no side effects of the import of the NLO UFO sm"""
        ufo_model = ufomodels.load_model(import_ufo.find_ufo_path('loop_sm'),False)
        original_all_particles = copy.copy(ufo_model.all_particles)
        original_all_vertices = copy.copy(ufo_model.all_vertices)
        original_all_couplings = copy.copy(ufo_model.all_couplings)
        original_all_lorentz = copy.copy(ufo_model.all_lorentz)
        original_all_parameters = copy.copy(ufo_model.all_parameters)
        original_all_orders = copy.copy(ufo_model.all_orders)
        original_all_functions = copy.copy(ufo_model.all_functions)
        original_all_CTvertices = copy.copy(ufo_model.all_CTvertices)
        original_all_CTparameters = copy.copy(ufo_model.all_CTparameters)


        ufo2mg5_converter = import_ufo.UFOMG5Converter(ufo_model)
        model = ufo2mg5_converter.load_model()
        # It is important to run import_ufo.OrganizeModelExpression(ufo_model).main() 
        # since this reverts some of the changes done in load_model()
        # There *is* side effects in-between, namely the expression of the CTcouplings
        # which contained CTparameters have been substituted to dictionaries.
        parameters, couplings = import_ufo.OrganizeModelExpression(ufo_model).main()        

        self.assertEqual(original_all_particles,ufo_model.all_particles)
        self.assertEqual(original_all_vertices,ufo_model.all_vertices)
        self.assertEqual(original_all_couplings,ufo_model.all_couplings)
        self.assertEqual(original_all_lorentz,ufo_model.all_lorentz)
        self.assertEqual(original_all_parameters,ufo_model.all_parameters)
        self.assertEqual(original_all_orders,ufo_model.all_orders)
        self.assertEqual(original_all_functions,ufo_model.all_functions)
        self.assertEqual(original_all_CTvertices,ufo_model.all_CTvertices)
        self.assertEqual(original_all_CTparameters,ufo_model.all_CTparameters)

        # Also test that one new lorentz structure has been added within the model
        # and that the associate optimization is working as expected.
        self.assertEqual(len(original_all_lorentz) + 1, len(model['lorentz']))
        new_l = [l for l  in model['lorentz'] if l not in original_all_lorentz][0]
        new_name = new_l.name
        self.assertEqual(new_l.name, 'R2RGA_VVVV1')
        # find interactions with that lorentz structure
        int_with_it = []
        for id, vertices in model.get('interaction_dict').items():
            if new_name in vertices['lorentz']:
                int_with_it.append(vertices)
        self.assertEqual(len(int_with_it), 2)
        # check the first one
        vert = int_with_it[0]
        pdg = [p['pdg_code'] for p in vert['particles']]
        self.assertEqual(pdg, [21,21,21,21])
        # check the equivalent vertex in the original model
        old_vert = [ v for v in ufo_model.all_CTvertices if pdg == [p.pdg_code for p in v.particles]]
        #pick one
        old_vert = old_vert[0]
        
        # find the number of coupling associate to this lorentz structure
        ind = vert['lorentz'].index(new_name)
        coup_name = [ c for ((l,col),c) in vert['couplings'].items() if l ==ind]
        nb_old = len([ c for ((l,col),c) in vert['couplings'].items() if c == coup_name[0]])
        nb_new = len([ c for ((l,col,k),c) in old_vert.couplings.items() if c.name == coup_name[0]])
        self.assertEqual(3*nb_old, nb_new)

#===============================================================================
# TestRestrictModel
#===============================================================================
class TestRestrictModel(unittest.TestCase):
    """Test class for the RestrictModel object"""

    def setUp(self):
        """Set up decay model"""
        #Read the full SM
        sm_path = import_ufo.find_ufo_path('sm', )
        self.base_model = import_ufo.import_full_model(sm_path,  options={'apply_flavor_grouping':False})

        model = copy.deepcopy(self.base_model)
        self.model = import_ufo.RestrictModel(model)
        self.restrict_file = os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'restrict_sm.dat')
        self.model.set_parameters_and_couplings(self.restrict_file)
         
        
    def test_detect_special_parameters(self):
        """ check that detect zero parameters works"""        
        
        expected = set(['I3x32', 'etaWS', 'conjg__CKM3x2', 'CKM1x2', 'WT', 'I1x32', 'I1x33', 'I1x31', 'I2x32', 'CKM3x1', 'I2x13', 'I2x12', 'I3x23', 'I3x22', 'I3x21', 'conjg__CKM2x1', 'lamWS', 'conjg__CKM2x3', 'I2x23', 'AWS', 'CKM1x3', 'conjg__CKM3x1', 'I4x23', 'ymc', 'ymb', 'yme', 'CKM3x2', 'CKM2x3', 'CKM2x1', 'ymm', 'conjg__CKM1x3', 'Me', 'ym', 'I2x22', 'WTau', 'lamWS__exp__2', 'lamWS__exp__3', 'yc', 'yb', 'ye', 'MC', 'MB', 'MM', 'conjg__CKM1x2', 'I3x31', 'rhoWS', 'I4x33', 'I4x13'])
        zero, one = self.model.detect_special_parameters()
        result = set(zero)
        self.assertEqual(len(result), len(expected))

        self.assertEqual(expected, result)
        
        expected = set(['conjg__CKM3x3', 'conjg__CKM2x2', 'CKM1x1', 'CKM2x2', 'CKM3x3', 'conjg__CKM1x1'])
        result = set(one)
        self.assertEqual(expected, result)

        
        
    def test_detect_identical_parameters(self):
        """ check that we detect correctly identical parameter """
        
        expected=set([('MZ','MH')])
        result = self.model.detect_identical_parameters()
        result = [tuple([obj[0].name for obj in obj_list]) for obj_list in result]
        
        self.assertEqual(expected, set(result))
        
    def test_merge_identical_parameters(self):
        """check that we treat correctly the identical parameters"""
        
        parameters = self.model.detect_identical_parameters()
        self.model.merge_iden_parameters(parameters[0])
        
        
        #check that both MZ and MH are not anymore in the external_parameter
        keeped = '1*%s' % parameters[0][0][0].name
        removed = parameters[0][1][0].name
        for dep,data in self.model['parameters'].items():
            if dep == ('external'):
                for param in data:
                    self.assertNotEqual(param.name, removed)
            elif dep == ():
                found=0      
                for param in data:
                    if removed == param.name:
                        found += 1
                        self.assertEqual(param.expr, keeped)
                self.assertEqual(found, 1)
        
        # checked that the mass (and the width) of those particles identical
        self.assertEqual(self.model['particle_dict'][23]['mass'],
                         self.model['particle_dict'][25]['mass'])
        self.assertNotEqual(self.model['particle_dict'][23]['width'],
                         self.model['particle_dict'][25]['width'])
        

        
    def test_detect_zero_iden_couplings(self):
        """ check that detect zero couplings works"""
        
        zero, iden = self.model.detect_identical_couplings(allow_minus_coupling=True)
        
        # check what is the zero coupling
        expected = set(['GC_17', 'GC_16', 'GC_15', 'GC_14', 'GC_13', 'GC_19', 'GC_18', 'GC_22', 'GC_30', 'GC_20', 'GC_89', 'GC_88', 'GC_101', 'GC_102', 'GC_103', 'GC_42', 'GC_106', 'GC_107', 'GC_82', 'GC_43', 'GC_84', 'GC_85', 'GC_86', 'GC_105', 'GC_28', 'GC_29', 'GC_48', 'GC_44', 'GC_23', 'GC_46', 'GC_47', 'GC_26', 'GC_24', 'GC_25', 'GC_83', 'GC_87', 'GC_93', 'GC_92', 'GC_91', 'GC_90'])
        result = set(zero)
        self.assertEqual(len(expected), len(result))
        for name in result:
            self.assertEqual(self.model['coupling_dict'][name], 0)
        
        self.assertEqual(expected, result)        
        
        # check what are the identical coupling
        expected = [[('GC_100',1), ('GC_108',1), ('GC_49',1), ('GC_45',1), ('GC_40',1), ('GC_41',1), ('GC_104',1)],
                    [('GC_21', 1), ('GC_27', -1)],
                    [('GC_3', 1), ('GC_4', -1)],
                    [('GC_39', 1), ('GC_38', -1)],
                    [('GC_51', 1), ('GC_50', -1)],
                    #[('GC_53', 1), ('GC_52', -1)], #GC_52 is not assigned to a vertex to they are consider as different coupling order and not merged... not relevant anyway
                    [('GC_56', 1), ('GC_54', -1)],
                    [('GC_66', 1), ('GC_67', -1)],
                    [('GC_7', 1), ('GC_9', -1)],
                    [('GC_70', 1), ('GC_73', -1)],
                    [('GC_75', 1), ('GC_74', -1)],
                    [('GC_76', 1), ('GC_79', -1)],
                    [('GC_77', 1), ('GC_78', -1)],
                    [('GC_97', 1), ('GC_96', -1)]]
        expected = [[('GC_100',1), ('GC_108',1), ('GC_49',1), ('GC_45',1), ('GC_40',1), ('GC_41',1), ('GC_104',1)],
                    [('GC_21', 1), ('GC_27', -1)],
                    [('GC_3', 1), ('GC_4', -1)],
                    [('GC_38', 1), ('GC_39', -1)],
                    [('GC_50', 1), ('GC_51', -1)],
                    #[('GC_53', 1), ('GC_52', -1)], #GC_52 is not assigned to a vertex to they are consider as different coupling order and not merged... not relevant anyway
                    [('GC_54', 1), ('GC_56', -1)],
                    [('GC_66', 1), ('GC_67', -1)],
                    [('GC_68', 1), ('GC_80', 1)],
                    [('GC_7', 1), ('GC_9', -1)],
                    [('GC_70', 1), ('GC_73', -1)],
                    [('GC_74', 1), ('GC_75', -1)],
                    [('GC_76', 1), ('GC_79', -1)],
                    [('GC_77', 1), ('GC_78', -1)],
                    [('GC_96', 1), ('GC_97', -1)]]
        
        for elem in expected:
            elem.sort(key=str)
        for elem in iden:
            elem.sort(key=str)
        
        expected.sort(key=str)
        iden.sort(key=str)
        
        self.assertEqual(expected, iden)

    def test_locate_couplings(self):
        """ check the creation of the coupling to vertex dict """
        
        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [5, 5, 25]:
                input_bbh = candidate
                coupling_bbh = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [23, 23, 25, 25]:
                input_zzhh = candidate
                coupling_zzhh = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [11, 12, 24]:
                input_wen = candidate
                coupling_wen = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [22, 24, 24]:
                input_aww = candidate
                coupling_aww = candidate['couplings'][(0,0)]            
        
        
        target = [coupling_bbh, coupling_zzhh, coupling_wen, coupling_aww]
        sol = {coupling_bbh: [input_bbh['id']],
               coupling_zzhh: [input_zzhh['id']],
               coupling_wen: [43, 44, 45, 66, 67, 68],
               coupling_aww: [input_aww['id']]}
        # b b~ h // z z h h //w- e+ ve // a w+ w-
        
        self.model.locate_coupling()
        for coup in target:
            self.assertIn(coup, self.model.coupling_pos)
            self.assertEqual(sol[coup], [v['id'] for v in self.model.coupling_pos[coup]])

  
    def test_merge_iden_couplings(self):
        """ check that the merged couplings are treated correctly:
             suppression and replacement in the vertex (allow_minus_coupling=False) """
        
        self.model.locate_coupling()
        zero, iden = self.model.detect_identical_couplings()
        self.assertEqual(len(iden), 2)
        
        # Check that All the code/model is the one intended for this test
        target = [i for i in iden if len(i)==7][0] 
        target2 = [i[0] for i in target]
        GC = target2[0]
        
        check_content = [['d', 'u', 'w+'], ['s', 'c', 'w+'], ['b', 't', 'w+'], ['u', 'd', 'w+'], ['c', 's', 'w+'], ['t', 'b', 'w+'], ['e-', 've', 'w+'], ['m-', 'vm', 'w+'], ['tt-', 'vt', 'w+'], ['ve', 'e-', 'w+'], ['vm', 'm-', 'w+'], ['vt', 'tt-', 'w+']]
        content =  [[p.get('name') for p in v.get('particles')] \
               for v in self.model.get('interactions') \
               if any([c in target2 for c in v['couplings'].values()])]

        self.assertEqual(len(check_content),len(content))#, 'test not up-to-date'      

        vertex_id = [v.get('id') \
               for v in self.model.get('interactions') \
               if any([c in target2 for c in v['couplings'].values()])]


        for id in vertex_id:
            is_in_target = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                if coup in target2:
                    is_in_target = True
            assert is_in_target == True, 'test not up-to-date'
        
        # check now that everything is fine
        self.model.merge_iden_couplings(target)
        for id in vertex_id:
            has_GC = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                self.assertNotIn(coup, target[1:])
                if coup == GC:
                    has_GC = True
            self.assertTrue(has_GC, True)

    def test_merge_iden_couplings_with_minus(self):
        """ check that the merged couplings are treated correctly with allow_minus_coupling=True:
             suppression and replacement in the vertex, including opposite-sign couplings """
        
        self.model.locate_coupling()
        zero, iden = self.model.detect_identical_couplings(allow_minus_coupling=True)
        self.assertEqual(len(iden), 14)
        
        # Check that All the code/model is the one intended for this test
        target = [i for i in iden if len(i)==7][0] 
        target2 = [i[0] for i in target]
        GC = target2[0]
        
        check_content = [['d', 'u', 'w+'], ['s', 'c', 'w+'], ['b', 't', 'w+'], ['u', 'd', 'w+'], ['c', 's', 'w+'], ['t', 'b', 'w+'], ['e-', 've', 'w+'], ['m-', 'vm', 'w+'], ['tt-', 'vt', 'w+'], ['ve', 'e-', 'w+'], ['vm', 'm-', 'w+'], ['vt', 'tt-', 'w+']]
        content =  [[p.get('name') for p in v.get('particles')] \
               for v in self.model.get('interactions') \
               if any([c in target2 for c in v['couplings'].values()])]

        self.assertEqual(len(check_content),len(content))#, 'test not up-to-date'      

        vertex_id = [v.get('id') \
               for v in self.model.get('interactions') \
               if any([c in target2 for c in v['couplings'].values()])]


        for id in vertex_id:
            is_in_target = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                if coup in target2:
                    is_in_target = True
            assert is_in_target == True, 'test not up-to-date'
        
        # check now that everything is fine
        self.model.merge_iden_couplings(target)
        for id in vertex_id:
            has_GC = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                self.assertNotIn(coup, target[1:])
                if coup == GC:
                    has_GC = True
            self.assertTrue(has_GC, True)
        
        # check that the same occur with opposite sign coupling
        target = [i for i in sorted(iden) if len(i)==2][1] 
        target2 = [i[0] for i in target]
        GC = target2[0]
        
        check_content = [['a', 'w+', 'w+'], ['e-', 'e-', 'a'], ['mu-', 'mu-', 'a'], ['ta-', 'ta-', 'a']]
        content =  [[p.get('name') for p in v.get('particles')] \
               for v in self.model.get('interactions') \
               if any([c in target2 for c in v['couplings'].values()])]
        self.assertEqual(len(check_content),len(content))#, 'test not up-to-date'
        
        vertex_id = [v.get('id') \
               for v in self.model.get('interactions') \
               if any([c in target2[1:] for c in v['couplings'].values()])]
        
        for id in vertex_id:
            is_in_target = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                if coup in target2:
                    is_in_target = True
            assert is_in_target == True, 'test not up-to-date'
        
        self.model.merge_iden_couplings(target)
        for id in vertex_id:
            has_GC = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                self.assertNotIn(coup, target[1:])
                if coup == '-%s' % GC:
                    has_GC = True
            self.assertTrue(has_GC, True)
        

    def test_remove_couplings(self):
        """ check that the detection of irrelevant interactions works """
        
        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [5, 5, 25]:
                input_bbh = candidate
                coupling_bbh = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [21, 21, 21, 21]:
                input_4g = candidate
                coupling_4g = candidate['couplings'][(0,0)]
        
        found_bbh = 0
        found_4g = 0
        for dep,data in self.model['couplings'].items():
            for param in data:
                if param.name == coupling_bbh: found_bbh +=1
                elif param.name == coupling_4g: found_4g +=1
        self.assertGreater(found_bbh, 0)
        self.assertGreater(found_4g, 0)
        
        # make the real test
        result = self.model.remove_couplings([coupling_bbh,coupling_4g])
        
        for dep,data in self.model['couplings'].items():
            for param in data:
                self.assertNotIn(param.name, [coupling_bbh, coupling_4g])

             
    def test_remove_interactions(self):
        """ check that the detection of irrelevant interactions works """
        
        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [5, 5, 25]:
                input_bbh = candidate
                coupling_bbh = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [21, 21, 21, 21]:
                input_4g = candidate
                coupling_4g = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [1, 1, 23]:
                input_ddz = candidate
                coupling_ddz_1 = candidate['couplings'][(0,0)]
                coupling_ddz_2 = candidate['couplings'][(0,1)]
            if [p['pdg_code'] for p in candidate['particles']] == [11, 11, 23]:
                input_eez = candidate
                coupling_eez_1 = candidate['couplings'][(0,0)]            
                coupling_eez_2 = candidate['couplings'][(0,1)]
        
        #security                                      
        found_4g = 0  
        found_bbh = 0 
        for dep,data in self.model['couplings'].items():
            for param in data:
                if param.name == coupling_4g: found_4g +=1
                elif param.name == coupling_bbh: found_bbh +=1
        self.assertGreater(found_bbh, 0)
        self.assertGreater(found_4g, 0)
        
        # make the real test
        self.model.locate_coupling()
        result = self.model.remove_interactions([coupling_bbh, coupling_4g])
        self.assertNotIn(input_bbh, self.model['interactions'])
        self.assertNotIn(input_4g, self.model['interactions'])
        
    
        # Now test case where some of them are deleted and some not
        if coupling_ddz_1 != coupling_eez_1:
            coupling_eez_1, coupling_eez_2 = coupling_eez_2, coupling_eez_1
        assert coupling_ddz_1 == coupling_eez_1
        
        result = self.model.remove_interactions([coupling_ddz_1, coupling_ddz_2])
        self.assertIn(coupling_eez_2, list(input_eez['couplings'].values()))
        self.assertNotIn(coupling_eez_1, list(input_eez['couplings'].values()))
        self.assertNotIn(coupling_ddz_1, list(input_ddz['couplings'].values()))
        self.assertNotIn(coupling_ddz_2, list(input_ddz['couplings'].values()))

    def test_remove_interactions2(self):
        """ check that the detection of irrelevant interactions works """
        
        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [5, 5, 25]:
                input_bbh = candidate
                coupling_bbh = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [21, 21, 21, 21]:
                input_4g = candidate
                coupling_4g = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [1, 1, 23]:
                input_ddz = candidate
                coupling_ddz_1 = candidate['couplings'][(0,0)]
                coupling_ddz_2 = candidate['couplings'][(0,1)]
            if [p['pdg_code'] for p in candidate['particles']] == [11, 11, 23]:
                input_eez = candidate
                coupling_eez_1 = candidate['couplings'][(0,0)]            
                coupling_eez_2 = candidate['couplings'][(0,1)]
        
        #security                                      
        found_4g = 0  
        found_bbh = 0 
        for dep,data in self.model['couplings'].items():
            for param in data:
                if param.name == coupling_4g: found_4g +=1
                elif param.name == coupling_bbh: found_bbh +=1
        self.assertGreater(found_bbh, 0)
        self.assertGreater(found_4g, 0)
        
        # make the real test
        self.model.locate_coupling()
        #result = self.model.remove_interactions([coupling_bbh, coupling_4g])
        #self.assertNotIn(input_bbh, self.model['interactions'])
        #self.assertNotIn(input_4g, self.model['interactions'])
        
    
        # Now test case where some of them are deleted and some not
        if coupling_ddz_1 != coupling_eez_1:
            coupling_eez_1, coupling_eez_2 = coupling_eez_2, coupling_eez_1
        assert coupling_ddz_1 == coupling_eez_1
        
        result = self.model.remove_interactions([coupling_ddz_1])
        self.assertIn(coupling_eez_2, list(input_eez['couplings'].values()))
        self.assertNotIn(coupling_eez_1, list(input_eez['couplings'].values()))
        self.assertNotIn(coupling_ddz_1, list(input_ddz['couplings'].values()))
        self.assertIn(coupling_ddz_2, list(input_ddz['couplings'].values()))

        self.assertEqual(len(input_ddz['couplings']), 1)
        self.assertEqual(len(input_ddz['lorentz']), 1)
        self.assertEqual(list(input_ddz['couplings'].keys())[0], (0,0))

    def test_remove_interactions3(self):
        """ check that the detection of irrelevant interactions works """
        
        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [5, 5, 25]:
                input_bbh = candidate
                coupling_bbh = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [21, 21, 21, 21]:
                input_4g = candidate
                coupling_4g = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [1, 1, 23]:
                input_ddz = candidate
                coupling_ddz_1 = candidate['couplings'][(0,0)]
                coupling_ddz_2 = candidate['couplings'][(0,1)]
            if [p['pdg_code'] for p in candidate['particles']] == [11, 11, 23]:
                input_eez = candidate
                coupling_eez_1 = candidate['couplings'][(0,0)]            
                coupling_eez_2 = candidate['couplings'][(0,1)]
        
        #security                                      
        found_4g = 0  
        found_bbh = 0 
        for dep,data in self.model['couplings'].items():
            for param in data:
                if param.name == coupling_4g: found_4g +=1
                elif param.name == coupling_bbh: found_bbh +=1
        self.assertGreater(found_bbh, 0)
        self.assertGreater(found_4g, 0)
        
        # make the real test
        self.model.locate_coupling()
        #result = self.model.remove_interactions([coupling_bbh, coupling_4g])
        #self.assertNotIn(input_bbh, self.model['interactions'])
        #self.assertNotIn(input_4g, self.model['interactions'])
        
    
        # Now test case where some of them are deleted and some not
        if coupling_ddz_1 != coupling_eez_1:
            coupling_eez_1, coupling_eez_2 = coupling_eez_2, coupling_eez_1
        assert coupling_ddz_1 == coupling_eez_1
        
        result = self.model.remove_interactions([coupling_ddz_2])
        self.assertIn(coupling_eez_2, list(input_eez['couplings'].values()))
        self.assertIn(coupling_eez_1, list(input_eez['couplings'].values()))
        self.assertIn(coupling_ddz_1, list(input_ddz['couplings'].values()))
        self.assertNotIn(coupling_ddz_2, list(input_ddz['couplings'].values()))

        self.assertEqual(len(input_ddz['couplings']), 1)
        self.assertEqual(len(input_ddz['lorentz']), 1)
        self.assertEqual(list(input_ddz['couplings'].keys())[0], (0,0))


    def test_put_parameters_to_zero(self):
        """check that we remove parameters correctly"""
        
        part_t = self.model.get_particle(6)
        # Check that we remove a mass correctly
        self.assertEqual(part_t['mass'], 'MT')
        self.model.fix_parameter_values(['MT'],[])
        self.assertEqual(part_t['mass'], 'ZERO')
        for dep,data in self.model['parameters'].items():
            for param in data:
                self.assertNotEqual(param.name, 'MT')
        
        for particle in self.model['particles']:
            self.assertNotEqual(particle['mass'], 'MT')
                    
        for pdg, particle in self.model['particle_dict'].items():
            self.assertNotEqual(particle['mass'], 'MT')
        
        # Check that we remove a width correctly
        self.assertEqual(part_t['width'], 'WT')
        self.model.fix_parameter_values(['WT'],[])
        self.assertEqual(part_t['width'], 'ZERO')
        for dep,data in self.model['parameters'].items():
            for param in data:
                self.assertNotEqual(param.name, 'WT')

        for pdg, particle in self.model['particle_dict'].items():
            self.assertNotEqual(particle['width'], 'WT')       
             
        # Check that we can remove correctly other external parameter
        self.model.fix_parameter_values(['ymb','yb'],[])
        for dep,data in self.model['parameters'].items():
            for param in data:
                self.assertNotIn(param.name, ['ymb'])
                if param.name == 'yb':
                    param.expr == 'ZERO'
                    
    def test_get_new_coupling_name(self):
        """ test that the static function get_new_coupling_name
            behaves as expected
        """
        
        # reject wrong input
        self.assertRaises(AssertionError, import_ufo.RestrictModel.get_new_coupling_name,
                         '','','',0)
        self.assertRaises(AssertionError, import_ufo.RestrictModel.get_new_coupling_name,
                         '','','',2.)
        self.assertRaises(AssertionError, import_ufo.RestrictModel.get_new_coupling_name,
                         '','1','2',1)      
        self.assertRaises(AssertionError, import_ufo.RestrictModel.get_new_coupling_name,
                         '',1,1,1)         
        self.assertRaises(AssertionError, import_ufo.RestrictModel.get_new_coupling_name,
                         1,'1','1',1)
        
        # real test
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', 'GC2', 'GC2', 1), 'GC1')        
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', 'GC2', '-GC2', 1), '-GC1') 
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', '-GC2', 'GC2', 1), '-GC1') 
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', '-GC2', '-GC2', 1), 'GC1') 
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', 'GC2', 'GC2', 1), '-GC1') 
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', 'GC2', '-GC2', 1), 'GC1') 
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', '-GC2', 'GC2', 1), 'GC1') 
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', '-GC2', '-GC2', 1), '-GC1') 
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', 'GC2', 'GC2', -1), '-GC1') 
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', 'GC2', '-GC2', -1), 'GC1') 
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', '-GC2', 'GC2', -1), 'GC1') 
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', '-GC2', '-GC2', -1), '-GC1') 
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', 'GC2', 'GC2', -1), 'GC1') 
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', 'GC2', '-GC2', -1), '-GC1') 
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', '-GC2', 'GC2', -1), '-GC1') 
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', '-GC2', '-GC2', -1), 'GC1')
                   
    def test_restrict_from_a_param_card(self):
        """ check the full restriction chain in one case b b~ h """
        
        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [5, 5, 25]:
                interaction = candidate
                coupling = interaction['couplings'][(0,0)]
    

        self.model.restrict_model(self.restrict_file)

        # check remove interactions
        self.assertNotIn(interaction, self.model['interactions'])
        
        # check remove parameters
        for dep,data in self.model['parameters'].items():
            for param in data:
                self.assertNotIn(param.name, ['yb','ymb','MB','WT'])

        # check remove couplings
        for dep,data in self.model['couplings'].items():
            for param in data:
                self.assertNotIn(param.name, [coupling])

        # check masses
        part_b = self.model.get_particle(5)
        part_t = self.model.get_particle(6)
        self.assertEqual(part_b['mass'], 'ZERO')
        self.assertEqual(part_t['width'], 'ZERO')
                
        # check identical masses
        keeped, rejected = None, None 
        for param in self.model['parameters'][('external',)]:
            if param.name == 'MH':
                self.assertEqual(keeped, None)
                keeped, rejected = 'MH','MZ'
            elif param.name == 'MZ':
                self.assertEqual(keeped, None)
                keeped, rejected = 'MZ','MH'
                
        self.assertNotEqual(keeped, None)
        
        found = 0
        for param in self.model['parameters'][()]:
            self.assertNotEqual(param.name, keeped)
            if param.name == rejected:
                found +=1
        self.assertEqual(found, 1)
       
class TestBenchmarkModel(unittest.TestCase):
    """Test class for the RestrictModel object"""

    def setUp(self):
        """Set up decay model"""
        #Read the full SM
        sm_path = import_ufo.find_ufo_path('sm')
        self.base_model = import_ufo.import_full_model(sm_path)
        model = copy.deepcopy(self.base_model)
        self.model = import_ufo.RestrictModel(model)
        self.restrict_file = os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'restrict_sm.dat')
        
        
    def test_use_as_benchmark(self):
        """check that the value inside the restrict card overwrite the default
        parameter such that this option can be use for benchmark point"""
        
        params_ext = self.model['parameters'][('external',)]
        value = {}
        [value.__setitem__(data.name, data.value) for data in params_ext] 
        self.model.restrict_model(self.restrict_file)
        #use the UFO -> MG4 converter class
        
        params_ext = self.model['parameters'][('external',)]
        value2 = {}
        [value2.__setitem__(data.name, data.value) for data in params_ext] 
        
        self.assertNotEqual(value['WW'], value2['WW'])
                
    def test_model_name(self):
        """ test that the model name is correctly set """
        self.assertEqual(self.base_model["name"], "sm")
        model = import_ufo.import_model('sm-full') 
        self.assertEqual(model["name"], "sm-full")
        model = import_ufo.import_model('sm-no_b_mass') 
        self.assertEqual(model["name"], "sm-no_b_mass")        


#===============================================================================
# TestRestrictModel_Merged
#===============================================================================
class TestRestrictModel_Merged(unittest.TestCase):
    """Duplicate of TestRestrictModel with the model fully merged via restrict_model.
    Verifies that detection/manipulation methods behave correctly on an already-
    restricted model."""

    def setUp(self):
        """Set up fully restricted SM model"""
        sm_path = import_ufo.find_ufo_path('sm')
        self.base_model = import_ufo.import_full_model(sm_path, options={'apply_flavor_grouping': True})

        model = copy.deepcopy(self.base_model)
        self.model = import_ufo.RestrictModel(model)
        self.restrict_file = os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'restrict_sm.dat')
        # Apply the full restriction pipeline (merge identical couplings/parameters,
        # remove zero couplings/interactions, etc.)
        self.model.restrict_model(self.restrict_file)

    def test_detect_special_parameters(self):
        """check that detect zero parameters works on a merged model"""

        expected = set(['I3x32', 'etaWS', 'conjg__CKM3x2', 'CKM1x2', 'WT', 'I1x32', 'I1x33', 'I1x31', 'I2x32', 'CKM3x1', 'I2x13', 'I2x12', 'I3x23', 'I3x22', 'I3x21', 'conjg__CKM2x1', 'lamWS', 'conjg__CKM2x3', 'I2x23', 'AWS', 'CKM1x3', 'conjg__CKM3x1', 'I4x23', 'ymc', 'ymb', 'yme', 'CKM3x2', 'CKM2x3', 'CKM2x1', 'ymm', 'conjg__CKM1x3', 'Me', 'ym', 'I2x22', 'WTau', 'lamWS__exp__2', 'lamWS__exp__3', 'yc', 'yb', 'ye', 'MC', 'MB', 'MM', 'conjg__CKM1x2', 'I3x31', 'rhoWS', 'I4x33', 'I4x13'])
        zero, one = self.model.detect_special_parameters()
        result = set(zero)
        self.assertEqual(len(result), len(expected))
        self.assertEqual(expected, result)

        expected = set(['conjg__CKM3x3', 'conjg__CKM2x2', 'CKM1x1', 'CKM2x2', 'CKM3x3', 'conjg__CKM1x1'])
        result = set(one)
        self.assertEqual(expected, result)

    def test_detect_identical_parameters(self):
        """After restrict_model, no identical parameters remain to be detected"""

        result = self.model.detect_identical_parameters()
        # MH/MZ were already merged in setUp's restrict_model
        self.assertEqual(result, [])

    def test_merge_identical_parameters(self):
        """After restrict_model, identical parameters are already merged; check state"""

        # No identical parameters remain
        parameters = self.model.detect_identical_parameters()
        self.assertEqual(parameters, [])

        # One of MH/MZ is kept as external, the other is a derived parameter
        keeped, rejected = None, None
        for param in self.model['parameters'][('external',)]:
            if param.name == 'MH':
                self.assertIsNone(keeped)
                keeped, rejected = 'MH', 'MZ'
            elif param.name == 'MZ':
                self.assertIsNone(keeped)
                keeped, rejected = 'MZ', 'MH'
        self.assertIsNotNone(keeped)

        found = 0
        for param in self.model['parameters'][()]:
            self.assertNotEqual(param.name, keeped)
            if param.name == rejected:
                found += 1
                self.assertEqual(param.expr, '1*%s' % keeped)
        self.assertEqual(found, 1)

        # The Z and H particles should share the same mass parameter
        self.assertEqual(self.model['particle_dict'][23]['mass'],
                         self.model['particle_dict'][25]['mass'])
        self.assertNotEqual(self.model['particle_dict'][23]['width'],
                         self.model['particle_dict'][25]['width'])

    def test_detect_zero_iden_couplings(self):
        """check that detect zero/identical couplings works on a merged model"""

        zero, iden = self.model.detect_identical_couplings(allow_minus_coupling=True)

        expected = set(['GC_17', 'GC_16', 'GC_15', 'GC_14', 'GC_13', 'GC_19', 'GC_18', 'GC_22', 'GC_30', 'GC_20', 'GC_89', 'GC_88', 'GC_101', 'GC_102', 'GC_103', 'GC_42', 'GC_106', 'GC_107', 'GC_82', 'GC_43', 'GC_84', 'GC_85', 'GC_86', 'GC_105', 'GC_28', 'GC_29', 'GC_48', 'GC_44', 'GC_23', 'GC_46', 'GC_47', 'GC_26', 'GC_24', 'GC_25', 'GC_83', 'GC_87', 'GC_93', 'GC_92', 'GC_91', 'GC_90'])
        result = set(zero)
        self.assertEqual(len(expected), len(result))
        for name in result:
            self.assertEqual(self.model['coupling_dict'][name], 0)
        self.assertEqual(expected, result)

        expected = [[('GC_100',1), ('GC_108',1), ('GC_49',1), ('GC_45',1), ('GC_40',1), ('GC_41',1), ('GC_104',1)],
                    [('GC_21', 1), ('GC_27', -1)],
                    [('GC_3', 1), ('GC_4', -1)],
                    [('GC_38', 1), ('GC_39', -1)],
                    [('GC_50', 1), ('GC_51', -1)],
                    [('GC_54', 1), ('GC_56', -1)],
                    [('GC_66', 1), ('GC_67', -1)],
                    [('GC_68', 1), ('GC_80', 1)],
                    [('GC_7', 1), ('GC_9', -1)],
                    [('GC_70', 1), ('GC_73', -1)],
                    [('GC_74', 1), ('GC_75', -1)],
                    [('GC_76', 1), ('GC_79', -1)],
                    [('GC_77', 1), ('GC_78', -1)],
                    [('GC_96', 1), ('GC_97', -1)]]

        for elem in expected:
            elem.sort(key=str)
        for elem in iden:
            elem.sort(key=str)

        expected.sort(key=str)
        iden.sort(key=str)

        self.assertEqual(expected, iden)

    def test_locate_couplings(self):
        """check the coupling-to-vertex dict on a merged model.
        bbh is absent (removed by restrict); wen uses the merged coupling GC_100."""

        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [23, 23, 25, 25]:
                input_zzhh = candidate
                coupling_zzhh = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [11, 12, 24]:
                # GC_40..GC_49 were merged into GC_100 by restrict_model
                input_wen = candidate
                coupling_wen = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [22, 24, 24]:
                input_aww = candidate
                coupling_aww = candidate['couplings'][(0,0)]

        sol = {coupling_zzhh: [input_zzhh['id']],
               coupling_aww: [input_aww['id']]}

        self.model.locate_coupling()

        # zzhh and aww: each maps to exactly one vertex
        for coup in [coupling_zzhh, coupling_aww]:
            self.assertIn(coup, self.model.coupling_pos)
            self.assertEqual(sol[coup], [v['id'] for v in self.model.coupling_pos[coup]])

        # The merged wen coupling covers all 12 quark/lepton charged-current vertices
        self.assertIn(coupling_wen, self.model.coupling_pos)
        self.assertEqual(len(self.model.coupling_pos[coupling_wen]), 12)

    def test_merge_iden_couplings(self):
        """check that re-applying merge_iden_couplings on an already-merged model
        is a no-op and leaves the canonical coupling in place."""

        self.model.locate_coupling()
        zero, iden = self.model.detect_identical_couplings()
        self.assertEqual(len(iden), 2)

        target = [i for i in iden if len(i)==7][0]
        target2 = [i[0] for i in target]
        GC = target2[0]

        check_content = [['d', 'u', 'w+'], ['s', 'c', 'w+'], ['b', 't', 'w+'], ['u', 'd', 'w+'], ['c', 's', 'w+'], ['t', 'b', 'w+'], ['e-', 've', 'w+'], ['m-', 'vm', 'w+'], ['tt-', 'vt', 'w+'], ['ve', 'e-', 'w+'], ['vm', 'm-', 'w+'], ['vt', 'tt-', 'w+']]
        content =  [[p.get('name') for p in v.get('particles')] \
               for v in self.model.get('interactions') \
               if any([c in target2 for c in v['couplings'].values()])]

        self.assertEqual(len(check_content), len(content))

        vertex_id = [v.get('id') \
               for v in self.model.get('interactions') \
               if any([c in target2 for c in v['couplings'].values()])]

        for id in vertex_id:
            is_in_target = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                if coup in target2:
                    is_in_target = True
            assert is_in_target == True, 'test not up-to-date'

        # Re-applying merge_iden_couplings should be a no-op: canonical GC_100 stays
        self.model.merge_iden_couplings(target)
        for id in vertex_id:
            has_GC = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                self.assertNotIn(coup, target[1:])
                if coup == GC:
                    has_GC = True
            self.assertTrue(has_GC, True)

    def test_merge_iden_couplings_with_minus(self):
        """check that re-applying merge_iden_couplings (allow_minus_coupling=True)
        on an already-merged model works correctly for both same-sign and
        opposite-sign coupling groups."""

        self.model.locate_coupling()
        zero, iden = self.model.detect_identical_couplings(allow_minus_coupling=True)
        self.assertEqual(len(iden), 14)

        target = [i for i in iden if len(i)==7][0]
        target2 = [i[0] for i in target]
        GC = target2[0]

        check_content = [['d', 'u', 'w+'], ['s', 'c', 'w+'], ['b', 't', 'w+'], ['u', 'd', 'w+'], ['c', 's', 'w+'], ['t', 'b', 'w+'], ['e-', 've', 'w+'], ['m-', 'vm', 'w+'], ['tt-', 'vt', 'w+'], ['ve', 'e-', 'w+'], ['vm', 'm-', 'w+'], ['vt', 'tt-', 'w+']]
        content =  [[p.get('name') for p in v.get('particles')] \
               for v in self.model.get('interactions') \
               if any([c in target2 for c in v['couplings'].values()])]

        self.assertEqual(len(check_content), len(content))

        vertex_id = [v.get('id') \
               for v in self.model.get('interactions') \
               if any([c in target2 for c in v['couplings'].values()])]

        for id in vertex_id:
            is_in_target = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                if coup in target2:
                    is_in_target = True
            assert is_in_target == True, 'test not up-to-date'

        self.model.merge_iden_couplings(target)
        for id in vertex_id:
            has_GC = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                self.assertNotIn(coup, target[1:])
                if coup == GC:
                    has_GC = True
            self.assertTrue(has_GC, True)

        # check opposite-sign coupling group (GC_3 / GC_4)
        target = [i for i in sorted(iden) if len(i)==2][1]
        target2 = [i[0] for i in target]
        GC = target2[0]

        check_content = [['a', 'w+', 'w+'], ['e-', 'e-', 'a'], ['mu-', 'mu-', 'a'], ['ta-', 'ta-', 'a']]
        content =  [[p.get('name') for p in v.get('particles')] \
               for v in self.model.get('interactions') \
               if any([c in target2 for c in v['couplings'].values()])]
        self.assertEqual(len(check_content), len(content))

        vertex_id = [v.get('id') \
               for v in self.model.get('interactions') \
               if any([c in target2[1:] for c in v['couplings'].values()])]

        for id in vertex_id:
            is_in_target = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                if coup in target2:
                    is_in_target = True
            assert is_in_target == True, 'test not up-to-date'

        self.model.merge_iden_couplings(target)
        for id in vertex_id:
            has_GC = False
            for coup in self.model.get_interaction(id)['couplings'].values():
                self.assertNotIn(coup, target[1:])
                if coup == '-%s' % GC:
                    has_GC = True
            self.assertTrue(has_GC, True)

    def test_remove_couplings(self):
        """check that the detection of irrelevant interactions works (merged model).
        Uses GC_12 (4g) and GC_65 (zzhh) which survive restrict_model."""

        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [23, 23, 25, 25]:
                coupling_zzhh = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [21, 21, 21, 21]:
                coupling_4g = candidate['couplings'][(0,0)]

        found_zzhh = 0
        found_4g = 0
        for dep, data in self.model['couplings'].items():
            for param in data:
                if param.name == coupling_zzhh: found_zzhh += 1
                elif param.name == coupling_4g: found_4g += 1
        self.assertGreater(found_zzhh, 0)
        self.assertGreater(found_4g, 0)

        result = self.model.remove_couplings([coupling_zzhh, coupling_4g])

        for dep, data in self.model['couplings'].items():
            for param in data:
                self.assertNotIn(param.name, [coupling_zzhh, coupling_4g])

    def test_remove_interactions(self):
        """check that the detection of irrelevant interactions works (merged model).
        Uses GC_65 (zzhh) and GC_12 (4g); with apply_flavor_grouping=True, ddz
        uses GC_FFV_0/GC_FFV_1 while eez uses GC_FFV_2/GC_FFV_3 (different), so
        removing ddz couplings does not affect eez."""

        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [23, 23, 25, 25]:
                input_zzhh = candidate
                coupling_zzhh = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [21, 21, 21, 21]:
                input_4g = candidate
                coupling_4g = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [1, 1, 23]:
                input_ddz = candidate
                coupling_ddz_1 = candidate['couplings'][(0,0)]
                coupling_ddz_2 = candidate['couplings'][(0,1)]
            if [p['pdg_code'] for p in candidate['particles']] == [11, 11, 23]:
                input_eez = candidate
                coupling_eez_1 = candidate['couplings'][(0,0)]
                coupling_eez_2 = candidate['couplings'][(0,1)]

        found_4g = 0
        found_zzhh = 0
        for dep, data in self.model['couplings'].items():
            for param in data:
                if param.name == coupling_4g: found_4g += 1
                elif param.name == coupling_zzhh: found_zzhh += 1
        self.assertGreater(found_zzhh, 0)
        self.assertGreater(found_4g, 0)

        self.model.locate_coupling()
        result = self.model.remove_interactions([coupling_zzhh, coupling_4g])
        self.assertNotIn(input_zzhh, self.model['interactions'])
        self.assertNotIn(input_4g, self.model['interactions'])

        # With apply_flavor_grouping=True, ddz and eez have different couplings
        # (GC_FFV_0/1 vs GC_FFV_2/3), so removing ddz couplings leaves eez intact
        assert coupling_ddz_1 != coupling_eez_1, \
            'With flavor grouping, ddz and eez should have different couplings'

        result = self.model.remove_interactions([coupling_ddz_1, coupling_ddz_2])
        # eez is entirely unaffected
        self.assertIn(coupling_eez_1, list(input_eez['couplings'].values()))
        self.assertIn(coupling_eez_2, list(input_eez['couplings'].values()))
        # ddz lost both its couplings
        self.assertNotIn(coupling_ddz_1, list(input_ddz['couplings'].values()))
        self.assertNotIn(coupling_ddz_2, list(input_ddz['couplings'].values()))

    def test_remove_interactions2(self):
        """check that the detection of irrelevant interactions works (merged model).
        With apply_flavor_grouping=True, ddz uses GC_FFV_0/GC_FFV_1 and eez uses
        GC_FFV_2/GC_FFV_3; removing coupling_ddz_1 does not affect eez."""

        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [21, 21, 21, 21]:
                coupling_4g = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [1, 1, 23]:
                input_ddz = candidate
                coupling_ddz_1 = candidate['couplings'][(0,0)]
                coupling_ddz_2 = candidate['couplings'][(0,1)]
            if [p['pdg_code'] for p in candidate['particles']] == [11, 11, 23]:
                input_eez = candidate
                coupling_eez_1 = candidate['couplings'][(0,0)]
                coupling_eez_2 = candidate['couplings'][(0,1)]

        found_4g = 0
        for dep, data in self.model['couplings'].items():
            for param in data:
                if param.name == coupling_4g: found_4g += 1
        self.assertGreater(found_4g, 0)

        self.model.locate_coupling()

        # With flavor grouping, ddz and eez have independent couplings
        assert coupling_ddz_1 != coupling_eez_1, \
            'With flavor grouping, ddz and eez should have different couplings'

        result = self.model.remove_interactions([coupling_ddz_1])
        # eez fully unaffected
        self.assertIn(coupling_eez_1, list(input_eez['couplings'].values()))
        self.assertIn(coupling_eez_2, list(input_eez['couplings'].values()))
        # ddz: lost coupling_ddz_1, kept coupling_ddz_2
        self.assertNotIn(coupling_ddz_1, list(input_ddz['couplings'].values()))
        self.assertIn(coupling_ddz_2, list(input_ddz['couplings'].values()))

        self.assertEqual(len(input_ddz['couplings']), 1)
        self.assertEqual(len(input_ddz['lorentz']), 1)
        self.assertEqual(list(input_ddz['couplings'].keys())[0], (0,0))

    def test_remove_interactions3(self):
        """check that the detection of irrelevant interactions works (merged model).
        With apply_flavor_grouping=True, ddz uses GC_FFV_0/GC_FFV_1 and eez uses
        GC_FFV_2/GC_FFV_3; removing coupling_ddz_2 does not affect eez."""

        for candidate in self.model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [21, 21, 21, 21]:
                coupling_4g = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [1, 1, 23]:
                input_ddz = candidate
                coupling_ddz_1 = candidate['couplings'][(0,0)]
                coupling_ddz_2 = candidate['couplings'][(0,1)]
            if [p['pdg_code'] for p in candidate['particles']] == [11, 11, 23]:
                input_eez = candidate
                coupling_eez_1 = candidate['couplings'][(0,0)]
                coupling_eez_2 = candidate['couplings'][(0,1)]

        found_4g = 0
        for dep, data in self.model['couplings'].items():
            for param in data:
                if param.name == coupling_4g: found_4g += 1
        self.assertGreater(found_4g, 0)

        self.model.locate_coupling()

        # With flavor grouping, ddz and eez have independent couplings
        assert coupling_ddz_1 != coupling_eez_1, \
            'With flavor grouping, ddz and eez should have different couplings'

        result = self.model.remove_interactions([coupling_ddz_2])
        # eez fully unaffected
        self.assertIn(coupling_eez_1, list(input_eez['couplings'].values()))
        self.assertIn(coupling_eez_2, list(input_eez['couplings'].values()))
        # ddz: lost coupling_ddz_2, kept coupling_ddz_1
        self.assertIn(coupling_ddz_1, list(input_ddz['couplings'].values()))
        self.assertNotIn(coupling_ddz_2, list(input_ddz['couplings'].values()))

        self.assertEqual(len(input_ddz['couplings']), 1)
        self.assertEqual(len(input_ddz['lorentz']), 1)
        self.assertEqual(list(input_ddz['couplings'].keys())[0], (0,0))

    def test_put_parameters_to_zero(self):
        """check that we remove parameters correctly on a merged model"""

        part_t = self.model.get_particle(6)
        # top quark mass is still present (not zeroed by restrict_model)
        self.assertEqual(part_t['mass'], 'MT')
        self.model.fix_parameter_values(['MT'], [])
        self.assertEqual(part_t['mass'], 'ZERO')
        for dep, data in self.model['parameters'].items():
            for param in data:
                self.assertNotEqual(param.name, 'MT')

        for particle in self.model['particles']:
            self.assertNotEqual(particle['mass'], 'MT')

        for pdg, particle in self.model['particle_dict'].items():
            self.assertNotEqual(particle['mass'], 'MT')

        # top quark width was already zeroed by restrict_model
        self.assertEqual(part_t['width'], 'ZERO')
        for dep, data in self.model['parameters'].items():
            for param in data:
                self.assertNotEqual(param.name, 'WT')

        for pdg, particle in self.model['particle_dict'].items():
            self.assertNotEqual(particle['width'], 'WT')

        # ymb and yb were already removed by restrict_model
        for dep, data in self.model['parameters'].items():
            for param in data:
                self.assertNotIn(param.name, ['ymb', 'yb'])

    def test_get_new_coupling_name(self):
        """test that the static function get_new_coupling_name behaves as expected
        (same as non-merged variant; no model state used)"""

        self.assertRaises(AssertionError, import_ufo.RestrictModel.get_new_coupling_name,
                         '','','',0)
        self.assertRaises(AssertionError, import_ufo.RestrictModel.get_new_coupling_name,
                         '','','',2.)
        self.assertRaises(AssertionError, import_ufo.RestrictModel.get_new_coupling_name,
                         '','1','2',1)
        self.assertRaises(AssertionError, import_ufo.RestrictModel.get_new_coupling_name,
                         '',1,1,1)
        self.assertRaises(AssertionError, import_ufo.RestrictModel.get_new_coupling_name,
                         1,'1','1',1)

        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', 'GC2', 'GC2', 1), 'GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', 'GC2', '-GC2', 1), '-GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', '-GC2', 'GC2', 1), '-GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', '-GC2', '-GC2', 1), 'GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', 'GC2', 'GC2', 1), '-GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', 'GC2', '-GC2', 1), 'GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', '-GC2', 'GC2', 1), 'GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', '-GC2', '-GC2', 1), '-GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', 'GC2', 'GC2', -1), '-GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', 'GC2', '-GC2', -1), 'GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', '-GC2', 'GC2', -1), 'GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        'GC1', '-GC2', '-GC2', -1), '-GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', 'GC2', 'GC2', -1), 'GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', 'GC2', '-GC2', -1), '-GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', '-GC2', 'GC2', -1), '-GC1')
        self.assertEqual(import_ufo.RestrictModel.get_new_coupling_name(\
                        '-GC1', '-GC2', '-GC2', -1), 'GC1')

    def test_restrict_from_a_param_card(self):
        """check that applying restrict_model a second time is idempotent"""

        # Find the coupling for an interaction that was already removed
        coupling_bbh = None
        for candidate in self.base_model['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [5, 5, 25]:
                coupling_bbh = candidate['couplings'][(0,0)]
                break

        # Confirm the already-restricted state before second call
        self.assertNotIn(coupling_bbh,
                         [c.name for dep, data in self.model['couplings'].items()
                          for c in data])

        # Apply restriction a second time
        self.model.restrict_model(self.restrict_file)

        # bbh interaction must still be absent
        for candidate in self.model['interactions']:
            self.assertNotEqual([p['pdg_code'] for p in candidate['particles']],
                                [5, 5, 25])

        # Parameters removed by restrict must still be absent
        for dep, data in self.model['parameters'].items():
            for param in data:
                self.assertNotIn(param.name, ['yb', 'ymb', 'MB', 'WT'])

        # b quark mass and t quark width are ZERO
        part_b = self.model.get_particle(5)
        part_t = self.model.get_particle(6)
        self.assertEqual(part_b['mass'], 'ZERO')
        self.assertEqual(part_t['width'], 'ZERO')

        # MH/MZ: one is kept as external, the other remains a derived parameter
        keeped, rejected = None, None
        for param in self.model['parameters'][('external',)]:
            if param.name == 'MH':
                self.assertEqual(keeped, None)
                keeped, rejected = 'MH', 'MZ'
            elif param.name == 'MZ':
                self.assertEqual(keeped, None)
                keeped, rejected = 'MZ', 'MH'
        self.assertNotEqual(keeped, None)

        found = 0
        for param in self.model['parameters'][()]:
            self.assertNotEqual(param.name, keeped)
            if param.name == rejected:
                found += 1
        self.assertEqual(found, 1)



class TestLorentzStructureCanonicalisation(unittest.TestCase):
    """Sorting the arguments of the symmetric lorentz structures.

    Renumbering the indices of a vertex can reorder the arguments of a
    symmetric function, so that the same object is written Metric(3,2) in one
    definition and Metric(2,3) in another. import_ufo compares the two
    structures when a lorentz name is defined twice and warns when they
    disagree; without canonicalisation that warning fires on every such
    renumbering and hides the real disagreements among the noise.
    """

    def test_symmetry_is_carried_by_the_structure(self):
        """is_symmetric lives on the aloha object, and defaults to False."""
        import aloha.aloha_object as aloha_object
        import aloha.aloha_lib as aloha_lib
        self.assertFalse(aloha_lib.FactoryLorentz.is_symmetric)
        self.assertTrue(aloha_object.Metric.is_symmetric)
        self.assertFalse(aloha_object.Gamma.is_symmetric)
        self.assertTrue(import_ufo.is_symmetric_lorentz_structure('Metric'))
        self.assertFalse(import_ufo.is_symmetric_lorentz_structure('Gamma'))
        # an unknown name must not be taken for a symmetric structure
        self.assertFalse(import_ufo.is_symmetric_lorentz_structure('NotAThing'))

    def test_argument_order_of_a_symmetric_function_is_ignored(self):
        """The two spellings of one Metric compare equal."""
        canon = import_ufo.canonicalize_lorentz_structure
        # the two cases actually met when importing the sm model
        self.assertEqual(canon('Metric(3,2)'), canon('Metric(2,3)'))
        self.assertEqual(canon('Metric(4,2)'), canon('Metric(2,4)'))
        # summed indices are negative, and must sort numerically (not as text)
        self.assertEqual(canon('Metric(-1,2)'), canon('Metric(2,-1)'))
        # and inside a larger expression
        self.assertEqual(canon('Metric(1,2)*Gamma(3,4,5)'),
                         canon('Metric(2,1)*Gamma(3,4,5)'))

    def test_real_differences_are_still_reported(self):
        """Canonicalisation must not silence a genuine redefinition."""
        canon = import_ufo.canonicalize_lorentz_structure
        # different indices, not a reordering
        self.assertNotEqual(canon('Metric(1,2)'), canon('Metric(1,3)'))
        # Gamma and ProjP are NOT symmetric: reordering them stays a difference
        self.assertNotEqual(canon('Gamma(1,2,3)'), canon('Gamma(3,2,1)'))
        self.assertNotEqual(canon('ProjP(1,2)'), canon('ProjP(2,1)'))
        # a symmetric part that matches does not excuse an asymmetric part
        self.assertNotEqual(canon('Metric(1,2)*ProjM(3,4)'),
                            canon('Metric(2,1)*ProjM(4,3)'))


class TestRestrictionDoesNotLeakIntoTheUFO(unittest.TestCase):
    """A Lorentz structure the restriction merges must stay in that model.

    A UFO Lorentz registers itself in its object_library's `all_lorentz`, a
    module global that stays in sys.modules.  RestrictModel.add_lorentz left
    the merged structure there, so every later import of the same model in the
    process started with more structures (658, 661, 664 for SMEFTatNLO-NLO)
    and named its own merged ones one number further on -- which made
    customize_model's stability check refuse SMEFTatNLO-NLO outright.
    """

    def setUp(self):
        import types

        self.name = 'fake_ufo_object_library_for_restriction_test'
        library = types.ModuleType(self.name)
        exec('all_lorentz = []\n'
             'class Lorentz(object):\n'
             '    def __init__(self, name, spins, structure="external", **opt):\n'
             '        self.name = name\n'
             '        self.spins = spins\n'
             '        self.structure = structure\n'
             '        global all_lorentz\n'
             '        all_lorentz.append(self)\n', library.__dict__)
        sys.modules[self.name] = library
        self.library = library

    def tearDown(self):
        sys.modules.pop(self.name, None)

    def test_a_merged_structure_is_not_registered_in_the_ufo(self):
        original = self.library.Lorentz('FFVV1', [2, 2, 3, 3], 'Gamma(3,2,1)')
        model = import_ufo.RestrictModel()
        model['lorentz'] = [original]

        model.add_lorentz('FFVV99', [2, 2, 3, 3], 'Gamma(4,2,1)')

        self.assertIn('FFVV99', [l.name for l in model['lorentz']])
        self.assertEqual([l.name for l in self.library.all_lorentz],
                         ['FFVV1'])
