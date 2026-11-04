################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""Unit tests for the madmatrix ALOHA writer (madmatrix/model_handling.py)."""

from __future__ import absolute_import

import re

import tests.unit_tests as unittest


class TestMomentaInDenomPrecision(unittest.TestCase):
    """The TMPs that only involve momenta are computed from the momenta in
    denominator precision (double for FPTYPE=v), and a custom propagator
    denominator made of them is evaluated from those double versions."""

    def setUp(self):
        from madmatrix import model_handling
        # the helpers only need the two sets filled by define_expression
        self.writer = object.__new__(model_handling.MadMatrixALOHAWriter)
        self.writer.dp_needed = set()
        self.writer.pure_momentum_tmps = set()

    def test_momenta_only_expression(self):
        """p^2 and p.q are rewritten on dP, the constants on their d version,
        and the legs they need are recorded."""
        writer = self.writer
        self.assertEqual(
            writer.momenta_in_denom_precision(
                '( P1[0] * P1[0] - P1[1] * P1[1] - P1[2] * P1[2] - P1[3] * P1[3] )'),
            '( dP1[0] * dP1[0] - dP1[1] * dP1[1] - dP1[2] * dP1[2] - dP1[3] * dP1[3] )')
        self.assertEqual(writer.dp_needed, set([1]))
        self.assertEqual(
            writer.momenta_in_denom_precision('two * ( P2[0] * P3[0] + 0.5 * P2[1] * P3[1] )'),
            'twod * ( dP2[0] * dP3[0] + 0.5 * dP2[1] * dP3[1] )')
        self.assertEqual(writer.dp_needed, set([1, 2, 3]))

    def test_not_momenta_only(self):
        """A wavefunction, cI, a mass or no momentum at all: left alone, and no
        dP array is asked for."""
        writer = self.writer
        for expr in ('( wV3[0] * P1[0] - wV3[1] * P1[1] )',
                     '( P1[0] * P1[0] - cI * P1[3] )',
                     '( P1[0] * P1[0] - M1 * M1 )',
                     '( one + two )',
                     '( dP1[0] * dP1[0] )'):
            self.assertIsNone(writer.momenta_in_denom_precision(expr), expr)
        self.assertEqual(writer.dp_needed, set())

    def test_loop_mode_untouched(self):
        """Loop mode has complex momenta: nothing is rewritten."""
        import aloha
        old = aloha.loop_mode
        aloha.loop_mode = True
        try:
            self.assertIsNone(self.writer.momenta_in_denom_precision('( P1[0] * P1[0] )'))
        finally:
            aloha.loop_mode = old

    def test_denominator_momentum_tmps(self):
        """Only the TMPs of a custom denominator made of momenta-only TMPs are
        selected; a numerator-only momenta TMP, a denominator involving anything
        else, a standard propagator or an amplitude routine select nothing."""
        import types
        writer = self.writer
        writer.write_obj = lambda obj: obj          # contracted holds C++ text here
        writer.offshell = 1
        writer.tag = []
        contracted = {'TMP9': '( P1[0] * P1[0] - P1[1] * P1[1] )',
                      'TMP10': '( P1[0] * P1[0] + P1[1] * P1[1] )',
                      'TMP13': '( P1[0] * P2[0] - P1[1] * P2[1] )',   # numerator only
                      'TMP6': '( wV3[0] * wV2[0] - wV3[1] * wV2[1] )'}

        def select(denominator, offshell=1, tag=()):
            writer.routine = types.SimpleNamespace(denominator=denominator,
                                                   contracted=contracted)
            writer.offshell = offshell
            writer.tag = list(tag)
            return writer.denominator_momentum_tmps()

        self.assertEqual(select('(TMP9 * TMP10)'), set(['TMP9', 'TMP10']))
        self.assertEqual(select('(TMP9 * TMP6)'), set())          # a wavefunction TMP
        self.assertEqual(select('(TMP9 - M1 * M1)'), set())       # a mass
        self.assertEqual(select('(TMP9 * TMP99)'), set())         # unknown TMP
        self.assertEqual(select(None), set())                     # standard propagator
        self.assertEqual(select('1'), set())
        self.assertEqual(select('(TMP9 * TMP10)', offshell=0), set())
        self.assertEqual(select('(TMP9 * TMP10)', tag=['L']), set())

    def test_custom_denominator(self):
        """The axial gauge denominator (P.PBar) P^2 uses the double TMPs; a
        denominator with anything else keeps the amplitude precision ones."""
        writer = self.writer
        writer.pure_momentum_tmps = set(['TMP9', 'TMP10'])
        self.assertEqual(writer.denominator_in_denom_precision('(TMP9 * TMP10)'),
                         '(dTMP9 * dTMP10)')
        self.assertEqual(writer.denominator_in_denom_precision('(half * TMP9 * TMP10)'),
                         '(halfd * dTMP9 * dTMP10)')
        for denominator in ('(TMP9 * TMP11)',              # TMP11 is not momenta only
                            '(TMP9 - M1 * M1)',            # a mass
                            '(TMP9 + cI * M1 * W1)',       # a width
                            '1'):                          # no TMP at all
            self.assertEqual(writer.denominator_in_denom_precision(denominator),
                             denominator)


class TestExternalWavefunctionCalls(unittest.TestCase):
    """The external wavefunctions are built with the generic HELAS routines,
    never with the z-axis-optimised ones.

    ipzxxx/imzxxx/ixzxxx (and their o counterparts) assume px == py == 0 and
    E == +-pz for the two incoming legs. That only holds in a frame where the
    beams run along z, and the matrix element is called on momenta boosted into
    whatever frame the run card's me_frame selects -- the rest frame of a single
    final-state particle, say, where the incoming partons point nowhere in
    particular. Using them would silently give the wrong amplitude there.
    """

    # the z-optimised variants defined in HelAmps_<model>.h
    OPTIMISED = ('ipzxxx', 'imzxxx', 'ixzxxx', 'opzxxx', 'omzxxx', 'oxzxxx')

    def external_calls(self, ids):
        """The HELAS calls madmatrix writes for the external wavefunctions of
        the process with the given (flavour grouped) particle ids."""
        import madgraph.core.base_objects as base_objects
        import madgraph.core.diagram_generation as diagram_generation
        import madgraph.core.helas_objects as helas_objects
        import madgraph.models.import_ufo as import_ufo
        from madmatrix import model_handling

        model = import_ufo.import_model('sm')
        legs = base_objects.LegList(
            [base_objects.Leg({'id': pdg, 'state': i > 1})
             for i, pdg in enumerate(ids)])
        amplitude = diagram_generation.Amplitude(
            base_objects.Process({'legs': legs, 'model': model}))
        self.assertTrue(amplitude.get('diagrams'), ids)
        matrix_element = helas_objects.HelasMatrixElement(amplitude)
        writer = model_handling.MadMatrixUFOHelasCallWriter(model)
        return [writer.get_wavefunction_call(wf)
                for wf in matrix_element.get_all_wavefunctions()
                if not wf.get('mothers')]

    def test_massless_fermions_use_the_generic_routine(self):
        """A massless quark in the initial state is exactly the case the base
        GPUFOHelasCallWriter optimises away; madmatrix must not."""
        # 81/-81 are the flavour-grouped light quarks, the mg7 model groups them
        calls = self.external_calls([81, -81, 23, 21])   # q q~ > z g
        self.assertEqual(len(calls), 4)
        self.assertTrue(calls[0].startswith('ixxxxx<'), calls[0])
        self.assertTrue(calls[1].startswith('oxxxxx<'), calls[1])
        for call in calls:
            for name in self.OPTIMISED:
                self.assertNotIn(name, call)

    def test_no_optimised_call_anywhere(self):
        for ids in ([81, -81, 21, 21],      # q q~ > g g
                    [21, 21, 21, 21],       # g g > g g
                    [81, -81, 23, 21]):     # q q~ > z g
            for call in self.external_calls(ids):
                for name in self.OPTIMISED:
                    self.assertNotIn(name, call, ids)

    def test_helicity_is_per_lane(self):
        """The external wavefunctions take the helicity of each SIMD lane/GPU
        thread (hel_sv, built in calculate_jamps), not one helicity for all
        events, so that events with different helicities run in lock-step."""
        calls = self.external_calls([81, -81, 23, 21])   # q q~ > z g
        for ipar, call in enumerate(calls):
            self.assertIn('hel_sv[%d]' % ipar, call)
            self.assertNotIn('cHel[', call)
            
class TestPropagatorDenominators(unittest.TestCase):
    """The propagator factor 'denom' of an off-shell routine.

    A standard propagator calls madmatrix::denom, which computes only
    P^2 - Re(m^2) in denominator precision (double for FPTYPE=v) and divides
    at vertex precision; the coupling goes in with it. A routine assembled
    from several Lorentz structures has one coupling per term, in the
    numerator, so madmatrix::denom gets none. In the complex mass scheme the
    mass is complex and carries the width. A custom denominator (axial gauge)
    is still divided in denominator precision, on the momenta-only TMPs.
    """

    @classmethod
    def setUpClass(cls):
        import madgraph.models.import_ufo as import_ufo
        cls.model = import_ufo.import_model('sm', options={'apply_flavor_grouping': False})

    def routines(self, request, complex_mass=False, gauge=True):
        """{routine name: C++ definition} of the ALOHA routines of *request*
        (the compute_subset format: Lorentz names, tags, off-shell leg) as
        madmatrix writes them; *gauge* is aloha.unitary_gauge (2 is axial)."""
        import aloha
        import aloha.create_aloha as create_aloha
        from madmatrix import model_handling
        saved = (aloha.complex_mass, aloha.unitary_gauge, aloha.loop_mode)
        aloha.complex_mass, aloha.unitary_gauge = complex_mass, gauge
        try:
            # explicit_combine as in write_aloha_routines outside of the FD gauge
            aloha_model = create_aloha.AbstractALOHAModel.from_model(
                self.model, explicit_combine=True)
            aloha_model.add_Lorentz_object(self.model.get('lorentz'))
            aloha_model.compute_subset(list(request))
            text = '\n'.join(
                routine.write(output_dir=None, mode='no_include',
                              language=model_handling.MadMatrixALOHAWriter)[1]
                for routine in dict(aloha_model).values())
        finally:
            aloha.complex_mass, aloha.unitary_gauge, aloha.loop_mode = saved
        # a definition runs from its name, at the start of a line, to the next one
        parts = re.split(r'\n  ([A-Za-z0-9_]+)\(', text)
        return dict(zip(parts[1::2], parts[2::2]))

    def test_standard_propagator(self):
        """The coupling, the mass and the width go to madmatrix::denom, and the
        wavefunction is denom times the numerator: no cId, nothing declared in
        denominator precision and cast down afterwards."""
        routines = self.routines([(('FFV2',), [], 3),         # Z
                                  (('FFV1',), ['P0'], 3)])    # photon
        for name in ('FFV2_3', 'FFV1P0_3'):
            body = routines[name]
            self.assertIn('const cxtype_amp_sv denom = '
                          'madmatrix::denom( Ccoeff, COUP, dP3, M3, W3 );', body, name)
            self.assertIn('wV3[0] = denom * ', body, name)
            for old in ('cxtype_denom_sv denom', 'cId',
                        'static_cast<cxtype_amp_sv>(denom)'):
                self.assertNotIn(old, body, name)

    def test_assembled_routine(self):
        """The Z coupling to fermions is assembled from FFV2 and FFV4: the two
        couplings multiply their own terms, and madmatrix::denom only gets the
        momentum, the mass and the width."""
        routines = self.routines([(('FFV2', 'FFV4'), [], 3),
                                  (('FFV2', 'FFV4'), [], 1)])
        body = routines['FFV2_4_3']
        self.assertIn('const cxtype_amp_sv denom = madmatrix::denom( dP3, M3, W3 );', body)
        self.assertIn('Ccoeff1 * static_cast<cxtype_amp_sv>( COUP1 )', body)
        self.assertIn('Ccoeff2 * static_cast<cxtype_amp_sv>( COUP2 )', body)
        self.assertIn('const cxtype_amp_sv denom = madmatrix::denom( dP1, M1, W1 );',
                      routines['FFV2_4_1'])
        # the two structures alone keep their coupling in the propagator
        self.assertIn('madmatrix::denom( Ccoeff, COUP, dP3, M3, W3 )', routines['FFV2_3'])

    def test_complex_mass_scheme(self):
        """The mass is a cxtype that carries the width: madmatrix::denom gets it
        and no width, with the coupling or without it."""
        routines = self.routines([(('FFV2',), [], 3), (('FFV2', 'FFV4'), [], 3)],
                                 complex_mass=True)
        body = routines['FFV2_3']
        self.assertIn('const cxtype & M3', body)
        self.assertIn('const cxtype_amp_sv denom = '
                      'madmatrix::denom( Ccoeff, COUP, dP3, M3 );', body)
        self.assertIn('const cxtype_amp_sv denom = madmatrix::denom( dP3, M3 );',
                      routines['FFV2_4_3'])
        for name in ('FFV2_3', 'FFV2_4_3'):
            self.assertNotIn('W3', routines[name], name)

    def test_axial_gauge_custom_denominator(self):
        """In axial gauge a massless vector has the custom denominator
        (P.PBar) P^2: it does not go through madmatrix::denom, and it is
        divided in denominator precision, on TMPs built from the double
        momenta."""
        routines = self.routines([(('VVV1',), ['P0'], 1), (('FFV1',), ['P0'], 3)],
                                 gauge=2)
        for name in ('VVV1P0_1', 'FFV1P0_3'):
            body = routines[name]
            self.assertNotIn('madmatrix::denom', body, name)
            found = re.search(
                r'const cxtype_amp_sv denom = static_cast<cxtype_amp_sv>\( '
                r'static_cast<fptype_denom>\(Ccoeff\) \* static_cast<cxtype_denom_sv>\(COUP\)'
                r' / \( \((dTMP\d+) \* (dTMP\d+)\) \) \);', body)
            self.assertTrue(found, name)
            for tmp in found.groups():
                self.assertIn('const fptype_denom_sv %s = ' % tmp, body, name)
