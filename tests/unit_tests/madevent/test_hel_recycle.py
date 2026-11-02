##############################################################################
#
# Copyright (c) 2010 The MadGraph Development team and Contributors
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
""" Fixed-form line wrapping of the helicity recycled matrix element """

from __future__ import absolute_import
import os
import shutil
import tempfile
import unittest

import madgraph.madevent.hel_recycle as hel_recycle


class TestDoMultiline(unittest.TestCase):
    """do_multiline breaks a statement over fixed-form continuation lines.

    A continuation line is attached to whatever statement precedes it, so a
    physical line which holds nothing but blanks must never be emitted: the
    continuation after it would silently extend the previous statement."""

    # the Kleiss-Kuijf color flow JAMPs of g g > g g g on the DDM basis: long
    # enough to wrap and with no space of their own to wrap at, so the only
    # candidate split point is inside the statement's indentation
    JAMPF = '        JAMPF(2,1)=+2D0*(-IMAG1*JAMP(3,1)-IMAG1*JAMP(4,1)' \
            '-IMAG1*JAMP(6,1))'

    def physical_lines(self, line):
        return hel_recycle.do_multiline(line).split('\n')

    def assertWrapIsValid(self, line):
        lines = self.physical_lines(line)
        for i, physical in enumerate(lines):
            self.assertLessEqual(len(physical), 132,
                                 'physical line %d is too long' % i)
            if i:
                self.assertTrue(physical.lstrip().startswith('$'),
                                'continuation line %d lost its marker' % i)
        for i, physical in enumerate(lines[:-1]):
            self.assertTrue(physical.strip(),
                            'physical line %d is blank, so the continuation '
                            'which follows it joins the previous statement' % i)
        # and nothing of the statement is lost on the way
        rebuilt = ''.join(p.lstrip()[1:] if i else p
                          for i, p in enumerate(lines))
        self.assertEqual(rebuilt.replace(' ', ''), line.replace(' ', ''))

    def test_no_blank_physical_line_without_a_space_to_wrap_at(self):
        """A statement whose only space is its indentation wraps mid-token"""

        self.assertWrapIsValid(self.JAMPF)
        self.assertEqual(len(self.physical_lines(self.JAMPF)), 2)

    def test_short_line_is_untouched(self):
        """Nothing happens below the limit"""

        short = '        JAMPF(1,1)=+2D0*(+IMAG1*JAMP(6,1))'
        self.assertEqual(hel_recycle.do_multiline(short), short)

    def test_wraps_at_a_space_when_there_is_one(self):
        """The usual case still breaks at the last space that fits"""

        line = '        JAMP(2,1) = (-1.000000000000000D+00)*AMP( K,8)+' \
               '(-1.000000000000000D+00)*AMP( K,11)+(-1.0D+00)*TMP_JAMP(20)'
        self.assertWrapIsValid(line)
        self.assertTrue(self.physical_lines(line)[0].endswith(' '))

    def test_every_wrap_width_around_the_limit_is_valid(self):
        """Sweep the statement length across the wrap width"""

        for pad in range(40):
            line = '        JAMPF(2,1)=+2D0*(' + 'X' * pad + ')'
            self.assertWrapIsValid(line)


class TestCppDialect(unittest.TestCase):
    """Reading and re-writing the C++ the madmatrix backend emits.

    The recycler's algorithm is language independent; what a dialect owns is
    only how a call is taken apart and put back together. The C++ one has to
    cope with three things fortran never shows it: a template argument list
    before the real one, a result that is not the last argument, and an
    amplitude whose name cannot say which diagram it belongs to."""

    EXTERNAL = ('      ixxxxx<M_ACCESS, W_ACCESS>( momenta, m_pars->MT,'
                ' cHel[ihel][2], -1, cFlavors[iflavor][2], aloha_obj[5], 2 );')
    SCALAR = ('      sxxxxx<M_ACCESS, W_ACCESS>( momenta, +1,'
              ' cFlavors[iflavor][3], aloha_obj[6], 3 );')
    CURRENT = ('      FFV1P0_3<W_ACCESS, CD_ACCESS>( aloha_obj[0],'
               ' aloha_obj[1], COUPs[0], 1.0, 0., 0., aloha_obj[4] );')
    AMPLITUDE = ('      FFV1_0<W_ACCESS, A_ACCESS, CD_ACCESS>( aloha_obj[2],'
                 ' aloha_obj[3], aloha_obj[4], COUPs[0], 1.0,'
                 ' &amp_fp[0] ); //HRAMP 7')

    def setUp(self):
        self.dialect = hel_recycle.CppDialect()
        self.tmpdir = tempfile.mkdtemp(prefix='hr_cpp')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_the_template_arguments_are_not_arguments(self):
        """A call's arguments start after the <...>, and the commas inside it
        are not separators"""

        self.assertEqual(self.dialect.called_function(self.CURRENT), 'FFV1P0_3')
        self.assertEqual(self.dialect.arguments(self.CURRENT),
                         ['aloha_obj[0]', 'aloha_obj[1]', 'COUPs[0]', '1.0',
                          '0.', '0.', 'aloha_obj[4]'])

    def test_an_arrow_is_not_a_closing_bracket(self):
        """m_pars->MT would end the template argument list if '>' were read
        without looking at what precedes it"""

        self.assertEqual(self.dialect.called_function(self.EXTERNAL), 'ixxxxx')
        self.assertEqual(self.dialect.arguments(self.EXTERNAL)[1], 'm_pars->MT')

    def test_the_result_is_not_the_last_argument(self):
        """An external ends with its leg index, a current with its output"""

        external = self.dialect.arguments(self.EXTERNAL)
        self.assertEqual(self.dialect.output_name(external), 'aloha_obj[5]')
        self.assertEqual(self.dialect.output_position(external),
                         len(external) - 2)
        current = self.dialect.arguments(self.CURRENT)
        self.assertEqual(self.dialect.output_name(current), 'aloha_obj[4]')

    def test_the_leg_a_call_builds(self):
        """Read off the trailing index, and 1-based like the fortran one"""

        self.assertEqual(self.dialect.external_leg(
                                    self.dialect.arguments(self.EXTERNAL)), 3)
        self.assertEqual(self.dialect.external_helicity_leg(
                                    self.dialect.arguments(self.EXTERNAL)), 3)

    def test_an_axial_gauge_external_ends_with_its_reference(self):
        """vxxxxxr carries its reference leg after its own: the leg it builds
        is the argument after the output slot, not the last one"""

        line = ('      vxxxxxr<M_ACCESS, W_ACCESS>( momenta, 0., cHel[ihel][2],'
                ' +1, cFlavors[iflavor][2], aloha_obj[2], 2, 0 );')
        args = self.dialect.arguments(line)
        self.assertEqual(self.dialect.called_function(line).upper(), 'VXXXXXR')
        self.assertIn('VXXXXXR', self.dialect.external_routines)
        self.assertEqual(self.dialect.external_leg(args), 3)
        self.assertEqual(self.dialect.external_helicity_leg(args), 3)
        self.assertEqual(self.dialect.output_name(args), 'aloha_obj[2]')

    def test_a_scalar_has_no_helicity_to_fix(self):
        """Unlike fortran, where the slot exists and is overwritten anyway"""

        args = self.dialect.arguments(self.SCALAR)
        self.assertIsNone(self.dialect.external_helicity_leg(args))
        self.assertEqual(self.dialect.set_external_helicity(list(args), 0),
                         args)

    def test_fixing_a_helicity_replaces_the_table_read(self):
        """The external wavefunctions take a helicity per SIMD lane: the row's
        own goes in every lane"""
        args = self.dialect.arguments(self.EXTERNAL)
        fixed = self.dialect.set_external_helicity(list(args), -1)
        self.assertEqual(fixed[2], '( fptype_amp_sv{} + -1 )')
        self.assertEqual(fixed[:2] + fixed[3:], args[:2] + args[3:])
        # the per-lane helicity of the plain code is the same slot
        lane = self.EXTERNAL.replace('cHel[ihel][2]', 'hel_sv[2]')
        self.assertNotEqual(lane, self.EXTERNAL)
        lane_args = self.dialect.arguments(lane)
        self.assertEqual(self.dialect.external_helicity_leg(lane_args),
                         self.dialect.external_helicity_leg(args))
        self.assertEqual(self.dialect.set_external_helicity(list(lane_args), -1), fixed)

    def test_an_amplitude_says_which_diagram_it_is(self):
        """Every amplitude writes the same scratch, so the name cannot"""

        self.assertTrue(self.dialect.is_amplitude(
                            self.dialect.called_function(self.AMPLITUDE)))
        self.assertEqual(self.dialect.amplitude_diagram(
            self.AMPLITUDE, self.dialect.arguments(self.AMPLITUDE)), 7)

    def test_a_fold_is_stamped_out_once_per_row(self):
        """The exporter writes what an amplitude contributes with a ${row}
        hole; the dialect fills it, once per row the amplitude survives in"""

        self.dialect.folds = {7: ['      jampAll_sv[${row} * ncolor] += amp_sv[0];']}

        class FakeAmp(object):
            def __init__(self, row, args):
                self.numbers, self.diag_num, self.args = (row,), 7, args

        args = self.dialect.arguments(self.AMPLITUDE)
        rendered = self.dialect.render_amplitudes(
            self.AMPLITUDE, [FakeAmp(1, args), FakeAmp(4, args)],
            gauge='U', amp_splt=False)
        self.assertEqual(rendered.count('FFV1_0'), 2)
        self.assertIn('jampAll_sv[0 * ncolor]', rendered)
        self.assertIn('jampAll_sv[3 * ncolor]', rendered)

    def test_a_line_that_is_not_a_call_is_not_one(self):
        for line in ('      // Amplitude(s) for diagram number 2',
                     '//HRFOLD 7',
                     '      jampAll_sv[( 0 * nParity + iParity ) * ncolor + 1]'
                     ' += amp_sv[0];',
                     '      if( storeChannelWeights )'):
            self.assertIsNone(self.dialect.called_function(line), line)

    def test_the_fold_block_is_read_out_and_then_skipped(self):
        """It is not a call and must not reach the unfolding, but it is needed
        before the pass that has to skip it -- hence the separate read"""

        path = os.path.join(self.tmpdir, 'orig.cc')
        with open(path, 'w') as fsock:
            fsock.write('\n'.join([self.AMPLITUDE, '//HRFOLD 7',
                                   '      jampAll_sv[${row}] += amp_sv[0];',
                                   '//HRENDFOLD']) + '\n')
        self.assertEqual(self.dialect.read_folds(path),
                         {7: ['      jampAll_sv[${row}] += amp_sv[0];']})
        skipped = [line for line in open(path)
                   if self.dialect.skip_line(line)]
        self.assertEqual(len(skipped), 3)


class TestCppAmplitudeSplit(unittest.TestCase):
    """The P1N amplitude split, rendered for C++.

    The decision -- which leg to peel, and which unfolded amplitudes then share
    a partial contraction -- is the fortran one (plan_amp_split); what is
    checked here is what the C++ dialect makes of it, and in particular the
    flavour test, which is NOT uniform across peel positions and whose failure
    mode is silent."""

    LINE = ('      FFV1_0<W_ACCESS, A_ACCESS, CD_ACCESS>( aloha_obj[0],'
            ' aloha_obj[1], aloha_obj[2], COUPs[0], 1.0,'
            ' &amp_fp[0] ); //HRAMP 1')

    class FakeAmp(object):
        def __init__(self, row, args):
            self.numbers, self.diag_num, self.args = (row,), 1, list(args)

    def setUp(self):
        self.dialect = hel_recycle.CppDialect()
        self.dialect.folds = {1: ['      jamp[${row}] += amp_sv[0];']}

    def amplitudes(self, peeled_slots, shared=('aloha_obj[1]', 'aloha_obj[2]')):
        """One amplitude per peeled wavefunction, all sharing the other legs"""
        return [self.FakeAmp(row + 1, [slot, shared[0], shared[1],
                                       'COUPs[0]', '1.0', '&amp_fp[0]'])
                for row, slot in enumerate(peeled_slots)]

    def render(self, amps, **env):
        old = hel_recycle.CppDialect.P1N_MIN_SHARING
        hel_recycle.CppDialect.P1N_MIN_SHARING = env.get('minshare', 2)
        try:
            return self.dialect.render_amplitudes(self.LINE, amps,
                                                  gauge='U', amp_splt=True)
        finally:
            hel_recycle.CppDialect.P1N_MIN_SHARING = old

    def test_the_column_with_most_wavefunctions_is_peeled(self):
        """Column 0 varies, the others do not, so column 0 is the one to peel
        and P1N_1 is the routine that leaves it off"""

        out = self.render(self.amplitudes(['aloha_obj[%d]' % i
                                           for i in (10, 11, 12, 13)]))
        self.assertIn('FFV1P1N_1<W_ACCESS, CD_ACCESS>( aloha_obj[1],'
                      ' aloha_obj[2], COUPs[0], 1.0, _p1n );', out)
        # one partial contraction for the four amplitudes, four contractions
        self.assertEqual(out.count('P1N_1<'), 1)
        self.assertEqual(out.count('amp_sv[0] ='), 4)

    def test_a_fermion_peel_re_applies_the_flavour_test(self):
        """A P1N whose output is a fermion propagates the partner's flv_index
        and defers the test, so the contraction has to make it"""

        out = self.render(self.amplitudes(['aloha_obj[%d]' % i
                                           for i in (10, 11, 12, 13)]))
        self.assertIn('_p1n.flv_index != aloha_obj[10].flv_index', out)
        self.assertIn('_p1n.flv_index == -1 ) ? cxzero_sv()', out)

    def test_a_boson_peel_does_not(self):
        """A P1N whose output is a boson applies the test itself and zeroes the
        current; testing again would read an flv_index it never set"""

        amps = [self.FakeAmp(row + 1, ['aloha_obj[0]', 'aloha_obj[1]', slot,
                                       'COUPs[0]', '1.0', '&amp_fp[0]'])
                for row, slot in enumerate('aloha_obj[%d]' % i
                                           for i in (10, 11, 12, 13))]
        out = self.render(amps)
        self.assertIn('FFV1P1N_3<W_ACCESS, CD_ACCESS>( aloha_obj[0],'
                      ' aloha_obj[1], COUPs[0], 1.0, _p1n );', out)
        self.assertNotIn('flv_index', out)

    def test_each_amplitude_still_gets_its_own_row_fold(self):
        out = self.render(self.amplitudes(['aloha_obj[%d]' % i
                                           for i in (10, 11, 12, 13)]))
        for row in range(4):
            self.assertIn('jamp[%d] += amp_sv[0];' % row, out)

    def test_a_group_below_the_threshold_is_not_split(self):
        """Shallow grouping is a measured net loss, so it falls back to
        evaluating the whole vertex per row"""

        amps = self.amplitudes(['aloha_obj[10]', 'aloha_obj[11]'])
        self.assertNotIn('P1N', self.render(amps, minshare=4))
        self.assertIn('P1N', self.render(amps, minshare=2))

    def test_the_split_is_declined_rather_than_guessed(self):
        """A spin the fortran renderer refuses too"""

        amps = self.amplitudes(['aloha_obj[10]', 'aloha_obj[11]',
                                'aloha_obj[12]', 'aloha_obj[13]'])
        line = self.LINE.replace('FFV1_0', 'TTT1_0')
        self.assertIsNone(self.dialect._p1n_group(line, amps, 0))


class TestCppTabulatedSplit(unittest.TestCase):
    """The P1N groups written as tables (CppDialect._p1n_table) rather than as
    one dot product and one fold per member: what the tables say has to be
    exactly what the straight-line fold says, entry by entry."""

    LINE = TestCppAmplitudeSplit.LINE
    FakeAmp = TestCppAmplitudeSplit.FakeAmp
    SLOT = '( ${row} * nParity + iParity )'
    FOLD = ['if( storeChannelWeights )', '{',
            '  numAll_sv[%s * ndiagrams + 3] += cxabs2( amp_sv[0] );' % SLOT, '}',
            'jampTmpAll_sv[%s * 7 + 2] += amp_sv[0];' % SLOT,
            'jampTmpAll_sv[%s * 7 + 5] += -amp_sv[0];' % SLOT,
            'jampAll_sv[%s * njampso + 1] -= 1. / 3. * amp_sv[0];' % SLOT,
            'jampAll_sv[%s * njampso + 4] += 2. * cxtype( 0, 1 ) * amp_sv[0];' % SLOT,
            'jampAll_sv[%s * njampso + 0] -= amp_sv[0];' % SLOT]
    # (kind, index, value) of the fold above; None for the numerator
    EXPECTED = [(2, 3, None), (1, 2, 1), (1, 5, -1), (0, 1, -1. / 3.),
                (0, 4, 2j), (0, 0, -1)]

    def setUp(self):
        self.dialect = hel_recycle.CppDialect()
        self.dialect.TABLES = True
        self.dialect.folds = {1: list(self.FOLD)}

    def amplitudes(self, peeled=(10, 11, 12, 13)):
        return [self.FakeAmp(row + 1, ['aloha_obj[%d]' % slot, 'aloha_obj[1]',
                                       'aloha_obj[2]', 'COUPs[0]', '1.0', '&amp_fp[0]'])
                for row, slot in enumerate(peeled)]

    def render(self, amps):
        old = hel_recycle.CppDialect.P1N_MIN_SHARING
        hel_recycle.CppDialect.P1N_MIN_SHARING = 2
        try:
            return self.dialect.render_amplitudes(self.LINE, amps, gauge='U',
                                                  amp_splt=True)
        finally:
            hel_recycle.CppDialect.P1N_MIN_SHARING = old

    @staticmethod
    def table(out, name):
        line = [l for l in out.split('\n') if ' %s[] = {' % name in l][0]
        return line.split('{', 1)[1].rsplit('}', 1)[0]

    def test_the_members_are_tables_and_one_call(self):
        out = self.render(self.amplitudes())
        self.assertEqual(out.count('FFV1P1N_1<'), 1)
        self.assertNotIn('amp_sv[0] =', out)
        self.assertNotIn('jampAll_sv[(', out)
        self.assertIn('hr_p1n_fold<W_ACCESS, 4, true>( _p1n, aloha_obj, hrM, 4,'
                      ' hrF, hrC, 7,', out)
        # slot of the peeled leg, row, fold entries
        self.assertIn('hrM[] = { { 10, 0, 6 }, { 11, 1, 6 }, { 12, 2, 6 },'
                      ' { 13, 3, 6 } };', out)

    def test_the_entries_decode_to_the_fold(self):
        out = self.render(self.amplitudes())
        entries = [int(e.strip().rstrip('u'), 16)
                   for e in self.table(out, 'hrF').split(',')]
        coefs = [float(c) for c in self.table(out, 'hrC').split(',')]
        self.assertEqual(len(entries), 4 * len(self.EXPECTED))
        for k, entry in enumerate(entries):
            kind, index, value = self.EXPECTED[k % len(self.EXPECTED)]
            self.assertEqual(entry >> 30, kind)
            self.assertEqual(entry & 0xfffff, index)
            if value is None:
                continue
            op, coef = (entry >> 28) & 3, (entry >> 20) & 0xff
            got = {0: 1, 1: -1}.get(op, complex(coefs[2 * coef], coefs[2 * coef + 1]))
            self.assertEqual(complex(got), complex(value))
            # +-1 adds or subtracts, a real factor is not multiplied as a complex
            self.assertEqual(op, {1: 0, -1: 1}.get(value, 2 if not complex(value).imag else 3))

    def test_a_boson_peel_says_so(self):
        amps = [self.FakeAmp(row + 1, ['aloha_obj[0]', 'aloha_obj[1]',
                                       'aloha_obj[%d]' % slot, 'COUPs[0]', '1.0',
                                       '&amp_fp[0]'])
                for row, slot in enumerate((10, 11, 12, 13))]
        out = self.render(amps)
        self.assertIn('hr_p1n_fold<W_ACCESS, 4, false>', out)
        self.assertEqual(self.table(out, 'hrM').split(',')[0].strip(), '{ 10')

    def test_an_unknown_fold_keeps_the_straight_form(self):
        self.dialect.folds = {1: self.FOLD + ['jamp[${row}] += amp_sv[0];']}
        out = self.render(self.amplitudes())
        self.assertNotIn('hr_p1n_fold', out)
        self.assertEqual(out.count('amp_sv[0] ='), 4)

    def test_off_keeps_the_straight_form(self):
        self.dialect.TABLES = False
        out = self.render(self.amplitudes())
        self.assertNotIn('hr_p1n_fold', out)
        self.assertIn('jampAll_sv[( 3 * nParity + iParity ) * njampso + 4]', out)

    def test_a_store_is_its_own_kind(self):
        self.dialect.folds = {1: ['ampAll_sv[%s * 40 + 6] = amp_sv[0];' % self.SLOT]}
        out = self.render(self.amplitudes())
        self.assertIn('hrC, 40, jampAll_sv, ampAll_sv, numAll_sv,', out)
        entries = [int(e.strip().rstrip('u'), 16) for e in self.table(out, 'hrF').split(',')]
        self.assertEqual(entries, [(3 << 30) | 6] * 4)
        # a store with a factor, or an addition into the amplitude table, is not one
        for bad in ('ampAll_sv[%s * 40 + 6] += amp_sv[0];',
                    'ampAll_sv[%s * 40 + 6] = 2. * amp_sv[0];'):
            self.dialect.folds = {1: [bad % self.SLOT]}
            self.assertNotIn('hr_p1n_fold', self.render(self.amplitudes()))

    def test_coefficients(self):
        value = hel_recycle.CppDialect.fold_coefficient
        self.assertEqual(value(''), 1)
        self.assertEqual(value('-'), -1)
        self.assertEqual(value('1. / 3. * '), 1. / 3.)
        self.assertEqual(value('-2. * cxtype( 0, 1 ) * '), -2j)
        self.assertEqual(value('cxtype_amp( 0, 1 ) * '), 1j)
        self.assertIsNone(value('cI * '))
        self.assertIsNone(value('__import__( "os" ) * '))


class TestRecycledChunkBoundaries(unittest.TestCase):
    """Where the recycled call block may be cut.

    The block goes into ordinary functions so that the compiler is not handed
    one body of a hundred thousand lines. A cut in the wrong place is not a
    subtle problem -- it does not compile -- but WHERE the wrong places are is
    subtle: brace depth alone says nothing about `if( x )`, whose body opens on
    the next line."""

    def setUp(self):
        from madmatrix.model_handling import OneProcessExporterMadMatrix
        self.split = OneProcessExporterMadMatrix.hr_split_statements

    @staticmethod
    def fold(i):
        """What one amplitude turns into: the call, a guarded multichannel
        block, then the color flows"""
        return ['      FFV1_0<W_ACCESS>( a, b, &amp_fp[0] ); //HRAMP %d' % i,
                '      if( storeChannelWeights )',
                '      {',
                '        numAll_sv[%d] += cxabs2( amp_sv[0] );' % i,
                '      }',
                '      jampAll_sv[%d] += amp_sv[0];' % i]

    def assertChunksAreWholeStatements(self, lines, per_chunk):
        chunks = self.split(lines, per_chunk)
        self.assertEqual([l for c in chunks for l in c], lines,
                         'the block must come back out unchanged')
        for n, chunk in enumerate(chunks):
            depth = sum(l.count('{') - l.count('}') for l in chunk)
            self.assertEqual(depth, 0, 'chunk %d leaves a brace open' % n)
            last = chunk[-1].split('//')[0].rstrip()
            self.assertTrue(last.endswith(';') or last.endswith('}'),
                            'chunk %d ends mid-statement: %r' % (n, last))
        return chunks

    def test_a_guard_is_never_split_from_its_body(self):
        """`if( storeChannelWeights )` leaves the brace depth at zero, so depth
        alone would allow a cut between it and the block it guards"""

        lines = sum((self.fold(i) for i in range(6)), [])
        for per_chunk in range(1, 8):
            self.assertChunksAreWholeStatements(lines, per_chunk)

    def test_a_braced_group_is_never_split(self):
        """A P1N group holds declarations its contractions read"""

        lines = ['      {',
                 '        FFV1P1N_1<W_ACCESS>( a, b, _p1n );',
                 '        const cxtype_sv* _pt = access( _p1n.w );',
                 '        { const cxtype_sv* _pw = access( w.w );',
                 '          amp_sv[0] = _pt[0] * _pw[0]; }',
                 '      }'] * 4
        for per_chunk in (1, 2, 3, 5):
            chunks = self.assertChunksAreWholeStatements(lines, per_chunk)
            for chunk in chunks:
                opens = ''.join(chunk).count('{')
                self.assertEqual(opens, ''.join(chunk).count('}'))

    def test_the_whole_block_survives_every_chunk_size(self):
        lines = sum((self.fold(i) for i in range(10)), [])
        for per_chunk in range(1, 30):
            self.assertChunksAreWholeStatements(lines, per_chunk)

class TestRecycledChunkPolicy(unittest.TestCase):
    """When the recycled block is cut up at all. Chunks cost run time on a small
    block and save the build of a big one, so a block stays inline up to
    HR_INLINE_MAX statements unless --hel_recycling_chunk says otherwise."""

    def setUp(self):
        from madmatrix.model_handling import OneProcessExporterMadMatrix
        self.cls = OneProcessExporterMadMatrix

    def exporter(self, options):
        class Writer(object):
            cmd_options = options
        exporter = self.cls.__new__(self.cls)
        exporter.helas_call_writer = Writer()
        return exporter

    def test_a_small_block_stays_inline(self):
        exporter = self.exporter({})
        self.assertEqual(exporter.hel_recycling_chunk_size(self.cls.HR_INLINE_MAX), 0)
        self.assertEqual(exporter.hel_recycling_chunk_size(self.cls.HR_INLINE_MAX + 1),
                         self.cls.HR_CHUNK_STMTS)

    def test_the_option_wins(self):
        self.assertEqual(self.exporter({'hel_recycling_chunk': '50'})
                         .hel_recycling_chunk_size(10), 50)
        self.assertEqual(self.exporter({'hel_recycling_chunk': '0'})
                         .hel_recycling_chunk_size(10**6), 0)

    def test_the_size_limit_and_its_option(self):
        self.assertEqual(self.exporter({}).hel_recycling_max_stmts(),
                         self.cls.HR_MAX_STMTS)
        self.assertEqual(self.exporter({'hel_recycling_max': '0'})
                         .hel_recycling_max_stmts(), 0)
        self.assertEqual(self.exporter({'hel_recycling_max': '12'})
                         .hel_recycling_max_stmts(), 12)
        self.assertEqual(self.exporter({'hel_recycling_max': 'x'})
                         .hel_recycling_max_stmts(), self.cls.HR_MAX_STMTS)
        # what is counted is what hr_chunk_calls counts: no comment, no blank
        calls = ('      // recycled row 0\n\n'
                 '      FFV1_0<W_ACCESS>( a, b, &amp_fp[0] ); //HRAMP 1\n'
                 '      if( storeChannelWeights )\n      {\n'
                 '        numAll_sv[0] += cxabs2( amp_sv[0] );\n      }\n')
        self.assertEqual(self.cls.hr_count_statements(calls), 5)
        # the tables of a tabulated P1N group are data, not code
        calls += ('      {\n        static constexpr HrMember hrM[] = { { 1, 0, 1 } };\n'
                  '        hr_p1n_fold<W_ACCESS, 4, false>( _p1n );\n      }\n')
        self.assertEqual(self.cls.hr_count_statements(calls), 8)

    def test_chunks_take_the_state_and_are_called_in_order(self):
        exporter = self.exporter({'hel_recycling_chunk': '2'})
        calls = '\n'.join('      FFV1_0<W_ACCESS>( a, b, &amp_fp[0] ); //HRAMP %d' % i
                          for i in range(5)) + '\n'
        decls, units, body = exporter.hr_chunk_calls(calls)
        self.assertEqual(len(units), 1)
        defs = units[0]
        self.assertEqual(defs.count('static void'), 3)
        unit = defs[defs.index('hr_unit_0'):]
        self.assertEqual([line.split('(')[0].strip() for line in unit.split('\n')
                          if line.startswith('    hr_chunk_')],
                         ['hr_chunk_0', 'hr_chunk_1', 'hr_chunk_2'])
        self.assertEqual([line.split('(')[0].strip() for line in body.split('\n') if line],
                         ['hr_unit_0'])
        for decl, name in self.cls.HR_CHUNK_STATE:
            self.assertIn(decl, defs)
            self.assertIn(name, body)
        self.assertEqual(self.exporter({}).hr_chunk_calls(calls), ('', [], calls))

    def test_chunks_are_declared_without_names_and_spread_over_files(self):
        exporter = self.exporter({'hel_recycling_chunk': '2'})
        exporter.HR_FILE_STMTS = 4
        calls = '\n'.join('      FFV1_0<W_ACCESS>( a, b, &amp_fp[0] ); //HRAMP %d' % i
                          for i in range(9)) + '\n'
        decls, units, body = exporter.hr_chunk_calls(calls)
        # chunks of 2 statements, files of 4: 2 + 2 + 1 chunks, one unit each
        self.assertEqual([unit.count('static void') for unit in units], [2, 2, 1])
        for k, unit in enumerate(units):
            self.assertIn('#include "HelRecyclingChunks.h"', unit)
            self.assertIn('visibility( "hidden" ) ) ) void\n  hr_unit_%d(' % k, unit)
        self.assertEqual(decls.count('visibility( "hidden" )'), 3)
        self.assertEqual(body.count('hr_unit_'), 3)
        # SigmaKin.cc's own tables are passed, but not named in its declarations
        for name in ('cIPD', 'cFlavors', 'cBWCUTOFF'):
            self.assertNotIn(name, decls)
            self.assertIn(name, body)
        self.assertIn('const short ( * )[npar]', decls)


class TestRecycledColourSubExpressions(unittest.TestCase):
    """The recycled folds keep the shared sub-expressions of the color flows
    (build_jamp_plan) that the plain build writes, one set per helicity row:
    each amplitude is added into its row's sub-expressions, and hr_post_lines
    then makes the row's color flows out of them. Checked by running the
    emitted statements -- they are valid Python as well as C++ -- against the
    color flows written out term by term, with some amplitudes left out of the
    row (the ones that vanish there are never folded in)."""

    def setUp(self):
        from fractions import Fraction
        from madmatrix.model_handling import MadMatrixUFOHelasCallWriter
        writer = MadMatrixUFOHelasCallWriter.__new__(MadMatrixUFOHelasCallWriter)
        writer.cmd_options = {'jamp_optim': True}
        writer.jamp_orbit = False
        writer.split_order_index = lambda matrix_element: None
        self.writer = writer
        # six color flows over eight amplitudes, sharing the obvious partial
        # sums; coefficients (sign, fraction, imaginary, power of Nc)
        self.color_amplitudes = [
            [((1, Fraction(1), False, 0), 1), ((1, Fraction(1), False, 0), 2),
             ((-1, Fraction(1), False, 0), 3)],
            [((1, Fraction(1), False, 0), 1), ((1, Fraction(1), False, 0), 2),
             ((1, Fraction(1, 3), False, 0), 4)],
            [((1, Fraction(1), False, 0), 1), ((1, Fraction(1), False, 0), 2),
             ((-1, Fraction(1), False, 0), 3), ((1, Fraction(1), True, 0), 5)],
            [((1, Fraction(1), False, 0), 6), ((1, Fraction(1), False, 0), 7),
             ((1, Fraction(1), False, 0), 1), ((1, Fraction(1), False, 0), 2)],
            [((1, Fraction(1), False, 0), 6), ((1, Fraction(1), False, 0), 7),
             ((-1, Fraction(2), False, 0), 8)],
            [((1, Fraction(1), False, 0), 6), ((1, Fraction(1), False, 0), 7),
             ((1, Fraction(1), False, 0), 8), ((-1, Fraction(1), False, 0), 3)],
        ]
        self.color = {}
        for njamp, coeff_list in enumerate(self.color_amplitudes):
            for coeff, namp in coeff_list:
                self.color.setdefault(namp, {})[njamp] = coeff

    def run_row(self, plan, amps, row=2, nrows=4):
        ntmp, post = self.writer.hr_post_lines(plan)
        njamp = len(self.color_amplitudes)
        scope = {'nParity': 1, 'iParity': 0, 'hrNtmp': ntmp, 'njampso': njamp,
                 'cxtype': complex, 'cxtype_amp': complex, 'fptype': float,
                 'jampAll_sv': [0j] * (nrows * njamp),
                 'jampTmpAll_sv': [0j] * (nrows * max(ntmp, 1))}

        class Diagram(object):
            def get(self, key):
                return 1

        from string import Template
        self.writer.nb_tmp_jamp = ntmp
        for namp, value in amps.items():
            captures = plan[1].get(namp, []) if plan is not None else None
            fold = self.writer._hr_orig_amplitude('FFV1_0( a ); ', namp, Diagram(),
                                                  False, self.color, captures)
            self.assertEqual(fold[-1], '//HRENDFOLD')
            scope['amp_sv'] = [value]
            for line in fold[2:-1]:
                exec(Template(line).substitute(row=row).rstrip(';'), scope)
        # the per-row post lines run in place on the row's slice of the tables
        tmp = scope['jampTmpAll_sv'][row * ntmp:(row + 1) * ntmp]
        jamp = scope['jampAll_sv'][row * njamp:(row + 1) * njamp]
        post_scope = {'jampTmp_sv': tmp, 'jamp_sv': jamp, 'cxtype': complex}
        for line in post:
            exec(line.strip().rstrip(';'), post_scope)
        return jamp

    def expected(self, amps):
        jamp = [0j] * len(self.color_amplitudes)
        for njamp, coeff_list in enumerate(self.color_amplitudes):
            for (sign, frac, imag, ncp), namp in coeff_list:
                jamp[njamp] += sign * float(frac) * (1j if imag else 1) * \
                               3**ncp * amps.get(namp, 0)
        return jamp

    def check(self, amps):
        plan = self.writer.build_jamp_plan(None, self.color_amplitudes)
        self.assertIsNotNone(plan)
        self.assertGreater(plan[0], 0)
        for got, want in zip(self.run_row(plan, amps), self.expected(amps)):
            self.assertAlmostEqual(got, want, places=12)
        # and the expanded fold (no plan) agrees with it
        for got, want in zip(self.run_row(None, amps), self.expected(amps)):
            self.assertAlmostEqual(got, want, places=12)

    def test_every_amplitude_in_the_row(self):
        self.check({n: complex(n, 0.5 * n - 1) for n in range(1, 9)})

    def test_amplitudes_missing_from_the_row(self):
        # the first amplitude of a sub-expression, which the plain build
        # assigns rather than adds, is among the ones left out
        self.check({n: complex(n, -n) for n in (2, 4, 5, 7)})

    def run_rows(self, plan, amps):
        """The rows mode: every amplitude of the row in ampRow (0 where it
        vanishes), then the plain build's color code once, reading it."""
        class Diagram(object):
            def get(self, key):
                return 1
        ncolor = len(self.color_amplitudes)
        lines = []
        for namp in range(1, 9):
            lines += self.writer._hr_row_lines(
                namp, Diagram(), True, self.color,
                plan[1].get(namp, []) if plan else None,
                plan[2].get(namp, []) if plan else [])
        if plan:
            lines += plan[3]
        scope = {'ampRow': [amps.get(n, 0j) for n in range(1, 9)],
                 'jamp_sv': [0j] * ncolor, 'jampTmp_sv': [0j] * max(plan[0] if plan else 0, 1),
                 'numerators_sv': [0.] * 2, 'storeChannelWeights': False,
                 'cxabs2': lambda z: abs(z) ** 2,
                 'cxtype': complex, 'cxtype_amp': complex, 'fptype': float}
        for line in lines:
            line = line.strip().rstrip(';')
            if line.startswith('if( storeChannelWeights ) '):
                line = line[len('if( storeChannelWeights ) '):]
            exec(line, scope)
        return scope['jamp_sv']

    def test_rows_mode(self):
        plan = self.writer.build_jamp_plan(None, self.color_amplitudes)
        for amps in ({n: complex(n, 0.5 * n - 1) for n in range(1, 9)},
                     {n: complex(n, -n) for n in (2, 4, 5, 7)}):
            for p in (plan, None):
                for got, want in zip(self.run_rows(p, amps), self.expected(amps)):
                    self.assertAlmostEqual(got, want, places=12)
        # the colour code reads the row's table, never the scratch amplitude
        rows = self.writer._hr_row_lines(1, None, False, self.color, plan[1][1], [])
        self.assertTrue(rows)
        self.assertTrue(all('amp_sv' not in l and 'ampRow[0]' in l for l in rows))

    def test_rows_store(self):
        fold = self.writer._hr_orig_store('FFV1_0( a ); ', 7, 40)
        self.assertEqual(fold[1:], ['//HRFOLD 7',
            'ampAll_sv[( ${row} * nParity + iParity ) * 40 + 6] = amp_sv[0];',
            '//HRENDFOLD'])

    def test_post_lines_only_ever_add(self):
        plan = self.writer.build_jamp_plan(None, self.color_amplitudes)
        _ntmp, post = self.writer.hr_post_lines(plan)
        self.assertTrue(post)
        for line in post:
            self.assertIn(' += ', line)
        self.assertEqual(self.writer.hr_post_lines(None), (0, []))


class TestRecycledOptions(unittest.TestCase):
    """--hel_recycling_jamp / --hel_recycling_tables, and the per-row include
    HelRecyclingJamps.inc each mode writes."""

    def test_jamp_mode(self):
        from madmatrix.model_handling import MadMatrixUFOHelasCallWriter as W
        writer = W.__new__(W)
        for options, mode in (({}, 'rows'), ({'hel_recycling_jamp': 'fold'}, 'fold'),
                              ({'hel_recycling_jamp': 'EXPANDED'}, 'expanded'),
                              ({'hel_recycling_jamp': 'bogus'}, 'rows')):
            writer.cmd_options = options
            self.assertEqual(writer.hr_jamp_mode(), mode)

    def test_tables(self):
        from madmatrix.model_handling import OneProcessExporterMadMatrix as E
        class Writer(object):
            cmd_options = {}
        exporter = E.__new__(E)
        exporter.helas_call_writer = Writer()
        self.assertTrue(exporter.hel_recycling_tables())
        for off in ('False', 'false', '0', 'no'):
            Writer.cmd_options = {'hel_recycling_tables': off}
            self.assertFalse(exporter.hel_recycling_tables())
        Writer.cmd_options = {'hel_recycling_tables': 'True'}
        self.assertTrue(exporter.hel_recycling_tables())

    def test_jamps_include(self):
        from madmatrix.model_handling import OneProcessExporterMadMatrix as E
        rows = E.hr_jamps_inc({'mode': 'rows', 'ntmp': 7, 'namp': 159,
                               'lines': ['jamp_sv[0] += ampRow[3];']})
        self.assertIn('const cxtype_amp_sv* ampRow = ampAll_sv + slot * 159;', rows)
        self.assertIn('cxtype_amp_sv jampTmp_sv[7];', rows)
        self.assertIn('jamp_sv[0] += ampRow[3];', rows)
        # no table of every row's numerators in the rows mode: hr_row hands the
        # event's own numerators_sv
        self.assertNotIn('numAll_sv', rows)
        fold = E.hr_jamps_inc({'mode': 'fold', 'ntmp': 7, 'namp': 0,
                               'lines': ['jampTmp_sv[1] += jampTmp_sv[0];']})
        self.assertIn('cxtype_amp_sv* jampTmp_sv = jampTmpAll_sv + (size_t)slot * 7;', fold)
        self.assertNotIn('ampRow', fold)
        expanded = E.hr_jamps_inc({'mode': 'expanded', 'ntmp': 0, 'namp': 0, 'lines': ()})
        self.assertTrue(all(l.strip().startswith('//') for l in expanded.splitlines() if l.strip()))


class TestHiddenRunCardEntries(unittest.TestCase):
    """The mg7 run card's hidden entries -- hel_recycling_max_mb among them --
    are read, and written back once the user set them (not before), so that
    rewriting the card does not drop them."""

    def test_round_trip(self):
        import madgraph.various.banner as banner_mod
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, 'run_card.toml')
            card = banner_mod.RunCardMG7()
            self.assertEqual(card['run'].get('hel_recycling_max_mb'), -1)
            card.write(path)
            self.assertNotIn('hel_recycling_max_mb', open(path).read())
            card.set('run.hel_recycling_max_mb', 0, user=True)
            card.write(path)
            text = open(path).read()
            self.assertIn('hel_recycling_max_mb = 0', text)
            # in its own section, so that the card stays valid TOML
            try:
                import tomllib
            except ImportError:
                tomllib = None
            if tomllib:
                self.assertEqual(tomllib.loads(text)['run']['hel_recycling_max_mb'], 0)
            again = banner_mod.RunCardMG7(path)
            self.assertEqual(again['run.hel_recycling_max_mb'], 0)
            again.write(path)
            self.assertIn('hel_recycling_max_mb = 0', open(path).read())
            # a legacy alias is read but never written
            card.set('generation.systematics', True, user=True)
            card.write(path)
            generation = text_section(open(path).read(), 'generation')
            self.assertFalse(any(l.strip().startswith('systematics =') for l in generation))
        finally:
            shutil.rmtree(tmpdir)


def text_section(text, name):
    """The lines of [name] in a TOML text."""
    out, inside = [], False
    for line in text.splitlines():
        if line.startswith('['):
            inside = line.strip() == '[%s]' % name
            continue
        if inside:
            out.append(line)
    return out


class TestRecyclingMakeArguments(unittest.TestCase):
    """What the mg7 launcher hands make for a --hel_recycling output, and when a
    library has to be rebuilt for other settings."""

    def setUp(self):
        try:
            import madgraph.iolibs.template_files.mg7.launch as launch
        except Exception as error:      # needs the mg7 runtime (madspace)
            self.skipTest('mg7 launch not importable: %s' % error)
        import madgraph.various.banner as banner_mod
        self.launch = launch
        self.process = launch.MadgraphProcess.__new__(launch.MadgraphProcess)
        self.process.run_card = banner_mod.RunCardMG7()

    def test_make_arguments(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('HRMAXMB', None)
            os.environ.pop('HRNOBLAS', None)
            self.assertEqual(self.process.hel_recycling_make_args(), ['HRMAXMB=', 'HRNOBLAS='])
            os.environ['HRMAXMB'] = '20'
            self.assertEqual(self.process.hel_recycling_make_args()[0], 'HRMAXMB=20')
            # the run card wins over the environment
            self.process.run_card.set('run.hel_recycling_max_mb', 0, user=True)
            self.process.run_card.set('run.hel_recycling_row_color_sum', True, user=True)
            self.assertEqual(self.process.hel_recycling_make_args(), ['HRMAXMB=0', 'HRNOBLAS=1'])

    def test_rebuild_when_the_settings_change(self):
        changed = self.launch.MadgraphProcess.hel_recycling_flags_changed
        tmpdir = tempfile.mkdtemp()
        try:
            self.assertFalse(changed(tmpdir, 'simd_128', ':'))      # not recycled
            open(os.path.join(tmpdir, 'HelRecycling.inc'), 'w').close()
            self.assertFalse(changed(tmpdir, 'simd_128', ':'))      # never built
            os.mkdir(os.path.join(tmpdir, 'build.simd_128'))
            with open(os.path.join(tmpdir, 'build.simd_128', '.hrflags'), 'w') as fsock:
                fsock.write(':\n')
            self.assertFalse(changed(tmpdir, 'simd_128', ':'))
            self.assertTrue(changed(tmpdir, 'simd_128', '0:'))
            self.assertFalse(changed(tmpdir, 'cuda', '0:'))           # gpu: plain anyway
        finally:
            shutil.rmtree(tmpdir)


class TestHelRecyclingWarmup(unittest.TestCase):
    """What the probe prints, read back: madevent's helicity-filter protocol."""

    def test_parse(self):
        from madmatrix.output import ProcessExporterMadMatrixStandalone
        output = '\n'.join(['Matrix Element/Good Helicity: 1 3',
                            'Matrix Element/Good Helicity: 1 1',
                            'HEL/ZEROAMP: 1 3 7',
                            'HEL/ZEROAMP: 1 1 2',
                            'Matrix element = 1.0 GeV^-2'])
        self.assertEqual(ProcessExporterMadMatrixStandalone._parse_hel_warmup(output),
                         ([1, 3], [(1, 2), (3, 7)]))


class TestHelRecyclingWarmupDispatch(unittest.TestCase):
    """Which directories an output warms up: an mg7 output its own (it has no
    check_sa.exe of its own, so it is lent one for the probe), and never the
    directory another output left registered."""

    def setUp(self):
        from madmatrix import output, model_handling
        self.output = output
        self.warmups = model_handling.OneProcessExporterMadMatrix._hr_warmups
        del self.warmups[:]
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.addCleanup(self.warmups.__delitem__, slice(None))

    def exporter(self, cls, name):
        exporter = cls.__new__(cls)
        exporter.dir_path = os.path.join(self.tmp, name)
        exporter._p_makefile_replace = {}
        return exporter

    def pdir(self, output_dir):
        path = os.path.join(output_dir, 'SubProcesses', 'P1_gg_ggg')
        os.makedirs(path)
        class Registered(object):
            pass
        registered = Registered()
        registered.path = path
        self.warmups.append(registered)
        return path

    def test_mg7_finalize_calls_the_warmup(self):
        from unittest import mock
        import madgraph.iolibs.export_cpp as export_cpp
        exporter = self.exporter(self.output.ProcessExporterMadMatrix, 'mg7')
        with mock.patch.object(export_cpp.ProcessExporterMG7, 'finalize') as base, \
             mock.patch.object(self.output.ProcessExporterMadMatrix,
                               '_run_hel_recycling_warmups') as warmup:
            exporter.finalize()
        base.assert_called_once_with()
        warmup.assert_called_once_with()

    def test_mg7_finalize_clears_the_list_and_builds_only_its_own(self):
        from unittest import mock
        import subprocess
        import madgraph.iolibs.export_cpp as export_cpp
        exporter = self.exporter(self.output.ProcessExporterMadMatrix, 'mg7')
        own = self.pdir(exporter.dir_path)
        foreign = self.pdir(os.path.join(self.tmp, 'other'))
        calls = []
        def run(cmd, cwd=None, **opts):
            calls.append((cmd, cwd, sorted(os.listdir(cwd))))
            return subprocess.CompletedProcess(cmd, 1, stdout=b'')
        with mock.patch.object(export_cpp.ProcessExporterMG7, 'finalize'), \
             mock.patch.object(self.output.subprocess, 'run', side_effect=run):
            exporter.finalize()
        self.assertEqual(self.warmups, [])
        self.assertEqual(set(cwd for _cmd, cwd, _ls in calls), set([own]))
        # the probe build ran on the lent standalone driver ...
        build, _cwd, listing = calls[0]
        self.assertEqual(build[:3], ['make', '-f', 'hrprobe.mk'])
        self.assertIn('HRPROBE=1', build)
        for name in ['check_sa.cc', 'hrprobe.mk', 'madmatrix.mk', 'timer.h']:
            self.assertIn(name, listing)
        # ... which is taken back with the probe build
        self.assertEqual(calls[-1][0][-1], 'cleanall')
        self.assertEqual(os.listdir(own), [])
        self.assertEqual(os.listdir(foreign), [])

    def test_a_new_output_drops_what_an_earlier_one_left(self):
        self.pdir(os.path.join(self.tmp, 'earlier'))
        self.output.ProcessExporterMadMatrix(os.path.join(self.tmp, 'new'),
                                             {'output_options': {}})
        self.assertEqual(self.warmups, [])


class TestGoodHelIds(unittest.TestCase):
    """NHEL(0,k) of the recycled matrix element is the 1-based id of that
    helicity in the original NHEL table.

    SMATRIX returns it as the selected helicity and get_nhel reads the LHE
    spin column back from it, where row 0 is the number of spin states: a
    0-based id puts every event on the previous combination."""

    # e+ e- > mu+ mu- ordering, only the four helicity conserving ones survive
    ALL_HEL = [(-1, 1, 1, -1), (-1, 1, 1, 1), (-1, 1, -1, -1), (-1, 1, -1, 1),
               (1, -1, 1, -1), (1, -1, 1, 1), (1, -1, -1, -1), (1, -1, -1, 1)]
    GOOD = ['1', '4', '5', '8']

    def recycled_nhel_lines(self, hel_filt):
        recycler = hel_recycle.HelicityRecycler(self.GOOD)
        recycler.hel_filt = hel_filt
        recycler.prepare_bools()
        for i, hel in enumerate(self.ALL_HEL, 1):
            recycler.get_good_hel('      DATA (NHEL(I,%4d),I=1,4) /%s/\n'
                                  % (i, ','.join('%2d' % h for h in hel)))
        recycler.get_good_hel('C     ----------\n')
        lines = recycler.template_dict['helicity_lines'].split('\n')
        return dict((tuple(int(h) for h in l.split('/')[1].split(',')[1:]),
                     int(l.split('/')[1].split(',')[0]))
                    for l in lines if 'NHEL' in l)

    def assertOriginalIds(self, nhel, expected_hel):
        self.assertEqual(sorted(nhel), sorted(expected_hel))
        for hel, old_id in nhel.items():
            self.assertEqual(self.ALL_HEL[old_id - 1], hel)

    def test_hel_recycling_ids_with_filtering(self):
        """hel_filtering = True keeps the good helicities with their own id"""

        nhel = self.recycled_nhel_lines(True)
        self.assertOriginalIds(nhel,
                               [self.ALL_HEL[int(i) - 1] for i in self.GOOD])

    def test_hel_recycling_ids_without_filtering(self):
        """hel_filtering = False keeps them all, ids still start at 1"""

        nhel = self.recycled_nhel_lines(False)
        self.assertOriginalIds(nhel, self.ALL_HEL)
        self.assertEqual(min(nhel.values()), 1)


class TestFortranRecyclerRegression(unittest.TestCase):
    """madevent's own recycler (the Fortran output, on by default in every
    madevent run) shares its algorithm with the C++ dialect: the DAG, the good-
    helicity and vanishing-amplitude filters and the P1N split plan. Its output
    must not change with that sharing. Each case is a matrix1_orig.f and
    template_matrix1.f of a madevent output, the arguments gen_ximprove passes
    (good helicities, vanishing amplitudes, vanishing (helicity, amplitude)
    pairs, gauge) and the matrix1_optim.f that main's recycler wrote for them
    before the C++ work (tests/input_files/hel_recycle_fortran); the cases run
    one after the other in this process, as the P directories of one output
    do in gen_ximprove."""

    CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                         os.pardir, 'input_files', 'hel_recycle_fortran')

    def test_the_fortran_output_is_unchanged(self):
        import json
        cases = sorted(os.listdir(self.CASES))
        self.assertEqual(len(cases), 5)
        tmpdir = tempfile.mkdtemp()
        try:
            for case in cases:
                src = os.path.join(self.CASES, case)
                args = json.load(open(os.path.join(src, 'recycler_args.json')))
                recycler = hel_recycle.HelicityRecycler(
                    args['good_hels'], args['bad_amps'],
                    [tuple(pair) for pair in args['bad_amps_perhel']],
                    gauge=args['gauge'])
                recycler.hel_filt = True
                recycler.amp_splt = True
                recycler.amp_filt = True
                out = os.path.join(tmpdir, case + '_optim.f')
                recycler.set_input(os.path.join(src, 'matrix1_orig.f'))
                recycler.set_output(out)
                recycler.set_template(os.path.join(src, 'template_matrix1.f'))
                recycler.generate_output_file()
                del recycler
                with open(os.path.join(src, 'matrix1_optim.f')) as fsock:
                    expected = fsock.read()
                with open(out) as fsock:
                    got = fsock.read()
                self.assertEqual(got, expected, 'the recycled Fortran of %s changed' % case)
        finally:
            shutil.rmtree(tmpdir)
