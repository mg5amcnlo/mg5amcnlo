################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
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
"""One place turns the systematics summary into percentages.

madspace reports cross sections and *absolute* uncertainties. Two things show
them as percentages -- the run's Systematics box and the parameter-scan row --
and they used to be two copies of the same arithmetic. They must not drift:
the same run printing one number in its box and another in the scan table
would be worse than either convention on its own.
"""

from __future__ import absolute_import

import inspect
import unittest

import madgraph.mg7.systematics_summary as systematics_summary


class _Summary(object):
    """The shape ``SystematicsCalculator::summary()`` writes."""

    NOMINAL = 500.0

    @classmethod
    def full(cls):
        return {
            'nominal': {'pdf_set': 'NNPDF23_lo', 'pdf_lhaid': 247000,
                        'cross_section': cls.NOMINAL},
            'event_count': 10000,
            'warnings': [],
            'scale': {'weight_ids': ['1', '2'],
                      'min': 0.785 * cls.NOMINAL,      # -21.5%
                      'max': 1.296 * cls.NOMINAL},     # +29.6%
            'pdf': [{'pdf_set': 'NNPDF23_lo', 'pdf_lhaid': 247000,
                     'error_type': 'replicas', 'weight_ids': ['11', '12'],
                     'central': cls.NOMINAL,
                     'uncertainty_up': 0.02 * cls.NOMINAL,
                     'uncertainty_down': 0.02 * cls.NOMINAL}],
        }


class TestPercentages(unittest.TestCase):

    def test_scale_is_measured_against_the_nominal(self):
        up, down = systematics_summary.scale_percentages(_Summary.full())
        self.assertAlmostEqual(up, 29.6, places=3)
        self.assertAlmostEqual(down, 21.5, places=3)

    def test_scale_comes_back_as_positive_magnitudes(self):
        """The caller supplies the sign: the box writes '+x% -y%', the scan
        row stores them in separate columns."""

        up, down = systematics_summary.scale_percentages(_Summary.full())
        self.assertGreater(up, 0)
        self.assertGreater(down, 0)

    def test_pdf_is_measured_against_the_sets_own_central(self):
        """For a replicas set that is the replica mean, not the nominal
        member, so the two are deliberately different numbers."""

        summary = _Summary.full()
        summary['pdf'][0]['central'] = 0.9 * _Summary.NOMINAL
        entry, up, down = systematics_summary.pdf_percentages(summary)[0]
        self.assertIs(entry, summary['pdf'][0])
        self.assertAlmostEqual(up, 0.02 * _Summary.NOMINAL / entry['central'] * 100)
        self.assertAlmostEqual(down, up)

    def test_a_negative_nominal_keeps_positive_magnitudes(self):
        """An interference (e.g. p p > u u~ QCD^2==2) has a negative total.
        Dividing by the signed nominal printed 'Scale variation: +-24.1%
        --30%'; the band must still read '+up% -down%'."""

        nominal = -9680.6
        summary = _Summary.full()
        summary['nominal']['cross_section'] = nominal
        # the envelope of a negative total: max is the least negative
        summary['scale']['min'] = 1.30 * nominal
        summary['scale']['max'] = 0.759 * nominal
        summary['pdf'][0]['central'] = nominal
        summary['pdf'][0]['uncertainty_up'] = 0.0695 * abs(nominal)
        summary['pdf'][0]['uncertainty_down'] = 0.0695 * abs(nominal)
        up, down = systematics_summary.scale_percentages(summary)
        self.assertAlmostEqual(up, 24.1, places=3)
        self.assertAlmostEqual(down, 30.0, places=3)
        _entry, pup, pdown = systematics_summary.pdf_percentages(summary)[0]
        self.assertAlmostEqual(pup, 6.95, places=3)
        self.assertAlmostEqual(pdown, 6.95, places=3)

    def test_a_set_without_an_uncertainty_is_left_out(self):
        """errorset on a one-member set: the entry exists, the uncertainty
        does not. Reporting zero would be a lie."""

        summary = _Summary.full()
        summary['pdf'] = [{'pdf_set': 'NNPDF40MC_lo_as_01180',
                           'error_type': '', 'weight_ids': []}]
        self.assertEqual(systematics_summary.pdf_percentages(summary), [])

    def test_a_set_without_a_central_is_left_out(self):
        """summary() omits "central" when it could not be computed."""

        summary = _Summary.full()
        summary['pdf'][0].pop('central')
        self.assertEqual(systematics_summary.pdf_percentages(summary), [])

    def test_a_run_with_no_events_reports_nothing(self):
        """summary() omits cross_section when event_count is 0."""

        summary = _Summary.full()
        summary['event_count'] = 0
        del summary['nominal']['cross_section']
        self.assertIsNone(systematics_summary.nominal_cross_section(summary))
        self.assertIsNone(systematics_summary.scale_percentages(summary))
        self.assertEqual(systematics_summary.pdf_percentages(summary), [])

    def test_a_pdf_only_run_has_no_scale_block(self):
        summary = _Summary.full()
        summary.pop('scale')
        self.assertIsNone(systematics_summary.scale_percentages(summary))
        self.assertTrue(systematics_summary.pdf_percentages(summary))

    def test_an_empty_summary_is_handled(self):
        for empty in ({}, None):
            self.assertIsNone(systematics_summary.nominal_cross_section(empty))
            self.assertIsNone(systematics_summary.scale_percentages(empty or {}))
            self.assertEqual(systematics_summary.pdf_percentages(empty or {}), [])


class TestOneImplementation(unittest.TestCase):
    """Neither caller may grow its own copy back."""

    def sources(self):
        import madgraph.mg7.launch as launch

        return {
            'the Systematics box':
                inspect.getsource(launch.MadgraphProcess.log_systematics_summary),
            'the scan row':
                inspect.getsource(launch.MadgraphProcess.get_result),
        }

    def test_neither_caller_does_the_arithmetic_itself(self):
        for name, source in self.sources().items():
            self.assertNotIn('* 100', source,
                             '%s computes percentages itself again' % name)
            self.assertIn('systematics_summary', source,
                          '%s no longer uses the shared helper' % name)

    def test_the_box_and_the_scan_row_agree(self):
        """Same summary in, same numbers out of both."""

        summary = _Summary.full()
        summary['pdf'][0]['central'] = 0.9 * _Summary.NOMINAL

        scale = systematics_summary.scale_percentages(summary)
        pdf = systematics_summary.pdf_percentages(summary)

        # what get_result() stores
        row = {'scale_up(%)': scale[0], 'scale_down(%)': scale[1],
               'pdf_up(%)': pdf[0][1], 'pdf_down(%)': pdf[0][2]}
        # what the box formats
        box_scale = ('+%.3g' % scale[0], '-%.3g' % scale[1])
        box_pdf = ('+%.3g' % pdf[0][1], '-%.3g' % pdf[0][2])

        self.assertEqual(box_scale,
                         ('+%.3g' % row['scale_up(%)'],
                          '-%.3g' % row['scale_down(%)']))
        self.assertEqual(box_pdf,
                         ('+%.3g' % row['pdf_up(%)'],
                          '-%.3g' % row['pdf_down(%)']))

    def test_the_helper_imports_nothing(self):
        """It has to be usable from anywhere that reads a summary."""

        import ast

        tree = ast.parse(inspect.getsource(systematics_summary))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
        self.assertEqual([n for n in names if n and n != '__future__'], [])
