##############################################################################
#
# Copyright (c) 2010 The MadGraph7 Development team and Contributors
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
"""Drawing the mg7 event-sample histograms into Events/<run>/plots.

The tests that actually draw are skipped when matplotlib is missing -- which
is the case for the interpreter the test runner uses on some setups, and is
exactly the situation a run has to survive without failing.
"""

from __future__ import absolute_import
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from madgraph import MG5DIR
from madgraph.mg7 import plots


def make_histogram(name='jet-pt', values=None, bin_count=4):
    """One histogram as EventHistograms::to_json() writes it: the arrays
    carry the underflow first and the overflow last."""
    if values is None:
        values = [1.0, 2.0, 3.0, 4.0]
    return {
        'name': name,
        'min': 0.0,
        'max': 100.0,
        'bin_count': bin_count,
        'bin_values': [99.0] + list(values) + [88.0],
        'bin_errors': [9.9] + [0.1 * v for v in values] + [8.8],
        'scale_envelope': {'low': [99.0] + [0.8 * v for v in values] + [88.0],
                           'high': [99.0] + [1.2 * v for v in values] + [88.0]},
        'pdf_uncertainty': [{'pdf_set': 'NNPDF40_lo_as_01180',
                             'pdf_lhaid': 331900,
                             'error_type': 'replicas',
                             'central': [99.0] + list(values) + [88.0],
                             'uncertainty_down': [0.] + [0.05 * v for v in values] + [0.],
                             'uncertainty_up': [0.] + [0.05 * v for v in values] + [0.]}],
        'weights': [],
    }


class TestPlotHelpers(unittest.TestCase):
    """the arithmetic, which needs no plotting backend"""

    def test_underflow_and_overflow_are_dropped(self):
        self.assertEqual(plots._inner([99.0, 1.0, 2.0, 88.0], 2), [1.0, 2.0])

    def test_short_column_is_padded(self):
        """a column that is not there must not shift the bins"""
        self.assertEqual(plots._inner([], 3), [0., 0., 0.])
        self.assertEqual(plots._inner([1.0, 2.0], 3), [1.0, 2.0, 0.])

    def test_values_are_differential(self):
        """the y axis is dsigma/dx, as in MadBoard's plots (not HwU)"""
        self.assertEqual(plots._scaled([99.0, 10.0, 30.0, 88.0], 2, 50.0),
                         [0.2, 0.6])
        self.assertEqual(plots._scaled([0.0, 1.0, 0.0], 1, 0.), [0.])

    def test_step_covers_the_last_edge(self):
        self.assertEqual(plots._stepped([1.0, 2.0]), [1.0, 2.0, 2.0])
        self.assertEqual(plots._stepped([]), [0.])

    def test_axis_scale_follows_the_spectrum(self):
        """a falling spectrum gets a log axis, a flat one does not"""
        self.assertEqual(plots._y_scale([1.0, 1.2, 0.9]), 'linear')
        self.assertEqual(plots._y_scale([1000.0, 1.0]), 'log')
        # empty bins are ignored rather than forcing a linear axis
        self.assertEqual(plots._y_scale([1000.0, 1.0, 0.0]), 'log')
        self.assertEqual(plots._y_scale([0.0, 5.0]), 'linear')
        # an interference: a log axis would hide the negative bins
        self.assertEqual(plots._y_scale([1000.0, 1.0, -3.0]), 'linear')

    def test_bands(self):
        bands = plots._bands(make_histogram(), 4, 25.0)
        self.assertEqual([label for label, _l, _h, _c in bands],
                         ['scale', 'PDF (NNPDF40_lo_as_01180)'])
        _label, low, high, _colour = bands[0]
        self.assertAlmostEqual(low[0], 0.8 * 1.0 / 25.0)
        self.assertAlmostEqual(high[3], 1.2 * 4.0 / 25.0)

    def test_file_names(self):
        self.assertEqual(plots._file_name('t_1-t_2-pair_mass'),
                         't_1-t_2-pair_mass.pdf')
        self.assertEqual(plots._file_name('a/b'), 'a_b.pdf')
        self.assertEqual(plots._file_name(''), 'histogram.pdf')


class TestPlotRendering(unittest.TestCase):
    """the files a run writes into Events/<run>/plots"""

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='test_mg7_plots')

    def tearDown(self):
        shutil.rmtree(self.path)

    def test_nothing_to_draw(self):
        """no histograms: no directory, and no backend needed to find out"""
        out = os.path.join(self.path, 'plots')
        self.assertEqual(plots.render([], out), [])
        self.assertFalse(os.path.exists(out))

    @unittest.skipUnless(plots.available(), 'matplotlib is not installed')
    def test_one_file_per_histogram(self):
        out = os.path.join(self.path, 'plots')
        written = plots.render(
            [make_histogram('jet-pt'), make_histogram('sqrt_s')], out)
        self.assertEqual(written, ['jet-pt.pdf', 'sqrt_s.pdf'])
        self.assertEqual(sorted(os.listdir(out)), ['jet-pt.pdf', 'sqrt_s.pdf'])
        for name in written:
            self.assertGreater(os.path.getsize(os.path.join(out, name)), 0)

    @unittest.skipUnless(plots.available(), 'matplotlib is not installed')
    def test_empty_bins_do_not_break_the_drawing(self):
        """a tail that ran out of events still has to draw"""
        out = os.path.join(self.path, 'plots')
        histogram = make_histogram('jet-pt', values=[5.0, 1.0, 0.0, 0.0])
        self.assertEqual(plots.render([histogram], out), ['jet-pt.pdf'])

    @unittest.skipUnless(plots.available(), 'matplotlib is not installed')
    def test_without_systematics(self):
        """no bands to draw, but the distribution is still a plot"""
        out = os.path.join(self.path, 'plots')
        histogram = make_histogram('sqrt_s')
        del histogram['scale_envelope']
        del histogram['pdf_uncertainty']
        self.assertEqual(plots.render([histogram], out), ['sqrt_s.pdf'])

    @unittest.skipUnless(plots.available(), 'matplotlib is not installed')
    def test_matplotlib_does_not_narrate_the_drawing(self):
        """matplotlib logs every font it embeds at DEBUG level: a dozen lines
        per plot on the run's stdout as soon as anything lets DEBUG through.

        Run in a subprocess, because the point is what a fresh interpreter
        with DEBUG logging prints -- the loggers of this one have been pinned
        by the tests above already.
        """
        script = (
            'import json, logging, os, shutil, sys\n'
            'logging.basicConfig(level=logging.DEBUG)\n'
            'sys.path.insert(0, %r)\n' % MG5DIR +
            'from madgraph.mg7 import plots\n'
            'out = %r\n' % os.path.join(self.path, 'sub') +
            'plots.render([%r], out)\n' % make_histogram()
        )
        result = subprocess.run([sys.executable, '-c', script],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('matplotlib', result.stdout + result.stderr)

    @unittest.skipUnless(plots.available(), 'matplotlib is not installed')
    def test_an_explicit_level_is_left_alone(self):
        """someone who silenced it harder keeps their setting"""
        quiet = logging.getLogger('matplotlib')
        saved = quiet.level
        quiet.setLevel(logging.CRITICAL)
        try:
            plots._pyplot()
            self.assertEqual(quiet.level, logging.CRITICAL)
        finally:
            quiet.setLevel(saved)

    def test_missing_backend_is_reported(self):
        """a run without matplotlib must be told, not crash"""
        saved = plots._pyplot

        def no_matplotlib():
            raise plots.BackendMissing("No module named 'matplotlib'")

        plots._pyplot = no_matplotlib
        try:
            self.assertFalse(plots.available())
            with self.assertRaises(plots.BackendMissing):
                plots.render([make_histogram()], os.path.join(self.path, 'p'))
        finally:
            plots._pyplot = saved
