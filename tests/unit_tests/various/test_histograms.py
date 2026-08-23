################################################################################
#
# Copyright (c) 2011 The MadGraph5_aMC@NLO Development team and Contributors
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
from __future__ import division

from __future__ import absolute_import
import madgraph.various.histograms as histograms
import inspect
import os
import subprocess
import sys
import unittest
from unittest import mock
import copy
import tests.IOTests as IOTests
import madgraph.various.misc as misc

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_HwU_source = os.path.join(_file_path,os.pardir,'input_files','MADatNLO.HwU')
pjoin = os.path.join

class TestHistograms(unittest.TestCase):
    """Test that Histograms are correctly read, parsed, manipulated, written
    out."""
    
    def setUp(self):
        """ Load the histograms"""

        # load the base histograms
        self.histo_list = histograms.HwUList(_HwU_source)
    
    def test_histograms_operations(self):
        """ We test that basic operations are correctly handled """
        
        histo_list = copy.copy(self.histo_list)
    
        # "Testing 'Hist1 - 2.0 + 2.0 == Hist1'"
        my_hist = histo_list[0]+2.0
        my_hist = my_hist-2.0
        self.assertLess(
            abs(2.0-(my_hist.bins[0].wgts['central']/histo_list[0].bins[0].wgts['central'])-\
        (my_hist.bins[0].wgts[('scale',1.0,2.0)]/histo_list[0].bins[0].wgts[('scale',1.0,2.0)])),
            1.0e-14
        )

        # "Testing 'Hist1 - Hist2 + Hist2 == Hist1'"
        my_hist = histo_list[0]+histo_list[1]
        my_hist = my_hist-histo_list[1]
        self.assertLess(
            abs(2.0-(my_hist.bins[0].wgts['central']/histo_list[0].bins[0].wgts['central'])-\
        (my_hist.bins[0].wgts[('scale',1.0,2.0)]/histo_list[0].bins[0].wgts[('scale',1.0,2.0)])),
            1.0e-14
        )
        
        #"Testing 'Hist1 * 2.0 / 2.0 == Hist1'"
        my_hist = histo_list[0]*2.0
        my_hist = my_hist/2.0
        self.assertLess(
            abs(2.0-(my_hist.bins[0].wgts['central']/histo_list[0].bins[0].wgts['central'])-\
        (my_hist.bins[0].wgts[('scale',1.0,2.0)]/histo_list[0].bins[0].wgts[('scale',1.0,2.0)])),
            1.0e-14
        )

        #"Testing 'Hist1 * Hist2 / Hist2 == Hist1'"
        my_hist = histo_list[0]*histo_list[1]
        my_hist = my_hist/histo_list[1]
        self.assertLess(
            abs(2.0-(my_hist.bins[0].wgts['central']/histo_list[0].bins[0].wgts['central'])-\
        (my_hist.bins[0].wgts[('scale',1.0,2.0)]/histo_list[0].bins[0].wgts[('scale',1.0,2.0)])),
            1.0e-14
        )
    
    def test_output_reload(self):
        """ Outputs existing HwU histograms in the gnuplot format and makes sure
        that they remain identical when reloading them."""
        
        one_histo = histograms.HwUList([copy.copy(self.histo_list[0])])
        
        with misc.TMP_directory() as tmpdir:
            one_histo.output(pjoin(tmpdir,'OUT'), format='gnuplot')
            new_histo = histograms.HwUList(pjoin(tmpdir,'OUT.HwU'))
        
        one_histo = one_histo[0][0]
        one_histo.trim_auxiliary_weights()
        new_histo = new_histo[0]
        self.assertEqual(one_histo.type, new_histo.type)
        self.assertEqual(one_histo.title,new_histo.title)
        self.assertEqual(one_histo.x_axis_mode,new_histo.x_axis_mode)
        self.assertEqual(one_histo.y_axis_mode,new_histo.y_axis_mode)
        self.assertEqual(one_histo.bins.weight_labels,
                                                  new_histo.bins.weight_labels)
        self.assertEqual(len(one_histo.bins),len(new_histo.bins))
        for i, bin in enumerate(one_histo.bins):
             self.assertEqual(set(bin.wgts.keys()),
                                             set(new_histo.bins[i].wgts.keys()))
             for label, wgt in bin.wgts.items():
                 self.assertEqual(wgt,new_histo.bins[i].wgts[label])


class TestHistogramRegressions(unittest.TestCase):
    """Regression tests for histogram arithmetic and output helpers."""

    @staticmethod
    def make_hwu(labels=None, values=None, title='same', x_axis_mode='LIN',
                 y_axis_mode='LOG', n_bins=1):
        if labels is None:
            labels = ['central', 'stat_error']
        if values is None:
            values = [1.0, 0.1]
        bins = []
        for i in range(n_bins):
            bins.append(histograms.Bin((float(i), float(i+1)),
                              dict(zip(labels, values))))
        histo = histograms.HwU(title=title, x_axis_mode=x_axis_mode,
                                             y_axis_mode=y_axis_mode)
        histo.bins = histograms.BinList(bins, weight_labels=list(labels))
        return histo

    def test_bin_range_requires_positive_width(self):
        for width in [0.0, -1.0]:
            with self.assertRaises(histograms.MadGraph5Error):
                histograms.BinList(bin_range=[0.0, 1.0, width])

    def test_statistical_errors_remain_nonnegative(self):
        divided = histograms.Histogram.DIVIDE(
            {'central': 2.0, 'stat_error': 0.2},
            {'central': -4.0, 'stat_error': 0.4})
        self.assertAlmostEqual(divided['stat_error'],
                                             0.07071067811865477)

        rescaled = histograms.Histogram.RESCALE(-2.0)(
                              {'central': 2.0, 'stat_error': 0.2})
        self.assertEqual(rescaled['central'], -4.0)
        self.assertEqual(rescaled['stat_error'], 0.4)

    def test_compatibility_checks_axes_and_bin_count(self):
        reference = self.make_hwu()
        different_axis = self.make_hwu(x_axis_mode='LOG')
        different_bin_count = self.make_hwu(n_bins=2)

        self.assertFalse(reference.test_plot_compability(different_axis))
        self.assertFalse(reference.test_plot_compability(different_bin_count))

    def test_mur_uncertainties_keep_dynamic_scale_labels(self):
        labels = [
            'central', 'stat_error',
            ('scale_adv', 0, 0.5, 1.0),
            ('scale_adv', 0, 2.0, 1.0),
            ('scale_adv', 1, 0.5, 1.0),
            ('scale_adv', 1, 2.0, 1.0),
            ('scale', 0.5, 1.0),
            ('scale', 2.0, 1.0)]
        histo = self.make_hwu(labels,
                     [1.0, 0.1, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0])

        histo.set_uncertainty('MUR')

        self.assertEqual(
            [histo.bins[0].wgts[label] for label in
                ['delta_mur_cen 0 @aux', 'delta_mur_min 0 @aux',
                 'delta_mur_max 0 @aux']],
            [10.0, 10.0, 11.0])
        self.assertEqual(
            [histo.bins[0].wgts[label] for label in
                ['delta_mur_cen 1 @aux', 'delta_mur_min 1 @aux',
                 'delta_mur_max 1 @aux']],
            [20.0, 20.0, 21.0])

    def test_pdf_fallback_handles_small_and_special_sets(self):
        small = self.make_hwu(
              ['central', 'stat_error', ('pdf', 1000)], [1.0, 0.1, 1.1])

        special_labels = ['central', 'stat_error'] + [
                                  ('pdf', 244400+i) for i in range(101)]
        special_values = [100.0, 0.1, 100.0] + [
                                  float(101+i) for i in range(100)]
        special = self.make_hwu(special_labels, special_values)

        with mock.patch.object(histograms.subprocess, 'check_output',
                               side_effect=OSError) as check_output, \
             mock.patch.object(histograms.subprocess, 'Popen') as popen, \
             mock.patch.object(histograms.logger, 'warning'):
            small.set_uncertainty('pdf')
            special.set_uncertainty('PDF')

        self.assertEqual(check_output.call_count, 2)
        popen.assert_not_called()
        self.assertEqual(small.bins[0].wgts['delta_pdf_min @aux'], 1.1)
        self.assertEqual(small.bins[0].wgts['delta_pdf_max @aux'], 1.1)
        self.assertEqual(special.bins[0].wgts['delta_pdf_min @aux'], 32.0)
        self.assertEqual(special.bins[0].wgts['delta_pdf_max @aux'], 168.0)

    def test_ratio_helper_uses_denominator_path(self):
        constructor_paths = []

        class FakeHistogram(object):
            def __init__(self, path):
                self.path = path

            def get(self, name):
                if name == 'bins':
                    return [0.0]
                if self.path == 'numerator.HwU':
                    return [6.0]
                return [3.0]

        class FakeHwUList(object):
            def __init__(self, path, raw_labels=False):
                constructor_paths.append(path)
                self.path = path

            def get(self, name):
                return FakeHistogram(self.path)

        class FakeAxes(object):
            plotted_ratio = None

            def plot(self, bins, ratio, *args, **opts):
                self.plotted_ratio = ratio

        axes = FakeAxes()
        with mock.patch.object(histograms, 'HwUList', FakeHwUList):
            histograms.plot_ratio_from_HWU(
                'numerator.HwU', axes, 'plot', 'num', 'den',
                hwu_denominator_path='denominator.HwU')

        self.assertEqual(constructor_paths,
                         ['numerator.HwU', 'denominator.HwU'])
        self.assertEqual(axes.plotted_ratio, [2.0])

    def test_matplotlib_output_writes_standalone_renderer(self):
        labels = ['central', 'stat_error',
                  ('scale', 0.5, 1.0), ('scale', 2.0, 1.0)]
        first = self.make_hwu(labels, [2.0, 0.2, 1.8, 2.2],
                              title='matplotlib plot', n_bins=2)
        second = self.make_hwu(labels, [1.0, 0.1, 0.9, 1.1],
                               title='matplotlib plot', n_bins=2)
        first.type = 'NLO'
        second.type = 'LO'

        with misc.TMP_directory() as tmpdir:
            output_base = pjoin(tmpdir, 'matplotlib_output')
            histograms.HwUList([first, second]).output(
                output_base, format='matplotlib', auto_open=False,
                uncertainties=['scale', 'statistical'])

            hwu_path = output_base+'.HwU'
            script_path = output_base+'.py'
            self.assertTrue(os.path.exists(hwu_path))
            self.assertTrue(os.path.exists(script_path))
            with open(script_path) as stream:
                script = stream.read()
            namespace = {
                '__file__': script_path,
                '__name__': 'generated_matplotlib_renderer_test'}
            exec(compile(script, script_path, 'exec'), namespace)

            self.assertEqual(namespace['DATA_FILE'],
                             'matplotlib_output.HwU')
            self.assertEqual(namespace['PDF_FILE'],
                             'matplotlib_output.pdf')
            self.assertFalse(namespace['OPEN_AFTER_RENDER'])
            self.assertEqual(len(namespace['PLOTS']), 1)
            plot = namespace['PLOTS'][0]
            self.assertEqual(len(plot['main']), 2)
            self.assertEqual(len(plot['ratios']), 1)
            self.assertEqual(plot['y_axis_mode'], 'LOG')
            self.assertTrue(plot['relative_uncertainties'])
            self.assertTrue(any(uncertainty['band'] for uncertainty in
                                plot['main'][0]['uncertainties']))
            self.assertEqual(len(namespace['_read_hwu_blocks'](hwu_path)), 3)

            # Empty component histograms are common in aMC@NLO output. They
            # must not disable the logarithmic scale requested by Y_AXIS@LOG.
            axis = mock.MagicMock()
            axis.get_legend_handles_labels.return_value = ([], [])
            namespace['_draw_main'](axis, {
                'main': [{
                    'block': 0,
                    'label': 'empty component',
                    'color': 'black',
                    'statistical': False,
                    'uncertainties': []}],
                'x_axis_mode': 'LIN',
                'y_axis_mode': 'LOG'},
                [[[0.0, 1.0, 0.0, 0.0]]])
            axis.set_yscale.assert_called_once_with('log')

            cli_base = pjoin(tmpdir, 'cli_matplotlib')
            result = subprocess.run(
                [sys.executable, histograms.__file__, hwu_path,
                 '--matplotlib', '--no_open', '--out='+cli_base,
                 '--only_stat'], stdout=subprocess.PIPE,
                 stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertTrue(os.path.exists(cli_base+'.HwU'))
            self.assertTrue(os.path.exists(cli_base+'.py'))

    def test_uncertainty_output_defaults_and_missing_merging_weights(self):
        output_default = inspect.signature(histograms.HwUList.output).\
                            parameters['uncertainties'].default
        group_default = inspect.signature(histograms.HwUList.output_group).\
                            parameters['uncertainties'].default
        for default in [output_default, group_default]:
            self.assertIn('statistical', default)
            self.assertNotIn('statitistical', default)

        hwu_output = []
        gnuplot_output = []
        histo_list = histograms.HwUList([self.make_hwu()])
        histo_list.output_group(hwu_output, gnuplot_output, 0, 'plots.HwU',
                                uncertainties=['merging_scale'])
        self.assertNotIn('Relative scale and PDF uncertainty',
                         '\n'.join(gnuplot_output))

    def test_cli_accepts_run_id_and_rejects_incompatible_sums(self):
        xml_template = """<histfile>
<run id="{run_id}" header="xmin;xmax;Weight;WeightError">
<jethistograms njet="0">
<histogram name="{name}" unit="pb" weight="all">
0.0 1.0 {value} 0.1
</histogram>
</jethistograms>
</run>
{remainder}</histfile>
"""
        second_run = """<run id="1" header="xmin;xmax;Weight;WeightError">
<jethistograms njet="0">
<histogram name="run_one" unit="pb" weight="all">
0.0 1.0 2.0 0.1
</histogram>
</jethistograms>
</run>
"""

        with misc.TMP_directory() as tmpdir:
            xml_path = pjoin(tmpdir, 'runs.xml')
            with open(xml_path, 'w') as stream:
                stream.write(xml_template.format(run_id=0, name='run_zero',
                              value=1.0, remainder=second_run))
            output_base = pjoin(tmpdir, 'selected')
            result = subprocess.run(
                [sys.executable, histograms.__file__,
                 xml_path+'@run_id=1', '--out='+output_base, '--HwU',
                 '--no_open'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                 universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            with open(output_base+'.HwU') as stream:
                selected_output = stream.read()
            self.assertIn('run_one pb', selected_output)
            self.assertNotIn('run_zero pb', selected_output)

            first_path = pjoin(tmpdir, 'first.HwU')
            second_path = pjoin(tmpdir, 'second.HwU')
            self.make_hwu(title='first').output(first_path)
            self.make_hwu(title='second').output(second_path)
            result = subprocess.run(
                [sys.executable, histograms.__file__, first_path, second_path,
                 '--sum', '--no_suffix', '--out='+pjoin(tmpdir, 'sum'),
                 '--HwU', '--no_open'], stdout=subprocess.PIPE,
                 stderr=subprocess.PIPE, universal_newlines=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('are not compatible and cannot be combined',
                          result.stdout+result.stderr)
