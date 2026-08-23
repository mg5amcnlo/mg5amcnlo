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
import io
import os
import re
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
        
        # Output preparation must not replace the input with nested groups or
        # append ratios/auxiliary uncertainty curves to it.
        self.assertEqual(len(one_histo), 1)
        one_histo = one_histo[0]
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

    def test_scalar_operations_do_not_mutate_the_left_operand(self):
        original = self.make_hwu(values=[2.0, 0.2])
        result = original*3.0

        self.assertIsNot(result, original)
        self.assertEqual(original.bins[0].wgts['central'], 2.0)
        self.assertEqual(original.bins[0].wgts['stat_error'], 0.2)
        self.assertEqual(result.bins[0].wgts['central'], 6.0)
        self.assertAlmostEqual(result.bins[0].wgts['stat_error'], 0.6)

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
            offsets = namespace['_index_hwu_blocks'](hwu_path)
            self.assertEqual(len(offsets), 3)
            required = namespace['_required_columns'](plot)
            self.assertEqual(set(required), set([0, 1, 2]))
            self.assertNotIn('_read_hwu_blocks', namespace)
            loaded = namespace['_load_plot_blocks'](hwu_path, offsets, plot)
            for block_index, positions in required.items():
                self.assertEqual(set(loaded[block_index]['columns']), positions)

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
                {0: {'columns': {
                    0: [0.0], 1: [1.0], 2: [0.0]}}})
            axis.set_yscale.assert_called_once_with('log')

            negative_axis = mock.MagicMock()
            negative_axis.get_legend_handles_labels.return_value = ([], [])
            namespace['_draw_main'](negative_axis, {
                'main': [{
                    'block': 0,
                    'label': 'signed component',
                    'color': 'red',
                    'statistical': False,
                    'uncertainties': []}],
                'x_axis_mode': 'LIN',
                'y_axis_mode': 'LOG'},
                {0: {'columns': {
                    0: [0.0, 1.0], 1: [1.0, 2.0],
                    2: [-2.0, 3.0]}}})
            self.assertTrue(any(call.kwargs.get('linestyle') == '--'
                                for call in negative_axis.step.call_args_list))

            cli_base = pjoin(tmpdir, 'cli_matplotlib')
            result = subprocess.run(
                [sys.executable, histograms.__file__, hwu_path,
                 '--matplotlib', '--no_open', '--out='+cli_base,
                 '--only_stat'], stdout=subprocess.PIPE,
                 stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertTrue(os.path.exists(cli_base+'.HwU'))
            self.assertTrue(os.path.exists(cli_base+'.py'))

    def test_dense_streaming_sum_handles_order_schemas_and_mmap(self):
        labels = ['central', 'stat_error', 'weight_a', 'weight_b']
        reversed_labels = ['central', 'stat_error', 'weight_b', 'weight_a']
        first_a = self.make_hwu(labels, [1.0, 0.1, 10.0, 20.0],
                                title='A', n_bins=2)
        first_b = self.make_hwu(labels, [4.0, 0.4, 40.0, 50.0],
                                title='B', n_bins=2)
        second_a = self.make_hwu(
            reversed_labels, [2.0, 0.2, 30.0, 25.0],
            title='A', n_bins=2)
        second_b = self.make_hwu(
            reversed_labels, [5.0, 0.5, 60.0, 55.0],
            title='B', n_bins=2)

        with misc.TMP_directory() as tmpdir:
            first_base = pjoin(tmpdir, 'first')
            second_base = pjoin(tmpdir, 'second')
            histograms.HwUList([first_a, first_b]).output(
                                                first_base, format='HwU')
            # Deliberately reverse both histogram and weight-column order.
            histograms.HwUList([second_b, second_a]).output(
                                               second_base, format='HwU')

            aggregator = histograms.StreamingHwUAggregator(memory_limit=0)
            try:
                aggregator.add_file(first_base+'.HwU')
                aggregator.add_file(second_base+'.HwU')
                summed = aggregator.to_hwu_list()
                by_title = dict((histo.title, histo) for histo in summed)

                self.assertEqual(by_title['A'].bins[0].wgts['central'], 3.0)
                self.assertEqual(by_title['A'].bins[0].wgts['weight_a'], 35.0)
                self.assertEqual(by_title['A'].bins[0].wgts['weight_b'], 50.0)
                self.assertAlmostEqual(
                    by_title['A'].bins[0].wgts['stat_error'],
                    (0.1**2+0.2**2)**0.5)
                self.assertEqual(len(aggregator._spill_arenas), 1)
                self.assertIsNotNone(aggregator._spill_arenas[0]._mapping)

                lazy_sum = by_title['A']+by_title['B']
                self.assertIsInstance(lazy_sum.bins[0].wgts,
                                      histograms.LazyCombinedWeightView)
                self.assertEqual(lazy_sum.bins[0].wgts['central'], 12.0)
                by_title['A'].rebin(2)
                self.assertIsInstance(by_title['A'].bins[0].wgts,
                                      histograms.LazyRebinnedWeightView)
                self.assertEqual(by_title['A'].bins[0].wgts['central'], 6.0)

                output_base = pjoin(tmpdir, 'summed')
                aggregator.output(output_base, format='HwU')
                reloaded = histograms.HwUList(output_base+'.HwU')
                reloaded_a = next(histo for histo in reloaded
                                  if histo.title == 'A')
                self.assertEqual(reloaded_a.bins[0].wgts['weight_a'], 35.0)
                self.assertEqual(reloaded_a.bins[0].wgts['weight_b'], 50.0)

                cli_base = pjoin(tmpdir, 'cli_sum')
                result = subprocess.run([
                    sys.executable, histograms.__file__, first_base+'.HwU',
                    second_base+'.HwU', '--sum', '--HwU', '--no_open',
                    '--keep_all_weights', '--memory_limit=0',
                    '--out='+cli_base],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    universal_newlines=True)
                self.assertEqual(result.returncode, 0,
                                 result.stdout+result.stderr)
                cli_sum = histograms.HwUList(cli_base+'.HwU')
                cli_a = next(histo for histo in cli_sum if histo.title == 'A')
                self.assertEqual(cli_a.bins[0].wgts['weight_a'], 35.0)
            finally:
                aggregator.close()

    def test_dense_parser_rejects_malformed_and_nonfinite_blocks(self):
        header = '##& xmin & xmax & central value & dy\n\n'
        opening = '<histogram> 1 "bad |X_AXIS@LIN |Y_AXIS@LOG"\n'

        with misc.TMP_directory() as tmpdir:
            malformed_header = pjoin(tmpdir, 'malformed_header.HwU')
            with open(malformed_header, 'w') as stream:
                stream.write('##& xmin & xmax & central value\n'+opening+
                             '0 1 2\n<\\histogram>\n')
            with self.assertRaisesRegex(histograms.HwU.ParseError,
                                        'mandatory weight names'):
                list(histograms.iter_hwu_dense(malformed_header))

            bad_rows = [
                ('nonnumeric', '0 1 2 0.1x\n<\\histogram>\n',
                 'non-numeric'),
                ('nonfinite', '0 1 inf 0.1\n<\\histogram>\n',
                 'non-finite'),
                ('premature_close', '<\\histogram>\n',
                 'before its closing tag'),
                ('decreasing', '1 0 2 0.1\n<\\histogram>\n',
                 'decreasing boundaries')]
            for name, body, message in bad_rows:
                path = pjoin(tmpdir, name+'.HwU')
                with open(path, 'w') as stream:
                    stream.write(header+opening+body)
                with self.subTest(name=name), self.assertRaisesRegex(
                                      histograms.HwU.ParseError, message):
                    list(histograms.iter_hwu_dense(path))

            commented = pjoin(tmpdir, 'commented.HwU')
            with open(commented, 'w') as stream:
                stream.write(header+opening+
                             '0 1 2 0.1 # an allowed row comment\n'
                             '<\\histogram>\n')
            records = list(histograms.iter_hwu_dense(commented))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].n_bins, 1)

    def test_streaming_aggregator_fails_closed_and_releases_resources(self):
        first = histograms.HwUList([
            self.make_hwu(title='A'), self.make_hwu(title='B')])
        incomplete = histograms.HwUList([self.make_hwu(title='A')])

        with misc.TMP_directory() as tmpdir:
            first_path = pjoin(tmpdir, 'first.HwU')
            incomplete_path = pjoin(tmpdir, 'incomplete.HwU')
            first.output(pjoin(tmpdir, 'first'), format='HwU')
            incomplete.output(pjoin(tmpdir, 'incomplete'), format='HwU')

            aggregator = histograms.StreamingHwUAggregator(memory_limit=0)
            aggregator.add_file(first_path)
            with self.assertRaisesRegex(histograms.MadGraph5Error,
                                        'expected histogram.*missing'):
                aggregator.add_file(incomplete_path)
            self.assertEqual(aggregator.accumulators, {})
            with self.assertRaisesRegex(histograms.MadGraph5Error,
                                        'previous input failed'):
                aggregator.to_hwu_list()
            with self.assertRaisesRegex(histograms.MadGraph5Error,
                                        'previous input failed'):
                len(aggregator)
            aggregator.close()
            aggregator.close()

            with histograms.StreamingHwUAggregator(
                                             memory_limit=0) as aggregator:
                with self.assertRaisesRegex(histograms.MadGraph5Error,
                                            'factors must be finite'):
                    aggregator.add_file(first_path, factor=float('inf'))
                aggregator.add_file(first_path)
                self.assertEqual(len(aggregator), 2)

        with self.assertRaisesRegex(histograms.MadGraph5Error,
                                    'must be non-negative'):
            histograms.StreamingHwUAggregator(memory_limit=-1)

        buffer_ = histograms._DenseDoubleBuffer(
                                      2, use_mmap=True, numpy=None)
        self.assertIsNone(buffer_._file)
        buffer_[0] = 1.0
        buffer_.close()
        buffer_.close()

    def test_arithmetic_validates_lazy_schemas_and_operand_protocol(self):
        left = self.make_hwu(['central', 'stat_error'], [1.0, 0.1])
        right = self.make_hwu(
            ['central', 'stat_error', 'extra'], [1.0, 0.1, 2.0])

        with self.assertRaisesRegex(histograms.MadGraph5Error,
                                    'different weight labels'):
            left+right
        self.assertIs(left.__add__(object()), NotImplemented)
        self.assertIs(left.__sub__(object()), NotImplemented)
        self.assertIs(left.__mul__(object()), NotImplemented)
        self.assertIs(left.__truediv__(object()), NotImplemented)

        lazy = histograms.LazyCombinedWeightView(
            {'central': 1.0}, {'central': 2.0}, histograms.Histogram.ADD,
            ('central',))
        with self.assertRaises(KeyError):
            lazy['unknown']
        with self.assertRaisesRegex(histograms.MadGraph5Error, "Element '1'"):
            histograms.Bin((0.0, 1))
        with self.assertRaisesRegex(histograms.MadGraph5Error,
                                    'title.*must be a string'):
            histograms.HwU(title=1)
        with self.assertRaisesRegex(histograms.MadGraph5Error,
                                    'type.*string or None'):
            histograms.HwU(type=1)

    def test_failed_streamed_output_preserves_existing_files(self):
        first = self.make_hwu(title='first')
        incompatible = self.make_hwu(
            ['central', 'stat_error', 'extra'], [1.0, 0.1, 2.0],
            title='second')

        with misc.TMP_directory() as tmpdir:
            output_base = pjoin(tmpdir, 'atomic')
            with open(output_base+'.HwU', 'w') as stream:
                stream.write('existing data')
            with open(output_base+'.gnuplot', 'w') as stream:
                stream.write('existing script')
            groups = iter([
                histograms.HwUList([first]),
                histograms.HwUList([incompatible])])
            with self.assertRaisesRegex(histograms.MadGraph5Error,
                                        'different base column schema'):
                histograms.HwUList([]).output(
                    output_base, format='gnuplot', uncertainties=[],
                    _histogram_groups=groups,
                    _weight_schema=['central', 'stat_error'])
            with open(output_base+'.HwU') as stream:
                self.assertEqual(stream.read(), 'existing data')
            with open(output_base+'.gnuplot') as stream:
                self.assertEqual(stream.read(), 'existing script')

    def test_generated_matplotlib_reader_is_strict(self):
        script = histograms.HwUList._get_matplotlib_script(
                                      'strict', [], False, '')
        namespace = {'__file__': 'strict.py', '__name__': 'strict_reader_test'}
        exec(compile(script, 'strict.py', 'exec'), namespace)

        with misc.TMP_directory() as tmpdir:
            cases = [
                ('nonfinite', 1, '0 1 inf\n<\\histogram>\n',
                 'non-finite'),
                ('wrong_count', 2, '0 1 2\n<\\histogram>\n',
                 'contains 1 rows but declares 2'),
                ('unclosed', 1, '0 1 2\n', 'missing its closing tag')]
            for name, n_bins, body, message in cases:
                path = pjoin(tmpdir, name+'.HwU')
                with open(path, 'w') as stream:
                    stream.write('<histogram> %d "bad"\n%s'%(n_bins, body))
                offsets = namespace['_index_hwu_blocks'](path)
                with self.subTest(name=name), self.assertRaisesRegex(
                                                   SystemExit, message):
                    namespace['_read_hwu_block'](
                                        path, offsets[0], set([0, 1, 2]))

    def test_output_canonicalizes_weight_order_and_is_repeatable(self):
        labels = ['central', 'stat_error', 'weight_a', 'weight_b']
        reversed_labels = ['central', 'stat_error', 'weight_b', 'weight_a']
        first = self.make_hwu(labels, [1.0, 0.1, 10.0, 20.0])
        second = self.make_hwu(
            reversed_labels, [2.0, 0.2, 40.0, 30.0])
        first.type = 'NLO'
        second.type = 'LO'
        source = histograms.HwUList([first, second])

        with misc.TMP_directory() as tmpdir:
            for name in ['one', 'two']:
                source.output(pjoin(tmpdir, name), format='gnuplot',
                              uncertainties=[], number_of_ratios=0,
                              auto_open=False)
            reloaded = histograms.HwUList(pjoin(tmpdir, 'one.HwU'))

        self.assertEqual(len(source), 2)
        self.assertTrue(all(isinstance(item, histograms.HwU)
                            for item in source))
        self.assertEqual(reloaded[1].bins[0].wgts['weight_a'], 30.0)
        self.assertEqual(reloaded[1].bins[0].wgts['weight_b'], 40.0)

    def test_plot_ranges_and_ratio_labels_use_every_curve(self):
        curves = histograms.HwUList([
            self.make_hwu(values=[1.0, 0.1]),
            self.make_hwu(values=[2.0, 0.1]),
            self.make_hwu(values=[1000.0, 0.1])])
        curves[0].type = 'NLO'
        curves[1].type = 'LO'
        curves[2].type = 'LO1'
        hwu_output = []
        gnuplot_output = []
        curves.output_group(hwu_output, gnuplot_output, 0, 'plots.HwU',
                            uncertainties=[], number_of_ratios=0)
        yrange = next(line for line in gnuplot_output
                      if 'set yrange [' in line and 'rendering subhistograms' in line)
        upper = float(re.search(r'set yrange \[[^:]+:([^\]]+)\]',
                                yrange).group(1))
        self.assertGreater(upper, 1000.0)

        prepared = copy.deepcopy(curves)
        prepared.output_group([], [], 0, 'plots.HwU', uncertainties=[],
                              _copy_group=False)
        ratio_titles = [histo.title for histo in prepared
                        if histo.type == 'AUX']
        self.assertTrue(any(title.endswith('1/K-factor')
                            for title in ratio_titles))
        self.assertTrue(any(title.endswith('LO1/NLO')
                            for title in ratio_titles))

    def test_explicit_hessian_weight_list(self):
        labels = ['central', 'stat_error'] + [
            'MUF=1_MUR=1_PDF=%d_MERGING=30'%member
            for member in [1, 2, 3]]
        histo = self.make_hwu(labels, [10.0, 0.1, 9.0, 12.0, 11.0])

        minimum, maximum = histo.get_uncertainty_band(labels[2:],
                                                       mode='hessian')

        self.assertEqual(minimum, [9.0])
        self.assertEqual(maximum, [12.0])

    def test_compound_xml_variations_are_skipped_cleanly(self):
        xml_source = """<histfile>
<run id="0" header="xmin;xmax;Weight_PDF=260000_MUR=1_MUF=1_ALPSFACT=1_MERGING=30;WeightError;MUR=2_MUF=1_ALPSFACT=2_PDF=260001_MERGING=30">
<jethistograms njet="0">
<histogram name="compound" unit="pb" weight="all">
0.0 1.0 2.0 0.1 2.5
</histogram>
</jethistograms>
</run>
</histfile>
"""
        parsed = histograms.HwUList(io.StringIO(xml_source))

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].bins.weight_labels,
                         ['central', 'stat_error'])
        with misc.TMP_directory() as tmpdir:
            xml_path = pjoin(tmpdir, 'compound.xml')
            with open(xml_path, 'w') as stream:
                stream.write(xml_source)
            dense = list(histograms.iter_hwu_dense(xml_path))
        self.assertEqual(len(dense), 1)
        self.assertEqual(dense[0].weight_labels,
                         ('central', 'stat_error'))

    def test_log_range_for_one_value_is_multiplicative(self):
        histo = self.make_hwu(values=[0.5, 0.0])

        self.assertEqual(histograms.HwU.get_y_optimal_range(
            [histo], labels=['central'], scale='LOG'), (0.05, 5.0))
        gnuplot_output = []
        histograms.HwUList([histo]).output_group(
            [], gnuplot_output, 0, 'plot.HwU', uncertainties=[])
        self.assertEqual(histo.y_axis_mode, 'LOG')
        self.assertIn('set logscale y', '\n'.join(gnuplot_output))

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
