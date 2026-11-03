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
"""The mg7 event-sample histograms written in the HwU format (MADatLO.HwU).

The point of the format is that the rest of MadGraph reads it, so the tests
that matter here parse the result back with madgraph.various.histograms --
the very code that plots an aMC@NLO MADatNLO.HwU -- rather than only checking
the text against itself.
"""

from __future__ import absolute_import
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import madgraph.various.histograms as histograms
from madgraph.mg7 import hwu_output


def make_histogram(name='jet-pt', bin_count=2, weights=True):
    """One histogram in the shape EventHistograms::to_json() writes: the
    arrays carry the underflow first and the overflow last."""
    out = {
        'name': name,
        'min': 0.0,
        'max': 100.0,
        'bin_count': bin_count,
        #        under  bin1  bin2  over
        'bin_values': [0.5, 10.0, 30.0, 1.5],
        'bin_errors': [0.05, 1.0, 3.0, 0.15],
        'scale_envelope': {'low': [0.4, 8.0, 24.0, 1.2],
                           'high': [0.6, 12.0, 36.0, 1.8]},
        'pdf_uncertainty': [{'pdf_set': 'NNPDF40_lo_as_01180',
                             'pdf_lhaid': 331900,
                             'error_type': 'replicas',
                             # member 0, not the nominal value: the PDF
                             # set of the variations need not be the run's
                             'central': [0.5, 10.5, 31.0, 1.5],
                             'uncertainty_down': [0.05, 1.0, 2.0, 0.1],
                             'uncertainty_up': [0.05, 2.0, 3.0, 0.1]}],
        'weights': [],
    }
    if weights:
        out['weights'] = [
            {'id': 1, 'bin_values': [0.4, 8.0, 24.0, 1.2], 'bin_errors': [0.] * 4},
            {'id': 2, 'bin_values': [0.6, 12.0, 36.0, 1.8], 'bin_errors': [0.] * 4},
            {'id': 3, 'bin_values': [0.5, 10.0, 30.0, 1.5], 'bin_errors': [0.] * 4},
            {'id': 4, 'bin_values': [0.5, 11.0, 29.0, 1.5], 'bin_errors': [0.] * 4},
        ]
    return out


SUMMARY = {
    'scale': {'weight_ids': [1, 2]},
    'pdf': [{'pdf_lhaid': 331900, 'error_type': 'replicas',
             'weight_ids': [3, 4]}],
    'variations': [
        {'id': 1, 'mur': 0.5, 'muf': 2.0, 'dyn': -1,
         'pdf_lhaid': 331900, 'pdf_member': 0},
        {'id': 2, 'mur': 2.0, 'muf': 0.5, 'dyn': -1,
         'pdf_lhaid': 331900, 'pdf_member': 0},
        {'id': 3, 'mur': 1.0, 'muf': 1.0, 'dyn': -1,
         'pdf_lhaid': 331900, 'pdf_member': 0},
        {'id': 4, 'mur': 1.0, 'muf': 1.0, 'dyn': -1,
         'pdf_lhaid': 331900, 'pdf_member': 1},
    ],
}


class TestHwUOutput(unittest.TestCase):
    """hwu_output.to_hwu: the text it writes"""

    def header(self, text):
        return [c.strip() for c in text.splitlines()[0][4:].split('&')]

    def bin_rows(self, text, index=0):
        rows, inside = [], False
        for line in text.splitlines():
            if line.startswith('<histogram>'):
                inside = True
                continue
            if line.startswith('<\\histogram>'):
                if inside and not index:
                    return rows
                inside = False
                index -= 1
                rows = []
                continue
            if inside and line.strip():
                rows.append([float(x) for x in line.split()])
        return rows

    def test_header_columns(self):
        text = hwu_output.to_hwu([make_histogram()], SUMMARY)
        self.assertEqual(self.header(text),
                         ['xmin', 'xmax', 'central value', 'dy',
                          'delta_mu_min @aux', 'delta_mu_max @aux',
                          'delta_pdf_min @aux', 'delta_pdf_max @aux',
                          'dyn=-1 muR= 1.000 muF= 1.000',
                          'dyn=-1 muR= 0.500 muF= 2.000',
                          'dyn=-1 muR= 2.000 muF= 0.500',
                          'PDF= 331900', 'PDF= 331901'])

    def test_pdf_central_member_is_not_a_scale_variation(self):
        """the central member carries mur = muf = 1: the group decides"""
        labels = hwu_output.weight_labels(SUMMARY)
        self.assertEqual(labels[3], 'PDF= 331900')
        self.assertEqual(labels[4], 'PDF= 331901')

    def test_dynamical_scale_label(self):
        summary = {'scale': {'weight_ids': [1]},
                   'variations': [{'id': 1, 'mur': 0.5, 'muf': 0.5, 'dyn': 3}]}
        self.assertEqual(hwu_output.weight_labels(summary)[1],
                         'dyn=3 muR= 0.500 muF= 0.500')

    def test_central_scale_column(self):
        """aMC@NLO lists muR = muF = 1 among its scale columns: so does this"""
        text = hwu_output.to_hwu([make_histogram()], SUMMARY)
        header = self.header(text)
        central = header.index('dyn=-1 muR= 1.000 muF= 1.000')
        for row in self.bin_rows(text):
            self.assertEqual(row[central], row[header.index('central value')])

    def test_central_scale_column_only_once(self):
        """not added when a variation is the central scale already, nor
        without scale variations"""
        summary = {'scale': {'weight_ids': [1, 2]},
                   'variations': [{'id': 1, 'mur': 1.0, 'muf': 1.0},
                                  {'id': 2, 'mur': 2.0, 'muf': 2.0}]}
        header = self.header(hwu_output.to_hwu([make_histogram()], summary))
        self.assertEqual(header.count('dyn=-1 muR= 1.000 muF= 1.000'), 1)
        summary = {'pdf': SUMMARY['pdf'], 'variations': SUMMARY['variations']}
        header = self.header(hwu_output.to_hwu([make_histogram()], summary))
        self.assertFalse([h for h in header if h.startswith('dyn=')])

    def test_values_are_sigma_per_bin(self):
        """HwU holds the cross section in the bin, as aMC@NLO writes it:
        sum(value) is the cross section in the range"""
        text = hwu_output.to_hwu([make_histogram()], SUMMARY)
        rows = self.bin_rows(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r[0] for r in rows], [0.0, 50.0])
        self.assertEqual([r[1] for r in rows], [50.0, 100.0])
        # 10 pb in a 50 GeV wide bin is written as 10, not as 10/50 pb/GeV
        self.assertEqual([r[2] for r in rows], [10.0, 30.0])
        self.assertEqual([r[3] for r in rows], [1.0, 3.0])
        self.assertAlmostEqual(sum(r[2] for r in rows), 40.0)
        # every column alike: the envelope and the variations too
        header = self.header(text)
        self.assertEqual([r[header.index('delta_mu_min @aux')] for r in rows],
                         [8.0, 24.0])
        self.assertEqual([r[header.index('PDF= 331901')] for r in rows],
                         [11.0, 29.0])

    def test_edges_have_amcatnlo_precision(self):
        """aMC@NLO writes its edges with Fortran e14.7, and histograms.py
        pairs histograms only when the edges agree exactly"""
        histogram = make_histogram()
        histogram.update({'min': 300.0, 'max': 1000.0, 'bin_count': 60})
        text = hwu_output.to_hwu([histogram], SUMMARY)
        rows = self.bin_rows(text)
        self.assertEqual(len(rows), 60)
        # what HwU_output prints for 300 + 700/60 in analysis_HwU_pp_ttx
        self.assertEqual(rows[0][1], float('0.3116667E+03'))
        self.assertEqual(rows[1][0], rows[0][1])
        self.assertEqual(rows[-1][1], 1000.0)
        self.assertIn('+3.1166670e+02', text)

    def test_underflow_and_overflow_are_dropped(self):
        """HwU has no such bins; an aMC@NLO analysis drops them too"""
        rows = self.bin_rows(hwu_output.to_hwu([make_histogram()], SUMMARY))
        self.assertNotIn(0.5, [r[2] for r in rows])   # the underflow
        self.assertNotIn(1.5, [r[2] for r in rows])   # the overflow

    def test_no_systematics(self):
        """without variations the file still holds the distribution"""
        histogram = make_histogram(weights=False)
        del histogram['scale_envelope']
        del histogram['pdf_uncertainty']
        text = hwu_output.to_hwu([histogram], None)
        self.assertEqual(self.header(text),
                         ['xmin', 'xmax', 'central value', 'dy'])
        self.assertEqual(len(self.bin_rows(text)), 2)

    def test_nothing_to_write(self):
        self.assertEqual(hwu_output.to_hwu([], SUMMARY), '')
        self.assertEqual(hwu_output.to_hwu(None, None), '')


class TestHwUOutputReadBack(unittest.TestCase):
    """the file is read by madgraph.various.histograms, which is the point"""

    def setUp(self):
        text = hwu_output.to_hwu(
            [make_histogram('jet-pt'), make_histogram('sqrt_s')], SUMMARY)
        handle, self.path = tempfile.mkstemp(suffix='.HwU')
        with os.fdopen(handle, 'w') as stream:
            stream.write(text)

    def tearDown(self):
        os.remove(self.path)

    def test_parsed_by_the_hwu_reader(self):
        parsed = histograms.HwUList(self.path, run_id=0)
        self.assertEqual([h.title for h in parsed], ['jet-pt', 'sqrt_s'])
        self.assertEqual(parsed[0].type, 'LO')
        self.assertEqual(len(parsed[0].bins), 2)
        self.assertAlmostEqual(parsed[0].bins[0].wgts['central'], 10.0)

    def test_weight_labels_are_recognised(self):
        """the labels must read as scale/PDF, or there is no band to draw"""
        parsed = histograms.HwUList(self.path, run_id=0)
        kinds = {}
        for label in parsed[0].bins.weight_labels:
            kind = histograms.HwU.get_HwU_wgt_label_type(label)
            kinds.setdefault(kind, []).append(label)
        self.assertEqual(sorted(kinds['scale_adv']),
                         [('scale_adv', -1, 0.5, 2.0),
                          ('scale_adv', -1, 1.0, 1.0),
                          ('scale_adv', -1, 2.0, 0.5)])
        self.assertEqual(sorted(kinds['pdf']), [('pdf', 331900), ('pdf', 331901)])

    def test_scale_envelope_survives_the_round_trip(self):
        """the envelope histograms.py rebuilds is the one madspace computed"""
        parsed = histograms.HwUList(self.path, run_id=0)
        histogram = parsed[0]
        histogram.set_uncertainty(type='all_scale')
        low = [l for l in histogram.bins[0].wgts
               if isinstance(l, str) and l.startswith('delta_mu_min')][0]
        high = [l for l in histogram.bins[0].wgts
                if isinstance(l, str) and l.startswith('delta_mu_max')][0]
        self.assertAlmostEqual(histogram.bins[0].wgts[low], 8.0)
        self.assertAlmostEqual(histogram.bins[0].wgts[high], 12.0)
        self.assertAlmostEqual(histogram.bins[1].wgts[low], 24.0)
        self.assertAlmostEqual(histogram.bins[1].wgts[high], 36.0)

    def test_scale_envelope_contains_the_central_value(self):
        """madspace's envelope starts from the central value; the rebuilt
        one does too, through the central-scale column"""
        histogram = make_histogram()
        # both variations above the central value in every bin
        histogram['weights'][0]['bin_values'] = [0.6, 11.0, 31.0, 1.6]
        histogram['weights'][1]['bin_values'] = [0.7, 12.0, 36.0, 1.8]
        with open(self.path, 'w') as stream:
            stream.write(hwu_output.to_hwu([histogram], SUMMARY))
        parsed = histograms.HwUList(self.path, run_id=0)[0]
        parsed.set_uncertainty(type='all_scale')
        low = [l for l in parsed.bins[0].wgts
               if isinstance(l, str) and l.startswith('delta_mu_min')][0]
        self.assertEqual([b.wgts[low] for b in parsed.bins], [10.0, 30.0])


# The header of the MADatNLO.HwU of an aMC@NLO fixed-order run,
# p p > t t~ [QCD] with analysis_HwU_pp_ttx, copied as written.
AMCATNLO_HEADER = (
    '##& xmin & xmax & central value & dy & delta_mu_cen -1 @aux & '
    'delta_mu_min -1 @aux & delta_mu_max -1 @aux & '
    'dyn=-1 muR= 1.000 muF= 1.000 & dyn=-1 muR= 2.000 muF= 1.000 & '
    'dyn=-1 muR= 0.500 muF= 1.000 & dyn=-1 muR= 1.000 muF= 2.000 & '
    'dyn=-1 muR= 2.000 muF= 2.000 & dyn=-1 muR= 0.500 muF= 2.000 & '
    'dyn=-1 muR= 1.000 muF= 0.500 & dyn=-1 muR= 2.000 muF= 0.500 & '
    'dyn=-1 muR= 0.500 muF= 0.500')

# the eight variations of the default mg7 systematics, central excluded
NINE_POINT_SUMMARY = {
    'scale': {'weight_ids': list(range(1, 9))},
    'variations': [{'id': i + 1, 'mur': mur, 'muf': muf, 'dyn': -1}
                   for i, (mur, muf) in enumerate(
                       [(mur, muf) for mur in (0.5, 1.0, 2.0)
                        for muf in (0.5, 1.0, 2.0)
                        if (mur, muf) != (1.0, 1.0)])],
}


def nine_point_histogram(name, central):
    """An mg7 histogram with the eight variations of NINE_POINT_SUMMARY:
    `central` holds the two inner bins, mu_R = 0.5 is 20% up, 2 is 20% down."""
    def inner(values):
        return [0.] + list(values) + [0.]
    weights = []
    for entry in NINE_POINT_SUMMARY['variations']:
        factor = {0.5: 1.2, 1.0: 1.0, 2.0: 0.8}[entry['mur']]
        weights.append({'id': entry['id'],
                        'bin_values': inner([c * factor for c in central]),
                        'bin_errors': [0.] * 4})
    return {'name': name, 'min': 0.0, 'max': 100.0, 'bin_count': 2,
            'bin_values': inner(central),
            'bin_errors': inner([0.01 * c for c in central]),
            'scale_envelope': {'low': inner([0.8 * c for c in central]),
                               'high': inner([1.2 * c for c in central])},
            'weights': weights}


def amcatnlo_hwu(title, central):
    """An HwU file as aMC@NLO writes it: sigma per bin, nine scale columns
    (mu_R = 0.5 20% up, 2 20% down, as in nine_point_histogram)."""
    labels = [h.strip() for h in AMCATNLO_HEADER[4:].split('&')]
    lines = [AMCATNLO_HEADER, '',
             '<histogram> 2 "%s |X_AXIS@LIN |Y_AXIS@LOG"' % title]
    for index, value in enumerate(central):
        row = [50. * index, 50. * (index + 1), value, 0.01 * value,
               value, 0.8 * value, 1.2 * value]
        for label in labels[7:]:
            mur = float(label.split('muR=')[1].split()[0])
            row.append(value * {0.5: 1.2, 1.0: 1.0, 2.0: 0.8}[mur])
        lines.append('  ' + '   '.join('%+.7e' % v for v in row))
    lines += ['<\\histogram>', '']
    return '\n'.join(lines)


class TestHwUOverlayOnAMCatNLO(unittest.TestCase):
    """what the file is for: an mg7 LO histogram drawn by histograms.py
    together with an aMC@NLO one of the same title and binning"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # 40 pb in the range for mg7 LO, 60 pb for aMC@NLO NLO
        self.mg7 = os.path.join(self.tmpdir, 'MADatLO.HwU')
        with open(self.mg7, 'w') as stream:
            stream.write(hwu_output.to_hwu(
                [nine_point_histogram('tt inv m', [10.0, 30.0])],
                NINE_POINT_SUMMARY))
        self.amc = os.path.join(self.tmpdir, 'MADatNLO.HwU')
        with open(self.amc, 'w') as stream:
            stream.write(amcatnlo_hwu('tt inv m', [15.0, 45.0]))

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_same_scale_columns_as_amcatnlo(self):
        """histograms.py stops on files with different weight columns"""
        mg7 = histograms.HwUList(self.mg7, run_id=0)[0]
        amc = histograms.HwUList(self.amc, run_id=0)[0]
        self.assertEqual(set(mg7.bins.weight_labels),
                         set(amc.bins.weight_labels))
        self.assertEqual(len([l for l in mg7.bins.weight_labels
                              if histograms.HwU.get_HwU_wgt_label_type(l)
                              == 'scale_adv']), 9)

    def test_same_normalisation_as_amcatnlo(self):
        """both hold sigma per bin, so their ratio is the K-factor and not
        the K-factor times a bin width"""
        mg7 = histograms.HwUList(self.mg7, run_id=0)[0]
        amc = histograms.HwUList(self.amc, run_id=0)[0]
        for lo, nlo in zip(mg7.bins, amc.bins):
            self.assertEqual(lo.boundaries, nlo.boundaries)
            self.assertAlmostEqual(nlo.wgts['central'] / lo.wgts['central'],
                                   1.5)

    def test_overlay_with_histograms_py(self):
        """one plot, two curves, a scale band on each, from the default
        command line"""
        out = os.path.join(self.tmpdir, 'overlay')
        result = subprocess.run(
            [sys.executable, histograms.__file__, self.mg7, self.amc,
             '--out=' + out, '--matplotlib', '--no_open',
             '--assign_types=LO,NLO'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        namespace = {'__file__': out + '.py',
                     '__name__': 'hwu_overlay_renderer_test'}
        with open(out + '.py') as stream:
            exec(compile(stream.read(), out + '.py', 'exec'), namespace)
        self.assertEqual(len(namespace['PLOTS']), 1)
        plot = namespace['PLOTS'][0]
        self.assertEqual(len(plot['main']), 2)
        for curve in plot['main']:
            self.assertIn('scale', [u['type'] for u in curve['uncertainties']
                                    if u['band']])

