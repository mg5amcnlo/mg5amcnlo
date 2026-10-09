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
"""Test the summary printed by the systematics module"""

from __future__ import absolute_import
import unittest
from six import StringIO

import madgraph.various.systematics as systematics


class FakePDFSet(object):
    """minimal stand-in for lhapdf.PDFSet"""

    def __init__(self, lhapdfID, size, name):
        self.lhapdfID = lhapdfID
        self.size = size
        self.name = name
        self.errorType = 'replicas'

    def uncertainty(self, values):
        class Err(object):
            pass
        err = Err()
        err.central = values[0]
        err.errplus = max(values) - values[0]
        err.errminus = values[0] - min(values)
        return err


class FakePDF(object):
    """minimal stand-in for lhapdf.PDF (member memberID of pdfset)"""

    def __init__(self, pdfset, memberID):
        self.pdfset = pdfset
        self.memberID = memberID
        self.lhapdfID = pdfset.lhapdfID + memberID

    def set(self):
        return self.pdfset


class FakeBanner(object):

    def __init__(self):
        self.run_card = {'pdlabel': 'lhapdf'}

    def get(self, card, name, default=None):
        return 'sum'


class TestPrintCrossSections(unittest.TestCase):
    """check the envelope summary of Systematics.print_cross_sections"""

    def get_summary(self, sign, central=None):
        """run print_cross_sections on a fixed set of variations, all the
        cross-sections being multiplied by sign (and the central one
        replaced by central if given)"""

        pdfset = FakePDFSet(1000, 3, 'FakeSet')
        pdf0, pdf1, pdf2 = [FakePDF(pdfset, i) for i in range(3)]
        obj = systematics.Systematics.__new__(systematics.Systematics)
        obj.banner = FakeBanner()
        obj.orig_pdf = pdf0
        obj.orig_dyn = -1
        obj.pdfsets = {pdfset.lhapdfID: pdfset}
        obj.log = lambda x: None
        # (mur, muf, alps, dyn, pdf), cross-section
        variations = [((1, 1, 1, -1, pdf0), 100.),
                      ((2, 2, 1, -1, pdf0), 90.),
                      ((0.5, 0.5, 1, -1, pdf0), 120.),
                      ((1, 1, 2, -1, pdf0), 95.),
                      ((1, 1, 0.5, -1, pdf0), 104.),
                      ((1, 1, 1, 1, pdf0), 80.),
                      ((2, 2, 1, 1, pdf0), 72.),
                      ((0.5, 0.5, 1, 1, pdf0), 96.),
                      ((1, 1, 1, -1, pdf1), 103.),
                      ((1, 1, 1, -1, pdf2), 98.)]
        obj.args = [arg for arg, _ in variations]
        all_cross = [sign * cross for _, cross in variations]
        if central is not None:
            all_cross[0] = central
        stdout = StringIO()
        obj.print_cross_sections(all_cross, len(all_cross), stdout)
        return stdout.getvalue()

    def test_positive_cross_section(self):
        """the summary of a positive cross-section"""

        text = self.get_summary(1)
        self.assertIn('#     scale variation: +20% -10%\n', text)
        self.assertIn('#     emission scale variation: + 4% - 5%\n', text)
        self.assertIn('#     central scheme variation: + 0% -20%\n', text)
        self.assertIn('# PDF variation: + 3% - 2%\n', text)
        self.assertIn('# dynamical scheme # 1 : 80 +20% -10% #', text)

    def test_negative_cross_section(self):
        """a negative cross-section (an interference for instance) gets the
        same lines, in percent of its size: '+' is the increase of the
        cross-section, so '+' and '-' are swapped compared to the positive
        case"""

        text = self.get_summary(-1)
        self.assertIn('# original cross-section: -100.0\n', text)
        self.assertIn('#     scale variation: +10% -20%\n', text)
        self.assertIn('#     emission scale variation: + 5% - 4%\n', text)
        self.assertIn('#     central scheme variation: +20% - 0%\n', text)
        self.assertIn('# PDF variation: + 2% - 3%\n', text)
        self.assertIn('# dynamical scheme # 1 : -80 +10% -20% #', text)
        self.assertNotIn('+-', text)
        self.assertNotIn('--', text)

    def test_zero_central_cross_section(self):
        """no relative variation around a zero central cross-section: the
        percentages are undefined (nan), the dynamical schemes with a non
        zero central value of their own are still reported"""

        text = self.get_summary(1, central=0.)
        self.assertIn('# original cross-section: 0.0\n', text)
        self.assertIn('#     scale variation: +nan% -nan%\n', text)
        self.assertIn('#     emission scale variation: +nan% -nan%\n', text)
        self.assertIn('#     central scheme variation: +nan% -nan%\n', text)
        self.assertIn('# PDF variation: +nan% -nan%\n', text)
        self.assertIn('# dynamical scheme # 1 : 80 +20% -10% #', text)
