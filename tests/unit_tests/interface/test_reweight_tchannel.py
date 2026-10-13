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
"""The reweighting must use the t-channel widths of the generation.

madevent sets the t-channel widths to zero (zerowidth_tchannel) since 2.8.0,
aMC@NLO keeps them. The standalone_rw matrix elements follow the sample.
"""

from __future__ import absolute_import
import os
import re
import unittest

import madgraph.interface.reweight_interface as reweight_interface
import madgraph.various.lhe_parser as lhe_parser

pjoin = os.path.join
_input_path = pjoin(os.path.dirname(os.path.dirname(os.path.dirname(
                                  os.path.realpath(__file__)))), 'input_files')


class TestKeepTchannelWidth(unittest.TestCase):

    keep = staticmethod(reweight_interface.ReweightInterface.keep_tchannel_width)

    def get_banner(self, name):
        return lhe_parser.EventFile(pjoin(_input_path, name)).get_banner()

    def test_recent_madevent_sample_has_zero_width(self):
        event_banner = self.get_banner('ttbar.lhe.gz')
        event_banner['mg5proccard'] = re.sub(r'VERSION\s+2\.5\.5', 'VERSION 3.8.1',
                                             event_banner['mg5proccard'])
        self.assertFalse(self.keep(event_banner))

    def test_madevent_sample_before_2_8_keeps_width(self):
        # generated with 2.5.5
        self.assertTrue(self.keep(self.get_banner('ttbar.lhe.gz')))

    def test_amcatnlo_sample_keeps_width(self):
        # generated with 3.3.0
        self.assertTrue(self.keep(self.get_banner('ttbar_nlo.lhe.gz')))
