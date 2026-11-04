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
"""When the native systematics vary the MLM pdf ladder.

The kernel walks the ladder under scale_scheme "madevent", and under
"clustering_mean" with clustering_mean_pdf_ladder (the default), both only with
pdf_reweighting. The systematics have to vary the ladder in exactly those
cases: without it, a ladder sample's mu_F variations would leave the ladder's
lowered leading density and its ratios untouched.
"""
from __future__ import absolute_import

import unittest

from madgraph.various.banner import RunCardMG7


class TestMLMPdfLadder(unittest.TestCase):

    def setUp(self):
        try:
            import madgraph.mg7.launch as launch
        except Exception as error:  # madspace not built or importable here
            self.skipTest('mg7 launch not importable: %s' % error)
        self.launch = launch

    def ladder(self, mlm=True, is_decay=False, leptonic=False, **beam):
        rc = RunCardMG7()
        for key, value in beam.items():
            rc.set('beam.%s' % key, value, user=True)
        return self.launch.mlm_pdf_ladder(rc, mlm, is_decay, leptonic)

    def test_both_scale_schemes_walk_the_ladder_by_default(self):
        self.assertTrue(self.ladder(scale_scheme='madevent'))
        self.assertTrue(self.ladder(scale_scheme='clustering_mean'))

    def test_clustering_mean_without_the_ladder(self):
        self.assertFalse(self.ladder(scale_scheme='clustering_mean',
                                     clustering_mean_pdf_ladder=False))
        # the switch is only read under clustering_mean
        self.assertTrue(self.ladder(scale_scheme='madevent',
                                    clustering_mean_pdf_ladder=False))

    def test_pdf_reweighting_off_turns_it_off_everywhere(self):
        for scheme in ('madevent', 'clustering_mean'):
            self.assertFalse(self.ladder(scale_scheme=scheme, pdf_reweighting=False))

    def test_only_merged_hadronic_production(self):
        self.assertFalse(self.ladder(mlm=False))
        self.assertFalse(self.ladder(is_decay=True))
        self.assertFalse(self.ladder(leptonic=True))


if __name__ == '__main__':
    unittest.main()
