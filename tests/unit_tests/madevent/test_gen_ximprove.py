################################################################################
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
"""Unit tests for gen_ximprove refine-event reuse helpers."""

from __future__ import absolute_import
import unittest

import madgraph.madevent.gen_ximprove as gen_ximprove
import madgraph.various.banner as bannermod


class DummyRefine(object):

    format_variable = staticmethod(bannermod.ConfigFile.format_variable)


class TestGenXImprove(unittest.TestCase):

    def test_refine_iteration_estimate_keeps_previous_events(self):
        dummy = DummyRefine()
        nunwgt, maxwgt, new_evt, efficiency, drop = \
            gen_ximprove.gen_ximprove._estimate_refine_iteration_events(
                dummy, 200, 1000, 100, 1.9, 80, 2.0)

        self.assertFalse(drop)
        self.assertAlmostEqual(maxwgt, 2.0)
        self.assertAlmostEqual(new_evt, 95.0)
        self.assertAlmostEqual(nunwgt, 175.0)
        self.assertAlmostEqual(efficiency, 0.095)

    def test_refine_iteration_estimate_drops_previous_events(self):
        dummy = DummyRefine()
        nunwgt, maxwgt, new_evt, efficiency, drop = \
            gen_ximprove.gen_ximprove._estimate_refine_iteration_events(
                dummy, 200, 1000, 100, 1.0, 80, 5.0)

        self.assertTrue(drop)
        self.assertAlmostEqual(maxwgt, 1.0)
        self.assertAlmostEqual(new_evt, 100.0)
        self.assertAlmostEqual(nunwgt, 100.0)
        self.assertAlmostEqual(efficiency, 0.1)

    def test_reuse_previous_refine_events_defaults(self):
        normal = DummyRefine()
        normal.run_card = bannermod.RunCard()
        self.assertFalse(
            gen_ximprove.gen_ximprove._reuse_previous_refine_events(normal))

        normal.run_card['keep_previous_refine_events'] = True
        normal.run_card.user_set.add('keep_previous_refine_events')
        self.assertTrue(
            gen_ximprove.gen_ximprove._reuse_previous_refine_events(normal))

        shared = object.__new__(gen_ximprove.gen_ximprove_share)
        shared.run_card = bannermod.RunCard()
        shared.format_variable = staticmethod(bannermod.ConfigFile.format_variable)
        self.assertTrue(
            gen_ximprove.gen_ximprove._reuse_previous_refine_events(shared))

        shared.run_card['keep_previous_refine_events'] = False
        shared.run_card.user_set.add('keep_previous_refine_events')
        self.assertFalse(
            gen_ximprove.gen_ximprove._reuse_previous_refine_events(shared))
