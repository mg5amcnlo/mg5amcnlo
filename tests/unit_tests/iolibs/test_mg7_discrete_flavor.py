################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""The mg7 launcher's discrete flavor sampler.

The Integrand conditions the flavor probabilities on the PDF prior, and only
passes that condition when there are PDFs.  A leptonic process (or a decay)
with more than one flavor index -- `z > q q~` with both u- and d-type rows --
used to declare the prior anyway and died building the event generator with
"keys and values must have the same size".
"""
from __future__ import absolute_import
import types
import unittest

from madgraph.mg7 import bootstrap as mg7_bootstrap

# Importing launch without a compiled madspace starts the madspace build, and
# test_manager imports every unit test module while collecting: check first.
launch = None
if mg7_bootstrap.madspace_is_installed():
    try:
        import madgraph.mg7.launch as launch
    except Exception as error:
        launch = None
        import_error = error


@unittest.skipIf(launch is None, 'madspace not available')
class TestDiscreteFlavorPrior(unittest.TestCase):

    @staticmethod
    def stub(leptonic):
        process = types.SimpleNamespace(
            leptonic=leptonic, contexts=[],
            run_card={'phasespace': {'adaptive_symmetry_sampling': False}})
        return types.SimpleNamespace(process=process)

    def build(self, leptonic):
        sym, flavor = launch.MadgraphSubprocess.build_discrete(
            self.stub(leptonic), 1, 2, 'test')
        self.assertIsNone(sym)
        return flavor

    def test_leptonic_has_no_prior(self):
        """random input only: no PDF prior to condition on"""
        flavor = self.build(leptonic=True)
        self.assertEqual(len(flavor.forward_function().inputs), 1)

    def test_hadronic_has_prior(self):
        """random input + the PDF prior"""
        flavor = self.build(leptonic=False)
        self.assertEqual(len(flavor.forward_function().inputs), 2)

    def test_single_flavor_has_no_sampler(self):
        sym, flavor = launch.MadgraphSubprocess.build_discrete(
            self.stub(True), 1, 1, 'test')
        self.assertIsNone(flavor)


if __name__ == '__main__':
    unittest.main()
