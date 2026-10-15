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
"""Unit tests for the selection of the particles in the density mode of the reweighting"""
from __future__ import absolute_import
import types

import tests.unit_tests as unittest

import madgraph.interface.reweight_interface as rwgt_interface
import madgraph.various.lhe_parser as lhe_parser

chose = rwgt_interface.DensityInterface.chose_particle_user_input


class TestDensitySelection(unittest.TestCase):
    """Test the conversion of the user input (pdg codes, observable, ranks) into positions in the original order"""

    # g g > t t~ t t~ with the pt of the particles: 6 at position 2 and 4, -6 at position 3 and 5
    pdg = [21, 21, 6, -6, 6, -6]
    momenta = [(100., 0., 0., 100.), (100., 0., 0., -100.),
               (200., 10., 0., 0.), (200., 0., 30., 0.),
               (200., 0., 50., 0.), (200., 20., 0., 0.)]

    def test_density_selection_not_used(self):
        self.assertEqual(chose(self.momenta, self.pdg, ([0], '', []), 'opt'), [-1])

    def test_density_selection_default_order(self):
        """the j-th position corresponds to the j-th pdg code given by the user (not the order of the process)"""
        self.assertEqual(chose(self.momenta, self.pdg, ([-6, 6], '', []), 'opt'), [3, 2])
        self.assertEqual(chose(self.momenta, self.pdg, ([-6, 6], '', []), 'opt', fortran_format=True), [4, 3])
        self.assertEqual(chose(self.momenta, self.pdg, ([6, -6, 6], '', []), 'opt'), [2, 3, 4])
        # the first incoming particle is at the python position 0
        self.assertEqual(chose(self.momenta, self.pdg, ([21, 21], '', []), 'opt'), [0, 1])
        # a ranking without observable is ignored
        self.assertEqual(chose(self.momenta, self.pdg, ([6, 6], '', [1, 0]), 'opt'), [2, 4])

    def test_density_selection_observable_order(self):
        """without ranks the particles of the same pdg are taken in decreasing order of the observable"""
        self.assertEqual(chose(self.momenta, self.pdg, ([6, 6], 'pt', []), 'opt'), [4, 2])
        self.assertEqual(chose(self.momenta, self.pdg, ([-6, 6], 'pt', []), 'opt'), [3, 4])
        self.assertEqual(chose(self.momenta, self.pdg, ([6, -6, 6], 'pt', []), 'opt', fortran_format=True), [5, 4, 3])

    def test_density_selection_observable_ranks(self):
        """with ranks, the j-th position is the particle of pdg user_input[0][j] with rank user_input[2][j] (0 = largest)"""
        self.assertEqual(chose(self.momenta, self.pdg, ([6, 6], 'pt', [1, 0]), 'opt'), [2, 4])
        self.assertEqual(chose(self.momenta, self.pdg, ([6, -6], 'pt', [1, 1]), 'opt'), [2, 5])
        # interleaved pdg codes are not grouped by pdg
        self.assertEqual(chose(self.momenta, self.pdg, ([6, -6, 6], 'pt', [0, 0, 1]), 'opt'), [4, 3, 2])

    def test_density_selection_tie(self):
        """identical particles with the same value of the observable can not be ranked: the original order is used"""
        momenta = list(self.momenta)
        momenta[4] = (200., 0., 10., 0.) # same pt as the top at position 2
        with self.assertLogs(rwgt_interface.logger, level='WARNING'):
            self.assertEqual(chose(momenta, self.pdg, ([6, 6], 'pt', []), 'opt'), [2, 4])

    def test_density_selection_wrong_inputs(self):
        # pdg not in the process
        self.assertRaises(ValueError, chose, self.momenta, self.pdg, ([5, -6], '', []), 'opt')
        # not enough identical particles
        self.assertRaises(ValueError, chose, self.momenta, self.pdg, ([6, 6, 6], '', []), 'opt')
        self.assertRaises(ValueError, chose, self.momenta, self.pdg, ([6], 'pt', [2]), 'opt')
        # the same particle selected twice
        self.assertRaises(ValueError, chose, self.momenta, self.pdg, ([6, 6], 'pt', [0, 0]), 'opt')
        # number of ranks different from the number of particles
        self.assertRaises(ValueError, chose, self.momenta, self.pdg, ([6, 6], 'pt', [0]), 'opt')
        # unknown observable
        self.assertRaises(ValueError, chose, self.momenta, self.pdg, ([6], 'not_an_observable', []), 'opt')

    def test_density_boost_original_order(self):
        """the positions used for the boost refer to orig_order, not to the event which contains the intermediate top"""
        event = lhe_parser.Event("""<event>
 6      1 +4.2873600e+02 2.58116800e+02 7.54677100e-03 1.10829800e-01
       21 -1    0    0  501  502 +0.0000000000e+00 +0.0000000000e+00 +1.0845458909e+02 1.0845458909e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
       21 -1    0    0  502  503 -0.0000000000e+00 -0.0000000000e+00 -6.4844659178e+02 6.4844659178e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
        6  2    1    2  501    0 -1.0996883743e+01 +1.9124424233e+02 -1.8275210837e+02 3.1616925127e+02 1.7282757494e+02 0.0000e+00 0.0000e+00
        5  1    3    3  501    0 +4.3637392514e+01 +4.9281743782e+00 -1.3060449857e+01 4.6056207819e+01 4.7000000000e+00 0.0000e+00 -1.0000e+00
       24  1    3    3    0    0 -5.4634276257e+01 +1.8631606795e+02 -1.6969165851e+02 2.7011304345e+02 8.0419002446e+01 0.0000e+00 -1.0000e+00
       -6  1    1    2    0  503 +1.0996883743e+01 -1.9124424233e+02 -3.5723989432e+02 4.4073192960e+02 1.7300000000e+02 0.0000e+00 -1.0000e+00
</event>""")
        orig_order = ([21, 21], [5, 24, -6])
        pdg = orig_order[0] + orig_order[1]
        all_p = [event.get_momenta(orig_order)]
        user_input = ([5, -6], '', [])
        positions = chose(all_p[0], pdg, user_input, 'momenta_boost')
        self.assertEqual(positions, [2, 4])

        stub = types.SimpleNamespace(momenta_boost=user_input, keep_ordering=True)
        new_all_p = rwgt_interface.DensityInterface.method_boost_event(stub, event, all_p, orig_order, 0, positions)
        p_b, p_tbar = new_all_p[0][2], new_all_p[0][4]
        for k in range(1, 4): # b + t~ is at rest after the boost
            self.assertAlmostEqual(p_b[k] + p_tbar[k], 0., places=6)
