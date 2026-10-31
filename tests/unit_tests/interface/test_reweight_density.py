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
import tests.unit_tests as unittest

import madgraph.interface.reweight_interface as rwgt_interface

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


class TestDensityOrderHelicities(unittest.TestCase):
    """Test the conversion of order_helicities into the helicity combinations passed to Fortran (ALLOW_HEL)"""

    class DensityStub(object):
        """The methods of DensityInterface used by order_helicities, without a full interface
        (whose __del__ would run do_quit)"""
        helicity_basis = rwgt_interface.DensityInterface.helicity_basis
        InvalidCmd = rwgt_interface.DensityInterface.InvalidCmd
        do_change_order_helicities = rwgt_interface.DensityInterface.do_change_order_helicities
        set_allowed_helicities = rwgt_interface.DensityInterface.set_allowed_helicities

    def density_interface(self, spins=None):
        """A DensityInterface with only what order_helicities needs, as set by particle_in_density_matrix"""
        cmd = self.DensityStub()
        cmd.order_helicities = None
        cmd.allowed_helicities = [0]
        cmd.spins = None
        if spins is not None:
            self.set_spins(cmd, spins)
        return cmd

    @staticmethod
    def set_spins(cmd, spins):
        cmd.spins = spins
        cmd.number_changing_helicities = len(spins)
        cmd.number_combinations = 1
        for spin in spins:
            cmd.number_combinations *= spin
        cmd.set_allowed_helicities()

    def order(self, spins, line):
        cmd = self.density_interface(spins)
        cmd.do_change_order_helicities(line.split())
        return cmd.allowed_helicities

    def test_order_helicities_default(self):
        """the first particle is the first index, each basis is (+1, -1) or (+1, 0, -1)"""
        self.assertEqual(self.density_interface([2, 2]).allowed_helicities, [1, 1, 1, -1, -1, 1, -1, -1])
        self.assertEqual(self.density_interface([2, 3]).allowed_helicities, [1, 1, 1, 0, 1, -1, -1, 1, -1, 0, -1, -1])
        for line in ['[0]', 'None']:
            self.assertEqual(self.order([2, 2], line), [1, 1, 1, -1, -1, 1, -1, -1])

    def test_order_helicities_inputs(self):
        # all the combinations, with or without spaces
        self.assertEqual(self.order([2, 2], '[-1, 1, -1, -1, 1, 1, 1, -1]'), [-1, 1, -1, -1, 1, 1, 1, -1])
        self.assertEqual(self.order([2, 2], '[-1,1,-1,-1,1,1,1,-1]'), [-1, 1, -1, -1, 1, 1, 1, -1])
        # an order which is not a tensor product of the bases of each particle is still a basis
        self.assertEqual(self.order([2, 2], '[1, 1, -1, -1, 1, -1, -1, 1]'), [1, 1, -1, -1, 1, -1, -1, 1])
        # the basis of each particle, for any number of particles
        self.assertEqual(self.order([2, 2], '[-1, 1] [1, -1]'), [-1, 1, -1, -1, 1, 1, 1, -1])
        self.assertEqual(self.order([2, 3], '[-1, 1] [1, 0, -1]'), [-1, 1, -1, 0, -1, -1, 1, 1, 1, 0, 1, -1])
        self.assertEqual(self.order([2, 2, 2], '[-1, 1] [1, -1] [1, -1]')[:6], [-1, 1, 1, -1, 1, -1])
        self.assertEqual(self.order([3], '[0, 1, -1]'), [0, 1, -1])

    def test_order_helicities_read_before_particles(self):
        """the order is kept until the particles in the density matrix are known, and follows them if they change"""
        cmd = self.density_interface()
        cmd.do_change_order_helicities(['[-1,', '1]', '[1,', '-1]'])
        self.set_spins(cmd, [2, 2])
        self.assertEqual(cmd.allowed_helicities, [-1, 1, -1, -1, 1, 1, 1, -1])
        cmd.do_change_order_helicities(['[0]'])
        self.set_spins(cmd, [2, 3])
        self.assertEqual(cmd.allowed_helicities, [1, 1, 1, 0, 1, -1, -1, 1, -1, 0, -1, -1])

    def test_order_helicities_wrong_inputs(self):
        InvalidCmd = rwgt_interface.DensityInterface.InvalidCmd
        # wrong length
        self.assertRaises(InvalidCmd, self.order, [2, 2], '[1, -1, 1]')
        # helicity 0 for a particle with 2 spin degrees of freedom (fermion or massless vector boson)
        self.assertRaises(InvalidCmd, self.order, [2, 2], '[1, -1] [0, 1]')
        # duplicated combination
        self.assertRaises(InvalidCmd, self.order, [2, 2], '[1, 1, 1, 1, -1, 1, -1, -1]')
        # not integers
        self.assertRaises(InvalidCmd, self.order, [2, 2], '[a, b]')


class TestDensityAverageFile(unittest.TestCase):
    """Test the file with the average density matrix, in single core and multicore mode"""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def read_average(self, name):
        import os
        lines = open(os.path.join(self.tmpdir, name)).read().splitlines()
        return lines[0], [[complex(x.strip(' ()')) for x in line.strip('\t[]').split(',')] for line in lines[1:]]

    def test_average_density_file_name(self):
        """the name of the file does not depend on the compression of the event file"""
        import os
        density = rwgt_interface.DensityInterface
        for event_file in ['unweighted_events.lhe.gz', 'unweighted_events.lhe']:
            density.write_average_density(os.path.join(self.tmpdir, event_file), [2, 0, 2], 4., [1, -1])
            self.assertEqual(os.listdir(self.tmpdir), ['Average_density_matrix_unweighted_events.txt'])
            header, rho = self.read_average('Average_density_matrix_unweighted_events.txt')
            self.assertIn('helicity basis [1, -1]', header)
            self.assertEqual(rho, [[0.5, 0], [0, 0.5]])

    def test_average_density_multicore(self):
        """the average of a multicore run is the one of the full event file, not the one of a chunk"""
        import os
        density = rwgt_interface.DensityInterface
        event_file = os.path.join(self.tmpdir, 'unweighted_events.lhe.gz')
        chunks = ['%s_%s.lhe' % (event_file, i) for i in range(3)]
        # weighted sums of normalised matrices and sums of weights of each chunk
        density.write_density_sum(chunks[0], [3, 1j, 0], 3., [1, -1])
        density.write_density_sum(chunks[1], [0, 0, 1], 1., [1, -1])
        # chunks[2] had no event
        density.combine_average_density(chunks, event_file)

        self.assertEqual(os.listdir(self.tmpdir), ['Average_density_matrix_unweighted_events.txt']) #the intermediate files are removed
        _, rho = self.read_average('Average_density_matrix_unweighted_events.txt')
        self.assertEqual(rho, [[0.75, 0.25j], [-0.25j, 0.25]])
