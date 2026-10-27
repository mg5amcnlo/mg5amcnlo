##############################################################################
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
"""Tests of the grid combination of the split survey/refine jobs
(madgraph/madevent/combine_grid.py, use_cut=-2 in dsample.f)"""
from __future__ import absolute_import
import unittest

import madgraph.madevent.combine_grid as combine_grid


class TestCombineGrid(unittest.TestCase):
    """new grid from the combined grid_information of the split jobs"""

    ng = 20

    def get_new_grid(self, weights, xmin, xmax):
        """uniform original grid on [0,1] with the given weight per bin and
        the range [xmin, xmax] of the points seen"""
        info = combine_grid.grid_information(mc_hel=0)
        info.ng = self.ng
        info.maxinvar = 1
        info.nonzero = 1000
        for i in range(self.ng):
            info.original_grid[(i, 0)] = (i + 1) / self.ng
            info.grid_base[(i, 0)] = weights[i]
            info.non_zero_grid[(i, 0)] = 1000 // self.ng if weights[i] else 0
        info.min_on_axis[0] = xmin
        info.max_on_axis[0] = xmax
        return info.get_new_grid_for_var(0)

    def check_grid(self, grid):
        self.assertEqual(len(grid), self.ng)
        self.assertGreater(grid[0], 0)
        for i in range(1, self.ng):
            self.assertGreater(grid[i], grid[i-1], grid)
        self.assertEqual(grid[-1], 1)

    def test_grid_interval(self):
        """the grid of dsample.f is on [xgmin, xgmax] = [0, 1]"""
        self.assertEqual((combine_grid.grid_information.start,
                          combine_grid.grid_information.stop), (0, 1))

    def test_points_in_first_and_last_bin(self):
        """points down to the first bin: the new grid starts at 0, not -1
        (x < 0 gave a Bjorken x above 1 for g g > h)"""
        weights = [10. / (i + 1) for i in range(self.ng)]
        grid = self.get_new_grid(weights, 1e-4, 0.999)
        self.check_grid(grid)
        # the weight is at small x
        self.assertLess(grid[self.ng // 2], 0.5)

    def test_flat(self):
        """a flat function keeps a uniform grid"""
        grid = self.get_new_grid([1.] * self.ng, 1e-4, 0.999)
        self.check_grid(grid)
        for i, x in enumerate(grid):
            self.assertAlmostEqual(x, (i + 1) / self.ng)

    def test_endpoints_not_reached(self):
        """points only in [0.3, 0.6]: the first/last grid points are put
        just outside of that range (as in dsample.f), and the last one
        stays at 1 such that the region above is still sampled"""
        weights = [1. if 6 <= i <= 11 else 0. for i in range(self.ng)]
        grid = self.get_new_grid(weights, 0.3, 0.6)
        self.check_grid(grid)
        width = 1. / self.ng
        self.assertAlmostEqual(grid[0], 0.3 - width / 5)
        self.assertAlmostEqual(grid[-2], 0.6 + width / 5)
        for x in grid[1:-2]:
            self.assertGreaterEqual(x, grid[0])
            self.assertLessEqual(x, grid[-2])


if __name__ == '__main__':
    unittest.main()
