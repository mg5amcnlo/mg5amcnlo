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
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Tests of the submission of the split refine jobs
(gen_ximprove.gensym.create_resubmit_one_iter, used by gen_ximprove_share)"""
from __future__ import absolute_import
import os
import shutil
import tempfile
import types
import unittest

import madgraph.madevent.gen_ximprove as gen_ximprove


class TestCreateResubmitOneIter(unittest.TestCase):
    """The refine jobs start from G/ftn26 (refine_splitted.sh copies it into
    each job's ftn25), so that is the grid that has to be there."""

    def setUp(self):
        self.me_dir = tempfile.mkdtemp()
        self.pdir = os.path.join(self.me_dir, 'SubProcesses', 'P1_gg_hg')
        self.gdir = os.path.join(self.pdir, 'G1')
        os.makedirs(self.gdir)
        self.launched = []

        def launch_job(exe, argument=None, cwd=None, packet_member=None):
            self.launched.append((os.path.basename(exe), argument, cwd))

        cmd = types.SimpleNamespace(run_card={'nhel': 0}, opts={'accuracy': 0.02},
                                    launch_job=launch_job)
        self.x_improve = types.SimpleNamespace(
            me_dir=self.me_dir, cmd=cmd, splitted_Pdir={}, lastoffset={},
            disable_multichannel=False, combine_iteration=lambda *args: None)
        self.x_improve.write_parameter_file = types.MethodType(
            gen_ximprove.gensym.write_parameter_file, self.x_improve)

    def tearDown(self):
        shutil.rmtree(self.me_dir)

    def resubmit(self, nb_job=3):
        gen_ximprove.gensym.create_resubmit_one_iter(
            self.x_improve, 'P1_gg_hg', '1', 1000, nb_job, step=0)

    def test_grid_left_by_an_unsplit_survey(self):
        """survey_splitting = 1: the survey ran in G1 itself and left the grid
        madevent wrote, ftn26, and no ftn25."""
        open(os.path.join(self.gdir, 'ftn26'), 'w').close()
        self.resubmit(nb_job=3)
        self.assertEqual([job[0] for job in self.launched],
                         ['refine_splitted.sh'] * 3)
        self.assertEqual([job[1] for job in self.launched],
                         [['G1_1', 'G1', 1], ['G1_2', 'G1', 2], ['G1_3', 'G1', 3]])
        with open(os.path.join(self.gdir, 'input_app.txt')) as fsock:
            self.assertIn('-2       !Grid Adjustment', fsock.read())

    def test_no_grid_is_refused(self):
        with self.assertRaises(AssertionError):
            self.resubmit()
        self.assertEqual(self.launched, [])


if __name__ == '__main__':
    unittest.main()
