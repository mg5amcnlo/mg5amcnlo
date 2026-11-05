from __future__ import absolute_import

import os
import tempfile
import unittest

from madgraph.interface.launch_ext_program import MadLoopLauncher, SALauncher


class TestSALauncherTimings(unittest.TestCase):

    def test_read_feynman_diagram_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'ngraphs.inc'), 'w') as stream:
                stream.write('       integer    n_max_cg\n')
                stream.write('parameter (n_max_cg=42)\n')

            launcher = SALauncher(None, tmpdir)
            self.assertEqual(launcher._read_feynman_diagram_count(tmpdir), 42)

    def test_read_feynman_diagram_count_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            launcher = SALauncher(None, tmpdir)
            self.assertIsNone(launcher._read_feynman_diagram_count(tmpdir))


class TestMadLoopLauncherDensity(unittest.TestCase):
    """The density mode must be detected from the output line of the proc
    card even when it is wrapped inside '--density', as older versions of
    the card writer did."""

    def write_proc_card(self, tmpdir, *lines):
        os.mkdir(os.path.join(tmpdir, 'Cards'))
        with open(os.path.join(tmpdir, 'Cards', 'proc_card_mg5.dat'), 'w') as stream:
            stream.write('#' + '*' * 60 + '\n')
            stream.write('\n'.join(lines) + '\n')

    def test_density_flag_split_by_the_wrap(self):
        # as older versions wrote it for a long MG5DIR/TEST_AMC/MGProcess: no
        # line of the file contains '--density'
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_proc_card(tmpdir,
                'import model loop_sm',
                'generate g g > z* g [sqrvirt=QCD]',
                'output standalone_fortran /Users/someone/worktrees/g/TEST_AMC/MGProcess -\\',
                '-density=3,4 -f')
            self.assertTrue(MadLoopLauncher(None, tmpdir).uses_density())

    def test_density_flag_on_one_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_proc_card(tmpdir,
                'generate g g > h [sqrvirt=QCD]',
                'output standalone_fortran /tmp/test1 --density=1,2 -f')
            self.assertTrue(MadLoopLauncher(None, tmpdir).uses_density())

    def test_no_density(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_proc_card(tmpdir,
                'generate g g > e+ e- g / a [sqrvirt=QCD]',
                'output standalone_fortran /tmp/density-study/MGProcess -f')
            self.assertFalse(MadLoopLauncher(None, tmpdir).uses_density())


if __name__ == '__main__':
    unittest.main()
