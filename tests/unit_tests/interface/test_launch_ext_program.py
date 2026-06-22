from __future__ import absolute_import

import os
import tempfile
import unittest

from madgraph.interface.launch_ext_program import SALauncher


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


if __name__ == '__main__':
    unittest.main()
