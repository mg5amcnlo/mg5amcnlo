################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
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

"""Outputting the same [virt=] process twice from one interface.

'set loop_color_flows' (like 'set loop_optimized_output') drops
_curr_matrix_elements, so the next 'output' rebuilds the LoopHelasMatrixElement
from self._curr_amps.  ML5export used to replace _curr_amps by the
'base_amplitude' of each matrix element after the first output.  For a loop
matrix element that amplitude is rebuilt from the helas diagrams by
LoopHelasMatrixElement.get_base_amplitude: it has no loop tags and no
structure repository, so the second output died with an IndexError on
tag[0][0] in process_loop_diagram.
"""

from __future__ import absolute_import

import filecmp
import glob
import os
import shutil
import sys
import tempfile

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.interface.master_interface as MGCmd
import madgraph.loop.loop_diagram_generation as loop_diagram_generation

from madgraph import MG5DIR

pjoin = os.path.join

PROCESS = 'u u~ > d d~ [virt=QCD]'
# The files carrying the loop diagrams themselves
COMPARED_FILES = ['coef_construction_1.f', 'loop_CT_calls_1.f',
                  'loop_matrix.f']


def get_interface():
    """A fresh MasterCmd with PROCESS generated."""

    interface = MGCmd.MasterCmd()
    interface.no_notification()
    for line in ['import model loop_sm', 'generate %s' % PROCESS]:
        interface.exec_cmd(line, printcmd=False, precmd=True)
    return interface


class TestLoopOutputTwice(unittest.TestCase):
    """A second output after 'set loop_color_flows' must match a fresh one."""

    def setUp(self):
        # 'output' chdir's into the directory it writes
        os.chdir(MG5DIR)
        self.tmpdir = tempfile.mkdtemp(prefix='loop_output_twice')

    def tearDown(self):
        os.chdir(MG5DIR)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def output(self, interface, name):
        out_dir = pjoin(self.tmpdir, name)
        interface.exec_cmd('output standalone_fortran %s -f' % out_dir,
                           printcmd=False, precmd=True)
        os.chdir(MG5DIR)
        proc_dirs = glob.glob(pjoin(out_dir, 'SubProcesses', 'P*'))
        self.assertEqual(len(proc_dirs), 1)
        return proc_dirs[0]

    def test_output_again_after_set_loop_color_flows(self):
        """The second output used to crash; it must equal a fresh output."""

        interface = get_interface()
        interface.exec_cmd('set loop_color_flows True', printcmd=False,
                           precmd=True)
        self.output(interface, 'A')

        # the amplitudes the next output starts from are still the generated
        # ones, with their loop tags
        for amp in interface._curr_amps:
            self.assertIsInstance(amp, loop_diagram_generation.LoopAmplitude)
            for diag in amp.get('loop_diagrams'):
                self.assertTrue(diag.get('tag'))

        interface.exec_cmd('set loop_color_flows False', printcmd=False,
                           precmd=True)
        second = self.output(interface, 'B')
        fresh = self.output(get_interface(), 'C')

        self.assertEqual(os.path.basename(second), os.path.basename(fresh))
        for name in COMPARED_FILES:
            self.assertTrue(filecmp.cmp(pjoin(second, name),
                                        pjoin(fresh, name), shallow=False),
                            '%s differs from a fresh output' % name)


if __name__ == '__main__':
    unittest.main()
