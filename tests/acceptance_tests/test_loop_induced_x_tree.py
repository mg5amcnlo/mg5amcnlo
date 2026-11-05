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
"""Numerical checks of the interference processes. First the loop-induced
x tree interference

    generate g g > h > t t~ [LIxtree=QCD]
    output standalone_fortran

at one fixed phase-space point (sqrt(s) = 1 TeV, default loop_sm parameters).
The standalone MadLoop check prints the three products it can form:

  INTERF=1  2 Re(A_loop A_tree^*)   the selected result
  INTERF=0  |A_loop|^2              must equal 'g g > h > t t~ [sqrvirt=QCD]'
  INTERF=2  |A_tree|^2              must equal the tree-level matrix element
                                    of 'g g > t t~' (with the helicity filter
                                    off: it is set on the interference)

The references were obtained independently: the tree-level matrix element
from 'output standalone_fortran' of 'g g > t t~', the loop squared from
'[sqrvirt=QCD]', and the interference from the original recipe of the wiki
page LoopInducedTimesTree (tree vertices copied as UVtree counterterms with a
'BKGQCD' order), all at this same point.
"""

from __future__ import absolute_import

import os
import re
import shutil
import subprocess
import tempfile

import tests.unit_tests as unittest

import madgraph.interface.master_interface as MGCmd

pjoin = os.path.join

PS_POINT = """5.0000000000000000e+02  0.0000000000000000e+00  0.0000000000000000e+00  5.0000000000000000e+02
5.0000000000000000e+02  0.0000000000000000e+00  0.0000000000000000e+00 -5.0000000000000000e+02
4.9999999999999989e+02  1.0407299191319960e+02  4.1735559881462342e+02 -1.8722744588420289e+02
4.9999999999999989e+02 -1.0407299191319960e+02 -4.1735559881462342e+02  1.8722744588420281e+02
"""

REFERENCES = {1: -2.3256785052943376e-04,   # wiki recipe
              0: 8.1044796923476000e-05,    # [sqrvirt=QCD]
              2: 5.9262610090352252e-01}    # tree-level g g > t t~


class LoopInducedXTreeStandaloneTest(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='lixtree_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_loop_induced_x_tree_standalone(self):
        out_dir = pjoin(self.tmpdir, 'gg_h_ttx')
        interface = MGCmd.MasterCmd()
        for command in ['import model loop_sm',
                        'generate g g > h > t t~ [LIxtree=QCD]',
                        'output standalone_fortran %s -f' % out_dir]:
            interface.exec_cmd(command, printcmd=False, precmd=True,
                               errorhandling=False)

        proc_dir = pjoin(out_dir, 'SubProcesses', 'P0_gg_h_ttx')
        # read the point above instead of the RAMBO one
        check_sa = pjoin(proc_dir, 'check_sa.f')
        text = open(check_sa).read()
        self.assertIn('PARAMETER (READPS = .FALSE.)', text)
        open(check_sa, 'w').write(text.replace('PARAMETER (READPS = .FALSE.)',
                                               'PARAMETER (READPS = .TRUE.)'))
        open(pjoin(proc_dir, 'PS.input'), 'w').write(PS_POINT)
        # no helicity filter, so that |A_tree|^2 is complete as well
        card = pjoin(proc_dir, 'MadLoop5_resources', 'MadLoopParams.dat')
        lines = open(card).read().split('\n')
        index = lines.index('#HelicityFilterLevel')
        lines[index + 1] = '0'
        open(card, 'w').write('\n'.join(lines))

        subprocess.check_call(['make', 'check'], cwd=proc_dir,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        output = subprocess.run(['./check'], cwd=proc_dir, timeout=600,
                                capture_output=True, text=True).stdout

        results = {}
        for match in re.finditer(r'Loop ME for orders \(INTERF=(\d) QCD=4\) :\s*'
                                 r'> accuracy\s*=\s*\S+\s*> finite\s*=\s*(\S+)',
                                 output):
            results[int(match.group(1))] = float(match.group(2).replace('D', 'E'))
        self.assertEqual(sorted(results), [0, 1, 2], output[-2000:])
        for interf, reference in REFERENCES.items():
            self.assertAlmostEqual(results[interf] / reference, 1., places=10,
                                   msg='INTERF=%d: %r instead of %r' %
                                       (interf, results[interf], reference))

        # the result returned is the interference alone
        total = re.search(r'Matrix element finite\s*=\s*(\S+)', output)
        self.assertAlmostEqual(float(total.group(1)) / REFERENCES[1], 1.,
                               places=10)



class TreeXTreeStandaloneTest(unittest.TestCase):
    """'u u~ > z > e+ e- [treextree] u u~ > a > e+ e-' at the phase-space
    point of the standalone check, against |A_Z + A_a|^2 - |A_Z|^2 - |A_a|^2
    from three ordinary standalone outputs evaluated at the same point."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='treextree_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    output_format = 'standalone_fortran'

    def evaluate(self, name, process):
        """(momenta, matrix element) printed by the standalone check."""
        out_dir = pjoin(self.tmpdir, name)
        interface = MGCmd.MasterCmd()
        for command in ['import model sm', 'generate %s' % process,
                        'output %s %s -f' % (self.output_format, out_dir)]:
            interface.exec_cmd(command, printcmd=False, precmd=True,
                               errorhandling=False)
        sub_dir = pjoin(out_dir, 'SubProcesses')
        proc_dir = [pjoin(sub_dir, d) for d in os.listdir(sub_dir)
                    if d.startswith('P')][0]
        if self.output_format == 'standalone_fortran':
            build, check = ['make', 'check'], ['./check']
        else:
            # madmatrix: the default build mixes precisions (float color
            # sum), which caps the comparison at ~1e-7; build it in double
            build, check = ['make', 'FPTYPE=d', '-j4'], ['./check_sa.exe']
        subprocess.check_call(build, cwd=proc_dir,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        output = subprocess.run(check, cwd=proc_dir, timeout=300,
                                capture_output=True, text=True).stdout
        momenta = re.findall(r'^\s*[1-4]\s+(\S+\s+\S+\s+\S+\s+\S+)', output,
                             re.MULTILINE)
        value = re.search(r'Matrix element =\s*(\S+)', output).group(1)
        return momenta, float(value.replace('D', 'E'))

    def test_tree_x_tree_standalone(self):
        point, interference = self.evaluate('interf',
                        'u u~ > z > e+ e- [treextree] u u~ > a > e+ e-')
        self.assertEqual(len(point), 4)
        references = {}
        for name, process in [('za', 'u u~ > e+ e- / h'),
                              ('z', 'u u~ > z > e+ e-'),
                              ('a', 'u u~ > a > e+ e-')]:
            momenta, references[name] = self.evaluate(name, process)
            self.assertEqual(momenta, point)
        expected = references['za'] - references['z'] - references['a']
        self.assertTrue(abs(expected) > 1e-3 * references['za'])
        self.assertAlmostEqual(interference / expected, 1., places=10)



class TreeXTreeMadmatrixStandaloneTest(TreeXTreeStandaloneTest):
    """The same comparison with the C++ matrix element of madmatrix ('output
    standalone', the code mg7 runs), built in double precision."""

    output_format = 'standalone'


if __name__ == '__main__':
    unittest.main()
