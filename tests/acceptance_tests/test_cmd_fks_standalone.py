################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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
"""Acceptance tests for the FKS Born building-block standalone output
('output standalone --fks').

Both tests compile and run Fortran (the lightweight 'check_fks' driver), so
they live in the acceptance tier rather than unit_tests (whose 1s budget would
silently skip them). The first is a focused regression on the hardcoded Born
building-block values at the driver's fixed RAMBO seed; the second checks that
those values agree with the ones obtained from a plain aMC@NLO output of the
same process.
"""
from __future__ import division
from __future__ import absolute_import
import glob
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import madgraph.interface.master_interface as MGCmd
import madgraph.various.misc as misc

pjoin = os.path.join


# Reference Born building blocks for each process, evaluated at the phase-space
# point built by the driver's fixed RAMBO seed (RMARIN(1802,9373) in
# check_sa_fks.f). If the driver's phase-space construction or the model
# defaults ever change, regenerate these by running 'check_fks' once and pasting
# the printed values.
#
# The list is the single source of truth for both tests below; adding a process
# here grows the value-regression and the standalone-vs-aMC@NLO cross-check at
# once. Three representative cases are covered:
#   * g g > t t~   : 2 -> 2, gluon legs (non-zero BORNTILDE, full color-link web)
#   * e+ e- > j j  : colour-singlet initial state, a single q-qbar colour link
#   * z > j j      : a 1 -> 2 decay (NINCOMING=1) with a mixed QCD/QED Born,
#                    which exercises the decay kinematics and the
#                    perturbed-split-order selection in the driver
PROCESSES = [
    {'id': 'gg_ttx',
     'process': 'g g > t t~ [QCD]',
     'model': 'loop_sm',
     'born': 0.56845354707929208,
     'borntilde': 0.022972741277183555,
     'bij': {(1, 1): 1.2643814370872453,
             (1, 2): -1.5674461029801903,
             (1, 3): 0.046354897525540478,
             (1, 4): -1.0076716687198406,
             (2, 2): 1.2643814370872453,
             (2, 3): -1.0076716687198406,
             (2, 4): 0.046354897525540478,
             (3, 3): 0.56194730537210891,
             (3, 4): -0.16257783954991759,
             (4, 4): 0.56194730537210891}},
    {'id': 'ee_jj',
     'process': 'e+ e- > j j [QCD]',
     'model': 'loop_sm',
     'born': 0.016492937363039667,
     'borntilde': 0.0,
     'bij': {(3, 3): 0.01630416725597162,
             (3, 4): -0.032608334511943241,
             (4, 4): 0.01630416725597162}},
    {'id': 'z_jj',
     'process': 'z > j j QED=2 [QCD]',
     'model': 'loop_sm',
     'born': 1705.2784047399880,
     'borntilde': 0.0,
     'bij': {(2, 2): 1685.7606208572352,
             (2, 3): -3371.5212417144703,
             (3, 3): 1685.7606208572352}},
    {'id': 'uux_ttx',
     'process': 'u u~ > t t~ [QCD]',
     'model': 'loop_sm',
     'born': 0.59212443086238242,
     'borntilde': 0.0,
     'bij': {(1, 1): 0.5853472637786112,
             (1, 2): 0.14633681594465273,
             (1, 3): -1.0243577116125695,
             (1, 4): -0.29267363188930562,
             (2, 2): 0.5853472637786112,
             (2, 3): -0.29267363188930562,
             (2, 4): -1.0243577116125695,
             (3, 3): 0.58534726377861090,
             (3, 4): 0.14633681594465273,
             (4, 4): 0.58534726377861090}},
    # an EW ([QED]) correction: the soft particle is a photon, so the
    # building blocks are charge-linked Borns (need_charge_links, itype=1).
    # This exercises the whole [QED] path: born_charges.inc, the charge
    # branch of sborn_sf, and the find_color_links(pert='QED') topology
    # (every charged pair, incl. the massive-W self-links). The W is
    # colourless, so its charge links are absent from the colour basis -
    # the value-set here would be wrong if they were ever dropped.
    # The (1,1)/(2,2) entries are the massless-u charge diagonals the driver
    # rebuilds from charge conservation (B_ii = -1/2 sum_{j!=i} B_ij), the
    # same identity used for the massless gluon/quark colour diagonals above.
    {'id': 'uux_wpwm',
     'process': 'u u~ > w+ w- [QED]',
     'model': 'loop_sm',
     'born': 0.0053974535211713691,
     'borntilde': 0.0,
     'bij': {(1, 1): 0.00011690478264832319,
             (1, 2): -0.00023380956529664638,
             (1, 3): -0.0003507143479449694,
             (1, 4): 0.0003507143479449694,
             (2, 2): 0.00011690478264832319,
             (2, 3): 0.0003507143479449694,
             (2, 4): -0.0003507143479449694,
             (3, 3): 0.0002630357609587269,
             (3, 4): -0.0005260715219174538,
             (4, 4): 0.0002630357609587269}},
]


class TestFKSStandalone(unittest.TestCase):
    """'output standalone --fks' end-to-end (generate + launch + check_fks)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='fkssa')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _new_cmd():
        cmd = MGCmd.MasterCmd()
        cmd.no_notification()
        cmd.run_cmd('set automatic_html_opening False --no_save')
        return cmd

    @staticmethod
    def _run(cmd, line):
        # precmd/postcmd=True so the command history is populated (the FKS
        # finalize reads history.get('generate') to parse the perturbation)
        cmd.exec_cmd(line, errorhandling=False, printcmd=False,
                     precmd=True, postcmd=True)

    def _output_fks_sa(self, process, model, path):
        """generate + 'output standalone --fks' + launch (builds & runs
        check_fks)."""
        cmd = self._new_cmd()
        self._run(cmd, 'import model %s' % model)
        self._run(cmd, 'generate %s' % process)
        self._run(cmd, 'output standalone --fks %s -f' % path)
        self._run(cmd, 'launch %s -f' % path)

    def _output_amcatnlo(self, process, model, path):
        """generate + plain (full) aMC@NLO output of the same process."""
        cmd = self._new_cmd()
        self._run(cmd, 'import model %s' % model)
        self._run(cmd, 'generate %s' % process)
        self._run(cmd, 'output %s -f' % path)

    def _born_dir(self, path):
        """the P* subprocess directory holding the standalone driver.

        A multi-flavour process produces one born dir per flavour family
        (e.g. P0_epem_ddx and P0_epem_uux); glob order is filesystem- and
        platform-dependent, so we sort and always take the first one. The
        reference values in PROCESSES are tied to that same deterministic
        choice (the alphabetically-first born dir)."""
        found = sorted(glob.glob(
            pjoin(path, 'SubProcesses', 'P*', 'check_sa_fks.f')))
        self.assertTrue(found, 'no FKS standalone born dir in %s' % path)
        return os.path.dirname(found[0])

    def _run_check_fks(self, born_dir):
        """run an already-built check_fks and return its stdout."""
        exe = pjoin(born_dir, 'check_fks')
        self.assertTrue(os.path.isfile(exe),
                        'check_fks not built in %s' % born_dir)
        out = subprocess.check_output([exe], cwd=born_dir,
                                      stderr=subprocess.STDOUT)
        return out.decode(errors='replace')

    @staticmethod
    def parse_check_fks(output):
        """extract (born, borntilde, {(m,n): b_ij}) from check_fks stdout."""
        born = borntilde = None
        bij = {}
        # 'BORN  =  ...' but NOT the 'BORN: keeping split order' debug lines
        m = re.search(r'BORN\s*=\s*([-+\d.EeDd]+)', output)
        if m:
            born = float(m.group(1).replace('D', 'E').replace('d', 'e'))
        m = re.search(r'BORNTILDE\s*=\s*([-+\d.EeDd]+)', output)
        if m:
            borntilde = float(m.group(1).replace('D', 'E').replace('d', 'e'))
        for line in output.splitlines():
            toks = line.split()
            if len(toks) == 4 and toks[0] == 'B_ij':
                key = (int(toks[1]), int(toks[2]))
                bij[key] = float(toks[3].replace('D', 'E').replace('d', 'e'))
        return born, borntilde, bij

    def assertClose(self, value, ref, rel=1e-6, msg=''):
        self.assertIsNotNone(value, 'missing value (%s)' % msg)
        self.assertAlmostEqual(value, ref, delta=abs(ref) * rel + 1e-9,
                               msg='%s: %r vs %r' % (msg, value, ref))

    # ------------------------------------------------------------------ #
    # tests
    # ------------------------------------------------------------------ #
    def _workdir(self, spec):
        """a fresh per-process working directory under the test tmpdir."""
        path = pjoin(self.tmpdir, spec['id'])
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def _check_born_values(self, spec):
        """check_fks reproduces the hardcoded Born building blocks for one
        process at the driver's fixed RAMBO seed."""
        path = pjoin(self._workdir(spec), 'fks_sa')
        self._output_fks_sa(spec['process'], spec['model'], path)
        born, borntilde, bij = self.parse_check_fks(
            self._run_check_fks(self._born_dir(path)))

        self.assertClose(born, spec['born'], msg='BORN')
        self.assertClose(borntilde, spec['borntilde'], msg='BORNTILDE')
        self.assertEqual(set(bij), set(spec['bij']),
                         'unexpected set of color links')
        for key, ref in spec['bij'].items():
            self.assertClose(bij[key], ref, msg='B_ij%s' % (key,))

    def test_fks_standalone_born_values(self):
        """check_fks reproduces the hardcoded Born building blocks at the
        driver's fixed RAMBO seed, for every process in PROCESSES."""
        for spec in PROCESSES:
            with self.subTest(process=spec['process']):
                self._check_born_values(spec)

    def _check_vs_amcatnlo(self, spec):
        """the standalone Born building blocks match the ones computed by the
        code of a plain aMC@NLO output of the same process.

        The '--fks' output is a full FKS directory (full-gen), so born.f /
        sborn_sf.f / b_sf_*.f are byte-for-byte the production ones. We bring
        the standalone driver into the plain aMC@NLO directory (whose P-dir
        makefile already carries the 'check_fks' target), build and run it
        there, and require identical values."""
        workdir = self._workdir(spec)
        # 1) standalone --fks (driver already built by launch)
        path_sa = pjoin(workdir, 'fks_sa')
        self._output_fks_sa(spec['process'], spec['model'], path_sa)
        born_dir_sa = self._born_dir(path_sa)
        val_sa = self.parse_check_fks(self._run_check_fks(born_dir_sa))

        # 2) plain aMC@NLO output of the same process
        path_nlo = pjoin(workdir, 'nlo')
        self._output_amcatnlo(spec['process'], spec['model'], path_nlo)

        # 3) drop the standalone driver + data into the full dir and build it
        pname = os.path.basename(born_dir_sa)
        born_dir_nlo = pjoin(path_nlo, 'SubProcesses', pname)
        for f in ('check_sa_fks.f', 'born_pmass.inc', 'born_links.dat',
                  'born_charges.inc'):
            shutil.copy(pjoin(born_dir_sa, f), pjoin(born_dir_nlo, f))
        # the plain output lacks the makefile-include stubs that the SA
        # finalize writes; the check_fks target does not use their content
        for stub in ('analyse_opts', 'pythia8_opts'):
            sp = pjoin(path_nlo, 'SubProcesses', stub)
            if not os.path.isfile(sp):
                open(sp, 'w').write('')
        misc.compile(cwd=pjoin(path_nlo, 'Source'))
        misc.compile(['check_fks'], cwd=born_dir_nlo)
        val_nlo = self.parse_check_fks(self._run_check_fks(born_dir_nlo))

        # 4) identical Born building blocks (same code, same seed)
        self.assertClose(val_sa[0], val_nlo[0], rel=1e-9, msg='BORN')
        self.assertClose(val_sa[1], val_nlo[1], rel=1e-9, msg='BORNTILDE')
        self.assertEqual(set(val_sa[2]), set(val_nlo[2]),
                         'color link sets differ between outputs')
        for key in val_sa[2]:
            self.assertClose(val_sa[2][key], val_nlo[2][key], rel=1e-9,
                             msg='B_ij%s' % (key,))

    def test_fks_standalone_vs_amcatnlo(self):
        """the standalone Born building blocks match a plain aMC@NLO output of
        the same process, for every process in PROCESSES."""
        for spec in PROCESSES:
            with self.subTest(process=spec['process']):
                self._check_vs_amcatnlo(spec)


if __name__ == '__main__':
    unittest.main()
