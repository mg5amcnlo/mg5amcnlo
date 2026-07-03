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
"""Consistency of the unrolled-helicity standalone output (output standalone
--unrolhel=True) against the standard standalone output.

In --unrolhel mode the helicity loop is unrolled into a vector dimension using
the cartesian-product (ALOHA tag 'H') routines: external wavefunctions carry all
their helicity states at once, vertices/amplitudes are built as type(aloha_H)
cartesian products, and the squared matrix element is summed over the NCOMB
helicity combinations after a per-diagram canonical reindexing.  For every phase
space point the summed/averaged |M|^2 must therefore agree with the standard
standalone to (numerical) round-off.

Each process below is generated twice (standard and --unrolhel), each output is
compiled (`make check`) and run (`./check`), and the printed 'Matrix element'
value(s) are compared.  The processes cover the features that historically broke:
combined (multi-Lorentz) routines, color, cross-topology helicity indexing,
massive (3-state) external vectors, identical particles, and flavor couplings.
"""

from __future__ import absolute_import

import os
import re
import shutil
import subprocess
import tempfile
import unittest
import logging
logger = logging.getLogger('madgraph.madevent')

import madgraph.interface.master_interface as cmd_interface
import madgraph.various.misc as misc

pjoin = os.path.join


def _sanitize_process_name(process):
    return re.sub(r'[^A-Za-z0-9]+', '_', process).strip('_').lower()


def unrolhel_consistency_test_factory(process, model='sm', tolerance=1e-9,
                                      unrol_opts=''):
    def test(self):
        self.check_process(process, model=model, tolerance=tolerance,
                           unrol_opts=unrol_opts)
    test.__name__ = 'test_%s' % _sanitize_process_name(process)
    test.__doc__ = 'Check --unrolhel and standard standalone |M|^2 agree for %s.' % process
    return test


class StandaloneUnrolhelConsistency(unittest.TestCase):

    debugging = getattr(unittest, 'debug', False)

    @classmethod
    def setUpClass(cls):
        # The whole test relies on a working Fortran compiler (make check).
        if not misc.which('gfortran') and not misc.which('g-fortran') \
                and not misc.which('f77'):
            raise unittest.SkipTest('no fortran compiler available')

    def setUp(self):
        self.cmd = cmd_interface.MasterCmd()
        self.cmd.no_notification()
        prefix = 'amc_unrolhel_debug_' if self.debugging else 'amc_unrolhel_'
        self.tmpdir = tempfile.mkdtemp(prefix=prefix)
        self.std_dir = pjoin(self.tmpdir, 'Standard')
        self.unrol_dir = pjoin(self.tmpdir, 'Unrolhel')

    def tearDown(self):
        if not self.debugging and os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def do(self, line):
        self.cmd.exec_cmd(line)

    def check_process(self, process, model='sm', tolerance=1e-9, unrol_opts=''):
        self.do('set automatic_html_opening False')
        self.do('set group_subprocesses False')
        self.do('import model %s' % model)
        self.do('generate %s' % process)
        self.do('output standalone %s -f' % self.std_dir)
        self.do('output standalone %s --unrolhel=True %s -f'
                % (self.unrol_dir, unrol_opts))

        std_subdirs = self._get_subprocess_dirs(pjoin(self.std_dir, 'SubProcesses'))
        unrol_subdirs = self._get_subprocess_dirs(pjoin(self.unrol_dir, 'SubProcesses'))
        self.assertEqual([os.path.basename(d) for d in std_subdirs],
                         [os.path.basename(d) for d in unrol_subdirs],
                         'Different subprocess structure for %s' % process)

        for std_sub, unrol_sub in zip(std_subdirs, unrol_subdirs):
            std_me = self._run_standalone(std_sub)
            unrol_me = self._run_standalone(unrol_sub)
            self.assertEqual(
                len(std_me), len(unrol_me),
                'Different number of matrix elements for %s (%s): std=%s unrolhel=%s'
                % (process, os.path.basename(std_sub), len(std_me), len(unrol_me)))
            self.assertTrue(std_me,
                            'No matrix element printed for %s' % process)
            for i, (s, u) in enumerate(zip(std_me, unrol_me)):
                scale = max(abs(s), abs(u), 1e-99)
                self.assertLessEqual(
                    abs(s - u) / scale, tolerance,
                    'Incompatible |M|^2 for %s (entry %s): standard=%s unrolhel=%s'
                    % (process, i, s, u))

    def _get_subprocess_dirs(self, root_dir):
        subproc_dirs = [pjoin(root_dir, name) for name in sorted(os.listdir(root_dir))
                        if name.startswith('P') and os.path.isdir(pjoin(root_dir, name))]
        self.assertTrue(subproc_dirs,
                        'No subprocess directory found in %s' % root_dir)
        return subproc_dirs

    def _run_standalone(self, subproc_dir):
        retcode = self._call_with_optional_redirection(['make', 'check'], subproc_dir)
        self.assertEqual(retcode, 0,
                         'Failed to compile standalone check in %s' % subproc_dir)
        # ./check <sqrts> <ntry> <flavor>: evaluate the same point several times
        # so the good-helicity warmup completes and the runtime helicity masking
        # actually engages (the printed value is from the last, masked, call).
        output = subprocess.Popen(['./check', '1000', '10', '0'],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  cwd=subproc_dir).communicate()[0].decode()
        values = [float(m.group('value')) for m in
                  re.finditer(r'Matrix element\s*=\s*(?P<value>[\d\.eEdD\+-]+)',
                              output.replace('D', 'E').replace('d', 'e'))]
        self.assertTrue(values,
                        'No matrix element printed by ./check in %s:\n%s'
                        % (subproc_dir, output))
        return values

    def _call_with_optional_redirection(self, command, cwd):
        if logger.isEnabledFor(logging.INFO):
            return subprocess.call(command, cwd=cwd)
        with open(os.devnull, 'w') as devnull:
            return subprocess.call(command, stdout=devnull, stderr=devnull, cwd=cwd)


class TestStandaloneUnrolhelConsistency(StandaloneUnrolhelConsistency):
    pass

    # single-topology, combined (gamma + Z = FFV6_2) routines
    test_unrolhel_ee_mumu = unrolhel_consistency_test_factory(
        'e+ e- > mu+ mu-', model='sm')

    # cross-topology + non-trivial color
    test_unrolhel_uux_ddxg = unrolhel_consistency_test_factory(
        'u u~ > d d~ g', model='sm')

    # massive (3-state) external vector + identical particles + reused wf slot
    test_unrolhel_uux_uuxz = unrolhel_consistency_test_factory(
        'u u~ > u u~ z', model='sm')

    # combined H+M (flavor-coupling) routines + massive externals + reused slot
    test_unrolhel_uux_uuxwpwm = unrolhel_consistency_test_factory(
        'u u~ > u u~ w+ w-', model='sm')

    # VBF-like multi-topology with a non-trivial broken-symmetry flavor factor
    # (BROKEN_SYM(FLAVOR) = 2 here, which must be applied to the summed |M|^2)
    test_unrolhel_ud_wpwmud = unrolhel_consistency_test_factory(
        'u d > w+ w- u d QCD=0', model='sm')

    # merged-flavor subprocess: exercises the per-flavor flavor mask (the
    # amplitude IAND guards must wrap the whole SCRATCH->AMP remap) and the
    # per-flavor good-helicity warmup reset (check_sa evaluates several flavors
    # with a shared, ever-incrementing NTRY).
    test_unrolhel_pp_wpwm = unrolhel_consistency_test_factory(
        'p p > w+ w- QCD=0', model='sm')

    # merged-flavor *with both same- and distinct-flavor assignments in a single
    # subprocess* (e.g. d d > w+ w- d d and d s > w+ w- d s in P1_QQ_wpwmQQ).
    # This is the process that historically exposed two independent bugs at once:
    #   - same-flavor lines: stale SCRATCH leaking into a flavor-skipped diagram
    #     (needs the IAND guard to wrap the whole CALL + SCRATCH->AMP remap);
    #   - distinct-flavor lines: the missing *BROKEN_SYM(FLAVOR) factor, which
    #     came out exactly a factor 2 low.
    # check_sa loops over all flavor combinations, so both classes are compared
    # against the standard standalone in one go.
    test_unrolhel_pp_wpwmjj = unrolhel_consistency_test_factory(
        'p p > w+ w- j j QCD=0', model='sm')

    # merged-flavor QCD process spanning several subprocess groups (q g, g q,
    # q q~) with a gluon t-channel and a b quark in the jet definition: checks
    # the flavor machinery together with QCD color flows and a massive merged
    # flavor member.
    test_unrolhel_pp_zj = unrolhel_consistency_test_factory(
        'p p > z j', model='sm')

    # --unrolhel has no flavor selection other than the per-flavor mask, so
    # --mask=False must be ignored (forced back on) rather than silently summing
    # all flavor channels.  Same merged process as above with --mask=False must
    # still agree with the standard standalone.
    test_unrolhel_pp_wpwm_nomask = unrolhel_consistency_test_factory(
        'p p > w+ w- QCD=0', model='sm', unrol_opts='--mask=False')

    # 4-point (VVVV) contact vertex: exercises the helicity-recycling amplitude
    # factorization for a 4-leg vertex (the H '_0' routine is built as the P1N
    # current of the last leg over the three other legs and closed by the final
    # contraction).  Must agree bit-for-bit with the standard standalone.
    test_unrolhel_aa_wpwm = unrolhel_consistency_test_factory(
        'a a > w+ w-', model='sm')

    # 2 -> 5 process: large helicity space (NCOMB=288), many diagrams and a deep
    # wavefunction chain -- exercises the good-helicity warmup/masking and the
    # per-wavefunction cartesian indexing at scale, well beyond the 2 -> 4 cases
    # above.  Single-flavor (no merge), so it must agree bit-for-bit.
    # NOTE: the *merged*-flavor 2 -> 5 case (e.g. q q > w+ w- q q g with a
    # multi-flavor q) currently disagrees with the standard standalone by ~1-4%;
    # that is a separate, pre-existing bug and is intentionally NOT covered here.
    test_unrolhel_uu_wpwmuug = unrolhel_consistency_test_factory(
        'u u > w+ w- u u g', model='sm')
