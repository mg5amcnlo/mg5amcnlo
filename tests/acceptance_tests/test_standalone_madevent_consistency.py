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
import madgraph.various.process_checks as process_checks


pjoin = os.path.join


def _sanitize_process_name(process):
    return re.sub(r'[^A-Za-z0-9]+', '_', process).strip('_').lower()


def matrix_element_consistency_test_factory(process, model='sm', tolerance=1e-6):
    def test(self):
        self.check_process(process, model=model, tolerance=tolerance)
    test.__name__ = 'test_%s' % _sanitize_process_name(process)
    test.__doc__ = 'Check standalone and madevent matrix elements agree for %s.' % process
    return test


class StandaloneMadeventMatrixElementConsistency(unittest.TestCase):

    debugging = getattr(unittest, 'debug', False)

    def setUp(self):
        self.cmd = cmd_interface.MasterCmd()
        self.cmd.no_notification()
        if not self.debugging:
            self.tmpdir = tempfile.mkdtemp(prefix='amc')
        else:
            self.tmpdir = tempfile.mkdtemp(prefix='amc_debug_')
        self.standalone_dir = pjoin(self.tmpdir, 'StandaloneProcess')
        self.madevent_dir = pjoin(self.tmpdir, 'MadEventProcess')

    def tearDown(self):
        if not self.debugging and os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def do(self, line):
        self.cmd.exec_cmd(line)

    def check_process(self, process, model='sm', tolerance=1e-6):
        self.do('set automatic_html_opening False')
        self.do('set group_subprocesses False')
        self.do('set apply_flavor_grouping True')
        self.do('set zerowidth_tchannel False')
        self.do('import model %s' % model)
        self.do('generate %s' % process)
        generated_process = self.cmd._curr_amps[0].get('process')
        self.do('output standalone %s -f' % self.standalone_dir)
        self.do('output madevent %s -f' % self.madevent_dir)

        standalone_dir = self._get_single_subprocess_dir(
            pjoin(self.standalone_dir, 'SubProcesses'))
        madevent_dir = self._get_single_subprocess_dir(
            pjoin(self.madevent_dir, 'SubProcesses'))

        standalone_rows, printed_phase_space = self._run_standalone(standalone_dir)
        seeded_phase_space = self._get_seeded_phase_space(generated_process)
        self._assert_phase_space_reasonable(
            printed_phase_space, seeded_phase_space, standalone_dir)
        madevent_by_iflav = self._run_hacked_madevent(madevent_dir, seeded_phase_space)

        self.assertTrue(len(standalone_rows) <= len(madevent_by_iflav),
                         'Flavor-count mismatch for %s: standalone=%s madevent=%s'
                         % (process, len(standalone_rows), len(madevent_by_iflav)))

        for iflav, standalone_row in enumerate(standalone_rows, start=1):
            self.assertIn(iflav, madevent_by_iflav,
                          'Missing madevent flavor index %s for %s' % (iflav, process))
            standalone_me = standalone_row['value']
            madevent_me = madevent_by_iflav[iflav]
            scale = max(abs(standalone_me), abs(madevent_me), 1e-99)
            misc.sprint('flavor=%s: diff=%f%%'%(
                         standalone_row['pdg'], 100 * abs(standalone_me - madevent_me) / scale if scale != 0 else 0))
            self.assertLessEqual(
                abs(standalone_me - madevent_me) / scale,
                tolerance,
                'Incompatible matrix elements for %s flavor=%s iflav=%s: standalone=%s madevent=%s'
                % (process, standalone_row['pdg'], iflav, standalone_me, madevent_me))

    def _get_single_subprocess_dir(self, root_dir):
        subproc_dirs = [pjoin(root_dir, name) for name in sorted(os.listdir(root_dir))
                        if name.startswith('P') and os.path.isdir(pjoin(root_dir, name))]
        self.assertEqual(len(subproc_dirs), 1,
                         'Expected a single subprocess directory in %s, got %s'
                         % (root_dir, subproc_dirs))
        return subproc_dirs[0]

    def _run_standalone(self, subproc_dir):
        retcode = self._call_with_optional_redirection(['make', 'check'], subproc_dir)
        self.assertEqual(retcode, 0, 'Failed to compile standalone check in %s' % subproc_dir)

        output = subprocess.Popen(['./check', '1000'],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  cwd=subproc_dir).communicate()[0].decode()

        ps_pattern = re.compile(
            r'^\s*\d+\s+'
            r'(?P<e>[\d\.eE\+-]+)\s+'
            r'(?P<px>[\d\.eE\+-]+)\s+'
            r'(?P<py>[\d\.eE\+-]+)\s+'
            r'(?P<pz>[\d\.eE\+-]+)',
            re.MULTILINE)
        phase_space = [[float(match.group(name)) for name in ('e', 'px', 'py', 'pz')]
                       for match in ps_pattern.finditer(output)]
        self.assertTrue(phase_space, 'No phase-space point found in %s' % subproc_dir)
        return self._extract_standalone_flavors(output, subproc_dir), phase_space

    def _get_seeded_phase_space(self, process_obj, energy=1000.0):
        evaluator = process_checks.MatrixElementEvaluator(
            process_obj.get('model'), cmd=self.cmd)
        phase_space = process_checks._get_seeded_python_momenta(
            process_obj, evaluator, energy)
        self.assertTrue(phase_space,
                        'Failed to generate seeded phase-space point for %s'
                        % process_obj.nice_string())
        return phase_space

    def _assert_phase_space_reasonable(self, printed, seeded, subproc_dir):
        self.assertEqual(len(printed), len(seeded),
                         'Mismatch in particle count for printed/seeded phase-space in %s'
                         % subproc_dir)
        for ipart, (printed_vec, seeded_vec) in enumerate(zip(printed, seeded), start=1):
            for icomp, (printed_val, seeded_val) in enumerate(zip(printed_vec, seeded_vec)):
                tolerance = max(1e-3, 1e-6 * max(abs(seeded_val), 1.0))
                self.assertLessEqual(
                    abs(printed_val - seeded_val), tolerance,
                    'Printed phase-space seems inconsistent in %s at particle=%s component=%s: '
                    'printed=%s seeded=%s'
                    % (subproc_dir, ipart, icomp, printed_val, seeded_val))

    def _run_hacked_madevent(self, subproc_dir, phase_space):
        source_dir = pjoin(self.madevent_dir, 'Source')
        retcode = self._call_with_optional_redirection(['make'], source_dir)
        self.assertEqual(retcode, 0, 'Failed to compile MadEvent source in %s' % source_dir)

        self._write_hacked_driver(pjoin(subproc_dir, 'driver.f'), phase_space)

        retcode = self._call_with_optional_redirection(['make', 'madevent'], subproc_dir)
        self.assertEqual(retcode, 0, 'Failed to compile hacked madevent in %s' % subproc_dir)

        output = subprocess.Popen(['./madevent'],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  cwd=subproc_dir).communicate()[0].decode()
        return self._extract_madevent_by_iflav(output, subproc_dir)

    def _call_with_optional_redirection(self, command, cwd):
        if logger.isEnabledFor(logging.INFO):
            return subprocess.call(command, cwd=cwd)
        with open(os.devnull, 'w') as devnull:
            return subprocess.call(command, stdout=devnull, stderr=devnull, cwd=cwd)

    def _extract_standalone_flavors(self, output, subproc_dir):
        lines = output.splitlines()
        standalone_rows = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith('PDG'):
                continue
            pdg_values = tuple(int(token) for token in re.findall(r'-?\d+', stripped))
            me_value = None
            for next_line in lines[index + 1:]:
                match = re.search(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)',
                                  next_line)
                if match:
                    me_value = float(match.group('value'))
                    break
                if next_line.strip().startswith('PDG'):
                    break
            if me_value is not None:
                standalone_rows.append({'pdg': pdg_values, 'value': me_value})
        self.assertTrue(standalone_rows, 'No flavor matrix elements found in %s' % subproc_dir)
        return standalone_rows

    def _extract_madevent_by_iflav(self, output, subproc_dir):
        lines = output.splitlines()
        by_iflav = {}
        current_iflav = None
        for line in lines:
            iflav_match = re.search(r'IFLAV\s*=\s*(\d+)', line)
            if iflav_match:
                current_iflav = int(iflav_match.group(1))
                continue
            me_match = re.search(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)', line)
            if me_match and current_iflav is not None:
                by_iflav[current_iflav] = float(me_match.group('value'))
                current_iflav = None
        self.assertTrue(by_iflav, 'No madevent flavor matrix elements found in %s' % subproc_dir)
        return by_iflav

    def _write_hacked_driver(self, driver_path, phase_space):
        lines = [
            '      PROGRAM DRIVER',
            '      use model_object',
            '      IMPLICIT NONE',
            "      INCLUDE 'genps.inc'",
            "      INCLUDE 'nexternal.inc'",
            "      INCLUDE 'maxamps.inc'",
            "      INCLUDE 'coupl.inc'",
            '      REAL*8 ZERO',
            '      PARAMETER (ZERO=0D0)',
            '      INTEGER SELECTED_HEL, SELECTED_COL, IFLAV, IVEC, J',
            '      INTEGER FLAVOR(NEXTERNAL)',
            '      REAL*8 P(0:3,NEXTERNAL), ANS',
            '      REAL*8 POL(2)',
            '      COMMON/TO_POLARIZATION/POL',
            '      INTEGER ISUM_HEL',
            '      LOGICAL MULTI_CHANNEL',
            '      COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL',
            '      LOGICAL INIT_MODE',
            '      COMMON /TO_DETERMINE_ZERO_HEL/INIT_MODE',
            '      LOGICAL ALLOW_HELICITY_GRID_ENTRIES',
            '      COMMON/TO_ALLOW_HELICITY_GRID_ENTRIES/ALLOW_HELICITY_GRID_ENTRIES',
            '      INTEGER MINCFIG, MAXCFIG',
            '      COMMON/TO_CONFIGS/MINCFIG, MAXCFIG',
            '      INTEGER NB_SPIN_STATE(2)',
            '      COMMON /NB_HEL_STATE/ NB_SPIN_STATE',
            '      CHARACTER*30 PARAM_CARD_NAME',
            '      COMMON/TO_PARAM_CARD_NAME/PARAM_CARD_NAME',
            '      REAL*8 PMASS(NEXTERNAL)',
            '      COMMON/TO_MASS/PMASS',
            "      PARAM_CARD_NAME='param_card.dat'",
            '      CALL SETRUN',
            '      CALL SETPARA(PARAM_CARD_NAME)',
            "      INCLUDE 'pmass.inc'",
            '      POL(1)=1D0',
            '      POL(2)=1D0',
            '      ISUM_HEL=0',
            '      MULTI_CHANNEL=.FALSE.',
            '      HEL_PICKED=0',
            '      HEL_JACOBIAN=1D0',
            '      INIT_MODE=.FALSE.',
            '      ALLOW_HELICITY_GRID_ENTRIES=.FALSE.',
            '      MINCFIG=1',
            '      MAXCFIG=1',
            '      NB_SPIN_STATE(1)=2',
            '      NB_SPIN_STATE(2)=2',
            '      IVEC=1']

        for index, momentum in enumerate(phase_space):
            iparticle = index + 1
            for component, value in enumerate(momentum):
                if isinstance(value, str):
                    formatted_value = value.replace('e', 'd').replace('E', 'D')
                else:
                    formatted_value = ('%.17E' % value).replace('E', 'D')
                lines.append('      P(%d,%d)=%s' %
                             (component, iparticle, formatted_value))

        lines.extend([
            '      DO IFLAV=1,MAXFLAVPERPROC',
            '         CALL GET_FLAVOR(IFLAV,FLAVOR)',
            '         CALL SMATRIX(P, IFLAV, 0.5D0, 0.5D0, 1, IVEC, ANS,',
            '     $    SELECTED_HEL, SELECTED_COL)',
            "         WRITE(*,*) 'IFLAV = ', IFLAV",
            "         WRITE(*,*) 'PDG', (FLAVOR(J),J=1,NEXTERNAL)",
            "         WRITE(*,*) 'Matrix element = ', ANS, ' GeV^',-(2*NEXTERNAL-8)",
            '      ENDDO',
            '      END',
            '',
            '      SUBROUTINE OPEN_FILE_LOCAL(LUN,FILENAME,FOPENED)',
            '      IMPLICIT NONE',
            '      INTEGER LUN',
            '      LOGICAL FOPENED',
            '      CHARACTER*(*) FILENAME',
            '      FOPENED=.FALSE.',
            "      OPEN(UNIT=LUN,FILE=FILENAME,STATUS='OLD',ERR=10)",
            '      FOPENED=.TRUE.',
            '      RETURN',
            ' 10   CONTINUE',
            '      RETURN',
            '      END',
            ''])

        with open(driver_path, 'w') as driver:
            driver.write('\n'.join(lines))


class TestStandaloneMadeventMatrixElementConsistency(
        StandaloneMadeventMatrixElementConsistency):
    pass    

    test_standalone_madevent_consistency_ee_ee = matrix_element_consistency_test_factory(
        'e+ e- > e+ e-', model='sm', tolerance=1e-6)

    test_standalone_madevent_consistency_ll_ll = matrix_element_consistency_test_factory(
        'l+ l- > l+ l-', model='sm', tolerance=1e-6)

    test_standalone_madevent_consistency_VBFZ_qqx = matrix_element_consistency_test_factory(
        '_quark _anti_quark > Z _quark _anti_quark QCD=0', model='sm', tolerance=1e-5)

    test_standalone_madevent_consistency_VBFZ_qq = matrix_element_consistency_test_factory(
        '_quark _quark > Z _quark _quark QCD=0', model='sm', tolerance=1e-5)
    
    test_standalone_madevent_consistency_VBFZ_qxqx = matrix_element_consistency_test_factory(
        '_anti_quark _anti_quark > Z _anti_quark _anti_quark QCD=0', model='sm', tolerance=1e-5)

    test_standalone_madevent_consistency_VBF_WW = matrix_element_consistency_test_factory(
        '_quark _quark > W+ W- _quark _quark QCD=0', model='sm', tolerance=1e-5)
    
    test_standalone_madevent_consistency_VBFH = matrix_element_consistency_test_factory(
        '_quark _anti_quark > H _quark _anti_quark QCD=0', model='sm', tolerance=1e-5)
    
    test_standalone_madevent_consistency_VBFHu = matrix_element_consistency_test_factory(
        'u u  > H u u QCD=0', model='sm', tolerance=1e-5)
    
    test_standalone_madevent_consistency_qq = matrix_element_consistency_test_factory(
        'u _quark  > u _quark QCD=0', model='sm', tolerance=1e-5)