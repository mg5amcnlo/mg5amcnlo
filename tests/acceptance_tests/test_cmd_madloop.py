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
from __future__ import division
from __future__ import absolute_import
import glob
import subprocess
import unittest
import os
import re
import shutil
import tempfile
import copy
import sys
import logging
import time
import tests.IOTests as IOTests
logger = logging.getLogger('test_cmd')

import tests.unit_tests.iolibs.test_file_writers as test_file_writers

import madgraph.interface.master_interface as MGCmd
import madgraph.interface.amcatnlo_run_interface as NLOCmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.various.misc as misc
import madgraph.core.color_amp as color_amp
import tests.IOTests as IOTests
import madgraph.various.lhe_parser as lhe_parser

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd

pjoin = os.path.join
path = os.path

#===============================================================================
# TestCmdLoop
#===============================================================================
class TestCmdLoop(unittest.TestCase):
    """this treats all the command not related to MG_ME"""
    logger_saved_info = {}
    debugging = True
    def setUp(self):
        """ Initialize the test """
        self.interface = MGCmd.MasterCmd()
        self.interface.no_notification()
        # Below the key is the name of the logger and the value is a tuple with
        # first the handlers and second the level.
        self.logger_saved_info = {}

        if not self.debugging:
            self.tmpdir = tempfile.mkdtemp(prefix='amc')
        else:
            if os.path.exists(pjoin(MG5DIR, 'TEST_AMC')):
                shutil.rmtree(pjoin(MG5DIR, 'TEST_AMC'))
            os.mkdir(pjoin(MG5DIR, 'TEST_AMC'))
            self.tmpdir = pjoin(MG5DIR, 'TEST_AMC')
            
        self.out_dir = pjoin(self.tmpdir,'MGProcess')
    
    def generate(self, process, model):
        """Create a process"""
        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception as error:
            pass

        self.interface.onecmd('import model %s' % model)
        if isinstance(process, str):
            self.interface.onecmd('generate %s' % process)
        else:
            for p in process:
                self.interface.onecmd('add process %s' % p)
        self.interface.onecmd('output /tmp/MGPROCESS -f')      
    
    def do(self, line):
        """ exec a line in the interface """        
        self.interface.exec_cmd(line)
    
    def run_cmd(self, line):
        """for third party call, call the line with pre and postfix treatment
        with global error handling"""
        
        return self.interface.exec_cmd(line, errorhandling=True, precmd=True, postcmd=True)
    
    @classmethod
    def setup_logFile_for_logger(cls,full_logname,restore=False,level=logging.DEBUG):
        """ Setup the logger by redirecting them all to logfiles in tmp """
        
        logs = full_logname.split('.')
        lognames = [ '.'.join(logs[:(len(logs)-i)]) for i in\
                                            range(len(full_logname.split('.')))]
        if not hasattr(cls, 'tmp_path'):
            # To store the path of the log files of each logger treated
            cls.tmp_path = {}

        for logname in lognames:
            my_logger = logging.getLogger(logname)       
            if restore:
                try:
                    if hasattr(cls, tmp_path) and logname in cls.tmp_path:
                        os.remove(cls.tmp_path[logname])
                except:
                    pass
                my_logger.removeHandler(cls.logger_saved_info[logname][0])
                my_logger.setLevel(cls.logger_saved_info[logname][1])
                for i, h in enumerate(my_logger.handlers):
                    h.setLevel(cls.logger_saved_info[logname][2][i])
            else:
                cls.tmp_path[logname] = tempfile.mktemp('', 'tmp', None)
                hdlr = logging.FileHandler(cls.tmp_path[logname])     
                # I assume below that the orders of the handlers in my_logger.handlers
                # remains the same after having added/removed the FileHandler
                cls.logger_saved_info[logname] = [hdlr,my_logger.level,\
                                                [h.level for h in my_logger.handlers]]
                for h in my_logger.handlers:
                    # This not elegant, but the only way I could find to mute this handlers
                    h.setLevel(logging.CRITICAL+1)
                my_logger.addHandler(hdlr)
                my_logger.setLevel(level)

        if not restore:
            for logname in lognames:
                logging.getLogger(logname).debug('Log of %s'%logname)
    
    def notest_ML_launch_gg_ddx(self):
        """test that the output works fine for g g > d d~ [virt=QCD]"""

        self.setup_logFile_for_logger('cmdprint.ext_program')
        try:
            cmd = os.getcwd()
            self.generate(['g g > d d~ [virt=QCD]'], 'loop_sm')
            self.assertEqual(cmd, os.getcwd())
            self.do('launch -f')
            
            # Test that the result of the run is present.
            self.assertTrue(path.isfile('/tmp/MGPROCESS/SubProcesses/P0_gg_ddx/result.dat'))
            # Test that the Helicity filter is correctly configured.
            self.assertTrue(path.isfile('/tmp/MGPROCESS/SubProcesses/P0_gg_ddx/HelFilter.dat'))
            # Test that the cmdprint log file is there
            self.assertTrue(path.isfile('/tmp/cmdprint.ext_program.log'))
            # Test that it contains the expected output. 
            # Of course I could setup a detailed regexpr to make sure the values
            # in the output are not NaN or so, but it is not really the idea of these
            # acceptance tests.
            self.assertIn(
                'Results for process gg > ddx',
                open('/tmp/cmdprint.ext_program.log').read()
            )
        except:
            self.setup_logFile_for_logger('cmdprint.ext_program',restore=True)      
            raise
        self.setup_logFile_for_logger('cmdprint.ext_program',restore=True)

    def test_ML_check_brs_gd_gd(self):
        """test that the brs check works fine on g d > g d"""
        
        self.setup_logFile_for_logger('madgraph.check_cmd')
        try:
            cmd = os.getcwd()
            self.do('import model loop_sm')
            self.do('check brs -reuse g d > g d [virt=QCD]')
            self.assertTrue(path.isfile(pjoin(MG5DIR,'TMP_CHECK',\
                                               'SubProcesses/P0_gd_gd/result.dat')))
            shutil.rmtree(pjoin(MG5DIR,'TMP_CHECK'))
            self.assertEqual(cmd, os.getcwd())
            self.assertTrue(path.isfile(self.tmp_path['madgraph.check_cmd']))
            res = open(self.tmp_path['madgraph.check_cmd']).read()
            self.assertIn('Process [virt=QCD]', res)
            self.assertIn('Summary: 1/1 passed, 0/1 failed', res)
            self.assertIn('BRS', res)
            self.assertIn('Passed', res)
        except:
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)      
            raise
        self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)

    def test_improve_ps_keeps_a_leg_at_rest(self):
        """IMPROVE_PS_POINT_PRECISION must not destroy an exactly-zero
        three-momentum.

        A polarised matrix element is evaluated in a frame where the selected
        leg is at rest, and boost_to_frame puts its three-momentum at exactly
        zero on purpose: HELAS reads the spin quantisation axis of a massive
        vector off 'pp.eq.rZero'. The PSMC branch of improve_ps used to
        restore momentum conservation by dumping the residual into leg
        NEXTERNAL unconditionally, so whenever the at-rest leg was the last
        one its exact zero came back as a ~1e-14 vector and the longitudinal
        polarisation vector was built along rounding noise. On g g > z{0}
        z{0} [noborn=QCD] that split me_frame=[3] from me_frame=[4] by 26%%.

        This drives the routine directly, so the check is exact rather than
        statistical, and it covers both entry points: the double precision
        wrapper and the quad routine reached through SET_MP_PS on the
        stability-escalation path.
        """
        out_dir = pjoin(self.tmpdir, 'ML_improve_ps')
        self.do('import model loop_sm')
        self.do('generate g g > z z [sqrvirt=QCD]')
        self.do('output standalone %s -f' % out_dir)

        proc_dir = pjoin(out_dir, 'SubProcesses', 'P0_gg_zz')
        prefix = open(pjoin(proc_dir, 'proc_prefix.txt')).read().strip()

        # SET_MP_PS is the second entry point; it improves the raw momenta in
        # quad, so it has to reach the same routine the driver calls.
        loop_matrix = open(pjoin(proc_dir, 'loop_matrix.f')).read()
        set_mp_ps = loop_matrix.split('SUBROUTINE %sSET_MP_PS(P)' % prefix)[1]
        set_mp_ps = set_mp_ps.split('\n      END\n')[0]
        self.assertIn('CALL %sMP_IMPROVE_PS_POINT_PRECISION' % prefix,
                      set_mp_ps)

        driver = open(pjoin(_pickle_path,
                            'improve_ps_at_rest_driver.f')).read()
        driver = driver.replace('__PFX__', prefix)
        driver_path = pjoin(proc_dir, 'improve_ps_at_rest_driver.f')
        open(driver_path, 'w').write(driver)

        exe = pjoin(proc_dir, 'improve_ps_at_rest')
        compile = subprocess.Popen(
            ['gfortran', '-ffixed-line-length-132',
             '-I%s' % proc_dir, '-I%s' % pjoin(out_dir, 'SubProcesses'),
             driver_path, pjoin(proc_dir, 'improve_ps.f'), '-o', exe],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=proc_dir)
        (out, _) = compile.communicate()
        self.assertEqual(compile.returncode, 0,
                         'could not build the improve_ps driver:\n%s'
                         % out.decode())

        run = subprocess.Popen([exe], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, cwd=proc_dir)
        (out, _) = run.communicate()
        out = out.decode()
        self.assertEqual(run.returncode, 0, out)

        results = [l.split() for l in out.split('\n')
                   if l.startswith('RESULT ')]
        # 2 at-rest legs x 2 ImprovePSPoint settings x 2 entry points
        self.assertEqual(len(results), 8, out)
        for _, entry, leg, mode, px, py, pz in results:
            for label, value in (('px', px), ('py', py), ('pz', pz)):
                self.assertEqual(
                    float(value), 0.0,
                    'improve_ps moved leg %s off rest (%s, entry %s, '
                    'ImprovePSPoint=%s): %s = %s'
                    % (leg, label, entry, mode, label, value))

    def test_polarised_rotation_stability_leg_at_rest(self):
        """MadLoop's rotation stability test must remain a symmetry when a
        polarised massive particle is exactly at rest.

        A one-leg me_frame puts the selected particle at rest, and HELAS then
        takes its spin axis from the frame z axis instead of its momentum.
        NRotations_DP/QP used to re-evaluate the loop after axis permutations
        that move z, i.e. in a different polarisation state. On
        u u~ > z{0} z{0} [virt=QCD] with leg 3 at rest this returned code 420
        (EPS), a relative accuracy of 6.1, and a finite part 56%% off.
        ROTATE_PS now rotates about z whenever a leg is exactly at rest.

        The point is the boost_to_frame configuration: a c.m. point boosted
        into the rest frame of leg 3, its three-momentum then set to exactly
        zero.
        """
        import math
        out_dir = pjoin(self.tmpdir, 'ML_rotation_at_rest')
        self.do('import model loop_sm')
        self.do('generate u u~ > z{0} z{0} [virt=QCD]')
        self.do('output standalone %s -f' % out_dir)
        proc_dir = pjoin(out_dir, 'SubProcesses', 'P0_uux_z0z0')

        # check reads PS.input only when READPS is set
        check_sa = pjoin(proc_dir, 'check_sa.f')
        text = open(check_sa).read()
        self.assertIn('PARAMETER (READPS = .FALSE.)', text)
        open(check_sa, 'w').write(text.replace('PARAMETER (READPS = .FALSE.)',
                                               'PARAMETER (READPS = .TRUE.)'))
        make = subprocess.Popen(['make', 'check'], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, cwd=proc_dir)
        (out, _) = make.communicate()
        self.assertEqual(make.returncode, 0, out.decode())

        mz = None
        in_mass = False
        for line in open(pjoin(out_dir, 'Cards', 'param_card.dat')):
            fields = line.split()
            if line.lower().startswith('block'):
                in_mass = fields[1].lower() == 'mass'
            elif in_mass and fields and fields[0] == '23':
                mz = float(fields[1])
        self.assertIsNotNone(mz)

        energy = 250.0
        k = math.sqrt(energy**2 - mz**2)
        theta, phi = 0.7, 0.3
        p3 = [energy, k*math.sin(theta)*math.cos(phi),
              k*math.sin(theta)*math.sin(phi), k*math.cos(theta)]
        ps = [[energy, 0., 0., energy], [energy, 0., 0., -energy],
              p3, [energy, -p3[1], -p3[2], -p3[3]]]
        beta = [p3[i]/p3[0] for i in (1, 2, 3)]
        beta2 = sum(b*b for b in beta)
        gamma = 1.0/math.sqrt(1.0 - beta2)
        def boost(p):
            bp = sum(beta[i]*p[i+1] for i in range(3))
            shift = (gamma - 1.0)*bp/beta2 - gamma*p[0]
            return [gamma*(p[0] - bp)] + [p[i+1] + shift*beta[i]
                                          for i in range(3)]
        at_rest = [boost(p) for p in ps]
        at_rest[2] = [mz, 0.0, 0.0, 0.0]

        def write_point(point):
            with open(pjoin(proc_dir, 'PS.input'), 'w') as handle:
                for p in point:
                    handle.write(' '.join('%.17E' % x for x in p) + '\n')

        params = [f for f in glob.glob(pjoin(out_dir, '**', 'MadLoopParams.dat'),
                                       recursive=True)
                  if 'default' not in os.path.basename(f)]
        self.assertTrue(params)

        def evaluate(nrot):
            for path_ in params:
                lines = open(path_).read().split('\n')
                for key in ('#NRotations_DP', '#NRotations_QP'):
                    lines[lines.index(key) + 1] = str(nrot)
                open(path_, 'w').write('\n'.join(lines))
            run = subprocess.Popen(['./check'], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, cwd=proc_dir)
            (out, _) = run.communicate()
            out = out.decode()
            self.assertEqual(run.returncode, 0, out)
            # check evaluates the point NPSPOINTS times; the last one is
            # past MadLoop's initialisation
            finite = [float(l.split()[4]) for l in out.split('\n')
                      if 'Matrix element finite' in l][-1]
            # no accuracy line while MadLoop is still initialising
            accuracy = ([float(l.split()[3]) for l in out.split('\n')
                         if 'Relative accuracy' in l] or [None])[-1]
            code = [int(l.split()[-1]) for l in out.split('\n')
                    if 'MadLoop return code' in l][-1]
            return finite, accuracy, code

        # Leave MadLoop's initialisation stage (return code 1xx, filters not
        # yet written) on an ordinary point, so both measured evaluations
        # below run the real stability test.
        write_point(ps)
        for attempt in range(6):
            code = evaluate(0)[2]
            if code // 100 != 1:
                break
        self.assertNotEqual(code // 100, 1,
            'MadLoop did not leave its initialisation stage: code %d' % code)

        write_point(at_rest)
        finite0, _, code0 = evaluate(0)
        finite2, accuracy2, code2 = evaluate(2)
        self.assertEqual(code0 // 100, 2,
            'reference evaluation not stable: return code %d' % code0)
        self.assertEqual(code2 // 100, 2,
            'rotations flagged the at-rest point: return code %d, relative '
            'accuracy %.2e' % (code2, accuracy2))
        self.assertLess(accuracy2, 1e-6)
        self.assertLess(abs(finite2/finite0 - 1.0), 1e-8,
            'rotations changed the finite part: %.16e vs %.16e'
            % (finite2, finite0))

    def test_generate_output_semicolon_preserves_loop_process(self):
        """Regression test for combined generate/output in one command line."""

        with self.assertRaises(InvalidCmd) as ctx:
            self.interface.exec_cmd(
                'import model sm;generate e+ e- > t t~ [virt=QCD]; output bad>path',
                precmd=True,
                errorhandling=False
            )

        self.assertIn('not allowed in the output path', str(ctx.exception))
        self.assertNotIn('No processes generated', str(ctx.exception))
        self.assertTrue(self.interface._curr_amps)

    def test_loop_nc_memo_not_poisoned_across_generations(self):
        """Two loop generations in one process. The first one runs with
        compute_loop_nc=False; it must not pin loop_Nc_power in the
        process-wide color memo for the loop-induced generation that follows,
        which needs the real value to write coloramps.inc.

        Value-level discrimination lives in the unit test
        (test_color_amp.LoopNcMemoTest); this one guards the reported crash."""

        # The poisoning generation must be the one that fills the memo: if a
        # good value lands first the poisoning is inert and this test would
        # pass on a broken tree.
        color_amp.ColorBasis._canonical_dict.clear()

        self.do('import model loop_sm')
        self.do('generate g g > h [sqrvirt=QCD]')
        self.do('output standalone_fortran %s -f' % pjoin(self.tmpdir, 'sqrvirt'))

        self.do('generate g g > h [noborn=QCD]')
        self.do('output madevent %s -f' % pjoin(self.tmpdir, 'noborn'))

        coloramps = glob.glob(pjoin(self.tmpdir, 'noborn', 'SubProcesses',
                                    'P*', 'coloramps.inc'))
        self.assertTrue(coloramps)
        for f in coloramps:
            self.assertIn('ICOLAMP', open(f).read())

    def test_ML_check_full_epem_ttx(self):
        """ Test that check full e+ e- > t t~ works fine """
        
        self.setup_logFile_for_logger('madgraph.check_cmd')
        try:
            cmd = os.getcwd()
            self.do('import model loop_sm')
            self.do('check full -reuse e+ e- > t t~ [virt=QCD]')
            self.assertEqual(cmd, os.getcwd())
            self.assertTrue(path.isfile(pjoin(MG5DIR,'TMP_CHECK',\
                                            'SubProcesses/P0_epem_ttx/result.dat')))
            shutil.rmtree(pjoin(MG5DIR,'TMP_CHECK'))
            self.assertTrue(path.isfile(self.tmp_path['madgraph.check_cmd']))
            res = open(self.tmp_path['madgraph.check_cmd']).read()
            # Needs the loop_sm feynman model to successfully run the gauge check.
            # self.assertTrue('Gauge results' in res)
            self.assertIn('Lorentz invariance results', res)
            self.assertIn('Process permutation results:', res)
            self.assertIn('Gauge results', res)
            self.assertIn('Summary: passed', res)
            self.assertIn('Passed', res)
            self.assertNotIn('Failed', res)
            self.assertNotIn('1/1 failed', res)
            self.assertIn('1/1 passed', res)
        except:
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
            raise
        self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)

    def test_ML_gauge_invariance_epem_ttx_all_modes(self):
        """Cross-check e+ e- > t t~ [virt=QCD] in all four combinations
        of gauge (Feynman/Unitary) and output mode (optimized/non-optimized).

        The matrix element squared should be identical (up to numerical
        noise) across all four combinations.  In particular, the
        non-optimized Unitary path goes through the bare-L ALOHA helpers
        and the type(aloha) / type(mp_aloha) WL wavefunction arrays — it
        is the path that the recent refactor specifically targets, and
        any layout / accessor mismatch there will show up as a finite
        relative deviation from the other three configurations."""

        import aloha
        import madgraph.various.process_checks as pcheck
        import madgraph.loop.loop_diagram_generation as ldg
        import madgraph.loop.loop_helas_objects as lho

        self.setup_logFile_for_logger('madgraph.check_cmd')
        try:
            self.do('import model loop_sm')

            # Build the loop amplitude once; we reuse it in every mode.
            self.interface.exec_cmd('generate e+ e- > t t~ [virt=QCD]')
            process = self.interface._curr_amps[0].get('process')

            cuttools_dir = pjoin(MG5DIR, 'vendor', 'CutTools')
            mg_options = self.interface.options

            orig_optimized = mg_options.get('loop_optimized_output')
            orig_unitary = aloha.unitary_gauge

            # Generate a shared phase-space point so the four evaluations
            # are comparable.
            tmp_evaluator = pcheck.LoopMatrixElementEvaluator(
                cuttools_dir=cuttools_dir,
                cmd=self.interface,
                model=self.interface._curr_model,
                auth_skipping=False,
                output_path=MG5DIR,
                reuse=False)
            shared_p, _ = tmp_evaluator.get_momenta(process)

            results = {}
            try:
                for optimized in (True, False):
                    for unitary in (False, True):
                        label = ('opt' if optimized else 'noopt') + \
                                ('_unitary' if unitary else '_feynman')

                        aloha.unitary_gauge = unitary
                        mg_options['loop_optimized_output'] = optimized

                        amplitude = ldg.LoopAmplitude(process)
                        matrix_element = lho.LoopHelasMatrixElement(
                            amplitude, optimized_output=optimized)

                        # Each evaluator gets its own temporary export
                        # directory; do not let proliferation reuse a
                        # directory generated under a different gauge.
                        for d in misc.glob('TMP_CHECK*', MG5DIR):
                            if path.isdir(d):
                                shutil.rmtree(d, ignore_errors=True)
                        evaluator = pcheck.LoopMatrixElementEvaluator(
                            cuttools_dir=cuttools_dir,
                            cmd=self.interface,
                            model=self.interface._curr_model,
                            auth_skipping=False,
                            output_path=MG5DIR,
                            reuse=False)

                        data = evaluator.evaluate_matrix_element(
                            matrix_element, p=shared_p,
                            output='jamp')
                        self.assertIsNotNone(
                            data,
                            '%s: evaluator returned None' % label)
                        self.assertIn(
                            'm2', data,
                            '%s: no m2 key in evaluator output' % label)
                        results[label] = data['m2']

                        for d in misc.glob('TMP_CHECK*', MG5DIR):
                            if path.isdir(d):
                                shutil.rmtree(d, ignore_errors=True)
            finally:
                mg_options['loop_optimized_output'] = orig_optimized
                aloha.unitary_gauge = orig_unitary

            # All four answers should agree to a few parts per million.
            self.assertEqual(len(results), 4)
            ref_label, ref_value = next(iter(results.items()))
            tol = 1e-6
            for label, value in results.items():
                denom = max(abs(value), abs(ref_value), 1e-30)
                relerr = abs(value - ref_value) / denom
                self.assertLess(
                    relerr, tol,
                    'm2 mismatch: %s=%.10e vs %s=%.10e (relerr=%.3e). '
                    'Full table: %s' %
                    (label, value, ref_label, ref_value, relerr, results))
        except:
            self.setup_logFile_for_logger('madgraph.check_cmd', restore=True)
            raise
        self.setup_logFile_for_logger('madgraph.check_cmd', restore=True)

    def test_ML_check_timing_epem_ttx(self):
        """ Test that check timing e+ e- > t t~ works fine """
        
        self.setup_logFile_for_logger('madgraph.check_cmd')
        try:
            cmd = os.getcwd()
            self.do('import model loop_sm')
            if path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')):
                shutil.rmtree(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx'))
            # Make sure it works for an initial run
            self.do('check timing -reuse e+ e- > t t~ [virt=QCD]')
            self.assertEqual(cmd, os.getcwd())
            self.assertTrue(path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')))
            self.assertTrue(path.isfile(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx',\
                                            'SubProcesses/P0_epem_ttx/result.dat')))
            self.assertTrue(path.isfile(self.tmp_path['madgraph.check_cmd']))
            res = open(self.tmp_path['madgraph.check_cmd']).read()
            self.assertIn('Generation time total', res)
            self.assertIn('Executable size', res)
            self.assertTrue(not 'NA' in res)
            
            # Now for a Reuse-run
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
            self.setup_logFile_for_logger('madgraph.check_cmd')
            self.do('check timing -reuse e+ e- > t t~ [virt=QCD]')
            self.assertEqual(cmd, os.getcwd())
            self.assertTrue(path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')))
            self.assertTrue(path.isfile(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx',\
                                            'SubProcesses/P0_epem_ttx/result.dat')))
            shutil.rmtree(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx'))
            self.assertTrue(path.isfile(self.tmp_path['madgraph.check_cmd']))
            res = open(self.tmp_path['madgraph.check_cmd']).read()
            self.assertIn('Generation time total', res)
            self.assertIn('Executable size', res)
            self.assertLessEqual(res.count('NA'), 10)
        except:
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
            if path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')):
                shutil.rmtree(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx'))
            raise
        self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)

    def test_ML_check_profile_epem_ttx(self):
        """ Test that check profile e+ e- > t t~ works fine """

        self.setup_logFile_for_logger('madgraph.check_cmd')
        try:
            cmd = os.getcwd()
            self.do('import model loop_sm')
            if path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')):
                shutil.rmtree(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx'))
            
            # Make sure it works for an initial run
            self.do('check profile -reuse e+ e- > t t~ [virt=QCD]')
            self.assertEqual(cmd, os.getcwd())
            self.assertTrue(path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')))
            self.assertTrue(path.isfile(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx',\
                                            'SubProcesses/P0_epem_ttx/result.dat')))
            self.assertTrue(path.isfile(self.tmp_path['madgraph.check_cmd']))
            res = open(self.tmp_path['madgraph.check_cmd']).read()
            self.assertIn('Generation time total', res)
            self.assertIn('Executable size', res)
            self.assertIn('Tool (DoublePrec for CT)', res)
            self.assertIn('Number of Unstable PS points', res)
            self.assertLessEqual(res.count('NA'), 3)

            # Now for a Reuse-run
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
            self.setup_logFile_for_logger('madgraph.check_cmd')
            self.do('check profile -reuse e+ e- > t t~ [virt=QCD]')
            self.assertEqual(cmd, os.getcwd())
            self.assertTrue(path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')))
            self.assertTrue(path.isfile(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx',\
                                            'SubProcesses/P0_epem_ttx/result.dat')))
            shutil.rmtree(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx'))
            self.assertTrue(path.isfile(self.tmp_path['madgraph.check_cmd']))
            res = open(self.tmp_path['madgraph.check_cmd']).read()
            self.assertIn('Generation time total', res)
            self.assertIn('Executable size', res)
            self.assertIn('Tool (DoublePrec for CT)', res)
            self.assertIn('Number of Unstable PS points', res)
            self.assertLessEqual(res.count('NA'), 11)
        except:
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
            if path.isdir(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx')):
                shutil.rmtree(pjoin(MG5DIR,'SAVEDTMP_CHECK_epem_ttx'))
            raise
        self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)

    def test_ML_check_cms_al_lvlvlx_LO(self):
        """ Test that check cms a l- > l- vl vl~ passes at leading order."""

        try:
            self.setup_logFile_for_logger('madgraph.check_cmd')
            cwd = os.getcwd()
            # Disable flavor grouping so multiparticles like l- expand to
            # individual flavors (e-, mu-) rather than the merged _lepton PDG.
            # Merged particles are not supported for CMS checks and would cause
            # only a single merged process to be tested instead of all flavor
            # combinations.
            self.do('set apply_flavor_grouping False')
            # Change this when we will make the CMS-ready EW model the default
            self.do('import model sm')
            self.do('define l- = e- mu-')
            self.do('define l+ = e+ mu+')
            self.do('define vl = ve vm vt')   
            self.do('define vl~ = ve~ vm~ vt~')       
            # Make sure it works for an initial run
            command = 'check cms -reuse a l- > l- vl vl~ '
            options = {'name':'acceptance_test_alm_lmvlvlx_LO',
                       'lambdaCMS':'(1.0e-5,2)',
                       'show_plot':'False',
                       'seed':'666',
                       'resonances':'all',
                       'recompute_width':'first_time',
                       'report':'full'}
            self.do(command+' '.join('--%s=%s'%(opt, value) for opt, value in 
                                                               options.items()))
            self.assertEqual(cwd, os.getcwd())
            self.assertTrue(path.isfile(self.tmp_path['madgraph.check_cmd']))
            res = open(self.tmp_path['madgraph.check_cmd']).read()
            self.assertEqual(res.count('=== FAILED ==='), 0)
            self.assertEqual(res.count('=== PASSED ==='), 10)
            self.assertIn('Summary: 10/10 passed.', res)
            self.assertTrue(path.isfile(pjoin(MG5DIR,
                                            'acceptance_test_alm_lmvlvlx_LO.log')))
            res = open(pjoin(MG5DIR,'acceptance_test_alm_lmvlvlx_LO.log')).read()
            self.assertEqual(res.count('=== FAILED ==='), 0)
            self.assertEqual(res.count('=== PASSED ==='), 10)
            self.assertIn('Summary: 10/10 passed.', res)
            self.assertTrue(path.isfile(pjoin(MG5DIR,
                                         'acceptance_test_alm_lmvlvlx_LO.pkl')))
            
            # Now for a reuse run using --analyze
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
            self.setup_logFile_for_logger('madgraph.check_cmd')
            os.remove(pjoin(MG5DIR,'acceptance_test_alm_lmvlvlx_LO.log'))
            self.do('check cms --analyze=%s --show_plot=False --report=full'%
                             pjoin(MG5DIR,'acceptance_test_alm_lmvlvlx_LO.pkl'))
            self.assertEqual(cwd, os.getcwd())
            self.assertTrue(path.isfile(self.tmp_path['madgraph.check_cmd']))
            res = open(self.tmp_path['madgraph.check_cmd']).read()
            self.assertEqual(res.count('=== FAILED ==='), 0)
            self.assertEqual(res.count('=== PASSED ==='), 10)
            self.assertIn('Summary: 10/10 passed.', res)
            self.assertTrue(path.isfile(pjoin(MG5DIR,
                                         'acceptance_test_alm_lmvlvlx_LO.pkl')))
            
            # Finally rerun it but this time using lambda_diff_power = 2
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
            self.setup_logFile_for_logger('madgraph.check_cmd')
            os.remove(pjoin(MG5DIR,'acceptance_test_alm_lmvlvlx_LO.pkl'))
            options['diff_lambda_power']='2'
            self.do(command+' '.join('--%s=%s'%(opt, value) for opt, value in 
                                                               options.items()))
            self.assertEqual(cwd, os.getcwd())
            self.assertTrue(path.isfile(self.tmp_path['madgraph.check_cmd']))
            res = open(self.tmp_path['madgraph.check_cmd']).read()
            self.assertEqual(res.count('=== FAILED ==='), 6)
            self.assertEqual(res.count('=== PASSED ==='), 4)
            self.assertIn('Summary: 4/10 passed, failed checks are for:', res)
            self.assertTrue(path.isfile(pjoin(MG5DIR,
                                         'acceptance_test_alm_lmvlvlx_LO.log')))
            res = open(pjoin(MG5DIR,'acceptance_test_alm_lmvlvlx_LO.log')).read()
            self.assertEqual(res.count('=== FAILED ==='), 6)
            self.assertEqual(res.count('=== PASSED ==='), 4)
            self.assertIn('Summary: 4/10 passed, failed checks are for:', res)
            self.assertTrue(path.isfile(pjoin(MG5DIR,
                                         'acceptance_test_alm_lmvlvlx_LO.pkl')))

            # Clean up duties
            os.remove(pjoin(MG5DIR,'acceptance_test_alm_lmvlvlx_LO.log'))
            os.remove(pjoin(MG5DIR,'acceptance_test_alm_lmvlvlx_LO.pkl'))
        except Exception as e:
            try:
                self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
                os.remove(pjoin(MG5DIR,'acceptance_test_alm_lmvlvlx_LO.log'))
                os.remove(pjoin(MG5DIR,'acceptance_test_alm_lmvlvlx_LO.pkl'))
            except:
                pass
            raise e
        self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)

    def test_ML_check_cms_aem_emvevex(self):
        """ Test that check cms a e- > e- ve ve~ [virt=QCD QED] works fine """

        self.setup_logFile_for_logger('madgraph.check_cmd')
        files = ['acceptance_test_aem_emvevex.pkl',
                 'acceptance_test_aem_emvevex.log',
                 'acceptance_test_aem_emvevex_widths_increased.pkl',
                 'acceptance_test_aem_emvevex_widths_increased.log']
        for f in files:
            if os.path.exists(f):
                os.remove(f)
                
        output_name = 'SAVEDTMP_CHECK_acceptance_test_aem_emvevex__%s__'
        
        try:
            cwd = os.getcwd()
            
            # Change this when we will make the CMS-ready EW model the default
            self.do('import model loop_qcd_qed_sm')
            for mode in ['NWA','CMS']:
                if path.isdir(pjoin(MG5DIR,output_name%mode)):
                    shutil.rmtree(pjoin(MG5DIR,output_name%mode))
            
            # Make sure it works for an initial run
            command = 'check cms -reuse a e- > e- ve ve~ [virt=QCD QED] '
            options = {'name':'acceptance_test_aem_emvevex',
                       'lambdaCMS':'(1.0e-6,2)',
                       'show_plot':'False',
                       'seed':'666',
                       'resonances':'2',
                       'recompute_width':'first_time',
                       'report':'full'}
            cmd = command+' '.join('--%s=%s'%(opt, value) for opt, value in 
                                                                options.items())
            
            #print "Running first CMS check cmd: %s" %cmd
            self.do(cmd)
            self.assertEqual(cwd, os.getcwd())
            for mode in ['NWA','CMS']:
                self.assertTrue(path.isdir(pjoin(MG5DIR,output_name%mode)))
            self.assertTrue(path.isfile(pjoin(MG5DIR,
                                            'acceptance_test_aem_emvevex.pkl')))
            self.assertTrue(path.isfile(self.tmp_path['madgraph.check_cmd']))
            res = open(self.tmp_path['madgraph.check_cmd']).read()
            self.assertEqual(res.count('=== FAILED ==='), 0)
            self.assertEqual(res.count('=== PASSED ==='), 2)
            self.assertTrue(path.isfile(pjoin(MG5DIR,
                                            'acceptance_test_aem_emvevex.log')))
            res = open(pjoin(MG5DIR,'acceptance_test_aem_emvevex.log')).read()
            self.assertEqual(res.count('=== FAILED ==='), 0)
            self.assertEqual(res.count('=== PASSED ==='), 2)
                        
            # Now for a Reuse-run with the widths modified by 1%
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
            self.setup_logFile_for_logger('madgraph.check_cmd')
            # Now copy the card with recomputed widths in it
            for mode in ['NWA','CMS']:
                self.assertTrue(path.isfile(pjoin(MG5DIR,output_name%mode,
                                   'Cards','param_card.dat_recomputed_widths')))
                shutil.copy(pjoin(MG5DIR,output_name%mode,'Cards',
                                     'param_card.dat_recomputed_widths'),
                        pjoin(MG5DIR,output_name%mode,'Cards','param_card.dat'))
            options['tweak']='allwidths->1.1*allwidths(widths_increased)'
            options['recompute_width']='never'
            cmd = command+' '.join('--%s=%s'%(opt, value) for opt, value in 
                                                                options.items())
            # print "Running second CMS check cmd: ",cmd
            self.do(cmd)
            self.assertEqual(cwd, os.getcwd())
            self.assertTrue(path.isfile(pjoin(MG5DIR,
                           'acceptance_test_aem_emvevex_widths_increased.pkl')))
            self.assertTrue(path.isfile(self.tmp_path['madgraph.check_cmd']))
            res = open(self.tmp_path['madgraph.check_cmd']).read()
            self.assertEqual(res.count('=== FAILED ==='), 2)
            self.assertEqual(res.count('=== PASSED ==='), 0)
            self.assertTrue(path.isfile(pjoin(MG5DIR,
                           'acceptance_test_aem_emvevex_widths_increased.log')))
            res = open(pjoin(MG5DIR,
                     'acceptance_test_aem_emvevex_widths_increased.log')).read()
            self.assertEqual(res.count('=== FAILED ==='), 2)
            self.assertEqual(res.count('=== PASSED ==='), 0)
        
            # Clean up duties
            for mode in ['NWA','CMS']:
                shutil.rmtree(pjoin(MG5DIR,output_name%mode))
            for file in files:
                try:
                    os.remove(pjoin(MG5DIR,file))
                except:
                    pass
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)

        except KeyError as e:
            self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)
            for mode in ['NWA','CMS']:
                try:
                    shutil.rmtree(pjoin(MG5DIR,output_name%mode))
                except:
                    pass
            for f in files:
                try:
                    os.remove(pjoin(MG5DIR,f))
                except:
                    pass
            raise e
        self.setup_logFile_for_logger('madgraph.check_cmd',restore=True)


    def test_density_mode_LI_validation(self):
        """
        This tests checks the convolution relation between the full matrix element
        and the density matrices of the production and the decay:
        M²_full = 3 · Σ_{λλ'} rho_prod(λ,λ')·rho_dec(λ,λ') / [(Q²-M_Z²)² + M_Z²Γ_Z²]
        The process is g g > (z > e+ e-) g, where the z is offshell.
        To check this relation we compute 5 events of g g > z* g that we decay with madspin,
        we then use the standalone mode to compute the density matrices of the decay and the production
        as well as the matrix element of the full process.
        This serves as a verification of the computation of density matrices at loop-induced level
        """

        import madgraph.various.Density_functions as dens
        rho_decay, rho_production = [], []
        matrix_elem_decay, matrix_elem_prod = [], []
        p2_decay, p2_production = [], []
        gammaz_decay, gammaz_production = 2.441404e+00, 2.441404e+00 #they are read from the param_card
        matrix_element_prod_and_decay = []

        ##First part, we need the matrix element of the full process g g > g e+ e-.
        # We will use the momenta and alphas/mu_r to generate the density matrices in standalone
        path_input = pjoin(MG5DIR, 'tests', 'input_files', 'madspin', 'unweighted_events_gg_epemg.lhe.gz')
        lhe = lhe_parser.EventFile(path_input)
        for event in lhe:
            momenta_full = []
            alphas = event.aqcd
            mu_r = event.scale
            # for each event we store the momenta of g g > z g into momenta_production
            momenta_full.append([event[0].E, event[0].px, event[0].py, event[0].pz]) # g
            momenta_full.append([event[1].E, event[1].px, event[1].py, event[1].pz]) # g
            momenta_full.append([event[2].E, event[2].px, event[2].py, event[2].pz]) # z
            momenta_full.append([event[3].E, event[3].px, event[3].py, event[3].pz]) # g
            momenta_full.append([event[4].E, event[4].px, event[4].py, event[4].pz]) # e+
            momenta_full.append([event[5].E, event[5].px, event[5].py, event[5].pz]) # e-

            if os.path.isdir(self.out_dir):
                shutil.rmtree(self.out_dir)
            self.do('import model loop_sm')
            self.do('generate g g > e+ e- g / a [sqrvirt=QCD]') ## Photon is removed to have only the z propagator
            self.run_cmd(f'output standalone_fortran {self.out_dir} -f')

            path_PS_card = pjoin(self.out_dir, "SubProcesses/P0_gg_epemg_no_a/PS.input")
            with open(path_PS_card, 'w') as psinput:
                psinput.write(str(momenta_full[0]).strip("[],") + "\n") #g
                psinput.write(str(momenta_full[1]).strip("[],") + "\n") #g
                psinput.write(str(momenta_full[4]).strip("[],") + "\n") #e+
                psinput.write(str(momenta_full[5]).strip("[],") + "\n") #e-
                psinput.write(str(momenta_full[3]).strip("[],") + "\n") #g

            # We cannot do output and launch at the same time because we need to modify PS.input beforehand
            text = f""" launch {self.out_dir}
            set param_card mu_r {mu_r}
            set param_card as {alphas}
            """

            command_card = open(f'{self.out_dir}/mg5_cmd_full.txt','w')
            command_card.write(text)
            command_card.close()

            logfile = 'test_madspin_full.log'
            subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'),
                            f'{self.out_dir}/mg5_cmd_full.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)

            with open(pjoin(self.out_dir, "SubProcesses/P0_gg_epemg_no_a/result.dat"), "r") as result:
                for line in result:
                    if line.strip()[:3] == 'FIN':
                        fin = float(line[3:].strip())
                    if line.strip()[:4] == '1EPS':
                        eps1 = float(line[4:].strip())
                    if line.strip()[:4] == '2EPS':
                        eps2 = float(line[4:].strip())

                matrix_element_prod_and_decay.append(fin + eps1 + eps2) #matrix element of the full process


            ##We now have the momenta for all the particles (including the offshell z) as well as alphas and mu_r
            # We can then compute the density matrices for the production part and the decay part.

            ## Production part, we run the standalone reweight module at loop-induced for the production g g > z* g [sqrvirt=QCD] to compute the production density matrix.
            momenta_production = [momenta_full[0], momenta_full[1], momenta_full[2], momenta_full[3]] # g g > z g

            p2_production.append(momenta_production[2][0]**2 - momenta_production[2][1]**2 - momenta_production[2][2]**2 - momenta_production[2][3]**2)

            # now that we have the momenta of the event, let us compute the production density matrix with the standalone mode
            if os.path.isdir(self.out_dir):
                shutil.rmtree(self.out_dir)
            self.do('import model loop_sm')
            self.do('generate g g > z* g  [sqrvirt=QCD]')
            self.run_cmd(f'output standalone_fortran {self.out_dir} --density=3 -f')

            path_PS_card = pjoin(self.out_dir, "SubProcesses/P0_gg_zg/PS.input")
            with open(path_PS_card, 'w') as psinput:
                psinput.write(str(momenta_production[0]).strip("[],") + "\n")
                psinput.write(str(momenta_production[1]).strip("[],") + "\n")
                psinput.write(str(momenta_production[2]).strip("[],") + "\n")
                psinput.write(str(momenta_production[3]).strip("[],") + "\n")

            # We cannot do output and launch at the same time because we need to modify PS.input beforehand
            text = f""" launch {self.out_dir}
            set param_card mu_r {mu_r}
            set param_card as {alphas}
            """

            command_card = open(f'{self.out_dir}/mg5_cmd_prod.txt','w')
            command_card.write(text)
            command_card.close()

            logfile = 'test_madspin_convolution.log'
            subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'),
                            f'{self.out_dir}/mg5_cmd_prod.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)


            with open(pjoin(self.out_dir, "SubProcesses/P0_gg_zg/result.dat"), "r") as result:
                for line in result:
                    if line.strip()[:3] == 'RHO':
                        rho_production_str = line.strip()[3:].strip()
            #The density matrix is written in the basis [-1, 0, +1] (the default setting)

            rho_production_str = rho_production_str.split()
            rho_production_float = [0.]*6
            for i in range(len(rho_production_str)):
                aux = rho_production_str[i].strip("()").split(",")
                rho_production_float[i] = float(aux[0]) + float(aux[1])*1j
            density_prod = dens.DensityMatrixObservables(rho_production_float)
            rho_production.append(density_prod.square_matrix())
            matrix_elem_prod.append(density_prod.get_trace())


            ### Decay part, we reweight the input_file of the decay z > e+ e- to get the density matrix of the decay
            momenta_decay = [momenta_full[2], momenta_full[4], momenta_full[5]] # z > e+ e-
            p2_decay.append(momenta_decay[0][0]**2 - momenta_decay[0][1]**2 - momenta_decay[0][2]**2 - momenta_decay[0][3]**2)

            if os.path.isdir(self.out_dir):
                shutil.rmtree(self.out_dir)

            self.do('generate z* > e+ e-')
            self.run_cmd(f'output standalone_fortran {self.out_dir} --density=1 -f')
            path_PS_card = pjoin(self.out_dir, "SubProcesses/P0_z_epem/PS.input")
            with open(path_PS_card, 'w') as psinput:
                psinput.write(str(momenta_decay[0]).strip("[],") + "\n")
                psinput.write(str(momenta_decay[1]).strip("[],") + "\n")
                psinput.write(str(momenta_decay[2]).strip("[],") + "\n")

            # We cannot do output and launch at the same time because we need to modify PS.input beforehand
            text = f""" launch {self.out_dir}
            set param_card as {alphas}
            """

            command_card = open(f'{self.out_dir}/mg5_cmd_decay.txt','w')
            command_card.write(text)
            command_card.close()

            logfile = 'test_madspin_convolution.log'
            subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'),
                            f'{self.out_dir}/mg5_cmd_decay.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)

            density_card_path = pjoin(self.out_dir, "SubProcesses/P0_z_epem/Density_matrix.dat")

            with open(density_card_path, "r") as result:
                density_matrix = []
                rho_text = result.readlines()[1].split()
                for i in range(len(rho_text)):
                    aux = rho_text[i].strip("()").split(",")
                    density_matrix.append(float(aux[0]) + float(aux[1])*1j)
            #The density matrix is written in the basis [-1, 0, +1] (the default setting)

            density_decay = dens.DensityMatrixObservables(density_matrix)
            rho_decay.append(density_decay.square_matrix())
            matrix_elem_decay.append(density_decay.get_trace())


        ###From here we have the density matrix of the production, the density matrix of the decay and the full matrix element, we have everything that we need

        def convolution(rho1, rho2):
            conv = 0.
            for i in range(len(rho1)):
                for j in range(len(rho1[0])):
                    conv += rho1[i][j] * rho2[i][j]
            return conv

        def propagator(Q2, mz, gammaz):
            return 1/((Q2 - mz**2)**2 + mz**2 * gammaz**2)

    #We check that the mass, the decay and the transferred momentum of the z are the same in the decay and the production for each of the 5 events
        for i in range(5):
            self.assertAlmostEqual(p2_decay[i], p2_production[i], places=7)
            comparison = 3 * propagator(p2_decay[i], 9.118800e+01, gammaz_decay) * convolution(rho_production[i], rho_decay[i])
            self.assertAlmostEqual(comparison.imag, 0., places=12)
            self.assertAlmostEqual(matrix_element_prod_and_decay[i], comparison.real)


    def test_standalone_rw_tree_then_virt(self):
        """The calls of a tree standalone_rw output (fk_ tag of the widths)
        must not leak into the next MadLoop output (NLO reweighting)"""

        self.do('import model loop_sm')
        self.do('generate u u~ > e+ e-')
        self.do('output standalone_rw %s -f' % pjoin(self.tmpdir, 'TREE'))
        self.do('generate u u~ > e+ e- [virt=QCD]')
        self.do('output standalone_rw %s -f' % pjoin(self.tmpdir, 'VIRT'))

        born = open(pjoin(self.tmpdir, 'VIRT', 'SubProcesses', 'P0_uux_epem',
                          'born_matrix.f')).read()
        self.assertEqual([l.strip() for l in born.split('\n') if 'FK_' in l.upper()], [])
        self.assertIn('MDL_MZ,MDL_WZ,', born)

    def test_density_mode_loop_induced_standalone1(self):
        """ Testing the density mode in standalone mode for loop induced.
            Process: g g > h [sqrvirt=QCD]
            We check the value of the non-normalised density matrix, its trace and the matrix element computed independently from the density matrix.
        """

        short_path = '/tmp/test1' #we use this path because there is a problem when using paths that are too long

        if os.path.isdir(short_path):
            shutil.rmtree(short_path)

        self.do('import model loop_sm')
        self.do('generate g g > h  [sqrvirt=QCD]')
        self.run_cmd(f'output standalone_fortran {short_path} --density=1,2 -f') # we need run_cmd here, else HelicityFilterLevel is not set to 1.
        self.run_cmd(f'launch {short_path} -f ')

        devnull = open(os.devnull,'w')    
        logfile = os.path.join(short_path,'SubProcesses', 'P0_gg_h',
                            'logfile_test.log')

        # the result of the run is stored inside the file result.dat
        # we want to check 2 things: 1) the density matrix, 2) the matrix element

        result_string = open(os.path.join(short_path, 'SubProcesses/P0_gg_h/result.dat'), 'r').readlines()

        for line in result_string:
            if line[0:4] == 'BORN':
                born_part = float(line.strip('BORN '))
            if line[0:3] == 'FIN':
                finite_part = float(line.strip('FIN '))
            if line[0:4] == '1EPS':
                eps1_part = float(line.strip('1EPS '))
            if line[0:4] == '2EPS':
                eps2_part = float(line.strip('2EPS '))
            if line[0:4].strip() == 'RHO':
                rho = line.strip('RHO ').split()
            
        for i in range(len(rho)):
            aux = rho[i].strip('()').split(",")
            rho[i] = complex(float(aux[0]), float(aux[1]))


        #we have here the 2 element we wanted, we now can compare them to reference values
        reference_born = 0.000000000000000E+000
        reference_fin = 9.370261322546434E-003
        reference_1eps = 1.290803679293600E-016
        reference_2eps = 8.325783139603451E-031
        reference_rho = [4.68513066127328095E-003, 0, 0, 4.68513066127328095E-003, 0, 0, 0, 0, 0, 4.68513066127328095E-003]

        self.assertAlmostEqual(reference_born, born_part, places=7)
        self.assertAlmostEqual(reference_fin, finite_part, places=7)
        self.assertAlmostEqual(reference_fin, rho[0] + rho[4] + rho[7] + rho[9], places=7) #the trace of the non-normalised density matrix must be equal to the matrix element
        self.assertTrue(eps1_part < 1e-10)
        self.assertTrue(eps2_part < 1e-10)
        for i in range(len(rho)):
            self.assertAlmostEqual(reference_rho[i], rho[i], places=7)


    def test_density_mode_loop_induced_standalone2(self):
        """ Testing the density mode in standalone mode for loop induced.
            Process: g g > w+ w- [sqrvirt=QCD]
            We check the value of the non-normalised density matrix, its trace and the matrix element computed independently from the density matrix.
        """

        short_path = '/tmp/test2' #we use this path because there is a problem when using paths that are too long

        if os.path.isdir(short_path):
            shutil.rmtree(short_path)

        self.do('import model loop_sm')
        self.do('generate g g > w+ w-  [sqrvirt=QCD]')
        self.run_cmd(f'output standalone_fortran {short_path} --density=3,4 -f') # we need run_cmd here, else HelicityFilterLevel is not set to 1.
        self.run_cmd(f'launch {short_path} -f ')

        devnull = open(os.devnull,'w')    
        logfile = os.path.join(short_path,'SubProcesses', 'P0_gg_wpwm',
                            'logfile_test.log')

        # the result of the run is stored inside the file result.dat
        # we want to check 2 things: 1) the density matrix, 2) the matrix element

        result_string = open(os.path.join(short_path, 'SubProcesses/P0_gg_wpwm/result.dat'), 'r').readlines()

        for line in result_string:
            if line[0:4] == 'BORN':
                born_part = float(line.strip('BORN '))
            if line[0:3] == 'FIN':
                finite_part = float(line.strip('FIN '))
            if line[0:4] == '1EPS':
                eps1_part = float(line.strip('1EPS '))
            if line[0:4] == '2EPS':
                eps2_part = float(line.strip('2EPS '))
            if line[0:4].strip() == 'RHO':
                rho = line.strip('RHO ').split()
            
        for i in range(len(rho)):
            aux = rho[i].strip('()').split(",")
            rho[i] = complex(float(aux[0]), float(aux[1]))


        #we have here the 2 element we wanted, we now can compare them to reference values
        reference_born = 0.000000000000000E+000
        reference_fin = 3.105375524420539E-004
        reference_1eps = 1.223802616982397E-017
        reference_2eps = 3.557950409013374E-019
        reference_rho_string = "(3.73213795707112307E-005,0.0000000000000000)  (-4.51413968742457291E-006,5.57674567023938250E-008)  (-1.14754836983286585E-005,1.23142567879820382E-005)  (-4.51413968742060710E-006,5.57674567064153786E-008)  (2.06647046233477647E-005,-2.39083683336853825E-006) (-6.97740942991315351E-007,-7.57233746251466554E-007)  (-1.14754836983295733E-005,1.23142567879801967E-005) (-6.97740942982686944E-007,-7.57233746271897306E-007)  (2.23922051764003669E-005,-5.33926435228035830E-007)         (2.21435090197597551E-006,0.0000000000000000)  (-4.93140434730487375E-007,4.96826885248896461E-006)   (1.67472536204813680E-006,7.94712119149096233E-019)  (-8.88853697963469205E-007,2.49830646735896084E-006)   (1.59956801378797867E-006,1.03677923085064894E-006) (-1.88857382800311938E-008,-2.45870015865194242E-006)   (1.59964460661341568E-006,9.27354326758565039E-007)  (2.01481289752838196E-006,-2.15157374088450262E-006)         (7.54389808529480892E-005,0.0000000000000000)  (-1.88857382804387159E-008,2.45870015865508660E-006)  (1.83273643633384266E-005,-1.54589035570191527E-005)  (2.48373343448314292E-006,-1.61678488066185346E-006) (-2.71966101483490324E-005,-1.70121273500105884E-018)  (9.77630338665281179E-007,-1.52166999389930935E-006) (-1.18203828299276593E-005,-1.19487829367985977E-005)         (2.21435090197053714E-006,0.0000000000000000)  (-8.88853697991626592E-007,2.49830646736104158E-006)   (1.59964460660996677E-006,9.27354326753081136E-007)  (-4.93140434740817471E-007,4.96826885249362330E-006)   (1.59956801378213224E-006,1.03677923084800916E-006)  (2.01481289751487348E-006,-2.15157374088780308E-006)         (7.58057935657595967E-005,0.0000000000000000)   (6.64375005056783532E-006,3.78231686136833000E-006)   (1.83273643633825638E-005,1.54589035569656372E-005)   (6.64375005057648522E-006,3.78231686133695844E-006)  (2.64034458170458516E-005,-2.25528548096681493E-006)         (3.03895638444625953E-006,0.0000000000000000)   (9.77630338683339709E-007,1.52166999389573869E-006)  (3.01676726510797787E-006,-2.18442931497799867E-018)  (4.74785584968493724E-006,-5.62829196758272748E-006)         (7.54389808529727277E-005,0.0000000000000000)  (2.48373343447933635E-006,-1.61678488066374806E-006) (-1.18203828299307730E-005,-1.19487829367974152E-005)         (3.03895638444237122E-006,0.0000000000000000)  (4.74785584969054629E-006,-5.62829196756257233E-006)         (3.60258030262078615E-005,0.0000000000000000)"
        reference_rho_string = reference_rho_string.split()
        reference_rho = []
        for elem in reference_rho_string:
            aux = elem.strip('()').split(",")
            reference_rho.append(complex(float(aux[0]), float(aux[1])))
        
        self.assertAlmostEqual(reference_born, born_part, places=7)
        self.assertAlmostEqual(reference_fin, finite_part, places=7)
        self.assertAlmostEqual(reference_fin, rho[0] + rho[9] + rho[17] + rho[24] + rho[30] + rho[35] + rho[39] + rho[42] + rho[44], places=7) #the trace of the non-normalised density matrix must be equal to the matrix element
        self.assertTrue(eps1_part < 1e-10)
        self.assertTrue(eps2_part < 1e-10)
        for i in range(len(rho)):
            self.assertAlmostEqual(reference_rho[i].real, rho[i].real, places=7)
            self.assertAlmostEqual(reference_rho[i].imag, rho[i].imag, places=7)



    def test_density_mode_loop_induced_standalone3(self):
        """ Testing the density mode in standalone mode for loop induced.
            Process: p p > h j [sqrvirt=QCD]
            We check the value of the non-normalised density matrix, its trace and the matrix element computed independently from the density matrix.
        """

        short_path = '/tmp/test3' #we use this path because there is a problem when using paths that are too long

        if os.path.isdir(short_path):
            shutil.rmtree(short_path)

        self.do('import model loop_sm')
        self.do('generate p p > h j  [sqrvirt=QCD]')
        self.run_cmd(f'output standalone_fortran {short_path} --density=2,4 -f') # we need run_cmd here, else HelicityFilterLevel is not set to 1.
        self.run_cmd(f'launch {short_path} -f ')

        # the result of the run is stored inside the file result.dat
        # we want to check 2 things: 1) the density matrix, 2) the matrix element

        result_string_gu = open(os.path.join(short_path, 'SubProcesses/P0_gu_hu/result.dat'), 'r').readlines()
        reference_gu = {'born': 0, 'fin': 4.911079822136591E-005, '1eps': 4.652514256760648E-020, '2eps': 1.889084601044776E-034,
                        'rho': [0, 0, 0 ,0, 2.45553991106829770E-005, -1.17925624055059263E-005 -6.27120991655205813E-006j, 0, 2.45553991106829872E-005, 0, 0]}
        result_string_gux = open(os.path.join(short_path, 'SubProcesses/P2_gux_hux/result.dat'), 'r').readlines()
        reference_gux = {'born': 0, 'fin': 4.911079822136591E-005, '1eps': 4.652514256760648E-020, '2eps': 1.889084601044776E-034,
                        'rho': [0, 0, 0 ,0, 2.45553991106829770E-005, -1.17925624055059263E-005 +6.27120991655205813E-006j, 0, 2.45553991106829872E-005, 0, 0]}
        result_string_uux = open(os.path.join(short_path, 'SubProcesses/P3_uux_hg/result.dat'), 'r').readlines()
        reference_uux = {'born': 0, 'fin': 7.679521988380547e-05, '1eps': 1.787557563159388E-019, '2eps': 1.117722363316526E-033,
                        'rho':   [(5.979704672209291e-06+0j), (1.392298457962328e-05+5.270427227360091e-21j), 0j, 0j, (3.241790526969405e-05+0j), 0j, 0j, (3.2417905269694064e-05+0j), (1.3922984579623275e-05+7.529181753371558e-22j), (5.97970467220928e-06+0j)]}
        result_string_gg = open(os.path.join(short_path, 'SubProcesses/P4_gg_hg/result.dat'), 'r').readlines()
        reference_gg = {'born': 0, 'fin': 1.1959960400593030e-03, '1eps': 1.672814917871989E-017, '2eps': 5.111506851363750E-031,
                        'rho':  [(0.00046955799704735543+0j), (-3.402074849160713e-05-1.592561249559833e-05j), (0.00023457059095329064-7.124241678531108e-05j), (-8.024579382478496e-05+4.267420435787805e-05j), (0.0001284400229822933+1.5881867761018131e-21j), (-1.1655765706238232e-05+6.198462299240582e-06j), (0.000190212542058378-0.0001546545600144973j), (0.00012844002298229327+2.329340604949326e-21j), (-3.402074849160712e-05+1.59256124955982e-05j), (0.0004695579970473552-7.058607893785836e-22j)]}
        


        #for gu
        for line in result_string_gu:
            if line[0:4] == 'BORN':
                born_part = float(line.strip('BORN '))
            if line[0:3] == 'FIN':
                finite_part = float(line.strip('FIN '))
            if line[0:4] == '1EPS':
                eps1_part = float(line.strip('1EPS '))
            if line[0:4] == '2EPS':
                eps2_part = float(line.strip('2EPS '))
            if line[0:4].strip() == 'RHO':
                rho = line.strip('RHO ').split()
            
        for i in range(len(rho)):
            aux = rho[i].strip('()').split(",")
            rho[i] = complex(float(aux[0]), float(aux[1]))


        self.assertAlmostEqual(reference_gu['born'], born_part, places=7)
        self.assertAlmostEqual(reference_gu['fin'], finite_part, places=7)
        self.assertAlmostEqual(reference_gu['fin'], rho[0] + rho[4] + rho[7] + rho[9], places=7) #the trace of the non-normalised density matrix must be equal to the matrix element
        self.assertTrue(eps1_part < 1e-10)
        self.assertTrue(eps2_part < 1e-10)

        for i in range(len(rho)):
            self.assertAlmostEqual(reference_gu['rho'][i].real, rho[i].real, places=7)
            self.assertAlmostEqual(reference_gu['rho'][i].imag, rho[i].imag, places=7)
        

        #for gux
        for line in result_string_gux:
            if line[0:4] == 'BORN':
                born_part = float(line.strip('BORN '))
            if line[0:3] == 'FIN':
                finite_part = float(line.strip('FIN '))
            if line[0:4] == '1EPS':
                eps1_part = float(line.strip('1EPS '))
            if line[0:4] == '2EPS':
                eps2_part = float(line.strip('2EPS '))
            if line[0:4].strip() == 'RHO':
                rho = line.strip('RHO ').split()
            
        for i in range(len(rho)):
            aux = rho[i].strip('()').split(",")
            rho[i] = complex(float(aux[0]), float(aux[1]))


        self.assertAlmostEqual(reference_gux['born'], born_part, places=7)
        self.assertAlmostEqual(reference_gux['fin'], finite_part, places=7)
        self.assertAlmostEqual(reference_gux['fin'], rho[0] + rho[4] + rho[7] + rho[9], places=7) #the trace of the non-normalised density matrix must be equal to the matrix element
        self.assertTrue(eps1_part < 1e-10)
        self.assertTrue(eps2_part < 1e-10)
        for i in range(len(rho)):
            self.assertAlmostEqual(reference_gux['rho'][i].real, rho[i].real, places=7)
            self.assertAlmostEqual(reference_gux['rho'][i].imag, rho[i].imag, places=7)

        #for uux
        for line in result_string_uux:
            if line[0:4] == 'BORN':
                born_part = float(line.strip('BORN '))
            if line[0:3] == 'FIN':
                finite_part = float(line.strip('FIN '))
            if line[0:4] == '1EPS':
                eps1_part = float(line.strip('1EPS '))
            if line[0:4] == '2EPS':
                eps2_part = float(line.strip('2EPS '))
            if line[0:4].strip() == 'RHO':
                rho = line.strip('RHO ').split()
        
        
        for i in range(len(rho)):
            aux = rho[i].strip('()').split(",")
            rho[i] = complex(float(aux[0]), float(aux[1]))
       

        self.assertAlmostEqual(reference_uux['born'], born_part, places=7)
        self.assertAlmostEqual(reference_uux['fin'], finite_part, places=7)
        self.assertAlmostEqual(reference_uux['fin'], rho[0] + rho[4] + rho[7] + rho[9], places=7) #the trace of the non-normalised density matrix must be equal to the matrix element
        self.assertTrue(eps1_part < 1e-10)
        self.assertTrue(eps2_part < 1e-10)
        for i in range(len(rho)):
            self.assertAlmostEqual(reference_uux['rho'][i].real, rho[i].real, places=7)
            self.assertAlmostEqual(reference_uux['rho'][i].imag, rho[i].imag, places=7)

        #for gg
        for line in result_string_gg:
            if line[0:4] == 'BORN':
                born_part = float(line.strip('BORN '))
            if line[0:3] == 'FIN':
                finite_part = float(line.strip('FIN '))
            if line[0:4] == '1EPS':
                eps1_part = float(line.strip('1EPS '))
            if line[0:4] == '2EPS':
                eps2_part = float(line.strip('2EPS '))
            if line[0:4].strip() == 'RHO':
                rho = line.strip('RHO ').split()
            
        for i in range(len(rho)):
            aux = rho[i].strip('()').split(",")
            rho[i] = complex(float(aux[0]), float(aux[1]))

        self.assertAlmostEqual(reference_gg['born'], born_part, places=7)
        self.assertAlmostEqual(reference_gg['fin'], finite_part, places=7)
        self.assertAlmostEqual(reference_gg['fin'], rho[0] + rho[4] + rho[7] + rho[9], places=7) #the trace of the non-normalised density matrix must be equal to the matrix element
        self.assertTrue(eps1_part < 1e-10)
        self.assertTrue(eps2_part < 1e-10)
        for i in range(len(rho)):
            self.assertAlmostEqual(reference_gg['rho'][i].real, rho[i].real, places=7)
            self.assertAlmostEqual(reference_gg['rho'][i].imag, rho[i].imag, places=7)

    def test_density_mode_vs_standalone_LI1(self):
        """ Comparing the value of density loop-induced to the standalone version.
            Process: g g > w+ w- [sqrvirt=QCD]
            We generate a single event from the python interface and use the value of alpha_s, mu_r and p to feed to standalone code. 
            We compare the non-normalised density matrices.
        """

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        text = f""" import model loop_sm
                    generate g g > w+ w- [noborn=QCD]
                    output madevent {self.out_dir}
                    launch
                    set run_card nevents 1
                    set run_card use_syst False
                """

        #This block of code launches MadGraph with the commands written in mg5_cmd.txt
        command_card = open('/tmp/mg5_cmd.txt','w')
        command_card.write(text)
        command_card.close()

        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                        '/tmp/mg5_cmd.txt'])
        

        #Here we replace the lhe file by the reference lhe file (stored in the input_files).
        os.remove(f"{self.out_dir}/Events/run_01/unweighted_events.lhe.gz")
        shutil.copyfile(pjoin(MG5DIR, "tests/input_files/density_mode/test_density_mode_LIvsSA.lhe.gz"), f"{self.out_dir}/Events/run_01/unweighted_events.lhe.gz")

        #Now we reweight the lhe file through the inline method
        text_rwgt = f"""launch {self.out_dir} -i
reweight run_01 --mode=density
set reweight_card particle_in_density_matrix [24, -24]
set reweight_card order_helicities [-1, 1, -1, 0, -1, -1, 0, 1, 0, 0, 0, -1, 1, 1, 1, 0, 1, -1]
set matrix_normalisation False
"""
        #This block of code launches MadGraph with the commands written in mg5_cmd_rwgt.txt
        command_card_rwgt = open('/tmp/mg5_cmd_rwgt.txt','w')
        command_card_rwgt.write(text_rwgt)
        command_card_rwgt.close()

        logfile = 'test_density_mode_LIvsSA.log'
        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                         '/tmp/mg5_cmd_rwgt.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)

        # We read the reweighted event with the density matrix computed with the python interface
        lhe_path = pjoin(self.out_dir, "Events/run_01/unweighted_events.lhe.gz")
        p_all = []
        for event in lhe_parser.EventFile(lhe_path):
            density_check = event.density
            alphas = event.aqcd
            mu_r = event.scale
            for particle in event:
                p_all.append([particle.E, particle.px, particle.py, particle.pz])
        
        
        #Now we want to compute this exact same density matrix with the standalone mode. We clean the directory and do it in here again
        shutil.rmtree(self.out_dir)
        
        self.do('import model loop_sm')
        self.do('generate g g > w+ w-  [sqrvirt=QCD]')
        self.run_cmd(f'output standalone_fortran {self.out_dir} --density=3,4 -f')
        
        path_PS_card = pjoin(self.out_dir, "SubProcesses/P0_gg_wpwm/PS.input")
        with open(path_PS_card, 'w') as psinput:
            psinput.write(str(p_all[0]).strip("[],") + "\n")
            psinput.write(str(p_all[1]).strip("[],") + "\n")
            psinput.write(str(p_all[2]).strip("[],") + "\n")
            psinput.write(str(p_all[3]).strip("[],") + "\n")
        
        # We cannot do output and launch at the same time because we need to modify PS.input beforehand
        text_bis = f""" launch {self.out_dir}
        set param_card mu_r {mu_r}
        set param_card as {alphas}
        """
        
        command_card_bis = open('/tmp/mg5_cmd_bis.txt','w')
        command_card_bis.write(text_bis)
        command_card_bis.close()

        logfile = 'test_density_vs_LI_standalone.log'
        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                        '/tmp/mg5_cmd_bis.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)
        

        # # the two are identical, make the comparison
        with open(pjoin(self.out_dir, "SubProcesses/P0_gg_wpwm/result.dat"), "r") as result:
            for line in result:
                if line.strip()[:3] == 'RHO':
                    rho_standalone_str = line.strip()[3:].strip()
        
        rho_standalone = rho_standalone_str.split()
        for i in range(len(rho_standalone)):
            aux = rho_standalone[i].strip("()").split(",")
            rho_standalone[i] = float(aux[0]) + float(aux[1])*1j
        
        misc.sprint(rho_standalone)

        for j in range(45): # 45 to raise error if density_check is an empty array
            self.assertAlmostEqual(density_check[j].real, rho_standalone[j].real, places=7)
            self.assertAlmostEqual(density_check[j].imag, rho_standalone[j].imag, places=7)


class TestCmdMatchBox(IOTests.IOTestManager):
    
    def setUp(self):
        """ Initialize the test """

        self.interface = MGCmd.MasterCmd()
        self.interface.no_notification()
        # Below the key is the name of the logger and the value is a tuple with
        # first the handlers and second the level.
        self.logger_saved_info = {}

        # Select the Tensor Integral to include in the test
        misc.deactivate_dependence('pjfry', cmd = self.interface, log='stdout')
        misc.deactivate_dependence('samurai', cmd = self.interface, log='stdout')        
        misc.deactivate_dependence('golem', cmd = self.interface, log='stdout')
        misc.activate_dependence('ninja', cmd = self.interface, log='stdout',MG5dir=MG5DIR)
        misc.activate_dependence('collier', cmd = self.interface, log='stdout',MG5dir=MG5DIR)

    @IOTests.createIOTest()
    def testIO_MatchBoxOutput(self):
        r""" target: TEST/SubProcesses/P1_uux_uux/[.+\.(inc|f)]
            target: TEST/SubProcesses/P0_wpwm_wpwm/[.+\.(inc|f)]"""
        
        cmd = """
        set apply_flavor_grouping False
        import model sm
        generate w+ w- > w+ w- @0
        output matchbox %(path)s/TEST --postpone_model
        generate u u~ > u u~ [virt=QCD] @1
        output matchbox %(path)s/TEST -f
        """ % {'path': self.IOpath}
        
        for line in cmd.split('\n'):
            self.interface.exec_cmd(line)
 
    
#===============================================================================
# IOTestMadLoopOutputFromInterface
#===============================================================================
class IOTestMadLoopOutputFromInterface(IOTests.IOTestManager):
    """Test MadLoop outputs when generated directly from the interface."""

    @IOTests.createIOTest(groupName='MadLoop_output_from_the_interface')
    def testIO_TIR_output(self):
        r""" target: [ggttx_IOTest/SubProcesses/(.*)\.f]
        """
        interface = MGCmd.MasterCmd()
        interface.no_notification()

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)
        
        # Select the Tensor Integral to include in the test
        misc.deactivate_dependence('pjfry', cmd = interface, log='stdout')
        misc.deactivate_dependence('samurai', cmd = interface, log='stdout')        
        misc.deactivate_dependence('golem', cmd = interface, log='stdout')
        misc.activate_dependence('ninja', cmd = interface, log='stdout',MG5dir=MG5DIR)

        # The reference includes COLLIER interfaces on every runner.
        misc.activate_dependence('collier', cmd = interface, log='stdout',MG5dir=MG5DIR)

        run_cmd('generate g g > t t~ [virt=QCD]')
        interface.onecmd('output %s -f' % str(pjoin(self.IOpath,'ggttx_IOTest')))

        #remove some function from some file:
        IOTests.IOTest.remove_f77_function_from_file(
                    pjoin(self.IOpath,'ggttx_IOTest', 'SubProcesses','MadLoopCommons.f'),
                    'PRINT_MADLOOP_BANNER')

    @IOTests.createIOTest(groupName='MadLoop_output_from_the_interface')
    def testIO_loop_induced_standalone_output(self):
        r""" target: gghLI_IOTest/SubProcesses/P0_gg_h/[(check_sa|loop_matrix)\.f]
        """
        # A loop-induced ([noborn=]) process is exported from the MadGraph
        # interface, which used to hand the loop matrix element to the
        # tree-level standalone exporter and crash.
        interface = MGCmd.MasterCmd()
        interface.no_notification()

        # Select the Tensor Integral to include in the test
        misc.deactivate_dependence('pjfry', cmd = interface, log='stdout')
        misc.deactivate_dependence('samurai', cmd = interface, log='stdout')
        misc.deactivate_dependence('golem', cmd = interface, log='stdout')
        misc.activate_dependence('ninja', cmd = interface, log='stdout',MG5dir=MG5DIR)

        # no 'import model': validate_model must bootstrap sm -> loop_sm itself
        interface.exec_cmd('generate g g > h [noborn=QCD]', errorhandling=False,
                           printcmd=False, precmd=True, postcmd=True)
        interface.onecmd('output standalone_fortran %s -f' %
                                    str(pjoin(self.IOpath,'gghLI_IOTest')))
        




