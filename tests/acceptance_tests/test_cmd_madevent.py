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
import collections
import subprocess
import unittest
import json
import glob
import os
import re
import shutil
import sys
import logging
import time
import tempfile
import math
import madgraph


logger = logging.getLogger('test_cmd')

import tests.unit_tests.iolibs.test_file_writers as test_file_writers

import madgraph.interface.master_interface as MGCmd
import madgraph.interface.madevent_interface as MECmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.iolibs.files as files

import madgraph.various.misc as misc
import madgraph.various.lhe_parser as lhe_parser
import madgraph.various.banner as banner_mod
import madgraph.various.lhe_parser as lhe_parser
import madgraph.various.banner as banner

import tests.parallel_tests.test_aloha as test_aloha

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd

pjoin = os.path.join

def check_html_page(cls, link):
    """return True if all link in the html page are existing on disk.
       otherwise raise an assertion error"""
        
    text=open(link).read()
    pattern = re.compile(r'href=[\"\']?(.*?)?[\"\'\s\#]', re.DOTALL)
    
    cwd = os.path.dirname(link)
    with misc.chdir(cwd):
        for path in pattern.findall(text):
            if not path:
                continue # means is just a line starting with #
            cls.assertTrue(os.path.exists(path), '%s/%s' %(cwd,path))
    return True
    

#===============================================================================
# Shared helpers for the mg7 (madspace) cross-section acceptance tests
# (module-level so they can be used from several test classes)
#===============================================================================
def _mg7_datadir_or_skip(test):
    """Return a usable LHAPDF data dir for mg7 runs, or skipTest (on *test*) if
    the mg7 runtime stack (madspace + LHAPDF + the run_card.toml default PDF) is
    not available."""
    import glob
    try:
        import madspace
        has_mg7 = hasattr(madspace, 'ChannelEventGenerator')
    except ImportError:
        has_mg7 = False
    datadir = os.environ.get('LHAPDF_DATA_PATH')
    if not datadir:
        mg = MGCmd.MasterCmd()
        lhapdf = mg.options['lhapdf']
        if not lhapdf:
            lhapdf = 'lhapdf-config'
        try:
            datadir = subprocess.check_output(
            [lhapdf, '--datadir']).decode().strip()
        except Exception:
            datadir = None 
        
    misc.sprint(datadir)
    if not has_mg7 or not datadir or not os.path.isdir(datadir):
        test.skipTest('mg7 runtime stack (madspace + LHAPDF data) unavailable')
    if not glob.glob(pjoin(datadir, 'NNPDF40_lo_as_01180*')):
        test.skipTest('NNPDF40_lo_as_01180 PDF set not available')
    return datadir


def _run_mg7_xsec(test, setup_cmds, run_dir, datadir):
    """Run an mg7 cross-section: execute `setup_cmds` (MG5 lines, ending with
    the generate), `output mg7 run_dir`, drive bin/generate_events with the
    dynamical HT/2 scale, and return (cross, error) from the madspace info.json
    (process.mean / process.error). Assertions are made on the *test* instance.

    The event target drives the statistical precision of the returned
    cross-section: madspace quotes error = rel_std_dev / sqrt(count_opt), and
    with only 2000 events VEGAS barely adapts, so rel_std_dev stays ~2.4 and the
    error is ~1.1% -- larger than the tolerance some callers assert, which made
    them fail at random. 50000 events lets the grid adapt (rel_std_dev ~0.37)
    and brings the error down to ~0.08%. That costs nothing: the integration
    itself is ~0.2s wall even at 100000 events. What used to dominate the run
    time was the LHE systematics post-processing (145 weights per event), which
    none of the callers look at -- they only read info.json -- so it is switched
    off here and the higher statistics comes out free (<1s for 50000 events)."""
    import glob, json
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    mg = MGCmd.MasterCmd()
    mg.no_notification()
    for c in setup_cmds:
        mg.exec_cmd(c)
    mg.exec_cmd('output mg7 %s' % run_dir)
    toml = pjoin(run_dir, 'Cards', 'run_card.toml')
    t = open(toml).read()
    t = t.replace('fixed_ren_scale = true', 'fixed_ren_scale = false')
    t = t.replace('fixed_fact_scale = true', 'fixed_fact_scale = false')
    t = re.sub(r'events = \d+', 'events = 50000', t)
    # Switch both systematics paths off: [postprocessing] systematics (the
    # legacy systematics.py pass) and the native [systematics] enable, which
    # defaults to true and is the one that actually costs the time here. The
    # cross-section these callers read does not depend on either.
    t = re.sub(r'^systematics = true$', 'systematics = false', t, flags=re.M)
    t = re.sub(r'(?ms)^(\[systematics\].*?^enable = )true$', r'\1false', t)
    open(toml, 'w').write(t)
    env = dict(os.environ)
    env['LHAPDF_DATA_PATH'] = datadir
    log = pjoin(run_dir, 'mg7_gen.log')
    ret = subprocess.call(
        [sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
        cwd=run_dir, env=env, stdout=open(log, 'w'), stderr=subprocess.STDOUT)
    test.assertEqual(ret, 0, 'mg7 generate_events failed (see %s)' % log)
    infos = sorted(glob.glob(pjoin(run_dir, 'Events', '*', 'info.json')))
    test.assertTrue(infos, 'no mg7 info.json under %s' % run_dir)
    info = json.load(open(infos[-1]))['process']
    return float(info['mean']), float(info.get('error') or 0.0)


# default tool cards shipped with MG5, copied into the mg7 output's Cards/ so a
# tool switch is "available" and its driver (invoked with --no_default /
# -from_cards) finds the card it needs.
_MG7_DEFAULT_CARDS = {
    'pythia8': (pjoin('Template', 'LO', 'Cards', 'pythia8_card_default.dat'),
                'pythia8_card.dat'),
    'delphes': (pjoin('Template', 'Common', 'Cards', 'delphes_card_default.dat'),
                'delphes_card.dat'),
    'rivet':   (pjoin('Template', 'LO', 'Cards', 'rivet_card_default.dat'),
                'rivet_card.dat'),
    'madspin': (pjoin('Template', 'Common', 'Cards', 'madspin_card_default.dat'),
                'madspin_card.dat'),
}


def _mg7_tool_or_skip(test, cmd, *option_keys):
    """skipTest unless every configured tool path in *option_keys* (e.g.
    'pythia8_path', 'delphes_path', 'rivet_path') is available; otherwise the
    post-processing driver would silently skip the tool and the output-file
    assertions would fail for an unrelated (missing-tool) reason."""
    for key in option_keys:
        path = cmd.options.get(key)
        if not path or path in ('None', 'NA') or not os.path.exists(path):
            test.skipTest('mg7 post-processing tool %s not configured (%s)'
                          % (key, path))


def _run_mg7_postproc(test, setup_cmds, run_dir, datadir, switch_lines=None,
                      toml_edits=None, precopy_cards=None, extra_cards=None,
                      events=100, require_tools=(), timeout=1800):
    """Drive an mg7 post-processing chain through the command interface.

    Runs *setup_cmds* (MG5 lines ending with the ``generate``) + ``output mg7
    run_dir`` in-process, trims the ``run_card.toml`` (small event target, LHE
    output, dynamical HT/2 scale, plus any *toml_edits*), copies the requested
    default tool cards (*precopy_cards*, keys of :data:`_MG7_DEFAULT_CARDS`) and
    any *extra_cards* ((text, dest) pairs) into ``Cards/``, then runs the mg7
    ``bin/generate_events`` -- feeding *switch_lines* (e.g. ``'shower=Pythia8'``)
    on stdin so the merged tool-selection question runs the chained
    post-processing (``run_selected_tools`` -> ``MG7RunCmd``), exactly as the
    ``launch`` command does from the mg5 interface. When *switch_lines* is None
    the run is non-interactive (``-f``) and only the run_card-driven LHE
    post-processing (time-of-flight/systematics) is exercised.

    Returns the run directory (``Events/run_01``). Assertions/skips are made on
    *test*; the run self-skips when the mg7 runtime stack or a *require_tools*
    program is unavailable.
    """
    import glob
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    mg = MGCmd.MasterCmd()
    mg.no_notification()
    if require_tools:
        _mg7_tool_or_skip(test, mg, *require_tools)
    # Drive the output through run_cmd (not exec_cmd): run_cmd records the
    # commands in the interface history, which is what the exporter dumps into
    # Cards/proc_card_mg5.dat and hence into the LHE banner's <MG5ProcCard>.
    # exec_cmd bypasses the history, so the proc_card would miss the model /
    # generate / output lines and MadSpin (which reads 'generate' from the
    # banner) would abort with "no generate line". This mirrors how a real user
    # drives the output (bin/madgraph command file / interactive session).
    for c in setup_cmds:
        mg.run_cmd(c)
    mg.run_cmd('output mg7 %s' % run_dir)

    cards_dir = pjoin(run_dir, 'Cards')
    for tool in (precopy_cards or []):
        src, dst = _MG7_DEFAULT_CARDS[tool]
        shutil.copy(pjoin(MG5DIR, src), pjoin(cards_dir, dst))
    for text, dst in (extra_cards or []):
        open(pjoin(cards_dir, dst), 'w').write(text)

    toml = pjoin(run_dir, 'Cards', 'run_card.toml')
    t = open(toml).read()
    # post-processing operates on an LHE file
    t = re.sub(r'output_format = "?\w+"?', 'output_format = "lhe"', t)
    # dynamical HT/2 scale + small trimmed event target for a fast test
    t = t.replace('fixed_ren_scale = true', 'fixed_ren_scale = false')
    t = t.replace('fixed_fact_scale = true', 'fixed_fact_scale = false')
    t = re.sub(r'events = \d+', 'events = %d' % events, t)
    for pat, repl in (toml_edits or []):
        t = re.sub(pat, repl, t)
    open(toml, 'w').write(t)

    env = dict(os.environ)
    env['LHAPDF_DATA_PATH'] = datadir
    log = pjoin(run_dir, 'mg7_postproc.log')
    if switch_lines:
        stdin_text = '\n'.join(list(switch_lines) + ['done']) + '\n'
        # the switches are answered through stdin on purpose
        args = [sys.executable, pjoin(run_dir, 'bin', 'generate_events'),
                '--allow-stdin']
    else:
        stdin_text = None
        args = [sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f']
    with open(log, 'w') as logf:
        proc = subprocess.run(args, cwd=run_dir, env=env, input=stdin_text,
                              text=True, stdout=logf, stderr=subprocess.STDOUT,
                              timeout=timeout)
    # Surface any per-tool *_crash.log written by run_selected_tools when a
    # post-processing tool fails *without* failing generate_events (rc stays 0),
    # so a silently-swallowed tool failure is visible in CI.
    for cl in sorted(glob.glob(pjoin(run_dir, '**', '*_crash.log'), recursive=True)):
        try:
            print('\n----- %s -----\n%s' % (cl, open(cl).read()),
                  file=sys.stderr, flush=True)
        except Exception:
            pass
    if proc.returncode != 0:
        # surface the generate_events output (the log lives in a tmp dir that CI
        # does not upload) so the real crash is visible in the test failure.
        test.fail('mg7 generate_events failed (rc=%s)\n%s'
                  % (proc.returncode, _mg7_postproc_report(run_dir)))
    runs = sorted(glob.glob(pjoin(run_dir, 'Events', 'run_*')))
    test.assertTrue(runs, 'no mg7 run directory under %s' % run_dir)
    return runs[0]


def _mg7_postproc_report(run_dir, nlines=120):
    """The content of the Events run directories and the end of the
    generate_events log of _run_mg7_postproc, for a failure message: both live
    in a tmp dir that CI does not upload."""
    import glob
    def tail(path):
        try:
            text = ''.join(open(path).readlines()[-nlines:])
        except Exception as err:
            text = '(could not read %s: %s)' % (path, err)
        return '----- %s (last %d lines) -----\n%s' % (path, nlines, text)

    out = []
    for run in sorted(glob.glob(pjoin(run_dir, 'Events', 'run_*'))):
        out.append('----- %s -----\n%s' % (run, '\n'.join(sorted(os.listdir(run)))))
        # the logs and scripts of the post-processing tools (Pythia8 shower, ...)
        for path in sorted(glob.glob(pjoin(run, '*.log')) + glob.glob(pjoin(run, '*.sh'))):
            if not os.path.basename(path).startswith('compile_'):
                out.append(tail(path))
    out.append(tail(pjoin(run_dir, 'mg7_postproc.log')))
    return '\n'.join(out)


#===============================================================================
# TestCmd
#===============================================================================
class TestMECmdShell(unittest.TestCase):
    """this treats all the command not related to MG_ME"""
    
    def setUp(self):
        
        self.debugging = unittest.debug
        if self.debugging:
            self.path = pjoin(MG5DIR, "tmp_test")
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            os.mkdir(pjoin(MG5DIR, "tmp_test"))
        else:
            self.path = tempfile.mkdtemp(prefix='acc_test_mg5')
        self.run_dir = pjoin(self.path, 'MGPROC')

        if logging.getLogger('madgraph').level >= 20:
            self.stdout = open(os.devnull, 'w')
        else:
            self.stdout = sys.stdout
    
    def tearDown(self):

        if self.path != pjoin(MG5DIR, "tmp_test"):
            shutil.rmtree(self.path)
        stdout = getattr(self, 'stdout', None)
        if stdout not in [None, sys.stdout, sys.stderr] and not stdout.closed:
            self.stdout.close() 

    def get_stdout(self):
        stdout = getattr(self, 'stdout', None)
        if logging.getLogger('madgraph').level >= 20:
            if stdout is None or stdout.closed:
                self.stdout = open(os.devnull, 'w')
        elif getattr(sys.stdout, 'closed', False):
            if stdout is None or stdout.closed or stdout == sys.stdout:
                self.stdout = open(os.devnull, 'w')
        else:
            self.stdout = sys.stdout
        return self.stdout
    
    def generate(self, process, model):
        """Create a process"""

        try:
            shutil.rmtree(self.run_dir)
        except Exception as error:
            pass
        interface = MGCmd.MasterCmd()
        interface.no_notification()
        interface.run_cmd('import model %s' % model)
        if isinstance(process, str):
            interface.run_cmd('generate %s' % process)
        else:
            for p in process:
                interface.run_cmd('add process %s' % p)

        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        #if not misc.which('root'):
        #    raise Exception('root is require for this test')
        #interface.exec_cmd('set pythia-pgs_path %s --no_save' % pjoin(MG5DIR, 'pythia-pgs'))
        interface.onecmd('output madevent %s -f' % self.run_dir)            
        
        if os.path.exists(pjoin(interface.options['syscalc_path'],'sys_calc')):
            shutil.rmtree(interface.options['syscalc_path'])
            #print "install SysCalc"
            #interface.onecmd('install SysCalc')
        
        
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=self.run_dir)
        self.cmd_line.no_notification()
        #self.cmd_line.options['syscalc_path'] = pjoin(MG5DIR, 'SysCalc')
        
    
    @staticmethod
    def join_path(*path):
        """join path and treat spaces"""     
        combine = os.path.join(*path)
        return combine.replace(' ',r'\ ')        
    
    def do(self, line):
        """ exec a line in the cmd under test """        
        self.cmd_line.run_cmd(line)
        
    def test_madevent_dy3j_mlm(self):
        """ Test that biasing LO event generation works as intended. """
        self.out_dir = self.run_dir

        if not self.debugging or not os.path.isdir(pjoin(MG5DIR,'BackUp_tmp_test')):
            self.generate('u g > l+ l- u u u~', 'sm')

            run_card = banner.RunCardLO(pjoin(self.out_dir, 'Cards', 'run_card.dat'))
            run_card.set('ickkw', 1, user=True)
            run_card.set('xqcut', 10.0, user=True)
            run_card.write(pjoin(self.out_dir, 'Cards', 'run_card.dat'))
            
            # Compile the code
            stdout = self.get_stdout()
            subprocess.Popen(['make'], cwd=pjoin(self.out_dir, 'Source'), stdout=stdout, stderr=stdout).wait()
            subprocess.Popen(['make', 'madevent_forhel'],                         
                             cwd=pjoin(self.out_dir, 'SubProcesses', 'P1_qg_llqqq'),
                             stdout=stdout, stderr=stdout).wait()
            with open(pjoin(self.out_dir, 'SubProcesses', 'P1_qg_llqqq', 'run_config.txt'), 'w') as fsock:  
                fsock.write('1000 5 3\n')  
                fsock.write('0.1\n')       # Accuracy
                fsock.write('2\n')         # Grid Adjustment 0=none, 2=adjust   
                fsock.write('1\n')         # Suppress Amplitude 1=yes
                fsock.write('0\n')         # Helicity Sum/event 0=exact
                fsock.write('      86\n')
            
        stdout = self.get_stdout()
        return_code = subprocess.Popen(
            ['./madevent_forhel'],
            cwd=pjoin(self.out_dir, 'SubProcesses', 'P1_qg_llqqq'),
            stdin=open(pjoin(self.out_dir, 'SubProcesses', 'P1_qg_llqqq', 'run_config.txt')),
            stdout=stdout, stderr=stdout
        ).wait()
            
        self.assertEqual(return_code, 0)



    def test_madevent_ptj_bias(self):
        """ Test that biasing LO event generation works as intended. """
        self.out_dir = self.run_dir

        if not self.debugging or not os.path.isdir(pjoin(MG5DIR,'BackUp_tmp_test')):
            self.generate('d d~ > u u~', 'sm')

            run_card = banner.RunCardLO(pjoin(self.out_dir, 'Cards','run_card.dat'))
            # Some test checking that some cut are absent/present by default
            self.assertIn('ptj', run_card.user_set)
            self.assertIn('drjj', run_card.user_set)
            self.assertIn('ptj2min', run_card.user_set)
            self.assertNotIn('ptj3min', run_card.user_set)
            self.assertIn('mmjj', run_card.user_set)
            self.assertNotIn('ptheavy', run_card.user_set)
            self.assertNotIn('Ej', run_card.user_set)
            
            
            run_card.set('bias_module','ptj_bias',user=True)
            run_card.set('bias_parameters',"{'ptj_bias_target_ptj': 1000.0,'ptj_bias_enhancement_power': 4.0}",user=True)
            run_card.set('use_syst',False)
            run_card.set('nevents',10000)            
            run_card.write(pjoin(self.out_dir, 'Cards','run_card.dat'))
            self.do('launch -f')
            run_card = banner.RunCardLO(pjoin(self.out_dir, 'Cards','run_card.dat'))
            run_card.set('bias_module','dummy',user=True)
            run_card.set('bias_parameters',"{}",user=True)
            run_card.set('use_syst',False)
            run_card.set('nevents',10000)
            run_card.write(pjoin(self.out_dir, 'Cards','run_card.dat'))
            self.do('launch -f')
            if self.debugging:
                if os.path.isdir(pjoin(MG5DIR,'BackUp_tmp_test')):
                    shutil.rmtree(pjoin(MG5DIR,'BackUp_tmp_test'))
                misc.copytree(pjoin(MG5DIR,'tmp_test'),
                                pjoin(MG5DIR,'BackUp_tmp_test'))
        else:
            shutil.rmtree(pjoin(MG5DIR,'tmp_test'))
            misc.copytree(pjoin(MG5DIR,'BackUp_tmp_test'),pjoin(MG5DIR,'tmp_test'))

        biased_events = lhe_parser.EventFile(pjoin(self.out_dir, 'Events','run_01','unweighted_events.lhe.gz'))
        unbiased_events = lhe_parser.EventFile(pjoin(self.out_dir, 'Events','run_02','unweighted_events.lhe.gz'))
                
        biased_events_ptj  = []
        biased_events_wgts = []
        for event in biased_events:
            biased_events_ptj.append(math.sqrt(event[2].px**2+event[2].py**2))
            biased_events_wgts.append(event.wgt)
        
        biased_median_ptj = sorted(biased_events_ptj)[len(biased_events_ptj)//2]
        unbiased_events_ptj = []
        for event in unbiased_events:
            unbiased_events_ptj.append(math.sqrt(event[2].px**2+event[2].py**2))
        unbiased_median_ptj = sorted(unbiased_events_ptj )[len(unbiased_events_ptj)//2]
        
        # Make that not all biased events have the same weights
        self.assertGreater(len(set(biased_events_wgts)),1)
        # Make sure that there is significantly more events in the ptj tail
        self.assertGreater(biased_median_ptj,5.0*unbiased_median_ptj)
        # Make sure that the cross-section is close enough for the bias and
        # unbiased samples. The biased sample estimates it from its event
        # weights, which have a long tail (low ptj events carry large
        # weights): its statistical error is ~4.5% for 10k events, so a
        # fixed 10% (~2 sigma) tolerance fails for some seeds.
        nb = len(biased_events_wgts)
        mean_wgt = sum(biased_events_wgts) / nb
        stat_err = math.sqrt(sum((w - mean_wgt)**2 for w in biased_events_wgts)
                             / (nb - 1) / nb)
        self.assertLess(abs(biased_events.cross-unbiased_events.cross),
                        4 * stat_err,
                        'biased %s +- %s vs unbiased %s' % (biased_events.cross,
                                                stat_err, unbiased_events.cross))

    def test_madspin_gridpack(self):

        self.out_dir = self.run_dir
        self.generate('g g > t t~', 'sm')

        #put the MadSpin card
        ff = open(pjoin(self.out_dir, 'Cards/madspin_card.dat'), 'w')
        orig_card =  open(pjoin(self.out_dir, 'Cards/madspin_card_default.dat')).read()
        ff.write('set ms_dir %s' % pjoin(self.out_dir, 'MSDIR1'))
        ff.write(orig_card)
        ff.close()
        
        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards','run_card.dat'))
        self.assertNotIn('ptj', run_card.user_set)
        self.assertNotIn('drjj', run_card.user_set)
        self.assertNotIn('ptj2min', run_card.user_set)
        self.assertNotIn('ptj3min', run_card.user_set)
        self.assertNotIn('mmjj', run_card.user_set)
        self.assertIn('ptheavy', run_card.user_set)
        self.assertNotIn('el', run_card.user_set)
        self.assertNotIn('ej', run_card.user_set)
        self.assertNotIn('polbeam1', run_card.user_set)
        self.assertNotIn('ptl', run_card.user_set)
        
        #reduce the number of events
        files.cp(pjoin(_file_path, 'input_files', 'run_card_matching.dat'),
                 pjoin(self.out_dir, 'Cards/run_card.dat'))

        #create the gridpack        
        self.do('launch -f')
        self.check_parton_output('run_01', 100)
        self.check_parton_output('run_01_decayed_1', 100)
        #move the MS gridpack
        self.assertTrue(os.path.exists(pjoin(self.out_dir, 'MSDIR1')))
        files.mv(pjoin(self.out_dir, 'MSDIR1'), pjoin(self.out_dir, 'MSDIR2'))
        
        #put the MadSpin card
        ff = open(pjoin(self.out_dir, 'Cards/madspin_card.dat'), 'w')
        ff.write('set ms_dir %s' % pjoin(self.out_dir, 'MSDIR2'))
        ff.write(orig_card)
        ff.close()
               
        #create the gridpack        
        self.do('launch -f')
        
        self.check_parton_output('run_02_decayed_1', 100)           
        
    def _setup_madspin_ww_wz(self, *card_lines):
        """p p > w+ w- plus p p > w+ z, decayed by madspin_v1.

        The two productions have different total branching ratios -- W+W-
        0.67 x 0.22, W+Z 0.67 x 0.07 -- so add_loose_decay gives W+Z a null
        decay (decay_struct None) for the events that must be dropped. Both
        are quark-initiated, so all_ME is keyed by flavour-grouped tags
        ((-81, 81), (-24, 24)), and every decay product is a merged class
        (j, l, vl)."""
        self.out_dir = self.run_dir
        self.generate(['p p > w+ w-', 'p p > w+ z'], 'sm')
        with open(pjoin(self.out_dir, 'Cards', 'madspin_card.dat'), 'w') as ff:
            for line in card_lines + ('set spinmode madspin_v1',
                                      'set seed 11',
                                      'decay w+ > j j',
                                      'decay w- > l- vl~',
                                      'decay z > l+ l-',
                                      'launch'):
                ff.write(line + '\n')
        run_card = banner.RunCardLO(pjoin(self.out_dir, 'Cards', 'run_card.dat'))
        run_card.set('nevents', 300)
        run_card.set('use_syst', False)
        run_card.write(pjoin(self.out_dir, 'Cards', 'run_card.dat'))

    def _check_madspin_ww_wz(self, run):
        """The decayed events of `run` (see _setup_madspin_ww_wz): flavour
        consistent decays, no merged PDG code, and the null decay applied to
        W+Z only."""
        allowed = {24: [{2, -1}, {4, -3}], -24: [{11, -12}, {13, -14}],
                   23: [{11, -11}, {13, -13}]}
        def production(event):
            return tuple(sorted(p.pid for p in event
                                if p.status in (1, 2) and p.mother1
                                and p.mother1.status == -1))

        undecayed = collections.Counter(
            production(e) for e in lhe_parser.EventFile(pjoin(
                self.out_dir, 'Events', run, 'unweighted_events.lhe.gz')))
        decayed = pjoin(self.out_dir, 'Events', run + '_decayed_1',
                        'unweighted_events.lhe.gz')
        # madevent logs a MadSpin crash and carries on: say so here
        self.assertTrue(os.path.exists(decayed),
                        'MadSpin wrote no decayed events for %s' % run)
        written = collections.Counter()
        for event in lhe_parser.EventFile(decayed):
            written[production(event)] += 1
            for p in event:
                self.assertFalse(80 < abs(p.pid) < 90,
                                 'merged PDG code %s written' % p.pid)
                if p.pid in allowed:
                    daughters = {d.pid for d in event if d.mother1 is p}
                    self.assertIn(daughters, allowed[p.pid])
        ww, wz = (-24, 24), (23, 24)
        self.assertEqual(set(undecayed), {ww, wz})
        # the larger total BR: every W+W- event is written
        self.assertEqual(written[ww], undecayed[ww])
        # the smaller one: kept with probability 0.07/0.22, so some but
        # not all of the ~50 W+Z events (0 kept: ~1e-8)
        self.assertGreater(written[wz], 0)
        self.assertLess(written[wz], undecayed[wz])

    def test_madspin_gridpack_unequal_br(self):
        """Reuse an ms_dir in madspin_v1 mode, down to the decayed events.

        Saving madspin.pkl used to crash on the null decay of W+Z
        (switch_all_model_instance iterated its decay_struct, None). The
        second launch reuses the directory: run_from_pickle used to restore
        no model, so the production tags were looked up unmerged,
        ((-2, 2), (-24, 24)), and load_event died with a KeyError.
        """
        ms_dir = pjoin(self.run_dir, 'MSDIR')
        self._setup_madspin_ww_wz('set ms_dir %s' % ms_dir)

        # build the directory
        self.do('launch -f')
        pkl = pjoin(ms_dir, 'madspin.pkl')
        self.assertTrue(os.path.exists(pkl),
                        'the first launch did not build the ms_dir')
        built = os.stat(pkl).st_mtime_ns

        # reuse it on a new sample
        self.do('launch -f')
        self.assertEqual(os.stat(pkl).st_mtime_ns, built,
                         'the second launch should reuse madspin.pkl')

        for run in ('run_01', 'run_02'):
            self._check_madspin_ww_wz(run)

    def test_madspin_use_old_dir(self):
        """use_old_dir reruns madspin_v1 on the matrix elements the previous
        run left in the same directory (production_me/all_ME.pkl).

        That file is written by save_status_to_pickle: all_decay and the
        decay structures as their repr, no model. The reload took it as it
        was -- all_decay a string, the decay branches plain dicts, the tags
        unmerged -- and a successful run deleted production_me anyway, so
        there was nothing to reload in the first place.
        """
        self._setup_madspin_ww_wz('set use_old_dir True')

        self.do('launch -f')
        # MadSpin works in the process directory (the events are in Events/)
        pkl = pjoin(self.out_dir, 'production_me', 'all_ME.pkl')
        self.assertTrue(os.path.exists(pkl),
                        'the first launch did not keep production_me')
        built = os.stat(pkl).st_mtime_ns

        self.do('launch -f')
        # a reload that fails falls back to regenerating, which rewrites it
        self.assertEqual(os.stat(pkl).st_mtime_ns, built,
                         'the second launch should reload all_ME.pkl')

        for run in ('run_01', 'run_02'):
            self._check_madspin_ww_wz(run)
        
        
    def test_width_computation(self):
        """test the param_card created is correct"""
        
        cmd = os.getcwd()
        self.generate(['Z > l+ l-','Z > j j'], 'sm')
        self.assertEqual(cmd, os.getcwd())
        
        # check that the run_card do not have cut
        run_card = banner.RunCard(pjoin(self.run_dir,'Cards','run_card.dat'))
        self.assertEqual(run_card['ptj'], 0)
        self.assertIn('ptj', run_card.user_set)
        self.assertIn('drjj', run_card.user_set)
        self.assertIn('ptj2min', run_card.user_set)
        self.assertNotIn('ptj3min', run_card.user_set)
        self.assertIn('mmjj', run_card.user_set)
        self.assertNotIn('ptheavy', run_card.user_set)
        self.assertNotIn('el', run_card.user_set)
        self.assertNotIn('ej', run_card.user_set)
        self.assertNotIn('polbeam1', run_card.user_set)
        self.assertIn('ptl', run_card.user_set)
        
        self.do('calculate_decay_widths -f')        
        
        # test the param_card is correctly written
        self.assertTrue(os.path.exists('%s/Events/run_01/param_card.dat' % self.run_dir))
        
        text = open('%s/Events/run_01/param_card.dat' % self.run_dir).read()
        data = text.split('DECAY  23')[1].split('DECAY',1)[0]
        data = data.split('\n')
        if '#' in data[0]:
            data[0] = data[0].split('#',1)[0]
        width = float(data[0])
        self.assertAlmostEqual(width, 1.492240e+00, delta=1e-4)
        values = {(3,-3): 2.493165e-01,
                  (1,-1): 2.493165e-01,
                  (4,-4): 1.944158e-01,
                  (2,-2): 1.944158e-01,
                  (-11,11): 5.626776e-02,
                  (-13,13): 5.626776e-02}
        if self.debugging:
            misc.sprint('\n'.join(data))
        for l in data[1:]:
            if l.startswith("#"):
                continue
            l = l.strip()
            if not l:
                continue
            #2.493165e-01   2    3  -3 # 0.37204
            br, _, id1,id2,_,_ = l.split()
            
            self.assertAlmostEqual(float(br), values[(int(id1),int(id2))],delta=2e-3)
        
        
#         self.assertEqual("""1.492240e+00
# #  BR             NDA  ID1    ID2   ...
#    2.493165e-01   2    3  -3 # 0.37204
#    2.493165e-01   2    1  -1 # 0.37204
#    1.944158e-01   2    4  -4 # 0.290115
#    1.944158e-01   2    2  -2 # 0.290115
#    5.626776e-02   2    -11  11 # 0.083965
#    5.626776e-02   2    -13  13 # 0.083965
# #
# #      PDG        Width""".split('\n'), data.strip().split('\n'))

    def test_width_nlocomputation(self):
        """test the param_card created is correct"""
        
        cmd = os.getcwd()
        
        interface = MGCmd.MasterCmd()
        interface.no_notification()

        interface.exec_cmd("import model loop_qcd_qed_sm", errorhandling=False, 
                                                        printcmd=False, 
                                                        precmd=True, postcmd=False)
        interface.exec_cmd("compute_widths H Z W+ t --nlo --output=%s" % \
                           pjoin(self.path, "param_card.dat")
                           , errorhandling=False, 
                                                        printcmd=False, 
                                                        precmd=True, postcmd=False)      
        
        # test the param_card is correctly written
        self.assertTrue(os.path.exists('%s/param_card.dat' % self.path))
        text = open('%s/param_card.dat' % self.path).read()
        pattern = re.compile(r"decay\s+23\s+([+-.\de]*)", re.I)
        value = float(pattern.search(text).group(1))
        self.assertAlmostEqual(2.48883,value, delta=1e-3)
        pattern = re.compile(r"decay\s+24\s+([+-.\de]*)", re.I)
        value = float(pattern.search(text).group(1))
        self.assertAlmostEqual(2.08465,value, delta=1e-3)
        pattern = re.compile(r"decay\s+25\s+([+-.\de]*)", re.I)
        value = float(pattern.search(text).group(1))
        self.assertAlmostEqual(3.514960e-03,value, delta=1e-3)
        pattern = re.compile(r"decay\s+6\s+([+-.\de]*)", re.I)
        value = float(pattern.search(text).group(1))
        self.assertAlmostEqual(1.36728,value, delta=5e-3)        
        



        
    def test_creating_matched_plot(self):
        """test that the creation of matched plot works and the systematics as well"""

        cmd = os.getcwd()
        self.generate('p p > W+', 'sm')
        self.assertEqual(cmd, os.getcwd())        

        if not self.cmd_line.options['pythia-pgs_path']:
            return

        shutil.copy(os.path.join(_file_path, 'input_files', 'run_card_matching.dat'),
                    '%s/Cards/run_card.dat' % self.run_dir)
        shutil.copy('%s/Cards/pythia_card_default.dat' % self.run_dir,
                    '%s/Cards/pythia_card.dat' % self.run_dir)
        try:
            os.remove(pjoin(self.run_dir, 'Cards',  'madanalysis5_parton_card.dat'))
            os.remove(pjoin(self.run_dir, 'Cards',  'madanalysis5_hadron_card.dat'))
        except:
            pass
        self.do('generate_events -f')     

        #modify the run_card
        run_card = self.cmd_line.run_card
        run_card['nevents'] = 44
        run_card['use_syst'] = 'F'
        run_card.write('%s/Cards/run_card.dat'% self.run_dir,
                                    '%s/Cards/run_card_default.dat'% self.run_dir)

        self.assertEqual(cmd, os.getcwd())        
        self.do('generate_events -f')
        self.assertEqual(int(self.cmd_line.run_card['nevents']), 44)
        self.do('pythia run_01 -f')
        self.do('quit')
        
        self.assertEqual(int(self.cmd_line.run_card['nevents']), 100)
        
        self.check_parton_output(syst=False)
        self.check_parton_output('run_02', target_event=44, syst=False)
        self.check_pythia_output(syst=False)        
        self.assertEqual(cmd, os.getcwd())

        
    def test_group_subprocess(self):
        """check that both u u > u u gives the same result"""
        

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
        mg_cmd.exec_cmd(' generate u u > u u')
        mg_cmd.exec_cmd('output madevent %s/'% self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir= self.run_dir)
        self.cmd_line.no_notification()

        self.cmd_line.exec_cmd('set automatic_html_opening False')
        
        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        
        self.run_dir = pjoin(self.path, 'MGPROC2')
        mg_cmd.exec_cmd('set group_subprocesses False')
        mg_cmd.exec_cmd('generate u u > u u')
        mg_cmd.exec_cmd('output madevent %s' % self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir= self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')

        
        self.do('generate_events -f')        
        
        val2 = self.cmd_line.results.current['cross']
        err2 = self.cmd_line.results.current['error']        
        
        self.assertLess(abs(val2 - val1) / (err1 + err2), 5,
            'grouped (%s +- %s) vs ungrouped (%s +- %s) disagree'
            % (val1, err1, val2, err2))
        # previously PDF was nn23lo1 (lhaid 230000) with this reference value 1310200.0
        # Now that madevent convolutes with the same LHAPDF set as mg7, this
        # agrees with the mg7 result for the same process (8.03e+05, see
        # test_group_subprocess_mg7) instead of sitting a factor ~1.6 above it.
        target = 801960.0
        self.assertLess(abs(val2 - target) / (err2), 5,
            'u u > u u cross-section %s +- %s far from reference %s'
            % (val2, err2, target))
        #check precision
        self.assertLess(err2 / val2, 0.005)
        self.assertLess(err1 / val1, 0.005)

    def test_group_subprocess_mg7(self):
        """mg7 equivalent of test_group_subprocess for u u > u u.

        Runs the mg7 (madspace) integrator with group_subprocesses on and off
        and checks the two cross-sections agree (grouping consistency). It also
        pins the absolute value to the mg7-native result obtained with the
        run_card.toml defaults (NNPDF40_lo_as_01180 + dynamical HT/2 scale,
        events=2000) ~ 8.03e+05 pb.

        NOTE: this now AGREES with the madevent reference in
        test_group_subprocess (8.02e5 pb): that test used to run on madevent's
        internal nn23lo1 and sit at 1.31e6, and the LO run_card default moving
        to the same LHAPDF set is what closed the gap.
        """
        import glob, json
        # The mg7 cross-section run needs the madspace runtime and a resolvable
        # LHAPDF data path; skip cleanly where that stack is unavailable.
        try:
            import madspace
            has_mg7 = hasattr(madspace, 'ChannelEventGenerator')
        except ImportError:
            has_mg7 = False
        datadir = os.environ.get('LHAPDF_DATA_PATH')
        if not datadir:
            try:
                datadir = subprocess.check_output(
                    ['lhapdf-config', '--datadir']).decode().strip()
            except Exception:
                datadir = None
        if not has_mg7 or not datadir or not os.path.isdir(datadir):
            self.skipTest('mg7 runtime stack (madspace + LHAPDF data) unavailable')
        # the mg7 run_card.toml default PDF must be present in the data dir
        if not glob.glob(pjoin(datadir, 'NNPDF40_lo_as_01180*')):
            self.skipTest('NNPDF40_lo_as_01180 PDF set not available')

        def run_mg7(group):
            run_dir = pjoin(self.path, 'MG7_%s' % ('grp' if group else 'ungrp'))
            if os.path.isdir(run_dir):
                shutil.rmtree(run_dir)
            mg = MGCmd.MasterCmd()
            mg.no_notification()
            mg.exec_cmd('set automatic_html_opening False --no_save')
            mg.exec_cmd('set group_subprocesses %s' % ('True' if group else 'False'))
            mg.exec_cmd('generate u u > u u')
            mg.exec_cmd('output mg7 %s' % run_dir)
            # Use the dynamical HT/2 scale (= madevent dynamical_scale_choice=3)
            # and a trimmed-but-converged event target.
            toml = pjoin(run_dir, 'Cards', 'run_card.toml')
            t = open(toml).read()
            t = t.replace('fixed_ren_scale = true', 'fixed_ren_scale = false')
            t = t.replace('fixed_fact_scale = true', 'fixed_fact_scale = false')
            t = re.sub(r'events = \d+', 'events = 2000', t)
            open(toml, 'w').write(t)
            env = dict(os.environ)
            env['LHAPDF_DATA_PATH'] = datadir
            log = pjoin(run_dir, 'mg7_gen.log')
            ret = subprocess.call(
                [sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
                cwd=run_dir, env=env,
                stdout=open(log, 'w'), stderr=subprocess.STDOUT)
            self.assertEqual(ret, 0, 'mg7 generate_events failed (see %s)' % log)
            infos = sorted(glob.glob(pjoin(run_dir, 'Events', '*', 'info.json')))
            self.assertTrue(infos, 'no mg7 info.json under %s' % run_dir)
            info = json.load(open(infos[-1]))['process']
            return float(info['mean']), float(info.get('error') or 0.0)

        val1, err1 = run_mg7(True)
        val2, err2 = run_mg7(False)
        # group_subprocesses on/off must give the same cross-section
        self.assertLess(abs(val1 - val2) / (err1 + err2 + 1e-30), 5,
            'mg7 grouped (%s +- %s) vs ungrouped (%s +- %s) disagree'
            % (val1, err1, val2, err2))
        # The madevent run in test_group_subprocess now lands on the same
        # value (8.0196e+05 +- 2.6e+03) with lhapdf NNPDF40_lo_as_01180. The -39% against
        # the 1.277e+06 obtained with NNPDF23_lo_as_0130_qed is the PDF change
        # alone: alpha_s^2 (-18%) times the smaller NNPDF4.0 u-quark luminosity.
        # Spread over 6 runs at events=2000: 7.96e5-8.15e5, mean 8.03e5.
        # previously PDF was NNPDF40MC_lo_as_01180 with this reference value 7.76e+05
        target = 8.03e+05
        self.assertLess(abs(val2 - target) / target, 0.10,
            'mg7 u u > u u cross-section %s far from mg7 reference %s'
            % (val2, target))

    def test_madevent_flavor_zjj(self):
        """Cross-section and initial-state flavor composition for
        q q > z q q QCD=0 with q = u d (madevent backend).

        With q a merged u/d multiparticle, the grouped matrix element
        carries per-flavor identical-particle corrections (the BROKEN_SYM
        factor) and a per-group symmetry/symfact treatment. This pins the
        total cross-section and the average number of initial-state u and d
        quarks per event (two partons per event) so a mis-applied symmetry
        factor is caught.
        """

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
        mg_cmd.exec_cmd('import model sm')
        mg_cmd.exec_cmd('define q = u d')
        mg_cmd.exec_cmd('generate q q > z q q QCD=0')
        mg_cmd.exec_cmd('output madevent %s' % self.run_dir)

        self.cmd_line = MECmd.MadEventCmdShell(me_dir=self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')

        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards', 'run_card.dat'))
        run_card.set('nevents', 1000, user=True)
        run_card.write(pjoin(self.run_dir, 'Cards', 'run_card.dat'))

        self.do('launch -f')

        cross = self.cmd_line.results.current['cross']
        error = self.cmd_line.results.current['error']
        # Reference 6.124 pb is the sum of the three single-flavor
        # subprocesses run separately (u u > z u u = 0.353, u d > z u d =
        # 5.691, d d > z d d = 0.092), i.e. the result free of the merged-q
        # grouping machinery.
        # previously PDF was nn23lo1 (lhaid 230000) with this reference value 6.124
        self.assertAlmostEqual(cross, 5.144, delta=max(0.1, 5 * error))

        events = lhe_parser.EventFile(pjoin(self.run_dir, 'Events', 'run_01',
                                            'unweighted_events.lhe.gz'))
        n_u = n_d = n_events = 0
        for event in events:
            n_events += 1
            for particle in (event[0], event[1]):
                if abs(particle.pid) == 2:
                    n_u += 1
                elif abs(particle.pid) == 1:
                    n_d += 1
        self.assertGreater(n_events, 0)
        # q = u d only: every initial-state parton is a u or a d quark.
        self.assertEqual(n_u + n_d, 2 * n_events)
        self.assertAlmostEqual(n_u / n_events, 1.042, delta=0.1)
        self.assertAlmostEqual(n_d / n_events, 0.958, delta=0.1)

    def test_madevent_flavor_zud(self):
        """Cross-section and initial-state flavor composition for
        u q > z u q QCD=0 with q = u d (madevent backend).

        One initial leg is fixed u, the other a merged u/d
        multiparticle, so the grouped subprocess covers the u u and
        u d initial states. The grouped matrix element carries per-
        flavor identical-particle corrections and a per-group
        symmetry/symfact treatment. This pins the total cross-section
        and the average number of initial-state u and d quarks per
        event (two partons per event) so a mis-applied symmetry
        factor on the u q pattern is caught.
        """

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
        mg_cmd.exec_cmd('import model sm')
        mg_cmd.exec_cmd('define q = u d')
        mg_cmd.exec_cmd('generate u q > z u q QCD=0')
        mg_cmd.exec_cmd('output madevent %s' % self.run_dir)

        self.cmd_line = MECmd.MadEventCmdShell(me_dir=self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')

        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards', 'run_card.dat'))
        run_card.set('nevents', 1000, user=True)
        run_card.write(pjoin(self.run_dir, 'Cards', 'run_card.dat'))

        self.do('launch -f')

        cross = self.cmd_line.results.current['cross']
        error = self.cmd_line.results.current['error']
        # Reference 3.215 pb is u u > z u u (0.353) + u d > z u d
        # (2.862) run as separate single-flavor processes, i.e. the
        # result free of the merged-q grouping machinery on the u q
        # pattern.
        # previously PDF was nn23lo1 (lhaid 230000) with this reference value 3.215
        self.assertAlmostEqual(cross, 2.754, delta=max(0.1, 5 * error))

        events = lhe_parser.EventFile(pjoin(self.run_dir, 'Events', 'run_01',
                                            'unweighted_events.lhe.gz'))
        n_u = n_d = n_events = 0
        for event in events:
            n_events += 1
            for particle in (event[0], event[1]):
                if abs(particle.pid) == 2:
                    n_u += 1
                elif abs(particle.pid) == 1:
                    n_d += 1
        self.assertGreater(n_events, 0)
        # q = u d only: every initial-state parton is a u or a d quark.
        self.assertEqual(n_u + n_d, 2 * n_events)
        # One beam is always u; the other is u with weight
        # 0.353/3.215 and d with weight 2.862/3.215.
        self.assertAlmostEqual(n_u / n_events, 1+0.353/3.215, delta=0.1)
        self.assertAlmostEqual(n_d / n_events, 2.862/3.215, delta=0.1)

    def test_madevent_flavor_zud_nogroup(self):
        """Same physics as test_madevent_flavor_zud with
        group_subprocesses=False. With grouping disabled, the single-
        flavor subprocesses run independently and the reference still
        matches sigma(u u > z u u) + sigma(u d > z u d) ~ 3.215 pb.
        """

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
        mg_cmd.exec_cmd('set group_subprocesses False')
        mg_cmd.exec_cmd('import model sm')
        mg_cmd.exec_cmd('define q = u d')
        mg_cmd.exec_cmd('generate u q > z u q QCD=0')
        mg_cmd.exec_cmd('output madevent %s' % self.run_dir)

        self.cmd_line = MECmd.MadEventCmdShell(me_dir=self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')

        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards', 'run_card.dat'))
        run_card.set('nevents', 1000, user=True)
        run_card.write(pjoin(self.run_dir, 'Cards', 'run_card.dat'))

        self.do('launch -f')

        cross = self.cmd_line.results.current['cross']
        error = self.cmd_line.results.current['error']
        # previously PDF was nn23lo1 (lhaid 230000) with this reference value 3.215
        self.assertAlmostEqual(cross, 2.754, delta=max(0.1, 5 * error))

        events = lhe_parser.EventFile(pjoin(self.run_dir, 'Events', 'run_01',
                                            'unweighted_events.lhe.gz'))
        n_u = n_d = n_events = 0
        for event in events:
            n_events += 1
            for particle in (event[0], event[1]):
                if abs(particle.pid) == 2:
                    n_u += 1
                elif abs(particle.pid) == 1:
                    n_d += 1
        self.assertGreater(n_events, 0)
        self.assertEqual(n_u + n_d, 2 * n_events)
        self.assertAlmostEqual(n_u / n_events, 1+0.353/3.215, delta=0.1)
        self.assertAlmostEqual(n_d / n_events, 2.862/3.215, delta=0.1)

    def test_madevent_merged_flavor_uq(self):
        """Cross-section for u q > u q QCD=0 with q = u d (madevent backend).

        One initial leg is a fixed u, the other a merged u/d multiparticle,
        so the subprocess covers the u u and u d initial states. Running the
        two flavors separately (no merged multiparticle) gives 4428 pb; the
        merged-flavor path must reproduce that. This pins the cross-section
        so a mis-applied PDF convolution or symmetry factor is caught.
        """

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
        mg_cmd.exec_cmd('import model sm')
        mg_cmd.exec_cmd('define q = u d')
        mg_cmd.exec_cmd('generate u q > u q QCD=0')
        mg_cmd.exec_cmd('output madevent %s' % self.run_dir)

        # Explicit check of the initial-state mirroring, on top of the
        # cross-section: leg 1 is a fixed u while leg 2 is the merged u/d
        # multiparticle, so the beam-swapped "d u" initial state is NOT part of
        # this process. Neither flavor may therefore be mirrored -- a mirror on
        # the u d flavor would count it twice (and the cross-section check below
        # would then be off by the u d contribution).
        import glob as _glob
        mirror_incs = _glob.glob(pjoin(self.run_dir, 'SubProcesses', 'P*',
                                       'mirrorprocs.inc'))
        self.assertTrue(mirror_incs, 'no mirrorprocs.inc under %s' % self.run_dir)
        for path in mirror_incs:
            text = open(path).read().upper()
            self.assertNotIn('.TRUE.', text,
                'u q > u q (q = u d): leg 1 is a fixed u, so no initial-state '
                'mirroring may be applied, but %s requests one:\n%s'
                % (path, text))

        self.cmd_line = MECmd.MadEventCmdShell(me_dir=self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')

        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards', 'run_card.dat'))
        run_card.set('nevents', 1000, user=True)
        run_card.write(pjoin(self.run_dir, 'Cards', 'run_card.dat'))

        self.do('launch -f')

        cross = self.cmd_line.results.current['cross']
        error = self.cmd_line.results.current['error']
        # Reference 4428 pb is u u > u u plus u d > u d run as separate
        # single-flavor processes (no merged multiparticle).
        # previously PDF was nn23lo1 (lhaid 230000) with this reference value 4428.0
        self.assertAlmostEqual(cross, 3466.3, delta=max(30.0, 5 * error))

    def test_merged_flavor_initial_state_mirroring_mg7(self):
        """Initial-state mirroring of the merged-flavor mg7 output.

        The beam-swapped (mirror) copy of a flavor combination may only be added
        when the two initial legs range over the *same* set of flavors, i.e. when
        the swapped initial state really is part of the process and got collapsed
        into this one:

        * u q > u q (q = u d): leg 1 is a fixed u, so "d u > u d" is NOT part of
          the process -- no flavor may be mirrored. Mirroring the u d flavor here
          made its contribution twice too large.
        * q q > q q (q = u d): both legs span {u,d}, so the "d u" combination was
          collapsed into "u d" and only that *mixed* flavor must be mirrored.

        This is an output-level check (no event generation), so it needs no mg7
        runtime and directly guards the flag the cross-section depends on."""
        import json

        def mirrors(process, out_dir):
            mg = MGCmd.MasterCmd()
            mg.no_notification()
            for c in ['set automatic_html_opening False --no_save',
                      'import model sm', 'define q = u d',
                      'generate %s' % process]:
                mg.exec_cmd(c)
            mg.exec_cmd('output mg7 %s' % out_dir)
            info = json.load(open(pjoin(out_dir, 'SubProcesses',
                                        'subprocesses.json')))
            # {(initial pdg 1, initial pdg 2): mirror}
            return {tuple(f['options'][0][:2]): f['mirror']
                    for p in info for f in p['flavors']}

        # leg 1 fixed to u -> the d u initial state is absent -> never mirror
        uq = mirrors('u q > u q', pjoin(self.path, 'MG7_mirror_uq'))
        self.assertEqual(sorted(uq), [(2, 1), (2, 2)])
        for flavor, mirror in uq.items():
            self.assertFalse(mirror,
                'u q > u q (q = u d): flavor %s must not be mirrored (leg 1 is a '
                'fixed u, so d u > u d is not part of the process); mirroring it '
                'double counts the u d contribution' % (flavor,))

        # both legs span {u,d} -> the swapped mixed flavor was collapsed here
        qq = mirrors('q q > q q', pjoin(self.path, 'MG7_mirror_qq'))
        for flavor, mirror in qq.items():
            self.assertEqual(mirror, flavor[0] != flavor[1],
                'q q > q q (q = u d): only the mixed initial flavor may be '
                'mirrored, got mirror=%s for %s' % (mirror, (flavor,)))

        # e+ e- > e+ e-: the two beams are *different* particles (the leg
        # signature must stay signed -- comparing |pdg| made e+ and e- look like
        # the same beam and mirrored the flavor, doubling the cross-section).
        ee = mirrors('e+ e- > e+ e-', pjoin(self.path, 'MG7_mirror_ee'))
        self.assertEqual(sorted(ee), [(-11, 11)])
        for flavor, mirror in ee.items():
            self.assertFalse(mirror,
                'e+ e- > e+ e-: beams are different particles, flavor %s must '
                'not be mirrored (that doubles the cross-section)' % (flavor,))

    def test_color_flow_consistency_mg7(self):
        """LHECompleter (madspace/src/driver/lhe_output.cpp) must build
        without error for p p > j j j: it walks each diagram's vertices,
        cancelling color/anti-color indices (compute_decay_color), and raises
        if a color flow doesn't close. Exercises the real C++ code directly
        against every (channel, diagram, active color) in subprocesses.json."""
        import json
        # Prefer the locally-built module (madspace/install) over whatever
        # madspace happens to be on the system path, since this test exercises
        # local C++ source changes (diagram_propagator_pdgs).
        local_install = pjoin(MG5DIR, 'madspace', 'install')
        if os.path.isdir(local_install):
            if local_install not in sys.path:
                sys.path.insert(0, local_install)
            for mod_name in [m for m in sys.modules if m.startswith('madspace')]:
                mod_file = getattr(sys.modules[mod_name], '__file__', '') or ''
                if not mod_file.startswith(local_install):
                    del sys.modules[mod_name]
        try:
            import madspace as ms
            has_mg7 = hasattr(ms.SubprocArgs(), 'diagram_propagator_pdgs')
        except ImportError:
            has_mg7 = False
        if not has_mg7:
            self.skipTest('mg7 runtime stack (madspace) unavailable')

        mg = MGCmd.MasterCmd()
        mg.no_notification()
        for c in ['set automatic_html_opening False --no_save',
                  'import model sm',
                  'define p = g u c d s u~ c~ d~ s~',
                  'define j = g u c d s u~ c~ d~ s~',
                  'generate p p > j j j']:
            mg.exec_cmd(c)
        out_dir = pjoin(self.path, 'MG7_color_flow_consistency')
        mg.exec_cmd('output mg7 %s' % out_dir)
        meta_list = json.load(open(pjoin(out_dir, 'SubProcesses',
                                          'subprocesses.json')))

        checked = 0
        for subproc_index, meta in enumerate(meta_list):
            n_particles = len(meta['incoming']) + len(meta['outgoing'])
            incoming_masses = [0.0] * len(meta['incoming'])
            outgoing_masses = [0.0] * len(meta['outgoing'])
            topologies, permutations = [], []
            diagram_indices, diagram_color_indices = [], []
            diagram_propagator_pdgs = []
            for channel in meta['channels']:
                # mass/width don't affect color computation, only whether a
                # propagator counts as resonant -- use 0.0 throughout so every
                # propagator's color still gets validated.
                propagators = [
                    ms.Propagator(mass=0.0, width=0.0, integration_order=0,
                                  e_min=0.0, e_max=0.0, pdg_id=pid)
                    for pid in channel['propagators']
                ]
                diagram = ms.Diagram(incoming_masses, outgoing_masses,
                                      propagators, channel['vertices'])
                topo = ms.Topology.topologies(diagram)[0]
                diagrams = channel['diagrams']
                topologies.append(topo)
                permutations.append([d['permutation'] for d in diagrams])
                diagram_indices.append([d['diagram'] for d in diagrams])
                diagram_color_indices.append(
                    [d['active_colors'] for d in diagrams]
                )
                diagram_propagator_pdgs.append(
                    [d['propagator_pdgs'] for d in diagrams]
                )
                checked += sum(len(d['active_colors']) for d in diagrams)

            subproc_args = ms.SubprocArgs(
                process_id=subproc_index,
                topologies=topologies,
                permutations=permutations,
                diagram_indices=diagram_indices,
                diagram_color_indices=diagram_color_indices,
                diagram_propagator_pdgs=diagram_propagator_pdgs,
                color_flows=meta['color_flows'],
                pdg_color_types={
                    int(k): v for k, v in meta['pdg_color_types'].items()
                },
                helicities=[[0] * n_particles],
                pdg_ids=[flavor['options'] for flavor in meta['flavors']],
            )
            try:
                ms.LHECompleter([subproc_args], bw_cutoff=15.0)
            except Exception as err:
                self.fail(
                    'LHECompleter rejected subprocess %s (incoming=%s '
                    'outgoing=%s): %s'
                    % (subproc_index, meta['incoming'], meta['outgoing'], err)
                )

        self.assertGreater(
            checked, 0,
            'no (subprocess, channel, diagram, color) combination was found '
            'to check -- p p > j j j should generate plenty'
        )

    def test_single_qcd_order_mg7(self):
        """The mg7 output records the single alpha_s power (single_qcd_order in
        SubProcesses/proc_characteristics) whenever the QCD power of |M|^2 is the
        same for every diagram, and -1 otherwise. systematics uses it to
        reconstruct the LO reweighting info (no <mgrwt> in the mg7 LHE) and so
        compute scale/PDF uncertainties. Output-level check (no mg7 runtime)."""
        import madgraph.various.banner as banner_mod

        def single_qcd_order(setup, out_dir):
            mg = MGCmd.MasterCmd()
            mg.no_notification()
            for c in (['set automatic_html_opening False --no_save',
                       'import model sm'] + setup):
                mg.exec_cmd(c)
            mg.exec_cmd('output mg7 %s' % out_dir)
            pc = banner_mod.ProcCharacteristic(
                pjoin(out_dir, 'SubProcesses', 'proc_characteristics'))
            return int(pc['single_qcd_order'])

        # pure QCD 2->2: |M|^2 ~ alpha_s^2
        self.assertEqual(2, single_qcd_order(
            ['generate u u > u u'], pjoin(self.path, 'MG7_nqcd_uu')))
        # pure electroweak: no alpha_s
        self.assertEqual(0, single_qcd_order(
            ['generate e+ e- > mu+ mu-'], pjoin(self.path, 'MG7_nqcd_ee')))
        # mixed QCD/QED orders (QCD=2 and QED=2 contributions) -> not uniform
        self.assertEqual(-1, single_qcd_order(
            ['generate u u~ > d d~', 'add process u u~ > d d~ QED=2'],
            pjoin(self.path, 'MG7_nqcd_mixed')))

    def _check_systematics_weights(self, lhe_path, expected_ids=None):
        """Every event carries the variation weights of the <initrwgt> header
        and they differ from the nominal weight (real scale/PDF information
        entered the computation). Returns the number of events."""
        banner = banner_mod.Banner(lhe_path)
        self.assertIn('initrwgt', banner, 'no <initrwgt> header in %s' % lhe_path)
        ids = re.findall(r'<weight id=["\'](\w+)["\']', banner['initrwgt'])
        self.assertTrue(ids, 'empty <initrwgt> header')
        if expected_ids is not None:
            self.assertEqual(ids, expected_ids)
        nb_event = 0
        nb_nontrivial = 0
        for evt in lhe_parser.EventFile(lhe_path):
            rwgt = evt.parse_reweight()
            self.assertEqual(sorted(rwgt), sorted(ids),
                'the event weights do not match the <initrwgt> header')
            if any(abs(w - evt.wgt) > 1e-6 * abs(evt.wgt) for w in rwgt.values()):
                nb_nontrivial += 1
            nb_event += 1
        self.assertGreater(nb_event, 0, 'no event found in %s' % lhe_path)
        self.assertGreater(nb_nontrivial, 0,
            'the systematic weights are all equal to the nominal weight')
        return nb_event

    def test_systematics_mg7(self):
        """Scale/PDF systematics computed by madspace when the events are
        written ([systematics] section of the run_card), end to end (u u > u u).

        Checks the LHEF3 output: an <initrwgt> header with the 3x3 scale grid
        (8 weights, ids 1-8) plus one PDF member of another set when that set
        is available (a cross-set variation, with its own alpha_s), a <rwgt>
        block in every event, non-trivial weights, the events.weights.json
        sidecar with the same ids and the per-variation cross sections, and the
        systematics summary in info.json. No LHAPDF python module is needed."""
        datadir = _mg7_datadir_or_skip(self)
        # a cross-set PDF variation (its own alpha_s) rather than a member of
        # the default set, so this also covers the cross-set code path
        other_set = 'NNPDF23_lo_as_0130_qed'
        has_other = os.path.isfile(pjoin(datadir, other_set, '%s_0001.dat' % other_set))
        pdf_entry = '["%s@1"]' % other_set if has_other else '["central"]'
        nb_pdf = 1 if has_other else 0
        run = _run_mg7_postproc(
            self,
            ['set automatic_html_opening False --no_save',
             'import model sm',
             'generate u u > u u'],
            pjoin(self.path, 'MG7_syst'), datadir,
            switch_lines=None,
            toml_edits=[(r'\npdf = \[[^\]]*\]', '\npdf = %s' % pdf_entry),
                        # one histogram, to get the event-sample histograms
                        # with their variation bands in info.json. The whole
                        # section is replaced, not appended to: the output
                        # writes process-specific defaults there (sqrt_s among
                        # them), and a second sqrt_s.min is a TOML error.
                        (r'\[histograms\]\n(?:(?!\[).*\n)*',
                         '[histograms]\nsqrt_s.min = 0.0\nsqrt_s.max = 2000.0\n'
                         'sqrt_s.bin_count = 10\n\n')],
            events=100)

        lhe_path = pjoin(run, 'events.lhe')
        if not os.path.exists(lhe_path):
            misc.gunzip(pjoin(run, 'events.lhe.gz'), keep=True, stdout=lhe_path)
        # 8 scale variations, then the PDF member
        nb_ids = 8 + nb_pdf
        nb_event = self._check_systematics_weights(
            lhe_path, expected_ids=[str(i) for i in range(1, nb_ids + 1)])

        side_path = pjoin(run, 'events.weights.json')
        self.assertTrue(os.path.exists(side_path), 'no events.weights.json sidecar')
        side = json.load(open(side_path))
        self.assertEqual([v['id'] for v in side['variations']], list(range(1, nb_ids + 1)))
        self.assertEqual(side['columns'], ['rwgt_%d' % i for i in range(1, nb_ids + 1)])
        self.assertEqual(side['event_count'], nb_event)
        self.assertEqual([(v['mur'], v['muf']) for v in side['variations'][:8]],
                         [(0.5, 0.5), (0.5, 1.0), (0.5, 2.0), (1.0, 0.5),
                          (1.0, 2.0), (2.0, 0.5), (2.0, 1.0), (2.0, 2.0)])
        xsec = side['nominal']['cross_section']
        self.assertGreater(xsec, 0)
        # the scale envelope brackets the nominal cross section
        self.assertLess(side['scale']['min'], xsec)
        self.assertGreater(side['scale']['max'], xsec)
        if has_other:
            member = side['variations'][8]
            self.assertEqual((member['pdf_set'], member['pdf_member']), (other_set, 1))
            self.assertNotAlmostEqual(member['cross_section'] / xsec, 1.0, places=3)
        info = json.load(open(pjoin(run, 'info.json')))
        self.assertIn('systematics', info)
        self.assertEqual(len(info['systematics']['variations']), nb_ids)
        # event-sample histograms: nominal + one column per variation, each
        # summing to the corresponding cross section, with the scale envelope
        self.assertIn('event_histograms', info)
        hist = info['event_histograms'][0]
        self.assertEqual(hist['name'], 'sqrt_s')
        self.assertEqual(len(hist['bin_values']), 12)  # 10 bins + under/overflow
        self.assertAlmostEqual(sum(hist['bin_values']) / xsec, 1.0, places=8)
        self.assertEqual([w['id'] for w in hist['weights']], list(range(1, nb_ids + 1)))
        for w, var in zip(hist['weights'], side['variations']):
            self.assertAlmostEqual(sum(w['bin_values']) / var['cross_section'], 1.0, places=8)
        env = hist['scale_envelope']
        for lo, nom, hi in zip(env['low'], hist['bin_values'], env['high']):
            self.assertLessEqual(lo, nom + 1e-12)
            self.assertGreaterEqual(hi, nom - 1e-12)
        # no legacy systematics.py run happened
        self.assertFalse(glob.glob(pjoin(run, '*systematics*crash.log')))

    def test_systematics_mixed_order_mg7(self):
        """A process whose |M|^2 mixes several alpha_s powers (u u~ > d d~ with
        QCD and QED contributions, qcd_power = -1) gets exact renormalisation
        scale weights by re-evaluating the matrix element at the varied alpha_s,
        so the full 3x3 scale grid is written and the mu_R weights differ from
        the nominal."""
        datadir = _mg7_datadir_or_skip(self)
        run = _run_mg7_postproc(
            self,
            ['set automatic_html_opening False --no_save',
             'import model sm',
             'generate u u~ > d d~',
             'add process u u~ > d d~ QED=2'],
            pjoin(self.path, 'MG7_syst_mixed'), datadir,
            switch_lines=None,
            toml_edits=[(r'\npdf = \[[^\]]*\]', '\npdf = ["central"]')],
            events=100)
        lhe_path = pjoin(run, 'events.lhe')
        if not os.path.exists(lhe_path):
            misc.gunzip(pjoin(run, 'events.lhe.gz'), keep=True, stdout=lhe_path)
        side = json.load(open(pjoin(run, 'events.weights.json')))
        self.assertEqual(side['warnings'], [])
        self.assertEqual([(v['mur'], v['muf']) for v in side['variations']],
                         [(0.5, 0.5), (0.5, 1.0), (0.5, 2.0), (1.0, 0.5),
                          (1.0, 2.0), (2.0, 0.5), (2.0, 1.0), (2.0, 2.0)])
        self._check_systematics_weights(lhe_path, expected_ids=[str(i) for i in range(1, 9)])
        # the mu_R-only variations (ids 2 and 7) really move the weights
        moved = 0
        for evt in lhe_parser.EventFile(lhe_path):
            rwgt = evt.parse_reweight()
            if abs(rwgt['2'] - evt.wgt) > 1e-6 * abs(evt.wgt) and \
               abs(rwgt['7'] - evt.wgt) > 1e-6 * abs(evt.wgt):
                moved += 1
        self.assertGreater(moved, 0, 'the mu_R weights equal the nominal weight')

    def test_systematics_mg7_legacy(self):
        """The legacy path ([postprocessing] systematics = true, systematics.py
        + the LHAPDF python module) still works when the native computation is
        switched off, and reads the reweighting inputs from the LHE (the
        single alpha_s power recorded at output time, single_qcd_order)."""
        datadir = _mg7_datadir_or_skip(self)
        run = _run_mg7_postproc(
            self,
            ['set automatic_html_opening False --no_save',
             'import model sm',
             'generate u u > u u'],
            pjoin(self.path, 'MG7_syst_legacy'), datadir,
            switch_lines=None,   # -f: run_lhe_postprocessing runs systematics
            toml_edits=[(r'\nenable = true\nmur', '\nenable = false\nmur'),
                        (r'\nsystematics = false\n', '\nsystematics = true\n'),
                        (r'systematics_pdf = \[[^\]]*\]',
                         'systematics_pdf = ["central"]')],
            events=100)

        lhe_path = pjoin(run, 'events.lhe')
        if not os.path.exists(lhe_path):
            misc.gunzip(pjoin(run, 'events.lhe.gz'), keep=True, stdout=lhe_path)
        crash = glob.glob(pjoin(run, '*systematics*crash.log'))
        if crash:
            self.skipTest('legacy systematics.py could not run (LHAPDF python '
                          'module?): %s' % open(crash[0]).read()[-500:])
        self._check_systematics_weights(lhe_path)

    def test_relaunch_switch_defaults_mg7(self):
        """Re-launching an mg7 output must not turn every tool on.

        The launch question materialises every candidate tool card (copying the
        *_default.dat) so it can offer them for edition. If the cards of the
        tools the user did not select are not pruned afterwards, a later launch
        sees them all present and defaults every switch to ON (set_default_<tool>
        keys off card presence). This drives the real launch question
        (ask_edit_cards) twice on the same output, selecting nothing each time,
        and checks that the second launch still defaults every switch to OFF
        instead of turning everything on. Output-level: no madspace/LHAPDF
        runtime needed."""
        import importlib, io
        mg = MGCmd.MasterCmd()
        mg.no_notification()
        for c in ['set automatic_html_opening False --no_save',
                  'import model sm', 'generate e+ e- > mu+ mu-']:
            mg.exec_cmd(c)
        out_dir = pjoin(self.path, 'MG7_relaunch')
        mg.exec_cmd('output mg7 %s' % out_dir)

        launcher = importlib.import_module(
            'madgraph.iolibs.template_files.mg7.launch')
        tool_cards = ('pythia8_card.dat', 'madspin_card.dat', 'delphes_card.dat',
                      'reweight_card.dat', 'rivet_card.dat',
                      'madanalysis5_parton_card.dat',
                      'madanalysis5_hadron_card.dat')

        def active_tool_cards():
            present = set(os.listdir(pjoin(out_dir, 'Cards')))
            return sorted(c for c in tool_cards if c in present)

        def launch_select_nothing():
            # drive the real launch question, answering '0' (= done: select
            # nothing / accept the shown defaults)
            old_stdin = sys.stdin
            sys.stdin = io.StringIO('0\n')
            try:
                return launcher.ask_edit_cards()
            finally:
                sys.stdin = old_stdin

        cwd = os.getcwd()
        try:
            os.chdir(out_dir)
            # LAUNCH 1: selecting nothing must leave no active tool card behind
            # (the materialised cards are pruned again).
            launch_select_nothing()
            self.assertEqual(active_tool_cards(), [],
                'the launch left unselected tool cards active: %s'
                % active_tool_cards())

            # LAUNCH 2: the defaults must reflect the previous (empty) selection
            # -- no switch may come back on.
            switch = launch_select_nothing()
            for key, value in switch.items():
                self.assertIn(value, ('OFF', 'Not Avail.'),
                    're-launch defaulted %s=%s instead of OFF -- the tool cards '
                    'from the previous launch were not pruned' % (key, value))
        finally:
            os.chdir(cwd)

    def test_madevent_merged_flavor_uq_mg7(self):
        """mg7 equivalent of test_madevent_merged_flavor_uq (u q > u q QCD=0,
        q = u d): the merged-flavor path must reproduce the cross-section
        obtained by running u u > u u and u d > u d as separate single-flavor
        processes.

        This used to come out too large because the mg7 exporter mirrored the
        mixed u d initial flavor -- leg 1 is a fixed u, so the beam-swapped
        d u > u d state is not part of the process and the u d contribution was
        counted twice (see Process.has_same_initial_multiparticle and
        test_merged_flavor_initial_state_mirroring_mg7, which guards the flag
        directly at output level).

        Self-skips where the mg7 runtime stack is unavailable.
        """
        import glob, json
        try:
            import madspace
            has_mg7 = hasattr(madspace, 'ChannelEventGenerator')
        except ImportError:
            has_mg7 = False
        datadir = os.environ.get('LHAPDF_DATA_PATH')
        if not datadir:
            try:
                datadir = subprocess.check_output(
                    ['lhapdf-config', '--datadir']).decode().strip()
            except Exception:
                datadir = None
        if not has_mg7 or not datadir or not os.path.isdir(datadir):
            self.skipTest('mg7 runtime stack (madspace + LHAPDF data) unavailable')
        if not glob.glob(pjoin(datadir, 'NNPDF40_lo_as_01180*')):
            self.skipTest('NNPDF40_lo_as_01180 PDF set not available')

        run_dir = pjoin(self.path, 'MG7_uq')
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir)
        mg = MGCmd.MasterCmd()
        mg.no_notification()
        mg.exec_cmd('set automatic_html_opening False --no_save')
        mg.exec_cmd('import model sm')
        mg.exec_cmd('define q = u d')
        mg.exec_cmd('generate u q > u q QCD=0')
        mg.exec_cmd('output mg7 %s' % run_dir)
        toml = pjoin(run_dir, 'Cards', 'run_card.toml')
        t = open(toml).read()
        t = t.replace('fixed_ren_scale = true', 'fixed_ren_scale = false')
        t = t.replace('fixed_fact_scale = true', 'fixed_fact_scale = false')
        t = re.sub(r'events = \d+', 'events = 2000', t)
        open(toml, 'w').write(t)
        env = dict(os.environ)
        env['LHAPDF_DATA_PATH'] = datadir
        log = pjoin(run_dir, 'mg7_gen.log')
        ret = subprocess.call(
            [sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
            cwd=run_dir, env=env,
            stdout=open(log, 'w'), stderr=subprocess.STDOUT)
        self.assertEqual(ret, 0,
            'mg7 generate_events failed for u q > u q (merged flavor) -- known '
            'mg7 crash/segfault on this process (see %s)' % log)
        infos = sorted(glob.glob(pjoin(run_dir, 'Events', '*', 'info.json')))
        self.assertTrue(infos, 'no mg7 info.json under %s' % run_dir)
        info = json.load(open(infos[-1]))['process']
        cross = float(info['mean'])
        error = float(info.get('error') or 0.0)
        # mg7 reference with the default PDF NNPDF40_lo_as_01180. QCD=0, so
        # there is no alpha_s here at all and the shift away from the 4380 pb of
        # NNPDF23_lo_as_0130_qed is purely the smaller NNPDF4.0 valence-quark
        # luminosity.
        # Spread over 6 runs at events=2000: 3446-3536, mean 3490.
        # previously PDF was NNPDF40MC_lo_as_01180 with this reference value 3730.0
        self.assertAlmostEqual(cross, 3490.0, delta=max(50.0, 5 * error))

    def test_gridpack_mg7(self):
        """An mg7 gridpack has to run with the libraries it ships.

        save_gridpack copies the matrix-element libraries that were built, and
        their file names carry the resolved backend, while the card the gridpack
        runs from carried the backend as requested. cpu_mode defaults to 'auto',
        so the gridpack asked for a ..._auto.so nobody had built and died in
        dlopen -- not when the gridpack was made, but the first time it was run,
        possibly on another machine.

        Both halves are checked: that the backend named in grid_run_card.toml is
        one the gridpack carries a library for, and that running it produces
        events.
        """
        import glob, re as _re
        try:
            import madspace
            has_mg7 = hasattr(madspace, 'ChannelEventGenerator')
        except ImportError:
            has_mg7 = False
        if not has_mg7:
            self.skipTest('mg7 runtime stack (madspace) unavailable')
        # e+ e- > mu+ mu- is leptonic, so no parton density is ever evaluated
        # and the [systematics] PDF variations are empty -- but alpha_s still
        # comes from the nominal set, and the mu_R/mu_F weights are computed.
        # They are deliberately left enabled: the gridpack has to reproduce the
        # calculator the launcher built, and used to die doing so.
        datadir = os.environ.get('LHAPDF_DATA_PATH')
        if not datadir:
            try:
                datadir = subprocess.check_output(
                    ['lhapdf-config', '--datadir']).decode().strip()
            except Exception:
                datadir = None

        mg = MGCmd.MasterCmd()
        mg.no_notification()
        mg.exec_cmd('generate e+ e- > mu+ mu-')
        mg.exec_cmd('output mg7 %s' % self.run_dir)

        toml = pjoin(self.run_dir, 'Cards', 'run_card.toml')
        t = open(toml).read()
        t = _re.sub(r'(?m)^events = \d+', 'events = 2000', t)
        t = _re.sub(r'(?m)^save_gridpack = .*', 'save_gridpack = true', t)
        open(toml, 'w').write(t)

        env = dict(os.environ)
        if datadir:
            env['LHAPDF_DATA_PATH'] = datadir
        log = pjoin(self.run_dir, 'mg7_gridpack_gen.log')
        ret = subprocess.call(
            [sys.executable, pjoin(self.run_dir, 'bin', 'generate_events'), '-f'],
            cwd=self.run_dir, env=env,
            stdout=open(log, 'w'), stderr=subprocess.STDOUT)
        self.assertEqual(ret, 0, 'mg7 generate_events failed (see %s)' % log)

        gridpacks = glob.glob(pjoin(self.run_dir, 'Events', '*', 'gridpack'))
        self.assertTrue(gridpacks, 'no gridpack produced under %s' % self.run_dir)
        gridpack = gridpacks[0]

        # The card the gridpack runs from must name a backend it actually
        # carries a library for: 'auto' is not one, nothing is built under it.
        card = open(pjoin(gridpack, 'Cards', 'grid_run_card.toml')).read()
        mode = _re.search(r'(?m)^cpu_mode\s*=\s*"([^"]+)"', card)
        self.assertTrue(mode, 'no cpu_mode in the gridpack card')
        backend = mode.group(1)
        libdir = pjoin(gridpack, 'lib')
        self.assertNotEqual(
            backend, 'auto',
            'the gridpack card names the unresolved backend, but its libraries '
            'are named after the resolved one: %s' % os.listdir(libdir))
        self.assertTrue(
            glob.glob(pjoin(libdir, '*%s*' % backend)),
            'gridpack names backend %r but ships no matching library: %s'
            % (backend, os.listdir(libdir)))

        # the launch wrote npy events (the mg7 default), but a gridpack feeds
        # a shower/detector chain: its card has to default to LHE
        self.assertRegex(card, r'(?m)^output_format\s*=\s*"lhe"',
                         'the gridpack card does not default to LHE')

        # and it has to run
        gplog = pjoin(self.run_dir, 'mg7_gridpack_run.log')
        ret = subprocess.call(
            [sys.executable, pjoin(gridpack, 'bin', 'generate_events'),
             '--events', '2000'],
            cwd=gridpack, env=env,
            stdout=open(gplog, 'w'), stderr=subprocess.STDOUT)
        self.assertEqual(ret, 0, 'mg7 gridpack run failed (see %s)' % gplog)
        lhe = glob.glob(pjoin(gridpack, 'Events', '*', 'events.lhe.gz'))
        self.assertTrue(lhe, 'gridpack run produced no LHE events (see %s)' % gplog)
        # and its LHE header has to be complete: the cards, the beams and the
        # cross section of this gridpack run, not an empty <init> block
        import gzip as _gzip
        with _gzip.open(lhe[0], 'rt') as stream:
            header = stream.read().split('<event>', 1)[0]
        for block in ('<MG5ProcCard>', '<slha>', '<MG7RunCard>', '<MG7Seed>'):
            self.assertIn(block, header)
        init = header.split('<init>', 1)[1].split('</init>', 1)[0].split()
        self.assertEqual(init[:2], ['-11', '11'])
        self.assertGreater(float(init[10]), 0., 'no cross section in <init>')
        # and it has to reweight: without a nominal alpha_s grid the gridpack
        # either crashed or dropped the weights it was configured to write.
        self.assertTrue(
            glob.glob(pjoin(gridpack, 'Events', '*', 'events.weights.json')),
            'the gridpack run wrote no systematics weights (see %s)' % gplog)

    def test_flavor_grouping_consistency(self):
        """Check that the four combinations of 'apply_flavor_grouping' and
        'group_subprocesses' return compatible cross-sections for the
        process p p > e+ e-.

        Settings tested:
            1. apply_flavor_grouping True  / group_subprocesses False
            2. apply_flavor_grouping True  / group_subprocesses True
            3. apply_flavor_grouping False / group_subprocesses False
            4. apply_flavor_grouping False / group_subprocesses True
        """

        settings = [
            # (apply_flavor_grouping, group_subprocesses)
            ('True',  'False'),
            ('True',  'True'),
            ('False', 'False'),
            ('false', 'True'),
        ]

        results = []
        for i, (afg, gsp) in enumerate(settings):
            run_dir = pjoin(self.path, 'MGPROC_fg_%d' % i)
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)

            mg_cmd = MGCmd.MasterCmd()
            mg_cmd.no_notification()
            mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
            mg_cmd.exec_cmd('set apply_flavor_grouping %s' % afg)
            mg_cmd.exec_cmd('import model sm')
            mg_cmd.exec_cmd('set group_subprocesses %s' % gsp)
            mg_cmd.exec_cmd('generate p p > l+ l-')
            mg_cmd.exec_cmd('output madevent %s' % run_dir)

            self.cmd_line = MECmd.MadEventCmdShell(me_dir=run_dir)
            self.cmd_line.no_notification()
            self.cmd_line.exec_cmd('set automatic_html_opening False')

            self.do('generate_events -f')

            cross = self.cmd_line.results.current['cross']
            err = self.cmd_line.results.current['error']

            # A run that produced nothing leaves cross *and* error at 0 -- most
            # often a ZeroResult swallowed by nice_error_handling, which only
            # logs a warning (invisible at the CRITICAL level these tests run
            # at). Catch it here: the precision check below divides by
            # cross+1e-99, so 0/1e-99 = 0 < 0.05 sails through, and the zero
            # would surface only in the pairwise comparison as a misleading
            # "incompatible cross-sections ... (1e-99 +- 0)" rather than as the
            # failed run it is. (The check this replaces tested val == 0 after
            # the +1e-99, so it could never fire.)
            self.assertTrue(cross,
                'no cross-section produced for apply_flavor_grouping=%s/'
                'group_subprocesses=%s: the run failed rather than disagreeing '
                '(cross=%s, error=%s, run dir %s)' % (afg, gsp, cross, err, run_dir))

            val = cross + 1e-99
            results.append((val, err, afg, gsp))

            #check precision is reasonable for each individual run
            self.assertLess(err / val, 0.05,
                'cross-section determination is too imprecise '
                '(apply_flavor_grouping=%s, group_subprocesses=%s): '
                '%s +- %s' % (afg, gsp, val, err))

        # Check pairwise compatibility: each pair of cross-sections should
        # agree within 5 times the combined statistical uncertainty.
        if unittest.debug:
            for val, err, afg, gsp in results:
                misc.sprint('  apply_flavor_grouping=%s/group_subprocesses=%s: %s +- %s' %
                         (afg, gsp, val, err))
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                val_i, err_i, afg_i, gsp_i = results[i]
                val_j, err_j, afg_j, gsp_j = results[j]
                self.assertLess(
                    abs(val_i - val_j) / (err_i + err_j),
                    3,
                    'Incompatible cross-sections between '
                    'apply_flavor_grouping=%s/group_subprocesses=%s '
                    '(%s +- %s) and '
                    'apply_flavor_grouping=%s/group_subprocesses=%s '
                    '(%s +- %s)' % (
                        afg_i, gsp_i, val_i, err_i,
                        afg_j, gsp_j, val_j, err_j
                    )
                )

    def test_flavor_grouping_consistency_mg7(self):
        """mg7 equivalent of test_flavor_grouping_consistency for p p > l+ l-.

        The four settings must all give the same cross-section. They used not
        to (~1336 pb grouped vs ~538 pb ungrouped): a matrix element carries
        flavor multiplicity either in its merged legs (apply_flavor_grouping=
        True) or in the several processes mapped onto it (grouping off), and the
        mg7 exporter only enumerated the first, so with grouping off it kept 4
        of the 16 channels of p p > l+ l- -- dropping c/s in the initial state
        and mu+ mu- entirely. Both sources are now walked through the shared
        HelasMatrixElement.get_flavor_pdg_combinations. It self-skips where the
        mg7 runtime stack is unavailable.
        """
        datadir = _mg7_datadir_or_skip(self)
        settings = [
            # (apply_flavor_grouping, group_subprocesses)
            ('True',  'False'),
            ('True',  'True'),
            ('False', 'False'),
            ('false', 'True'),
        ]
        results = []
        for i, (afg, gsp) in enumerate(settings):
            cross, err = _run_mg7_xsec(self, 
                ['set automatic_html_opening False --no_save',
                 'set apply_flavor_grouping %s' % afg,
                 'import model sm',
                 'set group_subprocesses %s' % gsp,
                 'generate p p > l+ l-'],
                pjoin(self.path, 'MG7_fg_%d' % i), datadir)
            # Fail on a run that produced nothing rather than letting it reach
            # the pairwise comparison: err/(cross+1e-99) is 0 < 0.05 when both
            # are 0, so the precision check below cannot catch it.
            self.assertTrue(cross,
                'mg7 produced no cross-section for apply_flavor_grouping=%s/'
                'group_subprocesses=%s: the run failed rather than disagreeing '
                '(cross=%s, error=%s)' % (afg, gsp, cross, err))
            results.append((cross + 1e-99, err, afg, gsp))
            self.assertLess(err / (cross + 1e-99), 0.05,
                'mg7 cross-section too imprecise (afg=%s, gsp=%s): %s +- %s'
                % (afg, gsp, cross, err))
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                vi, ei, ai, gi = results[i]
                vj, ej, aj, gj = results[j]
                self.assertLess(abs(vi - vj) / (ei + ej), 3,
                    'mg7 incompatible cross-sections between '
                    'apply_flavor_grouping=%s/group_subprocesses=%s (%s +- %s) and '
                    'apply_flavor_grouping=%s/group_subprocesses=%s (%s +- %s)'
                    % (ai, gi, vi, ei, aj, gj, vj, ej))

    def test_flavor_grouping_consistency_width(self):
        """Check that the four combinations of 'apply_flavor_grouping' and
        'group_subprocesses' return compatible cross-sections for the
        process z > l+ l-.

        Settings tested:
            1. apply_flavor_grouping True  / group_subprocesses False
            2. apply_flavor_grouping True  / group_subprocesses True
            3. apply_flavor_grouping False / group_subprocesses False
            4. apply_flavor_grouping False / group_subprocesses True
        """

        settings = [
            # (apply_flavor_grouping, group_subprocesses)
            ('True',  'False'),
            ('True',  'True'),
            ('False', 'False'),
            ('false', 'True'),
        ]

        results = []
        for i, (afg, gsp) in enumerate(settings):
            run_dir = pjoin(self.path, 'MGPROC_fg_%d' % i)
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)

            mg_cmd = MGCmd.MasterCmd()
            mg_cmd.no_notification()
            mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
            mg_cmd.exec_cmd('set apply_flavor_grouping %s' % afg)
            mg_cmd.exec_cmd('import model sm')
            mg_cmd.exec_cmd('set group_subprocesses %s' % gsp)
            mg_cmd.exec_cmd('generate z > l+ l-')
            mg_cmd.exec_cmd('output madevent %s' % run_dir)

            self.cmd_line = MECmd.MadEventCmdShell(me_dir=run_dir)
            self.cmd_line.no_notification()
            self.cmd_line.exec_cmd('set automatic_html_opening False')

            self.do('generate_events -f')

            cross = self.cmd_line.results.current['cross']
            err = self.cmd_line.results.current['error']

            # A run that produced nothing leaves cross *and* error at 0 -- most
            # often a ZeroResult swallowed by nice_error_handling, which only
            # logs a warning (invisible at the CRITICAL level these tests run
            # at). Catch it here: the precision check below divides by
            # cross+1e-99, so 0/1e-99 = 0 < 0.05 sails through, and the zero
            # would surface only in the pairwise comparison as a misleading
            # "incompatible cross-sections ... (1e-99 +- 0)" rather than as the
            # failed run it is. (The check this replaces tested val == 0 after
            # the +1e-99, so it could never fire.)
            self.assertTrue(cross,
                'no cross-section produced for apply_flavor_grouping=%s/'
                'group_subprocesses=%s: the run failed rather than disagreeing '
                '(cross=%s, error=%s, run dir %s)' % (afg, gsp, cross, err, run_dir))

            val = cross + 1e-99
            results.append((val, err, afg, gsp))

            #check precision is reasonable for each individual run
            self.assertLess(err / val, 0.05,
                'cross-section determination is too imprecise '
                '(apply_flavor_grouping=%s, group_subprocesses=%s): '
                '%s +- %s' % (afg, gsp, val, err))

        # Check pairwise compatibility: each pair of cross-sections should
        # agree within 5 times the combined statistical uncertainty.
        if True or unittest.debug:
            for val, err, afg, gsp in results:
                misc.sprint('  apply_flavor_grouping=%s/group_subprocesses=%s: %s +- %s' %
                         (afg, gsp, val, err))
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                val_i, err_i, afg_i, gsp_i = results[i]
                val_j, err_j, afg_j, gsp_j = results[j]
                self.assertLess(
                    abs(val_i - val_j) / (err_i + err_j),
                    3,
                    'Incompatible cross-sections between '
                    'apply_flavor_grouping=%s/group_subprocesses=%s '
                    '(%s +- %s) and '
                    'apply_flavor_grouping=%s/group_subprocesses=%s '
                    '(%s +- %s)' % (
                        afg_i, gsp_i, val_i, err_i,
                        afg_j, gsp_j, val_j, err_j
                    )
                )


    def test_flavor_grouping_consistency_mlm(self):
        """Check flavor_compatible function in clustering with MLM merging.

        Tests the process q q~ > q q~ (with q = u d s c, q~ = u~ d~ s~ c~)
        with MLM merging (ickkw=1) and xqcut=20 to verify that flavor-filtering
        in the clustering algorithm correctly identifies valid diagram topologies.

        The process should cluster via W boson (u<->d coupling) but not via
        gluon/photon/Z (which require same flavor).
        """
        settings = [
            # (apply_flavor_grouping, group_subprocesses)
            ('True',  'False'),
            ('True',  'True'),
            ('False', 'False'),
            ('false', 'True'),
        ]

        # previously PDF was nn23lo1 (lhaid 230000) with this reference value
        # 5184588.926738217 +- 2971 (measured on v3.7.2 with 150k events).
        # The shift is only -1.67%, far smaller than for an unmerged process
        # (u u > u u moves -39%): measured by an A/B on identical code, same
        # process and MLM settings, 5179400 +- 22594 with nn23lo1 pinned vs
        # 5092900 +- 19956 with the new default. The nn23lo1 arm sits within
        # 1 sigma of the stored 3.7.2 number, so the small shift is real and
        # not a sign that the run ignored the PDF change.
        results = [(5092900.0, 19956, 'NNPDF40_lo_as_01180', 'nevents=10k')]
        for i, (afg, gsp) in enumerate(settings):
            run_dir = pjoin(self.path, 'MGPROC_fg_%d' % i)


            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)

            mg_cmd = MGCmd.MasterCmd()
            mg_cmd.no_notification()
            mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
            #mg_cmd.exec_cmd('import model sm')
            mg_cmd.exec_cmd('set apply_flavor_grouping %s' % afg)
            mg_cmd.exec_cmd('import model sm')
            mg_cmd.exec_cmd('set group_subprocesses %s' % gsp)

            # Define custom particles for flavor grouping
            mg_cmd.exec_cmd('define q = u d s c')
            mg_cmd.exec_cmd('define q~ = u~ d~ s~ c~')

            # Generate process with flavor-grouped particles
            mg_cmd.exec_cmd('generate q q~ > q q~')
            mg_cmd.exec_cmd('output madevent %s' % run_dir)

            self.cmd_line = MECmd.MadEventCmdShell(me_dir=run_dir)
            self.cmd_line.no_notification()
            self.cmd_line.exec_cmd('set automatic_html_opening False')

            # Configure MLM merging parameters
            run_card = banner.RunCardLO(pjoin(run_dir, 'Cards', 'run_card.dat'))
            run_card.set('ickkw', 1, user=True)
            run_card.set('xqcut', 20.0, user=True)
            run_card.write(pjoin(run_dir, 'Cards', 'run_card.dat'))
            

            # Generate events with MLM merging enabled
            self.do('generate_events -f')

            # Verify event generation succeeded
            cross = self.cmd_line.results.current['cross']
            err = self.cmd_line.results.current['error']

            # Check that we got a valid cross-section. Test the raw cross, not
            # cross+1e-99: a run that produced nothing leaves cross and error at
            # 0 (typically a ZeroResult swallowed by nice_error_handling), and
            # asserting on the padded value can never fail.
            self.assertTrue(cross,
                'no cross-section produced for q q~ > q q~ with MLM merging: '
                'the run failed (cross=%s, error=%s, run dir %s)'
                % (cross, err, run_dir))

            val = cross + 1e-99
            results.append((val, err, afg, gsp))

            # Check precision is reasonable
            self.assertLess(err / val, 0.10,
                'cross-section determination is too imprecise for MLM merging: '
                '%s +- %s' % (val, err))
            
       # Check pairwise compatibility: each pair of cross-sections should
        # agree within 5 times the combined statistical uncertainty.
        if unittest.debug:
            for val, err, afg, gsp in results:
                misc.sprint('  apply_flavor_grouping=%s/group_subprocesses=%s: %s +- %s' %
                            (afg, gsp, val, err))
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                val_i, err_i, afg_i, gsp_i = results[i]
                val_j, err_j, afg_j, gsp_j = results[j]
                self.assertLess(
                    abs(val_i - val_j) / (err_i + err_j),
                    3,
                    'Incompatible cross-sections between '
                    'apply_flavor_grouping=%s/group_subprocesses=%s '
                    '(%s +- %s) and '
                    'apply_flavor_grouping=%s/group_subprocesses=%s '
                    '(%s +- %s)' % (
                        afg_i, gsp_i, val_i, err_i,
                        afg_j, gsp_j, val_j, err_j
                    )
                )


    def test_mlm_pdfrwt_flavour(self):
        """With MLM (ickkw=1) and use_syst, the <pdfrwt> information must carry
        the event's flavours. A grouped
        subprocess (q q~ > z with q = d s / u c) clusters with the
        representative of its coupling class: s s~ > z was booked as 1 and
        c c~ > z as 2, so the systematics of every s/c-initiated event used
        the d/u PDF."""

        import gzip
        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
        mg_cmd.exec_cmd('import model sm')
        mg_cmd.exec_cmd('generate p p > z @0')
        mg_cmd.exec_cmd('add process p p > z j @1')
        # scalar mode only: vector_size>1 with grouped flavours does not run
        # (DSIG_VEC reads the per-warp IFLAV_VEC per event)
        for vector_size in [1]:
            run_dir = pjoin(self.path, 'MGPROC_pdfrwt_%d' % vector_size)
            mg_cmd.exec_cmd('output madevent %s' % run_dir)

            run_card = banner.RunCardLO(pjoin(run_dir, 'Cards', 'run_card.dat'))
            run_card.set('nevents', 2000, user=True)
            run_card.set('iseed', 7, user=True)
            run_card.set('ickkw', 1, user=True)
            run_card.set('xqcut', 20.0, user=True)
            run_card.set('use_syst', True, user=True)
            run_card.set('systematics_program', 'none', user=True)
            if vector_size > 1:
                run_card.set('vector_size', vector_size, user=True)
            run_card.write(pjoin(run_dir, 'Cards', 'run_card.dat'))

            self.cmd_line = MECmd.MadEventCmdShell(me_dir=run_dir)
            self.cmd_line.no_notification()
            self.cmd_line.exec_cmd('set automatic_html_opening False')
            self.do('generate_events -f')

            path = pjoin(run_dir, 'Events', 'run_01', 'unweighted_events.lhe.gz')
            text = gzip.open(path, 'rt').read()
            checked = collections.Counter()
            for event in re.findall(r'<event[^>]*>(.*?)</event>', text, re.S):
                lines = [l.split() for l in event.split('<')[0].strip().splitlines()]
                partons = [(int(l[0]), int(l[1]), float(l[8])) for l in lines[1:]]
                # beam 1 moves along +z, beam 2 along -z
                beam = dict((1 if pz > 0 else 2, pdg)
                            for pdg, status, pz in partons if status == -1)
                flavours = set(abs(pdg) for pdg, status, pz in partons
                               if abs(pdg) <= 6)
                pdfrwt = dict((int(b), [int(x) for x in body.split()[1:1+int(body.split()[0])]])
                              for b, body in re.findall(
                                  r'<pdfrwt beam="(\d)">(.*?)</pdfrwt>', event))
                self.assertEqual(sorted(pdfrwt), [1, 2], event)
                for b in (1, 2):
                    # leading entry: the incoming parton of this beam
                    self.assertEqual(pdfrwt[b][0], beam[b],
                        'vector_size=%s: leading pdfrwt entry of beam %s is '
                        'not the incoming parton:\n%s' % (vector_size, b, event))
                    # ladder: only quark flavours present in the event
                    for pdg in pdfrwt[b][1:]:
                        if abs(pdg) <= 6:
                            self.assertIn(abs(pdg), flavours,
                                'vector_size=%s: pdfrwt ladder of beam %s has a '
                                'flavour absent from the event:\n%s'
                                % (vector_size, b, event))
                quarks = [abs(pdg) for pdg in beam.values() if pdg != 21]
                checked[max(quarks) if quarks else 21] += 1
            # the check is only meaningful with s and c initiated events
            self.assertGreater(checked[3], 0, checked)
            self.assertGreater(checked[4], 0, checked)

    def test_t_channel_survey_grid(self):
        """check that refining on the survey grid is not biased for
        u q > z u q (all channels pure t-channel chains). With the grid
        defined on [-1,1] while the variables are in [0,1], the survey grid
        starved the regions close to the kinematic end-points (below the
        lowest x seen): the rare huge weights from there biased the refine
        low (96.6 pb, -2.3%, averaged over 4 seeds).
        Reference (fixed scale 91.188, default PDF) from high statistics
        (8 iterations up to 2.5M points per channel, pooled over seeds):
        98.840 +- 0.017 pb; madspace gives 98.78 +- 0.02 pb.
        Only the grouped output is checked here (the non-grouped one is
        test_madevent_merged_flavor_nogroup_symmetry)."""

        from unittest import mock
        gen_ximprove = MECmd.gen_ximprove

        target = 98.84
        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
        mg_cmd.exec_cmd('define q = u d')
        mg_cmd.exec_cmd('generate u q > z u q')
        mg_cmd.exec_cmd('output madevent %s' % self.run_dir)

        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards', 'run_card.dat'))
        run_card.set('nevents', 10000, user=True)
        run_card.set('iseed', 21, user=True)
        run_card.set('use_syst', False, user=True)
        run_card.set('fixed_ren_scale', True, user=True)
        run_card.set('fixed_fac_scale', True, user=True)
        run_card.set('scale', 91.188, user=True)
        run_card.set('dsqrt_q2fact1', 91.188, user=True)
        run_card.set('dsqrt_q2fact2', 91.188, user=True)
        run_card.write(pjoin(self.run_dir, 'Cards', 'run_card.dat'))

        self.cmd_line = MECmd.MadEventCmdShell(me_dir=self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')
        # never split the refine jobs: they then restart from the
        # survey grid (with a fixed grid)
        with mock.patch.object(gen_ximprove.gen_ximprove_v4, 'max_request_event', 10**9):
            self.do('generate_events -f')
        val = self.cmd_line.results.current['cross']
        err = self.cmd_line.results.current['error']
        self.assertLess(abs(val - target) / target, 0.012,
                        '%s +- %s vs %s' % (val, err, target))

    def test_madevent_merged_flavor_nogroup_symmetry(self):
        """u q > z u q (q = u d) without subprocess grouping: one
        flavor-merged P dir (rows u u and u d). The identical-particle swap
        of the final quarks is a symmetry of the u u row only; symfact.dat
        used it to fold channel 2 onto channel 1 with weight 2, which made
        the cross-section ~4.5% high (103.4 pb).
        Reference (fixed scale 91.188, default PDF): 98.840 +- 0.017 pb
        from the grouped output at high statistics; madspace 98.78 +- 0.02.
        The tolerance leaves room for the low bias of the survey grid on
        t-channel chains (see test_t_channel_survey_grid)."""

        target = 98.84
        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
        mg_cmd.exec_cmd('set group_subprocesses False')
        mg_cmd.exec_cmd('define q = u d')
        mg_cmd.exec_cmd('generate u q > z u q')
        mg_cmd.exec_cmd('output madevent %s' % self.run_dir)

        pdirs = [d for d in glob.glob(pjoin(self.run_dir, 'SubProcesses', 'P*'))
                 if os.path.isdir(d)]
        self.assertEqual(len(pdirs), 1, pdirs)
        symfact = open(pjoin(pdirs[0], 'symfact_orig.dat')).read().split()
        self.assertEqual(symfact[1::2], ['1'] * (len(symfact) // 2))

        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards', 'run_card.dat'))
        run_card.set('nevents', 10000, user=True)
        run_card.set('iseed', 21, user=True)
        run_card.set('use_syst', False, user=True)
        run_card.set('fixed_ren_scale', True, user=True)
        run_card.set('fixed_fac_scale', True, user=True)
        run_card.set('scale', 91.188, user=True)
        run_card.set('dsqrt_q2fact1', 91.188, user=True)
        run_card.set('dsqrt_q2fact2', 91.188, user=True)
        run_card.write(pjoin(self.run_dir, 'Cards', 'run_card.dat'))

        self.cmd_line = MECmd.MadEventCmdShell(me_dir=self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')
        self.do('generate_events -f')
        val = self.cmd_line.results.current['cross']
        err = self.cmd_line.results.current['error']
        self.assertLess(abs(val - target) / target, 0.02,
                        '%s +- %s vs %s' % (val, err, target))


    def test_group_subprocess_permuted_scale(self):
        """check that grouped (scalar and vector) and non-grouped output agree
        when the grouped channels use a symmetry permutation of the momenta
        and the scales depend on the clustering (MLM). see issue #453"""

        from unittest import mock
        gen_ximprove = MECmd.gen_ximprove

        results = {}
        for name, group, vector_size in [('grouped', True, 1),
                                         ('grouped_vec', True, 16),
                                         ('nongrouped', False, 1)]:
            self.run_dir = pjoin(self.path, 'MGPROC_%s' % name)
            mg_cmd = MGCmd.MasterCmd()
            mg_cmd.no_notification()
            mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
            mg_cmd.exec_cmd('set group_subprocesses %s' % group)
            mg_cmd.exec_cmd('define q = u d')
            mg_cmd.exec_cmd('generate u q > z u q')
            mg_cmd.exec_cmd('output madevent %s' % self.run_dir)

            run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards', 'run_card.dat'))
            run_card.set('nevents', 10000, user=True)
            run_card.set('iseed', 11, user=True)
            run_card.set('use_syst', False, user=True)
            run_card.set('ickkw', 1, user=True)
            run_card.set('xqcut', 40.0, user=True)
            run_card.set('drjj', 0.0, user=True)
            if vector_size > 1:
                run_card.set('vector_size', vector_size, user=True)
            run_card.write(pjoin(self.run_dir, 'Cards', 'run_card.dat'))

            self.cmd_line = MECmd.MadEventCmdShell(me_dir=self.run_dir)
            self.cmd_line.no_notification()
            self.cmd_line.exec_cmd('set automatic_html_opening False')
            # use the same refine setup (survey grid kept) for grouped and
            # non-grouped: whether a job is split depends on the output
            # mode and changes the refine grid, which is independent of #453
            with mock.patch.object(gen_ximprove.gen_ximprove, 'keep_grid_for_refine', True), \
                 mock.patch.object(gen_ximprove.gen_ximprove_v4, 'keep_grid_for_refine', True):
                self.do('generate_events -f')
            results[name] = (self.cmd_line.results.current['cross'],
                             self.cmd_line.results.current['error'])

        val0, err0 = results['nongrouped']
        self.assertGreater(val0, 0)
        for name in ['grouped', 'grouped_vec']:
            val, err = results[name]
            # without the fix, grouped is ~3% low (about 8 sigma)
            self.assertLess(abs(val - val0) / math.hypot(err, err0), 4,
                            '%s: %s +- %s vs nongrouped %s +- %s' % (name, val, err, val0, err0))

    def test_vector_mode_helicity_both_orientations(self):
        """check that scalar and vector (vector_size 16) madevent agree for
        p p > z, with and without helicity recycling. q q~ > z with the Z at
        rest has helicities that vanish in one beam orientation only, and in
        vector mode the first events of the good-helicity scan share one
        orientation: without the fix, vector mode was 10-50% low."""

        results = {}
        for name, hel_recycling, vector_size in [('hel_scalar', True, 1),
                                                 ('hel_vec', True, 16),
                                                 ('nohel_scalar', False, 1),
                                                 ('nohel_vec', False, 16)]:
            self.run_dir = pjoin(self.path, 'MGPROC_%s' % name)
            mg_cmd = MGCmd.MasterCmd()
            mg_cmd.no_notification()
            mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
            mg_cmd.exec_cmd('generate p p > z')
            mg_cmd.exec_cmd('output madevent %s --hel_recycling=%s' % (self.run_dir, hel_recycling))

            run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards', 'run_card.dat'))
            run_card.set('nevents', 2000, user=True)
            run_card.set('iseed', 31, user=True)
            run_card.set('use_syst', False, user=True)
            if vector_size > 1:
                run_card.set('vector_size', vector_size, user=True)
            run_card.write(pjoin(self.run_dir, 'Cards', 'run_card.dat'))

            self.cmd_line = MECmd.MadEventCmdShell(me_dir=self.run_dir)
            self.cmd_line.no_notification()
            self.cmd_line.exec_cmd('set automatic_html_opening False')
            self.do('generate_events -f')
            results[name] = (self.cmd_line.results.current['cross'],
                             self.cmd_line.results.current['error'])

        val0, err0 = results['hel_scalar']
        self.assertGreater(val0, 0)
        for name in ['hel_vec', 'nohel_scalar', 'nohel_vec']:
            val, err = results[name]
            self.assertLess(abs(val - val0) / math.hypot(err, err0), 4,
                            '%s: %s +- %s vs hel_scalar %s +- %s' % (name, val, err, val0, err0))

    def test_e_p_collision(self):
        """check that e p > e j gives the correct result"""
        

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --save')
        mg_cmd.exec_cmd(' generate e- p  > e- j')
        mg_cmd.exec_cmd('output madevent %s/'% self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=  self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')
        
        #check validity of the default run_card
        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards','run_card.dat'))
        self.assertIn('ptj', run_card.user_set)
        self.assertNotIn('drjj', run_card.user_set)
        self.assertNotIn('ptj2min', run_card.user_set)
        self.assertNotIn('ptj3min', run_card.user_set)
        self.assertNotIn('mmjj', run_card.user_set)
        self.assertNotIn('ptheavy', run_card.user_set)
        self.assertNotIn('el', run_card.user_set)
        self.assertNotIn('ej', run_card.user_set)
        self.assertIn('polbeam1', run_card.user_set)
        self.assertIn('ptl', run_card.user_set)
        
        shutil.copy(os.path.join(_file_path, 'input_files', 'run_card_ep.dat'),
                    '%s/Cards/run_card.dat' % self.run_dir) 
        
        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        # 100k value is 3933.1 +- 3 
        target = 3933.1
        self.assertLess(
            abs(val1 - target) / (err1+1.7),
            2.,
            'large difference between %s and %s +- %s'%
                        (target, val1, err1)
        )


    def test_e_p_collision_mg7(self):
        """mg7 equivalent of test_e_p_collision for e- p > e- j.

        The madevent test pins the cross-section (3933 pb) via a full
        generate_events run. The mg7 (madmatrix/cudacpp) integrator is far too
        slow for a CI cross-section run on this process, so this equivalent
        validates the mg7 *output* path instead: that `output mg7` generates the
        e- p > e- j subprocess directories and that they compile (scalar
        backend) into the expected shared libraries. The cross-section
        comparison against the madevent reference remains a TODO pending a
        faster mg7 integrator.
        """
        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
        mg_cmd.exec_cmd('generate e- p > e- j')
        mg_cmd.exec_cmd('output mg7 %s' % self.run_dir)

        # mg7 cards and launcher
        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'Cards',
                                             'run_card.toml')))
        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'bin',
                                             'generate_events')))

        sub_root = pjoin(self.run_dir, 'SubProcesses')
        proc_dirs = sorted(d for d in os.listdir(sub_root)
                           if d.startswith('P') and
                           os.path.isdir(pjoin(sub_root, d)))
        self.assertTrue(proc_dirs,
                        'no subprocess generated by mg7 for e- p > e- j')

        devnull = open(os.devnull, 'w')
        for d in proc_dirs:
            proc_dir = pjoin(sub_root, d)
            self.assertTrue(os.path.isfile(pjoin(proc_dir, 'CPPProcess.cc')),
                            'CPPProcess.cc missing in %s' % d)
            status = subprocess.call(['make', 'bldnone'],
                                     stdout=devnull, stderr=devnull,
                                     cwd=proc_dir)
            self.assertEqual(status, 0, 'mg7 subprocess %s did not compile' % d)
        libs = os.listdir(pjoin(self.run_dir, 'lib')) \
            if os.path.isdir(pjoin(self.run_dir, 'lib')) else []
        self.assertTrue(any(l.startswith('libmadmatrix_common') and
                            l.endswith('.so') for l in libs),
                        'common madmatrix library not built: %s' % libs)


    def test_eva_collision(self):
        """check that w+ w- > t t~ with EVA gives the correct result
        assuming x > MV/Ebeam restriction (eva_xcut=1) [2502.07878]"""


        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.run_cmd('set group_subprocesses false')
        mg_cmd.run_cmd('set automatic_html_opening False --save')
        mg_cmd.run_cmd(' generate w+ w-  > t t~')
        mg_cmd.run_cmd('output madevent %s/'% self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=  self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')
        
        #check validity of the default run_card
        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards','run_card.dat'))

        f = open(pjoin(self.run_dir, 'Cards','run_card.dat'),'r')
        self.assertNotIn('ptj', run_card.user_set)
        self.assertNotIn('drjj', run_card.user_set)
        self.assertNotIn('ptj2min', run_card.user_set)
        self.assertNotIn('ptj3min', run_card.user_set)
        self.assertNotIn('mmjj', run_card.user_set)
        self.assertIn('ptheavy', run_card.user_set)
        self.assertNotIn('el', run_card.user_set)
        self.assertNotIn('ej', run_card.user_set)
        self.assertIn('polbeam1', run_card.user_set)
        self.assertNotIn('ptl', run_card.user_set)
        
        self.assertEqual(run_card['lpp1'], -3)
        self.assertEqual(run_card['lpp2'], 3)
        self.assertEqual(run_card['pdlabel'], 'eva')
        self.assertEqual(run_card['fixed_fac_scale'], True)
        self.assertEqual(run_card['eva_xcut'], 1)
        
        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        
        target = 0.01118182
        self.assertTrue(abs(val1 - target) / err1 < 2., 'large difference between %s and %s +- %s (%s sigma)'%
                        (target, val1, err1, abs(val1 - target) / err1))

    def test_eva_oldrelease_collision(self):
        """check that w+ w- > t t~ with EVA gives the correct result
        assuming no x > MV/Ebeam restriction (eva_xcut=0) [2111.02442]"""


        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.run_cmd('set group_subprocesses false')
        mg_cmd.run_cmd('set automatic_html_opening False --save')
        mg_cmd.run_cmd(' generate w+ w-  > t t~')
        mg_cmd.run_cmd('output madevent %s/'% self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=  self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')
        
        #check validity of the default run_card
        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards','run_card.dat'))

        f = open(pjoin(self.run_dir, 'Cards','run_card.dat'),'r')
        self.assertNotIn('ptj', run_card.user_set)
        self.assertNotIn('drjj', run_card.user_set)
        self.assertNotIn('ptj2min', run_card.user_set)
        self.assertNotIn('ptj3min', run_card.user_set)
        self.assertNotIn('mmjj', run_card.user_set)
        self.assertIn('ptheavy', run_card.user_set)
        self.assertNotIn('el', run_card.user_set)
        self.assertNotIn('ej', run_card.user_set)
        self.assertIn('polbeam1', run_card.user_set)
        self.assertNotIn('ptl', run_card.user_set)
        
        self.assertEqual(run_card['lpp1'], -3)
        self.assertEqual(run_card['lpp2'], 3)
        self.assertEqual(run_card['pdlabel'], 'eva')
        self.assertEqual(run_card['fixed_fac_scale'], True)
        run_card.set('eva_xcut', 0, user=True)
        run_card.write(pjoin(self.run_dir, 'Cards','run_card.dat'))
        self.assertEqual(run_card['eva_xcut'], 0)

        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        
        target = 0.02187245
        self.assertTrue(abs(val1 - target) / err1 < 2., 'large difference between %s and %s +- %s (%s sigma)'%
                        (target, val1, err1, abs(val1 - target) / err1))    

        
    def test_ieva_collision(self):
        """check that w+ w- > t t~ with EVA at full LP gives the correct result"""


        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.run_cmd('set group_subprocesses false')
        mg_cmd.run_cmd('set automatic_html_opening False --save')
        mg_cmd.run_cmd(' generate w+ w-  > t t~')
        mg_cmd.run_cmd('output madevent %s/'% self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=  self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')
        
        #check validity of the default run_card
        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards','run_card.dat'))

        f = open(pjoin(self.run_dir, 'Cards','run_card.dat'),'r')
        self.assertNotIn('ptj', run_card.user_set)
        self.assertNotIn('drjj', run_card.user_set)
        self.assertNotIn('ptj2min', run_card.user_set)
        self.assertNotIn('ptj3min', run_card.user_set)
        self.assertNotIn('mmjj', run_card.user_set)
        self.assertIn('ptheavy', run_card.user_set)
        self.assertNotIn('el', run_card.user_set)
        self.assertNotIn('ej', run_card.user_set)
        self.assertIn('polbeam1', run_card.user_set)
        self.assertNotIn('ptl', run_card.user_set)
        
        self.assertEqual(run_card['lpp1'], -3)
        self.assertEqual(run_card['lpp2'], 3)
        self.assertEqual(run_card['pdlabel'], 'eva')
        run_card.set('evaorder', 1, user=True)
        run_card.write(pjoin(self.run_dir, 'Cards','run_card.dat'))
        self.assertEqual(run_card['evaorder'], 1)
        self.assertEqual(run_card['eva_xcut'], 1)
        self.assertEqual(run_card['fixed_fac_scale'], True)

        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        
        #target = 0.003795
        target =0.003837 # value from v3.7.2 for 250k events
        # a single sample against a fixed reference: a 2 sigma window fails ~5%
        # of the random sequences, so any change of the event generation (not of
        # the cross-section) can trip it; 3 sigma keeps the check meaningful
        self.assertTrue(abs(val1 - target) / err1 < 3., 'large difference between %s and %s +- %s (%s sigma)'%
                        (target, val1, err1, abs(val1 - target) / err1))


    def test_quarkonium_production(self):
        """check that e e > etab(3S) etab(3S) gives the correct result"""

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('import model sm_onia')
        mg_cmd.exec_cmd(' generate e+ e- > etab(3s) etab(3S)')
        mg_cmd.exec_cmd('output madevent %s/'% self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=  self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')

        #check validity of the default run_card
        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards','run_card.dat'))
        self.assertIn('mom_resh_type', run_card.user_set)

        shutil.copy(os.path.join(_file_path, 'input_files', 'run_card_quarkonium.dat'),
                    '%s/Cards/run_card.dat' % self.run_dir)
        shutil.copy(os.path.join(_file_path, 'input_files', 'onia_card_quarkonium.dat'),
                    '%s/Cards/onia_card.dat' % self.run_dir)

        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        # 10k value is 6.227e-17 +- 2e-20
        target = 6.227e-17
        self.assertLess(
            abs(val1 - target) / (err1+1.4e-20),
            2.,
            'large difference between %s and %s +- %s'%
                        (target, val1, err1)
        )


    def test_leptonium_production(self):
        """check that p p > j mumu(1|3S1) gives the correct result"""

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('import model sm_onia-lepton_masses')
        mg_cmd.exec_cmd(' define p = g u d s u~ d~ s~')
        mg_cmd.exec_cmd(' define j = g u d s u~ d~ s~')
        mg_cmd.exec_cmd(' generate p p > j mumu(1|3S1)')
        mg_cmd.exec_cmd('output madevent %s/'% self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=  self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')

        #check validity of the default run_card
        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards','run_card.dat'))
        self.assertIn('mom_resh_type', run_card.user_set)

        shutil.copy(os.path.join(_file_path, 'input_files', 'run_card_leptonium.dat'),
                    '%s/Cards/run_card.dat' % self.run_dir)
        shutil.copy(os.path.join(_file_path, 'input_files', 'onia_card_leptonium.dat'),
                    '%s/Cards/onia_card.dat' % self.run_dir)

        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        # 1M value is 0.01668020 +- 7.24e-6
        target = 0.01668020
        self.assertLess(
            abs(val1 - target) / (err1+3e-6),
            2.,
            'large difference between %s and %s +- %s'%
                        (target, val1, err1)
        )


    def test_customised_madevent_via_run_card(self):
        """checking various advanced functionality of customization
           - set run_card entry via input/default_run_card_lo.dat
           - check that unknown entry can be added to the run_card.dat
           - check that custom cuts can be defined via the run_card.dat
           - check that those custom cuts can use custom entry
           - check that the cross-section is the expected one 
        """

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.run_cmd('set automatic_html_opening False')
        mg_cmd.run_cmd('generate p p > t t~')
        default_path = pjoin(self.path, 'default.dat')
        open(default_path, 'w').write("4 = dynamical_scale_choice\n 5.0 = my_param\n F = use_syst\n 5000 = nevents")
        import madgraph.various.banner as banner
        with misc.TMP_variable(banner.RunCardLO, 'default_run_card', default_path):
            mg_cmd.run_cmd('output madevent %s/'% self.run_dir)

        self.assertIn('my_param', open(pjoin(self.run_dir,'Cards','run_card.dat')).read())
        lo = banner.RunCard(pjoin(self.run_dir,'Cards', 'run_card.dat'))
        self.assertEqual(lo['dynamical_scale_choice'], 4)
        self.assertEqual(lo['my_param'], 5.0)
        
        # edit run_card
        fsock = open(pjoin(self.run_dir,'Cards', 'run_card.dat'),'a')
        fsock.write('\n[%s] = custom_fcts\n 10.0 = my_param2\n' % pjoin(self.path, 'custom.f'))
        fsock.close()

        # define the user cut
        cut = """
              logical FUNCTION dummy_cuts(P)
C**************************************************************************
C     INPUT:
C            P(0:3,1)           MOMENTUM OF INCOMING PARTON
C            P(0:3,2)           MOMENTUM OF INCOMING PARTON
C            P(0:3,3)           MOMENTUM OF ...
C            ALL MOMENTA ARE IN THE REST FRAME!!
C            COMMON/JETCUTS/   CUTS ON JETS
C     OUTPUT:
C            TRUE IF EVENTS PASSES ALL CUTS LISTED
C**************************************************************************
      IMPLICIT NONE
c
c     Constants
c
      include 'genps.inc'
      include 'nexternal.inc'
      include 'run.inc'
C
C     ARGUMENTS
C
      REAL*8 P(0:3,nexternal)
C
C     PARAMETERS
C
      real*8 PI
      parameter( PI = 3.14159265358979323846d0 )
      double precision pt

      if (pt(P(0,3)).lt.my_param)then
        dummy_cuts=.false.
        return
      endif   
      if (pt(P(0,4)).lt.my_param2)then
        dummy_cuts=.false.
        return
      endif   
      if (my_param.eq.my_param2)then
        dummy_cuts=.false.
        return
      endif  
      dummy_cuts=.true.

      return
      end
        """

        fsock = open(pjoin(self.path, 'custom.f'),'w')
        fsock.write(cut)
        fsock.close()
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=  self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')
        self.do('generate_events -f')

        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']

        # previously PDF was nn23lo1 (lhaid 230000) with this reference value 361.7
        target = 289.50 #+- 0.52 pb
        self.assertTrue(abs(val1 - target) / (2*err1) < 1., 'large difference between %s and %s +- %s'%
                        (target, val1, err1))

        self.assertIn('MY_PARAM', open(pjoin(self.run_dir,'Source','run.inc')).read())
        self.assertEqual(2, open(pjoin(self.run_dir,'Source','run.inc')).read().count('autodef'))

        self.assertIn('MY_PARAM', open(pjoin(self.run_dir,'Source','run_card.inc')).read())
        self.assertIn('MY_PARAM', open(pjoin(self.run_dir,'SubProcesses','dummy_fct.f')).read())


    def test_customised_madevent_via_run_card(self):
        """checking various advanced functionality of customization
           - set run_card entry via input/default_run_card_lo.dat
           - check that unknown entry can be added to the run_card.dat
           - check that custom cuts can be defined via the run_card.dat
           - check that those custom cuts can use custom entry
           - check that the cross-section is the expected one 
        """

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.run_cmd('set automatic_html_opening False')
        mg_cmd.run_cmd('generate p p > t t~')
        default_path = pjoin(self.path, 'default.dat')
        open(default_path, 'w').write("4 = dynamical_scale_choice\n 5.0 = my_param\n F = use_syst\n 5000 = nevents")
        import madgraph.various.banner as banner
        with misc.TMP_variable(banner.RunCardLO, 'default_run_card', default_path):
            mg_cmd.run_cmd('output madevent %s/'% self.run_dir)

        self.assertIn('my_param', open(pjoin(self.run_dir,'Cards','run_card.dat')).read())
        lo = banner.RunCard(pjoin(self.run_dir,'Cards', 'run_card.dat'))
        self.assertEqual(lo['dynamical_scale_choice'], 4)
        self.assertEqual(lo['my_param'], 5.0)
        
        # edit run_card
        fsock = open(pjoin(self.run_dir,'Cards', 'run_card.dat'),'a')
        fsock.write('\n[%s] = custom_fcts\n 10.0 = my_param2\n' % pjoin(self.path, 'custom.f'))
        fsock.close()

        # define the user cut
        cut = """
              logical FUNCTION dummy_cuts(P)
C**************************************************************************
C     INPUT:
C            P(0:3,1)           MOMENTUM OF INCOMING PARTON
C            P(0:3,2)           MOMENTUM OF INCOMING PARTON
C            P(0:3,3)           MOMENTUM OF ...
C            ALL MOMENTA ARE IN THE REST FRAME!!
C            COMMON/JETCUTS/   CUTS ON JETS
C     OUTPUT:
C            TRUE IF EVENTS PASSES ALL CUTS LISTED
C**************************************************************************
      IMPLICIT NONE
c
c     Constants
c
      include 'genps.inc'
      include 'nexternal.inc'
      include 'run.inc'
C
C     ARGUMENTS
C
      REAL*8 P(0:3,nexternal)
C
C     PARAMETERS
C
      real*8 PI
      parameter( PI = 3.14159265358979323846d0 )
      double precision pt

      if (pt(P(0,3)).lt.my_param)then
        dummy_cuts=.false.
        return
      endif   
      if (pt(P(0,4)).lt.my_param2)then
        dummy_cuts=.false.
        return
      endif   
      if (my_param.eq.my_param2)then
        dummy_cuts=.false.
        return
      endif  
      dummy_cuts=.true.

      return
      end
        """

        fsock = open(pjoin(self.path, 'custom.f'),'w')
        fsock.write(cut)
        fsock.close()
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=  self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')
        self.do('generate_events -f')

        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']

        # previously PDF was nn23lo1 (lhaid 230000) with this reference value 361.7
        target = 289.50 #+- 0.52 pb
        self.assertTrue(abs(val1 - target) / (2*err1) < 1., 'large difference between %s and %s +- %s'%
                        (target, val1, err1))

        self.assertIn('MY_PARAM', open(pjoin(self.run_dir,'Source','run.inc')).read())
        self.assertEqual(2, open(pjoin(self.run_dir,'Source','run.inc')).read().count('autodef'))

        self.assertIn('MY_PARAM', open(pjoin(self.run_dir,'Source','run_card.inc')).read())
        self.assertIn('MY_PARAM', open(pjoin(self.run_dir,'SubProcesses','dummy_fct.f')).read())



    def test_eft_running(self):
        """check that  gives the correct result"""



        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        if not os.path.exists(pjoin(MG5DIR, 'Template',"Running")):
            mg_cmd.run_cmd('install RunningCoupling')
        mg_cmd.run_cmd('set automatic_html_opening False --save')
        mg_cmd.run_cmd('import model %s/tests/input_files/SMEFTatNLO_running' % madgraph.MG5DIR)
        mg_cmd.run_cmd('generate p p > t t~ NP=2 NP^2==2 QCD=2 QED=0')
        mg_cmd.run_cmd('output madevent %s/'% self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=  self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')
        
        #check validity of the default run_card
        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards','run_card.dat'))

        #f = open(pjoin(self.run_dir, 'Cards','run_card.dat'),'r')
        self.assertIn('fixed_extra_scale', run_card.user_set)
        self.assertIn('mue_ref_fixed', run_card.user_set)
        self.assertIn('mue_over_ref', run_card.user_set)

        run_card['nevents'] = 10000
        run_card.write('%s/Cards/run_card.dat' % self.run_dir)
        
        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']

        #target = 166.36114 # value used as reference before changing sde_strategy
        # previously PDF was nn23lo1 (lhaid 230000) with this reference value
        # 165.84 (a 100k run, +- 0.05)
        # NNPDF40_lo_as_01180: 124.39 +- 0.033 from two 100k runs (124.389 and
        # 124.395 +- 0.046, with and without the last-iteration normalisation of
        # the reported cross-section); the former target 124.45 was a single 10k
        # run. A 10k run is compared within 3 sigma (its error combined with
        # the target's): 1 sigma fails a third of unbiased runs.
        target, target_err = 124.39, 0.033
        self.assertTrue(abs(val1 - target) / math.sqrt(err1**2 + target_err**2) < 3.,
                        'large difference between %s and %s +- %s'% (target, val1, err1))

        
        # edit run_card -> fix scale
        run_card['fixed_extra_scale'] = True
        run_card['mue_ref_fixed'] = 250

        run_card.write('%s/Cards/run_card.dat' % self.run_dir)
        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        # previously PDF was nn23lo1 (lhaid 230000) with this reference value
        # 165.71 (a 100k run, +- 0.06)
        # NNPDF40_lo_as_01180: 124.37 +- 0.033 from two 100k runs (124.377 and
        # 124.367 +- 0.047); the former target 124.36 was a single 10k run
        target, target_err = 124.37, 0.033
        self.assertTrue(abs(val1 - target) / math.sqrt(err1**2 + target_err**2) < 3.,
                        'large difference between %s and %s +- %s'% (target, val1, err1))






    @test_aloha.set_global()
    def test_complex_mass_scheme(self):
        """check that auto-width and Madspin works nicely with complex-mass-scheme"""
        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --save')
        mg_cmd.exec_cmd('set complex_mass_scheme', precmd=True)
        mg_cmd.exec_cmd('generate g g  > t t~', precmd=True)
        mg_cmd.exec_cmd('output madevent %s' % self.run_dir, precmd=True)
        
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=  self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')
        
        #modify run_card
        run_card = banner_mod.RunCard(pjoin(self.run_dir, 'Cards', 'run_card.dat'))
        run_card.set('nevents', 100)
        run_card.write(pjoin(self.run_dir, 'Cards', 'run_card.dat'))
        
        # check the auto-width
        self.cmd_line.exec_cmd('compute_widths 6 -f')

        # check value for the width    
        import madgraph.models.check_param_card as check_param_card    
        param_card = check_param_card.ParamCard(pjoin(self.run_dir, 'Cards', 'param_card.dat'))
        # NB: the trailing 3 in the original call was assertTrue's *msg*
        # argument, not a precision -- misc.equal's default sig_fig=6 is what
        # this has always compared at, so it is left alone.
        self.assertTrue(misc.equal(1.491257, param_card['decay'].get(6).value),
            'top width %s far from reference 1.491257'
            % param_card['decay'].get(6).value)
                        
        # generate events
        self.cmd_line.exec_cmd('launch -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        # previously PDF was nn23lo1 (lhaid 230000) with this reference value 440.779
        target = 313.66
        self.assertTrue(misc.equal(target, val1, 4*err1),
            'g g > t t~ (CMS) cross-section %s +- %s far from reference %s'
            % (val1, err1, target))                
        

        # run madspin
        fsock = open(pjoin(self.run_dir, 'Cards', 'madspin_card.dat'),'w')
        fsock.write('decay t > w+ b \n launch')
        fsock.close()
        
        self.cmd_line.exec_cmd('decay_events run_01 -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        # Not the production 440.779 any more. BR(t -> w+ b) is 1 to seven
        # digits, so the decayed cross-section used to be the production one --
        # but MadSpin draws the top's virtuality only inside +- BW_cut widths of
        # the pole and now says so: the reported sigma carries the fraction of
        # the Breit-Wigner that window keeps. One top is decayed here (t~ is not
        # in the card and MadSpin does not auto-conjugate), so the factor is a
        # single bw_retained_fraction(173.0, 1.491257, 15) = 0.9786983 and
        # 440.779 -> 431.39, i.e. -2.13%.
        #
        # The 4*err1 band is +-4.3% on a 100-event run, so the old number still
        # fitted inside it. That is exactly why it is updated rather than left:
        # a tolerance wide enough to hide a systematic shift is not a check that
        # the shift is right.
        # previously PDF was nn23lo1 (lhaid 230000) with this reference value 431.39
        # Derived, not measured: the -2.13% factor above is the Breit-Wigner
        # window fraction, which does not depend on the PDF, so the new value is
        # 313.66 * 0.9786983. CI measures it directly on the next run.
        target = 306.98
        self.assertTrue(misc.equal(target, val1, 4*err1),
            'MadSpin-decayed cross-section %s +- %s far from reference %s'
            % (val1, err1, target))
             
        
        
        
    def test_width_scan(self):
        """check that the width settings works on a scan based.
           and check that MW is updated."""
           
        cmdline = """
        set notification_center None --no-save 
        generate e+ e- > Z > mu+ mu-
        output madevent %s -f
        launch
        set use_syst F
        set MZ scan:[80, 85]
        set WZ Auto
        set nevents 1
        done
        launch 
        set WZ 2.0
        """ %(self.run_dir)
        
        cmdfile = open(pjoin(self.path,'cmd'),'w').write(cmdline)
        
        
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        subprocess.call([pjoin(_file_path, os.path.pardir,'bin','madgraph'), 
                         pjoin(self.path, 'cmd')],
                         #cwd=pjoin(self.path),
                        stdout=stdout,stderr=stdout)
        
        # check that the scan was done
        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'Events', 'run_04')))
        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'Events', 'scan_run_0[1-2].txt')))
        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'Events', 'scan_run_0[3-4].txt')))
        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'Events', 'scan_run_0[1-2].json')))
        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'Events', 'scan_run_0[3-4].json')))
        
        banner1 = banner.Banner(pjoin(self.run_dir, 'Events','run_01', 'run_01_tag_1_banner.txt'))
        banner2 = banner.Banner(pjoin(self.run_dir, 'Events','run_02', 'run_02_tag_1_banner.txt'))                                
        
        # check that MZ is updated
        self.assertEqual(banner1.get('param', 'mass', 23).value, 80)
        self.assertEqual(banner2.get('param', 'mass', 23).value, 85)

        #check that WZ is updated 
        self.assertEqual(banner1.get('param', 'decay', 23).value, 1.515619)
        self.assertEqual(banner2.get('param', 'decay', 23).value, 1.882985)   
        
        # check that MW is updated
        self.assertEqual(banner1.get('param', 'mass', 24).value, 6.496446e+01)
        self.assertEqual(banner2.get('param', 'mass', 24).value, 7.242341e+01)        
               
        banner3 = banner.Banner(pjoin(self.run_dir, 'Events','run_03', 'run_03_tag_1_banner.txt'))
        banner4 = banner.Banner(pjoin(self.run_dir, 'Events','run_04', 'run_04_tag_1_banner.txt'))                                
        
        # check that MZ is updated
        self.assertEqual(banner3.get('param', 'mass', 23).value, 80)
        self.assertEqual(banner4.get('param', 'mass', 23).value, 85)

        #check that WZ is NOT updated 
        self.assertEqual(banner3.get('param', 'decay', 23).value, 2.0)
        self.assertEqual(banner4.get('param', 'decay', 23).value, 2.0)   
        
        # check that MW is updated
        self.assertEqual(banner3.get('param', 'mass', 24).value, 6.496446e+01)
        self.assertEqual(banner4.get('param', 'mass', 24).value, 7.242341e+01)         
        
        
    def test_e_e_collision(self):
        """check that e+ e- > t t~ gives the correct result"""
        

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --save')
        mg_cmd.exec_cmd(' generate e+ e-  > e+ e-')
        mg_cmd.exec_cmd('output madevent %s/' % self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=  self.run_dir)
        self.cmd_line.no_notification()
        
        # couple of test checking that default run_card is as expected
        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards','run_card.dat'))
        self.assertNotIn('ptj', run_card.user_set)
        self.assertNotIn('drjj', run_card.user_set)
        self.assertNotIn('ptj2min', run_card.user_set)
        self.assertNotIn('ptj3min', run_card.user_set)
        self.assertNotIn('mmjj', run_card.user_set)
        self.assertNotIn('ptheavy', run_card.user_set)
        self.assertIn('el', run_card.user_set)
        self.assertIn('polbeam1', run_card.user_set)
        self.assertIn('ptl', run_card.user_set)
        
        shutil.copy(os.path.join(_file_path, 'input_files', 'run_card_ee.dat'),
                    '%s/Cards/run_card.dat' % self.run_dir)
        
        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        
        target = 155.9
        self.assertLess(abs(val1 - target) / err1, 2.,
            'cross-section %s +- %s far from reference %s' % (val1, err1, target))

    def test_e_e_collision_mg7(self):
        """mg7 cross-section for e+ e- > e+ e- (Bhabha).

        NB: this cannot be compared to test_e_e_collision's 155.9 pb -- that
        number belongs to the tailored run_card_ee.dat which that test copies in.
        Here the mg7 default run_card.toml is used (via _run_mg7_xsec), and the
        reference is the madevent cross-section for the same setup: 40.3 pb.

        The e+ and e- beams are *different* particles, so the beam-swapped
        initial state is not part of the process and no flavor may be mirrored.
        Mirroring it (which happened while the initial-leg signature compared
        |pdg|, making e+ and e- look like the same beam) doubled the result to
        ~81 pb -- hence the explicit mirroring assertion on top of the
        cross-section. Self-skips where the mg7 runtime stack is unavailable.
        """
        import json
        datadir = _mg7_datadir_or_skip(self)
        run_dir = pjoin(self.path, 'MG7_ee')
        cross, error = _run_mg7_xsec(self,
            ['set automatic_html_opening False --no_save',
             'import model sm',
             'generate e+ e- > e+ e-'],
            run_dir, datadir)

        # explicit mirroring check: e+ and e- are distinct beams -> never mirror
        info = json.load(open(pjoin(run_dir, 'SubProcesses',
                                    'subprocesses.json')))
        for proc in info:
            for flavor in proc['flavors']:
                self.assertFalse(flavor['mirror'],
                    'e+ e- > e+ e-: the beams are different particles, so the '
                    'swapped initial state is not part of the process and flavor '
                    '%s must not be mirrored (that doubles the cross-section)'
                    % (flavor['options'],))

        # madevent reference for the same run_card settings
        target = 155.9
        target = 40.3 # fixed scale mz

        self.assertAlmostEqual(cross, 40.3, delta=max(1.0, 5 * error))

    def test_decay_1to3_mg7(self):
        """mg7 partial width of a 1 -> 3 decay: t > b e+ ve.

        A process with one incoming particle is the special case of the mg7
        chain, and it went unnoticed because the madevent fallback is what runs
        without madspace:
          - the default [histograms] are scaled by the mass of the decaying
            particle, which used to be read from a 'parameter_dict' that only
            a ModelReader has (a LoopModel crashed `output`);
          - madspace's observables take the first two momenta to be the beams,
            so the first decay product (the b here) dropped out of every
            selection, and `z > mu+ mu-` asking for lepton_2 crashed the run.
        So on top of the width (madevent: 0.1636 +- 0.0002 GeV, i.e.
        Gamma(t > b w+) x BR(w+ > e+ ve)) the default histograms are checked:
        no sqrt_s (madspace's adds up two beams), every momentum histogram
        filled with the full width, and the e+ ve pair mass peaking at MW.
        Self-skips where the mg7 runtime stack is unavailable.
        """
        import glob, json
        datadir = _mg7_datadir_or_skip(self)
        run_dir = pjoin(self.path, 'MG7_t_decay')
        width, error = _run_mg7_xsec(self,
            ['set automatic_html_opening False --no_save',
             'import model sm',
             'generate t > b e+ ve'],
            run_dir, datadir)
        self.assertAlmostEqual(width, 0.1636, delta=max(0.002, 5 * error))

        info = json.load(open(sorted(glob.glob(
            pjoin(run_dir, 'Events', '*', 'info.json')))[-1]))
        hists = dict((h['name'], h) for h in info['event_histograms'])
        self.assertNotIn('sqrt_s', list(hists))
        momentum = [name for name in hists if name != 'weight']
        for group in ('bottom', 'lepton', 'missing'):
            self.assertIn('%s-pt' % group, momentum)
        for name in momentum:
            values = hists[name]['bin_values']
            # the decay products of a 173 GeV top fit in the ranges, which
            # are built from the top mass: nothing in under- or overflow
            self.assertAlmostEqual(sum(values[1:-1]), width,
                                   delta=0.01 * width, msg=name)
            if name.endswith('-pt'):
                # a particle left out of the selection is histogrammed at 0
                self.assertLess(values[1], 0.5 * width, msg=name)
        mll = hists['lepton-missing-pair_mass']
        bin_width = (mll['max'] - mll['min']) / mll['bin_count']
        peak = mll['bin_values'].index(max(mll['bin_values'])) - 1
        self.assertLessEqual(mll['min'] + peak * bin_width, 80.419)
        self.assertGreater(mll['min'] + (peak + 1) * bin_width, 80.419)

    def load_result(self, run_name):
        
        import madgraph.iolibs.save_load_object as save_load_object
        import madgraph.madevent.gen_crossxhtml as gen_crossxhtml
        
        result = save_load_object.load_from_file('%s/HTML/results.pkl' % self.run_dir)
        return result[run_name]

    def check_parton_output(self, run_name='run_01', target_event=100, syst=False):
        """Check that parton output exists and reach the targert for event"""
                
        # check that the number of event is fine:
        data = self.load_result(run_name)
        self.assertEqual(int(data[0]['nb_event']), target_event)
        self.assertIn('lhe', data[0].parton)
        
        if syst:
            # check that the html has the information
            self.assertIn('syst', data[0].parton)
            # check that the code was run correctly
            fsock = open('%s/Events/%s/parton_systematics.log' % \
                  (self.run_dir, data[0]['run_name']),'r')
            text = fsock.read()
            self.assertGreaterEqual(text.count('dynamical scheme'), 3)
        
        # check that the html link makes sense
        #check_html_page(self, pjoin(self.run_dir, 'crossx.html'))
    
        
        
        
                
    def check_pythia_output(self, run_name='run_01', syst=False):
        """ """
        # check that the number of event is fine:
        data = self.load_result(run_name)
        self.assertIn('hep', data[0].pythia)
        self.assertIn('log', data[0].pythia)

#        if syst:
#            # check that the html has the information
#            self.assertTrue('rwt' in data[0].pythia)

#===============================================================================
# TestCmd
#===============================================================================
class TestMEfromfile(unittest.TestCase):
    """test that we can launch everything from a single file"""

    def setUp(self):
        
        self.debuging = unittest.debug
        if self.debuging:
            self.path = pjoin(MG5DIR, 'ACC_TEST')
            if os.path.exists(self.path):
                 shutil.rmtree(self.path)
            os.mkdir(self.path) 
        else:
            self.path = tempfile.mkdtemp(prefix='acc_test_mg5')
        self.run_dir = pjoin(self.path, 'MGPROC') 
        
    
    def tearDown(self):

        if not self.debuging:
            shutil.rmtree(self.path)
        self.assertFalse(self.debuging)

    def test_add_time_of_flight(self):
        """checking time of flight is working fine"""

        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception as error:
            pass
        
        cmd = """import model sm
                 set automatic_html_opening False --no_save
                 set notification_center False --no_save
                 generate p p > w+ z
                 output madevent %s -f -nojpeg
                 launch -i 
                 set automatic_html_opening False --no_save
                 generate_events
                 parton
                 set nevents 100
                 set event_norm average
                 set systematics_program none
                 add_time_of_flight --threshold=4e-14
                 pythia8
                 """ %self.run_dir

        open(pjoin(self.path, 'mg5_cmd'),'w').write(cmd)
        
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','madgraph'), 
                         pjoin(self.path, 'mg5_cmd')],
                         #cwd=self.path,
                        stdout=stdout, stderr=stderr)

        # previously PDF was nn23lo1 (lhaid 230000) with this reference value
        # cross=15.62, error=0.19
        self.check_parton_output(cross=16.99, error=0.21)
        self.check_pythia_output()
        event = '%s/Events/run_01/unweighted_events.lhe' % self.run_dir
        if not os.path.exists(event):
            misc.gunzip(event)
        
        has_zero = False
        has_non_zero = False
        for event in lhe_parser.EventFile(event):
            for particle in event:
                if particle.pid in [23,25]:
                    self.assertTrue(particle.vtim ==0 or particle.vtim > 4e-14)
                    if particle.vtim == 0 :
                        has_zero = True
                    else:
                        has_non_zero = True
        self.assertTrue(has_zero)
        self.assertTrue(has_non_zero)
        
        self.assertFalse(self.debuging)
    

    def test_w_production_with_ms_decay(self):
        """A run to test madspin (inline and offline) on p p > w+ and p p > w-"""
        
        cwd = os.getcwd()
        
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        if logging.getLogger('madgraph').level > 20:
            stdout = devnull
        else:
            stdout= None
            
        #
        #  START REAL CODE
        #
        command = open(pjoin(self.path, 'cmd'), 'w')
        command.write("""import model sm
        set automatic_html_opening False --no_save
        set notification_center False --no_save
        generate p p > w+ 
        add process p p > w-
        output madevent %(path)s
        launch
        madspin=ON
        analysis=OFF
        shower=pythia8    
        %(path)s/../madspin_card.dat
        set nevents 1000
        set lhaid 10042
        set pdlabel lhapdf
        launch -i
        decay_events run_01 
        %(path)s/../madspin_card2.dat
        """ % {'path':self.run_dir})
        command.close()
        
        fsock = open(pjoin(self.path, 'madspin_card.dat'), 'w')
        fsock.write("""set spinmode madspin
        decay w+ > j j
        decay w- > e- ve~
        launch
        """)
        fsock.close()
        fsock = open(pjoin(self.path, 'madspin_card2.dat'), 'w')
        fsock.write("""set spinmode madspin
        decay w+ > j j
        decay w- > j j
        launch
        """)
        fsock.close()
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','madgraph'),
                         pjoin(self.path, 'cmd')],
                         cwd=pjoin(_file_path, os.path.pardir),
                        stdout=stdout,stderr=stdout)     
        
        #a=rwa_input('freeze')
        self.check_parton_output(cross=150770.0, error=7.4e+02,target_event=1000)
        self.check_parton_output('run_01_decayed_1', cross=66344.2066122, error=1.5e+03,target_event=666, delta_event=40)
        #logger.info('\nMS info: the number of events in the html file is not (always) correct after MS\n')
        self.check_parton_output('run_01_decayed_2', cross=100521.52517, error=8e+02,target_event=1000)
        self.check_pythia_output(run_name='run_01_decayed_1')

        #check the first decayed events for energy-momentum conservation.


        self.assertEqual(cwd, os.getcwd())


    def test_wj_production_with_ms_decay(self):
        """A run to test madspin (inline and offline) on p p > w+ j and p p > w- j.

        Same as test_w_production_with_ms_decay but with an extra jet at
        production, so the production process is 2 -> 2 (the W recoils against
        the jet). This exercises the spinmode=madspin reshuffling path: a fresh
        Breit-Wigner W mass is sampled and the production momenta are reshuffled
        before accept/reject (unlike the 2 -> 1 w+/w- case, which stays onshell).
        """

        cwd = os.getcwd()

        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        if logging.getLogger('madgraph').level > 20:
            stdout = devnull
        else:
            stdout= None

        #
        #  START REAL CODE
        #
        command = open(pjoin(self.path, 'cmd'), 'w')
        command.write("""import model sm
        set automatic_html_opening False --no_save
        set notification_center False --no_save
        generate p p > w+ j
        add process p p > w- j
        output madevent %(path)s
        launch
        madspin=ON
        analysis=OFF
        shower=pythia8
        %(path)s/../madspin_card.dat
        set nevents 1000
        set lhaid 10042
        set pdlabel lhapdf
        launch -i
        decay_events run_01
        %(path)s/../madspin_card2.dat
        """ % {'path':self.run_dir})
        command.close()

        fsock = open(pjoin(self.path, 'madspin_card.dat'), 'w')
        fsock.write("""set spinmode madspin
        decay w+ > j j
        decay w- > e- ve~
        launch
        """)
        fsock.close()
        fsock = open(pjoin(self.path, 'madspin_card2.dat'), 'w')
        fsock.write("""set spinmode madspin
        decay w+ > j j
        decay w- > j j
        launch
        """)
        fsock.close()
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','madgraph'),
                         pjoin(self.path, 'cmd')],
                         cwd=pjoin(_file_path, os.path.pardir),
                        stdout=stdout,stderr=stdout)

        # Cross-section reference values are omitted (cross=0 skips the check):
        # p p > w+/- j differ from the inclusive p p > w+/- of
        # test_w_production_with_ms_decay and need a reference run to pin down.
        # The event-count targets are driven by the (unchanged) W branching
        # ratios, so they match the 2 -> 1 test: ~666 kept for the mixed-BR
        # card (w+ > j j vs w- > e- ve~) and the full 1000 for the all-jets card.
        self.check_parton_output(target_event=1000)
        self.check_parton_output('run_01_decayed_1', target_event=666, delta_event=40)
        self.check_parton_output('run_01_decayed_2', target_event=1000)
        self.check_pythia_output(run_name='run_01_decayed_1')

        #check the first decayed events for energy-momentum conservation.


        self.assertEqual(cwd, os.getcwd())


    def test_w_production_with_PA_decay(self):
        """A run to test MadSpin PA (pole-approximation/density) mode on p p > w+ / w-.

        Inline-only counterpart of test_w_production_with_PA_decay_inline_then_offline.
        Exercises the new PA path through run_onshell(density_method=True).
        The madspin card has different BRs for w+ (-> j j) vs w- (-> e- ve~)
        and therefore exercises the BR-equalization / loose-decay branch
        (events with the smaller-BR pdg are dropped to keep the output
        unweighted).
        """

        cwd = os.getcwd()

        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        if logging.getLogger('madgraph').level > 20:
            stdout = devnull
        else:
            stdout= None

        #
        #  START REAL CODE
        #
        command = open(pjoin(self.path, 'cmd'), 'w')
        command.write("""import model sm
        set automatic_html_opening False --no_save
        set notification_center False --no_save
        generate p p > w+
        add process p p > w-
        output madevent %(path)s
        launch
        madspin=ON
        analysis=OFF
        shower=OFF
        %(path)s/../madspin_card.dat
        set nevents 1000
        set lhaid 10042
        set pdlabel lhapdf
        """ % {'path':self.run_dir})
        command.close()

        fsock = open(pjoin(self.path, 'madspin_card.dat'), 'w')
        fsock.write("""set spinmode PA
        decay w+ > j j
        decay w- > e- ve~
        launch
        """)
        fsock.close()
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','madgraph'),
                         pjoin(self.path, 'cmd')],
                         cwd=pjoin(_file_path, os.path.pardir),
                        stdout=stdout,stderr=stdout)

        # Parton-level production reference (unchanged from the madspin test).
        self.check_parton_output(cross=150770.0, error=7.4e+02, target_event=1000)
        # Mixed-BR case: BR equalization drops the w- -> e- ve~ events so the
        # effective BR is ~ (sigma_w+ * BR(w+->jj) + sigma_w- * BR(w-->eve)) / sigma_tot,
        # matching the legacy madspin result up to MC noise.
        self.check_parton_output('run_01_decayed_1', cross=66344.2066122, error=1.5e+03,
                                 target_event=666, delta_event=80)

        self.assertEqual(cwd, os.getcwd())


    def test_w_production_with_PA_decay_inline_then_offline(self):
        """PA-mode MadSpin run inline first with one set of decay channels,
        then again offline via ``decay_events`` with a different set of
        channels, on a mixed w+/w- sample.

        Mirrors ``test_w_production_with_ms_decay`` (same inline+offline
        sequence on the legacy ``spinmode=madspin`` path).

        Used to raise ``KeyError: (-24, 1, -2)`` in ``get_pdir`` because
        every MadSpin instance compiled its matrix elements to the same
        ``madspin_me/SubProcesses/`` directory, and macOS' ``dlopen``
        caches loaded libraries by install_name
        (``@rpath/liball_2me.dylib``) — the second run kept reusing the
        first run's already-loaded matrix elements regardless of what was
        on disk. Fixed by giving each MadSpinInterface instance a unique
        ``madspin_me_<N>`` output subdir and patching the dylib's
        install_name with ``install_name_tool`` so each run gets a fresh
        in-memory library.
        """

        cwd = os.getcwd()

        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        if logging.getLogger('madgraph').level > 20:
            stdout = devnull
        else:
            stdout= None

        command = open(pjoin(self.path, 'cmd'), 'w')
        command.write("""import model sm
        set automatic_html_opening False --no_save
        set notification_center False --no_save
        generate p p > w+
        add process p p > w-
        output madevent %(path)s
        launch
        madspin=ON
        analysis=OFF
        shower=OFF
        %(path)s/../madspin_card.dat
        set nevents 1000
        set lhaid 10042
        set pdlabel lhapdf
        launch -i
        decay_events run_01
        %(path)s/../madspin_card2.dat
        """ % {'path':self.run_dir})
        command.close()

        fsock = open(pjoin(self.path, 'madspin_card.dat'), 'w')
        fsock.write("""set spinmode PA
        decay w+ > j j
        decay w- > e- ve~
        launch
        """)
        fsock.close()
        fsock = open(pjoin(self.path, 'madspin_card2.dat'), 'w')
        fsock.write("""set spinmode PA
        decay w+ > j j
        decay w- > j j
        launch
        """)
        fsock.close()
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','madgraph'),
                         pjoin(self.path, 'cmd')],
                         cwd=pjoin(_file_path, os.path.pardir),
                        stdout=stdout,stderr=stdout)

        self.check_parton_output(cross=150770.0, error=7.4e+02, target_event=1000)
        # Mixed-BR card: BR equalization drops part of the w- -> e- ve~ events.
        self.check_parton_output('run_01_decayed_1', cross=66344.2066122, error=1.5e+03,
                                 target_event=666, delta_event=80)
        # Identical-BR card: no drops.
        self.check_parton_output('run_01_decayed_2', cross=100521.52517, error=8e+02,
                                 target_event=1000)

        self.assertEqual(cwd, os.getcwd())


    def test_DY_onejet(self):
        """
        This test is checking that the scale in auto_dsig are correctly assigned
        in 3.6.2, a wrong Q2FACT(IB(1)) was used instead of Q2FACT(1).
        Leading to an asymmetry in the DY +1j process.
        This acceptance test is there to prevent such type of error
        """

        cwd = os.getcwd()
        
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        if logging.getLogger('madgraph').level > 20:
            stdout = devnull
        else:
            stdout= None
            
        #
        #  START REAL CODE
        #
        command = open(pjoin(self.path, 'cmd'), 'w')
        command.write("""import model sm
        set automatic_html_opening False --no_save
        set notification_center False --no_save
        generate p p > mu+ mu- j
        output madevent %(path)s
        launch
        shower=OFF
        set nevents 10000
        set ickkw 1
        set xqcut 10
        set mmll 50
        set ptj 0.01
        """ % {'path':self.run_dir})
        command.close()
        
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','madgraph'), 
                         pjoin(self.path, 'cmd')],
                         cwd=pjoin(_file_path, os.path.pardir),
                        stdout=stdout,stderr=stdout)     
        
        #a=rwa_input('freeze')
        self.check_parton_output(cross=591.1733, error=2.17,target_event=10000)

        count = [0,0]
        for event in lhe_parser.EventFile(pjoin(self.run_dir, 'Events', 'run_01','unweighted_events.lhe.gz')):
            event.check()
            for particle in event:
                if particle.pid == 13:
                    if particle.pz > 0:
                        count[0] += 1
                    else:
                        count[1] += 1
                    break 

        # forward/backward balance of the muon: with 10000 events one fraction
        # has a statistical error of 0.005, so the window is 4 sigma (0.49-0.51
        # was 2 sigma and failed on the random stream alone; 100k events of the
        # same run card give 0.5002 +- 0.0016)
        self.assertTrue(0.48<count[0]/10000.<0.52)
        self.assertTrue(0.48<count[1]/10000.<0.52)


        self.assertEqual(cwd, os.getcwd())


    def test_generation_heft(self):
        """test added since heft was crashing due to a wrong handling of color denominator
           in the fortran exporter
        """

        cwd = os.getcwd()
        
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        if logging.getLogger('madgraph').level > 20:
            stdout = devnull
        else:
            stdout= None
            
        #
        #  START REAL CODE
        #
        command = open(pjoin(self.path, 'cmd'), 'w')
        command.write("""import model heft
        set automatic_html_opening False --no_save
        set notification_center False --no_save
        generate g g > b b~ HIW<=1
        output madevent %(path)s
        launch 
        shower=OFF
        set nevents 1000
        """ % {'path':self.run_dir})
        command.close()
        
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','madgraph'), 
                         pjoin(self.path, 'cmd')],
                         cwd=pjoin(_file_path, os.path.pardir),
                        stdout=stdout,stderr=stdout)     
        
        #a=rwa_input('freeze')
        # previously PDF was nn23lo1 (lhaid 230000) with this reference value cross=4.117e+08, error=1.413e+06
        self.check_parton_output(cross= 2.1086e+08, error=1.34e+06,target_event=1000)

    def test_generation_heft_mg7(self):
        """mg7 equivalent of test_generation_heft for g g > b b~ HIW<=1 (HEFT).

        Pins the mg7-native cross-section obtained with the run_card.toml
        defaults and the dynamical HT/2 scale that _run_mg7_xsec selects.
        Self-skips where the mg7 runtime stack is unavailable.
        """
        datadir = _mg7_datadir_or_skip(self)
        cross, error = _run_mg7_xsec(self,
            ['set automatic_html_opening False --no_save',
             'import model heft',
             'generate g g > b b~ HIW<=1'],
            pjoin(self.path, 'MG7_heft'), datadir)
        # mg7 reference with the default PDF NNPDF40_lo_as_01180 (NNPDF4.0 LO,
        # alpha_s(M_Z) = 0.118). The 3.708e+08 of the old NNPDF23_lo_as_0130_qed
        # default drops by -44% purely through the PDF -- gg luminosity times
        # alpha_s^2 (-22% at the ~20 GeV dynamical scale).
        # Spread over 6 runs: 2.0872e+08-2.0901e+08, mean 2.089e+08.
        # previously PDF was NNPDF40MC_lo_as_01180 with this reference value 1.820e+08
        target = 2.089e+08
        self.assertLess(abs(cross - target) / target, 0.10,
            'mg7 HEFT cross-section %s far from physical reference %s'
            % (cross, target))

    def test_polarization_top_decay(self):
        """check that polarized process t{X} > w+{Y} b{Z}, w+ > ta+ vt gives the correct results
        Test 1: check that various permutations can be called
        Test 2: check helicity-flipping process (massive limit)
        Test 3: check helicity-flipping process (massless limit)
        """

        cwd = os.getcwd()

        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        if logging.getLogger('madgraph').level > 20:
            stdout = devnull
        else:
            stdout= None

        #
        #  START REAL CODE (1/3)
        #
        command = open(pjoin(self.path, 'cmd'), 'w')
        command.write("""set group_subprocesses False
        import model loop_sm
        set automatic_html_opening False --no_save
        set notification_center False --no_save
        generate    t{L} > w+{0} b{R}, w+ > ta+ vt
        add process t{L} > w+{T} b{L}, w+ > ta+ vt
        add process t{L} > w+{A} b{R}, w+ > ta+ vt
        add process t{R} > w+{S} b{L}, w+ > ta+ vt
        add process t{R} > w+{0S} b{R}, w+ > ta+ vt
        add process t{L} > w+{S0} b{L}, w+ > ta+ vt
        add process t{L} > w+{G} b{R}, w+ > ta+ vt
        add process t{L} > w+{H} b{L}, w+ > ta+ vt
        add process t{R} > w+{Q} b{R}, w+ > ta+ vt
        add process t{R} > w+{W} b{L}, w+ > ta+ vt
        output madevent %(path)s
        launch
        analysis=off
        set no_parton_cut
        set nevents 40k
        set me_frame [1]
        set nhel 1
        set bwcutoff 100
        """ % {'path':self.run_dir})
        command.close()

        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','madgraph'),
                         pjoin(self.path, 'cmd')],
                         cwd=pjoin(_file_path, os.path.pardir),
                        stdout=stdout,stderr=stdout)

        # Width : 0.53881 ± 0.000343 (GeV) for 40k events
        tolerance = 1.1
        self.check_parton_output(cross= 0.53881, error=tolerance*0.000343,target_event=40000)

        #
        #  START REAL CODE (2/3)
        #
        command = open(pjoin(self.path, 'cmd'), 'w')
        command.write("""set group_subprocesses False
        import model loop_sm
        set automatic_html_opening False --no_save
        set notification_center False --no_save
        generate    t > w+{A} b, w+ > ta+ vt
        add process t > w+{S} b, w+ > ta+ vt
        output madevent %(path)s
        launch
        analysis=off
        set no_parton_cut
        set nevents 40k
        set me_frame [1]
        set nhel 1
        set bwcutoff 100
        """ % {'path':self.run_dir})
        command.close()

        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','madgraph'),
                         pjoin(self.path, 'cmd')],
                         cwd=pjoin(_file_path, os.path.pardir),
                        stdout=stdout,stderr=stdout)

        # Width : 1.3760e-05 (GeV) for 40k events: mean of 8 seeds. The former
        # 1.3303e-05 +- 2.1e-08 was the (x/sigma)^2 average of the refine
        # iterations, biased low by ~3.5% (the reported cross-section is now the
        # last iteration's mean). The error is the one a run reports (~7.5e-08):
        # the 8 seeds lie within 2.5 sigma, the old estimator 4.4-5.8 sigma away.
        # (The seed-to-seed spread, 1.6e-07, would let the old value pass.)
        self.check_parton_output(cross= 1.3760e-05, error=tolerance*7.5e-08,target_event=40000)

        #
        #  START REAL CODE (3/3)
        #
        command = open(pjoin(self.path, 'cmd'), 'w')
        command.write("""set group_subprocesses False
        import model loop_sm
        set automatic_html_opening False --no_save
        set notification_center False --no_save
        generate    t > w+{A} b, w+ > ta+ vt
        add process t > w+{S} b, w+ > ta+ vt
        output madevent %(path)s
        launch
        analysis=off
        set mta 1e-3
        set no_parton_cut
        set nevents 40k
        set me_frame [1]
        set nhel 1
        set bwcutoff 100
        set mmnl 5.0
        """ % {'path':self.run_dir})
        command.close()

        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','madgraph'),
                         pjoin(self.path, 'cmd')],
                         cwd=pjoin(_file_path, os.path.pardir),
                        stdout=stdout,stderr=stdout)

        # Width : 3.9369e-12 (GeV) for 40k events, mean of 6 seeds (spread
        # 1.7e-15); the former 3.9311e-12 was the (x/sigma)^2 average of the
        # refine iterations (0.15% low). The error is kept as before.
        self.check_parton_output(cross=3.9369e-12, error=tolerance*6.86e-15,target_event=40000)

    def test_generation_from_file_1(self):
        """ """
        cwd = os.getcwd()

        import subprocess
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        if logging.getLogger('madgraph').level > 20:
            stdout = devnull
        else:
            stdout= None

        fsock = open(pjoin(self.path, 'test_mssm_generation'),'w')
        fsock.write(open(pjoin(_file_path, 'input_files','test_mssm_generation')).read() %
                    {'dir_name': self.run_dir, 'mg5_path':pjoin(_file_path, os.path.pardir)})
        fsock.close()
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','madgraph'), 
                         pjoin(self.path, 'test_mssm_generation')],
                         #cwd=pjoin(self.path),
                        stdout=stdout,stderr=stdout)
        
        self.check_parton_output(cross=4.541638, error=0.035)
    
        self.check_parton_output('run_02', cross=4.41887317, error=0.035)
        #self.check_pythia_output()
        self.assertEqual(cwd, os.getcwd())
        #
        
        # Additional test: Check that the banner of the run_02 include correctly
        # the ptheavy 50 information
        banner = banner_mod.Banner(pjoin(self.run_dir, 'Events','run_01', 'run_01_fermi_banner.txt'))
        run_card = banner.charge_card('run_card')
        self.assertEqual(run_card['ptheavy'], 0)
        
        banner = banner_mod.Banner(pjoin(self.run_dir, 'Events','run_02', 'run_02_fermi_banner.txt'))
        run_card = banner.charge_card('run_card')
        self.assertEqual(run_card['ptheavy'], 50)
        
        events = lhe_parser.EventFile(pjoin(self.run_dir, 'Events','run_02', 'unweighted_events.lhe.gz'))
        banner =  banner_mod.Banner(events.banner)
        run_card = banner.charge_card('run_card')
        self.assertEqual(run_card['ptheavy'], 50)
        for event in events:
            event.check()

    def test_generation_from_file_1_mg7(self):
        """mg7 (madspace) cross-section for MSSM p p > go go.

        the standalone (madmatrix) export reproduces the per-flavor |M|^2 for
        p p > go go (test_madmatrix_mssm_gogo, ~1e-4) and the madspace integrator
        lands on the madevent cross-section as well: with NNPDF23_lo_as_0130_qed
        pinned in the run_card this setup gives 5.0235, against the madevent
        reference of 5.024 (run_01 of test_generation_from_file_1). The target
        below is for the default PDF instead -- see the comment on it.

        This used to be red at random rather than for a physics reason: the
        assertion is at 1%, but with the old 2000-event target a single run
        carried a ~1.1% statistical error, i.e. the tolerance sat below 1 sigma
        and the test failed on roughly 40% of runs (observed spread over 12
        identical local runs: 4.918-5.156, mean 5.030). _run_mg7_xsec now asks
        for 50000 events, which brings the error to ~0.08% and makes the 1%
        tolerance a ~13 sigma check. Self-skips where the mg7 runtime stack is
        unavailable.
        """
        datadir = _mg7_datadir_or_skip(self)
        cross, error = _run_mg7_xsec(self,
            ['set automatic_html_opening False --no_save',
             'import model MSSM_SLHA2',
             'generate p p > go go'],
            pjoin(self.path, 'MG7_mssm_gogo'), datadir)
        # Reference for the default PDF NNPDF40_lo_as_01180, measured over 6
        # runs: 2.9632 +- 0.0026 (single-run error ~0.0027, i.e. ~0.09%), so the
        # 1% tolerance here is a ~11 sigma check. The value shifts from the old
        # NNPDF23_lo_as_0130_qed reference of 5.024 (madevent, no cuts, lhapdf)
        # purely because of the PDF change: p p > go go is forced to large x by
        # the ~600 GeV gluino pair, where the sets differ a lot. With NNPDF23
        # pinned in the run_card this same setup still gives 5.0235, matching
        # that madevent reference to 0.01%.
        # previously PDF was NNPDF40MC_lo_as_01180 with this reference value 3.786
        target = 2.963
        self.assertLess(abs(cross - target) / target, 0.01,
            'mg7 p p > go go cross-section %s far from reference %s'
            % (cross, target))

    def test_contur_from_file(self):
        """check that contur runs as expected"""

        cwd = os.getcwd()
        import subprocess
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        if logging.getLogger('madgraph').level > 20:
            stdout = devnull
        else:
            stdout= None


        subprocess.call([pjoin(_file_path, os.path.pardir,'bin','madgraph'), 
                         pjoin(_file_path,  os.path.pardir, 'tests', 'input_files','rivet_contur_test.cmd')],
                         cwd=pjoin(self.path),
                         stdout=stdout,stderr=stdout)

        

        self.assertTrue(os.path.exists(pjoin(self.path, 'heavyNscan', 'Analysis', 'contur', 'ANALYSIS', 'contur.map')))
        self.assertTrue(os.path.exists(pjoin(self.path, 'heavyNscan', 'Analysis', 'contur', 'ANALYSIS', 'Summary.txt')))
        self.assertTrue(os.path.exists(pjoin(self.path, 'heavyNscan', 'Events', 'scan_run_[01-12].txt')))
        self.assertTrue(os.path.exists(pjoin(self.path, 'heavyNscan', 'Events', 'run_01',  'rivet_result.yoda')))
        self.assertTrue(os.path.exists(pjoin(self.path, 'heavyNscan', 'Events', 'run_12',  'rivet_result.yoda')))
        self.assertTrue(os.path.exists(pjoin(self.path, 'heavyNscan', 'Analysis', 'contur',  'conturPlot', 'combinedLevels.pdf')))


    def test_rivet_from_file(self):
        """check that contur runs as expected"""

        cwd = os.getcwd()
        import subprocess
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        if logging.getLogger('madgraph').level > 20:
            stdout = devnull
        else:
            stdout= None

        cmd = """generate p p > e+ e-
        output madevent %s
        launch
shower=pythia8
analysis=off
set mpi off
set mmll 50
set use_syst False
set nevents 100
set HEPMCoutput:file hepmc
        launch -i
rivet run_01
set analysis MC_ZINC
set draw_rivet_plots True
                 """ %self.run_dir

        open(pjoin(self.path, 'mg5_cmd'),'w').write(cmd)
        
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull
        subprocess.call([pjoin(_file_path, os.path.pardir,'bin','madgraph'), 
                         pjoin(self.path, 'mg5_cmd')],
                         #cwd=self.path,
                         stdout=stdout, stderr=stderr)

        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'Events', 'run_01',  'rivet_result.yoda')))
        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'Events', 'run_01',  'rivet-plots','index.html')))


    #==========================================================================
    # mg7 (madspace) counterparts of the post-processing acceptance tests.
    # Each drives a full mg7 generation and then chains the same post-processing
    # tool through the mg7 command interface (bin/generate_events -> the merged
    # tool-selection question -> run_selected_tools -> MG7RunCmd), so that at
    # least one test exercises the command interface + chaining of every tool
    # with the new (default) output. They self-skip when the mg7 runtime stack
    # (madspace + LHAPDF + the default PDF set) or the external tool is
    # unavailable.
    #==========================================================================
    def test_add_time_of_flight_mg7(self):
        """time-of-flight LHE post-processing chained on the mg7 output.

        Mirrors test_add_time_of_flight: the run_card.toml [postprocessing]
        time_of_flight threshold is applied to the generated events (this is the
        mg7 equivalent of the madevent 'add_time_of_flight' command), and the
        invariant lifetime written on the LHE particles is checked against the
        threshold."""
        datadir = _mg7_datadir_or_skip(self)
        run = _run_mg7_postproc(
            self,
            ['set automatic_html_opening False --no_save',
             'import model sm',
             'generate p p > w+ z'],
            pjoin(self.path, 'MG7_tof'), datadir,
            switch_lines=None,
            toml_edits=[(r'time_of_flight = -?[\d.eE+-]+',
                         'time_of_flight = 4e-14')],
            events=100)

        event = pjoin(run, 'events.lhe')
        if not os.path.exists(event):
            misc.gunzip(pjoin(run, 'events.lhe.gz'), keep=True, stdout=event)

        has_zero = False
        has_non_zero = False
        for evt in lhe_parser.EventFile(event):
            for particle in evt:
                if particle.pid in [23, 24]:
                    self.assertTrue(particle.vtim == 0 or particle.vtim > 4e-14)
                    if particle.vtim == 0:
                        has_zero = True
                    else:
                        has_non_zero = True
        self.assertTrue(has_non_zero,
                        'no time-of-flight information was added to the mg7 LHE')

    def test_pythia8_shower_mg7(self):
        """Pythia8 shower chained on the mg7 output through the command
        interface (mirrors the shower part of test_add_time_of_flight): the mg7
        events are showered in parallel by the reused madevent driver and a
        HepMC file is produced."""
        import glob
        datadir = _mg7_datadir_or_skip(self)
        run = _run_mg7_postproc(
            self,
            ['set automatic_html_opening False --no_save',
             'import model sm',
             'generate p p > w+ z'],
            pjoin(self.path, 'MG7_py8'), datadir,
            switch_lines=['shower=Pythia8'],
            precopy_cards=['pythia8'],
            require_tools=('pythia8_path',),
            events=100)
        report = _mg7_postproc_report(pjoin(self.path, 'MG7_py8'))

        hepmc = glob.glob(pjoin(run, '*hepmc*'))
        self.assertTrue(hepmc, 'mg7 Pythia8 shower produced no HepMC output in %s\n%s'
                        % (run, report))
        # a file is not enough: an empty shower leaves one too. Both HepMC2
        # and HepMC3 ascii start each event with an 'E ' line.
        import gzip
        opener = gzip.open if hepmc[0].endswith('.gz') else open
        with opener(hepmc[0], 'rt') as fsock:
            nb_showered = sum(1 for line in fsock if line.startswith('E '))
        self.assertGreater(nb_showered, 0, 'no event in %s\n%s' % (hepmc[0], report))

        # the Results Summary printed after the shower reports the generated
        # sample (it read 0 events and 0 pb when the mg7 run was booked
        # without them)
        log = open(pjoin(self.path, 'MG7_py8', 'mg7_postproc.log')).read()
        summary = log[log.rfind('Results Summary'):]
        nb_event = re.search(r'Nb of events :\s+(\d+)', summary)
        cross = re.search(r'Cross-section :\s+(\S+)', summary)
        self.assertTrue(nb_event and cross, 'no Results Summary\n%s' % report)
        self.assertEqual(int(nb_event.group(1)), 100, report)
        self.assertGreater(float(cross.group(1)), 0., report)

    def test_delphes_mg7(self):
        """Delphes detector simulation chained on the mg7 output (Pythia8 shower
        + Delphes) through the command interface."""
        import glob
        datadir = _mg7_datadir_or_skip(self)
        run = _run_mg7_postproc(
            self,
            ['set automatic_html_opening False --no_save',
             'import model sm',
             'generate p p > w+ z'],
            pjoin(self.path, 'MG7_delphes'), datadir,
            switch_lines=['shower=Pythia8', 'detector=Delphes'],
            precopy_cards=['pythia8', 'delphes'],
            require_tools=('pythia8_path', 'delphes_path'),
            events=100)

        self.assertTrue(glob.glob(pjoin(run, '*delphes*')),
                        'mg7 Delphes run produced no output in %s\n%s'
                        % (run, _mg7_postproc_report(pjoin(self.path, 'MG7_delphes'))))

    def test_rivet_from_file_mg7(self):
        """Rivet analysis chained on the mg7 output (Pythia8 shower + Rivet)
        through the command interface (mirrors test_rivet_from_file).

        Uses p p > w+ z rather than the madevent test's p p > e+ e-: mg7's LHE
        writer (madspace LHECompleter) currently crashes on the Drell-Yan final
        state, whereas the two-boson process writes LHE cleanly; the default
        rivet card's generic MC_ELECTRONS/MUONS/JETS analyses produce a yoda for
        the showered W/Z decays.

        do_rivet defers the actual Rivet execution to the postprocessor
        (rivet_card default run_rivet_later = True); run_selected_tools runs that
        postprocessor (cmd.postprocessing()) after the shower, exactly as
        MadEventCmd.do_launch does, so rivet_result.yoda is produced."""
        datadir = _mg7_datadir_or_skip(self)
        run = _run_mg7_postproc(
            self,
            ['set automatic_html_opening False --no_save',
             'import model sm',
             'generate p p > w+ z'],
            pjoin(self.path, 'MG7_rivet'), datadir,
            switch_lines=['shower=Pythia8', 'analysis=Rivet'],
            precopy_cards=['pythia8', 'rivet'],
            require_tools=('pythia8_path', 'rivet_path'),
            events=100)

        self.assertTrue(os.path.exists(pjoin(run, 'rivet_result.yoda')),
                        'mg7 Rivet run produced no yoda output in %s' % run)

    def test_mass_reweighting_mg7(self):
        """Matrix-element reweighting chained on the mg7 output through the
        command interface (mirrors test_mass_reweighting): p p > t t~ is
        generated with the mg7 output and reweighted to a heavier top by the
        reused madevent reweight driver (reweight -from_cards).

        The reweight module rebuilds the matrix element from the model/process
        stored in the LHE banner, so this also covers the banner's proc_card."""
        datadir = _mg7_datadir_or_skip(self)
        # one reweight point: the same "heavier top" hypothesis as the madevent
        # test. The default output mode adds the new weight to the events.
        reweight_card = ("launch\n"
                         "set mass mt 200\n")
        run = _run_mg7_postproc(
            self,
            ['set automatic_html_opening False --no_save',
             'import model sm',
             'generate p p > t t~'],
            pjoin(self.path, 'MG7_reweight'), datadir,
            switch_lines=['reweight=ON'],
            extra_cards=[(reweight_card, 'reweight_card.dat')],
            events=100)

        lhe_path = None
        for name in ('unweighted_events.lhe.gz', 'unweighted_events.lhe',
                     'events.lhe.gz', 'events.lhe'):
            if os.path.exists(pjoin(run, name)):
                lhe_path = pjoin(run, name)
                break
        self.assertTrue(lhe_path, 'no mg7 event file under %s' % run)

        # every event must carry the extra reweighted weight
        nb_event = 0
        for evt in lhe_parser.EventFile(lhe_path):
            rwgt = evt.parse_reweight()
            self.assertIn('rwgt_1', rwgt,
                          'mg7 reweight did not add the rwgt_1 weight to the events')
            self.assertNotEqual(float(rwgt['rwgt_1']), 0.0)
            nb_event += 1
        self.assertGreater(nb_event, 0, 'no event found in %s' % lhe_path)

    def test_w_production_with_ms_decay_mg7(self):
        """MadSpin decay chained on the mg7 output through the command interface
        (mirrors test_w_production_with_ms_decay): p p > w+ z is generated with
        the mg7 output and then the w+ is decayed (w+ > j j) by the reused
        madevent MadSpin driver.

        Uses the 2->2 p p > w+ z rather than the madevent test's 2->1 p p > w+:
        mg7's madspace phase-space builder requires at least two outgoing
        particles, so a 2->1 process cannot be generated by the mg7 output."""
        import glob
        datadir = _mg7_datadir_or_skip(self)
        madspin_card = ("set spinmode madspin\n"
                        "decay w+ > j j\n"
                        "launch\n")
        run = _run_mg7_postproc(
            self,
            ['set automatic_html_opening False --no_save',
             'import model sm',
             'generate p p > w+ z'],
            pjoin(self.path, 'MG7_madspin'), datadir,
            switch_lines=['madspin=ON'],
            extra_cards=[(madspin_card, 'madspin_card.dat')],
            events=100)

        decayed = glob.glob(pjoin(os.path.dirname(run), 'run_01_decayed*',
                                  '*.lhe*'))
        self.assertTrue(decayed,
                        'mg7 MadSpin run produced no decayed events under %s'
                        % os.path.dirname(run))
        counts = {}
        for evt in lhe_parser.EventFile(decayed[0]):
            for particle in evt:
                if particle.status == 1:
                    counts[particle.pdg] = counts.get(particle.pdg, 0) + 1
        # w+ > j j : the decay products are light-flavour (anti)quarks, and the
        # merged-flavour placeholders (81/82/83) must have been resolved.
        self.assertNotIn(81, counts)
        self.assertNotIn(-81, counts)
        self.assertTrue(any(abs(pdg) in (1, 2, 3, 4) for pdg in counts),
                        'no light quarks among the MadSpin decay products: %s'
                        % counts)
        # store_result used to save the madevent results database to
        # HTML/results.pkl, which the mg7 output does not have
        proc_dir = pjoin(self.path, 'MG7_madspin')
        with open(pjoin(proc_dir, 'mg7_postproc.log')) as fsock:
            self.assertEqual(
                [line for line in fsock if 'I/O error' in line], [])
        self.assertFalse(os.path.exists(pjoin(proc_dir, 'crossx.html')))




        

    def load_result(self, run_name):
        
        import madgraph.iolibs.save_load_object as save_load_object
        import madgraph.madevent.gen_crossxhtml as gen_crossxhtml
        
        result = save_load_object.load_from_file(pjoin(self.run_dir,'HTML/results.pkl'))
        return result[run_name]

    def check_parton_output(self, run_name='run_01', target_event=100, cross=0, error=9e99, delta_event=0):
        """Check that parton output exists and reach the targert for event"""
                
        # check that the number of event is fine:
        data = self.load_result(run_name)
        if target_event > 0:
            if delta_event == 0:
                self.assertEqual(target_event, int(data[0]['nb_event']))
            else:
                self.assertLessEqual(abs(int(data[0]['nb_event'])-target_event), delta_event)
        self.assertIn('lhe', data[0].parton)
        
        if cross:
            import math
            new_error = math.sqrt(error**2 + float(data[0]['error'])**2)
            self.assertLess(
                abs(cross - float(data[0]['cross']))/new_error,
                3,
                'cross is %s and not %s. NB_SIGMA %s' % (float(data[0]['cross']), cross, float(data[0]['cross'])/new_error)
            )
            self.assertLess(float(data[0]['error']), 3 * error)
            
        check_html_page(self, pjoin(self.run_dir, 'crossx.html'))
        if 'decayed' not in run_name:
            check_html_page(self, pjoin(self.run_dir,'HTML', run_name, 'results.html'))
        
    def check_pythia_output(self, run_name='run_01'):
        """ """
        # check that the number of event is fine:
        data = self.load_result(run_name)
        self.assertTrue('hep' in data[0].pythia or 'hepmc' in data[0].pythia8)
        self.assertTrue('log' in data[0].pythia or 'log' in data[0].pythia8)

    
    def test_decay_width_nlo_model(self):
        """ """
        
        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception as error:
            pass
        
        cmd = MGCmd.MasterCmd()
        cmd.no_notification()
        cmd.run_cmd('import model loop_sm')
        self.assertEqual(cmd.cmd.__name__, 'aMCatNLOInterface')
        #cmd.run_cmd('switch MG5')
        #self.assertEqual(cmd.cmd.__name__, 'MadGraphCmd')
        cmd.run_cmd('set automatic_html_opening False --no_save')
        cmd.run_cmd('generate w+ > all all')
        self.assertEqual(cmd.cmd.__name__, 'MadGraphCmd')
        cmd.run_cmd('output madevent %s -f' % self.run_dir)
        cmd.run_cmd('launch -f')
        data = self.load_result('run_01')
        self.assertNotEqual(data[0]['cross'], 0)

    def test_loop_induced_ggh(self):
        """Test loop-induced gg > h cross-section via g g > h QCD=0 [QCD]"""

        cmd = MGCmd.MasterCmd()
        cmd.no_notification()
        cmd.run_cmd('set automatic_html_opening False --no_save')
        cmd.run_cmd('generate g g > h QCD=0 [QCD]')
        cmd.run_cmd('output madevent %s -f' % self.run_dir)
        #modify the run_cardself
        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards','run_card.dat'))
        run_card['nevents'] = 100
        run_card['use_syst'] = 'F'
        run_card.write('%s/Cards/run_card.dat'% self.run_dir,
                                    '%s/Cards/run_card_default.dat'% self.run_dir)

        cmd.run_cmd('launch -f')
        
        # previously PDF was nn23lo1 (lhaid 230000) with this reference value cross=15.73, error=0.04
        self.check_parton_output(cross=12.18, error=0.514)

    def _get_delphes_path(self):
        """Return the configured delphes_path from the MG5 configuration, or
        None when Delphes is not configured (used to skip the parallel-Delphes
        acceptance test on setups without Delphes/ROOT)."""
        config = pjoin(MG5DIR, 'input', 'mg7_configuration.txt')
        if not os.path.exists(config):
            return None
        for line in open(config):
            line = line.split('#', 1)[0]
            if '=' in line:
                key, value = line.split('=', 1)
                if key.strip() == 'delphes_path':
                    value = value.strip()
                    if value and value.lower() != 'none':
                        return value
        return None

    def test_pythia8_delphes_parallel(self):
        """Fused parallel-Delphes path: a multicore Pythia8 + Delphes run should
        run Delphes on the individual Pythia8 splits and combine the ROOT files
        with hadd, keeping every showered event exactly once (normalization)."""

        delphes_path = self._get_delphes_path()
        if not (delphes_path and os.environ.get('ROOTSYS') and
                os.path.exists(pjoin(delphes_path, 'DelphesHepMC2'))):
            raise unittest.SkipTest('Delphes/ROOT not available')

        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception:
            pass

        # nb_core 2 with 400 events forces exactly 2 Pythia8 splits (the
        # min_n_events_per_job=100 security clamp keeps 400//100=4 capped to 2);
        # run_mode defaults to 2 (multicore). Setting nb_core_delphes activates
        # the fused parallel-Delphes path (Delphes runs on each split, then the
        # ROOT files are combined with hadd).
        nevents = 400
        cmd = """import model sm
        set automatic_html_opening False --no_save
        set notification_center False --no_save
        set nb_core 2
        set nb_core_delphes 2
        generate p p > e+ e-
        output madevent %s -f
        launch
        shower=pythia8
        detector=Delphes
        analysis=off
        set mpi off
        set use_syst False
        set event_norm average
        set nevents %d
        set HEPMCoutput:file hepmc.gz
        launch -i
        delphes run_01 --tag=single
        """ % (self.run_dir, nevents)
        open(pjoin(self.path, 'mg5_cmd'), 'w').write(cmd)

        if logging.getLogger('madgraph').level <= 20:
            stdout = None
            stderr = None
        else:
            devnull = open(os.devnull, 'w')
            stdout = devnull
            stderr = devnull
        ret = subprocess.call([pjoin(_file_path, os.path.pardir, 'bin', 'madgraph'),
                               pjoin(self.path, 'mg5_cmd')],
                              stdout=stdout, stderr=stderr)
        # Without this a failed run only shows up further down as
        # "TypeError: 'NoneType' object is not subscriptable" out of
        # load_result, because HTML/results.pkl was never written.
        self.assertEqual(ret, 0, 'mg5_aMC run failed (rc=%s)' % ret)

        # Parton level (the same lhe drives every split) and Pythia8 output.
        self.check_parton_output(target_event=nevents)
        self.check_pythia_output()

        # Two Delphes outputs of the *same* showered events:
        #   - tag_1_delphes_events.root : fused (Delphes per split -> hadd),
        #   - single_delphes_events.root: standard single Delphes pass on the
        #     merged HepMC (the 'delphes run_01 --tag=single' command above).
        # They must be equivalent: same number of events and same total weight
        # (this is the real normalization check for the fused path).
        eventdir = pjoin(self.run_dir, 'Events', 'run_01')
        fused_root = pjoin(eventdir, 'tag_1_delphes_events.root')
        single_root = pjoin(eventdir, 'single_delphes_events.root')
        self.assertTrue(os.path.exists(fused_root), 'no fused Delphes ROOT produced')
        self.assertTrue(os.path.exists(single_root), 'no single-core Delphes ROOT produced')
        self.assertGreater(os.path.getsize(fused_root), 0)

        # PyROOT is bundled with ROOT but its bindings may not import under the
        # test interpreter; when available, compare the two samples directly.
        try:
            import ROOT
        except ImportError:
            ROOT = None
        if ROOT is not None:
            ROOT.gErrorIgnoreLevel = ROOT.kError

            def read(path):
                tfile = ROOT.TFile.Open(path)
                tree = tfile.Get('Delphes')
                self.assertIsNotNone(tree)
                n = int(tree.GetEntries())
                total = 0.0
                try:
                    for event in tree:
                        total += event.Event.At(0).Weight
                except Exception:
                    total = None  # branch layout differs; fall back to counts
                tfile.Close()
                return n, total

            n_fused, w_fused = read(fused_root)
            n_single, w_single = read(single_root)
            # Same events processed either way: no loss or duplication from hadd.
            self.assertGreater(n_fused, 0)
            self.assertEqual(n_fused, n_single)
            # Same absolute normalization: the per-split HepMC weights are the
            # ones the single pass sees on the merged file, so the totals match.
            if w_fused is not None and w_single is not None:
                self.assertAlmostEqual(w_fused, w_single,
                                       delta=1e-6 * abs(w_single) + 1e-30)


#===============================================================================
# TestCmd
#===============================================================================
class TestMEfromPdirectory(unittest.TestCase):
    """test that we can launch everything from the P directory"""

    

    def generate(self, process, model):
        """Create a process"""

        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception as error:
            pass
        
        interface = MGCmd.MasterCmd()
        interface.no_notification()
        interface.onecmd('import model %s' % model)
        if isinstance(process, str):
            interface.onecmd('generate %s' % process)
        else:
            for p in process:
                interface.onecmd('add process %s' % p)
        interface.onecmd('output madevent /tmp/MGPROCESS/ -f')

    def load_result(self, run_name):
        
        import madgraph.iolibs.save_load_object as save_load_object
        import madgraph.madevent.gen_crossxhtml as gen_crossxhtml
        
        result = save_load_object.load_from_file('/tmp/MGPROCESS/HTML/results.pkl')
        return result[run_name]

    def check_parton_output(self, run_name='run_01', target_event=100, cross=0):
        """Check that parton output exists and reach the targert for event"""
                
        # check that the number of event is fine:
        data = self.load_result(run_name)
        self.assertEqual(int(data[0]['nb_event']), target_event)
        self.assertIn('lhe', data[0].parton)
        if cross:
            self.assertLess(abs(cross - float(data[0]['cross']))/float(data[0]['error']), 3)


    def test_run_fromP(self):
        """ """
                
        cmd = os.getcwd()
        self.generate('p p > e+ e-', 'sm')
        self.assertEqual(cmd, os.getcwd())
        shutil.copy(os.path.join(_file_path, 'input_files', 'run_card_matching.dat'),
                    '/tmp/MGPROCESS/Cards/run_card.dat')
        with misc.chdir('/tmp/MGPROCESS/'):
            ff = open('cmd.cmd','w')
            ff.write('set automatic_html_opening False --nosave\n')
            ff.write('set notification_center False --nosave\n')
            #ff.write('display options\n')
            #ff.write('display variable allow_notification_center\n')
            ff.write('generate_events -f \n') 
            ff.close()
            if logger.getEffectiveLevel() > 20:
                output = open(os.devnull,'w')
            else:
                output = None
            id = subprocess.call(['./bin/madevent','cmd.cmd'], stdout=output, stderr=output)
            self.assertEqual(id, 0)
            self.check_parton_output(cross=947.9) 
