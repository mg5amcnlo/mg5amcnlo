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
from __future__ import division
from __future__ import absolute_import
import subprocess
import unittest
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
                continue # means is just a linke starting with #
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
        try:
            datadir = subprocess.check_output(
                ['lhapdf-config', '--datadir']).decode().strip()
        except Exception:
            datadir = None
    if not has_mg7 or not datadir or not os.path.isdir(datadir):
        test.skipTest('mg7 runtime stack (madspace + LHAPDF data) unavailable')
    if not glob.glob(pjoin(datadir, 'NNPDF23_lo_as_0130_qed*')):
        test.skipTest('NNPDF23_lo_as_0130_qed PDF set not available')
    return datadir


def _run_mg7_xsec(test, setup_cmds, run_dir, datadir):
    """Run an mg7 cross-section: execute `setup_cmds` (MG5 lines, ending with
    the generate), `output mg7 run_dir`, drive bin/generate_events with the
    dynamical HT/2 scale + trimmed event target, and return (cross, error) from
    the madspace info.json (process.mean / process.error). Assertions are made
    on the *test* instance."""
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
    t = re.sub(r'events = \d+', 'events = 2000', t)
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

        #if not os.path.exists(pjoin(MG5DIR, 'MadAnalysis')):
        #    print("install MadAnalysis")
        #    p = subprocess.Popen([pjoin(MG5DIR,'bin','mg5_aMC')],
        #                     stdin=subprocess.PIPE,
        #                     stdout=stdout,stderr=stderr)
        #    out = p.communicate('install MadAnalysis4'.encode())
        #misc.compile(cwd=pjoin(MG5DIR,'MadAnalysis'))

        #if not misc.which('root'):
        #    raise Exception('root is require for this test')
        #interface.exec_cmd('set pythia-pgs_path %s --no_save' % pjoin(MG5DIR, 'pythia-pgs'))
        interface.exec_cmd('set madanalysis_path %s --no_save' % pjoin(MG5DIR, 'MadAnalysis'))
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
        # Make sure that the cross-section is close enough for the bias and unbiased samples
        self.assertLess((abs(biased_events.cross-unbiased_events.cross)/abs(unbiased_events.cross)),0.1)

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
        shutil.copy('%s/Cards/plot_card_default.dat' % self.run_dir,
                    '%s/Cards/plot_card.dat' % self.run_dir)        
        try:
            os.remove(pjoin(self.run_dir, 'Cards',  'madanalysis5_parton_card.dat'))
            os.remove(pjoin(self.run_dir, 'Cards',  'madanalysis5_hadron_card.dat'))
        except:
            pass
        self.do('generate_events -f')     


        f1 = self.check_matched_plot(tag='fermi')         
        start = time.time()
        
        #modify the run_card
        run_card = self.cmd_line.run_card
        run_card['nevents'] = 44
        run_card['use_syst'] = 'F'
        run_card.write('%s/Cards/run_card.dat'% self.run_dir,
                                    '%s/Cards/run_card_default.dat'% self.run_dir)

        self.assertEqual(cmd, os.getcwd())        
        self.do('generate_events -f')
        self.assertEqual(int(self.cmd_line.run_card['nevents']), 44)
        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'Cards', 'plot_card.dat')))
        self.do('pythia run_01 -f')
        self.do('quit')
        
        self.assertEqual(int(self.cmd_line.run_card['nevents']), 100)
        
        self.check_parton_output(syst=False)
        self.check_parton_output('run_02', target_event=44, syst=False)
        self.check_pythia_output(syst=False)        
        f2 = self.check_matched_plot(mintime=start, tag='tag_1')        
        
        self.assertNotEqual(f1.split('\n'), f2.split('\n'))
        
        
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
        
        self.assertLess(abs(val2 - val1) / (err1 + err2), 5)
        target = 1310200.0
        self.assertLess(abs(val2 - target) / (err2), 5)
        #check precision
        self.assertLess(err2 / val2, 0.005)
        self.assertLess(err1 / val1, 0.005)

    def test_group_subprocess_mg7(self):
        """mg7 equivalent of test_group_subprocess for u u > u u.

        Runs the mg7 (madspace) integrator with group_subprocesses on and off
        and checks the two cross-sections agree (grouping consistency). It also
        pins the absolute value to the mg7-native result obtained with the
        run_card.toml defaults (NNPDF23_lo_as_0130_qed + dynamical HT/2 scale,
        events=2000) ~ 2.21e5 pb.

        NOTE: this is NOT the madevent reference (1.31e6 pb in
        test_group_subprocess); the two are not directly comparable (different
        PDF/scale defaults) and there is moreover a known mg7 normalisation
        discrepancy. The check is intentionally a live (non-xfail) guard on the
        current mg7 result and should be revisited when the mg7 integrator
        normalisation is resolved.
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
        if not glob.glob(pjoin(datadir, 'NNPDF23_lo_as_0130_qed*')):
            self.skipTest('NNPDF23_lo_as_0130_qed PDF set not available')

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
        # mg7-native reference (see docstring) -- NOT the madevent 1.31e6 value.
        target = 221409.0
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
        self.assertAlmostEqual(cross, 6.124, delta=max(0.1, 5 * error))

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
        self.assertAlmostEqual(cross, 3.215, delta=max(0.1, 5 * error))

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
        self.assertAlmostEqual(cross, 3.215, delta=max(0.1, 5 * error))

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
        self.assertAlmostEqual(cross, 4428.0, delta=max(30.0, 5 * error))

    def test_madevent_merged_flavor_uq_mg7(self):
        """mg7 equivalent of test_madevent_merged_flavor_uq (u q > u q QCD=0,
        q = u d).

        KNOWN-FAILING, intentionally NOT marked xfail: the mg7 (madspace)
        integrator currently SEGFAULTS on this merged-flavor process, so
        generate_events does not produce the physical cross-section (4428 pb).
        The test asserts the physical reference and is therefore expected to
        fail until the mg7 integrator handles merged-flavor u q > u q; it is
        left undecorated so the failure stays visible in the mg7 workflow
        rather than being silently swallowed by expectedFailure. It still
        self-skips where the mg7 runtime stack is unavailable.
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
        if not glob.glob(pjoin(datadir, 'NNPDF23_lo_as_0130_qed*')):
            self.skipTest('NNPDF23_lo_as_0130_qed PDF set not available')

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
        # physical reference (same as the madevent test); mg7 must reproduce it
        self.assertAlmostEqual(cross, 4428.0, delta=max(30.0, 5 * error))

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

            val = self.cmd_line.results.current['cross'] + 1e-99
            err = self.cmd_line.results.current['error']
            results.append((val, err, afg, gsp))

            if val == 0:
                misc.sprint('Warning: cross-section is zero for '
                             'apply_flavor_grouping=%s/group_subprocesses=%s' % (afg, gsp))

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

        KNOWN-FAILING, intentionally NOT marked xfail: mg7 currently returns
        cross-sections that depend on the apply_flavor_grouping setting (e.g.
        ~1332 pb grouped vs ~511 pb ungrouped) because the broken-symmetry
        (flavour-consolidation) factor implemented for standalone /
        standalone_cpp is not yet applied on the mg7 (madmatrix) side. The four
        settings must agree; the test asserts that and is left undecorated so the
        mg7 flavour-grouping discrepancy stays visible until broken_sym is ported
        to mg7. It self-skips where the mg7 runtime stack is unavailable.
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

            val = self.cmd_line.results.current['cross'] + 1e-99
            err = self.cmd_line.results.current['error']
            results.append((val, err, afg, gsp))

            if val == 0:
                misc.sprint('Warning: cross-section is zero for '
                             'apply_flavor_grouping=%s/group_subprocesses=%s' % (afg, gsp))

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

        results = [(5184588.926738217,2971, '3.7.2', 'neventa=150k')]
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
            val = self.cmd_line.results.current['cross'] + 1e-99
            err = self.cmd_line.results.current['error']
            results.append((val, err, afg, gsp))

            # Check that we got a valid cross-section
            self.assertGreater(val, 0,
                'cross-section is zero for q q~ > q q~ with MLM merging')

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
            'large diference between %s and %s +- %s'%
                        (target, val1, err1)
        )


    def test_e_p_collision_mg7(self):
        """mg7 equivalent of test_e_p_collision for e- p > e- j.

        The madevent test pins the cross-section (3933 pb) via a full
        generate_events run. The mg7 (madmatrix/cudacpp) integrator is far too
        slow for a CI cross-section run on this process, so this equivalent
        validates the mg7 *output* path instead: that `output mg7` generates the
        e- p > e- j subprocess directories and that they compile (scalar cppnone
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
        self.assertTrue(abs(val1 - target) / err1 < 2., 'large diference between %s and %s +- %s (%s sigma)'%
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
        self.assertTrue(abs(val1 - target) / err1 < 2., 'large diference between %s and %s +- %s (%s sigma)'%
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
        self.assertTrue(abs(val1 - target) / err1 < 2., 'large diference between %s and %s +- %s (%s sigma)'%
                        (target, val1, err1, abs(val1 - target) / err1))


    def test_customised_madevent_via_run_card(self):
        """checking various advanced functionality of customization
           - set run_card entry via input/default_run_card_lo.dat
           - check that unknow entry can be added to the run_card.dat
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

        target = 361.7 #+- 0.1037 pb
        self.assertTrue(abs(val1 - target) / (2*err1) < 1., 'large diference between %s and %s +- %s'%
                        (target, val1, err1))

        self.assertIn('MY_PARAM', open(pjoin(self.run_dir,'Source','run.inc')).read())
        self.assertEqual(2, open(pjoin(self.run_dir,'Source','run.inc')).read().count('autodef'))

        self.assertIn('MY_PARAM', open(pjoin(self.run_dir,'Source','run_card.inc')).read())
        self.assertIn('MY_PARAM', open(pjoin(self.run_dir,'SubProcesses','dummy_fct.f')).read())


    def test_customised_madevent_via_run_card(self):
        """checking various advanced functionality of customization
           - set run_card entry via input/default_run_card_lo.dat
           - check that unknow entry can be added to the run_card.dat
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

        target = 361.7 #+- 0.1037 pb
        self.assertTrue(abs(val1 - target) / (2*err1) < 1., 'large diference between %s and %s +- %s'%
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
        # 100k value is 165.84 +- 0.05
        target = 165.84
        self.assertTrue(abs(val1 - target) / err1 < 1., 'large diference between %s and %s +- %s'%
                        (target, val1, err1))

        
        # edit run_card -> fix scale
        run_card['fixed_extra_scale'] = True
        run_card['mue_ref_fixed'] = 250

        run_card.write('%s/Cards/run_card.dat' % self.run_dir)
        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        # 100k value is  165.71 +- 0.06
        target = 165.71
        self.assertTrue(abs(val1 - target) / err1 < 1., 'large diference between %s and %s +- %s'%
                        (target, val1, err1))






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
        import models.check_param_card as check_param_card    
        param_card = check_param_card.ParamCard(pjoin(self.run_dir, 'Cards', 'param_card.dat'))
        self.assertTrue(misc.equal(1.491257, param_card['decay'].get(6).value),3)
                        
        # generate events
        self.cmd_line.exec_cmd('launch -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        target = 440.779
        self.assertTrue(misc.equal(target, val1, 4*err1))                
        

        # run madspin
        fsock = open(pjoin(self.run_dir, 'Cards', 'madspin_card.dat'),'w')
        fsock.write('decay t > w+ b \n launch')
        fsock.close()
        
        self.cmd_line.exec_cmd('decay_events run_01 -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        target = 440.779
        self.assertTrue(misc.equal(target, val1, 4*err1))          
             
        
        
        
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

        subprocess.call([pjoin(_file_path, os.path.pardir,'bin','mg5_aMC'), 
                         pjoin(self.path, 'cmd')],
                         #cwd=pjoin(self.path),
                        stdout=stdout,stderr=stdout)
        
        # check that the scan was done
        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'Events', 'run_04')))
        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'Events', 'scan_run_0[1-2].txt')))
        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'Events', 'scan_run_0[3-4].txt')))
        
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
        self.assertLess(abs(val1 - target) / err1, 2.)

    def test_e_e_collision_mg7(self):
        """mg7 equivalent of test_e_e_collision for e+ e- > e+ e-.

        KNOWN-FAILING, intentionally NOT marked xfail: mg7 has no lepton-beam
        (no-PDF / lpp=0) support yet -- its run_card.toml only carries a hadron
        PDF, so generate_events aborts with "PID 11 not found in pdf grid". The
        test asserts the physical cross-section (155.9 pb) and is expected to
        fail until mg7 supports lepton beams; left undecorated so the limitation
        stays visible. Self-skips where the mg7 runtime stack is unavailable.
        """
        datadir = _mg7_datadir_or_skip(self)
        cross, error = _run_mg7_xsec(self, 
            ['set automatic_html_opening False --no_save',
             'import model sm',
             'generate e+ e- > e+ e-'],
            pjoin(self.path, 'MG7_ee'), datadir)
        # physical reference (same as test_e_e_collision); mg7 must reproduce it
        self.assertAlmostEqual(cross, 155.9, delta=max(2.0, 5 * error))

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
            # check that the code was runned correctly
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

    def check_matched_plot(self, run_name='run_01', mintime=None, tag='fermi'):
        """ """
        path = '%(path)s/HTML/%(run)s/plots_pythia_%(tag)s/DJR1.ps' % \
                                {'path':self.run_dir,'run': run_name, 'tag': tag}

        self.assertTrue(os.path.exists(path))
        
        if mintime:
            self.assertGreater(os.path.getctime(path), mintime)
        
        return open(path).read()
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
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','mg5_aMC'), 
                         pjoin(self.path, 'mg5_cmd')],
                         #cwd=self.path,
                        stdout=stdout, stderr=stderr)

        self.check_parton_output(cross=15.62, error=0.19)
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
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','mg5_aMC'),
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
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','mg5_aMC'),
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
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','mg5_aMC'),
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
        Leading to an assymetry in the DY +1j process.
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
        set auto_ptj_mmjj False
        set ptj 0.01
        """ % {'path':self.run_dir})
        command.close()
        
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','mg5_aMC'), 
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

        self.assertTrue(0.49<count[0]/10000.<0.51)       
        self.assertTrue(0.49<count[1]/10000.<0.51)


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
        set nevents 1000
        set shower none
        """ % {'path':self.run_dir})
        command.close()
        
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','mg5_aMC'), 
                         pjoin(self.path, 'cmd')],
                         cwd=pjoin(_file_path, os.path.pardir),
                        stdout=stdout,stderr=stdout)     
        
        #a=rwa_input('freeze')
        self.check_parton_output(cross= 4.117e+08, error=1.413e+06,target_event=1000)

    def test_generation_heft_mg7(self):
        """mg7 equivalent of test_generation_heft for g g > b b~ HIW<=1 (HEFT).

        KNOWN-FAILING, intentionally NOT marked xfail: mg7 runs this HEFT process
        but its cross-section comes out ~257x below the physical value (~1.6e6 pb
        vs the madevent 4.117e8 pb) -- a large mg7 normalisation discrepancy for
        the effective ggH coupling. The test asserts the physical reference and is
        expected to fail until that is resolved; left undecorated to keep the
        discrepancy visible. Self-skips where the mg7 runtime stack is unavailable.
        """
        datadir = _mg7_datadir_or_skip(self)
        cross, error = _run_mg7_xsec(self, 
            ['set automatic_html_opening False --no_save',
             'import model heft',
             'generate g g > b b~ HIW<=1'],
            pjoin(self.path, 'MG7_heft'), datadir)
        # physical reference (same as test_generation_heft)
        target = 4.117e8
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

        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','mg5_aMC'),
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

        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','mg5_aMC'),
                         pjoin(self.path, 'cmd')],
                         cwd=pjoin(_file_path, os.path.pardir),
                        stdout=stdout,stderr=stdout)

        # Width : 1.3303e-05 ± 2.1e-08 (GeV) for 40k events
        self.check_parton_output(cross= 1.3303e-05, error=tolerance*2.1e-08,target_event=40000)

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

        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','mg5_aMC'),
                         pjoin(self.path, 'cmd')],
                         cwd=pjoin(_file_path, os.path.pardir),
                        stdout=stdout,stderr=stdout)

        # Width : 3.9311e-12 ± 6.86e-15  (GeV) for 40k events
        self.check_parton_output(cross=3.9311e-12, error=tolerance*6.86e-15,target_event=40000)

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
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','mg5_aMC'), 
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


        subprocess.call([pjoin(_file_path, os.path.pardir,'bin','mg5_aMC'), 
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
        subprocess.call([pjoin(_file_path, os.path.pardir,'bin','mg5_aMC'), 
                         pjoin(self.path, 'mg5_cmd')],
                         #cwd=self.path,
                         stdout=stdout, stderr=stderr)

        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'Events', 'run_01',  'rivet_result.yoda')))
        self.assertTrue(os.path.exists(pjoin(self.run_dir, 'Events', 'run_01',  'rivet-plots','index.html')))





        

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
        self.check_parton_output(cross=15.72, error=0.01514)


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
