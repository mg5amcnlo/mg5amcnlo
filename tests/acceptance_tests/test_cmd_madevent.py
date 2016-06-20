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
import subprocess
import unittest
import os
import re
import shutil
import sys
import logging
import time
import tempfile   

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

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd

pjoin = os.path.join

    
    


#===============================================================================
# TestCmd
#===============================================================================
class TestMECmdShell(unittest.TestCase):
    """this treats all the command not related to MG_ME"""
    
    def setUp(self):
        
        debugging = False
        if debugging:
            self.path = pjoin(MG5DIR, "tmp_test")
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            os.mkdir(pjoin(MG5DIR, "tmp_test"))
        else:
            self.path = tempfile.mkdtemp(prefix='acc_test_mg5')
        self.run_dir = pjoin(self.path, 'MGPROC') 
    
    def tearDown(self):

        if self.path != pjoin(MG5DIR, "tmp_test"):
            shutil.rmtree(self.path)
    
    def generate(self, process, model):
        """Create a process"""

        try:
            shutil.rmtree(self.run_dir)
        except Exception, error:
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

        if not os.path.exists(pjoin(MG5DIR, 'pythia-pgs')):
            print "install pythia-pgs"
            p = subprocess.Popen([pjoin(MG5DIR,'bin','mg5')],
                             stdin=subprocess.PIPE,
                             stdout=stdout,stderr=stderr)
            out = p.communicate('install pythia-pgs')
        misc.compile(cwd=pjoin(MG5DIR,'pythia-pgs'))
        if not os.path.exists(pjoin(MG5DIR, 'MadAnalysis')):
            print "install MadAnalysis"
            p = subprocess.Popen([pjoin(MG5DIR,'bin','mg5')],
                             stdin=subprocess.PIPE,
                             stdout=stdout,stderr=stderr)
            out = p.communicate('install MadAnalysis')
        misc.compile(cwd=pjoin(MG5DIR,'MadAnalysis'))

        if not misc.which('root'):
            raise Exception, 'root is require for this test'
        interface.exec_cmd('set pythia-pgs_path %s --no_save' % pjoin(MG5DIR, 'pythia-pgs'))
        interface.exec_cmd('set madanalysis_path %s --no_save' % pjoin(MG5DIR, 'MadAnalysis'))
        interface.onecmd('output madevent %s -f' % self.run_dir)            
        
        if not os.path.exists(pjoin(interface.options['syscalc_path'],'sys_calc')):
            print "install SysCalc"
            interface.onecmd('install SysCalc')
        
        
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.options['syscalc_path'] = pjoin(MG5DIR, 'SysCalc')
        
    
    @staticmethod
    def join_path(*path):
        """join path and treat spaces"""     
        combine = os.path.join(*path)
        return combine.replace(' ','\ ')        
    
    def do(self, line):
        """ exec a line in the cmd under test """        
        self.cmd_line.run_cmd(line)
        
  
    def test_madspin_gridpack(self):

        self.out_dir = self.run_dir
        self.generate('g g > t t~', 'sm')

        #put the MadSpin card
        ff = open(pjoin(self.out_dir, 'Cards/madspin_card.dat'), 'w')
        orig_card =  open(pjoin(self.out_dir, 'Cards/madspin_card_default.dat')).read()
        ff.write('set ms_dir %s' % pjoin(self.out_dir, 'MSDIR1'))
        ff.write(orig_card)
        ff.close()
        
        #reduce the number of events
        files.cp(pjoin(_file_path, 'input_files', 'run_card_matching.dat'),
                 pjoin(self.out_dir, 'Cards/run_card.dat'))

        #create the gridpack        
        self.do('launch -f')
        
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
        self.do('calculate_decay_widths -f')        
        
        # test the param_card is correctly written
        self.assertTrue(os.path.exists('%s/Events/run_01/param_card.dat' % self.run_dir))
        
        text = open('%s/Events/run_01/param_card.dat' % self.run_dir).read()
        data = text.split('DECAY  23')[1].split('DECAY',1)[0]
        data = data.split('\n')
        width = float(data[0])
        self.assertAlmostEqual(width, 1.492240e+00, delta=1e-4)
        values = {(3,-3): 2.493165e-01,
                  (1,-1): 2.493165e-01,
                  (4,-4): 1.944158e-01,
                  (2,-2): 1.944158e-01,
                  (-11,11): 5.626776e-02,
                  (-13,13): 5.626776e-02}
        for l in data[1:]:
            if l.startswith("#"):
                continue
            l = l.strip()
            if not l:
                continue
            #2.493165e-01   2    3  -3 # 0.37204
            br, _, id1,id2,_,_ = l.split()
            
            self.assertAlmostEqual(float(br), values[(int(id1),int(id2))],delta=1e-3)
        
        
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
        self.assertAlmostEqual(2.42823,value, delta=1e-3)
        pattern = re.compile(r"decay\s+24\s+([+-.\de]*)", re.I)
        value = float(pattern.search(text).group(1))
        self.assertAlmostEqual(2.028440,value, delta=1e-3)
        pattern = re.compile(r"decay\s+25\s+([+-.\de]*)", re.I)
        value = float(pattern.search(text).group(1))
        self.assertAlmostEqual(3.514960e-03,value, delta=1e-3)
        pattern = re.compile(r"decay\s+6\s+([+-.\de]*)", re.I)
        value = float(pattern.search(text).group(1))
        self.assertAlmostEqual(1.354080,value, delta=5e-3)        
        



        
    def test_creating_matched_plot(self):
        """test that the creation of matched plot works and the systematics as well"""

        cmd = os.getcwd()
        self.generate('p p > W+', 'sm')
        self.assertEqual(cmd, os.getcwd())        
        shutil.copy(os.path.join(_file_path, 'input_files', 'run_card_matching.dat'),
                    '%s/Cards/run_card.dat' % self.run_dir)
        shutil.copy('%s/Cards/pythia_card_default.dat' % self.run_dir,
                    '%s/Cards/pythia_card.dat' % self.run_dir)
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
        self.do('pythia run_01 -f')
        self.do('quit')
        
        self.assertEqual(int(self.cmd_line.run_card['nevents']), 100)
        
        self.check_parton_output(syst=True)
        self.check_parton_output('run_02', target_event=44, syst=False)
        self.check_pythia_output(syst=True)        
        f2 = self.check_matched_plot(mintime=start, tag='tag_1')        
        
        self.assertNotEqual(f1.split('\n'), f2.split('\n'))
        
        
        self.assertEqual(cmd, os.getcwd())
        
    def test_group_subprocess(self):
        """check that both u u > u u gives the same result"""
        

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
        mg_cmd.exec_cmd(' generate u u > u u')
        mg_cmd.exec_cmd('output %s/'% self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir= self.run_dir)
        self.cmd_line.no_notification()
        
        self.cmd_line.exec_cmd('set automatic_html_opening False')
        
        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        
        self.run_dir = pjoin(self.path, 'MGPROC2')
        mg_cmd.exec_cmd('set group_subprocesses False')
        mg_cmd.exec_cmd('generate u u > u u')
        mg_cmd.exec_cmd('output %s' % self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir= self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')

        
        self.do('generate_events -f')        
        
        val2 = self.cmd_line.results.current['cross']
        err2 = self.cmd_line.results.current['error']        
        
        self.assertTrue(abs(val2 - val1) / (err1 + err2) < 5)
        target = 1310200.0
        self.assertTrue(abs(val2 - target) / (err2) < 5)
        #check precision
        self.assertTrue(err2 / val2 < 0.005)
        self.assertTrue(err1 / val1 < 0.005)
        
    def test_e_p_collision(self):
        """check that e p > e j gives the correct result"""
        

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --save')
        mg_cmd.exec_cmd(' generate e- p  > e- j')
        mg_cmd.exec_cmd('output %s/'% self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=  self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')
        shutil.copy(os.path.join(_file_path, 'input_files', 'run_card_ep.dat'),
                    '%s/Cards/run_card.dat' % self.run_dir) 
        
        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        
        target = 3932.0
        self.assertTrue(abs(val1 - target) / err1 < 1., 'large diference between %s and %s +- %s'%
                        (target, val1, err1))
        
    def test_e_e_collision(self):
        """check that e+ e- > t t~ gives the correct result"""
        

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --save')
        mg_cmd.exec_cmd(' generate e+ e-  > e+ e-')
        mg_cmd.exec_cmd('output %s/' % self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=  self.run_dir)
        self.cmd_line.no_notification()
        shutil.copy(os.path.join(_file_path, 'input_files', 'run_card_ee.dat'),
                    '%s/Cards/run_card.dat' % self.run_dir)
        
        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        
        target = 155.9
        self.assertTrue(abs(val1 - target) / err1 < 2.)
        
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
        self.assertTrue('lhe' in data[0].parton)
        
        if syst:
            # check that the html has the information
            self.assertTrue('syst' in data[0].parton)
            # check that the code was runned correctly
            fsock = open('%s/Events/%s/%s_parton_syscalc.log' % \
                  (self.run_dir, data[0]['run_name'], data[0]['tag']),'r')
            text = fsock.read()
            self.assertTrue(text.count('cross-section') >= 3)
        
                
    def check_pythia_output(self, run_name='run_01', syst=False):
        """ """
        # check that the number of event is fine:
        data = self.load_result(run_name)
        self.assertTrue('lhe' in data[0].pythia)
        self.assertTrue('log' in data[0].pythia)

        if syst:
            # check that the html has the information
            self.assertTrue('rwt' in data[0].pythia)

    def check_matched_plot(self, run_name='run_01', mintime=None, tag='fermi'):
        """ """
        path = '%(path)s/HTML/%(run)s/plots_pythia_%(tag)s/DJR1.ps' % \
                                {'path':self.run_dir,'run': run_name, 'tag': tag}
        self.assertTrue(os.path.exists(path))
        
        if mintime:
            self.assertTrue(os.path.getctime(path) > mintime)
        
        return open(path).read()
#===============================================================================
# TestCmd
#===============================================================================
class TestMEfromfile(unittest.TestCase):
    """test that we can launch everything from a single file"""

    def setUp(self):
        
        self.path = tempfile.mkdtemp(prefix='acc_test_mg5')
        self.run_dir = pjoin(self.path, 'MGPROC') 
    
    def tearDown(self):

        shutil.rmtree(self.path)

    def test_add_time_of_flight(self):
        """checking time of flight is working fine"""

        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull
        if not os.path.exists(pjoin(MG5DIR, 'pythia-pgs')):
            p = subprocess.Popen([pjoin(MG5DIR,'bin','mg5')],
                             stdin=subprocess.PIPE,
                             stdout=stdout,stderr=stderr)
            out = p.communicate('install pythia-pgs')
        misc.compile(cwd=pjoin(MG5DIR,'pythia-pgs'))

        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception, error:
            pass
        
        cmd = """import model sm
                 set automatic_html_opening False --no_save
                 set notification_center False --no_save
                 generate p p > w+ z
                 output %s -f -nojpeg
                 launch -i 
                 set automatic_html_opening False --no_save
                 generate_events
                 parton
                 set nevents 100
                 add_time_of_flight --threshold=4e-14
                 pythia
                 """ %self.run_dir
        open(pjoin(self.path, 'mg5_cmd'),'w').write(cmd)
        
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull
        subprocess.call([pjoin(_file_path, os.path.pardir,'bin','mg5'), 
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

        if not os.path.exists(pjoin(MG5DIR, 'pythia-pgs')):
            p = subprocess.Popen([pjoin(MG5DIR,'bin','mg5')],
                             stdin=subprocess.PIPE,
                             stdout=stdout,stderr=stderr)
            out = p.communicate('install pythia-pgs')
        misc.compile(cwd=pjoin(MG5DIR,'pythia-pgs'))
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
        output %(path)s
        launch
        madspin=ON
        pythia=ON
        %(path)s/../madspin_card.dat
        set nevents 1000
        set pdlabel cteq6l1
        launch -i
        decay_events run_01 
        %(path)s/../madspin_card2.dat
        """ % {'path':self.run_dir})
        command.close()
        
        fsock = open(pjoin(self.path, 'madspin_card.dat'), 'w')
        fsock.write("""decay w+ > j j
        decay w- > e- ve~
        launch
        """)
        fsock.close()
        fsock = open(pjoin(self.path, 'madspin_card2.dat'), 'w')
        fsock.write("""decay w+ > j j
        decay w- > j j
        launch
        """)
        fsock.close()
                
        subprocess.call([pjoin(_file_path, os.path.pardir,'bin','mg5'), 
                         pjoin(self.path, 'cmd')],
                         cwd=pjoin(_file_path, os.path.pardir),
                        stdout=stdout,stderr=stdout)     
        
        #a=rwa_input('freeze')
        self.check_parton_output(cross=150770.0, error=7.4e+02,target_event=1000)
        self.check_parton_output('run_01_decayed_1', cross=66344.2066122, error=1.5e+03,target_event=1000)
        #logger.info('\nMS info: the number of events in the html file is not (always) correct after MS\n')
        self.check_parton_output('run_01_decayed_2', cross=100521.52517, error=8e+02,target_event=1000)
        self.check_pythia_output(run_name='run_01_decayed_1')
        
        #check the first decayed events for energy-momentum conservation.
        
        
        self.assertEqual(cwd, os.getcwd())
        
        
        
    def test_generation_from_file_1(self):
        """ """
        cwd = os.getcwd()
        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception, error:
            pass
        import subprocess
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        if not os.path.exists(pjoin(MG5DIR, 'pythia-pgs')):
            p = subprocess.Popen([pjoin(MG5DIR,'bin','mg5')],
                             stdin=subprocess.PIPE,
                             stdout=stdout,stderr=stderr)
            out = p.communicate('install pythia-pgs')
        misc.compile(cwd=pjoin(MG5DIR,'pythia-pgs'))
        if logging.getLogger('madgraph').level > 20:
            stdout = devnull
        else:
            stdout= None

        fsock = open(pjoin(self.path, 'test_mssm_generation'),'w')
        fsock.write(open(pjoin(_file_path, 'input_files','test_mssm_generation')).read() %
                    {'dir_name': self.run_dir, 'mg5_path':pjoin(_file_path, os.path.pardir)})
        fsock.close()

        subprocess.call([pjoin(_file_path, os.path.pardir,'bin','mg5'), 
                         pjoin(self.path, 'test_mssm_generation')],
                         #cwd=pjoin(self.path),
                        stdout=stdout,stderr=stdout)
        
        self.check_parton_output(cross=4.541638, error=0.035)
        self.check_parton_output('run_02', cross=4.41887317, error=0.035)
        self.check_pythia_output()
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
        
        
        
        

    def load_result(self, run_name):
        
        import madgraph.iolibs.save_load_object as save_load_object
        import madgraph.madevent.gen_crossxhtml as gen_crossxhtml
        
        result = save_load_object.load_from_file(pjoin(self.run_dir,'HTML/results.pkl'))
        return result[run_name]

    def check_parton_output(self, run_name='run_01', target_event=100, cross=0, error=9e99):
        """Check that parton output exists and reach the targert for event"""
                
        # check that the number of event is fine:
        data = self.load_result(run_name)
        if target_event > 0:
            self.assertEqual(int(data[0]['nb_event']), target_event)
        self.assertTrue('lhe' in data[0].parton)
        
        if cross:
            import math
            new_error = math.sqrt(error**2 + float(data[0]['error'])**2)
            self.assertTrue(abs(cross - float(data[0]['cross']))/new_error < 3,
                            'cross is %s and not %s. NB_SIGMA %s' % (float(data[0]['cross']), cross, float(data[0]['cross'])/new_error)
                            )
            self.assertTrue(float(data[0]['error']) < 3 * error)
                            
    def check_pythia_output(self, run_name='run_01'):
        """ """
        # check that the number of event is fine:
        data = self.load_result(run_name)
        self.assertTrue('lhe' in data[0].pythia)
        self.assertTrue('log' in data[0].pythia)
    
    def test_decay_width_nlo_model(self):
        """ """
        
        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception, error:
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
        cmd.run_cmd('output  %s -f' % self.run_dir)
        cmd.run_cmd('launch -f')
        data = self.load_result('run_01')
        self.assertNotEqual(data[0]['cross'], 0)
        
        

#===============================================================================
# TestCmd
#===============================================================================
class TestMEfromPdirectory(unittest.TestCase):
    """test that we can launch everything from the P directory"""

    

    def generate(self, process, model):
        """Create a process"""

        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception, error:
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
        self.assertTrue('lhe' in data[0].parton)
        if cross:
            self.assertTrue(abs(cross - float(data[0]['cross']))/float(data[0]['error']) < 3)


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
            ff.write('display options\n')
            ff.write('display variable allow_notification_center\n')
            ff.write('generate_events -f \n') 
            ff.close()
            if logger.getEffectiveLevel() > 20:
                output = open(os.devnull,'w')
            else:
                output = None
            id = subprocess.call(['./bin/madevent','cmd.cmd'], stdout=output, stderr=output)
            self.assertEqual(id, 0)
            self.check_parton_output(cross=947.9) 
