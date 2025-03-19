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
    
    def tearDown(self):

        if self.path != pjoin(MG5DIR, "tmp_test"):
            shutil.rmtree(self.path)
    
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

        if not os.path.exists(pjoin(MG5DIR, 'MadAnalysis')):
            print("install MadAnalysis")
            p = subprocess.Popen([pjoin(MG5DIR,'bin','mg5_aMC')],
                             stdin=subprocess.PIPE,
                             stdout=stdout,stderr=stderr)
            out = p.communicate('install MadAnalysis4'.encode())
        misc.compile(cwd=pjoin(MG5DIR,'MadAnalysis'))

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
        
        self.assertLess(abs(val2 - val1) / (err1 + err2), 5)
        target = 1310200.0
        self.assertLess(abs(val2 - target) / (err2), 5)
        #check precision
        self.assertLess(err2 / val2, 0.005)
        self.assertLess(err1 / val1, 0.005)
        
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
        
        target = 3932.0
        self.assertLess(
            abs(val1 - target) / (err1+1.7),
            2.,
            'large diference between %s and %s +- %s'%
                        (target, val1, err1)
        )
        

    def test_eva_collision(self):
        """check that e p > e j gives the correct result"""
        

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.run_cmd('set group_subprocesses false')
        mg_cmd.run_cmd('set automatic_html_opening False --save')
        mg_cmd.run_cmd(' generate w+ w-  > t t~')
        mg_cmd.run_cmd('output %s/'% self.run_dir)
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
        
        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        
        target = 0.02174605
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
            mg_cmd.run_cmd('output %s/'% self.run_dir)

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
            mg_cmd.run_cmd('output %s/'% self.run_dir)

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
        mg_cmd.run_cmd('set automatic_html_opening False --save')
        mg_cmd.run_cmd('import model %s/tests/input_files/SMEFTatNLO_running' % madgraph.MG5DIR)
        mg_cmd.run_cmd('generate p p > t t~ NP=2 NP^2==2 QCD=2 QED=0')
        mg_cmd.run_cmd('output %s/'% self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=  self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')
        
        #check validity of the default run_card
        run_card = banner.RunCardLO(pjoin(self.run_dir, 'Cards','run_card.dat'))

        #f = open(pjoin(self.run_dir, 'Cards','run_card.dat'),'r')
        self.assertIn('fixed_extra_scale', run_card.user_set)
        self.assertIn('mue_ref_fixed', run_card.user_set)
        self.assertIn('mue_over_ref', run_card.user_set)

        
        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']

        #target = 166.36114 # value used as reference before changing sde_strategy
        target = 165.7 # computed with sde_strategy #165.8 +- 0.02099 pb
        self.assertTrue(abs(val1 - target) / err1 < 2., 'large diference between %s and %s +- %s'%
                        (target, val1, err1))

        
        # edit run_card -> fix scale
        run_card['fixed_extra_scale'] = True
        run_card['mue_ref_fixed'] = 250

        run_card.write('%s/Cards/run_card.dat' % self.run_dir)
        self.do('generate_events -f')
        val1 = self.cmd_line.results.current['cross']
        err1 = self.cmd_line.results.current['error']
        target = 165.7 
        self.assertTrue(abs(val1 - target) / err1 < 1., 'large diference between %s and %s +- %s'%
                        (target, val1, err1))







    def test_complex_mass_scheme(self):
        """check that auto-width and Madspin works nicely with complex-mass-scheme"""
        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --save')
        mg_cmd.exec_cmd('set complex_mass_scheme', precmd=True)
        mg_cmd.exec_cmd('generate g g  > t t~', precmd=True)
        mg_cmd.exec_cmd('output %s' % self.run_dir, precmd=True)
        
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
        output %s -f
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
        mg_cmd.exec_cmd('output %s/' % self.run_dir)
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
                 output %s -f -nojpeg
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
        output %(path)s
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
        output %(path)s
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
        output %s
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
