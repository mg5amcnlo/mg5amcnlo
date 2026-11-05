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
import subprocess
import unittest
import os
import re
import shutil
import sys
import logging
import tempfile
import time
import math
from io import StringIO
from madgraph.various import lhe_parser

logger = logging.getLogger('test_cmd')

import tests.unit_tests.iolibs.test_file_writers as test_file_writers
from tests.parallel_tests.test_aloha import set_global
import tests.IOTests as IOTests

import madgraph
import madgraph.interface.master_interface as MGCmd
import madgraph.interface.amcatnlo_run_interface as NLOCmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.iolibs.files as files
import madgraph.various.misc as misc
import madgraph.various.banner as banner


_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd
from tests.acceptance_tests.test_cmd_madevent import check_html_page

pjoin = os.path.join

#===============================================================================
# TestCmd
#===============================================================================
class MECmdShell(IOTests.IOTestManager):
    """this treats all the command not related to MG_ME"""
    
    loadtime = time.time()
    
    def setUp(self):
        self.debugging = unittest.debug
    
        if not self.debugging:
            self.tmpdir = tempfile.mkdtemp(prefix='amc')
            #if os.path.exists(self.tmpdir):
            #    shutil.rmtree(self.tmpdir)
            #os.mkdir(self.tmpdir)
            self.path = pjoin(self.tmpdir,'MGProcess')
        else:
            if os.path.exists(pjoin(MG5DIR, 'TEST_AMC')):
                shutil.rmtree(pjoin(MG5DIR, 'TEST_AMC'))
            os.mkdir(pjoin(MG5DIR, 'TEST_AMC'))
            self.tmpdir = pjoin(MG5DIR, 'TEST_AMC')
            self.path = pjoin(self.tmpdir,'MGProcess')
        
    def tearDown(self):
        if not self.debugging:
            shutil.rmtree(self.tmpdir)
    
    
    def generate(self, process, model, multiparticles=[]):
        """Create a process"""

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)
            

        try:
            shutil.rmtree(self.path)
        except Exception as error:
            pass

        interface = MGCmd.MasterCmd()
        interface.no_notification()
        
        run_cmd('import model %s' % model)
        for multi in multiparticles:
            run_cmd('define %s' % multi)
        if isinstance(process, str):
            run_cmd('generate %s' % process)
        else:
            for p in process:
                run_cmd('add process %s' % p)
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        #else:
        #    devnull =open(os.devnull,'w')
        #    stdout=devnull
        #    stderr=devnull

        interface.onecmd('output %s -f' % self.path)
        proc_card = open('%s/Cards/proc_card_mg5.dat' % self.path).read()
        self.assertTrue('generate' in proc_card or 'add process' in proc_card)
        run_cmd('set automatic_html_opening False --no_save')
        self.cmd_line = NLOCmd.aMCatNLOCmdShell(me_dir= '%s' % self.path)
        self.cmd_line.no_notification()
        self.cmd_line.run_cmd('set automatic_html_opening False --no_save')
        self.assertFalse(self.cmd_line.options['automatic_html_opening'])

    @staticmethod
    def join_path(*path):
        """join path and treat spaces"""     
        combine = os.path.join(*path)
        return combine.replace(' ',r'\ ')        
    
    def set_parton_shower(self, shower):
        """Pin parton_shower in the run card.

        Rewriting a literal 'HERWIG6' only worked while that was the default;
        the moment it changed, the rewrite became a no-op and the test quietly
        showered with something other than the one it is named for.
        """

        path = '%s/Cards/run_card.dat' % self.path
        with open(path) as handle:
            card = handle.read()
        card, count = re.subn(r'^(\s*)\S+(\s*=\s*parton_shower)',
                              r'\g<1>%s\g<2>' % shower, card, flags=re.M)
        self.assertEqual(count, 1,
                         'could not set parton_shower in %s' % path)
        with open(path, 'w') as handle:
            handle.write(card)

    def do(self, line):
        """ exec a line in the cmd under test """        
        self.cmd_line.exec_cmd(line, errorhandling=False,precmd=True)


    @set_global()
    def test_check_singletop_fastjet(self):
        cmd = os.getcwd()
        self.generate(['p p > t j QED^2=4 QCD^2=0 [real=QCD]'], 'sm-no_b_mass', multiparticles=['p = p b b~', 'j = j b b~'])

        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertIn('10000 = nevents', card)
        card = card.replace('10000 = nevents', '100 = nevents')
        open('%s/Cards/run_card_default.dat' % self.path, 'w').write(card)
        os.system('cp  %s/Cards/run_card_default.dat %s/Cards/run_card.dat' % (self.path, self.path))

        card = open('%s/Cards/param_card_default.dat' % self.path).read()
        self.assertTrue( 'DECAY   6 1.491500e+00 # WT' in card)
        card = card.replace('DECAY   6 1.491500e+00 # WT', 'DECAY   6 0.0e+00 # WT')
        open('%s/Cards/param_card_default.dat' % self.path, 'w').write(card)
        os.system('cp  %s/Cards/param_card_default.dat %s/Cards/param_card.dat' % (self.path, self.path))

        card = open('%s/Cards/shower_card_default.dat' % self.path).read()
        self.assertIn('ANALYSE      =', card)
        card = card.replace('ANALYSE      =', 'ANALYSE     = mcatnlo_hwan_pp_tj.o myfastjetfortran.o mcatnlo_hbook_gfortran8.o')
        self.assertIn('EXTRALIBS    = stdhep Fmcfio', card)
        card = card.replace('EXTRALIBS    = stdhep Fmcfio', 'EXTRALIBS   = fastjet')
        open('%s/Cards/shower_card_default.dat' % self.path, 'w').write(card)
        os.system('cp  %s/Cards/shower_card_default.dat %s/Cards/shower_card.dat'% (self.path, self.path))

        self.set_parton_shower('HERWIG6')

        os.system('rm -rf %s/RunWeb' % self.path)
        os.system('rm -rf %s/Events/run_*' % self.path)
        self.do('generate_events -f')
        # test the lhe event file and plots exist
        self.assertTrue(os.path.exists('%s/Events/run_01/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_2.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/run_01_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/plot_HERWIG6_1_0.top' % self.path))

    #@IOTests.createIOTest()
    def test_check_html_long_process_strings(self):
        """ target: info.html
        """
        #check that the info.html file correctly lists all the subprocesses,
        #even when the process string has to be split on more lines (for length 
        #reasons)
        
        cmd = os.getcwd()
        self.generate(['p p > h w+ > ta+ ta- e+ ve QED^2=8 QCD^2=0 [QCD]'], 'sm')
        self.assertEqual(cmd, os.getcwd())

        #info_html_target = open(os.path.join(cmd, 'tests', 'input_files',
        #       'info_pp_to_hw_to_lvtata_nloqcd.html')).read()
        info_html_this = open(os.path.join(self.path, 'HTML', 'info.html')).read()
        #self.assertEqual(info_html_target, info_html_this)
        #open(pjoin(self.IOpath, "info.html"),"w").write(info_html_this)
        self.assertEqual(info_html_this.count('<TD> born </TD>'), 2)
        self.assertEqual(info_html_this.count('<TD> virt </TD>'), 2)
        self.assertEqual(info_html_this.count('<TD> real </TD>'), 6)
        self.assertEqual(info_html_this.count('<TR class=first>'), 2)
        self.assertEqual(info_html_this.count('<TR class=second>'), 8)


    def test_raise_invalid_path_hw7(self):
        """test that an exception is raised when trying to shower with hw7 without
        having set the corresponding paths"""
        cmd = os.getcwd()
        self.generate(['p p > e+ ve QED^2=4 QCD^2=0 [QCD] '], 'sm')
        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertIn('PYTHIA8   = parton_shower', card)
        card = card.replace('PYTHIA8   = parton_shower', 'HERWIG7   = parton_shower')
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card)
        self.cmd_line.exec_cmd('set  cluster_temp_path /tmp/ --no_save')
        self.do('generate_events -pf')
        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/run_01_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_2.html' % self.path))

        #no shower the file
        self.assertRaises(NLOCmd.aMCatNLOError, self.do, 'shower run_01 -f')


    def test_raise_invalid_path_py8(self):
        """test that an exception is raised when trying to shower with py8 without
        having set the corresponding paths"""
        cmd = os.getcwd()
        self.generate(['p p > e+ ve QED^2=4 QCD^2=0 [QCD] '], 'sm')
        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        # PYTHIA8 is now the default, so this test only has to keep it
        self.assertIn('PYTHIA8   = parton_shower', card)
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card)
        self.cmd_line.exec_cmd('set  cluster_temp_path /tmp/ --no_save')
        self.cmd_line.exec_cmd('set  pythia8_path None')
        self.do('generate_events -pf')
        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/run_01_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_2.html' % self.path))

        #no shower the file
        self.assertRaises(NLOCmd.aMCatNLOError, self.do, 'shower run_01 -f')




    def test_split_evt_gen(self):
        """test that the event generation splitting works"""
        cmd = os.getcwd()
        self.generate(['p p > e+ ve QED^2=4 QCD^2=0 [QCD] '], 'sm')
        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertIn(' -1 = nevt_job', card)
        card = card.replace(' -1 = nevt_job', '500 = nevt_job')
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card)
        self.cmd_line.exec_cmd('set  cluster_temp_path /tmp/ --no_save')
        self.do('generate_events -pf')
        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/run_01_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_2.html' % self.path))


    def test_ttbar_ewsudakov(self):
        self.generate(['p p > t t~ [QCD] '], 'loop_qcd_qed_sm_Gmu_forSudakov')
        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertIn('10000 = nevents', card)
        card = card.replace('10000 = nevents', '1000 = nevents')
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card)
        self.do('generate_events aMC@NLO --parton -f')
        card = open('%s/Cards/reweight_card_default.dat' % self.path).read()
        self.assertIn('#change include_sudakov True', card)
        card = card.replace('#change include_sudakov True', 'change include_sudakov True')
        open('%s/Cards/reweight_card.dat' % self.path, 'w').write(card)
        self.do('reweight run_01 -from_cards')



    def test_gen_evt_onlygen(self):
        """test that the event generation splitting works"""
        cmd = os.getcwd()
        self.generate(['p p > e+ ve [QCD]'], 'sm')
        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertTrue( '10000 = nevents' in card)
        card = card.replace('10000 = nevents', '100 = nevents')
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card)
        self.cmd_line.exec_cmd('set  cluster_temp_path /tmp/ --no_save')
        self.do('generate_events -f')
        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/run_01_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_2.html' % self.path))

        # now re-generate with -o
        self.do('generate_events -of')
        self.assertTrue(os.path.exists('%s/Events/run_02/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_02/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_02/run_02_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_02/alllogs_2.html' % self.path))



    def test_madspin_ON_and_onshell_atNLO(self):

        nevents =20
        text = """
        set crash_on_error True --no_save
        generate p p > t t~ [QCD]
        output %s
        launch
        madspin=ON
        shower=OFF
        set nevents %s
        set mt 174
        set wt = 1.5
        decay t > w+ b
        decay t~ > w- b~
        launch -i
        decay_events run_01
        add madspin --replace_line="set spinmode.*" --after_line=banner set spinmode=onshell 
        """ % (self.path,nevents)
        
        interface = MGCmd.MasterCmd()
        interface.no_notification()
        
        open(pjoin(self.tmpdir,'cmd'),'w').write(text)
        
        interface.exec_cmd('import command %s' % pjoin(self.tmpdir, 'cmd'))
        
        # perform some basic check
        orig_lhe = pjoin(self.path,'Events','run_01','events.lhe.gz')
        lhe_on = pjoin(self.path,'Events','run_01_decayed_1','events.lhe.gz')
        lhe_onshell= pjoin(self.path,'Events','run_01_decayed_2','events.lhe.gz')
        
        self.assertTrue(lhe_on)
        mt =174
        wt = 1.5
        
        # check that original event has top onshell
        nb_event = 0
        for event in lhe_parser.EventFile(orig_lhe):
            nb_event +=1
            nb_final = 0
            m_inv_t = 0
            m_inv_tbar = 0
            for p in event:
                if p.status == -1:
                    continue
                elif p.status == 1:
                    nb_final += 1
                    if p.pdg == 6:
                        #if m_inv_t != 0:
                        #    self.assertTrue(False, 'two top decaying')
                        m_inv_t = p.mass
                        self.assertTrue( mt - 0.1*wt < m_inv_t < mt + 0.1*wt)
                    elif p.pdg == -6:
                        #if m_inv_t != 0:
                        #    self.assertTrue(False, 'two antitop decaying')
                        m_inv_tbar = p.mass
                        self.assertTrue( mt - 0.1*wt < m_inv_tbar < mt + 0.1*wt)
                    #else:
                    #    self.assertTrue(False, 'not top-antitop decaying')
            self.assertIn(nb_final, [2,3])
        self.assertEqual(nb_event, nevents)        
        
        
        # check that each events has the decay ON mode
        nb_event = 0
        for event in lhe_parser.EventFile(lhe_on):
            nb_event +=1
            nb_final = 0
            m_inv_t = 0
            m_inv_tbar = 0
            for p in event:
                if p.status == -1:
                    continue
                elif p.status == 1:
                    nb_final += 1
                else:
                    if p.pdg == 6:
                        #if m_inv_t != 0:
                        #    self.assertTrue(False, 'two top decaying')
                        m_inv_t = p.mass
                        self.assertTrue( mt - 15*wt < m_inv_t < mt + 15*wt)
                    elif p.pdg == -6:
                        #if m_inv_t != 0:
                        #    self.assertTrue(False, 'two antitop decaying')
                        m_inv_tbar = p.mass
                        self.assertTrue( mt - 15*wt < m_inv_tbar < mt + 15*wt)
                    else:
                        self.assertTrue(False, 'not top-antitop decaying')
            self.assertIn(nb_final, [4,5])
        self.assertEqual(nb_event, nevents)

        # check that each events has the decay ON mode
        nb_event = 0
        for event in lhe_parser.EventFile(lhe_onshell):
            nb_event +=1
            nb_final = 0
            m_inv_t = 0
            m_inv_tbar = 0
            for p in event:
                if p.status == -1:
                    continue
                elif p.status == 1:
                    nb_final += 1
                else:
                    if p.pdg == 6:
                        #if m_inv_t != 0:
                        #    self.assertTrue(False, 'two top decaying')
                        m_inv_t = p.mass
                        self.assertTrue( mt - 1 < m_inv_t < mt + 1)
                    elif p.pdg == -6:
                        #if m_inv_t != 0:
                        #    self.assertTrue(False, 'two antitop decaying')
                        m_inv_tbar = p.mass
                        self.assertTrue( mt - 1 < m_inv_tbar < mt + 1)
                    else:
                        self.assertTrue(False, 'not top-antitop decaying')
            self.assertIn(nb_final, [4,5])
        self.assertEqual(nb_event, nevents)        

        
    def test_madspin_LOonly(self):
        
        text = """
        set crash_on_error True --no_save
        generate p p > w+ [LOonly]
        output %s
        launch
        madspin=ON
        set nevents 10
        decay w+ > e+ ve
        launch -i
        decay_events run_01
        add madspin --replace_line="set spinmode.*" --after_line="banner" set spinmode=onshell 
        """ % self.path
        
        interface = MGCmd.MasterCmd()
        interface.no_notification()
        
        open(pjoin(self.tmpdir,'cmd'),'w').write(text)
        
        interface.exec_cmd('import command %s' % pjoin(self.tmpdir, 'cmd'))
        
     # perform some basic check
        lhe_on = pjoin(self.path,'Events','run_01_decayed_1','events.lhe.gz')
        lhe_onshell= pjoin(self.path,'Events','run_01_decayed_2','events.lhe.gz')
        
        self.assertTrue(lhe_on)        
        # check that each events has the decay ON mode
        nb_event = 0
        for event in lhe_parser.EventFile(lhe_on):
            nb_event +=1
            nb_final = 0
            m_inv_w = 0
            for p in event:
                if p.status == -1:
                    continue
                elif p.status == 1:
                    nb_final += 1
                else:
                    if p.pdg == 24:
                        if m_inv_w != 0:
                            self.assertTrue(False, 'two W')
                        m_inv_w = p.mass
                        self.assertTrue( 30 < m_inv_w < 150)
                    else:
                        misc.sprint(p.pdg, p.status)
                        self.assertTrue(False, 'not W decaying')
            self.assertIn(nb_final, [2])
        self.assertEqual(nb_event, 10)      
        
        self.assertTrue(lhe_onshell)        
        # check that each events has the decay ON mode
        nb_event = 0
        for event in lhe_parser.EventFile(lhe_onshell):
            nb_event +=1
            nb_final = 0
            m_inv_w = 0
            for p in event:
                if p.status == -1:
                    continue
                elif p.status == 1:
                    nb_final += 1
                else:
                    if p.pdg == 24:
                        if m_inv_w != 0:
                            self.assertTrue(False, 'two W')
                        m_inv_w = p.mass
                        self.assertTrue( 80 < m_inv_w < 81)
                    else:
                        self.assertTrue(False, 'not W decaying')
            self.assertIn(nb_final, [2])
        self.assertEqual(nb_event, 10)  
          

    def generate_production(self):
        """production"""
        
        if os.path.exists('%s/Cards/proc_card_mg5.dat' % self.path):
            proc_path = '%s/Cards/proc_card_mg5.dat' % self.path
            if 'p p > e+ ve QED^2=4 QCD^2=0 [QCD]' in open(proc_path).read():
                if files.is_uptodate(proc_path, min_time=self.loadtime):
                    if hasattr(self, 'cmd_line'):
                        self.cmd_line.exec_cmd('quit')
                    os.system('rm -rf %s/RunWeb' % self.path)
                    os.system('rm -rf %s/Events/run_01' % self.path)
                    os.system('rm -rf %s/Events/run_01_LO' % self.path)                        
                    self.cmd_line = NLOCmd.aMCatNLOCmdShell(me_dir= '%s' % self.path)
                    self.cmd_line.run_cmd('set automatic_html_opening False --no_save')

                    card = open('%s/Cards/run_card_default.dat' % self.path).read()
                    self.assertIn('10000 = nevents', card)
                    card = card.replace('10000 = nevents', '100 = nevents')
                    open('%s/Cards/run_card_default.dat' % self.path, 'w').write(card)
                    os.system('cp  %s/Cards/run_card_default.dat %s/Cards/run_card.dat'% (self.path, self.path))
                    os.system('cp  %s/Cards/shower_card_default.dat %s/Cards/shower_card.dat'% (self.path, self.path))
                    
                    return

        cmd = os.getcwd()
        self.generate(['p p > e+ ve QED^2=4 QCD^2=0 [QCD]'], 'loop_sm')
        self.assertEqual(cmd, os.getcwd())
        self.do('quit')
        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertIn('10000 = nevents', card)
        card = card.replace('10000 = nevents', '100 = nevents')
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card)


 


    def test_amcantlo_script(self):
        """test that ./bin/aMCatNLO can be launched with some scripts.
        Check also that two runs run without border effects"""
        self.generate_production()
        script = "set notification_center False --no_save\n"
        script += "set automatic_html_opening False --no_save\n"
        script += "launch -p\n"
        script += "launch -p\n"
        open(pjoin(self.path, 'script.txt'), 'w').write(script)

        misc.call([pjoin('.','bin','aMCatNLO'), 'script.txt'], cwd='%s' % self.path,
                stdout = open(os.devnull, 'w'))

        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/run_01_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_2.html' % self.path))

        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_02/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_02/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_02/run_02_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_02/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_02/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_02/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_02/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_02/alllogs_2.html' % self.path))


    def test_eft_running_nlo(self):
        """aMC@NLO output of a model with running Wilson coefficients (running.py):
        run-card keys, generated code, and the effect of the extra scale on the
        Born interference (fixed extra scale, then dynamical extra scale)"""

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.run_cmd('set automatic_html_opening False --no_save')
        mg_cmd.run_cmd('import model %s/tests/input_files/SMEFTatNLO_running-NLO' % madgraph.MG5DIR)
        mg_cmd.run_cmd('generate p p > t t~ QED=0 NP^2<=2 [QCD]')
        mg_cmd.run_cmd('output %s/' % self.path)

        # the three keys of the extra scale are in the run_card and editable
        run_card = banner.RunCardNLO(pjoin(self.path, 'Cards', 'run_card.dat'))
        for key in ['fixed_extra_scale', 'mue_ref_fixed', 'mue_over_ref']:
            self.assertIn(key, run_card.user_set)
            self.assertNotIn(key, run_card.system_only)
        self.assertTrue(run_card['fixed_extra_scale'])
        self.assertEqual(run_card['mue_over_ref'], 1.)

        # generated code: the running routines must not include the vector.inc
        # of the madevent output, the multiple precision routine follows the
        # extra scale of the double precision one, and muR is passed to the model
        couplings = open(pjoin(self.path, 'Source', 'MODEL', 'couplings.f')).read()
        self.assertNotIn('vector.inc', couplings)
        self.assertIn('C_RUNNING_1', couplings)
        self.assertIn('GOTHER_MP', couplings)
        self.assertIn('CALL MP_C_RUNNING_1(GOTHER_EFF)', couplings)
        setscales = open(pjoin(self.path, 'SubProcesses', 'setscales.f')).read()
        self.assertIn('model_scale=muR', setscales)

        # Born interference of the four-fermion operators only: select the
        # squared orders (NP, QCD, QED) = (2, 2, 0) in FKS_params.dat
        fks = open(pjoin(self.path, 'Cards', 'FKS_params.dat')).read()
        fks = re.sub(r'(?m)^#SelectedCouplingOrders\n0\n-1 -1\n',
                     '#SelectedCouplingOrders\n1\n2 2 0\n', fks)
        self.assertIn('#SelectedCouplingOrders\n1\n2 2 0\n', fks)
        open(pjoin(self.path, 'Cards', 'FKS_params.dat'), 'w').write(fks)

        def set_run_card(**options):
            card = open(pjoin(self.path, 'Cards', 'run_card.dat')).read()
            for key, value in options.items():
                card, nb = re.subn(r'(?mi)^(\s*)\S+(\s*=\s*%s\b)' % key,
                                   lambda m: '%s%s%s' % (m.group(1), value, m.group(2)), card)
                self.assertEqual(nb, 1, key)
            open(pjoin(self.path, 'Cards', 'run_card.dat'), 'w').write(card)

        def run_lo(name):
            out = subprocess.run([pjoin(self.path, 'bin', 'generate_events'), 'LO', '-f',
                                  '--name=%s' % name], cwd=self.path,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True).stdout
            found = re.findall(r'Total cross section:\s+(\S+)\s+\+-\s+(\S+)\s+pb', out)
            self.assertTrue(found, out[-2000:])
            return float(found[-1][0]), float(found[-1][1])

        # the Wilson coefficients of the param_card are given at Q = 172.5 GeV:
        # no running for mue = 172.5, running up to 2 TeV otherwise
        set_run_card(dynamical_scale_choice=3, fixed_extra_scale='True',
                     mue_ref_fixed=172.5, req_acc_FO=0.005)
        norun, norun_err = run_lo('norun')
        set_run_card(mue_ref_fixed=2000.)
        fixed, fixed_err = run_lo('fixed')
        set_run_card(fixed_extra_scale='False', mue_over_ref=1.0)
        dyn, dyn_err = run_lo('dyn')

        for val, err in [(norun, norun_err), (fixed, fixed_err), (dyn, dyn_err)]:
            self.assertTrue(val != 0. and err < 0.02 * abs(val), (val, err))
        # the extra scale matters: running from 172.5 GeV up to 2 TeV changes the interference
        # (the dynamical scale, H_T/2 ~ 0.3 TeV here, runs little and is only required to work)
        self.assertTrue(abs(norun - fixed) > 5 * (norun_err ** 2 + fixed_err ** 2) ** 0.5,
                        (norun, fixed))


    def test_generate_events_lo_hw6_stdhep(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        self.set_parton_shower('HERWIG6')
        cmd = """generate_events aMC@LO
                 set nevents 100
                 """
        open('/tmp/mg5_cmd','w').write(cmd)
        self.cmd_line.import_command_file('/tmp/mg5_cmd')
        #self.do('import command %s/mg5_cmd')
        #self.do('generate_events LO -f')        
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/run_01_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_2.html' % self.path))
        # test the hep event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/events_HERWIG6_0.hep.gz' % self.path))
        # sanity check on the size
        self.assertGreater(
            os.path.getsize('%s/Events/run_01/events_HERWIG6_0.hep.gz' % self.path),
            os.path.getsize('%s/Events/run_01/events.lhe.gz' % self.path)
        )
        


    def test_generate_events_lo_py6_stdhep(self):
        """test the param_card created is correct"""
        
        self.generate_production()

        #change to py6
        self.set_parton_shower('PYTHIA6Q')
        self.do('generate_events aMC@LO -f')        
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/run_01_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_2.html' % self.path))
        # test the hep event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/events_PYTHIA6Q_0.hep.gz' % self.path))
        # sanity check on the size
        self.assertGreater(
            os.path.getsize('%s/Events/run_01/events_PYTHIA6Q_0.hep.gz' % self.path),
            os.path.getsize('%s/Events/run_01/events.lhe.gz' % self.path)
        )



    def test_generate_events_nlo_hw6_split(self):
        """test the param_card created is correct"""
        
        cmd = os.getcwd()
        self.generate(['p p > e+ ve QED^2=4 QCD^2=0 [QCD]'], 'loop_sm')
        self.assertEqual(cmd, os.getcwd())
        #change splitevent generation
        self.set_parton_shower('HERWIG6')
        card = open('%s/Cards/run_card.dat' % self.path).read()
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card.replace(' -1 = nevt_job', ' 1000 = nevt_job'))
        self.do('generate_events aMC@NLO -fp')        
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/run_01_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_2.html' % self.path))

        check_html_page(self, pjoin(self.path, 'crossx.html'))
        check_html_page(self, pjoin(self.path, 'HTML', 'run_01', 'results.html'))
        

    def test_calculate_xsect_nlo(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        
        self.do('calculate_xsect NLO -f')        
        
        # test the plot file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/MADatNLO.HwU' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/run_01_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_1.html' % self.path))

        check_html_page(self, pjoin(self.path, 'crossx.html'))
        check_html_page(self, pjoin(self.path, 'HTML', 'run_01', 'results.html'))


    def test_calculate_xsect_lo(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        
        self.do('calculate_xsect LO -f')        
        
        # test the plot file exists
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/MADatNLO.HwU' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/run_01_LO_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_1.html' % self.path))

    
    @unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
    def test_real_helicity_sum_survives_a_boost(self):
        """Compare a generated real-emission sum with every helicity evaluated
        explicitly after learning the zero mask in the real c.m. frame.

        In u d~ > w+ g g, two longitudinal-W helicities accidentally agree
        in that frame, but differ after the longitudinal boost used by an
        initial-state FKS sector. The branch removes equal-helicity reuse
        during export; this checks that rewrite on the physical process
        from PR #478, together with the generated model-state capture.
        """
        # Keep loop-reduction dependencies local and unbuilt: this driver only
        # needs the tree amplitudes, the model and BornSupport.
        interface = MGCmd.MasterCmd()
        interface.no_notification()
        for command in ('set automatic_html_opening False --no_save',
                        'set output_dependencies internal --no_save',
                        'import model sm',
                        'generate u d~ > w+ g [real=QCD]',
                        'output %s -f -nojpeg' % self.path):
            interface.exec_cmd(command, errorhandling=False, printcmd=False,
                               precmd=True, postcmd=True)
        proc_dir = pjoin(self.path, 'SubProcesses', 'P0_udx_wpg')
        mfile = [f for f in sorted(os.listdir(proc_dir))
                 if re.match(r'matrix_\d+\.f$', f) and
                 'Process: u d~ > w+ g g ' in open(pjoin(proc_dir, f)).read()]
        self.assertEqual(len(mfile), 1, 'no matrix file for u d~ > w+ g g')
        mfile = mfile[0]
        pfx = mfile[len('matrix_'):-len('.f')]
        body = open(pjoin(proc_dir, mfile)).read()
        body = body.split('SUBROUTINE SMATRIX%s_SPLITORDERS' % pfx)[1]
        body = body.split('\n      END\n')[0]

        driver = open(pjoin(_pickle_path, 'fks_real_helicity_frame_driver.f')).read()
        for key, value in (
                ('__NCOMB__', re.search(r'NCOMB=(\d+)', body).group(1)),
                ('__NSQAMPSO__', re.search(r'NSQAMPSO=(\d+)', body).group(1)),
                ('__IDEN__', re.search(r'DATA IDEN/(\d+)/', body).group(1)),
                ('__NHEL__', '\n'.join(l for l in body.split('\n')
                                       if 'DATA (NHEL(I,' in l)),
                ('__PFX__', pfx)):
            driver = driver.replace(key, value)
        open(pjoin(proc_dir, 'real_tie_frame.f'), 'w').write(driver)

        # The generated helicity loop captures the model through BornSupport.
        build = subprocess.run(['make', '../lib/libdhelas.a', '../lib/libmodel.a',
                                'born_support'],
                               cwd=pjoin(self.path, 'Source'),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(build.returncode, 0, build.stdout.decode())
        build = subprocess.run(
            ['gfortran', '-O', '-ffixed-line-length-none', '-I.', '-I../../lib',
             'real_tie_frame.f', mfile, 'born_support.f', '../../Source/timing.f',
             '-o', 'real_tie_frame', '-L../../lib', '-lmc_born_support',
             '-ldhelas', '-lmodel', '-Wl,-rpath,' + pjoin(self.path, 'lib')],
            cwd=proc_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(build.returncode, 0,
                         'could not build the driver:\n%s' % build.stdout.decode())

        # A generic event in its partonic c.m. frame (sqrt(s) = 400 GeV, the
        # g g system of mass 150 GeV), with the W on its mass shell.
        card = open(pjoin(self.path, 'Cards', 'param_card.dat')).read()
        mass_block = re.split(r'(?im)^\s*block\s+mass\b', card)[1]
        mw = float(re.search(r'(?m)^\s*24\s+(\S+)', mass_block).group(1))

        def boost(p, b):
            b2 = sum(x * x for x in b)
            gamma = 1. / math.sqrt(1. - b2)
            bp = sum(b[i] * p[i + 1] for i in range(3))
            f = (gamma - 1.) * bp / b2 + gamma * p[0]
            return [gamma * (p[0] + bp)] + [p[i + 1] + f * b[i] for i in range(3)]

        def direction(theta, phi):
            return [math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi), math.cos(theta)]

        sqrts, mgg = 400., 150.
        e_w = (sqrts**2 + mw**2 - mgg**2) / (2. * sqrts)
        p_w = math.sqrt(e_w**2 - mw**2)
        d_w = direction(0.7, 0.3)
        w = [e_w] + [p_w * x for x in d_w]
        q = [sqrts - e_w] + [-p_w * x for x in d_w]
        d_g = direction(1.9, 2.1)
        to_cm = [x / q[0] for x in q[1:]]
        g1 = boost([mgg / 2.] + [mgg / 2. * x for x in d_g], to_cm)
        g2 = boost([mgg / 2.] + [-mgg / 2. * x for x in d_g], to_cm)
        cm = [[sqrts / 2., 0., 0., sqrts / 2.], [sqrts / 2., 0., 0., -sqrts / 2.],
              w, g1, g2]
        boosted = [boost(p, [0., 0., 0.4]) for p in cm]

        momenta = '\n'.join(' '.join('%.17e' % x for p in event for x in p)
                            for event in (cm, boosted)) + '\n'
        run = subprocess.run([pjoin(proc_dir, 'real_tie_frame')],
                             input=momenta.encode(),
                             cwd=pjoin(self.path, 'Cards'),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = run.stdout.decode()
        self.assertEqual(run.returncode, 0, out)
        result = [l.split() for l in out.split('\n') if l.startswith('RESULT')]
        self.assertEqual(len(result), 1, out)
        smatrix, helsum = float(result[0][1]), float(result[0][2])
        self.assertGreater(helsum, 0., out)
        self.assertLess(abs(smatrix / helsum - 1.), 1e-10,
                        'the real emission after a first point in the c.m. '
                        'frame is %.16e, its helicity sum %.16e'
                        % (smatrix, helsum))


    @set_global()
    def test_polarised_nlo_me_frame(self):
        """Polarised fixed-order NLO in a chosen rest frame.

        p p > z{0} z{0} j [QCD] with me_frame = [3,4], i.e. the matrix
        elements evaluated in the ZZ rest frame. The Born, the real, every FKS
        counterterm and the virtual all have to be boosted to the *same*
        frame, rebuilt from each configuration's own momenta.

        What is actually being tested is check_poles and test_soft_col_limits,
        which the launch runs on its own and which fail the run if they do not
        pass. They are the only checks sensitive to this:

          - the infrared poles are proportional to the Born, so any frame
            mismatch between the Born and the virtual shows up as a per-point
            constant ratio between the MadFKS and the OLP poles;
          - the collinear Q term is the same order as the AP term, so a wrong
            azimuthal phase does not cancel and the soft/collinear ratios
            plateau off 1.

        A cross-section comparison would not catch either: both were wrong at
        some point during development while the total stayed plausible.
        """
        self.generate('p p > z{0} z{0} j [QCD]', 'loop_sm')

        # Ask for the ZZ rest frame. nn23lo1 rather than lhapdf, since the
        # python lhapdf bindings are broken under some interpreters and would
        # fail this test for an unrelated reason; scales fixed so the run is
        # reproducible; req_acc_fo loose because the assertion is on the
        # checks, not on the precision of the cross-section.
        card_path = pjoin(self.path, 'Cards', 'run_card.dat')
        run_card = banner.RunCardNLO(card_path)
        run_card.set('me_frame', [3, 4], user=True)
        run_card.set('pdlabel', 'nn23lo1', user=True)
        run_card.set('fixed_ren_scale', True, user=True)
        run_card.set('fixed_fac_scale', True, user=True)
        run_card.set('mur_ref_fixed', 91.188, user=True)
        run_card.set('muf_ref_fixed', 91.188, user=True)
        run_card.set('ptj', 30.0, user=True)
        run_card.set('etaj', 4.0, user=True)
        run_card.set('req_acc_fo', 0.05, user=True)
        run_card.write(card_path)

        self.do('calculate_xsect NLO -f')

        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))

        # check_poles writes one log per P directory; none may report a
        # miscancellation. This is the assertion that the frame handling is
        # consistent between the Born and the virtual.
        pole_logs = misc.glob(pjoin(self.path, 'SubProcesses', 'P*',
                                    'check_poles.log'))
        self.assertTrue(pole_logs, 'check_poles did not run')
        for log in pole_logs:
            self.assertNotIn('MISCANCELLATION', open(log).read(),
                             'poles do not cancel in %s' % log)

    @set_global()
    def test_polarised_nlo_ps_me_frame(self):
        """Polarised NLO+PS (MC@NLO matching) in a chosen rest frame.

        p p > z{0} j [QCD] with me_frame = [3], generating events rather than
        only a cross-section. On top of what the fixed-order test covers this
        exercises the MC counterterms and the LHE writing, both of which
        evaluate the Born and must do so in the same frame as everything else.

        Two assertions carry the physics:

          - the total cross-section must stay the fixed-order one, because the
            MC counterterm cancels between the S and the H event;
          - the ratio of the *absolute* cross-section to it must not, because
            that ratio measures how well the MC subtraction cancels locally,
            which is exactly what the azimuthal phase of the counterterm
            controls. With the phase left in the partonic c.m. the ratio was
            2.23; with it in the me_frame it is 2.00.

        p p > z j is the discriminating channel: the azimuthal term is
        non-zero only for a gluon-mother ISR configuration, so a process whose
        ISR mothers are all quarks passes with a completely wrong azimuth.
        """
        self.generate('p p > z{0} j [QCD]', 'loop_sm')

        # Ask for the Z rest frame. nn23lo1 rather than lhapdf, scales fixed
        # and the seed pinned so the run is reproducible.
        card_path = pjoin(self.path, 'Cards', 'run_card.dat')
        run_card = banner.RunCardNLO(card_path)
        run_card.set('me_frame', [3], user=True)
        run_card.set('pdlabel', 'nn23lo1', user=True)
        run_card.set('fixed_ren_scale', True, user=True)
        run_card.set('fixed_fac_scale', True, user=True)
        run_card.set('mur_ref_fixed', 91.188, user=True)
        run_card.set('muf_ref_fixed', 91.188, user=True)
        run_card.set('ptj', 30.0, user=True)
        run_card.set('etaj', 4.0, user=True)
        run_card.set('nevents', 1000, user=True)
        run_card.set('req_acc', 0.05, user=True)
        run_card.set('iseed', 33, user=True)
        run_card.set('parton_shower', 'PYTHIA8', user=True)
        run_card.write(card_path)

        # --parton stops before the shower: the shower knows nothing about the
        # frame, everything under test happens before it.
        self.do('generate_events aMC@NLO --parton -f')

        # The frame reached the matrix elements: bit 3 set, nothing else.
        run_inc = open(pjoin(self.path, 'Source', 'run_card.inc')).read()
        self.assertIn('FRAME_ID = 8', run_inc)

        self.assertTrue(os.path.exists('%s/Events/run_01/events.lhe.gz' % self.path))

        # check_poles and test_ME/test_MC run as part of the launch and would
        # abort it; assert on the logs anyway so a silent pass is visible.
        pole_logs = misc.glob(pjoin(self.path, 'SubProcesses', 'P*',
                                    'check_poles.log'))
        self.assertTrue(pole_logs, 'check_poles did not run')
        for log in pole_logs:
            self.assertNotIn('MISCANCELLATION', open(log).read(),
                             'poles do not cancel in %s' % log)

        # res_1.txt ends with the absolute cross-section and the cross-section.
        res = open(pjoin(self.path, 'Events', 'run_01', 'res_1.txt')).read()
        totals = re.findall(r'([-\d.eE+]+)\s+\+-', res.split('Total ABS')[-1])
        self.assertEqual(len(totals), 2, 'cannot read the totals from res_1.txt')
        xsec_abs, xsec = float(totals[0]), float(totals[1])
        # 2.19e3 while the Z rest frame was reached by a single boost from
        # the FKS frame, which depended on the FKS sector; reached along the
        # beam first (Collins-Soper, get_me_frame_boosts) it is 2.50e3
        self.assertAlmostEqual(xsec, 2.50e3, delta=1.0e2)
        self.assertLess(xsec_abs / xsec, 2.15,
                        'the MC counterterm does not cancel locally enough; '
                        'its azimuthal phase is probably not in the me_frame')

    def test_amcatnlo_from_file(self):
        """ """
        
        cwd = os.getcwd()
        try:
            os.remove('%s/test.log' % self.tmpdir)
        except Exception as error:
            pass
        import subprocess
        
        stdout = open('%s/test.log' % self.tmpdir,'w')
        if logging.getLogger('madgraph').level <= 20:
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stderr=devnull

            
        subprocess.call([sys.executable, pjoin(_file_path, os.path.pardir,'bin','madgraph'), 
                         pjoin(_file_path, 'input_files','test_amcatnlo')],
                         cwd=self.tmpdir,
                        stdout=stdout,stderr=stderr)

        stdout.close()
        text = open('%s/test.log' % self.tmpdir,'r').read()
        if logging.getLogger('madgraph').level <= 20:
            print(text)
        data = text.split('\n')
        for i,line in enumerate(data):
            if 'Summary:' in line:
                break
        #      Run at p-p collider (6500.0 + 6500.0 GeV)
        self.assertIn('Run at p-p collider (6500.0 + 6500.0 GeV)', data[i+2])
        #      Total cross section: 6.936e+03 +- ... pb
        cross_section = data[i+4]
        cross_section = float(cross_section.split(':')[1].split('+-')[0])
        # previously PDF was nn23nlo (lhaid 244600) with this reference value 6675.0
        # loop_sm gives the b a non-zero mass, so the NLO default here is the
        # 4-flavour set NNPDF40_nlo_as_01180_nf_4 (lhaid 334700), matching the
        # b-less proton MG5 already uses for this model.
        try:
            self.assertAlmostEqual(6936.0, cross_section,delta=50)
        except TypeError:
            self.assertTrue(abs(cross_section - 6936.0) <= 50)

        #      Number of events generated: 100
        self.assertIn('Number of events generated: 100', data[i+3])


    def load_result(self, run_name='run_01'):

        import madgraph.iolibs.save_load_object as save_load_object
        import madgraph.madevent.gen_crossxhtml as gen_crossxhtml

        result = save_load_object.load_from_file('%s/HTML/results.pkl' % self.path)
        return result[run_name]


    def test_generate_taggedph_nloew(self):
        """test the param_card created is correct"""
        

        text = """
        import model loop_qcd_qed_sm_Gmu-a0
        generate u u~ > !a! !a! [QED]
        output %s
        launch NLO
        set lepphreco False
        set quarkphreco False
        """ % (self.path)
        
        interface = MGCmd.MasterCmd()
        interface.no_notification()
        
        open(pjoin(self.tmpdir,'cmd'),'w').write(text)
        
        
        os.system('rm -rf %s/RunWeb' % self.path)
        os.system('rm -rf %s/Events/run_*' % self.path)

        interface.exec_cmd('import command %s' % pjoin(self.tmpdir, 'cmd'))
        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/run_01_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_1.html' % self.path))

        check_html_page(self, pjoin(self.path, 'crossx.html'))
        check_html_page(self, pjoin(self.path, 'HTML', 'run_01', 'results.html'))


    def test_generate_eeww_nlo_emela_noph(self):
        """ we will test the generation of NLO EW for w+w- production,
        using the gmu ren. scheme and the delta scheme for the factorisation 
        of IR singularities
        The expected generation/compilation/running time is about 5 mins with 8 cores
        """
        
        text = """
        import model loop_qcd_qed_sm_Gmu
        generate e+ e- > w+ w- [QED]
        output %s
        launch NLO
        set ebeam1 250
        set ebeam2 250
        set lpp1 -3
        set lpp2 3
        set fixed_ren_scale True
        set fixed_fac_scale True
        set req_acc_FO 0.0010
        set muf_ref_fixed 500
        set mur_ref_fixed 500
        set pdlabel emela
        set lhaid 137009
        set photons_from_lepton False
        set mw 80.379
        set mz 91.1876
        set wt 0
        set ww 0
        set wz 0
        set wh 0
        """ % (self.path)

        interface = MGCmd.MasterCmd()
        interface.no_notification()

        # skip if eMELA is not known to MadGraph7
        if not interface.options['eMELA']:
            self.skipTest("Skipping test, eMELA not available")

        open(pjoin(self.tmpdir,'cmd'),'w').write(text)
        os.system('rm -rf %s/RunWeb' % self.path)
        os.system('rm -rf %s/Events/run_*' % self.path)
        interface.exec_cmd('import command %s' % pjoin(self.tmpdir, 'cmd'))

        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        # read the cross section
        summary = open('%s/Events/run_01/summary.txt' % self.path).read()
        xsect = summary.split("Total cross section:")[1].split(" +-")[0]
        error = summary.split("Total cross section:")[1].split(" +-")[1].split()[0]

        # check within 3 sigma
        self.assertAlmostEqual(7.513, float(xsect), delta=3*float(error))

        check_html_page(self, pjoin(self.path, 'crossx.html'))
        check_html_page(self, pjoin(self.path, 'HTML', 'run_01', 'results.html'))


    def test_generate_eeww_nlo_emela_wph(self):
        """ we will test the generation of NLO EW for w+w- production,
        using the gmu ren. scheme and the delta scheme for the factorisation 
        of IR singularities.
        Photon in the initial state is included.
        The expected generation/compilation/running time is about 5 mins with 8 cores
        """
        
        text = """
        import model loop_qcd_qed_sm_Gmu
        define ep = e+ a
        define em = e- a
        generate ep em > w+ w- [QED]
        output %s
        launch NLO
        set ebeam1 250
        set ebeam2 250
        set lpp1 -3
        set lpp2 3
        set fixed_ren_scale True
        set fixed_fac_scale True
        set req_acc_FO 0.0010
        set muf_ref_fixed 500
        set mur_ref_fixed 500
        set pdlabel emela
        set lhaid 137009
        set photons_from_lepton True
        set mw 80.379
        set mz 91.1876
        set wt 0
        set ww 0
        set wz 0
        set wh 0
        """ % (self.path)

        interface = MGCmd.MasterCmd()
        interface.no_notification()

        # skip if eMELA is not known to MadGraph7
        if not interface.options['eMELA']:
            self.skipTest("Skipping test, eMELA not available")

        open(pjoin(self.tmpdir,'cmd'),'w').write(text)
        os.system('rm -rf %s/RunWeb' % self.path)
        os.system('rm -rf %s/Events/run_*' % self.path)
        interface.exec_cmd('import command %s' % pjoin(self.tmpdir, 'cmd'))

        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        # read the cross section
        summary = open('%s/Events/run_01/summary.txt' % self.path).read()
        xsect = summary.split("Total cross section:")[1].split(" +-")[0]
        error = summary.split("Total cross section:")[1].split(" +-")[1].split()[0]

        # check within 3 sigma
        self.assertAlmostEqual(7.694, float(xsect), delta=3*float(error))

        check_html_page(self, pjoin(self.path, 'crossx.html'))
        check_html_page(self, pjoin(self.path, 'HTML', 'run_01', 'results.html'))


    def test_generate_eett_nlo_emela_bs(self):
        """ we will test the generation of NLO EW for ttbar production,
        using the alphamz ren. scheme and the delta scheme for the factorisation 
        of IR singularities. (Actually, the amz scheme will be transformed 
        internally to the MSbar one)
        Beamstrahlung is included.
        The expected generation/compilation/running time is about 5 mins with 8 cores
        """
        
        text = """
        import model loop_qcd_qed_sm
        generate e+ e- > t t~ [QED]
        output %s
        launch NLO
        set ebeam1 250
        set ebeam2 250
        set lpp1 -3
        set lpp2 3
        set fixed_ren_scale True
        set fixed_fac_scale True
        set req_acc_FO 0.0010
        set muf_ref_fixed 500
        set mur_ref_fixed 500
        set pdlabel emela
        set lhaid 137010
        set photons_from_lepton False
        set mw 80.379
        set mz 91.1876
        set mt 173.3
        set wt 0
        set ww 0
        set wz 0
        set wh 0
        """ % (self.path)

        interface = MGCmd.MasterCmd()
        interface.no_notification()

        # skip if eMELA is not known to MadGraph7
        if not interface.options['eMELA']:
            self.skipTest("Skipping test, eMELA not available")

        open(pjoin(self.tmpdir,'cmd'),'w').write(text)
        os.system('rm -rf %s/RunWeb' % self.path)
        os.system('rm -rf %s/Events/run_*' % self.path)
        interface.exec_cmd('import command %s' % pjoin(self.tmpdir, 'cmd'))

        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        # read the cross section
        summary = open('%s/Events/run_01/summary.txt' % self.path).read()
        xsect = summary.split("Total cross section:")[1].split(" +-")[0]
        error = summary.split("Total cross section:")[1].split(" +-")[1].split()[0]

        # check within 3 sigma
        self.assertAlmostEqual(5.161e-1, float(xsect), delta=3*float(error))

        check_html_page(self, pjoin(self.path, 'crossx.html'))
        check_html_page(self, pjoin(self.path, 'HTML', 'run_01', 'results.html'))


    def test_generate_eett_lo_emela_clusterpath(self):
        """ we will test the generation of LO EW for ttbar production,
        using the alphamz ren. scheme and the delta scheme for the factorisation 
        of IR singularities. (Actually, the amz scheme will be transformed 
        internally to the MSbar one)
        We will set a cluster_temp_path 
        """

        
        text = """
        import model loop_qcd_qed_sm
        set cluster_temp_path /tmp/ --no_save
        generate e+ e- > t t~ [LOonly=QED]
        output %s
        launch LO --multicore
        set ebeam1 250
        set ebeam2 250
        set lpp1 -3
        set lpp2 3
        set fixed_ren_scale True
        set fixed_fac_scale True
        set req_acc_FO 0.010
        set muf_ref_fixed 500
        set mur_ref_fixed 500
        set pdlabel emela
        set lhaid 137010
        set photons_from_lepton False
        set mw 80.379
        set mz 91.1876
        set mt 173.3
        set wt 0
        set ww 0
        set wz 0
        set wh 0
        """ % (self.path)

        interface = MGCmd.MasterCmd()
        interface.no_notification()

        # skip if eMELA is not known to MadGraph7
        if not interface.options['eMELA']:
            self.skipTest("Skipping test, eMELA not available")

        open(pjoin(self.tmpdir,'cmd'),'w').write(text)
        os.system('rm -rf %s/RunWeb' % self.path)
        os.system('rm -rf %s/Events/run_*' % self.path)
        interface.exec_cmd('import command %s' % pjoin(self.tmpdir, 'cmd'))

        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/summary.txt' % self.path))

        check_html_page(self, pjoin(self.path, 'crossx.html'))
        check_html_page(self, pjoin(self.path, 'HTML', 'run_01_LO', 'results.html'))



    def test_generate_eett_nlo_qcd_noisr(self):
        """ we will test the generation of NLO QCD for ttbar production,
        (without ISR, fixed beam energy
        """
        
        text = """
        import model loop_sm-no_b_mass
        generate e+ e- > t t~ [QCD]
        output %s
        launch NLO
        set ebeam1 500
        set ebeam2 500
        set lpp1 0 
        set lpp2 0
        """ % (self.path)

        interface = MGCmd.MasterCmd()
        interface.no_notification()

        open(pjoin(self.tmpdir,'cmd'),'w').write(text)
        os.system('rm -rf %s/RunWeb' % self.path)
        os.system('rm -rf %s/Events/run_*' % self.path)
        interface.exec_cmd('import command %s' % pjoin(self.tmpdir, 'cmd'))

        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        # read the cross section
        summary = open('%s/Events/run_01/summary.txt' % self.path).read()
        xsect = summary.split("Total cross section:")[1].split(" +-")[0]
        error = summary.split("Total cross section:")[1].split(" +-")[1].split()[0]

        # check within 3 sigma
        self.assertAlmostEqual(1.745e-1, float(xsect), delta=3*float(error))

        check_html_page(self, pjoin(self.path, 'crossx.html'))
        check_html_page(self, pjoin(self.path, 'HTML', 'run_01', 'results.html'))
