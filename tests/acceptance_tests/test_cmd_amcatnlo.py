################################################################################
#
# Copyright (c) 2009 The MadGraph Development team and Contributors
#
# This file is a part of the MadGraph 5 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph license which should accompany this 
# distribution.
#
# For more information, please visit: http://madgraph.phys.ucl.ac.be
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
from cStringIO import StringIO

logger = logging.getLogger('test_cmd')

import tests.unit_tests.iolibs.test_file_writers as test_file_writers

import madgraph.interface.master_interface as MGCmd
import madgraph.interface.amcatnlo_run_interface as NLOCmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.iolibs.files as files
import madgraph.various.misc as misc

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd

pjoin = os.path.join

#===============================================================================
# TestCmd
#===============================================================================
class TestMECmdShell(unittest.TestCase):
    """this treats all the command not related to MG_ME"""
    
    loadtime = time.time()
    
    def generate(self, process, model, multiparticles=[]):
        """Create a process"""

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)
            

        
        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception, error:
            pass

        interface = MGCmd.MasterCmd()
        
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
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        interface.onecmd('output /tmp/MGPROCESS/ -f')
        proc_card = open('/tmp/MGPROCESS/Cards/proc_card_mg5.dat').read()
        self.assertTrue('generate' in proc_card or 'add process' in proc_card)
        
        self.cmd_line = NLOCmd.aMCatNLOCmdShell(me_dir= '/tmp/MGPROCESS')
        self.cmd_line.exec_cmd('set automatic_html_opening False --no_save')

    @staticmethod
    def join_path(*path):
        """join path and treat spaces"""     
        combine = os.path.join(*path)
        return combine.replace(' ','\ ')        
    
    def do(self, line):
        """ exec a line in the cmd under test """        
        self.cmd_line.exec_cmd(line, errorhandling=False,precmd=True)


    def test_check_singletop_fastjet(self):
        cmd = os.getcwd()
        self.generate(['p p > t j [real=QCD]'], 'sm-no_b_mass', multiparticles=['p = p b b~', 'j = j b b~'])

        card = open('/tmp/MGPROCESS/Cards/run_card_default.dat').read()
        self.assertTrue( '10000 = nevents' in card)
        card = card.replace('10000 = nevents', '100 = nevents')
        open('/tmp/MGPROCESS/Cards/run_card_default.dat', 'w').write(card)
        os.system('cp  /tmp/MGPROCESS/Cards/run_card_default.dat /tmp/MGPROCESS/Cards/run_card.dat')

        card = open('/tmp/MGPROCESS/Cards/shower_card_default.dat').read()
        self.assertTrue( 'ANALYSE     =' in card)
        card = card.replace('ANALYSE     =', 'ANALYSE     = mcatnlo_hwanstp.o myfastjetfortran.o mcatnlo_hbook_gfortran8.o')
        self.assertTrue( 'EXTRALIBS   = stdhep Fmcfio' in card)
        card = card.replace('EXTRALIBS   = stdhep Fmcfio', 'EXTRALIBS   = fastjet')
        open('/tmp/MGPROCESS/Cards/shower_card_default.dat', 'w').write(card)
        os.system('cp  /tmp/MGPROCESS/Cards/shower_card_default.dat /tmp/MGPROCESS/Cards/shower_card.dat')

        os.system('rm -rf /tmp/MGPROCESS/RunWeb')
        os.system('rm -rf /tmp/MGPROCESS/Events/run_*')
        self.do('generate_events -f')
        # test the lhe event file and plots exist
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/events.lhe.gz'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_abs.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_abs.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/plot_HERWIG6_1_0.top'))



    def test_check_ppzjj(self):
        """test that p p > z j j is correctly output without raising errors"""
        
        cmd = os.getcwd()
        self.generate(['p p > z p p [real=QCD]'], 'sm', multiparticles=['p = g u'])
        self.assertEqual(cmd, os.getcwd())
        self.do('compile -f')
        self.do('quit')

        pdirs = [dir for dir in \
                open('/tmp/MGPROCESS/SubProcesses/subproc.mg').read().split('\n') if dir]

        for pdir in pdirs:
            exe = os.path.join('/tmp/MGPROCESS/SubProcesses', pdir, 'madevent_mintMC')
            self.assertTrue(os.path.exists(exe))


    def test_split_evt_gen(self):
        """test that the event generation splitting works"""
        cmd = os.getcwd()
        self.generate(['p p > e+ ve [QCD] '], 'sm')
        card = open('/tmp/MGPROCESS/Cards/run_card_default.dat').read()
        self.assertTrue( ' -1 = nevt_job' in card)
        card = card.replace(' -1 = nevt_job', '500 = nevt_job')
        open('/tmp/MGPROCESS/Cards/run_card.dat', 'w').write(card)
        self.cmd_line.exec_cmd('set  cluster_temp_path /tmp/')
        self.do('generate_events -pf')
        # test the lhe event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/events.lhe.gz'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_abs.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_abs.txt'))



    def test_check_ppwy(self):
        """test that the p p > w y (spin 2 graviton) process works with loops. This
        is needed in order to test the correct wavefunction size setting for spin2
        particles"""
        cmd = os.getcwd()
        self.generate(['p p > w+ y [QCD] '], 'tests/loop_smgrav')
        card = open('/tmp/MGPROCESS/Cards/run_card_default.dat').read()
        self.assertTrue( '10000 = nevents' in card)
        card = card.replace('10000 = nevents', '100 = nevents')
        open('/tmp/MGPROCESS/Cards/run_card_default.dat', 'w').write(card)
        os.system('cp  /tmp/MGPROCESS/Cards/run_card_default.dat /tmp/MGPROCESS/Cards/run_card.dat')
        self.do('generate_events -pf')
        # test the lhe event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/events.lhe.gz'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_abs.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_abs.txt'))



    def generate_production(self):
        """production"""
        
        if os.path.exists('/tmp/MGPROCESS/Cards/proc_card_mg5.dat'):
            proc_path = '/tmp/MGPROCESS/Cards/proc_card_mg5.dat'
            if 'p p > e+ ve [QCD]' in open(proc_path).read():
                if files.is_uptodate(proc_path, min_time=self.loadtime):
                    if hasattr(self, 'cmd_line'):
                        self.cmd_line.exec_cmd('quit')
                        
                    self.cmd_line = NLOCmd.aMCatNLOCmdShell(me_dir= '/tmp/MGPROCESS')
                    self.cmd_line.exec_cmd('set automatic_html_opening False --no_save')
                    os.system('rm -rf /tmp/MGPROCESS/RunWeb')
                    os.system('rm -rf /tmp/MGPROCESS/Events/run_01')
                    os.system('rm -rf /tmp/MGPROCESS/Events/run_01_LO')
                    card = open('/tmp/MGPROCESS/Cards/run_card_default.dat').read()
                    self.assertTrue( '10000 = nevents' in card)
                    card = card.replace('10000 = nevents', '100 = nevents')
                    open('/tmp/MGPROCESS/Cards/run_card_default.dat', 'w').write(card)
                    os.system('cp  /tmp/MGPROCESS/Cards/run_card_default.dat /tmp/MGPROCESS/Cards/run_card.dat')
                    os.system('cp  /tmp/MGPROCESS/Cards/shower_card_default.dat /tmp/MGPROCESS/Cards/shower_card.dat')
                    
                    return

        cmd = os.getcwd()
        self.generate(['p p > e+ ve [QCD]'], 'loop_sm')
        self.assertEqual(cmd, os.getcwd())
        self.do('quit')
        card = open('/tmp/MGPROCESS/Cards/run_card_default.dat').read()
        self.assertTrue( '10000 = nevents' in card)
        card = card.replace('10000 = nevents', '100 = nevents')
        open('/tmp/MGPROCESS/Cards/run_card.dat', 'w').write(card)


    def test_ppgogo_amcatnlo(self):
        """tests if the p p > go go (in the mssm) process works"""
        self.generate(['p p > go go [real=QCD]'], 'mssm')

        card = open('/tmp/MGPROCESS/Cards/run_card_default.dat').read()
        self.assertTrue( '10000 = nevents' in card)
        card = card.replace('10000 = nevents', '100 = nevents')
        open('/tmp/MGPROCESS/Cards/run_card.dat', 'w').write(card)

        self.do('launch aMC@NLO -fp')

        # test the lhe event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/events.lhe.gz'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_abs.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_abs.txt'))


    def test_ppgogo_nlo(self):
        """tests if the p p > go go (in the mssm) process works at fixed order"""
        self.generate(['p p > go go [real=QCD]'], 'mssm')

        card = open('/tmp/MGPROCESS/Cards/run_card_default.dat').read()
        self.assertTrue( '10000  = npoints_FO' in card)
        card = card.replace('10000  = npoints_FO', '100  = npoints_FO')
        self.assertTrue( '6000   = npoints_FO' in card)
        card = card.replace('6000   = npoints_FO', '100  = npoints_FO')
        self.assertTrue( '5000   = npoints_FO' in card)
        card = card.replace('5000   = npoints_FO', '100  = npoints_FO')
        open('/tmp/MGPROCESS/Cards/run_card.dat', 'w').write(card)

        self.do('launch NLO -f')
        # test the plot file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/MADatNLO.top'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res.txt'))
        


    def test_calculate_xsect_script(self):
        """test if the calculate_xsect script in the bin directory
        works fine"""
        
        self.generate_production()
        misc.call([pjoin('.','bin','calculate_xsect'), '-f'], cwd='/tmp/MGPROCESS',
                stdout = open(os.devnull, 'w'))

        # test the plot file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/MADatNLO.top'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res.txt'))

        

    def test_generate_events_shower_scripts(self):
        """test if the generate_events and successively the shower script in 
        the bin directory works fine"""
        
        self.generate_production()
        # to check that the cleaning of files work well
        os.system('touch /tmp/MGPROCESS/SubProcesses/P0_udx_epve/GF1')
        self.do('quit')
        misc.call([pjoin('.','bin','generate_events'), '-f'], cwd='/tmp/MGPROCESS',
                stdout = open(os.devnull, 'w'))
        # test the lhe event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/events.lhe.gz'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_abs.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_abs.txt'))
        # test the hep event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/events_HERWIG6_0.hep.gz'))
        misc.call([pjoin('.','bin','shower'), 'run_01', '-f'], cwd='/tmp/MGPROCESS',
                stdout = open(os.devnull, 'w'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/events_HERWIG6_1.hep.gz'))
        # sanity check on the size
        self.assertTrue(os.path.getsize('/tmp/MGPROCESS/Events/run_01/events_HERWIG6_0.hep.gz') > \
                        os.path.getsize('/tmp/MGPROCESS/Events/run_01/events.lhe.gz'))
        self.assertTrue(os.path.getsize('/tmp/MGPROCESS/Events/run_01/events_HERWIG6_1.hep.gz') > \
                        os.path.getsize('/tmp/MGPROCESS/Events/run_01/events.lhe.gz'))


    def test_generate_events_lo_hwpp_set(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        cmd = """generate_events LO -p
                 set parton_shower herwigpp
                 set nevents 100
                 """
        open('/tmp/mg5_cmd','w').write(cmd)
        self.cmd_line.import_command_file('/tmp/mg5_cmd')
        #self.do('import command /tmp/mg5_cmd')
        #self.do('generate_events LO -f')        
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/events.lhe.gz'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_0_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_0_abs.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_1_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_1_abs.txt'))
    

    def test_generate_events_lo_hw6_set(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        cmd = """generate_events LO
                 set parton_shower herwig6
                 set nevents 100
                 """
        open('/tmp/mg5_cmd','w').write(cmd)
        self.cmd_line.import_command_file('/tmp/mg5_cmd')
        #self.do('import command /tmp/mg5_cmd')
        #self.do('generate_events LO -f')        
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/events.lhe.gz'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_0_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_0_abs.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_1_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_1_abs.txt'))

        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/events_HERWIG6_0.hep.gz'))
        # sanity check on the size
        self.assertTrue(os.path.getsize('/tmp/MGPROCESS/Events/run_01_LO/events_HERWIG6_0.hep.gz') > \
                        os.path.getsize('/tmp/MGPROCESS/Events/run_01_LO/events.lhe.gz'))



    def test_generate_events_lo_hw6_stdhep(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        cmd = """generate_events LO
                 set nevents 100
                 """
        open('/tmp/mg5_cmd','w').write(cmd)
        self.cmd_line.import_command_file('/tmp/mg5_cmd')
        #self.do('import command /tmp/mg5_cmd')
        #self.do('generate_events LO -f')        
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/events.lhe.gz'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_0_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_0_abs.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_1_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_1_abs.txt'))
        # test the hep event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/events_HERWIG6_0.hep.gz'))
        # sanity check on the size
        self.assertTrue(os.path.getsize('/tmp/MGPROCESS/Events/run_01_LO/events_HERWIG6_0.hep.gz') > \
                        os.path.getsize('/tmp/MGPROCESS/Events/run_01_LO/events.lhe.gz'))
        


    def test_generate_events_lo_py6_stdhep(self):
        """test the param_card created is correct"""
        
        self.generate_production()

        #change to py6
        card = open('/tmp/MGPROCESS/Cards/run_card.dat').read()
        open('/tmp/MGPROCESS/Cards/run_card.dat', 'w').write(card.replace('HERWIG6', 'PYTHIA6Q'))       
        self.do('generate_events LO -f')        
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/events.lhe.gz'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_0_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_0_abs.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_1_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res_1_abs.txt'))
        # test the hep event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/events_PYTHIA6Q_0.hep.gz'))
        # sanity check on the size
        self.assertTrue(os.path.getsize('/tmp/MGPROCESS/Events/run_01_LO/events_PYTHIA6Q_0.hep.gz') > \
                        os.path.getsize('/tmp/MGPROCESS/Events/run_01_LO/events.lhe.gz'))


    def test_generate_events_nlo_py6pt_stdhep(self):
        """check that py6pt event generation works in this case"""
        
        self.generate_production()

        #change to py6
        card = open('/tmp/MGPROCESS/Cards/run_card.dat').read()
        open('/tmp/MGPROCESS/Cards/run_card.dat', 'w').write(card.replace('HERWIG6', 'PYTHIA6PT'))       
        self.do('generate_events -f')        
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/events.lhe.gz'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_abs.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_abs.txt'))
        # test the hep event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/events_PYTHIA6PT_0.hep.gz'))
        # sanity check on the size
        self.assertTrue(os.path.getsize('/tmp/MGPROCESS/Events/run_01/events_PYTHIA6PT_0.hep.gz') > \
                        os.path.getsize('/tmp/MGPROCESS/Events/run_01/events.lhe.gz'))


    def test_check_generate_eventsnlo_py6pt_fsr(self):
        """check that py6pt event generation stops in this case (because of fsr)"""
        
        cmd = os.getcwd()
        self.generate(['e+ e- > t t~ [real=QCD]'], 'sm')
        #change to py6
        card = open('/tmp/MGPROCESS/Cards/run_card.dat').read()
        open('/tmp/MGPROCESS/Cards/run_card.dat', 'w').write(card.replace('HERWIG6', 'PYTHIA6PT'))       
        #self.do('generate_events -f')        
        self.assertRaises(NLOCmd.aMCatNLOError, self.do, 'generate_events -f')

        
    def test_generate_events_nlo_hw6_stdhep(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        self.do('generate_events NLO -f')
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/events.lhe.gz'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_abs.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_abs.txt'))
        # test the hep event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/events_HERWIG6_0.hep.gz'))


    def test_generate_events_nlo_hw6_split(self):
        """test the param_card created is correct"""
        
        cmd = os.getcwd()
        self.generate(['p p > e+ ve [QCD]'], 'loop_sm')
        self.assertEqual(cmd, os.getcwd())
        #change splitevent generation
        card = open('/tmp/MGPROCESS/Cards/run_card.dat').read()
        open('/tmp/MGPROCESS/Cards/run_card.dat', 'w').write(card.replace(' -1 = nevt_job', ' 100 = nevt_job'))
        self.do('generate_events NLO -fp')        
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/events.lhe.gz'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_abs.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_abs.txt'))
        

    def test_generate_events_nlo_py6_stdhep(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        #change to py6
        card = open('/tmp/MGPROCESS/Cards/run_card.dat').read()
        open('/tmp/MGPROCESS/Cards/run_card.dat', 'w').write(card.replace('HERWIG6', 'PYTHIA6Q'))
        
        self.do('generate_events NLO -f')        
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/events.lhe.gz'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_0_abs.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_tot.txt'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res_1_abs.txt'))
        # test the hep event file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/events_PYTHIA6Q_0.hep.gz'))
        
        

    def test_calculate_xsect_nlo(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        
        self.do('calculate_xsect NLO -f')        
        
        # test the plot file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/MADatNLO.top'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/res.txt'))


    def test_calculate_xsect_lo(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        
        self.do('calculate_xsect  LO -f')        
        
        # test the plot file exists
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/MADatNLO.top'))
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01_LO/res.txt'))
    
    def test_amcatnlo_from_file(self):
        """ """
        
        cwd = os.getcwd()
        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception, error:
            pass
        import subprocess
        
        stdout = open('/tmp/test.log','w')
        if logging.getLogger('madgraph').level <= 20:
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stderr=devnull

    
            
        subprocess.call([pjoin(_file_path, os.path.pardir,'bin','mg5'), 
                         pjoin(_file_path, 'input_files','test_amcatnlo')],
                         cwd=pjoin(MG5DIR),
                        stdout=stdout,stderr=stderr)
        stdout.close()
        text = open('/tmp/test.log','r').read()
        data = text.split('\n')
        for i,line in enumerate(data):
            if 'Summary:' in line:
                break
        #      Run at p-p collider (4000 + 4000 GeV)
        self.assertTrue('Run at p-p collider (4000 + 4000 GeV)' in data[i+2])
        #      Total cross-section: 1.249e+03 +- 3.2e+00 pb        
        cross_section = data[i+3]
        cross_section = float(cross_section.split(':')[1].split('+-')[0])
        # warning, delta may not be compatible with python 2.6 
        try:
            self.assertAlmostEqual(4232.0, cross_section,delta=50)
        except TypeError:
            self.assertTrue(cross_section < 4282. and cross_section > 4182.)

        #      Number of events generated: 10000        
        self.assertTrue('Number of events generated: 100' in data[i+4])
        

    def load_result(self, run_name):
        
        import madgraph.iolibs.save_load_object as save_load_object
        import madgraph.various.gen_crossxhtml as gen_crossxhtml
        
        result = save_load_object.load_from_file('/tmp/MGPROCESS/HTML/results.pkl')
        return result[run_name]
