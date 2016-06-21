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
import tempfile
import time
import math
from cStringIO import StringIO

logger = logging.getLogger('test_cmd')

import tests.unit_tests.iolibs.test_file_writers as test_file_writers
from tests.parallel_tests.test_aloha import set_global
import tests.IOTests as IOTests

import madgraph.interface.master_interface as MGCmd
import madgraph.interface.amcatnlo_run_interface as NLOCmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.iolibs.files as files
import madgraph.various.misc as misc
import madgraph.various.banner as banner


_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd

pjoin = os.path.join

#===============================================================================
# TestCmd
#===============================================================================
class MECmdShell(IOTests.IOTestManager):
    """this treats all the command not related to MG_ME"""
    
    loadtime = time.time()
    debugging = True
    
    def setUp(self):
        
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
        except Exception, error:
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
        return combine.replace(' ','\ ')        
    
    def do(self, line):
        """ exec a line in the cmd under test """        
        self.cmd_line.exec_cmd(line, errorhandling=False,precmd=True)


    @set_global()
    def test_check_singletop_fastjet(self):
        cmd = os.getcwd()
        self.generate(['p p > t j [real=QCD]'], 'sm-no_b_mass', multiparticles=['p = p b b~', 'j = j b b~'])

        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertTrue( '10000 = nevents' in card)
        card = card.replace('10000 = nevents', '100 = nevents')
        open('%s/Cards/run_card_default.dat' % self.path, 'w').write(card)
        os.system('cp  %s/Cards/run_card_default.dat %s/Cards/run_card.dat' % (self.path, self.path))

        card = open('%s/Cards/shower_card_default.dat' % self.path).read()
        self.assertTrue( 'ANALYSE      =' in card)
        card = card.replace('ANALYSE      =', 'ANALYSE     = mcatnlo_hwan_pp_tj.o myfastjetfortran.o mcatnlo_hbook_gfortran8.o')
        self.assertTrue( 'EXTRALIBS    = stdhep Fmcfio' in card)
        card = card.replace('EXTRALIBS    = stdhep Fmcfio', 'EXTRALIBS   = fastjet')
        open('%s/Cards/shower_card_default.dat' % self.path, 'w').write(card)
        os.system('cp  %s/Cards/shower_card_default.dat %s/Cards/shower_card.dat'% (self.path, self.path))

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

    @IOTests.createIOTest()
    def testIO_check_html_long_process_strings(self):
        """ target: info.html
        """
        #check that the info.html file correctly lists all the subprocesses,
        #even when the process string has to be split on more lines (for length 
        #reasons)
        
        cmd = os.getcwd()
        self.generate(['p p > h w+ > ta+ ta- e+ ve [QCD]'], 'sm')
        self.assertEqual(cmd, os.getcwd())

        #info_html_target = open(os.path.join(cmd, 'tests', 'input_files',
        #       'info_pp_to_hw_to_lvtata_nloqcd.html')).read()
        info_html_this = open(os.path.join(self.path, 'HTML', 'info.html')).read()
        #self.assertEqual(info_html_target, info_html_this)
        open(pjoin(self.IOpath, "info.html"),"w").write(info_html_this)

    def test_check_ppzjj(self):
        """test that p p > z j j is correctly output without raising errors"""
        
        cmd = os.getcwd()
        self.generate(['p p > z p p [real=QCD]'], 'sm', multiparticles=['p = g u'])
        self.assertEqual(cmd, os.getcwd())
        self.do('compile -f')
        self.do('quit')

        pdirs = [dir for dir in \
                open('%s/SubProcesses/subproc.mg' % self.path).read().split('\n') if dir]

        for pdir in pdirs:
            exe = os.path.join('%s/SubProcesses' % self.path, pdir, 'madevent_mintMC')
            self.assertTrue(os.path.exists(exe))


    def test_check_eejjj_lo_lhapdf(self):
        """test that e+ e- > j j j with pdlabel='lhapdf' runs ignoring the lhapdf setting
        """
        
        cmd = os.getcwd()
        self.generate(['e+ e- > p p p [real=QCD]'], 'sm' )
        self.assertEqual(cmd, os.getcwd())

        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        # this check that the value of lpp/beam are change automatically
        self.assertTrue('0   = lpp1' in card)
        self.assertTrue('500.0   = ebeam' in card)
        # pass to the object
        card = banner.RunCardNLO(card)
        card['pdlabel'] = "lhapdf"
        self.assertEqual(card['lpp1'], 0)
        self.assertEqual(card['lpp2'], 0)
        self.assertEqual(card['ebeam1'], 500)
        self.assertEqual(card['ebeam2'], 500)
        card.write('%s/Cards/run_card.dat' % self.path)
        

        self.do('calculate_xsect -f LO')
        self.do('quit')

        self.assertTrue(os.path.exists('%s/Events/run_01_LO/MADatNLO.HwU' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/run_01_LO_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_1.html' % self.path))

        # check the result
        res = open('%s/Events/run_01_LO/res_1.txt' % self.path).read()

        pat = re.compile('''\s*(\d+\.\d+e[+-]\d+) \+\- (\d+\.\d+e[+-]\d+)  \((\d+\.\d+e[+-]\d+)\%\)
        \s*(\-?\d+\.\d+e[+-]\d+) \+\- (\d+\.\d+e[+-]\d+)  \((\-?\d+\.\d+e[+-]\d+)\%\)''')

        match = re.search(pat, res)
        res_dict = {'xseca' : float(match.groups()[0]),
                    'erra' : float(match.groups()[1]),
                    'xsect' : float(match.groups()[3]),
                    'errt' : float(match.groups()[4])}

        self.assertEqual(res_dict['xseca'], res_dict['xsect'])
        self.assertTrue(math.fabs(res_dict['xseca']-3.811e-1) < 0.01)


    def test_raise_invalid_path_hwpp(self):
        """test that an exception is raised when trying to shower with hwpp without
        having set the corresponding pahts"""
        cmd = os.getcwd()
        self.generate(['p p > e+ ve [QCD] '], 'sm')
        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertTrue( 'HERWIG6   = parton_shower' in card)
        card = card.replace('HERWIG6   = parton_shower', 'HERWIGPP   = parton_shower')
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card)
        self.cmd_line.exec_cmd('set  cluster_temp_path /tmp/')
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
        having set the corresponding pahts"""
        cmd = os.getcwd()
        self.generate(['p p > e+ ve [QCD] '], 'sm')
        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertTrue( 'HERWIG6   = parton_shower' in card)
        card = card.replace('HERWIG6   = parton_shower', 'PYTHIA8   = parton_shower')
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card)
        self.cmd_line.exec_cmd('set  cluster_temp_path /tmp/')
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
        self.generate(['p p > e+ ve [QCD] '], 'sm')
        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertTrue( ' -1 = nevt_job' in card)
        card = card.replace(' -1 = nevt_job', '500 = nevt_job')
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card)
        self.cmd_line.exec_cmd('set  cluster_temp_path /tmp/')
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


    def test_split_evt_gen_zeroev(self):
        """test that the event generation splitting works when some channels have zero events"""
        cmd = os.getcwd()
        self.generate(['p p > e+ e- [real=QCD] '], 'sm')
        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertTrue( '-1 = nevt_job' in card)
        self.assertTrue( '10000 = nevents' in card)
        self.assertTrue( '-1.0 = req_acc' in card)
        card = card.replace(' -1 = nevt_job', '1 = nevt_job')
        card = card.replace('10000 = nevents', '6 = nevents')
        card = card.replace(' -1 = req_acc', '0.1 = req_acc')
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card)
        self.cmd_line.exec_cmd('set  cluster_temp_path /tmp/')
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


    def test_check_ppwy(self):
        """test that the p p > w y (spin 2 graviton) process works with loops. This
        is needed in order to test the correct wavefunction size setting for spin2
        particles"""
        cmd = os.getcwd()
        self.generate(['p p > w+ y [QCD] '], 'tests/loop_smgrav')
        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertTrue( '10000 = nevents' in card)
        card = card.replace('10000 = nevents', '100 = nevents')
        open('%s/Cards/run_card_default.dat' % self.path, 'w').write(card)
        os.system('cp  %s/Cards/run_card_default.dat %s/Cards/run_card.dat'% (self.path, self.path))
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


    def generate_production(self):
        """production"""
        
        if os.path.exists('%s/Cards/proc_card_mg5.dat' % self.path):
            proc_path = '%s/Cards/proc_card_mg5.dat' % self.path
            if 'p p > e+ ve [QCD]' in open(proc_path).read():
                if files.is_uptodate(proc_path, min_time=self.loadtime):
                    if hasattr(self, 'cmd_line'):
                        self.cmd_line.exec_cmd('quit')
                    os.system('rm -rf %s/RunWeb' % self.path)
                    os.system('rm -rf %s/Events/run_01' % self.path)
                    os.system('rm -rf %s/Events/run_01_LO' % self.path)                        
                    self.cmd_line = NLOCmd.aMCatNLOCmdShell(me_dir= '%s' % self.path)
                    self.cmd_line.run_cmd('set automatic_html_opening False --no_save')

                    card = open('%s/Cards/run_card_default.dat' % self.path).read()
                    self.assertTrue( '10000 = nevents' in card)
                    card = card.replace('10000 = nevents', '100 = nevents')
                    open('%s/Cards/run_card_default.dat' % self.path, 'w').write(card)
                    os.system('cp  %s/Cards/run_card_default.dat %s/Cards/run_card.dat'% (self.path, self.path))
                    os.system('cp  %s/Cards/shower_card_default.dat %s/Cards/shower_card.dat'% (self.path, self.path))
                    
                    return

        cmd = os.getcwd()
        self.generate(['p p > e+ ve [QCD]'], 'loop_sm')
        self.assertEqual(cmd, os.getcwd())
        self.do('quit')
        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertTrue( '10000 = nevents' in card)
        card = card.replace('10000 = nevents', '100 = nevents')
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card)


    @set_global()
    def test_ppgogo_amcatnlo(self):
        """tests if the p p > go go (in the mssm) process works"""
        self.generate(['p p > go go [real=QCD]'], 'mssm')

        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertTrue( '10000 = nevents' in card)
        card = card.replace('10000 = nevents', '100 = nevents')
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card)

        self.do('launch aMC@NLO -fp')

        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/run_01_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_2.html' % self.path))


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



    def test_launch_amcatnlo_name(self):
        """tests if the p p > e+ ve process works specifying the run name.
        """
        self.generate(['p p > e+ ve [QCD]'], 'sm')

        self.do('launch aMC@NLO -fp -n myrun')

        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/myrun/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/myrun/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/myrun/myrun_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/myrun/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/myrun/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/myrun/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/myrun/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/myrun/alllogs_2.html' % self.path))


    @set_global()
    def test_ppgogo_nlo(self):
        """tests if the p p > go go (in the mssm) process works at fixed order"""
        self.generate(['p p > go go [real=QCD]'], 'mssm')

        card = open('%s/Cards/run_card_default.dat' % self.path).read()
        self.assertTrue( '10000  = npoints_FO' in card)
        card = card.replace('10000  = npoints_FO', '100  = npoints_FO')
        self.assertTrue( '5000   = npoints_FO_grid' in card)
        card = card.replace('5000   = npoints_FO_grid', '100  = npoints_FO_grid')
        self.assertTrue( '0.01   = req_acc_FO' in card)
        card = card.replace('0.01   = req_acc_FO', '-1   = req_acc_FO')
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card)

        self.do('launch NLO -f')
        # test the plot file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/MADatNLO.HwU' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/run_01_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_1.html' % self.path))


    def test_calculate_xsect_script(self):
        """test if the calculate_xsect script in the bin directory
        works fine"""
        
        self.generate_production()
        misc.call([pjoin('.','bin','calculate_xsect'), '-f'], cwd='%s' % self.path,
                stdout = open(os.devnull, 'w'))

        # test the plot file exists
        self.assertTrue(os.path.exists('%s/Events/run_01/MADatNLO.HwU' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/run_01_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/res_1.txt' % self.path))
        

    def test_generate_events_shower_scripts(self):
        """test if the generate_events and successively the shower script in 
        the bin directory works fine.
        Also check the splitting of the shower for bot hep and top output"""

        self.generate_production()
        # to check that the cleaning of files work well
        os.system('touch %s/SubProcesses/P0_udx_epve/GF1' % self.path)
        self.do('quit')
        self.cmd_line.run_cmd('generate_events -f')
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
        misc.call([pjoin('.','bin','shower'), 'run_01', '-f'], cwd='%s' % self.path,
                stdout = open(os.devnull, 'w'))
        self.assertTrue(os.path.exists('%s/Events/run_01/events_HERWIG6_1.hep.gz' % self.path))
        # sanity check on the size
        self.assertTrue(os.path.getsize('%s/Events/run_01/events_HERWIG6_0.hep.gz' % self.path) > \
                        os.path.getsize('%s/Events/run_01/events.lhe.gz' % self.path))
        self.assertTrue(os.path.getsize('%s/Events/run_01/events_HERWIG6_1.hep.gz' % self.path) > \
                        os.path.getsize('%s/Events/run_01/events.lhe.gz' % self.path))

        #splitting of the shower
        # 1) hep output
        shower_card = open('%s/Cards/shower_card.dat' % self.path).read()
        shower_card = shower_card.replace('nsplit_jobs  = 1', 'nsplit_jobs  = 4')
        open('%s/Cards/shower_card.dat' % self.path, 'w').write(shower_card)
        self.cmd_line.run_cmd('shower run_01 -f')
        self.assertTrue(os.path.exists('%s/Events/run_01/events_HERWIG6_2__1.hep.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/events_HERWIG6_2__2.hep.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/events_HERWIG6_2__3.hep.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/events_HERWIG6_2__4.hep.gz' % self.path))

        # 2) top output
        shower_card = shower_card.replace('EXTRALIBS    = stdhep Fmcfio', 'EXTRALIBS    =')  
        shower_card = shower_card.replace('ANALYSE      =', 'ANALYSE      = mcatnlo_hwan_pp_lvl.o mcatnlo_hbook_gfortran8.o')  
        open('%s/Cards/shower_card.dat' % self.path, 'w').write(shower_card)
        self.cmd_line.run_cmd('shower run_01 -f')
        self.assertTrue(os.path.exists('%s/Events/run_01/plot_HERWIG6_1_0.tar.gz' % self.path))
        self.assertFalse(os.path.exists('%s/Events/run_01/plot_HERWIG6_1_0__1.top' % self.path))
        misc.call(['tar', '-xzpvf', '%s/Events/run_01/plot_HERWIG6_1_0.tar.gz' % self.path],
                  cwd='%s/Events/run_01/'% self.path,
                  stdout = open(os.devnull, 'w'))
        self.assertTrue(os.path.exists('%s/Events/run_01/plot_HERWIG6_1_0__1.top' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/plot_HERWIG6_1_0__2.top' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/plot_HERWIG6_1_0__3.top' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01/plot_HERWIG6_1_0__4.top' % self.path))


    def test_generate_events_name(self):
        """test if the generate_events and successively the shower script in 
        the bin directory works fine"""
        
        self.generate_production()
        # to check that the cleaning of files work well
        os.system('touch %s/SubProcesses/P0_udx_epve/GF1' % self.path)
        self.do('quit')
        misc.call([pjoin('.','bin','generate_events'), '-fp', '-n myrun'], cwd='%s' % self.path,
                stdout = open(os.devnull, 'w'))
        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/myrun/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/myrun/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/myrun/myrun_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/myrun/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/myrun/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/myrun/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/myrun/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/myrun/alllogs_2.html' % self.path))


    def test_generate_events_lo_hwpp_set(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        cmd = """generate_events aMC@LO -p
                 set parton_shower herwigpp
                 set nevents 100
                 """
        open('/tmp/mg5_cmd','w').write(cmd)
        self.assertFalse(self.cmd_line.options['automatic_html_opening'])
        self.cmd_line.import_command_file('/tmp/mg5_cmd')
        #self.do('import command %s/mg5_cmd')
        #self.do('generate_events LO -f')        
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/run_01_LO_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_2.html' % self.path))
    

    def test_generate_events_lo_hw6_set(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        cmd = """generate_events aMC@LO
                 set parton_shower herwig6
                 set nevents 100
                 """
        open('/tmp/mg5_cmd','w').write(cmd)
        self.cmd_line.import_command_file('/tmp/mg5_cmd')
        #self.do('import command %s/mg5_cmd')
        #self.do('generate_events LO -f')        
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/run_01_LO_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_2.html' % self.path))

        self.assertTrue(os.path.exists('%s/Events/run_01_LO/events_HERWIG6_0.hep.gz' % self.path))
        # sanity check on the size
        self.assertTrue(os.path.getsize('%s/Events/run_01_LO/events_HERWIG6_0.hep.gz' % self.path) > \
                        os.path.getsize('%s/Events/run_01_LO/events.lhe.gz' % self.path))



    def test_generate_events_lo_hw6_stdhep(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        cmd = """generate_events aMC@LO
                 set nevents 100
                 """
        open('/tmp/mg5_cmd','w').write(cmd)
        self.cmd_line.import_command_file('/tmp/mg5_cmd')
        #self.do('import command %s/mg5_cmd')
        #self.do('generate_events LO -f')        
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/run_01_LO_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_2.html' % self.path))
        # test the hep event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/events_HERWIG6_0.hep.gz' % self.path))
        # sanity check on the size
        self.assertTrue(os.path.getsize('%s/Events/run_01_LO/events_HERWIG6_0.hep.gz' % self.path) > \
                        os.path.getsize('%s/Events/run_01_LO/events.lhe.gz' % self.path))
        


    def test_generate_events_lo_py6_stdhep(self):
        """test the param_card created is correct"""
        
        self.generate_production()

        #change to py6
        card = open('%s/Cards/run_card.dat' % self.path).read()
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card.replace('HERWIG6', 'PYTHIA6Q'))       
        self.do('generate_events aMC@LO -f')        
        
        # test the lhe event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/events.lhe.gz' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/summary.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/run_01_LO_tag_1_banner.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/res_0.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/res_1.txt' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_0.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_1.html' % self.path))
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/alllogs_2.html' % self.path))
        # test the hep event file exists
        self.assertTrue(os.path.exists('%s/Events/run_01_LO/events_PYTHIA6Q_0.hep.gz' % self.path))
        # sanity check on the size
        self.assertTrue(os.path.getsize('%s/Events/run_01_LO/events_PYTHIA6Q_0.hep.gz' % self.path) > \
                        os.path.getsize('%s/Events/run_01_LO/events.lhe.gz' % self.path))


    def test_generate_events_nlo_py6pt_stdhep(self):
        """check that py6pt event generation works in this case"""
        
        self.generate_production()

        #change to py6
        card = open('%s/Cards/run_card.dat' % self.path).read()
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card.replace('HERWIG6', 'PYTHIA6PT'))       
        self.do('generate_events -f')        
        
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
        self.assertTrue(os.path.exists('%s/Events/run_01/events_PYTHIA6PT_0.hep.gz' % self.path))
        # sanity check on the size
        self.assertTrue(os.path.getsize('%s/Events/run_01/events_PYTHIA6PT_0.hep.gz' % self.path) > \
                        os.path.getsize('%s/Events/run_01/events.lhe.gz' % self.path))


    def test_check_generate_events_nlo_py6pt_fsr(self):
        """check that py6pt event generation stops in this case (because of fsr)"""
        
        cmd = os.getcwd()
        self.generate(['e+ e- > t t~ [real=QCD]'], 'sm')
        #change to py6
        card = open('%s/Cards/run_card.dat' % self.path).read()
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card.replace('HERWIG6', 'PYTHIA6PT'))       
        #self.do('generate_events -f')        
        self.assertRaises(NLOCmd.aMCatNLOError, self.do, 'generate_events -f')

        
    def test_generate_events_nlo_hw6_stdhep(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        self.do('generate_events aMC@NLO -f')
        
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


    def test_generate_events_nlo_hw6_split(self):
        """test the param_card created is correct"""
        
        cmd = os.getcwd()
        self.generate(['p p > e+ ve [QCD]'], 'loop_sm')
        self.assertEqual(cmd, os.getcwd())
        #change splitevent generation
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
        

    def test_generate_events_nlo_py6_stdhep(self):
        """test the param_card created is correct"""
        
        self.generate_production()
        #change to py6
        card = open('%s/Cards/run_card.dat' % self.path).read()
        open('%s/Cards/run_card.dat' % self.path, 'w').write(card.replace('HERWIG6', 'PYTHIA6Q'))
        
        self.do('generate_events aMC@NLO -f')        
        
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

    
    def test_amcatnlo_from_file(self):
        """ """
        
        cwd = os.getcwd()
        try:
            os.remove('%s/test.log' % self.tmpdir)
        except Exception, error:
            pass
        import subprocess
        
        stdout = open('%s/test.log' % self.tmpdir,'w')
        if logging.getLogger('madgraph').level <= 20:
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stderr=devnull

            
        subprocess.call([pjoin(_file_path, os.path.pardir,'bin','mg5'), 
                         pjoin(_file_path, 'input_files','test_amcatnlo')],
                         cwd=self.tmpdir,
                        stdout=stdout,stderr=stderr)

        stdout.close()
        text = open('%s/test.log' % self.tmpdir,'r').read()
        data = text.split('\n')
        for i,line in enumerate(data):
            if 'Summary:' in line:
                break
        #      Run at p-p collider (6500.0 + 6500.0 GeV)
        self.assertTrue('Run at p-p collider (6500.0 + 6500.0 GeV)' in data[i+2])
        #      Total cross-section: 1.249e+03 +- 3.2e+00 pb        
        cross_section = data[i+4]
        cross_section = float(cross_section.split(':')[1].split('+-')[0])
        # warning, delta may not be compatible with python 2.6 
        try:
            self.assertAlmostEqual(6700.0, cross_section,delta=50)
        except TypeError:
            self.assertTrue(cross_section < 6750.0 and cross_section > 6650.0)

        #      Number of events generated: 10000        
        self.assertTrue('Number of events generated: 100' in data[i+3])


    def test_jet_veto_xsec(self):
        """tests the jet-veto cross section at NNLL+NLO"""
        self.generate_production()
        cmd = """generate_events NLO
                 set ickkw -1
                 set ptj 10
                 """
        open('/tmp/mg5_cmd','w').write(cmd)
        self.assertFalse(self.cmd_line.options['automatic_html_opening'])
        self.cmd_line.import_command_file('/tmp/mg5_cmd')
        self.assertTrue(os.path.exists('%s/Events/run_01/summary.txt' % self.path))
        text=open('%s/Events/run_01/summary.txt' % self.path,'r').read()
        data=text.split('\n')
        for i,line in enumerate(data):
            if 'Process' in line:
                break
        #      Run at p-p collider (6500 + 6500 GeV)
        self.assertTrue('Run at p-p collider (6500.0 + 6500.0 GeV)' in data[i+1])
        cross_section = data[i+2]
        cross_section = float(cross_section.split(':')[1].split('+-')[0])
        try:
            # By default run to 1% precision: set delta=120 for ~2sigma effect.
            self.assertAlmostEqual(5933.0, cross_section,delta=120)
        except TypeError:
            self.assertTrue(cross_section < 4151. and cross_section > 4151.)

    def load_result(self, run_name):
        
        import madgraph.iolibs.save_load_object as save_load_object
        import madgraph.madevent.gen_crossxhtml as gen_crossxhtml
        
        result = save_load_object.load_from_file('%s/HTML/results.pkl' % self.path)
        return result[run_name]
