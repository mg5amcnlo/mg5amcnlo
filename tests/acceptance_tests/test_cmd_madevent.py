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

logger = logging.getLogger('test_cmd')

import tests.unit_tests.iolibs.test_file_writers as test_file_writers

import madgraph.interface.master_interface as MGCmd
import madgraph.interface.madevent_interface as MECmd
import madgraph.interface.launch_ext_program as launch_ext
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
    
    def generate(self, process, model):
        """Create a process"""

        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception, error:
            pass

        interface = MGCmd.MasterCmd()
        interface.onecmd('import model %s' % model)
        if isinstance(process, str):
            interface.onecmd('generate %s' % process)
        else:
            for p in process:
                interface.onecmd('add process %s' % p)
        interface.onecmd('output madevent /tmp/MGPROCESS/ -f')
        if not os.path.exists(pjoin(_file_path, os.path.pardir, 'pythia-pgs')):
            interface.onecmd('install pythia-pgs')
        if not misc.which('root'):
            raise Exception, 'root is require for this test'
        if not os.path.exists(pjoin(_file_path, os.path.pardir, 'MadAnalysis')):
            interface.onecmd('install MadAnalysis')
        
        self.cmd_line = MECmd.MadEventCmdShell(me_dir= '/tmp/MGPROCESS')
        self.cmd_line.exec_cmd('set automatic_html_opening False')

    @staticmethod
    def join_path(*path):
        """join path and treat spaces"""     
        combine = os.path.join(*path)
        return combine.replace(' ','\ ')        
    
    def do(self, line):
        """ exec a line in the cmd under test """        
        self.cmd_line.exec_cmd(line)
        
    def test_width_computation(self):
        """test the param_card created is correct"""
        
        cmd = os.getcwd()
        self.generate(['Z > l+ l-','Z > j j'], 'sm')
        self.assertEqual(cmd, os.getcwd())
        self.do('calculate_decay_widths -f')        
        
        # test the param_card is correctly written
        self.assertTrue(os.path.exists('/tmp/MGPROCESS/Events/run_01/param_card.dat'))
        
        text = open('/tmp/MGPROCESS/Events/run_01/param_card.dat').read()
        data = text.split('DECAY  23')[1].split('DECAY',1)[0]
        self.assertEqual("""1.492240e+00
#  BR             NDA  ID1    ID2   ...
   2.493165e-01   2    3  -3
   2.493165e-01   2    1  -1
   1.944158e-01   2    4  -4
   1.944158e-01   2    2  -2
   5.626776e-02   2    -11  11
   5.626776e-02   2    -13  13
#
#      PDG        Width""", data.strip())
        
    def test_creating_matched_plot(self):
        """test that the creation of matched plot works"""

        cmd = os.getcwd()
        self.generate('p p > W+ j', 'sm')
        self.assertEqual(cmd, os.getcwd())        
        shutil.copy(os.path.join(_file_path, 'input_files', 'run_card_matching.dat'),
                    '/tmp/MGPROCESS/Cards/run_card.dat')
        shutil.copy('/tmp/MGPROCESS/Cards/pythia_card_default.dat',
                    '/tmp/MGPROCESS/Cards/pythia_card.dat')
        self.do('generate_events -f')
        
        f1 = self.check_matched_plot(tag='fermi')         
        start = time.time()
                
        self.assertEqual(cmd, os.getcwd())
        self.do('generate_events -f')
        self.do('pythia run_01 -f')
        self.do('quit')
        
        self.check_parton_output()
        self.check_parton_output('run_02')
        self.check_pythia_output()        
        f2 = self.check_matched_plot(mintime=start, tag='tag_1')        
        
        self.assertNotEqual(f1.split('\n'), f2.split('\n'))
        
        
        self.assertEqual(cmd, os.getcwd())

    def load_result(self, run_name):
        
        import madgraph.iolibs.save_load_object as save_load_object
        import madgraph.various.gen_crossxhtml as gen_crossxhtml
        
        result = save_load_object.load_from_file('/tmp/MGPROCESS/HTML/results.pkl')
        return result[run_name]

    def check_parton_output(self, run_name='run_01', target_event=100):
        """Check that parton output exists and reach the targert for event"""
                
        # check that the number of event is fine:
        data = self.load_result(run_name)
        self.assertEqual(int(data[0]['nb_event']), target_event)
        self.assertTrue('lhe' in data[0].parton)
                
    def check_pythia_output(self, run_name='run_01'):
        """ """
        # check that the number of event is fine:
        data = self.load_result(run_name)
        self.assertTrue('lhe' in data[0].pythia)
        self.assertTrue('log' in data[0].pythia)

    def check_matched_plot(self, run_name='run_01', mintime=None, tag='fermi'):
        """ """
        path = '/tmp/MGPROCESS/HTML/%(run)s/plots_pythia_%(tag)s/DJR1.ps' % \
                                {'run': run_name, 'tag': tag}
        self.assertTrue(os.path.exists(path))
        
        if mintime:
            self.assertTrue(os.path.getctime(path) > mintime)
        
        return open(path).read()
#===============================================================================
# TestCmd
#===============================================================================
class TestMEfromfile(unittest.TestCase):
    """test that we can launch everything from a single file"""

    def test_generation_from_file_1(self):
        """ """
        cwd = os.getcwd()
        try:
            shutil.rmtree('/tmp/MGPROCESS/')
        except Exception, error:
            pass
        import subprocess
        
        devnull =open(os.devnull,'w')
        if not os.path.exists(pjoin(_file_path, os.path.pardir, 'pythia-pgs')):
            p = subprocess.Popen([pjoin(_file_path, os.path.pardir,'bin','mg5')],
                             stdin=subprocess.PIPE,
                             stdout=devnull,stderr=devnull)
            out = p.communicate('install pythia-pgs')
        

        subprocess.call([pjoin(_file_path, os.path.pardir,'bin','mg5'), 
                         pjoin(_file_path, 'input_files','test_mssm_generation')],
                         cwd=pjoin(_file_path, os.path.pardir),
                         stdout=devnull,stderr=devnull)

        
        self.check_parton_output(cross=4.541638, error=0.035)
        self.check_parton_output('run_02', cross=4.541638, error=0.035)
        self.check_pythia_output()
        self.assertEqual(cwd, os.getcwd())
        #

    def load_result(self, run_name):
        
        import madgraph.iolibs.save_load_object as save_load_object
        import madgraph.various.gen_crossxhtml as gen_crossxhtml
        
        result = save_load_object.load_from_file('/tmp/MGPROCESS/HTML/results.pkl')
        return result[run_name]

    def check_parton_output(self, run_name='run_01', target_event=100, cross=0, error=9e99):
        """Check that parton output exists and reach the targert for event"""
                
        # check that the number of event is fine:
        data = self.load_result(run_name)
        self.assertEqual(int(data[0]['nb_event']), target_event)
        self.assertTrue('lhe' in data[0].parton)
        
        if cross:
            self.assertTrue(abs(cross - float(data[0]['cross']))/error < 3)
                
    def check_pythia_output(self, run_name='run_01'):
        """ """
        # check that the number of event is fine:
        data = self.load_result(run_name)
        self.assertTrue('lhe' in data[0].pythia)
        self.assertTrue('log' in data[0].pythia)
        self.assertTrue('hep' in data[0].pythia)

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
        interface.onecmd('import model %s' % model)
        if isinstance(process, str):
            interface.onecmd('generate %s' % process)
        else:
            for p in process:
                interface.onecmd('add process %s' % p)
        interface.onecmd('set automatic_html_opening False')
        interface.onecmd('output madevent /tmp/MGPROCESS/ -f')

    def load_result(self, run_name):
        
        import madgraph.iolibs.save_load_object as save_load_object
        import madgraph.various.gen_crossxhtml as gen_crossxhtml
        
        result = save_load_object.load_from_file('/tmp/MGPROCESS/HTML/results.pkl')
        return result[run_name]

    def check_parton_output(self, run_name='run_01', target_event=100, cross=0, error=9e99):
        """Check that parton output exists and reach the targert for event"""
                
        # check that the number of event is fine:
        data = self.load_result(run_name)
        self.assertEqual(int(data[0]['nb_event']), target_event)
        self.assertTrue('lhe' in data[0].parton)
        if cross:
            self.assertTrue(abs(cross - float(data[0]['cross']))/error < 3)


        
    def test_run_fromP(self):
        """ """
                
        cmd = os.getcwd()
        self.generate('p p > e+ e-', 'sm')
        self.assertEqual(cmd, os.getcwd())
        shutil.copy(os.path.join(_file_path, 'input_files', 'run_card_matching.dat'),
                    '/tmp/MGPROCESS/Cards/run_card.dat')
        os.chdir('/tmp/MGPROCESS/')
        ff = open('cmd.cmd','w')
        ff.write('set automatic_html_opening False\n')
        ff.write('generate_events -f \n') 
        ff.close()
        if logger.getEffectiveLevel() > 20:
            output = open(os.devnull,'w')
        else:
            output = None
        id = subprocess.call(['./bin/madevent','cmd.cmd'], stdout=output, stderr=output)
        self.assertEqual(id, 0)
        self.check_parton_output(cross=946.58, error=1e-2)
        os.chdir(cmd)
        
