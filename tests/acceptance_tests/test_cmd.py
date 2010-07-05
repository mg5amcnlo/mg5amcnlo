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
import unittest
import os
import shutil
import logging

logger = logging.getLogger('test_cmd')

import madgraph.interface.cmd_interface as Cmd
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR

#===============================================================================
# TestCmd
#===============================================================================
class TestCmdShell1(unittest.TestCase):
    """this treats all the command not related to MG_ME"""

    def setUp(self):
        """ basic building of the class to test """
        
        self.cmd = Cmd.MadGraphCmdShell()
    
    @staticmethod
    def join_path(*path):
        """join path and treat spaces"""     
        combine = os.path.join(*path)
        return combine.replace(' ','\ ')        
    
    def do(self, line):
        """ exec a line in the cmd under test """        
        self.cmd.exec_cmd(line)
        
    def test_generate(self):
        """command 'generate' works"""
    
        self.do('load model %s' % self.join_path(_pickle_path, 'sm.pkl'))
        self.do('generate e+ e- > e+ e-')
        self.assertTrue(self.cmd._curr_amps)
        self.do('define P Z u')
        self.do('add process e+ e- > P')
        self.assertEqual(len(self.cmd._curr_amps), 2)
        self.do('add process mu+ mu- > P, Z>mu+mu-')
        self.assertEqual(len(self.cmd._curr_amps), 3)

    def test_draw(self):
        """ command 'draw' works """
        
        self.do('load model %s' % self.join_path(_pickle_path, \
                                                          'sm.pkl'))
        self.do('load processes %s' % self.join_path(_pickle_path,'e+e-_e+e-.pkl'))
        self.do('draw .')
        self.assertTrue(os.path.exists('diagrams_0_e+e-_e+e-.eps'))
        os.remove('diagrams_0_e+e-_e+e-.eps')
        
        self.do('generate g g > g g')
        self.do('draw .')
        self.assertTrue(os.path.exists('diagrams_0_gg_gg.eps'))
        os.remove('diagrams_0_gg_gg.eps')


class TestCmdShell2(unittest.TestCase):
    """The TestCase class for the test the FeynmanLine"""

    def setUp(self):
        """ basic building of the class to test """
        
        
        self.cmd = Cmd.MadGraphCmdShell()
        if  MG4DIR:
            logger.debug("MG_ME dir: " + MG4DIR)
            self.out_dir = os.path.join(MG4DIR, 'AUTO_TEST_MG5')
        else:
            raise Exception, 'NO MG_ME dir for this test'   
        self.do('import model_v4 sm')
        
        
    def tearDown(self):
        """ basic destruction after have run """
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
    
    join_path = TestCmdShell1.join_path

    def do(self, line):
        """ exec a line in the cmd under test """        
        self.cmd.exec_cmd(line)
    
    
    def test_setup_madevent_directory(self):
        """ command 'setup' works with path"""
        
        self.do('load processes %s' % self.join_path(_pickle_path,'e+e-_e+e-.pkl'))
        self.do('setup madevent_v4 %s -nojpeg' % self.out_dir)
        self.assertTrue(os.path.exists(self.out_dir))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'SubProcesses', 'P0_e+e-_e+e-')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'proc_card_mg5.dat')))
        self.assertFalse(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_e+e-_e+e-',
                                                    'matrix1.jpg')))
        self.do('finalize')
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_e+e-_e+e-',
                                                    'matrix1.jpg')))

        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'madevent.tar.gz')))        

    def test_read_madgraph4_proc_card(self):
        """ command 'setup' works with '.' """
        os.system('cp -rf %s %s' % (os.path.join(MG4DIR,'Template'),
                                    self.out_dir))
        os.system('cp -rf %s %s' % (
                            self.join_path(_pickle_path,'simple_v4_proc_card.dat'),
                            os.path.join(self.out_dir,'Cards','proc_card.dat')))
        
        self.do('import proc_v4 %s' % os.path.join(self.out_dir,
                                                       'Cards','proc_card.dat'))

        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                              'SubProcesses', 'P1_e-e+_vevex')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'proc_card_mg5.dat')))
        self.assertFalse(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_e+e-_e+e-',
                                                    'matrix1.jpg')))

        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'madevent.tar.gz')))


    def test_read_madgraph4_proc_card_with_setup(self):
        """ command 'setup' works with '.' """

        self.do('setup madevent_v4 %s' % self.out_dir)
        self.do('import proc_v4 %s' % self.join_path(_pickle_path, \
                                                     'simple_v4_proc_card.dat'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                              'SubProcesses', 'P1_e-e+_vevex')))        
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'proc_card_mg5.dat')))


    def test_setup_standalone_directory(self):
        """ command 'setup' works with path"""
        
        self.do('load processes %s' % self.join_path(_pickle_path,'e+e-_e+e-.pkl'))
        self.do('setup standalone_v4 %s' % self.out_dir)
        self.assertTrue(os.path.exists(self.out_dir))
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, 'lib', 'libdhelas3.a')))
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, 'lib', 'libmodel.a')))
        self.do('export')
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'SubProcesses', 'P0_e+e-_e+e-')))
        self.do('finalize')
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'Cards', 'proc_card_mg5.dat')))
