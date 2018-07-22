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
import math

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

#===============================================================================
# TestCmd
#===============================================================================
class TestMadSpin(unittest.TestCase):
    """test that we can launch everything from a single file"""

    def setUp(self):
        
        self.debuging = False
        if self.debuging:
            self.path = pjoin(MG5DIR, 'MS_TEST')
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            os.mkdir(self.path) 
        else:
            self.path = tempfile.mkdtemp(prefix='ms_test_mg5')
        self.run_dir = pjoin(self.path, 'MGPROC') 
        
    
    def tearDown(self):

        if not self.debuging:
            shutil.rmtree(self.path)
        self.assertFalse(self.debuging)


    def test_hepmc_decay(self):
        """ """
        
        cwd = os.getcwd()
        
        files.cp(pjoin(MG5DIR, 'tests', 'input_files', 'test.hepmc.gz'), self.path)


        fsock = open(pjoin(self.path, 'test_hepmc'),'w')
        text = """
        set spinmode none
        set cross_section {0:1.0}
        set new_wgt BR
        set input_format hepmc
        import ./test.hepmc.gz
        import model %s/tests/input_files/DM_pion %s/tests/input_files/DM_pion/param_pion.dat
        decay k0 > xr xr a
        launch
        """ % (MG5DIR, MG5DIR)
        
        fsock.write(text)
        fsock.close()

        import subprocess
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'),
                         pjoin(self.path, 'test_hepmc')],
                        cwd=pjoin(self.path),
                        stdout=stdout,stderr=stderr)
        self.assertTrue(os.path.exists(pjoin(self.path, 'test_decayed.lhe.gz')))
        lhe = lhe_parser.EventFile(pjoin(self.path, 'test_decayed.lhe.gz'))
        self.assertEqual(10, len(lhe))
        
        nb_dec = 0
        nb_photon = 0
        for event in lhe:
            self.assertEqual(event.nexternal, len(event))
            for particle in event:
                if particle.pdg == 130:
                    self.assertEqual(particle.status,2)
                    nb_dec +=1
                if particle.pdg ==22:
                    nb_photon += 1
                    
        self.assertEqual(nb_dec, 116)
        self.assertEqual(nb_photon, 116)

    def test_lhe_none_decay(self):
        """ """
        
        cwd = os.getcwd()
        
        files.cp(pjoin(MG5DIR, 'tests', 'input_files', 'test_spinmode_none.lhe.gz'), self.path)


        fsock = open(pjoin(self.path, 'test_hepmc'),'w')
        text = """
        set spinmode none
        import ./test_spinmode_none.lhe.gz
        decay z > mu+ mu-
        launch
        """
        
        fsock.write(text)
        fsock.close()

        import subprocess
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'),
                         pjoin(self.path, 'test_hepmc')],
                        cwd=pjoin(self.path),
                        stdout=stdout,stderr=stderr)

        self.assertTrue(os.path.exists(pjoin(self.path, 'test_spinmode_none_decayed.lhe.gz')))
        lhe = lhe_parser.EventFile(pjoin(self.path, 'test_spinmode_none_decayed.lhe.gz'))
        self.assertEqual(100, len(lhe))
        
        nb_dec = 0
        nb_muon = 0
        for event in lhe:
            muon_in = 0
            self.assertEqual(event.nexternal, len(event))
            for particle in event:
                if particle.pdg == 23:
                    self.assertEqual(particle.status,2)
                    nb_dec += 1
                if particle.pdg == 13:
                    nb_muon += 1
                    muon_in +=1
            self.assertEqual(muon_in, 1)
        self.assertEqual(nb_dec, 189)
        self.assertEqual(nb_muon, 100)