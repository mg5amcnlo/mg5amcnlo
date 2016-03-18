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
import madgraph.interface.common_run_interface as commonCmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.iolibs.files as files

import madgraph.various.misc as misc
import madgraph.various.lhe_parser as lhe_parser
import madgraph.various.banner as banner_mod

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd

pjoin = os.path.join

    
    


#===============================================================================
# TestCmd
#===============================================================================
class TestMECmdRWGT(unittest.TestCase):
    """this treats all the command not related to MG_ME"""
    
    def setUp(self):
        
        debugging = True
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


    def test_nlo_reweighting(self):
        """check that nlo reweighting is working.
           The main point is to check the recombination of the weights
           Since the rest should be either checked by the lhe_parser classs
           or by the various check of the standalone checks
        """
        
        # create a reweight directory   
        interface = MGCmd.MasterCmd()  
        interface.exec_cmd("import model loop_sm", errorhandling=False)
        interface.exec_cmd("set group_subprocesses False")
        interface.exec_cmd("generate u d~ > e+ ve [virt=QCD]", precmd=True, errorhandling=False)
        interface.exec_cmd('output standalone_rw %s/rw_mevirt -f' % self.path)
        # update the makefile:
        # update make_opts
        m_opts = {}
        if interface.options['lhapdf']:
            #lhapdfversion = subprocess.Popen([mgcmd.options['lhapdf'], '--version'], 
            #        stdout = subprocess.PIPE).stdout.read().strip()[0]
            m_opts['lhapdf'] = True
            m_opts['lhapdfversion'] = 5 # 6 always fail on my computer since 5 is compatible but slower always use 5
            m_opts['llhapdf'] = subprocess.Popen([interface.options['lhapdf'], '--libs'], 
                    stdout = subprocess.PIPE).stdout.read().strip().split()[0]
            m_opts['f2pymode'] = True
        else:
            raise Exception, "need LHAPDF"
            lhapdf = False
            lhapdfversion = 0

        path = pjoin(self.path,'rw_mevirt', 'Source', 'make_opts')
        
        commonCmd.CommonRunCmd.update_make_opts_full(path, m_opts)
    
    
        # Now compile the Source directory
        misc.compile(cwd=pjoin(self.path, 'rw_mevirt', 'Source'))
        #link it 
        with misc.chdir(pjoin(self.path)):
            if self.path not in sys.path:
                sys.path.insert(0, self.path)
            mymod = __import__('rw_mevirt.Source.rwgt2py', globals(), locals(), [],-1)
            mymod =  mymod.Source.rwgt2py
            #mymod.initialise([1,1], 244600)
       
            scales2 =  [[1283.6655, 1283.6655], [1283.6655, 1283.6655], [1283.6655, 1283.6655]] 
            pdg =  [[21, 21], [2, 2]] 
            bjx =  [[0.00036333765, 0.00036942677], [0.5007504, 0.51807252]] 
            wgt =  [[-1.0551457726, 0.890469566701], [0.0, 0.0], [0.0, 0.0]] 
            gs =  [1.3206738, 1.3206738] 
            qcdpower =  [2, 2] 
            orig_wgt =  -0.28722482722716736 
            ref_wgts =  [-1.8403634002861815, 1.5531385730590141] 
       

        #value = mymod.test_pdf() 
        #self.assertAlmostEqual(value, 1.0)
            
        out, partial = mymod.get_wgt(scales2, pdg, bjx, wgt, gs, qcdpower, 1., 1.)  
        #print ref_wgts, partial, [partial[i]/ref_wgts[i] for i in range(2)]
        #print out, orig_wgt, out/orig_wgt
        self.assertAlmostEqual(partial[0], ref_wgts[0], places=2)
        self.assertAlmostEqual(partial[1], ref_wgts[1], places=2)
        self.assertAlmostEqual(out, orig_wgt, places=2)
        
        
        
        
        #if True and __debug__: #this is only for trivial reweighting
        #    if not misc.equal(out, orig_wgt,1):
        #        misc.sprint(event)
        #        for i, computed in enumerate(partial):
        #            if not misc.equal(computed, ref_wgts[i], 3):
        #                misc.sprint("fail since %s != %s for wgt %s " % (computed, ref_wgts[i], i))
        #                misc.sprint(need_V, R, ratio_T)
        #        misc.sprint("fail since %s != %s for the sum." % (out, orig_wgt))
        #        misc.sprint( need_V, R, ratio_T)
        #        raw_input()
                

            
        
        
                           
        
           
           
           
        
        
        
