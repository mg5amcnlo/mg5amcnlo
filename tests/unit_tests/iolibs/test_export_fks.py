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
"""Unit test library for the export_FKS format routines"""

from __future__ import absolute_import
import six
StringIO = six
import copy
import fractions
import os 
import sys
import tempfile
import glob
import shutil
import six
from six.moves import zip
from tests import test_manager

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.various.misc as misc
import madgraph.iolibs.files as files
import tests.IOTests as IOTests
import madgraph.interface.master_interface as MGCmd

import madgraph.fks.fks_common as fks_common

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')

class IOExportFKSTest(IOTests.IOTestManager):
    """Test class for the export fks module"""


    def generate(self, process, model, multiparticles=[]):
        """Create a process"""

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        
        if model.endswith('CMS'):
            run_cmd('set complex_mass_scheme')
            model = model[:-3]

        run_cmd('import model %s' % model)
        for multi in multiparticles:
            run_cmd('define %s' % multi)
        if isinstance(process, str):
            run_cmd('generate %s' % process)
        else:
            for p in process:
                run_cmd('add process %s' % p)

        files.rm(self.IOpath)
        run_cmd('output %s -f' % self.IOpath)


    @IOTests.createIOTest()
    def testIO_test_pptt_fksreal(self):
        """ target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > t t~ QED^2=0 QCD^2=4 [real=QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_ppw_fksall(self):
        """ target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > w+ QED^2=2 QCD^2=0 [QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_tdecay_fksreal(self):
        """ target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['t > j j b QED^2=4 QCD^2=0 [real=QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_pptt_fks_loonly(self):
        """ target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > t t~ QED^2=0 QCD^2=4 [LOonly=QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_wprod_fksew(self):
        """ target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > e+ ve QED^2=4 QCD^2=0 [QED]'], 'loop_qcd_qed_smCMS')

    @IOTests.createIOTest()
    def testIO_test_pptt_fksrealew(self):
        """ target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > t t~ QED^2=0 QCD^2=4 [real=QED]'], 'sm', 
                      multiparticles = ['p = u u~ d d~ s s~ c c~ g a'])



class TestFKSOutput(unittest.TestCase):
    """ this class is to test that the new and old nlo generation give
    identical results
    """

    def tearDown(self):
        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        run_cmd('set low_mem_multicore_nlo_generation False')
        run_cmd('set OLP MadLoop')

    def test_w_nlo_gen_qcd(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > e+ ve [QCD]
        """
        path = tempfile.mkdtemp('', 'TMPWTest', None)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        
        run_cmd('set low_mem_multicore_nlo_generation True')
        run_cmd('generate p p > e+ ve QED^2=4 QCD^2=0 [QCD]')
        run_cmd('output %s' % os.path.join(path, 'W-newway'))
        run_cmd('set low_mem_multicore_nlo_generation False')
        run_cmd('generate p p > e+ ve QED^2=4 QCD^2=0 [QCD]')
        run_cmd('output %s' % os.path.join(path, 'W-oldway'))
        
        # the P0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', '*.f')) + \
           [os.path.join(path, 'W-oldway', 'SubProcesses', 'proc_characteristics')]):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

        # the V0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', 'V0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', 'V0*', '*.f'))):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)


#    @test_manager.bypass_for_py3
    def test_w_nlo_gen_qed(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > e+ ve [QED]
        """
        path = tempfile.mkdtemp('', 'TMPWTest', None)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        
        run_cmd('set low_mem_multicore_nlo_generation True')
        run_cmd('generate p p > e+ ve QED^2=4 QCD^2=0 [QED]')
        run_cmd('output %s' % os.path.join(path, 'W-newway'))
        run_cmd('set low_mem_multicore_nlo_generation False')
        run_cmd('generate p p > e+ ve QED^2=4 QCD^2=0 [QED]')
        run_cmd('output %s' % os.path.join(path, 'W-oldway'))
        
        # the P0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', '*.f')) + \
           [os.path.join(path, 'W-oldway', 'SubProcesses', 'proc_characteristics')]):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

        # the V0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', 'V0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', 'V0*', '*.f'))):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)


 #   @test_manager.bypass_for_py3
    def test_z_nlo_gen_qed(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > e+ e- [QED], in particular
        that b-initiated processes are NOT combined with d-s ones
        """
        path = tempfile.mkdtemp('', 'TMPWTest', None)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        
        run_cmd('define p3 = d s b d~ s~ b~')
        run_cmd('set low_mem_multicore_nlo_generation True')
        run_cmd('generate p3 p3 > e+ e- QED^2=4 QCD^2=0 [QED]')
        run_cmd('output %s' % os.path.join(path, 'Z-newway'))
        run_cmd('set low_mem_multicore_nlo_generation False')
        run_cmd('generate p3 p3 > e+ e- QED^2=4 QCD^2=0 [QED]')
        run_cmd('output %s' % os.path.join(path, 'Z-oldway'))
        
        # the P0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', '*.f')) + \
           [os.path.join(path, 'Z-oldway', 'SubProcesses', 'proc_characteristics')]):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

        # the V0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', 'V0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', 'V0*', '*.f'))):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

#    @test_manager.bypass_for_py3
    def test_z_nlo_gen_qcd(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > e+ e- [QED], in particular
        that the aa initiated folder is generated without virtuals
        """
        path = tempfile.mkdtemp('', 'TMPWTest', None)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        
        run_cmd('define p3 = d s b d~ s~ b~ a')
        run_cmd('set low_mem_multicore_nlo_generation True')
        run_cmd('generate p3 p3 > e+ e- QED^2=4 QCD^2=0 [QCD]')
        run_cmd('output %s' % os.path.join(path, 'Z-newway'))
        run_cmd('set low_mem_multicore_nlo_generation False')
        run_cmd('generate p3 p3 > e+ e- QED^2=4 QCD^2=0 [QCD]')
        run_cmd('output %s' % os.path.join(path, 'Z-oldway'))
        
        # the P0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', '*.f')) + \
           [os.path.join(path, 'Z-oldway', 'SubProcesses', 'proc_characteristics')]):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

        # the V0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', 'V0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', 'V0*', '*.f'))):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

#    @test_manager.bypass_for_py3
    def test_wj_loonly_gen(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > w j [LOonly=QCD]
        """
        path = tempfile.mkdtemp('', 'TMPWTest', None)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        

        run_cmd('set low_mem_multicore_nlo_generation True')
        run_cmd('generate p p > w+ j [LOonly=QCD]')
        run_cmd('output %s' % os.path.join(path, 'W-newway'))
        run_cmd('set low_mem_multicore_nlo_generation False')
        run_cmd('generate p p > w+ j [LOonly=QCD]')
        run_cmd('output %s' % os.path.join(path, 'W-oldway'))
                
        # the P0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', '*.f')) + \
           [os.path.join(path, 'W-oldway', 'SubProcesses', 'proc_characteristics')]):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

        # the V0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', 'V0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', 'V0*', '*.f'))):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

        # the Source dir 
        for oldf in \
          (glob.glob(os.path.join(path, 'W-oldway', 'Source', '*.inc')) + \
           glob.glob(os.path.join(path, 'W-oldway', 'Source', '*.f'))):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)


    @test_manager.bypass_for_py3
    def test_w_nlo_gen_gosam(self):
        """check that the new generation mode works when gosam is set 
        for p p > w [QCD] 
        """
        path = tempfile.mkdtemp('', 'TMPWTest', None)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        try:
            run_cmd('set low_mem_multicore_nlo_generation True')
            run_cmd('set OLP GoSam')
            run_cmd('generate p p > w+ QED^2=2 QCD^2=0 [QCD]')
            try:
                run_cmd('output %s' % os.path.join(path, 'W-newway'))
            except fks_common.FKSProcessError as err:
                # catch the error if gosam is not there
                if not 'Generation of the virtuals with GoSam failed' in str(err):
                    raise Exception(err)
        except Exception as e:
            run_cmd('set low_mem_multicore_nlo_generation False')
            run_cmd('set OLP MadLoop')
            raise e
        finally:
            run_cmd('set low_mem_multicore_nlo_generation False')
            run_cmd('set OLP MadLoop')

        shutil.rmtree(path)



