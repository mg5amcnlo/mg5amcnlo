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
import copy
import fractions
import os 
import sys
import tempfile
import glob
import shutil
import subprocess
from types import SimpleNamespace
from tests import test_manager

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.various.misc as misc
import madgraph.iolibs.files as files
import tests.IOTests as IOTests
import madgraph.interface.master_interface as MGCmd

import madgraph.fks.fks_common as fks_common
import madgraph.iolibs.export_fks as export_fks
import madgraph.iolibs.file_writers as writers

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')


class TestRealChannelExport(unittest.TestCase):
    """Check real topology ownership and the helicity-summed partition."""

    @staticmethod
    def real_me(vertices):
        class MatrixElement(dict):
            def get_number_of_amplitudes(self):
                return len(vertices) + 2

        diagrams = [{'vertices': [{'legs': [None] * size} for size in sizes]}
                    for sizes in vertices]
        return SimpleNamespace(matrix_element=MatrixElement(
            base_amplitude={'diagrams': diagrams},
            diagrams=[{'number': number + 1} for number in range(len(vertices))]))

    def test_channels_own_sectors_without_born_channel_factor(self):
        # The contact diagram leaves a hole in diagram IDs, but not configs.
        matrix_element = SimpleNamespace(
            real_processes=[self.real_me([[3, 3], [4], [3, 3]]),
                            self.real_me([[3]])],
            get_fks_info_list=lambda: [{'n_me': 1}, {'n_me': 1}, {'n_me': 2}])
        exporter = object.__new__(export_fks.ProcessExporterFortranFKS)
        channels, sectors, representatives = exporter.get_real_channels(matrix_element)
        self.assertEqual(channels, [(1, 1, 1, 1), (1, 1, 2, 3), (2, 3, 1, 1)])
        self.assertEqual(sectors, [1, 1, 2])
        self.assertEqual(representatives, [1, 3])
        with tempfile.TemporaryDirectory() as path:
            filename = os.path.join(path, 'real_channels.inc')
            exporter.write_real_channels_file(writers.FortranWriter(filename),
                                              matrix_element)
            with open(filename) as stream:
                output = stream.read().lower()
            self.assertIn('n_real_channels=3', output)
            self.assertIn('max_real_diagrams=5', output)

    def test_no_real_channels(self):
        exporter = object.__new__(export_fks.ProcessExporterFortranFKS)
        matrix_element = SimpleNamespace(real_processes=[],
                                         get_fks_info_list=lambda: [])
        with tempfile.TemporaryDirectory() as path:
            filename = os.path.join(path, 'real_channels.inc')
            exporter.write_real_channels_file(writers.FortranWriter(filename),
                                              matrix_element)
            with open(filename) as stream:
                output = stream.read().lower()
            self.assertIn('n_real_channels=0', output)
            self.assertIn('real_channel_me(1)', output)
            self.assertIn('real_sector_me(1)', output)

    def test_real_diagram_weights_include_cached_helicities(self):
        compiler = shutil.which('gfortran')
        if not compiler:
            self.skipTest('gfortran is needed for the real helicity regression')
        filename = os.path.join(export_fks._file_path,
                  'iolibs/template_files/realmatrix_splitorders_fks.inc')
        with open(filename) as stream:
            template = stream.read()
        template = template[template.index('      SUBROUTINE SMATRIX%(proc_prefix)s_SPLITORDERS'):]
        template = template[:template.index('C Helper functions')]
        # Both helicities have the same full matrix element, but unequal
        # individual diagram squares. Reusing just T would lose one helicity.
        source = template % {
            'proc_prefix': '1', 'info_lines': '', 'process_lines': '',
            'ncomb': 2, 'nSqAmpSplitOrders': 1, 'nAmpSplitOrders': 1,
            'helicity_lines': '      data nhel /1,-1/',
            'den_factor_line': '      data iden /1/',
            'ngraphs': 2, 'nwavefuncs': 1, 'ncolor': 1,
            'wavefunctionsize': 8, 'nb_temp_jamp': 1,
            'color_data_lines': '      data cf /1,0/\n      data denom /1/',
            'helas_calls': ('      amp(1)=dcmplx(p(0,1)+nhel(1),0d0)\n'
                            '      amp(2)=dcmplx(p(0,1)-nhel(1),0d0)'),
            'real_amp2_lines': ('      real_amp2(1)=real_amp2(1)+abs(amp(1))**2\n'
                               '      real_amp2(2)=real_amp2(2)+abs(amp(2))**2'),
            'jamp_lines': '      jamp(1,1)=amp(1)+amp(2)'}
        source += """
      integer function sqsoindex1(i,j)
      integer i,j
      sqsoindex1=1
      end
      program test_real_weights
      implicit none
      double precision p(0:3,1),ans(0:1)
      include 'real_matrix_weights.inc'
      p=0d0
      get_real_diagram_weights=.false.
      real_amp2=-1d0
      p(0,1)=2d0
      call smatrix1_splitorders(p,ans)
      if (ans(0).ne.32d0.or.any(real_amp2.ne.-1d0)) stop 1
      get_real_diagram_weights=.true.
      p(0,1)=3d0
      call smatrix1_splitorders(p,ans)
      if (ans(0).ne.72d0.or.any(real_amp2.ne.20d0)) stop 2
      p(0,1)=4d0
      call smatrix1_splitorders(p,ans)
      if (ans(0).ne.128d0.or.any(real_amp2.ne.34d0)) stop 3
      get_real_diagram_weights=.false.
      real_amp2=-1d0
      call smatrix1_splitorders(p,ans)
      if (ans(0).ne.128d0.or.any(real_amp2.ne.-1d0)) stop 4
      end
"""
        with tempfile.TemporaryDirectory() as path:
            for name, content in {
                'matrix.f': source,
                'nexternal.inc': '      integer nexternal\n      parameter(nexternal=1)\n',
                'coupl.inc': '',
                'real_matrix_weights.inc': (
                    '      double precision real_amp2(2)\n'
                    '      common /c_real_diagram_weights/ real_amp2\n'
                    '      logical get_real_diagram_weights\n'
                    '      common /c_get_real_diagram_weights/ get_real_diagram_weights\n')
            }.items():
                with open(os.path.join(path, name), 'w') as stream:
                    stream.write(content)
            subprocess.check_output([compiler, '-ffixed-line-length-none',
                                     '-fcheck=all', 'matrix.f', '-o', 'check'],
                                    cwd=path, stderr=subprocess.STDOUT)
            subprocess.check_output([os.path.join(path, 'check')], cwd=path,
                                    stderr=subprocess.STDOUT)

class IOExportFKSTest(IOTests.IOTestManager):
    """Test class for the export fks module"""


    def generate(self, process, model, multiparticles=[]):
        """Create a process"""

        def run_cmd(cmd):
            opt = dict(interface.options)
            opt['ninja'] = None
            opt['collier'] = None
            with misc.TMP_variable(interface, 'options', opt):
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
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > t t~ QED^2=0 QCD^2=4 [real=QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_ppw_fksall(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > w+ QED^2=2 QCD^2=0 [QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_tdecay_fksreal(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['t > j j b QED^2=4 QCD^2=0 [real=QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_pptt_fks_loonly(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > t t~ QED^2=0 QCD^2=4 [LOonly=QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_wprod_fksew(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > e+ ve QED^2=4 QCD^2=0 [QED]'], 'loop_qcd_qed_smCMS')

    @IOTests.createIOTest()
    def testIO_test_pptt_fksrealew(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
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
        
        run_cmd('define p3 = d s b d~ s~ b~ a')
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
        
        run_cmd('define p3 = d s b d~ s~ b~')
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



