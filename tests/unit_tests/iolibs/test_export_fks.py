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
"""Unit test library for the export_FKS format routines"""

from __future__ import absolute_import
import copy
import fractions
import os
import re
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
import madgraph.fks.fks_helas_objects as fks_helas
import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.export_fks as export_fks
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.file_writers as writers
from madgraph import MadGraph5Error

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')

class TestBornDirCollision(unittest.TestCase):
    """P<...> directory names must never be shared by two different matrix
    elements.

    shell_string() concatenates particle names and polarization labels with
    no separator, and every FKS born process carries id 0, so a clash is
    possible for some models.  os.mkdir always raises EEXIST, so the base
    died loudly but opaquely; the guard names both processes instead.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp('', 'TMPBornDir', None)
        self.cwd = os.getcwd()
        os.chdir(self.tmpdir)
        self.model = base_objects.Model()
        self.model.set('particles', base_objects.ParticleList([
            base_objects.Particle({'name': 'c', 'antiname': 'c~',
                                   'pdg_code': 3})]))

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir)

    def make_process(self, polarization):
        legs = base_objects.LegList(
            [base_objects.Leg({'id': 3, 'number': i + 1, 'state': i > 1})
             for i in range(4)])
        legs[2].set('polarization', polarization)
        return base_objects.Process({'legs': legs, 'model': self.model,
                                     'id': 0})

    def test_born_dir_collision_is_explicit(self):
        """A second matrix element asking for a taken directory must get an
        error naming both processes, not a bare FileExistsError."""

        exporter = export_fks.ProcessExporterFortranFKS.__new__(
                                    export_fks.ProcessExporterFortranFKS)

        first = self.make_process([1])
        second = self.make_process([-1])
        # force the clash: whatever shell_string() does, both want 'P0_clash'
        exporter.mkdir_born_dir('P0_clash', first)
        self.assertTrue(os.path.isdir('P0_clash'))

        try:
            exporter.mkdir_born_dir('P0_clash', second)
        except MadGraph5Error as error:
            message = str(error)
        else:
            self.fail('no error raised on a P directory name collision')

        self.assertIn('P0_clash', message)
        self.assertIn(first.nice_string(prefix=False).strip(), message)
        self.assertIn(second.nice_string(prefix=False).strip(), message)

    def test_born_dir_no_collision(self):
        """Distinct names are created without complaint, and the guard does
        not interfere with an ordinary (unpolarized) name."""

        exporter = export_fks.ProcessExporterFortranFKS.__new__(
                                    export_fks.ProcessExporterFortranFKS)
        plain = self.make_process([])
        self.assertEqual('0_cc_cc', plain.shell_string())
        exporter.mkdir_born_dir('P%s' % plain.shell_string(), plain)

        # 'c{S,S}' and 'c{A}' are different restrictions and used to render
        # the same string; they must now get one directory each
        left = self.make_process([9, 9])
        right = self.make_process([99])
        self.assertNotEqual(left.shell_string(), right.shell_string())
        exporter.mkdir_born_dir('P%s' % left.shell_string(), left)
        exporter.mkdir_born_dir('P%s' % right.shell_string(), right)
        self.assertEqual(sorted(os.listdir('.')),
                         ['P0_cc_c99c', 'P0_cc_c9c', 'P0_cc_cc'])


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

    def test_real_diagram_weights_follow_computed_helicities(self):
        compiler = shutil.which('gfortran')
        if not compiler:
            self.skipTest('gfortran is needed for the real helicity regression')
        filename = os.path.join(export_fks._file_path,
                  'iolibs/template_files/realmatrix_splitorders_fks.inc')
        aloha_functions = os.path.join(export_fks._file_path, os.pardir,
                          'aloha/template_files/aloha_functions.f')
        with open(filename) as stream:
            template = stream.read()
        template = template[template.index('      SUBROUTINE SMATRIX%(proc_prefix)s_SPLITORDERS'):]
        template = template[:template.index('C Helper functions')]
        # Helicity -1 has the same full matrix element as +1 and is not
        # recomputed; helicity 0 has none (its two diagrams cancel) and is
        # skipped. Both have nonzero diagram squares, which the weights
        # leave out at every point, the first one included.
        source = template % {
            'proc_prefix': '1', 'info_lines': '', 'process_lines': '',
            'ncomb': 3, 'nSqAmpSplitOrders': 1, 'nAmpSplitOrders': 1,
            'helicity_lines': '      data nhel /1,-1,0/',
            'den_factor_line': '      data iden /1/',
            'ngraphs': 2, 'nwavefuncs': 1, 'ncolor': 1,
            'wavefunctionsize': 8, 'nb_temp_jamp': 1,
            'color_data_lines': '      data cf /1,0/\n      data denom /1/',
            'helas_calls': ('      amp(1)=dcmplx(p(0,1)+nhel(1),0d0)\n'
                            '      amp(2)=dcmplx((2*nhel(1)**2-1)*p(0,1)'
                            '-nhel(1),0d0)'),
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
      get_real_diagram_weights=.true.
      real_amp2=-1d0
      p(0,1)=2d0
      call smatrix1_splitorders(p,ans)
      if (ans(0).ne.32d0.or.any(real_amp2.ne.(/9d0,1d0/))) stop 1
      p(0,1)=3d0
      call smatrix1_splitorders(p,ans)
      if (ans(0).ne.72d0.or.any(real_amp2.ne.(/16d0,4d0/))) stop 2
      get_real_diagram_weights=.false.
      real_amp2=-1d0
      p(0,1)=4d0
      call smatrix1_splitorders(p,ans)
      if (ans(0).ne.128d0.or.any(real_amp2.ne.-1d0)) stop 3
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
                                     '-fcheck=all', aloha_functions,
                                     'matrix.f', '-o', 'check'],
                                    cwd=path, stderr=subprocess.STDOUT)
            subprocess.check_output([os.path.join(path, 'check')], cwd=path,
                                    stderr=subprocess.STDOUT)


class TestBornGscale(unittest.TestCase):
    """gscale: when only alpha_S changed, born.f rescales its cached Born
    amplitudes by (G'/G)**n instead of recomputing them (BORN_GSCALE). With
    several split orders each one has its own power n, which must follow the
    QCD order of the amplitudes, and each coupling is checked against the
    QCD order of its interaction."""

    @classmethod
    def setUpClass(cls):
        interface = MGCmd.MasterCmd()
        interface.do_import('model loop_sm')
        # s-channel gluon (QCD order 2) and photon/Z (QCD order 0)
        interface.do_generate('u u~ > t t~ QCD^2<=4 QED^2<=4 [real=QCD]')
        helas = fks_helas.FKSHelasMultiProcess(interface._fks_multi_proc)
        cls.born_me = helas.get_matrix_elements()[0].born_me
        cls.model = interface._curr_model

    @staticmethod
    def data(lines, name):
        """The values of the Fortran array NAME in the DATA lines"""
        values = []
        for m in re.finditer(r'DATA \(%s\(I\),I=\d+,\d+\) /([-0-9,]+)/'
                             % name, lines):
            values += [int(v) for v in m.group(1).split(',')]
        return values

    def test_mixed_split_order_powers(self):
        """Each amplitude, amplitude split order and squared split order
        gets the power of G of its own QCD order."""

        out = export_fks.ProcessExporterFortranFKS.get_born_gscale_data(
                                                                self.born_me)
        lines = out['gscale_data_lines']
        amp_qcd = self.data(lines, 'AMPQCD')
        self.assertEqual(sorted(amp_qcd), [0, 0, 2])
        # the amplitude split orders are sorted by weight: QCD first
        squared_orders, amp_orders = self.born_me.get_split_orders_mapping()
        self.assertEqual(self.data(lines, 'AMPSOQCD'), [2, 0])
        for (orders, numbers), power in zip(amp_orders, [2, 0]):
            self.assertEqual(set(amp_qcd[n - 1] for n in numbers), {power})
        # QCD^2=4, QCD^2=2 QED^2=2 and QED^2=4
        self.assertEqual(self.data(lines, 'SQSOQCD'), [4, 2, 0])
        self.assertIn('DATA GSCALE_FAST /.TRUE./', lines)
        self.assertEqual(out['maxqcdpow'], 4)

    def test_coupling_powers(self):
        """The coupling of the gluon to the quarks must scale as G, the
        electroweak ones not at all."""

        out = export_fks.ProcessExporterFortranFKS.get_born_coupling_check_dict(
                                                                self.born_me)
        self.assertEqual(out['born_checks_amps'], '.TRUE.')
        self.assertEqual(out['born_gscale'], '.TRUE.')
        gluon = set()
        for inter in self.model.get('interactions'):
            pdgs = sorted(p.get_pdg_code() for p in inter.get('particles'))
            if pdgs in ([-6, 6, 21], [-2, 2, 21]):
                gluon.update(inter.get('couplings').values())
        powers = dict(re.findall(r'abs\((\w+)-saved_coup\(\d+\)\*rpow\((\d)\)',
                                 out['coup_gscale_lines']))
        self.assertTrue(gluon & set(powers))
        for name, power in powers.items():
            self.assertEqual(power, '1' if name in gluon else '0', name)


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


    def test_polarised_default_frame_is_the_massive_final_state(self):
        """The partonic c.m., me_frame = [1, 2], has no infrared-safe meaning
        for p p > w+{0} j at NLO: with initial-state QCD radiation the frame
        the FKS momenta come in depends on the sector (+9.5(4)% when only
        fks_b changed), and the only frame equal to the partonic c.m. of the
        Born contains the jet. So the output must write the rest frame of the
        massive final state, [3], as the run_card default, and the run_card
        must still refuse [1, 2] when asked for explicitly."""
        import madgraph.various.banner as banner_mod
        path = tempfile.mkdtemp('', 'TMPWPolTest', None)
        interface = MGCmd.MasterCmd()
        try:
            interface.exec_cmd('generate p p > w+{0} j [real=QCD]',
                               errorhandling=False, printcmd=False,
                               precmd=True, postcmd=True)
            interface.exec_cmd('output %s' % os.path.join(path, 'W'),
                               errorhandling=False, printcmd=False,
                               precmd=True, postcmd=True)

            charac = banner_mod.ProcCharacteristic(os.path.join(
                path, 'W', 'SubProcesses', 'proc_characteristics'))
            self.assertEqual(charac['me_frame_cm_legs'], [3, 4])
            self.assertEqual(charac['me_frame_unsafe_legs'], [1, 2, 4])
            self.assertEqual(charac['me_frame_default_legs'], [3])

            # the card as written: the W rest frame, accepted at launch
            run_card = banner_mod.RunCardNLO(os.path.join(
                path, 'W', 'Cards', 'run_card.dat'))
            self.assertEqual(run_card['me_frame'], [3])
            run_card.set_me_frame_process(charac)
            run_card.check_validity()
            run_card.update_system_parameter_for_include()
            self.assertEqual(run_card['frame_id'], 8)
            # [1, 2] by hand: refused with real emission
            run_card.set('me_frame', [1, 2], user=True)
            self.assertRaises(banner_mod.InvalidRunCard,
                              run_card.check_validity)
            # the Born alone (launch LO) has nothing to be infrared safe for
            run_card.set_me_frame_process(charac, real_emission=False)
            run_card.check_validity()

            # A bare 'treatcards run', as the reweighting and the FKS
            # standalone check compile their Source: no run_card is loaded
            # in that shell. It used to die on self.run_card; now the include
            # file resolves me_frame against the process found on disk.
            me_dir = os.path.join(path, 'W')
            def treatcards():
                with open(os.path.join(me_dir, 'treat.cmd'), 'w') as cmd:
                    cmd.write('treatcards run\n')
                return subprocess.run(
                    [sys.executable, os.path.join('bin', 'aMCatNLO'),
                     'treat.cmd'], cwd=me_dir, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, universal_newlines=True).stdout
            def frame_id():
                with open(os.path.join(me_dir, 'Source', 'run_card.inc')) as inc:
                    return [l.split('=')[1].strip() for l in inc
                            if 'FRAME_ID' in l.upper()]
            out = treatcards()
            self.assertNotIn('Error', out)
            self.assertEqual(frame_id(), ['8'])
            # a beam that radiates is refused on that path too
            card_path = os.path.join(me_dir, 'Cards', 'run_card.dat')
            with open(card_path) as card:
                text = card.read()
            text = re.sub(r'^(\s*)3(\s*=\s*me_frame)', r'\g<1>1, 3\2', text,
                          count=1, flags=re.M)
            with open(card_path, 'w') as card:
                card.write(text)
            self.assertIn('radiate', treatcards())
        finally:
            shutil.rmtree(path, ignore_errors=True)


    def test_low_mem_gen_with_respawning_pool(self):
        """the low-memory NLO generation reuses a maxtasksperchild=1 pool for
        several map_async calls. Between two calls its workers have exited and
        are replaced by the pool's handler thread, so len(pool._pool) can be
        zero, and without an explicit chunksize CPython divides by it
        (intermittent ZeroDivisionError). Simulate that state for every
        map_async call: the generation must not depend on it.
        """
        import multiprocessing.pool
        from unittest import mock

        map_async = multiprocessing.pool.Pool._map_async
        def map_async_while_respawning(pool, *args, **opts):
            workers, pool._pool = pool._pool, []
            try:
                return map_async(pool, *args, **opts)
            finally:
                pool._pool = workers

        path = tempfile.mkdtemp('', 'TMPWTest', None)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False,
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        try:
            run_cmd('set low_mem_multicore_nlo_generation True')
            run_cmd('generate p p > w+ [QCD]')
            with mock.patch.object(multiprocessing.pool.Pool, '_map_async',
                                   map_async_while_respawning):
                run_cmd('output %s' % os.path.join(path, 'W-newway'))
            self.assertTrue(glob.glob(os.path.join(path, 'W-newway',
                                                   'SubProcesses', 'P0*', 'V0*')))
        finally:
            run_cmd('set low_mem_multicore_nlo_generation False')
            shutil.rmtree(path)

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





class TestColorMatrixEncodingGate(unittest.TestCase):
    """The color matrix may be written in the compressed form -- one line per
    orbit of the color basis symmetry, the entries rebuilt on the first call by
    INIT_CF -- only by the templates that declare CF in the common block
    INIT_CF fills and call it.

    The FKS templates do none of that: CF is a plain local array whose only
    filling is the DATA statements of get_color_data_lines. Writing the
    compressed form there left CF at zero, so every real emission matrix
    element with a large enough color basis came out zero -- for
    g g > t t~ g g g (120 color structures) that is every soft and collinear
    check of test_soft_col_limits, which refuses to let the run start.
    """

    @classmethod
    def setUpClass(cls):
        # g g > g g: six color structures with a non-trivial symmetry, so the
        # compressed form is available whenever the exporter allows it
        interface = MGCmd.MasterCmd()
        interface.exec_cmd('import model sm', errorhandling=False,
                           printcmd=False, precmd=True, postcmd=True)
        interface.exec_cmd('generate g g > g g', errorhandling=False,
                           printcmd=False, precmd=True, postcmd=True)
        cls.matrix_element = helas_objects.HelasMatrixElement(
                                                        interface._curr_amps[0])

    @staticmethod
    def entry_lines(lines):
        """The DATA statements holding the color matrix entries themselves."""
        return [line for line in lines if 'CF(i)' in line or 'CF(i,' in line]

    def exporter(self, cls):
        """An exporter of that class with the compression always worth taking,
        so that only color_matrix_encoding_allowed decides."""
        instance = cls.__new__(cls)
        instance.color_encoding_margin = 0
        return instance

    def test_standalone_compresses(self):
        """The plain standalone output does rebuild CF at run time, so it only
        writes the denominator out. Without this the tests below would pass on
        a matrix element that is never compressed in the first place."""

        exporter = self.exporter(export_v4.ProcessExporterFortranSA)
        exporter.opt = {'export_format': 'standalone'}
        self.assertTrue(exporter.color_matrix_encoding_allowed(
                                                          self.matrix_element))
        lines = exporter.get_color_data_lines(self.matrix_element)
        self.assertEqual([], self.entry_lines(lines))
        self.assertTrue(any('Denom' in line for line in lines))

    def test_fks_writes_the_entries_out(self):
        """The FKS exporter has no INIT_CF, so the entries must be written."""

        exporter = self.exporter(export_fks.ProcessExporterFortranFKS)
        self.assertFalse(exporter.color_matrix_encoding_allowed(
                                                          self.matrix_element))
        lines = exporter.get_color_data_lines(self.matrix_element)
        self.assertNotEqual([], self.entry_lines(lines))

    def test_split_orders_and_madspin_write_the_entries_out(self):
        """The standalone variants select a template of their own -- split
        orders, MadSpin's msP/msF, matchbox -- and none of them rebuilds CF."""

        for export_format in ('standalone_msP', 'standalone_msF', 'matchbox',
                              'madloop_matchbox'):
            exporter = self.exporter(export_v4.ProcessExporterFortranSA)
            exporter.opt = {'export_format': export_format}
            self.assertFalse(exporter.color_matrix_encoding_allowed(
                                                           self.matrix_element),
                             'encoding allowed for %s' % export_format)

        # split orders send the plain standalone output to
        # matrix_standalone_splitOrders_v4.inc, which has no INIT_CF either
        split = copy.deepcopy(self.matrix_element)
        split.get('processes')[0].set('split_orders', ['QCD'])
        exporter = self.exporter(export_v4.ProcessExporterFortranSA)
        exporter.opt = {'export_format': 'standalone'}
        self.assertFalse(exporter.color_matrix_encoding_allowed(split))
        self.assertNotEqual([], self.entry_lines(
                                        exporter.get_color_data_lines(split)))

    def test_madevent_compresses_and_madweight_does_not(self):
        """madevent calls INIT_CF from every one of its templates; MadWeight
        writes a template of its own that does not."""

        exporter = self.exporter(export_v4.ProcessExporterFortranME)
        self.assertTrue(exporter.color_matrix_encoding_allowed(
                                                          self.matrix_element))

        exporter = self.exporter(export_v4.ProcessExporterFortranMW)
        self.assertFalse(exporter.color_matrix_encoding_allowed(
                                                          self.matrix_element))

    def test_whitelist_matches_the_templates(self):
        """COLOR_MATRIX_ENCODING_TEMPLATES is the list of templates that call
        INIT_CF. A template added or changed on either side has to move on the
        other, or an exporter is again free to drop the entries of a CF that
        nothing fills."""

        directory = os.path.join(os.path.dirname(export_v4.__file__),
                                 'template_files')
        seen = set()
        for name in os.listdir(directory):
            if not name.endswith('.inc'):
                continue
            text = open(os.path.join(directory, name)).read()
            if '%(color_data_lines)s' not in text:
                self.assertNotIn('%(color_init_routine)s', text,
                                 '%s rebuilds a color matrix it never writes'
                                 % name)
                continue
            rebuilds = 'INIT_CF' in text
            self.assertEqual(rebuilds,
                             name in export_v4.COLOR_MATRIX_ENCODING_TEMPLATES,
                             '%s: INIT_CF=%s but whitelisted=%s' % \
                             (name, rebuilds,
                              name in export_v4.COLOR_MATRIX_ENCODING_TEMPLATES))
            if rebuilds:
                self.assertIn('%(color_init_routine)s', text,
                              '%s calls INIT_CF but never defines it' % name)
                seen.add(name)
        self.assertEqual(seen, set(export_v4.COLOR_MATRIX_ENCODING_TEMPLATES))
