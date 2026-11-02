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
from __future__ import absolute_import
import array
import random
import subprocess
import unittest
import os
import re
import shutil
import sys
import logging
import tempfile

pjoin = os.path.join

logger = logging.getLogger('test_cmd')

import tests.unit_tests.iolibs.test_file_writers as test_file_writers
import tests.IOTests as IOTests

import madgraph.interface.master_interface as Cmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.iolibs.files as files
import madgraph.core.diagram_generation as diagram_generation
import madgraph.various.misc as misc
import madgraph.various.lhe_parser as lhe_parser
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd
from tests import test_manager

_v4_model_path = os.path.join(MG5DIR, 'tests', 'input_files', 'full_sm')

#===============================================================================
# TestCmd
#===============================================================================
class TestCmdShell1(unittest.TestCase):
    """this treats all the command not related to MG_ME"""

    def setUp(self):
        """ basic building of the class to test """
        
        self.cmd = Cmd.MasterCmd()
    
    @staticmethod
    def join_path(*path):
        """join path and treat spaces"""   

        combine = os.path.join(*path)
        return combine.replace(' ',r'\ ')        
    
    def do(self, line):
        """ exec a line in the cmd under test """        
        self.cmd.exec_cmd(line)
        
    def test_generate(self):
        """command 'generate' works"""
        
        self.do('import model sm')
        self.cmd._curr_model.pass_particles_name_in_mg_default()
        self.do('generate e+ e- > e+ e- QED<=2')
        self.assertTrue(self.cmd._curr_amps)
        self.do('define P Z u')
        self.do('define J P g')
        self.do('add process e+ e- > J')
        self.assertEqual(len(self.cmd._curr_amps), 2)
        self.do('add process mu+ mu- > P, Z > mu+ mu-')
        self.assertEqual(len(self.cmd._curr_amps), 3)
        self.do('generate e+ e- > Z > e+ e-')
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 1)
        # Test the "or" functionality for propagators
        self.do('define V z|a')
        self.do('generate e+ e- > V > e+ e-')
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 2)
        self.do('generate e+ e- > z | a > e+ e-')
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 2)
        self.do('generate d d~ > u u~ WEIGHTED^2>-1')
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 10) # only 4 have the correct flavor 
        self.do('generate d d~ > u u~ WEIGHTED^2>-2')
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 8) # only 3 have the correct flavor
        self.assertRaises(MadGraph5Error, self.do, 
                                           'generate d d~ > u u~ WEIGHTED^2>-4')
        self.assertRaises(MadGraph5Error, self.do, 'generate a V > e+ e-')
        self.assertRaises(MadGraph5Error, self.do, 'generate e+ e+|e- > e+ e-')
        self.assertRaises(MadGraph5Error, self.do, 'generate e+ e- > V a')
        self.assertRaises(MadGraph5Error, self.do, 'generate e+ e- > e+ e- / V')
        self.do('define V2 = w+ V')
        self.assertEqual(self.cmd._multiparticles['v2'],
                         [[24, 23], [24, 22]])
        
        self.do('generate e+ ve > V2 > e+ ve mu+ mu-')
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 16) # 8 with correct flavor
        
        self.do('generate e+ e- > e+ e- QED=2 [tree=QCD] QCD=0')
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 4)

        self.do('generate e+ e- > e+ e- @0 QCD<=2')
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 4)   
        
        self.do('generate u u~ > d d~ QED>0')
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 8) # 3 correct flavor          
        
        self.assertRaises(diagram_generation.NoDiagramException, self.do, 'generate u u~ > d d~ QED>0 QED^2==0')
        self.do('generate u u~ > d d~ QED==0 QCD>1 QED^2<=4')
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 2) # 1 correct flavor
        
        self.do('generate u u~ > d d~ c c~ QED==2')
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 288) # 28 with correct flavor
        
            
    def test_import_model(self):
        """check that old UFO model are loaded correctly"""

        # Test local models that exercise the FFV Lorentz structure handling
        # (Gamma(3,2,1) and Gamma5(-1,1)*Gamma(3,2,-1) projections)
        dm_pion_path = os.path.join(_file_path, 'input_files', 'DM_pion')
        self.do('import model %s' % dm_pion_path)

        # Test models requiring internet access; skip gracefully if unavailable
        try:
            self.do('''import model DY_SM''')
            self.do('''import model TopEffTh''')
            self.do('''import model uutt_tch_scalar''')
            self.do('''import model uutt_sch_4fermion''')
            self.do('''import model 2HDM''')
        except MadGraph5Error:
            pass  # Models not available locally and no internet connection

    def test_draw(self):
        """ command 'draw' works """

        self.do('set group_subprocesses False')
        self.do('import model sm')
        self.do('generate e+ e- > e+ e-')
        self.do('display diagrams . --no_open')
        self.assertTrue(os.path.exists('./diagrams_0_epem_epem.eps'))
        os.remove('./diagrams_0_epem_epem.eps')
        
        self.do('generate g g > g g')
        self.do('display diagrams . --no_open')
        self.assertTrue(os.path.exists('diagrams_0_gg_gg.eps'))
        os.remove('diagrams_0_gg_gg.eps')
        self.do('set group_subprocesses True')
        
    def test_config(self):
        """check that configuration file is at default value"""
        self.maxDiff=None
        self.cmd.options = {} #reset to None
        config = self.cmd.set_configuration(MG5DIR+'/input/.mg7_configuration_default.txt', final=False)
        config =dict(config)
        del config['stdout_level']
#        for key in config.keys():
#            if key.endswith('_path') and key != 'cluster_temp_path':
#                del config[key]
        expected = {'web_browser': None, 
                    'text_editor': None, 
                    'cluster_queue': None,
                    'nb_core': None,
                    #'pjfry': 'auto',
                    'golem': 'auto',
                    'run_mode': 2,
                    'pythia-pgs_path': './pythia-pgs', 
                    'delphes_path': './Delphes', 
                    'default_unset_couplings': 99,
                    'checkpointing': False,
                    'cluster_type': 'condor', 
                    'cluster_requirement': None,
                    'cluster_vacatetime': '120',
                    'enforce_shared_disk': False,
                    'cluster_status_update': (600, 30),
                    'cluster_temp_path': None, 
                    'fortran_compiler': None, 
                    'cpp_compiler': None,
                    'exrootanalysis_path': './ExRootAnalysis', 
                    'eps_viewer': None, 
                    'automatic_html_opening': True, 
                    'pythia8_path': './HEPTools/pythia8',
                    'madanalysis5_path': './HEPTools/madanalysis5/madanalysis5',
                    'group_subprocesses': 'Auto',
                    'complex_mass_scheme': False,
                    # set-option added by the DDM colour-basis work
                    # (ab161ac8a); this dict has to list every one.
                    'color_basis': 'auto',
                    'gauge': 'unitary',
                    'output_dependencies': 'external',
                    'plain': False,
                    'dmtcp': None,
                    'lhapdf': 'lhapdf-config',
                    'lhapdf_py2': None,
                    'lhapdf_py3': None,  
                    'loop_optimized_output': True,
                    'fastjet': 'fastjet-config',
                    'notification_center':True,
                    'timeout': 60,
                    'ignore_six_quark_processes': False,
                    'include_lepton_initiated_processes': False,
                    'OLP': 'MadLoop',
                    'crash_on_error': False,
                    'auto_update': 7,
                    'cluster_nb_retry': 1,
                    'f2py_compiler':None,
                    'f2py_compiler_py2':None,
                    'f2py_compiler_py3':None,
                    'cluster_retry_wait': 300,
                    'syscalc_path':'./SysCalc',
                    'collier':'./HEPTools/lib',
                    'hepmc_path': './hepmc',
                    'hepmc3_path': './HEPTools/hepmc3',
                    'hwpp_path': './herwigPP',
                    'thepeg_path': './thepeg',
                    #'applgrid': 'applgrid-config',
                    'pineappl': 'pineappl',
                    'cluster_size': 100,
                    'loop_color_flows': False,
                    'cluster_local_path': None,
                    'cvmfs_lhapdf_path':
                              '/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current',
                    'max_npoint_for_channel': 0,
                    'low_mem_multicore_nlo_generation': False,
                    'ninja': './HEPTools/lib',
                    'samurai': None,
                    'max_t_for_channel': 99,
                    'zerowidth_tchannel': True,
                    'auto_convert_model': True,
                    'nlo_mixed_expansion': True,
                    'acknowledged_v3.1_syntax': True,
                    'contur_path': './HEPTools/contur',
                    'rivet_path': './HEPTools/rivet',
                    'yoda_path':'./HEPTools/yoda',
                    'eMELA': 'eMELA-config',
                    'cluster_walltime': None,
                    'use_pigz': None,
                    'checkpointing': False,
                    'cluster_requirement': None,
                    'cluster_vacatetime': '120',
                    'enforce_shared_disk': False,
                    'heptools_install_dir': './HEPTools',
                    'apply_flavor_grouping': True,
                    'merge_same_topologies': True,
                    'merge_quartic_vertices': False,
                    'nb_core_pythia8': None,
                    'nb_core_delphes': None,
                        }

        self.assertEqual(config, expected)
        
        #text_editor = 'vi'
        #if 'EDITOR' in os.environ and os.environ['EDITOR']:
        #    text_editor = os.environ['EDITOR']
        
        #if sys.platform == 'darwin':
        #    self.assertEqual(launch_ext.open_file.web_browser, None)
        #    self.assertEqual(launch_ext.open_file.text_editor, text_editor)
        #    self.assertEqual(launch_ext.open_file.eps_viewer, None)
        #else:
        #    self.assertEqual(launch_ext.open_file.web_browser, 'firefox')
        #    self.assertEqual(launch_ext.open_file.text_editor, text_editor)
        #    self.assertEqual(launch_ext.open_file.eps_viewer, 'gv')
                        
class TestCmdShell2(unittest.TestCase,
                    test_file_writers.CheckFileCreate):
    """Test all command line related to MG_ME"""

    debugging = unittest.debug #set to True to keep the output directory after the test for debugging purpose
    def setUp(self):
        
        self.cmd = Cmd.MasterCmd()
        if not self.debugging:
            self.tmpdir = tempfile.mkdtemp(prefix='amc')
        else:
            if os.path.exists(pjoin(MG5DIR, 'TEST_AMC')):
                shutil.rmtree(pjoin(MG5DIR, 'TEST_AMC'))
            os.mkdir(pjoin(MG5DIR, 'TEST_AMC'))
            self.tmpdir = pjoin(MG5DIR, 'TEST_AMC')
            
        self.out_dir = pjoin(self.tmpdir,'MGProcess')
        
        
    def tearDown(self):
        if not self.debugging and os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
    
    join_path = TestCmdShell1.join_path

    @staticmethod
    def _dens_value_for_key(dm, key):
        """Return the complex value of DensityMatrix entry whose helicity label
        tuple matches ``key``.

        Replaces the old ``dm.matrix[ind][1]`` indexing, which relied on the
        legacy structured-array storage that was removed when DensityMatrix
        was refactored to parallel ``helicities`` / ``values`` arrays.
        """
        import numpy as np
        key_arr = np.asarray(key, dtype=np.int32)
        matches = np.where((dm.helicities == key_arr).all(axis=1))[0]
        if len(matches) == 0:
            raise KeyError('helicity key %s not found in DensityMatrix' % (key,))
        return complex(dm.values[matches[0]])

    def do(self, line, force=False):
        """ exec a line in the cmd under test """
        if force:
            self.cmd.exec_cmd(line, force=force)
        else:
           self.cmd.exec_cmd(line)

    @staticmethod
    def _dens_value_for_key(dm, key):
        """Return the complex value of DensityMatrix entry whose helicity label
        tuple matches ``key``.

        Replaces the old ``dm.matrix[ind][1]`` indexing, which relied on the
        legacy structured-array storage that was removed when DensityMatrix
        was refactored to parallel ``helicities`` / ``values`` arrays.
        """
        import numpy as np
        key_arr = np.asarray(key, dtype=np.int32)
        matches = np.where((dm.helicities == key_arr).all(axis=1))[0]
        if len(matches) == 0:
            raise KeyError('helicity key %s not found in DensityMatrix' % (key,))
        return complex(dm.values[matches[0]])
    
    def test_output_madevent_directory(self):
        """Test outputting a MadEvent directory"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)
        
        self.cmd.do_import('model sm', force=True)
        self.do('set group_subprocesses False')
        self.do('generate e+ e- > e+ e-')
#        self.do('load processes %s' % self.join_path(_pickle_path,'e+e-_e+e-.pkl'))
        self.do('output madevent %s -nojpeg' % self.out_dir)

        self.assertTrue(os.path.exists(self.out_dir))
        self.assertTrue(os.path.exists(pjoin(self.out_dir, 'Cards', 'me5_configuration.txt')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'SubProcesses', 'P0_epem_epem')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'proc_card_mg5.dat')))
        #self.assertFalse(os.path.exists(os.path.join(self.out_dir,
        #                                            'Cards',
        #                                            'ident_card.dat')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'run_card_default.dat')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'Source',
                                                    'maxconfigs.inc')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'maxconfigs.inc')))
        self.assertFalse(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'get_color.f')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'Source',
                                                    'MODEL',
                                                    'get_color.f')))
        if misc.which('gs'):
            self.assertFalse(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'matrix1.jpg')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'madevent.tar.gz')))
        self.do('output madevent %s -f' % self.out_dir)
        self.do('set group_subprocesses True')
        #if misc.which('gs'):
        #    self.assertTrue(os.path.exists(os.path.join(self.out_dir,
        #                                            'SubProcesses',
        #                                            'P0_epem_epem',
        #                                            'matrix1.jpg')))

        # Test the tar file
        os.mkdir(os.path.join(self.out_dir, 'temp'))
        devnull = open(os.devnull,'w')
        subprocess.call(['tar', 'xzf', os.path.join(os.path.pardir,
                                                    "madevent.tar.gz")],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'temp'))

        self.assertTrue(os.path.exists(pjoin(self.out_dir,'temp', 'Cards', 'me5_configuration.txt')))
        # Check that the Source directory compiles
        status = subprocess.call(['make'],
                                stdout=devnull, stderr=devnull, 
                                 cwd=os.path.join(self.out_dir, 'temp', 'Source'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                               'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                               'lib', 'libgeneric.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                               'lib', 'libcernlib.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                               'lib', 'libdsample.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                               'lib', 'libpdf.a')))
        # Check that gensym compiles
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, stderr=devnull, 
                                 cwd=os.path.join(self.out_dir, 'temp', 'SubProcesses',
                                                  'P0_epem_epem'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'gensym')))
        # Check that gensym runs
        proc = subprocess.Popen('./gensym', 
                                 stdout=devnull, stderr=devnull, stdin=subprocess.PIPE,
                                 cwd=os.path.join(self.out_dir, 'temp', 'SubProcesses',
                                                  'P0_epem_epem'), shell=True)
        proc.communicate('100 2 0.1 .false.\n'.encode())
        self.assertEqual(proc.returncode, 0)
        # Check that madevent compiles
        status = subprocess.call(['make', 'madevent'],
                                 stdout=devnull, stderr=devnull, 
                                 cwd=os.path.join(self.out_dir, 'temp', 'SubProcesses',
                                                  'P0_epem_epem'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'madevent')))

    def test_output_mg7_directory(self):
        """mg7 equivalent of test_output_madevent_directory.

        Checks that `output mg7` produces a complete, self-consistent mg7
        (madmatrix/cudacpp) directory for e+ e- > e+ e- -- the top-level layout
        (src/, SubProcesses/, lib/, Cards/, bin/), the mg7 cards and launcher,
        and that the generated subprocess compiles into the expected shared
        libraries (scalar backend).
        """
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.cmd.do_import('model sm', force=True)
        self.do('set group_subprocesses False')
        self.do('generate e+ e- > e+ e-')
        self.do('output mg7 %s' % self.out_dir)

        self.assertTrue(os.path.exists(self.out_dir))
        # mg7 cards and the event-generation launcher
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'param_card.dat')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'run_card.toml')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'bin', 'generate_events')))
        # mg7 C++ model/helas sources
        for f in ['HelAmps_sm.h', 'Parameters.h', 'Parameters.cc', 'makefile']:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir, 'src', f)),
                            '%s missing in mg7 src directory' % f)

        # Discover the subprocess directory (numbering is invocation dependent).
        sub_root = os.path.join(self.out_dir, 'SubProcesses')
        cand = [d for d in os.listdir(sub_root)
                if d.endswith('_epem_epem') and
                os.path.isdir(os.path.join(sub_root, d))]
        self.assertEqual(len(cand), 1,
                         'expected one epem_epem subprocess, got %s' % cand)
        proc_dir = os.path.join(sub_root, cand[0])
        for f in ['CPPProcess.cc', 'CPPProcess.h', 'makefile']:
            self.assertTrue(os.path.isfile(os.path.join(proc_dir, f)),
                            '%s missing in mg7 subprocess directory' % f)

        # Check that the subprocess compiles (scalar C++ backend).
        devnull = open(os.devnull, 'w')
        status = subprocess.call(['make', 'bldnone'],
                                 stdout=devnull, stderr=devnull, cwd=proc_dir)
        self.assertEqual(status, 0)
        libdir = os.path.join(self.out_dir, 'lib')
        libs = os.listdir(libdir) if os.path.isdir(libdir) else []
        self.assertTrue(any(l.startswith('libmadmatrix_common') and
                            l.endswith('.so') for l in libs),
                        'common madmatrix library not built: %s' % libs)
        self.assertTrue(any('epem_epem' in l and l.endswith('.so')
                            for l in libs),
                        'process madmatrix library not built: %s' % libs)

    def test_invalid_operations_for_add(self):
        """Test that errors are raised appropriately for add"""

        self.assertRaises(InvalidCmd,
                          self.do, 'add process')
        self.assertRaises(InvalidCmd,
                          self.do, 'add wrong wrong')

    def test_invalid_operations_for_generate(self):
        """Test that errors are raised appropriately for generate"""

        self.assertRaises(MadGraph5Error,
                          self.do, 'generate')
        self.assertRaises(MadGraph5Error,
                          self.do, 'generate q q > q q')
        self.assertRaises(MadGraph5Error,
                          self.do, 'generate u u~ >')
        self.assertRaises(MadGraph5Error,
                          self.do, 'generate > u u~')
        self.assertRaises(MadGraph5Error,
                          self.do, 'generate a|z > b b~')

    def test_invalid_operations_for_output(self):
        """Test that errors are raised appropriately for output"""

        self.assertRaises(InvalidCmd,
                          self.do, 'output')

    def test_import_model_v4_requires_debug(self):
        """Test that importing a v4 model is now debug-only."""

        self.assertRaises(InvalidCmd, self.do, 'import model_v4 %s' % _v4_model_path)
        self.do('import model_v4 %s --debug' % _v4_model_path)
        self.assertTrue(self.cmd._curr_model)
        self.assertTrue(self.cmd._model_v4_path)

    def test_check_generate_optimize(self):
        """Test that errors are raised appropriately for output"""

        # Invalid since forbiddent by the optimize option
        self.assertRaises(InvalidCmd,
                          self.do, 'generate a > e+ e- --optimize')

        self.assertRaises(InvalidCmd,
                          self.do, 'generate b > t w- --optimize')

        # Invalid since optimize is not allowed for cross-section
        self.assertRaises(InvalidCmd,
                          self.do, 'generate  p p > e+ e- --optimize') 
        
        # check that --optimize filter correctly
        self.do('generate t > all all --optimize')
        self.assertEqual(len(self.cmd._curr_amps), 1)
              
               
    def test_output_standalone_directory(self):
        """Test command 'output' with path"""
        
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('set group_subprocesses False')
        self.do('import model sm')
        self.do('generate e+ e- > e+ e-')
        self.do('output standalone_fortran %s' % self.out_dir)
        self.do('set group_subprocesses True')
        self.assertTrue(os.path.exists(self.out_dir))
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, 'lib', 'libdhelas.a')))
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, 'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'SubProcesses', 'P0_epem_epem')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'Cards', 'proc_card_mg5.dat')))

    def test_output_standalone_fortran_make_all_subprocesses(self):
        """A plain 'make' in the SubProcesses directory of a standalone_fortran
        output (without --prefix) must build the python module of every P
        directory. The per-directory rules used to come before 'all', so the
        default goal was the module of the first P directory only."""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('generate p p > t t~')
        self.do('output standalone_fortran %s' % self.out_dir)

        proc_root = pjoin(self.out_dir, 'SubProcesses')
        dirs = sorted(d for d in os.listdir(proc_root)
                      if d.startswith('P') and os.path.isdir(pjoin(proc_root, d)))
        self.assertGreaterEqual(len(dirs), 2, dirs)

        # a dry run is enough: the per-directory rules call a plain 'make'
        # (not $(MAKE)), so -n prints them without needing f2py
        dry_run = subprocess.run(['make', '-n'], cwd=proc_root,
                                 capture_output=True, text=True)
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        for d in dirs:
            self.assertIn('cd %s;make matrix2py.so' % d, dry_run.stdout)

    def test_custom_propa(self):
        """check that using custom propagator is working"""
        
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        path = os.path.join(MG5DIR, 'tests', 'input_files', 'sm_with_custom_propa')
        self.do('import model %s' % path)
        self.do('generate g g > t t~')
        self.do('output standalone_fortran %s ' % self.out_dir)        
        
        files = ['aloha_file.inc', 'aloha_functions.f','FFV1_0.f', 'FFV1_1.f',
                 'FFV1_2.f', 'makefile', 'VVV1PV2_1.f'] 

        for f in files:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir,
                                                        'Source', 'DHELAS',
                                                        f)), 
                            '%s file is not in aloha directory' % f)

        devnull = open(os.devnull,'w')
        # Check that the Model and Aloha output compile
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'Source'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        # Check that check_sa.f compiles
        subprocess.call(['make', 'check'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_gg_ttx'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses', 'P0_gg_ttx',
                                                    'check')))
        # Check that the output of check is correct 
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_gg_ttx',
                               'check.log')
        p = subprocess.Popen('./check', 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_gg_ttx'), shell=True)
        (log_output, err) = p.communicate()
        log_output = log_output.decode()

        #log_output = open(logfile, 'r').read()
        #misc.sprint(log_output)
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 0.592626100)
        
    def test_ufo_aloha_merged(self):
        """Test the import of models and the export of Helas Routine """

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('generate e+ e- > e+ e-')
        self.do('output standalone_fortran %s ' % self.out_dir)
        # Check that the needed ALOHA subroutines are generated
        files = ['FFV6_3.f', 'aloha_object.mod', 'FFV2_3.f', 'aloha_file.inc', 'makefile', 'FFV6_0.f', 'FFV1P0_3.f', 'FFV2_0.f', 'FFV1_0.f', 'aloha_functions.f']
        for f in files:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir,
                                                        'Source', 'DHELAS',
                                                        f)), 
                            '%s file is not in aloha directory' % f)
        # Check that unwanted ALOHA subroutines are not generated
        notfiles = ['FFV1_1.f', 'FFV1_2.f', 'FFV2_1.f', 'FFV2_2.f',
                    'FFV1_3.f','FFV2P0_3.f','FFV4P0_3.f'
                    'FFV4_1.f', 'FFV4_2.f', 
                    'VVV1_0.f', 'VVV1_1.f', 'VVV1_2.f', 'VVV1_3.f']
        for f in notfiles:
            self.assertFalse(os.path.isfile(os.path.join(self.out_dir,
                                                        'Source', 'DHELAS',
                                                        f)))
        devnull = open(os.devnull,'w')
        # Check that the Model and Aloha output compile
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'Source'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        # Check that check_sa.f compiles
        subprocess.call(['make', 'check'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_epem_epem'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses', 'P0_epem_epem',
                                                    'check')))
        # Check that the output of check is correct 
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_epem_epem',
                               'check.log')
        p = subprocess.Popen('./check', 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_epem_epem'), shell=True)
        (log_output, err) = p.communicate()
        log_output = log_output.decode()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 1.953735e-2)    
    
    def test_ufo_aloha(self):
        """Test the import of models and the export of Helas Routine """

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('set apply_flavor_grouping False')
        self.do('import model sm')
        self.do('generate e+ e- > e+ e-')
        self.do('output standalone_fortran %s ' % self.out_dir)
        # Check that the needed ALOHA subroutines are generated
        files = ['aloha_file.inc', 
                 #'FFS1C1_2.f', 'FFS1_0.f',
                 'FFV1_0.f', 'FFV1P0_3.f',
                 'FFV2_0.f', 'FFV2_3.f',
                 'FFV4_0.f', 'FFV4_3.f',
                 'makefile', 'aloha_functions.f']
        for f in files:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir,
                                                        'Source', 'DHELAS',
                                                        f)), 
                            '%s file is not in aloha directory' % f)
        # Check that unwanted ALOHA subroutines are not generated
        notfiles = ['FFV1_1.f', 'FFV1_2.f', 'FFV2_1.f', 'FFV2_2.f',
                    'FFV1_3.f','FFV2P0_3.f','FFV4P0_3.f'
                    'FFV4_1.f', 'FFV4_2.f', 
                    'VVV1_0.f', 'VVV1_1.f', 'VVV1_2.f', 'VVV1_3.f']
        for f in notfiles:
            self.assertFalse(os.path.isfile(os.path.join(self.out_dir,
                                                        'Source', 'DHELAS',
                                                        f)))
        devnull = open(os.devnull,'w')
        # Check that the Model and Aloha output compile
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'Source'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        # Check that check_sa.f compiles
        subprocess.call(['make', 'check'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_epem_epem'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses', 'P0_epem_epem',
                                                    'check')))
        # Check that the output of check is correct 
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_epem_epem',
                               'check.log')
        p = subprocess.Popen('./check', 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_epem_epem'), shell=True)
        (log_output, err) = p.communicate()
        log_output = log_output.decode()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 1.953735e-2)

    def test_standalone_spin2_loop_smgrav(self):
        """Regression test for spin-2 wavefunction storage in standalone
        Fortran output. ALOHA generates the tensor wavefunction routines
        TXXXXX / VVT2_* with a TYPE(ALOHA2D) (W(16)) parameter, but the
        caller's matrix.f stores every slot in a single TYPE(ALOHA)
        array; when TYPE(ALOHA) holds only W(4) the tensor routine
        overruns the slot and clobbers the caller's stack. This test
        compiles and runs ./check for p p > w+ y in loop_smgrav and
        asserts that the matrix element is the physical reference value
        (~16.95 GeV^0) rather than the order-of-magnitude-larger value
        produced by the stack corruption (~2.5e4 in our local repro)."""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        model_path = pjoin(MG5DIR, 'tests', 'input_files', 'loop_smgrav')
        self.do('import model %s' % model_path)
        self.do('generate p p > w+ y')
        self.do('output standalone_fortran %s ' % self.out_dir)

        # Pin down whichever P*_udx_wpy directory the exporter chose
        # (depends on flavor-grouping defaults).
        sub_root = pjoin(self.out_dir, 'SubProcesses')
        proc_candidates = [d for d in os.listdir(sub_root)
                           if d.endswith('_udx_wpy') and d.startswith('P')]
        self.assertTrue(proc_candidates,
                        'No P*_udx_wpy subprocess directory generated')
        proc_dir = pjoin(sub_root, proc_candidates[0])

        # aloha_functions.f (which contains TXXXXX/IXXXXX/OXXXXX/VXXXXX)
        # and the tensor-vertex routine VVT2_0 must have been generated.
        for f in ['aloha_functions.f', 'VVT2_0.f']:
            self.assertTrue(
                os.path.isfile(pjoin(self.out_dir, 'Source', 'DHELAS', f)),
                '%s missing under Source/DHELAS' % f)

        devnull = open(os.devnull, 'w')
        # Build libdhelas / libmodel
        subprocess.call(['make'], stdout=devnull, stderr=devnull,
                        cwd=pjoin(self.out_dir, 'Source'))
        # Build the standalone check binary
        subprocess.call(['make', 'check'], stdout=devnull, stderr=devnull,
                        cwd=proc_dir)
        self.assertTrue(os.path.isfile(pjoin(proc_dir, 'check')),
                        './check did not build for p p > w+ y in loop_smgrav')

        # Run ./check and parse the matrix-element value
        p = subprocess.Popen('./check', stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, cwd=proc_dir, shell=True)
        (log_output, _err) = p.communicate()
        log_output = log_output.decode()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)
        self.assertTrue(me_groups,
                        'check binary did not print a matrix-element value')
        # Reference value at the default 1 TeV check_sa PS point;
        # tolerance is generous because this is a regression guard, not
        # a precision check.
        self.assertAlmostEqual(float(me_groups.group('value')),
                               16.953243100346082, places=3)

    def test_standalone_wwjj(self):
        """test that standalone cpp is working"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('generate p p  > w+ w- j j  QCD=0')
        self.do('output standalone_fortran %s ' % self.out_dir)

        sub_root = os.path.join(self.out_dir, 'SubProcesses')
        proc_candidates = [d for d in os.listdir(sub_root)
                           if d.startswith('P') and 'qqx' in d.lower()
                           and 'wpwm' in d.lower()]
        self.assertTrue(proc_candidates,
                        'No P*_qqx_wpwmqqx subprocess directory generated')
        proc_dir = os.path.join(sub_root, sorted(proc_candidates)[0])
        logfile = os.path.join(proc_dir, 'check.log')

        # Check that check_sa.cc compiles
        with open(os.devnull, 'w') as devnull:
            subprocess.call(['make'],
                            stdout=devnull, stderr=devnull, 
                            cwd=proc_dir)
            with open(logfile, 'w') as logsock:
                subprocess.call('./check', stdout=logsock,
                                stderr=subprocess.STDOUT,
                                cwd=proc_dir, shell=True)
    
        log_output = open(logfile, 'r').read()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.findall(log_output)

        self.assertTrue(me_groups)
        misc.sprint(me_groups)
        solutions = ['9.7200631392208237E-011', '2.9914807602720421E-010', '2.5945761685827416E-013', '4.0548460794898514E-011', '3.8153429375138002E-011', '8.5331703693981318E-013', '9.6082597110903943E-011', '2.2765536852776125E-010', '3.6856683594456572E-011', '3.5849194817650895E-010', '1.3450540147059369E-011', '3.0986164866850253E-011', '1.1883567237597037E-010', '3.5802494898458324E-013', '5.2484040751611476E-012', '3.0731230518024460E-010', '2.8882822958717279E-011', '1.0295100732619021E-010']
        for val, sol in zip(me_groups, solutions):
            self.assertAlmostEqual(float(val), float(sol), 5)

    def test_standalone_merged_flavor_uq_zuq(self):
        """Regression test for grouped flavor filtering with mixed initial legs.

        Generates ``u q > Z u q QCD=0`` (q = u, d) so the initial state
        mixes a fixed u leg with a merged-quark leg, and asserts that the
        standalone matrix elements for the two surviving flavor
        assignments match the reference values obtained by running each
        flavor as its own explicit process:

            u d > Z u d  ->  1.4704291881825141E-006
            u u > Z u u  ->  3.5590322244693227E-008

        The same checks are repeated with ``--mask=False`` so the
        regression is guarded both with and without the per-flavor
        masking optimisation.  A previous bug in
        HelasMatrixElement.check_flavor_for_all_diagrams left stale
        ``flavortag`` attributes on shared wavefunctions across flavor
        iterations, which caused two diagrams that contribute only when
        all four external quarks are identical (the u u > Z u u case)
        to be wrongly trimmed.  That produced a u u matrix element that
        was 2x the correct value while the u d entry happened to be
        unaffected.
        """

        references = {
            (2, 1, 23, 2, 1): 1.4704291881825141e-06,
            (2, 2, 23, 2, 2): 3.5590322244693227e-08,
        }

        me_re = re.compile(
            r'PDG\s+([-+0-9 ]+?)\n[^\n]*Matrix element\s*=\s*'
            r'(?P<value>[\d\.eE\+-]+)', re.IGNORECASE)
        devnull = open(os.devnull, 'w')

        for mask_flag, label in [('', 'with mask'),
                                 ('--mask=False', 'without mask')]:
            if os.path.isdir(self.out_dir):
                shutil.rmtree(self.out_dir)

            # Re-create the MasterCmd each iteration so the second pass
            # starts from a clean process list.
            self.cmd = Cmd.MasterCmd()
            self.do('import model sm')
            self.do('define q = u d')
            self.do('generate u q > Z u q QCD=0')
            output_cmd = 'output standalone_fortran %s' % self.out_dir
            if mask_flag:
                output_cmd += ' ' + mask_flag
            self.do(output_cmd + ' -f')

            sub_root = pjoin(self.out_dir, 'SubProcesses')
            proc_candidates = [d for d in os.listdir(sub_root)
                               if d.startswith('P') and 'uQ_zuQ' in d]
            self.assertTrue(
                proc_candidates,
                '%s: no P*_uQ_zuQ subprocess directory generated'
                % label)
            proc_dir = pjoin(sub_root, sorted(proc_candidates)[0])

            subprocess.call(['make'], stdout=devnull, stderr=devnull,
                            cwd=pjoin(self.out_dir, 'Source'))
            subprocess.call(['make', 'check'], stdout=devnull,
                            stderr=devnull, cwd=proc_dir)
            self.assertTrue(
                os.path.isfile(pjoin(proc_dir, 'check')),
                '%s: ./check did not build for u q > Z u q' % label)

            p = subprocess.Popen('./check', stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 cwd=proc_dir, shell=True)
            (log_output, _err) = p.communicate()
            log_output = log_output.decode()

            results = {}
            for match in me_re.finditer(log_output):
                pdg = tuple(int(x) for x in match.group(1).split())
                results[pdg] = float(match.group('value'))

            for pdg, expected in references.items():
                self.assertIn(pdg, results,
                              '%s: missing PDG %s in check output'
                              % (label, pdg,))
                self.assertAlmostEqual(
                    results[pdg], expected,
                    delta=abs(expected) * 1e-7,
                    msg=('%s: matrix element for PDG %s is %s, '
                         'expected %s' % (label, pdg,
                                          results[pdg], expected)))

    def test_standalone_flavor_mask(self):
        """Acceptance test for the per-flavor masking optimization.

        Generates p p > j j QCD=0 and, for the q q~ > q q~ subprocess,
        exercises both the Fortran (standalone_fortran) and scalar C++
        (export_cpp.ProcessExporterCPP, driven through its internal API by
        _output_standalone_cpp) backends.
        The check_sa driver is patched to also evaluate two
        non-representative flavors -- s c~ > s c~ (flavor 3 4 3 4) and
        s c~ > c c~ (flavor 3 4 4 4) -- and the matrix-element source is
        patched to print the runtime flavor mask that gates the HELAS
        calls. The test asserts that

          * s c~ > s c~ reproduces the d u~ > d u~ value (PDG 1 -2 1 -2),
          * s c~ > c c~ vanishes,
          * the flavor mask is partial for s c~ > s c~ (a flavor present
            in the runtime flavor table), while s c~ > c c~ is *not* in the
            table: GET_FLAVOR_INDEX returns the 0 sentinel and SMATRIX /
            sigmaKin short-circuit to a zero matrix element before any HELAS
            or mask evaluation, so no MASKDBG line is emitted for it.
        """
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('generate p p > j j QCD=0')
        devnull = open(os.devnull, 'w')

        def find_qqx(sub_root):
            cand = [d for d in os.listdir(sub_root)
                    if d.startswith('P') and 'QQx' in d and d.endswith('QQx')]
            self.assertTrue(cand, 'no q q~ > q q~ subprocess in %s' % sub_root)
            return pjoin(sub_root, sorted(cand)[0])

        def parse_me_by_pdg(text):
            """Map each PDG tuple printed by check_sa to its matrix element."""
            result = {}
            pending = None
            for line in text.splitlines():
                line = line.strip()
                if line.startswith('PDG'):
                    try:
                        pending = tuple(int(x) for x in line.split()[1:])
                    except ValueError:
                        pending = None
                elif line.startswith('Matrix element') and pending is not None:
                    m = re.search(r'=\s*([-+0-9.eE]+)', line)
                    if m:
                        result[pending] = float(m.group(1))
                    pending = None
            return result

        def parse_mask(text, flavor):
            """Return (current, active) mask tuples for the MASKDBG line of
            the given flavor, or None if absent."""
            flavor = tuple(flavor)
            for line in text.splitlines():
                toks = line.split()
                if len(toks) < 2 or toks[0] != 'MASKDBG':
                    continue
                try:
                    nums = [int(x) for x in toks[1:]]
                except ValueError:
                    continue
                if tuple(nums[:len(flavor)]) != flavor:
                    continue
                rest = nums[len(flavor):]
                if not rest or len(rest) % 4 != 0:
                    continue
                nw = len(rest) // 4
                current = tuple(rest[0:nw]) + tuple(rest[2 * nw:3 * nw])
                active = tuple(rest[nw:2 * nw]) + tuple(rest[3 * nw:4 * nw])
                return current, active
            return None

        def run_check(proc_dir):
            log = pjoin(proc_dir, 'check.log')
            with open(log, 'w') as sock:
                subprocess.call('./check', stdout=sock, stderr=subprocess.STDOUT,
                                cwd=proc_dir, shell=True)
            return open(log).read()

        def assert_backend(text, known_flavor, zero_flavor):
            me = parse_me_by_pdg(text)
            for pdg in [(1, -2, 1, -2), (3, -4, 3, -4), (3, -4, 4, -4)]:
                self.assertIn(pdg, me, 'missing PDG %s in check output' % (pdg,))
            reference = me[(1, -2, 1, -2)]
            self.assertGreater(reference, 0.0)
            # s c~ > s c~ must reproduce d u~ > d u~ (flavor universality).
            self.assertAlmostEqual(me[(3, -4, 3, -4)], reference, places=6)
            # s c~ > c c~ is not a valid QCD=0 flavor -> vanishes.
            self.assertAlmostEqual(me[(3, -4, 4, -4)], 0.0, places=10)
            # Known flavor -> partial mask; lookup miss -> all-on fallback.
            #known = parse_mask(text, known_flavor)
            #self.assertIsNotNone(known, 'no MASKDBG line for %s' % (known_flavor,))
            #self.assertNotEqual(known[0], known[1],
            #                    'mask for %s should be partial' % (known_flavor,))
            #miss = parse_mask(text, zero_flavor)
            #self.assertIsNotNone(miss, 'no MASKDBG line for %s' % (zero_flavor,))
            #self.assertEqual(miss[0], miss[1],
            #                 'mask for %s should be all-on' % (zero_flavor,))

        # ---- Fortran standalone -------------------------------------
        self.do('output standalone_fortran %s -f' % self.out_dir)
        proc_dir = find_qqx(pjoin(self.out_dir, 'SubProcesses'))

        check_f = pjoin(proc_dir, 'check_sa.f')
        src = open(check_f).read()
        m = re.search(r'MAXFLAVOR=(\d+)\)', src)
        self.assertTrue(m, 'MAXFLAVOR not found in check_sa.f')
        nflav = int(m.group(1))
        src = src.replace('MAXFLAVOR=%d)' % nflav,
                          'MAXFLAVOR=%d)' % (nflav + 2), 1)
        extra = []
        for offset, (flavor, pdg) in enumerate(
                [((3, 4, 3, 4), (3, -4, 3, -4)),
                 ((3, 4, 4, 4), (3, -4, 4, -4))]):
            col = nflav + 1 + offset
            for leg in range(4):
                extra.append('        FLAVOR(%d,%d) = %d'
                             % (leg + 1, col, flavor[leg]))
                extra.append('        PDG_FOR_FLAVOR(%d,%d) = %d'
                             % (leg + 1, col, pdg[leg]))
        loop_marker = '      do I=1, MAXFLAVOR'
        self.assertIn(loop_marker, src)
        src = src.replace(loop_marker, '\n'.join(extra) + '\n' + loop_marker, 1)
        open(check_f, 'w').write(src)

        matrix_f = pjoin(proc_dir, 'matrix.f')
        src = open(matrix_f).read()
        amp_marker = '      AMP(:) = (0D0, 0D0)'
        self.assertIn(amp_marker, src)
        mask_write = ("      WRITE(*,*) 'MASKDBG', FLAVOR, CURRENT_WF_MASK, "
                      "ACTIVE_WF_MASK, CURRENT_AMP_MASK, ACTIVE_AMP_MASK\n")
        src = src.replace(amp_marker, mask_write + amp_marker, 1)
        open(matrix_f, 'w').write(src)

        subprocess.call(['make'], stdout=devnull, stderr=devnull,
                        cwd=pjoin(self.out_dir, 'Source'))
        subprocess.call(['make', 'check'], stdout=devnull, stderr=devnull,
                        cwd=proc_dir)
        self.assertTrue(os.path.isfile(pjoin(proc_dir, 'check')),
                        './check did not build for the Fortran standalone')
        assert_backend(run_check(proc_dir), (3, 4, 3, 4), (3, 4, 4, 4))

        # ---- C++ standalone -----------------------------------------
        self._output_standalone_cpp(self.out_dir, force=True)
        proc_dir = find_qqx(pjoin(self.out_dir, 'SubProcesses'))

        def extend_flavor_2d_array(text, name, dim_old, dim_new, extra_rows):
            """Append rows to a ``name[dim_old][4]`` C++ initializer (which may
            span several lines) and bump its first dimension to dim_new."""
            m = re.search(r'%s\s*\[%d\]\[4\]' % (re.escape(name), dim_old), text)
            self.assertTrue(m, '%s[%d][4] not found' % (name, dim_old))
            end = text.index('}};', m.start()) + 3
            block = text[m.start():end].replace(
                '[%d][4]' % dim_old, '[%d][4]' % dim_new, 1)
            # block ends in '}};' (last-row close, array close, semicolon); drop
            # the array close + ';' and re-add them after the appended rows.
            block = block[:-2] + ', ' + extra_rows + '};'
            return text[:m.start()] + block + text[end:]

        # The merged standalone_cpp evaluates a flavor by INDEX:
        # process.sigmaKin(iflav) reads CPPProcess's internal flavor_table and
        # the per-flavor bookkeeping arrays sized by nflavors. To exercise the
        # two non-representative flavors we therefore extend that internal table
        # (0-based codes d=0, u=1, s=2, c=3) and nflavors -- not a check_sa-local
        # array (the old `flavor_arr` no longer exists in the generated code):
        #   s c~ > s c~  -> flavor (2,3,2,3) / PDG (3,-4,3,-4)
        #   s c~ > c c~  -> flavor (2,3,3,3) / PDG (3,-4,4,-4)
        check_cpp = pjoin(proc_dir, 'check_sa.cpp')
        src = open(check_cpp).read()
        m = re.search(r'maxflavor\s*=\s*(\d+)', src)
        self.assertTrue(m, 'maxflavor not found in check_sa.cpp')
        nflav = int(m.group(1))
        new_nflav = nflav + 2
        src = re.sub(r'(maxflavor\s*=\s*)\d+',
                     r'\g<1>%d' % new_nflav, src, count=1)
        src = extend_flavor_2d_array(src, 'pdg_arr', nflav, new_nflav,
                                     '{3, -4, 3, -4}, {3, -4, 4, -4}')
        open(check_cpp, 'w').write(src)

        # The bookkeeping arrays (ntry/goodhel/...) and the sigmaKin flavor
        # index are sized by nflavors in the header -> bump it to match.
        cpp_h = pjoin(proc_dir, 'CPPProcess.h')
        src = open(cpp_h).read()
        self.assertIn('nflavors = %d' % nflav, src)
        src = src.replace('nflavors = %d' % nflav,
                          'nflavors = %d' % new_nflav, 1)
        open(cpp_h, 'w').write(src)

        cpp_proc = pjoin(proc_dir, 'CPPProcess.cc')
        src = open(cpp_proc).read()
        self.assertIn('#include "CPPProcess.h"', src)
        src = src.replace('#include "CPPProcess.h"',
                          '#include <iostream>\n#include "CPPProcess.h"', 1)
        # Extend the internal flavor table that sigmaKin indexes.
        src = extend_flavor_2d_array(src, 'flavor_table', nflav, new_nflav,
                                     '{2, 3, 2, 3}, {2, 3, 3, 3}')
        helas_marker = '  ixxxxx(p[perm[0]]'
        self.assertIn(helas_marker, src)
        mask_dump = (
            '  std::cout << "MASKDBG";\n'
            '  for (int mj = 0; mj < nexternal; ++mj)'
            ' std::cout << " " << flavor[mj];\n'
            '  for (int mk = 0; mk < nwords_wf; ++mk)'
            ' std::cout << " " << current_wf_mask[mk];\n'
            '  for (int mk = 0; mk < nwords_wf; ++mk)'
            ' std::cout << " " << active_wf_mask[mk];\n'
            '  for (int mk = 0; mk < nwords_amp; ++mk)'
            ' std::cout << " " << current_amp_mask[mk];\n'
            '  for (int mk = 0; mk < nwords_amp; ++mk)'
            ' std::cout << " " << active_amp_mask[mk];\n'
            '  std::cout << std::endl;\n')
        src = src.replace(helas_marker, mask_dump + helas_marker, 1)
        open(cpp_proc, 'w').write(src)

        subprocess.call(['make'], stdout=devnull, stderr=devnull, cwd=proc_dir)
        self.assertTrue(os.path.isfile(pjoin(proc_dir, 'check')),
                        './check did not build for the C++ standalone')
        # C++ flavor indices are 0-based: 3 4 3 4 -> 2 3 2 3.
        assert_backend(run_check(proc_dir), (2, 3, 2, 3), (2, 3, 3, 3))

    def test_standalone_goodhel_filter(self):
        """The good-helicity filter must not change the matrix element of any
        flavor served by a merged standalone matrix element.

        For each process below we output a Fortran standalone, compile every
        subprocess, and compare the per-flavor matrix element with the
        good-helicity filter OFF against the value with the filter ON, using
        the same ./check driver the ``launch <dir> --timings=N --nb_run=0``
        good-helicity check mode runs:

          * filter OFF -- ``--timings=1  --nb_run=0`` -> ``./check 1000 1 0``:
            a single SMATRIX call per flavor (NTRY below the warm-up
            threshold), so every helicity is summed;
          * filter ON  -- ``--timings=21 --nb_run=0`` -> ``./check 1000 21 0``:
            21 SMATRIX calls per flavor, so the per-flavor good-helicity filter
            is active for the final (printed) value.

        Both runs use the same deterministic RAMBO phase-space point, so the
        flavor-aware filter must reproduce the unfiltered value for every
        flavor. A flavor-blind filter would, after warming up on the first
        flavor, drop a helicity that vanishes for that flavor but contributes
        for another -- giving a different (too small) value, which this test
        would catch.
        """
        processes = [
            # q is not a default multiparticle: define the light quarks so
            # that "q q > q q" exercises the merged-quark matrix element.
            ('define q = u c d s u~ c~ d~ s~', 'q q > q q QCD=0'),
            (None, 'p p > w+ w+ j j QCD=0'),
            (None, 'p p > w+ w- j j QCD=0'),
        ]
        devnull = open(os.devnull, 'w')

        def parse_me_by_pdg(text):
            """Map each PDG tuple printed by ./check to its matrix element."""
            result = {}
            pending = None
            for line in text.splitlines():
                line = line.strip()
                if line.startswith('PDG'):
                    try:
                        pending = tuple(int(x) for x in line.split()[1:])
                    except ValueError:
                        pending = None
                elif line.startswith('Matrix element') and pending is not None:
                    m = re.search(r'=\s*([-+0-9.eEdD]+)', line)
                    if m:
                        result[pending] = float(
                            m.group(1).replace('D', 'E').replace('d', 'e'))
                    pending = None
            return result

        def run_check(proc_dir, nb_try):
            """Run ./check over all flavors with nb_try SMATRIX calls each
            (arg layout: <energy> <nb_try> <unique_flavor=0 -> all>)."""
            cproc = subprocess.Popen(
                ['./check', '1000', str(nb_try), '0'], cwd=proc_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            out = cproc.communicate()[0]
            if isinstance(out, bytes):
                out = out.decode('utf-8', errors='replace')
            return out

        for define_cmd, proc in processes:
            if os.path.isdir(self.out_dir):
                shutil.rmtree(self.out_dir)
            if define_cmd:
                self.do(define_cmd)
            self.do('generate %s' % proc)
            self.do('output standalone_fortran %s -f' % self.out_dir)
            subprocess.call(['make'], stdout=devnull, stderr=devnull,
                            cwd=pjoin(self.out_dir, 'Source'))
            sub_root = pjoin(self.out_dir, 'SubProcesses')
            proc_dirs = [pjoin(sub_root, d) for d in sorted(os.listdir(sub_root))
                         if d.startswith('P')
                         and os.path.isdir(pjoin(sub_root, d))]
            self.assertTrue(proc_dirs, 'no subprocess generated for %s' % proc)
            saw_nonzero = False
            for proc_dir in proc_dirs:
                tag = '%s [%s]' % (proc, os.path.basename(proc_dir))
                subprocess.call(['make', 'check'], stdout=devnull,
                                stderr=devnull, cwd=proc_dir)
                self.assertTrue(os.path.isfile(pjoin(proc_dir, 'check')),
                                './check did not build for %s' % tag)
                unfiltered = parse_me_by_pdg(run_check(proc_dir, 1))
                filtered = parse_me_by_pdg(run_check(proc_dir, 21))
                self.assertTrue(unfiltered,
                                'no matrix element parsed for %s' % tag)
                self.assertEqual(set(unfiltered), set(filtered),
                                 'the set of flavors changed by filtering '
                                 'for %s' % tag)
                for pdg, ref in unfiltered.items():
                    self.assertAlmostEqual(
                        filtered[pdg], ref,
                        delta=max(1e-6 * abs(ref), 1e-10),
                        msg='good-helicity filtering changed |M|^2 for PDG %s '
                            'of %s: %r (filter off) vs %r (filter on)'
                            % (pdg, tag, ref, filtered[pdg]))
                if any(v != 0.0 for v in unfiltered.values()):
                    saw_nonzero = True
            self.assertTrue(saw_nonzero,
                            'all matrix elements vanished for %s' % proc)

    def test_madmatrix_goodhel_filter(self):
        """The standalone (madmatrix/cudacpp) good-helicity filter must reproduce the
        per-flavor matrix element of every flavor served by a merged matrix
        element.

        the standalone (madmatrix) export computes a single global good-helicity list once, as the
        union over all flavor combinations (see sigmaKin_getGoodHel). A
        flavor-blind filter -- one that seeds the good helicities from only the
        flavor(s) of the first sampled events -- would drop a helicity that
        vanishes for the seeding flavor but contributes for another merged
        flavor, giving a too-small |M|^2 for that other flavor.

        We compare the standalone (madmatrix) per-flavor values (check_sa.exe 'matrix'
        mode) against the Fortran standalone ones, which
        test_standalone_goodhel_filter independently validates as
        filter-invariant. ``u u~ > j j QCD=0`` is a single merged matrix element
        whose flavors have *different* diagram content: ``u u~ > u u~`` has both
        s- and t-channel photon/Z exchange, while ``u u~ > d d~`` (the first
        flavor, which seeds the good helicities) is s-channel only. A
        flavor-blind filter warmed up on ``u u~ > d d~`` drops the t-channel
        helicities, making ``u u~ > u u~`` too small -- which this comparison
        catches. The massless final state makes both drivers evaluate the same
        fixed-energy RAMBO point.
        """
        devnull = open(os.devnull, 'w')
        energy = '1000'
        me_re = re.compile(r'Matrix element\s*=\s*([\d.eE+-]+)\s*GeV',
                           re.IGNORECASE)

        def get_values(output_format, check_exe, build_source=False):
            if os.path.isdir(self.out_dir):
                shutil.rmtree(self.out_dir)
            self.do('output %s %s -f' % (output_format, self.out_dir))
            if build_source:
                subprocess.call(['make'], stdout=devnull, stderr=devnull,
                                cwd=pjoin(self.out_dir, 'Source'))
            proc_root = pjoin(self.out_dir, 'SubProcesses')
            dirs = sorted(d for d in os.listdir(proc_root)
                          if d.startswith('P') and
                          os.path.isdir(pjoin(proc_root, d)))
            self.assertTrue(dirs, 'no subprocess for %s' % output_format)
            values = []
            for d in dirs:
                proc_dir = pjoin(proc_root, d)
                # standalone_fortran uses 'make check' + ./check; standalone ships a
                # UMAMI check_sa.exe whose 'matrix' mode == the Fortran driver.
                target = ['make', 'check'] if output_format == 'standalone_fortran' \
                    else ['make']
                subprocess.call(target, stdout=devnull, stderr=devnull,
                                cwd=proc_dir)
                log = pjoin(proc_dir, 'check.log')
                subprocess.call('%s %s' % (check_exe, energy),
                                stdout=open(log, 'w'), stderr=subprocess.STDOUT,
                                cwd=proc_dir, shell=True)
                found = me_re.findall(open(log).read())
                self.assertTrue(found, '%s produced no matrix element (see %s)'
                                % (output_format, log))
                values.extend(float(v) for v in found)
            return values

        self.do('import model sm')
        self.do('generate u u~ > j j QCD=0')
        mg7 = get_values('standalone', './check_sa.exe')
        standalone = get_values('standalone_fortran', './check', build_source=True)
        self.assertTrue(any(v != 0.0 for v in standalone),
                        'all matrix elements vanished for u u~ > j j')
        self._assert_me_lists_close(mg7, standalone, atol=1e-7)

    def test_standalone_split_orders_interference(self):
        """standalone (madmatrix) must return the squared-order contribution asked for.

        The madmatrix jamps carry an amplitude-order index and the color sum
        pairs them, so a '^2' constraint that keeps only some of the squared
        orders gets that contribution and not the total. The case that matters
        is an interference term, which cannot be reached by dropping diagrams
        at generation: ``u u~ > u u~ QED^2==2`` keeps every diagram and wants
        the QCD-EW cross term alone, -5.5828746494657265e-02 from the Fortran
        split-order driver, where a backend with no mask returns the whole
        +2.7756451199752394.

        The three components are checked to sum back to the unconstrained
        total *as computed by this same backend*. That comparison is the one
        that pins the pair loop: it uses one set of momenta and one set of
        parameters, so it is sensitive to the interference algebra alone --
        in particular to the fact that the color contraction keeps its
        doubled triangle while the loop over amplitude-order pairs must run
        over all ordered pairs, since for two different jamp vectors a pair
        and its transpose are not each other's conjugate.
        """
        devnull = open(os.devnull, 'w')
        me_re = re.compile(r'Matrix element\s*=\s*([\d.eE+-]+)\s*GeV',
                           re.IGNORECASE)

        def value(constraint):
            if os.path.isdir(self.out_dir):
                shutil.rmtree(self.out_dir)
            self.do('generate u u~ > u u~ %s' % constraint)
            self.do('output standalone %s -f' % self.out_dir)
            proc_root = pjoin(self.out_dir, 'SubProcesses')
            dirs = [d for d in os.listdir(proc_root)
                    if d.startswith('P') and os.path.isdir(pjoin(proc_root, d))]
            self.assertTrue(dirs, 'no subprocess for %s' % constraint)
            proc_dir = pjoin(proc_root, dirs[0])
            # FPTYPE=d: mixed precision hides and fakes differences here
            self.assertEqual(0, subprocess.call(['make', 'FPTYPE=d'],
                                                stdout=devnull, stderr=devnull,
                                                cwd=proc_dir),
                             'standalone %s did not build' % constraint)
            log = pjoin(proc_dir, 'check.log')
            subprocess.call('./check_sa.exe 1000', shell=True, cwd=proc_dir,
                            stdout=open(log, 'w'), stderr=subprocess.STDOUT)
            found = me_re.findall(open(log).read())
            self.assertTrue(found, 'no matrix element (see %s)' % log)
            return float(found[0])

        self.do('import model sm')
        interference = value('QED^2==2')
        # The Fortran split-order component, to the tolerance this backend is
        # compared at elsewhere (the EW couplings differ in the last digits)
        self.assertAlmostEqual(interference, -5.5828746494657265e-02, delta=1e-7)
        # ... and emphatically not the unmasked total
        self.assertLess(abs(interference), 1.0)

        components = [value('QED^2==0'), interference, value('QED^2==4')]
        total = value('QED^2<=4')
        self.assertAlmostEqual(sum(components), total,
                               delta=1e-12 * abs(total),
                               msg='the squared-order components do not add up '
                                   'to the total this backend computes')

    def test_standalone_interference_helicity_choice(self):
        """the helicity drawn for an interference |M|^2 follows madevent.

        A helicity can contribute negatively to an interference, so choosing it
        on the signed running sum over the helicities (the positive case)
        picks it with no meaningful probability. madmatrix follows madevent
        instead when the caller draws the helicity (umami's random helicity
        input, i.e. event generation): helicity i with probability
        |T_i| / sum_j |T_j|, and the event gets sign(T_i) * sum_j |T_j|. That
        averages back to the helicity sum sum_j T_j, which is what is returned
        when no random number is given.

        ``u u~ > t t~ g QED^2==2`` (the QCD-EW interference behind the tt~
        charge asymmetry) has helicities of both signs at almost every RAMBO
        point. Each point is evaluated on a uniform grid of helicity random
        numbers, through the umami interface of the library:

        * |M|^2 is the same, sum_j |T_j|, for every draw;
        * every helicity always comes with the same sign;
        * the average over the grid is the helicity sum, to the grid step;
        * the points with mixed signs give both signs, with
          sum_j |T_j| > |sum_j T_j|: the signed choice gives a constant |M|^2,
          and would fail here.
        """
        import ctypes
        import glob
        import struct

        self.do('import model sm')
        self.do('generate u u~ > t t~ g QED^2==2')
        self.do('output standalone %s -f' % self.out_dir)
        proc_root = pjoin(self.out_dir, 'SubProcesses')
        proc_dir = [pjoin(proc_root, d) for d in os.listdir(proc_root)
                    if d.startswith('P') and os.path.isdir(pjoin(proc_root, d))][0]
        devnull = open(os.devnull, 'w')
        self.assertEqual(0, subprocess.call(['make', 'FPTYPE=d'], stdout=devnull,
                                            stderr=devnull, cwd=proc_dir))
        npar, npoints, ngrid = 5, 16, 4000
        momfile = pjoin(proc_dir, 'momenta.bin')
        subprocess.call(['./check_sa.exe', 'perf', '--dump-momenta', momfile,
                         '1', str(npoints), '1'], cwd=proc_dir,
                        stdout=devnull, stderr=devnull)
        raw = open(momfile, 'rb').read()
        momenta = struct.unpack('%dd' % (len(raw) // 8), raw)
        self.assertEqual(len(momenta), npoints * npar * 4)

        # The process library finds its common library through its own
        # loader-relative rpath: no RTLD_GLOBAL preload, which would leak this
        # process's symbols into every library loaded later in this (shared)
        # test process.
        libdir = pjoin(self.out_dir, 'lib')
        # libmadmatrix_<model>_<Pdir>_<backend>_<fptype>.so
        lib = ctypes.CDLL([path for path in glob.glob(pjoin(libdir, 'libmadmatrix_*_P*.so'))
                           if 'libmadmatrix_common' not in path][0])
        handle = ctypes.c_void_p()
        card = pjoin(self.out_dir, 'Cards', 'param_card.dat').encode()
        self.assertEqual(0, lib.umami_initialize(ctypes.byref(handle), card))
        self.addCleanup(lib.umami_free, handle)
        # umami.h: UMAMI_IN_MOMENTA = 0, UMAMI_IN_RANDOM_HELICITY = 4,
        # UMAMI_OUT_MATRIX_ELEMENT = 0, UMAMI_OUT_HELICITY_INDEX = 3
        IN_MOMENTA, IN_RANDOM_HELICITY = 0, 4
        OUT_MATRIX_ELEMENT, OUT_HELICITY_INDEX = 0, 3

        def evaluate(point, draw):
            count = ngrid if draw else 1
            mom = (ctypes.c_double * (4 * npar * count))()
            for ipart in range(npar):  # momenta[count * (npar * mu + ipart) + ievt]
                for mu in range(4):
                    start = count * (npar * mu + ipart)
                    mom[start:start + count] = [point[ipart * 4 + mu]] * count
            me = (ctypes.c_double * count)()
            hel = (ctypes.c_int * count)()
            rnd = (ctypes.c_double * count)(
                *[(k + 0.5) / count for k in range(count)])
            in_keys = [IN_MOMENTA] + ([IN_RANDOM_HELICITY] if draw else [])
            inputs = [mom] + ([rnd] if draw else [])
            out_keys = [OUT_MATRIX_ELEMENT] + ([OUT_HELICITY_INDEX] if draw else [])
            outputs = [me] + ([hel] if draw else [])
            status = lib.umami_matrix_element(
                handle, ctypes.c_size_t(count), ctypes.c_size_t(count),
                ctypes.c_size_t(0), ctypes.c_size_t(len(in_keys)),
                (ctypes.c_int * len(in_keys))(*in_keys),
                (ctypes.c_void_p * len(inputs))(
                    *[ctypes.addressof(a) for a in inputs]),
                ctypes.c_size_t(len(out_keys)),
                (ctypes.c_int * len(out_keys))(*out_keys),
                (ctypes.c_void_p * len(outputs))(
                    *[ctypes.addressof(a) for a in outputs]))
            self.assertEqual(status, 0)
            return list(me), list(hel)

        mixed = 0
        for ipoint in range(npoints):
            point = momenta[ipoint * npar * 4:(ipoint + 1) * npar * 4]
            (helicity_sum,), _ = evaluate(point, False)
            me, hel = evaluate(point, True)
            abs_sum = abs(me[0])
            signs = {}
            for value, ihel in zip(me, hel):
                self.assertAlmostEqual(abs(value), abs_sum, delta=1e-12 * abs_sum)
                self.assertEqual(signs.setdefault(ihel, value > 0), value > 0,
                                 'helicity %d came with both signs' % ihel)
            # the grid step bounds the error of each helicity's interval
            self.assertAlmostEqual(sum(me) / ngrid, helicity_sum,
                                   delta=len(signs) * abs_sum / ngrid)
            if len(set(signs.values())) == 2:
                mixed += 1
                self.assertGreater(abs_sum, abs(helicity_sum) * (1 + 1e-6))
        # 15 of the 16 points when this test was written
        self.assertGreater(mixed, npoints // 2)

        # run card [generation] interference_helicity = "summed": the instance
        # keeps the helicity sum as the |M|^2 of every draw (mg7 then writes the
        # helicities as 9)
        lib.umami_set_parameter.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                            ctypes.c_double, ctypes.c_double]
        self.assertEqual(0, lib.umami_set_parameter(
            handle, b'interference_helicity_summed', 1., 0.))
        for ipoint in range(4):
            point = momenta[ipoint * npar * 4:(ipoint + 1) * npar * 4]
            (helicity_sum,), _ = evaluate(point, False)
            me, _hel = evaluate(point, True)
            for value in me:
                self.assertAlmostEqual(value, helicity_sum,
                                       delta=1e-12 * abs(helicity_sum))

    def test_standalone_umami_first_call(self):
        """the helicities kept, and the |M|^2, do not depend on the first umami call.

        The good helicities are chosen once per library: a helicity is kept if
        its |M|^2 is not 0 for one of the first events. They were chosen at
        the first umami_matrix_element call, so that
        * a first call of special points dropped, for the rest of the run, the
          helicities that vanish there: after one event in the exact forward
          direction, u u~ > e+ e- came out wrong by factors 5 to 100;
        * a first call shorter than a SIMD page was padded with zero events,
          which evaluate to nan, and every helicity was kept (twice the work
          for a q q~ initial state).
        They are now chosen at initialization, on RAMBO points, and the
        padding events of a batch are copies of real ones.

        p p > e+ e- has two flavor indices (d and u type) and only 4 of its 16
        helicities do not vanish. For the scalar backend and simd_128 in double
        and in mixed precision (one and two SIMD pages at a time), with a first
        call of one generic event, of 16 events, or of one forward event:
        * the same helicities are kept, fewer than 16
          (madmatrix_good_helicity_count);
        * the same |M|^2 come out afterwards, and a batch whose flavors
          alternate, leaving partial pages, gives the |M|^2 of single-event
          calls, event by event.

        The helicities are chosen once per loaded library, so each first call
        runs in its own process.
        """
        self._check_umami_first_call([('scalar', 'd', 1e-12),
                                      ('simd_128', 'd', 1e-12),
                                      ('simd_128', 'm', 1e-6)])

    def test_standalone_umami_first_call_gpu(self):
        """test_standalone_umami_first_call on GPUs (BACKEND=cuda, hip).

        There, umami_matrix_element takes device buffers, and umami_initialize
        copies its RAMBO points over to choose the good helicities. Skipped
        without nvcc and hipcc on the PATH, or without a device.
        """
        builds = [(backend, fptype, rtol)
                  for backend, compiler in (('cuda', 'nvcc'), ('hip', 'hipcc'))
                  if misc.which(compiler)
                  for fptype, rtol in (('d', 1e-12), ('m', 1e-6))]
        if not builds:
            raise unittest.SkipTest('neither nvcc nor hipcc on the PATH')
        self._check_umami_first_call(builds)

    def _check_umami_first_call(self, builds):
        """test_standalone_umami_first_call for these (backend, fptype, rtol)"""
        import glob
        import json
        import textwrap

        self.do('import model sm')
        self.do('generate p p > e+ e-')
        self.do('output standalone %s -f' % self.out_dir)
        proc_root = pjoin(self.out_dir, 'SubProcesses')
        proc_dir = [pjoin(proc_root, d) for d in os.listdir(proc_root)
                    if d.startswith('P') and os.path.isdir(pjoin(proc_root, d))][0]
        card = pjoin(self.out_dir, 'Cards', 'param_card.dat')
        helper = pjoin(self.out_dir, 'umami_first_call.py')
        with open(helper, 'w') as fsock:
            fsock.write(textwrap.dedent('''\
                import ctypes, ctypes.util, json, math, subprocess, sys
                lib = ctypes.CDLL(sys.argv[1])
                gpu = sys.argv[4] if sys.argv[4] in ('cuda', 'hip') else None
                if gpu:
                    # umami takes device buffers: the CUDA/HIP runtime of the library, or
                    # (cudart is linked statically by default) one sharing its device context
                    name = 'cudart' if gpu == 'cuda' else 'amdhip64'
                    ldd = subprocess.run(['ldd', sys.argv[1]], capture_output=True, text=True).stdout
                    paths = [line.split('=>')[1].split()[0] for line in ldd.splitlines()
                             if 'lib%s' % name in line and '=>' in line]
                    runtime = ctypes.CDLL(paths[0] if paths else
                                          ctypes.util.find_library(name) or 'lib%s.so' % name)
                    ndevice = ctypes.c_int(0)
                    if runtime[gpu + 'GetDeviceCount'](ctypes.byref(ndevice)) != 0 or not ndevice.value:
                        print(json.dumps({'skip': 'no %s device' % gpu}))
                        sys.exit(0)

                def buffer(host):  # what umami reads or writes: host, or a device copy
                    if not gpu:
                        return ctypes.addressof(host)
                    pointer = ctypes.c_void_p()
                    size = ctypes.c_size_t(ctypes.sizeof(host))
                    assert runtime[gpu + 'Malloc'](ctypes.byref(pointer), size) == 0
                    # cudaMemcpyHostToDevice == hipMemcpyHostToDevice == 1
                    assert runtime[gpu + 'Memcpy'](pointer, ctypes.c_void_p(ctypes.addressof(host)), size, 1) == 0
                    return pointer.value

                def release(pointers, output, host):  # output back into host, the device copies freed
                    if gpu:
                        assert runtime[gpu + 'DeviceSynchronize']() == 0
                        # cudaMemcpyDeviceToHost == hipMemcpyDeviceToHost == 2
                        assert runtime[gpu + 'Memcpy'](ctypes.c_void_p(ctypes.addressof(host)), ctypes.c_void_p(output),
                                                       ctypes.c_size_t(ctypes.sizeof(host)), 2) == 0
                        for pointer in pointers:
                            assert runtime[gpu + 'Free'](ctypes.c_void_p(pointer)) == 0

                handle = ctypes.c_void_p()
                assert lib.umami_initialize(ctypes.byref(handle), sys.argv[2].encode()) == 0
                npar = 4

                def point(k):  # q q~ > e+ e- at sqrt(s) = 1 TeV: k-th direction, or forward
                    e = 500.
                    ct, phi = (1., 0.) if k == 'forward' else (math.cos(0.3 + 0.37 * k), 0.7 * k)
                    st = math.sqrt(1 - ct * ct)
                    p3 = [e * st * math.cos(phi), e * st * math.sin(phi), e * ct]
                    return [[e, 0, 0, e], [e, 0, 0, -e], [e] + p3, [e] + [-x for x in p3]]

                def evaluate(ks, flavors):
                    n = len(ks)
                    mom = (ctypes.c_double * (4 * npar * n))()
                    for i, k in enumerate(ks):  # momenta[n * (npar * mu + ipart) + i]
                        for ipart, p in enumerate(point(k)):
                            for mu in range(4):
                                mom[n * (npar * mu + ipart) + i] = p[mu]
                    flav = (ctypes.c_uint * n)(*flavors)
                    me = (ctypes.c_double * n)()
                    pointers = [buffer(mom), buffer(flav), buffer(me)]
                    # umami.h: UMAMI_IN_MOMENTA = 0, UMAMI_IN_FLAVOR_INDEX = 2,
                    # UMAMI_OUT_MATRIX_ELEMENT = 0
                    status = lib.umami_matrix_element(
                        handle, ctypes.c_size_t(n), ctypes.c_size_t(n), ctypes.c_size_t(0),
                        ctypes.c_size_t(2), (ctypes.c_int * 2)(0, 2),
                        (ctypes.c_void_p * 2)(pointers[0], pointers[1]),
                        ctypes.c_size_t(1), (ctypes.c_int * 1)(0),
                        (ctypes.c_void_p * 1)(pointers[2]))
                    assert status == 0, status
                    release(pointers, pointers[2], me)
                    return list(me)

                first = {'single': [0], 'batch': list(range(16)), 'forward': ['forward']}[sys.argv[3]]
                evaluate(first, [k % 2 if k != 'forward' else 1 for k in first])
                flavors = [0, 1, 1, 0, 1, 0, 0]
                print(json.dumps({
                    'good': lib.madmatrix_good_helicity_count(),
                    'batch': evaluate(range(7), flavors),
                    'single': [evaluate([k], [f])[0] for k, f in enumerate(flavors)]}))
                '''))

        devnull = open(os.devnull, 'w')
        for backend, fptype, rtol in builds:
            build = 'BACKEND=%s FPTYPE=%s' % (backend, fptype)
            subprocess.call(['make', 'cleanall'], stdout=devnull,
                            stderr=devnull, cwd=proc_dir)
            # the end of the build log in the message (the GPU CI has nothing else)
            make = subprocess.run(['make', 'BACKEND=' + backend, 'FPTYPE=' + fptype],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  cwd=proc_dir)
            self.assertEqual(0, make.returncode, 'the %s library did not build:\n%s'
                             % (build, make.stdout.decode(errors='replace')[-3000:]))
            lib = glob.glob(pjoin(self.out_dir, 'lib', 'libmadmatrix_*_P*_%s_%s.so'
                                  % (backend, fptype)))[0]
            runs = {}
            for first in ('single', 'batch', 'forward'):
                out = subprocess.check_output(
                    [sys.executable, helper, lib, card, first, backend])
                runs[first] = json.loads(out.decode().strip().splitlines()[-1])
                if 'skip' in runs[first]:
                    raise unittest.SkipTest(runs[first]['skip'])
            reference = runs['single']
            self.assertGreater(reference['good'], 0, build)
            self.assertLess(reference['good'], 16,
                            '%s: no helicity was filtered out' % build)
            for first, run in runs.items():
                self.assertEqual(run['good'], reference['good'],
                                 '%s: a first call of %s events keeps other '
                                 'helicities' % (build, first))
                for batch, single, ref in zip(run['batch'], run['single'],
                                              reference['single']):
                    self.assertGreater(single, 0, build)
                    self.assertAlmostEqual(single, ref, delta=rtol * ref,
                                           msg='%s: the |M|^2 depend on the first '
                                               'call (%s)' % (build, first))
                    self.assertAlmostEqual(batch, single, delta=rtol * single,
                                           msg='%s: batch and single-event '
                                               '|M|^2 differ' % build)

    def test_standalone_cpp(self):
        """test that the scalar C++ standalone exporter is working

        `output standalone_cpp` is no longer a user-facing format, so the C++
        arm drives export_cpp.ProcessExporterCPP through its internal API
        (_output_standalone_cpp) instead of the command.
        """

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model MSSM_SLHA2-full')
        self.do('generate g g > go go QED=2')
        self._output_standalone_cpp(self.out_dir)
        devnull = open(os.devnull,'w')

        # Locate the subprocess directory: the merge shortened the C++
        # directory name (e.g. P0_Sigma_MSSM_SLHA2_full_gg_gogo -> P1_gg_gogo),
        # so discover it rather than hard-coding the number/prefix.
        proc_root = os.path.join(self.out_dir, 'SubProcesses')
        candidates = [d for d in os.listdir(proc_root)
                      if d.endswith('_gg_gogo') and
                      os.path.isdir(os.path.join(proc_root, d))]
        self.assertEqual(len(candidates), 1,
                         'expected exactly one gg_gogo subprocess, got %s'
                         % candidates)
        proc_dir = os.path.join(proc_root, candidates[0])
        logfile = os.path.join(proc_dir, 'check.log')
        # Check that check_sa.cc compiles
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull,
                        cwd=proc_dir)

        subprocess.call('./check',
                        stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                        cwd=proc_dir, shell=True)

        log_output = open(logfile, 'r').read()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)

        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 6.4739191,5)

        # Cross-check standalone (madmatrix) against standalone_fortran for
        # this massive BSM process. The Fortran ./check auto-bumps the CM energy
        # to 2*total_mass for the heavy gluinos, but check_sa.exe does not, so
        # evaluate BOTH at the same explicit above-threshold energy.
        energy = '5000'

        shutil.rmtree(self.out_dir)
        self.do('output standalone_fortran %s -f' % self.out_dir)
        subprocess.call(['make'], stdout=devnull, stderr=devnull,
                        cwd=os.path.join(self.out_dir, 'Source'))
        f_root = os.path.join(self.out_dir, 'SubProcesses')
        f_cand = [d for d in os.listdir(f_root)
                  if d.endswith('_gg_gogo') and
                  os.path.isdir(os.path.join(f_root, d))]
        self.assertEqual(len(f_cand), 1,
                         'expected one gg_gogo Fortran subprocess, got %s'
                         % f_cand)
        f_dir = os.path.join(f_root, f_cand[0])
        subprocess.call(['make', 'check'], stdout=devnull, stderr=devnull,
                        cwd=f_dir)
        # Same hard-coded reference at the auto-bumped default energy.
        f_log = os.path.join(f_dir, 'check.log')
        subprocess.call('./check',
                        stdout=open(f_log, 'w'), stderr=subprocess.STDOUT,
                        cwd=f_dir, shell=True)
        f_default = me_re.search(open(f_log).read())
        self.assertTrue(f_default,
                        'standalone_fortran produced no matrix element')
        self.assertAlmostEqual(float(f_default.group('value')), 6.4739191, 5)
        # Reference value at the explicit above-threshold energy.
        f_e_log = os.path.join(f_dir, 'check_e.log')
        subprocess.call('./check %s' % energy,
                        stdout=open(f_e_log, 'w'), stderr=subprocess.STDOUT,
                        cwd=f_dir, shell=True)
        f_me = me_re.search(open(f_e_log).read())
        self.assertTrue(f_me)

        shutil.rmtree(self.out_dir)
        self.do('output standalone %s -f' % self.out_dir)
        mg7_root = os.path.join(self.out_dir, 'SubProcesses')
        mg7_cand = [d for d in os.listdir(mg7_root)
                    if d.endswith('_gg_gogo') and
                    os.path.isdir(os.path.join(mg7_root, d))]
        self.assertEqual(len(mg7_cand), 1,
                         'expected one gg_gogo mg7 subprocess, got %s' % mg7_cand)
        mg7_dir = os.path.join(mg7_root, mg7_cand[0])
        subprocess.call(['make'], stdout=devnull, stderr=devnull, cwd=mg7_dir)
        mg7_log = os.path.join(mg7_dir, 'check.log')
        subprocess.call('./check_sa.exe %s' % energy,
                        stdout=open(mg7_log, 'w'), stderr=subprocess.STDOUT,
                        cwd=mg7_dir, shell=True)
        mg7_me = me_re.search(open(mg7_log).read())
        self.assertTrue(mg7_me, 'standalone (madmatrix) produced no matrix element')
        self._assert_me_lists_close([float(mg7_me.group('value'))],
                                    [float(f_me.group('value'))])

    
    def test_standalone_cpp_output_consistency(self):
        """test that standalone cpp is working"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        #step 0 cpp output
        self.do('generate p p > t t~, t > b mu+ vm, t~ > b~ mu- vm~')
        self._output_standalone_cpp(self.out_dir)
        devnull = open(os.devnull,'w')

        # Discover the subprocess directories: the merge shortened the C++
        # directory names (e.g. P0_Sigma_sm_gg_bmupvmbxmumvmx ->
        # P1_gg_bmupvmbxmumvmx), so list them rather than hard-coding.
        def get_values():
            proc_root = os.path.join(self.out_dir, 'SubProcesses')
            directories = sorted(d for d in os.listdir(proc_root)
                                 if d.startswith('P') and
                                 os.path.isdir(os.path.join(proc_root, d)))
            self.assertEqual(len(directories), 2,
                             'expected 2 subprocesses, got %s' % directories)
            values = []
            for oneproc in directories:
                logfile = os.path.join(self.out_dir,'SubProcesses', oneproc,
                                       'check.log')
                # Check that check_sa.cc compiles
                subprocess.call(['make'],
                                stdout=devnull, stderr=devnull, 
                                cwd=os.path.join(self.out_dir, 'SubProcesses', oneproc))
                
                subprocess.call('./check', 
                                stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                                cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                 oneproc), shell=True)
            
                log_output = open(logfile, 'r').read()
                me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                                   re.IGNORECASE)
                me_groups = me_re.search(log_output)
                self.assertTrue(me_groups)
                values.append(float(me_groups.group('value')))
            return values
        original = get_values()
        #step 1 standalone output
        shutil.rmtree(self.out_dir)
        self.do('output standalone_fortran %s -f' % self.out_dir)
        self._output_standalone_cpp(self.out_dir, force=True)
        new = get_values()
        
        for i,_ in enumerate(original):
            self.assertEqual(original[i], new[i])

    def _output_standalone_cpp(self, out_dir, force=False):
        """Write a scalar C++ standalone output for the processes currently
        held by the interface, driving export_cpp.ProcessExporterCPP through
        its internal API.

        `output standalone_cpp` is no longer a user-facing format, but the
        exporter class itself is very much alive: it is the base class of the
        madmatrix (`standalone`) export, and `check language` drives it exactly
        this way (see madgraph/various/process_checks.py and
        tests/unit_tests/various/test_process_checks.py).  Going through the API
        keeps the scalar-C++ coverage of these tests without the command.
        """
        import madgraph.iolibs.export_cpp as export_cpp
        import madgraph.iolibs.helas_call_writers as helas_call_writers
        import madgraph.core.helas_objects as helas_objects

        cmd = self.cmd
        model = cmd._curr_model

        if force and os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

        opt = dict(cmd.options)
        opt['output_options'] = {}
        opt.update({'sa_symmetry': False, 'export_format': 'standalone_cpp',
                    'mp': False, 'v5_model': True})
        exporter = export_cpp.ProcessExporterCPP(out_dir, opt)

        # Reuse the helas objects the interface already built, exactly like
        # do_output does: building a second HelasMultiProcess from the same
        # _curr_amps does NOT give the same matrix elements back (decay chains
        # in particular are lost), so the cache is what makes repeated exports
        # of one `generate` consistent.
        multi_me = cmd._curr_matrix_elements
        if not isinstance(multi_me, helas_objects.HelasMultiProcess) or \
                not multi_me.get_matrix_elements():
            # do_output sets this global from the exporter before building the
            # helas objects; mirror it.
            helas_objects.HelasMatrixElement.enumerate_all_flavors = \
                not getattr(exporter, 'use_flavor_mask', True)
            multi_me = helas_objects.HelasMultiProcess(cmd._curr_amps)
            for uid, me in enumerate(multi_me.get_matrix_elements()):
                me.get('processes')[0].set('uid', uid + 1)
            cmd._curr_matrix_elements = multi_me
        matrix_elements = multi_me.get_matrix_elements()
        self.assertTrue(matrix_elements, 'no matrix element to export')

        cpp_writer = helas_call_writers.CPPUFOHelasCallWriter(model)
        exporter.copy_template(model)
        for me_number, me in enumerate(matrix_elements):
            exporter.generate_subprocess_directory(me, cpp_writer, me_number)
        exporter.convert_model(model, multi_me.get_used_lorentz(),
                               multi_me.get_used_couplings())
        # ProcessExporterCPP.finalize() ignores its arguments and compiles src.
        exporter.finalize({'matrix_elements': matrix_elements}, '',
                          cmd.options, ['nojpeg'])
        return out_dir

    def _assert_me_lists_close(self, a, b, rtol=1e-5, atol=0.0):
        """Assert two matrix-element value lists agree as multisets (sorted),
        within a combined relative/absolute tolerance
        (|x-y| <= atol + rtol*max(|x|,|y|)).

        Backends print with different precision (the scalar C++ one 7 sig figs vs
        standalone (madmatrix) full double) and may emit the per-flavour values in a
        different order, so compare sorted rather than index-by-index / exact.
        `atol` lets callers treat numerically-tiny (vanishing-flavour) values as
        zero, where the different floating-point arithmetic of the Fortran vs
        cudacpp backends produces different noise."""
        a, b = sorted(a), sorted(b)
        self.assertEqual(len(a), len(b),
                         'different number of matrix elements: %d vs %d'
                         % (len(a), len(b)))
        for x, y in zip(a, b):
            self.assertLessEqual(abs(x - y), atol + rtol * max(abs(x), abs(y)),
                                 'matrix-element mismatch: %s vs %s' % (x, y))

    def test_standalone_cpp_fd_output_consistency(self):
        """test the standalone (madmatrix) output in FD gauge against standalone_fortran

        The standalone (madmatrix) matrix elements must agree with the
        Fortran standalone ones, both in FD gauge and in unitary gauge (and FD
        vs unitary, i.e. gauge invariance). madmatrix ships a UMAMI-based
        check_sa.exe whose 'matrix' mode is by design identical to the Fortran
        check driver; the per-flavour values are compared as sorted multisets
        (the backends may order flavours differently and print at different
        precision)."""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('set gauge FD')
        self.do('generate _quark _quark > h _quark _quark _quark _anti_quark  QCD=0')
        devnull = open(os.devnull,'w')
        energy = '1000'

        def get_values(output_format):
            self.do('output %s %s -f' % (output_format, self.out_dir))
            values = []
            proc_dir = os.path.join(self.out_dir, 'SubProcesses')
            directories = sorted([d for d in os.listdir(proc_dir) if d.startswith('P')])
            self.assertTrue(directories)
            if output_format == 'standalone_fortran':
                subprocess.call(['make'],
                                stdout=devnull, stderr=devnull,
                                cwd=os.path.join(self.out_dir, 'Source'))
            for oneproc in directories:
                logfile = os.path.join(proc_dir, oneproc, 'check.log')
                # standalone_fortran uses 'make check' + ./check; standalone ships a
                # UMAMI check_sa.exe whose 'matrix' mode == the Fortran driver.
                if output_format == 'standalone_fortran':
                    target = ['make', 'check']
                    check_exe = './check %s' % energy
                elif output_format == 'standalone':
                    target = ['make']
                    check_exe = './check_sa.exe %s' % energy
                else:
                    target = ['make']
                    check_exe = './check %s' % energy
                subprocess.call(target,
                                stdout=devnull, stderr=devnull,
                                cwd=os.path.join(proc_dir, oneproc))
                subprocess.call(check_exe,
                                stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                                cwd=os.path.join(proc_dir, oneproc), shell=True)
                log_output = open(logfile, 'r').read()
                me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                                   re.IGNORECASE)
                me_groups = me_re.findall(log_output)
                self.assertTrue(me_groups)
                values.extend(float(value) for value in me_groups)
            return values

        madmatrix = get_values('standalone')
        shutil.rmtree(self.out_dir)
        standalone = get_values('standalone_fortran')

        # atol: this process's matrix elements are O(1e-20), i.e. at the
        # floating-point noise floor, where the Fortran and cudacpp backends
        # differ; only require agreement above an absolute floor (the original
        # cpp-vs-standalone check used assertAlmostEqual, equally lenient here).
        self._assert_me_lists_close(madmatrix, standalone, atol=1e-7)

        self.do('set gauge unitary')
        self.do('generate _quark _quark > h _quark _quark _quark _anti_quark  QCD=0')
        devnull = open(os.devnull,'w')
        energy = '1000'

        shutil.rmtree(self.out_dir)
        madmatrix_no_fd = get_values('standalone')
        shutil.rmtree(self.out_dir)
        standalone_no_fd = get_values('standalone_fortran')

        self._assert_me_lists_close(madmatrix_no_fd, standalone_no_fd,
                                    atol=1e-7)
        # gauge invariance: unitary-gauge values must also match the FD ones.
        self._assert_me_lists_close(madmatrix_no_fd, standalone, atol=1e-7)

    def test_madmatrix_fd_vs_fortran(self):
        """FD gauge: madmatrix and the Fortran standalone must agree on the
        value, not merely both produce one.

        test_standalone_cpp_fd_output_consistency also compares the two
        backends in FD gauge, but its process (h + 4 jets) has matrix elements
        of order 1e-15 to 1e-21 while it asserts with atol=1e-7, so no wrong
        value can make it fail. Two madmatrix FD bugs lived through it: the
        Goldstone component of a longitudinal massive vector had the wrong sign
        on the scalar backends (0.7% off), and on the SIMD ones the FD block of
        vxxxxx broadcast its constants to the first lane only (nan elsewhere).

        u u~ > w+ w- is the smallest process that pins both down: the W's give
        a longitudinal polarisation (hence the Goldstone component), the
        massless initial state makes the two drivers evaluate the same RAMBO
        point, and |M|^2 is O(1e-3), far above the noise floor -- so the
        comparison is done with a relative tolerance and no absolute floor.
        The magnitude is asserted too, to keep this test from silently decaying
        into a vacuous one.

        madmatrix is checked once per backend: the FD wavefunctions are written
        twice in helas_fd.h, once under #ifndef MGONGPU_CPPSIMD and once for the
        vector types, and the two have drifted apart before. Which one a plain
        'make' builds depends on the host (cpu), so neither is exercised
        unless it is asked for by name.
        """
        energy = '1000'
        devnull = open(os.devnull, 'w')
        me_re = re.compile(r'Matrix element\s*=\s*([\d.eE+-]+)\s*GeV',
                           re.IGNORECASE)

        def get_values(output_format, check_exe, build_source=False,
                       backend=None):
            if os.path.isdir(self.out_dir):
                shutil.rmtree(self.out_dir)
            self.do('output %s %s -f' % (output_format, self.out_dir))
            if build_source:
                subprocess.call(['make'], stdout=devnull, stderr=devnull,
                                cwd=pjoin(self.out_dir, 'Source'))
            proc_root = pjoin(self.out_dir, 'SubProcesses')
            dirs = sorted(d for d in os.listdir(proc_root)
                          if d.startswith('P') and
                          os.path.isdir(pjoin(proc_root, d)))
            self.assertTrue(dirs, 'no subprocess for %s' % output_format)
            values = []
            for d in dirs:
                proc_dir = pjoin(proc_root, d)
                target = ['make', 'check'] if output_format == 'standalone_fortran' \
                    else ['make'] + (['BACKEND=%s' % backend] if backend else [])
                subprocess.call(target, stdout=devnull, stderr=devnull,
                                cwd=proc_dir)
                log = pjoin(proc_dir, 'check.log')
                subprocess.call('%s %s' % (check_exe, energy),
                                stdout=open(log, 'w'), stderr=subprocess.STDOUT,
                                cwd=proc_dir, shell=True)
                found = me_re.findall(open(log).read())
                self.assertTrue(found, '%s produced no matrix element (see %s)'
                                % (output_format, log))
                values.extend(float(v) for v in found)
            return values

        self.do('import model sm')
        self.do('set gauge FD')
        try:
            self.do('generate u u~ > w+ w-')
            standalone = get_values('standalone_fortran', './check', build_source=True)
            # scalar is the scalar code path, simd_128 the vector one (it maps
            # to NEON on arm)
            mg7 = dict((backend,
                        get_values('standalone', './check_sa.exe',
                                   backend=backend))
                       for backend in ('scalar', 'simd_128'))
        finally:
            self.do('set gauge unitary')

        # the comparison is only meaningful if the values are not noise
        self.assertGreater(max(abs(v) for v in standalone), 1e-6,
                           'matrix elements too small for a relative '
                           'comparison to mean anything: %s' % standalone)
        for backend, values in sorted(mg7.items()):
            # -ffast-math on the C++ side puts the backends ~3e-8 apart
            self._assert_me_lists_close(values, standalone, rtol=1e-6)

    def test_madmatrix_fd_simd_lanes(self):
        """FD gauge: the scalar and the vectorised madmatrix backends must
        compute the same thing, over many events.

        The FD wavefunctions are written twice in helas_fd.h -- once under
        #ifndef MGONGPU_CPPSIMD, once for the vector types -- and only the first
        event of the first lane is ever printed by check_sa.exe 'matrix' mode.
        A vector branch that is right in the first lane and wrong in the others
        is therefore invisible to a single point: the FD block of vxxxxx built
        its constants with a braced initialiser, which fills the first lane and
        zeroes the rest, and that gave nan (and later, once the surrounding code
        used selects, merely wrong numbers) in every other lane.

        'perf' mode runs many events through every lane, so comparing the mean
        matrix element of a scalar and a vectorised build covers them all. The
        two agree to the printed precision when the branches agree, and differed
        by 8e-5 relative with that bug in place.
        """
        devnull = open(os.devnull, 'w')
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('set gauge FD')
        try:
            self.do('generate u u~ > w+ w-')
            self.do('output standalone %s -f' % self.out_dir)
        finally:
            self.do('set gauge unitary')

        proc_root = pjoin(self.out_dir, 'SubProcesses')
        dirs = sorted(d for d in os.listdir(proc_root)
                      if d.startswith('P') and os.path.isdir(pjoin(proc_root, d)))
        self.assertTrue(dirs, 'no subprocess for standalone')

        def mean_me(proc_dir, backend):
            """mean |M|^2 over a multi-event run of the given backend"""
            exe = pjoin(proc_dir, 'check_sa.exe')
            if os.path.exists(exe):
                os.remove(exe)
            # USEBUILDDIR keeps the two backends' objects apart
            subprocess.call(['make', 'BACKEND=%s' % backend, 'USEBUILDDIR=1'],
                            stdout=devnull, stderr=devnull, cwd=proc_dir)
            if not os.path.exists(exe):
                return None  # backend not available on this host
            log = pjoin(proc_dir, 'perf_%s.log' % backend)
            subprocess.call('./check_sa.exe perf 1 32 300',
                            stdout=open(log, 'w'), stderr=subprocess.STDOUT,
                            cwd=proc_dir, shell=True)
            out = open(log).read()
            found = re.search(r'MeanMatrixElemValue\s*=\s*\(\s*([^\s+]+)', out)
            self.assertTrue(found, 'no mean matrix element (see %s)' % log)
            value = found.group(1)
            self.assertNotIn('nan', value.lower(),
                             'nan matrix element with BACKEND=%s (see %s)'
                             % (backend, log))
            return float(value)

        for d in dirs:
            proc_dir = pjoin(proc_root, d)
            # scalar is the scalar code path, simd_128 the vector one (it maps
            # to NEON on arm)
            scalar = mean_me(proc_dir, 'scalar')
            self.assertTrue(scalar, 'standalone did not build in %s' % proc_dir)
            self.assertGreater(scalar, 0.,
                               'null mean matrix element in %s' % proc_dir)
            vector = mean_me(proc_dir, 'simd_128')
            if vector is None:
                continue  # no vectorised backend here: nothing to compare
            self.assertLessEqual(abs(vector - scalar), 1e-5 * scalar,
                                 'the scalar and vectorised backends disagree '
                                 'in %s: %s vs %s' % (d, scalar, vector))

    def test_madmatrix_hel_recycling_vs_plain(self):
        """--hel_recycling must not change |M|^2, event by event.

        The recycled build makes every helicity row of a page at once and reads
        the rows back in the helicity loop; with one helicity per event
        (UMAMI_IN_HELICITY_INDEX, check_sa --hel-index: what MadTrex and a
        sampled-helicity run pass) each SIMD lane can have its own, and the
        recycled build has to take the plain per-lane code instead. Both are
        compared here to the plain output on the same points, at the scalar
        and a SIMD backend: the helicity sum, and one random helicity per event.

        g g > t t~ g is small enough to build in seconds and still has helicity
        rows that share wavefunctions, amplitudes that vanish at some rows and,
        at simd_128, two events per lane in mixed precision. FPTYPE=d keeps the
        two builds within rounding of each other (1e-12); the warm-up must have
        produced the recycled build (HelRecycling.inc), or the test would compare
        plain with plain.
        """
        devnull = open(os.devnull, 'w')
        root = self.out_dir + '_hr'
        dirs = {'plain': root + '_plain', 'recycled': root + '_recycled'}
        for path in dirs.values():
            if os.path.isdir(path):
                shutil.rmtree(path)
        self.do('import model sm')
        self.do('generate g g > t t~ g')
        self.do('output standalone %s -f' % dirs['plain'])
        self.do('output standalone %s -f --hel_recycling=True' % dirs['recycled'])
        proc = {}
        for name, path in dirs.items():
            proc_root = pjoin(path, 'SubProcesses')
            found = sorted(d for d in os.listdir(proc_root)
                           if d.startswith('P') and os.path.isdir(pjoin(proc_root, d)))
            self.assertEqual(len(found), 1, 'one subprocess expected in %s' % path)
            proc[name] = pjoin(proc_root, found[0])
        self.assertTrue(os.path.exists(pjoin(proc['recycled'], 'HelRecycling.inc')),
                        'the --hel_recycling warm-up did not produce the recycled build')
        ncomb = int(re.search(r'constexpr int ncomb = (\d+)',
                              open(pjoin(proc['plain'], 'ProcessData.h')).read()).group(1))
        nevt = 512
        random.seed(11)
        helicities = array.array('i', [random.randrange(0, ncomb) for _ in range(nevt)])
        hel_file = pjoin(root + '_hel.bin')
        with open(hel_file, 'wb') as fsock:
            helicities.tofile(fsock)

        def dump(proc_dir, backend, tag, extra=()):
            exe = pjoin(proc_dir, 'check_sa.exe')
            if os.path.exists(exe):
                os.remove(exe)
            subprocess.call(['make', 'BACKEND=%s' % backend, 'FPTYPE=d', 'USEBUILDDIR=1'],
                            stdout=devnull, stderr=devnull, cwd=proc_dir)
            if not os.path.exists(exe):
                return None  # backend not available on this host
            out = pjoin(proc_dir, 'me_%s_%s.bin' % (backend, tag))
            subprocess.call(['./check_sa.exe', 'perf'] + list(extra) +
                            ['--dump-me', out, '16', '32', '1'],
                            stdout=devnull, stderr=subprocess.STDOUT, cwd=proc_dir)
            self.assertTrue(os.path.exists(out), 'no |M|^2 dump in %s' % proc_dir)
            return array.array('d', open(out, 'rb').read())

        compared = 0
        for backend in ('scalar', 'simd_128'):
            for tag, extra in (('sum', ()), ('hel', ('--hel-index', hel_file))):
                plain = dump(proc['plain'], backend, tag, extra)
                recycled = dump(proc['recycled'], backend, tag, extra)
                if plain is None or recycled is None:
                    continue
                self.assertEqual(len(plain), nevt)
                self.assertEqual(len(recycled), nevt)
                self.assertTrue(any(plain), 'all-zero |M|^2 (%s, %s)' % (backend, tag))
                for i, (a, b) in enumerate(zip(plain, recycled)):
                    self.assertLessEqual(abs(a - b), 1e-12 * abs(a),
                                         '%s %s event %d: plain %r, recycled %r'
                                         % (backend, tag, i, a, b))
                compared += 1
        self.assertGreaterEqual(compared, 2, 'the scalar backend did not build')

    def test_standalone_cpp_perf_alpha_s_from_param_card(self):
        """check_sa.exe 'perf' mode must use the alpha_s of the param card,
        as the 'matrix' mode does.

        With --momenta (or -e) and --dump-me, perf mode is how the matrix
        element is evaluated at user-given points. It did not pass alpha_s to
        umami_matrix_element, so UMAMI fell back to its hard-coded g_s
        (alpha_s = 0.118) whatever the card said.

        g g > t t~ is compared to the analytic result (Ellis, Stirling, Webber,
        table 10.2) at the card alpha_s, then at twice that value. The top
        width in the t-channel propagators moves MadGraph away from it by about
        3e-6 at this point.
        """
        import array
        import math
        import madgraph.models.check_param_card as check_param_card

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)
        self.do('import model sm')
        self.do('generate g g > t t~')
        self.do('output standalone %s' % self.out_dir)

        proc_root = pjoin(self.out_dir, 'SubProcesses')
        dirs = [d for d in os.listdir(proc_root)
                if d.startswith('P') and os.path.isdir(pjoin(proc_root, d))]
        self.assertEqual(len(dirs), 1, dirs)
        proc_dir = pjoin(proc_root, dirs[0])
        devnull = open(os.devnull, 'w')
        subprocess.call(['make', 'FPTYPE=d'], stdout=devnull, stderr=devnull,
                        cwd=proc_dir)
        self.assertTrue(os.path.exists(pjoin(proc_dir, 'check_sa.exe')),
                        'check_sa.exe did not build in %s' % proc_dir)

        card_path = pjoin(self.out_dir, 'Cards', 'param_card.dat')
        card = check_param_card.ParamCard(card_path)
        mt = card['mass'].get(6).value

        # one on-shell point in the partonic centre-of-mass frame
        energy, theta, phi = 500., 0.7, 0.3
        pabs = math.sqrt(energy**2 - mt**2)
        p3 = [energy, pabs * math.sin(theta) * math.cos(phi),
              pabs * math.sin(theta) * math.sin(phi), pabs * math.cos(theta)]
        p4 = [energy] + [-x for x in p3[1:]]
        point = [energy, 0., 0., energy] + [energy, 0., 0., -energy] + p3 + p4
        # repeated 8 times: perf mode needs a multiple of the SIMD width
        with open(pjoin(proc_dir, 'pts.bin'), 'wb') as fsock:
            array.array('d', point * 8).tofile(fsock)

        s = 4 * energy**2
        tau1 = 2 * energy * (p3[0] - p3[3]) / s   # 2 p1.p3 / s
        tau2 = 2 * energy * (p3[0] + p3[3]) / s   # 2 p2.p3 / s
        rho = 4 * mt**2 / s
        def analytic(alpha_s):
            return (4 * math.pi * alpha_s)**2 \
                * (1. / (6 * tau1 * tau2) - 3. / 8.) \
                * (tau1**2 + tau2**2 + rho - rho**2 / (4 * tau1 * tau2))

        def perf_mes(tag):
            dump = 'me_%s.bin' % tag
            subprocess.call(['./check_sa.exe', 'perf', '--momenta', 'pts.bin',
                             '--dump-me', dump, '1', '8', '1'],
                            stdout=devnull, stderr=devnull, cwd=proc_dir)
            self.assertTrue(os.path.exists(pjoin(proc_dir, dump)),
                            'check_sa.exe perf wrote no %s' % dump)
            mes = array.array('d')
            with open(pjoin(proc_dir, dump), 'rb') as fsock:
                mes.frombytes(fsock.read())
            self.assertEqual(len(mes), 8)
            return mes

        alpha_s = card['sminputs'].get(3).value
        for tag, value in (('card', alpha_s), ('doubled', 2 * alpha_s)):
            card['sminputs'].get(3).value = value
            card.write(card_path)
            expected = analytic(value)
            for me in perf_mes(tag):
                self.assertLessEqual(abs(me - expected), 1e-4 * expected,
                                     'alpha_s = %s: %s instead of %s'
                                     % (value, me, expected))

    def test_madmatrix_vs_fortran(self):
        """Cross-check that standalone (madmatrix) reproduces the
        standalone_fortran matrix elements for p p > e+ e- QCD=0.

        Uses a massless final state so both check drivers evaluate the same
        default 1000 GeV phase-space point (no energy auto-bump mismatch), and
        compares the per-flavour matrix elements as sorted multisets (the two
        backends may emit them in a different order and at different printed
        precision). madmatrix ships a UMAMI-based check_sa.exe whose
        'matrix' mode is by design identical to the Fortran check driver.
        """
        energy = '1000'
        devnull = open(os.devnull, 'w')

        def get_values(output_format, check_exe, build_source=False):
            if os.path.isdir(self.out_dir):
                shutil.rmtree(self.out_dir)
            self.do('output %s %s' % (output_format, self.out_dir))
            if build_source:
                subprocess.call(['make'], stdout=devnull, stderr=devnull,
                                cwd=os.path.join(self.out_dir, 'Source'))
            proc_root = os.path.join(self.out_dir, 'SubProcesses')
            dirs = sorted(d for d in os.listdir(proc_root)
                          if d.startswith('P') and
                          os.path.isdir(os.path.join(proc_root, d)))
            self.assertTrue(dirs, 'no subprocess for %s' % output_format)
            values = []
            me_re = re.compile(r'Matrix element\s*=\s*([\d.eE+-]+)\s*GeV',
                               re.IGNORECASE)
            for d in dirs:
                proc_dir = os.path.join(proc_root, d)
                # standalone_fortran uses 'make check' + ./check; standalone
                # ships a UMAMI check_sa.exe whose 'matrix' mode == the Fortran
                # driver.
                target = ['make', 'check'] \
                    if output_format == 'standalone_fortran' else ['make']
                subprocess.call(target, stdout=devnull, stderr=devnull,
                                cwd=proc_dir)
                log = os.path.join(proc_dir, 'check.log')
                subprocess.call('%s %s' % (check_exe, energy),
                                stdout=open(log, 'w'), stderr=subprocess.STDOUT,
                                cwd=proc_dir, shell=True)
                found = me_re.findall(open(log).read())
                self.assertTrue(found, '%s produced no matrix element (see %s)'
                                % (output_format, log))
                values.extend(float(v) for v in found)
            return values

        self.do('import model sm')
        self.do('generate p p > e+ e- QCD=0')
        fortran = get_values('standalone_fortran', './check', build_source=True)
        self.assertTrue(any(v != 0.0 for v in fortran),
                        'all matrix elements vanished for p p > e+ e- QCD=0')
        mg7 = get_values('standalone', './check_sa.exe')
        self._assert_me_lists_close(mg7, fortran)

    def test_madmatrix_mssm_single_leg(self):
        """Single-merged-leg flavored couplings must give the same per-flavor
        |M|^2 in standalone (madmatrix) as in the Fortran standalone.

        p p > n1 n1 QCD=0 is a t-channel-squark process with single-merged-leg
        vertices (one merged light quark + an unmerged neutralino + a squark)
        whose flavored couplings are *independent* (electroweak), isolating the
        single-leg consumer-gating fix (the get_coupling_def merged-leg
        selection) from the still-guarded dependent-coupling case. Without the
        fix only the first flavor matches; with it all flavors do.
        """
        energy = '1000'
        devnull = open(os.devnull, 'w')
        me_re = re.compile(r'Matrix element\s*=\s*([\d.eE+-]+)\s*GeV',
                           re.IGNORECASE)

        def get_values(output_format, check_exe, build_source=False):
            if os.path.isdir(self.out_dir):
                shutil.rmtree(self.out_dir)
            self.do('output %s %s -f' % (output_format, self.out_dir))
            if build_source:
                subprocess.call(['make'], stdout=devnull, stderr=devnull,
                                cwd=os.path.join(self.out_dir, 'Source'))
            proc_root = os.path.join(self.out_dir, 'SubProcesses')
            dirs = sorted(d for d in os.listdir(proc_root)
                          if d.startswith('P') and
                          os.path.isdir(os.path.join(proc_root, d)))
            self.assertTrue(dirs, 'no subprocess for %s' % output_format)
            values = []
            for d in dirs:
                proc_dir = os.path.join(proc_root, d)
                target = ['make', 'check'] if output_format == 'standalone_fortran' \
                    else ['make']
                subprocess.call(target, stdout=devnull, stderr=devnull,
                                cwd=proc_dir)
                log = os.path.join(proc_dir, 'check.log')
                subprocess.call('%s %s' % (check_exe, energy),
                                stdout=open(log, 'w'), stderr=subprocess.STDOUT,
                                cwd=proc_dir, shell=True)
                found = me_re.findall(open(log).read())
                self.assertTrue(found, '%s produced no matrix element (see %s)'
                                % (output_format, log))
                values.extend(float(v) for v in found)
            return values

        self.do('import model MSSM_SLHA2')
        self.do('generate p p > n1 n1 QCD=0')
        mg7 = get_values('standalone', './check_sa.exe')
        standalone = get_values('standalone_fortran', './check', build_source=True)
        self.assertTrue(any(v != 0.0 for v in standalone),
                        'all matrix elements vanished for p p > n1 n1')
        self._assert_me_lists_close(mg7, standalone, rtol=1e-4)

    def test_madmatrix_mssm_gogo(self):
        """Dependent (event-by-event, running-alphas) flavored couplings must
        give the same per-flavor |M|^2 in standalone (madmatrix) as in the
        Fortran standalone.

        MSSM 'p p > go go' has single-merged-leg squark/gluino-quark vertices
        (one merged light quark + an unmerged gluino + a squark) whose flavored
        couplings are *dependent* (SUSY-QCD, running-alphas): the squark-quark-
        gluino coupling GC_106/GC_110 ~ g_s changes per event. These are not
        addressable as fixed value[] pointers, so they are gathered event-by-
        event into cDPF_* / flvCOUPs_dep (Step 3 of
        docs/mg7_merged_flavor_mssm_design.md). This is the dependent-coupling
        counterpart of test_madmatrix_mssm_single_leg (independent flavored
        couplings) and the consistency check matching test_madevent_mssm_gogo.

        The energy (sqrt(s)) is chosen above the gluino-pair threshold (Mgo ~
        608 GeV) so neither check driver auto-bumps it, i.e. both evaluate the
        same phase-space point.
        """
        energy = '3000'
        devnull = open(os.devnull, 'w')
        me_re = re.compile(r'Matrix element\s*=\s*([\d.eE+-]+)\s*GeV',
                           re.IGNORECASE)

        def get_values(output_format, check_exe, build_source=False):
            if os.path.isdir(self.out_dir):
                shutil.rmtree(self.out_dir)
            self.do('output %s %s -f' % (output_format, self.out_dir))
            if build_source:
                subprocess.call(['make'], stdout=devnull, stderr=devnull,
                                cwd=os.path.join(self.out_dir, 'Source'))
            proc_root = os.path.join(self.out_dir, 'SubProcesses')
            dirs = sorted(d for d in os.listdir(proc_root)
                          if d.startswith('P') and
                          os.path.isdir(os.path.join(proc_root, d)))
            self.assertTrue(dirs, 'no subprocess for %s' % output_format)
            values = []
            for d in dirs:
                proc_dir = os.path.join(proc_root, d)
                target = ['make', 'check'] if output_format == 'standalone_fortran' \
                    else ['make']
                subprocess.call(target, stdout=devnull, stderr=devnull,
                                cwd=proc_dir)
                log = os.path.join(proc_dir, 'check.log')
                subprocess.call('%s %s' % (check_exe, energy),
                                stdout=open(log, 'w'), stderr=subprocess.STDOUT,
                                cwd=proc_dir, shell=True)
                found = me_re.findall(open(log).read())
                self.assertTrue(found, '%s produced no matrix element (see %s)'
                                % (output_format, log))
                values.extend(float(v) for v in found)
            return values

        self.do('import model MSSM_SLHA2')
        self.do('generate p p > go go')
        mg7 = get_values('standalone', './check_sa.exe')
        standalone = get_values('standalone_fortran', './check', build_source=True)
        self.assertTrue(any(v != 0.0 for v in standalone),
                        'all matrix elements vanished for p p > go go')
        self._assert_me_lists_close(mg7, standalone, rtol=1e-4)

    def test_madevent_mssm_gogo(self):
        """The Fortran madevent output supports MSSM 'p p > go go' (merged-flavor
        squark/gluino vertices with single-merged-leg / event-by-event flavored
        couplings). The mg7/madmatrix C++ output now also supports it and is
        checked to agree per-flavor in test_madmatrix_mssm_gogo; this acts
        as the madevent counterpart.
        """
        self.do('import model MSSM_SLHA2')
        self.do('generate p p > go go')
        self.do('output madevent %s -f' % self.out_dir)
        self.assertTrue(
            os.path.isdir(os.path.join(self.out_dir, 'SubProcesses')),
            'madevent output should support MSSM p p > go go')
        proc_root = os.path.join(self.out_dir, 'SubProcesses')
        proc_dirs = [d for d in os.listdir(proc_root)
                     if d.startswith('P') and
                     os.path.isdir(os.path.join(proc_root, d))]
        self.assertTrue(proc_dirs,
                        'madevent produced no subprocess for p p > go go')



    def test_quarkonium_standalone(self):
        """Standalone test for various quarkonium production matrix elements"""

        import os
        import re
        import subprocess
        import glob
        import madgraph.interface.master_interface as MGCmd

        # --------------------------------------------------
        # Create MG5 interface
        # --------------------------------------------------
        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()

        mg_cmd.exec_cmd('import model sm_onia-c_mass')

        process_list = [
            ['g g > chic1(1|3P11) chib0(1|3P01)', 3.132275172481691e-16],
            ['g g > hc(1|1P11) g', 2.3637208371566567e-12],
            ['u u~ > a Jpsi(1|3P08) QCD=99 QED=99', 3.6650612158421924e-11],
            ['u a > Upsilon(1|3S11) u chib2(1|3P21) QCD=99 QED=99', 1.819597262304262e-20],
        ]
        for process in process_list:
            mg_cmd.exec_cmd('generate %s ' % process[0])

            # --------------------------------------------------
            # Generate process
            # --------------------------------------------------
            mg_cmd.exec_cmd('output standalone_fortran %s -f' % (self.out_dir))

            self.assertTrue(os.path.isdir(self.out_dir))

            # --------------------------------------------------
            # Find subprocess directory
            # --------------------------------------------------
            proc_dirs = glob.glob(os.path.join(self.out_dir,
                                            "SubProcesses",
                                            "P*"))

            self.assertTrue(proc_dirs, "No P* subprocess found")

            # --------------------------------------------------
            # Compile inside subprocess
            # --------------------------------------------------
            proc_dir = proc_dirs[0]
            subprocess.check_call(['make'], cwd=proc_dir)

            # --------------------------------------------------
            # Run check
            # --------------------------------------------------
            output = subprocess.check_output(
                ['./check'],
                cwd=proc_dir
            ).decode()

            # --------------------------------------------------
            # Extract matrix element
            # --------------------------------------------------
            match = re.search(r'Matrix element\s*=\s*([0-9Ee\+\-\.]+)', output)
            self.assertIsNotNone(match)

            val1 = float(match.group(1))
            target =  process[1]  # value to compare to
            passed = abs(val1-target)/(val1+target) < 1e-08

            self.assertTrue(passed)

         
    def test_standalone_density(self):
        """test that standalone density is working"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('generate p p > j t t~ ')
        self.do('output standalone_fortran %s --density=4,5 -f' % self.out_dir)
        devnull = open(os.devnull,'w')
    
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_gg_gttx',
                               'check_sa.log')
        # Check that check_sa.f compiles
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_gg_gttx'))
        
        subprocess.call(['./check'],
                        stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_gg_gttx'))
    
        log_output = open(logfile, 'r').read()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)    
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 0.0004188716594531423, 5)

        # check density matrix
        #
        #particle 3 has helicity -1 -1
        #particle 4 has helicity 1 1
        #value is 1 (2.78194898230116219E-011,0.0000000000000000)
        me_re = re.compile(r'particle\s+4\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'particle\s+5\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'value is\s+\d+\s*\(([\d\.eE\+-]+),([\d\.eE\+-]+)\)', re.MULTILINE)
        all_matches = me_re.findall(log_output)
        sol = {}
        for match in all_matches:
            sol[(int(match[0]), int(match[1]), int(match[2]), int(match[3]))] = (float(match[4]), float(match[5]))

        # We changed the value of the reference by a factor of 256, which is the inclusion of IDEN in get_inter in matrix.
        # original_sol = {(-1, -1, 1, 1): (0.02827952274928987, 0.0), (-1, -1, 1, -1): (-0.0041892876162345, -0.0041923830983622255), (-1, 1, 1, 1): (0.000469685615962711, 0.0006142055733429721), (-1, 1, 1, -1): (-0.01784029173125566, -0.00794999696313525), (-1, -1, -1, -1): (0.02532739017396033, 0.0), (-1, 1, -1, 1): (-0.00028182588524174187, 0.0024162264334765746), (-1, 1, -1, -1): (-0.00048593945847553023, -0.0006039982074415239), (1, 1, 1, 1): (0.025301510150454294, 0.0), (1, 1, 1, -1): (0.004212401136919661, 0.0042167644618831875), (1, 1, -1, -1): (0.028322721746299958, 0.0)}
        original_sol = {(-1, -1, 1, 1): (0.00011046688573941356, 0.0), (-1, -1, 1, -1): (-1.6364404750916015e-05, -1.6376496477977443e-05), (-1, 1, 1, 1): (1.83470943735434e-06, 2.3992405208709848e-06), (-1, 1, 1, -1): (-6.968863957521743e-05, -3.105467563724707e-05), (-1, -1, -1, -1): (9.893511786703254e-05, 0.0), (-1, 1, -1, 1): (-1.1008823642255542e-06, 9.43838450576787e-06), (-1, 1, -1, -1): (-1.89820100967004e-06, -2.359367997818453e-06), (1, 1, 1, 1): (9.883402402521209e-05, 0.0), (1, 1, 1, -1): (1.6454691941092424e-05, 1.64717361792312e-05), (1, 1, -1, -1): (0.00011063563182148421, 0.0)}

        for key in original_sol:
            self.assertIn(key, sol)
            self.assertAlmostEqual(original_sol[key][0], sol[key][0])
            self.assertAlmostEqual(original_sol[key][1], sol[key][1])

        ########################################################################
        ### check case with polarization vectors
        ########################################################################
        self.do('generate u u~ > z{0} z{T} g')
        self.do('output standalone_fortran %s --density=3,4,5 -f ' % self.out_dir)
        devnull = open(os.devnull,'w')
    
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_uux_z0zTg',
                               'check_sa.log')
        # Check that check_sa.f compiles
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_uux_z0zTg'))
        
        subprocess.call(['./check'],
                        stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_uux_z0zTg'))
    
        log_output = open(logfile, 'r').read()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)    
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 3.140334500813846e-08, 5)

        # check density matrix
        #
        #particle 3 has helicity -1 -1
        #particle 4 has helicity 1 1
        #value is 1 (2.78194898230116219E-011,0.0000000000000000)
        me_re = re.compile(r'particle\s+3\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'particle\s+4\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'particle\s+5\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'value is\s+\d+\s*\(([\d\.eE\+-]+),([\d\.eE\+-]+)\)', re.MULTILINE)
        all_matches = me_re.findall(log_output)
        sol = {}
        for match in all_matches:
            sol[(int(match[0]), int(match[1]), int(match[2]), int(match[3]), int(match[4]),int(match[5]))] = (float(match[6]), float(match[7]))
            self.assertTrue(len(match), 8)
            self.assertEqual(int(match[0]), int(match[1]))
            self.assertEqual(int(match[0]), 0)
            self.assertIn(int(match[2]), [-1,1])
            self.assertIn(int(match[3]), [-1,1])

        self.assertEqual(len(sol), 10)

        # We changed the value of the reference by a factor of 36, which is the inclusion of IDEN in get_inter in matrix.
        # original_sol =  {(0, 0, 1, 1, -1, -1): (3.4001167694559294e-07, 0.0), (0, 0, 1, 1, -1, 1): (-3.1461674759236455e-07, 4.8545818497513406e-09), (0, 0, 1, -1, -1, -1): (-3.069602710730949e-07, 4.832171908212019e-09), (0, 0, 1, -1, -1, 1): (2.7270589345996334e-07, -2.0762612868036477e-08), (0, 0, 1, 1, 1, 1): (2.914827381801457e-07, 0.0), (0, 0, 1, -1, 1, -1): (2.8446952890247865e-07, -7.217171184019204e-11), (0, 0, 1, -1, 1, 1): (-2.5326248092821595e-07, 1.519703606636158e-08), (0, 0, -1, -1, -1, -1): (2.7764709546297174e-07, 0.0), (0, 0, -1, -1, -1, 1): (-2.4727978606009926e-07, 1.4752928639145769e-08), (0, 0, -1, -1, 1, 1): (2.2137890970427402e-07, 0.0)}
        original_sol =  {(0, 0, 1, 1, -1, -1): (9.444768804044248e-09, 0.0), (0, 0, 1, 1, -1, 1): (-8.739354099787904e-09, 1.3484949582642612e-10), (0, 0, 1, -1, -1, -1): (-8.526674196474859e-09, 1.3422699745033388e-10), (0, 0, 1, -1, -1, 1): (7.575163707221203e-09, -5.767392463343466e-10), (0, 0, 1, 1, 1, 1): (8.09674272722627e-09, 0.0), (0, 0, 1, -1, 1, -1): (7.901931358402185e-09, -2.004769773338668e-12), (0, 0, 1, -1, 1, 1): (-7.035068914672666e-09, 4.2213989073226606e-10), (0, 0, -1, -1, -1, -1): (7.712419318415882e-09, 0.0), (0, 0, -1, -1, -1, 1): (-6.868882946113868e-09, 4.098035733096047e-10), (0, 0, -1, -1, 1, 1): (6.149414158452056e-09, 0.0)}
        for key in original_sol:
            
            self.assertIn(key, sol)
            self.assertAlmostEqual(original_sol[key][0], sol[key][0])
            self.assertAlmostEqual(original_sol[key][1], sol[key][1])

        ########################################################################
        ### check Z > t t~ case
        ######################################################################## 
        self.do('generate z > b b~')
        self.do('output standalone_fortran %s --density=1 -f ' % self.out_dir)
        devnull = open(os.devnull,'w')
    
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_z_bbx',
                               'check_sa.log')
        # Check that check_sa.f compiles
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_z_bbx'))
        
        subprocess.call(['./check'],
                        stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_z_bbx'))
    
        log_output = open(logfile, 'r').read()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output) 
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 1696.1593018147787, 5)

        # check density matrix
        #
        #particle 3 has helicity -1 -1
        #particle 4 has helicity 1 1
        #value is 1 (2.78194898230116219E-011,0.0000000000000000)
        me_re = re.compile(
                           r'particle\s+1\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'value is\s+\d+\s*\(([\d\.eE\+-]+),([\d\.eE\+-]+)\)', re.MULTILINE)
        all_matches = me_re.findall(log_output)
        sol = {}
        for match in all_matches:
            sol[(int(match[0]), int(match[1]))] = (float(match[2]), float(match[3]))


        self.assertEqual(len(sol), 6)

        # We changed the value of the reference by a factor of 3, which is the inclusion of IDEN in get_inter in matrix.
        # original_sol =  {(-1, -1): (520.1204099513759, 0.0), (-1, 0): (-217.2395066557355, 871.1782252029083), (-1, 1): (-939.258737149777, -499.49184104989456), (0, 0): (2136.628541206853, 0.0), (0, 1): (-534.1281427839928, 2141.9713873632077), (1, 1): (2431.7289542861076, 0.0)}
        original_sol =  {(-1, -1): (173.37346998379198, 0.0), (-1, 0): (-72.41316888524517, 290.3927417343028), (-1, 1): (-313.0862457165923, -166.49728034996485), (0, 0): (712.2095137356176, 0.0), (0, 1): (-178.04271426133093, 713.9904624544025), (1, 1): (810.5763180953692, 0.0)}
        for key in original_sol:    
            self.assertIn(key, sol)
            self.assertAlmostEqual(original_sol[key][0], sol[key][0])
            self.assertAlmostEqual(original_sol[key][1], sol[key][1])


        ########################################################################
        ### check case with interference computation
        ######################################################################## 
        self.do('generate u u~ > t t~ QCD^2==2')
        self.do('output standalone_fortran %s --density=3,4 -f ' % self.out_dir)
        devnull = open(os.devnull,'w')
    
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_uux_ttx',
                               'check_sa.log')
        # Check that check_sa.f compiles
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_uux_ttx'))
        
        subprocess.call(['./check'],
                        stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_uux_ttx'))
    
        log_output = open(logfile, 'r').read()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output) 
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 0, 5)

        # check density matrix
        #
        #particle 3 has helicity -1 -1
        #particle 4 has helicity 1 1
        #value is 1 (2.78194898230116219E-011,0.0000000000000000)
        me_re = re.compile(
                           r'particle\s+3\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'particle\s+4\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'value is\s+\d+\s*\(([\d\.eE\+-]+),([\d\.eE\+-]+)\)', re.MULTILINE)
        all_matches = me_re.findall(log_output)
        sol = {}
        for match in all_matches:
            sol[(int(match[0]), int(match[1]), int(match[2]), int(match[3]))] = (float(match[4]), float(match[5]))
            self.assertTrue(len(match), 6)
            self.assertIn(int(match[0]), [-1,1])
            self.assertIn(int(match[1]), [-1,1])
            self.assertIn(int(match[2]), [-1,1])
            self.assertIn(int(match[3]), [-1,1])


        self.assertEqual(len(sol), 10)
        # I am not sure why for this case I didn't have to introduce the normalisation term
        original_sol = sol =  {(-1, -1, 1, 1): (-5.551115123125783e-17, 0.0), (-1, -1, 1, -1): (0.00926133766116009, 0.0), (-1, 1, 1, 1): (-0.009292545047075378, 0.0), (-1, 1, 1, -1): (0.09884700885130696, 0.0), (-1, -1, -1, -1): (-1.6479873021779667e-17, 0.0), (-1, 1, -1, 1): (5.415404411525035e-07, 0.0), (-1, 1, -1, -1): (-0.012877349160785171, 0.0), (1, 1, 1, 1): (-1.734723475976807e-17, 0.0), (1, 1, 1, -1): (0.012880402732054583, 0.0), (1, 1, -1, -1): (-4.85722573273506e-17, 0.0)}
        for key in original_sol:
            self.assertIn(key, sol)
            self.assertAlmostEqual(original_sol[key][0] if abs(original_sol[key][0]) > 1e-12 else 0, sol[key][0])
            self.assertAlmostEqual(original_sol[key][1] if abs(original_sol[key][1]) > 1e-12 else 0, sol[key][1])

    def test_standalone_zerowidth(self):
        """output standalone_fortran --zerowidth_decay=: in a decay only the decaying
        particle's propagators get a zero width, in a production the listed
        final-state particles' ones; nothing changes without the option."""

        def calls(out_dir, pdir):
            # the P<n>_ numbering depends on the generation, not the suffix
            sub = os.path.join(out_dir, 'SubProcesses')
            pdir = [d for d in os.listdir(sub) if d.split('_', 1)[-1] == pdir][0]
            text = open(os.path.join(sub, pdir, 'matrix.f')).read()
            return [l.strip() for l in text.split('\n')
                    if l.strip().startswith('CALL') and 'XXXXX' not in l]

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)
        self.do('set group_subprocesses False')
        self.do('generate t > e+ ve b')
        self.do('add process t > w+ b g')
        for option in ('True', '6,24'):
            self.do('output standalone_fortran %s --prefix=int --density=1 --zerowidth_decay=%s -f'
                    % (self.out_dir, option))
            # the internal W of the top decay keeps its width
            self.assertTrue([c for c in calls(self.out_dir, 't_epveb')
                             if ',MDL_MW,MDL_WW,' in c])
            # the internal top of t > w+ b g does not
            self.assertTrue([c for c in calls(self.out_dir, 't_wpbg')
                             if ',MDL_MT, ZERO,' in c])
        self.do('output standalone_fortran %s --prefix=int --density=1 --zerowidth_decay=False -f'
                % self.out_dir)
        self.assertTrue([c for c in calls(self.out_dir, 't_wpbg')
                         if ',MDL_MT,MDL_WT,' in c])

        self.do('generate e+ e- > t t~ z')
        self.do('output standalone_fortran %s --prefix=int --density=1 --zerowidth_decay=6,24 -f'
                % self.out_dir)
        zeroed = [c for c in calls(self.out_dir, 'epem_ttxz') if 'MDL_MT' in c]
        self.assertEqual(len(zeroed), 2)
        self.assertTrue(all('MDL_MT, ZERO,' in c for c in zeroed), zeroed)

        self.do('output standalone_fortran %s --prefix=int --density=1 -f' % self.out_dir)
        kept = [c for c in calls(self.out_dir, 'epem_ttxz') if 'MDL_MT' in c]
        self.assertTrue(all('MDL_MT,MDL_WT,' in c for c in kept), kept)

    def test_standalone_density_uu(self):
        ############################################################################
        # Check convolution of density matrix with decay matrix 
        # reproduces the full matrix-element
        # testing case u u~ > z z, z > e+ e-
        ############################################################################
        self.do('generate u u~ > z z')
        self.do('output standalone_fortran %s_prod --density=3,4 -f ' % self.out_dir)
        self.do('generate u u~ > z z, z > e+ e-')
        self.do('output standalone_fortran %s_full -f ' % self.out_dir) 
        self.do('generate z > e+ e- --standalone') # --standalone allow mix 2>1 and 2>2 processes
        self.do('output standalone_fortran %s_dec1 --density=1 -f ' % self.out_dir)
        self.do('output standalone_fortran %s_dec2 --density=1 -f ' % self.out_dir)
        # Read a test event for u u~ > z z, z > e+ e-
        


        text_lhe = """ <event>
 8      1 +9.3182000e+00 9.58106800e+01 7.54677100e-03 1.28936100e-01
       -2 -1    0    0    0  501 -0.0000000000e+00 +0.0000000000e+00 +1.3133897947e+02 1.3133897947e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000000e+00 -0.0000000000e+00 -1.8928771624e+02 1.8928771624e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
       23  2    1    2    0    0 +2.7856507549e+01 +9.4048216813e+00 -1.5629560828e+02 1.8332485973e+02 9.1188000000e+01 0.0000e+00 0.0000e+00
       23  2    1    2    0    0 -2.7856507549e+01 -9.4048216813e+00 +9.8346871509e+01 1.3730183598e+02 9.1188000000e+01 0.0000e+00 0.0000e+00
      -11  1    3    3    0    0 +4.1207329404e+01 +1.3210392522e+01 -2.2428975695e+01 4.8740305887e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
       11  1    3    3    0    0 -1.3350821855e+01 -3.8055708401e+00 -1.3386663259e+02 1.3458455385e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
      -11  1    4    4    0    0 -5.7647128134e+01 -2.1962673907e+01 +7.6540783391e+01 9.8305859181e+01 0.0000000000e+00 0.0000e+00 -1.0000e+00
       11  1    4    4    0    0 +2.9790620585e+01 +1.2557852226e+01 +2.1806088117e+01 3.8995976798e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
       </event>"""

        
        Event = lhe_parser.Event()
        Event.parse(text_lhe.split('\n')) 
        # get the associate momenta for each matrix-element
        production_p = [Event[0], Event[1], Event[2], Event[3]]
        full_p = [Event[0], Event[1], Event[4], Event[5], Event[6], Event[7]]
        decay_z1 = [Event[2], Event[4], Event[5]]
        decay_z2 = [Event[3], Event[6], Event[7]]
        all_p = [production_p, full_p, decay_z1, decay_z2]


        all_me = []
        all_dens = []
        all_index= []
        # compiles the three directories:
        for i, (mdir, Pdir) in enumerate([(self.out_dir+'_prod', 'P0_uux_zz'), 
                                 (self.out_dir+'_full', 'P0_uux_zz_z_epem_z_epem'), 
                                 (self.out_dir+'_dec1', 'P0_z_epem'),
                                 (self.out_dir+'_dec2', 'P0_z_epem')]):
            p = all_p[i]
            self.edit_p_in_standalone(os.path.join(mdir, 'SubProcesses', Pdir), p)
            devnull = open(os.devnull,'w')
            subprocess.call(['make'],
                            stdout=devnull, stderr=devnull, 
                            cwd=os.path.join(mdir, 'SubProcesses',
                                             Pdir)) 
            self.assertTrue(os.path.exists(os.path.join(mdir,
                                                        'SubProcesses', Pdir,
                                                        'check')))
            #compute the matrix-element
            logfile = os.path.join(mdir,'SubProcesses', Pdir,
                                   'check.log')
            subprocess.call('./check', 
                            stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                            cwd=os.path.join(mdir, 'SubProcesses',
                                             Pdir), shell=True)
            log_output = open(logfile, 'r').read()
            misc.sprint(log_output)
            me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                               re.IGNORECASE)
            me_groups = me_re.findall(log_output) 
            dens_re =re.compile(r'value is\s+\d+\s*\(([\d\.eE\+-]+),([\d\.eE\+-]+)\)')
            density = dens_re.findall(log_output)

            hel_index = re.compile(
                           r'particle\s+\d\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*')
            index = hel_index.findall(log_output)
    
            if len(index) == len(density):
                all_index.append([[int(x[0]), int(x[1])] for x in index])
            else:
                curr_index = []
                for i in range(len(all_p), 2):
                    i1 = index[2*i]
                    i2 = index[2*i+1]
                    curr_index.append([[int(i1[0]), int(i1[1]), int(i2[0]), int(i2[1])]])
              
                all_index.append(curr_index)

            
            all_me.append(float(me_groups[0]))
            all_dens.append([complex(float(x[0]), float(x[1])) for x in density])

        import MadSpin.decay as madspin
        self.assertEqual(all_dens[1], []) 

        import numpy as np
        # array, nchanging, all_helicity_combinations, dimension):
        allow_hel = [-1,-1,   -1,0,   -1,1, 
                        0,-1,  0,0,   0,1, 
                        1,-1,  1,0,   1,1]
        
        #allow_hel = [-1,-1,-1,0,0,0,1,1,1,
        #             -1,0,1,-1,0,1,-1,0,1]

        misc.sprint(all_index[0])
        prod_dens = madspin.DensityMatrix(all_dens[0], 2, allow_hel, 9) 
        prod_dec1 = madspin.DensityMatrix(all_dens[2], 1, [-1,0,1], 3) 
        prod_dec2 = madspin.DensityMatrix(all_dens[3], 1, [-1,0,1], 3)  

        # GET_INTER divides each interference term by IDEN (initial-state spin
        # and colour averaging, including the identical-particle factor), so the
        # trace of each density matrix already equals the spin-averaged ME and
        # the IDEN normalisation that used to be applied here is now redundant.
        iden_prod = 72  # u u~ > z z : spin 4 * colour 9 * identical(ZZ) 2
        iden_dec = 3    # z > e+ e- : 3 Z helicity states

        self.assertAlmostEqual(prod_dec1.trace()/all_me[2],1,4)
        self.assertAlmostEqual(prod_dec2.trace()/all_me[3],1,4)
        self.assertAlmostEqual(prod_dens.trace()/all_me[0], 1,4)


        prod_dec = prod_dec1.tensor_product(prod_dec2)
        #self.assertNotEqual(str(prod_dec1), str(prod_dec2))
        #prod_dec_sym =prod_dec2.tensor_product(prod_dec1)
        mZ= 91.18800
        WZ = 2.44140
        nb_hel = 3*3
        symfact = 2 # 2 Z identical particles in the final state
        nb_spin = 2*2
        matrix = prod_dens.scalar_multiplication(prod_dec)/mZ**4/WZ**4/nb_hel/symfact/nb_spin
        #matrix_sym = prod_dens.scalar_multiplication(prod_dec_sym)/mZ**4/WZ**4/nb_hel/symfact/nb_spin

        misc.sprint(matrix, all_me[1], )
        # the three density matrices (production + 2 decays) each carry a 1/IDEN
        # factor that must be restored to match the unmodified full ME all_me[1]
        self.assertAlmostEqual(matrix * iden_prod * iden_dec**2 /all_me[1], 1,places=4)


    def test_standalone_density_dd(self):
        

        ############################################################################
        # Check convolution of density matrix with decay matrix 
        # reproduces the full matrix-element
        # testing case d d~ > z z, z > e+ e-
        ############################################################################
        self.do('generate d d~ > z z')
        self.do('output standalone_fortran %s_prod --prefix=int --density=3,4 -f ' % self.out_dir)
        self.do('generate d d~ > z z, z > e+ e-')
        self.do('output standalone_fortran %s_full -f ' % self.out_dir) 
        self.do('generate z > e+ e- --standalone') # --standalone allow mix 2>1 and 2>2 processes
        self.do('output standalone_fortran %s_dec1 --density=1 -f ' % self.out_dir)
        self.do('output standalone_fortran %s_dec2 --density=1 -f ' % self.out_dir)
        # Read a test event for u u~ > z z, z > e+ e-
        text_lhe = """<event>
        8      1 +9.3182000e+00 1.00474800e+02 7.54677100e-03 1.27930100e-01
       -1 -1    0    0    0  501 -0.0000000000e+00 +0.0000000000e+00 +4.4934420219e+01 4.4934420219e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
        1 -1    0    0  501    0 +0.0000000000e+00 -0.0000000000e+00 -2.2525427462e+02 2.2525427462e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
       23  2    1    2    0    0 -1.0803264452e+01 -4.0782658931e+01 -8.3249650133e+01 1.3048253287e+02 9.1188000000e+01 0.0000e+00 0.0000e+00
       23  2    1    2    0    0 +1.0803264452e+01 +4.0782658931e+01 -9.7070204270e+01 1.3970616197e+02 9.1188000000e+01 0.0000e+00 0.0000e+00
      -11  1    3    3    0    0 +1.0085745661e+01 +3.5438949841e+00 +1.6319184216e+01 1.9508901320e+01 0.0000000000e+00 0.0000e+00 -1.0000e+00
       11  1    3    3    0    0 -2.0889010113e+01 -4.4326553914e+01 -9.9568834346e+01 1.1097363155e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
      -11  1    4    4    0    0 +2.5139455461e+01 -6.3572870227e-01 +7.5662101007e+00 2.6261072087e+01 0.0000000000e+00 0.0000e+00 -1.0000e+00
       11  1    4    4    0    0 -1.4336191009e+01 +4.1418387636e+01 -1.0463641438e+02 1.1344508989e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
       </event>"""

        text_lhe ="""<event>
 8      1 +9.3182000e+00 1.24864100e+02 7.54677100e-03 1.23528200e-01
        1 -1    0    0  501    0 +0.0000000000e+00 +0.0000000000e+00 +1.1193143155e+02 1.1193143155e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
       -1 -1    0    0    0  501 -0.0000000000e+00 -0.0000000000e+00 -1.4479335429e+02 1.4479335429e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
       23  2    1    2    0    0 -3.9109420390e+01 -7.5804058190e+01 +8.5916956812e+00 1.2515938071e+02 9.1188000000e+01 0.0000e+00 0.0000e+00
       23  2    1    2    0    0 +3.9109420390e+01 +7.5804058190e+01 -4.1453618425e+01 1.3156540513e+02 9.1188000000e+01 0.0000e+00 0.0000e+00
      -11  1    3    3    0    0 -3.6688912923e+01 -2.9778691677e+01 -3.7238302752e+01 6.0162596365e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
       11  1    3    3    0    0 -2.4205074678e+00 -4.6025366514e+01 +4.5829998433e+01 6.4996784348e+01 0.0000000000e+00 0.0000e+00 -1.0000e+00
      -11  1    4    4    0    0 +5.2267279628e+01 +1.6179927919e+01 +4.8591069017e+00 5.4929677835e+01 0.0000000000e+00 0.0000e+00 -1.0000e+00
       11  1    4    4    0    0 -1.3157859241e+01 +5.9624130265e+01 -4.6312725324e+01 7.6635727286e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
       </event>"""
        Event = lhe_parser.Event()
        Event.parse(text_lhe.split('\n')) 
        # get the associate momenta for each matrix-element
        production_p = [Event[0], Event[1], Event[2], Event[3]]
        full_p = [Event[0], Event[1], Event[4], Event[5], Event[6], Event[7]]
        decay_z1 = [Event[2], Event[4], Event[5]]
        decay_z2 = [Event[3], Event[6], Event[7]]
        all_p = [production_p, full_p, decay_z1, decay_z2]


        all_me = []
        all_dens = []
        all_index= []
        # compiles the three directories:
        for i, (mdir, Pdir) in enumerate([(self.out_dir+'_prod', 'P0_ddx_zz'), 
                                 (self.out_dir+'_full', 'P0_ddx_zz_z_epem_z_epem'), 
                                 (self.out_dir+'_dec1', 'P0_z_epem'),
                                 (self.out_dir+'_dec2', 'P0_z_epem')]):
            p = all_p[i]
            self.edit_p_in_standalone(os.path.join(mdir, 'SubProcesses', Pdir), p)
            devnull = open(os.devnull,'w')
            subprocess.call(['make'],
                            stdout=devnull, stderr=devnull, 
                            cwd=os.path.join(mdir, 'SubProcesses',
                                             Pdir)) 
            self.assertTrue(os.path.exists(os.path.join(mdir,
                                                        'SubProcesses', Pdir,
                                                        'check')))
            #compute the matrix-element
            logfile = os.path.join(mdir,'SubProcesses', Pdir,
                                   'check.log')
            subprocess.call('./check', 
                            stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                            cwd=os.path.join(mdir, 'SubProcesses',
                                             Pdir), shell=True)
            log_output = open(logfile, 'r').read()
            misc.sprint(log_output)
            me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                               re.IGNORECASE)
            me_groups = me_re.findall(log_output) 
            dens_re =re.compile(r'value is\s+\d+\s*\(([\d\.eE\+-]+),([\d\.eE\+-]+)\)')
            density = dens_re.findall(log_output)

            hel_index = re.compile(
                           r'particle\s+\d\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*')
            index = hel_index.findall(log_output)
    
            if len(index) == len(density):
                all_index.append([[int(x[0]), int(x[1])] for x in index])
            else:
                curr_index = []
                for i in range(len(all_p), 2):
                    i1 = index[2*i]
                    i2 = index[2*i+1]
                    curr_index.append([[int(i1[0]), int(i1[1]), int(i2[0]), int(i2[1])]])
              
                all_index.append(curr_index)

            
            all_me.append(float(me_groups[0]))
            all_dens.append([complex(float(x[0]), float(x[1])) for x in density])

        import MadSpin.decay as madspin
        self.assertEqual(all_dens[1], []) 

        import numpy as np
        # array, nchanging, all_helicity_combinations, dimension):
        allow_hel = [-1,-1,   -1,0,   -1,1, 
                      0,-1,    0,0,    0,1, 
                      1,-1,    1,0,    1,1] #double checked with fortran code.
        

        prod_dens = madspin.DensityMatrix(all_dens[0], 2, allow_hel, 9) 
        prod_dec1 = madspin.DensityMatrix(all_dens[2], 1, [-1,0, 1], 3) 
        prod_dec2 = madspin.DensityMatrix(all_dens[3], 1, [-1,0, 1], 3)


        #consistency of the matrix-element and the density matrix
        # GET_INTER divides each interference term by IDEN (initial-state spin
        # and colour averaging, including the identical-particle factor), so the
        # trace of each density matrix already equals the spin-averaged ME and
        # the IDEN normalisation that used to be applied here is now redundant.
        iden_prod = 72  # d d~ > z z : spin 4 * colour 9 * identical(ZZ) 2
        iden_dec = 3    # z > e+ e- : 3 Z helicity states

        self.assertAlmostEqual(prod_dec1.trace()/all_me[2],1,4)
        self.assertAlmostEqual(prod_dec2.trace()/all_me[3],1,4)
        self.assertAlmostEqual(prod_dens.trace()/all_me[0],1,4)

        prod_dec =prod_dec1.tensor_product(prod_dec2)
        prod_dec_sym =prod_dec2.tensor_product(prod_dec1)
        mZ= 91.18800
        WZ = 2.44140
        nb_hel = 3*3
        symfact = 2 # 2 Z identical particles in the final state
        nb_spin = 2*2
        matrix = prod_dens.scalar_multiplication(prod_dec)/mZ**4/WZ**4/nb_hel/symfact/nb_spin
        #matrix_sym = prod_dens.scalar_multiplication(prod_dec_sym)/mZ**4/WZ**4/nb_hel/symfact/nb_spin

        misc.sprint(matrix, all_me[1])
        #misc.sprint(matrix/all_me[1], matrix_sym/all_me[1])

        # the three density matrices (production + 2 decays) each carry a 1/IDEN
        # factor that must be restored to match the unmodified full ME all_me[1]
        self.assertAlmostEqual(matrix * iden_prod * iden_dec**2 /all_me[1],1,places=4)
        #self.assertAlmostEqual(matrix_sym, all_me[1],4)

        #check how madspin build the full event:
        prod_lhe = """<event>
        4      1 +9.3182000e+00 1.24864100e+02 7.54677100e-03 1.23528200e-01
        1 -1    0    0  501    0 +0.0000000000e+00 +0.0000000000e+00 +1.1193143155e+02 1.1193143155e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
       -1 -1    0    0    0  501 -0.0000000000e+00 -0.0000000000e+00 -1.4479335429e+02 1.4479335429e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
       23  1    1    2    0    0 -3.9109420390e+01 -7.5804058190e+01 +8.5916956812e+00 1.2515938071e+02 9.1188000000e+01 0.0000e+00 -1.0000e+00
       23  1    1    2    0    0 +3.9109420390e+01 +7.5804058190e+01 -4.1453618425e+01 1.3156540513e+02 9.1188000000e+01 0.0000e+00 1.0000e+00
       </event>"""
        prod_Event = lhe_parser.Event()
        prod_Event.parse(prod_lhe.split('\n'))
        prod_p = [prod_Event[0], prod_Event[1], prod_Event[2], prod_Event[3]]
        
        for p1, pe in zip(production_p, prod_p):
                self.assertAlmostEqual(p1.px, pe.px)
                self.assertAlmostEqual(p1.py, pe.py)
                self.assertAlmostEqual(p1.pz, pe.pz)
                self.assertAlmostEqual(p1.E, pe.E)
                #self.assertEqual(p1.id, pe.id)
    
        dec1_lhe = """ <event>
 3      0 +8.3965000e-02 9.11880000e+01 7.54677100e-03 1.30000000e-01
       23 -1    0    0    0    0 -3.9109420391e+01 -7.5804058191e+01 +8.5916956813e+00 1.2515938071e+02 9.1188000000e+01 0.0000e+00 -1.0000e+00
      -11  1    1    0    0    0 -3.6688912923e+01 -2.9778691677e+01 -3.7238302752e+01 6.0162596365e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
       11  1    1    0    0    0 -2.4205074678e+00 -4.6025366514e+01 +4.5829998433e+01 6.4996784348e+01 0.0000000000e+00 0.0000e+00 -1.0000e+00
</event>"""

        dec1_Event = lhe_parser.Event()
        dec1_Event.parse(dec1_lhe.split('\n'))
        dec1_p = [dec1_Event[0], dec1_Event[1], dec1_Event[2]]
        for p1, pe in zip(decay_z1, dec1_p):
                self.assertAlmostEqual(p1.px, pe.px)
                self.assertAlmostEqual(p1.py, pe.py)
                self.assertAlmostEqual(p1.pz, pe.pz)
                self.assertAlmostEqual(p1.E, pe.E)
        self.assertAlmostEqual(decay_z1[0].px, prod_p[2].px)

        dec2_lhe = """<event>
 3      0 +8.3965000e-02 9.11880000e+01 7.54677100e-03 1.30000000e-01
       23 -1    0    0    0    0 +3.9109420387e+01 +7.5804058185e+01 -4.1453618422e+01 1.3156540512e+02 9.1188000000e+01 0.0000e+00 0.0000e+00
      -11  1    1    0    0    0 +5.2267279628e+01 +1.6179927919e+01 +4.8591069017e+00 5.4929677835e+01 0.0000000000e+00 0.0000e+00 -1.0000e+00
       11  1    1    0    0    0 -1.3157859241e+01 +5.9624130265e+01 -4.6312725324e+01 7.6635727286e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
</event>"""
                
        dec2_Event = lhe_parser.Event()
        dec2_Event.parse(dec2_lhe.split('\n'))
        dec2_p = [dec2_Event[0], dec2_Event[1], dec2_Event[2]]
        for p1, pe in zip(decay_z2, dec2_p):
                self.assertAlmostEqual(p1.px, pe.px)
                self.assertAlmostEqual(p1.py, pe.py)
                self.assertAlmostEqual(p1.pz, pe.pz)
                self.assertAlmostEqual(p1.E, pe.E)
        self.assertAlmostEqual(decay_z2[0].px, prod_p[3].px)


        # below number are computed by madspin...
        #prod_diag = 0.009896473385096114 , dec_diag = 148120.70052380403
        self.assertAlmostEqual(all_me[0]/0.009896473385096114,1, places=6)
        self.assertAlmostEqual(all_me[2]*all_me[3]/148120.70052380403,1, places=6)

        # check with printed version of madspin:
        # information from madspin log
        #    self.assertEqual(str(prod_dens),
        #    density_prod = [([-1, -1, -1, -1],  0.00155656-4.44342280e-21j)
        madspin_report= [([-1, -1, -1, -1],  0.00155656-4.44342280e-21j),
 ([-1, -1, -1,  0], -0.0061166 +3.18998398e-18j),
 ([-1, -1, -1,  1], -0.02385528+5.26004027e-18j),
 ([-1,  0, -1, -1],  0.00845586-1.72478531e-19j),
 ([-1,  0, -1,  0],  0.00353157-1.68518580e-18j),
 ([-1,  0, -1,  1], -0.01006808-3.31784584e-18j),
 ([-1,  1, -1, -1],  0.01352631-2.15122174e-18j),
 ([-1,  1, -1,  0],  0.01119138-1.93698754e-18j),
 ([-1,  1, -1,  1],  0.00186626-1.47534714e-18j),
 ([-1, -1,  0,  0],  0.02404441+8.32488915e-20j),
 ([-1, -1,  0,  1],  0.09364523+2.74460430e-17j),
 ([-1,  0,  0, -1], -0.03322776-1.67756927e-17j),
 ([-1,  0,  0,  0], -0.01387597-8.33384364e-19j),
 ([-1,  0,  0,  1],  0.039555  +3.40446067e-17j),
 ([-1,  1,  0, -1], -0.05306146-2.09403634e-17j),
 ([-1,  1,  0,  0], -0.04396074-1.67907887e-17j),
 ([-1,  1,  0,  1], -0.00733204+1.88201291e-18j),
 ([-1, -1,  1,  1],  0.3666445 +1.01936527e-18j),
 ([-1,  0,  1, -1], -0.1295935 -2.93067477e-17j),
 ([-1,  0,  1,  0], -0.05414116+1.18078428e-17j),
 ([-1,  0,  1,  1],  0.1543898 +8.85097549e-17j),
 ([-1,  1,  1, -1], -0.20829503-1.77720466e-17j),
 ([-1,  1,  1,  0], -0.1716968 -1.55059258e-17j),
 ([-1,  1,  1,  1], -0.02861856+1.56582097e-17j),
 ([ 0,  0, -1, -1],  0.04593575-6.65039823e-19j),
 ([ 0,  0, -1,  0],  0.01918499-8.80142009e-18j),
 ([ 0,  0, -1,  1], -0.0546941 -1.81837989e-17j),
 ([ 0,  1, -1, -1],  0.07348216-1.08546983e-17j),
 ([ 0,  1, -1,  0],  0.06079653-4.54388136e-18j),
 ([ 0,  1, -1,  1],  0.01013831-7.16399374e-18j),
 ([ 0,  0,  0,  0],  0.00801285-2.12259922e-19j),
 ([ 0,  0,  0,  1], -0.0228443 -1.87071346e-17j),
 ([ 0,  1,  0, -1],  0.03070557+9.95960116e-18j),
 ([ 0,  1,  0,  0],  0.02539443+7.89596503e-18j),
 ([ 0,  1,  0,  1],  0.00423451-1.40272182e-18j),
 ([ 0,  0,  1,  1],  0.06512972+1.75433965e-18j),
 ([ 0,  1,  1, -1], -0.08757587+4.34771986e-17j),
 ([ 0,  1,  1,  0], -0.07240333+3.17362500e-17j),
 ([ 0,  1,  1,  1], -0.01207272+1.30660225e-17j),
 ([ 1,  1, -1, -1],  0.11848886+3.00231717e-18j),
 ([ 1,  1, -1,  0],  0.09742438+2.01030469e-18j),
 ([ 1,  1, -1,  1],  0.01623366-9.84227552e-18j),
 ([ 1,  1,  0,  0],  0.08049559-9.13881031e-19j),
 ([ 1,  1,  0,  1],  0.01342101-8.17624195e-18j),
 ([ 1,  1,  1,  1],  0.00223785-8.42004594e-21j),
 ([-1, -1,  0, -1], -0.0061166 -3.18998398e-18j),
 ([-1, -1,  1, -1], -0.02385528-5.26004027e-18j),
 ([ 0, -1, -1, -1],  0.00845586+1.72478531e-19j),
 ([ 0, -1,  0, -1],  0.00353157+1.68518580e-18j),
 ([ 0, -1,  1, -1], -0.01006808+3.31784584e-18j),
 ([ 1, -1, -1, -1],  0.01352631+2.15122174e-18j),
 ([ 1, -1,  0, -1],  0.01119138+1.93698754e-18j),
 ([ 1, -1,  1, -1],  0.00186626+1.47534714e-18j),
 ([-1, -1,  1,  0],  0.09364523-2.74460430e-17j),
 ([ 0, -1, -1,  0], -0.03322776+1.67756927e-17j),
 ([ 0, -1,  0,  0], -0.01387597+8.33384364e-19j),
 ([ 0, -1,  1,  0],  0.039555  -3.40446067e-17j),
 ([ 1, -1, -1,  0], -0.05306146+2.09403634e-17j),
 ([ 1, -1,  0,  0], -0.04396074+1.67907887e-17j),
 ([ 1, -1,  1,  0], -0.00733204-1.88201291e-18j),
 ([ 0, -1, -1,  1], -0.1295935 +2.93067477e-17j),
 ([ 0, -1,  0,  1], -0.05414116-1.18078428e-17j),
 ([ 0, -1,  1,  1],  0.1543898 -8.85097549e-17j),
 ([ 1, -1, -1,  1], -0.20829503+1.77720466e-17j),
 ([ 1, -1,  0,  1], -0.1716968 +1.55059258e-17j),
 ([ 1, -1,  1,  1], -0.02861856-1.56582097e-17j),
 ([ 0,  0,  0, -1],  0.01918499+8.80142009e-18j),
 ([ 0,  0,  1, -1], -0.0546941 +1.81837989e-17j),
 ([ 1,  0, -1, -1],  0.07348216+1.08546983e-17j),
 ([ 1,  0,  0, -1],  0.06079653+4.54388136e-18j),
 ([ 1,  0,  1, -1],  0.01013831+7.16399374e-18j),
 ([ 0,  0,  1,  0], -0.0228443 +1.87071346e-17j),
 ([ 1,  0, -1,  0],  0.03070557-9.95960116e-18j),
 ([ 1,  0,  0,  0],  0.02539443-7.89596503e-18j),
 ([ 1,  0,  1,  0],  0.00423451+1.40272182e-18j),
 ([ 1,  0, -1,  1], -0.08757587-4.34771986e-17j),
 ([ 1,  0,  0,  1], -0.07240333-3.17362500e-17j),
 ([ 1,  0,  1,  1], -0.01207272-1.30660225e-17j),
 ([ 1,  1,  0, -1],  0.09742438-2.01030469e-18j),
 ([ 1,  1,  1, -1],  0.01623366+9.84227552e-18j),
 ([ 1,  1,  1,  0],  0.01342101+8.17624195e-18j)]
        madspin_report_dict = dict(((tuple(x), y) for x,y in madspin_report))

        # madspin_report holds the (pre-IDEN) density values reported by madspin;
        # the standalone prod_dens now carries the 1/IDEN normalisation from
        # GET_INTER, so we restore iden_prod (resp. iden_dec) when comparing.
        for key in madspin_report_dict:
            ref_val = self._dens_value_for_key(prod_dens, key)
            self.assertAlmostEqual(madspin_report_dict[key].real/(ref_val.real * iden_prod), 1, places=4)


        madspin_report = [([-1, -1], 296.70587 -7.1793691e-15j),
                          ([-1,  0], 102.16883 +4.6782249e+01j),
                          ([-1,  1], 187.98897 +2.1782800e+02j),
                         ([ 0,  0], 575.4612  -8.4577865e-16j),
                         ([ 0,  1],  60.377983+2.7646572e+01j),
                         ([ 1,  1], 282.4265  +2.6328346e-15j),
                         ([ 0, -1], 102.16883 -4.6782249e+01j),
                         ([ 1, -1], 187.98897 -2.1782800e+02j),
                         ([ 1,  0],  60.377983-2.7646572e+01j),]
        madspin_report_dict = dict(((tuple(x), y) for x,y in madspin_report))

        for key in madspin_report_dict:
            ref_val = self._dens_value_for_key(prod_dec1, key)
            self.assertAlmostEqual(madspin_report_dict[key].real/(ref_val.real * iden_dec), 1, places=4)

        madspin_report =[([-1, -1],  332.7482   -3.3880889e-16j),
                        ([-1,  0],  -84.79217  +1.5662439e+02j),
                        ([-1,  1], -149.53477  -2.2903455e+02j),
                        ([ 0,  0],  547.05566  -6.0317979e-15j),
                        ([ 0,  1],    1.8067838-3.3374121e+00j),
                        ([ 1,  1],  274.7897   -1.2771769e-15j),
                        ([ 0, -1],  -84.79217  -1.5662439e+02j),
                        ([ 1, -1], -149.53477  +2.2903455e+02j),
                        ([ 1,  0],    1.8067838+3.3374121e+00j),]
        madspin_report_dict = dict(((tuple(x), y) for x,y in madspin_report))

        for key in madspin_report_dict:
            ref_val = self._dens_value_for_key(prod_dec2, key)
            self.assertAlmostEqual(madspin_report_dict[key].real/(ref_val.real * iden_dec), 1, places=4)


        madspin_report = [([-1, -1, -1, -1],  9.8728344e+04-2.4894488e-12j),
                            ([-1, -1, -1,  0], -2.5158334e+04+4.6471375e+04j),
                            ([-1, -1, -1,  1], -4.4367844e+04-6.7955898e+04j),
                            ([-1, -1,  0,  0],  1.6231462e+05-5.7171845e-12j),
                            ([-1, -1,  0,  1],  5.3608337e+02-9.9022980e+02j),
                            ([-1, -1,  1,  1],  8.1531719e+04-2.3517624e-12j),
                            ([-1, -1,  0, -1], -2.5158334e+04-4.6471375e+04j),
                            ([-1, -1,  1, -1], -4.4367844e+04+6.7955898e+04j),
                            ([-1, -1,  1,  0],  5.3608337e+02+9.9022980e+02j),
                            ([-1,  0, -1, -1],  3.3996496e+04+1.5566709e+04j),
                            ([-1,  0, -1,  0], -1.5990357e+04+1.2035362e+04j),
                            ([-1,  0, -1,  1], -4.5630420e+03-3.0395766e+04j),
                            ([-1,  0,  0,  0],  5.5892039e+04+2.5592494e+04j),
                            ([-1,  0,  0,  1],  3.4072864e+02-2.5645407e+02j),
                            ([-1,  0,  1,  1],  2.8074943e+04+1.2855280e+04j),
                            ([-1,  0,  0, -1], -1.3358750e+03-1.9968898e+04j),
                            ([-1,  0,  1, -1], -2.5992543e+04+1.6404617e+04j),
                            ([-1,  0,  1,  0],  2.8465332e+01+4.2550491e+02j),
                            ([-1,  1, -1, -1],  6.2552992e+04+7.2481875e+04j),
                            ([-1,  1, -1,  0], -5.0057172e+04+1.0973549e+04j),
                            ([-1,  1, -1,  1],  2.1779248e+04-7.5628828e+04j),
                            ([-1,  1,  0,  0],  1.0284043e+05+1.1916404e+05j),
                            ([-1,  1,  0,  1],  1.0666372e+03-2.3382855e+02j),
                            ([-1,  1,  1,  1],  5.1657434e+04+5.9856891e+04j),
                            ([-1,  1,  0, -1],  1.8177188e+04-4.7913766e+04j),
                            ([-1,  1,  1, -1], -7.8001023e+04+1.0483107e+04j),
                            ([-1,  1,  1,  0], -3.8732639e+02+1.0209648e+03j),
                            ([ 0,  0, -1, -1],  1.9148367e+05-4.7640266e-13j),
                            ([ 0,  0, -1,  0], -4.8794602e+04+9.0131258e+04j),
                            ([ 0,  0, -1,  1], -8.6051461e+04-1.3180048e+05j),
                            ([ 0,  0,  0,  0],  3.1480931e+05-3.9337535e-12j),
                            ([ 0,  0,  0,  1],  1.0397339e+03-1.9205511e+03j),
                            ([ 0,  0,  1,  1],  1.5813081e+05-9.6737700e-13j),
                            ([ 0,  0,  0, -1], -4.8794602e+04-9.0131258e+04j),
                            ([ 0,  0,  1, -1], -8.6051461e+04+1.3180048e+05j),
                            ([ 0,  0,  1,  0],  1.0397339e+03+1.9205511e+03j),
                            ([ 0,  1, -1, -1],  2.0090666e+04+9.1993467e+03j),
                            ([ 0,  1, -1,  0], -9.4497070e+03+7.1124521e+03j),
                            ([ 0,  1, -1,  1], -2.6965884e+03-1.7962768e+04j),
                            ([ 0,  1,  0,  0],  3.3030117e+04+1.5124214e+04j),
                            ([ 0,  1,  0,  1],  2.0135797e+02-1.5155484e+02j),
                            ([ 0,  1,  1,  1],  1.6591248e+04+7.5969932e+03j),
                            ([ 0,  1,  0, -1], -7.8945264e+02-1.1800878e+04j),
                            ([ 0,  1,  1, -1], -1.5360629e+04+9.6945195e+03j),
                            ([ 0,  1,  1,  0],  1.6821953e+01+2.5145758e+02j),
                            ([ 1,  1, -1, -1],  9.3976914e+04+7.8038238e-13j),
                            ([ 1,  1, -1,  0], -2.3947557e+04+4.4234879e+04j),
                            ([ 1,  1, -1,  1], -4.2232586e+04-6.4685430e+04j),
                            ([ 1,  1,  0,  0],  1.5450303e+05-2.6323258e-13j),
                            ([ 1,  1,  0,  1],  5.1028366e+02-9.4257367e+02j),
                            ([ 1,  1,  1,  1],  7.7607898e+04+3.6276724e-13j),
                            ([ 1,  1,  0, -1], -2.3947557e+04-4.4234879e+04j),
                            ([ 1,  1,  1, -1], -4.2232586e+04+6.4685430e+04j),
                            ([ 1,  1,  1,  0],  5.1028366e+02+9.4257367e+02j),
                            ([ 0, -1, -1, -1],  3.3996496e+04-1.5566709e+04j),
                            ([ 0, -1, -1,  0], -1.3358750e+03+1.9968898e+04j),
                            ([ 0, -1, -1,  1], -2.5992543e+04-1.6404617e+04j),
                            ([ 0, -1,  0,  0],  5.5892039e+04-2.5592494e+04j),
                            ([ 0, -1,  0,  1],  2.8465332e+01-4.2550491e+02j),
                            ([ 0, -1,  1,  1],  2.8074943e+04-1.2855280e+04j),
                            ([ 0, -1,  0, -1], -1.5990357e+04-1.2035362e+04j),
                            ([ 0, -1,  1, -1], -4.5630420e+03+3.0395766e+04j),
                            ([ 0, -1,  1,  0],  3.4072864e+02+2.5645407e+02j),
                            ([ 1, -1, -1, -1],  6.2552992e+04-7.2481875e+04j),
                            ([ 1, -1, -1,  0],  1.8177188e+04+4.7913766e+04j),
                            ([ 1, -1, -1,  1], -7.8001023e+04-1.0483107e+04j),
                            ([ 1, -1,  0,  0],  1.0284043e+05-1.1916404e+05j),
                            ([ 1, -1,  0,  1], -3.8732639e+02-1.0209648e+03j),
                            ([ 1, -1,  1,  1],  5.1657434e+04-5.9856891e+04j),
                            ([ 1, -1,  0, -1], -5.0057172e+04-1.0973549e+04j),
                            ([ 1, -1,  1, -1],  2.1779248e+04+7.5628828e+04j),
                            ([ 1, -1,  1,  0],  1.0666372e+03+2.3382855e+02j),
                            ([ 1,  0, -1, -1],  2.0090666e+04-9.1993467e+03j),
                            ([ 1,  0, -1,  0], -7.8945264e+02+1.1800878e+04j),
                            ([ 1,  0, -1,  1], -1.5360629e+04-9.6945195e+03j),
                            ([ 1,  0,  0,  0],  3.3030117e+04-1.5124214e+04j),
                            ([ 1,  0,  0,  1],  1.6821953e+01-2.5145758e+02j),
                            ([ 1,  0,  1,  1],  1.6591248e+04-7.5969932e+03j),
                            ([ 1,  0,  0, -1], -9.4497070e+03-7.1124521e+03j),
                            ([ 1,  0,  1, -1], -2.6965884e+03+1.7962768e+04j),
                            ([ 1,  0,  1,  0],  2.0135797e+02+1.5155484e+02j),]
        madspin_report_dict = dict(((tuple(x), y) for x,y in madspin_report))

        # prod_dec is the tensor product of the two decay density matrices, so it
        # carries iden_dec**2 from the GET_INTER normalisation.
        for key in madspin_report_dict:
            ref_val = self._dens_value_for_key(prod_dec, key)
            self.assertAlmostEqual(madspin_report_dict[key].real/(ref_val.real * iden_dec**2), 1, places=4)
                                                                             
                                                                             
    # Run in a fresh interpreter: every output names its module all_matrix2py,
    # so it must not share a process with one an earlier test imported.
    _get_all_inter_script = '''
import itertools, json, math, os, sys
import numpy as np
import all_matrix2py as m
mdir, prefix, outfile = sys.argv[1:]
fct = lambda name: getattr(m, prefix + name)
fct('initialise')(os.path.join(mdir, 'Cards', 'param_card.dat'))
E, th, ph = 500., 0.7, 0.3
k = [E*math.sin(th)*math.cos(ph), E*math.sin(th)*math.sin(ph), E*math.cos(th)]
p = np.array([[E, 0, 0, E], [E, 0, 0, -E], [E] + k,
              [E] + [-x for x in k]]).T.copy()
kw = dict(pdgs=[2, -2, -11, 11], procid=-1, p=p, pos=[3, 4],
          allow_hel=[-1, -1, -1, 1, 1, -1, 1, 1])
cplx = lambda a: [[z.real, z.imag] for z in a]
res = {'density': cplx(fct('py_get_density')(alphas=0., scale2=0., **kw)),
       'inter': [], 'nhel': []}
for h1, h2 in itertools.product([-1, 1], repeat=2):
    # the entries at pos are ignored: 0 shows whether they were written to
    nhel = np.array([h1, h2, 0, 0], dtype=np.int32)
    res['inter'].append(cplx(fct('py_get_all_inter')(nhel=nhel, **kw)))
    res['nhel'].append([nhel.tolist(), [h1, h2, 0, 0]])
json.dump(res, open(outfile, 'w'))
'''

    def test_standalone_f2py_get_all_inter(self):
        """py_get_all_inter, summed over the helicities of the legs not in
        pos, must give py_get_density.

        It could not be called at all: f2py gave INTER the dimension (*), so
        every call raised a ValueError; the dispatch in all_matrix.f did not
        pass the NHEL the per-process GET_ALL_INTER takes; and with
        --prefixf2py the wrapper called the unprefixed f77_get_all_inter, so
        the whole module failed to import.
        """
        import json
        self.do('import model sm')
        self.do('generate p p > e+ e-')
        prefixed_dir = self.out_dir + '_prefixf2py'
        self.addCleanup(shutil.rmtree, prefixed_dir, ignore_errors=True)
        for out_dir, option, prefix in [(self.out_dir, '', ''),
                                        (prefixed_dir, ' --prefixf2py=1', 'f1_')]:
            self.do('output standalone_fortran %s --prefix=int -f%s'
                    % (out_dir, option))
            subdir = os.path.join(out_dir, 'SubProcesses')
            build = subprocess.run(['make', 'f2py'], cwd=subdir,
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0,
                             'make f2py failed:\n' + build.stdout[-2000:]
                             + build.stderr[-2000:])

            outfile = os.path.join(out_dir, 'get_all_inter.json')
            run = subprocess.run([sys.executable, '-c',
                                  self._get_all_inter_script,
                                  out_dir, prefix, outfile],
                                 cwd=subdir, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0,
                             run.stdout[-2000:] + run.stderr[-2000:])
            res = json.load(open(outfile))

            # f2py hands the int32 array straight through, and GET_ALL_INTER
            # writes the pos entries of the NHEL it is given
            for after, before in res['nhel']:
                self.assertEqual(after, before)

            dens = res['density']
            scale = max(abs(complex(*z)) for z in dens)
            self.assertGreater(scale, 0)
            # more than one helicity of the other legs contributes, so the
            # sum below does not reduce to a single call
            self.assertGreaterEqual(
                sum(any(re_ or im_ for re_, im_ in inter)
                    for inter in res['inter']), 2)
            for i, (re_, im_) in enumerate(dens):
                self.assertAlmostEqual(sum(inter[i][0] for inter in res['inter']),
                                       re_, delta=1e-12 * scale)
                self.assertAlmostEqual(sum(inter[i][1] for inter in res['inter']),
                                       im_, delta=1e-12 * scale)

    def test_standalone_density_f2py(self):       

        ############################################################################
        # Check convolution of density matrix with decay matrix 
        # reproduces the full matrix-element
        # testing case d d~ > z z, z > e+ e-
        ############################################################################
        self.do('generate d d~ > z z')
        self.do('output standalone_fortran %s_prod --prefix=int --density=3,4 -f ' % self.out_dir)
        self.do('generate d d~ > z z, z > e+ e-')
        #self.do('output standalone_fortran %s_full -f ' % self.out_dir) 
        #self.do('generate z > e+ e- --standalone') # --standalone allow mix 2>1 and 2>2 processes
        #self.do('output standalone_fortran %s_dec1 --density=1 -f ' % self.out_dir)
        #self.do('output standalone_fortran %s_dec2 --density=1 -f ' % self.out_dir)
        # Read a test event for u u~ > z z, z > e+ e-
        text_lhe = """<event>
        8      1 +9.3182000e+00 1.00474800e+02 7.54677100e-03 1.27930100e-01
       -1 -1    0    0    0  501 -0.0000000000e+00 +0.0000000000e+00 +4.4934420219e+01 4.4934420219e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
        1 -1    0    0  501    0 +0.0000000000e+00 -0.0000000000e+00 -2.2525427462e+02 2.2525427462e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
       23  2    1    2    0    0 -1.0803264452e+01 -4.0782658931e+01 -8.3249650133e+01 1.3048253287e+02 9.1188000000e+01 0.0000e+00 0.0000e+00
       23  2    1    2    0    0 +1.0803264452e+01 +4.0782658931e+01 -9.7070204270e+01 1.3970616197e+02 9.1188000000e+01 0.0000e+00 0.0000e+00
      -11  1    3    3    0    0 +1.0085745661e+01 +3.5438949841e+00 +1.6319184216e+01 1.9508901320e+01 0.0000000000e+00 0.0000e+00 -1.0000e+00
       11  1    3    3    0    0 -2.0889010113e+01 -4.4326553914e+01 -9.9568834346e+01 1.1097363155e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
      -11  1    4    4    0    0 +2.5139455461e+01 -6.3572870227e-01 +7.5662101007e+00 2.6261072087e+01 0.0000000000e+00 0.0000e+00 -1.0000e+00
       11  1    4    4    0    0 -1.4336191009e+01 +4.1418387636e+01 -1.0463641438e+02 1.1344508989e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
       </event>"""

        Event = lhe_parser.Event()
        Event.parse(text_lhe.split('\n')) 
        # get the associate momenta for each matrix-element
        production_p = [Event[0], Event[1], Event[2], Event[3]]
        #full_p = [Event[0], Event[1], Event[4], Event[5], Event[6], Event[7]]
        #decay_z1 = [Event[2], Event[4], Event[5]]
        #decay_z2 = [Event[3], Event[6], Event[7]]
        #all_p = [production_p, full_p, decay_z1, decay_z2]

        mdir = self.out_dir+'_prod'
        Pdir = 'P0_ddx_zz'
        p = production_p
        
        # start to do standalone fortran for comparison
        self.edit_p_in_standalone(os.path.join(mdir, 'SubProcesses', Pdir), p)
        devnull = open(os.devnull,'w')
        subprocess.call(['make'],
                            stdout=devnull, stderr=devnull, 
                            cwd=os.path.join(mdir, 'SubProcesses',
                                             Pdir)) 
        self.assertTrue(os.path.exists(os.path.join(mdir,
                                                        'SubProcesses', Pdir,
                                                        'check')))
        #compute the matrix-element
        logfile = os.path.join(mdir,'SubProcesses', Pdir,
                                   'check.log')
        subprocess.call('./check', 
                            stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                            cwd=os.path.join(mdir, 'SubProcesses',
                                             Pdir), shell=True)
        log_output = open(logfile, 'r').read()
        misc.sprint(log_output)
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.findall(log_output) 
        dens_re =re.compile(r'value is\s+\d+\s*\(([\d\.eE\+-]+),([\d\.eE\+-]+)\)')
        density = dens_re.findall(log_output)

        hel_index = re.compile(
                       r'particle\s+\d\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*')
        index = hel_index.findall(log_output)
        all_index = []
        if len(index) == len(density):
            all_index.append([[int(x[0]), int(x[1])] for x in index])
        else:
            curr_index = []
            for i in range(4, 2):
                i1 = index[2*i]
                i2 = index[2*i+1]
                curr_index.append([[int(i1[0]), int(i1[1]), int(i2[0]), int(i2[1])]])
          
            all_index.append(curr_index)

        fortran_me = float(me_groups[0])
        fortran_dens = [complex(float(x[0]), float(x[1])) for x in density]    

        # Do the computation via f2py linking
        sys.path.insert(0, os.path.join(mdir, 'SubProcesses', Pdir))
        make_rc = subprocess.call(['make', 'matrix2py.so'],
                            #stdout=devnull, stderr=devnull,
                            cwd=os.path.join(mdir, 'SubProcesses',
                                             Pdir))
        so_path = os.path.join(mdir, 'SubProcesses', Pdir, 'matrix2py.so')
        self.assertEqual(make_rc, 0,
            'make matrix2py.so failed (rc=%s); check that f2py is configured.' % make_rc)
        self.assertTrue(os.path.exists(so_path),
            'matrix2py.so was not produced at %s' % so_path)

        # Force re-import in case a stale module from another test is cached
        if 'matrix2py' in sys.modules:
            del sys.modules['matrix2py']
        import matrix2py
        #os.chdir(os.path.join(mdir, 'SubProcesses', Pdir))
        with misc.chdir(os.path.join(mdir, 'SubProcesses', Pdir)):
            matrix2py.py_m0_initialisemodel('../../Cards/param_card.dat')

            p = [[x.E, x.px, x.py, x.pz] for x in p]
            P =self.invert_momenta(p)
            alphas = 0.118
            nhel = -1 # means sum over all helicity
            # GET_value / SMATRIX now take a per-particle merged-flavor index
            # array (matches the flavor-grouping plumbing). Single-flavor
            # processes use 1 for every particle.
            flavor_value = [1] * len(P[0])
            me2 = matrix2py.py_m0_get_value(P, alphas, nhel, flavor_value)
            misc.sprint('fortran: ', fortran_me, ' f2py: ', me2)
            # compute density matrix
            self.assertAlmostEqual(fortran_me/me2, 1., places=5)

            pos = [3,4] # particle to get density matrix
            n_changing = 2 # why needed in f2py ?
            allow_hel = [-1,-1,   -1,0,   -1,1, 
                          0,-1,    0,0,    0,1, 
                          1,-1,    1,0,    1,1]
            ncomb = 9 # why needed in f2py ?
            alphas = 0.118 # no impact for ZZ
            scale2 = 0. # as passed by check_sa.f; no impact for ZZ

            # GET_DENSITY now takes a per-particle merged-flavor index array
            # (matches the SMATRIX/GET_AMP flavor-grouping plumbing).
            # Single-flavor processes use 1 for every particle.
            flavor = [1] * len(P[0])
            f2py_dens = matrix2py.py_m0_get_density(P, pos, n_changing, allow_hel, ncomb, flavor, alphas, scale2)
            misc.sprint('fortran: ', fortran_dens)
            misc.sprint('f2py:    ', f2py_dens)
            for i in range(9*5):
                misc.sprint(i, fortran_dens[i], f2py_dens[i])
                self.assertAlmostEqual(fortran_dens[i].real/f2py_dens[i].real, 1, places=3)
                self.assertAlmostEqual(fortran_dens[i].imag, f2py_dens[i].imag, places=5)
            import MadSpin.decay as madspin
            density_matrix = madspin.DensityMatrix(f2py_dens, n_changing, allow_hel, ncomb)
            # GET_INTER now divides by IDEN (9 colour * 4 spin * 2 ZZ identical = 72),
            # so the trace already equals the spin-averaged ME fortran_me.
            self.assertAlmostEqual(density_matrix.trace()/fortran_me, 1,4)
            misc.sprint(density_matrix.values[1], fortran_dens[1])

    def test_madmatrix_density_matches_fortran(self):
        """The madmatrix density matrices and |M|^2 (madmatrix.density, what
        MadSpin uses by default) against the Fortran f2py ones MadSpin used
        before, on the processes MadSpin generates for p p > t t~ with both tops
        decayed: off-shell ('*') tops, merged quark and lepton flavors, both
        quark orderings, on- and off-shell momenta."""
        import importlib.util
        import math
        import random
        import numpy as np
        import madmatrix.density as madmatrix_density

        os.makedirs(self.out_dir)
        self.do('set apply_flavor_grouping True')
        self.do('set group_subprocesses False')
        dirs = {}
        for name, process in (('prod', 'generate p p > t* t~*'),
                              ('dec', 'generate t* > b w+, w+ > l+ vl')):
            self.do(process)
            dirs[name] = (pjoin(self.out_dir, 'mm_' + name), pjoin(self.out_dir, 'f_' + name))
            self.do('output standalone %s -f --noeps=True' % dirs[name][0])
            self.do('output standalone_fortran %s --prefix=int --density=1 -f' % dirs[name][1])

        libtag = '_acc%d' % os.getpid()
        fortran, mm = {}, {}
        for menum, name in ((2, 'prod'), (1, 'dec')):
            mm_dir, f_dir = dirs[name]
            misc.compile(madmatrix_density.make_variables(libtag=libtag + name),
                         cwd=pjoin(mm_dir, 'SubProcesses'), mode='cpp', nb_core=4)
            mm[name] = madmatrix_density.MadDensityLibrary(mm_dir)
            mm[name].initialise(pjoin(mm_dir, 'Cards', 'param_card.dat'))
            misc.compile(cwd=pjoin(f_dir, 'Source'), nb_core=4)
            misc.compile(['MENUM=%d' % menum, 'all_matrix%dpy.so' % menum],
                         cwd=pjoin(f_dir, 'SubProcesses'))
            so = max(misc.glob('all_matrix%dpy*.so' % menum, pjoin(f_dir, 'SubProcesses')),
                     key=os.path.getsize)
            spec = importlib.util.spec_from_file_location('all_matrix%dpy' % menum, so)
            fortran[name] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fortran[name])
            fortran[name].initialise(pjoin(f_dir, 'Cards', 'param_card.dat'))
        self.assertEqual(mm['prod'].get_idens(), list(fortran['prod'].get_idens()))

        rng = random.Random(3)

        def boost(p, b):
            m = math.sqrt(b[0]**2 - b[1]**2 - b[2]**2 - b[3]**2)
            g, n = b[0] / m, [x / m for x in b[1:]]
            pb = sum(x * y for x, y in zip(p[1:], n))
            f = pb / (g + 1) + p[0]
            return (g * p[0] + pb,) + tuple(x + f * y for x, y in zip(p[1:], n))

        def two_body(M, m1, m2):
            k = math.sqrt((M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)) / (2 * M)
            c, phi = rng.uniform(-1, 1), rng.uniform(0, 2 * math.pi)
            s = math.sqrt(1 - c * c)
            v = (k * s * math.cos(phi), k * s * math.sin(phi), k * c)
            return ((math.sqrt(m1**2 + k * k),) + v,
                    (math.sqrt(m2**2 + k * k),) + tuple(-x for x in v))

        def production(mt1, mt2):
            shat = rng.uniform(mt1 + mt2 + 10, 1200.)
            p3, p4 = two_body(shat, mt1, mt2)
            beta = rng.uniform(-0.5, 0.5)
            gamma = 1. / math.sqrt(1 - beta**2)
            b = (gamma, 0., 0., gamma * beta)
            return [boost(p, b) for p in ((shat / 2, 0, 0, shat / 2),
                                          (shat / 2, 0, 0, -shat / 2), p3, p4)]

        def decay(mt, mw):
            w, b = two_body(mt, mw, 4.7)
            l, v = (boost(q, w) for q in two_body(mw, 0, 0))
            top = (math.sqrt(mt**2 + 250.**2), 100., -60., math.sqrt(250.**2 - 100.**2 - 60.**2))
            return [boost(q, top) for q in ((mt, 0, 0, 0), b, l, v)]

        def check(name, pdgs, p, pos, allow):
            P = np.array(p).T.copy()
            ref = np.asarray(fortran[name].py_get_density(pdgs=pdgs, procid=-1, p=P, pos=pos,
                                                          allow_hel=allow, alphas=0.118, scale2=0.))
            got = mm[name].density(pdgs, p, pos, allow, 0.118)
            self.assertLess(abs(ref - got).max(), 1e-10 * abs(ref).max(), (name, pdgs, p))
            ref = fortran[name].smatrixhel(pdgs, 0, P, 0.118, 0., -1)
            self.assertLess(abs(mm[name].matrix_element(pdgs, p, 0.118) - ref), 1e-10 * ref)

        two = [1, -1]
        both = [h for a in two for b in two for h in (a, b)]
        for mt1, mt2 in ((173., 173.), (168.5, 180.)):
            p = production(mt1, mt2)
            for pdgs in ([21, 21, 6, -6], [2, -2, 6, -6], [-1, 1, 6, -6], [3, -3, 6, -6]):
                check('prod', pdgs, p, [3, 4], both)
                check('prod', pdgs, p, [4], two)
        for mt, mw in ((173., 80.419), (176., 78.), (165., 84.)):
            for pdgs in ([6, 5, -11, 12], [6, 5, -13, 14]):
                check('dec', pdgs, decay(mt, mw), [1], two)
        # a decay at rest boosted by the library (MadSpin's decays) is the decay
        # boosted beforehand
        self.assertTrue(mm['dec'].supports_boost)
        w, b = two_body(171., 79., 4.7)
        l, v = (boost(q, w) for q in two_body(79., 0, 0))
        at_rest = [(171., 0., 0., 0.), b, l, v]
        top = (math.sqrt(171.**2 + 300.**2), 120., -80., math.sqrt(300.**2 - 120.**2 - 80.**2))
        lab = [boost(q, top) for q in at_rest]
        for pdgs in ([6, 5, -11, 12], [6, 5, -13, 14]):
            ref = mm['dec'].density(pdgs, lab, [1], two, 0.118)
            got = mm['dec'].density(pdgs, at_rest, [1], two, 0.118, boost=top)
            self.assertLess(abs(ref - got).max(), 1e-12 * abs(ref).max())
        # a batch is the same as the points one by one
        points = [decay(rng.uniform(165., 180.), rng.uniform(75., 85.)) for _ in range(9)]
        pdgs = [[6, 5, -11, 12] if k % 2 else [6, 5, -13, 14] for k in range(9)]
        batch = mm['dec'].density_batch(pdgs, points, [1], two, [0.118] * 9)
        for k in range(9):
            np.testing.assert_array_equal(batch[k],
                                          mm['dec'].density(pdgs[k], points[k], [1], two, 0.118))

    # Run in a fresh interpreter: every output names its module all_matrix2py,
    # so it must not share a process with one an earlier test imported.
    _pdgs_not_modified_script = '''
import json, math, os, sys
import numpy as np
import all_matrix2py as m
mdir, outfile = sys.argv[1:]
m.initialise(os.path.join(mdir, 'Cards', 'param_card.dat'))
E, th, ph = 500., 0.7, 0.3
k = [E*math.sin(th)*math.cos(ph), E*math.sin(th)*math.sin(ph), E*math.cos(th)]
p = np.array([[E, 0, 0, E], [E, 0, 0, -E], [E] + k,
              [E] + [-x for x in k]]).T.copy()
uu, dd = [2, -2, -11, 11], [1, -1, -11, 11]
cplx = lambda a: [[z.real, z.imag] for z in a]
res = {}
res['fresh_u'] = m.smatrixhel(uu, -1, p, 0.118, 0., -1)
res['fresh_d'] = m.smatrixhel(dd, -1, p, 0.118, 0., -1)
arr = np.array(uu, dtype=np.int32)
res['reused'] = [m.smatrixhel(arr, -1, p, 0.118, 0., -1) for _ in range(2)]
res['after'] = arr.tolist()
kw = dict(procid=-1, p=p, pos=[3], allow_hel=[-1, 1], alphas=0.118, scale2=0.)
res['dens_fresh_u'] = cplx(m.py_get_density(pdgs=uu, **kw))
res['dens_fresh_d'] = cplx(m.py_get_density(pdgs=dd, **kw))
arr = np.array(uu, dtype=np.int32)
res['dens_reused'] = [cplx(m.py_get_density(pdgs=arr, **kw)) for _ in range(2)]
res['dens_after'] = arr.tolist()
batch = np.asfortranarray(np.array([uu, uu], dtype=np.int32).T)
pb = np.asfortranarray(np.stack([p, p], axis=-1))
res['batch_reused'] = []
for _ in range(2):
    inter = m.py_get_density_batch(pdgs=batch, procid=-1, p=pb, pos=[3],
                                   allow_hel=[-1, 1], alphas=[0.118]*2,
                                   scale2=[0.]*2, nbatch=2, next=4)
    res['batch_reused'] += [cplx(inter[:, 0]), cplx(inter[:, 1])]
res['batch_after'] = batch.T.tolist()
json.dump(res, open(outfile, 'w'))
'''

    def test_standalone_f2py_pdgs_not_modified(self):
        """all_matrix2py must not write to the pdgs array it is handed.

        all_matrix.f rewrites every pdg to its merged-particle code (u -> 81)
        before dispatching. f2py passes an int32 numpy array straight through
        for intent(in), so while that rewrite was done on PDGS itself a reused
        [2,-2,-11,11] came back as [81,-81,-82,82] and the next call silently
        answered for d d~. A python list is copied by f2py and never showed it.
        """
        import json
        self.do('import model sm')
        self.do('generate p p > e+ e-')
        self.do('output standalone_fortran %s --prefix=int -f' % self.out_dir)
        subdir = os.path.join(self.out_dir, 'SubProcesses')
        build = subprocess.run(['make', 'f2py'], cwd=subdir,
                               capture_output=True, text=True)
        self.assertEqual(build.returncode, 0,
                         'make f2py failed:\n' + build.stdout[-2000:]
                         + build.stderr[-2000:])

        outfile = os.path.join(self.out_dir, 'pdgs_check.json')
        run = subprocess.run([sys.executable, '-c',
                              self._pdgs_not_modified_script,
                              self.out_dir, outfile],
                             cwd=subdir, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout[-2000:] + run.stderr[-2000:])
        res = json.load(open(outfile))

        uu = [2, -2, -11, 11]
        # u u~ and d d~ differ, so a rewritten array would be caught
        self.assertNotAlmostEqual(res['fresh_d'] / res['fresh_u'], 1, places=3)
        self.assertEqual(res['after'], uu)
        for value in res['reused']:
            self.assertAlmostEqual(value / res['fresh_u'], 1, places=12)

        ref_u = res['dens_fresh_u']
        self.assertNotAlmostEqual(res['dens_fresh_d'][0][0] / ref_u[0][0], 1, places=3)
        self.assertEqual(res['dens_after'], uu)
        self.assertEqual(res['batch_after'], [uu, uu])
        for dens in res['dens_reused'] + res['batch_reused']:
            self.assertEqual(len(dens), len(ref_u))
            for (re_, im_), (ref_re, ref_im) in zip(dens, ref_u):
                self.assertAlmostEqual(re_, ref_re, delta=1e-12 * abs(ref_u[0][0]))
                self.assertAlmostEqual(im_, ref_im, delta=1e-12 * abs(ref_u[0][0]))


    def test_density_mode_user_interface(self):
        ############################################################################
        # This test checks that the python interface of the density mode works properly ie.
        # it creates a LHE file with a tag <density> which contains the density matrix with the correct number of elements.
        # We also check that the average density matrix is stable.
        # To check if the value of the density matrix itself is correct see the other test_density_mode_* tests.
        ############################################################################
        
        text = f"""generate g g > t t~
output madevent {self.out_dir}_density0
launch
reweight=density
set run_card nevents 50000
set helicity_direction [6]
set particle_in_density_matrix [6, -6]
set boost_choice [6, -6]
"""

        #This block of code launches MadGraph with the commands written in mg5_cmd.txt
        command_card = open(pjoin(self.tmpdir, 'mg5_cmd.txt'),'w')
        command_card.write(text)
        command_card.close()

        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                         pjoin(self.out_dir+'_density0', '..', 'mg5_cmd.txt')])

        lhe_path = pjoin(self.out_dir+'_density0','Events','run_01','unweighted_events.lhe.gz')
        rho_mean_path = pjoin(self.out_dir+ '_density0','Events','run_01','Average_density_matrix_unweighted_events.txt')
        
        self.assertTrue(os.path.isfile(lhe_path), f"File not found {lhe_path}")
        self.assertTrue(os.path.isfile(rho_mean_path), f"File not found {rho_mean_path}")


        for event in lhe_parser.EventFile(lhe_path):
            density_check = event.density
            break #we only want the first one
        
        for elem in density_check:
            self.assertIsInstance(elem, complex)
        
        self.assertEqual(len(density_check), 10, f"The density matrix is not the correct length: {density_check}")

        # previously PDF was nn23lo1 (lhaid 230000) with this reference matrix
        # [[0.3670142422790588, 1.7429098337870793e-07-3.933851109770078e-05j, ...],
        #  ... diag(0.36701424, 0.13298576, 0.13298576, 0.36701424), off-diag 0.11514190 / 0.06344293]
        rho_avg_ref =  [[(0.3688357054745634+0j), (2.488456321669277e-07+8.149451446891586e-05j), (-2.488456322029901e-07-8.149451420327119e-05j), (0.1177535354898135-0j)],
                        [(2.488456321669277e-07-8.149451446891586e-05j), (0.13116429452559822+0j), (0.0635907988356563-0j), (-2.488456322029923e-07+8.149451420327103e-05j)],
                        [(-2.488456322029901e-07+8.149451420327119e-05j), (0.0635907988356563+0j), (0.13116429452559822+0j), (2.488456321669272e-07-8.149451446891567e-05j)],
                        [(0.1177535354898135+0j), (-2.488456322029923e-07-8.149451420327103e-05j), (2.488456321669272e-07+8.149451446891567e-05j), (0.3688357054745633+0j)]]

        #now let's read the average density matrix
        with open(rho_mean_path, 'r') as f:
            data = f.readlines()[1:]
            rho_avg = []
            for i in range(len(data)):
                aux = data[i].strip("\t\n[]").split(",")
                rho_avg.append([complex(aux[i].strip(" ()")) for i in range(len(aux))])
            
        # On a mismatch print the whole measured matrix, not just the first
        # element that differs: re-referencing this (a PDF change moves every
        # entry) otherwise needs one run per element.
        msg = 'measured rho_avg = %r' % (rho_avg,)
        for i in range(len(rho_avg)):
            for j in range(len(rho_avg[0])):
                # statistical error of a 50k-event average: at most 5e-4 per
                # element (measured), so 3 digits (+-5e-4) is ~1 sigma and fails for
                # most changes of the event generation; 2e-3 is 4 sigma
                self.assertAlmostEqual(rho_avg[i][j].real, rho_avg_ref[i][j].real, delta=2e-3, msg=msg)
                self.assertAlmostEqual(rho_avg[i][j].imag, rho_avg_ref[i][j].imag, delta=2e-3, msg=msg)


    def test_density_mode_user_interface(self):
        ############################################################################
        # This test checks that the python interface of the density mode works properly ie.
        # it creates a LHE file with a tag <density> which contains the density matrix with the correct number of elements.
        # We also check that the average density matrix is stable.
        # To check if the value of the density matrix itself is correct see the other test_density_mode_* tests.
        ############################################################################
        
        text = f"""generate g g > t t~
output madevent {self.out_dir}_density0
launch
reweight=density
set run_card nevents 50000
set helicity_direction [6]
set particle_in_density_matrix [6, -6]
set boost_choice [6, -6]
"""

        #This block of code launches MadGraph with the commands written in mg5_cmd.txt
        command_card = open('/tmp/mg5_cmd.txt','w')
        command_card.write(text)
        command_card.close()

        
        logfile = 'test_density_mode_ttbar.log'
        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                         '/tmp/mg5_cmd.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)


        

        lhe_path = pjoin(self.out_dir + '_density0/Events/run_01/unweighted_events.lhe.gz')
        rho_mean_path = pjoin(self.out_dir + '_density0/Events/run_01/Average_density_matrix_unweighted_events.txt')
        
        self.assertTrue(os.path.isfile(lhe_path), f"File not found {lhe_path}")
        self.assertTrue(os.path.isfile(rho_mean_path), f"File not found {rho_mean_path}")


        for event in lhe_parser.EventFile(lhe_path):
            density_check = event.density
            break #we only want the first one
        
        for elem in density_check:
            self.assertIsInstance(elem, complex)
        
        self.assertEqual(len(density_check), 10, f"The density matrix is not the correct length: {density_check}")

        # previously PDF was nn23lo1 (lhaid 230000) with this reference matrix
        # [[0.3670142422790588, 1.7429098337870793e-07-3.933851109770078e-05j, ...],
        #  ... diag(0.36701424, 0.13298576, 0.13298576, 0.36701424), off-diag 0.11514190 / 0.06344293]
        rho_avg_ref =  [[(0.3688357054745634+0j), (2.488456321669277e-07+8.149451446891586e-05j), (-2.488456322029901e-07-8.149451420327119e-05j), (0.1177535354898135-0j)],
                        [(2.488456321669277e-07-8.149451446891586e-05j), (0.13116429452559822+0j), (0.0635907988356563-0j), (-2.488456322029923e-07+8.149451420327103e-05j)],
                        [(-2.488456322029901e-07+8.149451420327119e-05j), (0.0635907988356563+0j), (0.13116429452559822+0j), (2.488456321669272e-07-8.149451446891567e-05j)],
                        [(0.1177535354898135+0j), (-2.488456322029923e-07-8.149451420327103e-05j), (2.488456321669272e-07+8.149451446891567e-05j), (0.3688357054745633+0j)]]

        #now let's read the average density matrix
        with open(rho_mean_path, 'r') as f:
            data = f.readlines()[1:]
            rho_avg = []
            for i in range(len(data)):
                aux = data[i].strip("\t\n[]").split(",")
                try:
                    rho_avg.append([complex(aux[i].strip(" ()")) for i in range(len(aux))])
                except: #if the values are like "np.complex128(value)"
                    print("aux", aux)
                    aux2 = [aux[i].strip(" ()[]").strip("'").replace("np.complex128(","").strip(" ()") for i in range(len(aux))]
                    try:
                        rho_avg.append([complex(aux2[i]) for i in range(len(aux2))])
                    except:
                        print("aux2", aux2)
                        raise ValueError
            

        # On a mismatch print the whole measured matrix, not just the first
        # element that differs: re-referencing this (a PDF change moves every
        # entry) otherwise needs one run per element.
        msg = 'measured rho_avg = %r' % (rho_avg,)
        for i in range(len(rho_avg)):
            for j in range(len(rho_avg[0])):
                # statistical error of a 50k-event average: at most 5e-4 per
                # element (measured), so 3 digits (+-5e-4) is ~1 sigma and fails for
                # most changes of the event generation; 2e-3 is 4 sigma
                self.assertAlmostEqual(rho_avg[i][j].real, rho_avg_ref[i][j].real, delta=2e-3, msg=msg)
                self.assertAlmostEqual(rho_avg[i][j].imag, rho_avg_ref[i][j].imag, delta=2e-3, msg=msg)


    def test_density_mode_ttbar(self):
        ############################################################################
        # Check working condition of the density mode
        # reproduces the full density matrix and computes quantum information observables
        # testing case g g > t t~
        ############################################################################
        import madgraph.various.Density_functions as dens
        #we generate just one event of the process  to create the process folder (it is fast enough)
        text = f"""generate g g > t t~
output madevent {self.out_dir}_density1
launch
set run_card nevents 1
set use_syst False
"""

        #This block of code launches MadGraph with the commands written in mg5_cmd.txt
        command_card = open(pjoin(self.tmpdir, 'mg5_cmd.txt'),'w')
        command_card.write(text)
        command_card.close()

        logfile = 'test_density_mode_ttbar1.log'
        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                         pjoin(self.tmpdir, 'mg5_cmd.txt')], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)


        #Here we replace the lhe file by the reference lhe file (stored in the input_files).
        os.remove(f"{self.out_dir}_density1/Events/run_01/unweighted_events.lhe.gz")
        shutil.copyfile(pjoin(MG5DIR, "tests/input_files/density_mode/test_density_mode_ttbar.lhe.gz"), f"{self.out_dir}_density1/Events/run_01/unweighted_events.lhe.gz")

        #Now we reweight the lhe file through the inline method
        text_rwgt = f"""launch {self.out_dir}_density1/ -i
reweight run_01 --mode=density
set helicity_direction [6]
set boost_choice [6, -6]
"""

        #This block of code launches MadGraph with the commands written in mg5_cmd.txt
        command_card_rwgt = open('/tmp/mg5_cmd_rwgt.txt','w')
        command_card_rwgt.write(text_rwgt)
        command_card_rwgt.close()

        logfile = 'test_density_mode_ttbar2.log'
        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                         '/tmp/mg5_cmd_rwgt.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)
        
        #The lhe file has been reweighted, now we read density_check, the density matrix to compare to the reference
        lhe_path = pjoin(self.out_dir + '_density1/Events/run_01/unweighted_events.lhe.gz')
        for event in lhe_parser.EventFile(lhe_path):
            density_check = event.density

        #reference density matrix
        # Regenerated when the reweighting (standalone_rw) started to set the
        # t-channel widths to zero, as madevent did for the input events
        # (zerowidth_tchannel): the t-channel top propagator lost MDL_WT.
        density_ref = [(0.45270438876343766+0.0j), (0.0+0.0024345422714880808j), (-0.0-0.002434542275983678j),
                       (0.28551318353826904+0.0j), (0.047295611236562354+0.0j), (0.047011148655689436+0.0j),
                       (0.0+0.0024345422759837264j), (0.047295611236562354+0.0j), (-0.0-0.0024345422714880972j),
                       (0.45270438876343766+0.0j)]

        #1) here we check that the density matrix is computed properly
        for i in range(len(density_ref)):
            self.assertAlmostEqual(density_ref[i].real, density_check[i].real, places=7)
            self.assertAlmostEqual(density_ref[i].imag, density_check[i].imag, places=7)
        
        rho_instance = dens.DensityMatrixObservables(density_check)

        #2) here we check that the concurrence is computed properly
        concurrence_ref = 0.47643514460330366
        concurrence_check = rho_instance.Get_Concurrence()
        self.assertAlmostEqual(concurrence_ref, concurrence_check, places=7)

        #3) here we check that purity is computed properly
        purity_ref = 0.5818593450086657
        purity_check = rho_instance.Get_Purity()
        self.assertAlmostEqual(purity_ref, purity_check, places=7)

        #4) here we check that magic is computed properly
        magic_ref = 0.4706253424888031
        magic_check = rho_instance.Magic_Mixed()
        self.assertAlmostEqual(magic_ref, magic_check, places=7)

    def test_density_mode_wpwm(self):
        ############################################################################
        # Check working condition of the density mode
        # reproduces the full density matrix and computes quantum information observables
        # testing case u u~ > w+ w-
        ############################################################################
        import madgraph.various.Density_functions as dens
        text = f"""generate u u~ > w+ w-
output madevent {self.out_dir}_density2
launch
set run_card nevents 1
set use_syst False
"""

        #This block of code launches MadGraph with the commands written in mg5_cmd.txt
        command_card = open('/tmp/mg5_cmd2.txt','w')
        command_card.write(text)
        command_card.close()


        logfile = 'test_density_mode_wpwm1.log'
        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                         '/tmp/mg5_cmd2.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)


        #Here we replace the lhe file by the reference lhe file (stored in the input_files).
        os.remove(f"{self.out_dir}_density2/Events/run_01/unweighted_events.lhe.gz")
        shutil.copyfile(pjoin(MG5DIR, "tests/input_files/density_mode/test_density_mode_wpwm.lhe.gz"), f"{self.out_dir}_density2/Events/run_01/unweighted_events.lhe.gz")

        #Now we reweight the lhe file through the inline method
        text_rwgt = f"""launch {self.out_dir}_density2/ -i
reweight run_01 --mode=density
set helicity_direction [24]
set particle_in_density_matrix [24, -24]
set boost_choice [24, -24]
set order_helicities [+1, -1, +1, 0, +1, +1, 0, -1, 0, 0, 0, +1, -1, -1, -1, 0, -1, +1]
set axis_referential [-1, -2]
"""

        #This block of code launches MadGraph with the commands written in mg5_cmd.txt
        command_card_rwgt = open('/tmp/mg5_cmd_rwgt2.txt','w')
        command_card_rwgt.write(text_rwgt)
        command_card_rwgt.close()

        logfile = 'test_density_mode_wpwm2.log'
        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                         '/tmp/mg5_cmd_rwgt2.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)
        
        #The lhe file has been reweighted, now we read density_check, the density matrix to compare to the reference
        lhe_path = pjoin(self.out_dir + '_density2/Events/run_01/unweighted_events.lhe.gz')
        for event in lhe_parser.EventFile(lhe_path):
            density_check = event.density

        #reference density matrix
        density_ref = [1.1377075855848739e-05+0j, -2.490684288126926e-09+4.954042982384063e-05j, -0.00010741895892651527-1.052802929701333e-08j, -2.490684288126926e-09+4.9540429845192674e-05j, -0.00020752750288735081-1.8249104252772033e-07j,
                       7.268812229221872e-07-0.00037506459555688727j, -0.0001074189589717972-1.052802929701333e-08j, 7.268812229221872e-07-0.0003750645955782362j, -0.003320285453600001+0j, 0.0002263297014919883+0j, 
                       -2.2327036098315108e-08+0.00046759229342787956j, 0.0002263297015849638+0j, -7.492082590489525e-07+0.0009009955880770624j, -0.001633220781778451-3.0830277624655218e-06j, -2.232703608840189e-08+0.00046759229362505555j, 
                       -0.0016332207818714126-3.083027762460848e-06j, 7.268812229221946e-07+0.014457877454041002j, 0.0010142200261194483+0j, -2.2327036118073693e-08-0.0004675922936294792j, 0.00195945157385873+1.5309858519097204e-06j, 
                       -6.515921500262154e-06+0.0035412482571934695j, 0.0010142200265469868+0j, -6.515921500242399e-06+0.0035412482573950388j, 0.03134914553472118-3.072499733046761e-06j, 0.00022632970167793935+0j, 
                       -7.492082593914445e-07+0.0009009955884665416j, -0.0016332207824823572-3.0830277638297043e-06j, -2.232703610816048e-08+0.0004675922938266552j, -0.001633220782575319-3.0830277638250304e-06j, 7.268812229221946e-07+0.01445787746027238j, 
                       0.003786149199326624+0j, -7.242802726675139e-06+0.006841498529578208j, 0.001959451574684711-1.530985852636054e-06j, -7.2428027263327e-06+0.006841498529967629j, 0.06056481979985879-5.3258180009900856e-05j, 
                       0.012364686884072221+0j, -6.515921503155216e-06-0.0035412482586862654j, 0.012364686884776021+0j, -0.00021213299283073748-0.10945883956181096j, 0.0010142200269745254+0j,
                       -6.515921503135461e-06+0.0035412482588878342j, 0.031349145547936254-3.072499733046761e-06j, 0.012364686885479825+0j, -0.00021213299283073748-0.10945883956804142j, 0.9689920004990016+0j]


        event_of_reference = """<event>
 4      1 +1.7477710e+01 8.42506000e+01 7.54677100e-03 1.31740400e-01
        2 -1    0    0  501    0 +0.0000000000e+00 +0.0000000000e+00 +2.6479266201e+02 2.6479266201e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
       -2 -1    0    0    0  501 -0.0000000000e+00 -0.0000000000e+00 -1.9946623190e+02 1.9946623190e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
       24  1    1    2    0    0 -7.2588221973e+00 +2.4046986127e+01 +2.4863204649e+02 2.6251868227e+02 8.0419002446e+01 0.0000e+00 -1.0000e+00
      -24  1    1    2    0    0 +7.2588221973e+00 -2.4046986127e+01 -1.8330561638e+02 2.0174021164e+02 8.0419002446e+01 0.0000e+00 1.0000e+00
<density> (1.1377075855848739e-05+0j) (-2.490684288126926e-09+4.954042982384063e-05j) (-0.00010741895892651527-1.052802929701333e-08j) (-2.490684288126926e-09+4.9540429845192674e-05j) (-0.00020752750288735081-1.8249104252772033e-07j) (7.268812229221872e-07-0.00037506459555688727j) (-0.0001074189589717972-1.052802929701333e-08j) (7.268812229221872e-07-0.0003750645955782362j) (-0.003320285453600001+0j) (0.0002263297014919883+0j) (-2.2327036098315108e-08+0.00046759229342787956j) (0.0002263297015849638+0j) (-7.492082590489525e-07+0.0009009955880770624j) (-0.001633220781778451-3.0830277624655218e-06j) (-2.232703608840189e-08+0.00046759229362505555j) (-0.0016332207818714126-3.083027762460848e-06j) (7.268812229221946e-07+0.014457877454041002j) (0.0010142200261194483+0j) (-2.2327036118073693e-08-0.0004675922936294792j) (0.00195945157385873+1.5309858519097204e-06j) (-6.515921500262154e-06+0.0035412482571934695j) (0.0010142200265469868+0j) (-6.515921500242399e-06+0.0035412482573950388j) (0.03134914553472118-3.072499733046761e-06j) (0.00022632970167793935+0j) (-7.492082593914445e-07+0.0009009955884665416j) (-0.0016332207824823572-3.0830277638297043e-06j) (-2.232703610816048e-08+0.0004675922938266552j) (-0.001633220782575319-3.0830277638250304e-06j) (7.268812229221946e-07+0.01445787746027238j) (0.003786149199326624+0j) (-7.242802726675139e-06+0.006841498529578208j) (0.001959451574684711-1.530985852636054e-06j) (-7.2428027263327e-06+0.006841498529967629j) (0.06056481979985879-5.3258180009900856e-05j) (0.012364686884072221+0j) (-6.515921503155216e-06-0.0035412482586862654j) (0.012364686884776021+0j) (-0.00021213299283073748-0.10945883956181096j) (0.0010142200269745254+0j) (-6.515921503135461e-06+0.0035412482588878342j) (0.031349145547936254-3.072499733046761e-06j) (0.012364686885479825+0j) (-0.00021213299283073748-0.10945883956804142j) (0.9689920004990016+0j) </density>
</event>
"""
        
        #1) here we check that the density matrix is computed properly
        for i in range(len(density_ref)):
            self.assertAlmostEqual(density_ref[i].real, density_check[i].real, places=7)
            self.assertAlmostEqual(density_ref[i].imag, density_check[i].imag, places=7)

        rho_instance = dens.DensityMatrixObservables(density_check)

        #2) here we check that the bounds of concurrence are computed properly
        lower_concurrence2_ref = 0.023941838572111518
        upper_concurrence2_ref = 0.02402940624369876
        lower_concurrence2_check = rho_instance.ConcLB2()
        upper_concurrence2_check = rho_instance.ConcUB2()
        self.assertAlmostEqual(lower_concurrence2_check, lower_concurrence2_ref, places=7)
        self.assertAlmostEqual(upper_concurrence2_check, upper_concurrence2_ref, places=7)
      
        #3) here we check that purity is computed properly
        purity_ref = 0.9999562161642062
        purity_check = rho_instance.Get_Purity()
        self.assertAlmostEqual(purity_check, purity_ref, places=7)

        #4) here we check that mana is computed properly
        mana_ref = 0.5113137785276397
        mana_check = rho_instance.Get_Mana()
        self.assertAlmostEqual(mana_check, mana_ref, places=7)

    def test_density_mode_decay1(self):
        ############################################################################
        # Check working condition of the density mode
        # reproduces the full density matrix and computes quantum information observables
        # testing case g g > t t~, t > b W+
        # particle_in_density_matrix = [5, -6]
        ############################################################################
        import madgraph.various.Density_functions as dens
        text = f"""generate g g > t t~, t > b w+
output madevent {self.out_dir}_density3
launch
set run_card nevents 1
set run_card use_syst False
"""


        #This block of code launches MadGraph with the commands written in mg5_cmd.txt
        command_card = open('/tmp/mg5_cmd.txt','w')
        command_card.write(text)
        command_card.close()

        logfile = 'test_density_mode_decay11.log'
        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                         '/tmp/mg5_cmd.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)


        #Here we replace the lhe file by the reference lhe file (stored in the input_files).
        os.remove(f"{self.out_dir}_density3/Events/run_01/unweighted_events.lhe.gz")
        shutil.copyfile(pjoin(MG5DIR, "tests/input_files/density_mode/test_density_mode_decay1.lhe.gz"), f"{self.out_dir}_density3/Events/run_01/unweighted_events.lhe.gz")

        #Now we reweight the lhe file through the inline method
        text_rwgt = f"""launch {self.out_dir}_density3/ -i
reweight run_01 --mode=density
set helicity_direction [5]
set particle_in_density_matrix [5, -6]
set boost_choice [5, -6]
"""

        #This block of code launches MadGraph with the commands written in mg5_cmd.txt
        command_card_rwgt = open('/tmp/mg5_cmd_rwgt.txt','w')
        command_card_rwgt.write(text_rwgt)
        command_card_rwgt.close()

        logfile = 'test_density_mode_decay12.log'
        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                         '/tmp/mg5_cmd_rwgt.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)
        
        #The lhe file has been reweighted, now we read density_check, the density matrix to compare to the reference
        lhe_path = pjoin(self.out_dir + '_density3/Events/run_01/unweighted_events.lhe.gz')
        for event in lhe_parser.EventFile(lhe_path):
            density_check = event.density

        #reference density matrix
        # Regenerated when the density mode's boost_choice started summing the
        # legs it names: it used to count LHE lines, status-2 ones included, so
        # the status-2 top of this decay chain shifted it onto t (the resonance
        # line) + W+ instead of b + t~. The previous numbers were in that frame.
        # Regenerated when the reweighting (standalone_rw) started to set the
        # t-channel widths to zero, as madevent did for the input events
        # (zerowidth_tchannel): the t-channel top propagator lost MDL_WT.
        density_ref = [(0.002572819258503627+0.0j), (0.00022287625051417933+0.000269373544283043j), (0.034620381554042186-0.0031148240475356633j),
                       (0.003493349987547038+0.0030943536658021155j), (0.002492794223097223+0.0j), (0.0022714731159374324-0.0025077711911968975j),
                       (0.03329882105505325-0.001669052089828375j), (0.5122804070927276+0.0j), (0.043564252872336236+0.03090169904947277j),
                       (0.48265397942567173+0.0j)]


        event_of_reference = """<event>
 6      1 +4.2873600e+02 2.58116800e+02 7.54677100e-03 1.10829800e-01
       21 -1    0    0  501  502 +0.0000000000e+00 +0.0000000000e+00 +1.0845458909e+02 1.0845458909e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
       21 -1    0    0  502  503 -0.0000000000e+00 -0.0000000000e+00 -6.4844659178e+02 6.4844659178e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
        6  2    1    2  501    0 -1.0996883743e+01 +1.9124424233e+02 -1.8275210837e+02 3.1616925127e+02 1.7282757494e+02 0.0000e+00 0.0000e+00
        5  1    3    3  501    0 +4.3637392514e+01 +4.9281743782e+00 -1.3060449857e+01 4.6056207819e+01 4.7000000000e+00 0.0000e+00 -1.0000e+00
       24  1    3    3    0    0 -5.4634276257e+01 +1.8631606795e+02 -1.6969165851e+02 2.7011304345e+02 8.0419002446e+01 0.0000e+00 -1.0000e+00
       -6  1    1    2    0  503 +1.0996883743e+01 -1.9124424233e+02 -3.5723989432e+02 4.4073192960e+02 1.7300000000e+02 0.0000e+00 -1.0000e+00
<density> (0.002572819258503627+0.0j) (0.00022287625051417933+0.000269373544283043j) (0.034620381554042186-0.0031148240475356633j) (0.003493349987547038+0.0030943536658021155j) (0.002492794223097223+0.0j) (0.0022714731159374324-0.0025077711911968975j) (0.03329882105505325-0.001669052089828375j) (0.5122804070927276+0.0j) (0.043564252872336236+0.03090169904947277j) (0.48265397942567173+0.0j) </density>
</event>
"""
        
        #1) here we check that the density matrix is computed properly
        for i in range(len(density_ref)):
            self.assertAlmostEqual(density_ref[i].real, density_check[i].real, places=7)
            self.assertAlmostEqual(density_ref[i].imag, density_check[i].imag, places=7)

        rho_instance = dens.DensityMatrixObservables(density_check)

        #2) here we check that the bounds of concurrence is computed properly
        concurrence_ref = 0.0
        concurrence_check = rho_instance.Get_Concurrence()
        self.assertAlmostEqual(concurrence_ref, concurrence_check, places=7)
      
        #3) here we check that purity is computed properly
        purity_ref = 0.505810869888376
        purity_check = rho_instance.Get_Purity()
        self.assertAlmostEqual(purity_ref, purity_check, places=7)

        #4) here we check that magic is computed properly
        magic_ref = 0.04539434702919854
        magic_check = rho_instance.Magic_Mixed()
        self.assertAlmostEqual(magic_ref, magic_check, places=7)

    def test_density_mode_decay2(self):
        ############################################################################
        # Check working condition of the density mode
        # reproduces the full density matrix and computes quantum information observables
        # testing case g g > t t~, t > b W+
        # particle_in_density_matrix = [24, -6]
        ############################################################################
        import madgraph.various.Density_functions as dens
        #we generate just one event of the process  to create the process folder (it is fast enough)
        text = f"""generate g g > t t~, t > b w+
output madevent {self.out_dir}_density4
launch
reweight=density
set run_card nevents 1
set run_card use_syst False
"""

        #This block of code launches MadGraph with the commands written in mg5_cmd.txt
        command_card = open('/tmp/mg5_cmd.txt','w')
        command_card.write(text)
        command_card.close()

        logfile = 'test_density_mode_decay21.log'
        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                         '/tmp/mg5_cmd.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)


        #Here we replace the lhe file by the reference lhe file (stored in the input_files).
        os.remove(f"{self.out_dir}_density4/Events/run_01/unweighted_events.lhe.gz")
        shutil.copyfile(pjoin(MG5DIR, "tests/input_files/density_mode/test_density_mode_decay2.lhe.gz"), f"{self.out_dir}_density4/Events/run_01/unweighted_events.lhe.gz")

        #Now we reweight the lhe file through the inline method
        text_rwgt = f"""launch {self.out_dir}_density4/ -i
reweight run_01 --mode=density
set helicity_direction [24]
set particle_in_density_matrix [24, -6]
set boost_choice [24, -6]
"""

        #This block of code launches MadGraph with the commands written in mg5_cmd.txt
        command_card_rwgt = open('/tmp/mg5_cmd_rwgt.txt','w')
        command_card_rwgt.write(text_rwgt)
        command_card_rwgt.close()

        logfile = 'test_density_mode_decay22.log'
        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                         '/tmp/mg5_cmd_rwgt.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)
        
        #The lhe file has been reweighted, now we read density_check, the density matrix to compare to the reference
        lhe_path = pjoin(self.out_dir + '_density4/Events/run_01/unweighted_events.lhe.gz')
        for event in lhe_parser.EventFile(lhe_path):
            density_check = event.density

        #reference density matrix
        # Regenerated when the density mode's boost_choice started summing the
        # legs it names: it used to count LHE lines, status-2 ones included, so
        # the status-2 top of this decay chain shifted it onto b + W+ instead of
        # W+ + t~. The previous numbers were in that frame.
        # Regenerated when the reweighting (standalone_rw) started to set the
        # t-channel widths to zero, as madevent did for the input events
        # (zerowidth_tchannel): the t-channel top propagator lost MDL_WT.
        density_ref = [(0.03462856093584298+0.0j), (0.004222692552048597-3.914851804525198e-06j), (-0.09315136922763956-0.06424541106056061j),
                       (-0.012161343076623666-0.007852907982114157j), (-0.0021548339817954705-0.02299216606257348j), (0.004000989959849888+0.0004271687621753295j),
                       (0.022124992142446936+0.0j), (-0.004976106543712062-0.007632363502328344j), (-0.05784211938393767-0.041606353301745905j),
                       (-0.034560622408768275-0.028594347672533223j), (-0.012757048104007053-0.018079926753134273j), (0.374857868204696+0.0j),
                       (0.029720084211203073-0.012771984245079535j), (0.03485290974186369+0.0484732992793134j), (-0.01064461469034376+0.0037920989793522506j),
                       (0.2350541853144974+0.0j), (0.14918389804980686+0.014260521941388894j), (0.045994627672498976+0.008149821823945263j),
                       (0.1306812643391364+0.0j), (0.0037983745012122867+0.02048399076640624j), (0.20265312906338015+0.0j)]

        
        #1) here we check that the density matrix is computed properly
        for i in range(len(density_ref)):
            self.assertAlmostEqual(density_ref[i].real, density_check[i].real, places=7)
            self.assertAlmostEqual(density_ref[i].imag, density_check[i].imag, places=7)

        rho_instance = dens.DensityMatrixObservables(density_check)

        #2) here we check that the smaller eigenvalue of the partially transposed density matrix is computed properly
        flag_ref, eigval_ref = False, [0.0001309474273079456, 0.00023369563417047353, 0.10046843670831423, 0.1278389488146789, 0.25565138370380525, 0.5156765877117229]
        flag_check, eigval_check = rho_instance.PeresHorodecki_criterion(['boson', 'fermion'])
        self.assertEqual(flag_ref, flag_check)
        for i in range(len(eigval_ref)):
            self.assertAlmostEqual(eigval_ref[i], eigval_check[i], places=7)
      
        #3) here we check that purity is computed properly
        purity_ref = 0.35771674847320956
        purity_check = rho_instance.Get_Purity()
        self.assertAlmostEqual(purity_ref, purity_check, places=7)

    def test_density_mode_doublettbar(self):
        ############################################################################
        # Check working condition of the density mode
        # reproduces the full density matrix and computes quantum information observables
        # testing case p p > t t t~ t~
        # helicity_direction [6] pt [0]
        # particle_in_density_matrix [6, -6] rapidity [0, 1]
        # boost_choice [6, -6] pt [0, 0]
        ############################################################################
        import madgraph.various.Density_functions as dens
        #we generate just one event of the process  to create the process folder (it is fast enough)
        text = f"""generate p p > t t t~ t~
output madevent {self.out_dir}_density5
launch
set run_card nevents 1
set run_card use_syst False
"""
        
        #This block of code launches MadGraph with the commands written in mg5_cmd.txt
        command_card = open('/tmp/mg5_cmd.txt','w')
        command_card.write(text)
        command_card.close()

        logfile = 'test_density_mode_ttbar1.log'
        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                         '/tmp/mg5_cmd.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)


        #Here we replace the lhe file by the reference lhe file (stored in the input_files).
        # The MG5 run above is not checked for success and its log is not part
        # of the test output, so a failed generation used to surface only as a
        # FileNotFoundError on the line below, with the actual error invisible.
        generated = f"{self.out_dir}_density5/Events/run_01/unweighted_events.lhe.gz"
        if not os.path.exists(generated):
            self.fail('MG5 did not produce %s. Tail of %s:\n%s'
                      % (generated, logfile, open(logfile).read()[-3000:]))
        os.remove(generated)
        shutil.copyfile(pjoin(MG5DIR, "tests/input_files/density_mode/test_density_mode_doublettbar.lhe.gz"), f"{self.out_dir}_density5/Events/run_01/unweighted_events.lhe.gz")

        #Now we reweight the lhe file through the inline method
        text_rwgt = f"""launch {self.out_dir}_density5/ -i
reweight run_01 --mode=density
set helicity_direction [6] pt [0]
set particle_in_density_matrix [6, -6] rapidity [0, 1]
set boost_choice [6, -6] pt [0, 0]
"""

        #This block of code launches MadGraph with the commands written in mg5_cmd.txt
        command_card_rwgt = open('/tmp/mg5_cmd_rwgt.txt','w')
        command_card_rwgt.write(text_rwgt)
        command_card_rwgt.close()

        logfile = 'test_density_mode_ttbar2.log'
        subprocess.call([sys.executable,pjoin(MG5DIR,'bin','madgraph'), 
                         '/tmp/mg5_cmd_rwgt.txt'], stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)
        

        ## With the chosen seed, the event must be:
        event_random = """<event>
 6      1 +8.8259995e-03 6.31093800e+02 7.54677100e-03 1.00159800e-01
       21 -1    0    0  504  503 +0.0000000000e+00 +0.0000000000e+00 +1.2135286086e+03 1.2135286086e+03 0.0000000000e+00 0.0000e+00 -1.0000e+00
       21 -1    0    0  502  504 -0.0000000000e+00 -0.0000000000e+00 -5.8371897887e+02 5.8371897887e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
        6  1    1    2  501    0 +3.6623210951e+02 -1.7709369819e+02 +2.0834055640e+02 4.8869512318e+02 1.7300000000e+02 0.0000e+00 -1.0000e+00
        6  1    1    2  502    0 -3.1599447381e+02 +3.9628896508e+02 +1.0466750690e+02 5.4569381371e+02 1.7300000000e+02 0.0000e+00 -1.0000e+00
       -6  1    1    2    0  501 -2.8790168023e+02 -5.3212030280e+01 +2.4116702458e+02 4.1690458307e+02 1.7300000000e+02 0.0000e+00 -1.0000e+00
       -6  1    1    2    0  503 +2.3766404453e+02 -1.6598323661e+02 +7.5634541823e+01 3.4595406749e+02 1.7300000000e+02 0.0000e+00 -1.0000e+00
<density> (0.4158995551694228+0.0j) (-0.038266416707557815-0.08667282347919017j) (0.018177344236518424+0.06949114218724137j) (-0.006036326465490495+0.02831843465941133j) (0.08410102856179515+0.0j) (-0.051324023693436596-0.010218495226221291j) (-0.018178695623423707-0.06949438656775611j) (0.0840990482542204+0.0j) (0.03826126259622656+0.08667076377998419j) (0.4159003680145617+0.0j) </density>
</event>
"""

        #The lhe file has been reweighted, now we read density_check, the density matrix to compare to the reference
        lhe_path = pjoin(self.out_dir + '_density5/Events/run_01/unweighted_events.lhe.gz')
        for event in lhe_parser.EventFile(lhe_path):
            density_check = event.density

        #reference density matrix
        # Regenerated when the reweighting (standalone_rw) started to set the
        # t-channel widths to zero, as madevent did for the input events
        # (zerowidth_tchannel): the t-channel top propagator lost MDL_WT.
        density_ref = [(0.4158995551694228+0.0j), (-0.038266416707557815-0.08667282347919017j), (0.018177344236518424+0.06949114218724137j),
                       (-0.006036326465490495+0.02831843465941133j), (0.08410102856179515+0.0j), (-0.051324023693436596-0.010218495226221291j),
                       (-0.018178695623423707-0.06949438656775611j), (0.0840990482542204+0.0j), (0.03826126259622656+0.08667076377998419j),
                       (0.4159003680145617+0.0j)]

        lhe_path = pjoin(self.out_dir + '_density5/Events/run_01/unweighted_events.lhe.gz')
        for event in lhe_parser.EventFile(lhe_path):
            density_check = event.density
        
        # 1) here we check that the density matrix is computed properly
        for i in range(len(density_ref)):
            self.assertAlmostEqual(density_ref[i].real, density_check[i].real, places=7)
            self.assertAlmostEqual(density_ref[i].imag, density_check[i].imag, places=7)

        rho_instance = dens.DensityMatrixObservables(density_check)

        #2) here we check that the bounds of concurrence is computed properly
        concurrence_ref = 0.028913877218595573
        concurrence_check = rho_instance.Get_Concurrence()
        self.assertAlmostEqual(concurrence_ref, concurrence_check, places=7)
      
        # #3) here we check that purity is computed properly
        purity_ref = 0.4237883009145146
        purity_check = rho_instance.Get_Purity()
        self.assertAlmostEqual(purity_ref, purity_check, places=7)

        # #4) here we check that magic is computed properly
        magic_ref = 0.4802316159074085
        magic_check = rho_instance.Magic_Mixed()
        self.assertAlmostEqual(magic_ref, magic_check, places=7)

    @staticmethod
    def invert_momenta(p):
        """ fortran/C-python do not order table in the same order"""
        new_p = []
        for i in range(len(p[0])):  new_p.append([0]*len(p))
        for i, onep in enumerate(p):
            for j, x in enumerate(onep):
                new_p[j][i] = x
        return new_p



    def edit_p_in_standalone(self, dir, p):
        """edit the check.f file to include the momenta p"""
        misc.sprint(dir)
        text = []
        done = False
        for line in open(os.path.join(dir, 'check_sa.f'), 'r'):
            if 'CALL GET_MOMENTA(SQRTS,PMASS,P)' in line:
                for i in range(len(p)):
                    real_p = lhe_parser.FourMomentum(p[i])
                    misc.sprint(real_p)
                    done = True
                    for j in range(4):
                        text.append('        p(%s,%s) = %e\n' % (j, i+1, real_p[j]))
            else:
                text.append(line)

        if not done:
            raise Exception('Could not find place to insert momenta in check_sa.f')

        checkf = os.path.join(dir, 'check_sa.f')
        open(checkf, 'w').write('\n'.join(text))

        
    def notest_v4_heft(self):
        """Test standalone directory for UFO HEFT model"""
        ##v4 not supported anymore

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model heft', force=True)
        self.do('generate g g > h g g')
        self.do('output standalone_fortran %s ' % self.out_dir)

        devnull = open(os.devnull,'w')
        # Check that the Model and Aloha output compile
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'Source'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        # Check that check_sa.f compiles
        subprocess.call(['make', 'check'],
#                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_gg_hgg'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses', 'P0_gg_hgg',
                                                    'check')))
        # Check that the output of check is correct 
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_gg_hgg',
                               'check.log')
        p = subprocess.Popen('./check', 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_gg_hgg'), shell=True)
        (log_output, err) = p.communicate()                                         
        log_output =log_output.decode()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)
        
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 1.10908942e-06)
        
    def test_madevent_ufo_aloha(self):
        """Test MadEvent output with UFO/ALOHA"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('set apply_flavor_grouping False')
        self.do('import model sm')
        self.do('set group_subprocesses False')
        self.do('generate e+ e- > e+ e-')
        self.do('output madevent %s ' % self.out_dir)
        # Check that the needed ALOHA subroutines are generated
        files = ['aloha_file.inc',
                 #'FFS1C1_2.f', 'FFS1_0.f',
                 'FFV1_0.f', 'FFV1P0_3.f',
                 'FFV2_0.f', 'FFV2_3.f',
                 'FFV4_0.f', 'FFV4_3.f',
                 'makefile', 'aloha_functions.f']
        for f in files:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir,
                                                        'Source', 'DHELAS',
                                                        f)), 
                            '%s file is not in aloha directory' % f)
        
        #check the content of FFV1P0_0.f
        self.check_aloha_file()
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'Cards',
                                                    'ident_card.dat')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'run_card_default.dat')))
        devnull = open(os.devnull,'w')
        # Check that the Source directory compiles
        status = subprocess.call(['make'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libgeneric.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libcernlib.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdsample.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libpdf.a')))
        # Check that gensym compiles
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_epem_epem'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'gensym')))
        # Check that gensym runs
        proc = subprocess.Popen('./gensym', 
                                 stdout=devnull, stdin=subprocess.PIPE,
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_epem_epem'), shell=True)
        proc.communicate('100 2 0.1 .false.\n'.encode())
        
        self.assertEqual(proc.returncode, 0)
        # Check that madevent compiles
        status = subprocess.call(['make', 'madevent'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_epem_epem'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'madevent')))


    def test_mg7_ufo_aloha(self):
        """Test mg7 (madmatrix/cudacpp) output with UFO/ALOHA.

        mg7's analogue of the Fortran ALOHA routines is the C++ ``HelAmps_sm.h``
        header: the same FFV* helicity-amplitude functions are emitted as inline
        ``ALOHAOBJ`` C++ routines. This mirrors test_madevent_ufo_aloha but for
        the mg7 backend: it checks the routines are generated, the parameters /
        process sources are present, and that the subprocess compiles (scalar
        backend) into the expected shared libraries.
        """

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('set apply_flavor_grouping False')
        self.do('import model sm')
        self.do('set group_subprocesses False')
        self.do('generate e+ e- > e+ e-')
        self.do('output mg7 %s ' % self.out_dir)

        # The C++ HelAmps header is the mg7 equivalent of the Fortran ALOHA
        # subroutines: check it exists and emits the expected FFV* routines.
        helamps = os.path.join(self.out_dir, 'src', 'HelAmps_sm.h')
        self.assertTrue(os.path.isfile(helamps),
                        'HelAmps_sm.h file is not in the mg7 src directory')
        helamps_content = open(helamps).read()
        for routine in ['FFV1_0', 'FFV1P0_3',
                        'FFV2_0', 'FFV2_3',
                        'FFV4_0', 'FFV4_3']:
            self.assertIn(routine, helamps_content,
                          '%s routine is not in HelAmps_sm.h' % routine)
        # the merged-ALOHA work emits these as C++ ALOHAOBJ routines
        self.assertIn('ALOHAOBJ', helamps_content,
                      'HelAmps_sm.h does not use the ALOHAOBJ interface')

        # Parameter / model sources and the src makefile
        for f in ['Parameters.h', 'Parameters.cc', 'makefile']:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir,
                                                        'src', f)),
                            '%s file is not in the mg7 src directory' % f)
        # mg7 cards
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'param_card.dat')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'run_card.toml')))

        # Locate the subprocess directory (P0/P1 numbering depends on the
        # invocation path, so discover it rather than hard-coding the number).
        subproc_root = os.path.join(self.out_dir, 'SubProcesses')
        candidates = [d for d in os.listdir(subproc_root)
                      if d.endswith('_epem_epem') and
                      os.path.isdir(os.path.join(subproc_root, d))]
        self.assertEqual(len(candidates), 1,
                         'expected exactly one epem_epem subprocess, got %s'
                         % candidates)
        pdir = os.path.join(subproc_root, candidates[0])
        for f in ['CPPProcess.cc', 'CPPProcess.h', 'makefile']:
            self.assertTrue(os.path.isfile(os.path.join(pdir, f)),
                            '%s file is not in the mg7 subprocess directory' % f)

        devnull = open(os.devnull, 'w')
        # Check that the subprocess compiles (scalar/no-SIMD C++ backend).
        status = subprocess.call(['make', 'bldnone'],
                                 stdout=devnull, stderr=devnull,
                                 cwd=pdir)
        self.assertEqual(status, 0)
        # The build produces the common and per-process shared libraries.
        libdir = os.path.join(self.out_dir, 'lib')
        libs = os.listdir(libdir) if os.path.isdir(libdir) else []
        self.assertTrue(any(l.startswith('libmadmatrix_common') and
                            l.endswith('.so') for l in libs),
                        'common madmatrix library not built: %s' % libs)
        self.assertTrue(any(l.startswith('libmadmatrix_') and
                            'epem_epem' in l and l.endswith('.so')
                            for l in libs),
                        'process madmatrix library not built: %s' % libs)


    def test_madevent_ufo_aloha_merged(self):
        """Test MadEvent output with UFO/ALOHA"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('set apply_flavor_grouping True')
        self.do('import model sm')
        self.do('set group_subprocesses False')
        self.do('generate e+ e- > e+ e-')
        self.do('output madevent %s ' % self.out_dir)
        # Check that the needed ALOHA subroutines are generated
        files = ['FFV6_3.f', 'FFV2_3.f', 'FFV1P1N_2.f', 'FFV6P1N_3.f', 'aloha_file.inc', 'FFV6_0.f', 'FFV2P1N_3.f', 'FFV1P0_3.f', 
                  'FFV2_0.f', 'FFV1P1N_1.f', 'FFV1_0.f', 'aloha_functions.f', 'FFV2P1N_2.f', 'FFV6P1N_1.f', 'FFV2P1N_1.f', 
                  'FFV6P1N_2.f', 'FFV1P1N_3.f']
        for f in files:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir,
                                                        'Source', 'DHELAS',
                                                        f)), 
                            '%s file is not in aloha directory' % f)
        
        #check the content of FFV1P0_0.f
        #self.check_aloha_file()
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'Cards',
                                                    'ident_card.dat')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'run_card_default.dat')))
        devnull = open(os.devnull,'w')
        # Check that the Source directory compiles
        status = subprocess.call(['make'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libgeneric.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libcernlib.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdsample.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libpdf.a')))
        # Check that gensym compiles
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_epem_epem'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'gensym')))
        # Check that gensym runs
        proc = subprocess.Popen('./gensym', 
                                 stdout=devnull, stdin=subprocess.PIPE,
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_epem_epem'), shell=True)
        proc.communicate('100 2 0.1 .false.\n'.encode())
        
        self.assertEqual(proc.returncode, 0)
        # Check that madevent compiles
        status = subprocess.call(['make', 'madevent'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_epem_epem'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'madevent')))
        
        


    def check_aloha_file(self):
        """check the content of aloha file FFV1P0_3.f and FFV2_3.f"""
        
        ffv1p0 = """C     This File is Automatically generated by ALOHA
C     The process calculated in this file is:
C     Gamma(3,2,1)
C
      SUBROUTINE FFV1P0_3(F1, F2, COUP, M3, W3,V3)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      TYPE(ALOHA) F1
      INTEGER FLV_INDEX1
      TYPE(ALOHA) F2
      INTEGER FLV_INDEX2
      REAL*8 M3
      REAL*8 P3(0:3)
      TYPE(ALOHA) V3
      REAL*8 W3
      COMPLEX*16 DENOM
      V3%P(:) = +F1%P(:)+F2%P(:)
      P3(:) = -V3 % P (:)
      FLV_INDEX1 = F1 %FLV_INDEX
      FLV_INDEX2 = F2 %FLV_INDEX
      IF(FLV_INDEX1.NE.FLV_INDEX2.OR.FLV_INDEX1.EQ.0)THEN
        V3%W(:) = (0D0,0D0)
        RETURN
      ENDIF
      DENOM = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI
     $ * W3))
      V3%W(1)= DENOM*(-CI)*(F1 % W(1)*F2 % W(3)+F1 % W(2)*F2 % W(4)+F1
     $  % W(3)*F2 % W(1)+F1 % W(4)*F2 % W(2))
      V3%W(2)= DENOM*(-CI)*(-F1 % W(1)*F2 % W(4)-F1 % W(2)*F2 % W(3)
     $ +F1 % W(3)*F2 % W(2)+F1 % W(4)*F2 % W(1))
      V3%W(3)= DENOM*(-CI)*(-CI*(F1 % W(1)*F2 % W(4)+F1 % W(4)*F2 %
     $  W(1))+CI*(F1 % W(2)*F2 % W(3)+F1 % W(3)*F2 % W(2)))
      V3%W(4)= DENOM*(-CI)*(-F1 % W(1)*F2 % W(3)-F1 % W(4)*F2 % W(2)
     $ +F1 % W(2)*F2 % W(4)+F1 % W(3)*F2 % W(1))
      END


"""
        text = open(os.path.join(self.out_dir,'Source', 'DHELAS', 'FFV1P0_3.f')).read()

        
        self.assertNotIn('OM3', text)
        ffv1p0 = [l.strip() for l in ffv1p0.strip().split('\n')]
        text = [l.strip() for l in text.strip().split('\n')]
        self.assertEqual(ffv1p0, text)
        
        ffv2 = """C     This File is Automatically generated by ALOHA
C     The process calculated in this file is:
C     Gamma(3,2,-1)*ProjM(-1,1)
C
      SUBROUTINE FFV2_3(F1, F2, COUP, M3, W3,V3)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      TYPE(ALOHA) F1
      INTEGER FLV_INDEX1
      TYPE(ALOHA) F2
      INTEGER FLV_INDEX2
      REAL*8 M3
      REAL*8 OM3
      REAL*8 P3(0:3)
      COMPLEX*16 TMP2
      TYPE(ALOHA) V3
      REAL*8 W3
      COMPLEX*16 DENOM
      OM3 = 0D0
      IF (M3.NE.0D0) OM3=1D0/M3**2
      V3%P(:) = +F1%P(:)+F2%P(:)
      P3(:) = -V3 % P (:)
      FLV_INDEX1 = F1 %FLV_INDEX
      FLV_INDEX2 = F2 %FLV_INDEX
      IF(FLV_INDEX1.NE.FLV_INDEX2.OR.FLV_INDEX1.EQ.0)THEN
        V3%W(:) = (0D0,0D0)
        RETURN
      ENDIF
      TMP2 = (F1 % W(1)*(F2 % W(3)*(P3(0)+P3(3))+F2 % W(4)*(P3(1)+CI
     $ *(P3(2))))+F1 % W(2)*(F2 % W(3)*(P3(1)-CI*(P3(2)))+F2 % W(4)
     $ *(P3(0)-P3(3))))
      DENOM = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI
     $ * W3))
      V3%W(1)= DENOM*(-CI)*(F1 % W(1)*F2 % W(3)+F1 % W(2)*F2 % W(4)
     $ -P3(0)*OM3*TMP2)
      V3%W(2)= DENOM*(-CI)*(-F1 % W(1)*F2 % W(4)-F1 % W(2)*F2 % W(3)
     $ -P3(1)*OM3*TMP2)
      V3%W(3)= DENOM*(-CI)*(-CI*(F1 % W(1)*F2 % W(4))+CI*(F1 % W(2)*F2
     $  % W(3))-P3(2)*OM3*TMP2)
      V3%W(4)= DENOM*(-CI)*(-F1 % W(1)*F2 % W(3)-P3(3)*OM3*TMP2+F1 %
     $  W(2)*F2 % W(4))
      END


C     This File is Automatically generated by ALOHA
C     The process calculated in this file is:
C     Coup(1) * (Gamma(3,2,-1)*ProjM(-1,1)) + Coup(2) *
C      (Gamma(3,2,-1)*ProjM(-1,1) + 2*Gamma(3,2,-1)*ProjP(-1,1))
C
      SUBROUTINE FFV2_4_3(F1, F2, COUP1, COUP2, M3, W3,V3)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP1
      COMPLEX*16 COUP2
      TYPE(ALOHA) F1
      INTEGER FLV_INDEX1
      TYPE(ALOHA) F2
      INTEGER FLV_INDEX2
      REAL*8 M3
      REAL*8 OM3
      REAL*8 P3(0:3)
      COMPLEX*16 TMP2
      COMPLEX*16 TMP5
      TYPE(ALOHA) V3
      REAL*8 W3
      COMPLEX*16 DENOM
      OM3 = 0D0
      IF (M3.NE.0D0) OM3=1D0/M3**2
      V3%P(:) = +F1%P(:)+F2%P(:)
      P3(:) = -V3 % P (:)
      FLV_INDEX1 = F1 %FLV_INDEX
      FLV_INDEX2 = F2 %FLV_INDEX
      IF(FLV_INDEX1.NE.FLV_INDEX2.OR.FLV_INDEX1.EQ.0)THEN
        V3%W(:) = (0D0,0D0)
        RETURN
      ENDIF
      TMP2 = (F1 % W(1)*(F2 % W(3)*(P3(0)+P3(3))+F2 % W(4)*(P3(1)+CI
     $ *(P3(2))))+F1 % W(2)*(F2 % W(3)*(P3(1)-CI*(P3(2)))+F2 % W(4)
     $ *(P3(0)-P3(3))))
      TMP5 = (F1 % W(3)*(F2 % W(1)*(P3(0)-P3(3))-F2 % W(2)*(P3(1)+CI
     $ *(P3(2))))+F1 % W(4)*(F2 % W(1)*(-P3(1)+CI*(P3(2)))+F2 % W(2)
     $ *(P3(0)+P3(3))))
      DENOM = 1D0/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI*
     $  W3))
      V3%W(1)= DENOM*(-2D0 * CI)*(COUP2*(OM3*-1D0/2D0 * P3(0)*(TMP2
     $ +2D0*(TMP5))+(+1D0/2D0*(F1 % W(1)*F2 % W(3)+F1 % W(2)*F2 % W(4))
     $ +F1 % W(3)*F2 % W(1)+F1 % W(4)*F2 % W(2)))+1D0/2D0*(COUP1*(F1 %
     $  W(1)*F2 % W(3)+F1 % W(2)*F2 % W(4)-P3(0)*OM3*TMP2)))
      V3%W(2)= DENOM*(-2D0 * CI)*(COUP2*(OM3*-1D0/2D0 * P3(1)*(TMP2
     $ +2D0*(TMP5))+(-1D0/2D0*(F1 % W(1)*F2 % W(4)+F1 % W(2)*F2 % W(3))
     $ +F1 % W(3)*F2 % W(2)+F1 % W(4)*F2 % W(1)))-1D0/2D0*(COUP1*(F1 %
     $  W(1)*F2 % W(4)+F1 % W(2)*F2 % W(3)+P3(1)*OM3*TMP2)))
      V3%W(3)= DENOM*CI*(COUP2*(OM3*P3(2)*(TMP2+2D0*(TMP5))+(+CI*(F1 %
     $  W(1)*F2 % W(4))-CI*(F1 % W(2)*F2 % W(3))-2D0 * CI*(F1 % W(3)
     $ *F2 % W(2))+2D0 * CI*(F1 % W(4)*F2 % W(1))))+COUP1*(+CI*(F1 %
     $  W(1)*F2 % W(4))-CI*(F1 % W(2)*F2 % W(3))+P3(2)*OM3*TMP2))
      V3%W(4)= DENOM*2D0 * CI*(COUP2*(OM3*1D0/2D0 * P3(3)*(TMP2+2D0
     $ *(TMP5))+(+1D0/2D0*(F1 % W(1)*F2 % W(3))-1D0/2D0*(F1 % W(2)*F2
     $  % W(4))-F1 % W(3)*F2 % W(1)+F1 % W(4)*F2 % W(2)))+1D0/2D0
     $ *(COUP1*(F1 % W(1)*F2 % W(3)+P3(3)*OM3*TMP2-F1 % W(2)*F2 % W(4))
     $ ))
      END


"""
        text = open(os.path.join(self.out_dir,'Source', 'DHELAS', 'FFV2_3.f')).read()
        #misc.sprint(text)
        self.assertIn('OM3', text)
        ffv2 = [l.strip() for l in ffv2.strip().split('\n')]
        text = [l.strip() for l in text.strip().split('\n')]
        self.assertEqual(ffv2, text) 
        
        
        
    def test_define_order(self):
        """Test the reordering of particles in the define"""

        self.do('import model sm')
        self.do('define p = u c~ g d s b~ b h')
        self.assertEqual(self.cmd._multiparticles['p'],
                         [21, 2, 1, 3, -4, 5, -5, 25])
        self.do('import model sm-no_masses')
        self.do('define p = u c~ g d s b~ b h')
        self.assertEqual(self.cmd._multiparticles['p'],
                         [21, 2, 1, 3, 5, -4, -5, 25])

    def test_madevent_decay_chain(self):
        """Test decay chain output"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('define p = u u~ d d~')
        self.do('set group_subprocesses False')
        self.do('generate p p > w+, w+ > l+ vl @1')
        self.do('output madevent %s ' % self.out_dir)
        devnull = open(os.devnull,'w')
        # Check that all subprocess directories have been created
        # (directory names use merged-particle naming convention: Q/Qx for quarks, L/Lx/N for leptons)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P1_QxQ_wp_wp_LxN')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P1_QQx_wp_wp_LxN')))
        # Check that the Source directory compiles
        status = subprocess.call(['make'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))

        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libgeneric.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libcernlib.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdsample.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libpdf.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libbias.a')))
        
        
        # Check that gensym compiles
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P1_QQx_wp_wp_LxN'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P1_QQx_wp_wp_LxN',
                                                    'gensym')))
        # Check that gensym runs
        proc = subprocess.Popen('./gensym',
                                  stdin=subprocess.PIPE, 
                                 stdout=devnull,
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P1_QQx_wp_wp_LxN'),
                                 shell=True)
        proc.communicate('100 4 0.1 .false.\n'.encode())
        
        self.assertEqual(proc.returncode, 0)
        # Check that madevent compiles
        status = subprocess.call(['make', 'madevent'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P1_QQx_wp_wp_LxN'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P1_QQx_wp_wp_LxN',
                                                    'madevent')))
        
    def test_complex_mass_SA(self):
        """ Test that the complex_mass compile in fortran """

        self.do('set apply_flavor_grouping False')
        self.do('import model sm --noprefix')
        self.do('set complex_mass_scheme')
        self.do('generate e+ e- > e+ e-')
        self.do('output standalone_fortran %s ' % self.out_dir)
        misc.compile(cwd=os.path.join(self.out_dir,'SubProcesses', 'P0_epem_epem'))
        p = subprocess.Popen(['./check'], cwd=os.path.join(self.out_dir,'SubProcesses', 'P0_epem_epem'),
                            stdout=subprocess.PIPE)
        #output = p.stdout.read()
        for line in p.stdout:
            line = line.decode('utf8')
            if 'Matrix element' in line:
                value = line.split('=')[1]
                value = value. split('GeV')[0]
                value = eval(value)
                self.assertAlmostEqual(value, 0.019538610404713896)
        
        self.do('import model sm')
        self.do('set complex_mass_scheme')
        self.do('generate e+ e- > e+ e-')
        self.do('output standalone_fortran %s -f' % self.out_dir)
        misc.compile(cwd=os.path.join(self.out_dir,'SubProcesses', 'P0_epem_epem'))
        p = subprocess.Popen(['./check'], cwd=os.path.join(self.out_dir,'SubProcesses', 'P0_epem_epem'),
                            stdout=subprocess.PIPE)
        #output = p.stdout.read()
        for line in p.stdout:
            line = line.decode('utf8')
            if 'Matrix element' in line:
                value = line.split('=')[1]
                value = value. split('GeV')[0]
                value = eval(value)
                self.assertAlmostEqual(value, 0.019538610404713896)

    def test_complex_mass_SA_merged(self):
        """ Test that the complex_mass compiles in fortran with flavor-grouped (merged) model """

        self.do('import model sm --noprefix')
        self.do('set complex_mass_scheme')
        self.do('generate e+ e- > e+ e-')
        self.do('output standalone_fortran %s ' % self.out_dir)
        subdir = os.path.join(self.out_dir, 'SubProcesses', 'P0_epem_epem')
        misc.compile(cwd=subdir)
        p = subprocess.Popen(['./check'], cwd=subdir, stdout=subprocess.PIPE)
        for line in p.stdout:
            line = line.decode('utf8')
            if 'Matrix element' in line:
                value = line.split('=')[1]
                value = value.split('GeV')[0]
                value = eval(value)
                self.assertAlmostEqual(value, 0.019538610404713896)

        self.do('import model sm')
        self.do('set complex_mass_scheme')
        self.do('generate e+ e- > e+ e-')
        self.do('output standalone_fortran %s -f' % self.out_dir)
        subdir = os.path.join(self.out_dir, 'SubProcesses', 'P0_epem_epem')
        misc.compile(cwd=subdir)
        p = subprocess.Popen(['./check'], cwd=subdir, stdout=subprocess.PIPE)
        for line in p.stdout:
            line = line.decode('utf8')
            if 'Matrix element' in line:
                value = line.split('=')[1]
                value = value.split('GeV')[0]
                value = eval(value)
                self.assertAlmostEqual(value, 0.019538610404713896)

    def test_load_feynman(self):
        """ Test that feynman gauge assignment works """
        
        self.do('import model sm')
        # check that the model is correctly loaded (has some goldstone)
        nb_goldstone = 0
        for part in self.cmd._curr_model['particles']:
            if part.get('pdg_code') in [250, 251]:
                nb_goldstone += 1
        self.assertEqual(nb_goldstone, 0)
        self.do('set gauge Feynman')
        self.do('import model sm')
        # check that the model is correctly loaded (has some goldstone)
        nb_goldstone = 0
        for part in self.cmd._curr_model['particles']:
            if part.get('pdg_code') in [250, 251]:
                nb_goldstone += 1
        self.assertEqual(nb_goldstone, 2)

    def test_check_gauge_epem_vevex_wpwm(self):
        """Test `check gauge e+ e- > ve ve~ w+ w-` includes axial and succeeds."""

        self.do('import model sm')
        with self.assertLogs('madgraph.check_cmd', level='INFO') as cm:
            self.do('check gauge e+ e- > ve ve~ w+ w-')

        log = '\n'.join(cm.output)
        self.assertIn('Gauge results (switching between Unitary/Feynman/Axial/FD gauge):', log)
        self.assertIn('Summary: 1/1 passed, 0/1 failed', log)

    def test_check_pp_wpwm(self):
        """Test `check p p > w+ w-` runs and gauge check succeeds.

        With apply_flavor_grouping on (the default), the four light-quark
        subprocesses are carried by the single merged matrix element
        Q Qx > w+ w-, so the gauge block checks one process, not four.  The
        per-flavor coverage lives in the flavor-grouping block, which compares
        the merged matrix element against the unmerged one for every flavor.
        """

        self.do('import model sm')
        with self.assertLogs('madgraph.check_cmd', level='DEBUG') as cm:
            self.do('check p p > w+ w-')

        log = '\n'.join(cm.output)
        self.assertIn('Gauge results (switching between Unitary/Feynman/Axial/FD gauge):', log)
        self.assertIn('Q Qx > w+ w-', log)
        self.assertIn('Summary: 1/1 passed, 0/1 failed', log)
        # the four flavors (both orderings) are still checked, here:
        self.assertIn('Flavor grouping check results:', log)
        self.assertIn('Summary: 8/8 passed, 0/8 failed', log)

    def test_check_gauge_pp_wpwm(self):
        """Test `check gauge p p > w+ w-` includes axial and succeeds.

        See test_check_pp_wpwm for why a single merged process is checked.
        """

        self.do('import model sm')
        with self.assertLogs('madgraph.check_cmd', level='INFO') as cm:
            self.do('check gauge p p > w+ w-')

        log = '\n'.join(cm.output)
        self.assertIn('Gauge results (switching between Unitary/Feynman/Axial/FD gauge):', log)
        self.assertIn('Q Qx > w+ w-', log)
        self.assertIn('Summary: 1/1 passed, 0/1 failed', log)

    def test_check_gauge_epem_aa_includes_axial(self):
        """Test `check gauge e+ e- > a a` includes axial gauge and succeeds."""

        self.do('import model sm')
        with self.assertLogs('madgraph.check_cmd', level='INFO') as cm:
            self.do('check gauge e+ e- > a a')

        log = '\n'.join(cm.output)
        self.assertIn('Gauge results (switching between Unitary/Feynman/Axial/FD gauge):', log)
        self.assertIn('Summary: 1/1 passed, 0/1 failed', log)
         

    def test_madevent_subproc_group(self):
        """Test MadEvent output using the SubProcess group functionality"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('define p = g u d u~ d~')
        self.do('set group_subprocesses True')
        self.do('generate g g > p p @2')
        self.do('output madevent %s ' % self.out_dir)
        self.do('set group_subprocesses False')
        devnull = open(os.devnull,'w')
        # Check that all subprocess directories have been created
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gg_gg')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gg_qq')))
        #if misc.which('gs'):
            #self.assertTrue(os.path.exists(os.path.join(self.out_dir,
            #                                        'SubProcesses',
            #                                        'P2_gg_gg',
            #                                        'matrix11.jpg')))
            #self.assertTrue(os.path.exists(os.path.join(self.out_dir,
            #                                        'HTML',
            #                                        'card.jpg')))
        # Check that the run_config.inc file has been modified correctly
        run_config = open(os.path.join(self.out_dir, 'Source',
                                       'run_config.inc')).read()
        self.assertTrue(run_config.find("ChanPerJob=2"))
        generate_events = open(os.path.join(self.out_dir, 'bin',
                                       'generate_events')).read()
        self.assertTrue(generate_events.find(\
                                            "$dirbin/refine $a $mode $n 1 $t"))
        # Check that the maxconfigs.inc file has been created properly
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'Source',
                                                    'maxconfigs.inc')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gg_qq',
                                                    'maxconfigs.inc')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'Source',
                                                    'MODEL',
                                                    'get_color.f')))
        # Check that the Source directory compiles
        status = subprocess.call(['make'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libgeneric.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libcernlib.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdsample.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libpdf.a')))
        # Check that combine_events, gen_ximprove, combine_runs 
        # compile
        #status = subprocess.call(['make', '../bin/internal/combine_events'],
        #                         stdout=devnull, 
        #                         cwd=os.path.join(self.out_dir, 'Source'))
        #self.assertEqual(status, 0)
        #self.assertTrue(os.path.exists(os.path.join(self.out_dir,
        #                                       'bin','internal', 'combine_events')))
        status = subprocess.call(['make', '../bin/internal/gen_ximprove'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'bin','internal', 'gen_ximprove')))
        # Check that gensym compiles
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P2_gg_qq'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gg_qq',
                                                    'gensym')))
        # Check that gensym runs
        proc = subprocess.Popen('./gensym', 
                                 stdout=devnull, stdin=subprocess.PIPE,
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P2_gg_qq'), shell=True)
        proc.communicate('100 4 0.1 .false.\n'.encode())
        self.assertEqual(proc.returncode, 0)
        # Check that madevent compiles
        status = subprocess.call(['make', 'madevent_forhel'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P2_gg_qq'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gg_qq',
                                                    'madevent_forhel')))
        
    def test_madevent_subproc_group_symmetry(self):
        """Check that symmetry.f gives right output"""

        def analyse(fsock):
            data = []
            for line in fsock:
                if line.strip():
                    data.append([int(i) for i in line.split()])
            return data

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model MSSM_SLHA2')
        self.do('define q = u d u~ d~')
        self.do('set group_subprocesses True')
        self.do('generate u u~ > g > go go, go > q q n1 / ur dr')
        self.do('output madevent %s ' % self.out_dir)
        self.do('set group_subprocesses False')
        devnull = open(os.devnull,'w')
        # Check that all subprocess directories have been created
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_qq_gogo_go_qqn1_go_qqn1')))
        
        target = """  1   1
 2   1
 3  -1
 4  -2
 5   1
 6  -5
 7  -1
 8  -2
 9  -1
10  -2
11  -5
12  -5 
"""

                                           
        self.assertEqual(analyse(target.split('\n')), 
                         analyse(open(os.path.join(self.out_dir,
                                           'SubProcesses',
                                           'P0_qq_gogo_go_qqn1_go_qqn1',
                                           'symfact_orig.dat'))))

        # Compile the Source directory
        status = subprocess.call(['make'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))
        self.assertEqual(status, 0)

        # Compile gensym
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_qq_gogo_go_qqn1_go_qqn1'))
        # Run gensym
        proc = subprocess.Popen('./gensym', 
                                 stdout=devnull, stdin=subprocess.PIPE,
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_qq_gogo_go_qqn1_go_qqn1'), shell=True)
        proc.communicate('100 4 0.1 .false.\n'.encode())
        self.assertEqual(proc.returncode, 0)


                
        # Check the new contents of the symfact.dat file
        self.assertEqual(analyse(open(os.path.join(self.out_dir,
                                           'SubProcesses',
                                           'P0_qq_gogo_go_qqn1_go_qqn1',
                                           'symfact.dat'))), 
                         analyse(target.split('\n')))

    def test_madevent_subproc_group_decay_chain(self):
        """Test decay chain output using the SubProcess group functionality"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('define p = g u d u~ d~')
        self.do('set group_subprocesses True')
        self.do('generate p p > w+, w+ > l+ vl @1')
        self.do('add process p p > w+ p, w+ > l+ vl @2')
        self.do('output madevent %s -nojpeg' % self.out_dir)
        self.do('set group_subprocesses False')
        devnull = open(os.devnull,'w')
        # Check that all subprocess directories have been created
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gq_wpq_wp_lvl')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gq_wpq_wp_lvl')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_qq_wpg_wp_lvl')))
        goal_subproc_mg = \
"""P2_gq_wpq_wp_lvl
P2_qq_wpg_wp_lvl
P1_qq_wp_wp_lvl
"""
        self.assertFileContains(os.path.join(self.out_dir,
                                             'SubProcesses',
                                             'subproc.mg'),
                                goal_subproc_mg)
        # Check that the Source directory compiles
        status = subprocess.call(['make'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libgeneric.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libcernlib.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdsample.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libpdf.a')))
        # Check that gensym compiles
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P2_qq_wpg_wp_lvl'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_qq_wpg_wp_lvl',
                                                    'gensym')))
        # Check that gensym runs
        proc = subprocess.Popen('./gensym', 
                                 stdout=devnull, stdin=subprocess.PIPE,
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P2_qq_wpg_wp_lvl'),
                                 shell=True)
        proc.communicate('100 4 0.1 .false.\n'.encode())
        self.assertEqual(proc.returncode, 0)
        # Check that madevent compiles
        status = subprocess.call(['make', 'madevent_forhel'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P2_qq_wpg_wp_lvl'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_qq_wpg_wp_lvl',
                                                    'madevent_forhel')))
        
    def test_ungroup_decay(self):
        """Test group_subprocesses=False for decay process"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('set group_subprocesses False')
        self.do('generate w+ > l+ vl')
        self.do('add process w+ > j j')
        self.do('output madevent %s ' % self.out_dir)
        # Check that all subprocesses have separate directories
        directories = ['P0_wp_LxN','P0_wp_QQx']
        for d in directories:
            self.assertTrue(os.path.isdir(os.path.join(self.out_dir,
                                                       'SubProcesses',
                                                       d)))
        self.do('set group_subprocesses True')
        self.do('generate w+ > l+ vl')
        self.do('add process w+ > j j')
        self.do('output madevent %s -f' % self.out_dir)
        # Check that all subprocesses are combined
        directories = ['P0_wp_lvl','P0_wp_qq']
        for d in directories:
            self.assertTrue(os.path.isdir(os.path.join(self.out_dir,
                                                       'SubProcesses',
                                                       d)))

        self.do('generate w+ > l+ vl')
        self.do('generate e+ e- > j j')
        self.do('output madevent %s -f' % self.out_dir)
        # Check that all subprocesses are combined
        directories = ['P0_wp_lvl','P0_wp_qq']
        for d in directories:
            self.assertFalse(os.path.isdir(os.path.join(self.out_dir,
                                                       'SubProcesses',
                                                       d)))
        # Check that all subprocesses are combined
        directories = ['P0_ll_qq']
        for d in directories:
            self.assertFalse(os.path.isdir(os.path.join(self.out_dir,
                                                       'SubProcesses',
                                                       d)))


    def test_ungroup_decay_mg7(self):
        """mg7 equivalent of test_ungroup_decay.

        Mirrors test_ungroup_decay for the mg7 (madmatrix/cudacpp) backend.
        Unlike madevent, mg7 uses the per-subprocess (ungrouped-style)
        directory naming regardless of the group_subprocesses flag: the
        flavour-merged madevent names (e.g. wp_lvl, wp_qq) are never produced,
        each subprocess keeps its own directory (wp_LxN, wp_QQx).
        """

        def has(d):
            return os.path.isdir(os.path.join(self.out_dir, 'SubProcesses', d))

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('set group_subprocesses False')
        self.do('generate w+ > l+ vl')
        self.do('add process w+ > j j')
        self.do('output mg7 %s ' % self.out_dir)
        # ungrouped: each subprocess in its own directory (same as madevent)
        for d in ['P0_wp_LxN', 'P0_wp_QQx']:
            self.assertTrue(has(d), '%s missing in ungrouped mg7 output' % d)

        self.do('set group_subprocesses True')
        self.do('generate w+ > l+ vl')
        self.do('add process w+ > j j')
        self.do('output mg7 %s -f' % self.out_dir)
        # mg7 does not merge subprocesses on grouping: the per-subprocess
        # directories persist and the madevent-style merged ones are absent.
        for d in ['P0_wp_LxN', 'P0_wp_QQx']:
            self.assertTrue(has(d), '%s missing in grouped mg7 output' % d)
        for d in ['P0_wp_lvl', 'P0_wp_qq']:
            self.assertFalse(has(d),
                             '%s should not exist in mg7 output (no merging)' % d)

        self.do('generate w+ > l+ vl')
        self.do('generate e+ e- > j j')
        self.do('output mg7 %s -f' % self.out_dir)
        # the second generate replaces the first: only e+ e- > j j survives
        self.assertTrue(has('P0_epem_QQx'),
                        'P0_epem_QQx missing after regenerate')
        for d in ['P0_wp_LxN', 'P0_wp_QQx', 'P0_wp_lvl', 'P0_wp_qq', 'P0_ll_qq']:
            self.assertFalse(has(d), '%s should not survive the regenerate' % d)


    def test_output_mg7_decay_subprocess_metadata(self):
        """`output mg7` of a decay must describe one initial leg, not two.

        The exporter used to hard-code two initial legs and offset the outgoing
        ones by 3, so a 1 -> n process silently produced
        incoming = [pdg, None] and lost its first outgoing particle (it landed
        on outgoing[-1], overwriting the last one). MadSpin generates its decay
        matrix elements exactly this way, so pin the metadata down.
        """
        import json

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        # Without this the leptons come out as merged-particle ids (-82, 83)
        # rather than their pdgs, which says nothing about the leg ordering.
        # It has to precede the model import, which is what builds the merges.
        self.do('set apply_flavor_grouping False')
        self.do('import model sm')
        self.do('set group_subprocesses False')
        self.do('generate t > b w+, w+ > e+ ve')
        self.do('output mg7 %s' % self.out_dir)

        with open(pjoin(self.out_dir, 'SubProcesses',
                        'subprocesses.json')) as fsock:
            subprocesses = json.load(fsock)
        self.assertEqual(len(subprocesses), 1)
        subproc = subprocesses[0]

        self.assertEqual(subproc['incoming'], [6])
        # The b is the first outgoing leg: it is the one the old offset dropped.
        self.assertEqual(subproc['outgoing'], [5, -11, 12])
        # No beam pair, so no beam-swapped mirror configuration.
        self.assertFalse(any(flav['mirror'] for flav in subproc['flavors']))
        # The channel topology must hang off the single incoming line i0.
        edges = set(edge for channel in subproc['channels']
                    for vertex in channel['vertices'] for edge in vertex)
        self.assertIn('i0', edges)
        self.assertNotIn('i1', edges)

        with open(pjoin(self.out_dir, 'SubProcesses',
                        'proc_characteristics')) as fsock:
            characteristics = fsock.read()
        self.assertIn('ninitial = 1', characteristics)
        self.assertIn('nexternal = 4', characteristics)

    def test_output_mg7_decay_run_card_has_no_cuts(self):
        """`output mg7` of a decay must ship a run card without any cut.

        A partial width is inclusive, so any kinematic cut biases it low: the
        hadron-collider defaults (ptj/ptl/eta/dR) used to survive into a decay
        directory and cost ~3% on the t > b w+, w+ > e+ ve width. The card is
        written at output time, so the emitted [cuts] section is what has to be
        empty -- the user still sees exactly what is run, and can add a cut back
        by hand. A collision must keep its defaults untouched.
        """
        import madgraph.various.banner as banner_mod

        def cuts_of(path):
            card = banner_mod.RunCardMG7(path, consistency=False)
            return card['cuts']

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('generate t > b w+, w+ > e+ ve')
        self.do('output mg7 %s' % self.out_dir)

        # both the working card and the "set <param> default" reference
        for name in ('run_card.toml', 'run_card_default.toml'):
            cuts = cuts_of(pjoin(self.out_dir, 'Cards', name))
            self.assertEqual(dict(cuts), {},
                             '%s of a 1 -> n decay must carry no cut, got %s'
                             % (name, dict(cuts)))

        # the Breit-Wigner cutoff is a sampling range for the off-shell
        # propagators, not a cut on the final state: it must survive.
        card = banner_mod.RunCardMG7(pjoin(self.out_dir, 'Cards',
                                           'run_card.toml'), consistency=False)
        self.assertEqual(card['phasespace']['bw_cutoff'], 15)

        # a 2 -> n collision keeps the standard cuts
        shutil.rmtree(self.out_dir)
        self.do('generate p p > t t~')
        self.do('output mg7 %s' % self.out_dir)
        cuts = cuts_of(pjoin(self.out_dir, 'Cards', 'run_card.toml'))
        self.assertEqual(cuts['jet-pt']['min'], 20.0)
        self.assertEqual(cuts['lepton-pt']['min'], 10.0)
        self.assertEqual(cuts['jet-eta_abs']['max'], 5.0)

    @test_manager.bypass_for_py3
    def test_madevent_triplet_diquarks(self):
        """Test MadEvent output of triplet diquarks"""

        self.do('import model triplet_diquarks')
        self.do('set group_subprocesses False')
        self.do('generate u t > trip~ > u t g')
        self.do('output %s ' % self.out_dir)

        devnull = open(os.devnull,'w')
        # Check that the Source directory compiles
        status = subprocess.call(['make'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libgeneric.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libcernlib.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdsample.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libpdf.a')))
        # Check that gensym compiles
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_ut_tripx_utg'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_ut_tripx_utg',
                                                    'gensym')))
        # Check that gensym runs
        proc = subprocess.Popen('./gensym', 
                                 stdout=devnull, stdin=subprocess.PIPE,
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_ut_tripx_utg'), shell=True)
        proc.communicate('100 4 0.1 .false.\n'.encode())
        self.assertEqual(proc.returncode, 0)
        
        # Check that madevent compiles
        status = subprocess.call(['make', 'madevent'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_ut_tripx_utg'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_ut_tripx_utg',
                                                    'madevent')))
        
    def test_leshouche_sextet_diquarks(self):
        """Test leshouche.inc output of sextet diquarks"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        devnull = open(os.devnull,'w')

        # Test sextet production
        self.do('import model sextet_diquarks')
        self.do('set group_subprocesses False')
        self.do('generate u u > six g')
        self.do('output madevent %s ' % self.out_dir)
        
        # Check that leshouche.inc exists
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_uu_sixg',
                                                    'leshouche.inc')))        
        # Test sextet decay
        self.do('generate six > u u g')
        self.do('output madevent %s -f' % self.out_dir)

        # Check that leshouche.inc exists
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_six_uug',
                                                    'leshouche.inc')))        

        # Test sextet production
        self.do('generate u g > six u~')
        self.do('output madevent %s -f' % self.out_dir)
        
        # Check that leshouche.inc exists
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_ug_sixux',
                                                    'leshouche.inc')))

    @unittest.expectedFailure
    def test_leshouche_sextet_diquarks_mg7(self):
        """mg7 equivalent of test_leshouche_sextet_diquarks (TODO).

        leshouche.inc is a MadEvent Fortran artifact (Les Houches colour/flavour
        table for the Fortran integrator); the mg7 (madmatrix/cudacpp) export
        does not produce it. This test is xfail to keep mg7 leshouche support on
        the to-do list: when mg7 grows an equivalent it will start passing and
        the expectedFailure decorator should be removed.
        """
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sextet_diquarks')
        self.do('set group_subprocesses False')
        self.do('generate u u > six g')
        self.do('output mg7 %s ' % self.out_dir)

        # Discover the sextet subprocess directory (numbering is invocation
        # dependent) and require a leshouche.inc -- currently absent in mg7.
        sub_root = os.path.join(self.out_dir, 'SubProcesses')
        cand = [d for d in os.listdir(sub_root)
                if d.endswith('_sixg') and
                os.path.isdir(os.path.join(sub_root, d))]
        self.assertTrue(cand, 'no u u > six g subprocess in mg7 output')
        self.assertTrue(os.path.exists(os.path.join(sub_root, cand[0],
                                                    'leshouche.inc')),
                        'leshouche.inc not generated by mg7 (TODO)')

    def test_ufo_standard_sm(self):
        """ check that we can use standard MG4 name """
        self.do('import model sm')
        self.do('generate mu+ mu- > ta+ ta-')       

    def test_decay_chain_identical_particle_outoforder(self):
        """ check that we can use standard MG4 name """
        
        self.do('import model sm')
        self.do('generate e+ e- > z z h, h > b b~, z > u u~, z > e+ e-')
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.do('output madevent %s ' % self.out_dir)
        Pdir = os.listdir(pjoin(self.out_dir, 'SubProcesses')) 
        self.assertNotIn('P0_ll_zzh_z_ll_z_ll_h_bbx',  Pdir)


    def run_standalone(self,commands):

        for command in commands:
            self.do(command)
        self.do('output standalone_fortran %s -f' % self.out_dir)
        Pdir = None
        for pdir in misc.glob('P*', pjoin(self.out_dir, 'SubProcesses')):
            Pdir = pdir
            break 
        subprocess.call(['make', 'check'], cwd=Pdir, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        stdout = subprocess.Popen(['./check'], cwd=Pdir,
                            stdout=subprocess.PIPE).communicate()[0].decode('utf8')
        value = None
        for line in stdout.split('\n'):
            if 'Matrix element' in line:
                value = line.split('=')[1]
                value = float(value.split('GeV')[0])
        return value

    def test_decay_chain_symmetry_factor(self):
        """ check that flavor symmetry factor matches the unflavor case """

        cmd = ['set apply_flavor_grouping False',
               'import model sm ',
               'generate e+ e- > z z, z > e+ e-, z > e+ e-']
        flavor_value = self.run_standalone(cmd)
        cmd[0] = 'set apply_flavor_grouping True'
        unflavor_value = self.run_standalone(cmd)
        self.assertAlmostEqual(flavor_value/unflavor_value, 1.0, places=5)
        # Absolute value check (apply_flavor_grouping=False reference)
        self.assertAlmostEqual(flavor_value/1.4452059645560334e-15, 1.0, places=5)

        #######################################################################
        cmd[0] = 'set apply_flavor_grouping False' 
        cmd[2] = 'generate e+ e- > z z, z > mu+ mu-, z > e+ e-'
        flavor_value = self.run_standalone(cmd)
        cmd[0] = 'set apply_flavor_grouping True'
        unflavor_value = self.run_standalone(cmd)
        self.assertAlmostEqual(flavor_value/unflavor_value, 1.0, places=5)
        self.assertAlmostEqual(flavor_value/2.8904119291120669e-15, 1.0, places=5)

        #######################################################################
        # Two Z bosons decaying to different quark/lepton species: tests that
        # the decay-tree fingerprint fix correctly sets COMP_OLD=1 for the
        # no-grouping case (preventing a spurious factor of 2).
        cmd[0] = 'set apply_flavor_grouping False'
        cmd[2] = 'generate e+ e- > z z, z > u u~, z > e+ e-'
        flavor_value = self.run_standalone(cmd)
        cmd[0] = 'set apply_flavor_grouping True'
        unflavor_value = self.run_standalone(cmd)
        self.assertAlmostEqual(flavor_value/unflavor_value, 1.0, places=5)
        self.assertAlmostEqual(flavor_value/1.3222299076541260e-14, 1.0, places=5)

        #######################################################################
        # Two Z bosons decaying to different quarks (both merge to _quark):
        # tests that no-grouping COMP_OLD=1 while grouping COMP_OLD=2 with
        # runtime correction, giving the same physical result.
        cmd[0] = 'set apply_flavor_grouping False'
        cmd[2] = 'generate e+ e- > z z, z > d d~, z > s s~'
        flavor_value = self.run_standalone(cmd)
        cmd[0] = 'set apply_flavor_grouping True'
        unflavor_value = self.run_standalone(cmd)
        self.assertAlmostEqual(flavor_value/unflavor_value, 1.0, places=5)
        self.assertAlmostEqual(flavor_value/1.5703482894659815e-14, 1.0, places=5)

        #######################################################################
        # Two Z bosons decaying to identical quarks (COMP_OLD=2 in both cases).
        cmd[0] = 'set apply_flavor_grouping False'
        cmd[2] = 'generate e+ e- > z z, z > u u~, z > u u~'
        flavor_value = self.run_standalone(cmd)
        cmd[0] = 'set apply_flavor_grouping True'
        unflavor_value = self.run_standalone(cmd)
        self.assertAlmostEqual(flavor_value/unflavor_value, 1.0, places=5)
        self.assertAlmostEqual(flavor_value/1.0543959064905002e-14, 1.0, places=5)

        #######################################################################
        cmd[0] = 'set apply_flavor_grouping False'
        cmd[2] = 'generate e+ e- > z a, z > e+ e-, a > e+ e-'
        flavor_value = self.run_standalone(cmd)
        cmd[0] = 'set apply_flavor_grouping True'
        unflavor_value = self.run_standalone(cmd)
        self.assertAlmostEqual(flavor_value/unflavor_value, 1.0, places=5)
        self.assertAlmostEqual(flavor_value/1.9524089070808569e-14, 1.0, places=5)

        #######################################################################
        cmd[0] = 'set apply_flavor_grouping False'
        cmd[2] = 'generate e+ e- > z a, z > mu+ mu-, a > e+ e-'
        flavor_value = self.run_standalone(cmd)
        cmd[0] = 'set apply_flavor_grouping True'
        unflavor_value = self.run_standalone(cmd)
        self.assertAlmostEqual(flavor_value/unflavor_value, 1.0, places=5)
        self.assertAlmostEqual(flavor_value/1.9524089070808569e-14, 1.0, places=5)
        #######################################################################
        cmd[0] = 'set apply_flavor_grouping False'
        cmd[2] = 'generate e+ e- > z a, z > mu+ mu-, a > u u~'
        flavor_value = self.run_standalone(cmd)
        cmd[0] = 'set apply_flavor_grouping True'
        unflavor_value = self.run_standalone(cmd)
        self.assertAlmostEqual(flavor_value/unflavor_value, 1.0, places=5)
        self.assertAlmostEqual(flavor_value/2.6032118761078096e-14, 1.0, places=5)
        #######################################################################
        cmd[0] = 'set apply_flavor_grouping False'
        cmd[2] = 'generate e+ e- > z a, z > mu+ mu-, a > t t~'
        flavor_value = self.run_standalone(cmd)
        cmd[0] = 'set apply_flavor_grouping True'
        unflavor_value = self.run_standalone(cmd)
        self.assertAlmostEqual(flavor_value/unflavor_value, 1.0, places=5)
        self.assertAlmostEqual(flavor_value/3.9704302268721535e-14, 1.0, places=5)
        ######################################################################
        cmd[0] = 'set apply_flavor_grouping False'
        cmd[2] = 'generate e+ e- > z z h, z > u u~, z > e+ e-, h > b b~'
        flavor_value = self.run_standalone(cmd)
        cmd[0] = 'set apply_flavor_grouping True'   
        unflavor_value = self.run_standalone(cmd)

        self.assertAlmostEqual(flavor_value/unflavor_value, 1.0, places=5)
        self.assertAlmostEqual(flavor_value/2.1127915184144537e-27, 1.0, places=5)
        ######################################################################
        cmd[0] = 'set apply_flavor_grouping False'
        cmd[2] = 'generate e+ e- > z z h, h > b b~, z > u u~, z > e+ e-'
        flavor_value = self.run_standalone(cmd)
        cmd[0] = 'set apply_flavor_grouping True'
        unflavor_value = self.run_standalone(cmd)

        self.assertAlmostEqual(flavor_value/unflavor_value, 1.0, places=5)
        self.assertAlmostEqual(flavor_value/2.1127915184144537e-27, 1.0, places=5)
        
        ######################################################################
        cmd[0] = 'set apply_flavor_grouping False'
        cmd[2] = 'generate e+ e- > t t~ z, z > e+ e-, (t > z t, z > e+ e- ), (t~ > t~ z, z > e+ e- )'
        flavor_value = self.run_standalone(cmd)
        cmd[0] = 'set apply_flavor_grouping True'
        unflavor_value = self.run_standalone(cmd)
        
        self.assertAlmostEqual(flavor_value/unflavor_value, 1.0, places=5)
        ######################################################################
        cmd[0] = 'set apply_flavor_grouping False'
        cmd[2] = 'generate e+ e- > t t~ z, z > e+ e-, (t > z t, z > e+ e- ), (t~ > t~ z, z > mu+ mu- )'
        flavor_value = self.run_standalone(cmd)
        cmd[0] = 'set apply_flavor_grouping True'
        unflavor_value = self.run_standalone(cmd)

        self.assertAlmostEqual(flavor_value/unflavor_value, 1.0, places=5)
        ######################################################################
        cmd[0] = 'set apply_flavor_grouping False'
        cmd[2] = 'generate e+ e- > t t~ z, z > d d~, (t > z t, z > e+ e- ), (t~ > t~ z, z > mu+ mu- )'
        flavor_value = self.run_standalone(cmd)
        cmd[0] = 'set apply_flavor_grouping True'
        unflavor_value = self.run_standalone(cmd)

        self.assertAlmostEqual(flavor_value/unflavor_value, 1.0, places=5)


    def test_save_load(self):
        """ check that we can use standard MG4 name """
        
        self.do('set apply_flavor_grouping False')
        self.do('import model sm')
        self.assertEqual(len(self.cmd._curr_model.get('particles')), 17)
        self.assertEqual(len(self.cmd._curr_model.get('interactions')), 56)
        self.do('set apply_flavor_grouping True')
        self.do('import model sm')
        self.assertEqual(len(self.cmd._curr_model.get('particles')), 20)
        self.assertEqual(len(self.cmd._curr_model.get('interactions')), 39)        
        #self.do('save model /tmp/model.pkl')
        self.do('import model sm')
        #self.do('load model /tmp/model.pkl')
        self.assertEqual(len(self.cmd._curr_model.get('particles')), 20)
        self.assertEqual(len(self.cmd._curr_model.get('interactions')), 39)
        self.do('generate mu+ mu- > ta+ ta-') 
        self.assertEqual(len(self.cmd._curr_amps), 1)
        nicestring = """Process: mu+ mu- > ta+ ta- WEIGHTED<=4
2 diagrams:
1  ((1(82),2(-82)>1(22),id:34),(3(-15),4(15),1(22),id:36)) (QCD=0,QED=2,WEIGHTED=4)
2  ((1(82),2(-82)>1(23),id:40),(3(-15),4(15),1(23),id:42)) (QCD=0,QED=2,WEIGHTED=4)"""
        self.do('generate e+ e- > ta+ ta-') 
        self.assertEqual(len(self.cmd._curr_amps), 1)
        nicestring = """Process: e+ e- > ta+ ta- WEIGHTED<=4
2 diagrams:
1  ((1(82),2(-82)>1(22),id:34),(3(-15),4(15),1(22),id:36)) (QCD=0,QED=2,WEIGHTED=4)
2  ((1(82),2(-82)>1(23),id:40),(3(-15),4(15),1(23),id:42)) (QCD=0,QED=2,WEIGHTED=4)"""


        #self.assertEqual(self.cmd._curr_amps[0].nice_string().split('\n'), nicestring.split('\n'))
        #self.do('save processes /tmp/model.pkl')
        #self.do('generate e+ e- > e+ e-')
        #self.do('load processes /tmp/model.pkl')
        #self.assertEqual(len(self.cmd._curr_amps), 1)
        #self.assertEqual(self.cmd._curr_amps[0].nice_string(), nicestring)
        
        #os.remove('/tmp/model.pkl')
        
    def test_standalone_cpp_output(self):
        """Test the C++ standalone output"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('generate e+ e- > e+ e- @2')
        self._output_standalone_cpp(self.out_dir)

        # Check that all needed src files are generated
        files = ['HelAmps_sm.h', 'HelAmps_sm.cc', 'Makefile',
                 'Parameters_sm.h', 'Parameters_sm.cc',
                 'rambo.h', 'rambo.cc', 'read_slha.h', 'read_slha.cc']

        for f in files:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir,
                                                       'src',
                                                        f)), 
                            '%s file is not in aloha directory' % f)

        devnull = open(os.devnull,'w')
        # Check that the Model and Aloha output has compiled
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel_sm.a')))

        # Locate the subprocess directory: the merge shortened the C++
        # directory name (P2_Sigma_sm_epem_epem -> P2_epem_epem), so discover it.
        proc_root = os.path.join(self.out_dir, 'SubProcesses')
        candidates = [d for d in os.listdir(proc_root)
                      if d.endswith('_epem_epem') and
                      os.path.isdir(os.path.join(proc_root, d))]
        self.assertEqual(len(candidates), 1,
                         'expected exactly one epem_epem subprocess, got %s'
                         % candidates)
        proc_dir = os.path.join(proc_root, candidates[0])
        # Check that check_sa.cpp compiles
        subprocess.call(['make', 'check'],
                        stdout=devnull, stderr=devnull,
                        cwd=proc_dir)


        self.assertTrue(os.path.exists(os.path.join(proc_dir, 'check')))

        # Check that the output of check is correct
        logfile = os.path.join(proc_dir, 'check.log')

        subprocess.call('./check',
                        stdout=open(logfile, 'w'), stderr=devnull,
                        cwd=proc_dir, shell=True)

        log_output = open(logfile, 'r').read()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.e\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 1.953735e-2)
        
    def test_import_banner_command(self):
        """check that the import banner command works"""
        
        cwd = os.getcwd()
        os.chdir(self.tmpdir)
        self.do('import banner %s --no_launch' % pjoin(MG5DIR, 'tests', 'input_files', 'tt_banner.txt'))
        
        # check that the output exists:
        self.assertTrue(os.path.exists(self.out_dir))
        
        # check that the Cards have been modified
        run_card = open(pjoin(self.out_dir,'Cards','run_card.dat')).read()
        self.assertIn("'tt'     = run_tag", run_card)
        self.assertIn("200       = nevents", run_card)
        os.chdir(cwd)
        


#===============================================================================
# IOTestFDGauge
#===============================================================================
class IOTestFDGauge(IOTests.IOTestManager):
    """Reference files for the FD gauge output, on both backends.

    FD gauge writes wavefunctions and routines that no other gauge does: a
    massive vector and its Goldstone share one 5 component object, the
    propagator factor is part of the routine, and the Lorentz structures of a
    vertex are assembled even though they act on different spins. None of that
    was covered by a reference file, and three madmatrix bugs (a Goldstone sign,
    a vector constant broadcast to the first lane only, and a gauge direction
    that differed between the scalar and vector branches) lived in the released
    templates because of it. Two of them show up as a wrong number, which
    test_madmatrix_fd_vs_fortran and test_madmatrix_fd_simd_lanes now
    catch; the third is a gauge choice, invisible to any matrix element, and
    only a reference file can hold it still.

    u u~ > w+ w- is small and covers what is specific to FD: the merged flavour
    routines, a same spin combination (FFV6_2M_3), two mixed spin ones
    (VVV1_VVS1_VSV2_VSS1_0 and VVV1_VSV2_VSS2_SVV2_SVS2_SSV3_0) and the inlined
    propagator factor of every offshell V/S routine.
    """

    def setUp(self):
        super(IOTestFDGauge, self).setUp()
        self.interface = Cmd.MasterCmd()
        self.interface.no_notification()

    def generate_fd(self, output_format, path):
        """u u~ > w+ w- in FD gauge, in the given output format"""
        cmds = ['import model sm',
                'set gauge FD',
                'generate u u~ > w+ w-',
                'output %s %s -f' % (output_format, path)]
        try:
            for cmd in cmds:
                self.interface.exec_cmd(cmd, errorhandling=False, printcmd=False,
                                        precmd=True, postcmd=True)
        finally:
            # the gauge is global: do not leak it into the next test
            self.interface.exec_cmd('set gauge unitary', errorhandling=False,
                                    printcmd=False, precmd=True, postcmd=True)

    @IOTests.createIOTest()
    def testIO_FDgauge_standalone_fortran(self):
        r""" target: FD_fortran/Source/DHELAS/FFV6M_3.f
             target: FD_fortran/Source/DHELAS/VVV1_0.f
        """
        # Two files rather than the whole DHELAS, chosen to cover what FD gauge
        # does that no other gauge does:
        #  - FFV6M_3.f: the propagator factor written into the body of an
        #    offshell routine (q, the gauge direction, js1/js2 and the update of
        #    the 5 components), and a combination merged into one expression
        #    (FFV6_2M_3);
        #  - VVV1_0.f: the combinations assembled out of structures that act on
        #    different spins (VVV1_VVS1_VSV2_VSS1_0 and
        #    VVV1_VSV2_VSS2_SVV2_SVS2_SSV3_0), which only exist because a
        #    massive vector and its Goldstone are the same wavefunction here.
        self.generate_fd('standalone_fortran', pjoin(self.IOpath, 'FD_fortran'))

    @IOTests.createIOTest()
    def testIO_FDgauge_madmatrix(self):
        r""" target: FD_madmatrix/src/HelAmps_sm.h
        """
        # one file holds all of it for madmatrix: the generated routines and,
        # pasted above them, the FD helpers of helas_fd.h
        self.generate_fd('standalone', pjoin(self.IOpath, 'FD_madmatrix'))
