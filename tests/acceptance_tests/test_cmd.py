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
from __future__ import absolute_import
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

import madgraph.interface.master_interface as Cmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.iolibs.files as files
import madgraph.core.diagram_generation as diagram_generation
import madgraph.various.misc as misc
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
        self.do('display diagrams . --generate_only')
        self.assertTrue(os.path.exists('./diagrams_0_epem_epem.eps'))
        os.remove('./diagrams_0_epem_epem.eps')
        
        self.do('generate g g > g g')
        self.do('display diagrams . --generate_only')
        self.assertTrue(os.path.exists('diagrams_0_gg_gg.eps'))
        os.remove('diagrams_0_gg_gg.eps')
        self.do('set group_subprocesses True')
        
    def test_config(self):
        """check that configuration file is at default value"""
        self.maxDiff=None
        self.cmd.options = {} #reset to None
        config = self.cmd.set_configuration(MG5DIR+'/input/.mg5_configuration_default.txt', final=False)
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
                    'td_path': './td', 
                    'delphes_path': './Delphes', 
                    'default_unset_couplings': 99,
                    'checkpointing': False,
                    'cluster_type': 'condor', 
                    'cluster_requirement': None,
                    'cluster_vacatetime': '120',
                    'enforce_shared_disk': False,
                    'cluster_status_update': (600, 30),
                    'madanalysis_path': './MadAnalysis', 
                    'cluster_temp_path': None, 
                    'fortran_compiler': None, 
                    'cpp_compiler': None,
                    'exrootanalysis_path': './ExRootAnalysis', 
                    'eps_viewer': None, 
                    'automatic_html_opening': True, 
                    'pythia8_path': './HEPTools/pythia8',
                    'mg5amc_py8_interface_path': './HEPTools/MG5aMC_PY8_interface',
                    'madanalysis5_path': './HEPTools/madanalysis5/madanalysis5',
                    'group_subprocesses': 'Auto',
                    'complex_mass_scheme': False,
                    'gauge': 'unitary',
                    'output_dependencies': 'external',
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
                    'hwpp_path': './herwigPP',
                    'thepeg_path': './thepeg',
                    #'applgrid': 'applgrid-config',
                    'pineappl': 'pineappl',
                    'cluster_size': 100,
                    'loop_color_flows': False,
                    'cluster_local_path': None,
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

    def do(self, line, force=False):
        """ exec a line in the cmd under test """
        if force:        
            self.cmd.exec_cmd(line, force=force)
        else:   
           self.cmd.exec_cmd(line) 
    
    def test_output_madevent_directory(self):
        """Test outputting a MadEvent directory"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)
        
        self.cmd.do_import('model sm', force=True)
        self.do('set group_subprocesses False')
        self.do('generate e+ e- > e+ e-')
#        self.do('load processes %s' % self.join_path(_pickle_path,'e+e-_e+e-.pkl'))
        self.do('output %s -nojpeg' % self.out_dir)
        
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
                                                 'Cards', 'plot_card_default.dat')))
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
        self.do('output %s -f' % self.out_dir)
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
        self.do('output standalone %s' % self.out_dir)
        self.do('set group_subprocesses True')
        self.assertTrue(os.path.exists(self.out_dir))
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, 'lib', 'libdhelas.a')))
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, 'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'SubProcesses', 'P0_epem_epem')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'Cards', 'proc_card_mg5.dat')))
    
    def test_custom_propa(self):
        """check that using custom propagator is working"""
        
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        path = os.path.join(MG5DIR, 'tests', 'input_files', 'sm_with_custom_propa')
        self.do('import model %s' % path)
        self.do('generate g g > t t~')
        self.do('output standalone %s ' % self.out_dir)        
        
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
        self.do('output standalone %s ' % self.out_dir)
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
        self.do('output standalone %s ' % self.out_dir)
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
        self.do('output standalone %s ' % self.out_dir)

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
        self.do('output standalone %s ' % self.out_dir)

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

    def test_standalone_flavor_mask(self):
        """Acceptance test for the per-flavor masking optimization.

        Generates p p > j j QCD=0 and, for the q q~ > q q~ subprocess,
        exercises both the Fortran (standalone) and C++ (standalone_cpp)
        backends. The check_sa driver is patched to also evaluate two
        non-representative flavors -- s c~ > s c~ (flavor 3 4 3 4) and
        s c~ > c c~ (flavor 3 4 4 4) -- and the matrix-element source is
        patched to print the runtime flavor mask that gates the HELAS
        calls. The test asserts that

          * s c~ > s c~ reproduces the d u~ > d u~ value (PDG 1 -2 1 -2),
          * s c~ > c c~ vanishes,
          * the flavor mask is partial for s c~ > s c~ (a flavor present
            in the runtime flavor table) and fully on for s c~ > c c~ (a
            lookup miss that falls back to the safe all-on mask).
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
            known = parse_mask(text, known_flavor)
            self.assertIsNotNone(known, 'no MASKDBG line for %s' % (known_flavor,))
            self.assertNotEqual(known[0], known[1],
                                'mask for %s should be partial' % (known_flavor,))
            miss = parse_mask(text, zero_flavor)
            self.assertIsNotNone(miss, 'no MASKDBG line for %s' % (zero_flavor,))
            self.assertEqual(miss[0], miss[1],
                             'mask for %s should be all-on' % (zero_flavor,))

        # ---- Fortran standalone -------------------------------------
        self.do('output standalone %s -f' % self.out_dir)
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
        shutil.rmtree(self.out_dir)
        self.do('output standalone_cpp %s -f' % self.out_dir)
        proc_dir = find_qqx(pjoin(self.out_dir, 'SubProcesses'))

        check_cpp = pjoin(proc_dir, 'check_sa.cpp')
        src = open(check_cpp).read()
        m = re.search(r'maxflavor\s*=\s*(\d+)', src)
        self.assertTrue(m, 'maxflavor not found in check_sa.cpp')
        nflav = int(m.group(1))
        out_lines = []
        for line in src.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith('static const int maxflavor'):
                line = line.replace('= %d' % nflav, '= %d' % (nflav + 2))
            elif stripped.startswith('static const int flavor_arr'):
                line = (line.replace('[%d][4]' % nflav, '[%d][4]' % (nflav + 2))
                        .replace('}};', '}, {2, 3, 2, 3}, {2, 3, 3, 3}};'))
            elif stripped.startswith('static const int pdg_arr'):
                line = (line.replace('[%d][4]' % nflav, '[%d][4]' % (nflav + 2))
                        .replace('}};', '}, {3, -4, 3, -4}, {3, -4, 4, -4}};'))
            out_lines.append(line)
        open(check_cpp, 'w').write(''.join(out_lines))

        cpp_proc = pjoin(proc_dir, 'CPPProcess.cc')
        src = open(cpp_proc).read()
        self.assertIn('#include "CPPProcess.h"', src)
        src = src.replace('#include "CPPProcess.h"',
                          '#include <iostream>\n#include "CPPProcess.h"', 1)
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

    def test_standalone_cpp(self):
        """test that standalone cpp is working"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model MSSM_SLHA2-full')
        self.do('generate g g > go go QED=2')
        self.do('output standalone_cpp %s ' % self.out_dir)
        devnull = open(os.devnull,'w')
    
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_Sigma_MSSM_SLHA2_full_gg_gogo',
                               'check.log')
        # Check that check_sa.cc compiles
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_Sigma_MSSM_SLHA2_full_gg_gogo'))
        
        subprocess.call('./check', 
                        stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_Sigma_MSSM_SLHA2_full_gg_gogo'), shell=True)
    
        log_output = open(logfile, 'r').read()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)
        
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 6.4739191,5)
    
    
    def test_standalone_cpp_output_consistency(self):
        """test that standalone cpp is working"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        #step 0 cpp output
        self.do('generate p p > t t~, t > b mu+ vm, t~ > b~ mu- vm~')
        self.do('output standalone_cpp %s ' % self.out_dir)
        devnull = open(os.devnull,'w')
    
        directories= ['P0_Sigma_sm_gg_bmupvmbxmumvmx', 'P0_Sigma_sm_QQx_bmupvmbxmumvmx']
        def get_values():
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
        self.do('output standalone %s -f' % self.out_dir)
        shutil.rmtree(self.out_dir)            
        self.do('output standalone_cpp %s -f' % self.out_dir)     
        new = get_values()
        
        for i,_ in enumerate(original):
            self.assertEqual(original[i], new[i])

    def test_standalone_cpp_fd_output_consistency(self):
        """test standalone_cpp in FD gauge against standalone"""

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
            if output_format == 'standalone':
                subprocess.call(['make'],
                                stdout=devnull, stderr=devnull,
                                cwd=os.path.join(self.out_dir, 'Source'))
            for oneproc in directories:
                logfile = os.path.join(proc_dir, oneproc, 'check.log')
                target = ['make', 'check'] if output_format == 'standalone' else ['make']
                subprocess.call(target,
                                stdout=devnull, stderr=devnull,
                                cwd=os.path.join(proc_dir, oneproc))
                subprocess.call('./check %s' % energy,
                                stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                                cwd=os.path.join(proc_dir, oneproc), shell=True)
                log_output = open(logfile, 'r').read()
                me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                                   re.IGNORECASE)
                me_groups = me_re.findall(log_output)
                self.assertTrue(me_groups)
                values.extend(float(value) for value in me_groups)
            return values

        standalone_cpp = get_values('standalone_cpp')
        shutil.rmtree(self.out_dir)
        standalone = get_values('standalone')

        self.assertEqual(len(standalone_cpp), len(standalone))
        for i, _ in enumerate(standalone_cpp):
            self.assertAlmostEqual(standalone_cpp[i], standalone[i])

        self.do('set gauge unitary')
        self.do('generate _quark _quark > h _quark _quark _quark _anti_quark  QCD=0')
        devnull = open(os.devnull,'w')
        energy = '1000'     
        
        shutil.rmtree(self.out_dir)
        standalone_cpp_no_fd = get_values('standalone_cpp')
        shutil.rmtree(self.out_dir)
        standalone_no_fd = get_values('standalone')

        self.assertEqual(len(standalone_cpp_no_fd), len(standalone_no_fd))
        for i, _ in enumerate(standalone_cpp_no_fd):
            self.assertAlmostEqual(standalone_cpp_no_fd[i], standalone_no_fd[i])
        for i, _ in enumerate(standalone_cpp_no_fd):
            self.assertAlmostEqual(standalone_cpp_no_fd[i], standalone[i])


         
    def test_v4_heft(self):
        """Test standalone directory for UFO HEFT model"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model heft', force=True)
        self.do('generate g g > h g g')
        self.do('output standalone %s ' % self.out_dir)

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
        self.do('output %s ' % self.out_dir)
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
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'plot_card_default.dat')))
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
        
        
     
    def test_madevent_ufo_aloha_merged(self):
        """Test MadEvent output with UFO/ALOHA"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('set apply_flavor_grouping True')
        self.do('import model sm')
        self.do('set group_subprocesses False')
        self.do('generate e+ e- > e+ e-')
        self.do('output %s ' % self.out_dir)
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
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'plot_card_default.dat')))
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
C     Gamma(3,2,-1)*ProjM(-1,1)
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
      TYPE(ALOHA) V3
      TYPE(ALOHA) VTMP
      REAL*8 W3
      COMPLEX*16 DENOM
      INTEGER*4 I
      CALL FFV2_3(F1,F2,COUP1,M3,W3,V3)
      CALL FFV4_3(F1,F2,COUP2,M3,W3,VTMP)
      DO I = 1, 4
        V3 %W(I) = V3%W(I) + VTMP%W(I)
      ENDDO
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
        self.do('output standalone %s ' % self.out_dir)
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
        self.do('output standalone %s -f' % self.out_dir)
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
        self.do('output standalone %s ' % self.out_dir)
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
        self.do('output standalone %s -f' % self.out_dir)
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
        self.do('output %s ' % self.out_dir)
        self.do('set group_subprocesses False')
        devnull = open(os.devnull,'w')
        # Check that all subprocess directories have been created
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_qq_gogo_go_qqn1_go_qqn1')))
        
        target=""" 1   1
 2   1
 3  -1
 4  -2
 5  -2
 6   1
 7  -6
 8  -1
 9  -2
10  -1
11  -2
12  -2
13  -6
14  -6
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
        self.do('output %s ' % self.out_dir)
        # Check that all subprocesses have separate directories
        directories = ['P0_wp_LxN','P0_wp_QQx']
        for d in directories:
            self.assertTrue(os.path.isdir(os.path.join(self.out_dir,
                                                       'SubProcesses',
                                                       d)))
        self.do('set group_subprocesses True')
        self.do('generate w+ > l+ vl')
        self.do('add process w+ > j j')
        self.do('output %s -f' % self.out_dir)
        # Check that all subprocesses are combined
        directories = ['P0_wp_lvl','P0_wp_qq']
        for d in directories:
            self.assertTrue(os.path.isdir(os.path.join(self.out_dir,
                                                       'SubProcesses',
                                                       d)))
            
        self.do('generate w+ > l+ vl')
        self.do('generate e+ e- > j j')
        self.do('output %s -f' % self.out_dir)
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
        self.do('output %s ' % self.out_dir)
        
        # Check that leshouche.inc exists
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_uu_sixg',
                                                    'leshouche.inc')))        
        # Test sextet decay
        self.do('generate six > u u g')
        self.do('output %s -f' % self.out_dir)

        # Check that leshouche.inc exists
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_six_uug',
                                                    'leshouche.inc')))        

        # Test sextet production
        self.do('generate u g > six u~')
        self.do('output %s -f' % self.out_dir)
        
        # Check that leshouche.inc exists
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_ug_sixux',
                                                    'leshouche.inc')))
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
        self.do('output standalone %s -f' % self.out_dir)
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
        
    def test_pythia8_output(self):
        """Test Pythia 8 output"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)
        # Create out_dir and out_dir/include
        os.makedirs(os.path.join(self.out_dir,'include'))
        # Touch the file Pythia.h, which is needed to verify that this is a Pythia dir
        py_h_file = open(os.path.join(self.out_dir,'include','Pythia.h'), 'w')
        py_h_file.close()

        self.do('set apply_flavor_grouping False')
        self.do('import model sm')
        self.do('define p g u d u~ d~')
        self.do('define j g u d u~ d~')
        self.do('generate p p > w+ j @2')
        self.do('output pythia8 %s' % self.out_dir)
        # Check that the needed files are generated
        files = ['Processes_sm/Sigma_sm_gq_wpq.h', 'Processes_sm/Sigma_sm_gq_wpq.cc',
                 'Processes_sm/Sigma_sm_qq_wpg.h', 'Processes_sm/Sigma_sm_qq_wpg.cc',
                 'Processes_sm/HelAmps_sm.h', 'Processes_sm/HelAmps_sm.cc',
                 'Processes_sm/Parameters_sm.h',
                 'Processes_sm/Parameters_sm.cc', 'Processes_sm/Makefile',
                 'examples/main_sm_1.cc', 'examples/Makefile_sm_1']
        for f in files:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir, f)), 
                            '%s file is not in directory' % f)
        self.do('generate u u~ > a a a a')
        self.assertRaises(MadGraph5Error,
                          self.do,
                          'output pythia8 %s' % self.out_dir)
        self.do('generate u u~ > w+ w-, w+ > e+ ve, w- > e- ve~ @1')
        self.assertRaises(MadGraph5Error,
                          self.do,
                          'output pythia8 %s' % self.out_dir)

    def test_standalone_cpp_output(self):
        """Test the C++ standalone output"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('generate e+ e- > e+ e- @2')
        self.do('output standalone_cpp %s' % self.out_dir)

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
        # Check that check_sa.cpp compiles
        subprocess.call(['make', 'check'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P2_Sigma_sm_epem_epem'))


        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_Sigma_sm_epem_epem',
                                                    'check')))

        # Check that the output of check is correct 
        logfile = os.path.join(self.out_dir, 'SubProcesses',
                               'P2_Sigma_sm_epem_epem', 'check.log')

        subprocess.call('./check', 
                        stdout=open(logfile, 'w'), stderr=devnull,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P2_Sigma_sm_epem_epem'), shell=True)

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
        
