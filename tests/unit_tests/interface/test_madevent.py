##############################################################################
#
# Copyright (c) 2010 The MadGraph7 Development team and Contributors
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
from cmd import Cmd
""" Basic test of the command interface """

import unittest
import madgraph
import madgraph.interface.master_interface as mgcmd
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.interface.madevent_interface as mecmd
import madgraph.various.cluster as cluster
import os
import shutil
import stat
import tempfile


root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.dirname(root_path)
# root_path is ./tests
pjoin = os.path.join

class TestMadEventCmd(unittest.TestCase):
    """ check if the ValidCmd works correctly """
    
    def test_card_type_recognition(self):
        """Check that the different card are recognize correctly"""

        #detect = mecmd.MadEventCmd.detect_card_type
        def detect(p):
            #print p
            return mecmd.MadEventCmd.detect_card_type(p)
        # run_card
        card_dir= pjoin(root_path,'..','Template/LO', 'Cards')
        self.assertEqual(detect(pjoin(card_dir, 'run_card.dat')),
                         'run_card.dat')
        self.assertEqual(detect(pjoin(root_path, 'input_files','run_card_matching.dat')),
                         'run_card.dat')

        # mg7 run_card (TOML format): concrete card + template
        self.assertEqual(detect(pjoin(root_path, 'input_files','mg7_run_card.toml')),
                         'run_card.toml')
        self.assertEqual(detect(pjoin(root_path,'..','madgraph','iolibs',
                                      'template_files','mg7','run_card.toml')),
                         'run_card.toml')

        # PYTHIA_CARD
        self.assertEqual(detect(pjoin(card_dir, 'pythia_card_default.dat')),
                         'pythia_card.dat')

        # PYTHIA8_CARD
        self.assertEqual(detect(pjoin(card_dir, 'pythia8_card_default.dat')),
                                                             'pythia8_card.dat')

        # PARAM_CARD
        self.assertEqual(detect(pjoin(card_dir, 'param_card.dat')),
                         'param_card.dat')
        self.assertEqual(detect(pjoin(root_path, 'input_files','sps1a_param_card.dat')),
                         'param_card.dat')
        self.assertEqual(detect(pjoin(root_path, 'input_files','restrict_sm.dat')),
                         'param_card.dat')

        card_dir= pjoin(root_path,'..','Template/Common', 'Cards')

        # PLOT_CARD (MadAnalysis4 is gone, but old banners still carry the card)
        self.assertEqual(detect(pjoin(root_path, 'input_files', 'plot_card.dat')),
                         'plot_card.dat')

        # Delphes
        self.assertEqual(detect(pjoin(card_dir, 'delphes_card_CMS.dat')),
                         'delphes_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'delphes_card_default.dat')),
                         'delphes_card.dat')
        # PGS
        self.assertEqual(detect(pjoin(card_dir, 'pgs_card_ATLAS.dat')),
                         'pgs_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'pgs_card_CMS.dat')),
                         'pgs_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'pgs_card_LHC.dat')),
                         'pgs_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'pgs_card_TEV.dat')),
                         'pgs_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'pgs_card_default.dat')),
                         'pgs_card.dat')
        
        # Reweight
        card_dir= pjoin(root_path,'..','Template','Common', 'Cards')
        self.assertEqual(detect(pjoin(card_dir, 'reweight_card_default.dat')),
                         'reweight_card.dat')
        
        #MadSpin card are tested in their specific routine. (in fact acceptance test)
        card_dir= pjoin(root_path,'..','Template', 'Common', 'Cards')
        self.assertEqual(detect(pjoin(card_dir, 'madspin_card_default.dat')),
                         'madspin_card.dat') 

        card_dir= pjoin(root_path,'..','Template', 'NLO', 'Cards')
        # NLO Card
        self.assertEqual(detect(pjoin(card_dir, 'run_card.dat')),
                         'run_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'shower_card.dat')),
                         'shower_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'FO_analyse_card.dat')),
                         'FO_analyse_card.dat')
        
        #MA5 card        
        card_dir= pjoin(root_path,'input_files')
        self.assertEqual(detect(pjoin(card_dir, 'madanalysis5_hadron_card.dat')),
                         'madanalysis5_hadron_card.dat')
        self.assertEqual(detect(pjoin(card_dir, 'madanalysis5_parton_card.dat')),
                         'madanalysis5_parton_card.dat')
        
        # Rivet card
        card_dir= pjoin(root_path,'..','Template', 'LO', 'Cards')
        self.assertEqual(detect(pjoin(card_dir, 'rivet_card_default.dat')),
                         'rivet_card.dat')        

        # Onia card
        self.assertEqual(detect(pjoin(root_path, 'input_files',
                                      'onia_card_quarkonium.dat')),
                         'onia_card.dat')
        self.assertEqual(detect(pjoin(root_path, 'input_files',
                                      'onia_card_leptonium.dat')),
                         'onia_card.dat')
        
        
    def test_help_category(self):
        """Check that no help category are introduced by mistake.
           If this test fails, this is due to a un-expected ':' in a command of
           the cmd interface.
        """
        cmd = mecmd.MadEventCmdShell
        category = set()
        valid_command = [c for c in dir(cmd) if c.startswith('do_')]
        
        for command in valid_command:
            obj = getattr(cmd,command)
            if obj.__doc__ and ':' in obj.__doc__:
                category.add(obj.__doc__.split(':',1)[0])
                
        target = set(['Main Commands','Advanced commands', 'Require MG5 directory', 'Not in help'])
        self.assertEqual(target, category)


class TestDelphesFusion(unittest.TestCase):
    """Unit tests for the fused parallel-Delphes path (is_delphes_fusion_active
    and run_delphes_on_splits). These use fake Delphes/hadd executables so they
    run everywhere, without a real ROOT/Delphes install."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='delphes_fusion_')
        self._orig_rootsys = os.environ.get('ROOTSYS')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        if self._orig_rootsys is None:
            os.environ.pop('ROOTSYS', None)
        else:
            os.environ['ROOTSYS'] = self._orig_rootsys

    def _make_stub(self, **opts):
        """A MadEventCmd instance with only the attributes the tested methods
        touch (bypassing the heavy __init__)."""
        stub = mecmd.MadEventCmd.__new__(mecmd.MadEventCmd)
        stub.me_dir = tempfile.mkdtemp(dir=self.tmp)
        stub.run_name = 'run_01'
        options = {'delphes_path': None, 'run_mode': 2, 'nb_core': 2,
                   'nb_core_pythia8': None, 'nb_core_delphes': None,
                   'cluster_temp_path': None}
        options.update(opts)
        stub.options = options
        stub.run_card = {'event_norm': 'average'}
        class _Banner(object):
            def add(self, *a, **k): pass
            def write(self, *a, **k): pass
        stub.banner = _Banner()
        stub.update_status = lambda *a, **k: None
        for sub in ['Cards', 'Source', pjoin('Events', 'run_01')]:
            os.makedirs(pjoin(stub.me_dir, sub))
        return stub

    # ---- is_delphes_fusion_active ---------------------------------------
    def test_is_delphes_fusion_active(self):
        def make(card=True, **opts):
            stub = self._make_stub(**opts)
            if card:
                open(pjoin(stub.me_dir, 'Cards', 'delphes_card.dat'), 'w').close()
            return stub

        # nb_core_delphes unset -> single core (off, the default)
        self.assertFalse(make(delphes_path='/d').is_delphes_fusion_active())
        # nb_core_delphes set -> parallel (on)
        self.assertTrue(make(delphes_path='/d', nb_core_delphes=2).is_delphes_fusion_active())
        # set, but various disqualifiers -> off
        self.assertFalse(make(delphes_path=None, nb_core_delphes=2).is_delphes_fusion_active())
        self.assertFalse(make(card=False, delphes_path='/d', nb_core_delphes=2).is_delphes_fusion_active())
        self.assertFalse(make(delphes_path='/d', nb_core_delphes=2, run_mode=0).is_delphes_fusion_active())
        stub = make(delphes_path='/d', nb_core_delphes=2)
        stub.run_card['event_norm'] = 'sum'
        self.assertFalse(stub.is_delphes_fusion_active())

    # ---- run_delphes_on_splits ------------------------------------------
    def _setup_run(self, n_splits=3, fail_split=None):
        """Build fake DelphesHepMC2 + hadd and n_splits split dirs holding a
        distinct events.hepmc. Returns (stub, split_dirs, parallelization_dir)."""
        # fake Delphes: args = card out in ; copies in->out, but produces no
        # output (yet exits 0) for the split whose name matches fail_split.
        ddir = pjoin(self.tmp, 'delphes')
        os.makedirs(ddir)
        exe = pjoin(ddir, 'DelphesHepMC2')
        fail = ('[[ "$3" == *%s* ]] && exit 0' % fail_split) if fail_split else 'false'
        with open(exe, 'w') as f:
            f.write('#!/bin/bash\n%s\necho "DELPHES_OUTPUT_$3"\ncp "$3" "$2"\n' % fail)
        os.chmod(exe, os.stat(exe).st_mode | stat.S_IEXEC)

        # fake hadd (ROOTSYS/bin/hadd): concatenate the input ROOTs into output.
        rootsys = pjoin(self.tmp, 'root')
        os.makedirs(pjoin(rootsys, 'bin'))
        hadd = pjoin(rootsys, 'bin', 'hadd')
        with open(hadd, 'w') as f:
            f.write('#!/bin/bash\n'
                    'out=""; skip=0; ins=()\n'
                    'echo "HADD_STDOUT"\n'
                    'echo "HADD_STDERR" >&2\n'
                    'for a in "$@"; do\n'
                    '  if [ "$skip" = 1 ]; then skip=0; continue; fi\n'
                    '  case "$a" in -f) ;; -j) skip=1;;\n'
                    '    *) if [ -z "$out" ]; then out="$a"; else ins+=("$a"); fi;; esac\n'
                    'done\n'
                    'cat "${ins[@]}" > "$out"\n')
        os.chmod(hadd, os.stat(hadd).st_mode | stat.S_IEXEC)
        os.environ['ROOTSYS'] = rootsys

        stub = self._make_stub(delphes_path=ddir, nb_core_delphes=2)
        open(pjoin(stub.me_dir, 'Cards', 'delphes_card.dat'), 'w').close()
        stub.cluster = cluster.MultiCore(nb_core=2, cluster_temp_path=None)

        pdir = pjoin(stub.me_dir, 'Events', 'run_01', 'PY8_parallelization')
        os.makedirs(pdir)
        split_dirs = []
        for i in range(n_splits):
            d = pjoin(pdir, 'split_%d' % i)
            os.makedirs(d)
            with open(pjoin(d, 'events.hepmc'), 'w') as f:
                f.write('CONTENT_%d\n' % i)
            split_dirs.append(d)
        return stub, split_dirs, pdir

    def test_run_delphes_on_splits_all_ok(self):
        stub, split_dirs, pdir = self._setup_run(n_splits=3)
        ok = stub.run_delphes_on_splits(split_dirs, pdir, 'tag_1')
        self.assertTrue(ok)
        final = pjoin(stub.me_dir, 'Events', 'run_01', 'tag_1_delphes_events.root')
        self.assertTrue(os.path.isfile(final))
        # hadd concatenated every split's Delphes output, in order.
        self.assertEqual(open(final).read(),
                         'CONTENT_0\nCONTENT_1\nCONTENT_2\n')
        log = pjoin(stub.me_dir, 'Events', 'run_01', 'tag_1_delphes.log')
        log_content = open(log).read()
        split_sections = []
        for i in range(3):
            section = ('=' * 35 + '\n'
                       ' -> Delphes log file for run %d <-\n' % i
                       + '=' * 35 + '\n'
                       'DELPHES_OUTPUT_' + pjoin(split_dirs[i], 'events.hepmc') + '\n\n')
            split_sections.append(section)
        self.assertEqual(log_content[:sum(map(len, split_sections))],
                         ''.join(split_sections))
        self.assertEqual(log_content.count('DELPHES_OUTPUT_'), 3)
        self.assertGreater(log_content.index('HADD_STDOUT'),
                   log_content.index('DELPHES_OUTPUT_' + pjoin(split_dirs[-1], 'events.hepmc')))
        self.assertIn('HADD_STDERR', log_content)

    def test_run_delphes_on_splits_partial_failure(self):
        # split_1's Delphes produces no ROOT (but exits 0): the fused path must
        # NOT hadd a partial set (that would silently drop events) and instead
        # fall back to the standard single Delphes pass.
        stub, split_dirs, pdir = self._setup_run(n_splits=3, fail_split='split_1')
        ok = stub.run_delphes_on_splits(split_dirs, pdir, 'tag_1')
        self.assertFalse(ok)
        final = pjoin(stub.me_dir, 'Events', 'run_01', 'tag_1_delphes_events.root')
        self.assertFalse(os.path.isfile(final))


class TestLhapdfInfoPatch(unittest.TestCase):
    """check that missing AlphaS_* metadata is added to the .info file of a
    PDF set everywhere the run time can read it (global dir and local copy)"""

    INFO_MISSING = """SetDesc: test set
Format: lhagrid1
FlavorScheme: variable
NumFlavors: 5
AlphaS_Type: ipol
"""

    def setUp(self):
        import madgraph.interface.common_run_interface as common_run
        self.common_run = common_run
        self.tmpdir = tempfile.mkdtemp(prefix='mg5_lhapdf_test')
        # a fake global lhapdf data directory with one set
        self.pdfsets_dir = pjoin(self.tmpdir, 'share', 'LHAPDF')
        os.makedirs(pjoin(self.pdfsets_dir, 'MYSET'))
        with open(pjoin(self.pdfsets_dir, 'MYSET', 'MYSET.info'), 'w') as f:
            f.write(self.INFO_MISSING)
        # a fake process directory
        self.me_dir = pjoin(self.tmpdir, 'PROC')
        os.makedirs(pjoin(self.me_dir, 'lib', 'PDFsets'))
        self.saved_datapath = os.environ.pop('LHAPDF_DATA_PATH', None)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        if self.saved_datapath is not None:
            os.environ['LHAPDF_DATA_PATH'] = self.saved_datapath

    def get_fake_cmd(self, **options):
        common_run = self.common_run
        class FakeRunCmd(object):
            patch_lhapdf_info_file = staticmethod(
                          common_run.CommonRunCmd.patch_lhapdf_info_file)
            use_shared_pdfsets_dir = staticmethod(
                          common_run.CommonRunCmd.use_shared_pdfsets_dir)
            get_shared_pdfsets_dirs = common_run.CommonRunCmd.get_shared_pdfsets_dirs
            copy_lhapdf_set = common_run.CommonRunCmd.copy_lhapdf_set
        cmd = FakeRunCmd()
        cmd.me_dir = self.me_dir
        # cvmfs_lhapdf_path is None unless a test asks for it: the default
        # points at a real mount which may exist on the machine running this
        cmd.options = {'cluster_local_path': None, 'run_mode': 2,
                       'cvmfs_lhapdf_path': None}
        cmd.options.update(options)
        cmd.lhapdf_pdfsets = {}
        return cmd

    def test_patch_lhapdf_info_file(self):
        """missing AlphaS_* keys are mirrored from their base counterpart,
        and the patching is idempotent"""

        setdir = pjoin(self.pdfsets_dir, 'MYSET')
        self.common_run.CommonRunCmd.patch_lhapdf_info_file(setdir)
        content = open(pjoin(setdir, 'MYSET.info')).read()
        self.assertIn('AlphaS_FlavorScheme: variable', content)
        self.assertIn('AlphaS_NumFlavors: 5', content)
        # calling it again should not duplicate the keys
        self.common_run.CommonRunCmd.patch_lhapdf_info_file(setdir)
        content = open(pjoin(setdir, 'MYSET.info')).read()
        self.assertEqual(content.count('AlphaS_FlavorScheme'), 1)
        self.assertEqual(content.count('AlphaS_NumFlavors'), 1)
        # a non existing directory should simply be ignored
        self.common_run.CommonRunCmd.patch_lhapdf_info_file(
                                             pjoin(self.tmpdir, 'DOESNOTEXIST'))

    def test_copy_lhapdf_set_patches_global_and_local(self):
        """with require_local, both the global set and the local copy end up
        with the required metadata"""

        cmd = self.get_fake_cmd()
        cmd.copy_lhapdf_set(['MYSET'], self.pdfsets_dir)
        local_info = pjoin(self.me_dir, 'lib', 'PDFsets', 'MYSET', 'MYSET.info')
        global_info = pjoin(self.pdfsets_dir, 'MYSET', 'MYSET.info')
        self.assertTrue(os.path.isfile(local_info))
        self.assertIn('AlphaS_FlavorScheme: variable', open(local_info).read())
        self.assertIn('AlphaS_FlavorScheme: variable', open(global_info).read())

    def test_copy_lhapdf_set_patches_global_without_local(self):
        """without require_local the set stays global but is still patched"""

        cmd = self.get_fake_cmd()
        cmd.copy_lhapdf_set(['MYSET'], self.pdfsets_dir, require_local=False)
        local_set = pjoin(self.me_dir, 'lib', 'PDFsets', 'MYSET')
        global_info = pjoin(self.pdfsets_dir, 'MYSET', 'MYSET.info')
        self.assertFalse(os.path.exists(local_set))
        self.assertIn('AlphaS_FlavorScheme: variable', open(global_info).read())
        self.assertIn('AlphaS_NumFlavors: 5', open(global_info).read())

class TestSharedPdfsetsDir(unittest.TestCase):
    """a PDF set readable from every node (CVMFS, or the user-declared
    cluster_local_path) is used in place, so it is never copied into
    lib/PDFsets -- which is exactly what would be shipped to the node."""

    def setUp(self):
        import madgraph.interface.common_run_interface as common_run
        self.common_run = common_run
        self.tmpdir = tempfile.mkdtemp(prefix='mg5_cvmfs_test')
        # stand-in for /cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current
        self.cvmfs = pjoin(self.tmpdir, 'cvmfs')
        os.makedirs(pjoin(self.cvmfs, 'MYSET'))
        # the local LHAPDF data directory, holding a different set
        self.pdfsets_dir = pjoin(self.tmpdir, 'share', 'LHAPDF')
        os.makedirs(pjoin(self.pdfsets_dir, 'OTHERSET'))
        self.me_dir = pjoin(self.tmpdir, 'PROC')
        os.makedirs(pjoin(self.me_dir, 'lib', 'PDFsets'))
        self.saved = {key: os.environ.get(key) for key in
                      ('LHAPATH', 'CLUSTER_LHAPATH', 'LHAPDF_DATA_PATH')}
        for key in self.saved:
            os.environ.pop(key, None)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def get_fake_cmd(self, **options):
        common_run = self.common_run
        class FakeRunCmd(object):
            patch_lhapdf_info_file = staticmethod(
                          common_run.CommonRunCmd.patch_lhapdf_info_file)
            use_shared_pdfsets_dir = staticmethod(
                          common_run.CommonRunCmd.use_shared_pdfsets_dir)
            get_shared_pdfsets_dirs = common_run.CommonRunCmd.get_shared_pdfsets_dirs
            copy_lhapdf_set = common_run.CommonRunCmd.copy_lhapdf_set
            get_pdf_input_filename = common_run.CommonRunCmd.get_pdf_input_filename
        cmd = FakeRunCmd()
        cmd.me_dir = self.me_dir
        cmd.options = {'cluster_local_path': None, 'run_mode': 2,
                       'cvmfs_lhapdf_path': self.cvmfs}
        cmd.options.update(options)
        cmd.lhapdf_pdfsets = {}
        return cmd

    def test_set_on_cvmfs_is_not_copied_locally(self):
        cmd = self.get_fake_cmd()
        cmd.copy_lhapdf_set(['MYSET'], self.pdfsets_dir)
        self.assertFalse(os.path.exists(
                              pjoin(self.me_dir, 'lib', 'PDFsets', 'MYSET')))
        # ... and LHAPDF is told where to read it, keeping the local dir too
        self.assertIn(self.cvmfs, os.environ['LHAPATH'].split(':'))
        self.assertIn(self.pdfsets_dir, os.environ['LHAPATH'].split(':'))
        self.assertEqual(os.environ['CLUSTER_LHAPATH'], os.environ['LHAPATH'])

    def test_nothing_is_shipped_to_the_node(self):
        """an empty lib/PDFsets means the node reads the PDF on its own"""

        cmd = self.get_fake_cmd()
        cmd.run_card = {'pdlabel': 'lhapdf'}
        # no Source/PDF/pdf_list.txt: the lhapdf branch is the one reached
        os.makedirs(pjoin(self.me_dir, 'Source', 'PDF'))
        open(pjoin(self.me_dir, 'Source', 'PDF', 'pdf_list.txt'), 'w').close()
        cmd.copy_lhapdf_set(['MYSET'], self.pdfsets_dir)
        self.assertEqual(cmd.get_pdf_input_filename(), '')
        # a set which is NOT on the mirror is copied, and then shipped
        cmd = self.get_fake_cmd()
        cmd.run_card = {'pdlabel': 'lhapdf'}
        cmd.copy_lhapdf_set(['OTHERSET'], self.pdfsets_dir)
        self.assertTrue(os.path.isdir(
                            pjoin(self.me_dir, 'lib', 'PDFsets', 'OTHERSET')))
        self.assertEqual(cmd.get_pdf_input_filename(),
                         pjoin(self.me_dir, 'lib', 'PDFsets'))

    def test_cvmfs_disabled(self):
        """with the mirror switched off, the ordinary local copy is made"""

        cmd = self.get_fake_cmd(cvmfs_lhapdf_path=None)
        self.assertEqual(cmd.get_shared_pdfsets_dirs(), [])
        # OTHERSET lives in the local data dir, not on the mirror
        cmd.copy_lhapdf_set(['OTHERSET'], self.pdfsets_dir)
        self.assertTrue(os.path.isdir(
                            pjoin(self.me_dir, 'lib', 'PDFsets', 'OTHERSET')))
        self.assertNotIn('CLUSTER_LHAPATH', os.environ)

    def test_cluster_local_path_only_for_a_cluster_run(self):
        cmd = self.get_fake_cmd(cvmfs_lhapdf_path=None,
                                cluster_local_path=self.cvmfs)
        self.assertEqual(cmd.get_shared_pdfsets_dirs(), [])
        cmd.options['run_mode'] = 1
        self.assertEqual(cmd.get_shared_pdfsets_dirs()[0], self.cvmfs)


class TestPythia8MergingPlugin(unittest.TestCase):
    """Since Pythia8 8.318, main164 loads its merging hook as a plugin, with
    a readString that Pythia8 ignores after reading a card with Main:subrun
    lines (all of ours): the card of a CKKW-L shower requests the plugin."""

    def setUp(self):
        import madgraph.various.banner as banner_mod
        self.banner_mod = banner_mod
        self.tmpdir = tempfile.mkdtemp(prefix='mg5_py8_merging')
        self.template = open(pjoin(madgraph.MG5DIR, 'Template', 'LO', 'Cards',
                                   'pythia8_card_default.dat')).read()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def make_pythia8(self, version):
        """a fake Pythia8 installation, with main164.cc as in this version"""
        py8 = pjoin(self.tmpdir, 'pythia%s' % version)
        os.makedirs(pjoin(py8, 'share', 'Pythia8', 'examples'))
        os.makedirs(pjoin(py8, 'lib'))
        if version == '8318':
            source = ('    pythia.readString("Init:plugins += {libpythia8amcatnloHooks.so"\n'
                      '      "::aMCatNLOHooks}");\n')
            open(pjoin(py8, 'lib', 'libpythia8amcatnloHooks.so'), 'w').close()
        else:
            source = '    pythia.setUserHooksPtr(mergingHooks);\n'
        with open(pjoin(py8, 'share', 'Pythia8', 'examples', 'main164.cc'), 'w') as fsock:
            fsock.write(source)
        return py8

    def setup_card(self, pythia8_path, run_type, **run_card):
        """the card of the shower, as do_pythia8 sets it up"""
        class FakeCmd(object):
            pass
        cmd = FakeCmd()
        cmd.run_tag, cmd.run_name, cmd.me_dir = 'tag_1', 'run_01', self.tmpdir
        cmd.to_store = []
        cmd.options = {'run_mode': 2, 'pythia8_path': pythia8_path}
        cmd.proc_characteristic = {'max_n_matched_jets': 2}
        cmd.run_card = self.banner_mod.RunCardLO()
        for name, value in run_card.items():
            cmd.run_card[name] = value
        card = self.banner_mod.PY8Card(self.template)
        card.read(self.template + 'Merging:Process = guess\n', setter='user')
        mecmd.MadEventCmd.setup_Pythia8RunAndCard(cmd, card, run_type)
        return card

    def test_ckkwl_requests_the_merging_plugin(self):
        ckkwl = {'ickkw': 0, 'xqcut': 0.0, 'ktdurham': 30.0}
        py8 = self.make_pythia8('8318')
        card = self.setup_card(py8, 'CKKW', **ckkwl)
        self.assertEqual(card.init_plugins,
                         ['%s::aMCatNLOHooks::add' %
                          os.path.abspath(pjoin(py8, 'lib', 'libpythia8amcatnloHooks.so'))])
        # up to Pythia8 8.317, main164 sets the hook itself
        card = self.setup_card(self.make_pythia8('8317'), 'CKKW', **ckkwl)
        self.assertEqual(card.init_plugins, [])
        # main164 does not merge an MLM shower
        card = self.setup_card(py8, 'MLM', ickkw=1, xqcut=30.0)
        self.assertEqual(card.init_plugins, [])
