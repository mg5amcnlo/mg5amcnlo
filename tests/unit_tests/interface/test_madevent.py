##############################################################################
#
# Copyright (c) 2010 The MadGraph5_aMC@NLO Development team and Contributors
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

        # PLOT_CARD
        self.assertEqual(detect(pjoin(card_dir, 'plot_card.dat')),
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
        
        
    def test_help_category(self):
        """Check that no help category are introduced by mistake.
           If this test failes, this is due to a un-expected ':' in a command of
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
            f.write('#!/bin/bash\n%s\ncp "$3" "$2"\n' % fail)
        os.chmod(exe, os.stat(exe).st_mode | stat.S_IEXEC)

        # fake hadd (ROOTSYS/bin/hadd): concatenate the input ROOTs into output.
        rootsys = pjoin(self.tmp, 'root')
        os.makedirs(pjoin(rootsys, 'bin'))
        hadd = pjoin(rootsys, 'bin', 'hadd')
        with open(hadd, 'w') as f:
            f.write('#!/bin/bash\n'
                    'out=""; skip=0; ins=()\n'
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

    def test_run_delphes_on_splits_partial_failure(self):
        # split_1's Delphes produces no ROOT (but exits 0): the fused path must
        # NOT hadd a partial set (that would silently drop events) and instead
        # fall back to the standard single Delphes pass.
        stub, split_dirs, pdir = self._setup_run(n_splits=3, fail_split='split_1')
        ok = stub.run_delphes_on_splits(split_dirs, pdir, 'tag_1')
        self.assertFalse(ok)
        final = pjoin(stub.me_dir, 'Events', 'run_01', 'tag_1_delphes_events.root')
        self.assertFalse(os.path.isfile(final))
