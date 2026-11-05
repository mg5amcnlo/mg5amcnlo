################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
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
"""Acceptance tests for LO MadEvent runs with a bound atomic electron in the
initial state (model sm_atom, impulse approximation in the atom rest frame).

The references come from the EAtom_PS benchmarks: the free-electron limit
(6 x the fixed-target e+ e- > mu+ mu- cross section) and a one-dimensional
quadrature of the standalone matrix element for a 2 -> 1 resonance."""
from __future__ import division
from __future__ import absolute_import

import gzip
import logging
import os
import shutil
import tempfile
import unittest

import madgraph
import madgraph.interface.master_interface as MGCmd
import madgraph.interface.madevent_interface as MECmd
import madgraph.various.banner as banner_mod
import madgraph.various.lhe_parser as lhe_parser
import models.check_param_card as param_card_mod
from madgraph import MG5DIR

pjoin = os.path.join
logger = logging.getLogger('test_cmd')

ME = 5.11e-4  # electron mass of the sm_atom-lepton_masses param_card


class TestEatomMadEvent(unittest.TestCase):
    """madevent output and launch for e+ e(atom) processes"""

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='acc_test_eatom')
        self.run_dir = pjoin(self.path, 'MGPROC')

    def tearDown(self):
        shutil.rmtree(self.path)

    def generate(self, process):
        interface = MGCmd.MasterCmd()
        interface.no_notification()
        interface.exec_cmd('set automatic_html_opening False --no_save')
        interface.run_cmd('import model sm_atom-lepton_masses')
        interface.run_cmd('generate %s' % process)
        interface.onecmd('output madevent %s -f' % self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False')

    def set_run(self, **params):
        path = pjoin(self.run_dir, 'Cards', 'run_card.dat')
        card = banner_mod.RunCard(path)
        for key, value in params.items():
            card.set(key, value, user=True)
        card.write(path)

    def test_eatom_output_defaults(self):
        """run_card and process information of an atom process"""
        self.generate('e+ eC12 > mu+ mu-')
        card = banner_mod.RunCardLO(pjoin(self.run_dir, 'Cards', 'run_card.dat'))
        self.assertEqual(card['lpp1'], 0)
        self.assertEqual(card['lpp2'], 0)
        self.assertEqual(card['eatom_iatom'], 2)
        self.assertEqual(card['eatom_a'], 12)
        self.assertEqual(card['eatom_z'], 6)
        self.assertEqual(card['ptl'], 0)
        self.assertEqual(card['eatom_ps_mode'], 1)
        init = card.get_banner_init_information()
        self.assertEqual(init['idbmup2'], 1000060120)
        pdir = [d for d in os.listdir(pjoin(self.run_dir, 'SubProcesses')) if d.startswith('P1_')][0]
        text = open(pjoin(self.run_dir, 'SubProcesses', pdir, 'eatom.inc')).read()
        self.assertIn('IATOM=2', text)
        self.assertIn('EATOM_PDG=1000601211', text)
        leshouche = open(pjoin(self.run_dir, 'SubProcesses', pdir, 'leshouche.inc')).read()
        self.assertIn('/-11,11,-13,13/', leshouche)

    def test_eatom_free_limit_and_events(self):
        """e+ eC12 > mu+ mu- at 100 GeV: cross section and event content"""
        self.generate('e+ eC12 > mu+ mu-')
        self.set_run(nevents=1000, ebeam1=100.0, iseed=33)
        self.cmd_line.run_cmd('generate_events -f')
        val = self.cmd_line.results.current['cross']
        err = self.cmd_line.results.current['error']
        # EAtom_PS benchmark (10k events): 4.98850e6 +- 1.0e3 pb; 6 x free
        # fixed target = 4.98684e6 pb
        target = 4.9885e6
        self.assertLess(abs(val - target) / (err + 1.0e3), 4.,
                        'cross section %s +- %s, expected %s' % (val, err, target))
        lhe = pjoin(self.run_dir, 'Events', 'run_01', 'unweighted_events.lhe.gz')
        with gzip.open(lhe, 'rt') as f:
            text = f.read()
        init = text.split('<init>')[1].split('\n')[1].split()
        self.assertEqual(init[:2], ['-11', '1000060120'])
        lhefile = lhe_parser.EventFile(lhe)
        nev = 0
        for event in lhefile:
            nev += 1
            event.check()
            bound = [p for p in event if p.status == -1 and p.pdg == 11]
            self.assertEqual(len(bound), 1)
            self.assertAlmostEqual(bound[0].E, ME, places=12)
            self.assertEqual(bound[0].helicity, 9)
        self.assertEqual(nev, 1000)

    def test_eatom_resonance_2to1(self):
        """e+ eC12 > z with M_Z = 10 MeV at E1 = E0: 1-D quadrature reference"""
        self.generate('e+ eC12 > z')
        path = pjoin(self.run_dir, 'Cards', 'param_card.dat')
        card = param_card_mod.ParamCard(path)
        card['mass'].get(23).value = 0.01
        card['sminputs'].get(2).value = 736.1099
        card['decay'].get(23).value = 0.0
        card.write(path)
        e0 = (0.01**2 - 2 * ME**2) / (2 * ME)
        self.set_run(nevents=500, ebeam1=e0, iseed=33)
        self.cmd_line.run_cmd('generate_events -f')
        val = self.cmd_line.results.current['cross']
        err = self.cmd_line.results.current['error']
        target = 1.468803e14   # quadrature (EAtom_PS/scripts/ref_2to1.py)
        self.assertLess(abs(val - target) / err, 4.,
                        'cross section %s +- %s, expected %s' % (val, err, target))
