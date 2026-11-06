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
"""Acceptance tests of photon-photon processes in ultraperipheral collisions
with gamma-UPC (arXiv:2207.03012) and forward neutron tagging, at LO
(MadEvent) and NLO (fixed order).

The references are 2000-event (LO) and 1% (NLO) runs with gamma-UPC 1.8.
Each test compares with 3 combined standard deviations. EDFF is used
throughout: building the ChFF luminosity grid for ion-ion collisions takes
tens of minutes; the ChFF labels are covered by the unit tests."""

from __future__ import absolute_import

import math
import os
import shutil
import tempfile
import unittest

import madgraph.interface.master_interface as MGCmd
import madgraph.interface.madevent_interface as MECmd
import madgraph.interface.amcatnlo_run_interface as NLOCmd
import madgraph.various.banner as banner
from madgraph import MG5DIR, InvalidCmd

pjoin = os.path.join

PB208, O16, PROTON = (82, 126), (8, 8), (1, 0)
# beam energies (GeV, total per beam): Pb at sqrt(s_NN)=5.02 TeV, pPb at
# 8.16 TeV, OO at 5.36 TeV
E_PBPB = (522080, 522080)
E_PPB = (6500, 532480)
E_OO = (42880, 42880)

# a a > mu+ mu-, pT(mu) > 5 GeV, |eta(mu)| < 2.5 (pb)
REF_LO = {
    ('PbPb', 'edff'): (2.340e7, 1.204e5),
    ('PbPb', 'edf0n0n'): (1.561e7, 8.126e4),
    ('PbPb', 'edf0nxn'): (2.955e6, 1.493e4),
    ('PbPb', 'edfxn0n'): (2.955e6, 1.493e4),
    ('PbPb', 'edfxnxn'): (1.888e6, 8674.),
    ('PbPb', 'edf1nin'): (1.338e6, 6762.),
    ('PbPb', 'edpxnxn'): (2.055e6, 9464.),
    ('PbPb', 'edmxnxn'): (1.729e6, 8017.),
    ('pPb', 'edfin2n'): (0.09758, 4.569e-4),   # Pb excited by the proton field only
    ('OO', 'edff'): (6187., 26.74),
    ('OO', 'edf1n1n'): (6.976e-5, 3.92e-7),  # mutual excitation of O16 is rare
}
# !a! !a! > b b~ [QCD] at fixed-order NLO, no cuts (pb)
REF_NLO = {
    ('PbPb', 'edff'): (2.093e6, 1.4e4),
    ('PbPb', 'edf0nxn'): (2.386e5, 1.3e3),
}


def compatible(test, value, error, ref, ref_error, nsigma=3.):
    """value +- error agrees with ref +- ref_error"""
    pull = abs(value - ref) / math.sqrt(error**2 + ref_error**2)
    test.assertLess(pull, nsigma, '%s +- %s vs reference %s +- %s (%.1f sigma)'
                    % (value, error, ref, ref_error, pull))


def set_upc_beams(card, pdlabel, beam1, beam2, ebeam):
    for i, (Z, N), E in ((1, beam1, ebeam[0]), (2, beam2, ebeam[1])):
        card.set('lpp%i' % i, 2, user=True)
        card.set('ebeam%i' % i, E, user=True)
        card.set('nb_proton%i' % i, Z, user=True)
        card.set('nb_neutron%i' % i, N, user=True)
    card.set('pdlabel', pdlabel, user=True)


class TestGammaUPCMadEvent(unittest.TestCase):
    """a a > mu+ mu- in UPC at LO"""

    def setUp(self):
        self.debugging = unittest.debug
        if self.debugging:
            self.path = pjoin(MG5DIR, 'tmp_test_upc')
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            os.mkdir(self.path)
        else:
            self.path = tempfile.mkdtemp(prefix='acc_test_upc')
        self.run_dir = pjoin(self.path, 'MGPROC')

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
        mg_cmd.exec_cmd('import model sm')
        mg_cmd.exec_cmd('generate a a > mu+ mu-')
        mg_cmd.exec_cmd('output %s -f' % self.run_dir)
        self.cmd_line = MECmd.MadEventCmdShell(me_dir=self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False --no_save')
        self.nrun = 0

    def tearDown(self):
        if not self.debugging:
            shutil.rmtree(self.path)

    def write_card(self, pdlabel, beam1=PB208, beam2=PB208, ebeam=E_PBPB, nevents=1000):
        path = pjoin(self.run_dir, 'Cards', 'run_card.dat')
        card = banner.RunCardLO(path)
        set_upc_beams(card, pdlabel, beam1, beam2, ebeam)
        for name, value in [('ptl', 5), ('etal', 2.5), ('drll', 0), ('mmll', 0),
                            ('nevents', nevents), ('use_syst', False), ('iseed', 0)]:
            card.set(name, value, user=True)
        card.write(path)

    def run_upc(self, pdlabel, **opts):
        """cross section and error (pb) of one run"""
        self.write_card(pdlabel, **opts)
        self.nrun += 1
        self.cmd_line.exec_cmd('generate_events upc_%i -f' % self.nrun, errorhandling=False,
                               precmd=True)
        return self.cmd_line.results.current['cross'], self.cmd_line.results.current['error']

    def test_upc_pbpb_edff(self):
        """inclusive PbPb: the regression anchor of the gamma-UPC upgrade"""

        compatible(self, *self.run_upc('edff'), *REF_LO[('PbPb', 'edff')])

    def test_upc_pbpb_neutron_classes(self):
        """the 0n0n, 0nXn, Xn0n and XnXn classes add up to the inclusive cross
        section (their probabilities sum to one at each impact parameter)"""

        total, total_error2 = 0., 0.
        incl, incl_error = REF_LO[('PbPb', 'edff')]
        for label in ['edf0n0n', 'edf0nxn', 'edfxn0n', 'edfxnxn']:
            value, error = self.run_upc(label)
            compatible(self, value, error, *REF_LO[('PbPb', label)])
            self.assertLess(value, incl)
            total += value
            total_error2 += error**2
        compatible(self, total, math.sqrt(total_error2), incl, incl_error)

        # one tagged beam, the other inclusive: used to stop in gamma-UPC
        value, error = self.run_upc('edf1nin')
        compatible(self, value, error, *REF_LO[('PbPb', 'edf1nin')])
        self.assertLess(value, REF_LO[('PbPb', 'edfxn0n')][0] + REF_LO[('PbPb', 'edfxnxn')][0])

    def test_upc_pbpb_neutron_band(self):
        """upper/lower neutron-emission cross sections: more (less) break-up,
        so XnXn goes up (down) and 0n0n down (up). For exact multiplicities
        (1n, 2n, ...) there is no such ordering."""

        central = REF_LO[('PbPb', 'edfxnxn')][0]
        up, up_error = self.run_upc('edpxnxn')
        compatible(self, up, up_error, *REF_LO[('PbPb', 'edpxnxn')])
        down, down_error = self.run_upc('edmxnxn')
        compatible(self, down, down_error, *REF_LO[('PbPb', 'edmxnxn')])
        self.assertGreater(up - 3 * up_error, central)
        self.assertLess(down + 3 * down_error, central)

    def test_upc_ppb_proton_tag_ignored(self):
        """pPb: the tag of the proton beam is ignored (edf1n2n = edfin2n)"""

        value, error = self.run_upc('edf1n2n', beam1=PROTON, beam2=PB208, ebeam=E_PPB)
        compatible(self, value, error, *REF_LO[('pPb', 'edfin2n')])

    def test_upc_oo(self):
        """O16+O16 with a 1n tag on each side; 3n is not available for O16
        and is refused when the run_card is checked, before any compilation"""

        value, error = self.run_upc('edf1n1n', beam1=O16, beam2=O16, ebeam=E_OO)
        compatible(self, value, error, *REF_LO[('OO', 'edf1n1n')])
        self.assertLess(value, REF_LO[('OO', 'edff')][0])

        # write the card by hand: RunCardLO.write would not let it through
        path = pjoin(self.run_dir, 'Cards', 'run_card.dat')
        text = open(path).read().replace(' edf1n1n ', ' edf3n3n ')
        self.assertIn('edf3n3n', text)
        open(path, 'w').write(text)
        with self.assertRaises(InvalidCmd):
            self.cmd_line.exec_cmd('generate_events upc_bad -f', errorhandling=False,
                                   precmd=True)


class TestGammaUPCNLO(unittest.TestCase):
    """!a! !a! > b b~ [QCD] in PbPb UPC at fixed-order NLO"""

    def setUp(self):
        self.debugging = unittest.debug
        if self.debugging:
            self.path = pjoin(MG5DIR, 'tmp_test_upc_nlo')
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            os.mkdir(self.path)
        else:
            self.path = tempfile.mkdtemp(prefix='acc_test_upc_nlo')
        self.run_dir = pjoin(self.path, 'MGPROC')

        mg_cmd = MGCmd.MasterCmd()
        mg_cmd.no_notification()
        mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
        # precmd/postcmd: the NLO output reads the generate line from the history
        for line in ['import model loop_sm', 'generate !a! !a! > b b~ [QCD]',
                     'output %s -f' % self.run_dir]:
            mg_cmd.exec_cmd(line, errorhandling=False, precmd=True, postcmd=True)
        self.cmd_line = NLOCmd.aMCatNLOCmdShell(me_dir=self.run_dir)
        self.cmd_line.no_notification()
        self.cmd_line.exec_cmd('set automatic_html_opening False --no_save')
        self.nrun = 0

    def tearDown(self):
        if not self.debugging:
            shutil.rmtree(self.path)

    def run_upc(self, pdlabel):
        path = pjoin(self.run_dir, 'Cards', 'run_card.dat')
        card = banner.RunCardNLO(path)
        set_upc_beams(card, pdlabel, PB208, PB208, E_PBPB)
        card.set('req_acc_FO', 0.01, user=True)
        card.set('iseed', 0, user=True)
        card.write(path)
        self.nrun += 1
        name = 'upc_%i' % self.nrun
        self.cmd_line.exec_cmd('calculate_xsect NLO -f --name=%s' % name, errorhandling=False,
                               precmd=True)
        summary = open(pjoin(self.run_dir, 'Events', name, 'summary.txt')).read()
        value, rest = summary.split('Total cross section:')[1].split('+-')
        return float(value), float(rest.split()[0])

    def test_upc_nlo_pbpb(self):
        """EDFF; EDFF with the upper neutron-emission cross section and no
        tagging (edpf: the same luminosity, and its library used not to be
        linked at NLO); EDFF with 0nXn"""

        compatible(self, *self.run_upc('edff'), *REF_NLO[('PbPb', 'edff')])
        compatible(self, *self.run_upc('edpf'), *REF_NLO[('PbPb', 'edff')])
        value, error = self.run_upc('edf0nxn')
        compatible(self, value, error, *REF_NLO[('PbPb', 'edf0nxn')])
        self.assertLess(value, REF_NLO[('PbPb', 'edff')][0])
