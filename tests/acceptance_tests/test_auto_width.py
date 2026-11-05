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
"""compute_widths with madspace and with madevent.

Both integrate the N-body decays of the same amplitudes, so the cards they write
have to agree: every channel within the integration error, and the totals
within it too. The processes are the SM ones with a real W or Z in the final
state (h > W d u~, Z > W e nu, ...), and the particles go in one call, which is
what makes the output of the second particle start from the matrix elements of
the first one if nothing resets them.
"""

from __future__ import absolute_import

import os
import re
import shutil
import tempfile
import unittest

import madgraph.interface.master_interface as mgcmd
import madgraph.models.check_param_card as check_param_card
from madgraph.mg7 import bootstrap as mg7_bootstrap

pjoin = os.path.join


def read_decays(path):
    """{pid: {'total': GeV, 'channels': {sorted products: GeV}}} of a param_card"""
    decays = {}
    current = None
    with open(path) as stream:
        for line in stream:
            match = re.match(r'DECAY\s+(\d+)\s+([-\d.eE+]+)', line)
            if match:
                current = int(match.group(1))
                decays[current] = {'total': float(match.group(2)), 'channels': {}}
                continue
            match = re.match(r'\s*[-\d.eE+]+\s+\d+\s+((?:-?\d+\s+)+)#\s*([-\d.eE+]+)', line)
            if match and current is not None:
                products = tuple(sorted(int(x) for x in match.group(1).split()))
                decays[current]['channels'][products] = float(match.group(2))
    return decays


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class TestAutoWidthEngines(unittest.TestCase):
    """The cards of --engine=madspace and --engine=madevent agree."""

    PARTICLES = {6: 't', 23: 'z', 24: 'w+', 25: 'h'}

    @classmethod
    def setUpClass(cls):
        cls.path = tempfile.mkdtemp(prefix='acc_test_auto_width')
        cmd = mgcmd.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False --no_save')
        cmd.exec_cmd('import model sm')
        cls.cards = {}
        for engine in ('madspace', 'madevent'):
            card = pjoin(cls.path, 'param_card_%s.dat' % engine)
            cmd.exec_cmd('compute_widths t w+ z h --body_decay=3 --engine=%s '
                         '--output=%s' % (engine, card))
            cls.cards[engine] = read_decays(card)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def test_every_particle_has_a_decay_table_with_both(self):
        for engine, decays in self.cards.items():
            for pid in self.PARTICLES:
                self.assertIn(pid, decays, '%s: no decays for %s' % (engine, pid))
                self.assertTrue(decays[pid]['channels'])

    def test_total_widths(self):
        for pid, name in self.PARTICLES.items():
            madspace = self.cards['madspace'][pid]['total']
            madevent = self.cards['madevent'][pid]['total']
            self.assertAlmostEqual(madspace / madevent, 1, delta=0.01,
                                   msg='total width of %s: madspace %g, madevent %g'
                                   % (name, madspace, madevent))

    def test_same_channels_in_both(self):
        for pid, name in self.PARTICLES.items():
            self.assertEqual(sorted(self.cards['madspace'][pid]['channels']),
                             sorted(self.cards['madevent'][pid]['channels']),
                             'channels of %s' % name)

    def test_channels(self):
        """Channel by channel, to 5%: madevent aims at 1% for each of them (and
        the smallest channels of the Z are only a few 1e-8 GeV)."""
        for pid, name in self.PARTICLES.items():
            madspace = self.cards['madspace'][pid]['channels']
            madevent = self.cards['madevent'][pid]['channels']
            for products, expected in madevent.items():
                self.assertAlmostEqual(
                    madspace[products] / expected, 1, delta=0.05,
                    msg='%s > %s: madspace %g, madevent %g'
                    % (name, products, madspace[products], expected))

    def test_the_decays_of_one_particle_are_not_those_of_another(self):
        """The three-body decays of the Z are 1e-7 GeV: they are small because
        the Z is barely above the W threshold. The widths of the Higgs, with
        the same final states, are a thousand times larger."""
        for engine, decays in self.cards.items():
            for products, width in decays[23]['channels'].items():
                if len(products) == 3:
                    self.assertLess(width, 1e-6, '%s: Z > %s' % (engine, products))


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
@unittest.skipUnless(shutil.which('gfortran'), 'no Fortran compiler (aMC@NLO)')
class TestAutoWidthNLO(unittest.TestCase):
    """compute_widths --nlo with MC@NLO.

    The SM of the repository has no dedicated code (SMWidth), so the two-body decays
    with a colour charge are run at NLO with aMC@NLO (loop_sm) and the decays into
    three bodies stay at LO, madspace's. t > w+ b is the known -9% (1 + 2 alpha_s/3pi
    (5/2 - 2 pi^2/3)); h > b b~ has the pole-mass Yukawa of loop_sm, which makes it
    1 + 4/3 alpha_s/pi (9/4 + 3/2 log(m_b^2/m_H^2)) = 0.64.
    """

    @classmethod
    def setUpClass(cls):
        cls.path = tempfile.mkdtemp(prefix='acc_test_auto_width_nlo')
        cmd = mgcmd.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False --no_save')
        cmd.exec_cmd('import model sm')
        cls.cards = {}
        for name, option in (('lo', ''), ('nlo', '--nlo')):
            card = pjoin(cls.path, 'param_card_%s.dat' % name)
            cmd.exec_cmd('compute_widths t h --body_decay=3 %s --output=%s' % (option, card))
            cls.cards[name] = read_decays(card)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def k_factor(self, pid, products):
        nlo = self.cards['nlo'][pid]['channels'][products]
        lo = self.cards['lo'][pid]['channels'][products]
        return nlo / lo

    def test_the_top(self):
        self.assertAlmostEqual(self.k_factor(6, (5, 24)), 0.909, delta=0.02)

    def test_the_higgs_to_bottoms(self):
        self.assertAlmostEqual(self.k_factor(25, (-5, 5)), 0.647, delta=0.04)

    def test_no_colour_no_correction(self):
        """h > ta+ ta-: the same width, exact in both"""
        self.assertAlmostEqual(self.k_factor(25, (-15, 15)), 1, places=9)

    def test_three_bodies_stay_at_lo(self):
        """Another madspace run, so the integration error: 5% on each channel"""
        for products, lo in self.cards['lo'][25]['channels'].items():
            if len(products) == 3:
                self.assertAlmostEqual(self.cards['nlo'][25]['channels'][products] / lo,
                                       1, delta=0.05, msg=str(products))

    def test_the_total_is_the_sum(self):
        for name, decays in self.cards.items():
            for pid in (6, 25):
                self.assertAlmostEqual(
                    decays[pid]['total'], sum(decays[pid]['channels'].values()),
                    delta=1e-4 * decays[pid]['total'], msg='%s %s' % (name, pid))


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
@unittest.skipUnless(shutil.which('gfortran'), 'no Fortran compiler (aMC@NLO)')
class TestAutoWidthOptionsInCard(unittest.TestCase):
    """DECAY 6 auto --nlo --body_decay=3 in the param_card of an mg7 output: each
    width is computed with its own options, and none is lost on the way."""

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='acc_test_auto_width_card')
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.path, ignore_errors=True)

    def test_options_of_the_widths(self):
        cmd = mgcmd.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False --no_save')
        cmd.exec_cmd('import model sm')
        cmd.exec_cmd('generate g g > t t~')
        run_dir = pjoin(self.path, 'PROC')
        cmd.exec_cmd('output %s -f' % run_dir)

        card = pjoin(run_dir, 'Cards', 'param_card.dat')
        with open(card) as stream:
            text = stream.read()
        text = re.sub(r'(?m)^DECAY\s+6\s+[^\n#]*', 'DECAY  6 auto --nlo --body_decay=3 ', text)
        text = re.sub(r'(?m)^DECAY\s+24\s+[^\n#]*', 'DECAY  24 auto --body_decay=2 ', text)
        with open(card, 'w') as stream:
            stream.write(text)

        from madgraph.mg7 import launch as mg7_launch
        os.chdir(run_dir)
        mg7_launch.compute_auto_widths(card)
        os.chdir(self.cwd)

        decays = read_decays(card)
        # t > w+ b at NLO: 1.356 GeV, the LO width being 1.4915 GeV
        self.assertAlmostEqual(decays[6]['total'], 1.356, delta=0.04)
        self.assertEqual(sorted(decays[6]['channels']), [(5, 24)])
        # w+ at LO, two bodies: the 2.0479 GeV of the formulas
        self.assertAlmostEqual(decays[24]['total'], 2.0479, delta=0.005)
        with open(card) as stream:
            self.assertEqual(len(check_param_card.group_auto_widths(stream.read())), 0)


@unittest.skipUnless(shutil.which('gfortran'), 'no Fortran compiler (SMWidth, aMC@NLO)')
class TestAutoWidthNLOvsSMWidth(unittest.TestCase):
    """The two NLO codes on the SM of loop_qcd_qed_sm: SMWidth (QCD and EW) and MC@NLO
    (QCD), the table of docs/source/madevent_to_mg7.rst.

    The QCD corrections of the two agree to 0.05% (SMWidth called with the orders
    (1, 0): 1.4136, 2.1586, 2.5793 GeV for t, W, Z), and what SMWidth has on top of them
    is the EW corrections, -3.3% for the top, -3.4% for the W, -3.5% for the Z. The card
    of this test is in a directory whose path is longer than the 100 characters that
    SMWidth reads.
    """

    # (SMWidth QCD+EW, MC@NLO QCD) in GeV, mass of the top 173.3 GeV
    EXPECTED = {6: (1.3673, 1.4130), 24: (2.0847, 2.1576), 23: (2.4888, 2.5783)}

    @classmethod
    def setUpClass(cls):
        cls.path = tempfile.mkdtemp(prefix='acc_test_auto_width_smwidth_' + 'x' * 70)
        cmd = mgcmd.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False --no_save')
        cls.cards = {}
        try:
            cmd.exec_cmd('import model loop_qcd_qed_sm')
        except Exception as error:
            shutil.rmtree(cls.path, ignore_errors=True)
            raise unittest.SkipTest('loop_qcd_qed_sm is not available: %s' % error)
        for name, option in (('smwidth', '--nlo=smwidth'),
                             ('mcatnlo', '--nlo=mcatnlo --engine=madevent')):
            card = pjoin(cls.path, 'param_card_%s.dat' % name)
            cmd.exec_cmd('compute_widths t w+ z %s --output=%s' % (option, card))
            cls.cards[name] = read_decays(card)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def test_smwidth(self):
        for pid, (expected, _) in self.EXPECTED.items():
            self.assertAlmostEqual(self.cards['smwidth'][pid]['total'], expected,
                                   delta=0.003 * expected, msg=str(pid))

    def test_mcatnlo(self):
        for pid, (_, expected) in self.EXPECTED.items():
            self.assertAlmostEqual(self.cards['mcatnlo'][pid]['total'], expected,
                                   delta=0.005 * expected, msg=str(pid))

    def test_the_qcd_corrections_agree(self):
        """MC@NLO (QCD) over SMWidth (QCD and EW) is what the EW corrections take
        away from SMWidth: 1.033 to 1.036 for t, W and Z. Without the QCD corrections
        it would be 1.13 for the top (LO over NLO QCD+EW) and 1.01 for W and Z."""
        for pid in self.EXPECTED:
            ratio = self.cards['mcatnlo'][pid]['total'] / self.cards['smwidth'][pid]['total']
            self.assertAlmostEqual(ratio, 1.035, delta=0.007, msg=str(pid))


@unittest.skipUnless(shutil.which('gfortran'), 'no Fortran compiler (SMWidth, aMC@NLO)')
class TestAutoWidthNLOEWvsSMWidth(unittest.TestCase):
    """The electroweak corrections of the two-body decays with MC@NLO against SMWidth,
    on the SM of loop_qcd_qed_sm: the NLO QCD+EW widths of the top quark, the W and the
    Z (17 decays, 8 runs of aMC@NLO, about two minutes).

    SMWidth with the orders (1, 1), 1.3918 / 2.0847 / 2.4888 GeV for t, W, Z, and with
    the orders (0, 1), 1.5245 / 2.0309 / 2.4234 GeV, for the electroweak corrections
    alone. The top quark of 'compute_widths --nlo' also has the finite width of the W,
    1.3673 GeV, a three-body effect that the two-body decays do not have: it is left out
    of the expected value. The electroweak correction of Z > b b~ is 1.5% below the one of
    Z > d d~ (the top quark is the isospin partner of the b), which a run of the b for
    the d and the s would miss by 0.5% on the width of the Z.
    """

    EXPECTED = {6: 1.3900, 24: 2.0839, 23: 2.4878}     # MC@NLO, QCD+EW
    SMWIDTH = {6: 1.3918, 24: 2.0847, 23: 2.4888}      # the orders (1, 1)

    @classmethod
    def setUpClass(cls):
        cls.path = tempfile.mkdtemp(prefix='acc_test_auto_width_ew')
        cmd = mgcmd.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False --no_save')
        try:
            cmd.exec_cmd('import model loop_qcd_qed_sm')
        except Exception as error:
            shutil.rmtree(cls.path, ignore_errors=True)
            raise unittest.SkipTest('loop_qcd_qed_sm is not available: %s' % error)
        cls.cards = {}
        for name, option in (
                ('smwidth', '--nlo=smwidth'),
                ('mcatnlo', '--nlo --corrections=QCD+QED --body_decay=2 --engine=madevent')):
            card = pjoin(cls.path, 'param_card_%s.dat' % name)
            cmd.exec_cmd('compute_widths t w+ z %s --output=%s' % (option, card))
            cls.cards[name] = read_decays(card)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def test_mcatnlo(self):
        for pid, expected in self.EXPECTED.items():
            self.assertAlmostEqual(self.cards['mcatnlo'][pid]['total'], expected,
                                   delta=0.004 * expected, msg=str(pid))

    def test_agrees_with_smwidth(self):
        for pid, expected in self.SMWIDTH.items():
            self.assertAlmostEqual(self.cards['mcatnlo'][pid]['total'] / expected, 1,
                                   delta=0.004, msg=str(pid))

    def test_the_built_in_widths_of_the_w_and_the_z(self):
        """no finite width of the W for them: the same widths in both"""
        for pid in (24, 23):
            self.assertAlmostEqual(
                self.cards['mcatnlo'][pid]['total'] / self.cards['smwidth'][pid]['total'],
                1, delta=0.004, msg=str(pid))

    def test_bottom_and_down_are_not_alike(self):
        """Z > b b~ has a larger electroweak correction than Z > d d~"""
        channels = self.cards['mcatnlo'][23]['channels']
        self.assertLess(channels[(-5, 5)], 0.99 * channels[(-1, 1)])
