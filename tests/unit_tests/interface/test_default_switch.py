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
"""Unit tests for the user/site defaults of the launch switches
   (input/default_switch.txt)."""

from __future__ import absolute_import
import os
import shutil
import tempfile
import unittest

import madgraph
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.various.misc as misc

pjoin = os.path.join
MG5DIR = madgraph.MG5DIR


class FakeMother(object):
    """The little the ControlSwitch needs from the interface asking."""

    def __init__(self, mg5_path=None):
        self.options = {'mg5_path': mg5_path}
        self.me_dir = mg5_path


class FakeSwitch(ext_cmd.ControlSwitch):
    """A ControlSwitch built without asking anything: only the default
       resolution is under test here."""

    to_control = [('shower', 'shower program'),
                  ('detector', 'detector simulation'),
                  ('analysis', 'analysis program')]

    def get_allowed_shower(self):
        return ['Pythia8', 'OFF']

    def get_allowed_detector(self):
        return ['Delphes', 'OFF']

    def get_allowed_analysis(self):
        return ['MadAnalysis5', 'OFF']

    def set_default_shower(self):
        self.switch['shower'] = 'OFF'

    def set_default_detector(self):
        # a derived default: no detector simulation without a shower
        self.switch['detector'] = 'OFF' if self.switch['shower'] == 'OFF' \
                                        else 'Delphes'

    def set_default_analysis(self):
        # like the real LO 'detector' default, this one also (re)computes
        # another switch -- the user file has to win over that too.
        self.set_default_shower()
        self.switch['analysis'] = 'OFF'

    def consistency_detector_shower(self, vdetector, vshower):
        return 'Pythia8' if vdetector == 'Delphes' and vshower == 'OFF' else None

    def consistency_shower_detector(self, vshower, vdetector):
        return 'OFF' if vshower == 'OFF' and vdetector == 'Delphes' else None

    @classmethod
    def build(cls, mg5_path=None):
        obj = cls.__new__(cls)
        obj.mother_interface = FakeMother(mg5_path)
        obj.switch = dict((key.lower(), 'temporary') for key, _ in cls.to_control)
        obj.inconsistent_keys = {}
        obj.inconsistent_details = {}
        obj.last_changed = []
        return obj


class TestDefaultSwitchFile(unittest.TestCase):
    """input/default_switch.txt sets the default of the launch question"""

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='test_default_switch')
        os.mkdir(pjoin(self.path, 'input'))

    def tearDown(self):
        shutil.rmtree(self.path)

    def write(self, content):
        with open(pjoin(self.path, 'input', 'default_switch.txt'), 'w') as fsock:
            fsock.write(content)

    def defaults(self, mg5_path=None):
        """Defaults of the question for an installation rooted at self.path.
        Patching madgraph.MG5DIR (rather than only passing mg5_path) is what
        a source installation looks like, and it is the branch that wins."""
        obj = FakeSwitch.build(mg5_path)
        with misc.TMP_variable(madgraph, 'MG5DIR',
                               None if mg5_path else self.path):
            obj.set_default_switch()
        return obj.switch

    def test_no_file(self):
        """without the file the computed defaults are kept"""
        self.assertEqual(self.defaults(),
                         {'shower': 'OFF', 'detector': 'OFF',
                          'analysis': 'OFF'})

    def test_value_set(self):
        """an entry of the file becomes the default of the question"""
        self.write('analysis = MadAnalysis5\n')
        self.assertEqual(self.defaults()['analysis'], 'MadAnalysis5')

    def test_comment_and_case(self):
        """comments/blank lines are skipped, the value is case insensitive"""
        self.write('# analysis = OFF\n\n  shower = pythia8  # my default\n')
        switch = self.defaults()
        self.assertEqual(switch['shower'], 'Pythia8')
        self.assertEqual(switch['analysis'], 'OFF')

    def test_not_overwritten_by_other_default(self):
        """a set_default_XXX touching another switch does not win over the file"""
        self.write('shower = Pythia8\n')
        self.assertEqual(self.defaults()['shower'], 'Pythia8')

    def test_derived_default_follows_the_file(self):
        """a default computed from another switch is recomputed on top of it"""
        self.write('shower = Pythia8\n')
        switch = self.defaults()
        self.assertEqual(switch['shower'], 'Pythia8')
        self.assertEqual(switch['detector'], 'Delphes')

    def test_conflict_is_resolved(self):
        """an inconsistent pair is resolved, the question opens clean"""
        self.write('detector = Delphes\nshower = OFF\n')
        obj = FakeSwitch.build()
        with misc.TMP_variable(madgraph, 'MG5DIR', self.path):
            obj.set_default_switch()
        self.assertEqual(obj.switch, {'shower': 'OFF', 'detector': 'OFF',
                                      'analysis': 'OFF'})
        self.assertEqual(obj.inconsistent_keys, {})

    def test_invalid_value_ignored(self):
        """a value that is not available falls back on the computed default"""
        self.write('shower = Herwig7\n')
        self.assertEqual(self.defaults()['shower'], 'OFF')

    def test_standalone_directory_uses_mg5_path(self):
        """a process directory has no input/: it reads the installation's"""
        self.write('analysis = MadAnalysis5\n')
        self.assertEqual(self.defaults(mg5_path=self.path)['analysis'],
                         'MadAnalysis5')

    def test_unknown_key_ignored(self):
        """the file is shared by the LO/NLO/mg7 questions: extra keys are ok"""
        self.write('madanalysis = ON\nanalysis = MadAnalysis5\n')
        switch = self.defaults()
        self.assertEqual(switch['analysis'], 'MadAnalysis5')
        self.assertNotIn('madanalysis', switch)

    def test_line_without_equal_ignored(self):
        """a malformed line does not break the launch question"""
        self.write('analysis MadAnalysis5\nanalysis = MadAnalysis5\n')
        self.assertEqual(self.defaults()['analysis'], 'MadAnalysis5')

    def test_template_is_fully_commented(self):
        """the shipped template must not change anybody's default"""
        template = pjoin(MG5DIR, 'input', '.default_switch.txt')
        self.assertTrue(os.path.exists(template))
        shutil.copy(template, pjoin(self.path, 'input', 'default_switch.txt'))
        self.assertEqual(self.defaults(),
                         {'shower': 'OFF', 'detector': 'OFF',
                          'analysis': 'OFF'})


class ShortcutSwitch(ext_cmd.ControlSwitch):
    """A switch question with an '@' shortcut, like the aMC@NLO one."""

    default_switch_file = None
    to_control = [('shower', 'shower program')]

    def get_allowed_shower(self):
        return ['PYTHIA8', 'OFF']

    def set_default_shower(self):
        self.switch['shower'] = 'OFF'

    def ans_amc__at__nlo(self, value):
        self.switch['shower'] = 'PYTHIA8'


class TestSwitchShortcuts(unittest.TestCase):
    """Mode shortcuts (lo, nlo, aMC@NLO, ...) answered from a script."""

    def test_at_shortcut_is_a_valid_scripted_answer(self):
        """'@' is spelled '__at__' in the ans_ method name. The answers a
           script may give were built from that name, so 'aMC@NLO' in a
           command file was rejected ('not valid for current question'), the
           launch ran with the previous mode, and every later line of the
           launch block was executed after the run."""
        mother = FakeMother()
        question = ShortcutSwitch(ShortcutSwitch.to_control, mother,
                                  mother_interface=mother)
        self.assertIn('aMC@NLO'.lower(), question.allow_arg)
        # the method-name spelling, which the switch help used to list,
        # stays valid for existing command files
        self.assertIn('amc__at__nlo', question.allow_arg)
