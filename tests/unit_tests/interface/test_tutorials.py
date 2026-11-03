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
"""Tests for the tutorial mode.

The important one is TestPortedTutorials: it pins the text the three ported
tutorials emit for a given command to what the pre-2026 command-name lookup in
CmdExtended.postcmd emitted, so the port cannot drift.  When the text is later
inlined into the tutorial modules (and tutorial_text*.py deleted) this test is
what makes that safe.
"""

from __future__ import absolute_import

import json
import logging
import re
import sys
import unittest

import madgraph
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.interface.tutorials as tutorials
import madgraph.interface.tutorials.bsm as bsm
import madgraph.interface.tutorials.mixin as tutorial_mixin
import madgraph.interface.tutorial_text_nlo as legacy_nlo
import madgraph.interface.tutorial_text_madloop as legacy_madloop

from madgraph.interface.tutorials._port import retarget
from madgraph.interface.tutorials.session import (Exercise, Step, Tutorial,
                                                  TutorialSession)


class _Capture(logging.Handler):
    """Collect what the tutorial logger emits."""

    def __init__(self):
        logging.Handler.__init__(self)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class _BareInterface(mg_interface.CmdExtended):
    """A CmdExtended that skips the (expensive) real __init__.

    postcmd only needs the instance to answer hasattr(self, 'post_XXX'), so
    this is enough to drive the tutorial hook.
    """

    def __init__(self):
        pass


class _TutorialTestCase(unittest.TestCase):

    def setUp(self):
        self.capture = _Capture()
        self.logger = logging.getLogger('tutorial')
        self._saved = (self.logger.handlers, self.logger.propagate,
                       self.logger.level)
        self.logger.handlers = [self.capture]
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        # attach() arms module-level switches in extended_cmd, and a test that
        # does not detach would leave them armed for the next one -- start and
        # end from a known state rather than inheriting one
        self._reset_question_hooks()

    def tearDown(self):
        (self.logger.handlers, self.logger.propagate,
         self.logger.level) = self._saved
        self._reset_question_hooks()

    @staticmethod
    def _reset_question_hooks():
        import madgraph.interface.extended_cmd as extended_cmd

        extended_cmd.question_hint = None
        extended_cmd.suppress_timeout = False

    def run_lines(self, tutorial_name, lines):
        """Feed command lines to a fresh session; return [(line, [text])]."""

        interface = _BareInterface()
        tutorial_mixin.attach(interface, tutorials.start(tutorial_name))
        out = []
        for line in lines:
            self.capture.messages = []
            interface.postcmd(True, line)
            out.append((line, list(self.capture.messages)))
        return out


#===============================================================================
# the port is faithful
#===============================================================================

# every command line the pre-2026 lookup could resolve, plus a few that it
# deliberately could not
COMMAND_LINES = [
    'generate p p > t t~',
    'add process p p > W+ j, W+ > l+ vl @2',
    'display processes',
    'display diagrams',
    'output MY_FIRST_MG5_RUN',
    'open index.html',
    'launch MY_FIRST_MG5_RUN',
    'import model MSSM_SLHA2',
    'import model_v4 sm',
    'customize_model',
    'define v = w+ w- z a',
    'history my_mg5_cmd.dat',
    'open ./my_mg5_cmd.dat',
    'check',
    'check profile p p > t t~',
    'tutorial',
    'set gauge Feynman',
    '',
    'quit',
]


def _legacy_text(module, line):
    """What CmdExtended.postcmd used to print for `line`, or None.

    A transcription of the pre-2026 lookup, kept here on purpose: it is the
    specification the port has to meet.  `retarget` applies the one change the
    port is allowed to make -- the stale MG5_aMC> prompt -- so that any other
    drift still fails.
    """

    args = line.split()
    if not args:
        return None
    if len(args) == 1:
        keys = [args[0]]
    else:
        keys = [args[0] + '_' + args[1].split('.')[0], args[0]]
    for key in keys:
        text = getattr(module, key, None)
        if isinstance(text, str):
            return retarget(text).replace('\n', '\n\t')
    return None


class TestPortedTutorials(_TutorialTestCase):
    """`nlo` and `madloop` emit exactly what the old lookup emitted.

    `lo` is deliberately absent: it was rewritten, not ported (see the plan in
    docs/tutorial-mode-plan.md), so it has TestLoTutorial below instead.
    """

    def _check(self, name, legacy_module):
        for line, emitted in self.run_lines(name, COMMAND_LINES):
            expected = _legacy_text(legacy_module, line)
            if expected is None:
                self.assertEqual(emitted, [],
                                 'tutorial %s should stay silent on %r, said %r'
                                 % (name, line, emitted))
            else:
                self.assertEqual(len(emitted), 1,
                                 'tutorial %s said %d things on %r'
                                 % (name, len(emitted), line))
                # the ported text must survive in full; a tutorial may append
                # to it (nlo adds shower guidance) but may not alter it
                self.assertIn(expected, emitted[0],
                              'tutorial %s drifted on %r' % (name, line))

    def test_nlo_matches_legacy(self):
        self._check('nlo', legacy_nlo)

    def test_madloop_matches_legacy(self):
        self._check('madloop', legacy_madloop)

    def test_no_stale_prompt_anywhere(self):
        """No tutorial still tells the user to type at an MG5_aMC> prompt."""

        for tutorial in tutorials.all_tutorials(include_hidden=True):
            for step in tutorial.steps:
                text = step.render(None)
                for stale in ('MG5_aMC>', 'MG_aMC>'):
                    self.assertNotIn(
                        stale, text,
                        'tutorial %s, step %r still uses the %s prompt'
                        % (tutorial.name, step.title, stale))


class TestLoTutorial(_TutorialTestCase):
    """The rewritten `lo`: a short, sequenced path to an event sample."""

    def test_is_sequenced_and_short(self):
        tutorial = tutorials.get('lo')
        self.assertEqual(tutorial.order, 'sequence')
        self.assertLessEqual(len(tutorial.steps), 8,
                             'lo has grown back into a grab-bag')

    def test_reaches_launch(self):
        """The point of `lo` is that it gets you to an event sample."""

        keys = [step.key for step in tutorials.get('lo').steps]
        for expected in ('generate', 'output', 'launch'):
            self.assertIn(expected, keys,
                          'lo no longer walks the user through %r' % expected)

    def test_intro_adapts_to_madspace(self):
        """Step 0 asks for `install madspace` only when it is missing."""

        import madgraph.interface.tutorials.lo as lo_module

        original = lo_module.madspace_is_installed
        try:
            lo_module.madspace_is_installed = lambda: False
            self.assertIn('install madspace', lo_module.intro(None))
            lo_module.madspace_is_installed = lambda: True
            self.assertNotIn('install madspace', lo_module.intro(None))
        finally:
            lo_module.madspace_is_installed = original

    def test_it_points_at_tutorial_list_rather_than_listing(self):
        """Tutorials used to close by listing the others with their
        descriptions -- eleven lines that grew with every new tutorial, and
        that `tutorial list` already prints on demand."""

        text = tutorials.get('lo').steps[-1].render(None)
        self.assertIn('tutorial list', text)
        self.assertNotIn('generate, output and run a process', text)

    def test_no_tutorial_names_one_that_does_not_exist(self):
        """Whatever a step mentions inline still has to be real."""

        class _Empty(object):
            _curr_amps = []
            options = {}

        registered = set(tutorials.names(include_aliases=True))
        # sub-commands, not tutorials
        registered |= {'stop', 'list', 'status', 'help', 'hint', 'solution',
                       'next', 'repeat', 'back', 'skip'}
        for tutorial in tutorials.all_tutorials(include_hidden=True):
            for step in tutorial.steps:
                for name in re.findall(r'`tutorial ([a-z0-9_]+)`',
                                       step.render(_Empty())):
                    self.assertIn(
                        name, registered,
                        '%s / %s points at `tutorial %s`, which does not exist'
                        % (tutorial.name, step.title, name))


#===============================================================================
# the registry
#===============================================================================

class TestRegistry(unittest.TestCase):

    def test_primary_names(self):
        names = [t.name for t in tutorials.all_tutorials()]
        for expected in ('lo', 'nlo', 'madloop'):
            self.assertIn(expected, names)

    def test_legacy_names_are_aliases(self):
        """The pre-2026 names keep working, and resolve to the new ones."""

        for old, new in (('MadGraph5', 'lo'), ('aMCatNLO', 'nlo'),
                         ('MadLoop', 'madloop')):
            self.assertIsNotNone(tutorials.get(old), '%s no longer accepted' % old)
            self.assertEqual(tutorials.get(old).name, new)

    def test_lookup_is_case_insensitive(self):
        self.assertEqual(tutorials.get('MADGRAPH5').name, 'lo')
        self.assertEqual(tutorials.get('LO').name, 'lo')

    def test_unknown_name(self):
        self.assertIsNone(tutorials.get('not-a-tutorial'))

    def test_every_step_has_a_title(self):
        for tutorial in tutorials.all_tutorials(include_hidden=True):
            for step in tutorial.steps:
                self.assertTrue(step.title,
                                'step %r of %s has no title' % (step.key, tutorial.name))


#===============================================================================
# the session
#===============================================================================

class TestKeys(unittest.TestCase):
    """The command line -> key reduction must match the old getattr chain."""

    def test_two_word_key_first(self):
        self.assertEqual(TutorialSession.keys_for('import model sm'),
                         ('import_model', 'import'))

    def test_dot_is_truncated(self):
        self.assertEqual(TutorialSession.keys_for('open index.html'),
                         ('open_index', 'open'))

    def test_single_word(self):
        self.assertEqual(TutorialSession.keys_for('customize_model'),
                         ('customize_model',))

    def test_empty(self):
        self.assertEqual(TutorialSession.keys_for('   '), ())


class TestOrdering(_TutorialTestCase):

    def _tutorial(self, order):
        return Tutorial(
            name='fake-%s' % order, title='fake', order=order,
            steps=[Step('generate', 'FIRST', title='one'),
                   Step('output', 'SECOND', title='two'),
                   Step('generate', 'THIRD', title='three')])

    def test_sequence_advances_through_repeated_commands(self):
        """A tutorial teaching `generate` twice gets two different lessons.

        This is what the old command-name lookup could not express and is the
        reason the engine exists.
        """

        session = TutorialSession(self._tutorial('sequence'))
        self.assertEqual(session.step_for('generate p p > t t~')[1].text, 'FIRST')
        session.advance(0)
        self.assertEqual(session.step_for('output PROC')[1].text, 'SECOND')
        session.advance(1)
        self.assertEqual(session.step_for('generate p p > z')[1].text, 'THIRD')

    def test_sequence_does_not_jump_past_a_lesson(self):
        """A command a later step is keyed on does not fire it from here.

        It used to: 'generate' at step one fired step three, skipping the
        `output` lesson between -- the way `display modellist`, mentioned in
        passing by the bsm intro, fired the next `display` step two lessons on.
        Only what the current step waits for moves the tutorial.
        """

        session = TutorialSession(self._tutorial('sequence'))
        session.advance(0)
        self.assertIsNone(session.step_for('generate p p > z'))

    def test_only_the_command_asked_for_moves_it(self):
        """Close enough is the command and its sub-command; the rest is the
        reader's to choose."""

        def at(name, index, line):
            session = TutorialSession(tutorials.get(name))
            session.index = index
            found = session.step_for(line)
            return found[0] if found else None

        self.assertIsNone(at('bsm', 0, 'display modellist'))
        self.assertEqual(at('bsm', 0, 'import model SMEFTatNLO-NLO'), 1)
        self.assertIsNone(at('checks', 2, 'check lorentz p p > e+ e-'))
        self.assertEqual(at('checks', 2, 'check gauge p p > mu+ mu-'), 3)
        # another directory is fine, another output format is not
        self.assertEqual(at('decays', 1, 'output MY_DIR'), 2)
        self.assertIsNone(at('decays', 1, 'output madevent MY_DIR'))
        # another process is fine too
        self.assertEqual(at('syntax', 0, 'generate p p > w+ w- QED<=4'), 1)

    def test_the_signature_of_a_command(self):
        from madgraph.interface.tutorials.session import command_signature

        self.assertEqual(command_signature('output TT'), ('output', 'mg7'))
        self.assertEqual(command_signature('output madevent TT'),
                         ('output', 'madevent'))
        self.assertEqual(command_signature('check p p > e+ e-'),
                         ('check', 'full'))
        self.assertEqual(command_signature('display modellist'),
                         ('display', 'modellist'))
        self.assertEqual(command_signature('generate p p > t t~'),
                         ('generate',))
        self.assertEqual(command_signature('add process p p > z'),
                         ('add', 'process'))

    def test_sequence_does_not_refire_the_current_step(self):
        session = TutorialSession(self._tutorial('sequence'))
        session.advance(2)
        self.assertIsNone(session.step_for('generate p p > z'))

    def test_free_always_takes_the_first_match(self):
        session = TutorialSession(self._tutorial('free'))
        session.advance(2)
        self.assertEqual(session.step_for('generate p p > z')[1].text, 'FIRST')

    def test_specific_key_wins_over_bare_one(self):
        """'open index.html' must resolve to the open_index step even though a
        plain 'open' step sits earlier in the list."""

        tutorial = Tutorial(name='fake-keys', title='fake', order='free',
                            steps=[Step('open', 'BARE', title='bare'),
                                   Step('open_index', 'SPECIFIC', title='specific')])
        session = TutorialSession(tutorial)
        self.assertEqual(session.step_for('open index.html')[1].text, 'SPECIFIC')
        self.assertEqual(session.step_for('open other.txt')[1].text, 'BARE')

    def test_no_match(self):
        session = TutorialSession(self._tutorial('free'))
        self.assertIsNone(session.step_for('display particles'))


#===============================================================================
# attach / detach
#===============================================================================

class TestAttach(_TutorialTestCase):

    def test_attach_and_detach_restore_the_class(self):
        interface = _BareInterface()
        original = interface.__class__
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        self.assertTrue(tutorial_mixin.is_attached(interface))
        self.assertIsInstance(interface, original)
        tutorial_mixin.detach(interface)
        self.assertFalse(tutorial_mixin.is_attached(interface))
        self.assertIs(interface.__class__, original)

    def test_no_output_once_detached(self):
        interface = _BareInterface()
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        tutorial_mixin.detach(interface)
        self.capture.messages = []
        interface.postcmd(True, 'generate p p > t t~')
        self.assertEqual(self.capture.messages, [])

    def test_attaching_twice_keeps_one_layer(self):
        interface = _BareInterface()
        original = interface.__class__
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        wrapped = interface.__class__
        tutorial_mixin.attach(interface, tutorials.start('nlo'))
        self.assertIs(interface.__class__, wrapped)
        tutorial_mixin.detach(interface)
        self.assertIs(interface.__class__, original)

    def test_switching_tutorial_switches_content(self):
        interface = _BareInterface()
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        tutorial_mixin.attach(interface, tutorials.start('madloop'))
        self.capture.messages = []
        interface.postcmd(True, 'check')
        self.assertEqual(self.capture.messages,
                         [retarget(legacy_madloop.check).replace('\n', '\n\t')])


#===============================================================================
# every tutorial is walkable
#===============================================================================

class TestTutorialsAreWalkable(_TutorialTestCase):
    """A tutorial has to be followable by doing what it says.

    Each step names, in `solution`, the command it asks the user to type; the
    step that command triggers is the next one.  So walking a sequenced
    tutorial by typing its own solutions must visit every step in order.  If a
    lesson ever asks for a command that does not lead anywhere, this fails.

    Two kinds of step sit off the main line and are stepped over.  A *side
    quest* is an optional detour offered by a step, whose own steps are titled
    'side quest: ...'; it still has to lead back.  A *sticky* step answers a
    command the lesson invited without consuming it, so it is never reached by
    walking solutions at all.
    """

    SIDE_QUEST = 'side quest'

    @classmethod
    def off_main_line(cls, step):
        return step.sticky or step.title.startswith(cls.SIDE_QUEST)

    @classmethod
    def main_line_after(cls, steps, index):
        """The step the main line reaches from `index`, skipping the detours."""

        following = index + 1
        while following < len(steps) and cls.off_main_line(steps[following]):
            following += 1
        return following

    def test_syntax_solutions_lead_to_the_next_step(self):
        tutorial = tutorials.get('syntax')
        self.assertIsNotNone(tutorial, 'the syntax tutorial is not registered')
        self.assertEqual(tutorial.order, 'sequence')

        steps = tutorial.steps
        session = TutorialSession(tutorial)
        for index, step in enumerate(steps[:-1]):
            solution = step.get_solution()
            self.assertTrue(solution,
                            'step %d (%s) asks for no command' % (index, step.title))
            session.index = index
            found = session.step_for(solution)
            self.assertIsNotNone(
                found, 'step %d (%s) asks for %r, which triggers nothing'
                % (index, step.title, solution))
            # inside the detour the rule is the plain one; on the main line the
            # detour is stepped over
            expected = (index + 1 if step.title.startswith(self.SIDE_QUEST)
                        else self.main_line_after(steps, index))
            if step.sticky:
                # a sticky step does not consume its lesson: it points back at
                # the command that does, which is the one it sits in front of
                expected = self.main_line_after(steps, index)
            self.assertEqual(
                found[0], expected,
                'step %d (%s) asks for %r, which jumps to step %d rather than %d'
                % (index, step.title, solution, found[0], expected))

    # `lo` puts the command mid-lesson in three of its steps and `exercises`
    # opens on a question rather than on a command; every other sequenced
    # tutorial follows the two rules below.
    OLDER_LAYOUT = ('lo', 'exercises')

    @classmethod
    def sequenced(cls):
        return [t for t in tutorials.all_tutorials(include_hidden=True)
                if t.order == 'sequence' and t.name not in cls.OLDER_LAYOUT]

    def test_every_step_ends_on_the_command_it_waits_for(self):
        """A lesson that ends on prose leaves the reader with nothing to type.

        Several of these steps run to thirty lines, and the command used to sit
        somewhere in the middle of them, under the caveats -- by the end of the
        lesson there was no telling what would make the next one appear.  Every
        step but the closing one ends on the command it is waiting for.
        """

        for tutorial in self.sequenced():
            for step in tutorial.steps[:-1]:
                if isinstance(step, Exercise):
                    continue        # an exercise ends on its question
                solution = step.get_solution()
                self.assertTrue(
                    solution, '%s step %r waits for nothing'
                    % (tutorial.name, step.title))
                lines = [line for line in step.render(None).splitlines()
                         if line.strip()]
                self.assertEqual(
                    lines[-1].strip(), 'MG7> %s' % solution,
                    '%s step %r ends on %r rather than on the command it '
                    'waits for'
                    % (tutorial.name, step.title, lines[-1].strip()))

    def test_a_lesson_after_a_process_opens_on_its_diagram_count(self):
        """And the step a command leads to has to say what the command did.

        The reader has just watched MG5 print a process and diagram count; the
        lesson that follows opens on those numbers -- read back from the
        session by counts_line(), never written into the text -- so it starts
        from what they did rather than from a standing start.
        """

        from madgraph.interface.tutorials.session import counts

        class _Amplitude(object):
            @staticmethod
            def get_number_of_diagrams():
                return 9

        class _Generated(object):
            _curr_amps = [_Amplitude(), _Amplitude()]
            options = {}
            _curr_model = None
            _comparisons = None
            _done_export = None

        interface = _Generated()
        expected = counts(interface)
        self.assertTrue(expected, 'the stub interface counts nothing')

        for tutorial in self.sequenced():
            session = TutorialSession(tutorial)
            for index, step in enumerate(tutorial.steps[:-1]):
                solution = step.get_solution() or ''
                if not (solution.startswith('generate')
                        or solution.startswith('add process')):
                    continue
                session.index = index
                found = session.step_for(solution)
                if found is None:
                    continue
                target = found[1]
                if target.sticky or isinstance(target, Exercise):
                    # a sticky step answers in its own words, and an exercise
                    # marks an answer rather than reporting a count
                    continue
                self.assertIn(
                    expected, target.render(interface),
                    '%s: %r leaves %r, which does not say what it produced'
                    % (tutorial.name, solution, target.title))

    def test_no_syntax_command_is_asked_for_twice_running(self):
        """Leaving a lesson on the command the next one teaches asks the reader
        for the same line twice -- once to get there, once to get out."""

        tutorial = tutorials.get('syntax')
        steps = tutorial.steps
        solutions = [step.get_solution() for step in steps[:-1]]
        for index, solution in enumerate(solutions[1:], start=1):
            if steps[index].sticky or steps[index - 1].sticky:
                # a sticky step does not consume the lesson: pointing back at
                # the same command is exactly its job
                continue
            self.assertNotEqual(
                solution, solutions[index - 1],
                'syntax steps %r and %r both wait for %r'
                % (steps[index - 1].title, steps[index].title, solution))

    def test_the_exclusion_extras_do_not_jump_a_lesson(self):
        """`generate ... $ a` and `... $$ a` are the comparison the exclusion
        lesson invites; `generate ... / a` is the way on.  All three are
        `generate` commands over the same particles, so only the line tells
        them apart -- before the sticky step they all counted as the way on,
        and trying the comparison skipped the reader a lesson."""

        tutorial = tutorials.get('syntax')
        titles = [step.title for step in tutorial.steps]
        lesson = titles.index('excluding particles and s-channels')

        for extra in ('generate p p > e+ e- $ a', 'generate p p > e+ e- $$ a'):
            session = TutorialSession(tutorial)
            session.index = lesson
            found = session.step_for(extra)
            self.assertIsNotNone(found, '%r is answered by nothing' % extra)
            self.assertTrue(found[1].sticky,
                            '%r reaches %r, which would end the lesson'
                            % (extra, found[1].title))
            # and it answers again, as many times as it is tried
            self.assertEqual(session.index, lesson)
            self.assertIsNotNone(session.step_for(extra))

        session = TutorialSession(tutorial)
        session.index = lesson
        found = session.step_for('generate p p > e+ e- / a')
        self.assertIsNotNone(found)
        self.assertFalse(found[1].sticky,
                         'the exclusion lesson no longer has a way on')
        self.assertEqual(found[0], self.main_line_after(tutorial.steps, lesson))

    def test_the_sticky_answer_says_which_operator_was_tried(self):
        """The two are answered differently or the step is not worth having:
        `$` keeps the diagram count, `$$` does not."""

        step = [s for s in tutorials.get('syntax').steps
                if s.title == 'what $ and $$ did'][0]

        class _Interface(object):
            _curr_amps = []
            options = {}

        single = step.render(_Interface(), 'generate p p > e+ e- $ a')
        double = step.render(_Interface(), 'generate p p > e+ e- $$ a')
        self.assertNotEqual(single, double)
        self.assertIn('on-shell', single)
        self.assertIn('dropped', double)

    def test_the_syntax_side_quest_is_reachable_and_comes_back(self):
        """A detour nothing can reach is dead text; one that does not come back
        strands the reader.  The offer is the `output standalone` the step
        before it names."""

        tutorial = tutorials.get('syntax')
        steps = tutorial.steps
        detour = [i for i, step in enumerate(steps)
                  if step.title.startswith(self.SIDE_QUEST)]
        self.assertTrue(detour, 'the syntax side quest has gone')
        self.assertEqual(detour, list(range(detour[0], detour[-1] + 1)),
                         'the side quest steps are not contiguous')

        offered_at = detour[0] - 1
        session = TutorialSession(tutorial)

        # the offer is taken
        session.index = offered_at
        found = session.step_for('output standalone')
        self.assertIsNotNone(found, 'the side quest cannot be reached')
        self.assertEqual(found[0], detour[0],
                         'the side quest offer lands on the wrong step')
        self.assertIn('output standalone', steps[offered_at].render(None),
                      'the step before the side quest does not name its command')

        # ... and it leads back to where skipping it would have gone
        session.index = detour[-1]
        found = session.step_for(steps[detour[-1]].get_solution())
        self.assertIsNotNone(found, 'the side quest does not lead back')
        self.assertEqual(found[0], self.main_line_after(steps, offered_at),
                         'the side quest comes back to the wrong step')

    def test_every_sequenced_step_makes_progress(self):
        """Weaker rule, applied to every sequenced tutorial: doing what a step
        asks must move you forward.  `lo` deliberately skips a step -- the
        intro jumps past `install madspace` when madspace is already there --
        so it cannot use the strict rule above."""

        from madgraph.interface.tutorials.session import Exercise

        for tutorial in tutorials.all_tutorials(include_hidden=True):
            if tutorial.order != 'sequence':
                continue
            session = TutorialSession(tutorial)
            for index, step in enumerate(tutorial.steps[:-1]):
                if isinstance(tutorial.steps[index + 1], Exercise):
                    # an exercise is triggered by its own answer, so the step
                    # before it only has to reach it; the exercise itself is
                    # covered by TestExercises
                    pass
                solution = step.get_solution()
                self.assertTrue(
                    solution,
                    '%s step %d (%s) asks for no command'
                    % (tutorial.name, index, step.title))
                session.index = index - 1 if isinstance(step, Exercise) else index
                found = session.step_for(solution)
                self.assertIsNotNone(
                    found, '%s step %d (%s) asks for %r, which triggers nothing'
                    % (tutorial.name, index, step.title, solution))
                expected = index if isinstance(step, Exercise) else index + 1
                self.assertGreaterEqual(
                    found[0], expected,
                    '%s step %d (%s) asks for %r, which does not move forward'
                    % (tutorial.name, index, step.title, solution))

    def test_sequenced_tutorials_start_on_the_tutorial_command(self):
        """The first step is the intro, triggered by `tutorial NAME` itself."""

        for tutorial in tutorials.all_tutorials(include_hidden=True):
            first = tutorial.steps[0]
            self.assertTrue(first.matches(('tutorial',)),
                            'tutorial %s does not open with an intro step'
                            % tutorial.name)

    def test_last_step_is_terminal(self):
        for tutorial in tutorials.all_tutorials(include_hidden=True):
            session = TutorialSession(tutorial)
            session.index = len(tutorial.steps) - 1
            self.assertTrue(session.finished)
            self.assertIsNone(session.next_step)


#===============================================================================
# the `tutorial` command itself
#===============================================================================

class _BareMadGraphCmd(mg_interface.MadGraphCmd):
    """A MadGraphCmd without the real (expensive) __init__.

    Carries only what check_tutorial / ask_tutorial touch -- and deliberately
    *not* `force`, which is set opportunistically elsewhere (by
    `import command -f`) and is not an attribute every interface has.  Assuming
    it existed crashed a bare `tutorial` on the real MasterCmd.
    """

    def __init__(self, use_rawinput=False):
        self.use_rawinput = use_rawinput
        self.asked = None

    def ask(self, question, default, choices=(), **opts):
        self.asked = (question, default, list(choices))
        self.ask_opts = dict(opts)
        return self._answer


class TestTutorialCommand(unittest.TestCase):

    def test_bare_tutorial_does_not_need_a_force_attribute(self):
        """The regression: `tutorial` with no argument on a fresh interface."""

        interface = _BareMadGraphCmd()
        self.assertFalse(hasattr(interface, 'force'))
        args = []
        interface.check_tutorial(args)          # must not raise
        self.assertEqual(args, [tutorials.all_tutorials()[0].name])

    def test_bare_tutorial_never_blocks_without_a_terminal(self):
        """In a command file or a pipe it keeps its historical meaning."""

        interface = _BareMadGraphCmd(use_rawinput=False)
        interface._answer = 'should not be asked'
        self.assertEqual(interface.ask_tutorial(), 'lo')
        self.assertIsNone(interface.asked, 'the menu blocked on a non-tty')

    def test_force_skips_the_menu(self):
        interface = _BareMadGraphCmd(use_rawinput=True)
        interface.force = True
        interface._answer = 'should not be asked'
        self.assertEqual(interface.ask_tutorial(), 'lo')
        self.assertIsNone(interface.asked)

    def test_a_number_picks_that_tutorial(self):
        """Numbers follow the order the menu prints, which is by section --
        not the order the registry happens to hold."""

        listed = [tutorial for _key, _title, group, _notice
                  in tutorials.by_section() for tutorial in group]
        for index, tutorial in enumerate(listed):
            interface = _BareMadGraphCmd()
            interface._answer = str(index + 1)
            self.assertEqual(self._ask_interactively(interface), tutorial.name)

    def _ask_interactively(self, interface):
        """Run ask_tutorial as though stdin were a terminal."""

        import sys as _sys

        class _Tty(object):
            def __getattr__(self, name):
                return getattr(_sys.__stdin__, name)

            def isatty(self):
                return True

        interface.use_rawinput = True
        saved = _sys.stdin
        _sys.stdin = _Tty()
        try:
            return interface.ask_tutorial()
        finally:
            _sys.stdin = saved

    def test_a_name_is_accepted_too(self):
        interface = _BareMadGraphCmd()
        interface._answer = 'syntax'
        self.assertEqual(self._ask_interactively(interface), 'syntax')

    def test_the_menu_offers_names_numbers_and_stop(self):
        interface = _BareMadGraphCmd()
        interface._answer = 'lo'
        self._ask_interactively(interface)
        _question, default, choices = interface.asked
        self.assertEqual(default, 'lo')
        for name in tutorials.names():
            self.assertIn(name, choices)
        self.assertIn('1', choices)
        self.assertIn('stop', choices)

    def test_the_menu_never_times_out(self):
        """A tutorial menu waits indefinitely.

        MG7 times most questions out so an unattended script cannot hang, but
        the menu only ever appears when there is a person at the keyboard --
        this path is skipped outright without a tty -- so a timer there just
        picks a tutorial for someone who went to make a coffee. timeout=0
        means no limit, and also drops the '[Ns to answer]' suffix.
        """

        interface = _BareMadGraphCmd()
        interface._answer = 'lo'
        self._ask_interactively(interface)
        self.assertEqual(interface.ask_opts.get('timeout'), 0,
                         'the tutorial menu asks with a timeout')

    def test_aliases_are_normalised_to_the_primary_name(self):
        for old, new in (('MadGraph5', 'lo'), ('aMCatNLO', 'nlo'),
                         ('MadLoop', 'madloop')):
            args = [old]
            _BareMadGraphCmd().check_tutorial(args)
            self.assertEqual(args, [new])

    def test_subcommands_pass_through(self):
        for name in ('stop', 'list', 'status', 'help'):
            args = [name]
            _BareMadGraphCmd().check_tutorial(args)
            self.assertEqual(args, [name])

    def test_an_unknown_name_is_refused(self):
        interface = _BareMadGraphCmd()
        self.assertRaises(madgraph.InvalidCmd,
                          interface.check_tutorial, ['not-a-tutorial'])

    def test_too_many_arguments_is_refused(self):
        interface = _BareMadGraphCmd()
        self.assertRaises(madgraph.InvalidCmd,
                          interface.check_tutorial, ['lo', 'syntax'])


#===============================================================================
# exercises: every solution passes, every diagnosis fires
#===============================================================================

class TestExercises(unittest.TestCase):
    """Drive the exercises through a real interface.

    This is the point of the exercise design: an exercise that stops working --
    because the syntax moved, or a check got too strict -- fails here rather
    than in front of a user.
    """

    def setUp(self):
        # a fresh interface per test: the NLO exercise switches the interface
        # to aMC@NLO and leaves _fks_multi_proc set, which a shared instance
        # would carry into the next test
        import madgraph.interface.master_interface as master
        self.interface = master.MasterCmd()
        self.interface.exec_cmd('import model sm', printcmd=False)

    def run_line(self, line):
        """Run a command and hand back the interface it left behind."""

        self.interface.exec_cmd(line, printcmd=False, precmd=True)
        return self.interface

    def exercises(self):
        from madgraph.interface.tutorials.session import Exercise
        return [step for step in tutorials.get('exercises').steps
                if isinstance(step, Exercise)]

    def test_every_solution_passes_its_own_check(self):
        for exercise in self.exercises():
            interface = self.run_line(exercise.solution)
            passed, message = exercise.evaluate(interface, exercise.solution)
            self.assertTrue(passed,
                            'exercise %r rejects its own solution %r: %s'
                            % (exercise.title, exercise.solution, message))

    def test_every_exercise_has_a_hint_and_a_solution(self):
        for exercise in self.exercises():
            self.assertTrue(exercise.hint,
                            'exercise %r has no hint' % exercise.title)
            self.assertTrue(exercise.solution,
                            'exercise %r has no solution' % exercise.title)

    def test_the_named_mistakes_are_reachable(self):
        """Feed each exercise the wrong answer it names, and check the right
        diagnosis comes back.

        A mistake entry that can no longer be triggered means the syntax moved
        under us, and this says so.
        """

        # (exercise title, the wrong command, a phrase the diagnosis must carry)
        wrong_answers = [
            ('letting the EW diagrams in', 'generate p p > t t~ QED=0',
             'goes the wrong way'),
            ('interference only', 'generate p p > j j QED=2 QCD=2',
             'constrains the *amplitude*'),
            ('interference only', 'generate p p > j j QCD^2=2 QED^2=2',
             'a bare `=` means `<=`'),
            ('decay chains', 'generate p p > t t~ > w+ b w- b~',
             'The separator is a **comma**'),
            ('forbidding a particle', 'generate p p > e+ e- $ a',
             'To forbid the photon *anywhere*'),
            ('excluding a resonance', 'generate p p > e+ e- / z',
             'forbids the Z everywhere'),
            ('going to NLO', 'generate p p > t t~',
             'still a leading-order process'),
        ]
        by_title = dict((exercise.title, exercise)
                        for exercise in self.exercises())

        for title, wrong, expected in wrong_answers:
            exercise = by_title[title]
            interface = self.run_line(wrong)
            passed, message = exercise.evaluate(interface, wrong)
            self.assertFalse(
                passed, 'exercise %r accepted the wrong answer %r'
                % (title, wrong))
            self.assertIn(
                expected, message,
                'exercise %r did not diagnose %r -- it said:\n%s'
                % (title, wrong, message))

    def test_the_fallback_says_what_the_command_produced(self):
        """An unanticipated wrong answer still gets told what it did."""

        by_title = dict((exercise.title, exercise)
                        for exercise in self.exercises())
        # a 2 -> 2 process that is not t t~: neither named mistake matches
        # (`p p > z` would, since it is not 2 -> 2), so this reaches the
        # fallback, which is what needs to be readable
        exercise = by_title['top pair production']
        interface = self.run_line('generate p p > z z')
        passed, message = exercise.evaluate(interface, 'generate p p > z z')
        self.assertFalse(passed)
        self.assertIn('What your command produced', message)
        self.assertIn('diagrams', message)


#===============================================================================
# only the user's own commands drive a tutorial
#===============================================================================

class TestOnlyUserCommandsCount(_TutorialTestCase):
    """MG5 runs commands for itself, and those must not move a tutorial.

    `display diagrams` issues an `open`, importing a model issues half a dozen
    `define`s. Before the depth guard those fired tutorial steps -- and in a
    sequenced tutorial an internal `open` matching a later step skipped the
    user to the end.
    """

    def _fire(self, depth, line='display diagrams'):
        interface = _BareInterface()
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        interface.exec_cmd_depth = depth
        self.capture.messages = []
        interface.postcmd(True, line)
        return self.capture.messages

    def test_a_user_command_fires(self):
        """Depth 0 is where a user command sits, typed or from a file."""

        self.assertTrue(self._fire(0, 'generate p p > t t~'),
                        'a user command did not reach the tutorial')

    def test_a_command_mg5_issued_does_not(self):
        """Depth 1 is MG5 talking to itself -- the interactive case, which the
        first version of the guard let through."""

        self.assertEqual(self._fire(1, 'generate p p > t t~'), [])
        self.assertEqual(self._fire(2, 'generate p p > t t~'), [])

    def test_display_diagrams_does_not_skip_to_the_end(self):
        """The reported bug: `display diagrams` runs `open` underneath, and
        `open` matches the last step of `lo`."""

        interface = _BareInterface()
        session = tutorials.start('lo')
        tutorial_mixin.attach(interface, session)

        interface.exec_cmd_depth = 0
        interface.postcmd(True, 'generate p p > t t~')
        reached = session.index

        # the `open` display issues for the .eps, one level down
        interface.exec_cmd_depth = 1
        interface.postcmd(True, 'open /tmp/whatever.eps')
        self.assertEqual(session.index, reached,
                         'an internal `open` moved the tutorial')

        interface.exec_cmd_depth = 0
        interface.postcmd(True, 'display diagrams')
        self.assertLess(session.index, len(session.tutorial.steps) - 1,
                        '`display diagrams` skipped to the last step')
        self.assertEqual(session.current.title, 'look at the diagrams')


#===============================================================================
# lessons describe what actually happened
#===============================================================================

class TestAppliedOrders(unittest.TestCase):
    """The coupling-order lesson must report the orders MG5 really applied.

    MG5 has more than one way of settling this, so the tutorial reads the
    process back rather than asserting what usually happens.
    """

    def _fake(self, orders):
        from madgraph.core import base_objects, diagram_generation

        process = base_objects.Process()
        process.set('orders', dict(orders))
        amplitude = diagram_generation.Amplitude()
        amplitude.set('process', process)

        class _Interface(object):
            _curr_amps = [amplitude]

        return _Interface()

    def test_reports_a_weighted_search(self):
        from madgraph.interface.tutorials.session import describe_applied_orders

        text = describe_applied_orders(self._fake({'WEIGHTED': 2}))
        # quoted the way MG5 prints it: an amplitude order means '<='
        self.assertIn('WEIGHTED<=2', text)
        self.assertIn('QCD + 2*QED', text)

    def test_reports_an_explicit_order(self):
        from madgraph.interface.tutorials.session import describe_applied_orders

        text = describe_applied_orders(self._fake({'QED': 0}))
        self.assertIn('QED<=0', text)
        self.assertNotIn('WEIGHTED', text)

    def test_reports_no_constraint(self):
        from madgraph.interface.tutorials.session import describe_applied_orders

        self.assertIn('no coupling-order constraint',
                      describe_applied_orders(self._fake({})))

    def test_no_tutorial_hardcodes_the_choice(self):
        """Neither lesson may assert which mechanism MG5 used."""

        class _Empty(object):
            _curr_amps = []

        for name in ('lo', 'syntax'):
            for step in tutorials.get(name).steps:
                text = step.render(_Empty())
                self.assertNotIn('it added `QED=0` on its own', text)
                self.assertNotIn('Trying coupling order WEIGHTED', text)


#===============================================================================
# `tutorial help`
#===============================================================================

class TestTutorialHelp(unittest.TestCase):
    """`tutorial help` explains the commands a running tutorial understands."""

    class _Recorder(mg_interface.MadGraphCmd):
        def __init__(self):
            self.use_rawinput = False
            self.lines = []

        def _record(self, message, *args):
            self.lines.append(message)

    def setUp(self):
        self.interface = self._Recorder()
        self._saved = mg_interface.logger.info
        mg_interface.logger.info = self.interface._record

    def tearDown(self):
        mg_interface.logger.info = self._saved

    def text(self):
        return '\n'.join(self.interface.lines)

    def test_it_names_every_in_tutorial_command(self):
        self.interface.print_tutorial_help()
        for command in ('hint', 'solution', 'next', 'repeat', 'back', 'skip'):
            self.assertIn(command, self.text(),
                          '`tutorial help` does not mention %r' % command)

    def test_it_names_every_subcommand(self):
        self.interface.print_tutorial_help()
        for command in ('tutorial list', 'tutorial status', 'tutorial help',
                        'tutorial stop'):
            self.assertIn(command, self.text())

    def test_it_says_when_nothing_is_running(self):
        self.interface.print_tutorial_help()
        self.assertIn('No tutorial is running', self.text())

    def test_it_reports_progress_when_one_is(self):
        self.interface._tutorial_session = tutorials.start('lo')
        self.interface.print_tutorial_session_progress = None
        self.interface.print_tutorial_help()
        self.assertIn("Running 'lo'", self.text())

    def test_help_is_an_accepted_argument(self):
        args = ['help']
        _BareMadGraphCmd().check_tutorial(args)
        self.assertEqual(args, ['help'])

    def test_the_in_tutorial_commands_all_exist(self):
        """Every command `tutorial help` advertises is really implemented."""

        from madgraph.interface.tutorials import mixin as tutorial_mixin_mod

        provided = set(tutorial_mixin_mod.mixin_command_names())
        for command in ('hint', 'solution', 'next', 'repeat', 'back', 'skip'):
            self.assertIn('do_%s' % command, provided,
                          '`tutorial help` advertises %r, which the mixin does '
                          'not provide' % command)


#===============================================================================
# the two uncertainties
#===============================================================================

class TestUncertaintyLesson(unittest.TestCase):
    """`lo` must distinguish the statistical and theoretical uncertainties, and
    say when the theory one will be missing."""

    class _WithLhapdf(object):
        # lhapdf_configured requires an *executable*, so a readable file will
        # not do; the interpreter running the tests always is one
        options = {'lhapdf': sys.executable}
        _done_export = ['/tmp/x/MYPROC', 'mg7']

    class _WithoutLhapdf(object):
        options = {'lhapdf': '/nonexistent/lhapdf-config'}
        _done_export = ['/tmp/x/MYPROC', 'mg7']

    def step(self):
        return [s for s in tutorials.get('lo').steps if s.title == 'run it'][0]

    def test_it_names_both_uncertainties(self):
        text = self.step().render(self._WithLhapdf())
        self.assertIn('statistical', text)
        self.assertIn('theoretical', text)
        self.assertIn('Scale variation', text)
        self.assertIn('PDF variation', text)

    def test_it_shows_the_notation_mg7_actually_uses(self):
        """madspace prints the error in the last digits (format.cpp), not with
        a +-. The lesson quotes it as-is: the notation is familiar to MadGraph
        users and does not need explaining."""

        text = self.step().render(self._WithLhapdf())
        self.assertIn('503.1(1.4)', text)

    def test_without_lhapdf_it_says_the_pdf_row_is_missing(self):
        """Since PR #89 madspace evaluates the PDF members itself, so the
        scale row survives without LHAPDF -- only the PDF one goes."""

        text = self.step().render(self._WithoutLhapdf())
        self.assertIn('Expect no PDF row', text)
        self.assertIn('install lhapdf6', text)
        self.assertIn('scale row and the cross section are unaffected', text)

    def test_with_lhapdf_it_says_nothing(self):
        """Nothing to warn about is nothing to say: the caveat is absent
        rather than reassuring."""

        text = self.step().render(self._WithLhapdf())
        self.assertNotIn('PDF row', text)
        self.assertNotIn('lhapdf', text.lower())

    def test_lhapdf_detection(self):
        from madgraph.interface.tutorials.session import lhapdf_configured

        self.assertTrue(lhapdf_configured(self._WithLhapdf()))
        self.assertFalse(lhapdf_configured(self._WithoutLhapdf()))


#===============================================================================
# a tutorial is a place to make mistakes
#===============================================================================

class TestCrashOnErrorSuspended(unittest.TestCase):
    """crash_on_error tears the session down on a mistyped command, which is
    exactly wrong while someone is learning. It is suspended for the life of
    the tutorial and put back on stop."""

    class _Interface(mg_interface.CmdExtended):
        def __init__(self, **options):
            self.options = dict(options)

    def test_it_is_off_while_a_tutorial_runs(self):
        interface = self._Interface(crash_on_error=True)
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        self.assertFalse(interface.options['crash_on_error'])

    def test_the_users_setting_comes_back(self):
        interface = self._Interface(crash_on_error=True)
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        tutorial_mixin.detach(interface)
        self.assertTrue(interface.options['crash_on_error'])

    def test_other_options_changed_meanwhile_are_kept(self):
        """The guard addresses one key, so a `set` during the tutorial sticks."""

        interface = self._Interface(crash_on_error=True, stdout_level='INFO')
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        interface.options['stdout_level'] = 'DEBUG'
        tutorial_mixin.detach(interface)
        self.assertEqual(interface.options['stdout_level'], 'DEBUG')
        self.assertTrue(interface.options['crash_on_error'])

    def test_switching_tutorial_does_not_lose_the_setting(self):
        interface = self._Interface(crash_on_error=True)
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        tutorial_mixin.attach(interface, tutorials.start('syntax'))
        self.assertFalse(interface.options['crash_on_error'])
        tutorial_mixin.detach(interface)
        self.assertTrue(interface.options['crash_on_error'])

    def test_an_interface_without_the_option_is_fine(self):
        interface = self._Interface()
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        tutorial_mixin.detach(interface)          # must not raise

    def test_already_off_stays_off(self):
        interface = self._Interface(crash_on_error=False)
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        tutorial_mixin.detach(interface)
        self.assertFalse(interface.options['crash_on_error'])


class TestTutorialStepCommandAliases(unittest.TestCase):
    """`tutorial hint` is as natural to type as `hint`, and used to be an
    InvalidCmd -- which, with crash_on_error set, killed the session."""

    def test_they_are_accepted_arguments(self):
        for name in ('hint', 'solution', 'next', 'repeat', 'back', 'skip'):
            args = [name]
            _BareMadGraphCmd().check_tutorial(args)
            self.assertEqual(args, [name])

    def test_each_alias_has_a_command_behind_it(self):
        provided = set(tutorial_mixin.mixin_command_names())
        for name in mg_interface.MadGraphCmd._tutorial_step_cmds:
            self.assertIn('do_%s' % name, provided,
                          '`tutorial %s` is accepted but nothing implements it'
                          % name)


#===============================================================================
# the optional detour in `lo`
#===============================================================================

class TestLoDetour(_TutorialTestCase):
    """`lo` offers an optional side-trip: regenerate allowing QED vertices and
    be told what changed. Taking it must report the real counts; skipping it
    must land on the next step as if it were not there."""

    def steps(self):
        return tutorials.get('lo').steps

    def index_of(self, title):
        for i, step in enumerate(self.steps()):
            if step.title == title:
                return i
        self.fail('no step titled %r' % title)

    def test_skipping_it_reaches_the_next_step(self):
        session = TutorialSession(tutorials.get('lo'))
        offered_at = self.index_of('generate a process')
        session.index = offered_at
        found = session.step_for('display diagrams')
        self.assertIsNotNone(found)
        self.assertEqual(found[1].title, 'look at the diagrams',
                         'skipping the detour did not reach the main line')

    def test_taking_it_reaches_the_detour(self):
        session = TutorialSession(tutorials.get('lo'))
        session.index = self.index_of('generate a process')
        found = session.step_for('generate p p > t t~ QED<=2')
        self.assertIsNotNone(found)
        self.assertEqual(found[1].title, 'the electroweak diagrams (detour)')

    def test_the_detour_leads_back_to_the_main_line(self):
        session = TutorialSession(tutorials.get('lo'))
        detour = self.index_of('the electroweak diagrams (detour)')
        session.index = detour
        found = session.step_for(self.steps()[detour].get_solution())
        self.assertIsNotNone(found)
        self.assertEqual(found[1].title, 'look at the diagrams')

    def test_it_reports_the_counts_it_was_given(self):
        """The before/after numbers come from the session, not from the text."""

        import madgraph.interface.tutorials.lo as lo_module

        class _Interface(object):
            _tutorial_lo_ndiag = 4
            _curr_amps = []

        interface = _Interface()
        original = lo_module.total_diagrams
        try:
            lo_module.total_diagrams = lambda i: 6
            text = lo_module._detour_text(interface)
        finally:
            lo_module.total_diagrams = original

        self.assertIn('**4** diagrams', text)
        self.assertIn('**6**', text)

    def test_it_copes_with_no_remembered_count(self):
        """`repeat` on the detour after a restart must not blow up."""

        import madgraph.interface.tutorials.lo as lo_module

        class _Interface(object):
            _curr_amps = []

        text = lo_module._detour_text(_Interface())
        self.assertIn('display diagrams', text)

    def test_the_offer_names_the_command(self):
        step = self.steps()[self.index_of('generate a process')]

        class _Empty(object):
            _curr_amps = []

        self.assertIn('QED<=2', step.render(_Empty()))


#===============================================================================
# what `lo` says about the launch question
#===============================================================================

class TestLaunchQuestion(unittest.TestCase):
    """The card question is explained after the fact, not before.

    An mg7 `launch` now runs in process (PR #131), so the tutorial logger does
    reach the question -- the "Need help here? type \'help\'" block appears
    there as it does for NLO. The explanation still belongs afterwards though:
    before the fact it is a wall of text about a screen the reader has not
    seen."""

    class _Interface(object):
        _done_export = ['/tmp/x/MYPROC', 'mg7']
        options = {'lhapdf': sys.executable}
        _curr_amps = []

    def step(self, title):
        return [s for s in tutorials.get('lo').steps if s.title == title][0]

    def before(self):
        return self.step('produce the output').render(self._Interface())

    def after(self):
        return self.step('run it').render(self._Interface())

    def test_the_preamble_stays_short(self):
        """The whole point: it used to describe the question at length before
        the reader had any use for it."""

        self.assertLess(len(self.before().strip().split('\n')), 30)

    def test_the_preamble_says_nothing_about_the_question(self):
        """The step no longer previews the card question at all: it is in
        process, so the tutorial is at the prompt with the user, and the
        explanation follows once they have seen it."""

        text = self.before()
        for detail in ('Not Avail.', 'set KEY VALUE', 'param_card.dat',
                       'run_card.toml'):
            self.assertNotIn(detail, text)

    def test_the_explanation_is_at_the_question(self):
        """It used to be recapped in the step after the run. Now that the
        tutorial can speak at the prompt (PR #131), it belongs there -- and
        the step after does not repeat it."""

        step = self.step('produce the output')
        self.assertTrue(step.question_hint)
        hint = step.question_hint
        if callable(hint):
            # it names the editor a card will open in, so it is worked out
            # when the question is asked
            hint = hint(None)
        for detail in ('param_card.dat', 'run_card.toml'):
            self.assertIn(detail, hint)
        self.assertNotIn('question you just answered', self.after())

    def test_it_does_not_reprint_the_question(self):
        """The reader has just seen it; repeating the switch list is noise."""

        text = self.after()
        for row in ('1   shower=', '6. param', 'The following switches'):
            self.assertNotIn(row, text)

    def test_it_does_not_claim_the_tutorial_stops(self):
        """It used to say launch handed over to a separate program. Since
        PR #131 that is false: the run is in this process."""

        text = self.before()
        self.assertNotIn('goes quiet', text)
        self.assertNotIn('separate program', text)

    def test_it_offers_launch_with_and_without_an_argument(self):
        """Bare `launch` takes _done_export -- see check_launch."""

        text = self.before()
        self.assertIn('launch\n', text)
        self.assertIn('takes the output you just made', text)


#===============================================================================
# the NLO shower guidance
#===============================================================================

class TestNloShowerGuidance(unittest.TestCase):
    """`nlo` must tell the reader what to do about the shower on *their*
    machine, before they type launch."""

    class _WithPy8(object):
        options = {'pythia8_path': '/somewhere'}

    class _WithoutPy8(object):
        options = {'pythia8_path': None}

    def step(self):
        return [s for s in tutorials.get('nlo').steps
                if s.title == 'produce an output'][0]

    def test_without_py8_it_says_how_to_install(self):
        text = self.step().render(self._WithoutPy8())
        self.assertIn('install pythia8', text)
        self.assertNotIn('install mg5amc_py8_interface', text)

    def test_without_py8_it_offers_parton_level_but_calls_it_unphysical(self):
        text = self.step().render(self._WithoutPy8())
        self.assertIn('launch -p', text)
        self.assertIn('UNPHYSICAL', text)
        self.assertIn('never to get a number', text)

    def test_with_py8_it_does_not_nag(self):
        text = self.step().render(self._WithPy8())
        self.assertNotIn('install pythia8', text)
        self.assertIn('has installed', text)

    def test_pythia8_path_is_the_requirement(self):
        """Pythia8 alone is enough: aMC@NLO builds its own Pythia8 driver."""

        from madgraph.interface.tutorials.session import pythia8_available

        self.assertTrue(pythia8_available(self._WithPy8()))
        self.assertFalse(pythia8_available(self._WithoutPy8()))

    def test_it_warns_off_herwig6(self):
        for interface in (self._WithPy8(), self._WithoutPy8()):
            self.assertIn('HERWIG6', self.step().render(interface))

    def test_the_ported_text_is_still_there(self):
        """Appending must not have dropped any of the original."""

        import madgraph.interface.tutorial_text_nlo as legacy

        text = self.step().render(self._WithPy8())
        self.assertIn(retarget(legacy.output), text)


#===============================================================================
# terminal styling
#===============================================================================

class TestTerminalStyling(unittest.TestCase):
    """Tutorials are authored with **emphasis** and `code` because that reads
    well in the source. A terminal shows those literally, so they are
    translated on the way out."""

    def convert(self, text):
        from madgraph.interface.tutorials._style import to_terminal

        return to_terminal(text)

    def test_bold_and_code_become_formatter_markers(self):
        self.assertEqual(self.convert('**loud**'), '$_BOLDloud$_RESET')
        self.assertEqual(self.convert('`cmd`'), '$GREENcmd$_RESET')

    def test_italic_becomes_underline(self):
        """There is no MG5 marker for italic, so the escape goes in direct."""

        self.assertEqual(self.convert('*soft*'), '\033[4msoft\033[0m')

    def test_a_star_inside_a_word_is_left_alone(self):
        self.assertEqual(self.convert('a*b*c'), 'a*b*c')

    def test_it_uses_the_unconditional_markers(self):
        """ColorFormatter drops $BOLD/$RESET/$COLOR for an INFO record with no
        colour argument, and always substitutes the underscored ones."""

        converted = self.convert('**a** and `b`')
        self.assertIn('$_BOLD', converted)
        self.assertIn('$_RESET', converted)
        self.assertNotIn('$BOLD', converted.replace('$_BOLD', ''))

    def test_a_bullet_is_not_emphasis(self):
        text = '  * a point\n  * another'
        self.assertEqual(self.convert(text), text)

    def test_the_dollar_syntax_survives(self):
        """`$` is MG5 process syntax for a forbidden s-channel, and the syntax
        tutorial is full of it."""

        for text in ('p p > e+ e- $ z', 'p p > e+ e- $$ z', '$ vs $$ vs /'):
            self.assertEqual(self.convert(text), text)

    def test_no_tutorial_emits_raw_markup(self):
        """Nothing a user sees should still carry asterisks or backticks."""

        class _Empty(object):
            _curr_amps = []
            options = {}

        for tutorial in tutorials.all_tutorials(include_hidden=True):
            for step in tutorial.steps:
                shown = self.convert(step.render(_Empty()))
                self.assertNotIn('**', shown,
                                 '%s / %s still shows ** in the terminal'
                                 % (tutorial.name, step.title))
                self.assertNotIn('`', shown,
                                 '%s / %s still shows a backtick in the terminal'
                                 % (tutorial.name, step.title))
                # a leftover single-* emphasis span
                self.assertIsNone(
                    re.search(r'(?<![\w*])\*(?=\S)[^*\n]+?(?<=\S)\*(?![\w*])',
                              shown),
                    '%s / %s still shows *emphasis* in the terminal'
                    % (tutorial.name, step.title))

    def test_no_span_wraps_a_line(self):
        """A span that wrapped would colour the next line's indentation, and
        the line-local regexes would miss it entirely."""

        import ast
        import madgraph.interface.tutorials as package

        # Only the tutorial text: scan the string literals rather than the
        # source lines, or `x ** 2` in the module's own code reads as an
        # unbalanced emphasis marker. Only the content modules, too -- the
        # engine's docstrings talk *about* the markup, backticks and all.
        for name in package._MODULES:
            module = sys.modules.get(
                'madgraph.interface.tutorials.%s' % name)
            if module is None:
                continue
            path = module.__file__.replace('.pyc', '.py')
            with open(path) as handle:
                tree = ast.parse(handle.read())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) \
                        or not isinstance(node.value, str):
                    continue
                for offset, line in enumerate(node.value.split('\n')):
                    where = '%s, string at line %s (+%d)' % (
                        path, getattr(node, 'lineno', '?'), offset)
                    self.assertEqual(line.count('`') % 2, 0,
                                     'unbalanced backtick in %s' % where)
                    self.assertEqual(line.count('**') % 2, 0,
                                     'unbalanced ** in %s' % where)

    def test_no_tutorial_text_collides_with_the_formatter(self):
        """A "$" followed by a formatter keyword would be eaten silently."""

        from madgraph.interface.tutorials._style import has_formatter_keyword

        class _Empty(object):
            _curr_amps = []
            options = {}

        for tutorial in tutorials.all_tutorials(include_hidden=True):
            for step in tutorial.steps:
                self.assertFalse(
                    has_formatter_keyword(step.render(_Empty())),
                    '%s / %s contains a $KEYWORD the formatter would swallow'
                    % (tutorial.name, step.title))


#===============================================================================
# the process lines a tutorial proposes have to exist in the model it runs on
#===============================================================================

class TestProposedProcessesAreValid(unittest.TestCase):
    """Every particle a tutorial tells the user to type must be in its model.

    `syntax` and `lo` run on the plain SM, so a process line quoting a charged
    Higgs -- as the alternative-s-channel lesson once did -- is a command the
    reader cannot run.  Only the particle names are checked here: the rest of
    the line (orders, `[QCD]`, decay chains) is the tutorial's own subject and
    is exercised by the exercises tutorial.
    """

    SM_TUTORIALS = ('syntax', 'lo')

    # a proposed command: after the prompt, or indented in an example block
    COMMAND = re.compile(
        r'^\s*(?:MG7>\s*)?(?:generate|add process|check +\w+)\s+(.*)$')

    @classmethod
    def setUpClass(cls):
        import madgraph.interface.master_interface as master
        cls.interface = master.MasterCmd()
        cls.interface.exec_cmd('import model sm', printcmd=False)

    def proposals(self, tutorial):
        """(step title, command) for every process line the steps quote."""

        out = []
        for step in tutorial.steps:
            for line in (step.render(self.interface) or '').splitlines():
                found = self.COMMAND.match(line)
                if found:
                    out.append((step.title, found.group(1).strip()))
            solution = step.get_solution()
            found = solution and self.COMMAND.match(solution)
            if found:
                out.append((step.title, found.group(1).strip()))
        return out

    @staticmethod
    def particle_names(process):
        """The particle names in a process line, orders and operators dropped."""

        # the example blocks align a comment after the command, two spaces or
        # more from it: 'generate p p > t t~     initial state, `>`, ...'
        process = re.split(r'\s{2,}', process)[0]
        process = process.split('[')[0]                  # [QCD] and friends
        process = re.sub(r'\{[^}]*\}', '', process)       # {0}, {T} ...
        process = re.sub(r'[(),]', ' ', process)          # decay chains
        names = []
        for token in process.split():
            if token in ('>', '|', '/', '$', '$$') or token.startswith('@'):
                continue
            if '=' in token or '^' in token:             # a coupling order
                continue
            names.append(token)
        return names

    def test_every_proposed_process_is_in_the_sm(self):
        for name in self.SM_TUTORIALS:
            tutorial = tutorials.get(name)
            self.assertIsNotNone(tutorial, '%s is not registered' % name)
            for title, process in self.proposals(tutorial):
                for particle in self.particle_names(process):
                    try:
                        self.interface.extract_particle_ids([particle])
                    except madgraph.InvalidCmd:
                        self.fail('%s / %s proposes %r, but %r is not in the '
                                  'SM' % (name, title, process, particle))

    def test_the_check_would_have_caught_the_charged_higgs(self):
        """The guard is only worth having if it fires -- this is the line that
        used to sit in the alternative-s-channel lesson."""

        names = self.particle_names('b b~ > W+ W- | H+ H- > ta+ vt ta- vt~')
        self.assertIn('H+', names)
        self.assertRaises(madgraph.InvalidCmd,
                          self.interface.extract_particle_ids, ['H+'])


#===============================================================================
# menu sections and provenance
#===============================================================================

class TestMenuSections(unittest.TestCase):
    """The menu is grouped, and the AI-generated notice has to be true of
    every tutorial it covers."""

    def sections(self):
        return list(tutorials.by_section())

    def named(self, key):
        for section_key, _title, group, notice in self.sections():
            if section_key == key:
                return [t.name for t in group], notice
        self.fail('no %s section' % key)

    def test_the_order_is_basic_advanced_more_exercises(self):
        keys = [key for key, _title, _group, _notice in self.sections()]
        self.assertEqual(keys, ['basic', 'advanced', 'more', 'exercises'])

    def test_basic_is_lo_and_nlo(self):
        self.assertEqual(self.named('basic')[0], ['lo', 'nlo'])

    def test_advanced_is_the_validated_ones(self):
        """'advanced' is the developer-validated group, not 'everything that
        is not basic' -- which is what it used to collect by default."""

        self.assertEqual(sorted(self.named('advanced')[0]),
                         ['bsm', 'decays', 'madloop', 'model', 'syntax'])

    def test_exercises_is_its_own_section(self):
        self.assertIn('exercises', self.named('exercises')[0])

    def test_basic_and_advanced_say_they_are_validated(self):
        for key in ('basic', 'advanced'):
            notice = self.named(key)[1]
            self.assertEqual(notice, tutorials.VALIDATED_NOTICE,
                             '%s does not claim validation' % key)

    def test_more_and_exercises_carry_the_ai_notice(self):
        for key in ('more', 'exercises'):
            notice = self.named(key)[1]
            self.assertIn('not yet validated', notice,
                          '%s is not flagged' % key)

    def test_every_tutorial_in_a_validated_section_is_validated(self):
        for key, title, group, _notice in self.sections():
            if key not in Tutorial.VALIDATED_SECTIONS:
                continue
            for tutorial in group:
                self.assertFalse(tutorial.ai_generated,
                                 '%s is AI-generated but sits under %s'
                                 % (tutorial.name, title))

    def test_the_ported_tutorials_are_not_called_ai_generated(self):
        """nlo and madloop are the pre-2026 text, near verbatim."""

        for name in ('nlo', 'madloop'):
            self.assertFalse(tutorials.get(name).ai_generated,
                             '%s is the original text, not AI-generated' % name)

    def test_an_unknown_section_is_refused(self):
        self.assertRaises(ValueError, Tutorial, name='x', title='x',
                          steps=[], section='nonsense')

    def test_an_ai_tutorial_cannot_claim_a_validated_section(self):
        """The heading promises validation, so the promise is enforced here
        rather than left to whoever adds the next tutorial."""

        for key in Tutorial.VALIDATED_SECTIONS:
            self.assertRaises(ValueError, Tutorial, name='x', title='x',
                              steps=[], section=key, ai_generated=True)
            Tutorial(name='x', title='x', steps=[], section=key,
                     ai_generated=False)

    def test_the_default_section_is_not_a_validated_one(self):
        """A new tutorial has to opt in to the validated groups."""

        self.assertNotIn(Tutorial(name='x', title='x', steps=[]).section,
                         Tutorial.VALIDATED_SECTIONS)


#===============================================================================
# what a tutorial says at a question
#===============================================================================

class TestQuestionHooks(_TutorialTestCase):
    """A question is often asked by an object the mixin is not attached to --
    the launch card question belongs to the run interface -- so the tutorial
    reaches it through module-level switches in extended_cmd."""

    def setUp(self):
        _TutorialTestCase.setUp(self)
        import madgraph.interface.extended_cmd as extended_cmd

        self.extended_cmd = extended_cmd

    def attach(self, name='lo'):
        interface = _BareInterface()
        session = tutorials.start(name)
        tutorial_mixin.attach(interface, session)
        return interface, session

    def test_the_generic_line_is_the_default(self):
        self.assertEqual(self.extended_cmd.get_question_hint(),
                         "Need help here? type 'help'")

    def test_a_step_can_speak_at_a_question(self):
        _interface, session = self.attach()
        for index, step in enumerate(session.tutorial.steps):
            if step.title == 'produce the output':
                session.index = index
        hint = self.extended_cmd.get_question_hint()
        self.assertIn('param_card.dat', hint)
        self.assertIn('run_card.toml', hint)
        self.assertIn('just press Enter', hint)

    def test_the_hint_is_styled_like_the_rest(self):
        """It goes through the same markup conversion as step text."""

        _interface, session = self.attach()
        for index, step in enumerate(session.tutorial.steps):
            if step.title == 'produce the output':
                session.index = index
        hint = self.extended_cmd.get_question_hint()
        self.assertNotIn('**', hint)
        self.assertNotIn('`', hint)

    def test_a_step_without_one_keeps_the_generic_line(self):
        _interface, session = self.attach()
        session.index = 0            # the intro has no question_hint
        self.assertEqual(self.extended_cmd.get_question_hint(),
                         "Need help here? type 'help'")

    def test_questions_do_not_time_out_during_a_tutorial(self):
        self.assertFalse(self.extended_cmd.suppress_timeout)
        interface, _session = self.attach()
        self.assertTrue(self.extended_cmd.suppress_timeout)
        tutorial_mixin.detach(interface)
        self.assertFalse(self.extended_cmd.suppress_timeout)

    def test_stopping_puts_the_generic_line_back(self):
        interface, _session = self.attach()
        tutorial_mixin.detach(interface)
        self.assertIsNone(self.extended_cmd.question_hint)
        self.assertEqual(self.extended_cmd.get_question_hint(),
                         "Need help here? type 'help'")

    def test_a_broken_hint_does_not_break_the_question(self):
        self.extended_cmd.question_hint = lambda: 1 / 0
        self.assertEqual(self.extended_cmd.get_question_hint(),
                         "Need help here? type 'help'")


#===============================================================================
# the post-run step shows the reader's own numbers
#===============================================================================

class TestRealRunNumbers(unittest.TestCase):
    """The uncertainties step reads the run's info.json rather than quoting
    invented values, the way the step before it reads the directory name."""

    # taken from a real `p p > t t~` run, which printed
    #   Result: 380.57(28)
    #   Scale variation: +26.6%  -19.8%
    #   PDF variation:   +1.96%  -1.96%
    INFO = {
        'channels': [{'mean': 300.0, 'error': 0.2},
                     {'mean': 80.56585501289203, 'error': 0.19442222}],
        'systematics': {
            'nominal': {'cross_section': 380.56585501289857},
            'event_count': 100000,
            'scale': {'min': 305.402009221479, 'max': 481.73208634756236},
            'pdf': [{'pdf_set': 'NNPDF40_lo_as_01180',
                     'error_type': 'replicas',
                     'central': 380.799971376436,
                     'uncertainty_up': 7.450190796275929,
                     'uncertainty_down': 7.450190796275929}],
        },
    }

    def module(self):
        import madgraph.interface.tutorials.lo as lo_module

        return lo_module

    def test_the_result_row_matches_what_the_run_printed(self):
        """mean is the channel means summed, error their quadrature sum --
        the way the run builds it. 380.5658 +- 0.2789 -> 380.57(28)."""

        row = self.module()._result_row(self.INFO)
        self.assertEqual(row, '380.57(28)')

    def test_the_systematics_rows_match(self):
        rows = self.module()._systematics_rows(self.INFO)
        self.assertIn('+26.6%', rows)
        self.assertIn('-19.8%', rows)
        self.assertIn('+1.96%', rows)

    def test_the_percentages_come_from_the_shared_helper(self):
        """Not a third implementation: the run's own box and the scan summary
        read the same one."""

        from madgraph.mg7 import systematics_summary

        up, down = systematics_summary.scale_percentages(self.INFO['systematics'])
        rows = self.module()._systematics_rows(self.INFO)
        self.assertIn('+%.3g%%' % up, rows)
        self.assertIn('-%.3g%%' % down, rows)

    def test_it_falls_back_when_there_is_no_run(self):
        """Nothing to read -- no output yet, or a run that made no events."""

        module = self.module()
        self.assertEqual(module._result_row(None), '503.1(1.4)')
        self.assertIn('+12.4%', module._systematics_rows(None))

    def test_it_falls_back_on_a_run_without_systematics(self):
        info = {'channels': self.INFO['channels']}
        module = self.module()
        self.assertEqual(module._result_row(info), '380.57(28)')
        self.assertIn('+12.4%', module._systematics_rows(info))

    def test_a_malformed_info_does_not_break_the_step(self):
        module = self.module()
        for broken in ({'channels': 'not a list'}, {'channels': []},
                       {'channels': [{'mean': 0.0, 'error': 0.0}]}):
            self.assertEqual(module._result_row(broken), '503.1(1.4)')

    def test_the_step_renders_with_a_real_run(self):
        import os
        import tempfile

        module = self.module()
        root = tempfile.mkdtemp()
        run = os.path.join(root, 'Events', 'run_01')
        os.makedirs(run)
        with open(os.path.join(run, 'info.json'), 'w') as handle:
            json.dump(self.INFO, handle)

        class _Interface(object):
            _done_export = [root, 'mg7']
            options = {}
            _curr_amps = []

        step = [s for s in tutorials.get('lo').steps
                if s.title == 'run it'][0]
        text = step.render(_Interface())
        self.assertIn('380.57(28)', text)
        self.assertIn('+26.6%', text)
        self.assertNotIn('503.1(1.4)', text)


class TutorialFailedCommandTest(unittest.TestCase):
    """A command which raises must not advance the tutorial, and must not
    leave the user in front of an error message with the tutorial silent."""

    class FakeBase(object):
        def postcmd(self, stop, line):
            return stop
        def notify_failed_command(self, line):
            pass

    def interface(self, name='model'):
        import madgraph.interface.tutorials as tutorials
        from madgraph.interface.tutorials.mixin import TutorialMixin

        class Fake(TutorialMixin, self.FakeBase):
            pass

        obj = Fake()
        obj._tutorial_session = tutorials.start(name)
        obj.exec_cmd_depth = 0
        return obj

    def test_a_failed_command_does_not_advance(self):
        obj = self.interface()
        session = obj._tutorial_session
        for line in ['import model sm', 'display particles',
                     'import model sm-no_b_mass', 'set gauge Feynman']:
            obj.postcmd(None, line)
        here = session.index

        obj.notify_failed_command('check gauge')   # needs a process: it raised
        obj.postcmd(None, 'check gauge')           # the interactive path
        self.assertEqual(session.index, here)

        # and the command the step waits for is still accepted afterwards
        obj.postcmd(None, 'define v = w+ w- z a')
        self.assertEqual(session.current.title, 'multiparticle labels')

    def test_a_successful_command_still_advances(self):
        obj = self.interface()
        session = obj._tutorial_session
        obj.postcmd(None, 'import model sm')
        self.assertEqual(session.current.title, 'load a model')

    def test_a_step_can_say_why_its_command_failed(self):
        """'check gauge' needs a process the user has no reason to guess"""

        import madgraph.interface.tutorials as tutorials

        step = [s for s in tutorials.get('model').steps
                if s.title == 'a detour: check gauge'][0]
        advice = step.get_failure_advice()
        self.assertTrue(advice)
        self.assertTrue('check gauge p p > e+ e-' in advice)

    def test_a_step_without_advice(self):
        import madgraph.interface.tutorials as tutorials

        step = [s for s in tutorials.get('model').steps
                if s.title == 'multiparticle labels'][0]
        self.assertEqual(step.get_failure_advice(), None)

    def test_only_the_first_report(self):
        """a script command fails, and the 'import' running it fails in turn"""

        obj = self.interface()
        obj.notify_failed_command('check gauge')
        self.assertEqual(obj._tutorial_failed_line, 'check gauge')
        obj.notify_failed_command('import /tmp/script.txt')
        self.assertEqual(obj._tutorial_failed_line, 'check gauge')

    def test_nothing_happens_without_a_tutorial(self):
        obj = self.interface()
        obj._tutorial_session = None
        obj.notify_failed_command('check gauge')
        self.assertEqual(getattr(obj, '_tutorial_failed_line', None), None)

    def test_the_hook_never_masks_the_real_error(self):
        """a failure inside the tutorial must not replace the user's error"""

        from madgraph.interface.extended_cmd import Cmd

        class Broken(object):
            def notify_failed_command(self, line):
                raise RuntimeError('tutorial is broken')

        Cmd.safe_notify_failed_command(Broken(), 'anything')  # must not raise
        Cmd.safe_notify_failed_command(object(), 'anything')  # no hook at all


class StickyStepTest(unittest.TestCase):
    """A lesson which invites the user to try several commands must answer them
    without being consumed, and without jumping to a later step that happens to
    share the key."""

    def session(self):
        import madgraph.interface.tutorials as tutorials
        return tutorials.start('model')

    def at_the_display_lesson(self):
        session = self.session()
        for line in ['import model sm', 'display particles']:
            session.advance(session.step_for(line)[0])
        self.assertEqual(session.current.title, 'look inside the model')
        return session

    def test_a_display_does_not_end_the_tutorial(self):
        """'display multiparticles' used to match the closing step"""

        session = self.at_the_display_lesson()
        for line in ['display multiparticles', 'display interactions',
                     'display couplings', 'display modellist']:
            index, step = session.step_for(line)
            self.assertEqual(step.title, 'trying the display commands', line)
            self.assertEqual(step.sticky, True)
            # sticky: the mixin renders it and does not advance
            self.assertEqual(session.current.title, 'look inside the model')

    def test_the_lesson_still_continues(self):
        session = self.at_the_display_lesson()
        index, step = session.step_for('import model sm-no_b_mass')
        self.assertEqual(step.title, 'restrictions')

    def test_the_closing_step_is_still_reachable(self):
        session = self.session()
        for line in ['import model sm', 'display particles',
                     'import model sm-no_b_mass', 'set gauge Feynman',
                     'define v = w+ w- z a']:
            session.advance(session.step_for(line)[0])
        index, step = session.step_for('display multiparticles')
        self.assertEqual(step.title, 'customise, merge, save')

    def test_a_later_display_does_not_end_it_either(self):
        """the closing step only answers the command it asks for"""

        session = self.session()
        for line in ['import model sm', 'display particles',
                     'import model sm-no_b_mass']:
            session.advance(session.step_for(line)[0])
        self.assertEqual(session.step_for('display particles'), None)

    def test_explain_restriction_is_answered_too(self):
        """the lesson names the command, so the user will try it"""

        session = self.session()
        for line in ['import model sm', 'display particles',
                     'import model sm-no_b_mass']:
            session.advance(session.step_for(line)[0])

        for line in ['explain_restriction', 'explain_restriction sm-full',
                     'explain_restriction --all']:
            index, step = session.step_for(line)
            self.assertEqual(step.title, 'a detour: explain_restriction', line)
            self.assertEqual(step.sticky, True)
        # the lesson is untouched and still continues
        self.assertEqual(session.current.title, 'restrictions')
        self.assertEqual(session.step_for('set gauge Feynman')[1].title,
                         'gauge and scheme')

    def test_the_define_lesson_says_it_too(self):
        """the lesson about multiparticles is where it matters most"""

        import madgraph.interface.tutorials as tutorials

        step = [s for s in tutorials.get('model').steps
                if s.title == 'multiparticle labels'][0]
        text = step.render(None)
        self.assertTrue('massless flavours' in text)
        self.assertTrue('sm-no_b_mass' in text)

    def test_explain_restriction_works_from_the_first_import(self):
        """the import of a restricted model suggests it right away"""

        session = self.session()
        session.advance(session.step_for('import model sm')[0])
        index, step = session.step_for('explain_restriction')
        self.assertEqual(step.title, 'a detour: explain_restriction')

    def test_the_multiparticle_note_says_what_p_and_j_follow(self):
        import madgraph.interface.tutorials.model as model

        note = model.display_note(None, 'display multiparticles')
        self.assertTrue('massless flavours' in note)

    def test_each_display_has_its_own_note(self):
        import madgraph.interface.tutorials.model as model

        for what in ['particles', 'interactions', 'couplings', 'parameters',
                     'multiparticles', 'modellist', 'coupling_order']:
            note = model.display_note(None, 'display %s' % what)
            self.assertTrue(note, what)
            # every one of them hands the user back to the tutorial
            self.assertTrue('import model sm-no_b_mass' in note, what)

    def test_coupling_order_points_at_the_syntax_tutorial(self):
        """the orders are what one constrains to filter the diagrams"""

        import madgraph.interface.tutorials as tutorials
        import madgraph.interface.tutorials.model as model

        note = model.display_note(None, 'display coupling_order')
        self.assertTrue('QED=0' in note)
        self.assertTrue('syntax' in note)
        # and that tutorial does exist, and does teach it
        self.assertNotEqual(tutorials.get('syntax'), None)
        self.assertTrue(any('coupling orders' in s.title
                            for s in tutorials.get('syntax').steps),
                        'the syntax tutorial no longer has an orders lesson')

    def test_an_unknown_display_still_answers(self):
        import madgraph.interface.tutorials.model as model

        self.assertTrue('import model sm-no_b_mass' in
                        model.display_note(None, 'display nonsense'))
        self.assertTrue('import model sm-no_b_mass' in
                        model.display_note(None, 'display'))

    def test_render_passes_the_line(self):
        """a sticky step's text takes the command, an ordinary one does not"""

        import madgraph.interface.tutorials as tutorials

        steps = tutorials.get('model').steps
        sticky = [s for s in steps if s.sticky][0]
        self.assertTrue('interactions' in
                        sticky.render(None, 'display interactions'))
        plain = [s for s in steps if s.title == 'customise, merge, save'][0]
        self.assertTrue(plain.render(None, 'display multiparticles'))


#===============================================================================
# the prompt, `tutorial index` and `tutorial skip N`
#===============================================================================

class _Recording(object):
    """The interface side the mixin needs, with commands recorded, not run."""

    prompt = 'MG7> '

    def __init__(self):
        self.ran = []
        self.exec_cmd_depth = 0
        self._curr_amps = []
        self.options = {}

    def postcmd(self, stop, line):
        return stop

    def notify_failed_command(self, line):
        pass

    def exec_cmd(self, line, printcmd=True, precmd=False, postcmd=True):
        self.ran.append(line)


class TutorialPromptTest(_TutorialTestCase):

    def interface(self, name='syntax'):
        interface = _Recording()
        tutorial_mixin.attach(interface, tutorials.start(name))
        return interface

    def test_the_prompt_says_where_the_reader_is(self):
        interface = self.interface()
        interface.postcmd(None, 'tutorial syntax')
        total = len(tutorials.get('syntax').steps)
        self.assertIn('TUTO [syntax] 1/%d>' % total, interface.prompt)
        interface.postcmd(None, 'generate p p > t t~ QED<=2')
        self.assertIn('TUTO [syntax] 2/%d>' % total, interface.prompt)

    def test_it_follows_a_move_that_fires_no_step(self):
        """`skip` and `back` move the session without a step firing; the
        prompt is refreshed after every command, not only the ones that
        advance."""

        interface = self.interface()
        interface.postcmd(None, 'tutorial syntax')
        interface.do_skip('')
        interface.postcmd(None, 'skip')
        self.assertIn('/%d>' % len(tutorials.get('syntax').steps),
                      interface.prompt)
        self.assertIn('2/', interface.prompt)

    def test_stopping_gives_the_prompt_back(self):
        interface = self.interface()
        interface.postcmd(None, 'tutorial syntax')
        tutorial_mixin.detach(interface)
        self.assertEqual(interface.prompt, 'MG7> ')


class TutorialPathTest(unittest.TestCase):
    """What `tutorial skip N` replays is the reader's own route to step N."""

    def path(self, name, number):
        tutorial = tutorials.get(name)
        return TutorialSession(tutorial).path_to(number - 1)

    def test_a_main_line_step(self):
        self.assertEqual(self.path('syntax', 3),
                         ['generate p p > t t~ QED<=2',
                          'generate p p > j j QCD^2==2 QED^2==2'])

    def test_a_side_quest_is_entered_by_its_entry(self):
        """The main line jumps over a side quest; its entry is the way in."""

        path = self.path('syntax', 4)
        self.assertEqual(path[-1], 'output standalone')
        self.assertEqual(len(path), 3)

    def test_a_sticky_step_is_nowhere_to_go(self):
        titles = [s.title for s in tutorials.get('syntax').steps]
        sticky = titles.index('what $ and $$ did') + 1
        self.assertIsNone(self.path('syntax', sticky))

    def test_a_free_order_tutorial_has_no_route(self):
        self.assertIsNone(self.path('nlo', 3))

    def test_every_step_is_reachable(self):
        """Apart from a sticky step, and the sign-off an exercise tutorial
        prints when its last answer passes, every step of every sequenced
        tutorial has a route -- a detour included, through its entry."""

        for tutorial in tutorials.all_tutorials(include_hidden=True):
            if tutorial.order != 'sequence':
                continue
            session = TutorialSession(tutorial)
            steps = tutorial.steps
            for index, step in enumerate(steps):
                if index == 0 or step.sticky:
                    continue
                if isinstance(steps[index - 1], Exercise):
                    continue        # the sign-off after the last exercise
                reach = index - 1 if isinstance(step, Exercise) else index
                if reach == 0:
                    continue
                self.assertIsNotNone(
                    session.path_to(reach),
                    '%s step %d (%s) cannot be reached by `tutorial skip`'
                    % (tutorial.name, index + 1, step.title))

    def test_what_is_replayed_and_what_is_skipped(self):
        from madgraph.interface.tutorials.session import replay_line

        self.assertEqual(replay_line('output madevent POL_ZZ'),
                         'output madevent POL_ZZ -f')
        self.assertEqual(replay_line('generate p p > t t~'),
                         'generate p p > t t~')
        for skipped in ('launch', 'display diagrams', 'history f.dat',
                        'check gauge p p > e+ e-', 'install madspace',
                        'compute_widths t'):
            self.assertIsNone(replay_line(skipped), skipped)


class TutorialSkipToStepTest(_TutorialTestCase):

    def interface(self, name='syntax'):
        interface = _Recording()
        tutorial_mixin.attach(interface, tutorials.start(name))
        interface._tutorial_session.advance(0)
        return interface

    def test_it_runs_the_route_and_lands_on_the_step(self):
        interface = self.interface()
        interface.do_skip('6')
        session = interface._tutorial_session
        self.assertEqual(session.index, 5)
        self.assertEqual(interface.ran,
                         ['generate p p > t t~ QED<=2',
                          'generate p p > j j QCD^2==2 QED^2==2',
                          'generate p p > j j QCD^2==4 QED^2==0'])
        self.assertIn('Step 6 of', self.capture.messages[0])
        self.assertIn('A comma opens a decay chain', self.capture.messages[1])

    def test_an_output_is_forced_and_a_launch_skipped(self):
        interface = self.interface('decays')
        interface.do_skip('5')          # after the decay-chain run
        self.assertIn('output TT_DECAY -f', interface.ran)
        self.assertNotIn('launch', interface.ran)
        self.assertIn('launch', self.capture.messages[0])   # said so

    def test_going_back_replays_from_the_start(self):
        interface = self.interface()
        interface.do_skip('6')
        interface.ran[:] = []
        interface.do_skip('3')
        self.assertEqual(interface._tutorial_session.index, 2)
        self.assertEqual(len(interface.ran), 2)

    def test_what_it_refuses(self):
        interface = self.interface()
        titles = [s.title for s in tutorials.get('syntax').steps]
        for bad in ('0', '99', 'three', str(titles.index('what $ and $$ did') + 1)):
            self.capture.messages = []
            interface.do_skip(bad)
            self.assertEqual(interface._tutorial_session.index, 0, bad)
            self.assertEqual(interface.ran, [], bad)
            self.assertTrue(self.capture.messages, bad)

    def test_the_command_accepts_a_step_number(self):
        interface = _BareMadGraphCmd()
        interface.check_tutorial(['skip', '3'])
        self.assertRaises(madgraph.InvalidCmd,
                          interface.check_tutorial, ['skip', 'three'])
        self.assertIn('index', interface._tutorial_opts)


#===============================================================================
# the text editor a card opens in
#===============================================================================

class TutorialEditorTest(unittest.TestCase):
    """`lo` says which editor a card will open in before the reader opens one,
    offers `help vi` when it is vi, and says how to change it afterwards."""

    def setUp(self):
        import os
        import madgraph.various.misc as misc

        self.misc = misc
        self._which = misc.which
        self._editor = os.environ.get('EDITOR')
        self.installed = set()
        misc.which = lambda program: program in self.installed

    def tearDown(self):
        import os

        self.misc.which = self._which
        if self._editor is None:
            os.environ.pop('EDITOR', None)
        else:
            os.environ['EDITOR'] = self._editor

    def resolve(self, configured=None):
        return self.misc.open_file.resolve_text_editor(configured, quiet=True)

    def test_the_order_mg7_resolves_it_in(self):
        import os

        os.environ.pop('EDITOR', None)
        self.installed = {'emacs', 'nano'}
        self.assertEqual(self.resolve(), 'emacs')          # first default found
        os.environ['EDITOR'] = 'nano'
        self.assertEqual(self.resolve(), 'nano')           # $EDITOR wins
        self.installed.add('code')
        self.assertEqual(self.resolve('code -w'), 'code -w')   # the option wins
        self.assertEqual(self.resolve('missing'), 'nano')  # unless not installed

    def test_the_question_names_the_editor_and_offers_help_vi(self):
        import os
        import madgraph.interface.tutorials.lo as lo

        class _Interface(object):
            options = {}

        self.installed = {'vi'}
        os.environ['EDITOR'] = 'vi'
        hint = lo._launch_question_hint(_Interface())
        self.assertIn('**vi**', hint)
        self.assertIn('`help vi`', hint)

        self.installed = {'nano'}
        os.environ['EDITOR'] = 'nano'
        hint = lo._launch_question_hint(_Interface())
        self.assertIn('**nano**', hint)
        self.assertNotIn('help vi', hint)

    def test_back_at_the_prompt_it_says_how_to_change_it(self):
        import madgraph.interface.tutorials.lo as lo

        class _Interface(object):
            options = {'text_editor': 'emacs'}

        self.installed = {'emacs'}
        text = lo._editor_back_at_the_prompt(_Interface())
        self.assertIn('**emacs**', text)
        self.assertIn('`text_editor` option', text)
        self.assertIn('set text_editor', text)
        self.assertIn('save options text_editor', text)

    def test_help_vi_answers(self):
        import madgraph.interface.extended_cmd as extended_cmd

        captured = []

        class _Handler(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        handler = _Handler()
        logger = logging.getLogger('cmdprint')
        logger.addHandler(handler)
        level = logger.level
        logger.setLevel(logging.INFO)
        try:
            extended_cmd.BasicCmd.help_vi(None)
            # the launch switch question calls help_X with an argument
            # (print_help_for_switch); without *args that crashed the launch
            extended_cmd.BasicCmd.help_vi(None, '')
        finally:
            logger.removeHandler(handler)
            logger.setLevel(level)
        self.assertEqual(len(captured), 2)
        for key in (':wq', ':q!', 'Esc'):
            self.assertIn(key, captured[0])


class TutorialWaitingForTest(unittest.TestCase):
    """What a failed command is told the tutorial is still waiting for."""

    def test_it_is_the_current_steps_command(self):
        """It used to quote the solution of the step *after* the current one:
        at the bsm intro, a failed command was told the tutorial waited for
        the step after's command rather than `import model`."""

        captured = []

        class _Handler(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        logger = logging.getLogger('tutorial')
        saved = (logger.handlers, logger.propagate, logger.level)
        logger.handlers = [_Handler()]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        try:
            interface = _Recording()
            tutorial_mixin.attach(interface, tutorials.start('bsm'))
            interface._tutorial_session.advance(0)
            interface.notify_failed_command('display modellist')
        finally:
            (logger.handlers, logger.propagate, logger.level) = saved
            tutorial_mixin.detach(interface)
        steps = tutorials.get('bsm').steps
        self.assertTrue(captured)
        self.assertIn(steps[0].get_solution(), captured[-1])
        self.assertNotIn(steps[1].get_solution(), captured[-1])


class TutorialGateTest(unittest.TestCase):
    """A lesson which needs the state its command was meant to leave.

    The bsm tutorial is the case: everything after its intro is about the
    EFT's `NP` order, so `import model sm` must not take the reader into it.
    """

    class _Model(object):

        def __init__(self, name):
            self.name = name

        def get(self, key):
            return {'name': self.name,
                    'particles': ['p'] * 10,
                    'interactions': ['v'] * 20}[key]

    def _run(self, model_name, line):
        """The bsm intro, then `line` with `model_name` loaded."""

        captured = []

        class _Handler(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        logger = logging.getLogger('tutorial')
        saved = (logger.handlers, logger.propagate, logger.level)
        logger.handlers = [_Handler()]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        try:
            interface = _Recording()
            interface._curr_model = None
            tutorial_mixin.attach(interface, tutorials.start('bsm'))
            interface.postcmd(None, 'tutorial bsm')
            if model_name:
                interface._curr_model = self._Model(model_name)
            interface.postcmd(None, line)
            return interface._tutorial_session.index, captured[-1]
        finally:
            (logger.handlers, logger.propagate, logger.level) = saved

    def test_the_standard_model_does_not_open_lesson_two(self):
        index, text = self._run('sm', 'import model sm')
        self.assertEqual(index, 0)
        self.assertIn('sm', text)
        self.assertIn('import model %s' % bsm.MODEL, text)
        # not the lesson itself
        self.assertNotIn('display coupling_order', text)

    def test_no_model_at_all_is_told_so(self):
        index, text = self._run(None, 'import model nowhere')
        self.assertEqual(index, 0)
        self.assertIn('No model is loaded', text)

    def test_an_eft_does_open_it(self):
        index, text = self._run(bsm.MODEL, 'import model %s' % bsm.MODEL)
        self.assertEqual(index, 1)
        self.assertIn('display coupling_order', text)

    def test_the_readers_own_restriction_is_an_eft_too(self):
        """`customize_model --save=top` reloads as SMEFTatNLO-top, and the
        gate has to let that through -- the tutorial ends on it."""

        self.assertIs(bsm._is_smeft(_ModelHolder(bsm.RELOAD.split()[-1])), True)

    def test_a_gate_which_raises_lets_the_step_through(self):
        """A broken check is not a reason to withhold a lesson."""

        def _boom(interface, line):
            raise RuntimeError('no')

        step = Step('generate', 'TEXT', gate=_boom)
        self.assertIsNone(step.refusal(None, 'generate p p > t t~'))

    def test_an_explicit_false_refuses_without_a_word(self):
        step = Step('generate', 'TEXT', gate=lambda i, l: False)
        self.assertIn('stays where it is', step.refusal(None, 'generate'))

    def test_a_gate_which_returns_nothing_lets_the_step_fire(self):
        """The shape a check is naturally written in returns None when it is
        happy; taking that for a refusal would strand the reader behind a
        message that names nothing."""

        def _gate(interface, line):
            if line == 'wrong':
                return 'that is not it'

        step = Step('generate', 'TEXT', gate=_gate)
        self.assertIsNone(step.refusal(None, 'right'))
        self.assertEqual(step.refusal(None, 'wrong'), 'that is not it')

    def test_an_empty_string_is_not_a_refusal_either(self):
        step = Step('generate', 'TEXT', gate=lambda i, l: '')
        self.assertIsNone(step.refusal(None, 'generate'))

    def test_the_index_of_a_gated_step_needs_no_interface(self):
        """step_for answers the same with no interface at all: `tutorial
        index` and `skip N` walk the tutorial without one."""

        session = TutorialSession(tutorials.get('bsm'))
        session.index = 0
        found = session.step_for('import model %s' % bsm.MODEL)
        self.assertEqual(found[0], 1)


class BsmInterferenceDetourTest(_TutorialTestCase):
    """The pure-interference lesson is a side quest, not a step of the line.

    The main line goes from `linear and quadratic` straight to the output; a
    reader who asks for `NP^2==2` gets the lesson and is put back on it.
    """

    name = 'bsm'

    def session_at(self, title):
        session = tutorials.start('bsm')
        session.index = [i for i, step in enumerate(session.tutorial.steps)
                         if step.title == title][0]
        return session

    def test_the_main_line_steps_over_it(self):
        session = self.session_at('linear and quadratic')
        found = session.step_for(bsm.OUTPUT_BEFORE)
        self.assertEqual(found[1].title, 'the parameters, on paper')

    def test_the_process_line_takes_the_detour(self):
        session = self.session_at('linear and quadratic')
        found = session.step_for(bsm.LINEAR)
        self.assertEqual(found[1].title, 'the interference on its own (detour)')

    def test_the_detour_leads_back_to_the_main_line(self):
        session = self.session_at('the interference on its own (detour)')
        found = session.step_for(bsm.OUTPUT_BEFORE)
        self.assertEqual(found[1].title, 'the parameters, on paper')

    def test_both_routes_reach_the_end(self):
        """path_to walks the main line; the detour is reached by its entry."""

        steps = tutorials.get('bsm').steps
        detour = [i for i, s in enumerate(steps)
                  if s.title == 'the interference on its own (detour)'][0]
        route = TutorialSession(tutorials.get('bsm')).path_to(detour)
        self.assertIsNotNone(route)
        self.assertEqual(route[-1], bsm.LINEAR)

    def test_the_regeneration_asks_for_the_readers_own_process(self):
        """After the detour the process in memory is not the main-line one,
        and the lesson which asks for it again has to ask for theirs."""

        class _History(object):
            def __init__(self, *lines):
                self.history = list(lines)

        main = _History('import model %s' % bsm.MODEL, bsm.ONE_INSERTION,
                        bsm.OUTPUT_BEFORE)
        detoured = _History('import model %s' % bsm.MODEL, bsm.ONE_INSERTION,
                            bsm.LINEAR, bsm.OUTPUT_BEFORE)

        step = [s for s in tutorials.get('bsm').steps
                if s.title == 'a restriction of your own'][0]
        self.assertEqual(step.get_solution(main), bsm.ONE_INSERTION)
        self.assertEqual(step.get_solution(detoured), bsm.LINEAR)
        self.assertIn(bsm.LINEAR, step.get_hint(detoured))
        # and with nothing to read, the main line's process
        self.assertEqual(step.get_solution(None), bsm.ONE_INSERTION)


class CallableHintTest(unittest.TestCase):
    """`hint` resolves a callable the way `solution` does."""

    def test_a_callable_hint_is_called(self):
        step = Step('generate', 'TEXT', hint=lambda interface: 'HINT')
        self.assertEqual(step.get_hint(None), 'HINT')

    def test_a_hint_which_raises_is_no_hint(self):
        def _boom(interface):
            raise RuntimeError('no')

        self.assertIsNone(Step('generate', 'TEXT', hint=_boom).get_hint(None))

    def test_a_plain_string_is_returned_as_is(self):
        self.assertEqual(Step('generate', 'TEXT', hint='HINT').get_hint(None),
                         'HINT')


class QuestionProgressTest(unittest.TestCase):
    """A step that says the next thing once the reader has done the previous.

    The hint at the top of a question is read once, when none of it applies
    yet; what follows an answer has to be printed when that answer is given.
    """

    def capture_tutorial(self, call):
        captured = []

        class _Handler(logging.Handler):
            def emit(self, record):
                # MG7 logs the style as a second argument ('$MG:BOLD'), which
                # getMessage() would try to interpolate into the text
                captured.append(record.msg)

        logger = logging.getLogger('tutorial')
        saved = (logger.handlers, logger.propagate, logger.level)
        logger.handlers = [_Handler()]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        try:
            call()
        finally:
            (logger.handlers, logger.propagate, logger.level) = saved
        return captured

    def test_a_question_prints_what_the_hook_returns(self):
        """The plumbing: extended_cmd calls it with the answer just given."""

        import madgraph.interface.extended_cmd as extended_cmd

        question = extended_cmd.SmartQuestion('A question?',
                                              allow_arg=['done'],
                                              default='done')
        extended_cmd.question_progress = lambda line: 'SAW [%s]' % line
        try:
            captured = self.capture_tutorial(
                lambda: question.reask(line='set DIM64F2L all 0'))
        finally:
            extended_cmd.question_progress = None
        self.assertTrue(any('SAW [set DIM64F2L all 0]' in text
                            for text in captured), captured)

    def test_the_answer_comes_from_the_caller_not_from_lastcmd(self):
        """An answer read from a command file never reaches lastcmd, so the
        hook has to be given the line rather than read it back."""

        import madgraph.interface.extended_cmd as extended_cmd

        question = extended_cmd.SmartQuestion('A question?',
                                              allow_arg=['done'],
                                              default='done')
        question.lastcmd = 'an older answer'
        seen = []
        extended_cmd.question_progress = lambda line: seen.append(line) or None
        try:
            self.capture_tutorial(lambda: question.reask(line='set a 0'))
        finally:
            extended_cmd.question_progress = None
        self.assertEqual(seen, ['set a 0'])

    def test_postcmd_reports_the_line_it_handled(self):
        """The path a `set` at a real question takes: postcmd -> reask."""

        import madgraph.interface.extended_cmd as extended_cmd

        class _Question(extended_cmd.SmartQuestion):
            def do_set(self, line):
                self.value = 'repeat'

        question = _Question('A question?', allow_arg=['done'], default='done')
        question.value = 'repeat'
        seen = []
        extended_cmd.question_progress = lambda line: seen.append(line) or None
        try:
            self.capture_tutorial(
                lambda: question.postcmd(False, 'set DIM64F2L all 0'))
        finally:
            extended_cmd.question_progress = None
        self.assertEqual(seen, ['set DIM64F2L all 0'])

    def test_no_hook_prints_nothing(self):
        import madgraph.interface.extended_cmd as extended_cmd

        self.assertIsNone(extended_cmd.get_question_progress('set x 0'))

    def test_a_hook_which_raises_prints_nothing(self):
        import madgraph.interface.extended_cmd as extended_cmd

        def _boom(line):
            raise RuntimeError('no')

        extended_cmd.question_progress = _boom
        try:
            self.assertIsNone(extended_cmd.get_question_progress('set x 0'))
        finally:
            extended_cmd.question_progress = None

    def test_attach_arms_it_and_detach_clears_it(self):
        import madgraph.interface.extended_cmd as extended_cmd

        interface = _Recording()
        tutorial_mixin.attach(interface, tutorials.start('bsm'))
        try:
            self.assertTrue(callable(extended_cmd.question_progress))
        finally:
            tutorial_mixin.detach(interface)
        self.assertIsNone(extended_cmd.question_progress)

    def test_the_session_asks_the_current_step(self):
        tutorial = Tutorial('t', 'title', order='sequence', steps=[
            Step('generate', 'ONE', title='one', solution='output X'),
            Step('output', 'TWO', title='two',
                 question_progress=lambda interface, line: 'GOT %s' % line)])
        session = TutorialSession(tutorial)
        session.advance(1)
        self.assertEqual(session.question_progress('set a 0'), 'GOT set a 0')

    def test_a_step_without_one_says_nothing(self):
        session = TutorialSession(Tutorial('t', 'title', order='sequence',
                                  steps=[Step('generate', 'ONE', title='one')]))
        session.advance(0)
        self.assertIsNone(session.question_progress('set a 0'))


class BsmCustomizeProgressTest(unittest.TestCase):
    """What the bsm lesson adds while customize_model is being answered."""

    class _Interface(object):
        _curr_model = None

    def progress(self, interface, *lines):
        return [bsm._customize_question_progress(interface, line)
                for line in lines]

    def test_the_first_set_earns_the_display_lines(self):
        interface = self._Interface()
        first, second = self.progress(interface, 'set DIM64F2L all 0',
                                      'set DIM64F4L all 0')
        self.assertIn('display parameters DIM64F2L', first)
        self.assertIn('display couplings', first)
        # said once: the reader is answering a question, not reading a page
        self.assertIsNone(second)

    def test_the_first_display_earns_how_to_read_it(self):
        interface = self._Interface()
        self.progress(interface, 'set DIM64F2L all 0')
        first, second = self.progress(interface, 'display parameters DIM64F2L',
                                      'display couplings GC_73')
        self.assertIn('unfolds it down to the', first)
        self.assertIn('before `done` commits it', first)
        self.assertIsNone(second)

    def test_nothing_else_is_commented_on(self):
        interface = self._Interface()
        for line in ('done', '', 'set_zero cQe1', 'help'):
            self.assertIsNone(
                bsm._customize_question_progress(interface, line), line)

    def test_the_opening_hint_stays_short(self):
        """The two blocks that follow an answer are not in it, and neither is
        the `-NLO` paragraph that used to close it."""

        hint = bsm._customize_question_hint(self._Interface())
        self.assertIn('set DIM64F2L all 0', hint)
        self.assertNotIn('display', hint)
        self.assertNotIn('set NAME free', hint)
        self.assertLess(len(hint.split('\n')), 15, hint)


class CustomizeQuestionHintTest(unittest.TestCase):
    """The bsm hint shown while customize_model asks its question.

    It invites a `display` of what the first `set` did, and the coupling it
    names has to be one of the reader's own model: `display couplings` matches
    on the coupling name, so a name out of thin air would send them nowhere.
    """

    class _Parameter(object):
        def __init__(self, name, lhablock):
            self.name = name
            self.lhablock = lhablock

    class _Coupling(object):
        def __init__(self, name, expr):
            self.name = name
            self.expr = expr

    def model(self, couplings):
        """A model whose DIM64F2L block holds cQe1, with those couplings."""

        parameters = {('external',): [
            self._Parameter('mdl_cQe1', 'DIM64F2L'),
            self._Parameter('mdl_Lambda', 'DIM6')]}

        class _Model(object):
            def get(inner, key):
                return {'parameters': parameters, 'couplings': couplings}[key]

        class _Interface(object):
            _curr_model = _Model()

        return _Interface()

    def test_it_names_a_coupling_the_restriction_kills(self):
        interface = self.model({('QED',): [
            self._Coupling('GC_1', '-(mdl_ee*mdl_complexi)/3.'),
            self._Coupling('GC_73', '(mdl_cQe1*mdl_complexi)/mdl_Lambda__exp__2')]})
        self.assertEqual(bsm._coupling_using(interface), 'GC_73')
        self.assertIn('display couplings GC_73',
                      bsm._customize_question_progress(interface,
                                                       'set DIM64F2L all 0'))

    def test_the_shortest_expression_wins(self):
        """One coefficient over Lambda^2 shows the point; a sum of four does
        not."""

        interface = self.model({('QED',): [
            self._Coupling('GC_9', '(2*mdl_cQe1*mdl_complexi)/mdl_Lambda__exp__2'
                                   ' + (mdl_cQe1*mdl_complexi)/mdl_Lambda__exp__2'),
            self._Coupling('GC_73', '(mdl_cQe1*mdl_complexi)/mdl_Lambda__exp__2')]})
        self.assertEqual(bsm._coupling_using(interface), 'GC_73')

    def test_a_model_with_no_such_coupling_still_invites_display(self):
        interface = self.model({('QED',): [
            self._Coupling('GC_1', '-(mdl_ee*mdl_complexi)/3.')]})
        self.assertIsNone(bsm._coupling_using(interface))
        said = bsm._customize_question_progress(interface, 'set DIM64F2L all 0')
        self.assertIn('display couplings NAME', said)
        self.assertIn('display parameters DIM64F2L', said)

    def test_it_survives_an_interface_with_no_model(self):
        class _Empty(object):
            _curr_model = None

        self.assertIsNone(bsm._coupling_using(_Empty()))
        self.assertIn('display parameters DIM64F2L',
                      bsm._customize_question_progress(_Empty(),
                                                       'set DIM64F2L all 0'))

    def test_the_step_uses_it(self):
        """The hint reaches the question through the step that asks for
        customize_model."""

        session = tutorials.start('bsm')
        step = [s for s in session.tutorial.steps
                if s.get_solution() == bsm.CUSTOMIZE][0]
        session.index = session.tutorial.steps.index(step)

        class _Empty(object):
            _curr_model = None

        self.assertIn('set DIM64F2L all 0', session.question_hint(_Empty()))


class _ModelHolder(object):
    """An interface holding one model, for the gate check above."""

    def __init__(self, name):
        self._curr_model = TutorialGateTest._Model(name)


class DisplayModellistTest(unittest.TestCase):

    def test_the_model_list_needs_no_model(self):
        """It lists model directories and the online database: the banner
        suggests it at startup, before any model is loaded."""

        interface = _BareMadGraphCmd()
        interface._curr_model = None
        interface.check_display(['modellist'])      # does not raise
        self.assertRaises(madgraph.InvalidCmd,
                          interface.check_display, ['particles'])
