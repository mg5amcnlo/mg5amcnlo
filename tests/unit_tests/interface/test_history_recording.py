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
"""What ends up in the command history.

Two things a history file has to get right to be replayable with
`import command`: it must not end by rewriting itself, and it must contain the
answers given to questions -- a `set` typed at the launch card question is part
of the run, and leaving it out produced a file that silently reproduced
something else.
"""

from __future__ import absolute_import

import os
import shutil
import tempfile
import unittest

import madgraph.interface.extended_cmd as extended_cmd


class _Interface(extended_cmd.Cmd):
    """A Cmd with just the bits history touches."""

    def __init__(self):
        self.history = extended_cmd.HistoryList() \
            if hasattr(extended_cmd, 'HistoryList') else []
        self.mother = None
        self.log = False
        self._export_dir = None

    def split_arg(self, line):
        return line.split()

    def check_history(self, args):
        return True

    def get_history_header(self):
        return '# header\n'


class TestHistoryDoesNotWriteItself(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.interface = _Interface()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def written(self, name='out.dat'):
        path = os.path.join(self.dir, name)
        # precmd records the command before do_history runs, as it does live
        self.interface.history.append('history %s' % path)
        self.interface.do_history(path)
        with open(path) as handle:
            return [l for l in handle.read().split('\n')
                    if l and not l.startswith('#')]

    def test_the_history_command_is_not_in_the_file(self):
        self.interface.history.extend(['generate p p > t t~', 'output PROC'])
        lines = self.written()
        self.assertEqual(lines, ['generate p p > t t~', 'output PROC'])

    def test_everything_else_survives(self):
        self.interface.history.extend(['import model sm', 'generate p p > z'])
        self.assertIn('import model sm', self.written())

    def test_it_only_drops_a_trailing_history_command(self):
        """An earlier `history` is not the one being run."""

        self.interface.history.extend(['history first.dat',
                                       'generate p p > t t~'])
        lines = self.written()
        self.assertIn('history first.dat', lines)
        self.assertIn('generate p p > t t~', lines)

    def test_an_empty_history_is_fine(self):
        self.assertEqual(self.written(), [])


class TestQuestionAnswersAreRecorded(unittest.TestCase):
    """A question runs its own cmdloop, so what is typed at it never reached
    the history of the interface that asked."""

    def test_a_typed_answer_reaches_the_mother(self):
        mother = _Interface()
        question = extended_cmd.SmartQuestion('pick one',
                                              mother_interface=mother)
        question.precmd('set nevents 321')
        self.assertEqual(list(mother.history), ['set nevents 321'])

    def test_an_empty_answer_is_not_recorded(self):
        """Pressing Enter takes the default; an empty line in a command file
        is skipped on replay anyway, so recording it would add noise that
        cannot mean anything."""

        mother = _Interface()
        question = extended_cmd.SmartQuestion('pick one',
                                              mother_interface=mother)
        for blank in ('', '   ', None):
            question.precmd(blank)
        self.assertEqual(list(mother.history), [])

    def test_precmd_still_returns_the_line(self):
        mother = _Interface()
        question = extended_cmd.SmartQuestion('pick one',
                                              mother_interface=mother)
        self.assertEqual(question.precmd('done'), 'done')

    def test_it_records_up_the_mother_chain(self):
        """The question's mother is often a child interface -- the run
        interface `launch` created -- while `history` is typed at the one the
        user started from. Both need it.
        """

        root, child = _Interface(), _Interface()
        child.mother = root
        question = extended_cmd.SmartQuestion('pick one',
                                              mother_interface=child)
        question.precmd('set nevents 321')
        self.assertEqual(list(child.history), ['set nevents 321'])
        self.assertEqual(list(root.history), ['set nevents 321'])

    def test_a_cycle_in_the_chain_does_not_hang(self):
        a, b = _Interface(), _Interface()
        a.mother, b.mother = b, a
        extended_cmd.record_answer_in_history(a, 'done')
        self.assertEqual(list(a.history), ['done'])
        self.assertEqual(list(b.history), ['done'])

    def test_an_interface_without_a_history_is_tolerated(self):
        class _Bare(object):
            mother = None

        extended_cmd.record_answer_in_history(_Bare(), 'done')   # must not raise

    def test_no_mother_is_tolerated(self):
        question = extended_cmd.SmartQuestion('pick one', mother_interface=None)
        self.assertEqual(question.precmd('done'), 'done')


class TestQuestionAnswersStayOutOfTheProcCard(unittest.TestCase):
    """An answer typed at a question is replayed by `history` but is not a
    command of the prompt whose history holds it.

    The failure this pins: `set width 6 auto` typed at the card question of a
    first `launch` was recorded into the MG5 history, the next `output` wrote
    it into proc_card_mg5.dat, and MadSpin -- which replays every `set` line of
    a proc card on a bare MG5 prompt -- died on it, since MG5 has no `set
    width`.
    """

    def setUp(self):
        import madgraph.various.banner as banner

        self.dir = tempfile.mkdtemp()
        self.card = banner.ProcCard()
        self.card.append('import model sm')
        self.card.append('generate p p > t t~, t > w+ b, t~ > w- b~')
        self.card.append('output TT_DECAY')
        self.card.append('launch TT_DECAY')

        class _Asker(object):
            mother = None

        asker = _Asker()
        asker.history = self.card
        for answer in ('set width 6 auto', 'done'):
            extended_cmd.record_answer_in_history(asker, answer)
        self.card.append('generate p p > t t~')
        self.card.append('output TT_MADSPIN')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def written(self):
        path = os.path.join(self.dir, 'proc_card_mg5.dat')
        self.card.write(path)
        with open(path) as handle:
            return [l for l in handle.read().split('\n')
                    if l and not l.startswith('#')]

    def test_the_proc_card_leaves_the_answers_out(self):
        written = self.written()
        self.assertNotIn('set width 6 auto', written)
        self.assertNotIn('done', written)
        self.assertIn('generate p p > t t~', written)

    def test_the_history_still_replays_them(self):
        """`history` writes the list itself, answers included -- otherwise the
        file reruns the launch with the defaults."""

        replay = '\n'.join(self.card)
        self.assertIn('set width 6 auto', replay)
        self.assertIn('done', replay)

    def test_they_stay_marked_through_the_proc_card_cleaning(self):
        """ProcCard.append rebuilds each line as it strips it; the marker has
        to survive that, and the popping a later `generate` does."""

        answers = [l for l in self.card if extended_cmd.is_question_answer(l)]
        self.assertEqual(answers, ['set width 6 auto', 'done'])

    def test_a_typed_command_is_not_an_answer(self):
        self.assertFalse(extended_cmd.is_question_answer('set nb_core 4'))
        self.assertTrue(extended_cmd.is_question_answer(
            extended_cmd.QuestionAnswer('set nb_core 4')))

    def test_the_set_lines_a_launch_copies_skip_them(self):
        """What launch_ext_program and the aMC@NLO hand-off pass on to the run
        they start: the MG5 settings, not an earlier run's card edits."""

        self.card.append('set nb_core 4')
        copied = [l for l in self.card if l.strip().startswith('set')
                  and not extended_cmd.is_question_answer(l)]
        self.assertIn('set nb_core 4', copied)
        self.assertNotIn('set width 6 auto', copied)
