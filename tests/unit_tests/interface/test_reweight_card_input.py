################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
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
"""The reweight card alone answers the param_card question of a launch block.

Each ``launch`` line of a reweight card opens the param_card question
(handle_param_card), and the ``set`` lines that follow answer it. The question
ends with the card or with the next launch block. The generic
check_answer_in_input_file reads stdin at that point whenever stdin is not a
tty (``haspiping``), so ``bin/madgraph script`` run with stdin left open (a
pipe nobody writes to) hung after "generating the square matrix element for
reweighting". These tests drive that question with a stdin that fails when
read.
"""

import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

import madgraph
import madgraph.interface.common_run_interface as common_run_interface
import madgraph.interface.reweight_interface as reweight_interface

pjoin = os.path.join


class _UnreadableStdin(io.StringIO):
    """stdin of a script run with its input left open: reading it would
    wait forever, so make it fail instead."""

    def readline(self, *args):
        raise AssertionError('the reweight card question read stdin')

    read = readline


class TestReweightCardAnswersParamCardQuestion(unittest.TestCase):

    def setUp(self):
        self.rw_dir = tempfile.mkdtemp(prefix='rw_me_')
        os.mkdir(pjoin(self.rw_dir, 'Cards'))
        shutil.copy(pjoin(madgraph.MG5DIR, 'models', 'sm', 'restrict_default.dat'),
                    pjoin(self.rw_dir, 'Cards', 'param_card.dat'))
        self.rw = reweight_interface.ReweightInterface()
        # as for a script run with stdin a pipe
        self.rw.haspiping = True

    def tearDown(self):
        # no event file was imported, so there is nothing for do_quit to do
        self.rw.exitted = True
        shutil.rmtree(self.rw_dir)

    def ask_param_card(self, card_lines):
        """the question of handle_param_card, with the rest of the reweight
        card after a launch line as the input file"""
        self.rw.inputfile = iter(card_lines)
        with mock.patch('sys.stdin', _UnreadableStdin()):
            return common_run_interface.CommonRunCmd.ask_edit_card_static(
                cards=['param_card.dat'], ask=self.rw.ask, pwd=self.rw_dir,
                first_cmd=None, write_file=False, return_instance=True)

    def test_end_of_card_closes_the_question(self):
        """after the last launch block, the end of the card means done"""
        answer, question = self.ask_param_card(['# Z width\n', '\n',
                                                'set decay 23 2.6\n'])
        self.assertEqual(answer, '0')
        self.assertEqual(question.param_card['decay'].get((23,)).value, 2.6)
        self.assertFalse(self.rw.get_stored_line())

    def test_next_launch_block_closes_the_question(self):
        """the next launch line ends this block and is kept for the
        interface, with the rest of the card left unread"""
        rest = iter(['set decay 23 2.6\n', 'launch --rwgt_name=b\n',
                     'set decay 23 2.7\n'])
        answer, question = self.ask_param_card(rest)
        self.assertEqual(answer, '0')
        self.assertEqual(question.param_card['decay'].get((23,)).value, 2.6)
        self.assertEqual(self.rw.get_stored_line(), 'launch --rwgt_name=b\n')
        self.assertEqual(list(rest), ['set decay 23 2.7\n'])


if __name__ == '__main__':
    unittest.main()
