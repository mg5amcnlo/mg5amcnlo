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
"""Syntax of the interference processes

    generate g g > h > t t~ [LIxtree=QCD] g g > t t~     (loop-induced x tree)
    generate p p > z > e+ e- [treextree] p p > a > e+ e-  (tree x tree)

Only the parsing is tested here (the generation is tested in
tests/unit_tests/core/test_tree_interference.py and
tests/unit_tests/loop/test_loop_induced_x_tree.py): the line is split in a
left-hand and a right-hand process, each with its own constraints, the two sides are checked
for consistency, and the command is routed to the right interface. The
keywords are case-insensitive and squared-order constraints are refused on
the right-hand process.
"""

from __future__ import absolute_import

import os
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.core.base_objects as base_objects
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.interface.master_interface as MGCmd

from madgraph import InvalidCmd

split = mg_interface.split_interference_line


class InterferenceLineSplitTest(unittest.TestCase):
    """split_interference_line, without any model."""

    def test_lixtree_with_right_process(self):
        self.assertEqual(split('g g > h > t t~ [LIxtree=QCD] g g > t t~'),
                         ('g g > h > t t~ [noborn= QCD ]', 'LIxtree', 'g g > t t~'))

    def test_lixtree_default_right_process(self):
        self.assertEqual(split('g g > h > t t~ [LIxtree=QCD]'),
                         ('g g > h > t t~ [noborn= QCD ]', 'LIxtree', None))

    def test_keywords_are_case_insensitive(self):
        for keyword in ['LIxtree', 'lixtree', 'LIXTREE', 'LiXTree']:
            self.assertEqual(
                split('g g > h > t t~ [%s=QCD] g g > t t~' % keyword)[1:],
                ('LIxtree', 'g g > t t~'))
        for keyword in ['treextree', 'TreeXTree', 'TREEXTREE']:
            self.assertEqual(
                split('p p > z > e+ e- [%s] p p > a > e+ e-' % keyword),
                ('p p > z > e+ e-', 'treextree', 'p p > a > e+ e-'))

    def test_constraints_go_to_their_side(self):
        """Constraints between ']' and the right-hand process belong to the
        left-hand one, those after it to the right-hand one; '@N' closes the
        whole definition and stays with the left-hand line."""
        self.assertEqual(
            split('g g > h > t t~ [lixtree = QCD] QCD^2==6 QED = 2 '
                  'g g > t t~ QED=0 / b @3'),
            ('g g > h > t t~ [noborn= QCD ] QCD^2==6 QED = 2 @3', 'LIxtree',
             'g g > t t~ QED=0 / b'))

    def test_pdg_code_processes_are_not_constraints(self):
        """An order name is an identifier, and a constraint is followed by a
        process: '25 > 6' or 'h > 5' starting a right-hand decay is no
        'ORDER > n' constraint."""
        self.assertEqual(split('25 > 6 -6 [treextree] 25 > 6 -6'),
                         ('25 > 6 -6', 'treextree', '25 > 6 -6'))
        self.assertEqual(split('h > b b~ [treextree] h > 5 -5'),
                         ('h > b b~', 'treextree', 'h > 5 -5'))
        self.assertEqual(split('u u~ > e+ e- [treextree] QED>2 u u~ > e+ e-'),
                         ('u u~ > e+ e- QED>2', 'treextree', 'u u~ > e+ e-'))
        self.assertEqual(split('g g > h > t t~ [LIxtree=QCD] QCD^2==6'),
                         ('g g > h > t t~ [noborn= QCD ] QCD^2==6', 'LIxtree', None))

    def test_ordinary_lines_are_untouched(self):
        for line in ['g g > z z [noborn=QCD]', 'p p > t t~ [QCD]',
                     'p p > t t~ [real=QCD] QCD^2<=4', 'p p > t t~ QED=0',
                     'p p > e+ e- [LOonly]']:
            self.assertEqual(split(line), (line, '', None))

    def test_malformed_brackets_are_refused(self):
        for line in ['g g > h [LIxtree=] g g > h',
                     'g g > h [LIxtree] g g > h',
                     'p p > z > e+ e- [treextree=QED] p p > a > e+ e-',
                     'p p > z > e+ e- [treextree]',
                     'g g > h > t t~ [LIxtree=QCD] g g > t t~ [QCD]']:
            self.assertRaises(InvalidCmd, split, line)

    def test_routing(self):
        """The master interface sends LIxtree where [noborn=] goes (MadLoop
        model validation, then the loop-induced generation) and treextree to
        the tree-level MadGraph interface."""
        process_type = MGCmd.MasterCmd.extract_process_type
        self.assertEqual(process_type('g g > h > t t~ [LIxtree] g g > t t~'),
                         ('NLO', 'LIxtree', []))
        self.assertEqual(process_type('g g > h > t t~ [LIxtree=QCD] g g > t t~'),
                         ('NLO', 'LIxtree', ['QCD']))
        self.assertEqual(process_type('g g > h > t t~ [lixtree= QCD] QCD^2==6 g g > t t~'),
                         ('NLO', 'LIxtree', ['QCD']))
        self.assertEqual(process_type('p p > z > e+ e- [TreeXTree] p p > a > e+ e-'),
                         ('tree', 'treextree', []))
        # unchanged
        self.assertEqual(process_type('g g > z z [noborn=QCD]'), ('NLO', 'noborn', ['QCD']))
        self.assertEqual(process_type('p p > t t~ [QCD]'), ('NLO', 'all', ['QCD']))
        self.assertEqual(process_type('p p > t t~'), ('tree', None, []))


class _StubModel(object):
    merged_particles = {81: [1, 3, 5]}


class InterferenceLegPdgsTest(unittest.TestCase):
    """Merged (flavour-grouped) legs are compared through their members."""

    def leg(self, ids, flavor=[]):
        return base_objects.MultiLeg({'ids': ids, 'flavor': list(flavor)})

    def test_expansion(self):
        pdgs = mg_interface.MadGraphCmd.interference_leg_pdgs
        self.assertEqual(pdgs(self.leg([81]), _StubModel), set([1, 3, 5]))
        self.assertEqual(pdgs(self.leg([-81]), _StubModel), set([-1, -3, -5]))
        self.assertEqual(pdgs(self.leg([81], [3]), _StubModel), set([3]))
        self.assertEqual(pdgs(self.leg([21, 81, 2]), _StubModel), set([21, 1, 3, 5, 2]))


class InterferenceProcessTest(unittest.TestCase):
    """extract_process on interference lines, with loop_sm."""

    interface = None

    def setUp(self):
        if InterferenceProcessTest.interface is None:
            interface = MGCmd.MasterCmd()
            interface.exec_cmd('import model loop_sm', printcmd=False,
                               precmd=True)
            InterferenceProcessTest.interface = interface
        self.cmd = InterferenceProcessTest.interface

    def test_process_defaults(self):
        """An ordinary process carries no interference."""
        procdef = self.cmd.extract_process('g g > t t~')
        self.assertEqual(procdef['interference_mode'], '')
        self.assertIsNone(procdef['interference_process'])
        self.assertEqual(base_objects.Process()['interference_mode'], '')

    def test_squared_order_selection(self):
        """What the run card treats as an interference: for ordinary
        processes exactly when the process string shows a '^2' (the squared
        order implied by an amplitude '==' is not one), and every interference
        process, whose constraint is hidden."""
        for line, expected in [('u u~ > e+ e-', False),
                               ('p p > j j QED==2', False),
                               ('p p > j j QCD^2==2', True),
                               ('p p > j j QED^2<=2', True),
                               ('u u~ > z > e+ e- [treextree] u u~ > a > e+ e-', True)]:
            procdef = self.cmd.extract_process(line)
            if procdef.get_interference_mode():
                import madgraph.core.diagram_generation as diagram_generation
                diagram_generation.prepare_interference_process(procdef)
            process = next(iter(procdef))
            self.assertEqual(process.has_squared_order_selection(), expected, line)
            if not procdef.get_interference_mode():
                self.assertEqual('^2' in process.nice_string(), expected, line)

    def test_process_from_an_older_pickle(self):
        """A process unpickled from a version without the interference keys
        still prints."""
        process = base_objects.Process()
        for key in ('interference_mode', 'interference_process'):
            dict.__delitem__(process, key)
        self.assertNotIn('interference_mode', str(process))
        self.assertEqual(process.get_interference_mode(), '')

    def test_lixtree(self):
        procdef = self.cmd.extract_process(
            'g g > h > t t~ [LIxtree=QCD] QCD^2==6 g g > t t~ QED=0 / b @2')
        self.assertEqual(procdef['interference_mode'], 'LIxtree')
        self.assertEqual(procdef['NLO_mode'], 'noborn')
        self.assertFalse(procdef['has_born'])
        self.assertEqual(procdef['perturbation_couplings'], ['QCD'])
        self.assertEqual(procdef['squared_orders'], {'QCD': 6})
        self.assertEqual(procdef['id'], 2)
        self.assertEqual(procdef['required_s_channels'], [[25]])

        right = procdef['interference_process']
        self.assertIsInstance(right, base_objects.ProcessDefinition)
        self.assertEqual(right['NLO_mode'], 'tree')
        self.assertTrue(right['has_born'])
        self.assertEqual(right['interference_mode'], '')
        self.assertEqual(right['orders'], {'QED': 0})
        self.assertEqual(right['squared_orders'], {})
        self.assertEqual(right['required_s_channels'], [])
        self.assertEqual(right['forbidden_particles'], [5])
        self.assertEqual(right['id'], 0)
        self.assertEqual([l['ids'] for l in right['legs']],
                         [[21], [21], [6], [-6]])

    def test_lixtree_default_right_process(self):
        """Without a right-hand process, the trees of the same external legs:
        the '> h >' requirement and the loop orders are not carried over."""
        procdef = self.cmd.extract_process('g g > h > t t~ QED=2 [lixtree=QCD]')
        right = procdef['interference_process']
        self.assertEqual([l['ids'] for l in right['legs']],
                         [l['ids'] for l in procdef['legs']])
        self.assertEqual(right['required_s_channels'], [])
        self.assertEqual(right['orders'], {})
        self.assertEqual(right['perturbation_couplings'], [])
        self.assertEqual(right['NLO_mode'], 'tree')

    def test_treextree(self):
        procdef = self.cmd.extract_process(
            'u u~ > z > e+ e- [TreeXTree] QED^2==4 p p > a > e+ e-')
        self.assertEqual(procdef['interference_mode'], 'treextree')
        self.assertEqual(procdef['NLO_mode'], 'tree')
        self.assertEqual(procdef['perturbation_couplings'], [])
        self.assertEqual(procdef['squared_orders'], {'QED': 4})
        self.assertEqual(procdef['required_s_channels'], [[23]])
        right = procdef['interference_process']
        self.assertEqual(right['required_s_channels'], [[22]])

    def test_display_round_trip(self):
        """The printed definition is the input syntax, and parses back to the
        same interference."""
        # (multiparticles are printed as 'u/d/...', which is not input
        # syntax: hence particles only)
        for line in ['g g > h > t t~ [LIxtree=QCD] QCD^2==6 g g > t t~ QED=0',
                     'u u~ > z > e+ e- [treextree] u u~ > a > e+ e- / h']:
            procdef = self.cmd.extract_process(line)
            printed = procdef.nice_string(prefix=False)
            self.assertIn('[ %s' % procdef['interference_mode'], printed)
            again = self.cmd.extract_process(printed)
            self.assertEqual(again['interference_mode'], procdef['interference_mode'])
            self.assertEqual(again['squared_orders'], procdef['squared_orders'])
            self.assertEqual(again['interference_process']['legs'],
                             procdef['interference_process']['legs'])
            self.assertEqual(again['interference_process']['required_s_channels'],
                             procdef['interference_process']['required_s_channels'])

    def test_right_hand_squared_orders_are_refused(self):
        for line in ['g g > h > t t~ [LIxtree=QCD] g g > t t~ QCD^2==4',
                     'u u~ > z > e+ e- [treextree] p p > a > e+ e- QED^2<=4']:
            self.assertRaisesRegex(InvalidCmd, 'right-hand process',
                                   self.cmd.extract_process, line)

    def test_inconsistent_legs_are_refused(self):
        for line in [
                # one more final-state particle
                'g g > h > t t~ [LIxtree=QCD] g g > t t~ g',
                # same legs, other order
                'g g > h > t t~ [LIxtree=QCD] g g > t~ t',
                # other initial state
                'g g > h > t t~ [LIxtree=QCD] u u~ > t t~',
                # a right-hand leg narrower than the left-hand one
                'p p > z > e+ e- [treextree] u u~ > a > e+ e-',
                # other polarisation
                'u u~ > z > e+{L} e- [treextree] u u~ > a > e+ e-']:
            self.assertRaisesRegex(InvalidCmd, 'two sides',
                                   self.cmd.extract_process, line)

    def test_wider_right_hand_legs_are_accepted(self):
        procdef = self.cmd.extract_process(
            'u u~ > z > e+ e- [treextree] p p > a > e+ e-')
        self.assertEqual(procdef['interference_mode'], 'treextree')

    def test_decay_chains_are_refused(self):
        self.assertRaises(InvalidCmd, self.cmd.extract_process,
                    'u u~ > z > e+ e- [treextree] u u~ > a > e+ e-, a > e+ e-')


if __name__ == '__main__':
    unittest.main()
