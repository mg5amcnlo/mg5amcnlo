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
"""Generation of the interference of a loop-induced amplitude with tree-level
ones:

    generate g g > h > t t~ [LIxtree=QCD]                 (right side implied)
    generate g g > h > t t~ [LIxtree=QCD] g g > t t~ QED=0

The tree-level diagrams of the right-hand process are stored among the
'loop_UVCT_diagrams' of the loop-induced amplitude, each with one unit of the
hidden order base_objects.INTERFERENCE_ORDER, and the squared-order constraint
INTERF^2==1 keeps only the loop x tree products. The numerical check of the
result (against the tree-level matrix element, [sqrvirt=] and the original
wiki recipe) is the acceptance test test_loop_induced_x_tree_standalone in
tests/acceptance_tests/test_cmd_madloop.py.
"""

from __future__ import absolute_import

import os
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.core.base_objects as base_objects
import madgraph.interface.master_interface as MGCmd
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.loop.loop_helas_objects as loop_helas_objects

from madgraph import InvalidCmd

INTERF = base_objects.INTERFERENCE_ORDER


def generate(line):
    """A fresh interface (the hidden order is added to its model) with 'line'
    generated."""
    interface = MGCmd.MasterCmd()
    interface.exec_cmd('import model loop_sm', printcmd=False, precmd=True)
    interface.exec_cmd('generate %s' % line, printcmd=False, precmd=True,
                       errorhandling=False)
    return interface


class LoopInducedXTreeGenerationTest(unittest.TestCase):

    gg_h_ttx = None

    def setUp(self):
        if LoopInducedXTreeGenerationTest.gg_h_ttx is None:
            LoopInducedXTreeGenerationTest.gg_h_ttx = \
                                generate('g g > h > t t~ [LIxtree=QCD]')
        self.interface = LoopInducedXTreeGenerationTest.gg_h_ttx

    def test_diagrams(self):
        """4 loops (the top triangles of g g > h, h > t t~) and the 3 QCD
        trees of g g > t t~, which carry one unit of the hidden order."""
        self.assertEqual(len(self.interface._curr_amps), 1)
        amp = self.interface._curr_amps[0]
        self.assertFalse(amp['has_born'])
        self.assertEqual(len(amp['loop_diagrams']), 4)
        self.assertEqual(len(amp['loop_UVCT_diagrams']), 3)
        for loop in amp['loop_diagrams']:
            self.assertEqual(loop.get_order(INTERF), 0)
        for tree in amp['loop_UVCT_diagrams']:
            self.assertIsInstance(tree, loop_base_objects.LoopUVCTDiagram)
            self.assertEqual(tree['UVCT_orders'], {INTERF: 1})
            self.assertEqual(tree['UVCT_couplings'], [1])
            self.assertEqual(tree.get_order(INTERF), 1)
            self.assertEqual(tree.get_order('QCD'), 2)
            self.assertEqual(tree.get_order('QED'), 0)
            # the hidden order does not change the weight
            self.assertEqual(tree.get_order('WEIGHTED'), 2)

    def test_process(self):
        """The left-hand process selects the interference with the hidden
        order, which is not shown; the subprocess keeps the right-hand tree
        process of its own legs, without the '> h >' requirement."""
        process = self.interface._curr_amps[0]['process']
        self.assertEqual(process['squared_orders'][INTERF], 1)
        self.assertEqual(process.get_squared_order_type(INTERF), '==')
        self.assertIn(INTERF, process['split_orders'])
        self.assertIn('QCD', process['split_orders'])
        self.assertNotIn(INTERF, process.nice_string())
        self.assertIn('[ LIxtree = QCD ] g g > t t~', process.nice_string())
        model = process['model']
        self.assertEqual(model['order_hierarchy'][INTERF], 0)
        self.assertIn(INTERF, model['coupling_orders'])

        right = process['interference_process']
        self.assertIsInstance(right, base_objects.Process)
        self.assertEqual([l['id'] for l in right['legs']], [21, 21, 6, -6])
        self.assertEqual(right['required_s_channels'], [])
        # minimal order, as 'generate g g > t t~' would choose
        self.assertEqual(right['orders'], {'WEIGHTED': 2})

    def test_helas_split_orders(self):
        """In the generated code each tree amplitude has INTERF=1 (not 2: the
        UVCT orders are added once), and the squared orders of the three
        products (loop x tree, loop x loop, tree x tree) are all defined, the
        first one being the selected one."""
        me = loop_helas_objects.LoopHelasMatrixElement(
                     self.interface._curr_amps[0], optimized_output=True)
        squared, amp_orders = me.get_split_orders_mapping()
        split_orders = me.get('processes')[0]['split_orders']
        interf = split_orders.index(INTERF)
        self.assertEqual(sorted(so[interf] for so, _ in squared), [0, 1, 2])
        loop_amp_orders = sorted(so[interf] for so, _ in
                                 amp_orders['loop_amp_orders'])
        self.assertEqual(loop_amp_orders, [0, 1])
        for diagram in me.get_loop_UVCT_diagrams():
            for amp in diagram.get_loop_UVCTamplitudes():
                self.assertEqual(amp.get('orders')[INTERF], 1)

    def test_alpha_s_power(self):
        """loop (QCD=2) x tree (QCD=2): alpha_s^2 for every event, which
        madevent writes in config_nqcd.inc for the systematics."""
        import madgraph.iolibs.export_v4 as export_v4
        me = loop_helas_objects.LoopHelasMatrixElement(
                     self.interface._curr_amps[0], optimized_output=True)
        self.assertEqual(export_v4.interference_alpha_s_power([me]), 2)
        self.assertEqual(export_v4.interference_alpha_s_powers([me]), [2])

    def test_orphan_trees_are_dropped(self):
        """Trees whose loop partners are all removed after the squared-order
        selection no longer contribute and are dropped."""
        import copy
        import madgraph.core.base_objects as base_objects
        amp = copy.copy(self.interface._curr_amps[0])
        amp['loop_UVCT_diagrams'] = base_objects.DiagramList(
                                            amp['loop_UVCT_diagrams'])
        self.assertEqual(amp.drop_orphan_interference_trees(), 0)
        self.assertEqual(len(amp['loop_UVCT_diagrams']), 3)
        amp['loop_diagrams'] = base_objects.DiagramList()
        self.assertEqual(amp.drop_orphan_interference_trees(), 3)
        self.assertEqual(len(amp['loop_UVCT_diagrams']), 0)

    def test_split_order_infos_do_not_print_the_hidden_order(self):
        """The debug summary of the squared-order selection leaves the hidden
        order out: loop x tree (INTERF=1) is selected, loop x loop (0) and
        tree x tree (2) are not. The returned combinations keep it."""
        amp = self.interface._curr_amps[0]
        with self.assertLogs('madgraph.loop_diagram_generation',
                             level='DEBUG') as log:
            infos = amp.print_split_order_infos()
        text = '\n'.join(log.output)
        self.assertNotIn(INTERF, text)
        self.assertIn('considered: (QCD,QED,WEIGHTED)', text)
        self.assertIn(' > loop : (4,2,W8)', text)
        self.assertIn('(4,4,W12)', text)
        self.assertIn('(4,0,W4)', text)
        self.assertEqual(infos[2], [(1, 4, 2, 8)])


class LoopInducedXTreeDisplayTest(unittest.TestCase):

    def test_subprocess_string_parses_back(self):
        """The string of a subprocess puts the left-hand exclusions before
        the bracket, so that they stay on the left-hand side when the string
        is read back."""
        interface = generate('g g > h > t t~ / b [LIxtree=QCD] g g > t t~')
        process = interface._curr_amps[0]['process']
        printed = process.nice_string(prefix=False)
        self.assertLess(printed.index('/ b'), printed.index('[ LIxtree'))
        again = interface.extract_process(printed)
        self.assertEqual(again['interference_mode'], 'LIxtree')
        self.assertEqual(again['forbidden_particles'], [5])
        right = again['interference_process']
        self.assertEqual(right['forbidden_particles'], [])
        self.assertEqual([l['ids'] for l in right['legs']],
                         [[21], [21], [6], [-6]])


class LoopInducedXTreeMultiProcessTest(unittest.TestCase):

    def test_subprocesses_without_loop_are_dropped(self):
        """p p > h > t t~: the light-quark channels have trees but no loop,
        so no interference: only g g survives (the trees alone would have
        kept the subprocess alive)."""
        interface = generate('p p > h > t t~ [LIxtree=QCD]')
        self.assertEqual([[l['id'] for l in amp['process']['legs']]
                          for amp in interface._curr_amps],
                         [[21, 21, 6, -6]])

    def test_right_hand_constraints(self):
        """The constraints of the right-hand process apply to the trees only,
        those between ']' and the right-hand process to the interference."""
        interface = generate('g g > h > t t~ [LIxtree=QCD] QCD^2==4 '
                             'g g > t t~ QED=0 / b')
        amp = interface._curr_amps[0]
        self.assertEqual(len(amp['loop_UVCT_diagrams']), 3)
        self.assertEqual(amp['process']['squared_orders'],
                         {'QCD': 4, INTERF: 1})
        right = amp['process']['interference_process']
        self.assertEqual(right['orders'], {'QED': 0})
        self.assertEqual(right['forbidden_particles'], [5])

    def test_no_tree_level_interference(self):
        """g g > z z has no tree-level diagram: nothing to interfere with."""
        self.assertRaisesRegex(InvalidCmd, 'no interference',
                               generate, 'g g > h > z z [LIxtree=QCD]')

    def test_later_nlo_process_ignores_the_hidden_order(self):
        """The hidden order the interference process added to the model is
        not one of the orders of a later NLO process: neither constrained,
        split nor perturbed."""
        interface = generate('g g > h > t t~ [LIxtree=QCD]')
        interface.exec_cmd('generate u u~ > e+ e- [QCD]', printcmd=False,
                           precmd=True, errorhandling=False)
        procdef = interface._curr_proc_defs[-1]
        self.assertEqual(sorted(procdef['perturbation_couplings']),
                         ['QCD', 'QED'])
        self.assertNotIn(INTERF, procdef['split_orders'])
        self.assertNotIn(INTERF, procdef['squared_orders'])
        self.assertNotIn(INTERF, procdef.nice_string())


if __name__ == '__main__':
    unittest.main()
