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
"""Generation of the interference of two tree-level amplitude sets:

    generate u u~ > z > e+ e- [treextree] u u~ > a > e+ e-

The diagrams of the right-hand process, generated with the legs of each
subprocess, are added to those of the left-hand one with one unit of the
hidden order base_objects.INTERFERENCE_ORDER, and the squared-order constraint
INTERF^2==1 keeps 2 Re(A_left A_right^*) only. The numerical check (against
|A_Z + A_a|^2 - |A_Z|^2 - |A_a|^2) is the acceptance test
test_tree_x_tree_standalone in tests/acceptance_tests/test_loop_induced_x_tree.py.
"""

from __future__ import absolute_import

import contextlib
import io
import os
import re
import shutil
import sys
import tempfile

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.interface.master_interface as MGCmd

from madgraph import InvalidCmd

INTERF = base_objects.INTERFERENCE_ORDER


def generate(line, model='sm'):
    interface = MGCmd.MasterCmd()
    interface.exec_cmd('import model %s' % model, printcmd=False, precmd=True)
    interface.exec_cmd('generate %s' % line, printcmd=False, precmd=True,
                       errorhandling=False)
    return interface


class TreeInterferenceGenerationTest(unittest.TestCase):

    def test_diagrams_and_orders(self):
        interface = generate('u u~ > z > e+ e- [treextree] u u~ > a > e+ e-')
        self.assertEqual(len(interface._curr_amps), 1)
        amp = interface._curr_amps[0]
        diagrams = amp.get('diagrams')
        self.assertEqual(len(diagrams), 2)
        # left-hand (Z) diagram first, then the tagged right-hand (photon) one
        self.assertEqual([d.get_order(INTERF) for d in diagrams], [0, 1])
        self.assertEqual([d.get_order('WEIGHTED') for d in diagrams], [4, 4])

        process = amp.get('process')
        self.assertEqual(process['squared_orders'], {INTERF: 1})
        self.assertEqual(process['split_orders'], [INTERF])
        self.assertNotIn(INTERF, process.nice_string())
        right = process['interference_process']
        self.assertIsInstance(right, base_objects.Process)
        self.assertEqual(right['required_s_channels'], [[22]])
        self.assertEqual(right['id'], 0)

        # the tag reaches the HELAS amplitudes (on a copy: the photon vertex
        # of the model is untouched)
        me = helas_objects.HelasMatrixElement(amp)
        squared, amp_orders = me.get_split_orders_mapping()
        self.assertEqual(sorted(so[0][0] for so in amp_orders), [0, 1])
        self.assertEqual(sorted(so[0] for so in squared), [0, 1, 2])
        model = process['model']
        for inter in model['interactions']:
            self.assertNotIn(INTERF, inter['orders'])

    def test_right_hand_constraints_apply_to_the_right_only(self):
        """'/ z' on the right does not remove the left-hand Z diagram; the
        left-hand squared-order constraint applies to the product."""
        interface = generate('u u~ > e+ e- / h [treextree] QED^2==4 '
                             'u u~ > e+ e- / h z')
        diagrams = interface._curr_amps[0].get('diagrams')
        # left: photon and Z; right: photon only
        self.assertEqual(sorted(d.get_order(INTERF) for d in diagrams),
                         [0, 0, 1])

    def test_multiprocess_without_crossing_reuse(self):
        """Every subprocess gets the right-hand diagrams of its own legs."""
        interface = generate('p p > z > e+ e- [treextree] p p > a > e+ e-')
        self.assertTrue(interface._curr_amps)
        for amp in interface._curr_amps:
            right = amp.get('process')['interference_process']
            self.assertEqual([l['id'] for l in right['legs']],
                             [l['id'] for l in amp.get('process')['legs']])
            self.assertEqual(sorted(d.get_order(INTERF)
                                    for d in amp.get('diagrams')), [0, 1])

    def test_left_hand_amplitude_constraints(self):
        """'==' on the left constrains the left-hand amplitudes only: it does
        not become a squared-order constraint on the product (QED^2==0 would
        remove every QCD x EW product here)."""
        interface = generate('u d > u d QED==0 [treextree] u d > u d QCD==0')
        amp = interface._curr_amps[0]
        process = amp.get('process')
        self.assertEqual(process['squared_orders'], {INTERF: 1})
        self.assertEqual(process['constrained_orders'], {'QED': (0, '==')})
        sides = [(d.get_order(INTERF), d.get_order('QED')) for d in amp.get('diagrams')]
        self.assertTrue(sides)
        self.assertTrue(all(qed == 0 for side, qed in sides if side == 0))
        self.assertTrue(all(qed == 2 for side, qed in sides if side == 1))
        self.assertEqual(set(side for side, _ in sides), set([0, 1]))

    def test_user_diagram_filter_applies_to_both_sides(self):
        """'--diagram_filter' runs the user filter on the right-hand diagrams
        too."""
        import madgraph.core.diagram_generation as diagram_generation
        filtered = []
        original = diagram_generation.Amplitude.apply_user_filter
        def record(amplitude, diag_list, diagram_filter=True):
            filtered.append(amplitude.get('process').get('required_s_channels'))
            return diag_list
        diagram_generation.Amplitude.apply_user_filter = record
        try:
            generate('u u~ > z > e+ e- [treextree] u u~ > a > e+ e- '
                     '--diagram_filter')
        finally:
            diagram_generation.Amplitude.apply_user_filter = original
        self.assertIn([[23]], filtered)
        self.assertIn([[22]], filtered)

    def test_plugin_diagram_filter_function(self):
        """A callable diagram_filter (--diagram_filter=PLUGINNAME) is used
        as the remove_diag function, instead of PLUGIN/user_filter.py."""
        import madgraph.core.diagram_generation as diagram_generation
        seen = []
        def remove_all(diag, model=None):
            seen.append(diag)
            return True
        amp = diagram_generation.Amplitude()
        amp.set('process', base_objects.Process())
        diags = base_objects.DiagramList([base_objects.Diagram()])
        res = amp.apply_user_filter(diags, remove_all)
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(res), 0)

    def test_hidden_order_does_not_leak(self):
        """Once registered by an interference process, the hidden order
        neither appears in the diagrams of later ordinary processes nor adds a
        level to the coupling hierarchy used for the default orders."""
        interface = MGCmd.MasterCmd()
        interface.exec_cmd('import model sm', printcmd=False, precmd=True)
        model = interface._curr_model
        before = model.get_particles_hierarchy()
        interface.exec_cmd('generate u u~ > z > e+ e- [treextree] u u~ > a > e+ e-',
                           printcmd=False, precmd=True, errorhandling=False)
        self.assertIn(INTERF, model.get('coupling_orders'))
        self.assertEqual(model.get_particles_hierarchy(), before)
        interface.exec_cmd('generate u u~ > e+ e-', printcmd=False,
                           precmd=True, errorhandling=False)
        for diagram in interface._curr_amps[0].get('diagrams'):
            self.assertNotIn(INTERF, diagram.get('orders'))

    def test_hidden_order_is_never_printed(self):
        """Registered in the model by an interference process, the hidden
        order (hierarchy 0) is neither part of the WEIGHTED definition printed
        by the minimal-order search, nor listed by 'display coupling_order',
        nor one of the orders that default_unset_couplings sets."""
        interface = MGCmd.MasterCmd()
        interface.exec_cmd('import model sm', printcmd=False, precmd=True)
        # (the left-hand process goes through WEIGHTED<=2 and 3 before 4)
        with self.assertLogs('madgraph.diagram_generation', level='INFO') as log:
            interface.exec_cmd('generate u d > u d [treextree] u d > u d QCD=0',
                               printcmd=False, precmd=True, errorhandling=False)
        self.assertEqual(interface._curr_model['order_hierarchy'][INTERF], 0)
        searches = [line for line in log.output
                    if 'Trying coupling order WEIGHTED' in line]
        self.assertTrue(searches)
        for line in searches:
            self.assertEqual(sorted(line.rsplit(' IS ', 1)[1].split('+')),
                             ['2*QED', 'QCD'])
            self.assertIn('WEIGHTED IS ', line)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            interface.exec_cmd('display coupling_order', printcmd=False,
                               precmd=True, errorhandling=False)
        self.assertIn('QED : weight = 2', stdout.getvalue())
        self.assertNotIn(INTERF, stdout.getvalue())

        interface.options['default_unset_couplings'] = 0
        interface.exec_cmd('generate u u~ > e+ e- QED=2', printcmd=False,
                           precmd=True, errorhandling=False)
        process = interface._curr_amps[0].get('process')
        self.assertEqual(process.get('orders'), {'QED': 2, 'QCD': 0})

    def test_never_grouped_with_an_ordinary_process(self):
        """Same '@N', same legs: still two subprocess groups, since the
        events of a group share the power of alpha_s of each channel."""
        import madgraph.iolibs.group_subprocs as group_subprocs
        import madgraph.core.diagram_generation as diagram_generation
        interface = generate('u u~ > z > e+ e- [treextree] u u~ > a > e+ e- @1')
        interface.exec_cmd('add process u u~ > z > e+ e- @1', printcmd=False,
                           precmd=True, errorhandling=False)
        amps = diagram_generation.AmplitudeList(interface._curr_amps)
        self.assertEqual(len(amps), 2)
        classes = group_subprocs.SubProcessGroup.find_process_classes(amps,
                                                                 'madevent')
        self.assertNotEqual(classes[0], classes[1])

    def test_alpha_s_power_per_subprocess(self):
        """One power per matrix element: None for an ordinary process (its
        channels keep the QCD order of their diagram), the power of the
        left x right products for an interference."""
        import madgraph.iolibs.export_v4 as export_v4
        interface = generate('u d > u d QED=0 [treextree] u d > u d QCD=0')
        interference = helas_objects.HelasMatrixElement(interface._curr_amps[0])
        interface.exec_cmd('add process u d > u d', printcmd=False,
                           precmd=True, errorhandling=False)
        ordinary = helas_objects.HelasMatrixElement(interface._curr_amps[-1])
        self.assertEqual(export_v4.interference_alpha_s_powers(
                                     [ordinary, interference]), [None, 1])

    def test_no_interference(self):
        self.assertRaisesRegex(InvalidCmd, 'no interference',
                               generate,
                               'u u~ > z > e+ e- [treextree] u u~ > h > e+ e-')


class TreeInterferenceOutputTest(unittest.TestCase):
    """Only the formats selecting squared split orders return the
    interference alone (madevent, standalone_fortran, mg7, standalone): the
    others are refused before anything is written."""

    def test_madmatrix_formats_are_accepted(self):
        """mg7 and the C++ standalone: the madmatrix color sum pairs the
        amplitude split orders (CPU and GPU backends)."""
        interface = generate('u u~ > z > e+ e- [treextree] u u~ > a > e+ e-')
        tmpdir = tempfile.mkdtemp(prefix='treextree_')
        try:
            for fmt in ['mg7', 'standalone']:
                out = os.path.join(tmpdir, fmt)
                interface.exec_cmd('output %s %s -f' % (fmt, out),
                                   printcmd=False, precmd=True,
                                   errorhandling=False)
                self.assertTrue(os.path.isdir(os.path.join(out, 'SubProcesses')),
                                fmt)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_unsupported_formats_are_refused(self):
        interface = generate('u u~ > z > e+ e- [treextree] u u~ > a > e+ e-')
        tmpdir = tempfile.mkdtemp(prefix='treextree_')
        try:
            for fmt in ['matchbox', 'matrix', 'mg7_v5']:
                out = os.path.join(tmpdir, fmt)
                self.assertRaisesRegex(InvalidCmd, 'cannot select the interference',
                                       interface.exec_cmd,
                                       'output %s %s -f' % (fmt, out),
                                       printcmd=False, precmd=True,
                                       errorhandling=False)
                self.assertFalse(os.path.exists(out))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


    def test_madevent_output(self):
        """madevent: the interference defaults of the run card (HT/2 scale,
        no systematics by default) and one power of alpha_s for all the
        channels, the one of the product of a left- and a right-hand
        amplitude: QCD (QCD=2) x electroweak (QCD=0) is alpha_s^1, whereas
        the channels of the two sides have QCD orders 2 and 0."""
        interface = generate('u d > u d QED=0 [treextree] u d > u d QCD=0')
        tmpdir = tempfile.mkdtemp(prefix='treextree_')
        try:
            out = os.path.join(tmpdir, 'me')
            interface.exec_cmd('output madevent %s -f' % out, printcmd=False,
                               precmd=True, errorhandling=False)
            run_card = open(os.path.join(out, 'Cards', 'run_card.dat')).read()
            self.assertRegex(run_card, r'\n\s*3\s*=\s*dynamical_scale_choice')
            self.assertRegex(run_card, r'\n\s*False\s*=\s*use_syst')
            self.assertRegex(run_card, r'\n\s*none\s*=\s*systematics_program')
            sub = os.path.join(out, 'SubProcesses')
            powers = set()
            for proc_dir in os.listdir(sub):
                if not proc_dir.startswith('P'):
                    continue
                text = open(os.path.join(sub, proc_dir, 'config_nqcd.inc')).read()
                powers.update(re.findall(r'NQCD\(\d+\)/(\d+)/', text))
            self.assertEqual(powers, set(['1']))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
