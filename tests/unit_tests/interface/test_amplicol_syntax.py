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
"""Syntax of the processes computed by AmpliCol

    generate p p > j j j [amplicol]

The process is kept as a definition for the mg7 output, which generates its
matrix elements with pyamplicol; no MG diagrams are generated for it.
"""

import os
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.interface.madgraph_interface as mg_interface
import madgraph.interface.master_interface as MGCmd

from madgraph import InvalidCmd

split = mg_interface.split_backend_line


class BackendLineSplitTest(unittest.TestCase):
    """split_backend_line, without any model."""

    def test_split(self):
        self.assertEqual(split('p p > j j j [amplicol]'), ('p p > j j j', 'amplicol'))
        self.assertEqual(split('p p > j j j [ AmpliCol ] @2'),
                         ('p p > j j j @2', 'amplicol'))
        self.assertEqual(split('p p > j j j'), ('p p > j j j', ''))
        self.assertEqual(split('p p > t t~ [QCD]'), ('p p > t t~ [QCD]', ''))

    def test_other_brackets_are_refused(self):
        self.assertRaises(InvalidCmd, split, 'p p > j j [amplicol] [QCD]')
        self.assertRaises(InvalidCmd, split, 'p p > j j [QCD] [amplicol]')
        # the pre-check must not route it to aMC@NLO with 'amplicol' as order
        process_type = MGCmd.MasterCmd.extract_process_type
        self.assertRaises(InvalidCmd, process_type, 'p p > t t~ [QCD] [amplicol]')

    def test_process_type(self):
        process_type = MGCmd.MasterCmd.extract_process_type
        self.assertEqual(process_type('p p > j j j [amplicol]'), ('tree', None, []))
        self.assertEqual(process_type('p p > t t~ [QCD]'), ('NLO', 'all', ['QCD']))


class AmplicolProcessTest(unittest.TestCase):
    """generate/add process with [amplicol], with the sm model."""

    def setUp(self):
        self.cmd = MGCmd.MasterCmd()
        self.cmd.exec_cmd('import model sm', printcmd=False, precmd=True)

    def test_no_diagrams_are_generated(self):
        self.cmd.exec_cmd('generate g g > g g g [amplicol]', printcmd=False,
                          precmd=True)
        self.cmd.exec_cmd('add process p p > e+ e-', printcmd=False, precmd=True)
        self.assertEqual(len(self.cmd._curr_amplicol_defs), 1)
        self.assertEqual(
            self.cmd._curr_amplicol_defs[0].nice_string(prefix=False).split('@')[0].strip(),
            'g g > g g g')
        # the madmatrix process keeps its own process number
        self.assertTrue(self.cmd._curr_amps)
        self.assertEqual(
            {amp.get('process').get('id') for amp in self.cmd._curr_amps}, {2})
        self.cmd.exec_cmd('generate p p > e+ e-', printcmd=False, precmd=True)
        self.assertEqual(len(self.cmd._curr_amplicol_defs), 0)

    def test_display_processes(self):
        """display processes lists the [amplicol] processes, also when they
        are the only ones; they have no diagrams to display"""
        import contextlib, io
        self.cmd.exec_cmd('generate g g > g g [amplicol]', printcmd=False,
                          precmd=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.cmd.do_display('processes')
        self.assertIn('g g > g g @1 [amplicol]', out.getvalue())
        self.assertRaisesRegex(InvalidCmd, 'amplicol', self.cmd.do_display,
                               'diagrams_text')

    def test_unsupported_options_are_refused(self):
        for line in ['p p > j j QED=0 [amplicol]',
                     'p p > z > e+ e- [amplicol]',
                     'p p > j j / b [amplicol]',
                     'z > e+ e- [amplicol]']:
            self.assertRaises(InvalidCmd, self.cmd.do_generate, line)


class AmplicolExportTest(unittest.TestCase):
    """The process definitions as pyamplicol sees them, without pyamplicol."""

    def setUp(self):
        self.cmd = MGCmd.MasterCmd()
        self.cmd.exec_cmd('import model sm', printcmd=False, precmd=True)
        self.cmd.exec_cmd('define q = u d', printcmd=False, precmd=True)
        self.cmd.exec_cmd('define qx = u~ d~', printcmd=False, precmd=True)

    def export(self, line):
        import madgraph.iolibs.export_amplicol as export_amplicol
        self.cmd.exec_cmd('generate %s [amplicol]' % line, printcmd=False,
                          precmd=True)
        procdef = self.cmd._curr_amplicol_defs[0]
        model = self.cmd._curr_model
        return (export_amplicol.process_name(procdef, model,
                                             self.cmd._multiparticles),
                export_amplicol.process_expression(procdef, model),
                export_amplicol.representative_processes([procdef], model))

    def test_flavour_groups_are_expanded(self):
        """MG replaces the light quarks by a flavour group, which pyamplicol
        does not know: the original flavours are passed on."""
        name, expression, processes = self.export('u u~ > e+ e- g')
        self.assertEqual(name, 'uux_epemg')
        self.assertEqual(expression, ('u u~ > e+ e- g', {}))
        self.assertEqual(len(processes), 1)
        name, expression, processes = self.export('q qx > e+ e- g')
        self.assertEqual(name, 'qqx_epemg')
        self.assertEqual(expression, ('mp0 mp1 > e+ e- g',
                                      {'mp0': ['u', 'd'], 'mp1': ['u~', 'd~']}))
        self.assertEqual(len(processes), 4)

    def test_multiparticles(self):
        name, (expression, multiparticles), processes = self.export('p p > j j')
        self.assertEqual(name, 'pp_jj')
        self.assertEqual(expression, 'mp0 mp0 > mp0 mp0')
        self.assertEqual(sorted(multiparticles['mp0']),
                         sorted(['g', 'u', 'd', 's', 'c', 'u~', 'd~', 's~', 'c~']))
        # every initial state, and every other particle of each final-state leg
        self.assertEqual(len(processes), 81 + 2 * 8)

    def test_coupling_orders(self):
        """The orders MG selects by default, as the largest non-QCD orders."""
        import madgraph.iolibs.export_amplicol as export_amplicol
        for line, orders in [('p p > j j j', {'QED': 0}),
                             ('p p > e+ e- j', {'QED': 2}),
                             ('p p > w+ j j', {'QED': 1})]:
            self.cmd.exec_cmd('generate %s [amplicol]' % line, printcmd=False,
                              precmd=True)
            self.assertEqual(export_amplicol.coupling_orders(
                self.cmd._curr_amplicol_defs[0], self.cmd._curr_model), orders)

    def test_default_histograms_see_every_final_state_particle(self):
        """The representative processes cover every particle of a final-state
        leg: the default histograms of p p > e+ e- j j are of jets, which
        every subprocess has, not of gluons, which most of them lack."""
        import madgraph.iolibs.export_amplicol as export_amplicol
        import madgraph.various.banner as banner_mod
        self.cmd.exec_cmd('generate p p > e+ e- j j [amplicol]', printcmd=False,
                          precmd=True)
        processes = export_amplicol.representative_processes(
            self.cmd._curr_amplicol_defs, self.cmd._curr_model)
        run_card = banner_mod.RunCardMG7()
        run_card.create_default_for_process(
            {'ninitial': 2, 'nexternal': 6}, '', [processes])
        histograms = list(run_card.dynamic_sections['histograms'])
        self.assertIn('jet_2-pt', histograms)
        self.assertFalse([name for name in histograms if name.startswith('g_')])

    def test_mirrored_beams(self):
        """pyamplicol keeps one process per unordered initial state; the other
        beam order is integrated as the mirror where the definition allows it."""
        import madgraph.iolibs.export_amplicol as export_amplicol
        mirrored = export_amplicol.mirrored
        partons = {21, 1, 2, -1, -2}
        exported = {((21, 2), (2, 21, 21)), ((2, 2), (2, 2, 21)),
                    ((2, -2), (-2, 2, 21)), ((-2, 2), (-2, 2, 21))}
        self.assertTrue(mirrored([21, 2, 21, 21, 2], 2, [partons, partons], exported))
        self.assertFalse(mirrored([2, 2, 2, 2, 21], 2, [partons, partons], exported))
        # both beam orders are exported
        self.assertFalse(mirrored([2, -2, 2, -2, 21], 2, [partons, partons], exported))
        # g u > u g g only
        self.assertFalse(mirrored([21, 2, 21, 21, 2], 2, [{21}, {2}], exported))

    def test_selector_rows_are_folded_into_weighted_options(self):
        """The rows of a UMAMI selector share its evaluation: they become the
        options of one flavour per mirror flag, written in proportion to their
        factors, which add up to the flavour factor."""
        import madgraph.iolibs.export_amplicol as export_amplicol
        metadata = {
            'pdg_ids': [[1, -1, 2, -2, 21], [1, -1, 3, -3, 21],
                        [1, -1, 4, -4, 21], [-1, 1, 3, -3, 21]],
            'provider': {'incoming_count': 2, 'masses': [0.] * 5,
                         'supports_alpha_s': True},
            'channels': [{
                'phasespace_order': [0, 2, 3, 4, 1],
                'phasespace_channels': [
                    {'phasespace_order': [0, 2, 3, 4, 1], 'resonances': []}],
                'processes': [
                    {'multichannels': [0], 'color_flows': 0, 'matrix_elements': [
                        {'pdg_ids': 0, 'factor': 1}, {'pdg_ids': 2, 'factor': 2},
                        {'pdg_ids': 1, 'factor': 1}]},
                    {'multichannels': [0], 'color_flows': 1, 'matrix_elements': [
                        {'pdg_ids': 3, 'factor': 1}]},
                ],
            }],
            'color_flows': [], 'helicities': [], 'runtime_processes': [],
        }
        partons = {21, 1, 2, 3, 4, -1, -2, -3, -4}
        entries = export_amplicol.subprocess_entries(
            metadata, self.cmd._curr_model, 'SubProcesses/A1', 'me.so', 'ratio.so',
            3, [partons, partons])
        self.assertEqual(len(entries), 1)
        flavors = entries[0]['flavors']
        # d d~ > s s~ g has its mirror exported, the other rows are mirrored
        common = {'channels': [0], 'resonance_pdgs': [[]]}
        self.assertEqual(flavors, [
            {'index': 0, 'options': [[1, -1, 2, -2, 21], [1, -1, 4, -4, 21]],
             'weights': [1, 2], 'factor': 3, 'mirror': True, 'color_index': 0,
             **common},
            {'index': 0, 'options': [[1, -1, 3, -3, 21]], 'weights': [1],
             'factor': 1, 'mirror': False, 'color_index': 0, **common},
            {'index': 1, 'options': [[-1, 1, 3, -3, 21]], 'weights': [1],
             'factor': 1, 'mirror': False, 'color_index': 1, **common},
        ])

    def test_resonance_charges_split_the_selector_rows(self):
        """A selector integrated over phase-space channels with resonances
        writes them with the charge of the decay products of each row: the
        charge-conjugated rows of a W selector become flavours of their own."""
        import madgraph.iolibs.export_amplicol as export_amplicol
        metadata = {
            'pdg_ids': [[2, -1, -11, 12, 21], [4, -3, -11, 12, 21],
                        [1, -2, 11, -12, 21]],
            'provider': {'incoming_count': 2, 'masses': [0.] * 5,
                         'supports_alpha_s': True},
            'channels': [{
                'phasespace_order': [0, 4, 1, 2, 3],
                'phasespace_channels': [
                    {'phasespace_order': [0, 4, 1, 2, 3], 'resonances': []},
                    {'phasespace_order': [0, 4, 1, 2, 3],
                     'resonances': [{'pdg': 24, 'legs': [2, 3]}]}],
                'colored_resonances': [{'pdg': 6, 'legs': [2, 3, 4], 'color': 3}],
                'processes': [
                    {'multichannels': [0, 1], 'color_flows': 0,
                     'matrix_elements': [{'pdg_ids': index, 'factor': 1}
                                         for index in range(3)]},
                ],
            }],
            'color_flows': [], 'helicities': [], 'runtime_processes': [],
        }
        partons = {21, 1, 2, 3, 4, -1, -2, -3, -4}
        entries = export_amplicol.subprocess_entries(
            metadata, self.cmd._curr_model, 'SubProcesses/A1', 'me.so', 'ratio.so',
            3, [partons, partons])
        flavors = entries[0]['flavors']
        self.assertEqual([f['options'] for f in flavors], [
            [[2, -1, -11, 12, 21], [4, -3, -11, 12, 21]], [[1, -2, 11, -12, 21]]])
        self.assertEqual([f['resonance_pdgs'] for f in flavors],
                         [[[], [24]], [[], [-24]]])
        self.assertEqual(entries[0]['phasespace_channels'][1],
                         {'order': [0, 4, 1, 2, 3],
                          'resonances': [{'pdg': 24, 'legs': [2, 3]}]})
        # the launch rejects a coloured resonance that goes on shell
        self.assertEqual(entries[0]['colored_resonances'],
                         [{'pdg': 6, 'legs': [2, 3, 4], 'color': 3}])


class AmplicolLaunchResonancesTest(unittest.TestCase):
    """The s-channel resonances of an [amplicol] subprocess are fixed at output;
    the launch keeps those that go on shell with the param card values."""

    # q q~ > e- e+ mu- mu+ through Z Z, Z and H > Z Z
    order = [0, 2, 3, 4, 5, 1]
    z_e, z_mu = {'pdg': 23, 'legs': [2, 3]}, {'pdg': 23, 'legs': [4, 5]}
    h = {'pdg': 25, 'legs': [2, 3, 4, 5]}

    def setUp(self):
        try:
            from madgraph.mg7 import subprocess_amplicol
        except ImportError:
            self.skipTest('madspace is not installed')
        self.module = subprocess_amplicol
        self.meta = {
            'incoming': [2, -2], 'outgoing': [11, -11, 13, -13],
            'phasespace_channels': [
                {'order': self.order, 'resonances': []},
                {'order': self.order, 'resonances': [self.z_e, self.z_mu]},
                {'order': self.order, 'resonances': [self.h, self.z_e]},
                {'order': self.order, 'resonances': [self.h]},
            ],
            'colored_resonances': [{'pdg': 6, 'legs': [2, 3], 'color': 3}],
            'flavors': [
                {'channels': [0, 1, 2, 3],
                 'resonance_pdgs': [[], [23, 23], [25, 23], [25]]},
                {'channels': [3, 1], 'resonance_pdgs': [[25], [23, 23]]},
            ],
        }

    def active(self, masses, widths):
        return self.module.active_resonances(
            self.meta, lambda pid: masses.get(pid, 0.), lambda pid: widths.get(pid, 0.))

    def test_all_resonances_on_shell(self):
        meta = self.active({23: 91., 25: 125.}, {23: 2.5, 25: 4e-3})
        self.assertEqual(meta['phasespace_channels'], self.meta['phasespace_channels'])
        self.assertEqual(meta['flavors'], self.meta['flavors'])

    def test_stable_resonance_is_dropped_and_channels_kept_apart(self):
        # without a Higgs width, H > Z Z becomes Z and H the non-resonant
        # channel: each still the channel of a copy of the flavours, so kept
        meta = self.active({23: 91., 25: 125.}, {23: 2.5})
        self.assertEqual(meta['phasespace_channels'], [
            {'order': self.order, 'resonances': []},
            {'order': self.order, 'resonances': [self.z_e, self.z_mu]},
            {'order': self.order, 'resonances': [self.z_e]},
            {'order': self.order, 'resonances': []},
        ])
        self.assertEqual([(f['channels'], f['resonance_pdgs']) for f in meta['flavors']],
                         [([0, 1, 2, 3], [[], [23, 23], [23], []]),
                          ([3, 1], [[], [23, 23]])])
        diagrams, masks = self.module.lhe_diagrams(meta)
        self.assertEqual(diagrams, [(0, ()), (1, (23, 23)), (2, (23,)), (3, ())])
        self.assertEqual(masks, [[1., 1., 1., 1.], [0., 1., 0., 1.]])

    def test_resonance_below_its_decay_threshold_is_dropped(self):
        # mu+ mu- of 2 x 70 GeV: neither Z > mu+ mu- nor H > 4l are on shell
        meta = self.active({13: 70., 23: 91., 25: 125.}, {23: 2.5, 25: 4e-3})
        self.assertEqual(
            [c['resonances'] for c in meta['phasespace_channels']],
            [[], [self.z_e], [self.z_e], []])
        self.assertEqual([f['channels'] for f in meta['flavors']], [[0, 1, 2, 3], [3, 1]])

    def test_coloured_resonance_on_shell_is_an_error(self):
        masses, widths = {6: 173., 23: 91.}, {23: 2.5}
        self.active(masses, widths)
        widths[6] = 1.5
        with self.assertRaisesRegex(RuntimeError, 'coloured s-channel resonance 6'):
            self.active(masses, widths)
