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
"""The integrator behind ``compute_widths``: madspace (default) or madevent.

The wiring tests stub the integrations, so they run without a compiled
madspace; the test of the alpha_s of a decay needs the launcher module and is
skipped when madspace is not installed. The comparison of the two integrators
on real processes is in tests/acceptance_tests/test_auto_width.py.
"""

from __future__ import absolute_import

import copy
import json
import logging
import math
import os
import shutil
import tempfile
import unittest
from unittest import mock

import madgraph
import madgraph.core.base_objects as base_objects
import madgraph.interface.madevent_interface as madevent_interface
import madgraph.interface.master_interface as mgcmd
from madgraph.mg7 import bootstrap as mg7_bootstrap
from madgraph.models.model_reader import Alphas_Runner

pjoin = os.path.join


class StopAtMadEvent(Exception):
    """Raised by the stubbed exec_cmd when the madevent integration starts."""


class AutoWidthEngineTest(unittest.TestCase):
    """Option, argument and dispatch of compute_widths."""

    @classmethod
    def setUpClass(cls):
        cls.cmd = mgcmd.MasterCmd()
        cls.cmd.no_notification()
        cls.cmd.exec_cmd('import model sm')

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='test_auto_width')
        self.card = pjoin(self.tmp, 'param_card.dat')
        with open(self.card, 'w') as stream:
            stream.write(self.cmd._curr_model.write_param_card())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --------------------------------------------------------------- arguments

    def check(self, line):
        return self.cmd.check_compute_widths(self.cmd.split_arg(line))[1]

    def test_engine_argument(self):
        self.assertIsNone(self.check('h --path=%s' % self.card)['engine'])
        self.assertEqual(self.check('h --path=%s --engine=madevent' % self.card)['engine'],
                         'madevent')
        self.assertEqual(self.check('h --path=%s --engine=MadSpace' % self.card)['engine'],
                         'madspace')

    def test_unknown_engine_argument(self):
        with self.assertRaises(madgraph.InvalidCmd):
            self.check('h --path=%s --engine=fortran' % self.card)

    # ---------------------------------------------------------------- dispatch

    def run_compute_widths(self, line, madspace=None):
        """compute_widths with both integrations stubbed.

        Returns the stub of the madspace integration, the stub of the card
        update and the lines of the warnings. The madevent integration is
        recognised by its 'output madevent', which stops the run."""
        if madspace is None:
            madspace = mock.Mock(return_value={25: [[[5, -5], 4.3e-3]]})

        def exec_cmd(command, *args, **opts):
            if command.startswith('output madevent'):
                raise StopAtMadEvent()

        def fake_decay_diagram(*args, **opts):
            self.cmd._curr_amps = ['amplitude']

        warnings = []
        handler = logging.Handler()
        handler.emit = lambda record: warnings.append(record.getMessage())
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger('cmdprint')
        logger.addHandler(handler)
        # another test may have muted the logger (MuteLogger, a level, ...)
        level, disabled = logger.level, logger.disabled
        logger.setLevel(logging.WARNING)
        logger.disabled = False
        update = mock.Mock()
        try:
            with mock.patch.object(self.cmd, 'do_decay_diagram', fake_decay_diagram), \
                 mock.patch.object(self.cmd, 'integrate_widths_madspace', madspace), \
                 mock.patch.object(self.cmd, 'exec_cmd', exec_cmd), \
                 mock.patch.object(madevent_interface.MadEventCmd,
                                   'update_width_in_param_card', update):
                result = self.cmd.do_compute_widths(
                    'h --body_decay=3 --path=%s %s' % (self.card, line))
        finally:
            logger.removeHandler(handler)
            logger.setLevel(level)
            logger.disabled = disabled
        return madspace, update, warnings, result

    def test_default_integrates_with_madspace(self):
        madspace, update, _, result = self.run_compute_widths('')
        self.assertEqual(madspace.call_count, 1)
        self.assertEqual(result, {25: [[[5, -5], 4.3e-3]]})
        # the model's two-body formulas write the card first, when it has any;
        # the madspace widths are what the card finally gets
        update.assert_called()
        self.assertEqual(update.call_args[0][0], {25: [[[5, -5], 4.3e-3]]})

    def test_the_argument_selects_madevent(self):
        madspace = mock.Mock()
        with self.assertRaises(StopAtMadEvent):
            self.run_compute_widths('--engine=madevent', madspace)
        madspace.assert_not_called()

    def test_the_argument_selects_madspace(self):
        madspace, _, _, _ = self.run_compute_widths('--engine=madspace')
        self.assertEqual(madspace.call_count, 1)

    def test_there_is_no_engine_option(self):
        """The engine is chosen by the argument of compute_widths only."""
        self.assertNotIn('auto_width_engine', self.cmd.options_madgraph)
        self.assertNotIn('auto_width_engine', self.cmd._set_options)
        self.assertFalse(hasattr(self.cmd, 'set2_auto_width_engine'))

    def test_the_engine_is_not_given_to_decay_diagram(self):
        """decay_diagram shares the arguments of compute_widths but would not
        understand an --engine it was handed back."""
        seen = []

        def fake_decay_diagram(line, *args, **opts):
            seen.append(line)
            self.cmd._curr_amps = ['amplitude']

        with mock.patch.object(self.cmd, 'do_decay_diagram', fake_decay_diagram), \
             mock.patch.object(self.cmd, 'integrate_widths_madspace',
                               mock.Mock(return_value={})), \
             mock.patch.object(madevent_interface.MadEventCmd,
                               'update_width_in_param_card'):
            self.cmd.do_compute_widths('h --body_decay=3 --path=%s --engine=madspace'
                                       % self.card)
        self.assertEqual(len(seen), 1)
        self.assertNotIn('engine', seen[0])
        self.assertNotIn('precision_channel', seen[0])

    def test_falls_back_on_madevent_when_madspace_fails(self):
        failing = mock.Mock(side_effect=RuntimeError('no compiler'))
        with self.assertRaises(StopAtMadEvent):
            self.run_compute_widths('', failing)
        failing.assert_called_once()

    def test_the_fallback_says_so(self):
        failing = mock.Mock(side_effect=RuntimeError('no compiler'))
        warnings = []
        handler = logging.Handler()
        handler.emit = lambda record: warnings.append(record.getMessage())
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger('cmdprint')
        logger.addHandler(handler)
        # another test may have muted the logger (MuteLogger, a level, ...)
        level, disabled = logger.level, logger.disabled
        logger.setLevel(logging.WARNING)
        logger.disabled = False
        try:
            with self.assertRaises(StopAtMadEvent):
                self.run_compute_widths('', failing)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(level)
            logger.disabled = disabled
        text = '\n'.join(warnings)
        self.assertIn('no compiler', text)
        self.assertIn('madevent', text)

    def test_complex_mass_scheme_uses_madevent(self):
        """The mg7 output has no complex mass scheme: not even an attempt."""
        madspace = mock.Mock()
        self.cmd.options['complex_mass_scheme'] = True
        try:
            with self.assertRaises(StopAtMadEvent):
                self.run_compute_widths('', madspace)
            with self.assertRaises(madgraph.InvalidCmd):
                self.run_compute_widths('--engine=madspace', madspace)
        finally:
            self.cmd.options['complex_mass_scheme'] = False
        madspace.assert_not_called()

    def test_no_fallback_when_madspace_was_asked_for(self):
        failing = mock.Mock(side_effect=RuntimeError('no compiler'))
        with self.assertRaises(RuntimeError):
            self.run_compute_widths('--engine=madspace', failing)


def fake_amplitude(pid, final_state):
    leg = base_objects.Leg({'id': pid, 'state': False})
    process = base_objects.Process({'legs': base_objects.LegList([leg])})
    amp = mock.MagicMock()
    amp.get.side_effect = lambda key: process if key == 'process' else None
    amp.final_state = final_state
    return amp


class MadspaceWidthsTest(unittest.TestCase):
    """integrate_widths_madspace: one output per decaying particle, widths
    gathered the way the two-body formulas fill decay_info."""

    @classmethod
    def setUpClass(cls):
        cls.cmd = mgcmd.MasterCmd()
        cls.cmd.no_notification()
        cls.cmd.exec_cmd('import model sm')

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='test_auto_width')
        self.card = pjoin(self.tmp, 'param_card.dat')
        with open(self.card, 'w') as stream:
            stream.write(self.cmd._curr_model.write_param_card())
        self.opts = {'path': self.card, 'output': self.card + '.out',
                     'precision_channel': 0.01}
        self.outputs = []
        self.originals = (self.cmd._curr_amps, self.cmd._curr_matrix_elements)

    def tearDown(self):
        self.cmd._curr_amps, self.cmd._curr_matrix_elements = self.originals
        shutil.rmtree(self.tmp, ignore_errors=True)

    def integrate(self, amps, results):
        """Run with the output and the integration stubbed: ``results`` maps
        the pid of the decaying particle to what the integration returns."""
        self.cmd._curr_amps = amps

        def exec_cmd(command, *args, **opts):
            # what the output sees: the amplitudes of one particle, and no
            # leftover matrix elements of the previous output
            self.assertTrue(command.startswith('output mg7'))
            self.outputs.append(
                (sorted(set(abs(a.get('process').get('legs')[0].get('id'))
                            for a in self.cmd._curr_amps)),
                 len(self.cmd._curr_matrix_elements.get_matrix_elements())))
            decay_dir = command.split()[2]
            os.makedirs(pjoin(decay_dir, 'Cards'))
            # whatever matrix elements 'output' would leave behind
            self.cmd._curr_matrix_elements = mock.MagicMock()
            self.cmd._curr_matrix_elements.get_matrix_elements.return_value = ['left over']

        def run_widths(decay_dir, precision):
            self.assertEqual(precision, 0.01)
            pid = self.outputs[-1][0][0]
            return results[pid]

        with mock.patch.object(self.cmd, 'exec_cmd', exec_cmd), \
             mock.patch.object(self.cmd, 'run_madspace_decay_widths', run_widths), \
             mock.patch.object(self.cmd, 'setup_mg7_environment'), \
             mock.patch.object(mg7_bootstrap, 'ensure_madspace'), \
             mock.patch('madgraph.core.diagram_generation.AmplitudeList', list):
            return self.cmd.integrate_widths_madspace(self.opts, {})

    def test_one_output_per_decaying_particle(self):
        amps = [fake_amplitude(25, 'a'), fake_amplitude(23, 'b'),
                fake_amplitude(25, 'c')]
        results = {25: [{'incoming': 25, 'outgoing': [-5, 5], 'width': 4e-3, 'error': 1e-6}],
                   23: [{'incoming': 23, 'outgoing': [-24, -11, 12], 'width': 2e-7, 'error': 1e-9}]}
        decay_info = self.integrate(amps, results)
        self.assertEqual([pids for pids, _ in self.outputs], [[25], [23]])
        # the second output starts from a cleared matrix element list
        self.assertEqual([count for _, count in self.outputs], [0, 0])
        self.assertEqual(sorted(decay_info), [23, 25])

    def test_the_state_of_the_interface_is_restored(self):
        amps = [fake_amplitude(25, 'a'), fake_amplitude(23, 'b')]
        results = {25: [], 23: []}
        self.integrate(amps, results)
        self.assertIs(self.cmd._curr_amps, amps)

    def test_widths_are_summed_per_final_state(self):
        """Two flavour assignments, or two subprocesses, of a decay give one
        line of the card; the final state is sorted like madevent writes it."""
        amps = [fake_amplitude(25, 'a')]
        results = {25: [
            {'incoming': 25, 'outgoing': [5, -5], 'width': 4.0e-3, 'error': 1e-6},
            {'incoming': 25, 'outgoing': [-24, 1, -2], 'width': 1.0e-4, 'error': 1e-7},
            {'incoming': 25, 'outgoing': [1, -24, -2], 'width': 2.0e-4, 'error': 1e-7},
        ]}
        decay_info = self.integrate(amps, results)
        widths = dict((tuple(products), width) for products, width in decay_info[25])
        self.assertEqual(sorted(widths), sorted([(-5, 5), (-24, -2, 1)]))
        self.assertAlmostEqual(widths[(-5, 5)], 4.0e-3)
        self.assertAlmostEqual(widths[(-24, -2, 1)], 3.0e-4)

    def test_a_failure_leaves_decay_info_alone(self):
        """The caller falls back on MadEvent with the decay_info it had: the
        particles integrated before the failure must not be in it twice."""
        amps = [fake_amplitude(25, 'a'), fake_amplitude(23, 'b')]
        known = {25: [[[-15, 15], 2.6e-4]]}
        self.cmd._curr_amps = amps

        def run_widths(decay_dir, precision):
            if len(self.outputs) == 2:
                raise madgraph.MadGraph5Error('the second particle failed')
            return [{'incoming': 25, 'outgoing': [5, -5], 'width': 4e-3, 'error': 0.}]

        def exec_cmd(command, *args, **opts):
            self.outputs.append(command)
            os.makedirs(pjoin(command.split()[2], 'Cards'))

        with mock.patch.object(self.cmd, 'exec_cmd', exec_cmd), \
             mock.patch.object(self.cmd, 'run_madspace_decay_widths', run_widths), \
             mock.patch.object(self.cmd, 'setup_mg7_environment'), \
             mock.patch.object(mg7_bootstrap, 'ensure_madspace'), \
             mock.patch('madgraph.core.diagram_generation.AmplitudeList', list):
            with self.assertRaises(madgraph.MadGraph5Error):
                self.cmd.integrate_widths_madspace(self.opts, known)
        self.assertEqual(known, {25: [[[-15, 15], 2.6e-4]]})
        self.assertIs(self.cmd._curr_amps, amps)

    def test_the_survey_asks_for_no_lhapdf(self):
        self.cmd._curr_amps = []
        setup = mock.Mock()
        with mock.patch.object(self.cmd, 'setup_mg7_environment', setup), \
             mock.patch.object(mg7_bootstrap, 'ensure_madspace'):
            self.cmd.integrate_widths_madspace(self.opts, {})
        setup.assert_called_once_with(lhapdf=False)

    def test_widths_are_added_to_what_is_there(self):
        """The two-body widths of the formulas are already in decay_info."""
        self.cmd._curr_amps = [fake_amplitude(25, 'a')]
        known = {25: [[[-15, 15], 2.6e-4]]}
        results = {25: [{'incoming': 25, 'outgoing': [5, -5], 'width': 4e-3, 'error': 0.}]}

        def exec_cmd(command, *args, **opts):
            os.makedirs(pjoin(command.split()[2], 'Cards'))

        with mock.patch.object(self.cmd, 'exec_cmd', exec_cmd), \
             mock.patch.object(self.cmd, 'run_madspace_decay_widths',
                               lambda decay_dir, precision: results[25]), \
             mock.patch.object(self.cmd, 'setup_mg7_environment'), \
             mock.patch.object(mg7_bootstrap, 'ensure_madspace'), \
             mock.patch('madgraph.core.diagram_generation.AmplitudeList', list):
            decay_info = self.cmd.integrate_widths_madspace(self.opts, known)
        self.assertEqual(len(decay_info[25]), 2)
        self.assertEqual(decay_info[25][0], [[-15, 15], 2.6e-4])


class MadspaceProcessTest(unittest.TestCase):
    """The survey of the decays runs in a process of its own."""

    @classmethod
    def setUpClass(cls):
        cls.cmd = mgcmd.MasterCmd()
        cls.cmd.no_notification()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='test_auto_width')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_results_are_read_back(self):
        results = [{'incoming': 25, 'outgoing': [5, -5], 'width': 4e-3, 'error': 1e-9}]

        def call(command, cwd=None, stdout=None, stderr=None, stdin=None):
            self.assertEqual(cwd, self.tmp)
            self.assertIn('--decay-widths=%s' % pjoin(self.tmp, 'decay_widths.json'),
                          command)
            self.assertIn('--precision=0.005', command)
            with open(pjoin(self.tmp, 'decay_widths.json'), 'w') as stream:
                json.dump(results, stream)
            return 0

        with mock.patch('subprocess.call', call):
            self.assertEqual(self.cmd.run_madspace_decay_widths(self.tmp, 0.005),
                             results)

    def test_the_cores_are_passed_on(self):
        seen = []

        def call(command, cwd=None, stdout=None, stderr=None, stdin=None):
            seen.append(command)
            with open(pjoin(self.tmp, 'decay_widths.json'), 'w') as stream:
                json.dump([], stream)
            return 0

        self.cmd.options['nb_core'] = 3
        try:
            with mock.patch('subprocess.call', call):
                self.cmd.run_madspace_decay_widths(self.tmp, 0.01)
        finally:
            self.cmd.options['nb_core'] = None
        self.assertIn('--nb-core=3', seen[0])

    def test_a_failure_shows_the_end_of_the_log(self):
        def call(command, cwd=None, stdout=None, stderr=None, stdin=None):
            stdout.write('compiling\n' * 40 + 'fatal error: no such file\n')
            stdout.flush()
            return 2

        with mock.patch('subprocess.call', call):
            with self.assertRaises(madgraph.MadGraph5Error) as caught:
                self.cmd.run_madspace_decay_widths(self.tmp, 0.01)
        message = str(caught.exception)
        self.assertIn('exit status 2', message)
        self.assertIn('fatal error: no such file', message)


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class DecayAlphaSTest(unittest.TestCase):
    """The alpha_s of a decay is the 2-loop running of madevent, at the mass of
    the decaying particle."""

    # alpha_s(m_Z) = 0.118 run with Source/alfas_functions.f (nloop = 2),
    # compiled and called at these scales
    FORTRAN = {125.0: 0.1126479113, 91.188: 0.118, 172.5: 0.1076705672,
               80.379: 0.1202897039, 3.0: 0.2522179973}

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='test_auto_width')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_grid(self, mass, alpha_s_mz=0.118):
        from madgraph.mg7 import launch as mg7_launch
        process = object.__new__(mg7_launch.MadgraphProcess)
        process.param_card = mock.Mock()
        process.param_card.get_value = lambda block, code: alpha_s_mz
        process.decaying_mass = mass
        process.e_cm = mass
        process.run_path = self.tmp
        path = process.write_fixed_alphas_info()
        info = {}
        with open(path) as stream:
            for line in stream:
                if ':' in line and not line.startswith('SetDesc'):
                    key, value = line.split(':', 1)
                    info[key.strip()] = [float(x) for x in
                                         value.strip().strip('[]').split(',')]
        return info['AlphaS_Qs'], info['AlphaS_Vals']

    def test_the_value_at_the_decaying_mass_is_the_running_one(self):
        for mass in (125.0, 172.5, 80.379, 3.0):
            qs, values = self.write_grid(mass)
            self.assertIn(mass, qs)
            self.assertAlmostEqual(values[qs.index(mass)], self.FORTRAN[mass], places=8)

    def test_the_running_is_not_constant(self):
        qs, values = self.write_grid(125.0)
        self.assertEqual(qs, sorted(qs))
        self.assertEqual(len(set(qs)), len(qs))
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertLess(values[qs.index(125.0)], 0.118)

    def test_alpha_s_of_the_card_is_the_value_at_mz(self):
        """The grid has no node at m_Z, so interpolate across it."""
        qs, values = self.write_grid(125.0, alpha_s_mz=0.1185)
        runner = Alphas_Runner(0.1185, 2, cmass=1.42, bmass=4.7)
        for q, value in zip(qs, values):
            self.assertAlmostEqual(value, runner(q), places=10)
        below = max(q for q in qs if q < 91.188)
        above = min(q for q in qs if q > 91.188)
        self.assertGreater(values[qs.index(below)], 0.1185)
        self.assertLess(values[qs.index(above)], 0.1185)

    def test_a_light_decaying_particle_extends_the_grid_down(self):
        qs, values = self.write_grid(0.5)
        self.assertEqual(qs[0], 0.5)
        self.assertTrue(all(v > 0 for v in values))

    def test_alpha_s_is_frozen_below_the_freeze_scale(self):
        """The muon has no QCD: its widths must not fail on the running of
        alpha_s, which is not defined at its mass."""
        runner = Alphas_Runner(0.118, 2, cmass=1.42, bmass=4.7)
        qs, values = self.write_grid(0.1057)
        self.assertEqual(qs[0], 0.1057)
        self.assertAlmostEqual(values[0], runner(1.0), places=10)
        self.assertEqual(values, sorted(values, reverse=True))
        for q, value in zip(qs, values):
            if q >= 1.0:
                self.assertAlmostEqual(value, runner(q), places=10)
            else:
                self.assertAlmostEqual(value, runner(1.0), places=10)

    def test_alpha_s_is_frozen_where_the_running_breaks_down(self):
        """A larger alpha_s(m_Z) moves the Landau pole up: below 1 GeV the
        running overflows for alpha_s(m_Z) = 0.13, and the grid still comes."""
        qs, values = self.write_grid(0.5, alpha_s_mz=0.13)
        self.assertEqual(qs[0], 0.5)
        self.assertTrue(all(v > 0 for v in values))
        self.assertEqual(values[0], values[qs.index(1.0)])

    def test_python_running_is_the_fortran_one(self):
        runner = Alphas_Runner(0.118, 2, cmass=1.42, bmass=4.7)
        for q, value in self.FORTRAN.items():
            self.assertAlmostEqual(runner(q), value, places=8)


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class DecayFlavorSplitTest(unittest.TestCase):
    """compute_decay_widths integrates every flavour of a subprocess on its
    own (the u u~ and d d~ of a flavour-grouped z > q q~ g have different
    couplings to the Z) and splits a flavour's width evenly between its
    assignments, which share its matrix element."""

    # as written by 'output mg7' for z > q qx g, q = u d c s
    GROUPED = {
        'path': 'SubProcesses/P1_z_QQxg', 'incoming': [23], 'outgoing': [81, -81, 21],
        'me_path': 'lib/libmadmatrix_P1_z_QQxg_{device}.so',
        'flavors': [{'index': 0, 'options': [[23, 1, -1, 21], [23, 3, -3, 21]],
                     'mirror': False},
                    {'index': 1, 'options': [[23, 2, -2, 21], [23, 4, -4, 21]],
                     'mirror': False}],
        'channels': [
            {'propagators': [81], 'diagrams': [
                {'diagram': 0, 'permutation': [0, 1, 2, 3], 'active_flavors': [0, 1]},
                {'diagram': 1, 'permutation': [0, 2, 1, 3], 'active_flavors': [0, 1]}]},
            {'propagators': [82], 'diagrams': [
                {'diagram': 2, 'permutation': [0, 1, 2, 3], 'active_flavors': [1]}]}],
    }

    def setUp(self):
        from madgraph.mg7 import launch as mg7_launch
        self.launch = mg7_launch

    def test_one_subprocess_per_flavour(self):
        original = copy.deepcopy(self.GROUPED)
        down, up = self.launch._single_flavor_metas(self.GROUPED)
        self.assertEqual(self.GROUPED, original)
        for single, flavor in zip((down, up), original['flavors']):
            self.assertEqual(single['flavors'], [flavor])
            self.assertEqual(single['me_path'], original['me_path'])
            self.assertEqual(single['path'], original['path'])
        # the diagrams the flavour is active in, renumbered to flavour 0
        self.assertEqual(len(down['channels']), 1)
        self.assertEqual([d['active_flavors'] for d in down['channels'][0]['diagrams']],
                         [[0], [0]])
        self.assertEqual(len(up['channels']), 2)
        self.assertEqual([d['diagram'] for d in up['channels'][1]['diagrams']], [2])
        self.assertEqual(self.launch._decay_flavor_options(down),
                         [[23, 1, -1, 21], [23, 3, -3, 21]])
        self.assertEqual(self.launch._decay_flavor_options(up),
                         [[23, 2, -2, 21], [23, 4, -4, 21]])

    def test_a_single_flavour_is_kept_as_it_is(self):
        meta = {'flavors': [self.GROUPED['flavors'][0]], 'channels': []}
        self.assertEqual(self.launch._single_flavor_metas(meta), [meta])

    def test_a_flavour_without_diagrams_has_no_subprocess(self):
        meta = copy.deepcopy(self.GROUPED)
        meta['channels'] = meta['channels'][1:]
        singles = self.launch._single_flavor_metas(meta)
        self.assertEqual([s['flavors'][0]['index'] for s in singles], [1])

    def test_split_needs_unmerged_subprocesses(self):
        process = object.__new__(self.launch.MadgraphProcess)
        process.merged_subprocess_data = [{'subprocesses': [0]}]
        process.subprocess_data = [self.GROUPED]
        with self.assertRaises(ValueError):
            process.split_decay_flavors()

    def test_several_flavours_left_are_refused(self):
        """never an even split across matrix elements"""
        with self.assertRaises(ValueError) as caught:
            self.launch._decay_flavor_options(self.GROUPED)
        self.assertIn('P1_z_QQxg', str(caught.exception))
