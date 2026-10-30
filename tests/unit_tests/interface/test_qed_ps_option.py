"""The experimental QED+PS override belongs to a single launch command."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph.interface import amcatnlo_interface, amcatnlo_run_interface
from madgraph.interface import extended_cmd, launch_ext_program, madgraph_interface
from madgraph.various.banner import RunCardNLO


class ConfigurationReached(Exception):
    """Stop after the real run selector, before compilation or event generation."""


class TestQEDPSOption(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='mg5_qed_ps_option_')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / 'Cards').mkdir()
        card = RunCardNLO()
        card['parton_shower'] = 'PYTHIA8'
        card.write(str(self.root / 'Cards/run_card.dat'))
        (self.root / 'Cards/shower_card.dat').touch()
        self.mother = SimpleNamespace(
            me_dir=str(self.root), last_mode='',
            proc_characteristics={'ninitial': 2, 'splitting_types': ['QED']},
            options={'madanalysis5_path': None, 'f2py_compiler': None,
                     'pythia8_path': str(self.root), 'herwig7_path': None,
                     'thepeg_path': None, 'hepmc_path': None})
        # Select installed modules deterministically and omit terminal rendering.
        for patch in (
                mock.patch.object(amcatnlo_run_interface.misc, 'has_f2py', return_value=False),
                mock.patch.object(amcatnlo_run_interface.misc, 'which', return_value='/bin/bc'),
                mock.patch.object(extended_cmd.ControlSwitch, 'create_question', return_value='')):
            patch.start()
            self.addCleanup(patch.stop)

    def selector(self, allow=False):
        return amcatnlo_run_interface.AskRunNLO(
            '', mother_interface=self.mother, allow_qed_ps=allow)

    def test_default_keeps_qed_fixed_order_and_hides_shower(self):
        selector = self.selector()
        self.assertEqual(selector.get_allowed_fixed_order(), ['ON'])
        self.assertEqual(selector.get_allowed_shower(), ['OFF'])
        self.assertIn('shower', selector.hide_line)
        selector.default('aMC@NLO')
        self.assertEqual(selector.answer['fixed_order'], 'ON')
        self.assertFalse(selector.answer['runshower'])

    def test_override_enables_matching_and_visible_shower(self):
        selector = self.selector(allow=True)
        self.assertIn('OFF', selector.get_allowed_fixed_order())
        self.assertIn('PYTHIA8', selector.get_allowed_shower())
        self.assertNotIn('shower', selector.hide_line)
        selector.default('aMC@NLO')
        self.assertEqual(selector.answer['fixed_order'], 'OFF')
        self.assertEqual(selector.answer['shower'], 'PYTHIA8')
        self.assertTrue(selector.answer['runshower'])

    def test_override_does_not_leak_into_next_run(self):
        self.assertTrue(self.selector(allow=True).answer['runshower'])
        self.assertFalse(self.selector().answer['runshower'])
        self.assertEqual(self.mother.proc_characteristics['splitting_types'], ['QED'])
        self.assertNotIn('allow_qed_ps', self.mother.options)

    def test_qcd_and_decay_restrictions_are_unchanged(self):
        self.mother.proc_characteristics['splitting_types'] = ['QCD']
        for allow in (False, True):
            self.assertTrue(self.selector(allow).answer['runshower'])
        self.mother.proc_characteristics['ninitial'] = 1
        for split in ('QCD', 'QED'):
            self.mother.proc_characteristics['splitting_types'] = [split]
            selector = self.selector(allow=True)
            self.assertEqual(selector.get_allowed_fixed_order(), ['ON'])
            self.assertEqual(selector.get_allowed_shower(), ['OFF'])
            self.assertFalse(selector.answer['runshower'])

    def test_parsers_default_off_and_accept_the_option(self):
        parsers = [amcatnlo_run_interface._launch_parser,
                   amcatnlo_run_interface._compile_parser,
                   amcatnlo_interface._launch_parser,
                   madgraph_interface._launch_parser]
        for parser in parsers:
            with self.subTest(parser=parser.get_usage()):
                options, args = parser.parse_args([])
                self.assertFalse(options.allow_qed_ps)
                options, args = parser.parse_args(['--allow_qed_ps'])
                self.assertTrue(options.allow_qed_ps)
                self.assertEqual(args, [])
                self.assertIn('--allow_qed_ps', parser.format_help())

    def command(self):
        command = object.__new__(amcatnlo_run_interface.aMCatNLOCmd)
        command.stop_for_runweb = True
        command.force = False
        command.proc_characteristics = self.mother.proc_characteristics
        command.cluster_mode = 0

        def observe(question, default, choices, **kwargs):
            selector = kwargs['ask_class'](
                question, mother_interface=self.mother,
                allow_qed_ps=kwargs['allow_qed_ps'])
            for line in kwargs['first_cmd']:
                selector.onecmd(line)
            self.selected = selector.answer
            raise ConfigurationReached

        command.ask = observe
        return command

    def test_launch_and_generate_events_route_override_per_invocation(self):
        command = self.command()
        for entry in (command.do_launch, command.do_generate_events):
            with self.subTest(command=entry.__name__):
                for mode in ('aMC@NLO', ''):
                    with self.subTest(mode=mode):
                        with self.assertRaises(ConfigurationReached):
                            entry(mode + ' --allow_qed_ps -f')
                        self.assertTrue(self.selected['runshower'])
                        self.assertEqual(self.selected['fixed_order'], 'OFF')
                with self.assertRaisesRegex(command.InvalidCmd, '--allow_qed_ps'):
                    entry('aMC@NLO -f')
                with self.assertRaises(ConfigurationReached):
                    entry('-f')
                self.assertFalse(self.selected['runshower'])
                self.assertEqual(self.selected['fixed_order'], 'ON')

    def test_explicit_matching_and_parton_modes_need_override(self):
        command = self.command()
        for mode in ('aMC@NLO', 'aMC@LO', 'noshower', 'noshowerLO', 'onlyshower'):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(command.InvalidCmd, '--allow_qed_ps'):
                    command.ask_run_configuration(mode, {'force': True})
        with self.assertRaisesRegex(command.InvalidCmd, '--allow_qed_ps'):
            command.do_launch('-p -f')
        for mode in ('LO', 'NLO'):
            with self.subTest(mode=mode):
                with self.assertRaises(ConfigurationReached):
                    command.do_launch(mode + ' -f')

    def test_outer_launcher_forwards_override_to_generated_process(self):
        options, _ = amcatnlo_interface._launch_parser.parse_args(
            ['aMC@NLO', '--allow_qed_ps', '--name=override_test', '-f'])
        parent = mock.Mock(options={}, history=[])
        runtime = mock.Mock(options_configuration={}, options_madgraph={}, options_madevent={})
        parent.define_child_cmd_interface.return_value = runtime
        launcher = launch_ext_program.aMCatNLOLauncher(
            str(self.root), parent, run_mode='aMC@NLO', shell=False, **vars(options))
        with mock.patch.object(amcatnlo_run_interface, 'aMCatNLOCmd', return_value=runtime), \
             mock.patch.object(launch_ext_program.os, 'remove'):
            launcher.launch_program()
        launch = runtime.run_cmd.call_args_list[0].args[0]
        options, args = amcatnlo_run_interface._launch_parser.parse_args(launch.split()[1:])
        self.assertTrue(options.allow_qed_ps)
        self.assertEqual(args, ['aMC@NLO'])
