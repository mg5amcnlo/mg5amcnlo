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
"""`launch` on an mg7 output.

The mg7 branch of :meth:`MadGraphCmd.do_launch` used to shell out to
``bin/generate_events`` and guess, from a hand-written prefix list, which of the
lines following ``launch`` in the user's command file belonged to the run. It
now builds the run interface in process and registers it through
``define_child_cmd_interface``, exactly like a madevent output, so the run reads
MG5's own script and hands back what it does not understand.

The wiring tests here stub the launcher module, so they run without a compiled
madspace; the tests that exercise the real :class:`MG7Cmd` are skipped when
madspace is not installed.
"""

from __future__ import absolute_import

import logging
import multiprocessing
import json
import os
import shutil
import subprocess
import sys
import tempfile
import copy
import types
import unittest
from unittest import mock

import madgraph
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.interface.master_interface as mgcmd
from madgraph.iolibs.template_files.mg7 import bootstrap as mg7_bootstrap

LAUNCH_MODULE = 'madgraph.iolibs.template_files.mg7.launch'
_MG_ROOT = str(mg7_bootstrap.MG_ROOT)


def make_mg7_dir(root):
    """The minimum that makes find_output_type() answer 'mg7'."""
    os.makedirs(os.path.join(root, 'SubProcesses'))
    os.makedirs(os.path.join(root, 'Cards'))
    with open(os.path.join(root, 'Cards', 'run_card.toml'), 'w') as stream:
        stream.write('[run]\nrun_name = "run"\n')
    return root


class FakeMG7Cmd(ext_cmd.Cmd):
    """Stands in for the real MG7Cmd: records what it was asked to run."""

    def __init__(self, me_dir='.', options=None):
        super(FakeMG7Cmd, self).__init__()
        self.me_dir = me_dir
        # the real MG7Cmd layers these over the output's own configuration;
        # keep the raw argument so the wiring can be checked
        self.raw_options = options
        self.options = options or {}
        self.commands = []
        self.raise_on_run = None

    def do_generate_events(self, line):
        self.commands.append(line)
        if self.raise_on_run is not None:
            raise self.raise_on_run

    def do_quit(self, line):
        self.commands.append('quit')
        return True


class MG7LaunchWiringTest(unittest.TestCase):
    """do_launch's mg7 branch: what it builds and what it hands the child."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.me_dir = make_mg7_dir(os.path.join(self.tmpdir.name, 'PROC'))

        self.cmd = mgcmd.MasterCmd()
        self.cmd.use_rawinput = False
        # MasterCmd reads the checkout's input/mg7_configuration.txt, and
        # crash_on_error decides whether an error inside the run stops MG5:
        # pin the shipped default so a developer's own setting cannot change
        # what these tests see. The tests that need it set it themselves.
        self.cmd.options['crash_on_error'] = False

        # MG5's error handling writes a 'debug' file into the working
        # directory; keep that (and anything else a test provokes) inside the
        # temporary directory instead of the checkout.
        self._saved_cwd = os.getcwd()
        os.chdir(self.tmpdir.name)

        # Stub the launcher module: importing the real one needs madspace.
        self.built = []
        stub = types.ModuleType(LAUNCH_MODULE)

        def MG7Cmd(me_dir='.', options=None):
            child = FakeMG7Cmd(me_dir, options)
            self.built.append(child)
            return child

        stub.MG7Cmd = MG7Cmd
        self.stub = stub
        self._saved_module = sys.modules.get(LAUNCH_MODULE)
        sys.modules[LAUNCH_MODULE] = stub
        # `from pkg import launch` resolves through the parent package's
        # attribute when the real module has already been imported (which
        # MG7CmdTest does), so sys.modules alone is not enough to stub it.
        import madgraph.iolibs.template_files.mg7 as mg7_pkg
        self.mg7_pkg = mg7_pkg
        self._saved_attr = getattr(mg7_pkg, 'launch', None)
        mg7_pkg.launch = stub

        # Never run the real madspace bootstrap from a unit test.
        self.bootstrap_calls = []
        self.bootstrap_jobs = []
        self._saved_ensure = mg7_bootstrap.ensure_madspace

        def fake_ensure(interactive=None, jobs=None):
            self.bootstrap_calls.append(interactive)
            self.bootstrap_jobs.append(jobs)

        mg7_bootstrap.ensure_madspace = fake_ensure

    def tearDown(self):
        mg7_bootstrap.ensure_madspace = self._saved_ensure
        if self._saved_module is None:
            sys.modules.pop(LAUNCH_MODULE, None)
        else:
            sys.modules[LAUNCH_MODULE] = self._saved_module
        if self._saved_attr is None:
            del self.mg7_pkg.launch
        else:
            self.mg7_pkg.launch = self._saved_attr
        os.chdir(self._saved_cwd)
        self.tmpdir.cleanup()

    def launch(self, line, script_lines=()):
        """Run `launch <line>` with ``script_lines`` following it in the file."""
        self.cmd.inputfile = iter(list(script_lines))
        self.cmd.do_launch('%s %s' % (self.me_dir, line) if line else self.me_dir)
        return self.built[-1] if self.built else None

    # -- the bug this change fixes ------------------------------------------
    def test_set_line_is_left_for_the_question_not_scavenged(self):
        """The lines after `launch` must stay in MG5's inputfile.

        The subprocess branch consumed every following line that did not start
        with one of a fixed list of MG5 keywords -- `set` deliberately among the
        consumed ones -- and piped them to the child's stdin. An MG5-level
        `set gauge Feynman` after `launch` was therefore swallowed with no
        message from either side. In process the child shares MG5's inputfile,
        so nothing is consumed up front.
        """
        remaining = ['set nevents 500\n', 'set gauge Feynman\n']
        self.launch('', remaining)
        # the child was handed the *same* iterator, still holding both lines
        child = self.built[-1]
        self.assertIs(child.inputfile, self.cmd.inputfile)
        self.assertEqual([l.strip() for l in child.inputfile],
                         ['set nevents 500', 'set gauge Feynman'])

    def test_child_is_registered_with_mg5(self):
        """define_child_cmd_interface is what gives the child MG5's script."""
        self.launch('')
        child = self.built[-1]
        self.assertIs(child.mother, self.cmd)
        self.assertIs(self.cmd.child, child)

    def test_child_runs_generate_events_then_quits(self):
        child = self.launch('')
        # do_generate_events is handed the arguments, so a bare launch is ''
        self.assertEqual(child.commands, ['', 'quit'])

    def test_me_dir_and_options_are_passed(self):
        child = self.launch('')
        # check_launch normalises the path with realpath
        self.assertEqual(child.me_dir, os.path.realpath(self.me_dir))
        self.assertIs(child.raw_options, self.cmd.options)

    # -- launch options, all of which the subprocess branch dropped ---------
    def test_force_flag_is_forwarded(self):
        child = self.launch('-f')
        self.assertEqual(child.commands[0], '-f')

    def test_run_name_is_forwarded(self):
        child = self.launch('-n myrun')
        self.assertEqual(child.commands[0], '--name=myrun')

    def test_laststep_is_forwarded(self):
        child = self.launch('--laststep=parton')
        self.assertEqual(child.commands[0], '--laststep=parton')

    def test_options_combine(self):
        child = self.launch('-f -n myrun --laststep=parton')
        self.assertEqual(child.commands[0],
                         '-f --name=myrun --laststep=parton')

    def test_interactive_hands_over_the_prompt(self):
        """`launch -i` must give the user the child's own command loop."""
        self.cmd.inputfile = iter([])
        self.cmd.do_launch('%s -i' % self.me_dir)
        child = self.built[-1]
        self.assertIs(self.cmd.child, child)
        # not driven with a canned command: the user drives it
        self.assertEqual(child.commands, [])

    # -- interrupts ---------------------------------------------------------
    def interrupted_launch(self):
        """Stub a child whose run is interrupted by Ctrl-C; return the list
        that records its stop_on_keyboard_stop calls."""
        stopped = []

        def MG7Cmd(me_dir='.', options=None):
            child = FakeMG7Cmd(me_dir, options)
            child.raise_on_run = KeyboardInterrupt()
            child.stop_on_keyboard_stop = lambda: stopped.append(True)
            self.built.append(child)
            return child

        self.stub.MG7Cmd = MG7Cmd
        self.cmd.inputfile = iter([])
        return stopped

    def test_keyboard_interrupt_reaches_the_error_handling(self):
        """Ctrl-C used to be swallowed by `except KeyboardInterrupt: pass`.

        Going through the child's run_cmd means it now reaches the standard
        cmd error handling, exactly as a madevent run does: the child's
        stop_on_keyboard_stop runs (that is where a run cleans up its jobs)
        and the child is still closed. With the default crash_on_error the
        interrupt stops the run, not MG5 -- the same as for madevent.
        """
        stopped = self.interrupted_launch()
        self.cmd.do_launch(self.me_dir)
        self.assertEqual(stopped, [True])
        self.assertEqual(self.built[-1].commands, ['', 'quit'])

    @unittest.skipUnless(__debug__, 'error_handling only consults '
                         'crash_on_error for Ctrl-C in debug mode')
    def test_keyboard_interrupt_with_crash_on_error_stops_mg5(self):
        """With crash_on_error = True the interrupt ends MG5, after the run
        has had its stop_on_keyboard_stop."""
        self.cmd.options['crash_on_error'] = True
        stopped = self.interrupted_launch()
        self.assertRaises(SystemExit, self.cmd.do_launch, self.me_dir)
        self.assertEqual(stopped, [True])

    # -- the environment the subprocess branch used to build ---------------
    def test_environment_is_exported_in_process(self):
        """MADGRAPH_HEPTOOLS_DIR / LHAPDF_DATA_PATH / MADGRAPH_LHAPDF_CONFIG
        used to be put in the subprocess env; in process they have to reach
        os.environ, which is the only channel the tools the run shells out to
        (the madspace build, the real LHAPDF library) can read."""
        saved = {k: os.environ.get(k) for k in
                 ('MADGRAPH_HEPTOOLS_DIR', 'LHAPDF_DATA_PATH',
                  'MADGRAPH_LHAPDF_CONFIG')}
        for key in saved:
            os.environ.pop(key, None)
        try:
            self.cmd.options['heptools_install_dir'] = self.tmpdir.name
            self.cmd.setup_mg7_environment()
            self.assertEqual(os.environ.get('MADGRAPH_HEPTOOLS_DIR'),
                             os.path.abspath(self.tmpdir.name))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_bootstrap_is_told_the_interactivity_explicitly(self):
        """The bootstrap must not guess from sys.argv/sys.stdin in process."""
        self.launch('')
        self.assertEqual(self.bootstrap_calls, [False])

        self.bootstrap_calls[:] = []
        self.cmd.use_rawinput = True
        self.cmd.inputfile = iter([])
        self.cmd.do_launch(self.me_dir)
        self.assertEqual(self.bootstrap_calls, [True])

    def test_bootstrap_gets_nb_core_for_the_madspace_build(self):
        """A source build of madspace is only parallel when it is told how many
        jobs it may use, so the launch path must pass MG5's nb_core on."""
        self.cmd.do_set('nb_core 3 --no_save')
        self.launch('')
        self.assertEqual(self.bootstrap_jobs, [3])

        # `set nb_core None` resolves to the machine's core count
        self.bootstrap_jobs[:] = []
        self.cmd.do_set('nb_core None --no_save')
        self.launch('')
        self.assertEqual(self.bootstrap_jobs,
                         [multiprocessing.cpu_count()])


class MG7BootstrapTest(unittest.TestCase):
    """The one-off madspace install that used to be a launch.py import side
    effect."""

    def setUp(self):
        self.saved = (mg7_bootstrap.INSTALL_DIR, mg7_bootstrap.MADSPACE_DIR,
                      subprocess.run)
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        (mg7_bootstrap.INSTALL_DIR, mg7_bootstrap.MADSPACE_DIR,
         subprocess.run) = self.saved
        sys.path[:] = [p for p in sys.path if self.tmpdir.name not in p]
        self.tmpdir.cleanup()

    def point_at_empty_install(self):
        from pathlib import Path
        root = Path(self.tmpdir.name)
        mg7_bootstrap.MADSPACE_DIR = root / 'madspace'
        mg7_bootstrap.INSTALL_DIR = root / 'madspace' / 'install'
        return root

    def record_runs(self):
        calls = []

        def fake_run(cmd, **opts):
            calls.append((cmd, opts))
            return types.SimpleNamespace(returncode=0)

        subprocess.run = fake_run
        return calls

    @staticmethod
    def quiet():
        """The bootstrap narrates the install on stdout; keep it out of the
        test output."""
        import contextlib
        import io
        return contextlib.redirect_stdout(io.StringIO())

    def test_no_install_when_already_present(self):
        root = self.point_at_empty_install()
        (root / 'madspace' / 'install' / 'madspace').mkdir(parents=True)
        calls = self.record_runs()
        with self.quiet():
            mg7_bootstrap.ensure_madspace(interactive=False)
        self.assertEqual(calls, [])
        self.assertIn(str(mg7_bootstrap.INSTALL_DIR), sys.path)

    def test_non_interactive_install_is_unattended(self):
        self.point_at_empty_install()
        calls = self.record_runs()
        with self.quiet():
            mg7_bootstrap.ensure_madspace(interactive=False)
        self.assertEqual(len(calls), 1)
        cmd, opts = calls[0]
        self.assertIn('--source', cmd)
        self.assertIn('--yes', cmd)
        # must not eat the caller's stdin, which carries the run's script
        self.assertEqual(opts['stdin'], subprocess.DEVNULL)

    def test_jobs_is_forwarded_to_the_installer(self):
        """Without a job count the installer's cmake build falls back to a
        serial make whenever ninja is missing."""
        self.point_at_empty_install()
        calls = self.record_runs()
        with self.quiet():
            mg7_bootstrap.ensure_madspace(interactive=False, jobs=4)
        cmd, _ = calls[0]
        self.assertIn('-j', cmd)
        self.assertEqual(cmd[cmd.index('-j') + 1], '4')

    def test_no_jobs_leaves_the_installer_default(self):
        self.point_at_empty_install()
        calls = self.record_runs()
        with self.quiet():
            mg7_bootstrap.ensure_madspace(interactive=False)
        cmd, _ = calls[0]
        self.assertNotIn('-j', cmd)

    def test_interactive_install_keeps_the_terminal(self):
        self.point_at_empty_install()
        calls = self.record_runs()
        with self.quiet():
            mg7_bootstrap.ensure_madspace(interactive=True)
        cmd, opts = calls[0]
        self.assertNotIn('--yes', cmd)
        self.assertIsNone(opts['stdin'])

    def test_explicit_interactivity_ignores_sys_argv(self):
        """In process, sys.argv is MG5's command line, not the run's: a `-f`
        meant for `mg5_aMC -f` must not be read as this run's force flag."""
        self.point_at_empty_install()
        calls = self.record_runs()
        saved_argv = sys.argv
        sys.argv = ['mg5_aMC', '-f']
        try:
            with self.quiet():
                mg7_bootstrap.ensure_madspace(interactive=True)
        finally:
            sys.argv = saved_argv
        cmd, opts = calls[0]
        self.assertNotIn('--yes', cmd)

    def test_drop_install_path_leaves_no_shadowing_entry(self):
        """The install dir also holds madspace's build dependencies (yaml,
        packaging, pathspec, ...), so leaving it on sys.path would shadow the
        caller's copies for the rest of what is now MG5's own session."""
        root = self.point_at_empty_install()
        (root / 'madspace' / 'install' / 'madspace').mkdir(parents=True)
        calls = self.record_runs()
        mg7_bootstrap.ensure_madspace(interactive=False)
        self.assertIn(str(mg7_bootstrap.INSTALL_DIR), sys.path)
        mg7_bootstrap.drop_install_path()
        self.assertNotIn(str(mg7_bootstrap.INSTALL_DIR), sys.path)
        self.assertEqual(calls, [])

    def test_failure_is_reported(self):
        self.point_at_empty_install()

        def failing_run(cmd, **opts):
            return types.SimpleNamespace(returncode=1)

        subprocess.run = failing_run
        with self.quiet():
            self.assertRaises(RuntimeError,
                              mg7_bootstrap.ensure_madspace, interactive=False)


class MadspaceInstallCommandTest(unittest.TestCase):
    """`install madspace` at the MG5 prompt: it runs madspace/install.py."""

    def setUp(self):
        self.cmd = mgcmd.MasterCmd()

    def installer_args(self, line):
        with mock.patch('subprocess.run') as run:
            run.return_value.returncode = 0
            self.cmd.do_install(line)
        return [str(a) for a in run.call_args[0][0]]

    def test_a_failing_installer_is_an_error(self):
        """A failed madspace/madtrex install.py (e.g. madtrex without
        madspace) is reported at the prompt, not passed over silently."""
        with mock.patch('subprocess.run') as run:
            run.return_value.returncode = 1
            with self.assertRaises(madgraph.InvalidCmd) as ctx:
                self.cmd.do_install('madtrex --yes')
        self.assertIn('install madtrex failed', str(ctx.exception))

    def test_nb_core_is_forwarded_as_the_job_count(self):
        """Without it the installer's cmake build is serial whenever ninja is
        missing, whatever nb_core says."""
        self.cmd.do_set('nb_core 5 --no_save')
        args = self.installer_args('madspace --source -y')
        self.assertIn('install.py', args[1])
        self.assertEqual(args[-2:], ['-j', '5'])

    def test_unset_nb_core_uses_every_core(self):
        self.cmd.do_set('nb_core None --no_save')
        args = self.installer_args('madspace --source -y')
        self.assertEqual(args[-2:], ['-j', str(multiprocessing.cpu_count())])

    def test_an_explicit_job_count_is_left_alone(self):
        self.cmd.do_set('nb_core 5 --no_save')
        for line in ('madspace --source -j 2', 'madspace --source -j2',
                     'madspace --source --jobs=2'):
            args = self.installer_args(line)
            self.assertEqual(len([a for a in args if a.startswith('-j')
                                  or a.startswith('--jobs')]), 1, args)
            self.assertNotIn('5', args)

    def test_madtrex_runs_its_own_installer(self):
        """`install madtrex` runs madtrex/install.py (not madspace's), with
        the same job-count forwarding."""
        self.cmd.do_set('nb_core 5 --no_save')
        args = self.installer_args('madtrex --yes --clean')
        self.assertEqual(os.path.basename(os.path.dirname(args[1])), 'madtrex')
        self.assertTrue(args[1].endswith('install.py'))
        self.assertEqual(args[2:], ['--yes', '--clean', '-j', '5'])


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class MG7CmdTest(unittest.TestCase):
    """The real run interface (needs a compiled madspace to import)."""

    def setUp(self):
        from madgraph.iolibs.template_files.mg7 import launch as mg7_launch
        self.launch = mg7_launch
        self.tmpdir = tempfile.TemporaryDirectory()
        self.me_dir = make_mg7_dir(os.path.join(self.tmpdir.name, 'PROC'))

    def tearDown(self):
        self.tmpdir.cleanup()

    def make_cmd(self):
        return self.launch.MG7Cmd(me_dir=self.me_dir)

    def test_option_parsing(self):
        cmd = self.make_cmd()
        none = {'reweight_card': None, 'reweight_backend': None}
        self.assertEqual(cmd._parse_run_options('-f'),
                         dict(none, force=True, name='', laststep=''))
        self.assertEqual(cmd._parse_run_options('--name=abc')['name'], 'abc')
        self.assertEqual(cmd._parse_run_options('--laststep=parton')['laststep'],
                         'parton')
        self.assertEqual(
            cmd._parse_run_options('-f --name=abc --laststep=parton'),
            dict(none, force=True, name='abc', laststep='parton'))
        # space-separated forms, as typed at the MG7> prompt
        self.assertEqual(cmd._parse_run_options('-n abc'),
                         dict(none, force=False, name='abc', laststep=''))
        opts = cmd._parse_run_options('--reweight=rw.dat --reweight-backend madtrex')
        self.assertEqual((opts['reweight_card'], opts['reweight_backend']),
                         ('rw.dat', 'madtrex'))
        opts = cmd._parse_run_options('--reweight rw.dat --reweight-backend=classic')
        self.assertEqual((opts['reweight_card'], opts['reweight_backend']),
                         ('rw.dat', 'classic'))
        self.assertEqual(cmd._parse_run_options('-s parton')['laststep'],
                         'parton')
        # a bare word is the run name
        self.assertEqual(cmd._parse_run_options('myrun')['name'], 'myrun')

    def test_crash_on_error_is_inherited_from_mg5(self):
        """The options are filtered down to the launcher's tool paths, which
        used to drop crash_on_error: a crash inside an mg7 launch then went
        back to the prompt and MG5 exited 0, even with crash_on_error True."""
        cmd = self.launch.MG7Cmd(me_dir=self.me_dir,
                                 options={'crash_on_error': True})
        self.assertIs(cmd.options['crash_on_error'], True)
        cmd = self.launch.MG7Cmd(me_dir=self.me_dir,
                                 options={'crash_on_error': 'never'})
        self.assertEqual(cmd.options['crash_on_error'], 'never')
        self.assertNotIn('crash_on_error', self.make_cmd().options)

    def test_laststep_parton_switches_every_tool_off(self):
        cmd = self.make_cmd()
        switch = {'shower': 'Pythia8', 'madspin': 'ON', 'reweight': 'OFF'}
        out = cmd._apply_laststep(switch, {'laststep': 'parton'})
        self.assertEqual(out, {'shower': 'OFF', 'madspin': 'OFF',
                               'reweight': 'OFF'})

    def test_laststep_empty_leaves_the_switch_alone(self):
        cmd = self.make_cmd()
        switch = {'shower': 'Pythia8'}
        self.assertEqual(cmd._apply_laststep(switch, {'laststep': ''}), switch)

    def output_format(self, switch, initial='compact_npy'):
        """Run force_lhe_output_if_needed on a run_card asking for `initial`
        and return the format the run ends up with."""
        path = os.path.join(self.me_dir, 'Cards', 'run_card.toml')
        with open(path, 'w') as stream:
            stream.write('[run]\nrun_name = "run"\n'
                         'output_format = "%s"\n' % initial)
        cwd = os.getcwd()
        os.chdir(self.me_dir)
        try:
            self.launch.force_lhe_output_if_needed(switch)
        finally:
            os.chdir(cwd)
        from madgraph.various.banner import RunCardMG7
        return RunCardMG7(path, consistency=False)['run']['output_format']

    def test_post_processing_forces_the_lhe_output(self):
        """the npy formats carry no event record a shower could read"""
        self.assertEqual(self.output_format({'shower': 'Pythia8'}), 'lhe')
        self.assertEqual(self.output_format({'madspin': 'ON'},
                                            initial='lhe_npy'), 'lhe')

    # -- the MADatLO.HwU switch ---------------------------------------------
    class FakeHistograms:
        """Stands in for the madspace EventHistograms of a finished run."""

        JSON = ('[{"name": "jet-pt", "min": 0.0, "max": 100.0, '
                '"bin_count": 2, "bin_values": [0.0, 1.0, 2.0, 0.0], '
                '"bin_errors": [0.0, 0.1, 0.2, 0.0], "weights": []}]')

        def to_json(self, systematics=None):
            return self.JSON

    def hwu_writer(self, write_hwu):
        """A MadgraphProcess reduced to what write_hwu() looks at."""
        from madgraph.various.banner import RunCardMG7
        process = self.launch.MadgraphProcess.__new__(self.launch.MadgraphProcess)
        process.run_card = RunCardMG7()
        process.run_card['run']['write_hwu'] = write_hwu
        process.run_path = self.me_dir
        process.systematics = None
        return process

    def hwu_path(self):
        return os.path.join(self.me_dir, self.launch.MadgraphProcess.hwu_file_name)

    def test_hwu_is_off_by_default(self):
        """the same numbers are in info.json: the second file is opt-in"""
        self.assertIs(self.launch.MadgraphProcess.__new__(
            self.launch.MadgraphProcess).__class__, self.launch.MadgraphProcess)
        from madgraph.various.banner import RunCardMG7
        self.assertIs(RunCardMG7()['run']['write_hwu'], False)
        process = self.hwu_writer(False)
        process.write_hwu(self.FakeHistograms())
        self.assertFalse(os.path.exists(self.hwu_path()))

    def test_hwu_is_written_when_asked(self):
        process = self.hwu_writer(True)
        process.write_hwu(self.FakeHistograms())
        with open(self.hwu_path()) as stream:
            text = stream.read()
        self.assertIn('<histogram> 2 "jet-pt', text)
        self.assertTrue(text.startswith('##& xmin & xmax'))

    def test_no_histograms_no_file(self):
        """nothing to write when [histograms] is empty, switch or no switch"""
        process = self.hwu_writer(True)
        process.write_hwu(None)
        self.assertFalse(os.path.exists(self.hwu_path()))

    # -- the Events/<run>/plots switch --------------------------------------
    def plot_maker(self, make_plots):
        from madgraph.various.banner import RunCardMG7
        process = self.launch.MadgraphProcess.__new__(self.launch.MadgraphProcess)
        process.run_card = RunCardMG7()
        process.run_card['run']['make_plots'] = make_plots
        process.run_path = self.me_dir
        process.systematics = None
        return process

    def plot_dir(self):
        return os.path.join(self.me_dir,
                            self.launch.MadgraphProcess.plot_dir_name)

    def test_plots_are_on_by_default(self):
        from madgraph.various.banner import RunCardMG7
        self.assertIs(RunCardMG7()['run']['make_plots'], True)

    def test_plots_can_be_switched_off(self):
        process = self.plot_maker(False)
        process.make_plots(self.FakeHistograms())
        self.assertFalse(os.path.exists(self.plot_dir()))

    def test_no_histograms_no_plot_directory(self):
        process = self.plot_maker(True)
        process.make_plots(None)
        self.assertFalse(os.path.exists(self.plot_dir()))

    def test_missing_matplotlib_does_not_fail_the_run(self):
        """the numbers are in info.json: a run without the backend carries on"""
        from madgraph.iolibs.template_files.mg7 import plots
        saved = plots._pyplot

        def no_matplotlib():
            raise plots.BackendMissing("No module named 'matplotlib'")

        plots._pyplot = no_matplotlib
        try:
            process = self.plot_maker(True)
            process.make_plots(self.FakeHistograms())   # must not raise
        finally:
            plots._pyplot = saved
        self.assertFalse(os.path.exists(self.plot_dir()))

    # -- "set histograms ..." in the launch question ------------------------
    def histogram_selector(self, card_text=None):
        """An MG7Selector reduced to what do_set needs, on this output's card."""
        from madgraph.various.banner import RunCardMG7
        path = os.path.join(self.me_dir, 'Cards', 'run_card.toml')
        default_path = os.path.join(self.me_dir, 'Cards',
                                    'run_card_default.toml')
        card = RunCardMG7()
        card.dynamic_sections['histograms'].update({
            'jet_1-pt': {'min': 0., 'max': 500., 'bin_count': 50},
            'sqrt_s': {'min': 0., 'max': 2000., 'bin_count': 50},
        })
        card.write(path)
        card.write(default_path)
        cwd = os.getcwd()
        os.chdir(self.me_dir)
        try:
            selector_class, _ = self.launch.build_selector_cmd()
        finally:
            os.chdir(cwd)
        obj = selector_class.__new__(selector_class)
        obj.run_card = RunCardMG7(path, consistency=False)
        obj.paths = {'run_default': default_path}
        obj.modified_card = set()
        return obj

    def test_set_histograms_off_removes_them_all(self):
        obj = self.histogram_selector()
        self.assertTrue(obj.run_card['histograms'])
        obj.do_set('histograms OFF')
        self.assertEqual(dict(obj.run_card['histograms']), {})
        self.assertIn('run', obj.modified_card)

    def test_set_histograms_default_puts_them_back(self):
        obj = self.histogram_selector()
        obj.do_set('histograms off')
        obj.do_set('histograms default')
        self.assertIn('jet_1-pt', obj.run_card['histograms'])

    def test_set_histograms_rejects_anything_else(self):
        """a value that is not OFF/default leaves the section alone"""
        obj = self.histogram_selector()
        before = dict(obj.run_card['histograms'])
        obj.do_set('histograms 42')
        self.assertEqual(dict(obj.run_card['histograms']), before)
        self.assertNotIn('run', obj.modified_card)

    # -- "set <madevent cut name> ..." in the launch question ---------------
    def cut_selector(self):
        """histogram_selector() whose generic editor must not be reached: it
        is what answered "WARNING: invalid set command mmll 200"."""
        obj = self.histogram_selector()
        generic = mock.patch.object(type(obj).__mro__[1], 'do_set')
        self.generic_do_set = generic.start()
        self.addCleanup(generic.stop)
        return obj

    def test_set_madevent_cut_names(self):
        """set mmll/mmllmax/... edit the [cuts] entry from_LO maps them to"""
        obj = self.cut_selector()
        obj.do_set('mmll 200')
        self.assertEqual(dict(obj.run_card['cuts']['lepton-sfos_pair_mass']),
                         {'min': 200.0})
        self.assertIn('run', obj.modified_card)
        obj.do_set('run_card mmllmax 2*100+100')
        self.assertEqual(dict(obj.run_card['cuts']['lepton-sfos_pair_mass']),
                         {'min': 200.0, 'max': 300.0})
        obj.do_set('mmll 0')
        obj.do_set('mmllmax -1')
        self.assertNotIn('lepton-sfos_pair_mass', obj.run_card['cuts'])
        # sqrt_s is also a histogram name: only the cut moves
        obj.do_set('dsqrt_shat 500')
        self.assertEqual(dict(obj.run_card['cuts']['sqrt_s']), {'min': 500.0})
        self.assertEqual(dict(obj.run_card['histograms']['sqrt_s']),
                         {'min': 0., 'max': 2000., 'bin_count': 50})
        self.generic_do_set.assert_not_called()

    def test_set_madevent_cut_name_rejects_non_numbers(self):
        obj = self.cut_selector()
        before = dict(obj.run_card['cuts'])
        obj.do_set('ptj abc')
        self.assertEqual(dict(obj.run_card['cuts']), before)
        self.assertNotIn('run', obj.modified_card)
        self.generic_do_set.assert_not_called()

    def test_set_madevent_scale_and_pdg_names(self):
        """the integer scale/strategy codes, maxjetflavor and the per-pdg
        cuts are translated. The generic editor aborted a launch script on
        the integer codes."""
        obj = self.cut_selector()
        obj.do_set('dynamical_scale_choice 1')
        self.assertEqual(obj.run_card['beam']['dynamical_scale_choice'],
                         'transverse_energy')
        obj.do_set('dynamical_scale_choice 0')     # no equivalent: kept
        self.assertEqual(obj.run_card['beam']['dynamical_scale_choice'],
                         'transverse_energy')
        obj.do_set('sde_strategy 2')
        self.assertEqual(obj.run_card['phasespace']['sde_strategy'], 'denominators')
        obj.do_set('maxjetflavor 5')
        self.assertIn(5, obj.run_card['multiparticles']['jet'])
        obj.do_set('pt_min_pdg {6: 100}')
        self.assertEqual(dict(obj.run_card['cuts']['pdg6-pt']), {'min': 100.0})
        self.assertIn('run', obj.modified_card)
        self.generic_do_set.assert_not_called()

    def test_set_mg7_values_still_reach_the_editor(self):
        obj = self.cut_selector()
        obj.do_set('dynamical_scale_choice transverse_mass')
        obj.do_set('sde_strategy denominators')
        obj.do_set('dynamical_scale_choice HT/4')
        self.assertEqual(self.generic_do_set.call_count, 3)

    def test_per_beam_fixed_fac_scale_is_reported_when_the_question_closes(self):
        obj = self.cut_selector()
        parent = type(obj).__mro__[1]
        with mock.patch.object(parent, 'check_card_consistency'):
            obj.do_set('fixed_fac_scale1 T')
            obj.do_set('fixed_fac_scale2 F')
            with self.assertLogs('madgraph7', level='WARNING') as logs:
                obj.check_card_consistency()
        self.assertEqual(len(logs.output), 1)
        self.assertIn('fixed_fac_scale1 = True and fixed_fac_scale2 = False',
                      logs.output[0])

    def test_set_madevent_cut_name_scan_is_refused_by_name(self):
        obj = self.cut_selector()
        before = dict(obj.run_card['cuts'])
        with self.assertLogs('madgraph.stdout', level='WARNING') as logs:
            obj.do_set('ptj scan:[20, 30]')
        self.assertIn('cannot be scanned', logs.output[0])
        self.assertEqual(dict(obj.run_card['cuts']), before)
        self.generic_do_set.assert_not_called()

    def test_set_lo_param_bugs_are_not_hidden(self):
        """only a bad value is reported as "ignoring set ..."; an error of
        the translation itself is not swallowed"""
        obj = self.cut_selector()
        with mock.patch.object(type(obj.run_card), 'set_lo_param',
                               side_effect=TypeError('bug')):
            self.assertRaises(TypeError, obj.do_set, 'maxjetflavor 4')

    def test_per_beam_state_is_cleared_when_the_check_fails(self):
        obj = self.cut_selector()
        parent = type(obj).__mro__[1]
        obj.do_set('fixed_fac_scale1 T')
        obj.do_set('fixed_fac_scale2 F')
        with mock.patch.object(parent, 'check_card_consistency',
                               side_effect=RuntimeError('card check')):
            with self.assertLogs('madgraph7', level='WARNING') as logs:
                self.assertRaises(RuntimeError, obj.check_card_consistency)
        self.assertIn('fixed_fac_scale1 = True', logs.output[0])
        self.assertEqual(obj.run_card.lo_set_warnings(), [])

    def test_set_madevent_beam_names(self):
        """set ebeam1/ebeam/pdlabel2/lpp, as a madevent launch script writes
        them, edit the per-beam [beam] entries"""
        obj = self.histogram_selector()
        obj.do_set('ebeam 6.8 TeV')
        obj.do_set('ebeam1 7 TeV')
        obj.do_set('pdlabel2 cteq6l1')
        obj.do_set('lpp 1')
        beam = obj.run_card['beam']
        self.assertEqual((beam['ebeam1'], beam['ebeam2']), (7000.0, 6800.0))
        self.assertEqual(beam['pdf2'], 'cteq6l1')
        self.assertNotEqual(beam['pdf1'], 'cteq6l1')
        self.assertFalse(beam['leptonic'])
        self.assertIn('run', obj.modified_card)
        # a value mg7 cannot represent leaves the card alone
        obj.modified_card.clear()
        obj.do_set('pdlabel1 none')
        self.assertNotEqual(obj.run_card['beam']['pdf1'], 'none')
        self.assertNotIn('run', obj.modified_card)

    def test_invalid_set_is_an_error_in_scripted_runs(self):
        """the set commands the mg7 editor rejects stop a launch script"""
        from madgraph import InvalidCmd
        obj = self.cut_selector()
        obj.mother_interface = types.SimpleNamespace(inputfile=iter([]))
        for line in ('pdf NNPDF', 'histograms 42', 'jet-pt.mni 20', 'jet-pt.min abc',
                     'mmll abc', 'ptj scan:[20, 30]', 'iseed xx',
                     'dynamical_scale_choice 99', 'sde_strategy 7', 'pdlabel1 none',
                     'custom_functions.bogus.x 1'):
            self.assertRaises(InvalidCmd, obj.do_set, line)
        self.assertNotIn('run', obj.modified_card)
        self.assertNotIn('jet-pt.mni', obj.run_card['cuts'])
        # madevent's -1/0 scale choices only warn, so ported scripts still run
        with self.assertLogs('madgraph7', level='WARNING'):
            obj.do_set('dynamical_scale_choice -1')
        self.generic_do_set.assert_not_called()

    def test_non_number_cut_bound_only_warns_interactively(self):
        """set jet-pt.min abc is rejected through invalid_set: a warning,
        not an exception, when nobody scripts the question"""
        obj = self.cut_selector()
        before = dict(obj.run_card['cuts']['jet-pt'])
        with self.assertLogs('madgraph.stdout', level='WARNING') as logs:
            obj.do_set('jet-pt.min abc')
        self.assertIn("invalid 'set jet-pt.min abc'", logs.output[0])
        self.assertEqual(dict(obj.run_card['cuts']['jet-pt']), before)
        self.assertNotIn('run', obj.modified_card)

    def test_energy_unit_on_a_dimensionless_entry_is_refused(self):
        """a unit converts an energy cut but stops a script on a dimensionless one"""
        obj = self.cut_selector()
        obj.do_set('jet-pt.min 1 TeV')
        self.assertEqual(obj.run_card['cuts']['jet-pt']['min'], 1000.)
        cuts = copy.deepcopy(obj.run_card['cuts'])
        obj.mother_interface = types.SimpleNamespace(inputfile=iter([]))
        for line in ('jet-eta_abs.max 4 TeV', 'etaj 2 TeV', 'drjj 1 MeV',
                     'iseed 1 GeV', 'nevents 1 TeV'):
            self.assertRaises(madgraph.InvalidCmd, obj.do_set, line)
        self.assertEqual(obj.run_card['cuts'], cuts)
        self.generic_do_set.assert_not_called()

    def test_collider_and_scale_shortcuts_go_through_invalid_set(self):
        """set lhc/fixed_scale/e_cm with a value that is not a number warn
        interactively (nothing raised) and stop a launch script"""
        lines = ('lhc abc', 'lep abc', 'fixed_scale abc', 'e_cm abc', 'e_cm scan:[1,2]')
        obj = self.cut_selector()
        beam = dict(obj.run_card['beam'])
        with self.assertLogs('madgraph.stdout', level='WARNING') as logs:
            for line in lines:
                obj.do_set(line)
        self.assertEqual(len(logs.output), len(lines), logs.output)
        obj.mother_interface = types.SimpleNamespace(inputfile=iter([]))
        for line in lines:
            self.assertRaises(madgraph.InvalidCmd, obj.do_set, line)
        self.assertEqual(dict(obj.run_card['beam']), beam)
        self.assertNotIn('run', obj.modified_card)
        self.generic_do_set.assert_not_called()

    def test_unknown_lhaid_only_warns_in_scripted_runs(self):
        from madgraph.various import banner as banner_mod
        obj = self.histogram_selector()
        obj.mother_interface = types.SimpleNamespace(inputfile=iter([]))
        error = banner_mod.UnknownPDFSet('lhaid = 1: unknown LHAPDF id')
        with mock.patch.object(type(obj.run_card), 'set_lo_beam_param',
                               side_effect=error):
            with self.assertLogs('madgraph7', level='WARNING') as logs:
                obj.do_set('lhaid 1')
        self.assertIn('unknown LHAPDF id', logs.output[0])
        self.assertNotIn('run', obj.modified_card)

    def test_set_custom_function_parameter(self):
        """custom function parameters are not run_card keys: set them anyway"""
        obj = self.cut_selector()
        obj.do_set('custom_functions.cut_parameters.pt_min 30')
        self.assertEqual(obj.run_card.custom_parameters('cut'), {'pt_min': 30.0})
        # the name is matched as the custom function writes it: case kept
        obj.do_set('custom_functions.cut_parameters.etaMax 2.5')
        self.assertEqual(obj.run_card.custom_parameters('cut'),
                         {'pt_min': 30.0, 'etaMax': 2.5})
        self.assertIn('run', obj.modified_card)
        self.generic_do_set.assert_not_called()

    def test_no_post_processing_keeps_the_npy_output(self):
        self.assertEqual(self.output_format({}), 'compact_npy')
        self.assertEqual(self.output_format(None), 'compact_npy')
        self.assertEqual(
            self.output_format({'shower': 'OFF', 'analysis': 'Not Avail.'}),
            'compact_npy')

    def test_run_name_is_written_to_the_run_card(self):
        cmd = self.make_cmd()
        cwd = os.getcwd()
        os.chdir(self.me_dir)
        try:
            cmd._set_run_name('myrun')
            with open(os.path.join('Cards', 'run_card.toml')) as stream:
                self.assertIn('myrun', stream.read())
        finally:
            os.chdir(cwd)

    def test_options_come_from_the_output_dir_not_the_cwd(self):
        """Cards/me5_configuration.txt of the *output* has the last word.

        bin/generate_events chdirs into the output first, so reading the config
        relative to the working directory was right there. In process MG5 has
        not moved when the interface is built, so the directory has to be
        passed explicitly or the output's own configuration is ignored.
        """
        with open(os.path.join(self.me_dir, 'Cards',
                               'me5_configuration.txt'), 'w') as stream:
            stream.write('delphes_path = /somewhere/delphes\n')
        cmd = self.make_cmd()
        self.assertNotEqual(os.getcwd(), self.me_dir)
        self.assertEqual(cmd.options['delphes_path'], '/somewhere/delphes')

    def test_status_file_path_is_absolute(self):
        """The status file must not depend on the working directory.

        ms.StatusFile finishes its write from a C++ destructor, which runs when
        Python collects the process object -- for an interrupted run, that is
        after the launch command has restored the working directory. A relative
        path there throws a filesystem_error out of a destructor, i.e. aborts
        the whole of MG5.
        """
        recorded = []
        ms = self.launch.ms
        saved = ms.StatusFile
        ms.StatusFile = lambda path: recorded.append(path)
        cwd = os.getcwd()
        os.chdir(self.me_dir)
        try:
            process = self.launch.MadgraphProcess.__new__(
                self.launch.MadgraphProcess)
            process.run_card = {
                "run": {"run_name": "run"},
                "gridpack": {"output_dir": "", "temp_output_dir": ""},
            }
            process.init_event_dir()
        finally:
            os.chdir(cwd)
            ms.StatusFile = saved
        self.assertEqual(len(recorded), 1)
        self.assertTrue(os.path.isabs(recorded[0]), recorded[0])

    # ------------------------------------------------------------------
    # the model of the process (SubProcesses/model.txt)
    # ------------------------------------------------------------------
    def write_model_txt(self, reference, model_hash=''):
        with open(os.path.join(self.me_dir, 'SubProcesses', 'model.txt'),
                  'w') as stream:
            stream.write('%s\n%s\n' % (reference, model_hash))

    def test_get_model_returns_the_model_of_the_process(self):
        """The hook 'update dependent' and the auto-width handler go through.

        Without it the card question could only warn that it had failed to
        update the dependent parameters of the param_card.
        """
        self.write_model_txt(os.path.join(_MG_ROOT, 'models', 'sm'))
        cmd = self.make_cmd()
        model = cmd.get_model()
        self.assertTrue(model)
        self.assertEqual(model.get('name'), 'sm')
        # imported once: the question asks again at every 'update dependent'
        self.assertIs(cmd.get_model(), model)

    def test_get_model_keeps_the_restriction(self):
        """'sm-no_b_mass' is not 'sm': the restriction is part of the model the
        process was generated with, so the reference recorded at output time
        carries it and reloading has to honour it."""
        self.write_model_txt(os.path.join(_MG_ROOT, 'models', 'sm-no_b_mass'))
        model = self.make_cmd().get_model()
        self.assertTrue(model)
        self.assertEqual(model.get('name'), 'sm-no_b_mass')

    def test_get_model_reads_the_process_characteristics(self):
        """The complex-mass scheme is a property of the process, not of the
        card, so it comes from SubProcesses/proc_characteristics -- which is a
        ConfigFile, whose get() is the parameter accessor and takes no
        default."""
        with open(os.path.join(self.me_dir, 'SubProcesses',
                               'proc_characteristics'), 'w') as stream:
            stream.write('complex_mass_scheme = False\nnexternal = 4\n')
        self.write_model_txt(os.path.join(_MG_ROOT, 'models', 'sm'))
        self.assertIs(self.launch._proc_characteristic(
            self.me_dir)['complex_mass_scheme'], False)
        self.assertEqual(self.make_cmd().get_model().get('name'), 'sm')

    def test_get_model_is_none_without_a_recorded_model(self):
        """An output written before SubProcesses/model.txt existed: the caller
        warns, nothing raises."""
        self.assertIsNone(self.make_cmd().get_model())

    def test_update_dependent_uses_the_model(self):
        """End to end: the dependent parameters of the param_card are the ones
        the model computes from the free ones (here M_W from G_F/M_Z/alpha)."""
        from madgraph.interface.common_run_interface import AskforEditCard
        from madgraph.models.check_param_card import ParamCard

        self.write_model_txt(os.path.join(_MG_ROOT, 'models', 'sm'))
        path = os.path.join(self.me_dir, 'Cards', 'param_card.dat')
        shutil.copyfile(os.path.join(_MG_ROOT, 'tests', 'input_files',
                                     'param_card_sm.dat'), path)
        card = ParamCard(path)
        self.assertRaises(KeyError, card['mass'].get, (24,))

        modified = AskforEditCard.update_dependent(
            self.make_cmd(), self.me_dir, card, path, timer=0)

        self.assertTrue(modified)
        self.assertAlmostEqual(ParamCard(path)['mass'].get((24,)).value,
                               80.419, places=2)

    def test_a_slow_model_stays_a_timeout(self):
        """'update dependent' loads the model under an alarm whose handler
        raises a TimeOutError defined inside the caller. Swallowing it would
        report a model that does not import instead of one that is slow (which
        the caller knows how to explain, and how to force)."""
        class TimeOutError(Exception):
            pass

        self.write_model_txt(os.path.join(_MG_ROOT, 'models', 'sm'))
        self.launch._model_cache.clear()
        with mock.patch('madgraph.models.import_ufo.import_model',
                        side_effect=TimeOutError):
            self.assertRaises(TimeOutError, self.launch.load_process_model,
                              self.me_dir)
        # ... and it is not remembered as a model that cannot be imported
        self.assertFalse(self.launch._model_cache)

    def test_a_changed_model_is_reported(self):
        """The hash on the second line is what tells a model that moved on from
        the one the matrix element was written for."""
        self.write_model_txt(os.path.join(_MG_ROOT, 'models', 'sm'),
                             'not-the-hash-of-that-model')
        self.launch._model_hash_warned.clear()
        with mock.patch.object(self.launch.logger, 'warning') as warning:
            self.assertEqual(self.launch.read_stored_model(self.me_dir),
                             os.path.join(_MG_ROOT, 'models', 'sm'))
            self.assertEqual(warning.call_count, 1)
            self.assertIn('has changed', warning.call_args[0][0])
            # only once per directory: this runs on every card question
            self.launch.read_stored_model(self.me_dir)
            self.assertEqual(warning.call_count, 1)

    def test_setup_logging_defers_to_an_existing_handler(self):
        """In process MG5 owns the loggers; adding a second handler would print
        every line twice."""
        import logging
        lg = logging.getLogger('madgraph')
        handler = logging.NullHandler()
        lg.addHandler(handler)
        saved = self.launch._TOOL_LOGGING_READY
        self.launch._TOOL_LOGGING_READY = False
        try:
            before = list(lg.handlers)
            self.launch._setup_logging()
            self.assertEqual(list(lg.handlers), before)
        finally:
            self.launch._TOOL_LOGGING_READY = saved
            lg.removeHandler(handler)


if __name__ == '__main__':
    unittest.main()


class TestPostProcessingIsQuiet(unittest.TestCase):
    """MG7RunCmd is an internal adapter, not an interface someone started.

    MadEventCmd.__init__ prints the MADEVENT welcome banner and a run of
    "load configuration from ..." lines, which is right for ./bin/madevent and
    wrong in the middle of a run that has already introduced itself.
    """

    def setUp(self):
        import io

        self.captured = io.StringIO()
        self.handler = logging.StreamHandler(self.captured)
        self.logger = logging.getLogger('madevent.stdout')
        self.saved = self.logger.level
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.saved)

    def test_info_is_dropped_during_setup(self):
        from madgraph.iolibs.template_files.mg7.run_interface import _quiet_setup

        with _quiet_setup():
            self.logger.info('W E L C O M E to')
            self.logger.info('load configuration from somewhere')
        self.assertNotIn('W E L C O M E', self.captured.getvalue())
        self.assertNotIn('load configuration', self.captured.getvalue())

    def test_warnings_still_get_through(self):
        """Quiet is not silent: anything that needs saying still says it."""

        from madgraph.iolibs.template_files.mg7.run_interface import _quiet_setup

        with _quiet_setup():
            self.logger.warning('something worth knowing')
        self.assertIn('something worth knowing', self.captured.getvalue())

    def test_the_level_is_restored_afterwards(self):
        from madgraph.iolibs.template_files.mg7.run_interface import _quiet_setup

        with _quiet_setup():
            pass
        self.logger.info('back to normal')
        self.assertIn('back to normal', self.captured.getvalue())

    def test_it_restores_the_level_after_a_failure(self):
        from madgraph.iolibs.template_files.mg7.run_interface import _quiet_setup

        try:
            with _quiet_setup():
                raise RuntimeError('setup blew up')
        except RuntimeError:
            pass
        self.logger.info('back to normal')
        self.assertIn('back to normal', self.captured.getvalue())


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class MG7ReweightingTest(unittest.TestCase):
    """Reweighting runs once per generation, after the systematics, by the
    backend prepare_reweighting() chose."""

    CARD = "change helicity False\nlaunch --rwgt_name=mt\nset MT 175\n"

    def setUp(self):
        from madgraph.iolibs.template_files.mg7 import launch
        self.launch = launch
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        self.me_dir = make_mg7_dir(os.path.join(self.tmpdir.name, 'PROC'))
        os.chdir(self.me_dir)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmpdir.cleanup()

    def write_card(self, text, path=os.path.join('Cards', 'reweight_card.dat')):
        with open(path, 'w') as f:
            f.write(text)
        return path

    def prepare(self, switch, cli=None, f2py=True, madtrex=True):
        from madgraph.various import reweight_backend as rb
        probe = rb.probe_environment

        def fake_probe(me_dir, options=None, **kwargs):
            env = probe(me_dir, options, has_f2py=f2py,
                        madtrex_install_dir=self.tmpdir.name)
            env.madtrex_installed = madtrex
            return env
        with mock.patch.object(rb, 'probe_environment', fake_probe):
            return self.launch.prepare_reweighting(switch, cli, {'mg5_path': '/mg5'})

    def test_off_or_without_launch_does_nothing(self):
        self.assertIsNone(self.prepare({'reweight': 'OFF'}))
        self.assertIsNone(self.prepare({'reweight': 'ON'}))  # no card
        self.write_card("# launch\n")
        self.assertIsNone(self.prepare({'reweight': 'ON'}))

    def test_auto_prefers_madtrex(self):
        self.write_card(self.CARD)
        decision = self.prepare({'reweight': 'ON'})
        self.assertEqual((decision.backend, decision.card_kind), ('madtrex', 'param'))
        self.assertEqual(self.prepare({'reweight': 'ON'}, madtrex=False).backend, 'classic')

    def test_switch_and_command_line(self):
        self.write_card("change backend madtrex\n" + self.CARD)
        self.assertEqual(self.prepare({'reweight': 'classic'}).backend, 'classic')
        self.assertEqual(self.prepare({'reweight': 'classic'}, cli='madtrex').backend,
                         'madtrex')
        self.assertEqual(self.prepare({'reweight': 'density'}).card_kind, 'density')
        decision = self.prepare({'reweight': 'my_plugin'})
        self.assertEqual((decision.backend, decision.plugin), ('classic', 'my_plugin'))

    def test_impossible_request_fails_before_generating(self):
        from madgraph.various import reweight_backend as rb
        self.write_card(self.CARD)
        with self.assertRaises(rb.ReweightBackendError):
            self.prepare({'reweight': 'madtrex'}, madtrex=False)

    def test_install_reweight_card_keeps_the_old_one(self):
        self.write_card("old\n")
        new = self.write_card(self.CARD, os.path.join(self.tmpdir.name, 'rw.dat'))
        self.launch.install_reweight_card(new)
        with open(os.path.join('Cards', 'reweight_card.dat')) as f:
            self.assertEqual(f.read(), self.CARD)
        with open(os.path.join('Cards', 'reweight_card.dat.bak')) as f:
            self.assertEqual(f.read(), "old\n")

    def test_reweight_option_with_switch_on_reweights_once(self):
        """--reweight=CARD with the reweight switch already on: one run,
        after the systematics, and no 'reweight' among the tools."""
        card = self.write_card(self.CARD, os.path.join(self.tmpdir.name, 'rw.dat'))
        calls = []

        class Process(object):
            gridpack_run_mode = 'regular'
            run_path = 'Events/run_01'

        launch = self.launch

        def fake_run_generation(switch, reweight):
            launch.run_single(switch, reweight)

        def fake_tools(switch, process):
            calls.append(('tools', tuple(launch.selected_tools(switch))))

        def fake_prepare(switch, cli, options):
            calls.append(('prepare', switch.get('reweight'), cli))
            return 'DECISION'
        with mock.patch.object(launch, 'ask_edit_cards', return_value={'reweight': 'ON'}), \
                mock.patch.object(launch, 'prepare_reweighting', fake_prepare), \
                mock.patch.object(launch, 'force_lhe_output_if_needed'), \
                mock.patch.object(launch, 'run_generation', fake_run_generation), \
                mock.patch.object(launch, 'compute_auto_widths'), \
                mock.patch.object(launch, 'MadgraphProcess', lambda: mock.Mock(**{
                    'gridpack_run_mode': 'regular'})), \
                mock.patch.object(launch, 'run_lhe_postprocessing',
                                  lambda p: calls.append(('systematics',))), \
                mock.patch.object(launch, 'run_reweighting',
                                  lambda p, c, decision: calls.append(('reweight', decision))), \
                mock.patch.object(launch, 'run_selected_tools', fake_tools):
            launch.MG7Cmd(me_dir=self.me_dir).do_generate_events('--reweight=%s' % card)
        self.assertEqual(calls, [('prepare', 'ON', None), ('systematics',),
                                 ('reweight', 'DECISION'), ('tools', ())])
        with open(os.path.join(self.me_dir, 'Cards', 'reweight_card.dat')) as f:
            self.assertEqual(f.read(), self.CARD)

    def test_reweight_option_switches_reweighting_on(self):
        card = self.write_card(self.CARD, os.path.join(self.tmpdir.name, 'rw.dat'))
        seen = []
        launch = self.launch
        with mock.patch.object(launch, 'prepare_reweighting',
                               lambda switch, cli, options: seen.append((switch, cli))), \
                mock.patch.object(launch, 'force_lhe_output_if_needed'), \
                mock.patch.object(launch, 'run_generation'):
            launch.MG7Cmd(me_dir=self.me_dir).do_generate_events(
                '-f --reweight=%s --reweight-backend=classic' % card)
        self.assertEqual(seen, [({'reweight': 'ON'}, 'classic')])

    def run_card(self, output_format='lhe', time_of_flight=-1.0, legacy=False,
                 native=True):
        return {'run': {'output_format': output_format},
                'postprocessing': {'time_of_flight': time_of_flight,
                                   'systematics': legacy},
                'systematics': {'enable': native}}

    def test_madtrex_lhe_run_is_generated_as_npy(self):
        """An "lhe" run MadtRex reweights is generated as npy, unless an
        LHE-only step runs before the reweighting."""
        from madgraph.various import reweight_backend as rb
        madtrex, classic = rb.Decision('madtrex', 'param'), rb.Decision('classic', 'param')
        process = mock.Mock(gridpack_run_mode='regular', run_card=self.run_card())
        via_binary = self.launch.lhe_via_binary
        self.assertTrue(via_binary(process, madtrex))
        self.assertFalse(via_binary(process, classic))
        self.assertFalse(via_binary(process, None))
        for card in (self.run_card('lhe_npy'), self.run_card('compact_npy'),
                     self.run_card(time_of_flight=1.0),
                     self.run_card(legacy=True, native=False)):
            process.run_card = card
            self.assertFalse(via_binary(process, madtrex), card)
        # the legacy systematics.py only runs without madspace's systematics
        process.run_card = self.run_card(legacy=True, native=True)
        self.assertTrue(via_binary(process, madtrex))
        process.run_card, process.gridpack_run_mode = self.run_card(), 'minimal'
        self.assertFalse(via_binary(process, madtrex))

    def run_single_as_npy(self, reweighting):
        """run_single() of an "lhe" run MadtRex reweights, with the
        generation, the reweighting (reweighting(process)) and the npy->LHE
        conversion faked; returns (process, calls)."""
        from madgraph.various import reweight_backend as rb
        from madgraph.iolibs.template_files.mg7 import npy_to_lhe
        run_path = os.path.abspath(os.path.join('Events', 'run_01'))
        os.makedirs(run_path)
        calls = []
        process = mock.Mock(gridpack_run_mode='regular', run_card=self.run_card(),
                            run_path=run_path)

        def generate_events(output_format=None):
            calls.append(('generate', output_format))
            process.events_path = os.path.join(run_path, 'events.npy')
            for name in ('events.npy', 'header.lhe'):
                open(os.path.join(run_path, name), 'w').close()
        process.generate_events = generate_events

        def convert(path):
            calls.append(('convert', os.path.basename(path)))
            lhe = os.path.join(run_path, 'events.lhe.gz')
            open(lhe, 'w').close()
            return lhe

        def run_reweighting(p, card, decision):
            calls.append(('reweight', p.events_path))
            reweighting(p)
        launch = self.launch
        with mock.patch.object(launch, 'compute_auto_widths'), \
                mock.patch.object(launch, 'MadgraphProcess', lambda: process), \
                mock.patch.object(launch, 'run_lhe_postprocessing'), \
                mock.patch.object(launch, 'run_reweighting', run_reweighting), \
                mock.patch.object(npy_to_lhe, 'convert', convert):
            launch.run_single(None, rb.Decision('madtrex', 'param'))
        return process, calls

    def test_madtrex_writes_the_requested_lhe(self):
        def reweighting(process):
            with open(os.path.join(process.run_path, 'reweighting.json'), 'w') as f:
                json.dump({'events': process.events_path}, f)
        process, calls = self.run_single_as_npy(reweighting)
        npy = os.path.join(process.run_path, 'events.npy')
        self.assertEqual(calls, [('generate', 'lhe_npy'), ('reweight', npy),
                                 ('convert', 'events.npy')])
        self.assertEqual(sorted(os.listdir(process.run_path)),
                         ['events.lhe.gz', 'reweighting.json'])
        lhe = os.path.join(process.run_path, 'events.lhe.gz')
        self.assertEqual(process.events_path, lhe)
        with open(os.path.join(process.run_path, 'reweighting.json')) as f:
            self.assertEqual(json.load(f)['events'], lhe)

    def test_failed_madtrex_still_writes_the_requested_lhe(self):
        def reweighting(process):
            raise RuntimeError('madtrex failed')
        with self.assertLogs('madevent', 'WARNING') as logs:
            process, calls = self.run_single_as_npy(reweighting)
        self.assertEqual([c[0] for c in calls], ['generate', 'reweight', 'convert'])
        self.assertTrue(any('reweighting failed: madtrex failed' in line
                            for line in logs.output))
        self.assertIn('events.lhe.gz', os.listdir(process.run_path))
        self.assertNotIn('events.npy', os.listdir(process.run_path))

    def test_the_decision_is_logged_and_recorded(self):
        from madgraph.various import reweight_backend as rb
        card = self.write_card("change backend auto\n" + self.CARD)
        run_path = os.path.join('Events', 'run_01')
        os.makedirs(run_path)
        decision = rb.Decision('madtrex', 'param',
                               ['MG7 output', 'madtrex installed',
                                'no classic-only directives in {card}'])

        class Process(object):
            events_path = os.path.join(run_path, 'events.npy')
            run_card = {'run': {'output_format': 'lhe_npy'}}
        Process.run_path = run_path
        seen = {}

        def fake_madtrex(card, events_path, card_kind, process_directory,
                         drop_stored_matrix_elements=False):
            with open(card) as f:
                seen['card'] = f.read()
            seen['kind'] = card_kind
        from madgraph.iolibs.template_files.mg7 import madtrex
        with mock.patch.object(madtrex, 'run_reweighting', fake_madtrex), \
                self.assertLogs('madevent', 'INFO') as logs:
            self.launch.run_reweighting(Process(), card, decision=decision)
        self.assertIn('Reweighting with madtrex (auto: MG7 output, madtrex installed, '
                      'no classic-only directives in reweight_card.dat)', logs.output[0])
        self.assertNotIn('\nchange backend', '\n' + seen['card'])
        self.assertEqual(seen['kind'], 'param')
        with open(os.path.join(run_path, 'reweighting.json')) as f:
            record = json.load(f)
        self.assertEqual((record['backend'], record['source'], record['weight_group']),
                         ('madtrex', 'auto', 'tearex_reweighting'))
        self.assertIn('change backend auto', record['card'])
        # the weights went into the run's own event file
        self.assertEqual(record['events'], Process.events_path)

    def test_madtrex_refreshes_the_tools_copy_of_lhe_events(self):
        import gzip
        from madgraph.various import reweight_backend as rb
        card = self.write_card(self.CARD)
        run_path = os.path.join('Events', 'run_01')
        os.makedirs(run_path)
        events = os.path.join(run_path, 'events.lhe.gz')
        copy = os.path.join(run_path, 'unweighted_events.lhe.gz')
        for path in (events, copy):
            with gzip.open(path, 'wt') as f:
                f.write('before\n')

        class Process(object):
            events_path = events
            run_card = {'run': {'output_format': 'lhe'}}
        Process.run_path = run_path

        def fake_madtrex(card, events_path, card_kind, process_directory,
                         drop_stored_matrix_elements=False):
            with gzip.open(events_path, 'wt') as f:
                f.write('reweighted\n')
        from madgraph.iolibs.template_files.mg7 import madtrex
        with mock.patch.object(madtrex, 'run_reweighting', fake_madtrex):
            self.launch.run_reweighting(Process(), card,
                                        decision=rb.Decision('madtrex', 'param'))
        with gzip.open(copy, 'rt') as f:
            self.assertEqual(f.read(), 'reweighted\n')

    def test_sync_lhe_copy(self):
        import gzip
        from madgraph.iolibs.template_files.mg7.run_interface import sync_lhe_copy
        with gzip.open('a.lhe.gz', 'wt') as f:
            f.write('events\n')
        sync_lhe_copy('a.lhe.gz', 'b.lhe')
        with open('b.lhe') as f:
            self.assertEqual(f.read(), 'events\n')
        sync_lhe_copy('b.lhe', 'c.lhe.gz')
        with gzip.open('c.lhe.gz', 'rt') as f:
            self.assertEqual(f.read(), 'events\n')
        sync_lhe_copy('c.lhe.gz', 'a.lhe.gz')
        self.assertFalse(os.path.exists('a.lhe.gz.tmp'))

    def test_compact_npy_is_refused_for_madtrex(self):
        from madgraph.various import reweight_backend as rb
        card = self.write_card(self.CARD)
        os.makedirs(os.path.join('Events', 'run_01'))

        class Process(object):
            run_path = os.path.join('Events', 'run_01')
            run_card = {'run': {'output_format': 'compact_npy'}}
        with self.assertRaisesRegex(ValueError, 'compact_npy'):
            self.launch.run_reweighting(Process(), card,
                                        decision=rb.Decision('madtrex', 'param'))

    def test_reweighting_is_not_a_post_processing_tool(self):
        self.assertEqual(self.launch.selected_tools({'reweight': 'ON'}), [])

    def test_a_failing_reweighting_does_not_stop_the_run(self):
        """The events are kept and MadSpin/shower/analyses still run; the
        failure is logged with how to reweight again."""
        os.makedirs(os.path.join('Events', 'run_01'))
        launch = self.launch
        process = mock.Mock(gridpack_run_mode='regular',
                            run_path=os.path.abspath(os.path.join('Events', 'run_01')))
        calls = []

        def failing(p, c, decision):
            raise RuntimeError('SPINUP 9 in event 3')
        with mock.patch.object(launch, 'compute_auto_widths'), \
                mock.patch.object(launch, 'MadgraphProcess', lambda: process), \
                mock.patch.object(launch, 'run_lhe_postprocessing'), \
                mock.patch.object(launch, 'run_reweighting', failing), \
                mock.patch.object(launch, 'run_selected_tools',
                                  lambda switch, p: calls.append('tools')), \
                self.assertLogs('madevent', 'WARNING') as logs:
            launch.run_single({'reweight': 'ON', 'shower': 'Pythia8'}, 'DECISION')
        self.assertEqual(calls, ['tools'])
        text = '\n'.join(logs.output)
        self.assertIn('reweighting failed: SPINUP 9 in event 3', text)
        self.assertIn("'reweight run_01'", text)
        self.assertTrue(os.path.exists(os.path.join('Events', 'run_01',
                                                    'reweighting_crash.log')))

    def test_existing_run_finds_the_event_file_and_its_format(self):
        import gzip
        for run, files in (('lhe', ['events.lhe.gz']),
                           ('lhe_npy', ['events.npy', 'header.lhe', 'events.lhe.gz']),
                           ('compact_npy', ['events.npy', 'lhe_completer.json'])):
            os.makedirs(os.path.join('Events', run))
            for name in files:
                open(os.path.join('Events', run, name), 'w').close()
        os.makedirs(os.path.join('Events', 'empty'))
        for run, events in (('lhe', 'events.lhe.gz'), ('lhe_npy', 'events.npy'),
                            ('compact_npy', 'events.npy')):
            existing = self.launch.ExistingRun(run)
            self.assertEqual(existing.run_card['run']['output_format'], run)
            self.assertEqual(os.path.basename(existing.events_path), events)
        self.assertEqual(self.launch.list_runs(), ['compact_npy', 'lhe', 'lhe_npy'])
        with self.assertRaisesRegex(ValueError, 'no event file'):
            self.launch.ExistingRun('empty')
        with self.assertRaisesRegex(ValueError, "no run 'run_09'.*runs: compact_npy"):
            self.launch.ExistingRun('run_09')

    def test_reweight_command_reweights_an_existing_run(self):
        os.makedirs(os.path.join('Events', 'run_01'))
        open(os.path.join('Events', 'run_01', 'events.lhe.gz'), 'w').close()
        card = self.write_card(self.CARD, os.path.join(self.tmpdir.name, 'rw.dat'))
        launch = self.launch
        cmd = launch.MG7Cmd(me_dir=self.me_dir)
        seen = []

        def fake_prepare(switch, cli, options):
            seen.append(('prepare', switch, cli))
            return 'DECISION'

        def fake_run(process, card_path, decision):
            seen.append(('run', os.path.basename(process.run_path),
                         process.run_card['run']['output_format'], decision))
        with mock.patch.object(launch, 'prepare_reweighting', fake_prepare), \
                mock.patch.object(launch, 'run_reweighting', fake_run):
            cmd.do_reweight('run_01 --reweight=%s --reweight-backend=madtrex' % card)
        self.assertEqual(seen, [('prepare', {'reweight': 'ON'}, 'madtrex'),
                                ('run', 'run_01', 'lhe', 'DECISION')])
        with open(os.path.join('Cards', 'reweight_card.dat')) as f:
            self.assertEqual(f.read(), self.CARD)
        with self.assertRaisesRegex(cmd.InvalidCmd, r'give the run to reweight \(run_01\)'):
            cmd.do_reweight('')
        with self.assertRaisesRegex(cmd.InvalidCmd, "no run 'run_02'"):
            cmd.do_reweight('run_02')


class ReweightSwitchTest(unittest.TestCase):
    """The reweight switch of the launch question: an MG7 output offers the
    backends that can run (ON is auto); a madevent output keeps its list."""

    OPTIONS = {'pythia-pgs_path': None, 'pythia8_path': None,
               'madanalysis5_path': None, 'exrootanalysis_path': None,
               'delphes_path': None, 'rivet_path': None, 'mg5_path': '/mg5',
               'f2py_compiler': None}

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.mg7 = make_mg7_dir(os.path.join(self.tmpdir.name, 'MG7'))
        self.madevent = os.path.join(self.tmpdir.name, 'ME')
        os.makedirs(os.path.join(self.madevent, 'Cards'))
        os.makedirs(os.path.join(self.madevent, 'bin'))
        open(os.path.join(self.madevent, 'bin', 'madevent'), 'w').close()

    def tearDown(self):
        self.tmpdir.cleanup()

    def question(self, me_dir, f2py, madtrex, card=None):
        from madgraph.interface import madevent_interface as me
        from madgraph.various import reweight_backend as rb
        if card is not None:
            with open(os.path.join(me_dir, 'Cards', 'reweight_card.dat'), 'w') as f:
                f.write(card)
        question = me.AskRun.__new__(me.AskRun)
        question.me_dir = me_dir
        question.mother_interface = mock.Mock(plugin_path=[])
        question.switch = {}
        with mock.patch.object(me.misc, 'has_f2py', return_value=f2py), \
                mock.patch.object(rb, 'madtrex_is_installed', return_value=madtrex):
            question.check_available_module(dict(self.OPTIONS))
            question.set_default_reweight()
        return question

    def test_mg7_with_both_backends(self):
        q = self.question(self.mg7, f2py=True, madtrex=True)
        self.assertEqual(q.get_allowed_reweight(),
                         ['OFF', 'ON', 'classic', 'madtrex', 'density'])
        self.assertEqual(q.switch['reweight'], 'OFF')

    def test_mg7_without_f2py(self):
        q = self.question(self.mg7, f2py=False, madtrex=True)
        self.assertEqual(q.get_allowed_reweight(), ['OFF', 'ON', 'madtrex'])

    def test_mg7_without_madtrex(self):
        q = self.question(self.mg7, f2py=True, madtrex=False)
        self.assertEqual(q.get_allowed_reweight(), ['OFF', 'ON', 'classic', 'density'])

    def test_mg7_with_neither(self):
        q = self.question(self.mg7, f2py=False, madtrex=False)
        self.assertEqual(q.get_allowed_reweight(), [])
        self.assertEqual(q.switch['reweight'], 'Not Avail.')

    def test_madevent_is_unchanged(self):
        for madtrex in (True, False):
            q = self.question(self.madevent, f2py=True, madtrex=madtrex)
            self.assertEqual(q.get_allowed_reweight(), ['OFF', 'ON', 'density'])
        q = self.question(self.madevent, f2py=False, madtrex=True)
        self.assertEqual(q.get_allowed_reweight(), [])

    def test_switch_value_from_the_card(self):
        q = self.question(self.mg7, True, True, card="change backend madtrex\nlaunch\n")
        self.assertEqual(q.switch['reweight'], 'madtrex')
        q = self.question(self.mg7, True, True, card="change backend auto\nlaunch\n")
        self.assertEqual(q.switch['reweight'], 'ON')
        q = self.question(self.mg7, True, True,
                          card="change particle_in_density_matrix 3\nlaunch\n")
        self.assertEqual(q.switch['reweight'], 'density')
        # a backend that cannot run here is just ON
        q = self.question(self.mg7, True, False, card="change backend madtrex\nlaunch\n")
        self.assertEqual(q.switch['reweight'], 'ON')
        # a madevent output does not read the backend line
        q = self.question(self.madevent, True, True, card="change backend classic\nlaunch\n")
        self.assertEqual(q.switch['reweight'], 'ON')

    def test_a_backend_picked_in_the_question_survives_the_card_sync(self):
        q = self.question(self.mg7, True, True, card="launch\nset MT 175\n")
        self.assertEqual(q.switch['reweight'], 'ON')
        for value in ('classic', 'madtrex'):
            q.switch['reweight'] = value
            self.assertEqual(q.switch_value_from_card_reweight(), value)
        # also over the card's own backend line, which only replaces ON
        with open(os.path.join(self.mg7, 'Cards', 'reweight_card.dat'), 'w') as f:
            f.write("change backend classic\nlaunch\n")
        q.switch['reweight'] = 'madtrex'
        self.assertEqual(q.switch_value_from_card_reweight(), 'madtrex')
        q.switch['reweight'] = 'ON'
        self.assertEqual(q.switch_value_from_card_reweight(), 'classic')

    def test_backend_values_enable_the_card(self):
        from madgraph.interface.madevent_interface import AskRunEditCard
        spec = [c for c in AskRunEditCard.switch_cards if c['key'] == 'reweight'][0]
        for value in ('ON', 'classic', 'madtrex', 'density'):
            self.assertTrue(spec['on']({'reweight': value}), value)
        q = self.question(self.mg7, True, True, card="launch\n")
        self.assertEqual(q.get_cardcmd_for_reweight('madtrex'), [])


class TestPostProcessingIsSkippedWhenThereIsNothingToDo(unittest.TestCase):
    """A plain generate/output/launch announced a post-processing step with
    nothing after the colon, and built the run interface to do nothing.

    The guard that was there counted any switch that is not "off", which is a
    different question from whether any driver will run: "Not Avail." is not
    off, and a shower switch set to anything but Pythia8 selects nothing here.
    """

    def tools_for(self, switch):
        """The tool list run_selected_tools builds, without running it."""

        from madgraph.iolibs.template_files.mg7 import launch
        return launch.selected_tools(switch)

    def test_a_not_available_switch_selects_no_tool(self):
        """The case from a real run: Delphes is not installed, so the detector
        switch reads "Not Avail." -- which is not "off"."""

        switch = {'shower': 'OFF', 'detector': 'Not Avail.',
                  'analysis': 'OFF', 'madspin': 'OFF', 'reweight': 'OFF'}
        self.assertEqual(self.tools_for(switch), [])

    def test_everything_off_selects_no_tool(self):
        switch = {'shower': 'OFF', 'detector': 'OFF', 'analysis': 'OFF',
                  'madspin': 'OFF', 'reweight': 'OFF'}
        self.assertEqual(self.tools_for(switch), [])

    def test_exroot_selects_no_tool(self):
        """analysis = ExRoot is the default when ExRootAnalysis is installed,
        but mg7 has no driver for it: it must neither run anything nor force
        the LHE output."""

        switch = {'shower': 'OFF', 'detector': 'OFF', 'analysis': 'ExRoot',
                  'madspin': 'OFF', 'reweight': 'OFF'}
        self.assertEqual(self.tools_for(switch), [])

    def test_a_real_selection_still_selects(self):
        switch = {'shower': 'Pythia8', 'detector': 'Not Avail.',
                  'analysis': 'OFF', 'madspin': 'ON', 'reweight': 'OFF'}
        self.assertEqual(self.tools_for(switch), ['MadSpin', 'Pythia8 shower'])

    def test_run_selected_tools_returns_before_building_anything(self):
        """With no tool selected it must not construct MG7RunCmd -- that is
        what printed the MADEVENT banner and the configuration lines."""

        from madgraph.iolibs.template_files.mg7 import launch

        class _Process(object):
            run_path = '/nonexistent/run_01'

        switch = {'shower': 'OFF', 'detector': 'Not Avail.',
                  'analysis': 'OFF', 'madspin': 'OFF', 'reweight': 'OFF'}
        # _find_event_file would fail on the fake path, and MG7RunCmd would
        # fail harder: returning early means neither is reached
        launch.run_selected_tools(switch, _Process())


class TestPostProcessingResultsStayInMemory(unittest.TestCase):
    """The tool drivers book their runs in a madevent results database, which
    madevent writes to HTML/results.pkl and renders as crossx.html. The mg7
    output has no HTML/ directory: store_result, at the end of every
    post-processing run, logged "ERROR: I/O error (2): No such file or
    directory", and a stale crossx.html was left at the process root.
    """

    def setUp(self):
        import io

        self.path = tempfile.mkdtemp(prefix='mg7_results_')
        os.mkdir(os.path.join(self.path, 'Events'))
        self.captured = io.StringIO()
        self.handler = logging.StreamHandler(self.captured)
        self.logger = logging.getLogger('madgraph.files')
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        shutil.rmtree(self.path)

    def test_store_result_bookkeeping_writes_nothing(self):
        import madgraph.various.banner as banner_mod
        from madgraph.iolibs.template_files.mg7.run_interface import MG7RunCmd

        # load_results_db only needs me_dir, process and the model name
        # (__del__ reads the other two); building a real MG7RunCmd needs a
        # whole mg7 output
        cmd = object.__new__(MG7RunCmd)
        cmd.stop_for_runweb = False
        cmd.force_run = True
        cmd.me_dir = self.path
        cmd.process = ['p p > t t~']
        cmd.find_model_name = lambda: 'sm'
        cmd.load_results_db()

        run_card = banner_mod.RunCardMG7()
        run_card['run_tag'] = 'tag_1'
        # what set_run_name, do_decay_events and store_result do with it
        cmd.results.add_run('run_01', run_card)
        cmd.results.add_run('run_01_decayed_1', run_card)
        cmd.results.add_detail('nb_event', 200)
        cmd.results.update('MadSpin Done', 'parton', makehtml=True)
        cmd.results.update('Done', 'pythia', makehtml=False, error=True)
        cmd.results.save()

        self.assertEqual(self.captured.getvalue(), '')
        self.assertEqual(os.listdir(self.path), ['Events'])
        self.assertEqual(cmd.results.current['run_name'], 'run_01_decayed_1')
        self.assertEqual(cmd.results.current['nb_event'], 200)


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class MG7MeFrameTest(unittest.TestCase):
    """run_card me_frame -> the particle list the matrix element is boosted
    into, and the number of incoming particles the boost starts from."""

    # a collision of two gluons into a Z plus gluons, so that particle 3 is
    # massive (a frame can be asked for it) and the others are not
    MASSES = {21: 0.0, 23: 91.188}

    def resolve(self, me_frame, is_decay, particle_count=4, outgoing=None):
        """``outgoing``, if given, lists the final states of several
        subprocesses of one output (g g > each of them)"""
        from madgraph.iolibs.template_files.mg7 import launch as mg7_launch
        process = object.__new__(mg7_launch.MadgraphProcess)
        process.run_card = {'run': {'me_frame': me_frame}}
        process.is_decay = is_decay
        incoming = [23] if is_decay else [21, 21]
        if outgoing is None:
            outgoing = [[23] + [21] * (particle_count - len(incoming) - 1)]
        process.subprocess_data = [{'incoming': incoming, 'outgoing': final}
                                   for final in outgoing]
        process.get_mass = lambda pid: self.MASSES[pid]
        process.init_me_frame()
        return process.me_frame, process.incoming_count

    def test_the_default_applies_no_boost(self):
        """[] is the default: the matrix element sees the momenta in the frame
        they are generated in, which is what mg7 did before me_frame existed."""
        self.assertEqual(self.resolve([], False), ([], 2))
        self.assertEqual(self.resolve([], True), ([], 1))

    def test_an_explicit_frame_is_passed_through(self):
        self.assertEqual(self.resolve([1, 2], False), ([1, 2], 2))
        self.assertEqual(self.resolve([3], False), ([3], 2))   # the Z, massive
        self.assertEqual(self.resolve([3, 4], False), ([3, 4], 2))

    def test_a_decay_keeps_what_the_card_asks_for(self):
        """[1] is a decay's equivalent of the partonic centre of mass, but the
        card is never second-guessed: [1, 2] on a decay is the rest frame of
        the decaying particle plus the first decay product, and is left alone."""
        self.assertEqual(self.resolve([1], True), ([1], 1))
        self.assertEqual(self.resolve([1, 2], True), ([1, 2], 1))

    def test_a_massless_particle_has_no_rest_frame(self):
        """Boosting to it divides by its vanishing mass and turns every
        momentum into a NaN, with nothing to say what went wrong."""
        with self.assertRaises(ValueError) as caught:
            self.resolve([4], False)          # leg 4 is a gluon
        self.assertIn('massless', str(caught.exception))
        # two particles whose sum is massive are fine, even if each is massless
        self.assertEqual(self.resolve([2, 4], False), ([2, 4], 2))

    def test_out_of_range_particle_is_rejected_before_the_compile(self):
        """madspace would catch this too, but only once the integrands are
        built -- after every matrix-element library has been compiled."""
        with self.assertRaises(ValueError) as caught:
            self.resolve([5], False, particle_count=4)
        self.assertIn('4 external particles', str(caught.exception))
        with self.assertRaises(ValueError):
            self.resolve([0], False)

    def test_mixed_multiplicities_check_every_subprocess(self):
        """An output can mix multiplicities (MLM merging). Every subprocess is
        checked against its own particles, whichever comes first: the bigger
        one first used to pass the range check, and the massless check then
        indexed past the end of the smaller one (a bare IndexError)."""
        mixed = [[23, 21, 21], [23, 21]]      # g g > z g g and g g > z g
        for outgoing in (mixed, mixed[::-1]):
            for me_frame in ([5], [3, 5]):
                with self.assertRaises(ValueError) as caught:
                    self.resolve(me_frame, False, outgoing=outgoing)
                message = str(caught.exception)
                self.assertIn('me_frame particle 5 out of range', message)
                self.assertIn('21 21 > 23 21 has 4 external particles',
                              message)
                self.assertIn('mixes subprocesses with 4, 5', message)
            # a particle every subprocess has is fine
            self.assertEqual(self.resolve([3], False, outgoing=outgoing),
                             ([3], 2))
            self.assertEqual(self.resolve([3, 4], False, outgoing=outgoing),
                             ([3, 4], 2))

    def test_the_massless_check_covers_every_subprocess(self):
        """Particle 4 is a Z in g g > z z g but a gluon in g g > z g."""
        with self.assertRaises(ValueError) as caught:
            self.resolve([4], False, outgoing=[[23, 23, 21], [23, 21]])
        self.assertIn('massless (pdg [21])', str(caught.exception))
        self.assertEqual(
            self.resolve([4], False, outgoing=[[23, 23, 21], [23, 23]]),
            ([4], 2))


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class TestForceLHEOutput(unittest.TestCase):
    """The events are written as npy by default; the run_card is switched to
    the LHE format only when something that reads LHE will run."""

    def setUp(self):
        from madgraph.iolibs.template_files.mg7 import launch
        from madgraph.various.banner import RunCardMG7
        self.launch = launch
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmpdir.name)
        os.mkdir('Cards')
        template = os.path.join(os.path.dirname(launch.__file__),
                                'run_card.toml')
        RunCardMG7().write(self.card, template=template)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmpdir.cleanup()

    card = os.path.join('Cards', 'run_card.toml')

    def output_format(self):
        import tomllib
        with open(self.card, 'rb') as f:
            return tomllib.load(f)['run']['output_format']

    def edit(self, old, new):
        with open(self.card) as f:
            text = f.read()
        self.assertIn(old, text)
        with open(self.card, 'w') as f:
            f.write(text.replace(old, new, 1))

    def test_default_is_npy(self):
        self.assertEqual(self.output_format(), 'lhe_npy')

    def test_nothing_to_do_keeps_npy(self):
        self.launch.force_lhe_output_if_needed({})
        self.launch.force_lhe_output_if_needed(
            {'shower': 'OFF', 'detector': 'Not Avail.', 'analysis': 'ExRoot',
             'madspin': 'OFF', 'reweight': 'OFF'})
        self.assertEqual(self.output_format(), 'lhe_npy')

    def test_tool_forces_lhe(self):
        self.launch.force_lhe_output_if_needed({'shower': 'Pythia8'})
        self.assertEqual(self.output_format(), 'lhe')

    def test_time_of_flight_forces_lhe(self):
        self.edit('time_of_flight = -1.0', 'time_of_flight = 0.5')
        self.launch.force_lhe_output_if_needed({})
        self.assertEqual(self.output_format(), 'lhe')

    def test_legacy_systematics_forces_lhe_only_without_native(self):
        self.edit('\nsystematics = false', '\nsystematics = true')
        self.launch.force_lhe_output_if_needed({})
        # madspace computes the weights itself: systematics.py never runs
        self.assertEqual(self.output_format(), 'lhe_npy')
        with open(self.card) as f:
            text = f.read()
        start = text.index('[systematics]')
        end = text.index('enable = true', start)
        with open(self.card, 'w') as f:
            f.write(text[:end] + 'enable = false' + text[end + 13:])
        self.launch.force_lhe_output_if_needed({})
        self.assertEqual(self.output_format(), 'lhe')

    def test_scan_card_survives(self):
        """A scan value is not a valid RunCardMG7 entry: the card is edited
        as text, and only on its output_format line."""
        self.edit('ebeam1 = 6500.0', 'ebeam1 = "scan:[6500, 7000]"')
        self.launch.force_lhe_output_if_needed({})
        self.launch.force_lhe_output_if_needed({'madspin': 'ON'})
        self.assertEqual(self.output_format(), 'lhe')
        with open(self.card) as f:
            self.assertIn('ebeam1 = "scan:[6500, 7000]"', f.read())

    def test_missing_output_format_line(self):
        with open(self.card, 'w') as f:
            f.write('[run]\nseed = 3\n')
        self.launch.force_lhe_output_if_needed({'madspin': 'ON'})
        self.assertEqual(self.output_format(), 'lhe')

    def test_reweighting_forces_lhe_for_the_classic_module_only(self):
        from madgraph.various import reweight_backend as rb
        madtrex = rb.Decision('madtrex', 'param')
        self.launch.force_lhe_output_if_needed({'reweight': 'ON'}, madtrex)
        self.assertEqual(self.output_format(), 'lhe_npy')
        # without a decision (no card, no launch) nothing is reweighted
        self.launch.force_lhe_output_if_needed({'reweight': 'ON'}, None)
        self.assertEqual(self.output_format(), 'lhe_npy')
        self.launch.force_lhe_output_if_needed({'reweight': 'ON'},
                                               rb.Decision('classic', 'classic'))
        self.assertEqual(self.output_format(), 'lhe')


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class TestNpyToLHE(unittest.TestCase):
    """npy_to_lhe rebuilds the LHE file of an npy run from the header.lhe
    written next to the events (the compact_npy completion is covered by comparing a
    real run against its output_format = "lhe" twin, which needs a process)."""

    def setUp(self):
        import numpy as np
        from madgraph.iolibs.template_files.mg7 import npy_to_lhe
        self.np = np
        self.npy_to_lhe = npy_to_lhe
        self.ms = npy_to_lhe._madspace()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.run = os.path.join(self.tmpdir.name, 'Events', 'run_01')
        os.makedirs(self.run)

    def tearDown(self):
        self.tmpdir.cleanup()

    def write_lhe_npy(self):
        """Two e+ e- > mu+ mu- events in the lhe_npy layout, the second one
        padded with an empty particle slot."""
        fields = [('process_id', '<i4'), ('weight', '<f8'), ('scale', '<f8'),
                  ('alpha_qed', '<f8'), ('alpha_qcd', '<f8'),
                  ('rwgt_1', '<f8'), ('rwgt_2', '<f8')]
        names = ('pdg_id', 'status_code', 'mother1', 'mother2', 'color',
                 'anti_color', 'px', 'py', 'pz', 'energy', 'mass', 'lifetime',
                 'spin')
        for i in range(5):
            fields += [('part%d_%s' % (i + 1, n), '<i4' if j < 6 else '<f8')
                       for j, n in enumerate(names)]
        events = self.np.zeros(2, dtype=fields)
        rows = [(-11, -1, 0, 0, 0, 0, 0., 0., 500., 500.),
                (11, -1, 0, 0, 0, 0, 0., 0., -500., 500.),
                (-13, 1, 1, 2, 0, 0, 100., 0., 489.9, 500.),
                (13, 1, 1, 2, 0, 0, -100., 0., -489.9, 500.)]
        for n in range(2):
            events['weight'][n] = 0.5
            events['scale'][n] = 1000.
            events['rwgt_1'][n] = 0.4 + n
            events['rwgt_2'][n] = 0.6
            for i, row in enumerate(rows):
                for name, value in zip(names, row):
                    events['part%d_%s' % (i + 1, name)][n] = value
        self.np.save(os.path.join(self.run, 'events.npy'), events)

    def save_meta(self):
        """The header.lhe an npy run writes next to its events."""
        ms = self.ms
        meta = ms.LHEMeta(
            beam1_pdg_id=-11, beam2_pdg_id=11, beam1_energy=500.,
            beam2_energy=500., beam1_pdf_id=-1, beam2_pdf_id=-1,
            weight_mode=3, processes=[ms.LHEProcess(0.5, 0.01, 0.5, 1)],
            headers=[ms.LHEHeader(name="MG7Seed", content="7"),
                     ms.LHEHeader(name="initrwgt",
                                  content='<weight id="1">a</weight>')])
        writer = ms.LHEFileWriter(os.path.join(self.run, 'header.lhe'), meta)
        del writer

    def test_lhe_npy_round_trip(self):
        from madgraph.various import lhe_parser
        self.write_lhe_npy()
        self.save_meta()
        output = self.npy_to_lhe.convert('run_01', me_dir=self.tmpdir.name)
        self.assertEqual(output, os.path.join(self.run, 'events.lhe.gz'))
        lhe = lhe_parser.EventFile(output)
        self.assertIn('<MG7Seed>', lhe.banner)
        self.assertIn('<initrwgt>', lhe.banner)
        events = list(lhe)
        self.assertEqual(len(events), 2)
        self.assertEqual([p.pid for p in events[0]], [-11, 11, -13, 13])
        self.assertEqual([p.status for p in events[1]], [-1, -1, 1, 1])
        self.assertAlmostEqual(events[0][2].px, 100.)
        self.assertAlmostEqual(events[1].wgt, 0.5)
        self.assertEqual(events[1].parse_reweight(), {'1': 1.4, '2': 0.6})

    def test_default_picks_unconverted_run(self):
        self.write_lhe_npy()
        self.save_meta()
        self.npy_to_lhe.main(['--no-gzip'], me_dir=self.tmpdir.name)
        self.assertTrue(os.path.exists(os.path.join(self.run, 'events.lhe')))
        with self.assertRaises(SystemExit):
            # nothing left to convert
            self.npy_to_lhe.main([], me_dir=self.tmpdir.name)

    def test_run_without_saved_header(self):
        self.write_lhe_npy()
        self.assertFalse(self.npy_to_lhe.can_convert(self.run))
        with self.assertRaises(FileNotFoundError):
            self.npy_to_lhe.convert(self.run)


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class TestHistogramSwitches(unittest.TestCase):
    """[run] weighted_histograms / postprocessing_histograms."""

    def test_defaults(self):
        from madgraph.various.banner import RunCardMG7
        card = RunCardMG7()
        self.assertIs(card['run']['weighted_histograms'], False)
        self.assertIs(card['run']['postprocessing_histograms'], True)

    def test_mean_ignores_under_and_overflow(self):
        """Both histogram kinds carry the under/overflow bins first and last;
        they have no position and must not shift the in-range bins."""
        from madgraph.iolibs.template_files.mg7 import launch
        mean = launch.MadgraphProcess._histogram_mean
        # bins [0,10) and [10,20) with weights 1 and 3, plus 100 in overflow
        self.assertAlmostEqual(mean(0., 20., [5., 1., 3., 100.]), 12.5)
        self.assertIsNone(mean(0., 20., [5., 0., 0., 100.]))

    def test_postprocessing_off_builds_no_event_histograms(self):
        from madgraph.iolibs.template_files.mg7 import launch
        from madgraph.various.banner import RunCardMG7
        process = launch.MadgraphProcess.__new__(launch.MadgraphProcess)
        process.run_card = RunCardMG7()
        process.run_card['run']['postprocessing_histograms'] = False
        process.hist_data = [launch.HistItem(None, 0., 1., 10, 'weight', True)]
        self.assertIsNone(process.build_event_histograms())
        self.assertIsNone(process.event_histograms)


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class MG7ObservablePidsTest(unittest.TestCase):
    """Which pdg ids a cut or histogram observable is built with.

    Group selections work on clean_pids' unsigned representatives, but a
    same-flavour opposite-sign pair needs the signed pdg ids of the actual
    flavours, consistent over every flavour a subprocess groups together."""

    def setUp(self):
        from madgraph.iolibs.template_files.mg7 import launch as mg7_launch
        self.launch = mg7_launch

    @staticmethod
    def meta(outgoing, options):
        return {
            'incoming': [81, -81],
            'outgoing': outgoing,
            'flavors': [{'index': i, 'options': [option], 'mirror': False}
                        for i, option in enumerate(options)],
        }

    @staticmethod
    def kwargs(observable, groups=([11, -11, 13, -13],)):
        return {'observable': observable, 'select_pids': list(groups),
                'name': 'lepton-%s' % observable}

    def test_unsigned_observables_keep_the_representatives(self):
        meta = self.meta([-82, 82, 21, 21], [[2, -2, -11, 11, 21, 21]])
        self.assertEqual(
            self.launch.observable_pids(meta, self.kwargs('pair_mass')),
            [1, 1, 11, 11, 21, 21])

    def test_sfos_gets_the_signed_flavours(self):
        meta = self.meta([-82, 82, 21, 21], [[2, -2, -11, 11, 21, 21],
                                             [4, -4, -13, 13, 21, 21]])
        self.assertEqual(
            self.launch.observable_pids(meta, self.kwargs('sfos_pair_mass')),
            [2, -2, -11, 11, 21, 21])

    def test_sfos_refuses_flavours_that_pair_differently(self):
        # e+ e- mu+ mu- and e+ mu- mu+ e-: the same momenta, different pairs
        meta = self.meta([-82, 82, -82, 82], [[2, -2, -11, 11, -13, 13],
                                              [2, -2, -11, 13, -13, 11]])
        with self.assertRaises(ValueError) as error:
            self.launch.observable_pids(meta, self.kwargs('sfos_pair_mass'))
        self.assertIn('apply_flavor_grouping', str(error.exception))


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class MG7DecayProductsTest(unittest.TestCase):
    """Which particles phasespace.cut_decays = false leaves uncut: those an
    on-shell (decay-chain) propagator decays into, as MadEvent's check_decay."""

    def setUp(self):
        from madgraph.iolibs.template_files.mg7 import launch as mg7_launch
        self.launch = mg7_launch

    @staticmethod
    def meta(vertices, on_shell, permutation=(0, 1, 2, 3, 4)):
        # p p > z j, z > e+ e-: outgoing e+ e- j
        return {
            'incoming': [21, 81],
            'outgoing': [-82, 82, 81],
            'channels': [{
                'propagators': [81, 23],
                'vertices': vertices,
                'on_shell_propagators': on_shell,
                'diagrams': [{'diagram': 1, 'permutation': list(permutation)}],
            }],
        }

    def test_the_decay_chain_products(self):
        meta = self.meta([['i1', 'i0', 'p0'], ['o0', 'o1', 'p1'],
                          ['p0', 'o2', 'p1']], [1])
        self.assertEqual(self.launch.decay_products(meta), {0, 1})

    def test_no_decay_chain(self):
        meta = self.meta([['i1', 'i0', 'p0'], ['o0', 'o1', 'p1'],
                          ['p0', 'o2', 'p1']], [])
        self.assertEqual(self.launch.decay_products(meta), set())

    def test_a_propagator_built_from_the_beams(self):
        # p1 = i0 + i1 - o2: it decays to everything but o2
        meta = self.meta([['i0', 'i1', 'p0'], ['p0', 'o2', 'p1'],
                          ['o0', 'o1', 'p1']], [1])
        self.assertEqual(self.launch.decay_products(meta), {0, 1})

    def test_read_through_the_permutation(self):
        # event[i] = topology[permutation[i]]: the topology's o0, o1 sit at
        # event outgoing positions 1 and 2
        meta = self.meta([['i1', 'i0', 'p0'], ['o0', 'o1', 'p1'],
                          ['p0', 'o2', 'p1']], [1], permutation=(0, 1, 4, 2, 3))
        self.assertEqual(self.launch.decay_products(meta), {1, 2})


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class MG7AutoPtjMjjCutNamesTest(unittest.TestCase):
    """The cuts auto_ptj_mjj writes must be ones madspace can build.

    apply_auto_ptj_mjj names a cut after the observable it wants, and
    parse_observable turns that name straight into an ms.Observable option.
    Nothing else checks the two agree, so renaming an observable in madspace
    (m_inv became pair_mass) left the launcher asking for a cut that no longer
    existed, and every merged run with xqcut > 0 died building its cuts --
    before reaching the clustering the cut exists for.

    The two methods are called unbound, on a stand-in carrying just the run
    card: building a real MadgraphProcess needs a process directory, which
    would put this test out of reach of a unit run.
    """

    OUTGOING_PDGS = [21, 21, 25, 1, -1]

    def setUp(self):
        from madgraph.iolibs.template_files.mg7 import launch as mg7_launch
        self.launch = mg7_launch

    def make_process(self, cuts, xqcut=20.0, auto=True):
        return types.SimpleNamespace(is_decay=False, run_card={
            'beam': {'dynamical_scale_choice': 'mlm'},
            'phasespace': {'xqcut': xqcut, 'auto_ptj_mjj': auto},
            'multiparticles': {'jet': [21, 1, 2, -1, -2],
                               'lepton': [11, 13, -11, -13]},
            'cuts': dict(cuts),
        })

    def apply(self, cuts, **kwargs):
        process = self.make_process(cuts, **kwargs)
        out = {key: dict(values) for key, values in cuts.items()}
        self.launch.MadgraphProcess.apply_auto_ptj_mjj(process, out)
        return process, out

    def test_every_cut_it_writes_builds_an_observable(self):
        import madspace as ms

        process, cuts = self.apply({'jet-pt': {'min': 10.0},
                                    'jet-delta_r': {'min': 0.4}})
        for key in cuts:
            kwargs = self.launch.MadgraphProcess.parse_observable(
                process, key, 'pt')
            # raises TypeError if the observable name is not one madspace knows
            ms.Observable(self.OUTGOING_PDGS, **kwargs)

    def test_it_tightens_the_jet_pt_and_pair_mass_to_xqcut(self):
        _, cuts = self.apply({'jet-pt': {'min': 10.0}}, xqcut=20.0)
        self.assertEqual(cuts['jet-pt']['min'], 20.0)
        self.assertEqual(cuts['jet-pair_mass']['min'], 20.0)

    def test_it_drops_the_jet_cone_cuts(self):
        """A dR cut would carve a hole out of the region the shower fills."""
        _, cuts = self.apply({'jet-delta_r': {'min': 0.4},
                              'jet-lepton-delta_r': {'min': 0.4}})
        self.assertEqual(cuts['jet-delta_r']['min'], 0.0)
        self.assertEqual(cuts['jet-lepton-delta_r']['min'], 0.0)

    def test_with_the_option_off_it_only_zeroes_cuts_above_xqcut(self):
        _, cuts = self.apply({'jet-pt': {'min': 50.0}}, xqcut=20.0, auto=False)
        self.assertEqual(cuts['jet-pt']['min'], 0.0)
        self.assertNotIn('jet-pair_mass', cuts)
        _, cuts = self.apply({'jet-pt': {'min': 10.0}}, xqcut=20.0, auto=False)
        self.assertEqual(cuts['jet-pt']['min'], 10.0)

    def test_it_leaves_an_unmerged_run_alone(self):
        for kwargs in ({'xqcut': 0.0}, {}):
            process = self.make_process({'jet-pt': {'min': 10.0}}, **kwargs)
            if not kwargs:
                process.run_card['beam']['dynamical_scale_choice'] = \
                    'transverse_mass'
            cuts = {'jet-pt': {'min': 10.0}, 'jet-delta_r': {'min': 0.4}}
            self.launch.MadgraphProcess.apply_auto_ptj_mjj(process, cuts)
            self.assertEqual(cuts, {'jet-pt': {'min': 10.0},
                                    'jet-delta_r': {'min': 0.4}})
