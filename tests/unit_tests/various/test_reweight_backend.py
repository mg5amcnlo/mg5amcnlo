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
"""madgraph/various/reweight_backend.py: which backend (the classic module or
MadtRex) runs a reweight card. The environments are built by hand."""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import unittest.mock
from pathlib import Path

from madgraph.various import reweight_backend as rb

MG_ROOT = Path(__file__).resolve().parents[3]

LAUNCH = "launch --rwgt_name=mt\nset MASS 6 175\n"


def env(**kwargs):
    """An MG7 lhe_npy output where both backends can run, unless overridden."""
    values = dict(output_type='mg7', nlo_events=False, output_format='lhe_npy',
                  classic_available=True, madtrex_installed=True)
    values.update(kwargs)
    return rb.Env(**values)


def select(card, environment=None, requested=None, plugin=None):
    return rb.select_backend(requested, rb.classify_card(card),
                             environment or env(), plugin=plugin)


class ClassifyCardTest(unittest.TestCase):

    def test_plain_card_is_shared(self):
        traits = rb.classify_card("change helicity False\n" + LAUNCH)
        self.assertEqual(traits.classic_only, [])
        self.assertEqual(traits.madtrex_only, [])
        self.assertIsNone(traits.backend)
        self.assertEqual(traits.madtrex_kind, 'param')
        self.assertEqual(traits.launches, 1)

    def test_backend_line(self):
        traits = rb.classify_card("# comment\nchange backend MadtRex # why not\n" + LAUNCH)
        self.assertEqual((traits.backend, traits.backend_line), ('madtrex', 2))

    def test_unknown_backend(self):
        with self.assertRaisesRegex(rb.ReweightBackendError, 'unknown reweighting backend'):
            rb.classify_card("change backend fortran\n" + LAUNCH)

    def test_two_backends(self):
        with self.assertRaisesRegex(rb.ReweightBackendError, 'two backends'):
            rb.classify_card("change backend classic\nchange backend madtrex\n" + LAUNCH)

    def test_classic_only_lines_are_numbered(self):
        card = ("change output 2.0\nchange rwgt_dir /tmp/x\nchange mode NLO\n"
                "change process p p > z --add\n" + LAUNCH + "/path/to/param_card.dat\n")
        traits = rb.classify_card(card)
        self.assertEqual([d.line for d in traits.classic_only], [1, 2, 3, 4, 7])
        self.assertEqual(traits.madtrex_only, [])
        self.assertFalse(traits.density)

    def test_density(self):
        traits = rb.classify_card("change particle_in_density_matrix 3\n" + LAUNCH)
        self.assertTrue(traits.density)
        self.assertEqual([d.line for d in traits.classic_only], [1])

    def test_madtrex_only_lines(self):
        card = ("change fptype d\nset helicity_reweighting false\nchange verbose true\n"
                + LAUNCH + "set MUF_FACTOR 2\n")
        traits = rb.classify_card(card)
        self.assertEqual([d.line for d in traits.madtrex_only], [1, 2, 3, 6])
        self.assertEqual(traits.classic_only, [])
        self.assertEqual(traits.madtrex_kind, 'pdf')

    def test_madtrex_kinds(self):
        self.assertEqual(rb.classify_card("change fptype d\n" + LAUNCH).madtrex_kind, 'param')
        self.assertEqual(rb.classify_card("launch\nset PDF_SET X\n").madtrex_kind, 'pdf')
        self.assertEqual(rb.classify_card("change scale_variation_output all\nlaunch\n")
                         .madtrex_kind, 'pdf')
        self.assertEqual(rb.classify_card("change mode pdf\nlaunch\nset PDF_SET X\n")
                         .madtrex_kind, 'generic')
        self.assertEqual(rb.classify_card("set mode parameter\n" + LAUNCH).madtrex_kind,
                         'generic')
        # LO is the classic default: both backends understand it
        traits = rb.classify_card("change mode LO\n" + LAUNCH)
        self.assertEqual((traits.classic_only, traits.madtrex_only), ([], []))

    def test_bang_comments(self):
        """A '!' comment does not hide what a line asks for."""
        traits = rb.classify_card("launch ! first\nset PDF_SET 331700 ! NNPDF\n")
        self.assertEqual([d.line for d in traits.madtrex_only], [2])
        self.assertEqual(traits.madtrex_kind, 'pdf')
        self.assertEqual(rb.classify_card("! launch\n").launches, 0)

    def test_named_parameters_are_shared(self):
        traits = rb.classify_card("launch\nset MT 175\nset param_card mass 6 175\n")
        self.assertEqual((traits.classic_only, traits.madtrex_only), ([], []))

    def test_strip_backend_line(self):
        card = "change backend auto\nchange helicity False\n  CHANGE BACKEND classic\n" + LAUNCH
        stripped = rb.strip_backend_line(card)
        self.assertEqual(len(stripped.splitlines()), len(card.splitlines()))
        self.assertIsNone(rb.classify_card(stripped).backend)
        self.assertIn("change helicity False", stripped)
        self.assertIn("launch --rwgt_name=mt", stripped)


class AutoRulesTest(unittest.TestCase):
    """The auto rules, in order."""

    def test_1_mixed_directives_are_an_error(self):
        with self.assertRaises(rb.ReweightBackendError) as ctx:
            select("change output 2.0\nchange fptype d\n" + LAUNCH)
        text = str(ctx.exception)
        self.assertIn('line 1', text)
        self.assertIn('line 2', text)

    def test_2_classic_only_directives(self):
        decision = select("change output 2.0\n" + LAUNCH)
        self.assertEqual((decision.backend, decision.card_kind), ('classic', 'classic'))
        self.assertIn('line 1', decision.describe())

    def test_2_density(self):
        decision = select("change particle_in_density_matrix 3\n" + LAUNCH)
        self.assertEqual((decision.backend, decision.card_kind), ('classic', 'density'))

    def test_2_plugin(self):
        decision = select(LAUNCH, plugin='my_plugin')
        self.assertEqual(decision.backend, 'classic')
        self.assertIn('my_plugin', decision.describe())

    def test_2_classic_only_without_classic(self):
        with self.assertRaisesRegex(rb.ReweightBackendError, 'f2py'):
            select("change output 2.0\n" + LAUNCH,
                   env(classic_available=False, classic_reason='needs f2py'))

    def test_3_madtrex_only_directives(self):
        decision = select("change fptype d\n" + LAUNCH,
                          env(classic_available=False))
        self.assertEqual((decision.backend, decision.card_kind), ('madtrex', 'param'))

    def test_3_madtrex_only_without_madtrex(self):
        with self.assertRaisesRegex(rb.ReweightBackendError, 'not installed'):
            select("change fptype d\n" + LAUNCH, env(madtrex_installed=False))

    def test_4_madevent_output(self):
        decision = select(LAUNCH, env(output_type='madevent', output_format=None))
        self.assertEqual(decision.backend, 'classic')
        self.assertIn('MG7', decision.describe())

    def test_4_nlo_events(self):
        decision = select(LAUNCH, env(output_type='aMC@NLO', nlo_events=True))
        self.assertEqual(decision.backend, 'classic')

    def test_4_madevent_without_classic(self):
        with self.assertRaises(rb.ReweightBackendError):
            select(LAUNCH, env(output_type='madevent', classic_available=False,
                               classic_reason='needs f2py'))

    def test_5_madtrex_preferred(self):
        for output_format in ('lhe', 'lhe_npy'):
            decision = select(LAUNCH, env(output_format=output_format))
            self.assertEqual(decision.backend, 'madtrex', output_format)
            self.assertEqual(decision.source, 'auto')
            self.assertEqual(
                decision.describe(),
                'Reweighting with madtrex (auto: MG7 output, madtrex installed, '
                'no classic-only directives in reweight_card.dat)')

    def test_6_classic_when_madtrex_unusable(self):
        decision = select(LAUNCH, env(madtrex_installed=False))
        self.assertEqual(decision.backend, 'classic')
        self.assertIn('not installed', decision.describe())
        decision = select(LAUNCH, env(output_format='compact_npy'))
        self.assertEqual(decision.backend, 'classic')
        self.assertIn('compact_npy', decision.describe())

    def test_7_nothing_available(self):
        with self.assertRaisesRegex(rb.ReweightBackendError, 'no backend'):
            select(LAUNCH, env(madtrex_installed=False, classic_available=False,
                               classic_reason='needs f2py'))


class ForcedBackendTest(unittest.TestCase):

    def test_madtrex_on_madevent(self):
        with self.assertRaisesRegex(rb.ReweightBackendError, 'MG7 output'):
            select(LAUNCH, env(output_type='madevent'), requested='madtrex')

    def test_madtrex_on_nlo(self):
        with self.assertRaisesRegex(rb.ReweightBackendError, 'NLO'):
            select(LAUNCH, env(nlo_events=True), requested='madtrex')

    def test_madtrex_on_compact_npy(self):
        with self.assertRaisesRegex(rb.ReweightBackendError, 'compact_npy'):
            select(LAUNCH, env(output_format='compact_npy'), requested='madtrex')

    def test_madtrex_not_installed(self):
        with self.assertRaisesRegex(rb.ReweightBackendError, 'not installed'):
            select(LAUNCH, env(madtrex_installed=False), requested='madtrex')

    def test_madtrex_with_classic_only_directive(self):
        with self.assertRaisesRegex(rb.ReweightBackendError, 'line 1'):
            select("change output 2.0\n" + LAUNCH, requested='madtrex')

    def test_madtrex_with_plugin(self):
        with self.assertRaisesRegex(rb.ReweightBackendError, 'plugin'):
            select(LAUNCH, requested='madtrex', plugin='my_plugin')

    def test_classic_with_madtrex_only_directive(self):
        with self.assertRaisesRegex(rb.ReweightBackendError, 'line 1'):
            select("change fptype d\n" + LAUNCH, requested='classic')

    def test_classic_not_available(self):
        with self.assertRaisesRegex(rb.ReweightBackendError, 'f2py'):
            select(LAUNCH, env(classic_available=False, classic_reason='needs f2py'),
                   requested='classic')

    def test_forced_classic_on_mg7(self):
        decision = select(LAUNCH, requested='classic')
        self.assertEqual((decision.backend, decision.source), ('classic', 'requested'))

    def test_every_problem_is_listed(self):
        with self.assertRaises(rb.ReweightBackendError) as ctx:
            select("change output 2.0\n" + LAUNCH,
                   env(output_type='madevent', madtrex_installed=False),
                   requested='madtrex')
        self.assertEqual(len(ctx.exception.reasons), 3)


class PrecedenceTest(unittest.TestCase):

    def test_card_backend(self):
        decision = select("change backend classic\n" + LAUNCH)
        self.assertEqual(decision.backend, 'classic')
        self.assertIn('change backend', decision.source)
        self.assertEqual(decision.warnings, [])

    def test_card_auto_is_auto(self):
        self.assertEqual(select("change backend auto\n" + LAUNCH).backend, 'madtrex')

    def test_switch_beats_card(self):
        decision = select("change backend classic\n" + LAUNCH,
                          requested=[('reweight switch', 'madtrex')])
        self.assertEqual(decision.backend, 'madtrex')
        self.assertEqual(len(decision.warnings), 1)
        self.assertIn('reweight switch', decision.warnings[0])
        self.assertIn('change backend', decision.warnings[0])

    def test_cli_beats_switch_and_card(self):
        decision = select("change backend madtrex\n" + LAUNCH,
                          requested=[('--reweight-backend', 'classic'),
                                     ('reweight switch', 'madtrex')])
        self.assertEqual((decision.backend, decision.source),
                         ('classic', '--reweight-backend'))
        self.assertEqual(len(decision.warnings), 2)

    def test_auto_and_none_requests_are_not_explicit(self):
        decision = select("change backend classic\n" + LAUNCH,
                          requested=[('--reweight-backend', 'auto'),
                                     ('reweight switch', None)])
        self.assertEqual(decision.backend, 'classic')
        self.assertEqual(decision.warnings, [])

    def test_agreeing_requests_do_not_warn(self):
        decision = select("change backend madtrex\n" + LAUNCH,
                          requested=[('reweight switch', 'madtrex')])
        self.assertEqual(decision.warnings, [])

    def test_unknown_request(self):
        with self.assertRaises(rb.ReweightBackendError):
            select(LAUNCH, requested='fortran')

    def test_conflicting_card_directives_under_auto(self):
        with self.assertRaisesRegex(rb.ReweightBackendError, 'mixes'):
            select("change particle_in_density_matrix 3\nset PDF_SET X\n" + LAUNCH)


class ProbeEnvironmentTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.dir, 'Cards'))

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_mg7(self):
        with open(os.path.join(self.dir, 'Cards', 'run_card.toml'), 'w') as f:
            f.write('[run]\noutput_format = "lhe" # options\n')
        madtrex = os.path.join(self.dir, 'mtx')
        os.makedirs(os.path.join(madtrex, 'madtrex'))
        found = rb.probe_environment(self.dir, {'mg5_path': '/mg5'},
                                     madtrex_install_dir=madtrex, has_f2py=True)
        self.assertEqual((found.output_type, found.output_format, found.nlo_events),
                         ('mg7', 'lhe', False))
        self.assertTrue(found.classic_available)
        self.assertTrue(found.madtrex_installed)

    def test_madevent_without_f2py(self):
        os.mkdir(os.path.join(self.dir, 'bin'))
        open(os.path.join(self.dir, 'bin', 'madevent'), 'w').close()
        found = rb.probe_environment(self.dir, {}, madtrex_install_dir=self.dir,
                                     has_f2py=False)
        self.assertEqual(found.output_type, 'madevent')
        self.assertFalse(found.classic_available)
        self.assertIn('f2py', found.classic_reason)
        self.assertFalse(found.madtrex_installed)

    def test_amcatnlo(self):
        open(os.path.join(self.dir, 'Cards', 'FKS_params.dat'), 'w').close()
        found = rb.probe_environment(self.dir, {}, has_f2py=True)
        self.assertEqual(found.output_type, 'aMC@NLO')
        self.assertTrue(found.nlo_events)

    def test_no_mg5_path(self):
        found = rb.probe_environment(self.dir, {'mg5_path': None}, has_f2py=True)
        self.assertFalse(found.classic_available)


class ClassicOptionsTest(unittest.TestCase):
    """The classic option names come from reweight_interface."""

    def test_every_classic_option_is_classified(self):
        from madgraph.interface.reweight_interface import (REWEIGHT_CHANGE_OPTIONS,
                                                           DENSITY_CHANGE_OPTIONS)
        for name in REWEIGHT_CHANGE_OPTIONS | DENSITY_CHANGE_OPTIONS:
            value = 'auto' if name == 'backend' else 'x'
            traits = rb.classify_card("change %s %s\n%s" % (name, value, LAUNCH))
            if name in rb.SHARED_CHANGE_OPTIONS:
                self.assertEqual(traits.classic_only, [], name)
            else:
                self.assertEqual([d.line for d in traits.classic_only], [1], name)


class ClassicSideTest(unittest.TestCase):
    """The classic module reads the shared vocabulary: it accepts 'change
    backend classic|auto', and says clearly what only MadtRex does."""

    def change(self, line):
        from madgraph.interface import reweight_interface as ri
        cmd = ri.ReweightInterface.__new__(ri.ReweightInterface)
        cmd.has_standalone_dir = False
        cmd.exitted = True  # nothing to clean up when it is collected
        return cmd.do_change(line)

    def test_backend_line(self):
        from madgraph.interface.extended_cmd import Cmd
        InvalidCmd = Cmd.InvalidCmd
        self.change('backend classic')
        self.change('backend AUTO')
        with self.assertRaisesRegex(InvalidCmd, 'MG7 output'):
            self.change('backend madtrex')
        with self.assertRaises(InvalidCmd):
            self.change('backend fortran')

    def test_madtrex_only_options(self):
        from madgraph.interface.extended_cmd import Cmd
        InvalidCmd = Cmd.InvalidCmd
        for line in ('mode pdf', 'fptype d', 'max_batch 10', 'alpha_s_power 2'):
            with self.assertRaisesRegex(InvalidCmd, 'MadtRex only', msg=line):
                self.change(line)

    def test_madtrex_only_options_match_the_selector(self):
        from madgraph.interface.reweight_interface import MADTREX_ONLY_CHANGE_OPTIONS
        madtrex_change, _, _ = rb._madtrex_card_vocabulary()
        self.assertEqual(MADTREX_ONLY_CHANGE_OPTIONS, madtrex_change)

    def test_default_f2py_is_the_running_pythons(self):
        """f2py3 on the PATH may build for another Python: the f2py installed
        with this one (e.g. a virtual environment's) comes first."""
        from madgraph.various import misc
        tmp = tempfile.mkdtemp()
        try:
            venv_bin = os.path.join(tmp, 'venv', 'bin')
            other_bin = os.path.join(tmp, 'usr', 'bin')
            for directory, name in ((venv_bin, 'f2py'), (other_bin, 'f2py3')):
                os.makedirs(directory)
                path = os.path.join(directory, name)
                open(path, 'w').close()
                os.chmod(path, 0o755)
            with unittest.mock.patch.object(sys, 'executable', os.path.join(venv_bin, 'python')), \
                    unittest.mock.patch.dict(os.environ, {'PATH': other_bin}):
                self.assertEqual(misc.default_f2py(), os.path.join(venv_bin, 'f2py'))
                self.assertTrue(misc.has_f2py())
                os.remove(os.path.join(venv_bin, 'f2py'))
                self.assertEqual(misc.default_f2py(), 'f2py3')
                os.remove(os.path.join(other_bin, 'f2py3'))
                self.assertIsNone(misc.default_f2py())
                self.assertFalse(misc.has_f2py())
        finally:
            shutil.rmtree(tmp)

    def test_do_reweight_backend_option(self):
        from madgraph import InvalidCmd
        from madgraph.interface.common_run_interface import CommonRunCmd
        cmd = unittest.mock.Mock(InvalidCmd=InvalidCmd, options={'mg5_path': '/mg5'},
                                 switch={'reweight': 'ON'})
        with self.assertRaisesRegex(InvalidCmd, 'MG7 output'):
            CommonRunCmd.do_reweight(cmd, '--backend=madtrex -from_cards')
        with self.assertRaises(InvalidCmd):
            CommonRunCmd.do_reweight(cmd, '--backend=fortran -from_cards')
        # classic: the option is dropped and the reweighting goes on (here it
        # stops at the missing card)
        cmd.me_dir = tempfile.mkdtemp()
        try:
            self.assertIsNone(CommonRunCmd.do_reweight(cmd, '--backend=classic -from_cards'))
        finally:
            shutil.rmtree(cmd.me_dir)


class DefaultCardTest(unittest.TestCase):
    """MG7 outputs get their own default reweight card; madevent keeps its."""

    MG7_CARD = MG_ROOT / 'madgraph' / 'iolibs' / 'template_files' / 'mg7' / \
        'reweight_card_default.dat'

    def test_mg7_card_runs_with_either_backend(self):
        traits = rb.classify_card(self.MG7_CARD.read_text())
        self.assertEqual((traits.classic_only, traits.madtrex_only, traits.backend),
                         ([], [], None))
        self.assertEqual(traits.launches, 1)
        from madgraph.iolibs.template_files.mg7 import param_reweight_card
        param_reweight_card.parse_reweight_card(self.MG7_CARD.read_text())

    def test_madmatrix_output_copies_the_mg7_card(self):
        text = (MG_ROOT / 'madmatrix' / 'output.py').read_text()
        self.assertIn("relative_path_list(mg7_templates, ['reweight_card_default.dat'])", text)
        classic = (MG_ROOT / 'Template' / 'Common' / 'Cards' / 'reweight_card_default.dat')
        self.assertIn('change mode NLO', classic.read_text())


class ImportHygieneTest(unittest.TestCase):
    """The selector never imports madtrex or madspace."""

    def test_selector_imports_neither(self):
        code = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(MG_ROOT)!r})
            from madgraph.various import reweight_backend as rb
            traits = rb.classify_card("change backend auto\\nlaunch\\nset MT 175\\n")
            env = rb.probe_environment({str(MG_ROOT)!r}, has_f2py=True)
            rb.select_backend(None, traits, rb.Env(output_type='madevent'))
            bad = [m for m in sys.modules if m.split('.')[0] in ('madtrex', 'madspace')
                   or m.endswith('.mg7.madtrex') or m.endswith('.mg7.madtrex_driver')]
            assert not bad, bad
        """)
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp,
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
