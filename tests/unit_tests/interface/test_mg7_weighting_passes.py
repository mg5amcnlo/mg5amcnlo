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
"""The [surrogate] and [amplicol] second passes of an mg7 run: the run card
settings of each pass and the checks done at launch."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from madgraph.mg7 import bootstrap as mg7_bootstrap


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class FirstPassSettingsTest(unittest.TestCase):
    """The first_pass_ settings apply to the subprocesses unweighted again in a
    second pass only, the [generation] ones to all others."""

    def setUp(self):
        import madgraph.various.banner as banner_mod
        from madgraph.mg7 import madgraph_process
        from madgraph.mg7.subprocess_amplicol import AmplicolSubprocess
        from madgraph.mg7.subprocess_madmatrix import MadgraphSubprocess
        self.ms = madgraph_process.ms
        self.process = object.__new__(madgraph_process.MadgraphProcess)
        run_card = banner_mod.RunCardMG7()
        run_card['generation']['max_overweight_truncation'] = 0.001
        run_card['generation']['freeze_max_weight_after'] = 500
        run_card['amplicol']['mode'] = 'unweight'
        run_card['amplicol']['first_pass_max_overweight_truncation'] = 0.02
        run_card['amplicol']['first_pass_freeze_max_weight_after'] = 70
        self.process.run_card = run_card
        self.process.surrogate = None
        self.process.subprocess_data = []
        step = self.ms.WeightingStep([], mode='unweight')

        amplicol = object.__new__(AmplicolSubprocess)
        amplicol.process = self.process
        amplicol.ratio_matrix_element = object()
        amplicol.weighting_steps = lambda: [step]
        madmatrix = object.__new__(MadgraphSubprocess)
        madmatrix.process = self.process
        madmatrix.surrogates = []
        self.process.subprocesses = [amplicol, madmatrix]

    def test_mixed_output(self):
        groups = self.process.weighting_groups([[0, 1], [2]])
        self.assertEqual(len(groups), 1)
        self.assertEqual(list(groups[0].channel_indices), [0, 1])
        self.assertAlmostEqual(groups[0].first_pass_max_overweight_truncation, 0.02)
        self.assertEqual(groups[0].first_pass_freeze_max_weight_after, 70)

        self.process.init_generator_config()
        cfg = self.process.event_generator_config
        self.assertAlmostEqual(cfg.max_overweight_truncation, 0.001)
        self.assertEqual(cfg.freeze_max_weight_after, 500)

    def test_reweight_mode_has_no_first_pass(self):
        self.process.run_card['amplicol']['mode'] = 'reweight'
        groups = self.process.weighting_groups([[0, 1], [2]])
        self.assertIsNone(groups[0].first_pass_max_overweight_truncation)
        self.assertIsNone(groups[0].first_pass_freeze_max_weight_after)


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class SurrogateWithAmplicolTest(unittest.TestCase):

    def test_is_refused(self):
        import madgraph.various.banner as banner_mod
        from madgraph.mg7 import madgraph_process
        process = object.__new__(madgraph_process.MadgraphProcess)
        process.run_card = banner_mod.RunCardMG7()
        process.run_card['surrogate']['function'] = 'surrogate.py'
        process.subprocess_data = [{'me_path': 'a'}, {'backend': 'amplicol'}]
        with self.assertRaisesRegex(RuntimeError, r'\[surrogate\] is not supported'):
            process.init_surrogate()
        # without [amplicol] subprocesses, the next check is reached
        process.subprocess_data = process.subprocess_data[:1]
        with self.assertRaisesRegex(RuntimeError, 'file not found'):
            process.init_surrogate()


_TEMPLATES = (Path(__file__).resolve().parents[3] / 'madgraph' / 'iolibs'
              / 'template_files' / 'mg7')


def _template_module(name):
    spec = importlib.util.spec_from_file_location(
        f'mg7_template_{name}', _TEMPLATES / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class GridpackFirstPassTest(unittest.TestCase):
    """A gridpack takes the first_pass_ settings from its card, for the groups
    that unweight in a later step, also if its groups.json predates them."""

    def load(self, groups):
        setup = _template_module('gridpack_setup')
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, 'data'))
            with open(os.path.join(tmp, 'data', 'groups.json'), 'w') as f:
                json.dump(groups, f)
            with mock.patch.object(setup, 'gridpack_path',
                                   lambda *parts: os.path.join(tmp, *parts)):
                return setup.load_weighting_groups(0.001, 500, 0.02, 70)

    def test_old_and_new_groups(self):
        ms = _template_module('gridpack_setup').ms
        step = ms.WeightingStep([], mode='unweight')
        stored = json.loads(ms.weighting_groups_to_json([
            ms.WeightingGroup([0], [step], 0.5, 3), ms.WeightingGroup([1], [step]),
        ]))
        reweight = json.loads(ms.weighting_groups_to_json([
            ms.WeightingGroup([2], [ms.WeightingStep([], mode='reweight')], 0.5, 3)
        ]))
        groups = self.load(stored + reweight)
        for group in groups[:2]:
            self.assertAlmostEqual(group.first_pass_max_overweight_truncation, 0.02)
            self.assertEqual(group.first_pass_freeze_max_weight_after, 70)
            self.assertAlmostEqual(group.steps[0].max_overweight_truncation, 0.001)
        self.assertIsNone(groups[2].first_pass_max_overweight_truncation)


class FakeApi:
    def __init__(self, device):
        self.device = device

    def device_type(self):
        return self.device


class FakeContext:
    def __init__(self, device='cpu'):
        self.device = device
        self.loaded = []

    def load_matrix_element(self, path, param_card, parameters):
        self.loaded.append((path, param_card, parameters))
        return FakeApi(self.device)


class SurrogateLibraryTest(unittest.TestCase):
    """A .so [surrogate]: path substitution, missing files, device and the
    run-time parameters of the madmatrix libraries."""

    def setUp(self):
        self.loader = _template_module('custom_function_loader')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, 'sur_{subprocess}_{device}.so')
        open(self.path.format(subprocess='P1_x', device='cpp512y'), 'w').close()

    def load(self, contexts, devices=('cpu',), exports=True, subprocess='P1_x'):
        with mock.patch.object(self.loader, '_exports', lambda path, name: exports):
            return self.loader.load_surrogate(
                None, contexts, ['cpp512y'] * len(contexts), list(devices),
                self.path, subprocess, [[1, 2]], 'card.dat', {'bwcutoff': 15},
            )

    def test_parameters_are_passed(self):
        context = FakeContext()
        self.load([context])
        library = self.path.format(subprocess='P1_x', device='cpp512y')
        self.assertEqual(context.loaded, [(library, 'card.dat', {'bwcutoff': 15.0})])

    def test_no_parameters_without_set_parameter(self):
        context = FakeContext()
        self.load([context], exports=False)
        self.assertEqual(context.loaded[0][2], {})

    def test_missing_library(self):
        with self.assertRaisesRegex(RuntimeError, 'sur_P2_y_cpp512y.so not found'):
            self.load([FakeContext()], subprocess='P2_y')

    def test_wrong_device(self):
        with self.assertRaisesRegex(RuntimeError, 'built for cpu, but the run uses cuda'):
            self.load([FakeContext('cpu')], devices=('cuda',))


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class CompileAmplicolTest(unittest.TestCase):
    """SubProcesses/check-pyamplicol runs before make, and its message is the
    error of the launch."""

    def setUp(self):
        from madgraph.mg7 import madgraph_process
        self.module = madgraph_process
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.subproc = os.path.join(self.tmp.name, 'SubProcesses')
        os.makedirs(os.path.join(self.subproc, 'P1_gg_ttx'))
        self.process = object.__new__(madgraph_process.MadgraphProcess)
        self.process.run_path = self.tmp.name
        self.process.subprocess_data = [{
            'backend': 'amplicol',
            'path': os.path.join(self.subproc, 'P1_gg_ttx'),
            'me_path': 'missing.so', 'ratio_path': 'missing.so',
        }]

    def write_check(self, body):
        path = os.path.join(self.subproc, 'check-pyamplicol')
        with open(path, 'w') as f:
            f.write('#!/bin/sh\n' + body)
        os.chmod(path, 0o755)

    def compile(self):
        with mock.patch.object(self.module.MadgraphProcess, 'make_subprocesses') as make:
            self.process.compile_amplicol()
        return make

    def test_stale_pyamplicol_stops_before_make(self):
        self.write_check('echo "reinstall pyamplicol 1.2" >&2\nexit 1\n')
        with mock.patch.object(self.module.MadgraphProcess, 'make_subprocesses') as make:
            with self.assertRaisesRegex(RuntimeError, 'reinstall pyamplicol 1.2'):
                self.process.compile_amplicol()
        make.assert_not_called()

    def test_matching_pyamplicol_compiles(self):
        self.write_check('exit 0\n')
        self.compile().assert_called_once()

    def test_older_output_without_check(self):
        self.compile().assert_called_once()


if __name__ == '__main__':
    unittest.main()
