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
################################################################################
"""MadNIS and a one-dimensional phase space.

A 2 -> 1 subprocess (the 0-jet `p p > z` of a merged `p p > z + jets` run)
samples a single random number. Once MadNIS was switched on for any other
subprocess, train_madnis built a normalizing flow for every subprocess, and a
one-dimensional flow has no coupling block: the run died in
Flow.initialize_from_vegas with "IndexError: vector".
"""
from __future__ import absolute_import
import collections
import logging
import types
import unittest
from unittest import mock

from madgraph.mg7 import bootstrap as mg7_bootstrap

# Importing launch without a compiled madspace starts the madspace build, and
# test_manager imports every unit test module while collecting: check first.
launch = None
if mg7_bootstrap.madspace_is_installed():
    try:
        import madgraph.mg7.launch as launch
    except Exception as error:
        launch = None
        import_error = error

MADNIS_AUTO = {
    key: 'auto' for key in [
        'flow_layers', 'flow_hidden_dim', 'discrete_hidden_dim',
        'cwnet_hidden_dim', 'lr', 'fixed_cwnet_fraction',
        'batch_size_per_channel',
    ]
}


def run_card(enable):
    madnis = collections.defaultdict(int, MADNIS_AUTO)
    madnis.update(enable=enable, train_batches=1000)
    return {
        'madnis': madnis,
        'generation': {'events': 10000, 'cpu_batch_size': 1, 'gpu_batch_size': 1},
        'gridpack': {'save_gridpack': False},
        'run': {'verbosity': 'log'},
    }


def subprocess_stub(subproc_id, n_out, card):
    stub = types.SimpleNamespace(
        subproc_id=subproc_id,
        meta={'outgoing': [None] * n_out},
        process=types.SimpleNamespace(run_card=card),
        madnis_supported=True,
    )
    stub.set_madnis_auto_settings = types.MethodType(
        launch.MadgraphSubprocess.set_madnis_auto_settings, stub)
    return stub


@unittest.skipIf(launch is None, 'madspace not available')
class TestMadnisOneDimension(unittest.TestCase):

    def test_auto_keeps_madnis_off(self):
        subproc = subprocess_stub(0, 1, run_card('auto'))
        subproc.set_madnis_auto_settings(100., 1)
        self.assertFalse(subproc.madnis_settings['enable'])
        self.assertFalse(subproc.madnis_supported)

    def test_explicit_enable_is_refused_with_a_warning(self):
        subproc = subprocess_stub(0, 1, run_card(True))
        with self.assertLogs('madgraph7', logging.WARNING) as logs:
            subproc.set_madnis_auto_settings(100., 1)
        self.assertFalse(subproc.madnis_settings['enable'])
        self.assertIn('random dimension', logs.output[0])

    def test_explicit_enable_kept_for_two_dimensions(self):
        subproc = subprocess_stub(0, 2, run_card(True))
        subproc.set_madnis_auto_settings(100., 2)
        self.assertTrue(subproc.madnis_settings['enable'])
        self.assertTrue(subproc.madnis_supported)

    def test_training_skips_the_one_dimensional_subprocess(self):
        """p p > z / p p > z j j: only z j j gets a flow, z keeps its VEGAS
        channels and the event generators of the survey."""
        card = run_card(True)
        z = subprocess_stub(0, 1, card)
        zjj = subprocess_stub(1, 3, card)
        z.set_madnis_auto_settings(1., 1)
        zjj.set_madnis_auto_settings(1., 7)

        survey_generator = object()
        z_channel = types.SimpleNamespace(event_generator=survey_generator)
        zjj_channel = types.SimpleNamespace(event_generator=object())
        z_ps = types.SimpleNamespace(channels=[z_channel])
        zjj_ps = types.SimpleNamespace(channels=[zjj_channel])
        trained_ps = types.SimpleNamespace(
            channels=['a', 'b'], cwnet=None)
        z.build_madnis = mock.Mock(side_effect=AssertionError('no flow for z'))
        zjj.build_madnis = mock.Mock(return_value=trained_ps)
        zjj.build_integrands = mock.Mock(return_value=[])

        process = types.SimpleNamespace(
            run_card=card, subprocesses=[z, zjj], phasespaces=[z_ps, zjj_ps],
            contexts=[mock.Mock()], devices=[None], pool_sizes=[1],
            status_file=None, run_seed=1,
            build_event_generator=mock.Mock(return_value='generator'))
        from madgraph.mg7 import madgraph_process
        with mock.patch.object(madgraph_process, 'ms') as ms:
            ms.MultiMadnisTraining.return_value.active_channels.return_value = [[1]]
            launch.MadgraphProcess.train_madnis(process)
            self.assertEqual(
                len(ms.TrainingArgs.call_args_list), 1,
                'only z j j is trained')

        zjj.build_madnis.assert_called_once_with(zjj_ps)
        self.assertEqual(process.phasespaces, [z_ps, trained_ps])
        self.assertIs(z_channel.event_generator, survey_generator)
        self.assertEqual(trained_ps.channels, ['b'])
        process.build_event_generator.assert_called_once_with([z_ps, trained_ps])
        self.assertTrue(process.madnis_trained)


if __name__ == '__main__':
    unittest.main()
