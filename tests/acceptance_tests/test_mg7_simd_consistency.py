################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
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
"""MG7 consistency under a change of the madspace SIMD mode.

Runs the same seeded mg7 generation once per ``madspace_cpu_mode`` supported on
the machine (``madspace.supported_simd_modes()``) and compares the cross section
with the one of the scalar run:

  * ``test_vegas_simd_consistency_mg7`` -- survey (VEGAS) + generate. With the
    same seed, the modes see the same random numbers and only differ by rounding,
    so the cross sections have to agree to a relative tolerance far below the
    statistical uncertainty (a SIMD-only mapping bug that still gave a
    statistically compatible result was off by 0.3%).
  * ``test_madnis_simd_consistency_mg7`` -- survey + madnis training + generate.
    The rounding differences grow during the training, so here the results only
    have to agree within their statistical uncertainties.
  * ``test_madnis_gridpack_simd_consistency_mg7`` -- a gridpack trained with
    madnis in scalar mode, then event generation from that gridpack in every
    mode. Without the training drift, the same tolerance as for VEGAS applies.

Process: ``g g > t t~ g``, whose multi-channel phase space uses momentum
permutations. Self-skips if the mg7 runtime stack / PDF is unavailable, or if
the machine supports no SIMD mode besides scalar.

Run locally with e.g.::

    ./tests/test_manager.py test_.*simd_consistency_mg7 -pA -t0 -l INFO

``MG7_SIMD_EVENTS`` sets the events per run (default 5000).
"""

from __future__ import absolute_import
from __future__ import division

import glob
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

import madgraph.interface.master_interface as MGCmd
from madgraph.various.banner import RunCardMG7
from tests.acceptance_tests.test_mg7_reproducibility import (
    _mg7_datadir_or_skip, _run)

pjoin = os.path.join

_PROCESS = 'g g > t t~ g'
_EVENTS = int(os.environ.get('MG7_SIMD_EVENTS', 5000))
_SEED = 424242
# relative tolerance for runs that only differ by rounding (VEGAS, gridpacks)
_ROUNDING_REL_TOL = 1e-4
# tolerance for the madnis runs, in combined standard deviations
_MADNIS_SIGMAS = 4.


class MG7SimdConsistencyTest(unittest.TestCase):
    """Cross sections of seeded mg7 runs agree between the madspace SIMD modes."""

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='mg7_simd_')

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def _modes_or_skip(self):
        import madspace
        modes = madspace.supported_simd_modes()
        if len(modes) < 2:
            self.skipTest('no SIMD mode besides scalar supported on this machine')
        return modes

    def _output_process(self, run_name, **settings):
        run_dir = pjoin(self.path, run_name)
        mg = MGCmd.MasterCmd()
        mg.no_notification()
        mg.exec_cmd('set automatic_html_opening False --no_save')
        mg.exec_cmd('generate %s' % _PROCESS)
        mg.exec_cmd('output mg7 %s' % run_dir)
        settings = dict({
            'run.seed': _SEED,
            'run.output_format': 'compact_npy',
            'run.postprocessing_histograms': False,
            'run.make_plots': False,
            'beam.pdf': 'NNPDF23_lo_as_0130_qed',
            'generation.events': _EVENTS,
            'systematics.enable': False,
            'postprocessing.systematics': False,
        }, **settings)
        toml = pjoin(run_dir, 'Cards', 'run_card.toml')
        rc = RunCardMG7(toml)
        for key, value in settings.items():
            rc.set(key, value, user=True)
        rc.write(toml)
        return run_dir

    def _set_mode(self, run_dir, mode):
        toml = pjoin(run_dir, 'Cards', 'run_card.toml')
        rc = RunCardMG7(toml)
        rc.set('run.madspace_cpu_mode', mode, user=True)
        rc.write(toml)

    def _cross_section(self, cmd, run_dir, datadir, mode, what):
        """Run *cmd* in *run_dir* from scratch and return the (cross section,
        error) from the info.json of that run."""
        shutil.rmtree(pjoin(run_dir, 'Events'), ignore_errors=True)
        env = dict(os.environ)
        env['LHAPDF_DATA_PATH'] = datadir
        _run(cmd, run_dir, pjoin(run_dir, 'gen_%s.log' % mode), env,
             '%s (madspace_cpu_mode=%s)' % (what, mode))
        infos = sorted(glob.glob(pjoin(run_dir, 'Events', '*', 'info.json')))
        self.assertTrue(infos, 'no info.json produced for mode %s' % mode)
        with open(infos[-1]) as f:
            process = json.load(f)['process']
        print('madspace_cpu_mode=%s: %.6g +- %.2g'
              % (mode, process['mean'], process['error']))
        return process['mean'], process['error']

    def _results(self, run_dir, modes):
        """Cross sections of full runs (bin/generate_events -f) per mode."""
        datadir = _mg7_datadir_or_skip(self)
        results = {}
        for mode in modes:
            self._set_mode(run_dir, mode)
            results[mode] = self._cross_section(
                [sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
                run_dir, datadir, mode, 'mg7 generate_events')
        return results

    def _assert_rounding_level(self, results):
        ref_mean, _ = results['scalar']
        for mode, (mean, _) in results.items():
            self.assertLessEqual(
                abs(mean - ref_mean), _ROUNDING_REL_TOL * abs(ref_mean),
                'madspace_cpu_mode=%s: cross section %g differs from the scalar '
                'one %g' % (mode, mean, ref_mean))

    def test_vegas_simd_consistency_mg7(self):
        """VEGAS runs: every SIMD mode reproduces the scalar cross section up
        to rounding."""
        modes = self._modes_or_skip()
        _mg7_datadir_or_skip(self)
        run_dir = self._output_process('vegas', **{'madnis.enable': False})
        self._assert_rounding_level(self._results(run_dir, modes))

    def test_madnis_simd_consistency_mg7(self):
        """madnis runs: every SIMD mode agrees with the scalar cross section
        within the statistical uncertainties."""
        modes = self._modes_or_skip()
        _mg7_datadir_or_skip(self)
        run_dir = self._output_process(
            'madnis', **{'madnis.enable': True, 'madnis.train_batches': 200})
        results = self._results(run_dir, modes)
        ref_mean, ref_error = results['scalar']
        for mode, (mean, error) in results.items():
            sigma = math.sqrt(ref_error ** 2 + error ** 2)
            self.assertLessEqual(
                abs(mean - ref_mean), _MADNIS_SIGMAS * sigma,
                'madspace_cpu_mode=%s: cross section %g +- %g is not compatible '
                'with the scalar one %g +- %g' % (mode, mean, error, ref_mean,
                                                  ref_error))

    def test_madnis_gridpack_simd_consistency_mg7(self):
        """A gridpack trained with madnis in scalar mode: event generation from
        it reproduces the scalar cross section up to rounding in every mode."""
        modes = self._modes_or_skip()
        datadir = _mg7_datadir_or_skip(self)
        run_dir = self._output_process(
            'madnis_gridpack',
            **{'madnis.enable': True, 'madnis.train_batches': 200,
               'gridpack.save_gridpack': True})
        self._set_mode(run_dir, 'scalar')
        env = dict(os.environ)
        env['LHAPDF_DATA_PATH'] = datadir
        _run([sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
             run_dir, pjoin(run_dir, 'train.log'), env,
             'mg7 madnis training run')
        gridpacks = glob.glob(pjoin(run_dir, 'Events', '*', 'gridpack'))
        self.assertTrue(gridpacks, 'gridpack was not produced under %s' % run_dir)
        gridpack_dir = gridpacks[0]

        results = {}
        for mode in modes:
            results[mode] = self._cross_section(
                [sys.executable, pjoin(gridpack_dir, 'bin', 'generate_events'),
                 '--seed', str(_SEED), '--events', str(_EVENTS),
                 '--output_format', 'compact_npy', '--madspace_cpu_mode', mode],
                gridpack_dir, datadir, mode, 'gridpack generate_events')
        self._assert_rounding_level(results)
