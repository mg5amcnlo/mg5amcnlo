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
"""MG7 (madspace) reproducibility tests.

Guards the seeded event-generation path: given the same run_card ``seed``,
independent mg7 runs must produce byte-identical LHE output (hashed with
sha256); a different seed must change the result. Covers:

  * ``test_vegas_reproducibility_mg7`` -- a plain survey (VEGAS-optimized)
    + generate run, through ``bin/generate_events``.
  * ``test_madnis_reproducible_mode_mg7`` -- a full survey + madnis training +
    generate run, through ``bin/generate_events`` (no gridpack export):
    training itself is required to be thread-count-independent and
    byte-identical for the same seed, both with online-only training and with
    buffered (off-policy replay) training enabled. GPU multi-channel batches
    are not covered by this test.
  * ``test_gridpack_reproducibility_mg7`` -- a gridpack trained with madnis,
    then standalone event generation from that gridpack (its own
    ``bin/generate_events --seed``), which is the normal way a gridpack is
    used and does not re-run survey/training.
  * ``test_gridpack_reproducibility_vegas_mg7`` -- same as above, but for a
    plain VEGAS-optimized gridpack (no madnis training).

Process: ``p p > t t~`` (hadronic, needs the NNPDF23_lo_as_0130_qed PDF grid;
self-skips if the mg7 runtime stack / PDF is unavailable, same as
test_check_xsec_processes_mg7.py).

Run locally with e.g.::

    ./tests/test_manager.py test_.*reproducib.*_mg7 -pA -t0 -l INFO

One knob is read from the environment so the CI can dial it without touching
the code:

  * ``MG7_REPRO_EVENTS`` -- events per run (default 10000).
"""

from __future__ import absolute_import
from __future__ import division

import glob
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import madgraph.interface.master_interface as MGCmd
from madgraph.various.banner import RunCardMG7

pjoin = os.path.join

_PROCESS = 'p p > t t~'
_EVENTS = int(os.environ.get('MG7_REPRO_EVENTS', 10000))
_SEED_A = 424242
_SEED_B = 909090


def _mg7_datadir_or_skip(test):
    """Return an LHAPDF data dir that contains the NNPDF23_lo_as_0130_qed set,
    or ``skipTest`` (on *test*) when the mg7 runtime stack (madspace + LHAPDF +
    the run_card.toml default PDF) is unavailable."""
    try:
        import madspace
        has_mg7 = hasattr(madspace, 'ChannelEventGenerator')
    except ImportError:
        has_mg7 = False
    if not has_mg7:
        test.skipTest('mg7 runtime stack (madspace) unavailable')

    candidates = []
    if os.environ.get('LHAPDF_DATA_PATH'):
        candidates.extend(os.environ['LHAPDF_DATA_PATH'].split(os.pathsep))
    try:
        out = subprocess.check_output(['lhapdf-config', '--datadir'],
                                      stderr=subprocess.DEVNULL).decode().strip()
        if out:
            candidates.append(out)
    except Exception:
        pass
    for d in candidates:
        if d and os.path.isdir(d) and glob.glob(pjoin(d, 'NNPDF23_lo_as_0130_qed*')):
            return d
    test.skipTest('NNPDF23_lo_as_0130_qed LHAPDF data not found '
                  '(set $LHAPDF_DATA_PATH)')


def _lhe_hash(path):
    """sha256 of an LHE file's physics content: everything from ``<init>``
    onward, i.e. excluding the ``<header>`` block. The header embeds the
    run_card.toml/param_card/proc_card verbatim, so it legitimately differs
    between runs that vary settings unrelated to the physics content (e.g.
    the seed itself, or the cpu thread pool size) -- only the generated
    events (and the <init> cross-section info, itself a deterministic
    function of the seed) should drive this hash.

    The file is gzipped by default, and the gzip header embeds an mtime, so
    always hash the decompressed content."""
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        content = f.read()
    marker = b'</header>\n'
    index = content.find(marker)
    if index == -1:
        raise AssertionError('no </header> found in %s' % path)
    return hashlib.sha256(content[index + len(marker):]).hexdigest()


def _find_lhe(run_path):
    """Locate the LHE file produced under Events/<run>/, gzipped or not, or
    fail."""
    matches = sorted(glob.glob(pjoin(run_path, 'Events', '*', 'events.lhe'))
                     + glob.glob(pjoin(run_path, 'Events', '*',
                                       'events.lhe.gz')))
    if not matches:
        raise AssertionError('no events.lhe[.gz] produced under %s' % run_path)
    return matches[-1]


def _run(cmd, cwd, log_path, env, what):
    """Run *cmd*, raising with the log tail on a non-zero exit."""
    with open(log_path, 'w') as logfh:
        ret = subprocess.call(cmd, cwd=cwd, env=env, stdout=logfh,
                              stderr=subprocess.STDOUT)
    if ret != 0:
        with open(log_path) as f:
            tail = ''.join(f.readlines()[-60:])
        raise AssertionError('%s failed (exit %d, see %s)\n\n%s'
                             % (what, ret, log_path, tail))


class MG7ReproducibilityTest(unittest.TestCase):
    """LHE-hash reproducibility of mg7 (madspace) event generation for a
    fixed run_card seed, and non-reproducibility across distinct seeds."""

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='mg7_repro_')

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def _output_process(self, run_name):
        run_dir = pjoin(self.path, run_name)
        mg = MGCmd.MasterCmd()
        mg.no_notification()
        mg.exec_cmd('set automatic_html_opening False --no_save')
        mg.exec_cmd('generate %s' % _PROCESS)
        mg.exec_cmd('output mg7 %s' % run_dir)
        # the hashes are taken on the LHE file; npy is the default output
        self._set_run_card(pjoin(run_dir, 'Cards', 'run_card.toml'),
                           **{'run.output_format': 'lhe'})
        return run_dir

    def _set_run_card(self, toml_path, **settings):
        """settings keys are 'section.key' RunCardMG7 names."""
        rc = RunCardMG7(toml_path)
        for key, value in settings.items():
            rc.set(key, value, user=True)
        rc.write(toml_path)

    def _generate_and_hash(self, run_dir, datadir, seed, tag, **run_card_settings):
        """Set run_card seed (plus any extra 'section.key' settings), run
        bin/generate_events -f from scratch, and return the sha256 of the
        resulting LHE file."""
        shutil.rmtree(pjoin(run_dir, 'Events'), ignore_errors=True)
        toml = pjoin(run_dir, 'Cards', 'run_card.toml')
        self._set_run_card(toml, **{'run.seed': seed}, **run_card_settings)
        env = dict(os.environ)
        env['LHAPDF_DATA_PATH'] = datadir
        _run([sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
             run_dir, pjoin(run_dir, 'gen_%s.log' % tag), env,
             'mg7 generate_events')
        return _lhe_hash(_find_lhe(run_dir))

    def _generate_and_hash_gridpack(self, gridpack_dir, datadir, seed, tag):
        """Run the gridpack's own bin/generate_events --seed from scratch, and
        return the sha256 of the resulting LHE file."""
        shutil.rmtree(pjoin(gridpack_dir, 'Events'), ignore_errors=True)
        env = dict(os.environ)
        env['LHAPDF_DATA_PATH'] = datadir
        _run([sys.executable, pjoin(gridpack_dir, 'bin', 'generate_events'),
              '--seed', str(seed), '--events', str(_EVENTS),
              '--output_format', 'lhe'],
             gridpack_dir, pjoin(gridpack_dir, 'gen_%s.log' % tag), env,
             'gridpack generate_events')
        return _lhe_hash(_find_lhe(gridpack_dir))

    def test_vegas_reproducibility_mg7(self):
        """Two VEGAS-optimized runs with the same seed produce byte-identical
        LHE files, including when the cpu thread pool is shrunk to 1; a
        different seed changes the result."""
        datadir = _mg7_datadir_or_skip(self)
        run_dir = self._output_process('vegas')
        toml = pjoin(run_dir, 'Cards', 'run_card.toml')
        self._set_run_card(
            toml,
            **{
                'generation.events': _EVENTS,
                # Keep the test scoped to madspace's own seeding: LHE-level
                # post-processing (systematics) is orthogonal and would only
                # add runtime here.
                'postprocessing.systematics': False,
            }
        )

        # Every call sets 'run.cpu_thread_pool_size' explicitly so it never
        # leaks from a previous call via the run_card left on disk.
        hash_a1 = self._generate_and_hash(
            run_dir, datadir, _SEED_A, 'a1', **{'run.cpu_thread_pool_size': -1})
        hash_a2 = self._generate_and_hash(
            run_dir, datadir, _SEED_A, 'a2', **{'run.cpu_thread_pool_size': -1})
        self.assertEqual(hash_a1, hash_a2,
                         'same seed produced different LHE content')

        hash_a_single = self._generate_and_hash(
            run_dir, datadir, _SEED_A, 'a_single',
            **{'run.cpu_thread_pool_size': 1})
        self.assertEqual(hash_a1, hash_a_single,
                         'same seed produced different LHE content with a '
                         'single-threaded cpu thread pool')

        hash_b = self._generate_and_hash(
            run_dir, datadir, _SEED_B, 'b', **{'run.cpu_thread_pool_size': -1})
        self.assertNotEqual(hash_a1, hash_b,
                            'different seeds produced identical LHE content')

    def test_madnis_reproducible_mode_mg7(self):
        """madnis training itself (CPU codepath) -- not just event generation
        -- is byte-identical for the same seed independent of the cpu thread
        pool size, both with online-only training and with buffered
        (off-policy replay) training enabled. Covers the deterministic
        scheduling in MadnisTraining::maybe_start_generator_jobs /
        start_generator_jobs (deterministic round dispatch + deferred buffer
        flush) and the seeding of BufferUnweighter/BatchSampler (see
        salt::madnis_train_unweight / salt::buffered_batch_sample in
        random.hpp) -- both required for this test to pass reliably."""
        datadir = _mg7_datadir_or_skip(self)
        run_dir = self._output_process('madnis_reproducible')
        toml = pjoin(run_dir, 'Cards', 'run_card.toml')
        base_settings = {
            'generation.events': _EVENTS,
            'postprocessing.systematics': False,
            'madnis.enable': True,
            'madnis.train_batches': 60,
        }

        # Online-only training (buffering disabled).
        self._set_run_card(toml, **base_settings, **{'madnis.buffer_capacity': 0})
        hash_online_multi = self._generate_and_hash(
            run_dir, datadir, _SEED_A, 'online_multi',
            **{'run.cpu_thread_pool_size': -1})
        hash_online_single = self._generate_and_hash(
            run_dir, datadir, _SEED_A, 'online_single',
            **{'run.cpu_thread_pool_size': 1})
        self.assertEqual(
            hash_online_multi, hash_online_single,
            'madnis training (buffering disabled) produced '
            'different LHE content with a single-threaded cpu thread pool')

        # Buffered (off-policy replay) training enabled.
        self._set_run_card(
            toml,
            **base_settings,
            **{
                'madnis.buffer_capacity': 3000,
                'madnis.buffered_steps_fraction': 0.75,
                # kept well below the capacity, as the buffered fraction ramps
                # up from zero at minimum_buffer_size
                'madnis.minimum_buffer_size': 200,
                # the default (1000) exceeds train_batches, leaving the buffer empty
                'madnis.buffer_skip_batches': 5,
            }
        )
        hash_buffered_multi = self._generate_and_hash(
            run_dir, datadir, _SEED_A, 'buffered_multi',
            **{'run.cpu_thread_pool_size': -1})
        hash_buffered_single = self._generate_and_hash(
            run_dir, datadir, _SEED_A, 'buffered_single',
            **{'run.cpu_thread_pool_size': 1})
        self.assertEqual(
            hash_buffered_multi, hash_buffered_single,
            'madnis training (buffering enabled) produced '
            'different LHE content with a single-threaded cpu thread pool')

        hash_buffered_b = self._generate_and_hash(
            run_dir, datadir, _SEED_B, 'buffered_b',
            **{'run.cpu_thread_pool_size': -1})
        self.assertNotEqual(
            hash_buffered_multi, hash_buffered_b,
            'different seeds produced identical LHE content')

    def test_gridpack_reproducibility_mg7(self):
        """A gridpack trained with madnis: two standalone event-generation
        runs from that gridpack with the same seed produce byte-identical LHE
        files; a different seed changes the result. Training itself is not
        required to be deterministic -- only the trained gridpack's event
        generation is under test here."""
        datadir = _mg7_datadir_or_skip(self)
        run_dir = self._output_process('gridpack')
        toml = pjoin(run_dir, 'Cards', 'run_card.toml')
        self._set_run_card(
            toml,
            **{
                'generation.events': _EVENTS,
                'postprocessing.systematics': False,
                'gridpack.save_gridpack': True,
                'madnis.enable': True,
                'madnis.train_batches': 100,
            }
        )

        env = dict(os.environ)
        env['LHAPDF_DATA_PATH'] = datadir
        _run([sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
             run_dir, pjoin(run_dir, 'train.log'), env,
             'mg7 madnis training run')

        gridpack_dir = pjoin(run_dir, 'Events', 'run_01', 'gridpack')
        self.assertTrue(os.path.isdir(gridpack_dir),
                        'gridpack was not produced under %s' % run_dir)

        hash_a1 = self._generate_and_hash_gridpack(
            gridpack_dir, datadir, _SEED_A, 'a1')
        hash_a2 = self._generate_and_hash_gridpack(
            gridpack_dir, datadir, _SEED_A, 'a2')
        self.assertEqual(hash_a1, hash_a2,
                         'same seed produced different LHE content')

        hash_b = self._generate_and_hash_gridpack(
            gridpack_dir, datadir, _SEED_B, 'b')
        self.assertNotEqual(hash_a1, hash_b,
                            'different seeds produced identical LHE content')

    def test_gridpack_reproducibility_vegas_mg7(self):
        """A plain VEGAS-optimized gridpack (no madnis training): two
        standalone event-generation runs from that gridpack with the same
        seed produce byte-identical LHE files; a different seed changes the
        result. Companion to test_gridpack_reproducibility_mg7, which covers
        the madnis-trained case; the survey/optimization itself is not
        required to be deterministic here either -- only the gridpack's own
        event generation is under test."""
        datadir = _mg7_datadir_or_skip(self)
        run_dir = self._output_process('gridpack_vegas')
        toml = pjoin(run_dir, 'Cards', 'run_card.toml')
        self._set_run_card(
            toml,
            **{
                'generation.events': _EVENTS,
                'postprocessing.systematics': False,
                'gridpack.save_gridpack': True,
            }
        )

        env = dict(os.environ)
        env['LHAPDF_DATA_PATH'] = datadir
        _run([sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
             run_dir, pjoin(run_dir, 'survey.log'), env,
             'mg7 vegas survey run')

        gridpack_dir = pjoin(run_dir, 'Events', 'run_01', 'gridpack')
        self.assertTrue(os.path.isdir(gridpack_dir),
                        'gridpack was not produced under %s' % run_dir)

        hash_a1 = self._generate_and_hash_gridpack(
            gridpack_dir, datadir, _SEED_A, 'a1')
        hash_a2 = self._generate_and_hash_gridpack(
            gridpack_dir, datadir, _SEED_A, 'a2')
        self.assertEqual(hash_a1, hash_a2,
                         'same seed produced different LHE content')

        hash_b = self._generate_and_hash_gridpack(
            gridpack_dir, datadir, _SEED_B, 'b')
        self.assertNotEqual(hash_a1, hash_b,
                            'different seeds produced identical LHE content')


if __name__ == '__main__':
    unittest.main()
