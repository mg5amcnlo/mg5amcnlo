"""Tests of barrier pooling, sampling normalization and frozen inference."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from madgraph.various import virtual_pool as pool


class VirtualPoolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_job(self, name, rows, ledger='1 1 100\n', orders=1, pdgs='21 21 6 -6'):
        directory = self.root / name
        directory.mkdir()
        header = 'VIRTUAL_POOL_SAMPLES 2\n5 8 %s 4 2\n%s\n0 0 173 173\n1 1 0 0\n' % (orders, pdgs)
        (directory / pool.SAMPLES).write_text(header + ''.join(
            ' '.join(str(x) for x in (*row[:4], 8, *row[4:], *([0.]*6), .3, .7)) + '\n' for row in rows))
        if ledger is not None:
            (directory / pool.ITERATIONS).write_text(ledger)
        return directory

    def test_normalization_uses_accepted_attempts_and_loop_probability(self):
        # Rows already contain all phase-space, channel and helicity factors.
        # Rejected iterations and uncaptured attempts are not interchangeable.
        first = self.write_job('first', [(1, 7, 1, 1, .5, 3., 9., 6., .1, .2),
                                        (2, 7, 2, 1, .1, 1.e30, 1.e30, 0., .3, .4)],
                               ledger='1 1 100\n2 0 400\n3 1 200\n')
        second = self.write_job('second', [(1, 7, 1, 1, .25, 5., 10., 7., .2, .4)],
                                ledger='1 1 700\n')
        third = self.write_job('third', [(1, 8, 1, 1, 1., 2., 3., 1., .3, .7)],
                               ledger='1 1 40\n')
        metadata, rows, reports = pool.collect_samples([first, second, third],
                    {first: '7', second: '7', third: '8'})
        self.assertEqual(metadata['nf'], 8)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r['born'] for r in rows], [3., 5., 2.])
        self.assertEqual([r['weight'] for r in rows], [.002, .004, .025])
        self.assertEqual(reports[0]['discarded_rows'], 1)

    def test_missing_incomplete_and_empty_captures_have_explicit_fallback(self):
        missing = self.root / 'missing'
        missing.mkdir()
        output, report_path = self.root / 'model', self.root / 'report.json'
        report = pool.train_pool([missing], output, report_path)
        self.assertEqual(report['status'], 'skipped')
        self.assertFalse(report['model_written'])
        self.assertFalse(output.exists())
        incomplete = self.write_job('incomplete', [(1, 1, 1, 1, 1., 1., 2., 1., 0., 0.)], ledger=None)
        report = pool.train_pool([incomplete], output, report_path)
        self.assertEqual(report['status'], 'fallback')
        if report.get('reason') == 'numpy_unavailable':
            self.assertFalse(report['model_written'])
            self.assertFalse(output.exists())
        else:
            self.assertTrue(report['model_written'])
            self.assertEqual(pool.load_model(output)['orders'][0]['blend'], 0.)
        self.assertEqual(report['accepted_rows'], 0)

    def test_empty_replicas_still_contribute_attempted_point_normalization(self):
        active = self.write_job('active', [(1, 1, 1, 1, .5, 1., 2., 1., 0., 0.)])
        empty = self.write_job('empty', [], ledger='1 1 300\n')
        (empty / pool.SAMPLES).write_text('')
        _, rows, reports = pool.collect_samples([active, empty], {active: 'same', empty: 'same'})
        self.assertEqual(rows[0]['weight'], 1. / 400. / .5)
        self.assertEqual(reports[1]['status'], 'empty_samples')

    def test_stratified_warmup_uses_each_channels_attempted_points(self):
        job = self.write_job('stratified', [
            (1, 7, 1, 1, .5, 3., 9., 6., .1, .2),
            (2, 8, 1, 1, .25, 2., 4., 0., .2, .4),
            (3, 7, 2, 1, 1., 6., 18., 12., .3, .4)],
            ledger='1 1 100\nCHANNEL 1 7 20\nCHANNEL 1 8 80\n2 1 300\n')
        _, rows, reports = pool.collect_samples([job])
        self.assertEqual(reports[0]['accepted_attempts'], 400)
        self.assertEqual([r['weight'] for r in rows], [5/400/.5, 1.25/400/.25, 1/400])
        self.assertEqual([r['born'] for r in rows], [3., 2., 6.])
        self.assertEqual(pool._read_iterations(job), {1: (1, 100), 2: (1, 300)})
        for ledger in ('1 1 100\nCHANNEL 1 7 20\n',
                       '1 1 100\nCHANNEL 1 7 100\nCHANNEL 1 7 0\n',
                       '1 1 100\nCHANNEL 2 7 100\n',
                       '1 1 100\nCHANNEL 1 7 100\n'):
            with self.subTest(ledger=ledger):
                (job/pool.ITERATIONS).write_text(ledger)
                with self.assertRaisesRegex(ValueError, 'stratified'):
                    pool.collect_samples([job])

    def test_canonical_capture_objective_is_invariant_under_job_splitting(self):
        rows = [(i + 1, 7, 1, 1, .25 if i % 2 else .5,
                 float(i + 1), float(3 * i - 2), float(i + 2), i / 8., .3)
                for i in range(8)]
        full = self.write_job('full', rows, ledger='1 1 80\n')
        # Production split replicas multiply integrands by wgt_mult=.5,
        # which capture removes. Physical PDF/Jacobian factors remain in
        # these Born/virtual/baseline values, before attempts are combined.
        canonical_rows = []
        for row in rows:
            runtime_values = tuple(value * .5 for value in row[5:8])
            canonical_rows.append(row[:5] + tuple(value / .5 for value in runtime_values) + row[8:])
        split1 = self.write_job('split1', canonical_rows[:4], ledger='1 1 40\n')
        split2 = self.write_job('split2', canonical_rows[4:], ledger='1 1 40\n')
        other = self.write_job('other', [(1, 8, 1, 1, .2, 7., -9., 5., .6, .7)],
                               ledger='1 1 250\n')
        _, unsplit, _ = pool.collect_samples([full, other], {full: '7', other: '8'})
        _, split, _ = pool.collect_samples([split1, split2, other],
                                           {split1: '7', split2: '7', other: '8'})
        key = lambda r: (r['channel'], r['sample'])
        unsplit, split = sorted(unsplit, key=key), sorted(split, key=key)
        for field in ('born', 'virtual', 'baseline', 'weight'):
            self.assertEqual([r[field] for r in unsplit], [r[field] for r in split])
        for power in (1, 2):
            objective = lambda data: sum(r['weight'] * abs(r['virtual'] - r['baseline']) ** power
                                         for r in data)
            self.assertEqual(objective(unsplit), objective(split))

    def test_metadata_mismatch_never_merges_different_born_orderings(self):
        first = self.write_job('first', [])
        second = self.write_job('second', [], pdgs='21 21 -6 6')
        with self.assertRaisesRegex(ValueError, 'incompatible Born'):
            pool.train_pool([first, second], self.root / 'model')

    def test_malformed_duplicate_and_nonfinite_data_rejected(self):
        row = (1, 1, 1, 1, 1., 1., 2., 1., 0., 0.)
        duplicate = self.write_job('duplicate', [row, row])
        with self.assertRaisesRegex(ValueError, 'duplicate sample'):
            pool.collect_samples([duplicate])
        nonfinite = self.write_job('nonfinite', [row[:-1] + (float('nan'),)])
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            pool.collect_samples([nonfinite])
        iteration = self.write_job('iteration', [row], ledger='1 1 100\n1 0 100\n')
        with self.assertRaisesRegex(ValueError, 'duplicate iteration'):
            pool.collect_samples([iteration])
        with self.assertRaisesRegex(ValueError, 'every supplied'):
            pool.collect_samples([iteration], {})

    def test_folds_are_stratified_and_sample_orders_stay_together(self):
        rows = [dict(directory='job', channel=channel, iteration=iteration,
                     sample=sample, order=order) for channel in (1, 2)
                for iteration in (1, 2) for sample in range(1, 14) for order in (1, 2)]
        folds = pool._folds(rows, 3)
        assignments = {}
        counts = {}
        for row, fold in zip(rows, folds):
            key = (row['channel'], row['iteration'], row['sample'])
            self.assertEqual(assignments.setdefault(key, fold), fold)
            counts.setdefault(key[:2], [0, 0, 0])[fold] += row['order'] == 1
        for count in counts.values():
            self.assertLessEqual(max(count) - min(count), 1)

    def test_import_does_not_load_optional_numpy(self):
        code = "import sys; import madgraph.various.virtual_pool; assert 'numpy' not in sys.modules"
        subprocess.check_call([sys.executable, '-c', code], cwd=str(Path(__file__).resolve().parents[3]))

    def test_discovery_without_optional_numpy(self):
        # CI imports every module before selecting tests. -S reproduces a
        # minimal runner with no installed optional packages.
        result = subprocess.run(
            [sys.executable, '-S', 'tests/test_manager.py', 'test_phi4_vertex', '-t0'],
            cwd=Path(__file__).resolve().parents[3], capture_output=True,
            text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Ran 1 test', result.stdout + result.stderr)

    def test_missing_numpy_keeps_adaptive_mint_without_a_model(self):
        for size in (0, 1, 60):
            with self.subTest(rows=size):
                rows = [(i + 1, 1, 1, 1, .5, 1., 2., 1., i / 60., .3)
                        for i in range(size)]
                job = self.write_job('no_numpy_%d' % size, rows)
                output = self.root / 'model'
                companion = self.root / 'neural'
                companion.write_text('stale')
                with mock.patch.dict(sys.modules, {'numpy': None}):
                    report = pool.train_pool([job], output, neural=True, output_neural=companion)
                self.assertEqual(report['status'], 'fallback')
                self.assertFalse(report['model_written'])
                self.assertFalse(report['enabled'])
                self.assertEqual(report['reason'], 'numpy_unavailable')
                self.assertEqual(report['call_probability'], 'adaptive MINT')
                self.assertEqual(report['neural']['reason'], 'numpy_unavailable')
                self.assertFalse(output.exists())
                self.assertFalse(companion.exists())

    def test_retired_capture_format_requires_new_integration(self):
        job = self.write_job('obsolete', [])
        path = job / pool.SAMPLES
        path.write_text(path.read_text().replace('VIRTUAL_POOL_SAMPLES 2', 'VIRTUAL_POOL_SAMPLES 1'))
        with self.assertRaisesRegex(ValueError, 'restart integration'):
            pool.collect_samples([job])


if __name__ == '__main__':
    unittest.main()
