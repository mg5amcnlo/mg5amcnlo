"""Survey quota collection, full-tail control, and AmpliCol LHE integrity."""

import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

from madgraph.various import ampli_pool
from tests.unit_tests.fks.test_ampli_lhe import candidate_file


def moments(points):
    n = len(points)
    absolute = sum(abs(x) for x in points) / n
    signed = sum(points) / n
    return dict(trials=n, mean_abs=absolute, mean_signed=signed,
                m2_abs=sum((abs(x)-absolute)**2 for x in points),
                m2_signed=sum((x-signed)**2 for x in points),
                covariance_sum=sum((abs(x)-absolute)*(x-signed) for x in points))


def native_pool_file(directory, epoch_points, final_quota, envelopes=None,
                     candidates=None, mask=(1, 0), log_z=0., targets=None,
                     proposal_ids=None, cutoffs=None):
    """Independent POOL4/5 fixture, including expired iteration records."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    envelopes = envelopes or [max(abs(value) for value in points) for points in epoch_points]
    if candidates is None:
        candidates = [(epoch_id, abs(value), math.log(abs(value))+math.log(2.),
                       1. if value >= 0. else -1.)
                      for epoch_id, points in enumerate(epoch_points, 1)
                      for value in points if value]
    event_epochs = sorted({row[0] for row in candidates})
    version = 5 if proposal_ids is not None else 4
    if version == 5:
        active = set(sorted({proposal_ids[index-1] for index in event_epochs})[-8:])
        eligible = [index for index in event_epochs if proposal_ids[index-1] in active]
    else:
        eligible = event_epochs[-8:]
    retained, rows = [], []
    for index, (epoch_id, weight, priority, factor) in enumerate(candidates):
        envelope = envelopes[epoch_id-1]
        threshold = envelope*math.exp(log_z)
        active = epoch_id in eligible
        keep = active and priority-math.log(envelope) > log_z
        correction = max(1., weight/threshold) if keep else 0.
        tail = int(active and weight > threshold)
        if keep:
            retained.append(index)
        rows.append([epoch_id, weight, priority, correction, tail, abs(factor)])
    target = len(retained)
    correction_sum = sum(row[3] for row in rows)
    for row in rows:
        row[3] *= target/correction_sum if correction_sum else 1.
    active_sum = sum(abs(value) for epoch_id, points in enumerate(epoch_points, 1)
                     for value in points if epoch_id in eligible)
    full_tail = sum(row[1] for row in rows if row[4])/active_sum
    weights = [rows[index][3]*rows[index][5] for index in retained]
    flags = [bool(rows[index][4]) for index in retained]
    heavy = [value for value, flag in zip(weights, flags) if flag]
    ordinary = sorted(value for value, flag in zip(weights, flags) if not flag)
    reserve_tail = sum(heavy)/sum(weights) if weights else 0.
    worst_tail = (0. if not heavy or not final_quota else 1. if len(heavy) >= final_quota
                  else sum(heavy)/(sum(heavy)+sum(ordinary[:final_quota-len(heavy)])))
    flat = [value for points in epoch_points for value in points]
    values = moments(flat)
    updates = ((proposal_ids[-1]-1 if proposal_ids else 0) if version == 5
               else len(epoch_points) if any(mask) else 0)
    text = ['MG5_AMPLI_POOL %d' % version, '%d %d %d' % (len(flat), len(rows), len(epoch_points)),
            ' '.join('%.16e' % values[key] for key in
                     ('mean_abs', 'mean_signed', 'm2_abs', 'm2_signed', 'covariance_sum')),
            '%d %d %.16e %.16e %.16e %.16e' %
            (target, final_quota, log_z, full_tail, reserve_tail, worst_tail),
            '%d %d' % (len(mask), updates),
            ' '.join(map(str, mask))]
    for epoch_id, points in enumerate(epoch_points, 1):
        values = moments(points)
        nonzero = sum(value != 0. for value in points)
        envelope = envelopes[epoch_id-1]
        cutoff = cutoffs[epoch_id-1] if cutoffs is not None else min(.1, envelope)
        target = targets[epoch_id-1] if targets is not None else nonzero
        text.append('%d %d %d %d ' % (epoch_id, len(points), nonzero, target) +
            ' '.join('%.16e' % values[key] for key in
                     ('mean_abs', 'mean_signed', 'm2_abs', 'm2_signed', 'covariance_sum')) +
            ' %.16e %.16e %.16e %d' %
            (cutoff, envelope, envelope*math.exp(log_z) if epoch_id in eligible else 0.,
             epoch_id in eligible) +
            (' %d' % proposal_ids[epoch_id-1] if version == 5 else ''))
    text.extend('%d %.16e %.16e %.16e %d %.16e' % tuple(row) for row in rows)
    (directory/'ampli_pool.dat').write_text('\n'.join(text)+'\n')
    (directory/'ampli_candidates.lhe').write_text(candidate_file([row[3] for row in candidates]))
    return directory


class TestAmpliPool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='mg5_ampli_pool_')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def pool_file(self, name, points, candidates, cutoff=1., event_weights=None):
        directory = self.work / name
        directory.mkdir()
        values = moments(points)
        content = 'MG5_AMPLI_POOL 1\n%d %d %.16e\n' % (
            len(points), len(candidates), cutoff)
        content += ' '.join('%.16e' % values[key] for key in (
            'mean_abs', 'mean_signed', 'm2_abs', 'm2_signed', 'covariance_sum')) + '\n'
        content += ''.join('%.16e %.16e\n' % candidate for candidate in candidates)
        (directory / 'ampli_pool.dat').write_text(content)
        if event_weights is None:
            event_weights = [1.] * len(candidates)
        (directory / 'ampli_candidates.lhe').write_text(candidate_file(event_weights))
        return directory

    def test_sidecar_reads_all_trial_moments_and_fortran_exponents(self):
        directory = self.pool_file('worker', [0., -1., 2., 0.],
                                   [(2., math.log(4.))])
        path = directory / 'ampli_pool.dat'
        path.write_text(path.read_text().replace('e+', 'D+').replace('e-', 'D-'))
        pool = ampli_pool.read_pool(directory)
        self.assertEqual(pool['trials'], 4)
        self.assertEqual(pool['ncandidates'], 1)
        self.assertEqual(pool['mean_abs'], .75)
        self.assertEqual(pool['mean_signed'], .25)
        self.assertEqual(pool['candidates'], [(2., math.log(4.))])
        self.assertEqual(Path(pool['lhe_path']), directory / 'ampli_candidates.lhe')

    def test_split_moments_match_every_point_and_include_zeroes(self):
        first = [0., -1., 2., 0.]
        second = [3., -5., 0., -2., 1., 0., 0.]
        combined = ampli_pool.combine_moments([moments(first), moments(second)])
        direct = moments(first + second)
        for key, value in direct.items():
            self.assertAlmostEqual(combined[key], value)
        n = len(first + second)
        self.assertAlmostEqual(combined['error_abs'],
                               math.sqrt(direct['m2_abs'] / (n*(n-1))))
        self.assertAlmostEqual(combined['error_signed'],
                               math.sqrt(direct['m2_signed'] / (n*(n-1))))
        self.assertAlmostEqual(combined['covariance'],
                               direct['covariance_sum'] / (n*(n-1)))

    def test_native_rates_include_the_survey_once_and_every_iteration(self):
        survey = dict(trials=100, absolute=5., signed=1., error_abs=.1, error_signed=.2)
        pools = [dict(epochs=[moments([0., -3., 6., 0.]),
                             moments([0., 0., 8., -2.])]),
                 dict(epochs=[moments([-2., 0., 5., 7., 0.])])]
        expected_means = [5., 1.]
        expected_variances = [.1**2, .2**2]
        count = 100
        for epoch in [row for pool in pools for row in pool['epochs']]:
            n = epoch['trials']
            for index, name in enumerate(('abs', 'signed')):
                value = epoch['mean_' + name]
                # Explicitly reproduce the upstream native formula, including
                # its population-variance and between-iteration terms.
                expected_variances[index] = (
                    expected_variances[index]*count**2 + epoch['m2_' + name]
                    )/(count+n)**2 + count*n*(expected_means[index]-value)**2/(count+n)**3
                expected_means[index] = (count*expected_means[index]+n*value)/(count+n)
            count += n
        combined = ampli_pool.combine_native_rates(survey, pools)
        self.assertEqual(combined['trials'], 113)
        self.assertEqual(combined['survey_trials'], 100)
        self.assertEqual(combined['production_trials'], 13)
        self.assertEqual(combined['production_iterations'], 3)
        self.assertAlmostEqual(combined['absolute'], expected_means[0])
        self.assertAlmostEqual(combined['signed'], expected_means[1])
        self.assertAlmostEqual(combined['error_abs'], math.sqrt(expected_variances[0]))
        self.assertAlmostEqual(combined['error_signed'], math.sqrt(expected_variances[1]))
        self.assertGreater(combined['error_abs'], .1*100/113)
        self.assertEqual(survey['trials'], 100)

    def test_native_pool_keeps_all_iteration_rates_and_last_eight_event_epochs(self):
        points = [[0., -float(index), float(index)] for index in range(1, 11)]
        directory = native_pool_file(self.work/'native', points, 14)
        pool = ampli_pool.read_pool(directory)
        self.assertEqual(pool['version'], 4)
        self.assertEqual(pool['trials'], 30)
        self.assertEqual(len(pool['epochs']), 10)
        self.assertEqual([row['id'] for row in pool['epochs'] if row['eligible']], list(range(3, 11)))
        self.assertEqual(ampli_pool.available_events([pool]), 16)
        self.assertTrue(all(row[2] == row[3] == 0. for row in pool['candidates'][:4]))
        self.assertEqual(pool['epochs'][0]['threshold'], 0.)
        survey = dict(trials=100, absolute=5., signed=1., error_abs=.1, error_signed=.1)
        combined = ampli_pool.combine_native_rates(survey, [pool])
        self.assertEqual(combined['production_trials'], 30)
        self.assertAlmostEqual(combined['absolute'], (500.+110.)/130.)
        self.assertAlmostEqual(combined['signed'], 100./130.)
        output = self.work/'native_events.lhe'
        ampli_pool.finalize_channel([pool], 14, output, 2., random.Random(13),
                                   cross_sections=[(42, 1., .1)])
        self.assertEqual(len(ET.fromstring(output.read_text()).findall('event')), 14)

    def test_native_collection_reallocates_pooled_reserve_without_worker_quota_constraints(self):
        first = ampli_pool.read_pool(native_pool_file(self.work/'native1', [[1., -1., 1.]], 2))
        second = ampli_pool.read_pool(native_pool_file(self.work/'native2', [[2., -2., 2., 2.]], 3))
        # Requested quota differs from the workers' nominal 2+3 allocation.
        selection, diagnostics = ampli_pool.select_candidates([first, second], 6, random.Random(2))
        self.assertEqual(len(selection), 6)
        self.assertEqual(diagnostics['available'], 7)
        with self.assertRaises(ampli_pool.PoolShortage):
            ampli_pool.pool_status([first, second], 8)

    def test_proposal_history_retains_more_than_eight_batches_with_individual_cutoffs(self):
        points = [[0., 1., -1.] for unused in range(12)]
        directory = native_pool_file(self.work/'stable_history', points, 20,
            proposal_ids=[1]*12, mask=(0, 0), cutoffs=[.1, .5]*6,
            targets=[2]*11+[1024])
        pool = ampli_pool.read_pool(directory)
        self.assertEqual(pool['version'], 5)
        self.assertEqual(pool['adaptation']['updates'], 0)
        self.assertEqual(pool['adaptation']['proposals'], 1)
        self.assertEqual(pool['adaptation']['history'], 'last_eight_event_proposals')
        self.assertEqual([epoch['id'] for epoch in pool['epochs'] if epoch['eligible']],
                         list(range(1, 13)))
        self.assertEqual([epoch['cutoff'] for epoch in pool['epochs']], [.1, .5]*6)
        self.assertEqual(pool['epochs'][-1]['target_nonzero'], 1024)
        self.assertEqual(pool['trials'], 36)
        self.assertEqual(ampli_pool.available_events([pool]), 24)
        output = self.work/'stable_events.lhe'
        result = ampli_pool.finalize_channel([pool], 20, output, 2., random.Random(17),
                                             cross_sections=[(42, 1., .1)])
        self.assertEqual(result['selected_tail'], 0.)
        self.assertEqual(len(ET.fromstring(output.read_text()).findall('event')), 20)

    def test_proposal_history_expires_whole_groups_and_skips_eventless_proposals(self):
        proposals = [1, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 10]
        points = [[1., -1.] for unused in proposals]
        # Proposal 2 has integration information but no stored events and
        # therefore must not displace a useful proposal from event history.
        candidates = [(epoch, 1., math.log(2.), 1.)
                      for epoch in range(1, 14) if epoch != 3
                      for unused in range(2)]
        directory = native_pool_file(self.work/'proposal_window', points, 18,
                                     proposal_ids=proposals, candidates=candidates)
        pool = ampli_pool.read_pool(directory)
        self.assertEqual([row['id'] for row in pool['epochs'] if row['eligible']],
                         list(range(4, 14)))
        self.assertEqual(pool['adaptation']['updates'], 9)
        self.assertEqual(pool['trials'], 26)
        self.assertEqual(ampli_pool.available_events([pool]), 20)
        self.assertTrue(all(row[2] == row[3] == 0. for row in pool['candidates'][:4]))
        # Even the expired/eventless observations still contribute to rates.
        self.assertEqual(pool['mean_abs'], 1.)
        self.assertEqual(pool['mean_signed'], 0.)

    def test_proposal_history_rejects_corrupt_ids_and_partial_group_eligibility(self):
        proposals = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
        directory = native_pool_file(self.work/'bad_proposals', [[1., -1.]]*11,
                                     16, proposal_ids=proposals)
        path = directory/'ampli_pool.dat'
        original = path.read_text().splitlines()
        mutations = [(6, 13, '0'), (6, 13, '2'), (7, 13, '3'),
                     (8, 13, '0'), (9, 13, '1'), (16, 13, '10'),
                     (4, 1, '7'), (7, 12, '1'), (9, 12, '0')]
        for line_index, word_index, value in mutations:
            with self.subTest(line=line_index, column=word_index, value=value):
                lines = original[:]
                row = lines[line_index].split()
                row[word_index] = value
                lines[line_index] = ' '.join(row)
                path.write_text('\n'.join(lines)+'\n')
                with self.assertRaises(ampli_pool.PoolError):
                    ampli_pool.read_pool(path)
        # Correctly formed IDs do not allow adaptation of folded grids.
        lines = original[:]
        lines[5] = '0 0'
        path.write_text('\n'.join(lines)+'\n')
        with self.assertRaisesRegex(ampli_pool.PoolError, 'adaptation mask'):
            ampli_pool.read_pool(path)

    def test_proposal_history_keeps_old_tails_and_all_independent_checks(self):
        points = [[2.]+[1.]*19] + [[1.]*20 for unused in range(9)]
        directory = native_pool_file(self.work/'stable_tail', points, 180,
                                     envelopes=[1.]*10, proposal_ids=[1]*10)
        pool = ampli_pool.read_pool(directory)
        self.assertTrue(pool['epochs'][0]['eligible'])
        self.assertEqual(pool['candidates'][0][3], 1.)
        self.assertAlmostEqual(pool['full_trial_tail'], 2./201.)
        self.assertGreater(pool['worst_subset_tail'], .01)
        with self.assertRaises(ampli_pool.PoolOverweight):
            ampli_pool.finalize_channel([pool], 180, self.work/'bad_tail.lhe', 1., random.Random(8))
        path = directory/'ampli_pool.dat'
        original = path.read_text().splitlines()
        for column in (3, 4, 5):
            lines = original[:]
            row = lines[3].split()
            row[column] = '0'
            lines[3] = ' '.join(row)
            path.write_text('\n'.join(lines)+'\n')
            with self.subTest(tail_column=column), self.assertRaises(ampli_pool.PoolError):
                ampli_pool.read_pool(path)

    def test_native_pool_versions_can_share_reallocated_collection(self):
        old = ampli_pool.read_pool(native_pool_file(self.work/'pool4', [[1., -1., 1.]], 2))
        new = ampli_pool.read_pool(native_pool_file(self.work/'pool5', [[1., -1., 1., 1.]], 3,
                                                   proposal_ids=[1]))
        selection, diagnostics = ampli_pool.select_candidates([old, new], 6, random.Random(19))
        self.assertEqual(len(selection), 6)
        self.assertEqual(diagnostics['available'], 7)
        self.assertEqual(diagnostics['overweight'], 0.)
        tightened = ampli_pool.tighten_native_pool(new)
        self.assertEqual(tightened['version'], 5)
        self.assertEqual(tightened['epochs'][0]['proposal_id'], 1)
        self.assertEqual(ampli_pool.pool_status([tightened], 3)['overweight'], 0.)

    def test_native_upward_rethresholding_preserves_files_and_uses_original_uniforms(self):
        candidates = [(1, 2., math.log(4.), -1.)] + [(1, 1., math.log(1.5), 1.)]*219
        directory = native_pool_file(self.work/'native_tail', [[2.]+[1.]*219], 200,
                                     envelopes=[1.], candidates=candidates)
        original = (directory/'ampli_pool.dat').read_bytes()
        pool = ampli_pool.read_pool(directory)
        self.assertLess(pool['reserve_tail'], .01)
        self.assertGreater(ampli_pool.pool_status([pool], 100)['worst_subset_tail'], .01)
        updated = ampli_pool.tighten_native_pool(pool)
        self.assertGreater(updated['log_z'], pool['log_z'])
        self.assertEqual(updated['generated_target'], 1)
        self.assertEqual(updated['candidates'][0][2:], (1., 0, 1.))
        self.assertTrue(all(row[2] == 0. for row in updated['candidates'][1:]))
        self.assertEqual(ampli_pool.pool_status([updated], 1)['overweight'], 0.)
        repeated = ampli_pool.tighten_native_pool(updated)
        self.assertEqual(repeated['log_z'], updated['log_z'])
        self.assertTrue(repeated['collection_rethresholded'])
        self.assertEqual(ampli_pool.pool_status([repeated], 1)['overweight'], 0.)
        self.assertEqual((directory/'ampli_pool.dat').read_bytes(), original)
        self.assertEqual(pool['generated_target'], 220)
        self.assertEqual(updated['epochs'][0]['mean_abs'], pool['epochs'][0]['mean_abs'])

    def test_native_pool_rejects_corrupt_epoch_and_candidate_evidence(self):
        directory = native_pool_file(self.work/'native_bad', [[0., 1., -1., 1.]], 2)
        path = directory/'ampli_pool.dat'
        original = path.read_text().splitlines()
        mutations = [(1, 0, '5'), (4, 1, '2'), (5, 0, '2'),
                     (6, 0, '2'), (6, 1, '3'), (6, 2, '5'),
                     (6, 4, '99'), (6, 10, '0'), (6, 11, '.01'),
                     (6, 12, '0'), (7, 0, '2'), (7, 3, '.5'), (7, 4, '1')]
        for line_index, word_index, value in mutations:
            with self.subTest(line=line_index, column=word_index, value=value):
                lines = original[:]
                words = lines[line_index].split()
                words[word_index] = value
                lines[line_index] = ' '.join(words)
                path.write_text('\n'.join(lines)+'\n')
                with self.assertRaises(ampli_pool.PoolError):
                    ampli_pool.read_pool(path)

    def test_native_final_iteration_can_finish_before_its_planned_budget(self):
        points = [[0., 1., -1., 1.], [1., 0., -1.]]
        directory = native_pool_file(self.work/'early_completion', points, 4,
                                     targets=[3, 1024])
        pool = ampli_pool.read_pool(directory)
        self.assertEqual(pool['epochs'][-1]['nonzero'], 2)
        self.assertEqual(pool['epochs'][-1]['target_nonzero'], 1024)
        self.assertEqual(pool['trials'], 7)
        self.assertEqual(pool['mean_abs'], 5./7.)
        status = ampli_pool.pool_status([pool], 4)
        self.assertEqual(status['overweight'], 0.)
        self.assertTrue(all(row[2] == 1. for row in pool['candidates']))
        survey = dict(trials=100, absolute=1., signed=.2,
                      error_abs=.1, error_signed=.1)
        combined = ampli_pool.combine_native_rates(survey, [pool])
        self.assertEqual(combined['production_trials'], 7)
        self.assertEqual(combined['trials'], 107)

    def test_native_early_completion_requires_last_epoch_and_positive_counts(self):
        for targets in ([4, 2], [0, 1024], [3, 0], [3, -1]):
            with self.subTest(targets=targets):
                directory = native_pool_file(self.work/'bad_early_completion',
                    [[0., 1., -1., 1.], [1., 0., -1.]], 4, targets=targets)
                with self.assertRaisesRegex(ampli_pool.PoolError, 'iteration state'):
                    ampli_pool.read_pool(directory)

    def test_native_early_completion_keeps_independent_tail_and_correction_checks(self):
        directory = native_pool_file(self.work/'early_bad_evidence',
                                     [[0., 1., -1., 1.]], 2, targets=[1024])
        path = directory/'ampli_pool.dat'
        original = path.read_text().splitlines()
        for line_index, word_index, value in [(3, 3, '.001'), (3, 4, '.001'),
                                             (3, 5, '.001'), (7, 3, '.5')]:
            with self.subTest(line=line_index, column=word_index):
                lines = original[:]
                row = lines[line_index].split()
                row[word_index] = value
                lines[line_index] = ' '.join(row)
                path.write_text('\n'.join(lines)+'\n')
                with self.assertRaises(ampli_pool.PoolError):
                    ampli_pool.read_pool(path)

    def test_empty_native_pool_has_no_trials_or_iterations(self):
        path = self.work/'ampli_pool.dat'
        for version in (4, 5):
            with self.subTest(version=version):
                path.write_text('MG5_AMPLI_POOL %d\n0 0 0\n0 0 0 0 0\n0 0 0 0 0 0\n2 0\n1 0\n' % version)
                pool = ampli_pool.read_pool(path)
                self.assertEqual(pool['version'], version)
                self.assertEqual(pool['epochs'], [])
                self.assertEqual(pool['candidates'], [])
                self.assertEqual(ampli_pool.pool_status([pool], 0)['overweight'], 0.)
                self.assertEqual(ampli_pool.tighten_native_pool(pool)['generated_target'], 0)

    def test_malformed_sidecars_fail_before_events_are_finalized(self):
        directory = self.pool_file('worker', [0., 1., 2.], [(2., math.log(4.))])
        path = directory / 'ampli_pool.dat'
        original = path.read_text()
        mutations = [
            original.replace('MG5_AMPLI_POOL 1', 'MG5_AMPLI_POOL 2'),
            original.replace('3 1 ', '0 1 '),
            original.rsplit('\n', 2)[0] + '\n',
            original + 'unexpected\n',
            original.replace('2.0000000000000000e+00', 'NaN'),
        ]
        for content in mutations:
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaises(ampli_pool.PoolError):
                    ampli_pool.read_pool(path)

    def test_lhe_scale_preserves_payload_and_bias_inverse(self):
        events = [part for kind, part in self.parts(candidate_file([2., -3.]))
                  if kind == 'event']
        for event, expected in zip(events, [10., -15.]):
            rewritten = ampli_pool._event_weight(event, 5.)
            before, after = ET.fromstring(event), ET.fromstring(rewritten)
            self.assertEqual(before.attrib, after.attrib)
            self.assertEqual(before.text.splitlines()[2:], after.text.splitlines()[2:])
            self.assertEqual(float(after.text.split()[2]), expected)
            self.assertEqual([ET.tostring(child) for child in before],
                             [ET.tostring(child) for child in after])

    def parts(self, content):
        path = self.work / 'lhe_test.lhe'
        path.write_text(content)
        return list(ampli_pool._lhe_parts(path))

    def test_init_updates_signed_rate_and_error_by_process_label(self):
        preamble = self.parts(candidate_file([1.]))[0][1]
        updated = ampli_pool._init_rates(preamble, [(42, -3.5, .07)], 2.)
        root = ET.fromstring(updated + '</LesHouchesEvents>')
        init_lines = root.find('init').text.splitlines()
        self.assertEqual(list(map(float, init_lines[2].split())), [-3.5, .07, 2., 42.])
        self.assertEqual(init_lines[1].split()[-2], '-4')
        with self.assertRaises(ampli_pool.PoolError):
            ampli_pool._init_rates(preamble, [(7, 1., .1)], 1.)

    def test_incomplete_lhe_is_rejected(self):
        content = candidate_file([1.])
        for truncated in (content[:content.index(' </event>')],
                          content[:content.index('</LesHouchesEvents>')]):
            with self.assertRaises(ampli_pool.PoolError):
                self.parts(truncated)

    def quota_pool(self, name, candidates, final_quota, threshold=1., cutoff=1.,
                   event_weights=None, points=None):
        """Write a v2 native worker with independently calculated diagnostics."""
        directory = self.work / name
        directory.mkdir()
        if points is None:
            points = [row[0] for row in candidates]
        values = moments(points)
        if event_weights is None:
            event_weights = [1.] * len(candidates)
        reserve = [index for index, (weight, priority) in enumerate(candidates)
                   if priority > math.log(threshold)]
        target = len(reserve)
        raw = [max(1., candidates[index][0] / threshold) for index in reserve]
        average = sum(raw)/target if target else 1.
        corrections = dict(zip(reserve, [value/average for value in raw]))
        flags = [weight > threshold for weight, priority in candidates]
        tail = sum(weight for (weight, unused), flag in zip(candidates, flags) if flag)
        full_tail = tail / sum(abs(point) for point in points) if tail else 0.
        physical = {index: corrections[index]*abs(event_weights[index]) for index in reserve}
        heavy = [physical[index] for index in reserve if flags[index]]
        ordinary = sorted(physical[index] for index in reserve if not flags[index])
        reserve_tail = sum(heavy)/sum(physical.values()) if heavy else 0.
        if not final_quota or not heavy:
            worst_tail = 0.
        elif len(heavy) >= final_quota:
            worst_tail = 1.
        else:
            worst_tail = sum(heavy)/(sum(heavy)+sum(ordinary[:final_quota-len(heavy)]))
        text = 'MG5_AMPLI_POOL 2\n%d %d %.16e\n' % (len(points), len(candidates), cutoff)
        text += ' '.join('%.16e' % values[key] for key in (
            'mean_abs', 'mean_signed', 'm2_abs', 'm2_signed', 'covariance_sum')) + '\n'
        text += '%d %d %.16e %.16e %.16e %.16e\n' % (
            target, final_quota, threshold, full_tail, reserve_tail, worst_tail)
        text += ''.join('%.16e %.16e %.16e %d %.16e\n' %
                        (weight, priority, corrections.get(index, 0.), flags[index],
                         abs(event_weights[index]))
                        for index, (weight, priority) in enumerate(candidates))
        (directory / 'ampli_pool.dat').write_text(text)
        (directory / 'ampli_candidates.lhe').write_text(candidate_file(event_weights))
        return directory

    def test_legacy_pools_are_diagnostic_only(self):
        pool = ampli_pool.read_pool(self.pool_file(
            'legacy', [1., 1.], [(1., math.log(2.)), (1., math.log(3.))]))
        self.assertEqual(pool['version'], 1)
        self.assertEqual(pool['adaptation'], {'mode': 'frozen'})
        self.assertEqual(ampli_pool.available_events([pool]), 2)
        with self.assertRaisesRegex(ampli_pool.PoolError, 'Legacy.*protocol 1'):
            ampli_pool.finalize_channel([pool], 1, self.work/'events.lhe', 1.,
                                       cross_sections=[(42, 1., .1)])

    def adaptive_pool(self, directory, record, mask):
        path = directory / 'ampli_pool.dat'
        lines = path.read_text().splitlines()
        lines[0] = 'MG5_AMPLI_POOL 3'
        lines[4:4] = [record, mask]
        path.write_text('\n'.join(lines) + '\n')
        return directory

    def test_adaptive_pool_keeps_draw_time_weights_and_tail_selection(self):
        candidates = [(1.015, math.log(2.))] + [(1., math.log(2.))]*219
        directory = self.quota_pool('adaptive', candidates, 200,
                                     event_weights=[-1.] + [1.]*219)
        frozen = ampli_pool.read_pool(directory)
        self.assertEqual(frozen['adaptation'], {'mode': 'frozen'})
        # 16+32+64 trials complete three batches, with 108 of 128 next trials.
        self.adaptive_pool(directory, '3 3 16 128 108', '1 0 1')
        adaptive = ampli_pool.read_pool(directory)
        self.assertEqual(adaptive['adaptation'], dict(
            mode='adaptive_unfolded', ndim=3, mask=[1, 0, 1], updates=3,
            interval=16, batch=128, points=108, completed_batches=3))
        self.assertEqual(adaptive['candidates'], frozen['candidates'])
        for key in ('threshold', 'full_trial_tail', 'reserve_tail', 'worst_subset_tail'):
            self.assertEqual(adaptive[key], frozen[key])
        for name, pool in (('frozen', frozen), ('adaptive', adaptive)):
            ampli_pool.finalize_channel([pool], 200, self.work/(name+'.lhe'),
                7., random.Random(42), cross_sections=[(42, 3.5, .07)])
        self.assertEqual((self.work/'frozen.lhe').read_text(),
                         (self.work/'adaptive.lhe').read_text())

    def test_adaptation_schedule_advances_even_without_successful_grid_updates(self):
        directory = self.quota_pool('zero_batches', [(1., math.log(2.))]*3, 2,
                                     points=[0.]*12 + [1.]*3)
        self.adaptive_pool(directory, '2 0 4 16 3', '1 0')
        adaptation = ampli_pool.read_pool(directory)['adaptation']
        self.assertEqual(adaptation['completed_batches'], 2)
        self.assertEqual(adaptation['updates'], 0)
        self.assertEqual(adaptation['points'], 3)

    def test_adaptation_schedule_caps_batches_and_handles_large_trial_counts(self):
        directory = self.quota_pool('capped', [(1., math.log(2.))]*3, 2)
        path = directory/'ampli_pool.dat'
        lines = path.read_text().splitlines()
        trials = 1024 * (2**6 - 1) + 65536 * 10**10 + 17
        lines[1] = '%d 3 1' % trials
        path.write_text('\n'.join(lines)+'\n')
        self.adaptive_pool(directory, '2 7 1024 65536 17', '0 1')
        adaptation = ampli_pool.read_pool(directory)['adaptation']
        self.assertEqual(adaptation['completed_batches'], 6 + 10**10)
        self.assertEqual(adaptation['updates'], 7)
        self.assertEqual(adaptation['points'], 17)

    def test_fully_folded_pool_records_disabled_adaptation(self):
        directory = self.quota_pool('folded', [(1., math.log(2.))]*3, 2)
        self.adaptive_pool(directory, '2 0 0 0 0', '0 0')
        adaptation = ampli_pool.read_pool(directory)['adaptation']
        self.assertEqual(adaptation['mode'], 'frozen')
        self.assertEqual(adaptation['mask'], [0, 0])
        self.assertEqual(adaptation['completed_batches'], 0)

    def test_corrupt_adaptation_metadata_is_rejected(self):
        directory = self.quota_pool('corrupt_adaptive', [(1., math.log(2.))]*3, 2,
                                     points=[1.]*15)
        path = directory/'ampli_pool.dat'
        original = path.read_text().splitlines()
        invalid = [
            ('2 2 4 16', '1 0'),          # missing counter
            ('2 2 4.0 16 3', '1 0'),      # noninteger interval
            ('2 2 4 16 3', '1'),          # mask dimension
            ('2 2 4 16 3', '1 2'),        # invalid mask value
            ('0 2 4 16 3', ''),           # empty dimension
            ('2 -1 4 16 3', '1 0'),       # negative updates
            ('2 3 4 16 3', '1 0'),        # too many successful updates
            ('2 2 4 8 3', '1 0'),         # stale batch size
            ('2 2 4 16 2', '1 0'),        # wrong accumulated trial count
            ('2 2 4 16 -1', '1 0'),       # negative batch count
            ('2 0 0 0 0', '1 0'),         # enabled mask without interval
            ('2 0 65537 65537 15', '1 0'), # interval exceeds cap
            ('2 0 4 16 3', '0 0'),        # disabled mask but live schedule
            ('2 1 0 0 0', '0 0'),         # disabled mask but updates
            ('2 0 0 0 15', '0 0'),        # disabled mask but points
        ]
        for record, mask in invalid:
            with self.subTest(record=record, mask=mask):
                lines = original[:]
                lines[0] = 'MG5_AMPLI_POOL 3'
                lines[4:4] = [record, mask]
                path.write_text('\n'.join(lines)+'\n')
                with self.assertRaises(ampli_pool.PoolError):
                    ampli_pool.read_pool(path)

    def test_full_tail_includes_entire_slightly_overweight_event(self):
        pool = ampli_pool.read_pool(self.quota_pool(
            'tail', [(1.015, math.log(4.)), (1., math.log(3.)), (1., math.log(2.))], 2))
        self.assertAlmostEqual(pool['full_trial_tail'], 1.015/3.015)
        self.assertAlmostEqual(pool['reserve_tail'], 1.015/3.015)
        self.assertAlmostEqual(pool['worst_subset_tail'], 1.015/2.015)
        # The obsolete mean excess is only 0.5% and would have passed.
        self.assertLess((1.015-1.)/3., .01)
        with self.assertRaises(ampli_pool.PoolOverweight):
            ampli_pool.finalize_channel([pool], 2, self.work/'events.lhe', 1.,
                                       cross_sections=[(42, 1.005, .005)])

    def test_all_trial_tail_uses_rejected_points_and_absolute_signs(self):
        pool = ampli_pool.read_pool(self.quota_pool(
            'tail', [(1.01, math.log(4.)), (1., math.log(3.)), (1., math.log(2.))], 2,
            points=[1.01, -1., 1., -.5, .5, 0., 0.]))
        self.assertAlmostEqual(pool['full_trial_tail'], 1.01/4.01)
        self.assertNotAlmostEqual(pool['full_trial_tail'], 1.01/1.01)
        self.assertEqual(pool['trials'], 7)

    def test_uniform_trimming_is_safe_for_every_subset_before_rng_draw(self):
        candidates = [(1.001, math.log(2.))] + [(1., math.log(2.))]*109
        pool = ampli_pool.read_pool(self.quota_pool('trim', candidates, 100))
        self.assertLess(pool['reserve_tail'], .01)
        self.assertGreater(pool['worst_subset_tail'], .01)
        output = self.work/'events.lhe'
        output.write_text('previous successful file')
        rng = random.Random(42)
        before = rng.getstate()
        with self.assertRaises(ampli_pool.PoolOverweight):
            ampli_pool.finalize_channel([pool], 100, output, 1., rng,
                                       cross_sections=[(42, 1., .01)])
        self.assertEqual(rng.getstate(), before)
        self.assertEqual(output.read_text(), 'previous successful file')
        self.assertEqual(list(self.work.glob('.ampli-*')), [])

    def test_small_tail_is_preserved_with_weighted_lhe_header(self):
        candidates = [(1.015, math.log(2.))] + [(1., math.log(2.))]*219
        pool = ampli_pool.read_pool(self.quota_pool('valid', candidates, 200,
                        event_weights=[-1.] + [1.]*219))
        output = self.work/'events.lhe'
        diagnostics = ampli_pool.finalize_channel([pool], 200, output, 7., random.Random(42),
                                                  cross_sections=[(42, 3.5, .07)])
        tree = ET.fromstring(output.read_text())
        events = tree.findall('event')
        self.assertEqual(len(events), 200)
        self.assertLess(diagnostics['selected_tail'], .01)
        self.assertLess(diagnostics['worst_subset_tail'], .01)
        self.assertEqual(tree.find('init').text.splitlines()[1].split()[-2], '-4')
        self.assertEqual(list(map(float, tree.find('init').text.splitlines()[2].split())),
                         [3.5, .07, 7., 42.])
        self.assertTrue(all(event.find('mgrwgt') is not None for event in events))
        self.assertTrue(all(event.find('rwgt') is not None for event in events))
        self.assertTrue(Path(pool['lhe_path']).exists())

    def test_workers_are_trimmed_independently_once_without_rethresholding(self):
        first = ampli_pool.read_pool(self.quota_pool(
            'first', [(1., math.log(3.))]*3, 2, threshold=2.,
            event_weights=[1., -1., 1.]))
        second = ampli_pool.read_pool(self.quota_pool(
            'second', [(10., math.log(40.))]*4, 3, threshold=30., cutoff=20.,
            event_weights=[-2., 2., -2., 2.]))
        class CountingRandom(random.Random):
            def __init__(self):
                super().__init__(42)
                self.calls = []
            def sample(self, population, k):
                self.calls.append((len(population), k))
                return super().sample(population, k)
        rng = CountingRandom()
        selected, info = ampli_pool.select_candidates([first, second], 5, rng)
        self.assertEqual(rng.calls, [(3, 2), (4, 3)])
        self.assertEqual(sum(worker == 0 for worker, unused in selected), 2)
        self.assertEqual(sum(worker == 1 for worker, unused in selected), 3)
        self.assertEqual(info['available'], 7)
        output = self.work/'events.lhe'
        ampli_pool.finalize_channel([first, second], 5, output, 7.,
                                   cross_sections=[(42, 3.5, .07)])
        weights = [float(event.text.split()[2]) for event in
                   ET.fromstring(output.read_text()).findall('event')]
        self.assertEqual([abs(value) for value in weights], [7., 7., 14., 14., 14.])
        with self.assertRaisesRegex(ampli_pool.PoolError, 'survey allocation'):
            ampli_pool.finalize_channel([first, second], 4, output, 7.,
                                       cross_sections=[(42, 3.5, .07)])

    def test_nominal_bias_factors_enter_both_tail_guards(self):
        candidates = [(1.001, math.log(2.))] + [(1., math.log(2.))]*219
        pool = ampli_pool.read_pool(self.quota_pool('bias', candidates, 200,
                        event_weights=[10.] + [1.]*219))
        self.assertLess(pool['full_trial_tail'], .01)
        self.assertGreater(pool['reserve_tail'], .01)
        self.assertGreater(pool['worst_subset_tail'], .01)
        with self.assertRaises(ampli_pool.PoolOverweight):
            ampli_pool.finalize_channel([pool], 200, self.work/'events.lhe', 1.,
                                       cross_sections=[(42, 1., .1)])

    def test_threshold_ties_are_not_overweight(self):
        pool = ampli_pool.read_pool(self.quota_pool(
            'ties', [(1., math.log(3.))]*3, 2, threshold=1.))
        self.assertEqual(pool['full_trial_tail'], 0.)
        self.assertEqual(pool['reserve_tail'], 0.)
        self.assertEqual(pool['worst_subset_tail'], 0.)
        self.assertTrue(all(not row[3] for row in pool['candidates']))

    def test_exact_one_percent_is_rejected(self):
        candidates = [(2., math.log(3.))] + [(1., math.log(3.))]*218
        pool = ampli_pool.read_pool(self.quota_pool('exact', candidates, 199))
        self.assertAlmostEqual(pool['worst_subset_tail'], .01)
        with self.assertRaises(ampli_pool.PoolOverweight):
            ampli_pool.finalize_channel([pool], 199, self.work/'events.lhe', 1.,
                                       cross_sections=[(42, 1., .1)])

    def test_saturated_threshold_retains_true_log_priority_order(self):
        directory = self.quota_pool('saturated', [(100., 713.)]*3, 2,
                                    threshold=sys.float_info.max,
                                    points=[100., 100., 100., 100.])
        path = directory/'ampli_pool.dat'
        lines = path.read_text().splitlines()
        counts = lines[1].split()
        counts[1] = '4'
        lines[1] = ' '.join(counts)
        # This excluded priority is above log(maxfloat), but below every
        # retained priority. Fortran saturates exp(712) at maxfloat.
        lines.append('100.0 712.0 0.0 0 1.0')
        path.write_text('\n'.join(lines)+'\n')
        Path(directory/'ampli_candidates.lhe').write_text(candidate_file([1.]*4))
        pool = ampli_pool.read_pool(directory)
        self.assertEqual(pool['full_trial_tail'], 0.)
        self.assertEqual(ampli_pool.available_events([pool]), 3)
        output = self.work/'events.lhe'
        ampli_pool.finalize_channel([pool], 2, output, 1.,
                                   cross_sections=[(42, 1., .1)])
        self.assertEqual(len(ET.fromstring(output.read_text()).findall('event')), 2)
        # Saturation never permits an excluded priority larger than a
        # retained one, nor relaxes ordinary finite-threshold validation.
        text = path.read_text()
        path.write_text(text.replace('100.0 712.0 0.0', '100.0 714.0 0.0'))
        with self.assertRaisesRegex(ampli_pool.PoolError, 'priority ordering'):
            ampli_pool.read_pool(directory)
        path.write_text(text.replace('1.7976931348623157e+308', '1.0000000000000000e+300'))
        with self.assertRaisesRegex(ampli_pool.PoolError, 'passing candidate'):
            ampli_pool.read_pool(directory)

    def test_actual_lhe_tail_is_rechecked_after_native_writer_rounding(self):
        candidates = [(2., math.log(3.))] + [(1., math.log(3.))]*218
        directory = self.quota_pool('rounded', candidates, 199,
                                    event_weights=[1.-1.e-9] + [1.]*218)
        pool = ampli_pool.read_pool(directory)
        self.assertLess(pool['worst_subset_tail'], .01)
        # e14.8 rounds the almost-unit factor to exactly one. Its resulting
        # final subset lies on the strict 1% boundary and must be rejected.
        Path(pool['lhe_path']).write_text(candidate_file([1.]*219))
        class FirstEvents:
            def sample(self, population, k):
                return population[:k]
        output = self.work/'events.lhe'
        output.write_text('previous successful file')
        with self.assertRaises(ampli_pool.PoolOverweight):
            ampli_pool.finalize_channel([pool], 199, output, 1., FirstEvents(),
                                       cross_sections=[(42, 1., .1)])
        self.assertEqual(output.read_text(), 'previous successful file')

    def test_unselected_candidates_are_retained_only_for_diagnostics(self):
        candidates = [(1., math.log(4.))]*3 + [(1., math.log(1.5))]
        pool = ampli_pool.read_pool(self.quota_pool('unused', candidates, 2, threshold=2.))
        self.assertEqual(pool['ncandidates'], 4)
        self.assertEqual(ampli_pool.available_events([pool]), 3)
        selected, unused = ampli_pool.select_candidates([pool], 2, random.Random(4))
        self.assertNotIn((0, 3), selected)

    def test_sidecar_tampering_fails_independent_reconstruction(self):
        directory = self.quota_pool('checked', [(1., math.log(3.))]*3, 2)
        path = directory/'ampli_pool.dat'
        original = path.read_text().splitlines()
        variants = []
        for line, field, replacement in ((3, 3, '.001'), (3, 4, '.002'), (3, 5, '.003'),
                                          (4, 2, '1.2'), (4, 3, '1'), (3, 1, '3')):
            changed = original[:]
            fields = changed[line].split()
            fields[field] = replacement
            changed[line] = ' '.join(fields)
            variants.append('\n'.join(changed)+'\n')
        for text in variants:
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(ampli_pool.PoolError):
                    ampli_pool.read_pool(directory)

    def test_lhe_factor_mismatch_and_corruption_preserve_existing_output(self):
        directory = self.quota_pool('checked', [(1., math.log(3.))]*3, 2)
        pool = ampli_pool.read_pool(directory)
        output = self.work/'events.lhe'
        output.write_text('previous successful file')
        candidate_path = Path(pool['lhe_path'])
        for weights in ([2., 1., 1.], [1., 1., 1., 1.], [1., 1.]):
            candidate_path.write_text(candidate_file(weights))
            with self.assertRaises(ampli_pool.PoolError):
                ampli_pool.finalize_channel([pool], 2, output, 1.,
                                           cross_sections=[(42, 1., .1)])
            self.assertEqual(output.read_text(), 'previous successful file')
            self.assertEqual(list(self.work.glob('.ampli-*')), [])

    def test_empty_worker_spool_borrows_nonempty_worker_header(self):
        empty = ampli_pool.read_pool(self.quota_pool('empty', [], 0, points=[0., 0.]))
        Path(empty['lhe_path']).write_text('</LesHouchesEvents>\n')
        filled = ampli_pool.read_pool(self.quota_pool(
            'filled', [(1., math.log(2.))]*2, 1))
        output = self.work/'events.lhe'
        ampli_pool.finalize_channel([empty, filled], 1, output, 1.,
                                   cross_sections=[(42, .25, .1)])
        tree = ET.fromstring(output.read_text())
        self.assertEqual(len(tree.findall('event')), 1)
        self.assertIsNotNone(tree.find('init'))

    def test_compact_worker_header_count_tracks_final_parent_quota(self):
        directory = self.quota_pool('worker', [(1., math.log(2.))]*2, 1)
        path = directory/'ampli_candidates.lhe'
        content = path.read_text().replace(
            '<header><generator>candidate test</generator></header>',
            '  <!--\n  <scalesfunctionalform>\n untouched\n  </scalesfunctionalform>\n'
            'PYTHIA8\n  -->\n  <header>\n     1500\n  </header>')
        path.write_text(content)
        output = self.work/'events.lhe'
        ampli_pool.finalize_channel([ampli_pool.read_pool(directory)], 1, output, 1.,
                                   cross_sections=[(42, .5, .1)])
        rewritten = output.read_text()
        self.assertIn('<header>\n        1\n  </header>', rewritten)
        self.assertIn(' untouched\n  </scalesfunctionalform>', rewritten)
        self.assertEqual(len(ET.fromstring(rewritten).findall('event')), 1)
