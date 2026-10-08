"""Reproduce the archived native AmpliCol run's rate and collection checks.

Run this script in place after the archive has been populated. --work and
--run allow the same checks against an unarchived run with identical metadata.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import sys

sys.dont_write_bytecode = True


WORK = Path(__file__).resolve().parent
OUT = WORK / 'drell_yan'
RUN = 'native_restart6000'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return float(value.replace('D', 'E').replace('d', 'e'))


def close(actual, expected, label, rel=2.e-10, absolute=2.e-12):
    assert math.isfinite(actual) and math.isfinite(expected), label
    assert math.isclose(actual, expected, rel_tol=rel, abs_tol=absolute), (
        label, actual, expected)


def combine_rates(survey, pools):
    """Independently reproduce upstream AmpliCol's iteration combination."""
    count = survey['trials']
    means = [survey['absolute'], survey['signed']]
    variances = [survey['error_abs']**2, survey['error_signed']**2]
    iterations = 0
    for pool in pools:
        for epoch in pool['epochs']:
            n = epoch['trials']
            for index, name in enumerate(('abs', 'signed')):
                value = epoch['mean_' + name]
                # Native epoch Var(mean)=M2/N**2; native combination also
                # includes the difference between successive iteration means.
                variances[index] = (
                    count**2 * variances[index] + epoch['m2_' + name]
                ) / (count+n)**2 + count*n*(means[index]-value)**2 / (count+n)**3
                means[index] = (count*means[index]+n*value)/(count+n)
            count += n
            iterations += 1
    return dict(absolute=means[0], signed=means[1],
                error_abs=math.sqrt(variances[0]), error_signed=math.sqrt(variances[1]),
                trials=count, survey_trials=survey['trials'],
                production_trials=count-survey['trials'], production_iterations=iterations)


def allocate(channels, field, nevents, seed):
    """Replay the common allocation uniforms in the persisted CDF order."""
    ordered = sorted(channels, key=lambda channel: channel['allocation_order'])
    assert sorted(channel['allocation_order'] for channel in channels) == list(range(len(channels)))
    positive = [channel for channel in ordered if channel[field] > 0.]
    total = math.fsum(channel[field] for channel in positive)
    result = {(channel['subprocess'], channel['channel']): 0 for channel in channels}
    rng = random.Random(seed ^ 0x414D504C)
    for unused in range(nevents):
        target = rng.random()*total
        cumulative = 0.
        for channel in positive:
            cumulative += channel[field]
            if target < cumulative:
                break
        result[channel['subprocess'], channel['channel']] += 1
    return result


def main(work=WORK, run=RUN):
    out = work / 'drell_yan'
    started = json.loads((work/'started.json').read_text())
    finished = json.loads((work/'finished.json').read_text())
    assert finished['returncode'] == 0 and finished['events_exist']
    settings = started['settings']
    snapshots = started['source_sha256']
    assert snapshots and finished['source_sha256'] == snapshots
    for name, expected in snapshots.items():
        assert digest(work/'sources'/name) == expected, ('source snapshot', name)
    # Read sidecars with the exact collector source that was used by this run.
    spec = importlib.util.spec_from_file_location(
        'tested_ampli_pool', work/'sources/madgraph/various/ampli_pool.py')
    ampli_pool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ampli_pool)

    manifest = json.loads((out/'Events'/run/'ampli_production.json').read_text())
    nevents = settings['nevents']
    assert manifest['version'] == 4
    assert manifest['rate_source'] == 'survey_plus_production'
    assert manifest['sampling'] == 'native_adaptive_unfolded'
    assert manifest['weight_convention'] == 'native_iteration_envelopes'
    assert manifest['uncertainty_convention'] == 'native_trial_weighted_iteration_errors'
    assert manifest['requested_events'] == nevents == 6000
    assert manifest['allowed_overweight_factor'] == .01
    assert sum(channel['quota'] for channel in manifest['channels']) == nevents
    assert sum(channel['initial_quota'] for channel in manifest['channels']) == nevents

    expected_weights, expected_tails, workers = [], [], []
    selection_rng = random.Random(manifest['seed'] ^ 0x504F4F4C)
    process_rates = defaultdict(lambda: [0., 0.])
    normalization = manifest['absolute_cross_section']
    if settings['event_norm'] == 'sum':
        normalization /= nevents
    elif settings['event_norm'] == 'unity':
        normalization = 1.
    for channel in manifest['channels']:
        parent = out/'SubProcesses'/channel['subprocess']/('GF'+channel['channel'])
        values = (parent/'res_1.dat').read_text().split()
        survey = dict(absolute=number(values[0]), error_abs=number(values[1]),
                      signed=number(values[2]), error_signed=number(values[3]),
                      trials=int(values[5]))
        assert survey['error_abs'] <= .03*survey['absolute']*(1.+1.e-12)
        assert survey['trials'] == channel['survey_points']
        close(channel['survey_absolute'], survey['absolute'], 'survey absolute')
        close(channel['survey_signed'], survey['signed'], 'survey signed')
        native, collection = [], []
        for batch in channel['batches']:
            pool = ampli_pool.read_pool(out/batch['directory'])
            assert pool['version'] == batch['pool_protocol'] == 4
            assert pool['trials'] == batch['trials']
            assert pool['generated_target'] == batch['generated_target']
            assert pool['final_quota'] == batch['nominal_final_quota']
            assert pool['generated_target'] == (11*pool['final_quota']+9)//10
            assert pool['adaptation'] == batch['adaptation']
            mask = pool['adaptation']['mask']
            assert mask[-3:] == [int(fold == 1) for fold in settings['folding']]
            assert all(mask[:-3])
            assert pool['adaptation']['schedule'] == 'nonzero_trials'
            assert sum(epoch['trials'] for epoch in pool['epochs']) == pool['trials']
            assert all(0 < epoch['target_nonzero'] <= epoch['nonzero'] <= epoch['trials']
                       for epoch in pool['epochs'])
            event_epochs = sorted(set(pool['candidate_epochs']))
            assert [epoch['id'] for epoch in pool['epochs'] if epoch['eligible']] == event_epochs[-8:]
            native_tails = {key: pool[key] for key in
                            ('full_trial_tail', 'reserve_tail', 'worst_subset_tail')}
            assert all(0. <= value < .01 for value in native_tails.values())
            close(batch['native_log_z'], pool['log_z'], 'native rejection level')
            view = pool
            if batch['collection_log_z'] != pool['log_z']:
                view = ampli_pool.tighten_native_pool(pool)
            close(batch['collection_log_z'], view['log_z'], 'collection rejection level')
            assert ampli_pool.available_events([view]) == batch['available']
            native.append(pool)
            collection.append(view)
            workers.append(dict(directory=batch['directory'], trials=pool['trials'],
                nonzero_trials=sum(epoch['nonzero'] for epoch in pool['epochs']),
                iterations=len(pool['epochs']), reserve=pool['generated_target'],
                nominal_quota=pool['final_quota'], collection_available=batch['available'],
                native_log_z=pool['log_z'], collection_log_z=view['log_z'],
                adaptation=pool['adaptation'], native_tail=native_tails))

        rates = combine_rates(survey, native)
        for key, value in rates.items():
            close(channel['updated_rates'][key], value, 'combined '+key)
        for key in ('absolute', 'signed', 'error_abs', 'error_signed'):
            close(channel[key], rates[key], 'published '+key)
        label = int(channel['subprocess'][1:].split('_')[0])
        process_rates[label][0] += channel['signed']
        process_rates[label][1] += channel['error_signed']**2
        assert sum(pool['trials'] for pool in native) == channel['generation_trials']
        assert ampli_pool.available_events(collection) == channel['available_candidates']
        if not channel['quota']:
            assert channel['selection'] is None
            continue
        status = ampli_pool.pool_status(collection, channel['quota'])
        assert status['overweight'] < .01
        selection, diagnostics = ampli_pool.select_candidates(collection, channel['quota'], selection_rng)
        assert len(selection) == channel['quota']
        for key, value in diagnostics.items():
            close(channel['selection'][key], value, 'selection '+key)
        # Sidecars record the nominal magnitudes after native LHE formatting,
        # so deterministic magnitude replay does not need large candidate
        # spools. The final LHE independently supplies signs and event count.
        for (pool_index, candidate_index), correction in selection.items():
            row = collection[pool_index]['candidates'][candidate_index]
            expected_weights.append(row[4]*correction*normalization)
            expected_tails.append(bool(row[3]))

    initial = allocate(manifest['channels'], 'survey_absolute', nevents, manifest['seed'])
    final = allocate(manifest['channels'], 'absolute', nevents, manifest['seed'])
    for channel in manifest['channels']:
        key = channel['subprocess'], channel['channel']
        assert initial[key] == channel['initial_quota'], ('initial CDF quota', key)
        assert final[key] == channel['quota'], ('updated CDF quota', key)
    close(manifest['cross_section'], sum(channel['signed'] for channel in manifest['channels']), 'total signed')
    close(manifest['absolute_cross_section'], sum(channel['absolute'] for channel in manifest['channels']), 'total absolute')
    close(manifest['uncertainty'], math.sqrt(sum(channel['error_signed']**2 for channel in manifest['channels'])), 'total uncertainty')
    assert sum(worker['trials'] for worker in workers) == manifest['generation_trials']
    assert any(worker['adaptation']['updates'] for worker in workers)
    assert any(worker['iterations'] > 1 for worker in workers)

    with gzip.open(out/'Events'/run/'events.lhe.gz', 'rt') as stream:
        text = stream.read()
    events = re.findall(r'<event(?:[ \t][^>]*)?>(.*?)</event>', text, re.S)
    weights = [number(event.strip().splitlines()[0].split()[2]) for event in events]
    assert len(weights) == len(expected_weights) == nevents
    assert all(math.isfinite(weight) and weight != 0. for weight in weights)
    assert any(weight < 0. for weight in weights)
    for actual, expected in zip(sorted(map(abs, weights)), sorted(expected_weights)):
        close(actual, expected, 'final LHE weight magnitude', rel=5.e-7)
    tail = sum(abs(weight) for weight, flag in zip(expected_weights, expected_tails) if flag)
    tail /= sum(map(abs, expected_weights))
    assert tail < .01
    init = re.search(r'<init>(.*?)</init>', text, re.S).group(1).strip().splitlines()
    assert int(init[0].split()[8]) == -4
    assert int(init[0].split()[9]) == len(init)-1 == len(process_rates)
    for line in init[1:]:
        rate, error, unused, label = line.split()
        expected_rate, variance = process_rates[int(label)]
        close(number(rate), expected_rate, 'LHE process rate', rel=5.e-8)
        close(number(error), math.sqrt(variance), 'LHE process error', rel=5.e-8)

    saved = json.loads((work/'survey_grids_before_generation.json').read_text())
    assert len(saved) == 2*len(manifest['channels'])
    for name, expected in saved.items():
        assert digest(out/name) == expected, ('saved grid changed', name)
    result = dict(run=run, events=len(events), negative_events=sum(weight < 0. for weight in weights),
        collected_tail_fraction=tail, survey_grids_unchanged=True,
        source_snapshots_match=True, source_sha256=snapshots,
        rates_recomputed_from_survey_once_and_all_production_iterations=True,
        initial_and_final_cdf_quotas_reproduced=True, deterministic_collection_reproduced=True,
        signed_rate_pb=manifest['cross_section'], error_pb=manifest['uncertainty'],
        absolute_rate_pb=manifest['absolute_cross_section'],
        generation_trials=manifest['generation_trials'], generation_cpu_seconds=manifest['generation_cpu_seconds'],
        generation_rounds=len(manifest['rounds']), iterations=sum(worker['iterations'] for worker in workers),
        grid_updates=sum(worker['adaptation']['updates'] for worker in workers),
        weight_abs_min=min(map(abs, weights)), weight_abs_max=max(map(abs, weights)),
        all_native_and_collected_tail_fractions_below_one_percent=True, workers=workers)
    (work/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--run', default=RUN)
    options = parser.parse_args()
    main(options.work.resolve(), options.run)
