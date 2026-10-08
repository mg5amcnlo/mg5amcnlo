"""Attribute an instrumented worker's final empirical tails to internal streams."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from madgraph.various import ampli_pool


def group():
    return dict(trials=0, abs_sum=0., signed_sum=0., weight_square_sum=0., maximum=0.,
                candidates=0, retained=0, retained_correction_sum=0.,
                tail_count=0, tail_abs_sum=0., tail_correction_sum=0.)


def analyze(worker):
    pool = ampli_pool.read_pool(worker)
    _, checks = ampli_pool._native_worker_status(pool)
    streams = {name: group() for name in ('novi', 'virt')}
    epochs = [{name: group() for name in streams} for _ in pool['epochs']]
    trace = worker / 'ampli_stream_trials.dat'
    probabilities = {name: set() for name in streams}
    count = 0
    stored_count = 0
    for line in trace.open():
        if line.startswith('#'):
            continue
        fields = line.split()
        trial, epoch_id = map(int, fields[:2])
        stream = fields[2]
        probability, weight, signed = map(float, fields[3:6])
        stored = fields[6] == 'T'
        candidate = int(fields[7])
        count += 1
        assert trial == count
        probabilities[stream].add(probability)
        epoch = pool['epochs'][epoch_id-1]
        tail = epoch['eligible'] and weight > epoch['threshold']
        retained = False
        correction = 0.
        if stored:
            stored_count += 1
            assert candidate == stored_count
            row = pool['candidates'][candidate-1]
            assert pool['candidate_epochs'][candidate-1] == epoch_id
            assert math.isclose(row[0], weight, rel_tol=1.e-13)
            assert bool(row[3]) == tail
            retained = bool(row[2])
            correction = row[2]*row[4]
        else:
            assert candidate == stored_count and not tail
        for target in (streams[stream], epochs[epoch_id-1][stream]):
            target['trials'] += 1
            target['abs_sum'] += weight
            target['signed_sum'] += signed
            target['weight_square_sum'] += weight*weight
            target['maximum'] = max(target['maximum'], weight)
            target['candidates'] += stored
            target['retained'] += retained
            target['retained_correction_sum'] += correction
            if tail:
                target['tail_count'] += 1
                target['tail_abs_sum'] += weight
                target['tail_correction_sum'] += correction
    assert count == pool['trials'] and stored_count == pool['ncandidates']
    all_abs = math.fsum(record['abs_sum'] for record in streams.values())
    tail_abs = math.fsum(record['tail_abs_sum'] for record in streams.values())
    tail_correction = math.fsum(record['tail_correction_sum'] for record in streams.values())
    retained_correction = math.fsum(record['retained_correction_sum'] for record in streams.values())
    for name, record in streams.items():
        record['probabilities'] = sorted(probabilities[name])
        assert len(probabilities[name]) == 1
        record['trial_fraction'] = record['trials']/count
        record['absolute_rate_fraction'] = record['abs_sum']/all_abs
        record['full_tail_mass_fraction'] = record['tail_abs_sum']/tail_abs if tail_abs else 0.
        record['reserve_tail_mass_fraction'] = record['tail_correction_sum']/tail_correction if tail_correction else 0.
        record['mean_abs_contribution'] = record['abs_sum']/count
        record['mean_signed_contribution'] = record['signed_sum']/count
        record['conditional_second_moment'] = record['weight_square_sum']/record['trials']*record['probabilities'][0]**2
    if all(epoch['eligible'] for epoch in pool['epochs']):
        assert math.isclose(tail_abs/all_abs, checks['full_trial_tail'], rel_tol=1.e-11)
    assert math.isclose(tail_correction/retained_correction, checks['reserve_tail'], rel_tol=1.e-11)
    return dict(worker=str(worker), trials=count, candidates=stored_count,
                streams=streams, epochs=epochs, final_checks=checks,
                note='Final thresholds attributed using original per-trial stream identity. '
                'No claim that final historical thresholds were available earlier.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('worker', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = analyze(args.worker.resolve())
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for name, stream in report['streams'].items():
        print(name, 'draws', stream['trials'], 'abs fraction', stream['absolute_rate_fraction'],
              'max', stream['maximum'], 'tail count', stream['tail_count'],
              'tail fraction', stream['full_tail_mass_fraction'])
