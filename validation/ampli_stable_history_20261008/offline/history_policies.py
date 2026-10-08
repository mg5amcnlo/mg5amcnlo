"""Compare fixed history policies at the final saved ttbar state.

The archive envelopes already include later observations; this is neither an
earlier stopping replay nor a CPU-saving estimate. Policies do not search for
favourable-tail epoch subsets. We examine every native rank threshold, without
assuming reserve or worst-subset tail tests are monotone. The archived ttbar
candidate LHE factors all equal one; this script asserts that simplification.
"""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from madgraph.various import ampli_pool


def read(path):
    with path.open() as stream:
        assert next(stream).strip() == 'MG5_AMPLI_POOL 4'
        n, ncand, nepoch = map(int, next(stream).split())
        next(stream)
        quota, final, *_ = next(stream).split()
        next(stream)
        next(stream)
        epochs = []
        for unused in range(nepoch):
            row = list(map(float, next(stream).split()))
            epochs.append(dict(id=int(row[0]), trials=int(row[1]),
                mean_abs=row[4], cutoff=row[9], envelope=row[10],
                eligible=bool(row[12])))
        candidates = np.loadtxt(stream, ndmin=2)
    assert len(candidates) == ncand
    assert np.all(candidates[:, 5] == 1.)
    return n, int(quota), int(final), epochs, candidates


def scan(epochs, candidates, selected_epochs, quota, final):
    ids = candidates[:, 0].astype(int) - 1
    selected_epochs = np.asarray(selected_epochs, dtype=bool)
    candidates = candidates[selected_epochs[ids]]
    ids = ids[selected_epochs[ids]]
    envelopes = np.asarray([e['envelope'] for e in epochs])[ids]
    ranks = candidates[:, 2] - np.log(envelopes)
    lr = np.log(candidates[:, 1]) - np.log(envelopes)
    ordered = np.sort(ranks)[::-1]
    floor = max(math.log(e['cutoff']/e['envelope']) for e, yes in zip(epochs, selected_epochs) if yes)
    denominator = sum(e['trials']*e['mean_abs'] for e, yes in zip(epochs, selected_epochs) if yes)
    ceiling = int(np.count_nonzero(ranks > floor))
    counts = np.arange(final, ceiling+1)
    if not len(counts):
        return dict(eligible_epochs=np.flatnonzero(selected_epochs).tolist(),
            ceiling=ceiling, capacity=0, quota_safe=False, quota_status=None,
            best_status=None, log_floor=floor)
    thresholds = np.maximum(ordered[np.minimum(counts, len(ordered)-1)], floor)
    thresholds[counts == len(ordered)] = floor
    tail_order = np.argsort(lr)[::-1]
    tail_counts = np.searchsorted(-lr[tail_order], -thresholds, side='left')
    raw_cumulative = np.concatenate(([0.], np.cumsum(candidates[tail_order, 1])))
    relative_cumulative = np.concatenate(([0.], np.cumsum(np.exp(lr[tail_order]))))
    full = raw_cumulative[tail_counts]/denominator
    mass = relative_cumulative[tail_counts]*np.exp(-thresholds)
    reserve = mass/(mass + counts - tail_counts)
    worst = np.where(tail_counts >= final, 1., mass/(mass + final - tail_counts))
    safe = np.maximum(np.maximum(full, reserve), worst) < .01
    def status(index):
        return dict(available=int(counts[index]), log_z=float(thresholds[index]),
            full_trial_tail=float(full[index]), reserve_tail=float(reserve[index]),
            worst_subset_tail=float(worst[index]), tail_events=int(tail_counts[index]),
            safe=bool(safe[index]))
    best = np.flatnonzero(safe)
    quota_status = status(quota-final) if final <= quota <= ceiling else None
    return dict(eligible_epochs=(np.flatnonzero(selected_epochs)+1).tolist(),
        eligible_trials=sum(e['trials'] for e, yes in zip(epochs, selected_epochs) if yes),
        eligible_candidates=len(candidates), ceiling=ceiling,
        capacity=int(counts[best[-1]]) if len(best) else 0,
        quota_safe=quota_status['safe'] if quota_status else False,
        quota_status=quota_status,
        best_status=status(best[-1]) if len(best) else None,
        log_floor=floor, safe_runs=int(np.count_nonzero(safe & ~np.r_[False, safe[:-1]])))


def verify(original, result, use_best=False):
    """Check the resulting pool through production correction/tail validation.

    The file reader still enforces the old eight-epoch protocol. Read the
    original valid pool, then change its eligibility and threshold in memory.
    """
    state = result['best_status'] if use_best else result['quota_status']
    if state is None or not state['safe']:
        return False
    pool = copy.deepcopy(original)
    eligible = set(result['eligible_epochs'])
    for epoch in pool['epochs']:
        epoch['eligible'] = epoch['id'] in eligible
        epoch['threshold'] = (max(epoch['cutoff'], math.exp(math.log(epoch['envelope']) + state['log_z']))
                              if epoch['eligible'] else 0.)
    quota = state['available']
    pool['generated_target'] = quota
    pool['log_z'] = state['log_z']
    ranks = [(row[1] - math.log(pool['epochs'][epoch-1]['envelope']), i)
        for i, (epoch, row) in enumerate(zip(pool['candidate_epochs'], pool['candidates'])) if epoch in eligible]
    selected = set(i for _, i in sorted(ranks, reverse=True)[:quota])
    logs = {i:max(0., math.log(pool['candidates'][i][0]) - math.log(pool['epochs'][pool['candidate_epochs'][i]-1]['threshold'])) for i in selected}
    maximum = max(logs.values())
    norm = quota/math.fsum(math.exp(v-maximum) for v in logs.values())
    rows = []
    for i, (epoch_id, row) in enumerate(zip(pool['candidate_epochs'], pool['candidates'])):
        epoch = pool['epochs'][epoch_id-1]
        weight, priority, _, _, factor = row
        tail = int(epoch['eligible'] and weight > epoch['threshold'])
        correction = norm*math.exp(logs[i]-maximum) if i in selected else 0.
        rows.append((weight, priority, correction, tail, factor))
    pool['candidates'] = rows
    for key in ['full_trial_tail', 'reserve_tail', 'worst_subset_tail']:
        pool[key] = state[key]
    retained, status = ampli_pool._native_worker_status(pool)
    assert len(retained) == quota and max(status.values()) < .01
    return True


def main():
    report = dict(note=__doc__, runs={})
    old = json.loads((ROOT/'validation/ampli_efficiency_review_20261008/statistics/surplus.json').read_text())
    for run in ['ttbar_bounded_300k_20261007', 'ttbar_30k_jobs_300k_20261007']:
        workers = []
        for path in sorted((ROOT/'validation'/run/'ampli').glob('SubProcesses/P*/GF*/ampli_pool.dat')):
            trials, quota, final, epochs, candidates = read(path)
            original = ampli_pool.read_pool(path)
            last8 = [e['eligible'] for e in epochs]
            floor = max(e['cutoff']/e['envelope'] for e, yes in zip(epochs, last8) if yes)
            event_epochs = set(candidates[:, 0].astype(int))
            policies = dict(last8=last8,
                all_history=[e['id'] in event_epochs for e in epochs],
                last8_floor_compatible=[yes or (e['id'] in event_epochs and e['cutoff']/e['envelope'] <= floor)
                    for e, yes in zip(epochs, last8)])
            worker = dict(path=str(path.relative_to(ROOT)), worker=path.parent.name,
                subprocess=path.parent.parent.name, trials=trials, quota=quota, final_quota=final, policies={})
            for name, mask in policies.items():
                result = scan(epochs, candidates, mask, quota, final)
                result['quota_validator_passed'] = verify(original, result)
                result['capacity_validator_passed'] = verify(original, result, True)
                worker['policies'][name] = result
            baseline = worker['policies']['last8']
            assert baseline['quota_safe']
            previous = next(w for w in old['runs'][run]['workers'] if w['worker'] == worker['worker'] and w['subprocess'] == worker['subprocess'])
            assert baseline['capacity'] == previous['maximum_native_rank_capacity']['available']
            workers.append(worker)
        summary = {}
        for name in policies:
            baseline = [w['policies']['last8'] for w in workers]
            rows = [w['policies'][name] for w in workers]
            summary[name] = dict(workers=len(workers), quota=sum(w['quota'] for w in workers),
                final_quota=sum(w['final_quota'] for w in workers), capacity=sum(r['capacity'] for r in rows),
                eligible_trials=sum(r['eligible_trials'] for r in rows),
                workers_quota_safe=sum(r['quota_safe'] for r in rows),
                improved_capacity=sum(r['capacity'] > b['capacity'] for r,b in zip(rows,baseline)),
                unchanged_capacity=sum(r['capacity'] == b['capacity'] for r,b in zip(rows,baseline)),
                regressed_capacity=sum(r['capacity'] < b['capacity'] for r,b in zip(rows,baseline)),
                extra_capacity=sum(r['capacity']-b['capacity'] for r,b in zip(rows,baseline)),
                extra_eligible_trials=sum(r['eligible_trials']-b['eligible_trials'] for r,b in zip(rows,baseline)),
                largest_capacity_gain=max(r['capacity']-b['capacity'] for r,b in zip(rows,baseline)),
                largest_capacity_loss=min(r['capacity']-b['capacity'] for r,b in zip(rows,baseline)),
                disconnected_safe_threshold_ranges=sum(r['safe_runs'] > 1 for r in rows))
        report['runs'][run] = dict(summary=summary, workers=workers)
        print(run, json.dumps(summary, indent=2), flush=True)
    (HERE/'history_policies.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
