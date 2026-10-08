"""Read saved native pools to separate event-history and rejection losses."""
from pathlib import Path
import collections, json, math, re, statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def read_pool(path):
    with path.open() as f:
        assert next(f).split() == ['MG5_AMPLI_POOL', '4']
        trials, ncandidates, nepochs = map(int, next(f).split())
        moments = list(map(float, next(f).split()))
        target, quota, *rest = next(f).split()
        next(f); next(f)
        epochs = []
        for _ in range(nepochs):
            row = next(f).split()
            epochs.append(dict(id=int(row[0]), trials=int(row[1]), nonzero=int(row[2]),
                target=int(row[3]), absolute=float(row[4]), signed=float(row[5]),
                cutoff=float(row[9]), envelope=float(row[10]), threshold=float(row[11]),
                eligible=bool(int(row[12])), candidates=0, retained=0, retained_tail=0))
        for _ in range(ncandidates):
            birth, weight, priority, correction, tail, factor = next(f).split()
            epoch = epochs[int(birth)-1]
            epoch['candidates'] += 1
            epoch['retained'] += int(float(correction) > 0)
            epoch['retained_tail'] += int(float(correction) > 0 and float(tail) > 0)
        assert not f.read().strip()
    return dict(trials=trials, candidates=ncandidates, target=int(target), quota=int(quota),
                epochs=epochs, absolute=moments[0], signed=moments[1])

reports = {}
for name in ('ttbar_bounded_300k_20261007', 'ttbar_30k_jobs_300k_20261007'):
    base = ROOT / 'validation' / name
    metrics = json.loads((base / 'metrics.json').read_text())
    workers = []
    for channel in metrics['production_manifest']['channels']:
        for batch in channel['batches']:
            directory = base / 'ampli' / batch['directory']
            pool = read_pool(directory / 'ampli_pool.dat')
            eligible = [e for e in pool['epochs'] if e['eligible']]
            expired = [e for e in pool['epochs'] if not e['eligible']]
            log = (directory / 'log_MINT2.txt').read_text()
            tails = [list(map(float, line.split())) for line in re.findall(
                r'AmpliCol native full-trial, reserve, worst-collected tails:\s*([^\n]+)', log)]
            quota_absent = sum(t == [1., 1., 1.] for t in tails[:-1])
            real_tail_fail = sum(max(t) >= .01 and t != [1., 1., 1.] for t in tails[:-1])
            actual = sum(e['retained'] for e in eligible)
            assert actual == pool['target']
            workers.append(dict(directory=batch['directory'], **pool,
                cpu_seconds=next(w['cpu_seconds'] for w in metrics['production_workers']
                                 if w['channel'] == str(Path(batch['directory']).relative_to('SubProcesses'))),
                expired_trials=sum(e['trials'] for e in expired),
                expired_candidates=sum(e['candidates'] for e in expired),
                eligible_trials=sum(e['trials'] for e in eligible),
                quota_absent_boundaries=quota_absent, real_tail_failure_boundaries=real_tail_fail,
                retained=actual))
    all_epochs = [e for w in workers for e in w['epochs']]
    active_epochs = [e for e in all_epochs if e['eligible']]
    totals = {key: sum(w[key] for w in workers) for key in (
        'trials','candidates','target','quota','expired_trials','expired_candidates',
        'eligible_trials','quota_absent_boundaries','real_tail_failure_boundaries','retained','cpu_seconds')}
    totals.update(workers=len(workers), epochs=len(all_epochs),
        expired_trial_fraction=totals['expired_trials']/totals['trials'],
        reserve_per_active_trial=totals['retained']/totals['eligible_trials'],
        final_per_all_trial=300000/totals['trials'],
        stored_but_not_retained=totals['candidates']-totals['retained'],
        first_iteration_trials=sum(w['epochs'][0]['trials'] for w in workers),
        final_iteration_trials=sum(w['epochs'][-1]['trials'] for w in workers),
        median_epoch_count=statistics.median(len(w['epochs']) for w in workers),
        max_epoch_count=max(len(w['epochs']) for w in workers),
        absolute_rate_weighted_threshold_ratio=sum(e['trials']*e['threshold'] for e in active_epochs)
            /sum(e['trials']*e['absolute'] for e in active_epochs),
        retained_by_epoch_age={str(age):sum(w['epochs'][-age-1]['retained'] for w in workers
            if len(w['epochs'])>age) for age in range(16)})
    reports[name] = dict(summary=totals, workers=workers,
        interpretation='Expired trials still enter the rate estimate. They cannot supply final events under the eight-epoch window. This is cost attribution, not a measured speedup from changing the window.')
(HERE / 'pool_diagnostics.json').write_text(json.dumps(reports, indent=2) + '\n')
print(json.dumps({name: data['summary'] for name,data in reports.items()}, indent=2))
