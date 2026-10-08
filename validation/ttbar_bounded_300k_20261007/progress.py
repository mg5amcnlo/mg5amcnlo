"""Read completed worker counters without disturbing the running benchmark."""
from pathlib import Path
import datetime, json, os

HERE = Path(__file__).resolve().parent
WORK = Path((HERE / 'work_directory.txt').read_text().strip()) / 'ampli'
totals = dict(completed_workers=0, trials=0, iterations=0, grid_updates=0,
              worker_cpu_seconds=0., maximum_finished_worker_tail=0.)
channels = {}
for p in sorted((WORK / 'SubProcesses').glob('P*/G*/log_MINT2.txt')):
    result = p.parent / 'res_2.dat'
    pool = p.parent / 'ampli_pool.dat'
    if not result.exists() or not pool.exists(): continue
    row = result.read_text().split()
    if len(row) < 7: continue
    with pool.open() as f:
        header = next(f).split()
        if header != ['MG5_AMPLI_POOL', '4']: continue
        trials, candidates, iterations = map(int, next(f).split())
        next(f)
        tails = list(map(float, next(f).split()))[3:]
        assert all(0. <= t < .01 for t in tails), p
        totals['maximum_finished_worker_tail'] = max(totals['maximum_finished_worker_tail'], *tails)
        dimensions, updates = map(int, next(f).split())
    totals['completed_workers'] += 1
    totals['trials'] += trials
    totals['iterations'] += iterations
    totals['grid_updates'] += updates
    totals['worker_cpu_seconds'] += float(row[6])
    key = p.parent.parent.name + '/' + p.parent.name.split('_')[0]
    channels[key] = channels.get(key, 0) + 1
started = json.loads((HERE / 'ampli_started.json').read_text())
end = datetime.datetime.now(datetime.timezone.utc)
if (HERE / 'ampli_finished.json').exists():
    end = datetime.datetime.fromisoformat(json.loads((HERE / 'ampli_finished.json').read_text())['finished'])
if (HERE / 'stopped.json').exists():
    end = datetime.datetime.fromisoformat(json.loads((HERE / 'stopped.json').read_text())['stopped'])
elapsed = (end-datetime.datetime.fromisoformat(started['started'])).total_seconds()
totals.update(elapsed_wall_seconds=elapsed, completed_by_channel=channels)
totals['generation_status'] = [x for x in (HERE / 'ampli.log').read_text().splitlines() if 'Idle:' in x][-1:]
totals['observed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
with (HERE / 'progress_history.jsonl').open('a') as history:
    history.write(json.dumps(totals) + '\n')
print(json.dumps(totals, indent=2))
