"""Build a snapshotted source variant and replay two fixed-input workers.

Use a fresh scratch copy of this directory to preserve the recorded results.
Run setup.py first, then run.py baseline, run.py candidate,
validate_collection.py and summarize.py, in that order.
Sources must first be snapshotted into <variant>_sources, preserving repo paths.
"""
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
paths = json.loads((HERE / 'paths.json').read_text())
variant = sys.argv[1]
if variant not in ('baseline', 'candidate'):
    raise ValueError('Expected baseline or candidate')
sources = HERE / (variant + '_sources')
export = Path(paths['isolated'])
process = export / paths['process']
out = HERE / variant
resolved_export = export.resolve()
for original_key in ('original', 'large'):
    original_export = Path(paths[original_key]).resolve()
    if resolved_export == original_export or original_export in resolved_export.parents:
        raise RuntimeError('Replay must use a separate isolated export')
for label in ('2500', '30000'):
    if (process / ('GF3.0_' + variant + label)).exists():
        raise RuntimeError('Replay worker already exists; run setup.py in fresh scratch space')
for source in (sources / 'Template/NLO/SubProcesses').iterdir():
    destination = (export/'SubProcesses'/source.name).resolve()
    if resolved_export not in destination.parents:
        raise RuntimeError('Destination source escapes the isolated export: ' + str(destination))
out.mkdir(exist_ok=True)

for source in (sources / 'Template/NLO/SubProcesses').iterdir():
    shutil.copyfile(source, export/'SubProcesses'/source.name)

with (out/'build.log').open('w') as log:
    result = subprocess.run(['make', '-j1', 'madevent_mintMC'], cwd=process, stdout=log, stderr=subprocess.STDOUT)
assert result.returncode == 0, (out/'build.log')

spec = importlib.util.spec_from_file_location('ampli_pool_snapshot', sources/'madgraph/various/ampli_pool.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
rows = {}
for label, original in [('2500', Path(paths['original'])), ('30000', Path(paths['large']))]:
    worker = process / ('GF3.0_' + variant + label)
    shutil.copytree(original/paths['process']/'GF3.0_1', worker, symlinks=True)
    target = out / label
    target.mkdir(exist_ok=True)
    start = time.monotonic()
    with (worker/'input_app.txt').open('rb') as data, (target/'run.log').open('wb') as log:
        run = subprocess.run(['../madevent_mintMC'], cwd=worker, stdin=data, stdout=log, stderr=subprocess.STDOUT)
    wall = time.monotonic() - start
    if run.returncode:
        raise RuntimeError('Worker failed: ' + str(target/'run.log'))
    pool = validator.read_pool(worker)
    retained, tails = validator._native_worker_status(pool)
    assert all(value < .01 for value in tails.values()), tails
    log = (target/'run.log').read_text()
    timing = dict((key, float(value)) for key, value in re.findall(r'Time spent in ([A-Za-z_0-9]+)\s*:\s*([\d.E+-]+)', log))
    epochs = pool['epochs']
    active_trials = sum(epoch['trials'] for epoch in epochs if epoch['eligible'])
    result_values = list(map(float, (worker/'res.dat').read_text().split()))
    row = dict(worker=str(worker), exit_code=run.returncode, wall_seconds=wall,
               timing=timing, trials=pool['trials'], candidates=pool['ncandidates'],
               quota=pool['final_quota'], reserve=pool['generated_target'],
               mean_signed=pool['mean_signed'], mean_abs=pool['mean_abs'],
               pooled_trial_error_signed=math.sqrt(pool['m2_signed']/(pool['trials']*(pool['trials']-1))),
               pooled_trial_error_abs=math.sqrt(pool['m2_abs']/(pool['trials']*(pool['trials']-1))),
               active_trials=active_trials, expired_trial_fraction=1-active_trials/pool['trials'],
               epochs=len(epochs), eligible_epochs=sum(epoch['eligible'] for epoch in epochs),
               adaptation=pool['adaptation'], tails=tails, validator_passed=True,
               published_rate=dict(absolute=result_values[0], error_abs=result_values[1],
                                   signed=result_values[2], error_signed=result_values[3]),
               res_dat=(worker/'res.dat').read_text().strip(),
               numerical_artifact_sha256={name: hashlib.sha256((worker/name).read_bytes()).hexdigest()
                                           for name in ['ampli_pool.dat', 'ampli_candidates.lhe']})
    for name in ['ampli_pool.dat', 'res.dat', 'input_app.txt', 'ampli_job.dat', 'moffset.dat']:
        shutil.copyfile(worker/name, target/name)
    (target/'candidate_lhe_path.txt').write_text(str(worker/'ampli_candidates.lhe')+'\n')
    rows[label] = row
    (out/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(label, json.dumps(row), flush=True)
