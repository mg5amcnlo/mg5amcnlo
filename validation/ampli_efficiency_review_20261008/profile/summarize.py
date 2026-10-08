"""Recompute timers and replay invariance from preserved logs and isolated files."""
import hashlib
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
paths = json.loads((HERE / 'paths.json').read_text())
benchmarks = {
    '2500': Path('/tmp/mg5-ttbar-bounded-300k-kqeozrhq/ampli'),
    '30000': Path('/tmp/mg5-ttbar-30k-jobs-bx5bg6kc/ampli'),
}
timing_pattern = re.compile(r'Time spent in ([A-Za-z_0-9]+)\s*:\s*([\d.E+-]+)')
existing = {}
for label, directory in benchmarks.items():
    total = {}
    channels = {}
    for path in directory.glob('SubProcesses/P*/G*_*/log_MINT2.txt'):
        channel = path.parent.parent.name + '/' + path.parent.name.split('_')[0]
        row = channels.setdefault(channel, {'jobs': 0, 'timing': {}, 'trials': 0, 'candidates': 0, 'quota': 0})
        row['jobs'] += 1
        log = path.read_text()
        for name, value in timing_pattern.findall(log):
            value = float(value)
            total[name] = total.get(name, 0.) + value
            row['timing'][name] = row['timing'].get(name, 0.) + value
        match = re.search(r'AmpliCol generation trials, candidates, requested events:\s+(\d+)\s+(\d+)\s+(\d+)', log)
        if match:
            for name, value in zip(['trials', 'candidates', 'quota'], match.groups()):
                row[name] += int(value)
    existing[label] = {'total_timing': total, 'channels': channels}
(HERE / 'existing_timing.json').write_text(json.dumps(existing, indent=2) + '\n')

summaries = {}
for label, worker in [('2500', 'GF3.0_1'), ('30000', 'GF3.0_profile30k')]:
    directory = Path(paths['profile']) / 'SubProcesses/P0_gg_ttx' / worker
    original = benchmarks[label] / 'SubProcesses/P0_gg_ttx/GF3.0_1'
    log = (directory / 'profile.log').read_text()
    if 'Time spent in Total' not in log:
        continue
    timings = dict((key, float(value)) for key, value in timing_pattern.findall(log))
    measured = {}
    for name, keys in [
        ('sample,map,fun,virt,consider,finish', ['sample', 'map', 'integrand', 'virtual_integrand', 'consider', 'finish']),
        ('envelope,select,map_updates,next_cutoff', ['envelope', 'select', 'map_updates', 'next_cutoff']),
        ('integrand trials,virtual trials', ['trials', 'virtual_trials']),
    ]:
        values = re.search('PROFILE ' + re.escape(name) + r':([^\n]+)', log).group(1).split()
        measured.update(zip(keys, map(float, values)))
    invariant = {}
    for filename in ['ampli_candidates.lhe', 'ampli_pool.dat']:
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        a, b = digest(original / filename), digest(directory / filename)
        invariant[filename] = {'original_sha256': a, 'instrumented_sha256': b, 'identical': a == b}
        assert a == b, (label, filename)
    # res.dat ends with CPU seconds: verify all numerical fields before that.
    a, b = (original / 'res.dat').read_text().split(), (directory / 'res.dat').read_text().split()
    assert a[:-1] == b[:-1]
    invariant['res.dat'] = {'physics_and_point_counts_identical': True, 'original_cpu': float(a[-1]), 'instrumented_cpu': timings['Total']}
    measured['sampler_total'] = sum(measured[key] for key in ['sample', 'map', 'consider', 'finish'])
    measured['sampler_fraction'] = measured['sampler_total'] / timings['Total']
    measured['integrand_fraction'] = measured['integrand'] / timings['Total']
    measured['write_events_fraction'] = timings['Write_events'] / timings['Total']
    summaries[label] = {'profile': measured, 'legacy_timing': timings, 'invariance': invariant}
    shutil.copy(directory / 'profile.log', HERE / f'gg_GF3_{label}_profile.log')
(HERE / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
print(json.dumps(summaries, indent=2))
