#!/usr/bin/env python3
"""Independent-seed folded signed/narrow-tail generation comparison."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from madgraph.various import ampli_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=16)
    parser.add_argument('--quota', type=int, default=30000)
    parser.add_argument('--tail-width', type=float, default=.005)
    parser.add_argument('--initial-nonzero', type=int, default=128)
    parser.add_argument('--seed-offset', type=int, default=100)
    parser.add_argument('--label', default='replicas')
    args = parser.parse_args()
    out = HERE / args.label
    out.mkdir(exist_ok=True)
    records = []
    provenance = {}
    for mode, variant in enumerate(('baseline', 'candidate')):
        sources = HERE.parent / 'physics' / (variant+'_sources') / 'Template/NLO/SubProcesses'
        build_temp = tempfile.TemporaryDirectory(prefix='mg5_history_analytic_'+variant+'_')
        build = Path(build_temp.name)
        executable = build/'analytic_history'
        driver_copy = out / 'driver.f90'
        driver_copy.write_bytes((HERE/'driver.f90').read_bytes())
        paths = [sources/'integrator_helpers.f90', sources/'simple_integrator.f90', driver_copy]
        provenance[variant] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths}
        command = ['gfortran', '-O2', '-fcheck=all,no-recursion', '-ffpe-trap=invalid,zero,overflow',
                   '-fbacktrace'] + list(map(str, paths)) + ['-o', str(executable)]
        subprocess.run(command, cwd=build, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for replica in range(1, args.seeds+1):
            seed = 1000003 + (replica+args.seed_offset)*7919 + mode*10000019
            directory = out / ('%s_%02d' % (variant, replica))
            directory.mkdir(exist_ok=True)
            begin = time.monotonic()
            with (directory/'run.log').open('w') as log:
                result = subprocess.run([str(executable), str(seed), str(args.quota), str(args.tail_width),
                                         str(args.initial_nonzero)], cwd=directory,
                                        stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if result.returncode:
                raise RuntimeError('Failed replica: '+str(directory/'run.log'))
            wall = time.monotonic()-begin
            pool = ampli_pool.read_pool(directory)
            status = ampli_pool.pool_status([pool], args.quota)
            assert status['overweight'] < .01, status
            selection, selected = ampli_pool.select_candidates([pool], args.quota, random.Random(seed ^ 0x123567))
            assert selected['selected_tail'] < .01, selected
            observations = [tuple(map(float, row.split())) for row in (directory/'observables.dat').read_text().splitlines()]
            assert len(observations) == pool['ncandidates']
            weights = [pool['candidates'][index][2] for index in range(pool['ncandidates'])]
            shape = [math.fsum(weight*row[column] for weight, row in zip(weights, observations))/math.fsum(weights)
                     for column in range(3)]
            selected_sum = math.fsum(selection.values())
            final_shape = [math.fsum(weight*observations[index][column] for (unused,index),weight in selection.items())/selected_sum
                           for column in range(3)]
            log = (directory/'run.log').read_text()
            row = dict(variant=variant, replica=replica, seed=seed, quota=args.quota,
                       trials=pool['trials'], nonzero=sum(epoch['nonzero'] for epoch in pool['epochs']),
                       epochs=len(pool['epochs']), updates=pool['adaptation']['updates'],
                       proposals=pool['adaptation'].get('proposals', len(pool['epochs'])),
                       skipped_adaptations=log.count('AmpliCol native adaptation held'),
                       eligible_epochs=sum(epoch['eligible'] for epoch in pool['epochs']),
                       active_trials=sum(epoch['trials'] for epoch in pool['epochs'] if epoch['eligible']),
                       means=[pool['mean_abs'],pool['mean_signed']],
                       quoted_errors=[math.sqrt(pool['m2_abs'])/pool['trials'],math.sqrt(pool['m2_signed'])/pool['trials']],
                       reserve_shapes=shape, final_shapes=final_shape,
                       tail_checks=status, selected_tail=selected['selected_tail'], wall_seconds=wall)
            records.append(row)
            (directory/'diagnostics.json').write_text(json.dumps(row, indent=2)+'\n')
            for name in ('ampli_pool.dat', 'observables.dat'):
                path = directory/name
                with path.open('rb') as source, gzip.open(str(path)+'.gz', 'wb') as target:
                    shutil.copyfileobj(source, target)
                path.unlink()
            print(json.dumps({key: row[key] for key in ('variant','replica','trials','epochs','updates','proposals','skipped_adaptations')}), flush=True)
            (out/'records.json').write_text(json.dumps(records, indent=2)+'\n')
        build_temp.cleanup()
    base = .1+(1.-math.exp(-30.))/30.+.01
    truth = [.58*base, .025*base]
    shape_truth = [(.01+(1.-math.exp(-3.))/30.)/base,
                   (.1*args.tail_width+(math.exp(-30.*.72)-math.exp(-30.*(.72+args.tail_width)))/30.+.01)/base, .025/.58]
    summary = dict(seeds_per_variant=args.seeds, final_quota=args.quota, tail_width=args.tail_width,
                   initial_nonzero=args.initial_nonzero, seed_offset=args.seed_offset, truth=truth,
                   shape_truth=shape_truth, provenance=provenance, variants={})
    for variant in ('baseline', 'candidate'):
        rows = [row for row in records if row['variant'] == variant]
        analysis = dict(total_trials=sum(row['trials'] for row in rows),
                        mean_trials=statistics.mean(row['trials'] for row in rows),
                        total_skipped=sum(row['skipped_adaptations'] for row in rows),
                        replicas_with_skips=sum(row['skipped_adaptations'] > 0 for row in rows),
                        mean_epochs=statistics.mean(row['epochs'] for row in rows),
                        mean_updates=statistics.mean(row['updates'] for row in rows),
                        mean_eligible_epochs=statistics.mean(row['eligible_epochs'] for row in rows),
                        expired_trials=sum(row['trials']-row['active_trials'] for row in rows),
                        max_tail=max(row['tail_checks']['overweight'] for row in rows),
                        max_selected_tail=max(row['selected_tail'] for row in rows), rates=[], shapes=[])
        for index, exact in enumerate(truth):
            means = [row['means'][index] for row in rows]
            variances = [row['quoted_errors'][index]**2 for row in rows]
            mean = statistics.mean(means)
            scatter = statistics.stdev(means)
            se = scatter/math.sqrt(len(rows))
            analysis['rates'].append(dict(mean=mean, truth=exact, empirical_scatter=scatter,
                                         empirical_mean_error=se, mean_bias_z=(mean-exact)/se,
                                         rms_quoted_error=math.sqrt(statistics.mean(variances)),
                                         scatter_over_rms_quote=scatter/math.sqrt(statistics.mean(variances))))
        for index, exact in enumerate(shape_truth):
            values = [row['final_shapes'][index] for row in rows]
            mean = statistics.mean(values)
            se = statistics.stdev(values)/math.sqrt(len(rows))
            analysis['shapes'].append(dict(mean=mean, truth=exact, empirical_mean_error=se,
                                          mean_bias_z=(mean-exact)/se))
        summary['variants'][variant] = analysis
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
