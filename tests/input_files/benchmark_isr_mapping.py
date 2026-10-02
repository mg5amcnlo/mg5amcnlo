#!/usr/bin/env python3
"""Manual, reproducible fixed-order ISR-map comparison (not a unit test).

Example:
  python3 benchmark_isr_mapping.py --mg5-root /path/to/mg5 --work /tmp/isr-bench \
      --cases dy dy_fiducial tt --seeds 271828 314159 161803 141421 173205

An exported process from this version can instead be supplied with
--existing-output PATH (one matching case only). That directory is copied,
never modified. Map 1 is symmetric; map 2 is asymmetric. The fiducial DY
case has pT_l > 25 GeV, |eta_l| < 2.5 and m_ll > 66 GeV, with no upper mass
cut. Both DY cases retain only the u ubar Born subprocess and its NLO real
channels; tt retains the gg Born subprocess and its NLO real channels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

PRESETS = {
    'dy': ('u u~ > e+ e- [QCD]', 91.188),
    'dy_fiducial': ('u u~ > e+ e- [QCD]', 91.188),
    'tt': ('g g > t t~ [QCD]', 173.0),
}


def invoke(args, cwd, log):
    with log.open('w') as stream:
        subprocess.run(args, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                       env={**os.environ, 'OMP_NUM_THREADS': '1'}, check=True)


def summarize(records):
    summaries = {}
    for case in sorted({r['case'] for r in records}):
        by_map = {m: {r['seed']: r for r in records
                      if r['case'] == case and r['mapping'] == m} for m in (1, 2)}
        seeds = sorted(by_map[1].keys() & by_map[2].keys())
        if not seeds:
            continue
        logs = [math.log(by_map[2][s]['error_pb'] ** 2 /
                         by_map[1][s]['error_pb'] ** 2) for s in seeds]
        mean_log = statistics.mean(logs)
        item = dict(paired_seeds=seeds,
                    geometric_variance_ratio=math.exp(mean_log),
                    median_variance_ratio=math.exp(statistics.median(logs)))
        if len(logs) > 1:
            se = statistics.stdev(logs) / math.sqrt(len(logs))
            item['approximate_95pct_interval_geometric_ratio'] = [
                math.exp(mean_log - 1.96 * se), math.exp(mean_log + 1.96 * se)]
        item['maps'] = {}
        for m in (1, 2):
            values = [by_map[m][s] for s in seeds]
            item['maps'][m] = dict(
                mean_xsec_pb=statistics.mean(r['xsec_pb'] for r in values),
                rms_error_pb=math.sqrt(statistics.mean(r['error_pb'] ** 2 for r in values)),
                mean_error_squared_times_points=statistics.mean(
                    r['error_squared_times_points'] for r in values),
                mean_integration_cpu_s=statistics.mean(r['integration_cpu_s'] for r in values))
        item['aggregate_variance_ratio'] = (
            sum(by_map[2][s]['error_pb'] ** 2 for s in seeds) /
            sum(by_map[1][s]['error_pb'] ** 2 for s in seeds))
        differences = [by_map[2][s]['xsec_pb'] - by_map[1][s]['xsec_pb'] for s in seeds]
        item['mean_paired_xsec_difference_pb'] = statistics.mean(differences)
        if len(differences) > 1:
            item['standard_error_paired_xsec_difference_pb'] = (
                statistics.stdev(differences) / math.sqrt(len(differences)))
        summaries[case] = item
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mg5-root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True,
                        help='New output directory; must not exist')
    parser.add_argument('--existing-output', type=Path)
    parser.add_argument('--cases', nargs='+', choices=PRESETS, default=['dy', 'dy_fiducial', 'tt'])
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[271828, 314159, 161803, 141421, 173205])
    parser.add_argument('--maps', nargs='+', type=int, choices=[0, 1, 2], default=[1, 2])
    parser.add_argument('--grid-points', type=int, default=5000)
    parser.add_argument('--grid-iterations', type=int, default=4)
    parser.add_argument('--points', type=int, default=50000)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--cores', type=int, default=2)
    parser.add_argument('--parallel', type=int, default=2)
    opts = parser.parse_args()
    if opts.existing_output and len(opts.cases) != 1:
        parser.error('--existing-output requires exactly one matching case')
    if any(s <= 0 for s in opts.seeds):
        parser.error('Use positive seeds so the run is reproducible')
    root = opts.mg5_root.resolve()
    work = opts.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    from madgraph.various.banner import RunCardNLO
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(opts).items()}
    metadata['python'] = sys.executable
    metadata['git_head'] = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    metadata['source_sha256'] = {
        name: hashlib.sha256((root/'Template/NLO/SubProcesses'/name).read_bytes()).hexdigest()
        for name in ['genps_fks.f', 'genps_fks_radiation.f', 'FKSParams.f90']}
    (work/'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    processes = {}
    for case in opts.cases:
        directory = work/('template_' + case)
        if opts.existing_output:
            shutil.copytree(opts.existing_output.resolve(), directory, symlinks=True)
        else:
            command = work/(case + '_generate.cmd')
            command.write_text('\n'.join([
                'set automatic_html_opening False --no_save',
                'set notification_center False --no_save',
                'set auto_update 0 --no_save', 'import model loop_sm',
                'generate ' + PRESETS[case][0], 'output ' + str(directory), 'quit']) + '\n')
            invoke([sys.executable, '-O', str(root/'bin/mg5_aMC'), str(command)],
                   root, work/(case + '_generate.log'))
        if not (directory/'bin/aMCatNLO').is_file():
            raise RuntimeError('Process generation failed; inspect generation log')
        processes[case] = directory

    def run_pair(task):
        case, mapping = task
        directory = work/f'{case}_map{mapping}'
        shutil.copytree(processes[case], directory, symlinks=True)
        # An existing output can contain absolute card symlinks. Materialize
        # all Cards content before changing it, so the donor remains untouched.
        cards = directory/'Cards'
        detached = directory/'Cards_isr_detached'
        shutil.copytree(cards, detached, symlinks=False)
        if cards.is_symlink():
            cards.unlink()
        else:
            shutil.rmtree(cards)
        detached.rename(cards)
        fks = directory/'Cards/FKS_params.dat'
        content = fks.read_text()
        if re.search(r'(?m)^#FKSISRMapping\s*$', content):
            content = re.sub(r'(?m)^(#FKSISRMapping\s*\n)[^\n]*',
                             lambda m: m.group(1) + str(mapping), content)
        else:
            content += '\n#FKSISRMapping\n' + str(mapping) + '\n'
        fks.write_text(content)
        records = []
        for seed in opts.seeds:
            name = f'isr_map{mapping}_{seed}'
            card = RunCardNLO(str(directory/'Cards/run_card_default.dat'))
            settings = dict(
                pdlabel='nn23nlo', reweight_scale=[False], reweight_pdf=[False],
                store_rwgt_info=False, iseed=seed, req_acc_fo=-1.,
                npoints_fo_grid=opts.grid_points, niters_fo_grid=opts.grid_iterations,
                npoints_fo=opts.points, niters_fo=opts.iterations,
                parton_shower='PYTHIA8', mcatnlo_delta=False, folding=[1, 1, 1],
                fixed_ren_scale=True, fixed_fac_scale=True,
                mur_ref_fixed=PRESETS[case][1], muf_ref_fixed=PRESETS[case][1],
                dynamical_scale_choice=[-2], ptl=25. if case == 'dy_fiducial' else 0.,
                etal=2.5 if case == 'dy_fiducial' else -1., drll=0., drll_sf=0.,
                mll=0., mll_sf=66. if case == 'dy_fiducial' else 30. if case == 'dy' else 0.,
                ptj=0., etaj=-1., ebeam1=6500., ebeam2=6500., lpp1=1, lpp2=1)
            if 'born_spreading' in card:
                settings['born_spreading'] = False
            for key, value in settings.items():
                card[key] = value
            card.write(str(directory/'Cards/run_card.dat'))
            command = work/f'{case}_{name}.cmd'
            command.write_text('\n'.join([
                'set automatic_html_opening False --no_save',
                'set notification_center False --no_save', 'set run_mode 2 --no_save',
                f'set nb_core {opts.cores} --no_save',
                f'launch NLO -f --name={name}', 'quit']) + '\n')
            log = work/f'{case}_{name}.log'
            started = time.monotonic()
            invocation = [sys.executable, '-O', str(directory/'bin/aMCatNLO'), str(command)]
            invoke(invocation, directory, log)
            elapsed = time.monotonic() - started
            event = directory/'Events'/name
            if not (event/'summary.txt').is_file():
                raise RuntimeError(f'Integration failed: inspect {log}')
            raw = work/'raw'/f'{case}_{name}'
            raw.mkdir(parents=True)
            channels = []
            for source in sorted((directory/'SubProcesses').glob('P*/*G*')):
                if not source.is_dir() or not (source/'res.dat').is_file():
                    continue
                target = raw/source.parent.name/source.name
                target.mkdir(parents=True)
                for path in source.iterdir():
                    if path.is_file() and not path.is_symlink() and path.name.startswith(
                            ('log', 'res', 'input', 'rand', 'mint', 'grid')):
                        shutil.copy2(path, target/path.name)
                values = [list(map(float, line.split())) for line in
                          (source/'res.dat').read_text().splitlines() if line.strip()]
                first = values[0]
                production_log = (source/'log_MINT1.txt').read_text()
                expected = 'symmetric' if mapping == 1 else 'asymmetric' if mapping == 2 else None
                if expected and f'FKS ISR mapping: {expected}' not in production_log:
                    raise RuntimeError('Requested mapping missing from production log; regenerate output')
                channels.append(dict(directory=str(source.relative_to(directory)),
                    xsec_pb=first[2], error_pb=first[3], iterations=int(first[4]),
                    points_per_iteration=int(first[5]), cpu_s=first[6],
                    configuration_rows=values[1:], input=(source/'input_app.txt').read_text()))
            if not channels:
                raise RuntimeError(f'No channel diagnostics found in {directory}')
            # Presets have one Born subprocess/integration group. Retain generic
            # aggregation for generated outputs with additional groups.
            points = sum(c['iterations'] * c['points_per_iteration'] for c in channels)
            error2 = sum(c['error_pb'] ** 2 for c in channels)
            record = dict(case=case, mapping=mapping, seed=seed,
                xsec_pb=sum(c['xsec_pb'] for c in channels), error_pb=math.sqrt(error2),
                production_points=points, error_squared_times_points=error2 * points,
                integration_cpu_s=sum(c['cpu_s'] for c in channels), elapsed_s=elapsed,
                invocation=invocation, cwd=str(directory), log=str(log),
                settings=settings, channels=channels)
            records.append(record)
            (work/f'{case}_map{mapping}_results.json').write_text(json.dumps(records, indent=2) + '\n')
            print(json.dumps({k: record[k] for k in ['case', 'mapping', 'seed', 'xsec_pb', 'error_pb']}), flush=True)
        return records

    with ThreadPoolExecutor(max_workers=opts.parallel) as pool:
        groups = list(pool.map(run_pair, [(case, m) for case in opts.cases for m in opts.maps]))
    records = [r for group in groups for r in group]
    (work/'results.json').write_text(json.dumps(records, indent=2) + '\n')
    (work/'summary.json').write_text(json.dumps(summarize(records), indent=2) + '\n')
    print(json.dumps(summarize(records), indent=2))


if __name__ == '__main__':
    main()
