#!/usr/bin/env python3
"""Native end-to-end benchmark for pooled virtual approximation and generation.

Examples (from the repository root)::

    python3 tests/input_files/benchmark_virtual_quality.py run
        --repo . --workdir /tmp/virtual_quality --processes ttbar eeqq eeqqg
        --events 10000 --seeds 1731 2731 --accuracy .02 --cores 4
    python3 tests/input_files/benchmark_virtual_quality.py compare
        /tmp/virtual_quality/ttbar_mint/mint_1731.json
        /tmp/virtual_quality/ttbar_pool/pool_1731.json

Join the wrapped example lines into one shell command.

Each process/mode gets a separate export. Native MC@NLO runs all three stages,
produces unweighted signed LHE events, and skips the shower. The MINT mode sets
UsePooledVirtual=False and UsePolyVirtual=False. Supply --baseline-repo to hold
the reference implementation fixed across runtime changes. Existing successful
runs are reused; partial or failed run directories are never overwritten.

JSON records per-stage MadLoop calls, residual-event acceptance, overflow excess,
component CPU times, wall time, cross sections, native grids, model reports and
source hashes. Per-run snapshots preserve training records and model/grid state.
Comparisons include signed LHE distributions; bin pulls use conservative sums of
squared weights and are descriptive checks, not a proof of universal accuracy.
All generated code, captures and LHE data stay under --workdir. Use a fresh
workdir to test a changed source snapshot; existing exports remain frozen.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

def configure(root, mode, events, seed, accuracy, cores, run_settings=None):
    run = root / 'Cards/run_card.dat'
    card = run.read_text()
    values = {'nevents': events, 'req_acc': accuracy, 'iseed': seed, 'reweight_scale': '.False.', 'reweight_pdf': '.False.', 'store_rwgt_info': '.False.'}
    values.update(run_settings or {})
    for key, value in values.items():
        card, n = re.subn('^\\s*[^#!\\n][^=\\n]*=\\s*' + re.escape(key) + '(?=\\s|!|$)', f' {value} = {key}', card, flags=re.M | re.I)
        if not n:
            card += f'\n {value} = {key}\n'
    run.write_text(card)
    fks = root / 'Cards/FKS_params.dat'
    data = fks.read_text()
    for key, value in {'UsePooledVirtual': '.True.' if mode == 'pool' else '.False.', 'UsePolyVirtual': '.False.'}.items():
        data, n = re.subn('(#' + key + '\\s*\\n)[^\\n]+', lambda m: m[1] + value, data, flags=re.I)
        if n != 1 and not (key == 'UsePooledVirtual' and mode == 'mint' and n == 0):
            raise RuntimeError(f'Cannot find {key}')
    fks.write_text(data)
    cfg = root / 'Cards/amcatnlo_configuration.txt'
    data = cfg.read_text()
    for key, value in {'automatic_html_opening': 'False', 'nb_core': cores, 'run_mode': 2}.items():
        data, n = re.subn('^\\s*#?\\s*' + key + '\\s*=.*$', f'{key} = {value}', data, flags=re.M)
        if not n:
            data += f'\n{key} = {value}\n'
    cfg.write_text(data)

def summarize(root, name):
    eventdir = root / 'Events' / name
    result = {'process_dir': str(root), 'run_name': name, 'stages': {}}
    for stage in range(3):
        f = eventdir / f'alllogs_{stage}.html'
        if not f.exists():
            continue
        data = f.read_text(errors='replace')
        points = [int(x) for x in re.findall('Total points tried\\s*[:=]\\s*(\\d+)', data)]
        result['stages'][str(stage)] = {'virtual_points': sum(points), 'madloop_stat_blocks': len(points)}
        result['stages'][str(stage)]['upper_bound_failure_lines'] = [line.strip() for line in data.splitlines() if re.search('upper.*(fail|violat)|bound.*(fail|violat)', line, re.I)]
    summary = eventdir / 'summary.txt'
    if summary.exists():
        result['summary'] = summary.read_text()
    native = eventdir / 'native_snapshot'
    if not native.is_dir():
        native = root / 'SubProcesses'
    result['model_reports'] = {'SubProcesses/' + str(f.relative_to(native)): json.loads(f.read_text()) for f in native.glob('P*/virtual_pool_report.json')}
    for stage, metrics in result['stages'].items():
        data = (eventdir / f'alllogs_{stage}.html').read_text(errors='replace')
        for kind in ('virt', 'novi'):
            metrics['accepted_' + kind + '_events'] = sum(map(int, re.findall('events generated, ' + kind + ':\\s*(\\d+)', data)))
            metrics['upper_bound_failures_' + kind] = sum(map(int, re.findall('upper bound failure, ' + kind + ':\\s*(\\d+)', data)))
        metrics['candidate_function_calls'] = sum(map(int, re.findall('another call to the function:\\s*(\\d+)', data)))
        if metrics['accepted_virt_events']:
            metrics['virtual_points_per_virtual_event'] = metrics['virtual_points'] / metrics['accepted_virt_events']
        metrics['channels'] = []
        channel_names = re.findall(r'LOG file for integration channel\s+(\S+),', data)
        for block_index, block in enumerate(data.split('<PRE>')[1:]):

            def ints(pattern):
                return [int(x) for x in re.findall(pattern, block)]
            calls = ints('Total points tried\\s*[:=]\\s*(\\d+)')
            accepted = ints('events generated, virt:\\s*(\\d+)')
            efficiencies = re.findall('Generation efficiencies:\\s*([\\d.Ee+-]+)\\s+([\\d.Ee+-]+)', block)
            overflow = re.search(r'Virtual envelope overflows / excess fraction:\s*(\d+)\s+([\d.Ee+-]+)', block)
            metrics['channels'].append({'overflow_count': int(overflow[1]) if overflow else None, 'overflow_excess_fraction': float(overflow[2]) if overflow else None, 'channel': channel_names[block_index] if block_index < len(channel_names) else None, 'virtual_points': sum(calls), 'accepted_virt_events': sum(accepted), 'efficiencies': efficiencies})
        metrics.pop('upper_bound_failure_lines', None)
        metrics['envelope_diagnostics'] = [line.strip() for line in data.splitlines() if 'Virtual envelope' in line or 'Virtual proposal attempts' in line or 'Virtual calibration' in line or 'Virtual Born carrier' in line or 'Virtual velocity carrier' in line]
        metrics['cpu_seconds_by_component'] = {}
        for component, seconds in re.findall('Time spent in (.+?)\\s*:\\s*([\\d.Ee+-]+)', data):
            metrics['cpu_seconds_by_component'][component] = metrics['cpu_seconds_by_component'].get(component, 0.0) + float(seconds)
        if metrics['virtual_points']:
            metrics['residual_event_acceptance_per_loop'] = metrics['accepted_virt_events'] / metrics['virtual_points']
    result['cross_section_pb'] = None
    result['cross_section_error_pb'] = None
    if summary.exists():
        cross = re.search('Total cross section:\\s*([\\d.Ee+-]+)\\s*\\+-\\s*([\\d.Ee+-]+)', summary.read_text())
        if cross:
            result['cross_section_pb'] = float(cross[1])
            result['cross_section_error_pb'] = float(cross[2])
    result['native_channel_grids'] = {}
    for f in native.glob('P*/GF*/mint_grids'):
        data = f.read_text()
        maxima = [float(x) for x in re.findall('^ MAX\\s+(\\S+)\\s*$', data, re.M)]
        sums = [[float(x) for x in line.split()] for line in re.findall('^ SUM\\s+(.+)$', data, re.M)]
        errors = [[float(x) for x in line.split()] for line in re.findall(r'^ QSM\s+(.+)$', data, re.M)]
        marker = re.search(r'^\s*VEP\s+(\d+)\s*$', data, re.M)
        envelope_version = int(marker[1]) if marker else None
        carrier_bounds = []
        velocity_bounds = []
        if envelope_version == 4 and sums:
            max_rows = [[float(x) for x in line.split()]
                        for line in re.findall(r'^\s*MAX\s+(.+)$', data, re.M)]
            bounds = max_rows[-len(sums):]
            if len(bounds) != len(sums) or any(len(row) != 2 for row in bounds):
                raise ValueError('Malformed VEP4 dual-carrier rows: %s' % f)
            carrier_bounds = [row[0] for row in bounds]
            velocity_bounds = [row[1] for row in bounds]
        elif envelope_version == 3 and sums:
            carrier_bounds = maxima[-len(sums):]
            maxima = maxima[:-len(sums)]
        split_replica = bool(re.search(r'_\d+$', f.parent.name))
        result['native_channel_grids']['SubProcesses/' + str(f.parent.relative_to(native))] = {'virtual_maxima': maxima, 'virtual_envelope_version': envelope_version, 'virtual_born_carrier_bounds': carrier_bounds, 'virtual_velocity_carrier_bounds': velocity_bounds, 'integrals': sums, 'errors': errors, 'split_replica': split_replica}
    tracked = [
        'SubProcesses/mint_module.f90', 'SubProcesses/driver_mintMC.f',
        'SubProcesses/driver_mintFO.f', 'SubProcesses/virtual_pooled.f90',
        'SubProcesses/virtual_surrogate.f90', 'SubProcesses/virtual_features.f90',
        'SubProcesses/virtual_approximation.f', 'SubProcesses/virtual_envelope_carrier.f90',
        'SubProcesses/fks_singular.f', 'SubProcesses/BinothLHA.f',
        'bin/internal/virtual_pool/__init__.py', 'bin/internal/virtual_pool/data.py',
        'bin/internal/virtual_pool/validation.py', 'bin/internal/virtual_pool/surrogate.py',
        'bin/internal/virtual_pool/polynomial.py', 'bin/internal/virtual_pool/helicity.py',
        'bin/internal/amcatnlo_run_interface.py',
    ]
    result['source_sha256'] = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in tracked if (root / f).exists()}
    archived_hashes = eventdir / 'native_snapshot' / 'source_sha256.json'
    if archived_hashes.is_file():
        result['source_sha256'] = json.loads(archived_hashes.read_text())
    result['input_sha256'] = {f: hashlib.sha256((root / 'Cards' / f).read_bytes()).hexdigest() for f in ('run_card.dat', 'param_card.dat', 'FKS_params.dat', 'MadLoopParams.dat') if (root / 'Cards' / f).is_file()}
    channel_grids = [g for g in result['native_channel_grids'].values() if g['integrals'] and not g['split_replica']]
    archived_inputs = eventdir / 'native_snapshot' / 'input_sha256.json'
    result['input_hashes_from'] = 'current_export_cards'
    if archived_inputs.is_file():
        result['input_sha256'] = json.loads(archived_inputs.read_text())
        result['input_hashes_from'] = 'run_snapshot'
    result['residual_absolute_integral_pb'] = sum(g['integrals'][0][4] for g in channel_grids)
    result['absolute_cross_section_pb'] = sum(g['integrals'][0][0] + g['integrals'][0][4] for g in channel_grids)
    if channel_grids:
        result['cross_section_pb'] = sum(g['integrals'][0][1] for g in channel_grids)
        result['cross_section_error_pb'] = math.sqrt(sum(g['errors'][0][1] ** 2 for g in channel_grids if g['errors']))
        result['residual_absolute_integral_error_pb'] = math.sqrt(sum(g['errors'][0][4] ** 2 for g in channel_grids if g['errors']))
    generated = result['stages'].get('2', {}).get('channels', [])
    excess_pb = 0.0
    complete_mapping = bool(generated)
    for channel in generated:
        name = (channel['channel'] or '').lstrip('/')
        grid = result['native_channel_grids'].get('SubProcesses/' + name)
        if grid is None or grid['split_replica']:
            complete_mapping = False
            continue
        channel['absolute_residual_pb'] = grid['integrals'][0][4]
        if channel['overflow_excess_fraction'] is None:
            # Older native logs do not report this diagnostic. Missing evidence
            # must not look like a measured zero-overflow result. A -1 residual
            # generation efficiency explicitly means no residual proposals were
            # made, including channels that only ran the calibration phase.
            not_sampled = bool(channel['efficiencies']) and all(
                float(pair[1]) < 0 for pair in channel['efficiencies'])
            channel['residual_generation_not_sampled'] = not_sampled
            if not not_sampled:
                complete_mapping = False
        else:
            excess_pb += channel['absolute_residual_pb'] * channel['overflow_excess_fraction']
    if complete_mapping and result['absolute_cross_section_pb'] > 0:
        result['estimated_envelope_excess_pb'] = excess_pb
        result['estimated_envelope_excess_fraction_of_absolute_total'] = excess_pb / result['absolute_cross_section_pb']
    for stage, metrics in result['stages'].items():
        text = (eventdir / f'alllogs_{stage}.html').read_text(errors='replace')
        metrics['virtual_calibration_attempts'] = sum(int(x) for x in re.findall(r'Virtual calibration phase-space attempts / evaluated residuals:\s*(\d+)', text))
        metrics['virtual_calibration_residual_evaluations'] = sum(int(x) for x in re.findall(r'Virtual calibration phase-space attempts / evaluated residuals:\s*\d+\s+(\d+)', text))
        metrics['virtual_calibration_channels'] = [{'overflows': int(n), 'largest_ratio': float(ratio), 'last_excess': float(excess)} for n, ratio, excess in re.findall(r'Virtual calibration overflows / largest ratio / last excess:\s*(\d+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)', text)]
        metrics['virtual_calibration_validated_channels'] = [v == 'T' for v in re.findall(r'Virtual calibration validated / scalar fallback:\s*([TF])', text)]
        metrics['virtual_born_carrier_attempts'] = sum(int(x) for x in re.findall(r'Virtual Born carrier attempts / vetoes:\s*(\d+)', text))
        metrics['virtual_born_carrier_vetoes'] = sum(int(x) for x in re.findall(r'Virtual Born carrier attempts / vetoes:\s*\d+\s+(\d+)', text))
        metrics['virtual_born_carrier_ratio_bounds_by_channel'] = [float(x) for x in re.findall(r'Virtual Born carrier ratio bound:\s*([\d.Ee+-]+)', text)]
        metrics['virtual_velocity_carrier_ratio_bounds_by_channel'] = [float(x) for x in re.findall(r'Virtual velocity carrier ratio bound:\s*([\d.Ee+-]+)', text)]
        metrics['virtual_proposal_attempts'] = sum((int(x) for x in re.findall('Virtual proposal attempts / evaluated residuals:\\s*(\\d+)', text)))
        metrics['virtual_proposal_evaluated_residuals'] = sum((int(x) for x in re.findall('Virtual proposal attempts / evaluated residuals:\\s*\\d+\\s+(\\d+)', text)))
        metrics['virtual_proposal_norms_by_channel'] = [{'base': float(base), 'effective_estimate': float(effective)} for base, effective in re.findall('Virtual proposal base norm / estimated effective norm:\\s*([\\d.Ee+-]+)\\s+([\\d.Ee+-]+)', text)]
        metrics['virtual_envelope_overflows_by_channel'] = [{'count': int(count), 'excess_fraction': float(excess)} for count, excess in re.findall('Virtual envelope overflows / excess fraction:\\s*(\\d+)\\s+([\\d.Ee+-]+)', text)]
    return result

def observables(path):
    import numpy as np
    values = []
    weights = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            if re.fullmatch(r'<event(?:\s[^>]*)?>', line.strip()) is None:
                continue
            header = next(f).split()
            n = int(header[0])
            weight = float(header[2])
            part = []
            for _ in range(n):
                fields = next(f).split()
                if int(fields[1]) != 1:
                    continue
                part.append((int(fields[0]), *[float(fields[i]) for i in [6, 7, 8, 9, 10]]))
            colored = [p for p in part if abs(p[0]) in (1, 2, 3, 4, 5, 6, 21)]
            ht = sum((math.hypot(p[1], p[2]) for p in colored))
            # Presets have colored final states; a colorless custom comparison
            # still has a useful leading-particle momentum.
            lead = max(colored or part, key=lambda p: math.hypot(p[1], p[2]))
            pt = math.hypot(lead[1], lead[2])
            eta = np.arcsinh(lead[3] / max(pt, 1e-15))
            tops = [p for p in part if abs(p[0]) == 6]
            sysp = tops if len(tops) == 2 else part
            total = [sum((p[i] for p in sysp)) for i in [1, 2, 3, 4]]
            mass = math.sqrt(max(0, total[3] ** 2 - sum((x * x for x in total[:3]))))
            values.append((ht, pt, eta, mass))
            weights.append(weight)
    return (np.array(values), np.array(weights))

def compare(a, b):
    import numpy as np
    output = {'baseline': str(a), 'candidate': str(b)}
    da, db = [json.loads(p.read_text()) for p in [a, b]]
    for name, d in [('baseline', da), ('candidate', db)]:
        output[name + '_metrics'] = {'calls_per_stage': {stage: x['virtual_points'] for stage, x in d['stages'].items()}, 'generation': d['stages']['2'], 'wall_seconds': d.get('wall_seconds'), 'summary': d.get('summary')}
    baseline_generation_calls = da['stages']['2']['virtual_points']
    baseline_total_calls = sum(s['virtual_points'] for s in da['stages'].values())
    output['generation_call_ratio'] = (db['stages']['2']['virtual_points'] / baseline_generation_calls
                                       if baseline_generation_calls else None)
    output['total_call_ratio'] = (sum(s['virtual_points'] for s in db['stages'].values()) / baseline_total_calls
                                 if baseline_total_calls else None)
    vals = []
    wgts = []
    for d in [da, db]:
        v, w = observables(Path(d['process_dir']) / 'Events' / d['run_name'] / 'events.lhe.gz')
        if not len(w):
            raise ValueError('No LHE events found for run %s' % d['run_name'])
        vals.append(v)
        wgts.append(w)
    output['lhe_events'] = {'baseline': len(wgts[0]), 'candidate': len(wgts[1])}
    output['negative_event_fractions'] = {'baseline': float(np.mean(wgts[0] < 0)), 'candidate': float(np.mean(wgts[1] < 0))}
    if all((d.get('cross_section_error_pb') is not None for d in (da, db))):
        error = math.hypot(da['cross_section_error_pb'], db['cross_section_error_pb'])
        output['cross_section_difference_sigma'] = (db['cross_section_pb'] - da['cross_section_pb']) / error if error else None
    output['distributions'] = {}
    for i, name in enumerate(('colored_ht', 'leading_colored_pt', 'leading_colored_eta', 'mass_tt_or_final_state')):
        combined = np.concatenate([v[:, i] for v in vals])
        edges = np.unique(np.quantile(combined, np.linspace(0, 1, 11)))
        if len(edges) < 3 or np.ptp(combined) < 1e-08 * max(1, abs(np.mean(combined))):
            continue
        edges[0] = -np.inf
        edges[-1] = np.inf
        content = []
        variances = []
        for v, w in zip(vals, wgts):
            content.append(np.histogram(v[:, i], bins=edges, weights=w)[0] / len(w))
            variances.append(np.histogram(v[:, i], bins=edges, weights=w * w)[0] / len(w) ** 2)
        deviation = np.sqrt(variances[0] + variances[1])
        # Quantile edges can leave empty bins when events share a value.
        # Both signed contents are then zero, so their pull is also zero.
        z = np.divide(content[1] - content[0], deviation,
                      out=np.zeros_like(deviation), where=deviation > 0)
        output['distributions'][name] = {'interior_edges': edges[1:-1].tolist(), 'baseline_variance': variances[0].tolist(), 'candidate_variance': variances[1].tolist(), 'baseline': content[0].tolist(), 'candidate': content[1].tolist(), 'pulls': z.tolist(), 'chi2': float(np.sum(z * z)), 'bins': len(z), 'largest_abs_pull': float(max(abs(z)))}
    return output
HADRON_SETTINGS = dict(lpp1=1, lpp2=1, ebeam1=6500, ebeam2=6500)
FOUR_FLAVOUR_SETTINGS = dict(HADRON_SETTINGS, pdlabel='lhapdf', lhaid=320500)
PRESETS = {
    'ttbar': ('p p > t t~ [QCD]', HADRON_SETTINGS),
    'ggtt': ('g g > t t~ [QCD]', HADRON_SETTINGS),
    'dy': ('p p > e+ e- [QCD]', HADRON_SETTINGS),
    'wjet': ('p p > w+ j [QCD]', dict(HADRON_SETTINGS, ptj=30)),
    'wbb_4fs': ('p p > w+ b b~ [QCD]', dict(FOUR_FLAVOUR_SETTINGS, ptb=20)),
    'eeqq': ('e+ e- > u u~ [QCD]', dict(lpp1=0, lpp2=0, ebeam1=125, ebeam2=125, ptj=10)),
    'eeqqg': ('e+ e- > u u~ g [QCD]', dict(lpp1=0, lpp2=0, ebeam1=125, ebeam2=125, ptj=10)),
}
FOUR_FLAVOUR_COMMANDS = ('import model loop_sm',
                        'define p = g u c d s u~ c~ d~ s~',
                        'define j = g u c d s u~ c~ d~ s~')
EXPORT_COMMANDS = dict(dy=('import model loop_sm-no_b_mass',),
                       ttbar=('import model loop_sm-no_b_mass',),
                       wjet=('import model loop_sm-no_b_mass',),
                       wbb_4fs=FOUR_FLAVOUR_COMMANDS)

def export_process(repo, root, process, cores, commands=()):
    """Export once, recording the exact source checkout and process command."""
    if root.exists():
        if not (root / 'bin/aMCatNLO').is_file():
            raise RuntimeError('Incomplete export exists: %s' % root)
        return
    if any((c.isspace() for c in str(root))):
        raise ValueError('MadGraph export paths must not contain whitespace')
    script = root.parent / (root.name + '_export.mg5')
    script.write_text('set automatic_html_opening False --no_save\nset crash_on_error True --no_save\nset nb_core %d --no_save\n' % cores+
                      ''.join(command+'\n' for command in commands)+
                      'generate %s\noutput %s -f\nquit\n' % (process, root))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with script.with_suffix('.log').open('w') as output:
        subprocess.run([sys.executable, str(repo / 'bin/mg5_aMC'), str(script)], cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    if not (root / 'bin/aMCatNLO').is_file():
        raise RuntimeError('MadGraph export failed; see %s' % script.with_suffix('.log'))
    (root / 'benchmark_export.json').write_text(json.dumps({'repo': str(repo), 'process': process,
                                                         'commands': list(commands)}, indent=2) + '\n')


def freeze_virtual_sources(repo, root, ref):
    """Freeze the pooled worker and its Python package before compilation/runs."""
    revision = subprocess.check_output(['git', 'rev-parse', '--verify', ref+'^{commit}'],
                                       cwd=repo, text=True).strip()
    mapping = {'SubProcesses/virtual_pooled.f90': 'Template/NLO/SubProcesses/virtual_pooled.f90'}
    mapping.update(('bin/internal/virtual_pool/'+name+'.py',
                    'madgraph/various/virtual_pool/'+name+'.py')
                   for name in ('__init__', 'data', 'validation', 'surrogate', 'polynomial', 'helicity'))
    payloads = {destination: subprocess.check_output(['git', 'show', revision+':'+source], cwd=repo)
                for destination, source in mapping.items()}
    hashes = {}
    for destination, payload in payloads.items():
        path = root/destination
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            path.unlink()
        path.write_bytes(payload)
        hashes[destination] = hashlib.sha256(payload).hexdigest()
    (root/'benchmark_virtual_ref.json').write_text(json.dumps(dict(revision=revision, source_sha256=hashes),
                                                            indent=2)+'\n')

def run_one(root, mode, events, seed, accuracy, cores, settings):
    name = '%s_%d' % (mode, seed)
    report = root / (name + '.json')
    if report.is_file():
        old = json.loads(report.read_text())
        expected = dict(events_requested=events, seed=seed, req_acc=accuracy,
                        mode=mode, run_settings=settings, returncode=0)
        if 'process_dir' in old:
            expected['process_dir'] = str(root)
        if old.get('completed') and all((old.get(k) == v for k, v in expected.items())):
            return report
        raise RuntimeError('Existing report has a different configuration: %s' % report)
    if (root / 'Events' / name).exists():
        raise RuntimeError('Run already exists without a matching report: %s' % (root / 'Events' / name))
    configure(root, mode, events, seed, accuracy, cores, settings)
    script = root / (name + '.cmd')
    script.write_text('set automatic_html_opening False --no_save\nset nb_core %d --no_save\nlaunch aMC@NLO -fp -n %s\nquit\n' % (cores, name))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    load_before = os.getloadavg() if hasattr(os, 'getloadavg') else None
    cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    with script.with_suffix('.log').open('w') as output:
        result = subprocess.run([sys.executable, str(root / 'bin/aMCatNLO'), str(script)], cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT)
    run_wall_seconds = time.monotonic() - started
    cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    metrics = summarize(root, name)
    complete = result.returncode == 0 and (root / 'Events' / name / 'events.lhe.gz').is_file()
    metrics.update(completed=complete, returncode=result.returncode,
                   wall_seconds=run_wall_seconds, events_requested=events,
                   seed=seed, req_acc=accuracy, mode=mode, run_settings=settings,
                   child_cpu_user_seconds=cpu_after.ru_utime-cpu_before.ru_utime,
                   child_cpu_system_seconds=cpu_after.ru_stime-cpu_before.ru_stime,
                   system_load_before=load_before,
                   system_load_after=os.getloadavg() if hasattr(os, 'getloadavg') else None,
                   native_cores=cores)
    snapshot = root / 'Events' / name / 'native_snapshot'
    for artifact in (root / 'SubProcesses').glob('P*/**/*'):
        if artifact.is_file() and (artifact.name.startswith('virtual_pool') or artifact.name in ('mint_grids', 'results.dat')):
            dest = snapshot / artifact.relative_to(root / 'SubProcesses')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact, dest)
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / 'source_sha256.json').write_text(json.dumps(metrics['source_sha256'], indent=2) + '\n')
    (snapshot / 'input_sha256.json').write_text(json.dumps(metrics['input_sha256'], indent=2) + '\n')
    metrics['input_hashes_from'] = 'run_snapshot'
    for card in metrics['input_sha256']:
        saved = snapshot / 'Cards' / card
        saved.parent.mkdir(exist_ok=True)
        shutil.copy2(root / 'Cards' / card, saved)
    report.write_text(json.dumps(metrics, indent=2) + '\n')
    if result.returncode or not complete:
        raise RuntimeError('Native benchmark failed; see %s' % script.with_suffix('.log'))
    print(json.dumps({'report': str(report), 'generation_virtual_calls': metrics['stages']['2']['virtual_points'], 'cross_section_pb': metrics['cross_section_pb'], 'wall_seconds': metrics['wall_seconds']}), flush=True)
    return report

def clone_export(source, destination):
    """Copy generated matrix elements without run output or source aliases.

    Links within the export are recreated inside the copy, preserving shared
    configuration files such as FKS_params.dat and randinit. External links
    are dereferenced so compiling or configuring the clone cannot mutate the
    source export or checkout.
    """
    source = source.resolve()
    destination = destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError('Clone must be outside its source export')
    if destination.exists():
        raise RuntimeError('Clone destination already exists: %s' % destination)

    def excluded(directory, names, dangling=True):
        current = Path(directory)
        # copytree(ignore_dangling_symlinks=True) checks relative targets
        # against the caller CWD in some Python versions. Resolve each link
        # at its actual parent here so valid relative links are retained.
        blocked = ({name for name in names if (current / name).is_symlink() and not (current / name).exists()}
                   if dangling else set())
        if current == source:
            blocked.update(set(names) & {'Events', 'HTML', 'RunWeb', 'RunMaterial', 'MCatNLO', 'MY_ANALYSIS', '.mg5'})
            # Reports belong to the source run. Copying one into a fresh
            # clone would let run_one reuse it without executing the clone.
            blocked.update(name for name in names if re.fullmatch(r'(pool|mint)_\d+\.(json|cmd|log)', name))
        if current == source / 'SubProcesses':
            blocked.update(set(names) & {'job_status.pkl', 'job_status2.pkl', 'results.dat', 'nevents_unweighted'})
        if current.parent.name == 'SubProcesses' and current.name.startswith('P'):
            blocked.update((name for name in names if name.startswith(('GF', 'GB', 'all_G', 'born_G')) and (current / name).is_dir()))
        return blocked
    shutil.copytree(source, destination, symlinks=False, ignore_dangling_symlinks=False, ignore=excluded)
    internal_links = 0
    for directory, directories, files in os.walk(source, followlinks=False):
        blocked = excluded(directory, directories + files, dangling=False)
        directories[:] = [name for name in directories if name not in blocked]
        for name in directories + files:
            if name in blocked:
                continue
            original = Path(directory) / name
            if not original.is_symlink():
                continue
            # Keep the direct target, rather than resolving an entire chain:
            # an internal link may end at an external library which was copied.
            target = Path(os.path.abspath(original.parent / os.readlink(original)))
            try:
                target_relative = target.relative_to(source)
            except ValueError:
                continue
            cloned_target = destination / target_relative
            copied = destination / original.relative_to(source)
            # Fresh exports contain links to includes generated only at launch
            # (notably run_card.inc). Preserve these internal dangling links;
            # the copy must resolve them when its own includes are written.
            if copied.is_dir():
                shutil.rmtree(copied)
            elif copied.exists() or copied.is_symlink():
                copied.unlink()
            copied.symlink_to(os.path.relpath(cloned_target, copied.parent),
                              target_is_directory=cloned_target.is_dir())
            internal_links += 1
    (destination / 'Events').mkdir(exist_ok=True)
    (destination / 'HTML').mkdir(exist_ok=True)
    (destination / 'benchmark_clone.json').write_text(json.dumps({'source': str(source), 'external_symlinks_dereferenced': True, 'internal_symlinks_preserved': internal_links}, indent=2) + '\n')
    return destination

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Export and run a native comparison suite')
    run.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    run.add_argument('--baseline-repo', type=Path)
    run.add_argument('--virtual-ref', help='freeze the pooled worker and Python package from this git revision in new exports')
    run.add_argument('--workdir', type=Path, required=True)
    run.add_argument('--processes', choices=sorted(PRESETS), nargs='+', default=['ttbar', 'eeqq', 'eeqqg'])
    run.add_argument('--modes', choices=['mint', 'pool'], nargs='+', default=['mint', 'pool'])
    run.add_argument('--events', type=int, default=10000)
    run.add_argument('--seeds', type=int, nargs='+', default=[1731, 2731])
    run.add_argument('--accuracy', type=float, default=0.02)
    run.add_argument('--cores', type=int, default=4)
    clone = commands.add_parser('clone', help='Make an isolated copy of an existing compiled export')
    clone.add_argument('source', type=Path)
    clone.add_argument('destination', type=Path)
    comparison = commands.add_parser('compare', help='Compare two existing benchmark JSON reports')
    comparison.add_argument('baseline', type=Path)
    comparison.add_argument('candidate', type=Path)
    comparison.add_argument('--output', type=Path)
    summary = commands.add_parser('summarize', help='Extract metrics from an existing native run')
    summary.add_argument('process_dir', type=Path)
    summary.add_argument('run_name')
    summary.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'clone':
        result = {'clone': str(clone_export(args.source, args.destination))}
    elif args.command == 'compare':
        result = compare(args.baseline, args.candidate)
    elif args.command == 'summarize':
        result = summarize(args.process_dir.resolve(), args.run_name)
    else:
        args.workdir = args.workdir.resolve()
        args.workdir.mkdir(parents=True, exist_ok=True)
        args.repo = args.repo.resolve()
        baseline = (args.baseline_repo or args.repo).resolve()
        # Freeze every export before running the first benchmark. This also
        # avoids mixing runtime snapshots if the checkout is edited during
        # long event generation. Per-run hashes identify the exact sources.
        for process_name in args.processes:
            for mode in args.modes:
                root = args.workdir / (process_name + '_' + mode)
                new = not root.exists()
                export_process(baseline if mode == 'mint' else args.repo, root, PRESETS[process_name][0], args.cores,
                               EXPORT_COMMANDS.get(process_name, ()))
                if args.virtual_ref:
                    if not new:
                        raise RuntimeError('--virtual-ref requires fresh exports: %s' % root)
                    freeze_virtual_sources(args.repo, root, args.virtual_ref)
        comparisons = []
        for process_name in args.processes:
            process, settings = PRESETS[process_name]
            reports = {}
            for mode in args.modes:
                root = args.workdir / (process_name + '_' + mode)
                export_process(baseline if mode == 'mint' else args.repo, root, process, args.cores,
                               EXPORT_COMMANDS.get(process_name, ()))
                for seed in args.seeds:
                    reports[mode, seed] = run_one(root, mode, args.events, seed, args.accuracy, args.cores, settings)
            if 'mint' in args.modes and 'pool' in args.modes:
                for seed in args.seeds:
                    result = compare(reports['mint', seed], reports['pool', seed])
                    path = args.workdir / ('%s_compare_%d.json' % (process_name, seed))
                    path.write_text(json.dumps(result, indent=2) + '\n')
                    comparisons.append(str(path))
        result = {'comparison_reports': comparisons}
    serialized = json.dumps(result, indent=2) + '\n'
    if getattr(args, 'output', None):
        args.output.write_text(serialized)
    else:
        print(serialized)
if __name__ == '__main__':
    main()
