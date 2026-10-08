#!/usr/bin/env python3
"""Read-only benchmark analysis for independent, freshly generated MC@NLO outputs.

Usage: analyze_pools.py --mint PATH --ampli PATH --output report.json
MINT rates use each unsplit stage-1 channel once. AmpliCol rates use the frozen
initial fixed-budget production moments recorded in ampli_production.json.
All production workers, including top-ups, count toward cost and efficiency.
"""
import argparse
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?'
FATAL = re.compile(r'AmpliCol production envelope exceeded|AmpliCol production budget exhausted|ERROR STOP|\bSTOP\s+[1-9]\d*\b|Traceback \(most recent call last\)|Segmentation fault|Fatal error', re.I)


def number(s):
    return float(s.replace('D', 'e').replace('d', 'e'))


def read(path):
    return path.read_text(errors='replace') if path.exists() else ''


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_run_card(path):
    result = {}
    for line in read(path).splitlines():
        s = line.split('!', 1)[0].split('#', 1)[0].strip()
        if '=' in s:
            value, key = s.split('=', 1)
            result[key.strip().lower()] = value.strip()
    return result


def parse_fks_card(path):
    result = {}
    name = None
    for line in read(path).splitlines():
        line = line.split('!', 1)[0].strip()
        if line.startswith('#'):
            name = line[1:].strip()
            result[name] = []
        elif line and name:
            result[name].append(line)
    return result


def parse_result(path):
    try:
        data = path.read_text().split()
        return dict(zip(('absolute_pb', 'absolute_error_pb', 'signed_pb', 'signed_error_pb', 'iterations', 'points_or_quota', 'cpu_seconds'), [number(v) for v in data[:7]])) if len(data) >= 7 else None
    except (OSError, ValueError):
        return None


def parse_lhe(path, event_norm, absolute_rate=None):
    out = {'path': str(path), 'events': 0, 'positive': 0, 'negative': 0, 'zero': 0, 'all_weights_finite': True,
           'sum_weights': 0.0, 'sum_squared_weights': 0.0, 'sum_absolute_weights': 0.0,
           'minimum_absolute_weight': None, 'maximum_absolute_weight': None, 'closed_document': False, 'truncated_event_header': False, 'init': [], 'negative_absolute_weight': 0., 'rwgt_blocks': 0, 'mgrwgt_blocks': 0}
    opening = gzip.open if path.suffix == '.gz' else open
    expect_header = False
    in_init = False
    init_header = False
    try:
        with opening(path, 'rt', errors='replace') as f:
            for line in f:
                stripped = line.strip()
                if stripped == '<init>':
                    in_init, init_header = True, True
                    continue
                if stripped == '</init>':
                    in_init = False
                    continue
                if in_init and stripped and not stripped.startswith('#'):
                    fields = stripped.split()
                    if init_header:
                        out['idwtup'] = int(fields[8])
                        out['init_processes'] = int(fields[9])
                        init_header = False
                    else:
                        out['init'].append(dict(signed_pb=number(fields[0]), signed_error_pb=number(fields[1]),
                                                maximum_weight=number(fields[2]), process=int(fields[3])))
                    continue
                out['rwgt_blocks'] += int('<rwgt>' in line)
                out['mgrwgt_blocks'] += int('<mgrwgt>' in line)
                if re.match(r'<event(?:\s[^>]*)?>', stripped):
                    expect_header = True
                    continue
                if expect_header and stripped and not stripped.startswith('#'):
                    try:
                        w = number(stripped.split()[2])
                    except (ValueError, IndexError):
                        out['truncated_event_header'] = True
                        expect_header = False
                        continue
                    expect_header = False
                    out['events'] += 1
                    out['positive' if w > 0 else 'negative' if w < 0 else 'zero'] += 1
                    out['all_weights_finite'] &= math.isfinite(w)
                    if not math.isfinite(w):
                        continue
                    out['sum_weights'] += w
                    out['sum_squared_weights'] += w*w
                    out['sum_absolute_weights'] += abs(w)
                    if w < 0: out['negative_absolute_weight'] += abs(w)
                    out['minimum_absolute_weight'] = abs(w) if out['minimum_absolute_weight'] is None else min(abs(w), out['minimum_absolute_weight'])
                    out['maximum_absolute_weight'] = abs(w) if out['maximum_absolute_weight'] is None else max(abs(w), out['maximum_absolute_weight'])
                if '</LesHouchesEvents>' in line:
                    out['closed_document'] = True
    except (OSError, EOFError) as exc:
        out['read_error'] = str(exc)
    out['truncated_event_header'] |= expect_header
    n = out['events']
    if n:
        mean_sign = (out['positive'] - out['negative']) / n
        out.update(positive_fraction=out['positive']/n, negative_fraction=out['negative']/n, mean_sign=mean_sign,
                   sign_effective_events=n*mean_sign**2,
                   signed_weight_effective_events=(out['sum_weights']**2/out['sum_squared_weights'] if out['sum_squared_weights'] else 0.0))
        out['absolute_weight_effective_events'] = (out['sum_absolute_weights']**2/out['sum_squared_weights']
                                                   if out['sum_squared_weights'] else 0.)
        out['weighted_negative_fraction'] = (out['negative_absolute_weight']/out['sum_absolute_weights']
                                             if out['sum_absolute_weights'] else 0.)
        sample_variance = max(0., out['sum_squared_weights']-out['sum_weights']**2/n)/max(n-1, 1)
        error_mean = math.sqrt(sample_variance/n)
        if absolute_rate is not None:

            out['rate_from_signs_pb'] = absolute_rate*mean_sign
            out['sign_sampling_error_pb'] = absolute_rate*math.sqrt(max(0.0, 1-mean_sign**2)/n)
            out['sign_error_note'] = 'Binomial sign fluctuation conditional on the integrated absolute normalization; excludes its integration uncertainty.'
        if event_norm == 'average':
            out['rate_from_weights_pb'] = out['sum_weights']/n
            out['weighted_event_sampling_error_pb'] = error_mean
        elif event_norm == 'sum':
            out['rate_from_weights_pb'] = out['sum_weights']
            out['weighted_event_sampling_error_pb'] = error_mean*n
        elif event_norm == 'unity':
            out['rate_from_weights_pb'] = absolute_rate*out['sum_weights']/n if absolute_rate else None
            out['weighted_event_sampling_error_pb'] = absolute_rate*error_mean if absolute_rate else None
    return out


def analyze(path, backend, expected, run_name):
    path = path.resolve()
    sp = path/'SubProcesses'
    if not sp.is_dir():
        raise ValueError(f'{path} is not an MG5 process directory')
    card = parse_run_card(path/'Cards/run_card.dat')
    out = {'path': str(path), 'backend': backend, 'run_name': run_name, 'run_card': card, 'fks_card': parse_fks_card(path/'Cards/FKS_params.dat'), 'cards_sha256': {}, 'stage_cpu_seconds': {},
           'stage_cpu_complete_workers': {}, 'stage_wall_seconds_estimates': {}, 'stage_logs': {}, 'integration_channels': [], 'production_workers': [], 'fatal_logs': [], 'warnings': []}
    out['selected_subprocesses'] = read(sp/'subproc.mg').split()
    out['export_process_commands'] = [line.strip() for line in read(path/'Cards/proc_card_mg5.dat').splitlines()
                                       if line.strip().startswith(('generate ', 'add process '))]
    for c in ('run_card.dat', 'param_card.dat', 'FKS_params.dat'):
        p=path/'Cards'/c
        if p.exists():
            out['cards_sha256'][c] = sha256(p)
    # Independent stages have separate logs. Split production directories may
    # link res_1.dat; stage-1 logs identify actual independent integration jobs.
    for stage in range(3):
        logs = sorted(p for p in sp.glob(f'P*/G*/log_MINT{stage}.txt') if not p.is_symlink())
        out['stage_logs'][str(stage)] = len(logs)
        seconds = 0.0
        completed = 0
        wall_records = []
        for log in logs:
            text = read(log)
            result = parse_result(log.parent/f'res_{stage}.dat')
            wall = re.findall(r'Time in seconds:\s*(\d+)', text)
            wall = int(wall[-1]) if wall else None
            if wall is not None:
                end = log.stat().st_mtime
                wall_records.append((end-wall, end, wall))
            cpu = re.findall(r'Time spent in Total\s*:\s*('+NUMBER+')', text)
            if cpu:
                seconds += number(cpu[-1]); completed += 1
            elif result:
                seconds += result['cpu_seconds']; completed += 1
            failures = [line.strip() for line in text.splitlines() if FATAL.search(line)]
            if failures:
                out['fatal_logs'].append({'log': str(log.relative_to(path)), 'lines': failures})
            if stage == 1:
                if result:
                    result['channel'] = str(log.parent.relative_to(sp))
                    result['result_path'] = str((log.parent/'res_1.dat').relative_to(path))
                    out['integration_channels'].append(result)
                else:
                    out['warnings'].append(f'No complete stage-1 result: {log.parent.relative_to(sp)}')
            if stage == 2:
                worker = {'channel': str(log.parent.relative_to(sp)), 'log': str(log.relative_to(path)),
                          'cpu_seconds': number(cpu[-1]) if cpu else result['cpu_seconds'] if result else None,
                          'wall_seconds': wall, 'trials': None, 'generated_events': None, 'quota': None,
                          'upper_bound_failures': 0, 'fatal': bool(failures)}
                quotas = re.findall(r'Generating\s+(\d+)\s+events', text)
                if quotas: worker['quota'] = int(quotas[-1])
                if backend == 'ampli':
                    stats = re.findall(r'AmpliCol production trials, candidates, retention cutoff:\s*(\d+)\s+(\d+)\s*('+NUMBER+')', text)
                    if stats:
                        t, c, cutoff = stats[-1]
                        worker.update(trials=int(t), generated_events=int(c), retention_cutoff=number(cutoff))
                    worker['envelope_exceeded'] = bool(re.search(r'production envelope exceeded', text, re.I))
                else:
                    trials = re.findall(r'another call to the function:\s*(\d+)', text)
                    counts = re.findall(r'events generated, (?:novi|virt|born):\s*(\d+)', text)
                    if trials: worker['trials'] = int(trials[-1])
                    if counts: worker['generated_events'] = sum(map(int, counts))
                    worker['upper_bound_failures'] = sum(map(int, re.findall(r'upper bound failure, (?:novi|virt|born):\s*(\d+)', text)))
                    effs = re.findall(r'Generation efficiencies:\s*('+NUMBER+r')\s*('+NUMBER+')', text)
                    if effs:
                        worker['nonvirtual_efficiency'], worker['virtual_efficiency'] = map(number, effs[-1])
                if worker['trials'] and worker['generated_events'] is not None:
                    worker['efficiency'] = worker['generated_events']/worker['trials']
                out['production_workers'].append(worker)
        out['stage_cpu_seconds'][str(stage)] = seconds
        out['stage_cpu_complete_workers'][str(stage)] = completed
        out['stage_wall_seconds_estimates'][str(stage)] = {'workers_with_wall_counters': len(wall_records),
            'sum_worker_seconds': sum(w[2] for w in wall_records),
            'phase_span_seconds': (max(w[1] for w in wall_records)-min(w[0] for w in wall_records)) if wall_records else None,
            'note': 'Span inferred from completed worker log modification times and integer wall durations; valid only when logs have not subsequently been copied/touched. In-progress workers are excluded.'}
    channels = out['integration_channels']
    if channels:
        out['integration'] = {'signed_pb': sum(x['signed_pb'] for x in channels),
                              'signed_error_pb': math.sqrt(sum(x['signed_error_pb']**2 for x in channels)),
                              'absolute_pb': sum(x['absolute_pb'] for x in channels),
                              'absolute_error_pb': math.sqrt(sum(x['absolute_error_pb']**2 for x in channels)),
                              'channels': len(channels)}
        sig = out['integration']['signed_pb']
        out['integration']['relative_signed_error'] = out['integration']['signed_error_pb']/abs(sig) if sig else None
        expected_channels = {str(p.parent.relative_to(sp)) for p in sp.glob('P*/G*/log_MINT0.txt') if not p.is_symlink()}
        expected_channels.update(str(p.parent.relative_to(sp)) for p in sp.glob('P*/G*/res_0.dat') if not p.is_symlink())
        completed_channels = {x['channel'] for x in channels}
        out['integration']['expected_channels'] = len(expected_channels)
        out['integration']['complete'] = bool(expected_channels) and completed_channels == expected_channels
    if backend == 'ampli' and (sp/'ampli_production.json').exists():
        manifest = json.loads((sp/'ampli_production.json').read_text())
        out['production_manifest'] = manifest
        out['survey'] = out.get('integration')
        frozen = manifest['channels']
        out['integration'] = dict(signed_pb=manifest['cross_section'],
            signed_error_pb=manifest['uncertainty'], absolute_pb=manifest['absolute_cross_section'],
            absolute_error_pb=math.sqrt(sum(c['error_abs']**2 for c in frozen)),
            channels=len(frozen), expected_channels=len(frozen), complete=True,
            relative_signed_error=manifest['uncertainty']/abs(manifest['cross_section']),
            estimator='Frozen initial fixed-budget production batches; top-up moments excluded',
            rate_estimation_trials=sum(c['rate_moments']['trials'] for c in frozen if c.get('rate_moments')))
    out['integration_complete'] = out.get('integration', {}).get('complete', False)
    workers = out['production_workers']
    known = [w for w in workers if w['trials'] is not None and w['generated_events'] is not None]
    trials = sum(w['trials'] for w in known)
    accepted = sum(w['generated_events'] for w in known)
    cpu = sum(w['cpu_seconds'] for w in known if w['cpu_seconds'] is not None)
    out['production'] = {'logged_workers': len(workers), 'workers_with_trial_counters': len(known),
                         'counter_events': accepted, 'trials': trials,
                         'efficiency': accepted/trials if trials else None,
                         'events_per_cpu_second_completed_workers': accepted/cpu if cpu else None,
                         'upper_bound_failures': sum(w['upper_bound_failures'] for w in workers),
                         'ampli_envelope_failure_workers': sum(bool(w.get('envelope_exceeded')) for w in workers),
                         'all_logged_workers_finished': bool(workers) and len(known)==len(workers) and not any(w['fatal'] for w in workers)}
    out['total_cpu_seconds_completed_workers'] = sum(out['stage_cpu_seconds'].values())
    out['cpu_note'] = 'Sum of worker CPU times from phase logs/results; incomplete workers without final CPU counters are excluded. Compilation and Python coordination are excluded.'
    assignments = []
    for line in read(sp/'nevents_unweighted').splitlines():
        fields = line.split()
        if len(fields)>=4:
            try:
                assignments.append({'file': fields[0], 'quota': int(fields[1]), 'allocated_absolute_pb': number(fields[2]), 'weight_fraction': number(fields[3])})
            except ValueError:
                pass
    out['assignments'] = assignments
    out['assigned_events'] = sum(a['quota'] for a in assignments)
    out['production']['all_assigned_workers_finished'] = bool(assignments) and all(any(w['channel']==str(Path(a['file']).parent) and w['generated_events']==a['quota'] and not w['fatal'] for w in workers) for a in assignments if a['quota']>0)
    event_files = [p for p in (path/'Events'/run_name/'events.lhe.gz', path/'Events'/run_name/'events.lhe') if p.is_file()]
    if event_files:
        latest = max(event_files, key=lambda p:p.stat().st_mtime_ns)
        out['lhe'] = parse_lhe(latest, card.get('event_norm','average').lower(), out.get('integration',{}).get('absolute_pb'))
        out['summary'] = read(latest.parent/'summary.txt')
        out['complete_expected_sample'] = out['lhe']['events']==expected and out['lhe']['closed_document'] and not out['lhe']['truncated_event_header'] and out['lhe']['all_weights_finite'] and not out['fatal_logs']
        if len(event_files)>1:
            out['warnings'].append('Both compressed and uncompressed event files found for the specified run; report uses most recently modified file.')
    else:
        out['lhe'] = None
        out['complete_expected_sample'] = False
        # Retain evidence of partial generation, but never treat candidate or
        # worker events as a final physical sample after an envelope failure.
        partials = []
        for p in sorted(sp.glob('P*/G*/*.lhe')):
            if p.name not in ('events.lhe','ampli_candidates.lhe'):
                continue
            sample = parse_lhe(p, card.get('event_norm','average').lower())
            partials.append({'path': str(p.relative_to(path)), 'events': sample['events'], 'closed_document': sample['closed_document']})
        out['partial_worker_files'] = partials
    production = out['production']
    production['stored_candidates'] = accepted
    production['candidate_retention_efficiency'] = accepted/trials if trials else None
    production['cpu_seconds'] = out['stage_cpu_seconds']['2']
    production['survey_cpu_seconds'] = sum(out['stage_cpu_seconds'][str(stage)] for stage in (0, 1))
    production['final_events'] = out['lhe']['events'] if out['complete_expected_sample'] else 0
    production['efficiency'] = production['final_events']/trials if trials and out['complete_expected_sample'] else None
    production['events_per_cpu_second_completed_workers'] = production['final_events']/production['cpu_seconds'] if out['complete_expected_sample'] and production['cpu_seconds'] else None
    production['events_per_total_cpu_second'] = production['final_events']/out['total_cpu_seconds_completed_workers'] if out['complete_expected_sample'] and out['total_cpu_seconds_completed_workers'] else None
    if 'production_manifest' in out:
        batches = [batch for channel in out['production_manifest']['channels'] for batch in channel['batches']]
        worker_index = {worker['channel']: worker for worker in out['production_workers']}
        initial = [batch for batch in batches if batch['include_rate']]
        topups = [batch for batch in batches if not batch['include_rate']]
        production['initial_rate_trials'] = sum(batch['trials'] for batch in initial)
        production['topup_trials'] = sum(batch['trials'] for batch in topups)
        production['initial_rate_cpu_seconds'] = sum(worker_index[str(Path(batch['directory']).relative_to('SubProcesses'))]['cpu_seconds'] for batch in initial)
        production['topup_cpu_seconds'] = sum(worker_index[str(Path(batch['directory']).relative_to('SubProcesses'))]['cpu_seconds'] for batch in topups)
        production['all_assigned_workers_finished'] = bool(batches) and all(
            worker_index[str(Path(batch['directory']).relative_to('SubProcesses'))]['trials'] == batch['trials']
            for batch in batches)
    if out['complete_expected_sample']:
        lhe = out['lhe']
        lhe['event_sampling_error_note'] = 'Sample variance of finalized signed event weights; conditional on the global absolute rate and neglecting finite-pool correlations. Not an independent integration uncertainty.'
        lhe['rate_minus_integration_pb'] = lhe['rate_from_weights_pb']-out['integration']['signed_pb']
        lhe['diagnostic_pull'] = lhe['rate_minus_integration_pb']/math.hypot(lhe['weighted_event_sampling_error_pb'], out['integration']['signed_error_pb'])
    source_paths = [
        'SubProcesses/mint_module.f90', 'SubProcesses/ampli_mint_adapter.f90',
        'SubProcesses/simple_integrator.f90', 'SubProcesses/integrator_helpers.f90',
        'SubProcesses/driver_mintMC.f', 'SubProcesses/genps_fks_radiation.f',
        'bin/internal/amcatnlo_run_interface.py', 'bin/internal/ampli_pool.py']
    for subprocess in out['selected_subprocesses']:
        source_paths.extend('SubProcesses/%s/%s' % (subprocess, name)
                            for name in ('fks_info.inc', 'born_support.json', 'born_support.f'))
        source_paths.extend(str(p.relative_to(path)) for p in sorted((sp/subprocess).glob('parton_lum_*.f')))
    out['exported_source_sha256'] = {name: sha256(path/name) for name in source_paths if (path/name).is_file()}
    metadata = {}
    for folder in (path, path.parent):
        for p in sorted(folder.glob('*.json')):
            if not any(word in p.name.lower() for word in ('tim','run','benchmark','status','wrapper','started','finished')):
                continue
            try:
                data = json.loads(p.read_text())
                if len(p.read_bytes()) < 100000:
                    metadata[str(p)] = data
            except (OSError, ValueError):
                pass
    out['wrapper_metadata'] = metadata
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mint', required=True, type=Path)
    ap.add_argument('--ampli', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--expected-events', type=int, default=10000)
    ap.add_argument('--run-name', default='benchmark_10k')
    args = ap.parse_args()
    report = {'expected_events_per_backend': args.expected_events, 'run_name': args.run_name,
              'rate_scope': 'Selected (u d~ + c s~) > W+ g g Born component, with this beam ordering, of the complete p p > W+ j j [QCD] FKS partition; not the inclusive physical W+2-jet cross section.',
              'mint': analyze(args.mint,'mint',args.expected_events,args.run_name), 'ampli': analyze(args.ampli,'ampli',args.expected_events,args.run_name), 'comparison': {}}
    m,a=report['mint'], report['ampli']
    comp=report['comparison']
    comp['both_integrations_complete'] = m['integration_complete'] and a['integration_complete']
    if comp['both_integrations_complete']:
        mi,ai=m['integration'],a['integration']
        err=math.hypot(mi['signed_error_pb'],ai['signed_error_pb'])
        delta=ai['signed_pb']-mi['signed_pb']
        comp.update(ampli_minus_mint_pb=delta, combined_signed_integration_error_pb=err, integration_pull=delta/err if err else None,
                    uncertainty_ratio_ampli_over_mint=ai['signed_error_pb']/mi['signed_error_pb'] if mi['signed_error_pb'] else None)
    for metric in ('efficiency','events_per_cpu_second_completed_workers', 'events_per_total_cpu_second'):
        mv,av=m['production'][metric],a['production'][metric]
        if mv and av is not None:
            comp[metric+'_ratio_ampli_over_mint']=av/mv
    comp['both_samples_complete']=m['complete_expected_sample'] and a['complete_expected_sample']
    comp['interpretation']='Integration errors are Monte Carlo integration uncertainties, not scale/PDF theory uncertainties. Final-event efficiency uses finalized event count divided by all attempted folded phase-space points, including top-ups. Candidate retention is reported separately. AmpliCol final rates use initial fixed-budget production moments, while MINT rates use stage-1 integration. Ratios from incomplete worker subsets are diagnostic only.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(args.output),'comparison':comp,'mint_complete':m['complete_expected_sample'],'ampli_complete':a['complete_expected_sample']},indent=2))

if __name__=='__main__': main()
