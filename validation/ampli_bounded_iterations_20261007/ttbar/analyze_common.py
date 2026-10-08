#!/usr/bin/env python3
"""Read-only MINT versus native AmpliCol MC@NLO benchmark analysis.

Usage: analyze.py --mint PATH --ampli PATH --output report.json
MINT rates use each unsplit stage-1 channel once. Native AmpliCol rates combine
the survey once with all production iterations, including fresh top-up workers.
Efficiencies use finalized events / all folded observations, including zeros.
No candidate spool is mistaken for a completed physical event sample.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import re
import random
import sys
from pathlib import Path

NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?'
FATAL = re.compile(r'AmpliCol production envelope exceeded|AmpliCol production budget exhausted|AmpliCol generation safety limit|ERROR STOP|\bSTOP\s+[1-9]\d*\b|Traceback \(most recent call last\)|Segmentation fault|Fatal error', re.I)


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
    result, key = {}, None
    for line in read(path).splitlines():
        text = line.split('!', 1)[0].strip()
        if text.startswith('#'):
            key = text[1:].strip()
        elif text and key:
            result[key], key = text, None
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
        out.update(negative_fraction=out['negative']/n, mean_sign=mean_sign,
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


def assert_close(actual, expected, label, relative=2.e-9):
    if not math.isclose(actual, expected, rel_tol=relative, abs_tol=2.e-11):
        raise ValueError('%s mismatch: %r != %r' % (label, actual, expected))


def combine_native_rates(survey, pools):
    """Independent replay of upstream trial-weighted iteration combination."""
    count = int(survey['points_or_quota'])
    means = [survey['absolute_pb'], survey['signed_pb']]
    variances = [survey['absolute_error_pb']**2, survey['signed_error_pb']**2]
    for pool in pools:
        for epoch in pool['epochs']:
            n = epoch['trials']
            for i, name in enumerate(('abs', 'signed')):
                value = epoch['mean_'+name]
                variances[i] = ((count**2*variances[i]+epoch['m2_'+name])/(count+n)**2
                    + count*n*(means[i]-value)**2/(count+n)**3)
                means[i] = (count*means[i]+n*value)/(count+n)
            count += n
    return dict(absolute=means[0], signed=means[1], error_abs=math.sqrt(variances[0]),
                error_signed=math.sqrt(variances[1]), trials=count)


def allocate(channels, field, nevents, seed):
    ordered = sorted(channels, key=lambda c: c['allocation_order'])
    assert [c['allocation_order'] for c in ordered] == list(range(len(ordered)))
    positive = [c for c in ordered if c[field] > 0.]
    total = math.fsum(c[field] for c in positive)
    result = {(c['subprocess'], c['channel']): 0 for c in channels}
    rng = random.Random(seed ^ 0x414D504C)
    for unused in range(nevents):
        target, cumulative = rng.random()*total, 0.
        for c in positive:
            cumulative += c[field]
            if target < cumulative:
                break
        result[c['subprocess'], c['channel']] += 1
    return result


def validate_native(path, out, pool_source=None):
    """Replay pool validation/collection and independently recompute native rates.

    The exact exported collector validates raw POOL4 metadata and replays the
    seeded collection. The rate combination here is separate from production.
    LHE magnitudes are checked against sidecar factors without candidate spools.
    """
    if pool_source is None:
        candidates = [path/'bin/internal/ampli_pool.py',
                      Path(__file__).resolve().parents[2]/'madgraph/various/ampli_pool.py']
        pool_source = next(p for p in candidates if p.is_file())
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('benchmark_ampli_pool', pool_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = out['production_manifest']
    assert manifest['version'] == 4
    assert manifest['rate_source'] == 'survey_plus_production'
    assert manifest['sampling'] == 'native_adaptive_unfolded'
    assert manifest['allowed_overweight_factor'] == .01
    nevents = manifest['requested_events']
    channels = manifest['channels']
    assert sum(c['quota'] for c in channels) == nevents
    assert sum(c['initial_quota'] for c in channels) == nevents
    initial = allocate(channels, 'survey_absolute', nevents, manifest['seed'])
    final = allocate(channels, 'absolute', nevents, manifest['seed'])
    selection_rng = random.Random(manifest['seed'] ^ 0x504F4F4C)
    expected_weights, expected_tails, workers = [], [], []
    normalization = manifest['absolute_cross_section']
    event_norm = out['run_card'].get('event_norm', 'average').lower()
    if event_norm == 'sum': normalization /= nevents
    elif event_norm == 'unity': normalization = 1.
    for c in channels:
        key = c['subprocess'], c['channel']
        assert c['initial_quota'] == initial[key]
        assert c['quota'] == final[key]
        parent = path/'SubProcesses'/c['subprocess']/('GF'+c['channel'])
        survey = parse_result(parent/'res_1.dat')
        assert survey['absolute_error_pb'] <= .03*survey['absolute_pb']*(1.+1.e-10)
        pools, collection = [], []
        for batch in c['batches']:
            pool = module.read_pool(path/batch['directory'])
            assert pool['version'] == batch['pool_protocol'] == 4
            assert pool['trials'] == batch['trials']
            assert pool['generated_target'] == batch['generated_target']
            assert pool['final_quota'] == batch['nominal_final_quota']
            assert pool['generated_target'] == (11*pool['final_quota']+9)//10
            assert pool['adaptation'] == batch['adaptation']
            assert pool['adaptation']['schedule'] == 'nonzero_trials'
            folds = list(map(int, re.findall(r'\d+', out['run_card']['folding'])))
            assert len(folds) == 3
            mask = pool['adaptation']['mask']
            assert mask[-3:] == [int(fold == 1) for fold in folds]
            assert all(mask[:-3])
            tails = {k: pool[k] for k in ('full_trial_tail','reserve_tail','worst_subset_tail')}
            assert all(0. <= value < .01 for value in tails.values())
            assert_close(batch['native_log_z'], pool['log_z'], 'native rejection level')
            view = pool if batch['collection_log_z'] == pool['log_z'] else module.tighten_native_pool(pool)
            assert_close(batch['collection_log_z'], view['log_z'], 'collection rejection level')
            assert module.available_events([view]) == batch['available']
            pools.append(pool)
            collection.append(view)
            workers.append(dict(directory=batch['directory'], trials=pool['trials'],
                nonzero_trials=sum(e['nonzero'] for e in pool['epochs']),
                iterations=len(pool['epochs']), stored_candidates=pool['ncandidates'],
                reserve_target=pool['generated_target'], nominal_quota=pool['final_quota'],
                available=batch['available'], native_tails=tails,
                rethresholded=bool(view.get('collection_rethresholded')),
                adaptation=pool['adaptation']))
        rates = combine_native_rates(survey, pools)
        for key, value in rates.items():
            assert_close(c['updated_rates'][key], value, 'combined '+key)
        for key in ('absolute','signed','error_abs','error_signed'):
            assert_close(c[key], rates[key], 'published '+key)
        assert sum(p['trials'] for p in pools) == c['generation_trials']
        assert module.available_events(collection) == c['available_candidates']
        if not c['quota']:
            assert c['selection'] is None
            continue
        status = module.pool_status(collection, c['quota'])
        assert status['overweight'] < .01
        selection, diagnostics = module.select_candidates(collection, c['quota'], selection_rng)
        for key, value in diagnostics.items():
            assert_close(c['selection'][key], value, 'collection '+key)
        for (pool_index, candidate_index), correction in selection.items():
            row = collection[pool_index]['candidates'][candidate_index]
            expected_weights.append(row[4]*correction*normalization)
            expected_tails.append(bool(row[3]))
    assert_close(manifest['cross_section'], sum(c['signed'] for c in channels), 'total signed')
    assert_close(manifest['absolute_cross_section'], sum(c['absolute'] for c in channels), 'total absolute')
    assert_close(manifest['uncertainty'], math.sqrt(sum(c['error_signed']**2 for c in channels)), 'total error')
    assert sum(w['trials'] for w in workers) == manifest['generation_trials'] == out['production']['trials']
    assert sum(w['stored_candidates'] for w in workers) == out['production']['counter_events']
    assert_close(manifest['generation_cpu_seconds'], out['production']['cpu_seconds'], 'generation CPU')
    final_path = Path(out['lhe']['path'])
    opening = gzip.open if final_path.suffix == '.gz' else open
    with opening(final_path, 'rt') as stream:
        text = stream.read()
    events = re.findall(r'<event(?:[ \t][^>]*)?>(.*?)</event>', text, re.S)
    weights = [number(event.strip().splitlines()[0].split()[2]) for event in events]
    assert len(weights) == len(expected_weights) == nevents
    for actual, expected in zip(sorted(map(abs, weights)), sorted(expected_weights)):
        assert_close(actual, expected, 'final LHE magnitude', relative=5.e-7)
    collected_tail = math.fsum(w for w, flag in zip(expected_weights, expected_tails) if flag)/math.fsum(expected_weights)
    assert collected_tail < .01
    assert out['lhe']['idwtup'] == -4
    assert_close(sum(p['signed_pb'] for p in out['lhe']['init']), manifest['cross_section'], 'LHE init rate', relative=5.e-8)
    assert_close(math.sqrt(sum(p['signed_error_pb']**2 for p in out['lhe']['init'])), manifest['uncertainty'], 'LHE init error', relative=5.e-8)
    return dict(collector_source=str(pool_source), collector_sha256=sha256(pool_source),
        rates_recomputed_from_survey_once_and_all_iterations=True,
        initial_and_updated_quotas_reproduced=True, deterministic_collection_reproduced=True,
        final_lhe_magnitudes_reproduced=True, all_native_and_collected_tail_fractions_below_one_percent=True,
        collected_tail_fraction=collected_tail,
        maximum_native_tail_fraction=max((max(w['native_tails'].values()) for w in workers), default=0.),
        maximum_final_channel_tail_bound=max((c['selection']['overweight'] for c in channels if c['selection']),default=0.),
        workers=workers, production_counters=dict(
            native_iterations=sum(w['iterations'] for w in workers),
            grid_updates=sum(w['adaptation']['updates'] for w in workers),
            nonzero_trials=sum(w['nonzero_trials'] for w in workers),
            zero_trials=sum(w['trials']-w['nonzero_trials'] for w in workers),
            stored_candidates=sum(w['stored_candidates'] for w in workers),
            native_reserve_target=sum(w['reserve_target'] for w in workers),
            total_nominal_worker_quota=sum(w['nominal_quota'] for w in workers),
            collection_available=sum(w['available'] for w in workers),
            collection_rounds=len(manifest['rounds']),
            rethresholded_workers=sum(w['rethresholded'] for w in workers),
            max_channel_quota_change=max((abs(c['quota']-c['initial_quota']) for c in channels),default=0.)))


def analyze(path, backend, expected, run_name=None, pool_source=None):
    path = path.resolve()
    sp = path/'SubProcesses'
    if not sp.is_dir():
        raise ValueError(f'{path} is not an MG5 process directory')
    card = parse_run_card(path/'Cards/run_card.dat')
    out = {'path': str(path), 'backend': backend, 'run_card': card, 'fks_card': parse_fks_card(path/'Cards/FKS_params.dat'), 'cards_sha256': {}, 'stage_cpu_seconds': {},
           'stage_cpu_complete_workers': {}, 'stage_wall_seconds_estimates': {}, 'stage_logs': {}, 'integration_channels': [], 'production_workers': [], 'fatal_logs': [], 'warnings': []}
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
                    stats = re.findall(r'AmpliCol generation trials, candidates, requested events:\s*(\d+)\s+(\d+)\s+(\d+)', text)
                    if stats:
                        t, c, target = stats[-1]
                        worker.update(trials=int(t), generated_events=int(c), reserve_target=int(target))
                    worker['envelope_exceeded'] = bool(re.search(r'production envelope exceeded', text, re.I))
                else:
                    trials = re.findall(r'another call to the function:\s*(\d+)', text)
                    counts = re.findall(r'events generated, (?:novi|virt|born):\s*(\d+)', text)
                    if trials: worker['trials'] = int(trials[-1])
                    if counts: worker['generated_events'] = sum(map(int, counts))
                    worker['zero_trials'] = sum(map(int, re.findall(r'failed generation cuts:\s*(\d+)', text)))
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
    manifest_path = path/'Events'/run_name/'ampli_production.json' if run_name else sp/'ampli_production.json'
    if backend == 'ampli' and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        out['production_manifest'] = manifest
        out['survey'] = out.get('integration')
        channels_final = manifest['channels']
        out['integration'] = dict(signed_pb=manifest['cross_section'],
            signed_error_pb=manifest['uncertainty'], absolute_pb=manifest['absolute_cross_section'],
            absolute_error_pb=math.sqrt(sum(c['error_abs']**2 for c in channels_final)),
            channels=len(channels_final), expected_channels=len(channels_final), complete=True,
            relative_signed_error=manifest['uncertainty']/abs(manifest['cross_section']),
            estimator='Survey once plus all native production iterations, including top-ups',
            rate_estimation_trials=sum(c['updated_rates']['trials'] for c in channels_final))
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
                         'zero_trials': sum(w.get('zero_trials', 0) for w in workers) if backend == 'mint' else None,
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
    event_directory = path/'Events'/run_name if run_name else path/'Events'
    pattern = 'events.lhe*' if run_name else '*/events.lhe*'
    event_files = [p for p in event_directory.glob(pattern) if p.name in ('events.lhe', 'events.lhe.gz')]
    if event_files:
        latest = max(event_files, key=lambda p:p.stat().st_mtime_ns)
        out['lhe'] = parse_lhe(latest, card.get('event_norm','average').lower(), out.get('integration',{}).get('absolute_pb'))
        out['lhe']['sha256'] = sha256(latest)
        out['summary'] = read(latest.parent/'summary.txt')
        out['complete_expected_sample'] = out['lhe']['events']==expected and out['lhe']['closed_document'] and not out['lhe']['truncated_event_header'] and out['lhe']['all_weights_finite'] and not out['fatal_logs']
        if len(event_files)>1:
            out['warnings'].append('Multiple event files found; report uses most recently modified file. Phase results must correspond to that run.')
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
        production['all_assigned_workers_finished'] = bool(batches) and all(
            worker_index.get(str(Path(batch['directory']).relative_to('SubProcesses')), {}).get('trials') == batch['trials']
            for batch in batches)
        if out['complete_expected_sample']:
            out['native_validation'] = validate_native(path, out, pool_source)
            production.update(out['native_validation']['production_counters'])
    if out['complete_expected_sample']:
        lhe = out['lhe']
        lhe['event_sampling_error_note'] = 'Sample variance of finalized signed event weights; conditional on the global absolute rate and neglecting finite-pool correlations. Not an independent integration uncertainty.'
        lhe['rate_minus_integration_pb'] = lhe['rate_from_weights_pb']-out['integration']['signed_pb']
        denominator = math.hypot(lhe['weighted_event_sampling_error_pb'], out['integration']['signed_error_pb'])
        lhe['diagnostic_pull'] = lhe['rate_minus_integration_pb']/denominator if denominator else None
        for name in ('signed_weight_effective_events', 'absolute_weight_effective_events'):
            lhe[name+'_per_generation_cpu_second'] = lhe[name]/production['cpu_seconds'] if production['cpu_seconds'] else None
            lhe[name+'_per_total_cpu_second'] = lhe[name]/out['total_cpu_seconds_completed_workers'] if out['total_cpu_seconds_completed_workers'] else None
    source_paths = [
        'SubProcesses/mint_module.f90', 'SubProcesses/ampli_mint_adapter.f90',
        'SubProcesses/simple_integrator.f90', 'SubProcesses/integrator_helpers.f90',
        'SubProcesses/driver_mintMC.f', 'SubProcesses/genps_fks_radiation.f',
        'bin/internal/amcatnlo_run_interface.py', 'bin/internal/ampli_pool.py']
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
    started = path.parent/(backend+'_started.json')
    finished = path.parent/(backend+'_finished.json')
    if started.exists(): out['wrapper_started'] = json.loads(started.read_text())
    if finished.exists(): out['wrapper_finished'] = json.loads(finished.read_text())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', type=Path, help='Parent of mint/ and ampli/ process directories')
    ap.add_argument('--mint', type=Path)
    ap.add_argument('--ampli', type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--expected-events', type=int, default=10000)
    ap.add_argument('--run-name', help='Explicit Events subdirectory used for both backends')
    ap.add_argument('--pool-source', type=Path, help='Exact ampli_pool.py used by the export; defaults to exported source or repository source')
    args = ap.parse_args()
    if args.work:
        args.mint = args.mint or args.work/'mint'
        args.ampli = args.ampli or args.work/'ampli'
    if not args.mint or not args.ampli:
        ap.error('Supply --work or both --mint and --ampli')
    report = {'expected_events_per_backend': args.expected_events, 'mint': analyze(args.mint,'mint',args.expected_events,args.run_name,args.pool_source), 'ampli': analyze(args.ampli,'ampli',args.expected_events,args.run_name,args.pool_source), 'comparison': {}}
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
    comp['run_card_differences'] = {key: {'mint': m['run_card'].get(key), 'ampli': a['run_card'].get(key)}
        for key in sorted(set(m['run_card']) | set(a['run_card']))
        if m['run_card'].get(key) != a['run_card'].get(key)}
    comp['fks_card_differences'] = {key: {'mint': m['fks_card'].get(key), 'ampli': a['fks_card'].get(key)}
        for key in sorted(set(m['fks_card']) | set(a['fks_card']))
        if m['fks_card'].get(key) != a['fks_card'].get(key)}
    comp['param_cards_identical'] = m['cards_sha256'].get('param_card.dat') == a['cards_sha256'].get('param_card.dat')
    comp['exported_sources_identical'] = m['exported_source_sha256'] == a['exported_source_sha256']
    if comp['both_samples_complete']:
        for backend in (m, a):
            backend['whole_launch_wall_seconds'] = backend.get('wrapper_finished', {}).get('wall_seconds')
        if m['whole_launch_wall_seconds'] and a['whole_launch_wall_seconds']:
            comp['whole_launch_wall_speedup_ampli'] = m['whole_launch_wall_seconds']/a['whole_launch_wall_seconds']
        comp['generation_cpu_speedup_ampli'] = m['production']['cpu_seconds']/a['production']['cpu_seconds']
        comp['total_worker_cpu_speedup_ampli'] = m['total_cpu_seconds_completed_workers']/a['total_cpu_seconds_completed_workers']
        for metric in ('signed_weight_effective_events', 'absolute_weight_effective_events'):
            for denominator in ('generation', 'total'):
                key = metric+'_per_'+denominator+'_cpu_second'
                comp[key+'_ratio_ampli_over_mint'] = a['lhe'][key]/m['lhe'][key]

    comp['interpretation']='Integration errors are Monte Carlo integration uncertainties, not scale/PDF theory uncertainties. Final-event efficiency uses finalized event count divided by all attempted folded phase-space points, including top-ups. Candidate retention is reported separately. AmpliCol final rates combine its 3% per-channel survey and all native production iterations, while MINT rates use stage-1 integration. Nonzero-count stopping and adaptation mean the native error is a conventional Monte Carlo estimate, not an exactly unbiased finite-sample uncertainty. MINT bound exceedance counts do not measure full overweight tail mass. Ratios from incomplete worker subsets are diagnostic only.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(args.output),'comparison':comp,'mint_complete':m['complete_expected_sample'],'ampli_complete':a['complete_expected_sample']},indent=2))

if __name__=='__main__': main()
