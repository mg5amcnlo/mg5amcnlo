#!/usr/bin/env python3
"""Verify completed MINT/AmpliCol samples and the per-channel overweight limit."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def event_digest(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    digest = hashlib.sha256()
    count = 0
    inside = False
    with opener(path, 'rb') as stream:
        for line in stream:
            stripped = line.strip()
            if stripped == b'<event>' or stripped.startswith(b'<event '):
                inside = True
                count += 1
            if inside:
                digest.update(line)
            if stripped == b'</event>':
                inside = False
    if inside:
        raise ValueError('Truncated event block in %s' % path)
    return {'events': count, 'event_blocks_sha256': digest.hexdigest()}


def raw_pool_quality(run_directory, channel):
    """Recompute the reported reserve excess directly from worker sidecars."""
    candidates = []
    cutoffs = []
    source_pools = []
    for batch in channel['batches']:
        pool_path = run_directory / batch['directory'] / 'ampli_pool.dat'
        source_pools.append({'path': str(pool_path), 'sha256': hashlib.sha256(pool_path.read_bytes()).hexdigest()})
        with pool_path.open() as stream:
            assert stream.readline().split() == ['MG5_AMPLI_POOL', '1']
            trials, count, cutoff = stream.readline().split()
            assert int(trials) == batch['trials']
            cutoffs.append(float(cutoff))
            stream.readline()  # All-point moments are checked against the initial snapshot.
            rows = [tuple(map(float, line.split())) for line in stream if line.strip()]
            assert len(rows) == int(count)
            candidates.extend(rows)
    log_cutoff = math.log(max(cutoffs)) if max(cutoffs) else -math.inf
    candidates = [row for row in candidates if row[1] >= log_cutoff]
    candidates.sort(key=lambda row: row[1], reverse=True)
    reserve = min(len(candidates), max(1000, math.ceil(1.1 * channel['quota'])))
    threshold = candidates[reserve][1] if reserve < len(candidates) else log_cutoff
    excess = math.fsum(math.expm1(max(0., math.log(weight) - threshold))
                       for weight, priority in candidates[:reserve]) / reserve
    return {'available': len(candidates), 'reserve': reserve, 'overweight': excess, 'source_pools': source_pools,
            'matches_manifest': (len(candidates) == channel['selection']['available'] and
                reserve == channel['selection']['reserve'] and math.isclose(
                    excess, channel['selection']['overweight'], rel_tol=1e-10, abs_tol=1e-12)),
            'within_one_percent': excess <= 0.01}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    checks = {'expected_events': metrics['expected_events_per_backend'], 'backends': {}}
    for backend in ('mint', 'ampli'):
        data = metrics[backend]
        lhe = data['lhe']
        entry = {
            'complete': data['complete_expected_sample'],
            'no_fatal_logs': not data['fatal_logs'],
            'integration_complete': data['integration_complete'],
            'workers_complete': data['production']['all_logged_workers_finished'],
            'assignments_complete': data['production']['all_assigned_workers_finished'],
            'assigned_events': data['assigned_events'],
            'finite_weights': bool(lhe and lhe['all_weights_finite']),
            'idwtup': lhe['idwtup'] if lhe else None,
            'event_payload': event_digest(lhe['path']) if lhe else None,
        }
        if lhe:
            entry['init_matches_rate'] = math.isclose(
                sum(row['signed_pb'] for row in lhe['init']), data['integration']['signed_pb'],
                rel_tol=1e-6, abs_tol=1e-10)
            entry['init_matches_error'] = math.isclose(
                math.sqrt(sum(row['signed_error_pb']**2 for row in lhe['init'])),
                data['integration']['signed_error_pb'], rel_tol=5e-5, abs_tol=1e-10)
            entry['init_error_tolerance_note'] = 'MINT writes the uncertainty to five significant figures; relative comparison tolerance is 5e-5.'
        checks['backends'][backend] = entry
    mint, ampli = metrics['mint'], metrics['ampli']
    checks['same_run_card'] = mint['cards_sha256']['run_card.dat'] == ampli['cards_sha256']['run_card.dat']
    mint_card, ampli_card = mint['run_card'].copy(), ampli['run_card'].copy()
    mint_accuracy = float(mint_card.pop('req_acc').replace('d', 'e').replace('D', 'e'))
    ampli_accuracy = float(ampli_card.pop('req_acc').replace('d', 'e').replace('D', 'e'))
    checks['run_cards_differ_only_in_req_acc'] = mint_card == ampli_card
    expected_ampli_accuracy = (mint_accuracy * mint['integration']['absolute_pb'] /
                               abs(mint['integration']['signed_pb']))
    checks['accuracy_policy'] = {
        'mint_req_acc': mint_accuracy,
        'ampli_req_acc': ampli_accuracy,
        'expected_ampli_req_acc': expected_ampli_accuracy,
        'reference_nominal_absolute_target_pb': mint_accuracy * mint['integration']['absolute_pb'],
        'ampli_survey_signed_pb': ampli.get('survey', {}).get('signed_pb'),
        'ampli_planner_nominal_absolute_target_pb': (
            ampli_accuracy * abs(ampli['survey']['signed_pb']) if ampli.get('survey') else None),
        'description': 'Match the requested absolute-rate accuracy: Ampli req_acc = MINT req_acc * MINT stage-1 absolute rate / abs(MINT stage-1 signed rate).',
        'matched_requested_absolute_accuracy': math.isclose(ampli_accuracy, expected_ampli_accuracy, rel_tol=1e-12),
    }
    checks['same_param_card'] = mint['cards_sha256']['param_card.dat'] == ampli['cards_sha256']['param_card.dat']
    checks['same_exported_sources'] = mint['exported_source_sha256'] == ampli['exported_source_sha256']
    checks['same_selected_subprocesses'] = mint['selected_subprocesses'] == ampli['selected_subprocesses']
    checks['selected_subprocesses'] = mint['selected_subprocesses']
    mint_fks, ampli_fks = mint['fks_card'].copy(), ampli['fks_card'].copy()
    checks['backend_switches'] = {'mint': mint_fks.pop('NLOPSIntegrator', None),
                                  'ampli': ampli_fks.pop('NLOPSIntegrator', None)}
    checks['fks_cards_differ_only_in_backend_switch'] = mint_fks == ampli_fks and checks['backend_switches'] == {'mint': ['0'], 'ampli': ['1']}
    manifest = ampli.get('production_manifest', {})
    checks['allowed_overweight_factor'] = manifest.get('allowed_overweight_factor')
    checks['channels'] = [{key: value for key, value in channel.items()
                           if key not in ('batches', 'rate_moments')}
                          for channel in manifest.get('channels', [])]
    excesses = [(channel.get('selection') or {}).get('overweight')
                for channel in checks['channels'] if channel['quota'] > 0]
    checks['zero_quota_channels'] = sum(channel['quota'] == 0 for channel in checks['channels'])
    checks['all_channels_within_one_percent'] = (
        checks['allowed_overweight_factor'] == 0.01 and bool(excesses) and
        all(value is not None and value <= 0.01 for value in excesses))
    checks['maximum_overweight_excess'] = max(excesses) if excesses and all(value is not None for value in excesses) else None
    checks['raw_pool_quality'] = {
        '%s/GF%s' % (channel['subprocess'], channel['channel']): raw_pool_quality(Path(ampli['path']), channel)
        for channel in manifest.get('channels', []) if channel['quota'] > 0}
    checks['raw_pools_confirm_overweight_enforcement'] = bool(checks['raw_pool_quality']) and all(
        row['matches_manifest'] and row['within_one_percent'] for row in checks['raw_pool_quality'].values())
    checks['initial_rate_trials'] = ampli['production'].get('initial_rate_trials')
    checks['topup_trials'] = ampli['production'].get('topup_trials')
    initial_path = args.metrics.parent / 'ampli_initial_production.json'
    if initial_path.is_file():
        initial = json.loads(initial_path.read_text())
        checks['rate_unchanged_by_topups'] = all(math.isclose(
            initial[key], ampli['integration'][key], rel_tol=1e-12, abs_tol=1e-12)
            for key in ('signed_pb', 'signed_error_pb', 'absolute_pb', 'absolute_error_pb'))
        checks['initial_trial_count_matches_recorded_snapshot'] = initial['trials'] == checks['initial_rate_trials']
    checks['all_required_checks_pass'] = (
        all(entry['complete'] and entry['no_fatal_logs'] and entry['integration_complete'] and
            entry['workers_complete'] and entry['assignments_complete'] and entry['finite_weights'] and
            entry['init_matches_rate'] and entry['init_matches_error'] and
            entry['assigned_events'] == checks['expected_events']
            for entry in checks['backends'].values()) and
        checks['backends']['ampli']['idwtup'] == -4 and checks['run_cards_differ_only_in_req_acc'] and
        checks['accuracy_policy']['matched_requested_absolute_accuracy'] and
        checks['same_param_card'] and checks['same_exported_sources'] and
        checks['same_selected_subprocesses'] and
        checks['fks_cards_differ_only_in_backend_switch'] and
        checks.get('rate_unchanged_by_topups', True) and
        checks.get('initial_trial_count_matches_recorded_snapshot', True) and
        checks['raw_pools_confirm_overweight_enforcement'] and
        checks['all_channels_within_one_percent'])
    args.output.write_text(json.dumps(checks, indent=2, allow_nan=False) + '\n')
    print(json.dumps(checks, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
