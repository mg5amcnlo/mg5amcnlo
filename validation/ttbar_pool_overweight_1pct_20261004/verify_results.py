#!/usr/bin/env python3
"""Read-only checks of the one-percent AmpliCol rerun against saved evidence."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def event_digest(path):
    """Hash exact event blocks, excluding timestamp/source-dependent banners."""
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    digest = hashlib.sha256()
    count = 0
    inside = False
    with opener(path, 'rb') as stream:
        for line in stream:
            if line.strip() == b'<event>':
                inside = True
                count += 1
            if inside:
                digest.update(line)
            if line.strip() == b'</event>':
                inside = False
    if inside:
        raise ValueError('Truncated event block in %s' % path)
    return {'events': count, 'event_blocks_sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--previous-metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    current = json.loads(args.metrics.read_text())
    previous = json.loads(args.previous_metrics.read_text())
    new, old = current['ampli'], previous['ampli']
    checks = {
        'expected_events': current['expected_events_per_backend'],
        'complete': new['complete_expected_sample'],
        'finite_weights': new['lhe']['all_weights_finite'],
        'idwtup': new['lhe']['idwtup'],
        'no_fatal_logs': not new['fatal_logs'],
        'cards_identical_to_previous_pool_run': new['cards_sha256'] == old['cards_sha256'],
        'same_production_trials': new['production']['trials'] == old['production']['trials'],
        'same_stored_candidates': new['production']['stored_candidates'] == old['production']['stored_candidates'],
        'topup_trials': new['production']['topup_trials'],
        'current_event_payload': event_digest(new['lhe']['path']),
        'previous_event_payload': event_digest(old['lhe']['path']),
        'rate_comparison': {},
        'cpu_comparison': {},
        'channels': [],
    }
    checks['identical_event_payloads'] = checks['current_event_payload'] == checks['previous_event_payload']
    for key in ('signed_pb', 'signed_error_pb', 'absolute_pb', 'absolute_error_pb'):
        checks['rate_comparison'][key] = {
            'current': new['integration'][key],
            'previous': old['integration'][key],
            'difference': new['integration'][key] - old['integration'][key],
        }
    for key in ('0', '1', '2'):
        checks['cpu_comparison'][key] = {
            'current_seconds': new['stage_cpu_seconds'][key],
            'previous_seconds': old['stage_cpu_seconds'][key],
        }
    manifest = new['production_manifest']
    checks['manifest_allowed_overweight_factor'] = manifest.get('allowed_overweight_factor')
    for channel in manifest['channels']:
        checks['channels'].append({key: value for key, value in channel.items()
                                   if key not in ('batches', 'rate_moments')})
    limit = checks['manifest_allowed_overweight_factor']
    excesses = [channel.get('selection', {}).get('overweight')
                for channel in manifest['channels']]
    checks['all_channels_within_one_percent'] = (
        limit == 0.01 and all(value is not None and value <= limit
                             for value in excesses))
    init = new['lhe']['init']
    checks['init_matches_production_rate'] = math.isclose(
        sum(row['signed_pb'] for row in init), new['integration']['signed_pb'],
        rel_tol=1e-6, abs_tol=1e-10)
    checks['init_matches_production_error'] = math.isclose(
        math.sqrt(sum(row['signed_error_pb']**2 for row in init)),
        new['integration']['signed_error_pb'], rel_tol=1e-6, abs_tol=1e-10)
    args.output.write_text(json.dumps(checks, indent=2, allow_nan=False) + '\n')
    print(json.dumps(checks, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
