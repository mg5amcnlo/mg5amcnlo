#!/usr/bin/env python3
"""Check archived benchmark consistency and independent audit agreement."""
from pathlib import Path
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name):
    return json.loads((HERE / name).read_text())


def numerical_contents(value):
    if isinstance(value, dict):
        return {key: numerical_contents(item) for key, item in value.items()
                if key not in {'path', 'process', 'collector_source', 'wrapper_metadata'}}
    if isinstance(value, list):
        return [numerical_contents(item) for item in value]
    return value


def equal(left, right):
    return math.isclose(left, right, rel_tol=2e-10, abs_tol=1e-12)


m = load('metrics.json')
t = load('mint_tail_metrics.json')
a = load('independent_native_audit.json')
it = load('independent_mint_audit.json')
provenance = load('source_provenance.json')
previous = json.loads((HERE.parent / 'ttbar_unfolded_10k_20261007/metrics.json').read_text())
n = m['ampli']['native_validation']
g = t['global_metrics']

checks = {
    'both_integrations_complete': m['comparison']['both_integrations_complete'],
    'both_300000_event_samples_complete': m['comparison']['both_samples_complete']
        and all(m[b]['lhe']['events'] == 300000 for b in ('mint', 'ampli')),
    'cards_identical_except_backend': not m['comparison']['run_card_differences']
        and m['comparison']['param_cards_identical']
        and m['comparison']['fks_card_differences'] == {
            'NLOPSIntegrator': {'mint': '0', 'ampli': '1'}},
    'automatic_accuracy': all(float(m[b]['run_card']['req_acc']) == -1. for b in ('mint', 'ampli')),
    'folding_disabled': all(m[b]['run_card']['folding'] == '1, 1, 1'
                            for b in ('mint', 'ampli')),
    'only_intended_card_changes_from_10k': all(
        {key for key in m[b]['run_card'] if m[b]['run_card'][key] != previous[b]['run_card'][key]} == {'nevents', 'req_acc'}
        and m[b]['fks_card'] == previous[b]['fks_card']
        and m[b]['cards_sha256']['param_card.dat'] == previous[b]['cards_sha256']['param_card.dat']
        for b in ('mint', 'ampli')),
    'sources_unchanged_from_10k': all(m[b]['exported_source_sha256'] == previous[b]['exported_source_sha256'] for b in ('mint', 'ampli')),
    'pristine_backend_sources_identical': m['comparison']['exported_sources_identical'],
    'current_sources_match_saved_provenance': all(
        hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
        for path, digest in provenance['working_tree_sources'].items()),
    'mint_diagnostic_worker_event_sequences_identical':
        t['reference']['exact_worker_event_sequences_identical'],
    'mint_diagnostic_final_event_sequence_identical':
        t['reference']['exact_final_event_sequence_identical'],
    'mint_diagnostic_cards_identical': all(t['reference']['cards_identical'].values()),
    'mint_diagnostic_uses_saved_instrumentation':
        (HERE / 'mint_tail/SubProcesses/mint_module.f90').read_bytes()
        == (HERE / 'mint_module_instrumented.f90').read_bytes(),
    'mint_diagnostic_other_exported_sources_unchanged': all(
        (HERE / 'mint' / path).read_bytes() == (HERE / 'mint_tail' / path).read_bytes()
        for path in m['mint']['exported_source_sha256']
        if path != 'SubProcesses/mint_module.f90'),
    'mint_all_parent_channels_and_positive_streams_covered':
        it['coverage'] == 1. and not it['missing_streams']
        and it['parent_channels'] == m['mint']['integration']['channels'],
    'ampli_all_coordinates_adaptive': all(
        all(w['adaptation']['mask'])
        and len(w['adaptation']['mask']) == w['adaptation']['ndim']
        and w['adaptation']['updates'] > 0 for w in n['workers']),
    'ampli_native_and_collected_tails_below_one_percent':
        n['all_native_and_collected_tail_fractions_below_one_percent'],
    'ampli_rate_and_collection_replay_passed': all(n[key] for key in (
        'rates_recomputed_from_survey_once_and_all_iterations',
        'initial_and_updated_quotas_reproduced', 'deterministic_collection_reproduced',
        'final_lhe_magnitudes_reproduced')),
    'archived_benchmark_analysis_matches': numerical_contents(m)
        == numerical_contents(load('archived_metrics.json')),
    'archived_tail_analysis_matches': numerical_contents(t)
        == numerical_contents(load('archived_mint_tail_metrics.json')),
    'archived_independent_audits_match':
        a == load('archived_independent_native_audit.json')
        and it == load('archived_independent_mint_audit.json'),
    'independent_native_audit_agrees': a['all_raw_checks_passed']
        and equal(a['signed_pb'], m['ampli']['integration']['signed_pb'])
        and equal(a['error_signed_pb'], m['ampli']['integration']['signed_error_pb'])
        and equal(a['collected_tail_fraction'], n['collected_tail_fraction'])
        and equal(a['max_tail_check'], n['maximum_native_tail_fraction'])
        and a['counts']['trials'] == m['ampli']['production']['trials'],
    'independent_mint_audit_agrees': it['all_raw_checks_passed']
        and it['exact_worker_and_final_event_replay']
        and equal(it['survey_full_tail'], g['survey_normalized_full_tail_fraction'])
        and equal(it['survey_full_tail_error'],
                  g['survey_normalized_full_tail_fraction_standard_error'])
        and equal(it['production_full_tail'], g['production_full_tail_fraction'])
        and it['tail_events'] == g['overweight_trials']
        and it['trials'] == m['mint']['production']['trials'],
    'all_launches_successful': all(load(b + '_finished.json')['cli_returncode'] == 0
        and load(b + '_finished.json')['events_exist']
        for b in ('mint', 'ampli', 'mint_tail')),
}
assert all(checks.values()), checks
report = dict(checks=checks, all_passed=True,
              scope='Consistency checks, not a requirement that MINT passes the 1% tail limit.',
              mint_full_tail_above_one_percent=g['survey_normalized_full_tail_fraction'] > .01)
(HERE / 'final_checks.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
