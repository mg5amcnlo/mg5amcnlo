"""Compare verified bounded 300K metrics with the audited earlier 300K runs."""
from pathlib import Path
import hashlib
import json
import math


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'ttbar_unfolded_300k_20261007'
STOPPED = HERE.parent / 'ttbar_two_stage_300k_20261007'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate_comparison(current, reference, kind):
    difference = current[kind + '_pb'] - reference[kind + '_pb']
    combined = math.hypot(current[kind + '_error_pb'], reference[kind + '_error_pb'])
    return dict(difference_pb=difference, quadrature_error_pb=combined,
                difference_over_quadrature_error=difference / combined,
                uncertainty_ratio=current[kind + '_error_pb'] / reference[kind + '_error_pb'])


def summary(metrics):
    production, events = metrics['production'], metrics['lhe']
    return dict(integration=metrics['integration'],
                generation_cpu_seconds=production['cpu_seconds'],
                survey_and_grid_cpu_seconds=production['survey_cpu_seconds'],
                total_worker_cpu_seconds=metrics['total_cpu_seconds_completed_workers'],
                whole_launch_wall_seconds=metrics.get(
                    'whole_launch_wall_seconds', metrics['wrapper_finished']['wall_seconds']),
                generation_trials=production['trials'],
                final_events=events['events'],
                final_events_per_generation_trial=production['efficiency'],
                production_workers=production['logged_workers'],
                signed_weight_effective_events=events['signed_weight_effective_events'],
                absolute_weight_effective_events=events['absolute_weight_effective_events'],
                negative_event_fraction=events['negative_fraction'],
                negative_absolute_weight_fraction=events['weighted_negative_fraction'],
                signed_effective_events_per_generation_cpu_second=
                events['signed_weight_effective_events_per_generation_cpu_second'],
                signed_effective_events_per_total_worker_cpu_second=
                events['signed_weight_effective_events_per_total_cpu_second'])


def compare(current, reference):
    c, r = summary(current), summary(reference)
    ratio_fields = (
        'generation_cpu_seconds', 'survey_and_grid_cpu_seconds',
        'total_worker_cpu_seconds', 'whole_launch_wall_seconds', 'generation_trials',
        'final_events_per_generation_trial', 'signed_weight_effective_events',
        'absolute_weight_effective_events',
        'signed_effective_events_per_generation_cpu_second',
        'signed_effective_events_per_total_worker_cpu_second')
    return dict(signed_rate=rate_comparison(c['integration'], r['integration'], 'signed'),
                absolute_rate=rate_comparison(c['integration'], r['integration'], 'absolute'),
                new_over_reference_ratios={key: c[key] / r[key] for key in ratio_fields})


def main():
    new = read(HERE / 'metrics.json')
    previous = read(PREVIOUS / 'metrics.json')
    old = previous['ampli']
    mint = previous['mint']
    tail = read(PREVIOUS / 'mint_tail_metrics.json')
    started = read(HERE / 'ampli_started.json')
    stopped_started = read(STOPPED / 'ampli_started.json')
    physics = read(STOPPED / 'physics_source_comparison.json')
    checks = {}

    checks['all_samples_complete'] = all(
        m['complete_expected_sample'] and m['integration_complete'] and
        m['lhe']['events'] == 300000 and not m['fatal_logs']
        for m in (new, old, mint))
    checks['run_cards_identical'] = new['run_card'] == old['run_card'] == mint['run_card']
    checks['ampli_fks_cards_identical'] = new['fks_card'] == old['fks_card']
    checks['mint_fks_differs_only_by_backend'] = (
        {k: v for k, v in new['fks_card'].items() if k != 'NLOPSIntegrator'} ==
        {k: v for k, v in mint['fks_card'].items() if k != 'NLOPSIntegrator'})
    parameter_hash = digest(HERE / 'ampli/Cards/param_card.dat')
    checks['parameter_cards_identical'] = all(
        m['cards_sha256']['param_card.dat'] == parameter_hash for m in (new, old, mint))
    checks['saved_parameter_cards_match_metrics'] = all(
        digest(PREVIOUS / backend / 'Cards/param_card.dat') == parameter_hash
        for backend in ('mint', 'ampli'))
    checks['same_seed_cores_and_request'] = (
        started['seed'] == 19727 and started['cores'] == 5 and
        started['req_acc'] == -1 and started['expected_events'] == 300000 and
        started['folding'] == [1, 1, 1] and all(
            m['wrapper_started']['settings']['iseed'] == started['seed'] and
            m['wrapper_started']['cores'] == started['cores'] and
            m['wrapper_started']['settings']['nevt_job'] == 2500
            for m in (old, mint)))
    # Persistent run cards reset iseed to zero after launch. Check the saved
    # prelaunch card and final randinit as well as launch metadata.
    checks['saved_input_cards_identical_to_stopped_run'] = all(
        digest(HERE / ('ampli_' + name)) == digest(STOPPED / ('ampli_' + name))
        for name in ('run_card.dat', 'param_card.dat', 'FKS_params.dat'))
    checks['actual_seed_confirmed'] = (
        (HERE / 'ampli/SubProcesses/randinit').read_text().strip() == 'r=19727')
    checks['prior_physics_comparison_passed'] = (
        physics['banner']['SLHA_identical'] and physics['banner']['run_parameters_identical'] and
        physics['banner']['old_seed'] == physics['banner']['new_seed'] == '19727' and
        physics['summary']['unexpected_changes'] == 0)
    checks['saved_new_sources_match_provenance'] = all(
        digest(HERE / 'sources' / path) == value
        for path, value in started['source_sha256'].items())
    checks['new_exported_sources_match_provenance'] = all(
        digest(HERE / 'ampli' / path) == value
        for path, value in started['exported_source_sha256'].items())
    changed_since_stopped = sorted(
        path for path, value in started['source_sha256'].items()
        if value != stopped_started['source_sha256'][path])
    checks['only_scheduler_changed_since_stopped_run'] = (
        changed_since_stopped == ['Template/NLO/SubProcesses/simple_integrator.f90'])
    changed_since_reference = sorted(
        path for path, value in new['exported_source_sha256'].items()
        if value != old['exported_source_sha256'][path])
    checks['only_expected_numerical_changes_since_three_stage_run'] = (
        changed_since_reference == sorted([
            'SubProcesses/ampli_mint_adapter.f90', 'SubProcesses/driver_mintMC.f',
            'SubProcesses/simple_integrator.f90', 'bin/internal/amcatnlo_run_interface.py']))
    checks['reference_mint_and_ampli_exported_sources_identical'] = (
        old['exported_source_sha256'] == mint['exported_source_sha256'])
    checks['reference_exported_sources_match_metrics'] = all(
        digest(PREVIOUS / backend / path) == value
        for backend, metrics in (('mint', mint), ('ampli', old))
        for path, value in metrics['exported_source_sha256'].items())
    checks['mint_tail_replay_identical_to_pristine'] = all(
        tail['reference'][name] for name in (
            'exact_worker_event_sequences_identical', 'exact_final_event_sequence_identical'))
    checks['new_workflow_has_only_two_stages'] = (
        new['stage_logs']['0'] == 0 and new['stage_logs']['1'] == 5 and
        new['production_manifest']['integration_stages'] == ['survey', 'generation'])
    checks['same_survey_and_maxima_as_stopped_run'] = (
        new['bounded_scheduler_validation']['same_survey_and_maxima_as_stopped_300k'])
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]

    report = dict(
        all_comparability_checks_passed=True, checks=checks,
        new_ampli=summary(new), saved_mint=summary(mint),
        saved_three_stage_ampli=summary(old),
        comparison_with_saved_mint=compare(new, mint),
        comparison_with_previous_three_stage_ampli=compare(new, old),
        overweight=dict(
            new_ampli={k: new['native_validation'][k] for k in (
                'collected_tail_fraction', 'maximum_native_tail_fraction',
                'maximum_final_channel_tail_bound')},
            previous_ampli={k: old['native_validation'][k] for k in (
                'collected_tail_fraction', 'maximum_native_tail_fraction',
                'maximum_final_channel_tail_bound')},
            mint_full_cross_section_tail=tail['global_metrics']['survey_normalized_full_tail_fraction'],
            mint_full_cross_section_tail_standard_error=
            tail['global_metrics']['survey_normalized_full_tail_fraction_standard_error'],
            mint_maximum_parent_channel_tail=
            tail['global_metrics']['maximum_channel_survey_normalized_full_tail_fraction'],
            mint_maximum_split_worker_tail=
            tail['global_metrics']['maximum_worker_survey_normalized_full_tail_fraction'],
            mint_collected_nominal_weight_tail=
            tail['final_lhe']['tail_flagged_absolute_weight_fraction']),
        source_changes=dict(since_stopped_run=changed_since_stopped,
                            since_three_stage_reference=changed_since_reference),
        references={str(path.relative_to(HERE.parent)): digest(path) for path in (
            HERE / 'metrics.json', PREVIOUS / 'metrics.json',
            PREVIOUS / 'mint_tail_metrics.json', STOPPED / 'physics_source_comparison.json')},
        interpretation=[
            'Only AmpliCol was rerun. MINT and previous AmpliCol are earlier completed 300K runs on the same machine and inputs.',
            'MINT timing is from the pristine run; its tail diagnostic replay has extra I/O and is used only for overweight measurements.',
            'Worker CPU excludes compilation and Python coordination; launch wall time includes them. Single-run timings are subject to host variation.',
            'Efficiencies include rejection and reserve costs. Native workflows achieve different precision; this is not an equal-precision timing comparison.',
            'Rate quadrature errors and difference/error diagnostics assume independent errors. Runs share a seed and integrand, so independence is not established; these are descriptive scales, not calibrated significances.',
            'AmpliCol retains native overweight correction factors. Its sample tail share, native worker bound and MINT global full-cross-section tail have different aggregation.',
        ])
    (HERE / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
