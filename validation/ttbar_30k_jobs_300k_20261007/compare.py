"""Compare verified 30K/job generation with the saved 2.5K/job 300K run.

The new invocation starts from the exact saved survey. Its actual execution
cost is generation only; the equivalent full-workflow cost adds the inherited
survey CPU explicitly. Whole-launch wall times are not compared as ratios.
"""
from pathlib import Path
import hashlib
import json
import math
import re


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'ttbar_bounded_300k_20261007'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_card(path):
    result = {}
    for line in path.read_text().splitlines():
        match = re.match(r'^\s*(.*?)\s*=\s*(\w+)\s*(?:!.*)?$', line)
        if match:
            result[match[2]] = match[1]
    return result


def differences(left, right):
    return sorted(key for key in left.keys() | right.keys()
                  if left.get(key) != right.get(key))


def rate_comparison(current, reference, kind):
    delta = current[kind + '_pb'] - reference[kind + '_pb']
    combined = math.hypot(current[kind + '_error_pb'],
                          reference[kind + '_error_pb'])
    return dict(difference_pb=delta, quadrature_error_pb=combined,
                difference_over_quadrature_error=delta / combined,
                uncertainty_ratio=current[kind + '_error_pb'] /
                reference[kind + '_error_pb'])


def summary(metrics, survey_cpu, generation_only):
    production, events = metrics['production'], metrics['lhe']
    cpu = production['cpu_seconds']
    equivalent_cpu = cpu + survey_cpu
    return dict(
        nevt_job=int(metrics['run_card']['nevt_job']),
        integration=metrics['integration'],
        generation_cpu_seconds=cpu,
        survey_cpu_seconds_for_equivalent_workflow=survey_cpu,
        survey_was_inherited=generation_only,
        actual_worker_cpu_seconds=cpu if generation_only else equivalent_cpu,
        equivalent_full_workflow_worker_cpu_seconds=equivalent_cpu,
        invocation_wall_seconds=metrics['wrapper_finished']['wall_seconds'],
        invocation_wall_scope=(
            'generation-only invocation including coordination and collection'
            if generation_only else
            'full invocation including compilation, survey, generation and collection'),
        generation_phase_span_seconds_estimate=
        metrics['stage_wall_seconds_estimates']['2']['phase_span_seconds'],
        generation_phase_span_note=
        metrics['stage_wall_seconds_estimates']['2']['note'],
        generation_trials=production['trials'],
        nonzero_generation_trials=production['nonzero_trials'],
        final_events=events['events'],
        final_events_per_generation_trial=events['events'] / production['trials'],
        final_events_per_generation_cpu_second=events['events'] / cpu,
        production_workers=production['logged_workers'],
        native_iterations=production['native_iterations'],
        grid_updates=production['grid_updates'],
        retained_candidates=production['stored_candidates'],
        generated_reserve_target=production['native_reserve_target'],
        collection_rounds=production['collection_rounds'],
        signed_weight_effective_events=events['signed_weight_effective_events'],
        absolute_weight_effective_events=events['absolute_weight_effective_events'],
        negative_event_fraction=events['negative_fraction'],
        negative_absolute_weight_fraction=events['weighted_negative_fraction'],
        signed_effective_events_per_generation_cpu_second=
        events['signed_weight_effective_events'] / cpu,
        signed_effective_events_per_equivalent_total_cpu_second=
        events['signed_weight_effective_events'] / equivalent_cpu,
        overweight={key: metrics['native_validation'][key] for key in (
            'collected_tail_fraction', 'maximum_native_tail_fraction',
            'maximum_final_channel_tail_bound')})


def main():
    new = read(HERE / 'metrics.json')
    old = read(PREVIOUS / 'metrics.json')
    started = read(HERE / 'ampli_started.json')
    old_started = read(PREVIOUS / 'ampli_started.json')
    new_out, old_out = HERE / 'ampli', PREVIOUS / 'ampli'
    checks = {}
    checks['both_samples_complete'] = all(
        metrics['complete_expected_sample'] and metrics['integration_complete']
        and metrics['lhe']['events'] == 300000 and not metrics['fatal_logs']
        and metrics['production']['all_assigned_workers_finished']
        for metrics in (new, old))
    checks['new_invocation_reuses_survey'] = (
        started['generation_only'] is True and started['fresh_survey'] is False)
    checks['restart_restored_saved_allocation_order'] = (
        started['restored_saved_allocation_order'] is True)
    checks['same_seed_cores_request_and_folding'] = all(
        metadata['seed'] == 19727 and metadata['cores'] == 5
        and metadata['expected_events'] == 300000 and metadata['req_acc'] == -1
        and metadata['folding'] == [1, 1, 1]
        for metadata in (started, old_started))
    checks['run_cards_differ_only_by_job_limit'] = (
        differences(new['run_card'], old['run_card']) == ['nevt_job']
        and new['run_card']['nevt_job'] == '30000'
        and old['run_card']['nevt_job'] == '2500')
    checks['prelaunch_cards_differ_only_by_job_limit'] = (
        differences(run_card(HERE / 'ampli_run_card.dat'),
                    run_card(PREVIOUS / 'ampli_run_card.dat')) == ['nevt_job'])
    checks['prelaunch_and_actual_seeds_confirmed'] = all(
        run_card(directory / 'ampli_run_card.dat')['iseed'] == '19727'
        and (directory / 'ampli/SubProcesses/randinit').read_text().strip() == 'r=19727'
        for directory in (HERE, PREVIOUS))
    checks['fks_cards_identical'] = new['fks_card'] == old['fks_card']
    checks['parameter_and_fks_files_identical'] = all(
        digest(new_out / 'Cards' / name) == digest(old_out / 'Cards' / name)
        == new['cards_sha256'][name] == old['cards_sha256'][name]
        for name in ('param_card.dat', 'FKS_params.dat'))
    checks['template_and_exported_source_fingerprints_identical'] = (
        started['source_sha256'] == old_started['source_sha256']
        and started['exported_source_sha256'] == old_started['exported_source_sha256']
        == new['exported_source_sha256'] == old['exported_source_sha256'])
    checks['source_snapshots_and_exports_match_fingerprints'] = all(
        digest(directory / 'sources' / path) == value
        for directory, metadata in ((HERE, started), (PREVIOUS, old_started))
        for path, value in metadata['source_sha256'].items()) and all(
        digest(directory / 'ampli' / path) == value
        for directory, metadata in ((HERE, started), (PREVIOUS, old_started))
        for path, value in metadata['exported_source_sha256'].items())
    survey_files = []
    for channel in old['production_manifest']['channels']:
        parent = Path('SubProcesses') / channel['subprocess'] / ('GF' + channel['channel'])
        for name in ('ampli_grids', 'grid.MC_integer', 'res_1.dat', 'log_MINT1.txt'):
            relative = parent / name
            survey_files.append(dict(path=str(relative),
                                     new_sha256=digest(new_out / relative),
                                     reference_sha256=digest(old_out / relative)))
    checks['all_twenty_survey_files_byte_identical'] = (
        len(survey_files) == 20 and all(
            record['new_sha256'] == record['reference_sha256'] for record in survey_files))
    checks['survey_statistics_identical'] = new['survey'] == old['survey']
    allocation_fields = ('subprocess', 'channel', 'allocation_order', 'initial_quota')
    allocations = [
        [tuple(channel[field] for field in allocation_fields) for channel in sorted(
            metrics['production_manifest']['channels'], key=lambda channel: channel['allocation_order'])]
        for metrics in (new, old)]
    checks['initial_channel_quotas_and_allocation_order_identical'] = (
        allocations[0] == allocations[1])
    checks['both_follow_two_stage_workflow'] = all(
        metrics['production_manifest']['integration_stages'] == ['survey', 'generation']
        and metrics['stage_logs']['0'] == 0 and metrics['stage_logs']['1'] == 5
        for metrics in (new, old))
    checks['native_tail_and_collection_replays_passed'] = all(
        metrics['native_validation'][key]
        for metrics in (new, old)
        for key in ('rates_recomputed_from_survey_once_and_all_iterations',
                    'initial_and_updated_quotas_reproduced',
                    'deterministic_collection_reproduced',
                    'final_lhe_magnitudes_reproduced',
                    'all_native_and_collected_tail_fractions_below_one_percent'))
    checks['each_batch_honors_its_job_limit'] = all(
        batch['generated_target'] <= int(metrics['run_card']['nevt_job'])
        for metrics in (new, old)
        for channel in metrics['production_manifest']['channels']
        for batch in channel['batches'])
    survey_cpu = old['production']['survey_cpu_seconds']
    checks['saved_survey_cpu_confirmed'] = math.isclose(
        survey_cpu, 59.93562422, rel_tol=1e-12)
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]

    current, reference = summary(new, survey_cpu, True), summary(old, survey_cpu, False)
    ratio_fields = (
        'generation_cpu_seconds', 'equivalent_full_workflow_worker_cpu_seconds',
        'generation_trials', 'final_events_per_generation_trial',
        'final_events_per_generation_cpu_second', 'production_workers',
        'native_iterations', 'grid_updates', 'signed_weight_effective_events',
        'absolute_weight_effective_events',
        'signed_effective_events_per_generation_cpu_second',
        'signed_effective_events_per_equivalent_total_cpu_second')
    report = dict(
        all_comparability_checks_passed=True, checks=checks,
        new_30000_per_job=current, saved_2500_per_job=reference,
        new_over_reference_ratios={key: current[key] / reference[key]
                                   for key in ratio_fields},
        generation_cpu_reduction_percent=100 * (1 - current['generation_cpu_seconds'] /
                                                reference['generation_cpu_seconds']),
        generation_speedup=reference['generation_cpu_seconds'] /
        current['generation_cpu_seconds'],
        equivalent_full_workflow_cpu_reduction_percent=100 * (
            1 - current['equivalent_full_workflow_worker_cpu_seconds'] /
            reference['equivalent_full_workflow_worker_cpu_seconds']),
        signed_rate=rate_comparison(new['integration'], old['integration'], 'signed'),
        absolute_rate=rate_comparison(new['integration'], old['integration'], 'absolute'),
        initial_allocation=[dict(zip(allocation_fields, row)) for row in allocations[0]],
        survey_file_comparison=survey_files,
        reference_hashes={str(path.relative_to(HERE.parent)): digest(path) for path in (
            HERE / 'metrics.json', PREVIOUS / 'metrics.json',
            HERE / 'ampli_started.json', PREVIOUS / 'ampli_started.json')},
        interpretation=[
            'Only AmpliCol generation with nevt_job=30000 was rerun. The nevt_job=2500 reference is the saved completed 300K run.',
            'Both runs use the exact same saved survey grids, maxima and statistics, the same source code, seed, physics cards and five-core setting.',
            'The copied restart job list was restored to its saved original allocation order, preserving the original initial channel quotas exactly before applying the new job-size limit.',
            'Generation CPU and trial counts are directly comparable. Single-run CPU timing remains subject to host variation.',
            'The new execution did not repeat the survey. Equivalent full-workflow CPU adds the inherited 59.93562422-second survey to generation; this is not newly measured survey time.',
            'Whole-invocation wall times have different scope: the reference includes compilation and survey, while the new invocation restarts generation. No wall-time ratio is reported.',
            'Generation-phase spans are estimates from worker log timestamps and integer wall counters, and require preserved timestamps; they exclude compilation, survey and final collection.',
            'nevt_job limits reserve events per worker, including the 10% overhead. Native correction weights are retained in both LHE samples.',
            'Final-event efficiency includes all production trials, reserve generation and any top-ups. Rates use the survey once plus all generation iterations.',
            'The two allocations change sampling histories and achieved precision; the CPU comparison is at fixed requested sample size, not fixed rate uncertainty.',
            'Quadrature rate-error diagnostics are descriptive scales, not calibrated significances: the runs share survey data and a seed, so independent errors are not established.',
            'The signed-rate shift is about 2.88 nominal quadrature errors, and the absolute-rate shift about 3.13. This single shared-survey comparison does not establish compatibility or bias; multiple independent seeds are needed to assess the variation.',
        ])
    (HERE / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
