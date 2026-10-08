"""Analyze the new run; compare with completed, previously audited references."""
from pathlib import Path
import hashlib, json, math
import analyze_common

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'ttbar_unfolded_300k_20261007'
previous_path = PREVIOUS / 'metrics.json'
previous = json.loads(previous_path.read_text())
new = analyze_common.analyze(HERE / 'ampli', 'ampli', 300000, 'benchmark_300k',
                             HERE / 'sources/madgraph/various/ampli_pool.py')
assert new['complete_expected_sample'] and new['integration_complete'] and not new['fatal_logs']
assert new['stage_logs']['0'] == 0 and new['stage_logs']['1'] == 5
assert new['production_manifest']['integration_stages'] == ['survey', 'generation']
assert new['production_manifest']['survey_min_iterations'] == 4
assert all(c['survey_iterations'] >= 4 for c in new['production_manifest']['channels'])

def compare(reference):
    n, r = new['integration'], reference['integration']
    difference = n['signed_pb'] - r['signed_pb']
    error = math.hypot(n['signed_error_pb'], r['signed_error_pb'])
    return dict(rate_difference_pb=difference, combined_error_pb=error, rate_pull=difference/error,
                uncertainty_ratio=n['signed_error_pb']/r['signed_error_pb'],
                generation_cpu_ratio=new['production']['cpu_seconds']/reference['production']['cpu_seconds'],
                total_worker_cpu_ratio=new['total_cpu_seconds_completed_workers']/reference['total_cpu_seconds_completed_workers'],
                efficiency_ratio=new['production']['efficiency']/reference['production']['efficiency'],
                trial_ratio=new['production']['trials']/reference['production']['trials'],
                signed_effective_event_ratio=new['lhe']['signed_weight_effective_events']/reference['lhe']['signed_weight_effective_events'])

old = previous['ampli']
assert new['run_card'] == old['run_card']
assert new['fks_card'] == old['fks_card']
assert new['cards_sha256']['param_card.dat'] == old['cards_sha256']['param_card.dat']
tail = json.loads((PREVIOUS / 'mint_tail_metrics.json').read_text())
report = dict(expected_events=300000, ampli=new,
              comparison_with_saved_mint=compare(previous['mint']),
              comparison_with_previous_three_stage_ampli=compare(old),
              references=dict(metrics_path=str(previous_path),
                              metrics_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                              mint_run_reused=True, previous_ampli_run_reused=True,
                              mint_full_trial_tail=tail['global_metrics']['survey_normalized_full_tail_fraction']),
              interpretation='Only AmpliCol was rerun. References are the preceding 300K runs on the same machine and inputs. Efficiencies include rejected trials and reserve costs. Precision differs across native workflows; this is not an equal-precision comparison.')
(HERE / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'ampli'}, indent=2))
