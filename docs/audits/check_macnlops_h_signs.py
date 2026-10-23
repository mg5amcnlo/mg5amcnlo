#!/usr/bin/env python3
"""Audit actual source-LHE H signs and cancellation bounds before showering.

Every nonzero H source weight must have the sign of the complete physical
R-D density. Negative H also requires (D-R) <= K. This is a source-sign
audit, not a shower audit or an event-selection step; source weights and
event blocks are never rewritten. The generated evaluators must be built.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from madgraph.various import macnlops as mc


def point_checks(weight, density):
    values = {name: getattr(density, name)
              for name in ('real', 'subtraction', 'shower')}
    if not all(math.isfinite(value) for value in values.values()):
        return ['nonfinite_density'], dict(
            densities={key: value if math.isfinite(value) else str(value)
                       for key, value in values.items()})
    real, subtraction, shower = (values[key] for key in values)
    if real < 0 or shower < 0:
        return ['negative_real_or_shower_density'], dict(densities=values)
    hard = real - subtraction
    scale = max(abs(real), abs(subtraction), abs(shower), sys.float_info.min)
    sign_residual = max(-math.copysign(1., weight) * hard, 0.) / scale
    negative = max(-hard, 0.)
    ratio = negative / shower if shower else (None if negative else 0.)
    bound_residual = max(negative - shower, 0.) / scale if weight < 0 else 0.
    errors = []
    if hard == 0:
        errors.append('zero_global_H_with_nonzero_source_weight')
    elif (weight > 0) != (hard > 0):
        errors.append('source_H_sign_disagrees_with_global_R_minus_D')
    # Match the rejection protocol's machine-roundoff allowance at H-/K=1.
    if weight < 0 and negative and (
            shower == 0 or ratio > 1. + 32 * math.ulp(1.)):
        errors.append('negative_H_exceeds_shower_bound')
    return errors, dict(densities=values, global_H=hard,
        density_scale=scale, sign_violation_scaled=sign_residual,
        negative_H_bound_excess_scaled=bound_residual,
        negative_H_over_K=ratio, abs_global_H_scaled=abs(hard) / scale)


def audit_source(source, evaluator, support, output, expected_events=None,
                 max_events=None):
    """Stream all H records; save exact failures and no selected event sample."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = mc.sha256(source)
    evaluator.configure(support['beam_xmax'], support['shower_cutoffs'])
    counts, maxima = Counter(), Counter()
    weights = defaultdict(list)
    first_failure = None
    complete = True
    minimum_h = None
    with mc.LHEStream(source) as stream, (output / 'failures.jsonl').open('w') as failures:
        for sequence, block in enumerate(stream, 1):
            if max_events is not None and sequence > max_events:
                complete = False
                break
            event = mc.Event(block)
            counts['source_events'] += 1
            counts['source_' + event.kind] += 1
            weights['source'].append(event.weight)
            if event.kind != 'H':
                continue
            weights['H'].append(event.weight)
            if event.weight == 0:
                counts['zero_weight_H'] += 1
                continue
            counts['negative_H' if event.weight < 0 else 'positive_H'] += 1
            weights['negative_H' if event.weight < 0 else 'positive_H'].append(event.weight)
            fatal = False
            try:
                density = evaluator.evaluate(event, sequence)
                errors, details = point_checks(event.weight, density)
                counts['evaluated_H'] += 1
            except (mc.MatchingError, OSError, ValueError) as error:
                errors, details = ['native_evaluator_error'], dict(error=str(error))
                fatal = True
            for key in ('sign_violation_scaled', 'negative_H_bound_excess_scaled'):
                maxima[key] = max(maxima[key], details.get(key, 0.))
            ratio = details.get('negative_H_over_K')
            if ratio is not None:
                maxima['negative_H_over_K'] = max(maxima['negative_H_over_K'], ratio)
            if 'abs_global_H_scaled' in details:
                minimum_h = min(minimum_h if minimum_h is not None else math.inf,
                                details['abs_global_H_scaled'])
            if not errors:
                counts['verified_H'] += 1
                continue
            counts['failed_H'] += 1
            counts['failed_negative_H' if event.weight < 0 else 'failed_positive_H'] += 1
            counts.update(errors)
            failure = dict(sequence=sequence, source_weight=event.weight,
                           process=event.process, errors=errors,
                           source_block=block, **details)
            failures.write(json.dumps(failure, allow_nan=False) + '\n')
            if first_failure is None:
                first_failure = failure
                (output / 'first_failure.json').write_text(
                    json.dumps(failure, indent=2, allow_nan=False) + '\n')
                (output / 'first_failure.lhe').write_text(
                    stream.header + block + '</LesHouchesEvents>\n')
            if fatal:
                complete = False
                break
    after = mc.sha256(source)
    if before != after:
        raise mc.MatchingError('Source file changed during its read-only H audit')
    if expected_events is not None and counts['source_events'] != expected_events:
        counts['unexpected_source_event_count'] += 1
        complete = False
    status = 'failed' if counts['failed_H'] else ('passed' if complete else 'incomplete')
    report = dict(schema='macnlops-source-H-sign-audit-v1',
        scope='source-H-sign-and-bound; no shower or event rejection',
        status=status, complete_source=complete, source=str(source.resolve()),
        source_sha256=before, source_unchanged=True,
        expected_events=expected_events, diagnostic_max_events=max_events,
        counts=dict(counts), maxima=dict(maxima),
        max_scaled_residual=max(maxima['sign_violation_scaled'],
                                maxima['negative_H_bound_excess_scaled']),
        minimum_abs_global_H_scaled=minimum_h,
        original_weight_sums={key: math.fsum(value) for key, value in weights.items()},
        first_failure=first_failure,
        failures_jsonl=str((output / 'failures.jsonl').resolve()))
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--process-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-events', type=int)
    parser.add_argument('--max-events', type=int,
        help='diagnostic prefix only; a truncated source cannot pass the full audit')
    parser.add_argument('source', type=Path)
    args = parser.parse_args(argv)
    if args.max_events is not None and args.max_events <= 0:
        parser.error('--max-events must be positive')
    process, source = args.process_dir.resolve(), args.source.resolve()
    manifest = mc.check_manifest(source, process)
    card = mc.check_source(mc.source_header(source, process, manifest), process)
    profile = mc.generation_profile(process)
    pdf, info = mc.pdf_name(process, card, with_info=True)
    if pdf != profile['pdf'] or mc.sha256(info) != profile['pdf_info_sha256'] or \
            mc.pdf_member_hash(pdf, info) != profile['pdf_member_sha256']:
        raise mc.MatchingError('Source PDF differs from the frozen generation profile')
    environment = dict(os.environ)
    environment['LHAPDF_DATA_PATH'] = str(info.parent.parent) + os.pathsep + \
                                     environment.get('LHAPDF_DATA_PATH', '')
    log_dir = args.output_dir.with_name(args.output_dir.name + '_evaluator_logs')
    log_dir.mkdir(parents=True, exist_ok=False)
    with mc.NativeEvaluator(process, log_dir, environment=environment) as evaluator:
        report = audit_source(source, evaluator, profile['support'], args.output_dir,
                              args.expected_events, args.max_events)
    report['evaluator_executables_sha256'] = {
        str(directory.relative_to(process)): mc.sha256(directory / 'macnlops_evaluator')
        for directory in evaluator.workers}
    report['evaluator_driver_sha256'] = mc.sha256(process / 'SubProcesses/driver_macnlops.f')
    report['audit_script_sha256'] = mc.sha256(Path(__file__))
    report['evaluator_logs'] = str(log_dir.resolve())
    report['generation_profile_sha256'] = mc.sha256(process / 'Cards/macnlops_profile.json')
    report['generation_lhe_sha256'] = manifest['lhe_sha256']
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in
        ('status', 'complete_source', 'counts', 'max_scaled_residual')}, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (mc.MatchingError, OSError) as error:
        sys.exit('Source H audit failed: ' + str(error))
