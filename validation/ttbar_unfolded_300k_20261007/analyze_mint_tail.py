#!/usr/bin/env python3
"""Analyze diagnostic MINT generation traces without changing event weights.

The trace records each complete folded trial, including rejected/zero points.
For each independently sampled channel/stream the proposal-corrected absolute
observation is a = f * Z / H. The overweight mass is a for f > H; the excess
mass is a - Z for f > H. These two quantities are intentionally distinct.

MINT chooses its virtual/nonvirtual stream once per requested accepted event,
then retries within that stream until acceptance. Its raw trial mixture is
therefore NOT the survey mixture. Aggregate trials across split workers within
each channel/stream first, estimate separate stream fractions, then combine
them using survey absolute rates. Also report the direct production estimate
formed by summing the separate stream means. Neither estimator is a new MINT
normalization: the generated sample retains its original survey normalization.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?'
EVENT = re.compile(r'<event(?:\s[^>]*)?>.*?</event>', re.S)
STREAMS = {1: 'virtual', 2: 'nonvirtual', 3: 'born'}


def number(value):
    return float(value.replace('D', 'e').replace('d', 'e'))


def close(actual, expected, label, rtol=2.e-12, atol=1.e-12):
    if not math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
        raise ValueError('%s: %r != %r' % (label, actual, expected))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_events(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        text = handle.read()
    if '</LesHouchesEvents>' not in text:
        raise ValueError('Unclosed LHE file: %s' % path)
    result = []
    for block in EVENT.findall(text):
        # Exact event payload, including comments and interior whitespace.
        digest = hashlib.sha256(block.encode()).hexdigest()
        header = next(line for line in block.splitlines()[1:]
                      if line.strip() and not line.lstrip().startswith('#'))
        result.append((digest, number(header.split()[2])))
    return result


def result_file(path):
    fields = path.read_text().split()
    return dict(zip(('absolute_pb', 'absolute_error_pb', 'signed_pb',
                     'signed_error_pb', 'iterations', 'points', 'cpu_seconds'),
                    map(number, fields[:7])))


def read_trace(path):
    lines = path.read_text().splitlines()
    if lines[0].strip() != 'MG5_MINT_TAIL 1':
        raise ValueError('Unsupported trace protocol: %s' % path)
    dimensions, configuration, channels = map(int, lines[1].split())
    if channels != 1:
        raise ValueError('Expected one channel per executable: %s' % path)
    survey_nonvirtual, survey_virtual, survey_signed = map(number, lines[2].split())
    normalizers = list(map(number, lines[3].split()))
    if len(normalizers) != 3:
        raise ValueError('Expected three stream normalizers: %s' % path)
    rows, event_count, footer = [], 0, None
    roundoff = dict(rows=0, maximum_absolute_excess=0., maximum_proposal_scaled_excess=0.)
    for line in lines[4:]:
        if not line.strip():
            continue
        if line.lstrip().startswith('#'):
            if line.lstrip().startswith('# END'):
                footer = line.strip()
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError('Malformed trial row in %s: %s' % (path, line))
        trial, stream, accepted, event = map(int, fields[:4])
        absolute, signed, bound, normalizer, observation = map(number, fields[4:])
        if not all(math.isfinite(x) for x in (absolute, signed, bound, normalizer, observation)):
            raise ValueError('Nonfinite trial in %s' % path)
        if trial != len(rows) + 1 or stream not in STREAMS or accepted not in (0, 1):
            raise ValueError('Invalid trial counters in %s' % path)
        if absolute < 0 or bound <= 0 or normalizer <= 0:
            raise ValueError('Invalid positive proposal weights in %s' % path)
        close(normalizer, normalizers[stream - 1], 'fixed stream normalizer')
        close(observation, absolute * normalizer / bound, 'proposal-corrected observation')
        if abs(signed) > absolute * (1 + 1.e-10) + 1.e-12:
            # Subtraction cancellations can leave a tiny signed/absolute
            # mismatch relative to the local proposal scale. Record it,
            # without altering either observation or any tail decision.
            excess = abs(signed) - absolute
            scaled = excess / max(absolute, bound, normalizer)
            # The 300K sample contains one residual at 2.36e-12 of this
            # scale; the raw scan is saved in mint_roundoff_diagnostics.json.
            if scaled > 1.e-11:
                raise ValueError('Signed folded observation exceeds absolute observation: %s' % path)
            roundoff['rows'] += 1
            roundoff['maximum_absolute_excess'] = max(roundoff['maximum_absolute_excess'], excess)
            roundoff['maximum_proposal_scaled_excess'] = max(roundoff['maximum_proposal_scaled_excess'], scaled)
        if accepted:
            event_count += 1
            if event != event_count:
                raise ValueError('Accepted event index does not match event sequence')
        # Instrumentation may mark rejected rows with zero, the current event
        # index, or the index being attempted; none is used for LHE mapping.
        elif event not in (0, event_count, event_count + 1):
            raise ValueError('Invalid rejected-event index')
        tail = absolute > bound
        if tail and not accepted:
            raise ValueError('An above-envelope point was unexpectedly rejected')
        rows.append(dict(trial=trial, stream=stream, accepted=bool(accepted),
                         event=event, absolute=absolute, signed=signed,
                         bound=bound, normalizer=normalizer, observation=observation,
                         ratio=absolute / bound, tail=tail))
    if footer is None:
        raise ValueError('Missing completed-run footer in %s' % path)
    footer_numbers = [int(x) for x in footer.split()[2:] if x.isdigit()]
    if footer_numbers:
        if footer_numbers[0] != len(rows):
            raise ValueError('Footer trial counter disagrees with records')
        if len(footer_numbers) > 1 and footer_numbers[1] != event_count:
            raise ValueError('Footer event counter disagrees with records')
        if len(footer_numbers) > 2 and footer_numbers[2] != sum(r['tail'] for r in rows):
            raise ValueError('Footer envelope-failure counter disagrees with records')
    return dict(dimensions=dimensions, configuration=configuration,
                survey_nonvirtual_pb=survey_nonvirtual, survey_virtual_pb=survey_virtual,
                survey_signed_pb=survey_signed, proposal_normalizers=normalizers,
                rows=rows, footer=footer, events=event_count,
                signed_absolute_roundoff=roundoff)


def summarize(rows):
    count = len(rows)
    absolute = [r['observation'] for r in rows]
    tail = [r['observation'] if r['tail'] else 0. for r in rows]
    excess = [r['observation'] - r['normalizer'] if r['tail'] else 0. for r in rows]
    signed = [r['signed'] * r['normalizer'] / r['bound'] for r in rows]
    sums = [math.fsum(values) for values in (absolute, tail, excess, signed)]
    result = dict(trials=count, accepted_events=sum(r['accepted'] for r in rows),
                  zero_trials=sum(r['absolute'] == 0 for r in rows),
                  overweight_trials=sum(r['tail'] for r in rows),
                  maximum_weight_over_envelope=max((r['ratio'] for r in rows), default=0.),
                  sum_proposal_corrected_absolute=sums[0],
                  sum_proposal_corrected_full_tail=sums[1],
                  sum_proposal_corrected_excess=sums[2],
                  sum_proposal_corrected_signed=sums[3])
    for key, value in zip(('absolute', 'full_tail', 'excess', 'signed'), sums):
        result['production_mean_' + key + '_pb'] = value / count if count else None
    mean_absolute = sums[0] / count if count else 0.
    result['production_variance_mean_absolute_pb2'] = math.fsum((a - mean_absolute)**2 for a in absolute) / (count * (count - 1)) if count > 1 else None
    result['trial_count_overweight_fraction'] = result['overweight_trials'] / count if count else None
    result['generation_efficiency'] = result['accepted_events'] / count if count else None
    for key, values, total in (('full_tail', tail, sums[1]), ('excess', excess, sums[2])):
        fraction = total / sums[0] if sums[0] else None
        result[key + '_fraction'] = fraction
        # Delta-method uncertainty of a self-normalized ratio. It conditions
        # on the observed trial count and frozen grids, and excludes survey
        # errors. Fixed-accepted-event stopping makes it approximate.
        residual_square_sum = math.fsum((v - fraction * a)**2 for v, a in zip(values, absolute)) if fraction is not None else None
        variance = count * residual_square_sum / ((count - 1) * sums[0]**2) if count > 1 and sums[0] else None
        result[key + '_fraction_standard_error'] = math.sqrt(variance) if variance is not None else None
        mean_value = total / count if count else 0.
        result['production_variance_mean_' + key + '_pb2'] = math.fsum((v - mean_value)**2 for v in values) / (count * (count - 1)) if count > 1 else None
        result['production_covariance_mean_absolute_' + key + '_pb2'] = math.fsum((a - mean_absolute) * (v - mean_value) for a, v in zip(absolute, values)) / (count * (count - 1)) if count > 1 else None
    return result


def combined_streams(streams):
    positive = [s for s in streams if s['survey_absolute_pb'] > 0.]
    if any(not s['trials'] or s['full_tail_fraction'] is None for s in positive):
        raise ValueError('A nonzero survey component has no usable production observations')
    total_survey = math.fsum(s['survey_absolute_pb'] for s in streams)
    total_production = math.fsum(s['production_mean_absolute_pb'] or 0. for s in streams)
    out = dict(survey_absolute_pb=total_survey,
               production_absolute_pb=total_production,
               production_signed_pb=math.fsum(s['production_mean_signed_pb'] or 0. for s in streams),
               trials=sum(s['trials'] for s in streams),
               accepted_events=sum(s['accepted_events'] for s in streams),
               overweight_trials=sum(s['overweight_trials'] for s in streams),
               zero_trials=sum(s['zero_trials'] for s in streams),
               maximum_weight_over_envelope=max((s['maximum_weight_over_envelope'] for s in streams), default=0.))
    for name in ('full_tail', 'excess'):
        survey_mass = math.fsum(s['survey_absolute_pb'] * s[name + '_fraction'] for s in positive)
        production_mass = math.fsum(s['production_mean_' + name + '_pb'] or 0. for s in streams)
        variance = math.fsum((s['survey_absolute_pb'] * (s[name + '_fraction_standard_error'] or 0.))**2 for s in positive)
        out['survey_normalized_' + name + '_pb'] = survey_mass
        out['survey_normalized_' + name + '_fraction'] = survey_mass / total_survey if total_survey else None
        out['survey_normalized_' + name + '_fraction_standard_error'] = math.sqrt(variance) / total_survey if total_survey else None
        out['production_' + name + '_pb'] = production_mass
        out['production_' + name + '_fraction'] = production_mass / total_production if total_production else None
        ratio = out['production_' + name + '_fraction']
        if ratio is not None:
            residual_variance = math.fsum((s['production_variance_mean_' + name + '_pb2'] or 0.)
                + ratio**2 * (s['production_variance_mean_absolute_pb2'] or 0.)
                - 2 * ratio * (s['production_covariance_mean_absolute_' + name + '_pb2'] or 0.) for s in streams)
            out['production_' + name + '_fraction_standard_error'] = math.sqrt(max(0., residual_variance)) / total_production
        else:
            out['production_' + name + '_fraction_standard_error'] = None
    out['trial_count_overweight_fraction'] = out['overweight_trials'] / out['trials'] if out['trials'] else None
    out['generation_efficiency'] = out['accepted_events'] / out['trials'] if out['trials'] else None
    return out


def stage_metrics(process):
    out = {}
    for stage in range(3):
        files = sorted(p for p in process.glob('SubProcesses/P*/GF*/log_MINT%d.txt' % stage) if not p.is_symlink())
        total = 0.
        for path in files:
            text = path.read_text()
            matches = re.findall(r'Time spent in Total\s*:\s*(' + NUMBER + ')', text)
            if matches:
                total += number(matches[-1])
            else:
                total += result_file(path.parent / ('res_%d.dat' % stage))['cpu_seconds']
        out[str(stage)] = dict(workers=len(files), cpu_seconds=total)
    return out


def find_final(process, run_name):
    choices = [process / 'Events' / run_name / name for name in ('events.lhe.gz', 'events.lhe')]
    return next(path for path in choices if path.is_file())


def analyze(process, reference, run_name, reference_run_name, trace_name, expected):
    process = process.resolve()
    paths = sorted(process.glob('SubProcesses/P*/GF*/' + trace_name))
    if not paths:
        raise ValueError('No MINT trial diagnostics found under %s' % process)
    by_channel = defaultdict(list)
    workers, worker_events, worker_tail_events = [], Counter(), Counter()
    event_weights = {}
    reference_workers_identical = True
    for path in paths:
        trace = read_trace(path)
        relative = path.parent.relative_to(process)
        parent_name = path.parent.name.split('_', 1)[0]
        channel = str(relative.parent / parent_name)
        by_channel[channel].append(trace)
        events = read_events(path.parent / 'events.lhe')
        if len(events) != trace['events']:
            raise ValueError('Accepted trace events disagree with worker LHE count')
        accepted = [row for row in trace['rows'] if row['accepted']]
        for (digest, weight), row in zip(events, accepted):
            worker_events[digest] += 1
            event_weights[digest] = weight
            if row['tail']:
                worker_tail_events[digest] += 1
        log_text = (path.parent / 'log_MINT2.txt').read_text()
        logged_trials = re.findall(r'another call to the function:\s*(\d+)', log_text)
        logged_tail = sum(map(int, re.findall(r'upper bound failure, (?:novi|virt|born):\s*(\d+)', log_text)))
        if not logged_trials or int(logged_trials[-1]) != len(trace['rows']):
            raise ValueError('Trial trace disagrees with MINT generation counters')
        if logged_tail != sum(r['tail'] for r in trace['rows']):
            raise ValueError('Tail trace disagrees with MINT envelope-failure counters')
        worker = dict(directory=str(relative), trace_sha256=sha256(path),
                      dimensions=trace['dimensions'], configuration=trace['configuration'],
                      survey_nonvirtual_pb=trace['survey_nonvirtual_pb'],
                      survey_virtual_pb=trace['survey_virtual_pb'],
                      proposal_normalizers=trace['proposal_normalizers'],
                      signed_absolute_roundoff=trace['signed_absolute_roundoff'],
                      footer=trace['footer'], streams=[])
        for stream in STREAMS:
            rows = [r for r in trace['rows'] if r['stream'] == stream]
            if rows:
                stream_rate = trace['survey_virtual_pb' if stream == 1 else 'survey_nonvirtual_pb']
                worker['streams'].append(dict(stream=STREAMS[stream], survey_absolute_pb=stream_rate, **summarize(rows)))
        worker.update(combined_streams(worker['streams']))
        if reference is not None:
            reference_events = read_events(reference / relative / 'events.lhe')
            worker['exact_reference_event_sequence_identical'] = events == reference_events
            reference_workers_identical &= events == reference_events
        workers.append(worker)
    channels, all_streams = [], []
    for name, traces in sorted(by_channel.items()):
        survey = traces[0]
        for trace in traces[1:]:
            for key in ('survey_nonvirtual_pb', 'survey_virtual_pb', 'survey_signed_pb'):
                close(trace[key], survey[key], 'split-worker survey ' + key)
        stage_one = result_file(process / name / 'res_1.dat')
        close(stage_one['absolute_pb'], survey['survey_nonvirtual_pb'] + survey['survey_virtual_pb'], 'survey total absolute rate')
        close(stage_one['signed_pb'], survey['survey_signed_pb'], 'survey signed rate')
        rows = [r for trace in traces for r in trace['rows']]
        if any(r['stream'] == 3 for r in rows) and any(r['stream'] == 2 for r in rows):
            raise ValueError('This analyzer requires exclusive Born or nonvirtual streams')
        active_nonvirtual = 3 if any(r['stream'] == 3 for r in rows) else 2
        streams = []
        for stream in (1, active_nonvirtual):
            absolute_rate = survey['survey_virtual_pb' if stream == 1 else 'survey_nonvirtual_pb']
            stream_rows = [r for r in rows if r['stream'] == stream]
            if absolute_rate or stream_rows:
                value = dict(channel=name, stream=STREAMS[stream], survey_absolute_pb=absolute_rate,
                             **summarize(stream_rows))
                streams.append(value)
                all_streams.append(value)
        channel = dict(channel=name, workers=len(traces), streams=streams, **combined_streams(streams))
        channel['survey_signed_pb'] = survey['survey_signed_pb']
        channel['survey_signed_error_pb'] = stage_one['signed_error_pb']
        channel['survey_absolute_error_pb'] = stage_one['absolute_error_pb']
        channels.append(channel)
    final_path = find_final(process, run_name)
    events = read_events(final_path)
    final_events = Counter(digest for digest, weight in events)
    if len(events) != expected or final_events != worker_events:
        raise ValueError('Final sample is not the complete worker event multiset')
    # Equality of the multisets above means all accepted events survived. Thus
    # duplicates cannot ambiguously affect membership or tail-weight totals.
    sum_absolute = math.fsum(abs(weight) for digest, weight in events)
    tail_absolute = math.fsum(abs(event_weights[digest]) * count for digest, count in worker_tail_events.items())
    lhe = dict(path=str(final_path.relative_to(process)), sha256=sha256(final_path),
               events=len(events), negative_events=sum(weight < 0 for digest, weight in events),
               sum_absolute_weights=sum_absolute, sum_signed_weights=math.fsum(w for d, w in events),
               tail_flagged_events=sum(worker_tail_events.values()), tail_flagged_absolute_weight=tail_absolute,
               tail_flagged_absolute_weight_fraction=tail_absolute / sum_absolute,
               minimum_absolute_weight=min(abs(w) for d, w in events),
               maximum_absolute_weight=max(abs(w) for d, w in events),
               exact_worker_event_multiset_preserved=True)
    comparison = None
    if reference is not None:
        reference_final = read_events(find_final(reference, reference_run_name))
        comparison = dict(path=str(reference),
                          exact_worker_event_sequences_identical=reference_workers_identical,
                          exact_final_event_sequence_identical=events == reference_final,
                          exact_final_event_multiset_identical=final_events == Counter(d for d, w in reference_final),
                          cards_identical={name: sha256(process / 'Cards' / name) == sha256(reference / 'Cards' / name)
                                           for name in ('run_card.dat', 'param_card.dat', 'FKS_params.dat')})
    global_metrics = combined_streams(all_streams)
    global_metrics.update(survey_signed_pb=math.fsum(c['survey_signed_pb'] for c in channels),
                          survey_signed_error_pb=math.sqrt(math.fsum(c['survey_signed_error_pb']**2 for c in channels)),
                          survey_absolute_error_pb=math.sqrt(math.fsum(c['survey_absolute_error_pb']**2 for c in channels)),
                          maximum_worker_survey_normalized_full_tail_fraction=max(w['survey_normalized_full_tail_fraction'] for w in workers),
                          maximum_channel_survey_normalized_full_tail_fraction=max(c['survey_normalized_full_tail_fraction'] for c in channels))
    return dict(process=str(process), run_name=run_name, protocol='MG5_MINT_TAIL 1',
                global_metrics=global_metrics, stage_metrics=stage_metrics(process),
                final_lhe=lhe, reference=comparison, channels=channels, workers=workers,
                method=dict(full_tail='Integral of |f| over fABS > pre-rejection envelope, divided by absolute integral.',
                            excess='Integral of max(fABS - envelope, 0), divided by absolute integral: omitted correction mass, not the full tail or a net total-rate bias. The actual LHE retains the survey normalization.',
                            proposal_correction='a = fABS * Z / H; virtual Z = H, nonvirtual Z = product of per-coordinate mean envelope heights.',
                            stream_combination='Pool split workers within each channel/stream, compute separate ratios, then weight by each survey absolute rate once.',
                            production_estimate='Sum independently estimated stream means; uses all attempted trials including rejections and zeros, without pooling unlike stream proposals.',
                            uncertainty='Approximate delta-method statistical error of survey-normalized ratios, conditional on grids, trial counts and survey rates. Does not include survey errors; accepted-event stopping can introduce finite-sample bias.',
                            final_lhe='Actual uncorrected LHE weight fraction from accepted trials tagged above envelope; distinct from the cross-section tail because MINT keeps nominal event magnitudes.',
                            timing='Instrumentation writes every trial. CPU timings include this diagnostic I/O overhead.'))


def self_test():
    # Distinct stream trial counts must not replace their survey mixture.
    def row(a, bound, z=1., accepted=True):
        return dict(observation=a*z/bound, absolute=a, signed=a, bound=bound,
                    normalizer=z, ratio=a/bound, tail=a > bound, accepted=accepted)
    one = summarize([row(2., 1.), row(0., 1., accepted=False)])
    two = summarize([row(.5, 1.) for unused in range(18)])
    one['survey_absolute_pb'], two['survey_absolute_pb'] = 1., 1.
    merged = combined_streams([one, two])
    close(merged['survey_normalized_full_tail_fraction'], .5, 'stream mixture')
    close(merged['survey_normalized_excess_fraction'], .25, 'excess mass')
    close(merged['production_full_tail_fraction'], 2./3., 'sum of stream means')
    close(one['full_tail_fraction'], 1., 'full mass includes the complete above-bound weight')
    close(one['excess_fraction'], .5, 'excess excludes the bound')
    # Equal-volume cells have F=(2,3), H=(1,9), so the envelope proposal
    # samples them with probabilities (.1,.9), and Z=5. The raw sampled
    # F-weighted tail is wrong: proposal-correcting restores 2/(2+3).
    nonflat = [row(2., 1., z=5.)] + [row(3., 9., z=5.) for unused in range(9)]
    corrected = summarize(nonflat)
    close(corrected['full_tail_fraction'], .4, 'nonflat proposal full tail')
    close(corrected['excess_fraction'], .2, 'nonflat proposal excess')
    close(sum(r['absolute'] for r in nonflat if r['tail']) / sum(r['absolute'] for r in nonflat), 2./29., 'incorrect raw-weight mixture diagnostic')
    print('Self-test passed: proposal correction, separate stream mixture and full/excess distinction.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--process', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--run-name', default='benchmark_10k')
    parser.add_argument('--reference-run-name', default='benchmark_10k')
    parser.add_argument('--trace-name', default='mint_overweight_trials.dat')
    parser.add_argument('--expected-events', type=int, default=10000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.process is None or args.output is None:
        parser.error('--process and --output are required')
    result = analyze(args.process, args.reference, args.run_name, args.reference_run_name,
                     args.trace_name, args.expected_events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('global_metrics', 'final_lhe', 'reference')}, indent=2))


if __name__ == '__main__':
    main()
