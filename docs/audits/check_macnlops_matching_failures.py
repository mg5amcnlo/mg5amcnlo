#!/usr/bin/env python3
"""Replay one saved point per failed matching channel, using temporary drivers.

No exports, source generation, Pythia evolution or accepted-sample publication.
Only the current native driver and counterterm object are compiled, serially;
all generated process sources, cards, objects and samples remain untouched.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from madgraph.various import macnlops as mc
from check_macnlops_component_endpoints import build_driver, command

# These failures occurred after successful native responses, so the old error
# does not include the sequence. The diagnostic replay identified these rows.
SEQUENCES = {
    ('DY_5FS', 'SubProcesses__P0_dxd_emep__GF2.0_24__events'): 1028,
    ('SingleTop_5FS', 'SubProcesses__P0_ub_td__GF1.0_6__events'): 5240,
    ('SingleTop_5FS', 'SubProcesses__P0_ub_td__GF1.0_39__events'): 4240,
}
SAMPLES = [('DY_5FS', 'delta'), ('TTbar_5FS', 'ordinary'),
           ('Wbb_4FS', 'ordinary'), ('SingleTop_5FS', 'ordinary'),
           ('WPlusJet_5FS', 'delta')]


def saved_points(campaign, limit):
    points = []
    for folder, mode in SAMPLES:
        process = campaign / 'processes' / folder
        matching = process / 'Events' / (mode + '_1M') / 'macnlops'
        summary = matching / 'summary.json'
        if not summary.exists():
            continue
        for row in json.loads(summary.read_text()):
            if row['status'] == 'complete':
                continue
            relative = Path(row['input']).relative_to(process).with_suffix('')
            channel = matching / '__'.join(relative.parts)
            error = row['error']
            match = re.search(r'MACNLOPS_DENSITY_V1\s+(\d+)\s+(\d+)', error)
            if match:
                sequence = int(match[1])
            elif 'returned non-zero exit status' in error:
                # Hard-input preflight is checked separately without Pythia.
                continue
            else:
                sequence = SEQUENCES[folder, channel.name]
            points.append((folder, mode, channel, sequence, error))
    if len(points) > limit:
        raise RuntimeError('Saved-point bound exceeded: %d > %d' % (len(points), limit))
    return points


def read_point(channel, sequence):
    with mc.LHEStream(channel / 'proposals.lhe') as events, (channel / 'shower.jsonl').open() as metadata:
        for index, (block, line) in enumerate(zip(events, metadata), 1):
            if index == sequence:
                event = mc.Event(block)
                event.shower_record = json.loads(line)
                event.check_proposal_metadata(event.shower_record, require_continuation=True)
                return event
    raise RuntimeError('Missing saved failure point: %s #%d' % (channel, sequence))


def check(campaign, limit=160):
    points = saved_points(campaign, limit)
    source = ROOT / 'Template/NLO/SubProcesses/montecarlocounter.f'
    report = dict(schema='macnlops-saved-failure-replay-v1', points=[],
                  scope='one saved point per failed density channel; no generation or showering',
                  counterterm_sha256=mc.sha256(source),
                  driver_sha256=mc.sha256(ROOT / 'Template/NLO/SubProcesses/driver_macnlops.f'),
                  python_sha256=mc.sha256(ROOT / 'madgraph/various/macnlops.py'))
    with contextlib.ExitStack() as stack:
        evaluators, binaries = {}, {}
        for folder, mode, channel, sequence, old_error in points:
            process = campaign / 'processes' / folder
            event = read_point(channel, sequence)
            support = event.shower_record
            if folder not in evaluators:
                evaluators[folder] = mc.NativeEvaluator(process, channel, partition='sampled')
            evaluator = evaluators[folder]
            evaluator.configure(support['beam_xmax'], support['shower_cutoffs'])
            native = []

            def query(directory, request, request_id):
                key = (folder, directory.name)
                if key not in binaries:
                    binaries[key] = build_driver(campaign / 'processes', folder,
                        directory.name, stack, trap_delta=True,
                        delta_matching=mode == 'delta', counter_source_text=source.read_text())
                binary, cwd, environment = binaries[key]
                configuration = 'MACNLOPS_CONFIG_V1 ' + ' '.join(
                    '%.17e' % x for x in evaluator.configuration) + '\n'
                output = command([str(binary)], cwd, input=configuration + request, env=environment)
                records = [line.split() for line in output.splitlines()
                           if line.startswith('MACNLOPS_DENSITY_V1 ')]
                if len(records) != 1 or int(records[0][1]) != request_id or int(records[0][2]):
                    raise RuntimeError('Saved failure replay failed: ' + output[-3000:])
                density = mc.Densities(*map(float, records[0][3:]))
                native.append(dict(directory=directory.name, densities=density.__dict__,
                                   request_sha256=hashlib.sha256(request.encode()).hexdigest()))
                return density

            evaluator._query = query
            density = evaluator.evaluate(event, sequence)
            acceptance = (density.hard_acceptance if event.kind == 'H' else density.acceptance)(
                signed_residuals=True)
            report['points'].append(dict(process=folder, mode=mode, channel=channel.name,
                sequence=sequence, kind=event.kind, old_error=old_error, native=native,
                acceptance=acceptance, component=event.matching_component, passed=True))
            print(folder, channel.name, sequence, 'PASS', flush=True)
    report.update(passed=bool(points) and all(p['passed'] for p in report['points']),
                  checked=len(points), numerical_delta_calls=0)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', type=Path, default=ROOT / 'production_1M_sampled')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=160)
    args = parser.parse_args()
    result = check(args.campaign_dir.resolve(), args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(passed=result['passed'], checked=result['checked'])))
    return int(not result['passed'])


if __name__ == '__main__':
    sys.exit(main())
