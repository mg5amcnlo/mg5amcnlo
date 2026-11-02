#!/usr/bin/env python3
"""Stream a native-t_c first-emission LHE and audit its retained source packets.

This checks recorded contracts, not the physics of the Sudakov or a Pythia
import.  In particular a positive-H shortcut does not record native t_c, and
the sampled source starting scale is not stored in the component packet.
"""

import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


DIAGNOSTIC_IDS = (
    'macnlops_veto_only', 'macnlops_removal_only',
    'native_tc_source_S', 'native_tc_source_H',
)
PROBABILITIES = {
    'keep', 'marking', 'delta_source_mean', 'delta_shower_mean',
    'starting_cdf',
}
NONNEGATIVE = {
    'pt', 'matching_scale', 'born_scale_factor', 'born_coupling_factor',
    'born_pdf_factor', 'kernel_scale', 'alpha_s_shower', 'alpha_s_kernel',
    'pdf_ratio',
}


def number(value):
    return float(value.replace('D', 'E').replace('d', 'e'))


def close(a, b):
    return abs(a - b) <= 2.e-12 * max(1., abs(a), abs(b))


class Sum:
    """Bounded-memory compensated block summation."""

    def __init__(self):
        self.block = []
        self.partials = []

    def add(self, value):
        self.block.append(value)
        if len(self.block) == 4096:
            self.partials.append(math.fsum(self.block))
            self.block.clear()

    def value(self):
        return math.fsum(self.partials + [math.fsum(self.block)])


def confirm_input_snapshot(path, initial_stat, scanned_digest):
    """Confirm content if the shower workflow recompressed the input in place."""
    def identity(info):
        return info.st_ino, info.st_size, info.st_mtime_ns

    final_stat = path.stat()
    stable = identity(initial_stat) == identity(final_stat)
    result = {'unchanged_during_scan': stable,
              'final_bytes': final_stat.st_size,
              'final_mtime_ns': final_stat.st_mtime_ns}
    if not stable:
        digest = hashlib.sha256()
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), ''):
                digest.update(chunk.encode('utf-8'))
        result['final_sha256_uncompressed'] = digest.hexdigest()
        result['final_digest_matches_scan'] = digest.hexdigest() == scanned_digest
        result['unchanged_during_confirmation'] = identity(final_stat) == identity(path.stat())
    result['confirmed'] = stable or (result['final_digest_matches_scan']
                                    and result['unchanged_during_confirmation'])
    return result


def audit(path, expected_events=None, label=None, roundtrip_controls=None):
    path = Path(path).resolve()
    before = path.stat()
    failures = collections.Counter()
    examples = []
    counts = collections.Counter()
    rules = collections.Counter()
    rules_by_source_sign = collections.defaultdict(collections.Counter)
    states = collections.Counter()
    registries = collections.Counter()
    ranges = {}
    totals = collections.defaultdict(Sum)
    max_errors = collections.defaultdict(float)
    digest = hashlib.sha256()
    header_lines = []
    controls = ({int(row['event']): row for row in roundtrip_controls['cases']}
                if roundtrip_controls else {})
    controlled_roundtrips = []

    def check(ok, name, event, detail=None):
        if not ok:
            failures[name] += 1
            if len(examples) < 25:
                examples.append({'event': event, 'check': name, 'detail': detail})

    def extent(name, value):
        if name not in ranges:
            ranges[name] = [value, value]
        else:
            ranges[name][0] = min(ranges[name][0], value)
            ranges[name][1] = max(ranges[name][1], value)

    opener = gzip.open if path.suffix == '.gz' else open
    block = []
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            digest.update(line.encode('utf-8'))
            if line.strip() == '<event>':
                check(not block, 'nested_event', counts['events'] + 1)
                block = [line]
            elif block:
                block.append(line)
                if line.strip() != '</event>':
                    continue
                counts['events'] += 1
                index = counts['events']
                text = ''.join(block)
                block = []
                try:
                    root = ET.fromstring(text)
                    head = root.text.strip().splitlines()
                    row = head[0].split()
                    n = int(row[0])
                    weight, scalup = number(row[2]), number(row[3])
                    particles = [r.split() for r in head[1:n + 1]]
                    kind_match = re.search(r'^\s*#aMCatNLO\s+([12])\s', text, re.M)
                    kind = 'S' if kind_match.group(1) == '1' else 'H'
                    hard, = root.findall('macnlops_hard_event')
                    comp, = root.findall('macnlops_component')
                    decision, = root.findall('macnlops_decision')
                    continuation, = root.findall('macnlops_continuation')
                    scales, = root.findall('scales')
                    source_scales, = root.findall('macnlops_source_scales')
                    endpoint, = root.findall('macnlops_endpoint')
                    weights = {e.get('id'): number(e.text.strip())
                               for e in root.findall('./rwgt/wgt')}
                    matched = weights['macnlops']
                    hard_rows = hard.text.strip().splitlines()
                    meta = list(map(int, hard_rows[0].split()))
                    hard_particles = [r.split() for r in hard_rows[1:]]
                    comp_rows = comp.text.strip().splitlines()
                    comp_ids = list(map(int, comp_rows[0].split()))
                    flow_probability, insertion, source_p, symmetry = map(
                        number, comp_rows[1].split())
                    keep = number(decision.get('keep'))
                    accepted = int(decision.get('accepted'))
                    scale = number(continuation.get('scale'))
                    state = continuation.get('state')
                    scale_values = {k: number(v) for k, v in scales.attrib.items()
                                    if k.startswith('scalup_')}
                except (AttributeError, ValueError, IndexError, ET.ParseError, KeyError) as error:
                    check(False, 'record_parse', index, str(error))
                    continue

                counts[kind] += 1
                counts[kind + '_negative_source'] += weight < 0
                counts[kind + '_negative_retained'] += matched < 0
                counts[kind + '_accepted'] += accepted == 1
                counts[kind + '_removed'] += accepted == 0
                rules[decision.get('rule')] += 1
                source_sign = kind + ('_negative' if weight < 0 else '_positive')
                rules_by_source_sign[source_sign][decision.get('rule')] += 1
                if meta[7] < 0:
                    counts[kind + '_signed_flow_marker'] += 1
                states[state] += 1
                registries[comp.get('registry')] += 1
                for key, value in [('source', weight), ('matched', matched),
                                   (kind + '_source', weight), (kind + '_matched', matched),
                                   ('expected_matched', weight * keep)]:
                    totals[key].add(value)
                check(math.isfinite(weight) and weight != 0, 'nonzero_finite_source_weight', index)
                check(math.isfinite(matched), 'finite_matched_weight', index)
                check(accepted in (0, 1), 'binary_acceptance', index)
                check(matched == (weight if accepted else 0.), 'accepted_weight', index)
                check(0 <= keep <= 1 and math.isfinite(keep), 'keep_probability', index)
                if decision.get('rule') in ('pdf_support_no_transfer', 'born_scale_no_transfer'):
                    check(keep == 1 and accepted == 1 and matched == weight,
                          'paired_support_guard_keeps_source', index)
                check(hard.get('source') == 'delta_native_tc', 'native_tc_source', index)
                check(hard.get('version') == '1' and hard.get('frame') == 'lab', 'hard_format', index)
                check(hard.get('partition') == 'sampled', 'sampled_partition', index)
                check(len(meta) == 14 and len(hard_particles) == meta[0], 'hard_packet_size', index)
                check(meta[1] == (1 if kind == 'S' else 2), 'source_kind', index)
                check(meta[8] == 0 and meta[9] == 1 and meta[11] == 1,
                      'fixed_flow_no_KL_coalesced', index, meta[8:12])
                check(len(comp_ids) == 4 and comp_ids[0] == meta[3]
                      and comp_ids[3] == abs(meta[7]), 'component_native_labels', index)
                check(kind == 'H' or meta[7] > 0, 'signed_flow_marker_only_H', index)
                check(bool(re.fullmatch(r'[0-9a-f]{64}', comp.get('registry', ''))),
                      'component_registry', index)
                for name, value in [('flow_probability', flow_probability),
                                    ('insertion', insertion), ('source_factor', source_p)]:
                    check(math.isfinite(value) and 0 <= value <= 1, name, index, value)
                    extent(name, value)
                check(math.isfinite(symmetry) and symmetry > 0, 'diagram_symmetry', index)
                if kind == 'S':
                    check(insertion == 0 and source_p == 1, 'S_source_unit_factor', index)
                for key, raw in decision.attrib.items():
                    if key in PROBABILITIES or key in NONNEGATIVE or key in ('r', 'd', 'k'):
                        value = number(raw)
                        check(math.isfinite(value), 'finite_' + key, index)
                        if key in PROBABILITIES:
                            check(0 <= value <= 1, 'range_' + key, index, value)
                        if key in NONNEGATIVE:
                            check(value >= 0, 'nonnegative_' + key, index, value)
                        extent(key, value)

                check(math.isfinite(scale) and scale > 0, 'positive_continuation', index)
                check(close(scalup, scale), 'global_scalup', index)
                max_errors['global_scalup'] = max(max_errors['global_scalup'], abs(scalup - scale))
                check(len(scale_values) >= n, 'particle_scale_count', index)
                check(all('scalup_' + str(i) in scale_values for i in range(1, n + 1)),
                      'particle_scale_anchors', index)
                check(all(math.isfinite(v) and 0 <= v <= scale + 2.e-12 * max(1., scale)
                          for v in scale_values.values()), 'finite_bounded_dipole_scales', index)
                check(all(scale_values.get('scalup_' + str(i), 0) > 0 for i in range(1, n + 1)),
                      'positive_particle_anchors', index)
                extent('continuation_scale', scale)
                if kind == 'H':
                    check(state == 'unshowered' and endpoint.get('origin') == 'hard_annotation',
                          'H_unshowered', index)
                    check(n == meta[0], 'H_particle_count', index)
                    check(all(v == scale for v in scale_values.values() if v > 0),
                          'H_common_positive_caps', index)
                    source_caps = [number(v) for k, v in source_scales.attrib.items()
                                   if k.startswith('scalup_') and number(v) > 0]
                    check(bool(source_caps) and all(v == scale for v in source_caps),
                          'H_source_to_continuation_cap_exact', index)
                    check(len(particles) == len(hard_particles) and all(
                        a[:6] == b[:6] and len(a) == len(b) == 13 and all(
                            close(number(x), number(y)) for x, y in zip(a[6:], b[6:]))
                        for a, b in zip(particles, hard_particles)), 'H_endpoint_preserved', index)
                    if decision.get('pt') is not None:
                        counts['H_native_pt_recorded'] += 1
                        native_pt = number(decision.get('pt'))
                        max_errors['H_native_pt_absolute'] = max(
                            max_errors['H_native_pt_absolute'], abs(native_pt-scale))
                        max_errors['H_native_pt_relative'] = max(
                            max_errors['H_native_pt_relative'],
                            abs(native_pt-scale)/max(1., native_pt, scale))
                        agrees = close(native_pt, scale)
                        if not agrees and index in controls:
                            control = controls[index]
                            envelope = control['envelopes'][str(control.get('selected_ulps', 1))]
                            agrees = (native_pt == control['nominal'] == control['decision']
                                      and scale == control['source_cap']
                                      and envelope['low'] <= scale <= envelope['high'])
                            if agrees:
                                counts['H_native_pt_roundtrip_controlled'] += 1
                                controlled_roundtrips.append(index)
                        check(agrees, 'H_native_pt_cap', index)
                    else:
                        counts['H_native_pt_not_recorded'] += 1
                elif state == 'emitted':
                    check(endpoint.get('origin') == 'shower' and n == meta[0] + 1,
                          'S_emitted_endpoint', index)
                    if decision.get('pt') is not None:
                        counts['S_winner_pt_recorded'] += 1
                        check(close(number(decision.get('pt')), scale), 'S_winner_scale', index)
                    else:
                        counts['S_winner_pt_not_recorded'] += 1
                elif state == 'exhausted':
                    check(n == meta[0], 'S_exhausted_particle_count', index)
                    check(scale < number(continuation.get('cutoff_isr'))
                          and scale < number(continuation.get('cutoff_fsr')), 'S_exhausted_cutoff', index)
                else:
                    check(False, 'S_continuation_state', index, state)

                diag = [key in weights for key in DIAGNOSTIC_IDS]
                if any(diag):
                    counts['diagnostic_weight_records'] += 1
                    check(all(diag), 'complete_diagnostic_weights', index)
                    if all(diag):
                        expected = [matched if kind == 'S' else weight,
                                    matched if kind == 'H' else weight,
                                    weight if kind == 'S' else 0., weight if kind == 'H' else 0.]
                        for key, value in zip(DIAGNOSTIC_IDS, expected):
                            check(weights[key] == value, 'diagnostic_' + key, index)
                            totals[key].add(weights[key])
            elif counts['events'] == 0:
                header_lines.append(line)

    check(not block, 'closed_final_event', counts['events'])
    if expected_events is not None:
        check(counts['events'] == expected_events, 'expected_event_count', 0, counts['events'])
    header = ''.join(header_lines)
    marker = re.search(r'<macnlops_first_emission\b[^>]*?/?>', header, re.S)
    first = ET.fromstring(marker.group(0)) if marker else None
    check(first is not None, 'first_emission_header', 0)
    if first is not None:
        check(first.get('prescription') == 'component-sudakov-ratio-v4', 'v4_prescription', 0)
        check(first.get('born_scale_correction') == 'native-born-v1', 'native_Born_correction', 0)
        check(first.get('weight') == 'macnlops' and first.get('rejected') == 'kept', 'retained_rejections', 0)
    snapshot = confirm_input_snapshot(path, before, digest.hexdigest())
    check(snapshot['confirmed'], 'input_snapshot_confirmed', 0)
    if roundtrip_controls:
        check(digest.hexdigest() == roundtrip_controls['input_sha256_uncompressed'],
              'roundtrip_control_input_sha256', 0)
    return {
        'schema': 'native-tc-event-audit-v1', 'label': label, 'input': str(path),
        'input_bytes': before.st_size, 'input_mtime_ns': before.st_mtime_ns,
        'input_snapshot': snapshot,
        'sha256_uncompressed': digest.hexdigest(), 'passed': not failures,
        'counts': dict(counts), 'rules': dict(rules), 'states': dict(states),
        'rules_by_source_sign': {k: dict(v) for k, v in rules_by_source_sign.items()},
        'registries': dict(registries), 'ranges': ranges,
        'weight_sums': {key: value.value() for key, value in totals.items()},
        'max_absolute_errors': dict(max_errors), 'failures': dict(failures),
        'failure_examples': examples,
        'controlled_native_roundtrips': controlled_roundtrips,
        'roundtrip_controls': ({k: v for k, v in roundtrip_controls.items() if k != 'cases'}
                              if roundtrip_controls else None),
        'first_emission_header': first.attrib if first is not None else None,
        'limitations': [
            'Source probability is checked for recorded identity/support, not recomputed: sampled Q is not stored.',
            'Common H caps are checked; a native t_c comparison requires a recorded decision pt or an independent provider probe.',
            'S continuation equals the recorded winning pt; independent Pythia import and competing-trial verification are separate probes.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--events', type=int)
    parser.add_argument('--label')
    parser.add_argument('--roundtrip-controls', type=Path,
                        help='Independent native precision controls, bound to this input SHA')
    args = parser.parse_args()
    controls = None
    if args.roundtrip_controls:
        controls = json.loads(args.roundtrip_controls.read_text())
        controls['control_file'] = str(args.roundtrip_controls.resolve())
        controls['control_file_sha256'] = hashlib.sha256(args.roundtrip_controls.read_bytes()).hexdigest()
    result = audit(args.input, args.events, args.label, controls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: result[key] for key in ('label', 'passed', 'counts', 'failures')}, sort_keys=True))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
