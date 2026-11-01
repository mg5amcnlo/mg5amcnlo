#!/usr/bin/env python3
"""Read-only Wbb4FS first-endpoint identity/probability/scale audit."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import xml.etree.ElementTree as ET


def number(value):
    return float(value.replace('D', 'E').replace('d', 'e'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def blocks(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        event = None
        for line in stream:
            s = line.strip()
            if s == '<event>' or s.startswith('<event '):
                assert event is None
                event = [line]
            elif event is not None:
                event.append(line)
                if s == '</event>':
                    yield event
                    event = None
        assert event is None


def particle(line):
    p = line.split()
    return dict(id=int(p[0]), status=int(p[1]), px=number(p[6]), py=number(p[7]),
                pz=number(p[8]), e=number(p[9]), mass=number(p[10]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--events', type=int, required=True)
    args = parser.parse_args()
    counts = Counter()
    maximum = defaultdict(float)
    weights, matched_weights = [], []
    for event in blocks(args.input):
        xml = ET.fromstring(''.join(event))
        fields = event[1].split()
        n, weight, scalup = int(fields[0]), number(fields[2]), number(fields[3])
        ps = [particle(line) for line in event[2:2+n]]
        kind = int(next(line.split()[1] for line in event if line.startswith('#aMCatNLO')))
        label = {1: 'S', 2: 'H'}[kind]
        counts[label] += 1
        counts[label + '_negative'] += weight < 0
        counts['events'] += 1
        assert math.isfinite(weight) and weight != 0
        assert math.isfinite(scalup) and scalup > 0
        weights.append(weight)
        final = [p for p in ps if p['status'] == 1]
        assert Counter(p['id'] for p in final if abs(p['id']) in (5,24)) == {5:1,-5:1,24:1}
        for p in ps:
            assert all(math.isfinite(p[k]) for k in ('px','py','pz','e','mass'))
            if p['status'] == -1:
                assert p['id'] == 21 or 1 <= abs(p['id']) <= 4
                counts['incoming_' + str(p['id'])] += 1
            if abs(p['id']) == 5:
                maximum['bottom_mass_error_GeV'] = max(maximum['bottom_mass_error_GeV'],abs(p['mass']-4.7))
                assert abs(p['mass'] - 4.7) < 1e-12
                mass2 = p['e']**2-p['px']**2-p['py']**2-p['pz']**2
                scale = max(1.,p['e']**2,p['px']**2+p['py']**2+p['pz']**2)
                relative = abs(mass2-4.7**2)/scale
                maximum['bottom_shell_error_over_energy2'] = max(maximum['bottom_shell_error_over_energy2'],relative)
                assert relative < 1e-10
            elif p['status'] == 1 and p['id'] != 24:
                assert p['id'] == 21 or 1 <= abs(p['id']) <= 4
        hard = xml.find('macnlops_hard_event')
        assert hard is not None and hard.get('source') == 'delta_native_ir' and hard.get('partition') == 'sampled'
        hardlines = [line for line in hard.text.splitlines() if line.strip()]
        nhard = int(hardlines[0].split()[0])
        hardps = [particle(line) for line in hardlines[1:1+nhard]]
        assert len(hardps) == nhard
        assert Counter(p['id'] for p in hardps if p['status'] == 1 and abs(p['id']) in (5,24)) == {5:1,-5:1,24:1}
        assert all(p['id'] == 21 or 1<=abs(p['id'])<=4 for p in hardps if p['status']==-1)
        if label == 'S':
            assert nhard == 5
            assert sorted(p['id'] for p in hardps if p['status']==1) == [-5,5,24]
        component = xml.find('macnlops_component')
        assert component is not None
        componentlines = [line.split() for line in component.text.splitlines() if line.strip()]
        assert len(componentlines) == 2
        source_factor = number(componentlines[1][1])
        assert math.isfinite(source_factor) and 0<=source_factor<=1
        if label == 'H' and weight < 0:
            assert source_factor > 0
        decision = xml.find('macnlops_decision')
        continuation = xml.find('macnlops_continuation')
        scales = xml.find('scales')
        assert decision is not None and continuation is not None and scales is not None
        keep = number(decision.get('keep'))
        assert math.isfinite(keep) and 0<=keep<=1
        counts['rule_' + decision.get('rule')] += 1
        counts['continuation_' + continuation.get('state')] += 1
        for name in ('r','d','k','pt','kernel_scale','delta_source_mean','delta_shower_mean','starting_cdf'):
            if name in decision.attrib:
                assert math.isfinite(number(decision.get(name)))
        for name in ('delta_source_mean','delta_shower_mean','starting_cdf'):
            if name in decision.attrib:
                assert -1e-12 <= number(decision.get(name)) <= 1+1e-12
        cscale = number(continuation.get('scale'))
        maximum['header_vs_continuation_relative'] = max(maximum['header_vs_continuation_relative'],abs(scalup-cscale)/max(scalup,cscale))
        assert abs(scalup-cscale) <= 1e-13*max(scalup,cscale)
        physical = {k:number(v) for k,v in scales.attrib.items() if k.startswith('scalup_')}
        assert physical and all(math.isfinite(v) and v>=0 for v in physical.values())
        for i,p in enumerate(ps,1):
            if abs(p['status'])==1:
                assert physical['scalup_'+str(i)]>0
        zero_pairs=sum(v==0 for k,v in physical.items() if k.count('_')==2)
        counts[label+'_zero_dipole_pairs']+=zero_pairs
        if label == 'H':
            # A zero directed pair is an inert production-scale entry but a
            # genuine FSR stop in stock Pythia; preserve that documented case.
            err = max(abs(v-scalup)/scalup for v in physical.values() if v>0)
            maximum['H_cap_relative_error'] = max(maximum['H_cap_relative_error'],err)
            assert err < 1e-13
        elif continuation.get('state') == 'emitted':
            pt = number(decision.get('pt'))
            err = abs(scalup-pt)/scalup
            maximum['S_winning_scale_relative_error'] = max(maximum['S_winning_scale_relative_error'],err)
            assert err < 1e-13
            assert max(physical.values()) <= scalup*(1+1e-13)
            if decision.get('rule') == 'below_matching_scale':
                counts['protected_below_boundary'] += 1
                assert number(decision.get('pt')) <= number(decision.get('matching_scale'))
        elif continuation.get('state') == 'exhausted':
            assert scalup < min(number(continuation.get('cutoff_isr')),number(continuation.get('cutoff_fsr')))
        rwgt = xml.find('rwgt')
        mac = next(c for c in rwgt if c.get('id') == 'macnlops')
        matched = number(mac.text.strip())
        accepted = decision.get('accepted') == '1'
        assert matched == (weight if accepted else 0.)
        matched_weights.append(matched)
    assert counts['events'] == args.events
    output = dict(input=dict(path=str(args.input),bytes=args.input.stat().st_size,sha256=digest(args.input)),
                  counts=dict(counts),maximum_errors=dict(maximum),expected_events=args.events,
                  sum_source_pb=math.fsum(weights),sum_matched_pb=math.fsum(matched_weights),
                  passed=True,limits='First-endpoint structural/probability/scale checks; not a transfer-closure or native-envelope certificate.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
