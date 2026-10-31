#!/usr/bin/env python3
"""Audit written S/H starts and sample transfer points for stock-Pythia import.

This reads existing first-emission files without changing them. Native probes
use the resulting LHE sample and its sidecar; input weights are the expected
S veto / H removal transfers, corrected for uniform sampling within pT bins.
"""
import argparse
import collections
import gzip
import json
import math
from pathlib import Path
import random
import re

import numpy as np

ATTR = re.compile(r'''([\w_]+)=["']([^"']*)["']''')
EDGES = np.array([0, .5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 10000.])
HARD = {6, 11, 12, 13, 14, 15, 16, 23, 24, 25}


def attrs(block, tag):
    match = re.search(r'<' + tag + r'\b([^<>]*)>', block, re.S)
    if match is None:
        raise ValueError('Missing ' + tag)
    return dict(ATTR.findall(match[1]))


def number(s):
    return float(s.replace('D', 'E').replace('d', 'e'))


def scan(source, output, key, per_bin):
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(310010 + ['DY', 'H', 'ttbar'].index(key))
    counts = collections.Counter()
    maxerr = collections.defaultdict(float)
    groups = collections.Counter()
    samples = collections.defaultdict(list)
    # Each entry: signed transfer, transfer*t, transfer*global_start,
    # transfer*beam1, transfer*beam2, transfer*maxFSR, transfer*(maxStart>t).
    sums = collections.defaultdict(lambda: np.zeros(7))
    header = []
    with gzip.open(source, 'rt') as f:
        for line in f:
            if line.strip() == '<event>':
                break
            header.append(line)
        else:
            raise ValueError('No events')
        sequence = 0
        while line:
            if line.strip() != '<event>':
                line = next(f, '')
                continue
            block = [line]
            for line in f:
                block.append(line)
                if line.strip() == '</event>':
                    break
            text = ''.join(block)
            info = block[1].split()
            n, w, scale = int(info[0]), number(info[2]), number(info[3])
            ps = [row.split() for row in block[2:2+n]]
            sh = 'S' if int(text.split('#aMCatNLO', 1)[1].split()[0]) == 1 else 'H'
            dec = attrs(text, 'macnlops_decision')
            cont = attrs(text, 'macnlops_continuation')
            active = attrs(text, 'scales')
            # Never accidentally select the inert macnlops_source_scales tag.
            active = {k: number(v) for k, v in active.items() if k.startswith('scalup_')}
            counts[sh] += 1
            counts[cont['state']] += 1
            e = abs(scale-number(cont['scale']))/max(scale, 1.)
            maxerr['header_vs_marker'] = max(maxerr['header_vs_marker'], e)
            assert e < 2e-14
            assert len(re.findall(r'<scales\b', text)) == 1
            assert all(0 <= v <= scale*(1+2e-14) for v in active.values())
            for i, p in enumerate(ps, 1):
                if abs(int(p[1])) == 1:
                    assert active['scalup_%d' % i] > 0
            t = number(dec.get('pt', '0'))
            if sh == 'S' and cont['state'] == 'emitted':
                e = abs(scale-t)/max(t, 1.)
                maxerr['S_scalup_vs_winning_evolution'] = max(maxerr['S_scalup_vs_winning_evolution'], e)
                assert e < 2e-14
            if sh == 'H':
                original = attrs(text, 'macnlops_source_scales')
                # The DY transport record may contain a colourless resonance
                # omitted from the flat hard endpoint. Its coloured legs keep
                # their relative order, but their LHE positions are relabelled.
                old_coloured = sorted({int(i) for k in original if k.startswith('scalup_')
                                       for i in k.split('_')[1:] if int(i) > 2})
                new_coloured = [i for i, p in enumerate(ps, 1)
                                if i > 2 and (int(p[4]) or int(p[5]))]
                assert len(old_coloured) == len(new_coloured)
                position = dict(zip(old_coloured, new_coloured))
                position.update({1: 1, 2: 2})
                for k, v in original.items():
                    if not k.startswith('scalup_'):
                        continue
                    k = 'scalup_'+'_'.join(str(position[int(i)]) for i in k.split('_')[1:])
                    expected = min(scale, number(v))
                    e = abs(active[k]-expected)/max(scale, 1.)
                    maxerr['H_pair_vs_source_pair'] = max(maxerr['H_pair_vs_source_pair'], e)
                    assert e < 2e-14
                if any(v == 0 for k, v in active.items() if k.count('_') == 2):
                    counts['H_with_zero_pair'] += 1
            # Reproduce the stock reader's minimal-positive production scale.
            legs = {}
            for k, v in sorted(active.items()):
                pieces = list(map(int, k.split('_')[1:]))
                i = pieces[0]
                if v > 0:
                    legs[i] = min(legs.get(i, v), v)
            pairs = [v for k, v in active.items()
                     if k.count('_') == 2 and int(k.split('_')[1]) > 2]
            keep = number(dec['keep'])
            transfer = (w if sh == 'S' else -w)*(1-keep)
            hard = [p for p in ps if int(p[1]) == 1 and abs(int(p[0])) in HARD]
            pt = math.hypot(sum(number(p[6]) for p in hard), sum(number(p[7]) for p in hard))
            if transfer != 0:
                assert t > 0
                bi = int(np.searchsorted(EDGES, pt, side='right')-1)
                group = '%s_%d' % (sh, bi)
                groups[group] += 1
                sums[group] += transfer*np.array([1, t, scale, legs[1], legs[2],
                                                 max(pairs, default=0.), float(scale > t*(1+1e-8))])
                reservoir = samples[group]
                index = len(reservoir) if len(reservoir) < per_bin else rng.randrange(groups[group])
                if index < per_bin:
                    record = dict(sequence=sequence, kind=sh, bin=bi, pt=pt,
                                  evolution=t, scale=scale, transfer=transfer, block=text)
                    if index == len(reservoir):
                        reservoir.append(record)
                    else:
                        reservoir[index] = record
            sequence += 1
            line = next(f, '')
    selected = sorted((dict(e, sample_factor=groups[g]/len(es))
                       for g, es in samples.items() for e in es), key=lambda e: e['sequence'])
    with (output/f'{key}_sample.lhe').open('w') as out, (output/f'{key}_sample.jsonl').open('w') as side:
        out.write(''.join(header))
        for e in selected:
            out.write(e['block'])
            side.write(json.dumps({k: v for k, v in e.items() if k != 'block'})+'\n')
        out.write('</LesHouchesEvents>\n')
    report = dict(source=str(source),events=sequence,counts=dict(counts),
                  max_relative_errors=dict(maxerr),sampled_events=len(selected),bins={})
    for group, a in sums.items():
        report['bins'][group] = dict(transfer=float(a[0]), events=groups[group],
            mean_evolution=float(a[1]/a[0]),mean_global=float(a[2]/a[0]),
            mean_beam1=float(a[3]/a[0]),mean_beam2=float(a[4]/a[0]),
            mean_max_FSR=float(a[5]/a[0]),fraction_global_above_evolution=float(a[6]/a[0]))
    (output/f'{key}_written_scales.json').write_text(json.dumps(report,indent=2)+'\n')
    print(key, sequence, 'events,',len(selected),'native probe points;',dict(maxerr),flush=True)


def controls(output, key):
    """Diagnostic H-only scale changes; source samples and matching stay fixed."""
    records = [json.loads(line) for line in (output/f'{key}_sample.jsonl').read_text().splitlines()]
    with (output/f'{key}_sample.lhe').open() as stream:
        text = stream.read()
    blocks = re.findall(r'<event>.*?</event>', text, re.S)
    assert len(blocks) == len(records)
    for mode in ['capped', 'common']:
        index = 0

        def transform(match):
            nonlocal index
            record = records[index]
            index += 1
            block = match[0]
            if record['kind'] != 'H':
                return block
            t = record['evolution']
            lines = block.splitlines()
            info = lines[1].split()
            info[3] = '%.17e' % t
            lines[1] = ' '.join(info)
            block = '\n'.join(lines)
            active = attrs(block, 'scales')
            for k, v in active.items():
                active[k] = '%.17e' % (min(t, number(v)) if mode == 'capped' and k.count('_') == 2 else t)
            tag = '<scales '+' '.join('%s="%s"' % pair for pair in active.items())+'>'
            block = re.sub(r'<scales\b[^<>]*>', lambda _: tag, block, count=1)
            marker = attrs(block, 'macnlops_continuation')
            marker['scale'] = '%.17e' % t
            tag = '<macnlops_continuation '+' '.join('%s="%s"' % pair for pair in marker.items())+'/>'
            return re.sub(r'<macnlops_continuation\b[^<>]*>', lambda _: tag, block, count=1)

        variant = re.sub(r'<event>.*?</event>', transform, text, flags=re.S)
        assert index == len(records)
        (output/f'{key}_{mode}.lhe').write_text(variant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('key', choices=['DY', 'H', 'ttbar'])
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-bin', type=int, default=1000)
    parser.add_argument('--controls-only', action='store_true')
    args = parser.parse_args()
    if not args.controls_only:
        scan(args.campaign/'backup'/f'{args.key}_run_1M_v4_events.lhe.gz',
             args.output,args.key,args.per_bin)
    controls(args.output, args.key)


if __name__ == '__main__':
    main()
