#!/usr/bin/env python3
"""Read the 30 October first-emission files without changing production data.

Cache the recorded decisions, check against the existing plotting arrays, and
retain deterministic samples of complete events for independent density and
Sudakov checks. Run with KEY=RUN arguments, e.g. DY=run_1M H=run_1M_fix.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np

BASE = Path('/export/slow1/rikkert/macnlops_1M_20261030')
PROCESSES = dict(DY='DY_5FS', H='Higgs_HEFT_5FS', ttbar='TTbar_5FS')
EDGES = (0., .5, 1., 1.5, 2., 3., 5., 7., 10., 15., 20., 50., 100., math.inf)
COLUMNS = ('keep', 'accepted', 'delta', 'r', 'd', 'k', 'kernel_scale',
           'pdf_ratio', 'alpha_s_shower', 'alpha_s_kernel', 'evolution',
           'start', 'x1', 'x2', 'id1', 'id2', 'mass', 'stop1', 'stop2', 'sector')
ATTR = re.compile(r'''([\w_]+)=["']([^"']*)["']''')
HARD = {6, 11, 12, 13, 14, 15, 16, 23, 24, 25}


def attributes(text, tag):
    return dict(ATTR.findall(text.split('<' + tag, 1)[1].split('>', 1)[0]))


def number(s):
    return float(s.replace('D', 'E').replace('d', 'e'))


def particles(lines):
    result = []
    for line in lines:
        f = line.split()
        result.append(dict(id=int(f[0]), status=int(f[1]),
                           colors=list(map(int, f[4:6])),
                           p=[number(f[k]) for k in (9, 6, 7, 8)], mass=number(f[10])))
    return result


def scan(argument, base, out):
    key, run = argument.split('=')
    with np.load(base / 'results' / f'{key}_{run}_lhe.npz') as archive:
        plotting = {k: archive[k] for k in ('pt', 'nominal', 'macnlops', 'is_h')}
    n = len(plotting['pt'])
    rows = np.full((n, len(COLUMNS)), np.nan)
    source = base / 'backup' / f'{key}_{run}_events.lhe.gz'
    rng = random.Random(301030 + list(PROCESSES).index(key))
    samples, seen = {}, {}
    checksum = hashlib.sha256()
    sequence = 0
    with gzip.open(source, 'rt') as stream:
        for line in stream:
            checksum.update(line.encode())
            if not line.startswith('<event'):
                continue
            block = [line]
            for line in stream:
                checksum.update(line.encode())
                block.append(line)
                if '</event>' in line:
                    break
            text = ''.join(block)
            f = block[1].split()
            count, w = int(f[0]), number(f[2])
            ps = particles(block[2:2+count])
            kind = int(text.split('#aMCatNLO', 1)[1].split()[0])
            hard = [p['p'] for p in ps if p['status'] == 1 and abs(p['id']) in HARD]
            momentum = np.sum(hard, axis=0)
            pt = math.hypot(*momentum[1:3])
            assert np.isclose(pt, plotting['pt'][sequence], rtol=1e-12, atol=1e-10)
            assert w == plotting['nominal'][sequence]
            assert (kind == 2) == plotting['is_h'][sequence]
            dec = attributes(text, 'macnlops_decision')
            scales = attributes(text, 'macnlops_source_scales')
            cont = attributes(text, 'macnlops_continuation')
            hard_lines = text.split('<macnlops_hard_event', 1)[1].split('>', 1)[1].split('</', 1)[0].strip().splitlines()
            original = particles(hard_lines[1:])
            component = text.split('<macnlops_component', 1)[1].split('>', 1)[1].split('</', 1)[0].strip().splitlines()
            delta = number(component[1].split()[2])
            values = {k: number(dec[k]) for k in COLUMNS if k in dec}
            values.update(delta=delta, evolution=number(dec.get('pt', '0')),
                          start=number(cont['source_scalup']),
                          x1=original[0]['p'][0]/6500., x2=original[1]['p'][0]/6500.,
                          id1=original[0]['id'], id2=original[1]['id'],
                          mass=math.sqrt(max(0., momentum[0]**2-np.dot(momentum[1:], momentum[1:]))),
                          stop1=number(scales.get('scalup_1_2', 'nan')),
                          stop2=number(scales.get('scalup_2_1', 'nan')))
            rows[sequence] = [values.get(k, math.nan) for k in COLUMNS]
            assert int(dec['accepted']) == (plotting['macnlops'][sequence] != 0.)
            # Independent uniform reservoirs in kind/bin groups. The all-S
            # sample is uniform over all S, regardless of the veto decision.
            bin_index = int(np.searchsorted(EDGES, pt, side='right')-1)
            groups = ['all_S'] if kind == 1 else []
            if dec['rule'] == 's_veto':
                groups.append(f'S_transfer_{bin_index}')
            if kind == 2 and w < 0:
                groups.append(f'H_negative_{bin_index}')
            for group in groups:
                limit = 5000 if group == 'all_S' else 500
                seen[group] = seen.get(group, 0) + 1
                reservoir = samples.setdefault(group, [])
                index = len(reservoir) if len(reservoir) < limit else rng.randrange(seen[group])
                if index < limit:
                    event = dict(sequence=sequence, weight=w, pt=pt, kind=kind,
                                 decision=dec, scales=scales, continuation=cont,
                                 original=original, endpoint=ps,
                                 component=[component[0].split(), component[1].split()])
                    if index == len(reservoir): reservoir.append(event)
                    else: reservoir[index] = event
            sequence += 1
    assert sequence == n
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'{key}_decisions.npz', rows=rows, columns=COLUMNS,
                        pt=plotting['pt'], weight=plotting['nominal'], is_h=plotting['is_h'],
                        matched=plotting['macnlops'])
    with gzip.open(out / f'{key}_samples.json.gz', 'wt') as stream:
        json.dump(dict(seen=seen, samples=samples), stream)
    (out / f'{key}_provenance.json').write_text(json.dumps(dict(source=str(source),
        uncompressed_sha256=checksum.hexdigest(), events=n, run=run,
        checks=['all event weights, S/H types, hard-system pT and accepted flags match plotting arrays']), indent=2)+'\n')
    print(key, n, 'decisions checked', flush=True)
    return key


def summarize(key, out):
    with np.load(out / f'{key}_decisions.npz') as archive:
        z = {k: archive[k] for k in archive.files}
    a = {k: z['rows'][:, i] for i, k in enumerate(z['columns'])}
    w, h, pt = z['weight'], z['is_h'], z['pt']
    x = np.where(~h, w*(1-a['keep']), 0.)
    y = np.where(h & (w < 0), -w*(1-a['keep']), 0.)
    output = []
    print(key, 'pT: expected S removal X, expected H addition Y, X/Y, <Delta>_H, largest 1/Delta share')
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = (pt >= lo) & (pt < hi)
        X, Y = x[s].sum(), y[s].sum()
        undamped = y[s & (y > 0)]/a['delta'][s & (y > 0)]
        Z = undamped.sum()
        ratio = X/Y if Y else None
        row = dict(lo=lo, hi=hi if math.isfinite(hi) else None, X=X, Y=Y,
                   X_error=float(np.linalg.norm(x[s])), Y_error=float(np.linalg.norm(y[s])),
                   ratio=ratio, Z=Z, delta=Y/Z if Z else None,
                   largest_undamped_share=undamped.max()/Z if Z else None,
                   S_realized=float((w-z['matched'])[s & ~h].sum()),
                   H_realized=float((z['matched']-w)[s & h].sum()))
        output.append(row)
        print(f'{lo:g}-{hi:g}: {X:.7g} {Y:.7g}',
              ' '.join('n/a' if q is None else f'{q:.5g}' for q in
                       [ratio, row['delta'], row['largest_undamped_share']]))
    (out / f'{key}_balance.json').write_text(json.dumps(output, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('samples', nargs='+')
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--output', type=Path, default=BASE/'investigation_lowpt')
    p.add_argument('--reuse', action='store_true')
    args = p.parse_args()
    if not args.reuse:
        with ProcessPoolExecutor(max_workers=min(3, len(args.samples))) as pool:
            list(pool.map(scan, args.samples, [args.base]*len(args.samples), [args.output]*len(args.samples)))
    for sample in args.samples:
        summarize(sample.split('=')[0], args.output)


if __name__ == '__main__':
    main()
