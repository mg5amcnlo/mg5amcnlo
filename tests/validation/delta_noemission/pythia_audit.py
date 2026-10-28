#!/usr/bin/env python3
"""Audit raw probe records before interpreting their no-emission CDFs."""
import argparse
import collections
import json
import math
from pathlib import Path


def audit(prefix):
    born = [json.loads(line) for line in Path(str(prefix)+'.born.jsonl').open()]
    types = collections.Counter()
    emitted_ids = collections.Counter()
    per_born = collections.Counter()
    max_momentum = max_mass2 = 0.
    ntrials = 0
    for line in Path(str(prefix)+'.trials.jsonl').open():
        trial = json.loads(line)
        assert trial['ok'], trial
        index = trial['event']
        assert trial['trial'] == per_born[index]
        per_born[index] += 1
        ntrials += 1
        types[trial['type']] += 1
        assert 0. <= trial['pt'] <= born[index]['scalup'] * (1.+1.e-6)
        if trial['type'] == 0:
            assert trial['pt'] == 0. and 'real' not in trial
            continue
        assert trial['type'] in (2, 3)
        real = trial['real']
        assert real['ids'][2:4] == [6, -6] and real['emitted'] == 5
        emitted = real['ids'][4]
        assert abs(emitted) <= 5 or emitted == 21, ('non-QCD emission', trial)
        emitted_ids[emitted] += 1
        vectors = real['p']
        assert all(math.isfinite(x) for v in vectors for x in v)
        energy = sum(v[0] for v in vectors[:2])
        mismatch = max(abs(sum(v[j] for v in vectors[:2]) -
                           sum(v[j] for v in vectors[2:])) for j in range(4))
        max_momentum = max(max_momentum, mismatch/energy)
        for v, mass in zip(vectors, real['masses']):
            residual = abs(v[0]**2-sum(x*x for x in v[1:])-mass*mass)
            max_mass2 = max(max_mass2, residual/max(1., v[0]**2))
    assert len(per_born) == len(born) and len(set(per_born.values())) == 1
    assert max_momentum < 1.e-8 and max_mass2 < 1.e-8
    return dict(born_events=len(born), trials=ntrials, repeats=per_born[0],
                emission_types=dict(types), emitted_ids=dict(emitted_ids),
                maximum_relative_momentum_residual=max_momentum,
                maximum_scaled_mass_squared_residual=max_mass2,
                negative_born_weights=sum(b['weight'] < 0. for b in born))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=Path)
    args = parser.parse_args()
    result = audit(args.prefix)
    text = json.dumps(result, indent=2, sort_keys=True)+'\n'
    Path(str(args.prefix)+'.audit.json').write_text(text)
    print(text, end='')
