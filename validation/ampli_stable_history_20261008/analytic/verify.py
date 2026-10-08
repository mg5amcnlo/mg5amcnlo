#!/usr/bin/env python3
"""Revalidate the compressed final pools and reproduce collected shapes."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from madgraph.various import ampli_pool

checksums = {}
records = json.loads((HERE/'replicas/records.json').read_text())
for row in records:
    folder = HERE/'replicas'/('%s_%02d' % (row['variant'], row['replica']))
    for name in ('ampli_pool.dat.gz', 'observables.dat.gz'):
        path = folder/name
        checksums[str(path.relative_to(HERE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='mg5_history_recheck_') as temporary:
        pool_path = Path(temporary)/'ampli_pool.dat'
        with gzip.open(folder/'ampli_pool.dat.gz', 'rb') as source, pool_path.open('wb') as target:
            shutil.copyfileobj(source, target)
        pool = ampli_pool.read_pool(pool_path)
    status = ampli_pool.pool_status([pool], row['quota'])
    assert status == row['tail_checks'], (folder, status)
    assert status['overweight'] < .01
    selection, selected = ampli_pool.select_candidates([pool], row['quota'], random.Random(row['seed'] ^ 0x123567))
    assert selected['selected_tail'] == row['selected_tail'] < .01
    with gzip.open(folder/'observables.dat.gz', 'rt') as stream:
        observations = [tuple(map(float, line.split())) for line in stream]
    assert len(observations) == pool['ncandidates']
    weight_sum = math.fsum(selection.values())
    for column in range(3):
        shape = math.fsum(weight*observations[index][column] for (unused,index),weight in selection.items())/weight_sum
        assert math.isclose(shape, row['final_shapes'][column], rel_tol=1.e-13, abs_tol=1.e-15)
(HERE/'replicas/evidence_sha256.json').write_text(json.dumps(checksums, indent=2)+'\n')
print('Revalidated %d compressed pools, three worker tail checks, final selection tails, and corrected shapes.' % len(records))
