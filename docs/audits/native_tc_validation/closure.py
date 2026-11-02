#!/usr/bin/env python3
"""Independent per-job inclusive closure at the actual HwU token precision.

No event reads or physics runs: compares saved first-stage chunk aggregates
with native full-shower totals and exact source-class counters.
"""
import argparse
from decimal import Decimal
import gzip
import json
from pathlib import Path
import re

import numpy as np
from run import DEFAULT_BASE, PROCESSES, dump, sha

NAMES=('central value','macnlops','macnlops_veto_only','macnlops_removal_only')


def total_tokens(path):
    names=None
    lines=iter(path.read_text().splitlines())
    for line in lines:
        if line.startswith('##&'):names=[x.strip() for x in line[3:].split('&')]
        if line.startswith('<histogram>') and re.search(r'"\s*total \|',line):
            tokens=next(lines).split()
            if names is None:raise RuntimeError('Missing native HwU labels')
            return [tokens[names.index(name)] for name in NAMES]
    raise RuntimeError('Missing native total histogram')


def token_half_spacing(token):
    # Decimal preserves printed trailing digits and the native exponent.
    value=Decimal(token.replace('D','E').replace('d','e'))
    return float(Decimal(5).scaleb(value.as_tuple().exponent-1))


def check(base,p,run):
    result=base/'results'/p['key']/run
    first=json.loads((result/'first.json').read_text())
    full=json.loads((result/'full.json').read_text())
    with gzip.open(result/'first_histograms.json.gz','rt') as f:hist=json.load(f)
    expected=np.asarray(hist['histograms']['total']['blocks'])[:,1,:]
    jobs=full['bindings']
    if len(jobs)!=len(expected):raise RuntimeError('First chunks and full jobs differ')
    records=[];largest_scaled=0.
    for i,binding in enumerate(jobs):
        path=Path(binding['path']);tokens=total_tokens(path)
        values=np.asarray([float(t.replace('D','E')) for t in tokens])
        tolerance=np.asarray([token_half_spacing(t) for t in tokens])
        # Allow native accumulation roundoff at a level far below printed precision.
        roundoff=1.e-10*np.maximum(1.,np.abs(expected[i]))
        difference=values-expected[i];allowed=tolerance+roundoff
        passed=bool(np.all(np.abs(difference)<=allowed))
        largest_scaled=max(largest_scaled,float(np.max(np.abs(difference)/allowed)))
        records.append(dict(job=i+1,path=str(path),sha256=sha(path),printed_tokens=tokens,
            first_expected_pb=expected[i].tolist(),full_printed_pb=values.tolist(),
            differences_pb=difference.tolist(),half_printed_spacing_pb=tolerance.tolist(),
            accumulation_tolerance_pb=roundoff.tolist(),passed=passed))
    count_names=('events','S','H','negative_S','negative_H',
                 'retained_negative_S','retained_negative_H','zero_matched')
    count_checks={name:dict(first=first['counts'].get(name,0),full=full['counts'].get(name,0),
        passed=first['counts'].get(name,0)==full['counts'].get(name,0)) for name in count_names}
    report=dict(process=p['key'],run=run,weights=NAMES,jobs=records,count_checks=count_checks,
        largest_fraction_of_allowed_print_rounding=largest_scaled,
        first_summary_sha256=sha(result/'first.json'),full_summary_sha256=sha(result/'full.json'),
        method='Compare paired contiguous first-stage chunk totals with each native shower-job total. Half-spacing is read from the actual decimal token, with a separate small accumulation allowance.',
        passed=all(x['passed'] for x in records) and all(x['passed'] for x in count_checks.values()))
    dump(result/'inclusive_closure.json',report)
    if not report['passed']:raise RuntimeError('Per-job inclusive closure failed')
    print('PASS',p['key'],run,'per-job token precision and all signed class counts',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--process',nargs='+',choices=[p['key'] for p in PROCESSES],default=['EE2j','DY'])
    parser.add_argument('--run',default='run_1M_native_tc');a=parser.parse_args()
    for p in PROCESSES:
        if p['key'] in a.process:check(a.base,p,a.run)


if __name__=='__main__':main()
