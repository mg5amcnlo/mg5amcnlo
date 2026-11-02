#!/usr/bin/env python3
"""Record current canonical payload hashes separately from gzip encoding hashes."""
import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from run import DEFAULT_BASE,PROCESSES,dump,sha
from specification import WEIGHTS,ADDED_IDS

WGT=re.compile(r'''<wgt\s+id=['"]([^'"]+)['"]\s*>([^<]+)</wgt>''')


def scan_payload(path,first):
    digest=hashlib.sha256();sequence=hashlib.sha256();size=0
    counts=Counter();totals=[0.]*6;inside=False;header=False
    with gzip.open(path,'rb') as stream:
        for raw in stream:
            digest.update(raw);size+=len(raw);line=raw.decode()
            if line.lstrip().startswith('<event'):
                inside=True;header=True;kind=None;weights={};continue
            if not inside:continue
            if header:
                nominal=float(line.split()[2].replace('D','E').replace('d','e'))
                header=False;continue
            if line.startswith('#aMCatNLO'):kind=int(line.split()[1])
            for match in WGT.finditer(line):weights[match[1]]=float(match[2].replace('D','E').replace('d','e'))
            if '</event>' not in line:continue
            if kind not in (1,2) or any(name not in weights for name in ('macnlops',*ADDED_IDS)):
                raise RuntimeError('Missing immutable source class or diagnostic weights')
            m=weights['macnlops'];label='S' if kind==1 else 'H'
            values=[nominal,m,*(weights[name] for name in ADDED_IDS)]
            expected=[nominal,m,m if kind==1 else nominal,nominal if kind==1 else m,
                      nominal if kind==1 else 0.,nominal if kind==2 else 0.]
            if not all(math.isfinite(x) for x in values) or any(x!=y for x,y in zip(values,expected)):
                raise RuntimeError('Per-event diagnostic weight identity/classification failed')
            if m!=0. and not math.isclose(m,nominal,rel_tol=1.e-7):raise RuntimeError('Invalid retained matching weight')
            counts['events']+=1;counts[label]+=1;counts['negative_'+label]+=nominal<0
            counts['retained_'+label]+=m!=0;counts['retained_negative_'+label]+=m<0
            counts['zero_matched']+=m==0
            for i,value in enumerate(values):totals[i]+=value
            sequence.update((str(kind)+' '+','.join(value.hex() for value in values)+'\n').encode())
            inside=False
    if inside:raise RuntimeError('Truncated canonical event')
    checks={key:dict(current=counts.get(key,0),first=value,passed=counts.get(key,0)==value)
            for key,value in first['counts'].items()}
    weight_checks={name:dict(current=totals[i],first=first['totals_pb'][name],
        passed=math.isclose(totals[i],first['totals_pb'][name],rel_tol=1.e-11,abs_tol=1.e-11))
        for i,name in enumerate(WEIGHTS)}
    passed=all(x['passed'] for x in checks.values()) and all(x['passed'] for x in weight_checks.values())
    return digest.hexdigest(),size,dict(events=counts['events'],weight_sequence_sha256=sequence.hexdigest(),
        sequence_definition='Per event: immutable S/H kind followed by six IEEE float.hex signed weights in source/matched/veto/removal/source_S/source_H order.',
        count_checks=checks,weight_sum_checks=weight_checks,per_event_identities_passed=True,passed=passed)


def audit(base,p,run):
    result=base/'results'/p['key']/run
    first=json.loads((result/'first.json').read_text())
    path=base/'processes'/p['directory']/'Events'/run/'events.lhe.gz'
    payload,size,checks=scan_payload(path,first)
    if not checks['passed']:raise RuntimeError('Current diagnostic sequence/counts do not match first.json; refusing legacy digest adoption')
    current=sha(path)
    dump(result/'encoding_audit.json',dict(schema='native-tc-encoding-audit-v2',process=p['key'],run=run,path=str(path),
        first_diagnostic_gzip_sha256=first['diagnostic_input_sha256'],
        current_gzip_sha256=current,current_uncompressed_sha256=payload,
        current_uncompressed_bytes=size,
        first_summary_sha256=sha(result/'first.json'),weight_sequence_audit=checks,
        gzip_encoding_matches_first=current==first['diagnostic_input_sha256'],
        reason='The native shower gunzips and regzips its canonical input; container encoding can change without changing LHE payload.',
        resume_policy='New diagnostics verify the saved payload digest. Legacy adoption requires this exact first-summary binding and successful weight-sequence/count checks; historical first.json is not rewritten.',
        evidence_limit='No pre-shower uncompressed hash was saved in this frozen epoch. This current payload hash is recorded for future comparisons; per-job weight/count closure separately verifies signed accounting but is not historical byte-sequence proof.'))
    print('Recorded payload and encoding:',p['key'],run,'same gzip:',current==first['diagnostic_input_sha256'],flush=True)


def main():
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--process',nargs='+',choices=[p['key'] for p in PROCESSES],default=['EE2j','DY'])
    parser.add_argument('--run',default='run_1M_native_tc');a=parser.parse_args()
    for p in PROCESSES:
        if p['key'] in a.process:audit(a.base,p,a.run)


if __name__=='__main__':main()
