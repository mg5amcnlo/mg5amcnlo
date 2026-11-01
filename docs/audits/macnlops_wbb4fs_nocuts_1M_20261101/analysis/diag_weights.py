#!/usr/bin/env python3
"""Add four-weight diagnostics to an uncut Wbb first-emission sample.

Usage: diag_weights.py EVENTS.lhe.gz BACKUP.lhe.gz ARRAYS.npz SUMMARY.json [Wbb]
The original is moved to BACKUP only after the replacement and arrays validate.
Every original signed nominal event is retained. No generation or analysis cuts.
"""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import math
import os
import re
import sys

import numpy as np
from jet_tools import observables,FIELDS

NAMES=('macnlops_veto_only','macnlops_removal_only')
GROUP=("    <weightgroup name='macnlops_diagnostics' combine='none'>\n"+
       ''.join("      <weight id='%s'> %s </weight>\n" % (name,name) for name in NAMES)+
       "    </weightgroup>\n")
WGT=re.compile(r"<wgt\s+id=['\"]macnlops['\"]\s*>([^<]+)</wgt>")
DECISION=re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')

def number(text):
    return float(text.replace('D','E').replace('d','e'))

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            result.update(chunk)
    return result.hexdigest()

def main(source,backup,arrays,summary,key='Wbb'):
    if key.lower()!='wbb':
        raise RuntimeError('This no-cut analysis is only for W+bbbar')
    source,backup,arrays,summary=map(Path,(source,backup,arrays,summary))
    if backup.exists():
        raise RuntimeError('Original-event backup already exists: '+str(backup))
    for path in (backup,arrays,summary):
        path.parent.mkdir(parents=True,exist_ok=True)
    temporary=source.with_name(source.name+'.diagnostics.tmp')
    values=[];nominal=[];matched=[];is_h=[];keeps=[];rules=[];rule_names={}
    counts=collections.Counter()
    with gzip.open(source,'rt') as stream,gzip.open(temporary,'wt',compresslevel=1) as out:
        header=[]
        for line in stream:
            if line.lstrip().startswith('<event'):
                break
            if any(re.search(r"<weight\s+id=['\"]"+name+r"['\"]",line) for name in NAMES):
                raise RuntimeError('Input already contains diagnostic weights')
            header.append(line)
        else:
            raise RuntimeError('Input contains no events')
        text=''.join(header)
        if len(re.findall(r"<weight\s+id=['\"]macnlops['\"]",text))!=1 or text.count('</initrwgt>')!=1:
            raise RuntimeError('Expected one matching weight in one initrwgt')
        out.write(text.replace('</initrwgt>',GROUP+'</initrwgt>'))
        while line:
            if not line.lstrip().startswith('<event'):
                out.write(line);line=next(stream,'');continue
            event=[line]
            event_header=next(stream);event.append(event_header)
            tokens=event_header.split()
            nparticles,nominal_text=int(tokens[0]),tokens[2]
            w=number(nominal_text)
            if not math.isfinite(w) or w==0.:
                raise RuntimeError('Nonfinite or zero nominal event weight')
            final=[]
            for _ in range(nparticles):
                particle=next(stream);event.append(particle);p=particle.split()
                if int(p[1])==1:
                    final.append([int(p[0])]+[number(p[j]) for j in (6,7,8,9)])
            kind=packet=matched_text=None
            for line in stream:
                event.append(line)
                if line.startswith('#aMCatNLO'):
                    kind=int(line.split()[1])
                match=WGT.search(line)
                if match:
                    if matched_text is not None:raise RuntimeError('Duplicate matching event weight')
                    matched_text=match[1].strip()
                if '<macnlops_decision' in line:
                    if packet is not None:raise RuntimeError('Duplicate matching decision')
                    packet=dict(DECISION.findall(line))
                if '</event>' in line:break
            if kind not in (1,2) or packet is None or matched_text is None:
                raise RuntimeError('Missing event kind, matching decision or matching weight')
            m=number(matched_text);keep=number(packet['keep'])
            if m!=0. and not math.isclose(m,w,rel_tol=1.e-7,abs_tol=0.):
                raise RuntimeError('Matching weight is neither original weight nor zero')
            if not math.isfinite(keep) or not 0.<=keep<=1.:
                raise RuntimeError('Invalid keep probability')
            v=observables(final)
            veto,removal=(matched_text,nominal_text) if kind==1 else (nominal_text,matched_text)
            written=0
            for line in event:
                out.write(line)
                if WGT.search(line):
                    indent=line[:len(line)-len(line.lstrip())]
                    out.write("%s<wgt id='%s'> %s </wgt>\n" % (indent,NAMES[0],veto))
                    out.write("%s<wgt id='%s'> %s </wgt>\n" % (indent,NAMES[1],removal))
                    written+=1
            assert written==1
            values.append(v);nominal.append(w);matched.append(m);is_h.append(kind==2);keeps.append(keep)
            rule='%s:%s' % ('H' if kind==2 else 'S',packet['rule'])
            rules.append(rule_names.setdefault(rule,len(rule_names)))
            counts['events']+=1;counts['H' if kind==2 else 'S']+=1
            if w<0:counts['negative_H' if kind==2 else 'negative_S']+=1
            line=next(stream,'')
    nominal,matched,is_h,keep=map(np.asarray,(nominal,matched,is_h,keeps))
    observed=np.asarray(values);veto=np.where(is_h,nominal,matched);removal=np.where(is_h,matched,nominal)
    assert np.array_equal(veto+removal,nominal+matched)
    np.savez_compressed(arrays,source_index=np.arange(len(nominal)),observables=observed,
        observable_names=np.asarray(FIELDS),pt=observed[:,0],nominal=nominal,macnlops=matched,
        is_h=is_h,keep=keep,rule=np.asarray(rules,dtype=np.int16),rule_names=np.asarray(list(rule_names)))
    report=dict(process='Wbb',cuts='none',jet_algorithm='anti-kT R=0.5, no jet threshold',
        primary='pT(W+b+bbar), using one stable massive b and bbar',events=len(nominal),counts=dict(counts),
        normalization='Original signed weights; no survivor, selection or range normalization',
        observable_fields=FIELDS,
        totals_pb={name:float(x.sum()) for name,x in dict(source=nominal,veto=veto,removal=removal,matched=matched).items()},
        expected_matched_pb=float(np.dot(nominal,keep)),
        matching_draw_variance_pb2=float(np.dot(nominal*nominal,keep*(1-keep))),
        missing_observables={name:int(np.count_nonzero(~np.isfinite(observed[:,i]))) for i,name in enumerate(FIELDS)},
        exact_zero_observables={name:int(np.count_nonzero(observed[:,i]==0.)) for i,name in enumerate(FIELDS)},
        classes={})
    for label,mask in [('S_positive',~is_h&(nominal>0)),('S_negative',~is_h&(nominal<0)),
                       ('H_positive',is_h&(nominal>0)),('H_negative',is_h&(nominal<0))]:
        report['classes'][label]=dict(events=int(mask.sum()),source_pb=float(nominal[mask].sum()),
            source_absolute_pb=float(np.abs(nominal[mask]).sum()),matched_pb=float(matched[mask].sum()),
            expected_matched_pb=float(np.dot(nominal[mask],keep[mask])))
    report['arrays_sha256']=sha(arrays)
    report['original_input_sha256']=sha(source)
    report['diagnostic_input_sha256']=sha(temporary)
    os.replace(source,backup)
    os.replace(temporary,source)
    summary.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('classes','missing_observables','exact_zero_observables')},indent=2))

if __name__=='__main__':
    main(*sys.argv[1:])
