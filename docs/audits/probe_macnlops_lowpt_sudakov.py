#!/usr/bin/env python3
"""Evaluate source Delta on stored S emissions at the known S starting scale.

Compare the actual per-leg stopping prescription with a common cutoff at the
winning evolution pT, and with independent massless-II backward quadrature.
The native production Delta routines are compiled verbatim by DeltaEvaluator.
"""
import argparse
import ctypes as C
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'tests/validation/delta_noemission'),
               '/nfs/home/rikkert/LHAPDF/lib/python3.12/dist-packages']
from delta_evaluator import DeltaEvaluator


def born_record(event, grouped=True):
    ps = event['original']
    born = dict(ids=[p['id'] for p in ps], colors=[p['colors'] for p in ps],
                p=[p['p'] for p in ps], masses=[p['mass'] for p in ps],
                x=[p['p'][0]/6500. for p in ps[:2]],
                scalup=float(event['continuation']['source_scalup']))
    if grouped:
        groups = {1:[1,3,5], 2:[2,4]}
        born['flavours'] = []
        born['table_ids'] = list(born['ids'])
        for i, pid in enumerate(born['ids']):
            group = groups[1 if abs(pid) in (1,3,5) else 2] if abs(pid)<=5 else [abs(pid)]
            born['flavours'].append([q if pid>0 else -q for q in group] if i<2 else [pid])
            if i<2: born['table_ids'][i] = group[0] if pid>0 else -group[0]
    return born


class Backward:
    def __init__(self, lib, n=24):
        self.lib=C.CDLL(str(lib.resolve()))
        self.lib.lowpt_init.argtypes=[C.c_char_p,C.c_char_p]
        if self.lib.lowpt_init(b'/nfs/home/rikkert/Downloads/pythia8/share/Pythia8/xmldoc',
                              b'LHAPDF6:NNPDF40_nlo_as_01180/0'):
            raise RuntimeError('PDF initialization failed')
        self.lib.lowpt_leg.argtypes=[C.c_int]+[C.c_double]*4+[C.c_int]+[C.POINTER(C.c_double)]*3
        self.nodes,self.weights=np.polynomial.legendre.leggauss(n)
        self.n=n
    def leg(self,pid,x,q,Q,m):
        r=np.zeros(4)
        ptr=lambda a:a.ctypes.data_as(C.POINTER(C.c_double))
        self.lib.lowpt_leg(pid,x,q,Q,m,self.n,ptr(self.nodes),ptr(self.weights),ptr(r))
        return dict(B=math.exp(-r[0]), F=math.exp(-r[1]), C=r[2], R=r[3])
    def born(self,born,q):
        m=2.*6500.*math.sqrt(math.prod(born['x']))
        legs=[self.leg(pid,x,q,born['scalup'],m) for pid,x in zip(born['ids'][:2],born['x'])]
        return dict(B=math.prod(l['B'] for l in legs),
                    FR=math.prod(l['F']*l['R'] for l in legs),
                    FRC=math.prod(l['F']*l['R']*math.exp(l['C']) for l in legs))


def evaluate(key, out, limit):
    import lhapdf
    lhapdf.setVerbosity(0)
    pdf=lhapdf.mkPDF('NNPDF40_nlo_as_01180',0)
    xf=lambda pid,x,q:pdf.xfxQ(pid,x,q) if x<1. else 0.
    with gzip.open(out/f'{key}_samples.json.gz','rt') as stream:
        sample=json.load(stream)
    backward=Backward(out/'backward.so')
    table=Path('/export/slow1/rikkert/macnlops_1M_20261030/processes')/dict(DY='DY_5FS',H='Higgs_HEFT_5FS',ttbar='TTbar_5FS')[key]/'SubProcesses/sudakov.f'
    result=[]
    with DeltaEvaluator(out/f'native_{key}',table=table) as native:
        for group, events in sample['samples'].items():
            if not group.startswith('S_transfer_'): continue
            for e in events[:limit]:
                born=born_record(e)
                real=dict(ids=[p['id'] for p in e['endpoint']],p=[p['p'] for p in e['endpoint']])
                q=float(e['decision']['pt'])
                src=native.evaluate_real(born,real,xf)
                common=native.evaluate(born,q,xf)
                # Change only the PDF flavour sum, retaining the production
                # table's massless representative for the grouped channel.
                definite=dict(born);definite.pop('flavours')
                common_def=native.evaluate(definite,q,xf)
                row=dict(sequence=e['sequence'],group=group,pt=e['pt'],evolution=q,
                         start=born['scalup'],ids=born['ids'][:2],
                         X=e['weight']*(1.-float(e['decision']['keep'])),
                         source_real=src['delta'],source_common=common['delta'],
                         source_common_definite=common_def['delta'],
                         connections=src['connections'],source_legs=src['legs'])
                if key!='ttbar': row.update(backward.born(born,q))
                result.append(row)
            print(key,group,len(events[:limit]),'done',flush=True)
        provenance=native.provenance
    (out/f'{key}_sudakov_probes.json').write_text(json.dumps(dict(rows=result,provenance=provenance),indent=1)+'\n')


def summarize(key,out):
    d=json.loads((out/f'{key}_sudakov_probes.json').read_text())
    groups={r['group'] for r in d['rows']}
    print(key,'bin: mean evolution/physical, src_real/src_common, source/common-backward, src_real/backward')
    for group in sorted(groups,key=lambda s:int(s.rsplit('_',1)[1])):
        rows=[r for r in d['rows'] if r['group']==group]
        w=np.array([r['X'] for r in rows]);w/=w.sum()
        avg=lambda f:sum(a*f(r) for a,r in zip(w,rows))
        print(group, 'N',len(rows),'ratio pTevol/physical',avg(lambda r:r['evolution']/r['pt']),
              'real/common',avg(lambda r:r['source_real']/r['source_common'] if r['source_common'] else 0.),
              'real/B',avg(lambda r:r['source_real']/r['B']) if 'B' in rows[0] else None,
              'common/B',avg(lambda r:r['source_common']/r['B']) if 'B' in rows[0] else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('keys',nargs='+')
    p.add_argument('--output',type=Path,default=Path('/export/slow1/rikkert/macnlops_1M_20261030/investigation_lowpt'))
    p.add_argument('--limit',type=int,default=500)
    p.add_argument('--reuse',action='store_true')
    args=p.parse_args()
    for key in args.keys:
        if not args.reuse:evaluate(key,args.output,args.limit)
        summarize(key,args.output)


if __name__=='__main__':main()
