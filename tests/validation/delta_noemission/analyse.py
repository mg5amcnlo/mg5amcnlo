#!/usr/bin/env python3
"""Paired comparison of branch Delta with independent Pythia first emissions."""
import argparse
import csv
import ctypes as C
import hashlib
import json
import re
from pathlib import Path
import subprocess
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from delta_evaluator import DeltaEvaluator, ROOT

Q = np.array([2.,3.,5.,7.,10.,15.,20.,30.,40.,50.,75.,100.,150.,200.,300.,500.,700.])
EDGES = np.array([1.,2.,5.,10.,20.,30.,50.,75.,100.,150.,200.,300.,500.,1000.])


def grouped(b):
    b=dict(b)
    # Undo the LHE writer's incoming shower-mass reshuffling for hard Delta.
    hard=np.asarray(b['p'],dtype=float).copy()
    total=hard[2:].sum(axis=0)
    ea=(total[0]+total[3])/2; eb=(total[0]-total[3])/2
    hard[0]=[ea,0,0,ea]; hard[1]=[eb,0,0,-eb]
    b['p']=hard.tolist(); b['masses']=[0.,0.,*b['masses'][2:]]
    b['x']=[ea/6500.,eb/6500.]
    if b['ids'][0]!=21:
        sign=1 if b['ids'][0]>0 else -1
        b['table_ids']=[sign*2,-sign*2,6,-6]
        b['flavours']=[[sign*i for i in (2,1,3,4,5)],[-sign*i for i in (2,1,3,4,5)],[6],[-6]]
    return b


def csvwrite(path, rows):
    with open(path,'w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def calculate(a):
    prefix=Path(a.prefix)
    born=[json.loads(l) for l in Path(str(prefix)+'.born.jsonl').open()]
    born=[grouped(b) for b in born]
    lib=C.CDLL(a.pdf_library)
    lib.pythia_pdf_init.argtypes=[C.c_char_p,C.c_char_p]
    lib.pythia_pdf_xf.argtypes=[C.c_int,C.c_double,C.c_double]
    lib.pythia_pdf_xf.restype=C.c_double
    assert lib.pythia_pdf_init(a.xmldir.encode(),a.pdf.encode())==0
    pdf=lib.pythia_pdf_xf
    if a.native_pdf_export:
        from delta_native_pdf import NativePDF
        pdf=NativePDF(a.native_pdf_export,prefix.parent/(prefix.name+'_native_pdf'))
    absweights=np.abs([b['weight'] for b in born])
    assert np.allclose(absweights,absweights[0],rtol=1e-10,atol=0), 'Nonuniform absolute weights'
    trials=[[] for b in born]
    bad=0
    for line in Path(str(prefix)+'.trials.jsonl').open():
        t=json.loads(line)
        if not t['ok']: bad+=1
        if t['trial']>=a.probes: t.pop('real',None)
        trials[t['event']].append(t)
    if bad: raise RuntimeError(f'{bad} failed showers: investigate before analysis')
    n=len(born); common=np.empty((n,len(Q),3)); refcdf=np.empty((n,len(Q)))
    nref=np.empty(n,dtype=int); probes=[]; flags=np.zeros((n,4),int)
    start_time=time.monotonic()
    with DeltaEvaluator(work=prefix.parent/(prefix.name+'_delta_build')) as e:
        for i,(b,ts) in enumerate(zip(born,trials)):
            assert len(ts)>a.probes
            assert [t['trial'] for t in ts]==list(range(len(ts)))
            reference=np.sort([t['pt'] for t in ts[a.probes:]])
            nref[i]=len(reference)
            refcdf[i]=np.searchsorted(reference,Q,side='right')/len(reference)
            for k,q in enumerate(Q):
                r=e.evaluate(b,q,pdf)
                common[i,k]=[r['delta'],r['isr'],r['fsr']]
                flags[i,0]+=any(r['clipped']); flags[i,1]+=any(r['nonfinite_raw'])
            for t in ts[:a.probes]:
                if t['pt']==0: continue
                real=t['real']
                assert real['ids'][-1]==21 or abs(real['ids'][-1])<=5,real
                r=e.evaluate_real(b,real,pdf,mc=0.,mb=0.)
                f=np.searchsorted(reference,t['pt'],side='right')/len(reference)
                p=real['p'][-1]
                probes.append([i,t['trial'],t['pt'],np.hypot(p[1],p[2]),t['type'],
                    r['delta'],f,r['isr'],r['fsr'],int(any(r['clipped'])),int(any(r['nonfinite_raw']))])
                flags[i,2]+=any(r['clipped']); flags[i,3]+=any(r['nonfinite_raw'])
            if (i+1)%50==0: print(f'evaluated {i+1}/{n} Born states ({time.monotonic()-start_time:.1f}s)',flush=True)
        provenance=e.provenance
        if hasattr(pdf,'provenance'): provenance['native_pdf']=pdf.provenance
    provenance.update(branch=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(str(prefix)+'.born.jsonl'),Path(str(prefix)+'.trials.jsonl')]},
        delta_pdf='native nn23nlo' if a.native_pdf_export else a.pdf,
        shower_pdf=re.search(r'^PDF:pSet = (.+)$',Path(str(prefix)+'.settings.cmnd').read_text(),re.M).group(1),
        probes_per_born=a.probes,reference_repeats=nref.tolist(),
        ensemble='absolute-weight S-event sample from actual branch, equally weighted conditional probabilities',
        pdf_flavours='production grouped flavours; first-flavour PDG for Sudakov lookup',
        columns=['event','trial','pt_evol','pt_lab','type','delta','pythia_cdf','isr','fsr','clipped','nonfinite'])
    np.savez_compressed(str(prefix)+'.analysis.npz',q=Q,common=common,refcdf=refcdf,
        probes=np.array(probes),flags=flags,channels=np.array([b['ids'][0]==21 for b in born]),
        weights=np.array([b['weight'] for b in born]),scalup=np.array([b['scalup'] for b in born]))
    Path(str(prefix)+'.analysis.json').write_text(json.dumps(provenance,indent=2)+'\n')


def summarize(a):
    z=np.load(a.prefix+'.analysis.npz'); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    n=len(z['common']); rng=np.random.default_rng(102727)
    boots=rng.integers(0,n,size=(a.bootstrap,n))
    rows=[]
    for channel,mask in [('all',np.ones(n,bool)),('gg',z['channels']),('qqbar',~z['channels'])]:
        idx=np.where(mask)[0]; m=len(idx)
        if m<2: continue
        for j,q in enumerate(z['q']):
            d=z['common'][mask,j,0]; p=z['refcdf'][mask,j]; diff=d-p
            rows.append(dict(channel=channel,pt_GeV=q,delta=d.mean(),pythia=p.mean(),
                delta_over_pythia=d.mean()/p.mean(),difference=diff.mean(),
                difference_se=diff.std(ddof=1)/np.sqrt(m),
                pythia_se=p.std(ddof=1)/np.sqrt(m),delta_se=d.std(ddof=1)/np.sqrt(m),born=m))
    csvwrite(out/'common_cut.csv',rows)
    r=z['probes']; event=r[:,0].astype(int); binned=[]
    for channel,mask in [('all',np.ones(n,bool)),('gg',z['channels']),('qqbar',~z['channels'])]:
        for lo,hi in zip(EDGES[:-1],EDGES[1:]):
            sel=(r[:,2]>=lo)&(r[:,2]<hi)&mask[event]
            if sum(sel)<20: continue
            s=r[sel]; ids=event[sel]
            counts=np.bincount(ids,minlength=n)
            ds=np.bincount(ids,weights=s[:,5],minlength=n)
            ps=np.bincount(ids,weights=s[:,6],minlength=n)
            bn=counts[boots].sum(axis=1); good=bn>0
            bd=ds[boots[good]].sum(axis=1)/bn[good]
            bp=ps[boots[good]].sum(axis=1)/bn[good]
            ratio=s[:,5].mean()/s[:,6].mean()
            binned.append(dict(channel=channel,pt_low=lo,pt_high=hi,pt_mean=s[:,2].mean(),
                delta=s[:,5].mean(),pythia=s[:,6].mean(),delta_over_pythia=ratio,
                difference=(s[:,5]-s[:,6]).mean(),difference_se=(bd-bp).std(ddof=1),
                ratio_se=(bd/bp).std(ddof=1) if np.all(bp>0) else None,pythia_se=bp.std(ddof=1),delta_se=bd.std(ddof=1),
                undefined_raw_count=int(sum(s[:,10]>0)),finite_clipped_count=int(sum(s[:,9]>0)),
                ratio_excluding_undefined=s[s[:,10]==0,5].mean()/s[s[:,10]==0,6].mean(),
                fsr_delta=s[:,8].mean(),actual_fsr_fraction=np.mean(s[:,4]==3),
                probes=len(s),born=np.sum(counts>0)))
    csvwrite(out/'real_emission.csv',binned)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex='col',gridspec_kw={'height_ratios':[2,1]})
    for col,(source,title) in enumerate([(rows,'Common cutoff on Born states'),(binned,'Delta on actual first emissions')]):
        for channel,color in [('all','black'),('gg','#0072B2'),('qqbar','#D55E00')]:
            s=[v for v in source if v['channel']==channel]
            x=[v.get('pt_GeV',v.get('pt_mean')) for v in s]
            axes[0,col].plot(x,[v['delta'] for v in s],color=color,label=f'Delta {channel}')
            axes[0,col].errorbar(x,[v['pythia'] for v in s],yerr=[v['pythia_se'] for v in s],
                fmt='o',ms=3,color=color,alpha=.8,label=f'Pythia {channel}')
            y=np.array([v['delta_over_pythia'] for v in s])
            axes[1,col].plot(x,y,color=color)
            if col==1:
                err=np.array([v['ratio_se'] for v in s],dtype=float); axes[1,col].fill_between(x,y-err,y+err,color=color,alpha=.15)
        axes[0,col].set_title(title); axes[0,col].set_ylim(-.03,1.05)
        axes[0,col].legend(fontsize=7,ncol=2)
        axes[1,col].axhline(1,color='gray',lw=.8); axes[1,col].set_ylim(0,2.5)
        axes[1,col].set_xlabel(r'First-emission shower $p_T$ [GeV]')
        for ax in axes[:,col]: ax.set_xscale('log'); ax.grid(alpha=.2)
    axes[0,0].set_ylabel('No-emission probability'); axes[1,0].set_ylabel('Delta / Pythia')
    fig.suptitle(r'$pp\to t\bar t$, 13 TeV, stable tops; branch 801a3417c; Pythia 8.313')
    fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=170); fig.savefig(out/'comparison.pdf'); plt.close(fig)
    summary=dict(born=n,gg=int(sum(z['channels'])),qqbar=int(sum(~z['channels'])),
        real_probes=len(r),first_fsr_fraction=float(np.mean(r[:,4]==3)),
        real_delta_fsr_unity_fraction=float(np.mean(np.abs(r[:,8]-1)<1e-12)),
        flags=dict(zip(['common_clipped','common_nonfinite_raw','real_clipped','real_nonfinite_raw'],z['flags'].sum(axis=0).tolist())),
        common_selected=[v for v in rows if v['channel']=='all' and v['pt_GeV'] in [5,10,20,50,100,200]],
        real_selected=[v for v in binned if v['channel']=='all'],
        caveats=['Common cutoff is auxiliary; real-emission comparison uses native real-dependent stopping scales.',
                 'Pythia CDF for each probe uses independent reference showers of same Born point and color.',
                 'Errors include Born-cluster variation; first-emission bins are correlated.',
                 'Ensemble uses absolute S-event measure, not a signed NLO cross-section.',
                 'Full table generation provenance is unavailable; primary comparison leaves tables unchanged.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item())+'\n')
    print(json.dumps(summary,indent=2,default=lambda x:x.item()))


def main():
    p=argparse.ArgumentParser(); p.add_argument('prefix'); p.add_argument('--pdf-library',required=True)
    p.add_argument('--xmldir',required=True); p.add_argument('--pdf',default='15');p.add_argument('--probes',type=int,default=100)
    p.add_argument('--output',required=True);p.add_argument('--reuse',action='store_true');p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--native-pdf-export')
    a=p.parse_args()
    if not a.reuse: calculate(a)
    summarize(a)
if __name__=='__main__': main()
