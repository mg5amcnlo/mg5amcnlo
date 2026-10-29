#!/usr/bin/env python3
"""Compare independent ISR/FSR/default-PDF CDFs with the unchanged Delta."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyse import Q,csvwrite

def readcdf(prefix,n):
    counts=np.zeros((n,len(Q)),int); totals=np.zeros(n,int)
    with Path(str(prefix)+'.trials.jsonl').open() as f:
        for line in f:
            t=json.loads(line); assert t['ok']; i=t['event']
            counts[i]+=t['pt']<=Q;totals[i]+=1
    assert min(totals)>0
    return counts/totals[:,None],totals

def main():
    p=argparse.ArgumentParser();p.add_argument('prefix');p.add_argument('output');a=p.parse_args()
    z=np.load(a.prefix+'.analysis.npz');n=len(z['common']);out=Path(a.output)
    controls={name:readcdf(a.prefix+'_'+name,n)[0] for name in ['isr','fsr','pdf15']}
    rows=[]
    for channel,mask in [('all',np.ones(n,bool)),('gg',z['channels']),('qqbar',~z['channels'])]:
        for j,q in enumerate(Q):
            r=dict(channel=channel,pt_GeV=q)
            for c,k in [('delta',0),('delta_isr',1),('delta_fsr',2)]:r[c]=z['common'][mask,j,k].mean()
            for c,d in [('pythia',z['refcdf']),*controls.items()]:
                r[c]=d[mask,j].mean();r[c+'_se']=d[mask,j].std(ddof=1)/np.sqrt(sum(mask))
            r['independence_residual']=(controls['isr'][mask,j]*controls['fsr'][mask,j]-z['refcdf'][mask,j]).mean()
            r['matched_pdf_ratio']=r['delta']/r['pdf15']
            rows.append(r)
    csvwrite(out/'controls.csv',rows)
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
    s=[r for r in rows if r['channel']=='all'];x=Q
    for ax,c,label,delta in zip(axs,['isr','fsr','pdf15'],['ISR only','FSR only','PDF sensitivity'],['delta_isr','delta_fsr','delta']):
        ax.plot(x,[r[delta] for r in s],color='black',label='Delta')
        ax.errorbar(x,[r[c] for r in s],yerr=[r[c+'_se'] for r in s],fmt='o-',ms=3,label='Pythia PDF15' if c=='pdf15' else 'Pythia PDF13')
        if c=='pdf15':ax.plot(x,[r['pythia'] for r in s],ls='--',label='Pythia PDF13')
        ax.set_xscale('log');ax.set_ylim(0,1.03);ax.set_title(label);ax.set_xlabel(r'Shower $p_T$ cutoff [GeV]');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axs[0].set_ylabel('No-emission probability');fig.tight_layout();fig.savefig(out/'controls.png',dpi=170);fig.savefig(out/'controls.pdf')
    print(json.dumps(s,indent=2))
if __name__=='__main__':main()
