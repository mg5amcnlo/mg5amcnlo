#!/usr/bin/env python3
"""Make the quantitative closure table and plot of the low-pT investigation.

Inputs are the recorded decisions, native Delta probes, and (for ttbar) the
independent ISR quadrature times FSR-only conditional shower measurements.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from diagnose_macnlops_lowpt import EDGES


def ratio_mean(weight, value, conditional_variance=None):
    mean=np.dot(weight,value)/weight.sum()
    var=len(weight)/(len(weight)-1)*np.sum((weight*(value-mean))**2)/weight.sum()**2
    if conditional_variance is not None:
        var+=np.dot(weight**2,conditional_variance)/weight.sum()**2
    return float(mean),float(np.sqrt(var))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('/export/slow1/rikkert/macnlops_1M_20261030/investigation_lowpt'))
    args=parser.parse_args();base=args.output;records=[]
    for key in ['DY','H','ttbar']:
        probes=json.loads((base/f'{key}_sudakov_probes.json').read_text())['rows']
        data=json.loads((base/f'{key}_balance.json').read_text())
        if key=='ttbar':
            with np.load(base/'ttbar_conditional_probabilities.npz') as f:
                B=f['B'];conditional_relative_var=(1.-f['fsr'])/(2000.*f['fsr'])
            with np.load(base/'ttbar_isr_probabilities.npz') as f:
                assert np.array_equal(f['sequence'],[r['sequence'] for r in probes])
        else:
            B=np.array([r['B'] for r in probes]);conditional_relative_var=np.zeros(len(B))
        assert len(B)==len(probes) and np.all(B>0)
        for i,row in enumerate(data):
            if not row['X']:continue
            chosen=np.array([r['group']==f'S_transfer_{i}' for r in probes])
            if chosen.sum()<2:continue
            w=np.array([r['X'] for r in probes])[chosen]
            real=np.array([r['source_real'] for r in probes])[chosen]/B[chosen]
            common=np.array([r['source_common'] for r in probes])[chosen]/B[chosen]
            prediction,error=ratio_mean(w,real,real**2*conditional_relative_var[chosen])
            common_prediction,common_error=ratio_mean(w,common,common**2*conditional_relative_var[chosen])
            measured=row['Y']/row['X']
            measured_error=np.sqrt(row['Y_error']**2/row['X']**2+row['Y']**2*row['X_error']**2/row['X']**4)
            records.append(dict(process=key,bin=i,lo=row['lo'],hi=row['hi'],X=row['X'],Y=row['Y'],
                                measured_Y_over_X=measured,measured_error=measured_error,
                                delta_prediction=prediction,delta_prediction_error=error,
                                common_cut_prediction=common_prediction,common_cut_error=common_error,
                                probe_events=int(chosen.sum())))
    (base/'closure.json').write_text(json.dumps(records,indent=2)+'\n')
    with (base/'closure.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.1),sharey=True,constrained_layout=True)
    for ax,key in zip(axes,['DY','H','ttbar']):
        rows=[r for r in records if r['process']==key and 2<=r['bin']<=10]
        x=np.array([np.sqrt(r['lo']*r['hi']) for r in rows])
        for field,error,offset,color,marker,label in [
            ('measured_Y_over_X','measured_error',.97,'black','o','Measured H / S transfer'),
            ('delta_prediction','delta_prediction_error',1.,'#c74440','s',r'$\langle\Delta_{\rm src}/\Delta_{\rm PS}\rangle_X$'),
            ('common_cut_prediction','common_cut_error',1.03,'#2478a5','^','Source at common evolution cutoff')]:
            ax.errorbar(x*offset,[r[field] for r in rows],yerr=[r[error] for r in rows],
                        color=color,marker=marker,ms=4,lw=1,capsize=2,label=label)
        ax.axhline(1.,color='.5',ls='--',lw=.8)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_ylim(.003,2.)
        ax.set_title(dict(DY='Drell–Yan',H='Higgs (HEFT)',ttbar=r'$t\bar t$')[key])
        ax.set_xlabel(r'First-emission hard-system $p_T$ [GeV]')
        ax.set_xticks([1,2,3,5,10,20,50],labels=['1','2','3','5','10','20','50'])
        ax.grid(alpha=.18)
    axes[0].set_ylabel('H weight added / S weight vetoed')
    axes[0].legend(loc='lower right',fontsize=8)
    fig.savefig(base/'lowpt_cause.png',dpi=190)
    fig.savefig(base/'lowpt_cause.pdf')
    for key in ['DY','H','ttbar']:
        for i in [4,6,7]:
            r=next(r for r in records if r['process']==key and r['bin']==i)
            print(key,f'{r["lo"]:g}-{r["hi"]:g}',
                  f'Y/X {r["measured_Y_over_X"]:.4f} ± {r["measured_error"]:.4f};',
                  f'predicted {r["delta_prediction"]:.4f} ± {r["delta_prediction_error"]:.4f};',
                  f'common {r["common_cut_prediction"]:.4f}')


if __name__=='__main__':main()
