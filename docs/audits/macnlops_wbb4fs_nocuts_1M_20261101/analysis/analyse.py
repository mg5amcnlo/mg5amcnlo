#!/usr/bin/env python3
"""First/full no-cut Wbb spectra, signed transfers and paired sampling errors."""
from pathlib import Path
import argparse
import glob
import hashlib
import json
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(__file__).resolve().parent
DEFINITION=json.loads((HERE/'observables.json').read_text())
FIELDS=DEFINITION['fields']
LABELS={'source':'Source','veto':'S veto only','removal':'H removal only','matched':'MAcNLOPS'}
STYLES={'source':('#111111','-'),'veto':('#d95f02','--'),'removal':('#1b9e77','-.'),'matched':('#2166ac','-')}
RANGES=[(0,1),(1,2),(2,5),(5,10),(10,20),(5,20),(20,50),(50,100),(0,None)]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def serial(x):
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x

def verify_hwu(path,values,names):
    labels=None;histograms={};rounding={};lines=iter(path.read_text().splitlines())
    for line in lines:
        if line.startswith('##&'):labels=[x.strip() for x in line[3:].split('&')]
        elif line.startswith('<histogram>'):
            title=line.split('"')[1].split('|')[0].strip();rows=[];bounds=[]
            for row in lines:
                if row.startswith('<\\histogram>'):break
                if not row.strip() or row.lstrip().startswith('#'):continue
                tokens=row.replace('D','E').split();rows.append([float(t) for t in tokens]);br=[]
                for token in tokens:
                    mantissa,exponent=token.upper().split('E') if 'E' in token.upper() else (token,'0')
                    decimals=len(mantissa.split('.')[1]) if '.' in mantissa else 0
                    br.append(.5*10.**(int(exponent)-decimals))
                bounds.append(br)
            histograms[title]=np.asarray(rows);rounding[title]=np.asarray(bounds)
    if not labels:raise RuntimeError('Missing HwU labels')
    nobs=len(FIELDS);maximum=0.;checked=0
    for spec in DEFINITION['histograms']:
        title=spec['name'];printed=histograms[title]
        for wi,name in enumerate(names):
            w=values[:,nobs+wi]
            if spec['field'] is None:exact=np.array([w.sum()])
            else:
                x=values[:,FIELDS.index(spec['field'])].copy();valid=np.isfinite(x)
                if spec['transform']=='log10':
                    valid&=x>0.;x=np.log10(np.where(x>0.,x,1.))
                exact=np.histogram(x[valid],bins=np.linspace(spec['lo'],spec['hi'],spec['bins']+1),weights=w[valid])[0]
            col=labels.index(name);bound=rounding[title][:,col]+1.e-11*max(1.,float(np.abs(w).sum()))
            difference=np.abs(printed[:,col]-exact)
            if not np.all(difference<=bound):
                raise RuntimeError('HwU vs binary mismatch '+str(path)+' '+title+' '+name+' '+str(float(difference.max())))
            maximum=max(maximum,float(np.max(difference/bound)));checked+=len(exact)
    return dict(path=str(path),sha256=sha(path),bins_weights_checked=checked,max_rounding_fraction=maximum,passed=True)

def read_full(root,first,out):
    root=Path(root)
    files=list(root.rglob('wbb_observables.bin'))
    def number(p):
        match=re.fullmatch(r'run_(\d+)',p.parent.name)
        return int(match[1]) if match else 0
    files.sort(key=number)
    if not files:raise RuntimeError('No full-shower binary streams in '+str(root))
    rows=[];groups=[];bindings=[];weight_names=None
    for job,path in enumerate(files):
        with path.open('rb') as f:
            header=np.fromfile(f,dtype=np.int32,count=2)
            if len(header)!=2 or header[0]!=len(FIELDS) or header[1]<4:
                raise RuntimeError('Invalid stream header: '+str(path))
            values=np.fromfile(f,dtype=np.float64)
        columns=int(header.sum())
        if len(values)%columns:raise RuntimeError('Truncated binary stream: '+str(path))
        values=values.reshape(-1,columns)
        names=[name.strip() for name in (path.parent/'wbb_weight_names.txt').read_text().splitlines()]
        if len(names)!=header[1]:raise RuntimeError('Wrong weight-label count')
        if weight_names is None:weight_names=names
        elif names!=weight_names:raise RuntimeError('Full jobs have different weight labels')
        rows.append(values);groups.append(np.full(len(values),job,dtype=np.int32))
        proof=verify_hwu(path.parent/'MADatNLO.HwU',values,names)
        bindings.append(dict(path=str(path),events=len(values),sha256=sha(path),weight_names=names,hwu_comparison=proof))
    rows=np.concatenate(rows);group=np.concatenate(groups)
    n=len(first['nominal'])
    if len(rows)!=n:raise RuntimeError('First/full event counts differ: %d vs%d'%(n,len(rows)))
    choices={'source':['central value'],'matched':['macnlops'],'veto':['macnlops_veto_only'],'removal':['macnlops_removal_only']}
    weights={}
    for key,candidates in choices.items():
        found=[weight_names.index(name) for name in candidates if name in weight_names]
        if len(found)!=1:raise RuntimeError('Unknown %s weight label in %r'%(key,weight_names))
        weights[key]=rows[:,len(FIELDS)+found[0]]
    target={'source':first['nominal'],'matched':first['macnlops'],
            'veto':np.where(first['is_h'],first['nominal'],first['macnlops']),
            'removal':np.where(first['is_h'],first['macnlops'],first['nominal'])}
    errors={key:float(np.max(np.abs(weights[key]-target[key]))) for key in choices}
    for key in choices:
        if not np.allclose(weights[key],target[key],rtol=2e-12,atol=1e-12):
            raise RuntimeError('First/full signed weights differ for'+key+':'+str(errors[key]))
    if not np.allclose(weights['veto']+weights['removal'],weights['source']+weights['matched'],rtol=2e-12,atol=1e-12):
        raise RuntimeError('Four-weight identity fails after shower')
    np.savez_compressed(out/'full_events.npz',source_index=np.arange(n),job_index=group,
        observables=rows[:,:len(FIELDS)],observable_names=np.asarray(FIELDS),
        nominal=weights['source'],macnlops=weights['matched'],is_h=first['is_h'],
        weights=np.stack([weights[name] for name in choices],axis=1),weight_names=np.asarray(list(choices)))
    return rows[:,:len(FIELDS)],weights,group,dict(events=n,jobs=len(files),bindings=bindings,
        weight_max_absolute_differences=errors,weight_order_preserved=True,
        four_weight_identity=True,events_per_job=[len(x) for x in groups])

def grouped(x,group,ng):
    return np.bincount(group,weights=x,minlength=ng)

def sampling_error(influence,group,ng,center=False):
    if ng==1:return float(np.linalg.norm(influence-influence.mean() if center else influence))
    sums=grouped(influence,group,ng)
    if center:sums-=sums.mean()
    return float(np.sqrt(ng/(ng-1))*np.linalg.norm(sums))

def summary(mask,weights,h,group,ng,keep):
    w,m=weights['source'],weights['matched']
    S=np.where(mask&~h,w-m,0.);H=np.where(mask&h,m-w,0.)
    W,M=float(w[mask].sum()),float(m[mask].sum());X,Y=float(S.sum()),float(H.sum())
    ratio=Y/X if X else None;relative=M/W-1 if W else None
    r=dict(events=int(mask.sum()),four_weights_pb={name:float(x[mask].sum()) for name,x in weights.items()},
        S_pb=X,H_pb=Y,net_H_minus_S_pb=Y-X,
        net_error_pb=sampling_error(H-S,group,ng,center=True),H_over_S=ratio,
        ratio_error=sampling_error((H-ratio*S)/X,group,ng) if ratio is not None and X>0 and Y>0 else None,
        first_eventwise_ratio_error=float(np.linalg.norm((H-ratio*S)/X)) if ratio is not None and X>0 and Y>0 else None,
        matched_source_minus1=relative,
        matched_source_error=sampling_error(np.where(mask,m-(M/W)*w,0.)/W,group,ng) if W else None,
        negative_S_events=int(np.count_nonzero(mask&~h&(w<0))),
        positive_S_events=int(np.count_nonzero(mask&~h&(w>0))),
        negative_H_events=int(np.count_nonzero(mask&h&(w<0))),
        positive_H_events=int(np.count_nonzero(mask&h&(w>0))),
        classes={})
    for label,selected in [('S_positive',mask&~h&(w>0)),('S_negative',mask&~h&(w<0)),
                           ('H_positive',mask&h&(w>0)),('H_negative',mask&h&(w<0))]:
        r['classes'][label]=dict(events=int(selected.sum()),source_pb=float(w[selected].sum()),
            absolute_source_pb=float(np.abs(w[selected]).sum()),matched_pb=float(m[selected].sum()),
            S_transfer_pb=float(S[selected].sum()),H_transfer_pb=float(H[selected].sum()))
    E=np.where(mask,w*keep,0.);ES=np.where(mask&~h,w-E,0.);EH=np.where(mask&h,E-w,0.)
    ex,ey=ES.sum(),EH.sum();er=ey/ex if ex else None
    r['expected']=dict(S_pb=float(ex),H_pb=float(ey),net_pb=float(ey-ex),H_over_S=er,
        ratio_error=sampling_error((EH-er*ES)/ex,group,ng) if er is not None and ex>0 and ey>0 else None,
        matched_source_minus1=float(E.sum()/W-1) if W else None)
    return r

def histogram(obs,weights,group,ng,spec):
    if spec['field'] is None:return None
    x=obs[:,FIELDS.index(spec['field'])].copy();missing=~np.isfinite(x);zero=x==0.
    if spec['transform']=='log10':
        missing|=x<=0
        x=np.log10(np.where(x>0,x,1.))
    edges=np.linspace(spec['lo'],spec['hi'],spec['bins']+1)
    selected=~missing&(x>=edges[0])&(x<edges[-1]);nb=spec['bins']
    index=np.searchsorted(edges,x[selected],side='right')-1
    jobindex=group[selected]*nb+index
    chunks={name:np.bincount(jobindex,weights=v[selected],minlength=ng*nb).reshape(ng,nb) for name,v in weights.items()}
    totals={name:v.sum(axis=0) for name,v in chunks.items()};base=totals['source']
    curves={}
    for name,total in totals.items():
        ratio=np.divide(total,base,out=np.full(nb,np.nan),where=base!=0)
        residual=chunks[name]-ratio*chunks['source']
        variance=np.sum(residual**2,axis=0)*(ng/(ng-1) if ng>1 else 1.)
        if ng==1:
            rr=np.nan_to_num(ratio)[index]
            variance=np.bincount(index,weights=(weights[name][selected]-rr*weights['source'][selected])**2,minlength=nb)
        curves[name]=dict(sum=total,ratio=ratio,ratio_error=np.divide(np.sqrt(variance),np.abs(base),out=np.full(nb,np.nan),where=base!=0))
    return dict(edges=edges,curves=curves,flows={name:dict(missing_events=int(missing.sum()),missing_pb=float(w[missing].sum()),
        underflow_events=int((~missing&(x<edges[0])).sum()),underflow_pb=float(w[~missing&(x<edges[0])].sum()),
        overflow_events=int((~missing&(x>=edges[-1])).sum()),overflow_pb=float(w[~missing&(x>=edges[-1])].sum()),
        zero_events=int(zero.sum()),zero_pb=float(w[zero].sum())) for name,w in weights.items()})

PLOT_LABELS={
 'primary scale 0-100':'pT(W+b+bbar) [GeV]',
 'radiation jet1 pT 0-50':'Leading radiation-jet pT [GeV]',
 'bb invariant mass':'m(b+bbar) [GeV]',
 'W pT 0-500':'W pT [GeV]',
 'bottom pT 0-500':'Bottom pT [GeV]',
 'antibottom pT 0-500':'Antibottom pT [GeV]',
 'Wbb invariant mass':'m(W+b+bbar) [GeV]',
 'bb pT 0-500':'pT(b+bbar) [GeV]',
 'all jet1 pT 0-500':'Leading all-parton jet pT [GeV]'}

def plot_page(pdf,first,full,specs,subtitle):
    # Explicit axes positions avoid cross-page GridSpec layout/clipping.
    fig=plt.figure(figsize=(12,12))
    for row,spec in enumerate(specs):
        for col,(hist,stage) in enumerate([(first,'First emission'),(full,'Full shower')]):
            item=hist[spec['name']];edges=np.asarray(item['edges']);width=np.diff(edges)
            left=.095+.47*col;bottom=.67-.285*row
            ax=fig.add_axes([left,bottom+.06,.36,.17])
            ra=fig.add_axes([left,bottom,.36,.055],sharex=ax)
            for name,(color,style) in STYLES.items():
                v=item['curves'][name]
                sums=np.asarray(v['sum']);ratio=np.asarray(v['ratio'],dtype=float);error=np.asarray(v['ratio_error'],dtype=float)
                ax.stairs(sums/width,edges,color=color,ls=style,label=LABELS[name],lw=1.)
                if name!='source':
                    center=.5*(edges[:-1]+edges[1:]);good=np.isfinite(ratio)&np.isfinite(error)
                    ra.plot(center[good],ratio[good],color=color,ls=style,lw=1.)
                    if name=='matched':ra.fill_between(center[good],(ratio-error)[good],(ratio+error)[good],color=color,alpha=.18)
            ax.axhline(0,color='black',lw=.5);ra.axhline(1,color='black',lw=.6)
            ax.set_ylabel('dσ/dx [pb/GeV]');ra.set_ylabel('/ source')
            ra.set_xlabel(PLOT_LABELS.get(spec['name'],spec['name']))
            ra.set_ylim(.9,1.1);ra.set_yticks([.9,1.,1.1]);ra.set_xlim(edges[0],edges[-1])
            ax.tick_params(labelbottom=False);ax.grid(alpha=.2);ra.grid(alpha=.2)
            if row==0:ax.set_title(stage,pad=8);ax.legend(frameon=False,fontsize=7)
    fig.suptitle('W+bbbar, 4FS; no generation or analysis cuts\n'+subtitle,y=.985)
    fig.text(.5,.036,'Absolute signed weights; no selection/range renormalization. Conditional sampling errors only.',ha='center',fontsize=8)
    fig.text(.5,.021,'Ratios displayed from 0.9 to 1.1; sparse-tail excursions outside this window are retained in comparison.json.',ha='center',fontsize=8)
    pdf.savefig(fig);plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first',required=True,type=Path);parser.add_argument('--full-root',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data=np.load(args.first);assert list(data['observable_names'])==FIELDS
    n=len(data['nominal']);w,m,h=data['nominal'],data['macnlops'],data['is_h'];keep=data['keep']
    firstweights=dict(source=w,veto=np.where(h,w,m),removal=np.where(h,m,w),matched=m)
    fullobs,fullweights,group,verification=read_full(args.full_root,data,args.out);ng=verification['jobs']
    results={};hist={}
    for stage,obs,weights in [('first',data['observables'],firstweights),('full',fullobs,fullweights)]:
        rows=[]
        for lo,hi in RANGES:
            mask=obs[:,0]>=lo
            if hi is not None:mask&=obs[:,0]<hi
            rows.append(dict(lo=lo,hi=hi,**summary(mask,weights,h,group,ng,keep)))
        results[stage]=dict(ranges=rows,all_events=summary(np.ones(n,dtype=bool),weights,h,group,ng,keep),
            missing_observable_counts={name:int(np.count_nonzero(~np.isfinite(obs[:,i]))) for i,name in enumerate(FIELDS)})
        hist[stage]={spec['name']:histogram(obs,weights,group,ng,spec) for spec in DEFINITION['histograms'] if spec['field'] is not None}
    drift=[]
    for lo,hi in RANGES:
        influences=[];ratios=[]
        for obs,weights in [(data['observables'],firstweights),(fullobs,fullweights)]:
            mask=obs[:,0]>=lo
            if hi is not None:mask&=obs[:,0]<hi
            S=np.where(mask&~h,w-weights['matched'],0.);H=np.where(mask&h,weights['matched']-w,0.)
            X,Y=S.sum(),H.sum();r=Y/X if X else None
            ratios.append(r);influences.append((H-r*S)/X if r is not None else None)
        drift.append(dict(lo=lo,hi=hi,full_minus_first=ratios[1]-ratios[0] if None not in ratios else None,
            paired_error=sampling_error(influences[1]-influences[0],group,ng) if all(x is not None for x in influences) else None))
    report=dict(events=n,jobs=ng,cuts='none',primary='pT(W+b+bbar)',jet_definition=DEFINITION['definition'],
        errors='Conditional paired source/shower chunks; one-job pilots use eventwise influences. Hard integration, channel-beta and native-envelope uncertainties separate.',
        verification=verification,results=results,paired_shower_drift=drift,
        inputs={str(args.first):sha(args.first)},histograms=hist)
    (args.out/'comparison.json').write_text(json.dumps(serial(report),indent=2,allow_nan=False)+'\n')
    index={x['name']:x for x in DEFINITION['histograms']}
    with PdfPages(args.out/'wbb_nocuts_validation.pdf') as pdf:
        for names,title in [(['primary scale 0-100','radiation jet1 pT 0-50','bb invariant mass'],'Radiation and bottom-pair mass'),
                            (['W pT 0-500','bottom pT 0-500','antibottom pT 0-500'],'Stable hard-object transverse momenta'),
                            (['Wbb invariant mass','bb pT 0-500','all jet1 pT 0-500'],'System and all-parton jet controls')]:
            plot_page(pdf,hist['first'],hist['full'],[index[name] for name in names],title)
    for stage in ['first','full']:
        row=next(r for r in results[stage]['ranges'] if r['lo']==5 and r['hi']==20)
        print(stage,'recoil5-20',json.dumps({k:row[k] for k in ['S_pb','H_pb','H_over_S','ratio_error','net_H_minus_S_pb','matched_source_minus1']}))
    print('Output:',args.out)

if __name__=='__main__':main()
