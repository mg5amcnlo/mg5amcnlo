#!/usr/bin/env python3
"""Read compact first/full histogram aggregates; produce paired shape plots."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from run import DEFAULT_BASE, PROCESSES, dump


def plotted_histograms(p):
    primary=['primary_0_100','primary_log10','primary_0_500']
    jets=['jet1_pt','jet2_pt','radiation_jet1_pt']
    if p['key'].startswith('EE'):
        return primary+['jet1_pt','jet2_pt','y23_log10','y34_log10','y45_log10',
                        'Durham_njets_ycut001','Cparameter']
    if p['key']=='H':return primary+['system_rapidity']+jets
    names=primary+['system_mass','system_rapidity','object1_pt','object2_pt']+jets
    if p['key']=='Wbb':names+=['pair_mass','pair_pt']
    names+=['pair_delta_phi','pair_delta_y']
    if p['key']=='Wj':names+=['fiducial40_primary_0_100','fiducial40_primary_log10','fiducial40_primary_0_500']
    return names


def plot_labels(p,name):
    key=p['key']
    primary={'DY':'pT(e−+e+)','H':'pT(h)','ttbar':'pT(t+tbar)',
        'Wbb':'pT(W+b+bbar)','Wj':'pT(W+leading jet)'}.get(key)
    if key.startswith('EE'):primary=f'sqrt(d{p["born_partons"]}{p["born_partons"]+1}) Durham'
    if name.startswith('fiducial40_'):primary='pT(W+leading qualifying jet), pTj>40 GeV, |yj|<4.5'
    if 'primary' in name:
        if name.endswith('log10'):return 'log10('+primary+' / GeV)','dσ/dlog10(scale/GeV) [pb]'
        return primary+' [GeV]','dσ/dscale [pb/GeV]'
    if name.startswith('y') and name.endswith('_log10'):
        return 'log10(Durham '+name[:3]+')','dσ/dlog10(y) [pb]'
    if name=='Cparameter':return 'C parameter','dσ/dC [pb]'
    if name=='Durham_njets_ycut001':return 'Durham jets at ycut=0.01','dσ/dNjets [pb]'
    if name=='system_rapidity':return 'hard-system rapidity','dσ/dy [pb]'
    if name=='pair_delta_y':return 'pair rapidity difference','dσ/dΔy [pb]'
    if name=='pair_delta_phi':return 'pair azimuthal separation [rad]','dσ/dΔphi [pb/rad]'
    objects={'DY':('e−','e+'),'ttbar':('t','tbar'),'Wbb':('b','bbar'),'Wj':('W','leading jet')}
    if name.startswith('object'):
        label=objects[key][0 if name.startswith('object1') else 1]
        return 'pT('+label+') [GeV]','dσ/dpT [pb/GeV]'
    labels={'system_mass':'hard-system invariant mass','pair_mass':'pair invariant mass',
        'pair_pt':'pair pT','jet1_pt':'leading jet pT','jet2_pt':'second jet pT',
        'radiation_jet1_pt':'leading radiation-jet pT'}
    return labels.get(name,name)+' [GeV]','dσ/dobservable [pb/GeV]'


def load(result,stage):
    with gzip.open(result/(stage+'_histograms.json.gz'),'rt') as stream:data=json.load(stream)
    for h in data['histograms'].values():
        h['sumw']=np.asarray(h['sumw']);h['blocks']=np.asarray(h['blocks'])
        edges=np.asarray(h['edges'])
        if stage=='first':
            h['sumw']=h['sumw'][1:-1];h['blocks']=h['blocks'][:,1:-1]
            h['edges']=np.column_stack((edges[:-1],edges[1:]))
        else:h['edges']=edges
    return data


def block_error(values):
    """Conditional error on a sum of independent block contributions."""
    n=len(values)
    if n<2:return None
    return float(np.sqrt(n/(n-1)*np.sum((values-values.mean())**2)))


def range_summary(h,lo=None,hi=None):
    edges=h['edges'];mask=np.ones(len(edges),dtype=bool)
    if lo is not None:mask &= edges[:,0]>=lo-1.e-8
    if hi is not None:mask &= edges[:,1]<=hi+1.e-8
    s=h['sumw'][mask].sum(axis=0);b=h['blocks'][:,mask].sum(axis=1)
    veto=s[0]-s[2];removal=s[3]-s[0];net=s[1]-s[0];n=len(b)
    bt=b[:,0]-b[:,2];bh=b[:,3]-b[:,0];bn=b[:,1]-b[:,0]
    ratio=removal/veto if veto else None
    ratio_error=(float(np.sqrt(n/(n-1)*np.sum(((bh-ratio*bt)/veto)**2)))
                 if n>1 and veto else None)
    return dict(source_pb=float(s[0]),matched_pb=float(s[1]),S_veto_pb=float(veto),
        H_removal_pb=float(removal),net_pb=float(net),net_error_pb=block_error(bn),
        S_veto_error_pb=block_error(bt),H_removal_error_pb=block_error(bh),H_over_S=ratio,
        H_over_S_error=ratio_error,matched_over_source=float(s[1]/s[0]) if s[0] else None,
        blocks=n,range=[lo,hi])


def paired_stage_drift(first,full,lo=None,hi=None):
    """Use the common source split to retain first/full covariance."""
    def vectors(h):
        edges=h['edges'];mask=np.ones(len(edges),dtype=bool)
        if lo is not None:mask &= edges[:,0]>=lo-1.e-8
        if hi is not None:mask &= edges[:,1]<=hi+1.e-8
        b=h['blocks'][:,mask].sum(axis=1)
        return np.column_stack((b[:,0]-b[:,2],b[:,3]-b[:,0],b[:,1]-b[:,0],b[:,0],b[:,1]))
    a,b=vectors(first),vectors(full)
    if a.shape!=b.shape:raise RuntimeError('First/full source blocks are not aligned')
    n=len(a)
    difference=b-a;out=dict(blocks=n,range=[lo,hi])
    for i,name in enumerate(('S_veto','H_removal','net','source','matched')):
        out[name+'_drift_pb']=float(difference[:,i].sum())
        out[name+'_drift_error_pb']=block_error(difference[:,i])
    aa,bb=a.sum(axis=0),b.sum(axis=0)
    if aa[0]!=0 and bb[0]!=0:
        ra,rb=aa[1]/aa[0],bb[1]/bb[0]
        influence=(b[:,1]-rb*b[:,0])/bb[0]-(a[:,1]-ra*a[:,0])/aa[0]
        out['H_over_S_drift']=float(rb-ra);out['H_over_S_drift_error']=block_error(influence)
    if aa[3]!=0 and bb[3]!=0:
        ra,rb=aa[4]/aa[3],bb[4]/bb[3]
        influence=(b[:,4]-rb*b[:,3])/bb[3]-(a[:,4]-ra*a[:,3])/aa[3]
        out['matched_over_source_drift']=float(rb-ra)
        out['matched_over_source_drift_error']=block_error(influence)
    return out


def report(base,p,run,plots=True):
    result=base/'results'/p['key']/run;data={}
    for stage in ('first','full'):
        if (result/(stage+'_histograms.json.gz')).is_file():data[stage]=load(result,stage)
    if not data:raise RuntimeError('No aggregate histograms found')
    target=json.loads((result/'first.json').read_text())['totals_pb']
    summary={}
    for stage,value in data.items():
        hs=value['histograms']
        summary[stage]=dict(inclusive=range_summary(hs['total']),
            primary_5_20=range_summary(hs['primary_0_100'],5.,20.),
            primary_20_50=range_summary(hs['primary_0_100'],20.,50.))
        if p['key']=='Wj':summary[stage]['fiducial40']=range_summary(hs['fiducial40_total'])
        # Full nominal and signed class totals must reproduce their first totals.
        # Matched/veto/removal inclusive weights must also remain unchanged.
        actual=hs['total']['sumw'][0]
        differences={name:float(actual[i]-target[name]) for i,name in enumerate(value['weights'])}
        max_relative=max(abs(differences[name])/max(1.,abs(target[name])) for name in differences)
        tolerance=2.e-6 if stage=='full' else 1.e-11
        summary[stage]['inclusive_weight_closure']=dict(differences_pb=differences,
            max_scaled_difference=max_relative,tolerance=tolerance,
            passed=max_relative<tolerance)
    paired={}
    if 'first' in data and 'full' in data:
        f=data['first']['histograms'];s=data['full']['histograms']
        paired=dict(inclusive=paired_stage_drift(f['total'],s['total']),
            primary_5_20=paired_stage_drift(f['primary_0_100'],s['primary_0_100'],5.,20.),
            primary_20_50=paired_stage_drift(f['primary_0_100'],s['primary_0_100'],20.,50.))
        if p['key']=='Wj':paired['fiducial40']=paired_stage_drift(f['fiducial40_total'],s['fiducial40_total'])
    dump(result/'shape_summary.json',dict(process=p['key'],run=run,stages=summary,
        paired_first_full_drift=paired,
        normalization='Signed original source cross sections; fixed analysis cuts only for Wj fiducial40.',
        uncertainty='Paired block jackknife variance; first contiguous source chunks, full independent shower jobs.',
        flow_policy='Inclusive totals include every event. Shape ranges omit out-of-range bins; no rescaling.'))
    if not plots:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    names=plotted_histograms(p)
    dump(result/'plot_metadata.json',dict(process=p['key'],plotted_histograms=names,
        axis_labels={name:dict(zip(('x','density'),plot_labels(p,name))) for name in names},
        omitted_histograms=[name for name in next(iter(data.values()))['histograms'] if name not in names],
        note='Generic coordinates that are inapplicable, fixed or duplicate primary/system quantities are omitted from presentation. Every underlying histogram and numerical summary is retained unchanged for audit.'))
    with PdfPages(result/'shapes.pdf') as pdf:
        for name in names:
            fig,axes=plt.subplots(2,len(data),figsize=(6.5*len(data),7.),squeeze=False,
                gridspec_kw={'height_ratios':[3,1]},sharex='col')
            for column,(stage,value) in enumerate(data.items()):
                h=value['histograms'][name];edges=h['edges'];s=h['sumw'];width=edges[:,1]-edges[:,0]
                upper,lower=axes[:,column]
                for i,label in ((0,'source'),(2,'S veto'),(3,'H removal'),(1,'combined')):
                    upper.stairs(s[:,i]/width,np.r_[edges[:,0],edges[-1,1]],label=label)
                xlabel,ylabel=plot_labels(p,name)
                upper.set_title(p['key']+' '+stage);upper.set_ylabel(ylabel)
                upper.legend(fontsize=8);upper.grid(alpha=.2)
                ratio=np.divide(s[:,1],s[:,0],out=np.full(len(s),np.nan),where=s[:,0]!=0)
                b=h['blocks'];n=len(b);err=np.zeros_like(ratio)
                if n>1:
                    residual=b[:,:,1]-ratio[None,:]*b[:,:,0]
                    err=np.divide(np.sqrt(n/(n-1)*(residual*residual).sum(axis=0)),np.abs(s[:,0]),
                        out=np.full(len(s),np.nan),where=s[:,0]!=0)
                center=edges.mean(axis=1)
                visible=np.isfinite(ratio)&np.isfinite(err)
                if np.any(visible):lower.errorbar(center[visible],ratio[visible],yerr=err[visible],fmt='.',ms=2,lw=.7)
                lower.axhline(1.,color='black',lw=.7);lower.set_ylabel('combined/source')
                lower.set_xlabel(xlabel);lower.grid(alpha=.2)
            fig.canvas.draw()
            fig.tight_layout(pad=1.5,w_pad=2.)
            pdf.savefig(fig,bbox_inches='tight',pad_inches=.2)
            plt.close(fig)
    print('Report ready:',result/'shape_summary.json',result/'shapes.pdf',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--process',nargs='+',default=['EE2j','DY'],choices=[p['key'] for p in PROCESSES])
    parser.add_argument('--run',default='run_1M_native_tc');parser.add_argument('--no-plots',action='store_true')
    a=parser.parse_args()
    for p in PROCESSES:
        if p['key'] in a.process:report(a.base,p,a.run,not a.no_plots)


if __name__=='__main__':main()
