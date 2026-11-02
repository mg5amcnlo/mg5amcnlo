#!/usr/bin/env python3
"""Read-only Wj inclusive/fiducial40 and qualified historical comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
from report import load,range_summary,paired_stage_drift
from run import DEFAULT_BASE,ROOT,dump,sha


def rebin(h,size):
    n=len(h['sumw'])
    if n%size:raise RuntimeError('Incompatible histogram binning')
    edges=h['edges'].reshape(n//size,size,2)
    return dict(edges=np.column_stack((edges[:,0,0],edges[:,-1,1])),
        sumw=h['sumw'].reshape(n//size,size,-1).sum(axis=1),
        blocks=h['blocks'].reshape(len(h['blocks']),n//size,size,-1).sum(axis=2))


def differential(h):
    s=h['sumw'];b=h['blocks'];n=len(b)
    S=s[:,0]-s[:,2];H=s[:,3]-s[:,0];net=s[:,1]-s[:,0]
    sb=b[:,:,0]-b[:,:,2];hb=b[:,:,3]-b[:,:,0];nb=b[:,:,1]-b[:,:,0]
    def error(values):return np.sqrt(n/(n-1)*((values-values.mean(axis=0))**2).sum(axis=0))
    ratio=np.divide(H,S,out=np.full(len(S),np.nan),where=S!=0)
    ratio_error=np.sqrt(n/(n-1)*(((hb-ratio[None,:]*sb)/S[None,:])**2).sum(axis=0))
    matched_ratio=np.divide(s[:,1],s[:,0],out=np.full(len(S),np.nan),where=s[:,0]!=0)
    matched_ratio_error=np.sqrt(n/(n-1)*(((b[:,:,1]-matched_ratio[None,:]*b[:,:,0])/s[None,:,0])**2).sum(axis=0))
    def serialise(values):return [float(x) if np.isfinite(x) else None for x in values]
    return dict(edges=h['edges'].tolist(),four_weights_pb=s[:,:4].tolist(),
        S_pb=S.tolist(),H_pb=H.tolist(),net_pb=net.tolist(),S_error_pb=error(sb).tolist(),
        H_error_pb=error(hb).tolist(),net_error_pb=error(nb).tolist(),H_over_S=serialise(ratio),
        H_over_S_error=serialise(ratio_error),matched_over_source=serialise(matched_ratio),
        matched_over_source_error=serialise(matched_ratio_error))


def compare(base,run):
    result=base/'results/Wj'/run;stages={};hists={}
    for stage in ('first','full'):
        if not (result/(stage+'_histograms.json.gz')).exists():continue
        data=load(result,stage)['histograms'];hists[stage]=data
        stages[stage]={}
        for label,prefix in (('inclusive',''),('fiducial40','fiducial40_')):
            h=data[prefix+'primary_0_100']
            stages[stage][label]=dict(selected_total=range_summary(data[prefix+'total']),
                recoil5_20=range_summary(h,5.,20.),recoil20_50=range_summary(h,20.,50.),
                curves_2GeV=differential(rebin(h,2)),transfers_5GeV=differential(rebin(h,5)))
        stages[stage]['paired_fiducial40_minus_inclusive']=dict(
            recoil5_20=paired_stage_drift(data['primary_0_100'],data['fiducial40_primary_0_100'],5.,20.),
            selected_total=paired_stage_drift(data['total'],data['fiducial40_total']))
    paired={}
    if len(hists)==2:
        for label,prefix in (('inclusive',''),('fiducial40','fiducial40_')):
            paired[label]=dict(recoil5_20=paired_stage_drift(hists['first'][prefix+'primary_0_100'],hists['full'][prefix+'primary_0_100'],5.,20.),
                selected_total=paired_stage_drift(hists['first'][prefix+'total'],hists['full'][prefix+'total']))
    oldpath=ROOT/'docs/audits/macnlops_wj_pt40_20261031/plots/histograms.json'
    old=json.loads(oldpath.read_text())
    historical=dict(path=str(oldpath),sha256=sha(oldpath),epoch=old['epoch'],
        selection=old['definition'],stages={stage:old['stages'][stage]['40'] for stage in ('first','full')},
        paired_full_minus_first=old['paired']['full_minus_first_cut40'],
        qualification='Independent historical pre-gluon-fix/native-IR source, before native Born G and explicit model-mass continuation. Same generation pT20/R.5 and fiducial pT40/|y|4.5 definitions; complete source/profile epochs differ. No attribution to t_c alone.')
    report=dict(process='Wj',run=run,stages=stages,paired_full_minus_first=paired,historical=historical,
        complete_first_full=len(stages)==2,
        selection='At least one anti-kT R=.5 jet with pT>40 GeV and |y|<4.5. Primary is pT(W+highest-pT qualifying jet).',
        normalization='Original signed weights; no survivor, fiducial or histogram-range renormalization.',
        errors='Paired conditional50-source-chunk/native-shower-job delta-method errors. Historical errors imported unchanged. No hard-integration or model uncertainty included.',
        weight_order=['source','matched','veto','removal'])
    dump(result/'fiducial40_comparison.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    def array(values):return np.asarray([np.nan if x is None else x for x in values],dtype=float)
    with PdfPages(result/'fiducial40_comparison.pdf') as pdf:
        for stage,data in stages.items():
            fig,axes=plt.subplots(2,2,figsize=(12,8))
            for column,label in enumerate(('inclusive','fiducial40')):
                h=data[label]['curves_2GeV'];e=np.asarray(h['edges']);w=np.asarray(h['four_weights_pb'])
                for i,name in enumerate(('source','matched','S veto','H removal')):
                    axes[0,column].stairs(w[:,i]/2.,np.r_[e[:,0],e[-1,1]],label=name)
                axes[0,column].set_title(stage+' '+label);axes[0,column].set_ylabel('d sigma / d recoil [pb/GeV]')
                axes[0,column].legend(fontsize=8)
                axes[1,column].errorbar(e.mean(axis=1),array(h['matched_over_source']),yerr=array(h['matched_over_source_error']),fmt='.',ms=3)
                axes[1,column].axhline(1.,color='black',lw=.7);axes[1,column].set_ylabel('matched / source')
                axes[1,column].set_xlabel('pT(W+leading selected jet) [GeV]')
            for ax in axes.flat:ax.grid(alpha=.2)
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
            fig,axes=plt.subplots(1,2,figsize=(12,4.5))
            for label in ('inclusive','fiducial40'):
                h=data[label]['transfers_5GeV'];e=np.asarray(h['edges']);xx=e.mean(axis=1)
                axes[0].errorbar(xx,array(h['H_over_S']),yerr=array(h['H_over_S_error']),fmt='.-',label='fresh '+label,ms=3)
            h=historical['stages'][stage]['transfers'];edge=np.asarray(old['transfer_edges'])
            axes[0].errorbar(.5*(edge[:-1]+edge[1:]),array(h['H_over_S']),yerr=array(h['H_over_S_error']),fmt='.-',label='historical fiducial40',ms=3)
            h=data['fiducial40']['curves_2GeV'];e=np.asarray(h['edges'])
            axes[1].errorbar(e.mean(axis=1),array(h['matched_over_source']),yerr=array(h['matched_over_source_error']),fmt='.',label='fresh fiducial40')
            oldh=historical['stages'][stage]['curves'];nom=np.asarray(oldh['source']['sum']);matched=np.asarray(oldh['matched']['sum'])
            edge=np.asarray(old['edges']);axes[1].plot(.5*(edge[:-1]+edge[1:]),matched/nom,'-',label='historical fiducial40')
            for ax,label in zip(axes,('H removal / S veto','matched / source')):
                ax.axhline(1.,color='black',lw=.7);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_ylabel(label);ax.set_xlabel('recoil [GeV]')
            fig.suptitle('Wj '+stage+': independent source/profile epochs');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    print('Wj fiducial40 report ready:',result/'fiducial40_comparison.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE);parser.add_argument('--run',default='run_1M_native_tc')
    a=parser.parse_args();compare(a.base,a.run)


if __name__=='__main__':main()
