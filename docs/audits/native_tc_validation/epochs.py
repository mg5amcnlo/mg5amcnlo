#!/usr/bin/env python3
"""Compare fresh native-t_c shapes with explicitly labelled frozen epochs.

Archived numbers are imported unchanged. Independent campaigns are compared
as independent estimates; no claim of a common-event regression is made.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from report import load,range_summary
from run import DEFAULT_BASE,PROCESSES,ROOT,dump,sha

AUDITS=ROOT/'docs/audits'
RANGES=((0.,5.),(5.,20.),(20.,50.),(50.,100.))


def historical(p):
    key=p['key']
    if key in ('DY','H','ttbar'):
        path=AUDITS/'macnlops_native_ir_1M_20261031/plots/native_transfer_comparison.json'
        data=json.loads(path.read_text())[key];out={}
        for epoch in ('v4','winner','native'):
            name=('historical '+epoch+' (2026-10-31 report)')
            out[name]=dict(reference=str(path.relative_to(ROOT)),sha256=sha(path),
                source_epoch=epoch,stages={})
            for stage,field in (('first','first_emission_realized_ranges'),('full','full_shower_ranges')):
                out[name]['stages'][stage]=[dict(lo=x['lo'],hi=x['hi'],H_over_S=x['ratio'],
                    H_over_S_error=x['error'],H_removal_pb=x['H_pb'],S_veto_pb=x['S_pb'],
                    matched_over_source=x.get('matched_over_source')) for x in data[epoch][field]]
        return out
    if key in ('EE2j','EE3j','EE4j','Wj'):
        path=AUDITS/'macnlops_native_ir_jets_1M_20261031/validation/summary.json'
        data=json.loads(path.read_text())[key]
        out=dict(reference=str(path.relative_to(ROOT)),sha256=sha(path),
            source_epoch=data['source_epoch'],stages={})
        for stage,field in (('first','first_emission'),('full','full_shower')):
            out['stages'][stage]=[dict(lo=x['lo'],hi=x['hi'],H_over_S=x[field]['ratio'],
                H_over_S_error=x[field]['error'],H_removal_pb=x[field]['H_pb'],S_veto_pb=x[field]['S_pb'])
                for x in data['raw_primary_ranges']]
        out['limitation']='Historical native-IR source before the general native Born-scale correction; packing epoch preserved explicitly.'
        if key=='Wj':out['limitation']+=' Both generation cuts are pTj>20 GeV, R=0.5. These archived rows are inclusive; the historical 25 GeV analysis fiducial definition is not compared with the new 40 GeV fiducial definition.'
        return {'historical native-IR '+data['source_epoch']:out}
    if key=='Wbb':
        path=AUDITS/'macnlops_wbb4fs_nocuts_1M_20261101/results/validation_1M/scalar_summary.json'
        data=json.loads(path.read_text());out=dict(reference=str(path.relative_to(ROOT)),sha256=sha(path),
            source_epoch='native-IR before native Born-scale correction',stages={})
        for stage,field in (('first','first_recoil5_20'),('full','full_recoil5_20')):
            if field not in data:continue
            x=data[field];out['stages'][stage]=[dict(lo=5.,hi=20.,H_over_S=x['H_over_S'],
                H_over_S_error=x['ratio_error'],H_removal_pb=x['H_pb'],S_veto_pb=x['S_pb'])]
        return {'historical native-IR Wbb no cuts':out}
    return {}


def compare(base,p,run):
    result=base/'results'/p['key']/run
    epochs=historical(p);current=dict(source_epoch='mcatnlo-delta-native-tc-v1; native-born-v1',stages={})
    for stage in ('first','full'):
        path=result/(stage+'_histograms.json.gz')
        if not path.exists():continue
        h=load(result,stage)['histograms']['primary_0_100'];rows=[]
        for lo,hi in RANGES:
            x=range_summary(h,lo,hi);x.update(lo=lo,hi=hi);rows.append(x)
        current['stages'][stage]=rows
    epochs['fresh native-t_c']=current
    profile_qualifications=[
        'This is a comparison of complete source/profile epochs, not an isolated change to t_c.',
        'The fresh campaign includes the native Born-scale correction and current gluon-FSR fixes; historical epochs may precede them.',
        'Fresh full showers explicitly set every hard-model particle mass. For the massless 5FS model this includes charm=0 and bottom=0. The archived DY native-IR full shower retained Pythia defaults charm~1.5 GeV and bottom~4.8 GeV.',
        'The explicit fresh model-mass profile is frozen across all eight processes; Wbb retains its massive hard bottom=4.7 GeV.',
        'Fresh and archived DY/H/tt/Wj full showers use LHAPDF6:NNPDF40_nlo_as_01180, as selected by pdfcode=1 with a hadronic LHAPDF run card. Bare-lepton EE cards contain inactive PDF13 with PDF:lepton off. Wbb uses the nf4 LHAPDF set.',
        'Historical full-shower shape differences cannot be attributed to t_c alone; inspect each archived profile/card.']
    dump(result/'epoch_comparison.json',dict(process=p['key'],run=run,epochs=epochs,
        profile_qualifications=profile_qualifications,
        interpretation='Independent source epochs, not paired events. Historical values and uncertainty estimates are preserved. Source definitions and fixes differ; see each reference.',
        observable='sqrt(Durham dmerge_N)' if p['key'].startswith('EE') else
            'pT(W+leading jet)' if p['key']=='Wj' else 'hard-system transverse recoil'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(result/'epoch_comparison.pdf') as pdf:
        for stage in current['stages']:
            fig,axes=plt.subplots(1,2,figsize=(12,4.5))
            for i,(label,data) in enumerate(epochs.items()):
                rows=data['stages'].get(stage,[])
                rows=[x for x in rows if (x['lo'],x['hi']) in RANGES and x['H_over_S'] is not None]
                if not rows:continue
                xx=[RANGES.index((x['lo'],x['hi']))+.06*(i-len(epochs)/2) for x in rows]
                axes[0].errorbar(xx,[x['H_over_S'] for x in rows],
                    yerr=[x['H_over_S_error'] or 0. for x in rows],fmt='o',label=label,ms=4)
                ratios=[x.get('matched_over_source') for x in rows]
                valid=[j for j,x in enumerate(ratios) if x is not None]
                if valid:axes[1].plot([xx[j] for j in valid],[ratios[j] for j in valid],'o',label=label,ms=4)
            for ax,title in zip(axes,('H removal / S veto','combined / source')):
                ax.axhline(1.,color='black',lw=.8);ax.grid(alpha=.2);ax.set_title(p['key']+' '+stage+' '+title)
                ax.set_xticks(np.arange(len(RANGES)));ax.set_xticklabels([f'{lo:g}–{hi:g}' for lo,hi in RANGES])
                ax.set_xlabel('primary scale range [GeV]')
            axes[0].legend(fontsize=7);fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    print('Epoch comparison ready:',result/'epoch_comparison.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--process',nargs='+',default=['EE2j','DY'],choices=[p['key'] for p in PROCESSES])
    parser.add_argument('--run',default='run_1M_native_tc');a=parser.parse_args()
    for p in PROCESSES:
        if p['key'] in a.process:compare(a.base,p,a.run)


if __name__=='__main__':main()
