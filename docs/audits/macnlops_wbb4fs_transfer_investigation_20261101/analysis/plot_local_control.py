#!/usr/bin/env python3
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent;d=json.loads((P/'local_prediction_crosscheck.json').read_text());models=[('baseline','Recorded matcher','#4477aa'),('Born_scale_factor','+ Born-scale density factor','#ee7733'),('predictions_mWbb_HT2_12x12','+ held-out local beta (12×12)','#228833')]
fig=plt.figure(figsize=(11,10));labels=['ISR g, beam 1','ISR g, beam 2','FSR bottom','FSR antibottom','ISR conversion, beam 1','ISR conversion, beam 2']
for n,scope in enumerate(['inclusive','recoil5_20']):
 ax=fig.add_axes([.26,.55-.40*n,.67,.30])
 for j,(key,label,color) in enumerate(models):
  rows=d['results'][scope][key][1:];x=[r['net_pb'] for r in rows];y=np.arange(6)+(j-1)*.2
  ax.errorbar(x,y,xerr=[r['conditional_net_error_pb'] for r in rows],fmt='o',ms=4,color=color,capsize=2,label=label)
 ax.set_yticks(np.arange(6),labels);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Expected H-removal minus signed S-veto [pb]');ax.set_title('Inclusive' if n==0 else '5 ≤ pT(W+b+bbar) < 20 GeV');ax.set_xlim(-.8,.2)
 if n==0:ax.legend(loc='upper right',bbox_to_anchor=(1.,1.24),fontsize=9,frameon=False)
fig.suptitle('No-cut W+bbbar in 4FS: density-factor controls\nLocal-beta correction improves closure; radiator residuals remain',y=.97)
fig.text(.5,.055,'Frozen first-emission probabilities; 1M source events. Local beta trained on independent 200k Born control with held-out S folds.',ha='center',fontsize=8)
fig.text(.5,.038,'Displayed errors condition on fitted factors; normalization-agent results include their fit/control uncertainty. No end-to-end matching rerun.',ha='center',fontsize=8)
fig.savefig(P/'local_density_control.pdf');fig.savefig(P/'local_density_control.png',dpi=120);plt.close(fig)
