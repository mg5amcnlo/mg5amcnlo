from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D=Path(__file__).parent
base=json.loads((D/'scales/S_scale_counterfactual.json').read_text())['results']
beta=json.loads((D/'normalisation/normalisation_summary.json').read_text())['models']['mWbb_HT2_12x12']
fig,axes=plt.subplots(1,2,figsize=(10,4.4))
for ax,hi,label in zip(axes,[20,None],['5 ≤ pT(W+b+bbar) < 20 GeV','All events']):
 rows=[next(r for r in base if r['lo']==(5 if hi else 0) and r['hi']==hi and r['control']==name) for name in ['baseline','Born_coupling_and_PDF']]
 z=beta['recoil5_20' if hi else 'inclusive']
 ys=[r['H_over_S'] for r in rows]+[z['H_over_S']]
 es=[r['ratio_error'] for r in rows]+[z['H_over_S_jackknife_error']]
 ax.errorbar(np.arange(3),ys,yerr=es,fmt='o',capsize=4,color='#2166ac')
 ax.axhline(1,color='black',lw=.8,ls='--');ax.set_ylim(.86,1.035);ax.set_xlim(-.4,2.4)
 ax.set_xticks(np.arange(3),['Current','Born-scale G','G + local S/B\n(two Born variables)']);ax.set_ylabel('Expected H-removal / signed S-veto');ax.set_title(label);ax.grid(alpha=.2)
 for i,y in enumerate(ys):ax.annotate(f'{y:.4f}',(i,y),xytext=(0,-20),textcoords='offset points',ha='center',fontsize=9)
fig.suptitle('W+bbbar, 4FS: diagnostic density corrections on fixed 1M no-cut inputs')
fig.subplots_adjust(top=.83,bottom=.25,wspace=.38)
fig.text(.5,.065,'Expected transfers average over the matching draw. Local S/B uses held-out signed S and independent 200k Born events.',ha='center',fontsize=8)
fig.text(.5,.025,'Conditional errors: paired 50 chunks for first two controls; 20-block fitted-ratio jackknife for local S/B. Radiator residuals remain.',ha='center',fontsize=8)
fig.savefig(D/'density_controls.pdf');fig.savefig(D/'density_controls.png',dpi=150);plt.close(fig)
