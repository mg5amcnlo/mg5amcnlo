#!/usr/bin/env python3
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent;d=json.loads((P/'decomposition.json').read_text());phase=json.loads((P/'phase_profiles.json').read_text())
fig=plt.figure(figsize=(11,10));labels=['ISR g, beam 1','ISR g, beam 2','FSR bottom','FSR antibottom','ISR conversion, beam 1','ISR conversion, beam 2']
for n,scope in enumerate(['inclusive','recoil5_20']):
 ax=fig.add_axes([.26,.66-.31*n,.67,.23]);r=[x for x in d['regions'][scope]['sector'] if x['class']!='No evaluated branch'];y=np.arange(6)
 ax.barh(y,[x['expected_net_pb'] for x in r],color=['#4477aa','#4477aa','#cc6677','#cc6677','#228833','#228833'],alpha=.8)
 ax.errorbar([x['expected_net_pb'] for x in r],y,xerr=[x['expected_net_error_pb'] for x in r],fmt='none',ecolor='black',capsize=2)
 ax.set_yticks(y,labels);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Expected H-removal minus signed S-veto [pb]')
 ax.set_title('Inclusive' if n==0 else '5 ≤ pT(W+b+bbar) < 20 GeV')
ax=fig.add_axes([.26,.085,.67,.19]);r=phase['regions']['inclusive']['endpoint_profiles']['system_mass'];labels=[f"{x['lo']:g}–{x['hi']:g}" if x['hi'] is not None else f"≥{x['lo']:g}" for x in r];x=np.arange(len(r))
ax.bar(x,[x['expected_net_pb'] for x in r],color='#4477aa',alpha=.8);ax.errorbar(x,[x['expected_net_pb'] for x in r],yerr=[x['expected_net_error_pb'] for x in r],fmt='none',ecolor='black',capsize=2);ax.axhline(0,color='black',lw=.8);ax.grid(axis='y',alpha=.2);ax.set_xticks(x,labels);ax.set_xlabel('Common first-endpoint m(W+b+bbar) [GeV]');ax.set_ylabel('Expected H − S [pb]')
fig.suptitle('No-cut W+bbbar in 4FS: location of the transfer deficit\n1M source events; recorded probabilities, before Bernoulli draws',y=.975)
fig.text(.5,.027,'Errors: paired 50 source chunks, conditional on hard integration, channel beta and the recorded shower model.',ha='center',fontsize=8)
fig.savefig(P/'transfer_decomposition.pdf');fig.savefig(P/'transfer_decomposition.png',dpi=120);plt.close(fig)
