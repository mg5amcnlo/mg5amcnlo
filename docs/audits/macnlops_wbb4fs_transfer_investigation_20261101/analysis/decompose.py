#!/usr/bin/env python3
"""Signed expected/realized transfer decomposition from recorded production tags."""
from pathlib import Path
import json,numpy as np
B=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101');O=B/'diagnosis_transfer_20261101/analysis'
z=np.load(O/'decision_metadata.npz');f=np.load(B/'results/run_1M_wbb4fs_nocuts_lhe.npz');v=z['values'];N={x:i for i,x in enumerate(z['fields'])};V=lambda k:v[:,N[k]]
w,m,h,p=f['nominal'],f['macnlops'],f['is_h'],f['keep'];n=len(w);groups=np.arange(n)//20000;G=50
actual=w-m;prob=w*(1-p);S=np.where(~h,prob,0.);H=np.where(h,-prob,0.);SA=np.where(~h,actual,0.);HA=np.where(h,-actual,0.)
def error(x,center=True):
 sums=np.bincount(groups,weights=x,minlength=G)
 if center:sums-=sums.mean()
 return float(np.sqrt(G/(G-1))*np.linalg.norm(sums))
def row(mask):
 s,t=float(S[mask].sum()),float(H[mask].sum());r=t/s if s else None;net=H-S
 a,b=float(SA[mask].sum()),float(HA[mask].sum());real=HA-SA
 return dict(events=int(mask.sum()),source_pb=float(w[mask].sum()),S_positive_pb=float(S[mask&(w>0)].sum()),S_negative_pb=float(S[mask&(w<0)].sum()),expected_S_pb=s,expected_H_pb=t,expected_net_pb=t-s,expected_net_error_pb=error(net*mask),expected_H_over_S=r,expected_ratio_error=error(mask*(H-r*S)/s,False) if r is not None and s>0 else None,realized_S_pb=a,realized_H_pb=b,realized_net_pb=b-a,draw_net_shift_pb=(b-a)-(t-s),draw_error_pb=float(np.sqrt(np.dot(w[mask]**2,p[mask]*(1-p[mask])))))
def table(cat,mask,labels=None):
 out=[]
 for x in np.unique(cat[mask]):
  r=row(mask&(cat==x));r['class']=str(x) if labels is None else labels.get(int(x),str(x));out.append(r)
 return out
u=np.maximum(V('d')-V('r'),0);k=V('k');C=np.minimum(u,k);A=V('delta_source_mean');P=V('delta_shower_mean');scale=V('kernel_scale')
valid=(u>0)&(k>0)&(A>0)&(P>0)&(scale>0)&np.isfinite(u+k+A+P+scale)
def acceptance(newscale=scale,newP=P):
 rej=np.zeros(n);ok=valid&(newscale>0)&(newP>0)&np.isfinite(newscale+newP)
 rej[ok&~h]=np.minimum(1.,C[ok&~h]*A[ok&~h]/(k[ok&~h]*newscale[ok&~h]*newP[ok&~h]))
 rej[ok&h]=np.minimum(C[ok&h]/u[ok&h],k[ok&h]*newscale[ok&h]*newP[ok&h]/(A[ok&h]*u[ok&h]))
 return 1.-np.minimum(rej,1.)
reconstructed=acceptance();assert np.allclose(reconstructed,p,atol=3e-14,rtol=1e-13),np.max(abs(reconstructed-p))
reg=z['registry'];config=V('config').astype(int);channel=reg*10+config
manifest=json.loads((O/'metadata_manifest.json').read_text());clabel={int(k.split(':')[0])*10+int(k.split(':')[1]):x['name'] for k,x in manifest['beta_groups'].items()}
sector=np.nan_to_num(V('sector')).astype(int)
sectorlabel={0:'No evaluated branch',1:'ISR beam1 gluon',2:'ISR beam2 gluon',3:'FSR bottom gluon',4:'FSR antibottom gluon',5:'ISR beam1 flavour conversion',6:'ISR beam2 flavour conversion'}
classcat=np.where(h,2,0)+(w<0);classlabel={0:'S_positive',1:'S_negative',2:'H_positive',3:'H_negative'}
ratht=V('endpoint_ht_half')/V('hard_ht_half');alpha=V('alpha_s_shower')/V('alpha_s_kernel');pdf=V('pdf_ratio')
result={'source_packet_coordinate_warning':'S hard packets are original Born while H hard packets are real. Source packet feature tables are not common-coordinate H/S cancellation tests; use phase_profiles.json endpoint_profiles and S-only Born profiles instead.','events':n,'primary':'pT(W+b+bbar)','errors':'Paired 50 source chunks, conditional on hard integration, channel normalization and recorded shower model; matching-draw variance is separate.','keep_reconstruction_max_abs':float(np.max(abs(reconstructed-p))),'regions':{}}
for name,mask in [('inclusive',np.ones(n,dtype=bool)),('recoil5_20',(f['pt']>=5)&(f['pt']<20))]:
 d={'total':row(mask),'sign':table(classcat,mask,classlabel),'channel':table(channel,mask,clabel),'sector':table(sector,mask,sectorlabel),'channel_sector':table(channel*10+sector,mask), 'rule':table(z['rule'],mask),'ranges':{},'counterfactuals':{}}
 for field,x,edges in [('emission_pt',V('pt'),[0,.5,1,2,5,10,20,50,100,1000,np.inf]),('hard_HT_half',V('hard_ht_half'),[0,50,60,80,100,150,250,500,np.inf]),('HT_real_over_hard',ratht,[0,1,1.05,1.1,1.2,1.5,2,3,5,np.inf]),('hard_bb_mass',V('hard_bb_mass'),[0,10,15,20,30,50,100,200,np.inf]),('hard_system_mass',V('hard_system_mass'),[0,100,120,150,200,300,500,1000,np.inf]),('hard_xmax',np.maximum(V('hard_x1'),V('hard_x2')),[0,.001,.003,.01,.03,.1,.3,1,np.inf]),('alpha_shower_over_kernel',alpha,[0,1,1.5,2,2.5,3,5,10,np.inf]),('delta_shower',P,[0,.01,.03,.1,.2,.4,.6,.8,1.,np.inf])]:
  d['ranges'][field]=[dict(lo=lo,hi=hi,**row(mask&np.isfinite(x)&(x>=lo)&(x<hi))) for lo,hi in zip(edges[:-1],edges[1:])]
 for cname,newscale,newP in [('beta_one',scale/z['beta'],P),('emission_coupling_one',scale/alpha,P),('emission_pdf_ratio_one',scale/pdf,P),('shower_sudakov_equals_source',scale,A)]:
  pp=acceptance(newscale,newP);delta=w*(1-pp);ss=np.where(~h,delta,0);hh=np.where(h,-delta,0)
  s,t=float(ss[mask].sum()),float(hh[mask].sum());d['counterfactuals'][cname]=dict(note='Expected-weight diagnostic on frozen source/endpoint sample; not a validated physical matching prescription.',S_pb=s,H_pb=t,H_over_S=t/s if s else None,net_pb=t-s,net_change_pb=float(((hh-ss)-(H-S))[mask].sum()),net_change_paired_error_pb=error(mask*((hh-ss)-(H-S))))
 d['saturation']={}
 for label,sel in [('S_keep_zero',~h&(p==0)),('S_keep_partial',~h&(p>0)&(p<1)),('H_keep_zero',h&(p==0)),('H_keep_partial',h&(p>0)&(p<1))]:d['saturation'][label]=row(mask&sel)
 d['source_packet_ranges_diagnostic_only']={key:d['ranges'].pop(key) for key in list(d['ranges']) if key.startswith('hard_') or key=='HT_real_over_hard'}
 result['regions'][name]=d
# JSON uses null for open upper bin edges.
def serial(x):
 if isinstance(x,float):return x if np.isfinite(x) else None
 if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
 if isinstance(x,list):return [serial(y) for y in x]
 return x
(O/'decomposition.json').write_text(json.dumps(serial(result),indent=2)+'\n')
for region,d in result['regions'].items():
 print(region,json.dumps(d['total']))
 for row in d['sector']:print('SECTOR',row['class'],row['expected_S_pb'],row['expected_H_pb'],row['expected_H_over_S'],row['expected_net_pb'],row['expected_net_error_pb'])
 print('COUNTERFACTUALS',json.dumps(d['counterfactuals']))
