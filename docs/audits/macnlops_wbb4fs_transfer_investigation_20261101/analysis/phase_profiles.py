#!/usr/bin/env python3
"""Common endpoint and S-only Born profiles; coherent beta-error counterfactual."""
exec((__import__('pathlib').Path(__file__).parent/'decompose.py').read_text().split("\nresult=")[0])
# No frozen validation outputs or input packets are written by this analysis.
fields=list(f['observable_names']);obs=f['observables'];result={'coordinate_warning':'H hard packet is real while S hard packet is Born. These source packet coordinates must not be compared as a common H/S cancellation variable. Common endpoint observables are used below; source Born profiles are S only.','regions':{}}
for scope,mask in [('inclusive',np.ones(n,dtype=bool)),('recoil5_20',(f['pt']>=5)&(f['pt']<20))]:
 d={'endpoint_profiles':{},'S_Born_profiles':{},'signed_shape_counterfactuals':{},'beta_error_counterfactuals':{}}
 specs=[('system_mass',[0,100,120,150,200,300,500,1000,np.inf]),('bb_mass',[0,10,15,20,30,50,100,200,np.inf]),('bb_pt',[0,5,10,20,30,50,100,200,np.inf]),('b_pt',[0,5,10,20,30,50,100,200,np.inf]),('bbar_pt',[0,5,10,20,30,50,100,200,np.inf]),('w_pt',[0,5,10,20,30,50,100,200,np.inf]),('bb_delta_r',[0,.25,.5,1,2,3,4,6,10,np.inf]),('system_y',[-np.inf,-3,-2,-1,0,1,2,3,np.inf])]
 for field,edges in specs:
  x=obs[:,fields.index(field)];d['endpoint_profiles'][field]=[dict(lo=lo,hi=hi,**row(mask&(x>=lo)&(x<hi))) for lo,hi in zip(edges[:-1],edges[1:])]
 for field,x,edges in [('hard_HT_half',V('hard_ht_half'),[0,45,50,55,60,70,80,100,150,250,np.inf]),('hard_bb_mass',V('hard_bb_mass'),[0,10,15,20,30,50,100,200,np.inf]),('hard_system_mass',V('hard_system_mass'),[0,100,120,150,200,300,500,1000,np.inf]),('hard_bb_delta_r',V('hard_bb_delta_r'),[0,.25,.5,1,2,3,4,6,10,np.inf]),('hard_xmax',np.maximum(V('hard_x1'),V('hard_x2')),[0,.01,.03,.1,.3,1,np.inf]),('HT_real_over_Born',ratht,[0,1,1.05,1.1,1.2,1.5,2,3,5,np.inf])]:
  rows=[]
  for lo,hi in zip(edges[:-1],edges[1:]):
   sel=mask&~h&(x>=lo)&(x<hi);pos=sel&(w>0);neg=sel&(w<0)
   out=dict(lo=lo,hi=hi,**row(sel));out.update(positive_source_pb=float(w[pos].sum()),negative_source_pb=float(w[neg].sum()),positive_average_rejection=float(prob[pos].sum()/w[pos].sum()) if pos.any() else None,negative_average_rejection=float(prob[neg].sum()/w[neg].sum()) if neg.any() else None)
   rows.append(out)
  d['S_Born_profiles'][field]=rows
 for label,cat in [('global',np.zeros(n,int)),('per_channel',channel)]:
  cf=0.;actual_s=0.;details=[];influence=np.zeros(n)
  for c in np.unique(cat[mask&~h]):
   sel=mask&~h&(cat==c);pos=sel&(w>0);neg=sel&(w<0);wp=w[pos].sum();wn=w[neg].sum();sp=S[pos].sum();sn=S[neg].sum();value=sp*(wp+wn)/wp;cf+=value;actual_s+=sp+sn
   influence[neg]=S[neg]-(sp/wp)*w[neg];influence[pos]=-(wn/wp)*(S[pos]-(sp/wp)*w[pos])
   details.append(dict(group=int(c),positive_mean_rejection=float(sp/wp),negative_mean_rejection=float(sn/wn) if wn else None,actual_S_pb=float(sp+sn),same_negative_shape_S_pb=float(value)))
  d['signed_shape_counterfactuals'][label]=dict(note='Diagnostic only: assign negative S the positive-S mean rejection in each group. Does not determine true local S/B and is not a matching prescription.',actual_S_pb=float(actual_s),same_negative_shape_S_pb=float(cf),S_bias_from_negative_shape_pb=float(actual_s-cf),details=details,paired_shape_bias_error_pb=error(influence))
 # Conservative coherent ratio excursion: use delta(S/B)/(S/B) <= dS/S+dB/B.
 relative={int(key.split(':')[0])*10+int(key.split(':')[1]):x['sigma_S_error']/x['sigma_S']+x['sigma_B_error']/x['sigma_B'] for key,x in manifest['beta_groups'].items()}
 shift=np.array([relative[int(c)] for c in channel])
 for sign in [-1,1]:
  pp=acceptance(scale*(1+sign*shift),P);delta=w*(1-pp);ss=np.where(~h,delta,0);hh=np.where(h,-delta,0)
  d['beta_error_counterfactuals'][str(sign)]=dict(relative_error_bound=relative,S_pb=float(ss[mask].sum()),H_pb=float(hh[mask].sum()),net_change_pb=float(((hh-ss)-(H-S))[mask].sum()))
 result['regions'][scope]=d
def serial(x):
 if isinstance(x,float):return x if np.isfinite(x) else None
 if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
 if isinstance(x,list):return [serial(y) for y in x]
 return x
(O/'phase_profiles.json').write_text(json.dumps(serial(result),indent=2)+'\n')
for scope,d in result['regions'].items():
 print(scope,'negative_shape',json.dumps(d['signed_shape_counterfactuals']))
 print('BETA_ERROR',json.dumps(d['beta_error_counterfactuals']))
 print('ENDPOINT MASS NET',[(x['lo'],x['hi'],x['expected_net_pb'],x['expected_net_error_pb']) for x in d['endpoint_profiles']['system_mass']])
