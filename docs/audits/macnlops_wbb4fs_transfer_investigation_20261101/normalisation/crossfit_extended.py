#!/usr/bin/env python3
"""Held-out signed-S / independent-positive-B density controls, no physics rerun."""
from pathlib import Path
import hashlib,json,numpy as np
OUT=Path(__file__).resolve().parent;DIAG=OUT.parent;BASE=OUT.parents[1]
D=np.load(DIAG/'analysis/decision_metadata.npz');B=np.load(OUT/'born_control.npz');G=np.load(DIAG/'scales/S_born_scale_factors.npz');FIRST=np.load(BASE/'results/run_1M_wbb4fs_nocuts_lhe.npz')
f=list(D['fields']);bf=list(B['fields']);v=D['values'];bv=B['values'];w=D['weight'];bw=B['weight'];s=D['kind']==1;h=~s;n=len(w);nb=len(bw)
assert n==1000000 and nb==200000 and np.array_equal(D['source_index'],np.arange(n))
ai=G['source_index'];raw=G['raw_rejection'];scale=G['values'][:,list(G['fields']).index('combined')];beta=D['beta'][ai]
assert np.max(abs(raw-(1-v[ai,f.index('keep')])))<1e-12
assert np.all(s[ai]) and np.all(scale>0) and np.all(beta>0)
recoil=FIRST['observables'][:,0];recoilbin=(recoil>=5)&(recoil<20)
ha=abs(w[h])*(1-v[h,f.index('keep')]);h_indices=np.flatnonzero(h)
fold=np.random.default_rng(4930301).integers(0,2,n);blocks=np.random.default_rng(4930302).integers(0,20,n);bblocks=np.random.default_rng(4930303).integers(0,20,nb)
models=[('mWbb_20',['hard_system_mass'],20),('mWbb_40',['hard_system_mass'],40),('mWbb_fixedlog',['hard_system_mass'],{'hard_system_mass':[90,100,110,120,140,160,200,250,350,500,700,1000,1500,3000,13000]}),('mWbb_y_6x6',['hard_system_mass','hard_system_y'],6),('mWbb_y_8x8',['hard_system_mass','hard_system_y'],8),('x1_x2_8x8',['hard_x1','hard_x2'],8)]

# Quantile boundaries depend only on the independent positive Born covariates.
def assignments(names,nq):
 if not names:return np.zeros(n,dtype=int),np.zeros(nb,dtype=int),1,{}
 codes=np.zeros(n,dtype=int);bcodes=np.zeros(nb,dtype=int);nodes={};mult=1
 for name in names:
  edges=np.array(nq[name],dtype=float) if isinstance(nq,dict) else np.unique(np.quantile(bv[:,bf.index(name)],np.arange(1,nq)/nq));nodes[name]=edges.tolist()
  codes+=mult*np.searchsorted(edges,v[:,f.index(name)],side='right');bcodes+=mult*np.searchsorted(edges,bv[:,bf.index(name)],side='right');mult*=len(edges)+1
 return codes,bcodes,mult,nodes

def evaluate(codes,bcodes,ncells,excluded=None,keepdetails=False):
 ok=np.ones(n,dtype=bool) if excluded is None else blocks!=excluded
 bok=np.ones(nb,dtype=bool) if excluded is None else bblocks!=excluded
 bwscale=nb/bok.sum();evalscale=n/ok.sum()
 bcounts=np.bincount(bcodes[bok],minlength=ncells);binnedB=np.bincount(bcodes[bok],weights=bw[bok]*bwscale,minlength=ncells)
 reject=np.empty(len(ai));ratios=np.ones(len(ai));fallback=np.zeros(len(ai),dtype=bool);details=[]
 for target in [0,1]:
  train=ok&(fold!=target);held=(fold[ai]==target);trainS=train&s
  trainfactor=n/train.sum()
  signed=np.bincount(codes[trainS],weights=w[trainS]*trainfactor,minlength=ncells)
  absolute=np.bincount(codes[trainS],weights=abs(w[trainS])*trainfactor,minlength=ncells)
  sq=np.bincount(codes[trainS],weights=(w[trainS]*trainfactor)**2,minlength=ncells)
  local=np.divide(signed,binnedB,out=np.full(ncells,np.nan),where=binnedB>0)
  valid=(bcounts>=100)&np.isfinite(local)&(local>0)
  bins=codes[ai[held]];good=valid[bins]
  factor=np.ones(held.sum());factor[good]=local[bins[good]]/beta[held][good]
  ratios[held]=factor;fallback[held]=~good
  reject[held]=np.minimum(1,raw[held]/scale[held]/factor)
  if keepdetails:
   details.append(dict(heldout_fold=target,train_source_events=int(train.sum()),bins=[dict(bin=i,B_events=int(bcounts[i]),B_pb=float(binnedB[i]),S_pb=float(signed[i]),S_abs_pb=float(absolute[i]),S_error_pb=float(np.sqrt(sq[i])),beta_local=float(local[i]) if np.isfinite(local[i]) else None,valid=bool(valid[i])) for i in range(ncells)]))
 active_ok=ok[ai];wt=w[ai]*evalscale
 Hweights=ha[ok[h_indices]]*evalscale;Hrecoil=recoilbin[h_indices[ok[h_indices]]]
 values={}
 for label,mask in [('inclusive',np.ones(len(ai),dtype=bool)),('recoil5_20',recoilbin[ai])]:
  selected=active_ok&mask;S=float(np.sum(wt[selected]*reject[selected]));H=float(np.sum(Hweights if label=='inclusive' else Hweights[Hrecoil]));values[label]=dict(S_pb=S,H_pb=H,H_over_S=H/S,net_H_minus_S_pb=H-S)
 if keepdetails:
  values['fit_details']=details;values['fallback_active_S_events']=int(fallback.sum());values['fallback_target_S_events']=int(np.sum(fallback&recoilbin[ai]));values['saturated_S_events']=int(np.sum(reject>=1));values['local_factor_range_active_S']=np.quantile(ratios,[0,.01,.1,.5,.9,.99,1]).tolist();values['min_global_local_factor']=float(min(row['beta_local'] for foldinfo in details for row in foldinfo['bins'] if row['valid'])/max(beta))
 return values

report=dict(method='Seeded twofold cross-fit: for each held-out source fold, estimate signed S/B from the opposite source fold and independent positive Born control. Fits use only Born covariates/all S source weights, never H/S transfer targets. Apply min(1,qraw/G * beta_channel/beta_local). Empty/nonpositive/under100-Born-event cells retain beta_channel (factor1), disclosed. H fixed because native H decisions are source-limited; generic H safety factor .42432 checked separately.',
 seeds=dict(folds=4930301,source_jackknife_blocks=4930302,Born_jackknife_blocks=4930303),jackknife_blocks=20,
 uncertainties='Delete-one of20 independent seeded source/Born blocks, renormalizing both densities and held-out event sums; includes transfer sampling and fitted-ratio statistics, omits hard integration uncertainties and higher-dimensional model bias.',
 Born_events=nb,source_S_events=int(s.sum()),active_S_events=len(ai),Born_total_pb=float(bw.sum()),source_S_total_pb=float(w[s].sum()),models={})
for name,names,nq in models:
 codes,bcodes,ncells,edges=assignments(names,nq);main=evaluate(codes,bcodes,ncells,keepdetails=True)
 reps=[evaluate(codes,bcodes,ncells,excluded=j) for j in range(20)]
 for label in ['inclusive','recoil5_20']:
  for key in ['S_pb','H_pb','H_over_S','net_H_minus_S_pb']:
   a=np.array([rep[label][key] for rep in reps]);main[label][key+'_jackknife_error']=float(np.sqrt(19/20*np.sum((a-a.mean())**2)))
 main.update(fields=names,quantile_internal_edges=edges,cells=ncells,jackknife_replicates=reps)
 report['models'][name]=main;print(name,{k:main[k] for k in ['inclusive','recoil5_20','fallback_active_S_events','saturated_S_events','min_global_local_factor']},flush=True)
report['inputs']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [DIAG/'analysis/decision_metadata.npz',OUT/'born_control.npz',DIAG/'scales/S_born_scale_factors.npz',Path(__file__)]}
(OUT/'crossfit_extended.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
