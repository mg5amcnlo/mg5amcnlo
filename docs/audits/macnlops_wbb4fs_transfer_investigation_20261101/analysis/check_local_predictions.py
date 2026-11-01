#!/usr/bin/env python3
"""Independent event-level reconciliation of held-out local-beta predictions."""
from pathlib import Path
import json,numpy as np,hashlib
B=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101');D=B/'diagnosis_transfer_20261101';A=D/'analysis'
f=np.load(B/'results/run_1M_wbb4fs_nocuts_lhe.npz');z=np.load(A/'decision_metadata.npz');v=z['values'];fields=list(z['fields']);w=f['nominal'];p=f['keep'];h=f['is_h'];sector=np.nan_to_num(v[:,fields.index('sector')]).astype(int);groups=np.arange(len(w))//20000
scale=np.load(D/'scales/S_born_scale_factors.npz');idx=scale['source_index'];G=scale['values'][:,list(scale['fields']).index('combined')]
assert np.array_equal(idx,np.flatnonzero(~h&(p<1)))
H=np.where(h,-w*(1-p),0);rejections={'baseline':1-p};rej=np.zeros(len(w));rej[idx]=(1-p[idx])/G;rejections['Born_scale_factor']=rej
proof={};filebindings=[]
for file in sorted((D/'normalisation').glob('predictions_*.npz')):
 pred=np.load(file);assert np.array_equal(pred['source_index'],idx)
 target=(1-p[idx])/G/pred['local_factor'];assert np.allclose(target,pred['rejection'],rtol=5e-12,atol=1e-15),file
 assert (target>0).all() and (target<1).all();rej=np.zeros(len(w));rej[idx]=pred['rejection'];rejections[file.stem]=rej
 proof[file.stem]=dict(max_abs_probability_difference=float(np.max(abs(target-pred['rejection']))),fallback_events=int(pred['fallback'].sum()),local_factor_range=[float(pred['local_factor'].min()),float(pred['local_factor'].max())])
 filebindings.append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
def err(x):
 sums=np.bincount(groups,weights=x,minlength=50);sums-=sums.mean();return float(np.sqrt(50/49)*np.linalg.norm(sums))
results={}
for scope,mask in [('inclusive',np.ones(len(w),bool)),('recoil5_20',(f['pt']>=5)&(f['pt']<20))]:
 models={}
 for model,rej in rejections.items():
  S=np.where(~h,w*rej,0);r=[]
  for sec in [0,1,2,3,4,5,6]:
   take=mask&((sector==sec) if sec else np.ones(len(w),bool));ss=float(S[take].sum());hh=float(H[take].sum());r.append(dict(sector=sec,S_pb=ss,H_pb=hh,H_over_S=hh/ss if ss else None,net_pb=hh-ss,conditional_net_error_pb=err((H-S)*take)))
  models[model]=r
 results[scope]=models
out=dict(probability_identity_proofs=proof,errors='Conditional paired50-source-chunk errors for frozen fitted factors. Fit/LO-control uncertainty is reported separately by normalization agent; do not replace it with these errors.',H_probabilities='Root independently verified all24003 inverse-projected H correction factors leave frozen H keeps bit-identical.',prediction_bindings=filebindings,results=results)
(A/'local_prediction_crosscheck.json').write_text(json.dumps(out,indent=2)+'\n')
for scope,models in results.items():
 print(scope)
 for model in ['baseline','Born_scale_factor','predictions_mWbb_HT2_8x8']:
  if model in models:print(model,[(x['sector'],x['H_over_S'],x['net_pb']) for x in models[model]])
