from pathlib import Path
import ctypes,os,json,hashlib
import numpy as np
O=Path(__file__).parent;B=O.parents[1];D=O/'H_getter'
first=np.load(B/'results/run_1M_wbb4fs_nocuts_lhe.npz');m=json.loads((D/'run_manifest.json').read_text())
rows=[];indices=[]
for group in m['runs']:
 ix=np.load(group['source_index'])['source_index']
 rr=[json.loads(s) for s in Path(group['output']).read_text().splitlines()]
 for r in rr:
  assert r['kind']==2
  rows.append(r);indices.append(ix[r['index']-1])
indices=np.asarray(indices);assert len(np.unique(indices))==len(indices)
os.environ['LHAPDF_DATA_PATH']='/nfs/home/rikkert/LHAPDF/share/LHAPDF'
lib=ctypes.CDLL(str(O/'born_scale_factors.so'));assert lib.init_born_scale_pdf()==0
lib.born_scale_factors.argtypes=[ctypes.c_long]+[ctypes.c_void_p]*6
arrays=[np.ascontiguousarray([r[name] for r in rows],dtype=dtype) for name,dtype in [('ids',np.int32),('x',float),('muBorn',float),('muReal',float),('alphaK',float)]]
f=np.empty((len(rows),6));p=lambda z:z.ctypes.data_as(ctypes.c_void_p)
assert lib.born_scale_factors(len(rows),*[p(z) for z in arrays],p(f))==0
u=np.array([r['d']-r['r'] for r in rows]);k=np.array([r['k'] for r in rows]);A=np.array([r['A'] for r in rows]);P=np.array([r['P'] for r in rows]);scale=np.array([r['kernel_scale'] for r in rows]);recorded=np.array([r['keep'] for r in rows])
C=np.minimum(u,k);uncapped=scale*k*P/(u*A)
old=np.minimum(C/u,uncapped);new=np.minimum(C/u,uncapped*f[:,2]);keep=first['keep'][indices]
assert np.allclose(old,1-keep,rtol=2.e-10,atol=1.e-10),np.max(np.abs(old-(1-keep)))
assert np.allclose(recorded,keep,rtol=1.e-10,atol=1.e-10),np.max(np.abs(recorded-keep))
remaining=np.flatnonzero(first['is_h']&(first['nominal']<0)&~np.isin(np.arange(len(first['is_h'])),indices));assert np.all(first['keep'][remaining]==1)
w=abs(first['nominal'][indices]);delta=new-old
r=dict(negative_H_events=int(np.count_nonzero(first['is_h']&(first['nominal']<0))),transfer_capable_H_contexts=len(rows),other_H_retained=len(remaining),all_positive_Born_support=True,combined_G_range=[float(f[:,2].min()),float(f[:,2].max())],muBorn_range=[float(arrays[2].min()),float(arrays[2].max())],max_alpha_real_relative_error=float(np.max(np.abs(f[:,5]))),max_original_keep_replay_difference=float(np.max(np.abs(recorded-keep))),changed_H_removal_probabilities=int(np.count_nonzero(np.abs(new-old)>1.e-12)),old_removed_weight_pb=float(np.dot(w,old)),new_removed_weight_pb=float(np.dot(w,new)),minimum_native_bound_over_C=float(np.min(uncapped/(C/u))),minimum_corrected_bound_over_C=float(np.min(uncapped*f[:,2]/(C/u))),extra_local_beta_factor_lower_bound_to_preserve_all_H=float(np.max((C/u)/(uncapped*f[:,2]))),inputs={str(D/'run_manifest.json'):hashlib.sha256((D/'run_manifest.json').read_bytes()).hexdigest()})
np.savez_compressed(O/'H_born_scale_factors.npz',source_index=indices,values=f,muBorn=arrays[2],muReal=arrays[3],ids=arrays[0],x=arrays[1],old_removal=old,new_removal=new,bound_over_C=uncapped/(C/u),corrected_bound_over_C=uncapped*f[:,2]/(C/u))
(O/'H_bound_all.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
