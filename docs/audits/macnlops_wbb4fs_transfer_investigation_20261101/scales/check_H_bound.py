from pathlib import Path
import json,ctypes,os
import numpy as np
O=Path(__file__).parent;B=O.parent
rows=[json.loads(l) for l in (B/'kernel/inverse_born.jsonl').read_text().splitlines()];rows=[r for r in rows if r['kind']==2]
os.environ['LHAPDF_DATA_PATH']='/nfs/home/rikkert/LHAPDF/share/LHAPDF'
lib=ctypes.CDLL(str(O/'born_scale_factors.so'));assert lib.init_born_scale_pdf()==0
lib.born_scale_factors.argtypes=[ctypes.c_long]+[ctypes.c_void_p]*6
arrays=[np.ascontiguousarray([r[name] for r in rows],dtype=dtype) for name,dtype in [('ids',np.int32),('x',float),('muBorn',float),('muReal',float),('alphaK',float)]]
f=np.empty((len(rows),6));p=lambda z:z.ctypes.data_as(ctypes.c_void_p)
assert lib.born_scale_factors(len(rows),*[p(z) for z in arrays],p(f))==0
u=np.array([r['d']-r['r'] for r in rows]);k=np.array([r['k'] for r in rows]);A=np.array([r['A'] for r in rows]);P=np.array([r['P'] for r in rows]);scale=np.array([r['kernel_scale'] for r in rows])
C=np.minimum(u,k);uncapped=scale*k*P/(u*A)
old=np.minimum(C/u,uncapped);new=np.minimum(C/u,uncapped*f[:,2]);keep=np.array([r['keep'] for r in rows]);assert np.allclose(old,1-keep,rtol=1.e-12,atol=1.e-13)
r=dict(H_contexts=len(rows),all_positive_Born_support=True,combined_G_range=[float(f[:,2].min()),float(f[:,2].max())],max_alpha_real_relative_error=float(np.max(np.abs(f[:,5]))),changed_H_removal_probabilities=int(np.count_nonzero(np.abs(new-old)>1.e-13)),old_removed_weight_pb=float(sum(abs(z['weight'])*a for z,a in zip(rows,old))),new_removed_weight_pb=float(sum(abs(z['weight'])*a for z,a in zip(rows,new))),minimum_native_bound_over_C=float(np.min(uncapped/(C/u))),minimum_corrected_bound_over_C=float(np.min(uncapped*f[:,2]/(C/u))),limits='223H from one9,973-source shard; not a fullHpopulation correction.')
(O/'H_bound_subset.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
