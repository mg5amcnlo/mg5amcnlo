from pathlib import Path
import json,numpy as np
O=Path(__file__).parent;B=O.parents[1]
x=np.load(B/'results/run_1M_wbb4fs_nocuts_lhe.npz');g=np.load(O/'S_born_scale_factors.npz')
w=x['nominal'];h=x['is_h'];q=1-x['keep'];pt=x['pt'];ix=g['source_index'];G=g['values'][:,2]
new=q.copy();new[ix]=np.minimum(1.,g['raw_rejection']/G);group=np.arange(len(w))//20000
err=lambda z:float(np.sqrt(50/49*np.sum(np.bincount(group,weights=z,minlength=50)**2)))
rows=[]
for lo,hi in [(5,20),(0,None)]:
 sel=pt>=lo
 if hi is not None:sel&=pt<hi
 H=np.where(sel&h,-w*q,0.);X0=np.where(sel&~h,w*q,0.);X1=np.where(sel&~h,w*new,0.)
 Y=H.sum();S0=X0.sum();S1=X1.sum();r0=Y/S0;r1=Y/S1
 z=X0-X1;chunks=np.bincount(group,weights=z,minlength=50)
 rows.append(dict(lo=lo,hi=hi,ratio_before=float(r0),ratio_after=float(r1),paired_ratio_change=float(r1-r0),paired_ratio_change_error=err((H-r1*X1)/S1-(H-r0*X0)/S0),net_improvement_pb=float(z.sum()),net_improvement_error_pb=float(np.sqrt(50/49*np.sum((chunks-chunks.mean())**2))),fraction_deficit_explained=float(z.sum()/(S0-Y))))
(O/'paired_scale_change.json').write_text(json.dumps({'method':'Expected transfers, same original1M endpoints; paired50contiguous20k sourcechunks. Independent all24003-H replay H_bound_all.json proves the same G leaves H unchanged.','results':rows},indent=2)+'\n');print(json.dumps(rows,indent=2))
