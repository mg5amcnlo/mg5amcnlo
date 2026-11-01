from pathlib import Path
import ctypes,json,os
import numpy as np
B=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101')
O=Path(__file__).parent
os.environ['LHAPDF_DATA_PATH']='/nfs/home/rikkert/LHAPDF/share/LHAPDF'
x=np.load(B/'diagnosis_transfer_20261101/analysis/decision_metadata.npz')
y=np.load(B/'results/run_1M_wbb4fs_nocuts_lhe.npz')
v=x['values'];keys={s:i for i,s in enumerate(x['fields'])};get=lambda s:v[:,keys[s]]
w=x['weight'];h=y['is_h'];keep=get('keep');pt=y['pt'];group=np.arange(len(w))//20000
active=(~h)&(keep<1)&np.isfinite(get('delta_source_mean'))
ix=np.flatnonzero(active)
ids=np.ascontiguousarray(x['incoming_ids'][ix,:2],dtype=np.int32)
frac=np.ascontiguousarray(np.column_stack((get('hard_x1')[ix],get('hard_x2')[ix])))
born=np.ascontiguousarray(get('hard_ht_half')[ix]);real=np.ascontiguousarray(get('endpoint_ht_half')[ix]);ak=np.ascontiguousarray(get('alpha_s_kernel')[ix])
a=np.empty((len(ix),6))
lib=ctypes.CDLL(str(O/'born_scale_factors.so'));assert lib.init_born_scale_pdf()==0
p=lambda z:z.ctypes.data_as(ctypes.c_void_p)
lib.born_scale_factors.argtypes=[ctypes.c_long]+[ctypes.c_void_p]*6
status=lib.born_scale_factors(len(ix),p(ids),p(frac),p(born),p(real),p(ak),p(a));assert status==0,status
maxalpha=float(np.max(np.abs(a[:,5])));print('active S',len(ix),'maximum alpha(HTreal)/alphaK−1',maxalpha,flush=True)
# Reconstruct the *unclipped* rejection, so counterfactuals handle saturation.
u=get('d')[ix]-get('r')[ix];k=get('k')[ix];A=get('delta_source_mean')[ix];P=get('delta_shower_mean')[ix];scale=get('kernel_scale')[ix]
raw=np.minimum(u,k)*A/(k*scale*P)
assert np.allclose(np.minimum(raw,1.),1.-keep[ix],rtol=1.e-11,atol=1.e-13)
rej={'baseline':1-keep}
for j,name in [(0,'Born_alpha_s_squared'),(1,'Born_PDF_luminosity'),(2,'Born_coupling_and_PDF')]:
 q=1-keep.copy();q[ix]=np.minimum(raw/a[:,j],1.);rej[name]=q
results=[]
for lo,hi in [(2,5),(5,10),(10,20),(5,20),(20,50),(0,None)]:
 selected=pt>=lo
 if hi is not None:selected&=pt<hi
 H=np.where(selected&h,-w*(1-keep),0.)
 for name,q in rej.items():
  S=np.where(selected&~h,w*q,0.)
  X,Y=S.sum(),H.sum();r=Y/X
  inf=(H-r*S)/X
  chunks=np.bincount(group,weights=inf,minlength=50)
  error=np.sqrt(50/49*np.dot(chunks,chunks))
  results.append(dict(lo=lo,hi=hi,control=name,S_pb=float(X),H_fixed_pb=float(Y),H_over_S=float(r),ratio_error=float(error),net_pb=float(Y-X),shift=float((Y-X)/w[selected].sum()),positive_S_pb=float(S[w>0].sum()),negative_S_pb=float(S[w<0].sum())))
np.savez_compressed(O/'S_born_scale_factors.npz',source_index=ix,values=a,fields=['coupling_squared','Born_PDF_luminosity','combined','alpha_Born','alpha_real_HT','alpha_real_over_alphaK_minus1'],raw_rejection=raw)
report=dict(events=len(w),active_S=len(ix),muBorn='Original canonical S Born H_T/2',muKernel_assumption='All final real endpoint H_T/2, validated independently with hard LHAPDF alpha_s',max_abs_alpha_real_over_alphaK_minus1=maxalpha,limitations='Expected first-emission control. Independent all24003-H replay H_bound_all.json proves the same G leaves all H bounds unchanged. Not a production change.',results=results)
(O/'S_scale_counterfactual.json').write_text(json.dumps(report,indent=2)+'\n')
for r in results:
 if r['lo']==5 and r['hi']==20:print(r)
