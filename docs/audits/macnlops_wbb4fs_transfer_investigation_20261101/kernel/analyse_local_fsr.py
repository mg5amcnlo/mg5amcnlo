from pathlib import Path
import json,numpy as np
from scipy.stats import kstest
D=Path(__file__).resolve().parent
points={r['index']:r for r in json.loads((D/'selected_profiles.json').read_text())}
a=np.genfromtxt(D/'local_fsr.csv.local.csv',delimiter=',',names=True)
cdf=np.full(len(a),np.nan);support=0;max_deviation=0
for index in np.unique(a['point']):
 for pdg,qi in [(5,0),(-5,1)]:
  mask=(a['point']==index)&(a['id']==pdg)
  if not mask.any():continue
  r=points[int(index)];m,mrec,M,cap=r['quarks'][qi];t=a['pt'][mask]**2;z=a['z'][mask];mr2=m*m;mk2=mrec*mrec;M2=M*M;A=M2+mr2-mk2;corrected=(M-mrec)**2-mr2
  disc=M2*M2*mr2*mr2+4*A*t*M2*(M2-mk2)
  smin=np.maximum(t/corrected,(M2*mr2-2*A*t+np.sqrt(np.maximum(0,disc)))/(2*A*A))
  smax=np.minimum(.25,t*(np.sqrt(t)/.5-1)/mr2)
  lo=2*smin/(1+np.sqrt(np.maximum(0,1-4*smin)))
  cut=2*smax/(1+np.sqrt(np.maximum(0,1-4*smax)))
  primitive=lambda z:-2*np.log1p(-z)-z-.5*z*z
  v0=primitive(lo);v1=primitive(cut);v2=primitive(1-cut);v3=primitive(1-lo)
  norm=v1-v0+v3-v2
  u=np.where(z<=cut,(primitive(z)-v0)/norm,(v1-v0+primitive(z)-v2)/norm)
  bad=((z<lo-1e-9)|(z>1-lo+1e-9)|((z>cut+1e-9)&(z<1-cut-1e-9)))
  support+=int(sum(bad));max_deviation=max(max_deviation,float(max(np.maximum(lo-z,z-(1-lo)))))
  cdf[mask]=u
assert np.all(np.isfinite(cdf))
result=dict(events=len(a),support_violations=support,cdf_range=[float(min(cdf)),float(max(cdf))],max_support_deviation=max_deviation,formula='Native z transformed by normalized integral of (1+z^2)/(1-z) over exact global-recoil massive-quark support and physical reduced-pT cutoff at each actual t; no survival factor or Born matrix element enters this conditional shape test.',groups=[])
for label,mask in [('all',np.ones(len(a),dtype=bool)),('b',a['id']==5),('bbar',a['id']==-5),('pt<5',a['pt']<5),('pt5-20',(a['pt']>=5)&(a['pt']<20)),('pt>=20',a['pt']>=20)]:
 u=cdf[mask];ks=kstest(u,'uniform');result['groups'].append(dict(label=label,events=int(sum(mask)),mean=float(np.mean(u)),mean_error=float(1/np.sqrt(12*len(u))),KS_statistic=float(ks.statistic),KS_p=float(ks.pvalue),decile_counts=np.histogram(u,np.linspace(0,1,11))[0].tolist()))
np.savez_compressed(D/'local_fsr_cdf.npz',point=a['point'],id=a['id'],pt=a['pt'],z=a['z'],cdf=cdf)
(D/'local_fsr_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
