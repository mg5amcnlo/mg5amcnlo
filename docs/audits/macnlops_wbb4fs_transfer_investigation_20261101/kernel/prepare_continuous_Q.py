from pathlib import Path
import json,importlib.util,xml.etree.ElementTree as ET
import numpy as np
D=Path(__file__).resolve().parent;B=D.parents[1]
points=json.loads((D/'selected_profiles.json').read_text());wanted={r['index'] for r in points}
spec=importlib.util.spec_from_file_location('audit',B/'validation/audit_matching_events.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
cdfs={}
for i,event in enumerate(a.blocks(D/'replay.lhe'),1):
 if i in wanted:cdfs[i]=float(ET.fromstring(''.join(event)).find('macnlops_decision').attrib['starting_cdf'])
gx,gw=np.polynomial.legendre.leggauss(12);report=[]
for p in points:
 q=np.array(p['Q']);Q=q[:,0];w=q[:,1];assert len(Q)%12==0
 width=2*(Q[-1]-Q[-2])/(gx[-1]-gx[-2]);hi=Q[-1]+.5*width*(1-gx[-1]);cdf=cdfs[p['index']]
 if cdf<1:
  odds=np.sqrt((1-cdf)/cdf);u=odds/(1+odds);lo=(p['pt']-u*hi)/(1-u)
 else:
  width=2*(Q[1]-Q[0])/(gx[1]-gx[0]);lo=Q[0]-.5*width*(1+gx[0])
 u=(Q-lo)/(hi-lo);raw=[]
 for k in range(0,len(Q),12):
  width=2*(Q[k+1]-Q[k])/(gx[1]-gx[0]);uk=u[k:k+12]
  raw.extend(gw*width/(hi-lo)*uk*(1-uk)/(uk*uk+(1-uk)**2)**2)
 raw=np.array(raw);error=np.max(np.abs(w/w.sum()-raw/raw.sum()));assert error<1e-9,error
 p['continuous_Q_bounds']=[float(lo),float(hi)];ut=max(0,min(1,(p['pt']-lo)/(hi-lo)));Ft=ut*ut/(ut*ut+(1-ut)**2)
 F=Ft+(np.arange(4096)+.5)/4096*(1-Ft);odds=np.sqrt(F/(1-F));us=odds/(1+odds);p['dense_Q']=lo+(hi-lo)*us
 report.append(dict(index=p['index'],lo=float(lo),hi=float(hi),source_cdf=cdf,max_normalized_weight_reconstruction_error=float(error),dense_quantile_points=4096))
def particle(p):
 e,x,y,z=p['p'];return ' '.join(map(str,[p['id'],p['status'],*p['mothers'],*p['colors'],x,y,z,e,p['mass'],0,9]))
for mode,key in [('ALL','P'),('ISR','P_ISR'),('FSR','P_FSR')]:
 lines=[str(len(points))]
 for p in points:
  lines.append(' '.join(map(str,[p['index'],len(p['born']),len(p['dense_Q']),p['pt'],p[key]])))
  lines.extend(particle(v) for v in p['born']);lines.extend(str(q)+' '+str(1/4096) for q in p['dense_Q'])
 (D/('continuous_points_'+mode+'.dat')).write_text('\n'.join(lines)+'\n')
(D/'continuous_Q_bounds.json').write_text(json.dumps(report,indent=2)+'\n')
print('Recovered all bounds; max normalized profile weight error',max(r['max_normalized_weight_reconstruction_error'] for r in report))
