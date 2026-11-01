from pathlib import Path
import sys,importlib.util,json,math
import numpy as np
import xml.etree.ElementTree as ET
D=Path(__file__).resolve().parent;B=D.parents[1]
spec=importlib.util.spec_from_file_location('audit',B/'validation/audit_matching_events.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def vec(p):return np.array([p['e'],p['px'],p['py'],p['pz']])
def dot(p,q):return p[0]*q[0]-np.dot(p[1:],q[1:])
def boost(v,b):
 b2=np.dot(b,b)
 if b2==0:return v
 gam=1/math.sqrt(1-b2);bp=np.dot(b,v[1:]);return np.r_[gam*(v[0]-bp),v[1:]+(((gam-1)*bp/b2)-gam*v[0])*b]
def coordinates(xi,y,M,m,mrec,k0):
 q=M*xi/2;aa=M-q;bb=q*y;cc=(M*M*(1-xi)+m*m-mrec*mrec)/2;den=aa*aa-bb*bb
 disc=cc*cc-den*m*m
 if disc<0:raise ValueError('disc')
 ks=[(-bb*cc+sgn*aa*math.sqrt(disc))/den for sgn in [1,-1]]
 ks=[k for k in ks if k>0 and abs(aa*math.sqrt(k*k+m*m)+bb*k-cc)<1e-6*max(M*M,1)]
 if not ks:raise ValueError('no branch')
 k=min(ks,key=lambda k:abs(k-k0));E=math.sqrt(k*k+m*m);eminus=m*m/(E+k);gap=eminus+(1-y)*k
 z=(eminus*eminus+2*E*k*(1-y))/(2*gap*(E+q));omz=(m*m/(2*gap)+q)/(E+q)
 return np.array([z,z*omz*M*xi*gap])
rows=[]
for index,event in enumerate(a.blocks(D/'replay.lhe'),1):
 xml=ET.fromstring(''.join(event));ds=xml.find('macnlops_decision').attrib
 if ds.get('sector') not in ('3','4') or ds['rule'] not in ('s_veto','no_transfer'):continue
 kind=int(next(s.split()[1] for s in event if s.startswith('#aMCatNLO')))
 if kind!=1:continue
 n=int(event[1].split()[0]);ps=[a.particle(s) for s in event[2:2+n]]
 total=vec(ps[0])+vec(ps[1]);beta=total[1:]/total[0];M=math.sqrt(dot(total,total))
 rid=5 if ds['sector']=='3' else -5;rad=next(p for p in ps if p['status']==1 and p['id']==rid);emt=next(p for p in ps if p['status']==1 and p['id']==21)
 pr=boost(vec(rad),beta);pg=boost(vec(emt),beta);rec=np.array([M,0.,0.,0.])-pr-pg;mrec=math.sqrt(max(0,dot(rec,rec)));m=rad['mass'];k=np.linalg.norm(pr[1:]);E=pr[0];xi=2*pg[0]/M;y=np.dot(pr[1:],pg[1:])/(k*pg[0])
 if abs(y)>=1 or xi<=0:continue
 c=coordinates(xi,y,M,m,mrec,k);hxi=min(xi,1-xi)*2e-5;hy=min(1-y,1+y)*2e-5
 if min(hxi,hy)<1e-12:continue
 try:
  dx=(coordinates(xi+hxi,y,M,m,mrec,k)-coordinates(xi-hxi,y,M,m,mrec,k))/(2*hxi)
  dy=(coordinates(xi,y+hy,M,m,mrec,k)-coordinates(xi,y-hy,M,m,mrec,k))/(2*hy)
 except ValueError:continue
 numeric=abs(np.linalg.det(np.c_[dx,dy]));geometry=(2-xi)*k+xi*y*E;z=c[0];analytic=abs(2*M*k*k/geometry*z*(1-z)**2)
 t=float(ds['pt'])**2
 pair=pr+pg;nativeZ=1-pg[0]/(pair[0]*(1-m*m/dot(pair,pair)))
 row=dict(index=index,id=rid,xi=xi,y=y,pt=math.sqrt(t),z=z,native_z=nativeZ,z_error=abs(z-nativeZ),t_relative_error=abs(c[1]/t-1),jacobian_numeric=numeric,jacobian_analytic=analytic,relative_error=abs(analytic/numeric-1),second_branch=bool(geometry<0))
 rows.append(row)
report=dict(points=len(rows),bottom=sum(r['id']==5 for r in rows),antibottom=sum(r['id']==-5 for r in rows),second_branch=sum(r['second_branch'] for r in rows),max_z_error=max(r['z_error'] for r in rows),max_t_relative_error=max(r['t_relative_error'] for r in rows),max_jacobian_relative_error=max(r['relative_error'] for r in rows),median_jacobian_relative_error=float(np.median([r['relative_error'] for r in rows])),rows=rows)
(D/'fsr_jacobian.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
