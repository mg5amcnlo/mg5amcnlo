#!/usr/bin/env python3
"""Describe matched DY residuals without changing showers or native physics."""
import argparse,ctypes as C,csv,hashlib,json
from pathlib import Path
import numpy as np
NAMES=['default','PDFmode0','PDFmode0_root','PDFmode2','PDFmode2_root','PDFmode1','PDFmode1_root']
ABS=[2,3,5,7,10,15,20,30,40,50,75,100,125,150]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('pythia',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 groups=[];details=[];files=[Path(__file__)]
 for pdf in [13,15]:
  study=a.study/f'pdf{pdf}';adir=study/('analysis_pdf13' if pdf==13 else 'analysis')
  bridge=adir/'pythia_pdf.so';lib=C.CDLL(str(bridge));lib.pythia_pdf_init.argtypes=[C.c_char_p,C.c_char_p];lib.pythia_pdf_init.restype=C.c_int;lib.pythia_pdf_xf.argtypes=[C.c_int,C.c_double,C.c_double];lib.pythia_pdf_xf.restype=C.c_double
  assert lib.pythia_pdf_init(str(a.pythia/'share/Pythia8/xmldoc').encode(),str(pdf).encode())==0
  tl=C.CDLL(str(adir/'PDFmode1/delta.so'));f=tl.pysudakov_;f.argtypes=[C.POINTER(C.c_double),C.POINTER(C.c_double),C.POINTER(C.c_int),C.POINTER(C.c_int),C.POINTER(C.c_double)];f.restype=C.c_double
  mass=(C.c_double*21)(.33,.33,.5,1.5,4.8,*([0.]*16))
  def table(q,m,pdg):
   qq,mm,ii,tt=C.c_double(q),C.c_double(m),C.c_int(pdg),C.c_int(1)
   return f(C.byref(qq),C.byref(mm),C.byref(ii),C.byref(tt),mass)
  for name in ['dy_fixed','dy_damped']:
   bp=study/(name+'.born.jsonl');zp=adir/(name+'.npz');files.extend([bp,zp,bridge,adir/'PDFmode1/delta.so'])
   born=[json.loads(l) for l in bp.open()]
   with np.load(zp) as z:arrays={k:z[k] for k in z.files}
   xs=np.array([b['x'] for b in born]);fl=abs(arrays['ids']);y=abs(arrays['rapidity']);qstart=arrays['scalup'];n=len(born)
   masks=[('all',np.ones(n,bool)),('light_ud',fl<=2),('sea_scb',fl>=3),('central_light_ud',(y<1)&(fl<=2)),('forward_light_ud',(y>=2)&(fl<=2)),('unclipped_mode1_q20',arrays['values'][:,6,5,3]==0)]
   masks +=[(f'flavour_{j}',fl==j) for j in range(1,6)]
   masks +=[(f'xmax_{lo:g}_{hi:g}',(xs.max(axis=1)>=lo)&(xs.max(axis=1)<hi)) for lo,hi in [(0,.02),(.02,.05),(.05,.1),(.1,.2),(.2,1)]]
   masks +=[(f'abs_y_{lo:g}_{hi:g}',(y>=lo)&(y<hi)) for lo,hi in [(0,1),(1,2),(2,10)]]
   for cut in [5,10,20,30,50]:
    j=ABS.index(cut)
    for group,mask in masks:
     ids=np.flatnonzero(mask)
     if len(ids)<10:continue
     boots=np.random.default_rng(102828).integers(0,len(ids),size=(500,len(ids)))
     ref=arrays['reference'][mask,j];d=arrays['values'][mask,j,5,0];den=ref[boots].mean(axis=1);br=np.divide(d[boots].mean(axis=1),den,out=np.full(500,np.nan),where=den>0)
     groups.append(dict(pdf=pdf,sample=name,cut=cut,group=group,n=len(ids),xmax_mean=float(xs[mask].max(axis=1).mean()),q_over_start_mean=float((cut/qstart[mask]).mean()),delta=float(d.mean()),shower=float(ref.mean()),ratio=float(d.mean()/ref.mean()),ratio_se=(float(br.std(ddof=1)) if np.isfinite(br).all() else None),clipped_legs=int(arrays['values'][mask,j,5,3].sum()),threshold_qualified=bool(pdf==15 and cut<=np.sqrt(2)*4.8),effective_extra_log_coefficient=(float(-np.log(d.mean()/ref.mean())/np.log(qstart[mask]/cut).mean()) if name=='dy_fixed' and np.all(qstart[mask]>cut) else None)))
   for cut in [10,20,50]:
    j=ABS.index(cut);leg=[];Fs=[];Rs=[]
    for b in born:
     if cut>=b['scalup']:
      leg.append([1.,1.]);Fs.append([1.,1.]);Rs.append([1.,1.]);continue
     M=np.sqrt(4*b['p'][0][0]*b['p'][1][0]);r=[table(cut,M,i)/table(b['scalup'],M,i) for i in b['ids'][:2]];ff=[lib.pythia_pdf_xf(i,x,cut)/lib.pythia_pdf_xf(i,x,b['scalup']) for i,x in zip(b['ids'][:2],b['x'])]
     Fs.append(ff);Rs.append(r);leg.append(np.array(ff)*r)
    leg=np.array(leg);Fs=np.array(Fs);Rs=np.array(Rs);direct=np.clip(leg,0,1).prod(axis=1);raw=leg.prod(axis=1)
    assert np.max(abs(direct-arrays['values'][:,j,5,0]))<1e-10
    for group,mask in masks:
     if sum(mask)<10:continue
     details.append(dict(pdf=pdf,sample=name,cut=cut,group=group,n=int(sum(mask)),mean_PDF_product=float(Fs[mask].prod(axis=1).mean()),mean_table_product=float(Rs[mask].prod(axis=1).mean()),mean_unclipped_delta=float(raw[mask].mean()),mean_clipped_delta=float(direct[mask].mean()),mean_shower=float(arrays['reference'][mask,j].mean()),ratio_without_clipping=float(raw[mask].mean()/arrays['reference'][mask,j].mean()),ratio_with_clipping=float(direct[mask].mean()/arrays['reference'][mask,j].mean()),clipped_legs=int(((leg[mask]<0)|(leg[mask]>1)).sum()),PDF_leg_percentiles=np.percentile(Fs[mask],[0,10,50,90,100]).tolist(),table_leg_percentiles=np.percentile(Rs[mask],[0,10,50,90,100]).tolist()))
 for filename,rows in [('groups.csv',groups),('decomposition.csv',details)]:
  with (a.output/filename).open('w') as o:
   w=csv.DictWriter(o,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 result=dict(groups=groups,decomposition=details,input_sha256={str(p):sha(p) for p in dict.fromkeys(files)},definition='Conditional equal-Born DY references; matched historical settings; native mode1 factors independently reconstructed as product clip(F_i R_i). Products of means are not substituted for means of products.',limitation='These patterns locate residuals and clipping effects; no per-radiator shower probability is measured. Effective per-leg comparisons would be symmetric diagnostics, not individual measured leg CDFs.')
 (a.output/'patterns.json').write_text(json.dumps(result,indent=2)+'\n')
 for r in groups:
  if r['pdf']==13 and r['sample']=='dy_fixed' and r['cut']==20:print(r)
 for r in details:
  if r['pdf']==13 and r['sample']=='dy_fixed' and r['cut']==20 and r['group'] in ['all','light_ud','central_light_ud','forward_light_ud']:print(r)
if __name__=='__main__':main()
