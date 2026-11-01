#!/usr/bin/env python3
from pathlib import Path
import hashlib,itertools,json,re,sys,math,struct
R=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS');sys.path.insert(0,str(R))
from madgraph.various import macnlops as mc
B=Path('/export/slow1/rikkert/macnlops_born_scale_fix_20261101/generalisation');OUT=Path(__file__).parent
manifest=json.loads((B/'baseline_manifest.json').read_text())
def tag(block):return mc.xml_attributes(re.search(r'<macnlops_decision\b([^>]*?)/>',block)[1])
def core(block):
 block=re.sub(r'<macnlops_decision\b[^>]*?/>','',block)
 return re.sub(r"(<wgt\s+id=['\"]macnlops['\"]\s*>).*?(</wgt>)",r'\1MATCH_WEIGHT\2',block)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
results=[];maxima={};mismatches=[];ee=0;protected=None
for process in manifest['processes']:
 key=process['key']
 if key not in ['Wj','Wbb','EE2','EE3','EE4']:continue
 for sample in process['samples']:
  rel=Path(key)/sample['provider']/sample['channel']/'events_macnlops.lhe'
  oldpath,newpath=B/'baseline'/rel,B/'corrected'/rel
  if key=='Wbb' and sample['channel']!='GF1.0_26':continue
  limit=None if key=='Wj' else (5295 if key=='Wbb' else 12)
  reviewed=0
  with mc.LHEStream(oldpath) as old,mc.LHEStream(newpath) as new:
   for index,(a,b) in enumerate(itertools.zip_longest(old,new),1):
    if limit is not None and index>limit:break
    if key=='Wbb' and index not in [1,2,5293,5294,5295]:continue
    assert a is not None and b is not None
    x,y=tag(a),tag(b);reviewed+=1
    assert core(a)==core(b),(rel,index,'physical/source/continuation payload')
    if key.startswith('EE'):
     for field in ['keep','accepted','kernel_scale','r','d','k','marking','delta_source_mean','delta_shower_mean','starting_cdf']:assert x.get(field)==y.get(field),(rel,index,field)
     if 'born_scale_factor' in y:assert float(y['born_scale_factor'])==float(y['born_coupling_factor'])==float(y['born_pdf_factor'])==1.
     ee+=1
    if 'born_scale_factor' in y:
     G=float(y['born_scale_factor']);assert G==float(y['born_coupling_factor'])*float(y['born_pdf_factor'])
     if 'kernel_scale' in x:assert float(y['kernel_scale'])==float(x['kernel_scale'])*G
     if y['rule'] in ['s_veto','h_removal']:
      u=max(float(y['d'])-float(y['r']),0);K=float(y['k']);A=float(y['delta_source_mean']);P=float(y['delta_shower_mean']);F=K*float(y['kernel_scale'])*P
      T=min(min(u,K)*A,F)
      expected=1-T/(u*A if mc.Event(b).kind=='H' else F)
      assert abs(float(y['keep'])-expected)<=1e-13,(rel,index,'common transfer',expected,y['keep'])
    if key=='Wbb' and index==5294:
     assert x['rule']==y['rule']=='pdf_support_no_transfer' and x['keep']==y['keep']==x['accepted']==y['accepted']=='1'
     protected=dict(path=str(rel),event=index,before=x,after=y,physical_payload_identical=True)
    for field in ['r','d','k','marking','total_k']:
     if x.get(field)==y.get(field):continue
     av,bv=float(x[field]),float(y[field]);absolute=abs(av-bv);relative=absolute/max(abs(av),abs(bv),1e-300);ulp=absolute/max(math.ulp(av),math.ulp(bv))
     record=dict(path=str(rel),event=index,field=field,before=av,after=bv,absolute=absolute,relative=relative,ulps=ulp)
     if field not in maxima or relative>maxima[field]['relative']:maxima[field]=record
     mismatches.append(record)
  results.append(dict(path=str(rel),events_reviewed=reviewed,baseline_sha256=sha(oldpath),corrected_sha256=sha(newpath)))
report=dict(reviewed_events=sum(x['events_reviewed'] for x in results),fixed_scale_ee_reviewed=ee,protected_point=protected,Wj_maximum_numeric_drifts=maxima,Wj_numeric_differences=mismatches,physical_payload_and_factor_pairing_assertions_pass=True,raw_bindings=results)
(OUT/'raw_review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['raw_bindings','Wj_numeric_differences','protected_point']},indent=2))
