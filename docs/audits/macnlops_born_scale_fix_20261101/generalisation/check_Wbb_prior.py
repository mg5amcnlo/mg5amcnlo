from pathlib import Path
import json,re,sys,numpy as np
ROOT=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS');sys.path.insert(0,str(ROOT));from madgraph.various import macnlops
b=Path(__file__).resolve().parent;wbb=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101/diagnosis_transfer_20261101');d=np.load(wbb/'analysis/decision_metadata.npz');g=np.load(wbb/'scales/S_born_scale_factors.npz');fields=list(d['fields']);v=d['values'];gv=g['values'];gi=g['source_index'];keys=['r','d','k','pt','sector','config'];columns=[fields.index(k) for k in keys];lookup={}
for p in (b/'corrected/Wbb').glob('P*/GF*/events_macnlops.lhe'):
 with macnlops.LHEStream(p) as f:
  for block in f:
   a=macnlops.xml_attributes(re.search(r'<macnlops_decision\b([^>]*?)/>',block)[1])
   if a.get('rule')!='s_veto' or 'born_scale_factor' not in a:continue
   config=float(re.search(r'<macnlops_component\b[^>]*>\s*([^\n]+)',block)[1].split()[1]);key=tuple([float(a[k]) for k in ['r','d','k','pt','sector']]+[config]);value=[float(a[k]) for k in ['born_coupling_factor','born_pdf_factor','born_scale_factor']]
   assert key not in lookup or lookup[key]==value;lookup[key]=value
errors=[];count=0
for j,index in enumerate(gi):
 key=tuple(v[index,columns])
 if key not in lookup:continue
 errors.append(abs(np.array(lookup.pop(key))/gv[j,:3]-1));count+=1
result=dict(compared_S_points=count,unmatched_unique_raw_points=len(lookup),max_relative_difference_per_factor=np.max(errors,axis=0).tolist(),factor_order=['coupling','Born_PDF','combined'],passed=bool(count>5000 and len(lookup)==0 and np.max(errors)<1e-9),
 reference='Original independentWbb diagnosis computedBorn alpha_s² andhardPDF ratios atnativeBornHT/2 versusrealHT/2; currentG usesnative selected-flow Born getter.',reference_files=[str(wbb/'scales/S_born_scale_factors.npz'),str(wbb/'analysis/decision_metadata.npz')])
(b/'Wbb_prior_factor_comparison.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(result)
