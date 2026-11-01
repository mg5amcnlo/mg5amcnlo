from pathlib import Path
import json,numpy as np
OUT=Path(__file__).resolve().parent;DIAG=OUT.parent
D=np.load(DIAG/'analysis/decision_metadata.npz');G=np.load(DIAG/'scales/S_born_scale_factors.npz');f=list(D['fields']);v=D['values'];ai=G['source_index'];beta=D['beta'][ai];fold=np.random.default_rng(4930301).integers(0,2,len(v));raw=G['raw_rejection'];scale=G['values'][:,list(G['fields']).index('combined')]
for filename,names in [('crossfit_extended.json',[]),('crossfit_flavour_geometry.json',['mWbb_HT2_8x8'])]:
 report=json.loads((OUT/filename).read_text())
 for name in names:
  model=report['models'][name];codes=np.zeros(len(ai),dtype=int);cells=1
  for field in model['fields']:
   edges=np.array(model['quantile_internal_edges'][field]);codes+=cells*np.searchsorted(edges,v[ai,f.index(field)],side='right');cells*=len(edges)+1
  if name.startswith('flavour'):
   cat=(np.max(abs(D['incoming_ids'][ai,:2]),axis=1)==4).astype(int)
   if name.startswith('flavour_order'):cat=2*cat+(D['incoming_ids'][ai,0]<0)
   codes+=cells*cat
  factor=np.ones(len(ai));fallback=np.ones(len(ai),dtype=bool)
  for detail in model['fit_details']:
   held=fold[ai]==detail['heldout_fold'];rows=detail['bins'];local=np.array([row['beta_local'] if row['beta_local'] is not None else np.nan for row in rows]);valid=np.array([row['valid'] for row in rows]);good=held&valid[codes];factor[good]=local[codes[good]]/beta[good];fallback[good]=False
  rejection=np.minimum(1,raw/scale/factor)
  p=OUT/f'predictions_{name}.npz';np.savez_compressed(p,source_index=ai,local_factor=factor,rejection=rejection,fallback=fallback,heldout_fold=fold[ai],model=name,model_report=filename)
  assert fallback.sum()==model['fallback_active_S_events'] and (rejection>=1).sum()==model['saturated_S_events']
  print(name,p,flush=True)
