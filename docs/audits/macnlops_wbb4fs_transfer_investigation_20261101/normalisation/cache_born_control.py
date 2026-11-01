from pathlib import Path
import gzip,json,hashlib,numpy as np
OUT=Path(__file__).resolve().parent;SOURCE=OUT/'Wbb_Born_LO/Events/run_born_control/unweighted_events.lhe.gz'
fields=['hard_ht_half','hard_x1','hard_x2','hard_system_mass','hard_system_y','hard_bb_mass','hard_bb_pt','hard_b_pt','hard_bbar_pt','hard_w_pt','hard_bb_delta_phi','hard_bb_delta_r']
rows=[];weights=[];incoming=[];alphas=[];scales=[];init=[]
m2=lambda p:p[3]*p[3]-np.dot(p[:3],p[:3])
pt=lambda p:np.hypot(p[0],p[1])
y=lambda p:.5*np.log((p[3]+p[2])/(p[3]-p[2]))
with gzip.open(SOURCE,'rt') as f:
 for line in f:
  if line.strip()=='<init>':
   header=next(f).split();init=[next(f).strip() for _ in range(int(header[-1]))]
  if line.strip()!='<event>':continue
  info=next(f).split();n=int(info[0]);assert n==5
  legs=[]
  while len(legs)<n:
   row=next(f).split()
   if row:legs.append(row)
  ids=np.array([int(q[0]) for q in legs]);status=np.array([int(q[1]) for q in legs]);mom=np.array([[float(q) for q in row[6:11]] for row in legs])
  assert np.array_equal(status,[-1,-1,1,1,1]);assert 5 not in abs(ids[:2]);assert np.all(mom[:2,4]==0)
  b=mom[ids==5][0];bb=mom[ids==-5][0];w=mom[ids==24][0]
  assert b[4]==bb[4]==4.7
  whole=b+bb+w;pair=b+bb
  ht=.5*sum(np.sqrt(mom[i,4]**2+pt(mom[i])**2) for i in [2,3,4])
  dphi=abs(np.arctan2(np.sin(np.arctan2(b[1],b[0])-np.arctan2(bb[1],bb[0])),np.cos(np.arctan2(b[1],b[0])-np.arctan2(bb[1],bb[0]))))
  rows.append([ht,(mom[0,3]+mom[0,2])/13000.,(mom[1,3]-mom[1,2])/13000.,np.sqrt(max(m2(whole),0.)),y(whole),np.sqrt(max(m2(pair),0.)),pt(pair),pt(b),pt(bb),pt(w),dphi,np.hypot(y(b)-y(bb),dphi)])
  weights.append(float(info[2]));incoming.append(ids[:2]);scales.append(float(info[3]));alphas.append(float(info[5]))
v=np.array(rows);weights=np.array(weights);assert len(v)==200000 and np.all(weights>0)
np.savez_compressed(OUT/'born_control.npz',values=v,fields=fields,weight=weights,incoming_ids=np.array(incoming),scale=np.array(scales),alpha_s=np.array(alphas))
result=dict(events=len(v),positive_weights=True,weight_sum_pb=float(weights.sum()),init_rows=init,
   scale_HT2_max_relative_difference=float(np.max(abs(np.array(scales)-v[:,0])/v[:,0])),
   minimum_observed_HT2_GeV=float(v[:,0].min()),minimum_observed_b_pt_GeV=float(v[:,7].min()),
   source=dict(path=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest()),
   fields=fields,beam_bottom_events=0,model_bottom_mass_GeV=4.7,
   normalization_note='LO unweighted conditional-sampling errors do not include the independently estimated hard integration normalization uncertainty.')
(OUT/'born_control_metadata.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2))
