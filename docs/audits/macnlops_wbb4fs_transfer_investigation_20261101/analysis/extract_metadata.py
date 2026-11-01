#!/usr/bin/env python3
"""Read frozen ordered LHE splits once for transfer and Born-scale diagnosis."""
from pathlib import Path
import concurrent.futures,hashlib,json,re,time
import numpy as np
B=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101')
OUT=B/'diagnosis_transfer_20261101/analysis'
ATTR=re.compile(r'''\b([A-Za-z_][A-Za-z0-9_]*)=["']([^"']*)["']''')
FIELDS=['keep','accepted','pt','matching_scale','r','d','k','sector','candidates','marking','total_k','kernel_scale','alpha_s_shower','alpha_s_kernel','pdf_ratio','delta_source_mean','delta_shower_mean','starting_cdf']
EXTRA=['component_sector','config','graph','flow','flow_probability','insertion','source_factor','symmetry','source_scalup','hard_ht_half','endpoint_ht_half','hard_x1','hard_x2','endpoint_x1','endpoint_x2','hard_system_mass','hard_system_y','hard_bb_mass','hard_bb_pt','hard_b_pt','hard_bbar_pt','hard_w_pt','hard_bb_delta_phi','hard_bb_delta_r']
REG={}
BETAS={}
for p in (B/'processes/Wbb_4FS/SubProcesses').glob('P*/macnlops_routes.dat'):
 registry=p.read_text().splitlines()[0].split()[1];REG[registry]=len(REG)
 for config in (1,2):
  f=next(p.parent.glob(f'GF{config}.0_*/events_macnlops.json'));d=json.loads(f.read_text());BETAS[f'{REG[registry]}:{config}']=dict(name=p.parent.name+f'/GF{config}.0',**d['channel'])
REGNAMES={v:k for k,v in REG.items()}
def num(s):return float(s.replace('D','E').replace('d','e'))
def particles(lines):
 arr=np.zeros((len(lines),7))
 for i,line in enumerate(lines):
  t=line.split();arr[i,:2]=int(t[0]),int(t[1]);arr[i,2:]=[num(t[j]) for j in range(6,11)]
 return arr
# particle array columns: id,status,px,py,pz,E,mass

def mass(q):return np.sqrt(max(q[3]**2-np.dot(q[:3],q[:3]),0.))
def rapidity(q):return .5*np.log((q[3]+q[2])/(q[3]-q[2])) if q[3]>abs(q[2]) else np.nan
def kin(a):
 final=a[a[:,1]==1];ht=.5*np.sqrt(final[:,6]**2+np.sum(final[:,2:4]**2,axis=1)).sum()
 incoming=a[a[:,1]==-1];x=[(incoming[0,5]+incoming[0,4])/13000.,(incoming[1,5]-incoming[1,4])/13000.]
 return ht,x

def worker(job):
 t0=time.time();path=B/f'results/full_1M_replay/input.lhe.gz_{job-1}.lhe'
 metadata=[];ids=[];hard_values=[];registries=[];rules=[];weights=[];kinds=[]
 with path.open() as f:
  for line in f:
   if line.strip()!='<event>':continue
   header=next(f).split();n=int(header[0]);weights.append(num(header[2]))
   endpoint=particles([next(f) for _ in range(n)])
   decision=component=continuation=None;hard=[];comp=[];state='';kind=None;registry=None
   for line in f:
    if '<macnlops_hard_event' in line:state='hard';continue
    if '</macnlops_hard_event>' in line:state='';continue
    if '<macnlops_component' in line:registry=REG[dict(ATTR.findall(line))['registry']];state='component';continue
    if '</macnlops_component>' in line:state='';continue
    if state=='hard':hard.append(line);continue
    if state=='component':comp.append(line);continue
    if '<macnlops_decision' in line:decision=dict(ATTR.findall(line))
    if '<macnlops_continuation' in line:continuation=dict(ATTR.findall(line))
    if line.startswith('#aMCatNLO'):kind=int(line.split()[1])
    if '</event>' in line:break
   assert kind in (1,2) and decision and continuation and registry is not None
   rows=[l for l in hard if l.strip()];nh=int(rows[0].split()[0]);assert len(rows)==nh+1
   hard=particles(rows[1:]);comp=[num(x) for line in comp for x in line.split()];assert len(comp)==8
   ht,x=kin(hard);eht,ex=kin(endpoint)
   b=hard[hard[:,0]==5][0,2:6];bb=hard[hard[:,0]==-5][0,2:6];w=hard[hard[:,0]==24][0,2:6];system=b+bb+w;pair=b+bb
   dphi=abs(np.arctan2(np.sin(np.arctan2(b[1],b[0])-np.arctan2(bb[1],bb[0])),np.cos(np.arctan2(b[1],b[0])-np.arctan2(bb[1],bb[0]))))
   extra=comp+[num(continuation['source_scalup']),ht,eht,*x,*ex,mass(system),rapidity(system),mass(pair),np.hypot(*pair[:2]),np.hypot(*b[:2]),np.hypot(*bb[:2]),np.hypot(*w[:2]),dphi,np.hypot(dphi,rapidity(b)-rapidity(bb))]
   values=[num(decision[k]) if k in decision else np.nan for k in FIELDS]+extra
   assert len(values)==len(FIELDS)+len(EXTRA)
   metadata.append(values);registries.append(registry);rules.append(decision['rule']);kinds.append(kind)
   incoming=hard[hard[:,1]==-1];endpoint_incoming=endpoint[endpoint[:,1]==-1];ids.append([*incoming[:,0],*endpoint_incoming[:,0]])
   hv=np.zeros((6,5));hv[:nh]=hard[:,2:];hard_values.append(hv)
 out=OUT/f'metadata_part_{job}.npz'
 np.savez(out,values=np.asarray(metadata),incoming_ids=np.asarray(ids,dtype=np.int16),hard_momenta=np.asarray(hard_values),registry=np.asarray(registries,dtype=np.int8),rule=np.asarray(rules),weight=np.asarray(weights),kind=np.asarray(kinds,dtype=np.int8))
 return job,len(metadata),round(time.time()-t0,2)

def main():
 start=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:
  for job,n,sec in pool.map(worker,range(1,51)):
   assert n==20000,(job,n)
   if job%10==0:print('PARSED',job,sec,'elapsed',time.time()-start,flush=True)
 arrays={};names=None
 for job in range(1,51):
  z=np.load(OUT/f'metadata_part_{job}.npz')
  for k in z.files:arrays.setdefault(k,[]).append(z[k])
 arrays={k:np.concatenate(v) for k,v in arrays.items()};first=np.load(B/'results/run_1M_wbb4fs_nocuts_lhe.npz')
 assert np.array_equal(arrays['weight'],first['nominal']);assert np.array_equal(arrays['kind']==2,first['is_h'])
 assert np.array_equal(arrays['values'][:,FIELDS.index('keep')],first['keep'])
 # Production-channel mapping is verified against the recorded beta factor.
 v=arrays['values'];present=np.isfinite(v[:,FIELDS.index('kernel_scale')])&(v[:,FIELDS.index('alpha_s_shower')]>0)
 beta=np.array([BETAS[f'{r}:{int(c)}']['beta'] for r,c in zip(arrays['registry'],v[:,len(FIELDS)+EXTRA.index('config')])])
 inferred=v[:,FIELDS.index('kernel_scale')]/(v[:,FIELDS.index('alpha_s_shower')]/v[:,FIELDS.index('alpha_s_kernel')]*v[:,FIELDS.index('pdf_ratio')])
 assert np.allclose(beta[present],inferred[present],rtol=2e-14,atol=0),(np.max(abs(beta[present]-inferred[present])))
 arrays.update(fields=np.asarray(FIELDS+EXTRA),registry_names=np.asarray([REGNAMES[i] for i in range(len(REGNAMES))]),beta=beta,source_index=np.arange(1000000))
 final=OUT/'decision_metadata.npz';np.savez_compressed(final,**arrays)
 manifest=dict(events=1000000,fields=FIELDS+EXTRA,registry_map=REG,beta_groups=BETAS,beta_inference_max_abs=float(np.max(abs(beta[present]-inferred[present]))),exact_first_event_order_weight_keep_verified=True,hard_momenta_columns=['px','py','pz','E','mass'],hard_momenta_note='First nhard rows. nhard=5 for S Born and6 for H real; row identities ordered as source hard packet: incoming1,incoming2,b,bbar,W[,extra].',endpoint_ht_note='H_T/2 of all stable final endpoint particles using their recorded masses.',seconds=time.time()-start,sha256=hashlib.sha256(final.read_bytes()).hexdigest())
 (OUT/'metadata_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('DONE',time.time()-start,manifest['sha256'],flush=True)
if __name__=='__main__':main()
