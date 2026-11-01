#!/usr/bin/env python3
"""Extract all negative H source packets without modifying production files."""
from pathlib import Path
import concurrent.futures,hashlib,json,re,shutil
import numpy as np
B=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101');A=B/'diagnosis_transfer_20261101/analysis';O=B/'diagnosis_transfer_20261101/scales/negative_H_inputs';P=B/'processes/Wbb_4FS/SubProcesses'
ATTR=re.compile(r'''\b([A-Za-z_][A-Za-z0-9_]*)=["']([^"']*)["']''')
EVENT=re.compile(r'<event>\s*\n(.*?)</event>',re.S);HARD=re.compile(r'<macnlops_hard_event[^>]*>(.*?)</macnlops_hard_event>',re.S);COMP=re.compile(r'<macnlops_component([^>]*)>(.*?)</macnlops_component>',re.S)
manifest=json.loads((A/'metadata_manifest.json').read_text());REG=manifest['registry_map'];z=np.load(A/'decision_metadata.npz');v=z['values'];kinds=z['kind'];weights=z['weight'];registries=z['registry'];incoming_ids=z['incoming_ids'];hard_momenta=z['hard_momenta'];F={s:i for i,s in enumerate(z['fields'])}
def key(reg,labels,incoming,mom):return np.asarray([reg,*labels,*incoming],dtype='<i4').tobytes()+np.asarray(mom,dtype='<f8').tobytes()
lookup={}
for i in np.flatnonzero((kinds==2)&(weights<0)):
 k=key(registries[i],[int(v[i,F[s]]) for s in ['component_sector','config','graph','flow']],incoming_ids[i,:2],hard_momenta[i])
 assert k not in lookup,('Ambiguous hard/component packet key',i,lookup.get(k));lookup[k]=int(i)
assert len(lookup)==24003,len(lookup)
def num(t):return float(t.replace('D','E').replace('d','e'))
def worker(path):
 text=path.read_text();first=text.index('<event>');header=text[:first];found=[];local=0
 for ev in EVENT.finditer(text):
  local+=1;body=ev[1];tokens=body.splitlines()[0].split();weight=num(tokens[2])
  if weight>=0 or not re.search(r'^#aMCatNLO\s+2\s',body,re.M):continue
  hard=HARD.search(body);comp=COMP.search(body);assert hard and comp
  reg=REG[dict(ATTR.findall(comp[1]))['registry']];rows=[line.split() for line in hard[1].splitlines() if line.strip()];nh=int(rows[0][0]);assert nh==6
  momenta=np.array([[num(row[j]) for j in range(6,11)] for row in rows[1:]]);incoming=[int(rows[j][0]) for j in [1,2]]
  labels=[int(t) for t in comp[2].splitlines()[1].split()] if not comp[2].splitlines()[0].strip() else [int(t) for t in comp[2].splitlines()[0].split()]
  assert len(labels)==4,labels
  k=key(reg,labels,incoming,momenta);assert k in lookup,(path,local,'Source packet absent in collected cache')
  idx=lookup[k];assert weight==weights[idx],(idx,weight,weights[idx])
  found.append((reg,labels[1],idx,local,ev[0],hashlib.sha256(k).hexdigest()))
 return str(path),header,found
paths=[]
for line in (P/'nevents_unweighted').read_text().splitlines():
 tokens=line.split()
 if tokens and int(tokens[1]):paths.append((P/tokens[0]).resolve())
assert len(paths)==101
O.mkdir(parents=True,exist_ok=True);groups={};seen=set()
with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:
 for n,(path,header,found) in enumerate(pool.map(worker,paths),1):
  for reg,config,idx,local,body,khash in found:
   assert idx not in seen,('Repeated source_index',idx);seen.add(idx)
   g=groups.setdefault((reg,config),dict(header=header,res_source=str(Path(path).parent/'res_1'),entries=[]))
   g['entries'].append(dict(source_index=idx,source_file=path,source_event_1based=local,packet_key_sha256=khash,event_text=body))
  if n%25==0:print('READ',n,'negativeH',len(seen),flush=True)
assert len(seen)==24003 and seen==set(lookup.values()),len(seen)
final=[]
for (reg,config),g in sorted(groups.items()):
 info=manifest['beta_groups'][f'{reg}:{config}'];name=info['name'].replace('/','_');folder=O/name;folder.mkdir(exist_ok=True)
 events=folder/'events.lhe'
 with events.open('w') as out:
  out.write(g['header'])
  for entry in g['entries']:out.write(entry.pop('event_text')+'\n')
  out.write('</LesHouchesEvents>\n')
 shutil.copy2(g['res_source'],folder/'res_1')
 np.savez(folder/'source_index.npz',source_index=np.array([x['source_index'] for x in g['entries']]),weight=weights[[x['source_index'] for x in g['entries']]])
 (folder/'mapping.json').write_text(json.dumps(g['entries'],indent=2)+'\n')
 final.append(dict(group=info['name'],registry=reg,config=config,beta=info['beta'],events=len(g['entries']),input=str(events),input_sha256=hashlib.sha256(events.read_bytes()).hexdigest(),res_original=g['res_source'],res_local=str(folder/'res_1'),source_index=str(folder/'source_index.npz'),mapping=str(folder/'mapping.json')))
report=dict(events=24003,groups=final,all_source_indices_unambiguous_unique=True,weight_and_original_hard_component_key_exact=True,production_inputs_untouched=True,header='Each group retains the complete original header of its first contributing source file.',source_index_sha256=hashlib.sha256((A/'decision_metadata.npz').read_bytes()).hexdigest())
(O/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
