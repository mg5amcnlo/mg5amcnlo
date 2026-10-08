from pathlib import Path
import gzip,hashlib,json,math,random,re,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from madgraph.various import ampli_pool
WORK=Path(__file__).resolve().parent
OUT=WORK/'drell_yan'
RUN='adaptive6000'
manifest=json.loads((OUT/'Events'/RUN/'ampli_production.json').read_text())
assert manifest['sampling']=='adaptive_unfolded'
assert manifest['weight_convention']=='draw_time_importance'
assert manifest['rate_source']=='survey'
assert manifest['requested_events']==6000
assert sum(c['quota'] for c in manifest['channels'])==6000
workers=[]
expected_weights=[]
expected_tail=[]
rng=random.Random(manifest['seed'] ^ 0x504F4F4C)
for c in manifest['channels']:
 channel_pools=[]
 assert c['error_abs'] <= .03*c['absolute']
 assert sum(b['final_quota'] for b in c['batches'])==c['quota']
 assert c['selection']['selected_tail']<.01
 for b in c['batches']:
  pool=ampli_pool.read_pool(OUT/b['directory'])
  channel_pools.append(pool)
  assert pool['version']==3
  assert pool['adaptation']==b['adaptation']
  assert pool['trials']==b['trials']
  assert pool['generated_target']==(11*pool['final_quota']+9)//10
  mask=pool['adaptation']['mask']
  assert mask[-3:]==[0,0,0] and all(mask[:-3])
  assert max(pool[k] for k in ('full_trial_tail','reserve_tail','worst_subset_tail'))<.01
  workers.append(dict(directory=b['directory'],trials=pool['trials'],quota=pool['final_quota'],reserve=pool['generated_target'],adaptation=pool['adaptation'],tail={k:pool[k] for k in ('full_trial_tail','reserve_tail','worst_subset_tail')}))
 if c['quota']:
  selection,diag=ampli_pool.select_candidates(channel_pools,c['quota'],rng)
  assert math.isclose(diag['selected_tail'],c['selection']['selected_tail'],rel_tol=1e-12,abs_tol=1e-15)
  for (ip,ic),correction in selection.items():
   row=channel_pools[ip]['candidates'][ic]
   expected_weights.append(correction*row[4]*manifest['absolute_cross_section']/6000)
   expected_tail.append(bool(row[3]))
assert any(w['adaptation']['updates']>=2 for w in workers)
assert sum(w['trials'] for w in workers)==manifest['generation_trials']
with gzip.open(OUT/'Events'/RUN/'events.lhe.gz','rt') as stream:
 text=stream.read()
events=re.findall(r'<event(?:[ \t][^>]*)?>(.*?)</event>',text,re.S)
assert len(events)==6000
weights=[float(e.strip().splitlines()[0].split()[2]) for e in events]
assert all(math.isfinite(w) and w!=0 for w in weights)
assert any(w<0 for w in weights)
assert all(math.isclose(a,b,rel_tol=5e-7) for a,b in zip(sorted(map(abs,weights)),sorted(expected_weights)))
collected_tail=sum(w for w,t in zip(expected_weights,expected_tail) if t)/sum(expected_weights)
assert collected_tail<.01
init=re.search(r'<init>(.*?)</init>',text,re.S).group(1).strip().splitlines()
assert int(init[0].split()[8])==-4
assert math.isclose(sum(float(line.split()[0]) for line in init[1:]),manifest['cross_section'],rel_tol=5e-8)
started=json.loads((WORK/'started.json').read_text())
assert all(hashlib.sha256((WORK/'sources'/name).read_bytes()).hexdigest()==digest for name,digest in started['source_sha256'].items()), 'Source snapshot differs from the tested export'
saved=json.loads((WORK/'survey_grids_before_generation.json').read_text())
assert len(saved)==16
assert all(hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest for name,digest in saved.items())
result=dict(collected_tail_fraction=collected_tail,survey_grids_unchanged=True,events=len(events),negative_events=sum(w<0 for w in weights),grid_updates=sum(w['adaptation']['updates'] for w in workers),workers=workers,signed_rate_pb=manifest['cross_section'],error_pb=manifest['uncertainty'],absolute_rate_pb=manifest['absolute_cross_section'],generation_trials=manifest['generation_trials'],generation_cpu_seconds=manifest['generation_cpu_seconds'],all_worker_tail_fractions_below_one_percent=True)
(WORK/'validation.json').write_text(json.dumps(result,indent=2)+chr(10))
print(json.dumps(result,indent=2))
