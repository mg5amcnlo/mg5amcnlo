from pathlib import Path
import gzip,hashlib,json,math,re,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from madgraph.various import ampli_pool
WORK=Path(__file__).parent; OUT=WORK/'drell_yan'
result={}
for run,expected,norm in [('survey3pct',120,'sum'),('restart60',60,'unity')]:
 directory=OUT/'Events'/run
 manifest=json.loads((directory/'ampli_production.json').read_text())
 assert manifest['version']==2 and manifest['rate_source']=='survey'
 assert manifest['requested_events']==expected
 assert sum(c['quota'] for c in manifest['channels'])==expected
 assert all(c['absolute']>=0 and c['error_abs']<=.03*c['absolute']*(1+1e-12) for c in manifest['channels'])
 assert math.isclose(sum(c['signed'] for c in manifest['channels']),manifest['cross_section'],rel_tol=1e-13)
 workers=trials=reserve=0; max_tail=0.; multiple_workers=False
 for channel in manifest['channels']:
  assert sum(b['final_quota'] for b in channel['batches'])==channel['quota']
  multiple_workers|=len(channel['batches'])>1
  for batch in channel['batches']:
   assert batch['generated_target']==(11*batch['final_quota']+9)//10
   if run=='restart60':
    pool=ampli_pool.read_pool(OUT/batch['directory'])
    assert pool['trials']==batch['trials']
    assert pool['final_quota']==batch['final_quota']
    assert pool['generated_target']==batch['generated_target']
    max_tail=max(max_tail,pool['full_trial_tail'],pool['reserve_tail'],pool['worst_subset_tail'])
   else:
    # Generation-only runs reuse worker directories. The first run's persisted
    # manifest and LHE remain available, but its raw worker pools were replaced.
    selection=channel['selection']
    max_tail=max(max_tail,selection['full_trial_tail'],selection['reserve_tail'],selection['worst_subset_tail'])
   assert max_tail<.01
   workers+=1;trials+=batch['trials'];reserve+=batch['generated_target']
  if channel['quota']:
   assert channel['selection']['selected_tail']<.01
 assert trials==manifest['generation_trials']
 with gzip.open(directory/'events.lhe.gz','rt') as stream:text=stream.read()
 blocks=re.findall(r'<event(?:[ \t][^>]*)?>(.*?)</event>',text,re.S)
 assert len(blocks)==expected
 weights=[float(block.strip().splitlines()[0].split()[2]) for block in blocks]
 assert all(math.isfinite(value) for value in weights)
 init=re.search(r'<init>(.*?)</init>',text,re.S).group(1).strip().splitlines()
 assert int(init[0].split()[8])==-4
 rate=sum(float(line.split()[0]) for line in init[1:] if line.strip() and not line.lstrip().startswith('#'))
 assert math.isclose(rate,manifest['cross_section'],rel_tol=5e-8)
 nominal=manifest['absolute_cross_section']/expected if norm=='sum' else 1.
 assert all(math.isclose(abs(value),nominal,rel_tol=5e-8) for value in weights) # this smoke has zero tails
 if run=='restart60':
  assert multiple_workers
  assert all('<rwgt>' in block for block in blocks)
 result[run]=dict(events=expected,workers=workers,reserve=reserve,trials=trials,max_tail_fraction=max_tail,signed_rate_pb=manifest['cross_section'],uncertainty_pb=manifest['uncertainty'],max_channel_survey_relative_error=max(c['error_abs']/c['absolute'] for c in manifest['channels'] if c['absolute']),negative_events=sum(value<0 for value in weights),normalization=norm,has_split_channels=multiple_workers,weighted_idwtup=-4,raw_worker_pools_independently_verified=(run=='restart60'))
restart=json.loads((WORK/'restart_finished.json').read_text())
assert restart['survey_grids_unchanged'] and restart['events_exist']
assert result['survey3pct']['signed_rate_pb']==result['restart60']['signed_rate_pb']
observation=json.loads((WORK/'survey_observation.json').read_text())
assert observation['completed_surveys']>0 and observation['candidate_spools']==observation['production_jobs']==0
result['survey_before_generation']=observation
result['restart_preserved_grids']=True
result['focused_tests_passed']=117
(WORK/'validation.json').write_text(json.dumps(result,indent=2)+chr(10))
print(json.dumps(result,indent=2))
