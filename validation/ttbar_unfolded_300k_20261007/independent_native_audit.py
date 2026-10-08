#!/usr/bin/env python3
"""Independent stdlib-only raw POOL4 audit; no MG5/benchmark imports.

Usage: independent_native_audit.py WORK_DIRECTORY > report.json

Each worker pool is parsed separately and final LHE weights are streamed, so
memory scales with one channel's reserve rather than the full event document.
The benchmark requests 300,000 events with no folding, and split workers retain
their original 2,500-event limit. All production/top-up batches are audited.
"""
from pathlib import Path
import math,json,random,gzip,re,sys
work=Path(sys.argv[1]);out=work/'ampli';sp=out/'SubProcesses'
run_name='benchmark_300k'; expected_events=300000
manifest=json.loads((out/'Events'/run_name/'ampli_production.json').read_text())
assert manifest['requested_events']==expected_events
assert manifest['allowed_overweight_factor']==.01
assert manifest['rate_source']=='survey_plus_production'
assert manifest['sampling']=='native_adaptive_unfolded'
rng=random.Random(manifest['seed'] ^ 0x504F4F4C)

def same(x,y):
 assert math.isclose(x,y,rel_tol=2e-8,abs_tol=2e-10),(x,y)

def frac(items):
 return math.fsum(x for x,t in items if t)/math.fsum(x for x,t in items) if items else 0.

def worst(items,n):
 ts=[x for x,t in items if t];rest=sorted(x for x,t in items if not t)
 return 1. if n and len(ts)>=n else frac([(x,True) for x in ts]+[(x,False) for x in rest[:n-len(ts)]]) if n else 0.

weights=[];tails=[];counts={'workers':0,'trials':0,'nonzero':0,'iterations':0,'updates':0,'candidates':0,'reserves':0,'rethresholded':0,'tail_events':0};maxchecks=0.;dims=set();rate_sums=[0.,0.,0.,0.]
ordered=sorted(manifest['channels'],key=lambda ch:ch['allocation_order'])
assert [ch['allocation_order'] for ch in ordered]==list(range(len(ordered)))
for field,quota_field in [('survey_absolute','initial_quota'),('absolute','quota')]:
 draw=random.Random(manifest['seed'] ^ 0x414D504C);expected=[0]*len(ordered)
 total=math.fsum(ch[field] for ch in ordered if ch[field]>0.)
 for _ in range(expected_events):
  point=draw.random()*total;cumulative=0.
  for i,ch in enumerate(ordered):
   cumulative+=max(ch[field],0.)
   if point<cumulative:expected[i]+=1;break
 for ch,count in zip(ordered,expected):assert ch[quota_field]==count
for ch in manifest['channels']:
 survey=list(map(float,(sp/ch['subprocess']/('GF'+ch['channel'])/'res_1.dat').read_text().split()))
 assert survey[1]<=.03*survey[0]*(1.+1e-10)
 n=int(survey[5]);means=[survey[0],survey[2]];variance=[survey[1]**2,survey[3]**2];channel_items=[];channel_full=[]
 for b in ch['batches']:
  p=out/b['directory']/'ampli_pool.dat';lines=iter(p.read_text().splitlines());assert next(lines).split()==['MG5_AMPLI_POOL','4']
  trials,nc,ne=map(int,next(lines).split());moments=list(map(float,next(lines).split()));target,quota,logz,full,reserve,subset=map(float,next(lines).split());target=int(target);quota=int(quota)
  ndim,updates=map(int,next(lines).split());mask=list(map(int,next(lines).split()));assert mask==[1]*ndim;dims.add(ndim)
  epochs=[]
  for _ in range(ne):
   row=next(lines).split();e=list(map(float,row));epochs.append(e)
   size=int(e[1]);merged=n+size
   for j,mi in enumerate((4,5)):
    v=e[mi];m2=e[mi+2]
    variance[j]=(n*n*variance[j]+m2)/(merged*merged)+n*size*(means[j]-v)**2/(merged**3)
    means[j]=(n*means[j]+size*v)/merged
   n=merged
  rows=[list(map(float,next(lines).split())) for _ in range(nc)];assert not list(lines)
  assert sum(int(e[1]) for e in epochs)==trials
  aggregate=[math.fsum(e[1]*e[j] for e in epochs)/trials for j in (4,5)]
  aggregate.extend(math.fsum(e[j+2]+e[1]*(e[j]-aggregate[j-4])**2 for e in epochs) for j in (4,5))
  aggregate.append(math.fsum(e[8]+e[1]*(e[4]-aggregate[0])*(e[5]-aggregate[1]) for e in epochs))
  for actual,expected in zip(moments,aggregate):same(actual,expected)
  assert target==(11*quota+9)//10
  assert trials==b['trials'] and target==b['generated_target'] and quota==b['nominal_final_quota']
  assert b['adaptation']['mask']==mask and b['adaptation']['updates']==updates
  assert b['adaptation']['schedule']=='nonzero_trials'
  active=sum(e[1]*e[4] for e in epochs if e[-1]);retained=[];rawcorr=[];tail_mass=0.
  for r in rows:
   eid,w,prio,c,t,factor=r;e=epochs[int(eid)-1]
   if not e[-1]: assert c==t==0.;continue
   same(math.log(e[11]),math.log(e[10])+logz)
   assert bool(t)==(w>e[11]);rank=prio-math.log(e[10])
   if c: assert rank>=logz-1e-10;retained.append(r);rawcorr.append(max(1.,w/e[11]))
   else: assert rank<=logz+1e-10 and not t
   if t:tail_mass+=w
  assert len(retained)==target
  scale=target/math.fsum(rawcorr)
  for r,c in zip(retained,rawcorr):same(r[3],c*scale)
  items=[(r[3]*r[5],bool(r[4])) for r in retained]
  measured=(tail_mass/active,frac(items),worst(items,quota))
  for x,y in zip(measured,(full,reserve,subset)):same(x,y);assert y<.01
  maxchecks=max(maxchecks,*measured)
  if b['collection_log_z']!=logz:
   counts['rethresholded']+=1
   newz=max([logz]+[math.log(r[1])-math.log(epochs[int(r[0])-1][10])+1e-12 for r in rows if epochs[int(r[0])-1][-1]])
   same(newz,b['collection_log_z']);retained=[];rawcorr=[]
   for r in rows:
    e=epochs[int(r[0])-1]
    if e[-1] and r[2]-math.log(e[10])>newz:
     threshold=max(e[9],math.exp(math.log(e[10])+newz));retained.append([*r[:3],0.,float(r[1]>threshold),r[5]]);rawcorr.append(max(1.,r[1]/threshold))
   scale=len(retained)/math.fsum(rawcorr) if rawcorr else 1.
   for r,c in zip(retained,rawcorr):r[3]=c*scale
  assert len(retained)==b['available']
  channel_items.extend((r[3]*r[5],bool(r[4])) for r in retained)
  channel_full.append(0. if b['collection_log_z']!=logz else measured[0])
  for key,val in [('workers',1),('trials',trials),('nonzero',sum(int(e[2]) for e in epochs)),('iterations',ne),('updates',updates),('candidates',nc),('reserves',target)]:counts[key]+=val
 for k,v in zip(('absolute','signed','error_abs','error_signed'),(*means,math.sqrt(variance[0]),math.sqrt(variance[1]))):same(v,ch[k]);same(v,ch['updated_rates'][k])
 assert n==ch['updated_rates']['trials']
 assert len(channel_items)==ch['available_candidates']
 if ch['quota']:
  diag=ch['selection'];checks=[max(channel_full),frac(channel_items),worst(channel_items,ch['quota'])]
  for key,value in zip(('full_trial_tail','reserve_tail','worst_subset_tail'),checks):same(diag[key],value);assert value<.01
  selected=rng.sample(channel_items,ch['quota']);same(diag['selected_tail'],frac(selected));assert frac(selected)<.01
  maxchecks=max(maxchecks,*checks)
  weights.extend(x*manifest['absolute_cross_section'] for x,t in selected);tails.extend(t for x,t in selected)
 for i,v in enumerate((means[0],means[1],variance[0],variance[1])):rate_sums[i]+=v
same(rate_sums[0],manifest['absolute_cross_section']);same(rate_sums[1],manifest['cross_section']);same(math.sqrt(rate_sums[3]),manifest['uncertainty'])
assert counts['trials']==manifest['generation_trials']
actual=[];expect_header=False;closed=False;event_open=False;closed_events=0
with gzip.open(out/'Events'/run_name/'events.lhe.gz','rt') as stream:
 for line in stream:
  line=line.strip()
  if re.fullmatch(r'<event(?:\s[^>]*)?>',line):
   assert not event_open
   event_open=True;expect_header=True
  elif expect_header and line and not line.startswith('#'):
   fields=line.split();assert len(fields)>=6
   actual.append(float(fields[2].replace('D','E')))
   assert math.isfinite(actual[-1]) and actual[-1]
   expect_header=False
  elif line=='</event>':
   assert event_open and not expect_header
   event_open=False;closed_events+=1
  elif line=='</LesHouchesEvents>':
   assert not event_open
   closed=True
assert closed and not event_open and not expect_header
assert len(actual)==len(weights)==closed_events==expected_events
for x,y in zip(sorted(map(abs,actual)),sorted(weights)):assert math.isclose(x,y,rel_tol=5e-7),(x,y)
collected=frac(list(zip(weights,tails)));assert collected<.01
counts['tail_events']=sum(tails)
result=dict(all_raw_checks_passed=True,run_name=run_name,expected_events=expected_events,channels=len(manifest['channels']),dimensions=sorted(dims),counts=counts,collection_rounds=len(manifest['rounds']),allocation_reproduced=True,absolute_pb=rate_sums[0],signed_pb=rate_sums[1],error_signed_pb=math.sqrt(rate_sums[3]),max_tail_check=maxchecks,collected_tail_fraction=collected,negative_events=sum(w<0 for w in actual),minimum_weight=min(map(abs,actual)),maximum_weight=max(map(abs,actual)),lhe_document_closed=closed)
cpu={};settings={}
for backend in ['mint','ampli']:
 proc=work/backend;banner=(proc/'Events'/run_name/(run_name+'_tag_1_banner.txt')).read_text();settings[backend]={}
 for key in ['req_acc','nevents','nevt_job','folding','iseed','lpp1','lpp2','ebeam1','ebeam2']:
  match=re.search(r'^\s*([^\n=!]+?)\s*=\s*'+key+r'\s*!',banner,re.M);assert match,key
  settings[backend][key]=match.group(1).strip()
 assert float(settings[backend]['req_acc'])==-1
 assert settings[backend]['iseed']=='19727'
 assert (proc/'SubProcesses/randinit').read_text().strip()=='r=19727'
 cpu[backend]={}
 for stage in range(3):
  seconds=[]
  for log in (proc/'SubProcesses').glob(f'P*/G*/log_MINT{stage}.txt'):
   if log.is_symlink():continue
   found=re.findall(r'Time spent in Total\s*:\s*([\d.eEdD+-]+)',log.read_text());assert found,log
   seconds.append(float(found[-1].replace('D','e')))
  cpu[backend][str(stage)]={'workers':len(seconds),'cpu_seconds':sum(seconds)}
assert settings['mint']==settings['ampli']
result.update(independent_log_cpu_seconds=cpu,independent_banner_settings=settings,
              randinit_seed_confirmed=19727,settings_identical=True)
print(json.dumps(result,indent=2))
