#!/usr/bin/env python3
"""Independent stdlib-only raw POOL4 audit; no MG5/benchmark imports.

Usage: independent_audit.py WORK_DIRECTORY [--mint SAVED_MINT_DIRECTORY]
       [--output report.json]

The new run must use survey stage 1 directly, at least four survey iterations,
and the final survey iteration alone as its saved statistical estimate. Saved stream maxima, first-epoch envelope floors, candidate cutoffs, and bounded
nonzero budgets are reconstructed independently from the raw epoch rows.
This script imports neither the integrator package nor the benchmark analyzer.

Each worker pool is parsed separately and final LHE weights are streamed, so
memory scales with one channel's reserve rather than the full event document.
The benchmark requests 300,000 events with no folding, and split workers retain
their original 2,500-event limit. All production/top-up batches are audited.
"""
from pathlib import Path
import argparse,hashlib,math,json,random,gzip,re,sys
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('work',type=Path,nargs='?')
parser.add_argument('--mint',type=Path,default=HERE.parent/'ttbar_unfolded_300k_20261007/mint')
parser.add_argument('--previous-ampli',type=Path,default=HERE.parent/'ttbar_unfolded_300k_20261007/ampli')
parser.add_argument('--run-name',default='bounded_300k')
parser.add_argument('--reference-run-name',default='benchmark_300k')
parser.add_argument('--stopped',type=Path,default=HERE.parent/'ttbar_two_stage_300k_20261007/partial')
parser.add_argument('--output',type=Path)
args=parser.parse_args()
work=args.work or Path((HERE/'work_directory.txt').read_text().strip())
out=work/'ampli';sp=out/'SubProcesses'
run_name=args.run_name;expected_events=300000
metadata=json.loads((HERE/'ampli_started.json').read_text())
assert metadata['expected_events']==expected_events and metadata['seed']==19727
assert metadata['req_acc']==-1 and metadata['folding']==[1,1,1]
assert metadata['run_name']==run_name and metadata['fresh_export'] and metadata['fresh_survey']
NUMBER=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?'
def real(value):return float(value.replace('D','E').replace('d','e'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
provenance={}
for name,expected in metadata['source_sha256'].items():
 actual=digest(HERE/'sources'/name);assert actual==expected,(name,actual,expected)
 provenance['sources/'+name]=actual
for name,expected in metadata['exported_source_sha256'].items():
 actual=digest(out/name);assert actual==expected,(name,actual,expected)
 provenance[name]=actual
assert provenance['SubProcesses/simple_integrator.f90']=='0e3a137fd13b8bdd5881c640de29ae9eb7a137006166d6f2f1d5ebb7ea7e54c8'
manifest=json.loads((out/'Events'/run_name/'ampli_production.json').read_text())
assert manifest['requested_events']==expected_events
assert manifest['allowed_overweight_factor']==.01
assert manifest['rate_source']=='survey_plus_production'
assert manifest['sampling']=='native_adaptive_unfolded'
assert manifest['integration_stages']==['survey','generation']
assert manifest['survey_min_iterations']==4
rng=random.Random(manifest['seed'] ^ 0x504F4F4C)

def same(x,y):
 assert math.isclose(x,y,rel_tol=2e-8,abs_tol=2e-10),(x,y)

def frac(items):
 return math.fsum(x for x,t in items if t)/math.fsum(x for x,t in items) if items else 0.

def worst(items,n):
 ts=[x for x,t in items if t];rest=sorted(x for x,t in items if not t)
 return 1. if n and len(ts)>=n else frac([(x,True) for x in ts]+[(x,False) for x in rest[:n-len(ts)]]) if n else 0.

weights=[];tails=[];counts={'workers':0,'trials':0,'nonzero':0,'iterations':0,'updates':0,'candidates':0,'reserves':0,'rethresholded':0,'tail_events':0};maxchecks=0.;dims=set();rate_sums=[0.,0.,0.,0.]
assert not list(sp.glob('P*/G*/log_MINT0.txt')), 'Unexpected stage 0 log'
assert not list(sp.glob('P*/G*/res_0.dat')), 'Unexpected stage 0 result'
surveys=[];first_epoch_forecasts=[]
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
 parent=sp/ch['subprocess']/('GF'+ch['channel'])
 survey=list(map(real,(parent/'res_1.dat').read_text().split()))
 assert survey[1]<=.03*survey[0]*(1.+1e-10)
 assert int(survey[4])==ch['survey_iterations']>=4
 assert int(survey[5])==ch['survey_points']
 surveylog=(parent/'log_MINT1.txt').read_text()
 iterations=re.findall(r'AmpliCol survey iteration, cumulative trials, ABS, signed, relative error:\s*(\d+)\s+(\d+)\s*('+NUMBER+r')\s*('+NUMBER+r')\s*('+NUMBER+')',surveylog)
 assert len(iterations)>=4 and int(iterations[-1][0])==int(survey[4])
 statistical=re.search(r'AmpliCol survey retained statistical points, total evaluations:\s*(\d+)\s+(\d+)',surveylog)
 assert statistical
 retained,total=map(int,statistical.groups())
 assert retained==int(survey[5])==int(iterations[-1][1])-int(iterations[-2][1])
 assert total==int(iterations[-1][1]) and retained<total
 same(real(iterations[-1][2]),survey[0]);same(real(iterations[-1][3]),survey[2])
 same(real(iterations[-1][4]),survey[1]/survey[0])
 checkpoint=(parent/'ampli_grids').read_text().splitlines()
 assert checkpoint[0].split()==['MG5_AMPLICOL','3','1']
 dimension,configuration,sector,nvalues=map(int,checkpoint[1].split())
 assert list(map(int,checkpoint[2].split()))==[1]*dimension
 for name in ('ampli_grids','grid.MC_integer'):
  reference=args.stopped/'SubProcesses'/ch['subprocess']/('GF'+ch['channel'])/name
  assert (parent/name).read_bytes()==reference.read_bytes(),('Changed survey checkpoint',parent,name)
 saved=list(map(real,checkpoint[3].split()));assert len(saved)==2*nvalues+2
 saved_abs=saved[0]+saved[4];same(saved_abs,survey[0]);same(saved[1],survey[2])
 same(real(checkpoint[4]),survey[1])
 probability=max(.001,min(.999,saved[4]/saved_abs))
 stream_maxima=saved[-2:]
 initial_envelope=max(stream_maxima[0]/(1.-probability),stream_maxima[1]/probability)
 surveys.append(dict(subprocess=ch['subprocess'],channel=ch['channel'],iterations=int(survey[4]),
  retained_statistical_points=retained,total_evaluations=total,relative_absolute_error=survey[1]/survey[0],
  stream_maxima=stream_maxima,virtual_probability=probability,initial_envelope=initial_envelope,
  checkpoint_sha256=digest(parent/'ampli_grids')))

 n=int(survey[5]);means=[survey[0],survey[2]];variance=[survey[1]**2,survey[3]**2];channel_items=[];channel_full=[]
 for b in ch['batches']:
  p=out/b['directory']/'ampli_pool.dat';lines=iter(p.read_text().splitlines());assert next(lines).split()==['MG5_AMPLI_POOL','4']
  trials,nc,ne=map(int,next(lines).split());moments=list(map(float,next(lines).split()));target,quota,logz,full,reserve,subset=map(float,next(lines).split());target=int(target);quota=int(quota)
  ndim,updates=map(int,next(lines).split());mask=list(map(int,next(lines).split()));assert mask==[1]*ndim;dims.add(ndim)
  epochs=[]
  for _ in range(ne):
   row=next(lines).split();assert len(row)==13;e=list(map(real,row));epochs.append(e)
   assert int(e[0])==len(epochs) and 0<int(e[3])<=int(e[2])<=int(e[1])
   assert e[-1] in (0.,1.)
   if e[-1]:assert e[11]>=e[9] and e[10]>0
   size=int(e[1]);merged=n+size
   for j,mi in enumerate((4,5)):
    v=e[mi];m2=e[mi+2]
    variance[j]=(n*n*variance[j]+m2)/(merged*merged)+n*size*(means[j]-v)**2/(merged**3)
    means[j]=(n*means[j]+size*v)/merged
   n=merged
  rows=[list(map(real,next(lines).split())) for _ in range(nc)];assert not list(lines)
  assert 0<=updates<=max(0,ne-1)
  assert all(len(r)==6 and all(math.isfinite(x) for x in r) and r[1]>0 and r[3]>=0 and r[4] in (0.,1.) and r[5]>0 for r in rows)
  event_epochs=sorted({int(r[0]) for r in rows})
  assert [int(e[0]) for e in epochs if e[-1]]==event_epochs[-8:]
  same(epochs[0][9],min(initial_envelope,saved_abs))
  assert epochs[0][10]>=initial_envelope*(1.-1.e-12)
  expected_nonzero=max(1024,min(8192,target))
  assert int(epochs[0][3])==expected_nonzero
  assert all(e[2]==e[3] for e in epochs)
  assert all(right[3]<=2*left[3] for left,right in zip(epochs,epochs[1:]))
  first_epoch_forecasts.append(dict(directory=b['directory'],initial_cutoff=epochs[0][9],
   target_nonzero=expected_nonzero,generated_target=target,
   survey_envelope_over_absolute=initial_envelope/saved_abs,
   iteration_targets=[int(e[3]) for e in epochs],
   previous_scheduler_initial_target=max(1024,math.ceil(target*max(1.,initial_envelope/saved_abs)))))
  workerlog=(out/b['directory']/'log_MINT2.txt').read_text()
  printed=re.search(r'AmpliCol saved survey stream maxima, initial production envelope:\s*('+NUMBER+r')\s*('+NUMBER+r')\s*('+NUMBER+')',workerlog)
  assert printed
  for actual,wanted in zip(map(real,printed.groups()),(*stream_maxima,initial_envelope)):same(actual,wanted)
  logged=re.search(r'AmpliCol generation trials, candidates, requested events:\s*(\d+)\s+(\d+)\s+(\d+)',workerlog)
  assert logged and tuple(map(int,logged.groups()))==(trials,nc,target)
  forecasts=re.findall(r'AmpliCol native effective events, remaining, nonzero acceptance:\s*([^\n]+)',workerlog)
  assert len(forecasts)==ne-1
  for i,row in enumerate(forecasts):
   effective,remaining,efficiency=map(real,row.split())
   same(remaining,max(1.,target-effective))
   expected=int(2*epochs[i][3])
   if efficiency>0:
    expected=max(min(1024,expected),min(expected,math.ceil(1.1*remaining/efficiency)))
   assert epochs[i+1][3]==expected,(p,i,epochs[i+1][3],expected)

  assert sum(int(e[1]) for e in epochs)==trials
  aggregate=[math.fsum(e[1]*e[j] for e in epochs)/trials for j in (4,5)]
  aggregate.extend(math.fsum(e[j+2]+e[1]*(e[j]-aggregate[j-4])**2 for e in epochs) for j in (4,5))
  aggregate.append(math.fsum(e[8]+e[1]*(e[4]-aggregate[0])*(e[5]-aggregate[1]) for e in epochs))
  for actual,expected in zip(moments,aggregate):same(actual,expected)
  assert target==(11*quota+9)//10 and 0<target<=2500
  assert trials==b['trials'] and target==b['generated_target'] and quota==b['nominal_final_quota']
  assert b['adaptation']['mask']==mask and b['adaptation']['updates']==updates
  assert b['adaptation']['schedule']=='nonzero_trials'
  active=sum(e[1]*e[4] for e in epochs if e[-1]);retained=[];rawcorr=[];tail_mass=0.
  for r in rows:
   eid,w,prio,c,t,factor=r;assert eid==int(eid) and 1<=eid<=ne;e=epochs[int(eid)-1]
   assert prio>=max(math.log(w),math.log(e[9]))-1.e-10
   if not e[-1]: assert c==t==0.;continue
   same(math.log(e[11]),math.log(e[10])+logz)
   assert bool(t)==(w>e[11]);rank=prio-math.log(e[10])
   if c: assert rank>=logz-1e-10;retained.append(r);rawcorr.append(max(1.,w/e[11]))
   else: assert rank<=logz+1e-10 and not t
   if t:tail_mass+=w
  assert len(retained)==target
  scale=target/math.fsum(rawcorr) if rawcorr else 1.
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
actual=[];expect_header=False;closed=False;event_open=False;closed_events=0;in_init=False;init_rows=[]
with gzip.open(out/'Events'/run_name/'events.lhe.gz','rt') as stream:
 for line in stream:
  line=line.strip()
  if line=='<init>':in_init=True;continue
  if line=='</init>':in_init=False;continue
  if in_init:
   if line and not line.startswith('#'):init_rows.append(line.split())
   continue
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
assert init_rows and int(init_rows[0][8])==-4
assert int(init_rows[0][9])==len(init_rows)-1
same(sum(real(row[0]) for row in init_rows[1:]),manifest['cross_section'])
# The LHE init formatting is less precise than the numerical sidecars.
assert math.isclose(math.sqrt(sum(real(row[1])**2 for row in init_rows[1:])),manifest['uncertainty'],rel_tol=5.e-5)
sum_weights=math.fsum(actual);sum_absolute=math.fsum(map(abs,actual));sum_squares=math.fsum(w*w for w in actual)
signed_effective_events=sum_weights**2/sum_squares
absolute_effective_events=sum_absolute**2/sum_squares
result=dict(lhe_init_rate_and_error_reproduced=True,signed_effective_events=signed_effective_events,absolute_effective_events=absolute_effective_events,sum_weights=sum_weights,sum_absolute_weights=sum_absolute,negative_absolute_weight_fraction=math.fsum(-w for w in actual if w<0)/sum_absolute,two_stage_survey_contract_passed=True,no_stage_zero=True,survey_minimum_four_iterations=True,final_survey_iteration_only=True,saved_stream_maxima_forecasts_reproduced=True,surveys=surveys,first_epoch_forecasts=first_epoch_forecasts,all_raw_checks_passed=True,run_name=run_name,expected_events=expected_events,channels=len(manifest['channels']),dimensions=sorted(dims),counts=counts,collection_rounds=len(manifest['rounds']),allocation_reproduced=True,absolute_pb=rate_sums[0],signed_pb=rate_sums[1],error_signed_pb=math.sqrt(rate_sums[3]),max_tail_check=maxchecks,collected_tail_fraction=collected,negative_events=sum(w<0 for w in actual),minimum_weight=min(map(abs,actual)),maximum_weight=max(map(abs,actual)),lhe_document_closed=closed)
cpu={};settings={}
for backend in ['mint','ampli']:
 proc=args.mint if backend=='mint' else out;backend_run=args.reference_run_name if backend=='mint' else run_name
 banner=(proc/'Events'/backend_run/(backend_run+'_tag_1_banner.txt')).read_text();settings[backend]={}
 for key in ['req_acc','nevents','nevt_job','folding','iseed','lpp1','lpp2','ebeam1','ebeam2','event_norm']:
  match=re.search(r'^\s*([^\n=!]+?)\s*=\s*'+key+r'\s*!',banner,re.M);assert match,key
  settings[backend][key]=match.group(1).strip()
 assert settings[backend]['event_norm'].lower()=='average'
 assert float(settings[backend]['req_acc'])==-1
 assert list(map(int,re.findall(r'\d+',settings[backend]['folding'])))==[1,1,1]
 assert int(settings[backend]['nevents'])==expected_events and int(settings[backend]['nevt_job'])==2500
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
assert cpu['ampli']['0']['workers']==0
assert cpu['ampli']['1']['workers']==len(manifest['channels'])
assert cpu['ampli']['2']['workers']==counts['workers']
same(cpu['ampli']['2']['cpu_seconds'],manifest['generation_cpu_seconds'])
card_comparison={}
for card in ('run_card.dat','param_card.dat','FKS_params.dat'):
 new_digest=digest(out/'Cards'/card)
 old_digest=digest(args.previous_ampli/'Cards'/card)
 card_comparison[card]=dict(new_ampli=new_digest,previous_ampli=old_digest,identical=new_digest==old_digest)
 assert new_digest==old_digest,('AmpliCol card changed',card)
result['new_ampli_cards_identical_to_previous']=card_comparison
mint_channels=[];mint_trials=0
for log in (args.mint/'SubProcesses').glob('P*/G*/log_MINT1.txt'):
 if log.is_symlink():continue
 mint_channels.append(list(map(real,(log.parent/'res_1.dat').read_text().split())))
for log in (args.mint/'SubProcesses').glob('P*/G*/log_MINT2.txt'):
 if log.is_symlink():continue
 found=re.findall(r'another call to the function:\s*(\d+)',log.read_text());assert found
 mint_trials+=int(found[-1])
mint_rate=math.fsum(row[2] for row in mint_channels)
mint_error=math.sqrt(math.fsum(row[3]**2 for row in mint_channels))
previous=json.loads((args.previous_ampli/'Events'/args.reference_run_name/'ampli_production.json').read_text())
previous_cpu={}
for stage in range(3):
 times=[]
 for log in (args.previous_ampli/'SubProcesses').glob(f'P*/G*/log_MINT{stage}.txt'):
  if log.is_symlink():continue
  found=re.findall(r'Time spent in Total\s*:\s*('+NUMBER+')',log.read_text());assert found
  times.append(real(found[-1]))
 previous_cpu[str(stage)]=dict(workers=len(times),cpu_seconds=math.fsum(times))
result['previous_three_stage_ampli_cpu']=previous_cpu
comparison={}
new_generation=cpu['ampli']['2']['cpu_seconds'];new_total=sum(c['cpu_seconds'] for c in cpu['ampli'].values())
for name,ref_trials,ref_rate,ref_error,ref_cpu in (
 ('saved_mint',mint_trials,mint_rate,mint_error,cpu['mint']),
 ('previous_three_stage_ampli',previous['generation_trials'],previous['cross_section'],previous['uncertainty'],previous_cpu)):
 ref_generation=ref_cpu['2']['cpu_seconds'];ref_total=sum(c['cpu_seconds'] for c in ref_cpu.values())
 comparison[name]=dict(reference_trials=ref_trials,reference_signed_pb=ref_rate,reference_error_pb=ref_error,
  final_efficiency_reference=expected_events/ref_trials,final_efficiency_new=expected_events/counts['trials'],
  final_efficiency_ratio_new_over_reference=ref_trials/counts['trials'],
  generation_cpu_ratio_new_over_reference=new_generation/ref_generation,
  total_worker_cpu_ratio_new_over_reference=new_total/ref_total,
  signed_rate_difference_pb=rate_sums[1]-ref_rate,
  nominal_combined_error_pull=(rate_sums[1]-ref_rate)/math.hypot(ref_error,math.sqrt(rate_sums[3])),
  signed_error_ratio_new_over_reference=math.sqrt(rate_sums[3])/ref_error)
result['comparison']=comparison
result['costs_and_efficiencies']=dict(generation_cpu_seconds=new_generation,total_worker_cpu_seconds=new_total,
 final_events_per_generation_trial=expected_events/counts['trials'],
 final_events_per_generation_cpu_second=expected_events/new_generation,
 final_events_per_total_cpu_second=expected_events/new_total,
 signed_effective_events_per_generation_cpu_second=signed_effective_events/new_generation,
 signed_effective_events_per_total_cpu_second=signed_effective_events/new_total)
result['source_provenance']=provenance
result['bounded_scheduler']=dict(survey_checkpoints_identical_to_stopped_300k=True,
 first_batch_uses_bounded_quota=True,first_epoch_retains_saved_maximum=True,
 candidate_cutoff_is_minimum_of_absolute_rate_and_maximum=True,
 continuation_forecasts_recomputed=True,maximum_growth_factor=2,
 shrinking_iterations=sum(sum(b<a for a,b in zip(w['iteration_targets'],w['iteration_targets'][1:])) for w in first_epoch_forecasts))
result['paths']=dict(work=str(work.resolve()),new_ampli=str(out.resolve()),mint=str(args.mint.resolve()),previous_ampli=str(args.previous_ampli.resolve()))
serialized=json.dumps(result,indent=2,allow_nan=False)+'\n'
if args.output:args.output.write_text(serialized)
print(serialized,end='')

