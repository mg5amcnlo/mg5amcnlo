#!/usr/bin/env python3
"""Independent stdlib-only proposal-corrected MINT tail and replay audit."""
from pathlib import Path
import math,json,re,gzip,sys
work=Path(sys.argv[1]);root=work/'mint_tail';ref=work/'mint';sp=root/'SubProcesses'
def same(x,y):assert math.isclose(x,y,rel_tol=2e-10,abs_tol=1e-12),(x,y)
def ev(p):
 s=gzip.open(p,'rt').read() if p.suffix=='.gz' else p.read_text();return re.findall(r'<event>.*?</event>',s,re.S)
records=[];parents={};all_events=[];tail_events=0;dims=set();maxratio=0.;missing=[]
roundoff_count=0;roundoff_max_abs=0.;roundoff_max_scaled=0.
for path in sorted(sp.glob('P*/GF*/mint_overweight_trials.dat')):
 lines=path.read_text().splitlines();assert lines[0]=='MG5_MINT_TAIL 1';ndim,config,nchan=map(int,lines[1].split());assert nchan==1;dims.add(ndim)
 rates=list(map(float,lines[2].split()));zs=list(map(float,lines[3].split()));assert lines[-1].startswith('# END')
 streams={};accepted=0;rows=[]
 for line in lines[4:-1]:
  row=list(map(float,line.split()));i,stream,acc,eid,f,sgn,h,z,a=row;i=int(i);stream=int(stream);acc=int(acc)
  excess=abs(sgn)-f;scale=max(f,h,z)
  if excess>1e-10*f+1e-12:
   assert excess<=1e-12*scale,(path,i,excess,scale)
   roundoff_count+=1;roundoff_max_abs=max(roundoff_max_abs,excess);roundoff_max_scaled=max(roundoff_max_scaled,excess/scale)
  same(a,f*z/h);same(z,zs[stream-1]);assert i==len(rows)+1;rows.append(row);flag=f>h;maxratio=max(maxratio,f/h)
  streams.setdefault(stream,[]).append((a,a if flag else 0.,(f-h)*z/h if flag else 0.,sgn*z/h))
  if flag:assert acc;tail_events+=1
  if acc:accepted+=1;assert eid==accepted
 actual=ev(path.parent/'events.lhe');reference=ev(ref/path.parent.relative_to(root)/'events.lhe');assert actual==reference;assert accepted==len(actual);all_events.extend(actual)
 parent=str(path.parent.relative_to(sp)).split('_ampli')[0];parent=str(Path(parent).parent/Path(parent).name.split('_')[0]);assert parent not in parents
 summary=[]
 for stream,rate in ((1,rates[1]),(2,rates[0])):
  pts=streams.get(stream,[])
  if rate and not pts:missing.append((parent,stream,rate));continue
  if not pts:continue
  n=len(pts);total=math.fsum(p[0] for p in pts);tail=math.fsum(p[1] for p in pts);fraction=tail/total
  serr=math.sqrt(n*math.fsum((p[1]-fraction*p[0])**2 for p in pts)/((n-1)*total*total)) if n>1 else 0.
  means=[math.fsum(p[j] for p in pts)/n for j in range(4)]
  cov=math.fsum((p[0]-means[0])*(p[1]-means[1]) for p in pts)/(n*(n-1)) if n>1 else 0.
  vara=math.fsum((p[0]-means[0])**2 for p in pts)/(n*(n-1)) if n>1 else 0.
  vart=math.fsum((p[1]-means[1])**2 for p in pts)/(n*(n-1)) if n>1 else 0.
  rec=dict(channel=parent,stream=stream,rate=rate,n=n,frac=fraction,frac_err=serr,mean=means,var_abs=vara,var_tail=vart,cov=cov);records.append(rec);summary.append(rec)
 parents[parent]=dict(events=accepted,trials=len(rows),rate=sum(r['rate'] for r in summary),tail=sum(r['rate']*r['frac'] for r in summary)/sum(r['rate'] for r in summary))
expected={str(p.parent.relative_to(sp)):float(p.read_text().split()[0]) for p in sp.glob('P*/GF*/res_1.dat') if not p.is_symlink()}
assert set(expected)==set(parents),(set(expected)-set(parents));assert not missing,missing
actual=ev(root/'Events/benchmark_10k/events.lhe.gz');reference=ev(ref/'Events/benchmark_10k/events.lhe.gz');assert actual==reference
assert sorted(actual)==sorted(all_events);assert len(actual)==10000
rate=math.fsum(r['rate'] for r in records);tail=math.fsum(r['rate']*r['frac'] for r in records);error=math.sqrt(math.fsum((r['rate']*r['frac_err'])**2 for r in records))/rate
prod_abs=math.fsum(r['mean'][0] for r in records);prod_tail=math.fsum(r['mean'][1] for r in records);ratio=prod_tail/prod_abs
proderr=math.sqrt(math.fsum(r['var_tail']+ratio**2*r['var_abs']-2*ratio*r['cov'] for r in records))/prod_abs
report=dict(all_raw_checks_passed=True,parent_channels=len(parents),streams=len(records),coverage=1.,missing_streams=missing,dimensions=sorted(dims),trials=sum(p['trials'] for p in parents.values()),events=len(actual),tail_events=tail_events,survey_rate=rate,survey_full_tail=tail/rate,survey_full_tail_error=error,production_full_tail=ratio,production_full_tail_error=proderr,production_absolute=prod_abs,production_signed=sum(r['mean'][3] for r in records),max_channel_tail=max(p['tail'] for p in parents.values()),max_weight_ratio=maxratio,exact_worker_and_final_event_replay=True,channels=parents)
report['signed_absolute_roundoff']=dict(rows=roundoff_count,maximum_absolute_excess=roundoff_max_abs,maximum_excess_over_proposal_scale=roundoff_max_scaled,note='Signed wgts and grouped absolute unwgt sums use different cancellation orders; observations are unchanged.')
print(json.dumps(report,indent=2))
