"""Record weighted ttbar shape diagnostics for each actual worker collection.

These channel-only, shared-seed samples are correlated and are not a coverage
study. Signed bins are normalized by the sample's absolute weight sum, retaining
negative events and native correction factors.
"""
import bisect
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
collections=json.loads((HERE/'collection_validation.json').read_text())
EDGES={'top_pt':[0.,50.,100.,200.,400.,800.,float('inf')],
       'ttbar_mass':[0.,400.,500.,700.,1000.,1500.,3000.,float('inf')]}

def events(path):
 with path.open() as stream:
  for line in stream:
   if line.strip()!='<event>':continue
   header=next(stream).split();n=int(header[0]);weight=float(header[2]);tops={}
   for _ in range(n):
    p=next(stream).split();pdg=int(p[0]);status=int(p[1])
    if abs(pdg)==6 and status==1:tops[pdg]=list(map(float,p[6:10]))
   assert set(tops)=={-6,6}
   yield weight,tops

def analyze(path):
 bins={name:[dict(signed=0.,absolute=0.,sum_squared=0.,count=0) for _ in edges[1:]] for name,edges in EDGES.items()}
 n=0;negative=0;total=0.;absolute=0.;squared=0.;max_abs=0.
 for w,tops in events(path):
  top=tops[6];pair=[tops[6][i]+tops[-6][i] for i in range(4)]
  values={'top_pt':math.hypot(top[0],top[1]),'ttbar_mass':math.sqrt(max(0.,pair[3]**2-sum(x*x for x in pair[:3])))}
  n+=1;negative+=(w<0);total+=w;absolute+=abs(w);squared+=w*w;max_abs=max(max_abs,abs(w))
  for name,value in values.items():
   idx=bisect.bisect_right(EDGES[name],value)-1;row=bins[name][idx]
   row['signed']+=w;row['absolute']+=abs(w);row['sum_squared']+=w*w;row['count']+=1
 for name,rows in bins.items():
  for row in rows:
   row['signed_fraction_of_abs']=row['signed']/absolute;row['absolute_fraction']=row['absolute']/absolute
 return dict(events=n,negative_events=negative,signed_over_absolute=total/absolute,
  mean_absolute_weight=absolute/n,max_absolute_weight=max_abs,weight_effective_count=absolute**2/squared,bins=bins)

result={}
for variant,rows in collections.items():
 result[variant]={}
 for label,row in rows.items():result[variant][label]=analyze(Path(row['destination']))
report=dict(note=__doc__,edges={k:[x if math.isfinite(x) else 'inf' for x in v] for k,v in EDGES.items()},samples=result)
(HERE/'weighted_shape_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
print('Analyzed',sum(map(len,result.values())),'collections')
