"""Independent scalar/paired-error check of the completed no-cut comparison."""
from pathlib import Path
import hashlib,json
import numpy as np
B=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101')
f=B/'results/run_1M_wbb4fs_nocuts_lhe.npz'
g=B/'results/validation_1M/full_events.npz'
c=B/'results/validation_1M/comparison.json'
x=np.load(f);y=np.load(g);report=json.loads(c.read_text())
n=len(x['nominal']);assert n==1000000==len(y['nominal'])
assert np.array_equal(y['source_index'],np.arange(n))
assert np.allclose(x['nominal'],y['nominal'],rtol=2.e-12,atol=1.e-12)
assert np.allclose(x['macnlops'],y['macnlops'],rtol=2.e-12,atol=1.e-12)
assert np.array_equal(x['is_h'],y['is_h'])
groups=y['job_index'];ng=len(np.unique(groups));assert ng==50
assert np.all(np.bincount(groups)==20000)
def err(z,center=False):
    chunks=np.array([z[groups==k].sum() for k in range(ng)])
    if center:chunks-=chunks.mean()
    return float(np.sqrt(ng/(ng-1)*np.dot(chunks,chunks)))
results={};influences={}
for stage,data in [('first',x),('full',y)]:
    w=data['nominal'];m=data['macnlops'];h=data['is_h']
    p=data['observables'][:,0];take=(p>=5)&(p<20)
    S=np.where(take&~h,w-m,0.);H=np.where(take&h,m-w,0.)
    v=H.sum()/S.sum();z=(H-v*S)/S.sum();influences[stage]=z
    r=dict(events=int(take.sum()),source_pb=float(w[take].sum()),matched_pb=float(m[take].sum()),S_pb=float(S.sum()),H_pb=float(H.sum()),H_over_S=float(v),ratio_error=err(z),net_H_minus_S_pb=float((H-S).sum()),net_error_pb=err(H-S,True),matched_source_minus1=float(m[take].sum()/w[take].sum()-1))
    target=next(z for z in report['results'][stage]['ranges'] if z['lo']==5 and z['hi']==20)
    for key in ('S_pb','H_pb','H_over_S','ratio_error','net_H_minus_S_pb','net_error_pb','matched_source_minus1'):
        assert np.isclose(r[key],target[key],rtol=1.e-11,atol=1.e-12),(stage,key,r[key],target[key])
    assert r['events']==target['events']
    assert np.isclose(r['source_pb'],target['four_weights_pb']['source'])
    assert np.isclose(r['matched_pb'],target['four_weights_pb']['matched'])
    results[stage]=r
change=results['full']['H_over_S']-results['first']['H_over_S']
change_error=err(influences['full']-influences['first'])
target=next(z for z in report['paired_shower_drift'] if z['lo']==5 and z['hi']==20)
assert np.isclose(change,target['full_minus_first'],rtol=1.e-11,atol=1.e-12)
assert np.isclose(change_error,target['paired_error'],rtol=1.e-11,atol=1.e-12)
results.update(passed=True,events=n,jobs=ng,cuts='none',paired_ratio_drift=change,paired_ratio_drift_error=change_error,inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (f,g,c)})
out=B/'validation/independent_summary.json';out.write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
