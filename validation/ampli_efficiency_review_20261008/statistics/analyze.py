"""Read-only diagnostics from the two saved ttbar runs; no physics rerun."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read_header(path):
    with path.open() as stream:
        assert stream.readline().strip() == 'MG5_AMPLI_POOL 4'
        n, ncand, nepoch = map(int, stream.readline().split())
        mean_abs, mean_signed, m2_abs, m2_signed, cov = map(float, stream.readline().split())
        generation = list(map(float, stream.readline().split()))
        stream.readline()
        stream.readline()
        epochs = []
        for unused in range(nepoch):
            row = list(map(float, stream.readline().split()))
            epochs.append(dict(id=int(row[0]), trials=int(row[1]), nonzero=int(row[2]),
                mean_abs=row[4], mean_signed=row[5], m2_abs=row[6], m2_signed=row[7],
                covariance=row[8], cutoff=row[9], envelope=row[10], threshold=row[11],
                eligible=bool(row[12])))
    return dict(path=str(path.relative_to(ROOT)), trials=n, candidates=ncand,
        mean_abs=mean_abs, mean_signed=mean_signed, m2_abs=m2_abs, m2_signed=m2_signed,
        epochs=epochs, full_trial_tail=generation[3], reserve_tail=generation[4],
        worst_subset_tail=generation[5])


def summary(workers, field):
    n = np.array([w['trials'] for w in workers], dtype=float)
    mean = np.array([w['mean_'+field] for w in workers])
    err = np.array([math.sqrt(w['m2_'+field])/w['trials'] for w in workers])
    pooled = float(np.dot(n, mean)/sum(n))
    within_var = sum(w['m2_'+field] for w in workers)/sum(n)**2
    pooled_var = (sum(w['m2_'+field] for w in workers) + np.dot(n, (mean-pooled)**2))/sum(n)**2
    result = dict(mean=pooled, error_pooled=float(math.sqrt(pooled_var)),
        error_within_only=float(math.sqrt(within_var)), worker_count=len(workers))
    if len(workers) > 1:
        # Diagnostic only: workers stop adaptively and errors are plug-in.
        invvar = 1/err**2
        ivmean = float(np.dot(invvar, mean)/sum(invvar))
        q = float(sum(((mean-ivmean)/err)**2))
        cluster_var = len(workers)/(len(workers)-1)*float(sum((n*(mean-pooled))**2))/sum(n)**2
        result.update(scatter_chi2=q, scatter_dof=len(workers)-1,
            nominal_scatter_p=float(chi2.sf(q, len(workers)-1)),
            error_worker_cluster=math.sqrt(cluster_var),
            trial_count_mean_correlation=float(np.corrcoef(n,mean)[0,1]))
    for lo, hi, label in [(1,1,'first'),(2,3,'second_third'),(4,8,'fourth_eighth'),(9,1000,'after_eighth')]:
        epochs=[e for w in workers for e in w['epochs'] if lo<=e['id']<=hi]
        if not epochs: continue
        nn=sum(e['trials'] for e in epochs)
        mu=sum(e['trials']*e['mean_'+field] for e in epochs)/nn
        var=sum(e['m2_'+field]+e['trials']*(e['mean_'+field]-mu)**2 for e in epochs)/nn**2
        result[label]=dict(trials=nn,mean=mu,error=math.sqrt(var))
    return result


all_runs = {}
for name in ['ttbar_bounded_300k_20261007', 'ttbar_30k_jobs_300k_20261007']:
    folder=ROOT/'validation'/name
    metrics=json.loads((folder/'metrics.json').read_text())
    channels={}
    for channel in metrics['production_manifest']['channels']:
        key=channel['subprocess']+'/GF'+channel['channel']
        workers=[read_header(folder/'ampli'/b['directory']/'ampli_pool.dat') for b in channel['batches']]
        channels[key]=dict(signed=summary(workers,'signed'),absolute=summary(workers,'abs'),
            workers=workers, published={k:channel[k] for k in ['signed','absolute','error_signed','error_abs']},
            expired_trials=sum(e['trials'] for w in workers for e in w['epochs'] if not e['eligible']),
            trials=sum(w['trials'] for w in workers))
    all_runs[name]=channels

output=dict(note='Scatter p values are nominal diagnostics, not calibrated tests under adaptive stopping. '
    'Worker cluster errors require many independent workers, unlike the 1 or 5 workers per channel of the large-job run. '
    'Epoch group errors ignore dependence induced by adaptation and stopping. '
    'Rates below exclude survey unless explicitly labelled published.',runs=all_runs)
(HERE/'diagnostics.json').write_text(json.dumps(output,indent=2)+'\n')
for name,channels in all_runs.items():
    print(name)
    for key,c in channels.items():
        s=c['signed']
        print(key,'Nworkers',s['worker_count'],'mean',round(s['mean'],5),'error',round(s['error_pooled'],5),
              'cluster',round(s.get('error_worker_cluster',0),5),'scatterp',round(s.get('nominal_scatter_p',0),5),
              'rho_Nmean',round(s.get('trial_count_mean_correlation',0),4))
        print('epoch groups',[(g,round(s[g]['mean'],4),round(s[g]['error'],4)) for g in ['first','second_third','fourth_eighth','after_eighth'] if g in s])
