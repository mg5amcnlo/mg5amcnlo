"""Offline final-pool capacity at fixed saved envelopes and eligible epochs.

This is a final-state diagnostic, not a replay of an alternate earlier stop:
the saved envelopes include information acquired throughout production.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from analyze import read_header

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from madgraph.various import ampli_pool


def capacity(path):
    header=read_header(path)
    epochs=header['epochs']
    with path.open() as f:
        for _ in range(3): f.readline()
        gen=list(map(float,f.readline().split()))
    quota,final=int(gen[0]),int(gen[1])
    logz=gen[2]
    candidates=np.loadtxt(path,skiprows=6+len(epochs),ndmin=2)
    ids=candidates[:,0].astype(int)-1
    eligible=np.array([e['eligible'] for e in epochs])[ids]
    candidates=candidates[eligible]
    ids=ids[eligible]
    envelope=np.array([e['envelope'] for e in epochs])[ids]
    logratio=np.log(candidates[:,1])-np.log(envelope)
    rank=candidates[:,2]-np.log(envelope)
    ranks=np.sort(rank)[::-1]
    floor=max(math.log(e['cutoff']/e['envelope']) for e in epochs if e['eligible'])
    total=sum(e['trials']*e['mean_abs'] for e in epochs if e['eligible'])
    factors=candidates[:,5]
    ceiling=int(np.count_nonzero(rank>floor))

    def status(z,k=None):
        select=rank>z
        # A first-excluded threshold can coincide within rounding with a rank.
        # Explicit top-k labels reproduce native selection at that boundary.
        if k is not None:
            inds=np.argpartition(rank,-k)[-k:]
            select=np.zeros(len(rank),dtype=bool)
            select[inds]=True
        tail=logratio>z
        assert not np.any(tail & ~select)
        ntail=int(np.count_nonzero(tail))
        tailmass=float(np.sum(np.exp(logratio[tail]-z)*factors[tail]))
        normal=factors[select & ~tail]
        full=float(np.sum(candidates[tail,1])/total)
        reserve=tailmass/(tailmass+float(sum(normal)))
        if ntail>=final:
            worst=1.
        elif ntail==0:
            worst=0.
        else:
            need=final-ntail
            assert need<=len(normal)
            ordinary=float(np.sum(np.partition(normal,need-1)[:need]))
            worst=tailmass/(tailmass+ordinary)
        return dict(available=int(np.count_nonzero(select)),full_trial_tail=full,
            reserve_tail=reserve,worst_subset_tail=worst,tail_events=ntail,
            log_z=float(z),safe=max(full,reserve,worst)<.01)

    initial=status(logz,quota)
    assert initial['safe']
    assert np.allclose([initial[k] for k in ['full_trial_tail','reserve_tail','worst_subset_tail']],gen[3:],rtol=1e-8,atol=1e-12)
    def ranked(k):
        z=max(float(ranks[k]),floor) if k<len(ranks) else floor
        return status(z,k)
    floor_status=status(floor)
    lower,upper=quota,ceiling
    while lower<upper:
        middle=(lower+upper+1)//2
        if ranked(middle)['safe']:
            lower=middle
        else:
            upper=middle-1
    best=ranked(lower)
    assert best['safe'] and lower>=quota
    next_status=ranked(lower+1) if lower<ceiling else None
    assert next_status is None or not next_status['safe']
    return dict(worker=path.parent.name,channel=path.parent.name.split('_')[0],
        subprocess=path.parent.parent.name,quota=quota,final_quota=final,
        trials=header['trials'],candidate_count=header['candidates'],active_candidate_count=len(rank),
        saved=initial,maximum_native_rank_capacity=best,
        extra_events=lower-quota,extra_fraction_of_requested_reserve=(lower-quota)/quota,
        storage_floor_capacity=floor_status,first_failing_rank=next_status,
        limit='storage_floor' if lower==ceiling else 'tail',
        factors_min=float(min(factors)),factors_max=float(max(factors)))


def independent_verify(path, result):
    """Replay the changed threshold with the production Python validator."""
    pool=ampli_pool.read_pool(str(path.parent))
    best=result['maximum_native_rank_capacity']
    pool['generated_target']=best['available']
    pool['log_z']=best['log_z']
    ranks=[]
    for i,(epoch_id,row) in enumerate(zip(pool['candidate_epochs'],pool['candidates'])):
        epoch=pool['epochs'][epoch_id-1]
        if epoch['eligible']:
            ranks.append((row[1]-math.log(epoch['envelope']),i))
    selected=set(i for _,i in sorted(ranks,reverse=True)[:best['available']])
    for epoch in pool['epochs']:
        epoch['threshold']=max(epoch['cutoff'],math.exp(math.log(epoch['envelope'])+best['log_z'])) if epoch['eligible'] else 0.
    logs={i:max(0.,math.log(pool['candidates'][i][0])-math.log(pool['epochs'][pool['candidate_epochs'][i]-1]['threshold'])) for i in selected}
    maximum=max(logs.values())
    norm=best['available']/math.fsum(math.exp(v-maximum) for v in logs.values())
    rows=[]
    for i,(epoch_id,row) in enumerate(zip(pool['candidate_epochs'],pool['candidates'])):
        epoch=pool['epochs'][epoch_id-1]
        weight,priority,_,_,factor=row
        tail=int(epoch['eligible'] and weight>epoch['threshold'])
        correction=norm*math.exp(logs[i]-maximum) if i in selected else 0.
        rows.append((weight,priority,correction,tail,factor))
    pool['candidates']=rows
    for key in ['full_trial_tail','reserve_tail','worst_subset_tail']:
        pool[key]=best[key]
    retained,status=ampli_pool._native_worker_status(pool)
    assert len(retained)==best['available'] and max(status.values())<.01


if __name__=='__main__':
    runs={}
    for run in ['ttbar_bounded_300k_20261007','ttbar_30k_jobs_300k_20261007']:
        workers=[]
        for path in sorted((ROOT/'validation'/run/'ampli').glob('SubProcesses/P*/GF*/ampli_pool.dat')):
            result=capacity(path)
            independent_verify(path,result)
            result['production_validator_replay_passed']=True
            workers.append(result)
        summary={k:sum(w[k] for w in workers) for k in ['quota','final_quota','trials','candidate_count','active_candidate_count','extra_events']}
        summary['maximum_native_rank_capacity']=sum(w['maximum_native_rank_capacity']['available'] for w in workers)
        summary['workers_with_surplus']=sum(w['extra_events']>0 for w in workers)
        summary['worker_count']=len(workers)
        summary['extra_fraction_of_reserve']=summary['extra_events']/summary['quota']
        summary['storage_limited_workers']=sum(w['limit']=='storage_floor' for w in workers)
        summary['per_channel']={}
        for w in workers:
            key=w['subprocess']+'/'+w['channel']
            row=summary['per_channel'].setdefault(key,dict(reserve=0,extra=0,workers=0))
            row['reserve']+=w['quota'];row['extra']+=w['extra_events'];row['workers']+=1
        runs[run]=dict(summary=summary,workers=workers)
        print(run,json.dumps(summary,indent=2))
    (HERE/'surplus.json').write_text(json.dumps(dict(note='Final-state capacity only. Uses saved final envelopes, final eligible epochs and existing candidates. '
        'Additional accepted events require a lower common normalized threshold and recomputed correction factors. '
        'Native first-excluded-rank convention, storage floor, strict 1% full-trial/reserve/worst-final-subset criteria, '
        'and original final quota are preserved. This does not quantify recoverable trials or an earlier stopping time.',runs=runs),indent=2)+'\n')
