#!/usr/bin/env python3
"""Focused native EE4, Wj and massive-Wbb component/source infrared probes.

Uses exact independent hard-rest-frame states with one emitted gluon and a
global recoil boost. Builds only an external audit executable against completed
providers. Component configurations/flows are taken from the actual pilot.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

from native_tc_limits_20261101 import boost,build,run
from native_tc_multileg_limits_20261101 import direction,physical_check


def masses(directory):
    result={};active=False
    for line in (directory/'param_card.dat').read_text().splitlines():
        f=line.split('#')[0].lower().split()
        if not f:continue
        if f[0]=='block':active=f[1]=='mass';continue
        if f[0]=='decay':active=False
        if active and len(f)>1:result[int(f[0])]=float(f[1].replace('d','e'))
    return result


def routes(directory,incoming):
    text=(directory/'macnlops_routes.dat').read_text().splitlines();registry=text[0].split()[1]
    sectors=[]
    for i,line in enumerate(text):
        f=line.split()
        if not f or f[0]!='SECTOR':continue
        owner,emitted,emitter=int(f[1]),int(f[2])-1,int(f[3])-1
        rows=[list(map(int,x.split())) for x in text[i+1:i+1+int(f[6])]]
        wanted=[row for row in rows if tuple(row[:2])==incoming and row[emitted]==21]
        if wanted:sectors.append(dict(owner=owner,emitted=emitted,emitter=emitter,ids=wanted[0]))
    return registry,sectors


def components(directory,registry):
    pilot=directory.parent.parent/'Events/run_pilot_native_tc/events.lhe.gz'
    with gzip.open(pilot,'rt') as stream:text=stream.read()
    found=Counter()
    for header,contents in re.findall(r'<macnlops_component\b([^>]*)>(.*?)</macnlops_component>',text,re.S):
        if registry not in header:continue
        first=contents.split();found[(int(first[1]),int(first[2]),int(first[3]))]+=1
    if not found:raise RuntimeError('No native pilot components for '+str(directory))
    configs=[]
    for config,graph,flow in found:
        if (config,graph) not in configs:configs.append((config,graph))
    configs.sort(key=lambda pair:-sum(n for (c,g,f),n in found.items() if (c,g)==pair))
    selected=set(configs[:2])
    return [(c,g,f) for c,g,f in found if (c,g) in selected],str(pilot)


def point(case,sector,sequence,lam,phi,card):
    ids=sector['ids'];emitted=sector['emitted'];hardlegs=[i for i in range(2,len(ids)) if i!=emitted]
    energy=91.188 if case=='ee4' else 300.;xi=lam if sequence=='soft' else .15
    restmass=energy*math.sqrt(1-xi);eg=energy*xi/2
    m=[card.get(abs(ids[i]),0.) for i in hardlegs]
    if case=='ee4':
        z=1/math.sqrt(3);axes=[[z,z,z],[z,-z,-z],[-z,z,-z],[-z,-z,z]]
        hard=[[restmass/4]+[restmass/4*x for x in axis] for axis in axes]
    elif case.startswith('wj'):
        a=[math.sin(.73)*math.cos(.31),math.sin(.73)*math.sin(.31),math.cos(.73)]
        axes=[a,[-x for x in a]];p=(restmass**2-card[24]**2)/(2*restmass)
        hard=[[math.sqrt(p*p+mass*mass)]+[p*x for x in axis] for mass,axis in zip(m,axes)]
    else:
        axes=[[1.,0.,0.],[-.5,math.sqrt(3)/2,0.],[-.5,-math.sqrt(3)/2,0.]]
        lo,hi=0.,restmass/2
        for _ in range(60):
            p=(lo+hi)/2
            if sum(math.sqrt(p*p+x*x) for x in m)>restmass:hi=p
            else:lo=p
        p=(lo+hi)/2;hard=[[math.sqrt(p*p+mass*mass)]+[p*x for x in axis] for mass,axis in zip(m,axes)]
    if sequence.startswith('beam'):
        c=1-2*lam if sequence=='beam1_collinear' else -1+2*lam
        d=[math.sqrt(1-c*c)*math.cos(phi),math.sqrt(1-c*c)*math.sin(phi),c]
    else:
        qcd=[j for j,i in enumerate(hardlegs) if ids[i]!=24]
        gluons=[j for j,i in enumerate(hardlegs) if ids[i]==21]
        target=(gluons[0] if sequence=='gluon1_collinear' else gluons[-1]
                if sequence=='gluon2_collinear' else qcd[-1]
                if sequence=='bbar_angular' else qcd[0])
        d=direction(axes[target],.23 if sequence=='soft' else 1-2*lam,phi)
    radiation=[eg]+[eg*x for x in d]
    hard=[boost(p,[-x/(energy-eg) for x in radiation[1:]]) for p in hard]
    momenta=[[energy/2,0.,0.,energy/2],[energy/2,0.,0.,-energy/2]]+[[0.]*4 for _ in hardlegs]+[[0.]*4]
    for leg,p in zip(hardlegs,hard):momenta[leg]=p
    momenta[emitted]=radiation
    external=[0.,0.]+[0. if i==emitted else card.get(abs(ids[i]),0.) for i in range(2,len(ids))]
    physical_check(momenta,external)
    return momenta,external


def audit(directory,output,case,max_power=5):
    directory,output=directory.resolve(),output.resolve();output.mkdir(parents=True,exist_ok=True)
    incoming=(-11,11) if case=='ee4' else (2,21) if case=='wj_qg' else (2,-1)
    registry,sectors=routes(directory,incoming);selected,pilot=components(directory,registry)
    if not sectors:raise RuntimeError('No requested soft-gluon real channel')
    sequences=(('soft','quark_collinear','gluon1_collinear','gluon2_collinear') if case=='ee4' else
               ('soft','beam1_collinear','beam2_collinear','final_collinear') if case.startswith('wj') else
               ('soft','beam1_collinear','beam2_collinear','b_angular','bbar_angular'))
    card=masses(directory);binary=build(directory,output);points=[];records=[]
    for sequence in sequences:
        for power in range(2,max_power+1):
            for phi in (0.,math.pi/2):
                for sector in sectors:
                    ps,external=point(case,sector,sequence,10.**-power,phi,card)
                    for config,graph,flow in selected:
                        request=len(points)+1
                        points.append(dict(request=request,sequence=sequence,lambda_=10.**-power,phi=phi,
                            owner=sector['owner'],emitter_leg=sector['emitter']+1,
                            emitted_leg=sector['emitted']+1,ids=sector['ids'],config=config,graph=graph,flow=flow))
                        records.append('MACNLOPS_POINT_V3 %d %d %d\n'%(request,sector['owner'],len(ps))+
                            ''.join('%d %s\n'%(pdg,' '.join('%.17g'%x for x in p)) for pdg,p in zip(sector['ids'],ps))+
                            'COMPONENT_LOCAL_V1 %d %d %d 4 1\nEND_POINT\n'%(config,graph,flow))
    text=''.join(records);(output/'points.txt').write_text(text)
    result=run(['./native_tc_limits_audit'],directory,executable=str(binary),input=text,timeout=300)
    (output/'native_stdout.log').write_text(result.stdout);(output/'native_stderr.log').write_text(result.stderr)
    densities=[];profiles=[];sources=[]
    for line in result.stdout.splitlines():
        f=line.split()
        if not f:continue
        if f[0]=='MACNLOPS_DENSITY_V1':densities.append((int(f[1]),int(f[2]),*map(float,f[3:])))
        if f[0]=='NATIVE_TC_LIMIT_V1':profiles.append((int(f[2]),*map(float,f[3:])))
        if f[0]=='NATIVE_TC_SOURCE_V1':sources.append(tuple(map(float,f[1:])))
    if any(len(x)!=len(points) for x in (densities,profiles,sources)):raise RuntimeError('Incomplete native points; inspect saved stdout')
    for row,density,profile,source in zip(points,densities,profiles,sources):
        request,status,r,d,k=density
        if status or request!=row['request']:raise RuntimeError('Invalid native point '+str(row))
        ps,tc,a,cdf,xi,y,sij,hij=profile;prob,sourceh=source;h=r-d
        row.update(status=status,profile_status=ps,r=r,d=d,k=k,h0=h,tc=tc,A=a,cdf=cdf,xi=xi,y=y,
            fks_sij=sij,fks_hij=hij,source_probability=prob,source_h=sourceh,
            source_factorization_error=abs(sourceh-prob*h)/max(abs(r),abs(d),1.),
            h0_log_shell=h*row['lambda_']**(2 if row['sequence']=='soft' else 1))
        row['source_log_shell']=a*row['h0_log_shell']
    tails=[]
    for sequence in sequences:
        for sector in sectors:
            for config,graph,flow in selected:
                group=[r for r in points if (r['sequence'],r['owner'],r['config'],r['flow'])==(sequence,sector['owner'],config,flow)]
                previous=[r for r in group if r['lambda_']==10.**-(max_power-1)]
                last=[r for r in group if r['lambda_']==10.**-max_power]
                a=sum(r['h0_log_shell'] for r in previous)/len(previous);b=sum(r['h0_log_shell'] for r in last)/len(last)
                ma=max(abs(r['h0_log_shell']) for r in previous);mb=max(abs(r['h0_log_shell']) for r in last)
                tails.append(dict(sequence=sequence,owner=sector['owner'],emitter_leg=sector['emitter']+1,config=config,flow=flow,
                    average_shell_ratio=abs(b/a) if a else None,max_abs_pointwise_shell_ratio=mb/ma if ma else None,
                    last_tc_range=[min(r['tc'] for r in last),max(r['tc'] for r in last)],last_A_range=[min(r['A'] for r in last),max(r['A'] for r in last)]))
    report=dict(case=case,directory=str(directory),pilot_components=pilot,selected_components=selected,
        actual_mass_card={str(k):card[k] for k in (5,24) if k in card},points=points,tails=tails,
        max_source_factorization_error=max(r['source_factorization_error'] for r in points),
        nonzero_profile_statuses=sorted(set(r['profile_status'] for r in points if r['profile_status'])),
        scope='Finite-grid diagnostics, not a proof. Both sampled azimuths retained, with max absolute pointwise and averaged shells. Massive angular limits are not assigned massless-collinear power assertions.',
        fingerprints={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__).resolve(),directory/'driver_macnlops.f',directory/'montecarlocounter.f',directory/'param_card.dat')})
    (output/'limits.json').write_text(json.dumps(report,indent=2)+'\n')
    return dict(case=case,points=len(points),max_source_factorization_error=report['max_source_factorization_error'],nonzero_profile_statuses=report['nonzero_profile_statuses'],max_pointwise_tail_ratio=max(r['max_abs_pointwise_shell_ratio'] for r in tails))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('subprocess_dir',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',choices=('ee4','wj_qqbar','wj_qg','wbb'),required=True)
    p.add_argument('--max-power',type=int,default=5);a=p.parse_args()
    print(json.dumps(audit(a.subprocess_dir,a.output,a.case,a.max_power),indent=2))
