#!/usr/bin/env python3
"""Exact physical Born-gluon and massive-top native-component limit probes.

Runs deterministic matrix-element evaluations only, using a completed provider
and an external audit binary. Collinear gluon spin terms are retained both per
azimuth and after a four-azimuth average; no strict pointwise-power assertion is
made. The source assembly check uses the physical selected-flow fraction.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from native_tc_limits_20261101 import boost, build, rotate, run


def direction(axis, cosine, phi):
    reference = [0.,0.,1.] if abs(axis[2]) < .8 else [1.,0.,0.]
    u = [axis[1]*reference[2]-axis[2]*reference[1],
         axis[2]*reference[0]-axis[0]*reference[2],
         axis[0]*reference[1]-axis[1]*reference[0]]
    size = math.sqrt(sum(x*x for x in u));u=[x/size for x in u]
    v = [axis[1]*u[2]-axis[2]*u[1],axis[2]*u[0]-axis[0]*u[2],
         axis[0]*u[1]-axis[1]*u[0]]
    return [cosine*axis[i]+math.sqrt(max(0.,1-cosine*cosine))*
            (math.cos(phi)*u[i]+math.sin(phi)*v[i]) for i in range(3)]


def ee3(sequence, lam, phi, energy=91.188):
    xi = lam if sequence=='soft' else .2
    hardmass=energy*math.sqrt(1-xi);eg=energy*xi/2
    axes=[[1.,0.,0.],[-.5,math.sqrt(3)/2,0.],[-.5,-math.sqrt(3)/2,0.]]
    axis=axes[2] if sequence=='gluon_collinear' else axes[0]
    d=direction(axis,.23 if sequence=='soft' else 1-2*lam,phi)
    emitted=[eg]+[eg*x for x in d]
    hard=[boost([hardmass/3]+[hardmass/3*x for x in a],
                [-x/(energy-eg) for x in emitted[1:]]) for a in axes]
    return ([[energy/2,0.,0.,energy/2],[energy/2,0.,0.,-energy/2]]+
            list(map(rotate,hard+[emitted])))


def ttbar(sequence,lam,phi,energy=800.,mass=172.5):
    xi=lam if sequence=='soft' else .2
    eg=energy*xi/2;pairmass=energy*math.sqrt(1-xi)
    c=(1-2*lam if sequence=='beam1_collinear' else
       -1+2*lam if sequence=='beam2_collinear' else .23)
    d=[math.sqrt(1-c*c)*math.cos(.29),math.sqrt(1-c*c)*math.sin(.29),c]
    emitted=[eg]+[eg*x for x in d]
    topaxis=(direction(d,1-2*lam,phi) if sequence=='top_angular' else
             direction(d,-1+2*lam,phi) if sequence=='antitop_angular' else
             direction(d,.21,phi))
    momentum=math.sqrt(pairmass**2/4-mass**2)
    tops=[boost([pairmass/2]+[sign*momentum*x for x in topaxis],
                [-x/(energy-eg) for x in emitted[1:]]) for sign in (1,-1)]
    return [[energy/2,0.,0.,energy/2],[energy/2,0.,0.,-energy/2]]+tops+[emitted]


def physical_check(momenta,masses):
    scale=momenta[0][0]+momenta[1][0]
    residual=max(abs(sum(p[k] for p in momenta[2:])-momenta[0][k]-momenta[1][k])
                 for k in range(4))
    shell=max(abs(p[0]**2-sum(x*x for x in p[1:])-m*m)
              for p,m in zip(momenta,masses))
    if residual>1e-11*scale or shell>1e-11*scale*scale:
        raise AssertionError('Invalid independently constructed physical point')


def audit(directory,output,process,max_power=5):
    directory,output=directory.resolve(),output.resolve();output.mkdir(parents=True,exist_ok=True)
    binary=build(directory,output)
    if process=='ee3':
        sequences=('soft','quark_collinear','gluon_collinear');owners=(1,2,3)
        configs=(2,4);flows=(1,);phis=[i*math.pi/2 for i in range(4)]
        ids=(-11,11,1,-1,21,21);masses=[0.]*6
    else:
        topmass=None;in_mass=False
        for line in (directory/'param_card.dat').read_text().splitlines():
            tokens=line.split('#')[0].lower().split()
            if not tokens:continue
            if tokens[0]=='block':in_mass=tokens[1]=='mass';continue
            if in_mass and tokens[0]=='6':topmass=float(tokens[1].replace('d','e'));break
        if topmass is None:raise RuntimeError('Missing native top mass')
        sequences=('soft','beam1_collinear','beam2_collinear','top_angular','antitop_angular')
        owners=(1,2,3,4);configs=(1,);flows=(1,2);phis=[0.,math.pi/2]
        ids=(21,21,6,-6,21);masses=[0.,0.,topmass,topmass,0.]
    records=[];points=[]
    for sequence in sequences:
        for power in range(2,max_power+1):
            lam=10.**-power
            for phi in phis:
                momenta=ee3(sequence,lam,phi) if process=='ee3' else ttbar(sequence,lam,phi,mass=topmass)
                physical_check(momenta,masses)
                if process=='ee3':
                    observable=min(2*min(p[0]**2,q[0]**2)*max(0.,
                        1-sum(p[k]*q[k] for k in range(1,4))/(p[0]*q[0]))/91.188**2
                        for i,p in enumerate(momenta[2:]) for q in momenta[2:][i+1:])
                else:observable=math.hypot(momenta[-1][1],momenta[-1][2])
                for config in configs:
                    for flow in flows:
                        for owner in owners:
                            request=len(points)+1
                            points.append(dict(request=request,sequence=sequence,lambda_=lam,
                                phi=phi,owner=owner,config=config,flow=flow,observable=observable))
                            records.append('MACNLOPS_POINT_V3 %d %d %d\n'%(request,owner,len(ids))+
                                ''.join('%d %s\n'%(pdg,' '.join('%.17g'%x for x in p))
                                        for pdg,p in zip(ids,momenta))+
                                'COMPONENT_LOCAL_V1 %d %d %d 4 1\nEND_POINT\n'%(config,config,flow))
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
    if any(len(x)!=len(points) for x in (densities,profiles,sources)):
        raise RuntimeError('Missing native points; inspect saved native logs')
    for row,density,profile,source in zip(points,densities,profiles,sources):
        request,status,r,d,k=density
        if status or request!=row['request']:raise RuntimeError('Invalid native point '+str(row))
        ps,tc,a,cdf,xi,y,sij,hij=profile;prob,sourceh=source;h=r-d
        row.update(status=status,profile_status=ps,r=r,d=d,k=k,h0=h,tc=tc,A=a,cdf=cdf,
            xi=xi,y=y,fks_sij=sij,fks_hij=hij,source_probability=prob,source_h=sourceh,
            source_factorization_error=abs(sourceh-prob*h)/max(abs(r),abs(d),1.),
            h0_log_shell=h*row['lambda_']**(2 if row['sequence']=='soft' else 1))
        row['source_log_shell']=row['h0_log_shell']*a
        row['irsafe_h0_log_shell']=row['h0_log_shell']*row['observable']
    averages=[]
    for sequence in sequences:
        for owner in owners:
            for config in configs:
                for flow in flows:
                    rows=[]
                    for power in range(2,max_power+1):
                        group=[r for r in points if (r['sequence'],r['owner'],r['config'],r['flow'],r['lambda_'])==
                               (sequence,owner,config,flow,10.**-power)]
                        rows.append(dict(lambda_=10.**-power,
                            h0=sum(r['h0'] for r in group)/len(group),
                            h0_log_shell=sum(r['h0_log_shell'] for r in group)/len(group),
                            max_abs_h0_log_shell=max(abs(r['h0_log_shell']) for r in group),
                            tc_min=min(r['tc'] for r in group),tc_max=max(r['tc'] for r in group),
                            A_min=min(r['A'] for r in group),A_max=max(r['A'] for r in group)))
                    a,b=rows[-2:]
                    averages.append(dict(sequence=sequence,owner=owner,config=config,flow=flow,
                        points=rows,tail_average_shell_ratio=abs(b['h0_log_shell']/a['h0_log_shell']) if a['h0_log_shell'] else None))
    final=dict(process=process,directory=str(directory),points=points,azimuth_averages=averages,
        external_masses=masses,observable='Durham y34' if process=='ee3' else 'ttbar recoil pT [GeV]',
        max_source_factorization_error=max(r['source_factorization_error'] for r in points),
        nonzero_profile_statuses=sorted(set(r['profile_status'] for r in points if r['profile_status'])),
        interpretation='Soft measure lambda d(lambda), massless collinear d(lambda). Massive angular scans have no physical collinear singularity. Gluon spin terms are diagnosed separately per azimuth and after averaging; no strict generic power failure is applied.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (Path(__file__).resolve(),directory/'driver_macnlops.f',directory/'montecarlocounter.f')})
    (output/'limits.json').write_text(json.dumps(final,indent=2)+'\n')
    return {k:final[k] for k in ('process','max_source_factorization_error','nonzero_profile_statuses')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('subprocess_dir',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--process',choices=('ee3','ttbar'),required=True)
    p.add_argument('--max-power',type=int,default=5);a=p.parse_args()
    print(json.dumps(audit(a.subprocess_dir,a.output,a.process,a.max_power),indent=2))
