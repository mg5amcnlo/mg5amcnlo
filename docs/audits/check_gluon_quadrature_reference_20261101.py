#!/usr/bin/env python3
"""Independent high-precision global-gluon reference and old/new rate parity.

The reference integrates x=|2z-1|, with s=(1-x*x)/4, rather than the production
logit coordinate. The two colour-end orientations are summed analytically.
Physical thresholds and the native 1e-12 floors are explicit integration
breakpoints. No shower, event generation, or production edits are performed.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess

import mpmath as mp

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/export/slow1/rikkert/macnlops_native_tc_1M_20261101')
LAMBDAS=('.3249942945718149','.28756663468675603','.22513027856717657')
CAPTURED_T=265.94255734488678
CAPTURED=(80.419002445775163,2679.8203470176318,28.056336658409535,0.,math.sqrt(CAPTURED_T),5,0.,0.,1)


def reference(point,dps=80,floors=True):
    """Return one rate, including active ends and the native one-loop coupling."""
    with mp.workdps(dps):
        rec,M,cap1,cap2,q,nfl,charm,bottom,_=point
        rec,M,q=map(lambda x:mp.mpf(str(x)),(rec,M,q));t=q*q
        ends=int(q<cap1)+int(q<cap2)
        if q<mp.mpf('.5') or not ends or M<=rec or q>=(M-rec)/2:return mp.mpf(0)
        r=(rec/M)**2;u=1-r;tau=t/M**2;s0=(mp.sqrt(tau)-tau)/u
        smin=max(tau/(1-mp.sqrt(r))**2,s0);xmax=mp.sqrt(1-4*smin)
        threshold=mp.mpf('1e-12');breaks=[mp.mpf(0),xmax]
        daughter=[mp.mpf(0)]*int(nfl)
        if nfl>=4:daughter[3]=mp.mpf(str(charm))
        if nfl>=5:daughter[4]=mp.mpf(str(bottom))
        for mass in daughter:
            if mass>q:
                x=mp.sqrt(1-t/(mass*mass))
                if x<xmax:breaks.append(x)
        if floors and rec:
            zfloor=(threshold+mp.sqrt(threshold**2+4*u*tau))/(2*u)
            x=1-2*zfloor
            if 0<x<xmax:breaks.append(x)
            sfloor=tau/threshold
            if smin<sfloor<mp.mpf('.25'):breaks.append(mp.sqrt(1-4*sfloor))
        def integrand(x):
            s=(1-x*x)/4
            if rec:
                z=(1-x)/2;omz=(1+x)/2;a=tau/s
                d1=u*z-tau/z;d2=u*omz-tau/omz
                if floors and min(d1,d2,a)<threshold:
                    dc=max(0,1-r*max(threshold,a)/(max(threshold,d1)*max(threshold,d2)))
                else:
                    numerator=u*(s-s0)*(u*s+tau+mp.sqrt(tau))
                    dc=max(0,numerator/(numerator+r*tau))
            else:dc=1
            flavours=sum(mp.sqrt(max(0,1-4*mass*mass*s/t)) for mass in daughter)
            return mp.mpf('1.5')*(1-s)**2/s*dc+flavours/4*(1-2*s)
        integral=mp.quad(integrand,sorted(set(breaks)))
        nf=5 if q>max(4.,bottom) else 4 if q>max(1.2,charm) else 3
        alpha=1/((11-mp.mpf(2)*nf/3)/2*mp.log(t/mp.mpf(LAMBDAS[nf-3])**2))
        return +ends*alpha*integral


def stress_points():
    rec,M,_,_,q,*_=CAPTURED
    return [CAPTURED]+[(rec,M,28.056336658409535,0.,q*f,5,0.,0.,1) for f in (.999999,1.000001)]+[
        (rec,5000.,6000.,0.,.51,5,0.,0.,1),
        (rec,13000.,14000.,0.,.51,5,0.,0.,1),
        (rec,13000.,14000.,0.,16.3,5,0.,0.,1)]+[
        (rec,M,M,0.,(M-rec)/2*f,5,0.,0.,1) for f in (.999999,.9999999999)]+[
        (1.3,13000.,14000.,0.,.51,5,0.,0.,1),
        (80.,125.,30.,0.,.6,5,2.,6.,1)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'diagnostics/gluon_quadrature/independent_reference.json')
    parser.add_argument('--original',type=Path,default=BASE/'diagnostics/gluon_quadrature/original/MacNLOPSMatch.cc')
    parser.add_argument('--random-points',type=int,default=1000);args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('gluon_rate_fixture',ROOT/'tests/unit_tests/various/test_macnlops_gluon_sudakov.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fingerprints={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                  (Path(__file__).resolve(),ROOT/'tests/unit_tests/various/test_macnlops_gluon_sudakov.py',
                   module.TestGlobalGluonRate.source_path,args.original)}
    class Current(module.TestGlobalGluonRate):pass
    class Original(module.TestGlobalGluonRate):source_path=args.original
    Current.setUpClass();Original.setUpClass();current=Current();original=Original()
    records=[]
    try:
        for point in stress_points():
            low=reference(point,60);high=reference(point,90)
            try:value=current.rates([point])[0];stable_error=None
            except subprocess.CalledProcessError as error:
                value=None;stable_error=error.stderr.strip()[-200:]
            try:old=original.rates([point])[0];old_error=None
            except subprocess.CalledProcessError as error:
                old=None;old_error=error.stderr.strip()[-200:]
            records.append(dict(point=point,reference_90_digits=mp.nstr(high,75),
                reference_60_vs_90=float(abs(high-low)),stable=value,
                absolute_error=float(abs(mp.mpf(value)-high)) if value is not None else None,
                floor_effect=float(high-reference(point,90,False)),stable_error=stable_error,old=old,old_error=old_error))
        rng=random.Random(20261101);grid=[]
        for index in range(args.random_points):
            M=91.188 if index<args.random_points//2 else math.exp(rng.uniform(math.log(100),math.log(13000)))
            rec=M*rng.uniform(0.,.95)
            q=math.exp(rng.uniform(math.log(.51),math.log(max(.5101,min(100.,.49*(M-rec))))))
            massive=index%7==0;charm,bottom=(2.,6.) if massive else (0.,0.)
            point=(rec,M,M,0. if index%2 else M,q,(0,4,5)[index%3],charm,bottom,1)
            new=current.rates([point])[0]
            try:old=original.rates([point])[0]
            except subprocess.CalledProcessError as error:
                grid.append(dict(point=point,old_converged=False,stable=new,old_error=error.stderr.strip()[-200:]));continue
            grid.append(dict(point=point,old_converged=True,old=old,stable=new,
                absolute_difference=abs(old-new),relative_difference=abs(old-new)/max(abs(old),1e-300)))
        good=[r for r in grid if r['old_converged']]
        result=dict(method='Independent 60/90-digit x=|2z−1| integration; production uses adaptive Gauss in logit(z).',
            references=records,parity_points=len(grid),old_converged=len(good),old_failed=len(grid)-len(good),
            max_parity_absolute=max(r['absolute_difference'] for r in good),max_parity_relative=max(r['relative_difference'] for r in good),
            worst_parity=sorted(good,key=lambda r:r['absolute_difference'],reverse=True)[:10],
            old_failures=[r for r in grid if not r['old_converged']],random_seed=20261101,
            fingerprints=fingerprints,
            production_tolerance='Unchanged: 1e-10/panels, 12/24 Gauss, maximum depth18',
            physics_generation_or_shower=False)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:Current.doClassCleanups();Original.doClassCleanups()


if __name__=='__main__':main()
