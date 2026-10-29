#!/usr/bin/env python3
"""Accuracy/cost study for fixed-quadrature light-quark PDF boundary factors.

Usage: dy_boundary_fast.py MATCHED_STUDY WORK PYTHIA_INSTALL OUTPUT
No production routine or shower is changed.
"""
import argparse
import csv
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.integrate import quad


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for key in ['study','work','pythia','output']:parser.add_argument(key,type=Path)
    a=parser.parse_args();a.work.mkdir(parents=True,exist_ok=True);a.output.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_suffix('.cc');so=a.work/'dy_boundary_fast.so'
    build=['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-I'+str(a.pythia/'include'),
           '-L'+str(a.pythia/'lib'),'-Wl,-rpath,'+str(a.pythia/'lib'),'-lpythia8','-o',str(so)]
    subprocess.run(build,check=True)
    lib=C.CDLL(str(so));lib.boundary_init.argtypes=[C.c_char_p]
    lib.boundary_xf.argtypes=[C.c_int,C.c_double,C.c_double];lib.boundary_xf.restype=C.c_double
    lib.boundary_eval.argtypes=[C.c_int,C.c_double,C.c_double,C.c_double,C.c_double,C.c_int,
                               C.POINTER(C.c_double),C.POINTER(C.c_double)]
    lib.boundary_eval.restype=C.c_double
    dptr=C.POINTER(C.c_double)
    lib.boundary_benchmark.argtypes=[C.c_int,C.POINTER(C.c_int),dptr,dptr,dptr,dptr,
        C.c_int,dptr,dptr,C.c_int,dptr]
    lib.boundary_benchmark.restype=C.c_double
    assert lib.boundary_init(str(a.pythia/'share/Pythia8/xmldoc').encode())==0
    nodes={n:tuple(np.ascontiguousarray(v) for v in np.polynomial.legendre.leggauss(n)) for n in [2,3,4,6]}
    def fast(c,n):
        x,w=nodes[n]
        return lib.boundary_eval(c['pid'],c['x'],c['q'],c['Q'],c['M'],n,
            x.ctypes.data_as(C.POINTER(C.c_double)),w.ctypes.data_as(C.POINTER(C.c_double)))
    cf=4./3.;beta=23./(12.*np.pi);lam2=91.188**2*np.exp(-1./(beta*.130))
    def exact(c):
        def outer(lt):
            t=np.exp(lt);r=t/c['M']**2;Z=1.-.5*r*(np.sqrt(1.+4./r)-1.)
            den=lib.boundary_xf(c['pid'],c['x'],t)
            assert den>0
            def inner(u):
                z=1.-(1.-Z)*u;y=c['x']/z
                fq=lib.boundary_xf(c['pid'],y,t);fg=lib.boundary_xf(21,y,t)
                return (1.-Z)*(cf*(1.+z*z)/(1.-z)*(fq/den-1.)
                      +.5*(z*z+(1.-z)**2)*fg/den)
            # At x/Z>1, the support boundary lies within this interval.
            point=(1-c['x'])/(1-Z)
            val=quad(inner,0,1,points=[point] if 0<point<1 else None,epsabs=2e-7,limit=150)[0]
            return (cf*Z*(1.-Z)+val)/(2*np.pi*beta*np.log(t/lam2))
        return quad(outer,np.log(c['q']**2),np.log(c['Q']**2),epsabs=2e-6,limit=150)
    cases=[];M=91.1876
    for q in [5.,10.,20.,50.]:
        for pid in [2,-2]:cases.append(dict(label=f'central_{pid}_{q:g}',kind='central',pid=pid,x=M/13000,q=q,Q=M,M=M))
    fixed=[json.loads(l) for l in (a.study/'pdf13/dy_fixed.born.jsonl').open()]
    damped=[json.loads(l) for l in (a.study/'pdf13/dy_damped.born.jsonl').open()]
    # Five x quantiles for each signed light flavour; no heavy-quark approximation.
    for pid in [1,-1,2,-2,3,-3]:
        choices=sorted([(b['x'][leg],b['event'],leg) for b in fixed for leg in [0,1] if b['ids'][leg]==pid])
        for quantile in np.linspace(0,1,5):
            x,event,leg=choices[round(quantile*(len(choices)-1))];b=fixed[event]
            mass=np.sqrt(4*b['p'][0][0]*b['p'][1][0])
            cases.append(dict(label=f'fixed_{pid}_{quantile:g}',kind='representative',pid=pid,x=x,q=20.,Q=b['scalup'],M=mass,event=event,leg=leg))
        for quantile in [.25,.75]:
            x,event,leg=choices[round(quantile*(len(choices)-1))];b=damped[event]
            if b['scalup']<=20:continue
            mass=np.sqrt(4*b['p'][0][0]*b['p'][1][0])
            cases.append(dict(label=f'damped_{pid}_{quantile:g}',kind='representative',pid=pid,x=x,q=20.,Q=b['scalup'],M=mass,event=event,leg=leg))
    rows=[];started=time.perf_counter()
    for i,c in enumerate(cases):
        value,error=exact(c)
        for n in nodes:
            approx=fast(c,n);rows.append(dict(**c,n=n,PDF_calls_max=n*(1+2*n),
                exact_exponent=value,quad_estimated_error=error,fast_exponent=approx,
                exponent_error=approx-value,factor_relative_error=float(np.expm1(approx-value))))
        print(f'{i+1}/{len(cases)} {c["label"]}: C={value:.8f}',flush=True)
    exact_seconds=time.perf_counter()-started
    benchmarks=[]
    pid_values=np.ascontiguousarray([c['pid'] for c in cases],dtype=np.int32)
    coordinate_values=[np.ascontiguousarray([c[k] for c in cases]) for k in ['x','q','Q','M']]
    # One ctypes entry per correction; all PDF calls and loops execute in C++.
    for n in nodes:
        count=20000;t0=time.perf_counter();checksum=0.
        for k in range(count):checksum+=fast(cases[k%len(cases)],n)
        elapsed=time.perf_counter()-t0
        benchmarks.append(dict(n=n,evaluations=count,seconds=elapsed,
            microseconds_per_leg=1e6*elapsed/count,checksum=checksum,
            note='Includes Python/ctypes entry; node and PDF loops are compiled C++. Built-in Pythia PDF13 only.'))
        xx,ww=nodes[n];checksum_native=C.c_double();native_count=200000
        native_seconds=lib.boundary_benchmark(len(cases),pid_values.ctypes.data_as(C.POINTER(C.c_int)),
            *[v.ctypes.data_as(dptr) for v in coordinate_values],n,
            xx.ctypes.data_as(dptr),ww.ctypes.data_as(dptr),native_count,C.byref(checksum_native))
        benchmarks[-1].update(native_evaluations=native_count,native_seconds=native_seconds,
            native_microseconds_per_leg=1e6*native_seconds/native_count,
            native_checksum=checksum_native.value)
    summaries=[]
    for n in nodes:
        selected=[r for r in rows if r['n']==n]
        errors=np.array([r['factor_relative_error'] for r in selected])
        central=np.array([r['factor_relative_error'] for r in selected if r['kind']=='central'])
        summaries.append(dict(n=n,PDF_calls_max=n*(1+2*n),max_abs_factor_relative_error=float(abs(errors).max()),
            p95_abs_factor_relative_error=float(np.percentile(abs(errors),95)),
            median_abs_factor_relative_error=float(np.median(abs(errors))),
            central_max_abs_factor_relative_error=float(abs(central).max())))
    # A physics diagnostic on every massless light-flavour Born state. Apply
    # exp(C) to each raw leg factor, then apply the existing probability bound.
    table_path=a.study/'pdf13/analysis_pdf13/PDFmode1/delta.so'
    tablelib=C.CDLL(str(table_path));tablefn=tablelib.pysudakov_;tablefn.restype=C.c_double
    tablefn.argtypes=[C.POINTER(C.c_double),C.POINTER(C.c_double),C.POINTER(C.c_int),C.POINTER(C.c_int),C.POINTER(C.c_double)]
    masses=(C.c_double*21)(.33,.33,.5,1.5,4.8,*([0.]*16))
    def table(q,m,pid):
        vals=[C.c_double(q),C.c_double(m),C.c_int(abs(pid)),C.c_int(1)]
        return tablefn(*[C.byref(v) for v in vals],masses)
    ensemble=[]
    for sample,born in [('dy_fixed',fixed),('dy_damped',damped)]:
        array_path=a.study/f'pdf13/analysis_pdf13/{sample}.npz'
        with np.load(array_path) as z:
            original=z['values'][:,6,5,0];reference=z['reference'][:,6]
        selected=[i for i,b in enumerate(born) if abs(b['ids'][0])<=3]
        predictions={n:[] for n in [2,3,4]};clips={n:0 for n in predictions}
        for i in selected:
            b=born[i];Q=b['scalup'];M=np.sqrt(4*b['p'][0][0]*b['p'][1][0])
            if Q<=20:
                for n in predictions:predictions[n].append(1.)
                continue
            raw=[];correction={n:[] for n in predictions}
            for pid,x in zip(b['ids'][:2],b['x']):
                raw.append(lib.boundary_xf(pid,x,400.)/lib.boundary_xf(pid,x,Q*Q)*table(20.,M,pid)/table(Q,M,pid))
                for n in predictions:correction[n].append(fast(dict(pid=pid,x=x,q=20.,Q=Q,M=M),n))
            assert abs(np.prod(np.clip(raw,0,1))-original[i])<1.e-10
            for n in predictions:
                legs=np.array(raw)*np.exp(correction[n]);clips[n]+=int(((legs<0)|(legs>1)).sum())
                predictions[n].append(float(np.clip(legs,0,1).prod()))
        ref=reference[selected];old=original[selected]
        boot=np.random.default_rng(102828).integers(0,len(selected),size=(500,len(selected)))
        for n in predictions:
            corrected=np.array(predictions[n]);ratio=corrected.mean()/ref.mean()
            difference=corrected/np.array(predictions[4])-1.
            ensemble.append(dict(sample=sample,n=n,q=20.,born=len(selected),flavours='d,u,s and antiparticles',
                original_ratio=float(old.mean()/ref.mean()),corrected_ratio=float(ratio),
                corrected_ratio_bootstrap_se=float((corrected[boot].mean(axis=1)/ref[boot].mean(axis=1)).std(ddof=1)),
                corrected_clipped_legs=clips[n],reference=float(ref.mean()),corrected=float(corrected.mean()),
                max_abs_relative_difference_vs_4x4=float(abs(difference).max()),
                p99_abs_relative_difference_vs_4x4=float(np.percentile(abs(difference),99))))
    with (a.output/'cheap_boundary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)
    result=dict(rows=rows,summaries=summaries,benchmarks=benchmarks,ensemble_diagnostic=ensemble,exact_reference_seconds=exact_seconds,
        cases=len(cases),definition='Light-quark massless-II mode1 boundary exponent C; multiply each leg factor by exp(C) before clipping. '
            'Outer Gauss-Legendre in log(t), inner Gauss-Legendre in u=(1-z)/(1-max(Z,x)), '
            'with diagonal real-minus-virtual cancellation explicit and the zero-PDF strip Z<z<x integrated analytically.',
        limitations=['Diagnostic for light quarks only, no heavy thresholds or general dipole topology',
                     'Accuracy test is against the derived continuous kernel correction, not a fresh shower for each representative event',
                     'One-loop alphaS=.130 and PDF13; performance depends on the selected PDF implementation',
                     'Maximum PDF-call count includes two same-x/scale flavour queries; Pythia caches the full flavour vector',
                     'No production routine or selected table changed'],
        build=build,input_sha256={str(p):sha(p) for p in [Path(__file__),source,so,table_path,
            a.study/'pdf13/dy_fixed.born.jsonl',a.study/'pdf13/dy_damped.born.jsonl',
            a.study/'pdf13/analysis_pdf13/dy_fixed.npz',a.study/'pdf13/analysis_pdf13/dy_damped.npz',
            a.pythia/'lib/libpythia8.so',a.pythia/'share/Pythia8/pdfdata/NNPDF23_lo_as_0130_qed_0000.dat']})
    (a.output/'cheap_boundary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summaries=summaries,benchmarks=benchmarks,ensemble_diagnostic=ensemble),indent=2),flush=True)


if __name__=='__main__':main()
