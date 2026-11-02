#!/usr/bin/env python3
"""Compare compiled generation Durham cuts to fjcore on physical RAMBO points.

No process is exported, generated, rebuilt, or modified. Only standalone audit
libraries are compiled in a temporary directory. Born and real states use the
actual fixed generated multiplicities, including a padded zero emitted slot.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from specification import durham_cut


BASE = Path('/export/slow1/rikkert/macnlops_native_tc_1M_20261101')
REFERENCE = r'''
#include "fjcore.hh"
#include <cmath>
#include <vector>
extern "C" int reference(int n,const double*p,const int*status,const int*id,
                          int target,int*count,double*distance){
  fjcore::ClusterSequence::set_fastjet_banner_stream(nullptr);
  std::vector<fjcore::PseudoJet> parts;
  for(int i=2;i<n;++i){
    if(status[i]!=1 || p[5*i]<=0.)continue;
    if(std::abs(id[i])>5 && id[i]!=21)continue;
    parts.emplace_back(p[5*i+1],p[5*i+2],p[5*i+3],p[5*i]);
  }
  const double e=p[0]+p[5],x=p[1]+p[6],y=p[2]+p[7],z=p[3]+p[8];
  const double q2=e*e-x*x-y*y-z*z;
  if(parts.empty() || q2<=0.){*count=0;*distance=0.;return 0;}
  fjcore::ClusterSequence cs(parts,fjcore::JetDefinition(fjcore::ee_kt_algorithm));
  // Use the incoming invariant mass even in status-filtering diagnostics.
  // exclusive_jets_ycut() instead uses the selected final state's energy.
  *count=cs.n_exclusive_jets(.01*q2);
  *distance=parts.size()>=unsigned(target) ? cs.exclusive_dmerge(target-1)/q2 : 0.;
  return *count>=target;
}
'''


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rambo(rng,number,multiplicity,energy=91.188):
    """Massless RAMBO: independent isotropic seeds, rest boost, common scaling."""
    u=rng.random((number,multiplicity,4));c=2*u[:,:,0]-1;phi=2*np.pi*u[:,:,1]
    e=-np.log(u[:,:,2]*u[:,:,3]);s=np.sqrt(1-c*c)
    q=np.stack((e,e*s*np.cos(phi),e*s*np.sin(phi),e*c),axis=2)
    return rest_and_scale(q,energy)


def rest_and_scale(q,energy):
    total=q.sum(axis=1);m=np.sqrt(total[:,0]**2-np.sum(total[:,1:]**2,axis=1))
    velocity=-total[:,1:]/total[:,0,None];b2=np.sum(velocity*velocity,axis=1)
    gamma=total[:,0]/m;bp=np.sum(q[:,:,1:]*velocity[:,None,:],axis=2)
    coefficient=np.divide(gamma-1,b2,out=np.zeros_like(gamma),where=b2>0)
    out=np.empty_like(q);out[:,:,0]=gamma[:,None]*(q[:,:,0]+bp)
    out[:,:,1:]=q[:,:,1:]+(coefficient[:,None]*bp+gamma[:,None]*q[:,:,0])[:,:,None]*velocity[:,None,:]
    out*=energy/m[:,None,None]
    return out


def compile_libraries(work,common):
    commands=[];cuts={}
    for target in (3,4):
        folder=work/str(target);folder.mkdir()
        (folder/'nexternal.inc').write_text('      integer nexternal,nincoming\n'
            f'      parameter(nexternal={target+3},nincoming=2)\n')
        source=folder/'cut.f'
        wrapper=f'''
      subroutine evaluate(p,status,ids,answer)bind(C,name='evaluate')
      use iso_c_binding
      implicit none
      real(c_double) p(0:4,{target+3})
      integer(c_int) status({target+3}),ids({target+3}),answer
      logical dummy_cuts
      external dummy_cuts
      answer=0
      if(dummy_cuts(p,status,ids))answer=1
      end
'''
        source.write_text(durham_cut(target)+wrapper)
        library=folder/'cut.so'
        command=['gfortran','-O2','-shared','-fPIC','-ffixed-line-length-none',
                 '-I'+str(folder),str(source),'-o',str(library)]
        subprocess.run(command,check=True,capture_output=True,text=True);commands.append(command)
        lib=ctypes.CDLL(str(library));fun=lib.evaluate
        fun.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int),
                      ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)]
        cuts[target]=(lib,fun)
    source=work/'reference.cc';source.write_text(REFERENCE);library=work/'reference.so'
    command=['g++','-O2','-std=c++11','-shared','-fPIC','-I'+str(common),
             str(source),str(common/'fjcore.cc'),'-o',str(library)]
    subprocess.run(command,check=True,capture_output=True,text=True);commands.append(command)
    lib=ctypes.CDLL(str(library));fun=lib.reference
    fun.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int),
                  ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.POINTER(ctypes.c_int),
                  ctypes.POINTER(ctypes.c_double)];fun.restype=ctypes.c_int
    return cuts,(lib,fun),commands


def event(final,target,zero_slot=None):
    n=target+3;p=np.zeros((n,5));p[0,:4]=[45.594,0,0,45.594];p[1,:4]=[45.594,0,0,-45.594]
    status=np.ones(n,dtype=np.int32);status[:2]=-1
    ids=np.full(n,21,dtype=np.int32);ids[:2]=[-11,11]
    slots=[i for i in range(target+1) if i!=zero_slot] if zero_slot is not None else list(range(len(final)))
    for k,(j,q) in enumerate(zip(slots,final)):
        p[j+2,:4]=q;ids[j+2]=(1,-1)[k] if k<2 else 21
    return p,status,ids


def audit(common,output,number=5000,seed=20261101):
    output.parent.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed)
    records={};mismatches=[];max_conservation=max_mass_shell=0.;comparisons=0
    with tempfile.TemporaryDirectory(prefix='native-tc-durham-') as temporary:
        cuts,reference,commands=compile_libraries(Path(temporary),common)
        def compare(target,kind,variant,p,status,ids):
            nonlocal comparisons
            pp=p.ctypes.data_as(ctypes.POINTER(ctypes.c_double));ss=status.ctypes.data_as(ctypes.POINTER(ctypes.c_int));ii=ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            count=ctypes.c_int();distance=ctypes.c_double();answer=ctypes.c_int()
            expected=reference[1](len(p),pp,ss,ii,target,ctypes.byref(count),ctypes.byref(distance))
            cuts[target][1](pp,ss,ii,ctypes.byref(answer));comparisons+=1
            key=f'{target}j_{kind}_{variant}';r=records.setdefault(key,dict(points=0,accepted=0,rejected=0,mismatches=0,minimum_threshold_margin=1.))
            r['points']+=1;r['accepted']+=bool(expected);r['rejected']+=not expected
            r['minimum_threshold_margin']=min(r['minimum_threshold_margin'],abs(distance.value-.01))
            if expected!=answer.value:
                r['mismatches']+=1
                if len(mismatches)<10:mismatches.append(dict(target=target,kind=kind,variant=variant,expected=expected,actual=answer.value,count=count.value,p=p.tolist(),status=status.tolist(),ids=ids.tolist()))
            return expected
        for target in (3,4):
            for kind,multiplicity in (('Born',target),('real',target+1)):
                points=rambo(rng,number,multiplicity)
                total=points.sum(axis=1);max_conservation=max(max_conservation,float(np.max(np.abs(total-np.array([91.188,0.,0.,0.])))))
                max_mass_shell=max(max_mass_shell,float(np.max(np.abs(points[:,:,0]**2-np.sum(points[:,:,1:]**2,axis=2)))))
                for final in points:
                    p,status,ids=event(final,target,zero_slot=target if kind=='Born' else None)
                    compare(target,kind,'canonical',p,status,ids)
                    # These controls retain the same incoming normalization.
                    status[2]=2;compare(target,kind,'status_filter',p,status,ids);status[2]=1
                    ids[2]=22;compare(target,kind,'nonparton_filter',p,status,ids)
                    if kind=='Born':
                        for slot in range(target):
                            p,status,ids=event(final,target,zero_slot=slot)
                            compare(target,kind,'zero_interior_'+str(slot),p,status,ids)
                    else:
                        p,status,ids=event(final[::-1],target)
                        compare(target,kind,'reversed_packing',p,status,ids)
        result=dict(passed=not mismatches,seed=seed,physical_points_per_Born_real_sample=number,
            independent_physical_points=4*number,comparisons=comparisons,records=records,
            mismatches=mismatches,max_four_momentum_residual_GeV=max_conservation,
            max_mass_shell_residual_GeV2=max_mass_shell,
            reference='Independent fjcore ee_kt E-scheme clustering at absolute dcut=.01*s_incoming; status/PDG/zero-energy filtering applied before clustering.',
            source='Compiled actual specification.durham_cut(3/4), nexternal=6/7 and nincoming=2.',
            fingerprints={str(p):sha(p) for p in (Path(__file__).resolve(),Path(__file__).with_name('specification.py'),common/'fjcore.cc',common/'fjcore.hh')},
            compile_commands=commands,process_files_modified=False,physics_generation_or_shower=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    if mismatches:raise AssertionError('Durham cut mismatches: '+str(output))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fjcore',type=Path,default=BASE/'processes/EE3Jets_5FS/MCatNLO/srcCommon')
    parser.add_argument('--output',type=Path,default=BASE/'validation/durham_cut_validation.json')
    parser.add_argument('--points',type=int,default=5000);parser.add_argument('--seed',type=int,default=20261101)
    args=parser.parse_args();result=audit(args.fjcore.resolve(),args.output.resolve(),args.points,args.seed)
    print(json.dumps({k:result[k] for k in ('passed','independent_physical_points','comparisons','max_four_momentum_residual_GeV','max_mass_shell_residual_GeV2')},indent=2))
