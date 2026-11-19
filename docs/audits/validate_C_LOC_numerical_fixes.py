#!/usr/bin/env python3
"""Read-only full-physics MINT checks of local-recoil numerical changes.

Copies the exported C_LOC process to scratch. Production resources and grids are
never written. Existing grids are only proposal distributions, not old integrals.
The baseline is the exported process; fixed objects are built from a frozen copy
of the selected current Template sources. No shower or event generation is run.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PROCESSES = ['P0_bdx_tux', 'P0_bu_td', 'P0_dxb_tux', 'P0_ub_td']
VARIABLES = ['FC','FFLAGS','LDFLAGS','MINTMC','NLOLIBS','APPLLIBS','LINKLIBS','FO_EXTRAPATHS','FO_EXTRALIBS','FJLIBS']
DEFAULT_SOURCES = ['genps_fks_helpers.f','genps_fks_radiation.f','genps_fks.f','initial_recoil.f','fks_Sij.f','fks_singular.f']
FAMILIES = {
 'native_reconstruction_mismatch': 'Native MC reconstruction mismatch',
 'counterevent_momenta_mismatch': 'Kinematics of counterevents',
 'cms_angle_outside_range': 'Warning in get_cms_costh_fks',
 'inconsistent_shat': 'Inconsistent shat',
 'nonfinite_weight_warning': 'Rejecting phase-space point: nonfinite weight',
 'native_H_projection_warning': 'invalid native MC H radiation projection',
 'mass_shell_warning': 'Mass shell violation',
 'missing_eikonal_counterevent': 'WARNING in eikonal_reduced: no cnt momenta',
 'oneloop_c0_all_zero': 'ERROR in OneLOop c0: all input equal zero',
 'oneloop_d0_permutation_attempt': 'WARNING from OneLOop d0: input does not seem to represent hard kinematics',
 'oneloop_d0_permutation_failure': 'WARNING from OneLOop d0: no permutation helps',
 'maxjetflavor': 'WARNING: the value of maxjetflavor',
}

def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot(args):
 export = args.work_root/'export_baseline'
 if not export.exists():
  shutil.copytree(args.process, export, symlinks=True)
  record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source':str(args.process),
          'source_sha256':{p.name:sha(p) for p in (export/'SubProcesses').iterdir() if p.is_file() and p.suffix in ('.f','.f90','.inc')},
          'binary_sha256':{p:sha(export/'SubProcesses'/p/'madevent_mintMC') for p in PROCESSES}}
  (args.work_root/'baseline_snapshot.json').write_text(json.dumps(record,indent=2)+'\n')
 return export

def variables(export, proc):
 mk=export/'print_variables.mk'
 if not mk.exists():mk.write_text('include makefile\n'+''.join('$(info '+k+'=$('+k+'))\n' for k in VARIABLES)+'\n.PHONY: codex_info\ncodex_info:\n\t@true\n')
 out=subprocess.run(['make','-f',str(mk),'codex_info'],cwd=export/'SubProcesses'/proc,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True).stdout
 return {k:next(x[len(k)+1:] for x in out.splitlines() if x.startswith(k+'=')) for k in VARIABLES}

def prepare_fixed(args, baseline):
 export=args.work_root/'export_fixed'
 if export.exists():
  raise RuntimeError('Fixed snapshot already exists; use a fresh scratch directory to preserve evidence')
 shutil.copytree(baseline,export,symlinks=True,ignore=shutil.ignore_patterns('numerical_validation'))
 sources={}
 for name in args.sources:
  source=ROOT/'Template/NLO/SubProcesses'/name
  target=export/'SubProcesses'/name
  shutil.copy2(source,target)
  sources[name]=sha(target)
 record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sources}
 (args.work_root/'fixed_snapshot.json').write_text(json.dumps(record,indent=2)+'\n')
 def build(proc):
  pd=export/'SubProcesses'/proc; cfg=variables(export,proc)
  with (args.work_root/(proc+'_build.log')).open('w') as log:
   compile_sources=list(args.sources)
   for obj in shlex.split(cfg['MINTMC']):
    if (pd/obj).exists():continue
    candidates=[Path(obj).with_suffix(suffix).name for suffix in ('.f','.f90')]
    source=next((name for name in candidates if (pd/name).exists()),None)
    if source is None:raise RuntimeError('Missing original object and source: '+str(pd/obj))
    if source not in compile_sources:compile_sources.append(source)
   for name in compile_sources:
    obj=Path(name).with_suffix('.o').name
    subprocess.run(shlex.split(cfg['FC'])+shlex.split(cfg['FFLAGS'])+['-fallow-argument-mismatch','-I.','-I'+str(export/'lib'),'-c',str(pd/name),'-o',str(pd/obj)],cwd=pd,stdout=log,stderr=subprocess.STDOUT,check=True)
   cmd=shlex.split(cfg['FC'])+['-o',str(pd/'madevent_mintMC')]+[str(pd/p) for p in shlex.split(cfg['MINTMC'])]
   for key in [*VARIABLES[4:],'LDFLAGS']:cmd+=shlex.split(cfg[key])
   subprocess.run(cmd,cwd=pd,stdout=log,stderr=subprocess.STDOUT,check=True)
  return proc
 variables(export,PROCESSES[0])  # Materialize make fragment before parallel readers.
 with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(build,PROCESSES))
 record['binary_sha256']={p:sha(export/'SubProcesses'/p/'madevent_mintMC') for p in PROCESSES}
 (args.work_root/'fixed_snapshot.json').write_text(json.dumps(record,indent=2)+'\n')
 return export

def run(args, export, proc):
 pd=export/'SubProcesses'/proc; work=pd/'numerical_validation'
 if work.exists():raise RuntimeError(str(work)+' already exists')
 work.mkdir()
 for name in ['FKS_params.dat','grid.MC_integer','param_card.dat','leshouche_info.dat','configs_and_props_info.dat','symfact.dat','initial_states_map.dat']:
  shutil.copy2(pd/'GF1.0'/name,work/name)
 (work/'mint_grids').write_text(''.join(line for line in (pd/'GF1.0/mint_grids').read_text().splitlines(True) if not line.lstrip().startswith('MAX')))
 (work/'randinit').write_text('r='+str(args.seed)+'\n')
 inp=f'{args.points} {args.iterations}\n-1.0\n1 -0.1\n-1 -0.1\n1\n1\n1.0\n1\n1 1 1\nall\n'
 (work/'input_app.txt').write_text(inp)
 start=time.monotonic()
 with (work/'validation.log').open('w') as log:
  result=subprocess.run([str(pd/'madevent_mintMC')],cwd=work,input=inp,text=True,stdout=log,stderr=subprocess.STDOUT)
 text=(work/'validation.log').read_text(); values={}
 if result.returncode or 'Detailed CPU timing:' not in text:
  raise RuntimeError('Incomplete integration; inspect '+str(work/'validation.log'))
 for line in (work/'mint_grids').read_text().splitlines():
  s=line.split()
  if s and s[0] in ['SUM','QSM']:values[s[0]]=[float(v.replace('D','E')) for v in s[1:]]
 if not all(math.isfinite(v) for name in ['SUM','QSM'] for v in values.get(name,[])):
  raise RuntimeError('Nonfinite MINT result; inspect '+str(work/'validation.log'))
 item={'process':proc,'returncode':result.returncode,'elapsed_seconds':time.monotonic()-start,'log':str(work/'validation.log'),
       'diagnostic_lines':{name:text.count(needle) for name,needle in FAMILIES.items()},
       'rejected_kinematics':sum(int(x) for x in re.findall(r'Total phase-space points rejected for invalid MC kinematics:\s*(\d+)',text)),
       'rejected_nonfinite':sum(int(x) for x in re.findall(r'Total phase-space points rejected for nonfinite weights:\s*(\d+)',text)),
       'sampled_points':re.findall(r'sampled.*?of\s+(\d+)\s+integration points',text),
       'SUM':values.get('SUM'),'QSM':values.get('QSM')}
 if 'SUM' in values:item['sigma_pb']=values['SUM'][1];item['error_pb']=values['QSM'][1]
 (args.work_root/(args.mode+'_'+proc+'.json')).write_text(json.dumps(item,indent=2)+'\n')
 print(json.dumps(item),flush=True)
 return item

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--process',type=Path,default=ROOT/'central_runs_fixed/C_LOC')
 p.add_argument('--work-root',type=Path,required=True)
 p.add_argument('--mode',choices=['baseline','fixed'],required=True)
 p.add_argument('--points',type=int,default=100000)
 p.add_argument('--iterations',type=int,default=3)
 p.add_argument('--seed',type=int,default=712925)
 p.add_argument('--sources',nargs='+',default=DEFAULT_SOURCES)
 args=p.parse_args();args.process=args.process.resolve();args.work_root=args.work_root.resolve();args.work_root.mkdir(parents=True,exist_ok=True)
 baseline=snapshot(args)
 export=baseline if args.mode=='baseline' else prepare_fixed(args,baseline)
 with concurrent.futures.ThreadPoolExecutor(4) as pool:results=list(pool.map(lambda proc:run(args,export,proc),PROCESSES))
 record={'mode':args.mode,'points_per_iteration':args.points,'iterations':args.iterations,'seed':args.seed,'processes':results,
         'sigma_pb':sum(x['sigma_pb'] for x in results),'error_pb':math.sqrt(sum(x['error_pb']**2 for x in results)),
         'rejected_kinematics':sum(x['rejected_kinematics'] for x in results),'rejected_nonfinite':sum(x['rejected_nonfinite'] for x in results),
         'diagnostic_lines':{name:sum(x['diagnostic_lines'][name] for x in results) for name in FAMILIES},
         'limitations':'Independent adaptive integrations; quoted cross-section difference uncertainty must include both errors. Rejection counts alone do not bound bias. Fixed-order closure is only as precise as this small controlled integration.'}
 (args.work_root/(args.mode+'_results.json')).write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record),flush=True)

if __name__=='__main__':main()
