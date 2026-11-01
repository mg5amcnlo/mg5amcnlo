#!/usr/bin/env python3
"""Replay frozen full-shower inputs after the sole diagnostic close->flush fix."""
from pathlib import Path
import os,sys,subprocess,concurrent.futures,hashlib,json,shutil,time
BASE=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101')
REPO=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS')
MC=BASE/'processes/Wbb_4FS/MCatNLO'; ORIGINAL=MC/'RUN_PYTHIA8_3'
OUT=BASE/'results/full_1M_replay'; BUILD=OUT/'build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(argv):
 print('BUILD', ' '.join(map(str,argv)),flush=True)
 subprocess.run(list(map(str,argv)),cwd=BUILD,check=True)
t0=time.time()
source=BASE/'processes/Wbb_4FS/Events/run_1M_wbb4fs_nocuts/events.lhe.gz'
inc=['-I'+str(BUILD/'include')]
if not (OUT/'input.lhe.gz_49.lhe').is_file():
 for src in ['srcCommon/HwU.f','PY8Analyzer/py8an_HwU_wbb.f','srcCommon/pineappl_interface_dummy.f']:
  command(['gfortran','-c',src,'-fPIC','-ffixed-line-length-132',*inc,'-o',Path(src).stem+'.o'])
 for src in ['PY8Analyzer/wbb_observables.cc','srcCommon/fjcore.cc']:
  command(['g++','-std=c++11','-c',src,'-fPIC','-I'+str(BUILD/'srcCommon'),'-o',Path(src).stem+'.o'])
 command(['g++','-std=c++11','-DNOPLUGIN','-I/nfs/home/rikkert/Downloads/pythia8/include',*inc,
  '-I'+str(BUILD/'srcCommon'),'Pythia83.cc','-o','Pythia8.exe','HwU.o','py8an_HwU_wbb.o',
  'wbb_observables.o','fjcore.o','pineappl_interface_dummy.o','-L/nfs/home/rikkert/Downloads/pythia8/lib',
  '-lpythia8','-lHepMCfio','-lgfortran','-I/export/slow1/rikkert/MCcntRefactor_Sfun/HEPTools/hepmc/include',
  '-L/export/slow1/rikkert/MCcntRefactor_Sfun/HEPTools/hepmc/lib',
  '-Wl,-rpath,/export/slow1/rikkert/MCcntRefactor_Sfun/HEPTools/hepmc/lib','-lHepMC','-L'+str(MC/'lib')])
 shutil.copy2(BUILD/'Pythia8.exe',OUT/'Pythia8.exe')
 # Use the same native splitter as the original shower interface; no event rewriting.
 sys.path.insert(0,str(REPO))
 from madgraph.various.lhe_parser import EventFile
 source=BASE/'processes/Wbb_4FS/Events/run_1M_wbb4fs_nocuts/events.lhe.gz'
 link=OUT/'input.lhe.gz'
 if not link.exists():link.symlink_to(source)
 os.chdir(OUT)
 f=EventFile('input.lhe.gz')
 try:n=f.split(20000,cwd='.')
 finally:f.close()
 assert n==50,n
print('SPLIT50',round(time.time()-t0,2),flush=True)
env=os.environ.copy();env['LD_LIBRARY_PATH']=':'.join(['/nfs/home/rikkert/Downloads/pythia8/lib','/export/slow1/rikkert/MCcntRefactor_Sfun/HEPTools/hepmc/lib','/nfs/home/rikkert/LHAPDF/lib',env.get('LD_LIBRARY_PATH','')]);env['PYTHIA8DATA']=str(ORIGINAL/'xmldoc')
shards=[]
for i in range(1,51):
 run=OUT/f'run_{i}';run.mkdir(exist_ok=True)
 shutil.copy2(ORIGINAL/f'run_{i}/Pythia8.cmd',run/'Pythia8.cmd')
 inp=OUT/f'input.lhe.gz_{i-1}.lhe'; assert inp.is_file(),inp
 
 if not (run/'events.lhe').exists():(run/'events.lhe').symlink_to(inp)
 
 if not (run/'Pythia8.exe').exists():(run/'Pythia8.exe').symlink_to('../Pythia8.exe')
 shards.append({'job':i,'input':str(inp),'input_sha256':sha(inp),'card_sha256':sha(run/'Pythia8.cmd')})
def job(i):
 run=OUT/f'run_{i}'
 with (run/'mcatnlo_run.log').open('wb') as log:
  result=subprocess.run(['./Pythia8.exe'],cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(f'job{i} failed: {result.returncode}')
 if (run/'MADatNLO.HwU').read_bytes()!=(ORIGINAL/f'run_{i}/MADatNLO.HwU').read_bytes():raise RuntimeError(f'job{i} HwU differs')
 stream=run/'wbb_observables.bin';assert stream.stat().st_size==8+20000*23*8,(i,stream.stat().st_size)
 assert not (run/'fort.97').exists(),i
 return i
# First job is an actual 20k-event checkpoint regression before the full replay.
print('PILOT_JOB',job(1),round(time.time()-t0,2),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as pool:
 for count,i in enumerate(pool.map(job,range(2,51)),start=2):
  if count%10==0 or count==50:print('REPLAY_DONE',count,round(time.time()-t0,2),flush=True)
manifest={'original_root':str(ORIGINAL),'replay_root':str(OUT),'source_input':str(source),'source_input_sha256':sha(source),'original_analyzer_sha256':sha(BASE/'results/full_1M_incomplete_streams/source_epoch/analysis_py8an_HwU_wbb.f'),'replay_analyzer_sha256':sha(BUILD/'PY8Analyzer/py8an_HwU_wbb.f'),'original_executable_sha256':sha(ORIGINAL/'Pythia8.exe'),'replay_executable_sha256':sha(OUT/'Pythia8.exe'),'driver_sha256':sha(BUILD/'Pythia83.cc'),'jobs':50,'events':1000000,'per_job_events':20000,'all_50_HwU_byte_identical':True,'no_fort_97':True,'seconds':time.time()-t0,'shards':shards}
(OUT/'replay_proof.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('REPLAY_ALL_PASS',round(time.time()-t0,2),flush=True)
