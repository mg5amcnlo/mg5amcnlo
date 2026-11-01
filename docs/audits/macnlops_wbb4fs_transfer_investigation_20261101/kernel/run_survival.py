from pathlib import Path
import os,subprocess,json,concurrent.futures,hashlib
D=Path(__file__).resolve().parent;B=D.parents[1]
env=dict(os.environ,LD_LIBRARY_PATH='/nfs/home/rikkert/Downloads/pythia8/lib:/nfs/home/rikkert/LHAPDF/lib',LHAPDF_DATA_PATH='/nfs/home/rikkert/LHAPDF/share/LHAPDF',PYTHIA8DATA='/nfs/home/rikkert/Downloads/pythia8/share/Pythia8/xmldoc')
settings=B/'processes/Wbb_4FS/Cards/macnlops_shower.cmnd'
def run(mode):
 cmd=[str(D/'conditional_survival'),str(D/('points_'+mode+'.dat')),str(D/('native_'+mode+'.csv')),'6000','4011227',str(settings),mode]
 with (D/('native_'+mode+'.log')).open('w') as log:
  p=subprocess.run(cmd,cwd=D,env=env,stdout=log,stderr=subprocess.STDOUT)
 print(mode,p.returncode,flush=True)
 return dict(mode=mode,command=cmd,returncode=p.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:result=list(pool.map(run,['ALL','ISR','FSR']))
(D/'survival_manifest.json').write_text(json.dumps(dict(runs=result,library_sha256=hashlib.sha256(Path('/nfs/home/rikkert/Downloads/pythia8/lib/libpythia8.so').read_bytes()).hexdigest()),indent=2)+'\n')
