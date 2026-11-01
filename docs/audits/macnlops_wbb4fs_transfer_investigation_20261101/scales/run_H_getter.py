from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,shlex,subprocess
O=Path(__file__).parent;B=O.parents[1];P=B/'processes/Wbb_4FS';K=O.parent/'kernel'
D=O/'H_getter';D.mkdir(exist_ok=True)
providers=['P0_dxu_bbxwp','P0_udx_bbxwp'];exes={}
for provider in providers:
 root=P/'SubProcesses'/provider;exe=D/('extract_'+provider)
 cmd=shlex.split((root/'macnlops_onestep.build.log').read_text().replace('\\\n',' '))
 cmd[cmd.index('-o')+1]=str(exe)
 target=cmd.index('../../MCatNLO/srcPythia8/MacNLOPSMatch.o')
 cmd[target:target+1]=[str(K/'InverseBornMatch.o'),str(K/'born_getter.o')]
 with (D/(provider+'_link.log')).open('w') as log:subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
 exes[provider]=exe
(D/'build.json').write_text(json.dumps({p:{'exe':str(e),'sha256':hashlib.sha256(e.read_bytes()).hexdigest()} for p,e in exes.items()},indent=2)+'\n')
print('Built isolated H getter for both providers.',flush=True)
# The input preparation tool supplies a group manifest with original data paths.
manifest=O/'negative_H_inputs/manifest.json'
if not manifest.exists():raise RuntimeError('Waiting for '+str(manifest))
m=json.loads(manifest.read_text())
print(json.dumps(m,indent=2)[:2000],flush=True)
envbase=dict(os.environ,LD_LIBRARY_PATH=str(P/'lib')+':/nfs/home/rikkert/Downloads/pythia8/lib:/nfs/home/rikkert/LHAPDF/lib',LHAPDF_DATA_PATH='/nfs/home/rikkert/LHAPDF/share/LHAPDF',PYTHIA8DATA='/nfs/home/rikkert/Downloads/pythia8/share/Pythia8/xmldoc')
def links(src,dst):
 dst.mkdir(parents=True,exist_ok=True)
 for p in src.iterdir():
  if not p.is_file() or p.suffix in ('.o','.mod','.log','.lhe') or 'events' in p.name or 'onestep' in p.name:continue
  q=dst/p.name
  if not q.exists():
   try:q.symlink_to(p.resolve())
   except FileExistsError:pass
def run(g):
 provider=g['group'].split('/')[0];name=g['group'].replace('/','_');work=D/provider/('GF'+str(g['config']))
 links(P/'SubProcesses'/provider,work.parent);links(Path(g['res_original']).parent,work)
 pdfpath=work/'PDFsets'
 if not pdfpath.exists():pdfpath.symlink_to(P/'lib/PDFsets')
 out=D/(name+'.jsonl');env=dict(envbase,MACNLOPS_INVERSE_BORN=str(out))
 cmd=[str(exes[provider]),'--input',g['input'],'--output',str(D/(name+'_events.lhe')),'--summary',str(D/(name+'_summary.json')),'--routes',str(P/'SubProcesses'/provider/'macnlops_routes.dat'),'--settings',str(P/'Cards/macnlops_shower.cmnd'),'--pdf','LHAPDF6:NNPDF40_nlo_as_01180_nf_4/0','--seed','4930401','--res',g['res_local']]
 with (D/(name+'_run.log')).open('w') as log:r=subprocess.run(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(name,r.returncode)
 summary=json.loads((D/(name+'_summary.json')).read_text());assert summary['status']=='complete',summary
 return dict(group=g['group'],command=cmd,cwd=str(work),output=str(out),source_index=g['source_index'],returncode=r.returncode,source_events=g['events'])
with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(run,m['groups']))
(D/'run_manifest.json').write_text(json.dumps(dict(runs=records,original_inputs_unmodified=True),indent=2)+'\n')
print('Completed H-only inverse-Born diagnostics for all4groups.',flush=True)
