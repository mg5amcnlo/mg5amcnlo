from pathlib import Path
import hashlib,json,os,subprocess,sys,time
B=Path(__file__).parent;R=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS')
old=json.loads(Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101/validation/pdf_support_existing_tests.json').read_text())['command']
modules=old[3:];modules=[m for m in modules if m!='-v']
modules+=['tests.unit_tests.various.test_macnlops_pdf_support','tests.unit_tests.various.test_macnlops_born_scales','tests.unit_tests.fks.test_macnlops_born_scale_factor']
cmd=[sys.executable,'-m','unittest',*modules,'-v']
env=dict(os.environ,PYTHIA8='/nfs/home/rikkert/Downloads/pythia8',PYTHIA8DATA='/nfs/home/rikkert/Downloads/pythia8/share/Pythia8/xmldoc',LHAPDF_DATA_PATH='/nfs/home/rikkert/LHAPDF/share/LHAPDF',LD_LIBRARY_PATH='/nfs/home/rikkert/Downloads/pythia8/lib:/nfs/home/rikkert/LHAPDF/lib')
files=['Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc','Template/NLO/SubProcesses/driver_macnlops.f','Template/NLO/SubProcesses/montecarlocounter.f','Template/NLO/SubProcesses/weight_lines.f','Template/NLO/SubProcesses/macnlops_capi.f','madgraph/various/macnlops.py']
before={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in files};start=time.monotonic();log=B/'logs/regressions.log'
with log.open('w') as out:p=subprocess.run(cmd,cwd=R,env=env,stdout=out,stderr=subprocess.STDOUT)
after={q:hashlib.sha256((R/q).read_bytes()).hexdigest() for q in files};assert after==before
record=dict(command=cmd,exit_code=p.returncode,elapsed_seconds=time.monotonic()-start,sources_sha256=before,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
(B/'logs/regressions.json').write_text(json.dumps(record,indent=2)+'\n');print(record);sys.exit(p.returncode)
