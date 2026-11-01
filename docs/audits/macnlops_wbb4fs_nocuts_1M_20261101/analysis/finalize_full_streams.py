#!/usr/bin/env python3
"""Freeze complete replay streams and prove surviving original rows are identical."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
B=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101')
R=B/'results/full_1M_replay';O=B/'processes/Wbb_4FS/MCatNLO/RUN_PYTHIA8_3';A=B/'results/full_1M'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((R/'replay_proof.json').read_text());assert proof['all_50_HwU_byte_identical']
A.mkdir(exist_ok=True);manifest=[];endpoints=[]
for i in range(1,51):
 old=O/f'run_{i}';new=R/f'run_{i}'
 initial=(old/'wbb_observables.bin').read_bytes();complete=(new/'wbb_observables.bin').read_bytes()
 assert len(initial)==8+5001*184;assert len(complete)==8+20000*184
 assert initial==complete[:len(initial)],i
 sequential=np.fromfile(old/'fort.97',dtype=np.dtype([('begin','<i4'),('row','<f8',(23,)),('end','<i4')]))
 assert len(sequential)==4999 and np.all(sequential['begin']==184) and np.all(sequential['end']==184),i
 assert sequential['row'].tobytes()==complete[8+15001*184:],i
 assert (old/'MADatNLO.HwU').read_bytes()==(new/'MADatNLO.HwU').read_bytes(),i
 endpoints.append(dict(job=i,events=20000,first_5001_identical=True,last_4999_identical=True,hwu_byte_identical=True))
 for name in ['wbb_observables.bin','wbb_weight_names.txt','MADatNLO.HwU','Pythia8.cmd','mcatnlo_run.log']:
  f=new/name;dest=A/f'run_{i}'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(f,dest)
  manifest.append(dict(source=str(f),path=str(dest.relative_to(A)),bytes=dest.stat().st_size,sha256=sha(dest)))
for f in [R/'Pythia8.exe',R/'replay_proof.json',R/'replay.log',O/'shower_card.dat']:
 dest=A/f.name;shutil.copy2(f,dest);manifest.append(dict(source=str(f),path=f.name,bytes=dest.stat().st_size,sha256=sha(dest)))
for f in [R/'build/PY8Analyzer/py8an_HwU_wbb.f',R/'build/PY8Analyzer/wbb_observables.cc',R/'build/Pythia83.cc',B/'analysis/observables.json']:
 dest=A/'source_epoch'/f.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(f,dest)
 manifest.append(dict(source=str(f),path=str(dest.relative_to(A)),bytes=dest.stat().st_size,sha256=sha(dest)))
(A/'manifest.json').write_text(json.dumps(dict(events=1000000,jobs=50,cuts='none',replay_proof=proof,original_surviving_rows=endpoints,files=manifest),indent=2)+'\n')
print(f'Archived {len(manifest)} files; all 1M rows complete; 500k surviving original rows and all 50 HwU files byte-identical.')
