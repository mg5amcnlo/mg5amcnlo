"""Join direct current SudGen API trials with the branch's unchanged table."""
from pathlib import Path
import ctypes as C
import csv,hashlib,json,math,sys
from delta_evaluator import DeltaEvaluator,_array,_ptr,ROOT
w=Path(sys.argv[1])
rows=[]
with DeltaEvaluator(work=w/'table_build') as e:
 fn=e.lib.pysudakov_;fn.restype=C.c_double
 masses=_array([.33,.33,.5,1.5,4.8,0,0,0,0,0,.000510998928,0,.1056583715,0,1.77682,0,0,0,0,0,0])
 def val(q,m,pdg,t):
  args=[C.c_double(q),C.c_double(m),C.c_int(pdg),C.c_int(t)]
  return fn(*[C.byref(x) for x in args],_ptr(masses))
 for filename in ['fresh_vendor.log','fresh_0118_pdf2.log','fresh_0118_pdf1.log','fresh_gluon_fsr.log']:
  for line in (w/filename).read_text().splitlines():
   if not line.startswith('RESULT '):continue
   a=line.split();q,Q,m=map(float,a[1:4]);pdg,t,seed=map(int,a[4:7]);p=float(a[7]);n=int(a[8])
   ratio=val(q,m,pdg,t)/val(Q,m,pdg,t)
   power=1./((2 if t==2 else 1)*(2 if pdg==21 else 1))
   norm=p**power
   normerr=power*p**(power-1)*math.sqrt(p*(1-p)/n) if p>0 else None
   rows.append(dict(setup=filename,q=q,Q=Q,mass=m,pdg=pdg,type=t,seed=seed,fresh_api=p,trials=n,
       fresh_error=math.sqrt(p*(1-p)/n),fresh_per_line=norm,fresh_per_line_error=normerr,api_to_line_power=power,shipped_ratio=ratio,
       shipped_per_leg=ratio**(2 if pdg==21 else 1)))
 provenance=e.provenance
with (w/'fresh_comparison.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
files=[ROOT/'tests/validation/delta_noemission/table_fresh_probe.cc',ROOT/'vendor/SudGen/pythia_interface_for_sudgen.cc',Path('/export/tmp/rikkert/Pythia/pythia8313/src/SimpleSpaceShower.cc'),Path('/export/tmp/rikkert/Pythia/pythia8313/src/SimpleTimeShower.cc'),Path('/export/tmp/rikkert/Pythia/pythia8313/lib/libpythia8.so'),w/'fresh_probe',w/'points.txt']
provenance['fresh_source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
provenance['version']='Pythia8.313'
provenance['normalization']='Current API ISR=one full radiator, FI=one full radiator, FF=both radiators. Per-colour-line probability is API^(1/((2 for FF else1)*(2 for gluon else1))); dummyFF ends symmetric. No heavy-top claim: API dummy radiators massless.'
provenance['settings_note']='vendor means ISRalpha.130 FSRalpha.1365, pdfMode2. The current wrapper explicitly removes PDF ratios via ISRpdfMode; TimeShowerpdfMode remains default0.'
provenance['fresh_setup']='Unmodified current vendor initializer; control replaces both shower alphaS=.118 order1 CMWoff and ISRpdfMode=1 or2 before reinitialization'
(w/'fresh_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
for r in rows:
 if r['q']==20:print(r)
