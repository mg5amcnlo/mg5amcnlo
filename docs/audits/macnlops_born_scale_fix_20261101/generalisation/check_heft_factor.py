from pathlib import Path
import os,json,ctypes,subprocess,sys,hashlib
import numpy as np
R=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS');sys.path.insert(0,str(R));from madgraph.various import macnlops
B=Path(__file__).resolve().parent;O=B/'direct/H_HEFT_diagnostic';rows=json.loads((O/'results.json').read_text())['points']
old=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101/diagnosis_transfer_20261101/scales/born_scale_factors.cc')
s=old.read_text().replace('NNPDF40_nlo_as_01180_nf_4','NNPDF40_nlo_as_01180');(O/'reference_factor.cc').write_text(s)
cmd=['g++','-O2','-shared','-fPIC','-I/nfs/home/rikkert/LHAPDF/include',str(O/'reference_factor.cc'),'-L/nfs/home/rikkert/LHAPDF/lib','-Wl,-rpath,/nfs/home/rikkert/LHAPDF/lib','-lLHAPDF','-o',str(O/'reference_factor.so')]
with (O/'reference_build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
with macnlops.LHEStream(B/'direct_inputs/H_HEFT/events.lhe') as stream:events=list(stream)
route=B/'processes/H_HEFT/SubProcesses/P0_gg_x0_no_t/macnlops_routes.dat';jmap={int(s.split()[1]):int(s.split()[3])-1 for s in route.read_text().splitlines() if s.startswith('SECTOR ')}
ids=np.full((len(rows),2),21,dtype=np.int32);x=[]
for r in rows:
 e=macnlops.Event(events[r['index']]);ps=e.hard.particles;fr=np.array([ps[0][2][0]/6500.,ps[1][2][0]/6500.])
 if r['kind']==2:
  shat=4*ps[0][2][0]*ps[1][2][0];fr[jmap[r['sector']]]*=125**2/shat
 x.append(fr)
x=np.ascontiguousarray(x);born=np.full(len(rows),62.5);real=np.array([r['muReal_HT2'] for r in rows]);ak=np.array([r['alphaK'] for r in rows]);v=np.empty((len(rows),6))
os.environ['LHAPDF_DATA_PATH']='/nfs/home/rikkert/LHAPDF/share/LHAPDF';lib=ctypes.CDLL(str(O/'reference_factor.so'));assert lib.init_born_scale_pdf()==0;lib.born_scale_factors.argtypes=[ctypes.c_long]+[ctypes.c_void_p]*6;p=lambda z:z.ctypes.data_as(ctypes.c_void_p);assert lib.born_scale_factors(len(rows),p(ids),p(x),p(born),p(real),p(ak),p(v))==0
c=np.array([r['coupling'] for r in rows]);f=np.array([r['pdf'] for r in rows]);maxerr=lambda a,b:float(np.max(np.abs(a/b-1)))
report=dict(points=len(rows),S=sum(r['kind']==1 for r in rows),H=sum(r['kind']==2 for r in rows),max_coupling_relative_error=maxerr(c,v[:,0]),max_pdf_relative_error=maxerr(f,v[:,1]),max_combined_relative_error=maxerr(c*f,v[:,2]),max_alpha_real_relative_error=float(np.max(np.abs(v[:,5]))),expected='Native HEFT GHGG proportional to G^2 implies Born |ME|^2 proportional to alpha_s^2; native dynamic Born H_T/2=125/2=62.5 GeV, including H inverse Born. Separate LHAPDF hard factors and independent incoming-x projection.',library_sha256=hashlib.sha256(Path('/nfs/home/rikkert/Downloads/pythia8/lib/libpythia8.so').read_bytes()).hexdigest(),compile_command=cmd)
(O/'reference_factor_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
