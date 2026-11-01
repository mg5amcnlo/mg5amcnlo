from pathlib import Path
import subprocess,os,json,hashlib
B=Path(__file__).resolve().parent;P=B/'processes/H_HEFT/SubProcesses/P0_gg_x0_no_t';S=P.parents[1]/'MCatNLO/srcPythia8';O=B/'direct/H_HEFT_diagnostic';O.mkdir(exist_ok=True)
s=(B/'direct_probe.cc').read_text()
s=s.replace('if(!identical)throw std::runtime_error("repeated request state drift");',r'''if(!identical)std::cout<<"DRIFT {\"index\":"<<event<<",\"r1\":"<<d.r<<",\"r2\":"<<repeat.r<<",\"d1\":"<<d.d<<",\"d2\":"<<repeat.d<<",\"k1\":"<<d.k<<",\"k2\":"<<repeat.k<<",\"a1\":"<<d.alphaK<<",\"a2\":"<<repeat.alphaK<<",\"A1\":"<<d.sudakov.source<<",\"A2\":"<<repeat.sudakov.source<<",\"coupling1\":"<<d.bornScale.coupling<<",\"coupling2\":"<<repeat.bornScale.coupling<<",\"pdf1\":"<<d.bornScale.pdf<<",\"pdf2\":"<<repeat.bornScale.pdf<<"}\n";''')
s=s.replace('<<d.bornScale.pdf<<",\\\"repeat_identical\\\":true,', '<<d.bornScale.pdf<<",\\\"repeat_identical\\\":"<<(identical?"true":"false")<<",')
(S/'HeftPointProbe.cc').write_text(s)
with (O/'build.log').open('w') as log:
 subprocess.run(['g++','-O','-std=c++17','-I/nfs/home/rikkert/Downloads/pythia8/include','-c',str(S/'HeftPointProbe.cc'),'-o',str(O/'HeftPointProbe.o')],stdout=log,stderr=subprocess.STDOUT,check=True)
 build=(P/'macnlops_onestep.build.log').read_text();link=build[build.index('gfortran -lstdc++'):].replace(' -o macnlops_onestep',' -o '+str(O/'probe')).replace('../../MCatNLO/srcPythia8/MacNLOPSMatch.o',str(O/'HeftPointProbe.o'))
 subprocess.run(link,shell=True,cwd=P,stdout=log,stderr=subprocess.STDOUT,check=True)
env=dict(os.environ,LD_LIBRARY_PATH=str(P.parents[1]/'lib')+':/nfs/home/rikkert/LHAPDF/lib:/nfs/home/rikkert/Downloads/pythia8/lib',LHAPDF_DATA_PATH='/nfs/home/rikkert/LHAPDF/share/LHAPDF')
cmd=[str(O/'probe'),str(B/'direct_inputs/H_HEFT/events.lhe'),str(P/'macnlops_routes.dat')]
with (O/'run.log').open('w') as log:r=subprocess.run(cmd,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT)
lines=(O/'run.log').read_text().splitlines();points=[json.loads(l[7:]) for l in lines if l.startswith('GPOINT ')];drift=[json.loads(l[6:]) for l in lines if l.startswith('DRIFT ')]
(O/'results.json').write_text(json.dumps(dict(command=cmd,returncode=r.returncode,points=points,drift=drift),indent=2)+'\n')
print('returncode',r.returncode,'points',len(points),'drift',len(drift));print(drift[:3])
