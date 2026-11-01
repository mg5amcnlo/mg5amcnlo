from pathlib import Path
import subprocess,os,json
B=Path(__file__).resolve().parent;P=B/'processes/H_HEFT/SubProcesses/P0_gg_x0_no_t';O=B/'direct/H_HEFT_diagnostic'
s=(P/'driver_macnlops.f').read_text();start=s.index('         call macnlops_born_scale_factor(p,p_born,');end=s.index('         if(kernel_status.ne.0)then',start)
s=s[:start]+'''         macnlops_born_scale_coupling=1d0\n         macnlops_born_scale_pdf=1d0\n         kernel_status=0\n'''+s[end:];(O/'driver_nog.f').write_text(s)
with (O/'nog_build.log').open('w') as log:
 subprocess.run(['gfortran','-O','-ffixed-line-length-132','-I'+str(P),'-I'+str(P.parents[1]/'lib'),'-c',str(O/'driver_nog.f'),'-o',str(O/'driver_nog.o')],cwd=O,stdout=log,stderr=subprocess.STDOUT,check=True)
 build=(P/'macnlops_onestep.build.log').read_text();link=build[build.index('gfortran -lstdc++'):].replace(' -o macnlops_onestep',' -o '+str(O/'nog_probe')).replace('../../MCatNLO/srcPythia8/MacNLOPSMatch.o',str(O/'HeftPointProbe.o')).replace('driver_macnlops.o',str(O/'driver_nog.o'))
 subprocess.run(link,shell=True,cwd=P,stdout=log,stderr=subprocess.STDOUT,check=True)
env=dict(os.environ,LD_LIBRARY_PATH=str(P.parents[1]/'lib')+':/nfs/home/rikkert/LHAPDF/lib:/nfs/home/rikkert/Downloads/pythia8/lib',LHAPDF_DATA_PATH='/nfs/home/rikkert/LHAPDF/share/LHAPDF')
cmd=[str(O/'nog_probe'),str(B/'direct_inputs/H_HEFT/events.lhe'),str(P/'macnlops_routes.dat')]
with (O/'nog_run.log').open('w') as log:r=subprocess.run(cmd,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT)
lines=(O/'nog_run.log').read_text().splitlines();points=[json.loads(l[7:]) for l in lines if l.startswith('GPOINT ')];drift=[json.loads(l[6:]) for l in lines if l.startswith('DRIFT ')]
(O/'nog_results.json').write_text(json.dumps(dict(command=cmd,returncode=r.returncode,points=points,drift=drift),indent=2)+'\n')
print('returncode',r.returncode,'points',len(points),'drift',len(drift));print(drift[:2])
