from pathlib import Path
import os,subprocess,json,hashlib,shutil
R=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS');B=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101');D=B/'diagnosis_transfer_20261101/kernel';P=B/'processes/Wbb_4FS/SubProcesses/P0_udx_bbxwp'
getter='''subroutine macnlops_diag_born(out) bind(C)\nuse iso_c_binding\nuse fks_phase_space_data, only: p_born\nimplicit none\ninclude 'nexternal.inc'\nreal(c_double), intent(out) :: out(4*(nexternal-1))\ninteger i,j\ndo i=1,nexternal-1\n do j=0,3\n  out(4*(i-1)+j+1)=p_born(j,i)\n enddo\nenddo\nend subroutine\n'''
(D/'born_getter.f90').write_text(getter)
source=(R/'Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc').read_text()
source=source.replace('extern "C" {','extern "C" {\nvoid macnlops_diag_born(double*);',1)
needle='      const bool accepted = decisions() < decision.keep;'
injection=r'''
      if(decision.evaluated && decision.density.sudakov.ready) {
        static std::ofstream diag(std::getenv("MACNLOPS_INVERSE_BORN"));
        const auto& z=decision.density.sudakov;
        std::vector<double> pb(4*(nlegs-1));macnlops_diag_born(pb.data());
        double htBorn=0.,htReal=0.,htInput=0.;
        for(int j=2;j<nlegs-1;++j)htBorn+=std::sqrt(std::max(0.,pb[4*j]*pb[4*j]-pb[4*j+3]*pb[4*j+3]));
        for(const auto& p:particles)if(p.status==1)htReal+=std::sqrt(std::max(0.,p.e*p.e-p.pz*p.pz));
        for(const auto& p:input.hardParticles)if(p.status==1)htInput+=std::sqrt(std::max(0.,p.e*p.e-p.pz*p.pz));
        diag<<std::setprecision(17)<<"{\"index\":"<<(counts["input_S"]+counts["input_H"]+1)
          <<",\"kind\":"<<kind<<",\"weight\":"<<input.weight<<",\"keep\":"<<decision.keep
          <<",\"muBorn\":"<<.5*htBorn<<",\"muReal\":"<<.5*htReal<<",\"muInput\":"<<.5*htInput
          <<",\"alphaK\":"<<decision.density.alphaK<<",\"pt\":"<<z.pt
          <<",\"q2fact\":["<<decision.density.q2fact[0]<<','<<decision.density.q2fact[1]
          <<"],\"ids\":["<<z.ids[0]<<','<<z.ids[1]<<"],\"x\":["<<z.x[0]<<','<<z.x[1]
          <<"],\"r\":"<<decision.density.r<<",\"d\":"<<decision.density.d<<",\"k\":"<<decision.density.k
          <<",\"kernel_scale\":"<<decision.kernelScale<<",\"pdf_ratio\":"<<decision.pdfRatio
          <<",\"A\":"<<decision.deltaSource<<",\"P\":"<<decision.deltaShower<<",\"born_p\":[";
        for(int j=0;j<nlegs-1;++j){if(j)diag<<',';diag<<'[';for(int k=0;k<4;++k){if(k)diag<<',';diag<<pb[4*j+k];}diag<<']';}
        diag<<"]}\n";
      }
'''
source=source.replace(needle,injection+needle)
(D/'InverseBornMatch.cc').write_text(source)
with (D/'inverse_build.log').open('w') as log:
 subprocess.run(['gfortran','-c','-I'+str(P),str(D/'born_getter.f90'),'-o',str(D/'born_getter.o')],cwd=D,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run(['g++','-O','-std=c++17','-I/nfs/home/rikkert/Downloads/pythia8/include','-c',str(D/'InverseBornMatch.cc'),'-o',str(D/'InverseBornMatch.o')],stdout=log,stderr=subprocess.STDOUT,check=True)
 link=(P/'macnlops_onestep.build.log').read_text().replace(' -o macnlops_onestep',' -o '+str(D/'extract_inverse_born')).replace('../../MCatNLO/srcPythia8/MacNLOPSMatch.o',str(D/'InverseBornMatch.o')+' '+str(D/'born_getter.o'))
 subprocess.run(link,shell=True,cwd=P,stdout=log,stderr=subprocess.STDOUT,check=True)
m=json.loads((B/'diagnostics/pdf_ratio/guard/manifest.json').read_text());cmd=m['command'];cmd[0]=str(D/'extract_inverse_born');cmd[cmd.index('--output')+1]=str(D/'inverse_replay.lhe');cmd[cmd.index('--summary')+1]=str(D/'inverse_replay.json')
env=dict(os.environ,LD_LIBRARY_PATH=str(B/'processes/Wbb_4FS/lib')+':/nfs/home/rikkert/Downloads/pythia8/lib:/nfs/home/rikkert/LHAPDF/lib',LHAPDF_DATA_PATH='/nfs/home/rikkert/LHAPDF/share/LHAPDF',PYTHIA8DATA='/nfs/home/rikkert/Downloads/pythia8/share/Pythia8/xmldoc',MACNLOPS_INVERSE_BORN=str(D/'inverse_born.jsonl'))
with (D/'inverse_run.log').open('w') as log:p=subprocess.run(cmd,cwd=m['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
assert p.returncode==0,p.returncode
rows=[json.loads(s) for s in (D/'inverse_born.jsonl').read_text().splitlines()]
s=[r for r in rows if r['kind']==1];h=[r for r in rows if r['kind']==2]
err=max(abs(r['muBorn']/r['muInput']-1) for r in s)
identical=(D/'inverse_replay.lhe').read_bytes()==(B/'diagnostics/pdf_ratio/guard/events.lhe').read_bytes()
report=dict(command=cmd,returncode=p.returncode,S_points=len(s),H_points=len(h),max_S_Born_vs_input_relative_mu_error=err,output_byte_identical=identical,notes='Isolated read-only p_born getter after density call, original production objects untouched.')
(D/'inverse_manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
