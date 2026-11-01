from pathlib import Path
import os,subprocess,json,shutil,hashlib
R=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS')
B=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101')
D=B/'diagnosis_transfer_20261101/kernel'
P=B/'processes/Wbb_4FS/SubProcesses/P0_udx_bbxwp'
source=(R/'Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc').read_text()
needle='      const bool accepted = decisions() < decision.keep;'
injection=r'''
      if(kind==1 && decision.sudakov && decision.keep<1. &&
         (counts["input_S"]+counts["input_H"])%13==0) {
        static std::ofstream diagnostic(std::getenv("MACNLOPS_KERNEL_PROFILE"));
        const auto& z=decision.density.sudakov;
        SudakovProfile onlyISR=z, onlyFSR=z;
        onlyISR.nquarks=0;onlyISR.ngluons=0;
        onlyFSR.ids[0]=onlyFSR.ids[1]=0;onlyFSR.ngluons=0;
        diagnostic<<std::setprecision(17)<<"{\"index\":"<<(counts["input_S"]+counts["input_H"]+1)
          <<",\"weight\":"<<input.weight<<",\"keep\":"<<decision.keep
          <<",\"P\":"<<decision.deltaShower<<",\"P_ISR\":"<<showerMarginal(c,onlyISR)
          <<",\"P_FSR\":"<<showerMarginal(c,onlyFSR)<<",\"A\":"<<decision.deltaSource
          <<",\"pt\":"<<z.pt<<",\"ids\":["<<z.ids[0]<<','<<z.ids[1]
          <<"],\"x\":["<<z.x[0]<<','<<z.x[1]<<"],\"mass\":"<<z.mass
          <<",\"gluon_count\":"<<z.ngluons<<",\"quarks\":[";
        for(int j=0;j<z.nquarks;++j){if(j)diagnostic<<',';diagnostic<<'[';
          for(int k=0;k<4;++k){if(k)diagnostic<<',';diagnostic<<z.quarks[4*j+k];}diagnostic<<']';}
        diagnostic<<"],\"born\":[";for(std::size_t j=0;j<input.hardParticles.size();++j){if(j)diagnostic<<',';particleJSON(diagnostic,input.hardParticles[j]);}
        diagnostic<<"],\"Q\":[";for(int j=0;j<z.count;++j){if(j)diagnostic<<',';diagnostic<<'['<<z.q[j]<<','<<z.w[j]<<']';}
        diagnostic<<"]}\n";diagnostic.flush();
      }
'''
assert source.count(needle)==1
(D/'MacNLOPSMatch.cc').write_text(source.replace(needle,injection+needle))
shutil.copy2(R/'Template/NLO/MCatNLO/srcPythia8/MacNLOPS.h',D/'MacNLOPS.h')
with (D/'extract_build.log').open('w') as log:
 subprocess.run(['g++','-O','-std=c++17','-I/nfs/home/rikkert/Downloads/pythia8/include','-c',str(D/'MacNLOPSMatch.cc'),'-o',str(D/'MacNLOPSMatch.o')],stdout=log,stderr=subprocess.STDOUT,check=True)
 link=(P/'macnlops_onestep.build.log').read_text().replace(' -o macnlops_onestep',' -o '+str(D/'extract_profiles')).replace('../../MCatNLO/srcPythia8/MacNLOPSMatch.o',str(D/'MacNLOPSMatch.o'))
 subprocess.run(link,shell=True,cwd=P,stdout=log,stderr=subprocess.STDOUT,check=True)
m=json.loads((B/'diagnostics/pdf_ratio/guard/manifest.json').read_text())
cmd=m['command'];cmd[0]=str(D/'extract_profiles');cmd[cmd.index('--output')+1]=str(D/'replay.lhe');cmd[cmd.index('--summary')+1]=str(D/'replay.json')
env=dict(os.environ,LD_LIBRARY_PATH=str(B/'processes/Wbb_4FS/lib')+':/nfs/home/rikkert/Downloads/pythia8/lib:/nfs/home/rikkert/LHAPDF/lib',LHAPDF_DATA_PATH='/nfs/home/rikkert/LHAPDF/share/LHAPDF',PYTHIA8DATA='/nfs/home/rikkert/Downloads/pythia8/share/Pythia8/xmldoc',MACNLOPS_KERNEL_PROFILE=str(D/'profiles.jsonl'))
with (D/'extract_run.log').open('w') as log:
 result=subprocess.run(cmd,env=env,cwd=m['cwd'],stdout=log,stderr=subprocess.STDOUT)
manifest=dict(command=cmd,cwd=m['cwd'],returncode=result.returncode,production_source_sha256=hashlib.sha256(source.encode()).hexdigest(),isolated_source_sha256=hashlib.sha256((D/'MacNLOPSMatch.cc').read_bytes()).hexdigest(),notes='Read-only diagnostic replay of frozen shard with unchanged decisions and extra profile output; no source or Pythia mutation.')
(D/'extract_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(result.returncode,flush=True)
print('Profiles',len((D/'profiles.jsonl').read_text().splitlines()),flush=True)
print('Replay byte-identical:',(D/'replay.lhe').read_bytes()==(B/'diagnostics/pdf_ratio/guard/events.lhe').read_bytes(),flush=True)
