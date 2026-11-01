from pathlib import Path
import os,subprocess,json
D=Path(__file__).resolve().parent;B=D.parents[1]
s=(D/'conditional_survival.cc').read_text()
s=s.replace(' for(const auto&point:points){const int pointSeed=', ' std::ofstream local(std::string(argv[2])+".local.csv");local<<std::setprecision(17)<<"point,id,pt,z\\n";\n for(const auto&point:points){const int pointSeed=')
needle='if(!hook->seen || hook->pt<point.cutoff)++below;'
inject=r'''
if(hook->seen && hook->type==3) {
 int ib=0,ia=0,ig=0;
 for(int j=1;j<hook->snapshot.size();++j)if(hook->snapshot[j].status()==51){ib=hook->snapshot[j].mother1();ia=hook->snapshot[ib].daughter1();ig=hook->snapshot[ib].daughter2();}
 if(!ib || !ia || !ig)throw std::runtime_error("Missing FSR branch");
 const auto& old=hook->snapshot[ib];const auto& rad=hook->snapshot[ia];const auto& em=hook->snapshot[ig];
 Vec4 total(point.born[0].px+point.born[1].px,point.born[0].py+point.born[1].py,point.born[0].pz+point.born[1].pz,point.born[0].e+point.born[1].e);
 const Vec4 pair=rad.p()+em.p();const double z=1.-(total*em.p())/((total*pair)*(1.-old.m2()/pair.m2Calc()));
 local<<point.id<<','<<old.id()<<','<<hook->pt<<','<<z<<'\n';
}
'''
assert s.count(needle)==1;s=s.replace(needle,inject+needle)
(D/'local_fsr.cc').write_text(s)
with (D/'local_fsr_build.log').open('w') as log:
 subprocess.run(['g++','-O2','-std=c++17','-I/nfs/home/rikkert/Downloads/pythia8/include',str(D/'local_fsr.cc'),'-L/nfs/home/rikkert/Downloads/pythia8/lib','-Wl,-rpath,/nfs/home/rikkert/Downloads/pythia8/lib','-lpythia8','-ldl','-o',str(D/'local_fsr')],stdout=log,stderr=subprocess.STDOUT,check=True)
env=dict(os.environ,LD_LIBRARY_PATH='/nfs/home/rikkert/Downloads/pythia8/lib:/nfs/home/rikkert/LHAPDF/lib',LHAPDF_DATA_PATH='/nfs/home/rikkert/LHAPDF/share/LHAPDF',PYTHIA8DATA='/nfs/home/rikkert/Downloads/pythia8/share/Pythia8/xmldoc')
cmd=[str(D/'local_fsr'),str(D/'points_FSR.dat'),str(D/'local_fsr.csv'),'6000','4011391',str(B/'processes/Wbb_4FS/Cards/macnlops_shower.cmnd'),'FSR']
with (D/'local_fsr_run.log').open('w') as log:run=subprocess.run(cmd,cwd=D,env=env,stdout=log,stderr=subprocess.STDOUT)
print(run.returncode)
(D/'local_fsr_manifest.json').write_text(json.dumps(dict(command=cmd,returncode=run.returncode),indent=2)+'\n')
