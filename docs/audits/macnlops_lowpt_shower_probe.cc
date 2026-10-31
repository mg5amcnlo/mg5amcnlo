// Repeated fixed singlet Borns with the production MAcNLOPS first-step engine.
// Usage: probe OUTPUT.csv PID MASS START TRIALS SEED
#include "MacNLOPS.h"

class FixedBorn : public LHAup {
public:
  FixedBorn(int id, double mass, double start):pid(id),m(mass),Q(start) {}
  bool setInit() override {
    setBeamA(2212,6500.); setBeamB(2212,6500.); setStrategy(-4);
    addProcess(1,1.,0.,1.); return true;
  }
  bool setEvent(int=0) override {
    setProcess(1,1.,Q,1./137.,.118);
    scales=LHAscales(Q);
    scales.attributes["scalup_1_2"]=Q;
    scales.attributes["scalup_2_1"]=Q;
    infoPtr->scales=&scales;
    addParticle(pid,-1,0,0,501,pid==21?502:0,0.,0.,.5*m,.5*m,0.,0.,9.,Q);
    addParticle(pid==21?21:-pid,-1,0,0,pid==21?502:0,501,0.,0.,-.5*m,.5*m,0.,0.,9.,Q);
    if (pid==21) addParticle(25,1,1,2,0,0,0.,0.,0.,m,m,0.,9.,Q);
    else {
      addParticle(11,1,1,2,0,0,.5*m,0.,0.,.5*m,0.,0.,9.,Q);
      addParticle(-11,1,1,2,0,0,-.5*m,0.,0.,.5*m,0.,0.,9.,Q);
    }
    setIdX(pid,pid==21?21:-pid,m/13000.,m/13000.);
    ++consumed; return true;
  }
  int pid; double m,Q; LHAscales scales; long consumed=0;
};

int main(int argc,char**argv) {try {
  if(argc!=7) throw std::runtime_error("See source usage");
  int pid=std::stoi(argv[2]),trials=std::stoi(argv[5]);
  double mass=std::stod(argv[3]),Q=std::stod(argv[4]);
  Options options; options.seed=std::stoi(argv[6]);
  options.pdf="LHAPDF6:NNPDF40_nlo_as_01180/0";
  options.alphaSMZ=.1365; options.alphaSorder=1; options.flavourCount=5;
  options.beamEnergyA=options.beamEnergyB=6500.;options.mcatnloDelta=true;
  Pythia p("/nfs/home/rikkert/Downloads/pythia8/share/Pythia8/xmldoc",false);configure(p,options);
  auto born=std::make_shared<FixedBorn>(pid,mass,Q);
  auto hook=std::make_shared<FirstEmission>();configureHook(p,options,*hook);
  p.setLHAupPtr(born);p.setUserHooksPtr(hook);
  if(!p.init()) throw std::runtime_error("Initialization failed");
  validateQCDOnlySettings(p);validateSourceMatchingSettings(p,options);
  hook->captureBeamLimits();
  std::ofstream output(argv[1]);output<<std::setprecision(17);
  output<<"evolution,physical,stop1,stop2,emitted_id\n";
  for(int i=0;i<trials;++i) {
    hook->reset();bool ok=p.next();validateNominalWeights(p);
    if(ok || !hook->finished || born->consumed!=i+1)
      throw std::runtime_error("First step failed or retried a Born");
    if(!hook->seen) {output<<"0,0,0,0,0\n";continue;}
    const Event&e=hook->snapshot;
    Vec4 hard,rad;
    int emitted=0;
    for(int j:hook->outgoing) {
      if(e[j].id()==25 || std::abs(e[j].id())==11) hard+=e[j].p();
      else {rad=e[j].p();emitted=e[j].id();}
    }
    Vec4 a=e[hook->incoming[0]].p(),b=e[hook->incoming[1]].p();
    const double z=(a+b-rad).m2Calc()/(a+b).m2Calc();
    double t1=std::sqrt(std::max(0.,-(a-rad).m2Calc()*(1.-z)));
    double t2=std::sqrt(std::max(0.,-(b-rad).m2Calc()*(1.-z)));
    output<<hook->pt<<','<<hard.pT()<<','<<t1<<','<<t2<<','<<emitted<<'\n';
  }
  if(!output)throw std::runtime_error("Output failed");
  p.settings.writeFile(std::string(argv[1])+".settings.cmnd",true);
  std::cout<<"Checked "<<trials<<" first steps and one Born consumption per step\n";
  return 0;
} catch(const std::exception&e) {std::cerr<<e.what()<<'\n';return 1;}}
