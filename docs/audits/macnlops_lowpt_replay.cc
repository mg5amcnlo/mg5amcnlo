// Conditional Sudakov measurements for saved, arbitrary Born configurations.
// Uses the production profile and first-step hook. Each input LHE Born has a
// matching cutoff in CUTOFFS.txt. Usage: probe LHE CUTOFFS OUTPUT TRIALS SEED [fsr-only]
#include "MacNLOPS.h"

class BornReplay : public LHAup {
public:
  BornReplay(const std::vector<InputEvent>& input,int repeats):events(input),n(repeats) {}
  bool setInit() override {
    setBeamA(2212,6500.);setBeamB(2212,6500.);setStrategy(-4);
    addProcess(1,1.,0.,1.);return true;
  }
  bool setEvent(int=0) override {
    if (count>=int(events.size())*n) return false;
    const InputEvent&e=events[count++/n];
    setProcess(1,1.,e.scale,e.alphaEM,e.alphaS);
    scales=LHAscales(e.scale);
    for (const auto& s:e.dipoleScales)
      scales.attributes[dipoleName(s.first.first,s.first.second)]=s.second;
    infoPtr->scales=&scales;
    for(const auto&p:e.particles)
      addParticle(p.id,p.status,p.mother1,p.mother2,p.col,p.acol,
                  p.px,p.py,p.pz,p.e,p.m,p.lifetime,p.spin,e.scale);
    setIdX(e.particles[0].id,e.particles[1].id,e.particles[0].e/6500.,e.particles[1].e/6500.);
    return true;
  }
  const std::vector<InputEvent>& events;int n,count=0;LHAscales scales;
};

int main(int argc,char**argv) {try {
  if(argc!=6 && argc!=7)throw std::runtime_error("See source usage");
  const bool fsrOnly=argc==7 && std::string(argv[6])=="fsr-only";
  if(argc==7 && !fsrOnly)throw std::runtime_error("Unknown probe mode");
  int repeats=std::stoi(argv[4]);
  Options o;o.seed=std::stoi(argv[5]);o.pdf="LHAPDF6:NNPDF40_nlo_as_01180/0";
  o.alphaSMZ=.1365;o.alphaSorder=1;o.flavourCount=5;
  o.beamEnergyA=o.beamEnergyB=6500.;o.mcatnloDelta=true;
  Pythia p("/nfs/home/rikkert/Downloads/pythia8/share/Pythia8/xmldoc",false);configure(p,o);
  validateQCDOnlySettings(p);
  if(fsrOnly)p.settings.flag("SpaceShower:QCDshower",false);
  RawReader raw(argv[1]);InputEvent input;std::vector<InputEvent> events;
  while(raw.next(input)) {
    prepareSourceMatching(input,true);validateHardInput(input,p);
    events.push_back(input);
  }
  std::ifstream cuts(argv[2]);std::vector<double> cutoff(events.size());
  for(auto&q:cutoff)if(!(cuts>>q))throw std::runtime_error("Missing cutoff");
  auto reader=std::make_shared<BornReplay>(events,repeats);
  auto hook=std::make_shared<FirstEmission>();configureHook(p,o,*hook);
  p.setLHAupPtr(reader);p.setUserHooksPtr(hook);
  if(!p.init())throw std::runtime_error("Initialization failed");
  if(!fsrOnly)validateQCDOnlySettings(p);
  validateSourceMatchingSettings(p,o);hook->captureBeamLimits();
  std::ofstream out(argv[3]);out<<"index,trials,below,cutoff\n"<<std::setprecision(17);
  for(std::size_t i=0;i<events.size();++i) {
    int below=0;
    for(int j=0;j<repeats;++j) {
      hook->reset();bool ok=p.next();validateNominalWeights(p);
      if(ok || !hook->finished || reader->count!=int(i)*repeats+j+1)
        throw std::runtime_error("First step failed or retried a Born");
      if(!hook->seen || hook->pt<cutoff[i])++below;
    }
    out<<i<<','<<repeats<<','<<below<<','<<cutoff[i]<<'\n';out.flush();
  }
  if(!out)throw std::runtime_error("Output failed");
  p.settings.writeFile(std::string(argv[3])+".settings.cmnd",true);
  std::cout<<"Checked "<<reader->count<<" first steps, one Born consumed per step\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
