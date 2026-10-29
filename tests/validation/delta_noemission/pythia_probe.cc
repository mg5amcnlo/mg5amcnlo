// Standalone first-accepted-emission measurement, Pythia 8.313+.
// Build: c++ -std=c++17 -O2 pythia_probe.cc -I$PYTHIA/include \
//   -L$PYTHIA/lib -Wl,-rpath,$PYTHIA/lib -lpythia8 -o pythia_probe
// Usage: pythia_probe XMLDIR generate born.lhe N SEED Q [PDFSET]
//        pythia_probe XMLDIR shower born.lhe output-prefix REPEATS SEED [PDFSET]
//                    [both|isr|fsr] [SETTINGS_OVERRIDE_FILE]
#include "Pythia8/Pythia.h"
#include "Pythia8/ShowerModel.h"
#include <fstream>
#include <iomanip>
#include <random>
#include <regex>
#include <stdexcept>
using namespace Pythia8;

struct Born {
  std::vector<LHAParticle> part;
  std::map<std::string,double> scales;
  double weight=1., q=173., aqed=1./137., as=.118;
};

void common(Pythia& p, int seed, const std::string& pdf) {
  for (const std::string s : {"Print:quiet = on", "Random:setSeed = on",
       "PartonShowers:model = 1", "SpaceShower:pdfMode = 0", "TimeShower:pdfMode = 0",
       "6:m0 = 173.", "6:mWidth = 0.", "6:doForceWidth = on", "6:mayDecay = off",
       "1:m0 = 0.33", "2:m0 = 0.33", "3:m0 = 0.50", "4:m0 = 1.5", "5:m0 = 4.8",
       "HadronLevel:all = off", "PartonLevel:MPI = off",
       "BeamRemnants:primordialKT = off", "PartonLevel:Remnants = on",
       "TimeShower:pTmaxMatch = 1", "TimeShower:pTmaxFudge = 1.",
       "TimeShower:alphaSvalue = 0.118", "TimeShower:alphaSorder = 1",
       "TimeShower:alphaSuseCMW = off", "TimeShower:dampenBeamRecoil = off",
       "TimeShower:globalRecoil = on", "TimeShower:nMaxGlobalRecoil = 1",
       "TimeShower:globalRecoilMode = 2", "TimeShower:nMaxGlobalBranch = 1",
       "TimeShower:nPartonsInBorn = -1", "TimeShower:limitPTmaxGlobal = on",
       "TimeShower:weightGluonToQuark = 1", "TimeShower:recoilDeadCone = on",
       "TimeShower:QEDshowerByQ = off", "TimeShower:QEDshowerByL = off",
       "TimeShower:QEDshowerByOther = off",
       "TimeShower:QEDshowerByGamma = off", "TimeShower:MEcorrections = off",
       "SpaceShower:pTmaxMatch = 1", "SpaceShower:pTmaxFudge = 1.",
       "SpaceShower:alphaSvalue = 0.118", "SpaceShower:alphaSorder = 1",
       "SpaceShower:alphaSuseCMW = off", "SpaceShower:rapidityOrder = off",
       "SpaceShower:QEDshowerByQ = off", "SpaceShower:QEDshowerByL = off",
       "SpaceShower:MEcorrections = off"})
    if (!p.readString(s)) throw std::runtime_error("Invalid setting: "+s);
  if (!p.readString("Random:seed = "+std::to_string(seed)) ||
      !p.readString("PDF:pSet = "+pdf)) throw std::runtime_error("Invalid seed/PDF");
}

std::vector<Born> readBorn(const std::string& filename, double& eA, double& eB) {
  std::ifstream in(filename); if (!in) throw std::runtime_error("Cannot open LHE");
  std::vector<Born> result; std::string line;
  while (std::getline(in,line)) {
    if (line.find("<init>") != std::string::npos) {
      std::getline(in,line); std::istringstream h(line); int a,b;
      h>>a>>b>>eA>>eB;
      if (a!=2212 || b!=2212) throw std::runtime_error("Need pp beams");
    }
    if (line.find("<event>") == std::string::npos) continue;
    Born b; int n,idproc; std::getline(in,line); std::istringstream h(line);
    h>>n>>idproc>>b.weight>>b.q>>b.aqed>>b.as;
    if (n!=4) throw std::runtime_error("Need four-leg Born ttbar event");
    for (int i=0;i<n;++i) {
      std::getline(in,line); std::istringstream r(line); LHAParticle a;
      r>>a.idPart>>a.statusPart>>a.mother1Part>>a.mother2Part>>a.col1Part>>a.col2Part
       >>a.pxPart>>a.pyPart>>a.pzPart>>a.ePart>>a.mPart>>a.tauPart>>a.spinPart;
      if (!r) throw std::runtime_error("Invalid particle line"); b.part.push_back(a);
    }
    if (b.part[0].statusPart!=-1 || b.part[1].statusPart!=-1 ||
        b.part[2].idPart!=6 || b.part[3].idPart!=-6)
      throw std::runtime_error("Born order must be beam A, beam B, top, antitop");
    while (std::getline(in,line) && line.find("</event>")==std::string::npos) {
      std::regex attr("scalup_([0-9]+)_([0-9]+)\\s*=\\s*['\"]([^'\"]+)['\"]");
      for (std::sregex_iterator it(line.begin(),line.end(),attr),end;it!=end;++it) {
        int i=std::stoi((*it)[1]); double q=std::stod((*it)[3]);
        b.scales["scalup_"+(*it)[1].str()+"_"+(*it)[2].str()]=q;
        if(i<1 || i>4 || !(q>0.)) throw std::runtime_error("Invalid dipole starting scale");
        double& partq=b.part[i-1].scalePart;partq=partq>0.?std::min(partq,q):q;
      }
    }
    result.push_back(b);
  }
  if (result.empty()) throw std::runtime_error("No Born events");
  return result;
}

class Replay : public LHAup {
public:
  Replay(const std::vector<Born>& b, double ea, double eb, int n)
    : born(b), eA(ea),eB(eb), repeats(n) {}
  bool setInit() override { setBeamA(2212,eA); setBeamB(2212,eB);
    setStrategy(3); addProcess(1); return true; }
  bool setEvent(int=0) override {
    if (next>=int(born.size())*repeats) return false;
    current=next++; const Born& b=born[current/repeats];
    setProcess(1,1.,b.q,b.aqed,b.as);
    scaleStorage=LHAscales(b.q);scaleStorage.attributes=b.scales;
    infoPtr->scales=&scaleStorage;
    for (auto a:b.part) addParticle(a);
    setIdX(b.part[0].idPart,b.part[1].idPart,b.part[0].ePart/eA,b.part[1].ePart/eB);
    return true;
  }
  const std::vector<Born>& born; double eA,eB; int repeats,next=0,current=-1;
  LHAscales scaleStorage;
};

class FirstEmission : public UserHooks {
public:
  double pt=0.; int type=0; std::vector<Particle> real;
  bool canVetoStep() override {return true;}
  int numberVetoStep() override {return 1;}
  bool doVetoStep(int pos,int nisr,int nfsr,const Event& e) override {
    if (nisr+nfsr!=1) return false;
    pt=infoPtr->pTnow(); type=pos; real.clear();
    real.push_back(e[partonSystemsPtr->getInA(0)]);
    real.push_back(e[partonSystemsPtr->getInB(0)]);
    int it=-1,ia=-1,ie=-1;
    for (int k=0;k<partonSystemsPtr->sizeOut(0);++k) {
      int i=partonSystemsPtr->getOut(0,k);
      if (e[i].id()==6) it=i;
      else if (e[i].id()==-6) ia=i;
      else {if(ie!=-1) throw std::runtime_error("More than one emission");ie=i;}
    }
    if (it<0 || ia<0 || ie<0) throw std::runtime_error("Invalid first-emission system");
    for (int i:{it,ia,ie}) real.push_back(e[i]);
    return false; // Observe accepted emissions without altering shower probabilities.
  }
  void reset() {pt=0.;type=0;real.clear();}
};

void writeParticles(std::ostream& o, const std::vector<LHAParticle>& a) {
  o<<"\"p\":[";
  for(size_t i=0;i<a.size();++i) {if(i)o<<',';o<<'['<<a[i].ePart<<','<<a[i].pxPart<<','<<a[i].pyPart<<','<<a[i].pzPart<<']';}
  o<<"],\"ids\":[";for(size_t i=0;i<a.size();++i){if(i)o<<',';o<<a[i].idPart;}
  o<<"],\"colors\":[";for(size_t i=0;i<a.size();++i){if(i)o<<',';o<<'['<<a[i].col1Part<<','<<a[i].col2Part<<']';}
  o<<"],\"masses\":[";for(size_t i=0;i<a.size();++i){if(i)o<<',';o<<a[i].mPart;}o<<']';
}
LHAParticle convert(const Particle& p,int status) {
  return LHAParticle(p.id(),status,status<0?0:1,status<0?0:2,p.col(),p.acol(),
    p.px(),p.py(),p.pz(),p.e(),p.m(),0.,9.,-1.);
}

int main(int argc,char** argv) {try {
  if(argc<7) throw std::runtime_error("See source usage");
  std::string xml=argv[1],mode=argv[2]; Pythia p(xml,false);
  if(mode=="generate") {
    int n=std::stoi(argv[4]),seed=std::stoi(argv[5]);
    bool damped=std::string(argv[6])=="damped";
    double q=damped?0.:std::stod(argv[6]);
    std::mt19937_64 scaleRng(static_cast<unsigned>(seed)^0x94e623a);
    std::uniform_real_distribution<double> uniform(0.,1.);
    common(p,seed,argc>7?argv[7]:"15");
    p.readString("Beams:eCM = 13000.");p.readString("Top:gg2ttbar = on");
    p.readString("Top:qqbar2ttbar = on");p.readString("PartonLevel:all = off");
    if(!p.init()) throw std::runtime_error("Generator init failed");
    std::ofstream o(argv[3]);o<<std::setprecision(17);
    o<<"<LesHouchesEvents version=\"1.0\">\n<header></header>\n<init>\n2212 2212 6500 6500 0 0 0 0 3 1\n1 0 1 1\n</init>\n";
    for(int k=0;k<n;++k) {
      if(!p.next()) throw std::runtime_error("Born generation failed");
      int it=-1,ia=-1;
      for(int i=0;i<p.process.size();++i){if(p.process[i].id()==6&&p.process[i].isFinal())it=i;if(p.process[i].id()==-6&&p.process[i].isFinal())ia=i;}
      if(it<0||ia<0)throw std::runtime_error("No stable top pair");
      if(damped) {double r=uniform(scaleRng);double u=std::sqrt(r)/(std::sqrt(r)+std::sqrt(1.-r));
        q=std::sqrt(173.*173.+p.process[it].pT2())*(.1+.9*u);}
      o<<"<event>\n4 1 1 "<<q<<" "<<p.info.alphaEM()<<" "<<p.info.alphaS()<<'\n';
      for(int i:{3,4,it,ia}){auto a=convert(p.process[i],i<5?-1:1);
        o<<a.idPart<<' '<<a.statusPart<<' '<<a.mother1Part<<' '<<a.mother2Part<<' '<<a.col1Part<<' '<<a.col2Part<<' '<<a.pxPart<<' '<<a.pyPart<<' '<<a.pzPart<<' '<<a.ePart<<' '<<a.mPart<<" 0 9\n";}
      o<<"</event>\n";
    }o<<"</LesHouchesEvents>\n";p.stat();
  } else if(mode=="shower") {
    int n=std::stoi(argv[5]),seed=std::stoi(argv[6]);
    common(p,seed,argc>7?argv[7]:"15");double ea=0.,eb=0.;
    auto born=readBorn(argv[3],ea,eb);auto lha=std::make_shared<Replay>(born,ea,eb,n);
    if(!born.front().scales.empty()) {
      p.readString("Beams:setProductionScalesFromLHEF = on");
      p.readString("Beams:setDipoleShowerStartingScalesFromLHEF = on");
    }
    if(argc>8) {std::string component=argv[8];
      if(component=="isr")p.readString("PartonLevel:FSR = off");
      else if(component=="fsr")p.readString("PartonLevel:ISR = off");
      else if(component!="both")throw std::runtime_error("Expected both/isr/fsr");}
    if(argc>9 && !p.readFile(argv[9]))
      throw std::runtime_error("Cannot read settings override file");
    if(p.settings.mode("PartonShowers:model")!=1 ||
       p.settings.mode("SpaceShower:pdfMode")!=0 ||
       p.settings.mode("TimeShower:pdfMode")!=0)
      throw std::runtime_error("Actual-shower probe requires Simple showers and ordinary PDF evolution");
    auto hook=std::make_shared<FirstEmission>(); p.setLHAupPtr(lha);p.setUserHooksPtr(hook);
    p.readString("Beams:frameType = 5");
    if(!p.init())throw std::runtime_error("Shower init failed");
    auto showerModel=p.getShowerModelPtr();
    if(!showerModel || !dynamic_cast<SimpleShowerModel*>(showerModel.get()) ||
       !dynamic_cast<SimpleSpaceShower*>(showerModel->getSpaceShower().get()) ||
       !dynamic_cast<SimpleTimeShower*>(showerModel->getTimeShower().get()))
      throw std::runtime_error("Simple shower runtime assertion failed");
    std::cout<<"PROOF SimpleShowerModel=1 SimpleSpaceShower=1 SimpleTimeShower=1 "
             <<"SpaceShower:pdfMode=0 TimeShower:pdfMode=0\n";
    std::string prefix=argv[4];p.settings.writeFile(prefix+".settings.cmnd",true);
    std::ofstream bo(prefix+".born.jsonl"),tr(prefix+".trials.jsonl");
    bo<<std::setprecision(17);tr<<std::setprecision(17);
    for(size_t k=0;k<born.size();++k){auto&b=born[k];bo<<"{\"event\":"<<k<<",\"weight\":"<<b.weight<<",\"scalup\":"<<b.q<<",\"x\":["<<b.part[0].ePart/ea<<','<<b.part[1].ePart/eb<<"],";writeParticles(bo,b.part);
      bo<<",\"starts\":[";for(int i=1;i<=4;++i){if(i>1)bo<<',';bo<<'[';for(int j=1;j<=4;++j){if(j>1)bo<<',';std::string key="scalup_"+std::to_string(i)+"_"+std::to_string(j);auto it=b.scales.find(key);bo<<(it==b.scales.end()?-1.:it->second);}bo<<']';}bo<<"]}\n";}
    long fail=0;
    for(size_t k=0;k<born.size()*n;++k){hook->reset();bool ok=p.next();
      if(lha->current!=int(k))throw std::runtime_error("Unexpected LHA event consumption/retry");
      if(!ok){++fail;tr<<"{\"event\":"<<k/n<<",\"trial\":"<<k%n<<",\"ok\":false}\n";continue;}
      if(p.info.nISR()+p.info.nFSRinProc()==0)hook->reset();
      tr<<"{\"event\":"<<k/n<<",\"trial\":"<<k%n<<",\"ok\":true,\"pt\":"<<hook->pt<<",\"type\":"<<hook->type;
      if(!hook->real.empty()){std::vector<LHAParticle>a;for(size_t i=0;i<hook->real.size();++i)a.push_back(convert(hook->real[i],i<2?-1:1));tr<<",\"real\":{\"emitted\":5,";writeParticles(tr,a);tr<<'}';}tr<<"}\n";
    }
    p.stat();std::cerr<<"Born="<<born.size()<<" repeats="<<n<<" failures="<<fail<<'\n';
    if(fail)return 2;
  } else throw std::runtime_error("Unknown mode");
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
