// Conditional first-QCD-emission trials for DY and gg -> stable H.
// Usage: probe XML generate dy|higgs PREFIX NBORN SEED
//        probe XML shower INPUT_LHE PREFIX REPEATS SEED
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
  double weight=1., q=125., aqed=1./137., as=.130;
};

void setting(Pythia& p, const std::string& s) {
  if (!p.readString(s)) throw std::runtime_error("Invalid setting: "+s);
}

void common(Pythia& p, int seed) {
  for (const std::string s : {"Print:quiet = on", "Random:setSeed = on",
       "PartonShowers:model = 1", "SpaceShower:pdfMode = 0", "TimeShower:pdfMode = 0",
       "PDF:pSet = 15", "25:m0 = 125.", "25:mayDecay = off",
       "1:m0 = 0.33", "2:m0 = 0.33", "3:m0 = 0.50", "4:m0 = 1.5", "5:m0 = 4.8",
       "HadronLevel:all = off", "PartonLevel:MPI = off",
       "BeamRemnants:primordialKT = off", "PartonLevel:Remnants = on",
       "TimeShower:pTmaxMatch = 1", "TimeShower:pTmaxFudge = 1.",
       "TimeShower:alphaSvalue = 0.1365", "TimeShower:alphaSorder = 1",
       "TimeShower:alphaSuseCMW = off", "TimeShower:dampenBeamRecoil = off",
       "TimeShower:globalRecoil = on", "TimeShower:nMaxGlobalRecoil = 1",
       "TimeShower:globalRecoilMode = 2", "TimeShower:nMaxGlobalBranch = 1",
       "TimeShower:nPartonsInBorn = -1", "TimeShower:limitPTmaxGlobal = on",
       "TimeShower:weightGluonToQuark = 1", "TimeShower:recoilDeadCone = on",
       "TimeShower:QEDshowerByQ = off", "TimeShower:QEDshowerByL = off",
       "TimeShower:QEDshowerByOther = off", "TimeShower:QEDshowerByGamma = off",
       "TimeShower:MEcorrections = off", "SpaceShower:pTmaxMatch = 1",
       "SpaceShower:pTmaxFudge = 1.", "SpaceShower:alphaSvalue = 0.130",
       "SpaceShower:alphaSorder = 1", "SpaceShower:alphaSuseCMW = off",
       "SpaceShower:rapidityOrder = off", "SpaceShower:QEDshowerByQ = off",
       "SpaceShower:QEDshowerByL = off", "SpaceShower:MEcorrections = off"}) setting(p,s);
  setting(p,"Random:seed = "+std::to_string(seed));
}

LHAParticle convert(const Particle& p, int status) {
  return LHAParticle(p.id(),status,status<0?0:1,status<0?0:2,p.col(),p.acol(),
    p.px(),p.py(),p.pz(),p.e(),p.m(),0.,9.,-1.);
}

void writeLHEHeader(std::ostream& o) {
  o<<"<LesHouchesEvents version=\"3.0\">\n<header></header>\n<init>\n"
   <<"2212 2212 6500 6500 0 0 0 0 3 1\n1 0 1 1\n</init>\n";
}

void writeEvent(std::ostream& o, const Born& b) {
  o<<"<event>\n"<<b.part.size()<<" 1 "<<b.weight<<' '<<b.q<<' '<<b.aqed<<' '<<b.as<<'\n';
  for(auto a:b.part) o<<a.idPart<<' '<<a.statusPart<<' '<<a.mother1Part<<' '
    <<a.mother2Part<<' '<<a.col1Part<<' '<<a.col2Part<<' '<<a.pxPart<<' '
    <<a.pyPart<<' '<<a.pzPart<<' '<<a.ePart<<' '<<a.mPart<<" 0 9\n";
  o<<"<scales scalup_1_2=\""<<b.q<<"\" scalup_2_1=\""<<b.q<<"\"/>\n</event>\n";
}

std::vector<Born> readBorn(const std::string& filename) {
  std::ifstream in(filename); if(!in)throw std::runtime_error("Cannot open LHE");
  std::string line;std::vector<Born> result;
  while(std::getline(in,line)) {
    if(line.find("<event>")==std::string::npos)continue;
    Born b;int n,id;std::getline(in,line);std::istringstream h(line);
    h>>n>>id>>b.weight>>b.q>>b.aqed>>b.as;
    if(n!=3 && n!=4)throw std::runtime_error("Need 3/4-leg singlet Born");
    for(int i=0;i<n;++i){std::getline(in,line);std::istringstream r(line);LHAParticle a;
      r>>a.idPart>>a.statusPart>>a.mother1Part>>a.mother2Part>>a.col1Part>>a.col2Part
       >>a.pxPart>>a.pyPart>>a.pzPart>>a.ePart>>a.mPart>>a.tauPart>>a.spinPart;
      if(!r)throw std::runtime_error("Invalid particle record");
      if(i<2){if(a.statusPart!=-1 || std::abs(a.mPart)>1e-12)
          throw std::runtime_error("Expected massless incoming Born momenta");a.scalePart=b.q;}
      else if(a.statusPart!=1 || a.col1Part || a.col2Part)
        throw std::runtime_error("Expected colourless final Born particles");
      b.part.push_back(a);
    }
    while(std::getline(in,line) && line.find("</event>")==std::string::npos){}
    b.scales["scalup_1_2"]=b.q;b.scales["scalup_2_1"]=b.q;result.push_back(b);
  }
  if(result.empty())throw std::runtime_error("Empty LHE");return result;
}

class Replay : public LHAup {
public:
  Replay(const std::vector<Born>& b,int n):born(b),repeats(n){}
  bool setInit()override{setBeamA(2212,6500.);setBeamB(2212,6500.);setStrategy(3);addProcess(1);return true;}
  bool setEvent(int=0)override{
    if(next>=int(born.size())*repeats)return false;
    current=next++;const Born&b=born[current/repeats];setProcess(1,1.,b.q,b.aqed,b.as);
    scales=LHAscales(b.q);scales.attributes=b.scales;infoPtr->scales=&scales;
    for(auto a:b.part)addParticle(a);
    setIdX(b.part[0].idPart,b.part[1].idPart,b.part[0].ePart/6500.,b.part[1].ePart/6500.);
    return true;
  }
  const std::vector<Born>&born;int repeats,next=0,current=-1;LHAscales scales;
};

class FirstEmission : public UserHooks {
public:
  explicit FirstEmission(std::shared_ptr<Replay> r):replay(r){}
  double pt=0.,maxBornResidual=0.;int type=0;long processChecks=0;
  std::shared_ptr<Replay> replay;
  bool canVetoProcessLevel()override{return true;}
  bool doVetoProcessLevel(Event&e)override{
    const Born&b=replay->born[replay->current/replay->repeats];
    if(e.size()!=int(b.part.size())+3)throw std::runtime_error("Unexpected process record size");
    for(size_t k=0;k<b.part.size();++k){const auto&a=b.part[k];const auto&p=e[k+3];
      if(p.id()!=a.idPart || p.col()!=a.col1Part || p.acol()!=a.col2Part)
        throw std::runtime_error("Replay changed Born identity/colour");
      double denom=std::max(1.,a.ePart);
      maxBornResidual=std::max(maxBornResidual,std::abs(p.e()-a.ePart)/denom);
      maxBornResidual=std::max(maxBornResidual,std::abs(p.px()-a.pxPart)/denom);
      maxBornResidual=std::max(maxBornResidual,std::abs(p.py()-a.pyPart)/denom);
      maxBornResidual=std::max(maxBornResidual,std::abs(p.pz()-a.pzPart)/denom);
    }
    ++processChecks;return false;
  }
  bool canVetoStep()override{return true;}
  int numberVetoStep()override{return 1;}
  bool doVetoStep(int pos,int nisr,int nfsr,const Event&)override{
    if(nisr+nfsr!=1)return false;
    pt=infoPtr->pTnow();type=pos;
    if(nisr!=1 || nfsr!=0 || pos!=2)throw std::runtime_error("First QCD branching is not ISR");
    return false;
  }
  void reset(){pt=0.;type=0;}
};

void writeBorn(std::ostream&o,const Born&b,int event){
  o<<"{\"event\":"<<event<<",\"weight\":"<<b.weight<<",\"scalup\":"<<b.q
   <<",\"x\":["<<b.part[0].ePart/6500.<<','<<b.part[1].ePart/6500.<<"],\"p\":[";
  for(size_t i=0;i<b.part.size();++i){if(i)o<<',';const auto&a=b.part[i];
    o<<'['<<a.ePart<<','<<a.pxPart<<','<<a.pyPart<<','<<a.pzPart<<']';}
  o<<"],\"ids\":[";for(size_t i=0;i<b.part.size();++i){if(i)o<<',';o<<b.part[i].idPart;}
  o<<"],\"colors\":[";for(size_t i=0;i<b.part.size();++i){if(i)o<<',';o<<'['<<b.part[i].col1Part<<','<<b.part[i].col2Part<<']';}
  o<<"],\"masses\":[";for(size_t i=0;i<b.part.size();++i){if(i)o<<',';o<<b.part[i].mPart;}
  o<<"],\"starts\":[";for(size_t i=0;i<b.part.size();++i){if(i)o<<',';o<<'[';
    for(size_t j=0;j<b.part.size();++j){if(j)o<<',';o<<((i<2 && j<2 && i!=j)?b.q:-1.);}o<<']';}
  o<<"]}\n";
}

int main(int argc,char**argv){try{
  if(argc!=7)throw std::runtime_error("See source usage");
  std::string xml=argv[1],mode=argv[2],input=argv[3],prefix=argv[4];
  int count=std::stoi(argv[5]),seed=std::stoi(argv[6]);Pythia p(xml,false);common(p,seed);
  if(mode=="generate"){
    setting(p,"Beams:eCM = 13000.");setting(p,"PartonLevel:all = off");
    if(input=="dy"){
      setting(p,"WeakSingleBoson:ffbar2gmZ = on");setting(p,"WeakZ0:gmZmode = 0");
      setting(p,"23:onMode = off");setting(p,"23:onIfMatch = 11 -11");
      setting(p,"PhaseSpace:mHatMin = 66.");setting(p,"PhaseSpace:mHatMax = 116.");
    }else if(input=="higgs")setting(p,"HiggsSM:gg2H = on");
    else throw std::runtime_error("Unknown hard process");
    if(!p.init())throw std::runtime_error("Born initialization failed");
    p.settings.writeFile(prefix+".generation.settings.cmnd",true);
    std::ofstream fixed(prefix+"_fixed.lhe"),damped(prefix+"_damped.lhe");
    fixed<<std::setprecision(17);damped<<std::setprecision(17);writeLHEHeader(fixed);writeLHEHeader(damped);
    std::mt19937_64 rng(static_cast<unsigned>(seed)^0x94e623a);std::uniform_real_distribution<double> uniform(0.,1.);
    double maxResidual=0.;
    for(int k=0;k<count;++k){
      if(!p.next())throw std::runtime_error("Born generation failed");
      Born b;b.aqed=p.info.alphaEM();b.as=p.info.alphaS();
      b.part={convert(p.process[3],-1),convert(p.process[4],-1)};
      if(input=="dy"){
        int em=-1,ep=-1;for(int i=5;i<p.process.size();++i){if(!p.process[i].isFinal())continue;
          if(p.process[i].id()==11)em=i;if(p.process[i].id()==-11)ep=i;}
        if(em<0 || ep<0)throw std::runtime_error("Missing Drell-Yan leptons");
        b.part.push_back(convert(p.process[em],1));b.part.push_back(convert(p.process[ep],1));
        if(std::abs(b.part[0].idPart)>5 || b.part[0].idPart!=-b.part[1].idPart)
          throw std::runtime_error("Unexpected Drell-Yan incoming channel");
      }else{
        int ih=-1;for(int i=5;i<p.process.size();++i)if(p.process[i].id()==25 && p.process[i].isFinal())ih=i;
        if(ih<0 || b.part[0].idPart!=21 || b.part[1].idPart!=21)throw std::runtime_error("Not gg->H");
        b.part.push_back(convert(p.process[ih],1));
      }
      // Internal Born kinematics must already be massless for the incoming partons.
      for(int i=0;i<2;++i)if(std::abs(b.part[i].mPart)>1e-12 || b.part[i].pxPart || b.part[i].pyPart ||
          std::abs(b.part[i].ePart-std::abs(b.part[i].pzPart))>1e-9*b.part[i].ePart)
        throw std::runtime_error("Massive internal incoming Born");
      Vec4 total;for(size_t i=2;i<b.part.size();++i){auto&a=b.part[i];total+=Vec4(a.pxPart,a.pyPart,a.pzPart,a.ePart);}
      b.q=total.mCalc();
      if(input=="dy" && !(b.q>66. && b.q<116.))throw std::runtime_error("DY mass cut violated");
      if(input=="higgs" && !(b.q>0.))throw std::runtime_error("Invalid Higgs mass");
      maxResidual=std::max(maxResidual,std::abs(total.e()-b.part[0].ePart-b.part[1].ePart)/total.e());
      maxResidual=std::max(maxResidual,std::abs(total.pz()-b.part[0].pzPart-b.part[1].pzPart)/total.e());
      writeEvent(fixed,b);double r=uniform(rng),u=std::sqrt(r)/(std::sqrt(r)+std::sqrt(1.-r));
      b.q*=.1+.9*u;writeEvent(damped,b);
    }
    fixed<<"</LesHouchesEvents>\n";damped<<"</LesHouchesEvents>\n";
    std::cout<<std::setprecision(17)<<"GENERATION process="<<input<<" Born="<<count
      <<" maximum_relative_momentum_residual="<<maxResidual<<'\n';p.stat();
  }else if(mode=="shower"){
    auto born=readBorn(input);auto lha=std::make_shared<Replay>(born,count);auto hook=std::make_shared<FirstEmission>(lha);
    p.setLHAupPtr(lha);p.setUserHooksPtr(hook);setting(p,"Beams:frameType = 5");
    setting(p,"Beams:setProductionScalesFromLHEF = on");setting(p,"Beams:setDipoleShowerStartingScalesFromLHEF = on");
    if(!p.init())throw std::runtime_error("Shower initialization failed");
    auto sm=p.getShowerModelPtr();
    if(!sm || !dynamic_cast<SimpleShowerModel*>(sm.get()) ||
      !dynamic_cast<SimpleSpaceShower*>(sm->getSpaceShower().get()) ||
      !dynamic_cast<SimpleTimeShower*>(sm->getTimeShower().get()) ||
      p.settings.mode("PartonShowers:model")!=1 || p.settings.mode("SpaceShower:pdfMode")!=0 ||
      p.settings.mode("TimeShower:pdfMode")!=0)throw std::runtime_error("Not ordinary Simple shower");
    std::cout<<"PROOF SimpleShowerModel=1 SimpleSpaceShower=1 SimpleTimeShower=1 SpaceShower:pdfMode=0 TimeShower:pdfMode=0\n";
    p.settings.writeFile(prefix+".settings.cmnd",true);
    std::ofstream bo(prefix+".born.jsonl"),tr(prefix+".trials.jsonl");bo<<std::setprecision(17);tr<<std::setprecision(17);
    for(size_t k=0;k<born.size();++k)writeBorn(bo,born[k],int(k));
    long failures=0,emitted=0;
    for(size_t k=0;k<born.size()*count;++k){hook->reset();bool ok=p.next();
      if(lha->current!=int(k))throw std::runtime_error("Unexpected LHA consumption/retry");
      if(!ok){++failures;tr<<"{\"event\":"<<k/count<<",\"trial\":"<<k%count<<",\"ok\":false}\n";continue;}
      if(p.info.nISR()+p.info.nFSRinProc()==0)hook->reset();
      if(hook->pt>born[k/count].q*(1.+1e-10) || hook->pt<0.)throw std::runtime_error("First emission outside start interval");
      if(hook->pt>0.)++emitted;
      tr<<"{\"event\":"<<k/count<<",\"trial\":"<<k%count<<",\"ok\":true,\"pt\":"<<hook->pt<<",\"type\":"<<hook->type<<"}\n";
    }
    p.stat();std::cout<<std::setprecision(17)<<"AUDIT process_checks="<<hook->processChecks<<" maximum_relative_Born_replay_difference="
      <<hook->maxBornResidual<<" emitted="<<emitted<<" Born="<<born.size()<<" repeats="<<count<<" failures="<<failures<<'\n';
    if(failures || hook->processChecks!=long(born.size()*count) || hook->maxBornResidual>1e-10)return 2;
  }else throw std::runtime_error("Unknown mode");
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
