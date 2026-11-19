// Actual Pythia first emissions from a stable-top Born colour configuration.
// Link to a stock or privately patched build. Record the first accepted branch,
// then terminate that diagnostic event through the standard event-veto hook.
#include "Pythia8/Pythia.h"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
using namespace Pythia8;

class SingleTopBorn : public LHAup {
public:
  SingleTopBorn(bool swapIn, bool antiIn) : swap(swapIn), anti(antiIn) {}
  bool setInit() override {
    setBeamA(2212,6500.); setBeamB(2212,6500.);
    setStrategy(3); addProcess(1,1.,0.,1.); return true;
  }
  bool setEvent(int = 0) override {
    setProcess(1,1.,300.,1./137.,0.118);
    int b=5, q=anti ? -1 : 2, jet=anti ? -2 : 1;
    auto incoming = [&](int id, int colour, double pz) {
      addParticle(id,-1,0,0,id>0 ? colour : 0,id<0 ? colour : 0,
                  0.,0.,pz,500.);
    };
    incoming(swap ? q : b,swap ? 502 : 501,500.);
    incoming(swap ? b : q,swap ? 501 : 502,-500.);
    double mass=173., k=(1.e6-mass*mass)/2000.;
    double px=k*std::sin(.6)*std::cos(.4), py=k*std::sin(.6)*std::sin(.4);
    double pz=k*std::cos(.6);
    addParticle(6,1,1,2,501,0,px,py,pz,(1.e6+mass*mass)/2000.,mass);
    addParticle(jet,1,1,2,jet>0 ? 502 : 0,jet<0 ? 502 : 0,-px,-py,-pz,k);
    return true;
  }
private:
  bool swap,anti;
};

class LocalCheck : public UserHooks {
public:
  int count=0, failures=0, fsr=0, gluon=0, conversion=0, ends=0, points=0;
  bool stopped=false;
  bool canVetoStep() override { return true; }
  int numberVetoStep() override { return 1; }
  bool doVetoStep(int, int nISR, int nFSR, const Event&) override {
    stopped=(nISR+nFSR==1);
    return stopped;
  }
  bool canVetoISREmission() override { return true; }
  bool canVetoFSREmission() override { return true; }
  bool doVetoFSREmission(int old, const Event& event, int system, bool=false) override {
    // The subsystem is updated after the hook; two outgoing Born quarks
    // identify a first emission independently of Info's cached counters.
    if(system==0 && partonSystemsPtr->sizeOut(system)==2) {
      ++fsr;
      std::cout<<"LOCAL_UNEXPECTED_FIRST_FSR old="<<old<<'\n';
      event.list();
      partonSystemsPtr->list();
    }
    return false;
  }
  bool doVetoISREmission(int old, const Event& event, int system) override {
    if(system!=0 || partonSystemsPtr->sizeOut(system)!=2)return false;
    ++count;
    int mother=event.back().mother1(), daughter=event[mother].daughter2();
    int end=-1;
    for(int j=0;j<partonSystemsPtr->sizeOut(system);++j) {
      int f=partonSystemsPtr->getOut(system,j);
      if ((event[daughter].col()!=0 && event[daughter].col()==event[f].col()) ||
          (event[daughter].acol()!=0 && event[daughter].acol()==event[f].acol()))end=j;
    }
    if(end<0) {++failures;return false;}
    int f=partonSystemsPtr->getOut(system,end);
    ends|=event[f].idAbs()==6 ? 1 : 2;
    // Massive emitted c/b conversion products are outside the analytic
    // massless-emission approximation; leave their shower evolution intact.
    if(event.back().id()!=21) {++conversion;return false;}
    ++gluon;
    int fnew=old+2+end;
    Vec4 pa=event[mother].p(), pf=event[fnew].p(), k=event.back().p();
    Vec4 ba=event[daughter].p(), bf=event[f].p();
    double z=ba.e()/pa.e();
    Vec4 residual=pf+k-(1.-z)*pa-bf;
    double error=std::abs(residual.e())+std::abs(residual.px())+
      std::abs(residual.py())+std::abs(residual.pz());
    if(error>1.e-6 || std::abs((z*pa-ba).e())>1.e-7)++failures;
    // Emit a small sample for the independent Fortran density comparison.
    if(points<25) {
      ++points;
      int beam=event[mother].pz()>0. ? 1 : 2;
      std::cout<<std::setprecision(17)<<"LOCAL_POINT "<<beam<<' '<<end+3<<' '
               <<event[f].m()<<' '<<(event.back().id()==21 ? 1 : 2);
      for(int j=0;j<2;++j) {
        int index=old+j;
        Vec4 p=event[index].p();
        std::cout<<' '<<p.e()<<' '<<p.px()<<' '<<p.py()<<' '<<p.pz();
      }
      for(int index : {old+2,old+3,event.size()-1}) {
        Vec4 p=event[index].p();
        std::cout<<' '<<p.e()<<' '<<p.px()<<' '<<p.py()<<' '<<p.pz();
      }
      std::cout<<' '<<z<<' '<<2.*(pa*k)<<' '<<2.*(ba*bf)<<'\n';
    }
    return false;
  }
};

bool configureLocal(Pythia& pythia) {
  const char* settings[]={
    "Beams:frameType=5", "PartonLevel:ISR=on", "PartonLevel:FSR=on",
    "PartonLevel:MPI=off", "HadronLevel:all=off", "Check:event=on",
    "Check:abortIfVeto=on",
    "SpaceShower:dipoleRecoil=on",
    "SpaceShower:pTmaxMatch=1", "SpaceShower:pTdampMatch=0",
    "SpaceShower:samePTasMPI=off", "SpaceShower:pT0parametrization=0",
    "SpaceShower:pT0Ref force=0.",
    "SpaceShower:rapidityOrder=off", "SpaceShower:alphaSuseCMW=off",
    "SpaceShower:alphaSvalue=0.118", "SpaceShower:alphaSorder=1",
    "TimeShower:alphaSvalue=0.118", "TimeShower:alphaSorder=1",
    "TimeShower:alphaSuseCMW=off",
    "SpaceShower:MEcorrections=off", "TimeShower:MEcorrections=off",
    "TimeShower:MEextended=off", "TimeShower:globalRecoil=off",
    "TimeShower:dampenBeamRecoil=off", "TimeShower:allowBeamRecoil=on",
    "SpaceShower:QEDshowerByQ=off", "SpaceShower:QEDshowerByL=off",
    "TimeShower:QEDshowerByQ=off", "TimeShower:QEDshowerByL=off",
    "TimeShower:QEDshowerByOther=off", "TimeShower:QEDshowerByGamma=off",
    "SpaceShower:weakShower=off", "TimeShower:weakShower=off",
    "6:mayDecay=off", "6:m0=173.",
    "PDF:pSet=15", "Random:setSeed=on", "Random:seed=172925",
    "Next:numberShowLHA=0", "Next:numberShowEvent=0", "Next:numberShowProcess=0",
    "Next:numberShowInfo=0", "Init:showChangedSettings=off",
    "Init:showChangedParticleData=off", "Init:showProcesses=off"};
  for(auto setting:settings)if(!pythia.readString(setting))return false;
  return true;
}

int main(int argc,char** argv) {
  if(argc!=4 && argc!=5)return 2;
  Pythia pythia(argv[1],false);
  pythia.setLHAupPtr(std::make_shared<SingleTopBorn>(std::atoi(argv[2]),std::atoi(argv[3])));
  auto check=std::make_shared<LocalCheck>();
  pythia.setUserHooksPtr(check);
  if(!configureLocal(pythia))return 3;
  if(argc==5 && std::atoi(argv[4]) && !pythia.readString(
       "SpaceShower:IFquarkCollinearCorrection=on"))return 3;
  if(argc==5 && std::atoi(argv[4])>1 && !pythia.readString(
       "SpaceShower:IFtopSoftCorrection=on"))return 3;
  if(!pythia.init())return 4;
  int failed=0;
  for(int i=0;i<300;++i) {
    check->stopped=false;
    if(!pythia.next() && !check->stopped)++failed;
    for(int j=0;j<pythia.event.size();++j)
      if(pythia.event[j].idAbs()==6 && pythia.event[j].isFinal() &&
         std::abs(pythia.event[j].m()-173.)>1.e-8)++check->failures;
  }
  std::cout<<"LOCAL_EMISSIONS first="<<check->count<<" gluon="<<check->gluon
           <<" conversion="<<check->conversion<<" both_ends="<<(check->ends==3)
           <<" first_fsr="<<check->fsr<<" failures="<<failed+check->failures<<'\n';
  return failed || check->failures || check->fsr ||
    !check->gluon || !check->conversion || check->ends!=3;
}
