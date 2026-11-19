// Follow complete showers and check current IF top connections after emission.
#define main first_emission_reference_main
#include "check_pythia8_singletop_local.cc"
#undef main

class SoftHistoryCheck : public UserHooks {
public:
  int firstTop=0,laterTop=0,gluonTop=0,ffGluonTop=0,violations=0;
  bool canVetoISREmission() override { return true; }
  bool canVetoFSREmission() override { return true; }
  bool doVetoISREmission(int old,const Event& event,int system) override {
    if(system!=0 || event.back().id()!=21)return false;
    int mother=event.back().mother1(),daughter=event[mother].daughter2();
    if(event[mother].id()!=event[daughter].id())return false;
    for(int j=0;j<partonSystemsPtr->sizeOut(system);++j) {
      int f=partonSystemsPtr->getOut(system,j);
      if(event[f].idAbs()!=6)continue;
      bool connected=(event[daughter].col()!=0 &&
                      event[daughter].col()==event[f].col()) ||
                     (event[daughter].acol()!=0 &&
                      event[daughter].acol()==event[f].acol());
      if(!connected)continue;
      // Incoming gluon-top IF configurations are outside this quark patch.
      if(event[daughter].idAbs()==21) {++gluonTop;continue;}
      if(event[daughter].idAbs()>6)continue;
      Vec4 pa=event[mother].p(),pf=event[old+2+j].p(),k=event.back().p();
      double weight=1.-event[f].m2()*(pa*k)/(2.*(pa*pf)*(pf*k));
      if(!std::isfinite(weight) || weight < -1.e-8 || weight > 1.+1.e-8)
        ++violations;
      if(partonSystemsPtr->sizeOut(system)>2)++laterTop;
      else ++firstTop;
    }
    return false;
  }
  bool doVetoFSREmission(int old,const Event& event,int system,bool=false) override {
    if(system==0 && event.size()>old+2 && event[old].id()==21 &&
       event[old+1].id()==21 && event[old+2].idAbs()==6 &&
       event[old+2].isFinal())++ffGluonTop;
    return false;
  }
};

int main(int argc,char** argv) {
  if(argc<4 || argc>6)return 2;
  Pythia pythia(argv[1],false);
  pythia.setLHAupPtr(std::make_shared<SingleTopBorn>(std::atoi(argv[2]),std::atoi(argv[3])));
  auto check=std::make_shared<SoftHistoryCheck>();
  pythia.setUserHooksPtr(check);
  if(!configureLocal(pythia))return 3;
  for(const char* setting : {"Check:abortIfVeto=off", "BeamRemnants:primordialKT=off"})
    if(!pythia.readString(setting))return 3;
  int mode=argc>4 ? std::atoi(argv[4]) : 2;
  if(mode>=1 && !pythia.readString("SpaceShower:IFquarkCollinearCorrection=on"))return 3;
  if(mode>=2 && !pythia.readString("SpaceShower:IFtopSoftCorrection=on"))return 3;
  if(!pythia.flag("TimeShower:recoilDeadCone"))return 3;
  if(!pythia.init())return 4;
  int failed=0,accepted=0;
  int requested=argc>5 ? std::atoi(argv[5]) : 300;
  for(int i=0;i<requested;++i) {
    if(!pythia.next()) {++failed;continue;}
    ++accepted;
    int tops=0;
    for(int j=0;j<pythia.event.size();++j)
      if(pythia.event[j].idAbs()==6 && pythia.event[j].isFinal()) {
        ++tops;
        if(std::abs(pythia.event[j].m()-173.)>1.e-8)++check->violations;
      }
    if(tops!=1)++check->violations;
  }
  std::cout<<"FULL_LOCAL_SOFT accepted="<<accepted<<" failed="<<failed
    <<" first_top_IF="<<check->firstTop<<" later_top_IF="<<check->laterTop
    <<" gluon_top_IF="<<check->gluonTop<<" FF_gluon_top="<<check->ffGluonTop
    <<" violations="<<check->violations<<'\n';
  return check->violations || !accepted || !check->firstTop ||
    !check->laterTop || !check->ffGluonTop;
}
