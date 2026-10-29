// Diagnostic access to prepared dipoles; no library source is modified.
#include "Pythia8/TimeShower.h"
#include "Pythia8/SimpleWeakShowerMEs.h"
#define private public
#include "Pythia8/SimpleTimeShower.h"
#undef private
#include "Pythia8/Pythia.h"
#include "Pythia8/ShowerModel.h"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <vector>
using namespace Pythia8;
extern "C" {extern Pythia pythia; void pythia_init_(double[21]);}
int main(int argc,char**argv){
 if(argc<2 || argc>3)return 1;
 double masses[21]={.33,.33,.5,1.5,4.8,0,0,0,0,0,.000510998928,0,.1056583715,0,1.77682,0,0,0,0,0,0};
 pythia_init_(masses);
 for(auto s:{"PartonShowers:model = 1","SpaceShower:pdfMode = 1","TimeShower:pdfMode = 0","PDF:pSet = 13",
   "SpaceShower:alphaSvalue = 0.130","TimeShower:alphaSvalue = 0.1365","SpaceShower:alphaSorder = 1","TimeShower:alphaSorder = 1",
   "SpaceShower:alphaSuseCMW = off","TimeShower:alphaSuseCMW = off","TimeShower:globalRecoil = off","TimeShower:weightGluonToQuark = 4",
   "TimeShower:dampenBeamRecoil = on","TimeShower:nMaxGlobalRecoil = 2","TimeShower:MEcorrections = off","SpaceShower:MEcorrections = off",
   "SpaceShower:phiIntAsym = off","SpaceShower:phiPolAsym = off","SpaceShower:phiPolAsymHard = off",
   "TimeShower:phiPolAsym = off","TimeShower:phiPolAsymHard = off","SpaceShower:rapidityOrder = off"})assert(pythia.readString(s));
 for(int id=1;id<=4;++id)assert(pythia.readString(std::to_string(id)+":m0 = 0."));
 pythia.settings.forceParm("SpaceShower:pT0Ref",0.);
 assert(pythia.init());assert(pythia.next());
 auto model=pythia.getShowerModelPtr();assert(dynamic_cast<SimpleShowerModel*>(model.get()));
 auto ts=dynamic_cast<SimpleTimeShower*>(model->getTimeShower().get());assert(ts);
 std::cout<<std::setprecision(17);
 const double Q=200.,q=20.,m=500.,s=7000.*7000.,x=.1;
 const int n=argc==3?std::stoi(argv[2]):200000;
 pythia.settings.writeFile(std::string(argv[1])+".settings.cmnd",true);
 for(int type:{2,4})for(int id:{2,21}){
   Event state;state.init("(line normalization)",&pythia.particleData);
   Vec4 pA(0,0,m/2,m/2),pB(0,0,-m/2,m/2);
   int colA=1,acolA=id==21?2:0;
   state.append(90,-11,0,0,0,0,0,0,pA+pB,m,m);
   if(type==2)state.append(id,23,0,0,0,0,colA,acolA,pA,0.,m);
   int idB=id==21?21:(type==4?id:-id);
   int colB=acolA,acolB=colA;if(type==4)std::swap(colB,acolB);
   state.append(idB,type==4?-21:23,0,0,0,0,colB,acolB,pB,0.,m);
   if(type==4)state.append(id,23,0,0,0,0,colA,acolA,pA,0.,m);
   auto& beamA=const_cast<BeamParticle&>(pythia.beamA);auto& beamB=const_cast<BeamParticle&>(pythia.beamB);
   beamA.clear();beamB.clear();beamA.append(1,id,x);beamB.append(2,idB,m*m/s/x);
   beamA.xfISR(0,id,x,Q*Q);beamB.xfISR(0,idB,m*m/s/x,Q*Q);
   beamA.pickValSeaComp();beamB.pickValSeaComp();
   pythia.partonSystems.clear();pythia.partonSystems.addSys();pythia.partonSystems.setInA(0,1);pythia.partonSystems.setInB(0,2);
   pythia.partonSystems.setSHat(0,m*m);pythia.partonSystems.setPTHat(0,Q);
   ts->useSystems=false;ts->prepare(0,state,true);
   std::vector<TimeDipoleEnd> full=ts->dipEnd;ts->dipEnd.clear();
   std::set<int> radiators;for(auto&dip:full)radiators.insert(dip.iRadiator);
   std::cout<<"COUNTS "<<type<<' '<<id<<' '<<full.size()<<' '<<radiators.size()<<'\n';
   for(auto&dip:full)std::cout<<"END "<<type<<' '<<id<<' '<<dip.iRadiator<<' '<<dip.iRecoiler<<' '<<dip.colType<<' '<<dip.isrType<<' '<<dip.pTmax<<'\n';
   int expected=(type==2?2:1)*(id==21?2:1);assert(full.size()==size_t(expected));
   std::vector<TimeDipoleEnd> radiator,line;
   for(auto&dip:full)if(dip.iRadiator==*radiators.begin())radiator.push_back(dip);
   line.push_back(full.front());
   int seq=0;
   for(auto vec:{line,radiator,full}){
     pythia.rndm.init(10283100+type*100+id*3+seq);
     ts->doTrialNow=true;model->getMergingHooks()->setShowerStoppingScale(q);
     int count=0;for(int trial=0;trial<n;++trial){double pt=ts->pTnext(vec,state,Q,q,m*m,id,type==2?1:-1,s,x);count+=pt<q;}
     std::cout<<"RESULT "<<type<<' '<<id<<' '<<seq<<' '<<vec.size()<<' '<<double(count)/n<<' '<<n<<'\n';++seq;
   }
   ts->useSystems=true;
 }
 pythia.stat();
}
