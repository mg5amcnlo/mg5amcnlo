// Whole-API normalization checks for all four SudGen dipole types.
// Link the unchanged production wrapper; no Pythia library edits.
#include "Pythia8/Pythia.h"
#include "Pythia8/ShowerModel.h"
#include "Pythia8/SimpleSpaceShower.h"
#include "Pythia8/SimpleTimeShower.h"
#include <cassert>
#include <iomanip>
#include <iostream>
extern "C" {
extern Pythia8::Pythia pythia;
void pythia_init_(double[21]);
void pythia_get_no_emission_prob_(double&, double&, double&, double&,
                                int&, int&, int&, double&);
}
int main(int argc,char**argv) {
  if(argc!=2)return 1;
  double masses[21]={.33,.33,.5,1.5,4.8,0,0,0,0,0,
    .000510998928,0,.1056583715,0,1.77682,0,0,0,0,0,0};
  pythia_init_(masses);
  for(std::string sector:{"SpaceShower:","TimeShower:"}) {
    assert(pythia.readString(sector+"alphaSvalue = "+
      (sector=="SpaceShower:"?"0.130":"0.1365")));
    for(std::string value:{"alphaSorder = 1","alphaSuseCMW = off",
        "renormMultFac = 1.","factorMultFac = 1.","MEcorrections = off",
        "phiPolAsym = off","phiPolAsymHard = off"})
      assert(pythia.readString(sector+value));
  }
  for(int i=1;i<=4;++i)assert(pythia.readString(std::to_string(i)+":m0 = 0."));
  for(std::string setting:{"5:m0 = 4.8","PDF:pSet = 13",
      "PartonShowers:model = 1","SpaceShower:nQuarkIn = 5",
      "StandardModel:alphaSnfmax = 6","SpaceShower:rapidityOrder = off",
      "SpaceShower:phiIntAsym = off","SpaceShower:samePTasMPI = off",
      "SpaceShower:pdfMode = 1","TimeShower:pdfMode = 0",
      "TimeShower:globalRecoil = off","TimeShower:dampenBeamRecoil = on",
      "TimeShower:weightGluonToQuark = 4",
      "TimeShower:QEDshowerByGamma = off","SpaceShower:QEDshowerByQ = off",
      "SpaceShower:QEDshowerByL = off"}) assert(pythia.readString(setting));
  pythia.settings.forceParm("SpaceShower:pT0Ref",0.);
  assert(pythia.init());assert(pythia.next());
  auto model=pythia.getShowerModelPtr();
  assert(dynamic_cast<Pythia8::SimpleShowerModel*>(model.get()));
  assert(dynamic_cast<Pythia8::SimpleSpaceShower*>(model->getSpaceShower().get()));
  assert(dynamic_cast<Pythia8::SimpleTimeShower*>(model->getTimeShower().get()));
  pythia.settings.writeFile(std::string(argv[1])+".settings.cmnd",true);
  std::cout<<std::setprecision(17)<<"SIMPLE_TYPES_CONFIRMED 1 1 1\n";
  double q,Q,m;int id,type,seed;
  while(std::cin>>q>>Q>>m>>id>>type>>seed) {
    double prob=0.,minimum=0.;
    pythia_get_no_emission_prob_(prob,Q,q,m,id,type,seed,minimum);
    assert(prob>=0.&&prob<=1.);
    int n=(type==1||type==3)?500000:10000;
    std::cout<<"RESULT "<<q<<' '<<Q<<' '<<m<<' '<<id<<' '<<type<<' '
      <<seed<<' '<<prob<<' '<<n<<std::endl;
  }
  pythia.stat();
}
