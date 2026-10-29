// Direct Simple-shower API probabilities with recovered table settings.
// Link with vendor/SudGen/pythia_interface_for_sudgen.cc.
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
int main(int argc, char** argv) {
  if (argc != 3) return 1;
  int mode = std::stoi(argv[1]);
  double masses[21] = {.33,.33,.5,1.5,4.8,0,0,0,0,0,
    .000510998928,0,.1056583715,0,1.77682,0,0,0,0,0,0};
  pythia_init_(masses);
  for (std::string sector : {"SpaceShower:", "TimeShower:"}) {
    assert(pythia.readString(sector+"alphaSvalue = "+
      (sector=="SpaceShower:"?"0.130":"0.1365")));
    assert(pythia.readString(sector+"alphaSorder = 1"));
    assert(pythia.readString(sector+"alphaSuseCMW = off"));
    assert(pythia.readString(sector+"renormMultFac = 1."));
    assert(pythia.readString(sector+"factorMultFac = 1."));
    assert(pythia.readString(sector+"MEcorrections = off"));
    assert(pythia.readString(sector+"phiPolAsym = off"));
    assert(pythia.readString(sector+"phiPolAsymHard = off"));
  }
  for (int i=1;i<=4;++i)
    assert(pythia.readString(std::to_string(i)+":m0 = 0."));
  assert(pythia.readString("5:m0 = 4.8"));
  assert(pythia.readString("PDF:pSet = 13"));
  assert(pythia.readString("SpaceShower:nQuarkIn = 5"));
  assert(pythia.readString("StandardModel:alphaSnfmax = 6"));
  assert(pythia.readString("SpaceShower:rapidityOrder = off"));
  assert(pythia.readString("SpaceShower:phiIntAsym = off"));
  assert(pythia.readString("SpaceShower:samePTasMPI = off"));
  pythia.settings.forceParm("SpaceShower:pT0Ref", 0.);
  assert(pythia.readString("SpaceShower:pdfMode = "+std::to_string(mode)));
  assert(pythia.init()); assert(pythia.next());
  auto model=pythia.getShowerModelPtr();
  assert(pythia.settings.mode("PartonShowers:model")==1);
  assert(dynamic_cast<Pythia8::SimpleShowerModel*>(model.get()));
  assert(dynamic_cast<Pythia8::SimpleSpaceShower*>(model->getSpaceShower().get()));
  assert(dynamic_cast<Pythia8::SimpleTimeShower*>(model->getTimeShower().get()));
  assert(pythia.settings.parm("SpaceShower:pT0Ref")==0.);
  for(int i=1;i<=4;++i) assert(pythia.particleData.m0(i)==0.);
  pythia.settings.writeFile(std::string(argv[2])+".settings.cmnd",true);
  std::cout<<std::setprecision(17)<<"SIMPLE_TYPES_CONFIRMED 1 1 1\n";
  double q,Q,m; int id,type,seed;
  while(std::cin>>q>>Q>>m>>id>>type>>seed) {
    double prob=0.,minimum=0.;
    pythia_get_no_emission_prob_(prob,Q,q,m,id,type,seed,minimum);
    assert(prob>=0. && prob<=1.);
    std::cout<<"RESULT "<<mode<<' '<<q<<' '<<Q<<' '<<m<<' '
      <<id<<' '<<type<<' '<<seed<<' '<<prob<<" 500000\n"<<std::flush;
  }
  pythia.stat();
}
