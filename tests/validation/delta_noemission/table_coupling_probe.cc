// Coupling/CMW audit of actual Pythia Simple Shower Sudakovs.
// Link with the unchanged vendor/SudGen/pythia_interface_for_sudgen.cc.
#include "Pythia8/Pythia.h"
#include "Pythia8/ShowerModel.h"
#include "Pythia8/SimpleSpaceShower.h"
#include "Pythia8/SimpleTimeShower.h"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
extern "C" {
extern Pythia8::Pythia pythia;
void pythia_init_(double[21]);
void pythia_get_no_emission_prob_(double&,double&,double&,double&,int&,int&,int&,double&);
}
int main(int argc,char**argv) {
 if(argc!=4) return 1;
 double alpha=std::stod(argv[1]);bool cmw=std::stoi(argv[2]);int pdfmode=std::stoi(argv[3]);
 double masses[21]={.33,.33,.5,1.5,4.8,0,0,0,0,0,.000510998928,0,.1056583715,0,1.77682,0,0,0,0,0,0};
 pythia_init_(masses);
 for(auto sector:{"SpaceShower:","TimeShower:"}) {
   assert(pythia.readString(std::string(sector)+"alphaSvalue = "+argv[1]));
   assert(pythia.readString(std::string(sector)+"alphaSorder = 1"));
   assert(pythia.readString(std::string(sector)+"alphaSuseCMW = "+(cmw?"on":"off")));
 }
 assert(pythia.readString("SpaceShower:pdfMode = "+std::string(argv[3])));
 assert(pythia.init()); assert(pythia.next());
 auto model=pythia.getShowerModelPtr();
 assert(pythia.settings.mode("PartonShowers:model")==1);
 assert(dynamic_cast<Pythia8::SimpleShowerModel*>(model.get()));
 assert(dynamic_cast<Pythia8::SimpleSpaceShower*>(model->getSpaceShower().get()));
 assert(dynamic_cast<Pythia8::SimpleTimeShower*>(model->getTimeShower().get()));
 Pythia8::AlphaStrong coupling;
 int nf=pythia.settings.mode("StandardModel:alphaSnfmax");
 coupling.init(alpha,1,nf,cmw);
 std::cout<<std::setprecision(16);
 std::cout<<"SIMPLE_TYPES_CONFIRMED 1 1 1\n";
 std::cout<<"COUPLING "<<alpha<<" "<<cmw<<" "<<pdfmode<<" "<<nf<<" "<<coupling.alphaS(400.)<<" "<<coupling.alphaS(40000.)<<" "<<coupling.facCMW(5)<<std::endl;
 for(int id:{2,21}) {
   double Q=200.,q=20.,m=500.,prob=0.,minimum=0.;int type=1,seed=1027271+(id==21);
   pythia_get_no_emission_prob_(prob,Q,q,m,id,type,seed,minimum);
   double err=std::sqrt(prob*(1.-prob)/500000.);
   double line=id==21?std::sqrt(prob):prob;
   double lineerr=id==21?err/(2.*line):err;
   std::cout<<"RESULT "<<alpha<<" "<<cmw<<" "<<pdfmode<<" "<<id<<" "<<prob<<" "<<err<<" "<<line<<" "<<lineerr<<std::endl;
 }
}
