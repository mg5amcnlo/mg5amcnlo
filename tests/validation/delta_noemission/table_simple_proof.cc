// Runtime proof that the current unmodified SudGen wrapper uses Simple showers.
#include "Pythia8/Pythia.h"
#include "Pythia8/ShowerModel.h"
#include "Pythia8/SimpleSpaceShower.h"
#include "Pythia8/SimpleTimeShower.h"
#include <cassert>
#include <iostream>
#include <iomanip>
extern "C" {
extern Pythia8::Pythia pythia;
void pythia_init_(double[21]);
void pythia_get_no_emission_prob_(double&,double&,double&,double&,int&,int&,int&,double&);
}
int main() {
 double masses[21]={.33,.33,.5,1.5,4.8,0,0,0,0,0,.000510998928,0,.1056583715,0,1.77682,0,0,0,0,0,0};
 pythia_init_(masses);
 auto model=pythia.getShowerModelPtr();
 assert(model);
 assert(pythia.settings.mode("PartonShowers:model")==1);
 assert(dynamic_cast<Pythia8::SimpleShowerModel*>(model.get()));
 assert(dynamic_cast<Pythia8::SimpleSpaceShower*>(model->getSpaceShower().get()));
 assert(dynamic_cast<Pythia8::SimpleTimeShower*>(model->getTimeShower().get()));
 std::cout<<"PROOF PartonShowers:model=1 SimpleShowerModel=1 SimpleSpaceShower=1 SimpleTimeShower=1\n";
 std::cout<<"PROOF Merging:useShowerPlugin="<<pythia.settings.flag("Merging:useShowerPlugin")<<" (does not change runtime shower types)\n";
 double Q=200.,q=20.,m=500.,p=0.,minimum=0.;int id=21,type=1,seed=1027102;
 pythia_get_no_emission_prob_(p,Q,q,m,id,type,seed,minimum);
 std::cout<<std::setprecision(16)<<"PROOF II gluon="<<p<<"\n";
 assert(p==0.374558);
 id=6;type=4;seed=1027108;
 pythia_get_no_emission_prob_(p,Q,q,m,id,type,seed,minimum);
 std::cout<<"PROOF FI top dummy="<<p<<"\n";
 assert(p==0.6533);
}
