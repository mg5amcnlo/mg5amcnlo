// Fresh point audit through the unmodified production SudGen interface.
// Link this driver with vendor/SudGen/pythia_interface_for_sudgen.cc.
#include "Pythia8/Pythia.h"
#include <iostream>
#include <iomanip>
#include <string>
extern "C" {
extern Pythia8::Pythia pythia;
void pythia_init_(double masses[21]);
void pythia_get_no_emission_prob_(double&, double&, double&, double&,
                                int&,int&,int&,double&);
}
int main(int argc,char**argv) {
 double mass[21]={.33,.33,.5,1.5,4.8,0,0,0,0,0,.000510998928,0,.1056583715,0,1.77682,0,0,0,0,0,0};
 pythia_init_(mass);
 std::string mode=argc>1?argv[1]:"vendor";
 if(mode!="vendor") {
   for(auto sector:{"SpaceShower:","TimeShower:"}) {
     pythia.readString(std::string(sector)+"alphaSvalue = 0.118");
     pythia.readString(std::string(sector)+"alphaSorder = 1");
     pythia.readString(std::string(sector)+"alphaSuseCMW = off");
   }
   pythia.readString("SpaceShower:pdfMode = "+mode);
   if(!pythia.init() || !pythia.next()) return 2;
 }
 std::cout<<std::setprecision(16);
 std::cout<<"SETTINGS "<<mode<<" "<<pythia.settings.parm("SpaceShower:alphaSvalue")<<" "<<pythia.settings.parm("TimeShower:alphaSvalue")<<" "<<pythia.settings.mode("SpaceShower:pdfMode")<<std::endl;
 double q,Q,m;
 int id,type,seed;
 while(std::cin>>q>>Q>>m>>id>>type>>seed) {
   double prob=0.,min=0.;
   pythia_get_no_emission_prob_(prob,Q,q,m,id,type,seed,min);
   int n=(type==1||type==3)?500000:10000;
   std::cout<<"RESULT "<<q<<" "<<Q<<" "<<m<<" "<<id<<" "<<type<<" "<<seed<<" "<<prob<<" "<<n<<std::endl;
 }
}
