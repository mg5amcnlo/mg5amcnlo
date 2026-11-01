#include "Pythia8/Pythia.h"
#include <iomanip>
#include <iostream>
#include <string>
int main(int argc,char**argv){
 if(argc!=4)return 1;
 Pythia8::Pythia p;
 if(!p.readFile(argv[1]))return 2;
 p.readString(std::string("Beams:LHEF = ")+argv[2]);
 if(p.flag("Main:spareFlag1"))p.settings.forceParm("SpaceShower:pT0Ref",0.);
 if(!p.init())return 3;
 p.settings.writeFile(argv[3],true);
 std::cout<<"AUDITJSON "<<std::setprecision(17)<<"{";
 bool first=true;
 for(const std::string k:{"SpaceShower:nQuarkIn","TimeShower:nGluonToQuark","StandardModel:alphaSnfmax","SpaceShower:alphaSorder","TimeShower:alphaSorder"}){if(!first)std::cout<<",";first=false;std::cout<<"\""<<k<<"\":"<<p.mode(k);}
 for(const std::string k:{"SpaceShower:alphaSvalue","TimeShower:alphaSvalue","SpaceShower:pTmin","TimeShower:pTmin","SpaceShower:pT0Ref"})std::cout<<",\""<<k<<"\":"<<p.parm(k);
 for(const std::string k:{"SpaceShower:alphaSuseCMW","TimeShower:alphaSuseCMW","PartonLevel:MPI","HadronLevel:Hadronize","TimeShower:globalRecoil","Beams:setProductionScalesFromLHEF","Beams:setDipoleShowerStartingScalesFromLHEF","24:mayDecay"})std::cout<<",\""<<k<<"\":"<<(k=="24:mayDecay"?p.particleData.mayDecay(24):p.flag(k));
 for(int id:{1,2,3,4,5,6,21,23,24})std::cout<<",\""<<id<<":m0\":"<<p.particleData.m0(id);
 std::cout<<"}\n";
 return 0;
}
