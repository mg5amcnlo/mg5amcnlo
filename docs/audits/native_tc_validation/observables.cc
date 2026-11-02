// Shared first/full-shower observables for the native-t_c validation.
// Inputs are ALL stable particles, p(4,n)=(px,py,pz,E). No event selection
// occurs here. Only Wj's separate fiducial fields impose pT>40, |y|<4.5.
#include "fjcore.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr int nobs=19;
struct Four {
  double x=0.,y=0.,z=0.,e=0.;
  Four()=default;
  Four(double a,double b,double c,double d):x(a),y(b),z(c),e(d) {}
  double pt() const {return std::hypot(x,y);}
  double rapidity() const {
    return e>std::abs(z) ? .5*std::log((e+z)/(e-z))
                        : std::numeric_limits<double>::quiet_NaN();
  }
};
Four operator+(const Four&a,const Four&b){return {a.x+b.x,a.y+b.y,a.z+b.z,a.e+b.e};}
Four four(const fjcore::PseudoJet& p){return {p.px(),p.py(),p.pz(),p.E()};}
double mass(const Four&p){
  const double q2=p.e*p.e-p.x*p.x-p.y*p.y-p.z*p.z;
  if(q2 < -1.e-10*std::max(1.,p.e*p.e))throw std::runtime_error("Spacelike system");
  return std::sqrt(std::max(0.,q2));
}
double deltaPhi(const Four&a,const Four&b){
  const double x=std::atan2(a.y,a.x)-std::atan2(b.y,b.x);
  return std::abs(std::atan2(std::sin(x),std::cos(x)));
}
bool parton(int id){return id==21 || (std::abs(id)>=1 && std::abs(id)<=5);}
std::vector<fjcore::PseudoJet> jets(const std::vector<fjcore::PseudoJet>& p){
  if(p.empty())return {};
  fjcore::ClusterSequence cs(p,fjcore::JetDefinition(fjcore::antikt_algorithm,.5));
  return fjcore::sorted_by_pt(cs.inclusive_jets(0.));
}
void pairValues(double*v,const Four&a,const Four&b){
  const Four p=a+b;
  v[14]=mass(p);v[15]=p.pt();v[16]=deltaPhi(a,b);
  v[17]=a.rapidity()-b.rapidity();
}
double cParameter(const std::vector<fjcore::PseudoJet>& p){
  double total=0.,numerator=0.;
  for(unsigned i=0;i<p.size();++i){
    const double a=std::sqrt(p[i].modp2());total+=a;
    if(a==0.)continue;
    for(unsigned j=0;j<i;++j){
      const double b=std::sqrt(p[j].modp2());if(b==0.)continue;
      const double dot=p[i].px()*p[j].px()+p[i].py()*p[j].py()+p[i].pz()*p[j].pz();
      const double cosine=std::max(-1.,std::min(1.,dot/(a*b)));
      numerator+=a*b*(1.-cosine*cosine);
    }
  }
  return total>0. ? std::max(0.,std::min(1.,3.*numerator/(total*total))) : 0.;
}
}

// process: DY1,H2,ttbar3,Wbb4,Wj5,EE2j6,EE3j7,EE4j8.
// values: primary,system_mass,system_y,object1_pt,object2_pt,jet1_pt,
// jet2_pt,fiducial_primary,fiducial_pass,y23,y34,y45,nDurhamJets,
// Cparameter,pair_mass,pair_pt,pair_delta_phi,pair_delta_y,radiation_jet1_pt.
// Nonapplicable values are zero. Undefined rapidities remain NaN coordinates
// with status0; callers may omit those coordinates without vetoing an event.
extern "C" void native_tc_observables_(const int*n,const int*ids,const double*p,
                                       const int*process,double*v,int*status){
  std::fill(v,v+nobs,0.);*status=1;
  try{
    if(*n<1 || *n>20000 || *process<1 || *process>8)return;
    fjcore::ClusterSequence::set_fastjet_banner_stream(nullptr);
    std::vector<fjcore::PseudoJet> all,radiation;
    Four w,b,bbar,top,antitop,lepton,antilepton,higgs;
    int nw=0,nb=0,nbbar=0,nt=0,ntbar=0,nl=0,nal=0,nh=0;
    for(int i=0;i<*n;++i){
      const double*a=p+4*i;
      for(int k=0;k<4;++k)if(!std::isfinite(a[k]))return;
      if(!(a[3]>0.))return;
      const Four q{a[0],a[1],a[2],a[3]};
      if(ids[i]==24){w=q;++nw;}
      if(ids[i]==5){b=q;++nb;}
      if(ids[i]==-5){bbar=q;++nbbar;}
      if(ids[i]==6){top=q;++nt;}
      if(ids[i]==-6){antitop=q;++ntbar;}
      if(ids[i]==11){lepton=q;++nl;}
      if(ids[i]==-11){antilepton=q;++nal;}
      if(ids[i]==25){higgs=q;++nh;}
      if(parton(ids[i])){
        all.emplace_back(a[0],a[1],a[2],a[3]);
        if(*process!=4 || std::abs(ids[i])!=5)
          radiation.emplace_back(a[0],a[1],a[2],a[3]);
      }
    }
    Four system;
    if(*process>=6){
      if(all.size()<2){*status=2;return;}
      for(const auto&q:all)system=system+four(q);
      const double Q2=mass(system)*mass(system);
      if(!(Q2>0.))return;
      fjcore::ClusterSequence cs(all,fjcore::JetDefinition(fjcore::ee_kt_algorithm));
      for(int target=2;target<=4;++target)
        v[9+target-2]=all.size()>unsigned(target) ? std::max(0.,cs.exclusive_dmerge(target))/Q2 : 0.;
      const int bornJets=*process-4;
      v[0]=std::sqrt(std::max(0.,v[9+bornJets-2]*Q2));
      const auto resolved=fjcore::sorted_by_pt(cs.exclusive_jets_ycut(.01));
      v[12]=resolved.size();v[13]=cParameter(all);
      if(!resolved.empty())v[5]=resolved[0].perp();
      if(resolved.size()>1)v[6]=resolved[1].perp();
    }else{
      const auto clustered=jets(all),rad=jets(radiation);
      if(!clustered.empty())v[5]=clustered[0].perp();
      if(clustered.size()>1)v[6]=clustered[1].perp();
      if(!rad.empty())v[18]=rad[0].perp();
      switch(*process){
      case 1:
        if(nl!=1 || nal!=1){*status=2;return;}
        system=lepton+antilepton;v[3]=lepton.pt();v[4]=antilepton.pt();
        pairValues(v,lepton,antilepton);break;
      case 2:
        if(nh!=1){*status=2;return;}
        system=higgs;v[3]=higgs.pt();break;
      case 3:
        if(nt!=1 || ntbar!=1){*status=2;return;}
        system=top+antitop;v[3]=top.pt();v[4]=antitop.pt();
        pairValues(v,top,antitop);break;
      case 4:
        if(nw!=1 || nb!=1 || nbbar!=1){*status=2;return;}
        system=w+b+bbar;v[3]=b.pt();v[4]=bbar.pt();pairValues(v,b,bbar);break;
      case 5:
        if(nw!=1){*status=2;return;}
        if(!clustered.empty()){
          system=w+four(clustered[0]);v[4]=clustered[0].perp();
          pairValues(v,w,four(clustered[0]));
        }else system=w;
        v[3]=w.pt();v[18]=v[6];v[8]=0.;
        for(const auto&j:clustered){
          if(j.perp()>40. && std::isfinite(j.rap()) && std::abs(j.rap())<4.5){
            v[7]=(w+four(j)).pt();v[8]=1.;break;
          }
        }
        break;
      }
      v[0]=system.pt();
    }
    v[1]=mass(system);v[2]=system.rapidity();
    if(*process!=5){v[7]=v[0];v[8]=1.;}
    for(int k=0;k<nobs;++k){
      if((k==2 || k==17) && std::isnan(v[k]))continue;
      if(!std::isfinite(v[k]))return;
    }
    *status=0;
  }catch(...){*status=3;}
}
