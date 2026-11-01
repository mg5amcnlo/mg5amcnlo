// Campaign-only no-cut W+ b bbar analysis. Shared by LHE and full shower.
#include "fjcore.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr int nobs = 19;
struct Four {
  double x,y,z,e;
  Four(double a=0.,double c=0.,double d=0.,double f=0.):x(a),y(c),z(d),e(f) {}
  double pt() const { return std::hypot(x,y); }
  double rap() const {
    if (e<=std::abs(z)) return std::numeric_limits<double>::quiet_NaN();
    return .5*std::log((e+z)/(e-z));
  }
};
Four add(const Four& a,const Four& b) {
  return {a.x+b.x,a.y+b.y,a.z+b.z,a.e+b.e};
}
double mass(const Four& a) {
  const double q2=a.e*a.e-a.x*a.x-a.y*a.y-a.z*a.z;
  if (q2 < -1.e-10*std::max(1.,a.e*a.e))
    throw std::runtime_error("Spacelike hard system");
  return std::sqrt(std::max(0.,q2));
}
std::array<double,2> leading(const std::vector<fjcore::PseudoJet>& p) {
  std::array<double,2> answer{};
  if (p.empty()) return answer;
  fjcore::ClusterSequence cs(p,fjcore::JetDefinition(fjcore::antikt_algorithm,.5));
  const auto jets=fjcore::sorted_by_pt(cs.inclusive_jets(0.));
  for (unsigned i=0;i<std::min(jets.size(),answer.size());++i) answer[i]=jets[i].perp();
  return answer;
}
}

// ids and p contain ALL stable particles; p is Fortran p(4,n)=(px,py,pz,E).
// No generation/analysis acceptance cuts, and no jet pT or rapidity thresholds.
extern "C" void wbb_observables_(const int* n,const int* ids,const double* p,
                                 double* values,int* status) {
  std::fill(values,values+nobs,std::numeric_limits<double>::quiet_NaN());
  *status=1;
  try {
    if (*n<3 || *n>20000) return;
    fjcore::ClusterSequence::set_fastjet_banner_stream(nullptr);
    Four w,b,bb;
    int nw=0,nb=0,nbb=0;
    std::vector<fjcore::PseudoJet> radiation,all;
    for (int i=0;i<*n;++i) {
      const double* a=p+4*i;
      for(int j=0;j<4;++j) if (!std::isfinite(a[j])) return;
      if (!(a[3]>0.)) return;
      const Four q{a[0],a[1],a[2],a[3]};
      if (ids[i]==24) {w=q;++nw;}
      if (ids[i]==5) {b=q;++nb;}
      if (ids[i]==-5) {bb=q;++nbb;}
      if ((std::abs(ids[i])>=1 && std::abs(ids[i])<=5) || ids[i]==21) {
        all.emplace_back(a[0],a[1],a[2],a[3]);
        if (std::abs(ids[i])!=5) radiation.emplace_back(a[0],a[1],a[2],a[3]);
      }
    }
    if (nw!=1 || nb!=1 || nbb!=1) {*status=2;return;}
    const Four pair=add(b,bb),system=add(w,pair);
    const auto rad=leading(radiation),inc=leading(all);
    const double phi0=std::atan2(b.y,b.x)-std::atan2(bb.y,bb.x);
    const double phi=std::abs(std::atan2(std::sin(phi0),std::cos(phi0)));
    const double dy=b.rap()-bb.rap();
    const std::array<double,nobs> result{{
      system.pt(),mass(system),system.rap(),pair.pt(),mass(pair),pair.rap(),
      w.pt(),w.rap(),b.pt(),b.rap(),bb.pt(),bb.rap(),phi,dy,
      std::hypot(dy,phi),rad[0],rad[1],inc[0],inc[1]}};
    std::copy(result.begin(),result.end(),values);
    for(int i=0;i<nobs;++i) {
      // Rapidity may be undefined only on a longitudinal endpoint. Preserve
      // that coordinate as missing without rejecting the event or its weight.
      if ((i==2 || i==5 || i==7 || i==9 || i==11 || i==13 || i==14)
          && std::isnan(values[i])) continue;
      if (!std::isfinite(values[i])) return;
    }
    *status=0;
  } catch (...) { *status=3; }
}
