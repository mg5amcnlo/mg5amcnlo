// On-the-fly parton-level histogram consumer. No event-record output.
#include "Pythia8/Pythia.h"
#include "fjcore.hh"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <openssl/sha.h>
#include <zlib.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using boost::property_tree::ptree;
using Pythia8::Pythia;
constexpr int NS=11;
constexpr int NO=21;
const std::array<std::string,NS> stages{{"reference","input","h_only","s_only","macnlops","H_positive","H_negative","S_positive","S_negative","H_total","H_negative_retained"}};
const double nanValue=std::numeric_limits<double>::quiet_NaN();
void require(bool ok,const std::string& what){if(!ok)throw std::runtime_error(what);}
std::string trimmed(std::string s){auto a=s.find_first_not_of(" \r\n\t");if(a==std::string::npos)return "";return s.substr(a,s.find_last_not_of(" \r\n\t")-a+1);}
double number(std::string s){std::replace(s.begin(),s.end(),'D','e');std::replace(s.begin(),s.end(),'d','e');return std::stod(s);}
class GzBuffer:public std::streambuf{
 gzFile file;char buffer[65536];
 int_type underflow()override{int n=gzread(file,buffer,sizeof(buffer));if(n<0)throw std::runtime_error("gzip read failed");if(!n)return traits_type::eof();setg(buffer,buffer,buffer+n);return traits_type::to_int_type(*gptr());}
 public:explicit GzBuffer(const std::string& path):file(gzopen(path.c_str(),"rb")){require(file,"Cannot open "+path);setg(buffer,buffer,buffer);}~GzBuffer(){gzclose(file);}
};
class Input:public std::istream{GzBuffer buffer;public:explicit Input(const std::string& p):std::istream(nullptr),buffer(p){rdbuf(&buffer);}};
struct Key{uint64_t a,b;bool operator==(const Key& o)const{return a==o.a&&b==o.b;}};
struct KeyHash{size_t operator()(const Key& k)const{return k.a^(k.b+0x9e3779b97f4a7c15ULL+(k.a<<6)+(k.a>>2));}};
struct Particle{int id,status;double px,py,pz,e,m;};
struct RawEvent{
 double weight=0,scale=0;int kind=0;std::vector<Particle> particles;std::string extra;
 Key identity()const{
  auto a=extra.find("<macnlops_hard_event"),b=extra.find("</macnlops_hard_event>",a);
  require(a!=std::string::npos&&b!=std::string::npos,"Missing original hard-event identity");
  const std::string endTag="</macnlops_hard_event>";std::string value;value.reserve(b-a+30);for(size_t i=a;i<b+endTag.size()&&i<extra.size();++i)if(!std::isspace(static_cast<unsigned char>(extra[i])))value.push_back(extra[i]);
  // A sampled source also fixes its diagram/kl/ij component. Include it in
  // the pairing key when comparing uninterrupted and staged showers.
  auto c=extra.find("<macnlops_component"),d=extra.find("</macnlops_component>",c);
  if(c!=std::string::npos){require(d!=std::string::npos,"Truncated component identity");const std::string end="</macnlops_component>";for(size_t i=c;i<d+end.size();++i)if(!std::isspace(static_cast<unsigned char>(extra[i])))value.push_back(extra[i]);}
  unsigned char hash[SHA256_DIGEST_LENGTH];SHA256(reinterpret_cast<const unsigned char*>(value.data()),value.size(),hash);
  Key key;std::memcpy(&key.a,hash,8);std::memcpy(&key.b,hash+8,8);return key;
 }
};
class RawReader{
 Input input;
 public:explicit RawReader(const std::string& p):input(p){}
 bool next(RawEvent& event){
  std::string line;bool found=false;while(std::getline(input,line))if(trimmed(line)=="<event>"){found=true;break;}if(!found)return false;
  require(bool(std::getline(input,line)),"Truncated event header");std::istringstream head(line);int n,proc;std::string w,s,ae,as;
  require(bool(head>>n>>proc>>w>>s>>ae>>as),"Malformed event header");event.weight=number(w);event.scale=number(s);event.particles.clear();event.extra.clear();event.kind=0;
  require(std::isfinite(event.weight),"Nonfinite event weight");
  for(int i=0;i<n;++i){require(bool(std::getline(input,line)),"Truncated particle");std::istringstream row(line);Particle p;int m1,m2,c1,c2;std::string px,py,pz,e,m,l,spin;
   require(bool(row>>p.id>>p.status>>m1>>m2>>c1>>c2>>px>>py>>pz>>e>>m>>l>>spin),"Malformed particle");p.px=number(px);p.py=number(py);p.pz=number(pz);p.e=number(e);p.m=number(m);event.particles.push_back(p);
  }
  bool closed=false;while(std::getline(input,line)){if(trimmed(line)=="</event>"){closed=true;break;}event.extra+=line+'\n';if(trimmed(line).find("#aMCatNLO")==0){std::istringstream tag(line);std::string name;tag>>name>>event.kind;}}
  require(closed&&(event.kind==1||event.kind==2),"Incomplete or unlabelled event");return true;
 }
};
struct Decision{long sequence;bool accepted;char kind;};
std::string field(const std::string& row,const std::string& key){auto p=row.find('"'+key+'"');require(p!=std::string::npos,"Missing decision "+key);p=row.find(':',p)+1;auto q=row.find_first_of(",}",p);return trimmed(row.substr(p,q-p));}
Decision decision(const std::string& row){Decision d;d.sequence=std::stol(field(row,"sequence"));auto a=field(row,"accepted");require(a=="true"||a=="false","Invalid decision");d.accepted=a=="true";auto k=field(row,"kind");require(k=="\"S\""||k=="\"H\"","Invalid decision kind");d.kind=k[1];return d;}
using Mask=std::array<bool,NS>;
Mask masks(const RawEvent& e,bool keep,bool reference){bool h=e.kind==2,n=e.weight<0;return Mask{{reference,true,!h||keep,h||keep,keep,h&&!n,h&&n,!h&&!n,!h&&n,h,h&&n&&keep}};}
struct Sum{long count=0,negative=0;double w=0,w2=0,abs=0;void fill(double v){++count;negative+=v<0;w+=v;w2+=v*v;abs+=std::abs(v);}};
struct Hist{
 std::string name,label,unit;std::vector<double> edges;bool log;
 std::vector<double> w,w2,abs,cov;std::vector<long> count;std::array<std::array<Sum,NS>,4> flow;
 explicit Hist(const ptree& c):name(c.get<std::string>("name")),label(c.get<std::string>("label")),unit(c.get<std::string>("unit")),log(c.get<bool>("log")){
  for(auto& x:c.get_child("edges"))edges.push_back(x.second.get_value<double>());
  require(edges.size()>1&&std::is_sorted(edges.begin(),edges.end()),"Invalid bins");int n=edges.size()-1;w.resize(NS*n);w2.resize(NS*n);abs.resize(NS*n);count.resize(NS*n);cov.resize(NS*NS*n);
 }
 int bins()const{return edges.size()-1;}
 int bin(double x)const{if(std::isnan(x))return -3;if(x<edges.front())return -1;if(x>edges.back())return -2;if(x==edges.back())return bins()-1;return std::upper_bound(edges.begin(),edges.end(),x)-edges.begin()-1;}
 void fill(double x,double weight,const Mask& mask,int referenceBin=-4,double referenceWeight=0){
  const int b=bin(x),n=bins();for(int i=0;i<NS;++i)if(mask[i]){
   if(b>=0){int k=i*n+b;w[k]+=weight;w2[k]+=weight*weight;abs[k]+=std::abs(weight);++count[k];
    for(int j=0;j<NS;++j)if(mask[j])cov[(i*NS+j)*n+b]+=weight*weight;
    if(i&&b==referenceBin){cov[i*NS*n+b]+=weight*referenceWeight;cov[i*n+b]+=weight*referenceWeight;}
   }
   if(b<0)flow[-b-1][i].fill(weight);if(x==0.)flow[3][i].fill(weight);
  }
 }
};
struct Four{double x=0,y=0,z=0,e=0;double pt()const{return std::hypot(x,y);}double rapidity()const{if(e<=std::abs(z))return z?std::copysign(std::numeric_limits<double>::infinity(),z):0.;return .5*std::log((e+z)/(e-z));}};
std::map<std::string,double> wjetObservables(const std::vector<Particle>& particles){
 std::vector<Four> w;std::vector<fjcore::PseudoJet> partons;
 for(const auto& p:particles)if(p.status==1){
  if(p.id==24)w.push_back(Four{p.px,p.py,p.pz,p.e});
  if((std::abs(p.id)>=1&&std::abs(p.id)<=5)||p.id==21)partons.emplace_back(p.px,p.py,p.pz,p.e);
 }
 require(w.size()==1,"Expected one stable W+ in W+jet sample");
 if(partons.empty())return {};
 fjcore::ClusterSequence cs(partons,fjcore::JetDefinition(fjcore::antikt_algorithm,.5));
 auto jets=fjcore::sorted_by_pt(cs.inclusive_jets());
 std::vector<fjcore::PseudoJet> central;
 for(const auto& jet:jets)if(std::abs(jet.rap())<4.5)central.push_back(jet);
 if(central.empty()||central[0].pt()<=25.)return {};
 Four j{central[0].px(),central[0].py(),central[0].pz(),central[0].e()};
 Four system{w[0].x+j.x,w[0].y+j.y,w[0].z+j.z,w[0].e+j.e};
 double m2=system.e*system.e-system.x*system.x-system.y*system.y-system.z*system.z;
 require(m2>=-1.e-10*system.e*system.e,"Spacelike W+jet system");
 double phi=std::atan2(w[0].y,w[0].x)-std::atan2(j.y,j.x);
 phi=std::abs(std::atan2(std::sin(phi),std::cos(phi)));
 // Additional-jet pT is inclusive down to zero to resolve the soft region.
 // The event selection itself always requires a jet above 25 GeV.
 double radiation=central.size()>1?central[1].pt():0.;
 double radiationY=central.size()>1&&radiation>25.?central[1].rap():nanValue;
 return {{"system_pt",system.pt()},{"system_pt_soft",system.pt()},
 {"system_mass",std::sqrt(std::max(0.,m2))},{"system_y",system.rapidity()},
 {"delta_phi",phi},{"delta_phi_deficit",std::max(0.,std::acos(-1.)-phi)},
 {"delta_y",w[0].rapidity()-j.rapidity()},
 {"object_1_pt",w[0].pt()},{"object_1_y",w[0].rapidity()},
 {"object_2_pt",j.pt()},{"object_2_y",j.rapidity()},
 {"jet_pt",radiation},{"jet_pt_soft",radiation},{"jet_y",radiationY}};
}
std::map<std::string,double> wbbObservables(const std::vector<Particle>& particles){
 std::vector<Four> w,b,bb;std::vector<fjcore::PseudoJet> radiation,all;
 for(const auto& p:particles)if(p.status==1){
  Four q{p.px,p.py,p.pz,p.e};
  if(p.id==24)w.push_back(q);if(p.id==5)b.push_back(q);if(p.id==-5)bb.push_back(q);
  if((std::abs(p.id)>=1&&std::abs(p.id)<=5)||p.id==21){
   all.emplace_back(p.px,p.py,p.pz,p.e);
   if(std::abs(p.id)!=5)radiation.emplace_back(p.px,p.py,p.pz,p.e);
  }
 }
 require(w.size()==1&&b.size()==1&&bb.size()==1,"Expected one stable W+, b and bbar in the four-flavour sample");
 auto add=[](const Four& x,const Four& y){return Four{x.x+y.x,x.y+y.y,x.z+y.z,x.e+y.e};};
 auto mass=[](const Four& x){double m2=x.e*x.e-x.x*x.x-x.y*x.y-x.z*x.z;require(m2>=-1.e-10*x.e*x.e,"Spacelike Wbb system");return std::sqrt(std::max(0.,m2));};
 auto leading=[](const std::vector<fjcore::PseudoJet>& partons){
  std::pair<double,double> result{0.,nanValue};
  if(!partons.empty()){
   fjcore::ClusterSequence cs(partons,fjcore::JetDefinition(fjcore::antikt_algorithm,.4));auto jets=fjcore::sorted_by_pt(cs.inclusive_jets());
   if(!jets.empty()){result.first=jets[0].pt();if(result.first>20.)result.second=jets[0].rap();}
  }
  return result;
 };
 Four pair=add(b[0],bb[0]),system=add(add(w[0],b[0]),bb[0]);auto jet=leading(radiation),inclusive=leading(all);
 double phi=std::atan2(b[0].y,b[0].x)-std::atan2(bb[0].y,bb[0].x);phi=std::abs(std::atan2(std::sin(phi),std::cos(phi)));
 return {{"system_pt",system.pt()},{"system_pt_soft",system.pt()},{"system_mass",mass(system)},{"system_y",system.rapidity()},
 {"delta_phi",phi},{"delta_phi_deficit",std::max(0.,std::acos(-1.)-phi)},{"delta_y",b[0].rapidity()-bb[0].rapidity()},
 {"jet_pt",jet.first},{"jet_pt_soft",jet.first},{"jet_y",jet.second},{"all_jet_pt",inclusive.first},{"all_jet_y",inclusive.second},
 {"object_1_pt",w[0].pt()},{"object_1_y",w[0].rapidity()},
 {"object_2_pt",b[0].pt()},{"object_2_y",b[0].rapidity()},{"object_3_pt",bb[0].pt()},{"object_3_y",bb[0].rapidity()},
 {"pair_pt",pair.pt()},{"pair_mass",mass(pair)},{"pair_y",pair.rapidity()}};
}
std::map<std::string,double> observables(const std::vector<Particle>& particles,const std::string& process){
 if(process=="wjet")return wjetObservables(particles);
 if(process=="wbb")return wbbObservables(particles);
 std::vector<Four> a,b;std::vector<fjcore::PseudoJet> partons;for(const auto& p:particles)if(p.status==1){
  Four q{p.px,p.py,p.pz,p.e};if(process=="zz"){if(p.id==23)a.push_back(q);}else{int target=process=="dy"?11:6;if(p.id==target)a.push_back(q);if(p.id==-target)b.push_back(q);}
  if((std::abs(p.id)>=1&&std::abs(p.id)<=5)||p.id==21)partons.emplace_back(p.px,p.py,p.pz,p.e);
 }
 if(process=="zz"){require(a.size()==2,"Expected two stable Z objects");std::sort(a.begin(),a.end(),[](const Four& x,const Four& y){return x.pt()>y.pt();});b.push_back(a[1]);a.resize(1);}
 double jp=0,jy=nanValue;
 if(!partons.empty()){
  fjcore::ClusterSequence cs(partons,fjcore::JetDefinition(fjcore::antikt_algorithm,.4));auto jets=fjcore::sorted_by_pt(cs.inclusive_jets());
  if(process=="single_top"){
   require(a.size()==1&&b.empty(),"Expected exactly one stable top and no antitop");
   require(!jets.empty(),"Single-top event has no spectator-jet candidate");
   // The leading inclusive jet is an infrared-safe spectator proxy; no
   // shower ancestry or quark flavour tag is used to identify it.
   b.push_back(Four{jets[0].px(),jets[0].py(),jets[0].pz(),jets[0].e()});
  }
  const size_t rank=process=="single_top"?1:0;
  if(jets.size()>rank){jp=jets[rank].pt();if(jp>20.)jy=jets[rank].rap();}
 }
 require(a.size()==1&&b.size()==1,"Expected two stable hard objects");Four sum{a[0].x+b[0].x,a[0].y+b[0].y,a[0].z+b[0].z,a[0].e+b[0].e};
 double m2=sum.e*sum.e-sum.x*sum.x-sum.y*sum.y-sum.z*sum.z;require(m2>=-1.e-10*sum.e*sum.e,"Spacelike hard system");
 double phi=std::atan2(a[0].y,a[0].x)-std::atan2(b[0].y,b[0].x);phi=std::abs(std::atan2(std::sin(phi),std::cos(phi)));
 return {{"system_pt",sum.pt()},{"system_pt_soft",sum.pt()},{"system_mass",std::sqrt(std::max(0.,m2))},{"system_y",sum.rapidity()},
 {"delta_phi",phi},{"delta_phi_deficit",std::max(0.,std::acos(-1.)-phi)},{"delta_y",a[0].rapidity()-b[0].rapidity()},
 {"jet_pt",jp},{"jet_pt_soft",jp},{"jet_y",jy},{"object_1_pt",a[0].pt()},{"object_2_pt",b[0].pt()},{"object_1_y",a[0].rapidity()},{"object_2_y",b[0].rapidity()}};
}
std::vector<Particle> finalParticles(const Pythia8::Event& event){std::vector<Particle> result;for(int i=1;i<event.size();++i)if(event[i].isFinal()){const auto& p=event[i];result.push_back(Particle{p.id(),1,p.px(),p.py(),p.pz(),p.e(),p.m()});}return result;}
struct Node{double weight;std::array<short,NO> bins;int next=-1;};
class Pairs{
 std::vector<Node> nodes;std::unordered_map<Key,std::pair<int,int>,KeyHash> positions;
 public:long consumed=0;Pairs(long n){nodes.reserve(n);positions.reserve(n);}
 void add(Key key,double weight,const std::vector<Hist>& h,const std::map<std::string,double>& values){Node node;node.weight=weight;require(h.size()<=NO,"Too many observables");for(size_t i=0;i<h.size();++i)node.bins[i]=values.empty()?-3:h[i].bin(values.at(h[i].name));int n=nodes.size();nodes.push_back(node);auto it=positions.find(key);if(it==positions.end())positions.emplace(key,std::make_pair(n,n));else{nodes[it->second.second].next=n;it->second.second=n;}}
 Node take(Key key,double weight){auto it=positions.find(key);require(it!=positions.end()&&it->second.first>=0,"Source/endpoint hard identities differ");int index=it->second.first;Node node=nodes[index];require(std::abs(node.weight-weight)<2.e-12*std::max(1.,std::abs(weight)),"Paired source weight differs");it->second.first=node.next;++consumed;return node;}
};
std::map<std::string,double> fillSelected(const RawEvent& event,const std::vector<Particle>& particles,
 const std::string& process,const Mask& mask,const Node* paired,std::vector<Hist>& hist,
 std::array<Sum,NS>& totals,std::array<Sum,NS>& fiducial){
 auto values=observables(particles,process);
 for(int i=0;i<NS;++i)if(mask[i])totals[i].fill(event.weight);
 if(values.empty())return values;
 for(int i=0;i<NS;++i)if(mask[i])fiducial[i].fill(event.weight);
 for(size_t i=0;i<hist.size();++i)hist[i].fill(values.at(hist[i].name),event.weight,mask,
   paired?paired->bins[i]:-4,paired?paired->weight:0.);
 return values;
}
class CountReader:public Pythia8::LHAupLHEF{
 public:long count=0;CountReader(Pythia8::Info* info,std::istream* stream,std::istream* header,bool scales):LHAupLHEF(info,stream,header,true,scales){}
 bool setEvent(int id=0)override{bool result=LHAupLHEF::setEvent(id);if(result)++count;return result;}
};
struct Audit{long events=0,exhausted=0,isr=0,fsr=0;double maxRatio=0,maxMomentum=0,maxShell=0;};
class Caps:public Pythia8::UserHooks{
 public:bool endpoint;double cap=0,maxpt=0;bool exhausted=false;long nISR=0,nFSR=0;
 explicit Caps(bool ep):endpoint(ep){}
 bool canVetoStep()override{return true;}int numberVetoStep()override{return 10000000;}
 bool doVetoStep(int pos,int ni,int nf,const Pythia8::Event&)override{require(pos==2||pos==3,"Non-QCD evolution");nISR=ni;nFSR=nf;double pt=infoPtr->pTnow();maxpt=std::max(maxpt,pt);if(endpoint)require(pt<=cap*(1.+1.e-10)+1.e-10,"Continuation exceeded written scale");return false;}
};
class Shower{
 Input input,header;std::shared_ptr<CountReader> reader;std::shared_ptr<Caps> hook;std::unique_ptr<Pythia> p;
 public:Audit audit;
 Shower(const std::string& file,const std::string& settings,bool endpoint,bool delta,int seed):input(file),header(file),hook(std::make_shared<Caps>(endpoint)),p(new Pythia){
  require(p->readFile(settings),"Cannot read shower settings");
  auto setting=[&](const std::string& s){require(p->readString(s),"Unsupported setting "+s);};
  setting("Beams:frameType=5");setting("Random:setSeed=on");setting("Random:seed="+std::to_string(seed));
  p->settings.forceParm("SpaceShower:pT0Ref",0.);
  for(const char* s:{"HadronLevel:all=off","PartonLevel:MPI=off","BeamRemnants:primordialKT=off","ColourReconnection:reconnect=off","ProcessLevel:resonanceDecays=off","PartonLevel:FSRinResonances=off","TimeShower:QEDshowerByQ=off","TimeShower:QEDshowerByL=off","TimeShower:QEDshowerByOther=off","TimeShower:QEDshowerByGamma=off","SpaceShower:QEDshowerByQ=off","SpaceShower:QEDshowerByL=off","PartonLevel:Remnants=on","Check:abortIfVeto=on","Next:numberShowInfo=0","Next:numberShowProcess=0","Next:numberShowEvent=0","Next:numberCount=0","Init:showChangedSettings=off","Init:showChangedParticleData=off","LesHouches:setQuarkMass=0","LesHouches:setLeptonMass=0"})setting(s);
  setting(std::string("Beams:setProductionScalesFromLHEF=")+(endpoint||delta?"on":"off"));setting(std::string("Beams:setDipoleShowerStartingScalesFromLHEF=")+(endpoint||delta?"on":"off"));
  if(endpoint)setting("TimeShower:globalRecoil=off");
  reader=std::make_shared<CountReader>(const_cast<Pythia8::Info*>(&p->info),&input,&header,endpoint||delta);p->setLHAupPtr(reader);p->setUserHooksPtr(hook);require(p->init(),"Pythia initialization failed");
  require(!p->flag("HadronLevel:all")&&!p->flag("PartonLevel:MPI")&&!p->flag("ProcessLevel:resonanceDecays")&&p->parm("SpaceShower:pT0Ref")==0.,"Parton-only profile mismatch");
 }
 std::vector<Particle> next(const RawEvent& event){
  hook->cap=event.scale;hook->exhausted=event.extra.find("state=\"exhausted\"")!=std::string::npos;hook->maxpt=0;hook->nISR=hook->nFSR=0;
  long before=reader->count;require(p->next(),"Pythia failed on input "+std::to_string(audit.events+1));require(reader->count==before+1&&p->info.nTried()==audit.events+1&&p->info.nSelected()==audit.events+1,"Skipped or repeated input");require(reader->weight()==event.weight,"Imported nominal weight changed");
  if(hook->endpoint){require(std::abs(reader->scale()-event.scale)<2.e-12*std::max(1.,event.scale),"Imported continuation cap differs");if(hook->exhausted)require(!hook->nISR&&!hook->nFSR,"Exhausted endpoint reradiated");}
  ++audit.events;audit.exhausted+=hook->exhausted;audit.isr+=hook->nISR;audit.fsr+=hook->nFSR;if(hook->endpoint&&event.scale>0)audit.maxRatio=std::max(audit.maxRatio,hook->maxpt/event.scale);
  auto result=finalParticles(p->event);std::array<double,4> momentum{{0,0,0,0}};
  for(const auto& q:result){momentum[0]+=q.px;momentum[1]+=q.py;momentum[2]+=q.pz;momentum[3]+=q.e;double norm=std::max(1.,q.e*q.e);audit.maxShell=std::max(audit.maxShell,std::abs(q.e*q.e-q.px*q.px-q.py*q.py-q.pz*q.pz-q.m*q.m)/norm);require(q.id!=22,"Unexpected final photon");}
  for(int i=1;i<=2;++i){momentum[0]-=p->event[i].px();momentum[1]-=p->event[i].py();momentum[2]-=p->event[i].pz();momentum[3]-=p->event[i].e();}
  for(double value:momentum)audit.maxMomentum=std::max(audit.maxMomentum,std::abs(value));require(audit.maxMomentum<1.e-4&&audit.maxShell<1.e-8,"Final conservation failure");return result;
 }
 void save(const std::string& prefix){p->settings.writeFile(prefix+".settings.cmnd",true);p->stat();}
};
void sumJSON(std::ostream& out,const Sum& s){out<<"{\"events\":"<<s.count<<",\"sumw\":"<<s.w<<",\"sumw2\":"<<s.w2<<",\"sumabsw\":"<<s.abs<<",\"negative_events\":"<<s.negative<<'}';}
template<typename T>void arrayJSON(std::ostream& out,const std::vector<T>& v){out<<'[';for(size_t i=0;i<v.size();++i){if(i)out<<',';out<<v[i];}out<<']';}
void auditJSON(std::ostream& out,const Audit& a){out<<"{\"events\":"<<a.events<<",\"exhausted\":"<<a.exhausted<<",\"isr\":"<<a.isr<<",\"fsr\":"<<a.fsr<<",\"max_pt_over_cap\":"<<a.maxRatio<<",\"max_momentum_GeV\":"<<a.maxMomentum<<",\"max_shell\":"<<a.maxShell<<'}';}
void addAudit(Audit& a,const Audit& b){a.events+=b.events;a.exhausted+=b.exhausted;a.isr+=b.isr;a.fsr+=b.fsr;a.maxRatio=std::max(a.maxRatio,b.maxRatio);a.maxMomentum=std::max(a.maxMomentum,b.maxMomentum);a.maxShell=std::max(a.maxShell,b.maxShell);}
void progress(const std::string& path,const std::string& phase,long n,long total){std::ofstream out(path+".tmp");out<<"{\"phase\":\""<<phase<<"\",\"events\":"<<n<<",\"target\":"<<total<<"}\n";out.close();require(std::rename((path+".tmp").c_str(),path.c_str())==0,"Cannot publish progress");}
int main(int argc,char** argv){try{
 require(argc==2,"Usage: StreamHist config.json");ptree config;boost::property_tree::read_json(argv[1],config);
 const std::string process=config.get<std::string>("process"),level=config.get<std::string>("level"),output=config.get<std::string>("output");const bool full=level=="full",delta=config.get<bool>("delta");long expected=config.get<long>("events"),acceptedExpected=config.get<long>("accepted_events");
 if(process=="wjet")require(config.get<std::string>("analysis_jets.algorithm")=="antikt"&&config.get<double>("analysis_jets.radius")==.5&&config.get<double>("analysis_jets.pt_min_GeV")==25.&&config.get<double>("analysis_jets.abs_rapidity_max")==4.5&&config.get<int>("analysis_jets.minimum_jets")==1,"W+jet analysis configuration differs from the implemented fiducial selection");
 std::vector<Hist> hist;for(auto& row:config.get_child("observables"))hist.emplace_back(row.second);require(hist.size()<=NO,"Observable cache size");std::array<Sum,NS> totals,fiducial;Pairs pairs(full?expected:0);Audit referenceAudit,endpointAudit;
 auto fill=[&](const RawEvent& e,const std::vector<Particle>& particles,const Mask& mask,const Node* paired){return fillSelected(e,particles,process,mask,paired,hist,totals,fiducial);};
 fjcore::ClusterSequence::set_fastjet_banner_stream(nullptr);
 if(full){const auto file=config.get<std::string>("source");RawReader raw(file);RawEvent event;Shower shower(file,config.get<std::string>("source_settings"),false,delta,config.get<int>("seed"));long n=0;
  while(raw.next(event)){auto particles=shower.next(event);Mask mask{};mask[0]=true;auto values=fill(event,particles,mask,nullptr);pairs.add(event.identity(),event.weight,hist,values);++n;if(n%5000==0)progress(output+".progress.json","direct_mc_shower",n,expected);}
  require(n==expected,"Direct MC source count differs");referenceAudit=shower.audit;shower.save(output+".source");
 }
 long n=0,accepted=0;for(auto& row:config.get_child("channels")){const auto& channel=row.second;std::string file=channel.get<std::string>("input");RawReader raw(file);std::ifstream decisions(channel.get<std::string>("decisions"));require(decisions.good(),"Missing decisions");std::unique_ptr<Shower> shower;if(full)shower.reset(new Shower(file,config.get<std::string>("endpoint_settings"),true,delta,channel.get<int>("seed")));RawEvent event;long local=0;std::string line;
  while(raw.next(event)){require(bool(std::getline(decisions,line)),"Missing matching decision");Decision d=decision(line);++local;require(d.sequence==local&&d.kind==(event.kind==1?'S':'H'),"Decision/source order differs");Mask mask=masks(event,d.accepted,!full);Node paired;std::vector<Particle> particles;if(full){paired=pairs.take(event.identity(),event.weight);particles=shower->next(event);}else particles=event.particles;fill(event,particles,mask,full?&paired:nullptr);++n;accepted+=d.accepted;if(n%5000==0)progress(output+".progress.json",full?"endpoint_shower":"first_emission_histograms",n,expected);}
  require(!std::getline(decisions,line),"Extra matching decisions");require(local==channel.get<long>("events"),"Channel count differs");if(full){addAudit(endpointAudit,shower->audit);if(endpointAudit.events==local)shower->save(output+".endpoint");}
 }
 require(n==expected&&accepted==acceptedExpected,"Source or accepted total differs");if(full)require(pairs.consumed==expected,"Incomplete paired covariance");
 std::ofstream out(output+".tmp");out<<std::setprecision(17)<<"{\"schema\":\"macnlops-stream-histograms-v1\",\"complete\":true,\"events\":"<<n<<",\"accepted\":"<<accepted<<",\"level\":\""<<level<<"\",\"totals\":{";
 for(int i=0;i<NS;++i){if(i)out<<',';out<<'"'<<stages[i]<<"\":";sumJSON(out,totals[i]);}out<<"},\"fiducial_totals\":{";
 for(int i=0;i<NS;++i){if(i)out<<',';out<<'"'<<stages[i]<<"\":";sumJSON(out,fiducial[i]);}out<<"},\"reference_audit\":";auditJSON(out,referenceAudit);out<<",\"endpoint_audit\":";auditJSON(out,endpointAudit);out<<",\"histograms\":{";
 for(size_t i=0;i<hist.size();++i){if(i)out<<',';auto& h=hist[i];out<<'"'<<h.name<<"\":{\"edges\":";arrayJSON(out,h.edges);out<<",\"sumw\":";arrayJSON(out,h.w);out<<",\"sumw2\":";arrayJSON(out,h.w2);out<<",\"sumabsw\":";arrayJSON(out,h.abs);out<<",\"count\":";arrayJSON(out,h.count);out<<",\"covariance\":";arrayJSON(out,h.cov);out<<",\"flows\":{";for(int f=0;f<4;++f){if(f)out<<',';const char* names[]={"underflow","overflow","missing","zero"};out<<'"'<<names[f]<<"\":{";for(int j=0;j<NS;++j){if(j)out<<',';out<<'"'<<stages[j]<<"\":";sumJSON(out,h.flow[f][j]);}out<<'}';}out<<"}}";}
 out<<"}}\n";out.close();require(bool(out),"Histogram output failed");require(std::rename((output+".tmp").c_str(),output.c_str())==0,"Cannot publish histograms");progress(output+".progress.json","complete",n,expected);std::cout<<"STREAM_HIST_PASS "<<n<<' '<<accepted<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<"STREAM_HIST_FAIL "<<e.what()<<'\n';return 1;}}
