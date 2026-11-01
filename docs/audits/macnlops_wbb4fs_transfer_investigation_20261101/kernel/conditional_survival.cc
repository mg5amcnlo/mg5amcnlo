// Diagnostic only: empirical stock survival with the exact saved conditional-Q quadrature mixture.
#include "/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS/Template/NLO/MCatNLO/srcPythia8/MacNLOPS.h"
#include <cstdint>
struct Point {int id;double cutoff,prediction;std::vector<LHEParticle> born;std::vector<double>q,w;};
class ConditionalBorn:public LHAup{
public:
 ConditionalBorn(std::vector<Point> points,int n):points(points),repeats(n){}
 bool setInit()override{setBeamA(2212,6500.);setBeamB(2212,6500.);setStrategy(-4);addProcess(1,1.,0.,1.);return true;}
 double uniform(){std::uint64_t z=(state+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return ((z^(z>>31))>>11)*(1. / 9007199254740992.);}
 bool setEvent(int=0)override{
  if(count>=int(points.size())*repeats)return false;
  const Point&point=points[count/repeats];double u=uniform(),s=0.;std::size_t selected=point.q.size()-1;
  for(std::size_t k=0;k<point.q.size();++k){s+=point.w[k];if(u<s){selected=k;break;}}
  const double Q=point.q[selected];qsum+=Q;const auto&ps=point.born;setProcess(1,1.,Q,1./137.,.118);scales=LHAscales(Q);
  for(std::size_t i=0;i<ps.size();++i)for(std::size_t j=0;j<ps.size();++j){
   if(i==j)continue;
   bool connected=(ps[i].col && (ps[i].col==ps[j].col || ps[i].col==ps[j].acol)) || (ps[i].acol && (ps[i].acol==ps[j].col || ps[i].acol==ps[j].acol));
   if(!connected)continue;
   double start=Q;
   if(i>=2){Vec4 p(ps[i].px+ps[j].px,ps[i].py+ps[j].py,ps[i].pz+ps[j].pz,ps[i].e+ps[j].e);double M=std::sqrt(std::max(0.,p.m2Calc()));double bound=.5*std::sqrt(std::max(0.,(M-ps[j].m-ps[i].m)*(M-ps[j].m+ps[i].m)));start=std::min(Q,bound);}
   scales.attributes[dipoleName(i+1,j+1)]=start;
  }
  infoPtr->scales=&scales;
  for(std::size_t i=0;i<ps.size();++i){const auto&p=ps[i];double legScale=-1.;for(const auto&entry:scales.attributes){const auto prefix=std::string("scalup_")+std::to_string(i+1)+"_";if(entry.first.compare(0,prefix.size(),prefix)==0 && entry.second>0.)legScale=legScale>0.?std::min(legScale,entry.second):entry.second;}addParticle(p.id,p.status,p.mother1,p.mother2,p.col,p.acol,p.px,p.py,p.pz,p.e,p.m,p.lifetime,p.spin,legScale);}
  setIdX(ps[0].id,ps[1].id,ps[0].e/6500.,ps[1].e/6500.);++count;return true;
 }
 std::vector<Point>points;int repeats,count=0;double qsum=0.;LHAscales scales;std::uint64_t state=0x7a9913ac;
};
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("probe POINTS OUTPUT REPEATS SEED SETTINGS MODE"); const std::string mode=argv[6]; if(mode!="ISR"&&mode!="FSR"&&mode!="ALL")throw std::runtime_error("mode");std::ifstream in(argv[1]);int n;in>>n;std::vector<Point>points;
 for(int i=0;i<n;++i){Point p;int np,nq;in>>p.id>>np>>nq>>p.cutoff>>p.prediction;std::string line;std::getline(in,line);for(int k=0;k<np;++k){std::getline(in,line);p.born.push_back(particleRecord(line));}for(int k=0;k<nq;++k){double q,w;in>>q>>w;p.q.push_back(q);p.w.push_back(w);}points.push_back(p);}
 if(!in)throw std::runtime_error("Point input failed");
 Options o;o.seed=std::stoi(argv[4]);o.pdf="LHAPDF6:NNPDF40_nlo_as_01180_nf_4/0";o.alphaSMZ=.1365;o.alphaSorder=1;o.flavourCount=4;o.beamEnergyA=o.beamEnergyB=6500.;o.mcatnloDelta=true;o.settings=argv[5];
 Pythia p("/nfs/home/rikkert/Downloads/pythia8/share/Pythia8/xmldoc",false);configure(p,o);if(mode!="ALL")p.settings.flag(mode=="FSR"?"SpaceShower:QCDshower":"TimeShower:QCDshower",false);auto reader=std::make_shared<ConditionalBorn>(points,std::stoi(argv[3]));auto hook=std::make_shared<FirstEmission>();configureHook(p,o,*hook);p.setLHAupPtr(reader);p.setUserHooksPtr(hook);if(!p.init())throw std::runtime_error("Init failed");validateSourceMatchingSettings(p,o);hook->captureBeamLimits();
 std::ofstream out(argv[2]);out<<std::setprecision(17)<<"index,trials,below,cutoff,prediction,meanQ,warnings,ISR,FSR\n";
 for(const auto&point:points){const int pointSeed=o.seed+1000*point.id;p.rndm.init(pointSeed);reader->state=0x6a09e667f3bcc909ULL ^ (std::uint64_t(pointSeed)*0x9e3779b97f4a7c15ULL);int below=0,isr=0,fsr=0;auto before=nativeTrialCounts(p);double q0=reader->qsum;
  for(int k=0;k<reader->repeats;++k){hook->reset();const int count0=reader->count;bool ok=p.next();validateNominalWeights(p);if(ok || !hook->finished || reader->count!=count0+1)throw std::runtime_error("Repeat/failure");if(!hook->seen || hook->pt<point.cutoff)++below;if(hook->type==2)++isr;if(hook->type==3)++fsr;}
  int warnings=0;for(const auto&d:warningDifference(nativeTrialCounts(p),before))warnings+=std::max(0,d.second);
  out<<point.id<<','<<reader->repeats<<','<<below<<','<<point.cutoff<<','<<point.prediction<<','<<(reader->qsum-q0)/reader->repeats<<','<<warnings<<','<<isr<<','<<fsr<<'\n';out.flush();
 }
 p.settings.writeFile(std::string(argv[2])+".settings.cmnd",true);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
