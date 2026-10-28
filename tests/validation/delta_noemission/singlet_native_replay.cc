// Cross-check the diagnostic Replay against Pythia's native LHE reader.
// Usage: executable XML INPUT_LHE PREFIX NBORN REPEATS SEED
#define main singlet_probe_original_main
#include "singlet_shower_probe.cc"
#undef main

struct Trial { double pt; int type; };

std::vector<Trial> runNativeCheck(const std::string& xml,
    const std::vector<Born>& born,const std::string& expanded,
    const std::string& prefix,int repeats,int seed,bool native) {
  Pythia p(xml,false);common(p,seed);
  auto replay=std::make_shared<Replay>(born,repeats);
  auto hook=std::make_shared<FirstEmission>(replay);p.setUserHooksPtr(hook);
  if(native){setting(p,"Beams:frameType = 4");setting(p,"Beams:LHEF = "+expanded);}
  else {p.setLHAupPtr(replay);setting(p,"Beams:frameType = 5");}
  setting(p,"Beams:setProductionScalesFromLHEF = on");
  setting(p,"Beams:setDipoleShowerStartingScalesFromLHEF = on");
  if(!p.init())throw std::runtime_error("Initialization failed");
  auto model=p.getShowerModelPtr();
  if(!model || !dynamic_cast<SimpleShowerModel*>(model.get()) ||
     !dynamic_cast<SimpleSpaceShower*>(model->getSpaceShower().get()) ||
     !dynamic_cast<SimpleTimeShower*>(model->getTimeShower().get()))
    throw std::runtime_error("Not Simple shower");
  std::string label=native?"native":"replay";
  p.settings.writeFile(prefix+"."+label+".settings.cmnd",true);
  std::ofstream tr(prefix+"."+label+".trials.jsonl");tr<<std::setprecision(17);
  std::vector<Trial> result;long emitted=0;
  for(size_t k=0;k<born.size()*repeats;++k){
    if(native)replay->current=int(k);
    hook->reset();if(!p.next())throw std::runtime_error("Failed shower");
    if(replay->current!=int(k))throw std::runtime_error("Unexpected input consumption");
    if(p.info.nISR()+p.info.nFSRinProc()==0)hook->reset();
    if(hook->pt>born[k/repeats].q*(1.+1e-10))throw std::runtime_error("Above SCALUP");
    if(hook->pt>0.)++emitted;
    result.push_back({hook->pt,hook->type});
    tr<<"{\"event\":"<<k/repeats<<",\"trial\":"<<k%repeats
      <<",\"pt\":"<<hook->pt<<",\"type\":"<<hook->type<<"}\n";
  }
  if(hook->processChecks!=long(result.size()) || hook->maxBornResidual>1e-10)
    throw std::runtime_error("Born process replay audit failed");
  std::cout<<std::setprecision(17)<<"AUDIT "<<label<<" process_checks="<<hook->processChecks
    <<" maximum_relative_Born_replay_difference="<<hook->maxBornResidual
    <<" emitted="<<emitted<<" trials="<<result.size()<<'\n';
  p.stat();return result;
}

int main(int argc,char**argv){try{
  if(argc!=7)throw std::runtime_error("See source usage");
  std::string xml=argv[1],input=argv[2],prefix=argv[3];
  int n=std::stoi(argv[4]),repeats=std::stoi(argv[5]),seed=std::stoi(argv[6]);
  auto born=readBorn(input);if(n<=0 || n>int(born.size()))throw std::runtime_error("Bad Born count");
  born.resize(n);std::string expanded=prefix+".expanded.lhe";
  {std::ofstream o(expanded);o<<std::setprecision(17);writeLHEHeader(o);
    for(auto&b:born)for(int i=0;i<repeats;++i)writeEvent(o,b);
    o<<"</LesHouchesEvents>\n";}
  auto a=runNativeCheck(xml,born,expanded,prefix,repeats,seed,false);
  auto b=runNativeCheck(xml,born,expanded,prefix,repeats,seed,true);
  int differences=0;double maxDifference=0.;
  for(size_t i=0;i<a.size();++i){
    if(a[i].pt!=b[i].pt || a[i].type!=b[i].type)++differences;
    maxDifference=std::max(maxDifference,std::abs(a[i].pt-b[i].pt));
  }
  std::ofstream o(prefix+".result.json");o<<std::setprecision(17)
    <<"{\"born\":"<<n<<",\"repeats\":"<<repeats<<",\"trials_per_reader\":"<<a.size()
    <<",\"seed\":"<<seed<<",\"different_trials\":"<<differences
    <<",\"maximum_absolute_pt_difference\":"<<maxDifference<<"}\n";
  std::cout<<"RESULT trials="<<a.size()<<" differences="<<differences
    <<" max_absolute_pt_difference="<<maxDifference<<'\n';
  return differences?2:0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
