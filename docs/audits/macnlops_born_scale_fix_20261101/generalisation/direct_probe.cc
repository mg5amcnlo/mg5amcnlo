#define main macnlops_original_main
#include "MacNLOPSMatch.native.cc"
#undef main
int main(int argc,char**argv){
 try{
  if(argc!=3)throw std::runtime_error("direct_probe INPUT ROUTES");
  int nlegs=0;char compiled[64];macnlops_c_setup(&nlegs,compiled);
  RouteTable routes;routes.read(argv[2]);if(routes.registry!=std::string(compiled,64))throw std::runtime_error("registry");
  double cut[2],xmax[2],floor[2];int nf[2];macnlops_c_profile(cut,xmax,floor,nf);macnlops_c_configure(xmax,&cut[0],&cut[1]);
  RawReader reader(argv[1]);InputEvent input;long index=0,used=0;
  std::cout<<std::setprecision(17);
  while(reader.next(input)){
   long event=index++;auto packet=componentPacket(input);if(packet.registry!=routes.registry)continue;
   auto marker=sourceXMLBlock(input.extra,"macnlops_decision");if(!marker.present)throw std::runtime_error("missing olddecision");auto old=marker.attributes;
   if(!old.count("delta_source_mean")||!old.count("sector")||!old.count("pt"))continue;
   int sector=integer(old.at("sector"));std::vector<LHEParticle> endpoint;
   if(input.kind==2)endpoint=input.hardParticles;else for(const auto&p:input.originalParticles)if(p.status==-1||p.status==1)endpoint.push_back(p);
   if(endpoint.size()!=std::size_t(nlegs))throw std::runtime_error("multiplicity");
   const auto& route=routes.sector(sector);std::vector<LHEParticle> ordered;int matches=0;
   for(const auto& row:route.rows){
    if(row[0]!=endpoint[0].id||row[1]!=endpoint[1].id)continue;
    std::vector<LHEParticle> trial={endpoint[0],endpoint[1]};std::vector<bool>taken(endpoint.size(),false);taken[0]=taken[1]=true;bool valid=true;
    for(int k=2;k<nlegs;++k){int found=-1;for(int j=2;j<nlegs;++j)if(!taken[j]&&endpoint[j].id==row[k]){if(found>=0)throw std::runtime_error("ambiguous outgoing PDG");found=j;}if(found<0){valid=false;break;}taken[found]=true;trial.push_back(endpoint[found]);}
    if(valid){ordered=trial;++matches;}
   }
   if(matches!=1)throw std::runtime_error("no unique native ordering");
   double pt=number(old.at("pt"));auto d=evaluate(ordered,sector,packet,packet.config,packet.graph,packet.flow,input.kind==1?&input.hardParticles:nullptr,true,pt);
   auto repeat=evaluate(ordered,sector,packet,packet.config,packet.graph,packet.flow,input.kind==1?&input.hardParticles:nullptr,true,pt);
   bool identical=d.r==repeat.r&&d.d==repeat.d&&d.k==repeat.k&&d.alphaK==repeat.alphaK&&d.q2fact[0]==repeat.q2fact[0]&&d.q2fact[1]==repeat.q2fact[1]&&d.sudakov.source==repeat.sudakov.source&&d.bornScale.ready==repeat.bornScale.ready&&d.bornScale.coupling==repeat.bornScale.coupling&&d.bornScale.pdf==repeat.bornScale.pdf;
   if(!identical)throw std::runtime_error("repeated request state drift");
   auto ht=[](const std::vector<LHEParticle>&parts){double result=0;for(const auto&p:parts)if(p.status==1)result+=std::sqrt(p.m*p.m+p.px*p.px+p.py*p.py);return result/2.;};
   std::cout<<"GPOINT {\"index\":"<<event<<",\"kind\":"<<input.kind<<",\"sector\":"<<sector<<",\"r\":"<<d.r<<",\"d\":"<<d.d<<",\"k\":"<<d.k<<",\"alphaK\":"<<d.alphaK<<",\"q2fact1\":"<<d.q2fact[0]<<",\"q2fact2\":"<<d.q2fact[1]<<",\"A\":"<<d.sudakov.source<<",\"old_r\":"<<number(old.at("r"))<<",\"old_d\":"<<number(old.at("d"))<<",\"old_k\":"<<number(old.at("k"))<<",\"old_alphaK\":"<<number(old.at("alpha_s_kernel"))<<",\"old_A\":"<<number(old.at("delta_source_mean"))<<",\"ready\":"<<d.bornScale.ready<<",\"coupling\":"<<d.bornScale.coupling<<",\"pdf\":"<<d.bornScale.pdf<<",\"repeat_identical\":true,\"muBorn_HT2\":"<<(input.kind==1?ht(input.hardParticles):0.)<<",\"muReal_HT2\":"<<ht(ordered)<<"}\n";
   ++used;
  }
  std::cout<<"DIRECT_DONE "<<used<<" "<<index<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<"DIRECT_ERROR "<<e.what()<<"\n";return 1;}
}
