// Inspect the stock LHE import and native shower caps, optionally continue
// each saved endpoint through the full shower. Pythia itself is unmodified.
// Usage: probe INPUT.lhe Pythia8.cmd OUTPUT.csv inspect|full
#include "MacNLOPS.h"

std::vector<NativeDipoleScale> auditScales(Pythia& engine, bool isr) {
  std::string listing;
  {
    CaptureNativeScaleListing capture;
    if (isr) engine.getShowerModelPtr()->getSpaceShower()->list();
    else engine.getShowerModelPtr()->getTimeShower()->list();
    listing = capture.str();
  }
  // Unlike the QCD-only matching profile, the ordinary shower card may
  // create additional non-QCD dipoles. Inspect the coloured rows only.
  std::vector<NativeDipoleScale> result;
  std::istringstream lines(listing);
  std::string line;
  while (std::getline(lines,line)) {
    const std::string row = trim(line);
    if (row.empty() || !std::isdigit(static_cast<unsigned char>(row[0]))) continue;
    const auto f = fields(row,isr ? 10 : 22,"native scale audit row");
    if (!integer(f[isr ? 6 : 4])) continue;
    result.push_back({integer(f[isr ? 3 : 1]),integer(f[isr ? 4 : 2]),
      isr ? integer(f[2]) : 0,number(f[isr ? 5 : 3])});
  }
  return result;
}

class ScaleAudit : public UserHooks {
public:
  Pythia* engine = nullptr;
  const InputEvent* input = nullptr;
  bool full = false, captured = false, finished = false;
  double global = 0., globalError = 0., productionError = 0., dipoleError = 0.;
  double first = 0., maximum = 0., beam[2] = {0.,0.}, fsrMax = 0.;
  int emissions = 0, missingPairs = 0, fsrPairs = 0;
  std::map<std::pair<int,int>,double> pairs;
  std::map<int,double> legs;

  void reset(const InputEvent& point) {
    input = &point; captured = finished = false;
    global = globalError = productionError = dipoleError = 0.;
    first = maximum = beam[0] = beam[1] = fsrMax = 0.;
    emissions = missingPairs = fsrPairs = 0;
    pairs.clear(); legs.clear();
    const auto scales = sourceXMLBlock(point.extra,"scales");
    for (const auto& a : scales.attributes) {
      if (a.first.compare(0,7,"scalup_") != 0) continue;
      const std::string tail = a.first.substr(7);
      const auto split = tail.find('_');
      const int rad = integer(tail.substr(0,split));
      const double value = number(a.second);
      if (split != std::string::npos)
        pairs[{rad,integer(tail.substr(split+1))}] = value;
      if (value > 0.)
        legs[rad] = legs.count(rad) ? std::min(legs[rad],value) : value;
    }
  }

  bool canVetoProcessLevel() override {return true;}
  bool doVetoProcessLevel(Event& process) override {
    global = process.scale();
    globalError = std::abs(global-input->scale)/std::max(1.,input->scale);
    const auto order = nativeProcessOrder(input->particles);
    if (process.size() != int(order.size())+2)
      throw std::runtime_error("Unexpected process multiplicity");
    for (std::size_t i=1; i<order.size(); ++i) {
      const auto& q = input->particles.at(order[i]-1);
      const auto& p = process[i+2];
      if (p.id()!=q.id || p.col()!=q.col || p.acol()!=q.acol)
        throw std::runtime_error("Process order/colour mismatch");
      if (order[i] != int(i))
        throw std::runtime_error("Probe requires flat canonical endpoint order");
      const double expected = legs.count(i) ? legs.at(i) : global;
      productionError = std::max(productionError,
        std::abs(p.scale()-expected)/std::max(1.,global));
    }
    if (!infoPtr->scales || infoPtr->scales->SCALUP!=input->scale)
      throw std::runtime_error("Raw SCALUP changed in the reader");
    return false;
  }

  bool canVetoPT() override {return true;}
  double scaleVetoPT() override {return 1.e20;}
  bool doVetoPT(int, const Event& event) override {
    if (captured) throw std::runtime_error("Duplicate initial-cap hook");
    captured = true;
    for (const auto& d : auditScales(*engine,true)) {
      if (d.side<1 || d.side>2) throw std::runtime_error("Bad ISR side");
      const int rad = d.radiator-2;
      const double expected = legs.count(rad) ? legs.at(rad) : global;
      dipoleError = std::max(dipoleError,std::abs(d.scale-expected)/std::max(1.,global));
      beam[d.side-1] = std::min(global,d.scale);
    }
    for (const auto& d : auditScales(*engine,false)) {
      const int rad = d.radiator-2, rec = d.recoiler-2;
      const auto key = std::make_pair(rad,rec);
      const double expected = pairs.count(key) ? pairs.at(key)
        : legs.count(rad) ? legs.at(rad) : global;
      dipoleError = std::max(dipoleError,std::abs(d.scale-expected)/std::max(1.,global));
      missingPairs += !pairs.count(key);
      ++fsrPairs; fsrMax = std::max(fsrMax,std::min(global,d.scale));
      if (event[d.radiator].id()!=input->particles.at(rad-1).id)
        throw std::runtime_error("Event/process ordering changed");
    }
    finished = !full;
    return !full;
  }

  bool canVetoStep() override {return full;}
  int numberVetoStep() override {return 100000;}
  bool doVetoStep(int, int, int, const Event&) override {
    const double pt = infoPtr->pTnow();
    if (!emissions) first = pt;
    maximum = std::max(maximum,pt); ++emissions;
    if (pt > global*(1.+1.e-9))
      throw std::runtime_error("Accepted emission exceeds imported global cap");
    return false;
  }
};

double hardPt(const Event& event) {
  Vec4 sum;
  for (int i=1; i<event.size(); ++i)
    if (event[i].isFinal()) {
      const int id = event[i].idAbs();
      if (id==6 || (id>=11 && id<=16) || id==23 || id==24 || id==25)
        sum += event[i].p();
    }
  return sum.pT();
}

int main(int argc,char** argv) {try {
  if (argc!=5 && argc!=6) throw std::runtime_error("Usage: probe INPUT CMD OUTPUT inspect|full [qed-off]");
  const bool full = std::string(argv[4])=="full";
  if (!full && std::string(argv[4])!="inspect") throw std::runtime_error("Unknown mode");
  Pythia p;
  if (!p.readFile(argv[2])) throw std::runtime_error("Cannot read settings");
  p.readString(std::string("Beams:LHEF = ")+argv[1]);
  p.readString("Check:abortIfVeto = on");
  p.readString("Next:numberShowEvent = 0");
  p.readString("Next:numberCount = 0");
  if (argc==6) {
    if (std::string(argv[5])!="qed-off") throw std::runtime_error("Unknown option");
    p.readString("TimeShower:QEDshowerByOther = off");
  }
  // The normal Pythia83 continuation driver applies the same override.
  p.settings.forceParm("SpaceShower:pT0Ref",0.);
  auto hook = std::make_shared<ScaleAudit>();
  hook->engine = &p; hook->full = full; p.setUserHooksPtr(hook);
  if (!p.init()) throw std::runtime_error("Pythia initialization failed");
  RawReader reader(argv[1]); InputEvent point;
  std::ofstream out(argv[3]); out<<std::setprecision(17);
  out<<"index,global,global_error,production_error,dipole_error,beam1,beam2,max_fsr,missing_fsr_pairs,fsr_pairs,first_emission,max_emission,emissions,final_pt,photons\n";
  int index = 0;
  while (reader.next(point)) {
    hook->reset(point);
    // Identical per-event random streams across scale-control variants.
    p.rndm.init(3100001+index);
    const bool ok = p.next();
    if ((full && !ok) || (!full && (ok || !hook->finished)) || !hook->captured)
      throw std::runtime_error("Shower failed or initial caps were not captured at event "+std::to_string(index));
    if (hook->globalError>2.e-12 || hook->productionError>2.e-12 || hook->dipoleError>2.e-12)
      throw std::runtime_error("Imported shower scale mismatch at event "+std::to_string(index));
    int photons=0;
    if (full) for (int j=1; j<p.event.size(); ++j)
      photons += p.event[j].isFinal() && p.event[j].id()==22;
    out<<index<<','<<hook->global<<','<<hook->globalError<<','<<hook->productionError
       <<','<<hook->dipoleError<<','<<hook->beam[0]<<','<<hook->beam[1]<<','<<hook->fsrMax
       <<','<<hook->missingPairs<<','<<hook->fsrPairs<<','<<hook->first<<','<<hook->maximum
       <<','<<hook->emissions<<','<<(full ? hardPt(p.event) : 0.)<<','<<photons<<'\n';
    ++index;
  }
  if (!out) throw std::runtime_error("Output failure");
  std::cout<<"Checked "<<index<<" stock-Pythia continuation imports; mode="<<argv[4]<<'\n';
  p.stat();
  return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
