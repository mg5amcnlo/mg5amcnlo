// Per-channel MAcNLOPS first emission for sampled MC@NLO-Delta sources.
// One stock Pythia 8.318 step per S event, then the H removal and S veto
// decided in process with this P* directory's densities (driver_macnlops.f
// through macnlops_capi.f). Every input event is written; a rejected event
// keeps its nominal weight and gets <wgt id='macnlops'> zero, unless
// --drop-rejected omits it. Build: make macnlops_onestep in SubProcesses/P*.
#include "MacNLOPS.h"

#include <cstdint>

extern "C" {
void macnlops_c_setup(int* nlegs, char* registry);
void macnlops_c_profile(double* cutoffs, double* xmax, double* masses, int* flavours);
void macnlops_c_configure(double* xmax, double* isr, double* fsr);
void macnlops_c_density(int owner, const int* ids, const double* lab, int config,
  int graph, int flow, int damping, double symmetry, int hasBorn, const int* bornIds,
  const int* bornColours, const double* bornLab, const double* bornMasses,
  double* r, double* d, double* k, double* alphas, double* q2fact, int* status);
}

namespace {

// The archived MC@NLO-Delta Sudakov tables behave as PYTHIA's default shower
// coupling at one loop without CMW, with pT0Ref = 0 (forced by configure()).
// The tables are never regenerated, so the one-step shower uses the same
// coupling: docs/macnlops_parameter_consistency.md. macnlops.py's
// TABLE_ALPHA_S and TABLE_ALPHA_S_ORDER must agree.
const double tableAlphaS = 0.1365;
const int tableAlphaSOrder = 1;
const char* prescriptionV3 = "component-undamped-delta-v3";
const char* prescriptionV2 = "component-undamped-delta-v2";

struct MatchOptions {
  std::string input, output, summary, routes, settings, pdf, res;
  int seed = 1;
  bool dropRejected = false, pdfCorrection = true;
};

MatchOptions matchOptions(int argc, char** argv) {
  MatchOptions m;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "macnlops_onestep --input IN.lhe --output OUT.lhe --summary OUT.json\n"
                << " --routes macnlops_routes.dat --settings macnlops_shower.cmnd\n"
                << " --pdf LHAPDF6:SET/MEMBER --seed N [--res res_1]\n"
                << " [--drop-rejected] [--s-veto v3|v2]\n"
                << "Showers one step with alpha_s(MZ) = 0.1365 at one loop, applies the\n"
                << "H removal and S veto, and writes all events with a 'macnlops' weight.\n";
      std::exit(0);
    }
    if (arg == "--drop-rejected") { m.dropRejected = true; continue; }
    if (++i == argc) throw std::runtime_error("Missing value for " + arg);
    const std::string value = argv[i];
    if (arg == "--input") m.input = value;
    else if (arg == "--output") m.output = value;
    else if (arg == "--summary") m.summary = value;
    else if (arg == "--routes") m.routes = value;
    else if (arg == "--settings") m.settings = value;
    else if (arg == "--pdf") m.pdf = value;
    else if (arg == "--res") m.res = value;
    else if (arg == "--seed") m.seed = std::stoi(value);
    else if (arg == "--s-veto") {
      if (value != "v2" && value != "v3")
        throw std::runtime_error("--s-veto must be v2 or v3");
      m.pdfCorrection = value == "v3";
    } else throw std::runtime_error("Unknown option: " + arg);
  }
  if (m.input.empty() || m.output.empty() || m.summary.empty() || m.routes.empty() ||
      m.settings.empty() || m.pdf.empty())
    throw std::runtime_error("--input, --output, --summary, --routes, --settings and --pdf are required");
  if (m.input == m.output || m.input == m.summary || m.output == m.summary)
    throw std::runtime_error("Input and output paths must be distinct");
  if (m.res.empty()) {
    const std::size_t slash = m.input.rfind('/');
    m.res = (slash == std::string::npos ? std::string() : m.input.substr(0, slash + 1)) + "res_1";
  }
  return m;
}

// sigma_S/sigma_B of the integration channel (macnlops.channel_normalisation).
struct ChannelNormalisation {
  double sigmaS = 0., errorS = 0., sigmaB = 0., errorB = 0., beta = 0.;
};

ChannelNormalisation channelNormalisation(const std::string& file) {
  std::ifstream input(file.c_str());
  if (!input) throw std::runtime_error("Cannot read the channel integrals in " + file);
  ChannelNormalisation result;
  bool foundS = false, foundB = false;
  std::string line;
  while (std::getline(input, line)) {
    const std::string row = trim(line);
    for (const std::string label : {"S", "Born"}) {
      const std::string prefix = "Final result [" + label + "]:";
      if (row.compare(0, prefix.size(), prefix)) continue;
      std::istringstream values(row.substr(prefix.size()));
      std::string value, separator, error;
      if (!(values >> value >> separator >> error) || separator != "+/-")
        throw std::runtime_error("Malformed channel integral in " + file);
      if (label == "S") { result.sigmaS = number(value); result.errorS = number(error); foundS = true; }
      else { result.sigmaB = number(value); result.errorB = number(error); foundB = true; }
    }
  }
  if (!foundS || !foundB)
    throw std::runtime_error(file + " lacks the channel S and Born integrals; regenerate the "
                             "source with the current integrator");
  if (!(result.sigmaS > 0.) || !(result.sigmaB > 0.) || !std::isfinite(result.sigmaS) ||
      !std::isfinite(result.sigmaB) || !std::isfinite(result.errorS) || !std::isfinite(result.errorB))
    throw std::runtime_error("Nonpositive channel S or Born cross section in " + file);
  result.beta = result.sigmaS / result.sigmaB;
  return result;
}

// Discrete routes of this P* directory's registry (macnlops.write_route_table).
struct RouteSector {
  int sector = 0, i = 0, j = 0, ij = 0;
  bool qcd = false;
  std::set<std::vector<int> > rows;
};

struct RouteTable {
  std::string registry, directory;
  int nexternal = 0, nborn = 0;
  std::set<std::vector<int> > bornPdgs;
  std::vector<int> stateClass;
  std::vector<RouteSector> sectors;
  // (a, b, config, graph, flow) -> transported (config, graph, flow), a < b.
  std::map<std::vector<int>, std::vector<int> > transport;

  void read(const std::string& file) {
    std::ifstream input(file.c_str());
    if (!input) throw std::runtime_error("Cannot read route table " + file);
    auto expect = [&](const std::string& word) {
      std::string found;
      if (!(input >> found) || found != word)
        throw std::runtime_error("Malformed route table " + file + ": expected " + word);
    };
    auto integers = [&](int count) {
      std::vector<int> values(count);
      for (int& value : values)
        if (!(input >> value)) throw std::runtime_error("Truncated route table " + file);
      return values;
    };
    expect("MACNLOPS_ROUTES_V1");
    if (!(input >> registry >> directory >> nexternal >> nborn) ||
        registry.size() != 64 || nexternal < 4 || nborn != nexternal - 1)
      throw std::runtime_error("Invalid route table header in " + file);
    int count = 0;
    expect("BORN_PDGS");
    if (!(input >> count) || count < 1) throw std::runtime_error("Invalid Born rows in " + file);
    for (int row = 0; row < count; ++row) bornPdgs.insert(integers(nborn));
    expect("STATE_CLASSES");
    stateClass = integers(nborn);
    expect("SECTORS");
    if (!(input >> count) || count < 1) throw std::runtime_error("Invalid sectors in " + file);
    for (int s = 0; s < count; ++s) {
      expect("SECTOR");
      RouteSector sector;
      int qcd = 0, rows = 0;
      if (!(input >> sector.sector >> sector.i >> sector.j >> sector.ij >> qcd >> rows) || rows < 1)
        throw std::runtime_error("Invalid sector row in " + file);
      sector.qcd = qcd != 0;
      for (int row = 0; row < rows; ++row) sector.rows.insert(integers(nexternal));
      sectors.push_back(sector);
    }
    expect("TRANSPORT");
    if (!(input >> count) || count < 0) throw std::runtime_error("Invalid transport size in " + file);
    for (int row = 0; row < count; ++row) {
      const std::vector<int> values = integers(8);
      if (!transport.emplace(std::vector<int>(values.begin(), values.begin() + 5),
                             std::vector<int>(values.begin() + 5, values.end())).second)
        throw std::runtime_error("Duplicate Born transport row in " + file);
    }
    expect("END");
  }

  const RouteSector& sector(int number) const {
    for (const auto& item : sectors) if (item.sector == number) return item;
    throw std::runtime_error("Component sector " + std::to_string(number) + " is not in the registry");
  }
};

struct ComponentPacket {
  std::string registry;
  int sector = 0, config = 0, graph = 0, flow = 0;
  double flowProbability = 0., insertion = 0., sourceFactor = 0., symmetry = 0.;
};

// prepareSourceMatching has already validated this packet's contract.
ComponentPacket componentPacket(const InputEvent& input) {
  const auto block = sourceXMLBlock(input.extra, "macnlops_component");
  if (!block.present || !input.hasHard || input.hardPartition != "sampled" ||
      input.hardSource != "delta")
    throw std::runtime_error("The first-emission program requires sampled MC@NLO-Delta "
                             "sources with component packets");
  std::istringstream rows(block.contents);
  std::vector<std::string> data;
  std::string line;
  while (std::getline(rows, line)) if (!trim(line).empty()) data.push_back(line);
  if (data.size() != 2) throw std::runtime_error("Invalid sampled-component row count");
  const auto labels = fields(data[0], 4, "sampled-component labels");
  const auto weights = fields(data[1], 4, "sampled-component weights");
  ComponentPacket c;
  c.registry = block.attributes.at("registry");
  c.sector = integer(labels[0]); c.config = integer(labels[1]);
  c.graph = integer(labels[2]); c.flow = integer(labels[3]);
  c.flowProbability = number(weights[0]); c.insertion = number(weights[1]);
  c.sourceFactor = number(weights[2]); c.symmetry = number(weights[3]);
  return c;
}

// R, the subtraction D (R - H) and the leading kernel K of one component,
// with K's emission coupling and squared factorisation scales.
struct Density {
  double r = 0., d = 0., k = 0., alphaK = 0., q2fact[2] = {0., 0.};
  bool transfers() const {return d > r && k > 0.;}
};

// Signed-residual acceptances of macnlops.Densities. A nonpositive K
// transfers nothing; kernelScale rescales only the S denominator.
double sAcceptance(const Density& x, double kernelScale) {
  if (!std::isfinite(kernelScale) || !(kernelScale > 0.))
    throw std::runtime_error("Invalid S-veto kernel normalisation");
  if (!std::isfinite(x.r) || !std::isfinite(x.d) || !std::isfinite(x.k) ||
      !std::isfinite(x.d - x.r))
    throw std::runtime_error("Nonfinite real, subtraction or shower density");
  const double negative = std::max(x.d - x.r, 0.);
  if (negative == 0. || x.k <= 0.) return 1.;
  return 1. - std::min(std::min(negative / x.k, 1.) / kernelScale, 1.);
}

double hAcceptance(const Density& x) {
  sAcceptance(x, 1.);
  const double negative = std::max(x.d - x.r, 0.);
  if (negative == 0.)
    throw std::runtime_error("Negative H event disagrees with reconstructed hard density");
  return std::max(1. - std::max(x.k, 0.) / negative, 0.);
}

Density evaluate(const std::vector<LHEParticle>& legs, int sector, const ComponentPacket& c,
                 int config, int graph, int flow, const std::vector<LHEParticle>* born) {
  const int n = legs.size();
  std::vector<int> ids(n), bornIds(n - 1, 0), bornColours(2 * (n - 1), 0);
  std::vector<double> lab(4 * n), bornLab(4 * (n - 1), 0.), bornMasses(n - 1, 0.);
  for (int i = 0; i < n; ++i) {
    ids[i] = legs[i].id;
    lab[4*i] = legs[i].e; lab[4*i+1] = legs[i].px; lab[4*i+2] = legs[i].py; lab[4*i+3] = legs[i].pz;
  }
  if (born) {
    if (int(born->size()) != n - 1) throw std::runtime_error("Invalid local Born particle count");
    for (int i = 0; i < n - 1; ++i) {
      const LHEParticle& p = (*born)[i];
      bornIds[i] = p.id; bornColours[2*i] = p.col; bornColours[2*i+1] = p.acol;
      bornLab[4*i] = p.e; bornLab[4*i+1] = p.px; bornLab[4*i+2] = p.py; bornLab[4*i+3] = p.pz;
      bornMasses[i] = p.m;
    }
  }
  Density x;
  int status = 1;
  macnlops_c_density(sector, ids.data(), lab.data(), config, graph, flow, 1, c.symmetry,
    born ? 1 : 0, bornIds.data(), bornColours.data(), bornLab.data(), bornMasses.data(),
    &x.r, &x.d, &x.k, &x.alphaK, x.q2fact, &status);
  if (status == 23)
    throw std::runtime_error("Pythia heavy-flavour threshold conversion needs its own density "
                             "and inverse map");
  if (status == 24)
    throw std::runtime_error("Hard-model mass-shell mismatch or unavailable massive "
                             "emitted-parton map");
  if (status)
    throw std::runtime_error("Native density evaluator failed with status " + std::to_string(status));
  if (!std::isfinite(x.r) || !std::isfinite(x.d) || !std::isfinite(x.k))
    throw std::runtime_error("Nonfinite native density");
  return x;
}

// macnlops.check_hard_mass_coordinates, against the hard-model masses that
// the shower settings install in particleData.
void checkModelMasses(const std::vector<LHEParticle>& legs, const Pythia& engine) {
  for (const auto& p : legs) {
    if (p.status != -1 && p.status != 1) continue;
    const double model = std::abs(engine.particleData.m0(p.id));
    if (std::abs(p.m - model) > 1.e-10 * std::max(1., std::max(std::abs(p.m), model)))
      throw std::runtime_error("LHE PDG " + std::to_string(p.id) +
        " has a shower mass differing from the hard model; an inverse write-time transport "
        "and its density Jacobian are required");
  }
}

bool sameMomentum(const Pythia8::Particle& a, const LHEParticle& b) {
  const double first[4] = {a.e(), a.px(), a.py(), a.pz()}, second[4] = {b.e, b.px, b.py, b.pz};
  const double tolerance = 5.e-10 * std::max(1., std::max(std::abs(first[0]), std::abs(second[0])));
  for (int i = 0; i < 4; ++i)
    if (std::abs(first[i] - second[i]) >
        std::max(5.e-10 * std::max(std::abs(first[i]), std::abs(second[i])), tolerance))
      return false;
  return true;
}

bool isQuark(int id) {return std::abs(id) >= 1 && std::abs(id) <= 6;}

bool qcdBranch(bool isr, int before, int after, int emitted) {
  if (isr)
    return (before == after && emitted == 21 && (before == 21 || isQuark(before))) ||
           (after == 21 && isQuark(before) && emitted == -before) ||
           (before == 21 && isQuark(after) && emitted == after);
  return (isQuark(before) && ((after == before && emitted == 21) ||
                              (emitted == before && after == 21))) ||
         (before == 21 && ((after == 21 && emitted == 21) ||
                           (isQuark(after) && emitted == -after)));
}

struct Branch {
  bool isr = false;
  int parent = -1, radiator = -1, sister = -1;
};

struct Candidate {
  int sector;
  std::vector<int> permutation, bornPermutation;
  std::string orientation;
};

// Port of macnlops_components.branch_candidates on the live snapshot.
std::vector<Candidate> branchCandidates(const RouteTable& routes, const InputEvent& input,
    const FirstEmission& hook, const std::vector<LHEParticle>& endpoint, Branch& branch) {
  const std::vector<LHEParticle>& source = input.particles;
  const int nborn = source.size(), nreal = endpoint.size();
  if (nborn < 3 || nreal != nborn + 1 || routes.nexternal != nreal)
    throw std::runtime_error("First branching must add exactly one external parton");
  std::vector<int> bornIds;
  for (const auto& p : source) bornIds.push_back(p.id);
  if (!routes.bornPdgs.count(bornIds))
    throw std::runtime_error("Source Born ordering does not match its native provider");
  const Event& snapshot = hook.snapshot;
  std::vector<int> endpointIndices = hook.incoming;
  endpointIndices.insert(endpointIndices.end(), hook.outgoing.begin(), hook.outgoing.end());
  if (int(endpointIndices.size()) != nreal ||
      int(std::set<int>(endpointIndices.begin(), endpointIndices.end()).size()) != nreal)
    throw std::runtime_error("Endpoint requires distinct external Pythia indices");
  for (int leg = 0; leg < nreal; ++leg) {
    const int index = endpointIndices[leg];
    if (index < 1 || index >= snapshot.size() || snapshot[index].id() != endpoint[leg].id ||
        snapshot[index].col() != endpoint[leg].col || snapshot[index].acol() != endpoint[leg].acol ||
        !sameMomentum(snapshot[index], endpoint[leg]))
      throw std::runtime_error("Endpoint particles disagree with the Pythia snapshot");
  }
  std::vector<int> origins;
  for (int leg = 0; leg < nborn; ++leg) {
    int found = 0, matches = 0;
    for (int i = 1; i < snapshot.size(); ++i) {
      const int status = snapshot[i].statusAbs();
      if ((leg < 2 ? status != 21 : status != 22 && status != 23) ||
          snapshot[i].id() != source[leg].id || snapshot[i].col() != source[leg].col ||
          snapshot[i].acol() != source[leg].acol || !sameMomentum(snapshot[i], source[leg]))
        continue;
      found = i; ++matches;
    }
    if (matches != 1)
      throw std::runtime_error("Ambiguous or missing original Born leg " + std::to_string(leg + 1));
    origins.push_back(found);
  }
  std::map<int, int> originToLeg;
  for (int leg = 0; leg < nborn; ++leg)
    if (!originToLeg.emplace(origins[leg], leg).second)
      throw std::runtime_error("Original Born legs do not have distinct ancestry");
  int before = 0, after = 0, emitted = 0;
  for (int i = 1; i < snapshot.size(); ++i) {
    const Pythia8::Particle& p = snapshot[i];
    if (hook.type == 2 && p.status() == 43) {
      emitted = i; after = p.mother1(); before = snapshot[after].daughter2();
    } else if (hook.type == 3 && p.status() == 51) {
      before = p.mother1(); after = snapshot[before].daughter1();
      emitted = snapshot[before].daughter2();
    }
  }
  const auto position = [&](int index) {
    return int(std::find(endpointIndices.begin(), endpointIndices.end(), index) -
               endpointIndices.begin());
  };
  if (!originToLeg.count(before) || position(after) == nreal || position(emitted) == nreal)
    throw std::runtime_error("Branch indices do not identify source and endpoint legs");
  branch.isr = hook.type == 2;
  branch.parent = originToLeg.at(before);
  branch.radiator = position(after);
  branch.sister = position(emitted);
  if ((hook.type != 2 && hook.type != 3) || branch.radiator == branch.sister ||
      branch.sister < 2 ||
      (branch.isr && (branch.parent >= 2 || branch.radiator != branch.parent)) ||
      (!branch.isr && (branch.parent < 2 || branch.radiator < 2)))
    throw std::runtime_error("Branch kind, beam or external daughter indices disagree");
  if (!qcdBranch(branch.isr, bornIds[branch.parent], endpoint[branch.radiator].id,
                 endpoint[branch.sister].id))
    throw std::runtime_error("Recorded flavours are not a QCD branching");

  // Incoming identities follow their beams. Final-state spectators, including
  // global-recoil copies, follow their own ancestry to one source Born leg.
  std::map<int, int> descendants = {{0, 0}, {1, 1}};
  for (int leg = 2; leg < nreal; ++leg) {
    if (leg == branch.sister) continue;
    std::set<int> found, visited;
    std::vector<int> pending(1, endpointIndices[leg]);
    while (!pending.empty()) {
      const int current = pending.back();
      pending.pop_back();
      if (current == 0 || !visited.insert(current).second) continue;
      const auto origin = originToLeg.find(current);
      if (origin != originToLeg.end()) { found.insert(origin->second); continue; }
      if (current < 0 || current >= snapshot.size())
        throw std::runtime_error("Ancestry refers to a missing Pythia record");
      pending.push_back(snapshot[current].mother1());
      pending.push_back(snapshot[current].mother2());
    }
    if (found.size() != 1)
      throw std::runtime_error("Endpoint spectator has ambiguous Born ancestry");
    const int sourceLeg = *found.begin();
    if (sourceLeg < 2 || !descendants.emplace(sourceLeg, leg).second)
      throw std::runtime_error("Two endpoint particles represent the same Born leg");
  }
  if (int(descendants.size()) != nborn || descendants.at(branch.parent) != branch.radiator)
    throw std::runtime_error("Endpoint does not preserve the source Born leg identities");

  struct Orientation { int i, j; std::string tag; };
  std::vector<Orientation> orientations = {{branch.sister, branch.radiator, "native"}};
  if (!branch.isr && bornIds[branch.parent] == 21)
    orientations.push_back({branch.radiator, branch.sister,
      endpoint[branch.sister].id == 21 ? "gg-swapped" : "gqq-swapped"});
  std::vector<Candidate> candidates;
  std::set<std::pair<int, std::vector<int> > > seen;
  for (const auto& sector : routes.sectors) {
    if (!sector.qcd) continue;
    const int i = sector.i - 1, j = sector.j - 1, father = sector.ij - 1;
    if (!(2 <= i && i < nreal && 0 <= j && j < nreal && i != j))
      throw std::runtime_error("Invalid native FKS pair");
    std::vector<int> bornSlots;
    for (int k = 0; k < nreal; ++k) if (k != i) bornSlots.push_back(k);
    const int slot = std::find(bornSlots.begin(), bornSlots.end(), j) - bornSlots.begin();
    if (father < 0 || father >= nborn || slot != father || branch.isr != (j < 2)) continue;
    std::vector<int> bornPermutation(nborn);
    for (int k = 0; k < nborn; ++k) bornPermutation[k] = k;
    if (father != branch.parent) {
      if (father < 2 || branch.parent < 2 || bornIds[father] != bornIds[branch.parent] ||
          routes.stateClass[father] != routes.stateClass[branch.parent])
        continue;
      std::swap(bornPermutation[father], bornPermutation[branch.parent]);
    }
    for (const auto& orientation : orientations) {
      std::vector<int> permutation(nreal, -1);
      for (int bornLeg = 0; bornLeg < nborn; ++bornLeg)
        permutation[bornSlots[bornLeg]] = descendants.at(bornPermutation[bornLeg]);
      permutation[i] = orientation.i;
      permutation[j] = orientation.j;
      std::vector<int> sorted = permutation;
      std::sort(sorted.begin(), sorted.end());
      bool valid = permutation[0] == 0 && permutation[1] == 1;
      for (int k = 0; k < nreal && valid; ++k) valid = sorted[k] == k;
      if (!valid) continue;
      std::vector<int> ids;
      for (int k : permutation) ids.push_back(endpoint[k].id);
      if (!sector.rows.count(ids) || !seen.insert(std::make_pair(sector.sector, permutation)).second)
        continue;
      candidates.push_back({sector.sector, permutation, bornPermutation, orientation.tag});
    }
  }
  if (candidates.empty())
    throw std::runtime_error("Actual branching has no certified native source component");
  return candidates;
}

std::uint64_t splitmix(std::uint64_t& state) {
  std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// Exactly representable uniform in [0,1); separate from Pythia's stream.
class Uniform {
public:
  explicit Uniform(std::uint64_t seed) : state(seed) {}
  double operator()() {return (splitmix(state) >> 11) * (1. / 9007199254740992.);}
private:
  std::uint64_t state;
};

struct Decision {
  std::string rule;
  double keep = 1., pt = 0., matchingScale = 0., kernelScale = 1.;
  double alphaShower = 0., pdfRatio = 1., marking = 1., totalKernel = 0.;
  bool evaluated = false;
  int candidates = 0, sector = 0;
  std::string orientation;
  Density density;
};

struct Context {
  RouteTable routes;
  ChannelNormalisation normalisation;
  double cutoffs[2] = {0., 0.}, floors[2] = {0., 0.};
  bool pdfCorrection = true;
  Pythia* engine = nullptr;
  double eBeam[2] = {0., 0.};
};

// macnlops.matching_scale: R transfers to S below the actual Born boundary.
double matchingScale(const Context& c, const InputEvent& input) {
  double scale = std::max(c.cutoffs[0], c.cutoffs[1]);
  for (const auto& p : input.particles)
    if (p.status == -1 && (std::abs(p.id) == 4 || std::abs(p.id) == 5))
      scale = std::max(scale, std::sqrt(2.) * c.floors[std::abs(p.id) - 4]);
  return scale;
}

double beamX(const Context& c, const LHEParticle& p, int side) {
  return (p.e + (side ? -p.pz : p.pz)) / (2. * c.eBeam[side]);
}

// The shower's backward-evolution PDF ratio at pT2 over K's at its muF2:
// x' f_a'(x', Q2) / x f_a(x, Q2) for the radiating beam. FSR has none.
double pdfRatio(const Context& c, const FirstEmission& hook, const InputEvent& input,
                const std::vector<LHEParticle>& endpoint, const Branch& branch, const Density& x) {
  if (!branch.isr || !c.pdfCorrection) return 1.;
  const int side = branch.parent;
  const LHEParticle& born = input.particles[side];
  const LHEParticle& real = endpoint[side];
  const double xBorn = beamX(c, born, side), xReal = beamX(c, real, side);
  const double q2Kernel = x.q2fact[side], q2Shower = hook.pt * hook.pt;
  const double shower = hook.beamXf(side, real.id, xReal, q2Shower) /
    hook.beamXf(side, born.id, xBorn, q2Shower);
  const double kernel = hook.beamXf(side, real.id, xReal, q2Kernel) /
    hook.beamXf(side, born.id, xBorn, q2Kernel);
  const double ratio = shower / kernel;
  if (!(xBorn > 0. && xReal > xBorn && xReal <= 1. && q2Kernel > 0.) || !std::isfinite(ratio) ||
      !(ratio > 0.))
    throw std::runtime_error("Invalid shower/kernel PDF ratio for the S veto");
  return ratio;
}

Decision decideS(const Context& c, Uniform& uniform, const InputEvent& input,
                 const FirstEmission& hook, const std::vector<LHEParticle>& endpoint) {
  Decision x;
  if (!hook.seen) { x.rule = "exhausted"; return x; }
  x.pt = hook.pt;
  x.matchingScale = matchingScale(c, input);
  if (!std::isfinite(x.pt) || x.pt < 0.)
    throw std::runtime_error("Invalid first-emission evolution scale");
  if (x.pt <= x.matchingScale) { x.rule = "below_matching_scale"; return x; }
  const ComponentPacket component = componentPacket(input);
  if (component.registry != c.routes.registry)
    throw std::runtime_error("Component registry differs from this directory's densities");
  checkModelMasses(endpoint, *c.engine);
  Branch branch;
  const std::vector<Candidate> candidates = branchCandidates(c.routes, input, hook, endpoint, branch);
  std::vector<Density> densities;
  for (const auto& candidate : candidates) {
    int config = component.config, graph = component.graph, flow = component.flow;
    std::vector<int> swapped;
    for (int k = 0; k < int(candidate.bornPermutation.size()); ++k)
      if (candidate.bornPermutation[k] != k) swapped.push_back(k);
    if (!swapped.empty()) {
      if (swapped.size() != 2)
        throw std::runtime_error("Born transport must exchange one pair of identical legs");
      const auto image = c.routes.transport.find({swapped[0], swapped[1], config, graph, flow});
      if (image == c.routes.transport.end())
        throw std::runtime_error("Born diagram/flow permutation has no unique certified image");
      config = image->second[0]; graph = image->second[1]; flow = image->second[2];
    }
    std::vector<LHEParticle> legs, born;
    for (int k : candidate.permutation) legs.push_back(endpoint[k]);
    for (int k : candidate.bornPermutation) born.push_back(input.hardParticles[k]);
    densities.push_back(evaluate(legs, candidate.sector, component, config, graph, flow, &born));
    sAcceptance(densities.back(), 1.);
  }
  // Equivalent daughter orientations share a signed luminosity. Mark them
  // with their physical kernels; the common sign cancels in the marking.
  bool positive = false, negative = false;
  for (const auto& density : densities) {
    x.totalKernel += density.k;
    positive = positive || density.k > 0.;
    negative = negative || density.k < 0.;
  }
  if (positive && negative)
    throw std::runtime_error("Equivalent shower orientations have incompatible PDF signs");
  const double sign = negative ? -1. : 1.;
  int selected = 0;
  if (candidates.size() > 1 && sign * x.totalKernel > 0.) {
    const double threshold = uniform() * sign * x.totalKernel;
    double cumulative = 0.;
    for (int k = 0; k < int(densities.size()); ++k)
      if (sign * densities[k].k > 0.) selected = k;
    for (int k = 0; k < int(densities.size()); ++k) {
      cumulative += sign * densities[k].k;
      if (threshold < cumulative) { selected = k; break; }
    }
    x.marking = sign * densities[selected].k / (sign * x.totalKernel);
  }
  x.evaluated = true;
  x.candidates = candidates.size();
  x.sector = candidates[selected].sector;
  x.orientation = candidates[selected].orientation;
  x.density = densities[selected];
  if (x.density.transfers()) {
    // K_sh: K with the shower's emission coupling and PDF ratio; the S
    // events carry beta = sigma_S/sigma_B times the Born density.
    x.alphaShower = hook.emissionAlphaS;
    if (!(x.alphaShower > 0.) || !(x.density.alphaK > 0.) || !std::isfinite(x.alphaShower) ||
        !std::isfinite(x.density.alphaK))
      throw std::runtime_error("The S veto needs positive shower and kernel emission couplings");
    x.pdfRatio = pdfRatio(c, hook, input, endpoint, branch, x.density);
    x.kernelScale = c.normalisation.beta * x.alphaShower / x.density.alphaK * x.pdfRatio;
  }
  x.keep = sAcceptance(x.density, x.kernelScale);
  x.rule = x.keep < 1. ? "s_veto" : "no_transfer";
  return x;
}

Decision decideH(const Context& c, const InputEvent& input) {
  Decision x;
  if (!(input.weight < 0.)) { x.rule = "positive_h"; return x; }
  const ComponentPacket component = componentPacket(input);
  if (component.registry != c.routes.registry)
    throw std::runtime_error("Component registry differs from this directory's densities");
  if (component.sourceFactor == 0.)
    throw std::runtime_error("Negative H event has a zero source multiplier");
  std::vector<int> ids;
  for (const auto& p : input.hardParticles) ids.push_back(p.id);
  if (!c.routes.sector(component.sector).rows.count(ids))
    throw std::runtime_error("H component registry/sector/ordered flavours disagree");
  checkModelMasses(input.hardParticles, *c.engine);
  x.density = evaluate(input.hardParticles, component.sector, component, component.config,
                       component.graph, component.flow, nullptr);
  x.evaluated = true;
  x.candidates = 1;
  x.sector = component.sector;
  x.orientation = "source";
  x.keep = hAcceptance(x.density);
  x.rule = "h_removal";
  return x;
}

std::string decisionTag(const Decision& x, bool accepted) {
  std::ostringstream out;
  out << std::setprecision(17) << "<macnlops_decision version=\"1\" rule=\"" << x.rule
      << "\" keep=\"" << x.keep << "\" accepted=\"" << (accepted ? 1 : 0) << '"';
  if (x.rule != "exhausted" && x.rule != "positive_h" && x.pt > 0.)
    out << " pt=\"" << x.pt << "\" matching_scale=\"" << x.matchingScale << '"';
  if (x.evaluated) {
    out << " r=\"" << x.density.r << "\" d=\"" << x.density.d << "\" k=\"" << x.density.k
        << "\" sector=\"" << x.sector << "\" orientation=\"" << x.orientation
        << "\" candidates=\"" << x.candidates << "\" marking=\"" << x.marking << '"';
    if (x.candidates > 1) out << " total_k=\"" << x.totalKernel << '"';
    if (x.alphaShower > 0.)
      out << " kernel_scale=\"" << x.kernelScale << "\" alpha_s_shower=\"" << x.alphaShower
          << "\" alpha_s_kernel=\"" << x.density.alphaK << "\" pdf_ratio=\"" << x.pdfRatio << '"';
  }
  out << "/>";
  return out.str();
}

const char* weightGroup =
  "    <weightgroup name='macnlops' combine='none'>\n"
  "      <weight id='macnlops'> macnlops </weight>\n"
  "    </weightgroup>\n";

// Declare the 'macnlops' weight and mark the sample in the pre-<header>
// region, where collect_events picks up these blocks from its first input.
std::string matchedPreamble(std::string text, bool dropRejected, bool pdfCorrection,
                            const double cutoffs[2]) {
  const std::size_t open = text.find("<LesHouchesEvents");
  const std::size_t header = text.find("<header");
  if (open == std::string::npos || header == std::string::npos || header < open)
    throw std::runtime_error("Input lacks the LesHouchesEvents/header preamble");
  if (text.find("<macnlops_first_emission") != std::string::npos)
    throw std::runtime_error("Input is already a MAcNLOPS first-emission sample");
  std::ostringstream marker;
  marker << std::setprecision(17) << "  <macnlops_first_emission version=\"1\" prescription=\""
         << (pdfCorrection ? prescriptionV3 : prescriptionV2) << "\" weight=\"macnlops\" rejected=\""
         << (dropRejected ? "dropped" : "kept") << "\" shower_alpha_s=\"" << tableAlphaS
         << "\" shower_alpha_s_order=\"" << tableAlphaSOrder << "\" pythia=\"8.318\" cutoff_isr=\""
         << cutoffs[0] << "\" cutoff_fsr=\"" << cutoffs[1] << "\"/>\n";
  std::size_t close = text.find("</initrwgt>");
  std::string blocks = marker.str();
  if (close != std::string::npos && close < header) {
    text.insert(text.rfind('\n', close) + 1, weightGroup);
  } else {
    blocks += std::string("  <initrwgt>\n") + weightGroup + "  </initrwgt>\n";
  }
  // Inside MG's leading comment when there is one, like its own blocks.
  std::size_t at = text.find('\n', open) + 1;
  const std::size_t comment = text.find("<!--", open);
  if (comment != std::string::npos && comment < header) at = text.find('\n', comment) + 1;
  text.insert(at, blocks);
  return text;
}

// The source XML stays provenance; the 'macnlops' weight follows the
// #aMCatNLO line (or joins an existing <rwgt>) for MG's LHEF weight reader.
std::string matchedExtras(const InputEvent& input, const std::string& weight) {
  std::string extra = inertSourceScales(input.extra);
  const std::string line = "   <wgt id='macnlops'> " + weight + " </wgt>\n";
  const std::size_t close = extra.find("</rwgt>");
  if (close != std::string::npos) {
    const std::size_t start = extra.rfind('\n', close);
    const bool own = start != std::string::npos &&
      trim(extra.substr(start + 1, close - start - 1)).empty();
    extra.insert(own ? start + 1 : close, own ? line : "\n" + line);
    return extra;
  }
  std::size_t at = 0;
  const std::size_t tag = extra.find("#aMCatNLO");
  if (tag != std::string::npos) at = extra.find('\n', tag) + 1;
  extra.insert(at, "  <rwgt>\n" + line + "  </rwgt>\n");
  return extra;
}

void writeMatchedEvent(std::ostream& out, const InputEvent& input,
    const std::vector<LHEParticle>& particles, const FirstEmission& hook,
    const std::string& weight, const std::string& decision) {
  const auto h = fields(input.header, 6, "LHE event header");
  const auto continuation = continuationState(input, hook, particles);
  out << input.open << '\n'
      << particles.size() << ' ' << h[1] << ' ' << h[2] << ' ' << continuation.scale
      << ' ' << h[4] << ' ' << h[5] << '\n';
  for (const LHEParticle& p : particles)
    out << p.id << ' ' << p.status << ' ' << p.mother1 << ' ' << p.mother2
        << ' ' << p.col << ' ' << p.acol << ' '
        << p.px << ' ' << p.py << ' ' << p.pz << ' ' << p.e << ' ' << p.m
        << ' ' << p.lifetime << ' ' << p.spin << '\n';
  out << matchedExtras(input, weight);
  if (input.hasHard)
    out << "<macnlops_endpoint version=\"1\" origin=\""
        << (hook.seen ? "shower" : "hard_annotation") << "\" stage=\"first\"/>\n";
  out << "<macnlops_continuation version=\"1\" state=\"" << continuation.state
      << "\" scale=\"" << continuation.scale << "\" source_scalup=\"" << input.scale
      << "\" cutoff_isr=\"" << hook.cutoffISR << "\" cutoff_fsr=\"" << hook.cutoffFSR << "\"/>\n";
  out << "<scales";
  for (const auto& entry : continuation.legScales)
    out << " scalup_" << entry.first << "=\"" << entry.second << '"';
  for (const auto& entry : continuation.dipoleScales)
    out << ' ' << dipoleName(entry.first.first, entry.first.second) << "=\"" << entry.second << '"';
  out << ">\n</scales>\n" << decision << '\n' << input.close << '\n';
}

std::string jsonString(const std::string& text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char ch : text) {
    if (ch == '"' || ch == '\\') out << '\\' << ch;
    else if (ch < 32) out << "\\u00" << "0123456789abcdef"[ch >> 4] << "0123456789abcdef"[ch & 15];
    else out << ch;
  }
  out << '"';
  return out.str();
}

struct ProcessTotals {
  double input = 0., kept = 0., keptSquares = 0., keptMax = 0.;
  long events = 0;
};

bool sameValue(double a, double b) {
  return std::abs(a - b) <= 1.e-12 * std::max(1., std::max(std::abs(a), std::abs(b)));
}

} // namespace

int main(int argc, char** argv) {
  MatchOptions m;
  std::map<std::string, long> counts;
  std::map<int, ProcessTotals> processes;
  std::map<std::string, int> warnings;
  double inputSum = 0., keptSum = 0.;
  Context c;
  auto writeSummary = [&](const std::string& status, const std::string& error) {
    if (m.summary.empty()) return;
    std::ofstream out(m.summary.c_str());
    out << std::setprecision(17) << "{\"schema\":\"macnlops-first-emission-channel-v1\""
        << ",\"status\":" << jsonString(status) << ",\"error\":" << jsonString(error)
        << ",\"input\":" << jsonString(m.input) << ",\"output\":" << jsonString(m.output)
        << ",\"seed\":" << m.seed << ",\"prescription\":"
        << jsonString(m.pdfCorrection ? prescriptionV3 : prescriptionV2)
        << ",\"rejected_events\":" << jsonString(m.dropRejected ? "dropped" : "kept")
        << ",\"shower\":{\"alpha_s_mz\":" << tableAlphaS << ",\"alpha_s_order\":" << tableAlphaSOrder
        << ",\"pdf\":" << jsonString(m.pdf) << ",\"cutoffs\":[" << c.cutoffs[0] << ',' << c.cutoffs[1]
        << "]},\"channel\":{\"res\":" << jsonString(m.res) << ",\"sigma_S\":" << c.normalisation.sigmaS
        << ",\"sigma_S_error\":" << c.normalisation.errorS << ",\"sigma_B\":" << c.normalisation.sigmaB
        << ",\"sigma_B_error\":" << c.normalisation.errorB << ",\"beta\":" << c.normalisation.beta
        << "},\"counts\":{";
    bool comma = false;
    for (const auto& entry : counts) {
      out << (comma ? "," : "") << jsonString(entry.first) << ':' << entry.second;
      comma = true;
    }
    out << "},\"weight_sums\":{\"input\":" << inputSum << ",\"macnlops\":" << keptSum
        << "},\"processes\":{";
    comma = false;
    for (const auto& entry : processes) {
      out << (comma ? "," : "") << '"' << entry.first << "\":{\"events\":" << entry.second.events
          << ",\"input\":" << entry.second.input << ",\"macnlops\":" << entry.second.kept
          << ",\"macnlops_squares\":" << entry.second.keptSquares
          << ",\"macnlops_max\":" << entry.second.keptMax << '}';
      comma = true;
    }
    out << "},\"native_trial_warnings\":{";
    comma = false;
    for (const auto& entry : warnings) {
      out << (comma ? "," : "") << jsonString(entry.first) << ':' << entry.second;
      comma = true;
    }
    out << "}}\n";
  };
  try {
    m = matchOptions(argc, argv);
    if (m.seed < 1 || m.seed > 900000000)
      throw std::runtime_error("Seed must be in [1,900000000]");
    int nlegs = 0;
    char registry[64];
    macnlops_c_setup(&nlegs, registry);
    c.routes.read(m.routes);
    if (c.routes.registry != std::string(registry, 64) || c.routes.nexternal != nlegs)
      throw std::runtime_error("Route table does not describe this directory's densities");
    double profileCutoffs[2], profileXmax[2], profileFloors[2];
    int profileFlavours[2];
    macnlops_c_profile(profileCutoffs, profileXmax, profileFloors, profileFlavours);
    c.normalisation = channelNormalisation(m.res);
    c.pdfCorrection = m.pdfCorrection;

    Options o;
    o.input = m.input; o.output = m.output; o.settings = m.settings; o.pdf = m.pdf;
    o.settingsOutput = m.output + ".settings.cmnd";
    o.seed = m.seed; o.alphaSMZ = tableAlphaS; o.alphaSorder = tableAlphaSOrder;
    o.flavourCount = profileFlavours[0]; o.mcatnloDelta = true;
    if (o.input.size() >= 3 && o.input.substr(o.input.size() - 3) == ".gz")
      throw std::runtime_error("Decompress the LHE input before this executable");
    if (o.flavourCount != 4 && o.flavourCount != 5)
      throw std::runtime_error("Invalid compiled flavour count");
    Pythia pythia;
    configure(pythia, o);
    RawReader raw(o.input);
    std::shared_ptr<CountingReader> reader = std::make_shared<CountingReader>(
      const_cast<Info*>(&pythia.info), o.input, pythia, true);
    std::shared_ptr<FirstEmission> hook = std::make_shared<FirstEmission>();
    configureHook(pythia, o, *hook);
    pythia.setLHAupPtr(reader);
    pythia.setUserHooksPtr(hook);
    if (!pythia.init()) throw std::runtime_error("Pythia initialization failed");
    validateQCDOnlySettings(pythia);
    validateSourceMatchingSettings(pythia, o);
    hook->captureBeamLimits();
    if (!pythia.settings.writeFile(o.settingsOutput, true))
      throw std::runtime_error("Cannot write Pythia settings snapshot");
    // The generation profile fixed the counterterm support; the shower
    // must reproduce it before any event is matched.
    c.cutoffs[0] = hook->cutoffISR; c.cutoffs[1] = hook->cutoffFSR;
    c.floors[0] = std::max(1.2, pythia.particleData.m0(4));
    c.floors[1] = std::max(4., pythia.particleData.m0(5));
    if (!sameValue(c.cutoffs[0], profileCutoffs[0]) || !sameValue(c.cutoffs[1], profileCutoffs[1]) ||
        !sameValue(hook->beamXMaxA, profileXmax[0]) || !sameValue(hook->beamXMaxB, profileXmax[1]) ||
        !sameValue(c.floors[0], profileFloors[0]) || !sameValue(c.floors[1], profileFloors[1]) ||
        pythia.mode("SpaceShower:nQuarkIn") != profileFlavours[0] ||
        pythia.mode("TimeShower:nGluonToQuark") != profileFlavours[1])
      throw std::runtime_error("The one-step shower differs from the generation profile; "
                               "prepare a fresh macnlops=True generation");
    double xmax[2] = {hook->beamXMaxA, hook->beamXMaxB};
    macnlops_c_configure(xmax, &hook->cutoffISR, &hook->cutoffFSR);
    c.engine = &pythia;
    c.eBeam[0] = reader->eBeamA(); c.eBeam[1] = reader->eBeamB();
    Uniform decisions(0x6d61636e6c6f7073ULL ^ std::uint64_t(m.seed));
    Uniform marking(0x6d61726b696e6721ULL ^ std::uint64_t(m.seed));

    std::ofstream output(o.output.c_str());
    if (!output) throw std::runtime_error("Cannot open output file");
    output << std::scientific << std::setprecision(17);
    InputEvent input;
    bool header = false;
    while (raw.next(input)) {
      output << (header ? input.prefix : matchedPreamble(input.prefix, m.dropRejected, m.pdfCorrection, c.cutoffs));
      header = true;
      const int kind = input.kind;
      if (kind != 1 && kind != 2) throw std::runtime_error("Missing #aMCatNLO S/H event kind");
      if (input.hasHard && kind != input.hardMetadata[1])
        throw std::runtime_error("S/H kind conflicts with hard-event annotation");
      hook->reset();
      hook->source = &input;
      std::vector<LHEParticle> particles = input.particles;
      reader->expect(input);
      const std::size_t before = reader->readCount;
      if (kind == 2) {
        if (!reader->skipOne()) throw std::runtime_error("Cannot consume H event");
      } else {
        const auto warningStart = nativeTrialCounts(pythia);
        const bool success = pythia.next();
        validateNominalWeights(pythia);
        for (const auto& entry : warningDifference(nativeTrialCounts(pythia), warningStart))
          if (entry.second > 0) warnings[entry.first] += entry.second;
        if (!success && !hook->finished)
          throw std::runtime_error("Shower failed before first emission at event " +
                                   std::to_string(counts["input_S"] + counts["input_H"] + 1));
        if (success && hook->finished)
          throw std::runtime_error("First-emission technical stop did not abort");
        if (hook->seen) {
          particles = hardParticles(*hook, input);
          if (particles.size() != input.particles.size() + 1)
            throw std::runtime_error("Shower changed external multiplicity unexpectedly");
        }
      }
      if (reader->readCount != before + 1)
        throw std::runtime_error("Pythia consumed more than one source event");
      validateQCDOnlyEndpoint(particles, input);
      const std::string label = kind == 1 ? "S" : "H";
      const Decision decision = kind == 1 ? decideS(c, marking, input, *hook, particles)
                                          : decideH(c, input);
      const bool accepted = decisions() < decision.keep;
      ++counts["input_" + label];
      ++counts[(accepted ? "accepted_" : "rejected_") + label];
      ++counts[decision.rule + "_" + label];
      if (decision.evaluated) ++counts["evaluated_" + label];
      inputSum += input.weight;
      ProcessTotals& totals = processes[input.process];
      ++totals.events;
      totals.input += input.weight;
      if (accepted) {
        keptSum += input.weight;
        totals.kept += input.weight;
        totals.keptSquares += input.weight * input.weight;
        totals.keptMax = std::max(totals.keptMax, std::abs(input.weight));
      }
      if (accepted || !m.dropRejected) {
        const auto h = fields(input.header, 6, "LHE event header");
        writeMatchedEvent(output, input, particles, *hook, accepted ? h[2] : "0.00000000E+00",
                          decisionTag(decision, accepted));
      }
      if (!output) throw std::runtime_error("Output write failure");
    }
    output << (header ? input.prefix : matchedPreamble(input.prefix, m.dropRejected, m.pdfCorrection, c.cutoffs));
    output.close();
    if (!output) throw std::runtime_error("Output write failure");
    writeSummary("complete", "");
    std::cout << "MAcNLOPS first emission: " << counts["input_S"] << " S and " << counts["input_H"]
              << " H events, " << counts["rejected_S"] << " S vetoed, " << counts["rejected_H"]
              << " H removed; beta = " << c.normalisation.beta << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "MAcNLOPS: " << error.what() << '\n';
    writeSummary("failed", error.what());
    return 1;
  }
}
