// One-step MAcNLOPS proposals. The final physics veto is performed separately.
// Build with: make MacNLOPS PYTHIA8LOCATION=/path/to/pythia8
#include "Pythia8/Pythia.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Pythia8;

namespace {

std::string trim(const std::string& s) {
  const std::size_t first = s.find_first_not_of(" \t\r\n");
  return first == std::string::npos ? "" :
    s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

bool eventStart(const std::string& line) {
  const std::string s = trim(line);
  return s.compare(0, 6, "<event") == 0 && s.size() > 6 &&
    (s[6] == '>' || s[6] == ' ' || s[6] == '\t');
}

double number(std::string s) {
  std::replace(s.begin(), s.end(), 'D', 'E');
  std::replace(s.begin(), s.end(), 'd', 'e');
  std::size_t used = 0;
  const double x = std::stod(s, &used);
  if (used != s.size() || !std::isfinite(x))
    throw std::runtime_error("Invalid LHE number: " + s);
  return x;
}

struct LHEParticle {
  int id, status, mother1, mother2, col, acol;
  double px, py, pz, e, m, lifetime, spin;
};

std::vector<std::string> fields(const std::string& line, std::size_t count,
                                const std::string& label) {
  std::istringstream stream(line);
  std::vector<std::string> result;
  std::string word;
  while (stream >> word) {
    if (word[0] == '#') break;
    result.push_back(word);
  }
  if (result.size() != count) throw std::runtime_error("Malformed " + label);
  return result;
}

int integer(const std::string& value) {
  std::size_t used = 0;
  const long n = std::stol(value, &used);
  if (used != value.size() || n < std::numeric_limits<int>::min() ||
      n > std::numeric_limits<int>::max())
    throw std::runtime_error("Invalid LHE integer: " + value);
  return int(n);
}

LHEParticle particleRecord(const std::string& line) {
  const auto f = fields(line, 13, "LHE particle record");
  LHEParticle p;
  p.id = integer(f[0]); p.status = integer(f[1]);
  p.mother1 = integer(f[2]); p.mother2 = integer(f[3]);
  p.col = integer(f[4]); p.acol = integer(f[5]);
  p.px = number(f[6]); p.py = number(f[7]); p.pz = number(f[8]);
  p.e = number(f[9]); p.m = number(f[10]);
  p.lifetime = number(f[11]); p.spin = number(f[12]);
  return p;
}

const std::vector<std::string>& qcdOnlyOff() {
  static const std::vector<std::string> keys = {
    "PartonLevel:MPI", "PartonLevel:Remnants", "PartonLevel:FSRinResonances",
    "PartonLevel:earlyResDec", "ProcessLevel:resonanceDecays",
    "HadronLevel:all", "HadronLevel:Hadronize", "HadronLevel:Decay",
    "HadronLevel:Rescatter", "HadronLevel:QED", "HadronLevel:BoseEinstein",
    "HadronLevel:DeuteronProduction", "ColourReconnection:reconnect",
    "ColourReconnection:forceHadronLevelCR", "ColourReconnection:forceResonance",
    "BeamRemnants:primordialKT", "TimeShower:interleaveResDec",
    "TimeShower:allowMPIdipole", "SpaceShower:QEDshowerByQ",
    "SpaceShower:QEDshowerByL", "TimeShower:QEDshowerByQ",
    "TimeShower:QEDshowerByL", "TimeShower:QEDshowerByOther",
    "TimeShower:QEDshowerByGamma", "SpaceShower:weakShower",
    "TimeShower:weakShower", "RHadrons:allow", "RHadrons:allowDecay",
    "Ropewalk:RopeHadronization", "Ropewalk:doShoving", "Ropewalk:doFlavour"
  };
  return keys;
}

void validateQCDOnlySettings(Pythia& p) {
  for (const auto& key : qcdOnlyOff())
    if (p.settings.flag(key))
      throw std::runtime_error("Non-QCD shower setting enabled: " + key);
  for (const char* key : {"PartonLevel:all", "ProcessLevel:all", "PartonLevel:ISR",
       "PartonLevel:FSR", "PartonLevel:FSRinProcess", "SpaceShower:QCDshower",
       "TimeShower:QCDshower", "TimeShower:interleave"})
    if (!p.settings.flag(key))
      throw std::runtime_error(std::string("QCD shower setting disabled: ") + key);
}

void validateQCDOnlyHistory(const Event& event, int systems) {
  if (systems != 1) throw std::runtime_error("QCD evolution changed the single hard system");
  for (int i = 3; i < event.size(); ++i) {
    const auto& p = event[i];
    const int status = p.statusAbs();
    if (p.isHadron() || (status >= 31 && status <= 39) || status >= 61)
      throw std::runtime_error("Non-QCD/MPI/remnant/hadron history in production shower");
    if (p.id() == 22) throw std::runtime_error("Photon in QCD-only shower history");
  }
}

std::map<std::string, std::string> xmlAttributes(const std::string& line,
    const std::string& name, bool empty) {
  const std::string s = trim(line), prefix = "<" + name;
  if (s.compare(0, prefix.size(), prefix) ||
      s.size() <= prefix.size() ||
      !std::isspace(static_cast<unsigned char>(s[prefix.size()])))
    throw std::runtime_error("Malformed " + name + " tag");
  const std::string ending = empty ? "/>" : ">";
  if (s.size() < ending.size() || s.substr(s.size()-ending.size()) != ending)
    throw std::runtime_error("Malformed " + name + " tag ending");
  const std::size_t end = s.size()-ending.size();
  std::size_t i = prefix.size();
  std::map<std::string, std::string> result;
  while (i < end) {
    while (i < end && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    if (i == end) break;
    const std::size_t begin = i;
    while (i < end && (std::isalnum(static_cast<unsigned char>(s[i])) || s[i]=='_')) ++i;
    const std::string key = s.substr(begin, i-begin);
    while (i < end && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    if (key.empty() || i == end || s[i++] != '=')
      throw std::runtime_error("Malformed " + name + " attribute");
    while (i < end && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    if (i == end || (s[i] != '\"' && s[i] != '\''))
      throw std::runtime_error("Unquoted " + name + " attribute");
    const char quote = s[i++];
    const std::size_t value = i;
    while (i < end && s[i] != quote) ++i;
    if (i == end || !result.emplace(key, s.substr(value, i-value)).second)
      throw std::runtime_error("Invalid or duplicate " + name + " attribute");
    ++i;
    if (i < end && !std::isspace(static_cast<unsigned char>(s[i])))
      throw std::runtime_error("Malformed " + name + " attribute separator");
  }
  return result;
}

struct InputEvent {
  std::string prefix, open, header, extra, close, raw;
  std::vector<LHEParticle> particles;
  std::vector<LHEParticle> originalParticles, hardParticles;
  std::vector<int> hardMetadata;
  bool hasHard = false, hardUsed = false, hasEndpoint = false;
  std::string hardSource, hardPartition, endpointOrigin;
  bool delta = false;
  double showerScale = 0.;
  std::vector<double> productionScales;
  std::map<std::pair<int, int>, double> dipoleScales;
  std::map<std::pair<int, int>, double> nativeDipoleScales;
  std::map<std::string, double> scaleAttributes;
  int process, kind;
  double weight, scale, alphaEM, alphaS;

  void parse() {
    std::istringstream input(raw);
    std::getline(input, open);
    std::getline(input, header);
    const auto h = fields(header, 6, "LHE event header");
    const int n = integer(h[0]);
    if (n < 2 || n > 100000)
      throw std::runtime_error("Malformed LHE event header");
    process = integer(h[1]); weight = number(h[2]); scale = number(h[3]);
    alphaEM = number(h[4]); alphaS = number(h[5]);
    for (int i = 0; i < n; ++i) {
      std::string line;
      if (!std::getline(input, line))
        throw std::runtime_error("Truncated LHE particle record");
      originalParticles.push_back(particleRecord(line));
    }
    kind = 0;
    std::string line;
    while (std::getline(input, line)) {
      if (trim(line) == "</event>") {
        close = line;
        break;
      }
      extra += line + "\n";
      if (line.find("macnlops_hard_event") != std::string::npos) {
        if (hasHard) throw std::runtime_error("Duplicate hard-event annotation");
        const auto attributes = xmlAttributes(line, "macnlops_hard_event", false);
        if (attributes.size() != 4 || attributes.count("version") != 1 ||
            attributes.count("frame") != 1 || attributes.count("source") != 1 ||
            attributes.count("partition") != 1 || attributes.at("version") != "1" ||
            attributes.at("frame") != "lab")
          throw std::runtime_error("Unsupported hard-event annotation schema");
        hardSource = attributes.at("source"); hardPartition = attributes.at("partition");
        if ((hardSource != "ordinary" && hardSource != "resolved" && hardSource != "delta") ||
            (hardPartition != "sampled" && hardPartition != "global_sum"))
          throw std::runtime_error("Invalid hard-event partition metadata");
        if (!std::getline(input, line)) throw std::runtime_error("Truncated hard-event annotation");
        extra += line + "\n";
        for (const auto& value : fields(line, 14, "hard-event metadata"))
          hardMetadata.push_back(integer(value));
        if (hardMetadata[0] < 3 || hardMetadata[0] > 500 ||
            (hardMetadata[1] != 1 && hardMetadata[1] != 2))
          throw std::runtime_error("Invalid hard-event multiplicity or kind");
        for (std::size_t i = 2; i < hardMetadata.size(); ++i)
          if (i != 7 && hardMetadata[i] < 0)
            throw std::runtime_error("Negative hard-event partition field");
        for (int i = 8; i <= 11; ++i)
          if (hardMetadata[i] > 1) throw std::runtime_error("Invalid hard-event partition flag");
        if (hardMetadata[12] < 1 ||
            (hardPartition == "global_sum" &&
             (hardMetadata[8] != 1 || hardMetadata[9] != 0 ||
              hardMetadata[11] != 1 || hardMetadata[13] < 1)))
          throw std::runtime_error("Conflicting hard-event partition metadata");
        for (int i = 0; i < hardMetadata[0]; ++i) {
          if (!std::getline(input, line)) throw std::runtime_error("Truncated hard-event particles");
          extra += line + "\n";
          hardParticles.push_back(particleRecord(line));
        }
        if (!std::getline(input, line) || trim(line) != "</macnlops_hard_event>")
          throw std::runtime_error("Unclosed hard-event annotation");
        extra += line + "\n"; hasHard = true;
        continue;
      }
      if (line.find("macnlops_endpoint") != std::string::npos) {
        if (hasEndpoint) throw std::runtime_error("Duplicate derived-endpoint marker");
        const auto attributes = xmlAttributes(line, "macnlops_endpoint", true);
        if (attributes.size() != 3 || attributes.count("version") != 1 ||
            attributes.count("origin") != 1 || attributes.count("stage") != 1 ||
            attributes.at("version") != "1")
          throw std::runtime_error("Unsupported derived-endpoint schema");
        endpointOrigin = attributes.at("origin");
        if ((endpointOrigin != "shower" && endpointOrigin != "hard_annotation") ||
            attributes.at("stage") != "first")
          throw std::runtime_error("Invalid derived-endpoint marker");
        hasEndpoint = true; continue;
      }
      const std::size_t tag = line.find("#aMCatNLO");
      if (tag != std::string::npos) {
        std::istringstream amc(line.substr(tag + 9));
        int found = 0;
        if (!(amc >> found) || (found != 1 && found != 2) || (kind && kind != found))
          throw std::runtime_error("Invalid #aMCatNLO S/H event kind");
        kind = found;
      }
    }
    if (close.empty()) throw std::runtime_error("Unclosed LHE event");
    if (hasEndpoint && !hasHard) throw std::runtime_error("Endpoint marker without hard-event provenance");
    if (hasHard) {
      if (kind && kind != hardMetadata[1]) throw std::runtime_error("Conflicting hard-event S/H kind");
      kind = hardMetadata[1];
      hardUsed = !hasEndpoint || endpointOrigin == "hard_annotation";
      if (hardUsed) {
        std::vector<LHEParticle> external;
        for (const auto& p : originalParticles) if (p.status == -1 || p.status == 1) external.push_back(p);
        if (external.size() != hardParticles.size())
          throw std::runtime_error("Hard-event external multiplicity conflicts with LHE body");
        for (std::size_t i = 0; i < external.size(); ++i)
          if (external[i].id != hardParticles[i].id || external[i].status != hardParticles[i].status ||
              external[i].col != hardParticles[i].col || external[i].acol != hardParticles[i].acol)
            throw std::runtime_error("Hard-event external identity conflicts with LHE body");
      }
      for (std::size_t i = 0; i < hardParticles.size(); ++i) {
        const auto& p = hardParticles[i];
        if (!p.id || p.id == std::numeric_limits<int>::min() || p.e <= 0. || p.m < 0. ||
            p.col < 0 || p.acol < 0 ||
            (i < 2 ? p.status != -1 : p.status != 1) ||
            p.mother1 < 0 || p.mother2 < 0 ||
            p.mother1 > int(hardParticles.size()) || p.mother2 > int(hardParticles.size()))
          throw std::runtime_error("Invalid hard-event external status or ancestry");
      }
    }
    particles = hardUsed ? hardParticles : originalParticles;
    showerScale = scale;
  }
};

// These packets remain byte-preserved source provenance in the written LHE.
// Only the native import receives reindexed scale attributes.
struct SourceXMLBlock {
  bool present = false;
  std::map<std::string, std::string> attributes;
  std::string contents;
};

SourceXMLBlock sourceXMLBlock(const std::string& text, const std::string& name) {
  SourceXMLBlock result;
  const std::string prefix = "<" + name, close = "</" + name + ">";
  const std::size_t start = text.find(prefix);
  if (start == std::string::npos) {
    if (text.find(close) != std::string::npos)
      throw std::runtime_error("Unopened " + name + " packet");
    return result;
  }
  if (text.find(prefix, start + prefix.size()) != std::string::npos)
    throw std::runtime_error("Duplicate " + name + " packet");
  const std::size_t end = text.find('>', start + prefix.size());
  if (end == std::string::npos) throw std::runtime_error("Truncated " + name + " packet");
  const std::string opening = text.substr(start, end-start+1);
  const bool empty = opening.size() >= 2 && opening[opening.size()-2] == '/';
  result.attributes = xmlAttributes(opening, name, empty);
  result.present = true;
  if (empty) {
    if (text.find(close) != std::string::npos)
      throw std::runtime_error("Conflicting " + name + " packet ending");
  } else {
    const std::size_t finish = text.find(close, end+1);
    if (finish == std::string::npos ||
        text.find(close, finish+close.size()) != std::string::npos)
      throw std::runtime_error("Unclosed or duplicate " + name + " packet ending");
    result.contents = text.substr(end+1, finish-end-1);
  }
  return result;
}

bool colourConnected(const LHEParticle& a, const LHEParticle& b) {
  return (a.col && (a.col == b.col || a.col == b.acol)) ||
         (a.acol && (a.acol == b.col || a.acol == b.acol));
}

std::string dipoleName(int radiator, int recoiler) {
  return "scalup_" + std::to_string(radiator) + "_" + std::to_string(recoiler);
}

void prepareSourceMatching(InputEvent& input, bool delta) {
  input.delta = delta;
  // MC@NLO-Delta H records use -1 as the unused scalar sentinel; their
  // directed scale matrix below supplies the actual native shower start.
  // Keep the sentinel as source provenance, not as a continuation scale.
  if (input.scale < 0. && !(delta && input.kind == 2 && input.scale == -1.))
    throw std::runtime_error("Negative LHE shower scale");
  if (input.hasEndpoint || sourceXMLBlock(input.extra, "macnlops_continuation").present)
    throw std::runtime_error("Derived endpoint requires shower continuation; do not repeat first-emission matching");
  if (input.hasHard && ((input.hardSource == "delta") != delta))
    throw std::runtime_error("Source matching mode conflicts with hard-event annotation");
  const auto scales = sourceXMLBlock(input.extra, "scales");
  const auto context = sourceXMLBlock(input.extra, "macnlops_delta_context");
  const auto component = sourceXMLBlock(input.extra, "macnlops_component");
  if (component.present) {
    const auto& a = component.attributes;
    if (!input.hasHard || input.hardPartition != "sampled" ||
        input.hardSource == "resolved" ||
        input.hardMetadata[8] != 0 || input.hardMetadata[9] != 1 ||
        a.size() != 2 || !a.count("version") || a.at("version") != "1" ||
        !a.count("registry") || a.at("registry").size() != 64 ||
        a.at("registry").find_first_not_of("0123456789abcdef") != std::string::npos)
      throw std::runtime_error("Invalid sampled-component source metadata");
    std::istringstream rows(component.contents);
    std::vector<std::string> data;
    std::string line;
    while (std::getline(rows, line)) if (!trim(line).empty()) data.push_back(line);
    if (data.size() != 2)
      throw std::runtime_error("Invalid sampled-component row count");
    const auto labels = fields(data[0], 4, "sampled-component labels");
    const auto weights = fields(data[1], 4, "sampled-component weights");
    for (const auto& value : labels)
      if (integer(value) < 1) throw std::runtime_error("Invalid sampled-component label");
    const double q = number(weights[0]), insertion = number(weights[1]);
    const double sourceFactor = number(weights[2]), symmetry = number(weights[3]);
    if (integer(labels[0]) != input.hardMetadata[2] ||
        integer(labels[3]) != std::abs(input.hardMetadata[7]) ||
        input.hardMetadata[11] != 1 || input.hardMetadata[13] < 1 ||
        (input.kind == 2 && input.hardMetadata[3] != integer(labels[0])) ||
        q <= 0. || q > 1. || insertion < 0. || insertion > 1. ||
        sourceFactor < 0. || sourceFactor > 1. || symmetry <= 0. ||
        (input.kind == 1 && (insertion != 0. || sourceFactor != 1.)))
      throw std::runtime_error("Invalid sampled-component source labels or weights");
  }
  bool hasDeltaScales = false;
  for (const auto& entry : scales.attributes)
    hasDeltaScales = hasDeltaScales || entry.first.compare(0, 7, "scalup_") == 0;
  if (!delta) {
    if (hasDeltaScales || context.present)
      throw std::runtime_error("Delta source packets require --mcatnlo-delta");
    return;
  }
  if (!scales.present || !trim(scales.contents).empty())
    throw std::runtime_error("Delta source requires an empty per-colour scales packet");
  std::vector<int> oldToNew(input.originalParticles.size()+1, 0);
  int next = 0;
  for (std::size_t i = 0; i < input.originalParticles.size(); ++i) {
    const auto& p = input.originalParticles[i];
    if (p.status == -1 || p.status == 1) oldToNew[i+1] = input.hardUsed ? ++next : int(i+1);
  }
  for (const auto& entry : scales.attributes) {
    const std::string& key = entry.first;
    const double value = number(entry.second);
    if (key == "muf" || key == "mur" || key == "mups") continue;
    if (key.compare(0, 7, "scalup_") != 0)
      throw std::runtime_error("Unsupported Delta starting-scale attribute: " + key);
    const std::size_t separator = key.find('_', 7);
    if (separator == std::string::npos)
      throw std::runtime_error("Malformed Delta dipole-scale index");
    const int i = integer(key.substr(7, separator-7)), j = integer(key.substr(separator+1));
    if (i < 1 || j < 1 || i == j || i >= int(oldToNew.size()) ||
        j >= int(oldToNew.size()) || !oldToNew[i] || !oldToNew[j] || value < 0. ||
        key != dipoleName(i, j) ||
        !colourConnected(input.originalParticles[i-1], input.originalParticles[j-1]))
      throw std::runtime_error("Invalid Delta dipole-scale entry: " + key);
    input.scaleAttributes[key] = value;
    input.dipoleScales[std::make_pair(oldToNew[i], oldToNew[j])] = value;
  }
  for (std::size_t i = 0; i < input.originalParticles.size(); ++i)
    for (std::size_t j = 0; j < input.originalParticles.size(); ++j)
      if (i != j && oldToNew[i+1] && oldToNew[j+1] &&
          colourConnected(input.originalParticles[i], input.originalParticles[j]) &&
          !input.scaleAttributes.count(dipoleName(i+1, j+1)))
        throw std::runtime_error("Incomplete Delta dipole-scale matrix");
  if (input.dipoleScales.empty())
    throw std::runtime_error("Delta source has no coloured dipole starting scales");
  if (input.kind == 2 && !context.present && !component.present)
    throw std::runtime_error("Delta H source requires density-context provenance");
  if (!context.present) return;
  const auto& a = context.attributes;
  if (a.size() != 4 || !a.count("version") || !a.count("owner") ||
      !a.count("insertion") || a.at("version") != "1" || integer(a.at("owner")) < 1 ||
      number(a.at("insertion")) < 0. || number(a.at("insertion")) > 1. ||
      !a.count("registry"))
    throw std::runtime_error("Invalid Delta density-context metadata");
  if (input.hasHard && integer(a.at("owner")) != input.hardMetadata[2])
    throw std::runtime_error("Conflicting Delta hard/context owner");
  if (a.count("registry")) {
    const auto& registry = a.at("registry");
    if (registry.size() != 64 || registry.find_first_not_of("0123456789abcdef") != std::string::npos)
      throw std::runtime_error("Invalid Delta context registry identity");
  }
  std::set<std::vector<int> > seen;
  std::istringstream rows(context.contents);
  std::string line;
  while (std::getline(rows, line)) {
    if (trim(line).empty()) continue;
    const auto f = fields(line, 4, "Delta density-context row");
    const std::vector<int> key = {integer(f[0]), integer(f[1]), integer(f[2])};
    const double probability = number(f[3]);
    if (*std::min_element(key.begin(), key.end()) < 1 || probability < 0. ||
        probability > 1. || !seen.insert(key).second)
      throw std::runtime_error("Invalid or duplicate Delta density-context row");
  }
}

// Match ProcessContainer's LHA ancestry traversal (native indices are LHA
// indices plus the two inserted beams). No momentum/identity guessing is
// used to route colour-dependent scales, including identical hard legs.
std::vector<int> nativeProcessOrder(const std::vector<LHEParticle>& particles) {
  std::vector<int> order(1, 0);
  std::vector<bool> seen(particles.size()+1, false);
  for (std::size_t parent = 0; parent < order.size(); ++parent)
    for (std::size_t i = 0; i < particles.size(); ++i)
      if (particles[i].mother1 == order[parent]) {
        if (seen[i+1]) throw std::runtime_error("Ambiguous Delta source ancestry");
        seen[i+1] = true; order.push_back(i+1);
      }
  if (order.size() != particles.size()+1)
    throw std::runtime_error("Unreachable Delta source ancestry");
  if (order.size() < 3 || order[1] != 1 || order[2] != 2)
    throw std::runtime_error("Delta import requires the two incoming legs first");
  return order;
}

void validateQCDOnlyEndpoint(const std::vector<LHEParticle>& endpoint,
                             const InputEvent& input) {
  std::map<int, int> originalEW, finalEW;
  for (const auto& p : input.particles)
    if (p.status == 1 && std::abs(p.id) > 6 && p.id != 21) ++originalEW[p.id];
  for (const auto& p : endpoint) {
    if (p.id == 22 || std::abs(p.id) > 100)
      throw std::runtime_error("Photon or hadron in QCD-only endpoint");
    if (p.status == 1 && std::abs(p.id) > 6 && p.id != 21) ++finalEW[p.id];
  }
  if (originalEW != finalEW)
    throw std::runtime_error("QCD shower changed the source electroweak particle content");
}

void validateHardInput(const InputEvent& input, const Pythia& engine) {
  if (input.hasHard && !input.hardUsed) {
    InputEvent provenance = input;
    provenance.particles = input.hardParticles;
    provenance.hardUsed = true; provenance.hasEndpoint = false;
    validateHardInput(provenance, engine);
  }
  double eventScale = 1.;
  for (const auto& p : input.particles)
    eventScale = std::max(eventScale, std::max(std::abs(p.e),
      std::max(std::abs(p.px), std::max(std::abs(p.py), std::abs(p.pz)))));
  // Pre-shuffle FKS coordinates can have chained boosts before annotation.
  // Allow their accumulated roundoff at the event scale; keep the tighter
  // plain-LHE/endpoint check. This is far below a physical c/b mass shift.
  const long double shellBudget = input.hasHard && input.hardUsed ? 16384.L : 4096.L;
  const long double tolerance = shellBudget * std::numeric_limits<double>::epsilon() *
    eventScale * eventScale;
  long double balance[4] = {0., 0., 0., 0.};
  int incoming = 0, outgoing = 0;
  for (std::size_t i = 0; i < input.particles.size(); ++i) {
    const auto& p = input.particles[i];
    if (p.e <= 0. || p.m < 0. || p.id == 0 || p.id == std::numeric_limits<int>::min() ||
        (p.status != -1 && p.status != 1 && p.status != 2))
      throw std::runtime_error("Invalid hard-model particle record");
    const long double mass2 = (long double)p.e*p.e - (long double)p.px*p.px -
      (long double)p.py*p.py - (long double)p.pz*p.pz;
    if (std::abs(mass2 - (long double)p.m*p.m) > tolerance)
      throw std::runtime_error("Hard-model momentum is off its declared mass shell");
    const int pdg = std::abs(p.id);
    if ((!input.hasEndpoint || input.hardUsed) &&
        (pdg <= 6 || pdg == 21 || (pdg >= 11 && pdg <= 16)) &&
        std::abs(p.m - engine.particleData.m0(p.id)) >
          1.e-9 * std::max(1., std::abs(engine.particleData.m0(p.id))))
      throw std::runtime_error("Noncanonical hard-model mass for PDG " +
        std::to_string(p.id) + "; supply a valid hard annotation and model settings");
    if (p.status == 2) continue;
    if (p.status == -1) {
      if (i >= 2) throw std::runtime_error("Hard incoming particles must be the first two records");
      ++incoming;
    } else ++outgoing;
    const int sign = p.status == -1 ? -1 : 1;
    balance[0] += sign*p.e; balance[1] += sign*p.px;
    balance[2] += sign*p.py; balance[3] += sign*p.pz;
  }
  if (incoming != 2 || !outgoing)
    throw std::runtime_error("Invalid hard-model incoming/final multiplicity");
  // An FKS hard annotation precedes the shower-mass reshuffle and its
  // momentum repair. Rare generated points retain a few 1e-8 relative
  // balance residuals even though every external leg is on shell. Keep
  // the strict plain-LHE check and admit that numerical source precision
  // for the annotated hard coordinates without changing their momenta.
  const long double balanceTolerance =
    (input.hasHard && input.hardUsed ? 1.e-7L : 1.e-9L) * eventScale;
  for (const auto& value : balance)
    if (std::abs(value) > balanceTolerance)
      throw std::runtime_error("Hard-model event does not conserve four-momentum");
}

// A second reader preserves the complete original event annotations. Keep its
// position synchronized with the Pythia LHA reader, including H pass-throughs.
class RawReader {
public:
  explicit RawReader(const std::string& file) : input(file.c_str()) {
    if (!input) throw std::runtime_error("Cannot open input: " + file);
  }

  bool next(InputEvent& event) {
    event = InputEvent();
    std::string line;
    while (std::getline(input, line)) {
      if (eventStart(line)) {
        event.raw = line + "\n";
        while (std::getline(input, line)) {
          event.raw += line + "\n";
          if (trim(line) == "</event>") {
            event.parse();
            return true;
          }
        }
        throw std::runtime_error("Unclosed input LHE event");
      }
      event.prefix += line + "\n";
    }
    return false;
  }
private:
  std::ifstream input;
};

class CountingReader : public LHAupLHEF {
public:
  CountingReader(Info* info, const std::string& file, const Pythia& pythia,
                 bool delta = false) :
    LHAupLHEF(info, file.c_str(), nullptr, true, delta), readCount(0),
    engine(pythia), delta(delta), expected(nullptr) {}
  void expect(InputEvent& input) {
    if (expected) throw std::runtime_error("Unconsumed hard source event");
    prepareSourceMatching(input, delta);
    validateHardInput(input, engine);
    expected = &input;
  }
  bool setEvent(int id = 0) override {
    if (!expected) throw std::runtime_error("Pythia requested an unqueued source event");
    const bool ok = LHAupLHEF::setEvent(id);
    if (!ok) return false;
    InputEvent& input = *expected;
    if (idProcess() != input.process || weight() != input.weight ||
        (!delta && scale() != input.scale) || alphaQED() != input.alphaEM || alphaQCD() != input.alphaS ||
        sizePart() != int(input.originalParticles.size())+1)
      throw std::runtime_error("Pythia/raw source event mismatch");
    if (delta && (!infoPtr->scales || infoPtr->LHEFversionSave < 2 ||
        infoPtr->scales->SCALUP != input.scale ||
        infoPtr->scales->attributes != input.scaleAttributes))
      throw std::runtime_error("Pythia/raw Delta starting-scale mismatch (LHEF 2/3 required)");
    input.showerScale = scale();
    if (delta && (!std::isfinite(input.showerScale) || input.showerScale <= 0.))
      throw std::runtime_error("Invalid effective Delta shower starting scale");
    for (std::size_t i = 0; i < input.originalParticles.size(); ++i) {
      const auto& p = input.originalParticles[i];
      const auto& q = particles[i+1];
      if (p.id != q.idPart || p.status != q.statusPart ||
          p.mother1 != q.mother1Part || p.mother2 != q.mother2Part ||
          p.col != q.col1Part || p.acol != q.col2Part ||
          p.px != q.pxPart || p.py != q.pyPart || p.pz != q.pzPart ||
          p.e != q.ePart || p.m != q.mPart || p.lifetime != q.tauPart || p.spin != q.spinPart)
        throw std::runtime_error("Pythia/raw source particle mismatch");
      if (!input.hardUsed || p.status == -1 || p.status == 1)
        input.productionScales.push_back(q.scalePart);
    }
    if (input.hardUsed) {
      // Replace only the LHA presentation. The source file, event header,
      // source header, weight and XML provenance are left intact. The native
      // reader's per-particle Delta production scales follow each hard leg.
      particles.clear(); addParticle(0);
      for (std::size_t i = 0; i < input.particles.size(); ++i) {
        const auto& p = input.particles[i];
        addParticle(p.id, p.status, p.mother1, p.mother2, p.col, p.acol,
          p.px, p.py, p.pz, p.e, p.m, p.lifetime, p.spin, input.productionScales[i]);
      }
      setIdX(input.particles[0].id, input.particles[1].id,
        (input.particles[0].e + input.particles[0].pz)/(2.*eBeamASave),
        (input.particles[1].e - input.particles[1].pz)/(2.*eBeamBSave));
    }
    if (delta) {
      const auto order = nativeProcessOrder(input.particles);
      std::vector<int> position(order.size(), 0);
      for (std::size_t i = 1; i < order.size(); ++i) position[order[i]] = i;
      // SimpleTimeShower looks up scalup_(iRad-2)_(iRec-2) after
      // ProcessContainer has ordered the source tree by ancestry.
      infoPtr->scales->attributes.clear();
      for (const auto& entry : input.dipoleScales) {
        const int i = position[entry.first.first], j = position[entry.first.second];
        input.nativeDipoleScales[std::make_pair(i, j)] = entry.second;
        infoPtr->scales->attributes[dipoleName(i, j)] = entry.second;
      }
    }
    expected = nullptr; ++readCount;
    return ok;
  }
  bool skipOne() {
    return setEvent();
  }
  std::size_t readCount;
private:
  const Pythia& engine;
  bool delta;
  InputEvent* expected;
};

// Preserve stock Pythia trial handling. These are the warnings that stock
// Pythia actually logs, not a certificate of every native trial bound.
std::map<std::string, int> nativeTrialCounts(const Pythia& p) {
  std::map<std::string, int> counts;
  for (const auto& message : p.logger)
    if (message.first.find("weight above unity") != std::string::npos ||
        message.first.find("weight is above unity") != std::string::npos ||
        message.first.find("reject denom") != std::string::npos)
      counts[message.first] = message.second;
  return counts;
}

std::map<std::string, int> warningDifference(
    std::map<std::string, int> after, const std::map<std::string, int>& before) {
  for (const auto& previous : before) after[previous.first] -= previous.second;
  return after;
}

// BEGIN MACNLOPS NATIVE SCALE LISTING HELPERS
// Stock 8.318 exposes its active dipoles only through the public list() API.
// Its three-decimal display precision is unsuitable for a shower handoff.
// This scoped facet preserves the actual doubles without changing Pythia.
class ScaleRoundTripNumPut : public std::num_put<char> {
protected:
  iter_type do_put(iter_type out, std::ios_base& stream, char_type,
                   double value) const override {
    std::ostringstream number;
    number.imbue(std::locale::classic());
    number << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    stream.width(0);
    *out++ = ' ';
    for (char c : number.str()) *out++ = c;
    *out++ = ' ';
    return out;
  }
};

class CaptureNativeScaleListing {
public:
  CaptureNativeScaleListing() : buffer(std::cout.rdbuf()), locale(std::cout.getloc()),
    flags(std::cout.flags()), precision(std::cout.precision()), width(std::cout.width()) {
    std::cout.rdbuf(text.rdbuf());
    std::cout.imbue(std::locale(std::locale::classic(), new ScaleRoundTripNumPut));
    std::cout << std::dec << std::noboolalpha;
  }
  ~CaptureNativeScaleListing() {
    std::cout.rdbuf(buffer); std::cout.imbue(locale); std::cout.flags(flags);
    std::cout.precision(precision); std::cout.width(width);
  }
  CaptureNativeScaleListing(const CaptureNativeScaleListing&) = delete;
  CaptureNativeScaleListing& operator=(const CaptureNativeScaleListing&) = delete;
  std::string str() const {return text.str();}
private:
  std::ostringstream text;
  std::streambuf* buffer;
  std::locale locale;
  std::ios::fmtflags flags;
  std::streamsize precision, width;
};

struct NativeDipoleScale {
  int radiator, recoiler, side;
  double scale;
};

std::vector<NativeDipoleScale> parseNativeDipoleScales(const std::string& listing,
                                                     bool isr) {
  const std::string name = isr ? "SimpleSpaceShower" : "SimpleTimeShower";
  const std::string heading = isr ? "i  syst  side   rad   rec       pTmax  col  chg  ME rec" :
    "i    rad    rec       pTmax  col  chg  gam weak  oni   hv  isr  sys sysR type  MErec     mix  ord  spl  ~gR  jun   flex  pol";
  std::vector<NativeDipoleScale> result;
  std::istringstream lines(listing);
  std::string line;
  enum class State { BeforeListing, BeforeRows, InRows, AfterListing };
  State state = State::BeforeListing;
  int nextDipole = 0;
  while (std::getline(lines, line)) {
    const std::string row = trim(line);
    if (row.empty()) continue;
    if (row.find("End PYTHIA " + name + " Dipole Listing") != std::string::npos) {
      if (state != State::InRows)
        throw std::runtime_error("Unexpected native scale listing footer");
      state = State::AfterListing; continue;
    }
    if (row.find("PYTHIA " + name + " Dipole Listing") != std::string::npos) {
      if (state != State::BeforeListing)
        throw std::runtime_error("Duplicate native scale listing");
      state = State::BeforeRows; continue;
    }
    if (row == heading) {
      if (state != State::BeforeRows)
        throw std::runtime_error("Unexpected native scale listing heading");
      state = State::InRows; continue;
    }
    if (state != State::InRows || !std::isdigit(static_cast<unsigned char>(row[0])))
      throw std::runtime_error("Malformed native shower scale listing");
    const auto f = fields(row, isr ? 10 : 22, "native shower scale row");
    for (std::size_t i = 0; i < f.size(); ++i)
      if (i == (isr ? 5u : 3u) || (!isr && (i == 15 || i == 20))) number(f[i]);
      else integer(f[i]);
    if (integer(f[0]) != nextDipole++)
      throw std::runtime_error("Nonconsecutive native dipoles");
    const int radiator = integer(f[isr ? 3 : 1]), recoiler = integer(f[isr ? 4 : 2]);
    const double scale = number(f[isr ? 5 : 3]);
    const int colour = integer(f[isr ? 6 : 4]);
    if (radiator < 1 || recoiler < 1 || radiator == recoiler || scale < 0. ||
        std::abs(colour) > 2 || integer(f[isr ? 7 : 5]) != 0 ||
        integer(f[isr ? 1 : 11]) != 0 ||
        (!isr && (integer(f[6]) || integer(f[7]) || integer(f[8]) || integer(f[9]) ||
                  integer(f[12]) != 0)))
      throw std::runtime_error("Unsupported native QCD dipole state");
    const int side = isr ? integer(f[2]) : 0;
    if (isr && side != 1 && side != 2)
      throw std::runtime_error("Unsupported native ISR dipole side");
    if (colour) result.push_back({radiator, recoiler, side, scale});
  }
  if (state != State::AfterListing)
    throw std::runtime_error("Incomplete native shower scale listing");
  return result;
}
// END MACNLOPS NATIVE SCALE LISTING HELPERS

std::vector<NativeDipoleScale> nativeDipoleScales(Pythia& engine, bool isr) {
  // configure() checks the stock shower version before any source is consumed.
  CaptureNativeScaleListing capture;
  if (isr) engine.getShowerModelPtr()->getSpaceShower()->list();
  else engine.getShowerModelPtr()->getTimeShower()->list();
  return parseNativeDipoleScales(capture.str(), isr);
}

void warningJSON(std::ostream& out, const std::map<std::string, int>& counts) {
  out << '[';
  bool comma = false;
  const char* hex = "0123456789abcdef";
  for (const auto& message : counts) {
    if (message.second <= 0) continue;
    if (comma) out << ',';
    out << "{\"message\":\"";
    for (unsigned char c : message.first) {
      if (c == '"' || c == '\\') out << '\\' << char(c);
      else if (c < 32) out << "\\u00" << hex[c >> 4] << hex[c & 15];
      else out << char(c);
    }
    out << "\",\"count\":" << message.second << '}';
    comma = true;
  }
  out << ']';
}

class FirstEmission : public UserHooks {
public:
  FirstEmission() : seen(false), finished(false), type(0), nISR(0), nFSR(0), pt(0.),
    beamXMaxA(0.), beamXMaxB(0.), cutoffISR(0.), cutoffFSR(0.) {}
  void captureBeamLimits() {
    // Before extracting the sole hard system, xMax(-1) is the exact beam
    // boundary later returned by xMax(0), including the hadron mass reserve.
    beamXMaxA = beamAPtr->xMax(-1);
    beamXMaxB = beamBPtr->xMax(-1);
  }
  bool canVetoStep() override { return true; }
  bool canVetoProcessLevel() override { return true; }
  bool doVetoProcessLevel(Event& process) override {
    if (source && source->delta) {
      const auto order = nativeProcessOrder(source->particles);
      if (process.size() != int(order.size())+2)
        throw std::runtime_error("Delta import changed source multiplicity");
      for (std::size_t i = 1; i < order.size(); ++i) {
        const auto& original = source->particles[order[i]-1];
        const auto& native = process[i+2];
        if (original.id != native.id() || original.col != native.col() ||
            original.acol != native.acol() ||
            (original.status == -1 ? native.statusAbs() != 21 :
             original.status == 2 ? native.statusAbs() != 22 : native.statusAbs() != 23) ||
            (source->productionScales[order[i]-1] >= 0. &&
             source->productionScales[order[i]-1] != native.scale()))
          throw std::runtime_error("Delta import changed particle or production-scale routing");
      }
      if (!infoPtr->scales || infoPtr->scales->attributes.size() != source->nativeDipoleScales.size())
        throw std::runtime_error("Delta import lost dipole starting scales");
      for (const auto& entry : source->nativeDipoleScales) {
        const auto found = infoPtr->scales->attributes.find(dipoleName(entry.first.first, entry.first.second));
        if (found == infoPtr->scales->attributes.end() || found->second != entry.second)
          throw std::runtime_error("Delta import changed a dipole starting scale");
      }
    }
    if (!source || !source->hardUsed) return false;
    // The annotation is already in physical model coordinates. Pythia must
    // import this flat external hard point without another mass reshuffle.
    if (process.size() != int(source->particles.size()) + 3)
      throw std::runtime_error("Annotated hard import changed multiplicity");
    double scale = 1.;
    for (const auto& p : source->particles) scale = std::max(scale, p.e);
    std::vector<bool> used(source->particles.size(), false);
    for (int i = 3; i < process.size(); ++i) {
      const auto& p = process[i];
      bool found = false;
      for (std::size_t j = 0; j < source->particles.size(); ++j) {
        const auto& q = source->particles[j];
        if (used[j] || p.id() != q.id || p.col() != q.col || p.acol() != q.acol ||
            (q.status == -1 ? p.statusAbs() != 21 :
             (p.statusAbs() != 22 && p.statusAbs() != 23))) continue;
        if (std::abs(p.e()-q.e) > 1.e-10*scale ||
            std::abs(p.px()-q.px) > 1.e-10*scale ||
            std::abs(p.py()-q.py) > 1.e-10*scale ||
            std::abs(p.pz()-q.pz) > 1.e-10*scale ||
            std::abs(p.m()-q.m) > 1.e-10*scale) continue;
        used[j] = found = true; break;
      }
      if (!found) throw std::runtime_error("Pythia changed annotated hard-model coordinates");
    }
    return false;
  }
  int numberVetoStep() override { return 1; }
  bool retryPartonLevel() override { return false; }
  bool canVetoPartonLevelEarly() override { return true; }
  bool doVetoPartonLevelEarly(const Event& event) override {
    validateQCDOnlyHistory(event, partonSystemsPtr->sizeSys());
    // The competing showers reached their cutoffs without an emission.
    // Stop before Pythia's complete-event checks, which require beam remnants.
    finished = true;
    return true;
  }

  bool doVetoStep(int pos, int isr, int fsr, const Event& event) override {
    validateQCDOnlyHistory(event, partonSystemsPtr->sizeSys());
    if (seen || isr + fsr != 1 || (pos != 2 && pos != 3))
      throw std::runtime_error("Unexpected first-emission shower hook");
    snapshot = event;
    incoming = {partonSystemsPtr->getInA(0), partonSystemsPtr->getInB(0)};
    outgoing.clear();
    for (int i = 0; i < partonSystemsPtr->sizeOut(0); ++i)
      outgoing.push_back(partonSystemsPtr->getOut(0, i));
    seen = finished = true; type = pos; nISR = isr; nFSR = fsr;
    pt = infoPtr->pTnow();
    // The simple showers' coupling at this emission: pT0Ref is forced to
    // zero and renormMultFac to one, so ISR and FSR evaluate alpha_s at pT2.
    emissionAlphaS = (pos == 2 ? isrAlphaS : fsrAlphaS).alphaS(pt * pt);
    if (!engine) throw std::runtime_error("Missing native shower for continuation export");
    continuationISR = nativeDipoleScales(*engine, true);
    continuationFSR = nativeDipoleScales(*engine, false);
    // This is a technical stop, not the MAcNLOPS event rejection.
    // Check:abortIfVeto makes next() return immediately on this first snapshot.
    return true;
  }

  void reset() {
    seen = finished = false; type = nISR = nFSR = 0; pt = emissionAlphaS = 0.;
    incoming.clear(); outgoing.clear();
    nativeTrialWarnings.clear();
    continuationISR.clear(); continuationFSR.clear();
    source = nullptr;
  }

  Event snapshot;
  bool seen, finished;
  int type, nISR, nFSR;
  double pt, emissionAlphaS = 0.;
  AlphaStrong isrAlphaS, fsrAlphaS;
  double beamXMaxA, beamXMaxB, cutoffISR, cutoffFSR;
  int flavourCount = 5;
  int alphaSNfmax = 5;
  std::map<std::string, int> nativeTrialWarnings;
  std::vector<int> incoming, outgoing;
  std::vector<NativeDipoleScale> continuationISR, continuationFSR;
  Pythia* engine = nullptr;
  const InputEvent* source = nullptr;
};

void particleJSON(std::ostream& out, const LHEParticle& p) {
  out << "{\"id\":" << p.id << ",\"status\":" << p.status
      << ",\"mothers\":[" << p.mother1 << ',' << p.mother2
      << "],\"colors\":[" << p.col << ',' << p.acol
      << "],\"p\":[" << p.e << ',' << p.px << ',' << p.py << ',' << p.pz
      << "],\"mass\":" << p.m << '}';
}

LHEParticle fromPythia(const Pythia8::Particle& p, int status) {
  LHEParticle r;
  r.id = p.id(); r.status = status;
  r.mother1 = status == -1 ? 0 : 1;
  r.mother2 = status == -1 ? 0 : 2;
  r.col = p.col(); r.acol = p.acol();
  r.px = p.px(); r.py = p.py(); r.pz = p.pz();
  r.e = p.e(); r.m = p.m(); r.lifetime = 0.; r.spin = 9.;
  return r;
}

Vec4 momentum(const LHEParticle& p) {
  return Vec4(p.px, p.py, p.pz, p.e);
}

void appendDecay(const InputEvent& input, int sourceParent, int outputParent,
                 std::vector<LHEParticle>& result) {
  const LHEParticle& before = input.particles[sourceParent];
  const LHEParticle after = result[outputParent];
  const Vec4 oldP = momentum(before), newP = momentum(after);
  if (oldP.m2Calc() <= 0. || newP.m2Calc() <= 0. ||
      std::abs(oldP.m2Calc() - newP.m2Calc()) >
        1.e-6 * std::max(1., std::abs(oldP.m2Calc())))
    throw std::runtime_error("Resonance mass changed during production shower");
  // This is exactly the boost used by PartonLevel::resonanceShowers when
  // restoring input decay trees after production radiation, without a decay
  // shower. A generic rest-frame boost pair introduces a different rotation.
  RotBstMatrix transform;
  transform.bst(oldP, newP);
  std::vector<std::pair<int, int> > nested;
  int daughters = 0;
  for (std::size_t i = 0; i < input.particles.size(); ++i) {
    const LHEParticle& source = input.particles[i];
    if (source.mother1 != sourceParent + 1) continue;
    if (source.mother2 != 0 && source.mother2 != sourceParent + 1)
      throw std::runtime_error("Ambiguous resonance-decay mothers");
    if (source.status != 1 && source.status != 2)
      throw std::runtime_error("Invalid resonance-decay daughter status");
    LHEParticle daughter = source;
    Vec4 p = momentum(source);
    p.rotbst(transform);
    daughter.px = p.px(); daughter.py = p.py();
    daughter.pz = p.pz(); daughter.e = p.e();
    daughter.mother1 = daughter.mother2 = outputParent + 1;
    if (daughter.col == before.col) daughter.col = after.col;
    if (daughter.acol == before.acol) daughter.acol = after.acol;
    if (daughter.col == -before.acol) daughter.col = -after.acol;
    if (daughter.acol == -before.col) daughter.acol = -after.col;
    const int position = result.size();
    result.push_back(daughter);
    if (source.status == 2) nested.push_back(std::make_pair(i, position));
    ++daughters;
  }
  if (!daughters) throw std::runtime_error("Source resonance has no daughters");
  for (const auto& decay : nested)
    appendDecay(input, decay.first, decay.second, result);
}

std::vector<LHEParticle> hardParticles(const FirstEmission& hook,
                                     const InputEvent& input) {
  std::map<int, int> resonanceSources;
  std::vector<int> usedOriginals;
  for (std::size_t source = 0; source < input.particles.size(); ++source) {
    const LHEParticle& resonance = input.particles[source];
    if (resonance.status != 2) continue;
    // Nested decays will be restored recursively with their parent's boost.
    if (resonance.mother1 > 0 &&
        input.particles.at(resonance.mother1 - 1).status == 2) continue;
    int original = 0;
    for (int i = 1; i < hook.snapshot.size(); ++i) {
      const Pythia8::Particle& candidate = hook.snapshot[i];
      if (candidate.statusAbs() != 22 || candidate.id() != resonance.id ||
          std::find(usedOriginals.begin(), usedOriginals.end(), i) != usedOriginals.end())
        continue;
      const Vec4 difference = candidate.p() - momentum(resonance);
      const double error = std::abs(difference.e()) + std::abs(difference.px()) +
        std::abs(difference.py()) + std::abs(difference.pz());
      if (error < 1.e-7 * std::max(1., resonance.e)) {original = i; break;}
    }
    if (!original) throw std::runtime_error("Cannot identify source resonance in shower history");
    usedOriginals.push_back(original);
    const int current = hook.snapshot[original].iBotCopyId();
    if (std::find(hook.outgoing.begin(), hook.outgoing.end(), current) == hook.outgoing.end())
      throw std::runtime_error("Shower resonance is missing from production final state");
    resonanceSources[current] = source;
  }
  std::vector<LHEParticle> result;
  for (int i : hook.incoming) {
    if (i <= 0 || i >= hook.snapshot.size())
      throw std::runtime_error("Invalid shower incoming index");
    result.push_back(fromPythia(hook.snapshot[i], -1));
  }
  std::vector<std::pair<int, int> > decays;
  for (int i : hook.outgoing) {
    if (i <= 0 || i >= hook.snapshot.size() || !hook.snapshot[i].isFinal())
      throw std::runtime_error("Unsupported non-final hard shower subsystem");
    const auto source = resonanceSources.find(i);
    result.push_back(fromPythia(hook.snapshot[i], source == resonanceSources.end() ? 1 : 2));
    if (source != resonanceSources.end())
      decays.push_back(std::make_pair(source->second, result.size() - 1));
  }
  for (const auto& decay : decays)
    appendDecay(input, decay.first, decay.second, result);
  return result;
}

void particlesJSON(std::ostream& out, const std::vector<LHEParticle>& particles) {
  out << '[';
  for (std::size_t i = 0; i < particles.size(); ++i) {
    if (i) out << ',';
    particleJSON(out, particles[i]);
  }
  out << ']';
}

void legScalesJSON(std::ostream& out, const std::map<int, double>& scales) {
  out << '[';
  bool comma = false;
  for (const auto& entry : scales) {
    if (comma) out << ',';
    out << '[' << entry.first << ',' << entry.second << ']';
    comma = true;
  }
  out << ']';
}

void dipoleScalesJSON(std::ostream& out,
    const std::map<std::pair<int, int>, double>& scales) {
  out << '[';
  bool comma = false;
  for (const auto& entry : scales) {
    if (comma) out << ',';
    out << '[' << entry.first.first << ',' << entry.first.second << ',' << entry.second << ']';
    comma = true;
  }
  out << ']';
}

void nativeScalesJSON(std::ostream& out, const std::vector<NativeDipoleScale>& scales) {
  out << '[';
  for (std::size_t i = 0; i < scales.size(); ++i) {
    if (i) out << ',';
    const auto& dipole = scales[i];
    out << '[' << dipole.radiator << ',' << dipole.recoiler << ',' << dipole.side
        << ',' << dipole.scale << ']';
  }
  out << ']';
}

struct Continuation {
  std::string state;
  double scale;
  std::map<int, double> legScales;
  std::map<std::pair<int, int>, double> dipoleScales;
};

double exhaustedScale(const FirstEmission& hook) {
  const double cutoff = std::min(hook.cutoffISR, hook.cutoffFSR);
  const double scale = 0.5 * cutoff;
  if (!std::isfinite(scale) || !(scale > 0. && scale < cutoff))
    throw std::runtime_error("No representable positive exhausted shower scale");
  return scale;
}

Continuation continuationState(const InputEvent& input, const FirstEmission& hook,
    const std::vector<LHEParticle>& particles) {
  Continuation result;
  const bool emitted = input.kind == 1 && hook.seen;
  const bool unshowered = input.kind == 2;
  if (unshowered) {
    result.state = "unshowered"; result.scale = input.showerScale;
  } else if (emitted) {
    result.state = "emitted"; result.scale = hook.pt;
  } else {
    result.state = "exhausted"; result.scale = exhaustedScale(hook);
  }
  if (!std::isfinite(result.scale) || result.scale <= 0.)
    throw std::runtime_error("Invalid continuation shower scale");
  for (std::size_t i = 0; i < particles.size(); ++i)
    if (std::abs(particles[i].status) == 1) result.legScales[i+1] = result.scale;
  if (unshowered) {
    if (input.productionScales.size() != particles.size())
      throw std::runtime_error("Missing source production scales for H continuation");
    if (input.delta) {
      // Positive global-valued leg anchors retain the reader's effective
      // global maximum, which can exceed all final production minima. The
      // original positive pair entries restore those minima on import.
      // Keep zero pairs zero: the stock production reader ignores them,
      // while its directed FSR lookup uses zero as a genuine dipole stop.
      for (const auto& entry : input.dipoleScales)
        result.dipoleScales[entry.first] = std::min(result.scale, entry.second);
    }
    return result;
  }
  if (!emitted) return result;
  // hardParticles writes the two incoming legs and all
  // production outgoing legs first, then append any restored decay trees.
  std::vector<int> native = hook.incoming;
  native.insert(native.end(), hook.outgoing.begin(), hook.outgoing.end());
  if (native.size() > particles.size() || hook.incoming.size() != 2)
    throw std::runtime_error("Invalid continuation hard-system ordering");
  std::map<int, int> position;
  for (std::size_t i = 0; i < native.size(); ++i) {
    const int index = native[i];
    if (index < 1 || index >= hook.snapshot.size() || !position.emplace(index, i+1).second)
      throw std::runtime_error("Invalid continuation snapshot index");
    const auto& p = hook.snapshot[index]; const auto& q = particles[i];
    if (p.id() != q.id || p.col() != q.col || p.acol() != q.acol ||
        p.px() != q.px || p.py() != q.py || p.pz() != q.pz || p.e() != q.e)
      throw std::runtime_error("Continuation body disagrees with native snapshot");
  }
  const double floor = std::min(result.scale, exhaustedScale(hook));
  std::map<int, double> isr;
  for (const auto& dipole : hook.continuationISR) {
    const auto leg = position.find(dipole.radiator);
    if (leg == position.end() || leg->second != dipole.side ||
        dipole.recoiler != hook.incoming.at(2-dipole.side))
      throw std::runtime_error("Native ISR cap does not map to continuation beams");
    const double scale = std::min(result.scale, dipole.scale);
    if (isr.count(leg->second) && isr.at(leg->second) != scale)
      throw std::runtime_error("Distinct native ISR caps cannot share one LHE leg");
    isr[leg->second] = scale;
    result.legScales[leg->second] = std::max(floor, scale);
  }
  for (const auto& dipole : hook.continuationFSR) {
    const auto rad = position.find(dipole.radiator), rec = position.find(dipole.recoiler);
    if (rad == position.end() || rec == position.end() || rad->second < 3)
      throw std::runtime_error("Native FSR cap does not map to continuation legs");
    const auto key = std::make_pair(rad->second, rec->second);
    const double scale = std::min(result.scale, dipole.scale);
    const double stored = std::max(floor, scale);
    if (result.dipoleScales.count(key) && result.dipoleScales.at(key) != stored)
      throw std::runtime_error("Distinct native FSR caps cannot share one LHE dipole");
    result.dipoleScales[key] = stored;
  }
  return result;
}

std::string inertSourceScales(std::string extra) {
  for (const auto& tags : {std::make_pair(std::string("<scales"), std::string("<macnlops_source_scales")),
                          std::make_pair(std::string("</scales>"), std::string("</macnlops_source_scales>"))}) {
    std::size_t start = 0;
    while ((start = extra.find(tags.first, start)) != std::string::npos) {
      extra.replace(start, tags.first.size(), tags.second);
      start += tags.second.size();
    }
  }
  return extra;
}

void writeLHE(std::ostream& out, const InputEvent& input,
              const std::vector<LHEParticle>& particles,
              const FirstEmission& hook) {
  const auto h = fields(input.header, 6, "LHE event header");
  const auto continuation = continuationState(input, hook, particles);
  out << input.open << '\n'
      << particles.size() << ' ' << h[1] << ' ' << h[2] << ' ' << continuation.scale
      << ' ' << h[4] << ' ' << h[5] << '\n';
  for (const LHEParticle& p : particles) {
    out << p.id << ' ' << p.status << ' ' << p.mother1 << ' ' << p.mother2
        << ' ' << p.col << ' ' << p.acol << ' '
        << p.px << ' ' << p.py << ' ' << p.pz << ' ' << p.e << ' ' << p.m
        << ' ' << p.lifetime << ' ' << p.spin << '\n';
  }
  // All source annotations/weights remain source data. They are not a new
  // matching evaluation at the showered point; the JSONL identifies that point.
  if (input.hasHard) {
    // The retained hard block is source provenance after radiation. Its
    // Born point must never override this emitted event body on reimport.
    out << "<macnlops_endpoint version=\"1\" origin=\""
        << (hook.seen ? "shower" : "hard_annotation") << "\" stage=\"first\"/>\n";
  }
  out << "<macnlops_continuation version=\"1\" state=\"" << continuation.state
      << "\" scale=\"" << continuation.scale << "\" source_scalup=\"" << input.scale
      << "\" cutoff_isr=\"" << hook.cutoffISR << "\" cutoff_fsr=\"" << hook.cutoffFSR << "\"/>\n";
  out << "<scales";
  for (const auto& entry : continuation.legScales)
    out << " scalup_" << entry.first << "=\"" << entry.second << '"';
  for (const auto& entry : continuation.dipoleScales)
    out << ' ' << dipoleName(entry.first.first, entry.first.second) << "=\"" << entry.second << '"';
  out << ">\n</scales>\n";
  std::istringstream extras(inertSourceScales(input.extra));
  std::string line;
  while (std::getline(extras, line))
    if (line.find("macnlops_endpoint") == std::string::npos) out << line << '\n';
  out << input.close << '\n';
}

void recordJSON(std::ostream& out, std::size_t sequence, int kind,
                const InputEvent& input, const FirstEmission& hook,
                const std::vector<LHEParticle>& particles) {
  const auto continuation = continuationState(input, hook, particles);
  out << "{\"schema\":\"macnlops-one-step-v1\",\"sequence\":" << sequence
      << ",\"event_kind\":\"" << (kind == 1 ? "S" : "H")
      << "\",\"emitted\":" << (hook.seen ? "true" : "false")
      << ",\"emission_type\":\"" << (hook.type == 2 ? "ISR" :
                                         hook.type == 3 ? "FSR" : "none")
      << "\",\"pt_evol\":" << hook.pt << ",\"alpha_s_emission\":" << hook.emissionAlphaS
      << ",\"source_scalup\":" << input.scale
      << ",\"shower_scalup\":" << input.showerScale
      << ",\"continuation_state\":\"" << continuation.state << '"'
      << ",\"continuation_scalup\":" << continuation.scale
      << ",\"continuation_cap_source\":\"" << (continuation.state == "emitted" ?
           "stock-pythia-8.318-public-dipole-list" : continuation.state == "unshowered" ?
           "source-imported-starting-scales" : "effective-shower-cutoffs") << '"'
      << ",\"source_matching\":\"" << (input.delta ? "delta" : "ordinary") << '"'
      << ",\"production_scales_from_lhef\":" << (input.delta ? "true" : "false")
      << ",\"dipole_scales_from_lhef\":" << (input.delta ? "true" : "false")
      << ",\"beam_xmax\":[" << hook.beamXMaxA << ',' << hook.beamXMaxB
      << "],\"shower_cutoffs\":[" << hook.cutoffISR << ',' << hook.cutoffFSR << ']'
      << ",\"nominal_shower_weight\":1,\"trial_headroom\":1"
      << ",\"qcd_only\":true,\"hard_systems\":1"
      << ",\"flavour_count\":" << hook.flavourCount
      << ",\"n_quark_in\":" << hook.flavourCount
      << ",\"n_gluon_to_quark\":" << hook.flavourCount
      << ",\"alpha_s_nfmax\":" << hook.alphaSNfmax
      << ",\"shower_backend\":\"stock-pythia8\""
      << ",\"native_trial_warning_scope\":\"matching-endpoint\""
      << ",\"native_trial_warning_coverage\":\"stock-logged-messages\""
      << ",\"native_trial_warning_counts\":";
  warningJSON(out, hook.nativeTrialWarnings);
  out << ",\"source_production_scales\":[";
  for (std::size_t i = 0; i < input.productionScales.size(); ++i) {
    if (i) out << ',';
    out << input.productionScales[i];
  }
  out << "],\"source_dipole_scales\":";
  dipoleScalesJSON(out, input.dipoleScales);
  out << ",\"native_dipole_scales\":";
  dipoleScalesJSON(out, input.nativeDipoleScales);
  out << ",\"continuation_leg_scales\":";
  legScalesJSON(out, continuation.legScales);
  out << ",\"continuation_dipole_scales\":";
  dipoleScalesJSON(out, continuation.dipoleScales);
  out << ",\"native_continuation_isr\":";
  nativeScalesJSON(out, hook.continuationISR);
  out << ",\"native_continuation_fsr\":";
  nativeScalesJSON(out, hook.continuationFSR);
  out
      << ",\"source_weight\":" << input.weight
      << ",\"hard_annotation_used\":" << (input.hardUsed ? "true" : "false")
      << ",\"source_coordinate_origin\":\"" << (input.hardUsed ? "hard_annotation" :
           input.hasEndpoint ? "derived_endpoint" : "canonical_lhe")
      << "\",\"endpoint_origin\":\"" << (hook.seen ? "shower" :
           input.hardUsed ? "hard_annotation" : "canonical_lhe") << "\""
      << ",\"source_partition\":";
  if (input.hasHard) {
    out << "{\"version\":1,\"frame\":\"lab\",\"source\":\"" << input.hardSource
        << "\",\"partition\":\"" << input.hardPartition << "\"";
    const char* names[] = {"nup", "kind", "owner", "native_sector", "native_provider",
      "native_context", "native_history", "born_flow", "mc_kl_sum", "fixed_flow",
      "fks_sum", "flavour_coalesced", "max_bcol", "group_size"};
    for (std::size_t i = 0; i < input.hardMetadata.size(); ++i)
      out << ",\"" << names[i] << "\":" << input.hardMetadata[i];
    out << '}';
  } else out << "null";
  out << ",\"original_lhe_particles\":";
  particlesJSON(out, input.originalParticles);
  out << ",\"source_hard_particles\":";
  particlesJSON(out, input.hardParticles);
  out << ",\"source_particles\":";
  particlesJSON(out, input.particles);
  out << ",\"particles\":";
  particlesJSON(out, particles);
  int radiatorBefore = 0, radiatorAfter = 0, emittedIndex = 0;
  std::vector<int> recoilers;
  if (hook.seen) {
    for (int i = 1; i < hook.snapshot.size(); ++i) {
      const Pythia8::Particle& p = hook.snapshot[i];
      if (hook.type == 2 && p.status() == 43) {
        emittedIndex = i;
        radiatorAfter = p.mother1();
        radiatorBefore = hook.snapshot[radiatorAfter].daughter2();
      } else if (hook.type == 3 && p.status() == 51) {
        radiatorBefore = p.mother1();
        radiatorAfter = hook.snapshot[radiatorBefore].daughter1();
        emittedIndex = hook.snapshot[radiatorBefore].daughter2();
      }
      if (std::abs(p.status()) == 42 || std::abs(p.status()) == 52 ||
          std::abs(p.status()) == 53) recoilers.push_back(i);
    }
  }
  out << ",\"branch_indices\":{\"radiator_before\":" << radiatorBefore
      << ",\"radiator_after\":" << radiatorAfter
      << ",\"emitted\":" << emittedIndex << ",\"recoilers\":[";
  for (std::size_t i = 0; i < recoilers.size(); ++i) {
    if (i) out << ',';
    out << recoilers[i];
  }
  out << "]},\"pythia_event\":[";
  if (hook.seen) {
    for (int i = 1; i < hook.snapshot.size(); ++i) {
      if (i > 1) out << ',';
      const Pythia8::Particle& p = hook.snapshot[i];
      out << "{\"index\":" << i << ",\"id\":" << p.id()
          << ",\"status\":" << p.status() << ",\"mothers\":["
          << p.mother1() << ',' << p.mother2() << "],\"daughters\":["
          << p.daughter1() << ',' << p.daughter2()
          << "],\"colors\":[" << p.col() << ',' << p.acol()
          << "],\"p\":[" << p.e() << ',' << p.px() << ',' << p.py()
          << ',' << p.pz() << "],\"mass\":" << p.m() << '}';
    }
  }
  out << "],\"hard_pythia_indices\":[";
  std::vector<int> indices = hook.incoming;
  indices.insert(indices.end(), hook.outgoing.begin(), hook.outgoing.end());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (i) out << ',';
    out << indices[i];
  }
  out << "]}\n";
}

struct Options {
  std::string input, output, metadata, settings, settingsOutput, pdf, kind, probe;
  int seed, alphaSorder, beamIdA, beamIdB, flavourCount;
  double alphaSMZ, beamEnergyA, beamEnergyB;
  bool masslessLightQuarks, mcatnloDelta;
  Options() : pdf("LHAPDF6:MSTW2008nlo90cl/0"), kind("auto"), seed(1),
    alphaSorder(-1), beamIdA(2212), beamIdB(2212), flavourCount(5), alphaSMZ(-1.),
    beamEnergyA(-1.), beamEnergyB(-1.), masslessLightQuarks(false), mcatnloDelta(false) {}
};

Options options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "MacNLOPS --input IN.lhe --output OUT.lhe --metadata OUT.jsonl\n"
                << " [--settings FILE] [--pdf LHAPDF6:SET/MEMBER] [--seed N]\n"
                << " [--alpha-s-mz VALUE] [--alpha-s-order 0|1|2|3]\n"
                << " [--flavour-count 4|5] (incoming/splitting flavours; stock alphaSnfmax=5)\n"
                << " [--settings-output FILE] [--massless-light-quarks]\n"
                << " [--mcatnlo-delta] (read required per-colour LHE starting scales)\n"
                << "Or --probe-config JSON --beam-id-a ID --beam-id-b ID\n"
                << " --beam-energy-a E --beam-energy-b E [same physics options]\n"
                << " [--event-kind auto|S|H]\n"
                << "S events receive exactly one competing first ISR/FSR proposal.\n"
                << "H events pass through at their original real phase-space point.\n"
                << "The final matching rejection is a separate evaluator stage.\n";
      std::exit(0);
    }
    if (arg == "--massless-light-quarks") {
      o.masslessLightQuarks = true;
      continue;
    }
    if (arg == "--mcatnlo-delta") {
      o.mcatnloDelta = true;
      continue;
    }
    if (++i == argc) throw std::runtime_error("Missing value for " + arg);
    const std::string value = argv[i];
    if (arg == "--input") o.input = value;
    else if (arg == "--output") o.output = value;
    else if (arg == "--metadata") o.metadata = value;
    else if (arg == "--settings") o.settings = value;
    else if (arg == "--settings-output") o.settingsOutput = value;
    else if (arg == "--pdf") o.pdf = value;
    else if (arg == "--event-kind") o.kind = value;
    else if (arg == "--seed") o.seed = std::stoi(value);
    else if (arg == "--alpha-s-mz") o.alphaSMZ = number(value);
    else if (arg == "--alpha-s-order") o.alphaSorder = std::stoi(value);
    else if (arg == "--flavour-count") o.flavourCount = std::stoi(value);
    else if (arg == "--probe-config") o.probe = value;
    else if (arg == "--beam-id-a") o.beamIdA = std::stoi(value);
    else if (arg == "--beam-id-b") o.beamIdB = std::stoi(value);
    else if (arg == "--beam-energy-a") o.beamEnergyA = number(value);
    else if (arg == "--beam-energy-b") o.beamEnergyB = number(value);
    else throw std::runtime_error("Unknown option: " + arg);
  }
  if (o.probe.empty() && (o.input.empty() || o.output.empty() || o.metadata.empty()))
    throw std::runtime_error("--input, --output and --metadata are required");
  if (o.probe.empty() && (o.input == o.output || o.input == o.metadata || o.output == o.metadata))
    throw std::runtime_error("Input and output paths must be distinct");
  if (o.input.size() >= 3 && o.input.substr(o.input.size() - 3) == ".gz")
    throw std::runtime_error("Decompress the LHE input before this executable");
  if (o.seed < 1 || o.seed > 900000000)
    throw std::runtime_error("Seed must be in [1,900000000]");
  if (o.kind != "auto" && o.kind != "S" && o.kind != "H")
    throw std::runtime_error("--event-kind must be auto, S, or H");
  if (o.pdf.compare(0, 8, "LHAPDF6:") != 0)
    throw std::runtime_error("Supply an ordinary MSbar PDF via LHAPDF6:SET/MEMBER");
  if (o.alphaSMZ != -1. && (o.alphaSMZ < 0.06 || o.alphaSMZ > 0.25))
    throw std::runtime_error("--alpha-s-mz must be in [0.06,0.25]");
  if (o.alphaSorder < -1 || o.alphaSorder > 3)
    throw std::runtime_error("--alpha-s-order must be 0, 1, 2 or 3");
  if (o.flavourCount != 4 && o.flavourCount != 5)
    throw std::runtime_error("--flavour-count must be 4 or 5");
  if (!o.probe.empty() && (o.beamEnergyA <= 0. || o.beamEnergyB <= 0.))
    throw std::runtime_error("Probe requires positive --beam-energy-a and --beam-energy-b");
  if (o.settingsOutput.empty())
    o.settingsOutput = (o.probe.empty() ? o.metadata : o.probe) + ".settings.cmnd";
  if (o.settingsOutput == o.input || o.settingsOutput == o.output ||
      o.settingsOutput == o.metadata)
    throw std::runtime_error("Settings snapshot needs a distinct output path");
  return o;
}

class ProbeLHAup : public LHAup {
public:
  explicit ProbeLHAup(const Options& options) : options(options) {}
  bool setInit() override {
    setBeamA(options.beamIdA, options.beamEnergyA, 0, 0);
    setBeamB(options.beamIdB, options.beamEnergyB, 0, 0);
    setStrategy(-4);
    addProcess(1, 1.e-9, 0., 1.e-9);
    return true;
  }
  bool setEvent(int = 0) override { return false; }
private:
  Options options;
};

void validateSourceMatchingSettings(Pythia& p, const Options& o) {
  for (const char* key : {"Beams:setProductionScalesFromLHEF",
                         "Beams:setDipoleShowerStartingScalesFromLHEF"})
    if (p.flag(key) != o.mcatnloDelta)
      throw std::runtime_error(std::string("Source matching scale setting changed: ") + key);
}

void applyPhysicalSettings(Pythia& p, const std::string& filename) {
  if (filename.empty()) return;
  std::ifstream input(filename.c_str());
  if (!input) throw std::runtime_error("Cannot read Pythia settings: " + filename);
  std::string line;
  while (std::getline(input, line)) {
    // Matching certifies one fixed shower profile. External files supply
    // only model masses and the two probed infrared support boundaries.
    const std::string command = trim(line.substr(0, line.find_first_of("!#")));
    if (command.empty()) continue;
    const std::size_t equal = command.find('=');
    const std::string key = trim(command.substr(0, equal));
    bool mass = false;
    const std::size_t colon = key.find(':');
    if (colon != std::string::npos && key.substr(colon) == ":m0") {
      const std::string id = key.substr(0, colon);
      const std::size_t first = !id.empty() && id[0] == '-' ? 1 : 0;
      mass = first < id.size() &&
        id.find_first_not_of("0123456789", first) == std::string::npos;
    }
    if (equal == std::string::npos ||
        (!mass && key != "SpaceShower:pTmin" && key != "TimeShower:pTmin"))
      throw std::runtime_error("Incompatible scalar shower setting: " + key +
        " (settings files permit only PDG:m0 and the ISR/FSR pTmin values)");
    const double value = number(trim(command.substr(equal + 1)));
    if (value < 0. || (!mass && value == 0.))
      throw std::runtime_error("Invalid shower mass or cutoff: " + command);
    if (!p.readString(command))
      throw std::runtime_error("Cannot apply Pythia setting: " + command);
  }
}

void configure(Pythia& p, const Options& o) {
  // Check before the source/proposal run rather than at its first emission.
  if (PYTHIA_VERSION_INTEGER != 8318)
    throw std::runtime_error("Native continuation scale export requires stock Pythia 8.318");
  // Explicit simple-shower profile. The evaluator must use these same settings;
  // a source MC counterterm is not automatically the actual shower density.
  const char* profile[] = {
    "PartonShowers:model = 1", "PartonLevel:MPI = off",
    "PartonLevel:Remnants = off", "HadronLevel:all = off",
    "ProcessLevel:resonanceDecays = off", "PartonLevel:FSRinResonances = off",
    "SpaceShower:QCDshower = on", "TimeShower:QCDshower = on",
    "TimeShower:interleave = on", "SpaceShower:MEcorrections = off",
    "TimeShower:MEcorrections = off", "SpaceShower:MEafterFirst = off",
    "TimeShower:MEafterFirst = off", "SpaceShower:phiPolAsym = off",
    "SpaceShower:phiPolAsymHard = off", "TimeShower:phiPolAsymHard = off",
    "SpaceShower:phiIntAsym = off", "TimeShower:phiPolAsym = off",
    "SpaceShower:rapidityOrder = off", "SpaceShower:samePTasMPI = off",
    "SpaceShower:pT0parametrization = 0", "SpaceShower:dipoleRecoil = off",
    "SpaceShower:pTmaxMatch = 1", "TimeShower:pTmaxMatch = 1",
    "SpaceShower:pTmaxFudge = 1.", "TimeShower:pTmaxFudge = 1.",
    "SpaceShower:renormMultFac = 1.", "TimeShower:renormMultFac = 1.",
    "SpaceShower:factorMultFac = 1.", "TimeShower:factorMultFac = 1.",
    "SpaceShower:pTdampMatch = 0", "TimeShower:pTdampMatch = 0",
    "SpaceShower:alphaSuseCMW = off", "TimeShower:alphaSuseCMW = off",
    "SpaceShower:alphaSvalue = 0.118", "TimeShower:alphaSvalue = 0.118",
    "SpaceShower:alphaSorder = 1", "TimeShower:alphaSorder = 1",
    "TimeShower:globalRecoil = on", "TimeShower:globalRecoilMode = 0",
    "TimeShower:allowBeamRecoil = on",
    "TimeShower:nMaxGlobalRecoil = 100000", "TimeShower:nMaxGlobalBranch = 1",
    "TimeShower:limitPTmaxGlobal = on", "TimeShower:dampenBeamRecoil = off",
    "TimeShower:weightGluonToQuark = 1", "TimeShower:recoilDeadCone = on",
    "TimeShower:nGluonToQuark = 5", "SpaceShower:nQuarkIn = 5",
    "TimeShower:useFixedFacScale = off", "SpaceShower:useFixedFacScale = off",
    "TimeShower:pdfMode = 0", "SpaceShower:pdfMode = 0",
    "TimeShower:cEmitQ = 0.", "TimeShower:cEmitC = 0.",
    "TimeShower:cEmitB = 0.", "TimeShower:cEmitG = 0.",
    "TimeShower:cSplit = 0.", "TimeShower:cSplitC = 0.",
    "TimeShower:cSplitB = 0.",
    "TimeShower:hEmitHard = 0.", "TimeShower:hEmitColl = 0.",
    "TimeShower:hEmitSoft = 0.",
    "TimeShower:hSplitHard = 0.", "TimeShower:hSplitColl = 0.",
    "SpaceShower:QEDshowerByQ = off", "SpaceShower:QEDshowerByL = off",
    "TimeShower:QEDshowerByQ = off", "TimeShower:QEDshowerByL = off",
    "TimeShower:QEDshowerByOther = off",
    "TimeShower:QEDshowerByGamma = off", "TimeShower:weakShower = off",
    "SpaceShower:weakShower = off",
    // The normalized forced-heavy-flavour sampler must retain its internal
    // accept/reject loop. UncertaintyBands defers that rejection to branch(),
    // and therefore cannot serve as generic central trial headroom here.
    "UncertaintyBands:doVariations = off",
    "Random:setSeed = on", "Check:abortIfVeto = on",
    "Init:showChangedSettings = off", "Init:showChangedParticleData = off",
    "Next:numberShowInfo = 0", "Next:numberShowProcess = 0",
    "Next:numberShowEvent = 0", "Next:numberCount = 0"
  };
  for (const char* command : profile)
    if (!p.readString(command))
      throw std::runtime_error(std::string("Unsupported Pythia setting: ") + command);
  for (const auto& key : qcdOnlyOff())
    if (!p.readString(key + " = off"))
      throw std::runtime_error("Unsupported QCD-only setting: " + key);
  // Zero is outside the tune-oriented XML range, but is the documented
  // unregularized analytic shower limit needed for matching.
  p.settings.forceParm("SpaceShower:pT0Ref", 0.);
  // The active-flavour hard scheme uses massless quarks and gluons. Other
  // model masses, including four-flavour bottom and top, come from settings.
  for (int id = 1; id <= o.flavourCount; ++id) p.particleData.m0(id, 0.);
  p.particleData.m0(21, 0.);
  p.particleData.m0(11, 0.); p.particleData.m0(13, 0.);
  applyPhysicalSettings(p, o.settings);
  // Enforce lifecycle and sample identity even when applying a settings file.
  p.readString("Beams:frameType = 5");
  p.readString("PDF:pSet = " + o.pdf);
  p.readString("Random:seed = " + std::to_string(o.seed));
  p.readString("Check:abortIfVeto = on");
  // Match the ordinary stock MC@NLO-Delta interface. With an external LHAup
  // reader these flags alone do not import production scales; CountingReader
  // also enables LHAupLHEF's setScalesFromLHEF constructor option.
  for (const char* key : {"Beams:setProductionScalesFromLHEF",
                         "Beams:setDipoleShowerStartingScalesFromLHEF"})
    if (!p.readString(std::string(key) + (o.mcatnloDelta ? " = on" : " = off")))
      throw std::runtime_error(std::string("Unsupported source matching scale setting: ") + key);
  if (o.alphaSMZ > 0.) {
    p.settings.parm("SpaceShower:alphaSvalue", o.alphaSMZ);
    p.settings.parm("TimeShower:alphaSvalue", o.alphaSMZ);
  }
  if (o.alphaSorder >= 0) {
    p.settings.mode("SpaceShower:alphaSorder", o.alphaSorder);
    p.settings.mode("TimeShower:alphaSorder", o.alphaSorder);
  }
  // --massless-light-quarks is retained as a compatibility spelling. The
  // declared flavour scheme already requires these model masses to be zero;
  // it must not silently overwrite a contradictory supplied model setting.
  for (int id = 1; id <= o.flavourCount; ++id)
    if (p.particleData.m0(id) != 0.)
      throw std::runtime_error("Active-flavour hard-model quarks must be massless");
  if (p.particleData.m0(21) != 0.)
    throw std::runtime_error("Hard-model gluons must be massless");
  p.settings.forceMode("SpaceShower:nQuarkIn", o.flavourCount);
  p.settings.forceMode("TimeShower:nGluonToQuark", o.flavourCount);
  // Stock AlphaStrong supports a minimum nfmax of five. The hard/PDF
  // flavour scheme still controls incoming flavours and g->qq species.
  p.settings.forceMode("StandardModel:alphaSnfmax", 5);
  const char* scalarOff[] = {
    "SpaceShower:MEcorrections", "TimeShower:MEcorrections",
    "SpaceShower:phiPolAsym", "TimeShower:phiPolAsym",
    "SpaceShower:phiPolAsymHard", "TimeShower:phiPolAsymHard",
    "SpaceShower:phiIntAsym", "SpaceShower:rapidityOrder", "SpaceShower:samePTasMPI",
    "SpaceShower:dipoleRecoil",
    "SpaceShower:alphaSuseCMW", "TimeShower:alphaSuseCMW",
    "TimeShower:dampenBeamRecoil", "SpaceShower:QEDshowerByQ",
    "SpaceShower:QEDshowerByL", "TimeShower:QEDshowerByQ",
    "TimeShower:QEDshowerByL", "TimeShower:QEDshowerByGamma",
    "TimeShower:QEDshowerByOther",
    "SpaceShower:weakShower", "TimeShower:weakShower",
    "TimeShower:useFixedFacScale", "SpaceShower:useFixedFacScale",
    "UncertaintyBands:doVariations"
  };
  for (const char* key : scalarOff)
    if (p.flag(key))
      throw std::runtime_error(std::string("Incompatible scalar shower setting: ") + key);
  const char* finiteZero[] = {
    "TimeShower:cEmitQ", "TimeShower:cEmitC", "TimeShower:cEmitB",
    "TimeShower:cEmitG", "TimeShower:cSplit", "TimeShower:cSplitC",
    "TimeShower:cSplitB", "SpaceShower:pT0Ref",
    "TimeShower:hEmitHard", "TimeShower:hEmitColl", "TimeShower:hEmitSoft",
    "TimeShower:hSplitHard", "TimeShower:hSplitColl"
  };
  for (const char* key : finiteZero)
    if (p.parm(key) != 0.)
      throw std::runtime_error(std::string("Incompatible scalar shower setting: ") + key);
  const char* scalarOn[] = {
    "TimeShower:globalRecoil", "TimeShower:limitPTmaxGlobal", "TimeShower:allowBeamRecoil",
    "TimeShower:recoilDeadCone", "PartonLevel:ISR", "PartonLevel:FSR",
    "PartonLevel:FSRinProcess", "SpaceShower:QCDshower", "TimeShower:QCDshower"
  };
  for (const char* key : scalarOn)
    if (!p.flag(key))
      throw std::runtime_error(std::string("Incompatible scalar shower setting: ") + key);
  const char* unitFactors[] = {
    "TimeShower:pTmaxFudge", "SpaceShower:pTmaxFudge",
    "TimeShower:renormMultFac", "SpaceShower:renormMultFac",
    "TimeShower:factorMultFac", "SpaceShower:factorMultFac"
  };
  for (const char* key : unitFactors)
    if (p.parm(key) != 1.)
      throw std::runtime_error(std::string("Incompatible scalar shower setting: ") + key);
  const std::pair<const char*, int> scalarModes[] = {
    {"PartonShowers:model", 1}, {"TimeShower:pTmaxMatch", 1},
    {"SpaceShower:pTmaxMatch", 1}, {"TimeShower:pTdampMatch", 0},
    {"SpaceShower:pTdampMatch", 0}, {"TimeShower:globalRecoilMode", 0},
    {"SpaceShower:pT0parametrization", 0},
    {"SpaceShower:nQuarkIn", o.flavourCount}, {"TimeShower:nGluonToQuark", o.flavourCount},
    {"StandardModel:alphaSnfmax", 5},
    {"SpaceShower:pdfMode", 0}, {"TimeShower:pdfMode", 0},
    {"TimeShower:nMaxGlobalBranch", 1}, {"TimeShower:nMaxGlobalRecoil", 100000},
    {"TimeShower:weightGluonToQuark", 1}
  };
  for (const auto& entry : scalarModes)
    if (p.mode(entry.first) != entry.second)
      throw std::runtime_error(std::string("Incompatible scalar shower setting: ") + entry.first);
  if (p.flag("PartonLevel:MPI") || p.flag("HadronLevel:all") ||
      !p.flag("TimeShower:interleave") || p.flag("PartonLevel:FSRinResonances"))
    throw std::runtime_error("One-step lifecycle requires MPI/hadronization/"
                             "decay showers off and interleaved ISR/FSR");
  validateQCDOnlySettings(p);
  validateSourceMatchingSettings(p, o);
}

void setEffectiveCutoffs(Pythia& p, FirstEmission& hook) {
  // The simple showers raise their PRIVATE cutoffs near the Landau pole and
  // do not update Settings. Reproduce their initialization with the same
  // public AlphaStrong implementation (Simple{Space,Time}Shower.cc, 8.318),
  // then expose the effective values in both Settings and event metadata.
  // The same couplings later give the alpha_s of each first emission.
  AlphaStrong& isr = hook.isrAlphaS;
  AlphaStrong& fsr = hook.fsrAlphaS;
  const int nf = p.mode("StandardModel:alphaSnfmax");
  isr.init(p.parm("SpaceShower:alphaSvalue"), p.mode("SpaceShower:alphaSorder"),
           nf, p.flag("SpaceShower:alphaSuseCMW"));
  fsr.init(p.parm("TimeShower:alphaSvalue"), p.mode("TimeShower:alphaSorder"),
           nf, p.flag("TimeShower:alphaSuseCMW"),
           p.parm("TimeShower:alphaSmax"), p.parm("TimeShower:alphaSrenormShift"));
  const double margin = p.mode("TimeShower:alphaSorder") < 2 ? 1.1 : 1.4;
  const double isrMin2 = 1.1 * 1.1 * isr.Lambda3() * isr.Lambda3() /
    p.parm("SpaceShower:renormMultFac");
  const double fsrMin2 = fsr.Lambda3() * fsr.Lambda3() /
    p.parm("TimeShower:renormMultFac") *
    (margin * margin - p.parm("TimeShower:alphaSrenormShift"));
  hook.cutoffISR = std::max(p.parm("SpaceShower:pTmin"), std::sqrt(std::max(0., isrMin2)));
  hook.cutoffFSR = std::max(p.parm("TimeShower:pTmin"), std::sqrt(std::max(0., fsrMin2)));
  // One ulp headroom avoids a second internal raise from rounding differences.
  hook.cutoffISR = std::nextafter(hook.cutoffISR, std::numeric_limits<double>::infinity());
  hook.cutoffFSR = std::nextafter(hook.cutoffFSR, std::numeric_limits<double>::infinity());
  p.settings.forceParm("SpaceShower:pTmin", hook.cutoffISR);
  p.settings.forceParm("TimeShower:pTmin", hook.cutoffFSR);
}

void configureHook(Pythia& p, const Options& o, FirstEmission& hook) {
  hook.engine = &p;
  hook.flavourCount = o.flavourCount;
  hook.alphaSNfmax = p.mode("StandardModel:alphaSnfmax");
  setEffectiveCutoffs(p, hook);
}

void validateNominalWeights(const Pythia& p) {
  // Variations are disabled and retained source weights must stay unchanged.
  // Stock trial-bound warnings are recorded, never used as an event veto.
  const WeightsSimpleShower& weights = p.info.weightContainerPtr->weightsSimpleShower;
  for (int i = 0; i < weights.getWeightsSize(); ++i)
    if (!std::isfinite(weights.getWeightsValue(i)) ||
        std::abs(weights.getWeightsValue(i) - 1.) > 1.e-12)
      throw std::runtime_error("Native shower changed the nominal auxiliary weight");
}

void probeConfiguration(const Options& o, Pythia& p) {
  std::shared_ptr<FirstEmission> hook = std::make_shared<FirstEmission>();
  configureHook(p, o, *hook);
  p.setLHAupPtr(std::make_shared<ProbeLHAup>(o));
  p.setUserHooksPtr(hook);
  if (!p.init()) throw std::runtime_error("Pythia configuration probe initialization failed");
  validateQCDOnlySettings(p);
  validateSourceMatchingSettings(p, o);
  hook->captureBeamLimits();
  if (!p.settings.writeFile(o.settingsOutput, true))
    throw std::runtime_error("Cannot write probed Pythia settings snapshot");
  std::ofstream out(o.probe.c_str());
  if (!out) throw std::runtime_error("Cannot write configuration probe");
  out << std::setprecision(17)
      << "{\"schema\":\"macnlops-shower-config-v1\",\"pythia_version\":\"8.318\","
      << "\"qcd_only\":true,\"trial_headroom\":1,\"shower_backend\":\"stock-pythia8\""
      << ",\"source_matching\":\"" << (o.mcatnloDelta ? "delta" : "ordinary") << '"'
      << ",\"production_scales_from_lhef\":" << (o.mcatnloDelta ? "true" : "false")
      << ",\"dipole_scales_from_lhef\":" << (o.mcatnloDelta ? "true" : "false")
      << ",\"flavour_count\":" << o.flavourCount
      << ",\"n_quark_in\":" << p.mode("SpaceShower:nQuarkIn")
      << ",\"n_gluon_to_quark\":" << p.mode("TimeShower:nGluonToQuark")
      << ",\"alpha_s_nfmax\":" << p.mode("StandardModel:alphaSnfmax")
      << ",\"beam_ids\":[" << o.beamIdA << ',' << o.beamIdB
      << "],\"beam_energies_lab\":[" << o.beamEnergyA << ',' << o.beamEnergyB
      << "],\"beam_energies_cm\":[" << p.beamA.e() << ',' << p.beamB.e()
      << "],\"beam_masses\":[" << p.beamA.m() << ',' << p.beamB.m()
      << "],\"beam_xmax\":[" << hook->beamXMaxA << ',' << hook->beamXMaxB
      << "],\"shower_cutoffs\":[" << hook->cutoffISR << ',' << hook->cutoffFSR
      << "],\"heavy_mass_floors\":{\"4\":" << std::max(1.2, p.particleData.m0(4))
      << ",\"5\":" << std::max(4., p.particleData.m0(5))
      << "},\"particle_masses\":{";
  bool first = true;
  for (const auto& entry : p.particleData) {
    const int id = entry.first;
    if (id <= 0) continue;
    if (!first) out << ',';
    first = false;
    out << '\"' << id << "\":" << p.particleData.m0(id);
  }
  out << "},\"qcd_only_off_switches\":{";
  first = true;
  for (const auto& key : qcdOnlyOff()) {
    if (!first) out << ',';
    first = false;
    out << '\"' << key << "\":" << (p.settings.flag(key) ? "true" : "false");
  }
  out << "}}\n";
  if (!out) throw std::runtime_error("Configuration probe write failed");
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options o = options(argc, argv);
    Pythia pythia;
    configure(pythia, o);
    if (!o.probe.empty()) {
      probeConfiguration(o, pythia);
      return 0;
    }
    RawReader raw(o.input);
    // Pythia exposes a const view of its mutable Info, whereas its LHAup
    // constructor requires the mutable pointer used by the native LHE reader.
    std::shared_ptr<CountingReader> reader = std::make_shared<CountingReader>(
      const_cast<Info*>(&pythia.info), o.input, pythia, o.mcatnloDelta);
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

    std::ofstream output(o.output.c_str()), metadata(o.metadata.c_str());
    if (!output || !metadata) throw std::runtime_error("Cannot open output files");
    output << std::scientific << std::setprecision(17);
    metadata << std::setprecision(17);
    std::size_t count = 0, emitted = 0, hard = 0;
    InputEvent input;
    while (raw.next(input)) {
      ++count;
      int kind = o.kind == "S" ? 1 : o.kind == "H" ? 2 : input.kind;
      if (!kind) throw std::runtime_error("Missing #aMCatNLO kind; use --event-kind");
      if (input.hasHard && kind != input.hardMetadata[1])
        throw std::runtime_error("Explicit kind conflicts with hard-event annotation");
      if (!input.kind) input.kind = kind;
      hook->reset();
      hook->source = &input;
      std::vector<LHEParticle> particles = input.particles;
      reader->expect(input);
      const std::size_t before = reader->readCount;
      if (kind == 2) {
        if (!reader->skipOne()) throw std::runtime_error("Cannot consume H event");
        ++hard;
      } else {
        const auto warningStart = nativeTrialCounts(pythia);
        const bool success = pythia.next();
        validateNominalWeights(pythia);
        hook->nativeTrialWarnings = warningDifference(nativeTrialCounts(pythia), warningStart);
        if (!success && !hook->finished)
          throw std::runtime_error("Shower failed before first emission at event " +
                                   std::to_string(count));
        if (success && hook->finished)
          throw std::runtime_error("First-emission technical stop did not abort");
        if (hook->seen) {
          particles = hardParticles(*hook, input);
          if (particles.size() != input.particles.size() + 1)
            throw std::runtime_error("Shower changed external multiplicity unexpectedly");
          ++emitted;
        }
      }
      if (reader->readCount != before + 1)
        throw std::runtime_error("Pythia consumed more than one source event");
      validateQCDOnlyEndpoint(particles, input);
      output << input.prefix;
      writeLHE(output, input, particles, *hook);
      recordJSON(metadata, count, kind, input, *hook, particles);
      if (!output || !metadata) throw std::runtime_error("Output write failure");
    }
    output << input.prefix;
    if (!output || !metadata) throw std::runtime_error("Output write failure");
    std::cout << "MAcNLOPS proposals: " << count << " source events, "
              << emitted << " S emissions, " << hard << " H pass-throughs\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "MAcNLOPS: " << error.what() << '\n';
    return 1;
  }
}
