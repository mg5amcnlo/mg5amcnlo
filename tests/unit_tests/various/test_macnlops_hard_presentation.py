"""Compiled hard-annotation presentation and same-input native-law checks.

Opt in with PYTHIA8=/prefix; no density/rejection evaluator is substituted.
"""
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'Template/NLO/MCatNLO/srcPythia8'
HEADER = '''<LesHouchesEvents version="3.0">
<header><generator>Hard-coordinate presentation regression</generator></header>
<init>
2212 2212 6500 6500 0 0 0 0 -4 1
1 0 1 1
</init>
'''


def particles(mass=0., final_mass=0., pdg=2):
    pin = math.sqrt(100.**2 - mass**2)
    pout = math.sqrt(100.**2 - final_mass**2)
    return [f'{pdg} -1 0 0 501 0 0 0 {pin:.17g} 100 {mass:.17g} 0 9',
            f'{-pdg} -1 0 0 0 501 0 0 {-pin:.17g} 100 {mass:.17g} 0 9',
            f'11 1 1 2 0 0 {pout:.17g} 0 0 100 {final_mass:.17g} 0 9',
            f'-11 1 1 2 0 0 {-pout:.17g} 0 0 100 {final_mass:.17g} 0 9']


def hard_block(kind=1, records=None, partition='sampled', group_size=0):
    records = records or particles()
    coalesced = int(partition == 'global_sum')
    return ('<macnlops_hard_event version="1" frame="lab" source="ordinary" '
            f'partition="{partition}">\n'
            f'{len(records)} {kind} 1 1 1 1 0 1 1 0 0 {coalesced} 1 {group_size}\n' +
            '\n'.join(records) + '\n</macnlops_hard_event>\n')


def event(kind=1, scale=100., weight=1., shuffled=True, annotation=True,
          hard=None, records=None, partition='sampled', group_size=0,
          resonance=False):
    records = records or particles(.5 if shuffled else 0., .000511 if shuffled else 0.)
    if resonance:
        records = records[:2] + ['23 2 1 2 0 0 0 0 0 200 200 0 9'] + [
            row.replace('1 1 2 0 0', '1 3 3 0 0', 1) for row in records[2:]]
    return (f'<event>\n{len(records)} 1 {weight:.17g} {scale:.17g} .0072973525693 .118\n' +
            '\n'.join(records) + '\n' +
            (hard_block(kind, hard, partition, group_size) if annotation else '') +
            f'#aMCatNLO {kind} 3 1 1 2 {scale:.17g} 0 -9 4 0 0 0 0 0 0\n'
            f'<rwgt><wgt id="nominal">{weight:.17g}</wgt></rwgt>\n'
            '# preserved exact source extras\n</event>\n')


def blocks(path):
    return re.findall(r'<event(?:\s[^>]*)?>.*?</event>', path.read_text(), re.S)


def handoff(block):
    matches = re.findall(r'<macnlops_continuation\b[^>]*/>', block)
    if len(matches) != 1:
        raise AssertionError('Expected exactly one first-endpoint continuation marker')
    marker = ET.fromstring(matches[0]).attrib
    if set(marker) != {'version', 'state', 'scale', 'source_scalup', 'cutoff_isr', 'cutoff_fsr'}:
        raise AssertionError('Unexpected continuation schema')
    return marker


def active_scales(block):
    matches = re.findall(r'<scales\b[^>]*>\s*</scales>', block, re.S)
    if len(matches) != 1:
        raise AssertionError('Expected exactly one active scale matrix')
    return {name: float(value) for name, value in ET.fromstring(matches[0]).attrib.items()}


def lhe_particles(block):
    lines = block.splitlines()
    nup = int(lines[1].split()[0])
    result = []
    for line in lines[2:2+nup]:
        values = line.split()
        if len(values) != 13:
            raise AssertionError('Malformed LHE particle fixture')
        result.append(tuple(map(int, values[:6])) + tuple(map(float, values[6:])))
    return result


def assert_handoff(test, source, endpoint, state):
    before, after = source.splitlines()[1].split(), endpoint.splitlines()[1].split()
    marker = handoff(endpoint)
    test.assertEqual(marker['version'], '1')
    test.assertEqual(marker['state'], state)
    test.assertEqual([after[i] for i in (1, 2, 4, 5)], [before[i] for i in (1, 2, 4, 5)])
    test.assertEqual(float(marker['source_scalup']), float(before[3]))
    test.assertEqual(float(after[3]), float(marker['scale']))
    test.assertGreater(float(marker['scale']), 0.)
    for key, value in active_scales(endpoint).items():
        if key.count('_') == 1:
            test.assertGreater(value, 0.)
        else:
            test.assertGreaterEqual(value, 0.)
        test.assertLessEqual(value, float(marker['scale']))
    if state == 'exhausted':
        test.assertLess(float(marker['scale']), min(float(marker['cutoff_isr']),
                                                   float(marker['cutoff_fsr'])))
    for pattern in (r'#aMCatNLO[^\n]*', r'<rwgt>.*?</rwgt>', r'<macnlops_hard_event\b.*?</macnlops_hard_event>'):
        test.assertEqual(re.findall(pattern, endpoint, re.S), re.findall(pattern, source, re.S))
    return marker


def delta_event(kind=1, resonance=False, scale=.01):
    # Four coloured external legs exercise outgoing-index changes when an
    # auxiliary resonance is inserted into the ordinary transport body.
    records = particles()
    records[1] = records[1].replace('0 501 ', '0 502 ')
    records[2] = records[2].replace('11 1 1 2 0 0 ', '1 1 1 2 501 0 ')
    records[3] = records[3].replace('-11 1 1 2 0 0 ', '-1 1 1 2 0 502 ')
    text = event(kind=kind, scale=scale, shuffled=False, records=records,
                 hard=records, resonance=resonance).replace('source="ordinary"', 'source="delta"')
    if resonance:
        body, annotation = text.split('<macnlops_hard_event', 1)
        body = body.replace('1 1 1 2 501 0 ', '1 1 3 3 501 0 ')
        body = body.replace('-1 1 1 2 0 502 ', '-1 1 3 3 0 502 ')
        text = body + '<macnlops_hard_event' + annotation
    shift = int(resonance)
    pairs = [(1, 3+shift, 40.), (3+shift, 1, 25.),
             (2, 4+shift, 30.), (4+shift, 2, 10.)]
    attributes = ' '.join(f"scalup_{i}_{j}='{value:.17g}'" for i, j, value in pairs)
    packet = f"<scales muf='-1' mur='-1' {attributes}>\n</scales>\n"
    if kind == 2:
        packet += ('<macnlops_delta_context version="1" owner="1" insertion=".25" '
                   f'registry="{"a" * 64}">\n1 1 1 .75\n</macnlops_delta_context>\n')
    return text.replace('#aMCatNLO', packet + '#aMCatNLO')


class TestNativeContinuationFormatting(unittest.TestCase):
    """Small standard-library-only checks; never link or initialize Pythia."""
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which(os.environ.get('CXX', 'g++'))
        if not compiler:
            raise unittest.SkipTest('A C++ compiler is needed for parser/format controls')
        text = (SOURCE / 'MacNLOPS.cc').read_text()
        begin = '// BEGIN MACNLOPS NATIVE SCALE LISTING HELPERS'
        end = '// END MACNLOPS NATIVE SCALE LISTING HELPERS'
        helpers = text.split(begin, 1)[1].split(end, 1)[0]
        continuation = text[text.index('struct Continuation {'):text.index('void recordJSON(')]
        particle_type = text[text.index('struct LHEParticle {'):text.index('std::vector<std::string> fields(')]
        pieces = []
        for start, stop in [('std::string trim(', 'bool eventStart('),
                            ('double number(', 'struct LHEParticle'),
                            ('std::vector<std::string> fields(', 'int integer('),
                            ('int integer(', 'LHEParticle particleRecord('),
                            ('std::string dipoleName(', 'void prepareSourceMatching(')]:
            pieces.append(text[text.index(start):text.index(stop)])
        cls.temporary = tempfile.TemporaryDirectory(prefix='macnlops-scale-format-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        source = cls.work / 'scale_format.cc'
        source.write_text('''#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
''' + '\n'.join(pieces) + particle_type + helpers + r'''
// Synthetic adapters exercise the actual serialization/mapping functions;
// they neither emulate a Pythia engine nor claim physical shower points.
struct InputEvent {
  int kind = 1;
  bool delta = false, hasHard = false;
  double showerScale = 100., scale = 100.;
  std::vector<double> productionScales;
  std::map<std::pair<int,int>,double> dipoleScales;
  std::string open = "<event>", close = "</event>", header, extra;
};
struct SyntheticParticle {
  LHEParticle p;
  int id() const {return p.id;} int col() const {return p.col;}
  int acol() const {return p.acol;}
  double px() const {return p.px;} double py() const {return p.py;}
  double pz() const {return p.pz;} double e() const {return p.e;}
};
struct FirstEmission {
  bool seen = true;
  double pt = 12., cutoffISR = .5, cutoffFSR = .6;
  std::vector<int> incoming, outgoing;
  std::vector<SyntheticParticle> snapshot;
  std::vector<NativeDipoleScale> continuationISR, continuationFSR;
};
''' + continuation + r'''
int main(int argc, char** argv) {
  try {
    if (argc == 2 && (std::string(argv[1]) == "continuation" ||
                      std::string(argv[1]) == "handoff-output")) {
      const LHEParticle dummy = {0, 0, 0, 0, 0, 0, 0., 0., 0., 0., 0., 0., 9.};
      std::vector<LHEParticle> particles(5, dummy);
      for (int i = 0; i < 5; ++i) {
        particles[i].id = i+1; particles[i].status = i < 2 ? -1 : 1;
        particles[i].col = 501+i; particles[i].e = 100.; particles[i].px = i;
      }
      InputEvent input;
      input.header = "5 7 -2.25 100 .0073 .118";
      input.extra = "<scales scalup_1_3='3.5'>\n</scales>\n#aMCatNLO 1 3 1\n"
                    "<rwgt><wgt id='nominal'>-2.25</wgt></rwgt>\n";
      FirstEmission hook; hook.incoming = {3,4}; hook.outgoing = {7,8,9};
      hook.snapshot.resize(10);
      const int indices[] = {3,4,7,8,9};
      for (int i = 0; i < 5; ++i) hook.snapshot[indices[i]].p = particles[i];
      hook.continuationISR = {{3,4,1,30.}, {4,3,2,7.}};
      hook.continuationFSR = {{7,3,0,.75}, {8,9,0,20.}, {9,8,0,0.}};
      auto state = continuationState(input, hook, particles);
      if (state.state != "emitted" || state.scale != 12. || state.legScales.at(1) != 12. ||
          state.legScales.at(2) != 7. || state.dipoleScales.at({3,1}) != .75 ||
          state.dipoleScales.at({4,5}) != 12. || state.dipoleScales.at({5,4}) != .25)
        throw std::runtime_error("residual caps were raised/lost or misindexed");
      if (std::string(argv[1]) == "handoff-output") {
        std::cout << std::scientific << std::setprecision(17);
        writeLHE(std::cout, input, particles, hook); return 0;
      }
      hook.continuationFSR.push_back({7,3,0,.8});
      bool conflict = false;
      try {continuationState(input, hook, particles);} catch (const std::runtime_error&) {conflict = true;}
      if (!conflict) throw std::runtime_error("unrepresentable duplicate caps were merged");
      hook.continuationFSR.pop_back();
      input.kind = 2; input.delta = true; input.showerScale = 40.;
      input.productionScales = {40.,30.,25.,10.,-1.};
      input.dipoleScales = {{{1,3},40.}, {{2,4},30.}, {{3,1},25.}, {{4,2},10.}};
      state = continuationState(input, hook, particles);
      if (state.state != "unshowered" || state.scale != 40. ||
          state.legScales.at(2) != 40. || state.dipoleScales.at({4,2}) != 10.)
        throw std::runtime_error("Delta H original limits were lost");
      input.showerScale = 100.; input.productionScales = {5.,5.,5.,5.,-1.};
      input.dipoleScales = {{{1,3},100.}, {{1,4},5.}, {{2,3},5.}, {{3,1},5.},
                           {{4,2},5.}, {{5,1},0.}};
      state = continuationState(input, hook, particles);
      for (const auto& leg : state.legScales)
        if (leg.second != 100.) throw std::runtime_error("Delta global scale was reduced to leg minima");
      if (state.dipoleScales.at({1,3}) != 100. || state.dipoleScales.at({5,1}) != 0.)
        throw std::runtime_error("Delta large/zero directed start changed");
      input.showerScale = 40.;
      input.delta = false;
      state = continuationState(input, hook, particles);
      if (state.state != "unshowered" || state.legScales.at(2) != 40. || !state.dipoleScales.empty())
        throw std::runtime_error("ordinary H original limit was lost");
      input.kind = 1; hook.seen = false;
      state = continuationState(input, hook, particles);
      if (state.state != "exhausted" || !(state.scale > 0. && state.scale < .5) ||
          state.legScales.at(1) != state.scale || !state.dipoleScales.empty())
        throw std::runtime_error("no-emission endpoint can restart above cutoff");
      hook.cutoffISR = 0.;
      bool invalid = false;
      try {continuationState(input, hook, particles);} catch (const std::runtime_error&) {invalid = true;}
      if (!invalid) throw std::runtime_error("zero exhausted anchor was accepted");
      std::cout << "PASS\n"; return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "format") {
      const double values[] = {std::nextafter(1., 2.), .12345678901234566,
        std::numeric_limits<double>::min(), 1.2345678901234567e128};
      const auto flags = std::cout.flags();
      const auto precision = std::cout.precision(), width = std::cout.width();
      const auto locale = std::cout.getloc();
      auto buffer = std::cout.rdbuf();
      std::string captured;
      {
        CaptureNativeScaleListing capture;
        for (double value : values)
          std::cout << std::fixed << std::setprecision(3) << std::setw(12) << value << '\n';
        captured = capture.str();
      }
      if (std::cout.flags() != flags || std::cout.precision() != precision ||
          std::cout.width() != width || std::cout.getloc() != locale || std::cout.rdbuf() != buffer)
        throw std::runtime_error("stream state was not restored");
      std::istringstream rows(captured); std::string token;
      for (double value : values) {
        if (!(rows >> token) || number(token) != value)
          throw std::runtime_error("round-trip precision was lost");
      }
      try {
        CaptureNativeScaleListing capture;
        std::cout << std::hex << std::setprecision(2) << std::setw(19) << 1.25;
        throw std::runtime_error("expected unwind control");
      } catch (const std::runtime_error&) {}
      if (std::cout.flags() != flags || std::cout.precision() != precision ||
          std::cout.width() != width || std::cout.getloc() != locale || std::cout.rdbuf() != buffer)
        throw std::runtime_error("stream state was not restored after exception");
      std::cout << "PASS\n";
      return 0;
    }
    if (argc != 3) throw std::runtime_error("bad parser control invocation");
    std::ifstream input(argv[2]);
    const std::string listing((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto rows = parseNativeDipoleScales(listing, std::string(argv[1]) == "isr");
    std::cout << std::setprecision(17);
    for (const auto& row : rows)
      std::cout << row.radiator << ' ' << row.recoiler << ' ' << row.side << ' ' << row.scale << '\n';
    return 0;
  } catch (const std::exception& error) {std::cerr << error.what(); return 1;}
}
''')
        cls.executable = cls.work / 'scale_format.exe'
        result = subprocess.run([compiler, '-std=c++11', '-O0', str(source), '-o',
                                 str(cls.executable)], capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    @staticmethod
    def listing(rows, isr=False):
        if isr:
            return ('\n --------  PYTHIA SimpleSpaceShower Dipole Listing  --------- \n'
                    '\n    i  syst  side   rad   rec       pTmax  col  chg  ME rec \n' +
                    '\n'.join(rows) + '\n\n --------  End PYTHIA SimpleSpaceShower Dipole Listing  -----\n')
        return ('\n --------  PYTHIA SimpleTimeShower Dipole Listing  -----------' + '-' * 64 +
                '-- \n \n    i    rad    rec       pTmax  col  chg  gam weak  oni'
                '   hv  isr  sys sysR type  MErec     mix  ord  spl  ~gR  jun   flex  pol\n' +
                '\n'.join(rows) + '\n\n --------  End PYTHIA SimpleTimeShower Dipole Listing  --------' +
                '-' * 65 + '\n')

    def parse_listing(self, text, isr=False):
        path = self.work / 'listing.txt'
        path.write_text(text)
        return subprocess.run([str(self.executable), 'isr' if isr else 'fsr', str(path)],
                              capture_output=True, text=True, timeout=5)

    def test_exact_numeric_facet_and_stream_state(self):
        result = subprocess.run([str(self.executable), 'format'], capture_output=True,
                                text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'PASS')

    def test_native_row_fields_and_small_residual_caps(self):
        time = self.listing(['0 7 4 .12345678901234566 1 0 0 0 0 0 2 0 0 0 0 0.0 0 0 0 0 1.0 0'])
        result = self.parse_listing(time)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split()[:3], ['7', '4', '0'])
        self.assertEqual(float(result.stdout.split()[3]), .12345678901234566)
        space = self.listing(['0 0 1 3 4 17.123456789012344 1 0 0 1',
                              '1 0 2 4 3 2.0000000000000004 -1 0 0 1'], True)
        result = self.parse_listing(space, True)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split() for line in result.stdout.splitlines()]
        self.assertEqual([row[:3] for row in rows], [['3', '4', '1'], ['4', '3', '2']])
        self.assertEqual(float(rows[1][3]), math.nextafter(2., 3.))

    def test_actual_continuation_mapping_and_exhausted_boundary(self):
        result = subprocess.run([str(self.executable), 'continuation'], capture_output=True,
                                text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'PASS')

    def test_actual_handoff_writer_preserves_weight_and_source_packets(self):
        result = subprocess.run([str(self.executable), 'handoff-output'], capture_output=True,
                                text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        header = result.stdout.splitlines()[1].split()
        self.assertEqual([header[i] for i in (0, 1, 2, 4, 5)], ['5', '7', '-2.25', '.0073', '.118'])
        self.assertEqual(float(header[3]), 12.)
        self.assertIn('state="emitted"', result.stdout)
        self.assertIn('source_scalup="1.00000000000000000e+02"', result.stdout)
        self.assertIn("<macnlops_source_scales scalup_1_3='3.5'>", result.stdout)
        self.assertIn('</macnlops_source_scales>', result.stdout)
        self.assertEqual(result.stdout.count('<scales '), 1)
        self.assertIn('scalup_3_1="7.50000000000000000e-01"', result.stdout)
        self.assertIn("<rwgt><wgt id='nominal'>-2.25</wgt></rwgt>", result.stdout)
        self.assertIn('#aMCatNLO 1 3 1\n', result.stdout)

    def test_malformed_native_listing_fails_closed(self):
        row = '0 7 4 .12345678901234566 1 0 0 0 0 0 2 0 0 0 0 0.0 0 0 0 0 1.0 0'
        valid = self.listing([row])
        controls = {
            'missing_footer': valid.split('End PYTHIA')[0],
            'bad_count': self.listing([row + ' 99']),
            'bad_integer': self.listing([row.replace('0 7 4', '0 7x 4', 1)]),
            'negative_scale': self.listing([row.replace('.12345678901234566', '-1')]),
            'nonfinite': self.listing([row.replace('.12345678901234566', 'NaN')]),
            'wrong_system': self.listing([row.replace('2 0 0 0 0 0.0', '2 1 0 0 0 0.0')]),
            'extra_qed': self.listing([row.replace('1 0 0 0 0 0 2', '1 0 1 0 0 0 2')]),
            'nonconsecutive': self.listing([row.replace('0 7 4', '1 7 4', 1)]),
            'lost_row': self.listing([row.replace('0 7 4', 'oops 7 4', 1)]),
            'negative_index': self.listing([row.replace('0 7 4', '-1 7 4', 1)]),
            'unused_column_type': self.listing([row.replace('0.0 0 0 0 0 1.0', '0.0 invalid 0 0 0 1.0')]),
            'trailing_garbage': valid + 'unexpected parser content\n',
            'duplicate_section': valid + valid,
        }
        for label, text in controls.items():
            with self.subTest(label=label):
                result = self.parse_listing(text)
                self.assertNotEqual(result.returncode, 0, result.stdout)


class TestHardPresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix = os.environ.get('PYTHIA8') or os.environ.get('PYTHIA8LOCATION')
        if not prefix:
            raise unittest.SkipTest('Set PYTHIA8 to run native hard-presentation checks')
        cls.prefix = Path(prefix).resolve()
        cls.tmp = tempfile.TemporaryDirectory(prefix='macnlops-hard-presentation-')
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.work = Path(cls.tmp.name)
        cls.environment = os.environ.copy()
        cls.pdf = cls.environment.get('MACNLOPS_TEST_PDF', 'LHAPDF6:MSTW2008nlo90cl/0')
        cls.executable = cls.work / 'one.exe'
        result = subprocess.run(['g++', '-std=c++11', '-O1',
            '-I' + str(cls.prefix / 'include'), str(SOURCE / 'MacNLOPS.cc'),
            '-o', str(cls.executable), '-L' + str(cls.prefix / 'lib'),
            '-Wl,-rpath,' + str(cls.prefix / 'lib'), '-lpythia8'],
            capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        parser = cls.work / 'parser.cc'
        parser.write_text('#define main unused_main\n#include "' +
            str(SOURCE / 'MacNLOPS.cc') + '"\n#undef main\n' + r'''
int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--native-import") {
      Pythia p;
      LHAupLHEF reader(const_cast<Info*>(&p.info), argv[2], nullptr, true, true);
      if (!reader.setInit()) throw std::runtime_error("Native LHE header import failed");
      while (reader.setEvent()) {
        std::cout << std::setprecision(17) << "{\"scale\":" << reader.scale() << ",\"legs\":[";
        for (int i = 1; i < reader.sizePart(); ++i) {
          if (i != 1) std::cout << ',';
          std::cout << reader.scale(i);
        }
        std::cout << "]}\n";
      }
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--setting") {
      Pythia p;
      for (const auto& key : qcdOnlyOff()) p.readString(key + " = off");
      p.readString(std::string(argv[2]) + " = on");
      validateQCDOnlySettings(p); return 0;
    }
    const bool matching = argc == 3 && (std::string(argv[1]) == "--delta-source" ||
                                       std::string(argv[1]) == "--ordinary-source");
    RawReader reader(argv[matching ? 2 : 1]); InputEvent input;
    while (reader.next(input)) {
      if (matching) {
        prepareSourceMatching(input, std::string(argv[1]) == "--delta-source");
        nativeProcessOrder(input.particles);
      } else if (argc == 3) {
        auto endpoint = input.particles;
        auto extra = endpoint.back(); extra.id = integer(argv[2]);
        endpoint.push_back(extra); validateQCDOnlyEndpoint(endpoint, input);
      }
      std::cout << input.hardUsed << " " << input.particles.size() << " "
                << input.hardParticles.size() << "\n";
    }
    return 0;
  } catch(const std::exception& e) { std::cerr << e.what(); return 1; }
}
''')
        cls.parser = cls.work / 'parser.exe'
        result = subprocess.run(['g++', '-std=c++11', '-O1',
            '-I' + str(cls.prefix / 'include'), str(parser), '-o', str(cls.parser),
            '-L' + str(cls.prefix / 'lib'), '-Wl,-rpath,' + str(cls.prefix / 'lib'),
            '-lpythia8'], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def run_native(self, label, document, extra=()):
        source = self.work / (label + '.input.lhe')
        source.write_text(HEADER + document + '</LesHouchesEvents>\n')
        original = source.read_bytes()
        output, metadata = (self.work / (label + suffix) for suffix in
                            ('.lhe', '.jsonl'))
        command = [str(self.executable), '--input', str(source),
            '--output', str(output), '--metadata', str(metadata), '--pdf', self.pdf,
            '--seed', '7349']
        result = subprocess.run(command + list(extra), env=self.environment,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(source.read_bytes(), original, 'Native import changed source bytes')
        return result, output, metadata, source

    def parse(self, text, label='parse'):
        path = self.work / (label + '.lhe')
        path.write_text(HEADER + text.rstrip() + '\n</LesHouchesEvents>\n')
        return subprocess.run([str(self.parser), str(path)], capture_output=True, text=True)

    def native_import(self, block, label):
        """Stock LHAupLHEF scale import only; no Pythia next() call."""
        path = self.work / (label + '.continuation.lhe')
        path.write_text(HEADER + block + '\n</LesHouchesEvents>\n')
        result = subprocess.run([str(self.parser), '--native-import', str(path)],
                                capture_output=True, text=True, env=self.environment)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_delta_native_scale_import_preserves_hard_leg_routing(self):
        controls = {}
        for resonance in (False, True):
            document = delta_event(resonance=resonance) + delta_event(2, resonance=resonance)
            label = 'delta_%s' % resonance
            result, output, metadata, source = self.run_native(
                label, document, ('--mcatnlo-delta',))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = [json.loads(line) for line in metadata.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row['source_matching'], 'delta')
                self.assertTrue(row['production_scales_from_lhef'])
                self.assertTrue(row['dipole_scales_from_lhef'])
                self.assertEqual(row['source_scalup'], .01)
                self.assertEqual(row['shower_scalup'], 40.)
                self.assertEqual(row['source_production_scales'], [40., 30., 25., 10.])
                self.assertEqual(row['source_dipole_scales'],
                    [[1, 3, 40.], [2, 4, 30.], [3, 1, 25.], [4, 2, 10.]])
                self.assertEqual(row['native_dipole_scales'], row['source_dipole_scales'])
                state = 'unshowered' if row['event_kind'] == 'H' else 'emitted' if row['emitted'] else 'exhausted'
                self.assertEqual(row['continuation_state'], state)
                self.assertEqual(row['continuation_cap_source'], {
                    'unshowered': 'source-imported-starting-scales',
                    'emitted': 'stock-pythia-8.318-public-dipole-list',
                    'exhausted': 'effective-shower-cutoffs'}[state])
                if row['emitted']:
                    self.assertLessEqual(row['pt_evol'], 40.)
            first_endpoints = blocks(output)
            for index, (old, endpoint, row) in enumerate(zip(blocks(source), first_endpoints, rows)):
                marker = assert_handoff(self, old, endpoint, row['continuation_state'])
                self.assertEqual(float(marker['scale']), row['continuation_scalup'])
                imported = self.native_import(endpoint, label + '_%d' % index)
                self.assertEqual(imported['scale'], float(marker['scale']))
                expected = {}
                for key, cap in active_scales(endpoint).items():
                    if cap <= 0.:  # native production-scale import ignores zero
                        continue
                    radiator = int(key.split('_')[1])
                    expected[radiator] = min(expected.get(radiator, cap), cap)
                self.assertEqual(imported['legs'], [expected[i] for i in range(1, len(imported['legs'])+1)])
                if row['emitted']:
                    self.assertEqual(float(marker['scale']), row['pt_evol'])
                if row['event_kind'] == 'H':
                    self.assertEqual(float(marker['scale']), row['shower_scalup'])
            # Original scale attributes are inert source provenance;
            # active endpoint scales use the new endpoint's leg indices.
            source_packets = re.findall(r'<scales\b.*?</scales>', document, re.S)
            context = re.search(r'<macnlops_delta_context\b.*?</macnlops_delta_context>',
                                document, re.S).group()
            text = output.read_text()
            inert = re.findall(r'<macnlops_source_scales\b.*?</macnlops_source_scales>', text, re.S)
            restored = [packet.replace('macnlops_source_scales', 'scales') for packet in inert]
            self.assertEqual(restored, source_packets)
            self.assertIn(context, text)
            controls[resonance] = rows
        reference = controls[False]
        for rows in controls.values():
            for row, control in zip(rows, reference):
                self.assertEqual(row['particles'], control['particles'])
                self.assertEqual(row['pythia_event'], control['pythia_event'])

    def test_delta_scale_routing_follows_native_ancestry_order(self):
        source = delta_event()
        before, hard = source.split('<macnlops_hard_event', 1)
        # Change the traversal order without changing any external momentum,
        # identity, colour or scale. Native order is now 1,2,4,3.
        hard = hard.replace('1 1 1 2 501 0 ', '1 1 2 1 501 0 ', 1)
        source = before + '<macnlops_hard_event' + hard
        result, _, metadata, _ = self.run_native(
            'delta_ancestry', source, extra=('--mcatnlo-delta',))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        row = json.loads(metadata.read_text())
        self.assertEqual(row['source_dipole_scales'],
            [[1, 3, 40.], [2, 4, 30.], [3, 1, 25.], [4, 2, 10.]])
        self.assertEqual(row['native_dipole_scales'],
            [[1, 4, 40.], [2, 3, 30.], [3, 2, 10.], [4, 1, 25.]])

    def test_delta_source_packets_fail_closed(self):
        valid = delta_event(2, resonance=True)
        controls = {
            'missing_scales': re.sub(r'<scales\b.*?</scales>\n', '', valid, flags=re.S),
            'missing_pair': valid.replace(" scalup_4_1='25'", ''),
            'nonexternal_index': valid.replace('scalup_1_4', 'scalup_1_3'),
            'disconnected_pair': valid.replace('scalup_1_4', 'scalup_1_5'),
            'negative_scale': valid.replace("scalup_1_4='40'", "scalup_1_4='-1'"),
            'nonfinite_scale': valid.replace("scalup_1_4='40'", "scalup_1_4='NaN'"),
            'duplicate_pair': valid.replace("scalup_1_4='40'", "scalup_1_4='40' scalup_1_4='40'"),
            'missing_context': re.sub(r'<macnlops_delta_context\b.*?</macnlops_delta_context>\n',
                                      '', valid, flags=re.S),
            'missing_registry': valid.replace(' registry="' + 'a' * 64 + '"', ''),
            'bad_registry': valid.replace('a' * 64, 'A' * 64),
            'context_owner': valid.replace('owner="1"', 'owner="2"'),
            'context_probability': valid.replace('1 1 1 .75', '1 1 1 1.01'),
            'context_duplicate': valid.replace('1 1 1 .75', '1 1 1 .75\n1 1 1 .75'),
            'derived_endpoint': valid.replace('#aMCatNLO',
                '<macnlops_endpoint version="1" origin="shower" stage="first"/>\n#aMCatNLO'),
            'ordinary_annotation': valid.replace('source="delta"', 'source="ordinary"'),
        }
        for label, document in controls.items():
            with self.subTest(label=label):
                path = self.work / (label + '.delta.lhe')
                path.write_text(HEADER + document + '</LesHouchesEvents>\n')
                result = subprocess.run([str(self.parser), '--delta-source', str(path)],
                    capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0, result.stdout)
        path = self.work / 'valid.delta.lhe'
        for document in (delta_event(), valid, valid.replace('1 1 1 .75\n', '')):
            path.write_text(HEADER + document + '</LesHouchesEvents>\n')
            result = subprocess.run([str(self.parser), '--delta-source', str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([str(self.parser), '--ordinary-source', str(path)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

    def test_identical_annotated_hard_presentation_before_any_veto(self):
        # Both comparison branches consume the same annotated hard point.
        # Only the untouched ordinary transport coordinates differ here.
        specs = [(1, 100., 1.)] * 4 + [(2, 100., -2.), (1, .01, 3.)]
        shuffled = ''.join(event(*spec, shuffled=True, resonance=True) for spec in specs)
        canonical = ''.join(event(*spec, shuffled=False) for spec in specs)
        a = self.run_native('transport', shuffled)
        b = self.run_native('canonical', canonical)
        self.assertEqual(a[0].returncode, 0, a[0].stdout + a[0].stderr)
        self.assertEqual(b[0].returncode, 0, b[0].stdout + b[0].stderr)
        self.assertEqual(a[1].read_bytes(), b[1].read_bytes())
        rows = [json.loads(line) for line in a[2].read_text().splitlines()]
        controls = [json.loads(line) for line in b[2].read_text().splitlines()]
        self.assertTrue(any(r['emitted'] for r in rows[:4]))
        self.assertFalse(rows[-1]['emitted'])
        for r, control in zip(rows, controls):
            self.assertTrue(r['hard_annotation_used'])
            self.assertTrue(r['qcd_only'])
            self.assertEqual(r['hard_systems'], 1)
            self.assertEqual(r['source_partition']['partition'], 'sampled')
            self.assertEqual(r['source_partition']['group_size'], 0)
            self.assertEqual(r['source_particles'], control['source_particles'])
            self.assertEqual(r['particles'], control['particles'])
            self.assertEqual(r['pythia_event'], control['pythia_event'])
            self.assertEqual(r['source_weight'], control['source_weight'])
            self.assertNotEqual(r['original_lhe_particles'], control['original_lhe_particles'])
        endpoints = blocks(a[1])
        self.assertIn('origin="hard_annotation" stage="first"', endpoints[-1])
        self.assertIn('origin="hard_annotation" stage="first"', endpoints[-2])
        from madgraph.various.macnlops import Event
        for index in (-2, -1):
            point = Event(endpoints[index])
            self.assertEqual([p[1] for p in point.external()],
                             [(100., 0., 0., 100.), (100., 0., 0., -100.),
                              (100., 100., 0., 0.), (100., -100., 0., 0.)])

    def test_emitted_endpoint_keeps_real_body_and_requires_continuation_consumer(self):
        result, output, metadata, _ = self.run_native('emitted', event())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        row = json.loads(metadata.read_text().splitlines()[0])
        self.assertTrue(row['emitted'])
        endpoint = blocks(output)[0]
        parsed = self.parse(endpoint)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        self.assertEqual(parsed.stdout.strip(), '0 5 4')
        marker = assert_handoff(self, event(), endpoint, 'emitted')
        self.assertEqual(float(marker['scale']), row['pt_evol'])
        # The proposal executable accepts fresh source points. A stock
        # continuation consumer uses this active matrix; treating it as a
        # fresh matching source would attempt another matching decision.
        rerun = self.run_native('reimport', endpoint + '\n')
        self.assertNotEqual(rerun[0].returncode, 0)
        self.assertIn('do not repeat first-emission matching', rerun[0].stderr)

    def test_malformed_or_conflicting_annotation_rejected(self):
        valid = event()
        controls = {
            'truncated': valid.replace('</macnlops_hard_event>', ''),
            'duplicate': valid.replace('#aMCatNLO', hard_block() + '#aMCatNLO'),
            'version': valid.replace('version="1"', 'version="2"'),
            'frame': valid.replace('frame="lab"', 'frame="cm"'),
            'unknown_attribute': valid.replace('frame="lab"', 'frame="lab" extra="x"'),
            'kind': valid.replace('4 1 1 1 1 1 0 1 1', '4 2 1 1 1 1 0 1 1'),
            'nup': valid.replace('4 1 1 1 1 1 0 1 1', '501 1 1 1 1 1 0 1 1'),
            'float_metadata': valid.replace('4 1 1 1 1 1 0 1 1', '4 1 1.0 1 1 1 0 1 1'),
            'negative_metadata': valid.replace('4 1 1 1 1 1 0 1 1', '4 1 -1 1 1 1 0 1 1'),
            'identity': valid.replace(hard_block(), hard_block(records=particles(pdg=1))),
            'status': valid.replace(hard_block(), hard_block(records=[
                p.replace('11 1 ', '11 2 ') if p.startswith('11 ') else p for p in particles()])),
            'global_flags': valid.replace('partition="sampled"', 'partition="global_sum"'),
            'duplicate_marker': valid.replace('#aMCatNLO',
                '<macnlops_endpoint version="1" origin="shower" stage="first"/>\n' * 2 + '#aMCatNLO'),
            'hard_full': valid.replace('#aMCatNLO',
                '<macnlops_endpoint version="1" origin="hard_annotation" stage="full"/>\n#aMCatNLO'),
            'marker_without_hard': event(annotation=False).replace('#aMCatNLO',
                '<macnlops_endpoint version="1" origin="shower" stage="first"/>\n#aMCatNLO'),
            'nonfinite_hard': valid.replace(hard_block(), hard_block().replace('100 100 0 0 9', '100 NaN 0 0 9')),
        }
        for label, text in controls.items():
            with self.subTest(label=label):
                result = self.parse(text, label)
                self.assertNotEqual(result.returncode, 0, result.stdout)
        for text in [valid, event(partition='global_sum', group_size=2)]:
            result = self.parse(text, 'valid')
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_transport_and_wrong_model_masses_fail_closed(self):
        for label, source in [('unannotated_transport', event(annotation=False)),
                              ('wrong_annotation_mass', event(hard=particles(.5)))]:
            with self.subTest(label=label):
                result = self.run_native(label, source)
                self.assertNotEqual(result[0].returncode, 0)
                self.assertIn('Noncanonical hard-model mass', result[0].stderr)
        result = self.run_native('canonical_legacy', event(shuffled=False, annotation=False))
        self.assertEqual(result[0].returncode, 0, result[0].stdout + result[0].stderr)

    def test_qcd_only_particle_and_runtime_setting_controls(self):
        source = self.work / 'qcd_endpoint.lhe'
        source.write_text(HEADER + event() + '</LesHouchesEvents>\n')
        for pdg in (22, 211, 24, 11):
            with self.subTest(pdg=pdg):
                result = subprocess.run([str(self.parser), str(source), str(pdg)],
                    capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
        result = subprocess.run([str(self.parser), str(source), '21'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for key in ('PartonLevel:MPI', 'PartonLevel:Remnants', 'HadronLevel:Hadronize',
                    'HadronLevel:Decay', 'ProcessLevel:resonanceDecays',
                    'ColourReconnection:reconnect', 'TimeShower:QEDshowerByOther',
                    'SpaceShower:weakShower'):
            with self.subTest(key=key):
                result = subprocess.run([str(self.parser), '--setting', key],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Non-QCD shower setting enabled', result.stderr)

    def test_model_top_and_four_flavour_bottom_masses_are_retained(self):
        cases = [(6, 173., 172.5, 5), (5, 4.7, 4.8, 4)]
        for pdg, hard_mass, mc_mass, nf in cases:
            with self.subTest(pdg=pdg, nf=nf):
                def pair(incoming_mass, final_mass):
                    pin = math.sqrt(250.**2 - incoming_mass**2)
                    pout = math.sqrt(250.**2 - final_mass**2)
                    return [
                        f'2 -1 0 0 501 0 0 0 {pin:.17g} 250 {incoming_mass:.17g} 0 9',
                        f'-2 -1 0 0 0 502 0 0 {-pin:.17g} 250 {incoming_mass:.17g} 0 9',
                        f'{pdg} 1 1 2 501 0 {pout:.17g} 0 0 250 {final_mass:.17g} 0 9',
                        f'{-pdg} 1 1 2 0 502 {-pout:.17g} 0 0 250 {final_mass:.17g} 0 9']
                settings = self.work / ('mass%d.cmnd' % pdg)
                settings.write_text('%d:m0 = %.17g\n' % (pdg, hard_mass))
                extra = ['--settings', str(settings), '--flavour-count', str(nf)]
                if nf == 4:
                    extra += ['--pdf', os.environ.get('MACNLOPS_TEST_PDF_4F',
                              'LHAPDF6:MSTW2008nlo68cl_nf4/0')]
                hard = pair(0., hard_mass)
                a = self.run_native('heavy%d_transport' % pdg,
                    event(2, .01, hard=hard, records=pair(.5, mc_mass)), extra)
                b = self.run_native('heavy%d_canonical' % pdg,
                    event(2, .01, hard=hard, records=hard), extra)
                for result in (a, b):
                    self.assertEqual(result[0].returncode, 0,
                                     result[0].stdout + result[0].stderr)
                self.assertEqual(a[1].read_bytes(), b[1].read_bytes())
                row = json.loads(a[2].read_text().splitlines()[0])
                self.assertEqual(row['flavour_count'], nf)
                self.assertEqual(row['n_quark_in'], nf)
                self.assertEqual(row['n_gluon_to_quark'], nf)
                self.assertEqual(row['alpha_s_nfmax'], 5)
                self.assertEqual([p['mass'] for p in row['source_particles']],
                                 [0., 0., hard_mass, hard_mass])
                self.assertEqual([p['mass'] for p in row['original_lhe_particles']],
                                 [.5, .5, mc_mass, mc_mass])


if __name__ == '__main__':
    unittest.main()
