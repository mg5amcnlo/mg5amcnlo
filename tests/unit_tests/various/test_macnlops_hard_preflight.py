"""Compile the hard-input check alone; no Pythia library or evolution."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('g++'), 'requires g++')
class TestHardPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='macnlops-preflight-')
        cls.addClassCleanup(cls.temporary.cleanup)
        work = Path(cls.temporary.name)
        text = (ROOT / 'Template/NLO/MCatNLO/srcPythia8/MacNLOPS.h').read_text()
        particles = text[text.index('struct LHEParticle {'):text.index('std::vector<std::string> fields(')]
        check = text[text.index('void validateHardInput('):text.index('// A second reader preserves')]
        source = work / 'check.cc'
        source.write_text('''#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
''' + particles + '''
struct InputEvent {
 bool hasHard=false, hardUsed=false, hasEndpoint=false;
 std::vector<LHEParticle> particles,hardParticles;
};
// The production preflight only queries canonical quark/lepton masses.
// This adapter supplies the massless hard model without initializing an engine.
struct Pythia {struct {double m0(int) const {return 0.;}} particleData;};
''' + check + '''
int main(){
 InputEvent input;int n,annotated;std::cin>>annotated>>n;
 input.hasHard=input.hardUsed=annotated;
 for(int i=0;i<n;++i){LHEParticle p{};
  std::cin>>p.id>>p.status>>p.e>>p.px>>p.py>>p.pz>>p.m;
  input.particles.push_back(p);
 }
 try{validateHardInput(input,Pythia{});std::cout<<"PASS\\n";return 0;}
 catch(const std::exception& e){std::cerr<<e.what()<<'\\n';return 1;}
}
''')
        cls.executable = work / 'check'
        result = subprocess.run(['g++', '-O0', '-std=c++11', str(source), '-o', str(cls.executable)],
                                text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.points = json.loads((ROOT / 'tests/input_files/macnlops_hard_preflight_20261025.json').read_text())['points']

    def invoke(self, point, annotated=True):
        rows = ['%d %d' % (annotated, len(point['particles']))]
        rows.extend('%d %d %s' % (p['id'], p['status'],
                    ' '.join('%.17g' % x for x in (*p['p'], p['mass'])))
                    for p in point['particles'])
        return subprocess.run([str(self.executable)], input='\n'.join(rows) + '\n',
                              capture_output=True, text=True)

    def test_saved_boosted_annotations_pass_without_altering_momenta(self):
        for point in self.points:
            with self.subTest(sequence=point['sequence']):
                result = self.invoke(point)
                self.assertEqual(result.returncode, 0, result.stderr)
                # The same precision in a plain LHE still fails its strict check.
                result = self.invoke(point, annotated=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('off its declared mass shell', result.stderr)

    def test_physical_offshellness_and_mass_changes_still_fail(self):
        for point in self.points:
            changed = copy.deepcopy(point)
            changed['particles'][-1]['p'][0] += 1.e-4
            result = self.invoke(changed)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('off its declared mass shell', result.stderr)
            changed = copy.deepcopy(point)
            changed['particles'][-1]['mass'] = 1.2
            result = self.invoke(changed)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('off its declared mass shell', result.stderr)


if __name__ == '__main__':
    unittest.main()
