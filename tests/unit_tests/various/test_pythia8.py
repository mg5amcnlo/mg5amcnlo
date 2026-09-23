"""Exercise the Pythia shower input and analysis reader without Pythia."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from madgraph.various.shower_card import ShowerCard


ROOT = Path(__file__).resolve().parents[3]
MCATNLO = ROOT / 'Template/NLO/MCatNLO'


@unittest.skipUnless(shutil.which('g++'), 'requires a C++ compiler')
class TestPythiaWeightReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(prefix='mg5_py8_reader_')
        cls.addClassCleanup(cls.work.cleanup)
        work = Path(cls.work.name)
        source = work / 'reader.cc'
        source.write_text(r'''
#include "LHEFRead.h"
extern "C" int get_wgts_info_len(void) { return 80; }
int main(int argc, char** argv) {
  MyReader reader(argv[1]);
  char labels[1024][wgts_info_len_used];
  int count = 0;
  reader.lhef_read_wgtsinfo_(count, labels);
  std::cout << "count " << count << '\n';
  double weights[1024] = {0.};
  for (int event = 0; event < 2; ++event) {
    reader.lhef_read_wgts_(weights);
    std::cout << "event " << event << ' ' << weights[1] << '\n';
  }
}
''')
        cls.executable = work / 'reader'
        subprocess.run([shutil.which('g++'), '-O0', '-I',
                        str(MCATNLO / 'include'), str(source), '-o',
                        str(cls.executable)], check=True, capture_output=True)

    def read_events(self, reweight, jwgtinfo=0, header=True):
        weights = ("<initrwgt>\n<weightgroup name='test'>\n"
                   "<weight id='1001'> scale variation </weight>\n"
                   "</weightgroup>\n</initrwgt>\n") if reweight else ''
        contents = '<LesHouchesEvents version="3.0">\n'
        if header:
            contents += '<header>\n' + weights + '</header>\n'
        contents += ('<init>\n2212 2212 6500 6500 0 0 0 0 -4 1\n'
                     '1 0 1 1\n</init>\n')
        for value in (2.5, -3.25):
            contents += ('<event>\n1 1 1 10 0.01 0.1\n'
                         '25 1 0 0 0 0 0 0 0 125 125 0 9\n'
                         '#aMCatNLO 1 0 0 0 0 0 0 %d 0 0 0 0 0 0 0\n'
                         % jwgtinfo)
            if reweight:
                contents += "<rwgt>\n<wgt id='1001'> %s </wgt>\n</rwgt>\n" % value
            contents += ("<scales scalup_1_2='20' scalup_2_1='30'>\n"
                         '</scales>\n</event>\n')
        contents += '</LesHouchesEvents>\n'
        path = Path(self.work.name) / 'events.lhe'
        path.write_text(contents)
        result = subprocess.run([str(self.executable), str(path)],
                                text=True, capture_output=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_delta_events_without_reweighting(self):
        self.assertEqual(self.read_events(False),
                         'count 1\nevent 0 0\nevent 1 0\n')

    def test_events_without_header(self):
        self.assertEqual(self.read_events(False, header=False),
                         'count 1\nevent 0 0\nevent 1 0\n')

    def test_reweighting_with_both_supported_weight_formats(self):
        for jwgtinfo in (0, 9):
            with self.subTest(jwgtinfo=jwgtinfo):
                self.assertEqual(self.read_events(True, jwgtinfo),
                                 'count 2\nevent 0 2.5\nevent 1 -3.25\n')


@unittest.skipUnless(shutil.which('bash'), 'requires bash')
class TestPythiaDeltaSteering(unittest.TestCase):
    def steering(self, delta, dipole_setting):
        with tempfile.TemporaryDirectory(prefix='mg5_py8_steering_') as tmp:
            work = Path(tmp)
            (work / 'xmldoc').mkdir()
            (work / 'xmldoc/BeamParameters.xml').write_text(
                '<flag name="Beams:setDipoleShowerStartingScalesFromLHEF" '
                'default="off"/>\n' if dipole_setting else '<chapter/>\n')
            card = ShowerCard()
            card.testing = True
            (work / 'shower.dat').write_text(card.write_card('PYTHIA8', ''))
            # Exercise the complete production steering function. Compilation
            # is unnecessary, and the default negative Lambda never enables
            # its optional override (the only use of bc in this function).
            command = r'''
source "$1"
source "$2"
function bc { cat >/dev/null; echo 0; }
thisdir="$PWD"
MCMODE=PYTHIA8
PY8VER=8.3
UsedPdfLib=THISLIB
BEAM1=1
BEAM2=1
ICKKW=0
runMCMadFKS
'''
            result = subprocess.run(
                [shutil.which('bash'), '-c', command, 'test',
                 str(MCATNLO / 'Scripts/MCatNLO_MadFKS_PYTHIA8.Script'),
                 str(work / 'shower.dat')], cwd=work,
                env=dict(os.environ, DELTA='ON' if delta else 'OFF'),
                text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            return (work / 'Pythia8.cmd').read_text()

    def test_delta_keeps_particle_and_dipole_scales(self):
        result = self.steering(True, True)
        self.assertIn('Beams:setProductionScalesFromLHEF = on', result)
        self.assertIn('Beams:setDipoleShowerStartingScalesFromLHEF = on', result)

    def test_legacy_pythia_has_no_unknown_setting(self):
        result = self.steering(True, False)
        self.assertIn('Beams:setProductionScalesFromLHEF = on', result)
        self.assertNotIn('Beams:setDipoleShowerStartingScalesFromLHEF', result)

    def test_ordinary_mcatnlo_keeps_scalar_scale(self):
        result = self.steering(False, True)
        self.assertNotIn('Beams:setProductionScalesFromLHEF', result)
        self.assertNotIn('Beams:setDipoleShowerStartingScalesFromLHEF', result)
