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
  if (argc > 2) std::cout << "label " << std::string(labels[1], 80) << '\n';
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

    def read_events(self, reweight, jwgtinfo=0, header=True,
                    label='scale variation', show_label=False):
        weights = ("<initrwgt>\n<weightgroup name='test'>\n"
                   f"<weight id='1001'> {label} </weight>\n"
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
        result = subprocess.run([str(self.executable), str(path)] +
                                (['labels'] if show_label else []),
                                text=True, capture_output=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_delta_events_without_reweighting(self):
        self.assertEqual(self.read_events(False),
                         'count 1\nevent 0 0\nevent 1 0\n')

    def test_events_without_header(self):
        self.assertEqual(self.read_events(False, header=False),
                         'count 1\nevent 0 0\nevent 1 0\n')

    def test_reweighting_with_supported_weight_formats(self):
        for jwgtinfo in (0, 9, -9):
            with self.subTest(jwgtinfo=jwgtinfo):
                self.assertEqual(self.read_events(True, jwgtinfo),
                                 'count 2\nevent 0 2.5\nevent 1 -3.25\n')

    def test_inclusive_scale_labels_are_recognised_and_order_tags_retained(self):
        for tag in (0, 400, 402):
            label = f'tag= {tag} dyn= -1 muR=0.20000E+01 muF=0.10000E+01'
            result = self.read_events(True, -9, label=label, show_label=True)
            output = next(line for line in result.splitlines() if line.startswith('label '))
            self.assertIn('dyn= -1 muR=0.20000E+01 muF=0.10000E+01', output)
            self.assertEqual('tag=' in output, tag != 0)


@unittest.skipUnless(shutil.which('bash'), 'requires bash')
class TestPythiaDeltaSteering(unittest.TestCase):
    """Starting scales of the Pythia8 command file written by MG5_aMC."""

    def steering(self, delta, dipole_setting):
        from madgraph.various import mcatnlo_shower

        class Installation(object):
            version = '8.312' if dipole_setting else '8.244'
            path = '/fake'
            def has_setting(self, name):
                return dipole_setting or \
                    name != 'Beams:setDipoleShowerStartingScalesFromLHEF'
            def has_lhapdf6(self):
                return True

        contract = mcatnlo_shower.MatchingContract(parton_shower='PYTHIA8',
                                                   mcatnlo_delta=delta)
        run = {'nevents': 10, 'nevents_total': 10, 'event_norm': 'average',
               'mc_masses': {5: 4.8}, 'pdf': ('internal', None, False), 'seed': 1}
        return mcatnlo_shower.pythia8_settings(
            contract, ShowerCard(), run, Installation()).render()

    def test_delta_keeps_particle_and_dipole_scales(self):
        result = self.steering(True, True)
        self.assertIn('Beams:setProductionScalesFromLHEF = on', result)
        self.assertIn('Beams:setDipoleShowerStartingScalesFromLHEF = on', result)

    def test_legacy_pythia_is_refused_for_delta(self):
        # Without per-dipole starting scales MC@NLO-Delta cannot be showered
        # consistently: this is an error, no longer a silent downgrade.
        from madgraph.various import mcatnlo_shower
        with self.assertRaises(mcatnlo_shower.ShowerInterfaceError):
            self.steering(True, False)

    def test_ordinary_mcatnlo_keeps_scalar_scale(self):
        result = self.steering(False, True)
        self.assertIn('Beams:setProductionScalesFromLHEF = off', result)
        self.assertNotIn('Beams:setDipoleShowerStartingScalesFromLHEF', result)
        result = self.steering(False, False)
        self.assertIn('Beams:setProductionScalesFromLHEF = off', result)

