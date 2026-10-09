"""Compiled hard-system QED reference-partner regressions.

Fixtures exercise the preference classes of SimpleTimeShower::setupQEDdip
and the global-II pair of SimpleSpaceShower::prepare. These partners set
the local evolution ceiling even when radiation uses a global recoil map.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


def leg(pdg, charge, p, mass=0.):
    return (pdg, charge, mass, *p)


BEAMS = [leg(21, 0., (100., 0., 0., 100.)),
         leg(21, 0., (100., 0., 0., -100.))]


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestPythia8QEDDipoles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_qed_dipoles_')
        cls.binary = Path(cls.tempdir.name) / 'check_qed_dipoles'
        subprocess.run([
            'gfortran', '-O0', '-g', '-fno-automatic', '-fcheck=all',
            '-ffpe-trap=invalid,zero,overflow',
            '-J', cls.tempdir.name,
            str(ROOT / 'Template/NLO/SubProcesses/qed_shower_support.f90'),
            str(ROOT / 'tests/input_files/check_pythia8_qed_dipoles.f90'),
            '-o', str(cls.binary)], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def run_event(self, event, beams=True, nincoming=2):
        data = '{} {} {}\n'.format(len(event), nincoming, int(beams))
        data += ''.join(' '.join(map(str, item))+'\n' for item in event)
        result = subprocess.run([str(self.binary)], input=data, check=True,
                                capture_output=True, text=True)
        values = list(map(int, result.stdout.split()))
        return values[0], values[1:]

    def test_colourless_leptons_and_directed_isr_pairs(self):
        event = [leg(2, 2/3, (100., 0., 0., 100.)),
                 leg(-2, -2/3, (100., 0., 0., -100.)),
                 leg(11, -1., (100., 100., 0., 0.)),
                 leg(-11, 1., (100., -100., 0., 0.))]
        self.assertEqual(self.run_event(event), (0, [2, 1, 4, 3]))
        # Initial photons also evolve against a neutral opposite beam.
        event[0] = leg(22, 0., event[0][3:])
        event[1] = BEAMS[1]
        self.assertEqual(self.run_event(event), (0, [2, 0, 4, 3]))

    def test_flavour_precedes_distance_and_beam_recoil_is_optional(self):
        event = [leg(11, -1., (100., 0., 0., 100.)), BEAMS[1],
                 leg(11, -1., (20., 0., 0., -20.)),
                 leg(-13, 1., (10., 0., 0., -10.)),
                 leg(-11, 1., (50., 0., 0., 50.))]
        # The oppositely directed e+ has invariant distance 2000, versus
        # 4000 for the initial electron. A collinear muon is not preferred.
        self.assertEqual(self.run_event(event)[1][2], 5)
        event[4] = leg(-11, 1., (150., 0., 0., 150.))
        self.assertEqual(self.run_event(event)[1][2], 1)
        self.assertEqual(self.run_event(event, beams=False)[1][2], 5)

    def test_charge_squared_weight_and_neutral_fallback(self):
        event = BEAMS + [leg(22, 0., (30., 30., 0., 0.)),
                         leg(1, -1/3, (10., 0., 10., 0.)),
                         leg(2, 2/3, (20., 0., -20., 0.)),
                         leg(21, 0., (1., 1., 0., 0.))]
        # Distances 300 and 600 become 2700 and 1350 after charge weights;
        # the collinear neutral spectator is considered only as a fallback.
        self.assertEqual(self.run_event(event)[1][2], 5)
        event[3] = leg(12, 0., event[3][3:])
        event[4] = leg(-12, 0., event[4][3:])
        self.assertEqual(self.run_event(event), (0, [0, 0, 6, 0, 0, 0]))

    def test_photon_flavour_priority_and_stable_ties(self):
        event = [leg(22, 0., (100., 0., 0., 100.)), BEAMS[1],
                 leg(22, 0., (20., 20., 0., 0.)),
                 leg(11, -1., (10., 10., 0., 0.)),
                 leg(13, -1., (10., 10., 0., 0.))]
        self.assertEqual(self.run_event(event)[1][2], 1)
        self.assertEqual(self.run_event(event, beams=False)[1][2], 4)

    def test_mass_subtracted_distance(self):
        event = BEAMS + [leg(15, -1., (5., 0., 0., 0.), mass=5.),
                         leg(-15, 1., (10., 0., 0., 6.), mass=8.),
                         leg(-15, 1., (4., 0., 0., 4.))]
        # p.p alone favours leg 5 (20 < 50), but subtracting m*m gives
        # invariant kinetic distances 20 versus 10 and selects leg 4.
        self.assertEqual(self.run_event(event)[1][2], 4)

    def test_charged_w_resonances_have_final_state_recoil_partners(self):
        event = BEAMS + [leg(24, 1., (100., 60., 0., 0.), mass=80.),
                         leg(-24, -1., (100., -60., 0., 0.), mass=80.)]
        self.assertEqual(self.run_event(event), (0, [0, 0, 4, 3]))
        # Without an opposite-flavour W, use the charged beam. This
        # reference dipole remains necessary with global recoil enabled.
        event[0] = leg(2, 2/3, event[0][3:])
        event[3] = leg(23, 0., event[3][3:], mass=80.)
        self.assertEqual(self.run_event(event), (0, [2, 0, 1, 0]))
        self.assertEqual(self.run_event(event, beams=False), (0, [2, 0, 4, 0]))

    def test_w_support_does_not_enable_incoming_w_evolution(self):
        event = [leg(24, 1., (100., 0., 0., 60.), mass=80.), BEAMS[1],
                 leg(-24, -1., (100., -60., 0., 0.), mass=80.),
                 leg(23, 0., (100., 60., 0., 0.), mass=80.)]
        self.assertEqual(self.run_event(event), (0, [0, 0, 1, 0]))

    def test_no_partner_and_unsupported_incoming_clear_result(self):
        event = BEAMS + [leg(22, 0., (100., 100., 0., 0.))]
        self.assertEqual(self.run_event(event), (3, [0, 0, 0]))
        event[0] = leg(11, -1., event[0][3:])
        self.assertEqual(self.run_event(event, nincoming=1), (1, [0, 0, 0]))
        # Initial charge is eligible ISR, but neutral final fallback may
        # not use an initial leg when allowBeamRecoil is off.
        self.assertEqual(self.run_event(event, beams=False), (3, [0, 0, 0]))

    def test_nonfinite_inputs_and_negative_mass_are_rejected(self):
        event = BEAMS + [leg(11, -1., (20., 20., 0., 0.), mass=-1.),
                         leg(-11, 1., (20., -20., 0., 0.))]
        self.assertEqual(self.run_event(event), (2, [0, 0, 0, 0]))
        event[2] = leg(11, -1., (float('nan'), 20., 0., 0.))
        self.assertEqual(self.run_event(event), (2, [0, 0, 0, 0]))


if __name__ == '__main__':
    unittest.main()
