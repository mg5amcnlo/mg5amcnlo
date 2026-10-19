"""Independent kinematic checks for the massive-pair residual carrier."""
from pathlib import Path
import itertools
import shutil
import subprocess
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None


ROOT = Path(__file__).resolve().parents[3]


def reference_factor(p, masses, incoming):
    """Direct NumPy invariant formula, independent of native scaling guards."""
    result = 1.
    for i, j in itertools.combinations(range(len(masses)), 2):
        if incoming[i] or incoming[j] or masses[i] <= 0 or masses[j] <= 0:
            continue
        dot = p[i, 0]*p[j, 0] - p[i, 1:]@p[j, 1:]
        result += dot / np.sqrt((dot-masses[i]*masses[j]) *
                                (dot+masses[i]*masses[j]))
    return result


@unittest.skipUnless(shutil.which('gfortran') and np is not None,
                     'requires gfortran and numpy')
class TestVirtualEnvelopeCarrier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix='mg5_velocity_carrier_')
        cls.executable = Path(cls.build.name)/'check'
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-frecursive', '-fcheck=all',
            '-ffpe-trap=invalid,zero,overflow',
            str(ROOT/'Template/NLO/SubProcesses/virtual_envelope_carrier.f90'),
            str(ROOT/'tests/input_files/check_virtual_envelope_carrier.f90'),
            '-o', str(cls.executable)], cwd=cls.build.name,
            text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout+result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def evaluate(self, cases):
        lines = [str(len(cases))]
        for p, mass, incoming in cases:
            lines.append(str(len(mass)))
            lines.extend(' '.join(map(str, [m, int(flag), *leg]))
                         for leg, m, flag in zip(p, mass, incoming))
        result = subprocess.run([str(self.executable)],
                                input='\n'.join(lines)+'\n', text=True,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return [(row.split()[0] == 'T', float(row.split()[1]))
                for row in result.stdout.splitlines()]

    @staticmethod
    def pair(beta=.4, mass=173.):
        energy = mass/np.sqrt(1-beta**2)
        p = np.array([[energy, 0, 0, energy], [energy, 0, 0, -energy],
                      [energy, energy*beta, 0, 0],
                      [energy, -energy*beta, 0, 0]])
        return p, np.array([0., 0., mass, mass]), np.array([1, 1, 0, 0], bool)

    def test_equal_mass_pair_matches_relative_velocity(self):
        betas = [.0001, .01, .2, .8, .99]
        results = self.evaluate([self.pair(beta) for beta in betas])
        for beta, (valid, factor) in zip(betas, results):
            self.assertTrue(valid)
            expected = 1 + (1+beta**2)/(2*beta)
            self.assertAlmostEqual(factor/expected, 1., delta=2e-8)

    def test_generic_one_or_two_incoming_and_unequal_masses_python_parity(self):
        rng = np.random.default_rng(4271)
        cases = []
        for n_incoming in (1, 2):
            for _ in range(60):
                masses = np.r_[np.zeros(n_incoming), rng.uniform(.2, 5., 3), 0.]
                vectors = rng.normal(size=(len(masses), 3))*3
                p = np.column_stack((np.sqrt(np.sum(vectors**2, axis=1)+masses**2), vectors))
                incoming = np.arange(len(masses)) < n_incoming
                cases.append((p, masses, incoming))
        for case, (valid, factor) in zip(cases, self.evaluate(cases)):
            self.assertTrue(valid)
            self.assertAlmostEqual(factor/reference_factor(*case), 1., delta=2e-13)

    def test_scale_boost_rotation_and_leg_permutation_invariance(self):
        p, mass, incoming = self.pair(.31)
        expected = reference_factor(p, mass, incoming)
        cases = [(p*scale, mass*scale, incoming) for scale in (1e-100, 1., 1e100)]
        beta = np.array([.13, -.21, .37])
        b2 = beta@beta
        gamma = 1/np.sqrt(1-b2)
        bp = p[:, 1:]@beta
        boosted = p.copy()
        boosted[:, 0] = gamma*(p[:, 0]-bp)
        boosted[:, 1:] += (((gamma-1)*bp/b2-gamma*p[:, 0])[:, None]*beta)
        cases.append((boosted, mass, incoming))
        angle = .817
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        rotated = p.copy()
        rotated[:, 1:] = p[:, 1:]@rotation
        cases.append((rotated, mass, incoming))
        order = [3, 0, 2, 1]
        cases.append((p[order], mass[order], incoming[order]))
        for valid, factor in self.evaluate(cases):
            self.assertTrue(valid)
            self.assertAlmostEqual(factor/expected, 1., delta=2e-13)

    def test_massless_or_single_massive_outgoing_carrier_is_one(self):
        p, mass, incoming = self.pair()
        massless = np.zeros(4)
        light_p = p.copy()
        light_p[:, 0] = np.linalg.norm(light_p[:, 1:], axis=1)
        one_mass = mass.copy()
        one_mass[3] = 0.
        one_p = p.copy()
        one_p[3, 0] = np.linalg.norm(one_p[3, 1:])
        for valid, factor in self.evaluate([(light_p, massless, incoming),
                                            (one_p, one_mass, incoming)]):
            self.assertTrue(valid)
            self.assertEqual(factor, 1.)

    def test_threshold_and_unresolved_roundoff_request_fallback_without_floor(self):
        p, mass, incoming = self.pair(0.)
        near = p.copy()
        near[3, 0] *= 1-2e-16
        for valid, factor in self.evaluate([(p, mass, incoming), (near, mass, incoming)]):
            self.assertFalse(valid)
            self.assertEqual(factor, 1.)  # unused sentinel, never a finite bound

    def test_invalid_mass_or_off_shell_momentum_requests_fallback(self):
        p, mass, incoming = self.pair()
        off_shell = p.copy()
        off_shell[2, 0] *= 1.01
        negative_mass = mass.copy()
        negative_mass[2] = -1.
        for valid, _ in self.evaluate([(off_shell, mass, incoming),
                                        (p, negative_mass, incoming)]):
            self.assertFalse(valid)
