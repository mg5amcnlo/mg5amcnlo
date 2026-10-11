"""Exercise the worker protocol and Fortran/Python physical-coordinate parity."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestVirtualPooled(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix='mg5_virtual_pool_build_')
        cls.executable = Path(cls.build.name) / 'check'
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-fno-automatic', '-fcheck=all',
            '-ffpe-trap=invalid,zero,overflow',
            str(TEMPLATE / 'virtual_pooled.f90'),
            str(ROOT / 'tests/input_files/check_virtual_pooled.f90'),
            '-o', str(cls.executable)], cwd=cls.build.name,
            capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix='mg5_virtual_pool_case_')
        self.path = Path(self.work.name)

    def tearDown(self):
        self.work.cleanup()

    def run_case(self, name, success=True):
        result = subprocess.run([str(self.executable), name], cwd=self.path,
                                capture_output=True, text=True, timeout=30)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result.stdout

    def mode(self, value):
        (self.path / 'virtual_pool_mode.dat').write_text(str(value) + '\n')

    def model(self, blend=.25):
        # ratio = (6/2)*(4+2*z_1), with z_1 = log(shat/mur2)/10.
        content = ('VIRTUAL_POOL_MODEL 2\n5 8 2 4 2\n21 21 6 -6\n0 0 172.5 172.5\n1 1 0 0\n'
                   f'1 {blend} 2 6 2 1\n' + '-10 '*8 + '\n' + '10 '*8 + '\n'
                   '0 0 4\n1 0 2\n2 0 1 1 0 0\n' + '-10 '*8 + '\n' + '10 '*8 + '\n')
        (self.path / 'virtual_pool_model.dat').write_text(content)
        self.mode(2)
        return content

    def test_absent_marker_is_inactive(self):
        self.run_case('inactive')
        self.assertFalse((self.path / 'virtual_pool_samples.dat').exists())

    def test_zero_marker_is_inactive(self):
        self.mode(0)
        self.run_case('inactive')

    def test_collection_reconstructs_virtual_and_iteration_resets(self):
        self.mode(1)
        self.run_case('collect')
        lines = (self.path / 'virtual_pool_samples.dat').read_text().splitlines()
        self.assertEqual(lines[0], 'VIRTUAL_POOL_SAMPLES 2')
        self.assertEqual(list(map(int, lines[1].split())), [5, 8, 2, 4, 2])
        data = np.array([list(map(float, line.split())) for line in lines[5:]])
        self.assertEqual(data.shape, (4, 19))
        np.testing.assert_array_equal(data[:, :4], [[1, 7, 1, 1], [1, 7, 1, 2],
                                                   [2, 8, 2, 1], [2, 8, 2, 2]])
        np.testing.assert_allclose(data[:, 5], .25)
        np.testing.assert_allclose(data[:, 6], [2, -3, 0, -1])
        np.testing.assert_allclose(data[:, 7], [0, 7, 5, -2])
        np.testing.assert_allclose(data[:, 8], [20, 9, 0, 3])
        ledger = np.loadtxt(self.path / 'virtual_pool_iterations.dat', dtype=int)
        np.testing.assert_array_equal(ledger, [[1, 0, 19], [2, 1, 23]])
        # A restarted collection launch replaces records instead of appending.
        self.run_case('collect')
        self.assertEqual((self.path / 'virtual_pool_samples.dat').read_text().splitlines(), lines)

    def test_frozen_prediction_scales_blend_and_boundaries(self):
        self.model()
        self.run_case('apply')
        self.assertFalse((self.path / 'virtual_pool_samples.dat').exists())

    def test_zero_blend_reproduces_mint(self):
        self.model(blend=0)
        self.run_case('zero_blend')

    def test_model_can_be_reinitialized_with_static_locals(self):
        self.model()
        self.run_case('reinitialize')

    def test_model_rejects_mismatched_point_metadata(self):
        self.model()
        self.assertIn('point/model particle mismatch', self.run_case('point_mismatch', success=False))

    def test_model_rejects_mismatched_mass_metadata(self):
        self.model()
        self.assertIn('point/model mass mismatch', self.run_case('mass_mismatch', success=False))

    def test_apply_requires_model(self):
        self.mode(2)
        self.assertIn('requires virtual_pool_model.dat', self.run_case('bad_model', success=False))

    def test_malformed_models_fail_closed(self):
        original = self.model()
        bad = [original.replace('5 8 2 4 2', '5 6 2 4 2'),
               original.replace('5 8 2 4 2', '5 8 2 4 3'),
               original.replace('VIRTUAL_POOL_MODEL 2', 'VIRTUAL_POOL_MODEL 1'),
               original.replace('1 0.25 2 6 2 1', '1 1.25 2 6 2 1'),
               original.replace('1 0.25 2 6 2 1', '1 0.25 0 6 2 1'),
               original.replace('1 0 2', '9 0 2'),
               original.replace('0 0 4', '0 1 4'),
               original.replace('0 0 4', '0 0 NaN'),
               original.replace('0 0 4', '0 0 4 5'),
               original.rsplit('\n', 2)[0] + '\n', original + 'junk\n']
        for content in bad:
            with self.subTest(content=content):
                (self.path / 'virtual_pool_model.dat').write_text(content)
                self.assertIn('Pooled virtual:', self.run_case('bad_model', success=False))

    def test_invariants_are_process_independent_and_lorentz_invariant(self):
        cases = [
            ([6, -6], [172.5, 172.5], [[100, 50, 90], [-100, -50, -90]]),
            ([-11, 12, 21], [0, 0, 0], [[30, 15, 20], [-10, 20, -15], [-20, -35, -5]]),
            ([6, -6, 5, -5], [172.5, 172.5, 4.7, 4.7],
             [[100, 50, 90], [-50, 20, -70], [-30, 80, 20], [-20, -150, -40]])]
        for final_pdg, final_mass, vectors in cases:
            vectors = np.array(vectors, dtype=float)
            energies = np.sqrt(np.sum(vectors**2, axis=1) + np.array(final_mass)**2)
            e = energies.sum()/2
            p = np.vstack(([e, 0, 0, e], [e, 0, 0, -e], np.column_stack((energies, vectors))))
            masses = [0., 0.] + final_mass
            pdg = [21, 21] + final_pdg
            shat = (2*e)**2
            mur2 = 8000.
            expected = [np.log(shat/mur2)] + [
                np.log(max(abs(2*(p[i, 0]*p[j, 0]-p[i, 1:]@p[j, 1:]))/shat, 1e-15))
                for i in range(len(p)) for j in range(i+1, len(p))] + [
                np.log(shat/m**2) for m in sorted(set(masses)) if m > 0]
            for beta in (np.zeros(3), np.array([.12, -.17, .28])):
                b2 = beta@beta
                gamma = 1/np.sqrt(1-b2)
                bp = p[:, 1:]@beta
                boosted = p.copy()
                boosted[:, 0] = gamma*(p[:, 0]-bp)
                boosted[:, 1:] += (((gamma-1)/b2*bp if b2 else bp)-gamma*p[:, 0])[:, None]*beta
                lines = ['%d %.17e' % (len(p), mur2)]
                lines += [' '.join(map(str, [pdg[i], masses[i], int(i < 2), *leg]))
                          for i, leg in enumerate(boosted)]
                (self.path / 'coordinates_input.dat').write_text('\n'.join(lines)+'\n')
                fields = self.run_case('invariants').split()
                self.assertEqual(list(map(int, fields[:2])), [5, len(expected)])
                np.testing.assert_allclose(list(map(float, fields[2:])), expected, rtol=2e-13, atol=2e-14)
