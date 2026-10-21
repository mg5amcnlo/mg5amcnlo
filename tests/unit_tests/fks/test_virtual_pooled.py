"""Exercise the worker protocol and Fortran/Python physical-coordinate parity."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestVirtualPooled(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix='mg5_virtual_pool_build_')
        cls.executable = Path(cls.build.name) / 'check'
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-frecursive', '-fcheck=all',
            '-ffpe-trap=invalid,zero,overflow',
            str(TEMPLATE / 'virtual_envelope_carrier.f90'),
            str(TEMPLATE / 'virtual_features.f90'),
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
        bounds = [' '.join(map(str, [-10.]*15+[-1.])),
                  ' '.join(map(str, [10.]*15+[1.]))]
        content = '\n'.join([
            'VIRTUAL_POOL_MODEL 4', '9 16 2 4 2', '21 21 6 -6',
            '0 0 172.5 172.5', '1 1 0 0', '8 8 3 -3',
            f'1 {blend} 2 6 2 1', *bounds, '0 0 0 4', '1 0 0 2', 'SURROGATE 0 0',
            '2 0 1 1 0 0', *bounds, 'SURROGATE 0 0', ''])
        (self.path / 'virtual_pool_model.dat').write_text(content)
        self.mode(2)
        return content

    def test_absent_marker_is_inactive(self):
        self.run_case('inactive')
        self.assertFalse((self.path / 'virtual_pool_samples.dat').exists())

    def test_zero_marker_is_inactive(self):
        self.mode(0)
        self.run_case('inactive')

    def test_fks_born_query_resets_the_previous_colour_request(self):
        work = self.path/'SubProcesses/P0_check'
        work.mkdir(parents=True)
        model = self.path/'Source/MODEL'
        model.mkdir(parents=True)
        (work/'nexternal.inc').write_text('      integer nexternal\n      parameter(nexternal=5)\n')
        (work/'orders.inc').write_text('      integer qcd_pos\n      parameter(qcd_pos=1)\n')
        (work/'coupl.inc').write_text('      real(8) g\n      common/test_g/g\n')
        (work/'q_es.inc').write_text('      real(8) qes2\n      common/test_qes/qes2\n')
        (model/'input.inc').write_text('      real(8) mu_r\n      common/test_mu/mu_r\n')
        # Compile the production subroutine, rather than a copy in the fixture.
        source = (TEMPLATE/'virtual_approximation.f').read_text()
        helper = source[source.index('      subroutine prepare_pooled_helicities('):]
        helper = helper[:helper.index('      subroutine deferred_virtual_context(')]
        (work/'prepare.f').write_text(helper)
        # Exercise production automatic storage and the original static-local
        # regression: neither may reuse a colour request for normalization.
        for storage in ('-frecursive', '-fno-automatic'):
            with self.subTest(storage=storage):
                executable = work/'check'
                result = subprocess.run([shutil.which('gfortran'), '-O1', storage, '-fcheck=all',
                    '-ffixed-line-length-none', '-I', str(work),
                    str(ROOT/'Template/NLO/Source/BornSupport/mc_born_types.f90'),
                    str(ROOT/'tests/input_files/check_virtual_physics_born.f90'), str(work/'prepare.f'),
                    '-o', str(executable)], cwd=work, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                result = subprocess.run([str(executable)], cwd=work, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    @unittest.skipUnless(np is not None, 'requires numpy')
    def test_physics_capture_dresses_the_fit_without_changing_mint(self):
        from madgraph.various import virtual_pool as pool
        from madgraph.various.virtual_pool.polynomial import Marginals
        self.mode(1)
        inputs = np.fromstring(self.run_case('collect_physics'), sep=' ')
        metadata, rows, _ = pool.collect_samples([self.path])
        self.assertEqual((metadata['version'], metadata['catalog'], metadata['nf']), (5, 9, 16))
        self.assertEqual(metadata['colours'], (8, 8, 3, -3))
        np.testing.assert_allclose([r['born'] for r in rows], [2., -3.])
        np.testing.assert_allclose([r['coupling_factor'] for r in rows], [.02, .03])
        ns = pool.scalar_feature_count(metadata)
        np.testing.assert_allclose(rows[0]['features'][ns-5:ns-1], inputs)
        _, dressed, _ = pool._physics_rows(metadata, rows)
        np.testing.assert_allclose([r['born'] for r in dressed], [.04, -.09])
        np.testing.assert_allclose([r['mint_born'] for r in dressed], [2., -3.])
        self.assertEqual(pool._physics_rows(metadata, dressed)[1], dressed)
        ordinary, carried = Marginals(2, 4), Marginals(2, 4)
        ordinary.update(rows); carried.update(dressed)
        np.testing.assert_array_equal(ordinary.average, carried.average)
        np.testing.assert_allclose(ordinary.predict_many(rows), carried.predict_many(dressed))

    @unittest.skipUnless(np is not None, 'requires numpy')
    def test_antenna_remainder_matches_crossed_continuations_and_invalid_domains(self):
        try:
            import scipy.special
        except ImportError:
            self.skipTest('optional scipy unavailable')
        import itertools
        from madgraph.various.virtual_pool.helicity import antenna_remainder
        p = np.zeros((5, 4))
        p[:2] = [[300., 0., 0., 300.], [300., 0., 0., -300.]]
        angle = np.arange(3)*2*np.pi/3
        p[2:, 0] = 200.
        p[2:, 1] = 200.*np.cos(angle)
        p[2:, 2] = 200.*np.sin(angle)
        triplets = list(itertools.permutations(range(5), 3))
        lines = [f'5 {len(triplets)+1}', '1 1 0 0 0']
        expected, valid = [], []
        for triplet in triplets:
            value, accepted = antenna_remainder(p[None], (1, 1, 0, 0, 0), *triplet)
            expected.append(value[0]); valid.append(accepted[0])
            lines.append(' '.join(str(i+1) for i in triplet))
            lines.extend(' '.join(map(str, point)) for point in p)
        # A zero three-parton invariant has no resolvable R(u,v).
        degenerate = p.copy(); degenerate[2:] = [200., 200., 0., 0.]
        lines.append('3 4 5')
        lines.extend(' '.join(map(str, point)) for point in degenerate)
        (self.path/'antenna_input.dat').write_text('\n'.join(lines)+'\n')
        answer = np.fromstring(self.run_case('antenna'), sep=' ').reshape(-1, 2)
        np.testing.assert_array_equal(answer[:, 0], list(valid)+[False])
        np.testing.assert_allclose(answer[:, 1], expected+[0.], rtol=2.e-13, atol=3.e-14)

    @unittest.skipUnless(np is not None, 'requires numpy')
    def test_physics_prediction_applies_the_known_coupling_once(self):
        from madgraph.various import virtual_pool as pool
        self.mode(1)
        self.run_case('collect_physics')
        metadata, _, _ = pool.collect_samples([self.path])
        ns = pool.scalar_feature_count(metadata)
        orders = [dict(order=k, blend=1., born_scale=1., target_scale=1., degree=0,
                       terms=[dict(pair=(-1, -1), coefficient=value)],
                       lower=[-1.e6]*ns+[-1.], upper=[1.e6]*ns+[1.])
                  for k, value in ((1, 5.), (2, -7.))]
        (self.path/'virtual_pool_model.dat').write_text(pool._model_text(metadata, orders))
        loaded = pool.load_model(self.path/'virtual_pool_model.dat')
        self.assertEqual(loaded['metadata']['colours'], metadata['colours'])
        self.mode(2)
        self.run_case('apply_physics')

    @unittest.skipUnless(np is not None, 'requires numpy')
    def test_collection_reconstructs_virtual_and_iteration_resets(self):
        self.mode(1)
        self.run_case('collect')
        lines = (self.path / 'virtual_pool_samples.dat').read_text().splitlines()
        self.assertEqual(lines[0], 'VIRTUAL_POOL_SAMPLES 5')
        self.assertEqual(list(map(int, lines[1].split())), [9, 16, 2, 4, 2])
        data = np.array([list(map(float, line.split())) for line in lines[8:]])
        self.assertEqual(data.shape, (4, 54))
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

    @unittest.skipUnless(np is not None, 'requires numpy')
    def test_rejected_grid_reset_keeps_monotonic_capture_ordinals(self):
        from madgraph.various import virtual_pool
        self.mode(1)
        self.run_case('collect_reset_zero')
        ledger = np.loadtxt(self.path/'virtual_pool_iterations.dat', dtype=int)
        np.testing.assert_array_equal(ledger, [[1, 0, 19], [2, 1, 23]])
        _, rows, attempted, _ = virtual_pool._read_job(str(self.path))
        self.assertEqual(attempted, 23)  # rejected attempts never enter fitting weights
        self.assertEqual(len(rows), 2)
        self.assertEqual({row['iteration'] for row in rows}, {2})

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
        bad = [original.replace('9 16 2 4 2', '9 15 2 4 2'),
               original.replace('9 16 2 4 2', '9 16 2 4 3'),
               original.replace('VIRTUAL_POOL_MODEL 4', 'VIRTUAL_POOL_MODEL 3'),
               original.replace('1 0.25 2 6 2 1', '1 1.25 2 6 2 1'),
               original.replace('1 0.25 2 6 2 1', '1 0.25 0 6 2 1'),
               original.replace('1 0 0 2', '17 0 0 2'),
               original.replace('0 0 0 4', '0 1 0 4'),
               original.replace('0 0 0 4', '0 0 0 NaN'),
               original.replace('0 0 0 4', '0 0 0 4 5'),
               original.rsplit('\n', 2)[0] + '\n', original + 'junk\n']
        for content in bad:
            with self.subTest(content=content):
                (self.path / 'virtual_pool_model.dat').write_text(content)
                self.assertIn('Pooled virtual:', self.run_case('bad_model', success=False))

    @unittest.skipUnless(np is not None, 'requires numpy')
    def test_invariants_are_process_independent_and_lorentz_invariant(self):
        from madgraph.various import virtual_pool as pool
        from madgraph.various.virtual_pool import helicity
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
            meta = dict(nlegs=len(p), masses=masses, incoming=[1, 1]+[0]*len(final_mass))
            threshold = pool.threshold_features(meta, p[None])[0].tolist()
            orientation = helicity.orientation_features(p[None], meta['incoming'])[0].tolist()
            expected += [np.log(shat/mur2)]+threshold+[0.]*4+[np.log(shat)]+orientation
            for beta in (np.zeros(3), np.array([.12, -.17, .28])):
                b2 = beta@beta
                gamma = 1/np.sqrt(1-b2)
                bp = p[:, 1:]@beta
                boosted = p.copy()
                boosted[:, 0] = gamma*(p[:, 0]-bp)
                boosted[:, 1:] += (((gamma-1)/b2*bp if b2 else bp)-gamma*p[:, 0])[:, None]*beta
                lines = ['%d %.17e %.17e' % (len(p), mur2, mur2)]
                lines += [' '.join(map(str, [pdg[i], masses[i], int(i < 2), *leg]))
                          for i, leg in enumerate(boosted)]
                (self.path / 'coordinates_input.dat').write_text('\n'.join(lines)+'\n')
                fields = self.run_case('invariants').split()
                self.assertEqual(list(map(int, fields[:2])), [9, len(expected)])
                np.testing.assert_allclose(list(map(float, fields[2:])), expected, rtol=2e-13, atol=2e-14)

    def test_invariants_distinguish_loop_scale_and_absolute_energy(self):
        p=np.array([[250.,0,0,250.],[250.,0,0,-250.],
                    [250.,150.,0.,200.],[250.,-150.,0.,-200.]])
        def coordinates(scale,qes):
            lines=['4 %.17e %.17e' % (10000.*scale**2,qes)]
            lines += [' '.join(map(str,[21,0.,int(i<2),*(leg*scale)]))
                      for i,leg in enumerate(p)]
            (self.path/'coordinates_input.dat').write_text('\n'.join(lines)+'\n')
            fields=self.run_case('invariants').split()
            self.assertEqual(list(map(int,fields[:2])),[9,14])
            return np.asarray(list(map(float,fields[2:])))
        first=coordinates(1.,40000.)
        changed_loop=coordinates(1.,90000.)
        np.testing.assert_allclose(first[:7],changed_loop[:7])
        np.testing.assert_allclose(first[8:],changed_loop[8:])
        self.assertAlmostEqual(changed_loop[7]-first[7],np.log(40000./90000.))
        scaled=coordinates(2.,160000.)
        np.testing.assert_allclose(first[:12],scaled[:12],atol=1e-14)
        np.testing.assert_allclose(first[13:],scaled[13:],atol=1e-14)
        self.assertAlmostEqual(scaled[12]-first[12],np.log(4.))
