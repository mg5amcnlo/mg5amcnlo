"""Native conditional predictions agree with Python and preserve estimator means."""
from pathlib import Path
import hashlib
import itertools
import math
import shutil
import subprocess
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None
from madgraph.various import virtual_pool as pool
from madgraph.various import virtual_pool_neural as neural

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'


@unittest.skipUnless(shutil.which('gfortran') and np is not None,
                     'requires gfortran and numpy')
class ConditionalVirtualIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix='mg5_conditional_build_')
        cls.addClassCleanup(cls.build.cleanup)
        cls.exe = Path(cls.build.name) / 'check'
        command = [shutil.which('gfortran'), '-O2', '-frecursive', '-fcheck=all',
                   '-ffpe-trap=invalid,zero,overflow',
                   str(TEMPLATE / 'virtual_envelope_carrier.f90'),
                   str(TEMPLATE / 'virtual_pooled.f90'),
                   str(ROOT / 'tests/input_files/check_conditional_virtual_integration.f90'),
                   '-o', str(cls.exe)]
        result = subprocess.run(command, cwd=cls.build.name, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='mg5_conditional_case_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.work = self.root / 'GF1.0'
        self.work.mkdir()
        self.resources = self.root / 'MadLoop5_resources'
        self.resources.mkdir()
        self.rng = np.random.default_rng(74618)

    def setup_case(self, widths=(5,), collect=False, incoming=(1, 1, 0, 0), symmetry=False,
                   physics=False, replacement_blend=None):
        n = 15
        mult = np.zeros(16, dtype=int)
        heads = np.array([2, 3, 5, 6, 8])  # Native gg -> tt representatives.
        mult[heads-1] = [2, 2, 2, 4, 2]
        if symmetry:
            # Complete native map, with nonunit multiplicities equal within
            # every orbit of the identical outgoing-leg transposition.
            heads = np.arange(1, 17)
            mult = np.array([1, 2, 2, 1, 1, 4, 4, 1, 1, 1, 1, 1, 1, 3, 3, 1])
        labels = np.array(list(itertools.product((-1, 1), repeat=4)), dtype=int)
        masses = np.array([0., 0., 173., 173.])
        metadata = dict(version=3, catalog=7, nf=11, norders=3, nlegs=4, nx=2,
                        pdgs=[21, 21, 6, -6], masses=masses.tolist(), incoming=list(incoming),
                        nhel=16, helicities=[(i+1, int(mult[i]), *labels[i]) for i in range(16)])
        if symmetry:
            metadata['pdgs'][3] = metadata['pdgs'][2]
        energy = self.rng.uniform(180., 900., n)
        direction = self.rng.normal(size=(n, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        v = np.sqrt(energy**2-173.**2)[:, None]*direction
        p = np.zeros((n, 4, 4))
        p[:, 0, 0] = p[:, 0, 3] = energy
        p[:, 1, 0] = energy
        p[:, 1, 3] = -energy
        p[:, 2] = np.column_stack((energy, v))
        p[:, 3] = np.column_stack((energy, -v))
        if sum(incoming) == 1:
            # Four-body fixture with a massive parent in an arbitrary slot.
            parent = int(np.flatnonzero(incoming)[0])
            p[:, parent] = p.sum(axis=1)
            masses[parent] = 0.  # Geometry permits varying parent invariant mass.
            metadata['masses'] = masses.tolist()
        mur2 = np.geomspace(1., 1.e8, n)
        qes2 = mur2*.7
        weights = self.rng.uniform(.01, 4., (n, 3, 16))
        weights[:, 2] = 0.  # A zero Born squared order must not dilute q.
        # Nonzero excluded-ID weights must never enter the native mixture.
        weights[:, :, mult == 0] = 100.
        normalized = weights*mult[None, None, :]
        totals = normalized.sum(axis=2)
        normalized = np.divide(normalized, totals[:, :, None], out=np.zeros_like(normalized),
                               where=totals[:, :, None] > 0)
        q = normalized.sum(axis=1)/(totals > 0).sum(axis=1)[:, None]
        mint = self.rng.normal(size=(n, 3))
        born = self.rng.normal(size=(n, 3))
        born[:3, 0] = [0., -1., 1.]
        target = self.rng.normal(size=(n, 3))
        descriptors = [(1, 1, 0, 0, -2., 2.), (2, 11, 0, 0, -1., 1.),
                       (3, 3, 2, 0, -1., 1.), (4, 1, 2, 3, -8., 1.),
                       (5, 1, 2, 3, -1., 1.), (6, 6, 0, 0, -1., 1.)]
        if incoming[2]:
            descriptors[2] = (3, 1, 2, 0, -1., 1.)
        if physics:
            metadata.update(catalog=8, nf=12)
            descriptors[1] = (2, 12, 0, 0, -1., 1.)
            descriptors += [(7, 1, 0, 0, -2., 2.), (1, 10, 0, 0, 1., 3.),
                            (7, 10, 0, 0, 1., 3.)]
        base_orders = [dict(order=k+1, blend=1., born_scale=1., target_scale=1., degree=0,
                            lower=[-2.]*(metadata['nf']-1)+[-1.],
                            upper=[2.]*(metadata['nf']-1)+[1.],
                            terms=[dict(pair=(-1, -1, -1), coefficient=(k+1)*.2)]) for k in range(3)]
        if physics:
            for k, entry in enumerate(base_orders):
                entry['degree'] = 1
                entry['lower'][9], entry['upper'][9] = 1., 3.
                entry['terms'].append(dict(pair=(9, -1, -1), coefficient=(k+1)*.15))
        if symmetry:
            for k, entry in enumerate(base_orders):
                entry['degree'] = 1
                entry['terms'].append(dict(pair=(2, -1, -1), coefficient=(k+1)*.11))
        base_text = pool._model_text(metadata, base_orders)
        (self.work / 'virtual_pool_model.dat').write_text(base_text)
        orders = []
        for k in range(3):
            if k == 2:
                orders.append(dict(order=3, enabled=False))
                continue
            size = sum(math.prod(s) for s in neural._shapes(len(descriptors), len(heads)-1, widths))
            trained = [1, 0, 1, 1]
            if symmetry:
                trained = [1]*15
                trained[1:3] = [0, 0]
            orders.append(dict(order=k+1, enabled=True, descriptors=descriptors, widths=list(widths),
                               heads=heads[:-1].tolist(), trained=trained,
                               theta=self.rng.normal(scale=.17, size=size).tolist(),
                               born_scale=2.+k, target_scale=3.-k))
            if physics:
                # The native affine skip implements the linear-head model;
                # hidden weights/output remain exactly zero.
                parts = neural.unpack(orders[-1]['theta'], len(descriptors), len(heads)-1, widths)
                for part in parts[:-1]:
                    part[:] = 0.
                orders[-1]['theta'] = np.concatenate([part.ravel() for part in parts]).tolist()
            if replacement_blend is not None:
                orders[-1].update(replaces_common=True, replacement_blend=replacement_blend)
        model = dict(metadata=metadata, base_sha256=hashlib.sha256(base_text.encode()).hexdigest(), orders=orders)
        if symmetry:
            model['permutations'] = [[1, 2, 3, 4], [1, 2, 4, 3]]
        if not collect:
            (self.work / 'virtual_pool_neural.dat').write_text(neural.model_text(model))
        (self.work / 'virtual_pool_mode.dat').write_text('1\n' if collect else '2\n')
        np.savetxt(self.resources / 'HelConfigs.dat', labels, fmt='%d')
        np.savetxt(self.resources / 'HelFilter.dat', np.where(mult > 0, mult, -10000), fmt='%d')
        with (self.work / 'points.dat').open('w') as f:
            f.write(f'{n} 4 3 16 3\n')
            f.write(' '.join(map(str, metadata['pdgs']))+'\n')
            f.write(' '.join(map(str, masses))+'\n')
            f.write(' '.join(map(str, incoming))+'\n')
            f.write(' '.join(map(str, mult))+'\n')
            for i in range(n):
                row = np.r_[mur2[i], qes2[i], p[i].ravel(), weights[i].ravel(order='F'),
                            mint[i], born[i], target[i]]
                f.write(' '.join('%.17e' % x for x in row)+'\n')
        features = self.point_features(p, masses, incoming, mur2, qes2)
        if physics:
            threshold = pool.threshold_features(metadata, p)
            features = np.column_stack((features[:, :9], threshold, features[:, 9:]))
        return dict(n=n, model=model, p=p, q=q, heads=heads, born=born, target=target,
                    mint=mint, features=features, mur2=mur2, qes2=qes2, base_orders=base_orders)

    @staticmethod
    def point_features(p, masses, incoming, mur2, qes2):
        shat = neural._dot(p[:, np.flatnonzero(incoming)].sum(1), p[:, np.flatnonzero(incoming)].sum(1))
        features = [np.log(shat/mur2)]
        features += [np.log(np.maximum(abs(2*neural._dot(p[:, i], p[:, j]))/shat, 1.e-15))
                     for i, j in itertools.combinations(range(4), 2)]
        features += [np.log(shat/m**2) for m in sorted(set(masses)) if m > 0]
        features += [np.log(shat/qes2), np.log(shat)]
        features = np.column_stack((*features, neural.orientation_features(p, incoming)))
        return features

    def run_native(self, mode='apply', success=True):
        result = subprocess.run([str(self.exe), mode], cwd=self.work,
                                capture_output=True, text=True, timeout=30)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def check_prediction(self, widths, incoming=(1, 1, 0, 0), physics=False):
        data = self.setup_case(widths=widths, incoming=incoming, physics=physics)
        result = self.run_native()
        actual = np.loadtxt(result.stdout.splitlines()).reshape(data['n'], 6, 6)
        model = data['model']
        common = pool.load_model(self.work/'virtual_pool_model.dat')
        for block, selected in enumerate([-1]+data['heads'].tolist()):
            for k, entry in enumerate(model['orders']):
                chosen, mean = neural.predict_correction(entry, model['metadata'], data['features'],
                                                         data['p'], data['q'], np.ones(data['n']),
                                                         np.full(data['n'], selected),
                                                         common=pool.predict(common, k+1, data['features'],
                                                                             np.ones(data['n']), data['mint'][:, k]))
                base = pool.predict(common, k+1, data['features'], np.ones(data['n']), data['mint'][:, k])
                expected_mean = base+mean
                expected_chosen = base+chosen
                np.testing.assert_allclose(actual[:, block, k], expected_mean, rtol=2.e-12, atol=2.e-13)
                np.testing.assert_allclose(actual[:, block, k+3], expected_chosen, rtol=2.e-12, atol=2.e-13)
        # Native q includes nonunit GOODHEL exactly once; unknown/masked heads
        # contribute their shared baseline and preserve the selected mean.
        summed = (actual[:, 1:, 3:]*data['q'][:, data['heads']-1, None]).sum(1)
        np.testing.assert_allclose(summed, actual[:, 0, :3], rtol=3.e-14, atol=3.e-14)
        signed = actual[:, 0, 3:]*data['born']
        self.assertTrue(np.isfinite(signed).all())
        self.assertEqual(signed[0, 0], 0.)

    def test_nonzero_one_layer_matches_python_with_native_multiplicities(self):
        self.check_prediction((5,))

    def test_nonzero_two_layers_masks_and_deferred_context_match_python(self):
        self.check_prediction((5, 3))

    def test_compact_linear_heads_with_thresholds_and_squared_inputs_match_python(self):
        self.check_prediction((1,), physics=True)
        model = neural.load_model(self.work/'virtual_pool_neural.dat', self.work/'virtual_pool_model.dat')
        self.assertEqual(model['metadata']['catalog'], 8)
        self.assertTrue(any(d[0] == 7 for d in model['orders'][0]['descriptors']))

    def test_colour_coupling_and_antenna_heads_match_python_and_survive_deferred_context(self):
        try:
            import scipy.special
        except ImportError:
            self.skipTest('optional scipy unavailable')
        data = self.setup_case(widths=(1,), physics=True, replacement_blend=.5)
        model = data['model']; meta = model['metadata']
        meta.update(version=4, catalog=9, nf=16, colours=[8, 8, 3, -3])
        coupling = np.array([.017, .023, .031])
        dipoles = np.zeros((4, 4)); dipoles[0, 1] = .75; dipoles[0, 2] = -.25; dipoles[2, 3] = .5
        with (self.work/'physics.dat').open('w') as stream:
            stream.write('8 8 3 -3\n'+' '.join(map(str, coupling))+'\n')
            stream.write(' '.join(map(str, dipoles.ravel(order='F')))+'\n')
        # Independent analytic colour kernels for the three nonzero links.
        p, scale = data['p'], data['qes2']
        kernels = np.zeros((data['n'], 4))
        for i, j in ((0, 1), (0, 2), (2, 3)):
            dot = neural._dot(p[:, i], p[:, j])
            logarithm = np.log(2*dot/scale)
            cusp = logarithm.copy()
            for leg in (i, j):
                if meta['masses'][leg]: cusp -= .5*np.log(meta['masses'][leg]**2/scale)
            if i == 2:
                velocity = np.sqrt(1-(173.**2/dot)**2)
                cusp = np.arctanh(velocity)/velocity
                kernels[:, 2] += dipoles[i, j]*cusp*logarithm
                kernels[:, 3] += dipoles[i, j]*np.pi**2/velocity
            kernels[:, 0] += dipoles[i, j]*cusp
            kernels[:, 1] += dipoles[i, j]*(logarithm**2-(np.pi**2 if i == 0 and j == 1 or i == 2 else 0.))
        data['features'] = np.column_stack((data['features'][:, :10], kernels, data['features'][:, 10:]))
        for entry in data['base_orders']:
            entry['lower'] = [-100.]*15+[-1.]; entry['upper'] = [100.]*15+[1.]
        base_text = pool._model_text(meta, data['base_orders'])
        (self.work/'virtual_pool_model.dat').write_text(base_text)
        model['base_sha256'] = hashlib.sha256(base_text.encode()).hexdigest()
        for entry in model['orders'][:2]:
            # New kinds replace existing inputs, retaining the affine shape.
            entry['descriptors'] = [(9, 11, 0, 0, -10., 10.), (2, 16, 0, 0, -1., 1.),
                (8, 1, 2, 3, -3., 3.), (8, 1, 3, 2, -3., 3.), (10, 1, 0, 0, -2., 2.),
                (9, 12, 0, 0, -100., 100.), (9, 13, 0, 0, -100., 100.),
                (9, 14, 0, 0, -100., 100.), (9, 15, 0, 0, -10., 20.)]
        text = neural.model_text(model)
        self.assertTrue(text.startswith('VIRTUAL_POOL_NEURAL 4\n'))
        (self.work/'virtual_pool_neural.dat').write_text(text)
        self.assertEqual(neural.model_text(neural.load_model(self.work/'virtual_pool_neural.dat',
                                                            self.work/'virtual_pool_model.dat')), text)
        actual = np.loadtxt(self.run_native().stdout.splitlines()).reshape(data['n'], 6, 6)
        common = pool.load_model(self.work/'virtual_pool_model.dat')
        for block, head in enumerate([-1]+data['heads'].tolist()):
            for k in range(3):
                chosen, mean = neural.predict_model(model, common, k+1, data['features'], data['p'], data['q'],
                    np.full(data['n'], coupling[k]), data['mint'][:, k], np.full(data['n'], head))
                np.testing.assert_allclose(actual[:, block, k], mean, rtol=3.e-12, atol=3.e-13)
                np.testing.assert_allclose(actual[:, block, k+3], chosen, rtol=3.e-12, atol=3.e-13)

    def test_absolute_linear_heads_blend_with_common_and_preserve_untrained_heads(self):
        data = self.setup_case(widths=(1,), physics=True, replacement_blend=.5)
        actual = np.loadtxt(self.run_native().stdout.splitlines()).reshape(data['n'], 6, 6)
        common = pool.load_model(self.work/'virtual_pool_model.dat')
        model = neural.load_model(self.work/'virtual_pool_neural.dat', self.work/'virtual_pool_model.dat')
        entry = model['orders'][0]
        inputs = neural.input_values(entry['descriptors'], data['features'], data['p'], data['q'],
                                      model['metadata']['incoming'])
        absolute = neural.forward(entry, inputs)[:, 0]*entry['target_scale']/entry['born_scale']
        base = pool.predict(common, 1, data['features'], np.ones(data['n']), data['mint'][:, 0])
        np.testing.assert_allclose(actual[:, 1, 3], .5*base+.5*absolute, rtol=2.e-12, atol=2.e-13)
        np.testing.assert_allclose(actual[:, 2, 3], base, rtol=2.e-12, atol=2.e-13)
        for block, head in enumerate([-1]+data['heads'].tolist()):
            selected, mean = neural.predict_model(model, common, 1, data['features'], data['p'], data['q'],
                                                  np.ones(data['n']), data['mint'][:, 0], np.full(data['n'], head))
            np.testing.assert_allclose(actual[:, block, 0], mean, rtol=2.e-12, atol=2.e-13)
            np.testing.assert_allclose(actual[:, block, 3], selected, rtol=2.e-12, atol=2.e-13)

    def test_unresolved_threshold_requests_exact_loop_and_uses_mint(self):
        data = self.setup_case(widths=(1,), physics=True)
        path = self.work/'points.dat'
        lines = path.read_text().splitlines()
        row = np.fromstring(lines[5], sep=' ')
        p = row[2:18].reshape(4, 4)
        p[2:] = [173., 0., 0., 0.]
        lines[5] = ' '.join('%.17e' % value for value in row)
        path.write_text('\n'.join(lines)+'\n')
        probabilities = np.loadtxt(self.run_native('ood').stdout.splitlines())
        self.assertEqual(probabilities[0], 1.)

    def test_absolute_head_is_preserved_when_the_common_baseline_is_huge(self):
        data = self.setup_case(widths=(1,), physics=True, replacement_blend=1.)
        for entry in data['base_orders']:
            entry.update(degree=0, terms=[dict(pair=(-1, -1, -1), coefficient=1.e25)])
        text = pool._model_text(data['model']['metadata'], data['base_orders'])
        (self.work/'virtual_pool_model.dat').write_text(text)
        data['model']['base_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        (self.work/'virtual_pool_neural.dat').write_text(neural.model_text(data['model']))
        actual = np.loadtxt(self.run_native().stdout.splitlines()).reshape(data['n'], 6, 6)
        common = pool.load_model(self.work/'virtual_pool_model.dat')
        entry = data['model']['orders'][0]
        inputs = neural.input_values(entry['descriptors'], data['features'], data['p'], data['q'],
                                      data['model']['metadata']['incoming'])
        expected = neural.forward(entry, inputs)[:, 0]*entry['target_scale']/entry['born_scale']
        prediction, _ = neural.predict_model(data['model'], common, 1, data['features'], data['p'], data['q'],
                                             np.ones(data['n']), data['mint'][:, 0], np.full(data['n'], 2))
        np.testing.assert_allclose(prediction, expected, rtol=2.e-12, atol=2.e-13)
        np.testing.assert_allclose(actual[:, 1, 3], expected, rtol=2.e-12, atol=2.e-13)

    def test_one_incoming_nonfirst_leg_matches_python(self):
        self.check_prediction((4, 3), incoming=(0, 1, 0, 0))

    def test_capture_three_reconstructs_selected_and_full_native_targets(self):
        data = self.setup_case(collect=True)
        self.run_native('capture')
        meta, rows, attempts, info = pool._read_job(str(self.work))
        self.assertEqual(meta['version'], 3)
        self.assertEqual(attempts, data['n'])
        self.assertEqual(meta['helicities'], tuple(tuple(x) for x in data['model']['metadata']['helicities']))
        self.assertEqual(len(rows), 3*data['n'])
        for j, row in enumerate(rows):
            i, k = divmod(j, 3)
            self.assertAlmostEqual(row['virtual'], data['target'][i, k], places=12)
            self.assertAlmostEqual(row['baseline'], data['mint'][i, k]*data['born'][i, k], places=12)
            np.testing.assert_allclose(row['q'], data['q'][i], rtol=3.e-15, atol=3.e-16)
            np.testing.assert_allclose(np.array(row['momenta']).reshape(4, 4), data['p'][i], rtol=0, atol=0)

    def test_native_map_mismatch_is_rejected(self):
        self.setup_case()
        path = self.resources / 'HelFilter.dat'
        values = np.loadtxt(path, dtype=int); values[1] = 1
        np.savetxt(path, values, fmt='%d')
        self.assertIn('helicity map mismatch', self.run_native(success=False).stdout)

    def test_foreign_native_context_uses_common_mint_fallback(self):
        data = self.setup_case()
        result = self.run_native('foreign')
        actual = np.loadtxt(result.stdout.splitlines()).reshape(data['n'], 6, 6)
        np.testing.assert_allclose(actual[:, 0, :3], data['mint'], rtol=0, atol=0)
        np.testing.assert_allclose(actual[:, :, 3:], np.broadcast_to(data['mint'][:, None, :], (data['n'], 6, 3)), rtol=0, atol=0)

    def replace_born_helicity_weights(self, data, invalid=False):
        path = self.work / 'points.dat'
        lines = path.read_text().splitlines()
        for i in range(data['n']):
            row = np.fromstring(lines[5+i], sep=' ')
            weights = row[18:66].reshape((3, 16), order='F')
            weights[:, data['heads']-1] = 0.
            if invalid:
                weights[0, data['heads'][0]-1] = -1.
            lines[5+i] = ' '.join('%.17e' % x for x in row)
        path.write_text('\n'.join(lines)+'\n')

    def test_zero_born_distribution_preserves_common_prediction(self):
        data = self.setup_case()
        self.replace_born_helicity_weights(data)
        result = self.run_native('zero_weights')
        actual = np.loadtxt(result.stdout.splitlines()).reshape(data['n'], 6, 6)
        np.testing.assert_allclose(actual, np.broadcast_to([.2, .4, .6, .2, .4, .6], actual.shape),
                                   rtol=2.e-15, atol=0)

    def test_negative_born_distribution_preserves_common_prediction(self):
        data = self.setup_case()
        self.replace_born_helicity_weights(data, invalid=True)
        result = self.run_native('invalid_weights')
        actual = np.loadtxt(result.stdout.splitlines())
        np.testing.assert_allclose(actual, np.broadcast_to([.2, .4, .6, .2, .4, .6], actual.shape),
                                   rtol=2.e-15, atol=0)

    def check_symmetry_prediction(self, widths):
        data = self.setup_case(widths=widths, symmetry=True)
        result = self.run_native()
        actual = np.loadtxt(result.stdout.splitlines()).reshape(data['n'], 17, 6)
        model = data['model']; meta = model['metadata']
        labels = np.array([h[2:] for h in meta['helicities']])
        lookup = {tuple(row): i+1 for i, row in enumerate(labels)}
        expected = np.zeros_like(actual)
        for permutation in model['permutations']:
            perm = np.array(permutation)-1
            mapped = np.array([lookup[tuple(row[perm])] for row in labels])
            p = data['p'][:, perm]
            features = self.point_features(p, meta['masses'], meta['incoming'], data['mur2'], data['qes2'])
            q = data['q'][:, mapped-1]
            for block, selected in enumerate([-1]+data['heads'].tolist()):
                transformed = -1 if selected < 0 else mapped[selected-1]
                for k, entry in enumerate(model['orders']):
                    chosen, mean = neural.predict_correction(entry, meta, features, p, q,
                                                             np.ones(data['n']), np.full(data['n'], transformed))
                    base = (k+1)*(.2+.11*np.clip(features[:, 2]/2., -1, 1))
                    expected[:, block, k] += (base+mean)/2
                    expected[:, block, k+3] += (base+chosen)/2
        np.testing.assert_allclose(actual, expected, rtol=3.e-12, atol=3.e-13)
        common = pool.load_model(str(self.work / 'virtual_pool_model.dat'))
        decoded = neural.load_model(str(self.work / 'virtual_pool_neural.dat'))
        for block, selected in enumerate([-1]+data['heads'].tolist()):
            for k in range(3):
                chosen, mean = neural.predict_model(decoded, common, k+1, data['features'], data['p'],
                                                     data['q'], np.ones(data['n']), data['mint'][:, k],
                                                     np.full(data['n'], selected))
                np.testing.assert_allclose(actual[:, block, k], mean, rtol=3.e-12, atol=3.e-13)
                np.testing.assert_allclose(actual[:, block, k+3], chosen, rtol=3.e-12, atol=3.e-13)
        summed = (actual[:, 1:, 3:]*data['q'][:, :, None]).sum(1)
        np.testing.assert_allclose(summed, actual[:, 0, :3], rtol=5.e-14, atol=5.e-14)
        # The absent ID16 and masked IDs2/3 retain the averaged complete base.
        np.testing.assert_allclose(actual[:, 16, 3:]/np.arange(1, 4),
                                   np.repeat(actual[:, 16, 3:4], 3, axis=1), rtol=2.e-14, atol=2.e-14)
        self.assertGreater(np.max(abs(actual[:, 16, 3]-(.2+.11*np.clip(data['features'][:, 2]/2., -1, 1)))), .01)

    def test_symmetry_averages_complete_scalar_and_one_layer_heads(self):
        self.check_symmetry_prediction((5,))

    def test_symmetry_two_layers_preserve_selected_mean_and_folded_context(self):
        self.check_symmetry_prediction((5, 3))

    def test_symmetry_rejects_incoming_exchange(self):
        self.setup_case(symmetry=True)
        path = self.work / 'virtual_pool_neural.dat'
        lines = path.read_text().splitlines()
        lines[24] = '2 1 3 4'
        path.write_text('\n'.join(lines)+'\n')
        self.assertIn('changes incoming legs', self.run_native(success=False).stdout)

    def test_symmetry_rejects_nonclosed_trained_mask(self):
        self.setup_case(symmetry=True)
        path = self.work / 'virtual_pool_neural.dat'
        lines = path.read_text().splitlines()
        # Header +16 native rows +count/two permutations +order +6 inputs.
        lines[33] = '2 1'
        path.write_text('\n'.join(lines)+'\n')
        self.assertIn('trained masks are not closed', self.run_native(success=False).stdout)

    def test_symmetry_rejects_nonclosed_output_heads(self):
        self.setup_case(symmetry=True)
        path = self.work / 'virtual_pool_neural.dat'
        lines = path.read_text().splitlines(); lines[34] = '16 0'
        path.write_text('\n'.join(lines)+'\n')
        self.assertIn('output heads are not closed', self.run_native(success=False).stdout)

    def test_symmetry_rejects_mismatched_native_multiplicity(self):
        self.setup_case(symmetry=True)
        path = self.work / 'virtual_pool_neural.dat'
        lines = path.read_text().splitlines(); fields = lines[7].split(); fields[1] = '3'
        lines[7] = ' '.join(fields); path.write_text('\n'.join(lines)+'\n')
        path = self.resources / 'HelFilter.dat'
        values = np.loadtxt(path, dtype=int); values[1] = 3
        np.savetxt(path, values, fmt='%d')
        self.assertIn('changes native helicity multiplicity', self.run_native(success=False).stdout)

    def test_symmetry_zero_born_q_retains_identity_common_fallback(self):
        data = self.setup_case(symmetry=True)
        self.replace_born_helicity_weights(data)
        actual = np.loadtxt(self.run_native('zero_weights').stdout.splitlines()).reshape(data['n'], 17, 6)
        base = .2+.11*np.clip(data['features'][:, 2]/2., -1, 1)
        expected = base[:, None, None]*np.array([1, 2, 3, 1, 2, 3])[None, None, :]
        np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=2.e-14, atol=2.e-14)

    def test_symmetry_out_of_domain_in_either_ordering_forces_exact(self):
        data = self.setup_case(symmetry=True)
        model = data['model']; meta = model['metadata']
        base_orders = data['base_orders']
        for entry in base_orders:
            entry['lower'] = [-1.e10]*10+[-1.]
            entry['upper'] = [1.e10]*10+[1.]
        base_text = pool._model_text(meta, base_orders)
        (self.work / 'virtual_pool_model.dat').write_text(base_text)
        model['base_sha256'] = hashlib.sha256(base_text.encode()).hexdigest()
        for entry in model['orders']:
            if not entry['enabled']:
                continue
            entry['descriptors'] = [tuple(d[:4])+(-1.e10, 1.e10) for d in entry['descriptors']]
            x = data['features'][0, 2]
            entry['descriptors'][0] = (1, 3, 0, 0, x-1.e-5, x+1.e-5)
        path = self.work / 'points.dat'
        lines = path.read_text().splitlines(); lines[0] = '1 4 3 16 3'
        path.write_text('\n'.join(lines[:6])+'\n')
        companion = self.work / 'virtual_pool_neural.dat'
        companion.write_text(neural.model_text(model))
        self.assertEqual(float(self.run_native('ood').stdout), 1.)
        del model['permutations']
        companion.write_text(neural.model_text(model))
        self.assertEqual(float(self.run_native('ood').stdout), .2)
