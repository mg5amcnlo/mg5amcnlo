"""Conditional estimator identities, analytic gradients and guarded training."""
from pathlib import Path
import itertools
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None
from madgraph.various import virtual_pool as pool
from madgraph.various import virtual_pool_polynomial as polynomial
from madgraph.various import virtual_pool_neural as neural


def metadata():
    return dict(version=3, catalog=7, nf=11, norders=1, nlegs=4, nx=2,
                pdgs=(21, 21, 6, -6), masses=(0., 0., 173., 173.), incoming=(1, 1, 0, 0),
                nhel=3, helicities=((1, 2, 1, -1, 1, -1), (2, 0, -1, 1, 1, -1),
                                    (3, 1, 1, 1, -1, -1)))


def momenta(n):
    p = np.array([[300., 0., 0., 300.], [300., 0., 0., -300.],
                  [300., 200., 0., 141.38953285], [300., -200., 0., -141.38953285]])
    return np.broadcast_to(p, (n, 4, 4)).copy()


def companion(widths=(3,)):
    descriptors = [(1, 1, 0, 0, -2., 2.), (2, 11, 0, 0, -1., 1.),
                   (3, 3, 2, 0, -1., 1.), (4, 1, 2, 3, -8., 1.),
                   (5, 1, 2, 3, -1., 1.), (6, 1, 0, 0, -1., 1.)]
    size = sum(np.prod(s) for s in neural._shapes(len(descriptors), 2, widths))
    return dict(metadata=metadata(), base_sha256='a'*64, orders=[dict(
        order=1, enabled=True, descriptors=descriptors, widths=list(widths),
        heads=[1, 3], trained=[1, 1], theta=np.linspace(-.1, .2, size).tolist(),
        born_scale=2., target_scale=3.)])


def symmetric_companion():
    meta = dict(version=3, catalog=7, nf=10, norders=1, nlegs=4, nx=2,
                pdgs=(21, 21, 21, 21), masses=(0.,)*4, incoming=(1, 1, 0, 0), nhel=16)
    meta['helicities'] = tuple((i, 2 if i in (2, 3) else 1)+labels
                              for i, labels in enumerate(itertools.product((-1, 1), repeat=4), 1))
    descriptors = [(1, 3, 0, 0, -2., 2.), (2, 10, 0, 0, -1., 1.),
                   (6, 2, 0, 0, -1., 1.), (6, 3, 0, 0, -1., 1.)]
    size = sum(np.prod(s) for s in neural._shapes(len(descriptors), 4, (3,)))
    return dict(metadata=meta, base_sha256='b'*64, permutations=[[1, 2, 3, 4], [1, 2, 4, 3]],
                orders=[dict(order=1, enabled=True, descriptors=descriptors, widths=[3],
                             heads=[1, 2, 3, 4], trained=[1, 1, 1, 0],
                             theta=np.linspace(-.1, .2, size).tolist(), born_scale=1., target_scale=1.)])


@unittest.skipUnless(np is not None, 'requires numpy')
class ConditionalPoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='conditional_pool_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_threshold_coordinates_are_invariant_and_unresolved_points_fall_back(self):
        meta = metadata()
        beta = np.array([.01, .1, .8])
        energy = 173./np.sqrt(1-beta**2)
        p = np.zeros((3, 4, 4))
        p[:, :, 0] = energy[:, None]
        p[:, 0, 3], p[:, 1, 3] = energy, -energy
        p[:, 2, 1], p[:, 3, 1] = beta*energy, -beta*energy
        expected = ((1+beta**2)/(2*beta))[:, None]
        np.testing.assert_allclose(pool.threshold_features(meta, p), expected, rtol=2.e-12)
        scaled = dict(meta, masses=tuple(m*1.e100 for m in meta['masses']))
        np.testing.assert_allclose(pool.threshold_features(scaled, p*1.e100), expected, rtol=3.e-12)
        boosted = p.copy()
        boosted[:, :, 0] = (p[:, :, 0]+.4*p[:, :, 3])/np.sqrt(1-.4**2)
        boosted[:, :, 3] = (p[:, :, 3]+.4*p[:, :, 0])/np.sqrt(1-.4**2)
        np.testing.assert_allclose(pool.threshold_features(meta, boosted), expected, rtol=3.e-12)
        rows = [dict(features=[0.]*11, momenta=point.ravel().tolist()) for point in p]
        enriched, converted, reason = pool._physics_rows(meta, rows)
        self.assertEqual((enriched['catalog'], enriched['nf']), (8, 12))
        np.testing.assert_allclose([r['features'][9] for r in converted], expected[:, 0])
        self.assertEqual(len(rows[0]['features']), 11)
        p[0, 2:] = [173., 0., 0., 0.]
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            pool.threshold_features(meta, p)
        rows[0]['momenta'] = p[0].ravel().tolist()
        unchanged, original, reason = pool._physics_rows(meta, rows)
        self.assertIs(unchanged, meta)
        self.assertIs(original, rows)
        self.assertEqual(reason, 'threshold_geometry_unresolved')

    def test_coupled_linear_heads_match_dense_and_sparse_solvers_with_signed_zero_born(self):
        try:
            import scipy.sparse.linalg
        except ImportError:
            self.skipTest('optional scipy unavailable')
        rng = np.random.default_rng(928)
        n = 1800
        inputs = rng.uniform(-1, 1, (n, 4))
        q = rng.dirichlet([2., 1., 2.], n)
        mixing = np.eye(3)[rng.integers(0, 3, n)]
        mixing[::7] = q[::7]
        born = rng.uniform(.2, 2., n)
        born[::13] *= -1
        born[::31] = 0.
        coefficients = rng.normal(size=(5, 3))
        coefficients[:, 1] = 0.
        # A large spread in column sizes exercises the preconditioner and
        # the conversion back to portable, unscaled inference coefficients.
        inputs[:, 0] *= 1.e-3
        inputs[:, 1] *= 1.e3
        truth = np.column_stack((inputs, np.ones(n)))@coefficients
        target = born*np.sum(mixing*truth, axis=1)
        target[::31] = .1  # irreducible observations still belong in V-B*f
        configuration = dict(neural.LINEAR_CONFIGURATION, ridge=1.e-10, column_scale=True)
        args = (inputs[:1200], mixing[:1200], born[:1200], target[:1200],
                np.ones(1200), [1, 2, 3], [1, 0, 1], configuration)
        dense = neural.fit(*args)
        with mock.patch.object(neural, 'MAX_DENSE_LINEAR_PARAMETERS', 0):
            sparse = neural.fit(*args)
        a, b = neural.forward(dense, inputs), neural.forward(sparse, inputs)
        a *= dense['target_scale']/dense['born_scale']
        b *= sparse['target_scale']/sparse['born_scale']
        np.testing.assert_allclose(a, b, rtol=2.e-6, atol=1.e-6)
        np.testing.assert_allclose(a[:, (0, 2)], truth[:, (0, 2)], rtol=1.e-5, atol=1.e-5)
        np.testing.assert_array_equal(a[:, 1], 0.)
        self.assertEqual(dense['optimizer_iterations'], 0)

    def test_native_probability_mean_selected_identity_and_signed_zero_born(self):
        model = companion((3, 2)); entry = model['orders'][0]
        x = np.zeros((4, 11)); x[:, 0] = [-4., -.5, .2, 8.]
        q = np.tile([.7, 0., .3], (4, 1)); b = np.array([1., -2., 0., 4.])
        first, mean = neural.predict_correction(entry, metadata(), x, momenta(4), q, b, np.ones(4))
        third, _ = neural.predict_correction(entry, metadata(), x, momenta(4), q, b, np.full(4, 3))
        np.testing.assert_allclose(.7*first+.3*third, mean, rtol=1.e-14, atol=1.e-14)
        full, _ = neural.predict_correction(entry, metadata(), x, momenta(4), q, b, np.full(4, -1))
        np.testing.assert_array_equal(full, mean)
        self.assertEqual(mean[2], 0.)
        # The GOODHEL=2 multiplicity is already inside q; multiplying by it
        # again would fail the equality above.
        entry['trained'][1] = 0
        unknown, mean = neural.predict_correction(entry, metadata(), x, momenta(4), q, b, np.full(4, 3))
        np.testing.assert_array_equal(unknown, 0.)
        np.testing.assert_allclose(mean, .7*first)
        # Native inference bypasses all heads when no Born distribution exists.
        q[0] = 0.
        absent, mean = neural.predict_correction(entry, metadata(), x, momenta(4), q, b, np.ones(4))
        self.assertEqual(absent[0], 0.)
        self.assertEqual(mean[0], 0.)
        q[0] = [.7, 0., .7]
        with self.assertRaises(ValueError):
            neural.predict_correction(entry, metadata(), x, momenta(4), q, b, np.ones(4))

    def test_one_and_two_layer_analytic_gradients_include_full_branches(self):
        rng = np.random.default_rng(902)
        z = rng.normal(size=(11, 3)); mix = np.zeros((11, 2)); mix[:5, 0] = 1; mix[5:9, 1] = 1
        mix[9:] = [.2, .8]
        b = rng.normal(size=11); b[3] = 0
        target = rng.normal(size=11); weight = np.ones(11)/11
        for widths in ((4,), (4, 3)):
            size = sum(np.prod(s) for s in neural._shapes(3, 2, widths))
            theta = rng.normal(size=size)*.1
            args = (z, mix, b, target, weight, widths, .03, [1, 1])
            _, gradient = neural.objective(theta, *args)
            numerical = []
            for i in range(size):
                plus = theta.copy(); minus = theta.copy(); plus[i] += 1.e-6; minus[i] -= 1.e-6
                numerical.append((neural.objective(plus, *args)[0]-neural.objective(minus, *args)[0])/2.e-6)
            np.testing.assert_allclose(gradient, numerical, rtol=2.e-6, atol=2.e-9)

    def test_prepared_sparse_objective_preserves_full_branch_gradients(self):
        rng = np.random.default_rng(927)
        nheads = 32
        z = rng.normal(size=(64, 4))
        probabilities = rng.uniform(size=(64, nheads))
        probabilities /= probabilities.sum(axis=1)[:, None]
        mixing = np.zeros_like(probabilities)
        mixing[np.arange(len(z)), rng.integers(nheads, size=len(z))] = 1.
        mixing[::11] = probabilities[::11]
        born = rng.normal(size=len(z))
        born[::13] = 0.
        target = rng.normal(size=len(z))
        weight = rng.uniform(.1, 1., len(z))/len(z)
        trained = [1]*nheads
        trained[3] = 0
        widths = (5, 3)
        size = sum(np.prod(shape) for shape in neural._shapes(z.shape[1], nheads, widths))
        theta = rng.normal(0., .1, size)
        cases = ((mixing, trained, True), (probabilities, trained, False),
                 (probabilities, [1]+[0]*(nheads-1), True),
                 (np.zeros_like(mixing), trained, True))
        for mixture, mask, sparse in cases:
            args = (z, mixture, born, target, weight, widths, .03, mask)
            prepared = neural._prepare_objective(z, mixture, mask)
            self.assertEqual(prepared[-1], sparse)
            expected, gradient = neural.objective(theta, *args)
            actual, prepared_gradient = neural.objective(theta, *args, prepared)
            self.assertAlmostEqual(actual, expected, places=14)
            np.testing.assert_allclose(prepared_gradient, gradient, rtol=1.e-12, atol=1.e-14)
            for index in (0, size//2, size-1):
                plus, minus = theta.copy(), theta.copy()
                plus[index] += 1.e-6
                minus[index] -= 1.e-6
                numerical = (neural.objective(plus, *args, prepared)[0]-
                             neural.objective(minus, *args, prepared)[0])/2.e-6
                self.assertAlmostEqual(prepared_gradient[index], numerical, places=8)

    def test_neural_domain_flags_match_clamped_and_degenerate_native_inputs(self):
        x = np.zeros((4, 11)); x[:, 0] = [-1.21, -1.1, 1.1, 1.21]
        p = momenta(4); q = np.tile([.6, 0., .4], (4, 1))
        values, outside = neural.input_values([(1, 1, 0, 0, -1., 1.)], x, p, q,
                                               metadata()['incoming'], return_outside=True)
        np.testing.assert_array_equal(values[:, 0], [-1., -1., 1., 1.])
        np.testing.assert_array_equal(outside, [True, False, False, True])
        p[0, 2:, 1:3] = 0
        values, outside = neural.input_values([(3, 3, 1, 0, -1., 1.), (3, 1, 0, 0, -1., 1.)],
            x, p, q, metadata()['incoming'], return_outside=True)
        np.testing.assert_array_equal(outside, [True, False, False, False])
        self.assertEqual(values[0, 0], 0.)
        np.testing.assert_array_equal(values[:, 1], 0.)

    def test_physical_inputs_preserve_logarithmic_continuation_and_exact_tail_flags(self):
        x = np.zeros((4, 11)); x[:, 0] = [-1.21, -1.1, 1.1, 1.21]
        descriptors = [(9, 1, 0, 0, -1., 1.), (10, 1, 0, 0, -1., 1.)]
        values, outside = neural.input_values(descriptors, x, momenta(4),
            np.tile([.6, 0., .4], (4, 1)), metadata()['incoming'], return_outside=True)
        np.testing.assert_allclose(values[:, 0], x[:, 0])
        np.testing.assert_allclose(values[:, 1], x[:, 0]**2)
        np.testing.assert_array_equal(outside, [True, False, False, True])

    def test_codec_roundtrip_and_strict_native_map_bounds(self):
        for widths in ((3,), (3, 2)):
            model = companion(widths); path = self.root/'neural'
            payload = neural.model_text(model); path.write_text(payload)
            loaded = neural.load_model(path)
            self.assertEqual(neural.model_text(loaded), payload)
            path.write_text(payload+'unexpected\n')
            with self.assertRaises(ValueError): neural.load_model(path)
        model = companion(); model['orders'][0]['heads'][0] = 2
        with self.assertRaises(ValueError): neural.model_text(model)

    def test_version2_symmetry_averages_common_and_remaps_native_helicities(self):
        model = symmetric_companion(); meta = model['metadata']
        path = self.root/'v2'; payload = neural.model_text(model); path.write_text(payload)
        self.assertTrue(payload.startswith('VIRTUAL_POOL_NEURAL 2\n'))
        loaded = neural.load_model(path)
        self.assertEqual(neural.model_text(loaded), payload)
        entry = dict(order=1, blend=1., born_scale=1., target_scale=1., degree=1,
                     lower=[-2.]*9+[-1.], upper=[2.]*9+[1.],
                     terms=[dict(pair=(2, -1), coefficient=1.)])
        common = dict(metadata=meta, orders=[entry])
        rng = np.random.default_rng(134); x = rng.uniform(-1, 1, (20, 10)); x[:, 9] = 0.
        q = rng.uniform(size=(20, 16)); q /= q.sum(axis=1)[:, None]
        born = rng.normal(size=20); p = momenta(20); mint = np.zeros(20)
        selected = np.array([1, 2, 3, 4, 7]*4)
        prediction, mean = neural.predict_model(loaded, common, 1, x, p, q, born, mint, selected)
        mapping = np.asarray(neural.helicity_permutation(meta, [1, 2, 4, 3]))
        xp, pp = neural.permuted_features(meta, x, p, [1, 2, 4, 3])
        mirrored, mirrored_mean = neural.predict_model(loaded, common, 1, xp, pp,
            q[:, np.argsort(mapping)], born, mint, mapping[selected-1])
        np.testing.assert_allclose(prediction, mirrored, rtol=1.e-14, atol=1.e-14)
        np.testing.assert_allclose(mean, mirrored_mean, rtol=1.e-14, atol=1.e-14)
        summed = np.zeros(20)
        for h in range(1, 17):
            individual, _ = neural.predict_model(loaded, common, 1, x, p, q, born, mint, np.full(20, h))
            summed += q[:, h-1]*individual
        np.testing.assert_allclose(summed, mean, rtol=1.e-14, atol=1.e-14)
        # An untrained/unknown head retains the averaged common polynomial,
        # not the unpermuted common value.
        unknown = selected == 7
        np.testing.assert_allclose(prediction[unknown], born[unknown]*(x[unknown, 2]+x[unknown, 3])/4)
        q[0] = 0.
        fallback, _ = neural.predict_model(loaded, common, 1, x, p, q, born, mint, selected)
        self.assertAlmostEqual(fallback[0], born[0]*x[0, 2]/2)
        loaded['orders'][0]['trained'][1] = 0  # native IDs 2<->3 must agree
        with self.assertRaises(ValueError): neural.model_text(loaded)

    def test_codec_rejects_native_incompatible_inputs_and_permutation_head_ids(self):
        model = companion()
        # Native canonical inputs must reference outgoing legs; incoming
        # components are zero auxiliaries, not portable model descriptors.
        model['orders'][0]['descriptors'][2] = (3, 1, 2, 0, -1., 1.)
        with self.assertRaisesRegex(ValueError, 'descriptor indices'):
            neural.model_text(model)
        model = symmetric_companion()
        model['orders'][0]['heads'][0] = model['metadata']['nhel']+1
        with self.assertRaisesRegex(ValueError, 'dimensions/scales'):
            neural.model_text(model)

    def test_symmetry_selection_guards_complete_predictions_on_all_forward_holds(self):
        model = symmetric_companion(); meta = model['metadata']; entry = model['orders'][0]
        entry['theta'] = [0.]*len(entry['theta'])
        n = 900; rng = np.random.default_rng(711); x = rng.uniform(-1, 1, (n, 10)); x[:, 9] = 0.
        p = momenta(n); q = np.full((n, 16), 1/16); b = np.ones(n); w = np.ones(n)
        y = (x[:, 2]+x[:, 3])/4
        common = dict(blend=1., born_scale=1., target_scale=1., degree=1,
                      lower=[-2.]*9+[-1.], upper=[2.]*9+[1.], terms=[dict(pair=(2, -1), coefficient=1.)])
        folds = [dict(valid=np.arange(n) >= int(f*n), descriptors=entry['descriptors'],
                      common_model=common, mint=np.zeros(n)) for f in neural.FRACTIONS]
        rows = [dict(selected_helicity=2, full_branch=False) for _ in range(n)]
        result = neural._validate_symmetry(rows, meta, x, p, q, b, y, w,
            folds, [entry]*3, 1., [1, 2, 4, 3])
        self.assertTrue(result['eligible'], result)
        rejected = neural._validate_symmetry(rows, meta, x, p, q, b, x[:, 2]/2, w,
            folds, [entry]*3, 1., [1, 2, 4, 3])
        self.assertFalse(rejected['eligible'])

    def test_composite_budget_rejects_before_evaluating_triples(self):
        meta = dict(metadata(), nlegs=8, masses=(0.,)*8, incoming=(1, 1, 0, 0, 0, 0, 0, 0), nf=46)
        with mock.patch.object(neural, '_dot', side_effect=AssertionError('must not evaluate over-budget triples')):
            with self.assertRaisesRegex(ValueError, 'input budget'):
                neural.descriptors_for(meta, np.zeros((20, 46)), np.zeros((20, 8, 4)),
                                       np.zeros((20, 3)), np.ones(20, bool), [1, 3], bank='composite')
        model = companion(); model['orders'][0]['descriptors'][0] = (6, 4, 0, 0, -1., 1.)
        with self.assertRaises(ValueError): neural.model_text(model)
        model = companion(); model['orders'][0]['theta'][0] = float('nan')
        with self.assertRaises(ValueError): neural.model_text(model)

    def test_catalog7_polynomial_retains_parity_and_sparse_budget(self):
        rng = np.random.default_rng(292)
        x = rng.uniform(-1, 1, (500, 11)); x[:, 2:10] = 0
        b = rng.normal(size=len(x)); b[::20] = 0
        y = b*(.3+.2*x[:, 0]**2+x[:, 10]*(.4-.6*x[:, 1]))
        model = polynomial.fit(x, b, y, np.ones(len(x)), 3, 1.e-6, 'square', nscalar=10)
        self.assertEqual(model['lower'][-1], -1.)
        self.assertEqual(model['upper'][-1], 1.)
        for term in model['terms']:
            self.assertLessEqual(sum(i >= 10 for i in term['pair']), 1)
        prediction = polynomial.predict(model, x, b, np.zeros(len(x)))
        self.assertLess(np.linalg.norm(prediction-y)/np.linalg.norm(y), 1.e-4)
        large = rng.normal(size=(2, 200))
        with self.assertRaises(ValueError): polynomial.fit(large, np.ones(2), np.ones(2), np.ones(2), 3, .01, 'square', nscalar=180)

    def test_capture3_preserves_full_map_and_rejects_double_multiplicity(self):
        job = self.root/'job'; job.mkdir()
        head = ('VIRTUAL_POOL_SAMPLES 3\n7 11 1 4 2\n21 21 6 -6\n0 0 173 173\n1 1 0 0\n3\n'
                '1 2 1 -1 1 -1\n2 0 -1 1 1 -1\n3 1 1 1 -1 -1\n')
        prefix = [1, 1, 1, 1, 8, .5, -2., 4., 7.]+[0.]*11+[.2, .3]
        suffix = [0, 1, .7, .7, 0., .3]+momenta(1).ravel().tolist()
        path = job/pool.SAMPLES
        path.write_text(head+' '.join(map(str, prefix+suffix))+'\n')
        (job/pool.ITERATIONS).write_text('1 1 20\n')
        meta, rows, _ = pool.collect_samples([job])
        self.assertEqual(meta['helicities'][0][1], 2)
        self.assertEqual(rows[0]['weight'], .1)
        self.assertEqual(rows[0]['q'], (.7, 0., .3))
        suffix[3] *= 2
        path.write_text(head+' '.join(map(str, prefix+suffix))+'\n')
        with self.assertRaises(ValueError): pool.collect_samples([job])

    def test_missing_dependencies_or_capture_removes_stale_companion(self):
        path = self.root/'neural'; path.write_text('stale')
        report = neural.train([], dict(version=2), [], [], self.root/'base', path)
        self.assertFalse(path.exists()); self.assertFalse(report['enabled'])
        path.write_text('stale')
        with mock.patch.dict('sys.modules', {'scipy.optimize': None}):
            report = neural.train([], metadata(), [], [], self.root/'base', path)
        self.assertEqual(report['reason'], 'numpy_or_scipy_unavailable')
        self.assertFalse(path.exists())

    def test_direct_fit_generalizes_with_native_full_and_selected_labels(self):
        try:
            import scipy.optimize
        except ImportError:
            self.skipTest('optional scipy unavailable')
        rng = np.random.default_rng(190)
        z = rng.uniform(-1, 1, (1100, 3)); q = .2+.6*(z[:, 0]+1)/2
        mix = np.zeros((1100, 2)); h = rng.random(1100) > q
        mix[np.arange(len(z)), h.astype(int)] = 1
        mix[::9] = np.column_stack((q, 1-q))[::9]
        b = rng.normal(size=len(z)); b[::31] = 0
        heads = np.column_stack((.3+.5*z[:, 1], -.4+.2*z[:, 2]))
        target = b*np.sum(mix*heads, axis=1)
        config = dict(widths=(8, 5), ridge=1.e-5, alpha=1., iterations=250)
        entry = neural.fit(z[:800], mix[:800], b[:800], target[:800], np.ones(800), [1, 3], [1, 1], config)
        predicted = b[800:]*entry['target_scale']/entry['born_scale']*np.sum(mix[800:]*neural.forward(entry, z[800:]), axis=1)
        self.assertLess(np.linalg.norm(predicted-target[800:])/np.linalg.norm(target[800:]), .02)

    def test_chronological_splits_preserve_replicate_samples(self):
        rows = [dict(directory=directory, channel=1, iteration=1+i//5, sample=i, order=order)
                for directory in ('a', 'b') for i in range(10) for order in (1, 2)]
        mask = neural.chronological_mask(rows, .5)
        for row, early in zip(rows, mask):
            self.assertEqual(early, row['sample'] < 5)

    def test_forward_base_recipe_and_coefficients_see_only_earlier_labels(self):
        rows = [dict(sample=i+1, directory='job', channel=1, iteration=1+i//10,
                     order=1, bins=8, x=[.3, .5], features=[0.]*11,
                     born=1., virtual=float(i), weight=1.) for i in range(30)]
        earlier = np.arange(30) < 20
        empty = polynomial.empty_order(1, 11, 'test')
        seen = []
        def checked(training, folds, meta, order):
            seen.extend(r['sample'] for r in training)
            self.assertEqual(seen, list(range(1, 21)))
            return empty, {}
        # Poison the globally chosen recipe: the outer fold must not use it.
        with mock.patch.object(polynomial, 'fit_order', side_effect=checked):
            before, recipe, blend, _, _ = neural._refit_base(rows, metadata(), empty,
                {'selected': {'degree': 999}}, earlier)
            for row in rows[20:]: row['virtual'] = 1.e90
            seen.clear()
            after, _, _, _, _ = neural._refit_base(rows, metadata(), empty,
                {'selected': {'degree': -999}}, earlier)
        np.testing.assert_array_equal(before, after)
        self.assertIsNone(recipe)
        self.assertEqual(blend, 0.)

    def test_production_capture_to_guarded_nonzero_companion(self):
        self.check_production_capture(3)

    def test_coupling_colour_capture_to_guarded_nonzero_companion(self):
        self.check_production_capture(4)

    def check_production_capture(self, version):
        try:
            import scipy.optimize
        except ImportError:
            self.skipTest('optional scipy unavailable')
        rng = np.random.default_rng(412)
        job = self.root/'production'; job.mkdir()
        lines = ['VIRTUAL_POOL_SAMPLES 3', '7 11 1 4 2', '21 21 6 -6',
                 '0 0 173 173', '1 1 0 0', '3', '1 2 1 -1 1 -1',
                 '2 0 -1 1 1 -1', '3 1 1 1 -1 -1']
        if version == 4:
            lines[:2] = ['VIRTUAL_POOL_SAMPLES 4', '9 16 1 4 2']
            lines.insert(5, '8 8 3 -3')
        for i in range(900):
            x = np.zeros(11); x[0] = rng.uniform(-1, 1)
            b = rng.uniform(.2, 1.5)*(-1 if i%13 == 0 else 1)
            h = 1 if rng.random() < .6 else 3
            full = i%11 == 0
            y = b*(.2*x[0] + (.6*.8-.4*.9 if full else .8 if h == 1 else -.9))
            native_q = [.6, 0., .4] if i%37 else [0., 0., 0.]
            row = [i+1, 1, 1+i//300, 1, 8, 1., b, y, 0.]+x.tolist()+rng.random(2).tolist()
            if version == 4:
                coupling = .02*(1+.2*x[0])
                row[7] *= coupling
                # Insert the massive-pair and four colour kernels before
                # absolute log(shat); the native Born stays undressed.
                row[18:18] = [1.1, 0., 0., 0., 0.]
                row.insert(9, coupling)
            row += [int(full), -1 if full else h, 1. if full else .6 if h == 1 else .4]+native_q+momenta(1).ravel().tolist()
            lines.append(' '.join(map(str, row)))
        (job/pool.SAMPLES).write_text('\n'.join(lines)+'\n')
        (job/pool.ITERATIONS).write_text('1 1 300\n2 1 300\n3 1 300\n')
        base = self.root/'virtual_pool_model.dat'; companion_path = self.root/'virtual_pool_neural.dat'
        report_path = self.root/'virtual_pool_report.json'
        report = pool.train_pool([job], base, report_path=report_path,
                                 neural=True, output_neural=companion_path)
        self.assertTrue(report_path.exists())
        self.assertTrue(report['neural']['enabled'], report['neural'])
        self.assertTrue(report['enabled'])
        self.assertEqual(report['status'], 'trained')
        self.assertEqual(report['neural']['additional_loop_calls'], 0)
        self.assertEqual(report['model_format_version'], 3)
        detail = report['neural']['orders'][0]
        self.assertEqual(detail['selected']['configuration']['family'], 'linear')
        self.assertEqual(detail['neural_screen'], 'compact_linear_residual_small')
        self.assertEqual(detail['skipped_unknown_native_probabilities'], 25)
        for fold in detail['selected']['folds']:
            self.assertLess(fold['l1'], fold['baseline']['l1'])
            self.assertLessEqual(fold['rms'], 1.05*fold['baseline']['rms'])
        model = neural.load_model(companion_path, base)
        self.assertTrue(model['orders'][0]['enabled'])
        if version == 4:
            self.assertEqual(report['neural']['format_version'], 4)
            self.assertEqual(model['metadata']['colours'], (8, 8, 3, -3))
            self.assertTrue(detail['selected']['configuration']['column_scale'])

    def test_nonlinear_fit_requires_forward_improvement_over_linear_heads(self):
        try:
            import scipy.optimize
        except ImportError:
            self.skipTest('optional scipy unavailable')
        rng = np.random.default_rng(8472)
        rows = []
        for i in range(900):
            x = rng.uniform(-1., 1.)
            head = 1 if rng.random() < .6 else 3
            full = i%11 == 0
            sign = .2 if full else 1. if head == 1 else -1.
            rows.append(dict(sample=i+1, directory='job', channel=1, iteration=1+i//300,
                             order=1, bins=8, x=[.2, .4], features=[x]+[0.]*10,
                             born=1., virtual=sign*(.2+.8*np.sin(4*x)), weight=1.,
                             q=(.6, 0., .4), selected_helicity=-1 if full else head,
                             full_branch=full, momenta=momenta(1).ravel().tolist()))
        common = polynomial.empty_order(1, 11, 'test')
        zeros = np.zeros(len(rows))
        configuration = dict(widths=(8,), ridge=1.e-4, alpha=1., iterations=250)
        with mock.patch.object(neural, '_refit_base', return_value=(zeros, None, 0., common, zeros)), \
                mock.patch.object(neural, 'DEFAULT_CONFIGURATIONS', (configuration,)):
            entry, report = neural.train_order(rows, metadata(), common, {'strategy': 'compact'})
        self.assertTrue(entry['enabled'], report)
        self.assertNotEqual(report['selected']['configuration'].get('family'), 'linear')
        self.assertTrue(report['selected']['versus_linear']['eligible'])
        linear = [c for c in report['candidates'] if c['configuration'].get('family') == 'linear' and c['eligible']]
        self.assertTrue(linear)
        self.assertLess(report['selected']['mean_l1'], .5*min(c['mean_l1'] for c in linear))

    def test_final_refit_cannot_activate_a_head_absent_from_earliest_fold(self):
        rows = []
        for i in range(600):
            rare = i < 20 or (i >= 300 and i%2 == 0)
            rows.append(dict(sample=i+1, channel=1, directory='job', iteration=1,
                order=1, bins=8, x=[.2, .4], features=[i/600]+[0.]*10,
                born=1., virtual=0. if rare else 1., weight=1., q=(.7, 0., .3),
                selected_helicity=3 if rare else 1, full_branch=False,
                momenta=momenta(1).ravel().tolist()))
        masks = []
        def fit(inputs, mixing, born, target, weight, heads, trained, configuration):
            masks.append(list(map(int, trained)))
            widths = (1,); parts = [np.zeros(shape) for shape in neural._shapes(inputs.shape[1], len(heads), widths)]
            parts[-2][0] = 1.
            return dict(widths=widths, heads=heads, trained=trained,
                        born_scale=1., target_scale=1., theta=np.concatenate([p.ravel() for p in parts]))
        common = polynomial.empty_order(1, 11, 'test')
        with mock.patch.object(neural, '_refit_base', return_value=(np.zeros(600), None, 0., common, np.zeros(600))), \
                mock.patch.object(neural, 'fit', side_effect=fit):
            entry, report = neural.train_order(rows, metadata(), common, {})
        self.assertTrue(entry['enabled'], report)
        self.assertEqual(masks, [[1, 0]]*4)  # three forward fits plus all-data fit
        self.assertEqual(report['untrained_native_heads'], [3])
        self.assertEqual(report['trained_heads'], [1])


if __name__ == '__main__':
    unittest.main()
