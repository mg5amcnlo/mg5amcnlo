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
from madgraph.various.virtual_pool import polynomial, surrogate
from madgraph.various.virtual_pool import helicity as neural


def metadata():
    return dict(version=5, catalog=9, nf=16, norders=1, nlegs=4, nx=2,
                pdgs=(21, 21, 6, -6), colours=(8, 8, 3, -3), masses=(0., 0., 173., 173.), incoming=(1, 1, 0, 0),
                nhel=3, helicities=((1, 2, 1, -1, 1, -1), (2, 0, -1, 1, 1, -1),
                                    (3, 1, 1, 1, -1, -1)))


def momenta(n):
    p = np.array([[300., 0., 0., 300.], [300., 0., 0., -300.],
                  [300., 200., 0., 141.38953285], [300., -200., 0., -141.38953285]])
    return np.broadcast_to(p, (n, 4, 4)).copy()


def companion(widths=(3,)):
    descriptors = [(1, 1, 0, 0, -2., 2.), (2, 16, 0, 0, -1., 1.),
                   (3, 3, 2, 0, -1., 1.), (4, 1, 2, 3, -8., 1.),
                   (5, 1, 2, 3, -1., 1.), (6, 1, 0, 0, -1., 1.)]
    size = sum(np.prod(s) for s in neural._shapes(len(descriptors), 2, widths))
    return dict(metadata=metadata(), base_sha256='a'*64, orders=[dict(
        order=1, enabled=True, descriptors=descriptors, widths=list(widths),
        heads=[1, 3], trained=[1, 1], theta=np.linspace(-.1, .2, size).tolist(),
        born_scale=2., target_scale=3.)])


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
        p[0, 2:] = [173., 0., 0., 0.]
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            pool.threshold_features(meta, p)

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
        x = np.zeros((4, 16)); x[:, 0] = [-4., -.5, .2, 8.]
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
        x = np.zeros((4, 16)); x[:, 0] = [-1.21, -1.1, 1.1, 1.21]
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
        x = np.zeros((4, 16)); x[:, 0] = [-1.21, -1.1, 1.1, 1.21]
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


    def test_codec_rejects_native_incompatible_inputs_and_head_ids(self):
        model = companion()
        # Native canonical inputs must reference outgoing legs; incoming
        # components are zero auxiliaries, not portable model descriptors.
        model['orders'][0]['descriptors'][2] = (3, 1, 2, 0, -1., 1.)
        with self.assertRaisesRegex(ValueError, 'descriptor indices'):
            neural.model_text(model)
        model = companion()
        model['orders'][0]['heads'][0] = model['metadata']['nhel']+1
        with self.assertRaisesRegex(ValueError, 'dimensions/scales'):
            neural.model_text(model)


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

    def test_residual_polynomial_retains_parity_and_coefficient_budget(self):
        rng = np.random.default_rng(292)
        x = rng.uniform(-1, 1, (500, 16)); x[:, 2:15] = 0
        b = rng.normal(size=len(x)); b[::20] = 0
        y = b*(.3+.2*x[:, 0]**2+x[:, 15]*(.4-.6*x[:, 1]))
        model = polynomial.fit(x, b, y, np.ones(len(x)), 2, 1.e-6, nscalar=15)
        self.assertEqual(model['lower'][-1], -1.)
        self.assertEqual(model['upper'][-1], 1.)
        for term in model['terms']:
            self.assertLessEqual(sum(i >= 15 for i in term['pair']), 1)
        prediction = polynomial.predict(model, x, b, np.zeros(len(x)))
        self.assertLess(np.linalg.norm(prediction-y)/np.linalg.norm(y), 1.e-4)
        large = rng.normal(size=(2, 200))
        with self.assertRaises(ValueError): polynomial.fit(large, np.ones(2), np.ones(2), np.ones(2), 2, .01, nscalar=180)


    def test_missing_dependencies_or_capture_removes_stale_companion(self):
        path = self.root/'neural'; path.write_text('stale')
        report = neural.train([], dict(version=2), [], [], self.root/'base', path)
        self.assertFalse(path.exists()); self.assertFalse(report['enabled'])
        path.write_text('stale')
        with mock.patch.dict('sys.modules', {'scipy.optimize': None}):
            report = neural.train([], metadata(), [], [], self.root/'base', path)
        self.assertEqual(report['reason'], 'all_orders_rejected')
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
                for directory in ('a', 'b') for i in range(25) for order in (1, 2)]
        mask = neural.chronological_mask(rows, .5)
        for row, early in zip(rows, mask):
            self.assertEqual(early, row['sample'] < 10)

    def test_forward_base_recipe_and_coefficients_see_only_earlier_labels(self):
        rows = [dict(sample=i+1, directory='job', channel=1, iteration=1+i//10,
                     order=1, bins=8, x=[.3, .5], features=[0.]*16, surrogate=[1., 0., 0., 0.],
                     surrogate_valid=True, surrogate_gs2=1., born=1., virtual=float(i), weight=1.) for i in range(30)]
        earlier = np.arange(30) < 20
        empty = polynomial.empty_order(1, 16, 'test')
        seen = []
        def checked(training, meta, order, fixed_order):
            seen.extend(r['sample'] for r in training)
            self.assertEqual(seen, list(range(1, 21)))
            return empty, {}
        # Poison the globally chosen recipe: the outer fold must not use it.
        with mock.patch.object(surrogate, 'fit_order', side_effect=checked):
            before, recipe, blend, _, _ = neural._refit_base(rows, metadata(), empty,
                {'selected': {'degree': 999}}, earlier)
            for row in rows[20:]: row['virtual'] = 1.e90
            seen.clear()
            after, _, _, _, _ = neural._refit_base(rows, metadata(), empty,
                {'selected': {'degree': -999}}, earlier)
        np.testing.assert_array_equal(before, after)
        self.assertIsNone(recipe)
        self.assertEqual(blend, 0.)


if __name__ == '__main__':
    unittest.main()
