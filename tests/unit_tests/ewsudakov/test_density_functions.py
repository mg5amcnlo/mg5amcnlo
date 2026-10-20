"""Packed-triangle density-matrix convention of the generated Fortran GET_INTER
routines and fxfx_ewsudakov.unpack_density_triangle, and DensityMatrix
contraction against amplitude-level expressions."""
from __future__ import division

import os
import sys
import unittest

import numpy as np

_REPO_ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        os.pardir,
    )
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import madgraph.various.Density_functions as density_functions  # noqa: E402
import madgraph.various.fxfx_ewsudakov as fx  # noqa: E402


def _upper_triangle(matrix):
    return [
        matrix[i, j]
        for i in range(matrix.shape[0])
        for j in range(i, matrix.shape[1])
    ]


class TestFortranTriangleConvention(unittest.TestCase):
    def test_upper_triangle_is_not_conjugated(self):
        """GET_INTER stores B[i,j]=M_i*conj(M_j) for i<=j."""
        rng = np.random.default_rng(20260728)
        amplitudes = rng.normal(size=9) + 1j * rng.normal(size=9)
        expected = np.outer(amplitudes, np.conj(amplitudes))

        reconstructed = fx.unpack_density_triangle(_upper_triangle(expected))

        np.testing.assert_allclose(reconstructed, expected, rtol=1e-13, atol=1e-13)
        self.assertGreater(
            np.max(np.abs(reconstructed - np.conj(expected))),
            1.0e-6,
        )

    def test_complex_sudakov_weight_uses_delta_not_conjugate_delta(self):
        """The packed-triangle reconstruction reproduces the complex-delta weight."""
        rng = np.random.default_rng(17)
        prod_amp = rng.normal(size=9) + 1j * rng.normal(size=9)
        decay_amp = rng.normal(size=9) + 1j * rng.normal(size=9)
        b_true = np.outer(prod_amp, np.conj(prod_amp))
        c_true = np.outer(decay_amp, np.conj(decay_amp))
        # Keep the contraction safely away from zero.
        c_true += 0.25 * np.trace(c_true).real / 9.0 * np.eye(9)
        delta = 0.08 * rng.normal(size=9) + 0.05j * rng.normal(size=9)

        b_reconstructed = fx.unpack_density_triangle(_upper_triangle(b_true))
        c_reconstructed = fx.unpack_density_triangle(_upper_triangle(c_true))

        def weight(b_matrix, c_matrix, correction):
            b = fx.DensityMatrix.from_full_matrix(b_matrix, [24, -24])
            c = fx.DensityMatrix.from_full_matrix(c_matrix, [24, -24])
            return b.apply_sudakov(correction).contract(c) / b.contract(c)

        factor = 1.0 + delta[:, None] + np.conj(delta)[None, :]
        expected = (
            np.sum(b_true * factor * c_true).real / np.sum(b_true * c_true).real
        )
        reconstructed = weight(b_reconstructed, c_reconstructed, delta)
        conjugated_matrices_result = weight(
            np.conj(b_true), np.conj(c_true), delta
        )
        conjugated_delta_result = weight(
            b_true, c_true, np.conj(delta)
        )
        real_delta_result = weight(b_true, c_true, np.real(delta))
        conjugated_matrices_real_delta_result = weight(
            np.conj(b_true), np.conj(c_true), np.real(delta)
        )

        self.assertAlmostEqual(reconstructed, expected, places=13)
        self.assertGreater(abs(conjugated_matrices_result - expected), 1.0e-7)
        self.assertAlmostEqual(
            conjugated_matrices_result, conjugated_delta_result, places=13
        )
        self.assertAlmostEqual(
            conjugated_matrices_real_delta_result, real_delta_result, places=13
        )

    def test_square_matrix_is_transpose_of_unpacked_triangle(self):
        """DensityMatrixObservables.square_matrix is the transpose of the
        unpacked triangle, and its elementwise conjugate for a Hermitian line."""
        rng = np.random.default_rng(29)
        for dim in (2, 3, 4, 6, 9):
            amplitudes = rng.normal(size=dim) + 1j * rng.normal(size=dim)
            hermitian = np.outer(amplitudes, np.conj(amplitudes))
            np.fill_diagonal(hermitian, hermitian.diagonal().real)
            line = _upper_triangle(hermitian)
            unpacked = fx.unpack_density_triangle(line)
            observables = density_functions.DensityMatrixObservables(line)
            square = np.asarray(observables.square_matrix())

            np.testing.assert_array_equal(square, unpacked.T)
            np.testing.assert_array_equal(square, np.conj(unpacked))
            self.assertGreater(np.max(np.abs(square - unpacked)), 1.0e-6)

    def test_unpacked_triangle_keeps_stored_entries_up_to_len_tri(self):
        """The upper triangle, diagonal included, is the stored line, the
        lower triangle its conjugate, and entries past len_tri are unused."""
        rng = np.random.default_rng(31)
        for dim in (1, 2, 3, 9, 27):
            size = dim * (dim + 1) // 2
            line = rng.normal(size=size) + 1j * rng.normal(size=size)
            extra = rng.normal(size=7) + 1j * rng.normal(size=7)

            unpacked = fx.unpack_density_triangle(line)
            truncated = fx.unpack_density_triangle(np.concatenate([line, extra]), size)

            self.assertEqual(unpacked.shape, (dim, dim))
            self.assertEqual(unpacked.dtype, np.complex128)
            np.testing.assert_array_equal(unpacked[np.triu_indices(dim)], line)
            lower = np.tril_indices(dim, -1)
            np.testing.assert_array_equal(unpacked[lower], np.conj(unpacked.T[lower]))
            self.assertEqual(truncated.tobytes(), unpacked.tobytes())


def _density(amplitudes, pdgs):
    """DensityMatrix built from amplitudes through the packed-triangle path."""
    line = _upper_triangle(np.outer(amplitudes, np.conj(amplitudes)))
    matrix = fx.unpack_density_triangle(line)
    return fx.DensityMatrix.from_full_matrix(matrix, pdgs)


class TestDensityContraction(unittest.TestCase):
    """DensityMatrix.contract and the density weight against amplitude-level
    expressions: B[h,h'] = M_h M*_h', C[h,h'] = D_h D*_h' and the coherent
    rate |sum_h M_h D_h|^2."""

    def test_pure_states_give_squared_amplitude(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            prod_amp = rng.normal(size=3) + 1j * rng.normal(size=3)
            decay_amp = rng.normal(size=3) + 1j * rng.normal(size=3)
            got = _density(prod_amp, [24]).contract(_density(decay_amp, [24]))
            want = abs(np.sum(prod_amp * decay_amp)) ** 2
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    def test_two_resonance_tensor_product(self):
        # Production index h = 3*h1 + h2 (first resonance varies slowest),
        # matching the itertools.product helicity enumeration and the np.kron
        # of build_decay_tensor_product.
        rng = np.random.default_rng(5)
        for _ in range(50):
            prod_amp = rng.normal(size=9) + 1j * rng.normal(size=9)
            decay1 = rng.normal(size=3) + 1j * rng.normal(size=3)
            decay2 = rng.normal(size=3) + 1j * rng.normal(size=3)
            decay_c = fx.build_decay_tensor_product(
                [_density(decay1, [24]), _density(decay2, [-24])]
            )
            got = _density(prod_amp, [24, -24]).contract(decay_c)
            want = abs(np.sum(prod_amp * np.kron(decay1, decay2))) ** 2
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    def test_sudakov_weight_matches_corrected_amplitudes(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            prod_amp = rng.normal(size=9) + 1j * rng.normal(size=9)
            decay_amp = rng.normal(size=9) + 1j * rng.normal(size=9)
            delta = 1.0e-4 * (rng.normal(size=9) + 1j * rng.normal(size=9))
            weight = fx.compute_density_weight(
                _density(prod_amp, [24, -24]), _density(decay_amp, [24, -24]), delta
            )
            born = abs(np.sum(prod_amp * decay_amp)) ** 2
            linear = np.sum(
                np.outer(prod_amp, np.conj(prod_amp))
                * (1.0 + delta[:, None] + np.conj(delta)[None, :])
                * np.outer(decay_amp, np.conj(decay_amp))
            ).real
            np.testing.assert_allclose(weight, linear / born, rtol=1e-12)
            # the linearised update is the full square minus |sum M delta D|^2
            corrected = abs(np.sum(prod_amp * (1.0 + delta) * decay_amp)) ** 2
            quadratic = abs(np.sum(prod_amp * delta * decay_amp)) ** 2
            np.testing.assert_allclose(weight * born + quadratic, corrected, rtol=1e-12)

    def test_azimuthal_mirror_pair(self):
        # Real production amplitudes (production plane as the xz plane) and a
        # decay configuration together with its mirror image through that
        # plane (conjugated decay amplitudes): the weight of each must follow
        # its own corrected amplitude.
        rng = np.random.default_rng(11)
        prod_amp = rng.normal(size=3).astype(complex)
        decay_amp = rng.normal(size=3) + 1j * rng.normal(size=3)
        delta = np.array([0.02 - 0.03j, 0.01 + 0.04j, -0.02 + 0.01j])
        prod = _density(prod_amp, [24])
        for amp in (decay_amp, np.conj(decay_amp)):
            weight = fx.compute_density_weight(prod, _density(amp, [24]), delta)
            linear = np.sum(
                np.outer(prod_amp, np.conj(prod_amp))
                * (1.0 + delta[:, None] + np.conj(delta)[None, :])
                * np.outer(amp, np.conj(amp))
            ).real
            np.testing.assert_allclose(
                weight, linear / abs(np.sum(prod_amp * amp)) ** 2, rtol=1e-12
            )
        mirror_difference = fx.compute_density_weight(
            prod, _density(decay_amp, [24]), delta
        ) - fx.compute_density_weight(prod, _density(np.conj(decay_amp), [24]), delta)
        self.assertGreater(abs(mirror_difference), 1.0e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
