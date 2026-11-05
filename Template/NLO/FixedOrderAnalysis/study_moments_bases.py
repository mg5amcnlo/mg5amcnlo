#!/usr/bin/env python3
"""Reproducible basis-choice diagnostics, independent of production analyses.

Requires NumPy. Run without arguments for JSON on stdout, or use --out FILE.
This studies density-ratio reconstruction r=f/q; its loss is integral q*(r-rhat)^2.
It does not add a Chebyshev or reference-weighted basis to the Fortran backend.

Equal polynomial span + equal projection measure + proper Gram inversion give
the same linear estimator and covariance for ANY fixed Monte Carlo sample.
Changing the orthogonality measure changes the projection and its noise.
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from numpy.polynomial.legendre import leggauss, legvander


def legendre(x, modes):
    return legvander(x, modes - 1) * np.sqrt(2 * np.arange(modes) + 1)


def chebyshev_orthonormal(x, modes):
    scale = np.ones(modes)
    scale[1:] = math.sqrt(2)
    return chebvander(x, modes - 1) * scale


def target_ratio(x):
    return np.exp(0.7 * x) * (1 + 0.12 * np.cos(7 * x))


def loss(prediction, truth, quadrature_weights):
    return float(np.dot(quadrature_weights, (prediction - truth) ** 2))


def mean_covariance(contributions):
    """Covariance of the sample mean, including correlations between modes."""
    mean = contributions.mean(axis=0)
    centered = contributions - mean
    size = len(contributions)
    return mean, centered.T @ centered / (size * (size - 1))


def study_polynomial_coordinates(rng, samples, trials, modes):
    nodes, quadrature = leggauss(max(128, 4 * modes))
    quadrature = quadrature / 2
    truth = target_ratio(nodes)
    legendre_grid = legendre(nodes, modes)
    chebyshev_grid = chebvander(nodes, modes - 1)
    gram = chebyshev_grid.T @ (quadrature[:, None] * chebyshev_grid)
    inverse_gram = np.linalg.inv(gram)

    # Comparing kernels at several points checks off-diagonal covariance too.
    check = np.linspace(-1, 1, 19)
    check_legendre = legendre(check, modes)
    check_chebyshev = chebvander(check, modes - 1)
    reconstruction_difference = 0.0
    covariance_difference = 0.0
    risks_legendre, risks_chebyshev = [], []
    legendre_time = chebyshev_time = 0.0
    for trial in range(trials):
        x = rng.uniform(-1, 1, samples)
        ratio = target_ratio(x)
        started = time.perf_counter()
        z_legendre = ratio[:, None] * legendre(x, modes)
        a_legendre, cov_legendre = mean_covariance(z_legendre)
        legendre_time += time.perf_counter() - started

        started = time.perf_counter()
        z_chebyshev = ratio[:, None] * chebvander(x, modes - 1)
        raw_chebyshev, raw_covariance = mean_covariance(z_chebyshev)
        a_chebyshev = inverse_gram @ raw_chebyshev
        cov_chebyshev = inverse_gram @ raw_covariance @ inverse_gram.T
        chebyshev_time += time.perf_counter() - started

        pred_legendre = legendre_grid @ a_legendre
        pred_chebyshev = chebyshev_grid @ a_chebyshev
        reconstruction_difference = max(
            reconstruction_difference,
            float(np.max(np.abs(pred_legendre - pred_chebyshev))),
        )
        if trial == 0:
            kernel_legendre = check_legendre @ cov_legendre @ check_legendre.T
            kernel_chebyshev = (
                check_chebyshev @ cov_chebyshev @ check_chebyshev.T
            )
            covariance_difference = float(
                np.max(np.abs(kernel_legendre - kernel_chebyshev))
            )
        risks_legendre.append(loss(pred_legendre, truth, quadrature))
        risks_chebyshev.append(loss(pred_chebyshev, truth, quadrature))

    scale = max(1.0, float(np.max(np.abs(truth))))
    if reconstruction_difference > 1e-9 * scale:
        raise AssertionError("Equal-span reconstruction invariance failed")
    if covariance_difference > 1e-9:
        raise AssertionError("Equal-span covariance invariance failed")

    return {
        "measure": "q(x) dx = dx/2 on [-1,1]",
        "estimator": "rhat = basis(x) @ Gram^{-1} @ mean[r(X)*basis(X)]",
        "same_samples": True,
        "modes": modes,
        "max_reconstruction_difference": reconstruction_difference,
        "max_covariance_kernel_difference": covariance_difference,
        "chebyshev_gram_condition_number": float(np.linalg.cond(gram)),
        "mean_integrated_squared_error": {
            "legendre": float(np.mean(risks_legendre)),
            "chebyshev_with_gram_inverse": float(np.mean(risks_chebyshev)),
        },
        "estimator_seconds_excluding_gram_setup": {
            "legendre": legendre_time,
            "chebyshev_with_gram_inverse": chebyshev_time,
        },
        "interpretation": (
            "Same polynomial span and projection measure give identical "
            "reconstruction and propagated covariance. Timings depend on the "
            "machine and implementation; they do not establish a universal "
            "advantage for either polynomial coordinate system."
        ),
    }


def study_changed_measure(rng, samples, trials, modes):
    nodes, quadrature = leggauss(max(128, 4 * modes))
    quadrature /= 2
    truth = target_ratio(nodes)
    phi_grid = chebyshev_orthonormal(nodes, modes)
    risks_uniform, risks_arcsine = [], []
    uniform_time = arcsine_time = 0.0
    for _ in range(trials):
        # Estimate the SAME arcsine-measure projection with two proposals.
        x = rng.uniform(-1, 1, samples)
        started = time.perf_counter()
        importance = 2 / (math.pi * np.sqrt(1 - x * x))
        coefficient = np.mean(
            (target_ratio(x) * importance)[:, None]
            * chebyshev_orthonormal(x, modes),
            axis=0,
        )
        uniform_time += time.perf_counter() - started
        risks_uniform.append(loss(phi_grid @ coefficient, truth, quadrature))

        theta = rng.uniform(0, math.pi, samples)
        x = np.cos(theta)
        started = time.perf_counter()
        coefficient = np.mean(
            target_ratio(x)[:, None] * chebyshev_orthonormal(x, modes), axis=0
        )
        arcsine_time += time.perf_counter() - started
        risks_arcsine.append(loss(phi_grid @ coefficient, truth, quadrature))

    return {
        "projection_measure": "dx/(pi*sqrt(1-x^2)) on [-1,1]",
        "evaluation_loss_measure": "dx/2, matching the preceding study",
        "uniform_proposal_importance": "2/(pi*sqrt(1-X^2))",
        "theoretical_coefficient_variance_under_uniform_sampling": "infinite",
        "reason": (
            "The target ratio and each normalized Chebyshev test have nonzero "
            "endpoint limits. Squared importance weights give a logarithmically "
            "divergent integral proportional to integral dx/(1-x^2)."
        ),
        "finite_trial_mean_integrated_squared_error": {
            "uniform_samples_with_arcsine_importance": float(np.mean(risks_uniform)),
            "arcsine_samples": float(np.mean(risks_arcsine)),
        },
        "estimator_seconds": {
            "uniform_samples_with_arcsine_importance": uniform_time,
            "arcsine_samples": arcsine_time,
        },
        "interpretation": (
            "Finite-trial errors with uniform sampling are diagnostics, not "
            "finite expected MSE estimates. Arcsine sampling changes the "
            "proposal; it is not a free basis change in an existing phase-space "
            "integrator. Neither experiment compares identical projection "
            "measures with the first study. A density envelope equal to the "
            "arcsine density is a further distinct construction."
        ),
    }


def study_reference_envelope(samples, beta, extent):
    """Compare two bases with the SAME truncated exponential reference q."""
    maximum_modes = 32
    nodes, weights = leggauss(384)
    z = extent * (nodes + 1) / 2
    normalization = -math.expm1(-extent)
    qweights = weights * extent / 2 * np.exp(-z) / normalization
    ratio = 1 + beta * z
    mu = float(np.dot(qweights, z))
    variance = float(np.dot(qweights, (z - mu) ** 2))
    sigma = math.sqrt(variance)

    physical_basis = np.column_stack((np.ones_like(z), (z - mu) / sigma))
    physical_coefficients = np.array((1 + beta * mu, beta * sigma))
    physical_error = loss(physical_basis @ physical_coefficients, ratio, qweights)
    physical_covariance = (
        physical_basis.T @ ((qweights * ratio**2)[:, None] * physical_basis)
        - np.outer(physical_coefficients, physical_coefficients)
    ) / samples

    # CDF computed stably on the finite physical interval [0,extent].
    t = 2 * (-np.expm1(-z) / normalization) - 1
    cdf_basis = legendre(t, maximum_modes)
    coefficients = cdf_basis.T @ (qweights * ratio)
    second_moments = cdf_basis.T @ (
        (qweights * ratio**2)[:, None] * cdf_basis
    )
    covariance = (second_moments - np.outer(coefficients, coefficients)) / samples
    rows = []
    for modes in (2, 4, 8, 16, 32):
        bias = loss(cdf_basis[:, :modes] @ coefficients[:modes], ratio, qweights)
        statistical_risk = float(np.trace(covariance[:modes, :modes]))
        endpoint_basis = np.sqrt(2 * np.arange(modes) + 1)
        endpoint = float(endpoint_basis @ coefficients[:modes])
        endpoint_truth = 1 + beta * extent
        endpoint_variance = float(
            endpoint_basis @ covariance[:modes, :modes] @ endpoint_basis
        )
        rows.append({
            "modes": modes,
            "exact_projection_squared_bias": bias,
            "expected_integrated_statistical_variance": statistical_risk,
            "expected_total_risk": bias + statistical_risk,
            "upper_endpoint_relative_bias": endpoint / endpoint_truth - 1,
            "upper_endpoint_relative_statistical_sigma": (
                math.sqrt(max(0, endpoint_variance)) / endpoint_truth
            ),
        })

    physical_endpoint = np.array((1.0, (extent - mu) / sigma))
    physical_statistical_risk = float(np.trace(physical_covariance))
    k = np.arange(1, 9)
    return {
        "physical_interval": [0, extent],
        "reference": "q(z)=exp(-z)/(1-exp(-extent))",
        "target": "f=q*(1+beta*z); f is an unnormalized cross-section density",
        "beta": beta,
        "proposal_for_variance_calculation": "the same q for both bases",
        "loss_for_both_bases": "integral q*(fhat/q-f/q)^2 dz",
        "physical_reference_weighted_polynomials": {
            "modes": 2,
            "basis": ["1", "(z-E_q[z])/sqrt(Var_q[z])"],
            "coefficients": physical_coefficients.tolist(),
            "exact_projection_squared_bias": physical_error,
            "expected_integrated_statistical_variance": physical_statistical_risk,
            "expected_total_risk": physical_error + physical_statistical_risk,
            "upper_endpoint_relative_bias": 0.0,
            "upper_endpoint_relative_statistical_sigma": float(
                math.sqrt(max(0, physical_endpoint @ physical_covariance
                              @ physical_endpoint)) / (1 + beta * extent)
            ),
        },
        "cdf_legendre": rows,
        "half_line_analytic_result": {
            "physical_basis": "Laguerre: a0=1+beta, a1=-beta, higher=0",
            "cdf_transformed_ratio": "1-beta*log((1-t)/2)",
            "cdf_coefficients_k_ge_1": "beta*sqrt(2*k+1)/(k*(k+1))",
            "decay": "k^(-3/2), algebraic rather than exponential",
            "first_eight_nonconstant_coefficients": (
                beta * np.sqrt(2 * k + 1) / (k * (k + 1))
            ).tolist(),
        },
        "interpretation": (
            "Envelope and physical-coordinate basis choice can matter more "
            "than polynomial naming. Both finite-interval bases use the same "
            "reference and loss. The physical basis resolves this particular "
            "linear correction exactly; it is not universally superior, and "
            "growing physical polynomials can amplify Monte Carlo variance. "
            "Variance results use independent samples from q, not an MG5 "
            "phase-space integration or a model of subtraction groups."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--extent", type=float, default=12.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.samples < 2 or args.trials < 1 or not 2 <= args.modes <= 40:
        parser.error("samples>=2, trials>=1 and 2<=modes<=40 are required")
    if not math.isfinite(args.beta) or args.beta < 0:
        parser.error("beta must be finite and nonnegative")
    if not math.isfinite(args.extent) or not 0.1 <= args.extent <= 30:
        parser.error("extent must lie between 0.1 and 30")
    rng = np.random.default_rng(args.seed)
    result = {
        "study": "orthogonal moment bases; research diagnostic, not production",
        "seed": args.seed,
        "samples_per_trial": args.samples,
        "trials": args.trials,
        "same_measure_coordinate_invariance": study_polynomial_coordinates(
            rng, args.samples, args.trials, args.modes
        ),
        "changed_chebyshev_projection_measure": study_changed_measure(
            rng, args.samples, args.trials, args.modes
        ),
        "finite_exponential_reference": study_reference_envelope(
            args.samples, args.beta, args.extent
        ),
    }
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
