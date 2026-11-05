import pytest
import torch
from madspace_testing import generator, rand
from pytest import approx

import madspace as ms

COUNT = 10000
SQRT_S_MAX = 13000.0


@pytest.fixture
def rng():
    return generator(1234)


def uniform(rng, low, high):
    return low + (high - low) * rand(COUNT, generator=rng)


@pytest.fixture
def s_min(rng):
    return uniform(rng, 173.0, SQRT_S_MAX) ** 2


@pytest.fixture
def s_max(rng, s_min):
    return uniform(rng, torch.sqrt(s_min), SQRT_S_MAX) ** 2


@pytest.fixture
def r_in(rng):
    return rand(COUNT, generator=rng)


@pytest.fixture(
    params=[
        {},
        {"mass": 173.0, "width": 1.4},
        {"mass": 173.0, "width": 1.4, "flat_window": 15.0},
        {"mass": 173.0, "power": 1.5},
        {"mass": 173.0, "power": 1.0},
        {"mass": 173.0, "power": 0.5},
        {"power": 1.5},
        {"power": 1.0},
        {"power": 0.5},
    ],
    ids=[
        "uniform",
        "breit wigner",
        "flattened breit wigner",
        "massive, power=1.5",
        "massive, power=1.0",
        "massive, power=0.5",
        "massless, power=1.5",
        "massless, power=1.0",
        "massless, power=0.5",
    ],
)
def invariant(request):
    return ms.Invariant(**request.param)


def test_invariant_min(invariant, r_in, s_min, s_max):
    s, det = invariant.map_forward([r_in], [s_min, s_max])
    assert torch.all(s_min < s)


def test_invariant_max(invariant, r_in, s_min, s_max):
    s, det = invariant.map_forward([r_in], [s_min, s_max])
    assert torch.all(s < s_max)


def test_invariant_finite(invariant, r_in, s_min, s_max):
    s, det = invariant.map_forward([r_in], [s_min, s_max])
    assert torch.all(torch.isfinite(s))
    assert torch.all(torch.isfinite(det))


def test_invariant_inverse(invariant, r_in, s_min, s_max):
    s, det = invariant.map_forward([r_in], [s_min, s_max])
    r_out, det_inv = invariant.map_inverse([s], [s_min, s_max])
    # the tolerances of pytest.approx
    torch.testing.assert_close(r_out, r_in, rtol=1e-6, atol=1e-4)
    torch.testing.assert_close(det_inv, 1 / det, rtol=1e-6, atol=1e-12)


def test_flat_window_breit_wigner():
    """The flattened Breit-Wigner of a $-excluded propagator: Breit-Wigner
    density outside |sqrt(s) - m| < w * width, a constant inside, and a
    normalised mapping."""
    mass, width, window = 91.188, 2.4414, 15.0
    n = 200000
    r_in = (torch.arange(n) + 0.5) / n  # midpoint grid: the mean below is a quadrature
    s_min = torch.full((n,), 20.0**2)
    s_max = torch.full((n,), 300.0**2)
    s, det = ms.Invariant(0.8, mass, width, window).map_forward([r_in], [s_min, s_max])
    # det = ds/dr, whose mean over r is the length of the range
    assert det.mean().item() == approx((s_max[0] - s_min[0]).item(), rel=1e-4)
    lo, hi = (mass - window * width) ** 2, (mass + window * width) ** 2
    inside = (s > lo) & (s < hi)
    # constant density in the window
    det_in = det[inside]
    ptp_in = (det_in.max() - det_in.min()).item()
    assert ptp_in == approx(0.0, abs=1e-6 * det_in.mean().item())
    # Breit-Wigner outside: det * BW is the same constant everywhere
    bw = 1.0 / ((s - mass**2) ** 2 + (mass * width) ** 2)
    outside = det[~inside] * bw[~inside]
    ptp_out = (outside.max() - outside.min()).item()
    assert ptp_out == approx(0.0, abs=1e-6 * outside.mean().item())
    # many more points outside the window, where a $-excluded matrix element
    # is nonzero, than a plain Breit-Wigner puts there (1.2% -> 34% here)
    s_bw, _ = ms.Invariant(0.8, mass, width).map_forward([r_in], [s_min, s_max])
    outside_bw = ((s_bw < lo) | (s_bw > hi)).double().mean()
    assert (~inside).double().mean() > 10 * outside_bw
