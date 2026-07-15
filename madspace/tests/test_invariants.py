import numpy as np
import pytest
from pytest import approx

import madspace as ms

COUNT = 10000
SQRT_S_MAX = 13000.0


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def s_min(rng):
    return rng.uniform(173.0, SQRT_S_MAX, COUNT) ** 2


@pytest.fixture
def s_max(rng, s_min):
    return rng.uniform(np.sqrt(s_min), SQRT_S_MAX, COUNT) ** 2


@pytest.fixture
def r_in(rng):
    return rng.random(COUNT)


@pytest.fixture(
    params=[
        {},
        {"mass": 173.0, "width": 1.4},
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
    np.testing.assert_array_less(s_min, s)


def test_invariant_max(invariant, r_in, s_min, s_max):
    s, det = invariant.map_forward([r_in], [s_min, s_max])
    np.testing.assert_array_less(s, s_max)


def test_invariant_finite(invariant, r_in, s_min, s_max):
    s, det = invariant.map_forward([r_in], [s_min, s_max])
    assert np.all(np.isfinite(s))
    assert np.all(np.isfinite(det))


def test_invariant_inverse(invariant, r_in, s_min, s_max):
    s, det = invariant.map_forward([r_in], [s_min, s_max])
    r_out, det_inv = invariant.map_inverse([s], [s_min, s_max])
    assert r_out == approx(r_in, abs=1e-4)
    assert det_inv == approx(1 / det)
