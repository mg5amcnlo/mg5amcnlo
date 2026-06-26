from collections import namedtuple

import numpy as np
import pytest
from pytest import approx

import madspace as ms

# ----------------------------
# Utilities
# ----------------------------


def inv_mass_sq(p):
    # p: (..., 4) with [E, px, py, pz]
    return p[..., 0] ** 2 - np.sum(p[..., 1:] ** 2, axis=-1)


def mass(p):
    return np.sqrt(np.maximum(inv_mass_sq(p), 0.0))


def boost(k: np.ndarray, p_boost: np.ndarray, inverse: bool = False) -> np.ndarray:
    # Change sign if inverse boost is performed
    sign = -1.0 if inverse else 1.0
    # Perform the boost
    EPS = 1e-16
    rsq = np.clip(mass(p_boost), min=EPS)
    kp = np.sum(k[..., 1:] * p_boost[..., 1:], axis=-1)
    k0 = (k[..., 0] * p_boost[..., 0] + sign * kp) / rsq
    c1 = (k[..., 0] + k0) / (rsq + p_boost[..., 0])
    k1 = k[..., 1] + sign * c1 * p_boost[..., 1]
    k2 = k[..., 2] + sign * c1 * p_boost[..., 2]
    k3 = k[..., 3] + sign * c1 * p_boost[..., 3]

    return np.stack((k0, k1, k2, k3), axis=-1)


# ----------------------------
# Fixtures
# ----------------------------

N = 10_000  # keep this moderate for CI speed; bump locally for tighter stats

InputPoint = namedtuple(
    "InputPoint",
    [
        "r_choice",
        "r_s23",
        "r_t1",
        "m0",
        "m12",
        "m1",
        "m2",
        "m3",
        "pa",
        "pb",
        "p0",
        "p3",
        "p12",
    ],
)

ZEROS = np.zeros(N)
PA = np.full((N, 4), [np.sqrt(100**2 + 200**2 + 300**2), 100.0, 200.0, 300.0])
PB = np.full((N, 4), [np.sqrt(100**2 + 200**2 + 400**2), 100.0, 200.0, -400.0])
P0 = PA + PB
M0 = mass(P0)
M1 = np.full(N, 10.0)
M2 = np.full(N, 20.0)
M3 = np.full(N, 30.0)
M12 = np.full(N, 200.0)


@pytest.fixture(
    params=[
        (M0, M12, ZEROS, ZEROS, M3, PA, PB, P0),
        (M0, M12, M1, ZEROS, M3, PA, PB, P0),
        (M0, M12, M1, M2, M3, PA, PB, P0),
    ],
    ids=[
        "both massless",
        "one massive",
        "both massive",
    ],
)
def fixed_input_points(rng, request):
    """
    Generate (pa, pb) and a valid 'spectator' p3 by first producing a 2->2 scattering.
    We then use pa, pb, and that p3 as conditions for the 2->3 peel-off.
    """
    m0, m12, m1, m2, m3, pA, pB, p0 = request.param

    map_22 = ms.TwoToTwoParticleScattering(com=False)
    r1 = rng.random(N)
    r2 = rng.random(N)
    p12, p3, det_22 = map_22.map_forward([r1, r2, m12, m3], [pA, pB])

    # Randoms for the 2->3 mapper
    r_choice = rng.random(N)
    r_s23 = rng.random(N)
    r_t1 = rng.random(N)

    return InputPoint(r_choice, r_s23, r_t1, m0, m12, m1, m2, m3, pA, pB, p0, p3, p12)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture(params=[False, True], ids=["base 2->2, LAB", "base 2->2, COM"])
def input_points(rng, request):
    """
    Generate (pa, pb) and a valid 'spectator' p3 by first producing a 2->2 scattering.
    We then use pa, pb, and that p3 as conditions for the 2->3 peel-off.
    """
    com = request.param

    # start with compatible incoming momenta
    # COM: pa=(E,0,0,+p), pb=(E,0,0,-p)
    pz = rng.uniform(1000.0, 4000.0, N)
    ma = rng.uniform(50.0, 300.0, N)
    mb = rng.uniform(200.0, 500.0, N)
    e1 = np.sqrt(pz**2 + ma**2)
    e2 = np.sqrt(pz**2 + mb**2)
    pa = np.stack([e1, np.zeros(N), np.zeros(N), +pz], axis=1)
    pb = np.stack([e2, np.zeros(N), np.zeros(N), -pz], axis=1)

    if not com:
        # boost to a some random frame
        pa = np.array([[0, 0, 4000.0]]) + rng.normal(0.0, 500.0, (N, 3))
        ea = np.sqrt(np.sum(pa**2, axis=1) + ma**2)
        pa = np.concatenate([ea[:, None], pa], axis=1)

        pb = np.array([[0, 0, -4000.0]]) + rng.normal(0.0, 500.0, (N, 3))
        eb = np.sqrt(np.sum(pb**2, axis=1) + mb**2)
        pb = np.concatenate([eb[:, None], pb], axis=1)

    # Build a valid 2->2 event to extract p3 as a spectator
    # Choose arbitrary outgoing masses for (Q, p3): make p3 light, Q heavyish
    m3 = rng.uniform(1.0, 10.0, N)
    m1 = rng.uniform(1.0, 40.0, N)
    m2 = rng.uniform(1.0, 40.0, N)
    m12 = rng.uniform(160, 400, N)

    map_22 = ms.TwoToTwoParticleScattering(com=com)
    r1 = rng.random(N)
    r2 = rng.random(N)
    p12, p3, det_22 = map_22.map_forward([r1, r2, m12, m3], [pa, pb])

    # Randoms for the 2->3 mapper
    r_choice = rng.random(N)  # decide branch (emitter choice)
    r_s23 = rng.random(N)
    r_t1 = rng.random(N)

    p0 = pa + pb
    m0 = mass(p0)

    return InputPoint(r_choice, r_s23, r_t1, m0, m12, m1, m2, m3, pa, pb, p0, p3, p12)


# ----------------------------
# Tests
# ----------------------------


def test_momentum_conservation(input_points):
    mapping = ms.TwoToThreeParticleScattering()

    inputs = [
        input_points.r_choice,
        input_points.r_s23,
        input_points.r_t1,
        input_points.m1,
        input_points.m2,
    ]
    conditions = [input_points.pa, input_points.pb, input_points.p3]

    m3 = mass(input_points.p3)
    p1, p2, det = mapping.map_forward(inputs, conditions)
    p_sum = p1 + p2 + input_points.p3

    assert p_sum == approx(input_points.p0)
    assert p1 + p2 == approx(input_points.p12)


def test_inverse(input_points):
    mapping = ms.TwoToThreeParticleScattering()

    inputs = [
        input_points.r_choice,
        input_points.r_s23,
        input_points.r_t1,
        input_points.m1,
        input_points.m2,
    ]
    conditions = [input_points.pa, input_points.pb, input_points.p3]

    p1, p2, det = mapping.map_forward(inputs, conditions)
    *inv_inputs, inv_det = mapping.map_inverse([p1, p2], conditions)

    assert inv_det == approx(1 / det, rel=1e-5)

    for i, (inp, inv_inp) in enumerate(zip(inputs, inv_inputs)):
        if i == 0:
            assert ((inp < 0.5) == (inv_inp < 0.5)).all()
            continue
        assert inp == approx(inv_inp), f"mismatch in input index {i}"


def test_on_shell_masses(input_points):
    mapping = ms.TwoToThreeParticleScattering()

    inputs = [
        input_points.r_choice,
        input_points.r_s23,
        input_points.r_t1,
        input_points.m1,
        input_points.m2,
    ]
    conditions = [input_points.pa, input_points.pb, input_points.p3]

    p1, p2, det = mapping.map_forward(inputs, conditions)

    # Outgoing masses must match m1, m2; spectator stays whatever it was.
    assert mass(p1 + p2) == approx(input_points.m12)
    assert mass(p1) == approx(input_points.m1)
    assert mass(p2) == approx(input_points.m2)


def test_phase_space_compare(rng, input_points):
    mapping23 = ms.TwoToThreeParticleScattering()
    mapping22 = ms.TwoToTwoParticleScattering(com=False)
    r1 = rng.random(N)
    r2 = rng.random(N)

    inputs = [
        input_points.r_choice,
        input_points.r_s23,
        input_points.r_t1,
        input_points.m1,
        input_points.m2,
    ]
    inputs22 = [
        r1,
        r2,
        input_points.m1,
        input_points.m2,
    ]
    conditions = [input_points.pa, input_points.pb, input_points.p3]
    conditions22 = [input_points.pa, input_points.pb - input_points.p3]

    p1, p2, det23 = mapping23.map_forward(inputs, conditions)
    p1s, p2s, det22 = mapping22.map_forward(inputs22, conditions22)

    # Outgoing masses must match m1, m2; spectator stays whatever it was.
    std_error_23 = np.std(det23) / np.sqrt(N)
    assert np.mean(det23) == approx(np.mean(det22), abs=3 * std_error_23, rel=1e-6)


def test_phase_space_volume(fixed_input_points):
    mapping23 = ms.TwoToThreeParticleScattering()

    inputs = [
        fixed_input_points.r_choice,
        fixed_input_points.r_s23,
        fixed_input_points.r_t1,
        fixed_input_points.m1,
        fixed_input_points.m2,
    ]

    conditions = [fixed_input_points.pa, fixed_input_points.pb, fixed_input_points.p3]

    p1, p2, det = mapping23.map_forward(inputs, conditions)

    s = fixed_input_points.m12**2
    m1_2 = fixed_input_points.m1**2
    m2_2 = fixed_input_points.m2**2
    phase_space_vol = (
        np.pi / (2 * s) * np.sqrt((s - m1_2 - m2_2) ** 2 - 4 * m1_2 * m2_2)
    )
    std_error = np.std(det) / np.sqrt(N)
    assert np.mean(det) == approx(phase_space_vol, abs=3 * std_error, rel=1e-6)
