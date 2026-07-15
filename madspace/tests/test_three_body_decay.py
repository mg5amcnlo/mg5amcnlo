from collections import namedtuple

import numpy as np
import pytest
from pytest import approx

import madspace as ms


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture(
    params=[
        {"decay": True, "com": False},
        {"decay": True, "com": True},
    ],
    ids=[
        "1->3 decay",
        "1->3 decay, COM",
    ],
)
def mapping_and_args(request):
    com = request.param["com"]
    zeros = np.zeros(N)
    if request.param["decay"]:
        mapping = ms.ThreeBodyDecay(com=com)
        if com:

            def make_args(point):
                p0 = np.stack([point.m0, zeros, zeros, zeros], axis=1)
                return (
                    [
                        point.r1,
                        point.r2,
                        point.r3,
                        point.r4,
                        point.r5,
                        point.m0,
                        point.m1,
                        point.m2,
                        point.m3,
                    ],
                    [],
                    p0,
                )

        else:

            def make_args(point):
                return (
                    [
                        point.r1,
                        point.r2,
                        point.r3,
                        point.r4,
                        point.r5,
                        point.m0,
                        point.m1,
                        point.m2,
                        point.m3,
                        point.p0,
                    ],
                    [],
                    point.p0,
                )

    else:
        raise NotImplementedError("Only 1->3 decay is implemented in this test.")
    return (mapping, make_args, com)


def mass(momentum):
    return np.sqrt(momentum[:, 0] ** 2 - np.sum(momentum[:, 1:] ** 2, axis=1))


InputPoint = namedtuple(
    "InputPoint",
    ["r1", "r2", "r3", "r4", "r5", "m0", "m1", "m2", "m3", "p0", "pa", "pb"],
)

N = 10000


@pytest.fixture
def input_points(rng):
    max_mass = 125.0
    r1 = rng.random(N)
    r2 = rng.random(N)
    r3 = rng.random(N)
    r4 = rng.random(N)
    r5 = rng.random(N)

    pa = np.array([[0, 0, 4000.0]]) + rng.normal(0.0, 500.0, (N, 3))
    ma = rng.uniform(2.5 * max_mass, 1000.0, N)
    ea = np.sqrt(np.sum(pa**2, axis=1) + ma**2)
    pa = np.concatenate([ea[:, None], pa], axis=1)

    pb = np.array([[0, 0, -4000.0]]) + rng.normal(0.0, 500.0, (N, 3))
    mb = rng.uniform(20.0, 500.0, N)
    eb = np.sqrt(np.sum(pb**2, axis=1) + mb**2)
    pb = np.concatenate([eb[:, None], pb], axis=1)

    p0 = pa + pb
    m0 = mass(p0)

    m1 = rng.uniform(1.0, max_mass, N)
    m2 = rng.uniform(1.0, max_mass, N)
    m3 = rng.uniform(1.0, max_mass, N)

    return InputPoint(r1, r2, r3, r4, r5, m0, m1, m2, m3, p0, pa, pb)


def test_momentum_conservation(mapping_and_args, input_points):
    mapping, make_args, com = mapping_and_args
    inputs, conditions, p0 = make_args(input_points)
    p1, p2, p3, det = mapping.map_forward(inputs, conditions)
    assert p1 + p2 + p3 == approx(p0)


def test_inverse(mapping_and_args, input_points):
    mapping, make_args, com = mapping_and_args
    inputs, conditions, p0 = make_args(input_points)
    p1, p2, p3, det = mapping.map_forward(inputs, conditions)
    *inv_inputs, inv_det = mapping.map_inverse([p1, p2, p3], conditions)
    assert inv_det == approx(1 / det)
    for i, (inp, inv_inp) in enumerate(zip(inputs, inv_inputs)):
        assert inp == approx(inv_inp), f"Mismatch at index {i}: {inp} vs {inv_inp}"


def test_outgoing_masses(mapping_and_args, input_points):
    mapping, make_args, com = mapping_and_args
    inputs, conditions, p0 = make_args(input_points)
    p1, p2, p3, det = mapping.map_forward(inputs, conditions)
    m0 = mass(p1 + p2 + p3)
    m1 = mass(p1)
    m2 = mass(p2)
    m3 = mass(p3)
    index = 5
    print(p3[index])
    print(m3[index], input_points.m3[index])
    assert m0 == approx(input_points.m0)
    assert m1 == approx(input_points.m1)
    assert m2 == approx(input_points.m2)
    assert m3 == approx(input_points.m3)


def test_phase_space_via_two_2body(mapping_and_args, rng):
    mapping_13, make_args, com = mapping_and_args  # your ThreeBodyDecay
    N = 50000
    r = rng.random((N, 5))  # 5 random numbers for 1->3 decay
    u = r[:, 0]  # first random number for s12

    # Fix parent in COM for clarity (or use your fixture’s p0)
    m0, m1, m2, m3 = 120.0, 10.0, 15.0, 5.0
    zeros = np.zeros(N)
    p0 = np.stack([np.full(N, m0), zeros, zeros, zeros], axis=1)

    smin = (m1 + m2) ** 2
    smax = (m0 - m3) ** 2
    s12 = smin + u * (smax - smin)
    mQ = np.sqrt(s12)

    # --- First 2-body: 0 -> Q + 3 ---
    mapping1 = ms.TwoBodyDecay(com=True)
    inputs1 = [r[:, 1], r[:, 2], np.full(N, m0), np.full(N, mQ), np.full(N, m3)]
    pQ, p3, det1 = mapping1.map_forward(inputs1, [])

    # Ensure pQ has invariant mass sqrt(s12)
    assert approx(np.sqrt(pQ[:, 0] ** 2 - (pQ[:, 1:] ** 2).sum(1))) == mQ

    # --- Second 2-body: Q -> 1 + 2 ---
    mapping2 = ms.TwoBodyDecay(com=False)  # use actual pQ (boosted)
    inputs2 = [r[:, 3], r[:, 4], mQ, np.full(N, m1), np.full(N, m2), pQ]
    p1, p2, det2 = mapping2.map_forward(inputs2, [])

    # Reference Jacobian
    J_ref = det1 * det2 * (smax - smin)

    # --- Our 1->3 mapping under test ---
    if com:
        inputs13 = [
            r[:, 0],
            r[:, 1],
            r[:, 2],
            r[:, 3],
            r[:, 4],
            np.full(N, m0),
            np.full(N, m1),
            np.full(N, m2),
            np.full(N, m3),
        ]
    else:
        inputs13 = [
            r[:, 0],
            r[:, 1],
            r[:, 2],
            r[:, 3],
            r[:, 4],
            np.full(N, m0),
            np.full(N, m1),
            np.full(N, m2),
            np.full(N, m3),
            p0,
        ]
    p1a, p2a, p3a, det = mapping_13.map_forward(inputs13, [])

    # Total volume match (mean Jacobian)
    error1 = np.sqrt(np.var(det) / N)
    error2 = np.sqrt(np.var(J_ref) / N)
    print(f"Mean det error: {error1:.3e}, ref: {error2:.3e}")
    assert np.mean(det) == approx(np.mean(J_ref), abs=3 * max(error1, error2))
