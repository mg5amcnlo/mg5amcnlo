import math
from collections import namedtuple

import pytest
import torch
from madspace_testing import generator, rand, randn

import madspace as ms


def assert_approx(actual, expected):
    """Elementwise pytest.approx with its default tolerances, for tensors."""
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-12)


@pytest.fixture
def rng():
    return generator(1234)


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
    zeros = torch.zeros(N)
    if request.param["decay"]:
        mapping = ms.ThreeBodyDecay(com=com)
        if com:

            def make_args(point):
                p0 = torch.stack([point.m0, zeros, zeros, zeros], dim=1)
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
    return torch.sqrt(momentum[:, 0] ** 2 - torch.sum(momentum[:, 1:] ** 2, dim=1))


InputPoint = namedtuple(
    "InputPoint",
    ["r1", "r2", "r3", "r4", "r5", "m0", "m1", "m2", "m3", "p0", "pa", "pb"],
)

N = 10000


def uniform(rng, low, high):
    return low + (high - low) * rand(N, generator=rng)


@pytest.fixture
def input_points(rng):
    max_mass = 125.0
    r1 = rand(N, generator=rng)
    r2 = rand(N, generator=rng)
    r3 = rand(N, generator=rng)
    r4 = rand(N, generator=rng)
    r5 = rand(N, generator=rng)

    pa = torch.tensor([[0, 0, 4000.0]]) + 500.0 * randn((N, 3), generator=rng)
    ma = uniform(rng, 2.5 * max_mass, 1000.0)
    ea = torch.sqrt(torch.sum(pa**2, dim=1) + ma**2)
    pa = torch.cat([ea[:, None], pa], dim=1)

    pb = torch.tensor([[0, 0, -4000.0]]) + 500.0 * randn((N, 3), generator=rng)
    mb = uniform(rng, 20.0, 500.0)
    eb = torch.sqrt(torch.sum(pb**2, dim=1) + mb**2)
    pb = torch.cat([eb[:, None], pb], dim=1)

    p0 = pa + pb
    m0 = mass(p0)

    m1 = uniform(rng, 1.0, max_mass)
    m2 = uniform(rng, 1.0, max_mass)
    m3 = uniform(rng, 1.0, max_mass)

    return InputPoint(r1, r2, r3, r4, r5, m0, m1, m2, m3, p0, pa, pb)


def test_momentum_conservation(mapping_and_args, input_points):
    mapping, make_args, com = mapping_and_args
    inputs, conditions, p0 = make_args(input_points)
    p1, p2, p3, det = mapping.map_forward(inputs, conditions)
    assert_approx(p1 + p2 + p3, p0)


def test_inverse(mapping_and_args, input_points):
    mapping, make_args, com = mapping_and_args
    inputs, conditions, p0 = make_args(input_points)
    p1, p2, p3, det = mapping.map_forward(inputs, conditions)
    *inv_inputs, inv_det = mapping.map_inverse([p1, p2, p3], conditions)
    assert_approx(inv_det, 1 / det)
    for i, (inp, inv_inp) in enumerate(zip(inputs, inv_inputs)):
        torch.testing.assert_close(
            inp, inv_inp, rtol=1e-6, atol=1e-12, msg=f"Mismatch at index {i}"
        )


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
    assert_approx(m0, input_points.m0)
    assert_approx(m1, input_points.m1)
    assert_approx(m2, input_points.m2)
    assert_approx(m3, input_points.m3)


def test_phase_space_via_two_2body(mapping_and_args, rng):
    mapping_13, make_args, com = mapping_and_args
    N = 50000
    r = rand((N, 5), generator=rng)  # 5 random numbers for 1->3 decay
    u = r[:, 0]  # first random number for s12

    # parent at rest with fixed masses
    m0, m1, m2, m3 = 120.0, 10.0, 15.0, 5.0
    zeros = torch.zeros(N)
    p0 = torch.stack([torch.full((N,), m0), zeros, zeros, zeros], dim=1)

    smin = (m1 + m2) ** 2
    smax = (m0 - m3) ** 2
    s12 = smin + u * (smax - smin)
    mQ = torch.sqrt(s12)

    # --- First 2-body: 0 -> Q + 3 ---
    mapping1 = ms.TwoBodyDecay(com=True)
    inputs1 = [r[:, 1], r[:, 2], torch.full((N,), m0), mQ, torch.full((N,), m3)]
    pQ, p3, det1 = mapping1.map_forward(inputs1, [])

    # Ensure pQ has invariant mass sqrt(s12)
    assert_approx(torch.sqrt(pQ[:, 0] ** 2 - (pQ[:, 1:] ** 2).sum(1)), mQ)

    # --- Second 2-body: Q -> 1 + 2 ---
    mapping2 = ms.TwoBodyDecay(com=False)  # use actual pQ (boosted)
    inputs2 = [r[:, 3], r[:, 4], mQ, torch.full((N,), m1), torch.full((N,), m2), pQ]
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
            torch.full((N,), m0),
            torch.full((N,), m1),
            torch.full((N,), m2),
            torch.full((N,), m3),
        ]
    else:
        inputs13 = [
            r[:, 0],
            r[:, 1],
            r[:, 2],
            r[:, 3],
            r[:, 4],
            torch.full((N,), m0),
            torch.full((N,), m1),
            torch.full((N,), m2),
            torch.full((N,), m3),
            p0,
        ]
    p1a, p2a, p3a, det = mapping_13.map_forward(inputs13, [])

    # Total volume match (mean Jacobian)
    error1 = math.sqrt(det.var(correction=0).item() / N)
    error2 = math.sqrt(J_ref.var(correction=0).item() / N)
    print(f"Mean det error: {error1:.3e}, ref: {error2:.3e}")
    assert det.mean().item() == pytest.approx(
        J_ref.mean().item(), abs=3 * max(error1, error2)
    )
