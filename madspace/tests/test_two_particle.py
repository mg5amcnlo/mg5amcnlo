import math
from collections import namedtuple

import pytest
import torch
from madspace_testing import generator, rand, randn

import madspace as ms


@pytest.fixture
def rng():
    return generator(1234)


def assert_approx(actual, expected):
    # the default tolerances of pytest.approx
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-12)


@pytest.fixture(
    params=[
        {"decay": True, "com": False},
        {"decay": True, "com": True},
        {"decay": False, "com": False},
        {"decay": False, "com": True},
        {"decay": False, "com": False, "has_cut": True},
        {"decay": False, "com": True, "has_cut": True},
    ],
    ids=[
        "1->2 decay",
        "1->2 decay, COM",
        "2->2 scattering",
        "2->2 scattering, COM",
        "2->2 scattering, cut",
        "2->2 scattering, COM, cut",
    ],
)
def mapping_and_args(request):
    com = request.param["com"]
    zeros = torch.zeros(N)
    if request.param["decay"]:
        mapping = ms.TwoBodyDecay(com=com)
        if com:

            def make_args(point):
                p0 = torch.stack([point.m0, zeros, zeros, zeros], dim=1)
                return ([point.r1, point.r2, point.m0, point.m1, point.m2], [], p0)

        else:

            def make_args(point):
                return (
                    [point.r1, point.r2, point.m0, point.m1, point.m2, point.p0],
                    [],
                    point.p0,
                )

    else:
        has_cut = request.param.get("has_cut", False)
        mapping = ms.TwoToTwoParticleScattering(com=com, has_cut=has_cut)
        cut = [zeros, zeros] if has_cut else []
        if com:

            def make_args(point):
                e0 = point.m0 / 2
                p0 = torch.stack([point.m0, zeros, zeros, zeros], dim=1)
                pa = torch.stack([e0, zeros, zeros, e0], dim=1)
                pb = torch.stack([e0, zeros, zeros, -e0], dim=1)
                return [point.r1, point.r2, point.m1, point.m2], [pa, pb, *cut], p0

        else:

            def make_args(point):
                return (
                    [point.r1, point.r2, point.m1, point.m2],
                    [point.pa, point.pb, *cut],
                    point.p0,
                )

    return (mapping, make_args)


def mass(momentum):
    return torch.sqrt(momentum[:, 0] ** 2 - torch.sum(momentum[:, 1:] ** 2, dim=1))


InputPoint = namedtuple("InputPoint", ["r1", "r2", "m0", "m1", "m2", "p0", "pa", "pb"])

N = 10_000

ZEROS = torch.zeros(N)
PA = torch.tensor(
    [math.sqrt(100**2 + 200**2 + 300**2), 100.0, 200.0, 300.0]
).repeat(N, 1)
PB = torch.tensor(
    [math.sqrt(100**2 + 200**2 + 400**2), 100.0, 200.0, -400.0]
).repeat(N, 1)
P0 = PA + PB
M0 = mass(P0)
M1 = torch.full((N,), 20.0)
M2 = torch.full((N,), 30.0)


@pytest.fixture(
    params=[
        (M0, ZEROS, ZEROS, P0, PA, PB),
        (M0, M1, ZEROS, P0, PA, PB),
        (M0, M1, M2, P0, PA, PB),
    ],
    ids=[
        "both massless",
        "one massive",
        "both massive",
    ],
)
def fixed_input_points(request, rng):
    r1 = rand(N, generator=rng)
    r2 = rand(N, generator=rng)
    return InputPoint(r1, r2, *request.param)


@pytest.fixture
def input_points(rng):
    max_mass = 125.0
    r1 = rand(N, generator=rng)
    r2 = rand(N, generator=rng)

    def uniform(low, high):
        return low + (high - low) * rand(N, generator=rng)

    pa = torch.tensor([[0, 0, 4000.0]]) + 500.0 * randn((N, 3), generator=rng)
    ma = uniform(2.5 * max_mass, 1000.0)
    ea = torch.sqrt(torch.sum(pa**2, dim=1) + ma**2)
    pa = torch.cat([ea[:, None], pa], dim=1)

    pb = torch.tensor([[0, 0, -4000.0]]) + 500.0 * randn((N, 3), generator=rng)
    mb = uniform(20.0, 500.0)
    eb = torch.sqrt(torch.sum(pb**2, dim=1) + mb**2)
    pb = torch.cat([eb[:, None], pb], dim=1)

    p0 = pa + pb
    m0 = mass(p0)

    m1 = uniform(1.0, max_mass)
    m2 = uniform(1.0, max_mass)

    return InputPoint(r1, r2, m0, m1, m2, p0, pa, pb)


def test_momentum_conservation(mapping_and_args, input_points):
    mapping, make_args = mapping_and_args
    inputs, conditions, p0 = make_args(input_points)
    p1, p2, det = mapping.map_forward(inputs, conditions)
    assert_approx(p1 + p2, p0)


def test_inverse(mapping_and_args, input_points):
    mapping, make_args = mapping_and_args
    inputs, conditions, p0 = make_args(input_points)
    p1, p2, det = mapping.map_forward(inputs, conditions)
    *inv_inputs, inv_det = mapping.map_inverse([p1, p2], conditions)
    assert_approx(inv_det, 1 / det)
    for inp, inv_inp in zip(inputs, inv_inputs):
        assert_approx(inp, inv_inp)


def test_outgoing_masses(mapping_and_args, input_points):
    mapping, make_args = mapping_and_args
    inputs, conditions, p0 = make_args(input_points)
    p1, p2, det = mapping.map_forward(inputs, conditions)
    m0 = mass(p1 + p2)
    m1 = mass(p1)
    m2 = mass(p2)
    assert_approx(m0, input_points.m0)
    assert_approx(m1, input_points.m1)
    assert_approx(m2, input_points.m2)


def test_phase_space_volume(mapping_and_args, fixed_input_points):
    mapping, make_args = mapping_and_args
    inputs, conditions, p0 = make_args(fixed_input_points)
    p1, p2, det = mapping.map_forward(inputs, conditions)

    s = fixed_input_points.m0**2
    m1_2 = fixed_input_points.m1**2
    m2_2 = fixed_input_points.m2**2
    phase_space_vol = (
        math.pi / (2 * s) * torch.sqrt((s - m1_2 - m2_2) ** 2 - 4 * m1_2 * m2_2)
    )

    assert_approx(det.mean().expand_as(phase_space_vol), phase_space_vol)
