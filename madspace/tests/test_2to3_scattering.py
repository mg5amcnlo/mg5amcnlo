import math
from collections import namedtuple

import pytest
import torch
from madspace_testing import generator, rand, randint, randn
from pytest import approx

import madspace as ms

# ----------------------------
# Utilities
# ----------------------------


def inv_mass_sq(p):
    # p: (..., 4) with [E, px, py, pz]
    return p[..., 0] ** 2 - torch.sum(p[..., 1:] ** 2, dim=-1)


def mass(p):
    return torch.sqrt(torch.clamp(inv_mass_sq(p), min=0.0))


def uniform(g, a, b, n):
    return a + (b - a) * rand(n, generator=g)


def assert_approx(actual, expected, rel=1e-6, abs=1e-12, msg=None):
    """pytest.approx-like elementwise comparison of tensors."""
    torch.testing.assert_close(actual, expected, rtol=rel, atol=abs, msg=msg)


def boost(
    k: torch.Tensor, p_boost: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    # Change sign if inverse boost is performed
    sign = -1.0 if inverse else 1.0
    # Perform the boost
    EPS = 1e-16
    rsq = torch.clamp(mass(p_boost), min=EPS)
    kp = torch.sum(k[..., 1:] * p_boost[..., 1:], dim=-1)
    k0 = (k[..., 0] * p_boost[..., 0] + sign * kp) / rsq
    c1 = (k[..., 0] + k0) / (rsq + p_boost[..., 0])
    k1 = k[..., 1] + sign * c1 * p_boost[..., 1]
    k2 = k[..., 2] + sign * c1 * p_boost[..., 2]
    k3 = k[..., 3] + sign * c1 * p_boost[..., 3]

    return torch.stack((k0, k1, k2, k3), dim=-1)


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

ZEROS = torch.zeros(N)
PA = torch.tensor(
    [math.sqrt(100**2 + 200**2 + 300**2), 100.0, 200.0, 300.0]
).repeat(N, 1)
PB = torch.tensor(
    [math.sqrt(100**2 + 200**2 + 400**2), 100.0, 200.0, -400.0]
).repeat(N, 1)
P0 = PA + PB
M0 = mass(P0)
M1 = torch.full((N,), 10.0)
M2 = torch.full((N,), 20.0)
M3 = torch.full((N,), 30.0)
M12 = torch.full((N,), 200.0)


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
    r1 = rand(N, generator=rng)
    r2 = rand(N, generator=rng)
    p12, p3, det_22 = map_22.map_forward([r1, r2, m12, m3], [pA, pB])

    # Discrete two-solution choice (int 0/1) + continuous randoms for the 2->3
    r_choice = randint(0, 2, (N,), generator=rng, dtype=torch.int32)
    r_s23 = rand(N, generator=rng)
    r_t1 = rand(N, generator=rng)

    return InputPoint(r_choice, r_s23, r_t1, m0, m12, m1, m2, m3, pA, pB, p0, p3, p12)


@pytest.fixture
def rng():
    return generator(1234)


@pytest.fixture(params=[False, True], ids=["base 2->2, LAB", "base 2->2, COM"])
def input_points(rng, request):
    """
    Generate (pa, pb) and a valid 'spectator' p3 by first producing a 2->2 scattering.
    We then use pa, pb, and that p3 as conditions for the 2->3 peel-off.
    """
    com = request.param

    # start with compatible incoming momenta
    # COM: pa=(Ea,0,0,+p), pb=(Eb,0,0,-p)
    pz = uniform(rng, 1000.0, 4000.0, N)
    ma = uniform(rng, 50.0, 300.0, N)
    mb = uniform(rng, 200.0, 500.0, N)
    e1 = torch.sqrt(pz**2 + ma**2)
    e2 = torch.sqrt(pz**2 + mb**2)
    pa = torch.stack([e1, torch.zeros(N), torch.zeros(N), +pz], dim=1)
    pb = torch.stack([e2, torch.zeros(N), torch.zeros(N), -pz], dim=1)

    if not com:
        # boost to a some random frame
        pa = torch.tensor([[0, 0, 4000.0]]) + 500.0 * randn((N, 3), generator=rng)
        ea = torch.sqrt(torch.sum(pa**2, dim=1) + ma**2)
        pa = torch.cat([ea[:, None], pa], dim=1)

        pb = torch.tensor([[0, 0, -4000.0]]) + 500.0 * randn((N, 3), generator=rng)
        eb = torch.sqrt(torch.sum(pb**2, dim=1) + mb**2)
        pb = torch.cat([eb[:, None], pb], dim=1)

    # Build a valid 2->2 event to extract p3 as a spectator
    # Choose arbitrary outgoing masses for (Q, p3): make p3 light, Q heavyish
    m3 = uniform(rng, 1.0, 10.0, N)
    m1 = uniform(rng, 1.0, 40.0, N)
    m2 = uniform(rng, 1.0, 40.0, N)
    m12 = uniform(rng, 160, 400, N)

    map_22 = ms.TwoToTwoParticleScattering(com=com)
    r1 = rand(N, generator=rng)
    r2 = rand(N, generator=rng)
    p12, p3, det_22 = map_22.map_forward([r1, r2, m12, m3], [pa, pb])

    # Randoms for the 2->3 mapper
    r_choice = randint(0, 2, (N,), generator=rng, dtype=torch.int32)
    r_s23 = rand(N, generator=rng)
    r_t1 = rand(N, generator=rng)

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

    assert_approx(p_sum, input_points.p0)
    assert_approx(p1 + p2, input_points.p12)


@pytest.mark.parametrize("arcsine", [False, True], ids=["flat s23", "arcsine s23"])
def test_inverse(input_points, arcsine):
    mapping = ms.TwoToThreeParticleScattering(arcsine_s23=arcsine)

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

    assert_approx(inv_det, 1 / det, rel=1e-5)

    for i, (inp, inv_inp) in enumerate(zip(inputs, inv_inputs)):
        if i == 0:
            # discrete two-solution choice: recovered exactly as an int
            assert torch.equal(inv_inp.long(), inp.long())
            continue
        assert_approx(inp, inv_inp, msg=f"mismatch in input index {i}")


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
    assert_approx(mass(p1 + p2), input_points.m12)
    assert_approx(mass(p1), input_points.m1)
    assert_approx(mass(p2), input_points.m2)


def test_phase_space_compare(rng, input_points):
    mapping23 = ms.TwoToThreeParticleScattering()
    mapping22 = ms.TwoToTwoParticleScattering(com=False)
    r1 = rand(N, generator=rng)
    r2 = rand(N, generator=rng)

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

    # det23 carries a factor 1/2 for the two solutions (kernel_two_to_three_particle_scattering),
    # so undo it here to compare against the 2->2 phase-space element.
    det23 = det23 * 2.0

    # The mean Jacobians of both mappings must agree within the statistical error.
    std_error_23 = det23.std(correction=0).item() / math.sqrt(N)
    assert det23.mean().item() == approx(
        det22.mean().item(), abs=3 * std_error_23, rel=1e-6
    )


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
    # undo the factor 1/2 for the two solutions contained in det
    det = det * 2.0

    s = fixed_input_points.m12**2
    m1_2 = fixed_input_points.m1**2
    m2_2 = fixed_input_points.m2**2
    phase_space_vol = (
        math.pi / (2 * s) * torch.sqrt((s - m1_2 - m2_2) ** 2 - 4 * m1_2 * m2_2)
    )
    std_error = det.std(correction=0).item() / math.sqrt(N)
    # fixed masses: the volume is the same for every point
    assert det.mean().item() == approx(
        phase_space_vol[0].item(), abs=3 * std_error, rel=1e-6
    )
