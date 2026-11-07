import math
from collections import namedtuple

import pytest
import torch
from madspace_testing import generator, rand, randn

import madspace as ms

# ----------------------------
# Utilities
# ----------------------------


def inv_mass_sq(p):
    return p[..., 0] ** 2 - torch.sum(p[..., 1:] ** 2, dim=-1)


def mass(p):
    return torch.sqrt(torch.clamp(inv_mass_sq(p), min=0.0))


def uniform(g, a, b, n):
    return a + (b - a) * rand(n, generator=g)


def beams(pz):
    """Massless beams (pz, 0, 0, +pz) and (pz, 0, 0, -pz)."""
    zeros = torch.zeros_like(pz)
    return (
        torch.stack([pz, zeros, zeros, +pz], dim=1),
        torch.stack([pz, zeros, zeros, -pz], dim=1),
    )


def lab_beams(g):
    """Massless beams with momenta smeared around (0, 0, +-4000)."""
    pa3 = torch.tensor([[0, 0, 4000.0]]) + 500.0 * randn((N, 3), generator=g)
    ea = torch.sqrt(torch.sum(pa3**2, dim=1))  # massless
    pb3 = torch.tensor([[0, 0, -4000.0]]) + 500.0 * randn((N, 3), generator=g)
    eb = torch.sqrt(torch.sum(pb3**2, dim=1))
    return torch.cat([ea[:, None], pa3], dim=1), torch.cat([eb[:, None], pb3], dim=1)


def assert_approx(actual, expected, rel=1e-6, abs=1e-12, msg=None):
    """pytest.approx-like elementwise comparison of tensors."""
    torch.testing.assert_close(actual, expected, rtol=rel, atol=abs, msg=msg)


# ----------------------------
# Fixtures
# ----------------------------

N = 10_000

InputPoint = namedtuple(
    "InputPoint",
    [
        "r_phi",
        "r_t1",
        "r_t2",
        "m1",
        "mir_min",
        "pa",
        "pb",
    ],
)


@pytest.fixture
def rng():
    return generator(1234)


@pytest.fixture(
    params=[False, True],
    ids=["LAB", "COM"],
)
def input_points(rng, request):
    """Generate (pa, pb) and a set of forward inputs for the DoubleT mapping.

    Picks m1, mir_min so that the kinematic ranges of t1 and t2 are non-empty.
    """
    com = request.param

    pz = uniform(rng, 1000.0, 4000.0, N)
    pa, pb = beams(pz)  # massless beams in COM

    if not com:
        # lab frame: beams with smeared momenta
        pa, pb = lab_beams(rng)

    # m1: the fixed single-particle mass
    # mir_min: lower bound on recoil mass (e.g. sum of constituent masses).
    # Pick mir_min well below the sqrt(s) so there's room for the recoil.
    m1 = uniform(rng, 1.0, 50.0, N)
    mir_min = uniform(rng, 50.0, 200.0, N)

    r_phi = rand(N, generator=rng)
    r_t1 = rand(N, generator=rng)
    r_t2 = rand(N, generator=rng)

    return InputPoint(r_phi, r_t1, r_t2, m1, mir_min, pa, pb)


@pytest.fixture(
    params=[
        (5.0, 80.0),
        (50.0, 150.0),
        (100.0, 250.0),
    ],
    ids=[
        "light single, light recoil min",
        "medium single, medium recoil min",
        "heavy single, heavy recoil min",
    ],
)
def fixed_input_points(rng, request):
    """Fixed (m1, mir_min) used by the phase-space-volume test."""
    m1, mir_min = (torch.full((N,), m) for m in request.param)

    pa, pb = beams(torch.full((N,), 3000.0))

    r_phi = rand(N, generator=rng)
    r_t1 = rand(N, generator=rng)
    r_t2 = rand(N, generator=rng)

    return InputPoint(r_phi, r_t1, r_t2, m1, mir_min, pa, pb)


# ----------------------------
# Tests
# ----------------------------


def test_momentum_conservation(input_points):
    mapping = ms.DoubleT()

    inputs = [
        input_points.r_phi,
        input_points.r_t1,
        input_points.r_t2,
    ]
    conditions = [
        input_points.pa,
        input_points.pb,
        input_points.m1,
        input_points.mir_min,
    ]

    p1, p2, det = mapping.map_forward(inputs, conditions)
    assert_approx(p1 + p2, input_points.pa + input_points.pb)


def test_on_shell_mass(input_points):
    mapping = ms.DoubleT()

    inputs = [
        input_points.r_phi,
        input_points.r_t1,
        input_points.r_t2,
    ]
    conditions = [
        input_points.pa,
        input_points.pb,
        input_points.m1,
        input_points.mir_min,
    ]

    p1, p2, det = mapping.map_forward(inputs, conditions)
    assert_approx(mass(p1), input_points.m1)


def test_recoil_mass_minimum(input_points):
    """The recoil mass m_ir should be >= mir_min, since that bound goes
    into the t1/t2 ranges."""
    mapping = ms.DoubleT()

    inputs = [
        input_points.r_phi,
        input_points.r_t1,
        input_points.r_t2,
    ]
    conditions = [
        input_points.pa,
        input_points.pb,
        input_points.m1,
        input_points.mir_min,
    ]

    p1, p2, det = mapping.map_forward(inputs, conditions)
    m_ir = mass(p2)
    # allow a small numerical tolerance below mir_min
    assert torch.all(m_ir >= input_points.mir_min - 1e-6 * input_points.mir_min)


def test_inverse(input_points):
    mapping = ms.DoubleT()

    inputs = [
        input_points.r_phi,
        input_points.r_t1,
        input_points.r_t2,
    ]
    conditions = [
        input_points.pa,
        input_points.pb,
        input_points.m1,
        input_points.mir_min,
    ]

    p1, p2, det = mapping.map_forward(inputs, conditions)
    *inv_inputs, inv_det = mapping.map_inverse([p1, p2], conditions)

    assert_approx(inv_det, 1 / det, rel=1e-5)

    for i, (inp, inv_inp) in enumerate(zip(inputs, inv_inputs)):
        assert_approx(inp, inv_inp, msg=f"mismatch in input index {i}")


def test_phase_space_volume_analytic(fixed_input_points):
    """The integral of DoubleT's det over (r_phi, r_t1, r_t2) should equal
    the 2-body phase-space volume integrated over m_ir^2 in [mir_min^2,
    (sqrt(s)-m1)^2]:

        V = int_{mir_min^2}^{(sqrt(s)-m1)^2} dm_ir^2 * (pi/(2s)) * sqrt(lambda(s, m1^2, mir^2))

    This matches the 2->2 volume convention (no (2pi)^n factors).
    """
    from scipy.integrate import quad

    mapping = ms.DoubleT()
    inputs = [
        fixed_input_points.r_phi,
        fixed_input_points.r_t1,
        fixed_input_points.r_t2,
    ]
    conditions = [
        fixed_input_points.pa,
        fixed_input_points.pb,
        fixed_input_points.m1,
        fixed_input_points.mir_min,
    ]
    p1, p2, det = mapping.map_forward(inputs, conditions)

    # All entries of fixed_input_points have the same s, m1, mir_min by construction.
    pa = fixed_input_points.pa[0]
    pb = fixed_input_points.pb[0]
    s = float((pa[0] + pb[0]) ** 2 - torch.sum((pa[1:] + pb[1:]) ** 2))
    m1_val = float(fixed_input_points.m1[0])
    mir_min_val = float(fixed_input_points.mir_min[0])

    def lam(s, m1_sq, m2_sq):
        return (s - m1_sq - m2_sq) ** 2 - 4 * m1_sq * m2_sq

    def integrand(mir_sq):
        l = lam(s, m1_val**2, mir_sq)
        if l <= 0:
            return 0.0
        return math.pi / (2 * s) * math.sqrt(l)

    analytic_volume, _ = quad(
        integrand,
        mir_min_val**2,
        (math.sqrt(s) - m1_val) ** 2,
    )

    mc_volume = det.mean().item()
    mc_err = det.std(correction=0).item() / math.sqrt(N)

    # 5-sigma agreement
    diff = abs(mc_volume - analytic_volume)
    assert diff < 5 * mc_err, (
        f"DoubleT volume mismatch: MC = {mc_volume:.6e} +/- {mc_err:.6e}, "
        f"analytic = {analytic_volume:.6e}, "
        f"diff = {diff:.4e} ({diff/mc_err:.2f} sigma), "
        f"ratio MC/analytic = {mc_volume/analytic_volume:.4f}"
    )


def test_inverse_lab_specifically(rng):
    """The round trip must work in a lab frame, not just the COM frame."""
    mapping = ms.DoubleT()

    pz = uniform(rng, 1000.0, 4000.0, N)
    pa, pb = lab_beams(rng)

    m1 = torch.full((N,), 30.0)
    mir_min = torch.full((N,), 100.0)
    r_phi = rand(N, generator=rng)
    r_t1 = rand(N, generator=rng)
    r_t2 = rand(N, generator=rng)

    inputs = [r_phi, r_t1, r_t2]
    conditions = [pa, pb, m1, mir_min]

    p1, p2, det = mapping.map_forward(inputs, conditions)
    *inv_inputs, inv_det = mapping.map_inverse([p1, p2], conditions)

    names = ["r_phi", "r_t1", "r_t2"]
    for name, inp, inv_inp in zip(names, inputs, inv_inputs):
        assert_approx(inp, inv_inp, msg=f"LAB mismatch in {name}")
    assert_approx(inv_det, 1 / det, rel=1e-5)


# ----------------------------
# Massless m1 stress tests (precision of the t bounds)
# ----------------------------


@pytest.fixture(
    params=[
        ("m1=0", 0.0),
        ("m1=1e-9", 1e-9),
        ("m1=1e-6", 1e-6),
        ("m1=1e-3", 1e-3),
    ],
    ids=["m1=0", "m1=1e-9", "m1=1e-6", "m1=1e-3"],
)
def massless_input_points(rng, request):
    """Near-massless m1 stresses the inverse: lsquare(p1) suffers
    catastrophic cancellation when m1 is recovered from it, so m1 is taken
    from the conditions instead."""
    _, m1_val = request.param

    pa, pb = beams(torch.full((N,), 6500.0))  # 13 TeV

    m1 = torch.full((N,), m1_val)
    mir_min = torch.full((N,), 50.0)

    r_phi = rand(N, generator=rng)
    r_t1 = rand(N, generator=rng)
    r_t2 = rand(N, generator=rng)

    return InputPoint(r_phi, r_t1, r_t2, m1, mir_min, pa, pb)


def test_inverse_massless(massless_input_points):
    """The DoubleT inverse must round-trip even for m1 ~= 0."""
    mapping = ms.DoubleT()

    inputs = [
        massless_input_points.r_phi,
        massless_input_points.r_t1,
        massless_input_points.r_t2,
    ]
    conditions = [
        massless_input_points.pa,
        massless_input_points.pb,
        massless_input_points.m1,
        massless_input_points.mir_min,
    ]

    p1, p2, det = mapping.map_forward(inputs, conditions)
    *inv_inputs, inv_det = mapping.map_inverse([p1, p2], conditions)

    # det round-trip
    assert_approx(inv_det, 1 / det, rel=1e-5)

    # random round-trip; r_t2 is the most sensitive, as its bounds depend on t1_abs
    for i, (inp, inv_inp) in enumerate(zip(inputs, inv_inputs)):
        assert_approx(
            inp, inv_inp, rel=1e-5, abs=1e-8, msg=f"mismatch in input index {i}"
        )


def test_phase_space_volume_massless(rng):
    """The volume integral matches the analytic one for massless m1."""
    from scipy.integrate import quad

    m1_val = 0.0
    mir_min_val = 80.0
    s_val = 13000.0**2

    pa = torch.tensor([6500.0, 0, 0, 6500.0]).repeat(N, 1)
    pb = torch.tensor([6500.0, 0, 0, -6500.0]).repeat(N, 1)

    m1 = torch.full((N,), m1_val)
    mir_min = torch.full((N,), mir_min_val)

    r_phi = rand(N, generator=rng)
    r_t1 = rand(N, generator=rng)
    r_t2 = rand(N, generator=rng)

    mapping = ms.DoubleT()
    p1, p2, det = mapping.map_forward(
        [r_phi, r_t1, r_t2],
        [pa, pb, m1, mir_min],
    )

    def lam(s, m1_sq, m2_sq):
        return (s - m1_sq - m2_sq) ** 2 - 4 * m1_sq * m2_sq

    def integrand(mir_sq):
        l = lam(s_val, m1_val**2, mir_sq)
        if l <= 0:
            return 0.0
        return math.pi / (2 * s_val) * math.sqrt(l)

    analytic_volume, _ = quad(
        integrand,
        mir_min_val**2,
        (math.sqrt(s_val) - m1_val) ** 2,
    )

    mc_volume = det.mean().item()
    mc_err = det.std(correction=0).item() / math.sqrt(N)

    diff = abs(mc_volume - analytic_volume)
    assert diff < 5 * mc_err, (
        f"DoubleT massless volume mismatch: MC = {mc_volume:.6e} +/- "
        f"{mc_err:.6e}, analytic = {analytic_volume:.6e}, "
        f"diff = {diff:.4e} ({diff/mc_err:.2f} sigma), "
        f"ratio MC/analytic = {mc_volume/analytic_volume:.4f}"
    )
