from collections import namedtuple

import numpy as np
import pytest
from pytest import approx

import madspace as ms

# ----------------------------
# Utilities
# ----------------------------


def inv_mass_sq(p):
    return p[..., 0] ** 2 - np.sum(p[..., 1:] ** 2, axis=-1)


def mass(p):
    return np.sqrt(np.maximum(inv_mass_sq(p), 0.0))


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
    return np.random.default_rng(1234)


@pytest.fixture(
    params=[False, True],
    ids=["LAB", "COM"],
)
def input_points(rng, request):
    """Generate (pa, pb) and a set of forward inputs for the DoubleT mapping.

    Picks m1, mir_min so that the kinematic ranges of t1 and t2 are non-empty.
    """
    com = request.param

    pz = rng.uniform(1000.0, 4000.0, N)
    e1 = pz  # massless beams in COM
    e2 = pz
    pa = np.stack([e1, np.zeros(N), np.zeros(N), +pz], axis=1)
    pb = np.stack([e2, np.zeros(N), np.zeros(N), -pz], axis=1)

    if not com:
        # boost to a random lab frame (the same recipe used in the 2->3 test)
        pa3 = np.array([[0, 0, 4000.0]]) + rng.normal(0.0, 500.0, (N, 3))
        ea = np.sqrt(np.sum(pa3**2, axis=1))  # massless
        pa = np.concatenate([ea[:, None], pa3], axis=1)
        pb3 = np.array([[0, 0, -4000.0]]) + rng.normal(0.0, 500.0, (N, 3))
        eb = np.sqrt(np.sum(pb3**2, axis=1))
        pb = np.concatenate([eb[:, None], pb3], axis=1)

    # m1: the fixed single-particle mass
    # mir_min: lower bound on recoil mass (e.g. sum of constituent masses).
    # Pick mir_min well below the sqrt(s) so there\'s room for the recoil.
    m1 = rng.uniform(1.0, 50.0, N)
    mir_min = rng.uniform(50.0, 200.0, N)

    r_phi = rng.random(N)
    r_t1 = rng.random(N)
    r_t2 = rng.random(N)

    return InputPoint(r_phi, r_t1, r_t2, m1, mir_min, pa, pb)


@pytest.fixture(
    params=[
        (np.full(N, 5.0), np.full(N, 80.0)),
        (np.full(N, 50.0), np.full(N, 150.0)),
        (np.full(N, 100.0), np.full(N, 250.0)),
    ],
    ids=[
        "light single, light recoil min",
        "medium single, medium recoil min",
        "heavy single, heavy recoil min",
    ],
)
def fixed_input_points(rng, request):
    """Fixed (m1, mir_min) used by the phase-space-volume test."""
    m1, mir_min = request.param

    pz = np.full(N, 3000.0)
    e1 = pz
    e2 = pz
    pa = np.stack([e1, np.zeros(N), np.zeros(N), +pz], axis=1)
    pb = np.stack([e2, np.zeros(N), np.zeros(N), -pz], axis=1)

    r_phi = rng.random(N)
    r_t1 = rng.random(N)
    r_t2 = rng.random(N)

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
    assert p1 + p2 == approx(input_points.pa + input_points.pb)


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
    assert mass(p1) == approx(input_points.m1)


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
    assert np.all(m_ir >= input_points.mir_min - 1e-6 * input_points.mir_min)


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

    assert inv_det == approx(1 / det, rel=1e-5)

    for i, (inp, inv_inp) in enumerate(zip(inputs, inv_inputs)):
        assert inp == approx(inv_inp), f"mismatch in input index {i}"


def test_phase_space_volume_analytic(fixed_input_points):
    """The integral of DoubleT\'s det over (r_phi, r_t1, r_t2) should equal
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
    s = (pa[0] + pb[0]) ** 2 - np.sum((pa[1:] + pb[1:]) ** 2)
    m1_val = fixed_input_points.m1[0]
    mir_min_val = fixed_input_points.mir_min[0]

    def lam(s, m1_sq, m2_sq):
        return (s - m1_sq - m2_sq) ** 2 - 4 * m1_sq * m2_sq

    def integrand(mir_sq):
        l = lam(s, m1_val**2, mir_sq)
        if l <= 0:
            return 0.0
        return np.pi / (2 * s) * np.sqrt(l)

    analytic_volume, _ = quad(
        integrand,
        mir_min_val**2,
        (np.sqrt(s) - m1_val) ** 2,
    )

    mc_volume = np.mean(det)
    mc_err = np.std(det) / np.sqrt(N)

    # 5-sigma agreement
    diff = abs(mc_volume - analytic_volume)
    assert diff < 5 * mc_err, (
        f"DoubleT volume mismatch: MC = {mc_volume:.6e} +/- {mc_err:.6e}, "
        f"analytic = {analytic_volume:.6e}, "
        f"diff = {diff:.4e} ({diff/mc_err:.2f} sigma), "
        f"ratio MC/analytic = {mc_volume/analytic_volume:.4f}"
    )


def test_inverse_lab_specifically(rng):
    """Explicit LAB test: round-trip should work in any frame, not just COM.

    This was the failing case for the 2->3 mapping before we fixed the
    rotation; keep it as a guard for double_t too.
    """
    mapping = ms.DoubleT()

    # boosted lab frame
    pz = rng.uniform(1000.0, 4000.0, N)
    pa3 = np.array([[0, 0, 4000.0]]) + rng.normal(0.0, 500.0, (N, 3))
    ea = np.sqrt(np.sum(pa3**2, axis=1))
    pa = np.concatenate([ea[:, None], pa3], axis=1)
    pb3 = np.array([[0, 0, -4000.0]]) + rng.normal(0.0, 500.0, (N, 3))
    eb = np.sqrt(np.sum(pb3**2, axis=1))
    pb = np.concatenate([eb[:, None], pb3], axis=1)

    m1 = np.full(N, 30.0)
    mir_min = np.full(N, 100.0)
    r_phi = rng.random(N)
    r_t1 = rng.random(N)
    r_t2 = rng.random(N)

    inputs = [r_phi, r_t1, r_t2]
    conditions = [pa, pb, m1, mir_min]

    p1, p2, det = mapping.map_forward(inputs, conditions)
    *inv_inputs, inv_det = mapping.map_inverse([p1, p2], conditions)

    names = ["r_phi", "r_t1", "r_t2"]
    for name, inp, inv_inp in zip(names, inputs, inv_inputs):
        assert inp == approx(inv_inp), f"LAB mismatch in {name}"
    assert inv_det == approx(1 / det, rel=1e-5)


# ----------------------------
# Massless m1 stress tests (regression for the t-bound precision issue)
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
    catastrophic cancellation when m1 is recovered from it. With m1 as a
    condition (Option B fix), this should round-trip cleanly."""
    _, m1_val = request.param

    pz = np.full(N, 6500.0)  # 13 TeV
    pa = np.stack([pz, np.zeros(N), np.zeros(N), +pz], axis=1)
    pb = np.stack([pz, np.zeros(N), np.zeros(N), -pz], axis=1)

    m1 = np.full(N, m1_val)
    mir_min = np.full(N, 50.0)

    r_phi = rng.random(N)
    r_t1 = rng.random(N)
    r_t2 = rng.random(N)

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
    assert inv_det == approx(1 / det, rel=1e-5)

    # random round-trip; r_t2 used to be the most sensitive (its bounds
    # depend on t1_abs which depended on the lossy m1^2 recompute).
    for i, (inp, inv_inp) in enumerate(zip(inputs, inv_inputs)):
        assert inp == approx(
            inv_inp, rel=1e-5, abs=1e-8
        ), f"mismatch in input index {i}"


def test_phase_space_volume_massless(rng):
    """Volume integral should still match analytic for massless m1."""
    from scipy.integrate import quad

    m1_val = 0.0
    mir_min_val = 80.0
    s_val = 13000.0**2

    pa = np.tile([6500.0, 0, 0, 6500.0], (N, 1))
    pb = np.tile([6500.0, 0, 0, -6500.0], (N, 1))

    m1 = np.full(N, m1_val)
    mir_min = np.full(N, mir_min_val)

    r_phi = rng.random(N)
    r_t1 = rng.random(N)
    r_t2 = rng.random(N)

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
        return np.pi / (2 * s_val) * np.sqrt(l)

    analytic_volume, _ = quad(
        integrand,
        mir_min_val**2,
        (np.sqrt(s_val) - m1_val) ** 2,
    )

    mc_volume = np.mean(det)
    mc_err = np.std(det) / np.sqrt(N)

    diff = abs(mc_volume - analytic_volume)
    assert diff < 5 * mc_err, (
        f"DoubleT massless volume mismatch: MC = {mc_volume:.6e} +/- "
        f"{mc_err:.6e}, analytic = {analytic_volume:.6e}, "
        f"diff = {diff:.4e} ({diff/mc_err:.2f} sigma), "
        f"ratio MC/analytic = {mc_volume/analytic_volume:.4f}"
    )
