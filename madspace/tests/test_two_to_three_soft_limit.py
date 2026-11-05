"""The 2->3 scattering block when one of its particles is soft.

TwoToThreeParticleScattering samples s23 inside its kinematic range and weights
the point with the Byckling-Kajantie Jacobian 1 / (8 sqrt(-G4)). Forming the 4x4
Gram determinant G4 (and cos(phi)) directly from invariants of order s cancels
terms of order s^4 down to a result proportional to the square of the s23 range.
Once a particle is soft that range is tiny and the difference drowns in
rounding, giving Jacobians of up to 1e14 in place of O(1); one such point in a
few million could throw a ColorOrderedMapping integration off by a factor of up
to 1e12.

The block therefore computes both from where s23 lies in its range, and that
range from momenta in the p12 rest frame: its Byckling-Kajantie form through 3x3
Gram determinants cancels the same way one level down and gives NaN or zero
Jacobians below a softness of about 1e-8. These tests pin that the Jacobian
follows its smooth soft limit, and that ColorOrderedMapping integrates the
massless n-body phase space to its analytic value.
"""

import math

import pytest
import torch
from madspace_testing import generator, rand, randint

import madspace as ms

E_BEAM = 500.0
PA = torch.tensor([E_BEAM, 0.0, 0.0, E_BEAM])
PB = torch.tensor([E_BEAM, 0.0, 0.0, -E_BEAM])
# deep enough into the soft limit to pass the softness (about 1e-8) where a
# Gram-determinant evaluation breaks down
DELTAS = 10.0 ** -torch.arange(2.0, 15.0)


def massless(energy, theta, phi):
    return energy * torch.tensor(
        [
            1.0,
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ]
    )


def lsquare(p):
    return p[..., 0] ** 2 - torch.sum(p[..., 1:] ** 2, dim=-1)


def scatter(p3, m1, m2):
    mapping = ms.TwoToThreeParticleScattering(0.8, 0.0, 0.0, 0.8, 0.0, 0.0)
    n = len(m1)
    inputs = [
        torch.zeros(n, dtype=torch.int32),
        torch.full((n,), 0.37),
        torch.full((n,), 0.61),
        m1,
        m2,
    ]
    conditions = [PA.repeat(n, 1), PB.repeat(n, 1), p3]
    _k1, _k2, det = mapping.map_forward(inputs, conditions)
    return det


def assert_smooth(values):
    """Every value is finite and within a factor 2 of the first one. (The s23
    sampling moves the limit by a few percent once s23 drops below the 1e-2
    GeV^2 offset of its power map, so this is not a tighter check.)"""
    assert torch.all(torch.isfinite(values)), values
    ratio = values / values[0]
    assert torch.all((ratio > 0.5) & (ratio < 2.0)), ratio


def test_jacobian_with_soft_peeled_particle():
    """m1 -> sqrt(s12) leaves particle 2 soft. Its phase space closes
    linearly, so det / delta must tend to a constant."""
    p3 = massless(350.0, 0.7, 0.3).repeat(len(DELTAS), 1)
    sqrt_s12 = math.sqrt(lsquare(PA + PB - p3[0]).item())
    det = scatter(p3, sqrt_s12 * (1.0 - DELTAS), torch.zeros(len(DELTAS)))
    assert_smooth(det / DELTAS)


def test_jacobian_with_soft_third_particle():
    """A soft third momentum (the particle peeled in the step before) leaves
    the rest of the kinematics regular, so the Jacobian must tend to a
    constant."""
    p3 = torch.stack([massless(350.0 * delta, 0.7, 0.3) for delta in DELTAS.tolist()])
    sqrt_s12 = torch.sqrt(lsquare(PA + PB - p3))
    det = scatter(p3, 0.4 * sqrt_s12, torch.zeros(len(DELTAS)))
    assert_smooth(det)


@pytest.mark.parametrize(
    "color_order",
    [[0, 2, 3, 4, 5, 1], [0, 2, 1, 3, 4, 5]],
    ids=["single chain", "1+3 split"],
)
def test_color_ordered_massless_volume(color_order):
    """The volume of massless 4-body phase space, (pi/2)^3 s^2 / 12. A
    bad Jacobian shows up in only a few points per million, so this needs more
    than a handful of batches to catch."""
    cm_energy = 1000.0
    n_out = len(color_order) - 2
    volume = (
        (math.pi / 2) ** (n_out - 1)
        * cm_energy ** (2 * (n_out - 2))
        / (math.factorial(n_out - 1) * math.factorial(n_out - 2))
    )
    mapping = ms.ColorOrderedMapping(ms.ColorOrderedMapping.Options(color_order))
    n = 2_000_000
    means = []
    for seed in range(100, 108):
        g = generator(seed)
        r = rand((n, mapping.random_dim()), generator=g)
        d = randint(0, 2, (n, mapping.discrete_dim()), generator=g, dtype=torch.int32)
        inputs = [r[:, i].contiguous() for i in range(r.shape[1])]
        inputs += [d[:, j].contiguous() for j in range(d.shape[1])]
        conditions = [torch.full((n,), cm_energy)] + [torch.zeros(n)] * n_out
        *momenta, det = mapping.map_forward(inputs, conditions)
        p = torch.stack(momenta, dim=1)
        det = det * 2.0 ** mapping.discrete_dim()
        det = torch.where(torch.isfinite(p).all(dim=2).all(dim=1), det, 0.0)
        means.append(torch.nan_to_num(det).mean().item())
    # The weights keep an integrable 1/sin(phi) peak at the edges of the s23
    # range, so the error estimate itself is noisy; a fixed tolerance well
    # above the spread (about 1%) is the robust check.
    assert sum(means) / len(means) / volume == pytest.approx(1.0, abs=0.05)


def extreme_points():
    """Soft and collinear limits down to 1e-16, at the corners of the random
    numbers: a soft peeled particle, a soft or vanishing p3, p3 along or
    against pa, a vanishing m1, and s12 -> 0."""
    deltas = [1e-8, 1e-12, 1e-15, 1e-16]
    corners = [0.0, 1e-16, 1e-8, 0.37, 1.0 - 1e-8, 1.0]
    points = []
    for delta in deltas:
        configs = [
            (massless(350.0, 0.7, 0.3), lambda s: math.sqrt(s) * (1.0 - delta)),
            (massless(350.0 * delta, 0.7, 0.3), lambda s: 0.4 * math.sqrt(s)),
            (massless(350.0, delta, 0.3), lambda s: 0.4 * math.sqrt(s)),
            (massless(350.0, math.pi - delta, 0.3), lambda s: 0.4 * math.sqrt(s)),
            (massless(350.0, 0.7, 0.3), lambda s: delta * math.sqrt(s)),
            (massless(500.0 * (1.0 - delta), 0.7, 0.3), lambda s: 0.0),
        ]
        for p3, m1 in configs:
            for r_s23 in corners:
                for r_t1 in corners:
                    for index in (0, 1):
                        s12 = lsquare(PA + PB - p3).item()
                        points.append((p3, m1(s12), r_s23, r_t1, index))
    return points


@pytest.mark.parametrize("arcsine", [True, False], ids=["arcsine", "flat"])
@pytest.mark.parametrize("s_power", [0.0, 0.8])
def test_finite_at_extreme_points(arcsine, s_power):
    """Forward and inverse stay finite in all these limits. Where p1 or p3 is
    along pa the azimuth is undefined and the weight is zero; there the s23
    range has zero width, and the inverse must not divide 0 / 0. A p3 below
    ~1e-12 GeV (or a p_12 that close to rest) must not hit absolute floors in the
    normalisation of the frame and the boost axis, which would put the momenta
    far off shell."""
    points = extreme_points()
    n = len(points)
    mapping = ms.TwoToThreeParticleScattering(
        0.8, 0.0, 0.0, s_power, 0.0, 0.0, arcsine_s23=arcsine
    )
    inputs = [
        torch.tensor([pt[4] for pt in points], dtype=torch.int32),
        torch.tensor([pt[2] for pt in points]),
        torch.tensor([pt[3] for pt in points]),
        torch.tensor([pt[1] for pt in points]),
        torch.zeros(n),
    ]
    conditions = [
        PA.repeat(n, 1),
        PB.repeat(n, 1),
        torch.stack([pt[0] for pt in points]),
    ]
    p1, p2, det = mapping.map_forward(inputs, conditions)
    assert torch.all(torch.isfinite(p1)) and torch.all(torch.isfinite(p2))
    assert torch.all(torch.isfinite(det)) and torch.all(det >= 0)
    # On shell where the weight is not zero. The residual |p^2 - m^2| / E^2 comes
    # from s12 -> 0, a pair collinear to ~1e-6 whose mass pa + pb - p3 fixes only
    # to ~1e-4, and stays below ~1e-9. (At zero weight the pair mass can be below
    # sqrt(eps) of its energy, which momenta cannot carry at all.)
    live = det > 0
    m1 = inputs[3][live]
    off1 = torch.abs(lsquare(p1[live]) - m1**2) / p1[live, 0] ** 2
    off2 = torch.abs(lsquare(p2[live])) / p2[live, 0] ** 2
    assert torch.max(off1) < 1e-8 and torch.max(off2) < 1e-8
    for out in mapping.map_inverse([p1, p2], conditions):
        assert torch.all(torch.isfinite(out))


def test_arcsine_rejects_breit_wigner():
    with pytest.raises(ValueError):
        ms.TwoToThreeParticleScattering(0.8, 0.0, 0.0, 0.8, 80.0, 2.0)
    ms.TwoToThreeParticleScattering(0.8, 0.0, 0.0, 0.8, 80.0, 2.0, arcsine_s23=False)
