"""The arcsine map of the s23 random number in the 2->3 scattering block.

On its kinematic range s23 is linear in cos(phi), and the Byckling-Kajantie
Jacobian is 1 / (sqrt(lambda) (s23_max - s23_min) |sin(phi)|): the measure is
flat in phi. Sampling s23 with any density that is finite at the kinematic
limits leaves an integrable 1/|sin(phi)| in the weight, whose variance diverges
logarithmically. With arcsine_s23 the random number is first mapped flat in
phi / 2 and only then handed to the s23 importance sampling. These tests pin
that
- with flat s23 sampling the block weight becomes exactly independent of r_s23,
- without the map the edge peak is still there,
- forward and inverse agree, with and without cuts,
- ColorOrderedMapping, which always uses the map, integrates to the volume.
"""

import math

import pytest
import torch
from madspace_testing import generator, rand, randint

import madspace as ms

E_BEAM = 500.0
PA = torch.tensor([E_BEAM, 0.0, 0.0, E_BEAM])
PB = torch.tensor([E_BEAM, 0.0, 0.0, -E_BEAM])
# r_s23 from the middle out to the edges
EDGE = 10.0 ** -torch.arange(1.0, 6.0)
R_S23 = torch.cat([EDGE, torch.tensor([0.5]), 1.0 - EDGE])


def massless(energy, theta, phi):
    return energy * torch.tensor(
        [
            1.0,
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ]
    )


def block_weights(r_s23, arcsine, s_power=0.0, m1=40.0, m2=0.0):
    mapping = ms.TwoToThreeParticleScattering(
        0.8, 0.0, 0.0, s_power, 0.0, 0.0, arcsine_s23=arcsine
    )
    n = len(r_s23)
    inputs = [
        torch.zeros(n, dtype=torch.int32),
        r_s23,
        torch.full((n,), 0.61),
        torch.full((n,), m1),
        torch.full((n,), m2),
    ]
    conditions = [
        PA.repeat(n, 1),
        PB.repeat(n, 1),
        massless(350.0, 0.7, 0.3).repeat(n, 1),
    ]
    p1, p2, det = mapping.map_forward(inputs, conditions)
    return mapping, inputs, conditions, det


def test_flat_weight_is_constant():
    """Flat s23 plus the arcsine map is flat in phi, which is exactly the
    measure: at fixed t1 the weight must not depend on r_s23 at all."""
    *_, det = block_weights(R_S23, arcsine=True)
    assert torch.all(torch.isfinite(det)) and torch.all(det > 0)
    ratio = det / det[len(EDGE)]
    torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize("s_power", [0.8, 1.0])
def test_weight_bounded_with_importance_sampling(s_power):
    """With 1/s23^nu importance sampling the weight is not constant,
    but it must stay finite and bounded out to the edges of the range."""
    *_, det = block_weights(R_S23, arcsine=True, s_power=s_power)
    assert torch.all(torch.isfinite(det)) and torch.all(det > 0)
    edges = torch.cat([det[: len(EDGE)], det[len(EDGE) + 1 :]])
    # converged to a finite limit: the last two decades agree
    assert det[len(EDGE) - 1].item() == pytest.approx(
        det[len(EDGE) - 2].item(), rel=1e-3
    )
    assert det[-1].item() == pytest.approx(det[-2].item(), rel=1e-3)
    assert torch.max(edges) / torch.min(edges) < 1e3


def test_edge_peak_without_arcsine():
    """Without the map the weight keeps its 1/|sin(phi)| = 1/(2 sqrt(u (1 - u)))
    peak, with u = r_s23 for flat sampling."""
    *_, det = block_weights(R_S23, arcsine=False)
    ratio = det * torch.sqrt(R_S23 * (1.0 - R_S23))
    torch.testing.assert_close(
        ratio, ratio[len(EDGE)].expand_as(ratio), rtol=1e-5, atol=1e-12
    )
    assert det[len(EDGE) - 1] / det[len(EDGE)] > 100.0


@pytest.mark.parametrize("arcsine", [True, False], ids=["arcsine", "flat"])
@pytest.mark.parametrize("s_power", [0.0, 0.8, 1.0])
def test_round_trip(arcsine, s_power):
    r_s23 = torch.cat([rand(10_000, generator=generator(7)), R_S23])
    mapping, inputs, conditions, det = block_weights(
        r_s23, arcsine=arcsine, s_power=s_power
    )
    p1, p2, _ = mapping.map_forward(inputs, conditions)
    *inv_inputs, det_inv = mapping.map_inverse([p1, p2], conditions)
    torch.testing.assert_close(inv_inputs[1], r_s23, rtol=1e-6, atol=1e-12)
    torch.testing.assert_close(inv_inputs[2], inputs[2], rtol=1e-6, atol=1e-8)
    rt = torch.abs(det * det_inv - 1.0)
    assert torch.max(rt) < 1e-8


@pytest.mark.parametrize("s_power", [0.0, 0.8, 1.0])
def test_round_trip_at_the_edges(s_power):
    """With the arcsine map the position in the s23 range is carried as u and
    1 - u in both directions, never through s23 itself, so r_s23 is recovered
    to the precision of the momenta also within 1e-12 of the edges, where
    going through s23 would lose absolute precision."""
    edge = 10.0 ** -torch.arange(6.0, 13.0)
    r_s23 = torch.cat([edge, 1.0 - edge])
    mapping, inputs, conditions, det = block_weights(
        r_s23, arcsine=True, s_power=s_power
    )
    p1, p2, _ = mapping.map_forward(inputs, conditions)
    *inv_inputs, det_inv = mapping.map_inverse([p1, p2], conditions)
    torch.testing.assert_close(inv_inputs[1], r_s23, rtol=1e-6, atol=1e-15)
    assert torch.max(torch.abs(det * det_inv - 1.0)) < 1e-9


def sample(color_order, seed, n, cuts=None):
    """Weights of ColorOrderedMapping for the massless n-body phase space at
    1 TeV, zero outside the cuts if given."""
    cuts = cuts or {}
    cm_energy = 1000.0
    n_out = len(color_order) - 2
    mapping = ms.ColorOrderedMapping(
        ms.ColorOrderedMapping.Options(color_order), **cuts
    )
    g = generator(seed)
    r = rand((n, mapping.random_dim()), generator=g)
    d = randint(0, 2, (n, mapping.discrete_dim()), generator=g, dtype=torch.int32)
    inputs = [r[:, i].contiguous() for i in range(r.shape[1])]
    inputs += [d[:, j].contiguous() for j in range(d.shape[1])]
    conditions = [torch.full((n,), cm_energy)] + [torch.zeros(n)] * n_out
    *momenta, det = mapping.map_forward(inputs, conditions)
    # momenta of the two beams, then of the outgoing particles
    p = torch.stack(momenta, dim=1)[:, 2:]
    det = det * 2.0 ** mapping.discrete_dim()
    ok = torch.isfinite(p).all(dim=2).all(dim=1) & torch.isfinite(det)
    if cuts:
        pt = torch.hypot(p[..., 1], p[..., 2])
        ok &= torch.all(pt > cuts["pt_min"][0], dim=1)
        eta = torch.asinh(p[..., 3] / torch.clamp(pt, min=1e-300))
        phi = torch.atan2(p[..., 2], p[..., 1])
        for i in range(n_out):
            for j in range(i + 1, n_out):
                q = p[:, i] + p[:, j]
                m2 = q[:, 0] ** 2 - torch.sum(q[:, 1:] ** 2, dim=1)
                dphi = phi[:, i] - phi[:, j]
                dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
                dr = torch.hypot(eta[:, i] - eta[:, j], dphi)
                ok &= m2 > cuts["m_inv_min"][i][j] ** 2
                ok &= dr > cuts["dr_min"][i][j]
    return torch.where(ok, det, 0.0)


@pytest.mark.parametrize(
    "color_order",
    [[0, 2, 3, 4, 5, 1], [0, 2, 1, 3, 4, 5]],
    ids=["single chain", "1+3 split"],
)
def test_color_ordered_massless_volume(color_order):
    """The analytic massless 4-body volume, (pi/2)^3 s^2 / 12, with the map."""
    volume = (math.pi / 2) ** 3 * 1000.0**4 / 12
    means = [
        sample(color_order, seed, 1_000_000).mean().item() for seed in range(200, 204)
    ]
    assert sum(means) / len(means) / volume == pytest.approx(1.0, abs=0.02)
