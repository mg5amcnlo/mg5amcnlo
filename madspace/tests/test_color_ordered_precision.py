"""Numerical precision of ColorOrderedMapping in long chains.

- Outgoing momenta stay on their mass shell and add up to the incoming ones.
- The forward/inverse round trip holds with cuts: the azimuthal frame of the
  2->2 block must not jump with the rounding of the incoming momentum, and
  the massless branch of the block-B s23 bound must not switch off when the
  inverse reads a massless particle's mass off its momentum.
"""

import pytest
import torch
from madspace_testing import generator, rand, randint

import madspace as ms

N = 200_000
CUTS = dict(
    pt_min=[20.0] * 4,
    m_inv_min=[[0.0 if i == j else 10.0 for j in range(4)] for i in range(4)],
    dr_min=[[0.0 if i == j else 0.4 for j in range(4)] for i in range(4)],
)


def run(color_order, cm_energy, seed, cuts=None):
    cuts = cuts or {}
    n_out = len(color_order) - 2
    mapping = ms.ColorOrderedMapping(
        ms.ColorOrderedMapping.Options(color_order), **cuts
    )
    g = generator(seed)
    r = rand((N, mapping.random_dim()), generator=g)
    d = randint(0, 2, (N, mapping.discrete_dim()), generator=g, dtype=torch.int32)
    inputs = [r[:, i].contiguous() for i in range(r.shape[1])]
    inputs += [d[:, j].contiguous() for j in range(d.shape[1])]
    conditions = [torch.full((N,), cm_energy)] + [torch.zeros(N)] * n_out
    *momenta, det = mapping.map_forward(inputs, conditions)
    return mapping, inputs, conditions, momenta, r, det


@pytest.mark.parametrize(
    "color_order",
    [[0, 2, 3, 4, 5, 1], [0, 2, 3, 4, 5, 6, 1], [0, 2, 3, 1, 4, 5, 6]],
    ids=["4 chain", "5 chain", "5 split"],
)
def test_massless_on_shell(color_order):
    cm_energy = 13000.0
    *_, momenta, _, det = run(color_order, cm_energy, 5)
    p = torch.stack(momenta, dim=1)[:, 2:]
    p = p[det > 0]
    # |p^2| / E^2 from (E - |p|) (E + |p|): ~1e-16 from rounding, up to ~1e-12
    # for particles with E ~ 1e-8 sqrt(s), which the 1/sqrt(s) sampling of the
    # rest masses reaches
    p_abs = torch.sqrt(torch.sum(p[..., 1:] ** 2, dim=-1))
    m2 = (p[..., 0] - p_abs) * (p[..., 0] + p_abs)
    assert torch.max(torch.abs(m2) / p[..., 0] ** 2) < 1e-10
    # momentum conservation to a few ulps of the total energy
    p_sum = torch.sum(p, dim=1)
    p_sum[:, 0] -= cm_energy
    assert torch.max(torch.abs(p_sum)) < 1e-14 * cm_energy


def test_round_trip_with_cuts():
    order = [0, 2, 3, 4, 5, 1]
    mapping, inputs, conditions, momenta, r, det = run(order, 1000.0, 11, CUTS)
    out = mapping.map_inverse(momenta, conditions)
    r_inv = torch.stack(out[: mapping.random_dim()], dim=1)
    det_inv = out[-1]
    ok = det > 0
    rt = torch.abs(det * det_inv - 1.0)[ok]
    assert (rt > 1e-2).double().mean() < 1e-4
    assert torch.quantile(rt, 0.999) < 1e-6
    r_err = torch.max(torch.abs(r_inv - r), dim=1).values[ok]
    assert (r_err > 1e-6).double().mean() < 1e-4


@pytest.mark.parametrize(
    "masses",
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 173.0, 173.0, 0.0, 0.0]],
    ids=["massless", "t tbar g g"],
)
def test_lab_frame_on_shell(masses):
    """The boost to the hadronic frame keeps every momentum on its mass shell
    and momentum conserved. Written as E cosh y + p_z sinh y, a particle moving
    against the boost would lose a factor e^{2|y|} of relative precision."""
    cm_energy = 13000.0
    mapping = ms.PhaseSpaceMapping(
        masses,
        cm_energy,
        mode=ms.PhaseSpaceMapping.color_ordered,
        color_ordered=ms.ColorOrderedMapping.Options([0, 2, 3, 4, 5, 1]),
    )
    g = generator(3)
    inputs = [rand((N, mapping.random_dim()), generator=g)]
    if mapping.discrete_dim():
        inputs.append(
            randint(0, 2, (N, mapping.discrete_dim()), generator=g, dtype=torch.int32)
        )
    p, x1, x2, det = mapping.map_forward(inputs)
    # zero-weight points at x1 or x2 ~ 1e-13 have no meaningful momenta
    p = p[det > 1e-20]
    m2 = p[..., 0] ** 2 - torch.sum(p[..., 1:] ** 2, dim=-1)
    # |p^2 - m^2| / E^2: a few 1e-15 for massless momenta, ~1e-12 for the top
    # that absorbs the rounding of the others
    off_shell = torch.abs(m2 - torch.tensor(masses) ** 2) / p[..., 0] ** 2
    assert torch.max(off_shell) < 1e-11
    violation = torch.abs(torch.sum(p[:, 2:], dim=1) - torch.sum(p[:, :2], dim=1))
    assert torch.max(violation) < 1e-14 * cm_energy
