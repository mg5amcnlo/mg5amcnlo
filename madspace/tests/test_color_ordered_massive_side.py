"""A side of two massive particles in ColorOrderedMapping with a cut elsewhere.

g g > t t~ g in the color order t g g g t~ (0 2 4 1 3 in particle numbers): the
gluon is alone on one side, the top pair is the other, built by the 2->3 step
off the opposite beam. Any cut turns on the transverse-energy floors of that
step, which are m_T >= m for a top without a pt cut. The floor on the momentum
transfer to the beam was etmin^2 = m_T^2, right only for massless particles:
the bound is m_T^2 - m^2 = pt^2. It removed the tops with |t| < m_t^2, about 6%
of the volume and 20% of it at small top pt, and the AmpliCol channels of
g g > t t~ g came out 25% low.
"""

import math

import pytest
import torch
from madspace_testing import generator, rand, randint

import madspace as ms

N = 400_000
CM_ENERGY = 600.0
ORDER = [2, 0, 4, 1, 3]
MASSES = [173.0, 173.0, 0.0]
PT_MIN = 20.0


def sample(pt_min, seed, masses=MASSES):
    mapping = ms.ColorOrderedMapping(
        ms.ColorOrderedMapping.Options(ORDER), pt_min=pt_min
    )
    g = generator(seed)
    r = rand((N, mapping.random_dim()), generator=g)
    d = randint(0, 2, (N, mapping.discrete_dim()), generator=g, dtype=torch.int32)
    inputs = [r[:, i].contiguous() for i in range(r.shape[1])]
    inputs += [d[:, j].contiguous() for j in range(d.shape[1])]
    conditions = [torch.full((N,), CM_ENERGY)] + [torch.full((N,), m) for m in masses]
    *momenta, det = mapping.map_forward(inputs, conditions)
    det = torch.nan_to_num(det) * 2.0 ** mapping.discrete_dim()
    return mapping, inputs, conditions, momenta, r, det


def pt(p):
    return torch.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2)


@pytest.mark.parametrize("top_pt_max", [math.inf, 50.0], ids=["all", "top pt < 50"])
def test_cut_handed_to_mapping_keeps_volume(top_pt_max):
    """The volume passing the gluon cut is the same whether the mapping knows
    the cut or it is applied afterwards."""
    _, _, _, p_free, _, det_free = sample([0.0, 0.0, 0.0], 1)
    _, _, _, p_cut, _, det_cut = sample([0.0, 0.0, PT_MIN], 2)
    # beams first: the top is momenta[2], the gluon momenta[4]
    external = torch.where(
        (pt(p_free[4]) >= PT_MIN) & (pt(p_free[2]) < top_pt_max), det_free, 0.0
    )
    piped = torch.where(
        (pt(p_cut[4]) >= PT_MIN) & (pt(p_cut[2]) < top_pt_max), det_cut, 0.0
    )
    error = math.sqrt((external.var(correction=0) + piped.var(correction=0)) / N)
    assert error > 0.0
    assert abs(piped.mean() - external.mean()).item() < 5.0 * error


def test_round_trip_massive_side_with_cut():
    mapping, _, conditions, momenta, r, det = sample([0.0, 0.0, PT_MIN], 3)
    out = mapping.map_inverse(momenta, conditions)
    r_inv = torch.stack(out[: mapping.random_dim()], dim=1)
    ok = det > 0
    rt = torch.abs(det / 2.0 ** mapping.discrete_dim() * out[-1] - 1.0)[ok]
    assert torch.mean((rt > 1e-2).double()) < 1e-4
    assert torch.quantile(rt, 0.999) < 1e-6
    r_err = torch.max(torch.abs(r_inv - r), dim=1).values[ok]
    assert torch.mean((r_err > 1e-6).double()) < 1e-4
