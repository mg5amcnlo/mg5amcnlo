"""Pairwise cuts handed from PhaseSpaceMapping to ColorOrderedMapping.

In color_ordered mode PhaseSpaceMapping reorders the pairwise invariant-mass
and delta-R cut matrices of Cuts (indexed by outgoing position) into the order
of the t-channel children and passes them to ColorOrderedMapping, which turns
them into floors on the invariants it samples. The tests guard against the
reordering skipping every odd child, which would silently drop any pair cut
involving the second, fourth, ... child.

The process is e+ e- > d u s c: leptonic, so every random number goes straight
to the t-channel mapping and PhaseSpaceMapping returns exactly the momenta of a
ColorOrderedMapping fed the same inputs, and every outgoing parton has its own
flavour, so a cut can name one specific pair.
"""

import math

import pytest
import torch
from madspace_testing import generator, rand, randint

import madspace as ms

O = ms.Observable
PSM = ms.PhaseSpaceMapping

CM_ENERGY = 1000.0
PIDS = [11, -11, 1, 2, 3, 4]
N_OUT = len(PIDS) - 2
MASSES = [0.0] * len(PIDS)
PAIR_MASS_CUT = 400.0
PT_CUT = 100.0
DELTA_R_CUT = 2.0
SEED = 20260918

# PhaseSpaceMapping adds a few graph operations of its own, which typically
# moves the momenta at the 1e-7 GeV level; a missing floor moves them by up to
# O(100 GeV).
ATOL = 1e-5

SINGLE_CHAIN = [0, 2, 3, 4, 5, 1]  # one chain d u s c
SPLIT_2_2 = [0, 2, 3, 1, 4, 5]  # {d u} | {s c}: central 2->2
SPLIT_1_3 = [0, 2, 1, 3, 4, 5]  # {d} | {u s c}: DoubleT + 2->3 peel


def pair_mass(a, b):
    """Cut on the invariant mass of outgoing children a and b."""
    selection = [[PIDS[a + 2]], [PIDS[b + 2]]]
    return ms.CutItem(O(PIDS, O.obs_pair_mass, selection), min=PAIR_MASS_CUT)


def delta_r(a, b):
    selection = [[PIDS[a + 2]], [PIDS[b + 2]]]
    return ms.CutItem(O(PIDS, O.obs_delta_r, selection), min=DELTA_R_CUT)


def pt_all():
    return ms.CutItem(O(PIDS, O.obs_pt, [O.jet_pids]), min=PT_CUT)


def matrix(pairs, value):
    m = [[0.0] * N_OUT for _ in range(N_OUT)]
    for a, b in pairs:
        m[a][b] = m[b][a] = value
    return m


def random_inputs(random_dim, discrete_dim, n, seed):
    g = generator(seed)
    r = rand((n, random_dim), generator=g)
    d = randint(0, 2, (n, discrete_dim), generator=g, dtype=torch.int32)
    return r, d


def psm_momenta(order, cut_items, r, d):
    mapping = PSM(
        MASSES,
        CM_ENERGY,
        leptonic=True,
        mode=PSM.color_ordered,
        cuts=ms.Cuts(cut_items),
        color_ordered=ms.ColorOrderedMapping.Options(order),
    )
    assert mapping.random_dim() == r.shape[1]
    assert mapping.discrete_dim() == d.shape[1]
    inputs = [r, d] if d.shape[1] else [r]
    return mapping.map_forward(inputs)[0]


def co_momenta(order, pt_min, m_inv_min, dr_min, r, d):
    mapping = ms.ColorOrderedMapping(
        ms.ColorOrderedMapping.Options(order), pt_min, m_inv_min, dr_min
    )
    n = r.shape[0]
    inputs = [r[:, i].contiguous() for i in range(r.shape[1])]
    inputs += [d[:, j].contiguous() for j in range(d.shape[1])]
    conditions = [torch.full((n,), CM_ENERGY)] + [torch.zeros(n) for _ in range(N_OUT)]
    out = mapping.map_forward(inputs, conditions)
    return torch.stack(out[:-1], dim=1)


def same_points(p, q):
    """Per point: both NaN, or both finite and equal within ATOL.

    The bound narrows ranges, so some points land where nothing is left to
    sample and come out as NaN (weight zero); those count as the same point."""
    nan_p = ~torch.isfinite(p).all(dim=2).all(dim=1)
    nan_q = ~torch.isfinite(q).all(dim=2).all(dim=1)
    close = (torch.nan_to_num(p - q).abs() <= ATOL).all(dim=2).all(dim=1)
    return (nan_p & nan_q) | (~nan_p & ~nan_q & close)


# Each case is a pair cut involving an odd child in a topology where the bound
# does shape the sampling: a set's composite mass, a 2->3 peel, or the rest
# system of a chain.
# (color order, cut items, pt_min, m_inv_min, dr_min, id)
CASES = [
    (SPLIT_2_2, [pair_mass(0, 1)], [0.0] * N_OUT,
     matrix([(0, 1)], PAIR_MASS_CUT), matrix([], 0.0), "2+2 m(d,u) set mass"),
    (SPLIT_2_2, [pair_mass(2, 3)], [0.0] * N_OUT,
     matrix([(2, 3)], PAIR_MASS_CUT), matrix([], 0.0), "2+2 m(s,c) set mass"),
    (SINGLE_CHAIN, [pair_mass(1, 2)], [0.0] * N_OUT,
     matrix([(1, 2)], PAIR_MASS_CUT), matrix([], 0.0), "chain m(u,s) peel"),
    (SINGLE_CHAIN, [pair_mass(1, 3)], [0.0] * N_OUT,
     matrix([(1, 3)], PAIR_MASS_CUT), matrix([], 0.0), "chain m(u,c) rest"),
    (SPLIT_1_3, [pt_all(), delta_r(1, 2)], [PT_CUT] * N_OUT,
     matrix([], 0.0), matrix([(1, 2)], DELTA_R_CUT), "1+3 dR(u,s) peel"),
]  # fmt: skip


@pytest.mark.parametrize(
    "order, cut_items, pt_min, m_inv_min, dr_min",
    [c[:5] for c in CASES],
    ids=[c[5] for c in CASES],
)
def test_pair_cut_reaches_color_ordered_mapping(
    order, cut_items, pt_min, m_inv_min, dr_min
):
    """PhaseSpaceMapping must hand ColorOrderedMapping the full pair-cut
    matrices: its momenta equal those of a ColorOrderedMapping built by hand
    with the cut, and differ from one built without it."""
    probe = ms.ColorOrderedMapping(ms.ColorOrderedMapping.Options(order))
    r, d = random_inputs(probe.random_dim(), probe.discrete_dim(), 2000, SEED)

    p_psm = psm_momenta(order, cut_items, r, d)
    p_with = co_momenta(order, pt_min, m_inv_min, dr_min, r, d)
    no_pair = matrix([], 0.0)
    p_without = co_momenta(order, pt_min, no_pair, no_pair, r, d)

    # Near degenerate configurations rounding can exceed ATOL, so allow a few points.
    assert same_points(p_psm, p_with).double().mean() > 0.99

    # Guards against a cut that does not affect this topology at all; a floor
    # only moves points where it lies above the kinematic bound.
    assert same_points(p_with, p_without).double().mean() < 0.9


def sample(order, cuts, n, seed):
    mapping = PSM(
        MASSES,
        CM_ENERGY,
        leptonic=True,
        mode=PSM.color_ordered,
        cuts=cuts,
        color_ordered=ms.ColorOrderedMapping.Options(order),
    )
    r, d = random_inputs(mapping.random_dim(), mapping.discrete_dim(), n, seed)
    inputs = [r, d] if mapping.discrete_dim() else [r]
    p_ext, _x1, _x2, det = mapping.map_forward(inputs)
    det = torch.nan_to_num(det) * 2.0 ** mapping.discrete_dim()
    return p_ext, det


def invariant_mass(p, a, b):
    total = p[:, a + 2] + p[:, b + 2]
    m2 = total[:, 0] ** 2 - torch.sum(total[:, 1:] ** 2, dim=1)
    return torch.sqrt(torch.clamp(m2, min=0.0))


# The central 2->2 split, whose set masses are the only place these cuts
# enter. The single chain is left out on purpose: its weights have a heavy
# tail with or without cuts, which would make a volume comparison flaky.
@pytest.mark.parametrize("pair", [(0, 1), (2, 3)], ids=["m(d,u)", "m(s,c)"])
def test_pair_cut_on_odd_child_raises_efficiency_not_volume(pair):
    """Handing the cut to the mapping must raise the fraction of points that
    pass it, and must leave the volume of the cut region unchanged."""
    n = 200_000
    a, b = pair

    p_free, det_free = sample(SPLIT_2_2, None, n, SEED)
    external = torch.where(invariant_mass(p_free, a, b) >= PAIR_MASS_CUT, det_free, 0.0)

    p_cut, piped = sample(SPLIT_2_2, ms.Cuts([pair_mass(a, b)]), n, SEED + 1)
    assert torch.all(invariant_mass(p_cut, a, b)[piped != 0.0] >= PAIR_MASS_CUT)
    assert torch.all(piped >= 0.0)

    # with the bound dropped, both fractions are the same (0.84 / 0.17)
    eff_external = (external > 0.0).double().mean()
    eff_piped = (piped > 0.0).double().mean()
    assert eff_piped > eff_external + 0.03

    error = math.sqrt((external.var(correction=0) + piped.var(correction=0)) / n)
    assert error > 0.0
    assert abs(piped.mean() - external.mean()).item() < 5.0 * error
