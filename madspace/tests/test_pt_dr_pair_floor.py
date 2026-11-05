"""The invariant-mass floors implied by pt and delta R cuts.

Two massless particles with pt_i >= pt_i,min and delta R >= R (R <= pi) have

    m^2 = 2 pt_1 pt_2 (cosh(d_eta) - cos(d_phi)) >= 2 pt_1,min pt_2,min (1 - cos R),

and in the partonic centre-of-mass frame the outgoing energies add up to
sqrt(s_hat), each at least the transverse mass, so

    sqrt(s_hat) >= sum_i sqrt(m_i^2 + pt_i,min^2).

Both are consequences of the cuts, so PhaseSpaceMapping uses them to bound the
s-channel invariants it samples. Pinned down here:

  * the bounds hold: no event passing the cuts lies below them, and the pair
    bound is reached, i.e. it is the right number and not only a safe one,
  * with the cuts handed to the mapping, no sampled point falls below them,
  * the volume of the cut region is the same with and without them,
  * the pair floor is left alone for massive legs and for R > pi.

The process is u u~ > e+ e- g g through the fully s-channel diagram
u u~ > a* > (a* > e+ e-) (g* > g g), so both cut pairs are exactly what a
propagator decays into.
"""

import math

import pytest
import torch
from madspace_testing import generator, rand

import madspace as ms

O = ms.Observable

CM_ENERGY = 13000.0
BATCH_SIZE = 100_000
SEED = 20261005

PIDS = [2, -2, -11, 11, 21, 21]
LEPTONS = (0, 1)  # outgoing positions
GLUONS = (2, 3)
PT_LEPTON = 10.0
PT_GLUON = 20.0
DR_LEPTON = 0.4
DR_GLUON = 0.4


def pair_floor(pt_1, pt_2, r):
    return math.sqrt(2.0 * pt_1 * pt_2 * (1.0 - math.cos(r)))


LEPTON_FLOOR = pair_floor(PT_LEPTON, PT_LEPTON, DR_LEPTON)
GLUON_FLOOR = pair_floor(PT_GLUON, PT_GLUON, DR_GLUON)


def topology(outgoing_masses=(0.0, 0.0, 0.0, 0.0)):
    return ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            list(outgoing_masses),
            [ms.Propagator(0.0, 0.0), ms.Propagator(0.0, 0.0), ms.Propagator(0.0, 0.0)],
            [
                ["i0", "i1", "p0"],
                ["p0", "p1", "p2"],
                ["p1", "o0", "o1"],
                ["p2", "o2", "o3"],
            ],
        )
    )


def cuts(dr_gluon=DR_GLUON):
    return ms.Cuts(
        [
            ms.CutItem(O(PIDS, O.obs_pt, [O.lepton_pids]), min=PT_LEPTON),
            ms.CutItem(O(PIDS, O.obs_pt, [O.jet_pids]), min=PT_GLUON),
            ms.CutItem(O(PIDS, O.obs_delta_r, [O.lepton_pids]), min=DR_LEPTON),
            ms.CutItem(O(PIDS, O.obs_delta_r, [O.jet_pids]), min=dr_gluon),
        ]
    )


def mapping(cuts=None, outgoing_masses=(0.0, 0.0, 0.0, 0.0)):
    return ms.PhaseSpaceMapping(topology(outgoing_masses), CM_ENERGY, cuts=cuts)


def sample(mapping, seed=SEED, n=BATCH_SIZE):
    r = rand((n, mapping.random_dim()), generator=generator(seed))
    p_ext, x1, x2, det = mapping.map_forward([r])
    ok = torch.isfinite(det) & torch.isfinite(p_ext).all(dim=2).all(dim=1)
    return p_ext, x1, x2, torch.where(ok, det, 0.0)


def invariant_mass(p, i, j):
    """Mass of outgoing particles i and j (the beams sit at 0 and 1)."""
    total = p[:, i + 2, :] + p[:, j + 2, :]
    m2 = total[:, 0] ** 2 - torch.sum(total[:, 1:] ** 2, dim=1)
    return torch.sqrt(torch.clamp(m2, min=0.0))


def passes(cuts, p_ext):
    return cuts(p_ext).reshape(-1) > 0.5


# --------------------------------------------------------------------------
# the bounds themselves
# --------------------------------------------------------------------------


def test_bounds_hold_and_the_pair_bound_is_reached():
    """Sampled without the floors and filtered by the cuts, no event lies below
    either bound, and the pair masses come close to theirs: the floor is the
    infimum of the cut region, not merely something below it."""
    c = cuts()
    p_ext, x1, x2, _ = sample(mapping(), n=400_000)
    keep = passes(c, p_ext)
    assert keep.sum() > 1000
    m_ll = invariant_mass(p_ext, *LEPTONS)[keep]
    m_gg = invariant_mass(p_ext, *GLUONS)[keep]
    sqrt_s_hat = CM_ENERGY * torch.sqrt(x1 * x2)[keep]
    assert m_ll.min() >= LEPTON_FLOOR * (1 - 1e-9)
    assert m_gg.min() >= GLUON_FLOOR * (1 - 1e-9)
    assert sqrt_s_hat.min() >= (2 * PT_LEPTON + 2 * PT_GLUON) * (1 - 1e-9)
    assert m_ll.min() < 1.5 * LEPTON_FLOOR
    assert m_gg.min() < 1.5 * GLUON_FLOOR


def test_without_the_floors_the_sampler_goes_below_them():
    """The counterpart of the test below: without the cuts the sampler does put
    points under both floors, so the next test is not passing trivially."""
    p_ext, x1, x2, _ = sample(mapping())
    assert (invariant_mass(p_ext, *LEPTONS) < LEPTON_FLOOR).double().mean() > 0.01
    assert (invariant_mass(p_ext, *GLUONS) < GLUON_FLOOR).double().mean() > 0.01
    sqrt_s_hat = CM_ENERGY * torch.sqrt(x1 * x2)
    assert (sqrt_s_hat < 2 * PT_LEPTON + 2 * PT_GLUON).double().mean() > 0.01


def test_cuts_bound_the_sampled_invariants():
    """With the cuts handed to the mapping nothing below the floors is
    generated, and the sampled minima sit on them: the mapping uses these
    numbers, not something stricter."""
    p_ext, x1, x2, _ = sample(mapping(cuts()))
    m_ll = invariant_mass(p_ext, *LEPTONS)
    m_gg = invariant_mass(p_ext, *GLUONS)
    sqrt_s_hat = CM_ENERGY * torch.sqrt(x1 * x2)
    for sampled, floor in [
        (m_ll, LEPTON_FLOOR),
        (m_gg, GLUON_FLOOR),
        (sqrt_s_hat, 2 * PT_LEPTON + 2 * PT_GLUON),
    ]:
        assert sampled.min() >= floor - 1e-6
        assert sampled.min() < 1.01 * floor


def propagator_weight(p_ext, x1, x2, det):
    """A test integrand shaped like the process: the phase-space weight times
    the three s-channel propagators, so that the low-mass region the floors
    remove carries weight. The bare volume grows with s_hat and would not see
    them at all."""
    s_hat = CM_ENERGY**2 * x1 * x2
    m_ll = invariant_mass(p_ext, *LEPTONS)
    m_gg = invariant_mass(p_ext, *GLUONS)
    weight = det / (m_ll**2 * m_gg**2 * s_hat)
    return torch.where(torch.isfinite(weight), weight, 0.0)


def test_cut_volume_is_unchanged_by_the_floors():
    """The floors must follow from the cuts, not add to them: integrating over
    the cut region gives the same answer whether the mapping knows about the
    cuts or only the filter does."""
    c = cuts()
    n = 400_000
    p_free, x1, x2, det_free = sample(mapping(), seed=SEED, n=n)
    full = propagator_weight(p_free, x1, x2, det_free)
    external = torch.where(passes(c, p_free), full, 0.0)
    p_cut, y1, y2, det_cut = sample(mapping(c), seed=SEED + 1, n=n)
    piped = torch.where(
        passes(c, p_cut), propagator_weight(p_cut, y1, y2, det_cut), 0.0
    )

    error = math.sqrt(
        external.var(correction=0).item() / n + piped.var(correction=0).item() / n
    )
    assert 0.0 < error < 0.02 * external.mean().item()
    assert abs(external.mean() - piped.mean()).item() < 5.0 * error
    # the cuts remove nearly all of this integrand, so they and the floors
    # they imply are what the comparison is about
    assert external.mean() < 0.1 * full.mean()


# --------------------------------------------------------------------------
# where the pair floor does not apply
# --------------------------------------------------------------------------


def test_no_pair_floor_for_massive_legs():
    """With a mass the bound does not follow (delta R is in pseudorapidity,
    the invariant in rapidity), so the pair is only held to its threshold.
    The sqrt(s_hat) floor still applies, with transverse masses."""
    mass = 1.0
    masses = (0.0, 0.0, mass, mass)
    p_ext, x1, x2, _ = sample(mapping(cuts(), outgoing_masses=masses))
    m_gg = invariant_mass(p_ext, *GLUONS)
    assert m_gg.min() >= 2 * mass - 1e-6
    assert (m_gg < GLUON_FLOOR).double().mean() > 0.001
    # the massless pair keeps its floor
    assert invariant_mass(p_ext, *LEPTONS).min() >= LEPTON_FLOOR - 1e-6
    sqrt_s_hat = CM_ENERGY * torch.sqrt(x1 * x2)
    transverse_mass_sum = 2 * PT_LEPTON + 2 * math.hypot(mass, PT_GLUON)
    assert sqrt_s_hat.min() >= transverse_mass_sum - 1e-6


def test_no_pair_floor_beyond_pi():
    """For R > pi the minimum is not at d_phi = R, so the formula does not
    apply; the sampler gets no pair floor from these cuts."""
    r = 3.5
    p_ext, _, _, _ = sample(mapping(cuts(dr_gluon=r)))
    m_gg = invariant_mass(p_ext, *GLUONS)
    assert (m_gg < pair_floor(PT_GLUON, PT_GLUON, r)).double().mean() > 0.001
    assert (m_gg < GLUON_FLOOR).double().mean() > 0.001
