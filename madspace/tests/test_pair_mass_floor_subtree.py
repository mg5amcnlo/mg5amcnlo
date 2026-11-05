"""A pair mass cut bounds every propagator whose decay contains the pair.

test_pair_mass_cut.py covers a propagator that decays into exactly the cut
pair. This file covers the general case: a propagator whose decay holds the
pair together with other particles. For future-pointing momenta the invariant
mass of a sum is at least the sum of the invariant masses, so such a node obeys

    m(node) >= m_cut + sum of the masses of its other final-state particles,

and handing that floor to the sampler is the difference between generating the
region the cut allows and generating almost nothing of it.

The channels come from p p > w+ w+ j j QCD=0 (subprocess u d~ > w+ w+ u d~ in
the output's own particle order: W+, W+, jet, jet), with a 200 GeV cut on the
jet pair:

  * W_CHANNEL: two t-channel quark lines meet in an s-channel W- that decays to
    the jet pair, so the cut is exactly a floor on the W- (100% efficiency).
  * Z_CHANNEL: a Z decays to a jet and an off-shell quark that decays to the
    other jet and a W+. The jet pair is two of the Z's three leaves. Without
    the floor on such nodes only 0.04% of this channel passes the cut.

The cut-region volume, integrated with each channel, must match the flat
RAMBO mapping that knows nothing about the cut: the floor follows from the
cut and must not cut anything of its own.

Further sections pin down the cases around it: the channel permutations (one
mapping serves several orderings of the event, so a floor must hold for all of
them), an on-shell window the cut excludes entirely, CutMode::any pair cuts,
the root of a mapping that does not sample it, which cuts Cuts reports as
floors at all, and the same-flavour opposite-sign pair mass.
"""

import math

import phasespace_helpers
import pytest
import torch
from phasespace_helpers import invariant_mass

import madspace as ms

O = ms.Observable
PSM = ms.PhaseSpaceMapping

CM_ENERGY = 13000.0
M_W, W_W = 80.419, 2.0476
M_Z, W_Z = 91.188, 2.441
BATCH_SIZE = 200_000
# the volume comparisons need the precision; 2M points still take under a second
VOLUME_BATCH_SIZE = 2_000_000
SEED = 20261003

PIDS = [2, -1, 24, 24, 2, -1]
JET_PIDS = [2, -1]
JETS = (4, 5)
W_PLUS = (2, 3)
OUTGOING_MASSES = [M_W, M_W, 0.0, 0.0]
PAIR_MASS_CUT = 200.0
BW_CUTOFF = 15.0

W_CHANNEL = (
    [(0.0, 0.0, -1), (0.0, 0.0, 1), (M_W, W_W, -24)],
    [["i0", "o0", "p0"], ["i1", "o1", "p1"], ["o3", "o2", "p2"], ["p0", "p1", "p2"]],
)
Z_CHANNEL = (
    [(0.0, 0.0, -1), (0.0, 0.0, -1), (M_Z, W_Z, 23)],
    [["o3", "o1", "p0"], ["i0", "o0", "p1"], ["p1", "i1", "p2"], ["p0", "o2", "p2"]],
)


def topology(channel, outgoing_masses=OUTGOING_MASSES, on_shell=()):
    """The channel's topology; propagators listed in on_shell get the bw_cutoff
    window build_topologies in the mg7 launcher gives an on-shell propagator."""
    propagators, vertices = channel
    return ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            outgoing_masses,
            [
                ms.Propagator(
                    mass=mass,
                    width=width,
                    integration_order=0,
                    e_min=mass - BW_CUTOFF * width if i in on_shell else 0.0,
                    e_max=mass + BW_CUTOFF * width if i in on_shell else 0.0,
                    pdg_id=pdg_id,
                )
                for i, (mass, width, pdg_id) in enumerate(propagators)
            ],
            vertices,
        )
    )


def pair_mass_cuts(minimum=PAIR_MASS_CUT):
    return ms.Cuts([ms.CutItem(O(PIDS, O.obs_pair_mass, [JET_PIDS]), min=minimum)])


def sample(mapping, seed, n=BATCH_SIZE, conditions=()):
    """Momenta and the weight with non-finite points set to 0."""
    p, det = phasespace_helpers.sample(mapping, seed, n, conditions)
    return p, phasespace_helpers.finite_weight(p, det)


def cut_volume(p, det):
    """Mean and standard error of the cut-region volume from one batch."""
    weights = torch.where(invariant_mass(p, *JETS) >= PAIR_MASS_CUT, det, 0.0)
    return weights.mean().item(), std(weights) / math.sqrt(len(weights))


def std(w):
    """numpy's standard deviation (no Bessel correction) as a float."""
    return w.std(correction=0).item()


@pytest.fixture(scope="module")
def rambo_volume():
    """The cut-region volume from a mapping with no propagator structure."""
    flat = PSM([0.0, 0.0] + OUTGOING_MASSES, CM_ENERGY, mode=PSM.rambo)
    return cut_volume(*sample(flat, SEED, n=VOLUME_BATCH_SIZE))


# --------------------------------------------------------------------------
# the propagator that decays into exactly the pair
# --------------------------------------------------------------------------


def test_w_channel_samples_only_the_allowed_region():
    p, det = sample(PSM(topology(W_CHANNEL), CM_ENERGY, cuts=pair_mass_cuts()), SEED)
    mjj = invariant_mass(p, *JETS)[det > 0]
    assert mjj.min() >= PAIR_MASS_CUT * (1 - 1e-9)


def test_w_channel_volume_matches_rambo(rambo_volume):
    mapping = PSM(topology(W_CHANNEL), CM_ENERGY, cuts=pair_mass_cuts())
    mean, error = cut_volume(*sample(mapping, SEED + 1, n=VOLUME_BATCH_SIZE))
    ref, ref_error = rambo_volume
    assert abs(mean - ref) < 5.0 * math.hypot(error, ref_error)


# --------------------------------------------------------------------------
# a propagator whose decay holds the pair and more
# --------------------------------------------------------------------------


def test_z_channel_propagator_gets_the_floor():
    """The Z's leaves are the jet pair and a W+, so m(Z) >= cut + m_W."""
    p, det = sample(PSM(topology(Z_CHANNEL), CM_ENERGY, cuts=pair_mass_cuts()), SEED)
    m_z = invariant_mass(p, JETS[0], JETS[1], W_PLUS[1])[det > 0]
    assert m_z.min() >= (PAIR_MASS_CUT + M_W) * (1 - 1e-9)


def test_z_channel_efficiency():
    """The jet pair is not a propagator of its own here, so the floor on the Z
    cannot make every point pass, but it has to lift the pass rate well above
    the ~0.04% of the unbounded channel."""
    p_free, _ = sample(PSM(topology(Z_CHANNEL), CM_ENERGY), SEED)
    p_cut, det = sample(
        PSM(topology(Z_CHANNEL), CM_ENERGY, cuts=pair_mass_cuts()), SEED
    )
    free = (invariant_mass(p_free, *JETS) >= PAIR_MASS_CUT).double().mean()
    bounded = (invariant_mass(p_cut, *JETS)[det > 0] >= PAIR_MASS_CUT).double().mean()
    assert free < 0.01
    assert bounded > 0.2


def test_z_channel_volume_matches_rambo(rambo_volume):
    mapping = PSM(topology(Z_CHANNEL), CM_ENERGY, cuts=pair_mass_cuts())
    mean, error = cut_volume(*sample(mapping, SEED + 2, n=VOLUME_BATCH_SIZE))
    ref, ref_error = rambo_volume
    assert abs(mean - ref) < 5.0 * math.hypot(error, ref_error)


@pytest.mark.parametrize("mode", [PSM.propagator, PSM.rambo, PSM.chili])
def test_floor_reaches_every_t_channel_mode(mode):
    mapping = PSM(
        topology(Z_CHANNEL), CM_ENERGY, cuts=pair_mass_cuts(), t_channel_mode=mode
    )
    p, det = sample(mapping, SEED, n=20_000)
    m_z = invariant_mass(p, JETS[0], JETS[1], W_PLUS[1])[det > 0]
    assert len(m_z) > 0
    assert m_z.min() >= (PAIR_MASS_CUT + M_W) * (1 - 1e-9)


# --------------------------------------------------------------------------
# which cuts count as floors
# --------------------------------------------------------------------------

JETS4 = [21, 21, 1, 2, 3, 4]
JET_GROUP = [1, 2, 3, 4]


def test_two_group_summed_mass_is_a_pair_floor():
    """ "jet-jet-sum-mass" is the same cut as "jet-pair_mass"."""
    summed = O(PIDS, O.obs_mass, [JET_PIDS, JET_PIDS], sum_momenta=True)
    pairwise = O(PIDS, O.obs_pair_mass, [JET_PIDS])
    floors = [
        ms.Cuts([ms.CutItem(obs, min=PAIR_MASS_CUT)]).m_inv_min()
        for obs in (summed, pairwise)
    ]
    assert floors[0] == floors[1]
    assert floors[0][2][3] == PAIR_MASS_CUT


def test_ordered_selection_is_not_a_floor():
    """ "jet_1-pt" bounds the leading jet, whichever particle that is, so it
    bounds no particle in particular and must stay a filter."""
    leading_pt = O(
        JETS4, O.obs_pt, [JET_GROUP], order_observable=O.obs_pt, order_indices=[1]
    )
    leading_pair = O(
        JETS4,
        O.obs_pair_mass,
        [JET_GROUP, JET_GROUP],
        order_observable=O.obs_pt,
        order_indices=[1, 2],
    )
    cuts = ms.Cuts(
        [ms.CutItem(leading_pt, min=300.0), ms.CutItem(leading_pair, min=500.0)]
    )
    assert cuts.pt_min() == [0.0] * 4
    assert torch.all(torch.tensor(cuts.m_inv_min()) == 0.0)


@pytest.mark.parametrize("mode", [ms.Cuts.CutMode.all, ms.Cuts.CutMode.any])
def test_any_mode_over_several_objects_is_not_a_floor(mode):
    """With mode "any" one passing jet (pair) is enough, so no individual jet
    (pair) is bounded."""
    cuts = ms.Cuts(
        [
            ms.CutItem(O(JETS4, O.obs_pt, [JET_GROUP]), min=50.0, mode=mode),
            ms.CutItem(O(JETS4, O.obs_pair_mass, [JET_GROUP]), min=500.0, mode=mode),
        ]
    )
    bounded = mode == ms.Cuts.CutMode.all
    assert cuts.pt_min() == [50.0 if bounded else 0.0] * 4
    m_inv = torch.tensor(cuts.m_inv_min())
    off_diagonal = m_inv[~torch.eye(4, dtype=torch.bool)]
    assert torch.all(off_diagonal == (500.0 if bounded else 0.0))


def test_any_mode_over_one_object_is_a_floor():
    cuts = ms.Cuts(
        [
            ms.CutItem(
                O(PIDS, O.obs_pair_mass, [JET_PIDS]),
                min=200.0,
                mode=ms.Cuts.CutMode.any,
            )
        ]
    )
    assert cuts.m_inv_min()[2][3] == 200.0


# --------------------------------------------------------------------------
# channel permutations
# --------------------------------------------------------------------------
#
# 2 -> 3 through one s-channel propagator X -> (o1, o2), all massless. The
# channel also serves the event with o0 and o1 swapped (the momenta are
# permuted after sampling: event[i] = topology[perm[i]]). A cut on the pair
# (event outgoing 1, event outgoing 2) is then the pair X decays into for the
# identity, but not for the swap, so X must not get the floor: it would remove
# 43% of the cut region from every point sampled for the swap.

PERM_ENERGY = 1000.0
PERM_CUT = 300.0
PERM_PIDS = [21, 21, 1, 2, 3]
PERMUTATIONS = [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]]


def x_topology():
    return ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            [0.0, 0.0, 0.0],
            [ms.Propagator(mass=0.0, width=0.0, integration_order=0) for _ in range(2)],
            [["i0", "i1", "p0"], ["o1", "o2", "p1"], ["p0", "p1", "o0"]],
        )
    )


def x_cuts():
    # PERM_PIDS names event outgoing 1 (pid 2) and 2 (pid 3)
    return ms.Cuts(
        [ms.CutItem(O(PERM_PIDS, O.obs_pair_mass, [[2], [3]]), min=PERM_CUT)]
    )


def test_floor_without_permutations():
    """The counterpart: with the identity only, X is exactly the cut pair."""
    p, det = sample(PSM(x_topology(), PERM_ENERGY, cuts=x_cuts()), SEED)
    assert invariant_mass(p, 3, 4)[det > 0].min() >= PERM_CUT * (1 - 1e-9)


def test_floor_must_hold_for_every_permutation():
    mapping = PSM(x_topology(), PERM_ENERGY, cuts=x_cuts(), permutations=PERMUTATIONS)
    n = 20_000
    p, det = sample(mapping, SEED, n, [torch.zeros(n, dtype=torch.int32)])
    # X gets no floor, so points with m(X) below the cut are still sampled
    assert invariant_mass(p, 3, 4).min() < 0.5 * PERM_CUT


@pytest.mark.parametrize("index", [0, 1])
def test_permuted_volume_matches_rambo(index):
    flat = PSM([0.0] * 5, PERM_ENERGY, mode=PSM.rambo)
    p, det = sample(flat, SEED, VOLUME_BATCH_SIZE)
    ref = torch.where(invariant_mass(p, 3, 4) >= PERM_CUT, det, 0.0)
    mapping = PSM(x_topology(), PERM_ENERGY, cuts=x_cuts(), permutations=PERMUTATIONS)
    n = VOLUME_BATCH_SIZE
    _, weight = sample(
        mapping, SEED + 1 + index, n, [torch.full((n,), index, dtype=torch.int32)]
    )
    error = math.hypot(std(weight), std(ref)) / math.sqrt(n)
    assert abs(weight.mean() - ref.mean()) < 5.0 * error


# --------------------------------------------------------------------------
# an on-shell window the cut excludes
# --------------------------------------------------------------------------


def test_window_below_the_floor_makes_the_channel_empty():
    """An on-shell W- (window m_W +- 15 Gamma, up to 111 GeV) cannot decay to
    a jet pair above 200 GeV: the channel is empty, and says so instead of
    sampling an inverted range."""
    mapping = PSM(topology(W_CHANNEL, on_shell=(2,)), CM_ENERGY, cuts=pair_mass_cuts())
    assert mapping.empty()
    p, det = phasespace_helpers.sample(mapping, SEED, 20_000)
    # the window is left as it was: a valid range, every point cut away
    assert torch.all(torch.isfinite(det))
    assert torch.all(det == 0.0)


def test_window_above_the_floor_is_narrowed():
    cut = 100.0
    mapping = PSM(
        topology(W_CHANNEL, on_shell=(2,)), CM_ENERGY, cuts=pair_mass_cuts(cut)
    )
    assert not mapping.empty()
    p, det = sample(mapping, SEED, 20_000)
    mjj = invariant_mass(p, *JETS)
    assert mjj.min() >= cut * (1 - 1e-9)
    assert mjj.max() <= (M_W + BW_CUTOFF * W_W) * (1 + 1e-9)
    assert not PSM(topology(W_CHANNEL, on_shell=(2,)), CM_ENERGY).empty()


# --------------------------------------------------------------------------
# CutMode::any pair mass cuts
# --------------------------------------------------------------------------
#
# A W+ - jet pair above 400 GeV, for any one of the four such pairs. No pair is
# bounded, but the whole final state is: one W+ - jet pair is at least 400 GeV
# and the other W+ at least m_W, whichever pair it is.

ANY_CUT = 400.0


def any_cuts():
    observable = O(PIDS, O.obs_pair_mass, [[24], JET_PIDS])
    return ms.Cuts([ms.CutItem(observable, min=ANY_CUT, mode=ms.Cuts.CutMode.any)])


def any_passes(p):
    pairs = [(w, j) for w in W_PLUS for j in JETS]
    masses = torch.stack([invariant_mass(p, w, j) for w, j in pairs])
    return masses.max(dim=0).values >= ANY_CUT


def test_any_mode_cut_is_reported_as_a_set_bound():
    (bound,) = any_cuts().pair_mass_any_min()
    assert bound.min == ANY_CUT
    assert sorted(bound.pairs) == [(0, 2), (0, 3), (1, 2), (1, 3)]
    assert torch.all(torch.tensor(any_cuts().m_inv_min()) == 0.0)


def test_any_mode_cut_bounds_s_hat():
    p, det = sample(PSM(topology(W_CHANNEL), CM_ENERGY, cuts=any_cuts()), SEED)
    assert invariant_mass(p, 0, 1)[det > 0].min() >= (ANY_CUT + M_W) * (1 - 1e-9)


def test_any_mode_volume_matches_rambo():
    flat = PSM([0.0, 0.0] + OUTGOING_MASSES, CM_ENERGY, mode=PSM.rambo)
    p, det = sample(flat, SEED, VOLUME_BATCH_SIZE)
    ref = torch.where(any_passes(p), det, 0.0)
    mapping = PSM(topology(W_CHANNEL), CM_ENERGY, cuts=any_cuts())
    # the weights are heavy-tailed, so the error estimate is noisy: SEED + 1 misses
    _, weight = sample(mapping, SEED + 2, VOLUME_BATCH_SIZE)
    error = math.hypot(std(weight), std(ref)) / math.sqrt(VOLUME_BATCH_SIZE)
    assert abs(weight.mean() - ref.mean()) < 5.0 * error


# --------------------------------------------------------------------------
# a root the mapping does not sample
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "masses_mode",
    [("leptonic", PSM.propagator), ("hadronic", PSM.chili)],
)
def test_unsampled_root_leaves_the_momenta_alone(masses_mode):
    """A leptonic collision fixes s_hat and chili reconstructs it, so a pair
    cut on a 2 -> 2 process can only filter: same random numbers, same
    momenta, and the weight changes only where the cut removes the point."""
    kind, mode = masses_mode
    pids = [21, 21, 1, 2]
    cuts = ms.Cuts([ms.CutItem(O(pids, O.obs_pair_mass, [[1], [2]]), min=300.0)])
    leptonic = kind == "leptonic"
    free = PSM([0.0] * 4, 1000.0, leptonic=leptonic, mode=mode)
    cut = PSM([0.0] * 4, 1000.0, leptonic=leptonic, mode=mode, cuts=cuts)
    p_free, det_free = phasespace_helpers.sample(free, SEED, 20_000)
    p_cut, det_cut = phasespace_helpers.sample(cut, SEED, 20_000)
    torch.testing.assert_close(p_cut, p_free, rtol=0, atol=0, equal_nan=True)
    passes = invariant_mass(p_free, 2, 3) >= 300.0
    assert torch.equal(det_cut[passes], det_free[passes])
    assert torch.all(det_cut[~passes] == 0.0)


# --------------------------------------------------------------------------
# same-flavour opposite-sign pairs
# --------------------------------------------------------------------------

# d d~ > e- e+ mu- mu+
LEPTON_PIDS = [1, -1, 11, -11, 13, -13]
LEPTONS = [11, -11, 13, -13]


def lepton_event(n=2_000):
    return sample(PSM([0.0] * 6, CM_ENERGY, mode=PSM.rambo), SEED, n)[0]


def test_sfos_pairs_are_particle_antiparticle():
    observable = O(LEPTON_PIDS, O.obs_sfos_pair_mass, [LEPTONS])
    p = lepton_event()
    values = observable(p).reshape(len(p), -1)
    # e- e+ and mu- mu+ only, not e- mu+ nor the like-sign pairs
    assert values.shape[1] == 2
    expected = torch.stack([invariant_mass(p, 2, 3), invariant_mass(p, 4, 5)], dim=1)
    # a nearly collinear pair loses digits in m^2 = E^2 - |p|^2, hence the atol
    torch.testing.assert_close(
        values.sort(dim=1).values, expected.sort(dim=1).values, rtol=1e-9, atol=1e-6
    )


def test_sfos_cut_floors_only_its_pairs():
    sfos = ms.Cuts(
        [ms.CutItem(O(LEPTON_PIDS, O.obs_sfos_pair_mass, [LEPTONS]), min=50.0)]
    )
    every = ms.Cuts([ms.CutItem(O(LEPTON_PIDS, O.obs_pair_mass, [LEPTONS]), min=50.0)])
    expected = torch.zeros((4, 4))
    expected[0, 1] = expected[1, 0] = expected[2, 3] = expected[3, 2] = 50.0
    assert torch.equal(torch.tensor(sfos.m_inv_min()), expected)
    assert torch.all(
        torch.tensor(every.m_inv_min())[~torch.eye(4, dtype=torch.bool)] == 50.0
    )


def test_sfos_cut_filters_like_the_numpy_definition():
    cuts = ms.Cuts(
        [ms.CutItem(O(LEPTON_PIDS, O.obs_sfos_pair_mass, [LEPTONS]), min=500.0)]
    )
    p = lepton_event(20_000)
    mask = cuts(p)
    expected = (invariant_mass(p, 2, 3) >= 500.0) & (invariant_mass(p, 4, 5) >= 500.0)
    assert torch.equal(mask.bool(), expected)


def test_without_an_sfos_pair_the_cut_is_inactive():
    """e- mu+: nothing to cut on, like MadEvent's mmll."""
    pids = [1, -1, 11, -13]
    cuts = ms.Cuts([ms.CutItem(O(pids, O.obs_sfos_pair_mass, [LEPTONS]), min=500.0)])
    p = sample(PSM([0.0] * 4, CM_ENERGY, mode=PSM.rambo), SEED, 2_000)[0]
    assert torch.all(cuts(p) == 1.0)


def test_sfos_cannot_be_ordered():
    with pytest.raises(ValueError, match="ordered"):
        O(
            LEPTON_PIDS,
            O.obs_sfos_pair_mass,
            [LEPTONS, LEPTONS],
            order_observable=O.obs_pt,
            order_indices=[1, 2],
        )


# --------------------------------------------------------------------------
# every pair cut at once
# --------------------------------------------------------------------------
#
# d d~ > e+ e- mu+ mu- through an s-channel Z (the root) decaying to a lepton
# and an off-shell lepton, which emits a photon decaying to the last pair (the
# 4-lepton output's channel 2). With every lepton pair above 50 GeV no single
# pair bounds the Z beyond 50 GeV, so the root's Breit-Wigner kept sampling
# near m_Z and only ~0.03% of the points passed. All six pairs together put the four
# leptons above sqrt(6) * 50 GeV, the three under the off-shell lepton above
# sqrt(3) * 50 GeV.

ALL_PAIRS_CUT = 50.0
FOUR_LEPTON_PIDS = [1, -1, -11, 11, -13, 13]
FOUR_LEPTON_CHANNEL = (
    [(M_Z, W_Z, 23), (0.0, 0.0, 11), (0.0, 0.0, 22)],
    [["i0", "i1", "p0"], ["o2", "p0", "p1"], ["o0", "o1", "p2"], ["p1", "o3", "p2"]],
)


def all_pairs_cuts():
    observable = O(FOUR_LEPTON_PIDS, O.obs_pair_mass, [FOUR_LEPTON_PIDS[2:]])
    return ms.Cuts([ms.CutItem(observable, min=ALL_PAIRS_CUT)])


def all_pairs_pass(p):
    return torch.stack(
        [
            invariant_mass(p, i, j) >= ALL_PAIRS_CUT
            for i in range(2, 6)
            for j in range(i + 1, 6)
        ]
    ).all(dim=0)


def test_all_pairs_bound_the_sets_holding_them():
    mapping = PSM(
        topology(FOUR_LEPTON_CHANNEL, outgoing_masses=[0.0] * 4),
        CM_ENERGY,
        cuts=all_pairs_cuts(),
    )
    p, det = sample(mapping, SEED)
    kept = det > 0
    assert invariant_mass(p, 0, 1)[kept].min() >= math.sqrt(6) * ALL_PAIRS_CUT * (
        1 - 1e-9
    )
    assert invariant_mass(p, 2, 3, 5)[kept].min() >= math.sqrt(3) * ALL_PAIRS_CUT * (
        1 - 1e-9
    )
    assert all_pairs_pass(p).double().mean() > 0.05


def test_all_pairs_volume_matches_rambo():
    flat = PSM([0.0] * 6, CM_ENERGY, mode=PSM.rambo)
    p, det = sample(flat, SEED, VOLUME_BATCH_SIZE)
    ref = torch.where(all_pairs_pass(p), det, 0.0)
    mapping = PSM(
        topology(FOUR_LEPTON_CHANNEL, outgoing_masses=[0.0] * 4),
        CM_ENERGY,
        cuts=all_pairs_cuts(),
    )
    _, weight = sample(mapping, SEED + 1, VOLUME_BATCH_SIZE)
    error = math.hypot(std(weight), std(ref)) / math.sqrt(VOLUME_BATCH_SIZE)
    assert abs(weight.mean() - ref.mean()) < 5.0 * error
