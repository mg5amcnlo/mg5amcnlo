"""Angular and rapidity ranges implied by pt and eta cuts.

The partonic centre-of-mass frame is the lab up to a boost along the beam with
rapidity Y = log(x1 / x2) / 2. Two consequences of the cuts follow, and
PhaseSpaceMapping samples only the region they leave:

  * every final-state particle has |y| <= |eta|, and the rapidity of a sum of
    momenta is a weighted mean of theirs, so when every particle has an eta
    cut, |Y| <= the largest of them;
  * in a two-body decay of the partonic system (the root of an s-channel
    topology) both products have pt = p sin(theta), and a product's lab
    rapidity is Y + atanh(beta cos(theta)), so its pt and eta cuts bound
    cos(theta) from both sides. A composite product has no pt cut of its own,
    but its rapidity is bounded like Y when every particle in it has an eta
    cut.

Also checked: sqrt(s_hat) >= sum_i m_T,i minimised over transverse momenta
that add up to zero, which is more than the plain sum when one pt cut
dominates.

Pinned down here: no generated point leaves the windows, the windows are
reached, the integral over the cut region does not change, the mapping still
inverts, and nothing is restricted when the cuts do not imply it.
"""

import math

import pytest
import torch
from madspace_testing import generator, rand, randint

import madspace as ms

O = ms.Observable

CM_ENERGY = 13000.0
BATCH_SIZE = 100_000
SEED = 20261006
M_Z = 91.188

PT_LEPTON = 10.0
ETA_LEPTON = 2.5
PT_JET = 20.0
ETA_JET = 2.5


def s_channel_two_body(masses=(0.0, 0.0)):
    return ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            list(masses),
            [ms.Propagator(0.0, 0.0)],
            [["i0", "i1", "p0"], ["p0", "o0", "o1"]],
        )
    )


def s_channel_four_body():
    """u u~ > a* > (a* > e+ e-) (g* > g g): the root decays into two
    composites."""
    return ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [ms.Propagator(0.0, 0.0), ms.Propagator(0.0, 0.0), ms.Propagator(0.0, 0.0)],
            [
                ["i0", "i1", "p0"],
                ["p0", "p1", "p2"],
                ["p1", "o0", "o1"],
                ["p2", "o2", "o3"],
            ],
        )
    )


DY_PIDS = [2, -2, -11, 11]
ZG_PIDS = [2, -2, 23, 21]
WG_PIDS = [2, -1, -11, 12, 21]
EEGG_PIDS = [2, -2, -11, 11, 21, 21]


def lepton_cuts(pids, pt=PT_LEPTON, eta=ETA_LEPTON):
    items = [ms.CutItem(O(pids, O.obs_pt, [O.lepton_pids]), min=pt)]
    if eta is not None:
        items.append(ms.CutItem(O(pids, O.obs_eta_abs, [O.lepton_pids]), max=eta))
    return items


def jet_cuts(pids, pt=PT_JET, eta=ETA_JET):
    items = [ms.CutItem(O(pids, O.obs_pt, [O.jet_pids]), min=pt)]
    if eta is not None:
        items.append(ms.CutItem(O(pids, O.obs_eta_abs, [O.jet_pids]), max=eta))
    return items


def finite(p_ext):
    return torch.isfinite(p_ext).all(dim=2).all(dim=1)


def sample(mapping, seed=SEED, n=BATCH_SIZE):
    r = rand((n, mapping.random_dim()), generator=generator(seed))
    p_ext, x1, x2, det = mapping.map_forward([r])
    ok = torch.isfinite(det) & finite(p_ext)
    return r, p_ext, x1, x2, torch.where(ok, det, 0.0)


def passes(cuts, p_ext):
    return cuts(p_ext).reshape(-1) > 0.5


def fraction(mask):
    """The fraction of True entries of a boolean tensor."""
    return mask.double().mean().item()


def mean_and_error(w):
    """Mean of a batch and its standard error (numpy's std, without Bessel)."""
    return w.mean().item(), w.std(correction=0).item() / math.sqrt(len(w))


def pt(p):
    return torch.hypot(p[..., 1], p[..., 2])


def rapidity(p):
    return 0.5 * torch.log((p[..., 0] + p[..., 3]) / (p[..., 0] - p[..., 3]))


def eta(p):
    p_mag = torch.sqrt(torch.sum(p[..., 1:] ** 2, dim=-1))
    return 0.5 * torch.log((p_mag + p[..., 3]) / (p_mag - p[..., 3]))


def beam_rapidity(x1, x2):
    return 0.5 * torch.log(x1 / x2)


def sqrt_s_hat(x1, x2):
    return CM_ENERGY * torch.sqrt(x1 * x2)


def assert_close(actual, expected, rel=0.0, abs=0.0):
    """Like pytest.approx for tensors: |actual - expected| <= abs + rel * |expected|."""
    expected = torch.as_tensor(expected, dtype=actual.dtype).expand_as(actual)
    torch.testing.assert_close(actual, expected, rtol=rel, atol=abs)


def weighted_cut_integral(mapping, cuts, seed, n=400_000):
    """det / s_hat over the cut region: a propagator-shaped test integrand,
    so the low-mass region carries weight."""
    _, p_ext, x1, x2, det = sample(mapping, seed=seed, n=n)
    weight = torch.where(passes(cuts, p_ext), det / (CM_ENERGY**2 * x1 * x2), 0.0)
    return mean_and_error(weight)


def assert_same_integral(topology, cuts, n=400_000):
    free = ms.PhaseSpaceMapping(topology, CM_ENERGY)
    piped = ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=cuts)
    mean_free, err_free = weighted_cut_integral(free, cuts, SEED, n)
    mean_piped, err_piped = weighted_cut_integral(piped, cuts, SEED + 1, n)
    error = math.hypot(err_free, err_piped)
    assert 0.0 < error < 0.02 * mean_free
    assert abs(mean_free - mean_piped) < 5.0 * error
    return mean_free, err_free, mean_piped, err_piped


# --------------------------------------------------------------------------
# Drell-Yan: the two windows are the whole cut
# --------------------------------------------------------------------------


def dy_cuts():
    return ms.Cuts(lepton_cuts(DY_PIDS))


def test_dy_every_generated_point_passes():
    """For u u~ > e+ e- the pt and eta cuts are exactly the cos(theta) and Y
    windows, so with the cuts handed to the mapping every point it generates
    with a weight passes them, while without them most fail."""
    c = dy_cuts()
    _, p_free, _, _, det_free = sample(
        ms.PhaseSpaceMapping(s_channel_two_body(), CM_ENERGY)
    )
    assert fraction(passes(c, p_free)[det_free > 0]) < 0.9
    _, p_ext, _, _, det = sample(
        ms.PhaseSpaceMapping(s_channel_two_body(), CM_ENERGY, cuts=c)
    )
    assert fraction(det > 0) > 0.999
    assert fraction(passes(c, p_ext)[det > 0]) > 0.999


def test_dy_windows_are_reached():
    """The bounds sampled are the cuts themselves, not something stricter."""
    # all sampled points, as the weight already contains the cut mask
    _, p_ext, x1, x2, _ = sample(
        ms.PhaseSpaceMapping(s_channel_two_body(), CM_ENERGY, cuts=dy_cuts())
    )
    leptons = p_ext[:, 2:4]
    y = beam_rapidity(x1, x2)
    assert torch.abs(y).max() <= ETA_LEPTON + 1e-9
    assert torch.abs(y).max() > 0.99 * ETA_LEPTON
    assert torch.abs(eta(leptons)).max() <= ETA_LEPTON + 1e-6
    assert torch.abs(eta(leptons)).max() > 0.999 * ETA_LEPTON
    assert pt(leptons).min() >= PT_LEPTON * (1 - 1e-9)
    assert pt(leptons).min() < 1.01 * PT_LEPTON


def test_dy_cut_integral_unchanged():
    assert_same_integral(s_channel_two_body(), dy_cuts())


def test_dy_round_trip():
    mapping = ms.PhaseSpaceMapping(s_channel_two_body(), CM_ENERGY, cuts=dy_cuts())
    r, p_ext, x1, x2, det = sample(mapping, n=10_000)
    keep = det > 0
    r_back, det_back = mapping.map_inverse([p_ext, x1, x2], [])
    # the cut mask is part of det but not of det_back
    assert_close(r_back[keep], r[keep], abs=1e-7)
    assert_close((det * det_back)[keep], 1.0, rel=1e-7)


# --------------------------------------------------------------------------
# a massive product with no cut: only the decay window applies
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pt_jet", [PT_JET, 200.0])
def test_massive_partner_without_cuts(pt_jet):
    """u u~ > Z g: the Z has no eta cut, so Y is unbounded, but the gluon's pt
    and eta cuts (with the Z's beta != 1 on the other side) still fix the
    cos(theta) range. Nearly every sampled point passes; the few that do not
    are those where |Y| is so large that no angle is left (weight zero)."""
    c = ms.Cuts(jet_cuts(ZG_PIDS, pt=pt_jet))
    topology = s_channel_two_body((M_Z, 0.0))
    _, p_free, _, _, _ = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY))
    assert fraction(passes(c, p_free)) < 0.9
    _, p_ext, x1, x2, det = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=c))
    assert fraction(passes(c, p_ext)) > 0.99
    assert fraction(passes(c, p_ext)[det > 0]) > 0.999
    # Y is not restricted
    assert torch.abs(beam_rapidity(x1, x2)[det > 0]).max() > ETA_JET + 0.5
    assert_same_integral(topology, c)


def test_balanced_transverse_mass_floor():
    """With pt_g >= 200 GeV the Z must carry 200 GeV of pt too, so
    sqrt(s_hat) >= sqrt(m_Z^2 + 200^2) + 200, above the m_Z + 200 the plain
    sum of transverse masses gives."""
    pt_jet = 200.0
    c = ms.Cuts(jet_cuts(ZG_PIDS, pt=pt_jet))
    topology = s_channel_two_body((M_Z, 0.0))
    _, _, x1, x2, _ = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=c))
    floor = math.hypot(M_Z, pt_jet) + pt_jet
    assert sqrt_s_hat(x1, x2).min() >= floor * (1 - 1e-9)
    assert sqrt_s_hat(x1, x2).min() < 1.01 * floor


def test_balanced_floor_with_massless_partners():
    """u d~ > e+ ve g with pt_g >= 200 and only the lepton cut otherwise: the
    neutrino (massless, no cut) and the lepton must make up the gluon's pt,
    at full cost, so sqrt(s_hat) >= 2 * 200 + 0."""
    topology = ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            [0.0, 0.0, 0.0],
            [ms.Propagator(0.0, 0.0), ms.Propagator(0.0, 0.0)],
            [["i0", "i1", "p0"], ["p0", "p1", "o2"], ["p1", "o0", "o1"]],
        )
    )
    c = ms.Cuts(lepton_cuts(WG_PIDS) + jet_cuts(WG_PIDS, pt=200.0, eta=None))
    _, _, x1, x2, _ = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=c))
    assert sqrt_s_hat(x1, x2).min() >= 400.0 * (1 - 1e-9)
    assert sqrt_s_hat(x1, x2).min() < 1.01 * 400.0


# --------------------------------------------------------------------------
# composite products: rapidity bounds without pt bounds
# --------------------------------------------------------------------------


def eegg_cuts(eta_jet=ETA_JET):
    return ms.Cuts(lepton_cuts(EEGG_PIDS) + jet_cuts(EEGG_PIDS, eta=eta_jet))


def test_composite_rapidity_windows():
    """The root of u u~ > (e+ e-) (g g) decays into two composites. With every
    particle eta-cut, |Y| and the rapidities of both composites stay within
    the largest eta cut, and the cut integral does not move."""
    c = eegg_cuts()
    topology = s_channel_four_body()
    _, p_ext, x1, x2, det = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=c))
    # all sampled points, as the weight already contains the cut mask
    keep = finite(p_ext)
    eta_max = max(ETA_LEPTON, ETA_JET)
    assert torch.abs(beam_rapidity(x1, x2)[keep]).max() <= eta_max + 1e-9
    for pair in ((2, 3), (4, 5)):
        system = p_ext[keep, pair[0]] + p_ext[keep, pair[1]]
        assert torch.abs(rapidity(system)).max() <= eta_max + 1e-6
    assert_same_integral(topology, c)


def test_composite_round_trip():
    mapping = ms.PhaseSpaceMapping(s_channel_four_body(), CM_ENERGY, cuts=eegg_cuts())
    r, p_ext, x1, x2, det = sample(mapping, n=10_000)
    keep = det > 0
    r_back, det_back = mapping.map_inverse([p_ext, x1, x2], [])
    # a handful of points lose ~1e-5 to 1e-4 (platform dependent) in the inner decays
    # with or without the windows (momentum reconstruction), hence quantiles
    r_diff = torch.abs(r_back[keep] - r[keep])
    det_diff = torch.abs((det * det_back)[keep] - 1.0)
    assert torch.quantile(r_diff, 0.999) < 1e-7 and r_diff.max() < 1e-3
    assert (
        torch.quantile(det_diff, 0.99) < 1e-7 and torch.quantile(det_diff, 0.999) < 1e-5
    )


# --------------------------------------------------------------------------
# where nothing follows
# --------------------------------------------------------------------------


def test_no_rapidity_window_with_an_uncut_particle():
    """Without an eta cut on the gluons, neither Y nor the gluon system's
    rapidity is bounded, and both go beyond the lepton cut."""
    c = ms.Cuts(lepton_cuts(EEGG_PIDS) + jet_cuts(EEGG_PIDS, eta=None))
    _, p_ext, x1, x2, det = sample(
        ms.PhaseSpaceMapping(s_channel_four_body(), CM_ENERGY, cuts=c)
    )
    keep = det > 0
    assert torch.abs(beam_rapidity(x1, x2)[keep]).max() > ETA_LEPTON + 1.0
    gluons = p_ext[keep, 4] + p_ext[keep, 5]
    assert torch.abs(rapidity(gluons)).max() > ETA_LEPTON + 1.0


def test_no_cuts_no_change():
    """Empty cuts give the same mapping as no cuts."""
    topology = s_channel_two_body()
    a = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY))
    b = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=ms.Cuts(4)))
    for u, v in zip(a, b):
        assert torch.equal(u, v)


def test_two_body_decay_window_alone():
    """TwoBodyDecay(com=True, pt_min=...) on its own: cos(theta) stays in
    |cos| <= sqrt(1 - (pt_min / p)^2), the weight is the kept fraction, and it
    inverts."""
    pt_min = 30.0
    decay = ms.TwoBodyDecay(True, pt_min=pt_min)
    n = 10_000
    g = generator(SEED)
    r_phi, r_cos = rand(n, generator=g), rand(n, generator=g)
    m0 = torch.full((n,), 100.0)
    zeros = torch.zeros(n)
    ones = torch.ones(n)
    p1, p2, det = decay.map_forward([r_phi, r_cos, m0, zeros, zeros], [ones, ones])
    plain = ms.TwoBodyDecay(True)
    _, _, det_plain = plain.map_forward([r_phi, r_cos, m0, zeros, zeros], [])
    kept = math.sqrt(1 - (pt_min / 50.0) ** 2)
    assert_close(det, det_plain * kept, rel=1e-12)
    assert pt(p1).min() >= pt_min * (1 - 1e-9)
    *inputs, det_back = decay.map_inverse([p1, p2], [ones, ones])
    assert_close(inputs[1], r_cos, abs=1e-8)
    assert_close(det * det_back, 1.0, rel=1e-8)


# --------------------------------------------------------------------------
# coverage: no event passing the cuts lies outside the windows
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topology,cuts",
    [
        (s_channel_two_body(), dy_cuts()),
        (s_channel_two_body((M_Z, 0.0)), ms.Cuts(jet_cuts(ZG_PIDS, pt=200.0))),
        (s_channel_four_body(), eegg_cuts()),
    ],
    ids=["dy", "zg", "eegg"],
)
def test_every_passing_event_is_inside_the_windows(topology, cuts):
    """Exactness, pointwise: events generated without the windows and passing
    the cuts all map back, through the windowed mapping's inverse, to random
    numbers inside the unit cube. With the round trip above this means the
    windowed mapping covers the whole cut region with the right density."""
    free = ms.PhaseSpaceMapping(topology, CM_ENERGY)
    windowed = ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=cuts)
    _, p_ext, x1, x2, det = sample(free, n=400_000)
    keep = passes(cuts, p_ext) & (det > 0)
    assert keep.sum() > 1000
    r_back, _ = windowed.map_inverse([p_ext[keep], x1[keep], x2[keep]], [])
    assert torch.all(torch.isfinite(r_back))
    assert r_back.min() > -1e-6 and r_back.max() < 1 + 1e-6


# --------------------------------------------------------------------------
# t channel: the first scattering, between the two beams
# --------------------------------------------------------------------------
#
# In the partonic centre-of-mass frame the first scattering of a t-channel
# chain, pa pb -> R k, has t = (pb - k)^2, which fixes pb.k, while
# (pa + pb).k follows from the masses, so pa.k is fixed too. The rapidity
# log(pb.k / pa.k) / 2 of the peeled particle, and that of the recoil R, are
# then monotonic functions of |t|, and their bounds give an interval of |t|.

AA_PIDS = [2, -2, 22, 22]
AAG_PIDS = [2, -2, 22, 21, 22]  # a g a along the chain
ETA_PHOTON = 2.5
PT_PHOTON = 20.0


def t_channel_two_body():
    return ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            [0.0, 0.0],
            [ms.Propagator(0.0, 0.0)],
            [["i0", "o0", "p0"], ["p0", "i1", "o1"]],
        )
    )


def t_channel_three_body():
    """u u~ > a g a (with AAG_PIDS) through two t-channel propagators."""
    return ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            [0.0, 0.0, 0.0],
            [ms.Propagator(0.0, 0.0), ms.Propagator(0.0, 0.0)],
            [["i0", "o0", "p0"], ["p0", "o1", "p1"], ["p1", "i1", "o2"]],
        )
    )


def photon_cuts(pids, eta=ETA_PHOTON):
    items = [ms.CutItem(O(pids, O.obs_pt, [O.photon_pids]), min=PT_PHOTON)]
    if eta is not None:
        items.append(ms.CutItem(O(pids, O.obs_eta_abs, [O.photon_pids]), max=eta))
    return items


def test_t_channel_every_generated_point_passes():
    """u u~ > a a through the t channel: with the pt cut and the photons' eta
    cuts bounding |t| and Y, every sampled point passes, while without the
    cuts handed over many do not."""
    c = ms.Cuts(photon_cuts(AA_PIDS))
    topology = t_channel_two_body()
    _, p_free, _, _, _ = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY))
    assert fraction(passes(c, p_free)) < 0.9
    _, p_ext, _, _, det = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=c))
    assert fraction(passes(c, p_ext)) > 0.999
    photons = p_ext[:, 2:4]
    assert torch.abs(eta(photons)).max() <= ETA_PHOTON + 1e-6
    assert torch.abs(eta(photons)).max() > 0.999 * ETA_PHOTON


def test_t_channel_round_trip():
    c = ms.Cuts(photon_cuts(AA_PIDS))
    mapping = ms.PhaseSpaceMapping(t_channel_two_body(), CM_ENERGY, cuts=c)
    r, p_ext, x1, x2, det = sample(mapping, n=10_000)
    keep = det > 0
    r_back, det_back = mapping.map_inverse([p_ext, x1, x2], [])
    assert_close(r_back[keep], r[keep], abs=1e-7)
    assert_close((det * det_back)[keep], 1.0, rel=1e-7)


@pytest.mark.parametrize(
    "topology,cuts",
    [
        (t_channel_two_body(), ms.Cuts(photon_cuts(AA_PIDS))),
        (
            t_channel_three_body(),
            ms.Cuts(photon_cuts(AAG_PIDS) + jet_cuts(AAG_PIDS, eta=ETA_JET)),
        ),
        # the gluon has no eta cut: only the peeled photon bounds the first |t|
        (
            t_channel_three_body(),
            ms.Cuts(photon_cuts(AAG_PIDS) + jet_cuts(AAG_PIDS, eta=None)),
        ),
    ],
    ids=["aa", "aga", "aga-gluon-uncut"],
)
def test_t_channel_window_is_exact(topology, cuts):
    """Every event passing the cuts lies inside the windows (it maps back into
    the unit cube), and the integral over the cut region is unchanged."""
    free = ms.PhaseSpaceMapping(topology, CM_ENERGY)
    windowed = ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=cuts)
    _, p_ext, x1, x2, det = sample(free, n=400_000)
    keep = passes(cuts, p_ext) & (det > 0)
    assert keep.sum() > 1000
    r_back, _ = windowed.map_inverse([p_ext[keep], x1[keep], x2[keep]], [])
    assert torch.all(torch.isfinite(r_back))
    assert r_back.min() > -1e-6 and r_back.max() < 1 + 1e-6
    assert_same_integral(topology, cuts)


def test_t_channel_first_step_only_bounds_what_it_peels():
    """In u u~ > a g a the first scattering peels a photon from one end of the
    chain; the gluon (no eta cut) makes the recoil unbounded, so the other
    photon leaves its eta range as often as without the window (the cut
    removes it), the peeled one only where the eta and pt bounds together
    leave no |t| at all: the clamp then keeps the full range rather than an
    empty one, and the cuts reject every such point."""
    topology = t_channel_three_body()
    order = topology.t_integration_order
    peeled = order[0] + (order[0] == len(order) - 1)
    assert peeled in (0, 2)  # a photon
    other = 2 - peeled
    cuts = ms.Cuts(photon_cuts(AAG_PIDS) + jet_cuts(AAG_PIDS, eta=None))
    _, p_ext, _, _, _ = sample(ms.PhaseSpaceMapping(topology, CM_ENERGY, cuts=cuts))
    outside = torch.abs(eta(p_ext[:, 2:5])) > ETA_PHOTON + 1e-6
    assert fraction(outside[:, peeled]) < 0.03
    assert fraction(outside[:, other]) > 0.2


# --------------------------------------------------------------------------
# colour-ordered chains: the blocks that scatter the two beams
# --------------------------------------------------------------------------
#
# ColorOrderedMapping uses the rapidity bounds where a block's incoming momenta
# are the beams: the central 2->2 (the two colour sides), the double-t central
# block (|t2| at fixed |t1|: pa.p1 and pb.p1 are (m1^2 + |t1|)/2 and
# (m1^2 + |t2|)/2) and the first peel of a single chain.

GGGG_PIDS = [2, -2, 21, 21, 21, 21]
CO_ORDERS = {
    "chain": [0, 2, 3, 4, 5, 1],  # one chain: first peel is beam-beam
    "2+2": [0, 2, 3, 1, 4, 5],  # central 2->2 between {0, 1} and {2, 3}
    "1+3": [0, 2, 1, 3, 4, 5],  # double-t: {0} against {1, 2, 3}
}
# outgoing particles whose summed momentum has to stay within the bound
CO_BOUNDED = {
    "chain": [[0], [1, 2, 3]],
    "2+2": [[0, 1], [2, 3]],
    "1+3": [[0], [1, 2, 3]],
}


def gggg_cuts():
    return ms.Cuts(
        jet_cuts(GGGG_PIDS)
        + [ms.CutItem(O(GGGG_PIDS, O.obs_delta_r, [O.jet_pids]), min=0.4)]
    )


def co_sample(mapping, n, seed):
    g = generator(seed)
    r = rand((n, mapping.random_dim()), generator=g)
    d = randint(0, 2, (n, mapping.discrete_dim()), generator=g, dtype=torch.int32)
    p_ext, x1, x2, det = mapping.map_forward([r, d] if mapping.discrete_dim() else [r])
    ok = torch.isfinite(det) & finite(p_ext)
    return p_ext, x1, x2, torch.where(ok, det, 0.0)


@pytest.mark.parametrize("name", list(CO_ORDERS))
def test_color_ordered_window_is_exact(name):
    """Every event passing the cuts maps back into the unit cube through the
    windowed colour-ordered mapping, and the cut-region integral is
    unchanged."""
    order = CO_ORDERS[name]
    cuts = gggg_cuts()
    masses = [0.0] * 6

    def mapping(c):
        return ms.PhaseSpaceMapping(
            masses,
            CM_ENERGY,
            mode=ms.PhaseSpaceMapping.color_ordered,
            cuts=c,
            color_ordered=ms.ColorOrderedMapping.Options(order),
        )

    n = 200_000
    p_free, y1, y2, det_free = co_sample(mapping(ms.Cuts(6)), n, SEED)
    keep = passes(cuts, p_free) & (det_free > 0)
    assert keep.sum() > 1000
    out = mapping(cuts).map_inverse([p_free[keep], y1[keep], y2[keep]], [])
    r_back = out[0]
    assert torch.all(torch.isfinite(r_back))
    assert r_back.min() > -1e-6 and r_back.max() < 1 + 1e-6

    def integral(p_ext, x1, x2, det):
        w = torch.where(passes(cuts, p_ext), det / (CM_ENERGY**2 * x1 * x2), 0.0)
        return mean_and_error(w)

    a, ea = integral(p_free, y1, y2, det_free)
    b, eb = integral(*co_sample(mapping(cuts), n, SEED + 1))
    assert abs(a - b) < 5.0 * math.hypot(ea, eb)


@pytest.mark.parametrize("name", list(CO_ORDERS))
def test_color_ordered_blocks_respect_the_bounds(name):
    """ColorOrderedMapping on its own, at fixed sqrt(s_hat) and boost: with
    y_max the momenta each beam-beam block emits stay within the bound (up
    to the points where the pt and rapidity bounds leave nothing, for which
    the block keeps its full range), and without it they do not."""
    order = CO_ORDERS[name]
    sqrt_s, y_boost = 500.0, 0.8
    tau = (sqrt_s / CM_ENERGY) ** 2
    x1, x2 = math.sqrt(tau) * math.exp(y_boost), math.sqrt(tau) * math.exp(-y_boost)
    n = 100_000
    g = generator(SEED)

    def outside(y_max):
        mapping = ms.ColorOrderedMapping(
            ms.ColorOrderedMapping.Options(order), [PT_JET] * 4, [], [], y_max
        )
        r = rand((n, mapping.random_dim()), generator=g)
        d = randint(0, 2, (n, mapping.discrete_dim()), generator=g, dtype=torch.int32)
        inputs = [r[:, i].clone() for i in range(r.shape[1])]
        inputs += [d[:, j].clone() for j in range(d.shape[1])]
        conditions = [torch.full((n,), sqrt_s)] + [torch.zeros(n)] * 4
        if y_max:
            conditions += [torch.full((n,), x1), torch.full((n,), x2)]
        out = mapping.map_forward(inputs, conditions)
        p = torch.stack(list(out[:-1]), dim=1)
        fractions = []
        for group in CO_BOUNDED[name]:
            system = p[finite(p)][:, [2 + i for i in group]].sum(dim=1)
            fractions.append(
                fraction(torch.abs(rapidity(system) + y_boost) > ETA_JET + 1e-9)
            )
        return fractions

    for with_bound, without in zip(outside([ETA_JET] * 4), outside([])):
        assert with_bound < 0.03
        assert without > 0.1 or with_bound <= without


def test_double_t_window_alone():
    """DoubleT between the beams, with transverse-energy floors: with rapidity
    bounds on the single particle and the recoil, |t1| is narrowed to values
    that leave some |t2| (using (2 pa.q)(2 pb.q) = s M_T(q)^2) and |t2| at
    fixed |t1|, so neither momentum ever leaves its bound; without them the
    single particle does in a fifth of the points."""
    n = 50_000
    sqrt_s, y_boost = 500.0, 0.8
    tau = (sqrt_s / CM_ENERGY) ** 2
    x1, x2 = math.sqrt(tau) * math.exp(y_boost), math.sqrt(tau) * math.exp(-y_boost)
    pa = torch.tensor([sqrt_s / 2, 0.0, 0.0, sqrt_s / 2]).repeat(n, 1)
    pb = torch.tensor([sqrt_s / 2, 0.0, 0.0, -sqrt_s / 2]).repeat(n, 1)
    g = generator(SEED)
    r = [rand(n, generator=g) for _ in range(3)]
    base = [
        pa,
        pb,
        torch.zeros(n),
        torch.zeros(n),
        torch.full((n,), 20.0),
        torch.full((n,), 40.0),
    ]

    def outside(mapping, conditions):
        p1, p2, det = mapping.map_forward(r, conditions)
        ok = torch.isfinite(det) & (det > 0)
        return [
            fraction(torch.abs(rapidity(p[ok]) + y_boost) > ETA_JET + 1e-9)
            for p in (p1, p2)
        ]

    bounded = ms.DoubleT(0.8, 0.0, 0.0, 0.8, 0.0, 0.0, True, ETA_JET, ETA_JET, 1.0)
    free = ms.DoubleT(0.8, 0.0, 0.0, 0.8, 0.0, 0.0, True)
    assert outside(bounded, base + [torch.full((n,), x1), torch.full((n,), x2)]) == [
        0.0,
        0.0,
    ]
    assert outside(free, base)[0] > 0.1
    *inputs, det_back = bounded.map_inverse(
        list(
            bounded.map_forward(r, base + [torch.full((n,), x1), torch.full((n,), x2)])[
                :2
            ]
        ),
        base + [torch.full((n,), x1), torch.full((n,), x2)],
    )
    for a, b in zip(inputs, r):
        assert_close(a, b, abs=1e-7)


# --------------------------------------------------------------------------
# channel permutations: the windows hold for every permutation
# --------------------------------------------------------------------------
#
# One mapping serves all permutations of a channel (event[i] = topology[perm[i]]),
# so the bounds it uses are the weakest over the permutations. Here the two
# outgoing particles swap places between a lepton slot (pt > 10, |eta| < 2.5)
# and a photon slot (pt > 30, |eta| < 1.5): each topology particle may end up in
# either, so it only gets pt > 10 and |eta| < 2.5.

LA_PIDS = [2, -2, 11, 22]
PERMUTATIONS = [[0, 1, 2, 3], [0, 1, 3, 2]]


def lepton_photon_cuts(pids=LA_PIDS):
    return ms.Cuts(
        lepton_cuts(pids)
        + [
            ms.CutItem(O(pids, O.obs_pt, [O.photon_pids]), min=30.0),
            ms.CutItem(O(pids, O.obs_eta_abs, [O.photon_pids]), max=1.5),
        ]
    )


def permuted_sample(mapping, index, n, seed):
    r = rand((n, mapping.random_dim()), generator=generator(seed))
    perm = torch.full((n,), index, dtype=torch.int32)
    p_ext, x1, x2, det = mapping.map_forward([r], [perm])
    ok = torch.isfinite(det) & finite(p_ext)
    return r, perm, p_ext, x1, x2, torch.where(ok, det, 0.0)


@pytest.mark.parametrize("index", [0, 1])
def test_windows_hold_for_every_permutation(index):
    """For each permutation: the windows bind (|Y| stays below 2.5, and far
    more points pass than without them), every event passing the cuts maps
    back into the unit cube, and the cut-region integral is unchanged."""
    cuts = lepton_photon_cuts()
    topology = s_channel_two_body()
    free = ms.PhaseSpaceMapping(topology, CM_ENERGY, permutations=PERMUTATIONS)
    windowed = ms.PhaseSpaceMapping(
        topology, CM_ENERGY, cuts=cuts, permutations=PERMUTATIONS
    )
    n = 200_000
    _, perm, p_free, y1, y2, det_free = permuted_sample(free, index, n, SEED)
    _, _, p_win, z1, z2, det_win = permuted_sample(windowed, index, n, SEED + 1)

    assert torch.abs(beam_rapidity(z1, z2)).max() <= ETA_LEPTON + 1e-9
    assert fraction(passes(cuts, p_win)) > fraction(passes(cuts, p_free)) + 0.1

    keep = passes(cuts, p_free) & (det_free > 0)
    assert keep.sum() > 1000
    r_back, _ = windowed.map_inverse([p_free[keep], y1[keep], y2[keep]], [perm[keep]])
    assert r_back.min() > -1e-6 and r_back.max() < 1 + 1e-6

    def integral(p_ext, x1, x2, det):
        w = torch.where(passes(cuts, p_ext), det / (CM_ENERGY**2 * x1 * x2), 0.0)
        return mean_and_error(w)

    a, ea = integral(p_free, y1, y2, det_free)
    b, eb = integral(p_win, z1, z2, det_win)
    assert abs(a - b) < 5.0 * math.hypot(ea, eb)


def test_strongest_bounds_would_lose_permuted_events():
    """The counterpart: windows built from the strongest bounds (the photon
    slot's for both particles) exclude events that pass the cuts in the swapped
    permutation, which the test above would catch."""
    cuts = lepton_photon_cuts()
    topology = s_channel_two_body()
    free = ms.PhaseSpaceMapping(topology, CM_ENERGY, permutations=PERMUTATIONS)
    _, _, p_free, y1, y2, det = permuted_sample(free, 1, 200_000, SEED)
    keep = passes(cuts, p_free) & (det > 0)
    too_tight = ms.PhaseSpaceMapping(
        topology,
        CM_ENERGY,
        cuts=ms.Cuts(
            [
                ms.CutItem(O(AA_PIDS, O.obs_pt, [O.photon_pids]), min=30.0),
                ms.CutItem(O(AA_PIDS, O.obs_eta_abs, [O.photon_pids]), max=1.5),
            ]
        ),
    )
    r_back, _ = too_tight.map_inverse([p_free[keep], y1[keep], y2[keep]], [])
    outside = ~torch.all((r_back > -1e-6) & (r_back < 1 + 1e-6), dim=1)
    assert fraction(outside) > 0.01


# --------------------------------------------------------------------------
# leptonic beams: no luminosity, the lab is the partonic frame (Y = 0)
# --------------------------------------------------------------------------

MUMU_PIDS = [11, -11, 13, -13]
LEPTONIC_ENERGY = 500.0


@pytest.mark.parametrize(
    "topology", [s_channel_two_body(), t_channel_two_body()], ids=["s", "t"]
)
def test_leptonic_windows(topology):
    """e+ e- > mu+ mu- at fixed energy: the decay-angle window (s channel) or
    the |t| window (t channel) makes every point pass, the mapping inverts,
    and every event passing the cuts lies inside the window."""
    # |eta| < 1 so that the window has something to remove at this energy
    cuts = ms.Cuts(lepton_cuts(MUMU_PIDS, eta=1.0))
    n = 50_000
    windowed = ms.PhaseSpaceMapping(topology, LEPTONIC_ENERGY, leptonic=True, cuts=cuts)
    free = ms.PhaseSpaceMapping(topology, LEPTONIC_ENERGY, leptonic=True)
    r = rand((n, windowed.random_dim()), generator=generator(SEED))
    p_ext, _, _, det = windowed.map_forward([r])
    assert fraction(passes(cuts, p_ext)) > 0.999
    ones = torch.ones(n)
    r_back, det_back = windowed.map_inverse([p_ext, ones, ones], [])
    assert_close(r_back, r, abs=1e-9)
    assert_close(det * det_back, 1.0, rel=1e-9)

    p_free, _, _, det_free = free.map_forward([r])
    keep = passes(cuts, p_free) & (det_free > 0)
    assert 1000 < keep.sum() < 0.9 * n
    m = int(keep.sum())
    r_back, _ = windowed.map_inverse([p_free[keep], torch.ones(m), torch.ones(m)], [])
    assert r_back.min() > -1e-6 and r_back.max() < 1 + 1e-6


# --------------------------------------------------------------------------
# kernel edge cases: huge bounds, and bounds that do not bind
# --------------------------------------------------------------------------


def beam_conditions(n, sqrt_s=500.0, y_boost=0.8):
    tau = (sqrt_s / CM_ENERGY) ** 2
    x1 = torch.full((n,), math.sqrt(tau) * math.exp(y_boost))
    x2 = torch.full((n,), math.sqrt(tau) * math.exp(-y_boost))
    pa = torch.tensor([sqrt_s / 2, 0.0, 0.0, sqrt_s / 2]).repeat(n, 1)
    pb = torch.tensor([sqrt_s / 2, 0.0, 0.0, -sqrt_s / 2]).repeat(n, 1)
    return pa, pb, x1, x2


@pytest.mark.parametrize("y_max", [100.0, 1e3, 1e6])
def test_double_t_huge_bound_is_no_bound(y_max):
    """A rapidity bound of 100 or more is no bound: no exp overflows into
    0 * inf = NaN, and the momenta are exactly those without a bound."""
    n = 20_000
    pa, pb, x1, x2 = beam_conditions(n)
    g = generator(SEED)
    r = [rand(n, generator=g) for _ in range(3)]
    base = [
        pa,
        pb,
        torch.zeros(n),
        torch.zeros(n),
        torch.full((n,), 20.0),
        torch.full((n,), 40.0),
    ]
    bounded = ms.DoubleT(0.8, 0.0, 0.0, 0.8, 0.0, 0.0, True, y_max, y_max, 1.0)
    free = ms.DoubleT(0.8, 0.0, 0.0, 0.8, 0.0, 0.0, True)
    out_bounded = bounded.map_forward(r, base + [x1, x2])
    out_free = free.map_forward(r, base)
    for a, b in zip(out_bounded, out_free):
        assert torch.all(torch.isfinite(a))
        assert torch.equal(a, b)


def test_loose_t_bound_leaves_the_range_untouched():
    """A rapidity bound looser than the kinematics changes nothing at all: the
    clamp compares in |t| and keeps the input range bit for bit, instead of
    rebuilding it through pb.k and moving its ends by rounding."""
    n = 20_000
    pa, pb, x1, x2 = beam_conditions(n)
    g = generator(SEED)
    r = [rand(n, generator=g) for _ in range(2)]
    masses = [torch.full((n,), M_Z), torch.zeros(n)]
    loose = ms.TwoToTwoParticleScattering(True, 0.8, 0.0, 0.0, False, 50.0, 50.0, 1.0)
    free = ms.TwoToTwoParticleScattering(True, 0.8, 0.0, 0.0, False)
    out_loose = loose.map_forward(r + masses, [pa, pb, x1, x2])
    out_free = free.map_forward(r + masses, [pa, pb])
    for a, b in zip(out_loose, out_free):
        assert torch.equal(a, b)
