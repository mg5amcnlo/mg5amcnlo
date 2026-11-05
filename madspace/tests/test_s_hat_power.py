"""The power of the s_hat sampling of PhaseSpaceMapping, separate from the
invariant_power of the propagators, and the options of a color-ordered
PhaseSpaceMapping. Unset, the s_hat power is invariant_power, except for a
color-ordered chain with a lower bound on s_hat, where it is AmpliCol's 1.2."""

import math

import pytest
import torch
from madspace_testing import generator, rand, randint

import madspace as ms

MASSES = [0.0] * 6
CM_ENERGY = 13000.0
ORDER = [0, 2, 3, 1, 4, 5]
MODES = [
    (ms.PhaseSpaceMapping.rambo, None),
    (ms.PhaseSpaceMapping.propagator, None),
    (ms.PhaseSpaceMapping.color_ordered, ms.ColorOrderedMapping.Options(ORDER)),
]
IDS = ["rambo", "propagator", "color_ordered"]


def run(mode, color_ordered, seed, n, cuts=None, **kwargs):
    mapping = ms.PhaseSpaceMapping(
        MASSES,
        CM_ENERGY,
        mode=mode,
        color_ordered=color_ordered,
        invariant_power=0.8,
        cuts=cuts,
        **kwargs,
    )
    g = generator(seed)
    inputs = [rand((n, mapping.random_dim()), generator=g)]
    if mapping.discrete_dim():
        inputs.append(
            randint(0, 2, (n, mapping.discrete_dim()), generator=g, dtype=torch.int32)
        )
    p, x1, x2, det = mapping.map_forward(inputs)
    return p, x1 * x2, det * 2.0 ** mapping.discrete_dim()


@pytest.mark.parametrize("mode, color_ordered", MODES, ids=IDS)
def test_unset_is_invariant_power(mode, color_ordered):
    """Without s_hat_power and without a lower bound on s_hat (massless, no
    cuts), s_hat is sampled with invariant_power, in every mode."""
    p0, tau0, det0 = run(mode, color_ordered, 1, 1000)
    p1, tau1, det1 = run(mode, color_ordered, 1, 1000, s_hat_power=0.8)
    assert torch.equal(p0, p1) and torch.equal(tau0, tau1) and torch.equal(det0, det1)


@pytest.mark.parametrize("mode, color_ordered", MODES, ids=IDS)
def test_only_s_hat_changes(mode, color_ordered):
    """A different s_hat power moves the points in s_hat, not the integral."""
    n = 400_000
    _, tau_a, det_a = run(mode, color_ordered, 2, n, s_hat_power=0.8)
    _, tau_b, det_b = run(mode, color_ordered, 3, n, s_hat_power=1.2)
    # more points at small s_hat
    assert torch.median(tau_b) < 0.5 * torch.median(tau_a)
    # the massless volume (~ s_hat^2) convolved with the luminosity, weighted
    # by 1/s_hat^2: finite variance for both powers down to s_hat = 0
    w_a = det_a / (tau_a * CM_ENERGY**2) ** 2
    w_b = det_b / (tau_b * CM_ENERGY**2) ** 2
    err = math.hypot(
        w_a.std(correction=0).item() / math.sqrt(n),
        w_b.std(correction=0).item() / math.sqrt(n),
    )
    assert abs(w_a.mean() - w_b.mean()).item() < 4.0 * err


@pytest.mark.parametrize(
    "masses",
    [[0.0, 0.0, 173.0, 173.0, 0.0, 0.0], [0.0] * 6],
    ids=["top masses", "jet cuts"],
)
def test_unset_color_ordered_bounded(masses):
    """With masses or cuts bounding s_hat from below, a color-ordered chain
    samples s_hat as 1/s_hat^1.2, like AmpliCol, unless told otherwise."""
    cuts = None if masses[2] > 0.0 else jet_cuts()
    options = ms.ColorOrderedMapping.Options(ORDER)
    tau = {}
    for power in (None, 1.2, 0.8):
        kwargs = {} if power is None else dict(s_hat_power=power)
        mapping = ms.PhaseSpaceMapping(
            masses,
            CM_ENERGY,
            mode=ms.PhaseSpaceMapping.color_ordered,
            color_ordered=options,
            cuts=cuts,
            **kwargs,
        )
        g = generator(4)
        r = rand((1000, mapping.random_dim()), generator=g)
        d = randint(
            0, 2, (1000, mapping.discrete_dim()), generator=g, dtype=torch.int32
        )
        tau[power] = mapping.map_forward([r, d])[1]
    assert torch.equal(tau[None], tau[1.2])
    assert not torch.equal(tau[None], tau[0.8])


def jet_cuts():
    pids = [21] * 6
    obs = ms.Observable
    return ms.Cuts(
        [
            ms.CutItem(obs(pids, obs.obs_pt, [obs.jet_pids]), min=20.0),
            ms.CutItem(obs(pids, obs.obs_delta_r, [obs.jet_pids]), min=0.4),
        ]
    )


def test_color_ordered_options():
    """The chain power and the branch choice of a color-ordered
    PhaseSpaceMapping can be set; with cuts the defaults are AmpliCol's (chain
    power 1, cut-aware branch)."""
    options = ms.ColorOrderedMapping.Options
    mode = ms.PhaseSpaceMapping.color_ordered

    def momenta(color_ordered):
        return run(mode, color_ordered, 5, 1000, cuts=jet_cuts())[0]

    default = momenta(options(ORDER))
    explicit = momenta(options(ORDER, chain_invariant_power=1.0, cut_aware_branch=True))
    assert torch.equal(default, explicit)
    assert not torch.equal(default, momenta(options(ORDER, chain_invariant_power=0.7)))
    # the two solutions of a side of two particles mirror each other and pass
    # the cuts together, so the branch choice shows in a chain of four
    chain = [0, 2, 3, 4, 5, 1]
    assert not torch.equal(
        momenta(options(chain)), momenta(options(chain, cut_aware_branch=False))
    )


def test_color_ordered_options_match_mode():
    """The options are required by the color-ordered mode and refused by the
    others."""
    with pytest.raises(ValueError):
        ms.PhaseSpaceMapping(MASSES, CM_ENERGY, mode=ms.PhaseSpaceMapping.color_ordered)
    with pytest.raises(ValueError):
        ms.PhaseSpaceMapping(
            MASSES,
            CM_ENERGY,
            mode=ms.PhaseSpaceMapping.rambo,
            color_ordered=ms.ColorOrderedMapping.Options(ORDER),
        )
