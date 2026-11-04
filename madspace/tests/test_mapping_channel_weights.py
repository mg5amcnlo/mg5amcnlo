"""MappingChannelWeights: channel weights from the densities of the channel mappings.

- The weights are the masked densities of the inverse mappings, normalised per
  event, and the sampled diagram follows them.
- Partner mappings enter the normalisation only: the weights of the channels
  are their share of the contribution.
- A color-ordered chain with a resonance among its children integrates to the
  same as the plain chain, and can be inverted at the events of the plain one.
"""

import pytest
import torch
from madspace_testing import default_context, generator, rand, randint

import madspace as ms

E_CM = 13000.0
M_Z, W_Z = 91.1876, 2.4952
PIDS = [2, -2, 21, 11, -11]


def cuts():
    O = ms.Observable
    return ms.Cuts(
        [
            ms.CutItem(O(PIDS, O.obs_pt, [O.jet_pids]), min=20.0),
            ms.CutItem(O(PIDS, O.obs_pt, [[11, -11]]), min=10.0),
        ]
    )


def chain(order, resonant):
    """q q~ > g e- e+ as a chain, the lepton pair from a Z if resonant."""
    if resonant:
        propagators = [ms.Propagator(), ms.Propagator(mass=M_Z, width=W_Z, pdg_id=23)]
        vertices = [["i0", "p0", "o0"], ["i1", "p0", "p1"], ["p1", "o1", "o2"]]
    else:
        propagators = [ms.Propagator(), ms.Propagator()]
        vertices = [["i0", "p0", "o0"], ["p0", "p1", "o1"], ["i1", "p1", "o2"]]
    diagram = ms.Diagram([0.0, 0.0], [0.0] * 3, propagators, vertices)
    return ms.PhaseSpaceMapping(
        ms.Topology(diagram),
        E_CM,
        t_channel_mode=ms.PhaseSpaceMapping.color_ordered,
        cuts=cuts(),
        permutations=[list(range(5))],
        color_ordered=ms.ColorOrderedMapping.Options(order),
    )


RESONANT = chain([0, 2, 3, 1], True)
PLAIN = chain([0, 2, 3, 4, 1], False)


def sample(mapping, n, seed):
    g = generator(seed)
    inputs = [rand((n, mapping.random_dim()), generator=g)]
    if mapping.discrete_dim():
        inputs.append(
            randint(0, 2, (n, mapping.discrete_dim()), generator=g, dtype=torch.int32)
        )
    p, x1, x2, det = mapping.map_forward(inputs)
    return p, x1, x2, det * 2 ** mapping.discrete_dim()


def density(mapping, p, x1, x2):
    return mapping.map_inverse([p, x1, x2])[-1] / 2 ** mapping.discrete_dim()


def test_resonant_chain_integrates_like_the_plain_one():
    def bw(p):
        q = p[:, 3] + p[:, 4]
        m2 = q[:, 0] ** 2 - torch.sum(q[:, 1:] ** 2, dim=1)
        return 1e6 / ((m2 - M_Z**2) ** 2 + (M_Z * W_Z) ** 2)

    n = 400_000
    results = []
    for mapping in (RESONANT, PLAIN):
        p, _, _, w = sample(mapping, n, 1)
        values = torch.where(w > 0, w * torch.nan_to_num(bw(p)), 0.0)
        results.append((values.mean(), values.std() / n**0.5))
    (res, res_err), (plain, plain_err) = results
    assert res_err < 0.02 * res
    assert abs(res - plain) < 3 * (res_err**2 + plain_err**2) ** 0.5


def test_resonant_chain_inverts_the_plain_events():
    p, x1, x2, w = sample(PLAIN, 100_000, 2)
    ok = w > 0
    out = RESONANT.map_inverse([p[ok], x1[ok], x2[ok]])
    r = out[0]
    assert torch.all((r >= 0.0) & (r <= 1.0))
    p_back, *_ = RESONANT.map_forward([r])
    assert torch.max(torch.abs(p_back - p[ok])) < 1e-8 * E_CM


@pytest.mark.parametrize("flavor", [0, 1])
def test_weights_are_the_normalised_masked_densities(flavor):
    masks = [[1.0, 1.0], [0.0, 1.0]]
    # diagram 0 on channel 0, diagrams 1 and 2 on channel 1; flavor 0 has
    # diagrams 0 and 1, flavor 1 has diagram 2
    weights = ms.MappingChannelWeights(
        [RESONANT, PLAIN], masks, [0, 1, 1], [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    runtime = ms.FunctionRuntime(weights.function(), default_context())
    p, x1, x2, w = sample(PLAIN, 40_000, 3)
    ok = w > 0
    p, x1, x2 = p[ok], x1[ok], x2[ok]
    n = p.shape[0]
    flavors = torch.full((n,), flavor, dtype=torch.int32)
    alpha, diagram = (torch.from_dlpack(t) for t in runtime.call([p, x1, x2, flavors]))
    expected = torch.stack(
        [density(RESONANT, p, x1, x2), density(PLAIN, p, x1, x2)], dim=1
    ) * torch.tensor(masks[flavor])
    expected /= expected.sum(dim=1, keepdim=True)
    torch.testing.assert_close(alpha, expected, rtol=1e-12, atol=1e-14)
    counts = torch.bincount(diagram.long(), minlength=3).double()
    if flavor == 1:
        assert counts.tolist() == [0.0, 0.0, n]
    else:
        assert counts[2] == 0
        mean = alpha[:, 0].sum()
        assert abs(counts[0] - mean) < 5 * mean**0.5


def test_channels_without_the_flavor_get_no_weight():
    weights = ms.MappingChannelWeights([PLAIN, PLAIN], [[0.0, 1.0]])
    runtime = ms.FunctionRuntime(weights.function(), default_context())
    p, x1, x2, w = sample(PLAIN, 1000, 4)
    ok = w > 0
    n = int(ok.sum())
    alpha = torch.from_dlpack(
        runtime.call(
            [
                p[ok],
                x1[ok],
                x2[ok],
                torch.zeros(n, dtype=torch.int32),
            ]
        )[0]
    )
    torch.testing.assert_close(alpha[:, 1], torch.ones(n, dtype=alpha.dtype))
    torch.testing.assert_close(alpha[:, 0], torch.zeros(n, dtype=alpha.dtype))


def test_uniform_weights_without_valid_densities():
    """If no inverse gives a valid density, the weights are uniform over the
    channels of the flavor: a function of the event alone."""
    weights = ms.MappingChannelWeights([RESONANT, PLAIN, PLAIN], [[1.0, 0.0, 1.0]])
    runtime = ms.FunctionRuntime(weights.function(), default_context())
    p, x1, x2, w = sample(PLAIN, 1000, 5)
    ok = w > 0
    n = int(ok.sum())
    flavors = torch.zeros(n, dtype=torch.int32)
    p, x1, x2 = p[ok], x1[ok], x2[ok]
    alpha = torch.from_dlpack(runtime.call([p, x1, x2, flavors])[0])
    assert not torch.allclose(alpha[:, 0], alpha[:, 2])
    # inverse mappings that fail numerically
    p = torch.full_like(p, float("nan"))
    alpha = torch.from_dlpack(runtime.call([p, x1, x2, flavors])[0])
    expected = torch.tensor([0.5, 0.0, 0.5], dtype=alpha.dtype).expand(n, 3)
    torch.testing.assert_close(alpha, expected)


def test_partners_only_enter_the_normalisation():
    """A single channel with a partner: its weight is its share of the
    contribution, and a diagram can sit on the partner."""
    weights = ms.MappingChannelWeights(
        [RESONANT],
        [[1.0, 1.0], [1.0, 0.0]],
        [0, 1],
        [[1.0, 1.0], [1.0, 0.0]],
        partner_mappings=[PLAIN],
    )
    assert weights.channel_count() == 1
    assert weights.has_partners()
    runtime = ms.FunctionRuntime(weights.function(), default_context())
    p, x1, x2, w = sample(RESONANT, 20_000, 6)
    ok = w > 0
    p, x1, x2 = p[ok], x1[ok], x2[ok]
    n = p.shape[0]
    resonant, plain = density(RESONANT, p, x1, x2), density(PLAIN, p, x1, x2)
    for flavor, expected in ((0, resonant / (resonant + plain)), (1, torch.ones(n))):
        flavors = torch.full((n,), flavor, dtype=torch.int32)
        alpha, diagram = (
            torch.from_dlpack(t) for t in runtime.call([p, x1, x2, flavors])
        )
        assert alpha.shape == (n, 1)
        torch.testing.assert_close(
            alpha[:, 0], expected.to(alpha.dtype), rtol=1e-12, atol=1e-14
        )
        if flavor == 1:
            assert torch.all(diagram == 0)
        else:
            on_partner = (diagram == 1).double().sum()
            mean = (1 - alpha[:, 0]).sum()
            assert abs(on_partner - mean) < 5 * mean**0.5


def test_partners_need_a_single_channel():
    with pytest.raises(ValueError, match="single channel"):
        ms.MappingChannelWeights(
            [RESONANT, PLAIN], [[1.0, 1.0, 1.0]], partner_mappings=[PLAIN]
        )
