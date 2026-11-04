"""ColorOrderedMapping steps aligned with AmpliCol's gen23 (issue #292).

- A side of exactly two particles, with the other side not empty, is one 2->3
  step with the opposite beam as the previous emission: both invariants next
  to the beams in the color order are sampled, instead of one and an azimuth.
  The block then samples a momentum transfer with an incoming recoil.
- Composite and rest masses are sampled with 1/sqrt(s) (flat in the mass),
  within the bounds the energies of the two sides set.
- With cut_aware_branch the 2->3 steps keep the azimuthal solution that passes
  the pt cuts, and pick at random only when both do.
- The fiducial volume agrees with RAMBO's, whatever the options.
- The function builder does not fold 0 - x to x.
"""

import math

import pytest
import torch
from madspace_testing import default_context, generator, rand, randint

import madspace as ms

E_BEAM = 500.0
PA = torch.tensor([E_BEAM, 0.0, 0.0, E_BEAM])
PB = torch.tensor([E_BEAM, 0.0, 0.0, -E_BEAM])
CUTS = dict(
    pt_min=[20.0] * 4,
    m_inv_min=[[0.0 if i == j else 10.0 for j in range(4)] for i in range(4)],
    dr_min=[[0.0 if i == j else 0.4 for j in range(4)] for i in range(4)],
)


def lsquare(p):
    return p[..., 0] ** 2 - torch.sum(p[..., 1:] ** 2, dim=-1)


def test_sub_of_zero_literal():
    """sub(0, x) is -x and not x, which matters for bounds built from massless
    literal masses."""
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("x", ms.batch_float)]),
        ms.NamedTypes([("minus_x", ms.batch_float), ("x_minus_0", ms.batch_float)]),
    )
    x = fb.input(0)
    fb.output(0, fb.sub(ms.Value(0.0), x))
    fb.output(1, fb.sub(x, ms.Value(0.0)))
    runtime = ms.FunctionRuntime(fb.function(), default_context())
    values = torch.tensor([1.0, -2.0, 3.5])
    minus_x, x_minus_0 = (torch.from_dlpack(t) for t in runtime.call([values]))
    torch.testing.assert_close(minus_x, -values)
    torch.testing.assert_close(x_minus_0, values)


@pytest.mark.parametrize(
    "order, n_discrete, n_random",
    [
        ([0, 2, 3, 1, 4, 5], 2, 8),
        ([0, 2, 3, 1, 4], 1, 5),
        ([0, 2, 1, 3, 4], 1, 5),
        ([0, 2, 3, 4, 1, 5], 1, 8),
        ([0, 2, 3, 1], 0, 2),
    ],
    ids=["2+2", "2+1", "1+2", "3+1", "2+0"],
)
def test_discrete_dim(order, n_discrete, n_random):
    """One discrete choice per side of two particles next to a non-empty side
    (a 2->3 step instead of a 2->2 peel, with the same random numbers); a
    single chain of two keeps its 2->2 step, as in AmpliCol."""
    mapping = ms.ColorOrderedMapping(ms.ColorOrderedMapping.Options(order))
    assert mapping.discrete_dim() == n_discrete
    assert mapping.random_dim() == n_random


def beam_block(r_s23, r_t1, arcsine, has_cut=False, s_power=0.0):
    """The 2->3 step of a side of two massless particles with the incoming
    recoil pb, for a side momentum p12 recoiling against the other side."""
    mapping = ms.TwoToThreeParticleScattering(
        0.8,
        0.0,
        0.0,
        s_power,
        0.0,
        0.0,
        has_cut=has_cut,
        arcsine_s23=arcsine,
        p12_condition=True,
        incoming_recoil=True,
    )
    n = len(r_s23)
    # p12 = pa + pb - k, k a massless momentum of the other side
    k = torch.tensor([300.0, 300.0 * math.sin(0.9), 0.0, 300.0 * math.cos(0.9)])
    p12 = PA + PB - k
    inputs = [
        torch.zeros(n, dtype=torch.int32),
        r_s23,
        r_t1,
        torch.zeros(n),
        torch.zeros(n),
    ]
    conditions = [PA.repeat(n, 1), p12.repeat(n, 1), PB.repeat(n, 1)]
    if has_cut:
        conditions += [torch.full((n,), 20.0), torch.full((n,), 20.0)]
        conditions += [torch.zeros(n), torch.zeros(n)]
    p1, p2, det = mapping.map_forward(inputs, conditions)
    return mapping, inputs, conditions, p1, p2, det, p12


def test_beam_step_kinematics():
    """p1 + p2 = p12 on shell, t1 = (pa - p1)^2 set by r_t1 alone, and the
    sampled invariant (p2 - pb)^2 runs monotonically with r_s23."""
    r_s23 = torch.linspace(0.01, 0.99, 50)
    r_t1 = torch.full((50,), 0.37)
    *_, p1, p2, det, p12 = beam_block(r_s23, r_t1, arcsine=False)
    assert torch.all(det > 0)
    torch.testing.assert_close(p1 + p2, p12.repeat(50, 1), rtol=0, atol=1e-10)
    torch.testing.assert_close(lsquare(p1), torch.zeros(50), rtol=0, atol=1e-8)
    torch.testing.assert_close(lsquare(p2), torch.zeros(50), rtol=0, atol=1e-8)
    t1 = lsquare(PA - p1)
    torch.testing.assert_close(t1, t1[0].expand(50), rtol=1e-10, atol=0)
    v = -lsquare(p2 - PB)
    assert torch.all(v > 0)
    assert torch.all(torch.diff(v) > 0)
    # flat sampling of v: equally spaced in r
    torch.testing.assert_close(
        torch.diff(v), torch.diff(v)[0].expand(49), rtol=1e-8, atol=0
    )


def test_beam_step_flat_weight_with_arcsine():
    """The measure is flat in the azimuth: with the arcsine map and flat
    sampling the weight does not depend on r_s23, as for an outgoing recoil."""
    edge = 10.0 ** -torch.arange(1.0, 6.0)
    r_s23 = torch.cat([edge, torch.tensor([0.5]), 1.0 - edge])
    r_t1 = torch.full((len(r_s23),), 0.37)
    *_, det, _ = beam_block(r_s23, r_t1, arcsine=True)
    assert torch.all(torch.isfinite(det)) and torch.all(det > 0)
    torch.testing.assert_close(
        det / det[len(edge)], torch.ones_like(det), rtol=1e-6, atol=0
    )


@pytest.mark.parametrize("arcsine", [True, False], ids=["arcsine", "flat"])
@pytest.mark.parametrize("has_cut", [False, True], ids=["no cut", "cut"])
@pytest.mark.parametrize("s_power", [0.0, 0.8])
def test_beam_step_round_trip(arcsine, has_cut, s_power):
    g = generator(11)
    r = rand((20_000, 2), generator=g)
    mapping, inputs, conditions, p1, p2, det, _ = beam_block(
        r[:, 0].contiguous(), r[:, 1].contiguous(), arcsine, has_cut, s_power
    )
    ok = det > 0
    assert ok.float().mean() > 0.5
    *inv, det_inv = mapping.map_inverse([p1, p2], conditions)
    # without the arcsine map the inverse goes through the invariant itself,
    # which leaves ~1e-8 near the edges of its range
    torch.testing.assert_close(
        inv[1][ok], r[ok, 0], rtol=0, atol=1e-8 if arcsine else 1e-7
    )
    torch.testing.assert_close(inv[2][ok], r[ok, 1], rtol=0, atol=1e-8)
    assert torch.max(torch.abs(det[ok] * det_inv[ok] - 1.0)) < 1e-7


def passes(p, cuts):
    """Outgoing momenta p (batch, n_out, 4) inside the cuts."""
    n_out = p.shape[1]
    pt = torch.hypot(p[..., 1], p[..., 2])
    ok = torch.all(pt > cuts["pt_min"][0], dim=1)
    eta = torch.asinh(p[..., 3] / torch.clamp(pt, min=1e-300))
    phi = torch.atan2(p[..., 2], p[..., 1])
    for i in range(n_out):
        for j in range(i + 1, n_out):
            q = p[:, i] + p[:, j]
            dphi = phi[:, i] - phi[:, j]
            dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
            ok &= lsquare(q) > cuts["m_inv_min"][i][j] ** 2
            ok &= torch.hypot(eta[:, i] - eta[:, j], dphi) > cuts["dr_min"][i][j]
    return ok


def sample(order, seed, n, cuts=None, **options):
    """Weights of ColorOrderedMapping for the massless phase space at 1 TeV,
    zero outside the cuts if given."""
    cuts = cuts or {}
    cm_energy = 1000.0
    n_out = len(order) - 2
    mapping = ms.ColorOrderedMapping(
        ms.ColorOrderedMapping.Options(order, **options), **cuts
    )
    g = generator(seed)
    r = rand((n, mapping.random_dim()), generator=g)
    d = randint(0, 2, (n, mapping.discrete_dim()), generator=g, dtype=torch.int32)
    inputs = [r[:, i].contiguous() for i in range(r.shape[1])]
    inputs += [d[:, j].contiguous() for j in range(d.shape[1])]
    conditions = [torch.full((n,), cm_energy)] + [torch.zeros(n)] * n_out
    *momenta, det = mapping.map_forward(inputs, conditions)
    p = torch.stack(momenta, dim=1)[:, 2:]
    det = det * 2.0 ** mapping.discrete_dim()
    ok = torch.isfinite(p).all(dim=2).all(dim=1) & torch.isfinite(det)
    if cuts:
        ok &= passes(p, cuts)
    return torch.where(ok, det, 0.0)


@pytest.mark.parametrize(
    "order",
    [[0, 2, 3, 1, 4, 5], [0, 2, 3, 1, 4], [0, 2, 1, 3, 4]],
    ids=["2+2", "2+1", "1+2"],
)
def test_massless_volume(order):
    """The analytic massless volume, (pi/2)^(n-1) s^(n-2) / ((n-1)! (n-2)!)."""
    n = len(order) - 2
    volume = (math.pi / 2) ** (n - 1) * 1000.0 ** (2 * n - 4)
    volume /= math.factorial(n - 1) * math.factorial(n - 2)
    w = torch.cat([sample(order, seed, 1_000_000) for seed in range(400, 404)])
    err = w.std(correction=0).item() / math.sqrt(len(w))
    assert abs(w.mean().item() - volume) < 4.0 * err
    assert err / volume < 0.01


def rambo_acceptance(n_out, n):
    """Fraction of the massless phase space at 1 TeV inside the cuts, from
    RAMBO (a constant weight)."""
    mapping = ms.PhaseSpaceMapping(
        [0.0] * (n_out + 2), 1000.0, leptonic=True, mode=ms.PhaseSpaceMapping.rambo
    )
    g = generator(600)
    p = mapping.map_forward([rand((n, mapping.random_dim()), generator=g)])[0]
    frac = passes(p[:, 2:], CUTS).double().mean().item()
    return frac, math.sqrt(frac * (1.0 - frac) / n)


@pytest.mark.parametrize(
    "order, options",
    [
        ([0, 2, 3, 1, 4, 5], {}),
        ([0, 2, 3, 4, 5, 1], {}),
        ([0, 2, 3, 4, 1, 5], {}),
        ([0, 2, 3, 1, 4, 5], dict(cut_aware_branch=False)),
        ([0, 2, 3, 4, 5, 1], dict(cut_aware_branch=False, chain_invariant_power=0.7)),
    ],
    ids=["2+2", "chain", "3+1", "2+2 no branch", "chain no branch, 0.7"],
)
def test_cut_volume(order, options):
    """With cuts, the fiducial volume is the massless volume times the
    fraction RAMBO finds inside the cuts, whatever the options."""
    n_out = len(order) - 2
    volume = (math.pi / 2) ** (n_out - 1) * 1000.0 ** (2 * n_out - 4)
    volume /= math.factorial(n_out - 1) * math.factorial(n_out - 2)
    frac, frac_err = rambo_acceptance(n_out, 2_000_000)
    n = 1_000_000
    w = torch.cat([sample(order, s, n, CUTS, **options) for s in range(500, 504)])
    ratio = w.mean().item() / volume
    err = math.hypot(w.std(correction=0).item() / math.sqrt(len(w)) / volume, frac_err)
    assert abs(ratio - frac) < 4.0 * err


@pytest.mark.parametrize(
    "order", [[0, 2, 3, 1, 4, 5], [0, 2, 3, 4, 5, 1]], ids=["2+2", "chain"]
)
def test_round_trip_with_cuts(order):
    """Forward and inverse agree for the 2->3 steps and the branch choice;
    where only one solution passes, the inverse returns the one taken."""
    n = 200_000
    mapping = ms.ColorOrderedMapping(ms.ColorOrderedMapping.Options(order), **CUTS)
    g = generator(21)
    r = rand((n, mapping.random_dim()), generator=g)
    d = randint(0, 2, (n, mapping.discrete_dim()), generator=g, dtype=torch.int32)
    inputs = [r[:, i].contiguous() for i in range(r.shape[1])]
    inputs += [d[:, j].contiguous() for j in range(d.shape[1])]
    conditions = [torch.full((n,), 1000.0)] + [torch.zeros(n)] * 4
    *momenta, det = mapping.map_forward(inputs, conditions)
    out = mapping.map_inverse(momenta, conditions)
    r_inv = torch.stack(out[: mapping.random_dim()], dim=1)
    ok = det > 0
    rt = torch.abs(det * out[-1] - 1.0)[ok]
    assert torch.mean((rt > 1e-2).double()) < 1e-4
    assert torch.quantile(rt, 0.999) < 1e-6
    assert (
        torch.mean((torch.max(torch.abs(r_inv - r), dim=1).values[ok] > 1e-6).double())
        < 1e-4
    )
