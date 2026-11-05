import math

import pytest
import torch
from madspace_testing import generator, rand, randint
from pytest import approx

import madspace as ms

torch.set_printoptions(linewidth=1000)


def assert_approx(actual, expected, rel=1e-6, abs=1e-12):
    """pytest.approx-like elementwise comparison of tensors."""
    torch.testing.assert_close(actual, expected, rtol=rel, atol=abs)


def _color_order(n_out, mode):
    """Single-chain color order for color_ordered mode; None otherwise."""
    if mode != ms.PhaseSpaceMapping.color_ordered:
        return None
    return ms.ColorOrderedMapping.Options([0] + [i + 2 for i in range(n_out)] + [1])


def _fwd_inputs(mapping, rng, n):
    """[random] (+ [discrete] when the mode declares discrete choices)."""
    r = rand((n, mapping.random_dim()), generator=rng)
    inputs = [r]
    disc = None
    dd = mapping.discrete_dim()
    if dd:
        disc = randint(0, 2, (n, dd), generator=rng, dtype=torch.int32)
        inputs.append(disc)
    return inputs, r, disc


@pytest.fixture
def rng():
    return generator(43)


@pytest.fixture(
    params=[
        [0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [173.0, 173.0],
        [173.0, 173.0, 0.0],
        [173.0, 173.0, 0.0, 0.0],
        [173.0, 173.0, 0.0, 0.0, 0.0],
        [80.0, 80.0],
        [80.0, 80.0, 80.0],
        [80.0, 80.0, 80.0, 80.0],
        [80.0, 80.0, 80.0, 80.0, 80.0],
    ],
    ids=[
        "2 particles, massless",
        "3 particles, massless",
        "4 particles, massless",
        "5 particles, massless",
        "2 particles, t tbar",
        "3 particles, t tbar",
        "4 particles, t tbar",
        "5 particles, t tbar",
        "2 particles, W",
        "3 particles, W",
        "4 particles, W",
        "5 particles, W",
    ],
)
def masses(request):
    return [0.0, 0.0, *request.param]


@pytest.fixture(
    params=[
        ms.PhaseSpaceMapping.propagator,
        ms.PhaseSpaceMapping.rambo,
        ms.PhaseSpaceMapping.chili,
        ms.PhaseSpaceMapping.color_ordered,
    ],
    ids=["propagator", "rambo", "chili", "color_ordered"],
)
def mode(request):
    return request.param


BATCH_SIZE = 1000
CM_ENERGY = 13000.0


def test_t_channel_masses(masses, rng, mode):
    mapping = ms.PhaseSpaceMapping(
        masses,
        CM_ENERGY,
        mode=mode,
        color_ordered=_color_order(len(masses) - 2, mode),
    )
    inputs, r, _ = _fwd_inputs(mapping, rng, BATCH_SIZE)
    p_ext, x1, x2, det = mapping.map_forward(inputs)

    batch_phys = BATCH_SIZE
    if mode == ms.PhaseSpaceMapping.chili:
        physical_mask = det != 0.0
        p_ext = p_ext[physical_mask]
        batch_phys = p_ext.shape[0]

    m_ext_true = torch.tensor(masses).expand(batch_phys, -1)
    m_ext = torch.sqrt(
        torch.clamp(p_ext[:, :, 0] ** 2 - torch.sum(p_ext[:, :, 1:] ** 2, dim=2), min=0)
    )
    # m = sqrt(E^2 - |p|^2) has a rounding floor that scales with the energy;
    # the deep color_ordered boost chains reach it for a few high-energy events,
    # so only color_ordered gets the energy-scaled tolerance.
    abs_tol = 1e-3
    if mode == ms.PhaseSpaceMapping.color_ordered:
        abs_tol = 1e-3 + 2e-6 * torch.abs(p_ext[:, :, 0])
    assert torch.all(
        torch.abs(m_ext - m_ext_true) <= abs_tol + 1e-3 * torch.abs(m_ext_true)
    )


def test_t_channel_incoming(masses, rng, mode):
    mapping = ms.PhaseSpaceMapping(
        masses,
        CM_ENERGY,
        mode=mode,
        color_ordered=_color_order(len(masses) - 2, mode),
    )
    inputs, r, _ = _fwd_inputs(mapping, rng, BATCH_SIZE)
    p_ext, x1, x2, det = mapping.map_forward(inputs)

    batch_phys = BATCH_SIZE
    if mode == ms.PhaseSpaceMapping.chili:
        physical_mask = det != 0.0
        x1 = x1[physical_mask]
        x2 = x2[physical_mask]
        p_ext = p_ext[physical_mask]
        batch_phys = p_ext.shape[0]

    zeros = torch.zeros(batch_phys)
    p_a = p_ext[:, 0]
    p_b = p_ext[:, 1]
    e_beam = 0.5 * CM_ENERGY

    assert_approx(p_a[:, 0], p_a[:, 3])
    assert_approx(p_b[:, 0], -p_b[:, 3])
    assert_approx(p_a[:, 1], zeros)
    assert_approx(p_a[:, 2], zeros)
    assert_approx(p_b[:, 1], zeros)
    assert_approx(p_b[:, 2], zeros)
    assert torch.all(x1 >= 0) and torch.all(x1 <= 1)
    assert torch.all(x2 >= 0) and torch.all(x2 <= 1)
    assert_approx(p_a[:, 0], e_beam * x1)
    assert_approx(p_b[:, 0], e_beam * x2)


def test_t_channel_momentum_conservation(masses, rng, mode):
    mapping = ms.PhaseSpaceMapping(
        masses,
        CM_ENERGY,
        mode=mode,
        color_ordered=_color_order(len(masses) - 2, mode),
    )
    inputs, r, _ = _fwd_inputs(mapping, rng, BATCH_SIZE)
    p_ext, x1, x2, det = mapping.map_forward(inputs)

    if mode == ms.PhaseSpaceMapping.chili:
        physical_mask = det != 0.0
        p_ext = p_ext[physical_mask]

    p_in = torch.sum(p_ext[:, :2], dim=1)
    p_out = torch.sum(p_ext[:, 2:], dim=1)

    # the transverse components are 0 up to the rounding of momenta of order CM_ENERGY
    assert_approx(p_out, p_in, rel=1e-6, abs=1e-12 * CM_ENERGY)


def test_t_channel_inverse(masses, rng, mode):
    mapping = ms.PhaseSpaceMapping(
        masses,
        CM_ENERGY,
        mode=mode,
        invariant_power=0.3,
        color_ordered=_color_order(len(masses) - 2, mode),
    )
    inputs, r, disc = _fwd_inputs(mapping, rng, BATCH_SIZE)
    p_ext, x1, x2, det = mapping.map_forward(inputs)

    if mode == ms.PhaseSpaceMapping.chili:
        physical_mask = det != 0.0
        p_ext = p_ext[physical_mask]
        x1 = x1[physical_mask]
        x2 = x2[physical_mask]
        r = r[physical_mask]
        if disc is not None:
            disc = disc[physical_mask]
        det = det[physical_mask]

    out = mapping.map_inverse((p_ext, x1, x2))
    det_inv = out[-1]
    r_inv = out[0]
    one_batch = torch.ones_like(det)
    # Continuous randoms round-trip; the discrete choices (color_ordered) are
    # recovered exactly.
    assert_approx(r_inv, r, abs=1e-3, rel=1e-3)
    assert_approx(det * det_inv, one_batch, rel=1e-5)
    if disc is not None:
        assert torch.equal(out[1].long(), disc.long())


@pytest.mark.parametrize(
    "particle_count",
    [2, 3, 4, 5],
    ids=["2 particles", "3 particles", "4 particles", "5 particles"],
)
@pytest.mark.parametrize(
    "energy", [10.0, 100.0, 1000.0], ids=["10GeV", "100GeV", "1TeV"]
)
def test_t_channel_phase_space_volume(particle_count, energy, rng, mode):
    co = _color_order(particle_count, mode)
    if mode == ms.PhaseSpaceMapping.chili:
        mapping = ms.PhaseSpaceMapping(
            [0.0] * (particle_count + 2),
            energy,
            mode=mode,
            leptonic=False,
            color_ordered=co,
        )
    else:
        mapping = ms.PhaseSpaceMapping(
            [0.0] * (particle_count + 2),
            energy,
            mode=mode,
            leptonic=True,
            color_ordered=co,
        )
    sample_count = 100000
    inputs, r, _ = _fwd_inputs(mapping, rng, sample_count)
    *rest, det = mapping.map_forward(inputs)
    # det is the per-branch Jacobian; multiply by 2^discrete_dim (two solutions
    # per 2->3 step) to get the full volume.
    det = det * 2.0 ** mapping.discrete_dim()
    ps_volume = (
        (2 * math.pi) ** (4 - 3 * particle_count)
        * (math.pi / 2.0) ** (particle_count - 1)
        * energy ** (2 * particle_count - 4)
        / (math.gamma(particle_count) * math.gamma(particle_count - 1))
    )
    if mode == ms.PhaseSpaceMapping.chili:
        ps_volume /= (particle_count - 1) ** 2  # integration over x1*x2 in [0,1]
    std_error = det.std(correction=0).item() / math.sqrt(sample_count)
    assert det.mean().item() == approx(ps_volume, abs=3 * std_error, rel=1e-6)


# ---------------------------------------------------------------------------
# Color-order variety (color_ordered mode, always via PhaseSpaceMapping).
#
# The mass/mode-fixture tests above only use the single-chain color order
# (beams cyclically adjacent -> all outgoing on one side). These cases also
# exercise *two-sided* splits (set1 and set2 both non-empty), which build a
# genuinely different t-channel topology, and a range of discrete-peel counts
# (one discrete choice per 2->3 step: each peel after the first, and a
# two-particle side when the other side is not empty).
# ColorOrderedMapping is never constructed directly here -- only through
# PhaseSpaceMapping, which is the only entry point it is meant to have.
# ---------------------------------------------------------------------------
_CO_VARIETY = [
    # (color_order, outgoing_masses, id)
    ([0, 2, 3, 4, 1], [0.0, 0.0, 0.0], "n3 chain {2,3,4}|{}"),
    ([0, 2, 3, 1, 4], [0.0, 0.0, 0.0], "n3 split {2,3}|{4}"),
    ([0, 2, 1, 3, 4], [173.0, 80.0, 0.0], "n3 split {2}|{3,4} massive"),
    ([0, 2, 3, 4, 5, 1], [0.0, 0.0, 0.0, 0.0], "n4 chain {2,3,4,5}|{}"),
    ([0, 2, 3, 1, 4, 5], [0.0, 0.0, 0.0, 0.0], "n4 split {2,3}|{4,5}"),
    ([0, 2, 3, 4, 1, 5], [80.0, 80.0, 80.0, 80.0], "n4 split {2,3,4}|{5} W"),
    ([0, 2, 3, 4, 5, 6, 1], [0.0, 0.0, 0.0, 0.0, 0.0], "n5 chain {2..6}|{}"),
    ([0, 2, 3, 4, 1, 5, 6], [0.0, 0.0, 0.0, 0.0, 0.0], "n5 split {2,3,4}|{5,6}"),
    ([0, 2, 3, 1, 4, 5, 6], [173.0, 0.0, 0.0, 0.0, 0.0], "n5 split {2,3}|{4,5,6} top"),
]
_CO_IDS = [c[2] for c in _CO_VARIETY]


@pytest.mark.parametrize("color_order,out_masses,_label", _CO_VARIETY, ids=_CO_IDS)
def test_color_ordered_color_orders(rng, color_order, out_masses, _label):
    masses = [0.0, 0.0, *out_masses]
    mapping = ms.PhaseSpaceMapping(
        masses,
        CM_ENERGY,
        mode=ms.PhaseSpaceMapping.color_ordered,
        invariant_power=0.3,
        color_ordered=ms.ColorOrderedMapping.Options(color_order),
    )
    # random_dim is color-order invariant; discrete_dim tracks the peel count.
    assert mapping.random_dim() == 3 * len(out_masses) - 2
    inputs, r, disc = _fwd_inputs(mapping, rng, BATCH_SIZE)

    p_ext, x1, x2, det = mapping.map_forward(inputs)

    # (1) momentum conservation
    p_in = torch.sum(p_ext[:, :2], dim=1)
    p_out = torch.sum(p_ext[:, 2:], dim=1)
    # the transverse components are 0 up to the rounding of momenta of order CM_ENERGY
    assert_approx(p_out, p_in, rel=1e-6, abs=1e-12 * CM_ENERGY)

    # (2) on-shell external masses (energy-scaled FP floor for deep boost chains)
    m_ext_true = torch.tensor(masses).expand(BATCH_SIZE, -1)
    m_ext = torch.sqrt(
        torch.clamp(p_ext[:, :, 0] ** 2 - torch.sum(p_ext[:, :, 1:] ** 2, dim=2), min=0)
    )
    abs_tol = 1e-3 + 2e-6 * torch.abs(p_ext[:, :, 0])
    assert torch.all(
        torch.abs(m_ext - m_ext_true) <= abs_tol + 1e-3 * torch.abs(m_ext_true)
    )

    # (3) inverse: continuous randoms round-trip, discrete choices exactly,
    out = mapping.map_inverse((p_ext, x1, x2))
    r_inv, det_inv = out[0], out[-1]
    assert_approx(r_inv, r, abs=2e-3, rel=1e-3)
    rt_err = torch.abs(det * det_inv - 1.0)
    assert torch.quantile(rt_err, 0.99) < 1e-5
    assert torch.isfinite(rt_err).all()
    assert torch.mean((rt_err > 1e-2).double()) < 0.01
    if disc is not None:
        assert torch.equal(out[1].long(), disc.long())
