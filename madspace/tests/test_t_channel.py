import math

import numpy as np
import pytest
from pytest import approx

import madspace as ms

np.set_printoptions(linewidth=1000)


def _color_order(n_out, mode):
    """Single-chain color order for color_ordered mode; None otherwise."""
    if mode != ms.PhaseSpaceMapping.color_ordered:
        return None
    return [0] + [i + 2 for i in range(n_out)] + [1]


def _fwd_inputs(mapping, rng, n):
    """[random] (+ [discrete] when the mode declares discrete choices)."""
    r = rng.random((n, mapping.random_dim()))
    inputs = [r]
    disc = None
    dd = mapping.discrete_dim()
    if dd:
        disc = rng.integers(0, 2, size=(n, dd)).astype(np.int32)
        inputs.append(disc)
    return inputs, r, disc


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


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
        color_order=_color_order(len(masses) - 2, mode),
    )
    inputs, r, _ = _fwd_inputs(mapping, rng, BATCH_SIZE)
    p_ext, x1, x2, det = mapping.map_forward(inputs)

    batch_phys = BATCH_SIZE
    if mode == ms.PhaseSpaceMapping.chili:
        physical_mask = det != 0.0
        p_ext = p_ext[physical_mask]
        batch_phys = p_ext.shape[0]

    m_ext_true = np.full((batch_phys, len(masses)), masses)
    m_ext = np.sqrt(
        np.maximum(0, p_ext[:, :, 0] ** 2 - np.sum(p_ext[:, :, 1:] ** 2, axis=2))
    )
    # The massless on-shell reconstruction m = sqrt(E^2 - |p|^2) has an FP floor
    # that scales with the particle energy (~2e-6 * E); the deep color_ordered
    # boost chains reach it for a few high-energy boundary events. Shallower
    # modes stay well under the flat 1e-3, so only color_ordered needs the
    # energy-scaled term (a genuine kinematic error would exceed it ~1000x).
    abs_tol = 1e-3
    if mode == ms.PhaseSpaceMapping.color_ordered:
        abs_tol = 1e-3 + 2e-6 * np.abs(p_ext[:, :, 0])
    assert np.all(np.abs(m_ext - m_ext_true) <= abs_tol + 1e-3 * np.abs(m_ext_true))


def test_t_channel_incoming(masses, rng, mode):
    mapping = ms.PhaseSpaceMapping(
        masses,
        CM_ENERGY,
        mode=mode,
        color_order=_color_order(len(masses) - 2, mode),
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

    zeros = np.zeros(batch_phys)
    p_a = p_ext[:, 0]
    p_b = p_ext[:, 1]
    e_beam = 0.5 * CM_ENERGY

    assert p_a[:, 0] == approx(p_a[:, 3]) and p_b[:, 0] == approx(-p_b[:, 3])
    assert p_a[:, 1] == approx(zeros) and p_a[:, 2] == approx(zeros)
    assert p_b[:, 1] == approx(zeros) and p_b[:, 2] == approx(zeros)
    assert np.all(x1 >= 0) and np.all(x1 <= 1)
    assert np.all(x2 >= 0) and np.all(x2 <= 1)
    assert p_a[:, 0] == approx(e_beam * x1)
    assert p_b[:, 0] == approx(e_beam * x2)


def test_t_channel_momentum_conservation(masses, rng, mode):
    mapping = ms.PhaseSpaceMapping(
        masses,
        CM_ENERGY,
        mode=mode,
        color_order=_color_order(len(masses) - 2, mode),
    )
    inputs, r, _ = _fwd_inputs(mapping, rng, BATCH_SIZE)
    p_ext, x1, x2, det = mapping.map_forward(inputs)

    if mode == ms.PhaseSpaceMapping.chili:
        physical_mask = det != 0.0
        p_ext = p_ext[physical_mask]

    p_in = np.sum(p_ext[:, :2], axis=1)
    p_out = np.sum(p_ext[:, 2:], axis=1)

    assert p_out == approx(p_in, rel=1e-6, abs=1e-9)


def test_t_channel_inverse(masses, rng, mode):
    mapping = ms.PhaseSpaceMapping(
        masses,
        CM_ENERGY,
        mode=mode,
        invariant_power=0.3,
        color_order=_color_order(len(masses) - 2, mode),
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
    one_batch = np.ones_like(det)
    # Continuous randoms round-trip; the discrete int channel (color_ordered)
    # is checked for exact bin/solution recovery separately.
    assert r_inv == approx(r, abs=1e-3, rel=1e-3)
    assert det * det_inv == approx(one_batch, rel=1e-5)
    if disc is not None:
        disc_inv = np.asarray(out[1]).astype(np.int64)
        assert np.array_equal(disc_inv, disc.astype(np.int64))


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
            color_order=co,
        )
    else:
        mapping = ms.PhaseSpaceMapping(
            [0.0] * (particle_count + 2),
            energy,
            mode=mode,
            leptonic=True,
            color_order=co,
        )
    sample_count = 100000
    inputs, r, _ = _fwd_inputs(mapping, rng, sample_count)
    *rest, det = mapping.map_forward(inputs)
    # det is the per-branch Jacobian; the solution multiplicity (2 per 2->3 peel)
    # is owned externally now, so apply 2^discrete_dim to recover the full volume.
    det = det * 2.0 ** mapping.discrete_dim()
    ps_volume = (
        (2 * math.pi) ** (4 - 3 * particle_count)
        * (math.pi / 2.0) ** (particle_count - 1)
        * energy ** (2 * particle_count - 4)
        / (math.gamma(particle_count) * math.gamma(particle_count - 1))
    )
    if mode == ms.PhaseSpaceMapping.chili:
        ps_volume /= (particle_count - 1) ** 2  # integration over x1*x2 in [0,1]
    std_error = np.std(det) / np.sqrt(sample_count)
    assert np.mean(det) == approx(ps_volume, abs=3 * std_error, rel=1e-6)


# ---------------------------------------------------------------------------
# Color-order variety (color_ordered mode, always via PhaseSpaceMapping).
#
# The mass/mode-fixture tests above only use the single-chain color order
# (beams cyclically adjacent -> all outgoing on one side). These cases also
# exercise *two-sided* splits (set1 and set2 both non-empty), which build a
# genuinely different t-channel topology, and a range of discrete-peel counts
# (discrete_dim grows by one for every set with > 2 outgoing partons).
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
        color_order=color_order,
    )
    # random_dim is color-order invariant; discrete_dim tracks the peel count.
    assert mapping.random_dim() == 3 * len(out_masses) - 2
    inputs, r, disc = _fwd_inputs(mapping, rng, BATCH_SIZE)

    p_ext, x1, x2, det = mapping.map_forward(inputs)

    # (1) momentum conservation
    p_in = np.sum(p_ext[:, :2], axis=1)
    p_out = np.sum(p_ext[:, 2:], axis=1)
    assert p_out == approx(p_in, rel=1e-6, abs=1e-9)

    # (2) on-shell external masses (energy-scaled FP floor for deep boost chains)
    m_ext_true = np.full((BATCH_SIZE, len(masses)), masses)
    m_ext = np.sqrt(
        np.maximum(0, p_ext[:, :, 0] ** 2 - np.sum(p_ext[:, :, 1:] ** 2, axis=2))
    )
    abs_tol = 1e-3 + 2e-6 * np.abs(p_ext[:, :, 0])
    assert np.all(np.abs(m_ext - m_ext_true) <= abs_tol + 1e-3 * np.abs(m_ext_true))

    # (3) inverse: continuous randoms round-trip, discrete choices exactly,
    out = mapping.map_inverse((p_ext, x1, x2))
    r_inv, det_inv = out[0], out[-1]
    assert r_inv == approx(r, abs=1e-3, rel=1e-3)
    rt_err = np.abs(det * det_inv - 1.0)
    assert np.quantile(rt_err, 0.99) < 1e-5
    assert np.isfinite(rt_err).all()
    assert np.mean(rt_err > 1e-2) < 0.01
    if disc is not None:
        disc_inv = np.asarray(out[1]).astype(np.int64)
        assert np.array_equal(disc_inv, disc.astype(np.int64))
