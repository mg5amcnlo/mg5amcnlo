"""2 -> 1 phase space and asymmetric beams (PhaseSpaceMapping beam_rapidity)."""

import math

import numpy as np
import pytest
from madspace_testing import to_numpy
from pytest import approx

import madspace as ms

N = 2000


@pytest.fixture
def rng():
    return np.random.default_rng(4321)


def boost_z(p, y):
    q = p.copy()
    q[..., 0] = p[..., 0] * math.cosh(y) + p[..., 3] * math.sinh(y)
    q[..., 3] = p[..., 3] * math.cosh(y) + p[..., 0] * math.sinh(y)
    return q


def two_to_one_topology(mass):
    return ms.Topology(ms.Diagram([0.0, 0.0], [mass], [], [["i0", "i1", "o0"]]))


def test_two_to_one(rng):
    mass, sqrt_s = 91.188, 13000.0
    mapping = ms.PhaseSpaceMapping(two_to_one_topology(mass), sqrt_s)
    assert mapping.random_dim() == 1
    r = rng.random((N, 1))
    p, x1, x2, det = map(to_numpy, mapping.map_forward([r], []))

    tau = mass**2 / sqrt_s**2
    assert x1 * x2 == approx(tau)
    assert p[:, 0] + p[:, 1] == approx(p[:, 2], abs=1e-9)
    assert p[:, 0, 0] == approx(x1 * sqrt_s / 2)
    assert p[:, 1, 0] == approx(x2 * sqrt_s / 2)
    mass2 = p[:, 2, 0] ** 2 - np.sum(p[:, 2, 1:] ** 2, axis=1)
    assert mass2 == approx(mass**2)
    # dPhi_1 = 2 pi delta(s_hat - m^2), dx1 dx2 = |ln tau| / s ds_hat dr
    assert det == approx(2 * math.pi * abs(math.log(tau)) / sqrt_s**2)

    r_inv, det_inv = mapping.map_inverse([p, x1, x2], [])
    assert to_numpy(r_inv) == approx(r)
    assert to_numpy(det_inv) * det == approx(1.0)

    flat = ms.PhaseSpaceMapping([0.0, 0.0, mass], sqrt_s)
    assert to_numpy(flat.map_forward([r], [])[0]) == approx(p)


def test_two_to_one_needs_pdf():
    with pytest.raises(ValueError):
        ms.PhaseSpaceMapping(two_to_one_topology(125.0), 13000.0, leptonic=True)
    with pytest.raises(ValueError):
        ms.PhaseSpaceMapping(two_to_one_topology(0.0), 13000.0)


@pytest.mark.parametrize("leptonic", [False, True], ids=["hadronic", "leptonic"])
@pytest.mark.parametrize("mode", ["rambo", "propagator"])
def test_beam_rapidity(rng, mode, leptonic):
    e1, e2 = 7000.0, 4000.0
    e_cm = 2 * math.sqrt(e1 * e2)
    y0 = 0.5 * math.log(e1 / e2)
    masses = [0.0, 0.0, 0.0, 0.0, 173.0]
    kwargs = dict(leptonic=leptonic, mode=getattr(ms.PhaseSpaceMapping, mode))
    sym = ms.PhaseSpaceMapping(masses, e_cm, **kwargs)
    lab = ms.PhaseSpaceMapping(masses, e_cm, beam_rapidity=y0, **kwargs)
    r = rng.random((N, sym.random_dim()))
    p_sym, x1_sym, x2_sym, det_sym = map(to_numpy, sym.map_forward([r], []))
    p_lab, x1, x2, det = map(to_numpy, lab.map_forward([r], []))
    x1, x2 = np.broadcast_to(x1, (N,)).copy(), np.broadcast_to(x2, (N,)).copy()
    ok = det_sym > 0

    # same point, same weight: the lab frame is a longitudinal boost away
    assert det == approx(det_sym)
    assert p_lab[ok] == approx(boost_z(p_sym[ok], y0), abs=1e-8)
    # the incoming partons carry the momentum fractions of their own beam
    assert p_lab[ok, 0, 0] == approx(x1[ok] * e1)
    assert p_lab[ok, 1, 0] == approx(x2[ok] * e2)

    r_inv, det_inv = lab.map_inverse([p_lab, x1, x2], [])
    assert to_numpy(r_inv)[ok] == approx(r[ok], abs=1e-6)
    assert to_numpy(det_inv)[ok] * det[ok] == approx(1.0, rel=1e-5)


def test_beam_rapidity_cut_in_lab_frame(rng):
    """An eta cut is applied to the boosted (lab) momenta."""
    e1, e2 = 7000.0, 1000.0
    e_cm = 2 * math.sqrt(e1 * e2)
    y0 = 0.5 * math.log(e1 / e2)
    pids = [2, -2, 11, -11]
    cuts = ms.Cuts(
        [ms.CutItem(observable=ms.Observable(pids, "eta", [[11, -11]]), min=0.0)]
    )
    mapping = ms.PhaseSpaceMapping(
        [0.0] * 4, e_cm, cuts=cuts, beam_rapidity=y0, mode=ms.PhaseSpaceMapping.rambo
    )
    free = ms.PhaseSpaceMapping(
        [0.0] * 4, e_cm, beam_rapidity=y0, mode=ms.PhaseSpaceMapping.rambo
    )
    r = rng.random((N, mapping.random_dim()))
    p, _, _, det = map(to_numpy, mapping.map_forward([r], []))
    p_free, _, _, det_free = map(to_numpy, free.map_forward([r], []))
    eta = np.arctanh(p_free[:, 2:, 3] / np.linalg.norm(p_free[:, 2:, 1:], axis=2))
    passed = (eta > 0).all(axis=1) & (det_free > 0)
    assert passed.any() and not passed.all()
    assert ((det > 0) == passed).all()


@pytest.mark.parametrize("pdf_asymmetric_only", [False, True], ids=["energies", "pdfs"])
def test_mirror_beams(rng, pdf_asymmetric_only):
    """mirror_index 1: leg 1 from beam 2, i.e. rotated in the beams' frame"""
    e1, e2 = (6500.0, 6500.0) if pdf_asymmetric_only else (7000.0, 4000.0)
    e_cm = 2 * math.sqrt(e1 * e2)
    y0 = 0.5 * math.log(e1 / e2)
    masses = [0.0, 0.0, 0.0, 0.0]
    plain = ms.PhaseSpaceMapping(masses, e_cm, beam_rapidity=y0)
    mirror = ms.PhaseSpaceMapping(masses, e_cm, beam_rapidity=y0, mirror_beams=True)
    assert mirror.mirror_beams() and not plain.mirror_beams()
    r = rng.random((N, plain.random_dim()))
    p, x1, x2, det = map(to_numpy, plain.map_forward([r], []))
    zeros, ones = np.zeros(N, dtype=np.int32), np.ones(N, dtype=np.int32)
    p0, x1_0, x2_0, det0 = map(to_numpy, mirror.map_forward([r], [zeros]))
    p1, x1_1, x2_1, det1 = map(to_numpy, mirror.map_forward([r], [ones]))
    ok = det > 0
    assert p0[ok] == approx(p[ok], abs=1e-8)
    assert det1 == approx(det)
    assert x1_1 == approx(x1)
    assert x2_1 == approx(x2)
    # leg 1 moves along -z with its fraction of beam 2, leg 2 along +z
    assert (p1[ok, 0, 3] < 0).all()
    assert p1[ok, 0, 0] == approx(x1[ok] * e2)
    assert (p1[ok, 1, 3] > 0).all()
    assert p1[ok, 1, 0] == approx(x2[ok] * e1)
    # which is the beams'-frame rotation of the unmirrored event
    rot = boost_z(p[ok], -y0)
    rot[..., 2:] *= -1
    assert p1[ok] == approx(boost_z(rot, y0), abs=1e-8)
    r_inv, det_inv = mirror.map_inverse([p1, x1_1, x2_1], [ones])
    assert to_numpy(r_inv)[ok] == approx(r[ok], abs=1e-6)


# --- the mirror and the cuts -------------------------------------------------
#
# The initial-state mirror is the rotation by pi about x, (E, px, py, pz) ->
# (E, px, -py, -pz), which moves each leg onto the other beam. With mirror_beams
# the mapping draws the orientation first, so the cuts and the written event see
# the same momenta. Integrand requires it for mirrored flavors.


def mirror(p):
    """Rotate by pi about the x axis, as kernel_mirror_momenta does."""
    q = p.copy()
    q[..., 2:] *= -1
    return q


def eta(p):
    return np.arctanh(p[..., 3] / np.linalg.norm(p[..., 1:], axis=-1))


PIDS = [2, -2, 11, -11]


def lepton_cut(observable, **bounds):
    return ms.Cuts(
        [
            ms.CutItem(
                observable=ms.Observable(
                    PIDS, observable, [[11, -11]], name=f"lepton-{observable}"
                ),
                **bounds,
            )
        ]
    )


@pytest.mark.parametrize(
    "observable, bounds, separates",
    [("eta", dict(min=0.0), True), ("eta_abs", dict(max=1.0), False)],
    ids=["eta", "eta_abs"],
)
def test_mirror_beams_cuts_the_written_orientation(rng, observable, bounds, separates):
    """With mirror_beams the cuts act on the momenta the mapping returns.

    A signed eta cut tells the two orientations apart, |eta| does not.
    """
    e_cm, mode = 13000.0, ms.PhaseSpaceMapping.rambo
    cut = ms.PhaseSpaceMapping(
        [0.0] * 4,
        e_cm,
        cuts=lepton_cut(observable, **bounds),
        mirror_beams=True,
        mode=mode,
    )
    free = ms.PhaseSpaceMapping([0.0] * 4, e_cm, mirror_beams=True, mode=mode)
    r = rng.random((N, cut.random_dim()))
    accepted, momenta, physical = {}, {}, {}
    for index in (0, 1):
        condition = np.full(N, index, dtype=np.int32)
        p_cut, _, _, det_cut = map(to_numpy, cut.map_forward([r], [condition]))
        accepted[index] = det_cut > 0
        p, _, _, det = map(to_numpy, free.map_forward([r], [condition]))
        momenta[index], physical[index] = p, det > 0
        # whatever the cut mapping accepts passes the cut as written
        obs = eta(p_cut[:, 2:])
        if observable == "eta_abs":
            passed = (np.abs(obs) < bounds["max"] + 1e-9).all(axis=1)
        else:
            passed = (obs > bounds["min"] - 1e-9).all(axis=1)
        assert passed[accepted[index]].all()
        assert accepted[index].any()

    ok = physical[0]
    assert ok.any()
    # orientation 1 is the pi rotation of orientation 0
    assert momenta[1][ok] == approx(mirror(momenta[0][ok]), abs=1e-7)
    if separates:
        # the cut is on the written momenta, so it tells the orientations apart
        assert (accepted[0] != accepted[1]).any()


@pytest.mark.parametrize("mirror_beams", [False, True], ids=["plain", "mirror"])
def test_cut_aware_sampling_loses_nothing_with_asymmetric_beams(rng, mirror_beams):
    """The rapidity windows are built from the cuts, which hold in the lab frame;
    the integral over the cut mapping must equal the cut applied afterwards to
    an uncut mapping."""
    n, e_cm, y0 = 400_000, 13000.0, 0.7
    mode = ms.PhaseSpaceMapping.rambo
    cuts = lepton_cut("eta_abs", max=1.5)
    kwargs = dict(mode=mode, beam_rapidity=y0, mirror_beams=mirror_beams)
    cut = ms.PhaseSpaceMapping([0.0] * 4, e_cm, cuts=cuts, **kwargs)
    free = ms.PhaseSpaceMapping([0.0] * 4, e_cm, **kwargs)

    def conditions():
        if not mirror_beams:
            return []
        return [rng.integers(0, 2, n).astype(np.int32)]

    det_cut = to_numpy(
        cut.map_forward([rng.random((n, cut.random_dim()))], conditions())[3]
    )
    p, _, _, det = map(
        to_numpy,
        free.map_forward([rng.random((n, free.random_dim()))], conditions()),
    )
    passed = (np.abs(eta(p[:, 2:])) < 1.5).all(axis=1)
    assert det_cut.mean() == approx((det * passed).mean(), rel=0.03)


# --- the mirror and the matrix element ---------------------------------------


@pytest.fixture(scope="module")
def running_coupling(tmp_path_factory):
    """A RunningCoupling over a hand-written alpha_s grid: the Integrand needs
    one to build, and nothing here depends on the values."""
    info = tmp_path_factory.mktemp("alphas") / "grid.info"
    info.write_text(
        "AlphaS_Qs: [1.0, 10.0, 100.0, 1000.0, 10000.0]\n"
        "AlphaS_Vals: [0.4, 0.2, 0.12, 0.09, 0.08]\n"
    )
    return ms.RunningCoupling(ms.AlphaSGrid(str(info)))


PID_OPTIONS = [[2, -2], [-2, 2]]


def build_integrand(
    running_coupling, cuts=None, mirror_beams=True, flavor_mirror=(True, True)
):
    """A 2 -> 2 integrand over a beam-swapped pair of flavors. The matrix
    element is never called; only the compute graph is built."""
    e_cm = 13000.0
    me = ms.MatrixElement(
        matrix_element_index=0,
        particle_count=4,
        inputs=[
            ms.MatrixElement.momenta_in,
            ms.MatrixElement.flavor_in,
            ms.MatrixElement.alpha_s_in,
        ],
        outputs=[
            ms.MatrixElement.matrix_element_out,
            ms.MatrixElement.diagram_amp2_out,
            ms.MatrixElement.color_index_out,
            ms.MatrixElement.helicity_index_out,
            ms.MatrixElement.diagram_index_out,
        ],
        diagram_count=2,
        sample_random_inputs=True,
    )
    diff_xs = ms.DifferentialCrossSection(
        matrix_element=me,
        cm_energy=e_cm,
        running_coupling=running_coupling,
        energy_scale=ms.CachedScale(),
        pid_options=PID_OPTIONS,
    )
    mapping = ms.PhaseSpaceMapping(
        [0.0] * 4, e_cm, cuts=cuts, mirror_beams=mirror_beams
    )
    return ms.Integrand(
        mapping=mapping,
        diff_xs=[diff_xs],
        pid_options=PID_OPTIONS,
        running_coupling=running_coupling,
        energy_scale=ms.EnergyScale(4),
        flavor_mirror=list(flavor_mirror),
        channel_indices=[0],
    )


def test_mirrored_flavors_need_mirror_beams(running_coupling):
    with pytest.raises(ValueError, match="mirror_beams"):
        build_integrand(running_coupling, mirror_beams=False)


def test_any_cut_is_allowed(running_coupling):
    """The orientation is drawn before the mapping, so the cuts act on the
    written event whatever they measure."""
    build_integrand(running_coupling, cuts=lepton_cut("eta", min=0.0))


def test_unmirrored_flavors_need_no_mirror_beams(running_coupling):
    build_integrand(running_coupling, mirror_beams=False, flavor_mirror=(False, False))


def instruction_names(function):
    return [instruction.instruction.name for instruction in function.instructions]


def find_instruction(function, name):
    names = instruction_names(function)
    assert names.count(name) == 1, f"expected one {name}, got {names.count(name)}"
    return function.instructions[names.index(name)]


def test_mirror_is_applied_before_the_cuts(running_coupling):
    """The only mirror is the one inside the mapping, so the matrix element is
    evaluated on the orientation the event is written with."""
    function = build_integrand(running_coupling).function()
    names = instruction_names(function)
    assert names.count("mirror_momenta") == 1
    assert names.index("mirror_momenta") < names.index("nonzero")
