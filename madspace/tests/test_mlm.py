"""Tests for the MLM back-clustering (madspace/src/phasespace/mlm_clustering.cpp
and madspace/src/kernels/mlm.hpp).

The clustering compiler turns the diagram topologies of a subprocess into a
state machine of possible clustering histories; the kernel walks that state
machine for one phase-space point, picks a history, and reads the
renormalisation / factorisation / per-jet scales off it.

There is no single "right answer" to compare against (the resonance handling and
the factorisation-scale definition differ from madevent's on purpose), so these
tests pin down properties the implementation must have:

  * the compiled state machine is well formed and reachable,
  * the kernel's outputs are finite, positive and ordered as documented,
  * the clustering is invariant under the symmetries of the phase-space point,
  * the individual clustering measures agree with the Fortran ones they were
    ported from (Template/NLO/SubProcesses/cluster.f),
  * resonant propagators are recognised inside the Breit-Wigner window.
"""

import json
import os

import numpy as np
import pytest

import madspace as ms

CM_ENERGY = 13000.0
BATCH_SIZE = 200
SEED = 5678

M_TOP = 173.0
M_Z, W_Z = 91.188, 2.4414

TEST_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data")

# g g > t t~ g g and g g > t t~ g g g, i.e. legs (g, g, t, t~, g, ...)
PROCESSES = {
    "ttgg": ("ttgg.json", [21, 21, 6, -6, 21, 21]),
    "ttggg": ("ttggg.json", [21, 21, 6, -6, 21, 21, 21]),
}


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------


def load_diagrams(file_name):
    with open(os.path.join(TEST_DATA, file_name)) as f:
        return json.load(f)


def make_topologies(diagrams):
    return [
        ms.Topology(
            ms.Diagram(
                d["incoming_masses"],
                d["outgoing_masses"],
                [ms.Propagator(*p) for p in d["propagators"]],
                d["vertices"],
            )
        )
        for d in diagrams
    ]


def make_diagram_indices(diagrams):
    """One running index per (topology, permutation) pair, matching what
    launch.py hands to MLMClustering."""
    offset = 0
    indices = []
    for d in diagrams:
        count = len(d["permutations"])
        indices.append(list(range(offset, offset + count)))
        offset += count
    return indices


def make_clustering(diagrams, **kwargs):
    kwargs.setdefault("cm_energy", CM_ENERGY)
    return ms.MLMClustering(
        make_topologies(diagrams),
        [d["permutations"] for d in diagrams],
        make_diagram_indices(diagrams),
        **kwargs,
    )


def sample_momenta(diagrams, batch_size=BATCH_SIZE, seed=SEED):
    """Phase-space points for the first topology of the process."""
    rng = np.random.default_rng(seed)
    topology = make_topologies(diagrams)[0]
    permutations = diagrams[0]["permutations"]
    mapping = ms.PhaseSpaceMapping(topology, CM_ENERGY, permutations=permutations)
    r = rng.random((batch_size, mapping.random_dim()))
    condition = (
        []
        if len(permutations) <= 1
        else [rng.integers(0, len(permutations), batch_size, dtype=np.int32)]
    )
    p_ext, *_ = mapping.map_forward([r], condition)
    return np.asarray(p_ext)


@pytest.fixture(params=sorted(PROCESSES))
def process(request):
    file_name, pdg_ids = PROCESSES[request.param]
    diagrams = load_diagrams(file_name)
    return request.param, diagrams, pdg_ids


def run(clustering, momenta):
    """The scale outputs of a clustering, as numpy arrays: ren_scale,
    fact_scale1, fact_scale2, outgoing_scales, diagram_index."""
    return run_all(clustering, momenta)[:5]


def run_all(clustering, momenta):
    """As run(), plus the xqcut_weight. The alpha_s and pdf outputs that follow
    are left out; see alphas_outputs() and pdf_outputs()."""
    return tuple(np.asarray(v) for v in clustering(momenta))[:6]


def alphas_outputs(clustering, momenta):
    """The alpha_s reweighting outputs: per-vertex scales, and the weight that
    drops an event whose vertices sit where the coupling is not usable."""
    out = clustering(momenta)
    return np.asarray(out[6]), np.asarray(out[7])


PDF_OUTPUTS = [
    "pdf_scale1",
    "pdf_scale2",
    "pdf_rw_flavor",
    "pdf_rw_x",
    "pdf_rw_q_num",
    "pdf_rw_q_den",
    "pdf_rw_active",
    "pdf_rw_beam",
    "pdf_rw_x_prev",
]


def pdf_outputs(clustering, momenta):
    """The pdf reweighting outputs (the trailing outputs of the clustering), by
    name: the scale each beam density is to be evaluated at, and one slot per
    step of the ladder it is walked back up."""
    out = clustering(momenta)
    first = len(out) - len(PDF_OUTPUTS)
    return {
        name: np.asarray(value) for name, value in zip(PDF_OUTPUTS, list(out)[first:])
    }


def assert_jet_scales_agree(reference, other, max_flip_fraction=0.05):
    """Compare two sets of per-jet clustering scales that should be identical.

    Which clustering wins a step is an argmin over the candidate measures, and
    processes with identical final-state partons produce exactly degenerate
    candidates. Recomputing the measures on a symmetry-transformed event
    perturbs them at the level of floating-point noise, which can flip the
    argmin and hand the scale to the other member of the degenerate pair. The
    set of clustering *scales* is unchanged (so mu_R and mu_F are), only which
    leg they are booked against, and it happens for a small fraction of events.
    """
    matches = np.isclose(reference, other, rtol=1e-6, atol=1e-6)
    flipped = np.mean(~matches.all(axis=1))
    assert flipped <= max_flip_fraction, (
        f"{flipped:.1%} of events changed their jet-scale assignment, "
        f"more than the {max_flip_fraction:.0%} expected from degenerate "
        "clustering candidates"
    )


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_construction_succeeds(process):
    """Every process in the test data compiles into a state machine. The
    constructor throws on an unreachable or dead-ended state, so this also
    covers the structure of the compiled machine."""
    _, diagrams, pdg_ids = process
    assert make_clustering(diagrams) is not None
    assert make_clustering(diagrams, external_pdg_ids=pdg_ids) is not None


def test_pdg_id_count_is_checked(process):
    """A wrong number of external pdg ids is a caller error, not something to
    silently truncate."""
    _, diagrams, pdg_ids = process
    with pytest.raises(Exception):
        make_clustering(diagrams, external_pdg_ids=pdg_ids[:-1])


def test_flavor_information_selects_which_legs_are_jets(process):
    """Only jets get a real clustering scale. Without pdg ids the clustering has
    to assume every leg is one; with them, the tops (legs 2 and 3 of these
    processes) never are and always fall back to sqrt(s)."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    *_, out_no_pdg, _ = run(make_clustering(diagrams), momenta)
    *_, out_pdg, _ = run(make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta)
    # the tops are not jets, so with flavor information they never get one
    assert np.all(out_pdg[:, 0] == CM_ENERGY)
    assert np.all(out_pdg[:, 1] == CM_ENERGY)
    # ... whereas without it they do, at least sometimes
    assert np.any(out_no_pdg[:, :2] < CM_ENERGY)
    # the gluons are jets either way and usually pick up a real scale
    assert np.mean(out_pdg[:, 2:].min(axis=1) < CM_ENERGY) > 0.5


def test_unclustered_legs_fall_back_to_the_collider_energy(process):
    """Every outgoing leg is reported at a scale, never at zero.

    pt_clust is what an MLM veto compares against qcut, so a leg that no QCD
    clustering assigned a scale to has to come out at a value no veto can trip
    on. madevent does the same, via the "ptclus = etot" fallback in
    Template/LO/SubProcesses/reweight.f.
    """
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=2000, seed=4321)
    for clustering in (
        make_clustering(diagrams),
        make_clustering(diagrams, external_pdg_ids=pdg_ids),
    ):
        *_, out, _ = run(clustering, momenta)
        assert np.all(out > 0.0)
        assert np.all(out <= CM_ENERGY)
        # the fallback value appears exactly, not as a rounded-down scale
        fallback = out == CM_ENERGY
        assert fallback.any()


def test_the_fallback_value_follows_the_collider_energy(process):
    """The fallback is the collider energy the clustering was built with, so
    changing it moves only the unassigned legs."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    *_, out, _ = run(make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta)
    *_, out_half, _ = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids, cm_energy=CM_ENERGY / 2),
        momenta,
    )
    assigned = out != CM_ENERGY
    assert np.array_equal(out[assigned], out_half[assigned])
    assert np.all(out_half[~assigned] == CM_ENERGY / 2)


# --------------------------------------------------------------------------
# kernel outputs
# --------------------------------------------------------------------------


def test_scales_are_finite_and_positive(process):
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    ren, fac1, fac2, out_scales, diagram = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta
    )
    for name, value in [("ren", ren), ("fac1", fac1), ("fac2", fac2)]:
        assert np.all(np.isfinite(value)), f"{name} scale not finite"
        assert np.all(value > 0.0), f"{name} scale not positive"
    assert np.all(np.isfinite(out_scales))
    assert np.all(out_scales >= 0.0)


def test_scales_are_below_the_collider_energy(process):
    """No clustering scale can exceed the collider energy."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    ren, fac1, fac2, out_scales, _ = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta
    )
    for value in (ren, fac1, fac2, out_scales):
        assert np.all(value <= CM_ENERGY)


def test_factorization_scale_does_not_exceed_renormalization_scale(process):
    """The kernel caps mu_F at mu_R, and uses one factorisation scale for both
    beams."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    ren, fac1, fac2, _, _ = run(
        make_clustering(
            diagrams,
            external_pdg_ids=pdg_ids,
            clustering_mean_pdf_ladder=False,
        ),
        momenta,
    )
    assert np.array_equal(fac1, fac2)
    assert np.all(fac1 <= ren * (1.0 + 1e-12))


def test_diagram_index_is_in_range(process):
    """The selected diagram index has to be one of the indices handed to the
    constructor, since it is passed straight to the matrix element."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    *_, diagram = run(make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta)
    total = sum(len(d["permutations"]) for d in diagrams)
    assert diagram.min() >= 0
    assert diagram.max() < total


def test_batch_entries_are_independent(process):
    """Each phase-space point is clustered on its own: running a subset of a
    batch must give the same answer as running the whole batch.

    The kernel keeps per-event scratch arrays (momenta_tmp, masses_tmp, the
    cluster history), so this catches state leaking between batch entries.
    """
    _, diagrams, pdg_ids = process
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    momenta = sample_momenta(diagrams)
    ren_full, fac_full, _, out_full, _ = run(clustering, momenta)
    # the diagram index draws a random number, so only compare the scales
    ren_half, fac_half, _, out_half, _ = run(clustering, momenta[: BATCH_SIZE // 2])
    n = BATCH_SIZE // 2
    assert np.array_equal(ren_full[:n], ren_half)
    assert np.array_equal(fac_full[:n], fac_half)
    assert np.array_equal(out_full[:n], out_half)


def test_longitudinal_boosts_barely_change_the_scales(process):
    """The clustering measures themselves are longitudinally boost invariant -
    they are built from mT, pseudorapidity differences and azimuthal angles -
    but the *choice* of clustering is not.

    compute_scale multiplies an initial-state candidate by 1.000001 when the
    emitted parton goes against the beam it is clustered onto, tested as
    sign(pz1) != sign(pz2). That sign is not boost invariant, so a boost can
    flip which candidate wins for events where two candidates are close. This is
    inherited from the Fortran, which uses the same test.

    So: pin that the effect stays small rather than asserting exact invariance.
    """
    _, diagrams, pdg_ids = process
    clustering = make_clustering(
        diagrams,
        external_pdg_ids=pdg_ids,
        clustering_mean_pdf_ladder=False,
    )
    momenta = sample_momenta(diagrams)

    rapidity = 0.3
    cosh_y, sinh_y = np.cosh(rapidity), np.sinh(rapidity)
    boosted = momenta.copy()
    energy, pz = momenta[..., 0], momenta[..., 3]
    boosted[..., 0] = cosh_y * energy + sinh_y * pz
    boosted[..., 3] = sinh_y * energy + cosh_y * pz

    ren, fac, _, out, _ = run(clustering, momenta)
    ren_b, fac_b, _, out_b, _ = run(clustering, boosted)

    unchanged = np.isclose(ren_b, ren, rtol=1e-9)
    assert unchanged.mean() > 0.8, (
        f"only {unchanged.mean():.0%} of events keep their renormalisation "
        "scale under a longitudinal boost"
    )
    # where the history is unchanged, the scales have to agree to full precision
    assert ren_b[unchanged] == pytest.approx(ren[unchanged], rel=1e-9)
    assert fac_b[unchanged] == pytest.approx(fac[unchanged], rel=1e-9)
    assert_jet_scales_agree(out[unchanged], out_b[unchanged])


def test_scales_are_azimuthally_invariant(process):
    """Rotating the event around the beam axis must not change any scale."""
    _, diagrams, pdg_ids = process
    clustering = make_clustering(
        diagrams,
        external_pdg_ids=pdg_ids,
        clustering_mean_pdf_ladder=False,
    )
    momenta = sample_momenta(diagrams)

    angle = 0.7
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotated = momenta.copy()
    px, py = momenta[..., 1], momenta[..., 2]
    rotated[..., 1] = cos_a * px - sin_a * py
    rotated[..., 2] = sin_a * px + cos_a * py

    ren, fac, _, out, _ = run(clustering, momenta)
    ren_r, fac_r, _, out_r, _ = run(clustering, rotated)
    assert ren_r == pytest.approx(ren, rel=1e-9)
    assert fac_r == pytest.approx(fac, rel=1e-9)
    assert_jet_scales_agree(out, out_r)


# The schemes whose factorisation scale comes out of madevent's beam-line walk
# (setclscales in Template/LO/SubProcesses/reweight.f) rather than from the
# softest clustering.
BEAM_LINE_SCHEMES = {
    "madevent": {"scale_scheme": "madevent"},
    "clustering_mean_pdf_ladder": {"clustering_mean_pdf_ladder": True},
}

# What the beam-line walk hands the pdf reweighting (built from transverse
# masses and invariants).
LADDER_OUTPUTS = [
    "pdf_scale1",
    "pdf_scale2",
    "pdf_rw_x",
    "pdf_rw_q_num",
    "pdf_rw_q_den",
]


def rotate_about_beam(momenta, angle):
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotated = momenta.copy()
    px, py = momenta[..., 1], momenta[..., 2]
    rotated[..., 1] = cos_a * px - sin_a * py
    rotated[..., 2] = sin_a * px + cos_a * py
    return rotated


@pytest.mark.parametrize("scheme", sorted(BEAM_LINE_SCHEMES))
def test_beam_line_scales_are_azimuthally_invariant(process, scheme):
    """The same for the schemes that walk the beam lines, which are more
    sensitive to the choice of history: which beam takes a parton decides where
    that beam's line stops being a QCD line, and with it mu_F.

    Equal measures are common there. With two massless partons left in the
    final state they balance in transverse momentum, so clustering either one
    into the beam it travels along scores the same. Ties are broken in the
    order of cluster.f's loops rather than by rounding, so nothing may change
    under a rotation, not even which leg a jet scale is booked against.
    """
    _, diagrams, pdg_ids = process
    clustering = make_clustering(
        diagrams, external_pdg_ids=pdg_ids, **BEAM_LINE_SCHEMES[scheme]
    )
    momenta = sample_momenta(diagrams)
    for angle in (0.7, -2.5):
        rotated = rotate_about_beam(momenta, angle)
        ren, fac1, fac2, out, diagram = run(clustering, momenta)
        ren_r, fac1_r, fac2_r, out_r, diagram_r = run(clustering, rotated)
        assert ren_r == pytest.approx(ren, rel=1e-9)
        assert fac1_r == pytest.approx(fac1, rel=1e-9)
        assert fac2_r == pytest.approx(fac2, rel=1e-9)
        assert np.array_equal(diagram_r, diagram)
        assert_jet_scales_agree(out, out_r, max_flip_fraction=0.0)
        ladder = pdf_outputs(clustering, momenta)
        ladder_r = pdf_outputs(clustering, rotated)
        for name in LADDER_OUTPUTS:
            assert ladder_r[name] == pytest.approx(ladder[name], rel=1e-9), name


@pytest.mark.parametrize("scheme", sorted(BEAM_LINE_SCHEMES))
def test_beam_line_scales_barely_change_under_longitudinal_boosts(process, scheme):
    """As test_longitudinal_boosts_barely_change_the_scales, for the schemes
    that walk the beam lines: the measures and the transverse masses the walk
    reads are boost invariant, but the 1 + 1e-6 against-the-beam penalty is
    not, so a boost may change the history of an event with two close
    candidates. Where it leaves the history alone, everything has to agree.
    """
    _, diagrams, pdg_ids = process
    clustering = make_clustering(
        diagrams, external_pdg_ids=pdg_ids, **BEAM_LINE_SCHEMES[scheme]
    )
    momenta = sample_momenta(diagrams)

    rapidity = 0.3
    cosh_y, sinh_y = np.cosh(rapidity), np.sinh(rapidity)
    boosted = momenta.copy()
    energy, pz = momenta[..., 0], momenta[..., 3]
    boosted[..., 0] = cosh_y * energy + sinh_y * pz
    boosted[..., 3] = sinh_y * energy + cosh_y * pz

    ren, fac1, fac2, out, diagram = run(clustering, momenta)
    ren_b, fac1_b, fac2_b, out_b, diagram_b = run(clustering, boosted)
    alphas, _ = alphas_outputs(clustering, momenta)
    alphas_b, _ = alphas_outputs(clustering, boosted)

    # The history, step by step. The vertex scales alone do not tell: when the
    # penalty moves from one beam to the other, the parton goes to the other
    # beam at the same unpenalised scale, and the later steps that move are
    # often non-QCD ones, which carry no alpha_s scale.
    unchanged = (diagram_b == diagram) & np.isclose(alphas_b, alphas, rtol=1e-9).all(
        axis=1
    )
    assert unchanged.mean() > 0.8, (
        f"only {unchanged.mean():.0%} of events keep their clustering history "
        "under a longitudinal boost"
    )
    assert ren_b[unchanged] == pytest.approx(ren[unchanged], rel=1e-9)
    assert fac1_b[unchanged] == pytest.approx(fac1[unchanged], rel=1e-9)
    assert fac2_b[unchanged] == pytest.approx(fac2[unchanged], rel=1e-9)
    assert_jet_scales_agree(out[unchanged], out_b[unchanged])


def test_scales_are_homogeneous_for_a_single_clustering():
    """Every clustering measure has mass dimension one, so scaling all momenta
    by a factor scales every returned scale by the same factor.

    This is exact only as long as update_momenta is not involved: it keeps the
    Fortran's absolute 100 GeV^2 guard against boosting into a degenerate frame
    ("prevent too extreme boost" in Template/NLO/SubProcesses/cluster.f), which
    is not scale free. A 2 -> 2 process needs a single clustering, whose
    update_momenta result is never used again, so there the homogeneity is
    exact.
    """
    diagrams = [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0],
            "propagators": [[0.0, 0.0]],
            "vertices": [["i0", "o0", "p0"], ["p0", "o1", "i1"]],
            "permutations": [[0, 1, 2, 3]],
        }
    ]
    pdg_ids = [21, 21, 21, 21]
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    momenta = sample_momenta(diagrams, batch_size=128, seed=11)
    factor = 3.0
    ren, fac, _, out, _ = run(clustering, momenta)
    ren_s, fac_s, _, out_s, _ = run(clustering, momenta * factor)
    # the only measure available here is djb_clus = (E - pz)(E + pz), which
    # cancels badly for forward partons, so this cannot be pushed much below
    # 1e-6 even though the relation is exact in real arithmetic
    assert ren_s == pytest.approx(ren * factor, rel=1e-6)
    assert fac_s == pytest.approx(fac * factor, rel=1e-6)
    # the sqrt(s) fallback is a fixed constant, so only the legs that were
    # actually assigned a clustering scale scale with the momenta
    assigned = (out != CM_ENERGY) & (out_s != CM_ENERGY)
    assert_jet_scales_agree(
        np.where(assigned, out * factor, 0.0),
        np.where(assigned, out_s, 0.0),
        max_flip_fraction=0.3,
    )


def test_scales_mostly_scale_with_the_momenta(process):
    """The same for a multi-step clustering, where the absolute boost guard in
    update_momenta breaks exact homogeneity for the events that cross it.

    Pin how often that happens: the guard is a deliberate numerical safeguard,
    but a large fraction here would mean the scales depend on the overall energy
    of the event far more than intended.
    """
    _, diagrams, _ = process
    # drop the top masses so nothing else sets an absolute scale
    massless = json.loads(json.dumps(diagrams))
    for d in massless:
        d["outgoing_masses"] = [0.0] * len(d["outgoing_masses"])
    pdg_massless = [21] * (len(massless[0]["outgoing_masses"]) + 2)

    clustering = make_clustering(massless, external_pdg_ids=pdg_massless)
    momenta = sample_momenta(massless)
    factor = 2.0
    ren, _, _, _, _ = run(clustering, momenta)
    ren_s, _, _, _, _ = run(clustering, momenta * factor)

    ratio = ren_s / (ren * factor)
    exact = np.isclose(ratio, 1.0, rtol=1e-9)
    assert np.median(ratio) == pytest.approx(1.0, rel=1e-9)
    assert exact.mean() > 0.5, (
        f"only {exact.mean():.0%} of events scale exactly; the absolute boost "
        "guard in update_momenta should affect far fewer"
    )


def test_renormalization_scale_is_a_geometric_mean(process):
    """mu_R is the geometric mean of the clustering scales, so it lies between
    the smallest and the largest of them. mu_F is the smallest QCD clustering
    scale (capped at mu_R), which bounds mu_R from below."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    ren, fac, _, out, _ = run(
        make_clustering(
            diagrams,
            external_pdg_ids=pdg_ids,
            clustering_mean_pdf_ladder=False,
        ),
        momenta,
    )
    assert np.all(ren >= fac * (1.0 - 1e-12))
    # mu_F is the smallest QCD clustering scale, so every jet scale - each of
    # which is a QCD clustering scale - is at least as large
    assigned = out != CM_ENERGY
    smallest_jet_scale = np.where(assigned, out, np.inf).min(axis=1)
    assert np.all(smallest_jet_scale >= fac * (1.0 - 1e-9))


# --------------------------------------------------------------------------
# resonances
# --------------------------------------------------------------------------


M_B = 4.7


def z_to_bb_diagram(width):
    """u u~ > Z(> b b~) g g with the Z either given its physical width, so it
    can be recognised as a resonance, or a zero width, so it cannot.

    The b quarks are massive on purpose. compute_scale returns the invariant
    mass of the pair both when the clustering is resonant and when the parent is
    massive and both children are massless (itype 8 in the Fortran), so a
    resonance decaying to massless particles cannot tell the two branches apart.
    With massive children the non-resonant case falls through to the kt measure
    instead, and the resonance branch becomes observable.

    Legs: (u, u~, b, b~, g, g); the Z propagator recombines legs 2 and 3.
    """
    return [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [M_B, M_B, 0.0, 0.0],
            # p0 = Z (b b~), p1/p2 = t-channel quark lines
            "propagators": [[M_Z, width], [0.0, 0.0], [0.0, 0.0]],
            "vertices": [
                ["o0", "o1", "p0"],
                ["i0", "o2", "p1"],
                ["p1", "o3", "p2"],
                ["p2", "p0", "i1"],
            ],
            "permutations": [[0, 1, 2, 3, 4, 5]],
        }
    ]


Z_TO_BB_PDGS = [2, -2, 5, -5, 21, 21]


def test_resonant_propagators_get_a_breit_wigner_index():
    """Only a propagator with a width can be resonant, and only in a final-state
    clustering: a t-channel line is spacelike and never goes on shell."""
    with_width = make_clustering(z_to_bb_diagram(W_Z), external_pdg_ids=Z_TO_BB_PDGS)
    without = make_clustering(z_to_bb_diagram(0.0), external_pdg_ids=Z_TO_BB_PDGS)

    assert list(with_width.bw_masses) == [pytest.approx(M_Z)]
    assert list(with_width.bw_widths) == [pytest.approx(W_Z)]
    assert list(without.bw_masses) == []

    # exactly the b b~ clustering (legs 2 and 3) carries the index
    resonant_pairs = set()
    non_terminal, _ = walk(np.asarray(with_width.cluster_state_machine), 6)
    for _, (_, transitions) in non_terminal.items():
        for data, _, _ in transitions:
            if field(data, BIT_MASS_INDEX) != 0:
                resonant_pairs.add(
                    (field(data, BIT_PARTICLE1), field(data, BIT_PARTICLE2))
                )
    assert resonant_pairs == {(2, 3)}


def test_resonant_clustering_uses_the_invariant_mass():
    """A clustering onto a propagator inside its Breit-Wigner window is scored
    with the invariant mass of the pair rather than the kt measure. Compare
    against the same topology with the width set to zero, which switches the
    resonance branch off entirely."""
    momenta = sample_momenta(z_to_bb_diagram(W_Z))
    resonant = make_clustering(
        z_to_bb_diagram(W_Z),
        external_pdg_ids=Z_TO_BB_PDGS,
        bw_cutoff=15.0,
        clustering_mean_pdf_ladder=False,
    )
    non_resonant = make_clustering(
        z_to_bb_diagram(0.0),
        external_pdg_ids=Z_TO_BB_PDGS,
        clustering_mean_pdf_ladder=False,
    )

    ren_res, fac_res, _, _, _ = run(resonant, momenta)
    ren_non, _, _, _, _ = run(non_resonant, momenta)

    # the resonance branch has to actually change the answer
    assert not np.allclose(ren_res, ren_non)
    assert np.all(np.isfinite(ren_res)) and np.all(ren_res > 0.0)
    assert np.all(fac_res <= ren_res * (1.0 + 1e-12))


def test_breit_wigner_cutoff_widens_the_resonance_window():
    """bw_cutoff scales the half width of the window in which a propagator counts
    as resonant, exactly as it does for the phase-space mapping. A cutoff of zero
    closes the window, which has to reproduce the zero-width result."""
    momenta = sample_momenta(z_to_bb_diagram(W_Z))

    closed = make_clustering(
        z_to_bb_diagram(W_Z), external_pdg_ids=Z_TO_BB_PDGS, bw_cutoff=0.0
    )
    zero_width = make_clustering(z_to_bb_diagram(0.0), external_pdg_ids=Z_TO_BB_PDGS)
    wide = make_clustering(
        z_to_bb_diagram(W_Z), external_pdg_ids=Z_TO_BB_PDGS, bw_cutoff=15.0
    )

    ren_closed, *_ = run(closed, momenta)
    ren_zero, *_ = run(zero_width, momenta)
    ren_wide, *_ = run(wide, momenta)

    assert ren_closed == pytest.approx(ren_zero, rel=1e-12)
    assert not np.allclose(ren_wide, ren_closed)


# --------------------------------------------------------------------------
# clustering measures, against the Fortran they were ported from
# --------------------------------------------------------------------------


def djb_clus_reference(p, hadronic=True):
    """djb_clus from Template/NLO/SubProcesses/cluster.f."""
    r = (p[0] - p[3]) * (p[0] + p[3]) if hadronic else p[0] * p[0]
    return max(r, 0.0)


def dj_clus_reference(p1, p2, mass1, mass2, jet_radius):
    """The both-massless / both-massive branch of dj_clus:
    min(pt1^2, pt2^2) * 2 * (cosh(deta) - cos(dphi)) / D^2, in the form the
    kernel evaluates it, 4 * (sinh^2(deta/2) + sin^2(dphi/2)), which keeps its
    digits for a nearly collinear pair."""
    pt1_sq = p1[1] ** 2 + p1[2] ** 2
    pt2_sq = p2[1] ** 2 + p2[2] ** 2
    if pt1_sq == 0.0 or pt2_sq == 0.0:
        return 0.0
    deta = np.arcsinh(p1[3] / np.sqrt(pt1_sq)) - np.arcsinh(p2[3] / np.sqrt(pt2_sq))
    dphi = np.arctan2(p1[1] * p2[2] - p1[2] * p2[1], p1[1] * p2[1] + p1[2] * p2[2])
    m_max_sq = max(mass1**2, mass2**2)
    return max(
        m_max_sq
        + min(pt1_sq, pt2_sq)
        * 4.0
        * (np.sinh(deta / 2) ** 2 + np.sin(dphi / 2) ** 2)
        / jet_radius**2,
        0.0,
    )


def durham_reference(p1, p2):
    """The lepton-collider branch of dj_clus, 2 * min(E1^2, E2^2) *
    (1 - cos_theta), as 4 * min(E1^2, E2^2) * sin^2(theta/2)."""
    cross = np.cross(p1[1:], p2[1:])
    theta = np.arctan2(np.linalg.norm(cross), np.dot(p1[1:], p2[1:]))
    return 4.0 * min(p1[0] ** 2, p2[0] ** 2) * np.sin(theta / 2) ** 2


def massless(px, py, pz):
    return np.array([np.sqrt(px * px + py * py + pz * pz), px, py, pz])


def collinear_pair_event(hadronic):
    """Two final-state partons about 1e-5 apart, legs 2 and 4 (e+ e- > d d~ g)
    or 2 and 3 (g g > g g g), with the third one recoiling and the beams along
    z: momentum is conserved. For such a pair cosh(deta) - cos(dphi), or
    1 - cos_theta, is of order 1e-10."""

    def pt_eta_phi(pt, eta, phi):
        return massless(pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta))

    a = pt_eta_phi(300.0, 2.5, 0.7)
    b = pt_eta_phi(250.0, 2.5 + 8e-6, 0.7 + 6e-6)
    if hadronic:
        c = massless(-(a[1] + b[1]), -(a[2] + b[2]), -400.0)
        final = [a, b, c]
    else:
        c = massless(-(a[1] + b[1]), -(a[2] + b[2]), -(a[3] + b[3]))
        final = [a, c, b]
    energy = sum(p[0] for p in final)
    pz = sum(p[3] for p in final)
    beam1 = np.array([(energy + pz) / 2, 0.0, 0.0, (energy + pz) / 2])
    beam2 = np.array([(energy - pz) / 2, 0.0, 0.0, -(energy - pz) / 2])
    return np.array([[beam1, beam2] + final])


@pytest.mark.parametrize("hadronic", [True, False], ids=["kt", "durham"])
def test_a_nearly_collinear_pair_keeps_its_clustering_scale(hadronic):
    """The clustering scale of two partons 1e-5 apart. Written as in cluster.f,
    cosh(deta) - cos(dphi) (or 1 - cos_theta) is the difference of two numbers
    within 1e-10 of one, which loses about 1e-6 of relative error, enough for
    scalar and SIMD math to give different scales and histories. The kernel's
    form of the measure has none of that cancellation."""
    jet_radius = 0.4
    if hadronic:
        pdg_ids = [21, 21, 21, 21, 21]
        vertices = [["o0", "o1", "p0"], ["i0", "o2", "p1"], ["p1", "p0", "i1"]]
        legs = (2, 3)
    else:
        pdg_ids = [-11, 11, 1, -1, 21]
        vertices = [["o0", "o2", "p0"], ["p0", "o1", "p1"], ["i0", "p1", "i1"]]
        legs = (2, 4)
    diagrams = [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0, 0.0],
            "propagators": [[0.0, 0.0], [0.0, 0.0]],
            "vertices": vertices,
            "permutations": [[0, 1, 2, 3, 4]],
        }
    ]
    clustering = make_clustering(
        diagrams,
        external_pdg_ids=pdg_ids,
        jet_radius=jet_radius,
        hadronic=hadronic,
        clustering_mean_pdf_ladder=False,
        jet_scale_scheme=ms.MLMClustering.JetScaleScheme.emission,
    )
    event = collinear_pair_event(hadronic)
    _, _, _, jets, _ = run(clustering, event)
    p1, p2 = event[0][legs[0]], event[0][legs[1]]
    if hadronic:
        expected = np.sqrt(dj_clus_reference(p1, p2, 0.0, 0.0, jet_radius))
    else:
        expected = np.sqrt(durham_reference(p1, p2))
    assert 1e-4 < expected < 1e-1, "not the collinear configuration intended"
    # the softest clustering is that pair: both its legs carry its scale
    for leg in legs:
        assert jets[0][leg - 2] == pytest.approx(expected, rel=1e-9)


def first_step_candidates(clustering, n_ext, event, jet_radius, masses):
    """The clustering measures the kernel evaluates in its first step.

    Which pairs are candidates at all is decided by the compiled state machine -
    only clusterings that some diagram topology allows are offered - so read
    them off it rather than enumerating every pair.
    """
    flat = np.asarray(clustering.cluster_state_machine)
    candidates = {}
    for data, _, _ in decode_transitions(flat, 0):
        p1 = field(data, BIT_PARTICLE1)
        p2 = field(data, BIT_PARTICLE2)
        if p1 < 2:
            # initial-state clustering: mT of the final-state parton, with a
            # 1.000001 penalty when it goes against the beam
            measure = np.sqrt(djb_clus_reference(event[p2]))
            if (event[p1][3] < 0.0) != (event[p2][3] < 0.0):
                measure *= 1.000001
        else:
            measure = np.sqrt(
                dj_clus_reference(
                    event[p1], event[p2], masses[p1], masses[p2], jet_radius
                )
            )
        candidates[(p1, p2)] = measure
    return candidates


def test_single_clustering_scale_matches_the_measures():
    """For a 2 -> 2 process there is exactly one clustering step, so the scale
    the kernel returns is fully predicted by the measures above.

    This is the tightest available check that the kernel walks the state machine
    and applies the same measure the Fortran does.
    """
    diagrams = [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0],
            "propagators": [[0.0, 0.0]],
            "vertices": [["i0", "o0", "p0"], ["p0", "o1", "i1"]],
            "permutations": [[0, 1, 2, 3]],
        }
    ]
    pdg_ids = [21, 21, 21, 21]
    clustering = make_clustering(
        diagrams,
        external_pdg_ids=pdg_ids,
        clustering_mean_pdf_ladder=False,
    )
    momenta = sample_momenta(diagrams, batch_size=128, seed=99)
    ren, fac, _, _, _ = run(clustering, momenta)

    # one clustering step: mu_R is that step's scale, and mu_F (the smallest QCD
    # clustering scale, capped at mu_R) has to equal it
    assert fac == pytest.approx(ren, rel=1e-12)

    masses = [0.0] * 4
    for event, scale in zip(momenta, ren):
        candidates = first_step_candidates(clustering, 4, event, 1.0, masses)
        assert candidates, "no clustering offered from the initial state"
        assert scale == pytest.approx(min(candidates.values()), rel=1e-9)


def test_the_default_jet_radius_is_madevents():
    """madevent's hidden run_card d (common/to_dj/D in kin_functions.f) and
    FxFx both put 1 in the final-state kt measure. A radius of 0.4 would make
    that measure 2.5 times larger, and the merging cut weaker for final-state
    splittings, than in madevent."""
    diagrams = load_diagrams(PROCESSES["ttggg"][0])
    pdg_ids = PROCESSES["ttggg"][1]
    momenta = sample_momenta(diagrams)
    default = run(make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta)
    one = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids, jet_radius=1.0), momenta
    )
    old = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids, jet_radius=0.4), momenta
    )
    for a, b in zip(default, one):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(default[3], old[3])


def test_kt_measure_matches_the_fortran():
    """The first clustering step of g g > g g g, where the (o0, o1) final-state
    pair competes with the initial-state candidates and is scored by the kt
    measure. Checked against a transcription of dj_clus / djb_clus from
    Template/NLO/SubProcesses/cluster.f.
    """
    jet_radius = 0.4
    diagrams = [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0, 0.0],
            "propagators": [[0.0, 0.0], [0.0, 0.0]],
            "vertices": [
                ["o0", "o1", "p0"],
                ["i0", "o2", "p1"],
                ["p1", "p0", "i1"],
            ],
            "permutations": [[0, 1, 2, 3, 4]],
        }
    ]
    pdg_ids = [21, 21, 21, 21, 21]
    clustering = make_clustering(
        diagrams,
        external_pdg_ids=pdg_ids,
        jet_radius=jet_radius,
        clustering_mean_pdf_ladder=False,
    )
    momenta = sample_momenta(diagrams, batch_size=128, seed=99)
    _, fac, _, _, _ = run(clustering, momenta)

    # the state machine has to offer the final-state pair, otherwise this test
    # would never exercise dj_clus at all
    flat = np.asarray(clustering.cluster_state_machine)
    offered = {
        (field(d, BIT_PARTICLE1), field(d, BIT_PARTICLE2))
        for d, _, _ in decode_transitions(flat, 0)
    }
    assert (2, 3) in offered

    masses = [0.0] * 5
    for event, scale in zip(momenta, fac):
        candidates = first_step_candidates(clustering, 5, event, jet_radius, masses)
        # two clustering steps here (n_ext - 3), and mu_F is the smallest QCD
        # clustering scale of the whole history, so it is at most the smallest
        # measure available in the first step
        assert scale <= min(candidates.values()) * (1.0 + 1e-9)


# --------------------------------------------------------------------------
# structure of the compiled state machine
# --------------------------------------------------------------------------
#
# The kernel walks a flat integer array. A non-terminal state is a run of
# (data, next_offset, trace_data) triples ending at the triple whose data has
# bit 30 set; a terminal state is a count followed by that many diagram indices. Which of the
# two a given offset is depends only on how many clusterings have been done, so
# the walker below tracks the depth the same way the kernel does.

BIT_PARTICLE1 = (0, 0xFF)
BIT_PARTICLE2 = (8, 0xFF)
BIT_MASS_INDEX = (16, 0xFF)
BIT_MASSIVE_IN = (24, 1)
BIT_MASSIVE_OUT1 = (25, 1)
BIT_MASSIVE_OUT2 = (26, 1)
BIT_IS_QCD = (27, 1)
BIT_IS_JET1 = (28, 1)
BIT_IS_JET2 = (29, 1)
BIT_IS_LAST = (30, 1)

# Words per transition: (data, next_offset, trace_data).
STATE_ITEM_SIZE = 3

# trace_data: the low two bits say which daughter the mother's parton line
# continues into, the next two carry the mother's own flavour.
TRACE_MODE_MASK = 0x3
TRACE_FIRST, TRACE_SECOND, TRACE_HARDER, TRACE_BOTH = 0, 1, 2, 3
TRACE_MODES = {TRACE_FIRST, TRACE_SECOND, TRACE_HARDER, TRACE_BOTH}
TRACE_IS_JET_IN = 1 << 2
TRACE_IS_COLORED_IN = 1 << 3


def trace_mode(trace):
    return trace & TRACE_MODE_MASK


def field(data, spec):
    shift, mask = spec
    return (data >> shift) & mask


def decode_transitions(machine, offset):
    """The (data, next_offset, trace) triples of the non-terminal state at
    `offset`."""
    transitions = []
    while True:
        assert offset + STATE_ITEM_SIZE <= len(
            machine
        ), "transition list runs past the machine"
        data, next_offset, trace = machine[offset : offset + STATE_ITEM_SIZE]
        transitions.append((data, next_offset, trace))
        if field(data, BIT_IS_LAST):
            return transitions
        offset += STATE_ITEM_SIZE


def walk(machine, n_ext):
    """Visit every reachable state, returning
    {offset: (depth, transitions)} for non-terminal states and
    {offset: diagram indices} for terminal ones."""
    cluster_max = n_ext - 3
    non_terminal, terminal = {}, {}
    todo = [(0, 0)]
    seen = set()
    while todo:
        offset, depth = todo.pop()
        if (offset, depth) in seen:
            continue
        seen.add((offset, depth))
        if depth == cluster_max:
            count = machine[offset]
            assert count > 0, f"terminal state at {offset} selects no diagram"
            assert offset + count < len(machine)
            terminal[offset] = list(machine[offset + 1 : offset + 1 + count])
            continue
        transitions = decode_transitions(machine, offset)
        non_terminal[offset] = (depth, transitions)
        for _, next_offset, _ in transitions:
            todo.append((next_offset, depth + 1))
    return non_terminal, terminal


@pytest.fixture
def machine(process):
    _, diagrams, pdg_ids = process
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    n_ext = len(pdg_ids)
    return clustering, np.asarray(clustering.cluster_state_machine), n_ext


def test_state_machine_is_fully_reachable(machine):
    """Every state the walk can reach is well formed, and a state is never
    reached at two different depths (which would make the terminal/non-terminal
    reading of an offset ambiguous)."""
    _, flat, n_ext = machine
    non_terminal, terminal = walk(flat, n_ext)
    assert non_terminal, "no clustering states compiled"
    assert terminal, "no terminal states compiled"
    depths = {}
    for offset, (depth, _) in non_terminal.items():
        assert depths.setdefault(offset, depth) == depth
    for offset in terminal:
        assert offset not in non_terminal


def test_state_machine_transitions_are_in_range(machine):
    """Every transition names two distinct external particles in range and
    points at an offset inside the machine."""
    _, flat, n_ext = machine
    non_terminal, _ = walk(flat, n_ext)
    for offset, (_, transitions) in non_terminal.items():
        for data, next_offset, _ in transitions:
            particle1 = field(data, BIT_PARTICLE1)
            particle2 = field(data, BIT_PARTICLE2)
            # the merged pseudo-particle is always named by the lower index, so
            # the kernel's is_initial test (particle1 < 2) is meaningful
            assert particle1 < particle2 < n_ext
            assert 0 <= next_offset < len(flat)


def test_state_machine_has_no_duplicate_transitions(machine):
    """A state reached along several paths must be expanded only once.
    Expanding it again appends a second copy of the same transitions, which
    grows the table by a factor that blows up with the multiplicity."""
    _, flat, n_ext = machine
    non_terminal, _ = walk(flat, n_ext)
    for offset, (_, transitions) in non_terminal.items():
        keys = [
            (field(d, BIT_PARTICLE1), field(d, BIT_PARTICLE2), nxt)
            for d, nxt, _ in transitions
        ]
        assert len(keys) == len(set(keys)), f"duplicate transitions at state {offset}"


def test_state_machine_marks_exactly_one_last_transition(machine):
    """The kernel decides a state is finished when it sees the is_last bit, so
    exactly one transition per state carries it, and it is the final one."""
    _, flat, n_ext = machine
    non_terminal, _ = walk(flat, n_ext)
    for offset, (_, transitions) in non_terminal.items():
        flags = [field(d, BIT_IS_LAST) for d, _, _ in transitions]
        assert flags[-1] == 1
        assert sum(flags) == 1


def test_state_machine_mass_indices_are_in_range(machine):
    """mass_index is a 1-based index into the Breit-Wigner tables, or 0 for
    'never resonant'. An out-of-range value would read past those tables."""
    clustering, flat, n_ext = machine
    bw_count = len(clustering.bw_masses)
    assert len(clustering.bw_widths) == bw_count
    non_terminal, _ = walk(flat, n_ext)
    for _, (_, transitions) in non_terminal.items():
        for data, _, _ in transitions:
            assert field(data, BIT_MASS_INDEX) <= bw_count


def test_initial_state_clusterings_are_never_resonant(machine):
    """A t-channel propagator is spacelike, so it can never be on shell."""
    _, flat, n_ext = machine
    non_terminal, _ = walk(flat, n_ext)
    for _, (_, transitions) in non_terminal.items():
        for data, _, _ in transitions:
            if field(data, BIT_PARTICLE1) < 2:
                assert field(data, BIT_MASS_INDEX) == 0
                # a beam leg is not a final-state particle and gets no jet scale
                assert field(data, BIT_IS_JET1) == 0


def test_state_machine_trace_modes_are_valid(machine):
    """Every transition says how the mother's parton line continues into its
    daughters, and an initial-state clustering always carries the beam line on,
    which is daughter 1 by construction."""
    _, flat, n_ext = machine
    non_terminal, _ = walk(flat, n_ext)
    for _, (_, transitions) in non_terminal.items():
        for data, _, trace in transitions:
            assert trace_mode(trace) in TRACE_MODES
            if field(data, BIT_PARTICLE1) < 2:
                assert trace_mode(trace) == TRACE_FIRST


def trace_modes_of(clustering, n_ext):
    non_terminal, _ = walk(np.asarray(clustering.cluster_state_machine), n_ext)
    return {
        (field(data, BIT_PARTICLE1), field(data, BIT_PARTICLE2)): trace_mode(trace)
        for _, (_, transitions) in non_terminal.items()
        for data, _, trace in transitions
    }


def three_gluon_diagram(propagator_pdgs):
    """g g > g g g through an s-channel gluon splitting, legs (g, g, g, g, g)."""
    return [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0, 0.0],
            "propagators": [[0.0, 0.0, 0, 0.0, 0.0, pdg] for pdg in propagator_pdgs],
            "vertices": [
                ["o0", "o1", "p0"],
                ["i0", "o2", "p1"],
                ["p1", "p0", "i1"],
            ],
            "permutations": [[0, 1, 2, 3, 4]],
        }
    ]


def test_gluon_splitting_follows_the_harder_gluon():
    """A g -> g g vertex carries the parton line into the harder gluon, as
    ipartupdate does. p0 recombines legs 2 and 3 into a gluon."""
    clustering = make_clustering(
        three_gluon_diagram([21, 21]), external_pdg_ids=[21] * 5
    )
    assert trace_modes_of(clustering, 5)[(2, 3)] == TRACE_HARDER


def test_gluon_to_quarks_carries_the_line_into_both():
    """A g -> q qbar vertex has no single continuing line, so both daughters
    carry it on and both get booked at later vertices."""
    diagrams = three_gluon_diagram([21, 21])
    diagrams[0]["propagators"][0] = [0.0, 0.0, 0, 0.0, 0.0, 21]
    clustering = make_clustering(diagrams, external_pdg_ids=[21, 21, 1, -1, 21])
    assert trace_modes_of(clustering, 5)[(2, 3)] == TRACE_BOTH


def test_quark_line_follows_the_quark():
    """A q -> q g vertex keeps the line on the quark, whichever daughter it is."""
    # p0 recombines legs 2 and 3 into a quark
    quark_first = make_clustering(
        three_gluon_diagram([1, 21]), external_pdg_ids=[21, 21, 1, 21, 21]
    )
    assert trace_modes_of(quark_first, 5)[(2, 3)] == TRACE_FIRST
    quark_second = make_clustering(
        three_gluon_diagram([1, 21]), external_pdg_ids=[21, 21, 21, 1, 21]
    )
    assert trace_modes_of(quark_second, 5)[(2, 3)] == TRACE_SECOND


def test_terminal_states_select_known_diagrams(machine):
    _, flat, n_ext = machine
    _, terminal = walk(flat, n_ext)
    for offset, diagrams in terminal.items():
        assert len(diagrams) == len(set(diagrams))
        assert all(index >= 0 for index in diagrams)


def massive_leg_permutation_diagram():
    """A topology whose permutations move a massive leg onto a slot that a
    different permutation gives to a massless one.

    The shipped fixtures cannot exercise the slot-versus-leg question at all:
    their permutations only exchange identical gluons, so the two orderings
    agree and indexing by the wrong one is invisible. Here leg 2 is the top
    under the first permutation and a gluon under the second.
    """
    return [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [M_TOP, 0.0, 0.0],
            "propagators": [[0.0, 0.0], [0.0, 0.0]],
            "vertices": [
                ["o0", "o1", "p0"],
                ["i0", "o2", "p1"],
                ["p1", "p0", "i1"],
            ],
            "permutations": [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
        }
    ]


def test_permutations_moving_a_mass_between_legs_are_rejected():
    """external_masses is indexed by external leg, while a topology's masses
    are indexed by topology slot, so the permutation has to be applied. The
    kernel keeps a single mass per leg, so permutations that disagree about a
    leg's mass cannot be represented at all and have to be refused rather than
    silently given one of the two answers.
    """
    with pytest.raises(Exception) as caught:
        make_clustering(
            massive_leg_permutation_diagram(), external_pdg_ids=[21, 21, 6, 21, 21]
        )
    assert "mass" in str(caught.value).lower()


def test_external_masses_are_in_leg_order(process):
    """The kernel indexes external_masses by external leg while a topology's
    masses are indexed by topology slot, so the permutation has to be applied.
    For these processes legs 2 and 3 are the tops.

    On its own this does not pin the ordering down: these fixtures permute only
    identical gluons, so slot and leg order coincide. The case that tells them
    apart is test_permutations_moving_a_mass_between_legs_are_rejected.
    """
    _, diagrams, pdg_ids = process
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    masses = list(clustering.external_masses)
    assert len(masses) == len(pdg_ids)
    assert masses[0] == 0.0 and masses[1] == 0.0
    assert masses[2] == pytest.approx(M_TOP)
    assert masses[3] == pytest.approx(M_TOP)
    assert all(m == 0.0 for m in masses[4:])


# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------


def test_extreme_kinematics_do_not_break_the_walk(process):
    """The boost and rotation applied after an initial-state clustering can
    leave a pair exactly collinear, making the clustering measure inf or NaN.
    The kernel has to keep choosing a clustering anyway: if nothing wins a step
    it would jump to a stale or unset state offset and read outside the state
    machine.

    Strongly asymmetric beam energies reach that configuration. Three guards
    stand between that crash and this assertion - dj_clus zeroing a NaN, the selection
    mapping a non-finite measure onto SCALE_MAX, and taking the first candidate
    unconditionally - and they are deliberately redundant, so dropping any one
    of them leaves the others covering it. What is pinned here is the behaviour
    they exist for, not each guard individually.
    """
    _, diagrams, pdg_ids = process
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    momenta = sample_momenta(diagrams, batch_size=2000, seed=1234)
    ren, fac1, fac2, out, diagram = run(clustering, momenta)
    assert np.all(np.isfinite(ren))
    assert np.all(np.isfinite(fac1)) and np.all(np.isfinite(fac2))
    assert np.all(np.isfinite(out))
    total = sum(len(d["permutations"]) for d in diagrams)
    assert np.all((diagram >= 0) & (diagram < total))


def test_exactly_collinear_momenta_are_handled(process):
    """The same thing, but forced: put two final-state partons exactly along the
    beam axis so the pseudorapidity of the pair is genuinely undefined."""
    _, diagrams, pdg_ids = process
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    momenta = sample_momenta(diagrams, batch_size=32, seed=7)
    degenerate = momenta.copy()
    # zero the transverse components of the last two legs
    degenerate[:, -2:, 1] = 0.0
    degenerate[:, -2:, 2] = 0.0
    ren, fac1, _, out, diagram = run(clustering, degenerate)
    assert np.all(np.isfinite(ren))
    assert np.all(np.isfinite(fac1))
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------
# the jet-scale scheme switch
# --------------------------------------------------------------------------


EMISSION = ms.MLMClustering.JetScaleScheme.emission
PRODUCTION = ms.MLMClustering.JetScaleScheme.production


def test_production_scheme_is_never_softer_than_emission(process):
    """The two schemes book the same vertices onto a leg plus, for "production",
    every later vertex on the line the leg belongs to. Since "production" keeps
    the hardest of those and "emission" keeps the one "production" starts from,
    the production scale of a leg can only be larger or equal.
    """
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=1000, seed=2468)
    *_, emission, _ = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=EMISSION),
        momenta,
    )
    *_, production, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=PRODUCTION
        ),
        momenta,
    )
    assert np.all(production >= emission * (1.0 - 1e-12))


def gluon_chain_diagram():
    """g g > g g g g through a chain of two s-channel gluon splittings, so that
    one gluon line carries on past the vertex that emitted it:
    p0 = g(o0, o1), p1 = g(p0, o2). Legs (g, g, g, g, g, g)."""
    return [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0, 0.0, 0.0],
            "propagators": [[0.0, 0.0, 0, 0.0, 0.0, 21] for _ in range(3)],
            "vertices": [
                ["o0", "o1", "p0"],
                ["p0", "o2", "p1"],
                ["i0", "o3", "p2"],
                ["p2", "p1", "i1"],
            ],
            "permutations": [[0, 1, 2, 3, 4, 5]],
        }
    ]


def test_the_two_schemes_disagree_on_continuing_lines():
    """The schemes differ only for a leg whose parton line carries on past the
    vertex that emitted it. In the chain above the harder of the first two
    gluons stays on the line and is booked again at the second vertex, so
    "production" gives it the harder of the two scales and "emission" the
    softer. That is the whole point of the switch."""
    diagrams = gluon_chain_diagram()
    pdg_ids = [21] * 6
    momenta = sample_momenta(diagrams, batch_size=1000, seed=2468)
    *_, emission, _ = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=EMISSION),
        momenta,
    )
    *_, production, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=PRODUCTION
        ),
        momenta,
    )
    differs = ~np.isclose(production, emission)
    assert differs.any(), "the two schemes gave identical jet scales"
    # and where they differ it is because production kept a harder vertex
    assert np.all(production[differs] > emission[differs])


def test_the_schemes_coincide_when_no_line_continues(process):
    """With propagator flavours unknown a merged node is not recognised as a
    jet, so nothing is booked onto a continuing line and the two schemes fall
    back to the same answer. The ttgg/ttggg fixtures carry propagator masses
    only, so this pins that degradation as intentional rather than accidental.
    """
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=500, seed=2468)
    *_, emission, _ = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=EMISSION),
        momenta,
    )
    *_, production, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=PRODUCTION
        ),
        momenta,
    )
    assert np.array_equal(production, emission)


def test_the_schemes_agree_for_a_single_clustering():
    """A 2 -> 3 process has one clustering per leg, so nothing continues past
    the vertex that emitted it and the two schemes have to coincide."""
    diagrams = three_gluon_diagram([21, 21])
    pdg_ids = [21] * 5
    momenta = sample_momenta(diagrams, batch_size=256, seed=1357)
    *_, emission, _ = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=EMISSION),
        momenta,
    )
    *_, production, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=PRODUCTION
        ),
        momenta,
    )
    assert np.array_equal(production, emission)


def test_the_scheme_does_not_change_the_scales(process):
    """The switch only moves which leg a clustering scale is booked against; the
    clustering history, and therefore mu_R and mu_F, are untouched."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=1000, seed=2468)
    ren_e, fac_e, _, _, _ = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=EMISSION),
        momenta,
    )
    ren_p, fac_p, _, _, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=PRODUCTION
        ),
        momenta,
    )
    assert np.array_equal(ren_e, ren_p)
    assert np.array_equal(fac_e, fac_p)


def test_production_is_the_default():
    """The default matches madevent, which is the validated reference."""
    diagrams = three_gluon_diagram([21, 21])
    assert ms.MLMClustering.JetScaleScheme.__members__["production"] == PRODUCTION
    default = make_clustering(diagrams, external_pdg_ids=[21] * 5)
    explicit = make_clustering(
        diagrams, external_pdg_ids=[21] * 5, jet_scale_scheme=PRODUCTION
    )
    momenta = sample_momenta(diagrams, batch_size=64, seed=11)
    assert np.array_equal(run(default, momenta)[3], run(explicit, momenta)[3])


# --------------------------------------------------------------------------
# the generation-level merging cut
# --------------------------------------------------------------------------


def xqcut_weights(diagrams, pdg_ids, momenta, xqcut, **kwargs):
    clustering = make_clustering(
        diagrams, external_pdg_ids=pdg_ids, xqcut=xqcut, **kwargs
    )
    return run_all(clustering, momenta)[5]


def test_xqcut_is_off_by_default(process):
    """Without a merging cut every event keeps its weight, so turning the
    feature on is the only thing that can change an existing result."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    weights = run_all(make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta)[5]
    assert np.all(weights == 1.0)


def test_xqcut_weight_is_a_veto(process):
    """It multiplies the event weight, so it has to be exactly zero or one."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    weights = xqcut_weights(diagrams, pdg_ids, momenta, 50.0)
    assert np.all((weights == 0.0) | (weights == 1.0))
    assert np.any(weights == 0.0), "a 50 GeV cut should reject something"


def test_xqcut_rejects_monotonically(process):
    """Raising the cut can only ever reject more events."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=1000, seed=808)
    kept = [
        float(xqcut_weights(diagrams, pdg_ids, momenta, cut).sum())
        for cut in (0.0, 10.0, 30.0, 60.0, 120.0)
    ]
    assert kept == sorted(kept, reverse=True), kept
    assert kept[0] == len(momenta)
    assert kept[-1] < kept[0]


def test_xqcut_rejects_exactly_the_soft_jet_emissions(process):
    """madevent's rule: a clustering step whose daughter is still a bare
    final-state jet has to be at or above xqcut. In the "emission" scheme the
    per-jet scales are booked at exactly those steps, so the veto can be
    recomputed from them and compared."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=500, seed=99)
    cut = 40.0
    clustering = make_clustering(
        diagrams,
        external_pdg_ids=pdg_ids,
        xqcut=cut,
        jet_scale_scheme=EMISSION,
    )
    *_, jet_scales, _, weights = run_all(clustering, momenta)
    # a leg left at the sqrt(s) fallback was never booked and cannot fail
    booked = jet_scales < CM_ENERGY
    softest = np.where(booked, jet_scales, np.inf).min(axis=1)
    expected = np.where(np.isfinite(softest) & (softest < cut), 0.0, 1.0)
    assert np.array_equal(weights, expected)


def test_xqcut_does_not_disturb_the_scales(process):
    """The cut only vetoes. Events that survive keep exactly the scales the
    clustering would have given without it."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    ren_off, fac_off, _, out_off, _ = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta
    )
    ren_on, fac_on, _, out_on, _, weights = run_all(
        make_clustering(diagrams, external_pdg_ids=pdg_ids, xqcut=40.0), momenta
    )
    keep = weights > 0
    assert keep.any()
    assert np.array_equal(ren_off[keep], ren_on[keep])
    assert np.array_equal(fac_off[keep], fac_on[keep])
    assert np.array_equal(out_off[keep], out_on[keep])


def test_xqcut_survivors_have_no_soft_jet_scale(process):
    """Every booked jet scale of a surviving event is at or above the cut,
    which is what makes the merging scale mean anything downstream."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=1000, seed=4242)
    cut = 35.0
    clustering = make_clustering(
        diagrams, external_pdg_ids=pdg_ids, xqcut=cut, jet_scale_scheme=EMISSION
    )
    *_, jet_scales, _, weights = run_all(clustering, momenta)
    booked = jet_scales[weights > 0]
    booked = booked[booked < CM_ENERGY]
    assert len(booked) > 0
    assert np.all(booked >= cut)


def test_xqcut_does_not_depend_on_the_jet_scale_scheme(process):
    """The veto is defined on the clustering history, which the scheme switch
    does not touch: it only moves which leg a scale is booked against."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=500, seed=17)
    emission = xqcut_weights(
        diagrams, pdg_ids, momenta, 40.0, jet_scale_scheme=EMISSION
    )
    production = xqcut_weights(
        diagrams, pdg_ids, momenta, 40.0, jet_scale_scheme=PRODUCTION
    )
    assert np.array_equal(emission, production)


# --------------------------------------------------------------------------
# the alpha_s and pdf reweighting a merged event carries
# --------------------------------------------------------------------------
#
# Both are corrections madevent applies to a merged event that a single scale
# does not describe: the coupling belongs at the scale of each emission, and
# the beam density belongs at the scale that took the parton out of the beam.
# The reference is the rewgt loop of Template/LO/SubProcesses/reweight.f.
#
# What is checkable here is the bookkeeping, not the densities: which one is
# asked for is only settled once the flavour has been sampled, which happens in
# the integrand and not in the clustering.


def t_channel_diagram():
    """u u~ > b b~ g g through a t-channel quark line, with the propagator
    flavours filled in.

    The pdf reweighting follows a beam's parton line through the clustering, so
    a test of it needs a process where a beam emits more than once and stays a
    parton in between. Here beam 0 emits both gluons before the Z attaches: the
    line is a u throughout, which is also the case the "same flavour as the
    beam" class exists for.

    The propagator pdg ids matter and the json fixtures do not carry them. A
    line whose flavour is unknown is not a parton as far as the reweighting is
    concerned, so on those fixtures the chain stops at the first clustering and
    nothing is exercised.
    """
    return [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [M_B, M_B, 0.0, 0.0],
            # p0 = Z (b b~), p1/p2 = the t-channel u line
            "propagators": [(M_Z, W_Z, 23), (0.0, 0.0, 2), (0.0, 0.0, 2)],
            "vertices": [
                ["o0", "o1", "p0"],
                ["i0", "o2", "p1"],
                ["p1", "o3", "p2"],
                ["p2", "p0", "i1"],
            ],
            "permutations": [[0, 1, 2, 3, 4, 5]],
        }
    ]


T_CHANNEL_PDGS = [2, -2, 5, -5, 21, 21]


def t_channel_clustering(**kwargs):
    kwargs.setdefault("scale_scheme", ms.MLMClustering.ScaleScheme.madevent)
    diagrams = t_channel_diagram()
    return ms.MLMClustering(
        [
            ms.Topology(
                ms.Diagram(
                    d["incoming_masses"],
                    d["outgoing_masses"],
                    [
                        ms.Propagator(mass=m, width=w, pdg_id=i)
                        for m, w, i in d["propagators"]
                    ],
                    d["vertices"],
                )
            )
            for d in diagrams
        ],
        [d["permutations"] for d in diagrams],
        make_diagram_indices(diagrams),
        cm_energy=CM_ENERGY,
        external_pdg_ids=T_CHANNEL_PDGS,
        **kwargs,
    )


@pytest.fixture(scope="module")
def t_channel_momenta():
    return sample_momenta(t_channel_diagram(), batch_size=2000)


def test_alphas_scales_are_clustering_scales_or_zero(process):
    """Every vertex is handed either its own scale or 0, the latter meaning
    "not reweighted" (evaluated at mu_R). Not mu_R itself: a reweighted vertex
    can sit exactly at mu_R, and the systematics vary the two differently."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    scales, weight = alphas_outputs(clustering, momenta)
    assert np.all(np.isfinite(scales))
    assert np.all(scales >= 0.0)
    assert np.all((weight == 0.0) | (weight == 1.0))
    # gluon-only processes: every vertex is a QCD one, so few of them should
    # have been left out
    assert np.mean(scales == 0.0) < 0.5
    # without alpha_s reweighting none is
    clustering = make_clustering(
        diagrams,
        external_pdg_ids=pdg_ids,
        alphas_scheme=ms.MLMClustering.AlphasScheme.none,
    )
    scales, weight = alphas_outputs(clustering, momenta)
    assert np.all(scales == 0.0) and np.all(weight == 1.0)


def test_pdf_reweighting_off_leaves_the_density_where_the_scale_is(
    t_channel_momenta,
):
    clustering = t_channel_clustering(pdf_reweighting=False)
    _, fact1, fact2, _, _ = run(clustering, t_channel_momenta)
    out = pdf_outputs(clustering, t_channel_momenta)
    assert out["pdf_scale1"] == pytest.approx(fact1)
    assert out["pdf_scale2"] == pytest.approx(fact2)
    assert np.all(out["pdf_rw_active"] == 0.0)


def test_clustering_mean_pdf_ladder_takes_only_the_densities_from_madevent(
    t_channel_momenta,
):
    """The ladder under clustering_mean is madevent's pdf half grafted onto
    clustering_mean: the densities, the factorisation scales they end at and
    every ladder slot are madevent's, while mu_R, the jet scales, the alpha_s
    vertex scales and the xqcut veto stay clustering_mean's."""
    scheme = ms.MLMClustering.ScaleScheme
    mean = t_channel_clustering(
        scale_scheme=scheme.clustering_mean,
        xqcut=20.0,
        clustering_mean_pdf_ladder=False,
    )
    madevent = t_channel_clustering(scale_scheme=scheme.madevent, xqcut=20.0)
    # on by default
    ladder = t_channel_clustering(scale_scheme=scheme.clustering_mean, xqcut=20.0)
    # switched off, clustering_mean keeps the density at its scale
    assert np.all(pdf_outputs(mean, t_channel_momenta)["pdf_rw_active"] == 0.0)

    ren, _, _, jets, diagram, veto = run_all(ladder, t_channel_momenta)
    ren_m, _, _, jets_m, diagram_m, veto_m = run_all(mean, t_channel_momenta)
    assert np.array_equal(ren, ren_m)
    assert np.array_equal(jets, jets_m)
    assert np.array_equal(diagram, diagram_m)
    assert np.array_equal(veto, veto_m)
    for got, want in zip(
        alphas_outputs(ladder, t_channel_momenta),
        alphas_outputs(mean, t_channel_momenta),
    ):
        assert np.array_equal(got, want)

    _, fact1, fact2, _, _ = run(ladder, t_channel_momenta)
    _, fact1_e, fact2_e, _, _ = run(madevent, t_channel_momenta)
    assert np.array_equal(fact1, fact1_e)
    assert np.array_equal(fact2, fact2_e)
    out = pdf_outputs(ladder, t_channel_momenta)
    out_e = pdf_outputs(madevent, t_channel_momenta)
    assert (out["pdf_rw_active"] == 1.0).any()
    for name in PDF_OUTPUTS:
        assert np.array_equal(out[name], out_e[name]), name

    # and the switch is inert without pdf_reweighting
    off = t_channel_clustering(
        scale_scheme=scheme.clustering_mean,
        xqcut=20.0,
        pdf_reweighting=False,
    )
    _, fact1_o, fact2_o, _, _ = run(off, t_channel_momenta)
    _, fact1_m, fact2_m, _, _ = run(mean, t_channel_momenta)
    assert np.array_equal(fact1_o, fact1_m)
    assert np.array_equal(fact2_o, fact2_m)


@pytest.mark.parametrize(
    "history", list(ms.MLMClustering.ClusteringHistory.__members__.values())
)
@pytest.mark.parametrize(
    "jet_scale_scheme", list(ms.MLMClustering.JetScaleScheme.__members__.values())
)
def test_the_ladder_leaves_every_other_clustering_mean_output_alone(
    process, history, jet_scale_scheme
):
    """The walk under clustering_mean + ladder only writes the densities and
    mu_F; everything else it computes is madevent's, and has to stay out of the
    event. Checked on the gluon-rich processes, for every history and both jet
    scale schemes, against plain clustering_mean (every non-pdf output, exactly)
    and against madevent (mu_F and every ladder output, exactly)."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    scheme = ms.MLMClustering.ScaleScheme

    def clustering(**kwargs):
        return make_clustering(
            diagrams,
            external_pdg_ids=pdg_ids,
            xqcut=20.0,
            clustering_history=history,
            jet_scale_scheme=jet_scale_scheme,
            **kwargs,
        )

    ladder = clustering(scale_scheme=scheme.clustering_mean)
    mean = clustering(
        scale_scheme=scheme.clustering_mean, clustering_mean_pdf_ladder=False
    )
    madevent = clustering(scale_scheme=scheme.madevent)

    ren, fact1, fact2, jets, diagram, veto = run_all(ladder, momenta)
    ren_m, _, _, jets_m, diagram_m, veto_m = run_all(mean, momenta)
    assert np.array_equal(ren, ren_m)
    assert np.array_equal(jets, jets_m)
    assert np.array_equal(diagram, diagram_m)
    assert np.array_equal(veto, veto_m)
    for got, want in zip(
        alphas_outputs(ladder, momenta), alphas_outputs(mean, momenta)
    ):
        assert np.array_equal(got, want)

    _, fact1_e, fact2_e, _, diagram_e, _ = run_all(madevent, momenta)
    # the same history, or mu_F and the ladder have nothing to agree on
    same = diagram == diagram_e
    assert same.mean() > 0.5
    assert np.array_equal(fact1[same], fact1_e[same])
    assert np.array_equal(fact2[same], fact2_e[same])
    # (no ladder step fires on these two fixtures, under madevent either: the
    # steps themselves are compared on the t-channel process above)
    out = pdf_outputs(ladder, momenta)
    out_e = pdf_outputs(madevent, momenta)
    for name in PDF_OUTPUTS:
        assert np.array_equal(out[name][same], out_e[name][same]), name


def test_pdf_reweighting_drops_the_density_down_the_ladder(t_channel_momenta):
    """The point of the whole thing: with it on, the density is asked for at or
    below the factorisation scale, and for a real part of the sample strictly
    below it. If it were never below there would be nothing to walk back up."""
    clustering = t_channel_clustering()
    _, fact1, fact2, _, _ = run(clustering, t_channel_momenta)
    out = pdf_outputs(clustering, t_channel_momenta)
    for scale, fact in [(out["pdf_scale1"], fact1), (out["pdf_scale2"], fact2)]:
        assert np.all(np.isfinite(scale))
        assert np.all(scale > 0.0)
        assert np.all(scale <= fact * (1.0 + 1e-9))
        assert np.mean(scale < fact * (1.0 - 1e-9)) > 0.1


def test_pdf_reweighting_steps_only_ever_raise_the_scale(t_channel_momenta):
    """A step exists precisely to move the density from where it was last
    evaluated up to this clustering, so its two scales are ordered. reweight.f
    reaches the same place through its pt2pdf(ida) < q2now test."""
    out = pdf_outputs(t_channel_clustering(), t_channel_momenta)
    active = out["pdf_rw_active"] == 1.0
    assert active.any(), "no step fired; this test would prove nothing"
    assert np.all(out["pdf_rw_q_num"][active] > out["pdf_rw_q_den"][active])


def test_pdf_reweighting_ends_at_the_factorisation_scale(t_channel_momenta):
    """A chain that fired at all has to arrive back at the scale the event is
    reported at, otherwise the lowering it started from is never undone."""
    clustering = t_channel_clustering()
    _, fact1, fact2, _, _ = run(clustering, t_channel_momenta)
    out = pdf_outputs(clustering, t_channel_momenta)
    fact = np.stack([fact1, fact2], axis=1)
    for beam in (0, 1):
        on_beam = (out["pdf_rw_active"] == 1.0) & (out["pdf_rw_beam"] == beam)
        fired = on_beam.any(axis=1)
        if not fired.any():
            continue
        top = np.max(np.where(on_beam, out["pdf_rw_q_num"], 0.0), axis=1)
        assert top[fired] == pytest.approx(fact[fired, beam], rel=1e-9)


def test_inert_pdf_slots_cannot_change_the_weight(t_channel_momenta):
    """A slot no step claimed still costs two density evaluations, because the
    graph is the same for every event. Its two scales are equal so that its
    ratio is exactly one whatever those come out as."""
    out = pdf_outputs(t_channel_clustering(), t_channel_momenta)
    inert = out["pdf_rw_active"] == 0.0
    assert inert.any()
    assert out["pdf_rw_q_num"][inert] == pytest.approx(out["pdf_rw_q_den"][inert])
    assert np.all(out["pdf_rw_q_den"][inert] > 0.0)


def test_pdf_reweighting_only_ever_takes_momentum_off_the_beam(
    t_channel_momenta,
):
    """The fraction travels down the ladder as a product of the clustering z,
    each of which reweight.f only applies when it lies in (0, 1)."""
    x = pdf_outputs(t_channel_clustering(), t_channel_momenta)["pdf_rw_x"]
    assert np.all(x > 0.0)
    assert np.all(x <= 1.0)
    # and it does move: a chain that never rescaled x would be asking for the
    # density of the beam parton at every rung
    assert np.mean(x < 1.0) > 0.05


def test_pdf_flavor_classes_stay_inside_the_table(t_channel_momenta):
    """The class is an index the consumer looks up, so an out-of-range one
    would read off the end of a table rather than fail visibly."""
    clustering = t_channel_clustering()
    class_count = 3 + len(clustering.pdf_absolute_pdgs)
    out = pdf_outputs(clustering, t_channel_momenta)
    assert np.all(out["pdf_rw_flavor"] >= 0)
    assert np.all(out["pdf_rw_flavor"] < class_count)
    # the t-channel line is a u throughout, which is beam 0's own flavour, so
    # no absolute flavour is needed and none should have been allocated
    assert list(clustering.pdf_absolute_pdgs) == []
    active = out["pdf_rw_active"] == 1.0
    assert np.all(out["pdf_rw_flavor"][active] == 1)


def test_pdf_steps_name_a_real_beam(t_channel_momenta):
    beam = pdf_outputs(t_channel_clustering(), t_channel_momenta)["pdf_rw_beam"]
    assert np.all((beam == 0.0) | (beam == 1.0))


def test_pdf_reweighting_is_off_under_the_other_scale_scheme(t_channel_momenta):
    """Under clustering_mean with the ladder switched off there is no beam
    parton line to walk, so asking for the reweighting there must do nothing
    rather than something arbitrary."""
    clustering = t_channel_clustering(
        scale_scheme=ms.MLMClustering.ScaleScheme.clustering_mean,
        clustering_mean_pdf_ladder=False,
    )
    _, fact1, fact2, _, _ = run(clustering, t_channel_momenta)
    out = pdf_outputs(clustering, t_channel_momenta)
    assert np.all(out["pdf_rw_active"] == 0.0)
    assert out["pdf_scale1"] == pytest.approx(fact1)
    assert out["pdf_scale2"] == pytest.approx(fact2)


def write_flat_pdf_grid(path, value):
    """An lhagrid1 file with x f(x, Q) = value for every parton, x and Q."""
    xs = np.geomspace(1e-7, 1.0, 30)
    qs = np.geomspace(1.0, 1e5, 20)
    pids = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 21]
    with open(path, "w") as f:
        f.write("PdfType: central\nFormat: lhagrid1\n---\n")
        f.write(" ".join(f"{x:.12e}" for x in xs) + "\n")
        f.write(" ".join(f"{q:.12e}" for q in qs) + "\n")
        f.write(" ".join(str(pid) for pid in pids) + "\n")
        row = " ".join(f"{value:.6e}" for _ in pids) + "\n"
        f.write(row * (len(xs) * len(qs)))
        f.write("---\n")


def t_channel_integrand_outputs(
    tmp_path,
    pdf_value,
    batch_size=4000,
    pdf_reweighting=True,
    partial_weights=False,
    min_scale=0.0,
    **clustering_kwargs,
):
    """Every output of the t-channel process run through a whole Integrand
    under MLM (pdf reweighting by default), with the built-in flat test matrix
    element and a flat PDF of the given value. clustering_kwargs go to
    t_channel_clustering (scale_scheme madevent unless given); min_scale is the
    scale floor of the EnergyScale, beam.min_scale in a run."""
    context = ms.Context(device=ms.cpu_device(), thread_count=1)
    alphas_file = tmp_path / "alphas.info"
    alphas_file.write_text(
        "AlphaS_Qs: [1.0, 10.0, 100.0, 1000.0, 100000.0]\n"
        "AlphaS_Vals: [0.3, 0.2, 0.12, 0.1, 0.08]\n"
    )
    alphas_grid = ms.AlphaSGrid(str(alphas_file))
    alphas_grid.initialize_globals(context)
    pdf_file = tmp_path / "flat.dat"
    write_flat_pdf_grid(pdf_file, pdf_value)
    pdf_grid = ms.PdfGrid(str(pdf_file))
    pdf_grid.initialize_globals(context)

    diagrams = t_channel_diagram()
    mapping = ms.PhaseSpaceMapping(
        make_topologies(diagrams)[0],
        CM_ENERGY,
        permutations=diagrams[0]["permutations"],
    )
    # 0xBADCAFE: the built-in flat test matrix element, no library needed
    matrix_element = ms.MatrixElement(
        0xBADCAFE,
        len(T_CHANNEL_PDGS),
        ms.Integrand.matrix_element_inputs,
        ms.Integrand.matrix_element_outputs,
        1,
        True,
    )
    diff_xs = ms.DifferentialCrossSection(
        matrix_element=matrix_element,
        cm_energy=CM_ENERGY,
        running_coupling=None,
        energy_scale=ms.CachedScale(),
        pdf1=ms.CachedPdf(),
        pdf2=ms.CachedPdf(),
    )
    integrand = ms.Integrand(
        mapping,
        [diff_xs],
        pid_options=[T_CHANNEL_PDGS],
        pdf_grid=pdf_grid,
        running_coupling=ms.RunningCoupling(alphas_grid),
        energy_scale=ms.EnergyScale(
            t_channel_clustering(pdf_reweighting=pdf_reweighting, **clustering_kwargs),
            min_scale=min_scale,
        ),
        channel_indices=[0],
        partial_weights=partial_weights,
    )
    return ms.FunctionRuntime(integrand.function(), context)(
        np.array([batch_size], dtype=np.int64)
    )


def t_channel_integrand_weights(tmp_path, pdf_value, batch_size=4000, **kwargs):
    """Weights and momenta of t_channel_integrand_outputs."""
    out = t_channel_integrand_outputs(tmp_path, pdf_value, batch_size, **kwargs)
    return np.asarray(out[0]), np.asarray(out[1])


# The ladder under clustering_mean (clustering_mean_pdf_ladder, on by default)
# goes through the same integrand path. This process has a single flavour and no
# mirror, which is where a weight_after_cuts left out of the weight product
# would drop the ladder's veto without a trace. With no jet cut, the geometric
# mean mu_R of clustering_mean falls below the alpha_s grid for a few percent of
# the events, so that case runs with the 2 GeV floor of beam.min_scale, which
# drops them as a run does (without it their weight is 0 * NaN).
LADDER_SCHEMES = [
    pytest.param(
        {"scale_scheme": ms.MLMClustering.ScaleScheme.madevent}, id="madevent"
    ),
    pytest.param(
        {
            "scale_scheme": ms.MLMClustering.ScaleScheme.clustering_mean,
            "min_scale": 2.0,
        },
        id="clustering_mean_pdf_ladder",
    ),
]


@pytest.mark.parametrize("scheme", LADDER_SCHEMES)
def test_a_density_too_small_to_divide_by_drops_the_event(tmp_path, scheme):
    """reweight.f sets rewgt = 0 when the density it would divide by is under
    1e-10. With x f = 1e-12 everywhere every step of the ladder is such a
    division, so every event with a step on it must come out at zero.

    A smooth ramp in the denominator would leave a small nonzero weight."""
    weights, momenta = t_channel_integrand_weights(tmp_path, 1e-12, **scheme)
    clustering = {k: v for k, v in scheme.items() if k != "min_scale"}
    out = pdf_outputs(t_channel_clustering(pdf_reweighting=True, **clustering), momenta)
    assert not np.isnan(weights).any()
    stepped = (out["pdf_rw_active"] == 1.0).any(axis=1)
    assert stepped.any() and not stepped.all(), "both kinds needed"
    assert np.all(weights[stepped] == 0.0)
    # The other vetoes (alpha_s and the 2 GeV scale floor) still drop the odd
    # event of the rest, so only most of those survive.
    assert np.mean(weights[~stepped] > 0.0) > 0.9


def test_no_pdf_product_for_the_systematics_under_pdf_reweighting(tmp_path):
    """The systematics divide each PDF variation, taken at (k times) the
    factorisation scale, by the event's partial_weight_product. Under pdf
    reweighting that product is of the densities at pdf_scale, low on the
    clustering ladder, so even the central member would not give a ratio of 1.
    The integrand leaves it out there, and the systematics take the nominal
    densities at the factorisation scale themselves."""

    def output_count(**kwargs):
        return len(t_channel_integrand_outputs(tmp_path, 1.0, 16, **kwargs))

    # the product is there without pdf reweighting ...
    assert (
        output_count(pdf_reweighting=False, partial_weights=True)
        == output_count(pdf_reweighting=False) + 1
    )
    # ... and left out with it
    assert output_count(pdf_reweighting=True, partial_weights=True) == output_count(
        pdf_reweighting=True
    )
    # the same under clustering_mean: left out with the ladder, which is on by
    # default, and there without it
    mean = {"scale_scheme": ms.MLMClustering.ScaleScheme.clustering_mean}
    assert output_count(partial_weights=True, **mean) == output_count(**mean)
    no_ladder = dict(mean, clustering_mean_pdf_ladder=False)
    assert (
        output_count(partial_weights=True, **no_ladder) == output_count(**no_ladder) + 1
    )


# --------------------------------------------------------------------------
# the merging cut must not fire at a vertex that produced no radiation
# --------------------------------------------------------------------------


def w_plus_jet_diagram():
    """g u > e+ ve d, the diagram in which the jet pairs into the W.

    Vertices as mg7 builds them: the two beams give the t-channel quark p0, the
    leptons give the W p1, and then p0 and the jet give the W again - so once
    the W exists, (W, jet) is a valid clustering. Its measure is a kt between
    the W and the jet and has nothing to do with the jet's own transverse
    momentum, so a jet well above the merging cut can be clustered there at a
    scale well below it.

    That vertex is not a QCD splitting - the W carries no color - and madevent
    never applies the merging cut to a leg that was not emitted at a jet vertex
    (its iqjets gate). Reading the cut off any vertex would veto jets that are
    already above the cut.
    """
    return [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0, 0.0],
            # p0 = t-channel quark, p1 = W
            "propagators": [(0.0, 0.0, 1), (M_W, W_W, 24)],
            "vertices": [
                ["i1", "i0", "p0"],
                ["o0", "o1", "p1"],
                ["p0", "o2", "p1"],
            ],
            "permutations": [[0, 1, 2, 3, 4]],
        }
    ]


M_W, W_W = 80.419002, 2.0476
W_PLUS_JET_PDGS = [21, 2, -11, 12, 1]


def w_plus_jet_clustering(xqcut):
    diagrams = w_plus_jet_diagram()
    return ms.MLMClustering(
        [
            ms.Topology(
                ms.Diagram(
                    d["incoming_masses"],
                    d["outgoing_masses"],
                    [
                        ms.Propagator(mass=m, width=w, pdg_id=i)
                        for m, w, i in d["propagators"]
                    ],
                    d["vertices"],
                )
            )
            for d in diagrams
        ],
        [d["permutations"] for d in diagrams],
        make_diagram_indices(diagrams),
        cm_energy=CM_ENERGY,
        external_pdg_ids=W_PLUS_JET_PDGS,
        scale_scheme=ms.MLMClustering.ScaleScheme.madevent,
        xqcut=xqcut,
    )


def test_a_jet_above_the_merging_cut_is_never_vetoed():
    """The property the whole merging rests on: an event whose only jet is
    already above xqcut belongs to the matrix element, and the merging cut must
    leave it alone whatever the clustering history turns out to be."""
    xqcut = 60.0
    diagrams = w_plus_jet_diagram()
    momenta = sample_momenta(diagrams, batch_size=4000)
    clustering = w_plus_jet_clustering(xqcut)
    weight = run_all(clustering, momenta)[5]

    # mT of the one final-state parton, which is its pt here since it is
    # massless, and which is the measure of every clustering that can take it
    # back into a beam
    jet = momenta[:, 4, :]
    mt = np.sqrt(np.maximum(jet[:, 0] ** 2 - jet[:, 3] ** 2, 0.0))
    above = mt > xqcut * (1.0 + 1e-9)

    assert above.sum() > 200, "too few jets above the cut to prove anything"
    assert np.all(
        weight[above] == 1.0
    ), "%d of %d events with the jet above the merging cut were vetoed" % (
        int((weight[above] != 1.0).sum()),
        int(above.sum()),
    )


def test_a_jet_emitted_at_the_w_vertex_is_not_a_merging_jet():
    """The other half of the same rule, and the reason the test above is not
    vacuous: in this one diagram the jet is produced at the W vertex, so it is
    an electroweak emission and not radiation the shower would have made. Nothing
    is vetoed here at any merging cut - which is madevent's behaviour, since its
    iqjets is only ever set for a leg emitted at a jet vertex.

    That the cut does still remove soft QCD emissions is covered by
    test_xqcut_rejects_exactly_the_soft_jet_emissions and its neighbours, which
    run on a process whose jets do sit on QCD vertices.
    """
    diagrams = w_plus_jet_diagram()
    momenta = sample_momenta(diagrams, batch_size=4000)
    for xqcut in (60.0, 500.0):
        weight = run_all(w_plus_jet_clustering(xqcut), momenta)[5]
        assert np.all(
            weight == 1.0
        ), "xqcut = %g vetoed %d events whose jet is not a QCD emission" % (
            xqcut,
            int((weight != 1.0).sum()),
        )


# --------------------------------------------------------------------------
# mt2last: an s-channel QCD root takes the mean transverse mass of the pair
# --------------------------------------------------------------------------


def s_channel_ttbar_diagram(propagator):
    """u u~ > t t~ through one s-channel propagator: a gluon, which is the only
    diagram q q~ > t t~ has, or a Z as the colourless control.

    The last recorded clustering is then the final-state (t, t~) pair and the
    root is the only other step, so this is exactly the case setclscales'
    mt2last rule exists for."""
    return [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [M_TOP, M_TOP],
            "propagators": [propagator],
            "vertices": [["i0", "i1", "p0"], ["o0", "o1", "p0"]],
            "permutations": [[0, 1, 2, 3]],
        }
    ]


S_CHANNEL_TTBAR_PDGS = [2, -2, 6, -6]


def s_channel_ttbar_clustering(propagator):
    diagrams = s_channel_ttbar_diagram(propagator)
    return ms.MLMClustering(
        [
            ms.Topology(
                ms.Diagram(
                    d["incoming_masses"],
                    d["outgoing_masses"],
                    [
                        ms.Propagator(mass=m, width=w, pdg_id=i)
                        for m, w, i in d["propagators"]
                    ],
                    d["vertices"],
                )
            )
            for d in diagrams
        ],
        [d["permutations"] for d in diagrams],
        make_diagram_indices(diagrams),
        cm_energy=CM_ENERGY,
        external_pdg_ids=S_CHANNEL_TTBAR_PDGS,
        scale_scheme=ms.MLMClustering.ScaleScheme.madevent,
    )


def transverse_mass(p):
    return np.sqrt(np.maximum(p[..., 0] ** 2 - p[..., 3] ** 2, 0.0))


def test_an_s_channel_qcd_root_uses_the_mean_transverse_mass_of_the_pair():
    """q q~ > g > t t~ is reported at sqrt(mT(t) mT(t~)), madevent's mt2last,
    for both scales, rather than at the mT of the whole final state, sqrt(shat)."""
    gluon = (0.0, 0.0, 21)
    momenta = sample_momenta(s_channel_ttbar_diagram(gluon), batch_size=500)
    ren, fac1, fac2, _, _ = run(s_channel_ttbar_clustering(gluon), momenta)

    expected = np.sqrt(transverse_mass(momenta[:, 2]) * transverse_mass(momenta[:, 3]))
    shat_mt = transverse_mass(momenta[:, 2] + momenta[:, 3])
    assert ren == pytest.approx(expected, rel=1e-9)
    assert fac1 == pytest.approx(expected, rel=1e-9)
    assert fac2 == pytest.approx(expected, rel=1e-9)
    # and it is a different answer from the one the root would otherwise give
    assert np.all(expected <= 0.5 * shat_mt * (1.0 + 1e-9))


def test_a_colourless_s_channel_root_keeps_the_transverse_mass_of_the_final_state():
    """The rule needs the last clustering to be a QCD one. Through a Z the pair
    joins into a colourless line, so the root keeps mT of the final state - the
    Drell-Yan-like answer, and the reason W and Z + 0 jets never needed it."""
    z = (M_Z, W_Z, 23)
    momenta = sample_momenta(s_channel_ttbar_diagram(z), batch_size=500)
    ren, fac1, fac2, _, _ = run(s_channel_ttbar_clustering(z), momenta)

    shat_mt = transverse_mass(momenta[:, 2] + momenta[:, 3])
    assert ren == pytest.approx(shat_mt, rel=1e-9)
    assert fac1 == pytest.approx(shat_mt, rel=1e-9)
    assert fac2 == pytest.approx(shat_mt, rel=1e-9)


# --------------------------------------------------------------------------
# t-channel propagator flavours along the chain
# --------------------------------------------------------------------------


def gg_ttbar_g_chain_diagram():
    """g g > t t~ g through a t-channel chain whose two propagators differ:

        beam 0 --emits the jet--> gluon --emits t--> top --meets beam 1--> t~

    A chain with a single propagator reads the same from both ends, so only a
    chain like this one can tell whether each end of it got its own flavour and
    mass. Legs: (g, g, t, t~, g)."""
    return [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [M_TOP, M_TOP, 0.0],
            "propagators": [(0.0, 0.0, 21), (M_TOP, 0.0, 6)],
            "vertices": [["i0", "o2", "p0"], ["p0", "o0", "p1"], ["p1", "o1", "i1"]],
            "permutations": [[0, 1, 2, 3, 4]],
        }
    ]


GG_TTBAR_G_PDGS = [21, 21, 6, -6, 21]


def gg_ttbar_g_chain_clustering():
    diagrams = gg_ttbar_g_chain_diagram()
    return ms.MLMClustering(
        [
            ms.Topology(
                ms.Diagram(
                    d["incoming_masses"],
                    d["outgoing_masses"],
                    [
                        ms.Propagator(mass=m, width=w, pdg_id=i)
                        for m, w, i in d["propagators"]
                    ],
                    d["vertices"],
                )
            )
            for d in diagrams
        ],
        [d["permutations"] for d in diagrams],
        make_diagram_indices(diagrams),
        cm_energy=CM_ENERGY,
        external_pdg_ids=GG_TTBAR_G_PDGS,
        scale_scheme=ms.MLMClustering.ScaleScheme.madevent,
    )


def first_initial_state_transitions(clustering, n_ext):
    """{(beam, leg): (massive_in, is_jet_in)} for the clusterings the walk can
    take first, which are the ones whose mother is the propagator next to a
    beam."""
    machine = np.asarray(clustering.cluster_state_machine)
    out = {}
    for data, _, trace in decode_transitions(machine, 0):
        p1, p2 = field(data, BIT_PARTICLE1), field(data, BIT_PARTICLE2)
        if p1 < 2:
            out[(p1, p2)] = (
                field(data, BIT_MASSIVE_IN),
                int(bool(trace & TRACE_IS_JET_IN)),
            )
    return out


def test_each_end_of_a_t_channel_chain_gets_its_own_flavour():
    """Beam 0 emitting the jet leaves the gluon propagator, massless and a jet;
    beam 1 taking back the t~ leaves the top propagator, massive and not a jet.
    With the two ends swapped the gluon would be read as the top, the beam's
    parton line would stop at the jet, and mu_R would pick up the jet's own
    scale."""
    first = first_initial_state_transitions(gg_ttbar_g_chain_clustering(), 5)
    assert first[(0, 4)] == (0, 1), "beam 0 + jet should leave a massless jet line"
    assert first[(1, 3)] == (1, 0), "beam 1 + t~ should leave a massive, non-jet line"


def test_a_jet_emitted_before_the_tops_does_not_set_the_renormalisation_scale():
    """setclscales carries a beam's parton line on past a jet emission to the
    vertex where the top attaches, so when the jet is clustered into a beam and
    the tops make up the rest, all four scales in the mu_R average are central
    ones and mu_R equals mu_F. The jet's own scale entering mu_R would break
    that. It also needs the root measured where cluster.f measures it: not
    boosted after the last clustering, and in the current frame after an
    initial-state one; measured in the lab, mu_R would sit below mu_F."""
    clustering = gg_ttbar_g_chain_clustering()
    momenta = sample_momenta(gg_ttbar_g_chain_diagram(), batch_size=4000)
    ren, fac1, fac2, outgoing, _ = run(clustering, momenta)
    jet = momenta[:, 4, :]
    jet_pt = np.sqrt(jet[:, 1] ** 2 + jet[:, 2] ** 2)
    # the jet went into a beam: its clustering scale is its transverse momentum
    jet_into_beam = np.abs(outgoing[:, 2] / jet_pt - 1.0) < 1e-9
    assert jet_into_beam.sum() > 500, "too few jet-into-beam histories to test"
    # the root is measured in the frame the boost after the jet left behind,
    # where t and t~ balance, so the equality holds up to rounding
    same = np.abs(ren / fac1 - 1.0) < 1e-6
    assert (
        np.mean(same[jet_into_beam]) > 0.5
    ), "mu_R == mu_F on only %.1f%% of jet-into-beam points" % (
        100 * np.mean(same[jet_into_beam])
    )


# --------------------------------------------------------------------------
# clustering_measure: FxFx (NLO cluster_scale) against madevent's LO DJ
# --------------------------------------------------------------------------


def w_plus_jet_two_diagrams():
    """g u > e+ ve d with both ways the jet can be clustered:

      s-channel: g u > u*, u* > d W - the jet pairs with the W (massless mother,
                 massive and massless daughters, the case the measures disagree
                 on)
      t-channel: u emits the W and becomes d*, d* g > d - the jet goes into
                 beam 0

    madevent scores the (W, d) pair by the d's transverse mass times 1 + 1e-6,
    which never beats taking the d into a beam; the FxFx measure scores it by
    sqrt(|p_d.(p_d+p_W)|)/2, which often does."""
    return [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0, 0.0],
            "propagators": [(0.0, 0.0, 2), (M_W, W_W, 24)],
            "vertices": [["i1", "i0", "p0"], ["o0", "o1", "p1"], ["p0", "o2", "p1"]],
            "permutations": [[0, 1, 2, 3, 4]],
        },
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0, 0.0],
            "propagators": [(0.0, 0.0, 1), (M_W, W_W, 24)],
            "vertices": [["o0", "o1", "p1"], ["i1", "p1", "p0"], ["p0", "i0", "o2"]],
            "permutations": [[0, 1, 2, 3, 4]],
        },
    ]


def w_plus_jet_two_diagram_clustering(**kwargs):
    diagrams = w_plus_jet_two_diagrams()
    return ms.MLMClustering(
        [
            ms.Topology(
                ms.Diagram(
                    d["incoming_masses"],
                    d["outgoing_masses"],
                    [
                        ms.Propagator(mass=m, width=w, pdg_id=i)
                        for m, w, i in d["propagators"]
                    ],
                    d["vertices"],
                )
            )
            for d in diagrams
        ],
        [d["permutations"] for d in diagrams],
        make_diagram_indices(diagrams),
        cm_energy=CM_ENERGY,
        external_pdg_ids=W_PLUS_JET_PDGS,
        scale_scheme=ms.MLMClustering.ScaleScheme.madevent,
        **kwargs,
    )


def jet_into_beam_fraction(clustering, momenta):
    outgoing = run(clustering, momenta)[3]
    jet = momenta[:, 4, :]
    jet_pt = np.sqrt(jet[:, 1] ** 2 + jet[:, 2] ** 2)
    return np.mean(np.abs(outgoing[:, 2] / jet_pt - 1.0) < 1e-5)


def test_fxfx_is_the_default_clustering_measure():
    clustering = w_plus_jet_two_diagram_clustering()
    assert clustering.clustering_measure == ms.MLMClustering.ClusteringMeasure.fxfx


def test_the_madevent_measure_takes_the_jet_into_the_beam():
    """Under madevent's measure the jet never pairs with the W: its scale is its
    transverse momentum on every point, as in madevent (99% there, the rest
    being histories without this choice). Half of these jets go against the
    beam, where only the LO ordering of the 1 + 1e-6 factors and madevent's
    keep-the-first tie-break decide it."""
    momenta = sample_momenta(w_plus_jet_two_diagrams(), batch_size=4000)
    clustering = w_plus_jet_two_diagram_clustering(
        clustering_measure=ms.MLMClustering.ClusteringMeasure.madevent
    )
    assert jet_into_beam_fraction(clustering, momenta) == 1.0


def test_the_fxfx_measure_lets_the_jet_pair_with_the_w():
    """The FxFx score of the (W, d) pair is often below the d's pt, so some jets
    pair with the W instead."""
    momenta = sample_momenta(w_plus_jet_two_diagrams(), batch_size=4000)
    fxfx = jet_into_beam_fraction(
        w_plus_jet_two_diagram_clustering(
            clustering_measure=ms.MLMClustering.ClusteringMeasure.fxfx
        ),
        momenta,
    )
    assert fxfx < 0.95, "the FxFx measure should pair a visible fraction with the W"


# --------------------------------------------------------------------------
# clustering_history: over every diagram, or along one diagram
# --------------------------------------------------------------------------

HISTORY = ms.MLMClustering.ClusteringHistory


def history_clustering(diagrams, pdg_ids, history, **kwargs):
    kwargs.setdefault("scale_scheme", ms.MLMClustering.ScaleScheme.madevent)
    return make_clustering(
        diagrams, external_pdg_ids=pdg_ids, clustering_history=history, **kwargs
    )


def along(clustering, momenta, diagram):
    diagram = np.broadcast_to(np.asarray(diagram, dtype=np.int32), len(momenta))
    return [
        np.asarray(v)
        for v in ms.MLMClusteringAlongDiagram(clustering)(
            momenta, np.ascontiguousarray(diagram)
        )
    ]


def test_all_diagrams_is_the_default_history(process):
    _, diagrams, pdg_ids = process
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    assert clustering.clustering_history == HISTORY.all_diagrams
    assert len(clustering.diagram_start_states) == 0


def test_every_diagram_gets_a_state_machine_after_the_union_one(process):
    _, diagrams, pdg_ids = process
    union = history_clustering(diagrams, pdg_ids, HISTORY.all_diagrams)
    per_diagram = history_clustering(diagrams, pdg_ids, HISTORY.diagram)
    machine = per_diagram.cluster_state_machine
    union_size = len(union.cluster_state_machine)
    assert machine[:union_size] == union.cluster_state_machine
    starts = np.asarray(per_diagram.diagram_start_states)
    assert len(starts) == sum(len(d["permutations"]) for d in diagrams)
    assert np.all(starts >= union_size)
    assert np.all(starts < len(machine))
    assert len(set(starts)) == len(starts)


def test_the_history_setting_leaves_the_union_walk_alone(process):
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    reference = run_all(
        history_clustering(diagrams, pdg_ids, HISTORY.all_diagrams), momenta
    )
    for history in (HISTORY.diagram, HISTORY.madevent):
        other = run_all(history_clustering(diagrams, pdg_ids, history), momenta)
        for a, b in zip(reference, other):
            np.testing.assert_array_equal(a, b)


def test_the_union_walk_cannot_be_walked_along_a_diagram(process):
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=4)
    clustering = history_clustering(diagrams, pdg_ids, HISTORY.all_diagrams)
    with pytest.raises(Exception):
        along(clustering, momenta, 0)


def test_a_history_along_a_diagram_reports_that_diagram(process):
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    clustering = history_clustering(diagrams, pdg_ids, HISTORY.diagram, xqcut=20.0)
    count = len(clustering.diagram_start_states)
    picked = np.random.default_rng(3).integers(0, count, len(momenta))
    out = along(clustering, momenta, picked)
    np.testing.assert_array_equal(out[4], picked)
    for value in out:
        assert np.all(np.isfinite(value))


def test_a_single_diagram_clusters_along_itself_as_its_union_does(process):
    """A process made of one diagram has nothing to merge, so the history over
    every diagram and the one along that diagram are the same walk."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    single = [dict(diagrams[0], permutations=diagrams[0]["permutations"][:1])]
    union = run_all(history_clustering(single, pdg_ids, HISTORY.all_diagrams), momenta)
    full = history_clustering(diagrams, pdg_ids, HISTORY.diagram)
    out = along(full, momenta, 0)
    for a, b in zip(union, out):
        np.testing.assert_array_equal(a, b)


def test_the_madevent_history_is_either_the_union_or_the_diagram(process):
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    count = sum(len(d["permutations"]) for d in diagrams)
    picked = np.random.default_rng(4).integers(0, count, len(momenta))
    union = run_all(history_clustering(diagrams, pdg_ids, HISTORY.madevent), momenta)
    diagram = along(
        history_clustering(diagrams, pdg_ids, HISTORY.diagram), momenta, picked
    )
    madevent = along(
        history_clustering(diagrams, pdg_ids, HISTORY.madevent), momenta, picked
    )
    is_union = np.all(
        [
            np.isclose(a, b).reshape(len(momenta), -1).all(axis=1)
            for a, b in zip(union, madevent)
        ],
        axis=0,
    )
    is_diagram = np.all(
        [
            np.isclose(a, b).reshape(len(momenta), -1).all(axis=1)
            for a, b in zip(diagram[:6], madevent[:6])
        ],
        axis=0,
    )
    assert np.all(is_union | is_diagram)
    # both outcomes occur for a process with this many diagrams
    assert 0 < np.mean(is_union) < 1


def test_a_diagram_history_uses_that_diagrams_own_line_flavours():
    """g u > e+ ve d. Along the t-channel diagram the jet goes into beam 0 and
    its clustering is a QCD vertex that alpha_s is reweighted at. Along the
    s-channel one (u* > d W) the only QCD vertex left is g u > u*, which the
    union can also pick; the two diagrams must never hand back each other's
    history."""
    diagrams = w_plus_jet_two_diagrams()
    momenta = sample_momenta(diagrams, batch_size=2000)
    clustering = w_plus_jet_two_diagram_clustering(
        clustering_history=HISTORY.diagram,
        clustering_measure=ms.MLMClustering.ClusteringMeasure.madevent,
    )
    s_channel = along(clustering, momenta, 0)
    t_channel = along(clustering, momenta, 1)
    jet = momenta[:, 4, :]
    jet_pt = np.sqrt(jet[:, 1] ** 2 + jet[:, 2] ** 2)
    # the t-channel history always takes the jet into a beam at its pt
    assert np.mean(np.abs(t_channel[3][:, 2] / jet_pt - 1.0) < 1e-5) == 1.0
    # the s-channel one never clusters the jet with beam 0 before the W
    assert np.mean(np.abs(s_channel[3][:, 2] / jet_pt - 1.0) < 1e-5) < 0.5


# --------------------------------------------------------------------------
# jet scales of legs that are not merging jets, under scale_scheme = madevent
# --------------------------------------------------------------------------


def vbf_like_fsr_diagram():
    """u d > u d g with a colourless t-channel Z between the two quark lines
    and the gluon radiated off the outgoing u:

      u* > u g      (final state, a jet vertex)
      u Z > u*      (beam 0 line, through the Z)
      Z d > d       (beam 1 line)

    Neither quark is a merging jet in madevent (the Z stops both beam lines
    being parton lines), the gluon is."""
    return [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0, 0.0],
            "propagators": [(0.0, 0.0, 2), (M_Z, 0.0, 23)],
            "vertices": [["o0", "o2", "p0"], ["i0", "p1", "p0"], ["p1", "i1", "o1"]],
            "permutations": [[0, 1, 2, 3, 4]],
        },
    ]


def vbf_like_fsr_clustering(**kwargs):
    diagrams = vbf_like_fsr_diagram()
    return ms.MLMClustering(
        [
            ms.Topology(
                ms.Diagram(
                    d["incoming_masses"],
                    d["outgoing_masses"],
                    [
                        ms.Propagator(mass=m, width=w, pdg_id=i)
                        for m, w, i in d["propagators"]
                    ],
                    d["vertices"],
                )
            )
            for d in diagrams
        ],
        [d["permutations"] for d in diagrams],
        make_diagram_indices(diagrams),
        cm_energy=CM_ENERGY,
        external_pdg_ids=[2, 1, 2, 1, 21],
        scale_scheme=ms.MLMClustering.ScaleScheme.madevent,
        **kwargs,
    )


def fsr_jet_scales(**kwargs):
    momenta = sample_momenta(vbf_like_fsr_diagram(), batch_size=500)
    return run(vbf_like_fsr_clustering(**kwargs), momenta)[3]


def test_a_quark_that_is_not_a_merging_jet_is_written_at_the_collider_energy():
    """madevent writes ptclus = sqrt(s) for every leg iqjets does not call a
    merging jet, even one that radiated at a QCD vertex: the shower's MLM
    matching must leave it alone. The gluon keeps the scale of the vertex it
    was emitted at. (The kt measure of these unconstrained points can itself
    exceed sqrt(s), so the scales are compared to each other, not to it.)"""
    production = fsr_jet_scales()
    emission = fsr_jet_scales(jet_scale_scheme=ms.MLMClustering.JetScaleScheme.emission)
    np.testing.assert_array_equal(production[:, 0], CM_ENERGY)
    np.testing.assert_array_equal(production[:, 1], CM_ENERGY)
    np.testing.assert_array_equal(production[:, 2], emission[:, 2])
    assert np.all(production[:, 2] != CM_ENERGY)


def test_the_emission_scheme_still_books_the_radiating_quark():
    """The emission scheme books the (u, g) vertex onto both of its legs,
    unlike madevent's."""
    emission = fsr_jet_scales(jet_scale_scheme=ms.MLMClustering.JetScaleScheme.emission)
    np.testing.assert_array_equal(emission[:, 0], emission[:, 2])
    np.testing.assert_array_equal(emission[:, 1], CM_ENERGY)


# --------------------------------------------------------------------------
# clusterings that only exist for some flavours of a merged subprocess
# --------------------------------------------------------------------------

TRACE_SAME_FLAVOR = 1 << 16

# q q~ > q q~ with a merged quark, as clean_pids in launch.py represents it
QQ_TO_QQ_PDGS = [1, -1, 1, -1]
GLUON = [0.0, 0.0, 0, 0.0, 0.0, 21]


def qq_to_qq_diagrams():
    """The s-channel gluon (diagram 0: legs 2 and 3) and the t-channel gluon
    (diagram 1: legs 0 and 2, 1 and 3) of q q~ > q q~. For a q q~ > q' q~'
    flavour only the first exists, for q q~' > q q~' only the second."""
    common = {
        "incoming_masses": [0.0, 0.0],
        "outgoing_masses": [0.0, 0.0],
        "propagators": [GLUON],
        "permutations": [[0, 1, 2, 3]],
    }
    return [
        dict(common, vertices=[["o0", "o1", "p0"], ["i0", "i1", "p0"]]),
        dict(common, vertices=[["i0", "o0", "p0"], ["p0", "o1", "i1"]]),
    ]


def with_flavors(clustering, leg_flavors, momenta, flavor_index):
    index = np.full(len(momenta), flavor_index, dtype=np.int32)
    return [
        np.asarray(v)
        for v in ms.MLMClusteringWithFlavors(clustering, leg_flavors)(momenta, index)
    ]


def test_flavour_diagonal_clusterings_of_fermions_are_marked():
    """Every clustering of q q~ > q q~ joins two quarks through a gluon, so all
    of them depend on the flavours of the event."""
    clustering = make_clustering(qq_to_qq_diagrams(), external_pdg_ids=QQ_TO_QQ_PDGS)
    non_terminal, _ = walk(np.asarray(clustering.cluster_state_machine), 4)
    marked, unmarked = set(), set()
    for _, (_, transitions) in non_terminal.items():
        for data, _, trace in transitions:
            pair = (field(data, BIT_PARTICLE1), field(data, BIT_PARTICLE2))
            (marked if trace & TRACE_SAME_FLAVOR else unmarked).add(pair)
    assert marked == {(0, 2), (1, 3), (2, 3)}
    assert unmarked == set()


def test_clustering_follows_the_diagrams_the_flavours_allow():
    """u u~ > d d~ only has the s-channel gluon and u d~ > u d~ only the
    t-channel one, whichever of the two the kt measure prefers; a flavour with
    every leg alike, or no flavour information, keeps the union.

    For a 2 -> 2 the initial-state measure (the pt of the jet) always beats the
    final-state one of the back-to-back pair, so the union always takes the
    t-channel gluon: the q-q' clustering a u u~ > d d~ event does not have."""
    diagrams = qq_to_qq_diagrams()
    clustering = make_clustering(diagrams, external_pdg_ids=QQ_TO_QQ_PDGS)
    momenta = sample_momenta(diagrams)
    leg_flavors = [1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1]
    union = run(clustering, momenta)
    assert np.all(union[4] == 1)

    same = with_flavors(clustering, leg_flavors, momenta, 0)
    s_channel = with_flavors(clustering, leg_flavors, momenta, 1)
    t_channel = with_flavors(clustering, leg_flavors, momenta, 2)
    unknown = with_flavors(clustering, [0, 0, 0, 0], momenta, 0)

    assert np.all(s_channel[4] == 0)
    assert np.all(t_channel[4] == 1)
    for other in (same, unknown):
        for a, b in zip(union, other[:5]):
            np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------
# 2 -> 1: the 0-jet sample of p p > z + jets
# --------------------------------------------------------------------------


def drell_yan_two_to_one_clustering(**kwargs):
    """u u~ > z: nothing to cluster. cluster.f handles it as its "special
    2 -> 1 case", pt2ijcl = the invariant mass squared of the z, and
    setclscales sets both scales from that."""
    kwargs.setdefault("cm_energy", CM_ENERGY)
    kwargs.setdefault("external_pdg_ids", [2, -2, 23])
    return ms.MLMClustering(
        [ms.Topology(ms.Diagram([0.0, 0.0], [M_Z], [], [["i0", "i1", "o0"]]))],
        [[[0, 1, 2]]],
        [[0]],
        **kwargs,
    )


def two_to_one_momenta(batch_size=BATCH_SIZE, seed=SEED):
    """Unequal beam energies, so that the event is not at rest in the lab."""
    rng = np.random.default_rng(seed)
    e1 = rng.uniform(20.0, 500.0, batch_size)
    e2 = rng.uniform(20.0, 500.0, batch_size)
    p = np.zeros((batch_size, 3, 4))
    p[:, 0] = np.stack([e1, 0 * e1, 0 * e1, e1], axis=1)
    p[:, 1] = np.stack([e2, 0 * e2, 0 * e2, -e2], axis=1)
    p[:, 2] = p[:, 0] + p[:, 1]
    return p, np.sqrt(4.0 * e1 * e2)


@pytest.mark.parametrize(
    "scale_scheme",
    [
        ms.MLMClustering.ScaleScheme.clustering_mean,
        ms.MLMClustering.ScaleScheme.madevent,
    ],
)
@pytest.mark.parametrize(
    "history",
    list(ms.MLMClustering.ClusteringHistory.__members__.values()),
)
def test_a_two_to_one_process_takes_its_scales_from_shat(scale_scheme, history):
    """A 2 -> 1 process has no clustering to find and must not raise "MLM
    clustering found no valid clustering for this process", which would kill
    the 0-jet sample of a p p > z @0 + jets run."""
    clustering = drell_yan_two_to_one_clustering(
        scale_scheme=scale_scheme, clustering_history=history, xqcut=20.0
    )
    momenta, shat_root = two_to_one_momenta()
    ren, fact1, fact2, outgoing, diagram, xqcut_weight = run_all(clustering, momenta)
    np.testing.assert_allclose(ren, shat_root, rtol=1e-12)
    np.testing.assert_allclose(fact1, shat_root, rtol=1e-12)
    np.testing.assert_allclose(fact2, shat_root, rtol=1e-12)
    # the z is no jet: written at the collider energy, where no veto trips
    assert np.all(outgoing == CM_ENERGY)
    assert np.all(diagram == 0)
    # nothing was emitted, so nothing can fail the merging cut
    assert np.all(xqcut_weight == 1.0)
