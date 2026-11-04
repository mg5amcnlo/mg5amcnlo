"""Scale/PDF systematics computed at combine time (SystematicsCalculator).

The tests need an LHAPDF data directory holding NNPDF40MC_lo_as_01180
(LHAPDF_DATA_PATH, or the paths of the lhapdf module). The calculator evaluates
PDFs and alpha_s with the regular batched madspace functions; the reference
values here come from the same functions on the default context, and are
checked against LHAPDF when the python module is available.
"""

import json
import os
import re

import numpy as np
import pytest
from pytest import approx

import madspace as ms

PDF_SET = "NNPDF40MC_lo_as_01180"


def _pdf_dir():
    paths = []
    if os.environ.get("LHAPDF_DATA_PATH"):
        paths += os.environ["LHAPDF_DATA_PATH"].split(os.pathsep)
    try:
        import lhapdf

        lhapdf.setVerbosity(0)
        paths += list(lhapdf.paths())
    except ImportError:
        pass
    for path in paths:
        if os.path.isfile(os.path.join(path, PDF_SET, f"{PDF_SET}_0000.dat")):
            return path
    return None


PDF_DIR = _pdf_dir()
# host-side: LHAPDF, the systematics calculator and explicit CPU contexts
pytestmark = [
    pytest.mark.skipif(
        PDF_DIR is None, reason=f"pdf set {PDF_SET} required to run this test"
    ),
    pytest.mark.cpu_only,
]


def grid_files(set_name=PDF_SET, member=0):
    return (
        os.path.join(PDF_DIR, set_name, f"{set_name}_{member:04d}.dat"),
        os.path.join(PDF_DIR, set_name, f"{set_name}.info"),
    )


class PdfRef:
    """Reference PDF / alpha_s values through the regular batched madspace
    functions (PartonDensity / RunningCoupling on the default context)."""

    def __init__(self, grid, alpha_s, pids=(21, 2, 1, -1, -2, 3, 4, 5), prefix="ref"):
        self.grid, self.alpha_s, self.pids = grid, alpha_s, list(pids)
        ctx = ms.default_context()
        grid.initialize_globals(ctx, prefix)
        alpha_s.initialize_globals(ctx, prefix)
        self._pdf = ms.PartonDensity(grid, self.pids, False, prefix)
        self._coupling = ms.RunningCoupling(alpha_s, prefix)

    def pdf(self, pid, x, q):
        """x f(x, q) for arrays (or scalars) x, q."""
        x, q = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(q, float))
        values = np.asarray(self._pdf(x, q))[:, self.pids.index(pid)]
        return values if values.size > 1 else float(values[0])

    def alphas(self, q):
        q = np.atleast_1d(np.asarray(q, float))
        values = np.asarray(self._coupling(q))
        return values if values.size > 1 else float(values[0])


@pytest.fixture(scope="module")
def grids():
    grid_file, info_file = grid_files()
    return ms.PdfGrid(grid_file), ms.AlphaSGrid(info_file)


@pytest.fixture(scope="module")
def ref(grids):
    return PdfRef(*grids)


def sample_points(grid, n, seed=1):
    rng = np.random.default_rng(seed)
    x = 10 ** rng.uniform(np.log10(grid.x[0]) + 1e-6, 0, n)
    q = 10 ** rng.uniform(np.log10(grid.q[0]) + 1e-6, np.log10(grid.q[-1]) - 1e-6, n)
    return x, q


def test_pdf_matches_lhapdf(grids, ref):
    """The batched PDF / alpha_s functions the calculator uses agree with LHAPDF
    where the python module exists."""
    lhapdf = pytest.importorskip("lhapdf")
    lhapdf.setVerbosity(0)
    try:
        lha = lhapdf.mkPDF(PDF_SET, 0)
    except RuntimeError:
        pytest.skip("lhapdf cannot load %s" % PDF_SET)
    grid, alpha_s = grids
    x, q = sample_points(grid, 200, seed=3)
    for pid in [21, 2, -1, 5]:
        values = ref.pdf(pid, x, q)
        for xi, qi, v in zip(x, q, values):
            assert v == approx(lha.xfxQ(pid, xi, qi), rel=1e-6, abs=1e-9)
    for qi, a in zip(q, ref.alphas(q)):
        assert a == approx(lha.alphasQ(qi), rel=1e-6)


def make_config(**kwargs):
    grid_file, info_file = grid_files()
    config = ms.SystematicsConfig()
    config.mur = [0.5, 1.0, 2.0]
    config.muf = [0.5, 1.0, 2.0]
    config.nominal_set_name = PDF_SET
    config.nominal_lhaid = 338500
    config.nominal_error_type = "replicas"
    config.nominal_description = "test set <x> => y"
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def make_events(ref, n=50, seed=4):
    """Random u u > u u like events with the columns the combined event buffer
    carries; the PDF product is the nominal one, as the integrand stores it."""
    rng = np.random.default_rng(seed)
    x1 = 10 ** rng.uniform(-3, -0.3, n)
    x2 = 10 ** rng.uniform(-3, -0.3, n)
    scale = 10 ** rng.uniform(1.5, 3, n)
    weight = rng.uniform(0.5, 1.5, n)
    product = ref.pdf(2, x1, scale) * ref.pdf(2, x2, scale)
    return dict(
        event_weight=list(weight),
        subprocess_index=[0] * n,
        flavor_index=[0] * n,
        ren_scale=list(scale),
        x1=list(x1),
        fact_scale1=list(scale),
        x2=list(x2),
        fact_scale2=list(scale),
        partial_weight_product=list(product),
        alpha_qcd=list(ref.alphas(scale)),
    )


def test_scale_variations(grids, ref):
    """The LO scale reweighting formula, checked against a direct evaluation."""
    grid, alpha_s = grids
    config = make_config()
    args = [ms.SubprocessSystArgs(qcd_power=2, beam_pdgs=[[2, 2]])]
    calc = ms.SystematicsCalculator(config, args, grid, alpha_s)
    variations = calc.variations
    assert len(variations) == 8
    assert [(v.mur, v.muf) for v in variations] == [
        (0.5, 0.5),
        (0.5, 1.0),
        (0.5, 2.0),
        (1.0, 0.5),
        (1.0, 2.0),
        (2.0, 0.5),
        (2.0, 1.0),
        (2.0, 2.0),
    ]
    assert calc.weight_ids == list(range(1, 9))
    events = make_events(ref)
    weights = calc.weights(**events)
    for i in range(len(events["event_weight"])):
        w0 = events["event_weight"][i]
        mu = events["ren_scale"][i]
        x1, x2 = events["x1"][i], events["x2"][i]
        for var, w in zip(variations, weights[i]):
            r_alpha = (ref.alphas(var.mur * mu) / ref.alphas(mu)) ** 2
            r_pdf = (ref.pdf(2, x1, var.muf * mu) * ref.pdf(2, x2, var.muf * mu)) / (
                ref.pdf(2, x1, mu) * ref.pdf(2, x2, mu)
            )
            assert w == approx(w0 * r_alpha * r_pdf, rel=1e-12)


def test_identities(grids, ref):
    """Trivial variations give exactly the nominal weight."""
    grid, alpha_s = grids
    events = make_events(ref, n=20)
    # no QCD coupling: mu_R variations are the identity
    config = make_config(mur=[0.5, 1.0, 2.0], muf=[1.0])
    calc = ms.SystematicsCalculator(
        config, [ms.SubprocessSystArgs(qcd_power=0, beam_pdgs=[[2, 2]])], grid, alpha_s
    )
    for w0, row in zip(events["event_weight"], calc.weights(**events)):
        assert row == [w0, w0]
    # leptonic beams: mu_F variations are dropped, mu_R ones remain
    config = make_config(has_pdf=False, pdf_members=[])
    calc = ms.SystematicsCalculator(
        config,
        [ms.SubprocessSystArgs(qcd_power=2, beam_pdgs=[[11, -11]])],
        None,
        alpha_s,
    )
    assert [(v.mur, v.muf) for v in calc.variations] == [(0.5, 1.0), (2.0, 1.0)]


def test_non_uniform_qcd_power_drops_mur(grids):
    grid, alpha_s = grids
    config = make_config()
    calc = ms.SystematicsCalculator(
        config,
        [ms.SubprocessSystArgs(qcd_power=-1, beam_pdgs=[[2, -2]])],
        grid,
        alpha_s,
    )
    assert [(v.mur, v.muf) for v in calc.variations] == [(1.0, 0.5), (1.0, 2.0)]
    assert any("renormalisation scale" in w for w in calc.warnings)


def test_pdf_member_and_header(grids, ref):
    """A PDF member variation (here the nominal member as a fake second set)
    reweights by the PDF ratio, and the <initrwgt> text follows the
    systematics.py conventions."""
    grid, alpha_s = grids
    grid_file, info_file = grid_files()
    other = ms.PdfMemberSpec(
        set_name="OtherSet",
        set_lhaid=900000,
        member=1,
        grid_file=grid_file,
        info_file=info_file,
        error_type="hessian",
        description="a => b > c",
    )
    same = ms.PdfMemberSpec(
        set_name=PDF_SET,
        set_lhaid=338500,
        member=1,
        grid_file=grid_file,
        info_file=info_file,
        error_type="replicas",
        description="",
    )
    config = make_config(mur=[1.0], muf=[1.0], pdf_members=[same, other])
    calc = ms.SystematicsCalculator(
        config, [ms.SubprocessSystArgs(qcd_power=2, beam_pdgs=[[2, 2]])], grid, alpha_s
    )
    # nominal-set member requested -> the group starts with the nominal member
    assert [(v.pdf_index) for v in calc.variations] == [-1, 0, 1]
    events = make_events(ref, n=10)
    for w0, row in zip(events["event_weight"], calc.weights(**events)):
        # same grid file everywhere: every ratio is exactly one
        assert row == approx([w0, w0, w0], rel=1e-12)
    header = calc.initrwgt()
    assert '<weightgroup name="%s" combine="replicas">' % PDF_SET in header
    assert (
        '<weightgroup name="OtherSet" combine="hessian"> # 900000: a ; b .gt. c'
        in header
    )
    assert '<weight id="1" MUR="1.0" MUF="1.0" PDF="338500" >  </weight>' in header
    assert (
        '<weight id="2" MUR="1.0" MUF="1.0" PDF="338501" > PDF=338500 MemberID=1 </weight>'
        in header
    )
    assert (
        '<weight id="3" MUR="1.0" MUF="1.0" PDF="900001" > PDF=900000 MemberID=1 </weight>'
        in header
    )
    assert header.count("<weightgroup") == header.count("</weightgroup>") == 2
    summary = json.loads(calc.summary())
    assert [v["id"] for v in summary["variations"]] == [1, 2, 3]
    assert summary["variations"][2]["pdf_set"] == "OtherSet"


def test_lhe_event_rwgt_format():
    event = ms.LHEEvent()
    event.particles = [ms.LHEParticle(pdg_id=21, status_code=-1, energy=1.0, pz=1.0)]
    event.rwgt_ids = [1, 2]
    event.rwgt = [1.5e3, -2.0]
    text = event.format()
    assert (
        "<rwgt>\n<wgt id='1'> +1.5000000e+03 </wgt>\n<wgt id='2'> -2.0000000e+00 </wgt>\n</rwgt>\n</event>"
        in text
    )


def test_config_json_roundtrip():
    grid_file, info_file = grid_files()
    config = make_config(write_inputs=True, together=False, alps=[0.5, 1.0, 2.0])
    config.pdf_members = [
        ms.PdfMemberSpec(PDF_SET, 338500, 1, grid_file, info_file, "replicas", "d")
    ]
    text = config.to_json()
    back = ms.SystematicsConfig.from_json(text)
    assert back.mur == config.mur and back.muf == config.muf
    assert back.alps == [0.5, 1.0, 2.0]
    assert back.write_inputs and not back.together
    assert back.pdf_members[0].grid_file == grid_file
    args = ms.SubprocessSystArgs(qcd_power=2, beam_pdgs=[[2, 2], [21, 2]])
    back_args = ms.SubprocessSystArgs.from_json(args.to_json())
    assert back_args.qcd_power == 2 and back_args.beam_pdgs == [[2, 2], [21, 2]]


def random_momenta(n, seed=5):
    """2 -> 2 massless momenta (E px py pz), incoming along z."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        e1, e2 = rng.uniform(50, 800, 2)
        p_in = [[e1, 0.0, 0.0, e1], [e2, 0.0, 0.0, -e2]]
        # back-to-back in the partonic frame, boosted along z
        s = 4 * e1 * e2
        pcm = np.sqrt(s) / 2
        theta, phi = np.arccos(rng.uniform(-1, 1)), rng.uniform(0, 2 * np.pi)
        px, py, pz = (
            pcm * np.sin(theta) * np.cos(phi),
            pcm * np.sin(theta) * np.sin(phi),
            pcm * np.cos(theta),
        )
        beta = (e1 - e2) / (e1 + e2)
        gamma = 1 / np.sqrt(1 - beta**2)

        def boost(e, pz):
            return gamma * (e + beta * pz), gamma * (pz + beta * e)

        ea, pza = boost(pcm, pz)
        eb, pzb = boost(pcm, -pz)
        out.append(p_in + [[ea, px, py, pza], [eb, -px, -py, pzb]])
    return out


def test_dynamical_scales(grids, ref):
    """Dynamical scale variations: the scales from the momenta match their
    definitions, and the weights follow the LO formula with the new scale."""
    grid, alpha_s = grids
    momenta = random_momenta(20)
    for p in momenta[:5]:
        p = np.array(p)
        fin = p[2:]
        et = sum(
            e * np.sqrt((px**2 + py**2) / (px**2 + py**2 + pz**2))
            for e, px, py, pz in fin
        )
        ht = sum(np.sqrt(max(0.0, e**2 - pz**2)) for e, px, py, pz in fin)
        shat = np.sqrt((p[0, 0] + p[1, 0]) ** 2 - (p[0, 3] + p[1, 3]) ** 2)
        assert ms.SystematicsCalculator.dynamical_scale(1, p.tolist()) == approx(
            et, rel=1e-12
        )
        assert ms.SystematicsCalculator.dynamical_scale(2, p.tolist()) == approx(
            ht, rel=1e-12
        )
        assert ms.SystematicsCalculator.dynamical_scale(3, p.tolist()) == approx(
            ht / 2, rel=1e-12
        )
        assert ms.SystematicsCalculator.dynamical_scale(4, p.tolist()) == approx(
            shat, rel=1e-12
        )

    config = make_config(mur=[0.5, 1.0], muf=[1.0, 2.0], dyn_scales=[3, 4])
    calc = ms.SystematicsCalculator(
        config, [ms.SubprocessSystArgs(qcd_power=2, beam_pdgs=[[2, 2]])], grid, alpha_s
    )
    # (mur, muf) grid at the generated scale minus the nominal point, then the
    # full grid for each dynamical choice
    assert [(v.mur, v.muf, v.dyn) for v in calc.variations] == [
        (0.5, 1.0, -1),
        (0.5, 2.0, -1),
        (1.0, 2.0, -1),
        (0.5, 1.0, 3),
        (0.5, 2.0, 3),
        (1.0, 1.0, 3),
        (1.0, 2.0, 3),
        (0.5, 1.0, 4),
        (0.5, 2.0, 4),
        (1.0, 1.0, 4),
        (1.0, 2.0, 4),
    ]
    header = calc.initrwgt()
    assert 'DYN_SCALE="3" PDF="338500" > dyn_scale_choice=HT/2' in header
    events = make_events(ref, n=20)
    events["momenta"] = momenta
    weights = calc.weights(**events)
    for i in range(20):
        w0, mu = events["event_weight"][i], events["ren_scale"][i]
        x1, x2 = events["x1"][i], events["x2"][i]
        nominal_pdf = ref.pdf(2, x1, mu) * ref.pdf(2, x2, mu)
        for var, w in zip(calc.variations, weights[i]):
            mu_dyn = (
                mu
                if var.dyn == -1
                else ms.SystematicsCalculator.dynamical_scale(var.dyn, momenta[i])
            )
            r_alpha = (ref.alphas(var.mur * mu_dyn) / ref.alphas(mu)) ** 2
            r_pdf = (
                ref.pdf(2, x1, var.muf * mu_dyn)
                * ref.pdf(2, x2, var.muf * mu_dyn)
                / nominal_pdf
            )
            assert w == approx(w0 * r_alpha * r_pdf, rel=1e-12)


def test_event_histograms(grids, ref):
    """Event-sample histograms: bin sums equal the mean weight per column,
    the scale envelope brackets the nominal, binomial errors."""
    grid, alpha_s = grids
    n = 200
    momenta = random_momenta(n, seed=6)
    pt = np.array([np.hypot(p[2][1], p[2][2]) for p in momenta])
    ctx = ms.Context(device=ms.cpu_device(), thread_count=1)
    pids = [2, 2, 2, 2]
    obs = ms.ObservableValues(
        [
            ms.Observable(
                pids,
                observable="pt",
                select_pids=[[2]],
                order_observable="pt",
                order_indices=[1],
                name="jet-pt",
            ),
            ms.Observable(pids, observable="sqrt_s", select_pids=[], name="sqrt_s"),
        ]
    )
    specs = [
        ms.EventHistogramSpec("jet-pt", 0.0, 500.0, 10),
        ms.EventHistogramSpec("sqrt_s", 0.0, 2000.0, 8),
    ]
    hists = ms.EventHistograms(ctx, specs, [ms.SubprocessObservables(obs, 4)])
    rng = np.random.default_rng(7)
    weight = np.full(n, 3.0)
    syst = np.stack(
        [weight * 0.8, weight * 1.3, weight], axis=1
    )  # two scale-like columns + nominal copy
    hists.fill(list(weight), [0] * n, momenta, syst.tolist())
    data = json.loads(hists.to_json())
    assert [h["name"] for h in data] == ["jet-pt", "sqrt_s"]
    jet = data[0]
    assert sum(jet["bin_values"]) == approx(3.0)
    assert [sum(w["bin_values"]) for w in jet["weights"]] == approx([2.4, 3.9, 3.0])
    # binning of the leading-pt observable against numpy
    counts, _ = np.histogram(pt, bins=10, range=(0.0, 500.0))
    assert jet["bin_values"][1:-1] == approx(list(3.0 * counts / n))
    # binomial error of an unweighted sample
    m = counts[counts.argmax()]
    assert jet["bin_errors"][1 + counts.argmax()] == approx(
        3.0 * np.sqrt(m * (1 - m / n)) / n
    )
    # a systematics calculator adds the envelope: fake one with two scale variations
    config = make_config(mur=[0.5, 1.0, 2.0], muf=[1.0], pdf_members=[])
    calc = ms.SystematicsCalculator(
        config, [ms.SubprocessSystArgs(qcd_power=2, beam_pdgs=[[2, 2]])], grid, alpha_s
    )
    assert calc.weight_count == 2
    hists2 = ms.EventHistograms(ctx, specs, [ms.SubprocessObservables(obs, 4)])
    hists2.fill(list(weight), [0] * n, momenta, syst[:, :2].tolist())
    data2 = json.loads(hists2.to_json(calc))
    env = data2[0]["scale_envelope"]
    for lo, nom, hi in zip(env["low"], data2[0]["bin_values"], env["high"]):
        assert lo <= nom + 1e-12 and hi >= nom - 1e-12
    assert [w["id"] for w in data2[0]["weights"]] == [1, 2]


def test_mlm_vertices_follow_the_emission_scale(grids, ref):
    """An MLM event carries alpha_s(mu_R)^(n - n_v) prod_v alpha_s(q_v). As in
    madevent's systematics.py, a mu_R variation moves only the n - n_v powers,
    and an emission scale variation (alpsfact) only the vertices. A vertex
    carrying 0 is one the clustering did not reweight: it is one of the mu_R
    powers. A reweighted vertex can sit exactly at mu_R (one clustering under
    clustering_mean) and is still a vertex."""
    grid, alpha_s = grids
    config = make_config(mur=[0.5, 1.0, 2.0], muf=[1.0], alps=[0.5, 1.0, 2.0])
    n = 4
    args = [ms.SubprocessSystArgs(qcd_power=n, beam_pdgs=[[2, 2]])]
    calc = ms.SystematicsCalculator(config, args, grid, alpha_s)
    variations = calc.variations
    assert [(v.mur, v.alps) for v in variations] == [
        (0.5, 1.0),
        (2.0, 1.0),
        (1.0, 0.5),
        (1.0, 2.0),
    ]
    assert [v.is_emission for v in variations] == [False, False, True, True]
    assert calc.emission_variation_indices == [2, 3]
    assert 'name="Emission scale variation"' in calc.initrwgt()
    assert 'ALPSFACT="0.5"' in calc.initrwgt()

    events = make_events(ref, n=30)
    rng = np.random.default_rng(7)
    vertices = []
    for i, mu in enumerate(events["ren_scale"]):
        # two reweighted vertices, the second at mu_R on every other event,
        # and one that is not reweighted
        second = mu if i % 2 else mu * rng.uniform(0.05, 0.9)
        vertices.append([mu * rng.uniform(0.05, 0.9), second, 0.0])
    weights = calc.weights(**events, alphas_scales=vertices)
    for i, row in enumerate(weights):
        w0 = events["event_weight"][i]
        mu = events["ren_scale"][i]
        q = np.array(vertices[i][:2])
        nominal = ref.alphas(mu) ** (n - 2) * np.prod(ref.alphas(q))
        for var, w in zip(variations, row):
            varied = ref.alphas(var.mur * mu) ** (n - 2) * np.prod(
                ref.alphas(var.alps * q)
            )
            assert w == approx(w0 * varied / nominal, rel=1e-12)

    # without the vertex scales in the record: every power follows mu_R, and
    # an emission variation is the nominal weight
    for w0, mu, row in zip(
        events["event_weight"], events["ren_scale"], calc.weights(**events)
    ):
        assert row[0] == approx(
            w0 * (ref.alphas(0.5 * mu) / ref.alphas(mu)) ** n, rel=1e-12
        )
        assert row[2] == approx(w0, rel=1e-15)


def test_mlm_pdf_ladder_follows_alpsfact_and_muf(grids, ref):
    """The pdf reweighting ladder of an MLM event, varied as madevent's
    systematics.py rebuilds it from the pdfrwt entries of each beam: the beam
    parton at x_beam and its leading scale, then per step the line's flavour,
    its momentum fraction before the step and the step's scale. With S the
    last entry's scale, a variation is
        f(x_N-1, k S) prod_i<N-1 f_i(x_i, Q_i) / f_i(x_i+1, Q_i),
        Q_i = min(alpsfact q_i, k S),
    relative to the same at k = alpsfact = 1, with every scale floored at
    ladder_min_scale. A beam with no step has only
    its leading density, at k S. An event without a ladder (the 2 -> 1
    sample) keeps the plain beam ratio."""
    grid, alpha_s = grids
    low = 2.0
    config = make_config(
        mur=[1.0],
        muf=[0.5, 1.0, 2.0],
        alps=[0.5, 1.0, 2.0],
        pdf_ladder=True,
        ladder_min_scale=low,
    )
    # no alpha_s: the weights move through the PDFs alone
    args = [ms.SubprocessSystArgs(qcd_power=0, beam_pdgs=[[2, 2]])]
    calc = ms.SystematicsCalculator(config, args, grid, alpha_s)
    variations = calc.variations
    assert [(v.muf, v.alps) for v in variations] == [
        (0.5, 1.0),
        (2.0, 1.0),
        (1.0, 0.5),
        (1.0, 2.0),
    ]

    events = make_events(ref, n=20)
    rng = np.random.default_rng(11)
    ladder, lead = [], []
    for i, (x1, mu) in enumerate(zip(events["x1"], events["ren_scale"])):
        if i == 0:
            ladder.append([])
            lead.append([0.0, 0.0])  # no ladder at all
            continue
        q1 = max(mu * rng.uniform(0.02, 0.8), low)
        q2 = max(mu * rng.uniform(0.02, 0.8), q1)
        z1, z2 = rng.uniform(0.3, 0.9, 2)
        # beam 1 lowered to q1, with two steps (a gluon line from q1 to q2,
        # then a quark line up to mu); beam 2 left at mu with no step.
        # Step rows: [x before the step, pdg, scale, active + 2 beam]
        lead.append([q1, mu])
        ladder.append([[0.0] * 4, [x1, 21, q2, 1.0], [x1 * z1, 2, mu, 1.0]])
        del z2
    weights = calc.weights(**events, ladder=ladder, ladder_lead=lead)

    def ladder_weight(i, k, a):
        x1, x2, mu = events["x1"][i], events["x2"][i], events["ren_scale"][i]
        clamp = lambda q: max(q, low)
        # beam 1: entries (2, x1, q1), (21, x1, q2), (2, x1 z1, mu)
        entries = [
            (2, x1, lead[i][0]),
            (21, x1, ladder[i][1][2]),
            (2, ladder[i][2][0], ladder[i][2][2]),
        ]
        top = k * entries[-1][2]
        w = ref.pdf(entries[-1][0], entries[-1][1], clamp(top))
        for (pdg, x, q), (_, x_next, _) in zip(entries, entries[1:]):
            scale = clamp(min(a * q, top))
            w *= ref.pdf(pdg, x, scale) / ref.pdf(pdg, x_next, scale)
        # beam 2: its leading density alone, at k mu
        return w * ref.pdf(2, x2, clamp(k * lead[i][1]))

    for i, row in enumerate(weights):
        w0 = events["event_weight"][i]
        x1, x2, mu = events["x1"][i], events["x2"][i], events["ren_scale"][i]
        for var, w in zip(variations, row):
            if i == 0:
                if var.alps != 1.0:
                    assert w == approx(w0, rel=1e-15)
                    continue
                expected = (
                    ref.pdf(2, x1, var.muf * mu)
                    * ref.pdf(2, x2, var.muf * mu)
                    / events["partial_weight_product"][i]
                )
            else:
                expected = ladder_weight(i, var.muf, var.alps) / ladder_weight(
                    i, 1.0, 1.0
                )
            assert w == approx(w0 * expected, rel=1e-12)
