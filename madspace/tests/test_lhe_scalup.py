"""SCALUP of the written events: the larger factorisation scale, as madevent's
unwgt.f writes sqrt(max(q2fact)), with mu_R only in <scales> and AQCDUP.

u u~ > d d~ g with the built-in flat test matrix element and a flat PDF, at
fixed scales chosen all different (mu_R 91.188, mu_F1 40, mu_F2 60), generated
through an EventStream like a gridpack does; e+ e- > d d~ g is the same without
a PDF, where SCALUP stays mu_R.
"""

import os
import re

import numpy as np
import pytest

import madspace as ms

CM_ENERGY = 1000.0
MU_R, MU_F1, MU_F2 = 91.188, 40.0, 60.0
SEED = 4321


def build_topology():
    propagators = [
        ms.Propagator(
            mass=0.0, width=0.0, integration_order=0, e_min=0.0, e_max=0.0, pdg_id=pid
        )
        for pid in (22, -1)
    ]
    diagram = ms.Diagram(
        [0.0, 0.0],
        [0.0, 0.0, 0.0],
        propagators,
        [["i0", "i1", "p0"], ["o0", "p1", "p0"], ["o1", "o2", "p1"]],
    )
    return ms.Topology.topologies(diagram)[0]


PERMUTATIONS = [[0, 1, 2, 3, 4]]


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


def make_config():
    config = ms.GeneratorConfig()
    config.cpu_batch_size = 1000
    config.freeze_max_weight_after = 1000
    config.max_overweight_truncation = 0.05
    return config


def lhe_completer(pids, hadronic):
    initial = [[501, 0], [0, 501]] if hadronic else [[0, 0], [0, 0]]
    subproc_args = ms.SubprocArgs(
        process_id=0,
        topologies=[build_topology()],
        permutations=[PERMUTATIONS],
        diagram_indices=[[0]],
        diagram_color_indices=[[[0]]],
        color_flows=[initial + [[502, 0], [0, 503], [503, 502]]],
        pdg_color_types={11: 1, -11: 1, 2: 3, -2: -3, 1: 3, -1: -3, 22: 1, 21: 8},
        helicities=[[1, -1, 1, -1, 1]],
        pdg_ids=[[pids]],
    )
    return ms.LHECompleter([subproc_args], bw_cutoff=15.0)


def make_stream(run_dir, hadronic):
    """An EventStream over one channel whose max weight was fixed by a survey,
    reloaded from its JSON like a fix_max_weight gridpack."""
    pids = [2, -2, 1, -1, 21] if hadronic else [11, -11, 1, -1, 21]
    context = ms.Context(ms.cpu_device(), 2)
    alphas_file = os.path.join(run_dir, "alphas.info")
    with open(alphas_file, "w") as f:
        f.write("AlphaS_Qs: [1.0, 10.0, 100.0, 1000.0, 10000.0]\n")
        f.write("AlphaS_Vals: [0.3, 0.2, 0.12, 0.1, 0.08]\n")
    ms.AlphaSGrid(alphas_file).initialize_globals(context)
    pdf_kwargs, xs_kwargs = {}, {}
    if hadronic:
        pdf_file = os.path.join(run_dir, "flat.dat")
        write_flat_pdf_grid(pdf_file, 0.5)
        pdf_grid = ms.PdfGrid(pdf_file)
        pdf_grid.initialize_globals(context)
        pdf_kwargs = dict(pid_options=[pids], pdf_grid=pdf_grid)
        xs_kwargs = dict(pdf1=ms.CachedPdf(), pdf2=ms.CachedPdf())
    cuts = ms.Cuts(
        [
            ms.CutItem(ms.Observable(pids, ms.Observable.obs_pt, [[i]]), min=20.0)
            for i in (2, 3, 4)
        ]
    )
    mapping = ms.PhaseSpaceMapping(
        build_topology(),
        CM_ENERGY,
        permutations=PERMUTATIONS,
        leptonic=not hadronic,
        cuts=cuts,
    )
    # 0xBADCAFE: the built-in flat test matrix element, no library needed
    matrix_element = ms.MatrixElement(
        0xBADCAFE,
        len(pids),
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
        **xs_kwargs,
    )
    integrand = ms.Integrand(
        mapping,
        [diff_xs],
        running_coupling=ms.RunningCoupling(ms.AlphaSGrid(alphas_file)),
        energy_scale=ms.EnergyScale(
            len(pids),
            ms.EnergyScale.DynamicalScaleType.transverse_energy,
            ren_scale_fixed=True,
            fact_scale_fixed=True,
            ren_scale=MU_R,
            fact_scale1=MU_F1,
            fact_scale2=MU_F2,
        ),
        channel_indices=[0],
        **pdf_kwargs,
    )
    channel = ms.ChannelEventGenerator(
        contexts=[context],
        integrand=integrand,
        event_file=os.path.join(run_dir, "events.npy"),
        weight_file=os.path.join(run_dir, "weights.npy"),
        config=make_config(),
        subprocess_index=0,
        name="a",
        histograms=None,
    )
    generator = ms.EventGenerator([context], [channel], seed=1, config=make_config())
    generator.survey()
    generator.fix_max_weights()
    globals_file = os.path.join(run_dir, "globals.tar")
    context.save_globals(globals_file)

    stream_context = ms.Context(ms.cpu_device(), 2)
    stream_context.load_globals(globals_file)
    config = make_config()
    channels = [
        ms.ChannelEventGenerator.load_json(
            channel.to_json(True), [stream_context], config=config
        )
    ]
    return ms.EventStream(
        stream_context, channels, SEED, lhe_completer(pids, hadronic), config
    )


def scales_attributes(text):
    """The <scales> attributes of one formatted event, pt_clust_i aside."""
    tag = text[text.index("<scales") : text.index("</scales>")]
    return {
        name: float(value)
        for name, value in re.findall(r'(\w+)="([^"]*)"', tag)
        if not name.startswith("pt_clust_")
    }


@pytest.fixture(scope="module")
def hadronic_stream(tmp_path_factory):
    return make_stream(str(tmp_path_factory.mktemp("lhe_scalup_pp")), True)


@pytest.fixture(scope="module")
def leptonic_stream(tmp_path_factory):
    return make_stream(str(tmp_path_factory.mktemp("lhe_scalup_ee")), False)


def test_scalup_is_the_larger_factorisation_scale(hadronic_stream):
    batch = hadronic_stream.next_batch(200)
    np.testing.assert_array_equal(batch["scale"], MU_F2)


def test_scales_tag_keeps_both_scales(hadronic_stream):
    for event in hadronic_stream.next_events(20):
        text = event.format()
        header = text.split("\n")[1].split()
        assert float(header[3]) == pytest.approx(MU_F2)  # SCALUP
        assert scales_attributes(text) == pytest.approx(
            {"mur": MU_R, "muf": MU_F2, "muf1": MU_F1, "muf2": MU_F2}
        )


def test_scalup_stays_the_renormalisation_scale_without_a_pdf(leptonic_stream):
    batch = leptonic_stream.next_batch(200)
    np.testing.assert_array_equal(batch["scale"], MU_R)
    for event in leptonic_stream.next_events(5):
        assert scales_attributes(event.format()) == pytest.approx(
            {"mur": MU_R, "muf": MU_R, "muf1": MU_R, "muf2": MU_R}
        )
