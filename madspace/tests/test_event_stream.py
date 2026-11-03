"""EventStream: in-memory event generation from channels with a fixed maximum
weight, as stored in a fix_max_weight gridpack.

The channels of an e+ e- -> d d~ g process (flat matrix element, so the weights
only vary through the phase-space mapping) are surveyed and their maximum
weights fixed with EventGenerator.fix_max_weights(), then reloaded from their
JSON like a gridpack does.
"""

import json
import os

import numpy as np
import pytest

import madspace as ms

CM_ENERGY = 1000.0
PIDS = [11, -11, 1, -1, 21]
SEED = 1234


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


TOPOLOGY = build_topology()
PERMUTATIONS = [[0, 1, 2, 3, 4]]


def lhe_completer():
    subproc_args = ms.SubprocArgs(
        process_id=0,
        topologies=[TOPOLOGY],
        permutations=[PERMUTATIONS],
        diagram_indices=[[0]],
        diagram_color_indices=[[[0]]],
        color_flows=[[[0, 0], [0, 0], [501, 0], [0, 502], [502, 501]]],
        pdg_color_types={11: 1, -11: 1, 1: 3, -1: -3, 22: 1, 21: 8},
        helicities=[[1, -1, 1, -1, 1]],
        pdg_ids=[[PIDS]],
    )
    return ms.LHECompleter([subproc_args], bw_cutoff=15.0)


def make_config():
    config = ms.GeneratorConfig()
    config.cpu_batch_size = 1000
    config.freeze_max_weight_after = 1000
    config.max_overweight_truncation = 0.05
    return config


def make_context(run_dir, thread_count):
    context = ms.Context(ms.cpu_device(), thread_count)
    # RunningCoupling only reads these two keys of an LHAPDF .info file
    alphas_file = os.path.join(run_dir, "alphas.info")
    with open(alphas_file, "w") as f:
        f.write("AlphaS_Qs: [1.0, 10.0, 100.0, 1000.0, 10000.0]\n")
        f.write("AlphaS_Vals: [0.3, 0.2, 0.12, 0.1, 0.08]\n")
    ms.AlphaSGrid(alphas_file).initialize_globals(context)
    return context, alphas_file


def make_channel(context, alphas_file, run_dir, name, pt_min):
    cuts = ms.Cuts(
        [
            ms.CutItem(ms.Observable(PIDS, ms.Observable.obs_pt, [[i]]), min=pt_min)
            for i in (2, 3, 4)
        ]
    )
    mapping = ms.PhaseSpaceMapping(
        TOPOLOGY, CM_ENERGY, permutations=PERMUTATIONS, leptonic=True, cuts=cuts
    )
    # 0xBADCAFE: the built-in flat test matrix element, no library needed
    matrix_element = ms.MatrixElement(
        0xBADCAFE,
        len(PIDS),
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
    )
    integrand = ms.Integrand(
        mapping,
        [diff_xs],
        running_coupling=ms.RunningCoupling(ms.AlphaSGrid(alphas_file)),
        energy_scale=ms.EnergyScale(len(PIDS), 91.188),
        channel_indices=[0],
    )
    return ms.ChannelEventGenerator(
        contexts=[context],
        integrand=integrand,
        event_file=os.path.join(run_dir, f"events.{name}.npy"),
        weight_file=os.path.join(run_dir, f"weights.{name}.npy"),
        config=make_config(),
        subprocess_index=0,
        name=name,
        histograms=None,
    )


@pytest.fixture(scope="module")
def gridpack(tmp_path_factory):
    """Channel JSONs with fixed max weights and the globals, like a gridpack."""
    run_dir = str(tmp_path_factory.mktemp("event_stream"))
    context, alphas_file = make_context(run_dir, 4)
    channels = [
        make_channel(context, alphas_file, run_dir, "a", 20.0),
        make_channel(context, alphas_file, run_dir, "b", 60.0),
    ]
    generator = ms.EventGenerator([context], channels, seed=1, config=make_config())
    generator.survey()
    generator.fix_max_weights()
    globals_file = os.path.join(run_dir, "globals.tar")
    context.save_globals(globals_file)
    return {
        "run_dir": run_dir,
        "channels": [channel.to_json(True) for channel in channels],
        "globals": globals_file,
        "integral": generator.status().mean,
    }


def make_stream(gridpack, thread_count=4, seed=SEED):
    context = ms.Context(ms.cpu_device(), thread_count)
    context.load_globals(gridpack["globals"])
    config = make_config()
    channels = [
        ms.ChannelEventGenerator.load_json(channel, [context], config=config)
        for channel in gridpack["channels"]
    ]
    return ms.EventStream(context, channels, seed, lhe_completer(), config)


def stream_batches(stream, batch_sizes):
    batches = [stream.next_batch(size) for size in batch_sizes]
    return {key: np.concatenate([b[key] for b in batches]) for key in batches[0]}


def test_reproducible_across_batch_sizes_and_threads(gridpack):
    reference = stream_batches(make_stream(gridpack, 4), [3000])
    for thread_count, batch_sizes in [(1, [1000, 1000, 1000]), (3, [1, 999, 2000])]:
        events = stream_batches(make_stream(gridpack, thread_count), batch_sizes)
        for key, values in reference.items():
            np.testing.assert_array_equal(events[key], values, err_msg=key)


def test_seed_changes_events(gridpack):
    events1 = make_stream(gridpack, seed=1).next_batch(100)
    events2 = make_stream(gridpack, seed=2).next_batch(100)
    assert not np.array_equal(events1["px"], events2["px"])


def test_lhe_events_match_batches(gridpack):
    lhe_events = make_stream(gridpack).next_events(50)
    batch = make_stream(gridpack).next_batch(50)
    for i, event in enumerate(lhe_events):
        assert event.weight == batch["weight"][i]
        assert [p.pdg_id for p in event.particles] == list(batch["pdg_id"][i])
        assert [p.px for p in event.particles] == list(batch["px"][i])


def test_weights_and_particles(gridpack):
    stream = make_stream(gridpack)
    batch = stream.next_batch(5000)
    weights = batch["weight"]
    # weights are 1, except for overweight events
    assert np.all(weights >= 1.0)
    assert np.mean(weights == 1.0) > 0.9
    assert np.all(batch["particle_count"] == len(PIDS))
    assert set(map(tuple, batch["pdg_id"])) == {tuple(PIDS)}
    assert batch["px"].shape == (5000, stream.max_particle_count())
    # momentum conservation
    outgoing = batch["status_code"] == 1
    assert np.allclose(np.sum(batch["px"] * outgoing, axis=1), 0.0, atol=1e-6)


def test_status(gridpack):
    stream = make_stream(gridpack)
    stream.next_batch(2000)
    status = stream.status()
    assert stream.event_count() == 2000
    assert status.count_unweighted == 2000
    assert status.count > 0
    assert status.mean == pytest.approx(gridpack["integral"], rel=0.1)
    channel_status = stream.channel_status()
    assert sum(s.count_unweighted for s in channel_status) == 2000
    assert sum(stream.channel_probabilities()) == pytest.approx(1.0)


def test_channel_fractions_follow_probabilities(gridpack):
    stream = make_stream(gridpack)
    stream.next_batch(20000)
    probs = np.array(stream.channel_probabilities())
    counts = np.array([s.count_unweighted for s in stream.channel_status()])
    fractions = counts / counts.sum()
    assert np.allclose(fractions, probs, atol=0.02)


def test_no_files_written(gridpack, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stream = make_stream(gridpack)
    stream.next_batch(1000)
    stream.close()
    assert os.listdir(tmp_path) == []


def test_requires_fixed_max_weight(gridpack):
    context = ms.Context(ms.cpu_device(), 1)
    context.load_globals(gridpack["globals"])
    config = make_config()
    channels = []
    for channel in gridpack["channels"]:
        data = json.loads(channel)
        del data["max_weight"]
        channels.append(
            ms.ChannelEventGenerator.load_json(
                json.dumps(data), [context], config=config
            )
        )
    with pytest.raises(ValueError, match="fixed maximum weight"):
        ms.EventStream(context, channels, SEED, lhe_completer(), config)
