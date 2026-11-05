"""Re- and unweighting steps after the first unweighting (WeightingStep).

The integrand uses a deformed "surrogate" matrix element, and a later step
re- or unweights the events with the "true" one. The results have to agree with
a run that uses the true matrix element in the integrand, and must not depend
on the number of threads. Uses the e+ e- -> d d~ g setup of test_event_stream
with PythonFunction matrix elements.
"""

import json
import os

import numpy as np
import pytest
from test_event_stream import (
    CM_ENERGY,
    PERMUTATIONS,
    PIDS,
    TOPOLOGY,
    lhe_completer,
    make_config,
    make_context,
)
from test_integrand_me_outputs import python_matrix_element

import madspace as ms


def true_me(momenta):
    return 1.0 + 3.0 * (momenta[:, 2, 0] / CM_ENERGY) ** 2


def surrogate_me(momenta):
    return true_me(momenta) * (1.0 + 0.3 * np.sin(momenta[:, 3, 1] / 50.0))


def make_integrand(alphas_file, matrix_element, pt_min=20.0):
    mapping = ms.PhaseSpaceMapping(
        TOPOLOGY,
        CM_ENERGY,
        permutations=PERMUTATIONS,
        leptonic=True,
        cuts=ms.Cuts(
            [
                ms.CutItem(ms.Observable(PIDS, ms.Observable.obs_pt, [[i]]), min=pt_min)
                for i in (2, 3, 4)
            ]
        ),
    )
    diff_xs = ms.DifferentialCrossSection(
        matrix_element=matrix_element,
        cm_energy=CM_ENERGY,
        running_coupling=None,
        energy_scale=ms.CachedScale(),
        return_matrix_element=True,
    )
    return ms.Integrand(
        mapping,
        [diff_xs],
        running_coupling=ms.RunningCoupling(ms.AlphaSGrid(alphas_file)),
        energy_scale=ms.EnergyScale(len(PIDS), 91.188),
        channel_indices=[0],
    )


def generate(
    run_dir,
    integrand_me,
    step_modes=(),
    step_me=true_me,
    target_count=5000,
    threads=2,
    pt_mins=(20.0,),
    grouped=(0,),
    seed=7,
    step_outputs=("matrix_element",),
    override_outputs=(),
    integrand_outputs=("matrix_element",),
    step_channel=-1,
    step_kwargs=None,
    config_kwargs=None,
    survey=True,
):
    """Survey and generate, returns the status, the combined events and the
    channels. step_me and step_outputs can be lists, one entry per step, and
    integrand_me one entry per channel."""
    os.makedirs(run_dir, exist_ok=True)
    context, alphas_file = make_context(run_dir, threads)
    if not isinstance(integrand_me, list):
        integrand_me = [integrand_me] * len(pt_mins)
    me0s = [
        python_matrix_element(context, me, integrand_outputs) for me in integrand_me
    ]
    if not isinstance(step_me, list):
        step_me = [step_me] * len(step_modes)
        step_outputs = [step_outputs] * len(step_modes)
    step_mes = [
        python_matrix_element(context, me, outputs, step_channel)
        for me, outputs in zip(step_me, step_outputs)
    ]
    config = make_config()
    config.target_count = target_count
    config.cpu_batch_size = 500
    for key, value in (config_kwargs or {}).items():
        setattr(config, key, value)
    channels = [
        ms.ChannelEventGenerator(
            contexts=[context],
            integrand=make_integrand(alphas_file, me0, pt_min),
            event_file=os.path.join(run_dir, f"events.{i}.npy"),
            weight_file=os.path.join(run_dir, f"weights.{i}.npy"),
            config=config,
            subprocess_index=0,
            name=str(i),
            histograms=None,
        )
        for i, (pt_min, me0) in enumerate(zip(pt_mins, me0s))
    ]
    steps = [
        ms.WeightingStep(
            [me],
            mode=mode,
            override_outputs=list(override_outputs),
            **(step_kwargs or {}),
        )
        for me, mode in zip(step_mes, step_modes)
    ]
    groups = [ms.WeightingGroup(list(grouped), steps)] if steps else []
    generator = ms.EventGenerator(
        [context], channels, seed=seed, config=config, groups=groups
    )
    if survey:
        generator.survey()
    generator.generate()
    output_file = os.path.join(run_dir, "combined.npy")
    generator.combine_to_compact_npy(output_file)
    return generator.status(), np.load(output_file), channels


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """The true matrix element in the integrand."""
    return generate(
        str(tmp_path_factory.mktemp("reference")), true_me, target_count=20000
    )


def check_compatible(status, events, reference):
    ref_status, ref_events, _ = reference
    error = np.hypot(status.error, ref_status.error)
    assert abs(status.mean - ref_status.mean) < 4 * error
    # the events are normalized to the cross section
    assert np.mean(events["weight"]) == pytest.approx(status.mean_abs, rel=1e-9)
    # mean energy of the first quark, weighted
    energies = [
        np.sum(ev["weight"] * ev["part3_energy"]) / np.sum(ev["weight"])
        for ev in (events, ref_events)
    ]
    assert energies[0] == pytest.approx(energies[1], rel=0.01)


def test_unweight_step(tmp_path, reference):
    status, events, channels = generate(
        str(tmp_path), surrogate_me, ["unweight"], target_count=20000
    )
    assert len(events) == 20000
    check_compatible(status, events, reference)
    # unweighted: apart from the overweights, all events have the same weight
    weights = events["weight"]
    assert np.mean(np.isclose(weights, np.min(weights))) > 0.98
    assert channels[0].step_max_weight(1) > 0


def test_reweight_step(tmp_path, reference):
    status, events, _ = generate(
        str(tmp_path), surrogate_me, ["reweight"], target_count=20000
    )
    check_compatible(status, events, reference)
    # the ratios of the matrix elements stay in the weights
    weights = events["weight"]
    assert np.std(weights) / np.mean(weights) > 0.1


def test_reweight_then_unweight(tmp_path, reference):
    """The unweighting step absorbs the ratio of the reweighting step: the
    second step has the same matrix element, so it only unweights."""
    status, events, _ = generate(
        str(tmp_path), surrogate_me, ["reweight", "unweight"], target_count=20000
    )
    check_compatible(status, events, reference)
    weights = events["weight"]
    assert np.mean(np.isclose(weights, np.min(weights))) > 0.98


@pytest.mark.parametrize("step_modes", [["unweight"], ["reweight", "unweight"]])
def test_reproducible(tmp_path, step_modes):
    """Two grouped channels and one without steps, which share the batches
    differently with different thread counts."""
    results = [
        generate(
            str(tmp_path / str(threads)),
            surrogate_me,
            step_modes,
            threads=threads,
            pt_mins=(20.0, 40.0, 60.0),
            grouped=(0, 2),
        )
        for threads in (1, 4)
    ]
    (status_a, events_a, _), (status_b, events_b, _) = results
    assert len(events_a) == 5000
    assert np.array_equal(events_a, events_b)
    assert status_a.mean == status_b.mean


def heavy_tailed_me(momenta):
    return surrogate_me(momenta) * (
        1.0 + 0.05 / (0.002 + np.abs(np.sin(momenta[:, 3, 1] / 7.0)))
    )


def test_reproducible_step_freeze(tmp_path):
    """The step maximum freezes while the first one still rises, in an order
    relative to the step decisions that depends on the threads."""
    results = [
        generate(
            str(tmp_path / str(threads)),
            heavy_tailed_me,
            ["unweight"],
            threads=threads,
            pt_mins=(20.0, 40.0, 60.0),
            grouped=(0, 1, 2),
            step_kwargs={"freeze_max_weight_after": 300},
            config_kwargs={
                "freeze_max_weight_after": 100000,
                "max_overweight_truncation": 0.002,
            },
        )
        for threads in (1, 2, 4, 8)
    ]
    status_a, events_a, channels_a = results[0]
    for status_b, events_b, channels_b in results[1:]:
        assert [c.step_max_weight(1) for c in channels_a] == [
            c.step_max_weight(1) for c in channels_b
        ]
        assert np.array_equal(events_a, events_b)
        assert status_a.mean == status_b.mean


def test_survey_skips_cleared_iterations(tmp_path):
    """Only the middle channel iterates again in the survey, so the steps of its
    earlier iteration are skipped, independently of the threads."""
    results = [
        generate(
            str(tmp_path / str(threads)),
            [surrogate_me, heavy_tailed_me, surrogate_me],
            ["reweight", "unweight"],
            threads=threads,
            pt_mins=(20.0, 40.0, 60.0),
            grouped=(0, 1, 2),
            config_kwargs={"survey_max_iters": 6, "survey_target_precision": 0.02},
        )
        for threads in (1, 4)
    ]
    (status_a, events_a, channels_a), (status_b, events_b, _) = results
    assert [c.status().iterations for c in channels_a] == [3, 4, 3]
    assert np.array_equal(events_a, events_b)
    assert status_a.mean == status_b.mean


def test_step_budget_of_smaller_target(tmp_path):
    """A smaller target applies its truncation budget to the step maxima too,
    and rescales the unweighted count accordingly. EventGenerator does this at
    the end of each round, as generate() starts with a generous target."""
    small = lambda momenta: surrogate_me(momenta) * (momenta[:, 3, 1] > 300.0)
    _, _, channels = generate(
        str(tmp_path),
        [surrogate_me, small],
        ["unweight"],
        step_me=heavy_tailed_me,
        pt_mins=(20.0, 20.0),
        grouped=(0, 1),
        target_count=20000,
        survey=False,
        step_kwargs={"freeze_max_weight_after": 10**6},
    )
    channel = channels[1]
    target = channel.status().count_target
    assert target < 5000
    # the budget of the final target is already applied
    max_weight, count = channel.step_max_weight(1), channel.status().count_unweighted
    channel.apply_step_truncation_budgets()
    assert channel.step_max_weight(1) == max_weight
    channel.set_target_count(target // 20)
    channel.apply_step_truncation_budgets()
    new_max_weight = channel.step_max_weight(1)
    assert new_max_weight > max_weight
    assert channel.status().count_unweighted == pytest.approx(
        count * max_weight / new_max_weight, rel=1e-12
    )


def broken_me(momenta, invalid=(np.nan, np.inf)):
    """true_me, with invalid values in two regions."""
    px = momenta[:, 3, 1]
    me = np.where(px > 100.0, invalid[0], true_me(momenta))
    return np.where(px < -100.0, invalid[1], me)


@pytest.mark.parametrize("step_modes", [["unweight"], ["reweight"]])
def test_invalid_step_values(tmp_path, capfd, step_modes):
    """Non-finite ratios count as weight 0, like a step matrix element of 0,
    and are counted and warned about."""
    results = [
        generate(
            str(tmp_path / name),
            surrogate_me,
            step_modes,
            step_me=step_me,
            pt_mins=(20.0, 40.0),
            grouped=(0, 1),
        )
        for name, step_me in [
            ("broken", broken_me),
            ("zero", lambda momenta: broken_me(momenta, (0.0, 0.0))),
        ]
    ]
    (status_broken, events_broken, channels), (status_zero, events_zero, _) = results
    assert np.array_equal(events_broken, events_zero)
    assert status_broken.mean == status_zero.mean
    assert status_zero.count_invalid_step_values == 0
    invalid_counts = [c.status().count_invalid_step_values for c in channels]
    assert all(count > 0 for count in invalid_counts)
    assert status_broken.count_invalid_step_values == sum(invalid_counts)
    if step_modes == ["unweight"]:
        assert all(np.isfinite(c.step_max_weight(1)) for c in channels)
    # after the survey and after generate()
    warnings = [line for line in capfd.readouterr().out.splitlines() if "WARN" in line]
    assert all("got the weight 0 in the weighting steps" in w for w in warnings)
    assert {w.split("'")[1] for w in warnings} == {"0", "1"}


def test_index_outputs(tmp_path):
    """A step fills in indices the integrand matrix element does not return,
    and replaces the others only with override_outputs."""

    def with_helicity(momenta):
        return true_me(momenta), np.full(len(momenta), 1, np.int32)

    outputs = ("matrix_element", "helicity_index")
    _, events, _ = generate(
        str(tmp_path / "fill"),
        surrogate_me,
        ["unweight"],
        step_me=with_helicity,
        step_outputs=outputs,
    )
    assert (events["helicity_index"] == 1).all()
    assert (events["color_index"] == -1).all()

    def surrogate_with_helicity(momenta):
        return surrogate_me(momenta), np.zeros(len(momenta), np.int32)

    for override, helicity in [((), 0), ((ms.MatrixElement.helicity_index_out,), 1)]:
        _, events, _ = generate(
            str(tmp_path / f"override{helicity}"),
            surrogate_with_helicity,
            ["unweight"],
            step_me=with_helicity,
            step_outputs=outputs,
            integrand_outputs=outputs,
            override_outputs=override,
        )
        assert (events["helicity_index"] == helicity).all()


def true_over_surrogate(momenta):
    return true_me(momenta) / surrogate_me(momenta)


@pytest.mark.parametrize("step_modes", [["unweight"], ["reweight"]])
def test_ratio_step(tmp_path, step_modes):
    """A step that returns |M_true|^2 / |M_surrogate|^2 gives the same events
    as a step that returns |M_true|^2: the pipeline computes the same ratio."""
    results = [
        generate(
            str(tmp_path / name),
            surrogate_me,
            step_modes,
            step_me=step_me,
            step_outputs=outputs,
        )
        for name, step_me, outputs in [
            ("me", true_me, ("matrix_element",)),
            ("ratio", true_over_surrogate, ("matrix_element_ratio",)),
        ]
    ]
    (status_me, events_me, _), (status_ratio, events_ratio, _) = results
    assert np.array_equal(events_me, events_ratio)
    assert status_me.mean == status_ratio.mean


def test_ratio_with_fixed_channel(tmp_path):
    """A step matrix element evaluated at a fixed channel gets it as an input,
    which is not an input of the step."""

    def ratio(momenta, channel):
        return true_over_surrogate(momenta) * (channel == 3)

    results = [
        generate(
            str(tmp_path / name),
            surrogate_me,
            ["reweight"],
            step_me=step_me,
            step_outputs=("matrix_element_ratio",),
            step_channel=channel,
        )
        for name, step_me, channel in [
            ("plain", true_over_surrogate, -1),
            ("channel", ratio, 3),
        ]
    ]
    (status_plain, events_plain, _), (status_channel, events_channel, _) = results
    assert np.array_equal(events_plain, events_channel)
    assert status_plain.mean == status_channel.mean


def test_ratio_then_matrix_element(tmp_path, reference):
    """After a ratio step, the matrix element it implies is the reference of the
    next step, so the ratios still multiply to |M_true|^2 / |M_surrogate|^2."""

    def intermediate_me(momenta):
        return true_me(momenta) * (1.0 + 0.2 * np.cos(momenta[:, 2, 2] / 40.0))

    status, events, _ = generate(
        str(tmp_path),
        surrogate_me,
        ["reweight", "unweight"],
        step_me=[lambda p: intermediate_me(p) / surrogate_me(p), true_me],
        step_outputs=[("matrix_element_ratio",), ("matrix_element",)],
        target_count=20000,
    )
    check_compatible(status, events, reference)
    weights = events["weight"]
    assert np.mean(np.isclose(weights, np.min(weights))) > 0.98


def test_ratio_matrix_element_rules(tmp_path):
    context, _ = make_context(str(tmp_path), 1)
    ratio_me = python_matrix_element(
        context, true_over_surrogate, ("matrix_element_ratio",)
    )
    assert ms.WeightingStep([ratio_me]).returns_ratio()
    assert not ms.WeightingStep(
        [python_matrix_element(context, true_me)]
    ).returns_ratio()
    with pytest.raises(ValueError, match="weighting step"):
        ms.DifferentialCrossSection(
            matrix_element=ratio_me,
            cm_energy=CM_ENERGY,
            running_coupling=None,
            energy_scale=ms.CachedScale(),
        )
    with pytest.raises(ValueError, match="not both"):
        ms.MatrixElement(
            ratio_me.matrix_element_index(),
            len(PIDS),
            [ms.MatrixElement.momenta_in],
            [
                ms.MatrixElement.matrix_element_out,
                ms.MatrixElement.matrix_element_ratio_out,
            ],
        )
    groups = [ms.WeightingGroup([0], [ms.WeightingStep([ratio_me])])]
    text = ms.weighting_groups_to_json(groups)
    loaded = ms.weighting_groups_from_json(text)
    assert loaded[0].steps[0].returns_ratio()
    assert ms.weighting_groups_to_json(loaded) == text
    # a fixed channel (AmpliCol colour ordering) survives the round trip
    channel_me = python_matrix_element(
        context, true_over_surrogate, ("matrix_element_ratio",), 3
    )
    text = ms.weighting_groups_to_json(
        [ms.WeightingGroup([0], [ms.WeightingStep([channel_me])])]
    )
    loaded = ms.weighting_groups_from_json(text)
    assert loaded[0].steps[0].matrix_elements[0].channel() == 3


def test_first_pass_settings_of_group(tmp_path):
    """The first unweighting of the channels of a group uses the settings of
    the group, the other channels the global ones."""
    context, alphas_file = make_context(str(tmp_path), 2)
    surrogate = python_matrix_element(context, surrogate_me)
    full = python_matrix_element(context, true_me)
    config = make_config()
    channels = [
        ms.ChannelEventGenerator(
            [context],
            make_integrand(alphas_file, surrogate),
            str(tmp_path / f"events.{name}.npy"),
            str(tmp_path / f"weights.{name}.npy"),
            config,
            0,
            name,
            None,
        )
        for name in ("group", "plain")
    ]
    group = ms.WeightingGroup(
        [0],
        [ms.WeightingStep([full], freeze_max_weight_after=300)],
        first_pass_max_overweight_truncation=0.2,
        first_pass_freeze_max_weight_after=2000,
    )
    generator = ms.EventGenerator(
        [context], channels, seed=3, config=config, groups=[group]
    )
    assert [c.max_overweight_truncation() for c in channels] == [0.2, 0.05]
    assert [c.freeze_max_weight_after() for c in channels] == [2000, 1000]
    generator.survey()
    # each channel runs until its own maximum freezes
    generator.fix_max_weights()
    assert [c.status().count_target for c in channels] == [2000, 1000]

    # stored with the group, old JSON without them still loads
    text = ms.weighting_groups_to_json([group])
    loaded = ms.weighting_groups_from_json(text)[0]
    assert loaded.first_pass_max_overweight_truncation == 0.2
    assert loaded.first_pass_freeze_max_weight_after == 2000
    old = json.loads(text)
    del old[0]["first_pass_max_overweight_truncation"]
    del old[0]["first_pass_freeze_max_weight_after"]
    loaded = ms.weighting_groups_from_json(json.dumps(old))[0]
    assert loaded.first_pass_max_overweight_truncation is None
    assert loaded.first_pass_freeze_max_weight_after is None
    assert "first_pass" not in ms.weighting_groups_to_json([loaded])


def test_invalid_groups(tmp_path):
    context, alphas_file = make_context(str(tmp_path), 1)
    me = python_matrix_element(context, true_me)
    config = make_config()

    def channel(return_matrix_element=True):
        integrand = make_integrand(alphas_file, me)
        if not return_matrix_element:
            diff_xs = ms.DifferentialCrossSection(
                matrix_element=me,
                cm_energy=CM_ENERGY,
                running_coupling=None,
                energy_scale=ms.CachedScale(),
            )
            integrand = ms.Integrand(
                ms.PhaseSpaceMapping(
                    TOPOLOGY, CM_ENERGY, permutations=PERMUTATIONS, leptonic=True
                ),
                [diff_xs],
                running_coupling=ms.RunningCoupling(ms.AlphaSGrid(alphas_file)),
                energy_scale=ms.EnergyScale(len(PIDS), 91.188),
                channel_indices=[0],
            )
        return ms.ChannelEventGenerator(
            [context], integrand, "", "", config, 0, "c", None
        )

    step = ms.WeightingStep([me])
    with pytest.raises(ValueError, match="matrix element"):
        ms.EventGenerator(
            [context], [channel(False)], seed=1, groups=[ms.WeightingGroup([0], [step])]
        )
    with pytest.raises(ValueError, match="one weighting group"):
        ms.EventGenerator(
            [context],
            [channel()],
            seed=1,
            groups=[ms.WeightingGroup([0], [step]), ms.WeightingGroup([0], [step])],
        )
    with pytest.raises(ValueError, match="sampled internally"):
        sampled = ms.MatrixElement(
            me.matrix_element_index(),
            len(PIDS),
            [ms.MatrixElement.momenta_in, ms.MatrixElement.random_helicity_in],
            [ms.MatrixElement.matrix_element_out],
            1,
            True,
        )
        ms.EventGenerator(
            [context],
            [channel()],
            seed=1,
            groups=[ms.WeightingGroup([0], [ms.WeightingStep([sampled])])],
        )


def full_me_with_indices(momenta):
    zeros = np.zeros(len(momenta), np.int32)
    return true_me(momenta), zeros, zeros, zeros


FULL_ME_OUTPUTS = ("matrix_element", "color_index", "helicity_index", "diagram_index")


def load_matrix_elements(context):
    """The matrix elements, always in the same order, so that their indices
    match the ones stored in the channels and groups."""
    surrogate = python_matrix_element(context, surrogate_me)
    full = python_matrix_element(context, full_me_with_indices, FULL_ME_OUTPUTS)
    return surrogate, full


@pytest.fixture(scope="module")
def fixed_gridpack(tmp_path_factory):
    """Two grouped channels with all maxima fixed by fix_max_weights(), stored
    like in a gridpack."""
    config = make_config()
    run_dir = str(tmp_path_factory.mktemp("fixed"))
    context, alphas_file = make_context(run_dir, 2)
    surrogate, full = load_matrix_elements(context)
    step = ms.WeightingStep(
        [full], freeze_max_weight_after=1000, max_overweight_truncation=0.05
    )
    groups = [ms.WeightingGroup([0, 1], [step])]
    channels = [
        ms.ChannelEventGenerator(
            contexts=[context],
            integrand=make_integrand(alphas_file, surrogate, pt_min),
            event_file=os.path.join(run_dir, f"events.{i}.npy"),
            weight_file=os.path.join(run_dir, f"weights.{i}.npy"),
            config=config,
            subprocess_index=0,
            name=str(i),
            histograms=None,
        )
        for i, pt_min in enumerate((20.0, 60.0))
    ]
    generator = ms.EventGenerator(
        [context], channels, seed=1, config=config, groups=groups
    )
    generator.survey()
    generator.fix_max_weights()
    globals_file = os.path.join(run_dir, "globals.tar")
    context.save_globals(globals_file)
    return {
        "channels": [channel.to_json(True) for channel in channels],
        "groups": ms.weighting_groups_to_json(groups),
        "globals": globals_file,
        "max_weights": [
            (channel.max_weight(), channel.step_max_weight(1)) for channel in channels
        ],
        "status": generator.status(),
    }


def load_gridpack(gridpack, thread_count, run_dir=None):
    context = ms.Context(ms.cpu_device(), thread_count)
    context.load_globals(gridpack["globals"])
    load_matrix_elements(context)
    config = make_config()
    channels = [
        ms.ChannelEventGenerator.load_json(
            channel_json,
            [context],
            event_file=os.path.join(run_dir, f"events.{i}.npy") if run_dir else "",
            weight_file=os.path.join(run_dir, f"weights.{i}.npy") if run_dir else "",
            config=config,
        )
        for i, channel_json in enumerate(gridpack["channels"])
    ]
    groups = ms.weighting_groups_from_json(gridpack["groups"])
    return context, channels, groups, config


def test_fixed_max_weights(fixed_gridpack, tmp_path):
    """fix_max_weights() freezes the maxima of all steps. Reloaded like in a
    gridpack, with the groups from JSON, the channels keep them."""
    max_weights = fixed_gridpack["max_weights"]
    assert all(m0 > 0 and m1 > 0 for m0, m1 in max_weights)
    for channel_json, (_, m1) in zip(fixed_gridpack["channels"], max_weights):
        assert json.loads(channel_json)["step_max_weights"] == [m1]

    context, channels, groups, config = load_gridpack(fixed_gridpack, 2, str(tmp_path))
    assert ms.weighting_groups_to_json(groups) == fixed_gridpack["groups"]
    generator = ms.EventGenerator(
        [context], channels, seed=2, config=config, groups=groups
    )
    generator.generate()
    assert [
        (channel.max_weight(), channel.step_max_weight(1)) for channel in channels
    ] == max_weights
    status, new_status = fixed_gridpack["status"], generator.status()
    assert new_status.count_unweighted == config.target_count
    assert abs(status.mean - new_status.mean) < 4 * np.hypot(
        status.error, new_status.error
    )


def make_stream(fixed_gridpack, thread_count=4, seed=5):
    context, channels, groups, config = load_gridpack(fixed_gridpack, thread_count)
    return ms.EventStream(
        context, channels, seed, lhe_completer(), config, groups=groups
    )


def stream_batches(stream, batch_sizes):
    batches = [stream.next_batch(size) for size in batch_sizes]
    return {key: np.concatenate([b[key] for b in batches]) for key in batches[0]}


def test_event_stream_reproducible(fixed_gridpack):
    reference = stream_batches(make_stream(fixed_gridpack, 4), [3000])
    for thread_count, batch_sizes in [(1, [1000, 2000]), (3, [1, 999, 2000])]:
        events = stream_batches(make_stream(fixed_gridpack, thread_count), batch_sizes)
        for key, values in reference.items():
            np.testing.assert_array_equal(events[key], values, err_msg=key)


def test_event_stream_cross_section(fixed_gridpack):
    """The streamed events have unit weights apart from overweights, and the
    integral after the step matches the generator's."""
    stream = make_stream(fixed_gridpack)
    batch = stream.next_batch(5000)
    weights = batch["weight"]
    assert np.all(weights >= 1.0)
    assert np.mean(weights == 1.0) > 0.9
    status, stream_status = fixed_gridpack["status"], stream.status()
    assert abs(status.mean - stream_status.mean) < 4 * np.hypot(
        status.error, stream_status.error
    )


def test_event_stream_status_without_inactive_channels(fixed_gridpack):
    """A channel without fixed step maxima returns no events, so it does not
    count in the cross section of the stream."""
    gridpack = dict(fixed_gridpack)
    channel_json = json.loads(gridpack["channels"][1])
    del channel_json["step_max_weights"]
    gridpack["channels"] = [gridpack["channels"][0], json.dumps(channel_json)]
    stream = make_stream(gridpack)
    integral = json.loads(gridpack["channels"][0])["integral"]
    status = stream.status()
    assert status.mean == pytest.approx(integral["mean"], rel=1e-12)
    assert status.error == pytest.approx(integral["error"], rel=1e-12)
    assert stream.channel_probabilities()[1] == 0.0
