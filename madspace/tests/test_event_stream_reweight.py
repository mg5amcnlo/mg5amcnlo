"""EventStream with a reweighting step after the first unweighting, like the
default [amplicol] mode = "reweight": the streamed events carry the ratio of the
step matrix element to the integrand one, not unit weights. Uses the setup of
test_weighting_steps.
"""

import os

import numpy as np
import pytest
from test_event_stream import CM_ENERGY, lhe_completer, make_config, make_context
from test_integrand_me_outputs import python_matrix_element
from test_weighting_steps import FULL_ME_OUTPUTS, generate, make_integrand, true_me

import madspace as ms

PT_MINS = (20.0, 60.0)


def biased_me(momenta):
    """Does not average to the true one, so that the ratio changes the norm."""
    return true_me(momenta) * (0.6 + 0.8 * momenta[:, 2, 0] / CM_ENERGY)


def full_me_with_indices(momenta):
    zeros = np.zeros(len(momenta), np.int32)
    return true_me(momenta), zeros, zeros, zeros


def load_matrix_elements(context):
    return (
        python_matrix_element(context, biased_me),
        python_matrix_element(context, full_me_with_indices, FULL_ME_OUTPUTS),
    )


@pytest.fixture(scope="module")
def reweight_gridpack(tmp_path_factory):
    """Two grouped channels with fixed maxima and a reweighting step."""
    config = make_config()
    run_dir = str(tmp_path_factory.mktemp("reweight"))
    context, alphas_file = make_context(run_dir, 2)
    biased, full = load_matrix_elements(context)
    groups = [ms.WeightingGroup([0, 1], [ms.WeightingStep([full], mode="reweight")])]
    channels = [
        ms.ChannelEventGenerator(
            contexts=[context],
            integrand=make_integrand(alphas_file, biased, pt_min),
            event_file=os.path.join(run_dir, f"events.{i}.npy"),
            weight_file=os.path.join(run_dir, f"weights.{i}.npy"),
            config=config,
            subprocess_index=0,
            name=str(i),
            histograms=None,
        )
        for i, pt_min in enumerate(PT_MINS)
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
    }


def make_stream(gridpack, thread_count=4, seed=5):
    context = ms.Context(ms.cpu_device(), thread_count)
    context.load_globals(gridpack["globals"])
    load_matrix_elements(context)
    config = make_config()
    channels = [
        ms.ChannelEventGenerator.load_json(
            channel_json, [context], event_file="", weight_file="", config=config
        )
        for channel_json in gridpack["channels"]
    ]
    groups = ms.weighting_groups_from_json(gridpack["groups"])
    return ms.EventStream(
        context, channels, seed, lhe_completer(), config, groups=groups
    )


def test_event_stream_reweight(reweight_gridpack, tmp_path):
    stream = make_stream(reweight_gridpack)
    batch = stream.next_batch(20000)
    weights = batch["weight"]
    # the ratio of the full to the biased matrix element, times the first-pass
    # weight (1, or more for the overweights)
    ratio = 1.0 / (0.6 + 0.8 * batch["energy"][:, 2] / CM_ENERGY)
    assert np.mean(np.isclose(weights, ratio, rtol=1e-6)) > 0.9
    assert np.all(weights >= ratio * (1.0 - 1e-6))

    # mean weight x first-pass cross section = full cross section
    first_pass, _, _ = generate(
        str(tmp_path / "biased"),
        biased_me,
        pt_mins=PT_MINS,
        grouped=(0, 1),
        target_count=1000,
    )
    full, full_events, _ = generate(
        str(tmp_path / "full"),
        true_me,
        pt_mins=PT_MINS,
        grouped=(0, 1),
        target_count=20000,
    )
    status = stream.status()
    mean_weight_error = np.std(weights) / np.sqrt(len(weights))
    xs_error = np.hypot(
        mean_weight_error * first_pass.mean, np.mean(weights) * first_pass.error
    )
    assert abs(np.mean(weights) * first_pass.mean - full.mean) < 4 * np.hypot(
        xs_error, full.error
    )
    assert abs(status.mean - full.mean) < 4 * np.hypot(status.error, full.error)

    # the weighted events follow the full matrix element
    energies = [
        np.sum(w * e) / np.sum(w)
        for w, e in (
            (weights, batch["energy"][:, 2]),
            (full_events["weight"], full_events["part3_energy"]),
        )
    ]
    assert energies[0] == pytest.approx(energies[1], rel=0.01)
