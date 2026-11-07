"""Unit tests for compute_generation_batch_event_count, the pure function that
sizes a channel's next steady-state generation batch (see
generator_data.hpp/.cpp). The result is a weighted event count in nominal
batch units; device-specific job splitting happens later, at dispatch time.
"""

import math

import madspace as ms


def make_config():
    config = ms.GeneratorConfig()
    config.finish_remaining_fraction = 0.05
    config.max_batch_fraction = 0.6
    config.batch_overshoot_sigma = 1.0
    return config


def batch_events(
    count_target,
    count_unweighted,
    count_requested,
    cross_section_count,
    cross_section_rel_error,
    config,
):
    return ms.compute_generation_batch_event_count(
        count_target,
        count_unweighted,
        count_requested,
        cross_section_count,
        cross_section_rel_error,
        config,
    )


def test_confident_channel_hits_cap():
    # A converged channel far from its target steps up to the max_batch_fraction cap.
    config = make_config()
    events = batch_events(100000, 0.0, 0, 10000, 0.001, config)
    # efficiency is 1 for count_requested == 0
    expected = math.ceil(0.6 * 100000)
    assert events == expected


def test_uncertain_channel_is_throttled():
    # A large cross-section relative error lowers the safe target, so the batch is smaller.
    config = make_config()
    confident_events = batch_events(100000, 0.0, 0, 10000, 0.001, config)
    uncertain_events = batch_events(100000, 0.0, 0, 10000, 0.5, config)
    assert uncertain_events < confident_events


def test_no_data_is_cautious_not_full_target():
    # Without cross-section data the safe target is zero, which must give a small
    # exploratory batch, neither "done" nor the full target.
    config = make_config()
    no_data_events = batch_events(100000, 0.0, 0, 0, 0.0, config)
    confident_events = batch_events(100000, 0.0, 0, 10000, 0.001, config)
    assert 0 < no_data_events < confident_events


def test_few_events_left_finishes_outright():
    # A remaining deficit below finish_remaining_fraction of the target is requested in full.
    config = make_config()
    events = batch_events(100000, 97000.0, 0, 10000, 0.5, config)
    assert events == 3000


def test_mid_generation_is_meaningfully_throttled():
    # Between the cap and finish-outright regimes the batch is nonzero and below the deficit.
    config = make_config()
    events = batch_events(100000, 50000.0, 0, 10000, 0.3, config)
    assert 0 < events < 50000


def test_always_at_least_one_event():
    # The result is at least 1 even when the safe target is zero.
    config = make_config()
    events = batch_events(100000, 0.0, 0, 0, 0.0, config)
    assert events >= 1


def test_never_overshoots_true_remaining():
    # The requested event count must never exceed the true remaining deficit.
    config = make_config()
    config.max_batch_fraction = 1.0
    config.batch_overshoot_sigma = 0.0
    events = batch_events(100000, 99500.0, 0, 10000, 0.001, config)
    assert events <= 500


def test_zero_efficiency_does_not_diverge():
    # Zero unweighted events out of count_requested > 0 must not divide by zero
    # or inflate the request without bound.
    config = make_config()
    events = batch_events(100000, 0.0, 5000, 10000, 0.5, config)
    assert events < 100000 * 5000
