"""The momentum fractions of a PhaseSpaceMapping are a batch, also where it
samples none (leptonic, a decay).

There x1 = x2 = 1, but like every output of a mapping they have the batch
dimension their type declares. Channels of a MultiChannelMapping (and of a
MultiChannelFunction) are concatenated output by output, which needs it:
leptonic and decay channels used to fail with "Argument must have batch
dimension".
"""

import numpy as np
import pytest
import torch
from madspace_testing import to_numpy

import madspace as ms

BATCH = 300


@pytest.mark.parametrize("beam_rapidity", [0.0, 0.3], ids=["symmetric", "boosted"])
def test_fractions_have_the_batch_dimension(beam_rapidity):
    mapping = ms.PhaseSpaceMapping(
        [0.0, 0.0, 0.0, 0.0, 0.0],
        1000.0,
        leptonic=True,
        beam_rapidity=beam_rapidity,
        mode=ms.PhaseSpaceMapping.rambo,
    )
    r = np.random.default_rng(1).random((BATCH, mapping.random_dim()))
    momenta, x1, x2, det = mapping.map_forward([torch.as_tensor(r)])
    for x in (x1, x2):
        x = to_numpy(x)
        assert x.shape == (BATCH,)
        np.testing.assert_array_equal(x, 1.0)
    # the inverse takes them as it took the constants
    r_inv, det_inv = mapping.map_inverse([momenta, x1, x2])
    ok = to_numpy(det) > 0
    np.testing.assert_allclose(to_numpy(r_inv)[ok], r[ok], rtol=0, atol=1e-9)


def test_leptonic_channels():
    masses = [0.0, 0.0, 0.0, 0.0, 0.0]
    channels = [
        ms.PhaseSpaceMapping(
            masses, 1000.0, leptonic=True, mode=ms.PhaseSpaceMapping.rambo
        ),
        ms.PhaseSpaceMapping(
            masses, 1000.0, leptonic=True, mode=ms.PhaseSpaceMapping.propagator
        ),
    ]
    mapping = ms.MultiChannelMapping(channels)
    r = np.random.default_rng(2).random((2 * BATCH, channels[0].random_dim()))
    # a list of batch sizes, read on the host
    sizes = np.full(2, BATCH, dtype=np.int64)
    outputs = mapping.map_forward([torch.as_tensor(r)], [sizes])
    for k, channel in enumerate(channels):
        part = slice(k * BATCH, (k + 1) * BATCH)
        ref = channel.map_forward([torch.as_tensor(r[part])])
        for ours, theirs in zip(outputs, ref):
            np.testing.assert_array_equal(to_numpy(ours)[part], to_numpy(theirs))


def decay(invariant_power):
    """t > b w+ (w+ > e+ ve), the W as a propagator, its virtuality sampled
    with the given power"""
    diagram = ms.Diagram(
        [173.0],
        [4.7, 0.0, 0.0],
        [ms.Propagator(mass=80.4, width=2.0, pdg_id=24)],
        [["i0", "o0", "p0"], ["p0", "o1", "o2"]],
    )
    return ms.PhaseSpaceMapping(
        ms.Topology(diagram), 173.0, invariant_power=invariant_power
    )


def test_decay_channels():
    channels = [decay(0.8), decay(0.0)]
    r = np.random.default_rng(3).random((2 * BATCH, channels[0].random_dim()))
    ref = channels[0].map_forward([torch.as_tensor(r[:BATCH])])
    for x in ref[1:3]:
        x = to_numpy(x)
        assert x.shape == (BATCH,)
        np.testing.assert_array_equal(x, 1.0)
    sizes = np.full(2, BATCH, dtype=np.int64)
    outputs = ms.MultiChannelMapping(channels).map_forward(
        [torch.as_tensor(r)], [sizes]
    )
    for k, channel in enumerate(channels):
        part = slice(k * BATCH, (k + 1) * BATCH)
        ref = channel.map_forward([torch.as_tensor(r[part])])
        for ours, theirs in zip(outputs, ref):
            np.testing.assert_array_equal(to_numpy(ours)[part], to_numpy(theirs))
