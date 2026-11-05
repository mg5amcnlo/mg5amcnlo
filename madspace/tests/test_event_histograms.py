"""Event-sample histograms of the event weight itself.

`EventHistogramSpec.from_weight` histograms the weight an event was written
with, divided by the reference weight given to `EventHistograms` (the cross
section, i.e. the mean weight): a fully unweighted sample is a spike at 1
whatever the process, and a partially unweighted one shows its spread. Unlike
the observables, this needs no momenta and no PDF, so it lives here rather
than in test_systematics.py.
"""

import json

import numpy as np
import pytest
from pytest import approx

import madspace as ms

# host-side: EventHistograms fills from python lists on an explicit CPU context
pytestmark = pytest.mark.cpu_only


def random_momenta(n, seed=5):
    """2 -> 2 massless momenta (E px py pz), incoming along z."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        e1, e2 = rng.uniform(50, 800, 2)
        p_in = [[e1, 0.0, 0.0, e1], [e2, 0.0, 0.0, -e2]]
        s = 4 * e1 * e2
        pcm = np.sqrt(s) / 2
        theta, phi = np.arccos(rng.uniform(-1, 1)), rng.uniform(0, 2 * np.pi)
        px = pcm * np.sin(theta) * np.cos(phi)
        py = pcm * np.sin(theta) * np.sin(phi)
        pz = pcm * np.cos(theta)
        beta = (e1 - e2) / (e1 + e2)
        gamma = 1 / np.sqrt(1 - beta**2)

        def boost(e, pz):
            return gamma * (e + beta * pz), gamma * (pz + beta * e)

        ea, pza = boost(pcm, pz)
        eb, pzb = boost(pcm, -pz)
        out.append(p_in + [[ea, px, py, pza], [eb, -px, -py, pzb]])
    return out


def test_weight_histogram():
    """A from_weight histogram bins the event weight over the reference one,
    so an unweighted sample is a spike at 1 whatever the cross section."""
    n = 40
    momenta = random_momenta(n, seed=11)
    ctx = ms.Context(device=ms.cpu_device(), thread_count=1)
    pids = [2, 2, 2, 2]
    obs = ms.ObservableValues(
        [
            ms.Observable(pids, observable="sqrt_s", select_pids=[], name="sqrt_s"),
        ]
    )
    specs = [
        ms.EventHistogramSpec("sqrt_s", 0.0, 2000.0, 8),
        ms.EventHistogramSpec("weight", 0.0, 5.0, 10, from_weight=True),
    ]
    # half the events carry twice the reference weight
    weight = np.full(n, 3.0)
    weight[::2] = 6.0
    mean = float(weight.mean())
    hists = ms.EventHistograms(
        ctx, specs, [ms.SubprocessObservables(obs, 4)], reference_weight=mean
    )
    hists.fill(list(weight), [0] * n, momenta, [[] for _ in range(n)])
    data = json.loads(hists.to_json())
    assert [h["name"] for h in data] == ["sqrt_s", "weight"]
    w_hist = data[1]
    # every histogram of the same sample integrates to the same cross section
    assert sum(w_hist["bin_values"]) == approx(sum(data[0]["bin_values"]))
    # w/<w> is 2/3 and 4/3 here: bins [0.5, 1.0) and [1.0, 1.5), i.e. 2 and 3
    # once the underflow bin is counted in
    filled = {i: v for i, v in enumerate(w_hist["bin_values"]) if v != 0}
    assert sorted(filled) == [2, 3]
    assert filled[2] == approx(3.0 * (n / 2) / n)
    assert filled[3] == approx(6.0 * (n / 2) / n)


def test_weight_histogram_alone():
    """Only the weight is histogrammed: no observable runtime is needed."""
    n = 10
    momenta = random_momenta(n, seed=12)
    ctx = ms.Context(device=ms.cpu_device(), thread_count=1)
    specs = [ms.EventHistogramSpec("weight", 0.0, 5.0, 10, from_weight=True)]
    hists = ms.EventHistograms(ctx, specs, [None], reference_weight=2.0)
    hists.fill([2.0] * n, [0] * n, momenta, [[] for _ in range(n)])
    data = json.loads(hists.to_json())
    values = data[0]["bin_values"]
    assert sum(values) == approx(2.0)
    assert values[3] == approx(2.0)  # the bin holding w/<w> = 1
