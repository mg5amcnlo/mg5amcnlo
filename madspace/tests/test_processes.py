import json
import math
import os
from glob import glob

import pytest
import torch
from madspace_testing import generator, rand, randint

import madspace as ms

BATCH_SIZE = 1000
CM_ENERGY = 13000.0
rng = generator(1234)


def assert_approx(actual, expected, rel=1e-6, abs=1e-12):
    """pytest.approx-like elementwise comparison of tensors."""
    torch.testing.assert_close(actual, expected, rtol=rel, atol=abs)


def load_processes():
    proc_files = glob(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data", "*.json")
    )
    ret = []
    for file in proc_files:
        proc_name = os.path.splitext(os.path.basename(file))[0]
        with open(file) as f:
            diagrams = json.load(f)
        ret.extend(
            pytest.param(diagram, id=f"{proc_name}-{i}")
            for i, diagram in enumerate(diagrams)
        )
    return ret


@pytest.fixture(params=load_processes())
def process(request):
    return request.param


@pytest.fixture
def mapping(process):
    diagram = ms.Diagram(
        process["incoming_masses"],
        process["outgoing_masses"],
        [ms.Propagator(*prop) for prop in process["propagators"]],
        process["vertices"],
    )
    topology = ms.Topology(diagram)
    return ms.PhaseSpaceMapping(
        topology, CM_ENERGY, permutations=process["permutations"]
    )


@pytest.fixture
def cut_mapping(process):
    # a mild pt cut keeps the events away from extremely soft gluons, for which the
    # inverse of the t-channel invariants is numerically ill-conditioned
    pids = [21, 21] + [6 if mass > 0 else 21 for mass in process["outgoing_masses"]]
    O = ms.Observable
    cuts = ms.Cuts([ms.CutItem(O(pids, O.obs_pt, [O.jet_pids]), min=1.0)])
    diagram = ms.Diagram(
        process["incoming_masses"],
        process["outgoing_masses"],
        [ms.Propagator(*prop) for prop in process["propagators"]],
        process["vertices"],
    )
    return ms.PhaseSpaceMapping(
        ms.Topology(diagram), CM_ENERGY, cuts=cuts, permutations=process["permutations"]
    )


@pytest.fixture
def masses(process):
    return process["incoming_masses"] + process["outgoing_masses"]


@pytest.fixture
def permutation_count(process):
    return len(process["permutations"])


def test_process_masses(mapping, masses, permutation_count):
    r = rand((BATCH_SIZE, mapping.random_dim()), generator=rng)
    condition = (
        []
        if permutation_count <= 1
        else [
            randint(
                0, permutation_count, (BATCH_SIZE,), generator=rng, dtype=torch.int32
            )
        ]
    )
    p_ext, x1, x2, det = mapping.map_forward([r], condition)
    m_ext_true = torch.tensor(masses, dtype=torch.float64).expand(BATCH_SIZE, -1)
    m_ext = torch.sqrt(
        torch.clamp(p_ext[:, :, 0] ** 2 - torch.sum(p_ext[:, :, 1:] ** 2, dim=2), min=0)
    )
    assert_approx(m_ext, m_ext_true, abs=1e-3, rel=1e-3)


def test_process_incoming(mapping, masses, permutation_count):
    r = rand((BATCH_SIZE, mapping.random_dim()), generator=rng)
    condition = (
        []
        if permutation_count <= 1
        else [
            randint(
                0, permutation_count, (BATCH_SIZE,), generator=rng, dtype=torch.int32
            )
        ]
    )
    p_ext, x1, x2, det = mapping.map_forward([r], condition)
    zeros = torch.zeros(BATCH_SIZE)
    p_a = p_ext[:, 0]
    p_b = p_ext[:, 1]
    e_beam = 0.5 * CM_ENERGY

    assert_approx(p_a[:, 0], p_a[:, 3])
    assert_approx(p_b[:, 0], -p_b[:, 3])
    assert_approx(p_a[:, 1], zeros)
    assert_approx(p_a[:, 2], zeros)
    assert_approx(p_b[:, 1], zeros)
    assert_approx(p_b[:, 2], zeros)
    assert torch.all(x1 >= 0) and torch.all(x1 <= 1)
    assert torch.all(x2 >= 0) and torch.all(x2 <= 1)
    assert_approx(p_a[:, 0], e_beam * x1)
    assert_approx(p_b[:, 0], e_beam * x2)


def test_process_momentum_conservation(mapping, masses, permutation_count):
    r = rand((BATCH_SIZE, mapping.random_dim()), generator=rng)
    condition = (
        []
        if permutation_count <= 1
        else [
            randint(
                0, permutation_count, (BATCH_SIZE,), generator=rng, dtype=torch.int32
            )
        ]
    )
    p_ext, x1, x2, det = mapping.map_forward([r], condition)
    p_in = torch.sum(p_ext[:, :2], dim=1)
    p_out = torch.sum(p_ext[:, 2:], dim=1)

    assert_approx(p_out, p_in, rel=1e-6, abs=1e-10)


def test_process_inverse(cut_mapping, masses, permutation_count):
    r = rand((BATCH_SIZE, cut_mapping.random_dim()), generator=rng)
    condition = (
        []
        if permutation_count <= 1
        else [
            randint(
                0, permutation_count, (BATCH_SIZE,), generator=rng, dtype=torch.int32
            )
        ]
    )
    *map_out, det = cut_mapping.map_forward([r], condition)
    r_inv, det_inv = cut_mapping.map_inverse(map_out, condition)
    # events removed by the cuts (zero weight) are outside the phase space
    passed = det > 0
    assert torch.any(passed)
    assert_approx(r_inv[passed], r[passed], abs=1e-3, rel=1e-3)
    assert_approx(det_inv[passed], 1 / det[passed], rel=1e-3)
