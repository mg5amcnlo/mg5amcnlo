# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pyamplicol.color import ColorGroupDescriptor
from pyamplicol.color.plan import (
    build_color_plan,
    build_color_topology_replay_certificate,
    color_topology_replay_partitions,
)
from pyamplicol.generation.runtime_amplitudes import (
    _color_topology_replay_amplitudes,
)
from pyamplicol.models.builtin.model import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


@pytest.mark.parametrize(
    ("n_final", "sector_count", "partition_count", "partition_size", "representatives"),
    [
        (4, 120, 5, 24, (0, 24, 30, 32, 33)),
        (5, 720, 6, 120, (0, 120, 144, 150, 152, 153)),
        (6, 5_040, 7, 720, (0, 720, 840, 864, 870, 872, 873)),
    ],
)
def test_full_color_pure_gluon_partitions_match_crossing_templates(
    n_final: int,
    sector_count: int,
    partition_count: int,
    partition_size: int,
    representatives: tuple[int, ...],
) -> None:
    process = build_process_ir(
        f"g g > {' '.join(['g'] * n_final)}",
        color_accuracy="full",
    )
    color_plan = build_color_plan(process, color_accuracy="full")
    partitions = color_topology_replay_partitions(color_plan)

    assert color_plan.sector_count == sector_count
    assert len(partitions) == partition_count
    assert (
        tuple(partition.representative_sector_id for partition in partitions)
        == representatives
    )
    assert {len(partition.active_sector_ids) for partition in partitions} == {
        partition_size
    }
    assert {
        sector_id
        for partition in partitions
        for sector_id in partition.active_sector_ids
    } == set(range(sector_count))

    initial_labels = {leg.label for leg in process.initial_legs}
    for partition in partitions:
        for permutation in partition.label_permutations:
            mapping = dict(permutation)
            assert {mapping[label] for label in initial_labels} == initial_labels


def test_existing_lc_runtime_wrapper_remains_lc_only() -> None:
    process = build_process_ir("g g > g g g g g", color_accuracy="full")
    color_plan = build_color_plan(process, color_accuracy="full")

    from pyamplicol.color.plan import lc_topology_replay_partitions

    assert lc_topology_replay_partitions(color_plan) == ()


def test_full_color_certificate_proves_six_n5_crossing_orbits() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("g g > g g g g g", color_accuracy="full")
    color_plan = build_color_plan(process, color_accuracy="full")

    certificate = build_color_topology_replay_certificate(color_plan, model)

    assert certificate is not None
    assert certificate.color_accuracy == "full"
    assert certificate.physical_sector_ids == tuple(range(720))
    assert certificate.materialized_sector_ids == (0, 120, 144, 150, 152, 153)
    assert certificate.residual_sector_ids == ()
    assert certificate.replayed_sector_count == 720
    assert all(
        partition.proof_algorithm
        == "canonical-model-contract-color-label-equivariance-v2"
        and partition.proof_digest is not None
        and len(partition.proof_digest) == 64
        and partition.signs == (1,) * 120
        for partition in certificate.partitions
    )


def test_n5_full_color_amplitude_gather_is_a_720_group_bijection() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("g g > g g g g g", color_accuracy="full")
    color_plan = build_color_plan(process, color_accuracy="full")
    certificate = build_color_topology_replay_certificate(color_plan, model)
    assert certificate is not None
    helicity_key = tuple((leg.label, 21, 0, 1, 1) for leg in process.legs)
    descriptors = tuple(
        ColorGroupDescriptor(
            group_id=group_id,
            helicity_key=helicity_key,
            sector_id=sector_id,
            word=tuple(
                color_plan.sector(sector_id).word_labels  # type: ignore[union-attr]
            ),
            helicity_weight=1.0,
        )
        for group_id, sector_id in enumerate(certificate.materialized_sector_ids)
    )
    dag = SimpleNamespace(
        color_topology_replay=certificate,
        color_plan=color_plan,
        process=process,
    )

    replay = _color_topology_replay_amplitudes(dag, descriptors)

    assert replay is not None
    assert len(replay.physical_descriptors) == 720
    assert len(replay.mappings) == 120
    assert {len(mapping.group_routes) for mapping in replay.mappings} == {6}
    assert sorted(
        route.target_group_id
        for mapping in replay.mappings
        for route in mapping.group_routes
    ) == list(range(720))


@pytest.mark.parametrize("color_accuracy", ["nlc", "full"])
def test_n3_amplitude_gather_order_matches_replay_proof(
    color_accuracy: str,
) -> None:
    model = BuiltinSMModel()
    process = build_process_ir(
        "g g > g g g",
        color_accuracy=color_accuracy,
    )
    color_plan = build_color_plan(process, color_accuracy=color_accuracy)
    certificate = build_color_topology_replay_certificate(color_plan, model)
    assert certificate is not None
    helicity_key = tuple((leg.label, 21, 0, 1, 1) for leg in process.legs)
    descriptors = tuple(
        ColorGroupDescriptor(
            group_id=group_id,
            helicity_key=helicity_key,
            sector_id=sector_id,
            word=tuple(
                color_plan.sector(sector_id).word_labels  # type: ignore[union-attr]
            ),
            helicity_weight=1.0,
        )
        for group_id, sector_id in enumerate(certificate.materialized_sector_ids)
    )
    dag = SimpleNamespace(
        color_topology_replay=certificate,
        color_plan=color_plan,
        process=process,
    )

    replay = _color_topology_replay_amplitudes(dag, descriptors)

    assert replay is not None
    proof_mappings = tuple(
        sorted(
            {
                tuple(
                    sorted(
                        (int(left), int(right))
                        for left, right in permutation
                        if int(left) != int(right)
                    )
                )
                for partition in certificate.partitions
                for permutation in partition.label_permutations
            }
        )
    )
    assert tuple(
        mapping.label_permutation for mapping in replay.mappings
    ) == proof_mappings


def test_color_accuracy_is_bound_into_generic_replay_proof() -> None:
    model = BuiltinSMModel()
    certificates = []
    for color_accuracy in ("nlc", "full"):
        process = build_process_ir(
            "g g > g g g",
            color_accuracy=color_accuracy,
        )
        certificate = build_color_topology_replay_certificate(
            build_color_plan(process, color_accuracy=color_accuracy),
            model,
        )
        assert certificate is not None
        certificates.append(certificate)

    assert [partition.proof_digest for partition in certificates[0].partitions] != [
        partition.proof_digest for partition in certificates[1].partitions
    ]
