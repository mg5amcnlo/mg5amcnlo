# SPDX-License-Identifier: 0BSD

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal

import pytest

from pyamplicol.api.errors import ArtifactError
from pyamplicol.runtime._color_topology_exact import (
    apply_exact_color_replay_input_mapping,
    parse_exact_color_topology_replay,
    reduce_exact_color_topology_replay,
)


def _fixture(*, repeated: bool = False) -> tuple[dict[str, object], dict[str, object]]:
    mappings = [
        {
            "label_permutation": [],
            "group_routes": [
                {"source_group_id": 10, "target_group_id": 0, "factor": [1, 0]}
            ],
        },
        {
            "label_permutation": [
                {"representative_label": 1, "sector_label": 2},
                {"representative_label": 2, "sector_label": 1},
            ],
            "group_routes": [
                {"source_group_id": 11, "target_group_id": 1, "factor": [1, 0]}
            ],
        },
    ]
    contraction: dict[str, object] = {
        "supported": True,
        "reason": None,
        "group_count": 2,
        "includes_color_factor": True,
        "entries": [],
        "repeated_block": None,
    }
    if repeated:
        contraction["repeated_block"] = {
            "component_count": 2,
            "component_group_ids": [0, 1],
            "entries": [
                {
                    "left_group_index": 0,
                    "right_group_index": 0,
                    "weight": [1, 0],
                    "symmetry_factor": 1,
                }
            ],
        }
    else:
        contraction["entries"] = [
            {
                "left_group_id": 0,
                "right_group_id": 0,
                "weight": [1, 0],
                "symmetry_factor": 1,
            },
            {
                "left_group_id": 1,
                "right_group_id": 1,
                "weight": [1, 0],
                "symmetry_factor": 1,
            },
            {
                "left_group_id": 0,
                "right_group_id": 1,
                "weight": [1, 0],
                "symmetry_factor": 2,
            },
        ]
    proof_mappings = [
        [],
        [
            {"representative_label": 1, "sector_label": 2},
            {"representative_label": 2, "sector_label": 1},
        ],
    ]
    execution: dict[str, object] = {
        "runtime_schema": {
            "color_topology_replay": {
                "enabled": True,
                "mode": "external-label-permutation",
                "contract_version": 3,
                "color_accuracy": "full",
                "physical_sector_count": 2,
                "replayed_sector_count": 2,
                "materialized_sector_ids": [0, 1],
                "residual_sector_ids": [],
                "groups": [
                    {
                        "sector_permutations": [
                            {"label_permutation": mapping} for mapping in proof_mappings
                        ]
                    }
                ],
            },
            "amplitude_stage": {
                "stage_kind": "amplitude-roots",
                "output_count": 2,
                "roots": [
                    {"root_id": 0, "output_index": 0, "coherent_group_id": 10},
                    {"root_id": 1, "output_index": 1, "coherent_group_id": 11},
                ],
                "color_topology_replay": {
                    "contract_version": 1,
                    "physical_group_count": 2,
                    "physical_groups": [
                        {
                            "group_id": 0,
                            "helicities": [1, -1],
                            "color_sector_id": 0,
                            "color_word": [1, 2],
                            "helicity_weight": 2,
                        },
                        {
                            "group_id": 1,
                            "helicities": [1, -1],
                            "color_sector_id": 1,
                            "color_word": [2, 1],
                            "helicity_weight": 2,
                        },
                    ],
                    "mappings": mappings,
                },
                "color_contraction": contraction,
            },
        }
    }
    physics: dict[str, object] = {
        "color_accuracy": "full",
        "external_particles": [{"pdg": 1}, {"pdg": -1}],
        "helicities": [
            {"id": "h-plus", "values": [1, -1], "coefficient": 1},
            {"id": "h-minus", "values": [-1, 1], "coefficient": 1},
        ],
        "color_components": [{"id": "contracted:full"}],
    }
    return execution, physics


@pytest.mark.parametrize(
    ("repeated", "expected"),
    [(False, Decimal("4.5")), (True, Decimal("2.5"))],
)
def test_exact_color_replay_gathers_routes_and_contracts_physical_groups(
    repeated: bool, expected: Decimal
) -> None:
    execution, physics = _fixture(repeated=repeated)
    plan = parse_exact_color_topology_replay(execution, physics, None)
    assert plan is not None

    values, helicities, colors = reduce_exact_color_topology_replay(
        (
            ((Decimal(1), Decimal(0)), (Decimal(99), Decimal(0))),
            ((Decimal(99), Decimal(0)), (Decimal(2), Decimal(0))),
        ),
        plan,
        1,
        Decimal(1),
        None,
        None,
    )

    assert values == (((expected,), (expected,)),)
    assert helicities == ("h-plus", "h-minus")
    assert colors == ("contracted:full",)


def test_exact_color_replay_maps_representative_helicity_to_public_alias() -> None:
    execution, physics = _fixture()
    replay = execution["runtime_schema"]["amplitude_stage"]["color_topology_replay"]
    for group in replay["physical_groups"]:
        group["helicity_weight"] = 1

    plan = parse_exact_color_topology_replay(execution, physics, (1, 0))

    assert plan is not None
    assert plan.physical_group_members == (((1, Decimal(1)),),) * 2


def test_exact_color_replay_rejects_proof_and_route_mapping_disagreement() -> None:
    execution, physics = _fixture()
    malformed = deepcopy(execution)
    proof = malformed["runtime_schema"]["color_topology_replay"]
    proof["groups"][0]["sector_permutations"] = [{"label_permutation": []}]

    with pytest.raises(
        ArtifactError,
        match="proof mappings do not match the amplitude gather",
    ):
        parse_exact_color_topology_replay(malformed, physics, None)


def test_exact_color_replay_applies_mapping_in_representative_order() -> None:
    point = (
        (Decimal(10), Decimal(1), Decimal(2), Decimal(3)),
        (Decimal(20), Decimal(4), Decimal(5), Decimal(6)),
    )

    assert apply_exact_color_replay_input_mapping(point, ((0, 1), (1, 0))) == (
        point[1],
        point[0],
    )
