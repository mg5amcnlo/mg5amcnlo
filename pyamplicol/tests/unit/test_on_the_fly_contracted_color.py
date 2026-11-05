# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib

import pytest

from pyamplicol.color import build_color_plan
from pyamplicol.generation import service as service_module
from pyamplicol.generation.recurrence_physics import (
    build_on_the_fly_color_contraction,
)
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.runtime.recurrence_exact._color import (
    _decode_recurrence_color_contraction,
)


@pytest.mark.parametrize("color_accuracy", ("nlc", "full"))
def test_contracted_open_lines_publish_one_group_per_structural_owner(
    color_accuracy: str,
) -> None:
    process = build_process_ir(
        "u u~ > u u~",
        color_accuracy=color_accuracy,
    )
    color_plan = build_color_plan(
        process,
        color_accuracy=color_accuracy,
        fold_trace_reflections=False,
    )

    contraction, owner_by_sector, owner_sector_ids = (
        build_on_the_fly_color_contraction(color_plan)
    )
    encoded = service_module._build_on_the_fly_contracted_color_payload_v1(
        color_plan
    )
    decoded = _decode_recurrence_color_contraction(encoded.payload)

    assert color_plan.sector_count == 4
    assert color_plan.trace_reflections_folded is False
    assert owner_by_sector == (0, 0, 2, 2)
    assert owner_sector_ids == (0, 2)
    assert contraction.repeated_block is None
    assert contraction.includes_color_factor is True
    assert decoded.storage == "expanded"
    assert decoded.component_count == 1
    assert decoded.group_count == decoded.destination_count == 2
    assert decoded.group_sector_ids == owner_sector_ids
    assert decoded.group_component_ids == (0, 0)
    assert decoded.owner_by_sector == owner_by_sector
    assert decoded.destination_by_group == encoded.destination_by_group == (0, 1)
    assert encoded.summary == {
        "abi": "pyamplicol-recurrence-color-contraction-v3",
        "color_accuracy": color_accuracy,
        "storage": "expanded",
        "includes_color_factor": True,
        "group_count": 2,
        "sector_count": 4,
        "active_sector_count": 2,
        "component_count": 1,
        "destination_count": 2,
        "entry_count": len(contraction.entries),
        "logical_entry_count": len(contraction.entries),
        "semantic_digest": hashlib.sha256(encoded.payload).hexdigest(),
        "factorization": None,
    }


@pytest.mark.parametrize("color_accuracy", ("nlc", "full"))
def test_contracted_trace_plan_retains_every_orientation(
    color_accuracy: str,
) -> None:
    process = build_process_ir(
        "g g > g g",
        color_accuracy=color_accuracy,
    )
    color_plan = build_color_plan(
        process,
        color_accuracy=color_accuracy,
        fold_trace_reflections=False,
    )
    encoded = service_module._build_on_the_fly_contracted_color_payload_v1(
        color_plan
    )
    decoded = _decode_recurrence_color_contraction(encoded.payload)

    assert color_plan.sector_count == 6
    assert decoded.group_count == 6
    assert decoded.group_sector_ids == tuple(range(6))
    assert decoded.owner_by_sector == tuple(range(6))
    assert decoded.destination_by_group == tuple(range(6))
    assert encoded.summary["active_sector_count"] == 6
