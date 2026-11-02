# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyamplicol.generation.service as service_module
from pyamplicol.generation.artifact_writer import (
    CompiledProcessArtifact,
    build_api_validation_points,
)
from pyamplicol.generation.contracts import RuntimeExpressionSchema
from pyamplicol.generation.validation import ValidationPointRecord
from pyamplicol.models.builtin.process_ir import build_process_ir


def test_api_validation_points_include_aliases_in_alias_order() -> None:
    vectors = (
        (20.0, 1.0, 2.0, 3.0),
        (21.0, 4.0, 5.0, 6.0),
        (22.0, 7.0, 8.0, 9.0),
        (23.0, 10.0, 11.0, 12.0),
    )
    validation = ValidationPointRecord(
        process_id="base",
        process="d d~ > z g",
        seed=7,
        particles=tuple(
            (pdg, vector) for pdg, vector in zip((1, -1, 23, 21), vectors, strict=True)
        ),
    )
    runtime_schema = RuntimeExpressionSchema.from_mapping(
        {
            "amplitude_stage": {},
            "current_storage": {},
            "momentum_slots": [],
            "parameter_layout": {},
            "stages": [],
            "value_storage": {},
        }
    )
    process = CompiledProcessArtifact(
        process_id="base",
        expression="d d~ > z g",
        color_accuracy="lc",
        external_pdgs=(1, -1, 23, 21),
        aliases=(
            {
                "id": "crossed",
                "expression": "d d~ > g z",
                "external_pdgs": [1, -1, 21, 23],
                "external_permutation": [0, 1, 3, 2],
            },
        ),
        runtime_schema=runtime_schema,
        stage_manifest={},
        model_parameter_evaluator=None,
        dag_summary={},
        evaluator_root=Path("."),
        validation_point=validation,
        generation_filters={},
    )

    points = build_api_validation_points((process,))

    assert points == {
        "base": vectors,
        "crossed": (vectors[0], vectors[1], vectors[3], vectors[2]),
    }


def test_api_validation_points_scatter_non_self_inverse_alias_permutation() -> None:
    vectors = tuple((float(index), 0.0, 0.0, 0.0) for index in range(5))
    validation = ValidationPointRecord(
        process_id="base",
        process="d d~ > z g g",
        seed=11,
        particles=tuple(
            (pdg, vector)
            for pdg, vector in zip(
                (1, -1, 23, 21, 21),
                vectors,
                strict=True,
            )
        ),
    )
    runtime_schema = RuntimeExpressionSchema.from_mapping(
        {
            "amplitude_stage": {},
            "current_storage": {},
            "momentum_slots": [],
            "parameter_layout": {},
            "stages": [],
            "value_storage": {},
        }
    )
    process = CompiledProcessArtifact(
        process_id="base",
        expression="d d~ > z g g",
        color_accuracy="lc",
        external_pdgs=(1, -1, 23, 21, 21),
        aliases=(
            {
                "id": "cycled",
                "expression": "d d~ > g z g",
                "external_pdgs": [1, -1, 21, 23, 21],
                "external_permutation": [0, 1, 3, 4, 2],
            },
        ),
        runtime_schema=runtime_schema,
        stage_manifest={},
        model_parameter_evaluator=None,
        dag_summary={},
        evaluator_root=Path("."),
        validation_point=validation,
        generation_filters={},
    )

    points = build_api_validation_points((process,))

    assert points["cycled"] == (
        vectors[0],
        vectors[1],
        vectors[4],
        vectors[2],
        vectors[3],
    )


def test_alias_identity_uses_public_external_order() -> None:
    process = build_process_ir("d d~ > z g a", color_accuracy="lc")

    expression, pdgs = service_module._permuted_process_identity(
        process,
        (0, 1, 3, 4, 2),
    )

    assert expression == "d d~ > a z g"
    assert pdgs == (1, -1, 22, 23, 21)


def test_recurrence_validation_selector_cases_are_axis_bounded() -> None:
    physics = SimpleNamespace(
        color_flows=tuple(
            SimpleNamespace(id=f"flow:{index}") for index in range(720)
        ),
        helicities=tuple(
            SimpleNamespace(
                id=f"h:{index}",
                structural_zero=index >= 384,
            )
            for index in range(768)
        ),
    )

    cases = service_module._recurrence_validation_selector_cases(physics)

    assert cases == (
        ("flow flow:0", {"color_flows": ("flow:0",)}),
        ("flow flow:360", {"color_flows": ("flow:360",)}),
        ("flow flow:719", {"color_flows": ("flow:719",)}),
        ("helicity h:0", {"helicities": ("h:0",)}),
        ("helicity h:192", {"helicities": ("h:192",)}),
        ("helicity h:383", {"helicities": ("h:383",)}),
        (
            "combined flow and helicity",
            {
                "color_flows": ("flow:0",),
                "helicities": ("h:0",),
            },
        ),
        (
            "structural-zero helicity",
            {
                "color_flows": ("flow:0",),
                "helicities": ("h:384",),
            },
        ),
    )
