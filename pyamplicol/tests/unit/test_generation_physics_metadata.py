# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import pyamplicol.generation.physics_metadata as physics_metadata
import pyamplicol.generation.runtime_schema as runtime_schema
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.physics_metadata import (
    EAGER_PLAN_V3_REDUCTION_ENTRIES_MEMBER,
    EAGER_PLAN_V3_REDUCTION_GROUPS_CONTAINER_PATH,
    EAGER_PLAN_V3_REDUCTION_GROUPS_KIND,
    EAGER_PLAN_V3_REDUCTION_GROUPS_MEMBER,
    EAGER_PLAN_V3_REDUCTION_GROUPS_RUNTIME_LAYOUT_ABI,
    EAGER_PLAN_V3_REDUCTION_GROUPS_SCHEMA_VERSION,
    EAGER_PLAN_V3_REDUCTION_GROUPS_STORAGE_ABI,
    NATIVE_REDUCTION_GROUPS_EXTENSION_KEY,
    _color_id,
    build_resolved_physics_from_dag,
)
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def test_public_lc_color_ids_are_physical_and_sector_independent() -> None:
    assert _color_id(()) == "flow:singlet"
    assert _color_id((2, 4, 1)) == "flow:2,4,1"


@pytest.mark.parametrize(
    ("color_accuracy", "expected_ids"),
    (
        ("lc", ("flow:2,1,3,4", "flow:2,4,3,1")),
        ("nlc", ("color:contracted",)),
        ("full", ("color:contracted",)),
    ),
)
def test_crossed_two_line_public_flow_is_lc_only(
    color_accuracy: str,
    expected_ids: tuple[str, ...],
) -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir("d d~ > t t~", color_accuracy=color_accuracy),
        model=model,
    )

    physics = build_resolved_physics_from_dag(dag, model)

    assert tuple(item["id"] for item in physics["color_components"]) == expected_ids
    assert {
        color_id
        for group in physics["reduction"]["groups"]
        for color_id in group["physical_color_ids"]
    } == set(expected_ids)


@pytest.mark.parametrize("color_accuracy", ["lc", "full"])
def test_bounded_public_physics_matches_expanded_schema_without_building_it(
    color_accuracy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir("d d~ > z g", color_accuracy=color_accuracy),
        model=model,
    )
    process_id = f"bounded-{color_accuracy}"
    expanded = runtime_schema.build_runtime_schema(
        dag,
        model,
        process_id=process_id,
    )["physics"]

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("bounded physics called expanded runtime-schema builder")

    monkeypatch.setattr(runtime_schema, "build_runtime_schema_layout", forbidden)
    bounded = build_resolved_physics_from_dag(
        dag,
        model,
        process_id=process_id,
    )

    assert bounded == expanded
    schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "schemas"
            / "runtime-physics-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(bounded)
    assert len(bounded["external_particles"]) == len(dag.process.legs)
    assert bounded["helicities"]
    assert bounded["color_components"]
    assert bounded["reduction"]["groups"]
    assert NATIVE_REDUCTION_GROUPS_EXTENSION_KEY not in bounded["extensions"]
    assert bounded["model_parameters"]
    assert set(bounded["selectors"]) == {
        "helicity",
        "color_flow",
        "contracted_color",
    }


def test_eager_plan_v3_reduction_groups_are_pacbin_backed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(build_process_ir("d d~ > z g"), model=model)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("plan-v3 constructed expanded reduction groups")

    monkeypatch.setattr(physics_metadata, "_reduction_groups", forbidden)
    physics = build_resolved_physics_from_dag(
        dag,
        model,
        process_id="compact-eager-v3",
        native_eager_plan_v3_reduction_groups=True,
    )

    assert physics["reduction"] == {"kind": "lc-diagonal", "groups": []}
    extensions = physics["extensions"]
    descriptor = extensions[NATIVE_REDUCTION_GROUPS_EXTENSION_KEY]
    assert descriptor == {
        "kind": EAGER_PLAN_V3_REDUCTION_GROUPS_KIND,
        "schema_version": EAGER_PLAN_V3_REDUCTION_GROUPS_SCHEMA_VERSION,
        "storage_abi": EAGER_PLAN_V3_REDUCTION_GROUPS_STORAGE_ABI,
        "runtime_layout_abi": EAGER_PLAN_V3_REDUCTION_GROUPS_RUNTIME_LAYOUT_ABI,
        "container_path": EAGER_PLAN_V3_REDUCTION_GROUPS_CONTAINER_PATH,
        "group_member": EAGER_PLAN_V3_REDUCTION_GROUPS_MEMBER,
        "entry_member": EAGER_PLAN_V3_REDUCTION_ENTRIES_MEMBER,
        "group_count": len(
            physics_metadata._mapping_sequence(
                physics_metadata.build_runtime_amplitude_metadata(dag, model).get(
                    "coherent_groups", ()
                )
            )
        ),
    }
    schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "schemas"
            / "runtime-physics-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(physics)


def test_eager_physics_uses_authoritative_runtime_parameter_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(build_process_ir("d d~ > z g"), model=model)
    imaginary = float.fromhex("0x1.d8fdbd004403dp-2")
    records = (
        {
            "name": "derived_coupling_88.real",
            "kind": "derived_parameter_component",
            "parameter_index": 0,
            "default": 0.0,
            "runtime_name": "derived_coupling_88",
            "complex_component": "real",
        },
        {
            "name": "derived_coupling_88.imag",
            "kind": "derived_parameter_component",
            "parameter_index": 1,
            "default": imaginary,
            "runtime_name": "derived_coupling_88",
            "complex_component": "imag",
        },
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("eager physics reevaluated model-parameter defaults")

    monkeypatch.setattr(
        physics_metadata,
        "build_runtime_model_parameter_records",
        forbidden,
    )
    physics = build_resolved_physics_from_dag(
        dag,
        model,
        process_id="authoritative-eager-parameters",
        native_eager_plan_v3_reduction_groups=True,
        runtime_model_parameters=records,
    )

    assert physics["model_parameters"] == [
        {
            "name": "derived_coupling_88",
            "kind": "derived",
            "default_real": 0.0,
            "default_imaginary": imaginary,
            "mutable": False,
        }
    ]
