# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import pickle

import pytest

from pyamplicol.evaluators.symbolica_compile import (
    _partitioned_output_chunk_ranges,
)
from pyamplicol.generation.contracts import RuntimeExpressionSchema
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.stage_compiler import (
    _fanout_aware_current_order,
    build_generic_stage_compiler_blueprint,
)
from pyamplicol.generation.stage_planning import (
    _stage_with_fanout_aware_output_order,
)
from pyamplicol.generation.stage_types import (
    GenericCompiledStageBlueprint,
    GenericStageOutputSlot,
)
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def _minimal_schema() -> dict[str, object]:
    return {
        "parameter_layout": {
            "value_component_count": 0,
            "momentum_parameter_count": 0,
            "model_parameter_count": 0,
        },
        "model_parameters": [],
        "current_storage": {"current_slots": []},
        "value_storage": {"value_slots": []},
        "momentum_slots": [],
        "stages": [],
        "amplitude_stage": {"roots": []},
    }


def test_generation_runtime_expression_schema_is_canonical_and_frozen() -> None:
    left = RuntimeExpressionSchema.from_mapping(_minimal_schema())
    right = RuntimeExpressionSchema.from_mapping(
        dict(reversed(tuple(_minimal_schema().items())))
    )
    assert left == right
    assert left.sha256 == right.sha256
    assert left.to_mapping()["stages"] == []
    assert left.canonical_json == json.dumps(
        _minimal_schema(),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert left.canonical_json == json.dumps(
        left.to_mapping(),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    detached = left.to_mapping()
    detached["stages"] = ["changed"]
    detached["current_storage"]["current_slots"] = ["changed"]
    assert left.to_mapping()["stages"] == []
    assert left.to_mapping()["current_storage"] == {"current_slots": []}


def test_generation_runtime_expression_schema_typed_construction_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = RuntimeExpressionSchema.from_mapping(_minimal_schema())

    def fail_json_dump(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("typed construction must not canonicalize again")

    monkeypatch.setattr(json, "dumps", fail_json_dump)
    assert RuntimeExpressionSchema.from_mapping(schema) is schema


def test_generation_runtime_expression_schema_reuses_decoded_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = RuntimeExpressionSchema.from_mapping(_minimal_schema())

    def fail_json_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("mapping access must not decode canonical JSON again")

    monkeypatch.setattr(json, "loads", fail_json_load)
    assert schema.to_mapping() == _minimal_schema()


@pytest.mark.parametrize(
    ("construct", "message"),
    (
        (
            lambda: RuntimeExpressionSchema(canonical_json="not-json"),
            "not valid JSON",
        ),
        (
            lambda: RuntimeExpressionSchema(canonical_json="[]"),
            "root must be an object",
        ),
        (
            lambda: RuntimeExpressionSchema(
                canonical_json=json.dumps(_minimal_schema()),
                contract_version=1,
            ),
            "unsupported runtime expression-schema contract version",
        ),
        (
            lambda: RuntimeExpressionSchema.from_mapping({"stages": []}),
            "runtime expression schema is missing",
        ),
        (
            lambda: RuntimeExpressionSchema.from_mapping(
                {**_minimal_schema(), "invalid": object()}
            ),
            "must contain canonical JSON values",
        ),
    ),
)
def test_generation_runtime_expression_schema_validation_is_preserved(
    construct: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        construct()


def test_generation_runtime_expression_schema_normalizes_json_inputs() -> None:
    payload = _minimal_schema()
    payload["tuple_value"] = (1, 2)
    payload["mapping_value"] = {2: "two", 1: "one"}
    schema = RuntimeExpressionSchema.from_mapping(payload)

    expected = json.loads(schema.canonical_json)
    assert schema.to_mapping() == expected
    assert list(schema.to_mapping()["mapping_value"]) == ["1", "2"]


def test_generation_runtime_expression_schema_remains_pickleable() -> None:
    schema = RuntimeExpressionSchema.from_mapping(_minimal_schema())
    restored = pickle.loads(pickle.dumps(schema))

    assert restored == schema
    assert restored.to_mapping() == schema.to_mapping()


def test_generation_stage_compiler_requires_schema_and_local_parameters() -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(build_process_ir("d d~ > z"), model=model)
    with pytest.raises(ValueError, match="explicit runtime schema"):
        build_generic_stage_compiler_blueprint(dag)
    with pytest.raises(ValueError, match="stage-local parameter layout"):
        build_generic_stage_compiler_blueprint(
            dag,
            runtime_schema=_minimal_schema(),
            stage_local_parameter_layout=False,
        )


def test_generation_fanout_order_preserves_evaluation_reuse() -> None:
    order, before, after = _fanout_aware_current_order(
        (0, 1, 2, 3),
        output_size_by_current={current_id: 2 for current_id in range(4)},
        evaluation_groups_by_current={
            0: frozenset((1, 10)),
            1: frozenset((2, 20)),
            2: frozenset((1, 30)),
            3: frozenset((2, 40)),
        },
        chunk_size=4,
    )
    assert order in {(0, 2, 1, 3), (1, 3, 0, 2)}
    assert (before, after) == (8, 6)


def _selector_partition_stage(
    *, amplitude: bool = False, color_domains: bool = False
) -> GenericCompiledStageBlueprint:
    signatures = ((3, 23), (10, 18), (3, 23))
    slots = tuple(
        GenericStageOutputSlot(
            value_slot_id=-1 if amplitude else index,
            current_id=-1 if amplitude else index,
            variant="amplitude-root" if amplitude else "propagated",
            component_start=index,
            component_stop=index + 1,
            output_start=index,
            output_stop=index + 1,
            selector_domain_ids=signature,
            color_selector_domain_ids=(
                ((0, 2) if index != 1 else (1,)) if color_domains else ()
            ),
        )
        for index, signature in enumerate(signatures)
    )
    return GenericCompiledStageBlueprint(
        stage_index=0 if amplitude else 1,
        stage_kind="amplitude-roots" if amplitude else "current-combine",
        subset_size=None if amplitude else 2,
        evaluator_label="selector-partition-test",
        parameter_layout="stage-local-value-momentum",
        output_length=3,
        output_slots=slots,
        input_value_slot_ids=(),
        output_value_slot_ids=(),
        interaction_ids=(),
        input_components=(),
        parameter_count=0,
        value_parameter_count=0,
        momentum_parameter_count=0,
        model_parameter_count=0,
        real_valued_inputs=(),
        expression_ready=True,
        blockers=(),
        first_output_previews=("a", "b", "c"),
        output_expressions=("a", "b", "c"),
    )


@pytest.mark.parametrize("amplitude", (False, True))
@pytest.mark.parametrize("color_domains", (False, True))
def test_selector_domains_define_evaluator_chunk_boundaries(
    amplitude: bool, color_domains: bool
) -> None:
    stage = _stage_with_fanout_aware_output_order(
        _selector_partition_stage(
            amplitude=amplitude,
            color_domains=color_domains,
        ),
        chunk_size=512,
    )

    assert stage.output_expressions == ("a", "c", "b")
    assert stage.selector_output_partitions == ((0, 2), (2, 3))
    if amplitude:
        assert tuple(slot.component_start for slot in stage.output_slots) == (0, 2, 1)
    else:
        assert tuple(slot.current_id for slot in stage.output_slots) == (0, 2, 1)


def test_selector_partitions_are_subdivided_without_crossing_domains() -> None:
    assert _partitioned_output_chunk_ranges(
        10,
        chunk_size=4,
        output_partitions=((0, 3), (3, 10)),
    ) == ((0, 3), (3, 7), (7, 10))
