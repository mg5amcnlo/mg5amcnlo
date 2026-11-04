# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
import json
from decimal import ROUND_UP, Decimal, localcontext
from types import MethodType, SimpleNamespace

import pytest

from pyamplicol.api.errors import ArtifactError, CompatibilityError
from pyamplicol.runtime._normalization_exact import _pi
from pyamplicol.runtime.symbolica_exact import (
    SymbolicaExactExecutor,
    _apply_lc_replay_input_mapping,
    _apply_lc_replay_resolved,
    _canonical_amplitude_outputs,
    _decimal,
    _diagnostic_schema_mass_bindings,
    _exact_helicity_plan,
    _ExactEvaluator,
    _ExactExpressionEvaluator,
    _ExactRuntimeSourceState,
    _fill_sources_with_states,
    _lc_replay_plan,
    _reduce_materialized_helicity,
    _reduce_resolved,
    _upcast_decimal,
    _validated_color_contraction_entries,
    _working_precision,
)


class _DiagnosticNativeRuntime:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = values

    def _exact_runtime_state_json(self) -> str:
        return json.dumps(
            {
                "model_parameter_values": self._values,
                "normalization_factor": 1.0,
            }
        )


def _diagnostic_schema() -> dict[str, object]:
    return {
        "model": {
            "particles": [
                {"pdg": 6, "mass": 173.0, "mass_parameter": "ufo.MT"},
                {"pdg": 21, "mass": 0.0, "mass_parameter": None},
            ]
        },
        "model_parameters": [
            {"name": "ufo.MT", "parameter_index": 0},
        ],
    }


def _diagnostic_physics() -> dict[str, object]:
    return {
        "external_particles": [
            {"label": 1, "pdg": 21, "role": "initial"},
            {"label": 2, "pdg": 6, "role": "final"},
        ]
    }


def test_binary64_decimal_tag_preserves_exact_float_value() -> None:
    parsed = _decimal("binary64:3fb999999999999a", "test scalar")

    assert parsed == Decimal.from_float(0.1)
    assert parsed != Decimal("0.1")


def test_diagnostic_projection_uses_current_ufo_mass_and_input_order() -> None:
    executor = object.__new__(SymbolicaExactExecutor)
    schema = _diagnostic_schema()
    physics = _diagnostic_physics()
    executor._native_runtime = _DiagnosticNativeRuntime((7.0,))
    executor._execution = {"runtime_schema": schema}
    executor._physics = physics
    executor._primary_execution = executor._execution
    executor._primary_physics = physics
    # Artifact order is (top, gluon); native physics is already caller order.
    executor._permutation = (1, 0)

    original = (
        (
            (-5.0, 0.0, 0.0, 5.0),
            (24.9, 24.0, 0.0, 0.0),
        ),
    )
    projected, metadata = executor._diagnostic_project_onshell(
        original,
        precision=200,
    )

    assert projected[0][0] == (
        Decimal("-5"),
        Decimal("0.0"),
        Decimal("0.0"),
        Decimal("5.0"),
    )
    assert projected[0][1] == (
        Decimal("25.0"),
        Decimal("24.0"),
        Decimal("0.0"),
        Decimal("0.0"),
    )
    assert metadata["precision_digits"] == 200
    assert metadata["working_precision_digits"] == 208
    assert metadata["spatial_components_unchanged"] is True
    assert metadata["global_rebalance_applied"] is False
    assert metadata["unchanged"] is False
    point = metadata["points"][0]
    assert point["global_residual_before"] != point["global_residual_after"]
    gluon, top = point["legs"]
    assert gluon["pdg"] == 21
    assert gluon["projected_energy"] == "-5.0"
    assert top["pdg"] == 6
    assert top["mass"] == "7.0"
    assert top["mass_parameter"] == "ufo.MT"
    assert top["mass_authority"] == ("authenticated-current-runtime-model-parameter")
    assert Decimal(top["shell_residual_after"]) == 0


def test_diagnostic_schema_mass_binding_rejects_missing_parameter() -> None:
    schema = _diagnostic_schema()
    schema["model_parameters"] = []

    with pytest.raises(ArtifactError, match="does not have one authenticated binding"):
        _diagnostic_schema_mass_bindings(
            schema,
            _diagnostic_physics(),
            (Decimal("7"),),
        )


class _RecordingEvaluator:
    def __init__(self) -> None:
        self.values: object = None
        self.precision: int | None = None
        self.result = (Decimal("1.25"), Decimal("-0.5"))

    def evaluate_complex_with_prec(
        self, values: object, precision: int
    ) -> list[SimpleNamespace]:
        self.values = values
        self.precision = precision
        return [SimpleNamespace(to_decimal_tuple=lambda: self.result)]


def test_upcast_decimal_preserves_value_and_carries_requested_precision() -> None:
    cases = (
        Decimal("500"),
        Decimal("0.00123"),
        Decimal("1e100"),
        Decimal("-7.25"),
    )

    for value in cases:
        upcast = _upcast_decimal(value, 40)
        assert upcast == value
        assert len(upcast.as_tuple().digits) == 40

    zero = _upcast_decimal(Decimal("0"), 40)
    assert zero == 0
    assert zero.as_tuple().exponent == -40


def test_exact_evaluator_upcasts_every_complex_input() -> None:
    recording = _RecordingEvaluator()
    evaluator = _ExactEvaluator(input_len=2, evaluator=recording)

    result = evaluator.evaluate(
        ((Decimal("500"), Decimal("0")), (Decimal("0.125"), Decimal("-2"))),
        80,
    )

    assert result == ((Decimal("1.25"), Decimal("-0.5")),)
    assert recording.precision == 80
    values = recording.values
    assert isinstance(values, tuple)
    for real, imaginary in values:
        assert isinstance(real, Decimal)
        assert isinstance(imaginary, Decimal)
        assert len(real.as_tuple().digits) == 80 or real.is_zero()
        assert len(imaginary.as_tuple().digits) == 80 or imaginary.is_zero()


def test_exact_chunked_evaluator_selects_parent_inputs() -> None:
    first = _RecordingEvaluator()
    second = _RecordingEvaluator()
    evaluator = _ExactEvaluator(
        input_len=3,
        chunks=(
            _ExactEvaluator(input_len=2, evaluator=first),
            _ExactEvaluator(input_len=1, evaluator=second),
        ),
        chunk_input_indices=((0, 2), (1,)),
    )

    result = evaluator.evaluate(
        (
            (Decimal("1"), Decimal("0")),
            (Decimal("10"), Decimal("0")),
            (Decimal("3"), Decimal("0")),
        ),
        40,
    )

    assert result == (
        (Decimal("1.25"), Decimal("-0.5")),
        (Decimal("1.25"), Decimal("-0.5")),
    )
    assert first.values is not None
    assert second.values is not None
    assert tuple(value[0] for value in first.values) == (Decimal("1"), Decimal("3"))
    assert tuple(value[0] for value in second.values) == (Decimal("10"),)


@pytest.mark.parametrize("precision", (32, 80))
def test_exact_evaluator_converts_complex_float_without_decimal_rounding(
    precision: int,
) -> None:
    recording = _RecordingEvaluator()
    recording.result = (
        Decimal("1.234567890123456789012345678901234567890123456789"),
        Decimal("-0"),
    )
    evaluator = _ExactEvaluator(input_len=0, evaluator=recording)

    with localcontext() as context:
        context.prec = 6
        result = evaluator.evaluate((), precision)[0]

    assert result[0] is recording.result[0]
    assert result[1] is recording.result[1]
    assert result[1].is_signed()


def test_exact_parameter_derivation_converts_complex_float_for_both_arithmetics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import symbolica

    recording = _RecordingEvaluator()
    recording.result = (
        Decimal("1.234567890123456789012345678901234567890123456789"),
        Decimal("-0"),
    )
    expression = SimpleNamespace(
        evaluate=lambda *_args, **_kwargs: SimpleNamespace(
            to_decimal_tuple=lambda: recording.result
        ),
        evaluator=lambda *_args, **_kwargs: recording,
    )
    monkeypatch.setattr(symbolica, "E", lambda _text: expression)
    evaluator = _ExactExpressionEvaluator(("constant",), ())

    with localcontext() as context:
        context.prec = 6
        arbitrary = evaluator.evaluate((), 80)[0]
        double_double = evaluator.evaluate_double_double(())[0]

    for result in (arbitrary, double_double):
        assert result[0] is recording.result[0]
        assert result[1] is recording.result[1]
        assert result[1].is_signed()
    assert recording.precision == 32


def test_exact_pi_converts_complex_float_without_decimal_rounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import symbolica

    expected = Decimal("3.14159265358979323846264338327950288419716939937510")
    calls: list[tuple[object, int]] = []

    def evaluate(constants: object, *, decimal_digit_precision: int) -> SimpleNamespace:
        calls.append((constants, decimal_digit_precision))
        return SimpleNamespace(to_decimal_tuple=lambda: (expected, Decimal(0)))

    monkeypatch.setattr(
        symbolica, "E", lambda _text: SimpleNamespace(evaluate=evaluate)
    )
    _pi.cache_clear()
    try:
        with localcontext() as context:
            context.prec = 6
            assert _pi(80) is expected
        assert calls == [({}, 80)]
    finally:
        _pi.cache_clear()


def test_exact_amplitude_outputs_follow_plane_arena_bindings() -> None:
    outputs = tuple((Decimal(index), Decimal(-index)) for index in range(4))
    stage = {
        "output_length": 4,
        "compiled_plane_arena": {
            "output_bindings": [
                {"output_index": 0, "arena": "amplitude", "component": 0},
                {"output_index": 1, "arena": "amplitude", "component": 2},
                {"output_index": 2, "arena": "amplitude", "component": 1},
                {"output_index": 3, "arena": "amplitude", "component": 3},
            ]
        },
    }

    assert _canonical_amplitude_outputs(outputs, stage) == (
        outputs[0],
        outputs[2],
        outputs[1],
        outputs[3],
    )


def test_exact_amplitude_outputs_fall_back_to_output_slots() -> None:
    outputs = tuple((Decimal(index), Decimal(-index)) for index in range(3))
    stage = {
        "output_length": 3,
        "output_slots": [
            {
                "output_start": 0,
                "output_stop": 1,
                "component_start": 2,
                "component_stop": 3,
            },
            {
                "output_start": 1,
                "output_stop": 3,
                "component_start": 0,
                "component_stop": 2,
            },
        ],
    }

    assert _canonical_amplitude_outputs(outputs, stage) == (
        outputs[1],
        outputs[2],
        outputs[0],
    )


def test_exact_amplitude_outputs_reject_non_permutation() -> None:
    outputs = (
        (Decimal(1), Decimal(0)),
        (Decimal(2), Decimal(0)),
    )
    stage = {
        "output_length": 2,
        "compiled_plane_arena": {
            "output_bindings": [
                {"output_index": 0, "arena": "amplitude", "component": 0},
                {"output_index": 1, "arena": "amplitude", "component": 0},
            ]
        },
    }

    with pytest.raises(ArtifactError, match="not a permutation"):
        _canonical_amplitude_outputs(outputs, stage)


def test_upcast_rounding_does_not_depend_on_ambient_decimal_context() -> None:
    with localcontext() as context:
        context.rounding = ROUND_UP
        assert _upcast_decimal(Decimal("1.25"), 2) == Decimal("1.2")


def test_every_exact_request_stays_above_symbolica_binary64_shortcut() -> None:
    assert all(_working_precision(precision) >= 40 for precision in range(1, 40))
    assert _working_precision(80) == 88


def _replay_metadata() -> tuple[dict[str, object], dict[str, object]]:
    execution: dict[str, object] = {
        "compiled": {
            "lc_topology_replay": {
                "enabled": True,
                "mode": "external-label-permutation",
                "replayed_sector_count": 2,
                "groups": [
                    {
                        "sector_permutations": [
                            {"weight": 2.0, "label_permutation": []},
                            {
                                "weight": 1.0,
                                "label_permutation": [
                                    {
                                        "representative_label": 3,
                                        "sector_label": 4,
                                    },
                                    {
                                        "representative_label": 4,
                                        "sector_label": 3,
                                    },
                                ],
                            },
                        ]
                    }
                ],
            }
        }
    }
    physics: dict[str, object] = {
        "color_accuracy": "lc",
        "external_particles": [{}, {}, {}, {}],
        "helicities": [
            {
                "id": "h:+1,-1,+1,-1",
                "values": [1, -1, 1, -1],
                "coefficient": 1.0,
            },
            {
                "id": "h:+1,-1,-1,+1",
                "values": [1, -1, -1, 1],
                "coefficient": 1.0,
            },
        ],
        "color_components": [
            {
                "kind": "lc-flow",
                "id": "flow:1,2,3,4",
                "word": [1, 2, 3, 4],
                "coefficient": 1.0,
            },
            {
                "kind": "lc-flow",
                "id": "flow:1,4,3,2",
                "word": [1, 4, 3, 2],
                "coefficient": 1.0,
            },
            {
                "kind": "lc-flow",
                "id": "flow:1,2,4,3",
                "word": [1, 2, 4, 3],
                "coefficient": 1.0,
            },
        ],
        "reduction": {
            "groups": [
                {
                    "physical_helicity_ids": ["h:+1,-1,+1,-1"],
                    "physical_color_ids": ["flow:1,2,3,4"],
                }
            ]
        },
    }
    return execution, physics


def _materialized_cell(value: str) -> tuple[tuple[Decimal, ...], ...]:
    return (
        (Decimal(value), Decimal(0), Decimal(0)),
        (Decimal(0), Decimal(0), Decimal(0)),
    )


def test_exact_lc_replay_routes_public_flows_and_selectors() -> None:
    execution, physics = _replay_metadata()
    plan = _lc_replay_plan(execution, physics, None)
    assert plan is not None
    assert len(plan.entries) == 2
    materialized = (
        _materialized_cell("3"),
        _materialized_cell("5"),
        _materialized_cell("7"),
        _materialized_cell("11"),
    )
    helicity_ids = ("h:+1,-1,+1,-1", "h:+1,-1,-1,+1")
    color_ids = ("flow:1,2,3,4", "flow:1,4,3,2", "flow:1,2,4,3")
    full, _, _ = _apply_lc_replay_resolved(
        materialized,
        plan,
        2,
        helicity_ids,
        color_ids,
        None,
        None,
    )
    assert tuple(
        sum((value for helicity in point for value in helicity), Decimal(0))
        for point in full
    ) == (Decimal(13), Decimal(21))

    values, helicities, colors = _apply_lc_replay_resolved(
        materialized,
        plan,
        2,
        helicity_ids,
        color_ids,
        ("h:+1,-1,-1,+1",),
        ("flow:1,2,4,3",),
    )

    assert helicities == ("h:+1,-1,-1,+1",)
    assert colors == ("flow:1,2,4,3",)
    assert values == (((Decimal(7),),), ((Decimal(11),),))

    point = tuple(
        (Decimal(index), Decimal(0), Decimal(0), Decimal(0)) for index in range(4)
    )
    assert _apply_lc_replay_input_mapping(point, plan.entries[1].input_mapping) == (
        point[0],
        point[1],
        point[3],
        point[2],
    )


def test_exact_lc_replay_accepts_eager_execution_contract() -> None:
    execution, physics = _replay_metadata()
    compiled = execution.pop("compiled")
    assert isinstance(compiled, dict)
    execution["lc_topology_replay"] = compiled["lc_topology_replay"]

    plan = _lc_replay_plan(execution, physics, None)

    assert plan is not None
    assert len(plan.entries) == 2


def test_exact_lc_replay_rejects_disagreeing_execution_lane_mirrors() -> None:
    execution, physics = _replay_metadata()
    compiled = execution["compiled"]
    assert isinstance(compiled, dict)
    execution["lc_topology_replay"] = {
        **compiled["lc_topology_replay"],
        "mode": "different-mode",
    }

    with pytest.raises(ArtifactError, match="disagrees between execution lanes"):
        _lc_replay_plan(execution, physics, None)


def test_exact_lc_replay_rejects_missing_public_flow_reduction() -> None:
    execution, physics = _replay_metadata()
    colors = list(physics["color_components"])  # type: ignore[arg-type]
    colors.pop(1)
    physics["color_components"] = colors

    with pytest.raises(CompatibilityError, match="missing replayed LC flow word"):
        _lc_replay_plan(execution, physics, None)


def _six_flow_replay_metadata() -> tuple[dict[str, object], dict[str, object]]:
    words = (
        (1, 2, 3, 4),
        (1, 2, 4, 3),
        (1, 3, 2, 4),
        (1, 3, 4, 2),
        (1, 4, 2, 3),
        (1, 4, 3, 2),
    )
    representative_words = (words[0], words[0], words[2], words[0], words[2], words[0])

    def flow(index: int) -> dict[str, object]:
        word = words[index]
        representative = representative_words[index]
        return {
            "id": "flow:" + ",".join(map(str, word)),
            "index": index,
            "kind": "lc-flow",
            "word": list(word),
            "computed": index in {0, 2},
            "representative_id": "flow:" + ",".join(map(str, representative)),
            "coefficient": 1.0,
        }

    identity = [
        {"representative_label": label, "sector_label": label} for label in range(1, 5)
    ]
    swap = [
        {"representative_label": 1, "sector_label": 1},
        {"representative_label": 2, "sector_label": 2},
        {"representative_label": 3, "sector_label": 4},
        {"representative_label": 4, "sector_label": 3},
    ]
    execution: dict[str, object] = {
        "compiled": {
            "lc_topology_replay": {
                "contract_version": 2,
                "enabled": True,
                "mode": "external-label-permutation",
                "physical_sector_count": 3,
                "replayed_sector_count": 2,
                "materialized_sector_ids": [0, 2],
                "residual_sector_ids": [2],
                "groups": [
                    {
                        "active_sector_ids": [0, 1],
                        "materialized_sector_id": 0,
                        "representative_sector_id": 0,
                        "proof": {
                            "status": "proven",
                            "algorithm": "canonical-test-proof-v1",
                            "digest": "a" * 64,
                        },
                        "sector_permutations": [
                            {
                                "sector_id": 0,
                                "label_permutation": identity,
                                "weight": 2.0,
                                "sign": 1,
                                "factor": [2.0, 0.0],
                            },
                            {
                                "sector_id": 1,
                                "label_permutation": swap,
                                "weight": 2.0,
                                "sign": 1,
                                "factor": [2.0, 0.0],
                            },
                        ],
                    }
                ],
            }
        },
        "runtime_schema": {
            "amplitude_stage": {
                "roots": [
                    {
                        "root_id": 0,
                        "output_index": 0,
                        "coherent_group_id": 0,
                        "color_sector_id": 0,
                        "all_sector_weight": 2.0,
                    },
                    {
                        "root_id": 1,
                        "output_index": 1,
                        "coherent_group_id": 1,
                        "color_sector_id": 2,
                        "all_sector_weight": 2.0,
                    },
                ]
            }
        },
    }
    color_ids = [str(flow(index)["id"]) for index in range(6)]
    physics: dict[str, object] = {
        "color_accuracy": "lc",
        "external_particles": [{}, {}, {}, {}],
        "helicities": [
            {
                "id": "h:+1,-1,+1,+1",
                "values": [1, -1, 1, 1],
                "coefficient": 1.0,
            }
        ],
        "color_components": [flow(index) for index in range(6)],
        "reduction": {
            "groups": [
                {
                    "id": "reduction:0",
                    "representative_helicity_id": "h:+1,-1,+1,+1",
                    "representative_color_id": color_ids[0],
                    "physical_helicity_ids": ["h:+1,-1,+1,+1"],
                    "physical_color_ids": [color_ids[0], color_ids[5]],
                },
                {
                    "id": "reduction:1",
                    "representative_helicity_id": "h:+1,-1,+1,+1",
                    "representative_color_id": color_ids[2],
                    "physical_helicity_ids": ["h:+1,-1,+1,+1"],
                    "physical_color_ids": [color_ids[2], color_ids[4]],
                },
            ]
        },
    }
    return execution, physics


def test_exact_lc_replay_routes_residual_folded_sector_once() -> None:
    execution, physics = _six_flow_replay_metadata()
    plan = _lc_replay_plan(execution, physics, None)
    assert plan is not None
    assert len(plan.entries) == 2
    target_indices = [
        route.target_index for entry in plan.entries for route in entry.routes
    ]
    assert sorted(target_indices) == list(range(6))
    assert {route.weight for entry in plan.entries for route in entry.routes} == {
        Decimal(1)
    }

    identity = (
        (
            Decimal(3),
            Decimal(300),
            Decimal(5),
            Decimal(700),
            Decimal(500),
            Decimal(900),
        ),
    )
    swapped = (
        (
            Decimal(7),
            Decimal(301),
            Decimal(999),
            Decimal(701),
            Decimal(501),
            Decimal(901),
        ),
    )
    full, helicities, colors = _apply_lc_replay_resolved(
        (identity, swapped),
        plan,
        1,
        ("h:+1,-1,+1,+1",),
        tuple(str(record["id"]) for record in physics["color_components"]),
        None,
        None,
    )
    assert full == (
        ((Decimal(3), Decimal(7), Decimal(5), Decimal(7), Decimal(5), Decimal(3)),),
    )
    assert helicities == ("h:+1,-1,+1,+1",)
    assert len(colors) == 6

    selected, selected_helicities, selected_colors = _apply_lc_replay_resolved(
        (identity, swapped),
        plan,
        1,
        ("h:+1,-1,+1,+1",),
        tuple(str(record["id"]) for record in physics["color_components"]),
        ("h:+1,-1,+1,+1",),
        (colors[4],),
    )
    assert selected == (((Decimal(5),),),)
    assert selected_helicities == ("h:+1,-1,+1,+1",)
    assert selected_colors == (colors[4],)


def test_exact_lc_replay_rejects_incorrect_residual_folded_weight() -> None:
    execution, physics = _six_flow_replay_metadata()
    execution["runtime_schema"]["amplitude_stage"]["roots"][1]["all_sector_weight"] = (
        1.0
    )

    with pytest.raises(ArtifactError, match="covers 5 of 6 public color components"):
        _lc_replay_plan(execution, physics, None)


def _quotient_metadata() -> tuple[dict[str, object], dict[str, object]]:
    source = {
        "current_id": 0,
        "source_id": 0,
        "leg_label": 1,
        "source_helicity": -1,
        "chirality": 0,
        "spin_state": -1,
        "side": "final",
        "particle_id": 1,
        "anti_particle_id": 1,
        "source_orientation": "self-conjugate",
        "wavefunction_kind": "scalar",
        "dimension": 1,
        "source_basis": "scalar",
        "crossing": "identity",
        "value_slot": {
            "component_start": 0,
            "component_stop": 1,
        },
        "source_ir": {
            "basis": "scalar",
            "component_dimension": 1,
            "wavefunction_family": "scalar",
            "identity": {
                "pdg_label": 1,
                "anti_pdg_label": 1,
                "orientation": "self-conjugate",
            },
            "crossing": {
                "momentum_transform": "identity",
                "helicity_factor": 1,
                "chirality_factor": 1,
                "spin_state_factor": 1,
                "phase": [1.0, 0.0],
            },
            "states": [
                {"helicity": -1, "chirality": 0, "spin_state": -1},
                {"helicity": 1, "chirality": 0, "spin_state": 1},
            ],
        },
        "applied_crossing": {
            "momentum_transform": "identity",
            "helicity_factor": 1,
            "chirality_factor": 1,
            "spin_state_factor": 1,
            "phase": [1.0, 0.0],
        },
    }
    materialization = {
        "kind": "pyamplicol-helicity-recurrence-materialization",
        "contract_version": 1,
        "materialized_current_count": 1,
        "materialized_root_count": 1,
        "proof_current_count": 2,
        "proof_root_count": 2,
        "proof_to_materialized_current": [0, 0],
        "selector_schedules": [
            {
                "selector_domain_id": 0,
                "structural_zero": False,
                "active_current_ids": [0],
                "active_root_ids": [0],
            },
            {
                "selector_domain_id": 1,
                "structural_zero": True,
                "active_current_ids": [],
                "active_root_ids": [],
            },
            {
                "selector_domain_id": 2,
                "structural_zero": False,
                "active_current_ids": [0],
                "active_root_ids": [0],
            },
        ],
        "source_routes": [
            {
                "materialized_current_id": 0,
                "external_label": 1,
                "helicity": -1,
                "chirality": 0,
                "spin_state": -1,
                "declared_state_index": 0,
                "selector_domain_id": 3,
                "factor": [2.0, 0.0],
            },
            {
                "materialized_current_id": 0,
                "external_label": 1,
                "helicity": 1,
                "chirality": 0,
                "spin_state": 1,
                "declared_state_index": 1,
                "selector_domain_id": 4,
                "factor": [3.0, 0.0],
            },
        ],
        "amplitude_routes": [
            {
                "materialized_root_id": 0,
                "selector_domain_ids": [0],
                "factor": [1.0, 0.0],
                "residual": False,
            },
            {
                "materialized_root_id": 0,
                "selector_domain_ids": [2],
                "factor": [0.0, 1.0],
                "residual": False,
            },
        ],
    }
    execution: dict[str, object] = {
        "compiled": {},
        "runtime_schema": {
            "source_fill": {"sources": [source]},
            "amplitude_stage": {
                "roots": [
                    {
                        "root_id": 0,
                        "output_index": 0,
                        "coherent_group_id": 0,
                        "all_sector_weight": 1.0,
                        "helicity_weight": 1.0,
                    }
                ],
                "color_contraction": None,
            },
            "helicity_recurrence": {
                "contract_version": 1,
                "selector_domains": [
                    {
                        "id": 0,
                        "complete": True,
                        "source_states": [{"external_label": 1, "helicity": -1}],
                    },
                    {
                        "id": 1,
                        "complete": True,
                        "source_states": [{"external_label": 1, "helicity": 0}],
                    },
                    {
                        "id": 2,
                        "complete": True,
                        "source_states": [{"external_label": 1, "helicity": 1}],
                    },
                    {
                        "id": 3,
                        "complete": False,
                        "source_states": [{"external_label": 1, "helicity": -1}],
                    },
                    {
                        "id": 4,
                        "complete": False,
                        "source_states": [{"external_label": 1, "helicity": 1}],
                    },
                ],
                "materialization": materialization,
            },
        },
    }
    physics: dict[str, object] = {
        "color_accuracy": "lc",
        "external_particles": [{}],
        "extensions": {"normalization": {"global_coupling_factor": 2}},
        "helicities": [
            {
                "id": "h:-1",
                "values": [-1],
                "coefficient": 1.0,
                "structural_zero": False,
            },
            {
                "id": "h:+0",
                "values": [0],
                "coefficient": 0.0,
                "structural_zero": True,
            },
            {
                "id": "h:+1",
                "values": [1],
                "coefficient": 1.0,
                "structural_zero": False,
            },
        ],
        "color_components": [{"id": "flow:1", "kind": "lc-flow", "coefficient": 1.0}],
        "reduction": {
            "groups": [
                {
                    "id": "reduction:0",
                    "physical_helicity_ids": ["h:-1", "h:+1"],
                    "physical_color_ids": ["flow:1"],
                }
            ]
        },
    }
    return execution, physics


class _ExactRuntimeState:
    def _exact_runtime_state_json(self) -> str:
        return json.dumps({"model_parameter_values": [], "normalization_factor": 2.0})


def _synthetic_quotient_executor() -> SymbolicaExactExecutor:
    execution, physics = _quotient_metadata()
    executor = object.__new__(SymbolicaExactExecutor)
    executor._execution = execution
    executor._physics = physics
    executor._native_runtime = _ExactRuntimeState()
    executor._permutation = None
    executor._lc_replay = None
    executor._color_replay = None
    executor._helicity_plan = _exact_helicity_plan(execution, physics, None)
    executor._stage_evaluators = ()
    executor._amplitude_evaluator = None
    executor._load_evaluators = MethodType(lambda _self: None, executor)

    def evaluate_point(
        _self: SymbolicaExactExecutor,
        _point: object,
        _parameters: object,
        _precision: int,
        source_states: object = None,
    ) -> tuple[tuple[Decimal, Decimal], ...]:
        assert isinstance(source_states, tuple)
        state = source_states[0]
        assert isinstance(state, _ExactRuntimeSourceState)
        return (state.factor,)

    executor._evaluate_point = MethodType(evaluate_point, executor)
    return executor


def test_exact_executor_switches_to_helicity_sum_color_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_execution: dict[str, object] = {"runtime_schema": {}, "compiled": {}}
    summed_execution: dict[str, object] = {"runtime_schema": {}, "compiled": {}}
    physics: dict[str, object] = {
        "color_accuracy": "full",
        "external_particles": [{}, {}],
        "extensions": {"normalization": {"global_coupling_factor": 2}},
        "helicities": [{"id": "h", "values": [1, -1]}],
        "color_components": [{"id": "contracted"}],
    }
    identity = SimpleNamespace(input_mapping=())
    swapped = SimpleNamespace(input_mapping=((0, 1), (1, 0)))
    replay = SimpleNamespace(mappings=(identity, swapped))
    parse_calls: list[tuple[object, object, object]] = []

    def parse_replay(
        execution: object, selected_physics: object, permutation: object
    ) -> object:
        parse_calls.append((execution, selected_physics, permutation))
        return replay

    monkeypatch.setattr(
        "pyamplicol.runtime.symbolica_exact.parse_exact_color_topology_replay",
        parse_replay,
    )
    monkeypatch.setattr(
        "pyamplicol.runtime.symbolica_exact._lc_replay_plan",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "pyamplicol.runtime.symbolica_exact._exact_helicity_plan",
        lambda *_args: None,
    )

    executor = object.__new__(SymbolicaExactExecutor)
    executor._execution = primary_execution
    executor._physics = physics
    executor._primary_execution = primary_execution
    executor._primary_physics = physics
    executor._helicity_sum_execution = summed_execution
    executor._helicity_sum_physics = physics
    executor._native_runtime = _ExactRuntimeState()
    executor._permutation = None
    executor._lc_replay = None
    executor._color_replay = None
    executor._helicity_plan = None
    executor._stage_evaluators = ()
    executor._amplitude_evaluator = None
    executor._load_evaluators = MethodType(lambda _self: None, executor)
    evaluated_points: list[object] = []

    def evaluate_point(
        _self: SymbolicaExactExecutor,
        point: object,
        _parameters: object,
        _precision: int,
        _source_states: object = None,
    ) -> tuple[tuple[Decimal, Decimal], ...]:
        evaluated_points.append(point)
        assert isinstance(point, tuple)
        return ((point[0][0], Decimal(0)),)

    executor._evaluate_point = MethodType(evaluate_point, executor)
    reduction_calls: list[tuple[object, ...]] = []

    def reduce_replay(*args: object) -> tuple[object, tuple[str, ...], tuple[str, ...]]:
        reduction_calls.append(args)
        return (((Decimal(42),),),), ("h",), ("contracted",)

    monkeypatch.setattr(
        "pyamplicol.runtime.symbolica_exact.reduce_exact_color_topology_replay",
        reduce_replay,
    )

    result = executor.evaluate_resolved(
        (
            (
                (1.0, 0.0, 0.0, 1.0),
                (2.0, 0.0, 0.0, -1.0),
            ),
        ),
        helicities=None,
        color_flows=None,
        precision=40,
    )

    assert executor._execution is summed_execution
    # Activation checks the gather; evaluation rebinds rational colour weights
    # inside the requested arithmetic context, not the load-time context.
    assert parse_calls == [(summed_execution, physics, None)] * 2
    assert tuple(point[0][0] for point in evaluated_points) == (
        Decimal(1),
        Decimal(2),
    )
    assert len(reduction_calls) == 1
    amplitudes, selected_replay, point_count, normalization, selected_h, selected_c = (
        reduction_calls[0]
    )
    assert len(amplitudes) == 2
    assert selected_replay is replay
    assert point_count == 1
    assert normalization == Decimal(2)
    assert selected_h is None
    assert selected_c is None
    assert result.values == (((Decimal(42),),),)
    assert result.helicity_ids == ("h",)
    assert result.color_ids == ("contracted",)


def test_exact_executor_replays_physical_helicities_and_selectors() -> None:
    executor = _synthetic_quotient_executor()
    result = executor.evaluate_resolved(
        (((1.0, 0.0, 0.0, 0.0),),),
        helicities=None,
        color_flows=None,
        precision=40,
    )

    assert result.helicity_ids == ("h:-1", "h:+0", "h:+1")
    assert result.color_ids == ("flow:1",)
    assert result.values == (
        (
            (Decimal(8),),
            (Decimal(0),),
            (Decimal(18),),
        ),
    )
    selected = executor.evaluate_resolved(
        (((1.0, 0.0, 0.0, 0.0),),),
        helicities=("h:+1",),
        color_flows=("flow:1",),
        precision=40,
    )
    assert selected.helicity_ids == ("h:+1",)
    assert selected.values == (((Decimal(18),),),)


def test_exact_executor_replays_retained_amplitude_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution, physics = _quotient_metadata()
    runtime_schema = execution["runtime_schema"]
    assert isinstance(runtime_schema, dict)
    recurrence = runtime_schema["helicity_recurrence"]
    assert isinstance(recurrence, dict)
    materialization = recurrence["materialization"]
    assert isinstance(materialization, dict)
    amplitude_routes = materialization["amplitude_routes"]
    assert isinstance(amplitude_routes, list)
    amplitude_routes[0]["selector_domain_ids"] = [0, 2]
    del amplitude_routes[1]
    helicities = physics["helicities"]
    assert isinstance(helicities, list)
    helicities[0].update(
        {
            "computed": True,
            "representative_id": "h:-1",
        }
    )
    helicities[1].update(
        {
            "computed": False,
            "representative_id": "h:+0",
        }
    )
    helicities[2].update(
        {
            "computed": False,
            "representative_id": "h:-1",
        }
    )
    executor = _synthetic_quotient_executor()
    executor._physics = physics
    executor._helicity_plan = _exact_helicity_plan(execution, physics, None)
    evaluate_point = executor._evaluate_point
    evaluation_count = 0

    def counted_evaluate_point(
        _self: SymbolicaExactExecutor,
        point: object,
        parameters: object,
        precision: int,
        source_states: object = None,
    ) -> tuple[tuple[Decimal, Decimal], ...]:
        nonlocal evaluation_count
        evaluation_count += 1
        return evaluate_point(point, parameters, precision, source_states)

    executor._evaluate_point = MethodType(counted_evaluate_point, executor)
    reduction_indices: list[int] = []

    def recording_reduce(*args: object, **kwargs: object) -> tuple[Decimal, ...]:
        reduction_indices.append(args[4])
        return _reduce_materialized_helicity(*args, **kwargs)

    monkeypatch.setattr(
        "pyamplicol.runtime.symbolica_exact._reduce_materialized_helicity",
        recording_reduce,
    )

    result = executor.evaluate_resolved(
        (((1.0, 0.0, 0.0, 0.0),),),
        helicities=None,
        color_flows=None,
        precision=40,
    )

    assert result.values == (
        (
            (Decimal(8),),
            (Decimal(0),),
            (Decimal(8),),
        ),
    )
    assert evaluation_count == 1
    assert reduction_indices == [0, 0]

    evaluation_count = 0
    reduction_indices.clear()
    selected = executor.evaluate_resolved(
        (((1.0, 0.0, 0.0, 0.0),),),
        helicities=("h:+1",),
        color_flows=None,
        precision=40,
    )
    assert selected.helicity_ids == ("h:+1",)
    assert selected.values == (((Decimal(8),),),)
    assert evaluation_count == 1
    assert reduction_indices == [0]


def test_exact_executor_does_not_treat_physics_representative_as_execution() -> None:
    execution, physics = _quotient_metadata()
    helicities = physics["helicities"]
    assert isinstance(helicities, list)
    helicities[0].update(
        {
            "computed": True,
            "representative_id": "h:-1",
        }
    )
    helicities[1].update(
        {
            "computed": False,
            "representative_id": "h:+0",
        }
    )
    helicities[2].update(
        {
            "computed": False,
            "representative_id": "h:-1",
        }
    )
    executor = _synthetic_quotient_executor()
    executor._physics = physics
    executor._helicity_plan = _exact_helicity_plan(execution, physics, None)
    evaluate_point = executor._evaluate_point
    evaluation_count = 0

    def counted_evaluate_point(
        _self: SymbolicaExactExecutor,
        point: object,
        parameters: object,
        precision: int,
        source_states: object = None,
    ) -> tuple[tuple[Decimal, Decimal], ...]:
        nonlocal evaluation_count
        evaluation_count += 1
        return evaluate_point(point, parameters, precision, source_states)

    executor._evaluate_point = MethodType(counted_evaluate_point, executor)

    result = executor.evaluate_resolved(
        (((1.0, 0.0, 0.0, 0.0),),),
        helicities=None,
        color_flows=None,
        precision=40,
    )

    assert result.values == (
        (
            (Decimal(8),),
            (Decimal(0),),
            (Decimal(18),),
        ),
    )
    assert evaluation_count == 2


def test_exact_helicity_plan_rejects_reduction_representative_drift() -> None:
    execution, physics = _quotient_metadata()
    helicities = physics["helicities"]
    assert isinstance(helicities, list)
    helicities[0].update(
        {
            "computed": True,
            "representative_id": "h:-1",
        }
    )
    helicities[2].update(
        {
            "computed": False,
            "representative_id": "h:-1",
        }
    )
    reduction = physics["reduction"]
    assert isinstance(reduction, dict)
    groups = reduction["groups"]
    assert isinstance(groups, list)
    groups[0]["representative_helicity_id"] = "h:+1"

    with pytest.raises(
        ArtifactError,
        match="representative disagrees with its reduction group",
    ):
        _exact_helicity_plan(execution, physics, None)


def test_exact_materialized_source_fill_applies_route_factor() -> None:
    execution, _physics = _quotient_metadata()
    schema = execution["runtime_schema"]
    assert isinstance(schema, dict)
    state = [(Decimal(0), Decimal(0))]
    _fill_sources_with_states(
        state,
        ((Decimal(1), Decimal(0), Decimal(0), Decimal(0)),),
        schema,
        (),
        (_ExactRuntimeSourceState(1, 0, 1, (Decimal(2), Decimal(3))),),
    )
    assert state == [(Decimal(2), Decimal(3))]


def test_exact_helicity_plan_uses_selected_source_route_in_active_closure() -> None:
    execution, physics = _quotient_metadata()
    plan = _exact_helicity_plan(execution, physics, None)

    assert plan is not None
    state = plan.schedules[2].source_states[0]
    assert state == _ExactRuntimeSourceState(
        helicity=1,
        chirality=0,
        spin_state=1,
        factor=(Decimal(3), Decimal(0)),
    )


def test_exact_helicity_plan_recovers_globally_flipped_anchor_from_source_ir() -> None:
    execution, physics = _quotient_metadata()
    recurrence = execution["runtime_schema"]["helicity_recurrence"]
    recurrence["materialization"]["source_routes"] = recurrence["materialization"][
        "source_routes"
    ][:1]

    plan = _exact_helicity_plan(execution, physics, None)

    assert plan is not None
    assert plan.schedules[2].source_states[0] == _ExactRuntimeSourceState(
        helicity=1,
        chirality=0,
        spin_state=1,
        factor=(Decimal(1), Decimal(0)),
    )


def test_exact_materialized_helicity_distributes_lc_color_weights() -> None:
    execution, physics = _quotient_metadata()
    physics["color_components"] = [
        {"id": "flow:a", "kind": "lc-flow", "coefficient": 1.0},
        {"id": "flow:b", "kind": "lc-flow", "coefficient": 3.0},
    ]
    physics["reduction"] = {
        "groups": [
            {
                "id": "reduction:0",
                "physical_helicity_ids": ["h:-1"],
                "physical_color_ids": ["flow:a", "flow:b"],
            }
        ]
    }
    value = _reduce_materialized_helicity(
        ((Decimal(2), Decimal(0)),),
        execution,
        physics,
        Decimal(2),
        0,
        ((Decimal(1), Decimal(0)),),
    )
    assert value == (Decimal(2), Decimal(6))


@pytest.mark.parametrize("accuracy", ["nlc", "full"])
def test_exact_materialized_helicity_applies_color_contraction(
    accuracy: str,
) -> None:
    execution, physics = _quotient_metadata()
    physics["color_accuracy"] = accuracy
    physics["color_components"] = [{"id": "contracted", "kind": "contracted"}]
    physics["reduction"] = {
        "groups": [
            {
                "id": "reduction:0",
                "physical_helicity_ids": ["h:-1"],
                "physical_color_ids": ["contracted"],
            },
            {
                "id": "reduction:1",
                "physical_helicity_ids": ["h:-1"],
                "physical_color_ids": ["contracted"],
            },
        ]
    }
    runtime_schema = execution["runtime_schema"]
    assert isinstance(runtime_schema, dict)
    amplitude = runtime_schema["amplitude_stage"]
    assert isinstance(amplitude, dict)
    amplitude["roots"] = [
        {
            "root_id": 0,
            "output_index": 0,
            "coherent_group_id": 0,
            "all_sector_weight": 1.0,
            "helicity_weight": 1.0,
        },
        {
            "root_id": 1,
            "output_index": 1,
            "coherent_group_id": 1,
            "all_sector_weight": 1.0,
            "helicity_weight": 1.0,
        },
    ]
    amplitude["color_contraction"] = {
        "entries": [
            {
                "left_group_id": 0,
                "right_group_id": 1,
                "weight": [2.0, 0.0],
                "symmetry_factor": 0.5,
            }
        ]
    }
    value = _reduce_materialized_helicity(
        ((Decimal(2), Decimal(1)), (Decimal(3), Decimal(-1))),
        execution,
        physics,
        Decimal(4),
        0,
        ((Decimal(1), Decimal(0)), (Decimal(0), Decimal(1))),
    )
    assert value == (Decimal(20),)


def test_exact_color_contraction_compact_and_expanded_parity() -> None:
    execution, physics = _quotient_metadata()
    physics["color_accuracy"] = "full"
    physics["color_components"] = [{"id": "contracted", "kind": "contracted"}]
    physics["reduction"] = {
        "groups": [
            {
                "id": "reduction:0",
                "physical_helicity_ids": ["h:-1"],
                "physical_color_ids": ["contracted"],
            },
            {
                "id": "reduction:1",
                "physical_helicity_ids": ["h:+1"],
                "physical_color_ids": ["contracted"],
            },
        ]
    }
    runtime_schema = execution["runtime_schema"]
    assert isinstance(runtime_schema, dict)
    amplitude = runtime_schema["amplitude_stage"]
    assert isinstance(amplitude, dict)
    amplitude["roots"] = [
        {
            "root_id": 0,
            "output_index": 0,
            "coherent_group_id": 0,
            "all_sector_weight": 1.0,
            "helicity_weight": 1.0,
        },
        {
            "root_id": 1,
            "output_index": 1,
            "coherent_group_id": 1,
            "all_sector_weight": 1.0,
            "helicity_weight": 1.0,
        },
    ]
    expanded = {
        "group_count": 2,
        "entries": [
            {
                "left_group_id": group_id,
                "right_group_id": group_id,
                "weight": [2.0, 0.0],
                "symmetry_factor": 1.0,
            }
            for group_id in range(2)
        ],
    }
    compact = {
        "group_count": 2,
        "entries": [],
        "logical_entry_count": 2,
        "repeated_block": {
            "component_count": 2,
            "component_group_ids": [0, 1],
            "entries": [
                {
                    "left_group_index": 0,
                    "right_group_index": 0,
                    "weight": [2.0, 0.0],
                    "symmetry_factor": 1.0,
                }
            ],
        },
    }
    raw_amplitudes = (
        (Decimal(2), Decimal(1)),
        (Decimal(3), Decimal(-1)),
    )
    root_factors = ((Decimal(1), Decimal(0)),) * 2

    amplitude["color_contraction"] = expanded
    expanded_materialized = _reduce_materialized_helicity(
        raw_amplitudes,
        execution,
        physics,
        Decimal(4),
        0,
        root_factors,
    )
    expanded_resolved = _reduce_resolved(
        (raw_amplitudes,),
        execution,
        physics,
        Decimal(4),
        None,
        None,
    )
    amplitude["color_contraction"] = compact
    compact_materialized = _reduce_materialized_helicity(
        raw_amplitudes,
        execution,
        physics,
        Decimal(4),
        0,
        root_factors,
    )
    compact_resolved = _reduce_resolved(
        (raw_amplitudes,),
        execution,
        physics,
        Decimal(4),
        None,
        None,
    )

    assert compact_materialized == expanded_materialized == (Decimal(40),)
    assert compact_resolved == expanded_resolved


def test_exact_color_contraction_rejects_mixed_storage() -> None:
    execution, physics = _quotient_metadata()
    physics["color_accuracy"] = "full"
    physics["color_components"] = [{"id": "contracted", "kind": "contracted"}]
    runtime_schema = execution["runtime_schema"]
    assert isinstance(runtime_schema, dict)
    amplitude = runtime_schema["amplitude_stage"]
    assert isinstance(amplitude, dict)
    amplitude["color_contraction"] = {
        "group_count": 1,
        "entries": [
            {
                "left_group_id": 0,
                "right_group_id": 0,
                "weight": [1.0, 0.0],
                "symmetry_factor": 1.0,
            }
        ],
        "repeated_block": {
            "component_count": 2,
            "component_group_ids": [0, 1],
            "entries": [],
        },
    }

    with pytest.raises(ArtifactError, match="mixes expanded and repeated entries"):
        _reduce_materialized_helicity(
            ((Decimal(1), Decimal(0)),),
            execution,
            physics,
            Decimal(1),
            0,
            ((Decimal(1), Decimal(0)),),
        )


def test_exact_compact_complex_off_diagonal_entries_match_expanded() -> None:
    groups = {group_id: object() for group_id in range(4)}
    expanded = {
        "entries": [
            {
                "left_group_id": left,
                "right_group_id": right,
                "weight": [2.0, -0.5],
                "symmetry_factor": 2.0,
            }
            for left, right in ((0, 2), (1, 3))
        ]
    }
    compact = {
        "group_count": 4,
        "entries": [],
        "logical_entry_count": 2,
        "repeated_block": {
            "component_count": 2,
            "component_group_ids": [0, 1, 2, 3],
            "entries": [
                {
                    "left_group_index": 0,
                    "right_group_index": 1,
                    "weight": [2.0, -0.5],
                    "symmetry_factor": 2.0,
                }
            ],
        },
    }

    assert tuple(_validated_color_contraction_entries(compact, groups)) == tuple(
        _validated_color_contraction_entries(expanded, groups)
    )


@pytest.mark.parametrize(
    "factorized_block",
    [
        {
            "kind": "klein-four-walsh",
            "cosets": [[0, 1, 2, 3]],
        },
        {
            "kind": "elementary-abelian-walsh",
            "rank": 3,
            "cosets": [list(range(8))],
        },
    ],
)
def test_exact_color_contraction_ignores_walsh_acceleration_metadata(
    factorized_block: dict[str, object],
) -> None:
    local_group_count = len(factorized_block["cosets"][0])
    group_count = 2 * local_group_count
    groups = {group_id: object() for group_id in range(group_count)}
    compact = {
        "group_count": group_count,
        "entries": [],
        "logical_entry_count": 2,
        "repeated_block": {
            "component_count": 2,
            "component_group_ids": list(range(group_count)),
            "entries": [
                {
                    "left_group_index": 0,
                    "right_group_index": 3,
                    "weight": [2.0, 0.0],
                    "symmetry_factor": 2.0,
                }
            ],
            "factorized_block": factorized_block,
        },
    }
    without_acceleration = copy.deepcopy(compact)
    del without_acceleration["repeated_block"]["factorized_block"]

    assert tuple(_validated_color_contraction_entries(compact, groups)) == tuple(
        _validated_color_contraction_entries(without_acceleration, groups)
    )


def test_exact_helicity_plan_fails_closed_on_inconsistent_routes() -> None:
    execution, physics = _quotient_metadata()
    malformed = copy.deepcopy(execution)
    recurrence = malformed["runtime_schema"]["helicity_recurrence"]
    recurrence["materialization"]["amplitude_routes"][0]["selector_domain_ids"] = [2]

    with pytest.raises(ArtifactError, match="do not match active roots"):
        _exact_helicity_plan(malformed, physics, None)


@pytest.mark.parametrize(
    ("domain_ids", "message"),
    [
        ([2, 0], "distinct ordered complete selector domains"),
        ([0, 0], "selector domains are invalid"),
        ([0, 3], "distinct ordered complete selector domains"),
    ],
)
def test_exact_helicity_plan_requires_ordered_complete_amplitude_route_domains(
    domain_ids: list[int],
    message: str,
) -> None:
    execution, physics = _quotient_metadata()
    malformed = copy.deepcopy(execution)
    recurrence = malformed["runtime_schema"]["helicity_recurrence"]
    materialization = recurrence["materialization"]
    materialization["amplitude_routes"][0]["selector_domain_ids"] = domain_ids

    with pytest.raises(ArtifactError, match=message):
        _exact_helicity_plan(malformed, physics, None)


def test_exact_helicity_plan_rejects_conflicting_execution_domains() -> None:
    execution, physics = _quotient_metadata()
    malformed = copy.deepcopy(execution)
    recurrence = malformed["runtime_schema"]["helicity_recurrence"]
    materialization = recurrence["materialization"]
    materialization["amplitude_routes"] = [
        {
            "materialized_root_id": 0,
            "selector_domain_ids": [0, 2],
            "factor": [1.0, 0.0],
            "residual": False,
        },
        {
            "materialized_root_id": 0,
            "selector_domain_ids": [1, 2],
            "factor": [1.0, 0.0],
            "residual": False,
        },
    ]

    with pytest.raises(ArtifactError, match="inconsistent execution domains"):
        _exact_helicity_plan(malformed, physics, None)


def test_exact_helicity_plan_rejects_primary_that_is_replay_only() -> None:
    execution, physics = _quotient_metadata()
    malformed = copy.deepcopy(execution)
    recurrence = malformed["runtime_schema"]["helicity_recurrence"]
    materialization = recurrence["materialization"]
    materialization["amplitude_routes"][0]["selector_domain_ids"] = [0, 2]

    with pytest.raises(ArtifactError, match="both a primary and replay-only domain"):
        _exact_helicity_plan(malformed, physics, None)


def test_exact_helicity_plan_wraps_malformed_scalar_metadata() -> None:
    execution, physics = _quotient_metadata()
    malformed = copy.deepcopy(execution)
    recurrence = malformed["runtime_schema"]["helicity_recurrence"]
    recurrence["materialization"]["source_routes"][0].pop("helicity")

    with pytest.raises(ArtifactError, match="malformed helicity recurrence"):
        _exact_helicity_plan(malformed, physics, None)


def test_exact_helicity_plan_leaves_legacy_artifacts_unchanged() -> None:
    _execution, physics = _quotient_metadata()
    assert _exact_helicity_plan({"runtime_schema": {}}, physics, None) is None
    assert (
        _exact_helicity_plan(
            {
                "runtime_schema": {
                    "helicity_recurrence": {
                        "contract_version": 1,
                        "selector_domains": [],
                    }
                }
            },
            physics,
            None,
        )
        is None
    )
