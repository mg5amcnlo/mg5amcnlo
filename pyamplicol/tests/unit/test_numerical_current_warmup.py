# SPDX-License-Identifier: 0BSD
"""Real high-precision warm-up observations for numerical current reuse."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from decimal import Decimal, localcontext
from itertools import pairwise
from math import isclose
from types import SimpleNamespace
from typing import Any

import pytest

from pyamplicol.generation import numerical_current_warmup
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_types import GenericDAG
from pyamplicol.generation.numerical_current_warmup import (
    GenericDAGCurrentObservationCapture,
    build_numerical_current_probe_points,
    capture_generic_dag_current_observations,
    generic_dag_numerical_current_opt_out_report,
    run_generic_dag_numerical_current_warmup,
    validate_independent_current_observation_captures,
)
from pyamplicol.generation.stage_types import GenericStageOutputSlot
from pyamplicol.generation.validation import ValidationPointRecord
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir

_PRECISION_DIGITS = 96
_SEED = 17


@pytest.fixture(scope="module")
def z_dag_and_model() -> tuple[GenericDAG, BuiltinSMModel]:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir("d d~ > z", color_accuracy="lc"),
        model=model,
    )
    return dag, model


def _points(
    dag: GenericDAG,
    model: BuiltinSMModel,
) -> tuple[
    tuple[ValidationPointRecord, ...],
    tuple[ValidationPointRecord, ...],
]:
    return build_numerical_current_probe_points(
        dag,
        model,
        process_id="ddbar_z_numerical_warmup",
        seed=_SEED,
        candidate_count=4,
        verification_count=4,
    )


def _synthetic_capture(
    value: Decimal,
) -> GenericDAGCurrentObservationCapture:
    point_sha256s = (hashlib.sha256(b"application-validation-point").hexdigest(),)
    observations = {0: ((value, Decimal(0)),)}
    return GenericDAGCurrentObservationCapture(
        precision_digits=_PRECISION_DIGITS,
        points=(),
        point_sha256s=point_sha256s,
        kinematic_sha256s=(
            hashlib.sha256(b"application-validation-kinematics").hexdigest(),
        ),
        parameter_contexts=((),),
        parameter_context_sha256s=(
            hashlib.sha256(b"application-validation-parameters").hexdigest(),
        ),
        observations=observations,
        runtime_schema_sha256=hashlib.sha256(b"runtime-schema").hexdigest(),
        model_parameter_schema_sha256=hashlib.sha256(
            b"model-parameter-schema"
        ).hexdigest(),
        source_dag_sha256=hashlib.sha256(b"source-dag").hexdigest(),
        evaluator_output_partition_abi=(
            numerical_current_warmup._CAPTURE_OUTPUT_PARTITION_ABI
        ),
        evaluator_output_partition_sha256=hashlib.sha256(
            b"evaluator-output-partition"
        ).hexdigest(),
        observation_batch_sha256=(
            numerical_current_warmup._current_observation_batch_sha256(
                observations,
                point_sha256s=point_sha256s,
            )
        ),
        capture_contract_sha256=hashlib.sha256(
            b"application-validation-capture"
        ).hexdigest(),
    )


def _output_slot(
    *,
    current_id: int,
    value_slot_id: int,
    output_start: int,
    output_stop: int,
) -> GenericStageOutputSlot:
    return GenericStageOutputSlot(
        value_slot_id=value_slot_id,
        current_id=current_id,
        variant="propagated",
        component_start=output_start,
        component_stop=output_stop,
        output_start=output_start,
        output_stop=output_stop,
    )


def _partition_stage(
    *slots: GenericStageOutputSlot,
    output_length: int | None = None,
    stage_index: int = 1,
) -> Any:
    return SimpleNamespace(
        stage_index=stage_index,
        stage_kind="current-combine",
        subset_size=2,
        output_length=(
            slots[-1].output_stop if output_length is None and slots else output_length
        ),
        output_slots=slots,
        output_value_slot_ids=tuple(slot.value_slot_id for slot in slots),
    )


def test_current_output_partition_policy_is_exhaustive_and_identity_bound() -> None:
    stage = _partition_stage(
        _output_slot(
            current_id=2,
            value_slot_id=20,
            output_start=0,
            output_stop=2,
        ),
        _output_slot(
            current_id=3,
            value_slot_id=30,
            output_start=2,
            output_stop=4,
        ),
    )

    partitions, contract = numerical_current_warmup._current_capture_output_partitions(
        stage
    )
    repeated_partitions, repeated_digest = (
        numerical_current_warmup._current_capture_partition_plan((stage,))
    )

    assert partitions == ((0, 2), (2, 4))
    assert repeated_partitions == (partitions,)
    assert len(repeated_digest) == 64
    assert contract["partitions"] == [
        {
            "current_id": 2,
            "output_start": 0,
            "output_stop": 2,
            "slots": [
                {
                    "value_slot_id": 20,
                    "variant": "propagated",
                    "component_start": 0,
                    "component_stop": 2,
                    "output_start": 0,
                    "output_stop": 2,
                }
            ],
        },
        {
            "current_id": 3,
            "output_start": 2,
            "output_stop": 4,
            "slots": [
                {
                    "value_slot_id": 30,
                    "variant": "propagated",
                    "component_start": 2,
                    "component_stop": 4,
                    "output_start": 2,
                    "output_stop": 4,
                }
            ],
        },
    ]


@pytest.mark.parametrize(
    ("slots", "output_length", "message"),
    (
        (
            (
                _output_slot(
                    current_id=2,
                    value_slot_id=20,
                    output_start=0,
                    output_stop=2,
                ),
                _output_slot(
                    current_id=3,
                    value_slot_id=30,
                    output_start=3,
                    output_stop=5,
                ),
            ),
            5,
            "overlap, gap",
        ),
        (
            (
                _output_slot(
                    current_id=2,
                    value_slot_id=20,
                    output_start=0,
                    output_stop=2,
                ),
                _output_slot(
                    current_id=3,
                    value_slot_id=30,
                    output_start=1,
                    output_stop=3,
                ),
            ),
            3,
            "overlap, gap",
        ),
        (
            (
                _output_slot(
                    current_id=3,
                    value_slot_id=30,
                    output_start=0,
                    output_stop=2,
                ),
                _output_slot(
                    current_id=2,
                    value_slot_id=20,
                    output_start=2,
                    output_stop=4,
                ),
            ),
            4,
            "current order",
        ),
        (
            (
                _output_slot(
                    current_id=2,
                    value_slot_id=20,
                    output_start=0,
                    output_stop=2,
                ),
                _output_slot(
                    current_id=3,
                    value_slot_id=30,
                    output_start=2,
                    output_stop=4,
                ),
                _output_slot(
                    current_id=2,
                    value_slot_id=40,
                    output_start=4,
                    output_stop=6,
                ),
            ),
            6,
            "current order",
        ),
    ),
)
def test_current_output_partition_policy_rejects_gap_overlap_and_order_drift(
    slots: tuple[GenericStageOutputSlot, ...],
    output_length: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        numerical_current_warmup._current_capture_output_partitions(
            _partition_stage(*slots, output_length=output_length)
        )


def test_partitioned_high_precision_replay_preserves_order_and_width() -> None:
    class ExactSource:
        def __init__(
            self,
            outputs: tuple[tuple[Decimal, Decimal], ...],
        ) -> None:
            self.outputs = outputs
            self.calls: list[tuple[tuple[tuple[Decimal, Decimal], ...], int]] = []

        def evaluate_complex_with_prec(
            self,
            values: tuple[tuple[Decimal, Decimal], ...],
            precision: int,
        ) -> tuple[tuple[Decimal, Decimal], ...]:
            self.calls.append((values, precision))
            return self.outputs

    class ExactLeaf:
        def __init__(self, input_len: int, source: ExactSource) -> None:
            self.input_len = input_len
            self._source_evaluator = source

    first_source = ExactSource(
        (
            (Decimal(1), Decimal(0)),
            (Decimal(2), Decimal(0)),
        )
    )
    second_source = ExactSource(((Decimal(3), Decimal(0)),))
    evaluator = numerical_current_warmup._PartitionedCurrentCaptureStageEvaluator(
        evaluators=(
            ExactLeaf(2, first_source),
            ExactLeaf(1, second_source),
        ),
        chunk_input_indices=((0, 2), (1,)),
        output_partitions=((0, 2), (2, 3)),
        input_len=3,
        output_len=3,
    )
    values = (
        (Decimal(10), Decimal(0)),
        (Decimal(20), Decimal(0)),
        (Decimal(30), Decimal(0)),
    )

    assert evaluator.evaluate_complex_with_prec(values, 112) == (
        (Decimal(1), Decimal(0)),
        (Decimal(2), Decimal(0)),
        (Decimal(3), Decimal(0)),
    )
    assert first_source.calls == [((values[0], values[2]), 112)]
    assert second_source.calls == [((values[1],), 112)]

    second_source.outputs = ()
    with pytest.raises(ValueError, match="wrong output width"):
        evaluator.evaluate_complex_with_prec(values, 112)


def test_interpreted_stage_converts_complex_float_without_decimal_rounding() -> None:
    from types import SimpleNamespace

    expected = (
        Decimal("1.234567890123456789012345678901234567890123456789"),
        Decimal("-0"),
    )
    evaluator = SimpleNamespace(
        evaluate_complex_with_prec=lambda _values, _precision: [
            SimpleNamespace(to_decimal_tuple=lambda: expected)
        ]
    )

    with localcontext() as context:
        context.prec = 6
        result = numerical_current_warmup._evaluate_interpreted_stage(
            evaluator, (), precision=80
        )[0]

    assert result[0] is expected[0]
    assert result[1] is expected[1]
    assert result[1].is_signed()


@pytest.mark.parametrize("ambient_precision", (28, 50, 96))
@pytest.mark.parametrize(
    ("optimized", "accepted"),
    (
        (Decimal("1.125"), True),
        (Decimal("1.125000000000000000000000000001"), False),
    ),
)
def test_application_validation_uses_configured_precision_at_tolerance_boundary(
    ambient_precision: int,
    optimized: Decimal,
    accepted: bool,
) -> None:
    reference = _synthetic_capture(Decimal(1))
    applied = _synthetic_capture(optimized)

    with localcontext() as context:
        context.prec = ambient_precision
        if accepted:
            result = numerical_current_warmup._validate_applied_current_observations(
                reference,
                applied,
                relative_tolerance=0.0,
                absolute_tolerance=0.125,
            )
            assert result["status"] == "verified"
            assert Decimal(str(result["maximum_tolerance_ratio"])) <= 1
        else:
            with pytest.raises(
                ValueError,
                match="beyond its authenticated tolerance",
            ):
                numerical_current_warmup._validate_applied_current_observations(
                    reference,
                    applied,
                    relative_tolerance=0.0,
                    absolute_tolerance=0.125,
                )


def test_probe_points_are_deterministic_physically_distinct_domains(
    z_dag_and_model: tuple[GenericDAG, BuiltinSMModel],
) -> None:
    dag, model = z_dag_and_model
    candidate, verification = _points(dag, model)

    assert (candidate, verification) == _points(dag, model)
    all_points = (*candidate, *verification)
    assert len({point.seed for point in all_points}) == 8
    assert len({point.four_vectors for point in all_points}) == 8
    for point in all_points:
        incoming = point.four_vectors[:2]
        outgoing = point.four_vectors[2:]
        for component in range(4):
            assert isclose(
                sum(momentum[component] for momentum in incoming),
                sum(momentum[component] for momentum in outgoing),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )


def test_capture_uses_only_interpreted_high_precision_stage_replay(
    z_dag_and_model: tuple[GenericDAG, BuiltinSMModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dag, model = z_dag_and_model
    candidate, _verification = _points(dag, model)
    real_compile = numerical_current_warmup._compile_symbolica_outputs
    compile_contracts: list[dict[str, Any]] = []

    def guarded_compile(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["jit_compile"] is False
        output_partitions = tuple(kwargs["output_partitions"])
        assert output_partitions
        assert output_partitions[0][0] == 0
        assert all(left[1] == right[0] for left, right in pairwise(output_partitions))
        assert output_partitions[-1][1] == len(args[0])
        settings = kwargs["symbolica_settings"]
        assert settings.direct_translation is False
        assert settings.jit_direct_translation is False
        assert settings.jit_optimization_level == 0
        compile_contracts.append(dict(kwargs))
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(
        numerical_current_warmup,
        "_compile_symbolica_outputs",
        guarded_compile,
    )
    capture = capture_generic_dag_current_observations(
        dag,
        model,
        candidate,
        precision_digits=_PRECISION_DIGITS,
    )

    assert compile_contracts
    assert capture.precision_digits == _PRECISION_DIGITS
    assert capture.point_count == 4
    assert capture.current_count == len(dag.currents)
    assert len(set(capture.kinematic_sha256s)) == 4
    assert set(capture.observations) == {current.id for current in dag.currents}
    for current in dag.currents:
        values = capture.observations[current.id]
        assert len(values) == capture.point_count * current.dimension
        assert all(
            isinstance(component, Decimal) and component.is_finite()
            for value in values
            for component in value
        )
    provenance = capture.to_provenance_dict()
    assert provenance["complete_current_component_digest"] is True
    assert provenance["components_embedded"] is False
    assert provenance["point_major"] is True
    assert str(provenance["abi"]).endswith("-v2")
    assert provenance["evaluator_output_partition"] == {
        "abi": numerical_current_warmup._CAPTURE_OUTPUT_PARTITION_ABI,
        "sha256": capture.evaluator_output_partition_sha256,
    }
    assert len(capture.evaluator_output_partition_sha256) == 64
    assert provenance["points"] == [point.to_mapping() for point in candidate]


def test_transaction_reuses_one_baseline_capture_session(
    z_dag_and_model: tuple[GenericDAG, BuiltinSMModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dag, model = z_dag_and_model
    real_builder = numerical_current_warmup._build_current_capture_session
    built_dags: list[GenericDAG] = []

    def counted_builder(*args: object, **kwargs: Any) -> Any:
        built_dags.append(args[0])  # type: ignore[arg-type]
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(
        numerical_current_warmup,
        "_build_current_capture_session",
        counted_builder,
    )
    result = run_generic_dag_numerical_current_warmup(
        dag,
        model,
        process_id="ddbar_z_session_reuse",
        mode="certified-reuse",
        execution_mode="compiled",
        precision_digits=_PRECISION_DIGITS,
        probe_count=4,
        verification_probe_count=4,
        seed=_SEED,
        relative_tolerance=1.0e-70,
        absolute_tolerance=1.0e-80,
    )

    assert result.application.report.applied_relation_count == 0
    assert built_dags == [dag]
    assert (
        result.candidate_capture.evaluator_output_partition_sha256
        == result.verification_capture.evaluator_output_partition_sha256
    )


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
def test_real_capture_drives_authenticated_discovery_and_application(
    z_dag_and_model: tuple[GenericDAG, BuiltinSMModel],
    execution_mode: str,
) -> None:
    dag, model = z_dag_and_model
    result = run_generic_dag_numerical_current_warmup(
        dag,
        model,
        process_id="ddbar_z_numerical_warmup",
        mode="certified-reuse",
        execution_mode=execution_mode,  # type: ignore[arg-type]
        precision_digits=_PRECISION_DIGITS,
        probe_count=4,
        verification_probe_count=4,
        seed=_SEED,
        relative_tolerance=1.0e-70,
        absolute_tolerance=1.0e-80,
    )
    candidate = result.candidate_capture
    verification = result.verification_capture
    discovery = result.discovery
    assert discovery.report.inspected_current_count == len(dag.currents)
    assert (
        discovery.report.candidate_observation_batch_sha256
        == candidate.observation_batch_sha256
    )
    assert (
        discovery.report.verification_observation_batch_sha256
        == verification.observation_batch_sha256
    )
    assert set(candidate.parameter_context_sha256s).isdisjoint(
        verification.parameter_context_sha256s
    )
    assert candidate.parameter_contexts
    assert verification.parameter_contexts
    parameter_count = len(candidate.parameter_contexts[0])
    assert parameter_count
    assert all(
        len(context) == parameter_count
        for context in (
            *candidate.parameter_contexts,
            *verification.parameter_contexts,
        )
    )
    for parameter_index in range(parameter_count):
        assert (
            len(
                {
                    context[parameter_index]
                    for context in (
                        *candidate.parameter_contexts,
                        *verification.parameter_contexts,
                    )
                }
            )
            > 1
        )

    application = result.application
    if discovery.certificates:
        assert application.report.state == "authenticated-numerical-applied"
        assert application.report.warning_required
        assert application.report.applied_relation_count == len(discovery.certificates)
    else:
        assert (
            discovery.report.state
            == application.report.state
            == "no_certified_numerical_relation"
        )
        assert not application.report.warning_required
        assert application.report.applied_relation_count == 0
        assert application.dag is dag
        assert result.application_capture is None
        assert result.application_validation["status"] == (
            "not-required-no-applied-relations"
        )
    payload = result.to_json_dict()
    disabled = generic_dag_numerical_current_opt_out_report(
        dag,
        execution_mode=execution_mode,  # type: ignore[arg-type]
    )
    assert set(payload) - {"source_semantics"} == set(disabled)
    assert "source_semantics" in payload
    assert payload["candidate_capture"]["points"]
    assert payload["verification_capture"]["points"]
    assert payload["warning"]["required"] is result.warning_required
    assert payload["effective_mode"] == "certified-reuse"
    assert payload["relation_correctness"]["applied_relation_count"] == (
        result.application.report.applied_relation_count
    )


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
def test_public_opt_out_records_disabled_unoptimized_path_without_capture(
    z_dag_and_model: tuple[GenericDAG, BuiltinSMModel],
    execution_mode: str,
) -> None:
    dag, _model = z_dag_and_model
    report = generic_dag_numerical_current_opt_out_report(
        dag,
        execution_mode=execution_mode,  # type: ignore[arg-type]
    )

    assert report["requested_mode"] == "off"
    assert report["state"] == "disabled-by-user"
    assert report["candidate_capture"] is None
    assert report["verification_capture"] is None
    assert report["application_capture"] is None
    assert report["application_validation"]["status"] == "disabled-by-user"
    assert report["certified_relation_count"] == 0
    assert report["applied_relation_count"] == 0
    assert report["effective_mode"] == "off"
    assert "source_semantics" not in report
    assert report["relation_correctness"] == {
        "abi": "pyamplicol-numerical-current-relation-correctness-v1",
        "state": "no-applied-relations",
        "applied_relation_count": 0,
    }
    assert report["warning"] == {
        "required": False,
        "emit": "never",
        "code": None,
        "message": None,
    }


def test_public_opt_out_does_not_serialize_wide_helicity_ancestry(
    z_dag_and_model: tuple[GenericDAG, BuiltinSMModel],
) -> None:
    dag, _model = z_dag_and_model
    current = dag.currents[0]
    wide_ancestry = 10**4300
    wide_current = replace(
        current,
        index=replace(current.index, helicity_ancestry=wide_ancestry),
    )
    wide_dag = replace(dag, currents=(wide_current, *dag.currents[1:]))

    report = generic_dag_numerical_current_opt_out_report(
        wide_dag,
        execution_mode="compiled",
    )

    assert report["state"] == "disabled-by-user"
    assert "source_semantics" not in report


def test_generic_application_scope_contains_unsafe_relation_kinds() -> None:
    equal = SimpleNamespace(relation_kind="equal")
    opposite = SimpleNamespace(relation_kind="opposite")
    zero = SimpleNamespace(relation_kind="zero")

    contracted = numerical_current_warmup._application_scoped_certificates(
        (equal, opposite, zero),  # type: ignore[arg-type]
        allowed_relation_kinds=("zero", "equal"),
    )
    multi_helicity = numerical_current_warmup._application_scoped_certificates(
        (equal, opposite, zero),  # type: ignore[arg-type]
        allowed_relation_kinds=(),
    )

    assert contracted == (equal, zero)
    assert multi_helicity == ()
    with pytest.raises(ValueError, match="relation kind is invalid"):
        numerical_current_warmup._application_scoped_certificates(
            (SimpleNamespace(relation_kind="projective"),),  # type: ignore[arg-type]
            allowed_relation_kinds=("equal",),
        )


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
def test_certified_zero_currents_are_applied_and_revalidated_by_default(
    execution_mode: str,
) -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir("g g > g g", color_accuracy="full"),
        model=model,
    )

    result = run_generic_dag_numerical_current_warmup(
        dag,
        model,
        process_id="gg_gg_numerical_warmup",
        mode="certified-reuse",
        execution_mode=execution_mode,  # type: ignore[arg-type]
        precision_digits=_PRECISION_DIGITS,
        probe_count=4,
        verification_probe_count=4,
        relative_tolerance=1.0e-70,
        absolute_tolerance=1.0e-80,
        seed=_SEED,
    )

    assert result.discovery.certificates
    assert any(
        certificate.relation_kind == "zero"
        for certificate in result.discovery.certificates
    )
    for certificate in result.discovery.certificates:
        assert certificate.candidate_probe_count == 4
        assert certificate.verification_probe_count == 4
        assert (
            certificate.current_dimension
            == dag.currents[certificate.current_id].dimension
        )
    assert result.application.report.applied_relation_count == len(
        result.discovery.certificates
    )
    assert (
        result.application.report.interaction_evaluation_count_projected
        < result.application.report.interaction_evaluation_count_before
    )
    assert result.warning_required
    assert result.application_capture is not None
    assert result.application_capture.parameter_contexts == (
        result.verification_capture.parameter_contexts
    )
    assert result.application_validation["status"] == "verified"
    assert result.application_validation["checked_current_count"] == len(dag.currents)
    assert Decimal(
        str(result.application_validation["maximum_tolerance_ratio"])
    ) <= Decimal(1)
    assert result.application_validation["maximum_absolute_residual"] == "0"
    assert (
        result.application_validation["evaluator_output_partition_sha256"]
        == result.verification_capture.evaluator_output_partition_sha256
    )


def test_capture_domains_fail_closed_on_replay_drift(
    z_dag_and_model: tuple[GenericDAG, BuiltinSMModel],
) -> None:
    dag, model = z_dag_and_model
    candidate_points, verification_points = _points(dag, model)
    with pytest.raises(ValueError, match="distinct physical momenta"):
        capture_generic_dag_current_observations(
            dag,
            model,
            (candidate_points[0], candidate_points[0]),
            precision_digits=_PRECISION_DIGITS,
        )
    with pytest.raises(ValueError, match="does not match"):
        capture_generic_dag_current_observations(
            dag,
            model,
            (
                replace(candidate_points[0], process="g g > z"),
                candidate_points[1],
            ),
            precision_digits=_PRECISION_DIGITS,
        )

    candidate = capture_generic_dag_current_observations(
        dag,
        model,
        candidate_points,
        precision_digits=_PRECISION_DIGITS,
    )
    verification = capture_generic_dag_current_observations(
        dag,
        model,
        verification_points,
        precision_digits=_PRECISION_DIGITS,
    )
    repeated_kinematics = replace(
        verification,
        kinematic_sha256s=candidate.kinematic_sha256s,
    )
    with pytest.raises(ValueError, match="independent replay domains"):
        validate_independent_current_observation_captures(
            candidate,
            repeated_kinematics,
        )

    changed_parameter_schema = replace(
        verification,
        model_parameter_schema_sha256=hashlib.sha256(
            b"changed-model-parameter-schema"
        ).hexdigest(),
    )
    with pytest.raises(ValueError, match="independent replay domains"):
        validate_independent_current_observation_captures(
            candidate,
            changed_parameter_schema,
        )

    changed_partitions = (
        replace(
            verification,
            evaluator_output_partition_abi=(
                "pyamplicol-stale-current-output-partitions-v0"
            ),
        ),
        replace(
            verification,
            evaluator_output_partition_sha256=hashlib.sha256(
                b"stale-output-partitions"
            ).hexdigest(),
        ),
    )
    for changed_partition in changed_partitions:
        with pytest.raises(ValueError, match="independent replay domains"):
            validate_independent_current_observation_captures(
                candidate,
                changed_partition,
            )
        with pytest.raises(ValueError, match="provenance drifted"):
            numerical_current_warmup._validate_applied_current_observations(
                verification,
                changed_partition,
                relative_tolerance=1.0e-70,
                absolute_tolerance=1.0e-80,
            )

    current_id = min(candidate.observations)
    incomplete = replace(
        verification,
        observations={
            **verification.observations,
            current_id: verification.observations[current_id][:-1],
        },
    )
    with pytest.raises(ValueError, match="widths drifted"):
        validate_independent_current_observation_captures(
            candidate,
            incomplete,
        )

    changed_values = dict(verification.observations)
    changed = list(changed_values[current_id])
    changed[0] = (changed[0][0] + Decimal(1), changed[0][1])
    changed_values[current_id] = tuple(changed)
    changed_capture = replace(
        verification,
        observations=changed_values,
    )
    with pytest.raises(ValueError, match="beyond its authenticated tolerance"):
        numerical_current_warmup._validate_applied_current_observations(
            verification,
            changed_capture,
            relative_tolerance=1.0e-70,
            absolute_tolerance=1.0e-80,
        )

    with pytest.raises(ValueError, match="provenance drifted"):
        numerical_current_warmup._validate_applied_current_observations(
            verification,
            changed_parameter_schema,
            relative_tolerance=1.0e-70,
            absolute_tolerance=1.0e-80,
        )
