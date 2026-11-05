# SPDX-License-Identifier: 0BSD
"""Numerical nomination and exact-certificate promotion contracts."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from decimal import Decimal, localcontext
from itertools import product
from typing import Any

import pytest

from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_equivalence import (
    NUMERICAL_CURRENT_CAPTURE_ABI,
    NUMERICAL_CURRENT_CAPTURE_OUTPUT_PARTITION_ABI,
    ExactCurrentRelationCertificate,
    _build_numerical_observation_candidate_index,
    _canonical_payload_sha256,
    _current_evaluation_contract,
    _numerical_current_observation_batch_sha256,
    _numerical_observation_tolerance_window_ids,
    apply_numerical_current_relation_certificates,
    assign_recursive_current_evaluation_reuse,
    certify_numerical_current_observations,
    discover_generic_dag_numerical_current_relations,
    discover_recursive_evaluation_relations,
    generic_dag_numerical_capture_output_partition_sha256,
    generic_dag_numerical_model_parameter_schema_sha256,
    generic_dag_numerical_runtime_schema_sha256,
    generic_dag_numerical_source_dag_sha256,
    generic_dag_numerical_source_semantics_sha256,
    verify_dag_relation_certificates,
)
from pyamplicol.generation.runtime_schema import build_runtime_expression_schema
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


class _RuntimeCouplingBuiltin(BuiltinSMModel):
    def runtime_parameter_names_for_vertex(
        self,
        kind: int,
    ) -> tuple[str, str]:
        return (
            f"runtime.vertex.{kind}.component_0",
            f"runtime.vertex.{kind}.component_1",
        )


def _ambiguous_projective_dag():
    """Build a tiny DAG whose existing projective index safely fails closed."""

    model = BuiltinSMModel()
    seed = compile_generic_dag(
        build_process_ir("d d~ > z g", color_accuracy="lc"),
        model=model,
    )
    source_templates = []
    source_keys = set()
    for source_id in seed.sources:
        source = seed.currents[source_id]
        key = (
            _current_evaluation_contract(source),
            source.source_leg_label,
            source.source_helicity,
        )
        if key in source_keys:
            continue
        source_keys.add(key)
        source_templates.append(source)
        if len(source_templates) == 3:
            break
    assert len(source_templates) == 3
    sources = tuple(
        replace(source, id=current_id)
        for current_id, source in enumerate(source_templates)
    )

    template = seed.interactions[0]
    generated_template = seed.currents[template.result_id]
    generated = tuple(
        replace(generated_template, id=current_id) for current_id in range(3, 8)
    )

    def interaction(
        interaction_id: int,
        left_id: int,
        right_id: int,
        result_id: int,
        color_weight: tuple[float, float],
    ):
        return replace(
            template,
            id=interaction_id,
            left_id=left_id,
            right_id=right_id,
            result_id=result_id,
            color_weight=color_weight,
            evaluation_group_id=None,
            evaluation_factor=(1.0, 0.0),
        )

    interactions = (
        interaction(0, 0, 1, 3, (3.0, 0.0)),
        interaction(1, 0, 2, 3, (6.0, 0.0)),
        interaction(2, 0, 1, 4, (1.0, 0.0)),
        interaction(3, 0, 2, 4, (2.0, 0.0)),
        interaction(4, 0, 1, 5, (6.0, 0.0)),
        interaction(5, 0, 2, 5, (12.0, 0.0)),
        interaction(6, 3, 1, 6, (1.0, 0.0)),
        interaction(7, 5, 1, 7, (1.0, 0.0)),
    )
    root = replace(
        seed.amplitude_roots[0],
        id=0,
        left_id=7,
        right_id=1,
    )
    dag = replace(
        seed,
        currents=(*sources, *generated),
        sources=(0, 1, 2),
        interactions=interactions,
        amplitude_roots=(root,),
        selected_source_helicities=(),
        selected_color_sector_ids=(),
        lc_topology_replay=None,
        helicity_recurrence=None,
        helicity_materialization=None,
    )
    return dag, model


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
def test_exact_certified_discovery_promotes_missed_dag_reuse(
    execution_mode,
) -> None:
    dag, model = _ambiguous_projective_dag()
    baseline = assign_recursive_current_evaluation_reuse(dag, model)
    assert (
        baseline.interactions[6].evaluation_group_id
        != baseline.interactions[7].evaluation_group_id
    )

    result = discover_recursive_evaluation_relations(
        dag,
        model,
        mode="certified-reuse",
        execution_mode=execution_mode,
        precision_digits=80,
        probe_count=4,
        seed=17,
    )

    report = result.report
    assert report.execution_mode == execution_mode
    assert report.color_accuracy == "lc"
    assert report.state == "exact-certified-applied"
    assert report.numerical_candidate_count >= 3
    assert report.uncertified_candidate_count >= 1
    assert report.exact_certified_relation_count == 1
    assert report.applied_relation_count == 1
    assert report.certificate_replay_status == "verified"
    assert (
        result.dag.interactions[6].evaluation_group_id
        == result.dag.interactions[7].evaluation_group_id
    )
    assert verify_dag_relation_certificates(dag, model, report.certificates)
    restored = ExactCurrentRelationCertificate.from_json_dict(
        report.certificates[0].to_json_dict()
    )
    assert restored == report.certificates[0]
    assert verify_dag_relation_certificates(dag, model, (restored,))

    tampered = replace(report.certificates[0], factor=(3.0, 0.0))
    assert not verify_dag_relation_certificates(dag, model, (tampered,))


def test_diagnostic_mode_reports_promotable_reuse_without_applying_it() -> None:
    dag, model = _ambiguous_projective_dag()

    result = discover_recursive_evaluation_relations(
        dag,
        model,
        mode="diagnostic",
        execution_mode="compiled",
        precision_digits=96,
        probe_count=3,
        seed=23,
    )

    assert result.dag is dag
    assert result.report.state == "diagnostic-only"
    assert result.report.applied_relation_count == 0
    assert result.report.exact_certified_relation_count == 1
    assert result.report.interaction_evaluation_count_after is not None
    assert result.report.interaction_evaluation_count_before is not None
    assert (
        result.report.interaction_evaluation_count_after
        < result.report.interaction_evaluation_count_before
    )
    payload = result.report.to_json_dict()
    assert payload["probe"] == {
        "status": "completed",
        "precision_digits": 96,
        "probe_count": 3,
        "seed": 23,
        "deterministic": True,
        "candidate_only": True,
    }


def _authenticated_relation_for_ambiguous_dag(
    relation_kind: str,
):
    dag, model = _ambiguous_projective_dag()
    certificate = _authenticated_relation_certificate(dag, relation_kind)
    return dag, model, certificate


def _authenticated_relation_certificate(
    dag,
    relation_kind: str,
):
    model = BuiltinSMModel()
    source_digest = generic_dag_numerical_source_semantics_sha256(
        dag,
        execution_mode="compiled",
    )
    representative_candidate = (
        (Decimal("1.25"), Decimal("-0.5")),
        (Decimal("-3.0"), Decimal("2.25")),
        (Decimal("0.75"), Decimal("4.0")),
        (Decimal("-2.5"), Decimal("-1.125")),
    )
    representative_verification = (
        (Decimal("5.5"), Decimal("0.25")),
        (Decimal("-1.75"), Decimal("-3.0")),
        (Decimal("2.125"), Decimal("6.25")),
        (Decimal("-4.0"), Decimal("1.5")),
    )
    if relation_kind == "zero":
        candidate = tuple((Decimal(0), Decimal(0)) for _ in range(4))
        verification = candidate
        candidate_representative = None
        verification_representative = None
        representative_id = None
    else:
        sign = Decimal(1) if relation_kind == "equal" else Decimal(-1)
        candidate = tuple(
            (sign * real, sign * imaginary)
            for real, imaginary in representative_candidate
        )
        verification = tuple(
            (sign * real, sign * imaginary)
            for real, imaginary in representative_verification
        )
        candidate_representative = representative_candidate
        verification_representative = representative_verification
        representative_id = 3
    certificate = certify_numerical_current_observations(
        current_id=5,
        representative_id=representative_id,
        relation_kind=relation_kind,  # type: ignore[arg-type]
        source_semantics_sha256=source_digest,
        runtime_schema_sha256=(
            generic_dag_numerical_runtime_schema_sha256(dag, model)
        ),
        source_dag_sha256=generic_dag_numerical_source_dag_sha256(dag),
        candidate_capture_sha256=hashlib.sha256(
            b"candidate-capture"
        ).hexdigest(),
        verification_capture_sha256=hashlib.sha256(
            b"verification-capture"
        ).hexdigest(),
        candidate_observation_batch_sha256=hashlib.sha256(
            b"candidate-batch"
        ).hexdigest(),
        verification_observation_batch_sha256=hashlib.sha256(
            b"verification-batch"
        ).hexdigest(),
        candidate_current_values=candidate,
        candidate_representative_values=candidate_representative,
        verification_current_values=verification,
        verification_representative_values=verification_representative,
        precision_digits=96,
        seed=0x5059414D,
        relative_tolerance=1.0e-70,
        absolute_tolerance=1.0e-80,
        candidate_probe_count=2,
        verification_probe_count=2,
    )
    assert certificate is not None
    return certificate


@pytest.mark.parametrize("relation_kind", ("equal", "opposite", "zero"))
def test_authenticated_numerical_relations_are_applied_by_default(
    relation_kind: str,
) -> None:
    dag, model, certificate = _authenticated_relation_for_ambiguous_dag(
        relation_kind
    )

    result = apply_numerical_current_relation_certificates(
        dag,
        model,
        (certificate,),
        mode="certified-reuse",
        execution_mode="compiled",
    )

    assert result.report.state == "authenticated-numerical-applied"
    assert result.report.certificate_replay_status == "verified"
    assert result.report.applied_relation_count == 1
    assert result.report.warning_required
    assert result.report.interaction_evaluation_count_before == 4
    assert result.report.interaction_evaluation_count_projected == 3
    payload = result.report.to_json_dict()
    assert payload["relation_kind_counts"][relation_kind] == 1
    assert payload["warning"] == {
        "required": True,
        "emit": "once-per-generated-artifact",
        "code": "proofless-numerical-current-relations-applied-v1",
        "message": (
            "applied authenticated numerical equal/opposite/zero current "
            "reuse without an exact structural proof; disable with "
            "--no-numerical-current-reuse"
        ),
    }
    mapping = payload["mappings"][0]
    assert mapping["current_id"] == 5
    assert mapping["relation_kind"] == relation_kind
    assert mapping["resolved_current_ids"] == [5]
    assert mapping["projected_interaction_count"] == 1
    assert len(mapping["projected_interaction_ids_sha256"]) == 64
    if relation_kind == "zero":
        assert mapping["representative_id"] is None
        assert mapping["execution_representative_id"] == 3
        assert mapping["factor_binary64"] == ["0x0.0p+0", "0x0.0p+0"]


@pytest.mark.parametrize("relation_kind", ("equal", "opposite", "zero"))
def test_authenticated_application_only_merges_or_suppresses_existing_groups(
    relation_kind: str,
) -> None:
    dag, model = _ambiguous_projective_dag()
    baseline = assign_recursive_current_evaluation_reuse(dag, model)
    certificate = _authenticated_relation_certificate(
        baseline,
        relation_kind,
    )

    result = apply_numerical_current_relation_certificates(
        baseline,
        model,
        (certificate,),
        mode="certified-reuse",
        execution_mode="compiled",
    )

    assert result.report.interaction_evaluation_count_before == 4
    assert result.report.interaction_evaluation_count_projected == 3
    assert (
        result.report.interaction_evaluation_count_projected
        <= result.report.interaction_evaluation_count_before
    )
    assert result.dag.interactions[:7] == baseline.interactions[:7]
    changed = result.dag.interactions[7]
    if relation_kind == "zero":
        assert changed.evaluation_group_id == (
            baseline.interactions[7].evaluation_group_id
        )
        assert changed.evaluation_factor == (0.0, 0.0)
    else:
        assert changed.evaluation_group_id == (
            baseline.interactions[6].evaluation_group_id
        )
        assert changed.evaluation_factor == (
            (1.0, 0.0)
            if relation_kind == "equal"
            else (-1.0, 0.0)
        )


def test_authenticated_zero_suppresses_only_its_downstream_attachment() -> None:
    dag, model = _ambiguous_projective_dag()
    baseline = assign_recursive_current_evaluation_reuse(dag, model)
    extra_contribution = replace(
        baseline.interactions[7],
        id=8,
        left_id=4,
        result_id=7,
        evaluation_group_id=91,
    )
    multi_term = replace(
        baseline,
        interactions=(*baseline.interactions, extra_contribution),
    )
    certificate = _authenticated_relation_certificate(multi_term, "zero")

    result = apply_numerical_current_relation_certificates(
        multi_term,
        model,
        (certificate,),
        mode="certified-reuse",
        execution_mode="compiled",
    )

    assert result.dag.interactions[7].evaluation_factor == (0.0, 0.0)
    assert result.dag.interactions[8] == extra_contribution
    assert result.dag.interactions[:7] == multi_term.interactions[:7]
    assert (
        result.report.interaction_evaluation_count_projected
        == result.report.interaction_evaluation_count_before - 1
    )


def test_authenticated_zero_propagates_through_exact_structural_class() -> None:
    dag, model = _ambiguous_projective_dag()
    duplicate_first = replace(
        dag.interactions[4],
        id=6,
        result_id=6,
    )
    duplicate_second = replace(
        dag.interactions[5],
        id=7,
        result_id=6,
    )
    consume_certified = replace(
        dag.interactions[7],
        id=8,
        left_id=5,
        result_id=7,
    )
    consume_structural_member = replace(
        consume_certified,
        id=9,
        left_id=6,
    )
    unaffected = replace(
        consume_certified,
        id=10,
        left_id=4,
    )
    structural_fanout = replace(
        dag,
        interactions=(
            *dag.interactions[:6],
            duplicate_first,
            duplicate_second,
            consume_certified,
            consume_structural_member,
            unaffected,
        ),
    )
    baseline = assign_recursive_current_evaluation_reuse(
        structural_fanout,
        model,
    )
    assert (
        baseline.interactions[8].evaluation_group_id
        == baseline.interactions[9].evaluation_group_id
    )
    certificate = _authenticated_relation_certificate(baseline, "zero")

    result = apply_numerical_current_relation_certificates(
        baseline,
        model,
        (certificate,),
        mode="certified-reuse",
        execution_mode="compiled",
    )

    assert result.dag.interactions[:8] == baseline.interactions[:8]
    assert result.dag.interactions[8].evaluation_factor == (0.0, 0.0)
    assert result.dag.interactions[9].evaluation_factor == (0.0, 0.0)
    assert result.dag.interactions[10] == baseline.interactions[10]
    mapping = result.report.to_json_dict()["mappings"][0]
    assert mapping["resolved_current_ids"] == [5, 6]
    assert mapping["projected_interaction_count"] == 2
    assert (
        result.report.interaction_evaluation_count_projected
        <= result.report.interaction_evaluation_count_before
    )


@pytest.mark.parametrize(
    ("index_change", "expected_value"),
    (
        ("ordered_external_labels", (3, 2)),
        ("helicity_ancestry", 21),
    ),
)
def test_authenticated_zero_does_not_cross_runtime_routing_contracts(
    index_change: str,
    expected_value: object,
) -> None:
    dag, model = _ambiguous_projective_dag()
    certified = dag.currents[5]
    distinct_index = replace(
        certified.index,
        **{index_change: expected_value},
    )
    assert _current_evaluation_contract(certified) != (
        _current_evaluation_contract(
            replace(certified, index=distinct_index)
        )
    )

    duplicate_first = replace(
        dag.interactions[4],
        id=6,
        result_id=6,
    )
    duplicate_second = replace(
        dag.interactions[5],
        id=7,
        result_id=6,
    )
    consume_certified = replace(
        dag.interactions[7],
        id=8,
        left_id=5,
        result_id=7,
    )
    consume_distinct_route = replace(
        consume_certified,
        id=9,
        left_id=6,
    )
    routed_currents = (
        *dag.currents[:6],
        replace(dag.currents[6], index=distinct_index),
        *dag.currents[7:],
    )
    routed = replace(
        dag,
        currents=routed_currents,
        interactions=(
            *dag.interactions[:6],
            duplicate_first,
            duplicate_second,
            consume_certified,
            consume_distinct_route,
        ),
    )
    baseline = assign_recursive_current_evaluation_reuse(routed, model)
    assert (
        baseline.interactions[8].evaluation_group_id
        != baseline.interactions[9].evaluation_group_id
    )
    certificate = _authenticated_relation_certificate(baseline, "zero")

    result = apply_numerical_current_relation_certificates(
        baseline,
        model,
        (certificate,),
        mode="certified-reuse",
        execution_mode="compiled",
    )

    assert result.dag.interactions[8].evaluation_factor == (0.0, 0.0)
    assert result.dag.interactions[9] == baseline.interactions[9]
    assert result.report.to_json_dict()["mappings"][0][
        "resolved_current_ids"
    ] == [5]


def test_authenticated_merge_rejects_unauthenticated_existing_group_factors() -> None:
    dag, model = _ambiguous_projective_dag()
    antisymmetric = replace(
        dag,
        interactions=(
            *dag.interactions[:6],
            replace(dag.interactions[6], vertex_kind=0),
            replace(
                dag.interactions[7],
                vertex_kind=0,
                left_id=1,
                right_id=5,
            ),
        ),
    )
    baseline = assign_recursive_current_evaluation_reuse(
        antisymmetric,
        model,
    )
    target = replace(
        baseline.interactions[6],
        evaluation_factor=(2.0, 0.0),
    )
    grouped = replace(
        baseline,
        interactions=(
            *baseline.interactions[:6],
            target,
            baseline.interactions[7],
        ),
    )
    certificate = _authenticated_relation_certificate(grouped, "equal")

    with pytest.raises(ValueError, match="unauthenticated"):
        apply_numerical_current_relation_certificates(
            grouped,
            model,
            (certificate,),
            mode="certified-reuse",
            execution_mode="compiled",
        )


def test_authenticated_merge_rejects_spliced_existing_group_membership() -> None:
    dag, model = _ambiguous_projective_dag()
    baseline = assign_recursive_current_evaluation_reuse(dag, model)
    spliced = replace(
        baseline,
        interactions=(
            *baseline.interactions[:7],
            replace(
                baseline.interactions[7],
                evaluation_group_id=(
                    baseline.interactions[0].evaluation_group_id
                ),
            ),
        ),
    )
    certificate = _authenticated_relation_certificate(spliced, "equal")

    with pytest.raises(ValueError, match="aliases distinct"):
        apply_numerical_current_relation_certificates(
            spliced,
            model,
            (certificate,),
            mode="certified-reuse",
            execution_mode="compiled",
        )


def test_authenticated_relation_without_a_safe_target_remains_unapplied() -> None:
    dag, model = _ambiguous_projective_dag()
    baseline = assign_recursive_current_evaluation_reuse(dag, model)
    no_target = replace(
        baseline,
        interactions=(
            *baseline.interactions[:6],
            replace(
                baseline.interactions[6],
                left_id=4,
                right_id=2,
            ),
            baseline.interactions[7],
        ),
    )
    certificate = _authenticated_relation_certificate(no_target, "equal")

    result = apply_numerical_current_relation_certificates(
        no_target,
        model,
        (certificate,),
        mode="certified-reuse",
        execution_mode="compiled",
    )

    assert result.dag is no_target
    assert result.report.state == (
        "authenticated-numerical-no-reuse-opportunity"
    )
    assert result.report.applied_relation_count == 0
    assert not result.report.warning_required
    assert result.report.to_json_dict()["mappings"][0][
        "projected_interaction_count"
    ] == 0


def test_empty_numerical_audit_is_explicit_and_does_not_warn_or_apply() -> None:
    dag, model = _ambiguous_projective_dag()

    result = apply_numerical_current_relation_certificates(
        dag,
        model,
        (),
        mode="certified-reuse",
        execution_mode="compiled",
    )

    assert result.dag is dag
    assert result.report.state == "no_certified_numerical_relation"
    assert (
        result.report.certificate_replay_status
        == "no_certified_numerical_relation"
    )
    assert result.report.applied_relation_count == 0
    assert not result.report.warning_required
    assert result.report.to_json_dict()["warning"] == {
        "required": False,
        "emit": "never",
        "code": None,
        "message": None,
    }


def test_numerical_diagnostic_mode_records_but_does_not_apply_or_warn() -> None:
    dag, model, certificate = _authenticated_relation_for_ambiguous_dag("equal")

    result = apply_numerical_current_relation_certificates(
        dag,
        model,
        (certificate,),
        mode="diagnostic",
        execution_mode="compiled",
    )

    assert result.dag is dag
    assert result.report.state == "authenticated-numerical-diagnostic-only"
    assert result.report.applied_relation_count == 0
    assert not result.report.warning_required
    assert result.report.interaction_evaluation_count_projected == 3


def test_numerical_relation_set_fails_closed_on_mode_or_source_drift() -> None:
    dag, model, certificate = _authenticated_relation_for_ambiguous_dag("equal")

    with pytest.raises(ValueError, match="does not replay"):
        apply_numerical_current_relation_certificates(
            dag,
            model,
            (certificate,),
            mode="certified-reuse",
            execution_mode="eager",
        )

    drifted = replace(
        dag,
        interactions=(
            replace(dag.interactions[0], color_weight=(7.0, 0.0)),
            *dag.interactions[1:],
        ),
    )
    with pytest.raises(ValueError, match="does not replay"):
        apply_numerical_current_relation_certificates(
            drifted,
            model,
            (certificate,),
            mode="certified-reuse",
            execution_mode="compiled",
        )

    with pytest.raises(ValueError, match="provenance"):
        apply_numerical_current_relation_certificates(
            dag,
            _RuntimeCouplingBuiltin(),
            (certificate,),
            mode="certified-reuse",
            execution_mode="compiled",
        )


def _observation_points(domain: str) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(f"{domain}:{index}".encode()).hexdigest()
        for index in range(4)
    )


def _complete_current_observations(
    dag,
    *,
    domain: int,
) -> dict[int, tuple[tuple[Decimal, Decimal], ...]]:
    return {
        current.id: tuple(
            (
                Decimal(
                    domain
                    + current.id * 10_000
                    + point_index * 100
                    + component_index
                    + 1
                ),
                Decimal(
                    -domain
                    - current.id * 20_000
                    - point_index * 200
                    - component_index
                    - 1
                ),
            )
            for point_index in range(4)
            for component_index in range(current.dimension)
        )
        for current in dag.currents
    }


def _complete_observation_evidence(
    dag,
    model,
    candidate,
    verification,
    *,
    candidate_points: tuple[str, ...] | None = None,
    verification_points: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    candidate_point_hashes = candidate_points or _observation_points(
        "candidate"
    )
    verification_point_hashes = verification_points or _observation_points(
        "verification"
    )
    candidate_kinematics = _observation_points("candidate-kinematics")
    verification_kinematics = _observation_points(
        "verification-kinematics"
    )
    candidate_parameters = _observation_points("candidate-parameters")
    verification_parameters = _observation_points(
        "verification-parameters"
    )
    runtime_schema = build_runtime_expression_schema(
        dag,
        model,
    )
    runtime_schema_digest = runtime_schema.sha256
    model_parameter_schema_digest = (
        generic_dag_numerical_model_parameter_schema_sha256(
            dag,
            model,
        )
    )
    source_dag_digest = generic_dag_numerical_source_dag_sha256(dag)
    candidate_batch_digest = _numerical_current_observation_batch_sha256(
        candidate,
        point_sha256s=candidate_point_hashes,
    )
    verification_batch_digest = _numerical_current_observation_batch_sha256(
        verification,
        point_sha256s=verification_point_hashes,
    )
    output_partition_digest = (
        generic_dag_numerical_capture_output_partition_sha256(
            runtime_schema.to_mapping()
        )
    )

    def capture_digest(
        point_hashes: tuple[str, ...],
        kinematic_hashes: tuple[str, ...],
        parameter_hashes: tuple[str, ...],
        batch_digest: str,
    ) -> str:
        return _canonical_payload_sha256(
            {
                "abi": NUMERICAL_CURRENT_CAPTURE_ABI,
                "precision_digits": 96,
                "point_sha256s": list(point_hashes),
                "kinematic_sha256s": list(kinematic_hashes),
                "parameter_context_sha256s": list(parameter_hashes),
                "runtime_schema_sha256": runtime_schema_digest,
                "model_parameter_schema_sha256": (
                    model_parameter_schema_digest
                ),
                "source_dag_sha256": source_dag_digest,
                "evaluator_output_partition_abi": (
                    NUMERICAL_CURRENT_CAPTURE_OUTPUT_PARTITION_ABI
                ),
                "evaluator_output_partition_sha256": (
                    output_partition_digest
                ),
                "observation_batch_sha256": batch_digest,
            }
        )

    return {
        "candidate_point_sha256s": candidate_point_hashes,
        "verification_point_sha256s": verification_point_hashes,
        "candidate_kinematic_sha256s": candidate_kinematics,
        "verification_kinematic_sha256s": verification_kinematics,
        "candidate_parameter_context_sha256s": candidate_parameters,
        "verification_parameter_context_sha256s": verification_parameters,
        "runtime_schema_sha256": runtime_schema_digest,
        "source_dag_sha256": source_dag_digest,
        "evaluator_output_partition_abi": (
            NUMERICAL_CURRENT_CAPTURE_OUTPUT_PARTITION_ABI
        ),
        "evaluator_output_partition_sha256": output_partition_digest,
        "candidate_capture_sha256": capture_digest(
            candidate_point_hashes,
            candidate_kinematics,
            candidate_parameters,
            candidate_batch_digest,
        ),
        "verification_capture_sha256": capture_digest(
            verification_point_hashes,
            verification_kinematics,
            verification_parameters,
            verification_batch_digest,
        ),
        "process_id": dag.process.key,
    }


@pytest.mark.parametrize("relation_kind", ("equal", "opposite", "zero"))
def test_complete_warmup_discovers_and_applies_each_relation_kind(
    relation_kind: str,
) -> None:
    dag, model = _ambiguous_projective_dag()
    candidate = _complete_current_observations(dag, domain=100_000)
    verification = _complete_current_observations(dag, domain=900_000)
    if relation_kind == "zero":
        candidate[5] = tuple(
            (Decimal(0), Decimal(0)) for _value in candidate[5]
        )
        verification[5] = tuple(
            (Decimal(0), Decimal(0)) for _value in verification[5]
        )
    else:
        sign = Decimal(1) if relation_kind == "equal" else Decimal(-1)
        candidate[5] = tuple(
            (sign * real, sign * imaginary)
            for real, imaginary in candidate[3]
        )
        verification[5] = tuple(
            (sign * real, sign * imaginary)
            for real, imaginary in verification[3]
        )

    discovery = discover_generic_dag_numerical_current_relations(
        dag,
        model,
        candidate_observations=candidate,
        verification_observations=verification,
        **_complete_observation_evidence(
            dag,
            model,
            candidate,
            verification,
        ),
        execution_mode="compiled",
        precision_digits=96,
        seed=0x5059414D,
        relative_tolerance=1.0e-70,
        absolute_tolerance=1.0e-80,
    )

    assert discovery.report.state == "certified_numerical_relations"
    assert len(discovery.certificates) == 1
    certificate = discovery.certificates[0]
    assert certificate.current_id == 5
    assert certificate.relation_kind == relation_kind
    applied = apply_numerical_current_relation_certificates(
        dag,
        model,
        discovery.certificates,
        mode="certified-reuse",
        execution_mode="compiled",
    )
    assert applied.report.applied_relation_count == 1
    assert applied.report.interaction_evaluation_count_projected == 3


def test_complete_warmup_negative_audit_is_first_class_and_warning_free() -> None:
    dag, model = _ambiguous_projective_dag()
    candidate = _complete_current_observations(
        dag,
        domain=100_000,
    )
    verification = _complete_current_observations(
        dag,
        domain=900_000,
    )
    discovery = discover_generic_dag_numerical_current_relations(
        dag,
        model,
        candidate_observations=candidate,
        verification_observations=verification,
        **_complete_observation_evidence(
            dag,
            model,
            candidate,
            verification,
        ),
        execution_mode="compiled",
        precision_digits=96,
        seed=0x5059414D,
        relative_tolerance=1.0e-70,
        absolute_tolerance=1.0e-80,
    )

    assert discovery.certificates == ()
    assert discovery.report.state == "no_certified_numerical_relation"
    assert discovery.report.numerical_candidate_count == 0
    assert discovery.report.nearest_rejected_hypothesis is not None
    payload = discovery.report.to_json_dict()
    assert payload["warning"]["required"] is False
    assert payload["certified_numerical_relation_count"] == 0
    candidate_index = payload["candidate_index"]
    assert candidate_index["completeness"] == (
        "complete-within-configured-tolerance"
    )
    assert (
        candidate_index["screened_pair_hypothesis_count"]
        < candidate_index["theoretical_pair_hypothesis_count"]
    )
    assert discovery.report.tested_hypothesis_count == (
        candidate_index["zero_hypothesis_count"]
        + candidate_index["screened_pair_hypothesis_count"]
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        (
            "evaluator_output_partition_abi",
            "pyamplicol-stale-current-output-partitions-v0",
        ),
        (
            "evaluator_output_partition_sha256",
            hashlib.sha256(b"stale-current-output-partitions").hexdigest(),
        ),
    ),
)
def test_discovery_recomputes_and_rejects_stale_output_partition_evidence(
    field: str,
    replacement: str,
) -> None:
    dag, model = _ambiguous_projective_dag()
    candidate = _complete_current_observations(dag, domain=100_000)
    verification = _complete_current_observations(dag, domain=900_000)
    evidence = _complete_observation_evidence(
        dag,
        model,
        candidate,
        verification,
    )
    evidence[field] = replacement
    model_parameter_schema_digest = (
        generic_dag_numerical_model_parameter_schema_sha256(dag, model)
    )

    for domain, observations in (
        ("candidate", candidate),
        ("verification", verification),
    ):
        point_hashes = tuple(evidence[f"{domain}_point_sha256s"])
        batch_digest = _numerical_current_observation_batch_sha256(
            observations,
            point_sha256s=point_hashes,
        )
        evidence[f"{domain}_capture_sha256"] = _canonical_payload_sha256(
            {
                "abi": NUMERICAL_CURRENT_CAPTURE_ABI,
                "precision_digits": 96,
                "point_sha256s": list(point_hashes),
                "kinematic_sha256s": list(
                    evidence[f"{domain}_kinematic_sha256s"]
                ),
                "parameter_context_sha256s": list(
                    evidence[
                        f"{domain}_parameter_context_sha256s"
                    ]
                ),
                "runtime_schema_sha256": evidence[
                    "runtime_schema_sha256"
                ],
                "model_parameter_schema_sha256": (
                    model_parameter_schema_digest
                ),
                "source_dag_sha256": evidence["source_dag_sha256"],
                "evaluator_output_partition_abi": evidence[
                    "evaluator_output_partition_abi"
                ],
                "evaluator_output_partition_sha256": evidence[
                    "evaluator_output_partition_sha256"
                ],
                "observation_batch_sha256": batch_digest,
            }
        )

    with pytest.raises(ValueError, match="point contract"):
        discover_generic_dag_numerical_current_relations(
            dag,
            model,
            candidate_observations=candidate,
            verification_observations=verification,
            **evidence,
            execution_mode="compiled",
            precision_digits=96,
            seed=0x5059414D,
            relative_tolerance=1.0e-70,
            absolute_tolerance=1.0e-80,
        )


@pytest.mark.parametrize(
    ("difference", "expected_relation_count"),
    (
        (Decimal("0.125"), 1),
        (Decimal("0.125000000000000000000000000001"), 0),
    ),
)
@pytest.mark.parametrize("ambient_precision", (28, 50, 96))
def test_candidate_index_is_complete_at_absolute_tolerance_boundary(
    difference: Decimal,
    expected_relation_count: int,
    ambient_precision: int,
) -> None:
    dag, model = _ambiguous_projective_dag()
    candidate = _complete_current_observations(dag, domain=100_000)
    verification = _complete_current_observations(dag, domain=900_000)
    with localcontext() as construction_context:
        construction_context.prec = 128
        candidate[5] = tuple(
            (real + difference, imaginary)
            for real, imaginary in candidate[3]
        )
        verification[5] = tuple(
            (real + difference, imaginary)
            for real, imaginary in verification[3]
        )

    with localcontext() as context:
        context.prec = ambient_precision
        discovery = discover_generic_dag_numerical_current_relations(
            dag,
            model,
            candidate_observations=candidate,
            verification_observations=verification,
            **_complete_observation_evidence(
                dag,
                model,
                candidate,
                verification,
            ),
            execution_mode="compiled",
            precision_digits=96,
            seed=0x5059414D,
            relative_tolerance=0.0,
            absolute_tolerance=0.125,
        )

    assert len(discovery.certificates) == expected_relation_count


def test_candidate_index_relative_window_uses_complete_complex_pair_scale() -> None:
    observations = {
        0: ((Decimal(0), Decimal("1e100")),),
        1: ((Decimal("1e30"), Decimal("1e100")),),
    }
    index = _build_numerical_observation_candidate_index(
        (0, 1),
        observations,
    )

    assert _numerical_observation_tolerance_window_ids(
        index,
        observations[1],
        relation_kind="equal",
        current_id=1,
        relative_tolerance=Decimal("1e-70"),
        absolute_tolerance=Decimal(0),
        precision_digits=96,
    ) == (0,)


@pytest.mark.parametrize("relative_tolerance", ("0", "0.1", "0.9"))
@pytest.mark.parametrize("relation_kind", ("equal", "opposite"))
def test_candidate_index_window_contains_every_small_exhaustive_match(
    relative_tolerance: str,
    relation_kind: str,
) -> None:
    values = tuple(map(Decimal, (-10, -1, 0, 1, 10)))
    relative = Decimal(relative_tolerance)
    absolute = Decimal("0.25")
    sign = Decimal(1) if relation_kind == "equal" else Decimal(-1)
    for components in product(values, repeat=4):
        (
            representative_real,
            representative_imaginary,
            current_real,
            current_imaginary,
        ) = components
        representative = (representative_real, representative_imaginary)
        current = (current_real, current_imaginary)
        reference = (
            sign * representative_real,
            sign * representative_imaginary,
        )
        difference = max(
            abs(current_real - reference[0]),
            abs(current_imaginary - reference[1]),
        )
        scale = max(
            abs(current_real),
            abs(current_imaginary),
            abs(reference[0]),
            abs(reference[1]),
        )
        if difference > absolute + relative * scale:
            continue
        observations = {0: (representative,), 1: (current,)}
        index = _build_numerical_observation_candidate_index(
            (0, 1),
            observations,
        )
        assert 0 in _numerical_observation_tolerance_window_ids(
            index,
            observations[1],
            relation_kind=relation_kind,  # type: ignore[arg-type]
            current_id=1,
            relative_tolerance=relative,
            absolute_tolerance=absolute,
            precision_digits=96,
        )


def test_candidate_only_relation_is_rejected_by_independent_points() -> None:
    dag, model = _ambiguous_projective_dag()
    candidate = _complete_current_observations(dag, domain=100_000)
    verification = _complete_current_observations(dag, domain=900_000)
    candidate[5] = candidate[3]

    discovery = discover_generic_dag_numerical_current_relations(
        dag,
        model,
        candidate_observations=candidate,
        verification_observations=verification,
        **_complete_observation_evidence(
            dag,
            model,
            candidate,
            verification,
        ),
        execution_mode="compiled",
        precision_digits=96,
        seed=0x5059414D,
        relative_tolerance=1.0e-70,
        absolute_tolerance=1.0e-80,
    )

    assert discovery.certificates == ()
    assert discovery.report.state == "no_certified_numerical_relation"
    assert discovery.report.numerical_candidate_count == 1
    assert discovery.report.verification_rejected_count == 1
    assert any(
        item["reason"] == "independent-verification-rejected-candidate"
        for item in discovery.report.rejected_candidates
    )
    payload = discovery.report.to_json_dict()
    assert payload["rejected_hypothesis_count"] == (
        payload["tested_hypothesis_count"]
        - payload["certified_numerical_relation_count"]
    )
    diagnostics = payload["rejected_candidate_diagnostics"]
    assert diagnostics["total_rejected_hypothesis_count"] == payload[
        "rejected_hypothesis_count"
    ]
    assert diagnostics["full_rejection_sha256"] == (
        discovery.report.rejected_decision_sha256
    )


def test_warmup_requires_disjoint_candidate_and_verification_points() -> None:
    dag, model = _ambiguous_projective_dag()
    observations = _complete_current_observations(dag, domain=100_000)
    points = _observation_points("shared")

    with pytest.raises(ValueError, match="point contract"):
        discover_generic_dag_numerical_current_relations(
            dag,
            model,
            candidate_observations=observations,
            verification_observations=observations,
            **_complete_observation_evidence(
                dag,
                model,
                observations,
                observations,
                candidate_points=points,
                verification_points=points,
            ),
            execution_mode="compiled",
            precision_digits=96,
            seed=0x5059414D,
            relative_tolerance=1.0e-70,
            absolute_tolerance=1.0e-80,
        )
