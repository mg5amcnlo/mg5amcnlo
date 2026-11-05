# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import gc
import hashlib
import json
import math
import tempfile
import weakref
import zlib
from collections.abc import Iterator, Mapping
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import pyamplicol.generation.recurrence_numerical_current_warmup as recurrence_warmup
import pyamplicol.generation.service as generation_service
from pyamplicol.api.errors import ArtifactError, GenerationError
from pyamplicol.generation.dag_equivalence import (
    _advance_rejection_digest,
    _rejected_candidate_diagnostics,
)
from pyamplicol.generation.numerical_candidate_index import (
    build_numerical_observation_candidate_index,
    numerical_observation_tolerance_window_ids,
)
from pyamplicol.generation.recurrence_numerical_current_warmup import (
    _COMPRESSED_EVIDENCE_HEADER,
    _MIN_RAW_EVIDENCE_WIRE_BYTES,
    _RAW_EVIDENCE_MEMORY_TARGET_BYTES,
    RecurrenceNumericalEvidenceEnvelopeExceeded,
    RecurrenceNumericalEvidenceGeometry,
    _build_candidate_indexes,
    _decimal_string,
    _discover_relations,
    _pair_residuals,
    _raw_evidence_memory_upper_bound,
    _raw_evidence_wire_byte_limit,
    _read_compressed_evidence_spool,
    _relation_residuals,
    _runtime_parameter_schema_payload,
    _spooled_capture_memory_upper_bound,
    _SpooledObservationMapping,
    _synthetic_raw_evidence_bytes,
    _synthetic_raw_evidence_bytes_by_dimension_counts,
    _validate_raw_evidence_canonical_size,
    _validate_raw_evidence_geometry,
    _write_compressed_canonical_evidence,
    capture_recurrence_current_observations,
    recurrence_numerical_current_envelope_fallback_report,
    recurrence_numerical_current_opt_out_report,
    recurrence_numerical_relation_correctness_summary,
    recurrence_numerical_relation_fallback_summary,
    recurrence_numerical_source_semantics_sha256,
    run_recurrence_numerical_current_warmup,
    validate_recurrence_numerical_current_application,
    validate_recurrence_numerical_evidence_fallback,
)
from pyamplicol.generation.service import (
    _recurrence_relation_reporting,
)
from pyamplicol.generation.validation import ValidationPointRecord
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.runtime.recurrence_exact._plan import (
    _RecurrenceExactPlan,
    _RuntimeParameterSchemaRow,
    _SourceTemplate,
)
from pyamplicol.runtime.recurrence_exact._plan_v2 import (
    DIRECT_NONE_U32,
    _Contribution,
    _Current,
    _ExactFactor,
    _Executor,
    _MomentumForm,
    _MomentumTerm,
    _RecurrenceExactSectionsV1,
    _ReplayTarget,
    _ResolvedHelicity,
    _RowGroup,
    _Source,
)


def _factor(value: int) -> _ExactFactor:
    return _ExactFactor(value, 1, 0, 1)


def test_opposite_residual_is_independent_of_ambient_decimal_context() -> None:
    exact = Decimal("0.123456789012345678901234567890123456789012345678901234567890")
    with localcontext() as context:
        context.prec = 7
        residuals = _pair_residuals(
            ((Decimal(f"-{exact}"), Decimal(0)),),
            ((exact, Decimal(0)),),
            sign=-1,
            relative_tolerance=Decimal("1e-70"),
            absolute_tolerance=Decimal("1e-80"),
        )

    assert residuals == (0, 0, 0)


@pytest.mark.parametrize(
    ("process_expression", "process_id"),
    (
        ("d d~ > z", "d_dbar_to_z"),
        ("u d~ > w+", "u_dbar_to_w"),
    ),
)
def test_massive_one_body_recurrence_probe_domains_are_physical_and_distinct(
    process_expression: str,
    process_id: str,
) -> None:
    model = BuiltinSMModel()
    process = build_process_ir(process_expression, color_accuracy="full")
    probes = recurrence_warmup.build_recurrence_numerical_current_probe_points(
        process,
        model,
        process_id=process_id,
        seed=17,
        candidate_count=4,
        verification_count=4,
    )

    assert probes == (
        recurrence_warmup.build_recurrence_numerical_current_probe_points(
            process,
            model,
            process_id=process_id,
            seed=17,
            candidate_count=4,
            verification_count=4,
        )
    )
    candidate, verification = probes
    candidate_hashes = {
        recurrence_warmup._kinematic_sha256(point) for point in candidate
    }
    verification_hashes = {
        recurrence_warmup._kinematic_sha256(point) for point in verification
    }
    assert len(candidate_hashes) == len(candidate)
    assert len(verification_hashes) == len(verification)
    assert candidate_hashes.isdisjoint(verification_hashes)

    for point in (*candidate, *verification):
        assert point.available
        assert point.process == process_expression
        assert point.process_id == process_id
        assert all(
            math.isfinite(component)
            for _pdg, momentum in point.particles
            for component in momentum
        )
        incoming = point.four_vectors[:2]
        outgoing = point.four_vectors[2:]
        for component in range(4):
            assert math.isclose(
                sum(momentum[component] for momentum in incoming),
                sum(momentum[component] for momentum in outgoing),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
        for pdg, (energy, x_component, y_component, z_component) in point.particles:
            invariant_mass_squared = (
                energy * energy
                - x_component * x_component
                - y_component * y_component
                - z_component * z_component
            )
            expected_mass_squared = float(model.mass(pdg)) ** 2
            assert math.isclose(
                invariant_mass_squared,
                expected_mass_squared,
                rel_tol=1.0e-12,
                abs_tol=1.0e-10,
            )


def test_zero_residual_specialization_matches_explicit_zero_comparison() -> None:
    current = (
        (Decimal("0"), Decimal("-0")),
        (Decimal("1.25"), Decimal("-3.5")),
        (
            Decimal("0.123456789012345678901234567890123456789"),
            Decimal("-9.87654321098765432109876543210987654321"),
        ),
    )
    relative = Fraction(1, 10**70)
    absolute = Fraction(1, 10**80)

    expected = _pair_residuals(
        current,
        tuple((Decimal(0), Decimal(0)) for _ in current),
        sign=1,
        relative_tolerance=Decimal("1e-70"),
        absolute_tolerance=Decimal("1e-80"),
    )
    actual = _relation_residuals(
        "zero",
        current,
        None,
        relative_tolerance=relative,
        absolute_tolerance=absolute,
    )

    assert actual == expected
    with pytest.raises(ValueError, match="relation width is invalid"):
        _relation_residuals(
            "equal",
            current,
            None,
            relative_tolerance=relative,
            absolute_tolerance=absolute,
        )


def _topology_replay_plan() -> _RecurrenceExactPlan:
    sections = _RecurrenceExactSectionsV1(
        process_id="synthetic_replay",
        strategy="topology-replay",
        semantic_digest="1" * 64,
        runtime_layout_digest="2" * 64,
        current_arena_components=4,
        amplitude_destination_count=1,
        parameter_value_count=0,
        external_source_count=2,
        currents=(
            _Current(0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, DIRECT_NONE_U32),
            _Current(1, 0, 0, 1, 1, 1, 0, 0, 0, 4, 1, DIRECT_NONE_U32),
            _Current(
                2,
                1,
                0,
                2,
                1,
                0,
                1,
                0,
                2,
                4,
                DIRECT_NONE_U32,
                DIRECT_NONE_U32,
            ),
            _Current(
                3,
                1,
                0,
                3,
                1,
                0,
                1,
                0,
                3,
                4,
                DIRECT_NONE_U32,
                DIRECT_NONE_U32,
            ),
        ),
        sources=(
            _Source(0, 0, 0, 0, 0, 0, 0),
            _Source(1, 1, 1, 0, 0, 0, 0),
        ),
        contributions=(
            _Contribution(
                0,
                DIRECT_NONE_U32,
                1,
                DIRECT_NONE_U32,
                2,
                0,
                0,
                3,
            ),
            _Contribution(
                0,
                DIRECT_NONE_U32,
                1,
                DIRECT_NONE_U32,
                3,
                0,
                0,
                3,
            ),
        ),
        finalizations=(),
        closures=(),
        row_groups=(
            _RowGroup(0, 0, 0, 0, 0, 2),
            _RowGroup(1, 1, 1, DIRECT_NONE_U32, 0, 2),
        ),
        momentum_forms=(_MomentumForm(0, 1), _MomentumForm(1, 1)),
        momentum_terms=(_MomentumTerm(0, 1), _MomentumTerm(1, 1)),
        replay_targets=(
            _ReplayTarget(10, 0, 0, 2, 0, 0, 0, 1, 0),
            _ReplayTarget(11, 0, 2, 2, 0, 0, 0, 1, 0),
        ),
        source_permutations=(0, 1, 1, 0),
        replay_momentum_signs=(1, 1, 1, 1),
        replay_helicity_map=(),
        amplitude_destinations=(),
        resolved_helicities=(),
        source_state_assignments=(),
        source_dispatch_variants=(),
        source_embeddings=(),
        source_projections=(),
        resolved_source_selections=(),
        public_helicities=(),
        exact_factors=(_factor(1),),
        public_flow_ids=(10, 11),
        executors=(
            _Executor(
                0,
                "source",
                "initialize",
                (),
                1,
                1,
                None,
                "source",
            ),
        ),
    )
    return _RecurrenceExactPlan(
        sections=sections,
        kernels={},
        executors={0: sections.executors[0]},
        executor_exact_kernel_ids={},
        executor_parent_permutations={0: (0, 1)},
        source_templates={
            0: _SourceTemplate(
                0,
                1,
                0,
                0,
                0,
                "scalar",
                "self-conjugate",
                None,
                1,
                1,
                1,
            )
        },
        initial_source_slots=frozenset({0, 1}),
        executor_couplings={},
        prepared_defaults=(),
        parameter_projection=(),
        parameter_derivation=None,
        runtime_defaults=(Decimal("1.25"), Decimal("2.5")),
        runtime_parameter_schema=(
            _RuntimeParameterSchemaRow(0, "probe_a", 0, None, 0),
            _RuntimeParameterSchemaRow(1, "probe_b", 1, None, 0),
        ),
    )


def _no_relation_plan() -> _RecurrenceExactPlan:
    plan = _topology_replay_plan()
    contributions = list(plan.sections.contributions)
    contributions[1] = replace(contributions[1], exact_factor_id=1)
    plan.sections = replace(
        plan.sections,
        contributions=tuple(contributions),
        exact_factors=(*plan.sections.exact_factors, _factor(2)),
    )
    return plan


def _signed_relation_plan() -> _RecurrenceExactPlan:
    plan = _topology_replay_plan()
    prototype = plan.sections.currents[3]
    contribution = plan.sections.contributions[0]
    plan.sections = replace(
        plan.sections,
        current_arena_components=6,
        currents=(
            *plan.sections.currents,
            replace(
                prototype,
                semantic_id=4,
                component_base=4,
                first_use=4,
                last_use=4,
            ),
            replace(
                prototype,
                semantic_id=5,
                component_base=5,
                first_use=5,
                last_use=5,
            ),
        ),
        contributions=(
            *plan.sections.contributions,
            replace(
                contribution,
                destination_base=4,
                exact_factor_id=1,
            ),
            replace(
                contribution,
                destination_base=5,
                exact_factor_id=2,
            ),
        ),
        row_groups=(
            plan.sections.row_groups[0],
            replace(plan.sections.row_groups[1], row_count=4),
        ),
        exact_factors=(
            *plan.sections.exact_factors,
            _factor(-1),
            _factor(0),
        ),
    )
    return plan


def _contracted_signed_relation_plan() -> _RecurrenceExactPlan:
    plan = _signed_relation_plan()
    plan.sections = replace(
        plan.sections, strategy="contracted-color-union", replay_targets=()
    )
    return plan


def _points(seed_start: int) -> tuple[ValidationPointRecord, ...]:
    return tuple(
        ValidationPointRecord(
            process_id="synthetic_replay",
            process="s s > s s",
            seed=seed,
            particles=(
                (1, (10.0 + seed, 0.0, 0.0, 10.0 + seed)),
                (-1, (20.0 + seed, 0.0, 0.0, -20.0 - seed)),
            ),
        )
        for seed in range(seed_start, seed_start + 2)
    )


def test_replay_capture_observes_contribution_only_currents() -> None:
    plan = _topology_replay_plan()
    source_digest = recurrence_numerical_source_semantics_sha256(plan.sections)
    captured = capture_recurrence_current_observations(
        plan,
        _points(1),
        precision_digits=80,
        source_semantics_sha256=source_digest,
        seed=41,
        domain="candidate-current-probes-v1",
    )

    assert captured.current_count == 4
    assert captured.current_dimensions == {0: 1, 1: 1, 2: 1, 3: 1}
    assert captured.observations[2] == captured.observations[3]
    assert captured.context_policy == "seeded-replay-target-per-physical-point-v1"
    assert len(captured.context_sha256s) == 2


def test_python_parameter_default_hex_matches_native_signed_binary64_contract() -> None:
    plan = _topology_replay_plan()
    plan.runtime_defaults = (
        Decimal.from_float(-1.5),
        Decimal.from_float(float.fromhex("-0x0.0000000000001p-1022")),
    )

    payload = _runtime_parameter_schema_payload(plan)

    assert [row["default_binary64"] for row in payload["parameters"]] == [
        "-0x1.8000000000000p+0",
        "-0x0.0000000000001p-1022",
    ]


def test_warmup_builds_static_semantics_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_calls = 0
    default_calls = 0
    source_semantics_calls = 0
    original_contracts = recurrence_warmup._current_contracts
    original_defaults = recurrence_warmup._runtime_parameter_defaults
    original_source_semantics = recurrence_warmup._source_semantics_payload

    def counted_contracts(
        sections: _RecurrenceExactSectionsV1,
    ) -> tuple[tuple[object, ...], ...]:
        nonlocal contract_calls
        contract_calls += 1
        return original_contracts(sections)

    def counted_defaults(plan: _RecurrenceExactPlan) -> tuple[Decimal, ...]:
        nonlocal default_calls
        default_calls += 1
        return original_defaults(plan)

    def counted_source_semantics(
        sections: _RecurrenceExactSectionsV1,
        *,
        contracts: tuple[tuple[object, ...], ...] | None = None,
    ) -> dict[str, object]:
        nonlocal source_semantics_calls
        source_semantics_calls += 1
        return original_source_semantics(sections, contracts=contracts)

    monkeypatch.setattr(recurrence_warmup, "_current_contracts", counted_contracts)
    monkeypatch.setattr(
        recurrence_warmup,
        "_runtime_parameter_defaults",
        counted_defaults,
    )
    monkeypatch.setattr(
        recurrence_warmup,
        "_source_semantics_payload",
        counted_source_semantics,
    )
    result = run_recurrence_numerical_current_warmup(
        _topology_replay_plan(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="lc",
        precision_digits=80,
        seed=53,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )
    try:
        assert contract_calls == 1
        assert default_calls == 1
        assert source_semantics_calls == 1
        candidate_dimensions = result.candidate_capture.current_dimensions
        verification_dimensions = result.verification_capture.current_dimensions
        assert candidate_dimensions == verification_dimensions
        assert candidate_dimensions is not verification_dimensions
        assert isinstance(candidate_dimensions, dict)
        assert isinstance(verification_dimensions, dict)
        current_id = next(iter(candidate_dimensions))
        verification_dimension = verification_dimensions[current_id]
        candidate_dimensions[current_id] = verification_dimension + 1
        assert verification_dimensions[current_id] == verification_dimension
    finally:
        result.close()


def test_warmup_phase_timings_are_diagnostic_only() -> None:
    arguments = {
        "candidate_points": _points(1),
        "verification_points": _points(101),
        "mode": "certified-reuse",
        "color_accuracy": "lc",
        "precision_digits": 80,
        "seed": 53,
        "relative_tolerance": 1.0e-60,
        "absolute_tolerance": 1.0e-70,
    }
    result = run_recurrence_numerical_current_warmup(
        _topology_replay_plan(),
        **arguments,
    )
    repeated = run_recurrence_numerical_current_warmup(
        _topology_replay_plan(),
        **arguments,
    )
    try:
        assert result.generation_profile_timings
        assert all(
            isinstance(seconds, float) and seconds >= 0.0
            for seconds in result.generation_profile_timings.values()
        )
        assert {
            key
            for key in result.generation_profile_timings
            if key.startswith("warmup_candidate_probe_")
        } == {"warmup_candidate_probe_0", "warmup_candidate_probe_1"}
        assert {
            key
            for key in result.generation_profile_timings
            if key.startswith("warmup_verification_probe_")
        } == {"warmup_verification_probe_0", "warmup_verification_probe_1"}
        assert "generation_profile_timings" not in json.dumps(
            result.to_json_dict(),
            sort_keys=True,
        )
        assert result.evidence_json == repeated.evidence_json
        assert result == repeated
        assert result == replace(
            result,
            generation_profile_timings={"diagnostic-only": 1.0},
        )
        with pytest.raises(TypeError):
            result.generation_profile_timings["not-mutable"] = 1.0  # type: ignore[index]
    finally:
        result.close()
        repeated.close()


def test_recurrence_certified_reuse_uses_two_independent_probe_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = Mock(wraps=recurrence_warmup._capture_recurrence_current_observations)
    monkeypatch.setattr(
        recurrence_warmup,
        "_capture_recurrence_current_observations",
        capture,
    )
    result = run_recurrence_numerical_current_warmup(
        _topology_replay_plan(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="lc",
        precision_digits=80,
        seed=53,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )

    assert result.applied_relation_count == 1
    certificate = result.certificates[0]
    assert certificate.current_id == 3
    assert certificate.representative_id == 2
    assert certificate.execution_representative_id == 2
    assert certificate.relation_kind == "equal"
    assert certificate.current_dimension == 1
    assert certificate.candidate_probe_count == 2
    assert certificate.verification_probe_count == 2
    assert result.candidate_capture.current_count == 4
    assert result.verification_capture.current_count == 4
    assert set(result.candidate_capture.observations) == {2, 3}
    assert set(result.verification_capture.observations) == {2, 3}
    assert result.candidate_capture.complete_observations_retained is False
    assert result.verification_capture.complete_observations_retained is False
    with pytest.raises(ValueError, match="cannot recreate raw evidence"):
        result.candidate_capture.to_evidence_dict()
    assert set(result.candidate_capture.kinematic_sha256s).isdisjoint(
        result.verification_capture.kinematic_sha256s
    )
    evidence = json.loads(result.evidence_json)
    assert evidence["requested_mode"] == "certified-reuse"
    assert "relative_tolerance" not in evidence
    assert "absolute_tolerance" not in evidence
    assert evidence["relative_tolerance_binary64"] == (1.0e-60).hex()
    assert evidence["absolute_tolerance_binary64"] == (1.0e-70).hex()
    assert evidence["mappings"][0]["factor_integer"] == [1, 0]
    assert len(evidence["candidate_capture"]["observations"]) == 4
    assert len(evidence["verification_capture"]["observations"]) == 4
    report = result.to_json_dict()
    assert report["warning"]["required"] is True
    rejected = report["discovery"]["rejected_candidate_diagnostics"]
    assert rejected["retained_count"] == len(report["discovery"]["rejected_candidates"])
    assert rejected["total_rejected_hypothesis_count"] >= rejected["retained_count"]
    assert rejected["full_decision_sha256"] == report["discovery"]["decision_sha256"]
    persisted = report["persisted_numerical_evidence"]
    assert persisted["raw_evidence_retained"] is False
    census = persisted["full_census"]
    assert (
        census["uncertified_candidate_count"] == census["verification_rejected_count"]
    )
    assert census["numerical_candidate_count"] == (
        census["certified_relation_count"] + census["uncertified_candidate_count"]
    )
    assert census["rejected_hypothesis_count"] == (
        census["tested_hypothesis_count"] - census["certified_relation_count"]
    )
    assert (
        census["rejection_decision_sha256"]
        == report["discovery"]["rejection_decision_sha256"]
    )
    assert persisted["measured_payload_bytes"] > 0
    assert {
        row["current_id"] for row in persisted["candidate_capture"]["observations"]
    } == {2, 3}
    assert persisted["candidate_capture"]["full_batch_commitment"]["current_count"] == 4
    assert len(json.dumps(persisted, separators=(",", ":"), sort_keys=True)) < len(
        result.evidence_json
    )
    assert [call.kwargs["domain"] for call in capture.call_args_list] == [
        "candidate-current-probes-v1",
        "independent-verification-current-probes-v1",
    ]

    validated = validate_recurrence_numerical_current_application(
        result,
        _topology_replay_plan(),
        baseline_plan=_topology_replay_plan(),
        precision_digits=80,
        seed=53,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )
    assert validated.application_validation["status"] == "verified"
    assert validated.application_validation["maximum_absolute_residual"] == "0"


def test_recurrence_diagnostic_certifies_without_applying_or_warning() -> None:
    result = run_recurrence_numerical_current_warmup(
        _topology_replay_plan(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="diagnostic",
        color_accuracy="lc",
        precision_digits=80,
        seed=67,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )

    assert len(result.certificates) == 1
    assert result.applied_relation_count == 0
    assert result.candidate_capture.current_count == 4
    assert result.verification_capture.current_count == 4
    assert result.candidate_capture.observations == {}
    assert result.verification_capture.observations == {}
    assert result.candidate_capture.complete_observations_retained is False
    assert result.verification_capture.complete_observations_retained is False
    report = result.to_json_dict()
    assert report["state"] == "authenticated-numerical-diagnostic-only"
    assert report["warning"]["required"] is False
    assert json.loads(result.evidence_json)["requested_mode"] == "diagnostic"


def test_recurrence_no_relation_skips_independent_verification_with_full_census(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = Mock(wraps=recurrence_warmup._capture_recurrence_current_observations)
    monkeypatch.setattr(
        recurrence_warmup,
        "_capture_recurrence_current_observations",
        capture,
    )
    result = run_recurrence_numerical_current_warmup(
        _no_relation_plan(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="lc",
        precision_digits=80,
        seed=71,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )

    assert result.certificates == ()
    assert result.candidate_capture.current_count == 4
    assert result.candidate_capture.observations == {}
    assert result.verification_capture is None
    assert result.evidence_json == b""
    assert result.evidence_transport_bytes == 0
    assert result.evidence_canonical_bytes == 0
    assert result.evidence_encoding is None
    assert result.discovery_report["inspected_current_count"] == 4
    assert result.discovery_report["tested_hypothesis_count"] > 0
    assert result.discovery_report["numerical_candidate_count"] == 0
    probe_contract = result.discovery_report["probe_contract"]
    assert probe_contract["independent_verification"] is False
    assert probe_contract["verification_status"] == (
        "not-required-no-candidate-hypothesis"
    )
    assert probe_contract["verification_point_sha256s"] == []
    assert probe_contract["verification_capture_sha256"] is None
    report = result.to_json_dict()
    assert report["verification_capture"] is None
    persisted = report["persisted_numerical_evidence"]
    assert persisted["verification_capture"] is None
    assert persisted["generation_raw_evidence_bytes"] == 0
    assert persisted["generation_evidence_transport_bytes"] == 0
    assert persisted["generation_evidence_encoding"] is None
    assert [call.kwargs["domain"] for call in capture.call_args_list] == [
        "candidate-current-probes-v1"
    ]
    result.close()
    result.close()


def test_recurrence_no_relation_returns_before_compressed_candidate_spooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recurrence_warmup,
        "_select_raw_evidence_storage_geometry",
        lambda *_args, **_kwargs: recurrence_warmup._RawEvidenceStorageGeometry(
            scalar_count=128,
            row_count=16,
            canonical_byte_limit=1 << 20,
            encoding="zlib-canonical-json-v1",
            producer_resident_upper_bound=1 << 20,
        ),
    )
    result = run_recurrence_numerical_current_warmup(
        _no_relation_plan(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="lc",
        precision_digits=80,
        seed=71,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )

    assert result.candidate_capture.observations == {}
    assert result.verification_capture is None
    assert "warmup_candidate_spool" not in result.generation_profile_timings
    assert "warmup_verification_spool" not in result.generation_profile_timings
    result.close()
    result.close()


def test_application_recapture_rejects_stale_plan_and_capture_commitment() -> None:
    result = run_recurrence_numerical_current_warmup(
        _topology_replay_plan(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="lc",
        precision_digits=80,
        seed=73,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )

    with pytest.raises(ValueError, match="did not reproduce"):
        validate_recurrence_numerical_current_application(
            result,
            _topology_replay_plan(),
            baseline_plan=_no_relation_plan(),
            precision_digits=80,
            seed=73,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
        )

    tampered = replace(
        result,
        verification_capture=replace(
            result.verification_capture,
            observation_batch_sha256="0" * 64,
        ),
    )
    with pytest.raises(ValueError, match="did not reproduce"):
        validate_recurrence_numerical_current_application(
            tampered,
            _topology_replay_plan(),
            baseline_plan=_topology_replay_plan(),
            precision_digits=80,
            seed=73,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
        )


def test_recurrence_opt_out_has_no_capture_or_warning() -> None:
    report = recurrence_numerical_current_opt_out_report(
        _topology_replay_plan().sections,
        color_accuracy="lc",
    )

    assert report["requested_mode"] == "off"
    assert report["state"] == "disabled-by-user"
    assert report["candidate_capture"] is None
    assert report["verification_capture"] is None
    assert report["applied_relation_count"] == 0
    assert report["warning"]["required"] is False


def test_raw_evidence_scaling_is_quantified_without_a_fixed_ceiling() -> None:
    common = {
        "component_count": 4,
        "candidate_probe_count": 4,
        "verification_probe_count": 4,
        "decimal_characters": 112,
    }
    b_like_bytes = _synthetic_raw_evidence_bytes(652, **common)
    a_like_bytes = _synthetic_raw_evidence_bytes(17_000, **common)

    assert b_like_bytes == 4_918_550
    assert a_like_bytes == 128_293_862
    assert a_like_bytes > 25 * b_like_bytes
    a_like_upper_bound = _raw_evidence_memory_upper_bound(
        a_like_bytes,
        scalar_count=1_088_000,
        row_count=34_000,
    )
    assert a_like_upper_bound == 1_003_870_156
    assert a_like_upper_bound < _RAW_EVIDENCE_MEMORY_TARGET_BYTES
    just_over = _synthetic_raw_evidence_bytes(50_000, **common)
    assert just_over > a_like_bytes
    _validate_raw_evidence_canonical_size(just_over)


def test_real_a_mixed_dimension_shape_uses_dynamic_wire_budget() -> None:
    dimension_counts = {4: 15_834, 6: 1_240}
    current_count = sum(dimension_counts.values())
    component_count = sum(
        dimension * count for dimension, count in dimension_counts.items()
    )
    point_count = 4 + 4
    runtime_parameter_count = 10
    scalar_count = (
        2 * component_count * point_count + runtime_parameter_count * point_count
    )
    row_count = current_count * 2 + runtime_parameter_count
    byte_limit = _raw_evidence_wire_byte_limit(
        scalar_count=scalar_count,
        row_count=row_count,
    )

    assert (current_count, component_count) == (17_074, 70_776)
    assert (scalar_count, row_count) == (1_132_496, 34_158)
    assert byte_limit == 148_950_528
    wire_sizes = {
        characters: _synthetic_raw_evidence_bytes_by_dimension_counts(
            dimension_counts,
            candidate_probe_count=4,
            verification_probe_count=4,
            decimal_characters=characters,
        )
        for characters in (96, 112, 128)
    }
    assert wire_sizes == {
        96: 115_356_478,
        112: 133_475_134,
        128: 151_593_790,
    }
    _validate_raw_evidence_canonical_size(
        wire_sizes[96],
        byte_limit=byte_limit,
    )
    _validate_raw_evidence_canonical_size(
        wire_sizes[112],
        byte_limit=byte_limit,
    )
    with pytest.raises(ValueError, match="memory envelope"):
        _validate_raw_evidence_canonical_size(
            wire_sizes[128],
            byte_limit=byte_limit,
        )


def test_runtime_parameter_metadata_reduces_dynamic_wire_budget() -> None:
    without_parameters = _raw_evidence_wire_byte_limit(
        scalar_count=1_132_416,
        row_count=34_148,
    )
    with_ten_parameters = _raw_evidence_wire_byte_limit(
        scalar_count=1_132_496,
        row_count=34_158,
    )

    assert without_parameters == 148_978_688
    assert with_ten_parameters == 148_950_528
    assert without_parameters - with_ten_parameters == 28_160


def test_parameter_context_geometry_is_preflighted_without_allocation() -> None:
    sections = _topology_replay_plan().sections
    scalar_count, row_count, byte_limit = _validate_raw_evidence_geometry(
        sections,
        candidate_probe_count=2,
        verification_probe_count=2,
        runtime_parameter_count=400_000,
    )
    assert scalar_count > 400_000
    assert row_count > 400_000
    assert byte_limit is None


def test_exact_z_n8_sequential_spool_bound_includes_the_global_index() -> None:
    resident = _spooled_capture_memory_upper_bound(
        current_count=38_581,
        component_count=162_976,
        maximum_probe_count=4,
        runtime_parameter_count=10,
    )

    assert resident == 935_671_296
    assert resident < _RAW_EVIDENCE_MEMORY_TARGET_BYTES
    assert _RAW_EVIDENCE_MEMORY_TARGET_BYTES - resident == 138_070_528


def test_large_geometry_selects_spool_without_a_default_memory_limit() -> None:
    geometry = RecurrenceNumericalEvidenceGeometry.from_counts(
        current_count=116_319,
        component_count=561_426,
        candidate_probe_count=4,
        verification_probe_count=4,
        runtime_parameter_count=10,
    )

    selected = recurrence_warmup._select_raw_evidence_storage_geometry_for_counts(
        geometry
    )
    assert selected.encoding == "zlib-canonical-json-v1"
    assert selected.canonical_byte_limit is None
    assert selected.producer_resident_upper_bound == 3_095_140_864
    explicitly_allowed = (
        recurrence_warmup._select_raw_evidence_storage_geometry_for_counts(
            geometry, memory_envelope_bytes=4 << 30
        )
    )
    assert explicitly_allowed.producer_resident_upper_bound == 3_095_140_864


@pytest.mark.parametrize(
    "memory_envelope_bytes",
    (True, 0, -1),
)
def test_invalid_injected_envelope_is_not_converted_to_geometry_fallback(
    memory_envelope_bytes: object,
) -> None:
    geometry = RecurrenceNumericalEvidenceGeometry.from_counts(
        current_count=4,
        component_count=4,
        candidate_probe_count=2,
        verification_probe_count=2,
        runtime_parameter_count=2,
    )

    with pytest.raises(ValueError, match="memory envelope is invalid") as captured:
        recurrence_warmup._select_raw_evidence_storage_geometry_for_counts(
            geometry,
            memory_envelope_bytes=memory_envelope_bytes,  # type: ignore[arg-type]
        )

    assert not isinstance(
        captured.value,
        RecurrenceNumericalEvidenceEnvelopeExceeded,
    )


def test_geometry_fallback_precedes_static_context_and_current_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = RecurrenceNumericalEvidenceGeometry.from_counts(
        current_count=4,
        component_count=4,
        candidate_probe_count=2,
        verification_probe_count=2,
        runtime_parameter_count=2,
    )
    outcome = RecurrenceNumericalEvidenceEnvelopeExceeded(
        geometry,
        memory_envelope_bytes=1,
        spooled_producer_resident_bytes=2,
        raw_reason="injected test envelope",
    )
    monkeypatch.setattr(
        recurrence_warmup,
        "_select_raw_evidence_storage_geometry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(outcome),
    )
    monkeypatch.setattr(
        recurrence_warmup,
        "_build_warmup_static_context",
        lambda *_args, **_kwargs: pytest.fail(
            "static context was built after an envelope fallback"
        ),
    )

    with pytest.raises(RecurrenceNumericalEvidenceEnvelopeExceeded) as raised:
        run_recurrence_numerical_current_warmup(
            _topology_replay_plan(),
            candidate_points=_points(1),
            verification_points=_points(101),
            mode="certified-reuse",
            color_accuracy="lc",
            precision_digits=80,
            seed=53,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
        )

    assert raised.value is outcome


def test_envelope_fallback_has_reuse_off_runtime_and_visible_provenance() -> None:
    geometry = RecurrenceNumericalEvidenceGeometry.from_counts(
        current_count=4,
        component_count=4,
        candidate_probe_count=2,
        verification_probe_count=2,
        runtime_parameter_count=2,
    )
    with pytest.raises(RecurrenceNumericalEvidenceEnvelopeExceeded) as captured:
        recurrence_warmup._select_raw_evidence_storage_geometry_for_counts(
            geometry,
            memory_envelope_bytes=1 << 20,
        )
    plan = _topology_replay_plan()
    fallback = recurrence_numerical_current_envelope_fallback_report(
        plan.sections,
        color_accuracy="lc",
        requested_mode="certified-reuse",
        outcome=captured.value,
    )
    explicit_off = recurrence_numerical_current_opt_out_report(
        plan.sections,
        color_accuracy="lc",
    )

    fallback_runtime, fallback_aggregate = _recurrence_relation_reporting(
        {},
        mode="certified-reuse",
        lane_report=fallback,
    )
    off_runtime, off_aggregate = _recurrence_relation_reporting(
        {},
        mode="off",
        lane_report=explicit_off,
    )

    assert fallback_runtime == off_runtime == {}
    assert fallback_aggregate["effective_mode"] == "off"
    assert fallback_aggregate["effective_reuse_state"] == "disabled"
    assert fallback_aggregate["applied_relation_count"] == 0
    assert fallback_aggregate["fallback"]["reason"] == ("evidence-envelope-fallback")
    assert fallback_aggregate["warning"]["required"] is True
    assert off_aggregate["applied_relation_count"] == 0
    assert recurrence_numerical_relation_correctness_summary(fallback_aggregate) == {
        "abi": "pyamplicol-numerical-current-relation-correctness-v1",
        "state": "no-applied-relations",
        "applied_relation_count": 0,
    }
    assert recurrence_numerical_relation_fallback_summary(fallback_aggregate) == {
        "abi": "pyamplicol-numerical-current-reuse-fallback-v1",
        "requested_mode": "certified-reuse",
        "effective_mode": "off",
        "effective_reuse_state": "disabled",
        "reason": "evidence-envelope-fallback",
        "geometry": geometry.to_json_dict(),
        "certified_relation_count": 0,
        "applied_relation_count": 0,
    }
    with pytest.raises(ValueError, match="fallback provenance is absent"):
        recurrence_numerical_relation_fallback_summary(
            {**fallback_aggregate, "fallback": None}
        )
    fallback_payload = fallback["fallback"]
    assert isinstance(fallback_payload, Mapping)
    fallback_geometry = fallback_payload["geometry"]
    assert isinstance(fallback_geometry, Mapping)

    for mutation in (
        {"effective_reuse_state": "enabled"},
        {"state": "disabled-by-user"},
        {"fallback": {"reason": "evidence-envelope-fallback"}},
        {
            "fallback": {
                **fallback_payload,
                "geometry": {
                    **fallback_geometry,
                    "scalar_count": 1,
                },
            }
        },
        {
            "fallback": {
                **fallback_payload,
                "spooled_producer_resident_bytes": (
                    int(fallback_payload["spooled_producer_resident_bytes"]) + 1
                ),
            }
        },
    ):
        with pytest.raises(GenerationError):
            _recurrence_relation_reporting(
                {},
                mode="certified-reuse",
                lane_report={**fallback, **mutation},
            )

    with pytest.raises(GenerationError, match="opt-out metadata"):
        _recurrence_relation_reporting(
            {},
            mode="off",
            lane_report={**explicit_off, "effective_reuse_state": "enabled"},
        )


def _fallback_lowering_output(
    tmp_path: Path,
    name: str,
    *,
    inspection: Mapping[str, object],
) -> generation_service._RustRecurrenceLoweringOutput:
    return generation_service._RustRecurrenceLoweringOutput(
        inspection_summary=inspection,
        resolved_helicities=(),
        amplitude_destinations=(),
        structural_zero_physical_sector_certificate=None,
        exact_sections={},
        generation_profile={"pass": name},
        numerical_current_reuse_report=None,
        payload_path=tmp_path / f"{name}.pacbin",
        payload_size_bytes=1,
        payload_sha256="a" * 64,
        member_count=1,
        unpacked_size_bytes=1,
        index_sha256="b" * 64,
    )


@pytest.mark.parametrize(
    ("plan_factory", "mode", "expected_lowering_calls"),
    (
        (_no_relation_plan, "certified-reuse", 1),
        (_topology_replay_plan, "diagnostic", 2),
    ),
)
def test_service_only_skips_second_lowering_for_zero_certificates(
    tmp_path: Path,
    plan_factory: object,
    mode: str,
    expected_lowering_calls: int,
) -> None:
    assert callable(plan_factory)
    warmup = run_recurrence_numerical_current_warmup(
        plan_factory(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode=mode,  # type: ignore[arg-type]
        color_accuracy="lc",
        precision_digits=80,
        seed=71,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )
    calls: list[tuple[Path, dict[str, object]]] = []

    def lower_once(
        destination: Path,
        **kwargs: object,
    ) -> generation_service._RustRecurrenceLoweringOutput:
        calls.append((destination, dict(kwargs)))
        pass_name = "baseline" if len(calls) == 1 else "final"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"{pass_name}-payload".encode("ascii"))
        return replace(
            _fallback_lowering_output(
                tmp_path,
                pass_name,
                inspection={"runtime": pass_name},
            ),
            resolved_helicities=((len(calls),),),
            exact_sections={"pass": pass_name},
            payload_path=destination,
            payload_size_bytes=destination.stat().st_size,
            payload_sha256=str(len(calls)) * 64,
        )

    baseline = lower_once(
        tmp_path / ".baseline" / "recurrence-runtime.pacbin",
        mode="off",
        evidence=None,
        report_progress=True,
        timing_name="native-baseline-generation",
    )
    final_path = tmp_path / "final" / "recurrence-runtime.pacbin"
    if not warmup.certificates:
        warmup = warmup.without_evidence_transport()
    try:
        output, application_validation_required = (
            generation_service._lower_recurrence_after_numerical_warmup(
                lower_once=lower_once,
                final_schedule_path=final_path,
                baseline=baseline,
                warmup=warmup,
                requested_mode=mode,
            )
        )
        assert len(calls) == expected_lowering_calls
        assert application_validation_required is bool(warmup.certificates)
        assert output.payload_path == final_path
        if not warmup.certificates:
            assert final_path.read_bytes() == b"baseline-payload"
            assert not baseline.payload_path.exists()
            assert output.resolved_helicities == baseline.resolved_helicities
            assert output.exact_sections == baseline.exact_sections
            assert output.inspection_summary == {"runtime": "baseline"}
            assert output.generation_profile == {
                "schema_version": 1,
                "native_passes": {"final": {"pass": "baseline"}},
            }
            report = output.numerical_current_reuse_report
            assert isinstance(report, Mapping)
            assert report["requested_mode"] == mode
            assert report["effective_mode"] == warmup.effective_mode
            assert report["certified_relation_count"] == 0
            assert report["applied_relation_count"] == 0
            assert report["native_relation_application"] is None
        else:
            assert mode == "diagnostic"
            assert final_path.read_bytes() == b"final-payload"
            assert output.numerical_current_reuse_report is None
            assert calls[1][1] == {
                "mode": warmup.effective_mode,
                "evidence": warmup.evidence_json,
                "report_progress": False,
                "timing_name": "native-final-generation",
            }
    finally:
        warmup.close()


def test_service_fallback_publishes_the_existing_reuse_off_lowering(
    tmp_path: Path,
) -> None:
    geometry = RecurrenceNumericalEvidenceGeometry.from_counts(
        current_count=4,
        component_count=4,
        candidate_probe_count=2,
        verification_probe_count=2,
        runtime_parameter_count=2,
    )
    outcome = RecurrenceNumericalEvidenceEnvelopeExceeded(
        geometry,
        memory_envelope_bytes=1,
        spooled_producer_resident_bytes=(
            geometry.spooled_producer_resident_upper_bound()
        ),
        raw_reason="injected test envelope",
    )

    calls: list[tuple[Path, dict[str, object]]] = []

    def lower_once(
        destination: Path,
        **kwargs: object,
    ) -> generation_service._RustRecurrenceLoweringOutput:
        calls.append((destination, dict(kwargs)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"validated-mode-off-payload")
        return replace(
            _fallback_lowering_output(
                tmp_path,
                "baseline",
                inspection={"runtime": "original-plan"},
            ),
            payload_path=destination,
            payload_size_bytes=destination.stat().st_size,
        )

    baseline_path = tmp_path / ".baseline" / "recurrence-runtime.pacbin"
    baseline = lower_once(
        baseline_path,
        mode="off",
        evidence=None,
        report_progress=True,
        timing_name="native-baseline-generation",
    )
    final_schedule_path = tmp_path / "recurrence-runtime.pacbin"
    completed = generation_service._complete_recurrence_evidence_envelope_fallback(
        final_schedule_path=final_schedule_path,
        baseline=baseline,
        baseline_plan=_topology_replay_plan(),
        color_accuracy="lc",
        outcome=outcome,
    )

    assert calls == [
        (
            baseline_path,
            {
                "mode": "off",
                "evidence": None,
                "report_progress": True,
                "timing_name": "native-baseline-generation",
            },
        )
    ]
    assert completed.payload_path == final_schedule_path
    assert final_schedule_path.read_bytes() == b"validated-mode-off-payload"
    assert not baseline_path.exists()
    assert completed.inspection_summary == {"runtime": "original-plan"}
    assert completed.generation_profile == {
        "schema_version": 1,
        "native_passes": {"final": {"pass": "baseline"}},
    }
    report = completed.numerical_current_reuse_report
    assert isinstance(report, Mapping)
    assert report["requested_mode"] == "certified-reuse"
    assert report["effective_mode"] == "off"
    assert report["effective_reuse_state"] == "disabled"
    assert report["applied_relation_count"] == 0
    report_fallback = report["fallback"]
    assert isinstance(report_fallback, Mapping)
    assert report_fallback["geometry"] == geometry.to_json_dict()


def test_python_envelope_fallback_detaches_exception_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetainedFrameResource:
        pass

    geometry = RecurrenceNumericalEvidenceGeometry.from_counts(
        current_count=4,
        component_count=4,
        candidate_probe_count=2,
        verification_probe_count=2,
        runtime_parameter_count=2,
    )
    retained: list[weakref.ReferenceType[RetainedFrameResource]] = []

    def fail_warmup() -> None:
        frame_resource = RetainedFrameResource()
        retained.append(weakref.ref(frame_resource))
        try:
            raise RuntimeError("canonical evidence writer overflow")
        except RuntimeError as cause:
            raise RecurrenceNumericalEvidenceEnvelopeExceeded(
                geometry,
                memory_envelope_bytes=1,
                spooled_producer_resident_bytes=(
                    geometry.spooled_producer_resident_upper_bound()
                ),
                raw_reason="injected Python warmup envelope",
            ) from cause

    try:
        fail_warmup()
    except RecurrenceNumericalEvidenceEnvelopeExceeded as exc:
        detached = generation_service._detached_recurrence_evidence_envelope_outcome(
            exc,
            source_error=exc,
        )
    else:  # pragma: no cover - the injected fallback is mandatory
        raise AssertionError("injected Python fallback did not fail")

    assert detached.__traceback__ is None
    assert detached.__cause__ is None
    assert detached.__context__ is None
    assert detached.to_json_dict()["geometry"] == geometry.to_json_dict()

    baseline = _fallback_lowering_output(tmp_path, "baseline", inspection={})
    baseline.payload_path.write_bytes(b"baseline")
    publish = generation_service._publish_validated_recurrence_baseline

    def publish_without_retained_exception(
        *,
        baseline: generation_service._RustRecurrenceLoweringOutput,
        final_schedule_path: Path,
    ) -> generation_service._RustRecurrenceLoweringOutput:
        gc.collect()
        assert retained[0]() is None
        return publish(
            baseline=baseline,
            final_schedule_path=final_schedule_path,
        )

    monkeypatch.setattr(
        generation_service,
        "_publish_validated_recurrence_baseline",
        publish_without_retained_exception,
    )

    generation_service._complete_recurrence_evidence_envelope_fallback(
        final_schedule_path=tmp_path / "python-fallback.pacbin",
        baseline=baseline,
        baseline_plan=_topology_replay_plan(),
        color_accuracy="lc",
        outcome=detached,
    )


def test_native_capacity_fallback_releases_warmup_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetainedResource:
        pass

    class WarmupOwner:
        def __init__(
            self,
            retained: list[weakref.ReferenceType[RetainedResource]],
            events: list[str],
        ) -> None:
            self.evidence_transport = RetainedResource()
            self.candidate_capture = RetainedResource()
            self.verification_capture = RetainedResource()
            self.open_spool: RetainedResource | None = RetainedResource()
            retained.extend(
                weakref.ref(value)
                for value in (
                    self.evidence_transport,
                    self.candidate_capture,
                    self.verification_capture,
                    self.open_spool,
                )
            )
            self._events = events

        def close(self) -> None:
            self._events.append("warmup-closed")
            self.open_spool = None

    message = (
        "raw numerical evidence exceeds the streaming 1 GiB resident memory envelope"
    )
    retained_frames: list[weakref.ReferenceType[RetainedResource]] = []

    def fail_native() -> None:
        frame_resource = RetainedResource()
        retained_frames.append(weakref.ref(frame_resource))
        raise GenerationError(message)

    plan = _topology_replay_plan()
    try:
        fail_native()
    except GenerationError as exc:
        outcome = recurrence_warmup.recurrence_numerical_evidence_capacity_outcome(
            plan.sections,
            candidate_probe_count=2,
            verification_probe_count=2,
            runtime_parameter_count=len(plan.runtime_parameter_schema),
            error=exc,
        )
        assert isinstance(outcome, RecurrenceNumericalEvidenceEnvelopeExceeded)
        detached = generation_service._detached_recurrence_evidence_envelope_outcome(
            outcome,
            source_error=exc,
        )
    else:  # pragma: no cover - the injected fallback is mandatory
        raise AssertionError("injected native fallback did not fail")

    events: list[str] = []
    retained_warmup: list[weakref.ReferenceType[RetainedResource]] = []
    warmup = WarmupOwner(retained_warmup, events)
    warmup_owner = [warmup]
    del warmup
    baseline = _fallback_lowering_output(tmp_path, "baseline", inspection={})
    baseline.payload_path.write_bytes(b"baseline")
    publish = generation_service._publish_validated_recurrence_baseline

    def publish_without_evidence(
        *,
        baseline: generation_service._RustRecurrenceLoweringOutput,
        final_schedule_path: Path,
    ) -> generation_service._RustRecurrenceLoweringOutput:
        gc.collect()
        events.append("publish")
        assert warmup_owner == []
        assert retained_frames[0]() is None
        assert all(reference() is None for reference in retained_warmup)
        assert detached.__traceback__ is None
        assert detached.__cause__ is None
        assert detached.__context__ is None
        return publish(
            baseline=baseline,
            final_schedule_path=final_schedule_path,
        )

    monkeypatch.setattr(
        generation_service,
        "_publish_validated_recurrence_baseline",
        publish_without_evidence,
    )

    generation_service._complete_recurrence_evidence_envelope_fallback(
        final_schedule_path=tmp_path / "native-fallback.pacbin",
        baseline=baseline,
        baseline_plan=plan,
        color_accuracy="lc",
        outcome=detached,
        warmup_owner=warmup_owner,  # type: ignore[arg-type]
    )
    assert events == ["warmup-closed", "publish"]


def test_compressed_canonical_transport_round_trips_with_length_and_digest() -> None:
    value = {
        "candidate_capture": {"observations": []},
        "verification_capture": {"observations": []},
    }
    expected = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    with tempfile.TemporaryFile() as stream:
        canonical_bytes = _write_compressed_canonical_evidence(
            value,
            stream,
            canonical_byte_limit=1 << 20,
        )
        encoded = _read_compressed_evidence_spool(stream)

    magic, declared_bytes, digest = _COMPRESSED_EVIDENCE_HEADER.unpack_from(encoded)
    assert magic == b"PACNCEZ1"
    assert canonical_bytes == declared_bytes == len(expected)
    assert zlib.decompress(encoded[_COMPRESSED_EVIDENCE_HEADER.size :]) == expected
    assert digest.hex() == hashlib.sha256(expected).hexdigest()


def test_compressed_warmup_keeps_relation_rows_spooled_through_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recurrence_warmup,
        "_select_raw_evidence_storage_geometry",
        lambda *_args, **_kwargs: recurrence_warmup._RawEvidenceStorageGeometry(
            scalar_count=128,
            row_count=16,
            canonical_byte_limit=1 << 20,
            encoding="zlib-canonical-json-v1",
            producer_resident_upper_bound=1 << 20,
        ),
    )
    plan = _topology_replay_plan()
    result = run_recurrence_numerical_current_warmup(
        plan,
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="lc",
        precision_digits=80,
        seed=53,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )
    candidate_spool = result.candidate_capture.observations
    assert result.verification_capture is not None
    verification_spool = result.verification_capture.observations
    assert isinstance(candidate_spool, _SpooledObservationMapping)
    assert isinstance(verification_spool, _SpooledObservationMapping)
    assert result.evidence_transport_bytes == len(result.evidence_json)
    consumed = result.without_evidence_transport()
    assert consumed.evidence_json == b""
    assert consumed.evidence_transport_bytes == result.evidence_transport_bytes
    try:
        validated = validate_recurrence_numerical_current_application(
            consumed,
            plan,
            baseline_plan=plan,
            precision_digits=80,
            seed=53,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
        )
        persisted = validated.to_json_dict()["persisted_numerical_evidence"]
        assert persisted["generation_evidence_encoding"] == ("zlib-canonical-json-v1")
        assert validated.application_validation["status"] == "verified"
    finally:
        result.close()
        result.close()
    with pytest.raises(RuntimeError, match="spool is closed"):
        candidate_spool[0]
    with pytest.raises(RuntimeError, match="spool is closed"):
        verification_spool[0]


def test_explicit_raw_wire_limit_is_honored_without_compression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recurrence_warmup,
        "_select_raw_evidence_storage_geometry",
        lambda *_args, **_kwargs: recurrence_warmup._RawEvidenceStorageGeometry(
            scalar_count=128,
            row_count=16,
            canonical_byte_limit=128,
            encoding="canonical-json-v3",
            producer_resident_upper_bound=1 << 20,
        ),
    )
    monkeypatch.setattr(
        recurrence_warmup,
        "_write_compressed_canonical_evidence",
        lambda *_args, **_kwargs: pytest.fail(
            "late raw overflow attempted compressed evidence transport"
        ),
    )

    with pytest.raises(
        recurrence_warmup._CanonicalEvidenceCapacityExceeded
    ) as captured:
        run_recurrence_numerical_current_warmup(
            _topology_replay_plan(),
            candidate_points=_points(1),
            verification_points=_points(101),
            mode="certified-reuse",
            color_accuracy="lc",
            precision_digits=80,
            seed=53,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
        )

    assert "canonical boundary" in str(captured.value)


def test_initial_compressed_capacity_error_is_propagated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recurrence_warmup,
        "_select_raw_evidence_storage_geometry",
        lambda *_args, **_kwargs: recurrence_warmup._RawEvidenceStorageGeometry(
            scalar_count=128,
            row_count=16,
            canonical_byte_limit=1 << 20,
            encoding="zlib-canonical-json-v1",
            producer_resident_upper_bound=1 << 20,
        ),
    )
    monkeypatch.setattr(
        recurrence_warmup,
        "_write_compressed_canonical_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            recurrence_warmup._CanonicalEvidenceCapacityExceeded(
                "injected compressed transport capacity"
            )
        ),
    )

    with pytest.raises(
        recurrence_warmup._CanonicalEvidenceCapacityExceeded
    ) as captured:
        run_recurrence_numerical_current_warmup(
            _topology_replay_plan(),
            candidate_points=_points(1),
            verification_points=_points(101),
            mode="certified-reuse",
            color_accuracy="lc",
            precision_digits=80,
            seed=53,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
        )

    assert "compressed transport capacity" in str(captured.value)


def test_native_capacity_classifier_excludes_integrity_and_structure_errors() -> None:
    sections = _topology_replay_plan().sections
    outcome = recurrence_warmup.recurrence_numerical_evidence_capacity_outcome(
        sections,
        candidate_probe_count=4,
        verification_probe_count=4,
        runtime_parameter_count=2,
        error=ArtifactError(
            "raw numerical evidence exceeds the streaming 1 GiB resident memory "
            "envelope"
        ),
    )

    assert isinstance(outcome, RecurrenceNumericalEvidenceEnvelopeExceeded)
    assert outcome.geometry.current_count == len(sections.currents)
    assert (
        validate_recurrence_numerical_evidence_fallback(outcome.to_json_dict())
        == outcome.to_json_dict()
    )
    for message in (
        "compressed numerical relation evidence digest mismatch",
        "numerical relation evidence JSON lexical structure is incomplete",
        "recurrence runtime parameter schema/defaults do not match",
    ):
        assert (
            recurrence_warmup.recurrence_numerical_evidence_capacity_outcome(
                sections,
                candidate_probe_count=4,
                verification_probe_count=4,
                runtime_parameter_count=2,
                error=ArtifactError(message),
            )
            is None
        )


def test_spooled_discovery_preserves_global_equal_opposite_and_zero_relations() -> None:
    plan = _signed_relation_plan()
    source_digest = recurrence_numerical_source_semantics_sha256(plan.sections)
    candidate = capture_recurrence_current_observations(
        plan,
        _points(1),
        precision_digits=80,
        source_semantics_sha256=source_digest,
        seed=67,
        domain="candidate-current-probes-v1",
    )
    verification = capture_recurrence_current_observations(
        plan,
        _points(101),
        precision_digits=80,
        source_semantics_sha256=source_digest,
        seed=67,
        domain="independent-verification-current-probes-v1",
    )
    expected, _expected_report = _discover_relations(
        plan.sections,
        candidate,
        verification,
        source_semantics_sha256=source_digest,
        precision_digits=80,
        seed=67,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
        color_accuracy="lc",
    )
    candidate_spool = _SpooledObservationMapping(
        candidate.observations,
        candidate_indexes=_build_candidate_indexes(
            plan.sections,
            candidate.observations,
        ),
    )
    verification_spool = _SpooledObservationMapping(
        verification.observations,
    )
    try:
        actual, _actual_report = _discover_relations(
            plan.sections,
            replace(candidate, observations=candidate_spool),
            replace(verification, observations=verification_spool),
            source_semantics_sha256=source_digest,
            precision_digits=80,
            seed=67,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
            color_accuracy="lc",
        )
    finally:
        candidate_spool.close()
        verification_spool.close()

    assert actual == expected
    assert {
        (certificate.current_id, certificate.relation_kind) for certificate in actual
    } >= {(3, "equal"), (4, "opposite"), (5, "zero")}


def test_synthetic_multi_helicity_all_flow_skips_unsafe_selector_hypotheses() -> None:
    plan = _signed_relation_plan()
    source_digest = recurrence_numerical_source_semantics_sha256(plan.sections)
    candidate = capture_recurrence_current_observations(
        plan,
        _points(1),
        precision_digits=80,
        source_semantics_sha256=source_digest,
        seed=67,
        domain="candidate-current-probes-v1",
    )
    verification = capture_recurrence_current_observations(
        plan,
        _points(101),
        precision_digits=80,
        source_semantics_sha256=source_digest,
        seed=67,
        domain="independent-verification-current-probes-v1",
    )
    sections = replace(
        plan.sections,
        strategy="all-flow-union",
        replay_targets=(),
        resolved_helicities=(
            _ResolvedHelicity(0, 0, 0, 0, 2, 2, 2, 0),
            _ResolvedHelicity(2, 2, 2, 1, 2, 2, 2, 0),
        ),
    )

    certificates, report = _discover_relations(
        sections,
        candidate,
        verification,
        source_semantics_sha256=source_digest,
        precision_digits=80,
        seed=67,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
        color_accuracy="lc",
    )

    assert certificates == ()
    scope = report["application_scope"]
    assert scope["multi_helicity_all_flow_domain_ids"] == [0]
    assert scope["multi_helicity_all_flow_application"] == "disabled"
    assert scope["suppressed_current_count"] == 4
    assert report["tested_hypothesis_count"] == 0
    candidate_index = report["candidate_index"]
    assert candidate_index["theoretical_pair_hypothesis_count"] == 0
    assert candidate_index["screened_pair_hypothesis_count"] == 0
    assert candidate_index["zero_hypothesis_count"] == 0
    assert (
        recurrence_warmup._effective_numerical_relation_mode(
            sections,
            requested_mode="certified-reuse",
            certificates=certificates,
        )
        == "diagnostic"
    )


def test_contracted_binary64_policy_keeps_census_but_contains_opposite_relations() -> (
    None
):
    plan = _signed_relation_plan()
    source_digest = recurrence_numerical_source_semantics_sha256(plan.sections)
    candidate = capture_recurrence_current_observations(
        plan,
        _points(1),
        precision_digits=80,
        source_semantics_sha256=source_digest,
        seed=67,
        domain="candidate-current-probes-v1",
    )
    verification = capture_recurrence_current_observations(
        plan,
        _points(101),
        precision_digits=80,
        source_semantics_sha256=source_digest,
        seed=67,
        domain="independent-verification-current-probes-v1",
    )
    sections = replace(
        plan.sections,
        strategy="contracted-color-union",
        replay_targets=(),
    )

    certificates, report = _discover_relations(
        sections,
        candidate,
        verification,
        source_semantics_sha256=source_digest,
        precision_digits=80,
        seed=67,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
        color_accuracy="full",
    )

    relation_kinds = {certificate.relation_kind for certificate in certificates}
    assert relation_kinds == {"equal", "opposite", "zero"}
    assert report["application_scope"]["contracted_opposite_application"] == "disabled"
    assert (
        recurrence_warmup._effective_numerical_relation_mode(
            sections,
            requested_mode="certified-reuse",
            certificates=certificates,
        )
        == "certified-reuse"
    )
    eligible = recurrence_warmup._applicable_numerical_certificates(
        certificates, application_scope=report["application_scope"]
    )
    assert {certificate.relation_kind for certificate in eligible} == {"equal", "zero"}
    assert len(eligible) == 2
    assert all(
        any(item is certificate for item in certificates) for certificate in eligible
    )
    assert (
        recurrence_warmup._effective_numerical_relation_mode(
            sections,
            requested_mode="certified-reuse",
            certificates=tuple(
                certificate
                for certificate in certificates
                if certificate.relation_kind == "opposite"
            ),
        )
        == "diagnostic"
    )


def test_contracted_partial_reuse_preserves_full_native_evidence_and_census() -> None:
    result = run_recurrence_numerical_current_warmup(
        _contracted_signed_relation_plan(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="full",
        precision_digits=80,
        seed=67,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )
    try:
        assert len(result.certificates) == 3
        assert result.applied_relation_count == 2
        assert {item.relation_kind for item in result.applied_certificates} == {
            "equal",
            "zero",
        }
        assert result.effective_mode_reason is None
        evidence = json.loads(result.evidence_json)
        assert len(evidence["certificates"]) == len(evidence["mappings"]) == 3
        assert {item["relation_kind"] for item in evidence["mappings"]} == {
            "equal",
            "opposite",
            "zero",
        }
        report = result.to_json_dict()
        assert report["certified_relation_count"] == 3
        assert report["applied_relation_count"] == 2
        assert report["application"]["certified_relation_count"] == 3
        assert report["application"]["applied_relation_count"] == 2
        assert len(report["application"]["certificates"]) == 3
        assert (
            report["persisted_numerical_evidence"]["full_census"][
                "certificate_set_sha256"
            ]
            == evidence["certificate_set_sha256"]
        )
        assert 4 not in result.candidate_capture.observations
        assert 5 in result.candidate_capture.observations
        assert result.warning_required is True
    finally:
        result.close()


def test_applicable_relation_preserves_a_skipped_opposite_representative() -> None:
    plan = _contracted_signed_relation_plan()
    result = run_recurrence_numerical_current_warmup(
        plan,
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="full",
        precision_digits=80,
        seed=67,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )
    try:
        opposite = next(
            item for item in result.certificates if item.relation_kind == "opposite"
        )
        # Application filtering does not create certificates or rewrite their
        # representative. The native mapper keeps skipped currents computed.
        dependent = replace(
            result.applied_certificates[0],
            current_id=opposite.current_id + 2,
            representative_id=opposite.current_id,
            execution_representative_id=opposite.current_id,
            relation_kind="equal",
        )
        filtered = recurrence_warmup._applicable_numerical_certificates(
            (opposite, dependent),
            application_scope=result.discovery_report["application_scope"],
        )
        assert filtered == (dependent,)
        assert filtered[0] is dependent
        assert filtered[0].execution_representative_id == opposite.current_id
    finally:
        result.close()


def test_discovery_reads_each_candidate_current_once_and_reuses_residual_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _no_relation_plan()
    source_digest = recurrence_numerical_source_semantics_sha256(plan.sections)
    candidate = capture_recurrence_current_observations(
        plan,
        _points(1),
        precision_digits=80,
        source_semantics_sha256=source_digest,
        seed=71,
        domain="candidate-current-probes-v1",
    )
    verification = capture_recurrence_current_observations(
        plan,
        _points(101),
        precision_digits=80,
        source_semantics_sha256=source_digest,
        seed=71,
        domain="independent-verification-current-probes-v1",
    )
    candidate_indexes = _build_candidate_indexes(
        plan.sections,
        candidate.observations,
    )
    expected_certificates, expected_report = _discover_relations(
        plan.sections,
        candidate,
        verification,
        source_semantics_sha256=source_digest,
        precision_digits=80,
        seed=71,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
        color_accuracy="lc",
        candidate_indexes=candidate_indexes,
    )
    spool = _SpooledObservationMapping(candidate.observations)

    class CountingRows(
        Mapping[int, tuple[tuple[Decimal, Decimal], ...]],
    ):
        def __init__(self) -> None:
            self.reads: dict[int, int] = {}

        def __getitem__(
            self,
            current_id: int,
        ) -> tuple[tuple[Decimal, Decimal], ...]:
            self.reads[current_id] = self.reads.get(current_id, 0) + 1
            return spool[current_id]

        def __iter__(self) -> Iterator[int]:
            return iter(spool)

        def __len__(self) -> int:
            return len(spool)

    rows = CountingRows()
    fraction_string_calls = 0
    fraction_string = recurrence_warmup._fraction_string

    def counted_fraction_string(value: Fraction) -> str:
        nonlocal fraction_string_calls
        fraction_string_calls += 1
        return fraction_string(value)

    monkeypatch.setattr(
        recurrence_warmup,
        "_fraction_string",
        counted_fraction_string,
    )
    try:
        actual_certificates, actual_report = _discover_relations(
            plan.sections,
            replace(candidate, observations=rows),
            verification,
            source_semantics_sha256=source_digest,
            precision_digits=80,
            seed=71,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
            color_accuracy="lc",
            candidate_indexes=candidate_indexes,
        )
    finally:
        spool.close()

    assert actual_certificates == expected_certificates == ()
    assert actual_report == expected_report
    assert rows.reads == {2: 1, 3: 1}
    assert fraction_string_calls == 3 * (
        int(actual_report["tested_hypothesis_count"]) + 1
    )


def test_dynamic_raw_geometry_rejects_adversarial_sizes_and_reserves_wire() -> None:
    with pytest.raises(ValueError, match="memory envelope"):
        _raw_evidence_wire_byte_limit(
            scalar_count=1 << 63,
            row_count=1 << 63,
        )

    maximum = _raw_evidence_wire_byte_limit(
        scalar_count=0,
        row_count=0,
    )
    assert maximum == (_RAW_EVIDENCE_MEMORY_TARGET_BYTES - (32 << 20)) // 2
    assert maximum >= _MIN_RAW_EVIDENCE_WIRE_BYTES


@pytest.mark.parametrize("text", ("1e+20000", "1e-20000"))
def test_raw_evidence_accepts_decimal_widths_above_the_old_ceiling(
    text: str,
) -> None:
    encoded = _decimal_string(Decimal(text))
    assert len(encoded) > 16_384
    assert Decimal(encoded) == Decimal(text)


def test_shared_candidate_index_is_complete_at_exact_signed_boundaries() -> None:
    large = Fraction(10**100)
    observations = {
        0: ((Fraction(100, 9), Fraction(0)), (large, Fraction(0))),
        1: ((Fraction(-100, 9), Fraction(0)), (-large, Fraction(0))),
        2: ((Fraction(1_000), Fraction(0)), (Fraction(0), Fraction(0))),
        4: ((Fraction(10), Fraction(0)), (large, Fraction(0))),
    }
    index = build_numerical_observation_candidate_index(
        (0, 1, 2, 4),
        observations,
        normalize=Fraction,
    )
    assert (index.observation_index, index.scalar_component) == (0, 0)
    common = {
        "current_id": 4,
        "relative_tolerance": Fraction(1, 10),
        "absolute_tolerance": Fraction(0),
        "normalize": Fraction,
    }
    assert numerical_observation_tolerance_window_ids(
        index,
        observations[4],
        relation_kind="equal",
        **common,
    ) == (0,)
    assert numerical_observation_tolerance_window_ids(
        index,
        observations[4],
        relation_kind="opposite",
        **common,
    ) == (1,)
    # The unrelated 10**100 component has zero residual for the accepted
    # signed pairs and cannot enlarge the selected pair's tolerance window.
    assert abs(Fraction(10) - Fraction(100, 9)) == Fraction(10, 9)
    assert Fraction(1, 10) * Fraction(100, 9) == Fraction(10, 9)


def test_rejected_diagnostic_full_digest_binds_unretained_tail() -> None:
    retained = tuple({"current_id": index} for index in range(16))
    root = "00" * 32
    common_tail = _advance_rejection_digest(root, {"current_id": 16})
    left_digest = _advance_rejection_digest(common_tail, {"current_id": 17})
    right_digest = _advance_rejection_digest(common_tail, {"current_id": 18})
    left = _rejected_candidate_diagnostics(
        retained,
        total_rejected=18,
        full_census={"tested": 18},
        full_rejection_sha256=left_digest,
    )
    right = _rejected_candidate_diagnostics(
        retained,
        total_rejected=18,
        full_census={"tested": 18},
        full_rejection_sha256=right_digest,
    )
    assert left["retained_sha256"] == right["retained_sha256"]
    assert left["full_census_sha256"] == right["full_census_sha256"]
    assert left["full_rejection_sha256"] != right["full_rejection_sha256"]
    assert left["truncated"] is True


@pytest.mark.parametrize(
    ("relative_tolerance", "absolute_tolerance"),
    (
        (-0.0, 1.0e-70),
        (1.0e-60, -0.0),
        (float("nan"), 1.0e-70),
        (1.0e-60, float("inf")),
    ),
)
def test_recurrence_raw_evidence_rejects_noncanonical_tolerances_before_capture(
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    with pytest.raises(ValueError, match="tolerances are invalid"):
        run_recurrence_numerical_current_warmup(
            _topology_replay_plan(),
            candidate_points=_points(1),
            verification_points=_points(101),
            mode="certified-reuse",
            color_accuracy="lc",
            precision_digits=80,
            seed=53,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )


@pytest.mark.parametrize("partial_contracted", (False, True))
def test_recurrence_aggregate_reports_applied_warning_and_explicit_opt_out(
    partial_contracted: bool,
) -> None:
    result = run_recurrence_numerical_current_warmup(
        (
            _contracted_signed_relation_plan()
            if partial_contracted
            else _topology_replay_plan()
        ),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="full" if partial_contracted else "lc",
        precision_digits=80,
        seed=71,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )
    lane = result.to_json_dict()
    discovery = lane["discovery"]
    application = lane["application"]
    assert isinstance(discovery, dict)
    assert isinstance(application, dict)
    certified = len(result.certificates)
    applied = result.applied_relation_count
    assert (certified, applied) == ((3, 2) if partial_contracted else (1, 1))
    native = {
        "requested_mode": "certified-reuse",
        "state": "native-applied",
        "exact_certified_relation_count": certified,
        "applied_relation_count": applied,
        "certificate_count": certified,
        "numerical_candidate_count": discovery["numerical_candidate_count"],
        "uncertified_candidate_count": discovery["verification_rejected_count"],
        "rejected_hypothesis_count": discovery["rejected_hypothesis_count"],
        "certificate_replay": application["certificate_replay"],
        "probe": {
            "probe_count": lane["candidate_capture"]["point_count"],
            "verification_probe_count": (lane["verification_capture"]["point_count"]),
            "verification_rejected_count": discovery["verification_rejected_count"],
            "tested_hypothesis_count": discovery["tested_hypothesis_count"],
            "runtime_parameter_schema_sha256": lane["candidate_capture"][
                "runtime_parameter_schema_sha256"
            ],
            "candidate_observation_batch_sha256": lane["candidate_capture"][
                "observation_batch_sha256"
            ],
            "verification_observation_batch_sha256": lane["verification_capture"][
                "observation_batch_sha256"
            ],
            "decision_sha256": discovery["decision_sha256"],
            "rejection_decision_sha256": discovery["rejection_decision_sha256"],
        },
        "rejected_candidate_diagnostics": {
            "total_rejected_hypothesis_count": discovery["rejected_hypothesis_count"],
            "full_rejection_sha256": discovery["rejection_decision_sha256"],
        },
    }
    runtime_inspection, aggregate = _recurrence_relation_reporting(
        {"relation_discovery": native},
        mode="certified-reuse",
        lane_report=lane,
    )
    assert "relation_discovery" not in runtime_inspection
    assert aggregate["execution_mode"] == "recurrence"
    assert aggregate["lanes"]["primary"]["applied_relation_count"] == applied
    assert aggregate["certified_relation_count"] == certified
    assert aggregate["applied_relation_count"] == applied
    assert aggregate["warning"]["required"] is True
    assert aggregate["warning"]["emit"] == "once-per-generated-artifact"
    assert aggregate["native_relation_application"] == native

    for mutation in (
        {"requested_mode": "diagnostic"},
        {"exact_certified_relation_count": 0},
        {"applied_relation_count": 0},
        {"certificate_count": 0},
        {
            "certificate_replay": {
                **native["certificate_replay"],
                "certificate_set_sha256": "0" * 64,
            }
        },
        {
            "probe": {
                **native["probe"],
                "verification_probe_count": 3,
            }
        },
    ):
        drifted = {**native, **mutation}
        with pytest.raises(GenerationError):
            _recurrence_relation_reporting(
                {"relation_discovery": drifted},
                mode="certified-reuse",
                lane_report=lane,
            )

    if partial_contracted:
        for incorrect_count in (1, 3):
            with pytest.raises(GenerationError):
                _recurrence_relation_reporting(
                    {
                        "relation_discovery": {
                            **native,
                            "applied_relation_count": incorrect_count,
                        }
                    },
                    mode="certified-reuse",
                    lane_report={**lane, "applied_relation_count": incorrect_count},
                )

    off_lane = recurrence_numerical_current_opt_out_report(
        _topology_replay_plan().sections,
        color_accuracy="lc",
    )
    off_runtime_inspection, off_aggregate = _recurrence_relation_reporting(
        {},
        mode="off",
        lane_report=off_lane,
    )
    assert "relation_discovery" not in off_runtime_inspection
    assert off_aggregate["requested_mode"] == "off"
    assert off_aggregate["lanes"]["primary"]["state"] == "disabled-by-user"
    assert off_aggregate["warning"]["required"] is False


def test_application_rejects_current_dimension_drift() -> None:
    result = run_recurrence_numerical_current_warmup(
        _topology_replay_plan(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="lc",
        precision_digits=80,
        seed=79,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )
    drifted = _topology_replay_plan()
    current = drifted.sections.currents[3]
    drifted.sections = replace(
        drifted.sections,
        currents=(
            *drifted.sections.currents[:3],
            replace(current, component_count=2),
        ),
    )

    try:
        validate_recurrence_numerical_current_application(
            result,
            drifted,
            baseline_plan=_topology_replay_plan(),
            precision_digits=80,
            seed=79,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
        )
    except (ArtifactError, ValueError):
        pass
    else:  # pragma: no cover - fail-closed contract
        raise AssertionError("current-dimension drift was accepted")


def test_recurrence_service_forwards_complete_numerical_contract_to_pyo3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def binding(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        Path(str(args[5])).write_bytes(b"pacbin")
        dispatch_destination = kwargs.get("helicity_dispatch_destination")
        if dispatch_destination is not None:
            Path(str(dispatch_destination)).write_bytes(b"dispatch")
        return object()

    native = SimpleNamespace(_lower_recurrence_direct_v2=binding)
    monkeypatch.setattr(
        generation_service.importlib,
        "import_module",
        lambda _name: native,
    )
    monkeypatch.setattr(
        generation_service,
        "verify_native_module",
        lambda _module: None,
    )
    monkeypatch.setattr(
        generation_service,
        "_validate_rust_recurrence_lowering_result",
        lambda *_args, **kwargs: {
            "inspection_summary": {},
            "resolved_helicities": (),
            "amplitude_destinations": (),
            "structural_zero_physical_sector_certificate": None,
            "exact_sections": {},
            "generation_profile": {"serialized_bytes": {"container": len(b"pacbin")}},
            "member_count": 1,
            "unpacked_size_bytes": 6,
            "index_sha256": "b" * 64,
            "helicity_dispatch": (
                {
                    "abi": "pyamplicol-recurrence-helicity-dispatch-v1",
                    "size_bytes": len(b"dispatch"),
                    "sha256": hashlib.sha256(b"dispatch").hexdigest(),
                    "base_runtime_layout_digest": "2" * 64,
                    "resolved_helicity_count": 4,
                }
                if kwargs["expect_helicity_dispatch"]
                else None
            ),
        },
    )
    builder = SimpleNamespace(digest="c" * 64, layout="topology-replay")
    template = SimpleNamespace(digest="d" * 64)
    catalog = SimpleNamespace(
        catalog_digest="e" * 64,
        to_dict=lambda: {"z": 1, "a": 2},
    )

    def progress(_payload: object) -> None:
        pass

    evidence = b'{"raw":"evidence"}'
    destination = tmp_path / "recurrence-runtime.pacbin"

    output = generation_service._invoke_rust_recurrence_lowering_v2(
        builder,
        template,
        catalog,
        "f" * 64,
        "1" * 64,
        destination,
        point_tile_size=17,
        workspace_mib=23,
        relation_discovery_mode="certified-reuse",
        relation_discovery_precision_digits=101,
        relation_discovery_probe_count=3,
        relation_discovery_verification_probe_count=5,
        relation_discovery_relative_tolerance=1.25e-70,
        relation_discovery_absolute_tolerance=2.5e-80,
        relation_discovery_seed=123456789,
        color_accuracy="nlc",
        relation_discovery_evidence_json=evidence,
        progress_callback=progress,
    )
    cached_destination = tmp_path / "recurrence-runtime-cached.pacbin"
    cached_catalog = SimpleNamespace(
        catalog_digest="e" * 64,
        to_dict=lambda: pytest.fail("cached direct-template JSON was rebuilt"),
    )
    cached_output = generation_service._invoke_rust_recurrence_lowering_v2(
        builder,
        template,
        cached_catalog,
        "f" * 64,
        "1" * 64,
        cached_destination,
        point_tile_size=17,
        workspace_mib=23,
        relation_discovery_mode="certified-reuse",
        relation_discovery_precision_digits=101,
        relation_discovery_probe_count=3,
        relation_discovery_verification_probe_count=5,
        relation_discovery_relative_tolerance=1.25e-70,
        relation_discovery_absolute_tolerance=2.5e-80,
        relation_discovery_seed=123456789,
        color_accuracy="nlc",
        relation_discovery_evidence_json=evidence,
        progress_callback=progress,
        direct_template_catalog_json=b'{"a":2,"z":1}',
    )
    dispatch_destination = tmp_path / "recurrence-helicity-dispatch-v1.bin"
    dispatched_output = generation_service._invoke_rust_recurrence_lowering_v2(
        builder,
        template,
        cached_catalog,
        "f" * 64,
        "1" * 64,
        tmp_path / "recurrence-runtime-dispatched.pacbin",
        point_tile_size=17,
        workspace_mib=23,
        relation_discovery_mode="off",
        relation_discovery_precision_digits=101,
        relation_discovery_probe_count=3,
        relation_discovery_verification_probe_count=5,
        relation_discovery_relative_tolerance=1.25e-70,
        relation_discovery_absolute_tolerance=2.5e-80,
        relation_discovery_seed=123456789,
        color_accuracy="full",
        helicity_dispatch_destination=dispatch_destination,
        direct_template_catalog_json=b'{"a":2,"z":1}',
    )

    assert output.payload_path == destination
    assert output.generation_profile == {
        "serialized_bytes": {"container": len(b"pacbin")}
    }
    assert cached_output.payload_path == cached_destination
    assert dispatched_output.helicity_dispatch is not None
    assert dispatched_output.helicity_dispatch.payload_path == dispatch_destination
    assert (
        dispatched_output.helicity_dispatch.payload_sha256
        == hashlib.sha256(b"dispatch").hexdigest()
    )
    assert len(calls) == 3
    args, kwargs = calls[0]
    assert args[:2] == (builder, template)
    assert args[2] == b'{"a":2,"z":1}'
    assert args[3:5] == ("f" * 64, "1" * 64)
    assert args[5] == str(destination)
    assert kwargs == {
        "point_tile_size": 17,
        "workspace_mib": 23,
        "relation_discovery_mode": "certified-reuse",
        "relation_discovery_precision_digits": 101,
        "relation_discovery_probe_count": 3,
        "relation_discovery_verification_probe_count": 5,
        "relation_discovery_relative_tolerance": 1.25e-70,
        "relation_discovery_absolute_tolerance": 2.5e-80,
        "relation_discovery_seed": 123456789,
        "color_accuracy": "nlc",
        "relation_discovery_evidence_json": evidence,
        "helicity_dispatch_destination": None,
        "progress_callback": progress,
    }
    cached_args, cached_kwargs = calls[1]
    assert cached_args[2] == args[2]
    assert cached_kwargs == kwargs
    assert calls[2][1]["helicity_dispatch_destination"] == str(dispatch_destination)


def test_recurrence_service_preserves_native_capacity_message_for_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    message = (
        "raw numerical evidence exceeds the streaming 1 GiB resident memory envelope"
    )

    def binding(*_args: object, **_kwargs: object) -> object:
        # Native evidence validation exposes invalid_argument as ValueError;
        # the service wrapper deliberately retains the exact marker in its
        # typed GenerationError for the bounded fallback classifier.
        raise ValueError(message)

    native = SimpleNamespace(_lower_recurrence_direct_v2=binding)
    monkeypatch.setattr(
        generation_service.importlib,
        "import_module",
        lambda _name: native,
    )
    monkeypatch.setattr(
        generation_service,
        "verify_native_module",
        lambda _module: None,
    )
    builder = SimpleNamespace(digest="c" * 64, layout="topology-replay")
    template = SimpleNamespace(digest="d" * 64)
    catalog = SimpleNamespace(
        catalog_digest="e" * 64,
        to_dict=lambda: {"z": 1, "a": 2},
    )

    with pytest.raises(GenerationError, match="streaming 1 GiB") as captured:
        generation_service._invoke_rust_recurrence_lowering_v2(
            builder,
            template,
            catalog,
            "f" * 64,
            "1" * 64,
            tmp_path / "recurrence-runtime.pacbin",
            point_tile_size=17,
            workspace_mib=23,
            relation_discovery_mode="certified-reuse",
            relation_discovery_precision_digits=101,
            relation_discovery_probe_count=3,
            relation_discovery_verification_probe_count=5,
            relation_discovery_relative_tolerance=1.25e-70,
            relation_discovery_absolute_tolerance=2.5e-80,
            relation_discovery_seed=123456789,
            color_accuracy="nlc",
            relation_discovery_evidence_json=b"{}",
            progress_callback=None,
        )
    outcome = recurrence_warmup.recurrence_numerical_evidence_capacity_outcome(
        _topology_replay_plan().sections,
        candidate_probe_count=3,
        verification_probe_count=5,
        runtime_parameter_count=2,
        error=captured.value,
    )
    assert isinstance(outcome, RecurrenceNumericalEvidenceEnvelopeExceeded)
