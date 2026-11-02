# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import gc
import hashlib
import importlib
import io
import json
import logging
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

from pyamplicol._internal.versions import (
    SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
    active_source_revision,
    verify_native_module,
)
from pyamplicol.api.errors import (
    ArtifactError,
    GenerationError,
    ModelError,
    PyAmpliColError,
)
from pyamplicol.api.models import CompiledModel, _compiled_model_payload
from pyamplicol.api.requests import (
    ModelSource,
    ProcessAlias,
    ProcessRequest,
    ProcessSet,
)
from pyamplicol.api.results import GenerationPlan, GenerationResult, ProcessPhysics
from pyamplicol.artifacts.process_groups import initial_process_groups
from pyamplicol.config import (
    ClampRequest,
    ConfigResolution,
    GenerationConfig,
    RunConfig,
    config_to_dict,
    resolve_config,
)
from pyamplicol.reporting import (
    ProgressEnd,
    ProgressSink,
    ProgressStart,
)

from ..color.fft_basis_strategy import best_general_fft_basis
from ..color.plan import (
    ColorTopologyReplayCertificate,
    GenericColorPlan,
    LCColorTopologyReplayPlan,
    build_color_plan,
    build_color_topology_replay_certificate,
    build_lc_topology_replay_plan,
)
from ..models.adjoint_color_certificate import adjoint_tree_color_basis_is_proven
from ..models.base import Model
from ..models.loading import CompiledModel as _CompiledModelPayload
from ..models.prepared import PreparedKernelPack, PreparedModelBundle
from ..models.prepared_target import PreparedTargetError, validate_prepared_target
from ..models.recurrence_direct_template import (
    RECURRENCE_DIRECT_TEMPLATE_ABI,
    RecurrenceDirectTemplateCatalogV1,
)
from ..models.recurrence_template import RecurrenceTemplateCatalog
from ..processes.ir import CanonicalProcessIR, ProcessLegIR
from ..runtime.recurrence_exact._plan import _RecurrenceExactPlan
from ..runtime.recurrence_exact._plan_v2 import DIRECT_NONE_U32, _parse_exact_sections
from .artifact_writer import (
    EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
    EAGER_PLAN_V3_ABI,
    EAGER_PLAN_V3_RUNTIME_CAPABILITY,
    EAGER_RUNTIME_CONTAINER_KIND,
    EAGER_RUNTIME_CONTAINER_SCHEMA_VERSION,
    EAGER_RUNTIME_LAYOUT_ABI,
    EAGER_RUNTIME_STORAGE_ABI,
    RECURRENCE_COLOR_RUNTIME_CAPABILITY,
    RECURRENCE_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
    RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
    RECURRENCE_PLAN_ABI,
    RECURRENCE_RUNTIME_CONTAINER_KIND,
    RECURRENCE_RUNTIME_CONTAINER_SCHEMA_VERSION,
    RECURRENCE_RUNTIME_LAYOUT_ABI,
    RECURRENCE_RUNTIME_STORAGE_ABI,
    ArtifactWriteResult,
    CompiledColorSelectorExecutionArtifact,
    CompiledExecutionArtifact,
    CompiledHelicitySelectorExecutionArtifact,
    CompiledProcessArtifact,
    EagerPlanV3ProcessArtifact,
    OnTheFlyProcessArtifact,
    RecurrenceHelicitySelectorPlanArtifact,
    RecurrenceProcessArtifact,
    _GenerationConfigProvenance,
    write_schema_v3_artifact,
)
from .contracts import RuntimeExpressionSchema, StageCompilationInput
from .dag_algorithms import (
    _infer_minimal_coupling_order_limits_from_color_plan,
    contributing_color_sector_ids,
    filter_dag_to_color_sectors,
    has_model_supported_amplitudes,
    infer_minimal_coupling_order_limits,
    prune_dag_to_amplitude_roots,
    prune_global_helicity_flip_equivalent_roots,
)
from .dag_compiler import _restrict_color_plan, compile_generic_dag
from .dag_equivalence import (
    NUMERICAL_CURRENT_RELATION_WARNING,
    NUMERICAL_CURRENT_RELATION_WARNING_CODE,
    dynamic_color_projection_not_applicable_report,
    project_rectangular_dynamic_color_classes,
)
from .dag_types import GenericDAG
from .eager_columnar import (
    EAGER_LOWERING_INPUT_ABI,
    EagerLoweringInputV1,
    build_eager_lowering_input_v1,
    runtime_model_parameter_records_from_eager_input,
)
from .eager_lowering import (
    PreparedCatalogEagerKernelIndex,
    PreparedCatalogEagerKernelResolver,
)
from .eager_tables import MISSING_U32
from .evaluator_container import (
    PacbinMemberKind,
    PacbinMemberSource,
    write_pacbin_atomic,
)
from .helicity_materialization import materialize_helicity_recurrence
from .helicity_replay import (
    HELICITY_RECURRENCE_CONTRACT_VERSION,
    build_helicity_recurrence_plan,
)
from .numerical_current_warmup import (
    generic_dag_numerical_current_opt_out_report,
    run_generic_dag_numerical_current_warmup,
)
from .on_the_fly_seed import (
    OnTheFlyGenerationProjectionV1,
    OnTheFlyProcessSeedProjectionV1,
    project_on_the_fly_process_seed_v1,
)
from .physics_metadata import build_resolved_physics_from_dag
from .process_grouping import (
    GroupingBasis,
    GroupingParticle,
    GroupingProcess,
    GroupMember,
    LegMapping,
    flavor_blind_classes,
    flavor_blind_mapping,
    particle_table,
)
from .progress import GenerationPhaseReporter, PhaseHandle
from .recurrence_color import (
    RECURRENCE_COLOR_CONTRACTION_CODEC_ABI,
    encode_recurrence_color_contraction,
    recurrence_color_contraction_digest,
)
from .recurrence_columnar import (
    RECURRENCE_BUILDER_INPUT_ABI,
    RecurrenceBuilderInputV1,
    RecurrenceBuilderLogicalInputV1,
    build_recurrence_builder_input_v1,
)
from .recurrence_numerical_current_warmup import (
    EVIDENCE_ENVELOPE_FALLBACK_REASON,
    EVIDENCE_ENVELOPE_FALLBACK_WARNING,
    RecurrenceNumericalCurrentWarmupResult,
    RecurrenceNumericalEvidenceEnvelopeExceeded,
    build_recurrence_numerical_current_probe_points,
    numerical_relation_correctness_payload,
    recurrence_numerical_current_envelope_fallback_report,
    recurrence_numerical_current_opt_out_report,
    recurrence_numerical_evidence_capacity_outcome,
    recurrence_numerical_relation_correctness_summary,
    run_recurrence_numerical_current_warmup,
    validate_recurrence_numerical_current_application,
    validate_recurrence_numerical_evidence_fallback,
)
from .recurrence_physics import (
    build_on_the_fly_color_contraction,
    build_on_the_fly_public_metadata,
    build_on_the_fly_runtime_metadata,
    build_recurrence_color_contraction,
    build_recurrence_normalization,
    build_recurrence_physics,
    build_recurrence_runtime_metadata,
    recurrence_color_contraction_destinations,
    recurrence_color_sector_owner_map,
    recurrence_exact_color_coefficients,
    recurrence_referenced_kernel_ids,
)
from .recurrence_projection import (
    RecurrenceGenerationSliceV1,
    project_recurrence_process_v1,
)
from .recurrence_projection import (
    _digest as _recurrence_projection_digest,
)
from .recurrence_schedule_sharing import (
    RecurrenceProcessRemap,
    RecurrenceScheduleLoweringCache,
    _recurrence_schedule_semantic_digests,
    build_recurrence_process_executor_pack,
    recurrence_helicity_selector_schedule_digest,
)
from .recurrence_template_columnar import (
    RECURRENCE_TEMPLATE_INPUT_ABI,
    RecurrenceTemplateInputV1,
    build_recurrence_template_input_v1,
)
from .runtime_schema import (
    RuntimeSchemaLayout,
    build_runtime_expression_schema,
    build_runtime_schema_layout,
)
from .stage_compiler import (
    build_and_write_generic_stage_evaluator_artifacts,
    write_model_parameter_evaluator_artifact,
)
from .validation import (
    ValidationPointRecord,
    build_process_validation_point,
    build_validation_point,
)

if TYPE_CHECKING:
    from ..licensing import SymbolicaLicenseState
    from .artifact_writer import ApiBundleHook, ProcessArtifact
    from .process_grouping import ProcessGroupMap


_ProcessInput = TypeVar("_ProcessInput")
_ProcessOutput = TypeVar("_ProcessOutput")
_MISSING_PROCESS_RESULT = object()
_LOGGER = logging.getLogger("pyamplicol.generation")
_MAX_FUSED_HELICITY_SELECTOR_SECTORS = 128
_MAX_POST_BUILD_RESOLVED_COMPONENTS = 1_000_000
_EAGER_LOWERING_RESULT_KIND = "pyamplicol-eager-runtime-lowering-result"
_EAGER_LOWERING_RESULT_SCHEMA_VERSION = 1
_RECURRENCE_LOWERING_RESULT_KIND = "pyamplicol-recurrence-direct-lowering-result"
_RECURRENCE_LOWERING_RESULT_SCHEMA_VERSION = 4
_RECURRENCE_HELICITY_DISPATCH_ABI = "pyamplicol-recurrence-helicity-dispatch-v1"
_RECURRENCE_STRUCTURAL_ZERO_SECTOR_PROOF_KIND = (
    "complete-recurrence-closure-target-domain-v1"
)
_RECURRENCE_GENERATION_PROFILE_TIMINGS = frozenset(
    {
        "transition-catalog",
        "structural-feasibility",
        "color-target-index",
        "structural-demand",
        "support-indexing",
        "candidate-processing",
        "closure-processing",
        "canonical-emission",
        "python-extraction",
        "catalog-authentication",
        "semantic-construction-total",
        "direct-lowering",
        "serialization",
        "native-total",
    }
)
_RECURRENCE_GENERATION_PROFILE_COUNTERS = frozenset(
    {
        "support_bucket_count",
        "support_bucket_probe_count",
        "support_bucket_cache_hit_count",
        "support_bucket_cache_miss_count",
        "candidate_parent_pair_theoretical_count",
        "candidate_parent_pair_visited_count",
        "structural_feasible_support_count",
        "structural_decomposition_count",
        "structural_forward_transition_probe_count",
        "structural_demand_support_count",
        "structural_demand_state_count",
        "structural_reject_count",
        "transition_index_hit_count",
        "transition_index_miss_count",
        "transition_candidate_count",
        "quantum_match_count",
        "coupling_match_count",
        "transition_accept_count",
        "color_shape_match_count",
        "color_result_count",
        "color_target_accept_count",
        "color_target_reject_count",
        "color_acceptance_cache_hit_count",
        "color_acceptance_cache_miss_count",
        "color_fragment_bucket_count",
        "color_fragment_hash_lookup_count",
        "color_posting_incidence_count",
        "color_sparse_posting_bucket_count",
        "color_dense_posting_bucket_count",
        "color_sparse_posting_bytes",
        "color_dense_posting_bytes",
        "accepted_parent_key_clone_count",
        "current_key_lookup_count",
        "current_key_hit_count",
        "current_insert_count",
        "current_key_clone_count",
        "indexed_hash_lookup_count",
        "contribution_attempt_count",
        "contribution_insert_count",
        "contribution_merge_count",
        "closure_candidate_theoretical_count",
        "closure_candidate_count",
        "closure_support_lookup_count",
        "closure_state_match_count",
        "closure_color_attempt_count",
        "closure_group_count",
        "closure_proof_contribution_count",
        "constructed_current_count",
        "constructed_contribution_count",
        "constructed_interaction_count",
        "constructed_dynamic_color_state_count",
        "emitted_current_count",
        "emitted_contribution_count",
        "emitted_interaction_count",
        "emitted_finalization_count",
        "emitted_closure_count",
    }
)
_RECURRENCE_GENERATION_PROFILE_SERIALIZED_BYTES = frozenset(
    {"plan_payload", "container", "unpacked_container"}
)
# Symbolica releases the GIL while retaining process-wide mutable symbol state.
# Keep each complete lowering/compilation transaction atomic across generators.
_SYMBOLICA_MATERIALIZATION_LOCK = Lock()
_RECURRENCE_PROFILE_LOCK = Lock()


def _post_build_validation_slices(
    physics: ProcessPhysics,
    sample_count: int,
) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
    """Keep resolved-output validation below a bounded component count."""

    helicities = physics.helicities
    color_flows = physics.color_flows
    contracted = physics.contracted_color_components
    color_count = len(color_flows) or len(contracted)
    full_component_count = sample_count * len(helicities) * color_count
    if full_component_count <= _MAX_POST_BUILD_RESOLVED_COMPONENTS:
        return (("complete", (), ()),)

    helicity = next(
        (item for item in helicities if item.computed and not item.structural_zero),
        next(
            (item for item in helicities if not item.structural_zero),
            helicities[0],
        ),
    )
    helicity_ids = (str(helicity.id),)
    slices: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    if sample_count * color_count <= _MAX_POST_BUILD_RESOLVED_COMPONENTS:
        slices.append(("selected-helicity", helicity_ids, ()))

    if color_flows:
        color_flow = next(
            (item for item in color_flows if item.computed),
            color_flows[0],
        )
        color_flow_ids = (str(color_flow.id),)
        if sample_count * len(helicities) <= _MAX_POST_BUILD_RESOLVED_COMPONENTS:
            slices.append(("selected-flow", (), color_flow_ids))
        if not slices:
            slices.append(("selected-helicity-and-flow", helicity_ids, color_flow_ids))
    elif not slices:
        slices.append(("selected-helicity", helicity_ids, ()))
    return tuple(slices)


class _ValidationSelector(Protocol):
    id: str


class _ValidationHelicity(_ValidationSelector, Protocol):
    structural_zero: bool


class _ValidationPhysics(Protocol):
    color_flows: Sequence[_ValidationSelector]
    helicities: Sequence[_ValidationHelicity]


def _spread_selector_ids(values: Sequence[_ValidationSelector]) -> tuple[str, ...]:
    """Pick deterministic endpoints and a midpoint without copying an axis."""

    if not values:
        return ()
    positions = (0, len(values) // 2, len(values) - 1)
    selected: list[str] = []
    for position in positions:
        identifier = str(values[position].id)
        if identifier not in selected:
            selected.append(identifier)
    return tuple(selected)


def _recurrence_validation_selector_cases(
    physics: _ValidationPhysics,
) -> tuple[tuple[str, dict[str, tuple[str, ...]]], ...]:
    """Bound recurrence validation by axis size instead of their product."""

    color_flows = tuple(physics.color_flows)
    helicities = tuple(physics.helicities)
    live_helicities = tuple(
        helicity for helicity in helicities if not helicity.structural_zero
    )
    structural_zeros = tuple(
        helicity for helicity in helicities if helicity.structural_zero
    )
    color_ids = _spread_selector_ids(color_flows)
    helicity_ids = _spread_selector_ids(live_helicities)
    cases: list[tuple[str, dict[str, tuple[str, ...]]]] = []
    cases.extend(
        (
            f"flow {identifier}",
            {"color_flows": (identifier,)},
        )
        for identifier in color_ids
    )
    cases.extend(
        (
            f"helicity {identifier}",
            {"helicities": (identifier,)},
        )
        for identifier in helicity_ids
    )
    if color_ids and helicity_ids:
        cases.append(
            (
                "combined flow and helicity",
                {
                    "color_flows": (color_ids[0],),
                    "helicities": (helicity_ids[0],),
                },
            )
        )
    if color_ids and structural_zeros:
        zero_id = str(structural_zeros[0].id)
        cases.append(
            (
                "structural-zero helicity",
                {
                    "color_flows": (color_ids[0],),
                    "helicities": (zero_id,),
                },
            )
        )
    return tuple(cases)


class _RustEagerLoweringBinding(Protocol):
    def __call__(
        self,
        lowering_input: EagerLoweringInputV1,
        destination: str,
        /,
    ) -> object: ...


class _RustRecurrenceLoweringBinding(Protocol):
    def __call__(
        self,
        builder_input: RecurrenceBuilderInputV1,
        template_input: RecurrenceTemplateInputV1,
        direct_template_catalog_json: bytes,
        prepared_kernel_pack_digest: str,
        schedule_semantic_digest: str,
        destination: str,
        /,
        *,
        point_tile_size: int,
        workspace_mib: int,
        relation_discovery_mode: str,
        relation_discovery_precision_digits: int,
        relation_discovery_probe_count: int,
        relation_discovery_verification_probe_count: int,
        relation_discovery_relative_tolerance: float,
        relation_discovery_absolute_tolerance: float,
        relation_discovery_seed: int,
        color_accuracy: str,
        relation_discovery_evidence_json: bytes | None,
        helicity_dispatch_destination: str | None,
        progress_callback: Callable[[Mapping[str, object]], None] | None,
    ) -> object: ...


class _RustOnTheFlySeedsBinding(Protocol):
    def __call__(
        self,
        ordered_source_projection_jsons: tuple[bytes, ...],
        recurrence_template_catalog_json: bytes,
        direct_template_catalog_json: bytes,
        prepared_kernel_pack_digest: str,
        /,
    ) -> object: ...


class _RustOnTheFlySeedInspectorBinding(Protocol):
    def __call__(self, payload: bytes, /) -> object: ...


@dataclass(frozen=True, slots=True)
class _RustEagerLoweringOutput:
    inspection_summary: Mapping[str, object]
    payload_path: Path
    payload_size_bytes: int
    payload_sha256: str
    member_count: int
    unpacked_size_bytes: int
    index_sha256: str


@dataclass(frozen=True, slots=True)
class _RustRecurrenceHelicityDispatchOutput:
    payload_path: Path
    payload_size_bytes: int
    payload_sha256: str
    base_runtime_layout_digest: str
    resolved_helicity_count: int


@dataclass(frozen=True, slots=True)
class _RustRecurrenceStructuralZeroSectorCertificate:
    sector_ids: tuple[int, ...]
    semantic_digest: str
    sector_count: int
    resolved_helicity_count: int


@dataclass(frozen=True, slots=True)
class _RustRecurrenceLoweringOutput:
    inspection_summary: Mapping[str, object]
    resolved_helicities: tuple[tuple[int, ...], ...]
    amplitude_destinations: tuple[tuple[int, int | None], ...]
    structural_zero_physical_sector_certificate: (
        _RustRecurrenceStructuralZeroSectorCertificate | None
    )
    exact_sections: Mapping[str, object]
    generation_profile: Mapping[str, object]
    numerical_current_reuse_report: Mapping[str, object] | None
    payload_path: Path
    payload_size_bytes: int
    payload_sha256: str
    member_count: int
    unpacked_size_bytes: int
    index_sha256: str
    helicity_dispatch: _RustRecurrenceHelicityDispatchOutput | None = None


def _process_scoped_recurrence_inspection_summary(
    inspection_summary: Mapping[str, object],
    *,
    process_id: str,
    semantic_digest: str,
    schedule_digest: str,
) -> dict[str, object]:
    """Bind native inspection metadata to its published process identity."""

    result = dict(inspection_summary)
    result["process_id"] = process_id
    result["semantic_digest"] = semantic_digest
    result["schedule_digest"] = schedule_digest
    return result


def _required_process_direct_executor_ids(
    exact_sections: Mapping[str, object],
    remap: object,
) -> tuple[int, ...]:
    raw_groups = exact_sections.get("row_groups")
    if isinstance(raw_groups, str | bytes) or not isinstance(raw_groups, Sequence):
        raise GenerationError(
            "recurrence exact sections have no typed row-group sequence"
        )
    root_ids: set[int] = set()
    for index, raw_group in enumerate(raw_groups):
        if (
            isinstance(raw_group, str | bytes)
            or not isinstance(raw_group, Sequence)
            or len(raw_group) != 6
        ):
            raise GenerationError(
                f"recurrence row group {index} does not match plan-v2"
            )
        raw_executor_id = raw_group[3]
        if (
            isinstance(raw_executor_id, bool)
            or not isinstance(raw_executor_id, int)
            or raw_executor_id < 0
            or raw_executor_id > DIRECT_NONE_U32
        ):
            raise GenerationError(
                f"recurrence row group {index} has an invalid executor ID"
            )
        if raw_executor_id != DIRECT_NONE_U32:
            root_ids.add(raw_executor_id)
    changes = getattr(remap, "direct_executor_changes", None)
    count = getattr(remap, "direct_executor_count", None)
    if (
        not isinstance(changes, tuple)
        or isinstance(count, bool)
        or not isinstance(count, int)
    ):
        raise GenerationError("recurrence process has no typed executor remap")
    mapping = dict(changes)
    result = tuple(
        sorted(mapping.get(executor_id, executor_id) for executor_id in root_ids)
    )
    if (
        len(result) != len(root_ids)
        or len(set(result)) != len(result)
        or any(executor_id < 0 or executor_id >= count for executor_id in result)
    ):
        raise GenerationError(
            "recurrence process executor remap does not preserve its required set"
        )
    return result


def _validated_recurrence_evidence_fallback(
    value: object,
) -> dict[str, object]:
    """Validate the one narrow recurrence evidence-fallback record."""

    try:
        return validate_recurrence_numerical_evidence_fallback(value)
    except ValueError as exc:
        raise GenerationError(str(exc)) from exc


def _recurrence_relation_reporting(
    inspection: Mapping[str, object],
    *,
    mode: str,
    lane_report: Mapping[str, object],
    zero_certificate_baseline_reused: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    """Separate strict runtime inspection from artifact-wide provenance."""

    if mode not in {"off", "diagnostic", "certified-reuse"}:
        raise GenerationError(f"unsupported recurrence numerical current mode {mode!r}")
    certified = _result_integer(
        lane_report.get("certified_relation_count"),
        "recurrence certified relation count",
        minimum=0,
    )
    applied = _result_integer(
        lane_report.get("applied_relation_count"),
        "recurrence applied relation count",
        minimum=0,
    )
    if lane_report.get("requested_mode") != mode:
        raise GenerationError(
            "recurrence numerical current lane mode disagrees with generation"
        )
    effective_mode = lane_report.get("effective_mode")
    effective_reuse_state = lane_report.get("effective_reuse_state")
    effective_mode_reason = lane_report.get("effective_mode_reason")
    lane_state = lane_report.get("state")
    fallback_payload = lane_report.get("fallback")
    warning = lane_report.get("warning")
    if not isinstance(warning, Mapping):
        raise GenerationError(
            "recurrence numerical current report has no warning contract"
        )
    try:
        relation_correctness = recurrence_numerical_relation_correctness_summary(
            lane_report
        )
    except ValueError as exc:
        raise GenerationError(str(exc)) from exc
    no_warning: dict[str, object] = {
        "required": False,
        "emit": "never",
        "code": None,
        "message": None,
    }
    validated_fallback: dict[str, object] | None = None
    if mode == "off":
        if (
            effective_mode != "off"
            or effective_reuse_state != "disabled"
            or lane_state != "disabled-by-user"
            or fallback_payload is not None
            or certified != 0
            or applied != 0
        ):
            raise GenerationError(
                "explicit recurrence numerical opt-out metadata is inconsistent"
            )
        aggregate_warning = no_warning
    elif effective_mode == "off":
        if (
            mode != "certified-reuse"
            or effective_reuse_state != "disabled"
            or lane_state != "disabled-evidence-envelope-fallback"
            or certified != 0
            or applied != 0
        ):
            raise GenerationError(
                "recurrence numerical reuse fallback metadata is inconsistent"
            )
        validated_fallback = _validated_recurrence_evidence_fallback(fallback_payload)
        aggregate_warning = {
            "required": True,
            "emit": "once-per-generated-artifact",
            "code": EVIDENCE_ENVELOPE_FALLBACK_REASON,
            "message": EVIDENCE_ENVELOPE_FALLBACK_WARNING,
        }
    elif effective_mode in {"diagnostic", "certified-reuse"}:
        expected_applied = certified if effective_mode == "certified-reuse" else 0
        application_scope = lane_report.get("application_scope")
        if (
            effective_mode == "certified-reuse"
            and isinstance(application_scope, Mapping)
            and application_scope.get("contracted_opposite_application") == "disabled"
        ):
            application = lane_report.get("application")
            relation_counts = (
                application.get("relation_kind_counts")
                if isinstance(application, Mapping)
                else None
            )
            if not isinstance(relation_counts, Mapping):
                raise GenerationError(
                    "recurrence numerical application has no relation-kind counts"
                )
            expected_applied -= _result_integer(
                relation_counts.get("opposite"),
                "recurrence suppressed opposite relation count",
                minimum=0,
            )
        expected_state = (
            "no_certified_numerical_relation"
            if certified == 0
            else (
                "authenticated-numerical-applied"
                if effective_mode == "certified-reuse"
                else "authenticated-numerical-diagnostic-only"
            )
        )
        expected_reuse_state = (
            "enabled"
            if effective_mode == "certified-reuse" or mode == "diagnostic"
            else "disabled"
        )
        if mode == "certified-reuse" and effective_mode == "diagnostic":
            if (
                effective_mode_reason
                not in {
                    "multi-helicity-all-flow-member-scope-unproven",
                    "contracted-opposite-binary64-disabled",
                }
                or not isinstance(application_scope, Mapping)
                or application_scope.get("effective_mode_reason")
                != effective_mode_reason
            ):
                raise GenerationError(
                    "recurrence numerical diagnostic containment has no "
                    "application-scope reason"
                )
        elif effective_mode_reason is not None:
            raise GenerationError(
                "recurrence numerical current mode has a spurious downgrade reason"
            )
        if (
            (effective_mode == "certified-reuse" and mode != "certified-reuse")
            or effective_reuse_state != expected_reuse_state
            or lane_state != expected_state
            or fallback_payload is not None
            or applied != expected_applied
        ):
            raise GenerationError(
                "active recurrence numerical current metadata is inconsistent"
            )
        aggregate_warning = (
            {
                "required": True,
                "emit": "once-per-generated-artifact",
                "code": NUMERICAL_CURRENT_RELATION_WARNING_CODE,
                "message": NUMERICAL_CURRENT_RELATION_WARNING,
            }
            if applied
            else no_warning
        )
    else:
        raise GenerationError(
            "recurrence numerical current effective mode is inconsistent"
        )
    if dict(warning) != aggregate_warning:
        raise GenerationError(
            "recurrence numerical current report changed its warning contract"
        )
    warning_required = bool(aggregate_warning["required"])
    warning_code = aggregate_warning["code"]
    runtime_inspection = dict(inspection)
    native = runtime_inspection.pop("relation_discovery", None)
    if native is not None and not isinstance(native, Mapping):
        raise GenerationError(
            "native recurrence relation application report is not a mapping"
        )
    if effective_mode == "off":
        if native is not None:
            raise GenerationError(
                "disabled recurrence numerical current lane published application state"
            )
    elif native is None and zero_certificate_baseline_reused:
        if certified != 0 or applied != 0:
            raise GenerationError(
                "recurrence baseline reuse requires an empty certificate set"
            )
    else:
        if not isinstance(native, Mapping):
            raise GenerationError(
                "recurrence numerical current lane has no native application report"
            )
        native_certified = _result_integer(
            native.get("exact_certified_relation_count"),
            "native recurrence certified relation count",
            minimum=0,
        )
        native_applied = _result_integer(
            native.get("applied_relation_count"),
            "native recurrence applied relation count",
            minimum=0,
        )
        native_certificate_count = _result_integer(
            native.get("certificate_count"),
            "native recurrence certificate count",
            minimum=0,
        )
        if (
            native.get("requested_mode") != effective_mode
            or native_certified != certified
            or native_applied != applied
            or native_certificate_count != certified
        ):
            raise GenerationError(
                "native recurrence relation counts disagree with the "
                "authenticated warm-up"
            )
        if warning_required != (applied > 0) or (
            warning_required and warning_code != NUMERICAL_CURRENT_RELATION_WARNING_CODE
        ):
            raise GenerationError(
                "recurrence numerical current application or warning state "
                "is inconsistent"
            )
        application = lane_report.get("application")
        native_replay = native.get("certificate_replay")
        if not isinstance(application, Mapping) or not isinstance(
            native_replay, Mapping
        ):
            raise GenerationError(
                "recurrence numerical current certificate replay metadata is absent"
            )
        lane_replay = application.get("certificate_replay")
        if not isinstance(lane_replay, Mapping) or any(
            native_replay.get(key) != lane_replay.get(key)
            for key in ("algorithm", "certificate_set_sha256")
        ):
            raise GenerationError(
                "native recurrence certificate replay disagrees with the warm-up"
            )
        candidate_capture = lane_report.get("candidate_capture")
        verification_capture = lane_report.get("verification_capture")
        native_probe = native.get("probe")
        if (
            not isinstance(candidate_capture, Mapping)
            or not isinstance(verification_capture, Mapping)
            or not isinstance(native_probe, Mapping)
            or native_probe.get("probe_count") != candidate_capture.get("point_count")
            or native_probe.get("verification_probe_count")
            != verification_capture.get("point_count")
        ):
            raise GenerationError(
                "native recurrence probe counts disagree with the warm-up"
            )
        discovery = lane_report.get("discovery")
        if not isinstance(discovery, Mapping) or any(
            native_value != discovery.get(discovery_key)
            for native_value, discovery_key in (
                (
                    native.get("numerical_candidate_count"),
                    "numerical_candidate_count",
                ),
                (
                    native_probe.get("verification_rejected_count"),
                    "verification_rejected_count",
                ),
                (
                    native.get("uncertified_candidate_count"),
                    "verification_rejected_count",
                ),
                (
                    native.get("rejected_hypothesis_count"),
                    "rejected_hypothesis_count",
                ),
                (
                    native_probe.get("tested_hypothesis_count"),
                    "tested_hypothesis_count",
                ),
            )
        ):
            raise GenerationError(
                "native recurrence candidate counters disagree with the warm-up"
            )
        candidate_commitment = candidate_capture.get("observation_batch_sha256")
        verification_commitment = verification_capture.get("observation_batch_sha256")
        if any(
            native_probe.get(key) != expected
            for key, expected in (
                (
                    "runtime_parameter_schema_sha256",
                    candidate_capture.get("runtime_parameter_schema_sha256"),
                ),
                (
                    "candidate_observation_batch_sha256",
                    candidate_commitment,
                ),
                (
                    "verification_observation_batch_sha256",
                    verification_commitment,
                ),
                ("decision_sha256", discovery.get("decision_sha256")),
                (
                    "rejection_decision_sha256",
                    discovery.get("rejection_decision_sha256"),
                ),
            )
        ):
            raise GenerationError(
                "native recurrence raw-evidence commitments disagree with "
                "the authenticated warm-up"
            )
        rejected_diagnostics = native.get("rejected_candidate_diagnostics")
        if (
            not isinstance(rejected_diagnostics, Mapping)
            or rejected_diagnostics.get("total_rejected_hypothesis_count")
            != discovery.get("rejected_hypothesis_count")
            or rejected_diagnostics.get("full_rejection_sha256")
            != discovery.get("rejection_decision_sha256")
        ):
            raise GenerationError(
                "native recurrence rejected-hypothesis diagnostics disagree "
                "with the authenticated warm-up"
            )
    aggregate = {
        "schema_version": 1,
        "abi": "pyamplicol-artifact-numerical-current-reuse-v1",
        "requested_mode": mode,
        "effective_mode": effective_mode,
        "effective_reuse_state": effective_reuse_state,
        "execution_mode": "recurrence",
        "lane_count": 1,
        "lanes": {"primary": dict(lane_report)},
        "certified_relation_count": certified,
        "applied_relation_count": applied,
        "relation_correctness": relation_correctness,
        "structural_exact_discovery": None,
        "opt_out_flag": "--no-numerical-current-reuse",
        "fallback": validated_fallback,
        "warning": aggregate_warning,
        "native_relation_application": (None if native is None else dict(native)),
    }
    return runtime_inspection, aggregate


def _publish_validated_recurrence_baseline(
    *,
    baseline: _RustRecurrenceLoweringOutput,
    final_schedule_path: Path,
) -> _RustRecurrenceLoweringOutput:
    """Move one validated mode-off payload to its final destination."""

    if final_schedule_path.exists() or final_schedule_path.is_symlink():
        raise GenerationError(
            f"recurrence schedule destination already exists: {final_schedule_path}"
        )
    try:
        final_schedule_path.parent.mkdir(parents=True, exist_ok=True)
        baseline.payload_path.replace(final_schedule_path)
    except OSError as exc:
        raise GenerationError(
            "failed to publish the validated recurrence baseline payload"
        ) from exc
    return replace(baseline, payload_path=final_schedule_path)


def _lower_recurrence_after_numerical_warmup(
    *,
    lower_once: Callable[..., _RustRecurrenceLoweringOutput],
    final_schedule_path: Path,
    baseline: _RustRecurrenceLoweringOutput,
    warmup: RecurrenceNumericalCurrentWarmupResult,
    requested_mode: str,
) -> tuple[_RustRecurrenceLoweringOutput, bool]:
    """Lower any certified relation set, or publish its unchanged baseline."""

    if warmup.certificates:
        return (
            lower_once(
                final_schedule_path,
                mode=warmup.effective_mode,
                evidence=warmup.evidence_json,
                report_progress=False,
                timing_name="native-final-generation",
            ),
            True,
        )
    runtime_inspection, aggregate_report = _recurrence_relation_reporting(
        baseline.inspection_summary,
        mode=requested_mode,
        lane_report=warmup.to_json_dict(),
        zero_certificate_baseline_reused=True,
    )
    published = _publish_validated_recurrence_baseline(
        baseline=baseline,
        final_schedule_path=final_schedule_path,
    )
    return (
        replace(
            published,
            inspection_summary=runtime_inspection,
            generation_profile={
                "schema_version": 1,
                "native_passes": {"final": dict(baseline.generation_profile)},
            },
            numerical_current_reuse_report=aggregate_report,
        ),
        False,
    )


def _complete_recurrence_evidence_envelope_fallback(
    *,
    final_schedule_path: Path,
    baseline: _RustRecurrenceLoweringOutput,
    baseline_plan: _RecurrenceExactPlan,
    color_accuracy: str,
    outcome: RecurrenceNumericalEvidenceEnvelopeExceeded,
    warmup_owner: list[RecurrenceNumericalCurrentWarmupResult] | None = None,
) -> _RustRecurrenceLoweringOutput:
    """Publish the validated mode-off baseline after a reuse fallback."""

    if warmup_owner is None:
        warmup = None
    else:
        if len(warmup_owner) != 1:
            raise AssertionError(
                "recurrence fallback requires exactly one warm-up owner"
            )
        warmup = warmup_owner.pop()
        warmup.close()
    # Drop the transport, both captures and their spool owners before the
    # baseline payload is published.  The caller has already cleared exception
    # chains and transferred its only warm-up reference through this owner.
    warmup = None
    lane_report = recurrence_numerical_current_envelope_fallback_report(
        baseline_plan.sections,
        color_accuracy=color_accuracy,
        requested_mode="certified-reuse",
        outcome=outcome,
    )
    runtime_inspection, aggregate_report = _recurrence_relation_reporting(
        baseline.inspection_summary,
        mode="certified-reuse",
        lane_report=lane_report,
    )
    published = _publish_validated_recurrence_baseline(
        baseline=baseline,
        final_schedule_path=final_schedule_path,
    )
    return replace(
        published,
        inspection_summary=runtime_inspection,
        generation_profile={
            "schema_version": 1,
            "native_passes": {"final": dict(baseline.generation_profile)},
        },
        numerical_current_reuse_report=aggregate_report,
    )


def _detached_recurrence_evidence_envelope_outcome(
    outcome: RecurrenceNumericalEvidenceEnvelopeExceeded,
    *,
    source_error: BaseException,
) -> RecurrenceNumericalEvidenceEnvelopeExceeded:
    """Copy fallback metadata while releasing exception-retained work state."""

    detached = RecurrenceNumericalEvidenceEnvelopeExceeded(
        outcome.geometry,
        memory_envelope_bytes=outcome.memory_envelope_bytes,
        spooled_producer_resident_bytes=(outcome.spooled_producer_resident_bytes),
        raw_reason=outcome.raw_reason,
    )
    pending = [source_error]
    if outcome is not source_error:
        pending.append(outcome)
    seen: set[int] = set()
    while pending:
        error = pending.pop()
        identity = id(error)
        if identity in seen:
            continue
        seen.add(identity)
        cause = error.__cause__
        context = error.__context__
        error.__traceback__ = None
        error.__cause__ = None
        error.__context__ = None
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)
    return detached


def _invoke_rust_eager_lowering_v1(
    lowering_input: EagerLoweringInputV1,
    destination: Path,
) -> _RustEagerLoweringOutput:
    try:
        module = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(module)
    except ImportError as exc:
        raise GenerationError(
            "eager generation requires pyamplicol._rusticol with the "
            "eager plan-v3 lowering binding"
        ) from exc
    candidate = getattr(module, "_lower_eager_runtime_v1", None)
    if not callable(candidate):
        raise GenerationError(
            "pyamplicol._rusticol does not provide the required private "
            "_lower_eager_runtime_v1 binding for eager plan-v3 generation"
        )
    binding = cast(_RustEagerLoweringBinding, candidate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise GenerationError(
            f"Rust eager lowering destination already exists: {destination}"
        )
    try:
        raw_result = binding(lowering_input, os.fspath(destination))
        result = _validate_rust_eager_lowering_result(raw_result, lowering_input)
        if destination.is_symlink() or not destination.is_file():
            raise GenerationError(
                "Rust eager lowering did not produce a regular eager-runtime.pacbin"
            )
        payload_size = destination.stat().st_size
        if payload_size <= 0:
            raise GenerationError(
                "Rust eager lowering produced an empty runtime payload"
            )
        payload_sha256 = _file_sha256(destination)
    except Exception as exc:
        _discard_rust_eager_output(destination)
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError(f"Rust eager plan-v3 lowering failed: {exc}") from exc
    return _RustEagerLoweringOutput(
        inspection_summary=result["inspection_summary"],
        payload_path=destination,
        payload_size_bytes=payload_size,
        payload_sha256=payload_sha256,
        member_count=cast(int, result["member_count"]),
        unpacked_size_bytes=cast(int, result["unpacked_size_bytes"]),
        index_sha256=cast(str, result["index_sha256"]),
    )


def _invoke_rust_on_the_fly_seed_batch_builder_v1(
    ordered_source_projection_jsons: Sequence[bytes],
    recurrence_template_catalog_json: bytes,
    direct_template_catalog_json: bytes,
    prepared_kernel_pack_digest: str,
    *,
    process_identities: Sequence[str] = (),
) -> tuple[bytes, ...]:
    ordered_sources = tuple(ordered_source_projection_jsons)
    identities = tuple(process_identities)
    if identities and len(identities) != len(ordered_sources):
        raise GenerationError(
            "on-the-fly process identities do not align with source projections"
        )
    try:
        module = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(module)
    except (ImportError, OSError, RuntimeError) as exc:
        raise GenerationError(
            "on-the-fly generation requires the current native "
            "_build_on_the_fly_process_seeds_v1 binding"
        ) from exc
    candidate = getattr(module, "_build_on_the_fly_process_seeds_v1", None)
    if not callable(candidate):
        raise GenerationError(
            "pyamplicol._rusticol does not provide the private "
            "_build_on_the_fly_process_seeds_v1 binding"
        )
    binding = cast(_RustOnTheFlySeedsBinding, candidate)
    try:
        result = binding(
            ordered_sources,
            recurrence_template_catalog_json,
            direct_template_catalog_json,
            prepared_kernel_pack_digest,
        )
    except Exception as exc:
        detail = str(exc)
        attributed = next(
            (
                (index, identities[index])
                for index in range(len(identities) - 1, -1, -1)
                if f"index {index} " in detail
            ),
            None,
        )
        context = (
            ""
            if attributed is None
            else f" for process index {attributed[0]} {attributed[1]!r}"
        )
        raise GenerationError(
            f"on-the-fly process-seed construction failed{context}: {detail}"
        ) from exc
    if not isinstance(result, (list, tuple)) or len(result) != len(ordered_sources):
        raise GenerationError(
            "native on-the-fly process-seed builder returned a result with the "
            "wrong process count"
        )
    payloads: list[bytes] = []
    for index, payload in enumerate(result):
        if not isinstance(payload, bytes) or not payload:
            identity = "" if not identities else f" {identities[index]!r}"
            raise GenerationError(
                "native on-the-fly process-seed builder returned an empty payload "
                f"for process index {index}{identity}"
            )
        payloads.append(payload)
    return tuple(payloads)


def _invoke_rust_on_the_fly_seed_builder_v1(
    source_projection_json: bytes,
    recurrence_template_catalog_json: bytes,
    direct_template_catalog_json: bytes,
    prepared_kernel_pack_digest: str,
) -> bytes:
    """Retain the original singleton bridge as a plural-path wrapper."""

    return _invoke_rust_on_the_fly_seed_batch_builder_v1(
        (source_projection_json,),
        recurrence_template_catalog_json,
        direct_template_catalog_json,
        prepared_kernel_pack_digest,
    )[0]


_ON_THE_FLY_PROCESS_SEED_IDENTITY_ABI = "pyamplicol-on-the-fly-process-seed-identity-v1"
_U64_MAX = (1 << 64) - 1
_ON_THE_FLY_PROCESS_SEED_IDENTITY_FIELDS = frozenset(
    {
        "abi",
        "process_digest",
        "compiled_model_digest",
        "recurrence_template_catalog_digest",
        "prepared_kernel_pack_digest",
        "recurrence_direct_template_catalog_digest",
        "semantic_digest",
        "external_permutation",
        "external_sources",
    }
)


def _on_the_fly_selector_census_v1(
    projection: OnTheFlyProcessSeedProjectionV1,
    process: CanonicalProcessIR,
) -> dict[str, int]:
    """Count complete compact selector axes without materializing either axis."""

    helicity_count = math.prod(
        len(source.source_states) for source in projection.external_sources
    )
    if len(projection.external_sources) != len(process.legs) or any(
        leg.color_role == "inclusive" for leg in process.legs
    ):
        raise GenerationError(
            "on-the-fly selector census requires one concrete source per process leg"
        )
    fundamental_count = len(process.fundamental_labels)
    antifundamental_count = len(process.antifundamental_labels)
    adjoint_count = len(process.adjoint_labels)
    if fundamental_count != antifundamental_count:
        raise GenerationError(
            "on-the-fly selector census found unbalanced LC open-line endpoints"
        )
    if fundamental_count:
        color_flow_count = (
            math.factorial(fundamental_count)
            * math.factorial(adjoint_count)
            * math.comb(
                adjoint_count + fundamental_count - 1,
                fundamental_count - 1,
            )
        )
    elif adjoint_count:
        color_flow_count = math.factorial(max(adjoint_count - 1, 0))
    else:
        color_flow_count = 1
    census = {
        "physical_helicity_count": helicity_count,
        # Contracted NLC/full exposes one physical result; ``color_flow_count``
        # remains the private structural-selector domain bound by the color
        # payload destinations.
        "physical_color_flow_count": (
            color_flow_count if getattr(process, "color_accuracy", "lc") == "lc" else 1
        ),
    }
    for name, count in census.items():
        if count < 1 or count > _U64_MAX:
            raise GenerationError(
                f"on-the-fly selector census {name} is outside the positive u64 domain"
            )
    return census


def _on_the_fly_validation_selectors_v1(
    projection: OnTheFlyProcessSeedProjectionV1,
    *,
    color_accuracy: str = "lc",
) -> Mapping[str, tuple[str, ...]]:
    """Choose one deterministic compact selector without opening dense axes."""

    helicities: list[int] = []
    for source_index, source in enumerate(projection.external_sources):
        if not source.source_states:
            raise GenerationError(
                "on-the-fly validation source "
                f"{source_index} exposes no public helicity states"
            )
        helicities.append(int(source.source_states[0].public_helicity))
    if not helicities:
        raise GenerationError("on-the-fly validation requires external sources")
    helicity_id = "h:" + ",".join(f"{value:+d}" for value in helicities)
    return MappingProxyType(
        {
            "helicities": (helicity_id,),
            "color_flows": (() if color_accuracy in {"nlc", "full"} else ("1",)),
        }
    )


def _build_on_the_fly_contracted_color_payload_v1(
    color_plan: GenericColorPlan,
    *,
    contraction: str = "direct",
) -> _OnTheFlyContractedColorPayloadV1:
    """Encode the contracted metric over compact structural selector ordinals."""

    contraction, sector_owner_ids, owner_sector_ids = (
        build_on_the_fly_color_contraction(
            color_plan,
            contraction=contraction,
        )
    )
    group_count = contraction.group_count
    destination_count = len(owner_sector_ids)
    destination_by_group = (
        tuple(range(group_count))
        if contraction.destination_by_group is None
        else contraction.destination_by_group
    )
    if (
        len(destination_by_group) != group_count
        or len(set(destination_by_group)) != group_count
        or any(
            destination_id < 0 or destination_id >= destination_count
            for destination_id in destination_by_group
        )
    ):
        raise ValueError(
            "on-the-fly color contraction has an invalid destination projection"
        )
    normalized_owner_sector_ids = tuple(
        owner_sector_ids[destination_id] for destination_id in destination_by_group
    )
    payload = encode_recurrence_color_contraction(
        contraction,
        sector_count=color_plan.sector_count,
        component_count=1,
        ordered_group_ids=tuple(range(group_count)),
        destination_by_group=destination_by_group,
        destination_count=destination_count,
        group_sector_ids=normalized_owner_sector_ids,
        group_component_ids=(0,) * group_count,
        sector_owner_ids=sector_owner_ids,
        exact_coefficients=recurrence_exact_color_coefficients(
            color_plan,
            contraction,
            normalized_owner_sector_ids,
        ),
    )
    semantic_digest = recurrence_color_contraction_digest(payload)
    symmetric_group = contraction.symmetric_group_block
    summary = MappingProxyType(
        {
            "abi": RECURRENCE_COLOR_CONTRACTION_CODEC_ABI,
            "color_accuracy": contraction.color_accuracy,
            "storage": (
                "convolution-kernels" if symmetric_group is not None else "expanded"
            ),
            "includes_color_factor": contraction.includes_color_factor,
            "group_count": group_count,
            "sector_count": color_plan.sector_count,
            "active_sector_count": group_count,
            "component_count": 1,
            "destination_count": destination_count,
            "entry_count": (
                len(contraction.entries)
                if symmetric_group is None
                else symmetric_group.stored_entry_count
            ),
            "logical_entry_count": contraction.logical_entry_count,
            "semantic_digest": semantic_digest,
            "factorization": (
                None
                if symmetric_group is None
                else {
                    "kind": "symmetric-group-fourier",
                    "rank": symmetric_group.degree,
                    "coset_count": symmetric_group.channel_count,
                }
            ),
            **(
                {}
                if symmetric_group is None
                else {
                    "fft_provenance": {
                        "method": "symmetric-group-fourier",
                        "degree": symmetric_group.degree,
                        "channel_count": symmetric_group.channel_count,
                        "covered_local_group_count": (
                            symmetric_group.channel_count
                            * math.factorial(symmetric_group.degree)
                        ),
                        "residual_group_count": len(
                            symmetric_group.residual_local_group_indices
                        ),
                        "residual_entry_count": len(symmetric_group.residual_entries),
                        "raw_kernel_bytes": len(symmetric_group.kernel_entries) * 16,
                        "transformed_kernel_bytes": (
                            len(symmetric_group.kernel_entries) * 8
                        ),
                        "capability": (SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY),
                    }
                }
            ),
        }
    )
    return _OnTheFlyContractedColorPayloadV1(
        payload=payload,
        summary=summary,
        owner_sector_ids=normalized_owner_sector_ids,
        destination_by_group=destination_by_group,
    )


def _validate_on_the_fly_process_seed_identity_v1(
    value: object,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != (
        _ON_THE_FLY_PROCESS_SEED_IDENTITY_FIELDS
    ):
        raise GenerationError(
            "native on-the-fly process-seed identity has the wrong fields"
        )
    result = {str(key): item for key, item in value.items()}
    if result["abi"] != _ON_THE_FLY_PROCESS_SEED_IDENTITY_ABI:
        raise GenerationError(
            "native on-the-fly process-seed identity has an unsupported ABI"
        )
    for field in (
        "process_digest",
        "compiled_model_digest",
        "recurrence_template_catalog_digest",
        "prepared_kernel_pack_digest",
        "recurrence_direct_template_catalog_digest",
        "semantic_digest",
    ):
        digest = result[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise GenerationError(
                f"native on-the-fly process-seed identity {field} is not SHA-256"
            )
    permutation = result["external_permutation"]
    sources = result["external_sources"]
    if (
        not isinstance(permutation, list)
        or not isinstance(sources, list)
        or not sources
        or len(permutation) != len(sources)
        or any(
            isinstance(slot, bool) or not isinstance(slot, int) for slot in permutation
        )
        or sorted(permutation) != list(range(len(permutation)))
    ):
        raise GenerationError(
            "native on-the-fly process-seed identity has an invalid external domain"
        )
    labels: set[int] = set()
    for source_index, raw_source in enumerate(sources):
        if not isinstance(raw_source, Mapping) or set(raw_source) != {
            "source_slot",
            "public_label",
            "is_initial",
            "states",
        }:
            raise GenerationError(
                "native on-the-fly process-seed identity has an invalid source row"
            )
        source_slot = raw_source["source_slot"]
        public_label = raw_source["public_label"]
        states = raw_source["states"]
        if (
            isinstance(source_slot, bool)
            or source_slot != source_index
            or isinstance(public_label, bool)
            or not isinstance(public_label, int)
            or public_label < 1
            or public_label in labels
            or not isinstance(raw_source["is_initial"], bool)
            or not isinstance(states, list)
            or not states
        ):
            raise GenerationError(
                "native on-the-fly process-seed identity has a noncanonical "
                "source domain"
            )
        labels.add(public_label)
        previous_state_index = -1
        for raw_state in states:
            if not isinstance(raw_state, Mapping) or set(raw_state) != {
                "state_index",
                "public_helicity",
                "prepared_mass_parameter_slot",
            }:
                raise GenerationError(
                    "native on-the-fly process-seed identity has an invalid "
                    "source state"
                )
            state_index = raw_state["state_index"]
            helicity = raw_state["public_helicity"]
            mass_slot = raw_state["prepared_mass_parameter_slot"]
            if (
                isinstance(state_index, bool)
                or not isinstance(state_index, int)
                or state_index <= previous_state_index
                or isinstance(helicity, bool)
                or not isinstance(helicity, int)
                or (
                    mass_slot is not None
                    and (
                        isinstance(mass_slot, bool)
                        or not isinstance(mass_slot, int)
                        or mass_slot < 0
                    )
                )
            ):
                raise GenerationError(
                    "native on-the-fly process-seed identity has a noncanonical "
                    "source state"
                )
            previous_state_index = state_index
    return result


def _bind_on_the_fly_process_seed_identity_v1(
    identity: Mapping[str, object],
    projection: OnTheFlyProcessSeedProjectionV1,
) -> None:
    """Bind the native seed identity to its authoritative Python projection."""

    if identity["process_digest"] != projection.process_digest:
        raise GenerationError(
            "native on-the-fly process-seed process digest disagrees with "
            "the source projection"
        )
    if identity["external_permutation"] != list(projection.external_permutation):
        raise GenerationError(
            "native on-the-fly process-seed external permutation disagrees "
            "with the source projection"
        )

    native_sources = cast(list[Mapping[str, object]], identity["external_sources"])
    native_domain = tuple(
        (
            source["source_slot"],
            source["public_label"],
            source["is_initial"],
            tuple(
                (state["state_index"], state["public_helicity"])
                for state in cast(list[Mapping[str, object]], source["states"])
            ),
        )
        for source in native_sources
    )
    projected_domain = tuple(
        (
            source.source_slot,
            source.public_label,
            source.is_initial,
            tuple(
                (state.state_index, state.public_helicity)
                for state in source.source_states
            ),
        )
        for source in projection.external_sources
    )
    if native_domain != projected_domain:
        raise GenerationError(
            "native on-the-fly process-seed external source domain "
            "disagrees with the source projection"
        )


def _invoke_rust_on_the_fly_seed_inspector_v1(
    seed_payload: bytes,
) -> dict[str, object]:
    try:
        module = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(module)
    except (ImportError, OSError, RuntimeError) as exc:
        raise GenerationError(
            "on-the-fly generation requires the current native "
            "_inspect_on_the_fly_process_seed_v1 binding"
        ) from exc
    candidate = getattr(module, "_inspect_on_the_fly_process_seed_v1", None)
    if not callable(candidate):
        raise GenerationError(
            "pyamplicol._rusticol does not provide the private "
            "_inspect_on_the_fly_process_seed_v1 binding"
        )
    binding = cast(_RustOnTheFlySeedInspectorBinding, candidate)
    try:
        encoded = binding(seed_payload)
        if not isinstance(encoded, str):
            raise TypeError("native inspector did not return JSON text")
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise GenerationError(
            f"native on-the-fly process-seed inspection failed: {exc}"
        ) from exc
    except Exception as exc:
        raise GenerationError(
            f"native on-the-fly process-seed inspection failed: {exc}"
        ) from exc
    return _validate_on_the_fly_process_seed_identity_v1(decoded)


def _invoke_rust_recurrence_lowering_v2(
    builder_input: RecurrenceBuilderInputV1,
    template_input: RecurrenceTemplateInputV1,
    direct_template_catalog: RecurrenceDirectTemplateCatalogV1,
    prepared_kernel_pack_digest: str,
    schedule_semantic_digest: str,
    destination: Path,
    *,
    point_tile_size: int,
    workspace_mib: int,
    relation_discovery_mode: str,
    relation_discovery_precision_digits: int,
    relation_discovery_probe_count: int,
    relation_discovery_verification_probe_count: int,
    relation_discovery_relative_tolerance: float,
    relation_discovery_absolute_tolerance: float,
    relation_discovery_seed: int,
    color_accuracy: str,
    relation_discovery_evidence_json: bytes | None = None,
    helicity_dispatch_destination: Path | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    direct_template_catalog_json: bytes | None = None,
) -> _RustRecurrenceLoweringOutput:
    try:
        module = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(module)
    except ImportError as exc:
        raise GenerationError(
            "recurrence generation requires the native "
            "_lower_recurrence_direct_v2 binding"
        ) from exc
    candidate = getattr(module, "_lower_recurrence_direct_v2", None)
    if not callable(candidate):
        raise GenerationError(
            "pyamplicol._rusticol does not provide the private "
            "_lower_recurrence_direct_v2 binding"
        )
    binding = cast(_RustRecurrenceLoweringBinding, candidate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise GenerationError(
            f"Rust recurrence lowering destination already exists: {destination}"
        )
    if helicity_dispatch_destination is not None:
        helicity_dispatch_destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            helicity_dispatch_destination.exists()
            or helicity_dispatch_destination.is_symlink()
        ):
            raise GenerationError(
                "Rust recurrence helicity-dispatch destination already exists: "
                f"{helicity_dispatch_destination}"
            )
    try:
        encoded_direct_template_catalog = (
            json.dumps(
                direct_template_catalog.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            if direct_template_catalog_json is None
            else direct_template_catalog_json
        )
        raw_result = binding(
            builder_input,
            template_input,
            encoded_direct_template_catalog,
            prepared_kernel_pack_digest,
            schedule_semantic_digest,
            os.fspath(destination),
            point_tile_size=point_tile_size,
            workspace_mib=workspace_mib,
            relation_discovery_mode=relation_discovery_mode,
            relation_discovery_precision_digits=(relation_discovery_precision_digits),
            relation_discovery_probe_count=relation_discovery_probe_count,
            relation_discovery_verification_probe_count=(
                relation_discovery_verification_probe_count
            ),
            relation_discovery_relative_tolerance=(
                relation_discovery_relative_tolerance
            ),
            relation_discovery_absolute_tolerance=(
                relation_discovery_absolute_tolerance
            ),
            relation_discovery_seed=relation_discovery_seed,
            color_accuracy=color_accuracy,
            relation_discovery_evidence_json=relation_discovery_evidence_json,
            helicity_dispatch_destination=(
                None
                if helicity_dispatch_destination is None
                else os.fspath(helicity_dispatch_destination)
            ),
            progress_callback=progress_callback,
        )
        result = _validate_rust_recurrence_lowering_result(
            raw_result,
            builder_input=builder_input,
            template_input=template_input,
            prepared_kernel_pack_digest=prepared_kernel_pack_digest,
            direct_template_catalog_digest=direct_template_catalog.catalog_digest,
            expect_helicity_dispatch=(helicity_dispatch_destination is not None),
        )
        if destination.is_symlink() or not destination.is_file():
            raise GenerationError(
                "Rust recurrence lowering did not produce a regular "
                "recurrence-runtime.pacbin"
            )
        payload_size = destination.stat().st_size
        if payload_size <= 0:
            raise GenerationError(
                "Rust recurrence lowering produced an empty runtime payload"
            )
        serialized_bytes = cast(
            Mapping[str, object],
            cast(Mapping[str, object], result["generation_profile"])[
                "serialized_bytes"
            ],
        )
        if serialized_bytes.get("container") != payload_size:
            raise GenerationError(
                "Rust recurrence generation serialized-byte telemetry disagrees "
                "with the emitted runtime container"
            )
        payload_sha256 = _file_sha256(destination)
        dispatch_output = None
        dispatch_metadata = result["helicity_dispatch"]
        if helicity_dispatch_destination is not None:
            if (
                helicity_dispatch_destination.is_symlink()
                or not helicity_dispatch_destination.is_file()
            ):
                raise GenerationError(
                    "Rust recurrence lowering did not produce a regular "
                    "helicity-dispatch payload"
                )
            if not isinstance(dispatch_metadata, Mapping):
                raise AssertionError(
                    "validated recurrence lowering omitted helicity dispatch"
                )
            dispatch_size = helicity_dispatch_destination.stat().st_size
            dispatch_sha256 = _file_sha256(helicity_dispatch_destination)
            if (
                dispatch_size != dispatch_metadata["size_bytes"]
                or dispatch_sha256 != dispatch_metadata["sha256"]
            ):
                raise GenerationError(
                    "Rust recurrence helicity-dispatch metadata disagrees with "
                    "the emitted payload"
                )
            dispatch_output = _RustRecurrenceHelicityDispatchOutput(
                payload_path=helicity_dispatch_destination,
                payload_size_bytes=dispatch_size,
                payload_sha256=dispatch_sha256,
                base_runtime_layout_digest=cast(
                    str,
                    dispatch_metadata["base_runtime_layout_digest"],
                ),
                resolved_helicity_count=cast(
                    int,
                    dispatch_metadata["resolved_helicity_count"],
                ),
            )
    except Exception as exc:
        _discard_rust_eager_output(destination)
        if helicity_dispatch_destination is not None:
            _discard_rust_eager_output(helicity_dispatch_destination)
        if isinstance(exc, GenerationError | ArtifactError):
            raise
        raise GenerationError(f"Rust recurrence lowering failed: {exc}") from exc
    return _RustRecurrenceLoweringOutput(
        inspection_summary=cast(Mapping[str, object], result["inspection_summary"]),
        resolved_helicities=cast(
            tuple[tuple[int, ...], ...], result["resolved_helicities"]
        ),
        amplitude_destinations=cast(
            tuple[tuple[int, int | None], ...],
            result["amplitude_destinations"],
        ),
        structural_zero_physical_sector_certificate=cast(
            _RustRecurrenceStructuralZeroSectorCertificate | None,
            result["structural_zero_physical_sector_certificate"],
        ),
        exact_sections=cast(Mapping[str, object], result["exact_sections"]),
        generation_profile=cast(Mapping[str, object], result["generation_profile"]),
        numerical_current_reuse_report=None,
        payload_path=destination,
        payload_size_bytes=payload_size,
        payload_sha256=payload_sha256,
        member_count=cast(int, result["member_count"]),
        unpacked_size_bytes=cast(int, result["unpacked_size_bytes"]),
        index_sha256=cast(str, result["index_sha256"]),
        helicity_dispatch=dispatch_output,
    )


def _validate_rust_recurrence_lowering_result(
    value: object,
    *,
    builder_input: RecurrenceBuilderInputV1,
    template_input: RecurrenceTemplateInputV1,
    prepared_kernel_pack_digest: str,
    direct_template_catalog_digest: str,
    expect_helicity_dispatch: bool,
) -> dict[str, object]:
    result = _strict_mapping(
        value,
        "Rust recurrence lowering result",
        {
            "kind",
            "schema_version",
            "builder_input_abi",
            "builder_input_sha256",
            "template_input_abi",
            "template_input_sha256",
            "prepared_kernel_pack_digest",
            "direct_template_abi",
            "direct_template_catalog_digest",
            "recurrence_plan_abi",
            "runtime_layout_abi",
            "required_runtime_capabilities",
            "runtime_container",
            "inspection_summary",
            "resolved_helicities",
            "amplitude_destinations",
            "structural_zero_physical_sector_certificate",
            "helicity_dispatch",
            "exact_sections",
            "generation_profile",
        },
    )
    expected = {
        "kind": _RECURRENCE_LOWERING_RESULT_KIND,
        "schema_version": _RECURRENCE_LOWERING_RESULT_SCHEMA_VERSION,
        "builder_input_abi": RECURRENCE_BUILDER_INPUT_ABI,
        "builder_input_sha256": builder_input.digest,
        "template_input_abi": RECURRENCE_TEMPLATE_INPUT_ABI,
        "template_input_sha256": template_input.digest,
        "prepared_kernel_pack_digest": prepared_kernel_pack_digest,
        "direct_template_abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
        "direct_template_catalog_digest": direct_template_catalog_digest,
        "recurrence_plan_abi": RECURRENCE_PLAN_ABI,
        "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_ABI,
    }
    mismatched = [
        name
        for name, expected_value in expected.items()
        if result[name] != expected_value
    ]
    if mismatched:
        raise GenerationError(
            "Rust recurrence lowering result has incompatible contract fields: "
            + ", ".join(mismatched)
        )
    capabilities = result["required_runtime_capabilities"]
    color_capability = (
        RECURRENCE_CONTRACTED_COLOR_RUNTIME_CAPABILITY
        if builder_input.layout == "contracted-color-union"
        else RECURRENCE_COLOR_RUNTIME_CAPABILITY
    )
    expected_capabilities = tuple(
        sorted(
            {
                RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
                color_capability,
            }
        )
    )
    if (
        isinstance(capabilities, str | bytes)
        or not isinstance(capabilities, Sequence)
        or tuple(capabilities) != expected_capabilities
    ):
        raise GenerationError(
            "Rust recurrence lowering result has incompatible runtime capabilities"
        )
    container = _strict_mapping(
        result["runtime_container"],
        "Rust recurrence runtime container metadata",
        {
            "kind",
            "schema_version",
            "storage_abi",
            "member_count",
            "unpacked_size_bytes",
            "index_sha256",
        },
    )
    if container["kind"] != RECURRENCE_RUNTIME_CONTAINER_KIND:
        raise GenerationError("Rust recurrence runtime container kind is incompatible")
    if container["schema_version"] != RECURRENCE_RUNTIME_CONTAINER_SCHEMA_VERSION:
        raise GenerationError(
            "Rust recurrence runtime container schema is incompatible"
        )
    if container["storage_abi"] != RECURRENCE_RUNTIME_STORAGE_ABI:
        raise GenerationError(
            "Rust recurrence runtime container storage ABI is incompatible"
        )
    member_count = _result_integer(
        container["member_count"],
        "Rust recurrence runtime member count",
        minimum=1,
    )
    unpacked_size = _result_integer(
        container["unpacked_size_bytes"],
        "Rust recurrence runtime unpacked size",
        minimum=0,
    )
    index_sha256 = _result_sha256(
        container["index_sha256"],
        "Rust recurrence runtime index SHA-256",
    )
    inspection = _canonical_json_mapping(
        result["inspection_summary"],
        "Rust recurrence inspection summary",
    )
    exact_sections = _canonical_json_mapping(
        result["exact_sections"],
        "Rust recurrence exact sections",
    )
    generation_profile = _validate_recurrence_generation_profile(
        result["generation_profile"],
        inspection=inspection,
        unpacked_size_bytes=unpacked_size,
    )
    raw_dispatch = result["helicity_dispatch"]
    dispatch: dict[str, object] | None
    if raw_dispatch is None:
        if expect_helicity_dispatch:
            raise GenerationError(
                "Rust recurrence lowering omitted the requested helicity dispatch"
            )
        dispatch = None
    else:
        if not expect_helicity_dispatch:
            raise GenerationError(
                "Rust recurrence lowering emitted an unrequested helicity dispatch"
            )
        dispatch = _strict_mapping(
            raw_dispatch,
            "Rust recurrence helicity-dispatch metadata",
            {
                "abi",
                "size_bytes",
                "sha256",
                "base_runtime_layout_digest",
                "resolved_helicity_count",
            },
        )
        if dispatch["abi"] != _RECURRENCE_HELICITY_DISPATCH_ABI:
            raise GenerationError(
                "Rust recurrence helicity dispatch has an incompatible ABI"
            )
        dispatch["size_bytes"] = _result_integer(
            dispatch["size_bytes"],
            "Rust recurrence helicity-dispatch size",
            minimum=1,
        )
        dispatch["sha256"] = _result_sha256(
            dispatch["sha256"],
            "Rust recurrence helicity-dispatch SHA-256",
        )
        dispatch["base_runtime_layout_digest"] = _result_sha256(
            dispatch["base_runtime_layout_digest"],
            "Rust recurrence helicity-dispatch runtime-layout digest",
        )
        dispatch["resolved_helicity_count"] = _result_integer(
            dispatch["resolved_helicity_count"],
            "Rust recurrence helicity-dispatch resolved-helicity count",
            minimum=1,
        )
        if dispatch["base_runtime_layout_digest"] != inspection.get(
            "runtime_layout_digest"
        ):
            raise GenerationError(
                "Rust recurrence helicity dispatch disagrees with its runtime layout"
            )
    raw_helicities = result["resolved_helicities"]
    if isinstance(raw_helicities, str | bytes) or not isinstance(
        raw_helicities, Sequence
    ):
        raise GenerationError("Rust recurrence resolved helicities must be a list")
    resolved_helicities = []
    for raw_row in raw_helicities:
        if isinstance(raw_row, str | bytes) or not isinstance(raw_row, Sequence):
            raise GenerationError(
                "Rust recurrence resolved helicity rows must be lists"
            )
        row = tuple(
            _result_signed_integer(value, "Rust recurrence helicity")
            for value in raw_row
        )
        resolved_helicities.append(row)
    if len(set(resolved_helicities)) != len(resolved_helicities):
        raise GenerationError("Rust recurrence resolved helicities are not unique")
    if dispatch is not None and dispatch["resolved_helicity_count"] != len(
        resolved_helicities
    ):
        raise GenerationError(
            "Rust recurrence helicity dispatch disagrees with the resolved-helicity "
            "domain"
        )
    raw_destinations = result["amplitude_destinations"]
    if isinstance(raw_destinations, str | bytes) or not isinstance(
        raw_destinations, Sequence
    ):
        raise GenerationError("Rust recurrence amplitude destinations must be a list")
    amplitude_destinations = []
    for destination_id, raw_row in enumerate(raw_destinations):
        if (
            isinstance(raw_row, str | bytes)
            or not isinstance(raw_row, Sequence)
            or len(raw_row) != 2
        ):
            raise GenerationError(
                "Rust recurrence amplitude destination rows must be "
                "[sector_id, helicity_id-or-null]"
            )
        sector_id = _result_integer(
            raw_row[0],
            f"Rust recurrence amplitude destination {destination_id} sector ID",
            minimum=0,
        )
        raw_helicity_id = raw_row[1]
        helicity_id = (
            None
            if raw_helicity_id is None
            else _result_integer(
                raw_helicity_id,
                (f"Rust recurrence amplitude destination {destination_id} helicity ID"),
                minimum=0,
            )
        )
        amplitude_destinations.append((sector_id, helicity_id))
    if len(set(amplitude_destinations)) != len(amplitude_destinations):
        raise GenerationError(
            "Rust recurrence amplitude destinations repeat a sector/helicity pair"
        )
    raw_zero_certificate = result["structural_zero_physical_sector_certificate"]
    zero_certificate: _RustRecurrenceStructuralZeroSectorCertificate | None
    if raw_zero_certificate is None:
        if builder_input.layout == "contracted-color-union":
            raise GenerationError(
                "Rust contracted recurrence lowering omitted its complete "
                "structural-zero physical-sector certificate"
            )
        zero_certificate = None
    else:
        if builder_input.layout != "contracted-color-union":
            raise GenerationError(
                "Rust recurrence lowering emitted a structural-zero physical-sector "
                "certificate outside contracted color"
            )
        certificate = _strict_mapping(
            raw_zero_certificate,
            "Rust recurrence structural-zero physical-sector certificate",
            {
                "proof_kind",
                "sector_ids",
                "semantic_digest",
                "sector_count",
                "resolved_helicity_count",
            },
        )
        if certificate["proof_kind"] != _RECURRENCE_STRUCTURAL_ZERO_SECTOR_PROOF_KIND:
            raise GenerationError(
                "Rust recurrence structural-zero physical-sector certificate "
                "has an incompatible proof kind"
            )
        sector_count = _result_integer(
            certificate["sector_count"],
            "Rust recurrence structural-zero certificate sector count",
            minimum=1,
        )
        inspection_sector_count = _result_integer(
            inspection.get("sector_count"),
            "Rust recurrence inspection sector count",
            minimum=1,
        )
        if sector_count != inspection_sector_count:
            raise GenerationError(
                "Rust recurrence structural-zero certificate disagrees with the "
                "physical-sector domain"
            )
        resolved_count = _result_integer(
            certificate["resolved_helicity_count"],
            "Rust recurrence structural-zero certificate resolved-helicity count",
            minimum=1,
        )
        if resolved_count != len(resolved_helicities):
            raise GenerationError(
                "Rust recurrence structural-zero certificate disagrees with the "
                "resolved-helicity domain"
            )
        raw_sector_ids = certificate["sector_ids"]
        if isinstance(raw_sector_ids, str | bytes) or not isinstance(
            raw_sector_ids, Sequence
        ):
            raise GenerationError(
                "Rust recurrence structural-zero physical-sector IDs must be a list"
            )
        sector_ids = tuple(
            _result_integer(
                value,
                "Rust recurrence certified structural-zero physical-sector ID",
                minimum=0,
            )
            for value in raw_sector_ids
        )
        if tuple(sorted(set(sector_ids))) != sector_ids or any(
            sector_id >= sector_count for sector_id in sector_ids
        ):
            raise GenerationError(
                "Rust recurrence certified structural-zero physical-sector IDs "
                "are not canonical"
            )
        active_sector_ids = {sector_id for sector_id, _ in amplitude_destinations}
        if active_sector_ids.intersection(sector_ids):
            raise GenerationError(
                "Rust recurrence structural-zero certificate overlaps active "
                "amplitude destinations"
            )
        zero_certificate = _RustRecurrenceStructuralZeroSectorCertificate(
            sector_ids=sector_ids,
            semantic_digest=_result_sha256(
                certificate["semantic_digest"],
                "Rust recurrence structural-zero certificate semantic digest",
            ),
            sector_count=sector_count,
            resolved_helicity_count=resolved_count,
        )
        provenance = _strict_mapping(
            inspection.get("structural_zero_physical_sector_certificate"),
            "Rust recurrence structural-zero physical-sector provenance",
            {
                "proof_kind",
                "semantic_digest",
                "sector_count",
                "resolved_helicity_count",
                "certified_structural_zero_sector_count",
            },
        )
        expected_provenance = {
            "proof_kind": _RECURRENCE_STRUCTURAL_ZERO_SECTOR_PROOF_KIND,
            "semantic_digest": zero_certificate.semantic_digest,
            "sector_count": zero_certificate.sector_count,
            "resolved_helicity_count": zero_certificate.resolved_helicity_count,
            "certified_structural_zero_sector_count": len(zero_certificate.sector_ids),
        }
        if provenance != expected_provenance:
            raise GenerationError(
                "Rust recurrence structural-zero physical-sector provenance "
                "disagrees with its certificate"
            )
    return {
        "inspection_summary": inspection,
        "resolved_helicities": tuple(resolved_helicities),
        "amplitude_destinations": tuple(amplitude_destinations),
        "structural_zero_physical_sector_certificate": zero_certificate,
        "exact_sections": exact_sections,
        "generation_profile": generation_profile,
        "member_count": member_count,
        "unpacked_size_bytes": unpacked_size,
        "index_sha256": index_sha256,
        "helicity_dispatch": dispatch,
    }


def _validate_recurrence_generation_profile(
    value: object,
    *,
    inspection: Mapping[str, object],
    unpacked_size_bytes: int,
) -> dict[str, object]:
    profile = _strict_mapping(
        value,
        "Rust recurrence generation profile",
        {
            "schema_version",
            "scope",
            "timings_seconds",
            "operation_counters",
            "serialized_bytes",
        },
    )
    if profile["schema_version"] != 1 or profile["scope"] != "generation-only":
        raise GenerationError(
            "Rust recurrence generation profile has an incompatible contract"
        )
    timings = _strict_mapping(
        profile["timings_seconds"],
        "Rust recurrence generation timings",
        set(_RECURRENCE_GENERATION_PROFILE_TIMINGS),
    )
    normalized_timings: dict[str, float] = {}
    for name, value in timings.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise GenerationError(
                f"Rust recurrence generation timing {name!r} is invalid"
            )
        normalized_timings[name] = float(value)

    counters = _strict_mapping(
        profile["operation_counters"],
        "Rust recurrence generation operation counters",
        set(_RECURRENCE_GENERATION_PROFILE_COUNTERS),
    )
    normalized_counters = {
        name: _result_integer(
            counter,
            f"Rust recurrence generation counter {name!r}",
            minimum=0,
        )
        for name, counter in counters.items()
    }
    serialized = _strict_mapping(
        profile["serialized_bytes"],
        "Rust recurrence generation serialized bytes",
        set(_RECURRENCE_GENERATION_PROFILE_SERIALIZED_BYTES),
    )
    normalized_serialized = {
        name: _result_integer(
            count,
            f"Rust recurrence generation serialized byte count {name!r}",
            minimum=0,
        )
        for name, count in serialized.items()
    }
    member = inspection.get("runtime_container_member")
    if not isinstance(member, Mapping):
        raise GenerationError(
            "Rust recurrence inspection has no runtime-container member metadata"
        )
    expected_serialized = {
        "plan_payload": _result_integer(
            member.get("size_bytes"),
            "Rust recurrence runtime plan payload size",
            minimum=0,
        ),
        "container": _result_integer(
            member.get("container_size_bytes"),
            "Rust recurrence runtime container size",
            minimum=0,
        ),
        "unpacked_container": unpacked_size_bytes,
    }
    if normalized_serialized != expected_serialized:
        raise GenerationError(
            "Rust recurrence generation serialized-byte telemetry disagrees "
            "with authenticated container metadata"
        )
    return {
        "schema_version": 1,
        "scope": "generation-only",
        "timings_seconds": normalized_timings,
        "operation_counters": normalized_counters,
        "serialized_bytes": normalized_serialized,
    }


def _validate_rust_eager_lowering_result(
    value: object,
    lowering_input: EagerLoweringInputV1,
) -> dict[str, object]:
    result = _strict_mapping(
        value,
        "Rust eager lowering result",
        {
            "kind",
            "schema_version",
            "lowering_input_abi",
            "lowering_input_sha256",
            "eager_plan_abi",
            "runtime_layout_abi",
            "required_runtime_capabilities",
            "runtime_container",
            "inspection_summary",
        },
    )
    expected = {
        "kind": _EAGER_LOWERING_RESULT_KIND,
        "schema_version": _EAGER_LOWERING_RESULT_SCHEMA_VERSION,
        "lowering_input_abi": EAGER_LOWERING_INPUT_ABI,
        "lowering_input_sha256": lowering_input.digest,
        "eager_plan_abi": EAGER_PLAN_V3_ABI,
        "runtime_layout_abi": EAGER_RUNTIME_LAYOUT_ABI,
    }
    mismatched = [
        name
        for name, expected_value in expected.items()
        if result[name] != expected_value
    ]
    if mismatched:
        raise GenerationError(
            "Rust eager lowering result has incompatible contract fields: "
            + ", ".join(mismatched)
        )
    capabilities = result["required_runtime_capabilities"]
    if (
        isinstance(capabilities, str | bytes)
        or not isinstance(capabilities, Sequence)
        or tuple(capabilities)
        != tuple(
            sorted(
                (
                    EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
                    EAGER_PLAN_V3_RUNTIME_CAPABILITY,
                )
            )
        )
    ):
        raise GenerationError(
            "Rust eager lowering result has an incompatible runtime capability"
        )
    container = _strict_mapping(
        result["runtime_container"],
        "Rust eager runtime container metadata",
        {
            "kind",
            "schema_version",
            "storage_abi",
            "member_count",
            "unpacked_size_bytes",
            "index_sha256",
        },
    )
    if container["kind"] != EAGER_RUNTIME_CONTAINER_KIND:
        raise GenerationError("Rust eager runtime container kind is incompatible")
    if container["schema_version"] != EAGER_RUNTIME_CONTAINER_SCHEMA_VERSION:
        raise GenerationError("Rust eager runtime container schema is incompatible")
    if container["storage_abi"] != EAGER_RUNTIME_STORAGE_ABI:
        raise GenerationError(
            "Rust eager runtime container storage ABI is incompatible"
        )
    member_count = _result_integer(
        container["member_count"],
        "Rust eager runtime member count",
        minimum=1,
    )
    unpacked_size = _result_integer(
        container["unpacked_size_bytes"],
        "Rust eager runtime unpacked size",
        minimum=0,
    )
    index_sha256 = _result_sha256(
        container["index_sha256"],
        "Rust eager runtime index SHA-256",
    )
    inspection = _canonical_json_mapping(
        result["inspection_summary"],
        "Rust eager inspection summary",
    )
    return {
        "inspection_summary": inspection,
        "member_count": member_count,
        "unpacked_size_bytes": unpacked_size,
        "index_sha256": index_sha256,
    }


def _strict_mapping(
    value: object,
    context: str,
    fields: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise GenerationError(f"{context} must be an object")
    result = {str(key): item for key, item in value.items()}
    if set(result) != fields:
        raise GenerationError(f"{context} fields do not match the v1 binding protocol")
    return result


def _canonical_json_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise GenerationError(f"{context} must be an object")
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        result = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise GenerationError(f"{context} is not canonical JSON: {exc}") from exc
    if not isinstance(result, dict):  # pragma: no cover - Mapping encoded above
        raise GenerationError(f"{context} must encode as an object")
    return cast(dict[str, object], result)


def _result_integer(value: object, context: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GenerationError(f"{context} must be at least {minimum}")
    return value


def _result_signed_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GenerationError(f"{context} must be an integer")
    return value


def _result_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GenerationError(f"{context} must be a lowercase hexadecimal digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _discard_rust_eager_output(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
    except OSError:
        pass


def _eager_lowering_kernel_ids(
    lowering_input: EagerLoweringInputV1,
) -> frozenset[int]:
    references: set[int] = set()
    for table_name, column_name in (
        ("currents", "propagator_kernel_id"),
        ("interactions", "kernel_id"),
        ("roots", "kernel_id"),
    ):
        for raw_value in lowering_input.table(table_name).column(column_name):
            value = int(raw_value)
            if value != MISSING_U32:
                references.add(value)
    return frozenset(references)


def _builtin_sm_model() -> Model:
    from ..models import BuiltinSMModel

    return BuiltinSMModel()


def _nested_config_value(values: Mapping[str, object], path: str) -> object:
    current: object = values
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise GenerationError(
                f"prepared model references unknown configuration path {path!r}"
            )
        current = current[component]
    return current


def _prepared_pack_effective_values(
    pack: PreparedKernelPack,
) -> dict[str, object]:
    settings = pack.optimization_settings

    def required(name: str) -> object:
        if name not in settings:
            raise GenerationError(
                f"prepared kernel pack omits optimization setting {name!r}"
            )
        return settings[name]

    def required_bool(name: str) -> bool:
        value = required(name)
        if not isinstance(value, bool):
            raise GenerationError(
                f"prepared kernel pack has a non-boolean optimization setting {name!r}"
            )
        return value

    values: dict[str, object] = {
        "evaluator.backend": pack.backend,
        "evaluator.optimization.horner_iterations": required("iterations"),
        "evaluator.optimization.cpe_iterations": required("cpe_iterations"),
        "evaluator.optimization.max_horner_variables": required(
            "max_horner_scheme_variables"
        ),
        "evaluator.optimization.max_common_pair_cache_entries": required(
            "max_common_pair_cache_entries"
        ),
        "evaluator.optimization.max_common_pair_distance": required(
            "max_common_pair_distance"
        ),
        "evaluator.optimization.collect_factors": required("collect_factors"),
    }
    if pack.backend == "jit":
        values["evaluator.jit.optimization_level"] = required("jit_optimization_level")
        values["evaluator.jit.compress"] = required_bool("jit_compress")
        return values
    optimization_level = required("compiled_optimization_level")
    if isinstance(optimization_level, bool) or not isinstance(optimization_level, int):
        raise GenerationError(
            "prepared native kernel pack has an invalid optimization level"
        )
    values.update(
        {
            "evaluator.cpp.optimization": f"O{optimization_level}",
            "evaluator.cpp.native_arch": required("compiled_native"),
            "evaluator.cpp.compiler": required("compiler_path"),
            "evaluator.cpp.extra_flags": required("compiler_flags"),
        }
    )
    return values


@dataclass(frozen=True, slots=True)
class _ResolvedModel:
    source: ModelSource
    model: Model | None
    compiled: _CompiledModelPayload | None = None
    use_compiled_process_catalog: bool = True
    eager_kernel_index: PreparedCatalogEagerKernelIndex | None = None


@dataclass(frozen=True, slots=True)
class _ProcessSelection:
    max_color_sectors: int | None = None
    reference_color_order: tuple[int, ...] | None = None
    selected_color_sector_ids: frozenset[int] | None = None
    selected_source_helicities: Mapping[int, int] | None = None

    def __post_init__(self) -> None:
        if self.reference_color_order is not None:
            object.__setattr__(
                self,
                "reference_color_order",
                tuple(int(label) for label in self.reference_color_order),
            )
        if self.selected_color_sector_ids is not None:
            object.__setattr__(
                self,
                "selected_color_sector_ids",
                frozenset(int(value) for value in self.selected_color_sector_ids),
            )
        if self.selected_source_helicities is not None:
            object.__setattr__(
                self,
                "selected_source_helicities",
                MappingProxyType(
                    {
                        int(label): int(helicity)
                        for label, helicity in self.selected_source_helicities.items()
                    }
                ),
            )


@dataclass(frozen=True, slots=True)
class _ExpandedProcess:
    request: ProcessRequest
    process_ir: CanonicalProcessIR
    aliases: tuple[Mapping[str, object], ...] = ()
    source_expansion_size: int = 1
    source_request: ProcessRequest | None = None
    # Flavour-blind members represented by this process, never generated.
    group_members: tuple[GroupMember, ...] = ()


class _NoModelSupportedAmplitudes(GenerationError):
    """A concrete tree-level subprocess has no amplitude in the active model."""


@dataclass(frozen=True, slots=True)
class _UnsupportedProcess:
    expanded: _ExpandedProcess
    reason: str


_SupportedT = TypeVar("_SupportedT", "_DagProcess", _ExpandedProcess)


def _source_request(expanded: _ExpandedProcess) -> ProcessRequest:
    return expanded.source_request or expanded.request


def _partition_model_supported_processes(
    results: Sequence[_SupportedT | _UnsupportedProcess],
) -> tuple[tuple[_SupportedT, ...], tuple[_UnsupportedProcess, ...]]:
    """Retain supported children without silently dropping a whole request."""

    unsupported = tuple(
        result for result in results if isinstance(result, _UnsupportedProcess)
    )
    supported = tuple(
        cast(_SupportedT, result)
        for result in results
        if not isinstance(result, _UnsupportedProcess)
    )
    supported_request_names = {
        _source_request(
            result if isinstance(result, _ExpandedProcess) else result.expanded
        ).name
        for result in supported
    }
    unsupported_by_request: dict[str, list[_UnsupportedProcess]] = {}
    for result in unsupported:
        source = _source_request(result.expanded)
        unsupported_by_request.setdefault(source.name, []).append(result)
    empty_requests = tuple(
        (request_name, entries)
        for request_name, entries in unsupported_by_request.items()
        if request_name not in supported_request_names
    )
    if empty_requests:
        details = []
        for _request_name, entries in empty_requests:
            source = _source_request(entries[0].expanded)
            reasons = "; ".join(entry.reason for entry in entries)
            details.append(f"{source.expression!r}: {reasons}")
        raise GenerationError(
            "process expansion produced no model-supported tree-level amplitudes "
            "for requested process"
            + ("es" if len(details) != 1 else "")
            + ": "
            + "; ".join(details)
        )
    return supported, unsupported


def _warn_unsupported_processes(unsupported: Sequence[_UnsupportedProcess]) -> None:
    if not unsupported:
        return
    rendered = ", ".join(
        f"{item.expanded.request.name} ({item.expanded.process_ir.process}"
        + (
            f", with {len(item.expanded.group_members)} flavor-blind member(s)"
            if item.expanded.group_members
            else ""
        )
        + ")"
        for item in unsupported
    )
    _LOGGER.warning(
        "Skipped %d concrete subprocess%s with no model-supported tree-level "
        "amplitudes: %s",
        len(unsupported),
        "" if len(unsupported) == 1 else "es",
        rendered,
    )


@dataclass(frozen=True, slots=True)
class _PreparedProcessConstruction:
    process: CanonicalProcessIR
    complete_color_plan: GenericColorPlan
    restricted_color_plan: GenericColorPlan
    materialized_sector_ids: frozenset[int] | None
    coupling_order_limits: Mapping[str, int]
    topology_replay: LCColorTopologyReplayPlan | ColorTopologyReplayCertificate | None


@dataclass(frozen=True, slots=True)
class _DagProcess:
    expanded: _ExpandedProcess
    dag: GenericDAG
    coverage: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _CompiledProcess:
    expanded: _ExpandedProcess
    dag: GenericDAG
    helicity_sum_dag: GenericDAG | None
    helicity_selector_union_dag: GenericDAG | None
    coverage: Mapping[str, object]
    filters: Mapping[str, object]
    validation_points: tuple[ValidationPointRecord, ...]


@dataclass(frozen=True, slots=True)
class _GeneratedRecurrenceProcess:
    expanded: _ExpandedProcess
    artifact: RecurrenceProcessArtifact
    validation_points: tuple[ValidationPointRecord, ...]


@dataclass(frozen=True, slots=True)
class _GeneratedOnTheFlyProcess:
    expanded: _ExpandedProcess
    artifact: OnTheFlyProcessArtifact
    validation_points: tuple[ValidationPointRecord, ...]
    validation_selectors: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _ProjectedOnTheFlyProcess:
    index: int
    expanded: _ExpandedProcess
    projection: OnTheFlyGenerationProjectionV1


@dataclass(frozen=True, slots=True)
class _OnTheFlyContractedColorPayloadV1:
    payload: bytes
    summary: Mapping[str, object]
    owner_sector_ids: tuple[int, ...]
    destination_by_group: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _RecurrenceModelInputs:
    bundle: PreparedModelBundle
    catalog: RecurrenceTemplateCatalog
    template_catalog_json: bytes
    template_input: RecurrenceTemplateInputV1
    direct_template_catalog: RecurrenceDirectTemplateCatalogV1
    direct_template_catalog_json: bytes
    prepared_kernel_pack_digest: str


@dataclass(frozen=True, slots=True)
class _EvaluatorProcess:
    compiled: _CompiledProcess
    runtime_schema: RuntimeExpressionSchema
    stage_input: StageCompilationInput
    color_contraction_payload: bytes | None = None
    helicity_sum_runtime_schema: RuntimeExpressionSchema | None = None
    helicity_sum_stage_input: StageCompilationInput | None = None
    helicity_selector_lanes: tuple[_HelicitySelectorLane, ...] = ()
    color_selector_lanes: tuple[_ColorSelectorLane, ...] = ()


@dataclass(frozen=True, slots=True)
class _ColorSelectorLane:
    materialized_sector_id: int
    dag: GenericDAG
    runtime_schema: RuntimeExpressionSchema
    stage_input: StageCompilationInput


@dataclass(frozen=True, slots=True)
class _HelicitySelectorLane:
    selector_domain_ids: tuple[int, ...]
    active_current_ids: tuple[int, ...]
    active_root_ids: tuple[int, ...]
    dag: GenericDAG
    runtime_schema: RuntimeExpressionSchema
    stage_input: StageCompilationInput
    schedule_mode: Literal["parent-closure", "nested-runtime"] = "parent-closure"
    child_lanes: tuple[_HelicitySelectorLane, ...] = ()


def _validate_compiled_symmetric_group_diagnostic_process(
    process: _CompiledProcess,
) -> None:
    """Fail closed outside the deliberately narrow compiled diagnostic lane."""

    dag = process.dag
    if dag.color_plan.color_accuracy != "full":
        raise GenerationError(
            "compiled symmetric-group FFT diagnostic execution requires FullColour"
        )
    if not dag.selected_source_helicities or dag.helicity_coverage != "selected":
        raise GenerationError(
            "compiled symmetric-group FFT diagnostic execution requires one "
            "generation-time selected source-helicity assignment"
        )
    selected_source_labels = {
        int(label) for label, _helicity in dag.selected_source_helicities
    }
    external_source_labels = {int(leg.label) for leg in dag.process.legs}
    if selected_source_labels != external_source_labels:
        raise GenerationError(
            "compiled symmetric-group FFT diagnostic execution requires the "
            "selected source-label domain to match every resolved external leg"
        )
    if (
        process.helicity_sum_dag is not None
        or process.helicity_selector_union_dag is not None
        or dag.helicity_recurrence is not None
        or dag.helicity_materialization is not None
    ):
        raise GenerationError(
            "compiled symmetric-group FFT diagnostic execution does not support "
            "helicity selector or materialization lanes"
        )
    if dag.lc_topology_replay is not None or dag.color_topology_replay is not None:
        raise GenerationError(
            "compiled symmetric-group FFT diagnostic execution does not support "
            "colour-topology replay"
        )


def _compiled_symmetric_group_color_contraction_payload(
    dag: GenericDAG,
    layout: RuntimeSchemaLayout,
) -> bytes:
    contraction = layout.color_contraction
    if contraction is None or not contraction.supported:
        reason = None if contraction is None else contraction.reason
        raise GenerationError(
            "compiled symmetric-group FFT colour contraction is unavailable"
            + ("" if reason is None else f": {reason}")
        )
    block = contraction.symmetric_group_block
    if block is None:
        raise GenerationError(
            "compiled symmetric-group FFT is not applicable to this process: "
            "color planning found no certified nontrivial symmetric-group orbit "
            "to Fourier transform; compiled direct contraction remains supported "
            "with color.contraction='direct'"
        )
    if block.component_count != 1:
        raise GenerationError(
            "compiled symmetric-group FFT diagnostic execution requires exactly "
            "one selected-helicity contraction component"
        )
    group_count = contraction.group_count
    descriptor_by_group = {
        descriptor.group_id: descriptor
        for descriptor in layout.color_contraction_descriptors
    }
    if len(descriptor_by_group) != group_count or set(descriptor_by_group) != set(
        range(group_count)
    ):
        raise GenerationError(
            "compiled symmetric-group FFT coherent group IDs are not a dense "
            "authenticated destination domain"
        )
    destination_by_group = (
        tuple(range(group_count))
        if contraction.destination_by_group is None
        else contraction.destination_by_group
    )
    if len(destination_by_group) != group_count or set(destination_by_group) != set(
        range(group_count)
    ):
        raise GenerationError(
            "compiled symmetric-group FFT destination projection is not a "
            "permutation of coherent group IDs"
        )
    group_sector_ids = tuple(
        descriptor_by_group[destination_id].sector_id
        for destination_id in destination_by_group
    )
    active_sector_ids = frozenset(group_sector_ids)
    return encode_recurrence_color_contraction(
        contraction,
        sector_count=dag.color_plan.sector_count,
        component_count=1,
        ordered_group_ids=block.component_group_ids,
        destination_by_group=destination_by_group,
        destination_count=group_count,
        group_sector_ids=group_sector_ids,
        group_component_ids=(0,) * group_count,
        sector_owner_ids=tuple(
            sector_id if sector_id in active_sector_ids else MISSING_U32
            for sector_id in range(dag.color_plan.sector_count)
        ),
        exact_coefficients=recurrence_exact_color_coefficients(
            dag.color_plan,
            contraction,
            group_sector_ids,
        ),
    )


def _compiled_lc_color_selector_lane_dags(
    dag: GenericDAG,
) -> tuple[tuple[int, GenericDAG], ...]:
    if dag.color_plan.color_accuracy != "lc" or dag.color_coverage != "complete":
        return ()
    lanes: list[tuple[int, GenericDAG]] = []
    for sector_id in contributing_color_sector_ids(dag):
        lane_dag = filter_dag_to_color_sectors(dag, (sector_id,))
        if not lane_dag.amplitude_roots:
            raise GenerationError(
                "compiled LC colour-selector lane has no amplitude roots for "
                f"materialized sector {sector_id}"
            )
        lanes.append((sector_id, lane_dag))
    return tuple(lanes)


def _compiled_helicity_selector_closure_lanes(
    dag: GenericDAG,
    schema: RuntimeExpressionSchema,
    model: Model,
) -> tuple[_HelicitySelectorLane, ...]:
    """Compile no-replay LC evaluator lanes for exact helicity closures."""

    materialization = dag.helicity_materialization
    if (
        materialization is None
        or dag.lc_topology_replay is not None
        or dag.color_topology_replay is not None
    ):
        return ()
    grouped_domains: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        list[int],
    ] = {}
    for schedule in materialization.selector_schedules:
        if schedule.structural_zero:
            continue
        closure = (schedule.active_current_ids, schedule.active_root_ids)
        grouped_domains.setdefault(closure, []).append(schedule.selector_domain_id)

    stripped_dag = replace(
        dag,
        helicity_recurrence=None,
        helicity_materialization=None,
    )
    compile_dag = replace(
        stripped_dag,
        lc_topology_replay=None,
        color_topology_replay=None,
    )
    lanes: list[_HelicitySelectorLane] = []
    for (active_current_ids, active_root_ids), domain_ids in sorted(
        grouped_domains.items(),
        key=lambda item: tuple(item[1]),
    ):
        lane_schema = _helicity_closure_runtime_schema(
            schema,
            dag,
            active_current_ids=active_current_ids,
        )
        if lane_schema is None:
            # Keep the exact partitioned primary path for an unusual closure
            # that does not traverse every compiled recursion stage.
            continue
        lanes.append(
            _HelicitySelectorLane(
                selector_domain_ids=tuple(sorted(domain_ids)),
                active_current_ids=active_current_ids,
                active_root_ids=active_root_ids,
                dag=stripped_dag,
                runtime_schema=lane_schema,
                stage_input=StageCompilationInput(
                    compile_dag,
                    model,
                    lane_schema,
                ),
            )
        )
    return tuple(lanes)


def _helicity_closure_runtime_schema(
    schema: RuntimeExpressionSchema,
    dag: GenericDAG,
    *,
    active_current_ids: Sequence[int],
) -> RuntimeExpressionSchema | None:
    """Filter compiled stages while retaining the parent's stable storage ABI."""

    payload = schema.to_mapping()
    if payload.pop("helicity_recurrence", None) is None:
        raise GenerationError(
            "compiled helicity closure lane has no recurrence metadata to strip"
        )
    active = {int(current_id) for current_id in active_current_ids}
    value_slots = {
        int(record["value_slot_id"]): record
        for record in cast(
            "Sequence[Mapping[str, object]]",
            cast("Mapping[str, object]", payload["value_storage"])["value_slots"],
        )
    }
    momentum_slot_by_mask = {
        int(record["momentum_mask"]): int(record["momentum_slot_id"])
        for record in cast(
            "Sequence[Mapping[str, object]]",
            payload["momentum_slots"],
        )
    }
    filtered_stages: list[dict[str, object]] = []
    for raw_stage in cast("Sequence[Mapping[str, object]]", payload["stages"]):
        stage = dict(raw_stage)
        interaction_ids = tuple(
            int(interaction_id)
            for interaction_id in cast(
                "Sequence[object]",
                stage.get("interaction_ids", ()),
            )
            if dag.interactions[int(interaction_id)].result_id in active
        )
        if not interaction_ids:
            return None
        interactions = tuple(dag.interactions[item] for item in interaction_ids)
        input_current_ids = {
            current_id
            for interaction in interactions
            for current_id in (interaction.left_id, interaction.right_id)
        }
        output_current_ids = {interaction.result_id for interaction in interactions}
        input_value_slot_ids = tuple(
            sorted(
                int(slot_id)
                for slot_id in cast(
                    "Sequence[object]",
                    stage["input_value_slot_ids"],
                )
                if int(value_slots[int(slot_id)]["current_id"]) in input_current_ids
            )
        )
        output_value_slot_ids = tuple(
            sorted(
                int(slot_id)
                for slot_id in cast(
                    "Sequence[object]",
                    stage["output_value_slot_ids"],
                )
                if int(value_slots[int(slot_id)]["current_id"]) in output_current_ids
            )
        )
        momentum_slot_ids = {
            momentum_slot_by_mask[dag.currents[current_id].index.momentum_mask]
            for interaction in interactions
            for current_id in (
                interaction.left_id,
                interaction.right_id,
                interaction.result_id,
            )
        }
        evaluation_groups = {
            (
                "group",
                int(interaction.evaluation_group_id),
            )
            if interaction.evaluation_group_id is not None
            else ("interaction", interaction.id)
            for interaction in interactions
        }
        stage.update(
            {
                "input_current_ids": sorted(input_current_ids),
                "output_current_ids": sorted(output_current_ids),
                "input_value_slot_ids": list(input_value_slot_ids),
                "output_value_slot_ids": list(output_value_slot_ids),
                "input_momentum_slot_ids": sorted(momentum_slot_ids),
                "interaction_count": len(interaction_ids),
                "interaction_evaluation_count": len(evaluation_groups),
                "interaction_ids": list(interaction_ids),
                "interactions_compacted": True,
                "interactions": [],
            }
        )
        filtered_stages.append(stage)
    payload["stages"] = filtered_stages
    return RuntimeExpressionSchema.from_mapping(payload)


def _complete_helicity_domain_contract(
    dag: GenericDAG,
) -> tuple[
    frozenset[tuple[tuple[int, int], ...]],
    frozenset[tuple[tuple[int, int], ...]],
]:
    plan = dag.helicity_recurrence
    if plan is None:
        raise GenerationError("helicity selector lane has no recurrence proof")
    complete_by_id = {
        domain.id: domain.source_states
        for domain in plan.selector_domains
        if domain.complete
    }
    complete = frozenset(complete_by_id.values())
    structural_zero = frozenset(
        complete_by_id[domain_id]
        for domain_id in plan.structural_zero_selector_domain_ids
        if domain_id in complete_by_id
    )
    return complete, structural_zero


def _validate_matching_helicity_selector_domains(
    parent: GenericDAG,
    nested: GenericDAG,
) -> None:
    parent_complete, parent_zero = _complete_helicity_domain_contract(parent)
    nested_complete, nested_zero = _complete_helicity_domain_contract(nested)
    if parent_complete != nested_complete:
        raise GenerationError(
            "complete-color helicity selector lane disagrees with the parent "
            "source-state domain"
        )
    if parent_zero != nested_zero:
        raise GenerationError(
            "complete-color helicity selector lane disagrees with the parent "
            "structural-zero domain"
        )


def _numerical_current_warning_required(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    warning = value.get("warning")
    return isinstance(warning, Mapping) and warning.get("required") is True


def _map_process_phase(
    items: Sequence[_ProcessInput],
    operation: Callable[[_ProcessInput], _ProcessOutput],
    *,
    executor: ThreadPoolExecutor | None,
    max_in_flight: int,
    phase_name: str,
    item_name: Callable[[_ProcessInput], str],
) -> tuple[_ProcessOutput, ...]:
    """Apply one process phase with bounded submissions and stable output order."""

    if max_in_flight < 1:
        raise ValueError("process phase max_in_flight must be positive")
    if executor is None:
        results: list[_ProcessOutput] = []
        for item in items:
            try:
                results.append(operation(item))
            except Exception as exc:
                raise GenerationError(
                    f"{phase_name} failed for process {item_name(item)!r}: {exc}"
                ) from exc
        return tuple(results)

    ordered: list[_ProcessOutput | object] = [_MISSING_PROCESS_RESULT for _ in items]
    active: dict[Future[_ProcessOutput], int] = {}
    next_index = 0

    def submit_one(index: int) -> None:
        item = items[index]
        try:
            future = executor.submit(operation, item)
        except Exception as exc:
            for pending in active:
                pending.cancel()
            if active:
                wait(tuple(active))
            raise GenerationError(
                f"{phase_name} failed to schedule process {item_name(item)!r}: {exc}"
            ) from exc
        active[future] = index

    while next_index < len(items) and len(active) < max_in_flight:
        submit_one(next_index)
        next_index += 1

    while active:
        done, _pending = wait(tuple(active), return_when=FIRST_COMPLETED)
        failures: list[tuple[int, Exception]] = []
        for future in done:
            index = active.pop(future)
            try:
                ordered[index] = future.result()
            except Exception as exc:
                failures.append((index, exc))
        if failures:
            for pending in active:
                pending.cancel()
            if active:
                wait(tuple(active))
            index, exc = min(failures, key=lambda failure: failure[0])
            item = items[index]
            raise GenerationError(
                f"{phase_name} failed for process {item_name(item)!r}: {exc}"
            ) from exc
        while next_index < len(items) and len(active) < max_in_flight:
            submit_one(next_index)
            next_index += 1

    if any(result is _MISSING_PROCESS_RESULT for result in ordered):
        raise GenerationError(f"{phase_name} did not produce every process result")
    return tuple(cast(_ProcessOutput, result) for result in ordered)


def _initial_process_groups(
    artifact_processes: Sequence[ProcessArtifact],
    expanded: Iterable[_ExpandedProcess],
) -> ProcessGroupMap:
    """Each written process with the members recorded at expansion."""

    members = {entry.request.name: entry.group_members for entry in expanded}
    return initial_process_groups(
        (
            (process.process_id, process.expression, process.external_pdgs)
            for process in artifact_processes
        ),
        members,
    )


class GenerationBackend:
    def __init__(
        self,
        config: GenerationConfig | RunConfig | ConfigResolution | None,
        progress: ProgressSink | None,
        *,
        api_bundle_hook: ApiBundleHook | None = None,
        process_selection: _ProcessSelection | None = None,
    ) -> None:
        self._resource_config = config
        self._configuration = _GenerationConfigProvenance.from_config(config)
        self._config = self._configuration.effective
        self._progress = progress
        self._api_bundle_hook = api_bundle_hook
        self._process_selection_override = process_selection
        self._prepared_pack_warning_emitted = False

    def plan(
        self,
        processes: ProcessSet,
        *,
        model: ModelSource | CompiledModel | None = None,
    ) -> GenerationPlan:
        source = model or self._configured_model_source()
        task_id = "generation-plan"
        if self._progress is not None:
            self._progress.emit(
                ProgressStart(
                    task_id,
                    "Planning processes",
                    total=len(processes.requests),
                )
            )
        try:
            self._validate_recurrence_request()
            self._validate_on_the_fly_process_selection()
            license_state = self._detect_symbolica_license()
            self._apply_symbolica_resource_policy(license_state)
            resolved_model = self._resolve_model_for_plan(source)
            self._require_eager_kernel_pack(resolved_model)
            self._apply_prepared_kernel_pack_policy(resolved_model)
            if resolved_model.model is None and (
                self._on_the_fly_execution_enabled
                or (
                    self._run_config is not None
                    and self._run_config.color.fft_basis.value == "adjoint"
                )
            ):
                compiled = resolved_model.compiled
                if compiled is None:  # pragma: no cover - model preflight invariant
                    raise GenerationError(
                        "on-the-fly or adjoint FFT planning requires a resolved model"
                    )
                from ..models.external import CompiledUFOModel

                resolved_model = replace(
                    resolved_model,
                    model=CompiledUFOModel(compiled),
                )
            expanded = self._expand_process_set(
                processes,
                resolved_model,
                PhaseHandle(task_id, self._progress, len(processes.requests)),
            )
            self._apply_symbolica_resource_policy(
                license_state,
                process_count=len(expanded),
            )
            coverage: list[dict[str, object]] = []
            if self._on_the_fly_execution_enabled:
                generation_model = resolved_model.model
                if generation_model is None:  # pragma: no cover - guarded above
                    raise GenerationError(
                        "on-the-fly planning requires a resolved model"
                    )
                model_inputs = self._recurrence_model_inputs(resolved_model)
                for index, entry in enumerate(expanded):
                    coverage.append(
                        self._plan_on_the_fly_process(
                            self._project_on_the_fly_process(
                                entry,
                                generation_model,
                                model_inputs,
                                index=index,
                            )
                        )
                    )
            else:
                for entry in expanded:
                    coverage.append(
                        self._plan_concrete_process(
                            entry.process_ir,
                            model=resolved_model.model,
                        )
                    )
            result = GenerationPlan(
                concrete_processes=tuple(entry.request for entry in expanded),
                estimated_coverage={
                    "model_kind": resolved_model.source.kind,
                    "color_accuracy": self._color_accuracy,
                    "process_count": len(expanded),
                    "alias_count": sum(len(entry.aliases) for entry in expanded),
                    "processes": tuple(coverage),
                },
                requested_settings=self._configuration.requested,
                effective_settings=self._config,
                adjustments=self._configuration.adjustments,
                unsupported_features=(),
            )
        except Exception as exc:
            if self._progress is not None:
                self._progress.emit(
                    ProgressEnd(task_id, success=False, message=str(exc))
                )
            if isinstance(exc, (GenerationError, ModelError)):
                raise
            raise GenerationError(str(exc)) from exc
        if self._progress is not None:
            self._progress.emit(ProgressEnd(task_id))
        return result

    def generate(
        self,
        processes: ProcessSet,
        output: os.PathLike[str] | str,
        *,
        model: ModelSource | CompiledModel | None = None,
        mode: str = "error",
    ) -> GenerationResult:
        if mode not in ("error", "append", "replace"):
            raise ValueError("generation mode must be 'error', 'append', or 'replace'")
        write_mode = cast(Literal["error", "append", "replace"], mode)
        reporter = GenerationPhaseReporter(self._progress)
        run_config = self._run_config
        execution_mode = (
            "compiled"
            if run_config is None
            else str(run_config.evaluator.execution_mode)
        )
        numerical_current_warning_required = False
        backend = "jit" if run_config is None else str(run_config.evaluator.backend)
        if run_config is None:
            optimization = "O2"
        elif backend == "jit":
            optimization = f"O{run_config.evaluator.jit.optimization_level}"
        else:
            optimization = str(run_config.evaluator.cpp.optimization)
        reporter.start_generation(
            total=(
                5
                if execution_mode in {"recurrence", "on-the-fly"}
                else 7
                if execution_mode == "eager"
                else 8
            )
            + int(self._numerical_process_grouping),
            details={
                "execution_mode": execution_mode,
                "backend": backend.upper(),
                "optimization": optimization.upper(),
            },
        )
        generation_started = time.perf_counter()
        try:
            self._validate_recurrence_request()
            self._validate_on_the_fly_process_selection()
            license_state = self._detect_symbolica_license()
            self._apply_symbolica_resource_policy(license_state)
            source = model or self._configured_model_source()
            with reporter.phase(
                "model-loading",
                "Loading and compiling model",
                total=1,
            ) as phase:
                resolved_model = self._resolve_model(source)
                artifact_model = self._artifact_model(resolved_model)
                self._require_eager_kernel_pack(resolved_model)
                resolved_model = self._index_eager_kernel_pack(resolved_model)
                self._apply_prepared_kernel_pack_policy(resolved_model)
                phase.update(1, message=artifact_model.name)
            generation_model = resolved_model.model
            if generation_model is None:
                raise GenerationError("generation model resolution produced no model")

            with reporter.phase(
                "process-expansion",
                "Expanding concrete processes",
                total=len(processes.requests),
            ) as phase:
                expanded = self._expand_process_set(processes, resolved_model, phase)
            self._apply_symbolica_resource_policy(
                license_state,
                process_count=len(expanded),
            )

            if self._on_the_fly_execution_enabled:
                return self._generate_on_the_fly_processes(
                    expanded,
                    output=Path(output),
                    write_mode=write_mode,
                    requested_processes=processes,
                    resolved_model=resolved_model,
                    artifact_model=artifact_model,
                    generation_model=generation_model,
                    reporter=reporter,
                    generation_started=generation_started,
                )

            if self._recurrence_execution_enabled:
                return self._generate_recurrence_processes(
                    expanded,
                    output=Path(output),
                    write_mode=write_mode,
                    requested_processes=processes,
                    resolved_model=resolved_model,
                    artifact_model=artifact_model,
                    generation_model=generation_model,
                    reporter=reporter,
                    generation_started=generation_started,
                )

            with TemporaryDirectory(prefix="pyamplicol-generation-") as temporary:
                temporary_root = Path(temporary)
                worker_count = self._process_worker_count(len(expanded))
                executor = (
                    ThreadPoolExecutor(
                        max_workers=worker_count,
                        thread_name_prefix="pyamplicol-generation",
                    )
                    if worker_count > 1
                    else None
                )
                try:
                    with reporter.phase(
                        "dag",
                        "Compiling process DAGs",
                        total=len(expanded),
                    ) as phase:
                        dag_results = _map_process_phase(
                            expanded,
                            lambda entry: self._compile_for_generation(
                                entry,
                                generation_model,
                                phase,
                            ),
                            executor=executor,
                            max_in_flight=worker_count,
                            phase_name="DAG compilation",
                            item_name=lambda entry: entry.request.name,
                        )
                        compiled, unsupported = _partition_model_supported_processes(
                            dag_results
                        )
                        _warn_unsupported_processes(unsupported)
                        del dag_results
                        del unsupported

                    indexed_compiled = tuple(enumerate(compiled))
                    with reporter.phase(
                        "warmup-filter",
                        (
                            "Applying structural reductions and preparing "
                            "validation points"
                        ),
                        total=len(compiled),
                    ) as phase:
                        prepared = _map_process_phase(
                            indexed_compiled,
                            lambda indexed: self._prepare_warmup_process(
                                indexed[1],
                                generation_model,
                                index=indexed[0],
                                phase=phase,
                            ),
                            executor=executor,
                            max_in_flight=worker_count,
                            phase_name="warmup and validation-point preparation",
                            item_name=lambda indexed: indexed[1].expanded.request.name,
                        )
                    numerical_current_warning_required = any(
                        _numerical_current_warning_required(
                            entry.filters.get("relation_discovery")
                        )
                        for entry in prepared
                    )

                    if self._eager_execution_enabled:
                        with reporter.phase(
                            "eager-lowering",
                            "Lowering prepared-kernel DAG invocation tables",
                            total=len(prepared),
                        ) as phase:
                            artifact_processes = _map_process_phase(
                                prepared,
                                lambda entry: self._construct_eager_artifact(
                                    entry,
                                    generation_model,
                                    resolved_model,
                                    temporary_root,
                                    phase,
                                ),
                                executor=executor,
                                max_in_flight=worker_count,
                                phase_name="eager DAG-table lowering",
                                item_name=lambda entry: entry.expanded.request.name,
                            )
                    else:
                        with reporter.phase(
                            "evaluator-construction",
                            "Constructing runtime evaluator schemas",
                            total=len(prepared),
                        ) as phase:
                            evaluators = _map_process_phase(
                                prepared,
                                lambda entry: self._construct_evaluator(
                                    entry,
                                    generation_model,
                                    phase,
                                ),
                                executor=executor,
                                max_in_flight=worker_count,
                                phase_name="runtime evaluator-schema construction",
                                item_name=lambda entry: entry.expanded.request.name,
                            )

                        jit_total = sum(
                            _runtime_stage_count(evaluator.runtime_schema)
                            + 1
                            + (
                                0
                                if evaluator.helicity_sum_runtime_schema is None
                                else _runtime_stage_count(
                                    evaluator.helicity_sum_runtime_schema
                                )
                                + 1
                            )
                            + sum(
                                _helicity_selector_lane_compile_task_count(lane)
                                for lane in evaluator.helicity_selector_lanes
                            )
                            + sum(
                                _runtime_stage_count(lane.runtime_schema) + 1
                                for lane in evaluator.color_selector_lanes
                            )
                            for evaluator in evaluators
                        )
                        with reporter.phase(
                            "jit",
                            "Compiling and materializing stage evaluators",
                            total=jit_total,
                        ) as phase:
                            # Symbolica expressions are created while resolving the
                            # model on this caller thread. Keep materialization on that
                            # same thread. The process-wide lock already made this phase
                            # serial; moving between worker threads can violate backend
                            # thread affinity.
                            artifact_processes = _map_process_phase(
                                evaluators,
                                lambda evaluator: self._materialize_evaluator(
                                    evaluator,
                                    generation_model,
                                    temporary_root,
                                    phase,
                                ),
                                executor=None,
                                max_in_flight=1,
                                phase_name="evaluator materialization",
                                item_name=lambda evaluator: (
                                    evaluator.compiled.expanded.request.name
                                ),
                            )
                finally:
                    if executor is not None:
                        executor.shutdown(wait=True, cancel_futures=True)

                with reporter.phase(
                    "artifact-writing",
                    "Writing schema-v3 artifact",
                    total=1,
                ) as phase:

                    def artifact_progress(event: dict[str, object]) -> None:
                        details = {
                            str(key): value
                            for key, value in event.items()
                            if isinstance(value, (str, int, float, bool, type(None)))
                        }
                        phase.update(
                            int(event.get("completed", phase.completed)),
                            total=(
                                None
                                if event.get("total") is None
                                else int(event["total"])
                            ),
                            message=str(event.get("step", "writing artifact")),
                            details=details,
                        )

                    write_result = self._write_artifact(
                        Path(output),
                        mode=write_mode,
                        source=resolved_model.source,
                        compiled_model=artifact_model,
                        configuration=self._configuration,
                        processes=artifact_processes,
                        timings=reporter.timings,
                        api_bundle_hook=self._api_bundle_hook,
                        progress_callback=(
                            artifact_progress if phase.sink is not None else None
                        ),
                        process_groups=_initial_process_groups(
                            artifact_processes,
                            (entry.expanded for entry in prepared),
                        ),
                    )
                    phase.update(
                        phase.total or phase.completed,
                        message=str(write_result.output),
                        details={
                            "step": "artifact ready",
                            "file_count": len(write_result.files),
                        },
                    )
                if numerical_current_warning_required:
                    _LOGGER.warning(
                        "%s [%s]",
                        NUMERICAL_CURRENT_RELATION_WARNING,
                        NUMERICAL_CURRENT_RELATION_WARNING_CODE,
                    )

                concrete_requests = tuple(entry.expanded.request for entry in prepared)
                validation_points_by_process = {
                    process.expanded.request.name: process.validation_points
                    for process in prepared
                }
                expected_process_ids = tuple(
                    process.process_id for process in artifact_processes
                )
                del compiled
                del indexed_compiled
                del prepared
                del artifact_processes
                gc.collect()
                # After releasing the generation state: analysis loads runtimes.
                self._analyze_process_groups(write_result.output, reporter)

            with reporter.phase(
                "validation",
                "Validating generated artifact",
                total=1,
            ) as phase:
                validation = self._generation_config.validation
                if validation.post_build_validation:
                    self._validate_generated_artifact(
                        write_result.output,
                        expected_process_ids,
                        validation_points=validation_points_by_process,
                        expected_api_bundle_path=write_result.api_bundle_path,
                        progress=phase,
                    )
                    message = (
                        f"schema and {validation.samples} numerical samples"
                        if validation.enabled
                        else "schema, references, hashes, and target"
                    )
                else:
                    message = "post-build validation disabled"
                phase.update(1, message=message)

            reporter.timings["total"] = time.perf_counter() - generation_started
        except KeyboardInterrupt:
            reporter.finish_generation(success=False, message="interrupted")
            raise
        except Exception as exc:
            reporter.finish_generation(success=False, message=str(exc))
            if isinstance(exc, PyAmpliColError):
                raise
            raise GenerationError(str(exc)) from exc

        reporter.finish_generation()
        return GenerationResult(
            output=write_result.output,
            processes=ProcessSet(
                requests=concrete_requests,
                aliases=processes.aliases,
            ),
            mode=write_mode,
            files=write_result.files,
        )

    def _generate_on_the_fly_processes(
        self,
        expanded: tuple[_ExpandedProcess, ...],
        *,
        output: Path,
        write_mode: Literal["error", "append", "replace"],
        requested_processes: ProcessSet,
        resolved_model: _ResolvedModel,
        artifact_model: _CompiledModelPayload,
        generation_model: Model,
        reporter: GenerationPhaseReporter,
        generation_started: float,
    ) -> GenerationResult:
        """Build source-only LC seeds before any color plan or process DAG."""

        inputs_started = time.perf_counter()
        model_inputs = self._recurrence_model_inputs(resolved_model)
        reporter.timings["on-the-fly-model-input-preparation"] = (
            time.perf_counter() - inputs_started
        )
        with TemporaryDirectory(prefix="pyamplicol-on-the-fly-generation-") as root:
            temporary_root = Path(root)
            indexed = tuple(enumerate(expanded))
            with reporter.phase(
                "on-the-fly-construction",
                "Constructing compact on-the-fly process seeds",
                total=len(indexed),
            ) as phase:
                projected = _map_process_phase(
                    indexed,
                    lambda item: self._project_on_the_fly_process(
                        item[1],
                        generation_model,
                        model_inputs,
                        index=item[0],
                    ),
                    executor=None,
                    max_in_flight=1,
                    phase_name="on-the-fly source projection",
                    item_name=lambda item: item[1].request.name,
                )
                seed_payloads = _invoke_rust_on_the_fly_seed_batch_builder_v1(
                    tuple(item.projection.seed.to_json_bytes() for item in projected),
                    model_inputs.template_catalog_json,
                    model_inputs.direct_template_catalog_json,
                    model_inputs.prepared_kernel_pack_digest,
                    process_identities=tuple(
                        item.expanded.request.name for item in projected
                    ),
                )
                packaged = tuple(zip(projected, seed_payloads, strict=True))
                generated = _map_process_phase(
                    packaged,
                    lambda item: self._construct_on_the_fly_artifact(
                        item[0],
                        item[1],
                        generation_model,
                        model_inputs,
                        temporary_root,
                        phase=phase,
                    ),
                    executor=None,
                    max_in_flight=1,
                    phase_name="on-the-fly artifact packaging",
                    item_name=lambda item: item[0].expanded.request.name,
                )

            artifact_processes = tuple(item.artifact for item in generated)
            with reporter.phase(
                "artifact-writing",
                "Writing schema-v3 artifact",
                total=1,
            ) as phase:

                def artifact_progress(event: dict[str, object]) -> None:
                    details = {
                        str(key): value
                        for key, value in event.items()
                        if isinstance(value, (str, int, float, bool, type(None)))
                    }
                    phase.update(
                        int(event.get("completed", phase.completed)),
                        total=(
                            None if event.get("total") is None else int(event["total"])
                        ),
                        message=str(event.get("step", "writing artifact")),
                        details=details,
                    )

                write_result = write_schema_v3_artifact(
                    output,
                    mode=write_mode,
                    source=resolved_model.source,
                    compiled_model=artifact_model,
                    configuration=self._configuration,
                    processes=artifact_processes,
                    timings=reporter.timings,
                    api_bundle_hook=self._api_bundle_hook,
                    progress_callback=(
                        artifact_progress if phase.sink is not None else None
                    ),
                    process_groups=_initial_process_groups(
                        artifact_processes,
                        (item.expanded for item in generated),
                    ),
                )
                phase.update(
                    phase.total or phase.completed,
                    message=str(write_result.output),
                    details={
                        "step": "artifact ready",
                        "file_count": len(write_result.files),
                    },
                )
            validation_points_by_process = {
                item.expanded.request.name: item.validation_points for item in generated
            }
            validation_selectors_by_process = {
                item.expanded.request.name: item.validation_selectors
                for item in generated
            }
            expected_process_ids = tuple(
                process.process_id for process in artifact_processes
            )
            concrete_requests = tuple(item.expanded.request for item in generated)
            del artifact_processes
            del generated
            gc.collect()
            self._analyze_process_groups(write_result.output, reporter)

        with reporter.phase(
            "validation",
            "Validating generated artifact",
            total=1,
        ) as phase:
            validation = self._generation_config.validation
            if validation.post_build_validation:
                numerical_sample_count = sum(
                    1
                    for points in validation_points_by_process.values()
                    if any(point.available for point in points)
                )
                self._validate_generated_artifact(
                    write_result.output,
                    expected_process_ids,
                    validation_points=validation_points_by_process,
                    compact_validation_selectors=(validation_selectors_by_process),
                    expected_api_bundle_path=write_result.api_bundle_path,
                    progress=phase,
                )
                if not validation.enabled:
                    message = "compact schema, seed, selectors, and prepared model"
                elif numerical_sample_count == 0:
                    message = "compact schema; zero numerical samples available"
                else:
                    suffix = "" if numerical_sample_count == 1 else "s"
                    message = (
                        f"compact schema and {numerical_sample_count} "
                        f"selector-bounded numerical sample{suffix}"
                    )
            else:
                message = "post-build validation disabled"
            phase.update(1, message=message)

        reporter.timings["total"] = time.perf_counter() - generation_started
        reporter.finish_generation()
        return GenerationResult(
            output=write_result.output,
            processes=ProcessSet(
                requests=concrete_requests,
                aliases=requested_processes.aliases,
            ),
            mode=write_mode,
            files=write_result.files,
        )

    def _project_on_the_fly_process(
        self,
        expanded: _ExpandedProcess,
        model: Model,
        model_inputs: _RecurrenceModelInputs,
        *,
        index: int,
    ) -> _ProjectedOnTheFlyProcess:
        run = self._run_config
        if run is None:  # pragma: no cover - guarded by execution mode
            raise GenerationError("on-the-fly generation requires RunConfig")
        self._validate_on_the_fly_process_selection()
        coupling_policy = str(run.process.coupling_order_policy)
        explicit_limits = self._coupling_order_limits
        projection = project_on_the_fly_process_seed_v1(
            expanded.process_ir,
            model_inputs.catalog,
            model,
            coupling_order_policy=cast(
                Literal["minimal", "explicit"],
                coupling_policy,
            ),
            coupling_order_limits=explicit_limits,
            reference_color_order=self._process_selection.reference_color_order,
            color_basis=self._fft_color_basis(expanded.process_ir, model),
        )
        return _ProjectedOnTheFlyProcess(
            index=index,
            expanded=expanded,
            projection=projection,
        )

    def _validate_on_the_fly_process_selection(self) -> None:
        if not self._on_the_fly_execution_enabled:
            return
        selection = self._process_selection
        incompatible = []
        if selection.max_color_sectors is not None:
            incompatible.append("process.max_color_sectors")
        if selection.selected_color_sector_ids is not None:
            incompatible.append("process.selected_color_sector_ids")
        if selection.selected_source_helicities is not None:
            incompatible.append("process.selected_source_helicities")
        if incompatible:
            raise GenerationError(
                "on-the-fly artifacts retain complete runtime selector coverage; "
                "remove " + ", ".join(incompatible)
            )

    def _recurrence_helicity_selector_plan_enabled(self) -> bool:
        """Return whether generation can retain the complete selector plan."""

        selection = self._process_selection
        return self._color_accuracy in {"nlc", "full"} and all(
            value is None
            for value in (
                selection.max_color_sectors,
                selection.selected_color_sector_ids,
                selection.selected_source_helicities,
            )
        )

    def _construct_on_the_fly_artifact(
        self,
        projected: _ProjectedOnTheFlyProcess,
        seed_payload: bytes,
        model: Model,
        model_inputs: _RecurrenceModelInputs,
        temporary_root: Path,
        *,
        phase: PhaseHandle,
    ) -> _GeneratedOnTheFlyProcess:
        expanded = projected.expanded
        projection = projected.projection
        index = projected.index
        process_id = expanded.request.name
        run = self._run_config
        if run is None:  # pragma: no cover - guarded by execution mode
            raise GenerationError("on-the-fly generation requires RunConfig")
        query_construction_threads = run.evaluator.optimization.cores
        if (
            isinstance(query_construction_threads, bool)
            or not isinstance(query_construction_threads, int)
            or query_construction_threads < 1
        ):
            raise GenerationError(
                "on-the-fly generation requires resolved positive "
                "evaluator.optimization.cores"
            )
        selection = self._process_selection
        coupling_policy = str(run.process.coupling_order_policy)
        explicit_limits = self._coupling_order_limits
        process_seed_identity = _invoke_rust_on_the_fly_seed_inspector_v1(seed_payload)
        _bind_on_the_fly_process_seed_identity_v1(
            process_seed_identity,
            projection.seed,
        )
        color_accuracy = str(expanded.process_ir.color_accuracy)
        contracted_color = None
        if color_accuracy in {"nlc", "full"}:
            if projection.color_plan is None:
                raise GenerationError(
                    "on-the-fly contracted-color projection has no color plan"
                )
            try:
                contracted_color = _build_on_the_fly_contracted_color_payload_v1(
                    projection.color_plan,
                    contraction=run.color.contraction.value,
                )
            except ValueError as exc:
                raise GenerationError(
                    f"process {process_id!r} on-the-fly color contraction failed: {exc}"
                ) from exc
        elif projection.color_plan is not None:
            raise GenerationError("LC on-the-fly projection carries a color plan")
        validation_selectors = _on_the_fly_validation_selectors_v1(
            projection.seed,
            color_accuracy=color_accuracy,
        )
        selector_census = _on_the_fly_selector_census_v1(
            projection.seed,
            expanded.process_ir,
        )
        runtime_path = (
            temporary_root
            / "on-the-fly-runtimes"
            / process_id
            / "on-the-fly-runtime.pacbin"
        )
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_index = write_pacbin_atomic(
            runtime_path,
            (
                PacbinMemberSource(
                    "on-the-fly/process-seed-v1.bin",
                    PacbinMemberKind.ON_THE_FLY_PROCESS_SEED,
                    io.BytesIO(seed_payload),
                ),
            ),
        )
        runtime_metadata = {
            **build_on_the_fly_runtime_metadata(
                projection.seed.external_sources,
                projection.seed.parameter_projection,
                model_inputs.catalog,
                model,
                projection.runtime_normalization,
            ),
            "process_seed_identity": process_seed_identity,
        }
        public_metadata = build_on_the_fly_public_metadata(
            expanded.process_ir,
            model_inputs.catalog,
            process_id=process_id,
        )
        umami_physics = None
        umami_sectors = ()
        if run.generation.emit_api_bundle:
            from ..artifacts.umami_otf import build_umami_on_the_fly_physics

            umami_physics, umami_sectors = build_umami_on_the_fly_physics(
                public_metadata, expanded.process_ir, projection.seed
            )
        validation = self._generation_config.validation
        sample_count = (
            validation.samples
            if validation.enabled and validation.post_build_validation
            else 1
        )
        points = tuple(
            build_process_validation_point(
                expanded.process_ir,
                model,
                process_id=process_id,
                seed=validation.seed + index * sample_count + sample_index,
            )
            for sample_index in range(sample_count)
        )
        phase.advance(
            message=process_id,
            details={
                "process": process_id,
                "step": "compact on-the-fly seed ready",
                "seed_size_bytes": len(seed_payload),
            },
        )
        artifact = OnTheFlyProcessArtifact(
            process_id=process_id,
            expression=expanded.process_ir.process,
            color_accuracy=str(expanded.process_ir.color_accuracy),
            external_pdgs=tuple(int(leg.pdg) for leg in expanded.process_ir.legs),
            aliases=expanded.aliases,
            physics=public_metadata,
            umami_physics=umami_physics,
            umami_color_sectors=umami_sectors,
            runtime_path=runtime_path,
            runtime_size_bytes=runtime_index.file_size,
            runtime_sha256=_file_sha256(runtime_path),
            runtime_member_count=len(runtime_index.members),
            runtime_unpacked_size_bytes=sum(
                member.length for member in runtime_index.members
            ),
            runtime_index_sha256=runtime_index.index_sha256,
            referenced_kernel_ids=frozenset(
                kernel.kernel_id for kernel in model_inputs.bundle.kernel_pack.kernels
            ),
            runtime_metadata=runtime_metadata,
            selector_policy={
                **(
                    {"color_basis": projection.color_plan.basis}
                    if projection.color_plan is not None
                    and projection.color_plan.basis != "trace"
                    else {}
                ),
                "color_coverage": (
                    "complete" if contracted_color is None else "contracted"
                ),
                "reference_color_word": (
                    None
                    if selection.reference_color_order is None
                    else list(
                        projection.color_plan.sectors[0].trace_labels
                        if projection.color_plan is not None
                        and projection.color_plan.basis == "adjoint"
                        else selection.reference_color_order
                    )
                ),
                "trace_reflections_folded": bool(
                    contracted_color is None
                    and model.lc_trace_reflection_equivalence_is_proven(
                        expanded.process_ir
                    )
                ),
                "selector_census": selector_census,
            },
            color_contraction_payload=(
                None if contracted_color is None else contracted_color.payload
            ),
            color_contraction_summary=(
                None if contracted_color is None else contracted_color.summary
            ),
            point_tile_size=run.evaluator.recurrence.point_tile_size,
            query_construction_threads=query_construction_threads,
            validation_point=points[0],
            generation_filters={
                **self._fft_basis_metadata(projection.color_plan),
                "on_the_fly": {
                    "coupling_order_policy": coupling_policy,
                    "coupling_order_limits": dict(explicit_limits),
                    "selector_coverage": (
                        "complete" if contracted_color is None else "contracted"
                    ),
                },
            },
        )
        return _GeneratedOnTheFlyProcess(
            expanded=expanded,
            artifact=artifact,
            validation_points=points,
            validation_selectors=validation_selectors,
        )

    def _generate_recurrence_processes(
        self,
        expanded: tuple[_ExpandedProcess, ...],
        *,
        output: Path,
        write_mode: Literal["error", "append", "replace"],
        requested_processes: ProcessSet,
        resolved_model: _ResolvedModel,
        artifact_model: _CompiledModelPayload,
        generation_model: Model,
        reporter: GenerationPhaseReporter,
        generation_started: float,
    ) -> GenerationResult:
        """Build and publish compact recurrence schedules before GenericDAG."""

        expanded = self._drop_processes_without_amplitudes(expanded, generation_model)
        model_inputs_started = time.perf_counter()
        model_inputs = self._recurrence_model_inputs(resolved_model)
        reporter.timings["recurrence-model-input-preparation"] = (
            time.perf_counter() - model_inputs_started
        )
        with TemporaryDirectory(
            prefix="pyamplicol-recurrence-generation-"
        ) as temporary:
            temporary_root = Path(temporary)
            worker_count = self._process_worker_count(len(expanded))
            executor = (
                ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="pyamplicol-recurrence-generation",
                )
                if worker_count > 1
                else None
            )
            try:
                indexed = tuple(enumerate(expanded))
                lowering_cache = RecurrenceScheduleLoweringCache[
                    _RustRecurrenceLoweringOutput
                ]()
                with reporter.phase(
                    "recurrence-construction",
                    "Constructing compact process recurrences",
                    total=len(expanded),
                ) as phase:
                    generated = _map_process_phase(
                        indexed,
                        lambda item: self._construct_recurrence_artifact(
                            item[1],
                            generation_model,
                            model_inputs,
                            temporary_root,
                            index=item[0],
                            phase=phase,
                            lowering_cache=lowering_cache,
                            generation_timings=reporter.timings,
                        ),
                        executor=executor,
                        max_in_flight=worker_count,
                        phase_name="compact recurrence construction",
                        item_name=lambda item: item[1].request.name,
                    )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

            artifact_processes = tuple(item.artifact for item in generated)
            recurrence_warnings: list[tuple[str, str]] = []
            for process in artifact_processes:
                relation_report = process.generation_filters.get("relation_discovery")
                if not isinstance(relation_report, Mapping):
                    continue
                warning = relation_report.get("warning")
                if (
                    not isinstance(warning, Mapping)
                    or warning.get("required") is not True
                ):
                    continue
                code = warning.get("code")
                message = warning.get("message")
                if not isinstance(code, str) or not isinstance(message, str):
                    raise GenerationError(
                        "recurrence numerical warning metadata is malformed"
                    )
                item = (code, message)
                if item not in recurrence_warnings:
                    recurrence_warnings.append(item)
            with reporter.phase(
                "artifact-writing",
                "Writing schema-v3 artifact",
                total=1,
            ) as phase:

                def artifact_progress(event: dict[str, object]) -> None:
                    details = {
                        str(key): value
                        for key, value in event.items()
                        if isinstance(value, (str, int, float, bool, type(None)))
                    }
                    phase.update(
                        int(event.get("completed", phase.completed)),
                        total=(
                            None if event.get("total") is None else int(event["total"])
                        ),
                        message=str(event.get("step", "writing artifact")),
                        details=details,
                    )

                write_result = write_schema_v3_artifact(
                    output,
                    mode=write_mode,
                    source=resolved_model.source,
                    compiled_model=artifact_model,
                    configuration=self._configuration,
                    processes=artifact_processes,
                    timings=reporter.timings,
                    api_bundle_hook=self._api_bundle_hook,
                    progress_callback=(
                        artifact_progress if phase.sink is not None else None
                    ),
                    process_groups=_initial_process_groups(
                        artifact_processes,
                        (item.expanded for item in generated),
                    ),
                )
                phase.update(
                    phase.total or phase.completed,
                    message=str(write_result.output),
                    details={
                        "step": "artifact ready",
                        "file_count": len(write_result.files),
                    },
                )
            for warning_code, warning_message in recurrence_warnings:
                _LOGGER.warning(
                    "%s [%s]",
                    warning_message,
                    warning_code,
                )

            validation_points_by_process = {
                item.expanded.request.name: item.validation_points for item in generated
            }
            expected_process_ids = tuple(
                process.process_id for process in artifact_processes
            )
            concrete_requests = tuple(item.expanded.request for item in generated)
            del artifact_processes
            del generated
            gc.collect()
            self._analyze_process_groups(write_result.output, reporter)

        with reporter.phase(
            "validation",
            "Validating generated artifact",
            total=1,
        ) as phase:
            validation = self._generation_config.validation
            if validation.post_build_validation:
                self._validate_generated_artifact(
                    write_result.output,
                    expected_process_ids,
                    validation_points=validation_points_by_process,
                    expected_api_bundle_path=write_result.api_bundle_path,
                    progress=phase,
                )
                message = (
                    f"schema and {validation.samples} numerical samples"
                    if validation.enabled
                    else "schema, references, hashes, and target"
                )
            else:
                message = "post-build validation disabled"
            phase.update(1, message=message)

        reporter.timings["total"] = time.perf_counter() - generation_started
        reporter.finish_generation()
        return GenerationResult(
            output=write_result.output,
            processes=ProcessSet(
                requests=concrete_requests,
                aliases=requested_processes.aliases,
            ),
            mode=write_mode,
            files=write_result.files,
        )

    def _artifact_model(self, resolved: _ResolvedModel) -> _CompiledModelPayload:
        if resolved.compiled is not None:
            return resolved.compiled
        from ..models.loading import compile_model_source

        return compile_model_source("BUILTIN_SM", use_cache=False)

    def _expand_process_set(
        self,
        processes: ProcessSet,
        resolved: _ResolvedModel,
        phase: PhaseHandle,
    ) -> tuple[_ExpandedProcess, ...]:
        aliases_by_target: dict[str, list[ProcessAlias]] = {}
        for alias in processes.aliases:
            aliases_by_target.setdefault(alias.process_name, []).append(alias)
        result: list[_ExpandedProcess] = []
        names: set[str] = set()
        for request_index, request in enumerate(processes.requests, start=1):
            expanded = self._expand_request(request, resolved)
            request_aliases = aliases_by_target.get(request.name, [])
            if request_aliases and len(expanded) != 1:
                raise GenerationError(
                    f"process aliases for {request.name!r} are ambiguous after "
                    f"expansion into {len(expanded)} concrete processes"
                )
            side_permutation_classes: set[tuple[tuple[str, ...], tuple[str, ...]]] = (
                set()
            )
            for concrete_index, process_ir in enumerate(expanded, start=1):
                permutation_class = _process_side_permutation_class(process_ir)
                if permutation_class in side_permutation_classes:
                    continue
                side_permutation_classes.add(permutation_class)
                name = _expanded_name(
                    request.name,
                    concrete_index,
                    len(expanded),
                )
                if name in names:
                    raise GenerationError(f"expanded process ID is not unique: {name}")
                names.add(name)
                concrete = ProcessRequest.parse(process_ir.process, name=name)
                alias_records: list[Mapping[str, object]] = []
                for alias in request_aliases:
                    permutation = tuple(alias.particle_permutation) or tuple(
                        range(len(process_ir.legs))
                    )
                    if len(permutation) != len(process_ir.legs):
                        raise GenerationError(
                            f"alias {alias.name!r} permutation has length "
                            f"{len(permutation)}, expected {len(process_ir.legs)}"
                        )
                    if any(
                        process_ir.legs[representative_index].side
                        != process_ir.legs[public_index].side
                        for representative_index, public_index in enumerate(permutation)
                    ):
                        raise GenerationError(
                            f"alias {alias.name!r} may only permute particles "
                            "within the incoming and outgoing sides"
                        )
                    alias_expression, alias_pdgs = _permuted_process_identity(
                        process_ir,
                        permutation,
                    )
                    alias_records.append(
                        {
                            "id": alias.name,
                            "expression": alias_expression,
                            "external_pdgs": list(alias_pdgs),
                            "external_permutation": list(permutation),
                        }
                    )
                result.append(
                    _ExpandedProcess(
                        request=concrete,
                        process_ir=process_ir,
                        aliases=tuple(alias_records),
                        source_expansion_size=len(expanded),
                        source_request=request,
                    )
                )
            phase.update(request_index, message=request.name)
        if self._flavor_blind:
            # Names are assigned above, so process IDs do not depend on the flag.
            table = self._grouping_particle_table(resolved)
            result = _collapse_flavor_blind_classes(
                result,
                table,
                consider_charge_conjugation=self._consider_charge_conjugation,
                relation_warning=self._flavor_blind_relation_warning(resolved, table),
            )
        return tuple(result)

    def _grouping_particle_table(
        self,
        resolved: _ResolvedModel,
    ) -> dict[int, GroupingParticle]:
        """Particle types for process grouping, from the compiled model."""

        model = self._artifact_model(resolved).ir
        return particle_table(
            model.particles,
            {parameter.name: parameter for parameter in model.parameters},
        )

    def _flavor_blind_relation_warning(
        self,
        resolved: _ResolvedModel,
        table: Mapping[int, GroupingParticle],
    ) -> _RelationWarning:
        """Check relations against the vertices each representative is built from."""

        from .process_grouping_symmetry import VertexSymmetryTable

        model = resolved.model
        compiled = resolved.compiled
        if compiled is not None and resolved.use_compiled_process_catalog:
            symmetry = VertexSymmetryTable.from_compiled_ir(compiled.ir)
        else:
            model = model or _builtin_sm_model()
            symmetry = VertexSymmetryTable.from_builtin_model(model)
        explicit = self._coupling_order_limits
        run = self._run_config
        minimal = run is not None and run.process.coupling_order_policy == "minimal"
        limits_by_representative: dict[str, Mapping[str, int]] = {}

        def warning(
            member: GroupingProcess,
            representative: GroupingProcess,
            mapping: LegMapping,
            representative_ir: CanonicalProcessIR,
        ) -> str | None:
            nonlocal model
            limits = limits_by_representative.get(representative.process_id)
            if limits is None:
                limits = explicit
                if minimal:
                    if model is None:
                        from ..models.external import CompiledUFOModel

                        model = CompiledUFOModel(cast(_CompiledModelPayload, compiled))
                    limits = (
                        infer_minimal_coupling_order_limits(
                            representative_ir,
                            model=model,
                            max_coupling_orders=explicit or None,
                        )
                        or explicit
                    )
                limits_by_representative[representative.process_id] = limits
            return symmetry.relation_warning(
                member, representative, mapping, table, limits=limits
            )

        return warning

    def _compile_for_generation(
        self,
        expanded: _ExpandedProcess,
        model: Model,
        phase: PhaseHandle,
    ) -> _DagProcess | _UnsupportedProcess:
        process_name = expanded.request.name
        with phase.child(
            process_name,
            f"{process_name}: DAG construction",
            details={
                "process": process_name,
                "step": "colour-plan",
            },
        ) as task:

            def report(details: Mapping[str, str | int]) -> None:
                payload = {"process": process_name, **details}
                completed = task.completed
                total: int | None = None
                if "mask_index" in details and "mask_total" in details:
                    completed = int(details["mask_index"])
                    total = int(details["mask_total"])
                task.update(
                    completed,
                    total=total,
                    message=str(details.get("step", "DAG construction")),
                    details=payload,
                )

            try:
                dag, coverage = self._compile_concrete_process(
                    expanded.process_ir,
                    model,
                    progress_callback=report if task.sink is not None else None,
                )
            except _NoModelSupportedAmplitudes as exc:
                if expanded.source_expansion_size <= 1:
                    raise
                result: _DagProcess | _UnsupportedProcess = _UnsupportedProcess(
                    expanded=expanded,
                    reason=str(exc),
                )
                step = "unsupported tree-level subprocess"
                task.update(
                    task.completed,
                    message=step,
                    details={
                        "process": process_name,
                        "step": step,
                        "amplitude_count": 0,
                    },
                )
            else:
                result = _DagProcess(
                    expanded=expanded,
                    dag=dag,
                    coverage=coverage,
                )
                step = "DAG complete"
                task.update(
                    task.completed,
                    message=step,
                    details={
                        "process": process_name,
                        "step": step,
                        "current_count": len(dag.currents),
                        "interaction_count": len(dag.interactions),
                        "amplitude_count": len(dag.amplitude_roots),
                    },
                )
        phase.advance(
            message=process_name,
            details={"process": process_name, "step": step},
        )
        return result

    def _prepare_warmup_process(
        self,
        process: _DagProcess,
        model: Model,
        *,
        index: int,
        phase: PhaseHandle,
    ) -> _CompiledProcess:
        process_name = process.expanded.request.name
        with phase.child(
            process_name,
            f"{process_name}: structural reduction",
            total=2,
            details={"process": process_name, "step": "structural reduction"},
        ) as task:
            result = self._prepare_warmup_process_inner(
                process,
                model,
                index=index,
                progress=task,
            )
        phase.advance(
            message=process_name,
            details={"process": process_name, "step": "validation points ready"},
        )
        return result

    def _write_artifact(self, *args: Any, **kwargs: Any) -> ArtifactWriteResult:
        """Keep optional payloads inside the existing publication transaction."""
        return write_schema_v3_artifact(*args, **kwargs)

    @property
    def _numerical_process_grouping(self) -> bool:
        run = self._run_config
        return run is not None and run.process.analyze_process_groups == "numerical"

    def _analyze_process_groups(
        self,
        artifact: Path,
        reporter: GenerationPhaseReporter,
    ) -> None:
        """Merge numerically found process groups into the written artifact."""

        run = self._run_config
        if run is None or not self._numerical_process_grouping:
            return
        from .process_grouping_numerical import update_artifact_process_groups

        with reporter.phase(
            "process-grouping",
            "Analyzing process groups numerically",
            total=1,
        ) as phase:
            update_artifact_process_groups(
                artifact,
                analysis=str(run.process.analyze_process_groups),
                only_unit_multiplier=run.process.only_unit_multiplier,
                consider_charge_conjugation=run.process.consider_charge_conjugation,
            )
            phase.update(1, message="process-groups.json updated")

    def _prepare_warmup_process_inner(
        self,
        process: _DagProcess,
        model: Model,
        *,
        index: int,
        progress: PhaseHandle,
    ) -> _CompiledProcess:
        process_name = process.expanded.request.name
        progress.update(
            0,
            message="structural reduction",
            details={
                "process": process_name,
                "step": "structural reduction",
                "current_count": len(process.dag.currents),
                "interaction_count": len(process.dag.interactions),
                "amplitude_count": len(process.dag.amplitude_roots),
            },
        )
        parity_preweighted = any(
            float(root.helicity_weight) > 1.0 for root in process.dag.amplitude_roots
        )
        reduced = (
            prune_dag_to_amplitude_roots(process.dag)
            if parity_preweighted
            else prune_global_helicity_flip_equivalent_roots(process.dag, model)
        )
        if reduced.process.color_accuracy == "lc":
            reduced, dynamic_color_projection_certificate = (
                project_rectangular_dynamic_color_classes(
                    reduced,
                    model,
                    source_revision=active_source_revision(),
                )
            )
            dynamic_color_projection = (
                dynamic_color_projection_certificate.to_json_dict()
            )
        else:
            dynamic_color_projection = dynamic_color_projection_not_applicable_report(
                reduced
            )
        helicity_sum_dag: GenericDAG | None = None
        helicity_selector_union_dag: GenericDAG | None = None
        all_flow_union = self._all_flow_union_enabled
        helicity_recurrence = build_helicity_recurrence_plan(reduced, model)
        if helicity_recurrence is not None:
            if self._eager_execution_enabled:
                # Eager invocation tables already carry selector domains. Keep
                # the exact recurrence proof, but do not serialize the
                # compiled lane's evaluator-chunk schedules into an eager
                # artifact.
                reduced = replace(
                    reduced,
                    helicity_recurrence=helicity_recurrence,
                    helicity_materialization=None,
                )
            else:
                # Topology replay keeps its existing fused helicity-sum lane.
                # The all-flow union is itself the primary reusable lane, so
                # compiling that second payload would duplicate the expensive
                # cross-flow recurrence without helping its target workload.
                if not all_flow_union:
                    helicity_sum_dag = replace(
                        reduced,
                        helicity_recurrence=None,
                        helicity_materialization=None,
                    )
                materialization = materialize_helicity_recurrence(
                    reduced,
                    helicity_recurrence,
                )
                reduced = replace(
                    materialization.dag,
                    helicity_recurrence=helicity_recurrence,
                    helicity_materialization=materialization,
                )
                if not all_flow_union:
                    helicity_selector_union_dag = (
                        self._compile_complete_color_helicity_selector_union(
                            process,
                            model,
                        )
                    )
                if helicity_selector_union_dag is not None:
                    _validate_matching_helicity_selector_domains(
                        reduced,
                        helicity_selector_union_dag,
                    )
        relation_config = self._generation_config.relation_discovery
        relation_mode = str(relation_config.mode)
        relation_execution_mode = cast(
            Literal["compiled", "eager"],
            "eager" if self._eager_execution_enabled else "compiled",
        )
        relation_lanes: dict[str, object] = {}
        certified_relation_count = 0
        applied_relation_count = 0
        relation_warning_required = False

        def apply_numerical_relations(
            lane: str,
            lane_dag: GenericDAG,
        ) -> GenericDAG:
            nonlocal applied_relation_count
            nonlocal certified_relation_count
            nonlocal relation_warning_required
            if relation_mode == "off":
                relation_lanes[lane] = generic_dag_numerical_current_opt_out_report(
                    lane_dag,
                    execution_mode=relation_execution_mode,
                )
                return lane_dag
            if all_flow_union or lane == "helicity-selector-union":
                application_relation_kinds: tuple[
                    Literal["equal", "opposite", "zero"], ...
                ] = ()
                application_scope_reason = "multi-helicity-all-flow-selector-domain"
            elif self._color_accuracy in {"nlc", "full"}:
                application_relation_kinds = ("zero", "equal")
                application_scope_reason = "contracted-binary64-opposite-contained"
            else:
                application_relation_kinds = ("equal", "opposite", "zero")
                application_scope_reason = "complete-generic-dag"
            try:
                # Symbolica evaluator construction is process-global. Keep
                # bounded high-precision captures serial even when independent
                # process DAGs are prepared by worker threads.
                with _SYMBOLICA_MATERIALIZATION_LOCK:
                    result = run_generic_dag_numerical_current_warmup(
                        lane_dag,
                        model,
                        process_id=process_name,
                        mode=cast(
                            Literal["diagnostic", "certified-reuse"],
                            relation_mode,
                        ),
                        execution_mode=relation_execution_mode,
                        precision_digits=relation_config.precision_digits,
                        probe_count=relation_config.probe_count,
                        verification_probe_count=(
                            relation_config.verification_probe_count
                        ),
                        relative_tolerance=relation_config.relative_tolerance,
                        absolute_tolerance=relation_config.absolute_tolerance,
                        seed=relation_config.seed,
                        application_relation_kinds=(application_relation_kinds),
                        application_scope_reason=application_scope_reason,
                    )
            except ValueError as exc:
                raise GenerationError(
                    f"process {process_name!r} numerical current warm-up "
                    f"failed closed in lane {lane!r}: {exc}"
                ) from exc
            certified_relation_count += len(result.discovery.certificates)
            applied_relation_count += result.application.report.applied_relation_count
            payload = result.to_json_dict()
            if result.warning_required:
                warning = payload.get("warning")
                if (
                    not isinstance(warning, Mapping)
                    or warning.get("code") != NUMERICAL_CURRENT_RELATION_WARNING_CODE
                    or warning.get("message") != NUMERICAL_CURRENT_RELATION_WARNING
                ):
                    raise GenerationError(
                        "numerical current warm-up requested a warning "
                        "without its stable warning contract"
                    )
                relation_warning_required = True
            relation_lanes[lane] = payload
            return result.dag

        progress.update(
            1,
            message="authenticated numerical current warm-up",
            details={
                "process": process_name,
                "step": "authenticated numerical current warm-up",
                "mode": relation_mode,
            },
        )
        reduced = apply_numerical_relations("primary", reduced)
        if helicity_sum_dag is not None:
            helicity_sum_dag = apply_numerical_relations(
                "helicity-sum",
                helicity_sum_dag,
            )
        if helicity_selector_union_dag is not None:
            helicity_selector_union_dag = apply_numerical_relations(
                "helicity-selector-union",
                helicity_selector_union_dag,
            )
        warning_required = relation_warning_required
        relation_discovery_payload: dict[str, object] = {
            "schema_version": 1,
            "abi": "pyamplicol-artifact-numerical-current-reuse-v1",
            "requested_mode": relation_mode,
            "effective_mode": relation_mode,
            "effective_reuse_state": (
                "disabled" if relation_mode == "off" else "enabled"
            ),
            "execution_mode": relation_execution_mode,
            "lane_count": len(relation_lanes),
            "lanes": relation_lanes,
            "certified_relation_count": certified_relation_count,
            "applied_relation_count": applied_relation_count,
            "relation_correctness": numerical_relation_correctness_payload(
                applied_relation_count
            ),
            "structural_exact_discovery": None,
            "opt_out_flag": "--no-numerical-current-reuse",
            "warning": (
                {
                    "required": True,
                    "emit": "once-per-generated-artifact",
                    "code": NUMERICAL_CURRENT_RELATION_WARNING_CODE,
                    "message": NUMERICAL_CURRENT_RELATION_WARNING,
                }
                if warning_required
                else {
                    "required": False,
                    "emit": "never",
                    "code": None,
                    "message": None,
                }
            ),
        }
        before_amplitude_roots = (
            round(
                sum(float(root.helicity_weight) for root in process.dag.amplitude_roots)
            )
            if parity_preweighted
            else len(process.dag.amplitude_roots)
        )
        validation = self._generation_config.validation
        filters: dict[str, object] = {
            **(
                {}
                if self._color_accuracy != "lc"
                else {
                    "lc_flow_layout": (
                        "all-flow-union" if all_flow_union else "topology-replay"
                    )
                }
            ),
            "structural_helicity_reduction": {
                "applied": parity_preweighted or reduced is not process.dag,
                "before_amplitude_roots": before_amplitude_roots,
                "after_amplitude_roots": len(reduced.amplitude_roots),
                "mode": "proven global-helicity-flip equivalence",
            },
            "dynamic_color_projection": dict(dynamic_color_projection),
            "relation_discovery": relation_discovery_payload,
            **(
                {}
                if reduced.helicity_recurrence is None
                else {
                    "helicity_recurrence": {
                        "contract_version": (HELICITY_RECURRENCE_CONTRACT_VERSION),
                        **reduced.helicity_recurrence.proof_counts(),
                        **(
                            {}
                            if reduced.helicity_materialization is None
                            else {
                                "materialized_current_count": (
                                    reduced.helicity_materialization.materialized_current_count
                                ),
                                "materialized_amplitude_count": (
                                    reduced.helicity_materialization.materialized_root_count
                                ),
                                "materialization_strategy": (
                                    reduced.helicity_materialization.strategy
                                ),
                            }
                        ),
                    }
                }
            ),
        }
        progress.update(
            1,
            message="validation-point preparation",
            details={
                "process": process_name,
                "step": "validation-point preparation",
                "current_count": len(reduced.currents),
                "interaction_count": len(reduced.interactions),
                "amplitude_count": len(reduced.amplitude_roots),
            },
        )
        sample_count = (
            validation.samples
            if validation.enabled and validation.post_build_validation
            else 1
        )
        points = tuple(
            build_validation_point(
                reduced,
                model,
                process_id=process.expanded.request.name,
                seed=validation.seed + index * sample_count + sample_index,
            )
            for sample_index in range(sample_count)
        )
        point = points[0]
        progress.update(
            2,
            message=(
                process_name
                if point.available
                else f"{process_name}: metadata-only validation"
            ),
            details={
                "process": process_name,
                "step": "validation points ready",
                "samples": len(points),
            },
        )
        coverage = {
            **dict(process.coverage),
            "current_count": len(reduced.currents),
            "interaction_count": len(reduced.interactions),
            "interaction_evaluation_count": reduced.interaction_evaluation_count,
            "amplitude_root_count": len(reduced.amplitude_roots),
            "dynamic_color_projection": dict(dynamic_color_projection),
            "relation_discovery": relation_discovery_payload,
            **(
                {}
                if reduced.helicity_recurrence is None
                else {
                    "helicity_recurrence_contract_version": (
                        HELICITY_RECURRENCE_CONTRACT_VERSION
                    ),
                    **reduced.helicity_recurrence.proof_counts(),
                    **(
                        {}
                        if reduced.helicity_materialization is None
                        else {
                            "proof_current_count": (
                                reduced.helicity_materialization.proof_current_count
                            ),
                            "proof_amplitude_count": (
                                reduced.helicity_materialization.proof_root_count
                            ),
                            "helicity_materialization_strategy": (
                                reduced.helicity_materialization.strategy
                            ),
                        }
                    ),
                }
            ),
        }
        return _CompiledProcess(
            expanded=process.expanded,
            dag=reduced,
            helicity_sum_dag=helicity_sum_dag,
            helicity_selector_union_dag=helicity_selector_union_dag,
            coverage=coverage,
            filters=filters,
            validation_points=points,
        )

    def _recurrence_model_inputs(
        self,
        resolved_model: _ResolvedModel,
    ) -> _RecurrenceModelInputs:
        compiled = resolved_model.compiled
        bundle = None if compiled is None else compiled.prepared_bundle
        if bundle is None:  # pragma: no cover - preflighted during model loading
            raise GenerationError("recurrence generation has no prepared model bundle")
        catalog = bundle.kernel_pack.recurrence_template_catalog
        if catalog is None:  # pragma: no cover - preflighted during model loading
            raise GenerationError(
                "recurrence generation has no recurrence-template-v1 catalog"
            )
        direct_catalog = bundle.kernel_pack.recurrence_direct_template_catalog
        if direct_catalog is None:  # pragma: no cover - preflighted during loading
            raise GenerationError(
                "recurrence generation has no recurrence-direct-template-v1 catalog"
            )
        return _RecurrenceModelInputs(
            bundle=bundle,
            catalog=catalog,
            template_catalog_json=json.dumps(
                catalog.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii"),
            template_input=build_recurrence_template_input_v1(catalog),
            direct_template_catalog=direct_catalog,
            direct_template_catalog_json=json.dumps(
                direct_catalog.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii"),
            prepared_kernel_pack_digest=(catalog.header.prepared_kernel_pack_digest),
        )

    def _lower_recurrence_helicity_selector_plan(
        self,
        logical: RecurrenceBuilderLogicalInputV1,
        color_plan: GenericColorPlan,
        model_inputs: _RecurrenceModelInputs,
        temporary_root: Path,
        *,
        process_id: str,
        progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    ) -> RecurrenceHelicitySelectorPlanArtifact:
        """Lower and bind one persisted physical-colour/runtime-H schedule.

        The native lowering emits the authenticated dispatch sidecar together
        with the ordinary Direct plan; no OTF seed or query family is built.
        """

        if logical.layout != "all-flow-union" or logical.replay_partitions:
            raise GenerationError(
                "recurrence helicity-selector plan requires physical "
                "all-flow-union without replay"
            )
        if color_plan.color_accuracy not in {"nlc", "full"}:
            raise GenerationError(
                "recurrence helicity-selector plan requires complete NLC/full color"
            )
        run = self._run_config
        if run is None:  # pragma: no cover - recurrence requires RunConfig
            raise GenerationError(
                "recurrence helicity-selector lowering has no evaluator config"
            )
        recurrence_evaluator = run.evaluator.recurrence
        builder_input = build_recurrence_builder_input_v1(logical)
        native_schedule_digest, base_schedule_digest, sharing_domain_digest = (
            _recurrence_schedule_semantic_digests(
                logical,
                prepared_kernel_pack_digest=(model_inputs.prepared_kernel_pack_digest),
                direct_template_catalog_digest=(
                    model_inputs.direct_template_catalog.catalog_digest
                ),
                point_tile_size=recurrence_evaluator.point_tile_size,
                workspace_mib=recurrence_evaluator.workspace_mib,
                relation_discovery=None,
            )
        )

        relation_discovery = self._generation_config.relation_discovery

        def lower_schedule() -> _RustRecurrenceLoweringOutput:
            output_root = (
                temporary_root
                / "recurrence-helicity-selector-schedules"
                / base_schedule_digest
            )
            return _invoke_rust_recurrence_lowering_v2(
                builder_input,
                model_inputs.template_input,
                model_inputs.direct_template_catalog,
                model_inputs.prepared_kernel_pack_digest,
                native_schedule_digest,
                output_root / "recurrence-runtime.pacbin",
                point_tile_size=recurrence_evaluator.point_tile_size,
                workspace_mib=recurrence_evaluator.workspace_mib,
                relation_discovery_mode="off",
                relation_discovery_precision_digits=(
                    relation_discovery.precision_digits
                ),
                relation_discovery_probe_count=relation_discovery.probe_count,
                relation_discovery_verification_probe_count=(
                    relation_discovery.verification_probe_count
                ),
                relation_discovery_relative_tolerance=(
                    relation_discovery.relative_tolerance
                ),
                relation_discovery_absolute_tolerance=(
                    relation_discovery.absolute_tolerance
                ),
                relation_discovery_seed=relation_discovery.seed,
                color_accuracy=str(color_plan.color_accuracy),
                helicity_dispatch_destination=(
                    output_root / "recurrence-helicity-dispatch-v1.bin"
                ),
                progress_callback=progress_callback,
                direct_template_catalog_json=(
                    model_inputs.direct_template_catalog_json
                ),
            )

        process_scoped_cache = RecurrenceScheduleLoweringCache[
            _RustRecurrenceLoweringOutput
        ]()
        shared = process_scoped_cache.lower_process(
            logical,
            schedule_digest=base_schedule_digest,
            sharing_domain_digest=sharing_domain_digest,
            native_schedule_semantic_digest=native_schedule_digest,
            direct_executor_count=len(model_inputs.direct_template_catalog.templates),
            parameter_slot_count=len(model_inputs.catalog.parameters),
            lower=lower_schedule,
        )
        expected_identity_remap = RecurrenceProcessRemap.identity(
            logical,
            direct_executor_count=len(model_inputs.direct_template_catalog.templates),
            parameter_slot_count=len(model_inputs.catalog.parameters),
        )
        if shared.remap != expected_identity_remap:
            raise GenerationError(
                "recurrence helicity-selector AllFlow plan acquired a nonidentity "
                "cross-process remap"
            )
        output = shared.output
        dispatch = output.helicity_dispatch
        if dispatch is None:
            raise GenerationError(
                "native recurrence helicity-selector lowering omitted its exact "
                "dispatch"
            )
        if output.inspection_summary.get("lc_flow_layout") != "all-flow-union":
            raise GenerationError(
                "native recurrence helicity-selector lowering changed its layout"
            )
        resolved_helicities = shared.remap.remap_resolved_helicities(
            output.resolved_helicities
        )
        if dispatch.resolved_helicity_count != len(resolved_helicities):
            raise GenerationError(
                "recurrence helicity dispatch does not cover the auxiliary full "
                "helicity domain"
            )
        try:
            destinations = tuple(
                (
                    shared.remap.physical_sector_ids[sector_id],
                    helicity_id,
                )
                for sector_id, helicity_id in output.amplitude_destinations
            )
        except IndexError as exc:
            raise GenerationError(
                "recurrence helicity-selector destination lies outside its "
                "physical-owner binding"
            ) from exc
        expected_destinations = tuple(
            (sector_id, None) for sector_id in range(len(logical.physical_sectors))
        )
        if destinations != expected_destinations:
            raise GenerationError(
                "recurrence helicity-selector destinations are not dense "
                "runtime-helicity physical owners"
            )

        required_direct_executor_ids = _required_process_direct_executor_ids(
            output.exact_sections,
            shared.remap,
        )
        process_executor_pack = build_recurrence_process_executor_pack(
            direct_catalog=model_inputs.direct_template_catalog,
            kernel_pack=model_inputs.bundle.kernel_pack,
            required_executor_ids=required_direct_executor_ids,
            runtime_layout_digest=dispatch.base_runtime_layout_digest,
        )
        selector_schedule_digest = recurrence_helicity_selector_schedule_digest(
            shared.schedule_digest,
            dispatch.payload_sha256,
        )
        inspection_summary = _process_scoped_recurrence_inspection_summary(
            output.inspection_summary,
            process_id=process_id,
            semantic_digest=builder_input.digest,
            schedule_digest=shared.schedule_digest,
        )
        return RecurrenceHelicitySelectorPlanArtifact(
            recurrence_schedule_path=output.payload_path,
            recurrence_base_schedule_digest=shared.schedule_digest,
            recurrence_schedule_digest=selector_schedule_digest,
            recurrence_native_schedule_semantic_digest=(
                shared.native_schedule_semantic_digest
            ),
            recurrence_schedule_size_bytes=output.payload_size_bytes,
            recurrence_schedule_sha256=output.payload_sha256,
            recurrence_schedule_member_count=output.member_count,
            recurrence_schedule_unpacked_size_bytes=output.unpacked_size_bytes,
            recurrence_schedule_index_sha256=output.index_sha256,
            helicity_dispatch_path=dispatch.payload_path,
            helicity_dispatch_size_bytes=dispatch.payload_size_bytes,
            helicity_dispatch_sha256=dispatch.payload_sha256,
            helicity_dispatch_base_runtime_layout_digest=(
                dispatch.base_runtime_layout_digest
            ),
            helicity_dispatch_resolved_helicity_count=(
                dispatch.resolved_helicity_count
            ),
            physical_destination_count=len(destinations),
            builder_input_sha256=builder_input.digest,
            inspection_summary=inspection_summary,
            referenced_kernel_ids=recurrence_referenced_kernel_ids(logical),
            recurrence_process_remap=shared.remap,
            recurrence_process_executor_pack=process_executor_pack,
        )

    def _drop_processes_without_amplitudes(
        self,
        expanded: tuple[_ExpandedProcess, ...],
        model: Model,
    ) -> tuple[_ExpandedProcess, ...]:
        """Skip children of a multiparticle request that the model cannot connect.

        The recurrence builders, unlike the DAG compiler, need an amplitude.
        """

        results: list[_ExpandedProcess | _UnsupportedProcess] = []
        for entry in expanded:
            process = entry.process_ir
            supported = entry.source_expansion_size <= 1 or (
                has_model_supported_amplitudes(
                    process,
                    model=model,
                    color_plan=build_color_plan(
                        process,
                        color_accuracy=self._color_accuracy,
                        fold_trace_reflections=(
                            model.lc_trace_reflection_equivalence_is_proven(process)
                        ),
                    ),
                    max_coupling_orders=self._coupling_order_limits or None,
                )
            )
            results.append(
                entry
                if supported
                else _UnsupportedProcess(
                    expanded=entry,
                    reason=(
                        f"process {process.process!r} has no model-supported "
                        "tree-level amplitudes"
                    ),
                )
            )
        kept, unsupported = _partition_model_supported_processes(results)
        _warn_unsupported_processes(unsupported)
        return kept

    def _construct_recurrence_artifact(
        self,
        expanded: _ExpandedProcess,
        model: Model,
        model_inputs: _RecurrenceModelInputs,
        temporary_root: Path,
        *,
        index: int,
        phase: PhaseHandle,
        lowering_cache: RecurrenceScheduleLoweringCache[_RustRecurrenceLoweringOutput],
        generation_timings: dict[str, float] | None = None,
    ) -> _GeneratedRecurrenceProcess:
        process_name = expanded.request.name
        profile_started = time.perf_counter()
        profile: dict[str, float] = {}
        selector_plan_enabled = self._recurrence_helicity_selector_plan_enabled()
        with phase.child(
            process_name,
            f"{process_name}: compact recurrence construction",
            total=3,
            details={"process": process_name, "step": "colour planning"},
        ) as task:
            preparation_started = time.perf_counter()
            prepared = self._prepare_process_construction(expanded.process_ir, model)
            profile["process-model-color-preparation"] = (
                time.perf_counter() - preparation_started
            )
            normalization_started = time.perf_counter()
            normalization_contract, normalization_payload = (
                build_recurrence_normalization(expanded.process_ir, model)
            )
            profile["normalization"] = time.perf_counter() - normalization_started
            layout = (
                "contracted-color-union"
                if self._color_accuracy in {"nlc", "full"}
                else (
                    "all-flow-union"
                    if self._all_flow_union_enabled
                    else "topology-replay"
                )
            )
            typed_layout = cast(
                Literal[
                    "topology-replay",
                    "all-flow-union",
                    "contracted-color-union",
                ],
                layout,
            )
            selection = self._process_selection
            if (
                layout == "contracted-color-union"
                and selection.selected_color_sector_ids is not None
            ):
                raise GenerationError(
                    "NLC/full recurrence contracts the complete color basis and "
                    "does not support generation-selected color sectors"
                )
            source_selection = (
                None
                if selection.selected_source_helicities is None
                else tuple(sorted(selection.selected_source_helicities.items()))
            )
            projection_color_plan = prepared.complete_color_plan
            if selection.selected_color_sector_ids is not None:
                # Compact recurrence tables require dense physical-sector IDs.
                # A generation-specialized artifact owns only its restricted
                # sector domain, so remap that domain before projection instead
                # of retaining the unrequested complete-plan aliases.
                projection_color_plan = replace(
                    prepared.restricted_color_plan,
                    sectors=tuple(
                        replace(sector, id=dense_id)
                        for dense_id, sector in enumerate(
                            prepared.restricted_color_plan.sectors
                        )
                    ),
                )
            projection_started = time.perf_counter()
            logical = project_recurrence_process_v1(
                expanded.process_ir,
                projection_color_plan,
                model_inputs.catalog,
                layout=typed_layout,
                normalization=normalization_contract,
                topology_replay=prepared.topology_replay,
                generation_slice=RecurrenceGenerationSliceV1(
                    selected_source_helicities=source_selection,
                ),
                coupling_order_limits=prepared.coupling_order_limits,
                model=model,
                prefer_symmetric_group_closure_anchor=(
                    self._run_config is not None
                    and self._run_config.color.contraction.value
                    in {"auto", "symmetric-group-fft"}
                ),
            )
            if selection.selected_color_sector_ids is not None:
                selected_public_flow_ids = tuple(
                    flow.flow_id for flow in logical.public_flows
                )
                logical = replace(
                    logical,
                    selected_public_flow_ids=selected_public_flow_ids,
                )
            profile["projection-semantic-catalog"] = (
                time.perf_counter() - projection_started
            )
            task.update(
                1,
                message="columnar recurrence input",
                details={
                    "process": process_name,
                    "step": "columnar recurrence input",
                    "color_sector_count": len(logical.physical_sectors),
                    "public_flow_count": len(logical.public_flows),
                },
            )
            columnar_started = time.perf_counter()
            builder_input = build_recurrence_builder_input_v1(logical)
            profile["columnar-normalization"] = time.perf_counter() - columnar_started
            native_process_id = logical.process_id
            runtime_metadata_started = time.perf_counter()
            runtime_metadata = build_recurrence_runtime_metadata(
                logical,
                model_inputs.catalog,
                model,
                normalization_payload,
            )
            profile["runtime-metadata"] = time.perf_counter() - runtime_metadata_started
            run = self._run_config
            if run is None:  # pragma: no cover - recurrence requires RunConfig
                raise GenerationError(
                    "recurrence generation has no evaluator configuration"
                )
            recurrence_evaluator = run.evaluator.recurrence
            relation_discovery = self._generation_config.relation_discovery
            relation_discovery_mode = str(relation_discovery.mode)
            relation_discovery_identity = (
                None
                if relation_discovery_mode == "off"
                else {
                    "mode": relation_discovery_mode,
                    "precision_digits": relation_discovery.precision_digits,
                    "probe_count": relation_discovery.probe_count,
                    "verification_probe_count": (
                        relation_discovery.verification_probe_count
                    ),
                    "relative_tolerance": (relation_discovery.relative_tolerance),
                    "absolute_tolerance": (relation_discovery.absolute_tolerance),
                    "seed": relation_discovery.seed,
                    "color_accuracy": str(expanded.process_ir.color_accuracy),
                    "probe_process_id": native_process_id,
                }
            )
            schedule_digest_started = time.perf_counter()
            (
                native_schedule_semantic_digest,
                request_schedule_digest,
                sharing_domain_digest,
            ) = _recurrence_schedule_semantic_digests(
                logical,
                prepared_kernel_pack_digest=(model_inputs.prepared_kernel_pack_digest),
                direct_template_catalog_digest=(
                    model_inputs.direct_template_catalog.catalog_digest
                ),
                point_tile_size=recurrence_evaluator.point_tile_size,
                workspace_mib=recurrence_evaluator.workspace_mib,
                relation_discovery=relation_discovery_identity,
            )
            profile["schedule-normalization-and-digests"] = (
                time.perf_counter() - schedule_digest_started
            )
            task.update(
                2,
                message="Rust recurrence builder",
                details={
                    "process": process_name,
                    "step": "Rust recurrence builder",
                },
            )
            native_phase_total = len(logical.external_legs) + 1
            with task.child(
                "rust-builder",
                f"{process_name}: Rust recurrence construction",
                total=native_phase_total,
                unit="phases",
                details={
                    "process": process_name,
                    "step": "source construction",
                },
            ) as native_task:

                def report_native(details: Mapping[str, object]) -> None:
                    progress_index = int(
                        details.get("phase_index", native_task.completed)
                    )
                    progress_total = int(details.get("phase_total", native_phase_total))
                    contribution_count = int(details.get("contribution_count", 0))
                    display_details = {
                        "process": process_name,
                        "step": str(details.get("step", "recurrence construction")),
                        "stage_total": int(details.get("stage_total", 0)),
                        "candidate_parent_pair_count": int(
                            details.get("candidate_parent_pair_count", 0)
                        ),
                        "current_count": int(details.get("current_count", 0)),
                        "contribution_count": contribution_count,
                        "dynamic_color_state_count": int(
                            details.get("dynamic_color_state_count", 0)
                        ),
                        "color_target_prune_count": int(
                            details.get("color_target_prune_count", 0)
                        ),
                    }
                    for name in (
                        "stage_index",
                        "subset_size",
                        "candidate_parent_pair_total",
                    ):
                        value = details.get(name)
                        if value is not None:
                            display_details[name] = int(value)
                    native_task.update(
                        progress_index,
                        total=progress_total,
                        message=str(details.get("step", "recurrence construction")),
                        details=display_details,
                    )

                final_schedule_path = (
                    temporary_root
                    / "recurrence-schedules"
                    / request_schedule_digest
                    / "recurrence-runtime.pacbin"
                )

                def lower_schedule() -> _RustRecurrenceLoweringOutput:
                    def lower_once(
                        destination: Path,
                        *,
                        mode: str,
                        evidence: bytes | None,
                        report_progress: bool,
                        timing_name: str,
                    ) -> _RustRecurrenceLoweringOutput:
                        lowering_started = time.perf_counter()
                        try:
                            return _invoke_rust_recurrence_lowering_v2(
                                builder_input,
                                model_inputs.template_input,
                                model_inputs.direct_template_catalog,
                                model_inputs.prepared_kernel_pack_digest,
                                native_schedule_semantic_digest,
                                destination,
                                point_tile_size=(recurrence_evaluator.point_tile_size),
                                workspace_mib=recurrence_evaluator.workspace_mib,
                                relation_discovery_mode=mode,
                                relation_discovery_precision_digits=(
                                    relation_discovery.precision_digits
                                ),
                                relation_discovery_probe_count=(
                                    relation_discovery.probe_count
                                ),
                                relation_discovery_verification_probe_count=(
                                    relation_discovery.verification_probe_count
                                ),
                                relation_discovery_relative_tolerance=(
                                    relation_discovery.relative_tolerance
                                ),
                                relation_discovery_absolute_tolerance=(
                                    relation_discovery.absolute_tolerance
                                ),
                                relation_discovery_seed=relation_discovery.seed,
                                color_accuracy=str(expanded.process_ir.color_accuracy),
                                relation_discovery_evidence_json=evidence,
                                progress_callback=(
                                    report_native
                                    if report_progress and native_task.sink is not None
                                    else None
                                ),
                                direct_template_catalog_json=(
                                    model_inputs.direct_template_catalog_json
                                ),
                            )
                        finally:
                            profile[timing_name] = (
                                time.perf_counter() - lowering_started
                            )

                    if relation_discovery_mode == "off":
                        output = lower_once(
                            final_schedule_path,
                            mode="off",
                            evidence=None,
                            report_progress=True,
                            timing_name="native-final-generation",
                        )
                        lane_report = recurrence_numerical_current_opt_out_report(
                            _parse_exact_sections(
                                output.exact_sections,
                                native_process_id,
                            ),
                            color_accuracy=str(expanded.process_ir.color_accuracy),
                        )
                        runtime_inspection, aggregate_report = (
                            _recurrence_relation_reporting(
                                output.inspection_summary,
                                mode=relation_discovery_mode,
                                lane_report=lane_report,
                            )
                        )
                        return replace(
                            output,
                            inspection_summary=runtime_inspection,
                            generation_profile={
                                "schema_version": 1,
                                "native_passes": {
                                    "final": dict(output.generation_profile)
                                },
                            },
                            numerical_current_reuse_report=aggregate_report,
                        )

                    baseline = lower_once(
                        temporary_root
                        / ".numerical-current-baseline"
                        / request_schedule_digest
                        / "recurrence-runtime.pacbin",
                        mode="off",
                        evidence=None,
                        report_progress=True,
                        timing_name="native-baseline-generation",
                    )
                    probe_points_started = time.perf_counter()
                    candidate_points, verification_points = (
                        build_recurrence_numerical_current_probe_points(
                            expanded.process_ir,
                            model,
                            process_id=native_process_id,
                            seed=relation_discovery.seed,
                            candidate_count=relation_discovery.probe_count,
                            verification_count=(
                                relation_discovery.verification_probe_count
                            ),
                        )
                    )
                    profile["numerical-probe-point-preparation"] = (
                        time.perf_counter() - probe_points_started
                    )
                    envelope_fallback: (
                        RecurrenceNumericalEvidenceEnvelopeExceeded | None
                    ) = None
                    warmup = None
                    try:
                        with _SYMBOLICA_MATERIALIZATION_LOCK:
                            baseline_plan_started = time.perf_counter()
                            baseline_plan = _RecurrenceExactPlan.from_generation(
                                raw_sections=baseline.exact_sections,
                                process_id=native_process_id,
                                bundle=model_inputs.bundle,
                                runtime_metadata=runtime_metadata,
                            )
                            profile["numerical-baseline-plan-load"] = (
                                time.perf_counter() - baseline_plan_started
                            )
                            warmup_started = time.perf_counter()
                            warmup = run_recurrence_numerical_current_warmup(
                                baseline_plan,
                                candidate_points=candidate_points,
                                verification_points=verification_points,
                                mode=cast(
                                    Literal[
                                        "diagnostic",
                                        "certified-reuse",
                                    ],
                                    relation_discovery_mode,
                                ),
                                color_accuracy=str(expanded.process_ir.color_accuracy),
                                precision_digits=(relation_discovery.precision_digits),
                                seed=relation_discovery.seed,
                                relative_tolerance=(
                                    relation_discovery.relative_tolerance
                                ),
                                absolute_tolerance=(
                                    relation_discovery.absolute_tolerance
                                ),
                            )
                            profile["numerical-warmup-total"] = (
                                time.perf_counter() - warmup_started
                            )
                            profile.update(
                                {
                                    f"numerical-{name}": seconds
                                    for name, seconds in (
                                        warmup.generation_profile_timings.items()
                                    )
                                }
                            )
                    except RecurrenceNumericalEvidenceEnvelopeExceeded as exc:
                        profile["numerical-warmup-total"] = (
                            time.perf_counter() - warmup_started
                        )
                        if relation_discovery_mode != "certified-reuse":
                            raise GenerationError(
                                f"process {process_name!r} recurrence numerical "
                                "diagnostic evidence exceeds its memory envelope"
                            ) from exc
                        envelope_fallback = (
                            _detached_recurrence_evidence_envelope_outcome(
                                exc,
                                source_error=exc,
                            )
                        )
                    except (ArtifactError, ValueError) as exc:
                        raise GenerationError(
                            f"process {process_name!r} recurrence numerical "
                            f"current warm-up failed closed: {exc}"
                        ) from exc
                    if envelope_fallback is not None:
                        fallback_details = envelope_fallback.to_json_dict()
                        geometry_details = cast(
                            Mapping[str, object],
                            fallback_details["geometry"],
                        )
                        native_task.update(
                            native_task.completed,
                            message=EVIDENCE_ENVELOPE_FALLBACK_WARNING,
                            details={
                                "process": process_name,
                                "step": "numerical-current reuse disabled",
                                "warning_code": (EVIDENCE_ENVELOPE_FALLBACK_REASON),
                                "current_count": int(geometry_details["current_count"]),
                                "component_count": int(
                                    geometry_details["component_count"]
                                ),
                                "scalar_count": int(geometry_details["scalar_count"]),
                                "row_count": int(geometry_details["row_count"]),
                            },
                        )
                        warmup_owner = None
                        if warmup is not None:
                            warmup_owner = [warmup]
                            warmup = None
                        return _complete_recurrence_evidence_envelope_fallback(
                            final_schedule_path=final_schedule_path,
                            baseline=baseline,
                            baseline_plan=baseline_plan,
                            color_accuracy=str(expanded.process_ir.color_accuracy),
                            outcome=envelope_fallback,
                            warmup_owner=warmup_owner,
                        )
                    if warmup is None:
                        raise AssertionError(
                            "recurrence numerical warm-up completed without a result"
                        )
                    try:
                        native_capacity_fallback: (
                            RecurrenceNumericalEvidenceEnvelopeExceeded | None
                        ) = None
                        try:
                            if not warmup.certificates:
                                warmup = warmup.without_evidence_transport()
                            output, application_validation_required = (
                                _lower_recurrence_after_numerical_warmup(
                                    lower_once=lower_once,
                                    final_schedule_path=final_schedule_path,
                                    baseline=baseline,
                                    warmup=warmup,
                                    requested_mode=relation_discovery_mode,
                                )
                            )
                        except (ArtifactError, GenerationError) as exc:
                            native_capacity_fallback = (
                                recurrence_numerical_evidence_capacity_outcome(
                                    baseline_plan.sections,
                                    candidate_probe_count=len(candidate_points),
                                    verification_probe_count=len(verification_points),
                                    runtime_parameter_count=len(
                                        baseline_plan.runtime_parameter_schema
                                    ),
                                    error=exc,
                                )
                            )
                            if (
                                native_capacity_fallback is None
                                or relation_discovery_mode != "certified-reuse"
                            ):
                                raise
                            native_capacity_fallback = (
                                _detached_recurrence_evidence_envelope_outcome(
                                    native_capacity_fallback,
                                    source_error=exc,
                                )
                            )
                        if native_capacity_fallback is not None:
                            # The failed native call can retain its evidence
                            # argument through exception frames.  The handler
                            # above detached those frames; close both trusted
                            # spools and drop the result (including the evidence
                            # transport) before publishing the already
                            # validated reuse-off baseline.
                            fallback_details = native_capacity_fallback.to_json_dict()
                            geometry_details = cast(
                                Mapping[str, object],
                                fallback_details["geometry"],
                            )
                            native_task.update(
                                native_task.completed,
                                message=EVIDENCE_ENVELOPE_FALLBACK_WARNING,
                                details={
                                    "process": process_name,
                                    "step": "numerical-current reuse disabled",
                                    "warning_code": (EVIDENCE_ENVELOPE_FALLBACK_REASON),
                                    "current_count": int(
                                        geometry_details["current_count"]
                                    ),
                                    "component_count": int(
                                        geometry_details["component_count"]
                                    ),
                                    "scalar_count": int(
                                        geometry_details["scalar_count"]
                                    ),
                                    "row_count": int(geometry_details["row_count"]),
                                },
                            )
                            fallback_color_accuracy = str(
                                expanded.process_ir.color_accuracy
                            )
                            warmup_owner = [warmup]
                            warmup = None
                            return _complete_recurrence_evidence_envelope_fallback(
                                final_schedule_path=final_schedule_path,
                                baseline=baseline,
                                baseline_plan=baseline_plan,
                                color_accuracy=fallback_color_accuracy,
                                outcome=native_capacity_fallback,
                                warmup_owner=warmup_owner,
                            )
                        if warmup is None:
                            raise AssertionError(
                                "recurrence numerical warm-up was released "
                                "without a fallback"
                            )
                        if not application_validation_required:
                            return output
                        warmup = warmup.without_evidence_transport()
                        try:
                            with _SYMBOLICA_MATERIALIZATION_LOCK:
                                applied_plan_started = time.perf_counter()
                                applied_plan = _RecurrenceExactPlan.from_generation(
                                    raw_sections=output.exact_sections,
                                    process_id=native_process_id,
                                    bundle=model_inputs.bundle,
                                    runtime_metadata=runtime_metadata,
                                )
                                profile["numerical-application-plan-load"] = (
                                    time.perf_counter() - applied_plan_started
                                )
                                application_validation_started = time.perf_counter()
                                warmup = (
                                    validate_recurrence_numerical_current_application(
                                        warmup,
                                        applied_plan,
                                        baseline_plan=baseline_plan,
                                        precision_digits=(
                                            relation_discovery.precision_digits
                                        ),
                                        seed=relation_discovery.seed,
                                        relative_tolerance=(
                                            relation_discovery.relative_tolerance
                                        ),
                                        absolute_tolerance=(
                                            relation_discovery.absolute_tolerance
                                        ),
                                    )
                                )
                                profile["numerical-application-validation"] = (
                                    time.perf_counter() - application_validation_started
                                )
                        except (ArtifactError, ValueError) as exc:
                            raise GenerationError(
                                f"process {process_name!r} optimized recurrence "
                                f"current replay failed closed: {exc}"
                            ) from exc
                        runtime_inspection, aggregate_report = (
                            _recurrence_relation_reporting(
                                output.inspection_summary,
                                mode=relation_discovery_mode,
                                lane_report=warmup.to_json_dict(),
                            )
                        )
                        return replace(
                            output,
                            inspection_summary=runtime_inspection,
                            generation_profile={
                                "schema_version": 1,
                                "native_passes": {
                                    "baseline": dict(baseline.generation_profile),
                                    "final": dict(output.generation_profile),
                                },
                            },
                            numerical_current_reuse_report=aggregate_report,
                        )
                    finally:
                        if warmup is not None:
                            warmup.close()

                schedule_lowering_started = time.perf_counter()
                shared = lowering_cache.lower_process(
                    logical,
                    schedule_digest=request_schedule_digest,
                    sharing_domain_digest=sharing_domain_digest,
                    native_schedule_semantic_digest=(native_schedule_semantic_digest),
                    direct_executor_count=len(
                        model_inputs.direct_template_catalog.templates
                    ),
                    parameter_slot_count=len(model_inputs.catalog.parameters),
                    lower=lower_schedule,
                )
                profile["schedule-sharing-and-lowering"] = (
                    time.perf_counter() - schedule_lowering_started
                )
                output = shared.output
                schedule_digest = shared.schedule_digest
                native_schedule_semantic_digest = shared.native_schedule_semantic_digest
                process_remap = shared.remap
            task.update(
                3,
                message="compact recurrence runtime ready",
                details={
                    "process": process_name,
                    "step": "compact recurrence runtime ready",
                },
            )
        companion_artifact = None
        if selector_plan_enabled:
            selector_phase_total = len(expanded.process_ir.legs) + 2
            with phase.child(
                f"{process_name}:helicity-selector",
                f"{process_name}: persisted helicity-selector plan",
                total=selector_phase_total,
                unit="phases",
                details={
                    "process": process_name,
                    "step": "physical-color selector projection",
                },
            ) as selector_task:
                companion_projection_started = time.perf_counter()
                companion_logical = project_recurrence_process_v1(
                    expanded.process_ir,
                    prepared.complete_color_plan,
                    model_inputs.catalog,
                    layout="all-flow-union",
                    normalization=normalization_contract,
                    topology_replay=None,
                    generation_slice=RecurrenceGenerationSliceV1(),
                    coupling_order_limits=prepared.coupling_order_limits,
                    model=model,
                    prefer_symmetric_group_closure_anchor=(
                        run.color.contraction.value in {"auto", "symmetric-group-fft"}
                    ),
                    contracted_physical_color_union=True,
                )
                profile["helicity-selector-physical-projection"] = (
                    time.perf_counter() - companion_projection_started
                )
                selector_task.update(
                    1,
                    message="physical-color selector input ready",
                    details={
                        "process": process_name,
                        "step": "physical-color selector input ready",
                        "color_sector_count": len(companion_logical.physical_sectors),
                    },
                )

                def report_selector(details: Mapping[str, object]) -> None:
                    phase_index = int(details.get("phase_index", 0))
                    phase_total = int(
                        details.get("phase_total", selector_phase_total - 1)
                    )
                    selector_task.update(
                        1 + phase_index,
                        total=1 + phase_total,
                        message=str(
                            details.get(
                                "step",
                                "persisted helicity-selector construction",
                            )
                        ),
                        details={
                            "process": process_name,
                            "step": str(
                                details.get(
                                    "step",
                                    "persisted helicity-selector construction",
                                )
                            ),
                            "current_count": int(details.get("current_count", 0)),
                            "contribution_count": int(
                                details.get("contribution_count", 0)
                            ),
                        },
                    )

                companion_lowering_started = time.perf_counter()
                companion_artifact = self._lower_recurrence_helicity_selector_plan(
                    companion_logical,
                    prepared.complete_color_plan,
                    model_inputs,
                    temporary_root / f"recurrence-helicity-selector-process-{index}",
                    process_id=process_name,
                    progress_callback=(
                        report_selector if selector_task.sink is not None else None
                    ),
                )
                profile["helicity-selector-physical-lowering"] = (
                    time.perf_counter() - companion_lowering_started
                )
                selector_task.update(
                    selector_task.total or selector_task.completed,
                    message="persisted helicity-selector plan ready",
                    details={
                        "process": process_name,
                        "step": "persisted helicity-selector plan ready",
                    },
                )
        phase.advance(
            message=process_name,
            details={"process": process_name, "step": "recurrence runtime ready"},
        )

        process_binding_started = time.perf_counter()
        validation = self._generation_config.validation
        sample_count = (
            validation.samples
            if validation.enabled and validation.post_build_validation
            else 1
        )
        points = tuple(
            build_process_validation_point(
                expanded.process_ir,
                model,
                process_id=process_name,
                seed=validation.seed + index * sample_count + sample_index,
            )
            for sample_index in range(sample_count)
        )
        process_digest = _recurrence_projection_digest(
            expanded.process_ir.to_json_dict()
        )
        resolved_helicities = process_remap.remap_resolved_helicities(
            output.resolved_helicities
        )
        try:
            amplitude_destinations = tuple(
                (
                    process_remap.physical_sector_ids[sector_id],
                    helicity_id,
                )
                for sector_id, helicity_id in output.amplitude_destinations
            )
        except IndexError as exc:
            raise GenerationError(
                "shared recurrence schedule references a physical sector outside "
                "the concrete process binding"
            ) from exc
        structural_zero_certificate = output.structural_zero_physical_sector_certificate
        if structural_zero_certificate is None:
            certified_structural_zero_sector_ids: tuple[int, ...] = ()
        else:
            if structural_zero_certificate.sector_count != len(
                process_remap.physical_sector_ids
            ):
                raise GenerationError(
                    "shared recurrence structural-zero certificate disagrees with "
                    "its physical-sector remap"
                )
            try:
                certified_structural_zero_sector_ids = tuple(
                    sorted(
                        process_remap.physical_sector_ids[sector_id]
                        for sector_id in structural_zero_certificate.sector_ids
                    )
                )
            except IndexError as exc:
                raise GenerationError(
                    "shared recurrence structural-zero certificate lies outside "
                    "the concrete process binding"
                ) from exc
        profile["process-binding-and-validation-points"] = (
            time.perf_counter() - process_binding_started
        )
        physics_started = time.perf_counter()
        physics = build_recurrence_physics(
            expanded.process_ir,
            logical,
            model_inputs.catalog,
            process_id=process_name,
            resolved_helicities=resolved_helicities,
            normalization=normalization_payload,
            selected_color_sector_ids=(
                None
                if selection.selected_color_sector_ids is None
                else tuple(sorted(selection.selected_color_sector_ids))
            ),
            color_plan=prepared.complete_color_plan,
        )
        profile["physics-metadata"] = time.perf_counter() - physics_started
        run = self._run_config
        assert run is not None
        schedule_summary = output.inspection_summary.get("schedule", {})
        recurrence_summary = {
            "lc_flow_layout": layout,
            "builder_input_abi": RECURRENCE_BUILDER_INPUT_ABI,
            "template_input_abi": RECURRENCE_TEMPLATE_INPUT_ABI,
            "current_count": (
                schedule_summary.get("current_count", 0)
                if isinstance(schedule_summary, Mapping)
                else 0
            ),
            "contribution_count": (
                schedule_summary.get("contribution_count", 0)
                if isinstance(schedule_summary, Mapping)
                else 0
            ),
            "closure_term_count": (
                schedule_summary.get("closure_term_count", 0)
                if isinstance(schedule_summary, Mapping)
                else 0
            ),
        }
        inspection_summary = _process_scoped_recurrence_inspection_summary(
            output.inspection_summary,
            process_id=process_name,
            semantic_digest=builder_input.digest,
            schedule_digest=schedule_digest,
        )
        color_contraction_started = time.perf_counter()
        color_contraction = build_recurrence_color_contraction(
            logical,
            prepared.complete_color_plan,
            resolved_helicities,
            amplitude_destinations,
            certified_structural_zero_sector_ids,
            contraction=run.color.contraction.value,
        )
        color_contraction_payload = None
        color_contraction_summary = None
        if color_contraction is not None:
            sector_count = len(logical.physical_sectors)
            component_count = len(resolved_helicities)
            group_count = color_contraction.group_count
            schedule_destination_count = int(
                schedule_summary.get("amplitude_destination_count", 0)
                if isinstance(schedule_summary, Mapping)
                else 0
            )
            if schedule_destination_count != len(amplitude_destinations):
                raise GenerationError(
                    "contracted recurrence destination metadata does not match "
                    "its Direct schedule"
                )
            contraction_destinations = recurrence_color_contraction_destinations(
                logical,
                resolved_helicities,
                amplitude_destinations,
            )
            destination_count = len(contraction_destinations)
            if destination_count < group_count:
                raise GenerationError(
                    "contracted recurrence physical replay domain is smaller "
                    "than its color-contraction group domain"
                )
            if any(helicity_id is None for _, helicity_id in amplitude_destinations):
                raise GenerationError(
                    "contracted recurrence exposes a destination without a "
                    "resolved helicity"
                )
            destination_by_group = (
                tuple(range(group_count))
                if color_contraction.destination_by_group is None
                else color_contraction.destination_by_group
            )
            if (
                len(destination_by_group) != group_count
                or len(set(destination_by_group)) != group_count
                or any(
                    destination_id < 0 or destination_id >= destination_count
                    for destination_id in destination_by_group
                )
            ):
                raise GenerationError(
                    "contracted recurrence color plan has an invalid destination "
                    "projection"
                )
            normalized_destinations = tuple(
                contraction_destinations[destination_id]
                for destination_id in destination_by_group
            )
            group_sector_ids = tuple(
                sector_id for sector_id, _ in normalized_destinations
            )
            group_component_ids = tuple(
                cast(int, helicity_id) for _, helicity_id in normalized_destinations
            )
            repeated = color_contraction.repeated_block
            symmetric_group = color_contraction.symmetric_group_block
            ordered_group_ids = (
                repeated.component_group_ids
                if repeated is not None
                else symmetric_group.component_group_ids
                if symmetric_group is not None
                else tuple(range(group_count))
            )
            color_contraction_payload = encode_recurrence_color_contraction(
                color_contraction,
                sector_count=sector_count,
                component_count=component_count,
                ordered_group_ids=ordered_group_ids,
                destination_by_group=destination_by_group,
                destination_count=destination_count,
                group_sector_ids=group_sector_ids,
                group_component_ids=group_component_ids,
                sector_owner_ids=recurrence_color_sector_owner_map(
                    logical,
                    set(group_sector_ids),
                    set(certified_structural_zero_sector_ids),
                ),
                exact_coefficients=recurrence_exact_color_coefficients(
                    prepared.complete_color_plan,
                    color_contraction,
                    group_sector_ids,
                ),
            )
            color_digest = recurrence_color_contraction_digest(
                color_contraction_payload
            )
            color_contraction_summary = {
                "abi": RECURRENCE_COLOR_CONTRACTION_CODEC_ABI,
                "color_accuracy": color_contraction.color_accuracy,
                "storage": (
                    "convolution-kernels"
                    if symmetric_group is not None
                    else "expanded"
                    if repeated is None
                    else "repeated"
                ),
                "includes_color_factor": color_contraction.includes_color_factor,
                "group_count": group_count,
                "sector_count": sector_count,
                "active_sector_count": len(set(group_sector_ids)),
                "component_count": component_count,
                "destination_count": destination_count,
                "materialized_destination_count": schedule_destination_count,
                "entry_count": (
                    symmetric_group.stored_entry_count
                    if symmetric_group is not None
                    else len(color_contraction.entries)
                    if repeated is None
                    else len(repeated.entries)
                ),
                "logical_entry_count": color_contraction.logical_entry_count,
                "semantic_digest": color_digest,
                "factorization": (
                    {
                        "kind": "symmetric-group-fourier",
                        "rank": symmetric_group.degree,
                        "coset_count": symmetric_group.channel_count,
                    }
                    if symmetric_group is not None
                    else None
                    if repeated is None or repeated.factorized_block is None
                    else {
                        "kind": repeated.factorized_block.kind,
                        "rank": (
                            2
                            if repeated.factorized_block.kind == "klein-four-walsh"
                            else repeated.factorized_block.rank
                        ),
                        "coset_count": len(repeated.factorized_block.cosets),
                    }
                ),
                **(
                    {}
                    if symmetric_group is None
                    else {
                        "fft_provenance": {
                            "method": "symmetric-group-fourier",
                            "degree": symmetric_group.degree,
                            "channel_count": symmetric_group.channel_count,
                            "covered_local_group_count": (
                                symmetric_group.channel_count
                                * math.factorial(symmetric_group.degree)
                            ),
                            "residual_group_count": len(
                                symmetric_group.residual_local_group_indices
                            ),
                            "residual_entry_count": len(
                                symmetric_group.residual_entries
                            ),
                            "raw_kernel_bytes": (
                                len(symmetric_group.kernel_entries) * 16
                            ),
                            "transformed_kernel_bytes": (
                                len(symmetric_group.kernel_entries) * 8
                            ),
                            "capability": (
                                SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY
                            ),
                        }
                    }
                ),
            }
        profile["color-contraction"] = time.perf_counter() - color_contraction_started
        relation_discovery_payload: dict[str, object]
        numerical_current_reuse = output.numerical_current_reuse_report
        if not isinstance(numerical_current_reuse, Mapping):
            raise GenerationError(
                "recurrence numerical current lowering did not publish its "
                "aggregate report"
            )
        relation_discovery_payload = dict(numerical_current_reuse)
        artifact_record_started = time.perf_counter()
        required_direct_executor_ids = _required_process_direct_executor_ids(
            output.exact_sections,
            process_remap,
        )
        process_executor_pack = build_recurrence_process_executor_pack(
            direct_catalog=model_inputs.direct_template_catalog,
            kernel_pack=model_inputs.bundle.kernel_pack,
            required_executor_ids=required_direct_executor_ids,
            runtime_layout_digest=str(inspection_summary["runtime_layout_digest"]),
        )
        umami_keys = None
        umami_physics = None
        umami_sectors = ()
        if run.generation.emit_api_bundle:
            umami_sectors = tuple(
                sector.to_json_dict() for sector in prepared.complete_color_plan.sectors
            )
            umami_physics = dict(physics)
            from .umami_semantics import recurrence_structural_keys

            umami_keys = recurrence_structural_keys(
                exact_sections=output.exact_sections,
                runtime_metadata=runtime_metadata,
                kernel_pack=model_inputs.bundle.kernel_pack,
                ir=model_inputs.bundle.compiled_model["ir"],
                physics=physics,
                remap=process_remap,
                color_contraction_payload=color_contraction_payload,
            )
            replay = (
                prepared.topology_replay
                if isinstance(prepared.topology_replay, LCColorTopologyReplayPlan)
                else build_lc_topology_replay_plan(prepared.complete_color_plan, model)
            )
            if replay is not None:
                umami_physics["extensions"] = {
                    **cast(Mapping[str, object], physics.get("extensions", {})),
                    "lc_topology_replay": replay.to_json_dict(),
                }
        artifact = RecurrenceProcessArtifact(
            process_id=process_name,
            expression=expanded.process_ir.process,
            color_accuracy=expanded.process_ir.color_accuracy,
            external_pdgs=(
                *expanded.process_ir.initial_pdgs,
                *expanded.process_ir.final_pdgs,
            ),
            aliases=expanded.aliases,
            physics=physics,
            umami_physics=umami_physics,
            umami_color_sectors=umami_sectors,
            umami_structural_keys=umami_keys,
            recurrence_schedule_path=output.payload_path,
            recurrence_schedule_digest=schedule_digest,
            recurrence_native_schedule_semantic_digest=(
                native_schedule_semantic_digest
            ),
            recurrence_schedule_size_bytes=output.payload_size_bytes,
            recurrence_schedule_sha256=output.payload_sha256,
            recurrence_schedule_member_count=output.member_count,
            recurrence_schedule_unpacked_size_bytes=output.unpacked_size_bytes,
            recurrence_schedule_index_sha256=output.index_sha256,
            builder_input_sha256=builder_input.digest,
            prepared_kernel_pack_digest=model_inputs.prepared_kernel_pack_digest,
            direct_template_catalog_digest=(
                model_inputs.direct_template_catalog.catalog_digest
            ),
            referenced_kernel_ids=(
                recurrence_referenced_kernel_ids(logical)
                if companion_artifact is None
                else frozenset(
                    kernel.kernel_id
                    for kernel in model_inputs.bundle.kernel_pack.kernels
                )
            ),
            inspection_summary=inspection_summary,
            runtime_metadata=runtime_metadata,
            color_contraction_payload=color_contraction_payload,
            color_contraction_summary=color_contraction_summary,
            point_tile_size=run.evaluator.recurrence.point_tile_size,
            workspace_mib=run.evaluator.recurrence.workspace_mib,
            recurrence_summary=recurrence_summary,
            validation_point=points[0],
            generation_filters={
                **self._fft_basis_metadata(prepared.complete_color_plan),
                "lc_flow_layout": layout,
                "recurrence": recurrence_summary,
                "relation_discovery": relation_discovery_payload,
            },
            generation_profile=output.generation_profile,
            process_support_mask=1 << index,
            recurrence_process_remap=process_remap,
            recurrence_process_executor_pack=process_executor_pack,
            helicity_selector_companion=companion_artifact,
            process_digest=process_digest,
        )
        profile["artifact-record-assembly"] = (
            time.perf_counter() - artifact_record_started
        )
        profile["process-total"] = time.perf_counter() - profile_started
        if generation_timings is not None:
            timing_prefix = f"recurrence-process-{index}:{process_name}:"
            with _RECURRENCE_PROFILE_LOCK:
                generation_timings.update(
                    {
                        f"{timing_prefix}{name}": seconds
                        for name, seconds in profile.items()
                    }
                )
        _LOGGER.debug(
            "recurrence generation profile for %s: %s",
            process_name,
            json.dumps(profile, allow_nan=False, sort_keys=True),
        )
        return _GeneratedRecurrenceProcess(
            expanded=expanded,
            artifact=artifact,
            validation_points=points,
        )

    def _construct_evaluator(
        self,
        process: _CompiledProcess,
        model: Model,
        phase: PhaseHandle,
    ) -> _EvaluatorProcess:
        process_name = process.expanded.request.name
        run = self._run_config
        compiled_symmetric_group = (
            run is not None and run.color.contraction.value == "symmetric-group-fft"
        )
        with phase.child(
            process_name,
            f"{process_name}: runtime schema",
            details={"process": process_name, "step": "runtime layout"},
        ) as task:
            color_contraction_payload = None
            if compiled_symmetric_group:
                _validate_compiled_symmetric_group_diagnostic_process(process)
                schema_layout = build_runtime_schema_layout(
                    process.dag,
                    model,
                    process_id=process_name,
                    color_contraction="symmetric-group-fft",
                )
                schema = RuntimeExpressionSchema.from_mapping(
                    schema_layout.runtime_schema
                )
                color_contraction_payload = (
                    _compiled_symmetric_group_color_contraction_payload(
                        process.dag,
                        schema_layout,
                    )
                )
            else:
                schema = build_runtime_expression_schema(
                    process.dag,
                    model,
                    process_id=process_name,
                )
            helicity_sum_schema = (
                None
                if process.helicity_sum_dag is None
                else build_runtime_expression_schema(
                    process.helicity_sum_dag,
                    model,
                    process_id=process_name,
                )
            )
            helicity_selector_lanes: tuple[_HelicitySelectorLane, ...] = ()
            if (
                process.dag.helicity_coverage == "complete"
                and not process.dag.selected_source_helicities
                and process.dag.helicity_recurrence is not None
                and process.dag.helicity_materialization is not None
            ):
                union_dag = process.helicity_selector_union_dag
                if union_dag is not None:
                    union_schema = build_runtime_expression_schema(
                        union_dag,
                        model,
                        process_id=process_name,
                    )
                    materialization = process.dag.helicity_materialization
                    assert materialization is not None
                    helicity_selector_lanes = (
                        _HelicitySelectorLane(
                            selector_domain_ids=tuple(
                                sorted(
                                    schedule.selector_domain_id
                                    for schedule in materialization.selector_schedules
                                    if not schedule.structural_zero
                                )
                            ),
                            active_current_ids=(),
                            active_root_ids=(),
                            dag=union_dag,
                            runtime_schema=union_schema,
                            stage_input=StageCompilationInput(
                                union_dag,
                                model,
                                union_schema,
                            ),
                            schedule_mode="nested-runtime",
                            child_lanes=(),
                        ),
                    )
                elif process.dag.process.color_accuracy == "lc":
                    helicity_selector_lanes = _compiled_helicity_selector_closure_lanes(
                        process.dag,
                        schema,
                        model,
                    )
            color_selector_dag = (
                process.dag
                if process.helicity_sum_dag is None
                else process.helicity_sum_dag
            )
            color_selector_lanes: list[_ColorSelectorLane] = []
            sector_lanes = (
                ()
                if self._all_flow_union_enabled
                else _compiled_lc_color_selector_lane_dags(color_selector_dag)
            )
            for sector_id, lane_dag in sector_lanes:
                lane_schema = build_runtime_expression_schema(
                    lane_dag,
                    model,
                    process_id=process_name,
                )
                color_selector_lanes.append(
                    _ColorSelectorLane(
                        materialized_sector_id=sector_id,
                        dag=lane_dag,
                        runtime_schema=lane_schema,
                        stage_input=StageCompilationInput(
                            lane_dag,
                            model,
                            lane_schema,
                        ),
                    )
                )
            stage_count = _runtime_stage_count(schema) + (
                0
                if helicity_sum_schema is None
                else _runtime_stage_count(helicity_sum_schema)
            )
            stage_count += sum(
                _helicity_selector_lane_stage_count(lane)
                for lane in helicity_selector_lanes
            )
            stage_count += sum(
                _runtime_stage_count(lane.runtime_schema)
                for lane in color_selector_lanes
            )
            task.update(
                stage_count,
                total=stage_count,
                message="runtime schema ready",
                details={
                    "process": process_name,
                    "step": "runtime schema ready",
                    "stage_index": stage_count,
                    "stage_total": stage_count,
                    "current_count": len(process.dag.currents),
                    "interaction_count": len(process.dag.interactions),
                    "output_count": len(process.dag.amplitude_roots),
                    "helicity_sum_lane": helicity_sum_schema is not None,
                    "helicity_selector_closure_lane_count": len(
                        helicity_selector_lanes
                    ),
                    "color_selector_lane_count": len(color_selector_lanes),
                },
            )
        phase.advance(
            message=process_name,
            details={"process": process_name, "step": "runtime schema ready"},
        )
        return _EvaluatorProcess(
            compiled=process,
            runtime_schema=schema,
            stage_input=StageCompilationInput(process.dag, model, schema),
            color_contraction_payload=color_contraction_payload,
            helicity_sum_runtime_schema=helicity_sum_schema,
            helicity_sum_stage_input=(
                None
                if helicity_sum_schema is None or process.helicity_sum_dag is None
                else StageCompilationInput(
                    process.helicity_sum_dag,
                    model,
                    helicity_sum_schema,
                )
            ),
            helicity_selector_lanes=helicity_selector_lanes,
            color_selector_lanes=tuple(color_selector_lanes),
        )

    def _construct_eager_artifact(
        self,
        process: _CompiledProcess,
        model: Model,
        resolved_model: _ResolvedModel,
        temporary_root: Path,
        phase: PhaseHandle,
    ) -> EagerPlanV3ProcessArtifact:
        eager_kernel_index = resolved_model.eager_kernel_index
        if eager_kernel_index is None:
            raise GenerationError(
                "eager DAG lowering requires an indexed prepared model bundle"
            )
        resolver = PreparedCatalogEagerKernelResolver(
            process.dag,
            eager_kernel_index,
        )
        return self._construct_eager_plan_v3_artifact(
            process,
            model,
            resolver,
            temporary_root,
            phase,
        )

    def _construct_eager_plan_v3_artifact(
        self,
        process: _CompiledProcess,
        model: Model,
        resolver: PreparedCatalogEagerKernelResolver,
        temporary_root: Path,
        phase: PhaseHandle,
    ) -> EagerPlanV3ProcessArtifact:
        process_name = process.expanded.request.name
        with phase.child(
            process_name,
            f"{process_name}: Rust eager runtime lowering",
            total=2,
            details={"process": process_name, "step": "columnar lowering input"},
        ) as task:
            lowering_input = build_eager_lowering_input_v1(
                dag=process.dag,
                model=model,
                resolver=resolver,
                process_id=process_name,
            )
            task.update(
                1,
                message="columnar lowering input ready",
                details={
                    "process": process_name,
                    "step": "columnar lowering input ready",
                },
            )
            output = _invoke_rust_eager_lowering_v1(
                lowering_input,
                temporary_root
                / "eager-plan-v3"
                / process_name
                / "eager-runtime.pacbin",
            )
            task.update(
                2,
                message="Rust eager runtime ready",
                details={"process": process_name, "step": "eager runtime ready"},
            )
        phase.advance(
            message=process_name,
            details={"process": process_name, "step": "eager runtime ready"},
        )
        run = self._run_config
        if run is None:  # pragma: no cover - eager mode requires RunConfig
            raise GenerationError("eager generation has no evaluator configuration")
        ir = process.expanded.process_ir
        dag = process.dag
        physics = build_resolved_physics_from_dag(
            dag,
            model,
            process_id=process_name,
            native_eager_plan_v3_reduction_groups=True,
            runtime_model_parameters=(
                runtime_model_parameter_records_from_eager_input(lowering_input)
            ),
        )
        return EagerPlanV3ProcessArtifact(
            process_id=process_name,
            expression=ir.process,
            color_accuracy=ir.color_accuracy,
            external_pdgs=(*ir.initial_pdgs, *ir.final_pdgs),
            aliases=process.expanded.aliases,
            physics=physics,
            umami_color_sectors=tuple(
                sector.to_json_dict() for sector in dag.color_plan.sectors
            )
            if run.generation.emit_api_bundle
            else (),
            eager_runtime_path=output.payload_path,
            eager_runtime_size_bytes=output.payload_size_bytes,
            eager_runtime_sha256=output.payload_sha256,
            eager_runtime_member_count=output.member_count,
            eager_runtime_unpacked_size_bytes=output.unpacked_size_bytes,
            eager_runtime_index_sha256=output.index_sha256,
            lowering_input_sha256=lowering_input.digest,
            referenced_kernel_ids=_eager_lowering_kernel_ids(lowering_input),
            inspection_summary=output.inspection_summary,
            point_tile_size=run.evaluator.eager.point_tile_size,
            workspace_mib=run.evaluator.eager.workspace_mib,
            dag_summary={
                "current_count": len(dag.currents),
                "source_count": len(dag.sources),
                "interaction_count": len(dag.interactions),
                "interaction_evaluation_count": dag.interaction_evaluation_count,
                "amplitude_root_count": len(dag.amplitude_roots),
                "truncated": False,
            },
            validation_point=process.validation_points[0],
            generation_filters=process.filters,
        )

    def _materialize_evaluator(
        self,
        process: _EvaluatorProcess,
        model: Model,
        temporary_root: Path,
        phase: PhaseHandle,
    ) -> CompiledProcessArtifact:
        process_id = process.compiled.expanded.request.name
        run = self._run_config
        backend = "JIT" if run is None else str(run.evaluator.backend).upper()
        with phase.child(
            process_id,
            f"{process_id}: evaluator compilation",
            details={
                "process": process_id,
                "step": "waiting for compiler",
                "backend": backend,
            },
        ) as task:
            task.update(
                0,
                message="waiting for compiler",
                details={
                    "process": process_id,
                    "step": "waiting for compiler",
                    "backend": backend,
                },
            )
            with _SYMBOLICA_MATERIALIZATION_LOCK:
                return self._materialize_evaluator_unlocked(
                    process,
                    model,
                    temporary_root,
                    phase,
                    task,
                    backend=backend,
                )

    def _materialize_evaluator_unlocked(
        self,
        process: _EvaluatorProcess,
        model: Model,
        temporary_root: Path,
        phase: PhaseHandle,
        progress: PhaseHandle,
        *,
        backend: str,
    ) -> CompiledProcessArtifact:
        process_id = process.compiled.expanded.request.name
        evaluator_root = temporary_root / process_id
        total_stage_count = _runtime_stage_count(process.runtime_schema) + 1
        if process.helicity_sum_runtime_schema is not None:
            total_stage_count += (
                _runtime_stage_count(process.helicity_sum_runtime_schema) + 1
            )
        total_stage_count += sum(
            _helicity_selector_lane_compile_task_count(lane)
            for lane in process.helicity_selector_lanes
        )
        total_stage_count += sum(
            _runtime_stage_count(lane.runtime_schema) + 1
            for lane in process.color_selector_lanes
        )

        def evaluator_progress(
            event: dict[str, object],
            *,
            execution_lane: str,
        ) -> None:
            details = {
                str(key): value
                for key, value in event.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            }
            details.update(
                {
                    "process": process_id,
                    "backend": backend,
                    "execution_lane": execution_lane,
                    "lane_stage_total": event.get("total"),
                }
            )
            if event.get("stage") == "stage complete":
                completed = progress.completed + int(event.get("increment", 1))
                progress.update(
                    completed,
                    total=total_stage_count,
                    message=str(event.get("item", "stage complete")),
                    details=details,
                )
                phase.advance(
                    message=str(event.get("item", process_id)),
                    details=details,
                )
                return
            progress.update(
                progress.completed,
                total=total_stage_count,
                message=str(event.get("item", event.get("stage", "compiling"))),
                details=details,
            )

        def compile_lane(
            stage_input: StageCompilationInput,
            runtime_schema: RuntimeExpressionSchema,
            output_root: Path,
            *,
            execution_lane: str,
        ) -> tuple[Mapping[str, object], Mapping[str, object] | None]:
            callback = (
                None
                if progress.sink is None
                else lambda event: evaluator_progress(
                    event,
                    execution_lane=execution_lane,
                )
            )
            _blueprint, stage_manifest = (
                build_and_write_generic_stage_evaluator_artifacts(
                    stage_input,
                    runtime_schema.to_mapping(),
                    output_root,
                    model=model,
                    stage_local_parameter_layout=True,
                    symbolica_settings=self._symbolica_settings(),
                    jit_compile=True,
                    evaluator_progress_callback=callback,
                )
            )
            if not bool(stage_manifest.get("runtime_available")):
                raise GenerationError(
                    f"{execution_lane} stage evaluator for {process_id!r} "
                    "is not runtime-available"
                )
            model_parameter_evaluator = write_model_parameter_evaluator_artifact(
                model,
                runtime_schema.to_mapping(),
                output_root,
                symbolica_settings=self._symbolica_settings(),
                jit_compile=True,
                progress_callback=callback,
            )
            return stage_manifest, model_parameter_evaluator

        stage_manifest, model_parameter_evaluator = compile_lane(
            process.stage_input,
            process.runtime_schema,
            evaluator_root,
            execution_lane="selected-helicity",
        )
        helicity_sum_execution: CompiledExecutionArtifact | None = None
        if (
            process.helicity_sum_runtime_schema is not None
            or process.helicity_sum_stage_input is not None
        ):
            if (
                process.helicity_sum_runtime_schema is None
                or process.helicity_sum_stage_input is None
                or process.compiled.helicity_sum_dag is None
            ):
                raise GenerationError(
                    f"helicity-sum execution inputs for {process_id!r} are inconsistent"
                )
            helicity_sum_root = temporary_root / ".helicity-sum" / process_id
            (
                helicity_sum_stage_manifest,
                helicity_sum_model_parameter_evaluator,
            ) = compile_lane(
                process.helicity_sum_stage_input,
                process.helicity_sum_runtime_schema,
                helicity_sum_root,
                execution_lane="helicity-sum",
            )
            helicity_sum_dag = process.compiled.helicity_sum_dag
            helicity_sum_execution = CompiledExecutionArtifact(
                runtime_schema=process.helicity_sum_runtime_schema,
                stage_manifest=helicity_sum_stage_manifest,
                model_parameter_evaluator=(helicity_sum_model_parameter_evaluator),
                dag_summary={
                    "current_count": len(helicity_sum_dag.currents),
                    "source_count": len(helicity_sum_dag.sources),
                    "interaction_count": len(helicity_sum_dag.interactions),
                    "interaction_evaluation_count": (
                        helicity_sum_dag.interaction_evaluation_count
                    ),
                    "amplitude_root_count": len(helicity_sum_dag.amplitude_roots),
                    "truncated": False,
                },
                evaluator_root=helicity_sum_root,
            )
        helicity_selector_executions: list[
            CompiledHelicitySelectorExecutionArtifact
        ] = []

        def compile_helicity_selector_lane(
            lane: _HelicitySelectorLane,
            lane_root: Path,
            *,
            execution_lane: str,
        ) -> CompiledHelicitySelectorExecutionArtifact:
            lane_stage_manifest, lane_model_parameter_evaluator = compile_lane(
                lane.stage_input,
                lane.runtime_schema,
                lane_root,
                execution_lane=execution_lane,
            )
            child_executions = tuple(
                compile_helicity_selector_lane(
                    child,
                    lane_root.parent / f"{lane_root.name}-closure-{child_index}",
                    execution_lane=(f"{execution_lane}-closure-{child_index}"),
                )
                for child_index, child in enumerate(lane.child_lanes)
            )
            return CompiledHelicitySelectorExecutionArtifact(
                selector_domain_ids=lane.selector_domain_ids,
                schedule_mode=lane.schedule_mode,
                execution=CompiledExecutionArtifact(
                    runtime_schema=lane.runtime_schema,
                    stage_manifest=lane_stage_manifest,
                    model_parameter_evaluator=(lane_model_parameter_evaluator),
                    # Closure lanes retain their parent's stable current and
                    # amplitude storage ABI while filtering stage interactions.
                    dag_summary=_runtime_schema_dag_summary(lane.runtime_schema),
                    evaluator_root=lane_root,
                    helicity_selector_executions=child_executions,
                ),
            )

        for lane_index, lane in enumerate(process.helicity_selector_lanes):
            helicity_selector_root = (
                temporary_root
                / ".helicity-selector-union"
                / process_id
                / f"class-{lane_index}"
            )
            helicity_selector_executions.append(
                compile_helicity_selector_lane(
                    lane,
                    helicity_selector_root,
                    execution_lane=f"helicity-selector-class-{lane_index}",
                )
            )
        color_selector_executions: list[CompiledColorSelectorExecutionArtifact] = []
        selector_root_name = (
            ".helicity-sum-color-selector"
            if helicity_sum_execution is not None
            else ".color-selector"
        )
        for lane in process.color_selector_lanes:
            lane_root = (
                temporary_root
                / selector_root_name
                / process_id
                / f"sector-{lane.materialized_sector_id}"
            )
            lane_stage_manifest, lane_model_parameter_evaluator = compile_lane(
                lane.stage_input,
                lane.runtime_schema,
                lane_root,
                execution_lane=(f"color-selector sector-{lane.materialized_sector_id}"),
            )
            color_selector_executions.append(
                CompiledColorSelectorExecutionArtifact(
                    materialized_sector_id=lane.materialized_sector_id,
                    execution=CompiledExecutionArtifact(
                        runtime_schema=lane.runtime_schema,
                        stage_manifest=lane_stage_manifest,
                        model_parameter_evaluator=(lane_model_parameter_evaluator),
                        dag_summary={
                            "current_count": len(lane.dag.currents),
                            "source_count": len(lane.dag.sources),
                            "interaction_count": len(lane.dag.interactions),
                            "interaction_evaluation_count": (
                                lane.dag.interaction_evaluation_count
                            ),
                            "amplitude_root_count": len(lane.dag.amplitude_roots),
                            "truncated": False,
                        },
                        evaluator_root=lane_root,
                    ),
                )
            )
        primary_color_selector_executions: tuple[
            CompiledColorSelectorExecutionArtifact, ...
        ] = tuple(color_selector_executions)
        if helicity_sum_execution is not None:
            helicity_sum_execution = replace(
                helicity_sum_execution,
                color_selector_executions=tuple(color_selector_executions),
            )
            primary_color_selector_executions = ()
        progress.update(
            progress.completed,
            message="model parameters ready",
            details={
                "process": process_id,
                "step": "model parameters ready",
                "backend": backend,
            },
        )
        phase.advance(
            message=f"{process_id}: model parameters",
            details={
                "process": process_id,
                "step": "model parameters ready",
                "backend": backend,
            },
        )
        ir = process.compiled.expanded.process_ir
        dag = process.compiled.dag
        return CompiledProcessArtifact(
            process_id=process_id,
            expression=ir.process,
            color_accuracy=ir.color_accuracy,
            external_pdgs=(*ir.initial_pdgs, *ir.final_pdgs),
            aliases=process.compiled.expanded.aliases,
            runtime_schema=process.runtime_schema,
            umami_color_sectors=tuple(
                sector.to_json_dict() for sector in dag.color_plan.sectors
            )
            if self._generation_config.emit_api_bundle
            else (),
            stage_manifest=stage_manifest,
            model_parameter_evaluator=model_parameter_evaluator,
            dag_summary={
                "current_count": len(dag.currents),
                "source_count": len(dag.sources),
                "interaction_count": len(dag.interactions),
                "interaction_evaluation_count": dag.interaction_evaluation_count,
                "amplitude_root_count": len(dag.amplitude_roots),
                "truncated": False,
            },
            evaluator_root=evaluator_root,
            color_contraction_payload=process.color_contraction_payload,
            validation_point=process.compiled.validation_points[0],
            generation_filters=process.compiled.filters,
            helicity_sum_execution=helicity_sum_execution,
            helicity_selector_executions=tuple(helicity_selector_executions),
            color_selector_executions=primary_color_selector_executions,
        )

    def _process_worker_count(self, process_count: int) -> int:
        if process_count < 1:
            return 1
        configured = self._generation_config.workers
        requested = (
            max(1, os.cpu_count() or 1)
            if configured == "auto"
            else max(1, int(configured))
        )
        return min(process_count, requested)

    def _detect_symbolica_license(self) -> SymbolicaLicenseState:
        from ..licensing import detect_symbolica_license

        run = self._run_config
        return detect_symbolica_license(
            suggest=True if run is None else run.symbolica.suggest_license,
            json_mode=False if run is None else str(run.output.format) == "json",
        )

    def _apply_symbolica_resource_policy(
        self,
        state: SymbolicaLicenseState,
        *,
        process_count: int | None = None,
    ) -> None:
        from ..licensing import resolve_symbolica_resource_config

        resolution = resolve_symbolica_resource_config(
            self._resource_config,
            state,
            process_count=process_count,
        )
        self._resource_config = resolution
        self._configuration = _GenerationConfigProvenance.from_config(resolution)
        self._config = resolution.effective

    def _symbolica_settings(self) -> object:
        from ..evaluators.symbolica import SymbolicaEvaluatorSettings

        run = self._run_config
        if run is None:
            return SymbolicaEvaluatorSettings(jit_compress=True)
        optimization = run.evaluator.optimization
        if optimization.horner_iterations < 1:
            raise GenerationError(
                "evaluator.optimization.horner_iterations must be positive "
                "for generation"
            )
        backend = str(run.evaluator.backend)
        evaluator_backend = "jit" if backend == "jit" else "compiled-complex"
        cores = (
            max(1, os.cpu_count() or 1)
            if optimization.cores == "auto"
            else int(optimization.cores)
        )
        collect_factors = (
            False
            if optimization.collect_factors == "auto"
            else bool(optimization.collect_factors)
        )
        cpp_level = _cpp_optimization_level(run.evaluator.cpp.optimization)
        return SymbolicaEvaluatorSettings(
            backend=evaluator_backend,
            iterations=optimization.horner_iterations,
            cpe_iterations=optimization.cpe_iterations,
            n_cores=cores,
            jit_direct_translation=False,
            jit_optimization_level=run.evaluator.jit.optimization_level,
            jit_compress=run.evaluator.resolved_jit_compress,
            max_horner_scheme_variables=optimization.max_horner_variables,
            max_common_pair_cache_entries=(optimization.max_common_pair_cache_entries),
            max_common_pair_distance=optimization.max_common_pair_distance,
            collect_factors=collect_factors,
            compiled_inline_asm="default" if backend == "asm" else "none",
            compiled_optimization_level=cpp_level,
            compiled_native=run.evaluator.cpp.native_arch,
            compiler_path=run.evaluator.cpp.compiler,
            compiler_flags=run.evaluator.cpp.extra_flags,
            compiled_output_chunk_size=run.evaluator.output_chunk_size,
            output_chunk_strategy="uniform",
            output_chunk_autotune_batch_size=run.evaluator.batch_size,
            compiled_chunk_compile_workers=1,
        )

    def _validate_generated_artifact(
        self,
        output: Path,
        process_ids: Sequence[str],
        *,
        validation_points: Mapping[str, Sequence[ValidationPointRecord]],
        expected_api_bundle_path: str | None,
        compact_validation_selectors: Mapping[str, Mapping[str, tuple[str, ...]]]
        | None = None,
        progress: PhaseHandle | None = None,
    ) -> None:
        import cmath

        from pyamplicol.api import Runtime
        from pyamplicol.artifacts import load_manifest

        manifest = load_manifest(output)
        expected_ids = set(process_ids)
        actual_ids = {str(process["id"]) for process in manifest.processes}
        if not expected_ids.issubset(actual_ids):
            raise GenerationError("generated artifact omitted concrete processes")
        by_path = {record.path: record for record in manifest.payloads}
        process_record_by_id = {
            str(process["id"]): process for process in manifest.processes
        }
        process_total = len(process_ids)
        for process_index, process_id in enumerate(process_ids, start=1):
            if progress is not None:
                progress.update(
                    process_index - 1,
                    total=process_total,
                    message=f"loading {process_id}",
                    details={
                        "process": process_id,
                        "step": "runtime validation",
                        "sample_index": 0,
                    },
                )
            prefix = f"processes/{process_id}"
            required = {
                f"{prefix}/physics.json",
                f"{prefix}/execution.json",
                f"{prefix}/validation-momenta.json",
            }
            if self._on_the_fly_execution_enabled:
                required.add(f"{prefix}/on-the-fly-runtime.pacbin")
                if process_record_by_id[process_id].get("color_accuracy") in {
                    "nlc",
                    "full",
                }:
                    required.add(f"{prefix}/on-the-fly-color.bin")
            if not required.issubset(by_path):
                raise GenerationError(
                    f"artifact payload set is incomplete for {process_id!r}"
                )
            runtime = Runtime.load(output, process=process_id)
            validation = self._generation_config.validation
            samples = tuple(
                point.four_vectors
                for point in validation_points.get(process_id, ())
                if point.available
            )
            if self._on_the_fly_execution_enabled:
                if runtime.artifact_id != manifest.artifact_id:
                    raise GenerationError(
                        "Rusticol loaded the wrong artifact while validating "
                        f"{process_id!r}"
                    )
                if runtime.execution_mode != "on-the-fly":
                    raise GenerationError(
                        f"Rusticol selected the wrong execution mode for {process_id!r}"
                    )
                if runtime.representative_process_key != process_id:
                    raise GenerationError(
                        f"Rusticol selected the wrong process for {process_id!r}"
                    )
                validated_samples = 0
                if validation.enabled and samples:
                    raw_selectors = (
                        None
                        if compact_validation_selectors is None
                        else compact_validation_selectors.get(process_id)
                    )
                    if not isinstance(raw_selectors, Mapping) or set(raw_selectors) != {
                        "helicities",
                        "color_flows",
                    }:
                        raise GenerationError(
                            "on-the-fly validation has no authenticated compact "
                            f"selector for {process_id!r}"
                        )
                    helicities = raw_selectors.get("helicities")
                    color_flows = raw_selectors.get("color_flows")
                    contracted_color = process_record_by_id[process_id].get(
                        "color_accuracy"
                    ) in {"nlc", "full"}
                    if (
                        not isinstance(helicities, tuple)
                        or len(helicities) != 1
                        or not isinstance(helicities[0], str)
                        or not helicities[0].startswith("h:")
                        or not isinstance(color_flows, tuple)
                        or color_flows != (() if contracted_color else ("1",))
                    ):
                        raise GenerationError(
                            "on-the-fly validation compact selector is malformed "
                            f"for {process_id!r}"
                        )
                    selector_label = (
                        "compact helicity+contracted-color"
                        if contracted_color
                        else "compact helicity+flow"
                    )
                    selectors = {"helicities": helicities}
                    if color_flows:
                        selectors["color_flows"] = color_flows
                    selected_samples = samples[:1]
                    total = runtime.evaluate(selected_samples, **selectors)
                    resolved_total = runtime.evaluate_resolved(
                        selected_samples, **selectors
                    ).total()
                    if len(total) != 1 or len(resolved_total) != 1:
                        raise GenerationError(
                            f"Rusticol returned an invalid validation shape for "
                            f"{process_id!r} ({selector_label})"
                        )
                    summed = total[0]
                    resolved = resolved_total[0]
                    if progress is not None:
                        progress.update(
                            process_index - 1,
                            total=process_total,
                            message=f"{process_id}: {selector_label}, sample 1",
                            details={
                                "process": process_id,
                                "step": "numerical validation",
                                "selector_case": selector_label,
                                "validation_slice": selector_label,
                                "sample_index": 1,
                                "sample_total": 1,
                            },
                        )
                    difference = abs(complex(summed) - complex(resolved))
                    if not cmath.isclose(
                        complex(summed),
                        complex(resolved),
                        rel_tol=validation.relative_tolerance,
                        abs_tol=validation.absolute_tolerance,
                    ):
                        raise GenerationError(
                            "resolved Rusticol validation does not reduce to "
                            f"the total for {process_id!r} ({selector_label}) "
                            "sample 1 "
                            f"(absolute difference {difference:.3e})"
                        )
                    validated_samples = 1
                if progress is not None:
                    progress.update(
                        process_index,
                        total=process_total,
                        message=process_id,
                        details={
                            "process": process_id,
                            "step": "validated",
                            "samples": validated_samples,
                        },
                    )
                continue
            if runtime.physics.process_id != process_id:
                raise GenerationError(
                    f"Rusticol selected the wrong process for {process_id!r}"
                )
            if validation.enabled and samples:
                selector_cases = (
                    _recurrence_validation_selector_cases(runtime.physics)
                    if self._recurrence_execution_enabled
                    else tuple(
                        (
                            validation_slice,
                            {
                                **({"helicities": helicities} if helicities else {}),
                                **({"color_flows": color_flows} if color_flows else {}),
                            },
                        )
                        for validation_slice, helicities, color_flows in (
                            _post_build_validation_slices(
                                runtime.physics,
                                len(samples),
                            )
                        )
                    )
                )
                if not selector_cases:
                    raise GenerationError(
                        f"generated recurrence exposes no selectable components "
                        f"for {process_id!r}"
                    )
                for selector_label, selectors in selector_cases:
                    total = runtime.evaluate(samples, **selectors)
                    resolved_total = runtime.evaluate_resolved(
                        samples, **selectors
                    ).total()
                    if len(total) != len(samples) or len(resolved_total) != len(
                        samples
                    ):
                        raise GenerationError(
                            f"Rusticol returned an invalid validation shape for "
                            f"{process_id!r} ({selector_label})"
                        )
                    for sample_index, (summed, resolved) in enumerate(
                        zip(total, resolved_total, strict=True),
                        start=1,
                    ):
                        if progress is not None:
                            progress.update(
                                process_index - 1,
                                total=process_total,
                                message=(
                                    f"{process_id}: {selector_label}, "
                                    f"sample {sample_index}"
                                ),
                                details={
                                    "process": process_id,
                                    "step": "numerical validation",
                                    "selector_case": selector_label,
                                    "validation_slice": selector_label,
                                    "sample_index": sample_index,
                                    "sample_total": len(samples),
                                },
                            )
                        difference = abs(complex(summed) - complex(resolved))
                        if not cmath.isclose(
                            complex(summed),
                            complex(resolved),
                            rel_tol=validation.relative_tolerance,
                            abs_tol=validation.absolute_tolerance,
                        ):
                            raise GenerationError(
                                "resolved Rusticol validation does not reduce to "
                                f"the total for {process_id!r} ({selector_label}) "
                                f"sample {sample_index} "
                                f"(absolute difference {difference:.3e})"
                            )
            if progress is not None:
                progress.update(
                    process_index,
                    total=process_total,
                    message=process_id,
                    details={
                        "process": process_id,
                        "step": "validated",
                        "samples": len(samples),
                    },
                )
        actual_api_bundle_path = manifest.runtime.get("api_bundle_path")
        if actual_api_bundle_path != expected_api_bundle_path:
            raise GenerationError("artifact API bundle outcome is inconsistent")

    @property
    def _run_config(self) -> RunConfig | None:
        return self._config if isinstance(self._config, RunConfig) else None

    @property
    def _eager_execution_enabled(self) -> bool:
        run = self._run_config
        return run is not None and str(run.evaluator.execution_mode) == "eager"

    @property
    def _recurrence_execution_enabled(self) -> bool:
        run = self._run_config
        return run is not None and str(run.evaluator.execution_mode) == "recurrence"

    @property
    def _on_the_fly_execution_enabled(self) -> bool:
        run = self._run_config
        return run is not None and str(run.evaluator.execution_mode) == "on-the-fly"

    @property
    def _prepared_execution_enabled(self) -> bool:
        return (
            self._eager_execution_enabled
            or self._recurrence_execution_enabled
            or self._on_the_fly_execution_enabled
        )

    def _validate_recurrence_request(self) -> None:
        """Keep recurrence-family requests out of compiled/eager lanes."""

        if self._on_the_fly_execution_enabled and self._color_accuracy not in {
            "lc",
            "nlc",
            "full",
        }:
            raise GenerationError(
                "on-the-fly execution supports LC, NLC, and full color; "
                f"got color.accuracy={self._color_accuracy!r}"
            )
        if self._recurrence_execution_enabled and self._color_accuracy not in {
            "lc",
            "nlc",
            "full",
        }:
            raise GenerationError(
                "recurrence execution supports LC, NLC, and full color; "
                f"got color.accuracy={self._color_accuracy!r}"
            )

    @property
    def _all_flow_union_enabled(self) -> bool:
        run = self._run_config
        if run is None:
            return False
        return str(getattr(run.color, "lc_flow_layout", "topology-replay")) == (
            "all-flow-union"
        )

    @property
    def _generation_config(self) -> GenerationConfig:
        config = self._config
        return config.generation if isinstance(config, RunConfig) else config

    @property
    def _builtin_process_options(self):
        from ..models.builtin.process_types import BuiltinProcessOptions

        run = self._run_config
        if run is None:
            return BuiltinProcessOptions()
        max_lines = run.process.max_quark_lines
        return BuiltinProcessOptions(
            flavour_scheme=run.process.flavor_scheme,
            include_3qqbar=max_lines is None or max_lines >= 3,
        )

    @property
    def _max_quark_pairs(self) -> int | None:
        run = self._run_config
        return None if run is None else run.process.max_quark_lines

    @property
    def _color_accuracy(self) -> str:
        run = self._run_config
        return "lc" if run is None else run.color.accuracy

    @property
    def _coupling_order_limits(self) -> dict[str, int]:
        run = self._run_config
        if run is None:
            return {}
        return {
            str(name).upper(): int(value)
            for name, value in run.process.max_coupling_orders.items()
        }

    @property
    def _flavor_blind(self) -> bool:
        run = self._run_config
        return run is not None and run.process.flavor_blind

    @property
    def _consider_charge_conjugation(self) -> bool:
        run = self._run_config
        return run is None or run.process.consider_charge_conjugation

    @property
    def _process_selection(self) -> _ProcessSelection:
        if self._process_selection_override is not None:
            return self._process_selection_override
        run = self._run_config
        if run is None:
            return _ProcessSelection()
        process = run.process
        return _ProcessSelection(
            max_color_sectors=process.max_color_sectors,
            reference_color_order=(
                tuple(int(label) for label in process.reference_color_order)
                if process.reference_color_order
                else None
            ),
            selected_color_sector_ids=(
                frozenset(
                    int(sector_id) for sector_id in process.selected_color_sector_ids
                )
                if process.selected_color_sector_ids
                else None
            ),
            selected_source_helicities=(
                {
                    int(label): int(helicity)
                    for label, helicity in process.selected_source_helicities.items()
                }
                if process.selected_source_helicities
                else None
            ),
        )

    def _configured_model_source(self) -> ModelSource:
        run = self._run_config
        if run is None:
            return ModelSource.built_in_sm()
        return ModelSource.from_config(run.model)

    def _resolve_model_for_plan(
        self,
        source: ModelSource | CompiledModel,
    ) -> _ResolvedModel:
        if isinstance(source, CompiledModel):
            compiled = self._private_compiled_model(source)
            is_builtin = self._is_builtin_compiled_model(compiled)
            return _ResolvedModel(
                self._source_for_compiled_model(compiled),
                _builtin_sm_model() if is_builtin else None,
                compiled,
                use_compiled_process_catalog=not is_builtin,
            )
        if source.kind in {"built-in-sm", "built-in-sm-heft"}:
            if source.kind == "built-in-sm" and not self._prepared_execution_enabled:
                return _ResolvedModel(source, _builtin_sm_model())
            if self._prepared_execution_enabled:
                source = self._packaged_builtin_prepared_source(source.kind)
        source_input: str | Path
        if source.kind == "built-in-sm-heft":
            source_input = "built-in-sm-heft"
        elif source.path is not None:
            source_input = source.path
        else:
            raise ModelError("external model source has no path")

        from .._internal.sm_heft import load_cached_sm_heft_source
        from ..models.loading import load_cached_model_source, load_compiled_model

        try:
            if source.kind in {"compiled", "prepared"}:
                compiled = load_compiled_model(source.path)
            else:
                run = self._run_config
                use_cache = True if run is None else run.model.cache
                if not use_cache:
                    raise ModelError(
                        "dry-run does not compile external model sources and model "
                        "caching is disabled; compile the ModelSource explicitly and "
                        "pass its CompiledModel"
                    )
                cache_dir = None if run is None else run.model.cache_dir
                compiled = (
                    load_cached_sm_heft_source(cache_dir=cache_dir)
                    if source.kind == "built-in-sm-heft"
                    else load_cached_model_source(
                        source_input,
                        restriction=(
                            "default"
                            if source.restriction is None
                            else str(source.restriction)
                        ),
                        simplify=source.simplify,
                        cache_dir=cache_dir,
                    )
                )
                if compiled is None:
                    raise ModelError(
                        "dry-run does not compile external model sources; call "
                        "ModelSource.compile() first and pass the returned "
                        "CompiledModel, or populate the configured model cache"
                    )
        except ModelError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ModelError(str(exc)) from exc
        is_builtin = self._is_builtin_compiled_model(compiled)
        return _ResolvedModel(
            source,
            _builtin_sm_model() if is_builtin else None,
            compiled,
            use_compiled_process_catalog=not is_builtin,
        )

    @staticmethod
    def _private_compiled_model(compiled: CompiledModel) -> _CompiledModelPayload:
        payload = _compiled_model_payload(compiled)
        if not isinstance(payload, _CompiledModelPayload):
            raise ModelError("compiled model handle has an invalid private payload")
        return payload

    @staticmethod
    def _source_for_compiled_model(compiled: _CompiledModelPayload) -> ModelSource:
        if compiled.prepared_bundle is not None:
            return ModelSource(kind="prepared", path=compiled.prepared_bundle.path)
        return ModelSource(kind="compiled", path=compiled._serialized_path)

    @staticmethod
    def _is_builtin_compiled_model(compiled: _CompiledModelPayload) -> bool:
        return compiled.source.get("kind") == "built-in-sm"

    def _resolve_model(
        self,
        source: ModelSource | CompiledModel,
    ) -> _ResolvedModel:
        if isinstance(source, CompiledModel):
            compiled = self._private_compiled_model(source)
            if self._is_builtin_compiled_model(compiled):
                return _ResolvedModel(
                    self._source_for_compiled_model(compiled),
                    _builtin_sm_model(),
                    compiled,
                    use_compiled_process_catalog=False,
                )

            from ..models.external import CompiledUFOModel

            return _ResolvedModel(
                self._source_for_compiled_model(compiled),
                CompiledUFOModel(compiled),
                compiled,
            )
        if source.kind in {"built-in-sm", "built-in-sm-heft"}:
            if source.kind == "built-in-sm" and not self._prepared_execution_enabled:
                return _ResolvedModel(source, _builtin_sm_model())
            if self._prepared_execution_enabled:
                source = self._packaged_builtin_prepared_source(source.kind)
        source_input: str | Path
        if source.kind == "built-in-sm-heft":
            source_input = "built-in-sm-heft"
        elif source.path is not None:
            source_input = source.path
        else:
            raise ModelError("external model source has no path")
        from .._internal.sm_heft import compile_sm_heft_source
        from ..models.external import CompiledUFOModel
        from ..models.loading import compile_model_source, load_compiled_model

        try:
            if source.kind in {"compiled", "prepared"}:
                compiled = load_compiled_model(source.path)
            else:
                run = self._run_config
                use_cache = True if run is None else run.model.cache
                cache_dir = None if run is None else run.model.cache_dir
                compiled = (
                    compile_sm_heft_source(
                        cache_dir=cache_dir,
                        use_cache=use_cache,
                    )
                    if source.kind == "built-in-sm-heft"
                    else compile_model_source(
                        source_input,
                        restriction=(
                            "default"
                            if source.restriction is None
                            else str(source.restriction)
                        ),
                        simplify=source.simplify,
                        cache_dir=cache_dir,
                        use_cache=use_cache,
                    )
                )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ModelError(str(exc)) from exc
        if self._is_builtin_compiled_model(compiled):
            return _ResolvedModel(
                source,
                _builtin_sm_model(),
                compiled,
                use_compiled_process_catalog=False,
            )
        return _ResolvedModel(source, CompiledUFOModel(compiled), compiled)

    @staticmethod
    def _packaged_builtin_prepared_source(source_kind: str) -> ModelSource:
        from ..assets.prepared_models import (
            BUILTIN_SM_HEFT_JIT_O2,
            BUILTIN_SM_JIT_O2,
            PackagedPreparedModelError,
            materialize_packaged_prepared_model,
        )

        identifiers = {
            "built-in-sm": BUILTIN_SM_JIT_O2,
            "built-in-sm-heft": BUILTIN_SM_HEFT_JIT_O2,
        }
        try:
            identifier = identifiers[source_kind]
        except KeyError as exc:  # pragma: no cover - private caller contract
            raise ModelError(
                f"no packaged prepared model is registered for {source_kind!r}"
            ) from exc
        try:
            path = materialize_packaged_prepared_model(identifier)
        except (OSError, PackagedPreparedModelError, RuntimeError, ValueError) as exc:
            raise ModelError(
                f"the wheel-owned prepared {source_kind} model is unavailable "
                f"or stale: {exc}"
            ) from exc
        return ModelSource.from_path(path)

    def _require_eager_kernel_pack(self, resolved: _ResolvedModel) -> None:
        run = self._run_config
        if run is None or not self._prepared_execution_enabled:
            return
        execution_mode = str(run.evaluator.execution_mode)
        compiled = resolved.compiled
        if compiled is None or compiled.prepared_bundle is None:
            source = resolved.source.path or resolved.source.kind
            backend = str(run.evaluator.backend)
            raise GenerationError(
                f"{execution_mode} generation requires a prepared model kernel "
                "pack; run: "
                f"pyamplicol model compile {source} MODEL.pyamplicol-model "
                f"--backend {backend}"
            )
        pack = compiled.prepared_bundle.kernel_pack
        recurrence_family = execution_mode in {"recurrence", "on-the-fly"}
        if recurrence_family and pack.recurrence_template is None:
            source = resolved.source.path or resolved.source.kind
            backend = str(run.evaluator.backend)
            raise GenerationError(
                "prepared model kernel pack does not contain "
                "pyamplicol-recurrence-template-v1; rebuild it with: "
                f"pyamplicol model compile {source} MODEL.pyamplicol-model "
                f"--backend {backend}"
            )
        if recurrence_family:
            direct_catalog = pack.recurrence_direct_template_catalog
            if direct_catalog is None:
                source = resolved.source.path or resolved.source.kind
                backend = str(run.evaluator.backend)
                raise GenerationError(
                    "prepared model kernel pack does not contain "
                    "pyamplicol-recurrence-direct-template-v1; rebuild it with: "
                    f"pyamplicol model compile {source} MODEL.pyamplicol-model "
                    f"--backend {backend}"
                )
            pending = tuple(
                template.direct_executor_id
                for template in direct_catalog.templates
                if not template.payload_binding.executable
            )
            if pending:
                raise GenerationError(
                    "prepared model recurrence direct-template catalog contains "
                    "non-executable bindings "
                    f"{pending}; rebuild the prepared model with a Direct-Arena "
                    "capable backend"
                )
        try:
            validate_prepared_target(
                pack.target,
                backend=pack.backend,
                symjit_application_abi=cast(
                    str | None,
                    pack.dependency_abis.get("symjit_application"),
                ),
            )
        except PreparedTargetError as exc:
            source = resolved.source.path or resolved.source.kind
            raise GenerationError(
                f"{execution_mode} prepared model is incompatible with this "
                f"host: {exc}; "
                "prepare a matching bundle with: "
                f"pyamplicol model compile {source} MODEL.pyamplicol-model "
                f"--backend {pack.backend}"
            ) from exc

    def _index_eager_kernel_pack(self, resolved: _ResolvedModel) -> _ResolvedModel:
        if not self._eager_execution_enabled:
            return resolved
        compiled = resolved.compiled
        bundle = None if compiled is None else compiled.prepared_bundle
        if bundle is None:  # pragma: no cover - guarded by _require_eager_kernel_pack
            raise GenerationError("eager generation has no prepared kernel pack")
        try:
            index = PreparedCatalogEagerKernelIndex.from_manifest(
                bundle.kernel_pack.resolver_manifest
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GenerationError(
                f"prepared model has an invalid eager kernel resolver: {exc}"
            ) from exc
        return replace(resolved, eager_kernel_index=index)

    def _apply_prepared_kernel_pack_policy(self, resolved: _ResolvedModel) -> None:
        if not self._prepared_execution_enabled:
            return
        run = self._run_config
        if run is None:  # pragma: no cover - guarded by property above
            return
        execution_mode = str(run.evaluator.execution_mode)
        compiled = resolved.compiled
        bundle = None if compiled is None else compiled.prepared_bundle
        if bundle is None:  # pragma: no cover - guarded by _require_eager_kernel_pack
            raise GenerationError(
                f"{execution_mode} generation has no prepared kernel pack"
            )
        resolution = self._resource_config
        if not isinstance(resolution, ConfigResolution):
            resolution = resolve_config(config_to_dict(self._config))
        effective_values = config_to_dict(resolution.effective)
        pack_values = _prepared_pack_effective_values(bundle.kernel_pack)
        replacement_paths = frozenset(pack_values)
        inherited = [
            ClampRequest(item.path, item.effective, item.reason)
            for item in resolution.clamps
            if item.path not in replacement_paths
        ]
        prepared_clamps: list[ClampRequest] = []
        for path, value in pack_values.items():
            if _nested_config_value(effective_values, path) == value:
                continue
            prepared_clamps.append(
                ClampRequest(
                    path,
                    value,
                    f"prepared {execution_mode} kernel pack is authoritative for "
                    "backend "
                    "and code-shaping optimization settings",
                )
            )
        if not prepared_clamps:
            return
        updated = resolve_config(
            config_to_dict(resolution.requested),
            clamps=(*inherited, *prepared_clamps),
        )
        self._resource_config = updated
        self._configuration = _GenerationConfigProvenance.from_config(updated)
        self._config = updated.effective
        if not self._prepared_pack_warning_emitted:
            rendered = ", ".join(clamp.path for clamp in prepared_clamps)
            _LOGGER.warning(
                "%s generation uses prepared model settings for: %s",
                execution_mode,
                rendered,
            )
            self._prepared_pack_warning_emitted = True

    def _expand_request(
        self,
        request: ProcessRequest,
        resolved_model: _ResolvedModel,
    ) -> tuple[CanonicalProcessIR, ...]:
        if (
            resolved_model.compiled is None
            or not resolved_model.use_compiled_process_catalog
        ):
            from ..models.builtin.process_catalog import PDGS
            from ..models.builtin.process_ir import build_process_ir
            from ..models.builtin.process_selection import (
                enumerate_generic_process_set,
            )
            from ..processes.core_syntax import expand_process_variants

            multiparticles = {"all": tuple(PDGS)}
            if self._run_config is not None:
                multiparticles.update(self._run_config.process.multiparticles)
            expanded_request = " | ".join(
                expand_process_variants(
                    request.expression,
                    aliases=multiparticles,
                )
            )

            enumeration = enumerate_generic_process_set(
                expanded_request,
                self._builtin_process_options,
                max_quark_pairs=self._max_quark_pairs,
            )
            return tuple(
                build_process_ir(
                    entry.process,
                    color_accuracy=self._color_accuracy,
                    options=self._builtin_process_options,
                )
                for entry in enumeration.entries
            )

        from ..processes.conservation import conserved_charges
        from ..processes.model import (
            ModelParticleCatalog,
            build_model_process_ir,
            expand_model_processes,
        )

        catalog = ModelParticleCatalog(
            resolved_model.compiled.ir.name,
            resolved_model.compiled.ir.particles,
            resolved_model.compiled.ir.parameters,
        )
        run = self._run_config
        multiparticles = None if run is None else run.process.multiparticles
        candidates = tuple(
            build_model_process_ir(
                process,
                resolved_model.compiled.ir,
                color_accuracy=self._color_accuracy,
            )
            for process in expand_model_processes(
                request.expression,
                catalog,
                multiparticles=multiparticles,
                charges=conserved_charges(resolved_model.compiled.ir),
            )
        )
        if not candidates:
            raise GenerationError(
                f"process request {request.expression!r} has no concrete process "
                "conserving the additive charges of the model vertices"
            )
        selected, rejected = _select_color_ready_processes(
            candidates,
            color_accuracy=self._color_accuracy,
            # Structural colour coverage is independent of the requested
            # reduction. Model/coupling-order certification happens below.
            basis="trace",
        )
        if not selected:
            detail = "; ".join(rejected) or "no concrete processes"
            raise GenerationError(
                f"process request {request.expression!r} has no usable color plan: "
                f"{detail}"
            )
        return selected

    def _fft_color_basis(
        self, process: CanonicalProcessIR, model: Model | None
    ) -> Literal["trace", "adjoint"]:
        """Select a suitable exact basis, never assume unproved identities."""

        if (
            self._run_config is None
            or self._run_config.color.contraction.value == "direct"
            or self._run_config.color.fft_basis.value == "trace"
        ):
            return "trace"
        if self._automatic_adjoint_basis_limit(process) is not None:
            return "trace"
        ddm_proven = (
            model is not None
            and not process.fundamental_labels
            and not process.antifundamental_labels
            and len(process.adjoint_labels) >= 2
            and adjoint_tree_color_basis_is_proven(
                model, process, max_coupling_orders=self._coupling_order_limits or None
            )
        )
        return best_general_fft_basis(process, ddm_proven=ddm_proven).actual_basis

    def _automatic_adjoint_basis_limit(self, process: CanonicalProcessIR) -> str | None:
        """Known representation limits must not make automatic requests fail."""

        run = self._run_config
        if run is None or run.color.contraction.value != "auto":
            return None
        if process.fundamental_labels or process.antifundamental_labels:
            return None
        from ..color.adjoint_kernel import MAX_ADJOINT_GLUONS

        labels = tuple(sorted(process.adjoint_labels))
        if len(labels) > MAX_ADJOINT_GLUONS:
            return "The adjoint metric size limit requires the original trace basis."
        reference = self._process_selection.reference_color_order
        if labels and reference and set(reference) == set(labels):
            pivot = reference.index(labels[0])
            if reference[pivot - 1] != labels[-1]:
                return (
                    "The requested colour ordering requires the original trace basis."
                )
        return None

    def _fft_basis_metadata(
        self, color_plan: GenericColorPlan | None
    ) -> dict[str, object]:
        """Expose the actual representation without changing runtime identity."""

        run = self._run_config
        if (
            color_plan is None
            or run is None
            or run.color.contraction.value == "direct"
            or run.color.fft_basis.value != "adjoint"
        ):
            return {}
        strategy = best_general_fft_basis(
            color_plan.process, ddm_proven=color_plan.basis == "adjoint"
        )
        return {
            "fft_basis_selection": {
                "requested": "adjoint",
                "actual_basis": strategy.actual_basis,
                "name": strategy.name,
                "reason": (
                    self._automatic_adjoint_basis_limit(color_plan.process)
                    or strategy.reason
                ),
                "permutation_degree": strategy.permutation_degree,
                "tensor_count": strategy.tensor_count,
            }
        }

    def _plan_concrete_process(
        self,
        process: CanonicalProcessIR,
        *,
        model: Model | None,
    ) -> dict[str, object]:
        selection = self._process_selection
        self._validate_lc_flow_layout(selection)
        if (
            selection.selected_color_sector_ids is not None
            and self._color_accuracy != "lc"
        ):
            raise GenerationError(
                "process.selected_color_sector_ids is available only for LC generation"
            )
        color_plan = build_color_plan(
            process,
            basis=self._fft_color_basis(process, model),
            color_accuracy=self._color_accuracy,
            max_sectors=selection.max_color_sectors,
            reference_color_order=selection.reference_color_order,
            fold_trace_reflections=(
                model is not None
                and model.lc_trace_reflection_equivalence_is_proven(process)
            ),
        )
        color_plan, missing_sector_ids = _restrict_color_plan(
            color_plan,
            selection.selected_color_sector_ids,
        )
        if missing_sector_ids:
            raise GenerationError(
                f"process {process.process!r} did not materialize requested LC "
                "colour sector ids: "
                + ", ".join(str(sector_id) for sector_id in missing_sector_ids)
            )
        if not color_plan.sectors:
            detail = "; ".join(color_plan.diagnostics) or "no color sectors"
            raise GenerationError(
                f"process {process.process!r} has no usable color plan: {detail}"
            )
        return {
            "key": process.key,
            "process": process.process,
            "external_particle_count": len(process.legs),
            "color_sector_count": color_plan.sector_count,
            "color_coverage": (
                "selected"
                if color_plan.truncated
                or selection.selected_color_sector_ids is not None
                else "complete"
            ),
            "helicity_coverage": (
                "selected"
                if selection.selected_source_helicities is not None
                else "complete"
            ),
            "color_diagnostics": tuple(color_plan.diagnostics),
            **self._fft_basis_metadata(color_plan),
            "coupling_order_limits": self._coupling_order_limits,
            "dag_compilation_deferred": True,
        }

    def _plan_on_the_fly_process(
        self,
        projected: _ProjectedOnTheFlyProcess,
    ) -> dict[str, object]:
        process = projected.expanded.process_ir
        census = _on_the_fly_selector_census_v1(
            projected.projection.seed,
            process,
        )
        color_plan = projected.projection.color_plan
        return {
            "key": process.key,
            "process": process.process,
            "external_particle_count": len(process.legs),
            "execution_mode": "on-the-fly",
            "selector_representation": "compact-query-local",
            "physical_helicity_count": census["physical_helicity_count"],
            "physical_color_flow_count": census["physical_color_flow_count"],
            "color_sector_count": census["physical_color_flow_count"],
            "color_coverage": "complete" if color_plan is None else "contracted",
            "helicity_coverage": "complete",
            "color_diagnostics": (
                () if color_plan is None else tuple(color_plan.diagnostics)
            ),
            **self._fft_basis_metadata(color_plan),
            "coupling_order_limits": self._coupling_order_limits,
            "color_plan_materialized": color_plan is not None,
            "source_projection_validated": True,
            "native_seed_construction_deferred": True,
            "dag_compilation_deferred": True,
        }

    def _compile_concrete_process(
        self,
        process: CanonicalProcessIR,
        model: Model,
        *,
        progress_callback: Callable[[Mapping[str, str | int]], None] | None = None,
    ) -> tuple[GenericDAG, dict[str, object]]:
        self._validate_recurrence_request()
        prepared = self._prepare_process_construction(process, model)
        if self._recurrence_execution_enabled:
            raise GenerationError(
                "recurrence execution is not available in this development "
                "milestone: prepared recurrence-template-v1 is validated, but "
                "Rust recurrence builder construction has not been installed"
            )
        complete_color_plan = prepared.complete_color_plan
        materialized_sector_ids = prepared.materialized_sector_ids
        limits = dict(prepared.coupling_order_limits)
        replay_plan = prepared.topology_replay
        selection = self._process_selection
        dag = compile_generic_dag(
            process,
            model=model,
            color_plan=complete_color_plan,
            max_color_sectors=selection.max_color_sectors,
            reference_color_order=selection.reference_color_order,
            selected_color_sector_ids=materialized_sector_ids,
            max_coupling_orders=limits or None,
            max_quark_pairs=self._max_quark_pairs,
            selected_source_helicities=selection.selected_source_helicities,
            online_evaluation_reuse=(
                self._eager_execution_enabled or self._all_flow_union_enabled
            ),
            backward_live_planning=self._eager_execution_enabled,
            progress_callback=progress_callback,
        )
        if replay_plan is not None and replay_plan.optimized:
            root_sector_ids = {
                int(root.color_sector_id)
                if root.color_sector_id is not None
                else int(dag.currents[root.left_id].index.color_state.sector_id)
                for root in dag.amplitude_roots
            }
            missing_representatives = {
                int(partition.materialized_sector_id)
                for partition in replay_plan.partitions
                if int(partition.materialized_sector_id) not in root_sector_ids
            }
            if missing_representatives:
                replay_plan = None
                dag = compile_generic_dag(
                    process,
                    model=model,
                    color_plan=complete_color_plan,
                    max_color_sectors=selection.max_color_sectors,
                    reference_color_order=selection.reference_color_order,
                    selected_color_sector_ids=None,
                    max_coupling_orders=limits or None,
                    max_quark_pairs=self._max_quark_pairs,
                    selected_source_helicities=selection.selected_source_helicities,
                    online_evaluation_reuse=self._eager_execution_enabled,
                    backward_live_planning=self._eager_execution_enabled,
                    progress_callback=progress_callback,
                )
        if replay_plan is not None and replay_plan.optimized:
            dag = replace(
                dag,
                color_plan=complete_color_plan,
                color_coverage="complete",
                selected_color_sector_ids=(),
                lc_topology_replay=(
                    replay_plan
                    if isinstance(replay_plan, LCColorTopologyReplayPlan)
                    else None
                ),
                color_topology_replay=(
                    replay_plan
                    if isinstance(replay_plan, ColorTopologyReplayCertificate)
                    else None
                ),
            )
        if dag.truncated:
            raise GenerationError(
                f"process {process.process!r} DAG was unexpectedly truncated"
            )
        if not dag.has_amplitudes:
            raise _NoModelSupportedAmplitudes(
                f"process {process.process!r} has no model-supported tree-level "
                "amplitudes"
            )
        return (
            dag,
            {
                "key": process.key,
                "process": process.process,
                "color_sector_count": dag.color_plan.sector_count,
                **(
                    {}
                    if self._color_accuracy != "lc"
                    else {
                        "lc_flow_layout": (
                            "all-flow-union"
                            if self._all_flow_union_enabled
                            else "topology-replay"
                        )
                    }
                ),
                **(
                    {}
                    if (
                        dag.lc_topology_replay is None
                        and dag.color_topology_replay is None
                    )
                    else {
                        "materialized_color_sector_count": len(
                            (
                                dag.lc_topology_replay or dag.color_topology_replay
                            ).materialized_sector_ids
                        ),
                        "replayed_color_sector_count": (
                            (
                                dag.lc_topology_replay or dag.color_topology_replay
                            ).replayed_sector_count
                        ),
                        "residual_color_sector_count": len(
                            (
                                dag.lc_topology_replay or dag.color_topology_replay
                            ).residual_sector_ids
                        ),
                    }
                ),
                "color_coverage": dag.color_coverage,
                "helicity_coverage": dag.helicity_coverage,
                "source_count": len(dag.sources),
                "current_count": len(dag.currents),
                "interaction_count": len(dag.interactions),
                "interaction_evaluation_count": dag.interaction_evaluation_count,
                "amplitude_root_count": len(dag.amplitude_roots),
                "coupling_order_limits": limits,
            },
        )

    def _prepare_process_construction(
        self,
        process: CanonicalProcessIR,
        model: Model,
    ) -> _PreparedProcessConstruction:
        """Prepare the shared model/color contract before choosing a builder."""

        selection = self._process_selection
        self._validate_lc_flow_layout(selection)
        if (
            selection.selected_color_sector_ids is not None
            and self._color_accuracy != "lc"
        ):
            raise GenerationError(
                "process.selected_color_sector_ids is available only for LC generation"
            )
        complete_color_plan = build_color_plan(
            process,
            basis=self._fft_color_basis(process, model),
            color_accuracy=self._color_accuracy,
            max_sectors=selection.max_color_sectors,
            reference_color_order=selection.reference_color_order,
            fold_trace_reflections=(
                model.lc_trace_reflection_equivalence_is_proven(process)
            ),
        )
        color_plan, missing_sector_ids = _restrict_color_plan(
            complete_color_plan,
            selection.selected_color_sector_ids,
        )
        if missing_sector_ids:
            raise GenerationError(
                f"process {process.process!r} did not materialize requested LC "
                "colour sector ids: "
                + ", ".join(str(sector_id) for sector_id in missing_sector_ids)
            )
        if not color_plan.sectors:
            detail = "; ".join(color_plan.diagnostics) or "no color sectors"
            raise GenerationError(
                f"process {process.process!r} has no usable color plan: {detail}"
            )
        limits = self._coupling_order_limits
        run = self._run_config
        if run is not None and run.process.coupling_order_policy == "minimal":
            if (
                selection.max_color_sectors is None
                and selection.reference_color_order is None
                and complete_color_plan.color_accuracy == process.color_accuracy
            ):
                inferred = _infer_minimal_coupling_order_limits_from_color_plan(
                    process,
                    model=model,
                    color_plan=complete_color_plan,
                    max_coupling_orders=limits or None,
                )
            else:
                inferred = infer_minimal_coupling_order_limits(
                    process,
                    model=model,
                    max_coupling_orders=limits or None,
                )
            limits = inferred or limits
        replay_plan = None
        materialized_sector_ids = selection.selected_color_sector_ids
        if (
            self._color_accuracy == "lc"
            and not self._all_flow_union_enabled
            and selection.selected_color_sector_ids is None
            and selection.selected_source_helicities is None
        ):
            replay_plan = build_lc_topology_replay_plan(
                complete_color_plan,
                model,
            )
            if replay_plan is not None and replay_plan.optimized:
                materialized_sector_ids = frozenset(replay_plan.materialized_sector_ids)
        elif (
            self._color_accuracy in {"nlc", "full"}
            and selection.selected_color_sector_ids is None
            and selection.selected_source_helicities is None
            and all(abs(int(leg.outgoing_pdg or 0)) == 21 for leg in process.legs)
        ):
            # The first contracted-color consumer is deliberately scoped to
            # pure adjoint amplitudes.  Their relabeling phase is exactly +1,
            # so recurrence, compiled, and eager modes can consume the same
            # certificate without introducing a fermion-crossing convention.
            replay_plan = build_color_topology_replay_certificate(
                complete_color_plan,
                model,
            )
            if replay_plan is not None and replay_plan.optimized:
                materialized_sector_ids = frozenset(replay_plan.materialized_sector_ids)
        return _PreparedProcessConstruction(
            process=process,
            complete_color_plan=complete_color_plan,
            restricted_color_plan=color_plan,
            materialized_sector_ids=materialized_sector_ids,
            coupling_order_limits=MappingProxyType(dict(limits)),
            topology_replay=replay_plan,
        )

    def _validate_lc_flow_layout(self, selection: _ProcessSelection) -> None:
        if not self._all_flow_union_enabled:
            return
        if self._color_accuracy != "lc":
            raise GenerationError(
                "color.lc_flow_layout='all-flow-union' is available only for "
                "LC generation"
            )
        incompatible: list[str] = []
        if selection.max_color_sectors is not None:
            incompatible.append("process.max_color_sectors")
        if not self._recurrence_execution_enabled:
            if selection.selected_color_sector_ids is not None:
                incompatible.append("process.selected_color_sector_ids")
            if selection.selected_source_helicities is not None:
                incompatible.append("process.selected_source_helicities")
        if incompatible:
            raise GenerationError(
                "color.lc_flow_layout='all-flow-union' requires complete runtime "
                "flow and helicity coverage; remove " + ", ".join(incompatible)
            )

    def _compile_complete_color_helicity_selector_union(
        self,
        process: _DagProcess,
        model: Model,
    ) -> GenericDAG | None:
        """Build a bounded all-colour lane for runtime helicity selection.

        The ordinary complete artifact keeps topology replay for fast runtime
        colour selection and helicity sums. Replaying that compact DAG once per
        physical colour sector is inefficient for the complementary
        all-colours/selected-helicity workload, so bounded colour spaces receive
        an auxiliary complete-colour recurrence quotient.
        """

        parent = process.dag
        accuracy = parent.process.color_accuracy
        replay = (
            parent.lc_topology_replay
            if accuracy == "lc"
            else parent.color_topology_replay
            if accuracy in {"nlc", "full"}
            else None
        )
        if (
            self._eager_execution_enabled
            or accuracy not in {"lc", "nlc", "full"}
            or parent.color_coverage != "complete"
            or parent.selected_color_sector_ids
            or parent.selected_source_helicities
            or parent.helicity_coverage != "complete"
            or replay is None
            or len(parent.color_plan.sectors) > _MAX_FUSED_HELICITY_SELECTOR_SECTORS
        ):
            return None

        limits_payload = process.coverage.get("coupling_order_limits", {})
        if not isinstance(limits_payload, Mapping):
            raise GenerationError("process coupling-order coverage is not a mapping")
        limits = {str(name): int(value) for name, value in limits_payload.items()}
        selection = self._process_selection
        union = compile_generic_dag(
            process.expanded.process_ir,
            model=model,
            color_plan=parent.color_plan,
            max_color_sectors=None,
            reference_color_order=selection.reference_color_order,
            selected_color_sector_ids=None,
            max_coupling_orders=limits or None,
            max_quark_pairs=self._max_quark_pairs,
            selected_source_helicities=None,
            online_evaluation_reuse=False,
            backward_live_planning=False,
            progress_callback=None,
        )
        if union.truncated or not union.has_amplitudes:
            raise GenerationError(
                "complete-color helicity selector lane did not produce a "
                "complete amplitude DAG"
            )
        parity_preweighted = any(
            float(root.helicity_weight) > 1.0 for root in union.amplitude_roots
        )
        reduced = (
            prune_dag_to_amplitude_roots(union)
            if parity_preweighted
            else prune_global_helicity_flip_equivalent_roots(union, model)
        )
        recurrence = build_helicity_recurrence_plan(reduced, model)
        if recurrence is None:
            return None
        materialization = materialize_helicity_recurrence(
            reduced,
            recurrence,
        )
        return replace(
            materialization.dag,
            lc_topology_replay=None,
            color_topology_replay=None,
            helicity_recurrence=recurrence,
            helicity_materialization=materialization,
        )


def _runtime_stage_count(schema: RuntimeExpressionSchema) -> int:
    stages = schema.to_mapping().get("stages")
    if isinstance(stages, str | bytes) or not isinstance(stages, Sequence):
        raise GenerationError("runtime expression schema stages must be a sequence")
    return len(stages)


def _helicity_selector_lane_stage_count(
    lane: _HelicitySelectorLane,
) -> int:
    return _runtime_stage_count(lane.runtime_schema) + sum(
        _helicity_selector_lane_stage_count(child) for child in lane.child_lanes
    )


def _helicity_selector_lane_compile_task_count(
    lane: _HelicitySelectorLane,
) -> int:
    return (
        _runtime_stage_count(lane.runtime_schema)
        + 1
        + sum(
            _helicity_selector_lane_compile_task_count(child)
            for child in lane.child_lanes
        )
    )


def _runtime_schema_dag_summary(
    schema: RuntimeExpressionSchema,
) -> dict[str, object]:
    payload = schema.to_mapping()
    current_storage = cast("Mapping[str, object]", payload["current_storage"])
    source_fill = cast("Mapping[str, object]", payload["source_fill"])
    amplitude_stage = cast("Mapping[str, object]", payload["amplitude_stage"])
    stages = cast("Sequence[Mapping[str, object]]", payload["stages"])
    current_slots = cast("Sequence[object]", current_storage["current_slots"])
    return {
        "current_count": len(current_slots),
        "source_count": int(source_fill["source_count"]),
        "interaction_count": sum(int(stage["interaction_count"]) for stage in stages),
        "interaction_evaluation_count": sum(
            int(stage["interaction_evaluation_count"]) for stage in stages
        ),
        "amplitude_root_count": int(amplitude_stage["output_count"]),
        "truncated": False,
    }


def _select_color_ready_processes(
    processes: Sequence[CanonicalProcessIR],
    *,
    color_accuracy: str,
    basis: str = "trace",
) -> tuple[tuple[CanonicalProcessIR, ...], tuple[str, ...]]:
    """Drop structurally impossible children of an external multiparticle request."""

    selected: list[CanonicalProcessIR] = []
    rejected: list[str] = []
    for process in processes:
        color_plan = build_color_plan(
            process,
            color_accuracy=color_accuracy,
            basis=basis,
            max_sectors=1,
        )
        # This is only a structural existence probe. Its deliberately capped
        # plan may be truncated; complete basis construction happens later.
        if color_plan.sectors:
            selected.append(process)
            continue
        detail = "; ".join(color_plan.diagnostics) or "no color sectors"
        rejected.append(f"{process.process}: {detail}")
    return tuple(selected), tuple(rejected)


def _expanded_name(base: str, index: int, count: int) -> str:
    return base if count == 1 else f"{base}_{index}"


def _process_side_permutation_class(
    process: CanonicalProcessIR,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Identify one reusable representative per unordered external side."""

    return (
        tuple(sorted(leg.particle.casefold() for leg in process.initial_legs)),
        tuple(sorted(leg.particle.casefold() for leg in process.final_legs)),
    )


_RelationWarning = Callable[
    [GroupingProcess, GroupingProcess, LegMapping, CanonicalProcessIR],
    str | None,
]


def _grouping_process(entry: _ExpandedProcess) -> GroupingProcess:
    pdgs = tuple(leg.pdg for leg in entry.process_ir.legs)
    if any(pdg is None for pdg in pdgs):
        raise GenerationError(
            f"process {entry.request.name!r} has an unresolved external particle"
        )
    return GroupingProcess(
        process_id=entry.request.name,
        expression=entry.request.expression,
        pdgs=tuple(int(cast(int, pdg)) for pdg in pdgs),
        initial_count=len(entry.process_ir.initial_legs),
    )


def _collapse_flavor_blind_classes(
    entries: Sequence[_ExpandedProcess],
    table: Mapping[int, GroupingParticle],
    *,
    consider_charge_conjugation: bool,
    relation_warning: _RelationWarning,
) -> list[_ExpandedProcess]:
    """Keep the representative of each flavour class and record the others on it.

    Processes with explicit aliases keep their own artifact entry.
    """

    by_name = {entry.request.name: entry for entry in entries}
    members: dict[str, list[GroupMember]] = {}
    warnings: list[str] = []
    for flavour_class in flavor_blind_classes(
        [_grouping_process(entry) for entry in entries],
        table,
        consider_charge_conjugation=consider_charge_conjugation,
    ):
        representative = flavour_class[0]
        for process in flavour_class[1:]:
            if by_name[process.process_id].aliases:
                continue
            mapping = flavor_blind_mapping(
                process,
                representative,
                table,
                consider_charge_conjugation=consider_charge_conjugation,
            )
            warning = relation_warning(
                process,
                representative,
                mapping,
                by_name[representative.process_id].process_ir,
            )
            if warning is not None:
                warnings.append(warning)
            members.setdefault(representative.process_id, []).append(
                GroupMember.identity(replace(process, generated=False)).related(
                    mapping,
                    factor=Fraction(1),
                    basis=GroupingBasis.FLAVOR_BLIND,
                    warning=warning,
                )
            )
    if warnings:
        _LOGGER.warning(
            "flavor-blind grouping assumes %d relation(s) that are not symmetries "
            "of the model vertices allowed by the coupling orders, e.g. %s",
            len(warnings),
            "; ".join(warnings[:3]),
        )
    grouped = {member.process_id for group in members.values() for member in group}
    conjugated = sum(
        member.conjugated for group in members.values() for member in group
    )
    _LOGGER.info(
        "Flavor-blind grouping: %d subprocess%s -> %d generated "
        "representative%s; %d routed member%s (%d by charge conjugation%s); "
        "%d relabelling warning%s",
        len(entries),
        "" if len(entries) == 1 else "es",
        len(entries) - len(grouped),
        "" if len(entries) - len(grouped) == 1 else "s",
        len(grouped),
        "" if len(grouped) == 1 else "s",
        conjugated,
        "" if consider_charge_conjugation else ", disabled",
        len(warnings),
        "" if len(warnings) == 1 else "s",
    )
    return [
        replace(entry, group_members=tuple(members.get(entry.request.name, ())))
        for entry in entries
        if entry.request.name not in grouped
    ]


def _permuted_process_identity(
    process: CanonicalProcessIR,
    representative_to_alias: Sequence[int],
) -> tuple[str, tuple[int, ...]]:
    """Return public process metadata in a side-preserving alias order."""

    legs = process.legs
    if len(representative_to_alias) != len(legs) or sorted(
        representative_to_alias
    ) != list(range(len(legs))):
        raise GenerationError("alias particle permutation is not complete")
    alias_legs: list[ProcessLegIR | None] = [None] * len(legs)
    for representative_index, alias_index in enumerate(representative_to_alias):
        alias_legs[alias_index] = legs[representative_index]
    if any(leg is None for leg in alias_legs):
        raise GenerationError("alias particle permutation is incomplete")
    ordered_legs = tuple(cast(ProcessLegIR, leg) for leg in alias_legs)
    if any(leg.pdg is None for leg in ordered_legs):
        raise GenerationError("process alias has an unresolved external particle")
    initial = " ".join(leg.particle for leg in ordered_legs if leg.is_initial)
    final = " ".join(leg.particle for leg in ordered_legs if leg.is_final)
    if not initial or not final:
        raise GenerationError("process alias must retain initial and final states")
    return (
        f"{initial} > {final}",
        tuple(int(leg.pdg) for leg in ordered_legs if leg.pdg is not None),
    )


def _cpp_optimization_level(value: str) -> int:
    normalized = value.strip().upper()
    if normalized.startswith("-"):
        normalized = normalized[1:]
    if normalized not in {"O0", "O1", "O2", "O3"}:
        raise GenerationError("evaluator.cpp.optimization must be O0, O1, O2, or O3")
    return int(normalized[1])


def create_generator_backend(
    config: GenerationConfig | RunConfig | ConfigResolution | None,
    progress: ProgressSink | None,
    *,
    api_bundle_hook: ApiBundleHook | None = None,
) -> GenerationBackend:
    return GenerationBackend(config, progress, api_bundle_hook=api_bundle_hook)


__all__ = ["GenerationBackend", "create_generator_backend"]
