# SPDX-License-Identifier: 0BSD
"""Read-only summaries of generated process artifacts."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pyamplicol._internal.versions import (
    EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
    ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
    ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
    ON_THE_FLY_RUNTIME_CAPABILITY,
    RECURRENCE_BUILDER_INPUT_ABI,
    RECURRENCE_COLOR_RUNTIME_CAPABILITY,
    RECURRENCE_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
    RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
    RECURRENCE_DIRECT_BACKEND_ABI,
    RECURRENCE_DIRECT_TEMPLATE_ABI,
    RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY,
    RECURRENCE_PLAN_ABI,
    RECURRENCE_RUNTIME_LAYOUT_ABI,
    SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
)
from pyamplicol.api.errors import ArtifactError
from pyamplicol.generation.process_grouping import GroupMember, ProcessGroupMap

from .manifest import MANIFEST_NAME, ArtifactManifest, ArtifactPayloads, load_manifest
from .process_groups import initial_process_groups, read_process_groups
from .security import confined_path, normalize_relative_path

_EAGER_RUNTIME_KIND = "pyamplicol-runtime-eager-execution"
_RECURRENCE_RUNTIME_KIND = "pyamplicol-runtime-recurrence-execution"
_ON_THE_FLY_RUNTIME_KIND = "pyamplicol-runtime-on-the-fly-execution"
_ON_THE_FLY_PUBLIC_METADATA_KIND = "pyamplicol-on-the-fly-public-metadata"
_RETIRED_RECURRENCE_RUNTIME_CAPABILITY = "rusticol.recurrence-runtime.complex-f64.v1"
_RECURRENCE_DIRECT_SCHEDULE_MEMBER_PATH = "schedule/recurrence-direct-schedule-v2.bin"
_RECURRENCE_COLOR_CONTRACTION_CODEC_ABI = "pyamplicol-recurrence-color-contraction-v3"
_EAGER_PLAN_V3_ABI = "pyamplicol-eager-plan-v3"
_EAGER_RUNTIME_CAPABILITIES = frozenset(
    {
        EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        "rusticol.eager-dag.complex-f64.v1",
        "rusticol.eager-runtime-layout.complex-f64.v1",
    }
)
_MISSING_U32 = (1 << 32) - 1
_EAGER_INVOCATION = struct.Struct("<IIIIIIIQQ")
_EAGER_FINALIZATION = struct.Struct("<IIIII")
_EAGER_CLOSURE = struct.Struct("<IIIIIIdd")
_EAGER_SELECTOR_DOMAINS_ABI = "pyamplicol-eager-selector-domains-v1"
_EAGER_SELECTOR_DOMAIN = struct.Struct("<QQ")
_EAGER_SELECTOR_GROUP = struct.Struct("<I")
_EAGER_SELECTOR_DOMAIN_ID = struct.Struct("<I")
_COMPILED_PROFILE_PHASES = (
    "source-fill",
    "momentum-setup",
    "stage-input-pack",
    "stage-evaluator-call",
    "output-assign",
    "amplitude-input-pack",
    "amplitude-evaluator-call",
    "reduction",
)
_EAGER_PROFILE_PHASES = (
    "source-fill",
    "momentum-setup",
    "eager-execution-aggregate",
)
_RECURRENCE_PROFILE_PHASES = (
    "selector-plan",
    "source-fill",
    "momentum-form-fill",
    "recurrence-direct-contribution",
    "recurrence-direct-finalization",
    "recurrence-direct-closure",
    "reduction",
)
# The compact lane feeds the same direct-profile counters from its dynamically
# prepared query family; only the plan materialization boundary differs.
_ON_THE_FLY_PROFILE_PHASES = _RECURRENCE_PROFILE_PHASES


@dataclass(frozen=True, slots=True)
class ArtifactAliasInspection:
    id: str
    expression: str
    representative_id: str
    external_pdgs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ArtifactProcessInspection:
    id: str
    expression: str
    color_accuracy: str
    external_pdgs: tuple[int, ...]
    default: bool
    physical_helicities: int
    computed_helicities: int
    physical_color_components: int
    computed_color_components: int
    helicity_coverage: str
    color_coverage: str
    aliases: tuple[ArtifactAliasInspection, ...]
    selector_provenance: str | None = None
    helicity_runtime_contract: str | None = None
    color_flow_runtime_contract: str | None = None
    generation_specialized_axes: tuple[str, ...] = ()
    selected_source_helicities: tuple[tuple[int, int], ...] = ()
    selected_color_sector_ids: tuple[int, ...] = ()
    helicity_recurrence_status: str | None = None
    helicity_optimized_class_count: int | None = None
    helicity_optimized_current_count: int | None = None
    helicity_residual_current_count: int | None = None
    helicity_optimized_amplitude_class_count: int | None = None
    helicity_residual_amplitude_count: int | None = None
    helicity_materialized_current_count: int | None = None
    helicity_materialized_amplitude_count: int | None = None
    lc_physical_sector_count: int | None = None
    lc_materialized_sector_count: int | None = None
    lc_replayed_sector_count: int | None = None
    lc_residual_sector_count: int | None = None
    lc_flow_layout: str | None = None
    lc_union_sector_count: int | None = None
    execution_mode: str = "compiled"
    prepared_backend: str | None = None
    prepared_kernel_count: int | None = None
    referenced_kernel_count: int | None = None
    invocation_count: int | None = None
    attachment_count: int | None = None
    evaluation_alias_count: int | None = None
    maximum_fanout: int | None = None
    finalization_count: int | None = None
    closure_count: int | None = None
    selector_closure_available: bool = False
    selector_domain_count: int | None = None
    selector_domain_membership_count: int | None = None
    requested_point_tile_size: int | None = None
    effective_point_tile_size: int | None = None
    requested_query_construction_threads: int | None = None
    # Historical field name: this is a soft batching target, not an upper limit.
    workspace_limit_bytes: int | None = None
    workspace_bytes: int | None = None
    arena_semantic_component_count: int | None = None
    arena_component_count: int | None = None
    arena_component_reuse_count: int | None = None
    momentum_form_count: int | None = None
    direct_source_row_count: int | None = None
    direct_contribution_row_count: int | None = None
    direct_finalization_row_count: int | None = None
    direct_closure_row_count: int | None = None
    direct_row_group_count: int | None = None
    recurrence_color_accuracy: str | None = None
    recurrence_color_storage: str | None = None
    recurrence_color_sector_count: int | None = None
    recurrence_color_active_sector_count: int | None = None
    recurrence_color_component_count: int | None = None
    recurrence_color_group_count: int | None = None
    recurrence_color_destination_count: int | None = None
    recurrence_color_entry_count: int | None = None
    recurrence_color_logical_entry_count: int | None = None
    recurrence_color_factorization_kind: str | None = None
    recurrence_color_factorization_rank: int | None = None
    recurrence_color_factorization_coset_count: int | None = None
    recurrence_color_contraction_method: str | None = None
    recurrence_color_fft_degree: int | None = None
    recurrence_color_fft_channel_count: int | None = None
    recurrence_color_fft_covered_local_group_count: int | None = None
    recurrence_color_direct_residual_group_count: int | None = None
    recurrence_color_direct_residual_entry_count: int | None = None
    recurrence_color_raw_kernel_bytes: int | None = None
    recurrence_color_transformed_kernel_bytes: int | None = None
    recurrence_color_fft_capability: str | None = None
    packed_input_bytes: int | None = None
    packed_output_bytes: int | None = None
    scatter_bytes: int | None = None
    native_profile_phases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactDependencyInspection:
    name: str
    version: str
    license: str
    source: str


@dataclass(frozen=True, slots=True)
class ArtifactProcessGroupMemberInspection:
    """One subprocess of a process group and its relation to the representative.

    ``ME(p) = factor * ME_representative(q)`` with ``q[r] = p[permutation[r]]``;
    ``factor`` is the reduced fraction text, such as ``"1"`` or ``"16/9"``.
    """

    id: str
    process: str
    pdgs: tuple[int, ...]
    initial_count: int
    factor: str
    permutation: tuple[int, ...]
    conjugated: bool
    basis: str
    generated: bool
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactProcessGroupInspection:
    """A generated representative and its members, the representative first."""

    representative_id: str
    representative: str
    members: tuple[ArtifactProcessGroupMemberInspection, ...]


@dataclass(frozen=True, slots=True)
class ArtifactProcessGroupsInspection:
    """The artifact's process-group map, in artifact process order.

    ``stored`` is false for an artifact without ``process-groups.json``, whose
    processes are then each their own group.
    """

    stored: bool
    groups: tuple[ArtifactProcessGroupInspection, ...]
    analyses: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactProcessGroupSelection:
    """The process-group relation of one selected process.

    ``role`` is ``representative``, ``routed member`` (not generated, evaluated
    through the representative) or ``generated member`` (related numerically,
    evaluated by its own runtime process). ``permutation`` relates the
    representative's legs to the selected leg order, as in a group member.
    """

    process_id: str
    role: str
    representative_id: str
    representative: str
    factor: str
    permutation: tuple[int, ...]
    conjugated: bool
    basis: str
    warning: str | None
    group_size: int
    routed_count: int
    conjugated_count: int


@dataclass(frozen=True, slots=True)
class ArtifactInspection:
    kind: str
    path: Path
    artifact_kind: str
    artifact_id: str
    created_utc: str
    producer_version: str
    target: str
    cpu_features: tuple[str, ...]
    model_name: str
    model_source: str
    model_restriction: str | None
    default_process_id: str | None
    runtime_engine: str
    runtime_version: str
    runtime_capabilities: tuple[str, ...]
    payload_count: int
    payload_size_bytes: int
    integrity: str
    processes: tuple[ArtifactProcessInspection, ...]
    dependencies: tuple[ArtifactDependencyInspection, ...]
    physical_file_count: int | None = None
    evaluator_container_path: str | None = None
    evaluator_container_abi: str | None = None
    evaluator_container_member_count: int | None = None
    evaluator_container_unpacked_size_bytes: int | None = None
    process_groups: ArtifactProcessGroupsInspection | None = None


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{context} must be an object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} must be an array")
    return value


def _computed(value: object, context: str) -> bool:
    record = _mapping(value, context)
    if record.get("kind") == "contracted-color":
        return True
    computed = record.get("computed")
    if not isinstance(computed, bool):
        raise ArtifactError(f"{context}.computed must be a boolean")
    return computed


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtifactError(f"{context} must be an integer >= {minimum}")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{context} must be a non-empty string")
    return value


# the payload reader of the last inspected manifest, which it keeps alive
_PAYLOAD_READER: list[tuple[ArtifactManifest, ArtifactPayloads]] = []


def _payloads(manifest: ArtifactManifest) -> ArtifactPayloads:
    if not _PAYLOAD_READER or _PAYLOAD_READER[0][0] is not manifest:
        _PAYLOAD_READER[:] = [(manifest, ArtifactPayloads(manifest))]
    return _PAYLOAD_READER[0][1]


def _read_payload(manifest: ArtifactManifest, path: Path) -> bytes:
    """Read a payload at ``path`` below the root, loose or packed."""

    return _payloads(manifest).read_bytes(path.relative_to(manifest.root).as_posix())


def _json_mapping(
    manifest: ArtifactManifest, path: Path, context: str
) -> Mapping[str, object]:
    try:
        value = json.loads(_read_payload(manifest, path))
    except OSError as exc:
        raise ArtifactError(f"cannot read {context}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid {context}: {exc}") from exc
    return _mapping(value, context)


def _artifact_path(
    manifest: ArtifactManifest,
    relative: str | Path,
    context: str,
) -> Path:
    try:
        normalized = normalize_relative_path(Path(relative).as_posix())
    except ArtifactError:
        raise
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{context} is not a valid artifact path") from exc
    if _payloads(manifest).is_packed(normalized):
        return manifest.root / normalized
    return confined_path(manifest.root, normalized)


def _execution_paths(manifest: ArtifactManifest) -> Mapping[str, Path]:
    evaluator_relative = _string(
        manifest.runtime.get("evaluator_manifest_path"),
        "runtime.evaluator_manifest_path",
    )
    evaluator_path = _artifact_path(
        manifest,
        evaluator_relative,
        "runtime.evaluator_manifest_path",
    )
    evaluator_set = _json_mapping(
        manifest, evaluator_path, "runtime evaluator manifest"
    )
    records = _sequence(
        evaluator_set.get("processes"),
        "runtime evaluator manifest.processes",
    )
    base = Path(evaluator_relative).parent
    result: dict[str, Path] = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, f"runtime evaluator manifest.processes[{index}]")
        process_id = _string(
            record.get("process_id"),
            f"runtime evaluator manifest.processes[{index}].process_id",
        )
        manifest_path = _string(
            record.get("manifest_path"),
            f"runtime evaluator manifest.processes[{index}].manifest_path",
        )
        if process_id in result:
            raise ArtifactError(
                f"runtime evaluator manifest repeats process {process_id!r}"
            )
        result[process_id] = _artifact_path(
            manifest,
            base / manifest_path,
            f"runtime evaluator manifest for {process_id!r}",
        )
    expected = {str(process["id"]) for process in manifest.processes}
    if set(result) != expected:
        raise ArtifactError(
            "runtime evaluator manifest process IDs do not match artifact processes"
        )
    return result


@dataclass(frozen=True, slots=True)
class _ExecutionInspection:
    execution_mode: str
    physical_helicity_count: int | None = None
    physical_color_flow_count: int | None = None
    prepared_backend: str | None = None
    prepared_kernel_count: int | None = None
    referenced_kernel_count: int | None = None
    invocation_count: int | None = None
    attachment_count: int | None = None
    evaluation_alias_count: int | None = None
    maximum_fanout: int | None = None
    finalization_count: int | None = None
    closure_count: int | None = None
    selector_closure_available: bool = False
    selector_domain_count: int | None = None
    selector_domain_membership_count: int | None = None
    requested_point_tile_size: int | None = None
    effective_point_tile_size: int | None = None
    requested_query_construction_threads: int | None = None
    # Historical field name: this is a soft batching target, not an upper limit.
    workspace_limit_bytes: int | None = None
    workspace_bytes: int | None = None
    arena_semantic_component_count: int | None = None
    arena_component_count: int | None = None
    arena_component_reuse_count: int | None = None
    momentum_form_count: int | None = None
    direct_source_row_count: int | None = None
    direct_contribution_row_count: int | None = None
    direct_finalization_row_count: int | None = None
    direct_closure_row_count: int | None = None
    direct_row_group_count: int | None = None
    recurrence_color_accuracy: str | None = None
    recurrence_color_storage: str | None = None
    recurrence_color_sector_count: int | None = None
    recurrence_color_active_sector_count: int | None = None
    recurrence_color_component_count: int | None = None
    recurrence_color_group_count: int | None = None
    recurrence_color_destination_count: int | None = None
    recurrence_color_entry_count: int | None = None
    recurrence_color_logical_entry_count: int | None = None
    recurrence_color_factorization_kind: str | None = None
    recurrence_color_factorization_rank: int | None = None
    recurrence_color_factorization_coset_count: int | None = None
    recurrence_color_contraction_method: str | None = None
    recurrence_color_fft_degree: int | None = None
    recurrence_color_fft_channel_count: int | None = None
    recurrence_color_fft_covered_local_group_count: int | None = None
    recurrence_color_direct_residual_group_count: int | None = None
    recurrence_color_direct_residual_entry_count: int | None = None
    recurrence_color_raw_kernel_bytes: int | None = None
    recurrence_color_transformed_kernel_bytes: int | None = None
    recurrence_color_fft_capability: str | None = None
    packed_input_bytes: int | None = None
    packed_output_bytes: int | None = None
    scatter_bytes: int | None = None
    native_profile_phases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _PhysicsInspection:
    physical_helicities: int
    computed_helicities: int
    physical_color_components: int
    computed_color_components: int
    helicity_coverage: str
    color_coverage: str
    selector_provenance: str | None = None
    helicity_runtime_contract: str | None = None
    color_flow_runtime_contract: str | None = None
    generation_specialized_axes: tuple[str, ...] = ()
    selected_source_helicities: tuple[tuple[int, int], ...] = ()
    selected_color_sector_ids: tuple[int, ...] = ()
    helicity_recurrence_status: str | None = None
    helicity_optimized_class_count: int | None = None
    helicity_optimized_current_count: int | None = None
    helicity_residual_current_count: int | None = None
    helicity_optimized_amplitude_class_count: int | None = None
    helicity_residual_amplitude_count: int | None = None
    helicity_materialized_current_count: int | None = None
    helicity_materialized_amplitude_count: int | None = None
    lc_physical_sector_count: int | None = None
    lc_materialized_sector_count: int | None = None
    lc_replayed_sector_count: int | None = None
    lc_residual_sector_count: int | None = None


def _table_record(
    value: object,
    context: str,
) -> tuple[str, int, int]:
    record = _mapping(value, context)
    return (
        _string(record.get("path"), f"{context}.path"),
        _integer(record.get("count"), f"{context}.count"),
        _integer(record.get("row_size"), f"{context}.row_size", minimum=1),
    )


def _table_rows(
    manifest: ArtifactManifest,
    execution_root: Path,
    table: object,
    layout: struct.Struct,
    context: str,
) -> tuple[tuple[int | float, ...], ...]:
    relative, count, row_size = _table_record(table, context)
    if row_size != layout.size:
        raise ArtifactError(f"{context}.row_size is {row_size}, expected {layout.size}")
    path = _artifact_path(
        manifest,
        execution_root.relative_to(manifest.root) / relative,
        f"{context}.path",
    )
    try:
        payload = _read_payload(manifest, path)
    except OSError as exc:
        raise ArtifactError(f"cannot read {context} payload: {exc}") from exc
    expected_size = count * row_size
    if len(payload) != expected_size:
        raise ArtifactError(
            f"{context} declares {count} rows but has {len(payload)} bytes"
        )
    return cast(
        tuple[tuple[int | float, ...], ...],
        tuple(tuple(row) for row in layout.iter_unpack(payload)),
    )


def _optional_positive_integer(
    value: object,
    context: str,
) -> int | None:
    if value is None:
        return None
    return _integer(value, context, minimum=1)


def _profile_phases(execution: Mapping[str, object]) -> tuple[str, ...]:
    raw = execution.get("native_profile_phases")
    if raw is None:
        return _EAGER_PROFILE_PHASES
    phases = tuple(
        _string(value, f"execution.native_profile_phases[{index}]")
        for index, value in enumerate(_sequence(raw, "execution.native_profile_phases"))
    )
    if len(phases) != len(set(phases)):
        raise ArtifactError("execution.native_profile_phases contains duplicates")
    return phases


def _selector_domain_ids(
    manifest: ArtifactManifest,
    execution_root: Path,
    table: object,
    *,
    expected_count: int,
    domain_count: int,
    context: str,
) -> tuple[int, ...]:
    rows = _table_rows(
        manifest,
        execution_root,
        table,
        _EAGER_SELECTOR_DOMAIN_ID,
        context,
    )
    if len(rows) != expected_count:
        raise ArtifactError(
            f"{context} has {len(rows)} rows, expected {expected_count}"
        )
    result = tuple(int(row[0]) for row in rows)
    invalid = next((value for value in result if value >= domain_count), None)
    if invalid is not None:
        raise ArtifactError(f"{context} references unknown domain {invalid}")
    return result


def _selector_domain_inspection(
    manifest: ArtifactManifest,
    execution: Mapping[str, object],
    execution_root: Path,
    selector_closures: object,
    *,
    stages: Sequence[
        tuple[
            Mapping[str, object],
            tuple[tuple[int | float, ...], ...],
            int,
            tuple[tuple[int | float, ...], ...],
        ]
    ],
    closures: tuple[tuple[int | float, ...], ...],
) -> tuple[int, int]:
    selector = _mapping(
        selector_closures,
        "eager execution.plan.selector_closures",
    )
    abi = _string(
        selector.get("abi"),
        "eager execution.plan.selector_closures.abi",
    )
    if abi != _EAGER_SELECTOR_DOMAINS_ABI:
        raise ArtifactError(f"unsupported eager selector-domain ABI {abi!r}")

    domain_rows = _table_rows(
        manifest,
        execution_root,
        selector.get("domains"),
        _EAGER_SELECTOR_DOMAIN,
        "eager execution.plan.selector_closures.domains",
    )
    group_rows = _table_rows(
        manifest,
        execution_root,
        selector.get("domain_group_ids"),
        _EAGER_SELECTOR_GROUP,
        "eager execution.plan.selector_closures.domain_group_ids",
    )
    memberships: list[frozenset[int]] = []
    cursor = 0
    for domain_id, row in enumerate(domain_rows):
        start, count = int(row[0]), int(row[1])
        if start != cursor:
            raise ArtifactError(
                f"eager selector domain {domain_id} starts at {start}, "
                f"expected {cursor}"
            )
        stop = start + count
        if stop > len(group_rows):
            raise ArtifactError(
                f"eager selector domain {domain_id} exceeds its membership table"
            )
        members = tuple(int(member[0]) for member in group_rows[start:stop])
        if members != tuple(sorted(set(members))):
            raise ArtifactError(
                f"eager selector domain {domain_id} members are not unique and sorted"
            )
        memberships.append(frozenset(members))
        cursor = stop
    if cursor != len(group_rows):
        raise ArtifactError(
            "eager selector domains do not cover their membership table"
        )
    if len(memberships) != len(set(memberships)):
        raise ArtifactError("eager selector domains contain duplicate memberships")

    runtime_schema = _mapping(
        execution.get("runtime_schema"), "eager execution.runtime_schema"
    )
    amplitude_stage = _mapping(
        runtime_schema.get("amplitude_stage"),
        "eager execution.runtime_schema.amplitude_stage",
    )
    roots = _sequence(
        amplitude_stage.get("roots"),
        "eager execution.runtime_schema.amplitude_stage.roots",
    )
    groups_by_output: dict[int, int] = {}
    for index, raw_root in enumerate(roots):
        root = _mapping(
            raw_root,
            f"eager execution.runtime_schema.amplitude_stage.roots[{index}]",
        )
        output_index = _integer(
            root.get("output_index"),
            f"eager execution.runtime_schema.amplitude_stage.roots[{index}]"
            ".output_index",
        )
        group_id = _integer(
            root.get("coherent_group_id"),
            f"eager execution.runtime_schema.amplitude_stage.roots[{index}]"
            ".coherent_group_id",
        )
        if output_index in groups_by_output:
            raise ArtifactError(
                f"eager amplitude roots repeat output index {output_index}"
            )
        groups_by_output[output_index] = group_id

    declared_groups = set(groups_by_output.values())
    unknown_groups = (
        set().union(*memberships) - declared_groups if memberships else set()
    )
    if unknown_groups:
        rendered = ", ".join(str(value) for value in sorted(unknown_groups))
        raise ArtifactError(
            f"eager selector domains reference undeclared coherent groups: {rendered}"
        )

    selector_stages = _sequence(
        selector.get("stages"),
        "eager execution.plan.selector_closures.stages",
    )
    if len(selector_stages) != len(stages):
        raise ArtifactError("eager selector domains do not cover every stage")
    domain_count = len(memberships)
    for index, (raw_selector_stage, execution_stage) in enumerate(
        zip(selector_stages, stages, strict=True)
    ):
        selector_stage = _mapping(
            raw_selector_stage,
            f"eager execution.plan.selector_closures.stages[{index}]",
        )
        stage, invocations, attachment_count, finalizations = execution_stage
        stage_index = _integer(
            stage.get("stage_index"),
            f"eager execution.plan.stages[{index}].stage_index",
        )
        selector_stage_index = _integer(
            selector_stage.get("stage_index"),
            f"eager execution.plan.selector_closures.stages[{index}].stage_index",
        )
        if selector_stage_index != stage_index:
            raise ArtifactError("eager selector-domain stage index mismatch")
        invocation_domains = _selector_domain_ids(
            manifest,
            execution_root,
            selector_stage.get("invocation_domains"),
            expected_count=len(invocations),
            domain_count=domain_count,
            context=(
                f"eager execution.plan.selector_closures.stages[{index}]"
                ".invocation_domains"
            ),
        )
        attachment_domains = _selector_domain_ids(
            manifest,
            execution_root,
            selector_stage.get("attachment_domains"),
            expected_count=attachment_count,
            domain_count=domain_count,
            context=(
                f"eager execution.plan.selector_closures.stages[{index}]"
                ".attachment_domains"
            ),
        )
        unpropagated_domains = _selector_domain_ids(
            manifest,
            execution_root,
            selector_stage.get("unpropagated_finalization_domains"),
            expected_count=len(finalizations),
            domain_count=domain_count,
            context=(
                f"eager execution.plan.selector_closures.stages[{index}]"
                ".unpropagated_finalization_domains"
            ),
        )
        propagated_domains = _selector_domain_ids(
            manifest,
            execution_root,
            selector_stage.get("propagated_finalization_domains"),
            expected_count=len(finalizations),
            domain_count=domain_count,
            context=(
                f"eager execution.plan.selector_closures.stages[{index}]"
                ".propagated_finalization_domains"
            ),
        )

        for invocation, domain_id in zip(invocations, invocation_domains, strict=True):
            start, count = int(invocation[7]), int(invocation[8])
            stop = start + count
            if stop > len(attachment_domains):
                raise ArtifactError(
                    "eager invocation selector range exceeds attachment domains"
                )
            expected_domain = frozenset(
                group_id
                for attachment_domain_id in attachment_domains[start:stop]
                for group_id in memberships[attachment_domain_id]
            )
            if memberships[domain_id] != expected_domain:
                raise ArtifactError(
                    "eager invocation selector domain does not equal its "
                    "attachment-domain union"
                )
        for finalization, unpropagated, propagated in zip(
            finalizations,
            unpropagated_domains,
            propagated_domains,
            strict=True,
        ):
            if int(finalization[2]) == _MISSING_U32 and memberships[unpropagated]:
                raise ArtifactError(
                    "missing eager unpropagated output has a nonempty selector domain"
                )
            if int(finalization[3]) == _MISSING_U32 and memberships[propagated]:
                raise ArtifactError(
                    "missing eager propagated output has a nonempty selector domain"
                )

    closure_domains = _selector_domain_ids(
        manifest,
        execution_root,
        selector.get("closure_domains"),
        expected_count=len(closures),
        domain_count=domain_count,
        context="eager execution.plan.selector_closures.closure_domains",
    )
    for closure, domain_id in zip(closures, closure_domains, strict=True):
        output_index = int(closure[3])
        try:
            expected = frozenset((groups_by_output[output_index],))
        except KeyError as exc:
            raise ArtifactError(
                f"eager closure references unknown amplitude output {output_index}"
            ) from exc
        if memberships[domain_id] != expected:
            raise ArtifactError(
                "eager closure selector domain does not match its coherent group"
            )

    return domain_count, len(group_rows)


def _eager_execution_inspection(
    manifest: ArtifactManifest,
    execution: Mapping[str, object],
    execution_path: Path,
) -> _ExecutionInspection:
    if execution.get("eager_plan_abi") == _EAGER_PLAN_V3_ABI:
        return _eager_v3_execution_inspection(manifest, execution)
    runtime_options = _mapping(
        execution.get("runtime_options"), "eager execution.runtime_options"
    )
    requested_tile = _integer(
        runtime_options.get("point_tile_size"),
        "eager execution.runtime_options.point_tile_size",
        minimum=1,
    )
    workspace_mib = _integer(
        runtime_options.get("workspace_mib"),
        "eager execution.runtime_options.workspace_mib",
        minimum=1,
    )
    effective_tile = _optional_positive_integer(
        runtime_options.get("effective_point_tile_size"),
        "eager execution.runtime_options.effective_point_tile_size",
    )
    workspace_bytes = _optional_positive_integer(
        runtime_options.get("workspace_bytes"),
        "eager execution.runtime_options.workspace_bytes",
    )
    plan = _mapping(execution.get("plan"), "eager execution.plan")
    stages = _sequence(plan.get("stages"), "eager execution.plan.stages")
    execution_root = execution_path.parent

    invocation_count = 0
    attachment_count = 0
    finalization_count = 0
    maximum_fanout = 0
    referenced_kernel_ids: set[int] = set()
    inspected_stages: list[
        tuple[
            Mapping[str, object],
            tuple[tuple[int | float, ...], ...],
            int,
            tuple[tuple[int | float, ...], ...],
        ]
    ] = []
    for index, raw_stage in enumerate(stages):
        stage = _mapping(raw_stage, f"eager execution.plan.stages[{index}]")
        invocations = _table_rows(
            manifest,
            execution_root,
            stage.get("invocations"),
            _EAGER_INVOCATION,
            f"eager execution.plan.stages[{index}].invocations",
        )
        _attachment_path, stage_attachments, _attachment_size = _table_record(
            stage.get("attachments"),
            f"eager execution.plan.stages[{index}].attachments",
        )
        finalizations = _table_rows(
            manifest,
            execution_root,
            stage.get("finalizations"),
            _EAGER_FINALIZATION,
            f"eager execution.plan.stages[{index}].finalizations",
        )
        inspected_stages.append((stage, invocations, stage_attachments, finalizations))
        invocation_count += len(invocations)
        attachment_count += stage_attachments
        finalization_count += len(finalizations)
        for row in invocations:
            referenced_kernel_ids.add(int(row[0]))
            maximum_fanout = max(maximum_fanout, int(row[-1]))
        referenced_kernel_ids.update(
            int(row[0]) for row in finalizations if int(row[0]) != _MISSING_U32
        )

    closures = _table_rows(
        manifest,
        execution_root,
        plan.get("closures"),
        _EAGER_CLOSURE,
        "eager execution.plan.closures",
    )
    referenced_kernel_ids.update(
        int(row[0]) for row in closures if int(row[0]) != _MISSING_U32
    )
    if attachment_count < invocation_count:
        raise ArtifactError(
            "eager execution has fewer attachments than canonical invocations"
        )

    kernel_pack = _mapping(execution.get("kernel_pack"), "eager execution.kernel_pack")
    pack_path = _artifact_path(
        manifest,
        _string(
            kernel_pack.get("manifest_path"),
            "eager execution.kernel_pack.manifest_path",
        ),
        "eager execution.kernel_pack.manifest_path",
    )
    pack = _json_mapping(manifest, pack_path, "prepared eager kernel pack")
    kernels = _sequence(pack.get("kernels"), "prepared eager kernel pack.kernels")
    kernel_ids = {
        _integer(
            _mapping(raw, f"prepared eager kernel pack.kernels[{index}]").get(
                "kernel_id"
            ),
            f"prepared eager kernel pack.kernels[{index}].kernel_id",
        )
        for index, raw in enumerate(kernels)
    }
    if not referenced_kernel_ids <= kernel_ids:
        missing = ", ".join(
            str(value) for value in sorted(referenced_kernel_ids - kernel_ids)
        )
        raise ArtifactError(
            "eager execution references kernels absent from its prepared pack: "
            f"{missing}"
        )

    selector_closures = plan.get("selector_closures")
    selector_closure_available = False
    selector_domain_count = None
    selector_domain_membership_count = None
    if selector_closures is not None:
        (
            selector_domain_count,
            selector_domain_membership_count,
        ) = _selector_domain_inspection(
            manifest,
            execution,
            execution_root,
            selector_closures,
            stages=tuple(inspected_stages),
            closures=closures,
        )
        selector_closure_available = True

    return _ExecutionInspection(
        execution_mode="eager",
        prepared_backend=_string(
            pack.get("backend"), "prepared eager kernel pack.backend"
        ),
        prepared_kernel_count=len(kernels),
        referenced_kernel_count=len(referenced_kernel_ids),
        invocation_count=invocation_count,
        attachment_count=attachment_count,
        evaluation_alias_count=attachment_count - invocation_count,
        maximum_fanout=maximum_fanout,
        finalization_count=finalization_count,
        closure_count=len(closures),
        selector_closure_available=selector_closure_available,
        selector_domain_count=selector_domain_count,
        selector_domain_membership_count=selector_domain_membership_count,
        requested_point_tile_size=requested_tile,
        effective_point_tile_size=effective_tile,
        workspace_limit_bytes=workspace_mib * 1024 * 1024,
        workspace_bytes=workspace_bytes,
        native_profile_phases=_profile_phases(execution),
    )


def _eager_v3_execution_inspection(
    manifest: ArtifactManifest,
    execution: Mapping[str, object],
) -> _ExecutionInspection:
    runtime_options = _mapping(
        execution.get("runtime_options"), "eager execution.runtime_options"
    )
    requested_tile = _integer(
        runtime_options.get("point_tile_size"),
        "eager execution.runtime_options.point_tile_size",
        minimum=1,
    )
    workspace_mib = _integer(
        runtime_options.get("workspace_mib"),
        "eager execution.runtime_options.workspace_mib",
        minimum=1,
    )
    plan = _mapping(execution.get("plan"), "eager execution.plan")
    summary = _mapping(
        plan.get("inspection_summary"),
        "eager execution.plan.inspection_summary",
    )
    pack_reference = _mapping(
        execution.get("kernel_pack"), "eager execution.kernel_pack"
    )
    pack_path = _artifact_path(
        manifest,
        _string(
            pack_reference.get("manifest_path"),
            "eager execution.kernel_pack.manifest_path",
        ),
        "eager execution.kernel_pack.manifest_path",
    )
    pack = _json_mapping(manifest, pack_path, "prepared eager kernel pack")
    kernels = _sequence(pack.get("kernels"), "prepared eager kernel pack.kernels")
    invocation_count = _integer(
        summary.get("invocation_count"),
        "eager inspection invocation_count",
    )
    attachment_count = _integer(
        summary.get("attachment_count"),
        "eager inspection attachment_count",
    )
    if attachment_count < invocation_count:
        raise ArtifactError(
            "eager execution has fewer attachments than canonical invocations"
        )
    selector_domain_count = _integer(
        summary.get("selector_domain_count"),
        "eager inspection selector_domain_count",
    )
    return _ExecutionInspection(
        execution_mode="eager",
        prepared_backend=_string(
            pack.get("backend"), "prepared eager kernel pack.backend"
        ),
        prepared_kernel_count=len(kernels),
        invocation_count=invocation_count,
        attachment_count=attachment_count,
        evaluation_alias_count=attachment_count - invocation_count,
        finalization_count=_integer(
            summary.get("finalization_count"),
            "eager inspection finalization_count",
        ),
        closure_count=_integer(
            summary.get("closure_count"), "eager inspection closure_count"
        ),
        selector_closure_available=selector_domain_count > 0,
        selector_domain_count=selector_domain_count,
        requested_point_tile_size=requested_tile,
        workspace_limit_bytes=workspace_mib * 1024 * 1024,
        native_profile_phases=_profile_phases(execution),
    )


def _recurrence_execution_inspection(
    manifest: ArtifactManifest,
    execution: Mapping[str, object],
) -> _ExecutionInspection:
    declared_capabilities = frozenset(
        str(value)
        for value in _sequence(
            execution.get("required_runtime_capabilities"),
            "recurrence execution.required_runtime_capabilities",
        )
    )
    _require_contract(
        execution,
        "builder_input_abi",
        RECURRENCE_BUILDER_INPUT_ABI,
        "recurrence execution",
    )
    _require_contract(
        execution,
        "recurrence_plan_abi",
        RECURRENCE_PLAN_ABI,
        "recurrence execution",
    )
    _require_contract(
        execution,
        "runtime_layout_abi",
        RECURRENCE_RUNTIME_LAYOUT_ABI,
        "recurrence execution",
    )
    _require_contract(
        execution,
        "direct_template_abi",
        RECURRENCE_DIRECT_TEMPLATE_ABI,
        "recurrence execution",
    )
    _require_contract(
        execution,
        "direct_backend_abi",
        RECURRENCE_DIRECT_BACKEND_ABI,
        "recurrence execution",
    )
    runtime_options = _mapping(
        execution.get("runtime_options"),
        "recurrence execution.runtime_options",
    )
    requested_tile = _integer(
        runtime_options.get("point_tile_size"),
        "recurrence execution.runtime_options.point_tile_size",
        minimum=1,
    )
    workspace_mib = _integer(
        runtime_options.get("workspace_mib"),
        "recurrence execution.runtime_options.workspace_mib",
        minimum=1,
    )
    plan = _mapping(execution.get("plan"), "recurrence execution.plan")
    _require_contract(
        plan,
        "builder_input_abi",
        RECURRENCE_BUILDER_INPUT_ABI,
        "recurrence execution.plan",
    )
    _require_contract(
        plan,
        "recurrence_plan_abi",
        RECURRENCE_PLAN_ABI,
        "recurrence execution.plan",
    )
    _require_contract(
        plan,
        "runtime_layout_abi",
        RECURRENCE_RUNTIME_LAYOUT_ABI,
        "recurrence execution.plan",
    )
    _require_contract(
        plan,
        "direct_template_abi",
        RECURRENCE_DIRECT_TEMPLATE_ABI,
        "recurrence execution.plan",
    )
    _require_contract(
        plan,
        "direct_backend_abi",
        RECURRENCE_DIRECT_BACKEND_ABI,
        "recurrence execution.plan",
    )
    runtime_schedule = _mapping(
        plan.get("runtime_schedule"),
        "recurrence execution.plan.runtime_schedule",
    )
    _require_contract(
        runtime_schedule,
        "plan_member_path",
        _RECURRENCE_DIRECT_SCHEDULE_MEMBER_PATH,
        "recurrence execution.plan.runtime_schedule",
    )
    binding = _mapping(
        plan.get("process_binding"),
        "recurrence execution.plan.process_binding",
    )
    _require_contract(
        binding,
        "abi",
        "pyamplicol-recurrence-process-binding-v4",
        "recurrence execution.plan.process_binding",
    )
    primary_process_digest = _string(
        binding.get("process_digest"),
        "recurrence execution.plan.process_binding.process_digest",
    )
    if len(primary_process_digest) != 64 or any(
        character not in "0123456789abcdef" for character in primary_process_digest
    ):
        raise ArtifactError(
            "recurrence primary process binding has an invalid process digest"
        )
    companion_process_digest = execution.get("process_digest")
    if (
        companion_process_digest is not None
        and companion_process_digest != primary_process_digest
    ):
        raise ArtifactError(
            "recurrence companion process digest disagrees with its primary binding"
        )
    summary = _mapping(
        plan.get("inspection_summary"),
        "recurrence execution.plan.inspection_summary",
    )
    if (
        _string(
            summary.get("execution_mode"),
            "recurrence inspection execution_mode",
        )
        != "recurrence"
    ):
        raise ArtifactError(
            "recurrence inspection summary has the wrong execution mode"
        )
    schedule = _mapping(
        summary.get("schedule"),
        "recurrence inspection schedule",
    )
    direct_arena_raw = summary.get("direct_arena")
    direct_arena = (
        {}
        if direct_arena_raw is None
        else _mapping(direct_arena_raw, "recurrence inspection direct_arena")
    )
    pack_reference = _mapping(
        execution.get("kernel_pack"),
        "recurrence execution.kernel_pack",
    )
    pack_path = _artifact_path(
        manifest,
        _string(
            pack_reference.get("manifest_path"),
            "recurrence execution.kernel_pack.manifest_path",
        ),
        "recurrence execution.kernel_pack.manifest_path",
    )
    pack = _json_mapping(manifest, pack_path, "prepared recurrence kernel pack")
    kernels = _sequence(
        pack.get("kernels"),
        "prepared recurrence kernel pack.kernels",
    )
    prepared_kernel_count = _integer(
        summary.get("prepared_kernel_count"),
        "recurrence inspection prepared_kernel_count",
    )
    kernel_ids = {
        _integer(
            _mapping(
                value,
                f"prepared recurrence kernel pack.kernels[{index}]",
            ).get("kernel_id"),
            f"prepared recurrence kernel pack.kernels[{index}].kernel_id",
        )
        for index, value in enumerate(kernels)
    }
    direct_catalog = _mapping(
        pack.get("recurrence_direct_template"),
        "prepared recurrence kernel pack.recurrence_direct_template",
    )
    direct_templates = _sequence(
        direct_catalog.get("templates"),
        "prepared recurrence direct templates",
    )
    referenced_kernel_ids: set[int] = set()
    for index, value in enumerate(direct_templates):
        template = _mapping(
            value,
            f"prepared recurrence direct templates[{index}]",
        )
        binding = _mapping(
            template.get("payload_binding"),
            f"prepared recurrence direct templates[{index}].payload_binding",
        )
        kernel_id = binding.get("prepared_kernel_id")
        if kernel_id is None:
            continue
        parsed_kernel_id = _integer(
            kernel_id,
            f"prepared recurrence direct templates[{index}].prepared_kernel_id",
        )
        if parsed_kernel_id not in kernel_ids:
            raise ArtifactError(
                "recurrence direct template references an absent prepared kernel"
            )
        referenced_kernel_ids.add(parsed_kernel_id)
    complete_companion_catalog = (
        RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY
        in declared_capabilities
    )
    if (
        not complete_companion_catalog
        and prepared_kernel_count != len(referenced_kernel_ids)
    ) or (
        complete_companion_catalog
        and not 0 < prepared_kernel_count <= len(referenced_kernel_ids)
    ):
        raise ArtifactError(
            "recurrence inspection prepared-kernel count does not match its "
            "direct template catalog"
        )
    semantic_components = _optional_counter(
        direct_arena,
        "semantic_component_count",
        "recurrence inspection direct_arena.semantic_component_count",
    )
    arena_components = _optional_counter(
        direct_arena,
        "current_arena_components",
        "recurrence inspection direct_arena.current_arena_components",
    )
    arena_reuse = _optional_counter(
        direct_arena,
        "arena_component_reuse_count",
        "recurrence inspection direct_arena.arena_component_reuse_count",
    )
    if (
        semantic_components is not None
        and arena_components is not None
        and arena_reuse is not None
        and semantic_components - arena_components != arena_reuse
    ):
        raise ArtifactError(
            "recurrence direct-arena component reuse count is inconsistent"
        )
    packed_input_bytes = _optional_counter(
        direct_arena,
        "packed_input_bytes",
        "recurrence inspection direct_arena.packed_input_bytes",
    )
    packed_output_bytes = _optional_counter(
        direct_arena,
        "packed_output_bytes",
        "recurrence inspection direct_arena.packed_output_bytes",
    )
    scatter_bytes = _optional_counter(
        direct_arena,
        "scatter_bytes",
        "recurrence inspection direct_arena.scatter_bytes",
    )
    for label, value in (
        ("packed-input bytes", packed_input_bytes),
        ("packed-output bytes", packed_output_bytes),
        ("scatter bytes", scatter_bytes),
    ):
        if value not in {None, 0}:
            raise ArtifactError(f"recurrence direct-arena {label} must be zero")
    source_rows = _optional_counter(
        schedule,
        "source_row_count",
        "recurrence inspection source row count",
    )
    contribution_rows = _optional_counter(
        schedule,
        "contribution_count",
        "recurrence inspection contribution count",
    )
    finalization_rows = _optional_counter(
        schedule,
        "finalization_count",
        "recurrence inspection finalization count",
    )
    closure_rows = _optional_counter(
        schedule,
        "closure_term_count",
        "recurrence inspection closure row count",
    )
    runtime_metadata = _mapping(
        execution.get("runtime_metadata"),
        "recurrence execution.runtime_metadata",
    )
    color_reference_raw = runtime_metadata.get("color_contraction")
    color_accuracy = None
    color_storage = None
    color_sector_count = None
    color_active_sector_count = None
    color_component_count = None
    color_group_count = None
    color_destination_count = None
    color_entry_count = None
    color_logical_entry_count = None
    color_factorization_kind = None
    color_factorization_rank = None
    color_factorization_coset_count = None
    color_fft_provenance = None
    if color_reference_raw is not None:
        color_reference = _mapping(
            color_reference_raw,
            "recurrence execution.runtime_metadata.color_contraction",
        )
        _require_contract(
            color_reference,
            "abi",
            _RECURRENCE_COLOR_CONTRACTION_CODEC_ABI,
            "recurrence execution.runtime_metadata.color_contraction",
        )
        color_accuracy = _string(
            color_reference.get("color_accuracy"),
            "recurrence color-contraction accuracy",
        )
        if color_accuracy not in {"nlc", "full"}:
            raise ArtifactError(
                "recurrence color-contraction accuracy must be nlc or full"
            )
        color_storage = _string(
            color_reference.get("storage"),
            "recurrence color-contraction storage",
        )
        if color_storage not in {"expanded", "repeated", "convolution-kernels"}:
            raise ArtifactError(
                "recurrence color-contraction storage must be expanded, repeated, "
                "or convolution-kernels"
            )
        color_sector_count = _integer(
            color_reference.get("sector_count"),
            "recurrence color-contraction sector count",
            minimum=1,
        )
        color_active_sector_count = _integer(
            color_reference.get("active_sector_count"),
            "recurrence color-contraction active sector count",
            minimum=1,
        )
        color_component_count = _integer(
            color_reference.get("component_count"),
            "recurrence color-contraction component count",
            minimum=1,
        )
        color_group_count = _integer(
            color_reference.get("group_count"),
            "recurrence color-contraction group count",
            minimum=1,
        )
        color_destination_count = _integer(
            color_reference.get("destination_count"),
            "recurrence color-contraction destination count",
            minimum=1,
        )
        color_entry_count = _integer(
            color_reference.get("entry_count"),
            "recurrence color-contraction entry count",
            minimum=1,
        )
        color_logical_entry_count = _integer(
            color_reference.get("logical_entry_count"),
            "recurrence color-contraction logical entry count",
            minimum=1,
        )
        if color_active_sector_count > color_sector_count:
            raise ArtifactError(
                "recurrence active color sectors exceed the physical sector count"
            )
        if color_destination_count < color_group_count:
            raise ArtifactError(
                "recurrence color destinations do not cover contraction groups"
            )
        if color_group_count > color_active_sector_count * color_component_count:
            raise ArtifactError(
                "recurrence color groups exceed the active sector-helicity domain"
            )
        factorization_raw = color_reference.get("factorization")
        if factorization_raw is not None:
            factorization = _mapping(
                factorization_raw,
                "recurrence color-contraction factorization",
            )
            color_factorization_kind = _string(
                factorization.get("kind"),
                "recurrence color-contraction factorization kind",
            )
            color_factorization_rank = _integer(
                factorization.get("rank"),
                "recurrence color-contraction factorization rank",
                minimum=2,
            )
            color_factorization_coset_count = _integer(
                factorization.get("coset_count"),
                "recurrence color-contraction factorization coset count",
                minimum=1,
            )
            valid_factorization = (
                (
                    color_storage == "repeated"
                    and color_factorization_kind == "klein-four-walsh"
                    and color_factorization_rank == 2
                )
                or (
                    color_storage == "repeated"
                    and color_factorization_kind == "elementary-abelian-walsh"
                    and 3 <= color_factorization_rank <= 31
                )
                or (
                    color_storage == "convolution-kernels"
                    and color_factorization_kind == "symmetric-group-fourier"
                    and 2 <= color_factorization_rank <= 10
                )
            )
            if not valid_factorization:
                raise ArtifactError(
                    "recurrence color factorization kind, rank, and storage disagree"
                )
        symmetric_group_fft = color_factorization_kind == "symmetric-group-fourier"
        if (color_storage == "convolution-kernels") != symmetric_group_fft:
            raise ArtifactError(
                "recurrence convolution-kernel storage requires symmetric-group "
                "Fourier factorization"
            )
        if (
            SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY in declared_capabilities
        ) != symmetric_group_fft:
            raise ArtifactError(
                "recurrence symmetric-group Fourier metadata and runtime capability "
                "disagree"
            )
        color_fft_provenance = _symmetric_group_fft_provenance(
            color_reference,
            factorization_kind=color_factorization_kind,
            degree=color_factorization_rank,
            channel_count=color_factorization_coset_count,
            group_count=color_group_count,
            component_count=color_component_count,
            entry_count=color_entry_count,
            logical_entry_count=color_logical_entry_count,
            context="recurrence color-contraction",
        )
    elif SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY in declared_capabilities:
        raise ArtifactError(
            "recurrence symmetric-group Fourier capability has no color payload"
        )
    return _ExecutionInspection(
        execution_mode="recurrence",
        prepared_backend=_string(
            pack.get("backend"),
            "prepared recurrence kernel pack.backend",
        ),
        prepared_kernel_count=prepared_kernel_count,
        invocation_count=contribution_rows,
        attachment_count=0,
        finalization_count=finalization_rows,
        closure_count=closure_rows,
        requested_point_tile_size=requested_tile,
        effective_point_tile_size=requested_tile,
        workspace_limit_bytes=workspace_mib * 1024 * 1024,
        arena_semantic_component_count=semantic_components,
        arena_component_count=arena_components,
        arena_component_reuse_count=arena_reuse,
        momentum_form_count=_optional_counter(
            direct_arena,
            "momentum_form_count",
            "recurrence inspection direct_arena.momentum_form_count",
        ),
        direct_source_row_count=source_rows,
        direct_contribution_row_count=contribution_rows,
        direct_finalization_row_count=finalization_rows,
        direct_closure_row_count=closure_rows,
        direct_row_group_count=_optional_counter(
            direct_arena,
            "row_group_count",
            "recurrence inspection direct_arena.row_group_count",
        ),
        recurrence_color_accuracy=color_accuracy,
        recurrence_color_storage=color_storage,
        recurrence_color_sector_count=color_sector_count,
        recurrence_color_active_sector_count=color_active_sector_count,
        recurrence_color_component_count=color_component_count,
        recurrence_color_group_count=color_group_count,
        recurrence_color_destination_count=color_destination_count,
        recurrence_color_entry_count=color_entry_count,
        recurrence_color_logical_entry_count=color_logical_entry_count,
        recurrence_color_factorization_kind=color_factorization_kind,
        recurrence_color_factorization_rank=color_factorization_rank,
        recurrence_color_factorization_coset_count=(color_factorization_coset_count),
        recurrence_color_contraction_method=(
            None
            if color_fft_provenance is None
            else cast(str, color_fft_provenance["method"])
        ),
        recurrence_color_fft_degree=(
            None
            if color_fft_provenance is None
            else cast(int, color_fft_provenance["degree"])
        ),
        recurrence_color_fft_channel_count=(
            None
            if color_fft_provenance is None
            else cast(int, color_fft_provenance["channel_count"])
        ),
        recurrence_color_fft_covered_local_group_count=(
            None
            if color_fft_provenance is None
            else cast(int, color_fft_provenance["covered_local_group_count"])
        ),
        recurrence_color_direct_residual_group_count=(
            None
            if color_fft_provenance is None
            else cast(int, color_fft_provenance["residual_group_count"])
        ),
        recurrence_color_direct_residual_entry_count=(
            None
            if color_fft_provenance is None
            else cast(int, color_fft_provenance["residual_entry_count"])
        ),
        recurrence_color_raw_kernel_bytes=(
            None
            if color_fft_provenance is None
            else cast(int, color_fft_provenance["raw_kernel_bytes"])
        ),
        recurrence_color_transformed_kernel_bytes=(
            None
            if color_fft_provenance is None
            else cast(int, color_fft_provenance["transformed_kernel_bytes"])
        ),
        recurrence_color_fft_capability=(
            None
            if color_fft_provenance is None
            else cast(str, color_fft_provenance["capability"])
        ),
        packed_input_bytes=0,
        packed_output_bytes=0,
        scatter_bytes=0,
        native_profile_phases=_RECURRENCE_PROFILE_PHASES,
    )


def _symmetric_group_fft_provenance(
    reference: Mapping[str, object],
    *,
    factorization_kind: str | None,
    degree: int | None,
    channel_count: int | None,
    group_count: int,
    component_count: int,
    entry_count: int,
    logical_entry_count: int,
    context: str,
) -> dict[str, int | str] | None:
    raw = reference.get("fft_provenance")
    if factorization_kind != "symmetric-group-fourier":
        if raw is not None or "fft_provenance" in reference:
            raise ArtifactError(f"{context} non-FFT summary carries FFT provenance")
        return None
    if degree is None or channel_count is None:
        raise ArtifactError(f"{context} symmetric-group factorization is incomplete")
    provenance = _mapping(raw, f"{context} FFT provenance")
    expected_fields = {
        "method",
        "degree",
        "channel_count",
        "covered_local_group_count",
        "residual_group_count",
        "residual_entry_count",
        "raw_kernel_bytes",
        "transformed_kernel_bytes",
        "capability",
    }
    if set(provenance) != expected_fields:
        raise ArtifactError(f"{context} FFT provenance has unsupported fields")
    if group_count % component_count:
        raise ArtifactError(f"{context} FFT component map is not rectangular")
    local_group_count = group_count // component_count
    group_order = math.factorial(degree)
    covered_local_group_count = channel_count * group_order
    if covered_local_group_count > local_group_count:
        raise ArtifactError(f"{context} FFT channels exceed the local group domain")
    residual_group_count = local_group_count - covered_local_group_count
    residual_entry_count = (
        local_group_count * (local_group_count + 1) // 2
        - covered_local_group_count * (covered_local_group_count + 1) // 2
    )
    kernel_entry_count = channel_count * (channel_count + 1) // 2 * group_order
    raw_kernel_bytes = kernel_entry_count * 16
    transformed_kernel_bytes = kernel_entry_count * 8
    values: dict[str, int | str] = {
        "method": _string(provenance.get("method"), f"{context} FFT method"),
        "degree": _integer(provenance.get("degree"), f"{context} FFT degree"),
        "channel_count": _integer(
            provenance.get("channel_count"), f"{context} FFT channel count"
        ),
        "covered_local_group_count": _integer(
            provenance.get("covered_local_group_count"),
            f"{context} FFT-covered local group count",
        ),
        "residual_group_count": _integer(
            provenance.get("residual_group_count"),
            f"{context} direct residual group count",
        ),
        "residual_entry_count": _integer(
            provenance.get("residual_entry_count"),
            f"{context} direct residual entry count",
        ),
        "raw_kernel_bytes": _integer(
            provenance.get("raw_kernel_bytes"), f"{context} raw kernel bytes"
        ),
        "transformed_kernel_bytes": _integer(
            provenance.get("transformed_kernel_bytes"),
            f"{context} transformed kernel bytes",
        ),
        "capability": _string(
            provenance.get("capability"), f"{context} FFT capability"
        ),
    }
    expected: dict[str, int | str] = {
        "method": "symmetric-group-fourier",
        "degree": degree,
        "channel_count": channel_count,
        "covered_local_group_count": covered_local_group_count,
        "residual_group_count": residual_group_count,
        "residual_entry_count": residual_entry_count,
        "raw_kernel_bytes": raw_kernel_bytes,
        "transformed_kernel_bytes": transformed_kernel_bytes,
        "capability": SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
    }
    if (
        values != expected
        or entry_count != kernel_entry_count + residual_entry_count
        or logical_entry_count != entry_count * component_count
    ):
        raise ArtifactError(f"{context} symmetric-group FFT provenance is inconsistent")
    return values


def _on_the_fly_color_reference(
    runtime_metadata: Mapping[str, object],
    *,
    color_accuracy: str,
    contracted: bool,
) -> dict[str, Any] | None:
    raw = runtime_metadata.get("color_contraction")
    if not contracted:
        if raw is not None or "color_contraction" in runtime_metadata:
            raise ArtifactError(
                "LC on-the-fly execution unexpectedly carries color contraction"
            )
        return None
    reference = _mapping(
        raw,
        "on-the-fly execution.runtime_metadata.color_contraction",
    )
    expected_fields = {
        "abi",
        "color_accuracy",
        "storage",
        "includes_color_factor",
        "group_count",
        "sector_count",
        "active_sector_count",
        "component_count",
        "destination_count",
        "entry_count",
        "logical_entry_count",
        "semantic_digest",
        "factorization",
        "path",
        "size_bytes",
        "sha256",
    }
    if reference.get("factorization") is not None:
        expected_fields.add("fft_provenance")
    if set(reference) != expected_fields:
        raise ArtifactError(
            "on-the-fly color-contraction reference has unsupported fields"
        )
    _require_contract(
        reference,
        "abi",
        _RECURRENCE_COLOR_CONTRACTION_CODEC_ABI,
        "on-the-fly execution.runtime_metadata.color_contraction",
    )
    accuracy = _string(
        reference.get("color_accuracy"),
        "on-the-fly color-contraction accuracy",
    )
    storage = _string(
        reference.get("storage"),
        "on-the-fly color-contraction storage",
    )
    sector_count = _integer(
        reference.get("sector_count"),
        "on-the-fly color-contraction sector count",
        minimum=1,
    )
    active_sector_count = _integer(
        reference.get("active_sector_count"),
        "on-the-fly color-contraction active sector count",
        minimum=1,
    )
    component_count = _integer(
        reference.get("component_count"),
        "on-the-fly color-contraction component count",
        minimum=1,
    )
    group_count = _integer(
        reference.get("group_count"),
        "on-the-fly color-contraction group count",
        minimum=1,
    )
    destination_count = _integer(
        reference.get("destination_count"),
        "on-the-fly color-contraction destination count",
        minimum=1,
    )
    entry_count = _integer(
        reference.get("entry_count"),
        "on-the-fly color-contraction entry count",
        minimum=1,
    )
    logical_entry_count = _integer(
        reference.get("logical_entry_count"),
        "on-the-fly color-contraction logical entry count",
        minimum=1,
    )
    semantic_digest = _string(
        reference.get("semantic_digest"),
        "on-the-fly color-contraction semantic digest",
    )
    payload_sha256 = _string(
        reference.get("sha256"),
        "on-the-fly color-contraction SHA-256",
    )
    factorization_raw = reference.get("factorization")
    symmetric_group_fft = False
    factorization_rank = None
    factorization_coset_count = None
    if factorization_raw is not None:
        factorization = _mapping(
            factorization_raw,
            "on-the-fly color-contraction factorization",
        )
        if set(factorization) != {"kind", "rank", "coset_count"}:
            raise ArtifactError(
                "on-the-fly color-contraction factorization has unsupported fields"
            )
        factorization_kind = _string(
            factorization.get("kind"),
            "on-the-fly color-contraction factorization kind",
        )
        factorization_rank = _integer(
            factorization.get("rank"),
            "on-the-fly color-contraction factorization degree",
            minimum=2,
        )
        factorization_coset_count = _integer(
            factorization.get("coset_count"),
            "on-the-fly color-contraction channel count",
            minimum=1,
        )
        symmetric_group_fft = (
            factorization_kind == "symmetric-group-fourier" and factorization_rank <= 10
        )
        if not symmetric_group_fft:
            raise ArtifactError(
                "on-the-fly color factorization must be symmetric-group Fourier"
            )
    if (
        accuracy != color_accuracy
        or accuracy not in {"nlc", "full"}
        or storage != ("convolution-kernels" if symmetric_group_fft else "expanded")
        or reference.get("includes_color_factor") is not True
        or component_count != 1
        or active_sector_count != group_count
        or destination_count != group_count
        or group_count > sector_count
        or logical_entry_count != entry_count
        or reference.get("path") != "on-the-fly-color.bin"
        or semantic_digest != payload_sha256
        or len(payload_sha256) != 64
        or any(character not in "0123456789abcdef" for character in payload_sha256)
    ):
        raise ArtifactError("on-the-fly color-contraction reference is noncanonical")
    fft_provenance = _symmetric_group_fft_provenance(
        reference,
        factorization_kind=("symmetric-group-fourier" if symmetric_group_fft else None),
        degree=factorization_rank,
        channel_count=factorization_coset_count,
        group_count=group_count,
        component_count=component_count,
        entry_count=entry_count,
        logical_entry_count=logical_entry_count,
        context="on-the-fly color-contraction",
    )
    _integer(
        reference.get("size_bytes"),
        "on-the-fly color-contraction payload size",
        minimum=1,
    )
    return {
        "color_accuracy": accuracy,
        "storage": storage,
        "sector_count": sector_count,
        "active_sector_count": active_sector_count,
        "component_count": component_count,
        "group_count": group_count,
        "destination_count": destination_count,
        "entry_count": entry_count,
        "logical_entry_count": logical_entry_count,
        "factorization_kind": (
            "symmetric-group-fourier" if symmetric_group_fft else None
        ),
        "factorization_rank": factorization_rank,
        "factorization_coset_count": factorization_coset_count,
        "contraction_method": (
            None if fft_provenance is None else fft_provenance["method"]
        ),
        "fft_degree": (None if fft_provenance is None else fft_provenance["degree"]),
        "fft_channel_count": (
            None if fft_provenance is None else fft_provenance["channel_count"]
        ),
        "fft_covered_local_group_count": (
            None
            if fft_provenance is None
            else fft_provenance["covered_local_group_count"]
        ),
        "direct_residual_group_count": (
            None if fft_provenance is None else fft_provenance["residual_group_count"]
        ),
        "direct_residual_entry_count": (
            None if fft_provenance is None else fft_provenance["residual_entry_count"]
        ),
        "raw_kernel_bytes": (
            None if fft_provenance is None else fft_provenance["raw_kernel_bytes"]
        ),
        "transformed_kernel_bytes": (
            None
            if fft_provenance is None
            else fft_provenance["transformed_kernel_bytes"]
        ),
        "fft_capability": (
            None if fft_provenance is None else fft_provenance["capability"]
        ),
    }


def _execution_inspection(
    manifest: ArtifactManifest,
    execution_path: Path,
    required_runtime_capabilities: Sequence[object] | None = None,
    *,
    process: Mapping[str, object] | None = None,
) -> _ExecutionInspection:
    capabilities = (
        required_runtime_capabilities
        if required_runtime_capabilities is not None
        else _sequence(
            manifest.runtime.get("required_runtime_capabilities"),
            "runtime.required_runtime_capabilities",
        )
    )
    capability_values = frozenset(str(value) for value in capabilities)
    on_the_fly_color_capabilities = {
        ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
        ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
    }
    if (
        ON_THE_FLY_RUNTIME_CAPABILITY in capability_values
        or not on_the_fly_color_capabilities.isdisjoint(capability_values)
    ):
        selected_color_capabilities = capability_values.intersection(
            on_the_fly_color_capabilities
        )
        if len(selected_color_capabilities) != 1:
            raise ArtifactError(
                "on-the-fly execution must declare exactly one color capability"
            )
        selected_color_capability = next(iter(selected_color_capabilities))
        on_the_fly_capabilities = {
            ON_THE_FLY_RUNTIME_CAPABILITY,
            selected_color_capability,
        }
        symmetric_group_fft_capability = (
            SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY in capability_values
        )
        if symmetric_group_fft_capability:
            on_the_fly_capabilities.add(SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY)
        if capability_values != on_the_fly_capabilities or len(capabilities) != len(
            on_the_fly_capabilities
        ):
            raise ArtifactError(
                "on-the-fly execution must declare exactly its required runtime "
                "capabilities"
            )
        contracted_color = (
            selected_color_capability == ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY
        )
        expected_color_accuracies = {"nlc", "full"} if contracted_color else {"lc"}
        execution = _json_mapping(
            manifest, execution_path, "process execution manifest"
        )
        expected_root_fields = {
            "schema_version",
            "kind",
            "required_runtime_capabilities",
            "process",
            "key",
            "color_accuracy",
            "external_pdg_order",
            "kernel_pack",
            "runtime_options",
            "selector_policy",
            "runtime_metadata",
            "runtime_container",
        }
        if process is None or set(execution) != expected_root_fields:
            raise ArtifactError(
                "on-the-fly process execution manifest has unsupported root fields"
            )
        kind = _string(execution.get("kind"), "process execution manifest.kind")
        declared_capabilities = _sequence(
            execution.get("required_runtime_capabilities"),
            "on-the-fly process execution manifest.required_runtime_capabilities",
        )
        declared_capability_values = frozenset(
            str(value) for value in declared_capabilities
        )
        external_pdgs = tuple(
            _sequence(
                execution.get("external_pdg_order"),
                "on-the-fly process execution manifest.external_pdg_order",
            )
        )
        expected_pdgs = tuple(
            _sequence(
                process.get("external_pdgs"),
                f"process {process.get('id')} external_pdgs",
            )
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (*external_pdgs, *expected_pdgs)
            )
            or execution.get("schema_version") != 3
            or kind != _ON_THE_FLY_RUNTIME_KIND
            or len(declared_capabilities) != len(on_the_fly_capabilities)
            or declared_capability_values != on_the_fly_capabilities
            or execution.get("process") != process.get("expression")
            or execution.get("key") != process.get("id")
            or execution.get("color_accuracy") not in expected_color_accuracies
            or process.get("color_accuracy") != execution.get("color_accuracy")
            or external_pdgs != expected_pdgs
        ):
            raise ArtifactError(
                "on-the-fly process execution manifest disagrees with its outer process"
            )
        kernel_pack = _mapping(
            execution.get("kernel_pack"),
            "on-the-fly execution.kernel_pack",
        )
        if kernel_pack != {
            "manifest_path": "model/eager-kernel-pack.json",
            "payload_root": "model/eager-kernels",
        }:
            raise ArtifactError("on-the-fly prepared-kernel paths are not canonical")
        runtime_container = _mapping(
            execution.get("runtime_container"),
            "on-the-fly execution.runtime_container",
        )
        if runtime_container != {
            "kind": "pyamplicol-on-the-fly-runtime-container",
            "schema_version": 1,
            "storage_abi": "pacbin-v1",
            "path": "on-the-fly-runtime.pacbin",
            "seed_member_path": "on-the-fly/process-seed-v1.bin",
        }:
            raise ArtifactError("on-the-fly runtime-container contract is noncanonical")
        runtime_metadata = _mapping(
            execution.get("runtime_metadata"),
            "on-the-fly execution.runtime_metadata",
        )
        color_reference = _on_the_fly_color_reference(
            runtime_metadata,
            color_accuracy=str(execution.get("color_accuracy")),
            contracted=contracted_color,
        )
        reference_uses_symmetric_group_fft = (
            color_reference is not None
            and color_reference["factorization_kind"] == "symmetric-group-fourier"
        )
        if reference_uses_symmetric_group_fft != symmetric_group_fft_capability:
            raise ArtifactError(
                "on-the-fly symmetric-group Fourier metadata and runtime "
                "capability disagree"
            )
        selector_policy = _mapping(
            execution.get("selector_policy"),
            "on-the-fly execution.selector_policy",
        )
        if set(selector_policy) - {"color_basis"} != {
            "color_coverage",
            "reference_color_word",
            "trace_reflections_folded",
            "selector_census",
        }:
            raise ArtifactError(
                "on-the-fly execution.selector_policy has unsupported fields"
            )
        color_basis = selector_policy.get("color_basis", "trace")
        if color_basis not in ("trace", "adjoint"):
            raise ArtifactError(
                "on-the-fly execution.selector_policy.color_basis "
                "must be trace or adjoint"
            )
        if color_basis == "adjoint" and not contracted_color:
            raise ArtifactError("adjoint on-the-fly color requires contracted NLC/full")
        expected_color_coverage = "contracted" if contracted_color else "complete"
        if selector_policy.get("color_coverage") != expected_color_coverage:
            raise ArtifactError(
                "on-the-fly execution.selector_policy.color_coverage disagrees "
                "with its color capability"
            )
        trace_reflections_folded = selector_policy.get("trace_reflections_folded")
        if not isinstance(trace_reflections_folded, bool):
            raise ArtifactError(
                "on-the-fly execution.selector_policy.trace_reflections_folded "
                "must be a boolean"
            )
        if contracted_color and trace_reflections_folded:
            raise ArtifactError(
                "contracted on-the-fly color cannot fold trace reflections"
            )
        reference = selector_policy.get("reference_color_word")
        if reference is not None:
            word = tuple(
                _integer(
                    value,
                    "on-the-fly execution.selector_policy.reference_color_word"
                    f"[{index}]",
                    minimum=1,
                )
                for index, value in enumerate(
                    _sequence(
                        reference,
                        "on-the-fly execution.selector_policy.reference_color_word",
                    )
                )
            )
            if not word or len(set(word)) != len(word):
                raise ArtifactError(
                    "on-the-fly execution selector reference must contain unique "
                    "positive labels"
                )
        census = _mapping(
            selector_policy.get("selector_census"),
            "on-the-fly execution.selector_policy.selector_census",
        )
        if set(census) != {
            "physical_helicity_count",
            "physical_color_flow_count",
        }:
            raise ArtifactError(
                "on-the-fly execution selector census has unsupported fields"
            )
        helicity_count = _integer(
            census.get("physical_helicity_count"),
            "on-the-fly execution selector census physical_helicity_count",
            minimum=1,
        )
        color_flow_count = _integer(
            census.get("physical_color_flow_count"),
            "on-the-fly execution selector census physical_color_flow_count",
            minimum=1,
        )
        if helicity_count > (1 << 64) - 1 or color_flow_count > (1 << 64) - 1:
            raise ArtifactError(
                "on-the-fly execution selector census exceeds the u64 domain"
            )
        if contracted_color and color_flow_count != 1:
            raise ArtifactError(
                "contracted on-the-fly execution must expose one physical color result"
            )
        runtime_options = _mapping(
            execution.get("runtime_options"),
            "on-the-fly execution.runtime_options",
        )
        if set(runtime_options) != {
            "point_tile_size",
            "query_construction_threads",
        }:
            raise ArtifactError("on-the-fly execution.runtime_options is invalid")
        point_tile_size = _integer(
            runtime_options.get("point_tile_size"),
            "on-the-fly execution.runtime_options.point_tile_size",
            minimum=1,
        )
        query_construction_threads = _integer(
            runtime_options.get("query_construction_threads"),
            "on-the-fly execution.runtime_options.query_construction_threads",
            minimum=1,
        )
        return _ExecutionInspection(
            execution_mode="on-the-fly",
            physical_helicity_count=helicity_count,
            physical_color_flow_count=color_flow_count,
            requested_point_tile_size=point_tile_size,
            requested_query_construction_threads=query_construction_threads,
            recurrence_color_accuracy=(
                None if color_reference is None else color_reference["color_accuracy"]
            ),
            recurrence_color_storage=(
                None if color_reference is None else color_reference["storage"]
            ),
            recurrence_color_sector_count=(
                None if color_reference is None else color_reference["sector_count"]
            ),
            recurrence_color_active_sector_count=(
                None
                if color_reference is None
                else color_reference["active_sector_count"]
            ),
            recurrence_color_component_count=(
                None if color_reference is None else color_reference["component_count"]
            ),
            recurrence_color_group_count=(
                None if color_reference is None else color_reference["group_count"]
            ),
            recurrence_color_destination_count=(
                None
                if color_reference is None
                else color_reference["destination_count"]
            ),
            recurrence_color_entry_count=(
                None if color_reference is None else color_reference["entry_count"]
            ),
            recurrence_color_logical_entry_count=(
                None
                if color_reference is None
                else color_reference["logical_entry_count"]
            ),
            recurrence_color_factorization_kind=(
                None
                if color_reference is None
                else color_reference["factorization_kind"]
            ),
            recurrence_color_factorization_rank=(
                None
                if color_reference is None
                else color_reference["factorization_rank"]
            ),
            recurrence_color_factorization_coset_count=(
                None
                if color_reference is None
                else color_reference["factorization_coset_count"]
            ),
            recurrence_color_contraction_method=(
                None
                if color_reference is None
                else color_reference["contraction_method"]
            ),
            recurrence_color_fft_degree=(
                None if color_reference is None else color_reference["fft_degree"]
            ),
            recurrence_color_fft_channel_count=(
                None
                if color_reference is None
                else color_reference["fft_channel_count"]
            ),
            recurrence_color_fft_covered_local_group_count=(
                None
                if color_reference is None
                else color_reference["fft_covered_local_group_count"]
            ),
            recurrence_color_direct_residual_group_count=(
                None
                if color_reference is None
                else color_reference["direct_residual_group_count"]
            ),
            recurrence_color_direct_residual_entry_count=(
                None
                if color_reference is None
                else color_reference["direct_residual_entry_count"]
            ),
            recurrence_color_raw_kernel_bytes=(
                None if color_reference is None else color_reference["raw_kernel_bytes"]
            ),
            recurrence_color_transformed_kernel_bytes=(
                None
                if color_reference is None
                else color_reference["transformed_kernel_bytes"]
            ),
            recurrence_color_fft_capability=(
                None if color_reference is None else color_reference["fft_capability"]
            ),
            native_profile_phases=_ON_THE_FLY_PROFILE_PHASES,
        )
    if RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY in capability_values:
        color_capabilities = capability_values.intersection(
            {
                RECURRENCE_COLOR_RUNTIME_CAPABILITY,
                RECURRENCE_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
            }
        )
        if len(color_capabilities) != 1:
            raise ArtifactError(
                "recurrence direct-arena execution must declare exactly one "
                "recurrence color capability"
            )
        required = {
            *color_capabilities,
            RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
        }
        if not required.issubset(capability_values):
            missing = ", ".join(sorted(required - capability_values))
            raise ArtifactError(
                "recurrence direct-arena execution is missing required "
                f"capabilities: {missing}"
            )
        execution = _json_mapping(
            manifest, execution_path, "process execution manifest"
        )
        kind = _string(execution.get("kind"), "process execution manifest.kind")
        if kind != _RECURRENCE_RUNTIME_KIND:
            raise ArtifactError(
                "recurrence capability is paired with an unsupported process "
                f"execution kind {kind!r}"
            )
        return _recurrence_execution_inspection(manifest, execution)
    if _EAGER_RUNTIME_CAPABILITIES.isdisjoint(capability_values):
        # Compiled execution manifests can be several GiB at high multiplicity.
        # Their capability contract already identifies the execution lane, and
        # the manifest hash validation above has verified the payload bytes.
        return _ExecutionInspection(
            execution_mode="compiled",
            native_profile_phases=_COMPILED_PROFILE_PHASES,
        )
    execution = _json_mapping(manifest, execution_path, "process execution manifest")
    kind = _string(execution.get("kind"), "process execution manifest.kind")
    if kind == _EAGER_RUNTIME_KIND:
        return _eager_execution_inspection(manifest, execution, execution_path)
    raise ArtifactError(f"unsupported process execution kind {kind!r}")


def _optional_nonnegative_integer(
    value: object,
    context: str,
) -> int | None:
    if value is None:
        return None
    return _integer(value, context, minimum=0)


def _optional_counter(
    record: Mapping[str, object],
    name: str,
    context: str,
) -> int | None:
    if name not in record:
        return None
    return _integer(record[name], context, minimum=0)


def _require_contract(
    record: Mapping[str, object],
    name: str,
    expected: str,
    context: str,
) -> None:
    actual = record.get(name)
    if actual != expected:
        raise ArtifactError(
            f"{context}.{name} must be {expected!r}; regenerate the recurrence "
            "artifact with the current pyAmpliCol"
        )


def _source_helicity_selection(
    value: object,
    context: str,
) -> tuple[tuple[int, int], ...]:
    selection = _mapping(value, context)
    result: list[tuple[int, int]] = []
    for raw_label, raw_helicity in selection.items():
        try:
            label = int(raw_label)
        except (TypeError, ValueError) as exc:
            raise ArtifactError(f"{context} has a non-integer source label") from exc
        if str(label) != str(raw_label) or label < 1:
            raise ArtifactError(f"{context} has an invalid source label {raw_label!r}")
        if isinstance(raw_helicity, bool) or not isinstance(raw_helicity, int):
            raise ArtifactError(f"{context}[{raw_label!r}] must be an integer")
        result.append((label, raw_helicity))
    result.sort()
    if len({label for label, _helicity in result}) != len(result):
        raise ArtifactError(f"{context} repeats a source label")
    return tuple(result)


def _color_sector_selection(
    value: object,
    context: str,
) -> tuple[int, ...]:
    result = tuple(
        _integer(item, f"{context}[{index}]")
        for index, item in enumerate(_sequence(value, context))
    )
    if result != tuple(sorted(set(result))):
        raise ArtifactError(f"{context} must contain unique sorted sector IDs")
    return result


def _physics_inspection(
    manifest: ArtifactManifest,
    process: Mapping[str, object],
    execution: _ExecutionInspection,
) -> _PhysicsInspection:
    relative = str(process["physics_path"])
    try:
        payload = json.loads(_payloads(manifest).read_bytes(relative))
    except OSError as exc:
        raise ArtifactError(
            f"cannot read runtime physics metadata {relative}: {exc}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"invalid runtime physics metadata {relative}: {exc}"
        ) from exc

    physics = _mapping(payload, f"runtime physics metadata {relative}")
    if execution.execution_mode == "on-the-fly":
        if set(physics) != {
            "schema_version",
            "kind",
            "process_id",
            "process",
            "color_accuracy",
            "external_particles",
            "model_parameters",
        }:
            raise ArtifactError(
                f"{relative} compact on-the-fly metadata has unsupported fields"
            )
        if (
            physics.get("schema_version") != 1
            or physics.get("kind") != _ON_THE_FLY_PUBLIC_METADATA_KIND
            or physics.get("process_id") != process.get("id")
            or physics.get("process") != process.get("expression")
            or physics.get("color_accuracy") not in {"lc", "nlc", "full"}
            or process.get("color_accuracy") != physics.get("color_accuracy")
        ):
            raise ArtifactError(
                f"{relative} compact on-the-fly metadata disagrees with its process"
            )
        external_pdgs = _sequence(
            process.get("external_pdgs"),
            f"process {process.get('id')} external_pdgs",
        )
        externals = _sequence(
            physics.get("external_particles"),
            f"{relative}.external_particles",
        )
        if len(externals) != len(external_pdgs):
            raise ArtifactError(
                f"{relative}.external_particles disagrees with the process arity"
            )
        for index, (raw_external, pdg) in enumerate(
            zip(externals, external_pdgs, strict=True)
        ):
            external = _mapping(
                raw_external,
                f"{relative}.external_particles[{index}]",
            )
            if (
                set(external)
                != {
                    "index",
                    "label",
                    "particle",
                    "pdg",
                    "role",
                    "momentum_slot",
                    "momentum_components",
                }
                or external.get("index") != index
                or external.get("label") != index + 1
                or isinstance(external.get("pdg"), bool)
                or not isinstance(external.get("pdg"), int)
                or external.get("pdg") != pdg
                or external.get("momentum_slot") != index
                or external.get("role") != ("initial" if index < 2 else "final")
                or external.get("momentum_components") != ["E", "px", "py", "pz"]
            ):
                raise ArtifactError(
                    f"{relative}.external_particles[{index}] is noncanonical"
                )
            _string(
                external.get("particle"),
                f"{relative}.external_particles[{index}].particle",
            )
        parameter_names: set[str] = set()
        for index, raw_parameter in enumerate(
            _sequence(physics.get("model_parameters"), f"{relative}.model_parameters")
        ):
            parameter = _mapping(
                raw_parameter,
                f"{relative}.model_parameters[{index}]",
            )
            if set(parameter) != {
                "name",
                "kind",
                "default_real",
                "default_imaginary",
                "mutable",
            }:
                raise ArtifactError(
                    f"{relative}.model_parameters[{index}] has unsupported fields"
                )
            name = _string(
                parameter.get("name"),
                f"{relative}.model_parameters[{index}].name",
            )
            kind = parameter.get("kind")
            mutable = parameter.get("mutable")
            defaults = (
                parameter.get("default_real"),
                parameter.get("default_imaginary"),
            )
            if (
                name in parameter_names
                or kind
                not in {
                    "normalization",
                    "mass",
                    "width",
                    "coupling",
                    "external",
                    "derived",
                }
                or not isinstance(mutable, bool)
                or (kind == "derived" and mutable)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(value)
                    for value in defaults
                )
            ):
                raise ArtifactError(
                    f"{relative}.model_parameters[{index}] is noncanonical"
                )
            parameter_names.add(name)
        helicity_count = execution.physical_helicity_count
        color_flow_count = execution.physical_color_flow_count
        if helicity_count is None or color_flow_count is None:  # pragma: no cover
            raise ArtifactError("on-the-fly execution has no selector census")
        contracted_color = process.get("color_accuracy") in {"nlc", "full"}
        return _PhysicsInspection(
            physical_helicities=helicity_count,
            computed_helicities=helicity_count,
            physical_color_components=color_flow_count,
            computed_color_components=color_flow_count,
            helicity_coverage="complete",
            color_coverage="contracted" if contracted_color else "complete",
            selector_provenance="on-the-fly-compact-seed",
            helicity_runtime_contract="complete-reusable",
            color_flow_runtime_contract=(
                None if contracted_color else "complete-reusable"
            ),
            lc_physical_sector_count=(None if contracted_color else color_flow_count),
        )
    helicities = _sequence(physics.get("helicities"), f"{relative}.helicities")
    colors = _sequence(physics.get("color_components"), f"{relative}.color_components")
    coverage = _mapping(physics.get("coverage"), f"{relative}.coverage")

    computed_helicities = sum(
        _computed(item, f"{relative}.helicities[{index}]")
        for index, item in enumerate(helicities)
    )
    computed_colors = sum(
        _computed(item, f"{relative}.color_components[{index}]")
        for index, item in enumerate(colors)
    )
    selector_fields: dict[str, object] = {}
    extensions_raw = physics.get("extensions")
    extensions = (
        {}
        if extensions_raw is None
        else _mapping(extensions_raw, f"{relative}.extensions")
    )
    selectors_raw = extensions.get("runtime_selectors")
    if selectors_raw is not None:
        selectors = _mapping(selectors_raw, f"{relative}.extensions.runtime_selectors")
        if selectors.get("kind") != "pyamplicol-runtime-selectors":
            raise ArtifactError(
                f"{relative}.extensions.runtime_selectors.kind is unsupported"
            )
        if selectors.get("contract_version") != 1:
            raise ArtifactError(
                f"{relative}.extensions.runtime_selectors.contract_version must be 1"
            )
        selector_fields["selector_provenance"] = _string(
            selectors.get("provenance"),
            f"{relative}.extensions.runtime_selectors.provenance",
        )
        axes = _mapping(
            selectors.get("axes"),
            f"{relative}.extensions.runtime_selectors.axes",
        )
        helicity_axis = _mapping(
            axes.get("helicity"),
            f"{relative}.extensions.runtime_selectors.axes.helicity",
        )
        color_axis = _mapping(
            axes.get("color_flow"),
            f"{relative}.extensions.runtime_selectors.axes.color_flow",
        )
        helicity_generation_coverage = _string(
            helicity_axis.get("generation_coverage"),
            f"{relative}.extensions.runtime_selectors.axes.helicity."
            "generation_coverage",
        )
        color_generation_coverage = _string(
            color_axis.get("generation_coverage"),
            f"{relative}.extensions.runtime_selectors.axes.color_flow."
            "generation_coverage",
        )
        helicity_selection = _source_helicity_selection(
            helicity_axis.get("generation_selection"),
            f"{relative}.extensions.runtime_selectors.axes.helicity."
            "generation_selection",
        )
        color_selection = _color_sector_selection(
            color_axis.get("generation_selection"),
            f"{relative}.extensions.runtime_selectors.axes.color_flow."
            "generation_selection",
        )
        selector_fields["selected_source_helicities"] = helicity_selection
        selector_fields["selected_color_sector_ids"] = color_selection
        selector_fields["helicity_runtime_contract"] = _string(
            helicity_axis.get("runtime_contract"),
            f"{relative}.extensions.runtime_selectors.axes.helicity.runtime_contract",
        )
        selector_fields["color_flow_runtime_contract"] = _string(
            color_axis.get("runtime_contract"),
            f"{relative}.extensions.runtime_selectors.axes.color_flow.runtime_contract",
        )
        selector_fields["generation_specialized_axes"] = tuple(
            _string(
                value,
                f"{relative}.extensions.runtime_selectors."
                f"generation_specialized_axes[{index}]",
            )
            for index, value in enumerate(
                _sequence(
                    selectors.get("generation_specialized_axes"),
                    f"{relative}.extensions.runtime_selectors."
                    "generation_specialized_axes",
                )
            )
        )
        helicity_contract = cast(str, selector_fields["helicity_runtime_contract"])
        color_contract = cast(str, selector_fields["color_flow_runtime_contract"])
        specialized_axes = cast(
            tuple[str, ...], selector_fields["generation_specialized_axes"]
        )
        expected_specialized_axes = (
            *(("helicity",) if helicity_contract == "generation-specialized" else ()),
            *(("color_flow",) if color_contract == "generation-specialized" else ()),
        )
        if helicity_contract not in {
            "complete-reusable",
            "generation-specialized",
        }:
            raise ArtifactError("unsupported helicity runtime-selector contract")
        if color_contract not in {
            "complete-reusable",
            "generation-specialized",
            "contracted-color",
        }:
            raise ArtifactError("unsupported color-flow runtime-selector contract")
        if helicity_generation_coverage != str(coverage.get("helicities")):
            raise ArtifactError(
                "helicity selector generation coverage disagrees with physics coverage"
            )
        if color_generation_coverage != str(coverage.get("color")):
            raise ArtifactError(
                "color-flow selector generation coverage disagrees with "
                "physics coverage"
            )
        if specialized_axes != expected_specialized_axes:
            raise ArtifactError(
                "runtime-selector specialized axes disagree with axis contracts"
            )
        if (helicity_contract == "complete-reusable") != (
            helicity_generation_coverage == "complete" and not helicity_selection
        ):
            raise ArtifactError(
                "helicity runtime contract disagrees with generation selection"
            )
        if color_contract == "complete-reusable" and (
            color_generation_coverage != "complete" or color_selection
        ):
            raise ArtifactError(
                "color-flow reusable contract carries a generation specialization"
            )
        if color_contract == "generation-specialized" and (
            color_generation_coverage != "selected" or not color_selection
        ):
            raise ArtifactError(
                "color-flow specialized contract lacks an exact sector selection"
            )
        if color_contract == "contracted-color" and (
            color_generation_coverage != "contracted" or color_selection
        ):
            raise ArtifactError(
                "contracted-color runtime contract carries LC selection metadata"
            )
        legacy_helicity_selection = _source_helicity_selection(
            extensions.get("selected_source_helicities", {}),
            f"{relative}.extensions.selected_source_helicities",
        )
        if legacy_helicity_selection != helicity_selection:
            raise ArtifactError(
                "runtime-selector helicity selection disagrees with physics provenance"
            )
        recurrence_raw = selectors.get("helicity_recurrence")
        if recurrence_raw is not None:
            recurrence = _mapping(
                recurrence_raw,
                f"{relative}.extensions.runtime_selectors.helicity_recurrence",
            )
            selector_fields["helicity_recurrence_status"] = _string(
                recurrence.get("status"),
                f"{relative}.extensions.runtime_selectors.helicity_recurrence.status",
            )
            proof_counts = _mapping(
                recurrence.get("proof_counts"),
                f"{relative}.extensions.runtime_selectors."
                "helicity_recurrence.proof_counts",
            )
            count_fields = {
                "helicity_optimized_class_count": ("optimized_recurrence_class_count"),
                "helicity_optimized_current_count": "optimized_current_count",
                "helicity_residual_current_count": "residual_current_count",
                "helicity_optimized_amplitude_class_count": (
                    "optimized_amplitude_class_count"
                ),
                "helicity_residual_amplitude_count": "residual_amplitude_count",
            }
            for field, key in count_fields.items():
                selector_fields[field] = _optional_nonnegative_integer(
                    proof_counts.get(key),
                    f"{relative}.extensions.runtime_selectors."
                    f"helicity_recurrence.proof_counts.{key}",
                )
            selector_fields["helicity_materialized_current_count"] = (
                _optional_nonnegative_integer(
                    recurrence.get("materialized_current_count"),
                    f"{relative}.extensions.runtime_selectors."
                    "helicity_recurrence.materialized_current_count",
                )
            )
            selector_fields["helicity_materialized_amplitude_count"] = (
                _optional_nonnegative_integer(
                    recurrence.get("materialized_amplitude_count"),
                    f"{relative}.extensions.runtime_selectors."
                    "helicity_recurrence.materialized_amplitude_count",
                )
            )

    replay_raw = extensions.get("lc_topology_replay")
    if replay_raw is not None:
        replay = _mapping(replay_raw, f"{relative}.extensions.lc_topology_replay")
        selector_fields.update(
            {
                "lc_physical_sector_count": len(
                    _sequence(
                        replay.get("physical_sector_ids"),
                        f"{relative}.extensions.lc_topology_replay.physical_sector_ids",
                    )
                ),
                "lc_materialized_sector_count": len(
                    _sequence(
                        replay.get("materialized_sector_ids"),
                        f"{relative}.extensions.lc_topology_replay."
                        "materialized_sector_ids",
                    )
                ),
                "lc_replayed_sector_count": _integer(
                    replay.get("replayed_sector_count"),
                    f"{relative}.extensions.lc_topology_replay.replayed_sector_count",
                    minimum=0,
                ),
                "lc_residual_sector_count": len(
                    _sequence(
                        replay.get("residual_sector_ids"),
                        f"{relative}.extensions.lc_topology_replay.residual_sector_ids",
                    )
                ),
            }
        )

    return _PhysicsInspection(
        physical_helicities=len(helicities),
        computed_helicities=computed_helicities,
        physical_color_components=len(colors),
        computed_color_components=computed_colors,
        helicity_coverage=str(coverage.get("helicities", "unknown")),
        color_coverage=str(coverage.get("color", "unknown")),
        **selector_fields,
    )


def _process_inspection(
    manifest: ArtifactManifest,
    process: Mapping[str, object],
    execution: _ExecutionInspection,
) -> ArtifactProcessInspection:
    process_id = str(process["id"])
    lc_flow_layout: str | None = None
    if str(process["color_accuracy"]) == "lc":
        if execution.execution_mode == "on-the-fly":
            lc_flow_layout = "compact/query-local"
        else:
            lc_flow_layout = "topology-replay"
            generation = manifest.extensions.get("generation")
            if isinstance(generation, Mapping):
                concrete = generation.get("concrete_processes")
                if isinstance(concrete, Sequence) and not isinstance(
                    concrete, (str, bytes)
                ):
                    for record in concrete:
                        if (
                            not isinstance(record, Mapping)
                            or record.get("id") != process_id
                        ):
                            continue
                        filters = record.get("filters")
                        if not isinstance(filters, Mapping):
                            continue
                        candidate = filters.get("lc_flow_layout")
                        if candidate is not None:
                            lc_flow_layout = str(candidate)
                    if lc_flow_layout not in {"topology-replay", "all-flow-union"}:
                        raise ArtifactError(
                            f"process {process_id}.filters.lc_flow_layout is "
                            "unsupported"
                        )
    aliases: list[ArtifactAliasInspection] = []
    for index, raw_alias in enumerate(
        _sequence(process.get("aliases"), f"process {process_id}.aliases")
    ):
        alias = _mapping(raw_alias, f"process {process_id}.aliases[{index}]")
        aliases.append(
            ArtifactAliasInspection(
                id=str(alias["id"]),
                expression=str(alias["expression"]),
                representative_id=process_id,
                external_pdgs=tuple(
                    cast(int, value)
                    for value in _sequence(
                        alias.get("external_pdgs"),
                        f"process {process_id}.aliases[{index}].external_pdgs",
                    )
                ),
            )
        )
    physics = _physics_inspection(manifest, process, execution)
    union_sector_count = (
        physics.physical_color_components
        if lc_flow_layout == "all-flow-union"
        else None
    )
    return ArtifactProcessInspection(
        id=process_id,
        expression=str(process["expression"]),
        color_accuracy=str(process["color_accuracy"]),
        external_pdgs=tuple(
            cast(int, value)
            for value in _sequence(
                process.get("external_pdgs"), f"process {process_id}.external_pdgs"
            )
        ),
        default=process_id == manifest.default_process_id,
        physical_helicities=physics.physical_helicities,
        computed_helicities=physics.computed_helicities,
        physical_color_components=physics.physical_color_components,
        computed_color_components=physics.computed_color_components,
        helicity_coverage=physics.helicity_coverage,
        color_coverage=physics.color_coverage,
        aliases=tuple(aliases),
        selector_provenance=physics.selector_provenance,
        helicity_runtime_contract=physics.helicity_runtime_contract,
        color_flow_runtime_contract=physics.color_flow_runtime_contract,
        generation_specialized_axes=physics.generation_specialized_axes,
        selected_source_helicities=physics.selected_source_helicities,
        selected_color_sector_ids=physics.selected_color_sector_ids,
        helicity_recurrence_status=physics.helicity_recurrence_status,
        helicity_optimized_class_count=physics.helicity_optimized_class_count,
        helicity_optimized_current_count=physics.helicity_optimized_current_count,
        helicity_residual_current_count=physics.helicity_residual_current_count,
        helicity_optimized_amplitude_class_count=(
            physics.helicity_optimized_amplitude_class_count
        ),
        helicity_residual_amplitude_count=(physics.helicity_residual_amplitude_count),
        helicity_materialized_current_count=(
            physics.helicity_materialized_current_count
        ),
        helicity_materialized_amplitude_count=(
            physics.helicity_materialized_amplitude_count
        ),
        lc_physical_sector_count=physics.lc_physical_sector_count,
        lc_materialized_sector_count=physics.lc_materialized_sector_count,
        lc_replayed_sector_count=physics.lc_replayed_sector_count,
        lc_residual_sector_count=physics.lc_residual_sector_count,
        lc_flow_layout=lc_flow_layout,
        lc_union_sector_count=union_sector_count,
        execution_mode=execution.execution_mode,
        prepared_backend=execution.prepared_backend,
        prepared_kernel_count=execution.prepared_kernel_count,
        referenced_kernel_count=execution.referenced_kernel_count,
        invocation_count=execution.invocation_count,
        attachment_count=execution.attachment_count,
        evaluation_alias_count=execution.evaluation_alias_count,
        maximum_fanout=execution.maximum_fanout,
        finalization_count=execution.finalization_count,
        closure_count=execution.closure_count,
        selector_closure_available=execution.selector_closure_available,
        selector_domain_count=execution.selector_domain_count,
        selector_domain_membership_count=execution.selector_domain_membership_count,
        requested_point_tile_size=execution.requested_point_tile_size,
        effective_point_tile_size=execution.effective_point_tile_size,
        requested_query_construction_threads=(
            execution.requested_query_construction_threads
        ),
        workspace_limit_bytes=execution.workspace_limit_bytes,
        workspace_bytes=execution.workspace_bytes,
        arena_semantic_component_count=execution.arena_semantic_component_count,
        arena_component_count=execution.arena_component_count,
        arena_component_reuse_count=execution.arena_component_reuse_count,
        momentum_form_count=execution.momentum_form_count,
        direct_source_row_count=execution.direct_source_row_count,
        direct_contribution_row_count=execution.direct_contribution_row_count,
        direct_finalization_row_count=execution.direct_finalization_row_count,
        direct_closure_row_count=execution.direct_closure_row_count,
        direct_row_group_count=execution.direct_row_group_count,
        recurrence_color_accuracy=execution.recurrence_color_accuracy,
        recurrence_color_storage=execution.recurrence_color_storage,
        recurrence_color_sector_count=execution.recurrence_color_sector_count,
        recurrence_color_active_sector_count=(
            execution.recurrence_color_active_sector_count
        ),
        recurrence_color_component_count=execution.recurrence_color_component_count,
        recurrence_color_group_count=execution.recurrence_color_group_count,
        recurrence_color_destination_count=(
            execution.recurrence_color_destination_count
        ),
        recurrence_color_entry_count=execution.recurrence_color_entry_count,
        recurrence_color_logical_entry_count=(
            execution.recurrence_color_logical_entry_count
        ),
        recurrence_color_factorization_kind=(
            execution.recurrence_color_factorization_kind
        ),
        recurrence_color_factorization_rank=(
            execution.recurrence_color_factorization_rank
        ),
        recurrence_color_factorization_coset_count=(
            execution.recurrence_color_factorization_coset_count
        ),
        recurrence_color_contraction_method=(
            execution.recurrence_color_contraction_method
        ),
        recurrence_color_fft_degree=execution.recurrence_color_fft_degree,
        recurrence_color_fft_channel_count=(
            execution.recurrence_color_fft_channel_count
        ),
        recurrence_color_fft_covered_local_group_count=(
            execution.recurrence_color_fft_covered_local_group_count
        ),
        recurrence_color_direct_residual_group_count=(
            execution.recurrence_color_direct_residual_group_count
        ),
        recurrence_color_direct_residual_entry_count=(
            execution.recurrence_color_direct_residual_entry_count
        ),
        recurrence_color_raw_kernel_bytes=(execution.recurrence_color_raw_kernel_bytes),
        recurrence_color_transformed_kernel_bytes=(
            execution.recurrence_color_transformed_kernel_bytes
        ),
        recurrence_color_fft_capability=(execution.recurrence_color_fft_capability),
        packed_input_bytes=execution.packed_input_bytes,
        packed_output_bytes=execution.packed_output_bytes,
        scatter_bytes=execution.scatter_bytes,
        native_profile_phases=execution.native_profile_phases,
    )


def _group_member_inspection(
    member: GroupMember,
) -> ArtifactProcessGroupMemberInspection:
    return ArtifactProcessGroupMemberInspection(
        id=member.process_id,
        process=member.expression,
        pdgs=member.pdgs,
        initial_count=member.initial_count,
        factor=str(member.factor),
        permutation=member.permutation,
        conjugated=member.conjugated,
        basis=str(member.basis),
        generated=member.generated,
        warning=member.warning,
    )


def _groups_inspection(
    group_map: ProcessGroupMap,
    *,
    stored: bool,
    process_ids: Sequence[str] = (),
) -> ArtifactProcessGroupsInspection:
    """Return ``group_map`` with its groups in ``process_ids`` order."""

    order = {process_id: index for index, process_id in enumerate(process_ids)}
    groups = sorted(
        group_map.groups.values(),
        key=lambda group: (
            order.get(group.representative.process_id, len(order)),
            group.representative.process_id,
        ),
    )
    return ArtifactProcessGroupsInspection(
        stored=stored,
        groups=tuple(
            ArtifactProcessGroupInspection(
                representative_id=group.representative.process_id,
                representative=group.representative.expression,
                members=tuple(
                    _group_member_inspection(member) for member in group.members
                ),
            )
            for group in groups
        ),
        analyses=tuple(dict(analysis) for analysis in group_map.analyses),
    )


def _process_groups_inspection(
    manifest: ArtifactManifest,
) -> ArtifactProcessGroupsInspection:
    """Read the stored process-group map; without one, each process is a group."""

    group_map = read_process_groups(manifest.root)
    stored = group_map is not None
    if group_map is None:
        group_map = initial_process_groups(
            (
                str(process["id"]),
                str(process["expression"]),
                cast(Sequence[int], process["external_pdgs"]),
            )
            for process in manifest.processes
        )
    return _groups_inspection(
        group_map,
        stored=stored,
        process_ids=[str(process["id"]) for process in manifest.processes],
    )


def stored_process_groups(
    artifact: str | Path,
) -> ArtifactProcessGroupsInspection | None:
    """Return the artifact's stored process-group map, or ``None`` without one.

    Unlike :func:`inspect_artifact`, this reads only ``artifact.json`` and
    ``process-groups.json`` and verifies no payload.
    """

    group_map = read_process_groups(artifact)
    return None if group_map is None else _groups_inspection(group_map, stored=True)


def process_group_selection(
    artifact: str | Path,
    process_id: str,
    *,
    permutation: Sequence[int] | None = None,
) -> ArtifactProcessGroupSelection | None:
    """Return how ``process_id`` relates to its stored process group.

    Returns ``None`` when the artifact stores no map or the process is alone in
    its group. ``permutation`` is the runtime's external permutation of the
    requested leg order: from the representative's legs for a routed member,
    whose runtime already composes the stored relation, and from the
    process's own legs otherwise. The returned permutation relates the
    representative to the requested leg order.
    """

    group_map = read_process_groups(artifact)
    group = None if group_map is None else group_map.group_of(process_id)
    if group is None or len(group.members) == 1:
        return None
    member = next(item for item in group.members if item.process_id == process_id)
    representative = group.representative
    relation_permutation = member.permutation
    if permutation is not None:
        requested = tuple(int(index) for index in permutation)
        relation_permutation = (
            tuple(requested[index] for index in member.permutation)
            if member.generated
            else requested
        )
    if member.process_id == representative.process_id:
        role = "representative"
    elif member.generated:
        role = "generated member"
    else:
        role = "routed member"
    return ArtifactProcessGroupSelection(
        process_id=member.process_id,
        role=role,
        representative_id=representative.process_id,
        representative=representative.expression,
        factor=str(member.factor),
        permutation=relation_permutation,
        conjugated=member.conjugated,
        basis=str(member.basis),
        warning=member.warning,
        group_size=len(group.members),
        routed_count=sum(not item.generated for item in group.members),
        conjugated_count=sum(item.conjugated for item in group.members),
    )


def _reject_retired_recurrence_metadata(artifact: str | Path) -> None:
    """Reject pre-release recurrence artifacts before payload verification."""

    root = Path(artifact).expanduser().resolve(strict=True)
    manifest_path = root / MANIFEST_NAME
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # The strict manifest loader below owns ordinary format diagnostics.
        return
    if not isinstance(raw, Mapping):
        return
    capabilities: set[str] = set()
    runtime = raw.get("runtime")
    if isinstance(runtime, Mapping):
        declared = runtime.get("required_runtime_capabilities")
        if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
            capabilities.update(str(value) for value in declared)
    processes = raw.get("processes")
    if isinstance(processes, Sequence) and not isinstance(processes, (str, bytes)):
        for process in processes:
            if not isinstance(process, Mapping):
                continue
            declared = process.get("required_runtime_capabilities")
            if isinstance(declared, Sequence) and not isinstance(
                declared, (str, bytes)
            ):
                capabilities.update(str(value) for value in declared)
    if _RETIRED_RECURRENCE_RUNTIME_CAPABILITY in capabilities:
        raise ArtifactError(
            "this pre-release recurrence artifact uses the retired packet ABI; "
            "regenerate it with the current pyAmpliCol direct-arena recurrence mode"
        )


def inspect_artifact(artifact: str | Path) -> ArtifactInspection:
    """Validate and summarize one generated artifact without loading evaluators."""

    _reject_retired_recurrence_metadata(artifact)
    manifest = load_manifest(artifact)
    target = _mapping(manifest.producer["target"], "producer.target")
    model = manifest.model
    runtime = manifest.runtime
    execution_paths = _execution_paths(manifest)
    processes = tuple(
        _process_inspection(
            manifest,
            process,
            _execution_inspection(
                manifest,
                execution_paths[str(process["id"])],
                _sequence(
                    process.get("required_runtime_capabilities"),
                    f"process {process['id']} required runtime capabilities",
                ),
                process=process,
            ),
        )
        for process in manifest.processes
    )
    dependencies = tuple(
        ArtifactDependencyInspection(
            name=str(dependency["name"]),
            version=str(dependency["version"]),
            license=str(dependency["license"]),
            source=str(dependency["source"]),
        )
        for dependency in manifest.dependencies
    )
    restriction = model.get("restriction")
    container_path: str | None = None
    container_abi: str | None = None
    container_member_count: int | None = None
    container_unpacked_size: int | None = None
    container_raw = manifest.extensions.get("evaluator_payload_container")
    if container_raw is not None:
        container = _mapping(
            container_raw,
            "extensions.evaluator_payload_container",
        )
        if (
            container.get("kind") != "pyamplicol-evaluator-payload-container"
            or container.get("schema_version") != 1
            or container.get("storage_abi") != "pacbin-v1"
        ):
            raise ArtifactError("unsupported evaluator payload container contract")
        container_path = _string(
            container.get("path"),
            "extensions.evaluator_payload_container.path",
        )
        container_abi = _string(
            container.get("storage_abi"),
            "extensions.evaluator_payload_container.storage_abi",
        )
        container_member_count = _integer(
            container.get("member_count"),
            "extensions.evaluator_payload_container.member_count",
            minimum=0,
        )
        container_unpacked_size = _integer(
            container.get("unpacked_size_bytes"),
            "extensions.evaluator_payload_container.unpacked_size_bytes",
            minimum=0,
        )
        matching = tuple(
            payload for payload in manifest.payloads if payload.path == container_path
        )
        if len(matching) != 1 or matching[0].role != "evaluator-state":
            raise ArtifactError(
                "evaluator payload container is not one declared "
                "evaluator-state payload"
            )
    return ArtifactInspection(
        kind="pyamplicol-artifact-inspection",
        path=manifest.root,
        artifact_kind=manifest.kind,
        artifact_id=manifest.artifact_id,
        created_utc=manifest.created_utc,
        producer_version=str(manifest.producer["version"]),
        target=str(target["triple"]),
        cpu_features=tuple(
            str(value)
            for value in _sequence(
                target.get("cpu_features"), "producer.target.cpu_features"
            )
        ),
        model_name=str(model["name"]),
        model_source=str(model["source_kind"]),
        model_restriction=None if restriction is None else str(restriction),
        default_process_id=manifest.default_process_id,
        runtime_engine=str(runtime["engine"]),
        runtime_version=str(runtime["engine_version"]),
        runtime_capabilities=tuple(
            str(value)
            for value in _sequence(
                runtime.get("required_runtime_capabilities"),
                "runtime.required_runtime_capabilities",
            )
        ),
        payload_count=len(manifest.payloads),
        payload_size_bytes=sum(payload.size_bytes for payload in manifest.payloads),
        integrity="verified",
        processes=processes,
        dependencies=dependencies,
        physical_file_count=len(manifest.payloads) + 1,
        evaluator_container_path=container_path,
        evaluator_container_abi=container_abi,
        evaluator_container_member_count=container_member_count,
        evaluator_container_unpacked_size_bytes=container_unpacked_size,
        process_groups=_process_groups_inspection(manifest),
    )


__all__ = [
    "ArtifactAliasInspection",
    "ArtifactDependencyInspection",
    "ArtifactInspection",
    "ArtifactProcessGroupInspection",
    "ArtifactProcessGroupMemberInspection",
    "ArtifactProcessGroupSelection",
    "ArtifactProcessGroupsInspection",
    "ArtifactProcessInspection",
    "inspect_artifact",
    "process_group_selection",
    "stored_process_groups",
]
