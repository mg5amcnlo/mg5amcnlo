# SPDX-License-Identifier: 0BSD
"""Transactional schema-v3 output for compiled concrete processes."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, cast

from pyamplicol.api.requests import ModelSource
from pyamplicol.artifacts import (
    ArtifactBuilder,
    ArtifactManifest,
    PayloadRecord,
    load_manifest,
)
from pyamplicol.artifacts.manifest import PORTABLE_64LE_TARGET, compute_artifact_id
from pyamplicol.artifacts.process_groups import (
    PROCESS_GROUPS_MEDIA_TYPE,
    initial_process_groups,
    process_groups_payload,
    validate_process_groups,
)
from pyamplicol.artifacts.security import sha256_file
from pyamplicol.config import (
    ConfigClamp,
    ConfigResolution,
    GenerationConfig,
    RunConfig,
    config_to_dict,
)

from .._internal.versions import (
    COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY,
    COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY,
    COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
    COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
    COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY,
    COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
    COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
    COMPILED_PLANE_DIRECT_APPLICATION_ABI,
    COMPILED_RUNTIME_SELECTORS_CAPABILITY,
    EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
    EAGER_DIRECT_TABLE_BINDING_ABI,
    EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
    EVALUATOR_RUNTIME_CAPABILITIES,
    NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
    ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
    ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
    ON_THE_FLY_RUNTIME_CAPABILITY,
    PROCESS_ARTIFACT_SCHEMA_VERSION,
    PYTHON_API_VERSION,
    RECURRENCE_BUILDER_INPUT_ABI,
    RECURRENCE_COLOR_RUNTIME_CAPABILITY,
    RECURRENCE_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
    RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
    RECURRENCE_DIRECT_BACKEND_ABI,
    RECURRENCE_DIRECT_TEMPLATE_ABI,
    RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY,
    RECURRENCE_PLAN_ABI,
    RECURRENCE_RUNTIME_LAYOUT_ABI,
    RUNTIME_PHYSICS_SCHEMA_VERSION,
    SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    SYMBOLICA_CPP_RUNTIME_CAPABILITY,
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    SYMBOLICA_SERIALIZATION_ABI,
    SYMJIT_APPLICATION_ABI,
    SYMJIT_F64_RUNTIME_CAPABILITY,
    SYMJIT_PLANE_APPLICATION_ABI,
    SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
    TOML_SCHEMA_VERSION,
    active_native_build_inputs_sha256,
    active_source_revision,
    package_version,
    verify_native_module,
)
from ..evaluators.execution_schema import evaluator_runtime_capabilities
from ..models.loading import COMPILED_MODEL_SCHEMA_VERSION, CompiledModel
from ..models.prepared import (
    PREPARED_KERNEL_PACK_IDENTITY_ABI,
    prepared_kernel_pack_manifest_identity_sha256,
)
from .contracts import RuntimeExpressionSchema
from .eager_columnar import EAGER_LOWERING_INPUT_ABI
from .eager_lowering import EAGER_RUNTIME_KIND
from .eager_tables import EAGER_KERNEL_ABI, EAGER_RUNTIME_CAPABILITY
from .evaluator_container import (
    PacbinIndex,
    PacbinMemberKind,
    PacbinMemberSource,
    PacbinReader,
    write_pacbin_atomic,
)
from .process_grouping import PROCESS_GROUPS_PATH, PROCESS_GROUPS_ROLE, ProcessGroupMap
from .recurrence_schedule_sharing import (
    RECURRENCE_PROCESS_BINDING_ABI,
    RECURRENCE_SCHEDULE_INDEX_PATH,
    RECURRENCE_SCHEDULE_SHARING_SCHEMA_VERSION,
    RecurrenceProcessBinding,
    RecurrenceProcessExecutorPack,
    RecurrenceProcessRemap,
    RecurrenceScheduleSharingPlan,
    intern_recurrence_schedules,
    recurrence_helicity_selector_schedule_digest,
)
from .structural_source_proof import (
    ROLE as STRUCTURAL_SOURCE_PROOF_ROLE,
)
from .structural_source_proof import build_generation_structural_proof
from .validation import ValidationPointRecord, validation_point_map

if TYPE_CHECKING:
    from pyamplicol.artifacts.transaction import ArtifactWriteMode

ApiBundleHook = Callable[
    [ArtifactBuilder, Mapping[str, Sequence[Sequence[float]]]],
    Sequence[object],
]
ArtifactPayloadHook = Callable[[ArtifactBuilder], Mapping[str, object]]

_CONFIG_REQUESTED_PATH = "config/requested.toml"
_CONFIG_EFFECTIVE_PATH = "config/effective.toml"
_COMPILED_MODEL_PATH = "model/compiled-model.json"
_MODEL_PARAMETERS_PATH = "model/parameters.json"
_EVALUATOR_SET_PATH = "processes/evaluators.json"
_EAGER_KERNEL_PACK_PATH = "model/eager-kernel-pack.json"
_EAGER_KERNEL_PAYLOAD_ROOT = "model/eager-kernels"
_HELICITY_SUM_PAYLOAD_ROOT = "helicity-sum"
_HELICITY_SELECTOR_UNION_PAYLOAD_ROOT = "helicity-selector-union"
_COLOR_SELECTOR_PAYLOAD_ROOT = "color-selector"
_EAGER_PACK_IDENTITY_EXTENSION = "eager_prepared_pack"
_EAGER_PACK_IDENTITY_KIND = "pyamplicol-prepared-kernel-pack-identity"
_EAGER_PACK_IDENTITY_SCHEMA_VERSION = 1
_EVALUATOR_PAYLOAD_CONTAINER_EXTENSION = "evaluator_payload_container"
_RECURRENCE_BOOTSTRAP_FILENAME = "recurrence-bootstrap.bin"
_EVALUATOR_PAYLOAD_CONTAINER_PATH = "evaluators.pacbin"
_COMPILED_COLOR_CONTRACTION_MEMBER_PATH = "compiled-color.pacrclr3"
_EVALUATOR_PAYLOAD_CONTAINER_KIND = "pyamplicol-evaluator-payload-container"
_EVALUATOR_PAYLOAD_CONTAINER_SCHEMA_VERSION = 1
_EVALUATOR_PAYLOAD_CONTAINER_STORAGE_ABI = "pacbin-v1"
EAGER_PLAN_V3_ABI = "pyamplicol-eager-plan-v3"
EAGER_RUNTIME_LAYOUT_ABI = "pyamplicol-eager-runtime-layout-v1"
EAGER_PLAN_V3_RUNTIME_CAPABILITY = "rusticol.eager-runtime-layout.complex-f64.v1"
_EAGER_PLAN_V3_RUNTIME_CAPABILITIES = tuple(
    sorted(
        (
            EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
            EAGER_PLAN_V3_RUNTIME_CAPABILITY,
        )
    )
)
EAGER_RUNTIME_CONTAINER_KIND = "pyamplicol-eager-runtime-container"
EAGER_RUNTIME_CONTAINER_SCHEMA_VERSION = 1
EAGER_RUNTIME_STORAGE_ABI = "pacbin-v1"
_EAGER_RUNTIME_CONTAINER_PATH = "eager-runtime.pacbin"
RECURRENCE_RUNTIME_KIND = "pyamplicol-runtime-recurrence-execution"
RECURRENCE_RUNTIME_CONTAINER_KIND = "pyamplicol-recurrence-runtime-container"
RECURRENCE_RUNTIME_CONTAINER_SCHEMA_VERSION = 1
RECURRENCE_RUNTIME_STORAGE_ABI = "pacbin-v1"
_RECURRENCE_RUNTIME_CONTAINER_PATH = "recurrence-runtime.pacbin"
_RECURRENCE_DIRECT_SCHEDULE_MEMBER_PATH = "schedule/recurrence-direct-schedule-v2.bin"
_RECURRENCE_COLOR_CONTRACTION_PATH = "recurrence-color.bin"
_RECURRENCE_SCHEDULE_SHARING_EXTENSION = "recurrence_schedule_sharing"
_RECURRENCE_HELICITY_SELECTOR_SCHEDULE_ROOT = "recurrence/schedules"
_RECURRENCE_HELICITY_SELECTOR_BINDING_PATH = "helicity-selector-binding.bin"
_RECURRENCE_HELICITY_DISPATCH_PATH = "recurrence-helicity-dispatch-v1.bin"
_RECURRENCE_HELICITY_DISPATCH_ABI = "pyamplicol-recurrence-helicity-dispatch-v1"
ON_THE_FLY_RUNTIME_KIND = "pyamplicol-runtime-on-the-fly-execution"
ON_THE_FLY_RUNTIME_CONTAINER_KIND = "pyamplicol-on-the-fly-runtime-container"
ON_THE_FLY_RUNTIME_CONTAINER_SCHEMA_VERSION = 1
ON_THE_FLY_RUNTIME_STORAGE_ABI = "pacbin-v1"
ON_THE_FLY_PUBLIC_METADATA_KIND = "pyamplicol-on-the-fly-public-metadata"
_RECURRENCE_HELICITY_SELECTOR_COMPANION_V2_KIND = (
    "pyamplicol-recurrence-helicity-selector-companion-v2"
)
_ON_THE_FLY_RUNTIME_CONTAINER_PATH = "on-the-fly-runtime.pacbin"
_ON_THE_FLY_PROCESS_SEED_MEMBER_PATH = "on-the-fly/process-seed-v1.bin"
_ON_THE_FLY_COLOR_CONTRACTION_PATH = "on-the-fly-color.bin"
_SAFE_TOML_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_SUPPORTED_ARTIFACT_TARGETS = frozenset(
    {
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu",
    }
)


@dataclass(frozen=True, slots=True)
class CompiledExecutionArtifact:
    runtime_schema: RuntimeExpressionSchema
    stage_manifest: Mapping[str, object]
    model_parameter_evaluator: Mapping[str, object] | None
    dag_summary: Mapping[str, object]
    evaluator_root: Path
    color_selector_executions: tuple[CompiledColorSelectorExecutionArtifact, ...] = ()
    helicity_selector_executions: tuple[
        CompiledHelicitySelectorExecutionArtifact, ...
    ] = ()


@dataclass(frozen=True, slots=True)
class CompiledColorSelectorExecutionArtifact:
    materialized_sector_id: int
    execution: CompiledExecutionArtifact


@dataclass(frozen=True, slots=True)
class CompiledHelicitySelectorExecutionArtifact:
    selector_domain_ids: tuple[int, ...]
    execution: CompiledExecutionArtifact
    schedule_mode: str = "parent-closure"


@dataclass(frozen=True, slots=True)
class CompiledProcessArtifact:
    process_id: str
    expression: str
    color_accuracy: str
    external_pdgs: tuple[int, ...]
    aliases: tuple[Mapping[str, object], ...]
    runtime_schema: RuntimeExpressionSchema
    stage_manifest: Mapping[str, object]
    model_parameter_evaluator: Mapping[str, object] | None
    dag_summary: Mapping[str, object]
    evaluator_root: Path
    validation_point: ValidationPointRecord
    generation_filters: Mapping[str, object]
    color_contraction_payload: bytes | None = None
    helicity_sum_execution: CompiledExecutionArtifact | None = None
    helicity_selector_executions: tuple[
        CompiledHelicitySelectorExecutionArtifact, ...
    ] = ()
    color_selector_executions: tuple[CompiledColorSelectorExecutionArtifact, ...] = ()
    umami_physics: Mapping[str, object] | None = None
    umami_color_sectors: tuple[Mapping[str, object], ...] = ()
    umami_structural_keys: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class EagerPlanV3ProcessArtifact:
    """One Rust-lowered eager runtime plus its bounded publication metadata."""

    process_id: str
    expression: str
    color_accuracy: str
    external_pdgs: tuple[int, ...]
    aliases: tuple[Mapping[str, object], ...]
    physics: Mapping[str, object]
    eager_runtime_path: Path
    eager_runtime_size_bytes: int
    eager_runtime_sha256: str
    eager_runtime_member_count: int
    eager_runtime_unpacked_size_bytes: int
    eager_runtime_index_sha256: str
    lowering_input_sha256: str
    referenced_kernel_ids: frozenset[int]
    inspection_summary: Mapping[str, object]
    point_tile_size: int
    workspace_mib: int
    dag_summary: Mapping[str, object]
    validation_point: ValidationPointRecord
    generation_filters: Mapping[str, object]
    umami_physics: Mapping[str, object] | None = None
    umami_color_sectors: tuple[Mapping[str, object], ...] = ()
    umami_structural_keys: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class RecurrenceProcessArtifact:
    """One compact recurrence runtime plus bounded public metadata."""

    process_id: str
    expression: str
    color_accuracy: str
    external_pdgs: tuple[int, ...]
    aliases: tuple[Mapping[str, object], ...]
    physics: Mapping[str, object]
    recurrence_schedule_path: Path
    recurrence_schedule_digest: str
    recurrence_native_schedule_semantic_digest: str
    recurrence_schedule_size_bytes: int
    recurrence_schedule_sha256: str
    recurrence_schedule_member_count: int
    recurrence_schedule_unpacked_size_bytes: int
    recurrence_schedule_index_sha256: str
    builder_input_sha256: str
    prepared_kernel_pack_digest: str
    direct_template_catalog_digest: str
    referenced_kernel_ids: frozenset[int]
    inspection_summary: Mapping[str, object]
    runtime_metadata: Mapping[str, object]
    color_contraction_payload: bytes | None
    color_contraction_summary: Mapping[str, object] | None
    point_tile_size: int
    workspace_mib: int
    recurrence_summary: Mapping[str, object]
    validation_point: ValidationPointRecord
    generation_filters: Mapping[str, object]
    generation_profile: Mapping[str, object]
    recurrence_process_remap: RecurrenceProcessRemap
    recurrence_process_executor_pack: RecurrenceProcessExecutorPack
    process_digest: str
    process_support_mask: int = 1
    helicity_selector_companion: RecurrenceHelicitySelectorPlanArtifact | None = None
    umami_physics: Mapping[str, object] | None = None
    umami_color_sectors: tuple[Mapping[str, object], ...] = ()
    umami_structural_keys: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class RecurrenceHelicitySelectorPlanArtifact:
    """Persisted physical-colour schedule for runtime singleton helicities.

    The schedule uses the ordinary recurrence Direct-plan ABI with runtime
    source-state dispatch.  It deliberately owns neither physics metadata nor
    prepared-model state: those identities remain authenticated by the
    enclosing recurrence process.
    """

    recurrence_schedule_path: Path
    recurrence_base_schedule_digest: str
    recurrence_schedule_digest: str
    recurrence_native_schedule_semantic_digest: str
    recurrence_schedule_size_bytes: int
    recurrence_schedule_sha256: str
    recurrence_schedule_member_count: int
    recurrence_schedule_unpacked_size_bytes: int
    recurrence_schedule_index_sha256: str
    helicity_dispatch_path: Path
    helicity_dispatch_size_bytes: int
    helicity_dispatch_sha256: str
    helicity_dispatch_base_runtime_layout_digest: str
    helicity_dispatch_resolved_helicity_count: int
    physical_destination_count: int
    builder_input_sha256: str
    inspection_summary: Mapping[str, object]
    referenced_kernel_ids: frozenset[int]
    recurrence_process_remap: RecurrenceProcessRemap
    recurrence_process_executor_pack: RecurrenceProcessExecutorPack


@dataclass(frozen=True, slots=True)
class OnTheFlyProcessArtifact:
    """One source-only runtime seed plus compact public metadata."""

    process_id: str
    expression: str
    color_accuracy: str
    external_pdgs: tuple[int, ...]
    aliases: tuple[Mapping[str, object], ...]
    physics: Mapping[str, object]
    runtime_path: Path
    runtime_size_bytes: int
    runtime_sha256: str
    runtime_member_count: int
    runtime_unpacked_size_bytes: int
    runtime_index_sha256: str
    referenced_kernel_ids: frozenset[int]
    runtime_metadata: Mapping[str, object]
    selector_policy: Mapping[str, object]
    color_contraction_payload: bytes | None
    color_contraction_summary: Mapping[str, object] | None
    point_tile_size: int
    query_construction_threads: int
    validation_point: ValidationPointRecord
    generation_filters: Mapping[str, object]
    umami_physics: Mapping[str, object] | None = None
    umami_color_sectors: tuple[Mapping[str, object], ...] = ()
    umami_structural_keys: Mapping[str, str] | None = None


ProcessArtifact = (
    CompiledProcessArtifact
    | EagerPlanV3ProcessArtifact
    | RecurrenceProcessArtifact
    | OnTheFlyProcessArtifact
)


@dataclass(frozen=True, slots=True)
class ArtifactWriteResult:
    output: Path
    files: tuple[Path, ...]
    validation_points: Mapping[str, Mapping[str, object]]
    api_bundle_path: str | None


@dataclass(frozen=True, slots=True)
class _GenerationConfigProvenance:
    requested: GenerationConfig | RunConfig
    effective: GenerationConfig | RunConfig
    adjustments: tuple[ConfigClamp, ...] = ()

    @classmethod
    def from_config(
        cls,
        config: GenerationConfig | RunConfig | ConfigResolution | None,
    ) -> _GenerationConfigProvenance:
        if isinstance(config, ConfigResolution):
            effective = replace(config.effective, color=config.effective.resolved_color)
            return cls(config.requested, effective, config.clamps)
        effective = GenerationConfig() if config is None else config
        if isinstance(effective, RunConfig):
            return cls(effective, replace(effective, color=effective.resolved_color))
        return cls(effective, effective)


class _EvaluatorPayloadCollector:
    """Collect evaluator payloads and publish one root pacbin container."""

    def __init__(
        self,
        builder: ArtifactBuilder,
        *,
        existing: ArtifactManifest | None,
        target: Mapping[str, object],
    ) -> None:
        self._builder = builder
        self._existing = existing
        self._target = dict(target)
        self._new_sources: dict[str, PacbinMemberSource] = {}
        self._staged_loose_paths: set[str] = set()
        self._discarded_prefixes: set[str] = set()

    def discard_prefix(self, prefix: str) -> None:
        normalized = prefix.strip("/")
        if not normalized:
            raise ValueError("packed evaluator discard prefix must not be empty")
        self._discarded_prefixes.add(normalized)
        owned_prefix = normalized + "/"
        self._new_sources = {
            path: source
            for path, source in self._new_sources.items()
            if path != normalized and not path.startswith(owned_prefix)
        }

    def add_file(
        self,
        relative: str,
        source: Path,
        *,
        process_id: str | None,
    ) -> PayloadRecord:
        kind = _packed_evaluator_member_kind(relative)
        if kind is None:
            return self._builder.add_file(
                relative,
                source,
                role="evaluator-state",
                media_type=_media_type(source),
                target=self._target,
                process_id=process_id,
            )
            return
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"evaluator payload must be a regular file: {source}")
        # Evaluator builders may reuse their temporary output paths while
        # materializing nested selector lanes.  Snapshot each payload when it
        # is registered so a later write cannot silently change an earlier
        # logical container member before publication.
        record = self._builder.add_file(
            relative,
            source,
            role="evaluator-state",
            media_type=_media_type(source),
            target=self._target,
            process_id=process_id,
        )
        self._register_staged_source(relative, kind)
        return record

    def add_bytes(
        self,
        relative: str,
        content: bytes,
        *,
        process_id: str | None,
        media_type: str | None = None,
    ) -> PayloadRecord:
        kind = _packed_evaluator_member_kind(relative)
        if kind is None:
            return self._builder.add_bytes(
                relative,
                content,
                role="evaluator-state",
                media_type=media_type or _media_type(Path(relative)),
                target=self._target,
                process_id=process_id,
            )
        record = self._builder.add_bytes(
            relative,
            content,
            role="evaluator-state",
            media_type=media_type or _media_type(Path(relative)),
            target=self._target,
            process_id=process_id,
        )
        self._register_staged_source(relative, kind)
        return record

    def add_stream(
        self,
        relative: str,
        source: BinaryIO,
        *,
        process_id: str | None,
        media_type: str | None = None,
    ) -> None:
        kind = _packed_evaluator_member_kind(relative)
        self._builder.add_stream(
            relative,
            source,
            role="evaluator-state",
            media_type=media_type or _media_type(Path(relative)),
            target=self._target,
            process_id=process_id,
        )
        if kind is not None:
            self._register_staged_source(relative, kind)

    def publish(self) -> dict[str, object] | None:
        existing_container = _existing_evaluator_container_path(self._existing)
        if existing_container is None:
            return self._publish_with_old_sources(())

        container_path = self._builder.staged_path(existing_container)
        with PacbinReader.open(container_path, verify_payloads=True) as reader:
            old_sources = tuple(
                PacbinMemberSource(
                    member.logical_path,
                    member.kind,
                    cast("BinaryIO", reader.open_member_stream(member.logical_path)),
                )
                for member in reader.members
                if not self._is_discarded(member.logical_path)
                and member.logical_path not in self._new_sources
            )
            return self._publish_with_old_sources(old_sources)

    def _publish_with_old_sources(
        self,
        old_container_sources: Sequence[PacbinMemberSource],
    ) -> dict[str, object] | None:
        combined = {source.logical_path: source for source in old_container_sources}
        loose_paths: set[str] = set(self._staged_loose_paths)
        if self._existing is not None:
            for record in self._existing.payloads:
                kind = _packed_evaluator_member_kind(record.path)
                if kind is None or self._is_discarded(record.path):
                    continue
                path = self._builder.staged_path(record.path)
                if not path.is_file() or path.is_symlink():
                    continue
                loose_paths.add(record.path)
                combined[record.path] = PacbinMemberSource(
                    record.path,
                    kind,
                    path,
                )
        combined.update(self._new_sources)
        if not combined:
            self._builder.discard_payloads(_EVALUATOR_PAYLOAD_CONTAINER_PATH)
            return None

        destination = self._builder.staged_path(
            _EVALUATOR_PAYLOAD_CONTAINER_PATH,
            create_parent=True,
        )
        written = write_pacbin_atomic(destination, combined.values())
        with PacbinReader.open(destination, verify_payloads=True) as verified:
            if verified.index != written:
                raise ValueError("published evaluator container failed verification")
            index = verified.index

        for relative in sorted(loose_paths):
            self._builder.discard_payloads(relative)
        self._builder.register_staged_file(
            _EVALUATOR_PAYLOAD_CONTAINER_PATH,
            role="evaluator-state",
            media_type="application/octet-stream",
            target=self._target,
            process_id=None,
        )
        return _evaluator_payload_container_extension(index)

    def _register_staged_source(
        self,
        relative: str,
        kind: PacbinMemberKind,
    ) -> None:
        path = self._builder.staged_path(relative)
        self._new_sources[relative] = PacbinMemberSource(relative, kind, path)
        self._staged_loose_paths.add(relative)

    def _is_discarded(self, relative: str) -> bool:
        return any(
            relative == prefix or relative.startswith(prefix + "/")
            for prefix in self._discarded_prefixes
        )


def _packed_evaluator_member_kind(relative: str) -> PacbinMemberKind | None:
    if relative.endswith(".symjit"):
        return PacbinMemberKind.SYMJIT_APPLICATION
    if relative.endswith(".evaluator.bin"):
        return PacbinMemberKind.SYMBOLICA_EXACT_STATE
    if relative.endswith(".color.pacrclr3") or relative.endswith(
        "/compiled-color.pacrclr3"
    ):
        return PacbinMemberKind.COLOR_CONTRACTION
    return None


def _existing_evaluator_container_path(
    existing: ArtifactManifest | None,
) -> str | None:
    if existing is None:
        return None
    raw = existing.extensions.get(_EVALUATOR_PAYLOAD_CONTAINER_EXTENSION)
    if raw is None:
        return None
    extension = _mapping(raw)
    if str(extension.get("path")) != _EVALUATOR_PAYLOAD_CONTAINER_PATH:
        raise ValueError("append artifact has an incompatible evaluator container path")
    return _EVALUATOR_PAYLOAD_CONTAINER_PATH


def _evaluator_payload_container_extension(index: PacbinIndex) -> dict[str, object]:
    return {
        "kind": _EVALUATOR_PAYLOAD_CONTAINER_KIND,
        "schema_version": _EVALUATOR_PAYLOAD_CONTAINER_SCHEMA_VERSION,
        "storage_abi": _EVALUATOR_PAYLOAD_CONTAINER_STORAGE_ABI,
        "path": _EVALUATOR_PAYLOAD_CONTAINER_PATH,
        "member_count": len(index.members),
        "unpacked_size_bytes": sum(member.length for member in index.members),
        "index_sha256": index.index_sha256,
    }


def _write_recurrence_schedule_roots(
    builder: ArtifactBuilder,
    evaluator_payloads: _EvaluatorPayloadCollector,
    plan: RecurrenceScheduleSharingPlan,
) -> dict[str, object]:
    """Stage each interned schedule once and publish its root binding index."""

    for schedule in plan.schedules:
        record = evaluator_payloads.add_file(
            schedule.artifact_path,
            schedule.source_path,
            process_id=None,
        )
        if record.sha256 != schedule.sha256 or record.size_bytes != schedule.size_bytes:
            raise ValueError(
                f"interned recurrence schedule {schedule.digest} changed before "
                "root publication"
            )
    index_record = builder.add_json(
        RECURRENCE_SCHEDULE_INDEX_PATH,
        plan.to_mapping(),
        role="evaluator-manifest",
        compact=True,
    )
    return plan.extension_mapping(index_sha256=index_record.sha256)


def _validated_recurrence_helicity_selector_schedule_digest(
    companion: RecurrenceHelicitySelectorPlanArtifact,
) -> str:
    base_digest = _canonical_sha256(
        companion.recurrence_base_schedule_digest,
        "recurrence helicity-selector base schedule digest",
    )
    native_digest = _canonical_sha256(
        companion.recurrence_native_schedule_semantic_digest,
        "recurrence helicity-selector native schedule digest",
    )
    if base_digest != native_digest:
        raise ValueError(
            "recurrence helicity-selector plan must use relation-free native "
            "schedule identity"
        )
    digest = _canonical_sha256(
        companion.recurrence_schedule_digest,
        "recurrence helicity-selector schedule digest",
    )
    expected = recurrence_helicity_selector_schedule_digest(
        base_digest,
        companion.helicity_dispatch_sha256,
    )
    if digest != expected:
        raise ValueError(
            "recurrence helicity-selector root digest does not bind its Direct "
            "plan and helicity dispatch"
        )
    return digest


def _recurrence_helicity_selector_schedule_path(
    companion: RecurrenceHelicitySelectorPlanArtifact,
) -> str:
    digest = _validated_recurrence_helicity_selector_schedule_digest(companion)
    return (
        f"{_RECURRENCE_HELICITY_SELECTOR_SCHEDULE_ROOT}/{digest}/"
        f"{_RECURRENCE_RUNTIME_CONTAINER_PATH}"
    )


def _recurrence_helicity_dispatch_path(
    companion: RecurrenceHelicitySelectorPlanArtifact,
) -> str:
    digest = _validated_recurrence_helicity_selector_schedule_digest(companion)
    return (
        f"{_RECURRENCE_HELICITY_SELECTOR_SCHEDULE_ROOT}/{digest}/"
        f"{_RECURRENCE_HELICITY_DISPATCH_PATH}"
    )


def _write_recurrence_helicity_selector_schedule_roots(
    evaluator_payloads: _EvaluatorPayloadCollector,
    processes: Sequence[ProcessArtifact],
    *,
    recurrence_sharing: RecurrenceScheduleSharingPlan | None,
) -> None:
    """Stage every persisted selector schedule once by semantic digest."""

    primary_by_digest = (
        {}
        if recurrence_sharing is None
        else {schedule.digest: schedule for schedule in recurrence_sharing.schedules}
    )
    selector_by_digest: dict[str, RecurrenceHelicitySelectorPlanArtifact] = {}
    reused_primary_digests: set[str] = set()
    for process in processes:
        if not isinstance(process, RecurrenceProcessArtifact):
            continue
        companion = process.helicity_selector_companion
        if not isinstance(companion, RecurrenceHelicitySelectorPlanArtifact):
            continue
        digest = _validated_recurrence_helicity_selector_schedule_digest(companion)
        primary = primary_by_digest.get(digest)
        if primary is not None:
            if (
                primary.sha256 != companion.recurrence_schedule_sha256
                or primary.size_bytes != companion.recurrence_schedule_size_bytes
                or primary.member_count != companion.recurrence_schedule_member_count
                or primary.unpacked_size_bytes
                != companion.recurrence_schedule_unpacked_size_bytes
                or primary.index_sha256 != companion.recurrence_schedule_index_sha256
            ):
                raise ValueError(
                    "recurrence selector and primary schedule share a digest but "
                    "not an identical payload"
                )
            selector_by_digest[digest] = companion
            reused_primary_digests.add(digest)
            continue
        previous = selector_by_digest.get(digest)
        if previous is not None:
            if (
                previous.recurrence_schedule_sha256
                != companion.recurrence_schedule_sha256
                or previous.recurrence_schedule_size_bytes
                != companion.recurrence_schedule_size_bytes
                or previous.recurrence_schedule_member_count
                != companion.recurrence_schedule_member_count
                or previous.recurrence_schedule_unpacked_size_bytes
                != companion.recurrence_schedule_unpacked_size_bytes
                or previous.recurrence_schedule_index_sha256
                != companion.recurrence_schedule_index_sha256
                or previous.helicity_dispatch_sha256
                != companion.helicity_dispatch_sha256
                or previous.helicity_dispatch_size_bytes
                != companion.helicity_dispatch_size_bytes
                or previous.helicity_dispatch_base_runtime_layout_digest
                != companion.helicity_dispatch_base_runtime_layout_digest
                or previous.helicity_dispatch_resolved_helicity_count
                != companion.helicity_dispatch_resolved_helicity_count
            ):
                raise ValueError(
                    "recurrence helicity-selector schedule digest maps to "
                    "different payloads"
                )
            continue
        selector_by_digest[digest] = companion

    for digest, companion in sorted(selector_by_digest.items()):
        if digest not in reused_primary_digests:
            record = evaluator_payloads.add_file(
                _recurrence_helicity_selector_schedule_path(companion),
                companion.recurrence_schedule_path,
                process_id=None,
            )
            if (
                record.sha256 != companion.recurrence_schedule_sha256
                or record.size_bytes != companion.recurrence_schedule_size_bytes
            ):
                raise ValueError(
                    f"persisted recurrence helicity-selector schedule {digest} "
                    "changed before publication"
                )
        dispatch_record = evaluator_payloads.add_file(
            _recurrence_helicity_dispatch_path(companion),
            companion.helicity_dispatch_path,
            process_id=None,
        )
        if (
            dispatch_record.sha256 != companion.helicity_dispatch_sha256
            or dispatch_record.size_bytes != companion.helicity_dispatch_size_bytes
        ):
            raise ValueError(
                f"persisted recurrence helicity dispatch {digest} changed before "
                "publication"
            )


def _write_umami_bundle(
    builder: ArtifactBuilder,
    *,
    processes: Sequence[ProcessArtifact],
    process_records: Sequence[Mapping[str, object]],
    compiled_model: CompiledModel,
    process_groups: ProcessGroupMap | None = None,
) -> None:
    """Add export-only selector tables after runtime physics has been staged."""
    from ..artifacts.umami import UmamiProcessInput, umami_bundle_payloads
    from ..artifacts.umami_otf import restore_umami_on_the_fly_input

    new_processes = {process.process_id: process for process in processes}
    inputs = []
    previous_umami = None
    for record in process_records:
        identifier = str(record["id"])
        process = new_processes.get(identifier)
        physics = (
            process.umami_physics
            if process is not None and process.umami_physics is not None
            else json.loads(
                builder.staged_path(str(record["physics_path"])).read_text(
                    encoding="utf-8"
                )
            )
        )
        umami_input = UmamiProcessInput(
            process_id=identifier,
            expression=str(record["expression"]),
            color_accuracy=str(record["color_accuracy"]),
            external_pdgs=tuple(int(pdg) for pdg in record["external_pdgs"]),
            physics=physics,
            aliases=tuple(record.get("aliases", ())),
            color_sectors=() if process is None else process.umami_color_sectors,
            structural_keys=(
                None if process is None else process.umami_structural_keys
            ),
            # Flavour-blind members were not generated: they are exported as
            # selectors evaluated through this representative's runtime.
            group_members=(
                ()
                if process_groups is None or identifier not in process_groups.groups
                else tuple(
                    member
                    for member in process_groups.groups[identifier].members[1:]
                    if not member.generated
                )
            ),
        )
        if physics.get("kind") == ON_THE_FLY_PUBLIC_METADATA_KIND:
            # Appended processes retain compact runtime physics. Their previous
            # SDK already contains every physical selector and colour tag.
            if previous_umami is None:
                try:
                    previous = json.loads(
                        builder.staged_path("API/umami/metadata.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    previous_umami = (
                        [
                            json.loads(
                                builder.staged_path(
                                    "API/umami/" + str(provider["metadata"])
                                ).read_text(encoding="utf-8")
                            )
                            for provider in previous["providers"]
                        ]
                        if previous.get("kind") == "pyamplicol-umami-provider-index"
                        else [previous]
                    )
                except FileNotFoundError as exc:
                    raise ValueError(
                        "appending this OTF artifact requires its existing UMAMI "
                        "metadata; regenerate the complete process set"
                    ) from exc
            umami_input = restore_umami_on_the_fly_input(umami_input, previous_umami)
        inputs.append(umami_input)
    for payload in umami_bundle_payloads(
        processes=inputs,
        compiled_model=compiled_model.to_dict(),
        artifact_id=compute_artifact_id(
            {"payloads": [record.as_dict() for record in builder.payload_records()]}
        ),
    ):
        builder.add_bytes(
            payload.path,
            payload.content,
            role=payload.role,
            media_type=payload.media_type,
            executable=payload.executable,
        )


def write_schema_v3_artifact(
    destination: str | Path,
    *,
    mode: Literal["error", "append", "replace"],
    source: ModelSource,
    compiled_model: CompiledModel,
    configuration: _GenerationConfigProvenance,
    processes: Sequence[ProcessArtifact],
    timings: Mapping[str, float],
    api_bundle_hook: ApiBundleHook | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    payload_hook: ArtifactPayloadHook | None = None,
    process_groups: ProcessGroupMap | None = None,
) -> ArtifactWriteResult:
    if not processes:
        raise ValueError("schema-v3 generation requires at least one concrete process")
    output = Path(destination).expanduser().resolve(strict=False)
    existing = load_manifest(output) if mode == "append" else None
    if existing is not None:
        _reject_legacy_eager_append(existing)
    hook = api_bundle_hook or _default_api_bundle_hook()
    requested_config = _config_payload(configuration.requested)
    effective_config = _config_payload(configuration.effective)
    bundle_requested = _bundle_requested(configuration.effective)
    existing_bundle = _existing_bundle_path(existing)
    can_emit_bundle = hook is not None or existing_bundle is not None
    effective_config = _effective_config_payload(
        effective_config,
        disable_api_bundle=bundle_requested and not can_emit_bundle,
    )
    adjustments = [
        {"path": adjustment.path, "reason": adjustment.reason}
        for adjustment in configuration.adjustments
    ]
    if bundle_requested and not can_emit_bundle:
        adjustments.append(
            {
                "path": "generation.emit_api_bundle",
                "reason": "no root API-bundle emitter is installed",
            }
        )
    requested_bytes = _toml_bytes(requested_config)
    effective_bytes = _toml_bytes(effective_config)
    required_runtime_capabilities = set(
        _existing_required_runtime_capabilities(existing)
    )
    for process in processes:
        required_runtime_capabilities.update(_process_runtime_capabilities(process))
    canonical_runtime_capabilities = tuple(sorted(required_runtime_capabilities))
    producer = _producer_metadata(
        configuration.effective,
        runtime_capabilities=canonical_runtime_capabilities,
        implicit_portable_jit_evidence=_implicit_generation_portable_jit_evidence(
            processes
        ),
    )
    model = _model_metadata(source, compiled_model)
    dependencies = _dependency_metadata(source)
    eager_pack_identity = _eager_prepared_pack_identity(
        existing,
        compiled_model=compiled_model,
        processes=processes,
    )
    recurrence_processes = tuple(
        process
        for process in processes
        if isinstance(process, RecurrenceProcessArtifact)
    )
    if existing is not None and recurrence_processes:
        raise ValueError(
            "recurrence root schedules and process bindings are immutable; "
            "regenerate the recurrence process set instead of appending"
        )
    recurrence_sharing_plan = (
        intern_recurrence_schedules(recurrence_processes)
        if existing is None and recurrence_processes
        else None
    )
    _validate_append_compatibility(
        existing,
        producer=producer,
        model=model,
        eager_pack_identity=eager_pack_identity,
        requested_bytes=requested_bytes,
        effective_bytes=effective_bytes,
        adjustments=adjustments,
        processes=processes,
    )
    if existing is not None:
        producer = _plain_mapping(existing.producer)
        model = _plain_mapping(existing.model)
        dependencies = tuple(_plain_mapping(item) for item in existing.dependencies)

    target = _mapping(producer["target"])
    process_records = _existing_process_records(existing)
    evaluator_entries = _existing_evaluator_entries(existing)
    execution_manifest_sha256_by_process: dict[str, str] = {}
    validation_records = tuple(process.validation_point for process in processes)
    validations = validation_point_map(validation_records)
    bundle_points = _existing_bundle_points(existing)
    bundle_points.update(build_api_validation_points(processes))
    api_bundle_path = existing_bundle
    write_mode = cast("ArtifactWriteMode", mode)
    with ArtifactBuilder(
        output,
        mode=write_mode,
        expected_artifact_id=(existing.artifact_id if existing is not None else None),
    ) as builder:
        evaluator_payloads = _EvaluatorPayloadCollector(
            builder,
            existing=existing,
            target=target,
        )
        recurrence_sharing_extension = (
            _write_recurrence_schedule_roots(
                builder,
                evaluator_payloads,
                recurrence_sharing_plan,
            )
            if recurrence_sharing_plan is not None
            else None
        )
        _write_recurrence_helicity_selector_schedule_roots(
            evaluator_payloads,
            processes,
            recurrence_sharing=recurrence_sharing_plan,
        )
        if existing is None:
            if progress_callback is not None:
                progress_callback(
                    {
                        "step": "global payloads",
                        "completed": 0,
                        "total": len(processes) + 2,
                    }
                )
            _write_global_payloads(
                builder,
                compiled_model=compiled_model,
                requested_bytes=requested_bytes,
                effective_bytes=effective_bytes,
            )
        retain_recurrence_templates = (
            RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY in required_runtime_capabilities
            or ON_THE_FLY_RUNTIME_CAPABILITY in required_runtime_capabilities
            or RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY
            in required_runtime_capabilities
        )
        eager_kernel_ids = _prepared_kernel_ids(
            output,
            existing,
            compiled_model=compiled_model,
            processes=processes,
            retain_recurrence_templates=retain_recurrence_templates,
        )
        if eager_kernel_ids:
            if progress_callback is not None:
                progress_callback(
                    {
                        "step": "prepared kernel pack",
                        "completed": 1,
                        "total": len(processes) + 2,
                        "kernel_count": len(eager_kernel_ids),
                    }
                )
            _write_eager_kernel_pack(
                builder,
                compiled_model,
                kernel_ids=eager_kernel_ids,
                evaluator_payloads=evaluator_payloads,
                require_eager_direct=(
                    EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY
                    in required_runtime_capabilities
                    or ON_THE_FLY_RUNTIME_CAPABILITY in required_runtime_capabilities
                    or RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY
                    in required_runtime_capabilities
                ),
                retain_recurrence_templates=retain_recurrence_templates,
            )
        for process_index, process in enumerate(processes, start=1):
            if progress_callback is not None:
                progress_callback(
                    {
                        "step": "process payloads",
                        "completed": process_index,
                        "total": len(processes) + 2,
                        "process": process.process_id,
                    }
                )
            record, evaluator_entry, execution_sha256 = _write_process_payloads(
                builder,
                process,
                evaluator_payloads=evaluator_payloads,
                recurrence_sharing=recurrence_sharing_plan,
            )
            process_records.append(record)
            evaluator_entries.append(evaluator_entry)
            execution_manifest_sha256_by_process[process.process_id] = execution_sha256
        builder.add_json(
            _EVALUATOR_SET_PATH,
            {
                "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
                "kind": "pyamplicol-runtime-execution-set",
                "required_runtime_capabilities": list(canonical_runtime_capabilities),
                "processes": evaluator_entries,
            },
            role="evaluator-manifest",
        )
        if bundle_requested and hook is not None:
            api_bundle_path = _call_api_bundle_hook(builder, hook, bundle_points)
        evaluator_payload_container = evaluator_payloads.publish()
        source_revision = producer.get("git_revision")
        native_build_inputs_sha256 = producer.get("native_build_inputs_sha256")
        evaluator_container_path = (
            None
            if evaluator_payload_container is None
            else str(evaluator_payload_container["path"])
        )
        evaluator_container_index_sha256 = (
            None
            if evaluator_payload_container is None
            else str(evaluator_payload_container["index_sha256"])
        )
        if isinstance(source_revision, str) and isinstance(
            native_build_inputs_sha256, str
        ):
            staged_root = builder.root
            if staged_root is None:  # pragma: no cover - builder invariant
                raise RuntimeError("artifact builder is not active")
            for process_record in process_records:
                process_id = str(process_record["id"])
                execution_path = f"processes/{process_id}/execution.json"
                execution_file = builder.staged_path(execution_path)
                execution_payload = json.loads(
                    execution_file.read_text(encoding="utf-8")
                )
                if not isinstance(execution_payload, Mapping):
                    raise ValueError(
                        f"execution payload for {process_id!r} is not an object"
                    )
                if execution_payload.get("kind") == ON_THE_FLY_RUNTIME_KIND:
                    continue
                proof = build_generation_structural_proof(
                    artifact_root=staged_root,
                    process_id=process_id,
                    source_revision=source_revision,
                    native_build_inputs_sha256=native_build_inputs_sha256,
                    execution_path=execution_path,
                    execution_sha256=sha256_file(execution_file),
                    execution=execution_payload,
                    evaluator_container_path=evaluator_container_path,
                    evaluator_container_index_sha256=(evaluator_container_index_sha256),
                )
                builder.add_json(
                    f"processes/{process_id}/structural-source-proof.json",
                    proof,
                    role=STRUCTURAL_SOURCE_PROOF_ROLE,
                    process_id=process_id,
                    compact=True,
                )
        _write_recurrence_bootstraps(
            builder,
            processes=recurrence_processes,
            process_records=process_records,
            producer=producer,
            runtime={
                "engine": "rusticol",
                "engine_version": str(producer["version"]),
                "evaluator_manifest_path": _EVALUATOR_SET_PATH,
                "api_bundle_path": api_bundle_path,
                "required_runtime_capabilities": list(canonical_runtime_capabilities),
            },
            evaluator_payload_container=evaluator_payload_container,
            target=target,
        )
        # Written after the recurrence bootstrap froze its payload inventory:
        # the map is rewritten in place later and is not part of that image.
        group_map = _write_process_groups(
            builder,
            process_groups=process_groups,
            process_records=process_records,
        )
        extensions = _extensions(
            existing,
            processes=processes,
            timings=timings,
            api_bundle_requested=bundle_requested,
            api_bundle_path=api_bundle_path,
            eager_pack_identity=eager_pack_identity,
            execution_manifest_sha256_by_process=(execution_manifest_sha256_by_process),
            evaluator_payload_container=evaluator_payload_container,
            recurrence_schedule_sharing=recurrence_sharing_extension,
        )
        if payload_hook is not None:
            extensions.update(payload_hook(builder))
        if bundle_requested and hook is not None:
            _write_umami_bundle(
                builder,
                processes=processes,
                process_records=process_records,
                compiled_model=compiled_model,
                process_groups=group_map,
            )
        builder.finalize(
            kind=(
                "pyamplicol-process"
                if len(process_records) == 1
                else "pyamplicol-process-set"
            ),
            producer=producer,
            model=model,
            configuration={
                "toml_schema_version": 1,
                "requested_path": _CONFIG_REQUESTED_PATH,
                "effective_path": _CONFIG_EFFECTIVE_PATH,
                "adjustments": adjustments,
            },
            processes=process_records,
            default_process_id=str(process_records[0]["id"]),
            runtime={
                "engine": "rusticol",
                "engine_version": str(producer["version"]),
                "evaluator_manifest_path": _EVALUATOR_SET_PATH,
                "api_bundle_path": api_bundle_path,
                "required_runtime_capabilities": list(canonical_runtime_capabilities),
            },
            dependencies=dependencies,
            extensions=extensions,
        )
        staged = load_manifest(builder.root)
        _validate_artifact_references(staged)
        if progress_callback is not None:
            progress_callback(
                {
                    "step": "publishing artifact",
                    "completed": len(processes) + 2,
                    "total": len(processes) + 2,
                    "file_count": len(staged.payloads) + 1,
                }
            )

    manifest = load_manifest(output)
    _validate_artifact_references(manifest)
    files = tuple(
        [output / record.path for record in manifest.payloads]
        + [output / "artifact.json"]
    )
    return ArtifactWriteResult(
        output=output,
        files=files,
        validation_points=validations,
        api_bundle_path=api_bundle_path,
    )


def _write_process_groups(
    builder: ArtifactBuilder,
    *,
    process_groups: ProcessGroupMap | None,
    process_records: Sequence[Mapping[str, object]],
) -> ProcessGroupMap:
    """Stage ``process-groups.json`` covering every process of the artifact.

    An append keeps the stored groups, and a newly generated process leaves a
    group in which it was only an assumed member. Processes without a recorded
    group map to themselves.
    """

    stored = builder.staged_path(PROCESS_GROUPS_PATH)
    group_map = (
        ProcessGroupMap.from_json(stored.read_text(encoding="utf-8"))
        if stored.is_file()
        else ProcessGroupMap()
    )
    record_ids = [str(record["id"]) for record in process_records]
    new_ids = set(record_ids).difference(
        member.process_id for member in group_map.members() if member.generated
    )
    for group in group_map.groups.values():
        group.members = [
            member
            for member in group.members
            if member.generated or member.process_id not in new_ids
        ]
    if process_groups is not None:
        for group in process_groups.groups.values():
            group_map.add_group(
                [
                    member
                    for member in group.members
                    if member.generated or group_map.group_of(member.process_id) is None
                ]
            )
        group_map.analyses.extend(process_groups.analyses)
    grouped = {member.process_id for member in group_map.members()}
    ungrouped = initial_process_groups(
        (
            str(record["id"]),
            str(record["expression"]),
            cast(Sequence[int], record["external_pdgs"]),
        )
        for record in process_records
        if str(record["id"]) not in grouped
    )
    for group in ungrouped.groups.values():
        group_map.add_group(group.members)
    validate_process_groups(group_map, record_ids)
    builder.add_bytes(
        PROCESS_GROUPS_PATH,
        process_groups_payload(group_map),
        role=PROCESS_GROUPS_ROLE,
        media_type=PROCESS_GROUPS_MEDIA_TYPE,
    )
    return group_map


def _write_global_payloads(
    builder: ArtifactBuilder,
    *,
    compiled_model: CompiledModel,
    requested_bytes: bytes,
    effective_bytes: bytes,
) -> None:
    builder.add_bytes(
        _CONFIG_REQUESTED_PATH,
        requested_bytes,
        role="configuration-requested",
        media_type="application/toml",
    )
    builder.add_bytes(
        _CONFIG_EFFECTIVE_PATH,
        effective_bytes,
        role="configuration-effective",
        media_type="application/toml",
    )
    builder.add_json(
        _COMPILED_MODEL_PATH,
        compiled_model.to_dict(),
        role="compiled-model",
    )
    builder.add_json(
        _MODEL_PARAMETERS_PATH,
        {
            "schema_version": 1,
            "kind": "pyamplicol-model-parameter-defaults",
            "parameters": {
                name: [float(value[0]), float(value[1])]
                for name, value in sorted(compiled_model.parameter_defaults.items())
            },
        },
        role="model-parameters",
    )


def _write_eager_kernel_pack(
    builder: ArtifactBuilder,
    compiled_model: CompiledModel,
    *,
    kernel_ids: frozenset[int],
    evaluator_payloads: _EvaluatorPayloadCollector,
    require_eager_direct: bool,
    retain_recurrence_templates: bool,
) -> None:
    bundle = compiled_model.prepared_bundle
    if bundle is None:
        raise ValueError(
            "prepared-kernel artifact writing requires a prepared model bundle"
        )
    selected = tuple(
        kernel
        for kernel in bundle.kernel_pack.kernels
        if kernel.kernel_id in kernel_ids
    )
    if {kernel.kernel_id for kernel in selected} != set(kernel_ids):
        missing = sorted(set(kernel_ids) - {kernel.kernel_id for kernel in selected})
        raise ValueError(f"prepared model omits referenced eager kernels {missing}")
    selected_variants = tuple(
        variant
        for variant in bundle.kernel_pack.kernel_variants
        if variant.base_kernel_id in kernel_ids
    )
    builder.discard_payloads(_EAGER_KERNEL_PACK_PATH)
    builder.discard_payloads(_EAGER_KERNEL_PAYLOAD_ROOT, recursive=True)
    evaluator_payloads.discard_prefix(_EAGER_KERNEL_PAYLOAD_ROOT)
    pack_payload = bundle.kernel_pack.to_dict()
    pack_payload["eager_kernel_abi"] = EAGER_KERNEL_ABI
    if not retain_recurrence_templates:
        # Process artifacts retain only the prepared kernels they execute.  A
        # model-wide recurrence catalog references kernels outside that
        # process-local inventory and therefore cannot remain a valid member
        # of an eager-only kernel pack.  Eager exact execution uses the
        # prepared kernel records and resolver manifest, not recurrence
        # templates.
        pack_payload["recurrence_template"] = None
        pack_payload["recurrence_direct_template"] = None
    direct_descriptors: dict[str, bytes] = {}
    kernel_payloads: list[dict[str, object]] = []
    for kernel in selected:
        payload = kernel.to_dict()
        if require_eager_direct and _requires_eager_direct_table(kernel.contract_kind):
            manifest = _mapping(payload["f64_evaluator_manifest"])
            if bundle.kernel_pack.backend == "jit":
                direct_manifest, descriptor_path, descriptor = (
                    _eager_direct_evaluator_manifest(
                        bundle=bundle,
                        kernel_id=kernel.kernel_id,
                        input_complex_count=kernel.input_arity,
                        output_complex_count=kernel.output_arity,
                        manifest=manifest,
                    )
                )
                direct_descriptors[descriptor_path] = descriptor
            else:
                direct_manifest = _eager_native_direct_evaluator_manifest(
                    backend=bundle.kernel_pack.backend,
                    bundle=bundle,
                    kernel_id=kernel.kernel_id,
                    input_complex_count=kernel.input_arity,
                    output_complex_count=kernel.output_arity,
                    manifest=manifest,
                )
            payload["f64_evaluator_manifest"] = direct_manifest
        kernel_payloads.append(payload)
    pack_payload["kernels"] = kernel_payloads
    pack_payload["kernel_variants"] = [
        variant.to_dict() for variant in selected_variants
    ]
    pack_payload["resolver_manifest"] = _filtered_eager_resolver_manifest(
        bundle.kernel_pack.resolver_manifest,
        kernel_ids,
    )
    from ..models.prepared import PreparedKernelPack

    validated_pack_payload = dict(pack_payload)
    validated_pack_payload.pop("eager_kernel_abi")
    PreparedKernelPack.from_dict(validated_pack_payload)
    builder.add_json(
        _EAGER_KERNEL_PACK_PATH,
        pack_payload,
        role="evaluator-manifest",
        compact=True,
    )
    for descriptor_path, descriptor in sorted(direct_descriptors.items()):
        evaluator_payloads.add_bytes(
            f"{_EAGER_KERNEL_PAYLOAD_ROOT}/{descriptor_path}",
            descriptor,
            process_id=None,
            media_type="application/octet-stream",
        )
    referenced_payloads = {
        path
        for record in (*selected, *selected_variants)
        for path in record.referenced_payload_paths
    }
    with zipfile.ZipFile(bundle.path, "r") as archive:
        for member_path in sorted(referenced_payloads):
            with archive.open(member_path, "r") as stream:
                evaluator_payloads.add_stream(
                    f"{_EAGER_KERNEL_PAYLOAD_ROOT}/{member_path}",
                    cast("BinaryIO", stream),
                    process_id=None,
                    media_type=_media_type(Path(member_path)),
                )


def _requires_eager_direct_table(contract_kind: str) -> bool:
    """Return whether a prepared kernel executes inside the eager point loop."""

    # Model-parameter derivation is evaluated separately when the artifact is
    # loaded or an override is applied.  It remains on the ordinary evaluator
    # path, which supports scalar operations such as square roots that are
    # deliberately outside the call-free eager DirectTable ABI.
    return contract_kind != "model-parameter"


def _eager_direct_evaluator_manifest(
    *,
    bundle: object,
    kernel_id: int,
    input_complex_count: int,
    output_complex_count: int,
    manifest: Mapping[str, object],
) -> tuple[dict[str, object], str, bytes]:
    if manifest.get("kind") != "symjit-application-evaluator":
        raise ValueError(
            f"eager-direct-arena-v1 kernel {kernel_id} is not a SymJIT application"
        )
    plane_application = manifest.get("plane_application")
    if not isinstance(plane_application, Mapping):
        raise ValueError(
            f"eager-direct-arena-v1 kernel {kernel_id} predates the "
            "SymJIT plane-application ABI; regenerate the prepared model"
        )
    application_path = plane_application.get("application_path")
    if not isinstance(application_path, str) or not application_path:
        raise ValueError(
            f"eager-direct-arena-v1 kernel {kernel_id} has no plane application"
        )
    if (
        plane_application.get("application_abi") != SYMJIT_PLANE_APPLICATION_ABI
        or plane_application.get("storage_abi") != SYMJIT_APPLICATION_ABI
        or plane_application.get("input_complex_count") != input_complex_count
        or plane_application.get("output_complex_count") != output_complex_count
    ):
        raise ValueError(
            f"eager-direct-arena-v1 kernel {kernel_id} has an incompatible "
            "plane-application contract"
        )
    reader = getattr(bundle, "read_payload", None)
    if not callable(reader):  # pragma: no cover - internal type invariant
        raise TypeError("prepared model bundle cannot read evaluator payloads")
    source = reader(application_path)
    if not isinstance(source, bytes):  # pragma: no cover - bundle invariant
        raise TypeError("prepared model bundle returned a non-byte evaluator payload")
    started = time.perf_counter()
    descriptor = _derive_eager_direct_descriptor(
        source,
        input_complex_count=input_complex_count,
        output_complex_count=output_complex_count,
    )
    descriptor_s = time.perf_counter() - started
    descriptor_path = f"kernels/{kernel_id}/eager-direct-table-descriptor-v1.bin"
    result = _plain_mapping(manifest)
    timing = _plain_mapping(_mapping(result.get("build_timing", {})))
    timing["eager_direct_descriptor_s"] = descriptor_s
    result["build_timing"] = timing
    result["direct_table"] = {
        "capability": EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        "source_application_abi": SYMJIT_PLANE_APPLICATION_ABI,
        "descriptor_abi": EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
        "binding_abi": EAGER_DIRECT_TABLE_BINDING_ABI,
        "descriptor_path": descriptor_path,
        "descriptor_size_bytes": len(descriptor),
        "descriptor_sha256": hashlib.sha256(descriptor).hexdigest(),
        "input_complex_count": input_complex_count,
        "output_complex_count": output_complex_count,
    }
    return result, descriptor_path, descriptor


def _eager_native_direct_evaluator_manifest(
    *,
    backend: str,
    bundle: object,
    kernel_id: int,
    input_complex_count: int,
    output_complex_count: int,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    del bundle
    if backend not in {"cpp", "asm"}:
        raise ValueError(f"unsupported eager native DirectTable backend {backend!r}")
    if manifest.get("kind") != "compiled-complex-evaluator":
        raise ValueError(
            f"eager-direct-arena-v1 kernel {kernel_id} is not a native "
            "compiled evaluator"
        )
    expected_capability = {
        "cpp": SYMBOLICA_CPP_RUNTIME_CAPABILITY,
        "asm": SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    }[backend]
    if manifest.get("runtime_capability") != expected_capability:
        raise ValueError(
            f"eager-direct-arena-v1 kernel {kernel_id} native capability "
            "does not match its prepared backend"
        )
    direct = _mapping(manifest.get("direct_table"))
    from .._internal.versions import (
        NATIVE_EAGER_DIRECT_TABLE_APPLICATION_ABI,
    )

    expected = {
        "capability": EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        "source_application_abi": (NATIVE_EAGER_DIRECT_TABLE_APPLICATION_ABI),
        "descriptor_abi": EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
        "binding_abi": EAGER_DIRECT_TABLE_BINDING_ABI,
    }
    for field, value in expected.items():
        if direct.get(field) != value:
            raise ValueError(
                f"eager-direct-arena-v1 kernel {kernel_id} native "
                f"DirectTable {field} is incompatible"
            )
    library_path = direct.get("library_path")
    function_name = direct.get("function_name")
    if (
        not isinstance(library_path, str)
        or not library_path
        or library_path != manifest.get("library_path")
        or not isinstance(function_name, str)
        or not function_name
    ):
        raise ValueError(
            f"eager-direct-arena-v1 kernel {kernel_id} native DirectTable "
            "library binding is invalid"
        )
    if (
        direct.get("input_complex_count") != input_complex_count
        or direct.get("output_complex_count") != output_complex_count
    ):
        raise ValueError(
            f"eager-direct-arena-v1 kernel {kernel_id} native DirectTable "
            "width is incompatible"
        )
    return _plain_mapping(manifest)


def _derive_eager_direct_descriptor(
    source_application: bytes,
    *,
    input_complex_count: int,
    output_complex_count: int,
) -> bytes:
    operation = _eager_direct_descriptor_operation()
    descriptor = operation(
        source_application,
        input_complex_count,
        output_complex_count,
    )
    if not isinstance(descriptor, bytes) or not descriptor:
        raise RuntimeError("Rusticol returned an invalid eager DirectTable descriptor")
    return descriptor


@cache
def _eager_direct_descriptor_operation() -> Callable[[bytes, int, int], bytes]:
    try:
        rusticol = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(rusticol)
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "eager-direct-arena-v1 descriptor generation requires the current "
            "pyamplicol native extension"
        ) from exc
    operation = getattr(rusticol, "_eager_direct_descriptor_v1", None)
    if not callable(operation):
        raise RuntimeError(
            "the current pyamplicol native extension does not expose "
            "_eager_direct_descriptor_v1; rebuild it before generating artifacts"
        )
    return operation


def _prepared_kernel_ids(
    output: Path,
    existing: ArtifactManifest | None,
    *,
    compiled_model: CompiledModel,
    processes: Sequence[ProcessArtifact],
    retain_recurrence_templates: bool,
) -> frozenset[int]:
    has_prepared_process = any(
        _is_prepared_kernel_process(process) for process in processes
    )
    kernel_ids = {
        kernel_id
        for process in processes
        if _is_prepared_kernel_process(process)
        for kernel_id in _prepared_referenced_kernel_ids(process)
    }
    has_existing_pack = (
        existing is not None and (output / _EAGER_KERNEL_PACK_PATH).is_file()
    )
    if has_existing_pack:
        try:
            prior = json.loads(
                (output / _EAGER_KERNEL_PACK_PATH).read_text(encoding="utf-8")
            )
            kernel_ids.update(
                int(_mapping(item)["kernel_id"])
                for item in _sequence(_mapping(prior)["kernels"])
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                "existing eager kernel pack is malformed; replace the artifact"
            ) from exc
    if has_prepared_process or has_existing_pack:
        bundle = compiled_model.prepared_bundle
        if bundle is None:
            raise ValueError(
                "prepared-kernel artifact writing requires a prepared model bundle"
            )
        parameter_kernel_id = bundle.kernel_pack.resolver_manifest.get(
            "model_parameter_kernel_id"
        )
        if parameter_kernel_id is not None:
            kernel_ids.add(int(parameter_kernel_id))
        if retain_recurrence_templates:
            # The authenticated recurrence companions are model-wide rather
            # than process-local.  Retaining them therefore requires their
            # complete prepared-kernel inventory, including bindings that a
            # particular process schedule does not happen to exercise.
            kernel_ids.update(kernel.kernel_id for kernel in bundle.kernel_pack.kernels)
    return frozenset(kernel_ids)


def _eager_prepared_pack_identity(
    existing: ArtifactManifest | None,
    *,
    compiled_model: CompiledModel,
    processes: Sequence[ProcessArtifact],
) -> dict[str, object] | None:
    existing_uses_eager = existing is not None and (
        bool(
            {
                EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
                EAGER_RUNTIME_CAPABILITY,
                EAGER_PLAN_V3_RUNTIME_CAPABILITY,
                RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
                RECURRENCE_COLOR_RUNTIME_CAPABILITY,
                ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
                ON_THE_FLY_RUNTIME_CAPABILITY,
                ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
                RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY,
            }.intersection(_required_runtime_capabilities(existing.runtime))
        )
        or any(record.path == _EAGER_KERNEL_PACK_PATH for record in existing.payloads)
    )
    incoming_uses_eager = any(
        _is_prepared_kernel_process(process) for process in processes
    )
    if not existing_uses_eager and not incoming_uses_eager:
        return None
    bundle = compiled_model.prepared_bundle
    if bundle is None:
        raise ValueError(
            "prepared-kernel artifact writing requires a prepared model bundle"
        )
    return {
        "kind": _EAGER_PACK_IDENTITY_KIND,
        "schema_version": _EAGER_PACK_IDENTITY_SCHEMA_VERSION,
        "abi": PREPARED_KERNEL_PACK_IDENTITY_ABI,
        "eager_kernel_abi": EAGER_KERNEL_ABI,
        "identity_sha256": prepared_kernel_pack_manifest_identity_sha256(
            bundle.manifest
        ),
        "backend": bundle.kernel_pack.backend,
        "kernel_count": len(bundle.kernel_pack.kernels),
    }


def _is_prepared_kernel_process(process: ProcessArtifact) -> bool:
    return isinstance(
        process,
        EagerPlanV3ProcessArtifact
        | RecurrenceProcessArtifact
        | OnTheFlyProcessArtifact,
    )


def _prepared_referenced_kernel_ids(
    process: ProcessArtifact,
) -> frozenset[int]:
    if isinstance(process, RecurrenceProcessArtifact):
        return process.referenced_kernel_ids
    if isinstance(process, EagerPlanV3ProcessArtifact):
        return process.referenced_kernel_ids
    if isinstance(process, OnTheFlyProcessArtifact):
        return process.referenced_kernel_ids
    raise TypeError("compiled process has no prepared-kernel references")


def _filtered_eager_resolver_manifest(
    manifest: Mapping[str, object],
    kernel_ids: frozenset[int],
) -> dict[str, object]:
    result = _plain_mapping(manifest)
    for field in (
        "vertex_bindings",
        "propagator_bindings",
        "closure_bindings",
    ):
        if field not in manifest:
            continue
        result[field] = [
            _plain_mapping(record)
            for item in _sequence(manifest[field])
            if (record := _mapping(item)).get("kernel_id") is None
            or int(record["kernel_id"]) in kernel_ids
        ]
    parameter_kernel_id = manifest.get("model_parameter_kernel_id")
    if parameter_kernel_id is not None and int(parameter_kernel_id) not in kernel_ids:
        result["model_parameter_kernel_id"] = None
    return result


def _write_process_payloads(
    builder: ArtifactBuilder,
    process: ProcessArtifact,
    *,
    evaluator_payloads: _EvaluatorPayloadCollector,
    recurrence_sharing: RecurrenceScheduleSharingPlan | None = None,
) -> tuple[
    dict[str, object],
    dict[str, object],
    str,
]:
    prefix = f"processes/{process.process_id}"
    physics_path = f"{prefix}/physics.json"
    execution_path = f"{prefix}/execution.json"
    validation_path = f"{prefix}/validation-momenta.json"
    schema: Mapping[str, object]
    if isinstance(
        process,
        EagerPlanV3ProcessArtifact
        | RecurrenceProcessArtifact
        | OnTheFlyProcessArtifact,
    ):
        schema = {}
        physics = _mapping(process.physics)
    else:
        schema = _runtime_schema_mapping(process.runtime_schema)
        physics = _mapping(schema.get("physics"))
    if physics.get("process_id") != process.process_id:
        raise ValueError(
            f"runtime physics process ID does not match {process.process_id!r}"
        )
    if isinstance(process, OnTheFlyProcessArtifact):
        if (
            physics.get("schema_version") != 1
            or physics.get("kind") != ON_THE_FLY_PUBLIC_METADATA_KIND
            or process.color_accuracy not in {"lc", "nlc", "full"}
            or physics.get("color_accuracy") != process.color_accuracy
        ):
            raise ValueError(
                "on-the-fly generation returned incompatible compact public metadata"
            )
    elif isinstance(
        process, EagerPlanV3ProcessArtifact | RecurrenceProcessArtifact
    ) and (
        physics.get("schema_version") != RUNTIME_PHYSICS_SCHEMA_VERSION
        or physics.get("kind") != "pyamplicol-resolved-physics"
    ):
        raise ValueError(
            "Rust prepared lowering returned incompatible physics metadata"
        )
    builder.add_json(
        physics_path,
        physics,
        role="runtime-physics",
        process_id=process.process_id,
        compact=True,
    )
    if isinstance(process, EagerPlanV3ProcessArtifact):
        runtime_path = f"{prefix}/{_EAGER_RUNTIME_CONTAINER_PATH}"
        runtime_record = evaluator_payloads.add_file(
            runtime_path,
            process.eager_runtime_path,
            process_id=process.process_id,
        )
        _validate_staged_eager_runtime(process, runtime_record)
        execution_record = builder.add_bytes(
            execution_path,
            _eager_execution_summary(process),
            role="evaluator-manifest",
            media_type="application/json",
            process_id=process.process_id,
        )
    elif isinstance(process, RecurrenceProcessArtifact):
        if recurrence_sharing is None:
            raise ValueError(
                "recurrence artifacts require root schedule and process-binding "
                "publication"
            )
        binding = recurrence_sharing.binding(process.process_id)
        schedule = recurrence_sharing.schedule(binding.schedule_digest)
        binding_record = evaluator_payloads.add_bytes(
            binding.artifact_path,
            binding.payload,
            process_id=process.process_id,
        )
        if binding_record.sha256 != binding.sha256 or binding_record.size_bytes != len(
            binding.payload
        ):
            raise ValueError(
                f"recurrence process binding {process.process_id!r} changed "
                "during publication"
            )
        color_contraction_record = None
        if process.color_contraction_payload is not None:
            if process.color_contraction_summary is None:
                raise ValueError(
                    "recurrence color-contraction payload has no bounded summary"
                )
            color_contraction_record = evaluator_payloads.add_bytes(
                f"{prefix}/{_RECURRENCE_COLOR_CONTRACTION_PATH}",
                process.color_contraction_payload,
                process_id=process.process_id,
                media_type="application/octet-stream",
            )
        elif process.color_contraction_summary is not None:
            raise ValueError(
                "recurrence color-contraction summary has no binary payload"
            )
        companion_binding = None
        companion = process.helicity_selector_companion
        if isinstance(companion, RecurrenceHelicitySelectorPlanArtifact):
            if process.color_accuracy not in {"nlc", "full"}:
                raise ValueError(
                    "persisted recurrence helicity-selector plan requires NLC/full"
                )
            companion_binding_record = RecurrenceProcessBinding(
                process_id=process.process_id,
                schedule_digest=companion.recurrence_schedule_digest,
                native_schedule_semantic_digest=(
                    companion.recurrence_native_schedule_semantic_digest
                ),
                process_digest=process.process_digest,
                process_semantic_digest=companion.builder_input_sha256,
                process_support_mask=process.process_support_mask,
                remap=companion.recurrence_process_remap,
                executor_pack=companion.recurrence_process_executor_pack,
            )
            stored_binding = evaluator_payloads.add_bytes(
                f"{prefix}/{_RECURRENCE_HELICITY_SELECTOR_BINDING_PATH}",
                companion_binding_record.payload,
                process_id=process.process_id,
            )
            if (
                stored_binding.sha256 != companion_binding_record.sha256
                or stored_binding.size_bytes != len(companion_binding_record.payload)
            ):
                raise ValueError(
                    "recurrence helicity-selector process binding changed during "
                    "publication"
                )
            companion_binding = {
                **companion_binding_record.to_mapping(),
                "path": _RECURRENCE_HELICITY_SELECTOR_BINDING_PATH,
            }
        elif companion is not None:  # pragma: no cover - dataclass typing guard
            raise TypeError(
                "recurrence helicity-selector companion must be a persisted Direct plan"
            )
        execution_record = builder.add_bytes(
            execution_path,
            _bounded_recurrence_execution_summary(
                process,
                schedule_path=schedule.artifact_path,
                binding=binding.to_mapping(),
                color_contraction_record=color_contraction_record,
                companion_binding=companion_binding,
            ),
            role="evaluator-manifest",
            media_type="application/json",
            process_id=process.process_id,
        )
    elif isinstance(process, OnTheFlyProcessArtifact):
        runtime_path = f"{prefix}/{_ON_THE_FLY_RUNTIME_CONTAINER_PATH}"
        runtime_record = evaluator_payloads.add_file(
            runtime_path,
            process.runtime_path,
            process_id=process.process_id,
        )
        _validate_staged_on_the_fly_runtime(process, runtime_record)
        color_contraction_record = None
        if process.color_contraction_payload is not None:
            if process.color_contraction_summary is None:
                raise ValueError(
                    "on-the-fly color-contraction payload has no bounded summary"
                )
            color_contraction_record = evaluator_payloads.add_bytes(
                f"{prefix}/{_ON_THE_FLY_COLOR_CONTRACTION_PATH}",
                process.color_contraction_payload,
                process_id=process.process_id,
                media_type="application/octet-stream",
            )
        elif process.color_contraction_summary is not None:
            raise ValueError(
                "on-the-fly color-contraction summary has no binary payload"
            )
        execution_record = builder.add_bytes(
            execution_path,
            _on_the_fly_execution_summary(
                process,
                color_contraction_record=color_contraction_record,
            ),
            role="evaluator-manifest",
            media_type="application/json",
            process_id=process.process_id,
        )
    elif isinstance(process, CompiledProcessArtifact):
        color_contraction_payload_path = None
        if process.color_contraction_payload is not None:
            color_contraction_payload_path = _COMPILED_COLOR_CONTRACTION_MEMBER_PATH
            evaluator_payloads.add_bytes(
                f"{prefix}/{color_contraction_payload_path}",
                process.color_contraction_payload,
                process_id=process.process_id,
                media_type="application/octet-stream",
            )
        execution_record = builder.add_json(
            execution_path,
            _execution_manifest(
                process,
                schema,
                color_contraction_payload_path=color_contraction_payload_path,
            ),
            role="evaluator-manifest",
            process_id=process.process_id,
            compact=True,
        )
    else:  # pragma: no cover - exhaustive ProcessArtifact union
        raise TypeError(f"unsupported process artifact {type(process).__name__}")
    builder.add_json(
        validation_path,
        process.validation_point.to_mapping(),
        role="validation-momenta",
        process_id=process.process_id,
    )
    if isinstance(process, CompiledProcessArtifact):
        _copy_evaluator_payloads(
            evaluator_payloads,
            process.evaluator_root,
            prefix=prefix,
            process_id=process.process_id,
        )
        _copy_color_selector_evaluator_payloads(
            evaluator_payloads,
            process.color_selector_executions,
            prefix=prefix,
            process_id=process.process_id,
        )
        if process.helicity_sum_execution is not None:
            _copy_evaluator_payloads(
                evaluator_payloads,
                process.helicity_sum_execution.evaluator_root,
                prefix=f"{prefix}/{_HELICITY_SUM_PAYLOAD_ROOT}",
                process_id=process.process_id,
            )
            _copy_color_selector_evaluator_payloads(
                evaluator_payloads,
                process.helicity_sum_execution.color_selector_executions,
                prefix=f"{prefix}/{_HELICITY_SUM_PAYLOAD_ROOT}",
                process_id=process.process_id,
            )
        _copy_helicity_selector_evaluator_payloads(
            evaluator_payloads,
            process.helicity_selector_executions,
            prefix=prefix,
            process_id=process.process_id,
        )
    return (
        {
            "id": process.process_id,
            "expression": process.expression,
            "color_accuracy": process.color_accuracy,
            "external_pdgs": list(process.external_pdgs),
            "physics_path": physics_path,
            "required_runtime_capabilities": list(
                _process_runtime_capabilities(process)
            ),
            "aliases": [dict(alias) for alias in process.aliases],
        },
        {
            "process_id": process.process_id,
            "manifest_path": f"{process.process_id}/execution.json",
            "required_runtime_capabilities": list(
                _process_runtime_capabilities(process)
            ),
        },
        execution_record.sha256,
    )


def _write_recurrence_bootstraps(
    builder: ArtifactBuilder,
    *,
    processes: Sequence[RecurrenceProcessArtifact],
    process_records: Sequence[Mapping[str, object]],
    producer: Mapping[str, object],
    runtime: Mapping[str, object],
    evaluator_payload_container: Mapping[str, object] | None,
    target: Mapping[str, object],
) -> None:
    """Publish authenticated native recurrence load images.

    This v1 image is deliberately emitted only for a single-process artifact:
    multiple self-authenticating sidecars would create a cross-hash cycle, and
    mixed process sets can have artifact-wide capabilities not owned by this
    recurrence process. Such artifacts retain the generic loader.

    The image contains a process-ready recipe referencing the authoritative
    schedule/binding payloads, compact physics, and the staged payload
    inventory *before* the image is added. Finalization then includes the
    image's own SHA-256 record in the ordinary artifact ID. The private staging
    root is supplied only while Rust lowers the recipe and is never serialized.
    """

    if not processes:
        return
    if len(processes) != 1 or len(process_records) != 1:
        return
    if evaluator_payload_container is None:
        raise ValueError("recurrence bootstrap requires an evaluator payload container")
    records_by_id = {str(record["id"]): record for record in process_records}
    payloads = [record.as_dict() for record in builder.payload_records()]
    if builder.root is None:
        raise RuntimeError("recurrence bootstrap requires an active artifact builder")
    build = _recurrence_bootstrap_operation()
    for process in processes:
        process_id = process.process_id
        prefix = f"processes/{process_id}"
        execution_path = f"{prefix}/execution.json"
        physics_path = f"{prefix}/physics.json"
        context = {
            "schema_version": 1,
            "staged_root": str(builder.root),
            "producer": dict(producer),
            "runtime": dict(runtime),
            "process": dict(records_by_id[process_id]),
            "payloads": payloads,
            "evaluator_payload_container": dict(evaluator_payload_container),
        }
        image = build(
            json.dumps(
                context,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
            builder.staged_path(execution_path).read_bytes(),
            builder.staged_path(physics_path).read_bytes(),
        )
        if not isinstance(image, bytes) or not image:
            raise RuntimeError(
                "Rusticol returned an invalid recurrence bootstrap image"
            )
        builder.add_bytes(
            f"{prefix}/{_RECURRENCE_BOOTSTRAP_FILENAME}",
            image,
            role="evaluator-state",
            media_type="application/octet-stream",
            executable=False,
            target=target,
            process_id=process_id,
        )


@cache
def _recurrence_bootstrap_operation() -> Callable[[bytes, bytes, bytes], bytes]:
    try:
        rusticol = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(rusticol)
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "recurrence bootstrap generation requires the current pyamplicol "
            "native extension"
        ) from exc
    operation = getattr(rusticol, "_build_recurrence_bootstrap_v1", None)
    if not callable(operation):
        raise RuntimeError(
            "the current pyamplicol native extension does not expose "
            "_build_recurrence_bootstrap_v1; rebuild it before generating artifacts"
        )
    return operation


def _runtime_schema_mapping(
    schema: RuntimeExpressionSchema | Mapping[str, object],
) -> Mapping[str, object]:
    if isinstance(schema, RuntimeExpressionSchema):
        return schema.to_mapping()
    return schema


def _validate_staged_eager_runtime(
    process: EagerPlanV3ProcessArtifact,
    record: PayloadRecord,
) -> None:
    if process.eager_runtime_size_bytes <= 0:
        raise ValueError("Rust eager runtime payload must not be empty")
    payload_sha256 = _canonical_sha256(
        process.eager_runtime_sha256,
        "Rust eager runtime payload SHA-256",
    )
    if (
        record.size_bytes != process.eager_runtime_size_bytes
        or record.sha256 != payload_sha256
    ):
        raise ValueError(
            "Rust eager runtime payload changed after lowering and before publication"
        )


def _validate_staged_on_the_fly_runtime(
    process: OnTheFlyProcessArtifact,
    record: PayloadRecord,
) -> None:
    payload_sha256 = _canonical_sha256(
        process.runtime_sha256,
        "on-the-fly runtime payload SHA-256",
    )
    if process.runtime_size_bytes <= 0:
        raise ValueError("on-the-fly runtime payload must not be empty")
    if (
        record.size_bytes != process.runtime_size_bytes
        or record.sha256 != payload_sha256
    ):
        raise ValueError(
            "on-the-fly runtime payload changed before artifact publication"
        )
    with PacbinReader.open(process.runtime_path, verify_payloads=True) as reader:
        index = reader.index
        if (
            len(index.members) != 1
            or index.members[0].logical_path != _ON_THE_FLY_PROCESS_SEED_MEMBER_PATH
            or index.members[0].kind is not PacbinMemberKind.ON_THE_FLY_PROCESS_SEED
        ):
            raise ValueError(
                "on-the-fly runtime must contain exactly one canonical process seed"
            )
        if (
            len(index.members) != process.runtime_member_count
            or sum(member.length for member in index.members)
            != process.runtime_unpacked_size_bytes
            or index.index_sha256 != process.runtime_index_sha256
        ):
            raise ValueError(
                "on-the-fly runtime container metadata changed before publication"
            )


def _on_the_fly_execution_summary(
    process: OnTheFlyProcessArtifact,
    *,
    color_contraction_record: PayloadRecord | None = None,
) -> bytes:
    capabilities = list(_on_the_fly_process_runtime_capabilities(process))
    selector_policy = _mapping(process.selector_policy)
    if set(selector_policy) - {"color_basis"} != {
        "color_coverage",
        "reference_color_word",
        "trace_reflections_folded",
        "selector_census",
    }:
        raise ValueError("on-the-fly selector policy fields are invalid")
    color_basis = selector_policy.get("color_basis", "trace")
    if color_basis not in ("trace", "adjoint"):
        raise ValueError("on-the-fly selector color basis must be trace or adjoint")
    if color_basis == "adjoint" and process.color_accuracy not in {"nlc", "full"}:
        raise ValueError("adjoint on-the-fly color requires contracted NLC/full")
    expected_coverage = "complete" if process.color_accuracy == "lc" else "contracted"
    if selector_policy.get("color_coverage") != expected_coverage or not isinstance(
        selector_policy.get("trace_reflections_folded"), bool
    ):
        raise ValueError("on-the-fly selector policy is invalid")
    census = _mapping(selector_policy.get("selector_census"))
    if set(census) != {
        "physical_helicity_count",
        "physical_color_flow_count",
    }:
        raise ValueError("on-the-fly selector census fields are invalid")
    census_counts = {}
    for field in ("physical_helicity_count", "physical_color_flow_count"):
        count = _nonnegative_integer(
            census.get(field),
            f"on-the-fly selector census {field}",
            minimum=1,
        )
        if count > (1 << 64) - 1:
            raise ValueError(f"on-the-fly selector census {field} exceeds u64")
        census_counts[field] = count
    if process.color_accuracy in {"nlc", "full"} and (
        census_counts["physical_color_flow_count"] != 1
        or selector_policy.get("trace_reflections_folded") is not False
    ):
        raise ValueError(
            "contracted on-the-fly selector policy must expose one color "
            "result without trace-reflection folding"
        )
    runtime_metadata = _deep_plain(process.runtime_metadata)
    if not isinstance(runtime_metadata, dict):
        raise TypeError("on-the-fly runtime metadata must be a mapping")
    if color_contraction_record is None:
        if process.color_accuracy != "lc":
            raise ValueError("contracted on-the-fly execution has no color payload")
    else:
        if process.color_accuracy not in {"nlc", "full"}:
            raise ValueError("LC on-the-fly execution carries a color payload")
        summary = _deep_plain(process.color_contraction_summary)
        if not isinstance(summary, dict):
            raise TypeError("on-the-fly color-contraction summary must be a mapping")
        expected_summary_fields = {
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
        }
        if (
            isinstance(summary.get("factorization"), Mapping)
            and summary["factorization"].get("kind") == "symmetric-group-fourier"
        ):
            expected_summary_fields.add("fft_provenance")
        if set(summary) != expected_summary_fields:
            raise ValueError("on-the-fly color-contraction summary fields are invalid")
        factorization = summary.get("factorization")
        symmetric_group_fft = (
            isinstance(factorization, Mapping)
            and factorization.get("kind") == "symmetric-group-fourier"
        )
        factorization_rank = (
            factorization.get("rank") if isinstance(factorization, Mapping) else None
        )
        factorization_coset_count = (
            factorization.get("coset_count")
            if isinstance(factorization, Mapping)
            else None
        )
        factorization_is_canonical = (
            factorization is None
            if not symmetric_group_fft
            else isinstance(factorization, Mapping)
            and set(factorization) == {"kind", "rank", "coset_count"}
            and isinstance(factorization_rank, int)
            and not isinstance(factorization_rank, bool)
            and 2 <= factorization_rank <= 10
            and isinstance(factorization_coset_count, int)
            and not isinstance(factorization_coset_count, bool)
            and factorization_coset_count >= 1
        )
        _validate_symmetric_group_fft_provenance(
            summary,
            context="on-the-fly color-contraction",
        )
        if (
            summary.get("abi") != "pyamplicol-recurrence-color-contraction-v3"
            or summary.get("color_accuracy") != process.color_accuracy
            or summary.get("storage")
            != ("convolution-kernels" if symmetric_group_fft else "expanded")
            or summary.get("includes_color_factor") is not True
            or summary.get("component_count") != 1
            or not factorization_is_canonical
            or summary.get("active_sector_count") != summary.get("group_count")
            or summary.get("destination_count") != summary.get("group_count")
            or summary.get("logical_entry_count") != summary.get("entry_count")
            or summary.get("semantic_digest") != color_contraction_record.sha256
        ):
            raise ValueError("on-the-fly color-contraction summary is noncanonical")
        runtime_metadata["color_contraction"] = {
            **summary,
            "path": _ON_THE_FLY_COLOR_CONTRACTION_PATH,
            "size_bytes": color_contraction_record.size_bytes,
            "sha256": color_contraction_record.sha256,
        }
    payload = {
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": ON_THE_FLY_RUNTIME_KIND,
        "required_runtime_capabilities": capabilities,
        "process": process.expression,
        "key": process.process_id,
        "color_accuracy": process.color_accuracy,
        "external_pdg_order": list(process.external_pdgs),
        "kernel_pack": {
            "manifest_path": _EAGER_KERNEL_PACK_PATH,
            "payload_root": _EAGER_KERNEL_PAYLOAD_ROOT,
        },
        "runtime_options": {
            "point_tile_size": _nonnegative_integer(
                process.point_tile_size,
                "on-the-fly point tile size",
                minimum=1,
            ),
            "query_construction_threads": _nonnegative_integer(
                process.query_construction_threads,
                "on-the-fly query construction threads",
                minimum=1,
            ),
        },
        "selector_policy": _deep_plain(selector_policy),
        "runtime_metadata": runtime_metadata,
        "runtime_container": {
            "kind": ON_THE_FLY_RUNTIME_CONTAINER_KIND,
            "schema_version": ON_THE_FLY_RUNTIME_CONTAINER_SCHEMA_VERSION,
            "storage_abi": ON_THE_FLY_RUNTIME_STORAGE_ABI,
            "path": _ON_THE_FLY_RUNTIME_CONTAINER_PATH,
            "seed_member_path": _ON_THE_FLY_PROCESS_SEED_MEMBER_PATH,
        },
    }
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"on-the-fly execution summary is not canonical JSON: {exc}"
        ) from exc


def _recurrence_helicity_selector_companion_v2_manifest(
    process: RecurrenceProcessArtifact,
    companion: RecurrenceHelicitySelectorPlanArtifact,
    *,
    binding: Mapping[str, object],
) -> dict[str, object]:
    """Describe one pre-lowered, runtime-helicity-dispatched selector plan."""

    process_digest = _canonical_sha256(
        process.process_digest,
        "recurrence helicity-selector process digest",
    )
    if binding.get("process_digest") != process_digest:
        raise ValueError(
            "recurrence helicity-selector binding disagrees with its process digest"
        )
    if binding.get("process_semantic_digest") != companion.builder_input_sha256:
        raise ValueError(
            "recurrence helicity-selector binding disagrees with its builder input"
        )
    if binding.get("schedule_digest") != companion.recurrence_schedule_digest:
        raise ValueError(
            "recurrence helicity-selector binding disagrees with its schedule"
        )

    companion_pack = companion.recurrence_process_executor_pack
    primary_pack = process.recurrence_process_executor_pack
    for name in (
        "compiled_model_digest",
        "recurrence_template_catalog_digest",
        "prepared_kernel_pack_digest",
        "direct_template_catalog_digest",
    ):
        if getattr(companion_pack, name) != getattr(primary_pack, name):
            raise ValueError(
                "recurrence helicity-selector executor pack disagrees with the "
                f"primary {name.replace('_', ' ')}"
            )
    if (
        companion_pack.prepared_kernel_pack_digest
        != process.prepared_kernel_pack_digest
    ):
        raise ValueError(
            "recurrence helicity-selector prepared-kernel identity disagrees "
            "with its process"
        )
    if (
        companion_pack.direct_template_catalog_digest
        != process.direct_template_catalog_digest
    ):
        raise ValueError(
            "recurrence helicity-selector direct-template identity disagrees "
            "with its process"
        )

    primary_color = _deep_plain(process.color_contraction_summary)
    if not isinstance(primary_color, dict):
        raise TypeError(
            "persisted recurrence helicity-selector plan requires the primary "
            "color-contraction summary"
        )
    if (
        process.color_accuracy not in {"nlc", "full"}
        or primary_color.get("abi") != "pyamplicol-recurrence-color-contraction-v3"
        or primary_color.get("color_accuracy") != process.color_accuracy
    ):
        raise ValueError(
            "recurrence helicity-selector primary local-color view requires a "
            "compatible complete-color reducer"
        )
    _validate_symmetric_group_fft_provenance(
        primary_color,
        context="recurrence primary color-contraction",
    )
    storage = primary_color.get("storage")
    factorization_raw = primary_color.get("factorization")
    factorization_kind = (
        factorization_raw.get("kind")
        if isinstance(factorization_raw, Mapping)
        else None
    )
    component_count = _nonnegative_integer(
        primary_color.get("component_count"),
        "recurrence primary color-contraction component count",
        minimum=1,
    )
    supported_primary_view = (
        (storage == "expanded" and factorization_raw is None and component_count == 1)
        or (
            storage == "repeated"
            and factorization_kind
            in {None, "klein-four-walsh", "elementary-abelian-walsh"}
        )
        or (
            storage == "convolution-kernels"
            and factorization_kind == "symmetric-group-fourier"
        )
    )
    if not supported_primary_view:
        raise ValueError(
            "recurrence helicity-selector primary local-color view requires a "
            "direct, Walsh, or symmetric-group Fourier reducer"
        )
    group_count = _nonnegative_integer(
        primary_color.get("group_count"),
        "recurrence primary color-contraction group count",
        minimum=1,
    )
    if group_count % component_count:
        raise ValueError(
            "recurrence primary color-contraction groups do not form complete "
            "local component blocks"
        )
    local_group_count = group_count // component_count
    physical_destination_count = _nonnegative_integer(
        companion.physical_destination_count,
        "recurrence helicity-selector physical destination count",
        minimum=1,
    )
    if physical_destination_count != local_group_count:
        raise ValueError(
            "recurrence helicity-selector physical destinations do not match "
            "the primary local color-group domain"
        )
    runtime_schedule = {
        "kind": RECURRENCE_RUNTIME_CONTAINER_KIND,
        "schema_version": RECURRENCE_RUNTIME_CONTAINER_SCHEMA_VERSION,
        "storage_abi": RECURRENCE_RUNTIME_STORAGE_ABI,
        "path": _recurrence_helicity_selector_schedule_path(companion),
        "plan_member_path": _RECURRENCE_DIRECT_SCHEDULE_MEMBER_PATH,
        "size_bytes": _nonnegative_integer(
            companion.recurrence_schedule_size_bytes,
            "recurrence helicity-selector schedule size",
            minimum=1,
        ),
        "sha256": _canonical_sha256(
            companion.recurrence_schedule_sha256,
            "recurrence helicity-selector schedule SHA-256",
        ),
        "member_count": _nonnegative_integer(
            companion.recurrence_schedule_member_count,
            "recurrence helicity-selector schedule member count",
            minimum=1,
        ),
        "unpacked_size_bytes": _nonnegative_integer(
            companion.recurrence_schedule_unpacked_size_bytes,
            "recurrence helicity-selector schedule unpacked size",
        ),
        "index_sha256": _canonical_sha256(
            companion.recurrence_schedule_index_sha256,
            "recurrence helicity-selector schedule index SHA-256",
        ),
    }
    inspection_summary = _deep_plain(companion.inspection_summary)
    if not isinstance(inspection_summary, dict):
        raise TypeError(
            "recurrence helicity-selector inspection summary must be a mapping"
        )
    dispatch_layout_digest = _canonical_sha256(
        companion.helicity_dispatch_base_runtime_layout_digest,
        "recurrence helicity-dispatch base runtime-layout digest",
    )
    if inspection_summary.get("runtime_layout_digest") != dispatch_layout_digest:
        raise ValueError(
            "recurrence helicity dispatch disagrees with its base runtime layout"
        )
    native_schedule_digest = _canonical_sha256(
        binding.get("native_schedule_semantic_digest"),
        "recurrence helicity-selector binding native schedule digest",
    )
    if inspection_summary.get("schedule_digest") != native_schedule_digest:
        raise ValueError(
            "recurrence helicity-selector inspection summary disagrees with its "
            "native Direct-plan identity"
        )
    helicity_dispatch = {
        "abi": _RECURRENCE_HELICITY_DISPATCH_ABI,
        "path": _recurrence_helicity_dispatch_path(companion),
        "size_bytes": _nonnegative_integer(
            companion.helicity_dispatch_size_bytes,
            "recurrence helicity-dispatch size",
            minimum=1,
        ),
        "sha256": _canonical_sha256(
            companion.helicity_dispatch_sha256,
            "recurrence helicity-dispatch SHA-256",
        ),
        "base_runtime_layout_digest": dispatch_layout_digest,
        "resolved_helicity_count": _nonnegative_integer(
            companion.helicity_dispatch_resolved_helicity_count,
            "recurrence helicity-dispatch resolved-helicity count",
            minimum=1,
        ),
    }
    capabilities = list(_recurrence_process_runtime_capabilities(process))
    plan = {
        "kind": RECURRENCE_RUNTIME_KIND,
        "builder_input_abi": RECURRENCE_BUILDER_INPUT_ABI,
        "recurrence_plan_abi": RECURRENCE_PLAN_ABI,
        "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_ABI,
        "direct_template_abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
        "direct_backend_abi": RECURRENCE_DIRECT_BACKEND_ABI,
        "builder_input_sha256": _canonical_sha256(
            companion.builder_input_sha256,
            "recurrence helicity-selector builder input SHA-256",
        ),
        "prepared_kernel_pack_digest": _canonical_sha256(
            process.prepared_kernel_pack_digest,
            "recurrence helicity-selector prepared-kernel pack SHA-256",
        ),
        "direct_template_catalog_digest": _canonical_sha256(
            process.direct_template_catalog_digest,
            "recurrence helicity-selector direct-template catalog SHA-256",
        ),
        "required_runtime_capabilities": capabilities,
        "runtime_schedule": runtime_schedule,
        "helicity_dispatch": helicity_dispatch,
        "process_binding": _deep_plain(binding),
        "inspection_summary": inspection_summary,
    }
    return {
        "schema_version": 2,
        "kind": _RECURRENCE_HELICITY_SELECTOR_COMPANION_V2_KIND,
        "process_digest": process_digest,
        "plan": plan,
        "color_contraction": {
            "source": "primary",
            "view": "primary-local-color-view-v1",
        },
    }


def _eager_execution_summary(
    process: EagerPlanV3ProcessArtifact,
) -> bytes:
    try:
        content = (
            json.dumps(
                _eager_plan_v3_execution_manifest(process),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Rust eager execution summary is not canonical JSON: {exc}"
        ) from exc
    return content


def _eager_plan_v3_execution_manifest(
    process: EagerPlanV3ProcessArtifact,
) -> dict[str, object]:
    lowering_input_sha256 = _canonical_sha256(
        process.lowering_input_sha256,
        "eager lowering input SHA-256",
    )
    payload_sha256 = _canonical_sha256(
        process.eager_runtime_sha256,
        "Rust eager runtime payload SHA-256",
    )
    index_sha256 = _canonical_sha256(
        process.eager_runtime_index_sha256,
        "Rust eager runtime index SHA-256",
    )
    member_count = _nonnegative_integer(
        process.eager_runtime_member_count,
        "Rust eager runtime member count",
        minimum=1,
    )
    unpacked_size = _nonnegative_integer(
        process.eager_runtime_unpacked_size_bytes,
        "Rust eager runtime unpacked size",
    )
    payload_size = _nonnegative_integer(
        process.eager_runtime_size_bytes,
        "Rust eager runtime payload size",
        minimum=1,
    )
    capabilities = list(_EAGER_PLAN_V3_RUNTIME_CAPABILITIES)
    plan = {
        "kind": EAGER_RUNTIME_KIND,
        "eager_plan_abi": EAGER_PLAN_V3_ABI,
        "lowering_input_abi": EAGER_LOWERING_INPUT_ABI,
        "lowering_input_sha256": lowering_input_sha256,
        "runtime_layout_abi": EAGER_RUNTIME_LAYOUT_ABI,
        "required_runtime_capabilities": capabilities,
        "runtime_container": {
            "kind": EAGER_RUNTIME_CONTAINER_KIND,
            "schema_version": EAGER_RUNTIME_CONTAINER_SCHEMA_VERSION,
            "storage_abi": EAGER_RUNTIME_STORAGE_ABI,
            "path": _EAGER_RUNTIME_CONTAINER_PATH,
            "size_bytes": payload_size,
            "sha256": payload_sha256,
            "member_count": member_count,
            "unpacked_size_bytes": unpacked_size,
            "index_sha256": index_sha256,
        },
        "inspection_summary": _deep_plain(process.inspection_summary),
        "materialization_census": _eager_materialization_census(process),
    }
    return {
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": EAGER_RUNTIME_KIND,
        "required_runtime_capabilities": capabilities,
        "process": process.expression,
        "key": process.process_id,
        "color_accuracy": process.color_accuracy,
        "external_pdg_order": list(process.external_pdgs),
        "eager_plan_abi": EAGER_PLAN_V3_ABI,
        "kernel_pack": {
            "manifest_path": _EAGER_KERNEL_PACK_PATH,
            "payload_root": _EAGER_KERNEL_PAYLOAD_ROOT,
        },
        "runtime_options": {
            "point_tile_size": process.point_tile_size,
            "workspace_mib": process.workspace_mib,
        },
        "plan": plan,
        "dag_summary": _dag_summary(process.dag_summary),
    }


def _bounded_recurrence_execution_summary(
    process: RecurrenceProcessArtifact,
    *,
    schedule_path: str,
    binding: Mapping[str, object],
    color_contraction_record: PayloadRecord | None,
    companion_binding: Mapping[str, object] | None = None,
) -> bytes:
    try:
        content = (
            json.dumps(
                _recurrence_execution_manifest(
                    process,
                    schedule_path=schedule_path,
                    binding=binding,
                    color_contraction_record=color_contraction_record,
                    companion_binding=companion_binding,
                ),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Rust recurrence execution summary is not canonical JSON: {exc}"
        ) from exc
    return content


def _recurrence_execution_manifest(
    process: RecurrenceProcessArtifact,
    *,
    schedule_path: str,
    binding: Mapping[str, object],
    color_contraction_record: PayloadRecord | None,
    companion_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    primary_process_digest = _canonical_sha256(
        process.process_digest,
        "recurrence primary process digest",
    )
    if binding.get("process_digest") != primary_process_digest:
        raise ValueError(
            "recurrence primary process binding disagrees with its process digest"
        )
    capabilities = list(_recurrence_process_runtime_capabilities(process))
    runtime_schedule = {
        "kind": RECURRENCE_RUNTIME_CONTAINER_KIND,
        "schema_version": RECURRENCE_RUNTIME_CONTAINER_SCHEMA_VERSION,
        "storage_abi": RECURRENCE_RUNTIME_STORAGE_ABI,
        "path": schedule_path,
        "plan_member_path": _RECURRENCE_DIRECT_SCHEDULE_MEMBER_PATH,
        "size_bytes": _nonnegative_integer(
            process.recurrence_schedule_size_bytes,
            "Rust recurrence runtime payload size",
            minimum=1,
        ),
        "sha256": _canonical_sha256(
            process.recurrence_schedule_sha256,
            "Rust recurrence runtime payload SHA-256",
        ),
        "member_count": _nonnegative_integer(
            process.recurrence_schedule_member_count,
            "Rust recurrence runtime member count",
            minimum=1,
        ),
        "unpacked_size_bytes": _nonnegative_integer(
            process.recurrence_schedule_unpacked_size_bytes,
            "Rust recurrence runtime unpacked size",
        ),
        "index_sha256": _canonical_sha256(
            process.recurrence_schedule_index_sha256,
            "Rust recurrence runtime index SHA-256",
        ),
    }
    plan = {
        "kind": RECURRENCE_RUNTIME_KIND,
        "builder_input_abi": RECURRENCE_BUILDER_INPUT_ABI,
        "recurrence_plan_abi": RECURRENCE_PLAN_ABI,
        "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_ABI,
        "direct_template_abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
        "direct_backend_abi": RECURRENCE_DIRECT_BACKEND_ABI,
        "builder_input_sha256": _canonical_sha256(
            process.builder_input_sha256,
            "recurrence builder input SHA-256",
        ),
        "prepared_kernel_pack_digest": _canonical_sha256(
            process.prepared_kernel_pack_digest,
            "recurrence prepared-kernel pack SHA-256",
        ),
        "direct_template_catalog_digest": _canonical_sha256(
            process.direct_template_catalog_digest,
            "recurrence direct-template catalog SHA-256",
        ),
        "required_runtime_capabilities": capabilities,
        "runtime_schedule": runtime_schedule,
        "process_binding": _deep_plain(binding),
        "inspection_summary": _deep_plain(process.inspection_summary),
    }
    runtime_metadata = _deep_plain(process.runtime_metadata)
    if not isinstance(runtime_metadata, dict):
        raise TypeError("recurrence runtime metadata must be a mapping")
    if color_contraction_record is None:
        runtime_metadata["color_contraction"] = None
    else:
        summary = _deep_plain(process.color_contraction_summary)
        if not isinstance(summary, dict):
            raise TypeError("recurrence color-contraction summary must be a mapping")
        _validate_symmetric_group_fft_provenance(
            summary,
            context="recurrence color-contraction",
        )
        runtime_metadata["color_contraction"] = {
            **summary,
            "path": _RECURRENCE_COLOR_CONTRACTION_PATH,
            "size_bytes": color_contraction_record.size_bytes,
            "sha256": color_contraction_record.sha256,
        }
    companion = process.helicity_selector_companion
    if companion is None:
        if companion_binding is not None:
            raise ValueError(
                "recurrence without a helicity-selector companion carries "
                "companion metadata"
            )
        companion_manifest = None
        process_digest = None
    elif isinstance(companion, RecurrenceHelicitySelectorPlanArtifact):
        if color_contraction_record is None or companion_binding is None:
            raise ValueError(
                "persisted recurrence helicity-selector companion is missing its "
                "authenticated plan or primary color metadata"
            )
        process_digest = primary_process_digest
        companion_manifest = _recurrence_helicity_selector_companion_v2_manifest(
            process,
            companion,
            binding=companion_binding,
        )
    else:  # pragma: no cover - dataclass typing guards ordinary construction
        raise TypeError(
            "recurrence helicity-selector companion must be a persisted Direct plan"
        )
    return {
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": RECURRENCE_RUNTIME_KIND,
        "required_runtime_capabilities": capabilities,
        "process": process.expression,
        "key": process.process_id,
        "color_accuracy": process.color_accuracy,
        "external_pdg_order": list(process.external_pdgs),
        "process_digest": process_digest,
        "builder_input_abi": RECURRENCE_BUILDER_INPUT_ABI,
        "recurrence_plan_abi": RECURRENCE_PLAN_ABI,
        "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_ABI,
        "direct_template_abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
        "direct_backend_abi": RECURRENCE_DIRECT_BACKEND_ABI,
        "compiled_model_digest": (
            process.recurrence_process_executor_pack.compiled_model_digest
        ),
        "recurrence_template_catalog_digest": (
            process.recurrence_process_executor_pack.recurrence_template_catalog_digest
        ),
        "prepared_kernel_pack_digest": _canonical_sha256(
            process.prepared_kernel_pack_digest,
            "recurrence prepared-kernel pack SHA-256",
        ),
        "direct_template_catalog_digest": _canonical_sha256(
            process.direct_template_catalog_digest,
            "recurrence direct-template catalog SHA-256",
        ),
        "kernel_pack": {
            "manifest_path": _EAGER_KERNEL_PACK_PATH,
            "payload_root": _EAGER_KERNEL_PAYLOAD_ROOT,
        },
        "runtime_options": {
            "point_tile_size": process.point_tile_size,
            "workspace_mib": process.workspace_mib,
        },
        "runtime_metadata": runtime_metadata,
        "helicity_selector_companion": companion_manifest,
        "plan": plan,
        "recurrence_summary": _deep_plain(process.recurrence_summary),
    }


def _canonical_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{context} must be a lowercase hexadecimal digest")
    return value


def _nonnegative_integer(
    value: object,
    context: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{context} must be an integer greater than or equal to {minimum}"
        )
    return value


def _execution_manifest(
    process: ProcessArtifact,
    compiler_schema: Mapping[str, object],
    *,
    color_contraction_payload_path: str | None = None,
) -> dict[str, object]:
    if isinstance(process, OnTheFlyProcessArtifact):
        raise TypeError("on-the-fly execution manifests use the compact seed writer")
    if isinstance(process, RecurrenceProcessArtifact):
        raise TypeError(
            "recurrence execution manifests require an interned root schedule "
            "and process binding"
        )
    if isinstance(process, EagerPlanV3ProcessArtifact):
        return _eager_plan_v3_execution_manifest(process)
    primary = _compiled_execution_lane_manifest(
        runtime_schema=compiler_schema,
        stage_manifest=process.stage_manifest,
        model_parameter_evaluator=process.model_parameter_evaluator,
        dag_summary=process.dag_summary,
        payload_prefix=None,
    )
    required_runtime_capabilities = set(_required_runtime_capabilities(primary))
    if color_contraction_payload_path is not None:
        required_runtime_capabilities.add(SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY)
    color_selector_executions = _compiled_color_selector_execution_manifests(
        process=process,
        executions=process.color_selector_executions,
        parent_payload_prefix=None,
    )
    if color_selector_executions:
        required_runtime_capabilities.add(COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY)
        required_runtime_capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        for record in color_selector_executions:
            required_runtime_capabilities.update(
                _required_runtime_capabilities(_mapping(record["execution"]))
            )
    helicity_sum_execution = process.helicity_sum_execution
    auxiliary: dict[str, object] | None = None
    if helicity_sum_execution is not None:
        auxiliary = _compiled_nested_execution_manifest(
            process=process,
            execution=helicity_sum_execution,
            payload_prefix=_HELICITY_SUM_PAYLOAD_ROOT,
        )
        required_runtime_capabilities.update(_required_runtime_capabilities(auxiliary))
        required_runtime_capabilities.add(COMPILED_HELICITY_DUAL_LANE_CAPABILITY)
    helicity_selector_executions = _compiled_helicity_selector_execution_manifests(
        process=process,
        executions=process.helicity_selector_executions,
        parent_payload_prefix=None,
    )
    if helicity_selector_executions:
        for record in helicity_selector_executions:
            required_runtime_capabilities.update(
                _required_runtime_capabilities(_mapping(record["execution"]))
            )
        required_runtime_capabilities.add(COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY)
        required_runtime_capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
    if _uses_primary_helicity_recurrence(process):
        required_runtime_capabilities.add(
            COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY
        )
    return {
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": "pyamplicol-runtime-execution",
        "required_runtime_capabilities": sorted(required_runtime_capabilities),
        "process": process.expression,
        "key": process.process_id,
        "color_accuracy": process.color_accuracy,
        "external_pdg_order": list(process.external_pdgs),
        "compiled": primary["compiled"],
        "dag_summary": primary["dag_summary"],
        "materialization_census": primary["materialization_census"],
        "runtime_schema": primary["runtime_schema"],
        **(
            {}
            if color_contraction_payload_path is None
            else {
                "color_contraction_payload": {
                    "path": color_contraction_payload_path,
                }
            }
        ),
        **({} if auxiliary is None else {"helicity_sum_execution": auxiliary}),
        **(
            {}
            if not helicity_selector_executions
            else {"helicity_selector_executions": helicity_selector_executions}
        ),
        **(
            {}
            if not color_selector_executions
            else {"color_selector_executions": color_selector_executions}
        ),
    }


def _compiled_helicity_selector_execution_manifests(
    *,
    process: CompiledProcessArtifact,
    executions: Sequence[CompiledHelicitySelectorExecutionArtifact],
    parent_payload_prefix: str | None,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for lane_index, record in enumerate(
        _ordered_helicity_selector_executions(executions)
    ):
        execution = record.execution
        if execution.color_selector_executions:
            raise ValueError(
                "compiled helicity-selector closure execution cannot contain "
                "nested execution lanes"
            )
        lane_prefix = f"{_HELICITY_SELECTOR_UNION_PAYLOAD_ROOT}/class-{lane_index}"
        payload_prefix = (
            lane_prefix
            if parent_payload_prefix is None
            else f"{parent_payload_prefix.rstrip('/')}/{lane_prefix}"
        )
        manifest = _compiled_nested_execution_manifest(
            process=process,
            execution=execution,
            payload_prefix=payload_prefix,
        )
        if (
            record.schedule_mode == "parent-closure"
            and COMPILED_RUNTIME_SELECTORS_CAPABILITY
            in _required_runtime_capabilities(manifest)
        ):
            raise ValueError(
                "compiled helicity-selector closure stage evaluators cannot require "
                "runtime selectors"
            )
        result.append(
            {
                "selector_domain_ids": list(record.selector_domain_ids),
                "schedule_mode": record.schedule_mode,
                "execution": manifest,
            }
        )
    return result


def _compiled_nested_execution_manifest(
    *,
    process: CompiledProcessArtifact,
    execution: CompiledExecutionArtifact,
    payload_prefix: str,
) -> dict[str, object]:
    runtime_schema = _runtime_schema_mapping(execution.runtime_schema)
    lane = _compiled_execution_lane_manifest(
        runtime_schema=runtime_schema,
        stage_manifest=execution.stage_manifest,
        model_parameter_evaluator=execution.model_parameter_evaluator,
        dag_summary=execution.dag_summary,
        payload_prefix=payload_prefix,
    )
    color_selector_executions = _compiled_color_selector_execution_manifests(
        process=process,
        executions=execution.color_selector_executions,
        parent_payload_prefix=payload_prefix,
    )
    required_runtime_capabilities = set(_required_runtime_capabilities(lane))
    if _runtime_schema_uses_primary_helicity_recurrence(
        runtime_schema,
        has_helicity_sum_execution=False,
    ):
        required_runtime_capabilities.add(
            COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY
        )
    helicity_selector_executions = _compiled_helicity_selector_execution_manifests(
        process=process,
        executions=execution.helicity_selector_executions,
        parent_payload_prefix=payload_prefix,
    )
    if helicity_selector_executions:
        required_runtime_capabilities.add(COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY)
        required_runtime_capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        for record in helicity_selector_executions:
            required_runtime_capabilities.update(
                _required_runtime_capabilities(_mapping(record["execution"]))
            )
    if color_selector_executions:
        required_runtime_capabilities.add(COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY)
        required_runtime_capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        for record in color_selector_executions:
            required_runtime_capabilities.update(
                _required_runtime_capabilities(_mapping(record["execution"]))
            )
    return {
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": "pyamplicol-runtime-execution",
        "required_runtime_capabilities": sorted(required_runtime_capabilities),
        "process": process.expression,
        "key": process.process_id,
        "color_accuracy": process.color_accuracy,
        "external_pdg_order": list(process.external_pdgs),
        "compiled": lane["compiled"],
        "dag_summary": lane["dag_summary"],
        "materialization_census": lane["materialization_census"],
        "runtime_schema": lane["runtime_schema"],
        "physics_reduction": _plain_mapping(
            _mapping(_mapping(runtime_schema["physics"])["reduction"])
        ),
        **(
            {}
            if not color_selector_executions
            else {"color_selector_executions": color_selector_executions}
        ),
        **(
            {}
            if not helicity_selector_executions
            else {"helicity_selector_executions": (helicity_selector_executions)}
        ),
    }


def _compiled_color_selector_execution_manifests(
    *,
    process: CompiledProcessArtifact,
    executions: Sequence[CompiledColorSelectorExecutionArtifact],
    parent_payload_prefix: str | None,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for record in _ordered_color_selector_executions(executions):
        lane_prefix = _color_selector_payload_prefix(record.materialized_sector_id)
        payload_prefix = (
            lane_prefix
            if parent_payload_prefix is None
            else f"{parent_payload_prefix.rstrip('/')}/{lane_prefix}"
        )
        result.append(
            {
                "materialized_sector_id": record.materialized_sector_id,
                "execution": _compiled_nested_execution_manifest(
                    process=process,
                    execution=record.execution,
                    payload_prefix=payload_prefix,
                ),
            }
        )
    return result


def _compiled_execution_lane_manifest(
    *,
    runtime_schema: Mapping[str, object],
    stage_manifest: Mapping[str, object],
    model_parameter_evaluator: Mapping[str, object] | None,
    dag_summary: Mapping[str, object],
    payload_prefix: str | None,
) -> dict[str, object]:
    serialized_model_parameters = (
        None
        if model_parameter_evaluator is None
        else _model_parameter_evaluator(model_parameter_evaluator)
    )
    stage_evaluators = _stage_evaluator_set(stage_manifest)
    if payload_prefix is not None:
        stage_evaluators = _prefix_evaluator_payload_paths(
            stage_evaluators,
            payload_prefix,
        )
        if serialized_model_parameters is not None:
            serialized_model_parameters = _prefix_evaluator_payload_paths(
                serialized_model_parameters,
                payload_prefix,
            )
    required_runtime_capabilities = set(
        _required_runtime_capabilities(stage_evaluators)
    )
    required_runtime_capabilities.update(
        _runtime_schema_walsh_color_contraction_capabilities(runtime_schema)
    )
    if serialized_model_parameters is not None:
        required_runtime_capabilities.update(
            _required_runtime_capabilities(serialized_model_parameters)
        )
    compiled_manifest: dict[str, object] = {
        "kind": "generic-dag-stage-blueprint",
        "runtime_available": True,
        "runtime_unavailable_message": None,
        "model_parameter_evaluator": serialized_model_parameters,
        "stage_evaluators": stage_evaluators,
    }
    if COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY in required_runtime_capabilities:
        layout = _mapping(runtime_schema["parameter_layout"])
        compiled_manifest["momentum_slot_ids"] = _compiled_momentum_slot_ids(
            stage_evaluators,
            value_component_count=int(layout["value_component_count"]),
            momentum_component_count=int(layout["momentum_parameter_count"]),
        )
    topology_replay = runtime_schema.get("lc_topology_replay")
    if topology_replay is not None:
        compiled_manifest["lc_topology_replay"] = _plain_mapping(
            _mapping(topology_replay)
        )
    color_topology_replay = runtime_schema.get("color_topology_replay")
    if color_topology_replay is not None:
        compiled_manifest["color_topology_replay"] = _plain_mapping(
            _mapping(color_topology_replay)
        )
    helicity_recurrence = runtime_schema.get("helicity_recurrence")
    if helicity_recurrence is not None:
        compiled_manifest["helicity_recurrence"] = _plain_mapping(
            _mapping(helicity_recurrence)
        )
    serialized_dag_summary = _dag_summary(
        dag_summary,
        require_interaction_evaluation_count=True,
    )
    return {
        "kind": "pyamplicol-runtime-compiled-execution",
        "required_runtime_capabilities": sorted(required_runtime_capabilities),
        "compiled": compiled_manifest,
        "dag_summary": serialized_dag_summary,
        "materialization_census": _fully_resident_materialization_census(
            serialized_dag_summary,
            basis="immutable-fully-resident-compiled-dag",
        ),
        "runtime_schema": _execution_plan(runtime_schema),
    }


def _compiled_momentum_slot_ids(
    stage_evaluators: Mapping[str, object],
    *,
    value_component_count: int,
    momentum_component_count: int,
) -> list[int]:
    """Retain the union needed by every final compiled leaf in this lane.

    Chunk input maps are final here. Keep canonical momentum-slot numbering
    and the complete lane's coverage, independently of runtime selectors.
    """
    live: set[int] = set()
    stages = (
        *_sequence(stage_evaluators["stages"]),
        stage_evaluators["amplitude_stage"],
    )
    for raw_stage in stages:
        direct = _mapping(_mapping(raw_stage)["compiled_plane_arena"])
        bindings = _sequence(direct["input_bindings"])
        for raw_leaf in _sequence(direct["leaves"]):
            for raw_index in _sequence(_mapping(raw_leaf)["input_indices"]):
                binding = _mapping(bindings[int(raw_index)])
                if binding["kind"] != "momentum":
                    continue
                component = int(binding["global_component"]) - value_component_count
                if not 0 <= component < momentum_component_count:
                    raise ValueError(
                        "compiled momentum input is outside momentum slots"
                    )
                live.add(component // 4)
    return sorted(live)


def _prefix_evaluator_payload_paths(
    record: Mapping[str, object],
    prefix: str,
) -> dict[str, object]:
    path_fields = {
        "application_path",
        "descriptor_path",
        "evaluator_state_path",
        "library_path",
        "source_path",
    }

    def visit(value: object, *, field: str | None = None) -> object:
        if field in path_fields and value is not None:
            if not isinstance(value, str):
                raise TypeError(f"evaluator payload path {field!r} must be a string")
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"evaluator payload path {value!r} is not artifact-relative"
                )
            return f"{prefix.rstrip('/')}/{path.as_posix()}"
        if isinstance(value, Mapping):
            return {
                str(key): visit(item, field=str(key)) for key, item in value.items()
            }
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, tuple):
            return [visit(item) for item in value]
        return value

    result = visit(record)
    if not isinstance(result, dict):  # pragma: no cover - internal invariant
        raise TypeError("compiled evaluator manifest must be an object")
    return result


def _execution_plan(schema: Mapping[str, object]) -> dict[str, object]:
    source_fill = _mapping(schema["source_fill"])
    source_records = tuple(_mapping(item) for item in _sequence(source_fill["sources"]))
    source_count = (
        int(source_records[-1]["source_parameter_stop"]) if source_records else 0
    )
    parameter_layout = _mapping(schema["parameter_layout"])
    value_count = int(parameter_layout["value_component_count"])
    momentum_count = int(parameter_layout["momentum_parameter_count"])
    model_parameter_count = int(parameter_layout.get("model_parameter_count", 0))
    parameters = tuple(
        _execution_model_parameter(_mapping(item))
        for item in _sequence(schema.get("model_parameters", ()))
    )
    mass_parameters = {
        int(item["pdg"]): str(item["name"])
        for item in parameters
        if item["kind"] == "particle_mass" and item.get("pdg") is not None
    }
    model = _mapping(schema.get("model", {}))
    particles = tuple(_mapping(item) for item in _sequence(model.get("particles", ())))
    normalization = _mapping(schema.get("normalization", {}))
    return {
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": "pyamplicol-runtime-execution-plan",
        "process_key": str(schema["process_key"]),
        "process": str(schema["process"]),
        "external_particles": [
            _select(
                _mapping(item),
                "label",
                "index",
                "pdg",
                "outgoing_pdg",
                "role",
                "momentum_slot",
            )
            for item in _sequence(schema["external_particles"])
        ],
        "model": {
            "particles": [
                {
                    "pdg": int(particle["pdg"]),
                    "mass": float(particle.get("mass", 0.0)),
                    "mass_parameter": (
                        str(particle["mass_parameter"])
                        if particle.get("mass_parameter") is not None
                        else mass_parameters.get(int(particle["pdg"]))
                    ),
                }
                for particle in particles
            ]
        },
        "model_parameters": list(parameters),
        "normalization": {
            "color_factor": float(normalization.get("color_factor", 1.0)),
            "global_coupling_factor": float(
                normalization.get("global_coupling_factor", 1.0)
            ),
            "average_factor": float(normalization.get("average_factor", 1.0)),
            "identical_factor": float(normalization.get("identical_factor", 1.0)),
            "qcd_coupling_power": int(normalization.get("qcd_coupling_power", 0)),
            "electroweak_coupling_power": int(
                normalization.get("electroweak_coupling_power", 0)
            ),
        },
        "parameter_layout": {
            "source_component_parameter_count": source_count,
            "momentum_parameter_count": momentum_count,
            "model_parameter_count": model_parameter_count,
            "parameter_count_if_flattened": (
                source_count + momentum_count + model_parameter_count
            ),
            "value_component_count": value_count,
            "source_components_complex": True,
            "momentum_components_real": True,
            "real_valued_inputs": list(
                range(
                    source_count,
                    source_count + momentum_count + model_parameter_count,
                )
            ),
        },
        "current_storage": _current_storage(_mapping(schema["current_storage"])),
        "value_storage": _value_storage(_mapping(schema["value_storage"])),
        "source_fill": {
            "source_count": int(source_fill["source_count"]),
            "sources": [_source_record(item) for item in source_records],
        },
        "momentum_slots": [
            _select(
                _mapping(item),
                "momentum_slot_id",
                "momentum_mask",
                "external_labels",
                "component_start",
                "component_stop",
                "real_valued",
            )
            for item in _sequence(schema["momentum_slots"])
        ],
        "stages": [
            _runtime_stage(_mapping(item)) for item in _sequence(schema["stages"])
        ],
        "amplitude_stage": _amplitude_stage(_mapping(schema["amplitude_stage"])),
        **(
            {}
            if schema.get("helicity_recurrence") is None
            else {
                "helicity_recurrence": _plain_mapping(
                    _mapping(schema["helicity_recurrence"])
                )
            }
        ),
    }


def _execution_model_parameter(record: Mapping[str, object]) -> dict[str, object]:
    result = _select(record, "name", "kind", "parameter_index", "default")
    for name in ("pdg", "runtime_name", "complex_component"):
        if record.get(name) is not None:
            result[name] = record[name]
    return result


def _current_storage(storage: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "current_id",
        "component_start",
        "component_stop",
        "dimension",
        "is_source",
        "particle_id",
        "external_mask",
        "external_labels",
        "helicity_ancestry",
        "chirality",
        "spin_state",
        "flavour_flow",
        "color_state",
        "momentum_mask",
        "auxiliary_kind",
    )
    return {
        "component_count": int(storage["component_count"]),
        "number_type": str(storage["number_type"]),
        "metadata_compacted": True,
        "current_slots": [
            _select(_mapping(item), *fields)
            for item in _sequence(storage["current_slots"])
        ],
    }


def _value_storage(storage: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "value_slot_id",
        "current_id",
        "variant",
        "component_start",
        "component_stop",
        "dimension",
        "current_component_start",
        "current_component_stop",
        "is_source",
        "applies_propagator",
        "particle_id",
        "external_mask",
        "external_labels",
        "momentum_mask",
        "chirality",
        "propagator",
    )
    return {
        "component_count": int(storage["component_count"]),
        "number_type": str(storage["number_type"]),
        "metadata_compacted": True,
        "value_slots": [
            _select(_mapping(item), *fields)
            for item in _sequence(storage["value_slots"])
        ],
    }


def _source_record(record: Mapping[str, object]) -> dict[str, object]:
    return _select(
        record,
        "source_id",
        "current_id",
        "current_component_start",
        "current_component_stop",
        "value_slot",
        "source_parameter_start",
        "source_parameter_stop",
        "leg_label",
        "input_momentum_slot",
        "side",
        "crossing",
        "physical_pdg",
        "outgoing_pdg",
        "particle_id",
        "anti_particle_id",
        "source_kind",
        "wavefunction_kind",
        "source_orientation",
        "source_basis",
        "source_ir",
        "applied_crossing",
        "source_helicity",
        "chirality",
        "spin_state",
        "dimension",
        "helicity_ancestry",
        "color_state",
    )


def _runtime_stage(stage: Mapping[str, object]) -> dict[str, object]:
    interactions_compacted = bool(stage.get("interactions_compacted", False))
    interaction_ids = (
        [int(value) for value in _sequence(stage.get("interaction_ids", []))]
        if interactions_compacted
        else [
            int(_mapping(item)["interaction_id"])
            for item in _sequence(stage["interactions"])
        ]
    )
    if len(interaction_ids) != int(stage["interaction_count"]):
        raise ValueError("runtime stage interaction count is inconsistent")
    return {
        **_select(
            stage,
            "stage_index",
            "stage_kind",
            "subset_size",
            "input_current_ids",
            "output_current_ids",
            "input_value_slot_ids",
            "output_value_slot_ids",
            "interaction_count",
        ),
        "interactions_compacted": True,
        "interaction_ids": interaction_ids,
        "interactions": [],
    }


def _amplitude_stage(stage: Mapping[str, object]) -> dict[str, object]:
    contraction = stage.get("color_contraction")
    color_topology_replay = stage.get("color_topology_replay")
    return {
        "stage_kind": str(stage["stage_kind"]),
        "output_count": int(stage["output_count"]),
        "color_contraction": (
            None if contraction is None else _color_contraction(_mapping(contraction))
        ),
        "color_topology_replay": (
            None
            if color_topology_replay is None
            else _plain_mapping(_mapping(color_topology_replay))
        ),
        "roots": [
            _select(
                _mapping(root),
                "output_index",
                "root_id",
                "kind",
                "left_current_id",
                "right_current_id",
                "left_slot",
                "right_slot",
                "left_value_slot",
                "right_value_slot",
                "vertex_kind",
                "vertex_particles",
                "coupling",
                "color_weight",
                "color_sector_id",
                "contraction",
                "contraction_ir",
                "coherent_group_id",
                "helicity_weight",
                "all_sector_weight",
            )
            for root in _sequence(stage["roots"])
        ],
    }


def _color_contraction(record: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        **_select(record, "supported", "reason", "group_count"),
        "includes_color_factor": bool(record.get("includes_color_factor", False)),
        "entries": [
            _color_contraction_entry(
                _mapping(item),
                "left_group_id",
                "right_group_id",
            )
            for item in _sequence(record["entries"])
        ],
    }
    repeated_block = record.get("repeated_block")
    if repeated_block is not None:
        repeated = _mapping(repeated_block)
        compact: dict[str, object] = {
            "component_count": int(repeated["component_count"]),
            "component_group_ids": [
                int(value) for value in _sequence(repeated["component_group_ids"])
            ],
            "entries": [
                _color_contraction_entry(
                    _mapping(item),
                    "left_group_index",
                    "right_group_index",
                )
                for item in _sequence(repeated["entries"])
            ],
        }
        factorized_block = repeated.get("factorized_block")
        if factorized_block is not None:
            factorized = _mapping(factorized_block)
            compact_factorized: dict[str, object] = {
                "kind": str(factorized["kind"]),
            }
            if "rank" in factorized:
                compact_factorized["rank"] = int(factorized["rank"])
            compact_factorized["cosets"] = [
                [int(value) for value in _sequence(coset)]
                for coset in _sequence(factorized["cosets"])
            ]
            compact["factorized_block"] = compact_factorized
        result["repeated_block"] = compact
    return result


def _color_contraction_entry(
    record: Mapping[str, object], left: str, right: str
) -> dict[str, object]:
    result = _select(record, left, right, "weight", "symmetry_factor")
    if "exact_weight" in record:
        result["exact_weight"] = record["exact_weight"]
    return result


def _stage_evaluator_set(record: Mapping[str, object]) -> dict[str, object]:
    result = {
        **_select(
            record,
            "kind",
            "runtime_available",
            "runtime_unavailable_message",
            "parameter_count",
            "value_parameter_count",
            "momentum_parameter_count",
            "model_parameter_count",
            "real_valued_inputs",
            "parameter_layout",
            "stage_count",
            "required_runtime_capabilities",
        ),
        "stages": [
            _serialized_stage(_mapping(item)) for item in _sequence(record["stages"])
        ],
        "amplitude_stage": _serialized_stage(_mapping(record["amplitude_stage"])),
    }
    evaluator_manifests = [
        _mapping(_mapping(stage)["evaluator"])
        for stage in (*_sequence(result["stages"]), result["amplitude_stage"])
    ]
    actual = tuple(
        sorted(
            {
                capability
                for manifest in evaluator_manifests
                for capability in evaluator_runtime_capabilities(manifest)
            }
        )
    )
    declared = set(_required_runtime_capabilities(result))
    direct_stages = [
        stage.get("compiled_plane_arena") is not None
        for stage in (*_sequence(result["stages"]), result["amplitude_stage"])
    ]
    has_direct_capability = COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY in declared
    direct_evaluator_capabilities = {
        SYMJIT_F64_RUNTIME_CAPABILITY,
        SYMBOLICA_CPP_RUNTIME_CAPABILITY,
        SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    }
    if set(actual) & direct_evaluator_capabilities and not bool(
        direct_stages and all(direct_stages)
    ):
        raise ValueError(
            "compiled f64 artifacts require compiled-plane-arena-v1 metadata "
            "for every fused stage"
        )
    if has_direct_capability != bool(direct_stages and all(direct_stages)):
        raise ValueError(
            "compiled plane-arena capability and fused-stage metadata disagree"
        )
    if any(direct_stages) and not all(direct_stages):
        raise ValueError("compiled plane-arena metadata must cover every fused stage")
    evaluator_capabilities = declared - {
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
        COMPILED_RUNTIME_SELECTORS_CAPABILITY,
    }
    if set(actual) != evaluator_capabilities:
        raise ValueError(
            "stage evaluator runtime capabilities do not match evaluator payloads"
        )
    return result


def _serialized_stage(record: Mapping[str, object]) -> dict[str, object]:
    result = {
        **_select(
            record,
            "stage_index",
            "stage_kind",
            "subset_size",
            "evaluator_label",
            "parameter_layout",
            "output_length",
            "output_slots",
            "input_value_slot_ids",
            "output_value_slot_ids",
            "interaction_ids",
            "input_components",
            "parameter_count",
            "value_parameter_count",
            "momentum_parameter_count",
            "model_parameter_count",
            "real_valued_inputs",
            "expression_ready",
            "blockers",
        ),
        "evaluator": _evaluator(_mapping(record["evaluator"])),
    }
    direct = record.get("compiled_plane_arena")
    if direct is not None:
        result["compiled_plane_arena"] = _compiled_plane_arena_stage(
            _mapping(direct),
            stage=result,
        )
    return result


def _compiled_plane_arena_stage(
    record: Mapping[str, object],
    *,
    stage: Mapping[str, object],
) -> dict[str, object]:
    result = {
        **_select(
            record,
            "schema_version",
            "kind",
            "application_abi",
            "source_application_abi",
            "element_layout",
            "output_operation",
            "output_factor",
            "input_output_aliasing",
            "output_output_aliasing",
        ),
        "input_bindings": [
            _select(
                _mapping(item),
                "parameter_index",
                "kind",
                "source_id",
                "component",
                "global_component",
                "real_valued",
            )
            for item in _sequence(record["input_bindings"])
        ],
        "output_bindings": [
            _select(
                _mapping(item),
                "output_index",
                "arena",
                "component",
            )
            for item in _sequence(record["output_bindings"])
        ],
        "leaves": [
            _select(
                _mapping(item),
                "application_path",
                "source_application_abi",
                "optimization_level",
                "direct_codegen_optimization_level",
                "input_len",
                "output_len",
                "input_indices",
                "output_start",
                "output_stop",
            )
            for item in _sequence(record["leaves"])
        ],
    }
    symjit_contract = (
        result["application_abi"] == COMPILED_PLANE_DIRECT_APPLICATION_ABI
        and result["source_application_abi"] == SYMJIT_PLANE_APPLICATION_ABI
    )
    native_contract = (
        result["application_abi"] == NATIVE_COMPILED_DIRECT_APPLICATION_ABI
        and result["source_application_abi"] == NATIVE_COMPILED_DIRECT_APPLICATION_ABI
    )
    if (
        result["schema_version"] != 1
        or result["kind"] != "compiled-plane-arena-stage"
        or not (symjit_contract or native_contract)
        or result["element_layout"] != "split-complex-component-major"
        or result["output_operation"] != "overwrite"
        or result["output_factor"] != "identity"
        or result["input_output_aliasing"] != "forbidden"
        or result["output_output_aliasing"] != "forbidden"
    ):
        raise ValueError("compiled plane-arena stage contract is incompatible")
    if len(result["input_bindings"]) != int(stage["parameter_count"]):
        raise ValueError("compiled plane-arena input binding count is invalid")
    if len(result["output_bindings"]) != int(stage["output_length"]):
        raise ValueError("compiled plane-arena output binding count is invalid")
    if not result["leaves"]:
        raise ValueError("compiled plane-arena stage has no fused leaves")
    output_cursor = 0
    for leaf in result["leaves"]:
        leaf_map = _mapping(leaf)
        input_indices = list(_sequence(leaf_map["input_indices"]))
        if (
            leaf_map["source_application_abi"] != result["source_application_abi"]
            or (
                symjit_contract
                and leaf_map["direct_codegen_optimization_level"]
                != leaf_map["optimization_level"]
            )
            or (native_contract and leaf_map["direct_codegen_optimization_level"] != 3)
            or len(input_indices) != int(leaf_map["input_len"])
            or leaf_map["output_start"] != output_cursor
            or leaf_map["output_stop"] != output_cursor + int(leaf_map["output_len"])
        ):
            raise ValueError("compiled plane-arena leaf bindings are invalid")
        output_cursor = int(leaf_map["output_stop"])
    if output_cursor != int(stage["output_length"]):
        raise ValueError("compiled plane-arena leaves do not cover stage outputs")
    return result


def _model_parameter_evaluator(record: Mapping[str, object]) -> dict[str, object]:
    result = {
        **_select(
            record,
            "kind",
            "required_runtime_capabilities",
            "input_parameter_indices",
            "outputs",
        ),
        "evaluator": _evaluator(_mapping(record["evaluator"])),
    }
    if evaluator_runtime_capabilities(_mapping(result["evaluator"])) != (
        _required_runtime_capabilities(result)
    ):
        raise ValueError(
            "model-parameter evaluator runtime capabilities do not match its payload"
        )
    return result


def _evaluator(record: Mapping[str, object]) -> dict[str, object]:
    kind = str(record.get("kind", ""))
    if kind == "symjit-application-evaluator":
        result = _select(
            record,
            "kind",
            "runtime_capability",
            "input_len",
            "output_len",
            "application_path",
            "application_abi",
            "element_layout",
            "batch_layout",
            "compiler_type",
            "translation_mode",
            "optimization_level",
            "word_bits",
            "endianness",
            "required_defuns",
            "evaluator_state_path",
            "evaluator_state_runtime_capability",
        )
        if result["runtime_capability"] != SYMJIT_F64_RUNTIME_CAPABILITY:
            raise ValueError(
                "direct SymJIT evaluator has an invalid runtime capability"
            )
        if result["application_abi"] != SYMJIT_APPLICATION_ABI:
            raise ValueError(
                "direct SymJIT evaluator has an incompatible application ABI"
            )
        if (
            result["evaluator_state_runtime_capability"]
            != SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY
        ):
            raise ValueError(
                "direct SymJIT evaluator has an invalid fallback capability"
            )
        if result["element_layout"] != "complex-f64":
            raise ValueError("direct SymJIT evaluator has an invalid element layout")
        if (
            result["batch_layout"] != "row-major"
            or result["compiler_type"] != "native"
            or result["word_bits"] != 64
            or result["endianness"] != "little"
            or result["required_defuns"] != []
        ):
            raise ValueError("direct SymJIT evaluator has invalid execution metadata")
        if result["translation_mode"] not in {"direct", "indirect"}:
            raise ValueError("direct SymJIT evaluator has an invalid translation mode")
        optimization_level = result["optimization_level"]
        if (
            isinstance(optimization_level, bool)
            or not isinstance(optimization_level, int)
            or optimization_level not in {0, 1, 2, 3}
        ):
            raise ValueError(
                "direct SymJIT evaluator has an invalid optimization level"
            )
        plane = record.get("plane_application")
        if not isinstance(plane, Mapping):
            raise ValueError(
                "direct SymJIT evaluator predates the SymJIT 2.22 plane "
                "binding ABI; regenerate this artifact"
            )
        result["plane_application"] = _symjit_plane_application(
            plane,
            input_complex_count=int(result["input_len"]),
            output_complex_count=int(result["output_len"]),
        )
        return result
    if kind == "jit-symbolica-evaluator":
        result = _select(
            record,
            "kind",
            "runtime_capability",
            "input_len",
            "output_len",
            "evaluator_state_path",
        )
        if result["runtime_capability"] != SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY:
            raise ValueError("legacy JIT evaluator has an invalid runtime capability")
        return result
    if kind == "compiled-complex-evaluator":
        result = _select(
            record,
            "kind",
            "runtime_capability",
            "function_name",
            "input_len",
            "output_len",
            "library_path",
            "evaluator_state_path",
            "number_type",
        )
        if result["runtime_capability"] not in {
            SYMBOLICA_CPP_RUNTIME_CAPABILITY,
            SYMBOLICA_ASM_RUNTIME_CAPABILITY,
        }:
            raise ValueError("compiled evaluator has an invalid runtime capability")
        direct = record.get("native_direct_application")
        if direct is not None:
            if result["runtime_capability"] not in {
                SYMBOLICA_CPP_RUNTIME_CAPABILITY,
                SYMBOLICA_ASM_RUNTIME_CAPABILITY,
            }:
                raise ValueError(
                    "native DirectApplication metadata requires a compiled "
                    "C++ or ASM evaluator"
                )
            direct_application = _native_compiled_direct_application(
                _mapping(direct),
                expected_function_name=str(result["function_name"]),
                expected_output_count=int(result["output_len"]),
            )
            if result["library_path"] != direct_application["library_path"]:
                raise ValueError(
                    "compiled process DirectApplication must be the sole native "
                    "library payload; dense/direct dual production is forbidden"
                )
            result["native_direct_application"] = direct_application
        return result
    if kind == "chunked-symbolica-evaluator":
        result = {
            "kind": kind,
            "input_len": record["input_len"],
            "chunk_input_indices": [
                list(_sequence(indices))
                for indices in _sequence(record["chunk_input_indices"])
            ],
            "required_runtime_capabilities": list(
                _required_runtime_capabilities(record)
            ),
            "chunks": [
                _evaluator(_mapping(item)) for item in _sequence(record["chunks"])
            ],
        }
        actual = evaluator_runtime_capabilities(result)
        if actual != _required_runtime_capabilities(record):
            raise ValueError(
                "chunked evaluator required runtime capabilities do not match chunks"
            )
        return result
    raise ValueError(f"unsupported evaluator artifact kind {kind!r}")


def _symjit_plane_application(
    record: Mapping[str, object],
    *,
    input_complex_count: int,
    output_complex_count: int,
) -> dict[str, object]:
    result = _select(
        record,
        "application_path",
        "application_abi",
        "storage_abi",
        "element_layout",
        "descriptor_order",
        "input_complex_count",
        "output_complex_count",
        "input_plane_count",
        "output_plane_count",
        "compiler_type",
        "translation_mode",
        "optimization_level",
        "simd",
        "complex",
        "fast_math",
        "fast_complex",
        "compression",
        "threading",
        "direct_arena",
        "source_digest",
        "target",
    )
    expected = {
        "application_abi": SYMJIT_PLANE_APPLICATION_ABI,
        "storage_abi": SYMJIT_APPLICATION_ABI,
        "element_layout": "split-complex-plane-major",
        "descriptor_order": "inputs-re-im-then-outputs-re-im",
        "input_complex_count": input_complex_count,
        "output_complex_count": output_complex_count,
        "input_plane_count": 2 * input_complex_count,
        "output_plane_count": 2 * output_complex_count,
        "compiler_type": "native",
        "translation_mode": "symbolica-structured-instructions",
        "simd": True,
        "complex": True,
        "fast_math": True,
        "fast_complex": False,
        "threading": False,
        "direct_arena": True,
    }
    for field, expected_value in expected.items():
        if result[field] != expected_value:
            raise ValueError(
                f"direct SymJIT plane application {field} is incompatible; "
                "regenerate this artifact"
            )
    path = result["application_path"]
    digest = result["source_digest"]
    optimization_level = result["optimization_level"]
    if not isinstance(path, str) or not path:
        raise ValueError("direct SymJIT plane application path is invalid")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("direct SymJIT plane source digest is invalid")
    if (
        isinstance(optimization_level, bool)
        or not isinstance(optimization_level, int)
        or optimization_level not in {0, 1, 2, 3}
    ):
        raise ValueError("direct SymJIT plane optimization level is invalid")
    if not isinstance(result["compression"], bool):
        raise ValueError("direct SymJIT plane compression flag is invalid")
    target = _mapping(result["target"])
    if target.get("word_bits") != 64 or target.get("endianness") != "little":
        raise ValueError("direct SymJIT plane target is incompatible")
    result["target"] = _plain_mapping(target)
    return result


def _native_compiled_direct_application(
    record: Mapping[str, object],
    *,
    expected_function_name: str,
    expected_output_count: int,
) -> dict[str, object]:
    result = _select(
        record,
        "application_abi",
        "function_name",
        "source_path",
        "library_path",
        "target",
        "evaluator_state_sha256",
        "instruction_count",
        "temporary_count",
        "input_plane_count",
        "scalar_input_count",
        "output_plane_count",
        "simd_lane_width",
        "logical_stack_bytes",
        "output_semantics",
    )
    if result["application_abi"] != NATIVE_COMPILED_DIRECT_APPLICATION_ABI:
        raise ValueError("native DirectApplication has an incompatible ABI")
    if result["function_name"] != expected_function_name:
        raise ValueError(
            "native DirectApplication function identity does not match its evaluator"
        )
    target = _mapping(result["target"])
    triple = target.get("triple")
    cpu_features = target.get("cpu_features")
    if (
        not isinstance(triple, str)
        or triple not in _SUPPORTED_ARTIFACT_TARGETS
        or isinstance(cpu_features, str | bytes)
        or not isinstance(cpu_features, Sequence)
    ):
        raise ValueError("native DirectApplication target metadata is invalid")
    features = [str(item) for item in cpu_features]
    if features != sorted(set(features)) or any(not item for item in features):
        raise ValueError(
            "native DirectApplication CPU features must be sorted and unique"
        )
    result["target"] = {
        "triple": triple,
        "cpu_features": features,
    }
    digest = result["evaluator_state_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("native DirectApplication evaluator-state digest is invalid")
    for name, minimum in (
        ("instruction_count", 1),
        ("temporary_count", 0),
        ("input_plane_count", 1),
        ("scalar_input_count", 0),
        ("output_plane_count", 2),
        ("logical_stack_bytes", 1),
    ):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"native DirectApplication {name.replace('_', ' ')} is invalid"
            )
    if result["output_plane_count"] != expected_output_count * 2:
        raise ValueError(
            "native DirectApplication output planes do not match evaluator outputs"
        )
    if result["simd_lane_width"] not in {2, 4}:
        raise ValueError("native DirectApplication SIMD width is unsupported")
    if result["logical_stack_bytes"] > 64 * 1024:
        raise ValueError("native DirectApplication logical stack exceeds 64 KiB")
    if result["output_semantics"] != "factor-free-overwrite":
        raise ValueError("native DirectApplication output semantics are incompatible")
    for name in ("source_path", "library_path"):
        value = result[name]
        if not isinstance(value, str):
            raise ValueError(f"native DirectApplication {name} must be a string")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"native DirectApplication {name} must be artifact-relative"
            )
    return result


def _dag_summary(
    record: Mapping[str, object],
    *,
    require_interaction_evaluation_count: bool = False,
) -> dict[str, object]:
    result = _select(
        record,
        "current_count",
        "source_count",
        "interaction_count",
        "amplitude_root_count",
        "truncated",
    )
    interaction_evaluation_count = record.get("interaction_evaluation_count")
    if interaction_evaluation_count is None:
        if require_interaction_evaluation_count:
            raise ValueError(
                "compiled DAG summary is missing interaction_evaluation_count"
            )
    else:
        result["interaction_evaluation_count"] = interaction_evaluation_count
    return result


def _fully_resident_materialization_census(
    counts: Mapping[str, object],
    *,
    basis: str,
) -> dict[str, object]:
    exact = {
        str(name): value
        for name, value in counts.items()
        if str(name).endswith("_count")
    }
    if not exact:
        raise ValueError("fully resident materialization has no exact counters")
    for name, value in exact.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"fully resident materialization counter {name!r} is invalid"
            )
    return {
        "abi": "pyamplicol-fully-resident-materialization-census-v1",
        "basis": basis,
        "final": exact,
        "peak": dict(exact),
        "final_equals_peak": True,
    }


def _eager_materialization_census(
    process: EagerPlanV3ProcessArtifact,
) -> dict[str, object]:
    inspection = process.inspection_summary
    counts: dict[str, object] = {
        "source_count": process.dag_summary["source_count"],
        "current_count": process.dag_summary["current_count"],
        "interaction_count": process.dag_summary["interaction_count"],
        "amplitude_root_count": process.dag_summary["amplitude_root_count"],
    }
    for name in (
        "invocation_count",
        "attachment_count",
        "closure_count",
        "finalization_count",
    ):
        value = inspection.get(name)
        if value is not None:
            counts[name] = value
    return _fully_resident_materialization_census(
        counts,
        basis="immutable-fully-resident-eager-plan",
    )


def _select(record: Mapping[str, object], *names: str) -> dict[str, object]:
    missing = [name for name in names if name not in record]
    if missing:
        raise ValueError(
            "runtime execution record is missing fields: " + ", ".join(missing)
        )
    return {name: record[name] for name in names}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError("runtime execution field must be a sequence")
    return value


def _copy_evaluator_payloads(
    evaluator_payloads: _EvaluatorPayloadCollector,
    root: Path,
    *,
    prefix: str,
    process_id: str,
) -> None:
    source_root = root.expanduser().resolve(strict=True)
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_root).as_posix()
        evaluator_payloads.add_file(
            f"{prefix}/{relative}",
            path,
            process_id=process_id,
        )


def _copy_color_selector_evaluator_payloads(
    evaluator_payloads: _EvaluatorPayloadCollector,
    executions: Sequence[CompiledColorSelectorExecutionArtifact],
    *,
    prefix: str,
    process_id: str,
) -> None:
    for record in _ordered_color_selector_executions(executions):
        _copy_evaluator_payloads(
            evaluator_payloads,
            record.execution.evaluator_root,
            prefix=(
                f"{prefix}/{_color_selector_payload_prefix(record.materialized_sector_id)}"
            ),
            process_id=process_id,
        )


def _copy_helicity_selector_evaluator_payloads(
    evaluator_payloads: _EvaluatorPayloadCollector,
    executions: Sequence[CompiledHelicitySelectorExecutionArtifact],
    *,
    prefix: str,
    process_id: str,
) -> None:
    for lane_index, record in enumerate(
        _ordered_helicity_selector_executions(executions)
    ):
        lane_prefix = (
            f"{prefix}/{_HELICITY_SELECTOR_UNION_PAYLOAD_ROOT}/class-{lane_index}"
        )
        _copy_evaluator_payloads(
            evaluator_payloads,
            record.execution.evaluator_root,
            prefix=lane_prefix,
            process_id=process_id,
        )
        _copy_helicity_selector_evaluator_payloads(
            evaluator_payloads,
            record.execution.helicity_selector_executions,
            prefix=lane_prefix,
            process_id=process_id,
        )


def _color_selector_payload_prefix(materialized_sector_id: int) -> str:
    if materialized_sector_id < 0:
        raise ValueError("materialized colour-sector ids must be non-negative")
    return f"{_COLOR_SELECTOR_PAYLOAD_ROOT}/sector-{materialized_sector_id}"


def _ordered_color_selector_executions(
    executions: Sequence[CompiledColorSelectorExecutionArtifact],
) -> tuple[CompiledColorSelectorExecutionArtifact, ...]:
    ordered = tuple(
        sorted(executions, key=lambda record: record.materialized_sector_id)
    )
    sector_ids = tuple(record.materialized_sector_id for record in ordered)
    if len(sector_ids) != len(set(sector_ids)):
        raise ValueError("compiled colour-selector lane ids must be unique")
    for record in ordered:
        if record.materialized_sector_id < 0:
            raise ValueError("materialized colour-sector ids must be non-negative")
        if record.execution.color_selector_executions:
            raise ValueError("compiled colour-selector execution lanes cannot nest")
    return ordered


def _ordered_helicity_selector_executions(
    executions: Sequence[CompiledHelicitySelectorExecutionArtifact],
) -> tuple[CompiledHelicitySelectorExecutionArtifact, ...]:
    ordered = tuple(sorted(executions, key=lambda record: record.selector_domain_ids))
    seen: set[int] = set()
    for record in ordered:
        domain_ids = tuple(sorted(set(record.selector_domain_ids)))
        if not domain_ids or domain_ids != record.selector_domain_ids:
            raise ValueError(
                "compiled helicity-selector domain ids must be non-empty, "
                "sorted, and unique"
            )
        overlap = seen.intersection(domain_ids)
        if overlap:
            raise ValueError(
                "compiled helicity-selector lanes overlap selector domains: "
                + ", ".join(str(item) for item in sorted(overlap))
            )
        seen.update(domain_ids)
        if record.schedule_mode not in {"parent-closure", "nested-runtime"}:
            raise ValueError(
                "compiled helicity-selector schedule mode must be "
                "'parent-closure' or 'nested-runtime'"
            )
        if record.execution.color_selector_executions:
            raise ValueError("compiled helicity-selector execution lanes cannot nest")
        children = record.execution.helicity_selector_executions
        if children:
            if record.schedule_mode != "nested-runtime":
                raise ValueError(
                    "only a nested-runtime helicity-selector execution may "
                    "contain closure lanes"
                )
            for child in _ordered_helicity_selector_executions(children):
                if (
                    child.schedule_mode != "parent-closure"
                    or child.execution.helicity_selector_executions
                    or child.execution.color_selector_executions
                ):
                    raise ValueError(
                        "nested helicity-selector closure lanes must be "
                        "terminal parent-closure executions"
                    )
    return ordered


def build_api_validation_points(
    processes: Sequence[ProcessArtifact],
) -> dict[str, tuple[tuple[float, float, float, float], ...]]:
    """Return concrete and crossing-alias points in each public external order."""

    points: dict[str, tuple[tuple[float, float, float, float], ...]] = {}
    for process in processes:
        vectors = process.validation_point.four_vectors
        if not vectors:
            continue
        points[process.process_id] = vectors
        _add_alias_points(points, vectors=vectors, aliases=process.aliases)
    return points


def _existing_bundle_points(
    existing: ArtifactManifest | None,
) -> dict[str, tuple[tuple[float, float, float, float], ...]]:
    if existing is None:
        return {}
    points: dict[str, tuple[tuple[float, float, float, float], ...]] = {}
    for process in existing.processes:
        process_id = str(process["id"])
        validation_path = (
            existing.root / f"processes/{process_id}/validation-momenta.json"
        )
        if not validation_path.is_file():
            continue
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
        raw_points = payload.get("points") if isinstance(payload, Mapping) else None
        if not isinstance(raw_points, list) or not raw_points:
            continue
        raw_point = raw_points[0]
        if not isinstance(raw_point, list):
            raise ValueError(f"validation point for {process_id!r} is invalid")
        vectors = tuple(_validation_four_vector(item) for item in raw_point)
        points[process_id] = vectors
        aliases = process.get("aliases")
        if not isinstance(aliases, Sequence):
            raise ValueError(f"aliases for {process_id!r} are invalid")
        _add_alias_points(
            points,
            vectors=vectors,
            aliases=tuple(_mapping(alias) for alias in aliases),
        )
    return points


def _add_alias_points(
    points: dict[str, tuple[tuple[float, float, float, float], ...]],
    *,
    vectors: tuple[tuple[float, float, float, float], ...],
    aliases: Sequence[Mapping[str, object]],
) -> None:
    for alias in aliases:
        alias_id = str(alias["id"])
        raw_permutation = alias.get("external_permutation", ())
        if not isinstance(raw_permutation, Sequence):
            raise ValueError(f"alias {alias_id!r} permutation is invalid")
        permutation = tuple(int(index) for index in raw_permutation)
        if not permutation:
            permutation = tuple(range(len(vectors)))
        if sorted(permutation) != list(range(len(vectors))):
            raise ValueError(
                f"alias {alias_id!r} permutation does not match its external momenta"
            )
        if alias_id in points:
            raise ValueError(f"duplicate validation-point ID {alias_id!r}")
        alias_vectors: list[tuple[float, float, float, float] | None] = [
            None for _ in vectors
        ]
        for representative_index, alias_index in enumerate(permutation):
            alias_vectors[alias_index] = vectors[representative_index]
        if any(vector is None for vector in alias_vectors):
            raise ValueError(f"alias {alias_id!r} permutation is incomplete")
        points[alias_id] = tuple(
            vector for vector in alias_vectors if vector is not None
        )


def _validation_four_vector(
    record: object,
) -> tuple[float, float, float, float]:
    raw = _mapping(record).get("momentum")
    if not isinstance(raw, Sequence) or len(raw) != 4:
        raise ValueError("serialized validation momentum is not a four-vector")
    values = tuple(float(component) for component in raw)
    return values[0], values[1], values[2], values[3]


def _producer_metadata(
    config: GenerationConfig | RunConfig,
    *,
    runtime_capabilities: Sequence[str] = (),
    implicit_portable_jit_evidence: bool = False,
) -> dict[str, object]:
    version = package_version()
    target, c_abi = _artifact_target_metadata(
        config,
        runtime_capabilities=runtime_capabilities,
        implicit_portable_jit_evidence=implicit_portable_jit_evidence,
    )
    producer: dict[str, object] = {
        "distribution": "pyamplicol",
        "version": version,
        "versions": {
            "python_api": PYTHON_API_VERSION,
            "toml": TOML_SCHEMA_VERSION,
            "compiled_model": COMPILED_MODEL_SCHEMA_VERSION,
            "process_artifact": PROCESS_ARTIFACT_SCHEMA_VERSION,
            "runtime_physics": RUNTIME_PHYSICS_SCHEMA_VERSION,
            "symbolica_serialization": SYMBOLICA_SERIALIZATION_ABI,
            "c_abi": c_abi,
        },
        "target": target,
    }
    producer["native_build_inputs_sha256"] = active_native_build_inputs_sha256()
    source_revision = active_source_revision()
    if source_revision is not None:
        producer["git_revision"] = source_revision
    return producer


def _artifact_target_metadata(
    config: GenerationConfig | RunConfig,
    *,
    runtime_capabilities: Sequence[str] = (),
    implicit_portable_jit_evidence: bool = False,
) -> tuple[dict[str, object], int]:
    target, c_abi = _target_metadata(config)
    requested_portable_jit = (
        implicit_portable_jit_evidence
        if isinstance(config, GenerationConfig)
        else (
            str(config.evaluator.backend) == "jit"
            and config.evaluator.jit.optimization_level in {1, 2}
        )
    )
    portable_64le = requested_portable_jit and not {
        SYMBOLICA_ASM_RUNTIME_CAPABILITY,
        SYMBOLICA_CPP_RUNTIME_CAPABILITY,
        SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    }.intersection(runtime_capabilities)
    if portable_64le:
        target = {
            "triple": PORTABLE_64LE_TARGET,
            "cpu_features": [],
        }
    return target, c_abi


def _implicit_generation_portable_jit_evidence(
    processes: Sequence[ProcessArtifact],
) -> bool:
    if not processes or any(
        not isinstance(process, CompiledProcessArtifact) for process in processes
    ):
        return False
    records: list[Mapping[str, object]] = []
    for process in cast(Sequence[CompiledProcessArtifact], processes):
        records.append(process.stage_manifest)
        if process.model_parameter_evaluator is not None:
            records.append(process.model_parameter_evaluator)
        for selector in process.color_selector_executions:
            records.extend(_compiled_execution_evidence_records(selector.execution))
        if process.helicity_sum_execution is not None:
            records.extend(
                _compiled_execution_evidence_records(process.helicity_sum_execution)
            )
        for selector in process.helicity_selector_executions:
            records.extend(_compiled_execution_evidence_records(selector.execution))
    evaluator_count = 0
    for record in records:
        valid, count = _mapping_has_only_portable_symjit_evaluators(record)
        if not valid or count == 0:
            return False
        evaluator_count += count
    return evaluator_count > 0


def _compiled_execution_evidence_records(
    execution: CompiledExecutionArtifact,
) -> list[Mapping[str, object]]:
    records: list[Mapping[str, object]] = [execution.stage_manifest]
    if execution.model_parameter_evaluator is not None:
        records.append(execution.model_parameter_evaluator)
    for selector in execution.color_selector_executions:
        records.extend(_compiled_execution_evidence_records(selector.execution))
    for selector in execution.helicity_selector_executions:
        records.extend(_compiled_execution_evidence_records(selector.execution))
    return records


def _mapping_has_only_portable_symjit_evaluators(
    value: Mapping[str, object],
) -> tuple[bool, int]:
    evaluator_count = 0
    valid = True

    def visit(item: object) -> None:
        nonlocal evaluator_count, valid
        if not valid:
            return
        if isinstance(item, Mapping):
            kind = item.get("kind")
            if kind == "symjit-application-evaluator":
                evaluator_count += 1
                plane = item.get("plane_application")
                if (
                    item.get("optimization_level") not in (1, 2)
                    or not isinstance(plane, Mapping)
                    or plane.get("optimization_level") not in (1, 2)
                ):
                    valid = False
                    return
            elif kind in {"compiled-complex-evaluator", "jit-symbolica-evaluator"}:
                valid = False
                return
            if item.get("source_application_abi") == SYMJIT_PLANE_APPLICATION_ABI:
                for field in (
                    "optimization_level",
                    "direct_codegen_optimization_level",
                ):
                    if field in item and item[field] not in (1, 2):
                        valid = False
                        return
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, Sequence) and not isinstance(item, str | bytes):
            for nested in item:
                visit(nested)

    visit(value)
    return valid, evaluator_count


def _target_metadata(
    config: GenerationConfig | RunConfig,
) -> tuple[dict[str, object], int]:
    requires_native_features = (
        isinstance(config, RunConfig)
        and str(config.evaluator.backend) != "jit"
        and config.evaluator.cpp.native_arch
    )
    try:
        rusticol = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(rusticol)
        info = rusticol.target_info()
        triple = str(info.triple)
        available_features = tuple(str(item) for item in info.cpu_features)
        if available_features != tuple(sorted(set(available_features))):
            raise RuntimeError(
                "Rusticol returned non-canonical target CPU feature metadata"
            )
        c_abi = int(rusticol.abi_version())
    except (
        AttributeError,
        ImportError,
        OSError,
        importlib.metadata.PackageNotFoundError,
    ) as rusticol_error:
        try:
            from pyamplicol._sdk import load_sdk_info

            sdk = load_sdk_info()
        except (
            ImportError,
            OSError,
            RuntimeError,
            importlib.metadata.PackageNotFoundError,
        ) as sdk_error:
            raise RuntimeError(
                "Rusticol target metadata is unavailable; install or build the "
                "pyamplicol native extension before generating process artifacts"
            ) from sdk_error
        if requires_native_features:
            raise RuntimeError(
                "native C++ evaluator generation requires Rusticol CPU-feature "
                "introspection; SDK target metadata alone is insufficient"
            ) from rusticol_error
        triple = sdk.target
        available_features = ()
        c_abi = int(sdk.abi_version)
    required_features = available_features if requires_native_features else ()
    if triple not in _SUPPORTED_ARTIFACT_TARGETS:
        raise RuntimeError(
            f"Rusticol process artifacts are not supported on target {triple!r}"
        )
    if requires_native_features and not required_features:
        raise RuntimeError(
            "Rusticol did not detect any CPU features for a native C++ evaluator; "
            "refusing to emit incomplete target metadata"
        )
    return {
        "triple": triple,
        "cpu_features": list(required_features),
    }, c_abi


def _model_metadata(
    source: ModelSource,
    compiled: CompiledModel,
) -> dict[str, object]:
    source_kind = {
        "built-in-sm": "built-in-sm",
        "built-in-sm-heft": "built-in-sm-heft",
        "ufo": "ufo",
        "json": "ufo-json",
        "compiled": "compiled-model",
        "prepared": "compiled-model",
    }[source.kind]
    digest = str(compiled.source.get("digest", ""))
    if source.kind == "compiled" and source.path is not None:
        digest = hashlib.sha256(source.path.read_bytes()).hexdigest()
    if len(digest) != 64:
        raise ValueError("compiled model source has no canonical SHA-256 digest")
    restriction = (
        (
            source.restriction.name
            if isinstance(source.restriction, Path)
            else source.restriction
        )
        if source.restriction is not None
        else "default"
        if source.kind in {"ufo", "json"}
        else None
    )
    return {
        "name": compiled.name,
        "source_kind": source_kind,
        "content_sha256": digest,
        "compiled_schema_version": COMPILED_MODEL_SCHEMA_VERSION,
        "restriction": restriction,
    }


def _dependency_metadata(source: ModelSource) -> tuple[dict[str, object], ...]:
    dependencies: list[dict[str, object]] = [
        {
            "name": "symbolica",
            "version": _distribution_version("symbolica", "unknown"),
            "source": "https://symbolica.io/",
            "license": "Symbolica Software License Agreement",
        }
    ]
    if source.kind in {"ufo", "json"}:
        dependencies.append(
            {
                "name": "ufo-model-loader",
                "version": _distribution_version("ufo-model-loader", "unknown"),
                "source": "https://github.com/alphal00p/ufo_model_loader",
                "license": "MIT",
            }
        )
    return tuple(dependencies)


def _validate_append_compatibility(
    existing: ArtifactManifest | None,
    *,
    producer: Mapping[str, object],
    model: Mapping[str, object],
    eager_pack_identity: Mapping[str, object] | None,
    requested_bytes: bytes,
    effective_bytes: bytes,
    adjustments: Sequence[Mapping[str, str]],
    processes: Sequence[ProcessArtifact],
) -> None:
    if existing is None:
        return
    _reject_legacy_eager_append(existing)
    existing_producer = _plain_mapping(existing.producer)
    producer_identity_fields = (
        ("distribution", "distribution"),
        ("version", "version"),
        ("git_revision", "source revision"),
        ("native_build_inputs_sha256", "native build-input digest"),
    )
    for field, label in producer_identity_fields:
        if existing_producer.get(field) != producer.get(field):
            raise ValueError(
                f"append producer {label} differs from the existing artifact"
            )
    existing_capabilities = _required_runtime_capabilities(existing.runtime)
    existing_uses_eager = (
        EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY in existing_capabilities
        or any(record.path == _EAGER_KERNEL_PACK_PATH for record in existing.payloads)
    )
    existing_identity = existing.extensions.get(_EAGER_PACK_IDENTITY_EXTENSION)
    if existing_uses_eager or existing_identity is not None:
        if not isinstance(existing_identity, Mapping):
            raise ValueError(
                "append eager artifact has no canonical prepared-pack identity; "
                "replace the artifact"
            )
        if eager_pack_identity is None or _plain_mapping(
            existing_identity
        ) != _plain_mapping(eager_pack_identity):
            raise ValueError(
                "append prepared kernel pack identity differs from the existing "
                "artifact; replace the artifact"
            )
    if _plain_mapping(existing.model) != dict(model):
        raise ValueError("append model provenance differs from the existing artifact")
    if existing_producer.get("target") != producer.get("target"):
        raise ValueError("append target differs from the existing artifact")
    requested_path = existing.root / str(existing.configuration["requested_path"])
    effective_path = existing.root / str(existing.configuration["effective_path"])
    if requested_path.read_bytes() != requested_bytes:
        raise ValueError("append requested configuration differs from the artifact")
    if effective_path.read_bytes() != effective_bytes:
        raise ValueError("append effective configuration differs from the artifact")
    existing_adjustments: list[dict[str, str]] = []
    for item in _sequence(existing.configuration["adjustments"]):
        adjustment = _mapping(item)
        existing_adjustments.append(
            {
                "path": str(adjustment["path"]),
                "reason": str(adjustment["reason"]),
            }
        )
    if existing_adjustments != list(adjustments):
        raise ValueError("append configuration adjustments differ from the artifact")
    existing_ids = {str(record["id"]) for record in existing.processes}
    duplicates = existing_ids.intersection(process.process_id for process in processes)
    if duplicates:
        raise ValueError(
            "append process IDs already exist: " + ", ".join(sorted(duplicates))
        )


def _reject_legacy_eager_append(existing: ArtifactManifest) -> None:
    raw_capabilities = existing.runtime.get("required_runtime_capabilities", ())
    if (
        not isinstance(raw_capabilities, str | bytes)
        and isinstance(raw_capabilities, Sequence)
        and EAGER_RUNTIME_CAPABILITY in raw_capabilities
    ):
        raise ValueError(
            "legacy eager plan-v2 artifacts cannot be extended; regenerate the "
            "artifact with the current eager plan-v3 runtime or use replace mode"
        )


def _existing_process_records(
    existing: ArtifactManifest | None,
) -> list[dict[str, object]]:
    if existing is None:
        return []
    return [_plain_mapping(record) for record in existing.processes]


def _existing_evaluator_entries(
    existing: ArtifactManifest | None,
) -> list[dict[str, object]]:
    if existing is None:
        return []
    path = existing.root / str(existing.runtime["evaluator_manifest_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PROCESS_ARTIFACT_SCHEMA_VERSION
        or payload.get("kind") != "pyamplicol-runtime-execution-set"
    ):
        raise ValueError("append artifact has an incompatible evaluator-set manifest")
    entries = payload.get("processes")
    if not isinstance(entries, list) or not all(
        isinstance(item, dict) for item in entries
    ):
        raise ValueError("append evaluator-set process list is invalid")
    return [dict(item) for item in entries]


def _existing_required_runtime_capabilities(
    existing: ArtifactManifest | None,
) -> tuple[str, ...]:
    if existing is None:
        return ()
    return _required_runtime_capabilities(existing.runtime)


def _compiled_process_runtime_capabilities(
    process: CompiledProcessArtifact,
) -> tuple[str, ...]:
    capabilities = set(_required_runtime_capabilities(process.stage_manifest))
    capabilities.update(
        _runtime_schema_walsh_color_contraction_capabilities(
            _runtime_schema_mapping(process.runtime_schema)
        )
    )
    if process.color_contraction_payload is not None:
        capabilities.add(SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY)
    if process.model_parameter_evaluator is not None:
        capabilities.update(
            _required_runtime_capabilities(process.model_parameter_evaluator)
        )
    if _uses_primary_helicity_recurrence(process):
        capabilities.add(COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY)
    if process.color_selector_executions:
        capabilities.add(COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY)
        capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        for record in _ordered_color_selector_executions(
            process.color_selector_executions
        ):
            capabilities.update(
                _compiled_execution_runtime_capabilities(record.execution)
            )
    auxiliary = process.helicity_sum_execution
    if auxiliary is not None:
        capabilities.add(COMPILED_HELICITY_DUAL_LANE_CAPABILITY)
        capabilities.update(_compiled_execution_runtime_capabilities(auxiliary))
    selector_lanes = _ordered_helicity_selector_executions(
        process.helicity_selector_executions
    )
    if selector_lanes:
        capabilities.add(COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY)
        capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        for record in selector_lanes:
            capabilities.update(
                _compiled_execution_runtime_capabilities(record.execution)
            )
    return tuple(sorted(capabilities))


def _uses_primary_helicity_recurrence(process: CompiledProcessArtifact) -> bool:
    return _runtime_schema_uses_primary_helicity_recurrence(
        _runtime_schema_mapping(process.runtime_schema),
        has_helicity_sum_execution=process.helicity_sum_execution is not None,
    )


def _runtime_schema_uses_primary_helicity_recurrence(
    runtime_schema: Mapping[str, object],
    *,
    has_helicity_sum_execution: bool,
) -> bool:
    if has_helicity_sum_execution:
        return False
    recurrence = runtime_schema.get("helicity_recurrence")
    return isinstance(recurrence, Mapping) and isinstance(
        recurrence.get("materialization"), Mapping
    )


def _runtime_schema_uses_walsh_color_contraction(
    runtime_schema: Mapping[str, object],
) -> bool:
    return bool(_runtime_schema_walsh_color_contraction_capabilities(runtime_schema))


def _runtime_schema_walsh_color_contraction_capabilities(
    runtime_schema: Mapping[str, object],
) -> frozenset[str]:
    amplitude_stage = runtime_schema.get("amplitude_stage")
    if not isinstance(amplitude_stage, Mapping):
        return frozenset()
    contraction = amplitude_stage.get("color_contraction")
    if not isinstance(contraction, Mapping):
        return frozenset()
    repeated_block = contraction.get("repeated_block")
    if not isinstance(repeated_block, Mapping):
        return frozenset()
    factorized_block = repeated_block.get("factorized_block")
    if not isinstance(factorized_block, Mapping):
        return frozenset()
    kind = factorized_block.get("kind")
    if kind == "klein-four-walsh":
        return frozenset({COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY})
    if kind == "elementary-abelian-walsh":
        return frozenset({COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY})
    return frozenset()


def _compiled_execution_runtime_capabilities(
    execution: CompiledExecutionArtifact,
) -> tuple[str, ...]:
    capabilities = set(_required_runtime_capabilities(execution.stage_manifest))
    capabilities.update(
        _runtime_schema_walsh_color_contraction_capabilities(
            _runtime_schema_mapping(execution.runtime_schema)
        )
    )
    if _runtime_schema_uses_primary_helicity_recurrence(
        _runtime_schema_mapping(execution.runtime_schema),
        has_helicity_sum_execution=False,
    ):
        capabilities.add(COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY)
    if execution.model_parameter_evaluator is not None:
        capabilities.update(
            _required_runtime_capabilities(execution.model_parameter_evaluator)
        )
    if execution.color_selector_executions:
        capabilities.add(COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY)
        capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        for record in _ordered_color_selector_executions(
            execution.color_selector_executions
        ):
            capabilities.update(
                _compiled_execution_runtime_capabilities(record.execution)
            )
    selector_lanes = _ordered_helicity_selector_executions(
        execution.helicity_selector_executions
    )
    if selector_lanes:
        capabilities.add(COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY)
        capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        for record in selector_lanes:
            capabilities.update(
                _compiled_execution_runtime_capabilities(record.execution)
            )
    return tuple(sorted(capabilities))


def _process_runtime_capabilities(
    process: ProcessArtifact,
) -> tuple[str, ...]:
    if isinstance(process, OnTheFlyProcessArtifact):
        return _on_the_fly_process_runtime_capabilities(process)
    if isinstance(process, RecurrenceProcessArtifact):
        return _recurrence_process_runtime_capabilities(process)
    if isinstance(process, EagerPlanV3ProcessArtifact):
        return _EAGER_PLAN_V3_RUNTIME_CAPABILITIES
    return _compiled_process_runtime_capabilities(process)


def _on_the_fly_process_runtime_capabilities(
    process: OnTheFlyProcessArtifact,
) -> tuple[str, ...]:
    if process.color_accuracy == "lc":
        color_capability = ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY
    elif process.color_accuracy in {"nlc", "full"}:
        color_capability = ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY
    else:
        raise ValueError(
            f"unsupported on-the-fly color accuracy {process.color_accuracy!r}"
        )
    capabilities = {
        color_capability,
        ON_THE_FLY_RUNTIME_CAPABILITY,
    }
    if _uses_symmetric_group_fft_color_contraction(process):
        capabilities.add(SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY)
    return tuple(sorted(capabilities))


def _recurrence_process_runtime_capabilities(
    process: RecurrenceProcessArtifact,
) -> tuple[str, ...]:
    color_capability = (
        RECURRENCE_COLOR_RUNTIME_CAPABILITY
        if process.color_accuracy == "lc"
        else RECURRENCE_CONTRACTED_COLOR_RUNTIME_CAPABILITY
    )
    capabilities = {
        RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
        color_capability,
    }
    if process.helicity_selector_companion is not None:
        capabilities.add(RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY)
    if _uses_symmetric_group_fft_color_contraction(process):
        capabilities.add(SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY)
    return tuple(sorted(capabilities))


def _uses_symmetric_group_fft_color_contraction(
    process: OnTheFlyProcessArtifact | RecurrenceProcessArtifact,
) -> bool:
    summary = process.color_contraction_summary
    if not isinstance(summary, Mapping):
        return False
    factorization = summary.get("factorization")
    return isinstance(factorization, Mapping) and (
        factorization.get("kind") == "symmetric-group-fourier"
    )


def _validate_symmetric_group_fft_provenance(
    summary: Mapping[str, object],
    *,
    context: str,
) -> None:
    factorization_raw = summary.get("factorization")
    symmetric_group_fft = (
        isinstance(factorization_raw, Mapping)
        and factorization_raw.get("kind") == "symmetric-group-fourier"
    )
    provenance_raw = summary.get("fft_provenance")
    if not symmetric_group_fft:
        if provenance_raw is not None or "fft_provenance" in summary:
            raise ValueError(f"{context} non-FFT summary carries FFT provenance")
        return
    factorization = _mapping(factorization_raw)
    provenance = _mapping(provenance_raw)
    if set(provenance) != {
        "method",
        "degree",
        "channel_count",
        "covered_local_group_count",
        "residual_group_count",
        "residual_entry_count",
        "raw_kernel_bytes",
        "transformed_kernel_bytes",
        "capability",
    }:
        raise ValueError(f"{context} FFT provenance fields are invalid")
    degree = _nonnegative_integer(
        factorization.get("rank"), f"{context} FFT degree", minimum=2
    )
    channel_count = _nonnegative_integer(
        factorization.get("coset_count"), f"{context} FFT channel count", minimum=1
    )
    group_count = _nonnegative_integer(
        summary.get("group_count"), f"{context} group count", minimum=1
    )
    component_count = _nonnegative_integer(
        summary.get("component_count"), f"{context} component count", minimum=1
    )
    if degree > 10 or group_count % component_count:
        raise ValueError(f"{context} FFT group shape is invalid")
    group_order = math.factorial(degree)
    covered_local_group_count = channel_count * group_order
    local_group_count = group_count // component_count
    if covered_local_group_count > local_group_count:
        raise ValueError(f"{context} FFT channels exceed local groups")
    residual_group_count = local_group_count - covered_local_group_count
    residual_entry_count = (
        local_group_count * (local_group_count + 1) // 2
        - covered_local_group_count * (covered_local_group_count + 1) // 2
    )
    kernel_entry_count = channel_count * (channel_count + 1) // 2 * group_order
    raw_kernel_bytes = kernel_entry_count * 16
    transformed_kernel_bytes = kernel_entry_count * 8
    expected = {
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
        dict(provenance) != expected
        or summary.get("entry_count") != kernel_entry_count + residual_entry_count
        or summary.get("logical_entry_count")
        != (kernel_entry_count + residual_entry_count) * component_count
    ):
        raise ValueError(f"{context} symmetric-group FFT provenance is inconsistent")


def _required_runtime_capabilities(
    record: Mapping[str, object],
) -> tuple[str, ...]:
    raw = record.get("required_runtime_capabilities")
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise ValueError("runtime capability metadata must be a sequence")
    values = tuple(str(item) for item in raw)
    if values != tuple(sorted(set(values))):
        raise ValueError("runtime capabilities must be sorted and unique")
    unknown = set(values) - (
        EVALUATOR_RUNTIME_CAPABILITIES
        | {
            EAGER_PLAN_V3_RUNTIME_CAPABILITY,
            RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
            RECURRENCE_COLOR_RUNTIME_CAPABILITY,
        }
    )
    if unknown:
        raise ValueError(
            "unsupported evaluator runtime capabilities: " + ", ".join(sorted(unknown))
        )
    return values


def _extensions(
    existing: ArtifactManifest | None,
    *,
    processes: Sequence[ProcessArtifact],
    timings: Mapping[str, float],
    api_bundle_requested: bool,
    api_bundle_path: str | None,
    eager_pack_identity: Mapping[str, object] | None,
    execution_manifest_sha256_by_process: Mapping[str, str],
    evaluator_payload_container: Mapping[str, object] | None,
    recurrence_schedule_sharing: Mapping[str, object] | None,
) -> dict[str, object]:
    result = {} if existing is None else _plain_mapping(existing.extensions)
    previous = result.get("generation")
    generation = dict(previous) if isinstance(previous, Mapping) else {}
    concrete = generation.get("concrete_processes")
    process_records = list(concrete) if isinstance(concrete, list) else []
    for process in processes:
        record: dict[str, object] = {
            "id": process.process_id,
            "expression": process.expression,
            "validation_momenta_path": (
                f"processes/{process.process_id}/validation-momenta.json"
            ),
            "filters": dict(process.generation_filters),
        }
        if _is_prepared_kernel_process(process):
            record["execution_manifest_sha256"] = execution_manifest_sha256_by_process[
                process.process_id
            ]
        else:
            record["runtime_schema_sha256"] = process.runtime_schema.sha256
            if process.helicity_sum_execution is not None:
                record["helicity_sum_runtime_schema_sha256"] = (
                    process.helicity_sum_execution.runtime_schema.sha256
                )
            if process.helicity_selector_executions:
                record["helicity_selector_runtime_schema_sha256s"] = [
                    lane.execution.runtime_schema.sha256
                    for lane in _ordered_helicity_selector_executions(
                        process.helicity_selector_executions
                    )
                ]
        process_records.append(record)
    recurrence_schedule_profiles: dict[str, object] = {}
    for process in processes:
        if not isinstance(process, RecurrenceProcessArtifact):
            continue
        schedule_digest = _canonical_sha256(
            process.recurrence_schedule_digest,
            f"recurrence schedule {process.process_id!r} generation-profile digest",
        )
        profile = _deep_plain(process.generation_profile)
        previous_profile = recurrence_schedule_profiles.get(schedule_digest)
        if previous_profile is not None and previous_profile != profile:
            raise ValueError(
                "shared recurrence schedule has inconsistent generation telemetry: "
                f"{schedule_digest}"
            )
        recurrence_schedule_profiles[schedule_digest] = profile
    generation.update(
        {
            "schema_version": 1,
            "concrete_processes": process_records,
            "phase_timings_seconds": {
                str(name): float(value) for name, value in timings.items()
            },
            "api_bundle": {
                "requested": api_bundle_requested,
                "emitted": api_bundle_path is not None,
                "path": api_bundle_path,
                "scope": "root-artifact",
            },
        }
    )
    if recurrence_schedule_profiles:
        generation["recurrence_schedule_profiles"] = recurrence_schedule_profiles
    result["generation"] = generation
    if eager_pack_identity is not None:
        result[_EAGER_PACK_IDENTITY_EXTENSION] = _plain_mapping(eager_pack_identity)
    if recurrence_schedule_sharing is not None:
        result[_RECURRENCE_SCHEDULE_SHARING_EXTENSION] = _plain_mapping(
            recurrence_schedule_sharing
        )
    if evaluator_payload_container is None:
        result.pop(_EVALUATOR_PAYLOAD_CONTAINER_EXTENSION, None)
    else:
        result[_EVALUATOR_PAYLOAD_CONTAINER_EXTENSION] = _plain_mapping(
            evaluator_payload_container
        )
    return result


def _default_api_bundle_hook() -> ApiBundleHook | None:
    try:
        from pyamplicol.artifacts.api_bundle import emit_api_bundle
    except ImportError:
        return None
    return cast("ApiBundleHook", emit_api_bundle)


def _call_api_bundle_hook(
    builder: ArtifactBuilder,
    hook: ApiBundleHook,
    validation_points: Mapping[str, Sequence[Sequence[float]]],
) -> str:
    result = hook(builder, validation_points)
    paths = tuple(
        str(item["path"])
        if isinstance(item, Mapping)
        else str(getattr(item, "path", ""))
        for item in result
    )
    if not paths or any(not path.startswith("API/") for path in paths):
        raise ValueError("root API-bundle emitter returned an invalid payload set")
    return "API"


def _validate_artifact_references(manifest: ArtifactManifest) -> None:
    declared = {record.path for record in manifest.payloads}
    _validate_evaluator_payload_container(manifest)
    _validate_recurrence_schedule_sharing(manifest)
    required = {
        str(manifest.configuration["requested_path"]),
        str(manifest.configuration["effective_path"]),
        str(manifest.runtime["evaluator_manifest_path"]),
        *(str(process["physics_path"]) for process in manifest.processes),
        *(
            f"processes/{process['id']}/execution.json"
            for process in manifest.processes
        ),
    }
    api_path = manifest.runtime.get("api_bundle_path")
    if api_path is not None:
        api_prefix = str(api_path).rstrip("/") + "/"
        if not any(path.startswith(api_prefix) for path in declared):
            raise ValueError("artifact API bundle has no declared payloads")
    missing = required - declared
    if missing:
        raise ValueError(
            "artifact references undeclared payloads: " + ", ".join(sorted(missing))
        )
    actual = {
        path.relative_to(manifest.root).as_posix()
        for path in manifest.root.rglob("*")
        if path.is_file() and path.name != "artifact.json"
    }
    undeclared = actual - declared
    if undeclared:
        raise ValueError(
            "artifact contains undeclared files: " + ", ".join(sorted(undeclared))
        )


def _validate_recurrence_schedule_sharing(manifest: ArtifactManifest) -> None:
    raw = manifest.extensions.get(_RECURRENCE_SCHEDULE_SHARING_EXTENSION)
    if raw is None:
        return
    extension = _mapping(raw)
    expected_fields = {
        "kind",
        "schema_version",
        "index_path",
        "index_sha256",
        "schedule_count",
        "binding_count",
        "schedule_alias_count",
        "runtime_ownership",
        "interning_phase",
    }
    if set(extension) != expected_fields:
        raise ValueError("recurrence schedule-sharing extension fields are invalid")
    if (
        extension.get("kind") != "pyamplicol-recurrence-schedule-sharing"
        or extension.get("schema_version") != RECURRENCE_SCHEDULE_SHARING_SCHEMA_VERSION
        or extension.get("index_path") != RECURRENCE_SCHEDULE_INDEX_PATH
        or extension.get("runtime_ownership") != "root-schedule-plus-process-binding"
        or extension.get("interning_phase") != "before-direct-lowering"
    ):
        raise ValueError("recurrence schedule-sharing extension is incompatible")
    records = {record.path: record for record in manifest.payloads}
    index_record = records.get(RECURRENCE_SCHEDULE_INDEX_PATH)
    if (
        index_record is None
        or index_record.role != "evaluator-manifest"
        or index_record.process_id is not None
        or index_record.sha256 != extension.get("index_sha256")
    ):
        raise ValueError("recurrence schedule-sharing index record is invalid")
    index = _mapping(
        json.loads(
            (manifest.root / RECURRENCE_SCHEDULE_INDEX_PATH).read_text(encoding="utf-8")
        )
    )
    if (
        index.get("kind") != extension["kind"]
        or index.get("schema_version") != extension["schema_version"]
        or index.get("runtime_ownership") != extension["runtime_ownership"]
        or index.get("interning_phase") != extension["interning_phase"]
    ):
        raise ValueError("recurrence schedule-sharing index is incompatible")
    schedules = _sequence(index.get("schedules"))
    bindings = _sequence(index.get("bindings"))
    counts = {
        "schedule_count": len(schedules),
        "binding_count": len(bindings),
        "schedule_alias_count": len(bindings) - len(schedules),
    }
    if any(index.get(name) != value for name, value in counts.items()) or any(
        extension.get(name) != value for name, value in counts.items()
    ):
        raise ValueError("recurrence schedule-sharing counts are inconsistent")

    schedule_paths: dict[str, str] = {}
    declared_processes_by_schedule: dict[str, tuple[str, ...]] = {}
    for position, raw_schedule in enumerate(schedules):
        schedule = _mapping(raw_schedule)
        digest = _canonical_sha256(
            schedule.get("digest"),
            f"recurrence shared schedule {position} digest",
        )
        if digest in schedule_paths:
            raise ValueError(f"recurrence shared schedule {digest} is duplicated")
        expected_path = f"recurrence/schedules/{digest}/recurrence-runtime.pacbin"
        if schedule.get("path") != expected_path:
            raise ValueError(
                f"recurrence shared schedule {digest} has an invalid root path"
            )
        raw_process_ids = _sequence(schedule.get("process_ids"))
        declared_process_ids = tuple(str(value) for value in raw_process_ids)
        if not declared_process_ids or declared_process_ids != tuple(
            sorted(set(declared_process_ids))
        ):
            raise ValueError(
                f"recurrence shared schedule {digest} has invalid process ownership"
            )
        record = records.get(expected_path)
        if (
            record is None
            or record.role != "evaluator-state"
            or record.process_id is not None
            or record.sha256 != schedule.get("sha256")
            or record.size_bytes != schedule.get("size_bytes")
        ):
            raise ValueError(
                f"recurrence shared schedule {digest} has an invalid payload record"
            )
        schedule_paths[digest] = expected_path
        declared_processes_by_schedule[digest] = declared_process_ids

    process_ids = {str(process["id"]) for process in manifest.processes}
    binding_process_ids: set[str] = set()
    support_masks: set[tuple[int, ...]] = set()
    bound_processes_by_schedule: dict[str, list[str]] = {}
    for position, raw_binding in enumerate(bindings):
        binding = _mapping(raw_binding)
        process_id = str(binding.get("process_id"))
        if process_id not in process_ids or process_id in binding_process_ids:
            raise ValueError(
                f"recurrence process binding {position} has an invalid process ID"
            )
        binding_process_ids.add(process_id)
        if binding.get("abi") != RECURRENCE_PROCESS_BINDING_ABI:
            raise ValueError(
                f"recurrence process binding {process_id!r} has an invalid ABI"
            )
        _canonical_sha256(
            binding.get("process_digest"),
            f"recurrence process binding {process_id!r} process digest",
        )
        _canonical_sha256(
            binding.get("process_semantic_digest"),
            f"recurrence process binding {process_id!r} semantic digest",
        )
        _recurrence_binding_native_schedule_semantic_digest(
            binding,
            process_id=process_id,
        )
        _validate_recurrence_process_remap(
            binding.get("remap"),
            context=f"recurrence process binding {process_id!r}",
        )
        schedule_digest = _canonical_sha256(
            binding.get("schedule_digest"),
            f"recurrence process binding {process_id!r} schedule digest",
        )
        root_path = schedule_paths.get(schedule_digest)
        if root_path is None:
            raise ValueError(
                f"recurrence process binding {process_id!r} references an unknown "
                "schedule"
            )
        bound_processes_by_schedule.setdefault(schedule_digest, []).append(process_id)
        raw_support_words = _sequence(binding.get("process_support_words"))
        support_mask = tuple(raw_support_words)
        if (
            not support_mask
            or any(
                isinstance(word, bool)
                or not isinstance(word, int)
                or word < 0
                or word > (1 << 64) - 1
                for word in support_mask
            )
            or sum(int(word).bit_count() for word in support_mask) != 1
            or support_mask[-1] == 0
            or support_mask in support_masks
        ):
            raise ValueError(
                f"recurrence process binding {process_id!r} has an invalid support mask"
            )
        support_masks.add(support_mask)
        relative_binding_path = binding.get("path")
        if relative_binding_path != "recurrence-binding.bin":
            raise ValueError(
                f"recurrence process binding {process_id!r} has an invalid binding path"
            )
        binding_path = f"processes/{process_id}/{relative_binding_path}"
        binding_record = records.get(binding_path)
        if (
            binding_record is None
            or binding_record.role != "evaluator-state"
            or binding_record.process_id != process_id
            or binding_record.sha256 != binding.get("sha256")
            or binding_record.size_bytes != binding.get("size_bytes")
        ):
            raise ValueError(
                f"recurrence process binding {process_id!r} has an invalid payload"
            )
    for digest, declared_process_ids in declared_processes_by_schedule.items():
        bound_process_ids = tuple(sorted(bound_processes_by_schedule.get(digest, ())))
        if bound_process_ids != declared_process_ids:
            raise ValueError(
                f"recurrence shared schedule {digest} binding ownership is inconsistent"
            )


def _recurrence_binding_native_schedule_semantic_digest(
    binding: Mapping[str, object],
    *,
    process_id: str,
) -> str:
    return _canonical_sha256(
        binding.get("native_schedule_semantic_digest"),
        (f"recurrence process binding {process_id!r} native schedule semantic digest"),
    )


def _validate_recurrence_process_remap(
    value: object,
    *,
    context: str,
) -> None:
    remap = _mapping(value)
    expected = {
        "bijection_digest",
        "source_slots",
        "source_momentum_signs",
        "source_helicity_signs",
        "source_state_offsets",
        "source_state_indices",
        "public_flow_ids",
        "physical_sector_ids",
        "state_templates",
        "source_templates",
        "direct_executors",
        "parameter_slots",
    }
    if set(remap) != expected:
        raise ValueError(f"{context} remap fields are invalid")
    _canonical_sha256(
        remap.get("bijection_digest"),
        f"{context} process-bijection digest",
    )

    def permutation(name: str) -> tuple[int, ...]:
        values = tuple(_sequence(remap.get(name)))
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in values
        ) or tuple(sorted(values)) != tuple(range(len(values))):
            raise ValueError(f"{context} {name} is not a permutation")
        return cast(tuple[int, ...], values)

    sources = permutation("source_slots")
    permutation("public_flow_ids")
    permutation("physical_sector_ids")
    for name in ("source_momentum_signs", "source_helicity_signs"):
        signs = tuple(_sequence(remap.get(name)))
        if len(signs) != len(sources) or any(sign not in {-1, 1} for sign in signs):
            raise ValueError(f"{context} {name} is invalid")
    source_state_offsets = tuple(_sequence(remap.get("source_state_offsets")))
    source_state_indices = tuple(_sequence(remap.get("source_state_indices")))
    if (
        len(source_state_offsets) != len(sources) + 1
        or not source_state_offsets
        or source_state_offsets[0] != 0
        or source_state_offsets[-1] != len(source_state_indices)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (*source_state_offsets, *source_state_indices)
        )
        or any(left > right for left, right in pairwise(source_state_offsets))
    ):
        raise ValueError(f"{context} source-state remap is invalid")
    for start, stop in pairwise(source_state_offsets):
        if tuple(sorted(source_state_indices[start:stop])) != tuple(
            range(stop - start)
        ):
            raise ValueError(f"{context} source-state remap is not bijective")
    for name in (
        "state_templates",
        "source_templates",
        "direct_executors",
        "parameter_slots",
    ):
        sparse = _mapping(remap.get(name))
        if set(sparse) != {"count", "changes"}:
            raise ValueError(f"{context} {name} fields are invalid")
        count = sparse.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{context} {name} count is invalid")
        changes = tuple(_sequence(sparse.get("changes")))
        mapping = list(range(count))
        previous = -1
        for raw_change in changes:
            change = tuple(_sequence(raw_change))
            if (
                len(change) != 2
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or item < 0
                    or item >= count
                    for item in change
                )
                or change[0] <= previous
                or change[0] == change[1]
            ):
                raise ValueError(f"{context} {name} change is invalid")
            source, target = cast(tuple[int, int], change)
            mapping[source] = target
            previous = source
        if tuple(sorted(mapping)) != tuple(range(count)):
            raise ValueError(f"{context} {name} is not bijective")


def _validate_evaluator_payload_container(manifest: ArtifactManifest) -> None:
    raw = manifest.extensions.get(_EVALUATOR_PAYLOAD_CONTAINER_EXTENSION)
    if raw is None:
        return
    extension = _mapping(raw)
    expected_fields = {
        "kind",
        "schema_version",
        "storage_abi",
        "path",
        "member_count",
        "unpacked_size_bytes",
        "index_sha256",
    }
    if set(extension) != expected_fields:
        raise ValueError("evaluator payload container extension fields are invalid")
    if extension != {
        **extension,
        "kind": _EVALUATOR_PAYLOAD_CONTAINER_KIND,
        "schema_version": _EVALUATOR_PAYLOAD_CONTAINER_SCHEMA_VERSION,
        "storage_abi": _EVALUATOR_PAYLOAD_CONTAINER_STORAGE_ABI,
        "path": _EVALUATOR_PAYLOAD_CONTAINER_PATH,
    }:
        raise ValueError("evaluator payload container extension contract is invalid")
    records = {
        record.path: record
        for record in manifest.payloads
        if record.path == _EVALUATOR_PAYLOAD_CONTAINER_PATH
    }
    if len(records) != 1:
        raise ValueError("evaluator payload container is not a declared payload")
    record = records[_EVALUATOR_PAYLOAD_CONTAINER_PATH]
    if (
        record.role != "evaluator-state"
        or record.media_type != "application/octet-stream"
        or record.process_id is not None
        or record.target != manifest.producer["target"]
    ):
        raise ValueError("evaluator payload container manifest record is invalid")
    loose = sorted(
        payload.path
        for payload in manifest.payloads
        if payload.path != _EVALUATOR_PAYLOAD_CONTAINER_PATH
        and _packed_evaluator_member_kind(payload.path) is not None
    )
    if loose:
        raise ValueError(
            "artifact declares loose packed evaluator payloads: " + ", ".join(loose)
        )
    with PacbinReader.open(
        manifest.root / _EVALUATOR_PAYLOAD_CONTAINER_PATH,
        verify_payloads=False,
    ) as reader:
        expected = _evaluator_payload_container_extension(reader.index)
    if extension != expected:
        raise ValueError(
            "evaluator payload container metadata does not match its index"
        )


def _config_payload(config: GenerationConfig | RunConfig) -> dict[str, object]:
    payload = _plain_mapping(_mapping(config_to_dict(config)))
    if isinstance(config, GenerationConfig):
        return {"schema_version": 1, "generation": payload}
    return payload


def _effective_config_payload(
    requested: Mapping[str, object],
    *,
    disable_api_bundle: bool,
) -> dict[str, object]:
    result = _deep_plain(requested)
    if disable_api_bundle:
        generation = result.get("generation")
        if not isinstance(generation, dict):
            raise ValueError("generation configuration section is missing")
        generation["emit_api_bundle"] = False
    return result


def _bundle_requested(config: GenerationConfig | RunConfig) -> bool:
    generation = config if isinstance(config, GenerationConfig) else config.generation
    return bool(generation.emit_api_bundle)


def _existing_bundle_path(existing: ArtifactManifest | None) -> str | None:
    if existing is None:
        return None
    value = existing.runtime.get("api_bundle_path")
    return None if value is None else str(value)


def _toml_bytes(payload: Mapping[str, object]) -> bytes:
    lines: list[str] = []
    _write_toml_table(lines, (), payload, emit_header=False)
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _write_toml_table(
    lines: list[str],
    path: tuple[str, ...],
    payload: Mapping[str, object],
    *,
    emit_header: bool,
) -> None:
    scalars = [
        (str(key), value)
        for key, value in payload.items()
        if value is not None and not isinstance(value, Mapping)
    ]
    tables = [
        (str(key), _mapping(value))
        for key, value in payload.items()
        if isinstance(value, Mapping)
    ]
    if emit_header:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append("[" + ".".join(_toml_key(part) for part in path) + "]")
    lines.extend(f"{_toml_key(key)} = {_toml_value(value)}" for key, value in scalars)
    for key, table in tables:
        _write_toml_table(lines, (*path, key), table, emit_header=True)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, os.PathLike):
        return json.dumps(os.fspath(value), ensure_ascii=True)
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, Mapping):
        entries = (
            f"{_toml_key(str(key))} = {_toml_value(entry)}"
            for key, entry in value.items()
            if entry is not None
        )
        return "{ " + ", ".join(entries) + " }"
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"configuration value is not TOML serializable: {value!r}")


def _toml_key(value: str) -> str:
    return value if _SAFE_TOML_KEY.fullmatch(value) else json.dumps(value)


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".json"}:
        return "application/json"
    if suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}:
        return "text/x-c++src"
    if suffix == ".symjit":
        return "application/vnd.symjit.application"
    return "application/octet-stream"


def _distribution_version(name: str, fallback: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("artifact metadata value must be an object")
    return {str(key): item for key, item in value.items()}


def _plain_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return _deep_plain(value)


def _deep_plain(value: Mapping[str, object]) -> dict[str, object]:
    def convert(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): convert(entry) for key, entry in item.items()}
        if isinstance(item, Sequence) and not isinstance(item, str | bytes):
            return [convert(entry) for entry in item]
        return item

    return {str(key): convert(item) for key, item in value.items()}


__all__ = [
    "EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY",
    "EAGER_PLAN_V3_ABI",
    "EAGER_PLAN_V3_RUNTIME_CAPABILITY",
    "EAGER_RUNTIME_CONTAINER_KIND",
    "EAGER_RUNTIME_CONTAINER_SCHEMA_VERSION",
    "EAGER_RUNTIME_LAYOUT_ABI",
    "EAGER_RUNTIME_STORAGE_ABI",
    "ON_THE_FLY_PUBLIC_METADATA_KIND",
    "ON_THE_FLY_RUNTIME_CONTAINER_KIND",
    "ON_THE_FLY_RUNTIME_CONTAINER_SCHEMA_VERSION",
    "ON_THE_FLY_RUNTIME_KIND",
    "ON_THE_FLY_RUNTIME_STORAGE_ABI",
    "RECURRENCE_COLOR_RUNTIME_CAPABILITY",
    "RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY",
    "RECURRENCE_DIRECT_BACKEND_ABI",
    "RECURRENCE_DIRECT_TEMPLATE_ABI",
    "RECURRENCE_PLAN_ABI",
    "RECURRENCE_RUNTIME_CONTAINER_KIND",
    "RECURRENCE_RUNTIME_CONTAINER_SCHEMA_VERSION",
    "RECURRENCE_RUNTIME_KIND",
    "RECURRENCE_RUNTIME_LAYOUT_ABI",
    "RECURRENCE_RUNTIME_STORAGE_ABI",
    "ApiBundleHook",
    "ArtifactWriteResult",
    "CompiledExecutionArtifact",
    "CompiledProcessArtifact",
    "EagerPlanV3ProcessArtifact",
    "OnTheFlyProcessArtifact",
    "ProcessArtifact",
    "RecurrenceHelicitySelectorPlanArtifact",
    "RecurrenceProcessArtifact",
    "build_api_validation_points",
    "write_schema_v3_artifact",
]
