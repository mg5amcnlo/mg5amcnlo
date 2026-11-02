# SPDX-License-Identifier: 0BSD
"""Validated in-memory representation of an eager exact-execution plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import cast

from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.artifacts.manifest import ArtifactManifest
from pyamplicol.artifacts.security import confined_path, normalize_relative_path
from pyamplicol.generation.eager_tables import (
    EAGER_KERNEL_ABI,
    EAGER_OUTPUT_FACTOR_NONE,
    MISSING_U32,
    EagerAttachmentRow,
    EagerClosureRow,
    EagerCouplingRow,
    EagerFinalizationRow,
    EagerInvocationRow,
)
from pyamplicol.models.prepared import (
    PreparedKernelPack,
    PreparedKernelRecord,
    PreparedModelBundleError,
)
from pyamplicol.runtime._evaluator_payloads import ExactEvaluatorPayloadResolver
from pyamplicol.runtime.eager_exact._contracts import (
    _PREPARED_CATALOG_ABI,
    _SUPPORTED_PREPARED_BACKENDS,
    _artifact_kernel_loader,
    _complex_pair,
    _component_slots,
    _ComponentSlot,
    _direct_coefficients,
    _ExactAttachmentRow,
    _ExactClosureRow,
    _ExactCouplingRow,
    _ExactFinalizationRow,
    _ExactInvocationRow,
    _integer,
    _joined_payload_path,
    _KernelLoader,
    _LazyExactKernel,
    _load_table,
    _mapping,
    _PayloadIndex,
    _read_json,
    _sequence,
    _validate_execution_header,
)
from pyamplicol.runtime.eager_exact._plan_v3 import (
    EAGER_PLAN_V3_ABI,
    _load_eager_exact_sections_v1,
    _NativeExactSectionsLoader,
    _validate_plan_v3_execution_header,
)


@dataclass(frozen=True, slots=True)
class _ExactStage:
    stage_index: int
    invocations: tuple[_ExactInvocationRow, ...]
    attachments: tuple[_ExactAttachmentRow, ...]
    finalizations: tuple[_ExactFinalizationRow, ...]


@dataclass(frozen=True, slots=True)
class _RuntimeParameterSlots:
    real: int
    imaginary: int | None
    kind: str


@dataclass(frozen=True, slots=True)
class _PreparedParameterProjectionEntry:
    name: str
    prepared_index: int
    runtime_real_index: int
    runtime_imaginary_index: int | None


@dataclass(frozen=True, slots=True)
class _PreparedParameterProjection:
    parameter_count: int
    runtime_parameter_count: int
    entries: tuple[_PreparedParameterProjectionEntry, ...]

    def project(
        self, runtime_parameters: Sequence[Decimal]
    ) -> tuple[tuple[Decimal, Decimal], ...]:
        if len(runtime_parameters) != self.runtime_parameter_count:
            raise ArtifactError(
                f"eager runtime has {len(runtime_parameters)} model parameters, "
                f"expected {self.runtime_parameter_count}"
            )
        zero = Decimal(0)
        result = [(zero, zero) for _ in range(self.parameter_count)]
        for entry in self.entries:
            imaginary = (
                runtime_parameters[entry.runtime_imaginary_index]
                if entry.runtime_imaginary_index is not None
                else zero
            )
            result[entry.prepared_index] = (
                runtime_parameters[entry.runtime_real_index],
                imaginary,
            )
        return tuple(result)

    def entry(self, name: str) -> _PreparedParameterProjectionEntry | None:
        return next((entry for entry in self.entries if entry.name == name), None)


@dataclass(frozen=True, slots=True)
class _DerivedParameterTarget:
    name: str
    prepared_index: int | None
    runtime_real_index: int
    runtime_imaginary_index: int | None


@dataclass(frozen=True, slots=True)
class _ExactModelParameterState:
    runtime: tuple[Decimal, ...]
    prepared: tuple[tuple[Decimal, Decimal], ...]


@dataclass(frozen=True, slots=True)
class _ExactParameterDerivation:
    kernel: _LazyExactKernel
    input_parameter_indices: tuple[int, ...]
    targets: tuple[_DerivedParameterTarget | None, ...]

    def evaluate(
        self,
        runtime_parameters: Sequence[Decimal],
        prepared_parameters: Sequence[tuple[Decimal, Decimal]],
        precision: int,
    ) -> _ExactModelParameterState:
        inputs = tuple(
            prepared_parameters[index] for index in self.input_parameter_indices
        )
        outputs = self.kernel.evaluate(inputs, precision)
        if len(outputs) != len(self.targets):
            raise ArtifactError(
                "exact model-parameter output count does not match its layout"
            )
        runtime = list(runtime_parameters)
        prepared = list(prepared_parameters)
        zero = Decimal(0)
        for output, target in zip(outputs, self.targets, strict=True):
            if target is None:
                continue
            if target.prepared_index is not None:
                prepared[target.prepared_index] = output
            runtime[target.runtime_real_index] = output[0]
            if target.runtime_imaginary_index is not None:
                runtime[target.runtime_imaginary_index] = output[1]
            elif output[1] != zero:
                raise EvaluationError(
                    f"exact derived scalar parameter {target.name!r} has a "
                    "nonzero imaginary component"
                )
        return _ExactModelParameterState(tuple(runtime), tuple(prepared))


@dataclass(slots=True)
class _EagerExactPlan:
    runtime_schema: Mapping[str, object]
    physics_reduction_groups: tuple[Mapping[str, object], ...] | None
    selector_group_ids: tuple[int, ...] | None
    selector_domains: tuple[frozenset[int], ...] | None
    kernels: Mapping[int, _LazyExactKernel]
    value_slots: tuple[_ComponentSlot, ...]
    momentum_slots: tuple[_ComponentSlot, ...]
    current_slots: tuple[_ComponentSlot, ...]
    value_component_count: int
    momentum_component_count: int
    current_component_count: int
    parameter_count: int
    parameter_projection: _PreparedParameterProjection
    parameter_derivation: _ExactParameterDerivation | None
    amplitude_count: int
    couplings: tuple[_ExactCouplingRow, ...]
    stages: tuple[_ExactStage, ...]
    closures: tuple[_ExactClosureRow, ...]

    @classmethod
    def load_for_execution(
        cls,
        *,
        artifact_root: Path,
        process_root: Path,
        process_id: str,
        execution: Mapping[str, object],
        manifest: ArtifactManifest,
        kernel_loader: _KernelLoader | None,
        exact_payloads: ExactEvaluatorPayloadResolver,
        native_sections_loader: _NativeExactSectionsLoader | None = None,
    ) -> _EagerExactPlan:
        if execution.get("eager_plan_abi") == EAGER_PLAN_V3_ABI:
            return cls.load_v3(
                artifact_root=artifact_root,
                process_id=process_id,
                execution=execution,
                manifest=manifest,
                kernel_loader=kernel_loader,
                exact_payloads=exact_payloads,
                native_sections_loader=native_sections_loader,
            )
        del process_root
        raise CompatibilityError(
            "eager exact execution requires a compact plan-v3 artifact; "
            f"found eager plan ABI {execution.get('eager_plan_abi')!r}. "
            "Regenerate the artifact with the current `pyamplicol generate`."
        )

    @classmethod
    def load_v3(
        cls,
        *,
        artifact_root: Path,
        process_id: str,
        execution: Mapping[str, object],
        manifest: ArtifactManifest,
        kernel_loader: _KernelLoader | None,
        exact_payloads: ExactEvaluatorPayloadResolver,
        native_sections_loader: _NativeExactSectionsLoader | None,
    ) -> _EagerExactPlan:
        _validate_plan_v3_execution_header(execution)
        sections = _load_eager_exact_sections_v1(
            artifact_root,
            process_id,
            loader=native_sections_loader,
        )
        (
            pack,
            payload_root,
            effective_kernel_loader,
        ) = _load_exact_kernel_pack(
            artifact_root=artifact_root,
            execution=execution,
            manifest=manifest,
            kernel_loader=kernel_loader,
            exact_payloads=exact_payloads,
        )
        return cls._from_exact_sections(
            runtime_schema=sections.exact_schema,
            physics_reduction_groups=sections.reduction_groups,
            selector_group_ids=sections.selector_group_ids,
            selector_domains=sections.selector_domains,
            pack=pack,
            payload_root=payload_root,
            kernel_loader=effective_kernel_loader,
            couplings=sections.couplings,
            stages=tuple(
                _ExactStage(
                    stage.stage_index,
                    stage.invocations,
                    stage.attachments,
                    stage.finalizations,
                )
                for stage in sections.stages
            ),
            closures=sections.closures,
        )

    @classmethod
    def _from_exact_sections(
        cls,
        *,
        runtime_schema: Mapping[str, object],
        physics_reduction_groups: tuple[Mapping[str, object], ...] | None,
        selector_group_ids: tuple[int, ...] | None,
        selector_domains: tuple[frozenset[int], ...] | None,
        pack: PreparedKernelPack,
        payload_root: Path,
        kernel_loader: _KernelLoader,
        couplings: tuple[_ExactCouplingRow, ...],
        stages: tuple[_ExactStage, ...],
        closures: tuple[_ExactClosureRow, ...],
    ) -> _EagerExactPlan:
        layout = _mapping(runtime_schema.get("parameter_layout"), "parameter_layout")
        value_count = _integer(
            layout.get("value_component_count"), "value_component_count"
        )
        momentum_count = _integer(
            layout.get("momentum_parameter_count"), "momentum_parameter_count"
        )
        parameter_count = _integer(
            layout.get("model_parameter_count"), "model_parameter_count"
        )
        current_storage = _mapping(
            runtime_schema.get("current_storage"), "current_storage"
        )
        value_storage = _mapping(runtime_schema.get("value_storage"), "value_storage")
        current_count = _integer(
            current_storage.get("component_count"), "current component count"
        )
        values = _component_slots(
            value_storage.get("value_slots"),
            id_field="value_slot_id",
            start_field="component_start",
            stop_field="component_stop",
            dimension_field="dimension",
            component_count=value_count,
            context="value slots",
        )
        momenta = _component_slots(
            runtime_schema.get("momentum_slots"),
            id_field="momentum_slot_id",
            start_field="component_start",
            stop_field="component_stop",
            dimension_field=None,
            component_count=momentum_count,
            context="momentum slots",
        )
        currents = _component_slots(
            current_storage.get("current_slots"),
            id_field="current_id",
            start_field="component_start",
            stop_field="component_stop",
            dimension_field="dimension",
            component_count=current_count,
            context="current slots",
        )
        active_kernel_ids = _active_process_kernel_ids(stages, closures)
        include_parameter_kernel = _runtime_has_derived_parameters(runtime_schema)
        process_kernels = _process_prepared_kernels(
            pack.kernels,
            active_kernel_ids,
            include_parameter_kernel=include_parameter_kernel,
        )
        kernels = {
            kernel.kernel_id: _LazyExactKernel(kernel, payload_root, kernel_loader)
            for kernel in process_kernels
            if kernel.contract_kind != "model-parameter"
        }
        parameter_projection = _prepared_parameter_projection(
            pack.kernels,
            runtime_schema,
            parameter_count,
            active_kernel_ids=active_kernel_ids,
            include_parameter_kernel=include_parameter_kernel,
        )
        parameter_derivation = _exact_parameter_derivation(
            process_kernels,
            runtime_schema,
            parameter_projection,
            payload_root,
            kernel_loader,
        )
        amplitude_stage = _mapping(
            runtime_schema.get("amplitude_stage"), "amplitude_stage"
        )
        amplitude_count = _integer(
            amplitude_stage.get("output_count"),
            "amplitude output count",
            minimum=1,
        )
        result = cls(
            runtime_schema=runtime_schema,
            physics_reduction_groups=physics_reduction_groups,
            selector_group_ids=selector_group_ids,
            selector_domains=selector_domains,
            kernels=kernels,
            value_slots=values,
            momentum_slots=momenta,
            current_slots=currents,
            value_component_count=value_count,
            momentum_component_count=momentum_count,
            current_component_count=current_count,
            parameter_count=parameter_count,
            parameter_projection=parameter_projection,
            parameter_derivation=parameter_derivation,
            amplitude_count=amplitude_count,
            couplings=couplings,
            stages=stages,
            closures=closures,
        )
        result._validate()
        return result

    @classmethod
    def load(
        cls,
        *,
        artifact_root: Path,
        process_root: Path,
        process_id: str,
        execution: Mapping[str, object],
        manifest: ArtifactManifest,
        kernel_loader: _KernelLoader | None,
        exact_payloads: ExactEvaluatorPayloadResolver,
    ) -> _EagerExactPlan:
        _validate_execution_header(execution)
        payloads = _PayloadIndex.from_manifest(manifest)
        kernel_reference = _mapping(execution.get("kernel_pack"), "kernel_pack")
        pack_path = cast(str, kernel_reference.get("manifest_path"))
        if not isinstance(pack_path, str):
            raise ArtifactError("kernel_pack.manifest_path must be a string")
        payload_root_name = kernel_reference.get("payload_root")
        if not isinstance(payload_root_name, str):
            raise ArtifactError("kernel_pack.payload_root must be a string")
        pack_path = normalize_relative_path(pack_path)
        payload_root_name = normalize_relative_path(payload_root_name)
        payloads.require(pack_path, role="evaluator-manifest", process_id=None)
        pack_payload = _read_json(
            confined_path(artifact_root, pack_path), "eager kernel pack"
        )
        raw_backend = pack_payload.get("backend")
        if raw_backend not in _SUPPORTED_PREPARED_BACKENDS:
            raise CompatibilityError(
                f"unsupported prepared eager backend {raw_backend!r}"
            )
        mutable_pack = dict(pack_payload)
        kernel_abi = mutable_pack.pop("eager_kernel_abi", None)
        if kernel_abi != EAGER_KERNEL_ABI:
            raise CompatibilityError(f"unsupported eager kernel ABI {kernel_abi!r}")
        try:
            pack = PreparedKernelPack.from_dict(mutable_pack)
        except PreparedModelBundleError as exc:
            raise ArtifactError(f"eager kernel pack is malformed: {exc}") from exc
        if pack.resolver_manifest.get("abi") != _PREPARED_CATALOG_ABI:
            raise CompatibilityError("unsupported prepared eager kernel catalog ABI")
        payload_root = artifact_root / payload_root_name
        if payload_root.exists() and (
            not payload_root.is_dir() or payload_root.is_symlink()
        ):
            raise ArtifactError("eager kernel payload root is invalid")
        for kernel in pack.kernels:
            exact_path = _joined_payload_path(
                payload_root_name, kernel.exact_evaluator_state_path
            )
            exact_payloads.require_exact_state(exact_path, process_id=None)

        effective_kernel_loader = kernel_loader or _artifact_kernel_loader(
            exact_payloads,
            payload_root_name,
        )

        runtime_schema = _mapping(execution.get("runtime_schema"), "runtime_schema")
        layout = _mapping(runtime_schema.get("parameter_layout"), "parameter_layout")
        value_count = _integer(
            layout.get("value_component_count"), "value_component_count"
        )
        momentum_count = _integer(
            layout.get("momentum_parameter_count"), "momentum_parameter_count"
        )
        parameter_count = _integer(
            layout.get("model_parameter_count"), "model_parameter_count"
        )
        current_storage = _mapping(
            runtime_schema.get("current_storage"), "current_storage"
        )
        value_storage = _mapping(runtime_schema.get("value_storage"), "value_storage")
        current_count = _integer(
            current_storage.get("component_count"), "current component_count"
        )
        values = _component_slots(
            value_storage.get("value_slots"),
            id_field="value_slot_id",
            start_field="component_start",
            stop_field="component_stop",
            dimension_field="dimension",
            component_count=value_count,
            context="value slots",
        )
        momenta = _component_slots(
            runtime_schema.get("momentum_slots"),
            id_field="momentum_slot_id",
            start_field="component_start",
            stop_field="component_stop",
            dimension_field=None,
            component_count=momentum_count,
            context="momentum slots",
        )
        currents = _component_slots(
            current_storage.get("current_slots"),
            id_field="current_id",
            start_field="component_start",
            stop_field="component_stop",
            dimension_field="dimension",
            component_count=current_count,
            context="current slots",
        )
        plan_record = _mapping(execution.get("plan"), "plan")
        process_prefix = f"processes/{process_id}"
        raw_couplings = _load_table(
            process_root,
            process_prefix,
            _mapping(plan_record.get("couplings"), "plan.couplings"),
            EagerCouplingRow,
            payloads,
            process_id,
            "couplings",
        )
        couplings = tuple(
            _ExactCouplingRow(
                row.real_parameter_id,
                row.imag_parameter_id,
                _complex_pair(
                    row.constant_real,
                    row.constant_imag,
                    f"eager coupling {index} constant",
                ),
            )
            for index, row in enumerate(raw_couplings)
        )
        raw_runtime_stages = _sequence(runtime_schema.get("stages"), "runtime stages")
        raw_stages = _sequence(plan_record.get("stages"), "plan.stages")
        if len(raw_stages) != len(raw_runtime_stages):
            raise ArtifactError("eager plan stage count does not match runtime schema")
        stages: list[_ExactStage] = []
        previous_stage = -1
        for index, (raw_stage, raw_runtime_stage) in enumerate(
            zip(raw_stages, raw_runtime_stages, strict=True)
        ):
            stage = _mapping(raw_stage, f"plan.stages[{index}]")
            runtime_stage = _mapping(
                raw_runtime_stage, f"runtime_schema.stages[{index}]"
            )
            stage_index = _integer(stage.get("stage_index"), "stage_index")
            if stage_index <= previous_stage:
                raise ArtifactError("eager stage indices must be strictly increasing")
            previous_stage = stage_index
            if stage_index != _integer(runtime_stage.get("stage_index"), "stage_index"):
                raise ArtifactError("eager plan stage does not match runtime schema")
            raw_invocations = _load_table(
                process_root,
                process_prefix,
                _mapping(stage.get("invocations"), "stage.invocations"),
                EagerInvocationRow,
                payloads,
                process_id,
                f"stage {stage_index} invocations",
            )
            raw_attachments = _load_table(
                process_root,
                process_prefix,
                _mapping(stage.get("attachments"), "stage.attachments"),
                EagerAttachmentRow,
                payloads,
                process_id,
                f"stage {stage_index} attachments",
            )
            raw_finalizations = _load_table(
                process_root,
                process_prefix,
                _mapping(stage.get("finalizations"), "stage.finalizations"),
                EagerFinalizationRow,
                payloads,
                process_id,
                f"stage {stage_index} finalizations",
            )
            stages.append(
                _ExactStage(
                    stage_index,
                    tuple(
                        _ExactInvocationRow(
                            row.kernel_id,
                            row.left_value_slot_id,
                            row.right_value_slot_id,
                            row.left_momentum_slot_id,
                            row.right_momentum_slot_id,
                            row.coupling_slot_id,
                            row.output_factor_source,
                            row.attachment_start,
                            row.attachment_count,
                        )
                        for row in raw_invocations
                    ),
                    tuple(
                        _ExactAttachmentRow(
                            row.result_current_id,
                            _complex_pair(
                                row.factor_real,
                                row.factor_imag,
                                f"stage {stage_index} attachment factor",
                            ),
                        )
                        for row in raw_attachments
                    ),
                    tuple(
                        _ExactFinalizationRow(
                            row.kernel_id,
                            row.current_id,
                            row.unpropagated_value_slot_id,
                            row.propagated_value_slot_id,
                            row.momentum_slot_id,
                        )
                        for row in raw_finalizations
                    ),
                )
            )
        raw_closures = _load_table(
            process_root,
            process_prefix,
            _mapping(plan_record.get("closures"), "plan.closures"),
            EagerClosureRow,
            payloads,
            process_id,
            "closures",
        )
        amplitude_stage = _mapping(
            runtime_schema.get("amplitude_stage"), "amplitude_stage"
        )
        amplitude_count = _integer(
            amplitude_stage.get("output_count"), "amplitude output count", minimum=1
        )
        roots = _sequence(amplitude_stage.get("roots"), "amplitude roots")
        if len(roots) != len(raw_closures):
            raise ArtifactError("eager closure rows do not match amplitude roots")
        closures = tuple(
            _ExactClosureRow(
                kernel_id=row.kernel_id,
                left_value_slot_id=row.left_value_slot_id,
                right_value_slot_id=row.right_value_slot_id,
                amplitude_index=row.amplitude_index,
                coupling_slot_id=row.coupling_slot_id,
                output_factor_source=row.output_factor_source,
                factor=_complex_pair(
                    row.factor_real,
                    row.factor_imag,
                    f"eager closure {index} factor",
                ),
                direct_coefficients=(
                    _direct_coefficients(
                        _mapping(root, f"amplitude root {index}"),
                        index,
                    )
                    if row.kernel_id == MISSING_U32
                    else None
                ),
            )
            for index, (row, root) in enumerate(zip(raw_closures, roots, strict=True))
        )
        stage_records = tuple(stages)
        active_kernel_ids = _active_process_kernel_ids(stage_records, closures)
        include_parameter_kernel = _runtime_has_derived_parameters(runtime_schema)
        process_kernels = _process_prepared_kernels(
            pack.kernels,
            active_kernel_ids,
            include_parameter_kernel=include_parameter_kernel,
        )
        kernel_map = {
            kernel.kernel_id: _LazyExactKernel(
                kernel,
                payload_root,
                effective_kernel_loader,
            )
            for kernel in process_kernels
            if kernel.contract_kind != "model-parameter"
        }
        parameter_projection = _prepared_parameter_projection(
            pack.kernels,
            runtime_schema,
            parameter_count,
            active_kernel_ids=active_kernel_ids,
            include_parameter_kernel=include_parameter_kernel,
        )
        parameter_derivation = _exact_parameter_derivation(
            process_kernels,
            runtime_schema,
            parameter_projection,
            payload_root,
            effective_kernel_loader,
        )
        result = cls(
            runtime_schema=runtime_schema,
            physics_reduction_groups=None,
            selector_group_ids=None,
            selector_domains=None,
            kernels=kernel_map,
            value_slots=values,
            momentum_slots=momenta,
            current_slots=currents,
            value_component_count=value_count,
            momentum_component_count=momentum_count,
            current_component_count=current_count,
            parameter_count=parameter_count,
            parameter_projection=parameter_projection,
            parameter_derivation=parameter_derivation,
            amplitude_count=amplitude_count,
            couplings=couplings,
            stages=stage_records,
            closures=closures,
        )
        result._validate()
        return result

    def project_model_parameters(
        self, runtime_parameters: Sequence[Decimal]
    ) -> tuple[tuple[Decimal, Decimal], ...]:
        return self.parameter_projection.project(runtime_parameters)

    def resolve_model_parameters(
        self,
        runtime_parameters: Sequence[Decimal],
        precision: int,
    ) -> _ExactModelParameterState:
        runtime = tuple(runtime_parameters)
        prepared = self.parameter_projection.project(runtime)
        if self.parameter_derivation is None:
            return _ExactModelParameterState(runtime, prepared)
        return self.parameter_derivation.evaluate(runtime, prepared, precision)

    def _validate(self) -> None:
        if (self.selector_group_ids is None) != (self.selector_domains is None):
            raise ArtifactError(
                "eager exact selector groups and domains must be provided together"
            )
        selector_groups = (
            set() if self.selector_group_ids is None else set(self.selector_group_ids)
        )
        if self.selector_group_ids is not None:
            if not self.selector_group_ids:
                raise ArtifactError("eager exact selector group catalog is empty")
            if len(selector_groups) != len(self.selector_group_ids):
                raise ArtifactError("eager exact selector group IDs must be unique")
            assert self.selector_domains is not None
            if not self.selector_domains:
                raise ArtifactError("eager exact selector domain catalog is empty")
            for index, members in enumerate(self.selector_domains):
                unknown = members - selector_groups
                if unknown:
                    raise ArtifactError(
                        f"eager exact selector domain {index} references unknown "
                        f"group {min(unknown)}"
                    )

        def validate_domain(domain_id: int | None, context: str) -> None:
            if self.selector_domains is None:
                if domain_id is not None:
                    raise ArtifactError(f"{context} unexpectedly has a selector domain")
                return
            if domain_id is None or domain_id >= len(self.selector_domains):
                raise ArtifactError(f"{context} selector domain is out of range")

        for index, coupling in enumerate(self.couplings):
            for component, parameter_id in (
                ("real", coupling.real_parameter_id),
                ("imaginary", coupling.imag_parameter_id),
            ):
                if parameter_id != MISSING_U32 and parameter_id >= self.parameter_count:
                    raise ArtifactError(
                        f"eager coupling {index} {component} parameter is out of range"
                    )
        finalized: set[int] = set()
        stored_values: set[int] = set()
        for stage in self.stages:
            cursor = 0
            attached: set[int] = set()
            for index, invocation in enumerate(stage.invocations):
                validate_domain(
                    invocation.selector_domain_id,
                    f"eager invocation {index}",
                )
                kernel = self._require_kernel(
                    invocation.kernel_id, "vertex", f"invocation {index}"
                )
                left = self._slot(
                    self.value_slots,
                    invocation.left_value_slot_id,
                    "invocation left value",
                )
                right = self._slot(
                    self.value_slots,
                    invocation.right_value_slot_id,
                    "invocation right value",
                )
                left_momentum = self._slot(
                    self.momentum_slots,
                    invocation.left_momentum_slot_id,
                    "invocation left momentum",
                )
                right_momentum = self._slot(
                    self.momentum_slots,
                    invocation.right_momentum_slot_id,
                    "invocation right momentum",
                )
                if invocation.coupling_slot_id >= len(self.couplings):
                    raise ArtifactError("eager invocation coupling is out of range")
                if (
                    invocation.attachment_count == 0
                    or invocation.attachment_start != cursor
                ):
                    raise ArtifactError(
                        f"eager stage {stage.stage_index} has invalid attachment ranges"
                    )
                stop = cursor + invocation.attachment_count
                if stop > len(stage.attachments):
                    raise ArtifactError(
                        "eager invocation attachment range is out of bounds"
                    )
                self._validate_kernel_inputs(
                    kernel.record,
                    first_current=left.width,
                    second_current=right.width,
                    first_momentum=left_momentum.width,
                    second_momentum=right_momentum.width,
                    has_coupling=True,
                    context=f"invocation {index}",
                )
                for attachment in stage.attachments[cursor:stop]:
                    validate_domain(
                        attachment.selector_domain_id,
                        "eager attachment",
                    )
                    current = self._slot(
                        self.current_slots,
                        attachment.result_current_id,
                        "attachment result current",
                    )
                    if current.width != kernel.record.output_arity:
                        raise ArtifactError(
                            "eager invocation output width does not match "
                            "attached current"
                        )
                    attached.add(attachment.result_current_id)
                cursor = stop
            if cursor != len(stage.attachments):
                raise ArtifactError("eager attachment table is not fully referenced")
            stage_finalized: set[int] = set()
            for index, finalization in enumerate(stage.finalizations):
                validate_domain(
                    finalization.unpropagated_selector_domain_id,
                    f"eager finalization {index} unpropagated",
                )
                validate_domain(
                    finalization.propagated_selector_domain_id,
                    f"eager finalization {index} propagated",
                )
                if (
                    finalization.current_id in stage_finalized
                    or finalization.current_id in finalized
                ):
                    raise ArtifactError(
                        f"eager current {finalization.current_id} is finalized "
                        "more than once"
                    )
                stage_finalized.add(finalization.current_id)
                finalized.add(finalization.current_id)
                current = self._slot(
                    self.current_slots,
                    finalization.current_id,
                    "finalization current",
                )
                outputs = []
                for name, slot_id in (
                    ("unpropagated", finalization.unpropagated_value_slot_id),
                    ("propagated", finalization.propagated_value_slot_id),
                ):
                    if slot_id == MISSING_U32:
                        continue
                    if slot_id in stored_values:
                        raise ArtifactError(
                            f"eager value slot {slot_id} is finalized more than once"
                        )
                    stored_values.add(slot_id)
                    output = self._slot(self.value_slots, slot_id, f"{name} value")
                    if output.width != current.width:
                        raise ArtifactError("finalization output/current widths differ")
                    outputs.append((name, output))
                if not outputs:
                    raise ArtifactError("eager finalization stores no value")
                momentum = self._slot(
                    self.momentum_slots,
                    finalization.momentum_slot_id,
                    "finalization momentum",
                )
                if finalization.kernel_id == MISSING_U32:
                    if finalization.propagated_value_slot_id != MISSING_U32:
                        raise ArtifactError(
                            "eager propagated current has no finalization kernel"
                        )
                else:
                    kernel = self._require_kernel(
                        finalization.kernel_id,
                        "propagator",
                        f"finalization {index}",
                    )
                    if finalization.propagated_value_slot_id == MISSING_U32:
                        raise ArtifactError(
                            "eager finalization kernel has no propagated output"
                        )
                    if kernel.record.output_arity != current.width:
                        raise ArtifactError(
                            "eager finalization kernel/current widths differ"
                        )
                    self._validate_kernel_inputs(
                        kernel.record,
                        first_current=current.width,
                        second_current=0,
                        first_momentum=momentum.width,
                        second_momentum=0,
                        has_coupling=False,
                        context=f"finalization {index}",
                    )
            if not attached.issubset(stage_finalized):
                missing = min(attached - stage_finalized)
                raise ArtifactError(
                    f"eager stage {stage.stage_index} does not finalize "
                    f"current {missing}"
                )

        amplitude_stage = _mapping(
            self.runtime_schema.get("amplitude_stage"), "amplitude_stage"
        )
        roots = _sequence(amplitude_stage.get("roots"), "amplitude roots")
        if len(roots) != len(self.closures):
            raise ArtifactError("eager closure rows do not match amplitude roots")
        for index, (closure, raw_root) in enumerate(
            zip(self.closures, roots, strict=True)
        ):
            validate_domain(closure.selector_domain_id, f"eager closure {index}")
            if self.selector_domains is not None:
                if closure.coherent_group_id not in selector_groups:
                    raise ArtifactError(
                        f"eager closure {index} coherent group is unknown"
                    )
                assert closure.selector_domain_id is not None
                if (
                    closure.coherent_group_id
                    not in self.selector_domains[closure.selector_domain_id]
                ):
                    raise ArtifactError(
                        f"eager closure {index} domain excludes its coherent group"
                    )
            root = _mapping(raw_root, f"amplitude root {index}")
            left = self._slot(
                self.value_slots, closure.left_value_slot_id, "closure left value"
            )
            right = self._slot(
                self.value_slots, closure.right_value_slot_id, "closure right value"
            )
            if closure.amplitude_index >= self.amplitude_count:
                raise ArtifactError("eager closure amplitude index is out of range")
            if closure.kernel_id == MISSING_U32:
                if closure.coupling_slot_id != MISSING_U32:
                    raise ArtifactError("direct eager closure references a coupling")
                if closure.output_factor_source != EAGER_OUTPUT_FACTOR_NONE:
                    raise ArtifactError(
                        "direct eager closure has a dynamic output factor"
                    )
                if root.get("kind") != "direct-contraction":
                    raise ArtifactError(
                        "direct eager closure lacks contraction metadata"
                    )
                coefficients = closure.direct_coefficients
                if not coefficients:
                    raise ArtifactError(
                        "direct eager closure has no exact contraction coefficients"
                    )
                if left.width != right.width or left.width != len(coefficients):
                    raise ArtifactError("direct eager closure component widths differ")
            else:
                if closure.direct_coefficients is not None:
                    raise ArtifactError(
                        "kernel eager closure carries direct coefficients"
                    )
                if root.get("kind") == "direct-contraction":
                    raise ArtifactError("kernel eager closure has direct metadata")
                if closure.coupling_slot_id >= len(self.couplings):
                    raise ArtifactError("eager closure coupling is out of range")
                kernel = self._require_kernel(
                    closure.kernel_id, "closure", f"closure {index}"
                )
                if kernel.record.output_arity != 1:
                    raise ArtifactError(
                        "eager closure kernels must return one component"
                    )
                self._validate_kernel_inputs(
                    kernel.record,
                    first_current=left.width,
                    second_current=right.width,
                    first_momentum=0,
                    second_momentum=0,
                    has_coupling=True,
                    context=f"closure {index}",
                )

    def _require_kernel(
        self, kernel_id: int, kind: str, context: str
    ) -> _LazyExactKernel:
        kernel = self.kernels.get(kernel_id)
        if kernel is None:
            raise ArtifactError(
                f"{context} references missing eager kernel {kernel_id}"
            )
        if kernel.record.contract_kind != kind:
            raise ArtifactError(
                f"{context} requires a {kind} kernel, but {kernel_id} is "
                f"{kernel.record.contract_kind}"
            )
        return kernel

    @staticmethod
    def _slot(
        slots: Sequence[_ComponentSlot],
        slot_id: int,
        context: str,
    ) -> _ComponentSlot:
        if slot_id < 0 or slot_id >= len(slots):
            raise ArtifactError(f"{context} references unknown slot {slot_id}")
        return slots[slot_id]

    def _validate_kernel_inputs(
        self,
        record: PreparedKernelRecord,
        *,
        first_current: int,
        second_current: int,
        first_momentum: int,
        second_momentum: int,
        has_coupling: bool,
        context: str,
    ) -> None:
        seen: set[tuple[str, int]] = set()
        for input_index, contract in enumerate(record.input_contracts):
            role = str(contract["role"])
            component = _integer(
                contract["component"],
                f"kernel {record.kernel_id} input component",
            )
            if role == "left-current":
                allowed, bound, descriptor = True, first_current, (role, component)
            elif role == "right-current":
                allowed, bound, descriptor = (
                    record.contract_kind != "propagator",
                    second_current,
                    (role, component),
                )
            elif role == "current":
                allowed, bound, descriptor = (
                    record.contract_kind == "propagator",
                    first_current,
                    ("left-current", component),
                )
            elif role == "left-momentum":
                allowed, bound, descriptor = (
                    record.contract_kind != "closure",
                    first_momentum,
                    (role, component),
                )
            elif role == "right-momentum":
                allowed, bound, descriptor = (
                    record.contract_kind == "vertex",
                    second_momentum,
                    (role, component),
                )
            elif role == "momentum":
                allowed, bound, descriptor = (
                    record.contract_kind == "propagator",
                    first_momentum,
                    ("left-momentum", component),
                )
            elif role in {"coupling-real", "coupling-imag"}:
                allowed, bound, descriptor = has_coupling, 1, (role, 0)
            elif role == "model-parameter":
                parameter_id = contract.get("model_parameter_index")
                if isinstance(parameter_id, bool) or not isinstance(parameter_id, int):
                    raise ArtifactError(
                        f"kernel {record.kernel_id} model-parameter input lacks "
                        "an index"
                    )
                allowed, bound, component, descriptor = (
                    parameter_id < self.parameter_projection.parameter_count,
                    self.parameter_projection.parameter_count,
                    parameter_id,
                    (role, parameter_id),
                )
            else:
                raise ArtifactError(
                    f"kernel {record.kernel_id} has unsupported input role {role!r}"
                )
            if not allowed or component >= bound:
                raise ArtifactError(
                    f"{context} kernel {record.kernel_id} input {input_index} "
                    "is out of range"
                )
            if descriptor in seen:
                raise ArtifactError(
                    f"kernel {record.kernel_id} repeats eager input {descriptor!r}"
                )
            seen.add(descriptor)


def _load_exact_kernel_pack(
    *,
    artifact_root: Path,
    execution: Mapping[str, object],
    manifest: ArtifactManifest,
    kernel_loader: _KernelLoader | None,
    exact_payloads: ExactEvaluatorPayloadResolver,
) -> tuple[PreparedKernelPack, Path, _KernelLoader]:
    payloads = _PayloadIndex.from_manifest(manifest)
    kernel_reference = _mapping(execution.get("kernel_pack"), "kernel_pack")
    pack_path = kernel_reference.get("manifest_path")
    if not isinstance(pack_path, str):
        raise ArtifactError("kernel_pack.manifest_path must be a string")
    payload_root_name = kernel_reference.get("payload_root")
    if not isinstance(payload_root_name, str):
        raise ArtifactError("kernel_pack.payload_root must be a string")
    pack_path = normalize_relative_path(pack_path)
    payload_root_name = normalize_relative_path(payload_root_name)
    payloads.require(pack_path, role="evaluator-manifest", process_id=None)
    pack_payload = _read_json(
        confined_path(artifact_root, pack_path), "eager kernel pack"
    )
    raw_backend = pack_payload.get("backend")
    if raw_backend not in _SUPPORTED_PREPARED_BACKENDS:
        raise CompatibilityError(f"unsupported prepared eager backend {raw_backend!r}")
    mutable_pack = dict(pack_payload)
    kernel_abi = mutable_pack.pop("eager_kernel_abi", None)
    if kernel_abi != EAGER_KERNEL_ABI:
        raise CompatibilityError(f"unsupported eager kernel ABI {kernel_abi!r}")
    try:
        pack = PreparedKernelPack.from_dict(mutable_pack)
    except PreparedModelBundleError as exc:
        raise ArtifactError(f"eager kernel pack is malformed: {exc}") from exc
    if pack.resolver_manifest.get("abi") != _PREPARED_CATALOG_ABI:
        raise CompatibilityError("unsupported prepared eager kernel catalog ABI")
    payload_root = artifact_root / payload_root_name
    if payload_root.exists() and (
        not payload_root.is_dir() or payload_root.is_symlink()
    ):
        raise ArtifactError("eager kernel payload root is invalid")
    for kernel in pack.kernels:
        exact_path = _joined_payload_path(
            payload_root_name,
            kernel.exact_evaluator_state_path,
        )
        exact_payloads.require_exact_state(exact_path, process_id=None)
    effective_loader = kernel_loader or _artifact_kernel_loader(
        exact_payloads,
        payload_root_name,
    )
    return pack, payload_root, effective_loader


def _prepared_parameter_projection(
    kernels: Sequence[PreparedKernelRecord],
    runtime_schema: Mapping[str, object],
    runtime_parameter_count: int,
    *,
    active_kernel_ids: frozenset[int] | None = None,
    include_parameter_kernel: bool = False,
) -> _PreparedParameterProjection:
    runtime_slots = _runtime_parameter_slots(
        runtime_schema,
        runtime_parameter_count,
    )
    by_name: dict[str, int] = {}
    by_index: dict[int, str] = {}
    active_names: set[str] = set()
    for kernel in kernels:
        active = (
            active_kernel_ids is None
            or kernel.kernel_id in active_kernel_ids
            or (include_parameter_kernel and kernel.contract_kind == "model-parameter")
        )
        for contract in kernel.input_contracts:
            if contract.get("role") != "model-parameter":
                continue
            name = contract.get("model_parameter_name")
            if not isinstance(name, str) or not name:
                raise ArtifactError(
                    "prepared model-parameter input lacks its logical name"
                )
            index = contract.get("model_parameter_index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ArtifactError(
                    "prepared model-parameter input lacks its stable index"
                )
            previous_index = by_name.setdefault(name, index)
            if previous_index != index:
                raise ArtifactError(
                    f"prepared parameter {name!r} has conflicting stable indices"
                )
            previous_name = by_index.setdefault(index, name)
            if previous_name != name:
                raise ArtifactError(
                    f"prepared parameter index {index} names multiple parameters"
                )
            if active:
                active_names.add(name)

    entries = []
    active_parameters = sorted(
        ((name, by_name[name]) for name in active_names),
        key=lambda item: item[1],
    )
    for name, prepared_index in active_parameters:
        slots = runtime_slots.get(name)
        if slots is None:
            raise ArtifactError(
                f"prepared parameter {name!r} is absent from the process runtime schema"
            )
        entries.append(
            _PreparedParameterProjectionEntry(
                name=name,
                prepared_index=prepared_index,
                runtime_real_index=slots.real,
                runtime_imaginary_index=slots.imaginary,
            )
        )
    parameter_count = max((index for _, index in active_parameters), default=-1) + 1
    return _PreparedParameterProjection(
        parameter_count=parameter_count,
        runtime_parameter_count=runtime_parameter_count,
        entries=tuple(entries),
    )


def _active_process_kernel_ids(
    stages: Sequence[_ExactStage],
    closures: Sequence[_ExactClosureRow],
) -> frozenset[int]:
    kernel_ids = {
        invocation.kernel_id for stage in stages for invocation in stage.invocations
    }
    kernel_ids.update(
        finalization.kernel_id
        for stage in stages
        for finalization in stage.finalizations
        if finalization.kernel_id != MISSING_U32
    )
    kernel_ids.update(
        closure.kernel_id for closure in closures if closure.kernel_id != MISSING_U32
    )
    return frozenset(kernel_ids)


def _process_prepared_kernels(
    kernels: Sequence[PreparedKernelRecord],
    active_kernel_ids: frozenset[int],
    *,
    include_parameter_kernel: bool,
) -> tuple[PreparedKernelRecord, ...]:
    if sum(kernel.contract_kind == "model-parameter" for kernel in kernels) > 1:
        raise ArtifactError(
            "eager kernel pack declares multiple model-parameter kernels"
        )
    process_kernels = tuple(
        kernel
        for kernel in kernels
        if (include_parameter_kernel and kernel.contract_kind == "model-parameter")
        or kernel.kernel_id in active_kernel_ids
    )
    available_ids = {
        kernel.kernel_id
        for kernel in process_kernels
        if kernel.contract_kind != "model-parameter"
    }
    missing_ids = active_kernel_ids - available_ids
    if missing_ids:
        raise ArtifactError(
            f"eager process references missing prepared kernel {min(missing_ids)}"
        )
    return process_kernels


def _runtime_has_derived_parameters(
    runtime_schema: Mapping[str, object],
) -> bool:
    return any(
        _mapping(record, f"model parameters[{index}]").get("kind")
        == "derived_parameter_component"
        for index, record in enumerate(
            _sequence(runtime_schema.get("model_parameters"), "model parameters")
        )
    )


def _runtime_parameter_slots(
    runtime_schema: Mapping[str, object],
    parameter_count: int,
) -> Mapping[str, _RuntimeParameterSlots]:
    direct: dict[str, _RuntimeParameterSlots] = {}
    complex_components: dict[str, list[int | None]] = {}
    complex_kinds: dict[str, str] = {}
    seen_indices: set[int] = set()
    records = _sequence(runtime_schema.get("model_parameters"), "model parameters")
    for record_index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"model parameters[{record_index}]")
        parameter_index = _integer(
            record.get("parameter_index"),
            f"model parameters[{record_index}].parameter_index",
        )
        if parameter_index >= parameter_count or parameter_index in seen_indices:
            raise ArtifactError("runtime model-parameter indices are invalid")
        seen_indices.add(parameter_index)
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ArtifactError("runtime model parameter has no name")
        runtime_name = record.get("runtime_name")
        kind = record.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ArtifactError("runtime model parameter has no kind")
        if runtime_name is None:
            if name in direct or name in complex_components:
                raise ArtifactError(f"runtime model parameter {name!r} is duplicated")
            direct[name] = _RuntimeParameterSlots(parameter_index, None, kind)
            continue
        if not isinstance(runtime_name, str) or not runtime_name:
            raise ArtifactError("complex runtime model parameter has no logical name")
        component = record.get("complex_component")
        if component not in {"real", "imag"}:
            raise ArtifactError(
                f"runtime model parameter {runtime_name!r} has invalid component "
                f"{component!r}"
            )
        slots = complex_components.setdefault(runtime_name, [None, None])
        previous_kind = complex_kinds.setdefault(runtime_name, kind)
        if previous_kind != kind:
            raise ArtifactError(
                f"runtime model parameter {runtime_name!r} has mixed component kinds"
            )
        component_index = 0 if component == "real" else 1
        if slots[component_index] is not None:
            raise ArtifactError(
                f"runtime model parameter {runtime_name!r} repeats a complex component"
            )
        slots[component_index] = parameter_index

    if seen_indices != set(range(parameter_count)):
        raise ArtifactError(
            "runtime model-parameter indices must be contiguous from zero"
        )
    for name, (real, imaginary) in complex_components.items():
        if real is None:
            raise ArtifactError(
                f"complex runtime model parameter {name!r} lacks a real component"
            )
        if name in direct:
            raise ArtifactError(
                f"runtime model parameter {name!r} has scalar and complex records"
            )
        if imaginary is None:
            raise ArtifactError(
                f"complex runtime model parameter {name!r} lacks an imaginary component"
            )
        direct[name] = _RuntimeParameterSlots(real, imaginary, complex_kinds[name])
    return direct


def _exact_parameter_derivation(
    kernels: Sequence[PreparedKernelRecord],
    runtime_schema: Mapping[str, object],
    projection: _PreparedParameterProjection,
    payload_root: Path,
    kernel_loader: _KernelLoader,
) -> _ExactParameterDerivation | None:
    records = tuple(
        kernel for kernel in kernels if kernel.contract_kind == "model-parameter"
    )
    if not records:
        return None
    if len(records) != 1:
        raise ArtifactError(
            "eager kernel pack declares multiple model-parameter kernels"
        )
    record = records[0]
    runtime_slots = _runtime_parameter_slots(
        runtime_schema,
        projection.runtime_parameter_count,
    )
    output_names: set[str] = set()
    parsed_outputs: list[str] = []
    prefix = "model-parameter:"
    if not record.output_layout:
        raise ArtifactError("model-parameter kernel declares no outputs")
    for index, layout in enumerate(record.output_layout):
        if not layout.startswith(prefix) or len(layout) == len(prefix):
            raise ArtifactError(
                f"model-parameter kernel output {index} has invalid layout {layout!r}"
            )
        output_name = layout[len(prefix) :]
        if output_name != output_name.strip():
            raise ArtifactError(
                f"model-parameter kernel output {index} has invalid layout {layout!r}"
            )
        if output_name in output_names:
            raise ArtifactError(
                f"model-parameter kernel repeats output parameter {output_name!r}"
            )
        output_names.add(output_name)
        parsed_outputs.append(output_name)

    input_names: set[str] = set()
    input_parameter_indices: list[int] = []
    for index, contract in enumerate(record.input_contracts):
        if contract.get("role") != "model-parameter":
            raise ArtifactError(
                f"model-parameter kernel input {index} is not a model parameter"
            )
        input_name = contract.get("model_parameter_name")
        stable_index = contract.get("model_parameter_index")
        component = contract.get("component")
        if not isinstance(input_name, str) or not input_name:
            raise ArtifactError(
                f"model-parameter kernel input {index} lacks a logical name"
            )
        if (
            isinstance(stable_index, bool)
            or not isinstance(stable_index, int)
            or stable_index < 0
        ):
            raise ArtifactError(
                f"model-parameter kernel input {index} lacks a stable index"
            )
        if component != 0:
            raise ArtifactError(
                f"model-parameter kernel input {index} has invalid component "
                f"{component!r}"
            )
        entry = projection.entry(input_name)
        if entry is None or entry.prepared_index != stable_index:
            raise ArtifactError(
                f"model-parameter kernel input {input_name!r} has no matching "
                "projection"
            )
        input_slots = runtime_slots[input_name]
        if input_slots.kind not in {
            "external_parameter",
            "external_parameter_component",
        }:
            raise ArtifactError(
                f"model-parameter kernel input {input_name!r} is not an "
                "external/base "
                "parameter"
            )
        if input_name in input_names:
            raise ArtifactError(
                f"model-parameter kernel repeats input parameter {input_name!r}"
            )
        input_names.add(input_name)
        input_parameter_indices.append(stable_index)
    overlap = input_names & output_names
    if overlap:
        raise ArtifactError(
            "model-parameter kernel inputs overlap its derived outputs: "
            + ", ".join(sorted(overlap))
        )

    derived_runtime_names = {
        runtime_name
        for runtime_name, runtime_slots_record in runtime_slots.items()
        if runtime_slots_record.kind == "derived_parameter_component"
    }
    missing_outputs = derived_runtime_names - output_names
    if missing_outputs:
        raise ArtifactError(
            "model-parameter kernel does not output runtime derived parameters: "
            + ", ".join(repr(name) for name in sorted(missing_outputs))
        )

    targets: list[_DerivedParameterTarget | None] = []
    for output_name in parsed_outputs:
        output_slots = runtime_slots.get(output_name)
        entry = projection.entry(output_name)
        if output_slots is None:
            if entry is not None:
                raise ArtifactError(
                    f"derived prepared parameter {output_name!r} has no runtime slots"
                )
            targets.append(None)
            continue
        if output_slots.kind != "derived_parameter_component":
            raise ArtifactError(
                f"model-parameter kernel output {output_name!r} does not target "
                "derived runtime slots"
            )
        if output_slots.imaginary is None:
            raise ArtifactError(
                f"derived runtime parameter {output_name!r} lacks an imaginary "
                "component"
            )
        targets.append(
            _DerivedParameterTarget(
                name=output_name,
                prepared_index=(None if entry is None else entry.prepared_index),
                runtime_real_index=output_slots.real,
                runtime_imaginary_index=output_slots.imaginary,
            )
        )
    return _ExactParameterDerivation(
        _LazyExactKernel(record, payload_root, kernel_loader),
        tuple(input_parameter_indices),
        tuple(targets),
    )
