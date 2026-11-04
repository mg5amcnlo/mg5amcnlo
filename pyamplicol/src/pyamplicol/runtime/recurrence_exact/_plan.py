# SPDX-License-Identifier: 0BSD
"""Validated exact-execution view of a topology-replay recurrence artifact."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from pyamplicol._internal.versions import (
    PROCESS_ARTIFACT_SCHEMA_VERSION,
    RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY,
    SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
)
from pyamplicol.api.errors import ArtifactError, CompatibilityError
from pyamplicol.artifacts.manifest import ArtifactManifest
from pyamplicol.models.prepared import (
    PreparedKernelPack,
    PreparedKernelRecord,
    PreparedModelBundle,
)
from pyamplicol.models.recurrence_direct_template import (
    RECURRENCE_DIRECT_IDENTITY_FINALIZER,
)
from pyamplicol.runtime._evaluator_payloads import ExactEvaluatorPayloadResolver
from pyamplicol.runtime.eager_exact._contracts import (
    _complex_pair,
    _KernelLoader,
    _LazyExactKernel,
    _mapping,
    _sequence,
)
from pyamplicol.runtime.eager_exact._plan import (
    _ExactModelParameterState,
    _load_exact_kernel_pack,
)
from pyamplicol.runtime.symbolica_exact import _ExactEvaluator

from ._color import (
    RECURRENCE_CONTRACTED_COLOR_CAPABILITY,
    _load_recurrence_color_contraction,
    _RecurrenceColorContraction,
)
from ._plan_v2 import (
    RECURRENCE_DIRECT_RUNTIME_CAPABILITY,
    RECURRENCE_PLAN_V2_ABI,
    RECURRENCE_RUNTIME_KIND,
    RECURRENCE_RUNTIME_LAYOUT_V2_ABI,
    _Current,
    _Executor,
    _load_recurrence_exact_sections_v1,
    _NativeExactSectionsLoader,
    _parse_exact_sections,
    _RecurrenceExactSectionsV1,
)

_RECURRENCE_COLOR_CAPABILITY = "rusticol.recurrence-color.lc.v1"


@dataclass(frozen=True, slots=True)
class _ParameterProjectionRow:
    runtime_slot: int
    prepared_slot: int
    component: int


@dataclass(frozen=True, slots=True)
class _RuntimeParameterSchemaRow:
    runtime_slot: int
    runtime_name: str
    parameter_template_id: int
    prepared_parameter_id: int | None
    component: int
    probe_policy: str = "native-template-default-perturbed-v1"


@dataclass(frozen=True, slots=True)
class _SourceTemplate:
    template_id: int
    dimension: int
    helicity: int
    chirality: int
    spin_state: int
    family: str
    orientation: str
    mass_prepared_parameter_id: int | None
    crossing_helicity_factor: int
    crossing_chirality_factor: int
    crossing_spin_state_factor: int
    mass_parameter_name: str | None = None


@dataclass(frozen=True, slots=True)
class _PreparedParameterDerivation:
    kernel: _LazyExactKernel
    input_indices: tuple[int, ...]
    output_indices: tuple[int | None, ...]

    def evaluate(
        self,
        prepared_parameters: Sequence[tuple[Decimal, Decimal]],
        precision: int,
    ) -> tuple[tuple[Decimal, Decimal], ...]:
        try:
            inputs = tuple(prepared_parameters[index] for index in self.input_indices)
        except IndexError as exc:
            raise ArtifactError(
                "recurrence parameter derivation input is out of range"
            ) from exc
        outputs = self.kernel.evaluate(inputs, precision)
        if len(outputs) != len(self.output_indices):
            raise ArtifactError(
                "recurrence parameter derivation output count is inconsistent"
            )
        prepared = list(prepared_parameters)
        for output, target in zip(outputs, self.output_indices, strict=True):
            if target is not None:
                prepared[target] = output
        return tuple(prepared)


@dataclass(slots=True)
class _RecurrenceExactPlan:
    sections: _RecurrenceExactSectionsV1
    kernels: Mapping[int, _LazyExactKernel]
    executors: Mapping[int, _Executor]
    executor_exact_kernel_ids: Mapping[int, int]
    executor_parent_permutations: Mapping[int, tuple[int, int]]
    source_templates: Mapping[int, _SourceTemplate]
    initial_source_slots: frozenset[int]
    executor_couplings: Mapping[int, tuple[Decimal, Decimal]]
    prepared_defaults: tuple[tuple[Decimal, Decimal], ...]
    parameter_projection: tuple[_ParameterProjectionRow, ...]
    parameter_derivation: _PreparedParameterDerivation | None
    runtime_defaults: tuple[Decimal, ...] = ()
    runtime_parameter_schema: tuple[_RuntimeParameterSchemaRow, ...] = ()
    color_contraction: _RecurrenceColorContraction | None = None
    external_source_slots: tuple[int, ...] = ()
    # Populated lazily only after the complete plan boundary has validated.
    # The recipes are construction/runtime-session state, never serialized.
    trusted_executor_recipes: dict[int, object] = field(
        default_factory=dict,
        repr=False,
    )
    trusted_currents_by_stage: dict[int, tuple[_Current, ...]] = field(
        default_factory=dict,
        repr=False,
    )

    @classmethod
    def load(
        cls,
        *,
        artifact_root: Path,
        process_id: str,
        execution_path: str,
        execution: Mapping[str, object],
        manifest: ArtifactManifest,
        kernel_loader: _KernelLoader | None,
        exact_payloads: ExactEvaluatorPayloadResolver,
        native_sections_loader: _NativeExactSectionsLoader | None = None,
    ) -> _RecurrenceExactPlan:
        _validate_execution(execution, process_id)
        sections = _load_recurrence_exact_sections_v1(
            artifact_root,
            process_id,
            loader=native_sections_loader,
        )
        color_contraction = _load_recurrence_color_contraction(
            artifact_root=artifact_root,
            process_id=process_id,
            execution_path=execution_path,
            execution=execution,
            manifest=manifest,
        )
        pack, payload_root, effective_kernel_loader = _load_exact_kernel_pack(
            artifact_root=artifact_root,
            execution=execution,
            manifest=manifest,
            kernel_loader=kernel_loader,
            exact_payloads=exact_payloads,
        )
        kernels = {
            record.kernel_id: _LazyExactKernel(
                record,
                payload_root,
                effective_kernel_loader,
            )
            for record in pack.kernels
            if record.contract_kind != "model-parameter"
        }
        metadata = _mapping(execution.get("runtime_metadata"), "runtime metadata")
        runtime_parameters = tuple(
            _mapping(value, f"runtime parameter {index}")
            for index, value in enumerate(
                _sequence(metadata.get("runtime_parameters"), "runtime parameters")
            )
        )
        runtime_defaults = _parse_runtime_parameter_defaults(runtime_parameters)
        runtime_parameter_schema = _runtime_parameter_schema(
            metadata,
            runtime_parameters,
        )
        defaults = tuple(
            _parse_complex_default(value, index)
            for index, value in enumerate(
                _sequence(
                    metadata.get("prepared_parameter_defaults"),
                    "prepared parameter defaults",
                )
            )
        )
        if len(defaults) != sections.parameter_value_count:
            raise ArtifactError(
                "prepared parameter defaults do not match the recurrence plan"
            )
        projection_rows, external_prepared_by_name = _parameter_projection(
            metadata,
            len(runtime_parameters),
            len(defaults),
        )
        prepared_by_name = dict(
            _prepared_parameter_indices(
                pack.kernels,
                len(defaults),
            )
        )
        for name, prepared_index in external_prepared_by_name.items():
            previous = prepared_by_name.setdefault(name, prepared_index)
            if previous != prepared_index:
                raise ArtifactError(
                    f"prepared parameter {name!r} has conflicting stable indices"
                )
        derivation = _prepared_parameter_derivation(
            pack.kernels,
            payload_root,
            effective_kernel_loader,
            prepared_by_name,
        )
        executor_exact_kernel_ids = _executor_exact_kernel_ids(pack)
        external_source_slots, initial_source_slots = _external_source_layout(
            metadata,
            sections.external_source_count,
        )
        result = cls(
            sections=sections,
            kernels=kernels,
            executors={row.executor_id: row for row in sections.executors},
            executor_exact_kernel_ids=executor_exact_kernel_ids,
            executor_parent_permutations=_executor_parent_permutations(pack),
            source_templates=_source_templates(metadata, prepared_by_name),
            initial_source_slots=initial_source_slots,
            executor_couplings=_executor_couplings(
                pack,
                executor_exact_kernel_ids,
            ),
            prepared_defaults=defaults,
            parameter_projection=projection_rows,
            parameter_derivation=derivation,
            runtime_defaults=runtime_defaults,
            runtime_parameter_schema=runtime_parameter_schema,
            color_contraction=color_contraction,
            external_source_slots=external_source_slots,
        )
        result._validate(pack)
        return result

    @classmethod
    def from_generation(
        cls,
        *,
        raw_sections: object,
        process_id: str,
        bundle: PreparedModelBundle,
        runtime_metadata: Mapping[str, object],
    ) -> _RecurrenceExactPlan:
        """Construct an unpublished exact plan for generation-time probes."""

        sections = _parse_exact_sections(raw_sections, process_id)
        pack = bundle.kernel_pack
        payload_root = Path()

        def kernel_loader(
            record: PreparedKernelRecord,
            root: Path,
        ) -> _ExactEvaluator:
            return _ExactEvaluator.load(
                {
                    "evaluator_state_path": record.exact_evaluator_state_path,
                    "input_len": record.input_arity,
                },
                root,
                state_loader=bundle.read_payload,
            )

        kernels = {
            record.kernel_id: _LazyExactKernel(
                record,
                payload_root,
                kernel_loader,
            )
            for record in pack.kernels
            if record.contract_kind != "model-parameter"
        }
        runtime_parameters = tuple(
            _mapping(value, f"runtime parameter {index}")
            for index, value in enumerate(
                _sequence(
                    runtime_metadata.get("runtime_parameters"),
                    "runtime parameters",
                )
            )
        )
        runtime_defaults = _parse_runtime_parameter_defaults(runtime_parameters)
        runtime_parameter_schema = _runtime_parameter_schema(
            runtime_metadata,
            runtime_parameters,
        )
        defaults = tuple(
            _parse_complex_default(value, index)
            for index, value in enumerate(
                _sequence(
                    runtime_metadata.get("prepared_parameter_defaults"),
                    "prepared parameter defaults",
                )
            )
        )
        if len(defaults) != sections.parameter_value_count:
            raise ArtifactError(
                "prepared parameter defaults do not match the recurrence plan"
            )
        projection_rows, external_prepared_by_name = _parameter_projection(
            runtime_metadata,
            len(runtime_parameters),
            len(defaults),
        )
        prepared_by_name = dict(
            _prepared_parameter_indices(pack.kernels, len(defaults))
        )
        for name, prepared_index in external_prepared_by_name.items():
            previous = prepared_by_name.setdefault(name, prepared_index)
            if previous != prepared_index:
                raise ArtifactError(
                    f"prepared parameter {name!r} has conflicting stable indices"
                )
        derivation = _prepared_parameter_derivation(
            pack.kernels,
            payload_root,
            kernel_loader,
            prepared_by_name,
        )
        executor_exact_kernel_ids = _executor_exact_kernel_ids(pack)
        external_source_slots, initial_source_slots = _external_source_layout(
            runtime_metadata,
            sections.external_source_count,
        )
        result = cls(
            sections=sections,
            kernels=kernels,
            executors={row.executor_id: row for row in sections.executors},
            executor_exact_kernel_ids=executor_exact_kernel_ids,
            executor_parent_permutations=_executor_parent_permutations(pack),
            source_templates=_source_templates(
                runtime_metadata,
                prepared_by_name,
            ),
            initial_source_slots=initial_source_slots,
            executor_couplings=_executor_couplings(
                pack,
                executor_exact_kernel_ids,
            ),
            prepared_defaults=defaults,
            parameter_projection=projection_rows,
            parameter_derivation=derivation,
            runtime_defaults=runtime_defaults,
            runtime_parameter_schema=runtime_parameter_schema,
            color_contraction=None,
            external_source_slots=external_source_slots,
        )
        result._validate(pack, allow_missing_color_contraction=True)
        return result

    def resolve_model_parameters(
        self,
        runtime_parameters: Sequence[Decimal],
        precision: int,
    ) -> _ExactModelParameterState:
        prepared = list(self.prepared_defaults)
        for row in self.parameter_projection:
            try:
                value = runtime_parameters[row.runtime_slot]
            except IndexError as exc:
                raise ArtifactError(
                    "recurrence runtime parameter projection is out of range"
                ) from exc
            real, imaginary = prepared[row.prepared_slot]
            prepared[row.prepared_slot] = (
                value if row.component == 0 else real,
                value if row.component == 1 else imaginary,
            )
        runtime = tuple(runtime_parameters)
        if self.parameter_derivation is None:
            return _ExactModelParameterState(runtime, tuple(prepared))
        return _ExactModelParameterState(
            runtime,
            self.parameter_derivation.evaluate(
                prepared,
                precision,
            ),
        )

    def _validate(
        self,
        pack: PreparedKernelPack,
        *,
        allow_missing_color_contraction: bool = False,
    ) -> None:
        if (
            self.sections.strategy == "contracted-color-union"
            and self.color_contraction is None
            and not allow_missing_color_contraction
        ) or (
            self.sections.strategy != "contracted-color-union"
            and self.color_contraction is not None
        ):
            raise ArtifactError(
                "recurrence exact plan and contracted-color payload disagree"
            )
        if len(self.executors) != len(self.sections.executors):
            raise ArtifactError("recurrence direct executor IDs are duplicated")
        pack_by_id = {record.kernel_id: record for record in pack.kernels}
        expected_kind = {
            "contribution": "vertex",
            "finalization": "propagator",
            "closure": "closure",
        }
        for executor in self.sections.executors:
            parent_permutation = self.executor_parent_permutations.get(
                executor.executor_id
            )
            if parent_permutation not in {(0, 1), (1, 0)}:
                raise ArtifactError(
                    f"direct executor {executor.executor_id} has no authenticated "
                    "parent permutation"
                )
            if executor.role != "contribution" and parent_permutation != (0, 1):
                raise ArtifactError(
                    f"non-contribution executor {executor.executor_id} has a "
                    "non-identity parent permutation"
                )
            exact_kernel_id = self.executor_exact_kernel_ids.get(
                executor.executor_id,
                executor.prepared_kernel_id,
            )
            if exact_kernel_id is None:
                if executor.runtime_template is None:
                    raise ArtifactError(
                        f"direct executor {executor.executor_id} has no exact binding"
                    )
                continue
            if (
                executor.runtime_template is not None
                and executor.executor_id not in self.executor_exact_kernel_ids
            ):
                raise ArtifactError(
                    f"direct executor {executor.executor_id} mixes exact bindings"
                )
            record = pack_by_id.get(exact_kernel_id)
            if record is None:
                raise ArtifactError(
                    f"direct executor {executor.executor_id} references absent "
                    f"prepared kernel {exact_kernel_id}"
                )
            if record.contract_kind != expected_kind.get(executor.role):
                raise ArtifactError(
                    f"direct executor {executor.executor_id} has the wrong "
                    "prepared kernel role"
                )
            if record.output_arity != executor.destination_component_count:
                raise ArtifactError(
                    f"direct executor {executor.executor_id} output width disagrees "
                    "with its prepared kernel"
                )
        if self.sections.strategy in {"topology-replay", "contracted-color-union"}:
            for source in self.sections.sources:
                if (
                    source.source_template_or_dispatch_domain
                    not in self.source_templates
                ):
                    raise ArtifactError(
                        "recurrence source references absent template "
                        f"{source.source_template_or_dispatch_domain}"
                    )
        else:
            for variant in self.sections.source_dispatch_variants:
                if variant.source_template_id not in self.source_templates:
                    raise ArtifactError(
                        "recurrence source-dispatch variant references absent "
                        f"template {variant.source_template_id}"
                    )


def _executor_exact_kernel_ids(
    pack: PreparedKernelPack,
) -> Mapping[int, int]:
    """Recover exact kernels hidden behind certified native intrinsics."""

    semantic_catalog = pack.recurrence_template_catalog
    direct_catalog = pack.recurrence_direct_template_catalog
    if semantic_catalog is None or direct_catalog is None:
        raise ArtifactError(
            "exact recurrence execution requires semantic and direct template catalogs"
        )
    kernels = {record.kernel_id: record for record in pack.kernels}
    expected_kind = {
        "contribution": "vertex",
        "finalization": "propagator",
        "closure": "closure",
    }
    result: dict[int, int] = {}
    for direct in direct_catalog.templates:
        kernel_id = direct.payload_binding.prepared_kernel_id
        binding_id = direct.evaluator_binding_id
        if kernel_id is None:
            if binding_id >= len(semantic_catalog.evaluator_bindings):
                if (
                    binding_id == len(semantic_catalog.evaluator_bindings)
                    and direct.role == "finalization"
                    and direct.payload_binding.runtime_template
                    == RECURRENCE_DIRECT_IDENTITY_FINALIZER
                ):
                    continue
                raise ArtifactError(
                    f"direct executor {direct.direct_executor_id} references "
                    f"absent evaluator binding {binding_id}"
                )
            binding = semantic_catalog.evaluator_bindings[binding_id]
            if binding.callable_kind != "prepared-kernel":
                continue
            kernel_id = binding.prepared_kernel_id
        else:
            if binding_id >= len(semantic_catalog.evaluator_bindings):
                raise ArtifactError(
                    f"direct executor {direct.direct_executor_id} references "
                    f"absent evaluator binding {binding_id}"
                )
            binding = semantic_catalog.evaluator_bindings[binding_id]
        if kernel_id is None:
            raise ArtifactError(
                f"direct executor {direct.direct_executor_id} has an empty exact "
                "kernel binding"
            )
        if binding.resolver_key != direct.evaluator_resolver_key:
            raise ArtifactError(
                f"direct executor {direct.direct_executor_id} resolver does not "
                "match its semantic evaluator binding"
            )
        record = kernels.get(kernel_id)
        if record is None:
            raise ArtifactError(
                f"direct executor {direct.direct_executor_id} references absent "
                f"prepared kernel {kernel_id}"
            )
        if record.contract_kind != expected_kind.get(direct.role):
            raise ArtifactError(
                f"direct executor {direct.direct_executor_id} exact kernel has "
                "the wrong role"
            )
        result[direct.direct_executor_id] = kernel_id
    return result


def _executor_parent_permutations(
    pack: PreparedKernelPack,
) -> Mapping[int, tuple[int, int]]:
    """Recover authenticated prepared-kernel parent order for exact execution."""

    direct_catalog = pack.recurrence_direct_template_catalog
    if direct_catalog is None:
        raise ArtifactError(
            "exact recurrence execution requires a direct template catalog"
        )
    result: dict[int, tuple[int, int]] = {}
    for direct in direct_catalog.templates:
        permutation = direct.payload_binding.contribution_parent_permutation
        if permutation not in {(0, 1), (1, 0)}:
            raise ArtifactError(
                f"direct executor {direct.direct_executor_id} has an invalid "
                "parent permutation"
            )
        if direct.role != "contribution" and permutation != (0, 1):
            raise ArtifactError(
                f"non-contribution executor {direct.direct_executor_id} has a "
                "non-identity parent permutation"
            )
        previous = result.setdefault(direct.direct_executor_id, permutation)
        if previous != permutation:
            raise ArtifactError(
                f"direct executor {direct.direct_executor_id} has conflicting "
                "parent permutations"
            )
    return result


def _executor_couplings(
    pack: PreparedKernelPack,
    executor_exact_kernel_ids: Mapping[int, int],
) -> Mapping[int, tuple[Decimal, Decimal]]:
    semantic_catalog = pack.recurrence_template_catalog
    direct_catalog = pack.recurrence_direct_template_catalog
    if semantic_catalog is None or direct_catalog is None:
        raise ArtifactError(
            "exact recurrence execution requires semantic and direct template catalogs"
        )
    semantic_records = {
        record.template_id: record
        for records in (
            semantic_catalog.transitions,
            semantic_catalog.closures,
        )
        for record in records
    }
    kernels = {record.kernel_id: record for record in pack.kernels}
    result: dict[int, tuple[Decimal, Decimal]] = {}
    for direct in direct_catalog.templates:
        kernel_id = executor_exact_kernel_ids.get(direct.direct_executor_id)
        if kernel_id is None:
            continue
        kernel = kernels.get(kernel_id)
        if kernel is None:
            raise ArtifactError(
                f"direct executor {direct.direct_executor_id} references absent "
                f"prepared kernel {kernel_id}"
            )
        if not any(
            contract.get("role") in {"coupling-real", "coupling-imag"}
            for contract in kernel.input_contracts
        ):
            continue
        records = tuple(
            semantic_records.get(template_id)
            for template_id in direct.semantic_template_ids
        )
        if any(record is None for record in records):
            raise ArtifactError(
                f"direct executor {direct.direct_executor_id} has no uniform exact "
                "semantic coupling"
            )
        couplings = {
            record.binding_coupling for record in records if record is not None
        }
        if len(couplings) != 1:
            raise ArtifactError(
                f"direct executor {direct.direct_executor_id} has no uniform exact "
                "semantic coupling"
            )
        coupling = next(iter(couplings))
        result[direct.direct_executor_id] = (
            Decimal(coupling.real_numerator) / Decimal(coupling.real_denominator),
            Decimal(coupling.imag_numerator) / Decimal(coupling.imag_denominator),
        )
    return result


def _validate_execution(
    execution: Mapping[str, object],
    process_id: str,
) -> None:
    if execution.get("schema_version") != PROCESS_ARTIFACT_SCHEMA_VERSION:
        raise CompatibilityError(
            f"unsupported recurrence process schema {execution.get('schema_version')!r}"
        )
    if execution.get("kind") != RECURRENCE_RUNTIME_KIND:
        raise CompatibilityError(
            f"unsupported exact recurrence kind {execution.get('kind')!r}"
        )
    if execution.get("key") != process_id:
        raise ArtifactError("recurrence execution metadata selects the wrong process")
    if execution.get("recurrence_plan_abi") != RECURRENCE_PLAN_V2_ABI:
        raise CompatibilityError(
            f"unsupported recurrence plan ABI {execution.get('recurrence_plan_abi')!r}"
        )
    if execution.get("runtime_layout_abi") != RECURRENCE_RUNTIME_LAYOUT_V2_ABI:
        raise CompatibilityError(
            f"unsupported recurrence runtime-layout ABI "
            f"{execution.get('runtime_layout_abi')!r}"
        )
    summary = _mapping(execution.get("recurrence_summary"), "recurrence summary")
    layout = summary.get("lc_flow_layout")
    if layout not in {
        "topology-replay",
        "all-flow-union",
        "contracted-color-union",
    }:
        raise CompatibilityError(
            "unsupported exact recurrence LC flow layout "
            f"{summary.get('lc_flow_layout')!r}"
        )
    capabilities = execution.get("required_runtime_capabilities")
    color_capability = (
        RECURRENCE_CONTRACTED_COLOR_CAPABILITY
        if layout == "contracted-color-union"
        else _RECURRENCE_COLOR_CAPABILITY
    )
    required_capabilities = {
        RECURRENCE_DIRECT_RUNTIME_CAPABILITY,
        color_capability,
    }
    actual_capabilities = set(_sequence(capabilities, "recurrence capabilities"))
    permitted_capabilities = required_capabilities | {
        RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY,
    }
    if layout == "contracted-color-union":
        permitted_capabilities.add(SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY)
    if not required_capabilities <= actual_capabilities or not (
        actual_capabilities <= permitted_capabilities
    ):
        raise CompatibilityError("unsupported recurrence runtime capability contract")
    color_accuracy = execution.get("color_accuracy")
    if (layout == "contracted-color-union") != (color_accuracy in {"nlc", "full"}):
        raise ArtifactError(
            "recurrence layout and color-accuracy contracts are inconsistent"
        )


def _parse_complex_default(
    raw: object,
    index: int,
) -> tuple[Decimal, Decimal]:
    values = _sequence(raw, f"prepared parameter default {index}")
    if len(values) != 2:
        raise ArtifactError(
            f"prepared parameter default {index} must be a complex pair"
        )
    return _complex_pair(
        values[0],
        values[1],
        f"prepared parameter default {index}",
    )


def _parse_runtime_parameter_defaults(
    runtime_parameters: Sequence[Mapping[str, object]],
) -> tuple[Decimal, ...]:
    defaults = []
    for index, parameter in enumerate(runtime_parameters):
        if parameter.get("parameter_index") != index:
            raise ArtifactError(
                f"runtime parameter {index} has a non-canonical parameter index"
            )
        defaults.append(
            _complex_pair(
                parameter.get("default"),
                0,
                f"runtime parameter {index} default",
            )[0]
        )
    return tuple(defaults)


def _parameter_projection(
    metadata: Mapping[str, object],
    runtime_count: int,
    prepared_count: int,
) -> tuple[tuple[_ParameterProjectionRow, ...], Mapping[str, int]]:
    rows = []
    by_name: dict[str, int] = {}
    for index, raw in enumerate(
        _sequence(metadata.get("parameter_projection"), "parameter projection")
    ):
        row = _mapping(raw, f"parameter projection {index}")
        runtime_slot = _nonnegative_int(
            row.get("runtime_slot"), f"parameter projection {index} runtime slot"
        )
        component = _nonnegative_int(
            row.get("component"), f"parameter projection {index} component"
        )
        prepared_slot = row.get("prepared_parameter_id")
        name = row.get("runtime_name")
        if not isinstance(name, str) or not name:
            raise ArtifactError(f"parameter projection {index} has no runtime name")
        if prepared_slot is None:
            continue
        prepared_slot = _nonnegative_int(
            prepared_slot, f"parameter projection {index} prepared slot"
        )
        if (
            runtime_slot >= runtime_count
            or prepared_slot >= prepared_count
            or component not in {0, 1}
        ):
            raise ArtifactError(f"parameter projection {index} is out of range")
        previous = by_name.setdefault(name, prepared_slot)
        if previous != prepared_slot:
            raise ArtifactError(
                f"prepared parameter {name!r} has conflicting stable indices"
            )
        rows.append(_ParameterProjectionRow(runtime_slot, prepared_slot, component))
    return tuple(rows), by_name


def _runtime_parameter_schema(
    metadata: Mapping[str, object],
    runtime_parameters: Sequence[Mapping[str, object]],
) -> tuple[_RuntimeParameterSchemaRow, ...]:
    raw_projection = tuple(
        _mapping(value, f"parameter projection {index}")
        for index, value in enumerate(
            _sequence(metadata.get("parameter_projection"), "parameter projection")
        )
    )
    if len(raw_projection) != len(runtime_parameters):
        raise ArtifactError(
            "recurrence runtime parameter schema does not cover every flattened slot"
        )
    rows: list[_RuntimeParameterSchemaRow] = []
    for index, (projection, parameter) in enumerate(
        zip(raw_projection, runtime_parameters, strict=True)
    ):
        runtime_slot = _nonnegative_int(
            projection.get("runtime_slot"),
            f"parameter projection {index} runtime slot",
        )
        parameter_index = _nonnegative_int(
            parameter.get("parameter_index"),
            f"runtime parameter {index} parameter index",
        )
        runtime_name = projection.get("runtime_name")
        if (
            runtime_slot != index
            or parameter_index != index
            or not isinstance(runtime_name, str)
            or not runtime_name
        ):
            raise ArtifactError(
                f"recurrence runtime parameter schema row {index} is not canonical"
            )
        prepared_parameter_id = projection.get("prepared_parameter_id")
        if prepared_parameter_id is not None:
            prepared_parameter_id = _nonnegative_int(
                prepared_parameter_id,
                f"parameter projection {index} prepared parameter ID",
            )
        parameter_kind = parameter.get("kind")
        if not isinstance(parameter_kind, str) or not parameter_kind:
            raise ArtifactError(
                f"runtime parameter {index} has no canonical parameter kind"
            )
        probe_policy = (
            "derived-overwritten-fixed-zero-v1"
            if parameter_kind in {"derived", "derived_parameter_component"}
            else "native-template-default-perturbed-v1"
        )
        rows.append(
            _RuntimeParameterSchemaRow(
                runtime_slot=runtime_slot,
                runtime_name=runtime_name,
                parameter_template_id=_nonnegative_int(
                    projection.get("parameter_template_id"),
                    f"parameter projection {index} parameter template ID",
                ),
                prepared_parameter_id=prepared_parameter_id,
                component=_nonnegative_int(
                    projection.get("component"),
                    f"parameter projection {index} component",
                ),
                probe_policy=probe_policy,
            )
        )
    return tuple(rows)


def _prepared_parameter_indices(
    kernels: Sequence[PreparedKernelRecord],
    prepared_count: int,
) -> Mapping[str, int]:
    by_name: dict[str, int] = {}
    by_index: dict[int, str] = {}
    for kernel in kernels:
        for input_index, contract in enumerate(kernel.input_contracts):
            if contract.get("role") != "model-parameter":
                continue
            name = contract.get("model_parameter_name")
            prepared_index = contract.get("model_parameter_index")
            if not isinstance(name, str) or not name:
                raise ArtifactError(
                    f"prepared kernel {kernel.kernel_id} input {input_index} "
                    "has no model-parameter name"
                )
            if (
                isinstance(prepared_index, bool)
                or not isinstance(prepared_index, int)
                or prepared_index < 0
                or prepared_index >= prepared_count
            ):
                raise ArtifactError(
                    f"prepared kernel {kernel.kernel_id} input {input_index} "
                    "has an invalid model-parameter index"
                )
            previous_index = by_name.setdefault(name, prepared_index)
            if previous_index != prepared_index:
                raise ArtifactError(
                    f"prepared parameter {name!r} has conflicting stable indices"
                )
            previous_name = by_index.setdefault(prepared_index, name)
            if previous_name != name:
                raise ArtifactError(
                    f"prepared parameter index {prepared_index} has conflicting names"
                )
    return by_name


def _prepared_parameter_derivation(
    kernels: Sequence[PreparedKernelRecord],
    payload_root: Path,
    kernel_loader: _KernelLoader,
    prepared_by_name: Mapping[str, int],
) -> _PreparedParameterDerivation | None:
    records = tuple(
        kernel for kernel in kernels if kernel.contract_kind == "model-parameter"
    )
    if not records:
        return None
    if len(records) != 1:
        raise ArtifactError(
            "recurrence prepared pack declares multiple model-parameter kernels"
        )
    record = records[0]
    input_indices = []
    for input_index, contract in enumerate(record.input_contracts):
        if contract.get("role") != "model-parameter":
            raise ArtifactError(
                f"model-parameter kernel input {input_index} has an invalid role"
            )
        name = contract.get("model_parameter_name")
        prepared_index = contract.get("model_parameter_index")
        if (
            not isinstance(name, str)
            or isinstance(prepared_index, bool)
            or not isinstance(prepared_index, int)
            or prepared_by_name.get(name) != prepared_index
        ):
            raise ArtifactError(
                f"model-parameter kernel input {input_index} has no stable "
                "prepared-parameter binding"
            )
        input_indices.append(prepared_index)

    output_indices = []
    seen_outputs: set[str] = set()
    prefix = "model-parameter:"
    for output_index, layout in enumerate(record.output_layout):
        if not layout.startswith(prefix) or len(layout) == len(prefix):
            raise ArtifactError(
                f"model-parameter kernel output {output_index} has invalid layout"
            )
        name = layout[len(prefix) :]
        if name != name.strip() or name in seen_outputs:
            raise ArtifactError(
                f"model-parameter kernel output {output_index} has invalid name"
            )
        seen_outputs.add(name)
        output_indices.append(prepared_by_name.get(name))
    return _PreparedParameterDerivation(
        kernel=_LazyExactKernel(record, payload_root, kernel_loader),
        input_indices=tuple(input_indices),
        output_indices=tuple(output_indices),
    )


def _source_templates(
    metadata: Mapping[str, object],
    prepared_by_name: Mapping[str, int],
) -> Mapping[int, _SourceTemplate]:
    result = {}
    for index, raw in enumerate(
        _sequence(metadata.get("source_templates"), "source templates")
    ):
        source = _mapping(raw, f"source template {index}")
        template_id = _nonnegative_int(
            source.get("source_template_id"), f"source template {index} ID"
        )
        source_ir = _mapping(source.get("source_ir"), f"source template {index} IR")
        identity = _mapping(
            source_ir.get("identity"), f"source template {index} identity"
        )
        orientation = identity.get("orientation")
        family = source_ir.get("wavefunction_family")
        if orientation not in {"particle", "antiparticle", "self-conjugate"}:
            raise ArtifactError(f"source template {index} has invalid orientation")
        if family not in {"scalar", "fermion", "vector", "spin2"}:
            raise CompatibilityError(
                f"exact recurrence source family {family!r} is unsupported"
            )
        crossing = _mapping(source.get("crossing"), f"source template {index} crossing")
        mass_name = source_ir.get("mass_parameter")
        if mass_name is not None and not isinstance(mass_name, str):
            raise ArtifactError(f"source template {index} has invalid mass parameter")
        mass_prepared_parameter_id = (
            None if mass_name is None else prepared_by_name.get(mass_name)
        )
        if mass_name is not None and mass_prepared_parameter_id is None:
            raise ArtifactError(
                f"source template {index} mass parameter {mass_name!r} "
                "has no stable prepared binding"
            )
        if template_id in result:
            raise ArtifactError(f"source template ID {template_id} is duplicated")
        result[template_id] = _SourceTemplate(
            template_id=template_id,
            dimension=_nonnegative_int(
                source.get("dimension"), f"source template {index} dimension"
            ),
            helicity=_signed_int(
                source.get("helicity"), f"source template {index} helicity"
            ),
            chirality=_signed_int(
                source.get("chirality"), f"source template {index} chirality"
            ),
            spin_state=_signed_int(
                source.get("spin_state"), f"source template {index} spin state"
            ),
            family=str(family),
            orientation=str(orientation),
            mass_prepared_parameter_id=mass_prepared_parameter_id,
            crossing_helicity_factor=_signed_int(
                crossing.get("helicity_factor"),
                f"source template {index} crossing helicity",
            ),
            crossing_chirality_factor=_signed_int(
                crossing.get("chirality_factor"),
                f"source template {index} crossing chirality",
            ),
            crossing_spin_state_factor=_signed_int(
                crossing.get("spin_state_factor"),
                f"source template {index} crossing spin",
            ),
            mass_parameter_name=mass_name,
        )
    return result


def _external_source_layout(
    metadata: Mapping[str, object],
    expected_count: int,
) -> tuple[tuple[int, ...], frozenset[int]]:
    initial = set()
    ordered_slots = []
    seen = set()
    for index, raw in enumerate(
        _sequence(metadata.get("external_legs"), "external legs")
    ):
        leg = _mapping(raw, f"external leg {index}")
        slot = _nonnegative_int(leg.get("source_slot"), f"external leg {index} slot")
        is_initial = leg.get("is_initial")
        if not isinstance(is_initial, bool) or slot in seen:
            raise ArtifactError(f"external leg {index} has invalid source metadata")
        ordered_slots.append(slot)
        seen.add(slot)
        if is_initial:
            initial.add(slot)
    if len(seen) != expected_count or seen != set(range(expected_count)):
        raise ArtifactError("external source slots must be dense from zero")
    return tuple(ordered_slots), frozenset(initial)


def _nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactError(f"{context} must be a non-negative integer")
    return value


def _signed_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{context} must be an integer")
    return value


__all__ = ["_RecurrenceExactPlan"]
