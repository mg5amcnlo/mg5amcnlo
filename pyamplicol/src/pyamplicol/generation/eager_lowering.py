# SPDX-License-Identifier: 0BSD
"""Lower proven process DAGs into backend-independent eager execution tables."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..models.base import Model
from ..models.prepared_catalog import (
    ClosureKernelKey,
    PropagatorKernelKey,
    VertexKernelKey,
)
from .contracts import runtime_coupling_parameter_names
from .dag_types import CurrentNode, GenericDAG, InteractionNode
from .eager_tables import (
    EAGER_OUTPUT_FACTOR_COUPLING_IMAG,
    EAGER_OUTPUT_FACTOR_COUPLING_REAL,
    EAGER_OUTPUT_FACTOR_NONE,
    EAGER_PLAN_ABI,
    EAGER_RUNTIME_CAPABILITY,
    EAGER_SELECTOR_DOMAINS_ABI,
    MISSING_U32,
    EagerAttachmentRow,
    EagerClosureRow,
    EagerCouplingRow,
    EagerFinalizationRow,
    EagerInvocationRow,
    EagerSelectorDomainIdRow,
    EagerSelectorDomainRow,
    EagerSelectorGroupRow,
    pack_rows,
)
from .runtime_schema import RuntimeSchemaLayout, build_runtime_schema_layout

EAGER_RUNTIME_KIND = "pyamplicol-runtime-eager-execution"
EagerProgressCallback = Callable[[Mapping[str, str | int]], None]

# Dense Python integer masks are substantially cheaper than materialized
# membership tuples for the selector universes seen in complete LC artifacts.
# Keep the sparse interning path for unusually large universes, where retaining
# one arbitrary-width integer per distinct domain would dominate memory.
_DENSE_SELECTOR_DOMAIN_GROUP_LIMIT = 4096


class EagerKernelResolver(Protocol):
    """Resolve model-local prepared kernels without compiling them."""

    def vertex_kernel(self, interaction: InteractionNode) -> EagerResolvedKernel: ...

    def propagator_kernel_id(
        self,
        current: CurrentNode,
        propagator: Mapping[str, object],
    ) -> int | None: ...

    def closure_kernel(
        self, root: Mapping[str, object]
    ) -> EagerResolvedKernel | None: ...


@dataclass(frozen=True, slots=True)
class EagerResolvedKernel:
    """Prepared callable plus the exact transformation into its canonical ABI."""

    kernel_id: int
    canonical_input_order: tuple[int, int] = (0, 1)
    normalization_factor: tuple[float, float] = (1.0, 0.0)
    output_factor_source: int = EAGER_OUTPUT_FACTOR_NONE

    def __post_init__(self) -> None:
        if self.kernel_id < 0:
            raise ValueError("prepared kernel ID must be nonnegative")
        if self.canonical_input_order not in ((0, 1), (1, 0)):
            raise ValueError("prepared binary input order must be a permutation")
        if complex(*self.normalization_factor) == 0j:
            raise ValueError("prepared kernel normalization factor must be nonzero")
        if self.output_factor_source not in (
            EAGER_OUTPUT_FACTOR_NONE,
            EAGER_OUTPUT_FACTOR_COUPLING_REAL,
            EAGER_OUTPUT_FACTOR_COUPLING_IMAG,
        ):
            raise ValueError("unsupported prepared output factor source")


@dataclass(frozen=True, slots=True)
class MappingEagerKernelResolver:
    """Small immutable resolver used by prepared kernel-pack manifests."""

    vertex_kernels: Mapping[int, int]
    propagator_kernels: Mapping[tuple[int, int], int]
    closure_kernels: Mapping[tuple[str, int | None], int]

    def vertex_kernel(self, interaction: InteractionNode) -> EagerResolvedKernel:
        try:
            return EagerResolvedKernel(
                int(self.vertex_kernels[interaction.vertex_kind])
            )
        except KeyError as error:
            raise ValueError(
                f"prepared model has no vertex kernel for kind "
                f"{interaction.vertex_kind}"
            ) from error

    def propagator_kernel_id(
        self,
        current: CurrentNode,
        propagator: Mapping[str, object],
    ) -> int | None:
        if not bool(propagator.get("applies_propagator", False)):
            return None
        key = (int(current.index.particle_id), int(current.index.chirality))
        try:
            return int(self.propagator_kernels[key])
        except KeyError as error:
            raise ValueError(
                f"prepared model has no propagator kernel for particle/chirality {key}"
            ) from error

    def closure_kernel(self, root: Mapping[str, object]) -> EagerResolvedKernel | None:
        kind = str(root.get("kind"))
        vertex_kind = root.get("vertex_kind")
        key = (kind, None if vertex_kind is None else int(vertex_kind))
        if kind == "direct-contraction":
            return None
        try:
            return EagerResolvedKernel(int(self.closure_kernels[key]))
        except KeyError as error:
            raise ValueError(
                f"prepared model has no closure kernel for {key}"
            ) from error


@dataclass(frozen=True, slots=True)
class PreparedCatalogEagerKernelIndex:
    """Model-local resolver index parsed once while loading a prepared pack."""

    vertices: Mapping[VertexKernelKey, EagerResolvedKernel]
    propagators: Mapping[PropagatorKernelKey, int | None]
    closures: Mapping[ClosureKernelKey, EagerResolvedKernel]

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, object],
    ) -> PreparedCatalogEagerKernelIndex:
        if manifest.get("abi") != "pyamplicol-prepared-kernel-catalog-v1":
            raise ValueError("prepared model has an incompatible resolver ABI")
        vertices: dict[VertexKernelKey, EagerResolvedKernel] = {}
        propagators: dict[PropagatorKernelKey, int | None] = {}
        closures: dict[ClosureKernelKey, EagerResolvedKernel] = {}
        for raw in _mapping_sequence(
            manifest.get("vertex_bindings"), "resolver.vertex_bindings"
        ):
            key = _vertex_kernel_key(_mapping(raw.get("key"), "resolver.vertex.key"))
            vertices[key] = _resolved_kernel(raw, "resolver.vertex")
        for raw in _mapping_sequence(
            manifest.get("propagator_bindings"), "resolver.propagator_bindings"
        ):
            key_data = _mapping(raw.get("key"), "resolver.propagator.key")
            key = PropagatorKernelKey(
                int(key_data["particle_id"]), int(key_data["chirality"])
            )
            kernel_id = raw.get("kernel_id")
            propagators[key] = None if kernel_id is None else int(kernel_id)
        for raw in _mapping_sequence(
            manifest.get("closure_bindings"), "resolver.closure_bindings"
        ):
            key = _closure_kernel_key(_mapping(raw.get("key"), "resolver.closure.key"))
            closures[key] = _resolved_kernel(raw, "resolver.closure")
        return cls(vertices=vertices, propagators=propagators, closures=closures)


class PreparedCatalogEagerKernelResolver:
    """Bind one process DAG to a validated prepared-model resolver index."""

    def __init__(
        self,
        dag: GenericDAG,
        manifest: Mapping[str, object] | PreparedCatalogEagerKernelIndex,
    ) -> None:
        index = (
            manifest
            if isinstance(manifest, PreparedCatalogEagerKernelIndex)
            else PreparedCatalogEagerKernelIndex.from_manifest(manifest)
        )
        self._dag = dag
        self._vertices = index.vertices
        self._propagators = index.propagators
        self._closures = index.closures

    def vertex_kernel(self, interaction: InteractionNode) -> EagerResolvedKernel:
        left = self._dag.currents[interaction.left_id]
        right = self._dag.currents[interaction.right_id]
        result = self._dag.currents[interaction.result_id]
        key = VertexKernelKey(
            kind=int(interaction.vertex_kind),
            particles=tuple(int(value) for value in interaction.vertex_particles),
            left_chirality=int(left.index.chirality),
            right_chirality=int(right.index.chirality),
            result_chirality=int(result.index.chirality),
            coupling=tuple(float(value) for value in interaction.coupling),
        )
        try:
            return self._vertices[key]
        except KeyError as error:
            raise ValueError(
                f"prepared model has no vertex binding for {key}"
            ) from error

    def propagator_kernel_id(
        self,
        current: CurrentNode,
        propagator: Mapping[str, object],
    ) -> int | None:
        if not bool(propagator.get("applies_propagator", False)):
            return None
        key = PropagatorKernelKey(
            int(current.index.particle_id), int(current.index.chirality)
        )
        try:
            kernel_id = self._propagators[key]
        except KeyError as error:
            raise ValueError(
                f"prepared model has no propagator binding for {key}"
            ) from error
        if kernel_id is None:
            raise ValueError(
                f"prepared propagator binding {key} does not provide a kernel"
            )
        return kernel_id

    def closure_kernel(self, root: Mapping[str, object]) -> EagerResolvedKernel | None:
        if str(root.get("kind")) == "direct-contraction":
            return None
        kind = root.get("vertex_kind")
        particles = root.get("vertex_particles")
        if kind is None or particles is None:
            raise ValueError("prepared vertex closure lacks vertex identity")
        left = self._dag.currents[int(root["left_current_id"])]
        right = self._dag.currents[int(root["right_current_id"])]
        key = ClosureKernelKey(
            kind=int(kind),
            particles=tuple(
                int(value) for value in _sequence(particles, "closure.vertex_particles")
            ),
            left_chirality=int(left.index.chirality),
            right_chirality=int(right.index.chirality),
            coupling=tuple(
                float(value)
                for value in _sequence(root.get("coupling"), "closure.coupling")
            ),
        )
        try:
            return self._closures[key]
        except KeyError as error:
            raise ValueError(
                f"prepared model has no closure binding for {key}"
            ) from error


@dataclass(frozen=True, slots=True)
class EagerStageTables:
    stage_index: int
    subset_size: int
    invocations: tuple[EagerInvocationRow, ...]
    attachments: tuple[EagerAttachmentRow, ...]
    finalizations: tuple[EagerFinalizationRow, ...]

    def __post_init__(self) -> None:
        cursor = 0
        for invocation in self.invocations:
            if invocation.attachment_start != cursor:
                raise ValueError("eager attachment ranges must be contiguous")
            cursor += invocation.attachment_count
        if cursor != len(self.attachments):
            raise ValueError("eager attachment ranges must cover the attachment table")


@dataclass(frozen=True, slots=True)
class EagerStageSelectorDomains:
    """Domain references aligned one-for-one with one eager execution stage."""

    stage_index: int
    invocation_domains: tuple[EagerSelectorDomainIdRow, ...]
    attachment_domains: tuple[EagerSelectorDomainIdRow, ...]
    unpropagated_finalization_domains: tuple[EagerSelectorDomainIdRow, ...]
    propagated_finalization_domains: tuple[EagerSelectorDomainIdRow, ...]


@dataclass(frozen=True, slots=True)
class EagerSelectorClosureTables:
    """Interned coherent-group dependency domains for selector pruning."""

    domains: tuple[EagerSelectorDomainRow, ...]
    domain_group_ids: tuple[EagerSelectorGroupRow, ...]
    stages: tuple[EagerStageSelectorDomains, ...]
    closure_domains: tuple[EagerSelectorDomainIdRow, ...]

    def __post_init__(self) -> None:
        cursor = 0
        for domain_id, domain in enumerate(self.domains):
            if domain.member_start != cursor:
                raise ValueError(
                    f"eager selector domain {domain_id} does not start at {cursor}"
                )
            stop = cursor + domain.member_count
            if stop > len(self.domain_group_ids):
                raise ValueError(
                    f"eager selector domain {domain_id} exceeds its membership table"
                )
            previous = -1
            for member_index in range(cursor, stop):
                member = self.domain_group_ids[member_index].coherent_group_id
                if member <= previous:
                    raise ValueError(
                        f"eager selector domain {domain_id} members must be unique "
                        "and sorted"
                    )
                previous = member
            cursor = stop
        if cursor != len(self.domain_group_ids):
            raise ValueError(
                "eager selector domains do not cover their membership table"
            )

        domain_count = len(self.domains)
        for reference in self._domain_references():
            if reference.domain_id >= domain_count:
                raise ValueError(
                    f"eager selector domain ID {reference.domain_id} is out of range"
                )

    def _domain_references(self) -> Iterator[EagerSelectorDomainIdRow]:
        yield from self.closure_domains
        for stage in self.stages:
            for references in (
                stage.invocation_domains,
                stage.attachment_domains,
                stage.unpropagated_finalization_domains,
                stage.propagated_finalization_domains,
            ):
                yield from references

    def binary_payloads(self, *, prefix: str) -> dict[str, bytes]:
        payloads = {
            f"{prefix}/selector-domains.bin": pack_rows(self.domains),
            f"{prefix}/selector-domain-group-ids.bin": pack_rows(self.domain_group_ids),
            f"{prefix}/closure-domains.bin": pack_rows(self.closure_domains),
        }
        for stage in self.stages:
            base = f"{prefix}/stage-{stage.stage_index}"
            payloads[f"{base}-invocation-domains.bin"] = pack_rows(
                stage.invocation_domains
            )
            payloads[f"{base}-attachment-domains.bin"] = pack_rows(
                stage.attachment_domains
            )
            payloads[f"{base}-unpropagated-finalization-domains.bin"] = pack_rows(
                stage.unpropagated_finalization_domains
            )
            payloads[f"{base}-propagated-finalization-domains.bin"] = pack_rows(
                stage.propagated_finalization_domains
            )
        return payloads

    def to_metadata(self, *, prefix: str) -> dict[str, object]:
        def table(path: str, count: int, row_size: int) -> dict[str, object]:
            return {"path": path, "count": count, "row_size": row_size}

        return {
            "abi": EAGER_SELECTOR_DOMAINS_ABI,
            "domains": table(
                f"{prefix}/selector-domains.bin",
                len(self.domains),
                EagerSelectorDomainRow._STRUCT.size,
            ),
            "domain_group_ids": table(
                f"{prefix}/selector-domain-group-ids.bin",
                len(self.domain_group_ids),
                EagerSelectorGroupRow._STRUCT.size,
            ),
            "stages": [
                {
                    "stage_index": stage.stage_index,
                    "invocation_domains": table(
                        f"{prefix}/stage-{stage.stage_index}-invocation-domains.bin",
                        len(stage.invocation_domains),
                        EagerSelectorDomainIdRow._STRUCT.size,
                    ),
                    "attachment_domains": table(
                        f"{prefix}/stage-{stage.stage_index}-attachment-domains.bin",
                        len(stage.attachment_domains),
                        EagerSelectorDomainIdRow._STRUCT.size,
                    ),
                    "unpropagated_finalization_domains": table(
                        f"{prefix}/stage-{stage.stage_index}"
                        "-unpropagated-finalization-domains.bin",
                        len(stage.unpropagated_finalization_domains),
                        EagerSelectorDomainIdRow._STRUCT.size,
                    ),
                    "propagated_finalization_domains": table(
                        f"{prefix}/stage-{stage.stage_index}"
                        "-propagated-finalization-domains.bin",
                        len(stage.propagated_finalization_domains),
                        EagerSelectorDomainIdRow._STRUCT.size,
                    ),
                }
                for stage in self.stages
            ],
            "closure_domains": table(
                f"{prefix}/closure-domains.bin",
                len(self.closure_domains),
                EagerSelectorDomainIdRow._STRUCT.size,
            ),
        }


@dataclass(frozen=True, slots=True)
class EagerExecutionTables:
    process_key: str
    couplings: tuple[EagerCouplingRow, ...]
    stages: tuple[EagerStageTables, ...]
    closures: tuple[EagerClosureRow, ...]
    selector_closures: EagerSelectorClosureTables | None = None

    def __post_init__(self) -> None:
        selector_closures = self.selector_closures
        if selector_closures is None:
            return
        if len(selector_closures.stages) != len(self.stages):
            raise ValueError("eager selector domains do not cover every stage")
        for stage, domains in zip(
            self.stages,
            selector_closures.stages,
            strict=True,
        ):
            if domains.stage_index != stage.stage_index:
                raise ValueError("eager selector-domain stage index mismatch")
            expected_counts = (
                len(stage.invocations),
                len(stage.attachments),
                len(stage.finalizations),
                len(stage.finalizations),
            )
            actual_counts = (
                len(domains.invocation_domains),
                len(domains.attachment_domains),
                len(domains.unpropagated_finalization_domains),
                len(domains.propagated_finalization_domains),
            )
            if actual_counts != expected_counts:
                raise ValueError(
                    f"eager selector-domain row counts do not match stage "
                    f"{stage.stage_index}"
                )
        if len(selector_closures.closure_domains) != len(self.closures):
            raise ValueError("eager selector domains do not cover every closure")

    @property
    def invocation_count(self) -> int:
        return sum(len(stage.invocations) for stage in self.stages)

    @property
    def attachment_count(self) -> int:
        return sum(len(stage.attachments) for stage in self.stages)

    @property
    def referenced_kernel_ids(self) -> frozenset[int]:
        return frozenset(
            {
                *(row.kernel_id for stage in self.stages for row in stage.invocations),
                *(
                    row.kernel_id
                    for stage in self.stages
                    for row in stage.finalizations
                    if row.kernel_id != MISSING_U32
                ),
                *(
                    row.kernel_id
                    for row in self.closures
                    if row.kernel_id != MISSING_U32
                ),
            }
        )

    def binary_payloads(self, *, prefix: str = "eager") -> dict[str, bytes]:
        payloads = {f"{prefix}/couplings.bin": pack_rows(self.couplings)}
        for stage in self.stages:
            base = f"{prefix}/stage-{stage.stage_index}"
            payloads[f"{base}-invocations.bin"] = pack_rows(stage.invocations)
            payloads[f"{base}-attachments.bin"] = pack_rows(stage.attachments)
            payloads[f"{base}-finalizations.bin"] = pack_rows(stage.finalizations)
        payloads[f"{prefix}/closures.bin"] = pack_rows(self.closures)
        if self.selector_closures is not None:
            payloads.update(self.selector_closures.binary_payloads(prefix=prefix))
        return payloads

    def to_metadata(self, *, prefix: str = "eager") -> dict[str, object]:
        metadata: dict[str, object] = {
            "kind": EAGER_RUNTIME_KIND,
            "eager_plan_abi": EAGER_PLAN_ABI,
            "required_runtime_capabilities": [EAGER_RUNTIME_CAPABILITY],
            "process_key": self.process_key,
            "couplings": {
                "path": f"{prefix}/couplings.bin",
                "count": len(self.couplings),
                "row_size": EagerCouplingRow._STRUCT.size,
            },
            "stages": [
                {
                    "stage_index": stage.stage_index,
                    "subset_size": stage.subset_size,
                    "invocations": {
                        "path": (f"{prefix}/stage-{stage.stage_index}-invocations.bin"),
                        "count": len(stage.invocations),
                        "row_size": EagerInvocationRow._STRUCT.size,
                    },
                    "attachments": {
                        "path": (f"{prefix}/stage-{stage.stage_index}-attachments.bin"),
                        "count": len(stage.attachments),
                        "row_size": EagerAttachmentRow._STRUCT.size,
                    },
                    "finalizations": {
                        "path": (
                            f"{prefix}/stage-{stage.stage_index}-finalizations.bin"
                        ),
                        "count": len(stage.finalizations),
                        "row_size": EagerFinalizationRow._STRUCT.size,
                    },
                }
                for stage in self.stages
            ],
            "closures": {
                "path": f"{prefix}/closures.bin",
                "count": len(self.closures),
                "row_size": EagerClosureRow._STRUCT.size,
            },
        }
        if self.selector_closures is not None:
            metadata["selector_closures"] = self.selector_closures.to_metadata(
                prefix=prefix
            )
        return metadata


class _CouplingCatalog:
    def __init__(self, model_parameters: Sequence[Mapping[str, object]]) -> None:
        self.rows: list[EagerCouplingRow] = []
        self._row_ids: dict[EagerCouplingRow, int] = {}
        self._input_ids: dict[
            tuple[tuple[float, float], tuple[str | None, ...]],
            int,
        ] = {}
        self._direct = {
            str(record["name"]): int(record["parameter_index"])
            for record in model_parameters
        }
        logical: dict[str, dict[str, int]] = {}
        for record in model_parameters:
            runtime_name = record.get("runtime_name")
            component = record.get("complex_component")
            if isinstance(runtime_name, str) and component in {"real", "imag"}:
                logical.setdefault(runtime_name, {})[str(component)] = int(
                    record["parameter_index"]
                )
        self._logical = logical

    def add(
        self,
        coupling: Sequence[object],
        parameter_names: Sequence[object] | None,
    ) -> int:
        if len(coupling) != 2:
            raise ValueError("eager coupling metadata must contain two components")
        constants = (float(coupling[0]), float(coupling[1]))
        names = tuple(
            name if isinstance(name, str) else None for name in (parameter_names or ())
        )
        input_key = (constants, names)
        existing = self._input_ids.get(input_key)
        if existing is not None:
            return existing
        real_parameter = MISSING_U32
        imag_parameter = MISSING_U32

        first_name = names[0] if names else None
        if isinstance(first_name, str) and first_name in self._logical:
            components = self._logical[first_name]
            if set(components) != {"real", "imag"}:
                raise ValueError(
                    f"logical coupling {first_name!r} lacks real/imaginary slots"
                )
            real_parameter = components["real"]
            imag_parameter = components["imag"]
        else:
            for component, name in enumerate(names[:2]):
                if not isinstance(name, str):
                    continue
                parameter_id = self._direct.get(name)
                if parameter_id is None:
                    continue
                if component == 0:
                    real_parameter = parameter_id
                else:
                    imag_parameter = parameter_id

        row = EagerCouplingRow._from_trusted_values(
            real_parameter,
            imag_parameter,
            constants[0],
            constants[1],
        )
        existing = self._row_ids.get(row)
        if existing is not None:
            self._input_ids[input_key] = existing
            return existing
        row_id = len(self.rows)
        self.rows.append(row)
        self._row_ids[row] = row_id
        self._input_ids[input_key] = row_id
        return row_id


@dataclass(frozen=True, slots=True)
class _EagerStageLoweringInput:
    record: Mapping[str, object]
    evaluation_groups: Sequence[Sequence[InteractionNode]]
    input_value_slot_by_current: Mapping[int, int]
    output_current_ids: Sequence[int]


@dataclass(frozen=True, slots=True)
class _EagerLoweringInput:
    process_key: str
    value_slots: Mapping[int, Mapping[str, object]] | Sequence[Mapping[str, object]]
    value_slots_by_current_variant: Mapping[tuple[int, str], int]
    momentum_slot_by_mask: Mapping[int, int]
    model_parameters: Sequence[Mapping[str, object]]
    stages: Sequence[_EagerStageLoweringInput]
    amplitude_stage: Mapping[str, object]


def lower_fused_eager_execution(
    *,
    dag: GenericDAG,
    model: Model,
    resolver: EagerKernelResolver,
    process_id: str | None = None,
    progress_callback: EagerProgressCallback | None = None,
) -> tuple[dict[str, object], EagerExecutionTables]:
    """Assemble runtime metadata and eager tables from one owned layout."""

    if progress_callback is not None:
        progress_callback({"step": "runtime-layout"})
    layout = build_runtime_schema_layout(
        dag,
        model,
        process_id=process_id,
    )
    tables = _lower_eager_execution_tables(
        dag,
        model,
        resolver,
        _lowering_input_from_layout(layout),
        progress_callback=progress_callback,
    )
    return layout.runtime_schema, tables


def lower_eager_execution_tables(
    dag: GenericDAG,
    model: Model,
    runtime_schema: Mapping[str, object],
    resolver: EagerKernelResolver,
    progress_callback: EagerProgressCallback | None = None,
) -> EagerExecutionTables:
    """Lower a proven DAG without constructing any backend evaluator."""

    return _lower_eager_execution_tables(
        dag,
        model,
        resolver,
        _lowering_input_from_schema(dag, runtime_schema),
        progress_callback=progress_callback,
    )


def _lowering_input_from_schema(
    dag: GenericDAG,
    runtime_schema: Mapping[str, object],
) -> _EagerLoweringInput:
    value_slots = {
        int(slot["value_slot_id"]): slot
        for slot in _mapping_sequence(
            _mapping(runtime_schema.get("value_storage"), "value_storage").get(
                "value_slots"
            ),
            "value_storage.value_slots",
        )
    }
    value_slots_by_current_variant = {
        (int(slot["current_id"]), str(slot["variant"])): int(slot_id)
        for slot_id, slot in value_slots.items()
    }
    momentum_slot_by_mask = {
        int(slot["momentum_mask"]): int(slot["momentum_slot_id"])
        for slot in _mapping_sequence(
            runtime_schema.get("momentum_slots"),
            "momentum_slots",
        )
    }
    model_parameters = _mapping_sequence(
        runtime_schema.get("model_parameters"),
        "model_parameters",
    )
    stages: list[_EagerStageLoweringInput] = []
    for stage_record in _mapping_sequence(runtime_schema.get("stages"), "stages"):
        interactions = tuple(
            dag.interactions[int(value)]
            for value in _sequence(
                stage_record.get("interaction_ids"),
                "interaction_ids",
            )
        )
        groups: dict[tuple[str, int], list[InteractionNode]] = {}
        for interaction in interactions:
            group_key = (
                ("group", int(interaction.evaluation_group_id))
                if interaction.evaluation_group_id is not None
                else ("interaction", interaction.id)
            )
            groups.setdefault(group_key, []).append(interaction)
        input_value_slot_by_current = {
            int(value_slots[slot_id]["current_id"]): slot_id
            for slot_id in (
                int(value)
                for value in _sequence(
                    stage_record.get("input_value_slot_ids"),
                    "input_value_slot_ids",
                )
            )
        }
        stages.append(
            _EagerStageLoweringInput(
                record=stage_record,
                evaluation_groups=tuple(tuple(group) for group in groups.values()),
                input_value_slot_by_current=input_value_slot_by_current,
                output_current_ids=tuple(
                    int(value)
                    for value in _sequence(
                        stage_record.get("output_current_ids"),
                        "output_current_ids",
                    )
                ),
            )
        )
    return _EagerLoweringInput(
        process_key=str(runtime_schema.get("process_key", dag.process.key)),
        value_slots=value_slots,
        value_slots_by_current_variant=value_slots_by_current_variant,
        momentum_slot_by_mask=momentum_slot_by_mask,
        model_parameters=model_parameters,
        stages=tuple(stages),
        amplitude_stage=_mapping(
            runtime_schema.get("amplitude_stage"),
            "amplitude_stage",
        ),
    )


def _lowering_input_from_layout(
    layout: RuntimeSchemaLayout,
) -> _EagerLoweringInput:
    stages = tuple(
        _EagerStageLoweringInput(
            record=stage.record,
            evaluation_groups=stage.evaluation_groups,
            input_value_slot_by_current=stage.input_value_slot_by_current,
            output_current_ids=stage.output_current_ids,
        )
        for stage in layout.stages
    )
    return _EagerLoweringInput(
        process_key=str(layout.runtime_schema["process_key"]),
        value_slots=layout.value_slots_by_id,
        value_slots_by_current_variant=layout.value_slot_ids_by_current_variant,
        momentum_slot_by_mask=layout.momentum_slot_by_mask,
        model_parameters=layout.model_parameters,
        stages=stages,
        amplitude_stage=layout.amplitude_stage,
    )


def _lower_eager_execution_tables(
    dag: GenericDAG,
    model: Model,
    resolver: EagerKernelResolver,
    lowering: _EagerLoweringInput,
    *,
    progress_callback: EagerProgressCallback | None = None,
) -> EagerExecutionTables:
    value_slots = lowering.value_slots
    value_slots_by_current_variant = lowering.value_slots_by_current_variant
    momentum_slot_by_mask = lowering.momentum_slot_by_mask
    coupling_catalog = _CouplingCatalog(lowering.model_parameters)
    coupling_parameter_names: dict[
        tuple[int, tuple[int, ...], tuple[float, ...]],
        tuple[str | None, ...],
    ] = {}
    resolved_vertex_kernels: dict[
        tuple[
            int,
            tuple[int, int, int],
            int,
            int,
            int,
            tuple[float, float],
        ],
        EagerResolvedKernel,
    ] = {}
    propagator_kernel_ids: dict[tuple[int, int], int | None] = {}

    def parameter_names_for(interaction: InteractionNode) -> tuple[str | None, ...]:
        coupling_key = (
            interaction.vertex_kind,
            tuple(interaction.vertex_particles),
            tuple(interaction.coupling),
        )
        parameter_names = coupling_parameter_names.get(coupling_key)
        if parameter_names is None:
            parameter_names = tuple(
                runtime_coupling_parameter_names(
                    interaction.vertex_kind,
                    interaction.vertex_particles,
                    interaction.coupling,
                    model=model,
                )
            )
            coupling_parameter_names[coupling_key] = parameter_names
        return parameter_names

    def kernel_for(interaction: InteractionNode) -> EagerResolvedKernel:
        left = dag.currents[interaction.left_id]
        right = dag.currents[interaction.right_id]
        result = dag.currents[interaction.result_id]
        key = (
            int(interaction.vertex_kind),
            interaction.vertex_particles,
            int(left.index.chirality),
            int(right.index.chirality),
            int(result.index.chirality),
            interaction.coupling,
        )
        resolved = resolved_vertex_kernels.get(key)
        if resolved is None:
            resolved = resolver.vertex_kernel(interaction)
            resolved_vertex_kernels[key] = resolved
        return resolved

    stages: list[EagerStageTables] = []

    stage_total = len(lowering.stages)
    for stage_position, stage_input in enumerate(lowering.stages, start=1):
        stage_record = stage_input.record
        stage_index = int(stage_record["stage_index"])
        subset_size = int(stage_record["subset_size"])
        input_slot_by_current = stage_input.input_value_slot_by_current

        invocations: list[EagerInvocationRow] = []
        attachments: list[EagerAttachmentRow] = []
        for interactions in stage_input.evaluation_groups:
            live_interactions = tuple(
                interaction
                for interaction in interactions
                if complex(*interaction.evaluation_factor) != 0j
            )
            if not live_interactions:
                continue
            representative = live_interactions[0]
            resolved_kernel = kernel_for(representative)
            representative_factor = complex(*representative.evaluation_factor)
            parameter_names = parameter_names_for(representative)
            coupling_slot_id = coupling_catalog.add(
                representative.coupling,
                parameter_names,
            )
            attachment_start = len(attachments)
            group_factor = (
                complex(*resolved_kernel.normalization_factor)
                / representative_factor
            )
            for interaction in live_interactions:
                # The compiler's evaluation-group key already proves identical
                # kernel, coupling, and mutable-parameter provenance. Repeating
                # those resolver lookups for every fan-out attachment is pure
                # lowering overhead.
                factor = (
                    complex(*interaction.color_weight)
                    * complex(*interaction.evaluation_factor)
                    * group_factor
                )
                attachments.append(
                    EagerAttachmentRow._from_trusted_values(
                        interaction.result_id,
                        factor.real,
                        factor.imag,
                    )
                )
            left = dag.currents[representative.left_id]
            right = dag.currents[representative.right_id]
            input_currents = (left, right)
            ordered_currents = tuple(
                input_currents[index] for index in resolved_kernel.canonical_input_order
            )
            invocations.append(
                EagerInvocationRow._from_trusted_values(
                    resolved_kernel.kernel_id,
                    input_slot_by_current[ordered_currents[0].id],
                    input_slot_by_current[ordered_currents[1].id],
                    momentum_slot_by_mask[ordered_currents[0].index.momentum_mask],
                    momentum_slot_by_mask[ordered_currents[1].index.momentum_mask],
                    coupling_slot_id,
                    resolved_kernel.output_factor_source,
                    attachment_start,
                    len(live_interactions),
                )
            )

        finalizations = tuple(
            _finalization_row(
                dag.currents[int(current_id)],
                value_slots,
                value_slots_by_current_variant,
                momentum_slot_by_mask,
                resolver,
                propagator_kernel_ids,
            )
            for current_id in stage_input.output_current_ids
        )
        stages.append(
            EagerStageTables(
                stage_index=stage_index,
                subset_size=subset_size,
                invocations=tuple(invocations),
                attachments=tuple(attachments),
                finalizations=finalizations,
            )
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "step": "invocation lowering",
                    "stage_index": stage_position,
                    "stage_total": stage_total,
                    "subset_size": subset_size,
                    "invocation_count": len(invocations),
                    "attachment_count": len(attachments),
                    "finalization_count": len(finalizations),
                }
            )

    amplitude_stage = lowering.amplitude_stage
    if progress_callback is not None:
        progress_callback(
            {
                "step": "amplitude closures",
                "stage_index": stage_total,
                "stage_total": stage_total,
            }
        )
    closures: list[EagerClosureRow] = []
    for root in _mapping_sequence(
        amplitude_stage.get("roots"), "amplitude_stage.roots"
    ):
        closure_kernel = resolver.closure_kernel(root)
        coupling_slot_id = MISSING_U32
        if closure_kernel is not None:
            coupling_slot_id = coupling_catalog.add(
                _sequence(root.get("coupling"), "root.coupling"),
                _optional_sequence(root.get("coupling_parameter_names")),
            )
        color_weight = _complex_pair(root.get("color_weight"), "root.color_weight")
        left_slot_id = int(
            _mapping(root.get("left_value_slot"), "left_value_slot")["value_slot_id"]
        )
        right_slot_id = int(
            _mapping(root.get("right_value_slot"), "right_value_slot")["value_slot_id"]
        )
        if closure_kernel is not None:
            slots = (left_slot_id, right_slot_id)
            left_slot_id, right_slot_id = (
                slots[index] for index in closure_kernel.canonical_input_order
            )
            color_weight *= complex(*closure_kernel.normalization_factor)
        closures.append(
            EagerClosureRow._from_trusted_values(
                MISSING_U32 if closure_kernel is None else closure_kernel.kernel_id,
                left_slot_id,
                right_slot_id,
                int(root["output_index"]),
                coupling_slot_id,
                (
                    EAGER_OUTPUT_FACTOR_NONE
                    if closure_kernel is None
                    else closure_kernel.output_factor_source
                ),
                color_weight.real,
                color_weight.imag,
            )
        )

    stage_tables = tuple(stages)
    closure_rows = tuple(closures)
    if progress_callback is not None:
        progress_callback(
            {
                "step": "selector closures",
                "stage_index": stage_total,
                "stage_total": stage_total,
                "closure_count": len(closure_rows),
            }
        )
    selector_closures = _build_selector_closure_tables(
        stage_tables,
        closure_rows,
        amplitude_stage,
        value_slot_count=len(value_slots),
        current_count=len(dag.currents),
    )
    if progress_callback is not None:
        progress_callback(
            {
                "step": "eager plan ready",
                "stage_index": stage_total,
                "stage_total": stage_total,
                "closure_count": len(closure_rows),
            }
        )
    return EagerExecutionTables(
        process_key=lowering.process_key,
        couplings=tuple(coupling_catalog.rows),
        stages=stage_tables,
        closures=closure_rows,
        selector_closures=selector_closures,
    )


def _build_selector_closure_tables(
    stages: tuple[EagerStageTables, ...],
    closures: tuple[EagerClosureRow, ...],
    amplitude_stage: Mapping[str, object],
    *,
    value_slot_count: int,
    current_count: int,
) -> EagerSelectorClosureTables:
    """Propagate coherent-group dependencies backwards through eager rows."""

    roots = _mapping_sequence(amplitude_stage.get("roots"), "amplitude_stage.roots")
    if len(roots) != len(closures):
        raise ValueError("eager closures do not match runtime amplitude roots")

    declared_groups = tuple(
        int(group["group_id"])
        for group in _mapping_sequence(
            amplitude_stage.get("coherent_groups"),
            "amplitude_stage.coherent_groups",
        )
    )
    if len(declared_groups) != len(set(declared_groups)) or any(
        group_id < 0 or group_id > MISSING_U32 for group_id in declared_groups
    ):
        raise ValueError("runtime coherent-group IDs must be unique unsigned integers")
    ordered_group_ids = tuple(sorted(declared_groups))
    group_bit_indices = {
        group_id: bit_index for bit_index, group_id in enumerate(ordered_group_ids)
    }

    dense_mask_interning = (
        len(ordered_group_ids) <= _DENSE_SELECTOR_DOMAIN_GROUP_LIMIT
    )
    domain_masks: list[int] = []
    domain_ids_by_mask: dict[int, int] = {}
    domain_members: list[tuple[int, ...]] = []
    domain_ids_by_members: dict[tuple[int, ...], int] = {}

    def intern_members(members: tuple[int, ...]) -> int:
        existing = domain_ids_by_members.get(members)
        if existing is not None:
            return existing
        domain_id = len(domain_members)
        domain_members.append(members)
        domain_ids_by_members[members] = domain_id
        return domain_id

    def members(domain: int) -> tuple[int, ...]:
        result: list[int] = []
        remaining = domain
        while remaining:
            lowest = remaining & -remaining
            result.append(ordered_group_ids[lowest.bit_length() - 1])
            remaining ^= lowest
        return tuple(result)

    def intern_mask(domain: int) -> int:
        existing = domain_ids_by_mask.get(domain)
        if existing is not None:
            return existing
        domain_id = len(domain_masks)
        domain_masks.append(domain)
        domain_ids_by_mask[domain] = domain_id
        return domain_id

    def intern_stage_mask(
        domain: int,
        stage_domain_ids_by_mask: dict[int, int],
    ) -> int:
        if dense_mask_interning:
            return intern_mask(domain)
        existing = stage_domain_ids_by_mask.get(domain)
        if existing is not None:
            return existing
        domain_id = intern_members(members(domain))
        stage_domain_ids_by_mask[domain] = domain_id
        return domain_id

    value_domains = [0] * value_slot_count
    empty_domain_id = intern_mask(0) if dense_mask_interning else intern_members(())
    closure_domain_ids: list[int] = []
    for closure, root in zip(closures, roots, strict=True):
        if closure.amplitude_index != int(root["output_index"]):
            raise ValueError(
                "eager closure order does not match runtime amplitude roots"
            )
        group_id = int(root["coherent_group_id"])
        bit_index = group_bit_indices.get(group_id)
        if bit_index is None:
            raise ValueError(
                f"amplitude root references undeclared coherent group {group_id}"
            )
        domain = 1 << bit_index
        closure_domain_ids.append(
            intern_mask(domain)
            if dense_mask_interning
            else intern_members((group_id,))
        )
        value_domains[closure.left_value_slot_id] |= domain
        value_domains[closure.right_value_slot_id] |= domain

    stage_domain_ids_reversed: list[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
        ],
    ] = []
    current_domains = [0] * current_count
    current_domain_ids = [empty_domain_id] * current_count

    for stage in reversed(stages):
        stage_domain_ids_by_mask = {0: empty_domain_id}

        unpropagated_domain_ids: list[int] = []
        propagated_domain_ids: list[int] = []
        for finalization in stage.finalizations:
            unpropagated = (
                value_domains[finalization.unpropagated_value_slot_id]
                if finalization.stores_unpropagated
                else 0
            )
            propagated = (
                value_domains[finalization.propagated_value_slot_id]
                if finalization.stores_propagated
                else 0
            )
            unpropagated_domain_ids.append(
                intern_stage_mask(unpropagated, stage_domain_ids_by_mask)
            )
            propagated_domain_ids.append(
                intern_stage_mask(propagated, stage_domain_ids_by_mask)
            )
            current_domain = unpropagated | propagated
            current_domains[finalization.current_id] = current_domain
            current_domain_ids[finalization.current_id] = intern_stage_mask(
                current_domain,
                stage_domain_ids_by_mask,
            )
            if finalization.stores_unpropagated:
                value_domains[finalization.unpropagated_value_slot_id] = 0
            if finalization.stores_propagated:
                value_domains[finalization.propagated_value_slot_id] = 0

        attachment_domain_ids = [
            current_domain_ids[attachment.result_current_id]
            for attachment in stage.attachments
        ]

        invocation_domain_ids: list[int] = []
        for invocation in stage.invocations:
            start = invocation.attachment_start
            stop = start + invocation.attachment_count
            domain = 0
            for attachment_index in range(start, stop):
                result_current_id = stage.attachments[
                    attachment_index
                ].result_current_id
                domain |= current_domains[result_current_id]
            invocation_domain_ids.append(
                intern_stage_mask(domain, stage_domain_ids_by_mask)
            )
            value_domains[invocation.left_value_slot_id] |= domain
            value_domains[invocation.right_value_slot_id] |= domain

        stage_domain_ids_reversed.append(
            (
                tuple(invocation_domain_ids),
                tuple(attachment_domain_ids),
                tuple(unpropagated_domain_ids),
                tuple(propagated_domain_ids),
            )
        )
        for finalization in stage.finalizations:
            current_domains[finalization.current_id] = 0
            current_domain_ids[finalization.current_id] = empty_domain_id
        stage_domain_ids_by_mask.clear()

    # Construction-only masks and interning indexes do not participate in final
    # ID assignment or emission.
    value_domains.clear()
    current_domains.clear()
    current_domain_ids.clear()
    group_bit_indices.clear()

    # Match the prior dense-mask ordering exactly. For equal-cardinality masks,
    # integer comparison examines the highest differing bit first, equivalent
    # to lexicographically comparing the reversed, sorted member tuples.
    if dense_mask_interning:
        ordered_provisional_ids = tuple(
            sorted(
                range(len(domain_masks)),
                key=lambda domain_id: (
                    domain_masks[domain_id].bit_count(),
                    domain_masks[domain_id],
                ),
            )
        )
    else:
        ordered_provisional_ids = tuple(
            sorted(
                range(len(domain_members)),
                key=lambda domain_id: (
                    len(domain_members[domain_id]),
                    tuple(reversed(domain_members[domain_id])),
                ),
            )
        )
    final_domain_ids = [0] * len(ordered_provisional_ids)
    for final_domain_id, provisional_domain_id in enumerate(ordered_provisional_ids):
        final_domain_ids[provisional_domain_id] = final_domain_id

    domain_rows: list[EagerSelectorDomainRow] = []
    group_rows: list[EagerSelectorGroupRow] = []
    group_rows_by_bit_index = tuple(
        EagerSelectorGroupRow._from_trusted_values(group_id)
        for group_id in ordered_group_ids
    )
    group_row_by_id = (
        {}
        if dense_mask_interning
        else {
            group_id: group_rows_by_bit_index[bit_index]
            for bit_index, group_id in enumerate(ordered_group_ids)
        }
    )
    for provisional_domain_id in ordered_provisional_ids:
        if dense_mask_interning:
            domain = domain_masks[provisional_domain_id]
            member_count = domain.bit_count()
        else:
            resolved_members = domain_members[provisional_domain_id]
            member_count = len(resolved_members)
        domain_rows.append(
            EagerSelectorDomainRow._from_trusted_values(
                len(group_rows), member_count
            )
        )
        if dense_mask_interning:
            while domain:
                lowest = domain & -domain
                group_rows.append(
                    group_rows_by_bit_index[lowest.bit_length() - 1]
                )
                domain ^= lowest
        else:
            group_rows.extend(
                group_row_by_id[group_id] for group_id in resolved_members
            )

    domain_count = len(ordered_provisional_ids)
    domain_masks.clear()
    domain_ids_by_mask.clear()
    domain_members.clear()
    domain_ids_by_members.clear()
    group_row_by_id.clear()
    domain_reference_rows = tuple(
        EagerSelectorDomainIdRow._from_trusted_values(domain_id)
        for domain_id in range(domain_count)
    )

    def references(
        provisional_domain_ids: Sequence[int],
    ) -> tuple[EagerSelectorDomainIdRow, ...]:
        return tuple(
            domain_reference_rows[final_domain_ids[provisional_domain_id]]
            for provisional_domain_id in provisional_domain_ids
        )

    selector_stages = []
    for stage, domain_ids in zip(
        stages,
        reversed(stage_domain_ids_reversed),
        strict=True,
    ):
        (
            stage_invocation_domain_ids,
            stage_attachment_domain_ids,
            stage_unpropagated_domain_ids,
            stage_propagated_domain_ids,
        ) = domain_ids
        selector_stages.append(
            EagerStageSelectorDomains(
                stage_index=stage.stage_index,
                invocation_domains=references(stage_invocation_domain_ids),
                attachment_domains=references(stage_attachment_domain_ids),
                unpropagated_finalization_domains=references(
                    stage_unpropagated_domain_ids
                ),
                propagated_finalization_domains=references(stage_propagated_domain_ids),
            )
        )

    return EagerSelectorClosureTables(
        domains=tuple(domain_rows),
        domain_group_ids=tuple(group_rows),
        stages=tuple(selector_stages),
        closure_domains=references(closure_domain_ids),
    )


def _finalization_row(
    current: CurrentNode,
    value_slots: Mapping[int, Mapping[str, object]] | Sequence[Mapping[str, object]],
    value_slots_by_current_variant: Mapping[tuple[int, str], int],
    momentum_slot_by_mask: Mapping[int, int],
    resolver: EagerKernelResolver,
    propagator_kernel_ids: dict[tuple[int, int], int | None],
) -> EagerFinalizationRow:
    unpropagated = value_slots_by_current_variant.get(
        (current.id, "unpropagated"),
        MISSING_U32,
    )
    propagated = value_slots_by_current_variant.get(
        (current.id, "propagated"),
        MISSING_U32,
    )
    if unpropagated == MISSING_U32 and propagated == MISSING_U32:
        raise ValueError(f"eager current {current.id} has no output value slot")
    propagator_slot_id = propagated if propagated != MISSING_U32 else unpropagated
    output_slot = value_slots[propagator_slot_id]
    propagator = _mapping(
        output_slot.get("propagator"),
        f"current {current.id} propagator",
    )
    propagator_key = (current.index.particle_id, current.index.chirality)
    if not bool(output_slot.get("applies_propagator", False)):
        kernel_id = None
    elif propagator_key in propagator_kernel_ids:
        kernel_id = propagator_kernel_ids[propagator_key]
    else:
        kernel_id = resolver.propagator_kernel_id(current, propagator)
        propagator_kernel_ids[propagator_key] = kernel_id
    if propagated != MISSING_U32 and kernel_id is None:
        raise ValueError(
            f"eager propagated current {current.id} has no prepared propagator kernel"
        )
    return EagerFinalizationRow._from_trusted_values(
        MISSING_U32 if kernel_id is None else kernel_id,
        current.id,
        unpropagated,
        propagated,
        momentum_slot_by_mask[current.index.momentum_mask],
    )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be a sequence")
    return value


def _optional_sequence(value: object) -> Sequence[object] | None:
    if value is None:
        return None
    return _sequence(value, "optional sequence")


def _mapping_sequence(value: object, context: str) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _mapping(item, f"{context}[{index}]")
        for index, item in enumerate(_sequence(value, context))
    )


def _complex_pair(value: object, context: str) -> complex:
    components = _sequence(value, context)
    if len(components) != 2:
        raise ValueError(f"{context} must contain two components")
    return complex(float(components[0]), float(components[1]))


def _resolved_kernel(
    record: Mapping[str, object],
    context: str,
) -> EagerResolvedKernel:
    order = tuple(
        int(value)
        for value in _sequence(
            record.get("canonical_input_order"),
            f"{context}.canonical_input_order",
        )
    )
    factor = tuple(
        float(value)
        for value in _sequence(
            record.get("equivalence_factor"),
            f"{context}.equivalence_factor",
        )
    )
    if len(order) != 2 or len(factor) != 2:
        raise ValueError(f"{context} has malformed input transformation metadata")
    factor_source = {
        "none": EAGER_OUTPUT_FACTOR_NONE,
        "coupling-real": EAGER_OUTPUT_FACTOR_COUPLING_REAL,
        "coupling-imag": EAGER_OUTPUT_FACTOR_COUPLING_IMAG,
    }.get(str(record.get("output_factor_source", "none")))
    if factor_source is None:
        raise ValueError(f"{context} has an unsupported output factor source")
    return EagerResolvedKernel(
        kernel_id=int(record["kernel_id"]),
        canonical_input_order=(order[0], order[1]),
        normalization_factor=(factor[0], factor[1]),
        output_factor_source=factor_source,
    )


def _vertex_kernel_key(record: Mapping[str, object]) -> VertexKernelKey:
    particles = tuple(
        int(value) for value in _sequence(record.get("particles"), "vertex.particles")
    )
    coupling = tuple(
        float(value) for value in _sequence(record.get("coupling"), "vertex.coupling")
    )
    if len(particles) != 3 or len(coupling) != 2:
        raise ValueError("prepared vertex key has malformed particles or coupling")
    return VertexKernelKey(
        kind=int(record["kind"]),
        particles=(particles[0], particles[1], particles[2]),
        left_chirality=int(record["left_chirality"]),
        right_chirality=int(record["right_chirality"]),
        result_chirality=int(record["result_chirality"]),
        coupling=(coupling[0], coupling[1]),
    )


def _closure_kernel_key(record: Mapping[str, object]) -> ClosureKernelKey:
    particles = tuple(
        int(value) for value in _sequence(record.get("particles"), "closure.particles")
    )
    coupling = tuple(
        float(value) for value in _sequence(record.get("coupling"), "closure.coupling")
    )
    if len(particles) != 3 or len(coupling) != 2:
        raise ValueError("prepared closure key has malformed particles or coupling")
    return ClosureKernelKey(
        kind=int(record["kind"]),
        particles=(particles[0], particles[1], particles[2]),
        left_chirality=int(record["left_chirality"]),
        right_chirality=int(record["right_chirality"]),
        coupling=(coupling[0], coupling[1]),
    )


__all__ = [
    "EAGER_RUNTIME_KIND",
    "EagerExecutionTables",
    "EagerKernelResolver",
    "EagerResolvedKernel",
    "EagerSelectorClosureTables",
    "EagerStageSelectorDomains",
    "EagerStageTables",
    "MappingEagerKernelResolver",
    "PreparedCatalogEagerKernelIndex",
    "PreparedCatalogEagerKernelResolver",
    "lower_eager_execution_tables",
    "lower_fused_eager_execution",
]
