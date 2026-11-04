# SPDX-License-Identifier: 0BSD
"""Typed adapter for authenticated recurrence plan-v2 exact sections."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar, cast

from pyamplicol._internal.versions import verify_native_module
from pyamplicol.api.errors import ArtifactError, CompatibilityError
from pyamplicol.runtime.eager_exact._contracts import (
    _integer,
    _mapping,
    _sequence,
)

RECURRENCE_EXACT_SECTIONS_ABI = "pyamplicol-recurrence-exact-sections-v1"
RECURRENCE_PLAN_V2_ABI = "pyamplicol-recurrence-plan-v2"
RECURRENCE_RUNTIME_LAYOUT_V2_ABI = "pyamplicol-recurrence-runtime-layout-v2"
RECURRENCE_RUNTIME_KIND = "pyamplicol-runtime-recurrence-execution"
RECURRENCE_DIRECT_RUNTIME_CAPABILITY = "rusticol.recurrence-direct-arena.complex-f64.v1"
DIRECT_NONE_U32 = (1 << 32) - 1
_NATIVE_BINDING_NAME = "_load_recurrence_exact_sections_v1"


class _NativeExactSectionsBinding(Protocol):
    def __call__(self, artifact_root: str, process_id: str, /) -> object: ...


_NativeExactSectionsLoader = Callable[[Path, str], object]
_RowT = TypeVar("_RowT")


@dataclass(frozen=True, slots=True)
class _ExactFactor:
    real_numerator: int
    real_denominator: int
    imaginary_numerator: int
    imaginary_denominator: int


@dataclass(frozen=True, slots=True)
class _Current:
    semantic_id: int
    node_kind: int
    state_template_id: int
    component_base: int
    component_count: int
    momentum_form_id: int
    stage: int
    selector_domain_id: int
    first_use: int
    last_use: int
    source_row: int
    finalization_row: int


@dataclass(frozen=True, slots=True)
class _Source:
    source_slot: int
    destination_base: int
    momentum_form_id: int
    source_template_or_dispatch_domain: int
    spin_state_class: int
    exact_factor_id: int
    selector_domain_id: int


@dataclass(frozen=True, slots=True)
class _Contribution:
    parent0_base: int
    parent1_base: int
    parent0_momentum: int
    parent1_momentum: int
    destination_base: int
    exact_factor_id: int
    selector_domain_id: int
    flags: int


@dataclass(frozen=True, slots=True)
class _Finalization:
    component_base: int
    component_count: int
    momentum_form_id: int
    exact_factor_id: int
    selector_domain_id: int
    flags: int


@dataclass(frozen=True, slots=True)
class _Closure:
    parent0_base: int
    parent1_base: int
    parent0_momentum: int
    parent1_momentum: int
    amplitude_destination_id: int
    exact_factor_id: int
    component_factor_start: int
    component_count: int
    selector_domain_id: int
    flags: int


@dataclass(frozen=True, slots=True)
class _RowGroup:
    stage: int
    role: int
    destination_operation: int
    executor_id: int
    row_start: int
    row_count: int


@dataclass(frozen=True, slots=True)
class _MomentumForm:
    term_start: int
    term_count: int


@dataclass(frozen=True, slots=True)
class _MomentumTerm:
    source_slot: int
    coefficient: int


@dataclass(frozen=True, slots=True)
class _ReplayTarget:
    public_flow_id: int
    representative_id: int
    source_permutation_start: int
    source_permutation_count: int
    helicity_map_start: int
    helicity_map_count: int
    phase_factor_id: int
    multiplicity: int
    selector_domain_id: int


@dataclass(frozen=True, slots=True)
class _AmplitudeDestination:
    closure_row_start: int
    destination_id: int
    target_sector_id: int
    target_helicity_id: int
    closure_row_count: int
    selector_domain_id: int


@dataclass(frozen=True, slots=True)
class _ResolvedHelicity:
    source_state_start: int
    source_selection_start: int
    public_helicity_start: int
    helicity_id: int
    source_state_count: int
    source_selection_count: int
    public_helicity_count: int
    selector_domain_id: int


@dataclass(frozen=True, slots=True)
class _SourceStateAssignment:
    source_slot: int
    state_index: int


@dataclass(frozen=True, slots=True)
class _SourceDispatchVariant:
    embedding_start: int
    projection_start: int
    source_row_id: int
    dispatch_domain_id: int
    runtime_variant_id: int
    source_state_index: int
    source_template_id: int
    source_state_template_id: int
    crossed_state_template_id: int
    crossed_spin_state_class: int
    executor_id: int
    crossing_exact_factor_id: int
    embedding_count: int
    projection_count: int


@dataclass(frozen=True, slots=True)
class _SourceEmbedding:
    full_component: int
    source_component: int
    exact_factor_id: int


@dataclass(frozen=True, slots=True)
class _SourceProjection:
    source_component: int
    full_component: int


@dataclass(frozen=True, slots=True)
class _ResolvedSourceSelection:
    source_slot: int
    dispatch_variant_id: int


@dataclass(frozen=True, slots=True)
class _Executor:
    executor_id: int
    role: str
    destination_operation: str
    parent_component_counts: tuple[int, ...]
    destination_component_count: int
    momentum_operand_count: int
    prepared_kernel_id: int | None
    runtime_template: str | None


@dataclass(frozen=True, slots=True)
class _RecurrenceExactSectionsV1:
    process_id: str
    strategy: str
    semantic_digest: str
    runtime_layout_digest: str
    current_arena_components: int
    amplitude_destination_count: int
    parameter_value_count: int
    external_source_count: int
    currents: tuple[_Current, ...]
    sources: tuple[_Source, ...]
    contributions: tuple[_Contribution, ...]
    finalizations: tuple[_Finalization, ...]
    closures: tuple[_Closure, ...]
    row_groups: tuple[_RowGroup, ...]
    momentum_forms: tuple[_MomentumForm, ...]
    momentum_terms: tuple[_MomentumTerm, ...]
    replay_targets: tuple[_ReplayTarget, ...]
    source_permutations: tuple[int, ...]
    replay_momentum_signs: tuple[int, ...]
    replay_helicity_map: tuple[int, ...]
    amplitude_destinations: tuple[_AmplitudeDestination, ...]
    resolved_helicities: tuple[_ResolvedHelicity, ...]
    source_state_assignments: tuple[_SourceStateAssignment, ...]
    source_dispatch_variants: tuple[_SourceDispatchVariant, ...]
    source_embeddings: tuple[_SourceEmbedding, ...]
    source_projections: tuple[_SourceProjection, ...]
    resolved_source_selections: tuple[_ResolvedSourceSelection, ...]
    public_helicities: tuple[int, ...]
    exact_factors: tuple[_ExactFactor, ...]
    public_flow_ids: tuple[int, ...]
    executors: tuple[_Executor, ...]


def _load_recurrence_exact_sections_v1(
    artifact_root: Path,
    process_id: str,
    *,
    loader: _NativeExactSectionsLoader | None = None,
) -> _RecurrenceExactSectionsV1:
    raw = (loader or _native_exact_sections_loader)(artifact_root, process_id)
    return _parse_exact_sections(raw, process_id)


def _native_exact_sections_loader(artifact_root: Path, process_id: str) -> object:
    try:
        module = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(module)
    except ImportError as exc:
        raise CompatibilityError(
            "compact recurrence exact execution requires pyamplicol._rusticol"
        ) from exc
    candidate = getattr(module, _NATIVE_BINDING_NAME, None)
    if not callable(candidate):
        raise CompatibilityError(
            "compact recurrence exact execution requires the private native binding "
            f"{_NATIVE_BINDING_NAME}"
        )
    binding = cast(_NativeExactSectionsBinding, candidate)
    try:
        return binding(os.fspath(artifact_root), process_id)
    except (ArtifactError, CompatibilityError):
        raise
    except Exception as exc:
        raise ArtifactError(
            "could not load recurrence exact sections for process "
            f"{process_id!r}: {exc}"
        ) from exc


def _parse_exact_sections(
    raw: object,
    process_id: str,
) -> _RecurrenceExactSectionsV1:
    root = _mapping(raw, "recurrence exact sections")
    if root.get("abi") != RECURRENCE_EXACT_SECTIONS_ABI:
        raise CompatibilityError(
            f"unsupported recurrence exact-sections ABI {root.get('abi')!r}"
        )
    if root.get("runtime_layout_abi") != RECURRENCE_RUNTIME_LAYOUT_V2_ABI:
        raise CompatibilityError(
            f"unsupported recurrence runtime-layout ABI "
            f"{root.get('runtime_layout_abi')!r}"
        )
    if root.get("process_id") != process_id:
        raise ArtifactError("recurrence exact sections select the wrong process")
    strategy = root.get("strategy")
    if strategy not in {
        "topology-replay",
        "all-flow-union",
        "contracted-color-union",
    }:
        raise CompatibilityError(f"unsupported exact recurrence strategy {strategy!r}")
    counts = _row(root.get("counts"), 4, "recurrence exact counts")
    sections = _RecurrenceExactSectionsV1(
        process_id=process_id,
        strategy=strategy,
        semantic_digest=_digest(root.get("semantic_digest"), "semantic digest"),
        runtime_layout_digest=_digest(
            root.get("runtime_layout_digest"), "runtime-layout digest"
        ),
        current_arena_components=_integer(counts[0], "current arena count", minimum=1),
        amplitude_destination_count=_integer(
            counts[1], "amplitude destination count", minimum=1
        ),
        parameter_value_count=_integer(counts[2], "parameter count"),
        external_source_count=_integer(counts[3], "external source count", minimum=1),
        currents=_rows(root, "currents", 12, _Current),
        sources=_rows(root, "sources", 7, _Source, signed_fields={4}),
        contributions=_rows(root, "contributions", 8, _Contribution),
        finalizations=_rows(root, "finalizations", 6, _Finalization),
        closures=_rows(root, "closures", 10, _Closure),
        row_groups=_rows(root, "row_groups", 6, _RowGroup),
        momentum_forms=_rows(root, "momentum_forms", 2, _MomentumForm),
        momentum_terms=_rows(
            root, "momentum_terms", 2, _MomentumTerm, signed_fields={1}
        ),
        replay_targets=_rows(root, "replay_targets", 9, _ReplayTarget),
        source_permutations=tuple(
            _integer(value, f"source permutation {index}")
            for index, value in enumerate(
                _sequence(root.get("source_permutations"), "source permutations")
            )
        ),
        replay_momentum_signs=tuple(
            _integer(value, f"replay momentum sign {index}", minimum=-1)
            for index, value in enumerate(
                _sequence(
                    root.get("replay_momentum_signs"),
                    "replay momentum signs",
                )
            )
        ),
        replay_helicity_map=tuple(
            _integer(value, f"replay helicity map {index}")
            for index, value in enumerate(
                _sequence(root.get("replay_helicity_map"), "replay helicity map")
            )
        ),
        amplitude_destinations=_rows(
            root, "amplitude_destinations", 6, _AmplitudeDestination
        ),
        resolved_helicities=_rows(root, "resolved_helicities", 8, _ResolvedHelicity),
        source_state_assignments=_rows(
            root,
            "source_state_assignments",
            2,
            _SourceStateAssignment,
        ),
        source_dispatch_variants=_rows(
            root,
            "source_dispatch_variants",
            14,
            _SourceDispatchVariant,
            signed_fields={9},
        ),
        source_embeddings=_rows(root, "source_embeddings", 3, _SourceEmbedding),
        source_projections=_rows(root, "source_projections", 2, _SourceProjection),
        resolved_source_selections=_rows(
            root,
            "resolved_source_selections",
            2,
            _ResolvedSourceSelection,
        ),
        public_helicities=tuple(
            _signed_integer(value, f"public helicity {index}")
            for index, value in enumerate(
                _sequence(root.get("public_helicities"), "public helicities")
            )
        ),
        exact_factors=tuple(
            _parse_factor(row, index)
            for index, row in enumerate(
                _sequence(root.get("exact_factors"), "exact factors")
            )
        ),
        public_flow_ids=tuple(
            _integer(value, f"public flow {index}")
            for index, value in enumerate(
                _sequence(root.get("public_flow_ids"), "public flow IDs")
            )
        ),
        executors=tuple(
            _parse_executor(row, index)
            for index, row in enumerate(
                _sequence(root.get("executors"), "direct executors")
            )
        ),
    )
    _validate_sections(sections)
    return sections


def _row(raw: object, width: int, context: str) -> Sequence[object]:
    values = _sequence(raw, context)
    if len(values) != width:
        raise ArtifactError(f"{context} must contain {width} fields")
    return values


def _rows(
    root: Mapping[str, object],
    name: str,
    width: int,
    row_type: Callable[..., _RowT],
    *,
    signed_fields: set[int] | None = None,
) -> tuple[_RowT, ...]:
    signed = signed_fields or set()
    parsed = []
    for index, raw in enumerate(_sequence(root.get(name), f"recurrence {name}")):
        values = _row(raw, width, f"recurrence {name}[{index}]")
        parsed.append(
            row_type(
                *(
                    _signed_integer(value, f"{name}[{index}][{field}]")
                    if field in signed
                    else _integer(value, f"{name}[{index}][{field}]")
                    for field, value in enumerate(values)
                )
            )
        )
    return tuple(parsed)


def _signed_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{context} must be an integer")
    return value


def _decimal_integer(value: object, context: str) -> int:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{context} must be a decimal integer string")
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise ArtifactError(f"{context} must be a decimal integer string") from exc
    if str(parsed) != value:
        raise ArtifactError(f"{context} must use canonical decimal syntax")
    return parsed


def _parse_factor(raw: object, index: int) -> _ExactFactor:
    values = _row(raw, 4, f"exact factor {index}")
    factor = _ExactFactor(
        *(
            _decimal_integer(value, f"exact factor {index} field {field}")
            for field, value in enumerate(values)
        )
    )
    if factor.real_denominator <= 0 or factor.imaginary_denominator <= 0:
        raise ArtifactError(f"exact factor {index} has a non-positive denominator")
    return factor


def _parse_executor(raw: object, index: int) -> _Executor:
    values = _row(raw, 8, f"direct executor {index}")
    role = values[1]
    operation = values[2]
    if not isinstance(role, str) or role not in {
        "source",
        "contribution",
        "finalization",
        "closure",
    }:
        raise ArtifactError(f"direct executor {index} has an invalid role")
    if not isinstance(operation, str) or operation not in {
        "initialize",
        "add",
        "finalize-in-place",
        "closure-add",
    }:
        raise ArtifactError(f"direct executor {index} has an invalid operation")
    parent_counts = tuple(
        _integer(value, f"direct executor {index} parent count")
        for value in _sequence(values[3], f"direct executor {index} parent counts")
    )
    prepared_kernel_id = values[6]
    if prepared_kernel_id is not None:
        prepared_kernel_id = _integer(
            prepared_kernel_id, f"direct executor {index} prepared kernel"
        )
    runtime_template = values[7]
    if runtime_template is not None and (
        not isinstance(runtime_template, str) or not runtime_template
    ):
        raise ArtifactError(f"direct executor {index} has an invalid runtime template")
    return _Executor(
        executor_id=_integer(values[0], f"direct executor {index} ID"),
        role=role,
        destination_operation=operation,
        parent_component_counts=parent_counts,
        destination_component_count=_integer(
            values[4], f"direct executor {index} destination count", minimum=1
        ),
        momentum_operand_count=_integer(
            values[5], f"direct executor {index} momentum count"
        ),
        prepared_kernel_id=prepared_kernel_id,
        runtime_template=runtime_template,
    )


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactError(f"recurrence {context} must be a lowercase SHA-256")
    return value


def _validate_sections(sections: _RecurrenceExactSectionsV1) -> None:
    if tuple(row.semantic_id for row in sections.currents) != tuple(
        range(len(sections.currents))
    ):
        raise ArtifactError("recurrence current IDs are not dense")
    if tuple(row.executor_id for row in sections.executors) != tuple(
        range(len(sections.executors))
    ):
        raise ArtifactError("recurrence direct executor IDs are not dense")
    if tuple(row.destination_id for row in sections.amplitude_destinations) != tuple(
        range(sections.amplitude_destination_count)
    ):
        raise ArtifactError("recurrence amplitude destination IDs are not dense")
    if sections.strategy == "topology-replay":
        _validate_replay_sections(sections)
    elif sections.strategy == "all-flow-union":
        _validate_union_sections(sections)
    else:
        _validate_contracted_sections(sections)


def _validate_replay_sections(sections: _RecurrenceExactSectionsV1) -> None:
    if len(sections.public_flow_ids) != len(sections.replay_targets):
        raise ArtifactError("recurrence public flow and replay axes differ in size")
    _validate_replay_tables(sections)


def _validate_replay_tables(sections: _RecurrenceExactSectionsV1) -> None:
    if not sections.replay_targets and (
        sections.source_permutations
        or sections.replay_momentum_signs
        or sections.replay_helicity_map
    ):
        raise ArtifactError("recurrence replay tables have no replay targets")
    if any(
        len(permutation) != sections.external_source_count
        or sorted(permutation) != list(range(sections.external_source_count))
        for target in sections.replay_targets
        for permutation in (
            sections.source_permutations[
                target.source_permutation_start : target.source_permutation_start
                + target.source_permutation_count
            ],
        )
    ):
        raise ArtifactError(
            "recurrence replay permutation is not a complete source bijection"
        )
    if len(sections.replay_momentum_signs) != len(sections.source_permutations) or any(
        sign not in (-1, 1) for sign in sections.replay_momentum_signs
    ):
        raise ArtifactError(
            "recurrence replay momentum signs do not match source permutations"
        )
    helicity_count = len(sections.resolved_helicities)
    for target in sections.replay_targets:
        mapped = sections.replay_helicity_map[
            target.helicity_map_start : target.helicity_map_start
            + target.helicity_map_count
        ]
        if len(mapped) != helicity_count or sorted(mapped) != list(
            range(helicity_count)
        ):
            raise ArtifactError(
                "recurrence replay helicity mapping is not a complete bijection"
            )
    if (
        sections.source_dispatch_variants
        or sections.source_embeddings
        or sections.source_projections
        or sections.resolved_source_selections
    ):
        raise ArtifactError("topology-replay plan carries union source-dispatch tables")
    if any(row.source_selection_count for row in sections.resolved_helicities):
        raise ArtifactError(
            "topology-replay resolved helicity carries source selections"
        )


def _validate_union_sections(sections: _RecurrenceExactSectionsV1) -> None:
    if (
        sections.replay_targets
        or sections.source_permutations
        or sections.replay_momentum_signs
        or sections.replay_helicity_map
    ):
        raise ArtifactError("all-flow-union plan carries topology replay tables")
    if not sections.source_dispatch_variants:
        raise ArtifactError("all-flow-union plan has no source-dispatch variants")
    if not sections.resolved_source_selections:
        raise ArtifactError("all-flow-union plan has no resolved source selections")

    factor_count = len(sections.exact_factors)
    for index, variant in enumerate(sections.source_dispatch_variants):
        if variant.source_row_id >= len(sections.sources):
            raise ArtifactError(
                f"source-dispatch variant {index} references an absent source row"
            )
        source = sections.sources[variant.source_row_id]
        if source.source_template_or_dispatch_domain != variant.dispatch_domain_id:
            raise ArtifactError(
                f"source-dispatch variant {index} has the wrong dispatch domain"
            )
        if (
            variant.source_template_id == DIRECT_NONE_U32
            or variant.crossing_exact_factor_id >= factor_count
        ):
            raise ArtifactError(
                f"source-dispatch variant {index} has an invalid exact binding"
            )
        embeddings = _table_slice(
            sections.source_embeddings,
            variant.embedding_start,
            variant.embedding_count,
            f"source-dispatch variant {index} embeddings",
        )
        projections = _table_slice(
            sections.source_projections,
            variant.projection_start,
            variant.projection_count,
            f"source-dispatch variant {index} projections",
        )
        if tuple(row.full_component for row in embeddings) != tuple(
            range(variant.embedding_count)
        ):
            raise ArtifactError(
                f"source-dispatch variant {index} embeddings are not ordered"
            )
        for full_component, embedding in enumerate(embeddings):
            if embedding.exact_factor_id >= factor_count:
                raise ArtifactError(
                    f"source-dispatch variant {index} has an invalid embedding factor"
                )
            factor = sections.exact_factors[embedding.exact_factor_id]
            is_zero = factor.real_numerator == 0 and factor.imaginary_numerator == 0
            if (embedding.source_component == DIRECT_NONE_U32) != is_zero:
                raise ArtifactError(
                    f"source-dispatch variant {index} has an inconsistent "
                    "zero embedding"
                )
            if (
                embedding.source_component != DIRECT_NONE_U32
                and embedding.source_component >= variant.projection_count
            ):
                raise ArtifactError(
                    f"source-dispatch variant {index} embedding is out of range"
                )
            if full_component != embedding.full_component:
                raise ArtifactError(
                    f"source-dispatch variant {index} embedding order is invalid"
                )
        for source_component, projection in enumerate(projections):
            if (
                projection.source_component != source_component
                or projection.full_component >= variant.embedding_count
                or embeddings[projection.full_component].source_component
                != source_component
            ):
                raise ArtifactError(
                    f"source-dispatch variant {index} projection is not inverse"
                )

    for index, helicity in enumerate(sections.resolved_helicities):
        if (
            helicity.source_state_count != sections.external_source_count
            or helicity.source_selection_count != sections.external_source_count
            or helicity.source_state_start + helicity.source_state_count
            > len(sections.source_state_assignments)
            or helicity.source_selection_start + helicity.source_selection_count
            > len(sections.resolved_source_selections)
        ):
            raise ArtifactError(
                f"resolved helicity {index} does not cover every external source"
            )
        assignments = _table_slice(
            sections.source_state_assignments,
            helicity.source_state_start,
            helicity.source_state_count,
            f"resolved helicity {index} source states",
        )
        selections = _table_slice(
            sections.resolved_source_selections,
            helicity.source_selection_start,
            helicity.source_selection_count,
            f"resolved helicity {index} source selections",
        )
        if (
            len(assignments) != sections.external_source_count
            or len(selections) != sections.external_source_count
        ):
            raise ArtifactError(
                f"resolved helicity {index} does not cover every external source"
            )
        for source_slot, (assignment, selection) in enumerate(
            zip(assignments, selections, strict=True)
        ):
            if (
                assignment.source_slot != source_slot
                or selection.source_slot != source_slot
            ):
                raise ArtifactError(
                    f"resolved helicity {index} source rows are not ordered"
                )
            if selection.dispatch_variant_id >= len(sections.source_dispatch_variants):
                raise ArtifactError(
                    f"resolved helicity {index} selects an absent source variant"
                )
            variant = sections.source_dispatch_variants[selection.dispatch_variant_id]
            source = sections.sources[variant.source_row_id]
            if (
                source.source_slot != source_slot
                or variant.source_state_index != assignment.state_index
            ):
                raise ArtifactError(
                    f"resolved helicity {index} source selection is inconsistent"
                )

    sectors = [row.target_sector_id for row in sections.amplitude_destinations]
    if any(
        row.target_helicity_id != DIRECT_NONE_U32
        for row in sections.amplitude_destinations
    ):
        raise ArtifactError(
            "all-flow-union amplitude destination fixes a numerical helicity"
        )
    if len(sectors) != len(set(sectors)) or set(sectors) != set(
        sections.public_flow_ids
    ):
        raise ArtifactError(
            "all-flow-union destinations are not completely covered by public flows"
        )


def _validate_contracted_sections(sections: _RecurrenceExactSectionsV1) -> None:
    if sections.public_flow_ids:
        raise ArtifactError(
            "contracted-color recurrence carries a public color-flow axis"
        )
    _validate_replay_tables(sections)
    if (
        sections.source_dispatch_variants
        or sections.source_embeddings
        or sections.source_projections
        or sections.resolved_source_selections
    ):
        raise ArtifactError(
            "contracted-color recurrence carries union source-dispatch tables"
        )
    helicity_count = len(sections.resolved_helicities)
    if helicity_count == 0:
        raise ArtifactError("contracted-color recurrence has no resolved helicities")
    for index, helicity in enumerate(sections.resolved_helicities):
        if (
            helicity.source_state_count != sections.external_source_count
            or helicity.source_selection_count != 0
            or helicity.source_state_start + helicity.source_state_count
            > len(sections.source_state_assignments)
        ):
            raise ArtifactError(
                f"contracted resolved helicity {index} has invalid source coverage"
            )
        assignments = _table_slice(
            sections.source_state_assignments,
            helicity.source_state_start,
            helicity.source_state_count,
            f"contracted resolved helicity {index} source states",
        )
        if tuple(row.source_slot for row in assignments) != tuple(
            range(sections.external_source_count)
        ):
            raise ArtifactError(
                f"contracted resolved helicity {index} source rows are not ordered"
            )

    actual = [
        (row.target_sector_id, row.target_helicity_id)
        for row in sections.amplitude_destinations
    ]
    if (
        len(set(actual)) != len(actual)
        or any(
            helicity_id == DIRECT_NONE_U32 or helicity_id >= helicity_count
            for _, helicity_id in actual
        )
        or {helicity_id for _, helicity_id in actual} != set(range(helicity_count))
    ):
        raise ArtifactError(
            "contracted-color destinations have invalid sparse sector/helicity coverage"
        )


def _table_slice(
    rows: Sequence[_RowT],
    start: int,
    count: int,
    context: str,
) -> Sequence[_RowT]:
    stop = start + count
    if start < 0 or stop > len(rows):
        raise ArtifactError(f"{context} is out of bounds")
    return rows[start:stop]


__all__ = [
    "DIRECT_NONE_U32",
    "RECURRENCE_DIRECT_RUNTIME_CAPABILITY",
    "RECURRENCE_PLAN_V2_ABI",
    "RECURRENCE_RUNTIME_KIND",
    "RECURRENCE_RUNTIME_LAYOUT_V2_ABI",
    "_NativeExactSectionsLoader",
    "_RecurrenceExactSectionsV1",
    "_load_recurrence_exact_sections_v1",
]
