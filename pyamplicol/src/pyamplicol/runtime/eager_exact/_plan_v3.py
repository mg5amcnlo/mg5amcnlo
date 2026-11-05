# SPDX-License-Identifier: 0BSD
"""Typed adapter for exact-required sections of compact eager plan-v3."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from pyamplicol._internal.versions import (
    EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
    verify_native_module,
)
from pyamplicol.api.errors import ArtifactError, CompatibilityError
from pyamplicol.runtime.eager_exact._contracts import (
    _complex_pair,
    _ExactAttachmentRow,
    _ExactClosureRow,
    _ExactComplexProduct,
    _ExactCouplingRow,
    _ExactFinalizationRow,
    _ExactInvocationRow,
    _integer,
    _mapping,
    _sequence,
)

EAGER_EXACT_SECTIONS_ABI = "pyamplicol-eager-exact-sections-v1"
EAGER_REDUCTION_GROUPS_ABI = "pyamplicol-eager-reduction-groups-v1"
EAGER_PLAN_V3_ABI = "pyamplicol-eager-plan-v3"
EAGER_RUNTIME_LAYOUT_ABI = "pyamplicol-eager-runtime-layout-v1"
EAGER_PLAN_V3_RUNTIME_CAPABILITY = "rusticol.eager-runtime-layout.complex-f64.v1"
_EAGER_PLAN_V3_RUNTIME_CAPABILITIES = sorted(
    (
        EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        EAGER_PLAN_V3_RUNTIME_CAPABILITY,
    )
)
_NATIVE_BINDING_NAME = "_load_eager_exact_sections_v1"
_NATIVE_REDUCTION_BINDING_NAME = "_load_eager_reduction_groups_v1"


class _NativeExactSectionsBinding(Protocol):
    def __call__(self, artifact_root: str, process_id: str, /) -> object: ...


class _NativeReductionGroupsBinding(Protocol):
    def __call__(self, artifact_root: str, process_id: str, /) -> object: ...


_NativeExactSectionsLoader = Callable[[Path, str], object]
_NativeReductionGroupsLoader = Callable[[Path, str], object]


@dataclass(frozen=True, slots=True)
class _ExactStageV3:
    stage_index: int
    invocations: tuple[_ExactInvocationRow, ...]
    attachments: tuple[_ExactAttachmentRow, ...]
    finalizations: tuple[_ExactFinalizationRow, ...]


@dataclass(frozen=True, slots=True)
class _EagerExactSectionsV1:
    exact_schema: Mapping[str, object]
    reduction_groups: tuple[Mapping[str, object], ...]
    selector_group_ids: tuple[int, ...]
    selector_domains: tuple[frozenset[int], ...]
    couplings: tuple[_ExactCouplingRow, ...]
    stages: tuple[_ExactStageV3, ...]
    closures: tuple[_ExactClosureRow, ...]


def _load_eager_exact_sections_v1(
    artifact_root: Path,
    process_id: str,
    *,
    loader: _NativeExactSectionsLoader | None = None,
) -> _EagerExactSectionsV1:
    raw = (loader or _native_exact_sections_loader)(artifact_root, process_id)
    return _parse_exact_sections(raw, process_id)


def _load_eager_reduction_groups_v1(
    artifact_root: Path,
    process_id: str,
    *,
    loader: _NativeReductionGroupsLoader | None = None,
) -> tuple[Mapping[str, object], ...]:
    raw = (loader or _native_reduction_groups_loader)(artifact_root, process_id)
    root = _mapping(raw, "compact eager reduction groups")
    if root.get("abi") != EAGER_REDUCTION_GROUPS_ABI:
        raise CompatibilityError(
            f"unsupported compact eager reduction-groups ABI {root.get('abi')!r}"
        )
    if root.get("runtime_layout_abi") != EAGER_RUNTIME_LAYOUT_ABI:
        raise CompatibilityError(
            "unsupported compact eager reduction runtime-layout ABI "
            f"{root.get('runtime_layout_abi')!r}"
        )
    if root.get("process_id") != process_id:
        raise ArtifactError("compact eager reduction groups select the wrong process")
    return tuple(
        _mapping(value, f"compact eager reduction group {index}")
        for index, value in enumerate(
            _sequence(
                root.get("reduction_groups"),
                "compact eager reduction groups",
            )
        )
    )


def _native_exact_sections_loader(artifact_root: Path, process_id: str) -> object:
    try:
        module = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(module)
    except ImportError as exc:
        raise CompatibilityError(
            "compact eager exact execution requires pyamplicol._rusticol"
        ) from exc
    candidate = getattr(module, _NATIVE_BINDING_NAME, None)
    if not callable(candidate):
        raise CompatibilityError(
            "compact eager exact execution requires the private native binding "
            f"{_NATIVE_BINDING_NAME}"
        )
    binding = cast(_NativeExactSectionsBinding, candidate)
    try:
        return binding(os.fspath(artifact_root), process_id)
    except (ArtifactError, CompatibilityError):
        raise
    except Exception as exc:
        raise ArtifactError(
            f"could not load compact exact sections for process {process_id!r}: {exc}"
        ) from exc


def _native_reduction_groups_loader(artifact_root: Path, process_id: str) -> object:
    try:
        module = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(module)
    except ImportError as exc:
        raise CompatibilityError(
            "compact eager reduction loading requires pyamplicol._rusticol"
        ) from exc
    candidate = getattr(module, _NATIVE_REDUCTION_BINDING_NAME, None)
    if not callable(candidate):
        raise CompatibilityError(
            "compact eager reduction loading requires the private native binding "
            f"{_NATIVE_REDUCTION_BINDING_NAME}"
        )
    binding = cast(_NativeReductionGroupsBinding, candidate)
    try:
        return binding(os.fspath(artifact_root), process_id)
    except (ArtifactError, CompatibilityError):
        raise
    except Exception as exc:
        raise ArtifactError(
            f"could not load compact eager reduction groups for process "
            f"{process_id!r}: {exc}"
        ) from exc


def _parse_exact_sections(raw: object, process_id: str) -> _EagerExactSectionsV1:
    root = _mapping(raw, "compact eager exact sections")
    if root.get("abi") != EAGER_EXACT_SECTIONS_ABI:
        raise CompatibilityError(
            f"unsupported compact eager exact-sections ABI {root.get('abi')!r}"
        )
    if root.get("runtime_layout_abi") != EAGER_RUNTIME_LAYOUT_ABI:
        raise CompatibilityError(
            f"unsupported compact eager runtime-layout ABI "
            f"{root.get('runtime_layout_abi')!r}"
        )
    if root.get("process_id") != process_id:
        raise ArtifactError("compact eager exact sections select the wrong process")

    couplings = tuple(
        _parse_coupling(row, index)
        for index, row in enumerate(
            _sequence(root.get("couplings"), "compact eager couplings")
        )
    )
    invocations = tuple(
        _parse_invocation(row, index)
        for index, row in enumerate(
            _sequence(root.get("invocations"), "compact eager invocations")
        )
    )
    attachments = tuple(
        _parse_attachment(row, index)
        for index, row in enumerate(
            _sequence(root.get("attachments"), "compact eager attachments")
        )
    )
    finalizations = tuple(
        _parse_finalization(row, index)
        for index, row in enumerate(
            _sequence(root.get("finalizations"), "compact eager finalizations")
        )
    )
    stages = _parse_stages(root.get("stages"), invocations, attachments, finalizations)
    closures = tuple(
        _parse_closure(row, index)
        for index, row in enumerate(
            _sequence(root.get("closures"), "compact eager closures")
        )
    )
    selector_group_ids = tuple(
        _integer(value, f"compact eager selector group {index}")
        for index, value in enumerate(
            _sequence(
                root.get("selector_group_ids"),
                "compact eager selector groups",
            )
        )
    )
    selector_domains = tuple(
        frozenset(
            _integer(member, f"compact eager selector domain {index} member")
            for member in _sequence(
                raw_domain,
                f"compact eager selector domain {index}",
            )
        )
        for index, raw_domain in enumerate(
            _sequence(root.get("selector_domains"), "compact eager selector domains")
        )
    )
    return _EagerExactSectionsV1(
        exact_schema=_mapping(root.get("exact_schema"), "compact eager exact schema"),
        reduction_groups=tuple(
            _mapping(value, f"compact eager reduction group {index}")
            for index, value in enumerate(
                _sequence(
                    root.get("reduction_groups"),
                    "compact eager reduction groups",
                )
            )
        ),
        selector_group_ids=selector_group_ids,
        selector_domains=selector_domains,
        couplings=couplings,
        stages=stages,
        closures=closures,
    )


def _row(raw: object, width: int, context: str) -> Sequence[object]:
    values = _sequence(raw, context)
    if len(values) != width:
        raise ArtifactError(f"{context} must contain {width} fields")
    return values


def _parse_coupling(raw: object, index: int) -> _ExactCouplingRow:
    context = f"compact eager coupling {index}"
    values = _row(raw, 4, context)
    return _ExactCouplingRow(
        _integer(values[0], f"{context} real parameter"),
        _integer(values[1], f"{context} imaginary parameter"),
        _complex_pair(values[2], values[3], f"{context} constant"),
    )


def _parse_invocation(raw: object, index: int) -> _ExactInvocationRow:
    context = f"compact eager invocation {index}"
    values = _row(raw, 10, context)
    return _ExactInvocationRow(
        *(
            _integer(value, f"{context} field {field}")
            for field, value in enumerate(values)
        )
    )


def _parse_attachment(raw: object, index: int) -> _ExactAttachmentRow:
    context = f"compact eager attachment {index}"
    values = _row(raw, 4, context)
    return _ExactAttachmentRow(
        _integer(values[0], f"{context} current"),
        _parse_factor_product(values[1], values[2], f"{context} factor"),
        _integer(values[3], f"{context} selector domain"),
    )


def _parse_finalization(raw: object, index: int) -> _ExactFinalizationRow:
    context = f"compact eager finalization {index}"
    values = _row(raw, 7, context)
    return _ExactFinalizationRow(
        *(
            _integer(value, f"{context} field {field}")
            for field, value in enumerate(values)
        )
    )


def _parse_closure(raw: object, index: int) -> _ExactClosureRow:
    context = f"compact eager closure {index}"
    values = _row(raw, 11, context)
    raw_coefficients = values[8]
    coefficients = (
        None
        if raw_coefficients is None
        else tuple(
            _complex_pair(
                _row(value, 2, f"{context} direct coefficient {coefficient_index}")[0],
                _row(value, 2, f"{context} direct coefficient {coefficient_index}")[1],
                f"{context} direct coefficient {coefficient_index}",
            )
            for coefficient_index, value in enumerate(
                _sequence(raw_coefficients, f"{context} direct coefficients")
            )
        )
    )
    return _ExactClosureRow(
        kernel_id=_integer(values[0], f"{context} kernel"),
        left_value_slot_id=_integer(values[1], f"{context} left value"),
        right_value_slot_id=_integer(values[2], f"{context} right value"),
        amplitude_index=_integer(values[3], f"{context} amplitude"),
        coupling_slot_id=_integer(values[4], f"{context} coupling"),
        output_factor_source=_integer(values[5], f"{context} output factor source"),
        factor=_parse_factor_product(values[6], values[7], f"{context} factor"),
        direct_coefficients=coefficients,
        coherent_group_id=_integer(values[9], f"{context} coherent group"),
        selector_domain_id=_integer(values[10], f"{context} selector domain"),
    )


def _parse_factor_product(
    raw_numerators: object,
    raw_denominator: object,
    context: str,
) -> _ExactComplexProduct:
    numerators = []
    for index, raw in enumerate(_sequence(raw_numerators, f"{context} numerators")):
        values = _row(raw, 2, f"{context} numerator {index}")
        numerators.append(
            _complex_pair(values[0], values[1], f"{context} numerator {index}")
        )
    if not numerators:
        raise ArtifactError(f"{context} must contain at least one numerator")
    denominator = None
    if raw_denominator is not None:
        values = _row(raw_denominator, 2, f"{context} denominator")
        denominator = _complex_pair(values[0], values[1], f"{context} denominator")
    return _ExactComplexProduct(tuple(numerators), denominator)


def _parse_stages(
    raw_stages: object,
    invocations: tuple[_ExactInvocationRow, ...],
    attachments: tuple[_ExactAttachmentRow, ...],
    finalizations: tuple[_ExactFinalizationRow, ...],
) -> tuple[_ExactStageV3, ...]:
    stages = []
    invocation_cursor = 0
    attachment_cursor = 0
    finalization_cursor = 0
    previous_stage = -1
    for index, raw in enumerate(_sequence(raw_stages, "compact eager stages")):
        context = f"compact eager stage {index}"
        values = _row(raw, 7, context)
        stage_index = _integer(values[0], f"{context} index")
        if stage_index <= previous_stage:
            raise ArtifactError("compact eager stage indices must increase")
        previous_stage = stage_index

        invocation_start = _integer(values[1], f"{context} start")
        invocation_count = _integer(values[2], f"{context} count")
        if (
            invocation_start != invocation_cursor
            or invocation_start + invocation_count > len(invocations)
        ):
            raise ArtifactError(f"{context} has a non-contiguous table range")
        invocation_cursor = invocation_start + invocation_count
        stage_invocation_rows = invocations[invocation_start:invocation_cursor]

        attachment_start = _integer(values[3], f"{context} start")
        attachment_count = _integer(values[4], f"{context} count")
        if (
            attachment_start != attachment_cursor
            or attachment_start + attachment_count > len(attachments)
        ):
            raise ArtifactError(f"{context} has a non-contiguous table range")
        attachment_cursor = attachment_start + attachment_count
        stage_attachments = attachments[attachment_start:attachment_cursor]

        finalization_start = _integer(values[5], f"{context} start")
        finalization_count = _integer(values[6], f"{context} count")
        if (
            finalization_start != finalization_cursor
            or finalization_start + finalization_count > len(finalizations)
        ):
            raise ArtifactError(f"{context} has a non-contiguous table range")
        finalization_cursor = finalization_start + finalization_count
        stage_finalizations = finalizations[
            finalization_start:finalization_cursor
        ]

        if any(
            invocation.attachment_start < attachment_start
            for invocation in stage_invocation_rows
        ):
            raise ArtifactError(
                f"{context} has an invocation before its attachment range"
            )
        stage_invocations = tuple(
            _ExactInvocationRow(
                invocation.kernel_id,
                invocation.left_value_slot_id,
                invocation.right_value_slot_id,
                invocation.left_momentum_slot_id,
                invocation.right_momentum_slot_id,
                invocation.coupling_slot_id,
                invocation.output_factor_source,
                invocation.attachment_start - attachment_start,
                invocation.attachment_count,
                invocation.selector_domain_id,
            )
            for invocation in stage_invocation_rows
        )
        stages.append(
            _ExactStageV3(
                stage_index,
                stage_invocations,
                stage_attachments,
                stage_finalizations,
            )
        )
    if (
        invocation_cursor != len(invocations)
        or attachment_cursor != len(attachments)
        or finalization_cursor != len(finalizations)
    ):
        raise ArtifactError("compact eager stages do not cover their exact tables")
    return tuple(stages)


def _validate_plan_v3_execution_header(execution: Mapping[str, object]) -> None:
    from pyamplicol._internal.versions import PROCESS_ARTIFACT_SCHEMA_VERSION

    if execution.get("schema_version") != PROCESS_ARTIFACT_SCHEMA_VERSION:
        raise CompatibilityError(
            f"unsupported eager process schema {execution.get('schema_version')!r}"
        )
    if execution.get("kind") != "pyamplicol-runtime-eager-execution":
        raise CompatibilityError(
            f"unsupported exact eager execution kind {execution.get('kind')!r}"
        )
    if execution.get("eager_plan_abi") != EAGER_PLAN_V3_ABI:
        raise CompatibilityError(
            f"unsupported eager plan ABI {execution.get('eager_plan_abi')!r}"
        )
    plan = _mapping(execution.get("plan"), "plan")
    if (
        plan.get("kind") != execution.get("kind")
        or plan.get("eager_plan_abi") != EAGER_PLAN_V3_ABI
        or plan.get("runtime_layout_abi") != EAGER_RUNTIME_LAYOUT_ABI
    ):
        raise CompatibilityError("outer and compact eager plan contracts do not match")
    if execution.get("key") is None:
        raise ArtifactError("compact eager execution has no process key")
    expected = _EAGER_PLAN_V3_RUNTIME_CAPABILITIES
    if execution.get("required_runtime_capabilities") != expected:
        raise CompatibilityError(
            "unsupported compact eager runtime capability contract"
        )
    if plan.get("required_runtime_capabilities") != expected:
        raise CompatibilityError("unsupported compact eager plan capability contract")


__all__ = [
    "EAGER_EXACT_SECTIONS_ABI",
    "EAGER_PLAN_V3_ABI",
    "EAGER_PLAN_V3_RUNTIME_CAPABILITY",
    "EAGER_RUNTIME_LAYOUT_ABI",
]
