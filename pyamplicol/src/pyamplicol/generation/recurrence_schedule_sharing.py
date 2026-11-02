# SPDX-License-Identifier: 0BSD
"""Content-addressed recurrence schedules and compact process bindings."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import struct
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Generic, Protocol, TypeVar

from .recurrence_columnar import (
    RecurrenceBuilderLogicalInputV1,
    RecurrenceExternalLegV1,
    RecurrencePhysicalLCSectorV1,
    RecurrencePublicLCFlowV1,
    RecurrenceSourceStateV1,
)

RECURRENCE_SCHEDULE_SHARING_KIND = "pyamplicol-recurrence-schedule-sharing"
RECURRENCE_SCHEDULE_SHARING_SCHEMA_VERSION = 3
RECURRENCE_SCHEDULE_INDEX_PATH = "recurrence/schedule-index.json"
RECURRENCE_PROCESS_BINDING_ABI = "pyamplicol-recurrence-process-binding-v4"
RECURRENCE_PROCESS_BINDING_MAGIC = b"PACRDBN4"
_PROCESS_BINDING_VERSION = 4
_PROCESS_BINDING_FIXED_SIZE = 344
_MAX_SOURCE_BIJECTIONS = 4096

_DIRECT_ROLE = {
    "source": 0,
    "contribution": 1,
    "finalization": 2,
    "closure": 3,
}
_DIRECT_BACKEND = {"jit": 0, "cpp": 1, "asm": 2}
_DIRECT_BINDING_SOURCE = 0
_DIRECT_BINDING_INTRINSIC = 1
_DIRECT_BINDING_JIT = 2
_DIRECT_BINDING_NATIVE = 3
_DIRECT_FLAG_USES_EXACT_FACTOR = 1 << 0
_MISSING_U32 = (1 << 32) - 1
_PROCESS_VARIABLE_SEMANTIC_ROLES = frozenset(
    {
        "process",
        "color-plan",
        "fermion-pairing-semantic",
        "fermion-pairing-topology",
        "closure-reconstruction",
        "helicity-support:pure-massless-adjoint-tree-v1",
        "helicity-equivalence:global-flip-v1",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")


class RecurrenceScheduleSharingError(ValueError):
    """Raised when recurrence schedules cannot be shared exactly."""


@dataclass(frozen=True, slots=True)
class RecurrenceProcessExecutorPack:
    """Process-owned sparse Direct executor records for one fixed schedule."""

    compiled_model_digest: str
    recurrence_template_catalog_digest: str
    prepared_kernel_pack_digest: str
    direct_template_catalog_digest: str
    runtime_layout_digest: str
    backend: str
    target_triple: str
    portable: bool
    cpu_features: tuple[str, ...]
    catalog_executor_count: int
    executor_ids: tuple[int, ...]
    descriptor_payloads: tuple[bytes, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.compiled_model_digest, "compiled-model digest"),
            (
                self.recurrence_template_catalog_digest,
                "recurrence-template catalog digest",
            ),
            (self.prepared_kernel_pack_digest, "prepared-kernel pack digest"),
            (self.direct_template_catalog_digest, "direct-template catalog digest"),
            (self.runtime_layout_digest, "runtime-layout digest"),
        ):
            _require_sha256(value, label)
        if self.backend not in _DIRECT_BACKEND:
            raise RecurrenceScheduleSharingError(
                f"unsupported process executor backend {self.backend!r}"
            )
        _nonempty_text(self.target_triple, "process executor target triple")
        if type(self.portable) is not bool:
            raise RecurrenceScheduleSharingError(
                "process executor portability must be boolean"
            )
        if self.cpu_features != tuple(sorted(set(self.cpu_features))) or any(
            not isinstance(feature, str) or not feature for feature in self.cpu_features
        ):
            raise RecurrenceScheduleSharingError(
                "process executor CPU features must be sorted and unique"
            )
        count = _nonnegative_integer(
            self.catalog_executor_count, "direct-executor catalog count"
        )
        if count > _MISSING_U32:
            raise RecurrenceScheduleSharingError(
                "direct-executor catalog count exceeds the u32 wire domain"
            )
        if self.executor_ids != tuple(sorted(set(self.executor_ids))):
            raise RecurrenceScheduleSharingError(
                "process executor IDs must be sorted and unique"
            )
        if len(self.executor_ids) > _MISSING_U32 or any(
            isinstance(executor_id, bool)
            or not isinstance(executor_id, int)
            or executor_id < 0
            or executor_id >= count
            for executor_id in self.executor_ids
        ):
            raise RecurrenceScheduleSharingError(
                "process executor ID is outside the complete catalog domain"
            )
        if len(self.executor_ids) != len(self.descriptor_payloads) or any(
            not isinstance(payload, bytes) or len(payload) < 16
            for payload in self.descriptor_payloads
        ):
            raise RecurrenceScheduleSharingError(
                "process executor descriptors do not match their selected IDs"
            )
        for expected_id, payload in zip(
            self.executor_ids, self.descriptor_payloads, strict=True
        ):
            record_size, executor_id = struct.unpack_from("<II", payload)
            if record_size != len(payload) or executor_id != expected_id:
                raise RecurrenceScheduleSharingError(
                    "process executor descriptor header disagrees with its selected ID"
                )


@dataclass(frozen=True, slots=True)
class RecurrenceProcessRemap:
    """Authenticated root-schedule to concrete-process binding.

    Dense source, flow, and sector maps are deliberately explicit. Model-wide
    template/executor/parameter domains use sparse changes over an identity
    bijection so a binding remains small even for a large prepared catalog.
    """

    source_slots: tuple[int, ...]
    source_momentum_signs: tuple[int, ...]
    source_helicity_signs: tuple[int, ...]
    source_state_offsets: tuple[int, ...]
    source_state_indices: tuple[int, ...]
    public_flow_ids: tuple[int, ...]
    physical_sector_ids: tuple[int, ...]
    state_template_count: int
    source_template_count: int
    direct_executor_count: int
    parameter_slot_count: int
    state_template_changes: tuple[tuple[int, int], ...] = ()
    source_template_changes: tuple[tuple[int, int], ...] = ()
    direct_executor_changes: tuple[tuple[int, int], ...] = ()
    parameter_slot_changes: tuple[tuple[int, int], ...] = ()
    bijection_digest: str = ""

    def __post_init__(self) -> None:
        _permutation(self.source_slots, "source-slot remap")
        _signs(self.source_momentum_signs, len(self.source_slots), "momentum signs")
        _signs(self.source_helicity_signs, len(self.source_slots), "helicity signs")
        _ragged_permutations(
            self.source_state_offsets,
            self.source_state_indices,
            len(self.source_slots),
            "source-state remap",
        )
        _permutation(self.public_flow_ids, "public-flow remap")
        _permutation(self.physical_sector_ids, "physical-sector remap")
        for count, label in (
            (self.state_template_count, "state-template count"),
            (self.source_template_count, "source-template count"),
            (self.direct_executor_count, "direct-executor count"),
            (self.parameter_slot_count, "parameter-slot count"),
        ):
            _nonnegative_integer(count, label)
        for changes, count, label in (
            (
                self.state_template_changes,
                self.state_template_count,
                "state-template remap",
            ),
            (
                self.source_template_changes,
                self.source_template_count,
                "source-template remap",
            ),
            (
                self.direct_executor_changes,
                self.direct_executor_count,
                "direct-executor remap",
            ),
            (
                self.parameter_slot_changes,
                self.parameter_slot_count,
                "parameter-slot remap",
            ),
        ):
            _sparse_permutation(changes, count, label)
        if self.bijection_digest:
            _require_sha256(self.bijection_digest, "process-bijection digest")

    @classmethod
    def identity(
        cls,
        logical: RecurrenceBuilderLogicalInputV1,
        *,
        direct_executor_count: int,
        parameter_slot_count: int,
    ) -> RecurrenceProcessRemap:
        state_count = _template_count(logical, "current-state")
        source_count = _template_count(logical, "source")
        source_state_offsets = [0]
        source_state_indices: list[int] = []
        for leg in logical.external_legs:
            source_state_indices.extend(range(len(leg.source_states)))
            source_state_offsets.append(len(source_state_indices))
        result = cls(
            source_slots=tuple(range(len(logical.external_legs))),
            source_momentum_signs=(1,) * len(logical.external_legs),
            source_helicity_signs=(1,) * len(logical.external_legs),
            source_state_offsets=tuple(source_state_offsets),
            source_state_indices=tuple(source_state_indices),
            public_flow_ids=tuple(range(len(logical.public_flows))),
            physical_sector_ids=tuple(range(_physical_sector_domain_count(logical))),
            state_template_count=state_count,
            source_template_count=source_count,
            direct_executor_count=_nonnegative_integer(
                direct_executor_count, "direct-executor count"
            ),
            parameter_slot_count=_nonnegative_integer(
                parameter_slot_count, "parameter-slot count"
            ),
        )
        return result.with_digest(
            _canonical_digest(
                {
                    "contract": "pyamplicol-recurrence-process-bijection-v1",
                    "process": logical.process_id,
                    "remap": result._digest_payload(),
                    "relation": "identity",
                }
            )
        )

    def with_digest(self, digest: str) -> RecurrenceProcessRemap:
        return RecurrenceProcessRemap(
            source_slots=self.source_slots,
            source_momentum_signs=self.source_momentum_signs,
            source_helicity_signs=self.source_helicity_signs,
            source_state_offsets=self.source_state_offsets,
            source_state_indices=self.source_state_indices,
            public_flow_ids=self.public_flow_ids,
            physical_sector_ids=self.physical_sector_ids,
            state_template_count=self.state_template_count,
            source_template_count=self.source_template_count,
            direct_executor_count=self.direct_executor_count,
            parameter_slot_count=self.parameter_slot_count,
            state_template_changes=self.state_template_changes,
            source_template_changes=self.source_template_changes,
            direct_executor_changes=self.direct_executor_changes,
            parameter_slot_changes=self.parameter_slot_changes,
            bijection_digest=_require_sha256(digest, "process-bijection digest"),
        )

    def remap_resolved_helicities(
        self,
        rows: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        result: list[tuple[int, ...]] = []
        for row in rows:
            if len(row) != len(self.source_slots):
                raise RecurrenceScheduleSharingError(
                    "resolved helicity width disagrees with process binding"
                )
            target = [0] * len(row)
            for root_slot, target_slot in enumerate(self.source_slots):
                target[target_slot] = (
                    row[root_slot] * self.source_helicity_signs[root_slot]
                )
            result.append(tuple(target))
        if len(set(result)) != len(result):
            raise RecurrenceScheduleSharingError(
                "process binding collapses distinct resolved helicities"
            )
        return tuple(result)

    def to_mapping(self) -> dict[str, object]:
        return {
            "bijection_digest": self.bijection_digest,
            "source_slots": list(self.source_slots),
            "source_momentum_signs": list(self.source_momentum_signs),
            "source_helicity_signs": list(self.source_helicity_signs),
            "source_state_offsets": list(self.source_state_offsets),
            "source_state_indices": list(self.source_state_indices),
            "public_flow_ids": list(self.public_flow_ids),
            "physical_sector_ids": list(self.physical_sector_ids),
            "state_templates": _sparse_mapping(
                self.state_template_count, self.state_template_changes
            ),
            "source_templates": _sparse_mapping(
                self.source_template_count, self.source_template_changes
            ),
            "direct_executors": _sparse_mapping(
                self.direct_executor_count, self.direct_executor_changes
            ),
            "parameter_slots": _sparse_mapping(
                self.parameter_slot_count, self.parameter_slot_changes
            ),
        }

    def _digest_payload(self) -> dict[str, object]:
        payload = self.to_mapping()
        payload["bijection_digest"] = None
        return payload


@dataclass(frozen=True, slots=True)
class RecurrenceSharedLoweringResult(Generic[_T]):
    output: _T
    schedule_digest: str
    native_schedule_semantic_digest: str
    remap: RecurrenceProcessRemap


@dataclass(slots=True)
class _PendingLowering(Generic[_T]):
    logical: RecurrenceBuilderLogicalInputV1
    direct_executor_count: int
    parameter_slot_count: int
    schedule_digest: str
    sharing_domain_digest: str
    native_schedule_semantic_digest: str
    future: Future[_T]


class RecurrenceScheduleLoweringCache(Generic[_T]):
    """Thread-safe single-flight cache keyed before native lowering."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._pending: dict[str, Future[_T]] = {}
        self._processes: list[_PendingLowering[_T]] = []

    def lower_once(self, digest: str, lower: Callable[[], _T]) -> _T:
        key = _require_sha256(digest, "pre-lowering schedule digest")
        with self._lock:
            pending = self._pending.get(key)
            if pending is None:
                pending = Future()
                self._pending[key] = pending
                owner = True
            else:
                owner = False
        if not owner:
            return pending.result()
        try:
            result = lower()
        except BaseException as exc:
            pending.set_exception(exc)
            try:
                pending.exception()
            finally:
                raise
        pending.set_result(result)
        return result

    def lower_process(
        self,
        logical: RecurrenceBuilderLogicalInputV1,
        *,
        schedule_digest: str,
        sharing_domain_digest: str,
        native_schedule_semantic_digest: str,
        direct_executor_count: int,
        parameter_slot_count: int,
        lower: Callable[[], _T],
    ) -> RecurrenceSharedLoweringResult[_T]:
        """Lower once across exact process-isomorphic recurrence schedules."""

        digest = _require_sha256(schedule_digest, "pre-lowering schedule digest")
        sharing_domain = _require_sha256(
            sharing_domain_digest,
            "pre-lowering schedule-sharing domain digest",
        )
        native_semantics = _require_sha256(
            native_schedule_semantic_digest,
            "native schedule semantic digest",
        )
        executor_count = _nonnegative_integer(
            direct_executor_count, "direct-executor count"
        )
        parameter_count = _nonnegative_integer(
            parameter_slot_count, "parameter-slot count"
        )
        with self._lock:
            owner: _PendingLowering[_T] | None = None
            remap: RecurrenceProcessRemap | None = None
            for candidate in self._processes:
                if (
                    candidate.sharing_domain_digest != sharing_domain
                    or candidate.direct_executor_count != executor_count
                    or candidate.parameter_slot_count != parameter_count
                ):
                    continue
                remap = exact_recurrence_process_bijection(
                    candidate.logical,
                    logical,
                    direct_executor_count=executor_count,
                    parameter_slot_count=parameter_count,
                )
                if remap is not None:
                    owner = candidate
                    break
            if owner is None:
                future: Future[_T] = Future()
                owner = _PendingLowering(
                    logical=logical,
                    direct_executor_count=executor_count,
                    parameter_slot_count=parameter_count,
                    schedule_digest=digest,
                    sharing_domain_digest=sharing_domain,
                    native_schedule_semantic_digest=native_semantics,
                    future=future,
                )
                self._processes.append(owner)
                remap = RecurrenceProcessRemap.identity(
                    logical,
                    direct_executor_count=executor_count,
                    parameter_slot_count=parameter_count,
                )
                owns_lowering = True
            else:
                owns_lowering = False
        assert remap is not None
        if owns_lowering:
            try:
                output = lower()
            except BaseException as exc:
                owner.future.set_exception(exc)
                try:
                    owner.future.exception()
                finally:
                    raise
            owner.future.set_result(output)
        else:
            output = owner.future.result()
        return RecurrenceSharedLoweringResult(
            output=output,
            schedule_digest=owner.schedule_digest,
            native_schedule_semantic_digest=(owner.native_schedule_semantic_digest),
            remap=remap,
        )


class RecurrenceScheduleProcess(Protocol):
    process_id: str
    recurrence_schedule_path: Path
    recurrence_schedule_digest: str
    recurrence_native_schedule_semantic_digest: str
    recurrence_schedule_size_bytes: int
    recurrence_schedule_sha256: str
    recurrence_schedule_member_count: int
    recurrence_schedule_unpacked_size_bytes: int
    recurrence_schedule_index_sha256: str
    builder_input_sha256: str
    process_digest: str
    process_support_mask: int
    recurrence_process_remap: RecurrenceProcessRemap
    recurrence_process_executor_pack: RecurrenceProcessExecutorPack


def _recurrence_schedule_identity_payload(
    logical: RecurrenceBuilderLogicalInputV1,
    *,
    prepared_kernel_pack_digest: str,
    direct_template_catalog_digest: str,
    point_tile_size: int,
    workspace_mib: int,
) -> dict[str, object]:
    domain = _recurrence_schedule_domain_payload(
        prepared_kernel_pack_digest=prepared_kernel_pack_digest,
        direct_template_catalog_digest=direct_template_catalog_digest,
        point_tile_size=point_tile_size,
        workspace_mib=workspace_mib,
    )
    return {
        "contract": "pyamplicol-recurrence-prelower-schedule-identity-v1",
        "logical": _schedule_plain(logical),
        **domain,
    }


def _recurrence_schedule_domain_payload(
    *,
    prepared_kernel_pack_digest: str,
    direct_template_catalog_digest: str,
    point_tile_size: int,
    workspace_mib: int,
) -> dict[str, object]:
    return {
        "prepared_kernel_pack_digest": _require_sha256(
            prepared_kernel_pack_digest, "prepared-kernel pack digest"
        ),
        "direct_template_catalog_digest": _require_sha256(
            direct_template_catalog_digest, "direct-template catalog digest"
        ),
        "point_tile_size": _positive_integer(point_tile_size, "point tile size"),
        "workspace_mib": _positive_integer(workspace_mib, "workspace MiB"),
    }


def _recurrence_schedule_semantic_digests(
    logical: RecurrenceBuilderLogicalInputV1,
    *,
    prepared_kernel_pack_digest: str,
    direct_template_catalog_digest: str,
    point_tile_size: int,
    workspace_mib: int,
    relation_discovery: Mapping[str, object] | None = None,
) -> tuple[str, str, str]:
    """Return native, request, and sharing-domain schedule identities."""

    domain = _recurrence_schedule_domain_payload(
        prepared_kernel_pack_digest=prepared_kernel_pack_digest,
        direct_template_catalog_digest=direct_template_catalog_digest,
        point_tile_size=point_tile_size,
        workspace_mib=workspace_mib,
    )
    payload = {
        "contract": "pyamplicol-recurrence-prelower-schedule-identity-v1",
        "logical": _schedule_plain(logical),
        **domain,
    }
    native_digest = _canonical_digest(payload)
    request_payload = dict(payload)
    sharing_domain_payload = {
        "contract": "pyamplicol-recurrence-schedule-sharing-domain-v1",
        **domain,
    }
    if relation_discovery is not None:
        plain_relation = _schedule_plain(relation_discovery)
        request_payload["relation_discovery"] = plain_relation
        sharing_domain_payload["relation_discovery"] = plain_relation
    return (
        native_digest,
        _canonical_digest(request_payload),
        _canonical_digest(sharing_domain_payload),
    )


def recurrence_helicity_selector_schedule_digest(
    base_schedule_digest: str,
    helicity_dispatch_sha256: str,
) -> str:
    """Bind one Direct plan and its exact-helicity dispatch into one root ID."""

    return _canonical_digest(
        {
            "contract": "pyamplicol-recurrence-helicity-selector-schedule-v1",
            "base_schedule_digest": _require_sha256(
                base_schedule_digest,
                "helicity-selector base schedule digest",
            ),
            "helicity_dispatch": {
                "abi": "pyamplicol-recurrence-helicity-dispatch-v1",
                "sha256": _require_sha256(
                    helicity_dispatch_sha256,
                    "helicity-selector dispatch SHA-256",
                ),
            },
        }
    )


def recurrence_native_schedule_semantic_digest(
    logical: RecurrenceBuilderLogicalInputV1,
    *,
    prepared_kernel_pack_digest: str,
    direct_template_catalog_digest: str,
    point_tile_size: int,
    workspace_mib: int,
) -> str:
    """Return the relation-independent semantic identity embedded by native lowering.

    Only concrete process ownership is normalized. All model, crossing,
    selector, color, proof, parameter, and runtime-option semantics remain in
    the digest. Relation-discovery policy is deliberately excluded: it governs
    the lowering request, while certified relations themselves are authenticated
    separately and a request that applies none must retain the source schedule's
    byte identity.
    """

    return _canonical_digest(
        _recurrence_schedule_identity_payload(
            logical,
            prepared_kernel_pack_digest=prepared_kernel_pack_digest,
            direct_template_catalog_digest=direct_template_catalog_digest,
            point_tile_size=point_tile_size,
            workspace_mib=workspace_mib,
        )
    )


def recurrence_schedule_semantic_digest(
    logical: RecurrenceBuilderLogicalInputV1,
    *,
    prepared_kernel_pack_digest: str,
    direct_template_catalog_digest: str,
    point_tile_size: int,
    workspace_mib: int,
    relation_discovery: Mapping[str, object] | None = None,
) -> str:
    """Return the policy-bearing request/cache identity for schedule lowering.

    This identity isolates discovery policies in temporary paths, lowering
    cache ownership, and artifact grouping. Native serialization receives
    :func:`recurrence_native_schedule_semantic_digest` instead, so policy alone
    cannot perturb a schedule when no certified relation is applied.
    """

    payload = _recurrence_schedule_identity_payload(
        logical,
        prepared_kernel_pack_digest=prepared_kernel_pack_digest,
        direct_template_catalog_digest=direct_template_catalog_digest,
        point_tile_size=point_tile_size,
        workspace_mib=workspace_mib,
    )
    if relation_discovery is not None:
        payload["relation_discovery"] = _schedule_plain(relation_discovery)
    return _canonical_digest(payload)


def build_recurrence_process_executor_pack(
    *,
    direct_catalog: object,
    kernel_pack: object,
    required_executor_ids: Sequence[int],
    runtime_layout_digest: str,
) -> RecurrenceProcessExecutorPack:
    """Project one validated model catalog to its exact fixed-plan executor set."""

    direct_mapping = _object_mapping(direct_catalog, "direct-template catalog")
    pack_mapping = _object_mapping(kernel_pack, "prepared-kernel pack")
    raw_templates = _object_sequence(
        direct_mapping.get("templates"), "direct-template catalog templates"
    )
    catalog_count = len(raw_templates)
    templates: dict[int, Mapping[str, object]] = {}
    for expected_id, raw_template in enumerate(raw_templates):
        template = _require_mapping(raw_template, "direct-template descriptor")
        executor_id = _wire_u32(
            template.get("direct_executor_id"), "direct-template executor ID"
        )
        if executor_id != expected_id or executor_id in templates:
            raise RecurrenceScheduleSharingError(
                "direct-template executor IDs must be dense, ordered, and unique"
            )
        templates[executor_id] = template

    selected_ids = tuple(
        _wire_u32(value, "required direct-executor ID")
        for value in required_executor_ids
    )
    if selected_ids != tuple(sorted(set(selected_ids))):
        raise RecurrenceScheduleSharingError(
            "required direct-executor IDs must be sorted and unique"
        )
    if any(executor_id not in templates for executor_id in selected_ids):
        raise RecurrenceScheduleSharingError(
            "required direct-executor ID is absent from its complete catalog"
        )

    raw_kernels = _object_sequence(
        pack_mapping.get("kernels"), "prepared-kernel pack kernels"
    )
    kernels: dict[int, Mapping[str, object]] = {}
    for raw_kernel in raw_kernels:
        kernel = _require_mapping(raw_kernel, "prepared-kernel descriptor")
        kernel_id = _wire_u32(kernel.get("kernel_id"), "prepared-kernel ID")
        if kernel_id in kernels:
            raise RecurrenceScheduleSharingError(
                f"prepared-kernel pack repeats kernel {kernel_id}"
            )
        kernels[kernel_id] = kernel

    backend = _wire_text(direct_mapping.get("backend"), "direct backend")
    if backend != pack_mapping.get("backend") or backend not in _DIRECT_BACKEND:
        raise RecurrenceScheduleSharingError(
            "direct-template and prepared-kernel backends disagree"
        )
    target = _require_mapping(pack_mapping.get("target"), "prepared-kernel target")
    target_triple = _wire_text(target.get("target_triple"), "target triple")
    if target_triple != direct_mapping.get("target_triple"):
        raise RecurrenceScheduleSharingError(
            "direct-template and prepared-kernel targets disagree"
        )
    portable = target.get("portable")
    if type(portable) is not bool or portable != direct_mapping.get("portable"):
        raise RecurrenceScheduleSharingError(
            "direct-template and prepared-kernel portability disagree"
        )
    if target.get("word_bits") != 64 or target.get("endianness") != "little":
        raise RecurrenceScheduleSharingError(
            "process executor packs require a 64-bit little-endian target"
        )
    cpu_features = tuple(
        _wire_text(value, "target CPU feature")
        for value in _object_sequence(target.get("cpu_features"), "target CPU features")
    )
    if cpu_features != tuple(sorted(set(cpu_features))):
        raise RecurrenceScheduleSharingError(
            "target CPU features must be sorted and unique"
        )

    descriptors = tuple(
        _encode_process_executor_descriptor(
            templates[executor_id],
            backend=backend,
            kernels=kernels,
        )
        for executor_id in selected_ids
    )
    return RecurrenceProcessExecutorPack(
        compiled_model_digest=_wire_sha256(
            direct_mapping.get("compiled_model_digest"), "compiled-model digest"
        ),
        recurrence_template_catalog_digest=_wire_sha256(
            direct_mapping.get("recurrence_template_catalog_digest"),
            "recurrence-template catalog digest",
        ),
        prepared_kernel_pack_digest=_wire_sha256(
            direct_mapping.get("prepared_kernel_pack_digest"),
            "prepared-kernel pack digest",
        ),
        direct_template_catalog_digest=_wire_sha256(
            direct_mapping.get("catalog_digest"), "direct-template catalog digest"
        ),
        runtime_layout_digest=_wire_sha256(
            runtime_layout_digest, "runtime-layout digest"
        ),
        backend=backend,
        target_triple=target_triple,
        portable=portable,
        cpu_features=cpu_features,
        catalog_executor_count=catalog_count,
        executor_ids=selected_ids,
        descriptor_payloads=descriptors,
    )


def _encode_process_executor_descriptor(
    template: Mapping[str, object],
    *,
    backend: str,
    kernels: Mapping[int, Mapping[str, object]],
) -> bytes:
    executor_id = _wire_u32(template.get("direct_executor_id"), "direct executor ID")
    role = _wire_text(template.get("role"), "direct executor role")
    role_tag = _DIRECT_ROLE.get(role)
    if role_tag is None:
        raise RecurrenceScheduleSharingError(
            f"unsupported direct executor role {role!r}"
        )
    destination_count = _wire_u32(
        template.get("destination_component_count"),
        "direct destination component count",
        positive=True,
    )
    binding = _require_mapping(
        template.get("payload_binding"), "direct executor payload binding"
    )
    binding_kind = _wire_text(binding.get("kind"), "direct binding kind")
    exact_factor_slots = _object_sequence(
        binding.get("exact_factor_scalar_slots"), "exact-factor scalar slots"
    )
    flags = _DIRECT_FLAG_USES_EXACT_FACTOR if len(exact_factor_slots) != 0 else 0
    if role == "source":
        record_kind = _DIRECT_BINDING_SOURCE
        body = b""
    elif binding_kind == "rusticol-intrinsic":
        record_kind = _DIRECT_BINDING_INTRINSIC
        body = _encode_intrinsic_binding(binding, role=role)
    elif binding_kind == "prepared-direct-call" and backend == "jit":
        record_kind = _DIRECT_BINDING_JIT
        body = _encode_jit_binding(binding, template=template, kernels=kernels)
    elif binding_kind == "prepared-direct-call" and backend in {"cpp", "asm"}:
        record_kind = _DIRECT_BINDING_NATIVE
        body = _encode_native_binding(binding, template=template, kernels=kernels)
    else:
        raise RecurrenceScheduleSharingError(
            f"unsupported executable direct binding {binding_kind!r} for {backend!r}"
        )
    size = 16 + len(body)
    if size > _MISSING_U32:
        raise RecurrenceScheduleSharingError(
            "process direct-executor descriptor exceeds the u32 wire domain"
        )
    return (
        struct.pack(
            "<IIBBHI",
            size,
            executor_id,
            role_tag,
            record_kind,
            flags,
            destination_count,
        )
        + body
    )


def _encode_intrinsic_binding(binding: Mapping[str, object], *, role: str) -> bytes:
    runtime_template = _wire_text(
        binding.get("runtime_template"), "intrinsic runtime template"
    )
    raw_projections = _object_sequence(
        binding.get("scalar_projections"), "intrinsic scalar projections"
    )
    if role == "contribution":
        if len(raw_projections) != 1:
            raise RecurrenceScheduleSharingError(
                "contribution intrinsic requires exactly one scale projection"
            )
        projection = _require_mapping(
            raw_projections[0], "contribution intrinsic scale"
        )
        if projection.get("kind") != "intrinsic-scale-v1":
            raise RecurrenceScheduleSharingError(
                "contribution intrinsic has an unsupported scale projection"
            )
        real_bits = _wire_u64(
            projection.get("constant_real_bits"), "intrinsic real bits"
        )
        imag_bits = _wire_u64(
            projection.get("constant_imag_bits"), "intrinsic imaginary bits"
        )
        raw_parameter = projection.get("parameter_index")
        parameter = (
            _MISSING_U32
            if raw_parameter is None
            else _wire_u32(raw_parameter, "intrinsic parameter index")
        )
        has_scale = 1
    else:
        if raw_projections:
            raise RecurrenceScheduleSharingError(
                "non-contribution intrinsic cannot carry scalar projections"
            )
        real_bits = 0
        imag_bits = 0
        parameter = _MISSING_U32
        has_scale = 0
    return struct.pack(
        "<B3xQQI", has_scale, real_bits, imag_bits, parameter
    ) + _encode_wire_text(runtime_template, "intrinsic runtime template")


def _encode_jit_binding(
    binding: Mapping[str, object],
    *,
    template: Mapping[str, object],
    kernels: Mapping[int, Mapping[str, object]],
) -> bytes:
    kernel_id = _wire_u32(binding.get("prepared_kernel_id"), "prepared JIT kernel ID")
    kernel = kernels.get(kernel_id)
    if kernel is None:
        raise RecurrenceScheduleSharingError(
            f"prepared JIT kernel {kernel_id} is absent"
        )
    evaluator = _require_mapping(
        kernel.get("f64_evaluator_manifest"), "prepared JIT evaluator"
    )
    plane = _require_mapping(
        evaluator.get("plane_application"), "prepared JIT plane application"
    )
    compression = plane.get("compression")
    if type(compression) is not bool:
        raise RecurrenceScheduleSharingError(
            "prepared JIT plane compression must be boolean"
        )
    optimization_level = _wire_u32(
        template.get("optimization_level"), "prepared JIT optimization level"
    )
    source_path = _wire_text(
        binding.get("source_application_path"), "prepared JIT source path"
    )
    source_sha256 = _wire_sha256(
        binding.get("source_application_sha256"), "prepared JIT source digest"
    )
    source_abi = _wire_text(
        binding.get("source_application_abi"), "prepared JIT source ABI"
    )
    parameter_bindings = _object_sequence(
        binding.get("parameter_bindings"), "prepared JIT parameter bindings"
    )
    plane_projections = _object_sequence(
        binding.get("input_plane_projections"), "prepared JIT plane projections"
    )
    scalar_projections = _object_sequence(
        binding.get("scalar_projections"), "prepared JIT scalar projections"
    )
    output_aliases = tuple(
        _wire_u32(value, "prepared JIT output alias")
        for value in _object_sequence(
            binding.get("output_alias_inputs"), "prepared JIT output aliases"
        )
    )
    counts = tuple(
        _wire_u32(len(values), f"prepared JIT {label} count")
        for values, label in (
            (parameter_bindings, "parameter-binding"),
            (plane_projections, "plane-projection"),
            (scalar_projections, "scalar-projection"),
            (output_aliases, "output-alias"),
        )
    )
    return b"".join(
        (
            struct.pack(
                "<IIB3x",
                kernel_id,
                optimization_level,
                int(compression),
            ),
            bytes.fromhex(source_sha256),
            _encode_wire_text(source_path, "prepared JIT source path"),
            _encode_wire_text(source_abi, "prepared JIT source ABI"),
            struct.pack("<4I", *counts),
            *(_encode_parameter_binding(value) for value in parameter_bindings),
            *(_encode_plane_projection(value) for value in plane_projections),
            *(_encode_scalar_projection(value) for value in scalar_projections),
            struct.pack(f"<{len(output_aliases)}I", *output_aliases),
        )
    )


def _encode_native_binding(
    binding: Mapping[str, object],
    *,
    template: Mapping[str, object],
    kernels: Mapping[int, Mapping[str, object]],
) -> bytes:
    kernel_id = _wire_u32(
        binding.get("prepared_kernel_id"), "prepared native kernel ID"
    )
    kernel = kernels.get(kernel_id)
    if kernel is None:
        raise RecurrenceScheduleSharingError(
            f"prepared native kernel {kernel_id} is absent"
        )
    source_path = _wire_text(
        binding.get("source_application_path"), "prepared native library path"
    )
    entry_point = _wire_text(
        binding.get("native_entry_point"), "prepared native entry point"
    )
    coupling = _prepared_native_coupling(binding, kernel)
    if coupling is None:
        has_coupling = 0
        coupling_real_bits = 0
        coupling_imag_bits = 0
    else:
        has_coupling = 1
        coupling_real_bits = _f64_bits(coupling[0])
        coupling_imag_bits = _f64_bits(coupling[1])
    return b"".join(
        (
            struct.pack(
                "<IB3xQQ",
                kernel_id,
                has_coupling,
                coupling_real_bits,
                coupling_imag_bits,
            ),
            _encode_wire_text(source_path, "prepared native library path"),
            _encode_wire_text(entry_point, "prepared native entry point"),
        )
    )


def _prepared_native_coupling(
    binding: Mapping[str, object],
    kernel: Mapping[str, object],
) -> tuple[float, float] | None:
    parameter_bindings = _object_sequence(
        binding.get("parameter_bindings"), "native parameter bindings"
    )
    input_contracts = _object_sequence(
        kernel.get("input_contracts"), "native kernel input contracts"
    )
    if len(parameter_bindings) != 2 * len(input_contracts):
        raise RecurrenceScheduleSharingError(
            "native direct parameter bindings do not match the kernel inputs"
        )
    scalars = _object_sequence(
        binding.get("scalar_projections"), "native scalar projections"
    )
    coupling_real: float | None = None
    coupling_imag: float | None = None
    for input_index, raw_contract in enumerate(input_contracts):
        contract = _require_mapping(raw_contract, "native kernel input contract")
        role = contract.get("role")
        if role not in {"coupling-real", "coupling-imag"}:
            continue
        value = _native_literal_binding(
            parameter_bindings,
            scalars,
            2 * input_index,
        )
        imaginary = _native_literal_binding(
            parameter_bindings,
            scalars,
            2 * input_index + 1,
        )
        if imaginary != 0.0:
            raise RecurrenceScheduleSharingError(
                "native direct coupling scalar lane has a nonzero imaginary part"
            )
        if role == "coupling-real":
            if coupling_real is not None:
                raise RecurrenceScheduleSharingError(
                    "native direct kernel repeats its real coupling input"
                )
            coupling_real = value
        else:
            if coupling_imag is not None:
                raise RecurrenceScheduleSharingError(
                    "native direct kernel repeats its imaginary coupling input"
                )
            coupling_imag = value
    if coupling_real is None and coupling_imag is None:
        return None
    return (
        0.0 if coupling_real is None else coupling_real,
        0.0 if coupling_imag is None else coupling_imag,
    )


def _native_literal_binding(
    parameter_bindings: Sequence[object],
    scalars: Sequence[object],
    parameter_index: int,
) -> float:
    binding = _require_mapping(
        parameter_bindings[parameter_index], "native scalar parameter binding"
    )
    if binding.get("kind") != "scalar":
        raise RecurrenceScheduleSharingError(
            "native direct coupling is not bound to a scalar"
        )
    scalar_index = _wire_u32(binding.get("index"), "native scalar index")
    if scalar_index >= len(scalars):
        raise RecurrenceScheduleSharingError(
            "native direct coupling scalar index is out of bounds"
        )
    scalar = _require_mapping(scalars[scalar_index], "native literal scalar")
    raw_value = scalar.get("value")
    if (
        scalar.get("kind") != "literal"
        or isinstance(raw_value, bool)
        or not isinstance(raw_value, int | float)
    ):
        raise RecurrenceScheduleSharingError(
            "native direct coupling is not an immutable literal"
        )
    value = float(raw_value)
    if not math.isfinite(value):
        raise RecurrenceScheduleSharingError(
            "native direct coupling literal is not finite"
        )
    return value


def _encode_parameter_binding(value: object) -> bytes:
    binding = _require_mapping(value, "JIT parameter binding")
    kind = binding.get("kind")
    tag = 0 if kind == "plane" else 1 if kind == "scalar" else None
    if tag is None:
        raise RecurrenceScheduleSharingError(
            f"unsupported JIT parameter binding {kind!r}"
        )
    index = _wire_u32(binding.get("index"), "JIT parameter-binding index")
    return struct.pack("<B3xI", tag, index)


def _encode_plane_projection(value: object) -> bytes:
    projection = _require_mapping(value, "JIT plane projection")
    kind = projection.get("kind")
    if kind == "parent-current":
        tag = 0
        operand = _wire_u8(projection.get("parent"), "JIT parent index")
        component = _wire_u16(projection.get("component"), "JIT parent component")
        imaginary = _wire_bool(projection.get("imaginary"), "JIT imaginary flag")
    elif kind == "momentum":
        tag = 1
        operand = _wire_u8(projection.get("operand"), "JIT momentum operand")
        component = _wire_u16(
            projection.get("lorentz_component"), "JIT Lorentz component"
        )
        imaginary = False
    elif kind in {"destination-current", "destination-amplitude"}:
        tag = 2 if kind == "destination-current" else 3
        operand = 0
        component = _wire_u16(projection.get("component"), "JIT destination component")
        imaginary = _wire_bool(projection.get("imaginary"), "JIT imaginary flag")
    else:
        raise RecurrenceScheduleSharingError(
            f"unsupported JIT plane projection {kind!r}"
        )
    return struct.pack("<BBHB3x", tag, operand, component, int(imaginary))


def _encode_scalar_projection(value: object) -> bytes:
    projection = _require_mapping(value, "JIT scalar projection")
    kind = projection.get("kind")
    if kind == "exact-factor":
        tag = 0
        imaginary = _wire_bool(projection.get("imaginary"), "JIT imaginary flag")
        index = 0
        bits = 0
    elif kind == "parameter":
        tag = 1
        imaginary = _wire_bool(projection.get("imaginary"), "JIT imaginary flag")
        index = _wire_u32(projection.get("index"), "JIT parameter index")
        bits = 0
    elif kind == "literal":
        tag = 2
        imaginary = False
        index = 0
        raw_value = projection.get("value")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise RecurrenceScheduleSharingError("JIT literal must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise RecurrenceScheduleSharingError("JIT literal must be finite")
        bits = _f64_bits(value)
    else:
        raise RecurrenceScheduleSharingError(
            f"unsupported JIT scalar projection {kind!r}"
        )
    return struct.pack("<BBHIQ", tag, int(imaginary), 0, index, bits)


def _object_mapping(value: object, context: str) -> Mapping[str, object]:
    operation = getattr(value, "to_dict", None)
    if not callable(operation):
        raise RecurrenceScheduleSharingError(f"{context} is not a typed model object")
    return _require_mapping(operation(), context)


def _require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrenceScheduleSharingError(f"{context} must be a mapping")
    return value


def _object_sequence(value: object, context: str) -> tuple[object, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise RecurrenceScheduleSharingError(f"{context} must be a sequence")
    return tuple(value)


def _wire_text(value: object, context: str) -> str:
    return _nonempty_text(value, context)


def _wire_sha256(value: object, context: str) -> str:
    return _require_sha256(value, context)


def _wire_bool(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise RecurrenceScheduleSharingError(f"{context} must be boolean")
    return value


def _wire_integer(
    value: object,
    context: str,
    maximum: int,
    *,
    positive: bool = False,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < int(positive)
        or value > maximum
    ):
        qualifier = "positive " if positive else ""
        raise RecurrenceScheduleSharingError(
            f"{context} must be a {qualifier}integer no larger than {maximum}"
        )
    return value


def _wire_u8(value: object, context: str) -> int:
    return _wire_integer(value, context, (1 << 8) - 1)


def _wire_u16(value: object, context: str) -> int:
    return _wire_integer(value, context, (1 << 16) - 1)


def _wire_u32(value: object, context: str, *, positive: bool = False) -> int:
    return _wire_integer(value, context, _MISSING_U32, positive=positive)


def _wire_u64(value: object, context: str) -> int:
    return _wire_integer(value, context, (1 << 64) - 1)


def _f64_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _encode_wire_text(value: str, context: str) -> bytes:
    encoded = value.encode("utf-8")
    if not encoded or len(encoded) > _MISSING_U32:
        raise RecurrenceScheduleSharingError(
            f"{context} has an invalid UTF-8 wire length"
        )
    return struct.pack("<I", len(encoded)) + encoded


def exact_recurrence_process_bijection(
    root: RecurrenceBuilderLogicalInputV1,
    target: RecurrenceBuilderLogicalInputV1,
    *,
    direct_executor_count: int,
    parameter_slot_count: int,
) -> RecurrenceProcessRemap | None:
    """Return an exact root-to-target schedule binding or fail closed.

    V1 sharing is intentionally limited to complete topology-replay schedules.
    The relation is an all-outgoing external-state isomorphism, not a particle
    label erasure: every source state, color forest, replay selector, parameter
    projection, and fermion-pairing row must map bijectively.
    """

    if (
        root.layout != "topology-replay"
        or target.layout != "topology-replay"
        or len(root.external_legs) != len(target.external_legs)
        or root.coupling_limits != target.coupling_limits
        or root.parameter_projection != target.parameter_projection
        or root.semantic_template_references != target.semantic_template_references
        or root.selected_public_flow_ids is not None
        or target.selected_public_flow_ids is not None
        or root.selected_source_coverage is not None
        or target.selected_source_coverage is not None
        or not _fixed_semantic_digests_match(root, target)
    ):
        return None

    candidates = _source_slot_bijections(root.external_legs, target.external_legs)
    for source_slots in candidates:
        source_relation = _external_source_relation(
            root.external_legs,
            target.external_legs,
            source_slots,
        )
        if source_relation is None:
            continue
        (
            momentum_signs,
            helicity_signs,
            source_state_offsets,
            source_state_indices,
            source_template_changes,
        ) = source_relation
        sector_ids = _sector_bijection(
            root.physical_sectors,
            target.physical_sectors,
            source_slots,
        )
        if sector_ids is None:
            continue
        flow_ids = _flow_bijection(
            root.public_flows,
            target.public_flows,
            source_slots,
            sector_ids,
        )
        if flow_ids is None:
            continue
        # Direct-Arena proof metadata still owns physical sector IDs. Until
        # that metadata has an explicit remap ABI, share only schedules whose
        # physical/public sector numbering is already identical.
        if sector_ids != tuple(range(len(sector_ids))) or flow_ids != tuple(
            range(len(flow_ids))
        ):
            continue
        if not _replay_contracts_match(
            root,
            target,
            source_slots=source_slots,
            sector_ids=sector_ids,
        ):
            continue
        if not _fermion_pairing_contracts_match(
            root.fermion_pairing_catalog,
            target.fermion_pairing_catalog,
            source_slots,
        ):
            continue

        result = RecurrenceProcessRemap(
            source_slots=source_slots,
            source_momentum_signs=momentum_signs,
            source_helicity_signs=helicity_signs,
            source_state_offsets=source_state_offsets,
            source_state_indices=source_state_indices,
            public_flow_ids=flow_ids,
            physical_sector_ids=tuple(range(_physical_sector_domain_count(root))),
            state_template_count=_template_count(root, "current-state"),
            source_template_count=_template_count(root, "source"),
            direct_executor_count=_nonnegative_integer(
                direct_executor_count, "direct-executor count"
            ),
            parameter_slot_count=_nonnegative_integer(
                parameter_slot_count, "parameter-slot count"
            ),
            source_template_changes=source_template_changes,
        )
        relation_digest = _canonical_digest(
            {
                "contract": "pyamplicol-recurrence-process-bijection-v1",
                "root_process": root.process_id,
                "target_process": target.process_id,
                "root_variable_semantics": _variable_semantic_digests(root),
                "target_variable_semantics": _variable_semantic_digests(target),
                "root_pairing": _pairing_proof_digests(root.fermion_pairing_catalog),
                "target_pairing": _pairing_proof_digests(
                    target.fermion_pairing_catalog
                ),
                "root_sector_proofs": [
                    [sector.closure_proof_algorithm, sector.closure_proof_digest]
                    for sector in root.physical_sectors
                ],
                "target_sector_proofs": [
                    [sector.closure_proof_algorithm, sector.closure_proof_digest]
                    for sector in target.physical_sectors
                ],
                "root_replay_proofs": [
                    [partition.proof_algorithm, partition.proof_digest]
                    for partition in root.replay_partitions
                ],
                "target_replay_proofs": [
                    [partition.proof_algorithm, partition.proof_digest]
                    for partition in target.replay_partitions
                ],
                "remap": result._digest_payload(),
            }
        )
        return result.with_digest(relation_digest)
    return None


def _fixed_semantic_digests_match(
    root: RecurrenceBuilderLogicalInputV1,
    target: RecurrenceBuilderLogicalInputV1,
) -> bool:
    root_rows = {row.role: row.digest for row in root.semantic_digests}
    target_rows = {row.role: row.digest for row in target.semantic_digests}
    return set(root_rows) == set(target_rows) and {
        role: digest
        for role, digest in root_rows.items()
        if role not in _PROCESS_VARIABLE_SEMANTIC_ROLES
    } == {
        role: digest
        for role, digest in target_rows.items()
        if role not in _PROCESS_VARIABLE_SEMANTIC_ROLES
    }


def _variable_semantic_digests(
    logical: RecurrenceBuilderLogicalInputV1,
) -> dict[str, str]:
    return {
        row.role: row.digest
        for row in logical.semantic_digests
        if row.role in _PROCESS_VARIABLE_SEMANTIC_ROLES
    }


def _source_slot_bijections(
    root: tuple[RecurrenceExternalLegV1, ...],
    target: tuple[RecurrenceExternalLegV1, ...],
) -> tuple[tuple[int, ...], ...]:
    root_groups: dict[tuple[object, ...], list[int]] = {}
    target_groups: dict[tuple[object, ...], list[int]] = {}
    for leg in root:
        root_groups.setdefault(_external_leg_orbit_key(leg), []).append(leg.source_slot)
    for leg in target:
        target_groups.setdefault(_external_leg_orbit_key(leg), []).append(
            leg.source_slot
        )
    if set(root_groups) != set(target_groups):
        return ()
    groups: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    count = 1
    for key in sorted(root_groups, key=repr):
        left = tuple(root_groups[key])
        right = tuple(target_groups[key])
        if len(left) != len(right):
            return ()
        permutations = tuple(itertools.permutations(right))
        count *= len(permutations)
        if count > _MAX_SOURCE_BIJECTIONS:
            return ()
        groups.append((left, permutations))
    result: list[tuple[int, ...]] = []
    for selection in itertools.product(*(rows[1] for rows in groups)):
        mapping = [0] * len(root)
        for (root_slots, _), target_slots in zip(groups, selection, strict=True):
            for root_slot, target_slot in zip(root_slots, target_slots, strict=True):
                mapping[root_slot] = target_slot
        result.append(tuple(mapping))
    return tuple(result)


def _external_leg_orbit_key(leg: RecurrenceExternalLegV1) -> tuple[object, ...]:
    return (
        leg.outgoing_pdg,
        leg.is_fermionic,
        tuple(
            sorted(
                (
                    state.current_state_template_id,
                    state.chirality,
                    state.spin_state,
                    state.crossing_phase.canonical_key,
                )
                for state in leg.source_states
            )
        ),
    )


def _external_source_relation(
    root: tuple[RecurrenceExternalLegV1, ...],
    target: tuple[RecurrenceExternalLegV1, ...],
    source_slots: tuple[int, ...],
) -> (
    tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, int], ...],
    ]
    | None
):
    momentum_signs: list[int] = []
    helicity_signs: list[int] = []
    source_state_offsets = [0]
    source_state_indices: list[int] = []
    source_templates: dict[int, int] = {}
    reverse_source_templates: dict[int, int] = {}
    for root_slot, target_slot in enumerate(source_slots):
        left = root[root_slot]
        right = target[target_slot]
        if (
            left.outgoing_pdg != right.outgoing_pdg
            or left.is_fermionic != right.is_fermionic
            or len(left.source_states) != len(right.source_states)
        ):
            return None
        right_states: dict[tuple[object, ...], RecurrenceSourceStateV1] = {}
        for state in right.source_states:
            key = _source_state_execution_key(state)
            if key in right_states:
                return None
            right_states[key] = state
        momentum_factor: int | None = None
        helicity_factor: int | None = None
        state_indices = [0] * len(left.source_states)
        for state in left.source_states:
            other = right_states.get(_source_state_execution_key(state))
            if other is None:
                return None
            state_indices[state.state_index] = other.state_index
            candidate_momentum = other.momentum_sign * state.momentum_sign
            if momentum_factor is None:
                momentum_factor = candidate_momentum
            elif momentum_factor != candidate_momentum:
                return None
            candidate_helicity = _signed_value_relation(
                state.public_helicity,
                other.public_helicity,
            )
            if candidate_helicity is None:
                return None
            if helicity_factor is None:
                helicity_factor = candidate_helicity
            elif helicity_factor != candidate_helicity:
                return None
            previous = source_templates.setdefault(
                state.source_template_id, other.source_template_id
            )
            reverse = reverse_source_templates.setdefault(
                other.source_template_id, state.source_template_id
            )
            if (
                previous != other.source_template_id
                or reverse != state.source_template_id
            ):
                return None
        momentum_signs.append(1 if momentum_factor is None else momentum_factor)
        helicity_signs.append(1 if helicity_factor is None else helicity_factor)
        if not _is_permutation(state_indices):
            return None
        source_state_indices.extend(state_indices)
        source_state_offsets.append(len(source_state_indices))
    changes = tuple(
        (source, target_id)
        for source, target_id in sorted(source_templates.items())
        if source != target_id
    )
    return (
        tuple(momentum_signs),
        tuple(helicity_signs),
        tuple(source_state_offsets),
        tuple(source_state_indices),
        changes,
    )


def _source_state_execution_key(
    state: RecurrenceSourceStateV1,
) -> tuple[object, ...]:
    return (
        state.current_state_template_id,
        state.chirality,
        state.spin_state,
        state.crossing_phase.canonical_key,
    )


def _signed_value_relation(left: int, right: int) -> int | None:
    if left == 0 or right == 0:
        return 1 if left == right else None
    if right == left:
        return 1
    if right == -left:
        return -1
    return None


def _sector_bijection(
    root: tuple[RecurrencePhysicalLCSectorV1, ...],
    target: tuple[RecurrencePhysicalLCSectorV1, ...],
    source_slots: tuple[int, ...],
) -> tuple[int, ...] | None:
    inverse = _inverse_permutation(source_slots)
    targets: dict[tuple[object, ...], int] = {}
    for sector in target:
        key = _sector_key(sector, inverse)
        if key in targets:
            return None
        targets[key] = sector.sector_id
    result: list[int] = []
    for sector in root:
        target_id = targets.get(_sector_key(sector, tuple(range(len(source_slots)))))
        if target_id is None:
            return None
        result.append(target_id)
    return tuple(result) if _is_permutation(result) else None


def _sector_key(
    sector: RecurrencePhysicalLCSectorV1,
    slots: tuple[int, ...],
) -> tuple[object, ...]:
    def mapped(values: Sequence[int]) -> tuple[int, ...]:
        return tuple(slots[value] for value in values)

    return (
        sector.kind,
        slots[sector.closure_source_slot],
        sector.closure_proof_algorithm,
        tuple(
            (
                slots[row.fundamental_source_slot],
                slots[row.antifundamental_source_slot],
                mapped(row.adjoint_source_slots),
                mapped(row.singlet_source_slots),
            )
            for row in sector.open_strings
        ),
        mapped(sector.trace_source_slots),
        mapped(sector.singlet_source_slots),
        mapped(sector.word_source_slots),
    )


def _flow_bijection(
    root: tuple[RecurrencePublicLCFlowV1, ...],
    target: tuple[RecurrencePublicLCFlowV1, ...],
    source_slots: tuple[int, ...],
    sector_ids: tuple[int, ...],
) -> tuple[int, ...] | None:
    inverse_slots = _inverse_permutation(source_slots)
    inverse_sectors = _inverse_permutation(sector_ids)
    targets: dict[tuple[object, ...], int] = {}
    for flow in target:
        permutation = tuple(
            inverse_slots[flow.source_slot_permutation[source_slots[root_slot]]]
            for root_slot in range(len(source_slots))
        )
        key = (
            inverse_sectors[flow.construction_sector_id],
            tuple(inverse_slots[value] for value in flow.word_source_slots),
            permutation,
            flow.reduction_weight.canonical_key,
        )
        if key in targets:
            return None
        targets[key] = flow.flow_id
    result = []
    for flow in root:
        key = (
            flow.construction_sector_id,
            flow.word_source_slots,
            flow.source_slot_permutation,
            flow.reduction_weight.canonical_key,
        )
        target_id = targets.get(key)
        if target_id is None:
            return None
        result.append(target_id)
    return tuple(result) if _is_permutation(result) else None


def _replay_contracts_match(
    root: RecurrenceBuilderLogicalInputV1,
    target: RecurrenceBuilderLogicalInputV1,
    *,
    source_slots: tuple[int, ...],
    sector_ids: tuple[int, ...],
) -> bool:
    inverse_sectors = _inverse_permutation(sector_ids)
    left = tuple(
        _replay_partition_key(
            row,
            tuple(range(len(source_slots))),
            tuple(range(len(source_slots))),
            tuple(range(len(sector_ids))),
        )
        for row in root.replay_partitions
    )
    right = tuple(
        _replay_partition_key(
            row,
            source_slots,
            _inverse_permutation(source_slots),
            inverse_sectors,
        )
        for row in target.replay_partitions
    )
    return left == right


def _replay_partition_key(
    row: object,
    root_to_concrete: tuple[int, ...],
    concrete_to_root: tuple[int, ...],
    sectors: tuple[int, ...],
) -> tuple[object, ...]:
    def conjugate(values: Sequence[int]) -> tuple[int, ...]:
        return tuple(
            concrete_to_root[values[root_to_concrete[root_slot]]]
            for root_slot in range(len(root_to_concrete))
        )

    return (
        sectors[row.representative_sector_id],
        sectors[row.materialized_sector_id],
        row.proof_algorithm,
        tuple(
            (
                sectors[target.sector_id],
                conjugate(target.external_permutation),
                conjugate(target.source_slot_permutation),
                target.amplitude_phase.canonical_key,
                target.fermion_sign,
            )
            for target in row.targets
        ),
    )


def _fermion_pairing_contracts_match(
    root: object,
    target: object,
    source_slots: tuple[int, ...],
) -> bool:
    if (root is None) != (target is None):
        return False
    if root is None:
        return True
    inverse = _inverse_permutation(source_slots)
    return _pairing_key(root, tuple(range(len(source_slots)))) == _pairing_key(
        target, inverse
    )


def _pairing_key(catalog: object, slots: tuple[int, ...]) -> tuple[object, ...]:
    concrete_to_root = slots
    root_to_concrete = _inverse_permutation(concrete_to_root)
    endpoints = tuple(
        sorted(
            (
                concrete_to_root[row.source_slot],
                row.species_id,
                row.particle_orientation,
                row.color_orientation,
                row.state_template_ids,
                row.anti_state_template_ids,
                row.basis_ids,
                row.color_representations,
            )
            for row in catalog.endpoints
        )
    )
    classes = tuple(
        sorted(
            (
                row.species_id,
                tuple(
                    concrete_to_root[value] for value in row.fundamental_source_slots
                ),
                tuple(
                    concrete_to_root[value]
                    for value in row.antifundamental_source_slots
                ),
                tuple(
                    (
                        concrete_to_root[left],
                        concrete_to_root[right],
                    )
                    for left, right in row.reference_pairings
                ),
                row.pairing_count,
            )
            for row in catalog.pairing_classes
        )
    )
    rules = tuple(
        sorted(
            (
                row.class_pairing_indices,
                tuple(
                    (
                        concrete_to_root[left],
                        concrete_to_root[right],
                    )
                    for left, right in row.endpoint_pairings
                ),
                tuple(
                    concrete_to_root[
                        row.source_slot_permutation[root_to_concrete[root_slot]]
                    ]
                    for root_slot in range(len(slots))
                ),
                tuple(
                    row.lineage_by_source_slot[root_to_concrete[root_slot]]
                    for root_slot in range(len(slots))
                ),
                row.fermion_parity,
                (
                    row.exact_factor.real_numerator,
                    row.exact_factor.real_denominator,
                    row.exact_factor.imag_numerator,
                    row.exact_factor.imag_denominator,
                ),
                row.multiplicity,
                row.proof_algorithm,
            )
            for row in catalog.rules
        )
    )
    return catalog.source_count, endpoints, classes, rules


def _pairing_proof_digests(catalog: object) -> object:
    if catalog is None:
        return None
    return {
        "semantic": catalog.semantic_digest,
        "topology": catalog.topology_digest,
        "endpoints": [row.contract_digest for row in catalog.endpoints],
        "classes": [row.proof_digest for row in catalog.pairing_classes],
        "rules": [row.proof_digest for row in catalog.rules],
    }


class RecurrenceSharedSchedule:
    """One immutable root Direct-Arena schedule."""

    __slots__ = (
        "digest",
        "index_sha256",
        "member_count",
        "process_ids",
        "sha256",
        "size_bytes",
        "source_path",
        "unpacked_size_bytes",
    )

    def __init__(
        self,
        *,
        digest: str,
        source_path: Path,
        sha256: str,
        size_bytes: int,
        member_count: int,
        unpacked_size_bytes: int,
        index_sha256: str,
        process_ids: tuple[str, ...],
    ) -> None:
        self.digest = _require_sha256(digest, "schedule digest")
        self.source_path = source_path
        self.sha256 = _require_sha256(sha256, "schedule payload SHA-256")
        self.size_bytes = _positive_integer(size_bytes, "schedule payload size")
        self.member_count = _positive_integer(member_count, "schedule member count")
        self.unpacked_size_bytes = _positive_integer(
            unpacked_size_bytes, "schedule unpacked size"
        )
        self.index_sha256 = _require_sha256(
            index_sha256, "schedule PACBIN index SHA-256"
        )
        self.process_ids = process_ids

    @property
    def artifact_path(self) -> str:
        return f"recurrence/schedules/{self.digest}/recurrence-runtime.pacbin"

    def to_mapping(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "path": self.artifact_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "member_count": self.member_count,
            "unpacked_size_bytes": self.unpacked_size_bytes,
            "index_sha256": self.index_sha256,
            "process_ids": list(self.process_ids),
        }


class RecurrenceProcessBinding:
    """One independent concrete-process binding to a root schedule."""

    __slots__ = (
        "executor_pack",
        "native_schedule_semantic_digest",
        "payload",
        "process_digest",
        "process_id",
        "process_semantic_digest",
        "process_support_mask",
        "process_support_words",
        "remap",
        "schedule_digest",
        "sha256",
    )

    def __init__(
        self,
        *,
        process_id: str,
        schedule_digest: str,
        native_schedule_semantic_digest: str,
        process_digest: str,
        process_semantic_digest: str,
        process_support_mask: int,
        remap: RecurrenceProcessRemap,
        executor_pack: RecurrenceProcessExecutorPack,
    ) -> None:
        self.process_id = _nonempty_text(process_id, "process ID")
        self.schedule_digest = _require_sha256(schedule_digest, "schedule digest")
        self.native_schedule_semantic_digest = _require_sha256(
            native_schedule_semantic_digest,
            "native schedule semantic digest",
        )
        self.process_digest = _require_sha256(process_digest, "process digest")
        self.process_semantic_digest = _require_sha256(
            process_semantic_digest, "process semantic digest"
        )
        self.process_support_mask = _single_bit(
            process_support_mask, "process support mask"
        )
        self.process_support_words = tuple(
            (self.process_support_mask >> shift) & ((1 << 64) - 1)
            for shift in range(0, max(1, self.process_support_mask.bit_length()), 64)
        )
        if not isinstance(remap, RecurrenceProcessRemap):
            raise RecurrenceScheduleSharingError(
                "recurrence process binding requires an exact process remap"
            )
        if not remap.bijection_digest:
            raise RecurrenceScheduleSharingError(
                "recurrence process remap has no authenticated bijection digest"
            )
        self.remap = remap
        if not isinstance(executor_pack, RecurrenceProcessExecutorPack):
            raise RecurrenceScheduleSharingError(
                "recurrence process binding requires a typed executor pack"
            )
        if executor_pack.catalog_executor_count != remap.direct_executor_count:
            raise RecurrenceScheduleSharingError(
                "recurrence process executor domain disagrees with its remap"
            )
        self.executor_pack = executor_pack
        self.payload = encode_recurrence_process_binding(
            process_id=self.process_id,
            schedule_digest=self.schedule_digest,
            process_digest=self.process_digest,
            process_semantic_digest=self.process_semantic_digest,
            process_support_mask=self.process_support_mask,
            remap=self.remap,
            executor_pack=self.executor_pack,
        )
        self.sha256 = hashlib.sha256(self.payload).hexdigest()

    @property
    def artifact_path(self) -> str:
        return f"processes/{self.process_id}/recurrence-binding.bin"

    def to_mapping(self) -> dict[str, object]:
        return {
            "abi": RECURRENCE_PROCESS_BINDING_ABI,
            "process_id": self.process_id,
            "schedule_digest": self.schedule_digest,
            "native_schedule_semantic_digest": (self.native_schedule_semantic_digest),
            "process_digest": self.process_digest,
            "process_semantic_digest": self.process_semantic_digest,
            "process_support_words": list(self.process_support_words),
            "remap": self.remap.to_mapping(),
            "path": "recurrence-binding.bin",
            "size_bytes": len(self.payload),
            "sha256": self.sha256,
        }


class RecurrenceScheduleSharingPlan:
    __slots__ = ("bindings", "schedules")

    def __init__(
        self,
        *,
        schedules: tuple[RecurrenceSharedSchedule, ...],
        bindings: tuple[RecurrenceProcessBinding, ...],
    ) -> None:
        self.schedules = schedules
        self.bindings = bindings

    def schedule(self, digest: str) -> RecurrenceSharedSchedule:
        for schedule in self.schedules:
            if schedule.digest == digest:
                return schedule
        raise RecurrenceScheduleSharingError(f"unknown recurrence schedule {digest!r}")

    def binding(self, process_id: str) -> RecurrenceProcessBinding:
        for binding in self.bindings:
            if binding.process_id == process_id:
                return binding
        raise RecurrenceScheduleSharingError(
            f"process {process_id!r} has no recurrence binding"
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "kind": RECURRENCE_SCHEDULE_SHARING_KIND,
            "schema_version": RECURRENCE_SCHEDULE_SHARING_SCHEMA_VERSION,
            "schedule_count": len(self.schedules),
            "binding_count": len(self.bindings),
            "schedule_alias_count": len(self.bindings) - len(self.schedules),
            "runtime_ownership": "root-schedule-plus-process-binding",
            "interning_phase": "before-direct-lowering",
            "schedules": [item.to_mapping() for item in self.schedules],
            "bindings": [item.to_mapping() for item in self.bindings],
        }

    def extension_mapping(self, *, index_sha256: str) -> dict[str, object]:
        return {
            "kind": RECURRENCE_SCHEDULE_SHARING_KIND,
            "schema_version": RECURRENCE_SCHEDULE_SHARING_SCHEMA_VERSION,
            "index_path": RECURRENCE_SCHEDULE_INDEX_PATH,
            "index_sha256": _require_sha256(index_sha256, "schedule index SHA-256"),
            "schedule_count": len(self.schedules),
            "binding_count": len(self.bindings),
            "schedule_alias_count": len(self.bindings) - len(self.schedules),
            "runtime_ownership": "root-schedule-plus-process-binding",
            "interning_phase": "before-direct-lowering",
        }


def intern_recurrence_schedules(
    processes: Sequence[RecurrenceScheduleProcess],
) -> RecurrenceScheduleSharingPlan:
    """Validate and intern already pre-lowered root schedules."""

    schedules: dict[str, RecurrenceSharedSchedule] = {}
    schedule_processes: dict[str, list[str]] = {}
    schedule_native_semantics: dict[str, str] = {}
    bindings: list[RecurrenceProcessBinding] = []
    process_ids: set[str] = set()
    support_masks: set[int] = set()

    for process in processes:
        process_id = _nonempty_text(process.process_id, "recurrence process ID")
        if process_id in process_ids:
            raise RecurrenceScheduleSharingError(
                f"duplicate recurrence process binding {process_id!r}"
            )
        process_ids.add(process_id)
        support_mask = _single_bit(
            process.process_support_mask, f"process {process_id!r} support mask"
        )
        if support_mask in support_masks:
            raise RecurrenceScheduleSharingError(
                f"recurrence support mask {support_mask} is reused"
            )
        support_masks.add(support_mask)

        digest = _require_sha256(
            process.recurrence_schedule_digest,
            f"process {process_id!r} schedule digest",
        )
        native_semantic_digest = _require_sha256(
            process.recurrence_native_schedule_semantic_digest,
            f"process {process_id!r} native schedule semantic digest",
        )
        candidate = RecurrenceSharedSchedule(
            digest=digest,
            source_path=process.recurrence_schedule_path,
            sha256=process.recurrence_schedule_sha256,
            size_bytes=process.recurrence_schedule_size_bytes,
            member_count=process.recurrence_schedule_member_count,
            unpacked_size_bytes=process.recurrence_schedule_unpacked_size_bytes,
            index_sha256=process.recurrence_schedule_index_sha256,
            process_ids=(),
        )
        previous = schedules.get(digest)
        if previous is not None and (
            previous.sha256 != candidate.sha256
            or previous.size_bytes != candidate.size_bytes
            or previous.member_count != candidate.member_count
            or previous.unpacked_size_bytes != candidate.unpacked_size_bytes
            or previous.index_sha256 != candidate.index_sha256
        ):
            raise RecurrenceScheduleSharingError(
                f"schedule digest {digest} maps to different payloads"
            )
        previous_native_semantic_digest = schedule_native_semantics.get(digest)
        if (
            previous_native_semantic_digest is not None
            and previous_native_semantic_digest != native_semantic_digest
        ):
            raise RecurrenceScheduleSharingError(
                f"schedule digest {digest} maps to different native semantics"
            )
        schedules.setdefault(digest, candidate)
        schedule_native_semantics.setdefault(digest, native_semantic_digest)
        schedule_processes.setdefault(digest, []).append(process_id)
        bindings.append(
            RecurrenceProcessBinding(
                process_id=process_id,
                schedule_digest=digest,
                native_schedule_semantic_digest=native_semantic_digest,
                process_digest=process.process_digest,
                process_semantic_digest=process.builder_input_sha256,
                process_support_mask=support_mask,
                remap=process.recurrence_process_remap,
                executor_pack=process.recurrence_process_executor_pack,
            )
        )

    shared = tuple(
        RecurrenceSharedSchedule(
            digest=digest,
            source_path=schedule.source_path,
            sha256=schedule.sha256,
            size_bytes=schedule.size_bytes,
            member_count=schedule.member_count,
            unpacked_size_bytes=schedule.unpacked_size_bytes,
            index_sha256=schedule.index_sha256,
            process_ids=tuple(sorted(schedule_processes[digest])),
        )
        for digest, schedule in sorted(schedules.items())
    )
    return RecurrenceScheduleSharingPlan(
        schedules=shared,
        bindings=tuple(sorted(bindings, key=lambda item: item.process_id)),
    )


def encode_recurrence_process_binding(
    *,
    process_id: str,
    schedule_digest: str,
    process_digest: str,
    process_semantic_digest: str,
    process_support_mask: int,
    remap: RecurrenceProcessRemap,
    executor_pack: RecurrenceProcessExecutorPack,
) -> bytes:
    process = _nonempty_text(process_id, "process ID").encode("utf-8")
    if len(process) > 4096:
        raise RecurrenceScheduleSharingError("process ID exceeds 4096 UTF-8 bytes")
    mask = _single_bit(process_support_mask, "process support mask")
    words = tuple(
        (mask >> shift) & ((1 << 64) - 1)
        for shift in range(0, max(1, mask.bit_length()), 64)
    )
    if not isinstance(remap, RecurrenceProcessRemap) or not remap.bijection_digest:
        raise RecurrenceScheduleSharingError(
            "recurrence process binding requires an authenticated process remap"
        )
    if not isinstance(executor_pack, RecurrenceProcessExecutorPack):
        raise RecurrenceScheduleSharingError(
            "recurrence process binding requires a typed executor pack"
        )
    if executor_pack.catalog_executor_count != remap.direct_executor_count:
        raise RecurrenceScheduleSharingError(
            "recurrence process executor domain disagrees with its remap"
        )
    target = executor_pack.target_triple.encode("utf-8")
    if len(target) > _MISSING_U32:
        raise RecurrenceScheduleSharingError(
            "process executor target triple exceeds the u32 wire domain"
        )
    counts = (
        len(remap.source_slots),
        len(remap.public_flow_ids),
        len(remap.physical_sector_ids),
        remap.state_template_count,
        remap.source_template_count,
        remap.direct_executor_count,
        remap.parameter_slot_count,
        len(remap.state_template_changes),
        len(remap.source_template_changes),
        len(remap.direct_executor_changes),
        len(remap.parameter_slot_changes),
    )
    for value in counts:
        if value > (1 << 32) - 1:
            raise RecurrenceScheduleSharingError(
                "recurrence process remap exceeds the u32 binding ABI"
            )
    for value, context in (
        (len(words), "support-word count"),
        (len(executor_pack.cpu_features), "CPU-feature count"),
        (len(executor_pack.executor_ids), "executor-descriptor count"),
    ):
        if value > _MISSING_U32:
            raise RecurrenceScheduleSharingError(
                f"recurrence process {context} exceeds the u32 binding ABI"
            )
    header = b"".join(
        (
            RECURRENCE_PROCESS_BINDING_MAGIC,
            struct.pack(
                "<8I",
                _PROCESS_BINDING_VERSION,
                _PROCESS_BINDING_FIXED_SIZE,
                len(process),
                len(words),
                len(target),
                len(executor_pack.cpu_features),
                len(executor_pack.executor_ids),
                executor_pack.catalog_executor_count,
            ),
            struct.pack(
                "<4B",
                _DIRECT_BACKEND[executor_pack.backend],
                int(executor_pack.portable),
                64,
                0,
            ),
            struct.pack("<11I", *counts),
            bytes.fromhex(_require_sha256(schedule_digest, "schedule digest")),
            bytes.fromhex(
                _require_sha256(process_semantic_digest, "process semantic digest")
            ),
            bytes.fromhex(
                _require_sha256(
                    executor_pack.compiled_model_digest,
                    "compiled-model digest",
                )
            ),
            bytes.fromhex(executor_pack.recurrence_template_catalog_digest),
            bytes.fromhex(executor_pack.prepared_kernel_pack_digest),
            bytes.fromhex(executor_pack.direct_template_catalog_digest),
            bytes.fromhex(executor_pack.runtime_layout_digest),
            bytes.fromhex(_require_sha256(process_digest, "process digest")),
        )
    )
    if len(header) != _PROCESS_BINDING_FIXED_SIZE:
        raise AssertionError("recurrence process-binding header size drifted")
    sparse_changes = (
        remap.state_template_changes,
        remap.source_template_changes,
        remap.direct_executor_changes,
        remap.parameter_slot_changes,
    )
    return b"".join(
        (
            header,
            process,
            target,
            *(
                _encode_wire_text(feature, "target CPU feature")
                for feature in executor_pack.cpu_features
            ),
            struct.pack(f"<{len(words)}Q", *words),
            struct.pack(f"<{len(remap.source_slots)}I", *remap.source_slots),
            struct.pack(
                f"<{len(remap.source_momentum_signs)}i",
                *remap.source_momentum_signs,
            ),
            struct.pack(
                f"<{len(remap.source_helicity_signs)}i",
                *remap.source_helicity_signs,
            ),
            struct.pack(
                f"<{len(remap.source_state_offsets)}I",
                *remap.source_state_offsets,
            ),
            struct.pack(
                f"<{len(remap.source_state_indices)}I",
                *remap.source_state_indices,
            ),
            struct.pack(
                f"<{len(remap.public_flow_ids)}I",
                *remap.public_flow_ids,
            ),
            struct.pack(
                f"<{len(remap.physical_sector_ids)}I",
                *remap.physical_sector_ids,
            ),
            *(
                struct.pack(
                    f"<{2 * len(changes)}I",
                    *(value for pair in changes for value in pair),
                )
                for changes in sparse_changes
            ),
            *executor_pack.descriptor_payloads,
        )
    )


def _schedule_plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, object] = {}
        for item in fields(value):
            if item.name == "process_id":
                result[item.name] = "<shared-schedule>"
            elif item.name in {"process_support_mask", "support_mask"}:
                result[item.name] = 1
            else:
                result[item.name] = _schedule_plain(getattr(value, item.name))
        return result
    if isinstance(value, Mapping):
        return {
            str(key): _schedule_plain(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_schedule_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_schedule_plain(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _nonempty_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecurrenceScheduleSharingError(f"{context} must be nonempty text")
    return value


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RecurrenceScheduleSharingError(
            f"{context} must be a lowercase hexadecimal SHA-256 digest"
        )
    return value


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrenceScheduleSharingError(f"{context} must be a positive integer")
    return value


def _nonnegative_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecurrenceScheduleSharingError(f"{context} must be a nonnegative integer")
    return value


def _single_bit(value: object, context: str) -> int:
    result = _positive_integer(value, context)
    if result & (result - 1):
        raise RecurrenceScheduleSharingError(f"{context} must contain one bit")
    return result


def _permutation(values: Sequence[int], context: str) -> tuple[int, ...]:
    result = tuple(_nonnegative_integer(value, f"{context} value") for value in values)
    if not _is_permutation(result):
        raise RecurrenceScheduleSharingError(
            f"{context} must be a dense zero-based permutation"
        )
    return result


def _is_permutation(values: Sequence[int]) -> bool:
    return tuple(sorted(values)) == tuple(range(len(values)))


def _inverse_permutation(values: Sequence[int]) -> tuple[int, ...]:
    permutation = _permutation(values, "permutation")
    inverse = [0] * len(permutation)
    for source, target in enumerate(permutation):
        inverse[target] = source
    return tuple(inverse)


def _signs(
    values: Sequence[int],
    expected_count: int,
    context: str,
) -> tuple[int, ...]:
    result = tuple(values)
    if len(result) != expected_count or any(value not in {-1, 1} for value in result):
        raise RecurrenceScheduleSharingError(
            f"{context} must contain {expected_count} signs in {{-1, 1}}"
        )
    return result


def _ragged_permutations(
    offsets: Sequence[int],
    values: Sequence[int],
    row_count: int,
    context: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    checked_offsets = tuple(
        _nonnegative_integer(value, f"{context} offset") for value in offsets
    )
    checked_values = tuple(
        _nonnegative_integer(value, f"{context} value") for value in values
    )
    if (
        len(checked_offsets) != row_count + 1
        or not checked_offsets
        or checked_offsets[0] != 0
        or checked_offsets[-1] != len(checked_values)
        or any(left > right for left, right in itertools.pairwise(checked_offsets))
    ):
        raise RecurrenceScheduleSharingError(
            f"{context} offsets do not partition {row_count} rows"
        )
    for row, (start, stop) in enumerate(itertools.pairwise(checked_offsets)):
        if not _is_permutation(checked_values[start:stop]):
            raise RecurrenceScheduleSharingError(
                f"{context} row {row} is not a dense zero-based permutation"
            )
    return checked_offsets, checked_values


def _sparse_permutation(
    changes: Sequence[tuple[int, int]],
    count: int,
    context: str,
) -> tuple[tuple[int, int], ...]:
    size = _nonnegative_integer(count, f"{context} domain size")
    result = tuple(changes)
    if result != tuple(sorted(result)):
        raise RecurrenceScheduleSharingError(f"{context} changes must be ordered")
    sources: set[int] = set()
    mapping = list(range(size))
    for source, target in result:
        source = _nonnegative_integer(source, f"{context} source")
        target = _nonnegative_integer(target, f"{context} target")
        if source >= size or target >= size:
            raise RecurrenceScheduleSharingError(
                f"{context} change is outside its declared domain"
            )
        if source == target:
            raise RecurrenceScheduleSharingError(
                f"{context} must omit identity changes"
            )
        if source in sources:
            raise RecurrenceScheduleSharingError(f"{context} repeats source {source}")
        sources.add(source)
        mapping[source] = target
    if not _is_permutation(mapping):
        raise RecurrenceScheduleSharingError(
            f"{context} changes do not define a bijection"
        )
    return result


def _sparse_mapping(
    count: int,
    changes: Sequence[tuple[int, int]],
) -> dict[str, object]:
    return {
        "count": _nonnegative_integer(count, "sparse mapping count"),
        "changes": [list(pair) for pair in changes],
    }


def _template_count(
    logical: RecurrenceBuilderLogicalInputV1,
    kind: str,
) -> int:
    identifiers = tuple(
        row.template_id
        for row in logical.semantic_template_references
        if row.kind == kind
    )
    if not identifiers:
        raise RecurrenceScheduleSharingError(
            f"recurrence process has no {kind!r} template domain"
        )
    if identifiers != tuple(range(len(identifiers))):
        raise RecurrenceScheduleSharingError(
            f"recurrence {kind!r} template IDs are not dense and ordered"
        )
    return len(identifiers)


def _physical_sector_domain_count(
    logical: RecurrenceBuilderLogicalInputV1,
) -> int:
    if logical.layout == "topology-replay":
        return len(logical.public_flows)
    return len(logical.physical_sectors)


__all__ = [
    "RECURRENCE_PROCESS_BINDING_ABI",
    "RECURRENCE_PROCESS_BINDING_MAGIC",
    "RECURRENCE_SCHEDULE_INDEX_PATH",
    "RECURRENCE_SCHEDULE_SHARING_KIND",
    "RECURRENCE_SCHEDULE_SHARING_SCHEMA_VERSION",
    "RecurrenceProcessBinding",
    "RecurrenceProcessExecutorPack",
    "RecurrenceProcessRemap",
    "RecurrenceScheduleLoweringCache",
    "RecurrenceScheduleSharingError",
    "RecurrenceScheduleSharingPlan",
    "RecurrenceSharedLoweringResult",
    "RecurrenceSharedSchedule",
    "build_recurrence_process_executor_pack",
    "encode_recurrence_process_binding",
    "exact_recurrence_process_bijection",
    "intern_recurrence_schedules",
    "recurrence_helicity_selector_schedule_digest",
    "recurrence_native_schedule_semantic_digest",
    "recurrence_schedule_semantic_digest",
]
