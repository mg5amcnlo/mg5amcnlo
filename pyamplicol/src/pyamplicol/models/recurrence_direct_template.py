# SPDX-License-Identifier: 0BSD
"""Prepared model metadata for Direct-Arena recurrence executors.

This module deliberately describes executable ownership without adapting the
existing packed eager-kernel ABI. Portable JIT bindings reference a standard
SymJIT direct-arena P-kernel and authenticate Rusticol-owned plane, broadcast,
scratch-output, and destination-policy metadata. Target-native C++/ASM bindings
reference a split-real shared library that exports the same typed arena/row
contract directly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias, cast

from .._internal.versions import (
    RECURRENCE_DIRECT_BINDING_ABI,
    SYMJIT_PLANE_APPLICATION_ABI,
)
from .recurrence_direct_intrinsics import (
    RECURRENCE_INTRINSIC_SCALE_KIND,
    CertifiedRecurrenceIntrinsic,
    certify_recurrence_contribution_intrinsic,
    certify_recurrence_finalization_intrinsic,
)
from .recurrence_template import (
    ExactComplexRationalV1,
    RecurrenceTemplateCatalog,
)

RECURRENCE_DIRECT_TEMPLATE_ABI = "pyamplicol-recurrence-direct-template-v1"
RECURRENCE_DIRECT_BACKEND_ABI = "rusticol.recurrence-direct-backend.v1"
RECURRENCE_DIRECT_CANONICALIZATION_ABI = "pyamplicol-canonical-json-v1"
RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI = RECURRENCE_DIRECT_BINDING_ABI
RECURRENCE_DIRECT_IDENTITY_FINALIZER = "rusticol.identity-finalize-in-place.v1"
SYMJIT_DIRECT_APPLICATION_ABI = SYMJIT_PLANE_APPLICATION_ABI
NATIVE_DIRECT_APPLICATION_ABI = "pyamplicol-recurrence-native-direct-library-v1"

DirectRole: TypeAlias = Literal["source", "contribution", "finalization", "closure"]
DirectDestinationOperation: TypeAlias = Literal[
    "initialize", "add", "finalize-in-place", "closure-add"
]
DirectBackend: TypeAlias = Literal["jit", "cpp", "asm"]
DirectPayloadBindingKind: TypeAlias = Literal[
    "rusticol-intrinsic",
    "prepared-direct-call",
    "pending-direct-call-abi",
]

_ROLES = ("source", "contribution", "finalization", "closure")
_ROLE_INDEX = {role: index for index, role in enumerate(_ROLES)}
_DESTINATION_OPERATIONS = {
    "source": "initialize",
    "contribution": "add",
    "finalization": "finalize-in-place",
    "closure": "closure-add",
}
_CONTRACT_ROLES = {
    "source": "source",
    "vertex": "contribution",
    "propagator": "finalization",
    "closure": "closure",
}
_BACKENDS = frozenset({"jit", "cpp", "asm"})
_PAYLOAD_BINDING_KINDS = frozenset(
    {"rusticol-intrinsic", "prepared-direct-call", "pending-direct-call-abi"}
)
_NATIVE_SOURCE_APPLICATION_ABIS = frozenset(
    {
        "symbolica.compiled-cpp.complex-f64.v1",
        "symbolica.compiled-asm.complex-f64.v1",
    }
)
_HEX = frozenset("0123456789abcdef")


class RecurrenceDirectTemplateError(ValueError):
    """Raised when a prepared Direct-Arena companion is not canonical."""


def _canonical_json(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RecurrenceDirectTemplateError(
            "direct recurrence template payload is not canonical JSON"
        ) from exc


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _require_nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RecurrenceDirectTemplateError(f"{name} must be a nonempty string")
    return value


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise RecurrenceDirectTemplateError(f"{name} must be a lowercase SHA-256")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise RecurrenceDirectTemplateError(f"{name} must be a nonnegative integer")
    return value


def _require_positive_int(name: str, value: object) -> int:
    result = _require_nonnegative_int(name, value)
    if result == 0:
        raise RecurrenceDirectTemplateError(f"{name} must be positive")
    return result


def _require_string_tuple(
    name: str,
    value: object,
    *,
    nonempty: bool = False,
    sorted_unique: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise RecurrenceDirectTemplateError(
            f"{name} must be an immutable nonempty-string tuple"
        )
    result = tuple(value)
    if nonempty and not result:
        raise RecurrenceDirectTemplateError(f"{name} must not be empty")
    if sorted_unique and result != tuple(sorted(set(result))):
        raise RecurrenceDirectTemplateError(f"{name} must be sorted and unique")
    return result


def _require_int_tuple(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple) or any(type(item) is not int for item in value):
        raise RecurrenceDirectTemplateError(
            f"{name} must be an immutable integer tuple"
        )
    if any(item < 0 for item in value):
        raise RecurrenceDirectTemplateError(f"{name} must be nonnegative")
    return value


def _encode_canonical_objects(value: Sequence[object]) -> tuple[str, ...]:
    return tuple(_canonical_json(item) for item in value)


def _decode_canonical_objects(value: Sequence[str]) -> list[object]:
    return [json.loads(item) for item in value]


def _require_canonical_object_tuple(name: str, value: object) -> tuple[str, ...]:
    strings = _require_string_tuple(name, value)
    for item in strings:
        try:
            decoded = json.loads(item)
        except json.JSONDecodeError as exc:
            raise RecurrenceDirectTemplateError(
                f"{name} must contain canonical JSON objects"
            ) from exc
        if not isinstance(decoded, Mapping) or _canonical_json(decoded) != item:
            raise RecurrenceDirectTemplateError(
                f"{name} must contain canonical JSON objects"
            )
    return strings


def _require_empty_prepared_call_metadata(
    binding: RecurrenceDirectPayloadBindingV1,
) -> None:
    fields = {
        "destination_operation": binding.destination_operation,
        "direct_application_abi": binding.direct_application_abi,
        "exact_factor_scalar_slots": binding.exact_factor_scalar_slots,
        "input_plane_count": binding.input_plane_count,
        "input_plane_projections": binding.input_plane_projections,
        "native_entry_point": binding.native_entry_point,
        "output_alias_inputs": binding.output_alias_inputs,
        "parameter_bindings": binding.parameter_bindings,
        "prepared_template_semantic_digest": (
            binding.prepared_template_semantic_digest
        ),
        "role": binding.role,
        "scalar_input_count": binding.scalar_input_count,
        "scalar_projections": binding.scalar_projections,
        "source_application_abi": binding.source_application_abi,
        "source_application_path": binding.source_application_path,
        "source_application_sha256": binding.source_application_sha256,
        "state_plane_indices": binding.state_plane_indices,
    }
    if any(value not in (None, (), 0) for value in fields.values()):
        raise RecurrenceDirectTemplateError(
            "non-executable direct bindings cannot carry prepared-call metadata"
        )


@dataclass(frozen=True, slots=True)
class RecurrenceDirectPayloadBindingV1:
    """Typed ownership of one executor's f64 implementation payload."""

    kind: DirectPayloadBindingKind
    payload_digest: str
    prepared_kernel_id: int | None = None
    runtime_template: str | None = None
    payload_paths: tuple[str, ...] = ()
    source_application_path: str | None = None
    source_application_sha256: str | None = None
    source_application_abi: str | None = None
    direct_application_abi: str | None = None
    native_entry_point: str | None = None
    role: DirectRole | None = None
    destination_operation: DirectDestinationOperation | None = None
    exact_factor_scalar_slots: tuple[int, ...] = ()
    state_plane_indices: tuple[int, ...] = ()
    parameter_bindings: tuple[str, ...] = ()
    input_plane_count: int = 0
    scalar_input_count: int = 0
    output_alias_inputs: tuple[int, ...] = ()
    input_plane_projections: tuple[str, ...] = ()
    scalar_projections: tuple[str, ...] = ()
    intrinsic_contract_digest: str | None = None
    prepared_template_semantic_digest: str | None = None
    contribution_parent_permutation: tuple[int, int] = (0, 1)
    abi: str = RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI

    def __post_init__(self) -> None:
        if self.abi != RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI:
            raise RecurrenceDirectTemplateError(
                f"unsupported direct payload-binding ABI {self.abi!r}; "
                "regenerate the prepared model with this pyAmpliCol version"
            )
        if self.kind not in _PAYLOAD_BINDING_KINDS:
            raise RecurrenceDirectTemplateError(
                f"unsupported direct payload-binding kind {self.kind!r}"
            )
        _require_sha256("direct payload digest", self.payload_digest)
        paths = _require_string_tuple(
            "direct payload paths", self.payload_paths, sorted_unique=True
        )
        parameter_bindings = _require_canonical_object_tuple(
            "direct parameter bindings", self.parameter_bindings
        )
        input_projections = _require_canonical_object_tuple(
            "direct input-plane projections", self.input_plane_projections
        )
        scalar_projections = _require_canonical_object_tuple(
            "direct scalar projections", self.scalar_projections
        )
        exact_factor_slots = _require_int_tuple(
            "direct exact-factor scalar slots", self.exact_factor_scalar_slots
        )
        state_planes = _require_int_tuple(
            "direct state-plane indices", self.state_plane_indices
        )
        output_aliases = _require_int_tuple(
            "direct output-alias inputs", self.output_alias_inputs
        )
        parent_permutation = _require_int_tuple(
            "direct contribution parent permutation",
            self.contribution_parent_permutation,
        )
        if parent_permutation not in {(0, 1), (1, 0)}:
            raise RecurrenceDirectTemplateError(
                "direct contribution parent permutation must be (0, 1) or (1, 0)"
            )
        input_plane_count = _require_nonnegative_int(
            "direct input-plane count", self.input_plane_count
        )
        scalar_input_count = _require_nonnegative_int(
            "direct scalar-input count", self.scalar_input_count
        )
        if self.kind == "rusticol-intrinsic":
            if self.prepared_kernel_id is not None or not self.runtime_template:
                raise RecurrenceDirectTemplateError(
                    "Rusticol direct intrinsics require a runtime template and "
                    "cannot reference a prepared kernel"
                )
            if paths:
                raise RecurrenceDirectTemplateError(
                    "Rusticol direct intrinsics cannot reference bundle payloads"
                )
            if self.role == "contribution":
                if (
                    self.destination_operation != "add"
                    or self.scalar_input_count != 1
                    or len(scalar_projections) != 1
                    or self.intrinsic_contract_digest is None
                ):
                    raise RecurrenceDirectTemplateError(
                        "contribution intrinsics require one certified scale"
                    )
                projection = json.loads(scalar_projections[0])
                if projection.get("kind") != RECURRENCE_INTRINSIC_SCALE_KIND:
                    raise RecurrenceDirectTemplateError(
                        "contribution intrinsic scale has an unsupported kind"
                    )
                _require_sha256(
                    "intrinsic contract digest", self.intrinsic_contract_digest
                )
                expected_payload_digest = _digest(
                    {
                        "abi": self.abi,
                        "destination_operation": self.destination_operation,
                        "contribution_parent_permutation": list(parent_permutation),
                        "intrinsic_contract_digest": (self.intrinsic_contract_digest),
                        "kind": self.kind,
                        "role": self.role,
                        "runtime_template": self.runtime_template,
                        "scalar_input_count": self.scalar_input_count,
                        "scalar_projections": [json.loads(scalar_projections[0])],
                    }
                )
                if self.payload_digest != expected_payload_digest:
                    raise RecurrenceDirectTemplateError(
                        "contribution intrinsic payload digest does not match "
                        "its certified metadata"
                    )
                if any(
                    value not in (None, (), 0)
                    for value in (
                        self.direct_application_abi,
                        exact_factor_slots,
                        input_plane_count,
                        input_projections,
                        output_aliases,
                        parameter_bindings,
                        self.prepared_template_semantic_digest,
                        self.source_application_abi,
                        self.source_application_path,
                        self.source_application_sha256,
                        state_planes,
                    )
                ):
                    raise RecurrenceDirectTemplateError(
                        "contribution intrinsics carry prepared-call metadata"
                    )
            else:
                if parent_permutation != (0, 1):
                    raise RecurrenceDirectTemplateError(
                        "non-contribution intrinsics require the identity "
                        "parent permutation"
                    )
                if self.intrinsic_contract_digest is not None:
                    raise RecurrenceDirectTemplateError(
                        "non-contribution intrinsics cannot carry a contract digest"
                    )
                _require_empty_prepared_call_metadata(self)
        else:
            if parent_permutation != (0, 1):
                raise RecurrenceDirectTemplateError(
                    "prepared direct payloads require the identity parent permutation"
                )
            _require_nonnegative_int(
                "direct prepared kernel id", self.prepared_kernel_id
            )
            if self.intrinsic_contract_digest is not None:
                raise RecurrenceDirectTemplateError(
                    "prepared direct payloads cannot carry an intrinsic contract"
                )
            if self.runtime_template is not None:
                raise RecurrenceDirectTemplateError(
                    "prepared direct payloads cannot name a Rusticol template"
                )
            if self.kind == "pending-direct-call-abi" and paths:
                raise RecurrenceDirectTemplateError(
                    "pending direct-call bindings cannot claim executable payloads"
                )
            if self.kind == "prepared-direct-call" and not paths:
                raise RecurrenceDirectTemplateError(
                    "prepared direct-call bindings require executable payload paths"
                )
            if self.kind == "pending-direct-call-abi":
                _require_empty_prepared_call_metadata(self)
            else:
                source_path = _require_nonempty(
                    "direct source application path", self.source_application_path
                )
                if paths != (source_path,):
                    raise RecurrenceDirectTemplateError(
                        "prepared direct-call payload paths must contain exactly "
                        "the source application"
                    )
                _require_sha256(
                    "direct source application digest",
                    self.source_application_sha256,
                )
                _require_nonempty(
                    "direct source application ABI", self.source_application_abi
                )
                if self.direct_application_abi not in {
                    SYMJIT_DIRECT_APPLICATION_ABI,
                    NATIVE_DIRECT_APPLICATION_ABI,
                }:
                    raise RecurrenceDirectTemplateError(
                        "prepared direct-call binding has an unsupported direct "
                        "application ABI"
                    )
                if self.direct_application_abi == SYMJIT_DIRECT_APPLICATION_ABI:
                    if self.native_entry_point is not None:
                        raise RecurrenceDirectTemplateError(
                            "prepared JIT direct-call bindings cannot name a native "
                            "entry point"
                        )
                else:
                    _require_nonempty(
                        "native direct-call entry point", self.native_entry_point
                    )
                if self.role not in _ROLE_INDEX or self.role == "source":
                    raise RecurrenceDirectTemplateError(
                        "prepared direct-call role must be a non-source executor role"
                    )
                if self.destination_operation != _DESTINATION_OPERATIONS[self.role]:
                    raise RecurrenceDirectTemplateError(
                        "prepared direct-call destination operation does not match "
                        "its role"
                    )
                if exact_factor_slots != (0, 1):
                    raise RecurrenceDirectTemplateError(
                        "prepared direct-call bindings must reserve exact-factor "
                        "scalar slots 0 and 1"
                    )
                if len(input_projections) != input_plane_count:
                    raise RecurrenceDirectTemplateError(
                        "direct input-plane projection count does not match "
                        "input_plane_count"
                    )
                if len(scalar_projections) != scalar_input_count:
                    raise RecurrenceDirectTemplateError(
                        "direct scalar projection count does not match "
                        "scalar_input_count"
                    )
                for name, values, upper_bound in (
                    ("state-plane", state_planes, input_plane_count),
                    ("output-alias", output_aliases, input_plane_count),
                ):
                    if any(value >= upper_bound for value in values):
                        raise RecurrenceDirectTemplateError(
                            f"direct {name} index is out of bounds"
                        )
                _require_sha256(
                    "prepared direct template semantic digest",
                    self.prepared_template_semantic_digest,
                )
                expected_payload_digest = _digest(
                    self._prepared_call_fields(include_payload_digest=False)
                )
                if self.payload_digest != expected_payload_digest:
                    raise RecurrenceDirectTemplateError(
                        "prepared direct-call payload digest does not match metadata"
                    )
                if len(parameter_bindings) == 0:
                    raise RecurrenceDirectTemplateError(
                        "prepared direct-call binding must map source parameters"
                    )

    @property
    def executable(self) -> bool:
        return self.kind != "pending-direct-call-abi"

    def to_dict(self) -> dict[str, object]:
        return self._prepared_call_fields(include_payload_digest=True)

    def _prepared_call_fields(
        self, *, include_payload_digest: bool
    ) -> dict[str, object]:
        payload = {
            "abi": self.abi,
            "contribution_parent_permutation": list(
                self.contribution_parent_permutation
            ),
            "destination_operation": self.destination_operation,
            "direct_application_abi": self.direct_application_abi,
            "exact_factor_scalar_slots": list(self.exact_factor_scalar_slots),
            "input_plane_count": self.input_plane_count,
            "input_plane_projections": _decode_canonical_objects(
                self.input_plane_projections
            ),
            "intrinsic_contract_digest": self.intrinsic_contract_digest,
            "kind": self.kind,
            "output_alias_inputs": list(self.output_alias_inputs),
            "parameter_bindings": _decode_canonical_objects(self.parameter_bindings),
            "payload_paths": list(self.payload_paths),
            "prepared_kernel_id": self.prepared_kernel_id,
            "prepared_template_semantic_digest": (
                self.prepared_template_semantic_digest
            ),
            "role": self.role,
            "runtime_template": self.runtime_template,
            "scalar_input_count": self.scalar_input_count,
            "scalar_projections": _decode_canonical_objects(self.scalar_projections),
            "source_application_abi": self.source_application_abi,
            "source_application_path": self.source_application_path,
            "source_application_sha256": self.source_application_sha256,
            "state_plane_indices": list(self.state_plane_indices),
        }
        if self.native_entry_point is not None:
            payload["native_entry_point"] = self.native_entry_point
        if include_payload_digest:
            payload["payload_digest"] = self.payload_digest
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> RecurrenceDirectPayloadBindingV1:
        if not isinstance(payload, Mapping):
            raise RecurrenceDirectTemplateError(
                "direct payload binding must be a JSON object"
            )
        expected = {
            "abi",
            "contribution_parent_permutation",
            "destination_operation",
            "direct_application_abi",
            "exact_factor_scalar_slots",
            "input_plane_count",
            "input_plane_projections",
            "intrinsic_contract_digest",
            "kind",
            "output_alias_inputs",
            "parameter_bindings",
            "payload_digest",
            "payload_paths",
            "prepared_kernel_id",
            "prepared_template_semantic_digest",
            "role",
            "runtime_template",
            "scalar_input_count",
            "scalar_projections",
            "source_application_abi",
            "source_application_path",
            "source_application_sha256",
            "state_plane_indices",
        }
        optional = {"native_entry_point"}
        if not expected.issubset(payload) or not set(payload).issubset(
            expected | optional
        ):
            raise RecurrenceDirectTemplateError(
                "direct payload-binding fields do not match v1"
            )
        array_fields = (
            "contribution_parent_permutation",
            "exact_factor_scalar_slots",
            "input_plane_projections",
            "output_alias_inputs",
            "parameter_bindings",
            "payload_paths",
            "scalar_projections",
            "state_plane_indices",
        )
        if any(not isinstance(payload[field], list) for field in array_fields):
            raise RecurrenceDirectTemplateError(
                "direct payload-binding arrays must be JSON arrays"
            )
        return cls(
            abi=payload["abi"],  # type: ignore[arg-type]
            kind=payload["kind"],  # type: ignore[arg-type]
            native_entry_point=payload.get("native_entry_point"),  # type: ignore[arg-type]
            payload_digest=payload["payload_digest"],  # type: ignore[arg-type]
            contribution_parent_permutation=tuple(
                payload["contribution_parent_permutation"]  # type: ignore[arg-type]
            ),
            payload_paths=tuple(payload["payload_paths"]),  # type: ignore[arg-type]
            prepared_kernel_id=payload["prepared_kernel_id"],  # type: ignore[arg-type]
            runtime_template=payload["runtime_template"],  # type: ignore[arg-type]
            source_application_path=payload["source_application_path"],  # type: ignore[arg-type]
            source_application_sha256=payload["source_application_sha256"],  # type: ignore[arg-type]
            source_application_abi=payload["source_application_abi"],  # type: ignore[arg-type]
            direct_application_abi=payload["direct_application_abi"],  # type: ignore[arg-type]
            role=payload["role"],  # type: ignore[arg-type]
            destination_operation=payload["destination_operation"],  # type: ignore[arg-type]
            exact_factor_scalar_slots=tuple(
                payload["exact_factor_scalar_slots"]  # type: ignore[arg-type]
            ),
            state_plane_indices=tuple(
                payload["state_plane_indices"]  # type: ignore[arg-type]
            ),
            parameter_bindings=_encode_canonical_objects(
                payload["parameter_bindings"]  # type: ignore[arg-type]
            ),
            input_plane_count=payload["input_plane_count"],  # type: ignore[arg-type]
            scalar_input_count=payload["scalar_input_count"],  # type: ignore[arg-type]
            output_alias_inputs=tuple(
                payload["output_alias_inputs"]  # type: ignore[arg-type]
            ),
            input_plane_projections=_encode_canonical_objects(
                payload["input_plane_projections"]  # type: ignore[arg-type]
            ),
            intrinsic_contract_digest=payload["intrinsic_contract_digest"],  # type: ignore[arg-type]
            scalar_projections=_encode_canonical_objects(
                payload["scalar_projections"]  # type: ignore[arg-type]
            ),
            prepared_template_semantic_digest=payload[
                "prepared_template_semantic_digest"
            ],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class PreparedJitDirectSourceV1:
    """Authenticated source application and fixed prepared-kernel contract."""

    prepared_kernel_id: int
    source_application_path: str
    source_application_sha256: str
    source_application_abi: str
    input_contracts: tuple[str, ...]
    exact_expressions: tuple[str, ...]
    output_arity: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(
            "prepared JIT direct source kernel ID", self.prepared_kernel_id
        )
        _require_nonempty(
            "prepared JIT direct source path", self.source_application_path
        )
        _require_sha256(
            "prepared JIT direct source digest", self.source_application_sha256
        )
        _require_nonempty("prepared JIT direct source ABI", self.source_application_abi)
        _require_canonical_object_tuple(
            "prepared JIT direct input contracts", self.input_contracts
        )
        output_arity = _require_positive_int(
            "prepared JIT direct output arity", self.output_arity
        )
        expressions = _require_string_tuple(
            "prepared JIT direct exact expressions",
            self.exact_expressions,
            nonempty=True,
        )
        if len(expressions) != output_arity:
            raise RecurrenceDirectTemplateError(
                "prepared JIT direct exact-expression count does not match output arity"
            )


@dataclass(frozen=True, slots=True)
class PreparedNativeDirectCallableSpecV1:
    """Compile-time contract for one target-native Direct-Arena export."""

    prepared_kernel_id: int
    role: DirectRole
    native_entry_point: str
    input_contracts: tuple[str, ...]
    exact_expressions: tuple[str, ...]
    output_arity: int
    parent_component_shapes: tuple[tuple[int, ...], ...]
    destination_component_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        kernel_id = _require_nonnegative_int(
            "prepared native direct kernel ID", self.prepared_kernel_id
        )
        if self.role not in _ROLE_INDEX or self.role == "source":
            raise RecurrenceDirectTemplateError(
                "prepared native direct callables require a non-source role"
            )
        expected_entry_point = native_direct_entry_point(self.role, kernel_id)
        if self.native_entry_point != expected_entry_point:
            raise RecurrenceDirectTemplateError(
                "prepared native direct entry point does not authenticate its "
                "role and kernel ID"
            )
        _require_canonical_object_tuple(
            "prepared native direct input contracts", self.input_contracts
        )
        output_arity = _require_positive_int(
            "prepared native direct output arity", self.output_arity
        )
        expressions = _require_string_tuple(
            "prepared native direct exact expressions",
            self.exact_expressions,
            nonempty=True,
        )
        if len(expressions) != output_arity:
            raise RecurrenceDirectTemplateError(
                "prepared native direct exact-expression count does not match "
                "output arity"
            )
        if (
            not isinstance(self.parent_component_shapes, tuple)
            or not self.parent_component_shapes
        ):
            raise RecurrenceDirectTemplateError(
                "prepared native direct parent shapes must be nonempty"
            )
        shapes = tuple(
            _require_int_tuple(
                "prepared native direct parent component shape", shape
            )
            for shape in self.parent_component_shapes
        )
        if (
            shapes != tuple(sorted(set(shapes)))
            or len({len(shape) for shape in shapes}) != 1
        ):
            raise RecurrenceDirectTemplateError(
                "prepared native direct parent shapes must be sorted, unique, "
                "and have one arity"
            )
        if any(count == 0 for shape in shapes for count in shape):
            raise RecurrenceDirectTemplateError(
                "prepared native direct parent component counts must be positive"
            )
        destination_counts = _require_int_tuple(
            "prepared native direct destination component counts",
            self.destination_component_counts,
        )
        if (
            not destination_counts
            or destination_counts != tuple(sorted(set(destination_counts)))
            or any(count == 0 for count in destination_counts)
        ):
            raise RecurrenceDirectTemplateError(
                "prepared native direct destination counts must be sorted, "
                "unique, and positive"
            )
        if self.output_arity > min(destination_counts):
            raise RecurrenceDirectTemplateError(
                "prepared native direct output exceeds a destination shape"
            )


@dataclass(frozen=True, slots=True)
class PreparedNativeDirectSourceV1:
    """Authenticated native library and one typed Direct-Arena export."""

    prepared_kernel_id: int
    role: DirectRole
    native_entry_point: str
    source_application_path: str
    source_application_sha256: str
    source_application_abi: str
    input_contracts: tuple[str, ...]
    exact_expressions: tuple[str, ...]
    output_arity: int

    def __post_init__(self) -> None:
        kernel_id = _require_nonnegative_int(
            "prepared native direct source kernel ID", self.prepared_kernel_id
        )
        if self.role not in _ROLE_INDEX or self.role == "source":
            raise RecurrenceDirectTemplateError(
                "prepared native direct sources require a non-source role"
            )
        if self.native_entry_point != native_direct_entry_point(self.role, kernel_id):
            raise RecurrenceDirectTemplateError(
                "prepared native direct source entry point is not canonical"
            )
        _require_canonical_object_tuple(
            "prepared native direct source input contracts", self.input_contracts
        )
        output_arity = _require_positive_int(
            "prepared native direct source output arity", self.output_arity
        )
        if len(
            _require_string_tuple(
                "prepared native direct source exact expressions",
                self.exact_expressions,
                nonempty=True,
            )
        ) != output_arity:
            raise RecurrenceDirectTemplateError(
                "prepared native direct source expression count does not match "
                "its output arity"
            )
        _require_nonempty(
            "prepared native direct source path", self.source_application_path
        )
        _require_sha256(
            "prepared native direct source digest", self.source_application_sha256
        )
        if self.source_application_abi not in _NATIVE_SOURCE_APPLICATION_ABIS:
            raise RecurrenceDirectTemplateError(
                "prepared native direct source has an unsupported application ABI"
            )


def native_direct_entry_point(role: DirectRole, prepared_kernel_id: int) -> str:
    """Return the role- and kernel-authenticated native C export."""

    kernel_id = _require_nonnegative_int(
        "prepared native direct kernel ID", prepared_kernel_id
    )
    if kernel_id == 0xFFFFFFFF:
        raise RecurrenceDirectTemplateError(
            "prepared native direct kernel ID cannot use the missing-ID sentinel"
        )
    if role not in _ROLE_INDEX or role == "source":
        raise RecurrenceDirectTemplateError(
            "prepared native Direct-Arena exports require a non-source role"
        )
    return f"pyamplicol_recurrence_direct_{role}_k{kernel_id:08x}_v1"


@dataclass(frozen=True, slots=True)
class RecurrenceDirectTemplateV1:
    """One model-prepared callable that operates directly on recurrence arenas."""

    template_id: str
    direct_executor_id: int
    evaluator_binding_id: int
    evaluator_resolver_key: str
    role: DirectRole
    parent_arity: int
    parent_component_counts: tuple[int, ...]
    destination_component_count: int
    momentum_operand_count: int
    destination_operation: DirectDestinationOperation
    coupling_slot_count: int
    parameter_slot_count: int
    semantic_template_ids: tuple[str, ...]
    exact_expression_digest: str
    payload_binding: RecurrenceDirectPayloadBindingV1
    backend: DirectBackend
    target_triple: str
    portable: bool
    optimization_level: int
    alignment_bytes: int
    simd_axis: str
    destination_aliasing: bool
    semantic_digest: str = ""
    abi: str = RECURRENCE_DIRECT_TEMPLATE_ABI

    def __post_init__(self) -> None:
        if self.abi != RECURRENCE_DIRECT_TEMPLATE_ABI:
            raise RecurrenceDirectTemplateError(
                f"unsupported direct template ABI {self.abi!r}"
            )
        _require_nonempty("direct template_id", self.template_id)
        _require_nonnegative_int("direct executor id", self.direct_executor_id)
        _require_nonnegative_int(
            "direct evaluator binding id", self.evaluator_binding_id
        )
        _require_nonempty("direct evaluator resolver key", self.evaluator_resolver_key)
        if self.role not in _ROLE_INDEX:
            raise RecurrenceDirectTemplateError(
                f"unsupported direct template role {self.role!r}"
            )
        _require_nonnegative_int("direct parent arity", self.parent_arity)
        counts = _require_int_tuple(
            "direct parent component counts", self.parent_component_counts
        )
        if len(counts) != self.parent_arity or any(count == 0 for count in counts):
            raise RecurrenceDirectTemplateError(
                "direct parent component counts must cover every nonempty parent"
            )
        _require_positive_int(
            "direct destination component count", self.destination_component_count
        )
        _require_nonnegative_int(
            "direct momentum operand count", self.momentum_operand_count
        )
        expected_operation = _DESTINATION_OPERATIONS[self.role]
        if self.destination_operation != expected_operation:
            raise RecurrenceDirectTemplateError(
                f"direct {self.role} template must use {expected_operation!r}"
            )
        _require_nonnegative_int("direct coupling slot count", self.coupling_slot_count)
        _require_nonnegative_int(
            "direct parameter slot count", self.parameter_slot_count
        )
        _require_string_tuple(
            "direct semantic template ids",
            self.semantic_template_ids,
            nonempty=True,
            sorted_unique=True,
        )
        _require_sha256("direct exact expression digest", self.exact_expression_digest)
        if not isinstance(self.payload_binding, RecurrenceDirectPayloadBindingV1):
            raise RecurrenceDirectTemplateError(
                "direct template requires a typed payload binding"
            )
        if self.backend not in _BACKENDS:
            raise RecurrenceDirectTemplateError(
                f"unsupported direct backend {self.backend!r}"
            )
        _require_nonempty("direct target triple", self.target_triple)
        if type(self.portable) is not bool:
            raise RecurrenceDirectTemplateError("direct portable flag must be boolean")
        _require_nonnegative_int("direct optimization level", self.optimization_level)
        if self.backend == "jit":
            if self.optimization_level != 2 or not self.portable:
                raise RecurrenceDirectTemplateError(
                    "prepared direct JIT templates must use portable SymJIT O2"
                )
        elif self.portable:
            raise RecurrenceDirectTemplateError(
                "prepared direct C++/ASM templates must be target-native"
            )
        alignment = _require_positive_int(
            "direct alignment bytes", self.alignment_bytes
        )
        if alignment & (alignment - 1):
            raise RecurrenceDirectTemplateError(
                "direct alignment bytes must be a power of two"
            )
        if self.simd_axis != "points-contiguous":
            raise RecurrenceDirectTemplateError(
                "direct template SIMD axis must be points-contiguous"
            )
        if type(self.destination_aliasing) is not bool:
            raise RecurrenceDirectTemplateError(
                "direct destination_aliasing must be boolean"
            )
        if self.destination_aliasing != (self.role == "finalization"):
            raise RecurrenceDirectTemplateError(
                "only direct finalization templates may alias their destination"
            )
        calculated = _digest(self._semantic_fields())
        if self.semantic_digest:
            _require_sha256("direct semantic digest", self.semantic_digest)
            if self.semantic_digest != calculated:
                raise RecurrenceDirectTemplateError(
                    "direct semantic digest does not match template contents"
                )
        else:
            object.__setattr__(self, "semantic_digest", calculated)

    @property
    def f64_payload_digest(self) -> str:
        return self.payload_binding.payload_digest

    @property
    def executable(self) -> bool:
        return self.payload_binding.executable

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "abi": self.abi,
            "alignment_bytes": self.alignment_bytes,
            "backend": self.backend,
            "coupling_slot_count": self.coupling_slot_count,
            "destination_aliasing": self.destination_aliasing,
            "destination_component_count": self.destination_component_count,
            "destination_operation": self.destination_operation,
            "direct_executor_id": self.direct_executor_id,
            "evaluator_binding_id": self.evaluator_binding_id,
            "evaluator_resolver_key": self.evaluator_resolver_key,
            "exact_expression_digest": self.exact_expression_digest,
            "momentum_operand_count": self.momentum_operand_count,
            "optimization_level": self.optimization_level,
            "parameter_slot_count": self.parameter_slot_count,
            "parent_arity": self.parent_arity,
            "parent_component_counts": list(self.parent_component_counts),
            "payload_binding": self.payload_binding.to_dict(),
            "portable": self.portable,
            "role": self.role,
            "semantic_template_ids": list(self.semantic_template_ids),
            "simd_axis": self.simd_axis,
            "target_triple": self.target_triple,
            "template_id": self.template_id,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._semantic_fields(), "semantic_digest": self.semantic_digest}

    @classmethod
    def from_dict(cls, payload: object) -> RecurrenceDirectTemplateV1:
        if not isinstance(payload, Mapping):
            raise RecurrenceDirectTemplateError("direct template must be a JSON object")
        expected = {
            "abi",
            "alignment_bytes",
            "backend",
            "coupling_slot_count",
            "destination_aliasing",
            "destination_component_count",
            "destination_operation",
            "direct_executor_id",
            "evaluator_binding_id",
            "evaluator_resolver_key",
            "exact_expression_digest",
            "momentum_operand_count",
            "optimization_level",
            "parameter_slot_count",
            "parent_arity",
            "parent_component_counts",
            "payload_binding",
            "portable",
            "role",
            "semantic_digest",
            "semantic_template_ids",
            "simd_axis",
            "target_triple",
            "template_id",
        }
        if set(payload) != expected:
            raise RecurrenceDirectTemplateError(
                "direct template fields do not match direct-template-v1"
            )
        parent_counts = payload["parent_component_counts"]
        semantic_ids = payload["semantic_template_ids"]
        if not isinstance(parent_counts, list) or not isinstance(semantic_ids, list):
            raise RecurrenceDirectTemplateError(
                "direct component counts and semantic IDs must be JSON arrays"
            )
        return cls(
            abi=payload["abi"],  # type: ignore[arg-type]
            template_id=payload["template_id"],  # type: ignore[arg-type]
            direct_executor_id=payload["direct_executor_id"],  # type: ignore[arg-type]
            evaluator_binding_id=payload["evaluator_binding_id"],  # type: ignore[arg-type]
            evaluator_resolver_key=payload["evaluator_resolver_key"],  # type: ignore[arg-type]
            role=payload["role"],  # type: ignore[arg-type]
            parent_arity=payload["parent_arity"],  # type: ignore[arg-type]
            parent_component_counts=tuple(parent_counts),  # type: ignore[arg-type]
            destination_component_count=payload["destination_component_count"],  # type: ignore[arg-type]
            momentum_operand_count=payload["momentum_operand_count"],  # type: ignore[arg-type]
            destination_operation=payload["destination_operation"],  # type: ignore[arg-type]
            coupling_slot_count=payload["coupling_slot_count"],  # type: ignore[arg-type]
            parameter_slot_count=payload["parameter_slot_count"],  # type: ignore[arg-type]
            semantic_template_ids=tuple(semantic_ids),  # type: ignore[arg-type]
            exact_expression_digest=payload["exact_expression_digest"],  # type: ignore[arg-type]
            payload_binding=RecurrenceDirectPayloadBindingV1.from_dict(
                payload["payload_binding"]
            ),
            backend=payload["backend"],  # type: ignore[arg-type]
            target_triple=payload["target_triple"],  # type: ignore[arg-type]
            portable=payload["portable"],  # type: ignore[arg-type]
            optimization_level=payload["optimization_level"],  # type: ignore[arg-type]
            alignment_bytes=payload["alignment_bytes"],  # type: ignore[arg-type]
            simd_axis=payload["simd_axis"],  # type: ignore[arg-type]
            destination_aliasing=payload["destination_aliasing"],  # type: ignore[arg-type]
            semantic_digest=payload["semantic_digest"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RecurrenceDirectTemplateCatalogV1:
    """Deterministic model-wide direct-executor metadata."""

    templates: tuple[RecurrenceDirectTemplateV1, ...]
    backend: DirectBackend
    target_triple: str
    portable: bool
    optimization_level: int
    compiled_model_digest: str
    recurrence_template_catalog_digest: str
    prepared_kernel_pack_digest: str
    prepared_kernel_contract_digest: str
    prepared_kernel_payload_digest: str
    optimization_settings_digest: str
    catalog_digest: str = ""
    abi: str = RECURRENCE_DIRECT_TEMPLATE_ABI
    backend_abi: str = RECURRENCE_DIRECT_BACKEND_ABI
    canonicalization_abi: str = RECURRENCE_DIRECT_CANONICALIZATION_ABI

    def __post_init__(self) -> None:
        if self.abi != RECURRENCE_DIRECT_TEMPLATE_ABI:
            raise RecurrenceDirectTemplateError(
                f"unsupported direct catalog ABI {self.abi!r}"
            )
        if self.backend_abi != RECURRENCE_DIRECT_BACKEND_ABI:
            raise RecurrenceDirectTemplateError(
                f"unsupported direct backend ABI {self.backend_abi!r}"
            )
        if self.canonicalization_abi != RECURRENCE_DIRECT_CANONICALIZATION_ABI:
            raise RecurrenceDirectTemplateError(
                "unsupported direct canonicalization ABI"
            )
        if not self.templates:
            raise RecurrenceDirectTemplateError(
                "direct template catalog must not be empty"
            )
        if self.backend not in _BACKENDS:
            raise RecurrenceDirectTemplateError(
                f"unsupported direct catalog backend {self.backend!r}"
            )
        _require_nonempty("direct catalog target triple", self.target_triple)
        if type(self.portable) is not bool:
            raise RecurrenceDirectTemplateError(
                "direct catalog portable flag must be boolean"
            )
        _require_nonnegative_int(
            "direct catalog optimization level", self.optimization_level
        )
        if self.backend == "jit":
            if not self.portable or self.optimization_level != 2:
                raise RecurrenceDirectTemplateError(
                    "prepared direct JIT catalogs must use portable SymJIT O2"
                )
        elif self.portable:
            raise RecurrenceDirectTemplateError(
                "prepared direct C++/ASM catalogs must be target-native"
            )
        for name in (
            "compiled_model_digest",
            "recurrence_template_catalog_digest",
            "prepared_kernel_pack_digest",
            "prepared_kernel_contract_digest",
            "prepared_kernel_payload_digest",
            "optimization_settings_digest",
        ):
            _require_sha256(f"direct catalog {name}", getattr(self, name))
        expected_order = tuple(
            sorted(
                self.templates,
                key=lambda template: (
                    template.direct_executor_id,
                    _ROLE_INDEX[template.role],
                    template.evaluator_binding_id,
                    template.template_id,
                ),
            )
        )
        if expected_order != self.templates:
            raise RecurrenceDirectTemplateError(
                "direct templates must be sorted by dense executor identity"
            )
        ids = [template.direct_executor_id for template in self.templates]
        names = [template.template_id for template in self.templates]
        binding_keys = [
            (template.role, template.evaluator_binding_id)
            for template in self.templates
        ]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise RecurrenceDirectTemplateError(
                "direct template executor IDs and names must be unique"
            )
        if len(set(binding_keys)) != len(binding_keys):
            raise RecurrenceDirectTemplateError(
                "direct (role, evaluator_binding_id) mappings must be unique"
            )
        if ids != list(range(len(ids))):
            raise RecurrenceDirectTemplateError(
                "direct template executor IDs must form a dense zero-based catalog"
            )
        for template in self.templates:
            if (
                template.backend != self.backend
                or template.target_triple != self.target_triple
                or template.portable != self.portable
                or template.optimization_level != self.optimization_level
            ):
                raise RecurrenceDirectTemplateError(
                    "direct template backend/target policy does not match its catalog"
                )
        calculated = _digest(self._semantic_fields())
        if self.catalog_digest:
            _require_sha256("direct catalog digest", self.catalog_digest)
            if self.catalog_digest != calculated:
                raise RecurrenceDirectTemplateError(
                    "direct catalog digest does not match its templates"
                )
        else:
            object.__setattr__(self, "catalog_digest", calculated)

    def direct_executor_id_for(
        self,
        role: DirectRole,
        evaluator_binding_id: int,
    ) -> int:
        """Resolve an authenticated semantic evaluator binding deterministically."""

        for template in self.templates:
            if (
                template.role == role
                and template.evaluator_binding_id == evaluator_binding_id
            ):
                return template.direct_executor_id
        raise RecurrenceDirectTemplateError(
            "direct executor catalog has no mapping for "
            f"({role!r}, evaluator_binding_id={evaluator_binding_id})"
        )

    @property
    def executable(self) -> bool:
        return all(template.executable for template in self.templates)

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "abi": self.abi,
            "backend": self.backend,
            "backend_abi": self.backend_abi,
            "canonicalization_abi": self.canonicalization_abi,
            "compiled_model_digest": self.compiled_model_digest,
            "optimization_level": self.optimization_level,
            "optimization_settings_digest": self.optimization_settings_digest,
            "portable": self.portable,
            "prepared_kernel_contract_digest": self.prepared_kernel_contract_digest,
            "prepared_kernel_pack_digest": self.prepared_kernel_pack_digest,
            "prepared_kernel_payload_digest": self.prepared_kernel_payload_digest,
            "recurrence_template_catalog_digest": (
                self.recurrence_template_catalog_digest
            ),
            "target_triple": self.target_triple,
            "templates": [template.to_dict() for template in self.templates],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._semantic_fields(), "catalog_digest": self.catalog_digest}

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: object) -> RecurrenceDirectTemplateCatalogV1:
        if not isinstance(payload, Mapping):
            raise RecurrenceDirectTemplateError(
                "direct template catalog must be a JSON object"
            )
        expected = {
            "abi",
            "backend",
            "backend_abi",
            "canonicalization_abi",
            "catalog_digest",
            "compiled_model_digest",
            "optimization_level",
            "optimization_settings_digest",
            "portable",
            "prepared_kernel_contract_digest",
            "prepared_kernel_pack_digest",
            "prepared_kernel_payload_digest",
            "recurrence_template_catalog_digest",
            "target_triple",
            "templates",
        }
        if set(payload) != expected:
            raise RecurrenceDirectTemplateError(
                "direct template catalog fields do not match v1"
            )
        raw_templates = payload["templates"]
        if not isinstance(raw_templates, list):
            raise RecurrenceDirectTemplateError(
                "direct template catalog templates must be a JSON array"
            )
        return cls(
            abi=payload["abi"],  # type: ignore[arg-type]
            backend_abi=payload["backend_abi"],  # type: ignore[arg-type]
            canonicalization_abi=payload["canonicalization_abi"],  # type: ignore[arg-type]
            templates=tuple(
                RecurrenceDirectTemplateV1.from_dict(item) for item in raw_templates
            ),
            backend=payload["backend"],  # type: ignore[arg-type]
            target_triple=payload["target_triple"],  # type: ignore[arg-type]
            portable=payload["portable"],  # type: ignore[arg-type]
            optimization_level=payload["optimization_level"],  # type: ignore[arg-type]
            compiled_model_digest=payload["compiled_model_digest"],  # type: ignore[arg-type]
            recurrence_template_catalog_digest=payload[
                "recurrence_template_catalog_digest"
            ],  # type: ignore[arg-type]
            prepared_kernel_pack_digest=payload["prepared_kernel_pack_digest"],  # type: ignore[arg-type]
            prepared_kernel_contract_digest=payload["prepared_kernel_contract_digest"],  # type: ignore[arg-type]
            prepared_kernel_payload_digest=payload["prepared_kernel_payload_digest"],  # type: ignore[arg-type]
            optimization_settings_digest=payload["optimization_settings_digest"],  # type: ignore[arg-type]
            catalog_digest=payload["catalog_digest"],  # type: ignore[arg-type]
        )


def build_prepared_native_direct_callable_specs(
    recurrence_catalog: RecurrenceTemplateCatalog,
    prepared_kernels: Mapping[int, object],
) -> dict[int, PreparedNativeDirectCallableSpecV1]:
    """Project model-generic semantic bindings into one native export per kernel.

    Inline couplings remain evaluator-binding-specific immutable contexts.
    They are validated here but deliberately excluded from callable identity,
    allowing one prepared Lorentz kernel to serve every compatible transition.
    """

    if not isinstance(recurrence_catalog, RecurrenceTemplateCatalog):
        raise TypeError(
            "native direct callable construction requires a validated recurrence "
            "catalog"
        )
    states = {state.template_id: state for state in recurrence_catalog.current_states}
    semantic_records = {
        record.template_id: record
        for records in (
            recurrence_catalog.sources,
            recurrence_catalog.transitions,
            recurrence_catalog.propagators,
            recurrence_catalog.closures,
        )
        for record in records
    }
    result: dict[int, PreparedNativeDirectCallableSpecV1] = {}
    for binding in recurrence_catalog.evaluator_bindings:
        if binding.callable_kind != "prepared-kernel":
            continue
        role = _CONTRACT_ROLES.get(binding.contract_kind)
        if role is None:
            continue
        if role == "source":
            raise RecurrenceDirectTemplateError(
                "prepared native recurrence sources are unsupported; source "
                "filling must remain Rusticol-owned"
            )
        assert binding.prepared_kernel_id is not None
        kernel_id = binding.prepared_kernel_id
        kernel = prepared_kernels.get(kernel_id)
        if kernel is None:
            raise RecurrenceDirectTemplateError(
                f"native direct callable references absent prepared kernel {kernel_id}"
            )
        inputs = tuple(getattr(kernel, "inputs", ()))
        exact_expressions = tuple(getattr(kernel, "exact_expressions", ()))
        output_arity = len(exact_expressions)
        if getattr(kernel, "kernel_id", None) != kernel_id:
            raise RecurrenceDirectTemplateError(
                f"native direct kernel inventory entry {kernel_id} identifies "
                f"{getattr(kernel, 'kernel_id', None)!r}"
            )
        if getattr(kernel, "contract_kind", None) != binding.contract_kind:
            raise RecurrenceDirectTemplateError(
                f"native direct kernel {kernel_id} contract kind does not match "
                "its recurrence evaluator binding"
            )
        input_contracts = tuple(
            _canonical_json(item.to_dict()) for item in inputs
        )
        concrete_parent_component_counts = tuple(
            states[state_id].dimension for state_id in binding.input_state_template_ids
        )
        parent_component_counts = _canonical_parent_component_counts(
            binding.semantic_template_ids,
            semantic_records,
            concrete_parent_component_counts,
        )
        destination_component_count = (
            states[binding.output_state_template_id].dimension
            if binding.output_state_template_id is not None
            else 1
        )
        _uniform_binding_coupling(
            binding.semantic_template_ids,
            semantic_records,
            required=_input_contracts_use_inline_coupling(input_contracts),
        )
        spec = PreparedNativeDirectCallableSpecV1(
            prepared_kernel_id=kernel_id,
            role=cast(DirectRole, role),
            native_entry_point=native_direct_entry_point(
                cast(DirectRole, role), kernel_id
            ),
            input_contracts=input_contracts,
            exact_expressions=exact_expressions,
            output_arity=output_arity,
            parent_component_shapes=(parent_component_counts,),
            destination_component_counts=(destination_component_count,),
        )
        _validate_prepared_native_direct_projection_bounds(spec)
        previous = result.get(kernel_id)
        result[kernel_id] = (
            spec
            if previous is None
            else _merge_prepared_native_direct_callable_specs(previous, spec)
        )
    return result


def _merge_prepared_native_direct_callable_specs(
    left: PreparedNativeDirectCallableSpecV1,
    right: PreparedNativeDirectCallableSpecV1,
) -> PreparedNativeDirectCallableSpecV1:
    identity_fields = (
        "prepared_kernel_id",
        "role",
        "native_entry_point",
        "input_contracts",
        "exact_expressions",
        "output_arity",
    )
    conflicts = tuple(
        name for name in identity_fields if getattr(left, name) != getattr(right, name)
    )
    left_parent_arity = len(left.parent_component_shapes[0])
    right_parent_arity = len(right.parent_component_shapes[0])
    if left_parent_arity != right_parent_arity:
        conflicts += ("parent_arity",)
    if conflicts:
        raise RecurrenceDirectTemplateError(
            f"prepared native kernel {left.prepared_kernel_id} has incompatible "
            "recurrence bindings for "
            + ", ".join(conflicts)
        )
    merged = replace(
        left,
        parent_component_shapes=tuple(
            sorted(set(left.parent_component_shapes + right.parent_component_shapes))
        ),
        destination_component_counts=tuple(
            sorted(
                set(
                    left.destination_component_counts
                    + right.destination_component_counts
                )
            )
        ),
    )
    _validate_prepared_native_direct_projection_bounds(merged)
    return merged


def _validate_prepared_native_direct_projection_bounds(
    spec: PreparedNativeDirectCallableSpecV1,
) -> None:
    parent_arity = len(spec.parent_component_shapes[0])
    for raw_contract in spec.input_contracts:
        try:
            contract = json.loads(raw_contract)
        except (TypeError, ValueError) as exc:
            raise RecurrenceDirectTemplateError(
                "prepared native direct input contract is not valid JSON"
            ) from exc
        if not isinstance(contract, Mapping):
            raise RecurrenceDirectTemplateError(
                "prepared native direct input contract must be an object"
            )
        role = contract.get("role")
        if role not in {"left-current", "right-current", "current"}:
            continue
        component = contract.get("component")
        if type(component) is not int or component < 0:
            raise RecurrenceDirectTemplateError(
                "prepared native direct current projection must use a "
                "nonnegative component"
            )
        parent = 1 if role == "right-current" else 0
        if parent >= parent_arity:
            raise RecurrenceDirectTemplateError(
                f"prepared native direct {role} projection has no parent"
            )
        if any(
            component >= shape[parent] for shape in spec.parent_component_shapes
        ):
            raise RecurrenceDirectTemplateError(
                f"prepared native direct {role} component {component} is outside "
                "an admitted parent shape"
            )


def build_recurrence_direct_template_catalog(
    recurrence_catalog: RecurrenceTemplateCatalog,
    *,
    backend: DirectBackend,
    target_triple: str,
    portable: bool,
    optimization_level: int,
    prepared_kernel_pack_digest: str,
    prepared_kernel_contract_digest: str,
    prepared_kernel_payload_digest: str,
    optimization_settings_digest: str,
    prepared_kernel_payload_digests: Mapping[int, str],
    prepared_direct_payload_bindings: (
        Mapping[int, RecurrenceDirectPayloadBindingV1] | None
    ) = None,
    prepared_jit_sources: Mapping[int, PreparedJitDirectSourceV1] | None = None,
    prepared_native_sources: Mapping[int, PreparedNativeDirectSourceV1] | None = None,
    alignment_bytes: int = 64,
) -> RecurrenceDirectTemplateCatalogV1:
    """Derive stable direct executors from one authenticated semantic catalog.

    ``prepared_jit_sources`` references each portable O2 plane application and
    derives its authenticated P-kernel arena binding. ``prepared_native_sources``
    references target-native split-real libraries and their typed exported
    entry points. Omitting both records prepared kernels as pending and never
    treats their packed eager evaluator as a direct call.
    """

    if not isinstance(recurrence_catalog, RecurrenceTemplateCatalog):
        raise TypeError(
            "direct template construction requires a validated recurrence catalog"
        )
    states = {state.template_id: state for state in recurrence_catalog.current_states}
    semantic_records = {
        record.template_id: record
        for records in (
            recurrence_catalog.sources,
            recurrence_catalog.transitions,
            recurrence_catalog.propagators,
            recurrence_catalog.closures,
        )
        for record in records
    }
    supplied_direct = dict(prepared_direct_payload_bindings or {})
    jit_sources = dict(prepared_jit_sources or {})
    native_sources = dict(prepared_native_sources or {})
    if backend == "jit" and native_sources:
        raise RecurrenceDirectTemplateError(
            "prepared JIT direct catalogs cannot reference native direct sources"
        )
    if backend == "jit":
        for kernel_id, source in jit_sources.items():
            if source.source_application_abi != SYMJIT_PLANE_APPLICATION_ABI:
                raise RecurrenceDirectTemplateError(
                    f"prepared JIT direct source for kernel {kernel_id} is not "
                    "a SymJIT plane application; regenerate the prepared model"
                )
    if backend in {"cpp", "asm"} and jit_sources:
        raise RecurrenceDirectTemplateError(
            "prepared native direct catalogs cannot reference SymJIT sources"
        )
    expected_native_source_abi = {
        "cpp": "symbolica.compiled-cpp.complex-f64.v1",
        "asm": "symbolica.compiled-asm.complex-f64.v1",
    }.get(backend)
    if expected_native_source_abi is not None:
        for kernel_id, source in native_sources.items():
            if source.source_application_abi != expected_native_source_abi:
                raise RecurrenceDirectTemplateError(
                    f"prepared {backend} direct source for kernel {kernel_id} "
                    "has the wrong target-native application ABI"
                )
    candidates: list[dict[str, object]] = []

    for evaluator_binding_id, binding in enumerate(
        recurrence_catalog.evaluator_bindings
    ):
        role = _CONTRACT_ROLES.get(binding.contract_kind)
        if role is None:
            continue
        concrete_parent_component_counts = tuple(
            states[state_id].dimension for state_id in binding.input_state_template_ids
        )
        parent_component_counts = _canonical_parent_component_counts(
            binding.semantic_template_ids,
            semantic_records,
            concrete_parent_component_counts,
        )
        destination_component_count = (
            states[binding.output_state_template_id].dimension
            if binding.output_state_template_id is not None
            else 1
        )
        if binding.callable_kind == "rusticol-template":
            assert binding.runtime_template is not None
            payload_binding = RecurrenceDirectPayloadBindingV1(
                kind="rusticol-intrinsic",
                runtime_template=binding.runtime_template,
                payload_digest=_digest(
                    {
                        "abi": RECURRENCE_DIRECT_BACKEND_ABI,
                        "callable_signature": binding.callable_signature,
                        "contribution_parent_permutation": [0, 1],
                        "runtime_template": binding.runtime_template,
                    }
                ),
            )
        else:
            assert binding.prepared_kernel_id is not None
            kernel_id = binding.prepared_kernel_id
            kernel_payload_digest = _require_sha256(
                f"prepared kernel {kernel_id} payload digest",
                prepared_kernel_payload_digests.get(kernel_id),
            )
            payload_binding = supplied_direct.get(kernel_id)
            if (
                payload_binding is not None
                and payload_binding.prepared_kernel_id != kernel_id
            ):
                raise RecurrenceDirectTemplateError(
                    f"direct payload binding for kernel {kernel_id} identifies "
                    f"kernel {payload_binding.prepared_kernel_id}"
                )
        coupling_slots, parameter_slots = _slot_counts(
            binding.semantic_template_ids,
            semantic_records,
            binding.input_layout,
        )
        candidate: dict[str, object] = {
            "evaluator_binding_id": evaluator_binding_id,
            "evaluator_resolver_key": binding.resolver_key,
            "role": role,
            "parent_component_counts": parent_component_counts,
            "destination_component_count": destination_component_count,
            "momentum_operand_count": (
                1
                if role in {"source", "finalization"}
                else len(parent_component_counts)
            ),
            "coupling_slot_count": coupling_slots,
            "parameter_slot_count": parameter_slots,
            "semantic_template_ids": binding.semantic_template_ids,
            "exact_expression_digest": _digest(
                {
                    "callable_signature": binding.callable_signature,
                    "exact_expression_digests": list(binding.exact_expression_digests),
                }
            ),
            "template_id": f"direct:{role}:{binding.semantic_digest[:24]}",
        }
        if binding.callable_kind != "rusticol-template":
            assert binding.prepared_kernel_id is not None
            kernel_id = binding.prepared_kernel_id
            if payload_binding is None:
                source = (
                    jit_sources.get(kernel_id)
                    if backend == "jit"
                    else native_sources.get(kernel_id)
                )
                if source is not None:
                    if source.prepared_kernel_id != kernel_id:
                        raise RecurrenceDirectTemplateError(
                            f"prepared direct source for kernel {kernel_id} "
                            f"identifies kernel {source.prepared_kernel_id}"
                        )
                    binding_coupling = _uniform_binding_coupling(
                        binding.semantic_template_ids,
                        semantic_records,
                        required=_input_contracts_use_inline_coupling(
                            source.input_contracts
                        ),
                    )
                    certified_intrinsic = None
                    finalization_intrinsic = None
                    if role == "contribution":
                        certified_intrinsic = certify_recurrence_contribution_intrinsic(
                            exact_expressions=source.exact_expressions,
                            input_contracts=source.input_contracts,
                            parent_component_counts=parent_component_counts,
                            destination_component_count=destination_component_count,
                            binding_coupling=binding_coupling,
                            allow_nontrivial_parent_permutation=True,
                        )
                    elif role == "finalization":
                        finalization_intrinsic = (
                            certify_recurrence_finalization_intrinsic(
                                exact_expressions=source.exact_expressions,
                                input_contracts=source.input_contracts,
                                component_count=destination_component_count,
                            )
                        )
                    if certified_intrinsic is not None:
                        payload_binding = _build_certified_intrinsic_binding(
                            certified_intrinsic
                        )
                    elif finalization_intrinsic is not None:
                        payload_binding = _build_runtime_intrinsic_binding(
                            runtime_template=finalization_intrinsic
                        )
                    elif isinstance(source, PreparedJitDirectSourceV1):
                        payload_binding = _build_prepared_jit_direct_binding(
                            source=source,
                            role=cast(DirectRole, role),
                            parent_component_counts=parent_component_counts,
                            destination_component_count=destination_component_count,
                            binding_coupling=binding_coupling,
                            prepared_template_semantic_digest=(
                                _prepared_template_contract_digest(
                                    candidate,
                                    backend=backend,
                                    target_triple=target_triple,
                                    portable=portable,
                                    optimization_level=optimization_level,
                                    alignment_bytes=alignment_bytes,
                                )
                            ),
                        )
                    else:
                        payload_binding = _build_prepared_native_direct_binding(
                            source=source,
                            role=cast(DirectRole, role),
                            parent_component_counts=parent_component_counts,
                            destination_component_count=destination_component_count,
                            binding_coupling=binding_coupling,
                            prepared_template_semantic_digest=(
                                _prepared_template_contract_digest(
                                    candidate,
                                    backend=backend,
                                    target_triple=target_triple,
                                    portable=portable,
                                    optimization_level=optimization_level,
                                    alignment_bytes=alignment_bytes,
                                )
                            ),
                        )
            if payload_binding is None:
                payload_binding = RecurrenceDirectPayloadBindingV1(
                    kind="pending-direct-call-abi",
                    prepared_kernel_id=kernel_id,
                    payload_digest=_digest(
                        {
                            "contribution_parent_permutation": [0, 1],
                            "kind": "pending-direct-call-abi",
                            "prepared_kernel_id": kernel_id,
                            "prepared_kernel_payload_digest": kernel_payload_digest,
                            "required_backend_abi": RECURRENCE_DIRECT_BACKEND_ABI,
                        }
                    ),
                )
        candidate["payload_binding"] = payload_binding
        candidates.append(candidate)

    synthetic_binding_id = len(recurrence_catalog.evaluator_bindings)
    identity_propagators = tuple(
        propagator
        for propagator in recurrence_catalog.propagators
        if not propagator.applies_propagator
    )
    if identity_propagators:
        identity_states = tuple(
            states[propagator.state_template_id] for propagator in identity_propagators
        )
        maximum_component_count = max(state.dimension for state in identity_states)
        identity_semantics = {
            "abi": RECURRENCE_DIRECT_BACKEND_ABI,
            "component_count_mode": "row",
            "maximum_component_count": maximum_component_count,
            "operation": RECURRENCE_DIRECT_IDENTITY_FINALIZER,
            "state_semantic_digests": sorted(
                state.semantic_digest for state in identity_states
            ),
        }
        runtime_template = RECURRENCE_DIRECT_IDENTITY_FINALIZER
        candidates.append(
            {
                "evaluator_binding_id": synthetic_binding_id,
                "evaluator_resolver_key": runtime_template,
                "role": "finalization",
                "parent_component_counts": (maximum_component_count,),
                "destination_component_count": maximum_component_count,
                "momentum_operand_count": 1,
                "coupling_slot_count": 0,
                "parameter_slot_count": 0,
                "semantic_template_ids": tuple(
                    sorted(
                        propagator.template_id for propagator in identity_propagators
                    )
                ),
                "exact_expression_digest": _digest(identity_semantics),
                "payload_binding": RecurrenceDirectPayloadBindingV1(
                    kind="rusticol-intrinsic",
                    runtime_template=runtime_template,
                    payload_digest=_digest(
                        {
                            **identity_semantics,
                            "contribution_parent_permutation": [0, 1],
                            "runtime_template": runtime_template,
                        }
                    ),
                ),
                "template_id": "direct:identity-finalization",
            }
        )

    candidates.sort(
        key=lambda item: (
            _ROLE_INDEX[cast(str, item["role"])],
            cast(int, item["evaluator_binding_id"]),
            cast(str, item["template_id"]),
        )
    )
    templates = tuple(
        RecurrenceDirectTemplateV1(
            template_id=cast(str, item["template_id"]),
            direct_executor_id=direct_executor_id,
            evaluator_binding_id=cast(int, item["evaluator_binding_id"]),
            evaluator_resolver_key=cast(str, item["evaluator_resolver_key"]),
            role=cast(DirectRole, item["role"]),
            parent_arity=len(cast(tuple[int, ...], item["parent_component_counts"])),
            parent_component_counts=cast(
                tuple[int, ...], item["parent_component_counts"]
            ),
            destination_component_count=cast(int, item["destination_component_count"]),
            momentum_operand_count=cast(int, item["momentum_operand_count"]),
            destination_operation=cast(
                DirectDestinationOperation,
                _DESTINATION_OPERATIONS[cast(str, item["role"])],
            ),
            coupling_slot_count=cast(int, item["coupling_slot_count"]),
            parameter_slot_count=cast(int, item["parameter_slot_count"]),
            semantic_template_ids=cast(tuple[str, ...], item["semantic_template_ids"]),
            exact_expression_digest=cast(str, item["exact_expression_digest"]),
            payload_binding=cast(
                RecurrenceDirectPayloadBindingV1, item["payload_binding"]
            ),
            backend=backend,
            target_triple=target_triple,
            portable=portable,
            optimization_level=optimization_level,
            alignment_bytes=alignment_bytes,
            simd_axis="points-contiguous",
            destination_aliasing=item["role"] == "finalization",
        )
        for direct_executor_id, item in enumerate(candidates)
    )
    return RecurrenceDirectTemplateCatalogV1(
        templates=templates,
        backend=backend,
        target_triple=target_triple,
        portable=portable,
        optimization_level=optimization_level,
        compiled_model_digest=recurrence_catalog.header.compiled_model_digest,
        recurrence_template_catalog_digest=recurrence_catalog.catalog_digest,
        prepared_kernel_pack_digest=prepared_kernel_pack_digest,
        prepared_kernel_contract_digest=prepared_kernel_contract_digest,
        prepared_kernel_payload_digest=prepared_kernel_payload_digest,
        optimization_settings_digest=optimization_settings_digest,
    )


def _prepared_template_contract_digest(
    candidate: Mapping[str, object],
    *,
    backend: DirectBackend,
    target_triple: str,
    portable: bool,
    optimization_level: int,
    alignment_bytes: int,
) -> str:
    """Authenticate template semantics without creating a payload-digest cycle."""

    return _digest(
        {
            "abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
            "alignment_bytes": alignment_bytes,
            "backend": backend,
            "coupling_slot_count": candidate["coupling_slot_count"],
            "destination_aliasing": candidate["role"] == "finalization",
            "destination_component_count": candidate["destination_component_count"],
            "destination_operation": _DESTINATION_OPERATIONS[
                cast(str, candidate["role"])
            ],
            "evaluator_binding_id": candidate["evaluator_binding_id"],
            "evaluator_resolver_key": candidate["evaluator_resolver_key"],
            "exact_expression_digest": candidate["exact_expression_digest"],
            "momentum_operand_count": candidate["momentum_operand_count"],
            "optimization_level": optimization_level,
            "parameter_slot_count": candidate["parameter_slot_count"],
            "parent_component_counts": list(
                cast(tuple[int, ...], candidate["parent_component_counts"])
            ),
            "portable": portable,
            "role": candidate["role"],
            "semantic_template_ids": list(
                cast(tuple[str, ...], candidate["semantic_template_ids"])
            ),
            "simd_axis": "points-contiguous",
            "target_triple": target_triple,
            "template_id": candidate["template_id"],
        }
    )


def _build_prepared_jit_direct_binding(
    *,
    source: PreparedJitDirectSourceV1,
    role: DirectRole,
    parent_component_counts: tuple[int, ...],
    destination_component_count: int,
    binding_coupling: ExactComplexRationalV1 | None,
    prepared_template_semantic_digest: str,
) -> RecurrenceDirectPayloadBindingV1:
    if role == "source":
        raise RecurrenceDirectTemplateError(
            "recurrence sources must remain Rusticol SourceIR intrinsics"
        )
    contracts = _decode_canonical_objects(source.input_contracts)
    input_plane_projections: list[dict[str, object]] = []
    scalar_projections: list[dict[str, object]] = [
        {"imaginary": False, "kind": "exact-factor"},
        {"imaginary": True, "kind": "exact-factor"},
    ]
    parameter_bindings: list[dict[str, object]] = []
    zero_scalar_index: int | None = None

    def append_zero_scalar_binding() -> None:
        nonlocal zero_scalar_index
        if zero_scalar_index is None:
            zero_scalar_index = len(scalar_projections)
            scalar_projections.append({"kind": "literal", "value": 0.0})
        parameter_bindings.append({"index": zero_scalar_index, "kind": "scalar"})

    for contract in contracts:
        if not isinstance(contract, Mapping):
            raise RecurrenceDirectTemplateError(
                "prepared JIT direct input contract must be an object"
            )
        input_role = contract.get("role")
        component = _require_nonnegative_int(
            "prepared JIT direct input component", contract.get("component")
        )
        if input_role in {"left-current", "right-current", "current"}:
            parent = 1 if input_role == "right-current" else 0
            if parent >= len(parent_component_counts):
                raise RecurrenceDirectTemplateError(
                    f"prepared JIT direct {input_role} input has no parent"
                )
            for imaginary in (False, True):
                parameter_bindings.append(
                    {"index": len(input_plane_projections), "kind": "plane"}
                )
                input_plane_projections.append(
                    {
                        "component": component,
                        "imaginary": imaginary,
                        "kind": "parent-current",
                        "parent": parent,
                    }
                )
        elif input_role in {"left-momentum", "right-momentum", "momentum"}:
            operand = 1 if input_role == "right-momentum" else 0
            parameter_bindings.append(
                {"index": len(input_plane_projections), "kind": "plane"}
            )
            input_plane_projections.append(
                {
                    "kind": "momentum",
                    "lorentz_component": component,
                    "operand": operand,
                }
            )
            # Portable complex SymJIT applications expose every original input
            # as adjacent real/imaginary parameters. Physical momenta are real.
            append_zero_scalar_binding()
        elif input_role in {"coupling-real", "coupling-imag"}:
            if binding_coupling is None:
                raise RecurrenceDirectTemplateError(
                    "prepared JIT direct coupling input has no uniform exact "
                    "semantic binding"
                )
            scalar_index = len(scalar_projections)
            parameter_bindings.append({"index": scalar_index, "kind": "scalar"})
            scalar_projections.append(
                {
                    "kind": "literal",
                    "value": float(
                        binding_coupling.imag
                        if input_role == "coupling-imag"
                        else binding_coupling.real
                    ),
                }
            )
            append_zero_scalar_binding()
        elif input_role == "model-parameter":
            parameter_index = _require_nonnegative_int(
                "prepared JIT direct model-parameter index",
                contract.get("model_parameter_index"),
            )
            for imaginary in (False, True):
                scalar_index = len(scalar_projections)
                parameter_bindings.append({"index": scalar_index, "kind": "scalar"})
                scalar_projections.append(
                    {
                        "imaginary": imaginary,
                        "index": parameter_index,
                        "kind": "parameter",
                    }
                )
        else:
            raise RecurrenceDirectTemplateError(
                f"unsupported prepared JIT direct input role {input_role!r}"
            )

    destination_kind = (
        "destination-amplitude" if role == "closure" else "destination-current"
    )
    output_alias_inputs: list[int] = []
    for component in range(source.output_arity):
        if component >= destination_component_count:
            raise RecurrenceDirectTemplateError(
                "prepared JIT direct output exceeds destination component count"
            )
        for imaginary in (False, True):
            output_alias_inputs.append(len(input_plane_projections))
            input_plane_projections.append(
                {
                    "component": component,
                    "imaginary": imaginary,
                    "kind": destination_kind,
                }
            )

    metadata: dict[str, object] = {
        "abi": RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI,
        "contribution_parent_permutation": [0, 1],
        "destination_operation": _DESTINATION_OPERATIONS[role],
        "direct_application_abi": SYMJIT_DIRECT_APPLICATION_ABI,
        "exact_factor_scalar_slots": [0, 1],
        "input_plane_count": len(input_plane_projections),
        "input_plane_projections": input_plane_projections,
        "intrinsic_contract_digest": None,
        "kind": "prepared-direct-call",
        "output_alias_inputs": output_alias_inputs,
        "parameter_bindings": parameter_bindings,
        "payload_paths": [source.source_application_path],
        "prepared_kernel_id": source.prepared_kernel_id,
        "prepared_template_semantic_digest": prepared_template_semantic_digest,
        "role": role,
        "runtime_template": None,
        "scalar_input_count": len(scalar_projections),
        "scalar_projections": scalar_projections,
        "source_application_abi": source.source_application_abi,
        "source_application_path": source.source_application_path,
        "source_application_sha256": source.source_application_sha256,
        "state_plane_indices": [],
    }
    return RecurrenceDirectPayloadBindingV1(
        kind="prepared-direct-call",
        payload_digest=_digest(metadata),
        prepared_kernel_id=source.prepared_kernel_id,
        payload_paths=(source.source_application_path,),
        source_application_path=source.source_application_path,
        source_application_sha256=source.source_application_sha256,
        source_application_abi=source.source_application_abi,
        direct_application_abi=SYMJIT_DIRECT_APPLICATION_ABI,
        role=role,
        destination_operation=cast(
            DirectDestinationOperation, _DESTINATION_OPERATIONS[role]
        ),
        exact_factor_scalar_slots=(0, 1),
        state_plane_indices=(),
        parameter_bindings=_encode_canonical_objects(parameter_bindings),
        input_plane_count=len(input_plane_projections),
        scalar_input_count=len(scalar_projections),
        output_alias_inputs=tuple(output_alias_inputs),
        input_plane_projections=_encode_canonical_objects(input_plane_projections),
        scalar_projections=_encode_canonical_objects(scalar_projections),
        prepared_template_semantic_digest=prepared_template_semantic_digest,
    )


def _build_prepared_native_direct_binding(
    *,
    source: PreparedNativeDirectSourceV1,
    role: DirectRole,
    parent_component_counts: tuple[int, ...],
    destination_component_count: int,
    binding_coupling: ExactComplexRationalV1 | None,
    prepared_template_semantic_digest: str,
) -> RecurrenceDirectPayloadBindingV1:
    """Bind one native export while authenticating JIT-equivalent projections."""

    if source.role != role:
        raise RecurrenceDirectTemplateError(
            "prepared native direct source role does not match its template"
        )
    if source.native_entry_point != native_direct_entry_point(
        role, source.prepared_kernel_id
    ):
        raise RecurrenceDirectTemplateError(
            "prepared native direct source entry point is not canonical"
        )
    # The scalar and plane projection contract is backend-independent. Reuse the
    # established JIT projection derivation only as authenticated metadata; the
    # native executable consumes arena views and typed rows directly and never
    # packs these projections at runtime.
    projected = _build_prepared_jit_direct_binding(
        source=PreparedJitDirectSourceV1(
            prepared_kernel_id=source.prepared_kernel_id,
            source_application_path=source.source_application_path,
            source_application_sha256=source.source_application_sha256,
            source_application_abi=source.source_application_abi,
            input_contracts=source.input_contracts,
            exact_expressions=source.exact_expressions,
            output_arity=source.output_arity,
        ),
        role=role,
        parent_component_counts=parent_component_counts,
        destination_component_count=destination_component_count,
        binding_coupling=binding_coupling,
        prepared_template_semantic_digest=prepared_template_semantic_digest,
    )
    metadata = projected._prepared_call_fields(include_payload_digest=False)
    metadata["direct_application_abi"] = NATIVE_DIRECT_APPLICATION_ABI
    metadata["native_entry_point"] = source.native_entry_point
    return replace(
        projected,
        direct_application_abi=NATIVE_DIRECT_APPLICATION_ABI,
        native_entry_point=source.native_entry_point,
        payload_digest=_digest(metadata),
    )


def _build_certified_intrinsic_binding(
    certified: CertifiedRecurrenceIntrinsic,
) -> RecurrenceDirectPayloadBindingV1:
    runtime_template = _require_nonempty(
        "certified intrinsic runtime template",
        certified.runtime_template,
    )
    contract_digest = _require_sha256(
        "certified intrinsic contract digest",
        certified.contract_digest,
    )
    scale = certified.scale_projection()
    metadata = {
        "abi": RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI,
        "contribution_parent_permutation": list(certified.parent_permutation),
        "destination_operation": "add",
        "intrinsic_contract_digest": contract_digest,
        "kind": "rusticol-intrinsic",
        "role": "contribution",
        "runtime_template": runtime_template,
        "scalar_input_count": 1,
        "scalar_projections": [scale],
    }
    return RecurrenceDirectPayloadBindingV1(
        kind="rusticol-intrinsic",
        payload_digest=_digest(metadata),
        runtime_template=runtime_template,
        role="contribution",
        destination_operation="add",
        scalar_input_count=1,
        scalar_projections=_encode_canonical_objects((scale,)),
        intrinsic_contract_digest=contract_digest,
        contribution_parent_permutation=certified.parent_permutation,
    )


def _build_runtime_intrinsic_binding(
    *,
    runtime_template: str,
) -> RecurrenceDirectPayloadBindingV1:
    runtime_template = _require_nonempty(
        "direct intrinsic runtime template", runtime_template
    )
    metadata = {
        "abi": RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI,
        "contribution_parent_permutation": [0, 1],
        "kind": "rusticol-intrinsic",
        "runtime_template": runtime_template,
    }
    return RecurrenceDirectPayloadBindingV1(
        kind="rusticol-intrinsic",
        payload_digest=_digest(metadata),
        runtime_template=runtime_template,
    )


def _source_uses_inline_coupling(source: PreparedJitDirectSourceV1) -> bool:
    return _input_contracts_use_inline_coupling(source.input_contracts)


def _input_contracts_use_inline_coupling(
    input_contracts: Sequence[str],
) -> bool:
    for contract in _decode_canonical_objects(input_contracts):
        if isinstance(contract, Mapping) and contract.get("role") in {
            "coupling-real",
            "coupling-imag",
        }:
            return True
    return False


def _uniform_binding_coupling(
    semantic_template_ids: Sequence[str],
    semantic_records: Mapping[str, object],
    *,
    required: bool,
) -> ExactComplexRationalV1 | None:
    if not required:
        return None
    couplings: set[ExactComplexRationalV1] = set()
    missing: list[str] = []
    for template_id in semantic_template_ids:
        record = semantic_records.get(template_id)
        coupling = getattr(record, "binding_coupling", None)
        if not isinstance(coupling, ExactComplexRationalV1):
            missing.append(template_id)
        else:
            couplings.add(coupling)
    if missing:
        raise RecurrenceDirectTemplateError(
            "prepared JIT direct coupling input is not owned by every semantic "
            "template: " + ", ".join(sorted(missing))
        )
    if len(couplings) != 1:
        raise RecurrenceDirectTemplateError(
            "one prepared JIT direct evaluator binding has conflicting exact "
            "semantic couplings"
        )
    return next(iter(couplings))


def _slot_counts(
    semantic_template_ids: Sequence[str],
    semantic_records: Mapping[str, object],
    input_layout: Sequence[str],
) -> tuple[int, int]:
    coupling_ids: set[str] = set()
    parameter_ids: set[str] = set()
    for template_id in semantic_template_ids:
        record = semantic_records.get(template_id)
        if record is None:
            continue
        coupling_ids.update(getattr(record, "coupling_parameter_ids", ()))
        for name in ("mass_parameter_id", "width_parameter_id"):
            value = getattr(record, name, None)
            if value is not None:
                parameter_ids.add(str(value))
    for raw in input_layout:
        try:
            contract = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(contract, Mapping):
            continue
        model_parameter = contract.get("model_parameter_name")
        role = contract.get("role")
        if isinstance(model_parameter, str) and model_parameter:
            if isinstance(role, str) and "coupling" in role:
                coupling_ids.add(model_parameter)
            else:
                parameter_ids.add(model_parameter)
    return len(coupling_ids), len(parameter_ids)


def _canonical_parent_component_counts(
    semantic_template_ids: Sequence[str],
    semantic_records: Mapping[str, object],
    concrete_counts: tuple[int, ...],
) -> tuple[int, ...]:
    canonical_shapes: set[tuple[int, ...]] = set()
    for template_id in semantic_template_ids:
        record = semantic_records.get(template_id)
        order = getattr(record, "canonical_input_order", None)
        if order is None:
            continue
        if len(order) != len(concrete_counts) or set(order) != set(
            range(len(concrete_counts))
        ):
            raise RecurrenceDirectTemplateError(
                f"semantic template {template_id!r} has an invalid canonical "
                "input order"
            )
        canonical_shapes.add(tuple(concrete_counts[index] for index in order))
    if len(canonical_shapes) > 1:
        raise RecurrenceDirectTemplateError(
            "one evaluator binding has incompatible canonical parent shapes"
        )
    return next(iter(canonical_shapes), concrete_counts)


def prepared_kernel_payload_digest(
    *,
    kernel_id: int,
    payload_records: Mapping[str, tuple[int, str]],
    referenced_paths: Sequence[str],
) -> str:
    """Digest exactly the payload bytes referenced by one prepared kernel."""

    rows: list[dict[str, object]] = []
    for path in sorted(set(referenced_paths)):
        try:
            size, digest = payload_records[path]
        except KeyError as exc:
            raise RecurrenceDirectTemplateError(
                f"prepared kernel {kernel_id} payload {path!r} is absent"
            ) from exc
        _require_nonnegative_int(f"prepared payload {path!r} size", size)
        _require_sha256(f"prepared payload {path!r} digest", digest)
        rows.append({"path": path, "sha256": digest, "size": size})
    if not rows:
        raise RecurrenceDirectTemplateError(
            f"prepared kernel {kernel_id} has no payload identity records"
        )
    return _digest(
        {
            "abi": RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI,
            "kernel_id": kernel_id,
            "payloads": rows,
        }
    )


__all__ = [
    "NATIVE_DIRECT_APPLICATION_ABI",
    "RECURRENCE_DIRECT_BACKEND_ABI",
    "RECURRENCE_DIRECT_CANONICALIZATION_ABI",
    "RECURRENCE_DIRECT_IDENTITY_FINALIZER",
    "RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI",
    "RECURRENCE_DIRECT_TEMPLATE_ABI",
    "SYMJIT_DIRECT_APPLICATION_ABI",
    "PreparedJitDirectSourceV1",
    "PreparedNativeDirectCallableSpecV1",
    "PreparedNativeDirectSourceV1",
    "RecurrenceDirectPayloadBindingV1",
    "RecurrenceDirectTemplateCatalogV1",
    "RecurrenceDirectTemplateError",
    "RecurrenceDirectTemplateV1",
    "build_prepared_native_direct_callable_specs",
    "build_recurrence_direct_template_catalog",
    "native_direct_entry_point",
    "prepared_kernel_payload_digest",
]
