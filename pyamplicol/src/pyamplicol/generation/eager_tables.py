# SPDX-License-Identifier: 0BSD
"""Fixed-width little-endian tables for eager DAG execution."""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar, Protocol, TypeVar

from .._internal.versions import EAGER_DAG_F64_RUNTIME_CAPABILITY
from ..models.prepared import EAGER_KERNEL_ABI

EAGER_PLAN_ABI = "pyamplicol-eager-plan-v2"
EAGER_RUNTIME_CAPABILITY = EAGER_DAG_F64_RUNTIME_CAPABILITY
EAGER_SELECTOR_DOMAINS_ABI = "pyamplicol-eager-selector-domains-v1"
MISSING_U32 = (1 << 32) - 1
EAGER_OUTPUT_FACTOR_NONE = 0
EAGER_OUTPUT_FACTOR_COUPLING_REAL = 1
EAGER_OUTPUT_FACTOR_COUPLING_IMAG = 2
_EAGER_OUTPUT_FACTOR_SOURCES = frozenset(
    (
        EAGER_OUTPUT_FACTOR_NONE,
        EAGER_OUTPUT_FACTOR_COUPLING_REAL,
        EAGER_OUTPUT_FACTOR_COUPLING_IMAG,
    )
)


class _FixedWidthRow(Protocol):
    _STRUCT: ClassVar[struct.Struct]

    def _values(self) -> tuple[int | float, ...]: ...

    @classmethod
    def _from_values(cls, values: tuple[int | float, ...]) -> _FixedWidthRow: ...


_RowT = TypeVar("_RowT", bound=_FixedWidthRow)


def _u32(value: int, field: str) -> int:
    integer = int(value)
    if integer < 0 or integer > MISSING_U32:
        raise ValueError(f"{field} must fit in an unsigned 32-bit integer")
    return integer


def _u64(value: int, field: str) -> int:
    integer = int(value)
    if integer < 0 or integer >= 1 << 64:
        raise ValueError(f"{field} must fit in an unsigned 64-bit integer")
    return integer


@dataclass(frozen=True, slots=True)
class EagerInvocationRow:
    """One canonical local-kernel evaluation and its attachment range."""

    kernel_id: int
    left_value_slot_id: int
    right_value_slot_id: int
    left_momentum_slot_id: int
    right_momentum_slot_id: int
    coupling_slot_id: int
    output_factor_source: int
    attachment_start: int
    attachment_count: int

    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<IIIIIIIQQ")

    def __post_init__(self) -> None:
        for field in (
            "kernel_id",
            "left_value_slot_id",
            "right_value_slot_id",
            "left_momentum_slot_id",
            "right_momentum_slot_id",
            "coupling_slot_id",
            "output_factor_source",
        ):
            object.__setattr__(self, field, _u32(getattr(self, field), field))
        if self.output_factor_source not in _EAGER_OUTPUT_FACTOR_SOURCES:
            raise ValueError("unsupported eager invocation output factor source")
        for field in ("attachment_start", "attachment_count"):
            object.__setattr__(self, field, _u64(getattr(self, field), field))

    def _values(self) -> tuple[int | float, ...]:
        return (
            self.kernel_id,
            self.left_value_slot_id,
            self.right_value_slot_id,
            self.left_momentum_slot_id,
            self.right_momentum_slot_id,
            self.coupling_slot_id,
            self.output_factor_source,
            self.attachment_start,
            self.attachment_count,
        )

    @classmethod
    def _from_values(cls, values: tuple[int | float, ...]) -> EagerInvocationRow:
        return cls(*(int(value) for value in values))

    @classmethod
    def _from_trusted_values(
        cls,
        kernel_id: int,
        left_value_slot_id: int,
        right_value_slot_id: int,
        left_momentum_slot_id: int,
        right_momentum_slot_id: int,
        coupling_slot_id: int,
        output_factor_source: int,
        attachment_start: int,
        attachment_count: int,
    ) -> EagerInvocationRow:
        """Construct an internally proven row without repeating boundary checks."""

        row = object.__new__(cls)
        object.__setattr__(row, "kernel_id", kernel_id)
        object.__setattr__(row, "left_value_slot_id", left_value_slot_id)
        object.__setattr__(row, "right_value_slot_id", right_value_slot_id)
        object.__setattr__(row, "left_momentum_slot_id", left_momentum_slot_id)
        object.__setattr__(row, "right_momentum_slot_id", right_momentum_slot_id)
        object.__setattr__(row, "coupling_slot_id", coupling_slot_id)
        object.__setattr__(row, "output_factor_source", output_factor_source)
        object.__setattr__(row, "attachment_start", attachment_start)
        object.__setattr__(row, "attachment_count", attachment_count)
        return row


@dataclass(frozen=True, slots=True)
class EagerAttachmentRow:
    """Fan-out from one canonical evaluation to one accumulated current."""

    result_current_id: int
    factor_real: float
    factor_imag: float

    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<Idd")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result_current_id",
            _u32(self.result_current_id, "result_current_id"),
        )
        object.__setattr__(self, "factor_real", float(self.factor_real))
        object.__setattr__(self, "factor_imag", float(self.factor_imag))
        if not isfinite(self.factor_real) or not isfinite(self.factor_imag):
            raise ValueError("attachment factor must be finite")

    def _values(self) -> tuple[int | float, ...]:
        return self.result_current_id, self.factor_real, self.factor_imag

    @classmethod
    def _from_values(cls, values: tuple[int | float, ...]) -> EagerAttachmentRow:
        return cls(int(values[0]), float(values[1]), float(values[2]))

    @classmethod
    def _from_trusted_values(
        cls,
        result_current_id: int,
        factor_real: float,
        factor_imag: float,
    ) -> EagerAttachmentRow:
        row = object.__new__(cls)
        object.__setattr__(row, "result_current_id", result_current_id)
        object.__setattr__(row, "factor_real", factor_real)
        object.__setattr__(row, "factor_imag", factor_imag)
        return row


@dataclass(frozen=True, slots=True)
class EagerCouplingRow:
    """Resolve one complex coupling from constants and/or model parameters."""

    real_parameter_id: int
    imag_parameter_id: int
    constant_real: float
    constant_imag: float

    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<IIdd")

    def __post_init__(self) -> None:
        for field in ("real_parameter_id", "imag_parameter_id"):
            object.__setattr__(self, field, _u32(getattr(self, field), field))
        object.__setattr__(self, "constant_real", float(self.constant_real))
        object.__setattr__(self, "constant_imag", float(self.constant_imag))
        if not isfinite(self.constant_real) or not isfinite(self.constant_imag):
            raise ValueError("coupling constants must be finite")

    def _values(self) -> tuple[int | float, ...]:
        return (
            self.real_parameter_id,
            self.imag_parameter_id,
            self.constant_real,
            self.constant_imag,
        )

    @classmethod
    def _from_values(cls, values: tuple[int | float, ...]) -> EagerCouplingRow:
        return cls(
            int(values[0]),
            int(values[1]),
            float(values[2]),
            float(values[3]),
        )

    @classmethod
    def _from_trusted_values(
        cls,
        real_parameter_id: int,
        imag_parameter_id: int,
        constant_real: float,
        constant_imag: float,
    ) -> EagerCouplingRow:
        row = object.__new__(cls)
        object.__setattr__(row, "real_parameter_id", real_parameter_id)
        object.__setattr__(row, "imag_parameter_id", imag_parameter_id)
        object.__setattr__(row, "constant_real", constant_real)
        object.__setattr__(row, "constant_imag", constant_imag)
        return row


@dataclass(frozen=True, slots=True)
class EagerFinalizationRow:
    """Finalize one accumulated current, applying its propagator at most once."""

    kernel_id: int
    current_id: int
    unpropagated_value_slot_id: int
    propagated_value_slot_id: int
    momentum_slot_id: int

    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<IIIII")

    def __post_init__(self) -> None:
        for field in (
            "kernel_id",
            "current_id",
            "unpropagated_value_slot_id",
            "propagated_value_slot_id",
            "momentum_slot_id",
        ):
            object.__setattr__(self, field, _u32(getattr(self, field), field))

    @property
    def applies_kernel(self) -> bool:
        return self.kernel_id != MISSING_U32

    @property
    def stores_unpropagated(self) -> bool:
        return self.unpropagated_value_slot_id != MISSING_U32

    @property
    def stores_propagated(self) -> bool:
        return self.propagated_value_slot_id != MISSING_U32

    def _values(self) -> tuple[int | float, ...]:
        return (
            self.kernel_id,
            self.current_id,
            self.unpropagated_value_slot_id,
            self.propagated_value_slot_id,
            self.momentum_slot_id,
        )

    @classmethod
    def _from_values(cls, values: tuple[int | float, ...]) -> EagerFinalizationRow:
        return cls(*(int(value) for value in values))

    @classmethod
    def _from_trusted_values(
        cls,
        kernel_id: int,
        current_id: int,
        unpropagated_value_slot_id: int,
        propagated_value_slot_id: int,
        momentum_slot_id: int,
    ) -> EagerFinalizationRow:
        row = object.__new__(cls)
        object.__setattr__(row, "kernel_id", kernel_id)
        object.__setattr__(row, "current_id", current_id)
        object.__setattr__(
            row,
            "unpropagated_value_slot_id",
            unpropagated_value_slot_id,
        )
        object.__setattr__(row, "propagated_value_slot_id", propagated_value_slot_id)
        object.__setattr__(row, "momentum_slot_id", momentum_slot_id)
        return row


@dataclass(frozen=True, slots=True)
class EagerClosureRow:
    """One prepared amplitude-closure call and its physical output slot."""

    kernel_id: int
    left_value_slot_id: int
    right_value_slot_id: int
    amplitude_index: int
    coupling_slot_id: int
    output_factor_source: int
    factor_real: float
    factor_imag: float

    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<IIIIIIdd")

    def __post_init__(self) -> None:
        for field in (
            "kernel_id",
            "left_value_slot_id",
            "right_value_slot_id",
            "amplitude_index",
            "coupling_slot_id",
            "output_factor_source",
        ):
            object.__setattr__(self, field, _u32(getattr(self, field), field))
        if self.output_factor_source not in _EAGER_OUTPUT_FACTOR_SOURCES:
            raise ValueError("unsupported eager closure output factor source")
        object.__setattr__(self, "factor_real", float(self.factor_real))
        object.__setattr__(self, "factor_imag", float(self.factor_imag))
        if not isfinite(self.factor_real) or not isfinite(self.factor_imag):
            raise ValueError("closure factor must be finite")

    def _values(self) -> tuple[int | float, ...]:
        return (
            self.kernel_id,
            self.left_value_slot_id,
            self.right_value_slot_id,
            self.amplitude_index,
            self.coupling_slot_id,
            self.output_factor_source,
            self.factor_real,
            self.factor_imag,
        )

    @classmethod
    def _from_values(cls, values: tuple[int | float, ...]) -> EagerClosureRow:
        return cls(
            int(values[0]),
            int(values[1]),
            int(values[2]),
            int(values[3]),
            int(values[4]),
            int(values[5]),
            float(values[6]),
            float(values[7]),
        )

    @classmethod
    def _from_trusted_values(
        cls,
        kernel_id: int,
        left_value_slot_id: int,
        right_value_slot_id: int,
        amplitude_index: int,
        coupling_slot_id: int,
        output_factor_source: int,
        factor_real: float,
        factor_imag: float,
    ) -> EagerClosureRow:
        row = object.__new__(cls)
        object.__setattr__(row, "kernel_id", kernel_id)
        object.__setattr__(row, "left_value_slot_id", left_value_slot_id)
        object.__setattr__(row, "right_value_slot_id", right_value_slot_id)
        object.__setattr__(row, "amplitude_index", amplitude_index)
        object.__setattr__(row, "coupling_slot_id", coupling_slot_id)
        object.__setattr__(row, "output_factor_source", output_factor_source)
        object.__setattr__(row, "factor_real", factor_real)
        object.__setattr__(row, "factor_imag", factor_imag)
        return row


@dataclass(frozen=True, slots=True)
class EagerSelectorDomainRow:
    """One range into the flattened coherent-group membership table."""

    member_start: int
    member_count: int

    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<QQ")

    def __post_init__(self) -> None:
        for field in ("member_start", "member_count"):
            object.__setattr__(self, field, _u64(getattr(self, field), field))

    def _values(self) -> tuple[int | float, ...]:
        return self.member_start, self.member_count

    @classmethod
    def _from_values(cls, values: tuple[int | float, ...]) -> EagerSelectorDomainRow:
        return cls(int(values[0]), int(values[1]))

    @classmethod
    def _from_trusted_values(
        cls,
        member_start: int,
        member_count: int,
    ) -> EagerSelectorDomainRow:
        row = object.__new__(cls)
        object.__setattr__(row, "member_start", member_start)
        object.__setattr__(row, "member_count", member_count)
        return row


@dataclass(frozen=True, slots=True)
class EagerSelectorGroupRow:
    """One coherent reduction-group member of an interned selector domain."""

    coherent_group_id: int

    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<I")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coherent_group_id",
            _u32(self.coherent_group_id, "coherent_group_id"),
        )

    def _values(self) -> tuple[int | float, ...]:
        return (self.coherent_group_id,)

    @classmethod
    def _from_values(cls, values: tuple[int | float, ...]) -> EagerSelectorGroupRow:
        return cls(int(values[0]))

    @classmethod
    def _from_trusted_values(cls, coherent_group_id: int) -> EagerSelectorGroupRow:
        row = object.__new__(cls)
        object.__setattr__(row, "coherent_group_id", coherent_group_id)
        return row


@dataclass(frozen=True, slots=True)
class EagerSelectorDomainIdRow:
    """One reference to an interned selector domain."""

    domain_id: int

    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<I")

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain_id", _u32(self.domain_id, "domain_id"))

    def _values(self) -> tuple[int | float, ...]:
        return (self.domain_id,)

    @classmethod
    def _from_values(cls, values: tuple[int | float, ...]) -> EagerSelectorDomainIdRow:
        return cls(int(values[0]))

    @classmethod
    def _from_trusted_values(cls, domain_id: int) -> EagerSelectorDomainIdRow:
        row = object.__new__(cls)
        object.__setattr__(row, "domain_id", domain_id)
        return row


def pack_rows(rows: Iterable[_FixedWidthRow]) -> bytes:
    """Serialize homogeneous rows without padding or a redundant header."""

    records = tuple(rows)
    if not records:
        return b""
    row_type = type(records[0])
    if any(type(row) is not row_type for row in records):
        raise TypeError("eager binary tables must contain one row type")
    layout = records[0]._STRUCT
    payload = bytearray(len(records) * layout.size)
    for row_index, row in enumerate(records):
        layout.pack_into(payload, row_index * layout.size, *row._values())
    return bytes(payload)


def unpack_rows(payload: bytes, row_type: type[_RowT]) -> tuple[_RowT, ...]:
    """Deserialize one table, rejecting truncated or extended row payloads."""

    layout = row_type._STRUCT
    if len(payload) % layout.size:
        raise ValueError(
            f"eager table has {len(payload)} bytes, not a multiple of {layout.size}"
        )
    return tuple(
        row_type._from_values(tuple(values)) for values in layout.iter_unpack(payload)
    )


__all__ = [
    "EAGER_KERNEL_ABI",
    "EAGER_OUTPUT_FACTOR_COUPLING_IMAG",
    "EAGER_OUTPUT_FACTOR_COUPLING_REAL",
    "EAGER_OUTPUT_FACTOR_NONE",
    "EAGER_PLAN_ABI",
    "EAGER_RUNTIME_CAPABILITY",
    "EAGER_SELECTOR_DOMAINS_ABI",
    "MISSING_U32",
    "EagerAttachmentRow",
    "EagerClosureRow",
    "EagerCouplingRow",
    "EagerFinalizationRow",
    "EagerInvocationRow",
    "EagerSelectorDomainIdRow",
    "EagerSelectorDomainRow",
    "EagerSelectorGroupRow",
    "pack_rows",
    "unpack_rows",
]
