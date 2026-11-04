# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import marshal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..models.base import Model
from .dag_types import GenericDAG

LC_SECTOR_SELECTOR_PARAMETER = "runtime.lc_sector_id"
EXPRESSION_SCHEMA_CONTRACT_VERSION = 2
_REQUIRED_RUNTIME_SCHEMA_KEYS = frozenset(
    {
        "amplitude_stage",
        "current_storage",
        "momentum_slots",
        "parameter_layout",
        "stages",
        "value_storage",
    }
)
@dataclass(frozen=True, slots=True)
class RuntimeExpressionSchema:
    """Canonical, dependency-neutral stage compiler input schema."""

    canonical_json: str
    contract_version: int = EXPRESSION_SCHEMA_CONTRACT_VERSION
    _payload_blob: bytes = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if self.contract_version != EXPRESSION_SCHEMA_CONTRACT_VERSION:
            raise ValueError(
                "unsupported runtime expression-schema contract version "
                f"{self.contract_version}"
            )
        try:
            payload = json.loads(self.canonical_json)
        except json.JSONDecodeError as exc:
            raise ValueError("runtime expression schema is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("runtime expression schema root must be an object")
        missing = sorted(_REQUIRED_RUNTIME_SCHEMA_KEYS - payload.keys())
        if missing:
            raise ValueError(
                "runtime expression schema is missing: " + ", ".join(missing)
            )
        # The blob is an immutable, process-local cache of already validated JSON.
        # It is never persisted or loaded from untrusted input.
        object.__setattr__(self, "_payload_blob", marshal.dumps(payload))

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object] | RuntimeExpressionSchema,
    ) -> RuntimeExpressionSchema:
        if isinstance(payload, cls):
            return payload
        try:
            canonical = json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "runtime expression schema must contain canonical JSON values"
            ) from exc
        return cls(canonical_json=canonical)

    def to_mapping(self) -> dict[str, object]:
        payload = marshal.loads(self._payload_blob)
        if not isinstance(payload, dict):  # pragma: no cover - internal invariant
            raise RuntimeError("runtime expression schema cache is not an object")
        return {str(key): value for key, value in payload.items()}

    def __reduce__(self) -> tuple[object, tuple[str, int]]:
        return type(self), (self.canonical_json, self.contract_version)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class StageCompilationInput:
    dag: GenericDAG
    model: Model
    runtime_schema: RuntimeExpressionSchema

    def __post_init__(self) -> None:
        if not isinstance(self.dag, GenericDAG):
            raise TypeError("stage compilation input requires a GenericDAG")
        if not isinstance(self.model, Model):
            raise TypeError("stage compilation input requires a Model")
        if not isinstance(self.runtime_schema, RuntimeExpressionSchema):
            raise TypeError(
                "stage compilation input requires a RuntimeExpressionSchema"
            )


def runtime_coupling_parameter_names(
    vertex_kind: int,
    vertex_particles: Sequence[int],
    coupling: Sequence[object],
    *,
    model: Model | None = None,
) -> list[str | None]:
    runtime_names = getattr(model, "runtime_parameter_names_for_vertex", None)
    if callable(runtime_names):
        return [str(name) for name in runtime_names(int(vertex_kind))]
    lowering_rule = getattr(model, "vertex_lowering_rule", None)
    if callable(lowering_rule):
        rule = lowering_rule(int(vertex_kind))
        if rule.coupling_mode in {"fixed", "none"}:
            return [None for _value in coupling]
    particles = "_".join(str(int(pdg)) for pdg in vertex_particles)
    base = f"coupling.{int(vertex_kind)}.{particles}"
    names: list[str | None] = []
    for component, value in enumerate(coupling):
        if not isinstance(value, str | int | float):
            raise TypeError(
                f"runtime coupling components must be numeric, got {value!r}"
            )
        numeric_value = float(value)
        if component == 1 and numeric_value == -10.0:
            names.append(None)
        else:
            names.append(f"{base}.component_{component}")
    return names


__all__ = [
    "EXPRESSION_SCHEMA_CONTRACT_VERSION",
    "LC_SECTOR_SELECTOR_PARAMETER",
    "RuntimeExpressionSchema",
    "StageCompilationInput",
    "runtime_coupling_parameter_names",
]
