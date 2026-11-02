# SPDX-License-Identifier: 0BSD
"""Private eager-plan contracts and exact-kernel loading utilities."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Protocol, TypeVar, cast

from pyamplicol._internal.versions import (
    EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
    PROCESS_ARTIFACT_SCHEMA_VERSION,
)
from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.artifacts.manifest import ArtifactManifest, PayloadRecord
from pyamplicol.artifacts.security import confined_path, normalize_relative_path
from pyamplicol.generation.eager_tables import (
    EAGER_PLAN_ABI,
    EAGER_RUNTIME_CAPABILITY,
    EagerAttachmentRow,
    EagerClosureRow,
    EagerCouplingRow,
    EagerFinalizationRow,
    EagerInvocationRow,
    unpack_rows,
)
from pyamplicol.models.prepared import PreparedKernelRecord
from pyamplicol.runtime._evaluator_payloads import ExactEvaluatorPayloadResolver
from pyamplicol.runtime.symbolica_exact import (
    _ComplexDecimal,
    _decimal,
    _ExactEvaluator,
    _ExactExpressionEvaluator,
)

_ZERO = Decimal(0)
_EAGER_RUNTIME_KIND = "pyamplicol-runtime-eager-execution"
_SUPPORTED_PREPARED_BACKENDS = frozenset(("jit", "asm", "cpp"))
_PREPARED_CATALOG_ABI = "pyamplicol-prepared-kernel-catalog-v1"


@dataclass(frozen=True, slots=True)
class _ExactCouplingRow:
    real_parameter_id: int
    imag_parameter_id: int
    constant: _ComplexDecimal


@dataclass(frozen=True, slots=True)
class _ExactInvocationRow:
    kernel_id: int
    left_value_slot_id: int
    right_value_slot_id: int
    left_momentum_slot_id: int
    right_momentum_slot_id: int
    coupling_slot_id: int
    output_factor_source: int
    attachment_start: int
    attachment_count: int
    selector_domain_id: int | None = None


@dataclass(frozen=True, slots=True)
class _ExactComplexProduct:
    numerators: tuple[_ComplexDecimal, ...]
    denominator: _ComplexDecimal | None = None


_ExactFactor = _ComplexDecimal | _ExactComplexProduct


@dataclass(frozen=True, slots=True)
class _ExactAttachmentRow:
    result_current_id: int
    factor: _ExactFactor
    selector_domain_id: int | None = None


@dataclass(frozen=True, slots=True)
class _ExactFinalizationRow:
    kernel_id: int
    current_id: int
    unpropagated_value_slot_id: int
    propagated_value_slot_id: int
    momentum_slot_id: int
    unpropagated_selector_domain_id: int | None = None
    propagated_selector_domain_id: int | None = None


@dataclass(frozen=True, slots=True)
class _ExactClosureRow:
    kernel_id: int
    left_value_slot_id: int
    right_value_slot_id: int
    amplitude_index: int
    coupling_slot_id: int
    output_factor_source: int
    factor: _ExactFactor
    direct_coefficients: tuple[_ComplexDecimal, ...] | None
    coherent_group_id: int | None = None
    selector_domain_id: int | None = None


class _KernelEvaluator(Protocol):
    input_len: int

    def evaluate(
        self,
        values: Sequence[_ComplexDecimal],
        precision: int,
    ) -> Sequence[_ComplexDecimal]: ...


_KernelCallable = Callable[[Sequence[_ComplexDecimal], int], Sequence[_ComplexDecimal]]
_KernelLoader = Callable[
    [PreparedKernelRecord, Path], _KernelEvaluator | _KernelCallable
]


def _complex_zero() -> _ComplexDecimal:
    return (_ZERO, _ZERO)


def _complex_add(left: _ComplexDecimal, right: _ComplexDecimal) -> _ComplexDecimal:
    return left[0] + right[0], left[1] + right[1]


def _complex_mul(left: _ComplexDecimal, right: _ComplexDecimal) -> _ComplexDecimal:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _complex_div(left: _ComplexDecimal, right: _ComplexDecimal) -> _ComplexDecimal:
    denominator = right[0] * right[0] + right[1] * right[1]
    if denominator == _ZERO:
        raise ArtifactError("exact eager factor has a zero denominator")
    return (
        (left[0] * right[0] + left[1] * right[1]) / denominator,
        (left[1] * right[0] - left[0] * right[1]) / denominator,
    )


def _resolve_exact_factor(factor: _ExactFactor) -> _ComplexDecimal:
    if not isinstance(factor, _ExactComplexProduct):
        return factor
    result = (Decimal(1), _ZERO)
    for numerator in factor.numerators:
        result = _complex_mul(result, numerator)
    if factor.denominator is not None:
        result = _complex_div(result, factor.denominator)
    return result


def _complex_pair(real: object, imaginary: object, context: str) -> _ComplexDecimal:
    return _decimal(real, f"{context} real part"), _decimal(
        imaginary, f"{context} imaginary part"
    )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{context} must be an object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} must be an array")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtifactError(f"{context} must be an integer >= {minimum}")
    return value


def _read_json(path: Path, context: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"could not load {context}: {exc}") from exc
    return _mapping(value, context)


def _joined_payload_path(prefix: str, member: str) -> str:
    normalized_prefix = normalize_relative_path(prefix)
    normalized_member = normalize_relative_path(member)
    return normalize_relative_path(
        (PurePosixPath(normalized_prefix) / normalized_member).as_posix()
    )


@dataclass(frozen=True, slots=True)
class _PayloadIndex:
    records: Mapping[str, PayloadRecord]

    @classmethod
    def from_manifest(cls, manifest: ArtifactManifest) -> _PayloadIndex:
        return cls({record.path: record for record in manifest.payloads})

    def require(
        self,
        path: str,
        *,
        role: str,
        process_id: str | None,
    ) -> PayloadRecord:
        normalized = normalize_relative_path(path)
        record = self.records.get(normalized)
        if record is None:
            raise ArtifactError(
                f"eager metadata references undeclared payload {normalized}"
            )
        if record.role != role or record.process_id != process_id:
            raise ArtifactError(
                f"eager payload {normalized} has role/process "
                f"{record.role!r}/{record.process_id!r}, expected "
                f"{role!r}/{process_id!r}"
            )
        return record


@dataclass(frozen=True, slots=True)
class _ComponentSlot:
    slot_id: int
    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start

    def read(self, values: Sequence[_ComplexDecimal]) -> tuple[_ComplexDecimal, ...]:
        if self.stop > len(values):
            raise ArtifactError(f"eager slot {self.slot_id} is outside its storage")
        return tuple(values[self.start : self.stop])

    def write(
        self,
        values: list[_ComplexDecimal],
        entries: Sequence[_ComplexDecimal],
        context: str,
    ) -> None:
        if len(entries) != self.width or self.stop > len(values):
            raise EvaluationError(
                f"{context} produced {len(entries)} components, expected {self.width}"
            )
        values[self.start : self.stop] = entries


def _component_slots(
    records: object,
    *,
    id_field: str,
    start_field: str,
    stop_field: str,
    dimension_field: str | None,
    component_count: int,
    context: str,
) -> tuple[_ComponentSlot, ...]:
    parsed: list[_ComponentSlot] = []
    for index, raw in enumerate(_sequence(records, context)):
        record = _mapping(raw, f"{context}[{index}]")
        slot_id = _integer(record.get(id_field), f"{context}[{index}].{id_field}")
        start = _integer(record.get(start_field), f"{context}[{index}].{start_field}")
        stop = _integer(record.get(stop_field), f"{context}[{index}].{stop_field}")
        if stop <= start:
            raise ArtifactError(f"{context}[{index}] has an empty/inverted range")
        if dimension_field is not None:
            dimension = _integer(
                record.get(dimension_field),
                f"{context}[{index}].{dimension_field}",
                minimum=1,
            )
            if stop - start != dimension:
                raise ArtifactError(f"{context}[{index}] has inconsistent dimension")
        parsed.append(_ComponentSlot(slot_id, start, stop))
    parsed.sort(key=lambda slot: slot.slot_id)
    cursor = 0
    for expected_id, slot in enumerate(parsed):
        if slot.slot_id != expected_id:
            raise ArtifactError(f"{context} IDs must be contiguous from zero")
        if slot.start != cursor:
            raise ArtifactError(f"{context} component ranges must be contiguous")
        cursor = slot.stop
    if cursor != component_count:
        raise ArtifactError(
            f"{context} spans {cursor} components, expected {component_count}"
        )
    return tuple(parsed)


@dataclass(slots=True)
class _CallableEvaluator:
    input_len: int
    function: _KernelCallable

    def evaluate(
        self,
        values: Sequence[_ComplexDecimal],
        precision: int,
    ) -> Sequence[_ComplexDecimal]:
        return self.function(values, precision)


def _default_kernel_loader(
    record: PreparedKernelRecord,
    payload_root: Path,
) -> _KernelEvaluator:
    if record.contract_kind == "model-parameter":
        return _ExactExpressionEvaluator(
            record.exact_expressions,
            tuple(str(item["symbol"]) for item in record.input_contracts),
        )
    return _ExactEvaluator.load(
        {
            "evaluator_state_path": record.exact_evaluator_state_path,
            "input_len": record.input_arity,
        },
        payload_root,
    )


def _artifact_kernel_loader(
    resolver: ExactEvaluatorPayloadResolver,
    payload_root_name: str,
) -> _KernelLoader:
    def load(
        record: PreparedKernelRecord,
        payload_root: Path,
    ) -> _KernelEvaluator:
        if record.contract_kind == "model-parameter":
            return _ExactExpressionEvaluator(
                record.exact_expressions,
                tuple(str(item["symbol"]) for item in record.input_contracts),
            )
        return _ExactEvaluator.load(
            {
                "evaluator_state_path": record.exact_evaluator_state_path,
                "input_len": record.input_arity,
            },
            payload_root,
            state_loader=lambda relative: resolver.read_exact_state(
                _joined_payload_path(payload_root_name, relative),
                process_id=None,
            ),
        )

    return load


@dataclass(slots=True)
class _LazyExactKernel:
    record: PreparedKernelRecord
    payload_root: Path
    loader: _KernelLoader
    _evaluator: _KernelEvaluator | None = field(default=None, init=False, repr=False)

    def evaluate(
        self,
        values: Sequence[_ComplexDecimal],
        precision: int,
    ) -> tuple[_ComplexDecimal, ...]:
        if len(values) != self.record.input_arity:
            raise EvaluationError(
                f"eager kernel {self.record.kernel_id} received {len(values)} "
                f"inputs, expected {self.record.input_arity}"
            )
        evaluator = self._load()
        try:
            raw_outputs = evaluator.evaluate(values, precision)
        except (ArtifactError, CompatibilityError, EvaluationError):
            raise
        except Exception as exc:
            raise EvaluationError(
                f"exact eager kernel {self.record.kernel_id} failed: {exc}"
            ) from exc
        if isinstance(raw_outputs, (str, bytes, bytearray)) or not isinstance(
            raw_outputs, Sequence
        ):
            raise EvaluationError(
                f"eager kernel {self.record.kernel_id} returned a non-array result"
            )
        outputs_list = []
        for output_index, value in enumerate(raw_outputs):
            if (
                isinstance(value, (str, bytes, bytearray))
                or not isinstance(value, Sequence)
                or len(value) != 2
            ):
                raise EvaluationError(
                    f"eager kernel {self.record.kernel_id} output {output_index} "
                    "is not a complex pair"
                )
            outputs_list.append(
                _complex_pair(
                    value[0],
                    value[1],
                    f"kernel {self.record.kernel_id} output {output_index}",
                )
            )
        outputs = tuple(outputs_list)
        if len(outputs) != self.record.output_arity:
            raise EvaluationError(
                f"eager kernel {self.record.kernel_id} produced {len(outputs)} "
                f"outputs, expected {self.record.output_arity}"
            )
        return outputs

    def _load(self) -> _KernelEvaluator:
        if self._evaluator is not None:
            return self._evaluator
        loaded = self.loader(self.record, self.payload_root)
        if callable(loaded) and not hasattr(loaded, "evaluate"):
            loaded = _CallableEvaluator(self.record.input_arity, loaded)
        if not hasattr(loaded, "evaluate") or not hasattr(loaded, "input_len"):
            raise ArtifactError(
                "exact eager kernel loader returned an invalid evaluator for "
                f"kernel {self.record.kernel_id}"
            )
        evaluator = cast(_KernelEvaluator, loaded)
        if (
            isinstance(evaluator.input_len, bool)
            or evaluator.input_len != self.record.input_arity
        ):
            raise ArtifactError(
                f"exact evaluator for eager kernel {self.record.kernel_id} has input "
                f"arity {evaluator.input_len}, expected {self.record.input_arity}"
            )
        self._evaluator = evaluator
        return evaluator


_RowT = TypeVar(
    "_RowT",
    EagerInvocationRow,
    EagerAttachmentRow,
    EagerCouplingRow,
    EagerFinalizationRow,
    EagerClosureRow,
)


def _load_table(
    process_root: Path,
    process_prefix: str,
    record: Mapping[str, object],
    row_type: type[_RowT],
    payloads: _PayloadIndex,
    process_id: str,
    context: str,
) -> tuple[_RowT, ...]:
    raw_path = record.get("path")
    if not isinstance(raw_path, str):
        raise ArtifactError(f"{context} path must be a string")
    path = normalize_relative_path(raw_path)
    count = _integer(record.get("count"), f"{context} count")
    row_size = _integer(record.get("row_size"), f"{context} row size", minimum=1)
    expected_size = row_type._STRUCT.size
    if row_size != expected_size:
        raise ArtifactError(
            f"{context} row size is {row_size}, expected {expected_size}"
        )
    global_path = _joined_payload_path(process_prefix, path)
    payloads.require(global_path, role="evaluator-state", process_id=process_id)
    table_path = confined_path(process_root, path)
    try:
        content = table_path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"could not read {context}: {exc}") from exc
    if len(content) != count * expected_size:
        raise ArtifactError(
            f"{context} declares {count} rows but contains {len(content)} bytes"
        )
    try:
        return unpack_rows(content, row_type)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"could not decode {context}: {exc}") from exc


def _direct_coefficients(
    root: Mapping[str, object], index: int
) -> tuple[_ComplexDecimal, ...]:
    contraction = _mapping(
        root.get("contraction_ir"), f"amplitude root {index} contraction_ir"
    )
    coefficients = []
    for coefficient_index, raw in enumerate(
        _sequence(
            contraction.get("coefficients"),
            f"amplitude root {index} contraction coefficients",
        )
    ):
        pair = _sequence(raw, f"direct coefficient {coefficient_index}")
        if len(pair) != 2:
            raise ArtifactError("direct contraction coefficient must be complex pair")
        coefficients.append(
            _complex_pair(pair[0], pair[1], "direct contraction coefficient")
        )
    if not coefficients:
        raise ArtifactError("direct eager closure has no contraction coefficients")
    return tuple(coefficients)


def _validate_execution_header(execution: Mapping[str, object]) -> None:
    if execution.get("schema_version") != PROCESS_ARTIFACT_SCHEMA_VERSION:
        raise CompatibilityError(
            f"unsupported eager process schema {execution.get('schema_version')!r}"
        )
    if execution.get("kind") != _EAGER_RUNTIME_KIND:
        raise CompatibilityError(
            f"unsupported exact eager execution kind {execution.get('kind')!r}"
        )
    if execution.get("eager_plan_abi") != EAGER_PLAN_ABI:
        raise CompatibilityError(
            f"unsupported eager plan ABI {execution.get('eager_plan_abi')!r}"
        )
    plan = _mapping(execution.get("plan"), "plan")
    if (
        plan.get("kind") != _EAGER_RUNTIME_KIND
        or plan.get("eager_plan_abi") != EAGER_PLAN_ABI
    ):
        raise CompatibilityError("outer and inner eager plan contracts do not match")
    if plan.get("process_key") != execution.get("key"):
        raise ArtifactError("eager plan process key does not match execution key")
    expected_capabilities = [EAGER_RUNTIME_CAPABILITY]
    if execution.get("lc_topology_replay") is not None:
        expected_capabilities.append(EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY)
    expected_capabilities.sort()
    if execution.get("required_runtime_capabilities") != expected_capabilities:
        raise CompatibilityError("unsupported eager runtime capability contract")
    if plan.get("required_runtime_capabilities") != expected_capabilities:
        raise CompatibilityError("unsupported eager plan capability contract")
