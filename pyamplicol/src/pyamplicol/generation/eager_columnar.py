# SPDX-License-Identifier: 0BSD
"""Columnar Python-to-Rust input contract for eager DAG lowering.

This module deliberately stops before executable eager-table construction.  It
turns the immutable, proof-carrying Python DAG into deterministic primitive
columns that a later Rust lowerer can consume without first constructing the
expanded runtime schema.  The production generation path does not use this
contract yet.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from ..color import build_color_contraction_plan
from ..color.contraction_types import (
    ColorContractionEntry,
    ColorContractionTemplateEntry,
)
from ..models.base import Model
from .contracts import runtime_coupling_parameter_names
from .dag_types import GenericDAG
from .eager_lowering import EagerKernelResolver, EagerResolvedKernel
from .helicity_replay import (
    HELICITY_RECURRENCE_CONTRACT_VERSION,
    HELICITY_RECURRENCE_PROOF_ALGORITHM,
)
from .runtime_amplitudes import (
    _amplitude_groups,
    _color_topology_replay_amplitudes,
    _has_multiple_lc_root_sectors,
    _root_all_sector_weight,
)

EAGER_LOWERING_INPUT_ABI = "pyamplicol-eager-lowering-input-v1"
EAGER_LOWERING_INPUT_SCHEMA_SHA256 = (
    "59d5a3193ca25af6224c296297425782c394a34e352be622f47045e90d2bce29"
)
MISSING_U32 = (1 << 32) - 1
MISSING_I32 = -(1 << 31)

FACTOR_EXACT_SOURCE_BINARY64 = 0
FACTOR_EXACT_SOURCE_CANONICAL_IR = 1

_U8 = np.dtype("u1")
_U32 = np.dtype("<u4")
_U64 = np.dtype("<u8")
_I32 = np.dtype("<i4")
_F64 = np.dtype("<f8")
_ALLOWED_DTYPES = frozenset(dtype.str for dtype in (_U8, _U32, _U64, _I32, _F64))

_SEMANTIC_LIMITATIONS = (
    "Every numeric factor carries both its f64 payload and a canonical exact-IR "
    "reference. Values whose source contract is intrinsically f64 use an exact "
    "IEEE-754 binary64 literal, preserving signed zero and every payload bit.",
    "Rational colour-contraction weights retain their pre-f64 exact source. "
    "Other GenericDAG proof weights, contraction coefficients, and "
    "EagerKernelResolver normalization factors do not expose their pre-f64 "
    "Symbolica expressions. Their exact-IR entries therefore certify the f64 "
    "value from which current exact and double-double execution deliberately "
    "starts, not an unavailable algebraic source expression.",
    "Prepared-kernel exact expressions and evaluator states remain owned by the "
    "prepared model pack and are selected by kernel ID. The current resolver "
    "does not expose independent exact references for binding normalization "
    "factors; extending that resolver is the remaining provenance blocker.",
    "EagerKernelResolver also does not expose the prepared model-parameter "
    "derivation kernel ID. Parameter definitions and defaults are represented, "
    "but integration must supply that pack-owned kernel binding separately.",
)


class EagerColumnarInputError(ValueError):
    """The proven Python input cannot be represented by the v1 wire contract."""


@dataclass(frozen=True, slots=True)
class EagerColumn:
    """One immutable primitive column."""

    name: str
    values: np.ndarray[Any, Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise EagerColumnarInputError("column name must not be empty")
        values = self.values
        if not isinstance(values, np.ndarray):
            raise TypeError(f"column {self.name!r} must be a NumPy array")
        if values.ndim < 1:
            raise EagerColumnarInputError(
                f"column {self.name!r} must have a row dimension"
            )
        if not values.flags.c_contiguous:
            raise EagerColumnarInputError(f"column {self.name!r} must be C-contiguous")
        if values.dtype.str not in _ALLOWED_DTYPES:
            raise EagerColumnarInputError(
                f"column {self.name!r} has unsupported dtype {values.dtype.str!r}"
            )
        if values.flags.writeable:
            raise EagerColumnarInputError(f"column {self.name!r} must be read-only")


@dataclass(frozen=True, slots=True)
class EagerColumnarTable:
    """A deterministic structure-of-arrays table."""

    name: str
    row_count: int
    columns: tuple[EagerColumn, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise EagerColumnarInputError("table name must not be empty")
        _checked_u64(self.row_count, f"{self.name}.row_count")
        names = tuple(column.name for column in self.columns)
        if len(set(names)) != len(names):
            raise EagerColumnarInputError(
                f"table {self.name!r} contains duplicate column names"
            )
        for column in self.columns:
            if len(column.values) != self.row_count:
                raise EagerColumnarInputError(
                    f"table {self.name!r} column {column.name!r} has "
                    f"{len(column.values)} rows, expected {self.row_count}"
                )

    def column(self, name: str) -> np.ndarray[Any, Any]:
        for column in self.columns:
            if column.name == name:
                return column.values
        raise KeyError(f"table {self.name!r} has no column {name!r}")


@dataclass(frozen=True, slots=True)
class EagerLoweringInputV1:
    """Complete immutable input consumed by the future Rust eager lowerer."""

    abi: str
    process_key: str
    model_name: str
    string_catalog: tuple[str, ...]
    canonical_ir_catalog: tuple[str, ...]
    tables: tuple[EagerColumnarTable, ...]
    semantic_limitations: tuple[str, ...] = _SEMANTIC_LIMITATIONS

    def __post_init__(self) -> None:
        if self.abi != EAGER_LOWERING_INPUT_ABI:
            raise EagerColumnarInputError(
                f"unsupported eager lowering input ABI {self.abi!r}"
            )
        if not self.process_key or not self.model_name:
            raise EagerColumnarInputError(
                "process key and model name must not be empty"
            )
        names = tuple(table.name for table in self.tables)
        if names != tuple(sorted(names)):
            raise EagerColumnarInputError(
                "columnar table names must be sorted deterministically"
            )
        if len(set(names)) != len(names):
            raise EagerColumnarInputError("columnar table names must be unique")
        _validate_catalog(self.string_catalog, "string")
        _validate_catalog(self.canonical_ir_catalog, "canonical IR")
        if not self.semantic_limitations:
            raise EagerColumnarInputError(
                "the v1 exact-factor limitation must be reported explicitly"
            )
        self._validate_references()

    def table(self, name: str) -> EagerColumnarTable:
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(f"eager lowering input has no table {name!r}")

    @property
    def digest(self) -> str:
        """Content digest independent of Python object identity and hash order."""

        digest = hashlib.sha256()
        for value in (self.abi, self.process_key, self.model_name):
            _hash_text(digest, value)
        for catalog in (self.string_catalog, self.canonical_ir_catalog):
            digest.update(len(catalog).to_bytes(8, "little"))
            for value in catalog:
                _hash_text(digest, value)
        digest.update(len(self.tables).to_bytes(8, "little"))
        for table in self.tables:
            _hash_text(digest, table.name)
            digest.update(table.row_count.to_bytes(8, "little"))
            digest.update(len(table.columns).to_bytes(4, "little"))
            for column in table.columns:
                _hash_text(digest, column.name)
                _hash_text(digest, column.values.dtype.str)
                digest.update(len(column.values.shape).to_bytes(1, "little"))
                for size in column.values.shape:
                    digest.update(int(size).to_bytes(8, "little"))
                digest.update(column.values.tobytes(order="C"))
        for limitation in self.semantic_limitations:
            _hash_text(digest, limitation)
        return digest.hexdigest()

    @property
    def schema_sha256(self) -> str:
        """Digest the fixed table/column ABI independently of row counts."""

        digest = hashlib.sha256()
        digest.update(len(self.tables).to_bytes(8, "little"))
        for table in self.tables:
            _hash_text(digest, table.name)
            digest.update(len(table.columns).to_bytes(4, "little"))
            for column in table.columns:
                _hash_text(digest, column.name)
                _hash_text(digest, column.values.dtype.str)
                digest.update(len(column.values.shape).to_bytes(1, "little"))
                for size in column.values.shape[1:]:
                    digest.update(int(size).to_bytes(8, "little"))
        return digest.hexdigest()

    def _validate_references(self) -> None:
        current_count = self.table("currents").row_count
        interaction_count = self.table("interactions").row_count
        factor_count = self.table("exact_factors").row_count
        string_count = len(self.string_catalog)
        ir_count = len(self.canonical_ir_catalog)

        _validate_dense_ids(self.table("currents"), "id")
        _validate_dense_ids(self.table("interactions"), "id")
        _validate_dense_ids(self.table("roots"), "id")
        _validate_bounded_columns(
            self.table("interactions"),
            ("left_current_id", "right_current_id", "result_current_id"),
            current_count,
        )
        _validate_bounded_columns(
            self.table("roots"),
            ("left_current_id", "right_current_id"),
            current_count,
        )
        _validate_bounded_columns(self.table("sources"), ("current_id",), current_count)
        _validate_bounded_columns(
            self.table("interaction_group_members"),
            ("interaction_id",),
            interaction_count,
        )
        for table in self.tables:
            factor_columns = tuple(
                column.name
                for column in table.columns
                if column.name.endswith("factor_id")
            )
            if factor_columns:
                _validate_bounded_columns(table, factor_columns, factor_count)
        _validate_bounded_columns(
            self.table("exact_factors"), ("exact_ir_id",), ir_count
        )
        _validate_optional_bounded_columns(
            self.table("exact_factors"), ("source_ir_id",), ir_count
        )
        _validate_bounded_columns(
            self.table("exact_factors"), ("canonical_string_id",), string_count
        )
        exact_sources = set(
            int(value) for value in self.table("exact_factors").column("exact_source")
        )
        if not exact_sources <= {
            FACTOR_EXACT_SOURCE_BINARY64,
            FACTOR_EXACT_SOURCE_CANONICAL_IR,
        }:
            raise EagerColumnarInputError(
                "exact-factor catalog contains an unsupported provenance code"
            )
        for table in self.tables:
            string_columns = tuple(
                column.name
                for column in table.columns
                if column.name.endswith("string_id")
            )
            if string_columns:
                _validate_optional_bounded_columns(table, string_columns, string_count)
        _validate_optional_bounded_columns(
            self.table("currents"), ("propagator_ir_id",), ir_count
        )
        _validate_bounded_columns(self.table("roots"), ("contraction_ir_id",), ir_count)
        _validate_bounded_columns(
            self.table("contraction_coefficients"),
            ("contraction_ir_id",),
            ir_count,
        )
        _validate_bounded_columns(
            self.table("sources"), ("source_ir_id", "crossing_ir_id"), ir_count
        )


@dataclass(frozen=True, slots=True)
class _ModelParameterSpec:
    name: str
    kind: str
    default: float
    runtime_name: str | None = None
    complex_component: int = -1
    parameter_type: str | None = None
    pdg: int | None = None
    vertex_kind: int | None = None
    vertex_particles: tuple[int, int, int] | None = None
    coupling_component: int | None = None
    derived: bool = False
    complex_domain: str | None = None
    definition: str | None = None


class _StringCatalog:
    def __init__(self) -> None:
        self.values: list[str] = []
        self.ids: dict[str, int] = {}

    def add(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError(f"catalog value must be a string, got {value!r}")
        existing = self.ids.get(value)
        if existing is not None:
            return existing
        result = _checked_u32(len(self.values), "catalog ID")
        self.values.append(value)
        self.ids[value] = result
        return result

    def optional(self, value: str | None) -> int:
        return MISSING_U32 if value is None else self.add(value)


class _SequenceCatalog:
    def __init__(self, *, signed: bool) -> None:
        self.signed = signed
        self.values: list[tuple[int, ...]] = []
        self.ids: dict[tuple[int, ...], int] = {}

    def add(self, values: Iterable[int]) -> int:
        item = tuple(int(value) for value in values)
        checker = _checked_i32 if self.signed else _checked_u32
        for index, value in enumerate(item):
            checker(value, f"sequence component {index}")
        existing = self.ids.get(item)
        if existing is not None:
            return existing
        result = _checked_u32(len(self.values), "sequence ID")
        self.values.append(item)
        self.ids[item] = result
        return result

    def tables(self, prefix: str) -> tuple[EagerColumnarTable, EagerColumnarTable]:
        ranges = _allocate(len(self.values), start=_U64, count=_U64)
        total = sum(len(value) for value in self.values)
        dtype = _I32 if self.signed else _U32
        flat = _allocate(total, value=dtype)
        cursor = 0
        for index, values in enumerate(self.values):
            ranges["start"][index] = cursor
            ranges["count"][index] = len(values)
            stop = cursor + len(values)
            flat["value"][cursor:stop] = values
            cursor = stop
        return (
            _freeze_table(f"{prefix}_sequence_ranges", ranges),
            _freeze_table(f"{prefix}_sequence_values", flat),
        )


class _BitsetCatalog:
    def __init__(self) -> None:
        self.values: list[int] = []
        self.ids: dict[int, int] = {}

    def add(self, value: int, context: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{context} must be an integer")
        if value < 0:
            raise EagerColumnarInputError(f"{context} must be nonnegative")
        existing = self.ids.get(value)
        if existing is not None:
            return existing
        result = _checked_u32(len(self.values), "bitset ID")
        self.values.append(value)
        self.ids[value] = result
        return result

    def tables(self) -> tuple[EagerColumnarTable, EagerColumnarTable]:
        ranges = _allocate(len(self.values), start=_U64, count=_U64, bit_count=_U64)
        total = sum((value.bit_length() + 63) // 64 for value in self.values)
        words = _allocate(total, value=_U64)
        cursor = 0
        for index, value in enumerate(self.values):
            count = (value.bit_length() + 63) // 64
            ranges["start"][index] = cursor
            ranges["count"][index] = count
            ranges["bit_count"][index] = value.bit_count()
            for word_index in range(count):
                words["value"][cursor] = (value >> (64 * word_index)) & ((1 << 64) - 1)
                cursor += 1
        return (
            _freeze_table("bitset_ranges", ranges),
            _freeze_table("bitset_words", words),
        )


class _FactorCatalog:
    def __init__(self, canonical_ir: _CanonicalIRCatalog) -> None:
        self.canonical_ir = canonical_ir
        self.values: list[tuple[tuple[float, float], int, int, int]] = []
        self.ids: dict[tuple[int, int, int, int, int], int] = {}
        self.binary_ids: dict[tuple[int, int, int], int] = {}

    def add(
        self,
        value: Sequence[object],
        context: str,
        *,
        exact_ir: object | None = None,
        source_ir_id: int = MISSING_U32,
    ) -> int:
        if len(value) != 2:
            raise EagerColumnarInputError(f"{context} must be a complex pair")
        pair = (float(value[0]), float(value[1]))
        if not all(math.isfinite(component) for component in pair):
            raise EagerColumnarInputError(f"{context} must be finite")
        if source_ir_id != MISSING_U32:
            _checked_u32(source_ir_id, f"{context} source IR ID")
        bits = (_f64_bits(pair[0]), _f64_bits(pair[1]))
        if exact_ir is None:
            binary_key = (*bits, source_ir_id)
            existing = self.binary_ids.get(binary_key)
            if existing is not None:
                return existing
        exact_source = (
            FACTOR_EXACT_SOURCE_BINARY64
            if exact_ir is None
            else FACTOR_EXACT_SOURCE_CANONICAL_IR
        )
        exact_payload = (
            _binary64_exact_ir(pair)
            if exact_ir is None
            else {
                "abi": "pyamplicol-exact-factor-v1",
                "kind": "canonical-source-ir",
                "source": exact_ir,
                "f64_fallback": _binary64_exact_ir(pair),
            }
        )
        exact_ir_id = self.canonical_ir.add(exact_payload, f"{context} exact IR")
        key = (*bits, exact_source, exact_ir_id, source_ir_id)
        existing = self.ids.get(key)
        if existing is not None:
            return existing
        result = _checked_u32(len(self.values), "exact-factor ID")
        self.values.append((pair, exact_source, exact_ir_id, source_ir_id))
        self.ids[key] = result
        if exact_ir is None:
            self.binary_ids[(*bits, source_ir_id)] = result
        return result

    def pair(self, factor_id: int) -> tuple[float, float]:
        return self.values[factor_id][0]

    def table(self, strings: _StringCatalog) -> EagerColumnarTable:
        columns = _allocate(
            len(self.values),
            real=_F64,
            imaginary=_F64,
            canonical_string_id=_U32,
            exact_source=_U8,
            exact_ir_id=_U32,
            source_ir_id=_U32,
        )
        for index, (
            (real, imaginary),
            exact_source,
            exact_ir_id,
            source_ir_id,
        ) in enumerate(self.values):
            columns["real"][index] = real
            columns["imaginary"][index] = imaginary
            columns["canonical_string_id"][index] = strings.add(
                f"complex-f64:{real.hex()}:{imaginary.hex()}"
            )
            columns["exact_source"][index] = exact_source
            columns["exact_ir_id"][index] = exact_ir_id
            columns["source_ir_id"][index] = source_ir_id
        return _freeze_table("exact_factors", columns)


class _CanonicalIRCatalog:
    def __init__(self) -> None:
        self.values: list[str] = []
        self.ids: dict[str, int] = {}

    def add(self, value: object, context: str) -> int:
        try:
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise EagerColumnarInputError(
                f"{context} cannot be represented canonically: {error}"
            ) from error
        existing = self.ids.get(encoded)
        if existing is not None:
            return existing
        result = _checked_u32(len(self.values), "canonical IR ID")
        self.values.append(encoded)
        self.ids[encoded] = result
        return result


class _Builder:
    def __init__(
        self,
        dag: GenericDAG,
        model: Model,
        resolver: EagerKernelResolver,
        process_id: str | None,
    ) -> None:
        if not isinstance(dag, GenericDAG):
            raise TypeError("eager columnar extraction requires a GenericDAG")
        if not isinstance(model, Model):
            raise TypeError("eager columnar extraction requires a Model")
        self.dag = dag
        self.model = model
        self.resolver = resolver
        self.process_key = process_id or dag.process.key
        self.strings = _StringCatalog()
        self.ir = _CanonicalIRCatalog()
        self.u32_sequences = _SequenceCatalog(signed=False)
        self.i32_sequences = _SequenceCatalog(signed=True)
        self.bitsets = _BitsetCatalog()
        self.factors = _FactorCatalog(self.ir)
        self.tables: list[EagerColumnarTable] = []
        self.coupling_specs: list[tuple[int, tuple[int, int, int], int, int]] = []
        self.coupling_ids: dict[
            tuple[int, tuple[int, int, int], tuple[float, float]], int
        ] = {}

        self._validate_dense_dag()
        self.group_ids, self.group_descriptors = _amplitude_groups(
            dag, dag.amplitude_roots
        )
        self.color_topology_replay = _color_topology_replay_amplitudes(
            dag, self.group_descriptors
        )

    def build(self) -> EagerLoweringInputV1:
        self._metadata()
        self._currents()
        self._sources()
        self._interactions()
        self._roots()
        self._interaction_groups()
        self._color_plan()
        self._topology_replay()
        self._color_topology_replay_amplitudes()
        self._helicity_proof()
        self._helicity_materialization()
        self._selectors_and_reductions()
        self._couplings()
        self._model_parameters()
        self.tables.append(self.factors.table(self.strings))
        self.tables.extend(self.bitsets.tables())
        self.tables.extend(self.u32_sequences.tables("u32"))
        self.tables.extend(self.i32_sequences.tables("i32"))
        return EagerLoweringInputV1(
            abi=EAGER_LOWERING_INPUT_ABI,
            process_key=self.process_key,
            model_name=self.model.name,
            string_catalog=tuple(self.strings.values),
            canonical_ir_catalog=tuple(self.ir.values),
            tables=tuple(sorted(self.tables, key=lambda table: table.name)),
        )

    def _validate_dense_dag(self) -> None:
        for context, records in (
            ("current", self.dag.currents),
            ("interaction", self.dag.interactions),
            ("amplitude root", self.dag.amplitude_roots),
        ):
            _checked_u32(len(records), f"{context} count")
            for expected, record in enumerate(records):
                _checked_u32(record.id, f"{context} id")
                if record.id != expected:
                    raise EagerColumnarInputError(
                        f"{context} ids must be dense; found {record.id} at {expected}"
                    )
        current_count = len(self.dag.currents)
        for interaction in self.dag.interactions:
            for field, current_id in (
                ("left", interaction.left_id),
                ("right", interaction.right_id),
                ("result", interaction.result_id),
            ):
                _require_index(current_id, current_count, f"interaction {field}")
        for root in self.dag.amplitude_roots:
            _require_index(root.left_id, current_count, "root left current")
            _require_index(root.right_id, current_count, "root right current")
        for source_id in self.dag.sources:
            _require_index(source_id, current_count, "source current")
            if not self.dag.currents[source_id].is_source:
                raise EagerColumnarInputError(
                    f"DAG source {source_id} is not marked as a source current"
                )

    def _metadata(self) -> None:
        columns = _allocate(
            1,
            process_ir_id=_U32,
            normalization_ir_id=_U32,
            model_name_string_id=_U32,
            process_key_string_id=_U32,
            helicity_coverage_string_id=_U32,
            color_coverage_string_id=_U32,
            color_accuracy_string_id=_U32,
            truncated=_U8,
            color_plan_truncated=_U8,
            trace_reflections_folded=_U8,
        )
        columns["process_ir_id"][0] = self.ir.add(
            self.dag.process.to_json_dict(), "process IR"
        )
        columns["normalization_ir_id"][0] = self.ir.add(
            dict(self.model.runtime_normalization_payload(self.dag)),
            "runtime normalization",
        )
        columns["model_name_string_id"][0] = self.strings.add(self.model.name)
        columns["process_key_string_id"][0] = self.strings.add(self.process_key)
        columns["helicity_coverage_string_id"][0] = self.strings.add(
            self.dag.helicity_coverage
        )
        columns["color_coverage_string_id"][0] = self.strings.add(
            self.dag.color_coverage
        )
        columns["color_accuracy_string_id"][0] = self.strings.add(
            self.dag.process.color_accuracy
        )
        columns["truncated"][0] = self.dag.truncated
        columns["color_plan_truncated"][0] = self.dag.color_plan.truncated
        columns["trace_reflections_folded"][0] = (
            self.dag.color_plan.trace_reflections_folded
        )
        self.tables.append(_freeze_table("metadata", columns))

        diagnostics = _allocate(
            len(self.dag.color_plan.diagnostics), message_string_id=_U32
        )
        for row, message in enumerate(self.dag.color_plan.diagnostics):
            diagnostics["message_string_id"][row] = self.strings.add(str(message))
        self.tables.append(_freeze_table("color_plan_diagnostics", diagnostics))

        selected_helicities = _allocate(
            len(self.dag.selected_source_helicities),
            external_label=_U32,
            helicity=_I32,
        )
        for index, (label, helicity) in enumerate(self.dag.selected_source_helicities):
            selected_helicities["external_label"][index] = _checked_u32(
                label, "selected-helicity external label"
            )
            selected_helicities["helicity"][index] = _checked_i32(
                helicity, "selected helicity"
            )
        self.tables.append(
            _freeze_table("selected_source_helicities", selected_helicities)
        )
        selected_colors = _allocate(
            len(self.dag.selected_color_sector_ids), sector_id=_U32
        )
        selected_colors["sector_id"][:] = self.dag.selected_color_sector_ids
        self.tables.append(_freeze_table("selected_color_sectors", selected_colors))

    def _currents(self) -> None:
        columns = _allocate(
            len(self.dag.currents),
            id=_U32,
            particle_id=_I32,
            dimension=_U32,
            is_source=_U8,
            source_leg_label=_U32,
            source_helicity=_I32,
            external_mask_bitset_id=_U32,
            external_labels_sequence_id=_U32,
            ordered_external_labels_sequence_id=_U32,
            helicity_ancestry_bitset_id=_U32,
            chirality=_I32,
            spin_state_sequence_id=_U32,
            flavour_flow_sequence_id=_U32,
            momentum_mask_bitset_id=_U32,
            coupling_order_start=_U64,
            coupling_order_count=_U64,
            quantum_flow_start=_U64,
            quantum_flow_count=_U64,
            color_accuracy_string_id=_U32,
            color_sector_id=_U32,
            color_line_groups_sequence_id=_U32,
            color_basis_keys_sequence_id=_U32,
            auxiliary_kind_string_id=_U32,
            propagator_ir_id=_U32,
            propagator_kernel_id=_U32,
        )
        coupling_order_count = sum(
            len(current.index.coupling_orders) for current in self.dag.currents
        )
        quantum_flow_count = sum(
            len(current.index.quantum_number_flow) for current in self.dag.currents
        )
        coupling_orders = _allocate(
            coupling_order_count, name_string_id=_U32, value=_I32
        )
        quantum_flows = _allocate(
            quantum_flow_count, name_string_id=_U32, expression_string_id=_U32
        )
        coupling_cursor = 0
        quantum_cursor = 0
        for row, current in enumerate(self.dag.currents):
            current_index = current.index
            columns["id"][row] = current.id
            columns["particle_id"][row] = _checked_i32(
                current_index.particle_id, "current particle id"
            )
            columns["dimension"][row] = _positive_u32(
                current.dimension, "current dimension"
            )
            columns["is_source"][row] = current.is_source
            columns["source_leg_label"][row] = (
                MISSING_U32
                if current.source_leg_label is None
                else _checked_u32(current.source_leg_label, "source leg label")
            )
            columns["source_helicity"][row] = (
                MISSING_I32
                if current.source_helicity is None
                else _checked_i32(current.source_helicity, "source helicity")
            )
            columns["external_mask_bitset_id"][row] = self.bitsets.add(
                current_index.external_mask, "current external mask"
            )
            columns["external_labels_sequence_id"][row] = self.u32_sequences.add(
                current_index.external_labels
            )
            columns["ordered_external_labels_sequence_id"][row] = (
                self.u32_sequences.add(current_index.ordered_external_labels)
            )
            columns["helicity_ancestry_bitset_id"][row] = self.bitsets.add(
                current_index.helicity_ancestry, "current helicity ancestry"
            )
            columns["chirality"][row] = _checked_i32(
                current_index.chirality, "current chirality"
            )
            spin_state = (
                current_index.spin_state
                if isinstance(current_index.spin_state, tuple)
                else (current_index.spin_state,)
            )
            columns["spin_state_sequence_id"][row] = self.i32_sequences.add(spin_state)
            columns["flavour_flow_sequence_id"][row] = self.i32_sequences.add(
                current_index.flavour_flow
            )
            columns["momentum_mask_bitset_id"][row] = self.bitsets.add(
                current_index.momentum_mask, "current momentum mask"
            )
            columns["coupling_order_start"][row] = coupling_cursor
            columns["coupling_order_count"][row] = len(current_index.coupling_orders)
            for name, value in current_index.coupling_orders:
                coupling_orders["name_string_id"][coupling_cursor] = self.strings.add(
                    str(name)
                )
                coupling_orders["value"][coupling_cursor] = _checked_i32(
                    value, "coupling order"
                )
                coupling_cursor += 1
            columns["quantum_flow_start"][row] = quantum_cursor
            columns["quantum_flow_count"][row] = len(current_index.quantum_number_flow)
            for name, expression in current_index.quantum_number_flow:
                quantum_flows["name_string_id"][quantum_cursor] = self.strings.add(
                    str(name)
                )
                quantum_flows["expression_string_id"][quantum_cursor] = (
                    self.strings.add(str(expression))
                )
                quantum_cursor += 1
            color = current_index.color_state
            if not all(isinstance(value, str) for value in color.basis_key):
                raise EagerColumnarInputError(
                    f"current {current.id} color basis keys must be strings"
                )
            columns["color_accuracy_string_id"][row] = self.strings.add(color.accuracy)
            columns["color_sector_id"][row] = _checked_u32(
                color.sector_id, "current color sector"
            )
            columns["color_line_groups_sequence_id"][row] = self.i32_sequences.add(
                color.line_groups
            )
            columns["color_basis_keys_sequence_id"][row] = self.u32_sequences.add(
                self.strings.add(value) for value in color.basis_key
            )
            columns["auxiliary_kind_string_id"][row] = self.strings.optional(
                current_index.auxiliary_kind
            )
            propagator = self.model._propagator_ir(
                current_index.particle_id, current_index.chirality
            )
            columns["propagator_ir_id"][row] = self.ir.add(
                propagator.to_json_dict(), "propagator IR"
            )
            kernel_id = self.resolver.propagator_kernel_id(
                current, propagator.to_json_dict()
            )
            columns["propagator_kernel_id"][row] = (
                MISSING_U32
                if kernel_id is None
                else _checked_u32(kernel_id, "prepared propagator kernel id")
            )
        self.tables.append(_freeze_table("currents", columns))
        self.tables.append(_freeze_table("coupling_orders", coupling_orders))
        self.tables.append(_freeze_table("quantum_flows", quantum_flows))

        momentum_masks = sorted(
            {current.index.momentum_mask for current in self.dag.currents},
            key=lambda value: (value.bit_count(), value),
        )
        momentum_columns = _allocate(len(momentum_masks), slot_id=_U32, bitset_id=_U32)
        for slot_id, mask in enumerate(momentum_masks):
            momentum_columns["slot_id"][slot_id] = slot_id
            momentum_columns["bitset_id"][slot_id] = self.bitsets.add(
                mask, "momentum mask"
            )
        self.tables.append(_freeze_table("momentum_masks", momentum_columns))

    def _sources(self) -> None:
        columns = _allocate(
            len(self.dag.sources),
            source_id=_U32,
            current_id=_U32,
            external_label=_U32,
            input_momentum_slot=_U32,
            source_ir_id=_U32,
            crossing_ir_id=_U32,
            crossing_factor_id=_U32,
            declared_state_index=_U32,
        )
        legs = {leg.label: leg for leg in self.dag.process.legs}
        for source_id, current_id in enumerate(self.dag.sources):
            current = self.dag.currents[current_id]
            if current.source_leg_label is None or current.source_helicity is None:
                raise EagerColumnarInputError(
                    f"source current {current_id} lacks leg/helicity provenance"
                )
            leg = legs.get(current.source_leg_label)
            if leg is None:
                raise EagerColumnarInputError(
                    f"source current {current_id} refers to unknown external leg"
                )
            source_ir = self.model._source_ir(current.index.particle_id)
            crossing = (
                source_ir.crossing
                if leg.is_initial
                else type(source_ir.crossing).identity()
            )
            expected = (
                current.source_helicity,
                current.index.chirality,
                current.index.spin_state,
            )
            state_index = next(
                (
                    index
                    for index, state in enumerate(source_ir.states)
                    if (
                        (applied := crossing.apply(state)).helicity,
                        applied.chirality,
                        applied.spin_state,
                    )
                    == expected
                ),
                None,
            )
            if state_index is None:
                raise EagerColumnarInputError(
                    f"source current {current_id} state {expected!r} is not declared"
                )
            columns["source_id"][source_id] = source_id
            columns["current_id"][source_id] = current_id
            columns["external_label"][source_id] = current.source_leg_label
            columns["input_momentum_slot"][source_id] = leg.label - 1
            source_ir_id = self.ir.add(source_ir.to_json_dict(), "source IR")
            crossing_ir_id = self.ir.add(crossing.to_json_dict(), "source crossing IR")
            columns["source_ir_id"][source_id] = source_ir_id
            columns["crossing_ir_id"][source_id] = crossing_ir_id
            columns["crossing_factor_id"][source_id] = self.factors.add(
                crossing.phase,
                "source crossing phase",
                source_ir_id=crossing_ir_id,
            )
            columns["declared_state_index"][source_id] = state_index
        self.tables.append(_freeze_table("sources", columns))

    def _interactions(self) -> None:
        group_by_key: dict[tuple[str, int], int] = {}
        self._interaction_group_members: list[list[int]] = []
        columns = _allocate(
            len(self.dag.interactions),
            id=_U32,
            stage_subset_size=_U32,
            vertex_kind=_I32,
            vertex_particles=(_I32, (3,)),
            left_current_id=_U32,
            right_current_id=_U32,
            result_current_id=_U32,
            coupling_id=_U32,
            coupling_factor_id=_U32,
            color_factor_id=_U32,
            evaluation_factor_id=_U32,
            evaluation_group_id=_U32,
            lowering_backend_string_id=_U32,
            full_tensor_network_ready=_U8,
            kernel_id=_U32,
            canonical_input_order=(_U8, (2,)),
            kernel_normalization_factor_id=_U32,
            output_factor_source=_U8,
        )
        for row, interaction in enumerate(self.dag.interactions):
            resolved = self.resolver.vertex_kernel(interaction)
            coupling_id = self._coupling(
                interaction.vertex_kind,
                interaction.vertex_particles,
                interaction.coupling,
            )
            columns["id"][row] = interaction.id
            columns["stage_subset_size"][row] = len(
                self.dag.currents[interaction.result_id].index.external_labels
            )
            columns["vertex_kind"][row] = _checked_i32(
                interaction.vertex_kind, "vertex kind"
            )
            columns["vertex_particles"][row] = interaction.vertex_particles
            columns["left_current_id"][row] = interaction.left_id
            columns["right_current_id"][row] = interaction.right_id
            columns["result_current_id"][row] = interaction.result_id
            columns["coupling_id"][row] = coupling_id
            columns["coupling_factor_id"][row] = self.coupling_specs[coupling_id][2]
            columns["color_factor_id"][row] = self.factors.add(
                interaction.color_weight, "interaction color weight"
            )
            columns["evaluation_factor_id"][row] = self.factors.add(
                interaction.evaluation_factor, "interaction evaluation factor"
            )
            columns["lowering_backend_string_id"][row] = self.strings.add(
                interaction.lowering_backend
            )
            columns["full_tensor_network_ready"][row] = (
                interaction.full_tensor_network_ready
            )
            _fill_resolved_kernel(columns, row, resolved, self.factors)
            group_key = (
                ("group", int(interaction.evaluation_group_id))
                if interaction.evaluation_group_id is not None
                else ("interaction", interaction.id)
            )
            group_id = group_by_key.get(group_key)
            if group_id is None:
                group_id = len(group_by_key)
                group_by_key[group_key] = group_id
                self._interaction_group_members.append([])
            self._interaction_group_members[group_id].append(interaction.id)
            columns["evaluation_group_id"][row] = group_id
        self.tables.append(_freeze_table("interactions", columns))

    def _roots(self) -> None:
        contraction_records: dict[int, object] = {}
        columns = _allocate(
            len(self.dag.amplitude_roots),
            id=_U32,
            kind_string_id=_U32,
            left_current_id=_U32,
            right_current_id=_U32,
            color_factor_id=_U32,
            contraction_ir_id=_U32,
            color_sector_id=_U32,
            vertex_kind=_I32,
            vertex_particles=(_I32, (3,)),
            coupling_id=_U32,
            coupling_factor_id=_U32,
            helicity_weight_factor_id=_U32,
            coherent_group_id=_U32,
            kernel_id=_U32,
            canonical_input_order=(_U8, (2,)),
            kernel_normalization_factor_id=_U32,
            output_factor_source=_U8,
        )
        for row, root in enumerate(self.dag.amplitude_roots):
            resolved = self.resolver.closure_kernel(root.to_json_dict())
            columns["id"][row] = root.id
            columns["kind_string_id"][row] = self.strings.add(root.kind)
            columns["left_current_id"][row] = root.left_id
            columns["right_current_id"][row] = root.right_id
            columns["color_factor_id"][row] = self.factors.add(
                root.color_weight, "root color weight"
            )
            contraction_ir_id = self.ir.add(
                root.contraction_ir.to_json_dict(), "contraction IR"
            )
            columns["contraction_ir_id"][row] = contraction_ir_id
            contraction_records.setdefault(contraction_ir_id, root.contraction_ir)
            sector_id = (
                root.color_sector_id
                if root.color_sector_id is not None
                else self.dag.currents[root.left_id].index.color_state.sector_id
            )
            columns["color_sector_id"][row] = _checked_u32(
                sector_id, "root color sector"
            )
            columns["vertex_kind"][row] = (
                MISSING_I32
                if root.vertex_kind is None
                else _checked_i32(root.vertex_kind, "root vertex kind")
            )
            columns["vertex_particles"][row] = (
                (MISSING_I32, MISSING_I32, MISSING_I32)
                if root.vertex_particles is None
                else root.vertex_particles
            )
            coupling_id = (
                MISSING_U32
                if root.vertex_kind is None or root.vertex_particles is None
                else self._coupling(
                    root.vertex_kind, root.vertex_particles, root.coupling
                )
            )
            columns["coupling_id"][row] = coupling_id
            columns["coupling_factor_id"][row] = (
                self.factors.add(root.coupling, "direct root coupling")
                if coupling_id == MISSING_U32
                else self.coupling_specs[coupling_id][2]
            )
            columns["helicity_weight_factor_id"][row] = self.factors.add(
                (root.helicity_weight, 0.0), "root helicity weight"
            )
            columns["coherent_group_id"][row] = self.group_ids[root.id]
            if resolved is None:
                columns["kernel_id"][row] = MISSING_U32
                columns["canonical_input_order"][row] = (0, 1)
                columns["kernel_normalization_factor_id"][row] = self.factors.add(
                    (1.0, 0.0), "direct closure normalization"
                )
                columns["output_factor_source"][row] = 0
            else:
                _fill_resolved_kernel(columns, row, resolved, self.factors)
        self.tables.append(_freeze_table("roots", columns))

        coefficient_count = sum(
            len(record.coefficients) for record in contraction_records.values()
        )
        coefficients = _allocate(
            coefficient_count,
            contraction_ir_id=_U32,
            component_index=_U32,
            factor_id=_U32,
        )
        cursor = 0
        for contraction_ir_id, record in sorted(contraction_records.items()):
            for component_index, coefficient in enumerate(record.coefficients):
                coefficients["contraction_ir_id"][cursor] = contraction_ir_id
                coefficients["component_index"][cursor] = component_index
                coefficients["factor_id"][cursor] = self.factors.add(
                    coefficient,
                    "contraction coefficient",
                    source_ir_id=contraction_ir_id,
                )
                cursor += 1
        self.tables.append(_freeze_table("contraction_coefficients", coefficients))

    def _interaction_groups(self) -> None:
        members_by_group = self._interaction_group_members
        groups = _allocate(
            len(members_by_group),
            representative_interaction_id=_U32,
            member_start=_U64,
            member_count=_U64,
        )
        members = _allocate(len(self.dag.interactions), interaction_id=_U32)
        cursor = 0
        for group_id, raw_group_members in enumerate(members_by_group):
            # Rust normalizes attachment factors by the representative
            # evaluation factor.  Keep a live member first for mixed groups;
            # an all-zero group remains explicit in the retained input and is
            # elided before runtime-plan invocation emission.
            group_members = sorted(
                raw_group_members,
                key=lambda interaction_id: (
                    self.dag.interactions[
                        interaction_id
                    ].evaluation_factor
                    == (0.0, 0.0)
                ),
            )
            groups["representative_interaction_id"][group_id] = group_members[0]
            groups["member_start"][group_id] = cursor
            groups["member_count"][group_id] = len(group_members)
            stop = cursor + len(group_members)
            members["interaction_id"][cursor:stop] = group_members
            cursor = stop
        self.tables.append(_freeze_table("interaction_groups", groups))
        self.tables.append(_freeze_table("interaction_group_members", members))

    def _color_plan(self) -> None:
        sectors = _allocate(
            len(self.dag.color_plan.sectors),
            id=_U32,
            kind_string_id=_U32,
            open_line_start=_U64,
            open_line_count=_U64,
            trace_labels_sequence_id=_U32,
            singlet_labels_sequence_id=_U32,
            word_labels_sequence_id=_U32,
        )
        open_line_count = sum(
            len(sector.open_color_lines) for sector in self.dag.color_plan.sectors
        )
        open_lines = _allocate(
            open_line_count,
            fundamental_label=_U32,
            antifundamental_label=_U32,
            adjoint_labels_sequence_id=_U32,
            singlet_labels_sequence_id=_U32,
        )
        cursor = 0
        for index, sector in enumerate(self.dag.color_plan.sectors):
            if sector.id != index:
                raise EagerColumnarInputError("color-sector ids must be dense")
            sectors["id"][index] = sector.id
            sectors["kind_string_id"][index] = self.strings.add(sector.kind)
            sectors["open_line_start"][index] = cursor
            sectors["open_line_count"][index] = len(sector.open_color_lines)
            sectors["trace_labels_sequence_id"][index] = self.u32_sequences.add(
                sector.trace_labels
            )
            sectors["singlet_labels_sequence_id"][index] = self.u32_sequences.add(
                sector.singlet_labels
            )
            word = sector.word_labels or sector.color_words[0]
            sectors["word_labels_sequence_id"][index] = self.u32_sequences.add(word)
            for line in sector.open_color_lines:
                open_lines["fundamental_label"][cursor] = line.fundamental_label
                open_lines["antifundamental_label"][cursor] = line.antifundamental_label
                open_lines["adjoint_labels_sequence_id"][cursor] = (
                    self.u32_sequences.add(line.adjoint_labels)
                )
                open_lines["singlet_labels_sequence_id"][cursor] = (
                    self.u32_sequences.add(line.singlet_labels)
                )
                cursor += 1
        self.tables.append(_freeze_table("color_sectors", sectors))
        self.tables.append(_freeze_table("color_open_lines", open_lines))

    def _topology_replay(self) -> None:
        replay = self.dag.lc_topology_replay or self.dag.color_topology_replay
        replay_kind = (
            0
            if replay is None
            else (1 if self.dag.lc_topology_replay is not None else 2)
        )
        replay_metadata = _allocate(
            1,
            present=_U8,
            replay_kind=_U8,
            physical_sector_ids_sequence_id=_U32,
            materialized_sector_ids_sequence_id=_U32,
        )
        replay_metadata["present"][0] = replay is not None
        replay_metadata["replay_kind"][0] = replay_kind
        replay_metadata["physical_sector_ids_sequence_id"][0] = (
            MISSING_U32
            if replay is None
            else self.u32_sequences.add(replay.physical_sector_ids)
        )
        replay_metadata["materialized_sector_ids_sequence_id"][0] = (
            MISSING_U32
            if replay is None
            else self.u32_sequences.add(replay.materialized_sector_ids)
        )
        self.tables.append(_freeze_table("lc_replay_metadata", replay_metadata))
        diagnostics_values = () if replay is None else replay.diagnostics
        diagnostics = _allocate(len(diagnostics_values), message_string_id=_U32)
        for row, message in enumerate(diagnostics_values):
            diagnostics["message_string_id"][row] = self.strings.add(str(message))
        self.tables.append(_freeze_table("lc_replay_diagnostics", diagnostics))

        partitions = () if replay is None else replay.partitions
        partition_columns = _allocate(
            len(partitions),
            representative_sector_id=_U32,
            materialized_sector_id=_U32,
            member_start=_U64,
            member_count=_U64,
            proof_algorithm_string_id=_U32,
            proof_digest_string_id=_U32,
        )
        member_count = sum(len(partition.active_sector_ids) for partition in partitions)
        members = _allocate(
            member_count,
            sector_id=_U32,
            factor_id=_U32,
            sign=_I32,
            permutation_start=_U64,
            permutation_count=_U64,
        )
        permutation_count = sum(
            len(permutation)
            for partition in partitions
            for permutation in partition.label_permutations
        )
        permutations = _allocate(
            permutation_count, representative_label=_U32, sector_label=_U32
        )
        member_cursor = 0
        permutation_cursor = 0
        for partition_id, partition in enumerate(partitions):
            partition_columns["representative_sector_id"][partition_id] = (
                partition.representative_sector_id
            )
            partition_columns["materialized_sector_id"][partition_id] = int(
                partition.materialized_sector_id
            )
            partition_columns["member_start"][partition_id] = member_cursor
            partition_columns["member_count"][partition_id] = len(
                partition.active_sector_ids
            )
            partition_columns["proof_algorithm_string_id"][partition_id] = (
                self.strings.add(str(partition.proof_algorithm))
            )
            partition_columns["proof_digest_string_id"][partition_id] = (
                self.strings.add(str(partition.proof_digest))
            )
            for sector_id, permutation, weight, sign in zip(
                partition.active_sector_ids,
                partition.label_permutations,
                partition.weights,
                partition.signs,
                strict=True,
            ):
                members["sector_id"][member_cursor] = sector_id
                members["factor_id"][member_cursor] = self.factors.add(
                    (weight * sign, 0.0), "LC replay factor"
                )
                members["sign"][member_cursor] = sign
                members["permutation_start"][member_cursor] = permutation_cursor
                members["permutation_count"][member_cursor] = len(permutation)
                for representative_label, sector_label in permutation:
                    permutations["representative_label"][permutation_cursor] = (
                        representative_label
                    )
                    permutations["sector_label"][permutation_cursor] = sector_label
                    permutation_cursor += 1
                member_cursor += 1
        residual_ids = () if replay is None else replay.residual_sector_ids
        residuals = _allocate(len(residual_ids), sector_id=_U32)
        residuals["sector_id"][:] = residual_ids
        self.tables.append(_freeze_table("lc_replay_partitions", partition_columns))
        self.tables.append(_freeze_table("lc_replay_members", members))
        self.tables.append(_freeze_table("lc_replay_permutations", permutations))
        self.tables.append(_freeze_table("lc_replay_residual_sectors", residuals))

    def _color_topology_replay_amplitudes(self) -> None:
        replay = self.color_topology_replay
        metadata = _allocate(
            1,
            present=_U8,
            contract_version=_U32,
            physical_group_count=_U64,
            mapping_count=_U64,
        )
        metadata["present"][0] = replay is not None
        metadata["contract_version"][0] = 0 if replay is None else 1
        metadata["physical_group_count"][0] = (
            0 if replay is None else len(replay.physical_descriptors)
        )
        metadata["mapping_count"][0] = (
            0 if replay is None else len(replay.mappings)
        )
        self.tables.append(_freeze_table("color_replay_metadata", metadata))

        physical_groups = () if replay is None else replay.physical_descriptors
        group_columns = _allocate(
            len(physical_groups),
            group_id=_U32,
            color_sector_id=_U32,
            color_word_sequence_id=_U32,
            helicity_values_sequence_id=_U32,
            helicity_weight_factor_id=_U32,
        )
        for row, descriptor in enumerate(physical_groups):
            group_columns["group_id"][row] = descriptor.group_id
            group_columns["color_sector_id"][row] = descriptor.sector_id
            group_columns["color_word_sequence_id"][row] = self.u32_sequences.add(
                descriptor.word
            )
            group_columns["helicity_values_sequence_id"][row] = (
                self.i32_sequences.add(
                    _descriptor_helicity_vector(self.dag, descriptor.helicity_key)
                )
            )
            group_columns["helicity_weight_factor_id"][row] = self.factors.add(
                (descriptor.helicity_weight, 0.0),
                "color replay physical-group helicity weight",
            )
        self.tables.append(
            _freeze_table("color_replay_physical_groups", group_columns)
        )

        mappings = () if replay is None else replay.mappings
        mapping_columns = _allocate(
            len(mappings),
            permutation_start=_U64,
            permutation_count=_U64,
            route_start=_U64,
            route_count=_U64,
        )
        permutation_count = sum(len(mapping.label_permutation) for mapping in mappings)
        route_count = sum(len(mapping.group_routes) for mapping in mappings)
        permutations = _allocate(
            permutation_count,
            representative_label=_U32,
            sector_label=_U32,
        )
        routes = _allocate(
            route_count,
            source_group_id=_U32,
            target_group_id=_U32,
            factor_id=_U32,
        )
        permutation_cursor = 0
        route_cursor = 0
        for mapping_id, mapping in enumerate(mappings):
            mapping_columns["permutation_start"][mapping_id] = permutation_cursor
            mapping_columns["permutation_count"][mapping_id] = len(
                mapping.label_permutation
            )
            mapping_columns["route_start"][mapping_id] = route_cursor
            mapping_columns["route_count"][mapping_id] = len(mapping.group_routes)
            for representative_label, sector_label in mapping.label_permutation:
                permutations["representative_label"][permutation_cursor] = (
                    representative_label
                )
                permutations["sector_label"][permutation_cursor] = sector_label
                permutation_cursor += 1
            for route in mapping.group_routes:
                routes["source_group_id"][route_cursor] = route.source_group_id
                routes["target_group_id"][route_cursor] = route.target_group_id
                routes["factor_id"][route_cursor] = self.factors.add(
                    route.factor,
                    "color replay amplitude route factor",
                )
                route_cursor += 1
        self.tables.append(_freeze_table("color_replay_mappings", mapping_columns))
        self.tables.append(
            _freeze_table("color_replay_permutations", permutations)
        )
        self.tables.append(_freeze_table("color_replay_routes", routes))

        contraction = (
            None
            if replay is None
            else build_color_contraction_plan(
                self.dag.color_plan, replay.physical_descriptors
            )
        )
        if replay is not None and (
            contraction is None
            or not contraction.supported
            or contraction.group_count != len(replay.physical_descriptors)
        ):
            raise EagerColumnarInputError(
                "color replay requires a complete physical color contraction"
            )
        contraction_metadata = _allocate(
            1,
            present=_U8,
            group_count=_U64,
            includes_color_factor=_U8,
            repeated=_U8,
            component_count=_U64,
        )
        contraction_metadata["present"][0] = contraction is not None
        contraction_metadata["group_count"][0] = (
            0 if contraction is None else contraction.group_count
        )
        contraction_metadata["includes_color_factor"][0] = (
            False if contraction is None else contraction.includes_color_factor
        )
        repeated = None if contraction is None else contraction.repeated_block
        contraction_metadata["repeated"][0] = repeated is not None
        contraction_metadata["component_count"][0] = (
            0 if repeated is None else repeated.component_count
        )
        self.tables.append(
            _freeze_table(
                "color_replay_contraction_metadata", contraction_metadata
            )
        )
        component_group_ids = () if repeated is None else repeated.component_group_ids
        component_groups = _allocate(
            len(component_group_ids),
            group_id=_U32,
        )
        for row, group_id in enumerate(component_group_ids):
            component_groups["group_id"][row] = group_id
        self.tables.append(
            _freeze_table(
                "color_replay_contraction_component_groups",
                component_groups,
            )
        )

        entries = (
            ()
            if contraction is None
            else (contraction.entries if repeated is None else repeated.entries)
        )
        contraction_entries = _allocate(
            len(entries),
            left_group_id=_U32,
            right_group_id=_U32,
            left_group_index=_U32,
            right_group_index=_U32,
            weight_factor_id=_U32,
            symmetry_factor_id=_U32,
        )
        for row, entry in enumerate(entries):
            if repeated is None:
                contraction_entries["left_group_id"][row] = entry.left_group_id
                contraction_entries["right_group_id"][row] = entry.right_group_id
                contraction_entries["left_group_index"][row] = MISSING_U32
                contraction_entries["right_group_index"][row] = MISSING_U32
            else:
                contraction_entries["left_group_id"][row] = MISSING_U32
                contraction_entries["right_group_id"][row] = MISSING_U32
                contraction_entries["left_group_index"][row] = (
                    entry.left_group_index
                )
                contraction_entries["right_group_index"][row] = (
                    entry.right_group_index
                )
            contraction_entries["weight_factor_id"][row] = self.factors.add(
                (entry.weight_re, entry.weight_im),
                "color replay contraction weight",
                exact_ir=_color_weight_ir(entry),
            )
            contraction_entries["symmetry_factor_id"][row] = self.factors.add(
                (entry.symmetry_factor, 0.0),
                "color replay contraction symmetry",
            )
        self.tables.append(
            _freeze_table(
                "color_replay_contraction_entries", contraction_entries
            )
        )

    def _helicity_proof(self) -> None:
        proof = self.dag.helicity_recurrence
        metadata = _allocate(
            1,
            present=_U8,
            contract_version=_U32,
            proof_algorithm_string_id=_U32,
            proof_current_count=_U64,
            proof_root_count=_U64,
        )
        metadata["present"][0] = proof is not None
        metadata["contract_version"][0] = (
            0 if proof is None else HELICITY_RECURRENCE_CONTRACT_VERSION
        )
        metadata["proof_algorithm_string_id"][0] = (
            MISSING_U32
            if proof is None
            else self.strings.add(HELICITY_RECURRENCE_PROOF_ALGORITHM)
        )
        metadata["proof_current_count"][0] = 0 if proof is None else proof.current_count
        metadata["proof_root_count"][0] = (
            0 if proof is None else proof.amplitude_root_count
        )
        self.tables.append(_freeze_table("helicity_proof_metadata", metadata))
        diagnostic_values = () if proof is None else proof.diagnostics
        diagnostics = _allocate(len(diagnostic_values), message_string_id=_U32)
        for row, message in enumerate(diagnostic_values):
            diagnostics["message_string_id"][row] = self.strings.add(str(message))
        self.tables.append(_freeze_table("helicity_proof_diagnostics", diagnostics))

        domains = () if proof is None else proof.selector_domains
        domain_columns = _allocate(
            len(domains), id=_U32, complete=_U8, state_start=_U64, state_count=_U64
        )
        state_count = sum(len(domain.source_states) for domain in domains)
        states = _allocate(state_count, external_label=_U32, helicity=_I32)
        state_cursor = 0
        for domain in domains:
            domain_columns["id"][domain.id] = domain.id
            domain_columns["complete"][domain.id] = domain.complete
            domain_columns["state_start"][domain.id] = state_cursor
            domain_columns["state_count"][domain.id] = len(domain.source_states)
            for label, helicity in domain.source_states:
                states["external_label"][state_cursor] = label
                states["helicity"][state_cursor] = helicity
                state_cursor += 1
        self.tables.append(_freeze_table("helicity_domains", domain_columns))
        self.tables.append(_freeze_table("helicity_domain_states", states))

        classes = () if proof is None else proof.recurrence_classes
        class_columns = _allocate(
            len(classes),
            representative_current_id=_U32,
            external_labels_sequence_id=_U32,
            source_class=_U8,
            class_id_string_id=_U32,
            proof_digest_string_id=_U32,
            transition_contract_ids_sequence_id=_U32,
            member_start=_U64,
            member_count=_U64,
        )
        member_count = sum(len(item.members) for item in classes)
        members = _allocate(
            member_count,
            class_index=_U32,
            current_id=_U32,
            selector_domain_id=_U32,
            factor_id=_U32,
        )
        class_index_by_id: dict[str, int] = {}
        cursor = 0
        for class_index, recurrence in enumerate(classes):
            class_index_by_id[recurrence.class_id] = class_index
            class_columns["representative_current_id"][class_index] = (
                recurrence.representative_current_id
            )
            class_columns["external_labels_sequence_id"][class_index] = (
                self.u32_sequences.add(recurrence.external_labels)
            )
            class_columns["source_class"][class_index] = recurrence.source_class
            class_columns["class_id_string_id"][class_index] = self.strings.add(
                recurrence.class_id
            )
            class_columns["proof_digest_string_id"][class_index] = self.strings.add(
                recurrence.proof_digest
            )
            class_columns["transition_contract_ids_sequence_id"][class_index] = (
                self.u32_sequences.add(
                    self.strings.add(value)
                    for value in recurrence.transition_contract_ids
                )
            )
            class_columns["member_start"][class_index] = cursor
            class_columns["member_count"][class_index] = len(recurrence.members)
            for member in recurrence.members:
                members["class_index"][cursor] = class_index
                members["current_id"][cursor] = member.current_id
                members["selector_domain_id"][cursor] = member.selector_domain_id
                members["factor_id"][cursor] = self.factors.add(
                    member.factor, "helicity recurrence factor"
                )
                cursor += 1
        self.tables.append(_freeze_table("helicity_recurrence_classes", class_columns))
        self.tables.append(_freeze_table("helicity_recurrence_members", members))

        source_mappings = () if proof is None else proof.source_state_mappings
        source_columns = _allocate(
            len(source_mappings),
            current_id=_U32,
            external_label=_U32,
            helicity=_I32,
            chirality=_I32,
            spin_state_sequence_id=_U32,
            declared_state_index=_U32,
            selector_domain_id=_U32,
            recurrence_class_index=_U32,
            representative_current_id=_U32,
            source_contract_digest_string_id=_U32,
            factor_id=_U32,
        )
        for row, mapping in enumerate(source_mappings):
            spin_state = (
                mapping.spin_state
                if isinstance(mapping.spin_state, tuple)
                else (mapping.spin_state,)
            )
            source_columns["current_id"][row] = mapping.current_id
            source_columns["external_label"][row] = mapping.external_label
            source_columns["helicity"][row] = mapping.helicity
            source_columns["chirality"][row] = mapping.chirality
            source_columns["spin_state_sequence_id"][row] = self.i32_sequences.add(
                spin_state
            )
            source_columns["declared_state_index"][row] = mapping.declared_state_index
            source_columns["selector_domain_id"][row] = mapping.selector_domain_id
            source_columns["recurrence_class_index"][row] = class_index_by_id[
                mapping.recurrence_class_id
            ]
            source_columns["representative_current_id"][row] = (
                mapping.representative_current_id
            )
            source_columns["source_contract_digest_string_id"][row] = self.strings.add(
                mapping.source_contract_digest
            )
            source_columns["factor_id"][row] = self.factors.add(
                mapping.factor, "helicity source factor"
            )
        self.tables.append(_freeze_table("helicity_source_mappings", source_columns))

        amplitude_classes = () if proof is None else proof.amplitude_classes
        amplitude_class_columns = _allocate(
            len(amplitude_classes),
            representative_root_id=_U32,
            class_id_string_id=_U32,
            proof_digest_string_id=_U32,
            transition_contract_ids_sequence_id=_U32,
            member_start=_U64,
            member_count=_U64,
        )
        amplitude_member_count = sum(len(item.members) for item in amplitude_classes)
        amplitude_members = _allocate(
            amplitude_member_count,
            class_index=_U32,
            root_id=_U32,
            selector_domain_start=_U64,
            selector_domain_count=_U64,
            factor_id=_U32,
        )
        amplitude_domain_count = sum(
            len(member.selector_domain_ids)
            for item in amplitude_classes
            for member in item.members
        )
        amplitude_domains = _allocate(amplitude_domain_count, selector_domain_id=_U32)
        member_cursor = 0
        domain_cursor = 0
        for class_index, recurrence in enumerate(amplitude_classes):
            amplitude_class_columns["representative_root_id"][class_index] = (
                recurrence.representative_root_id
            )
            amplitude_class_columns["class_id_string_id"][class_index] = (
                self.strings.add(recurrence.class_id)
            )
            amplitude_class_columns["proof_digest_string_id"][class_index] = (
                self.strings.add(recurrence.proof_digest)
            )
            amplitude_class_columns["transition_contract_ids_sequence_id"][
                class_index
            ] = self.u32_sequences.add(
                self.strings.add(value) for value in recurrence.transition_contract_ids
            )
            amplitude_class_columns["member_start"][class_index] = member_cursor
            amplitude_class_columns["member_count"][class_index] = len(
                recurrence.members
            )
            for member in recurrence.members:
                amplitude_members["class_index"][member_cursor] = class_index
                amplitude_members["root_id"][member_cursor] = member.root_id
                amplitude_members["selector_domain_start"][member_cursor] = (
                    domain_cursor
                )
                amplitude_members["selector_domain_count"][member_cursor] = len(
                    member.selector_domain_ids
                )
                for domain_id in member.selector_domain_ids:
                    amplitude_domains["selector_domain_id"][domain_cursor] = domain_id
                    domain_cursor += 1
                amplitude_members["factor_id"][member_cursor] = self.factors.add(
                    member.factor, "helicity amplitude factor"
                )
                member_cursor += 1
        self.tables.append(
            _freeze_table("helicity_amplitude_classes", amplitude_class_columns)
        )
        self.tables.append(
            _freeze_table("helicity_amplitude_members", amplitude_members)
        )
        self.tables.append(
            _freeze_table("helicity_amplitude_member_domains", amplitude_domains)
        )

        for name, values, field_name in (
            (
                "helicity_residual_currents",
                () if proof is None else proof.residual_current_ids,
                "current_id",
            ),
            (
                "helicity_residual_roots",
                () if proof is None else proof.residual_root_ids,
                "root_id",
            ),
            (
                "helicity_structural_zero_domains",
                () if proof is None else proof.structural_zero_selector_domain_ids,
                "selector_domain_id",
            ),
        ):
            table = _allocate(len(values), **{field_name: _U32})
            table[field_name][:] = values
            self.tables.append(_freeze_table(name, table))

    def _helicity_materialization(self) -> None:
        materialization = self.dag.helicity_materialization
        metadata = _allocate(
            1,
            present=_U8,
            strategy_string_id=_U32,
            proof_current_count=_U64,
            proof_root_count=_U64,
            materialized_current_count=_U64,
            materialized_root_count=_U64,
        )
        metadata["present"][0] = materialization is not None
        metadata["strategy_string_id"][0] = (
            MISSING_U32
            if materialization is None
            else self.strings.add(materialization.strategy)
        )
        metadata["proof_current_count"][0] = (
            0 if materialization is None else materialization.proof_current_count
        )
        metadata["proof_root_count"][0] = (
            0 if materialization is None else materialization.proof_root_count
        )
        metadata["materialized_current_count"][0] = (
            0 if materialization is None else materialization.materialized_current_count
        )
        metadata["materialized_root_count"][0] = (
            0 if materialization is None else materialization.materialized_root_count
        )
        self.tables.append(_freeze_table("helicity_materialization_metadata", metadata))

        mapping = (
            ()
            if materialization is None
            else materialization.proof_to_materialized_current
        )
        mapping_columns = _allocate(len(mapping), materialized_current_id=_U32)
        mapping_columns["materialized_current_id"][:] = mapping
        self.tables.append(
            _freeze_table("helicity_materialized_current_map", mapping_columns)
        )

        source_routes = () if materialization is None else materialization.source_routes
        source_columns = _allocate(
            len(source_routes),
            materialized_current_id=_U32,
            external_label=_U32,
            helicity=_I32,
            chirality=_I32,
            spin_state_sequence_id=_U32,
            declared_state_index=_U32,
            selector_domain_id=_U32,
            factor_id=_U32,
        )
        for row, route in enumerate(source_routes):
            spin_state = (
                route.spin_state
                if isinstance(route.spin_state, tuple)
                else (route.spin_state,)
            )
            source_columns["materialized_current_id"][row] = (
                route.materialized_current_id
            )
            source_columns["external_label"][row] = route.external_label
            source_columns["helicity"][row] = route.helicity
            source_columns["chirality"][row] = route.chirality
            source_columns["spin_state_sequence_id"][row] = self.i32_sequences.add(
                spin_state
            )
            source_columns["declared_state_index"][row] = route.declared_state_index
            source_columns["selector_domain_id"][row] = route.selector_domain_id
            source_columns["factor_id"][row] = self.factors.add(
                route.factor, "materialized source factor"
            )
        self.tables.append(
            _freeze_table("helicity_materialized_source_routes", source_columns)
        )

        amplitude_routes = (
            () if materialization is None else materialization.amplitude_routes
        )
        amplitude_columns = _allocate(
            len(amplitude_routes),
            materialized_root_id=_U32,
            selector_domain_start=_U64,
            selector_domain_count=_U64,
            factor_id=_U32,
            residual=_U8,
        )
        domain_count = sum(len(route.selector_domain_ids) for route in amplitude_routes)
        domains = _allocate(domain_count, selector_domain_id=_U32)
        cursor = 0
        for row, route in enumerate(amplitude_routes):
            amplitude_columns["materialized_root_id"][row] = route.materialized_root_id
            amplitude_columns["selector_domain_start"][row] = cursor
            amplitude_columns["selector_domain_count"][row] = len(
                route.selector_domain_ids
            )
            for domain_id in route.selector_domain_ids:
                domains["selector_domain_id"][cursor] = domain_id
                cursor += 1
            amplitude_columns["factor_id"][row] = self.factors.add(
                route.factor, "materialized amplitude factor"
            )
            amplitude_columns["residual"][row] = route.residual
        self.tables.append(
            _freeze_table("helicity_materialized_amplitude_routes", amplitude_columns)
        )
        self.tables.append(
            _freeze_table("helicity_materialized_amplitude_domains", domains)
        )

        schedules = (
            () if materialization is None else materialization.selector_schedules
        )
        schedule_columns = _allocate(
            len(schedules),
            selector_domain_id=_U32,
            active_current_sequence_id=_U32,
            active_root_sequence_id=_U32,
            structural_zero=_U8,
        )
        for row, schedule in enumerate(schedules):
            schedule_columns["selector_domain_id"][row] = schedule.selector_domain_id
            schedule_columns["active_current_sequence_id"][row] = (
                self.u32_sequences.add(schedule.active_current_ids)
            )
            schedule_columns["active_root_sequence_id"][row] = self.u32_sequences.add(
                schedule.active_root_ids
            )
            schedule_columns["structural_zero"][row] = schedule.structural_zero
        self.tables.append(
            _freeze_table("helicity_materialized_schedules", schedule_columns)
        )

    def _selectors_and_reductions(self) -> None:
        selector_records, selector_id_by_vector = self._helicity_selectors()
        helicity_selectors = _allocate(
            len(selector_records),
            values_sequence_id=_U32,
            representative_sequence_id=_U32,
            coefficient_factor_id=_U32,
            computed=_U8,
            structural_zero=_U8,
        )
        for row, (vector, representative, coefficient, computed, zero) in enumerate(
            selector_records
        ):
            helicity_selectors["values_sequence_id"][row] = self.i32_sequences.add(
                vector
            )
            helicity_selectors["representative_sequence_id"][row] = (
                self.i32_sequences.add(representative)
            )
            helicity_selectors["coefficient_factor_id"][row] = self.factors.add(
                (coefficient, 0.0), "helicity selector coefficient"
            )
            helicity_selectors["computed"][row] = computed
            helicity_selectors["structural_zero"][row] = zero
        self.tables.append(_freeze_table("helicity_selectors", helicity_selectors))

        color_records, color_id_by_word = self._color_selectors()
        color_selectors = _allocate(
            len(color_records),
            word_sequence_id=_U32,
            representative_word_sequence_id=_U32,
            coefficient_factor_id=_U32,
            computed=_U8,
        )
        for row, (word, representative, coefficient, computed) in enumerate(
            color_records
        ):
            color_selectors["word_sequence_id"][row] = self.u32_sequences.add(word)
            color_selectors["representative_word_sequence_id"][row] = (
                self.u32_sequences.add(representative)
            )
            color_selectors["coefficient_factor_id"][row] = self.factors.add(
                (coefficient, 0.0), "color selector coefficient"
            )
            color_selectors["computed"][row] = computed
        self.tables.append(_freeze_table("color_selectors", color_selectors))

        multiple_lc_sectors = _has_multiple_lc_root_sectors(self.dag)
        group_weights: dict[int, tuple[float, float]] = {}
        for root in self.dag.amplitude_roots:
            group_weights.setdefault(
                self.group_ids[root.id],
                (
                    root.helicity_weight,
                    _root_all_sector_weight(
                        self.dag,
                        root,
                        has_multiple_lc_root_sectors=multiple_lc_sectors,
                    ),
                ),
            )
        coherent_groups = _allocate(
            len(self.group_descriptors),
            id=_U32,
            color_sector_id=_U32,
            color_word_sequence_id=_U32,
            helicity_values_sequence_id=_U32,
            helicity_weight_factor_id=_U32,
            all_sector_weight_factor_id=_U32,
        )
        reduction_rows: list[tuple[int, int, int]] = []
        for descriptor in self.group_descriptors:
            group_id = descriptor.group_id
            vector = _descriptor_helicity_vector(self.dag, descriptor.helicity_key)
            word = tuple(int(value) for value in descriptor.word)
            helicity_weight, all_sector_weight = group_weights[group_id]
            coherent_groups["id"][group_id] = group_id
            coherent_groups["color_sector_id"][group_id] = descriptor.sector_id
            coherent_groups["color_word_sequence_id"][group_id] = (
                self.u32_sequences.add(word)
            )
            coherent_groups["helicity_values_sequence_id"][group_id] = (
                self.i32_sequences.add(vector)
            )
            coherent_groups["helicity_weight_factor_id"][group_id] = self.factors.add(
                (helicity_weight, 0.0), "group helicity weight"
            )
            coherent_groups["all_sector_weight_factor_id"][group_id] = self.factors.add(
                (all_sector_weight, 0.0), "group all-sector weight"
            )
            helicity_members = [vector]
            if helicity_weight > 1.0 + 1.0e-12:
                flipped = tuple(-value for value in vector)
                if flipped != vector:
                    helicity_members.append(flipped)
            color_members = [word]
            if (
                word
                and all_sector_weight > helicity_weight + 1.0e-12
                and (reflected := (word[0], *reversed(word[1:]))) != word
            ):
                color_members.append(reflected)
            for helicity in helicity_members:
                for color in color_members:
                    reduction_rows.append(
                        (
                            group_id,
                            selector_id_by_vector[helicity],
                            (
                                color_id_by_word[color]
                                if self.dag.process.color_accuracy == "lc"
                                else 0
                            ),
                        )
                    )
        self.tables.append(_freeze_table("coherent_groups", coherent_groups))
        reductions = _allocate(
            len(reduction_rows),
            coherent_group_id=_U32,
            helicity_selector_id=_U32,
            color_selector_id=_U32,
        )
        for row, values in enumerate(reduction_rows):
            (
                reductions["coherent_group_id"][row],
                reductions["helicity_selector_id"][row],
                reductions["color_selector_id"][row],
            ) = values
        self.tables.append(_freeze_table("reduction_members", reductions))

        contraction = (
            None
            if self.color_topology_replay is not None
            else build_color_contraction_plan(
                self.dag.color_plan, self.group_descriptors
            )
        )
        contraction_metadata = _allocate(
            1,
            present=_U8,
            color_accuracy_string_id=_U32,
            supported=_U8,
            reason_string_id=_U32,
            group_count=_U64,
            includes_color_factor=_U8,
        )
        contraction_metadata["present"][0] = contraction is not None
        contraction_metadata["color_accuracy_string_id"][0] = (
            MISSING_U32
            if contraction is None
            else self.strings.add(contraction.color_accuracy)
        )
        contraction_metadata["supported"][0] = (
            False if contraction is None else contraction.supported
        )
        contraction_metadata["reason_string_id"][0] = (
            MISSING_U32
            if contraction is None or contraction.reason is None
            else self.strings.add(contraction.reason)
        )
        contraction_metadata["group_count"][0] = (
            0 if contraction is None else contraction.group_count
        )
        contraction_metadata["includes_color_factor"][0] = (
            False if contraction is None else contraction.includes_color_factor
        )
        self.tables.append(
            _freeze_table("color_contraction_metadata", contraction_metadata)
        )
        entry_count = 0 if contraction is None else contraction.logical_entry_count
        contraction_columns = _allocate(
            entry_count,
            left_group_id=_U32,
            right_group_id=_U32,
            weight_factor_id=_U32,
            symmetry_factor_id=_U32,
        )
        logical_entries = (
            () if contraction is None else contraction.iter_logical_entries()
        )
        for row, entry in enumerate(logical_entries):
            contraction_columns["left_group_id"][row] = entry.left_group_id
            contraction_columns["right_group_id"][row] = entry.right_group_id
            contraction_columns["weight_factor_id"][row] = self.factors.add(
                (entry.weight_re, entry.weight_im),
                "color contraction weight",
                exact_ir=_color_weight_ir(entry),
            )
            contraction_columns["symmetry_factor_id"][row] = self.factors.add(
                (entry.symmetry_factor, 0.0), "color contraction symmetry"
            )
        self.tables.append(
            _freeze_table("color_contraction_entries", contraction_columns)
        )

    def _helicity_selectors(
        self,
    ) -> tuple[
        tuple[tuple[tuple[int, ...], tuple[int, ...], float, bool, bool], ...],
        dict[tuple[int, ...], int],
    ]:
        represented: dict[tuple[int, ...], tuple[tuple[int, ...], bool, float]] = {}
        materialization = self.dag.helicity_materialization
        recurrence = self.dag.helicity_recurrence
        if materialization is None:
            for descriptor in self.group_descriptors:
                vector = _descriptor_helicity_vector(self.dag, descriptor.helicity_key)
                weight = float(descriptor.helicity_weight)
                represented.setdefault(vector, (vector, True, 1.0))
                if weight > 1.0 + 1.0e-12:
                    if not math.isclose(weight, 2.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
                        raise EagerColumnarInputError(
                            f"unsupported helicity reuse weight {weight!r}"
                        )
                    flipped = tuple(-value for value in vector)
                    represented.setdefault(flipped, (vector, False, 1.0))
        else:
            if recurrence is None:
                raise EagerColumnarInputError(
                    "helicity materialization lacks its recurrence proof"
                )
            domain_by_id = {domain.id: domain for domain in recurrence.selector_domains}
            group_by_root = self.group_ids
            representative_by_group = {
                descriptor.group_id: _descriptor_helicity_vector(
                    self.dag, descriptor.helicity_key
                )
                for descriptor in self.group_descriptors
            }
            for route in materialization.amplitude_routes:
                group_id = group_by_root[route.materialized_root_id]
                representative = representative_by_group[group_id]
                coefficient = route.factor[0] ** 2 + route.factor[1] ** 2
                for domain_id in route.selector_domain_ids:
                    domain = domain_by_id[domain_id]
                    by_label = dict(domain.source_states)
                    vector = tuple(
                        by_label.get(int(leg.label), 0) for leg in self.dag.process.legs
                    )
                    candidate = (
                        representative,
                        vector == representative,
                        coefficient,
                    )
                    previous = represented.setdefault(vector, candidate)
                    if previous != candidate:
                        raise EagerColumnarInputError(
                            f"inconsistent materialized helicity route {vector}"
                        )

        possible = _possible_helicity_vectors(self.dag, self.model)
        rows = []
        for vector in sorted(possible):
            if vector in represented:
                representative, computed, coefficient = represented[vector]
                rows.append((vector, representative, coefficient, computed, False))
            else:
                rows.append((vector, vector, 0.0, False, True))
        result = tuple(rows)
        return result, {row[0]: index for index, row in enumerate(result)}

    def _color_selectors(
        self,
    ) -> tuple[
        tuple[tuple[tuple[int, ...], tuple[int, ...], float, bool], ...],
        dict[tuple[int, ...], int],
    ]:
        if self.dag.process.color_accuracy != "lc":
            records = (((), (), 1.0, True),)
            return records, {(): 0}
        replay = self.dag.lc_topology_replay
        materialized = (
            set(replay.materialized_sector_ids) if replay is not None else None
        )
        records: dict[tuple[int, ...], tuple[tuple[int, ...], float, bool]] = {}
        for sector in self.dag.color_plan.sectors:
            word = tuple(sector.word_labels or sector.color_words[0])
            representative = sector
            if replay is not None:
                representative = self.dag.color_plan.sector(
                    replay.representative_for(sector.id)
                )
                if representative is None:
                    raise EagerColumnarInputError(
                        f"missing replay representative for sector {sector.id}"
                    )
            representative_word = tuple(
                representative.word_labels or representative.color_words[0]
            )
            records.setdefault(
                word,
                (
                    representative_word,
                    1.0,
                    materialized is None or sector.id in materialized,
                ),
            )
            if (
                self.dag.color_plan.trace_reflections_folded
                and sector.kind == "single-trace"
                and len(word) > 2
                and (reflected := (word[0], *reversed(word[1:]))) != word
            ):
                records.setdefault(reflected, (representative_word, 1.0, False))
        rows = tuple((word, *records[word]) for word in sorted(records))
        return rows, {row[0]: index for index, row in enumerate(rows)}

    def _coupling(
        self,
        kind: int,
        particles: tuple[int, int, int],
        coupling: tuple[float, float],
    ) -> int:
        key = (int(kind), tuple(int(value) for value in particles), tuple(coupling))
        existing = self.coupling_ids.get(key)
        if existing is not None:
            return existing
        names = runtime_coupling_parameter_names(
            kind, particles, coupling, model=self.model
        )
        exact_ir = (
            None
            if not any(name is not None for name in names)
            else {
                "abi": "pyamplicol-runtime-coupling-v1",
                "kind": "model-parameter-components",
                "components": [
                    (
                        {"kind": "binary64-constant", "component": component}
                        if name is None
                        else {"kind": "model-parameter", "name": str(name)}
                    )
                    for component, name in enumerate(names)
                ],
            }
        )
        name_sequence_id = self.u32_sequences.add(
            MISSING_U32 if name is None else self.strings.add(name) for name in names
        )
        result = _checked_u32(len(self.coupling_specs), "coupling id")
        self.coupling_specs.append(
            (
                _checked_i32(kind, "coupling vertex kind"),
                tuple(_checked_i32(value, "coupling particle") for value in particles),
                self.factors.add(
                    coupling,
                    "coupling constant",
                    exact_ir=exact_ir,
                ),
                name_sequence_id,
            )
        )
        self.coupling_ids[key] = result
        return result

    def _couplings(self) -> None:
        columns = _allocate(
            len(self.coupling_specs),
            vertex_kind=_I32,
            vertex_particles=(_I32, (3,)),
            constant_factor_id=_U32,
            parameter_name_ids_sequence_id=_U32,
        )
        for row, (kind, particles, factor_id, names_id) in enumerate(
            self.coupling_specs
        ):
            columns["vertex_kind"][row] = kind
            columns["vertex_particles"][row] = particles
            columns["constant_factor_id"][row] = factor_id
            columns["parameter_name_ids_sequence_id"][row] = names_id
        self.tables.append(_freeze_table("couplings", columns))

    def _model_parameters(self) -> None:
        specs = _model_parameter_specs(
            self.dag,
            self.model,
            tuple(
                (
                    kind,
                    particles,
                    self.factors.pair(factor_id),
                    tuple(
                        None if name_id == MISSING_U32 else self.strings.values[name_id]
                        for name_id in self.u32_sequences.values[names_id]
                    ),
                )
                for kind, particles, factor_id, names_id in self.coupling_specs
            ),
        )
        columns = _allocate(
            len(specs),
            name_string_id=_U32,
            kind_string_id=_U32,
            default_value=_F64,
            default_factor_id=_U32,
            runtime_name_string_id=_U32,
            complex_component=_I32,
            parameter_type_string_id=_U32,
            pdg=_I32,
            vertex_kind=_I32,
            vertex_particles=(_I32, (3,)),
            coupling_component=_I32,
            derived=_U8,
            complex_domain_string_id=_U32,
            definition_string_id=_U32,
        )
        for row, spec in enumerate(specs):
            columns["name_string_id"][row] = self.strings.add(spec.name)
            columns["kind_string_id"][row] = self.strings.add(spec.kind)
            columns["default_value"][row] = _finite(spec.default, "parameter default")
            exact_ir = (
                None
                if not spec.derived or spec.definition is None
                else {
                    "abi": "pyamplicol-derived-parameter-v1",
                    "kind": "canonical-expression-component",
                    "name": spec.runtime_name or spec.name,
                    "component": spec.complex_component,
                    "expression": spec.definition,
                    "domain": spec.complex_domain,
                }
            )
            columns["default_factor_id"][row] = self.factors.add(
                (spec.default, 0.0),
                "model parameter default",
                exact_ir=exact_ir,
            )
            columns["runtime_name_string_id"][row] = self.strings.optional(
                spec.runtime_name
            )
            columns["complex_component"][row] = spec.complex_component
            columns["parameter_type_string_id"][row] = self.strings.optional(
                spec.parameter_type
            )
            columns["pdg"][row] = MISSING_I32 if spec.pdg is None else spec.pdg
            columns["vertex_kind"][row] = (
                MISSING_I32 if spec.vertex_kind is None else spec.vertex_kind
            )
            columns["vertex_particles"][row] = (
                (MISSING_I32, MISSING_I32, MISSING_I32)
                if spec.vertex_particles is None
                else spec.vertex_particles
            )
            columns["coupling_component"][row] = (
                MISSING_I32
                if spec.coupling_component is None
                else spec.coupling_component
            )
            columns["derived"][row] = spec.derived
            columns["complex_domain_string_id"][row] = self.strings.optional(
                spec.complex_domain
            )
            columns["definition_string_id"][row] = self.strings.optional(
                spec.definition
            )
        self.tables.append(_freeze_table("model_parameters", columns))


def build_eager_lowering_input_v1(
    *,
    dag: GenericDAG,
    model: Model,
    resolver: EagerKernelResolver,
    process_id: str | None = None,
) -> EagerLoweringInputV1:
    """Build the v1 lowering input without constructing a runtime schema."""

    return _Builder(dag, model, resolver, process_id).build()


def runtime_model_parameter_records_from_eager_input(
    lowering_input: EagerLoweringInputV1,
) -> tuple[dict[str, object], ...]:
    """Project the authoritative eager parameter columns into physics records."""

    table = lowering_input.table("model_parameters")
    names = table.column("name_string_id")
    kinds = table.column("kind_string_id")
    defaults = table.column("default_value")
    runtime_names = table.column("runtime_name_string_id")
    components = table.column("complex_component")
    records: list[dict[str, object]] = []
    for row in range(table.row_count):
        runtime_name_id = int(runtime_names[row])
        component = int(components[row])
        record: dict[str, object] = {
            "name": lowering_input.string_catalog[int(names[row])],
            "kind": lowering_input.string_catalog[int(kinds[row])],
            "parameter_index": row,
            "default": float(defaults[row]),
        }
        if runtime_name_id != MISSING_U32:
            record["runtime_name"] = lowering_input.string_catalog[runtime_name_id]
        if component == 0:
            record["complex_component"] = "real"
        elif component == 1:
            record["complex_component"] = "imag"
        elif component != -1:
            raise EagerColumnarInputError(
                f"model parameter {row} has invalid complex component {component}"
            )
        records.append(record)
    return tuple(records)


def _model_parameter_specs(
    dag: GenericDAG,
    model: Model,
    couplings: Sequence[
        tuple[
            int,
            tuple[int, int, int],
            tuple[float, float],
            tuple[str | None, ...],
        ]
    ],
) -> tuple[_ModelParameterSpec, ...]:
    result: list[_ModelParameterSpec] = []
    seen: set[str] = set()

    def add(spec: _ModelParameterSpec) -> None:
        if spec.name in seen:
            return
        seen.add(spec.name)
        result.append(spec)

    def add_complex(
        name: str,
        value: object,
        *,
        kind: str,
        derived: bool = False,
        domain: str | None = None,
        definition: str | None = None,
    ) -> None:
        pair = _complex_default(value, name)
        for component, default in enumerate(pair):
            add(
                _ModelParameterSpec(
                    name=f"{name}.{'real' if component == 0 else 'imag'}",
                    kind=kind,
                    default=default,
                    runtime_name=name,
                    complex_component=component,
                    derived=derived,
                    complex_domain=domain,
                    definition=definition,
                )
            )

    for name, value in sorted(model.runtime_normalization_parameter_defaults().items()):
        add(
            _ModelParameterSpec(
                name=str(name), kind="normalization", default=float(value)
            )
        )

    defaults_provider = getattr(model, "runtime_parameter_defaults", None)
    if callable(defaults_provider):
        type_provider = getattr(model, "runtime_parameter_type", None)
        for raw_name, value in sorted(defaults_provider().items()):
            name = str(raw_name)
            declared = (
                str(type_provider(name)).lower()
                if callable(type_provider)
                else "complex"
            )
            if declared == "complex":
                add_complex(name, value, kind="external_parameter_component")
            else:
                real, imaginary = _complex_default(value, name)
                if imaginary != 0.0:
                    raise EagerColumnarInputError(
                        f"real model parameter {name!r} has imaginary default"
                    )
                add(
                    _ModelParameterSpec(
                        name=name,
                        kind="external_parameter",
                        default=real,
                        parameter_type=declared,
                    )
                )
        used = {name for *_prefix, names in couplings for name in names if name}
        particle_names = getattr(model, "runtime_parameter_names_for_particle", None)
        if callable(particle_names):
            for particle_id in sorted(
                {current.index.particle_id for current in dag.currents}
            ):
                used.update(str(name) for name in particle_names(particle_id))
        derived_defaults = getattr(
            model, "runtime_derived_parameter_defaults_for", None
        )
        if callable(derived_defaults):
            names = tuple(sorted(used))
            values = derived_defaults(names)
            domains_provider = getattr(
                model, "runtime_derived_parameter_domains_for", None
            )
            domains = domains_provider(names) if callable(domains_provider) else {}
            definitions_provider = getattr(
                model, "runtime_derived_parameter_definitions_for", None
            )
            definitions = (
                definitions_provider(names) if callable(definitions_provider) else {}
            )
            for raw_name, value in sorted(values.items()):
                name = str(raw_name)
                add_complex(
                    name,
                    value,
                    kind="derived_parameter_component",
                    derived=True,
                    domain=str(domains.get(raw_name, domains.get(name, "complex"))),
                    definition=(
                        None
                        if definitions.get(raw_name, definitions.get(name)) is None
                        else str(definitions.get(raw_name, definitions.get(name)))
                    ),
                )
        return tuple(result)

    for particle in sorted(model.particles.values(), key=lambda item: item.pdg):
        if float(particle.mass) != 0.0:
            add(
                _ModelParameterSpec(
                    name=f"particle.{particle.pdg}.mass",
                    kind="particle_mass",
                    default=float(particle.mass),
                    pdg=particle.pdg,
                )
            )
        if float(particle.width) != 0.0:
            add(
                _ModelParameterSpec(
                    name=f"particle.{particle.pdg}.width",
                    kind="particle_width",
                    default=float(particle.width),
                    pdg=particle.pdg,
                )
            )
    for kind, particles, coupling, names in sorted(couplings):
        for component, name in enumerate(names):
            if name is None:
                continue
            add(
                _ModelParameterSpec(
                    name=name,
                    kind="coupling_component",
                    default=float(coupling[component]),
                    vertex_kind=kind,
                    vertex_particles=particles,
                    coupling_component=component,
                )
            )
    return tuple(result)


def _possible_helicity_vectors(
    dag: GenericDAG, model: Model
) -> tuple[tuple[int, ...], ...]:
    selected = dict(dag.selected_source_helicities)
    per_leg: list[tuple[int, ...]] = []
    for leg in dag.process.legs:
        if leg.outgoing_pdg is None:
            per_leg.append((0,))
            continue
        source_ir = model._source_ir(int(leg.outgoing_pdg))
        values = {
            int((source_ir.crossing.apply(state) if leg.is_initial else state).helicity)
            for state in source_ir.states
        }
        if leg.label in selected:
            values.intersection_update((selected[leg.label],))
            if not values:
                raise EagerColumnarInputError(
                    f"selected helicity is unavailable for leg {leg.label}"
                )
        per_leg.append(tuple(sorted(values)))
    return tuple(tuple(values) for values in product(*per_leg))


def _descriptor_helicity_vector(
    dag: GenericDAG, helicity_key: tuple[object, ...]
) -> tuple[int, ...]:
    by_label: dict[int, int] = {}
    for item in helicity_key:
        if isinstance(item, tuple) and len(item) >= 5:
            by_label[int(item[0])] = int(item[4])
    return tuple(by_label.get(leg.label, 0) for leg in dag.process.legs)


def _fill_resolved_kernel(
    columns: Mapping[str, np.ndarray[Any, Any]],
    row: int,
    resolved: EagerResolvedKernel,
    factors: _FactorCatalog,
) -> None:
    columns["kernel_id"][row] = _checked_u32(resolved.kernel_id, "prepared kernel id")
    columns["canonical_input_order"][row] = resolved.canonical_input_order
    columns["kernel_normalization_factor_id"][row] = factors.add(
        resolved.normalization_factor, "prepared kernel normalization"
    )
    columns["output_factor_source"][row] = _checked_u8(
        resolved.output_factor_source, "prepared output factor source"
    )


def _allocate(
    row_count: int,
    **specifications: np.dtype[Any] | tuple[np.dtype[Any], tuple[int, ...]],
) -> dict[str, np.ndarray[Any, Any]]:
    _checked_u64(row_count, "table row count")
    result: dict[str, np.ndarray[Any, Any]] = {}
    for name, specification in specifications.items():
        if isinstance(specification, tuple):
            dtype, tail_shape = specification
        else:
            dtype, tail_shape = specification, ()
        result[name] = np.empty((row_count, *tail_shape), dtype=dtype, order="C")
    return result


def _freeze_table(
    name: str,
    columns: Mapping[str, np.ndarray[Any, Any]],
) -> EagerColumnarTable:
    row_counts = {len(values) for values in columns.values()}
    row_count = row_counts.pop() if row_counts else 0
    if row_counts:
        raise EagerColumnarInputError(f"table {name!r} columns have unequal lengths")
    frozen: list[EagerColumn] = []
    for column_name, values in columns.items():
        values.flags.writeable = False
        frozen.append(EagerColumn(column_name, values))
    return EagerColumnarTable(name, row_count, tuple(frozen))


def _validate_catalog(values: tuple[str, ...], context: str) -> None:
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"{context} catalog must contain strings")
    if len(set(values)) != len(values):
        raise EagerColumnarInputError(f"{context} catalog contains duplicates")


def _validate_dense_ids(table: EagerColumnarTable, column_name: str) -> None:
    values = table.column(column_name)
    expected = np.arange(table.row_count, dtype=_U32)
    if not np.array_equal(values, expected):
        raise EagerColumnarInputError(f"table {table.name!r} ids are not dense")


def _validate_bounded_columns(
    table: EagerColumnarTable,
    columns: Sequence[str],
    bound: int,
) -> None:
    for column_name in columns:
        values = table.column(column_name)
        if values.size and int(values.max()) >= bound:
            raise EagerColumnarInputError(
                f"{table.name}.{column_name} references an absent row"
            )


def _validate_optional_bounded_columns(
    table: EagerColumnarTable,
    columns: Sequence[str],
    bound: int,
) -> None:
    for column_name in columns:
        values = table.column(column_name)
        present = values[values != MISSING_U32]
        if present.size and int(present.max()) >= bound:
            raise EagerColumnarInputError(
                f"{table.name}.{column_name} references an absent row"
            )


def _require_index(value: int, bound: int, context: str) -> int:
    result = _checked_u32(value, context)
    if result >= bound:
        raise EagerColumnarInputError(f"{context} {result} is outside [0, {bound})")
    return result


def _checked_u8(value: int, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 256:
        raise EagerColumnarInputError(f"{context} does not fit u8: {value!r}")
    return value


def _checked_u32(value: int, context: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MISSING_U32
    ):
        raise EagerColumnarInputError(f"{context} does not fit u32: {value!r}")
    return value


def _positive_u32(value: int, context: str) -> int:
    result = _checked_u32(value, context)
    if result == 0:
        raise EagerColumnarInputError(f"{context} must be positive")
    return result


def _checked_u64(value: int, context: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < 1 << 64
    ):
        raise EagerColumnarInputError(f"{context} does not fit u64: {value!r}")
    return value


def _checked_i32(value: int, context: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not -(1 << 31) <= value < 1 << 31
    ):
        raise EagerColumnarInputError(f"{context} does not fit i32: {value!r}")
    return value


def _finite(value: float, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise EagerColumnarInputError(f"{context} must be finite")
    return result


def _complex_default(value: object, context: str) -> tuple[float, float]:
    if isinstance(value, complex):
        pair = (float(value.real), float(value.imag))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if len(value) != 2:
            raise EagerColumnarInputError(f"{context} must be a complex pair")
        pair = (float(value[0]), float(value[1]))
    else:
        pair = (float(value), 0.0)
    if not all(math.isfinite(component) for component in pair):
        raise EagerColumnarInputError(f"{context} must be finite")
    return pair


def _f64_bits(value: float) -> int:
    """Return the exact little-endian IEEE-754 payload for one finite f64."""

    return int.from_bytes(struct.pack("<d", value), "little", signed=False)


def _binary64_exact_ir(value: tuple[float, float]) -> dict[str, object]:
    """Canonical exact representation of a complex value sourced as f64."""

    return {
        "abi": "pyamplicol-exact-factor-v1",
        "kind": "complex-ieee754-binary64",
        "real_bits": f"{_f64_bits(value[0]):016x}",
        "imaginary_bits": f"{_f64_bits(value[1]):016x}",
    }


def _color_weight_ir(
    entry: ColorContractionEntry | ColorContractionTemplateEntry,
) -> dict[str, object] | None:
    if entry.exact_weight is None:
        return None
    return {"kind": "rational-complex", "value": entry.to_json_dict()["exact_weight"]}


def _hash_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


__all__ = [
    "EAGER_LOWERING_INPUT_ABI",
    "EAGER_LOWERING_INPUT_SCHEMA_SHA256",
    "FACTOR_EXACT_SOURCE_BINARY64",
    "FACTOR_EXACT_SOURCE_CANONICAL_IR",
    "EagerColumn",
    "EagerColumnarInputError",
    "EagerColumnarTable",
    "EagerLoweringInputV1",
    "build_eager_lowering_input_v1",
    "runtime_model_parameter_records_from_eager_input",
]
