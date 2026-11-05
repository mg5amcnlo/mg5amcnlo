# SPDX-License-Identifier: 0BSD
"""Compact Python-to-Rust projection of recurrence-template-v1.

The prepared recurrence catalog is model-wide, while
``recurrence_columnar`` is process-specific.  Keeping the two authenticated
inputs separate lets a process set reuse one decoded model catalog without
copying it into every process builder input.
"""

from __future__ import annotations

import hashlib
import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from ..models.recurrence_template import (
    ExactComplexRationalV1,
    RecurrenceTemplateCatalog,
)
from .recurrence_columnar import (
    MISSING_U32,
    RecurrenceColumn,
    RecurrenceColumnarInputError,
    RecurrenceColumnarTable,
)

RECURRENCE_TEMPLATE_INPUT_ABI: Final = "pyamplicol-recurrence-template-input-v1"
RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION: Final = 2

_U8 = np.dtype("u1")
_U32 = np.dtype("<u4")
_U64 = np.dtype("<u8")
_I32 = np.dtype("<i4")

_PARAMETER_KIND = {"external": 0, "derived": 1, "constant": 2}
_PARAMETER_VALUE_TYPE = {"real": 0, "complex": 1}
_ORIENTATION = {"particle": 0, "antiparticle": 1, "self-conjugate": 2}
_STATISTICS = {"boson": 0, "fermion": 1}
_CONTRACT_KIND = {
    "source": 0,
    "vertex": 1,
    "propagator": 2,
    "closure": 3,
    "model-parameter": 4,
}
_CALLABLE_KIND = {"prepared-kernel": 0, "rusticol-template": 1}
_OUTPUT_FACTOR_SOURCE = {"none": 0, "coupling-real": 1, "coupling-imag": 2}
_LC_COLOR_COMPONENT_OPERATION = {
    "concatenate-join": 0,
    "concatenate-keep": 1,
    "inherit-left": 2,
    "inherit-right": 3,
    "empty": 4,
    "close": 5,
}
_LC_COLOR_COMPONENT_KIND = {"open-string": 0, "adjoint-segment": 1, "trace": 2}
_LC_COLOR_COMPONENT_ROLE = {"active": 0, "passive": 1, "none": 2}
_LC_COLOR_SOURCE_SEED_OPERATION = {"empty": 0, "singleton": 1}
_I128_MAX = (1 << 127) - 1
_I128_MIN = -_I128_MAX


@dataclass(frozen=True, slots=True)
class RecurrenceTemplateInputV1:
    """Owned immutable model-wide recurrence columns."""

    abi: str
    catalog_digest: str
    compiled_model_digest: str
    prepared_kernel_pack_digest: str
    tables: tuple[RecurrenceColumnarTable, ...]

    def __post_init__(self) -> None:
        if self.abi != RECURRENCE_TEMPLATE_INPUT_ABI:
            raise RecurrenceColumnarInputError(
                f"unsupported recurrence template input ABI {self.abi!r}"
            )
        _require_sha256(self.catalog_digest, "recurrence catalog digest")
        _require_sha256(self.compiled_model_digest, "compiled-model digest")
        _require_sha256(self.prepared_kernel_pack_digest, "prepared kernel pack digest")
        names = tuple(table.name for table in self.tables)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise RecurrenceColumnarInputError(
                "recurrence template tables must be uniquely named and sorted"
            )

    @property
    def canonical_digest(self) -> str:
        digest = hashlib.sha256()
        _hash_text(digest, self.abi)
        digest.update(bytes.fromhex(self.catalog_digest))
        digest.update(bytes.fromhex(self.compiled_model_digest))
        digest.update(bytes.fromhex(self.prepared_kernel_pack_digest))
        digest.update(len(self.tables).to_bytes(8, "little"))
        for table in self.tables:
            _hash_text(digest, table.name)
            digest.update(table.row_count.to_bytes(8, "little"))
            digest.update(len(table.columns).to_bytes(4, "little"))
            for column in table.columns:
                _hash_text(digest, column.name)
                _hash_text(digest, column.values.dtype.str)
                digest.update(len(column.values.shape).to_bytes(1, "little"))
                for dimension in column.values.shape:
                    digest.update(int(dimension).to_bytes(8, "little"))
                digest.update(column.values.tobytes(order="C"))
        return digest.hexdigest()

    @property
    def digest(self) -> str:
        return self.canonical_digest


class _StringCatalog:
    def __init__(self, values: Iterable[str]) -> None:
        self.values = tuple(sorted(set(values)))
        self._ids = {value: index for index, value in enumerate(self.values)}

    def id(self, value: str | None) -> int:
        if value is None:
            return MISSING_U32
        return self._ids[value]


class _DigestCatalog:
    def __init__(self, values: Iterable[str]) -> None:
        self.values = tuple(sorted(set(values)))
        self._ids = {value: index for index, value in enumerate(self.values)}

    def id(self, value: str | None) -> int:
        if value is None:
            return MISSING_U32
        return self._ids[value]


class _FactorCatalog:
    def __init__(self, values: Iterable[ExactComplexRationalV1]) -> None:
        by_key = {_factor_key(value): value for value in values}
        self.values = tuple(by_key[key] for key in sorted(by_key))
        self._ids = {
            _factor_key(value): index for index, value in enumerate(self.values)
        }

    def id(self, value: ExactComplexRationalV1 | None) -> int:
        if value is None:
            return MISSING_U32
        return self._ids[_factor_key(value)]


class _SequenceCatalog:
    def __init__(self, values: Iterable[Sequence[int]]) -> None:
        self.values = tuple(
            sorted(set(tuple(int(item) for item in value) for value in values))
        )
        self._ids = {value: index for index, value in enumerate(self.values)}

    def id(self, values: Sequence[int]) -> int:
        return self._ids[tuple(int(value) for value in values)]


class _QuantumNumberFlowCatalog:
    def __init__(self, values: Iterable[Sequence[tuple[str, str]]]) -> None:
        self.values = tuple(
            sorted(
                set(
                    tuple((str(name), str(expression)) for name, expression in value)
                    for value in values
                )
            )
        )
        self._ids = {value: index for index, value in enumerate(self.values)}

    def id(self, values: Sequence[tuple[str, str]]) -> int:
        key = tuple((str(name), str(expression)) for name, expression in values)
        return self._ids[key]


class _CouplingOrderCatalog:
    def __init__(self, values: Iterable[Sequence[tuple[str, int]]]) -> None:
        self.values = tuple(
            sorted(
                set(
                    tuple((str(name), int(power)) for name, power in value)
                    for value in values
                )
            )
        )
        self._ids = {value: index for index, value in enumerate(self.values)}

    def id(self, values: Sequence[tuple[str, int]]) -> int:
        key = tuple((str(name), int(power)) for name, power in values)
        return self._ids[key]


def build_recurrence_template_input_v1(
    catalog: RecurrenceTemplateCatalog,
) -> RecurrenceTemplateInputV1:
    """Flatten a validated semantic catalog into deterministic primitive columns."""

    if not isinstance(catalog, RecurrenceTemplateCatalog):
        raise TypeError("recurrence template extraction requires a validated catalog")
    _validate_i128_factors(catalog)

    sections = {
        "parameters": tuple(catalog.parameters),
        "current_states": tuple(catalog.current_states),
        "sources": tuple(catalog.sources),
        "quantum_flows": tuple(catalog.quantum_flows),
        "contact_orbit_certificates": tuple(catalog.contact_orbit_certificates),
        "contact_orbit_steps": tuple(catalog.contact_orbit_steps),
        "transitions": tuple(catalog.transitions),
        "propagators": tuple(catalog.propagators),
        "closures": tuple(catalog.closures),
        "color_contractions": tuple(catalog.color_contractions),
        "symmetry_proofs": tuple(catalog.symmetry_proofs),
        "runtime_helicity_contracts": tuple(catalog.runtime_helicity_contracts),
        "evaluator_bindings": tuple(catalog.evaluator_bindings),
    }
    ids = {
        name: {
            _record_identity(record, name): index
            for index, record in enumerate(records)
        }
        for name, records in sections.items()
    }

    strings = _StringCatalog(_all_strings(catalog))
    digests = _DigestCatalog(_all_digests(catalog))
    factors = _FactorCatalog(_all_factors(catalog))
    flavour_flows = _SequenceCatalog(_all_flavour_flows(catalog))
    quantum_number_flows = _QuantumNumberFlowCatalog(_all_quantum_number_flows(catalog))
    u32_sequences = _SequenceCatalog(
        _all_u32_sequences(
            catalog,
            ids,
            strings,
            digests,
            factors,
            flavour_flows,
            quantum_number_flows,
        )
    )
    i32_sequences = _SequenceCatalog(_all_i32_sequences(catalog))
    coupling_orders = _CouplingOrderCatalog(_all_coupling_orders(catalog))
    string_ranges, string_bytes = _string_tables(strings)

    tables = [
        _header_table(catalog, strings, digests, sections),
        _catalog_ranges("coupling_order", coupling_orders.values, 2),
        _coupling_order_terms(coupling_orders, strings),
        _contact_orbit_certificates_table(
            catalog, strings, digests, factors, u32_sequences
        ),
        _contact_orbit_steps_table(
            catalog, ids, strings, digests, factors, u32_sequences, i32_sequences
        ),
        _current_states_table(catalog, ids, strings, digests, u32_sequences),
        _digest_table(digests),
        _evaluator_bindings_table(catalog, ids, strings, digests, u32_sequences),
        _exact_factor_table(factors, strings),
        *_flatten_sequence_tables("flavour_flow", flavour_flows, _I32),
        *_flatten_sequence_tables("i32_sequence", i32_sequences, _I32),
        _parameters_table(catalog, ids, strings, digests, factors, u32_sequences),
        _propagators_table(catalog, ids, strings, digests),
        _quantum_flows_table(
            catalog,
            ids,
            strings,
            digests,
            factors,
            u32_sequences,
            i32_sequences,
            coupling_orders,
            flavour_flows,
            quantum_number_flows,
        ),
        _quantum_number_flow_ranges(quantum_number_flows),
        _quantum_number_flow_terms(quantum_number_flows, strings),
        *_runtime_helicity_tables(catalog, ids, strings, digests, factors),
        _sources_table(
            catalog,
            ids,
            strings,
            digests,
            flavour_flows,
            quantum_number_flows,
            u32_sequences,
        ),
        string_ranges,
        string_bytes,
        _symmetry_proofs_table(catalog, strings, digests, factors, u32_sequences),
        _transitions_table(
            catalog,
            ids,
            strings,
            digests,
            factors,
            u32_sequences,
            coupling_orders,
        ),
        _closures_table(
            catalog,
            ids,
            strings,
            digests,
            factors,
            u32_sequences,
            coupling_orders,
        ),
        _color_contractions_table(catalog, strings, digests, factors, i32_sequences),
        _lc_color_transition_witnesses_table(
            catalog,
            strings,
            digests,
            factors,
            u32_sequences,
        ),
        _color_nc_terms_table(catalog, factors),
        *_flatten_sequence_tables("u32_sequence", u32_sequences, _U32),
    ]
    tables.sort(key=lambda table: table.name)
    return RecurrenceTemplateInputV1(
        abi=RECURRENCE_TEMPLATE_INPUT_ABI,
        catalog_digest=catalog.header.catalog_digest,
        compiled_model_digest=catalog.header.compiled_model_digest,
        prepared_kernel_pack_digest=catalog.header.prepared_kernel_pack_digest,
        tables=tuple(tables),
    )


def _record_identity(record: object, section: str) -> str:
    attribute = "resolver_key" if section == "evaluator_bindings" else "template_id"
    return str(getattr(record, attribute))


def _all_strings(catalog: RecurrenceTemplateCatalog) -> Iterable[str]:
    yield RECURRENCE_TEMPLATE_INPUT_ABI
    yield catalog.header.abi
    yield catalog.header.canonicalization_abi
    yield catalog.header.exact_scalar_abi
    for record in catalog.parameters:
        yield from (
            record.template_id,
            record.name,
            *record.dependency_parameter_ids,
        )
    for record in catalog.current_states:
        yield from (
            record.template_id,
            record.species_id,
            record.basis,
            record.lc_color_shape_kind,
            *record.tensor_ordering,
        )
        yield from _present(
            record.auxiliary_kind, record.mass_parameter_id, record.width_parameter_id
        )
    for record in catalog.sources:
        yield from (
            record.template_id,
            record.crossing,
            record.wavefunction_family,
            record.evaluator_resolver_key,
            record.lc_color_seed.output_shape_kind,
        )
        for key, value in record.lc_color_seed.provenance:
            yield key
            yield value
        for name, expression in record.quantum_number_flow:
            yield name
            yield expression
    for record in catalog.quantum_flows:
        yield record.template_id
        yield record.flavour_flow_operation
        yield record.quantum_number_flow_operation
        for flow in (
            *record.input_quantum_number_flows,
            record.result_quantum_number_flow,
        ):
            for name, expression in flow:
                yield name
                yield expression
        yield from (name for name, _ in record.coupling_orders)
    for record in catalog.contact_orbit_certificates:
        yield from (
            record.template_id,
            record.algorithm,
            record.vertex,
            *record.particles,
            record.evaluator_class,
        )
    for record in catalog.contact_orbit_steps:
        yield from (
            record.template_id,
            record.certificate_template_id,
            record.stage,
        )
    for record in catalog.transitions:
        yield from (
            record.template_id,
            record.evaluator_resolver_key,
            *record.momentum_convention,
            record.output_factor_source,
            record.equivalence_class,
            record.output_projection,
        )
        yield from (name for name, _ in record.coupling_orders)
    for record in catalog.propagators:
        yield record.template_id
        yield from _present(record.evaluator_resolver_key, record.gauge)
    for record in catalog.closures:
        yield from (
            record.template_id,
            record.evaluator_resolver_key,
            record.output_factor_source,
            record.equivalence_class,
            record.projection,
            record.chirality_relation,
        )
        yield from _present(record.metric_signature)
        yield from (name for name, _ in record.coupling_orders)
    for record in catalog.color_contractions:
        yield from (record.template_id, record.rule_kind)
        for witness in record.transition_witnesses:
            yield from witness.input_shape_kinds
            yield from _present(witness.result_shape_kind)
            for key, value in witness.provenance:
                yield key
                yield value
    for record in catalog.symmetry_proofs:
        yield from (
            record.template_id,
            record.proof_algorithm,
            *record.subject_template_ids,
        )
    for record in catalog.runtime_helicity_contracts:
        yield from (
            record.template_id,
            record.full_state_template_id,
            record.proof_algorithm,
        )
        for variant in record.variants:
            yield variant.source_template_id
            yield variant.source_state_template_id
    for record in catalog.evaluator_bindings:
        yield from (
            record.resolver_key,
            *record.input_layout,
            *record.output_layout,
            *record.semantic_template_ids,
        )
        yield from _present(record.runtime_template)
    for factor in _all_factors(catalog):
        yield from (str(value) for value in _factor_key(factor))


def _all_digests(catalog: RecurrenceTemplateCatalog) -> Iterable[str]:
    yield catalog.header.catalog_digest
    yield catalog.header.compiled_model_digest
    yield catalog.header.prepared_kernel_pack_digest
    for records in (
        catalog.parameters,
        catalog.current_states,
        catalog.sources,
        catalog.quantum_flows,
        catalog.contact_orbit_certificates,
        catalog.contact_orbit_steps,
        catalog.transitions,
        catalog.propagators,
        catalog.closures,
        catalog.color_contractions,
        catalog.symmetry_proofs,
        catalog.runtime_helicity_contracts,
        catalog.evaluator_bindings,
    ):
        for record in records:
            yield record.semantic_digest
    for record in catalog.parameters:
        yield from _present(record.exact_expression_digest)
    for record in catalog.sources:
        yield record.wavefunction_expression_digest
        yield record.lc_color_seed.proof_digest
    for record in catalog.quantum_flows:
        yield record.predicate_digest
    for record in catalog.propagators:
        yield from _present(
            record.numerator_expression_digest, record.denominator_expression_digest
        )
    for record in catalog.color_contractions:
        yield record.expression_digest
        yield from (witness.proof_digest for witness in record.transition_witnesses)
    for record in catalog.symmetry_proofs:
        yield record.witness_digest
        yield from record.expression_digests
    for record in catalog.runtime_helicity_contracts:
        yield record.proof_digest
        yield from (variant.proof_digest for variant in record.variants)
    for record in catalog.evaluator_bindings:
        yield record.callable_signature
        yield from record.exact_expression_digests


def _all_factors(
    catalog: RecurrenceTemplateCatalog,
) -> Iterable[ExactComplexRationalV1]:
    for record in catalog.parameters:
        if record.default_value is not None:
            yield record.default_value
    for record in catalog.transitions:
        yield record.binding_coupling
        yield record.exact_factor
        if record.input_exchange_factor is not None:
            yield record.input_exchange_factor
    for record in catalog.contact_orbit_certificates:
        yield record.reconstruction_factor
    for record in catalog.contact_orbit_steps:
        yield record.reconstruction_factor
    for record in catalog.quantum_flows:
        yield record.exact_coupling
    for record in catalog.closures:
        yield record.binding_coupling
        yield record.exact_factor
        if record.input_exchange_factor is not None:
            yield record.input_exchange_factor
        yield from record.component_coefficients
    for record in catalog.color_contractions:
        yield record.exact_coefficient
        yield from (factor for _, factor in record.nc_polynomial)
        yield from (witness.exact_factor for witness in record.transition_witnesses)
    for record in catalog.symmetry_proofs:
        yield record.exact_phase
    for record in catalog.runtime_helicity_contracts:
        for variant in record.variants:
            yield from variant.embedding_factors


def _all_coupling_orders(
    catalog: RecurrenceTemplateCatalog,
) -> Iterable[Sequence[tuple[str, int]]]:
    yield ()
    yield from (record.coupling_orders for record in catalog.quantum_flows)
    yield from (record.coupling_orders for record in catalog.transitions)
    yield from (record.coupling_orders for record in catalog.closures)


def _all_u32_sequences(
    catalog: RecurrenceTemplateCatalog,
    ids: dict[str, dict[str, int]],
    strings: _StringCatalog,
    digests: _DigestCatalog,
    factors: _FactorCatalog,
    flavour_flows: _SequenceCatalog,
    quantum_number_flows: _QuantumNumberFlowCatalog,
) -> Iterable[Sequence[int]]:
    yield ()
    for record in catalog.parameters:
        yield tuple(
            ids["parameters"][value] for value in record.dependency_parameter_ids
        )
    for record in catalog.current_states:
        yield tuple(strings.id(value) for value in record.tensor_ordering)
    for record in catalog.sources:
        yield tuple(
            strings.id(item)
            for pair in record.lc_color_seed.provenance
            for item in pair
        )
    for record in catalog.quantum_flows:
        yield tuple(
            ids["current_states"][value] for value in record.input_state_template_ids
        )
        yield tuple(flavour_flows.id(value) for value in record.input_flavour_flows)
        yield tuple(
            quantum_number_flows.id(value)
            for value in record.input_quantum_number_flows
        )
    for record in catalog.transitions:
        yield tuple(
            ids["current_states"][value] for value in record.input_state_template_ids
        )
        yield record.canonical_input_order
        yield tuple(strings.id(value) for value in record.momentum_convention)
        yield tuple(ids["parameters"][value] for value in record.coupling_parameter_ids)
        yield tuple(
            ids["contact_orbit_steps"][value]
            for value in record.contact_orbit_step_template_ids
        )
        yield tuple(
            digests.id(value)
            for value in record.contact_orbit_step_semantic_digests
        )
    for record in catalog.contact_orbit_certificates:
        yield tuple(strings.id(value) for value in record.particles)
        yield record.physical_leg_equivalence_classes
    for record in catalog.contact_orbit_steps:
        yield record.left_covered_legs
        yield record.right_covered_legs
    for record in catalog.closures:
        yield tuple(
            ids["current_states"][value] for value in record.input_state_template_ids
        )
        yield record.canonical_input_order
        yield tuple(ids["parameters"][value] for value in record.coupling_parameter_ids)
        yield tuple(
            ids["quantum_flows"][value]
            for value in record.eligible_quantum_flow_template_ids
        )
        yield tuple(factors.id(value) for value in record.component_coefficients)
    for record in catalog.color_contractions:
        for witness in record.transition_witnesses:
            yield tuple(
                value
                for pairing in witness.input_port_pairings
                for port in pairing
                for value in port
            )
            yield tuple(
                value for port in witness.result_port_bindings for value in port
            )
            yield tuple(
                strings.id(item) for pair in witness.provenance for item in pair
            )
    for record in catalog.symmetry_proofs:
        yield tuple(strings.id(value) for value in record.subject_template_ids)
        yield record.input_permutation
        yield tuple(digests.id(value) for value in record.expression_digests)
    for record in catalog.evaluator_bindings:
        yield tuple(
            ids["current_states"][value] for value in record.input_state_template_ids
        )
        yield tuple(strings.id(value) for value in record.input_layout)
        yield tuple(strings.id(value) for value in record.output_layout)
        yield tuple(digests.id(value) for value in record.exact_expression_digests)
        yield tuple(strings.id(value) for value in record.semantic_template_ids)


def _all_i32_sequences(catalog: RecurrenceTemplateCatalog) -> Iterable[Sequence[int]]:
    yield ()
    yield from (record.input_spin_states for record in catalog.quantum_flows)
    yield from (record.input_representations for record in catalog.color_contractions)
    yield from (
        record.source_particle_legs for record in catalog.contact_orbit_steps
    )


def _all_flavour_flows(
    catalog: RecurrenceTemplateCatalog,
) -> Iterable[Sequence[int]]:
    yield from (record.flavour_flow for record in catalog.sources)
    for record in catalog.quantum_flows:
        yield from record.input_flavour_flows
        yield record.result_flavour_flow


def _all_quantum_number_flows(
    catalog: RecurrenceTemplateCatalog,
) -> Iterable[Sequence[tuple[str, str]]]:
    yield from (record.quantum_number_flow for record in catalog.sources)
    for record in catalog.quantum_flows:
        yield from record.input_quantum_number_flows
        yield record.result_quantum_number_flow


def _header_table(catalog, strings, digests, sections):
    names = (
        "schema_version",
        "abi_string_id",
        "canonicalization_abi_string_id",
        "exact_scalar_abi_string_id",
        "compiled_model_digest_id",
        "prepared_kernel_pack_digest_id",
        "catalog_digest_id",
        "parameter_count",
        "current_state_count",
        "source_count",
        "quantum_flow_count",
        "contact_orbit_certificate_count",
        "contact_orbit_step_count",
        "transition_count",
        "propagator_count",
        "closure_count",
        "color_contraction_count",
        "symmetry_proof_count",
        "runtime_helicity_contract_count",
        "evaluator_binding_count",
    )
    values = (
        RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION,
        strings.id(catalog.header.abi),
        strings.id(catalog.header.canonicalization_abi),
        strings.id(catalog.header.exact_scalar_abi),
        digests.id(catalog.header.compiled_model_digest),
        digests.id(catalog.header.prepared_kernel_pack_digest),
        digests.id(catalog.header.catalog_digest),
        *(len(sections[name]) for name in sections),
    )
    return _table(
        "catalog_header",
        {
            name: _array([value], _U32)
            for name, value in zip(names, values, strict=True)
        },
    )


def _parameters_table(catalog, ids, strings, digests, factors, sequences):
    rows = []
    for index, record in enumerate(catalog.parameters):
        rows.append(
            (
                index,
                strings.id(record.template_id),
                strings.id(record.name),
                _PARAMETER_KIND[record.parameter_kind],
                _PARAMETER_VALUE_TYPE[record.value_type],
                int(record.mutable),
                factors.id(record.default_value),
                digests.id(record.exact_expression_digest),
                sequences.id(
                    tuple(
                        ids["parameters"][value]
                        for value in record.dependency_parameter_ids
                    )
                ),
                MISSING_U32
                if record.prepared_parameter_id is None
                else record.prepared_parameter_id,
                digests.id(record.semantic_digest),
            )
        )
    return _rows(
        "parameters",
        rows,
        (
            ("id", _U32),
            ("template_string_id", _U32),
            ("name_string_id", _U32),
            ("kind", _U8),
            ("value_type", _U8),
            ("mutable", _U8),
            ("default_factor_id", _U32),
            ("exact_expression_digest_id", _U32),
            ("dependency_sequence_id", _U32),
            ("prepared_parameter_id", _U32),
            ("semantic_digest_id", _U32),
        ),
    )


def _current_states_table(catalog, ids, strings, digests, sequences):
    del ids
    rows = []
    parameter_ids = {
        record.template_id: i for i, record in enumerate(catalog.parameters)
    }
    for index, record in enumerate(catalog.current_states):
        rows.append(
            (
                index,
                strings.id(record.template_id),
                record.particle_id,
                record.anti_particle_id,
                strings.id(record.species_id),
                _ORIENTATION[record.orientation],
                _STATISTICS[record.statistics],
                record.color_representation,
                strings.id(record.basis),
                sequences.id(
                    tuple(strings.id(value) for value in record.tensor_ordering)
                ),
                record.dimension,
                record.chirality,
                strings.id(record.lc_color_shape_kind),
                strings.id(record.auxiliary_kind),
                _optional_reference(record.mass_parameter_id, parameter_ids),
                _optional_reference(record.width_parameter_id, parameter_ids),
                digests.id(record.semantic_digest),
            )
        )
    return _rows(
        "current_states",
        rows,
        (
            ("id", _U32),
            ("template_string_id", _U32),
            ("particle_id", _I32),
            ("anti_particle_id", _I32),
            ("species_string_id", _U32),
            ("orientation", _U8),
            ("statistics", _U8),
            ("color_representation", _I32),
            ("basis_string_id", _U32),
            ("tensor_ordering_sequence_id", _U32),
            ("dimension", _U32),
            ("chirality", _I32),
            ("lc_color_shape_string_id", _U32),
            ("auxiliary_kind_string_id", _U32),
            ("mass_parameter_id", _U32),
            ("width_parameter_id", _U32),
            ("semantic_digest_id", _U32),
        ),
    )


def _runtime_helicity_tables(catalog, ids, strings, digests, factors):
    contract_rows = []
    variant_rows = []
    embedding_rows = []
    projection_rows = []
    variant_offset = 0
    embedding_offset = 0
    projection_offset = 0
    for contract_id, contract in enumerate(catalog.runtime_helicity_contracts):
        contract_rows.append(
            (
                contract_id,
                strings.id(contract.template_id),
                ids["current_states"][contract.full_state_template_id],
                variant_offset,
                len(contract.variants),
                strings.id(contract.proof_algorithm),
                digests.id(contract.proof_digest),
                digests.id(contract.semantic_digest),
            )
        )
        for variant in contract.variants:
            variant_id = len(variant_rows)
            variant_rows.append(
                (
                    variant_id,
                    contract_id,
                    ids["sources"][variant.source_template_id],
                    ids["current_states"][variant.source_state_template_id],
                    embedding_offset,
                    len(variant.embedding_source_components),
                    projection_offset,
                    len(variant.projection_full_components),
                    digests.id(variant.proof_digest),
                )
            )
            for full_component, (source_component, factor) in enumerate(
                zip(
                    variant.embedding_source_components,
                    variant.embedding_factors,
                    strict=True,
                )
            ):
                embedding_rows.append(
                    (
                        variant_id,
                        full_component,
                        MISSING_U32 if source_component is None else source_component,
                        factors.id(factor),
                    )
                )
            for source_component, full_component in enumerate(
                variant.projection_full_components
            ):
                projection_rows.append((variant_id, source_component, full_component))
            embedding_offset += len(variant.embedding_source_components)
            projection_offset += len(variant.projection_full_components)
        variant_offset += len(contract.variants)
    return (
        _rows(
            "runtime_helicity_contracts",
            contract_rows,
            (
                ("id", _U32),
                ("template_string_id", _U32),
                ("full_state_template_id", _U32),
                ("variant_offset", _U32),
                ("variant_count", _U32),
                ("proof_algorithm_string_id", _U32),
                ("proof_digest_id", _U32),
                ("semantic_digest_id", _U32),
            ),
        ),
        _rows(
            "runtime_helicity_variants",
            variant_rows,
            (
                ("id", _U32),
                ("contract_id", _U32),
                ("source_template_id", _U32),
                ("source_state_template_id", _U32),
                ("embedding_offset", _U32),
                ("embedding_count", _U32),
                ("projection_offset", _U32),
                ("projection_count", _U32),
                ("proof_digest_id", _U32),
            ),
        ),
        _rows(
            "runtime_helicity_embeddings",
            embedding_rows,
            (
                ("variant_id", _U32),
                ("full_component", _U32),
                ("source_component", _U32),
                ("factor_id", _U32),
            ),
        ),
        _rows(
            "runtime_helicity_projections",
            projection_rows,
            (
                ("variant_id", _U32),
                ("source_component", _U32),
                ("full_component", _U32),
            ),
        ),
    )


def _sources_table(
    catalog,
    ids,
    strings,
    digests,
    flavour_flows,
    quantum_number_flows,
    sequences,
):
    parameter_ids = ids["parameters"]
    rows = []
    for index, record in enumerate(catalog.sources):
        rows.append(
            (
                index,
                strings.id(record.template_id),
                ids["current_states"][record.state_template_id],
                strings.id(record.crossing),
                strings.id(record.wavefunction_family),
                record.helicity,
                record.spin_state,
                flavour_flows.id(record.flavour_flow),
                quantum_number_flows.id(record.quantum_number_flow),
                _LC_COLOR_SOURCE_SEED_OPERATION[record.lc_color_seed.operation],
                strings.id(record.lc_color_seed.output_shape_kind),
                (
                    255
                    if record.lc_color_seed.component_kind is None
                    else _LC_COLOR_COMPONENT_KIND[record.lc_color_seed.component_kind]
                ),
                _LC_COLOR_COMPONENT_ROLE[record.lc_color_seed.component_role],
                digests.id(record.lc_color_seed.proof_digest),
                sequences.id(
                    tuple(
                        strings.id(item)
                        for pair in record.lc_color_seed.provenance
                        for item in pair
                    )
                ),
                digests.id(record.wavefunction_expression_digest),
                ids["evaluator_bindings"][record.evaluator_resolver_key],
                _optional_reference(record.mass_parameter_id, parameter_ids),
                _optional_reference(record.width_parameter_id, parameter_ids),
                digests.id(record.semantic_digest),
            )
        )
    return _rows(
        "sources",
        rows,
        (
            ("id", _U32),
            ("template_string_id", _U32),
            ("state_template_id", _U32),
            ("crossing_string_id", _U32),
            ("wavefunction_family_string_id", _U32),
            ("helicity", _I32),
            ("spin_state", _I32),
            ("flavour_flow_id", _U32),
            ("quantum_number_flow_id", _U32),
            ("lc_color_seed_operation", _U8),
            ("lc_color_seed_shape_string_id", _U32),
            ("lc_color_seed_component_kind", _U8),
            ("lc_color_seed_component_role", _U8),
            ("lc_color_seed_proof_digest_id", _U32),
            ("lc_color_seed_provenance_sequence_id", _U32),
            ("wavefunction_expression_digest_id", _U32),
            ("evaluator_binding_id", _U32),
            ("mass_parameter_id", _U32),
            ("width_parameter_id", _U32),
            ("semantic_digest_id", _U32),
        ),
    )


def _quantum_flows_table(
    catalog,
    ids,
    strings,
    digests,
    factors,
    u32_sequences,
    i32_sequences,
    coupling_orders,
    flavour_flows,
    quantum_number_flows,
):
    rows = []
    for index, record in enumerate(catalog.quantum_flows):
        rows.append(
            (
                index,
                strings.id(record.template_id),
                u32_sequences.id(
                    tuple(
                        ids["current_states"][value]
                        for value in record.input_state_template_ids
                    )
                ),
                i32_sequences.id(record.input_spin_states),
                u32_sequences.id(
                    tuple(
                        flavour_flows.id(value) for value in record.input_flavour_flows
                    )
                ),
                u32_sequences.id(
                    tuple(
                        quantum_number_flows.id(value)
                        for value in record.input_quantum_number_flows
                    )
                ),
                strings.id(record.flavour_flow_operation),
                strings.id(record.quantum_number_flow_operation),
                coupling_orders.id(record.coupling_orders),
                ids["current_states"][record.result_state_template_id],
                record.result_spin_state,
                flavour_flows.id(record.result_flavour_flow),
                quantum_number_flows.id(record.result_quantum_number_flow),
                factors.id(record.exact_coupling),
                digests.id(record.predicate_digest),
                digests.id(record.semantic_digest),
            )
        )
    return _rows(
        "quantum_flows",
        rows,
        (
            ("id", _U32),
            ("template_string_id", _U32),
            ("input_state_sequence_id", _U32),
            ("input_spin_sequence_id", _U32),
            ("input_flavour_sequence_id", _U32),
            ("input_quantum_sequence_id", _U32),
            ("flavour_flow_operation_string_id", _U32),
            ("quantum_number_flow_operation_string_id", _U32),
            ("coupling_order_set_id", _U32),
            ("result_state_template_id", _U32),
            ("result_spin_state", _I32),
            ("result_flavour_flow_id", _U32),
            ("result_quantum_number_flow_id", _U32),
            ("exact_coupling_factor_id", _U32),
            ("predicate_digest_id", _U32),
            ("semantic_digest_id", _U32),
        ),
    )


def _transitions_table(
    catalog, ids, strings, digests, factors, sequences, coupling_orders
):
    rows = []
    for index, record in enumerate(catalog.transitions):
        rows.append(
            (
                index,
                strings.id(record.template_id),
                sequences.id(
                    tuple(
                        ids["current_states"][value]
                        for value in record.input_state_template_ids
                    )
                ),
                ids["current_states"][record.result_state_template_id],
                ids["quantum_flows"][record.quantum_flow_template_id],
                ids["evaluator_bindings"][record.evaluator_resolver_key],
                sequences.id(record.canonical_input_order),
                sequences.id(
                    tuple(strings.id(value) for value in record.momentum_convention)
                ),
                sequences.id(
                    tuple(
                        ids["parameters"][value]
                        for value in record.coupling_parameter_ids
                    )
                ),
                coupling_orders.id(record.coupling_orders),
                ids["color_contractions"][record.color_contraction_template_id],
                factors.id(record.binding_coupling),
                factors.id(record.exact_factor),
                _OUTPUT_FACTOR_SOURCE[record.output_factor_source],
                strings.id(record.equivalence_class),
                factors.id(record.input_exchange_factor),
                strings.id(record.output_projection),
                sequences.id(
                    tuple(
                        ids["contact_orbit_steps"][value]
                        for value in record.contact_orbit_step_template_ids
                    )
                ),
                sequences.id(
                    tuple(
                        digests.id(value)
                        for value in record.contact_orbit_step_semantic_digests
                    )
                ),
                digests.id(record.semantic_digest),
            )
        )
    return _rows(
        "transitions",
        rows,
        (
            ("id", _U32),
            ("template_string_id", _U32),
            ("input_state_sequence_id", _U32),
            ("result_state_template_id", _U32),
            ("quantum_flow_template_id", _U32),
            ("evaluator_binding_id", _U32),
            ("canonical_input_order_sequence_id", _U32),
            ("momentum_convention_sequence_id", _U32),
            ("coupling_parameter_sequence_id", _U32),
            ("coupling_order_set_id", _U32),
            ("color_contraction_template_id", _U32),
            ("binding_coupling_factor_id", _U32),
            ("exact_factor_id", _U32),
            ("output_factor_source", _U8),
            ("equivalence_class_string_id", _U32),
            ("input_exchange_factor_id", _U32),
            ("output_projection_string_id", _U32),
            ("contact_orbit_step_sequence_id", _U32),
            ("contact_orbit_step_semantic_digest_sequence_id", _U32),
            ("semantic_digest_id", _U32),
        ),
    )


def _contact_orbit_certificates_table(
    catalog, strings, digests, factors, sequences
):
    rows = []
    for index, record in enumerate(catalog.contact_orbit_certificates):
        rows.append(
            (
                index,
                strings.id(record.template_id),
                strings.id(record.algorithm),
                record.algorithm_version,
                record.term_id,
                strings.id(record.vertex),
                sequences.id(tuple(strings.id(value) for value in record.particles)),
                strings.id(record.evaluator_class),
                sequences.id(record.physical_leg_equivalence_classes),
                factors.id(record.reconstruction_factor),
                digests.id(record.semantic_digest),
            )
        )
    return _rows(
        "contact_orbit_certificates",
        rows,
        (
            ("id", _U32),
            ("template_string_id", _U32),
            ("algorithm_string_id", _U32),
            ("algorithm_version", _U32),
            ("term_id", _U32),
            ("vertex_string_id", _U32),
            ("particle_string_sequence_id", _U32),
            ("evaluator_class_string_id", _U32),
            ("physical_leg_equivalence_sequence_id", _U32),
            ("reconstruction_factor_id", _U32),
            ("semantic_digest_id", _U32),
        ),
    )


def _contact_orbit_steps_table(
    catalog, ids, strings, digests, factors, u32_sequences, i32_sequences
):
    rows = []
    stage_ids = {"partial": 0, "final": 1}
    for index, record in enumerate(catalog.contact_orbit_steps):
        rows.append(
            (
                index,
                strings.id(record.template_id),
                ids["contact_orbit_certificates"][record.certificate_template_id],
                stage_ids[record.stage],
                record.result_leg,
                u32_sequences.id(record.left_covered_legs),
                u32_sequences.id(record.right_covered_legs),
                i32_sequences.id(record.source_particle_legs),
                factors.id(record.reconstruction_factor),
                digests.id(record.semantic_digest),
            )
        )
    return _rows(
        "contact_orbit_steps",
        rows,
        (
            ("id", _U32),
            ("template_string_id", _U32),
            ("certificate_id", _U32),
            ("stage", _U8),
            ("result_leg", _U8),
            ("left_covered_leg_sequence_id", _U32),
            ("right_covered_leg_sequence_id", _U32),
            ("source_particle_leg_sequence_id", _U32),
            ("reconstruction_factor_id", _U32),
            ("semantic_digest_id", _U32),
        ),
    )


def _propagators_table(catalog, ids, strings, digests):
    rows = []
    for index, record in enumerate(catalog.propagators):
        rows.append(
            (
                index,
                strings.id(record.template_id),
                ids["current_states"][record.state_template_id],
                int(record.applies_propagator),
                _optional_reference(
                    record.evaluator_resolver_key, ids["evaluator_bindings"]
                ),
                digests.id(record.numerator_expression_digest),
                digests.id(record.denominator_expression_digest),
                _optional_reference(record.mass_parameter_id, ids["parameters"]),
                _optional_reference(record.width_parameter_id, ids["parameters"]),
                strings.id(record.gauge),
                _optional_reference(
                    record.linearity_proof_template_id, ids["symmetry_proofs"]
                ),
                digests.id(record.semantic_digest),
            )
        )
    return _rows(
        "propagators",
        rows,
        (
            ("id", _U32),
            ("template_string_id", _U32),
            ("state_template_id", _U32),
            ("applies_propagator", _U8),
            ("evaluator_binding_id", _U32),
            ("numerator_expression_digest_id", _U32),
            ("denominator_expression_digest_id", _U32),
            ("mass_parameter_id", _U32),
            ("width_parameter_id", _U32),
            ("gauge_string_id", _U32),
            ("linearity_proof_template_id", _U32),
            ("semantic_digest_id", _U32),
        ),
    )


def _closures_table(
    catalog, ids, strings, digests, factors, sequences, coupling_orders
):
    rows = []
    for index, record in enumerate(catalog.closures):
        rows.append(
            (
                index,
                strings.id(record.template_id),
                sequences.id(
                    tuple(
                        ids["current_states"][value]
                        for value in record.input_state_template_ids
                    )
                ),
                _optional_reference(
                    record.result_state_template_id, ids["current_states"]
                ),
                ids["evaluator_bindings"][record.evaluator_resolver_key],
                sequences.id(record.canonical_input_order),
                sequences.id(
                    tuple(
                        ids["parameters"][value]
                        for value in record.coupling_parameter_ids
                    )
                ),
                coupling_orders.id(record.coupling_orders),
                sequences.id(
                    tuple(
                        ids["quantum_flows"][value]
                        for value in record.eligible_quantum_flow_template_ids
                    )
                ),
                ids["color_contractions"][record.color_contraction_template_id],
                factors.id(record.binding_coupling),
                factors.id(record.exact_factor),
                _OUTPUT_FACTOR_SOURCE[record.output_factor_source],
                strings.id(record.equivalence_class),
                factors.id(record.input_exchange_factor),
                strings.id(record.projection),
                sequences.id(
                    tuple(factors.id(value) for value in record.component_coefficients)
                ),
                strings.id(record.chirality_relation),
                strings.id(record.metric_signature),
                digests.id(record.semantic_digest),
            )
        )
    return _rows(
        "closures",
        rows,
        (
            ("id", _U32),
            ("template_string_id", _U32),
            ("input_state_sequence_id", _U32),
            ("result_state_template_id", _U32),
            ("evaluator_binding_id", _U32),
            ("canonical_input_order_sequence_id", _U32),
            ("coupling_parameter_sequence_id", _U32),
            ("coupling_order_set_id", _U32),
            ("eligible_quantum_flow_sequence_id", _U32),
            ("color_contraction_template_id", _U32),
            ("binding_coupling_factor_id", _U32),
            ("exact_factor_id", _U32),
            ("output_factor_source", _U8),
            ("equivalence_class_string_id", _U32),
            ("input_exchange_factor_id", _U32),
            ("projection_string_id", _U32),
            ("component_coefficient_sequence_id", _U32),
            ("chirality_relation_string_id", _U32),
            ("metric_signature_string_id", _U32),
            ("semantic_digest_id", _U32),
        ),
    )


def _color_contractions_table(catalog, strings, digests, factors, i32_sequences):
    rows = []
    witness_offset = 0
    nc_offset = 0
    for index, record in enumerate(catalog.color_contractions):
        rows.append(
            (
                index,
                strings.id(record.template_id),
                strings.id(record.rule_kind),
                i32_sequences.id(record.input_representations),
                int(record.output_representation is not None),
                0
                if record.output_representation is None
                else record.output_representation,
                record.ordered_open_string_arity,
                factors.id(record.exact_coefficient),
                witness_offset,
                len(record.transition_witnesses),
                nc_offset,
                len(record.nc_polynomial),
                digests.id(record.expression_digest),
                digests.id(record.semantic_digest),
            )
        )
        witness_offset += len(record.transition_witnesses)
        nc_offset += len(record.nc_polynomial)
    return _rows(
        "color_contractions",
        rows,
        (
            ("id", _U32),
            ("template_string_id", _U32),
            ("rule_kind_string_id", _U32),
            ("input_representation_sequence_id", _U32),
            ("has_output_representation", _U8),
            ("output_representation", _I32),
            ("ordered_open_string_arity", _U32),
            ("exact_coefficient_factor_id", _U32),
            ("witness_start", _U64),
            ("witness_count", _U64),
            ("nc_term_start", _U64),
            ("nc_term_count", _U64),
            ("expression_digest_id", _U32),
            ("semantic_digest_id", _U32),
        ),
    )


def _lc_color_transition_witnesses_table(
    catalog,
    strings,
    digests,
    factors,
    sequences,
):
    rows = []
    for color_id, record in enumerate(catalog.color_contractions):
        for ordinal, witness in enumerate(record.transition_witnesses):
            rows.append(
                (
                    color_id,
                    ordinal,
                    strings.id(witness.input_shape_kinds[0]),
                    strings.id(witness.input_shape_kinds[1]),
                    0 if witness.input_permutation == (0, 1) else 1,
                    witness.reverse_parent_mask,
                    _LC_COLOR_COMPONENT_OPERATION[witness.component_operation],
                    (
                        255
                        if witness.result_component_kind is None
                        else _LC_COLOR_COMPONENT_KIND[witness.result_component_kind]
                    ),
                    _LC_COLOR_COMPONENT_ROLE[witness.result_component_role],
                    strings.id(witness.result_shape_kind),
                    factors.id(witness.exact_factor),
                    digests.id(witness.proof_digest),
                    sequences.id(
                        tuple(
                            value
                            for pairing in witness.input_port_pairings
                            for port in pairing
                            for value in port
                        )
                    ),
                    sequences.id(
                        tuple(
                            value
                            for port in witness.result_port_bindings
                            for value in port
                        )
                    ),
                    sequences.id(
                        tuple(
                            strings.id(item)
                            for pair in witness.provenance
                            for item in pair
                        )
                    ),
                )
            )
    return _rows(
        "lc_color_transition_witnesses",
        rows,
        (
            ("color_contraction_id", _U32),
            ("ordinal", _U32),
            ("left_shape_string_id", _U32),
            ("right_shape_string_id", _U32),
            ("input_permutation", _U8),
            ("reverse_parent_mask", _U8),
            ("component_operation", _U8),
            ("result_component_kind", _U8),
            ("result_component_role", _U8),
            ("result_shape_string_id", _U32),
            ("exact_factor_id", _U32),
            ("proof_digest_id", _U32),
            ("input_port_pairing_sequence_id", _U32),
            ("result_port_binding_sequence_id", _U32),
            ("provenance_sequence_id", _U32),
        ),
    )


def _color_nc_terms_table(catalog, factors):
    rows = [
        (color_id, exponent, factors.id(factor))
        for color_id, record in enumerate(catalog.color_contractions)
        for exponent, factor in record.nc_polynomial
    ]
    return _rows(
        "color_nc_terms",
        rows,
        (
            ("color_contraction_id", _U32),
            ("exponent", _I32),
            ("factor_id", _U32),
        ),
    )


def _symmetry_proofs_table(catalog, strings, digests, factors, sequences):
    rows = []
    for index, record in enumerate(catalog.symmetry_proofs):
        rows.append(
            (
                index,
                strings.id(record.template_id),
                strings.id(record.proof_algorithm),
                sequences.id(
                    tuple(strings.id(value) for value in record.subject_template_ids)
                ),
                sequences.id(record.input_permutation),
                factors.id(record.exact_phase),
                sequences.id(
                    tuple(digests.id(value) for value in record.expression_digests)
                ),
                digests.id(record.witness_digest),
                digests.id(record.semantic_digest),
            )
        )
    return _rows(
        "symmetry_proofs",
        rows,
        (
            ("id", _U32),
            ("template_string_id", _U32),
            ("proof_algorithm_string_id", _U32),
            ("subject_template_sequence_id", _U32),
            ("input_permutation_sequence_id", _U32),
            ("exact_phase_factor_id", _U32),
            ("expression_digest_sequence_id", _U32),
            ("witness_digest_id", _U32),
            ("semantic_digest_id", _U32),
        ),
    )


def _evaluator_bindings_table(catalog, ids, strings, digests, sequences):
    rows = []
    for index, record in enumerate(catalog.evaluator_bindings):
        rows.append(
            (
                index,
                strings.id(record.resolver_key),
                MISSING_U32
                if record.prepared_kernel_id is None
                else record.prepared_kernel_id,
                _CONTRACT_KIND[record.contract_kind],
                digests.id(record.callable_signature),
                sequences.id(
                    tuple(
                        ids["current_states"][value]
                        for value in record.input_state_template_ids
                    )
                ),
                _optional_reference(
                    record.output_state_template_id, ids["current_states"]
                ),
                sequences.id(tuple(strings.id(value) for value in record.input_layout)),
                sequences.id(
                    tuple(strings.id(value) for value in record.output_layout)
                ),
                sequences.id(
                    tuple(
                        digests.id(value) for value in record.exact_expression_digests
                    )
                ),
                sequences.id(
                    tuple(strings.id(value) for value in record.semantic_template_ids)
                ),
                _CALLABLE_KIND[record.callable_kind],
                strings.id(record.runtime_template),
                digests.id(record.semantic_digest),
            )
        )
    return _rows(
        "evaluator_bindings",
        rows,
        (
            ("id", _U32),
            ("resolver_key_string_id", _U32),
            ("prepared_kernel_id", _U32),
            ("contract_kind", _U8),
            ("callable_signature_digest_id", _U32),
            ("input_state_sequence_id", _U32),
            ("output_state_template_id", _U32),
            ("input_layout_sequence_id", _U32),
            ("output_layout_sequence_id", _U32),
            ("exact_expression_digest_sequence_id", _U32),
            ("semantic_template_sequence_id", _U32),
            ("callable_kind", _U8),
            ("runtime_template_string_id", _U32),
            ("semantic_digest_id", _U32),
        ),
    )


def _coupling_order_terms(catalog: _CouplingOrderCatalog, strings: _StringCatalog):
    rows = [
        (set_id, strings.id(name), power)
        for set_id, values in enumerate(catalog.values)
        for name, power in values
    ]
    return _rows(
        "coupling_order_terms",
        rows,
        (
            ("set_id", _U32),
            ("name_string_id", _U32),
            ("power", _U32),
        ),
    )


def _catalog_ranges(prefix: str, values: Sequence[Sequence[object]], _width: int):
    del _width
    rows = []
    offset = 0
    for index, value in enumerate(values):
        rows.append((index, offset, len(value)))
        offset += len(value)
    return _rows(
        f"{prefix}_ranges", rows, (("id", _U32), ("start", _U64), ("count", _U64))
    )


def _flatten_sequence_tables(
    prefix: str,
    catalog: _SequenceCatalog,
    dtype: np.dtype[Any],
) -> tuple[RecurrenceColumnarTable, RecurrenceColumnarTable]:
    ranges = _catalog_ranges(prefix, catalog.values, 1)
    values = _table(
        f"{prefix}_values",
        {
            "value": _array(
                [value for sequence in catalog.values for value in sequence], dtype
            )
        },
    )
    return ranges, values


def _quantum_number_flow_ranges(
    catalog: _QuantumNumberFlowCatalog,
) -> RecurrenceColumnarTable:
    return _catalog_ranges("quantum_number_flow", catalog.values, 2)


def _quantum_number_flow_terms(
    catalog: _QuantumNumberFlowCatalog,
    strings: _StringCatalog,
) -> RecurrenceColumnarTable:
    rows = [
        (flow_id, strings.id(name), strings.id(expression))
        for flow_id, flow in enumerate(catalog.values)
        for name, expression in flow
    ]
    return _rows(
        "quantum_number_flow_terms",
        rows,
        (
            ("flow_id", _U32),
            ("name_string_id", _U32),
            ("expression_string_id", _U32),
        ),
    )


def _string_tables(
    strings: _StringCatalog,
) -> tuple[RecurrenceColumnarTable, RecurrenceColumnarTable]:
    encoded = tuple(value.encode("utf-8") for value in strings.values)
    ranges = []
    offset = 0
    for value in encoded:
        ranges.append((offset, len(value)))
        offset += len(value)
    return (
        _rows("string_ranges", ranges, (("start", _U64), ("count", _U64))),
        _table("string_bytes", {"value": _array(b"".join(encoded), _U8)}),
    )


def _digest_table(catalog: _DigestCatalog) -> RecurrenceColumnarTable:
    values = np.empty((len(catalog.values), 32), dtype=_U8)
    for row, value in enumerate(catalog.values):
        values[row, :] = np.frombuffer(bytes.fromhex(value), dtype=_U8)
    return _table(
        "digest_catalog",
        {"id": _array(range(len(catalog.values)), _U32), "value": values},
    )


def _exact_factor_table(catalog: _FactorCatalog, strings: _StringCatalog):
    rows = [
        (index, *(strings.id(str(value)) for value in _factor_key(factor)))
        for index, factor in enumerate(catalog.values)
    ]
    return _rows(
        "exact_factors",
        rows,
        (
            ("id", _U32),
            ("real_numerator_string_id", _U32),
            ("real_denominator_string_id", _U32),
            ("imag_numerator_string_id", _U32),
            ("imag_denominator_string_id", _U32),
        ),
    )


def _table(
    name: str, columns: dict[str, np.ndarray[Any, Any]]
) -> RecurrenceColumnarTable:
    frozen = []
    for column_name, values in columns.items():
        owned = (
            values
            if values.flags.owndata and values.flags.c_contiguous
            else np.ascontiguousarray(values).copy()
        )
        owned.flags.writeable = False
        frozen.append(RecurrenceColumn(column_name, owned))
    row_count = 0 if not frozen else len(frozen[0].values)
    return RecurrenceColumnarTable(name, row_count, tuple(frozen))


def _rows(
    name: str,
    rows: Sequence[Sequence[int]],
    schema: Sequence[tuple[str, np.dtype[Any]]],
):
    columns = {
        column_name: _array(
            (row[index] for row in rows),
            dtype,
            context=f"{name}.{column_name}",
        )
        for index, (column_name, dtype) in enumerate(schema)
    }
    return _table(name, columns)


def _array(
    values: Iterable[int] | bytes,
    dtype: np.dtype[Any],
    *,
    context: str = "recurrence template column",
) -> np.ndarray[Any, Any]:
    if isinstance(values, bytes):
        return np.frombuffer(values, dtype=_U8).copy()
    materialized = tuple(values)
    limits = np.iinfo(dtype)
    checked: list[int] = []
    for row, value in enumerate(materialized):
        try:
            integer = operator.index(value)
        except TypeError as error:
            raise RecurrenceColumnarInputError(
                f"{context} row {row} must be an integer, got {value!r}"
            ) from error
        if integer < limits.min or integer > limits.max:
            raise RecurrenceColumnarInputError(
                f"{context} row {row} value {integer} does not fit {dtype.str}"
            )
        checked.append(integer)
    return np.asarray(checked, dtype=dtype)


def _optional_reference(value: str | None, values: dict[str, int]) -> int:
    return MISSING_U32 if value is None else values[value]


def _present(*values: str | None) -> Iterable[str]:
    return (value for value in values if value is not None)


def _factor_key(value: ExactComplexRationalV1) -> tuple[int, int, int, int]:
    return (
        value.real_numerator,
        value.real_denominator,
        value.imag_numerator,
        value.imag_denominator,
    )


def _validate_i128_factors(catalog: RecurrenceTemplateCatalog) -> None:
    for factor_index, factor in enumerate(_all_factors(catalog)):
        for component in (
            "real_numerator",
            "real_denominator",
            "imag_numerator",
            "imag_denominator",
        ):
            value = getattr(factor, component)
            if value < _I128_MIN or value > _I128_MAX:
                raise RecurrenceColumnarInputError(
                    "recurrence exact factor cannot cross the Rust ABI as i128: "
                    f"factor {factor_index} {component}={value}"
                )


def _require_sha256(value: str, context: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RecurrenceColumnarInputError(f"{context} must be a lowercase SHA-256")


def _hash_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


__all__ = [
    "RECURRENCE_TEMPLATE_INPUT_ABI",
    "RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION",
    "RecurrenceTemplateInputV1",
    "build_recurrence_template_input_v1",
]
