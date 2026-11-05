# SPDX-License-Identifier: 0BSD
"""Private fixed-width encoder for external-fermion pairing catalogs.

This module is an intentionally unintegrated staging contract for a future
Rust recurrence-builder ABI.  It turns :class:`FermionPairingCatalogV1` into
owned, immutable NumPy structure-of-arrays tables without object dtypes.
Variable-length records use contiguous ``u64`` ranges, exact integers use
signed-magnitude little-endian ``u64`` limbs, and textual model contracts use
a deterministic UTF-8 catalog.

The leading underscore on the encoder and payload is deliberate.  Consumers
must not depend on this ABI until a matching Rust decoder is frozen.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from pyamplicol.generation.recurrence_columnar import (
    RecurrenceColumn,
    RecurrenceColumnarInputError,
    RecurrenceColumnarTable,
)
from pyamplicol.generation.recurrence_fermion_pairing import (
    NO_FERMION_LINE,
    PAIRING_PROOF_ALGORITHM,
    ExternalFermionEndpointRowV1,
    FermionPairingCatalogV1,
    FermionPairingClassRowV1,
    FermionPairingRuleRowV1,
)
from pyamplicol.models.recurrence_template import ExactComplexRationalV1

_PAIRING_COLUMNAR_ABI: Final = "pyamplicol-recurrence-fermion-pairing-columnar-v1"
_PAIRING_COLUMNAR_SCHEMA_VERSION: Final = 1
_U8 = np.dtype("u1")
_U32 = np.dtype("<u4")
_U64 = np.dtype("<u8")
_I32 = np.dtype("<i4")

_PARTICLE_ORIENTATION_CODES: Final = {"particle": 0, "antiparticle": 1}
_COLOR_ORIENTATION_CODES: Final = {"fundamental": 0, "antifundamental": 1}

_ColumnSpec = tuple[str, np.dtype[Any], tuple[int, ...]]
_TABLE_SCHEMAS: Final[dict[str, tuple[_ColumnSpec, ...]]] = {
    "header": (
        ("schema_version", _U32, ()),
        ("abi_string_id", _U32, ()),
        ("process_key_string_id", _U32, ()),
        ("proof_algorithm_string_id", _U32, ()),
        ("source_count", _U32, ()),
        ("endpoint_count", _U32, ()),
        ("pairing_class_count", _U32, ()),
        ("rule_count", _U32, ()),
        ("endpoint_state_template_count", _U64, ()),
        ("endpoint_anti_state_template_count", _U64, ()),
        ("endpoint_basis_count", _U64, ()),
        ("endpoint_color_representation_count", _U64, ()),
        ("class_fundamental_slot_count", _U64, ()),
        ("class_antifundamental_slot_count", _U64, ()),
        ("class_reference_pairing_count", _U64, ()),
        ("rule_class_pairing_index_count", _U64, ()),
        ("rule_endpoint_pairing_count", _U64, ()),
        ("rule_source_permutation_count", _U64, ()),
        ("rule_lineage_count", _U64, ()),
        ("exact_integer_count", _U32, ()),
        ("exact_integer_limb_count", _U64, ()),
        ("string_count", _U32, ()),
        ("string_byte_count", _U64, ()),
        ("no_fermion_line", _U32, ()),
        ("topology_digest", _U8, (32,)),
        ("semantic_digest", _U8, (32,)),
    ),
    "endpoints": (
        ("endpoint_id", _U32, ()),
        ("source_slot", _U32, ()),
        ("public_label", _U32, ()),
        ("species_class_id", _U32, ()),
        ("species_string_id", _U32, ()),
        ("particle_orientation", _U8, ()),
        ("color_orientation", _U8, ()),
        ("state_template_start", _U64, ()),
        ("state_template_count", _U64, ()),
        ("anti_state_template_start", _U64, ()),
        ("anti_state_template_count", _U64, ()),
        ("basis_start", _U64, ()),
        ("basis_count", _U64, ()),
        ("color_representation_start", _U64, ()),
        ("color_representation_count", _U64, ()),
        ("contract_digest", _U8, (32,)),
    ),
    "endpoint_state_template_ids": (("string_id", _U32, ()),),
    "endpoint_anti_state_template_ids": (("string_id", _U32, ()),),
    "endpoint_basis_ids": (("string_id", _U32, ()),),
    "endpoint_color_representations": (("value", _I32, ()),),
    "pairing_classes": (
        ("class_id", _U32, ()),
        ("species_class_id", _U32, ()),
        ("species_string_id", _U32, ()),
        ("fundamental_slot_start", _U64, ()),
        ("fundamental_slot_count", _U64, ()),
        ("antifundamental_slot_start", _U64, ()),
        ("antifundamental_slot_count", _U64, ()),
        ("reference_pairing_start", _U64, ()),
        ("reference_pairing_count", _U64, ()),
        ("pairing_count", _U64, ()),
        ("proof_digest", _U8, (32,)),
    ),
    "class_fundamental_slots": (("source_slot", _U32, ()),),
    "class_antifundamental_slots": (("source_slot", _U32, ()),),
    "class_reference_pairings": (
        ("fundamental_source_slot", _U32, ()),
        ("antifundamental_source_slot", _U32, ()),
    ),
    "rules": (
        ("rule_id", _U32, ()),
        ("class_pairing_index_start", _U64, ()),
        ("class_pairing_index_count", _U64, ()),
        ("endpoint_pairing_start", _U64, ()),
        ("endpoint_pairing_count", _U64, ()),
        ("source_permutation_start", _U64, ()),
        ("source_permutation_count", _U64, ()),
        ("lineage_start", _U64, ()),
        ("lineage_count", _U64, ()),
        ("fermion_parity", _I32, ()),
        ("real_numerator_integer_id", _U32, ()),
        ("real_denominator_integer_id", _U32, ()),
        ("imag_numerator_integer_id", _U32, ()),
        ("imag_denominator_integer_id", _U32, ()),
        ("multiplicity", _U64, ()),
        ("proof_algorithm_string_id", _U32, ()),
        ("proof_digest", _U8, (32,)),
    ),
    "rule_class_pairing_indices": (
        ("class_id", _U32, ()),
        ("pairing_index", _U64, ()),
    ),
    "rule_endpoint_pairings": (
        ("fundamental_source_slot", _U32, ()),
        ("antifundamental_source_slot", _U32, ()),
    ),
    "rule_source_slot_permutations": (("source_slot", _U32, ()),),
    "rule_lineages": (("line_id", _U32, ()),),
    "exact_integers": (
        ("integer_id", _U32, ()),
        ("sign", _I32, ()),
        ("limb_start", _U64, ()),
        ("limb_count", _U64, ()),
    ),
    "exact_integer_limbs": (("value", _U64, ()),),
    "string_ranges": (
        ("string_id", _U32, ()),
        ("start", _U64, ()),
        ("count", _U64, ()),
    ),
    "string_bytes": (("value", _U8, ()),),
}
_TABLE_ORDER: Final = tuple(_TABLE_SCHEMAS)


@dataclass(frozen=True, slots=True)
class _FermionPairingColumnarV1:
    """Owned fixed-width pairing payload awaiting a future Rust decoder."""

    abi: str
    tables: tuple[RecurrenceColumnarTable, ...]

    def __post_init__(self) -> None:
        if self.abi != _PAIRING_COLUMNAR_ABI:
            raise RecurrenceColumnarInputError(
                f"unsupported fermion-pairing columnar ABI {self.abi!r}"
            )
        _validate_encoded_tables(self.tables)

    def table(self, name: str) -> RecurrenceColumnarTable:
        """Return one immutable table by its private ABI name."""

        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(f"fermion-pairing payload has no table {name!r}")


class _StringCatalog:
    def __init__(self, values: Iterable[str]) -> None:
        canonical = {_nonempty_text(value, "pairing string") for value in values}
        self.values = tuple(sorted(canonical))
        _checked_u32(len(self.values), "pairing string count")
        self._ids = {value: index for index, value in enumerate(self.values)}

    def id(self, value: str) -> int:
        try:
            return self._ids[value]
        except KeyError as error:
            raise RecurrenceColumnarInputError(
                f"pairing string {value!r} was not collected"
            ) from error


class _ExactIntegerCatalog:
    def __init__(self, values: Iterable[int]) -> None:
        canonical = {_checked_int(value, "exact factor integer") for value in values}
        self.values = tuple(sorted(canonical))
        _checked_u32(len(self.values), "exact integer count")
        self._ids = {value: index for index, value in enumerate(self.values)}

    def id(self, value: int) -> int:
        try:
            return self._ids[value]
        except KeyError as error:
            raise RecurrenceColumnarInputError(
                f"exact integer {value} was not collected"
            ) from error


def _encode_fermion_pairing_catalog_v1(
    catalog: FermionPairingCatalogV1,
) -> _FermionPairingColumnarV1:
    """Validate and encode a canonical pairing catalog.

    The function is private because no Rust consumer is frozen yet.  It is
    deterministic for semantically identical input and fails before allocation
    of large flat arrays when IDs, ranges, proofs, or exact factors are stale.
    """

    _validate_logical_catalog(catalog)
    strings = _StringCatalog(_catalog_strings(catalog))
    exact_integers = _ExactIntegerCatalog(_catalog_exact_integers(catalog))

    endpoint_state_ids = [
        strings.id(value)
        for endpoint in catalog.endpoints
        for value in endpoint.state_template_ids
    ]
    endpoint_anti_state_ids = [
        strings.id(value)
        for endpoint in catalog.endpoints
        for value in endpoint.anti_state_template_ids
    ]
    endpoint_basis_ids = [
        strings.id(value)
        for endpoint in catalog.endpoints
        for value in endpoint.basis_ids
    ]
    endpoint_color_representations = [
        value
        for endpoint in catalog.endpoints
        for value in endpoint.color_representations
    ]
    class_fundamental_slots = [
        value
        for pairing_class in catalog.pairing_classes
        for value in pairing_class.fundamental_source_slots
    ]
    class_antifundamental_slots = [
        value
        for pairing_class in catalog.pairing_classes
        for value in pairing_class.antifundamental_source_slots
    ]
    class_reference_pairings = [
        value
        for pairing_class in catalog.pairing_classes
        for value in pairing_class.reference_pairings
    ]
    rule_class_indices = [
        value for rule in catalog.rules for value in rule.class_pairing_indices
    ]
    rule_endpoint_pairings = [
        value for rule in catalog.rules for value in rule.endpoint_pairings
    ]
    rule_source_permutations = [
        value for rule in catalog.rules for value in rule.source_slot_permutation
    ]
    rule_lineages = [
        value for rule in catalog.rules for value in rule.lineage_by_source_slot
    ]

    integer_limbs = [
        limb for value in exact_integers.values for limb in _unsigned_limbs(value)
    ]
    string_payloads = [value.encode("utf-8") for value in strings.values]
    string_bytes = b"".join(string_payloads)

    tables: list[RecurrenceColumnarTable] = []
    tables.append(
        _build_header_table(
            catalog,
            strings,
            exact_integers,
            endpoint_state_ids=endpoint_state_ids,
            endpoint_anti_state_ids=endpoint_anti_state_ids,
            endpoint_basis_ids=endpoint_basis_ids,
            endpoint_color_representations=endpoint_color_representations,
            class_fundamental_slots=class_fundamental_slots,
            class_antifundamental_slots=class_antifundamental_slots,
            class_reference_pairings=class_reference_pairings,
            rule_class_indices=rule_class_indices,
            rule_endpoint_pairings=rule_endpoint_pairings,
            rule_source_permutations=rule_source_permutations,
            rule_lineages=rule_lineages,
            integer_limbs=integer_limbs,
            string_bytes=string_bytes,
        )
    )
    tables.append(_build_endpoints_table(catalog.endpoints, strings))
    tables.append(_one_column_table("endpoint_state_template_ids", endpoint_state_ids))
    tables.append(
        _one_column_table("endpoint_anti_state_template_ids", endpoint_anti_state_ids)
    )
    tables.append(_one_column_table("endpoint_basis_ids", endpoint_basis_ids))
    tables.append(
        _one_column_table(
            "endpoint_color_representations", endpoint_color_representations
        )
    )
    tables.append(_build_pairing_classes_table(catalog.pairing_classes, strings))
    tables.append(_one_column_table("class_fundamental_slots", class_fundamental_slots))
    tables.append(
        _one_column_table("class_antifundamental_slots", class_antifundamental_slots)
    )
    tables.append(_pair_table("class_reference_pairings", class_reference_pairings))
    tables.append(_build_rules_table(catalog.rules, strings, exact_integers))
    tables.append(_class_index_table(rule_class_indices))
    tables.append(_pair_table("rule_endpoint_pairings", rule_endpoint_pairings))
    tables.append(
        _one_column_table("rule_source_slot_permutations", rule_source_permutations)
    )
    tables.append(_one_column_table("rule_lineages", rule_lineages))
    tables.extend(_build_exact_integer_tables(exact_integers))
    tables.extend(_build_string_tables(strings, string_payloads))
    return _FermionPairingColumnarV1(_PAIRING_COLUMNAR_ABI, tuple(tables))


def _validate_logical_catalog(catalog: FermionPairingCatalogV1) -> None:
    if not isinstance(catalog, FermionPairingCatalogV1):
        raise TypeError("pairing columnar encoding requires FermionPairingCatalogV1")
    _nonempty_text(catalog.process_key, "pairing process key")
    source_count = _checked_u32(catalog.source_count, "pairing source count")
    topology_digest = _digest_bytes(catalog.topology_digest, "topology digest")
    semantic_digest = _digest_bytes(catalog.semantic_digest, "semantic digest")
    del topology_digest, semantic_digest

    endpoints = tuple(catalog.endpoints)
    pairing_classes = tuple(catalog.pairing_classes)
    rules = tuple(catalog.rules)
    _checked_u32(len(endpoints), "pairing endpoint count")
    _checked_u32(len(pairing_classes), "pairing class count")
    _checked_u32(len(rules), "pairing rule count")
    if not rules:
        raise RecurrenceColumnarInputError(
            "a fermion pairing catalog must contain its trivial or physical rule"
        )

    endpoint_by_slot: dict[int, ExternalFermionEndpointRowV1] = {}
    for expected_id, endpoint in enumerate(endpoints):
        if not isinstance(endpoint, ExternalFermionEndpointRowV1):
            raise TypeError("pairing endpoints must be endpoint row v1 values")
        if endpoint.endpoint_id != expected_id:
            raise RecurrenceColumnarInputError(
                "pairing endpoint IDs must be dense and canonical"
            )
        source_slot = _bounded_u32(
            endpoint.source_slot, source_count, "pairing endpoint source slot"
        )
        if source_slot in endpoint_by_slot:
            raise RecurrenceColumnarInputError(
                "pairing endpoint source slots must be unique"
            )
        endpoint_by_slot[source_slot] = endpoint
        _checked_u32(endpoint.public_label, "pairing endpoint public label")
        _bounded_u32(
            endpoint.species_class_id,
            len(pairing_classes),
            "endpoint species class",
        )
        _nonempty_text(endpoint.species_id, "endpoint species ID")
        if endpoint.particle_orientation not in _PARTICLE_ORIENTATION_CODES:
            raise RecurrenceColumnarInputError(
                "endpoint particle orientation is not canonical"
            )
        if endpoint.color_orientation not in _COLOR_ORIENTATION_CODES:
            raise RecurrenceColumnarInputError(
                "endpoint color orientation is not canonical"
            )
        _canonical_strings(endpoint.state_template_ids, "state template IDs")
        _canonical_strings(
            endpoint.anti_state_template_ids, "antiparticle state template IDs"
        )
        _canonical_strings(endpoint.basis_ids, "endpoint basis IDs")
        _canonical_i32s(
            endpoint.color_representations, "endpoint color representations"
        )
        _digest_bytes(endpoint.contract_digest, "endpoint contract digest")
    if tuple(endpoint_by_slot) != tuple(sorted(endpoint_by_slot)):
        raise RecurrenceColumnarInputError(
            "pairing endpoints must be ordered by source slot"
        )

    expected_class_order: list[tuple[int, ...]] = []
    class_permutations: dict[int, tuple[tuple[int, ...], ...]] = {}
    for expected_id, pairing_class in enumerate(pairing_classes):
        if not isinstance(pairing_class, FermionPairingClassRowV1):
            raise TypeError("pairing classes must be pairing class row v1 values")
        if (
            pairing_class.class_id != expected_id
            or pairing_class.species_class_id != expected_id
        ):
            raise RecurrenceColumnarInputError(
                "pairing class IDs must be dense, aligned, and canonical"
            )
        _nonempty_text(pairing_class.species_id, "pairing class species ID")
        fundamental = _canonical_u32_slots(
            pairing_class.fundamental_source_slots,
            source_count,
            "fundamental pairing slots",
        )
        antifundamental = _canonical_u32_slots(
            pairing_class.antifundamental_source_slots,
            source_count,
            "antifundamental pairing slots",
        )
        if len(fundamental) != len(antifundamental):
            raise RecurrenceColumnarInputError(
                "pairing classes require balanced endpoint orientations"
            )
        expected_reference = tuple(zip(fundamental, antifundamental, strict=True))
        if pairing_class.reference_pairings != expected_reference:
            raise RecurrenceColumnarInputError(
                "pairing class reference pairings are stale"
            )
        expected_count = math.factorial(len(fundamental))
        if pairing_class.pairing_count != expected_count:
            raise RecurrenceColumnarInputError("pairing class pairing count is stale")
        _checked_u64(pairing_class.pairing_count, "pairing class pairing count")
        class_endpoints = tuple(
            endpoint
            for endpoint in endpoints
            if endpoint.species_class_id == expected_id
        )
        expected_fundamental = tuple(
            endpoint.source_slot
            for endpoint in class_endpoints
            if endpoint.color_orientation == "fundamental"
        )
        expected_antifundamental = tuple(
            endpoint.source_slot
            for endpoint in class_endpoints
            if endpoint.color_orientation == "antifundamental"
        )
        if (
            fundamental != expected_fundamental
            or antifundamental != expected_antifundamental
            or any(
                endpoint.species_id != pairing_class.species_id
                for endpoint in class_endpoints
            )
        ):
            raise RecurrenceColumnarInputError(
                "pairing class endpoint membership is inconsistent"
            )
        expected_class_order.append(tuple(sorted((*fundamental, *antifundamental))))
        class_payload = {
            "algorithm": PAIRING_PROOF_ALGORITHM,
            "antifundamental_source_slots": antifundamental,
            "fundamental_source_slots": fundamental,
            "reference_pairings": expected_reference,
            "species_contract_digests": tuple(
                sorted(endpoint.contract_digest for endpoint in class_endpoints)
            ),
        }
        if pairing_class.proof_digest != _digest(class_payload):
            raise RecurrenceColumnarInputError("pairing class proof digest is stale")
        class_permutations[expected_id] = tuple(itertools.permutations(antifundamental))
    if expected_class_order != sorted(expected_class_order):
        raise RecurrenceColumnarInputError(
            "pairing classes are not in canonical source-slot order"
        )
    if len({row.species_id for row in pairing_classes}) != len(pairing_classes):
        raise RecurrenceColumnarInputError("pairing class species IDs must be unique")

    expected_rule_count = math.prod(
        pairing_class.pairing_count for pairing_class in pairing_classes
    )
    if len(rules) != expected_rule_count:
        raise RecurrenceColumnarInputError("pairing rule product is incomplete")
    for expected_id, rule in enumerate(rules):
        _validate_rule(
            rule,
            expected_id=expected_id,
            source_count=source_count,
            pairing_classes=pairing_classes,
            class_permutations=class_permutations,
        )

    topology_payload = _topology_payload(
        source_count, endpoints, pairing_classes, rules
    )
    if catalog.topology_digest != _digest(topology_payload):
        raise RecurrenceColumnarInputError("pairing topology digest is stale")
    semantic_payload = {
        "process_key": catalog.process_key,
        "topology": topology_payload,
        "endpoint_contracts": tuple(
            {
                "anti_state_template_ids": endpoint.anti_state_template_ids,
                "basis_ids": endpoint.basis_ids,
                "color_representations": endpoint.color_representations,
                "contract_digest": endpoint.contract_digest,
                "particle_orientation": endpoint.particle_orientation,
                "species_id": endpoint.species_id,
                "state_template_ids": endpoint.state_template_ids,
            }
            for endpoint in endpoints
        ),
    }
    if catalog.semantic_digest != _digest(semantic_payload):
        raise RecurrenceColumnarInputError("pairing semantic digest is stale")


def _validate_rule(
    rule: FermionPairingRuleRowV1,
    *,
    expected_id: int,
    source_count: int,
    pairing_classes: tuple[FermionPairingClassRowV1, ...],
    class_permutations: Mapping[int, tuple[tuple[int, ...], ...]],
) -> None:
    if not isinstance(rule, FermionPairingRuleRowV1):
        raise TypeError("pairing rules must be pairing rule row v1 values")
    if rule.rule_id != expected_id:
        raise RecurrenceColumnarInputError(
            "pairing rule IDs must be dense and canonical"
        )
    expected_class_ids = tuple(range(len(pairing_classes)))
    if tuple(item[0] for item in rule.class_pairing_indices) != expected_class_ids:
        raise RecurrenceColumnarInputError(
            "pairing rule class indices must cover every class in order"
        )

    endpoint_pairings: list[tuple[int, int]] = []
    source_permutation = list(range(source_count))
    parity = 1
    for class_id, pairing_index in rule.class_pairing_indices:
        _checked_u32(class_id, "rule pairing class ID")
        _checked_u64(pairing_index, "rule class-local pairing index")
        permutations = class_permutations[class_id]
        if pairing_index >= len(permutations):
            raise RecurrenceColumnarInputError(
                "rule class-local pairing index is out of bounds"
            )
        pairing_class = pairing_classes[class_id]
        selected = permutations[pairing_index]
        parity *= _permutation_parity(
            pairing_class.antifundamental_source_slots, selected
        )
        endpoint_pairings.extend(
            zip(pairing_class.fundamental_source_slots, selected, strict=True)
        )
        for reference_slot, selected_slot in zip(
            pairing_class.antifundamental_source_slots, selected, strict=True
        ):
            source_permutation[reference_slot] = selected_slot
    expected_pairings = tuple(sorted(endpoint_pairings))
    if rule.endpoint_pairings != expected_pairings:
        raise RecurrenceColumnarInputError("rule endpoint pairings are stale")
    if rule.source_slot_permutation != tuple(source_permutation):
        raise RecurrenceColumnarInputError("rule source-slot permutation is stale")

    lineage = [NO_FERMION_LINE] * source_count
    for line_id, (fundamental_slot, antifundamental_slot) in enumerate(
        expected_pairings
    ):
        lineage[fundamental_slot] = line_id
        lineage[antifundamental_slot] = line_id
    if rule.lineage_by_source_slot != tuple(lineage):
        raise RecurrenceColumnarInputError("rule fermion lineage is stale")
    if rule.fermion_parity != parity or rule.fermion_parity not in {-1, 1}:
        raise RecurrenceColumnarInputError("rule fermion parity is stale")
    expected_factor = ExactComplexRationalV1(parity, 1, 0, 1)
    if rule.exact_factor != expected_factor:
        raise RecurrenceColumnarInputError(
            "rule exact factor does not reproduce its fermion parity"
        )
    if rule.multiplicity != 1:
        raise RecurrenceColumnarInputError("pairing rule multiplicity is not canonical")
    _checked_u64(rule.multiplicity, "pairing rule multiplicity")
    if rule.proof_algorithm != PAIRING_PROOF_ALGORITHM:
        raise RecurrenceColumnarInputError("pairing rule proof algorithm is stale")
    rule_payload = {
        "algorithm": PAIRING_PROOF_ALGORITHM,
        "class_pairing_indices": rule.class_pairing_indices,
        "endpoint_pairings": rule.endpoint_pairings,
        "fermion_parity": rule.fermion_parity,
        "lineage_by_source_slot": rule.lineage_by_source_slot,
        "source_slot_permutation": rule.source_slot_permutation,
    }
    if rule.proof_digest != _digest(rule_payload):
        raise RecurrenceColumnarInputError("pairing rule proof digest is stale")


def _build_header_table(
    catalog: FermionPairingCatalogV1,
    strings: _StringCatalog,
    exact_integers: _ExactIntegerCatalog,
    **flat_values: Sequence[object] | bytes,
) -> RecurrenceColumnarTable:
    columns = _allocate("header", 1)
    columns["schema_version"][0] = _PAIRING_COLUMNAR_SCHEMA_VERSION
    columns["abi_string_id"][0] = strings.id(_PAIRING_COLUMNAR_ABI)
    columns["process_key_string_id"][0] = strings.id(catalog.process_key)
    columns["proof_algorithm_string_id"][0] = strings.id(PAIRING_PROOF_ALGORITHM)
    columns["source_count"][0] = catalog.source_count
    columns["endpoint_count"][0] = len(catalog.endpoints)
    columns["pairing_class_count"][0] = len(catalog.pairing_classes)
    columns["rule_count"][0] = len(catalog.rules)
    count_columns = {
        "endpoint_state_template_count": "endpoint_state_ids",
        "endpoint_anti_state_template_count": "endpoint_anti_state_ids",
        "endpoint_basis_count": "endpoint_basis_ids",
        "endpoint_color_representation_count": "endpoint_color_representations",
        "class_fundamental_slot_count": "class_fundamental_slots",
        "class_antifundamental_slot_count": "class_antifundamental_slots",
        "class_reference_pairing_count": "class_reference_pairings",
        "rule_class_pairing_index_count": "rule_class_indices",
        "rule_endpoint_pairing_count": "rule_endpoint_pairings",
        "rule_source_permutation_count": "rule_source_permutations",
        "rule_lineage_count": "rule_lineages",
        "exact_integer_limb_count": "integer_limbs",
        "string_byte_count": "string_bytes",
    }
    for column_name, value_name in count_columns.items():
        columns[column_name][0] = _checked_u64(
            len(flat_values[value_name]), column_name
        )
    columns["exact_integer_count"][0] = len(exact_integers.values)
    columns["string_count"][0] = len(strings.values)
    columns["no_fermion_line"][0] = NO_FERMION_LINE
    columns["topology_digest"][0] = np.frombuffer(
        _digest_bytes(catalog.topology_digest, "topology digest"), dtype=_U8
    )
    columns["semantic_digest"][0] = np.frombuffer(
        _digest_bytes(catalog.semantic_digest, "semantic digest"), dtype=_U8
    )
    return _freeze_table("header", columns)


def _build_endpoints_table(
    endpoints: Sequence[ExternalFermionEndpointRowV1],
    strings: _StringCatalog,
) -> RecurrenceColumnarTable:
    columns = _allocate("endpoints", len(endpoints))
    state_cursor = anti_cursor = basis_cursor = color_cursor = 0
    for row, endpoint in enumerate(endpoints):
        columns["endpoint_id"][row] = endpoint.endpoint_id
        columns["source_slot"][row] = endpoint.source_slot
        columns["public_label"][row] = endpoint.public_label
        columns["species_class_id"][row] = endpoint.species_class_id
        columns["species_string_id"][row] = strings.id(endpoint.species_id)
        columns["particle_orientation"][row] = _PARTICLE_ORIENTATION_CODES[
            endpoint.particle_orientation
        ]
        columns["color_orientation"][row] = _COLOR_ORIENTATION_CODES[
            endpoint.color_orientation
        ]
        for prefix, cursor, values in (
            ("state_template", state_cursor, endpoint.state_template_ids),
            ("anti_state_template", anti_cursor, endpoint.anti_state_template_ids),
            ("basis", basis_cursor, endpoint.basis_ids),
            (
                "color_representation",
                color_cursor,
                endpoint.color_representations,
            ),
        ):
            columns[f"{prefix}_start"][row] = cursor
            columns[f"{prefix}_count"][row] = len(values)
        state_cursor += len(endpoint.state_template_ids)
        anti_cursor += len(endpoint.anti_state_template_ids)
        basis_cursor += len(endpoint.basis_ids)
        color_cursor += len(endpoint.color_representations)
        columns["contract_digest"][row] = np.frombuffer(
            _digest_bytes(endpoint.contract_digest, "endpoint contract digest"),
            dtype=_U8,
        )
    return _freeze_table("endpoints", columns)


def _build_pairing_classes_table(
    pairing_classes: Sequence[FermionPairingClassRowV1],
    strings: _StringCatalog,
) -> RecurrenceColumnarTable:
    columns = _allocate("pairing_classes", len(pairing_classes))
    fundamental_cursor = antifundamental_cursor = reference_cursor = 0
    for row, pairing_class in enumerate(pairing_classes):
        columns["class_id"][row] = pairing_class.class_id
        columns["species_class_id"][row] = pairing_class.species_class_id
        columns["species_string_id"][row] = strings.id(pairing_class.species_id)
        columns["fundamental_slot_start"][row] = fundamental_cursor
        columns["fundamental_slot_count"][row] = len(
            pairing_class.fundamental_source_slots
        )
        columns["antifundamental_slot_start"][row] = antifundamental_cursor
        columns["antifundamental_slot_count"][row] = len(
            pairing_class.antifundamental_source_slots
        )
        columns["reference_pairing_start"][row] = reference_cursor
        columns["reference_pairing_count"][row] = len(pairing_class.reference_pairings)
        columns["pairing_count"][row] = pairing_class.pairing_count
        columns["proof_digest"][row] = np.frombuffer(
            _digest_bytes(pairing_class.proof_digest, "pairing class proof digest"),
            dtype=_U8,
        )
        fundamental_cursor += len(pairing_class.fundamental_source_slots)
        antifundamental_cursor += len(pairing_class.antifundamental_source_slots)
        reference_cursor += len(pairing_class.reference_pairings)
    return _freeze_table("pairing_classes", columns)


def _build_rules_table(
    rules: Sequence[FermionPairingRuleRowV1],
    strings: _StringCatalog,
    exact_integers: _ExactIntegerCatalog,
) -> RecurrenceColumnarTable:
    columns = _allocate("rules", len(rules))
    class_cursor = pair_cursor = permutation_cursor = lineage_cursor = 0
    for row, rule in enumerate(rules):
        columns["rule_id"][row] = rule.rule_id
        columns["class_pairing_index_start"][row] = class_cursor
        columns["class_pairing_index_count"][row] = len(rule.class_pairing_indices)
        columns["endpoint_pairing_start"][row] = pair_cursor
        columns["endpoint_pairing_count"][row] = len(rule.endpoint_pairings)
        columns["source_permutation_start"][row] = permutation_cursor
        columns["source_permutation_count"][row] = len(rule.source_slot_permutation)
        columns["lineage_start"][row] = lineage_cursor
        columns["lineage_count"][row] = len(rule.lineage_by_source_slot)
        columns["fermion_parity"][row] = rule.fermion_parity
        factor = rule.exact_factor
        columns["real_numerator_integer_id"][row] = exact_integers.id(
            factor.real_numerator
        )
        columns["real_denominator_integer_id"][row] = exact_integers.id(
            factor.real_denominator
        )
        columns["imag_numerator_integer_id"][row] = exact_integers.id(
            factor.imag_numerator
        )
        columns["imag_denominator_integer_id"][row] = exact_integers.id(
            factor.imag_denominator
        )
        columns["multiplicity"][row] = rule.multiplicity
        columns["proof_algorithm_string_id"][row] = strings.id(rule.proof_algorithm)
        columns["proof_digest"][row] = np.frombuffer(
            _digest_bytes(rule.proof_digest, "pairing rule proof digest"), dtype=_U8
        )
        class_cursor += len(rule.class_pairing_indices)
        pair_cursor += len(rule.endpoint_pairings)
        permutation_cursor += len(rule.source_slot_permutation)
        lineage_cursor += len(rule.lineage_by_source_slot)
    return _freeze_table("rules", columns)


def _build_exact_integer_tables(
    catalog: _ExactIntegerCatalog,
) -> tuple[RecurrenceColumnarTable, RecurrenceColumnarTable]:
    columns = _allocate("exact_integers", len(catalog.values))
    flat_limbs: list[int] = []
    for row, value in enumerate(catalog.values):
        limbs = _unsigned_limbs(value)
        columns["integer_id"][row] = row
        columns["sign"][row] = -1 if value < 0 else (1 if value > 0 else 0)
        columns["limb_start"][row] = len(flat_limbs)
        columns["limb_count"][row] = len(limbs)
        flat_limbs.extend(limbs)
    return (
        _freeze_table("exact_integers", columns),
        _one_column_table("exact_integer_limbs", flat_limbs),
    )


def _build_string_tables(
    strings: _StringCatalog,
    payloads: Sequence[bytes],
) -> tuple[RecurrenceColumnarTable, RecurrenceColumnarTable]:
    ranges = _allocate("string_ranges", len(strings.values))
    cursor = 0
    for row, payload in enumerate(payloads):
        ranges["string_id"][row] = row
        ranges["start"][row] = cursor
        ranges["count"][row] = len(payload)
        cursor += len(payload)
    values = _allocate("string_bytes", cursor)
    if cursor:
        values["value"][:] = np.frombuffer(b"".join(payloads), dtype=_U8)
    return (
        _freeze_table("string_ranges", ranges),
        _freeze_table("string_bytes", values),
    )


def _one_column_table(name: str, values: Sequence[int]) -> RecurrenceColumnarTable:
    columns = _allocate(name, len(values))
    column_name = _TABLE_SCHEMAS[name][0][0]
    if values:
        columns[column_name][:] = values
    return _freeze_table(name, columns)


def _pair_table(
    name: str, values: Sequence[tuple[int, int]]
) -> RecurrenceColumnarTable:
    columns = _allocate(name, len(values))
    left_name, right_name = (item[0] for item in _TABLE_SCHEMAS[name])
    for row, (left, right) in enumerate(values):
        columns[left_name][row] = left
        columns[right_name][row] = right
    return _freeze_table(name, columns)


def _class_index_table(
    values: Sequence[tuple[int, int]],
) -> RecurrenceColumnarTable:
    columns = _allocate("rule_class_pairing_indices", len(values))
    for row, (class_id, pairing_index) in enumerate(values):
        columns["class_id"][row] = class_id
        columns["pairing_index"][row] = pairing_index
    return _freeze_table("rule_class_pairing_indices", columns)


def _allocate(name: str, row_count: int) -> dict[str, np.ndarray[Any, Any]]:
    _checked_u64(row_count, f"{name} row count")
    result: dict[str, np.ndarray[Any, Any]] = {}
    for column_name, dtype, tail_shape in _TABLE_SCHEMAS[name]:
        result[column_name] = np.empty((row_count, *tail_shape), dtype=dtype, order="C")
    return result


def _freeze_table(
    name: str, columns: Mapping[str, np.ndarray[Any, Any]]
) -> RecurrenceColumnarTable:
    expected_names = tuple(item[0] for item in _TABLE_SCHEMAS[name])
    if tuple(columns) != expected_names:
        raise RecurrenceColumnarInputError(
            f"table {name!r} columns do not match its private ABI schema"
        )
    frozen: list[RecurrenceColumn] = []
    for column_name, expected_dtype, expected_tail_shape in _TABLE_SCHEMAS[name]:
        values = columns[column_name]
        can_adopt = (
            values.flags.owndata
            and values.flags.c_contiguous
            and values.dtype == expected_dtype
            and values.shape[1:] == expected_tail_shape
        )
        owned = (
            values
            if can_adopt
            else np.array(values, dtype=values.dtype, order="C", copy=True)
        )
        owned.flags.writeable = False
        frozen.append(RecurrenceColumn(column_name, owned))
    row_count = len(frozen[0].values) if frozen else 0
    return RecurrenceColumnarTable(name, row_count, tuple(frozen))


def _validate_encoded_tables(tables: tuple[RecurrenceColumnarTable, ...]) -> None:
    if tuple(table.name for table in tables) != _TABLE_ORDER:
        raise RecurrenceColumnarInputError(
            "fermion-pairing tables do not match the private ABI order"
        )
    by_name = {table.name: table for table in tables}
    for table in tables:
        schema = _TABLE_SCHEMAS[table.name]
        if tuple(column.name for column in table.columns) != tuple(
            item[0] for item in schema
        ):
            raise RecurrenceColumnarInputError(
                f"table {table.name!r} does not match its column schema"
            )
        for column, (_, dtype, tail_shape) in zip(table.columns, schema, strict=True):
            if column.values.dtype != dtype or column.values.shape[1:] != tail_shape:
                raise RecurrenceColumnarInputError(
                    f"table {table.name!r} column {column.name!r} has the wrong "
                    "little-endian fixed-width representation"
                )
    header = by_name["header"]
    if header.row_count != 1:
        raise RecurrenceColumnarInputError("pairing header must contain one row")
    if int(header.column("schema_version")[0]) != _PAIRING_COLUMNAR_SCHEMA_VERSION:
        raise RecurrenceColumnarInputError("pairing columnar schema version is stale")
    if int(header.column("no_fermion_line")[0]) != NO_FERMION_LINE:
        raise RecurrenceColumnarInputError("pairing lineage sentinel is stale")
    row_counts = {
        "endpoint_count": "endpoints",
        "pairing_class_count": "pairing_classes",
        "rule_count": "rules",
        "exact_integer_count": "exact_integers",
        "string_count": "string_ranges",
    }
    for count_column, table_name in row_counts.items():
        if int(header.column(count_column)[0]) != by_name[table_name].row_count:
            raise RecurrenceColumnarInputError(
                f"header {count_column!r} does not match table {table_name!r}"
            )
    flat_counts = {
        "endpoint_state_template_count": "endpoint_state_template_ids",
        "endpoint_anti_state_template_count": "endpoint_anti_state_template_ids",
        "endpoint_basis_count": "endpoint_basis_ids",
        "endpoint_color_representation_count": "endpoint_color_representations",
        "class_fundamental_slot_count": "class_fundamental_slots",
        "class_antifundamental_slot_count": "class_antifundamental_slots",
        "class_reference_pairing_count": "class_reference_pairings",
        "rule_class_pairing_index_count": "rule_class_pairing_indices",
        "rule_endpoint_pairing_count": "rule_endpoint_pairings",
        "rule_source_permutation_count": "rule_source_slot_permutations",
        "rule_lineage_count": "rule_lineages",
        "exact_integer_limb_count": "exact_integer_limbs",
        "string_byte_count": "string_bytes",
    }
    for count_column, table_name in flat_counts.items():
        if int(header.column(count_column)[0]) != by_name[table_name].row_count:
            raise RecurrenceColumnarInputError(
                f"header {count_column!r} does not match table {table_name!r}"
            )
    _validate_dense_ids(by_name["endpoints"], "endpoint_id")
    _validate_dense_ids(by_name["pairing_classes"], "class_id")
    _validate_dense_ids(by_name["rules"], "rule_id")
    _validate_dense_ids(by_name["exact_integers"], "integer_id")
    _validate_dense_ids(by_name["string_ranges"], "string_id")
    _validate_ranges(
        by_name["string_ranges"],
        "start",
        "count",
        by_name["string_bytes"].row_count,
        "pairing string catalog",
    )
    _validate_ranges(
        by_name["exact_integers"],
        "limb_start",
        "limb_count",
        by_name["exact_integer_limbs"].row_count,
        "pairing exact-integer catalog",
    )
    range_groups = (
        ("endpoints", "state_template", "endpoint_state_template_ids"),
        (
            "endpoints",
            "anti_state_template",
            "endpoint_anti_state_template_ids",
        ),
        ("endpoints", "basis", "endpoint_basis_ids"),
        (
            "endpoints",
            "color_representation",
            "endpoint_color_representations",
        ),
        ("pairing_classes", "fundamental_slot", "class_fundamental_slots"),
        (
            "pairing_classes",
            "antifundamental_slot",
            "class_antifundamental_slots",
        ),
        (
            "pairing_classes",
            "reference_pairing",
            "class_reference_pairings",
        ),
        ("rules", "class_pairing_index", "rule_class_pairing_indices"),
        ("rules", "endpoint_pairing", "rule_endpoint_pairings"),
        ("rules", "source_permutation", "rule_source_slot_permutations"),
        ("rules", "lineage", "rule_lineages"),
    )
    for parent_name, prefix, child_name in range_groups:
        _validate_ranges(
            by_name[parent_name],
            f"{prefix}_start",
            f"{prefix}_count",
            by_name[child_name].row_count,
            f"{parent_name}.{prefix}",
        )


def _validate_dense_ids(table: RecurrenceColumnarTable, column: str) -> None:
    if not np.array_equal(table.column(column), np.arange(table.row_count, dtype=_U32)):
        raise RecurrenceColumnarInputError(
            f"table {table.name!r} IDs are not dense and canonical"
        )


def _validate_ranges(
    table: RecurrenceColumnarTable,
    start_column: str,
    count_column: str,
    child_count: int,
    context: str,
) -> None:
    cursor = 0
    for row in range(table.row_count):
        start = int(table.column(start_column)[row])
        count = int(table.column(count_column)[row])
        if start != cursor or start + count > child_count:
            raise RecurrenceColumnarInputError(
                f"{context} ranges are not contiguous or are out of bounds"
            )
        cursor += count
    if cursor != child_count:
        raise RecurrenceColumnarInputError(
            f"{context} ranges do not cover every flattened row"
        )


def _catalog_strings(catalog: FermionPairingCatalogV1) -> Iterable[str]:
    yield _PAIRING_COLUMNAR_ABI
    yield catalog.process_key
    yield PAIRING_PROOF_ALGORITHM
    for endpoint in catalog.endpoints:
        yield endpoint.species_id
        yield from endpoint.state_template_ids
        yield from endpoint.anti_state_template_ids
        yield from endpoint.basis_ids
    for pairing_class in catalog.pairing_classes:
        yield pairing_class.species_id
    for rule in catalog.rules:
        yield rule.proof_algorithm


def _catalog_exact_integers(catalog: FermionPairingCatalogV1) -> Iterable[int]:
    for rule in catalog.rules:
        yield rule.exact_factor.real_numerator
        yield rule.exact_factor.real_denominator
        yield rule.exact_factor.imag_numerator
        yield rule.exact_factor.imag_denominator


def _topology_payload(
    source_count: int,
    endpoints: tuple[ExternalFermionEndpointRowV1, ...],
    pairing_classes: tuple[FermionPairingClassRowV1, ...],
    rules: tuple[FermionPairingRuleRowV1, ...],
) -> dict[str, object]:
    return {
        "algorithm": PAIRING_PROOF_ALGORITHM,
        "source_count": source_count,
        "endpoints": tuple(
            (
                endpoint.source_slot,
                endpoint.public_label,
                endpoint.species_class_id,
                endpoint.color_orientation,
            )
            for endpoint in endpoints
        ),
        "pairing_classes": tuple(
            (
                row.class_id,
                row.fundamental_source_slots,
                row.antifundamental_source_slots,
                row.reference_pairings,
                row.pairing_count,
            )
            for row in pairing_classes
        ),
        "rules": tuple(
            (
                rule.class_pairing_indices,
                rule.endpoint_pairings,
                rule.source_slot_permutation,
                rule.lineage_by_source_slot,
                rule.fermion_parity,
            )
            for rule in rules
        ),
    }


def _permutation_parity(reference: Sequence[int], candidate: Sequence[int]) -> int:
    ranks = {value: index for index, value in enumerate(reference)}
    permutation = tuple(ranks[value] for value in candidate)
    inversions = sum(
        left > right
        for index, left in enumerate(permutation)
        for right in permutation[index + 1 :]
    )
    return -1 if inversions % 2 else 1


def _unsigned_limbs(value: int) -> tuple[int, ...]:
    magnitude = abs(_checked_int(value, "exact integer"))
    limbs: list[int] = []
    while magnitude:
        limbs.append(magnitude & ((1 << 64) - 1))
        magnitude >>= 64
    return tuple(limbs)


def _canonical_strings(values: Sequence[str], context: str) -> tuple[str, ...]:
    materialized = tuple(_nonempty_text(value, context) for value in values)
    if materialized != tuple(sorted(set(materialized))):
        raise RecurrenceColumnarInputError(f"{context} are not canonical")
    return materialized


def _canonical_i32s(values: Sequence[int], context: str) -> tuple[int, ...]:
    materialized = tuple(_checked_i32(value, context) for value in values)
    if materialized != tuple(sorted(set(materialized))):
        raise RecurrenceColumnarInputError(f"{context} are not canonical")
    return materialized


def _canonical_u32_slots(
    values: Sequence[int], bound: int, context: str
) -> tuple[int, ...]:
    materialized = tuple(_bounded_u32(value, bound, context) for value in values)
    if materialized != tuple(sorted(set(materialized))):
        raise RecurrenceColumnarInputError(f"{context} are not canonical")
    return materialized


def _nonempty_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecurrenceColumnarInputError(f"{context} must be nonempty text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise RecurrenceColumnarInputError(
            f"{context} is not valid UTF-8 text"
        ) from error
    return value


def _checked_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecurrenceColumnarInputError(f"{context} must be an integer")
    return value


def _checked_u32(value: object, context: str) -> int:
    parsed = _checked_int(value, context)
    if not 0 <= parsed < (1 << 32):
        raise RecurrenceColumnarInputError(f"{context} does not fit u32")
    return parsed


def _checked_u64(value: object, context: str) -> int:
    parsed = _checked_int(value, context)
    if not 0 <= parsed < (1 << 64):
        raise RecurrenceColumnarInputError(f"{context} does not fit u64")
    return parsed


def _checked_i32(value: object, context: str) -> int:
    parsed = _checked_int(value, context)
    if not -(1 << 31) <= parsed < (1 << 31):
        raise RecurrenceColumnarInputError(f"{context} does not fit i32")
    return parsed


def _bounded_u32(value: object, bound: int, context: str) -> int:
    parsed = _checked_u32(value, context)
    if parsed >= bound:
        raise RecurrenceColumnarInputError(f"{context} is out of bounds")
    return parsed


def _digest_bytes(value: object, context: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise RecurrenceColumnarInputError(
            f"{context} must be a lowercase SHA-256 digest"
        )
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise RecurrenceColumnarInputError(
            f"{context} must be a lowercase SHA-256 digest"
        ) from error
    if decoded.hex() != value:
        raise RecurrenceColumnarInputError(
            f"{context} must be a lowercase SHA-256 digest"
        )
    return decoded


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


__all__: tuple[str, ...] = ()
