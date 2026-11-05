# SPDX-License-Identifier: 0BSD
"""Exact external-fermion pairing rules for recurrence closure reconstruction.

The catalog in this module is intentionally process-local and bounded.  It
uses canonical process roles to bind external source slots to model-owned
current-state contracts, then enumerates only species-compatible Wick
pairings.  Particle numbers are used solely to bind a process leg to its
authenticated state contract; all physics matching uses the contract's
species, antiparticle relation, statistics, orientation, and LC color shape.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeVar, cast

from pyamplicol.models.recurrence_template import (
    CurrentStateTemplateV1,
    ExactComplexRationalV1,
    QuantumFlowTemplateV1,
)
from pyamplicol.processes.ir import CanonicalProcessIR, ProcessLegIR

FermionColorOrientation = Literal["fundamental", "antifundamental"]
_ValueT = TypeVar("_ValueT")

NO_FERMION_LINE: Final = 0xFFFF_FFFF
PAIRING_PROOF_ALGORITHM: Final = "canonical-external-fermion-pairing-v1"


class RecurrenceFermionPairingError(ValueError):
    """A process/model contract cannot define a bounded exact pairing catalog."""


@dataclass(frozen=True, slots=True)
class FermionPairingLimitsV1:
    """Explicit guards against factorial closure-rule materialization."""

    max_endpoints: int = 16
    max_pairings_per_species: int = 720
    max_total_rules: int = 4096

    def __post_init__(self) -> None:
        for name, value in (
            ("max_endpoints", self.max_endpoints),
            ("max_pairings_per_species", self.max_pairings_per_species),
            ("max_total_rules", self.max_total_rules),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ExternalFermionEndpointRowV1:
    """One colored external fermion bound to canonical model state contracts."""

    endpoint_id: int
    source_slot: int
    public_label: int
    species_class_id: int
    species_id: str
    particle_orientation: str
    color_orientation: FermionColorOrientation
    state_template_ids: tuple[str, ...]
    anti_state_template_ids: tuple[str, ...]
    basis_ids: tuple[str, ...]
    color_representations: tuple[int, ...]
    contract_digest: str


@dataclass(frozen=True, slots=True)
class FermionPairingClassRowV1:
    """One model-compatible fundamental/antifundamental endpoint set."""

    class_id: int
    species_class_id: int
    species_id: str
    fundamental_source_slots: tuple[int, ...]
    antifundamental_source_slots: tuple[int, ...]
    reference_pairings: tuple[tuple[int, int], ...]
    pairing_count: int
    proof_digest: str


@dataclass(frozen=True, slots=True)
class FermionPairingRuleRowV1:
    """One exact product of species-local Wick pairings."""

    rule_id: int
    class_pairing_indices: tuple[tuple[int, int], ...]
    endpoint_pairings: tuple[tuple[int, int], ...]
    source_slot_permutation: tuple[int, ...]
    lineage_by_source_slot: tuple[int, ...]
    fermion_parity: int
    exact_factor: ExactComplexRationalV1
    multiplicity: int
    proof_algorithm: str
    proof_digest: str


@dataclass(frozen=True, slots=True)
class FermionPairingCatalogV1:
    """Immutable endpoint, pairing-class, and closure reconstruction rows."""

    process_key: str
    source_count: int
    endpoints: tuple[ExternalFermionEndpointRowV1, ...]
    pairing_classes: tuple[FermionPairingClassRowV1, ...]
    rules: tuple[FermionPairingRuleRowV1, ...]
    topology_digest: str
    semantic_digest: str


@dataclass(frozen=True, slots=True)
class FermionPairingRootEndpointV1:
    """One compact endpoint retained before any pairing is selected."""

    source_slot: int
    color_orientation: FermionColorOrientation
    contract_digest: str


@dataclass(frozen=True, slots=True)
class FermionPairingRootClassV1:
    """One model-proven pairing class without factorial rule expansion."""

    species: str
    proof_digest: str
    fundamental_source_slots: tuple[int, ...]
    antifundamental_source_slots: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FermionPairingRootsV1:
    """O(external-leg) roots from which one runtime pairing can be derived."""

    endpoints: tuple[FermionPairingRootEndpointV1, ...]
    classes: tuple[FermionPairingRootClassV1, ...]


@dataclass(frozen=True, slots=True)
class _EndpointContract:
    source_slot: int
    public_label: int
    species_id: str
    particle_orientation: str
    color_orientation: FermionColorOrientation
    state_template_ids: tuple[str, ...]
    anti_state_template_ids: tuple[str, ...]
    basis_ids: tuple[str, ...]
    color_representations: tuple[int, ...]
    contract_digest: str


@dataclass(frozen=True, slots=True)
class _LocalPairing:
    index: int
    antifundamental_slots: tuple[int, ...]
    parity: int


def build_recurrence_fermion_pairing_roots_v1(
    process: CanonicalProcessIR,
    current_states: Sequence[CurrentStateTemplateV1],
    *,
    quantum_flows: Sequence[QuantumFlowTemplateV1] = (),
) -> FermionPairingRootsV1:
    """Derive compact model-compatible pairing roots without enumerating rules.

    The ordinary recurrence builder expands permutations because it publishes a
    complete process schedule.  The on-the-fly lane instead selects one pairing
    at runtime, so retaining endpoints and compatibility classes is sufficient
    and scales linearly with the external process.  No resource-policy ceiling
    belongs at this semantic projection boundary.
    """

    if not isinstance(process, CanonicalProcessIR):
        raise TypeError("fermion pairing roots require a CanonicalProcessIR")
    states = tuple(current_states)
    if any(not isinstance(state, CurrentStateTemplateV1) for state in states):
        raise TypeError(
            "fermion pairing root current states must be CurrentStateTemplateV1 rows"
        )
    flows = tuple(quantum_flows)
    if any(not isinstance(flow, QuantumFlowTemplateV1) for flow in flows):
        raise TypeError(
            "fermion pairing root quantum flows must be QuantumFlowTemplateV1 rows"
        )

    contracts = tuple(
        contract
        for source_slot, leg in enumerate(process.legs)
        if (contract := _derive_endpoint(source_slot, leg, states)) is not None
    )
    compatibility_ids = _fermion_compatibility_ids(states, flows)
    grouped: dict[str, list[_EndpointContract]] = {}
    for contract in contracts:
        species = compatibility_ids.get(contract.species_id, contract.species_id)
        grouped.setdefault(species, []).append(contract)

    classes: list[FermionPairingRootClassV1] = []
    for species, group in sorted(
        grouped.items(),
        key=lambda item: tuple(sorted(row.source_slot for row in item[1])),
    ):
        fundamental = tuple(
            sorted(
                row.source_slot
                for row in group
                if row.color_orientation == "fundamental"
            )
        )
        antifundamental = tuple(
            sorted(
                row.source_slot
                for row in group
                if row.color_orientation == "antifundamental"
            )
        )
        if not fundamental or len(fundamental) != len(antifundamental):
            raise RecurrenceFermionPairingError(
                "incompatible fermion species endpoints: species contract "
                f"{species!r} has {len(fundamental)} fundamental and "
                f"{len(antifundamental)} antifundamental endpoints"
            )
        proof_payload = {
            "algorithm": PAIRING_PROOF_ALGORITHM,
            "antifundamental_source_slots": antifundamental,
            "fundamental_source_slots": fundamental,
            "reference_pairings": tuple(
                zip(fundamental, antifundamental, strict=True)
            ),
            "species_contract_digests": tuple(
                sorted(row.contract_digest for row in group)
            ),
        }
        classes.append(
            FermionPairingRootClassV1(
                species=species,
                proof_digest=_digest(proof_payload),
                fundamental_source_slots=fundamental,
                antifundamental_source_slots=antifundamental,
            )
        )

    return FermionPairingRootsV1(
        endpoints=tuple(
            FermionPairingRootEndpointV1(
                source_slot=contract.source_slot,
                color_orientation=contract.color_orientation,
                contract_digest=contract.contract_digest,
            )
            for contract in sorted(contracts, key=lambda row: row.source_slot)
        ),
        classes=tuple(classes),
    )


def build_recurrence_fermion_pairing_catalog_v1(
    process: CanonicalProcessIR,
    current_states: Sequence[CurrentStateTemplateV1],
    *,
    quantum_flows: Sequence[QuantumFlowTemplateV1] = (),
    limits: FermionPairingLimitsV1 | None = None,
) -> FermionPairingCatalogV1:
    """Derive all bounded, model-compatible external-fermion pairings.

    Color-singlet fermions do not define LC open-string endpoints and are
    intentionally absent.  A process with no colored fermion endpoints has one
    trivial parity-even rule, which is convenient for later closure lowering.

    Fermion species connected by an authenticated model quantum flow share a
    pairing class.  This is required for charged-current lines, where a
    fundamental endpoint can carry a different species from its
    antifundamental endpoint.  In the absence of such a witness the original
    same-species rule remains exact.
    """

    if not isinstance(process, CanonicalProcessIR):
        raise TypeError("fermion pairing requires a CanonicalProcessIR")
    active_limits = limits or FermionPairingLimitsV1()
    if not isinstance(active_limits, FermionPairingLimitsV1):
        raise TypeError("fermion pairing limits must be FermionPairingLimitsV1")
    states = tuple(current_states)
    if any(not isinstance(state, CurrentStateTemplateV1) for state in states):
        raise TypeError(
            "fermion pairing current states must be CurrentStateTemplateV1 rows"
        )
    flows = tuple(quantum_flows)
    if any(not isinstance(flow, QuantumFlowTemplateV1) for flow in flows):
        raise TypeError(
            "fermion pairing quantum flows must be QuantumFlowTemplateV1 rows"
        )

    contracts = tuple(
        contract
        for source_slot, leg in enumerate(process.legs)
        if (contract := _derive_endpoint(source_slot, leg, states)) is not None
    )
    if len(contracts) > active_limits.max_endpoints:
        raise RecurrenceFermionPairingError(
            "colored fermion endpoint count exceeds the configured pairing bound: "
            f"{len(contracts)} > {active_limits.max_endpoints}"
        )

    compatibility_ids = _fermion_compatibility_ids(states, flows)
    grouped: dict[str, list[_EndpointContract]] = {}
    for contract in contracts:
        grouped.setdefault(
            compatibility_ids.get(contract.species_id, contract.species_id),
            [],
        ).append(contract)
    ordered_groups = sorted(
        grouped.values(),
        key=lambda group: tuple(sorted(item.source_slot for item in group)),
    )
    species_class_ids = {
        compatibility_ids.get(group[0].species_id, group[0].species_id): class_id
        for class_id, group in enumerate(ordered_groups)
    }
    endpoints = tuple(
        ExternalFermionEndpointRowV1(
            endpoint_id=endpoint_id,
            source_slot=contract.source_slot,
            public_label=contract.public_label,
            species_class_id=species_class_ids[
                compatibility_ids.get(contract.species_id, contract.species_id)
            ],
            species_id=compatibility_ids.get(
                contract.species_id, contract.species_id
            ),
            particle_orientation=contract.particle_orientation,
            color_orientation=contract.color_orientation,
            state_template_ids=contract.state_template_ids,
            anti_state_template_ids=contract.anti_state_template_ids,
            basis_ids=contract.basis_ids,
            color_representations=contract.color_representations,
            contract_digest=contract.contract_digest,
        )
        for endpoint_id, contract in enumerate(
            sorted(contracts, key=lambda item: item.source_slot)
        )
    )

    pairing_classes: list[FermionPairingClassRowV1] = []
    local_options: list[tuple[_LocalPairing, ...]] = []
    total_rule_count = 1
    for class_id, group in enumerate(ordered_groups):
        class_species_id = compatibility_ids.get(
            group[0].species_id, group[0].species_id
        )
        fundamental = tuple(
            sorted(
                item.source_slot
                for item in group
                if item.color_orientation == "fundamental"
            )
        )
        antifundamental = tuple(
            sorted(
                item.source_slot
                for item in group
                if item.color_orientation == "antifundamental"
            )
        )
        if len(fundamental) != len(antifundamental):
            raise RecurrenceFermionPairingError(
                "incompatible fermion species endpoints: species contract "
                f"{class_species_id!r} has {len(fundamental)} fundamental "
                f"and {len(antifundamental)} antifundamental endpoints"
            )
        pairing_count = math.factorial(len(fundamental))
        if pairing_count > active_limits.max_pairings_per_species:
            raise RecurrenceFermionPairingError(
                "model-compatible fermion pairings exceed the configured bound: "
                f"{pairing_count} > {active_limits.max_pairings_per_species}"
            )
        total_rule_count *= pairing_count
        if total_rule_count > active_limits.max_total_rules:
            raise RecurrenceFermionPairingError(
                "combined fermion pairing rules exceed the configured bound: "
                f"{total_rule_count} > {active_limits.max_total_rules}"
            )

        reference = tuple(zip(fundamental, antifundamental, strict=True))
        class_payload = {
            "algorithm": PAIRING_PROOF_ALGORITHM,
            "antifundamental_source_slots": antifundamental,
            "fundamental_source_slots": fundamental,
            "reference_pairings": reference,
            "species_contract_digests": tuple(
                sorted(item.contract_digest for item in group)
            ),
        }
        pairing_classes.append(
            FermionPairingClassRowV1(
                class_id=class_id,
                species_class_id=class_id,
                species_id=class_species_id,
                fundamental_source_slots=fundamental,
                antifundamental_source_slots=antifundamental,
                reference_pairings=reference,
                pairing_count=pairing_count,
                proof_digest=_digest(class_payload),
            )
        )
        local_options.append(
            tuple(
                _LocalPairing(
                    index=index,
                    antifundamental_slots=permutation,
                    parity=_permutation_parity(antifundamental, permutation),
                )
                for index, permutation in enumerate(
                    itertools.permutations(antifundamental)
                )
            )
        )

    rules = _build_rules(
        tuple(pairing_classes),
        tuple(local_options),
        source_count=len(process.legs),
    )
    topology_payload = _topology_payload(
        len(process.legs), endpoints, pairing_classes, rules
    )
    semantic_payload = {
        "process_key": process.key,
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
    return FermionPairingCatalogV1(
        process_key=process.key,
        source_count=len(process.legs),
        endpoints=endpoints,
        pairing_classes=tuple(pairing_classes),
        rules=rules,
        topology_digest=_digest(topology_payload),
        semantic_digest=_digest(semantic_payload),
    )


def _fermion_compatibility_ids(
    states: tuple[CurrentStateTemplateV1, ...],
    flows: tuple[QuantumFlowTemplateV1, ...],
) -> dict[str, str]:
    """Return exact model-proven fermion-line compatibility components."""

    state_by_id = {state.template_id: state for state in states}
    fermion_species = tuple(
        sorted({state.species_id for state in states if state.statistics == "fermion"})
    )
    parent = {species_id: species_id for species_id in fermion_species}

    def find(species_id: str) -> str:
        root = species_id
        while parent[root] != root:
            root = parent[root]
        while parent[species_id] != species_id:
            next_species = parent[species_id]
            parent[species_id] = root
            species_id = next_species
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for flow in flows:
        referenced = (
            *flow.input_state_template_ids,
            flow.result_state_template_id,
        )
        try:
            flow_states = tuple(state_by_id[template_id] for template_id in referenced)
        except KeyError as error:
            raise RecurrenceFermionPairingError(
                "fermion pairing quantum flow references an unknown current-state "
                f"template {error.args[0]!r}"
            ) from error
        species = tuple(
            sorted(
                {
                    state.species_id
                    for state in flow_states
                    if state.statistics == "fermion"
                }
            )
        )
        if len(species) == 2:
            union(*species)

    components: dict[str, list[str]] = {}
    for species_id in fermion_species:
        components.setdefault(find(species_id), []).append(species_id)
    result: dict[str, str] = {}
    for members in components.values():
        canonical_members = tuple(sorted(members))
        compatibility_id = (
            canonical_members[0]
            if len(canonical_members) == 1
            else f"model-fermion-transition-component:{_digest(canonical_members)}"
        )
        result.update(
            (species_id, compatibility_id) for species_id in canonical_members
        )
    return result


def _derive_endpoint(
    source_slot: int,
    leg: ProcessLegIR,
    states: tuple[CurrentStateTemplateV1, ...],
) -> _EndpointContract | None:
    statistics_is_fermion = leg.statistics == "fermion"
    family_is_fermion = leg.wavefunction_family == "fermion"
    if statistics_is_fermion != family_is_fermion:
        raise RecurrenceFermionPairingError(
            f"external leg {leg.label} has inconsistent fermion contracts"
        )
    if not statistics_is_fermion:
        return None
    if leg.color_role == "singlet":
        return None
    if leg.color_role not in {"fundamental", "antifundamental"}:
        raise RecurrenceFermionPairingError(
            f"fermion leg {leg.label} has unsupported LC color role {leg.color_role!r}"
        )
    if leg.outgoing_pdg is None:
        raise RecurrenceFermionPairingError(
            f"fermion leg {leg.label} has no concrete outgoing state identifier"
        )

    candidates = tuple(
        state
        for state in states
        if state.particle_id == leg.outgoing_pdg
        and state.statistics == "fermion"
        and state.auxiliary_kind is None
    )
    if not candidates:
        raise RecurrenceFermionPairingError(
            f"fermion leg {leg.label} has no canonical current-state contract"
        )
    species = _one_value(
        (state.species_id for state in candidates),
        f"fermion leg {leg.label} species",
    )
    orientation = _one_value(
        (state.orientation for state in candidates),
        f"fermion leg {leg.label} particle orientation",
    )
    anti_particle_id = _one_value(
        (state.anti_particle_id for state in candidates),
        f"fermion leg {leg.label} antiparticle state",
    )
    expected_shape = {
        "fundamental": "fundamental-open-string",
        "antifundamental": "antifundamental-open-string",
    }[leg.color_role]
    if any(state.lc_color_shape_kind != expected_shape for state in candidates):
        raise RecurrenceFermionPairingError(
            f"fermion leg {leg.label} color role disagrees with its state contract"
        )
    if leg.source_orientation != "inclusive" and orientation != leg.source_orientation:
        raise RecurrenceFermionPairingError(
            f"fermion leg {leg.label} source orientation disagrees with its state "
            "contract"
        )
    if orientation not in {"particle", "antiparticle"}:
        raise RecurrenceFermionPairingError(
            "self-conjugate or inclusive fermion pairing is not certified"
        )

    anti_orientation = "antiparticle" if orientation == "particle" else "particle"
    anti_candidates = tuple(
        state
        for state in states
        if state.particle_id == anti_particle_id
        and state.anti_particle_id == leg.outgoing_pdg
        and state.species_id == species
        and state.statistics == "fermion"
        and state.orientation == anti_orientation
        and state.auxiliary_kind is None
    )
    if not anti_candidates:
        raise RecurrenceFermionPairingError(
            f"fermion leg {leg.label} has no involutive antiparticle state contract"
        )

    state_template_ids = tuple(sorted(state.template_id for state in candidates))
    anti_state_template_ids = tuple(
        sorted(state.template_id for state in anti_candidates)
    )
    basis_ids = tuple(sorted({state.basis for state in candidates}))
    color_representations = tuple(
        sorted({state.color_representation for state in candidates})
    )
    payload = {
        "anti_state_contract_digests": tuple(
            sorted(state.semantic_digest for state in anti_candidates)
        ),
        "color_orientation": leg.color_role,
        "public_label": leg.label,
        "source_slot": source_slot,
        "state_contract_digests": tuple(
            sorted(state.semantic_digest for state in candidates)
        ),
    }
    return _EndpointContract(
        source_slot=source_slot,
        public_label=int(leg.label),
        species_id=species,
        particle_orientation=orientation,
        color_orientation=cast(FermionColorOrientation, leg.color_role),
        state_template_ids=state_template_ids,
        anti_state_template_ids=anti_state_template_ids,
        basis_ids=basis_ids,
        color_representations=color_representations,
        contract_digest=_digest(payload),
    )


def _build_rules(
    pairing_classes: tuple[FermionPairingClassRowV1, ...],
    local_options: tuple[tuple[_LocalPairing, ...], ...],
    *,
    source_count: int,
) -> tuple[FermionPairingRuleRowV1, ...]:
    combinations: Sequence[tuple[_LocalPairing, ...]] = (
        tuple(itertools.product(*local_options)) if local_options else ((),)
    )
    rules: list[FermionPairingRuleRowV1] = []
    for rule_id, selected in enumerate(combinations):
        endpoint_pairings: list[tuple[int, int]] = []
        source_permutation = list(range(source_count))
        parity = 1
        class_pairing_indices: list[tuple[int, int]] = []
        for pairing_class, local in zip(pairing_classes, selected, strict=True):
            class_pairing_indices.append((pairing_class.class_id, local.index))
            parity *= local.parity
            endpoint_pairings.extend(
                zip(
                    pairing_class.fundamental_source_slots,
                    local.antifundamental_slots,
                    strict=True,
                )
            )
            for reference_slot, selected_slot in zip(
                pairing_class.antifundamental_source_slots,
                local.antifundamental_slots,
                strict=True,
            ):
                source_permutation[reference_slot] = selected_slot
        endpoint_pairings.sort()
        lineage = [NO_FERMION_LINE] * source_count
        for line_id, (fundamental_slot, antifundamental_slot) in enumerate(
            endpoint_pairings
        ):
            lineage[fundamental_slot] = line_id
            lineage[antifundamental_slot] = line_id
        rule_payload = {
            "algorithm": PAIRING_PROOF_ALGORITHM,
            "class_pairing_indices": tuple(class_pairing_indices),
            "endpoint_pairings": tuple(endpoint_pairings),
            "fermion_parity": parity,
            "lineage_by_source_slot": tuple(lineage),
            "source_slot_permutation": tuple(source_permutation),
        }
        rules.append(
            FermionPairingRuleRowV1(
                rule_id=rule_id,
                class_pairing_indices=tuple(class_pairing_indices),
                endpoint_pairings=tuple(endpoint_pairings),
                source_slot_permutation=tuple(source_permutation),
                lineage_by_source_slot=tuple(lineage),
                fermion_parity=parity,
                exact_factor=ExactComplexRationalV1(parity, 1, 0, 1),
                multiplicity=1,
                proof_algorithm=PAIRING_PROOF_ALGORITHM,
                proof_digest=_digest(rule_payload),
            )
        )
    return tuple(rules)


def _permutation_parity(
    reference: tuple[int, ...],
    candidate: tuple[int, ...],
) -> int:
    if len(reference) != len(candidate) or set(reference) != set(candidate):
        raise RecurrenceFermionPairingError(
            "fermion parity requires a permutation of the reference endpoints"
        )
    ranks = {value: index for index, value in enumerate(reference)}
    permutation = tuple(ranks[value] for value in candidate)
    inversions = sum(
        left > right
        for index, left in enumerate(permutation)
        for right in permutation[index + 1 :]
    )
    return -1 if inversions % 2 else 1


def _one_value(values: Iterable[_ValueT], context: str) -> _ValueT:
    materialized = tuple(values)
    unique = set(materialized)
    if len(unique) != 1:
        raise RecurrenceFermionPairingError(
            f"{context} must be uniquely defined, found {materialized!r}"
        )
    return materialized[0]


def _topology_payload(
    source_count: int,
    endpoints: tuple[ExternalFermionEndpointRowV1, ...],
    pairing_classes: Sequence[FermionPairingClassRowV1],
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


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "NO_FERMION_LINE",
    "PAIRING_PROOF_ALGORITHM",
    "ExternalFermionEndpointRowV1",
    "FermionPairingCatalogV1",
    "FermionPairingClassRowV1",
    "FermionPairingLimitsV1",
    "FermionPairingRootClassV1",
    "FermionPairingRootEndpointV1",
    "FermionPairingRootsV1",
    "FermionPairingRuleRowV1",
    "RecurrenceFermionPairingError",
    "build_recurrence_fermion_pairing_catalog_v1",
    "build_recurrence_fermion_pairing_roots_v1",
]
