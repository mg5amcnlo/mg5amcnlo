# SPDX-License-Identifier: 0BSD
"""Proof-gated reusable-helicity recurrence metadata.

This module derives the model-generic contract describing which currents have
the same recurrence shape once runtime source states are treated as inputs.
Compiled generation retains the shared summed-helicity graph and attaches the
certified selector closures to it. Rusticol routes explicit runtime helicity
selectors through those closures; independently unproven currents remain
localized residuals rather than disabling proven reuse classes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import fsum
from typing import TypeAlias

from ..models.base import Model
from .contracts import runtime_coupling_parameter_names
from .dag_types import AmplitudeRoot, CurrentNode, GenericDAG, InteractionNode

HELICITY_RECURRENCE_CONTRACT_VERSION = 1
HELICITY_RECURRENCE_PROOF_ALGORITHM = (
    "canonical-source-transition-dependency-shape-v1"
)
RUNTIME_SELECTOR_PROVENANCE = "pyamplicol-runtime-selectors-v1"

_ComplexWeight: TypeAlias = tuple[float, float]
_SelectorState: TypeAlias = tuple[tuple[int, int], ...]
_ColorSectorAliases: TypeAlias = Mapping[int, int]


@dataclass(frozen=True, slots=True)
class HelicitySelectorDomain:
    """One partial or complete assignment of external source helicities."""

    id: int
    source_states: _SelectorState
    complete: bool

    def to_runtime_manifest(self) -> dict[str, object]:
        return {
            "id": self.id,
            "complete": self.complete,
            "source_states": [
                {"external_label": label, "helicity": helicity}
                for label, helicity in self.source_states
            ],
        }


@dataclass(frozen=True, slots=True)
class HelicityCurrentReplayMember:
    current_id: int
    selector_domain_id: int
    factor: _ComplexWeight = (1.0, 0.0)

    def to_runtime_manifest(self) -> dict[str, object]:
        return {
            "current_id": self.current_id,
            "selector_domain_id": self.selector_domain_id,
            "factor": list(self.factor),
        }


@dataclass(frozen=True, slots=True)
class HelicityRecurrenceClass:
    """Currents sharing one exact source/transition dependency shape."""

    class_id: str
    representative_current_id: int
    external_labels: tuple[int, ...]
    source_class: bool
    members: tuple[HelicityCurrentReplayMember, ...]
    proof_digest: str
    transition_contract_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.class_id:
            raise ValueError("helicity recurrence class id must not be empty")
        if not self.members:
            raise ValueError("helicity recurrence class must have members")
        if self.representative_current_id not in {
            member.current_id for member in self.members
        }:
            raise ValueError("helicity recurrence representative is not a member")
        _validate_digest(self.proof_digest, context="helicity recurrence proof")

    @property
    def optimized(self) -> bool:
        return len(self.members) > 1

    def to_runtime_manifest(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "representative_current_id": self.representative_current_id,
            "external_labels": list(self.external_labels),
            "source_class": self.source_class,
            "members": [member.to_runtime_manifest() for member in self.members],
            "proof": {
                "status": "proven",
                "algorithm": HELICITY_RECURRENCE_PROOF_ALGORITHM,
                "digest": self.proof_digest,
                "transition_contract_ids": list(self.transition_contract_ids),
            },
        }


@dataclass(frozen=True, slots=True)
class HelicitySourceStateMapping:
    current_id: int
    external_label: int
    helicity: int
    chirality: int
    spin_state: int | tuple[int, ...]
    declared_state_index: int
    selector_domain_id: int
    recurrence_class_id: str
    representative_current_id: int
    source_contract_digest: str
    factor: _ComplexWeight = (1.0, 0.0)

    def __post_init__(self) -> None:
        _validate_digest(
            self.source_contract_digest,
            context="helicity source contract",
        )

    def to_runtime_manifest(self) -> dict[str, object]:
        spin_state: object = self.spin_state
        if isinstance(spin_state, tuple):
            spin_state = list(spin_state)
        return {
            "current_id": self.current_id,
            "external_label": self.external_label,
            "helicity": self.helicity,
            "chirality": self.chirality,
            "spin_state": spin_state,
            "declared_state_index": self.declared_state_index,
            "selector_domain_id": self.selector_domain_id,
            "recurrence_class_id": self.recurrence_class_id,
            "representative_current_id": self.representative_current_id,
            "source_contract_digest": self.source_contract_digest,
            "factor": list(self.factor),
        }


@dataclass(frozen=True, slots=True)
class HelicityAmplitudeReplayMember:
    root_id: int
    selector_domain_ids: tuple[int, ...]
    factor: _ComplexWeight = (1.0, 0.0)

    def to_runtime_manifest(self) -> dict[str, object]:
        return {
            "root_id": self.root_id,
            "selector_domain_ids": list(self.selector_domain_ids),
            "factor": list(self.factor),
        }


@dataclass(frozen=True, slots=True)
class HelicityAmplitudeReplayClass:
    class_id: str
    representative_root_id: int
    members: tuple[HelicityAmplitudeReplayMember, ...]
    proof_digest: str
    transition_contract_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("helicity amplitude recurrence class must have members")
        if self.representative_root_id not in {
            member.root_id for member in self.members
        }:
            raise ValueError("helicity amplitude representative is not a member")
        _validate_digest(self.proof_digest, context="helicity amplitude proof")

    @property
    def optimized(self) -> bool:
        return len(self.members) > 1

    def to_runtime_manifest(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "representative_root_id": self.representative_root_id,
            "members": [member.to_runtime_manifest() for member in self.members],
            "proof": {
                "status": "proven",
                "algorithm": HELICITY_RECURRENCE_PROOF_ALGORITHM,
                "digest": self.proof_digest,
                "transition_contract_ids": list(self.transition_contract_ids),
            },
        }


@dataclass(frozen=True, slots=True)
class HelicityRecurrencePlan:
    """Strict additive contract for reusable complete-helicity artifacts."""

    current_count: int
    amplitude_root_count: int
    selector_domains: tuple[HelicitySelectorDomain, ...]
    source_state_mappings: tuple[HelicitySourceStateMapping, ...]
    recurrence_classes: tuple[HelicityRecurrenceClass, ...]
    amplitude_classes: tuple[HelicityAmplitudeReplayClass, ...]
    residual_current_ids: tuple[int, ...]
    residual_root_ids: tuple[int, ...]
    structural_zero_selector_domain_ids: tuple[int, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.current_count < 0 or self.amplitude_root_count < 0:
            raise ValueError("helicity recurrence object counts must be nonnegative")
        domain_ids = tuple(domain.id for domain in self.selector_domains)
        if domain_ids != tuple(range(len(domain_ids))):
            raise ValueError("helicity selector domain ids must be contiguous")
        known_domains = set(domain_ids)
        referenced_domains = {
            member.selector_domain_id
            for recurrence in self.recurrence_classes
            for member in recurrence.members
        } | {
            domain_id
            for recurrence in self.amplitude_classes
            for member in recurrence.members
            for domain_id in member.selector_domain_ids
        } | set(self.structural_zero_selector_domain_ids)
        if not referenced_domains <= known_domains:
            raise ValueError("helicity replay references an unknown selector domain")
        class_current_ids = {
            member.current_id
            for recurrence in self.recurrence_classes
            for member in recurrence.members
        }
        if class_current_ids & set(self.residual_current_ids):
            raise ValueError("helicity recurrence classes overlap residual currents")
        if class_current_ids | set(self.residual_current_ids) != set(
            range(self.current_count)
        ):
            raise ValueError(
                "helicity recurrence current ids do not cover the final dense DAG"
            )
        class_root_ids = {
            member.root_id
            for recurrence in self.amplitude_classes
            for member in recurrence.members
        }
        if class_root_ids & set(self.residual_root_ids):
            raise ValueError("helicity amplitude classes overlap residual roots")
        if class_root_ids | set(self.residual_root_ids) != set(
            range(self.amplitude_root_count)
        ):
            raise ValueError(
                "helicity recurrence root ids do not cover the final dense DAG"
            )
        source_ids = {mapping.current_id for mapping in self.source_state_mappings}
        if not source_ids <= class_current_ids:
            raise ValueError("helicity source mapping refers to a residual current")

    @property
    def optimized_class_count(self) -> int:
        return sum(item.optimized for item in self.recurrence_classes)

    @property
    def optimized_current_count(self) -> int:
        return sum(
            len(item.members) for item in self.recurrence_classes if item.optimized
        )

    @property
    def optimized_amplitude_class_count(self) -> int:
        return sum(item.optimized for item in self.amplitude_classes)

    @property
    def physical_helicity_count(self) -> int:
        return sum(domain.complete for domain in self.selector_domains)

    def proof_counts(self) -> dict[str, int]:
        return {
            "recurrence_class_count": len(self.recurrence_classes),
            "optimized_recurrence_class_count": self.optimized_class_count,
            "optimized_current_count": self.optimized_current_count,
            "residual_current_count": len(self.residual_current_ids),
            "amplitude_class_count": len(self.amplitude_classes),
            "optimized_amplitude_class_count": (
                self.optimized_amplitude_class_count
            ),
            "residual_amplitude_count": len(self.residual_root_ids),
            "source_state_mapping_count": len(self.source_state_mappings),
            "physical_helicity_count": self.physical_helicity_count,
            "structural_zero_helicity_count": len(
                self.structural_zero_selector_domain_ids
            ),
        }

    def to_json_dict(self) -> dict[str, object]:
        return self.to_runtime_manifest()

    def to_runtime_manifest(self) -> dict[str, object]:
        return {
            "kind": "pyamplicol-helicity-recurrence",
            "contract_version": HELICITY_RECURRENCE_CONTRACT_VERSION,
            "proof_algorithm": HELICITY_RECURRENCE_PROOF_ALGORITHM,
            "current_count": self.current_count,
            "amplitude_root_count": self.amplitude_root_count,
            "proof_counts": self.proof_counts(),
            "selector_domains": [
                domain.to_runtime_manifest() for domain in self.selector_domains
            ],
            "source_state_mappings": [
                mapping.to_runtime_manifest()
                for mapping in self.source_state_mappings
            ],
            "recurrence_classes": [
                recurrence.to_runtime_manifest()
                for recurrence in self.recurrence_classes
            ],
            "amplitude_classes": [
                recurrence.to_runtime_manifest()
                for recurrence in self.amplitude_classes
            ],
            "residual_current_ids": list(self.residual_current_ids),
            "residual_root_ids": list(self.residual_root_ids),
            "structural_zero_selector_domain_ids": list(
                self.structural_zero_selector_domain_ids
            ),
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class _CurrentProof:
    class_index: int | None
    factor: _ComplexWeight
    selector_state: _SelectorState
    residual_reason: str | None = None


@dataclass(slots=True)
class _PendingCurrentClass:
    class_id: str
    representative_current_id: int
    external_labels: tuple[int, ...]
    source_class: bool
    proof_digest: str
    transition_contract_ids: tuple[str, ...]
    members: list[tuple[int, _SelectorState, _ComplexWeight]]


@dataclass(slots=True)
class _PendingAmplitudeClass:
    class_id: str
    representative_root_id: int
    proof_digest: str
    transition_contract_ids: tuple[str, ...]
    members: list[tuple[int, tuple[_SelectorState, ...], _ComplexWeight]]


@dataclass(frozen=True, slots=True)
class _SemanticAmplitudeRootSignature:
    """Exact model-certified identity used only for construction aliases."""

    proof_digest: str
    selector_states: tuple[_SelectorState, ...]
    factor: _ComplexWeight


def _derive_current_replay_proofs(
    dag: GenericDAG,
    model: Model,
    *,
    color_sector_aliases: _ColorSectorAliases | None = None,
) -> tuple[
    dict[int, tuple[int, int]],
    dict[int, dict[str, object]],
    list[_PendingCurrentClass],
    tuple[_CurrentProof, ...],
    list[int],
    list[str],
]:
    """Build the shared exact recursive-current proof graph.

    ``color_sector_aliases`` projects only the traversal-sector identifier.
    Every external label, ordered traversal, line group, flavour flow, kernel
    contract, coupling and coefficient remains part of the proof.  The normal
    helicity-replay path passes no projection; contracted open-line
    construction uses the projection to recognize only proven block-order
    aliases.
    """

    source_by_bit, source_contracts, source_diagnostics = _source_contracts(
        dag,
        model,
    )
    interactions_by_result: dict[int, list[InteractionNode]] = defaultdict(list)
    for interaction in dag.interactions:
        interactions_by_result[int(interaction.result_id)].append(interaction)

    pending_classes: list[_PendingCurrentClass] = []
    class_by_signature: dict[str, int] = {}
    current_proofs: list[_CurrentProof | None] = [None] * len(dag.currents)
    residual_current_ids: list[int] = []
    diagnostics = list(source_diagnostics)

    ordered_currents = sorted(
        dag.currents,
        key=lambda current: (len(current.index.external_labels), current.id),
    )
    for current in ordered_currents:
        selector_state, selector_error = _selector_state_for_ancestry(
            int(current.index.helicity_ancestry),
            source_by_bit,
        )
        if selector_error is not None:
            _record_residual_current(
                current,
                selector_state,
                selector_error,
                current_proofs,
                residual_current_ids,
                diagnostics,
            )
            continue

        if current.is_source:
            source_contract = source_contracts.get(current.id)
            if source_contract is None:
                _record_residual_current(
                    current,
                    selector_state,
                    "source state has no exact SourceIR contract",
                    current_proofs,
                    residual_current_ids,
                    diagnostics,
                )
                continue
            signature_payload = {
                "kind": "source",
                "current_contract": _current_contract(
                    current,
                    include_spin_state=False,
                    color_sector_aliases=color_sector_aliases,
                ),
                "source_topology": source_contract["topology"],
                "source_contract_digest": source_contract["digest"],
            }
            class_index, factor = _register_current_class(
                current,
                selector_state,
                signature_payload,
                transition_contract_ids=(),
                source_class=True,
                class_by_signature=class_by_signature,
                pending_classes=pending_classes,
            )
            current_proofs[current.id] = _CurrentProof(
                class_index=class_index,
                factor=factor,
                selector_state=selector_state,
            )
            continue

        candidate = _generated_current_signature(
            dag,
            current,
            interactions_by_result[current.id],
            model,
            current_proofs,
            color_sector_aliases=color_sector_aliases,
        )
        if isinstance(candidate, str):
            _record_residual_current(
                current,
                selector_state,
                candidate,
                current_proofs,
                residual_current_ids,
                diagnostics,
            )
            continue
        signature_payload, transition_contract_ids = candidate
        class_index, factor = _register_current_class(
            current,
            selector_state,
            signature_payload,
            transition_contract_ids=transition_contract_ids,
            source_class=False,
            class_by_signature=class_by_signature,
            pending_classes=pending_classes,
        )
        current_proofs[current.id] = _CurrentProof(
            class_index=class_index,
            factor=factor,
            selector_state=selector_state,
        )

    if any(item is None for item in current_proofs):
        raise ValueError("helicity recurrence derivation left a current unclassified")
    return (
        source_by_bit,
        source_contracts,
        pending_classes,
        tuple(item for item in current_proofs if item is not None),
        residual_current_ids,
        diagnostics,
    )


def build_helicity_recurrence_plan(
    dag: GenericDAG,
    model: Model,
) -> HelicityRecurrencePlan | None:
    """Derive reusable recurrence classes for a complete-helicity DAG.

    The selected-helicity generation path deliberately returns ``None`` and is
    therefore byte-for-byte unchanged.  Proof failures are attached only to the
    affected current and descendants that depend on it; independent classes are
    retained.
    """

    if dag.helicity_coverage != "complete" or dag.selected_source_helicities:
        return None

    (
        source_by_bit,
        source_contracts,
        pending_classes,
        proven_currents,
        residual_current_ids,
        diagnostics,
    ) = _derive_current_replay_proofs(dag, model)

    pending_amplitudes, residual_root_ids, root_diagnostics = _amplitude_classes(
        dag,
        model,
        proven_currents,
        source_by_bit,
    )
    diagnostics.extend(root_diagnostics)

    residual_root_selector_states: dict[int, tuple[_SelectorState, ...]] = {}
    for root_id in residual_root_ids:
        selector_states, selector_error = _root_selector_states(
            dag.amplitude_roots[root_id],
            dag,
            source_by_bit,
        )
        if selector_error is not None:
            diagnostics.append(
                f"amplitude root {root_id}: residual selector routing failed: "
                f"{selector_error}"
            )
            continue
        residual_root_selector_states[root_id] = selector_states

    possible_helicities = _possible_helicity_states(dag, model)
    represented_helicities = {
        state
        for recurrence in pending_amplitudes
        for _root_id, states, _factor in recurrence.members
        for state in states
    } | {
        state
        for states in residual_root_selector_states.values()
        for state in states
    }
    structural_zeros = tuple(
        sorted(set(possible_helicities) - represented_helicities)
    )
    all_selector_states = {
        proof.selector_state for proof in proven_currents
    } | {
        state
        for recurrence in pending_amplitudes
        for _root_id, states, _factor in recurrence.members
        for state in states
    } | {
        state
        for states in residual_root_selector_states.values()
        for state in states
    } | set(structural_zeros)
    domain_id_by_state = {
        state: index for index, state in enumerate(sorted(all_selector_states))
    }
    complete_label_count = len(
        [leg for leg in dag.process.legs if leg.outgoing_pdg is not None]
    )
    selector_domains = tuple(
        HelicitySelectorDomain(
            id=index,
            source_states=state,
            complete=len(state) == complete_label_count,
        )
        for state, index in sorted(
            domain_id_by_state.items(),
            key=lambda item: item[1],
        )
    )

    recurrence_classes = tuple(
        HelicityRecurrenceClass(
            class_id=item.class_id,
            representative_current_id=item.representative_current_id,
            external_labels=item.external_labels,
            source_class=item.source_class,
            members=tuple(
                HelicityCurrentReplayMember(
                    current_id=current_id,
                    selector_domain_id=domain_id_by_state[state],
                    factor=factor,
                )
                for current_id, state, factor in item.members
            ),
            proof_digest=item.proof_digest,
            transition_contract_ids=item.transition_contract_ids,
        )
        for item in pending_classes
    )
    amplitude_classes = tuple(
        HelicityAmplitudeReplayClass(
            class_id=item.class_id,
            representative_root_id=item.representative_root_id,
            members=tuple(
                HelicityAmplitudeReplayMember(
                    root_id=root_id,
                    selector_domain_ids=tuple(
                        domain_id_by_state[state] for state in states
                    ),
                    factor=factor,
                )
                for root_id, states, factor in item.members
            ),
            proof_digest=item.proof_digest,
            transition_contract_ids=item.transition_contract_ids,
        )
        for item in pending_amplitudes
    )

    class_by_current = {
        member.current_id: recurrence
        for recurrence in recurrence_classes
        for member in recurrence.members
    }
    source_state_mappings: list[HelicitySourceStateMapping] = []
    for current in dag.currents:
        if not current.is_source or current.id not in source_contracts:
            continue
        recurrence = class_by_current.get(current.id)
        if recurrence is None:
            continue
        member = next(
            member for member in recurrence.members if member.current_id == current.id
        )
        source_contract = source_contracts[current.id]
        source_state_mappings.append(
            HelicitySourceStateMapping(
                current_id=current.id,
                external_label=int(current.source_leg_label or 0),
                helicity=int(current.source_helicity or 0),
                chirality=int(current.index.chirality),
                spin_state=current.index.spin_state,
                declared_state_index=int(source_contract["declared_state_index"]),
                selector_domain_id=member.selector_domain_id,
                recurrence_class_id=recurrence.class_id,
                representative_current_id=recurrence.representative_current_id,
                source_contract_digest=str(source_contract["digest"]),
                factor=member.factor,
            )
        )

    return HelicityRecurrencePlan(
        current_count=len(dag.currents),
        amplitude_root_count=len(dag.amplitude_roots),
        selector_domains=selector_domains,
        source_state_mappings=tuple(source_state_mappings),
        recurrence_classes=recurrence_classes,
        amplitude_classes=amplitude_classes,
        residual_current_ids=tuple(sorted(residual_current_ids)),
        residual_root_ids=tuple(sorted(residual_root_ids)),
        structural_zero_selector_domain_ids=tuple(
            domain_id_by_state[state] for state in structural_zeros
        ),
        diagnostics=tuple(diagnostics),
    )


def _source_contracts(
    dag: GenericDAG,
    model: Model,
) -> tuple[
    dict[int, tuple[int, int]],
    dict[int, dict[str, object]],
    tuple[str, ...],
]:
    """Return ancestry-bit source selectors and exact SourceIR contracts."""

    leg_by_label = {int(leg.label): leg for leg in dag.process.legs}
    source_by_bit: dict[int, tuple[int, int]] = {}
    contracts: dict[int, dict[str, object]] = {}
    diagnostics: list[str] = []
    for current in dag.currents:
        if not current.is_source:
            continue
        ancestry = int(current.index.helicity_ancestry)
        label = current.source_leg_label
        helicity = current.source_helicity
        if (
            ancestry <= 0
            or ancestry & (ancestry - 1)
            or label is None
            or helicity is None
        ):
            diagnostics.append(
                f"source current {current.id} has non-canonical ancestry metadata"
            )
            continue
        bit = ancestry.bit_length() - 1
        selector = (int(label), int(helicity))
        previous = source_by_bit.setdefault(bit, selector)
        if previous != selector:
            diagnostics.append(
                f"source ancestry bit {bit} maps to inconsistent physical states"
            )
            continue
        leg = leg_by_label.get(int(label))
        if leg is None or leg.outgoing_pdg is None:
            diagnostics.append(
                f"source current {current.id} refers to absent external leg {label}"
            )
            continue
        try:
            source_ir = model._source_ir(int(leg.outgoing_pdg))
            matched_index = None
            for index, declared in enumerate(source_ir.states):
                state = (
                    source_ir.crossing.apply(declared)
                    if leg.is_initial
                    else declared
                )
                if (
                    int(state.helicity) == int(helicity)
                    and int(state.chirality) == int(current.index.chirality)
                    and state.spin_state == current.index.spin_state
                ):
                    matched_index = index
                    break
            if matched_index is None:
                diagnostics.append(
                    f"source current {current.id} is not declared by its SourceIR"
                )
                continue
            exact_payload = {
                "source_ir": source_ir.to_json_dict(),
                "initial_state_crossed": bool(leg.is_initial),
            }
            contracts[current.id] = {
                "digest": _digest(exact_payload),
                "declared_state_index": matched_index,
                "topology": {
                    "external_label": int(label),
                    "statistics": source_ir.statistics,
                    "wavefunction_family": source_ir.wavefunction_family,
                    "component_dimension": int(source_ir.component_dimension),
                    "basis": source_ir.basis,
                    "orientation": source_ir.identity.orientation,
                    "self_conjugate": bool(source_ir.identity.self_conjugate),
                    "chirality": int(current.index.chirality),
                    "spin_state_shape": _spin_state_shape(
                        current.index.spin_state
                    ),
                    "crossing": source_ir.crossing.to_json_dict(),
                    "has_mass_parameter": source_ir.mass_parameter is not None,
                    "has_width_parameter": source_ir.width_parameter is not None,
                },
            }
        except Exception as error:
            diagnostics.append(
                f"source current {current.id} contract derivation failed: {error}"
            )
    return source_by_bit, contracts, tuple(diagnostics)


def _generated_current_signature(
    dag: GenericDAG,
    current: CurrentNode,
    interactions: Sequence[InteractionNode],
    model: Model,
    current_proofs: Sequence[_CurrentProof | None],
    *,
    color_sector_aliases: _ColorSectorAliases | None = None,
) -> tuple[dict[str, object], tuple[str, ...]] | str:
    terms_by_key: dict[str, list[_ComplexWeight]] = defaultdict(list)
    transition_contracts: set[str] = set()
    for interaction in interactions:
        left = current_proofs[interaction.left_id]
        right = current_proofs[interaction.right_id]
        if left is None or right is None:
            return "transition depends on an unclassified parent current"
        if left.class_index is None or right.class_index is None:
            return "transition depends on a localized residual parent current"
        try:
            equivalence = model.vertex_evaluation_equivalence(
                int(interaction.vertex_kind)
            )
        except Exception as error:
            return f"vertex transition contract failed: {error}"
        if not equivalence.verified or not equivalence.class_id:
            return (
                "vertex transition has no verified canonical kernel contract: "
                f"kind {interaction.vertex_kind}"
            )
        transition_contracts.add(equivalence.class_id)
        parent_classes = (left.class_index, right.class_index)
        if equivalence.input_order == (1, 0):
            parent_classes = (parent_classes[1], parent_classes[0])
        kernel_factor = equivalence.factor
        if (
            equivalence.input_exchange_factor is not None
            and parent_classes[1] < parent_classes[0]
        ):
            parent_classes = (parent_classes[1], parent_classes[0])
            kernel_factor = _complex_mul(
                kernel_factor,
                equivalence.input_exchange_factor,
            )
        coupling_identity = _runtime_coupling_identity(model, interaction)
        term_key = _canonical_json(
            {
                "kernel_contract_id": equivalence.class_id,
                "parent_recurrence_classes": list(parent_classes),
                "result_particle_id": int(current.index.particle_id),
                "result_chirality": int(current.index.chirality),
                "coupling_identity": coupling_identity,
            }
        )
        coefficient = _complex_mul(
            interaction.color_weight,
            _complex_mul(kernel_factor, _complex_mul(left.factor, right.factor)),
        )
        terms_by_key[term_key].append(coefficient)

    terms = []
    for term_key in sorted(terms_by_key):
        coefficient = _sum_weights(terms_by_key[term_key])
        if coefficient == (0.0, 0.0):
            continue
        terms.append(
            {
                "transition": json.loads(term_key),
                "coefficient": list(coefficient),
            }
        )
    try:
        propagator = model._propagator_ir(
            int(current.index.particle_id),
            int(current.index.chirality),
        ).to_json_dict()
    except Exception as error:
        return f"current finalization contract failed: {error}"
    return (
        {
            "kind": "generated-current",
            "current_contract": _current_contract(
                current,
                color_sector_aliases=color_sector_aliases,
            ),
            "propagator_contract": propagator,
            "terms": terms,
        },
        tuple(sorted(transition_contracts)),
    )


def _register_current_class(
    current: CurrentNode,
    selector_state: _SelectorState,
    signature_payload: Mapping[str, object],
    *,
    transition_contract_ids: tuple[str, ...],
    source_class: bool,
    class_by_signature: dict[str, int],
    pending_classes: list[_PendingCurrentClass],
) -> tuple[int, _ComplexWeight]:
    direct_signature = _canonical_json(signature_payload)
    opposite_payload = _negated_signature_payload(signature_payload)
    opposite_signature = _canonical_json(opposite_payload)
    class_index = class_by_signature.get(direct_signature)
    factor: _ComplexWeight = (1.0, 0.0)
    if class_index is None and opposite_signature != direct_signature:
        class_index = class_by_signature.get(opposite_signature)
        if class_index is not None:
            factor = (-1.0, 0.0)
    if class_index is None:
        proof_digest = hashlib.sha256(direct_signature.encode("ascii")).hexdigest()
        class_index = len(pending_classes)
        class_by_signature[direct_signature] = class_index
        pending_classes.append(
            _PendingCurrentClass(
                class_id=f"helicity-current-sha256:{proof_digest}",
                representative_current_id=current.id,
                external_labels=tuple(current.index.external_labels),
                source_class=source_class,
                proof_digest=proof_digest,
                transition_contract_ids=transition_contract_ids,
                members=[],
            )
        )
    pending_classes[class_index].members.append(
        (current.id, selector_state, factor)
    )
    return class_index, factor


def _record_residual_current(
    current: CurrentNode,
    selector_state: _SelectorState,
    reason: str,
    current_proofs: list[_CurrentProof | None],
    residual_current_ids: list[int],
    diagnostics: list[str],
) -> None:
    current_proofs[current.id] = _CurrentProof(
        class_index=None,
        factor=(1.0, 0.0),
        selector_state=selector_state,
        residual_reason=reason,
    )
    residual_current_ids.append(current.id)
    diagnostics.append(f"current {current.id}: {reason}")


def _amplitude_classes(
    dag: GenericDAG,
    model: Model,
    current_proofs: Sequence[_CurrentProof],
    source_by_bit: Mapping[int, tuple[int, int]],
    *,
    color_sector_aliases: _ColorSectorAliases | None = None,
) -> tuple[list[_PendingAmplitudeClass], list[int], list[str]]:
    pending: list[_PendingAmplitudeClass] = []
    class_by_signature: dict[str, int] = {}
    residual: list[int] = []
    diagnostics: list[str] = []
    for root in dag.amplitude_roots:
        left = current_proofs[root.left_id]
        right = current_proofs[root.right_id]
        selector_states, selector_error = _root_selector_states(
            root,
            dag,
            source_by_bit,
        )
        if selector_error is not None:
            residual.append(root.id)
            diagnostics.append(f"amplitude root {root.id}: {selector_error}")
            continue
        if left.class_index is None or right.class_index is None:
            residual.append(root.id)
            diagnostics.append(
                f"amplitude root {root.id}: closure depends on a residual current"
            )
            continue

        parent_classes = (left.class_index, right.class_index)
        factor = _complex_mul(left.factor, right.factor)
        contract_ids: tuple[str, ...] = ()
        if root.vertex_kind is not None:
            try:
                equivalence = model.vertex_evaluation_equivalence(root.vertex_kind)
            except Exception as error:
                residual.append(root.id)
                diagnostics.append(
                    f"amplitude root {root.id}: closure contract failed: {error}"
                )
                continue
            if not equivalence.verified or not equivalence.class_id:
                residual.append(root.id)
                diagnostics.append(
                    f"amplitude root {root.id}: unverified closure kernel "
                    f"{root.vertex_kind}"
                )
                continue
            if equivalence.input_order == (1, 0):
                parent_classes = (parent_classes[1], parent_classes[0])
            kernel_factor = equivalence.factor
            if (
                equivalence.input_exchange_factor is not None
                and parent_classes[1] < parent_classes[0]
            ):
                parent_classes = (parent_classes[1], parent_classes[0])
                kernel_factor = _complex_mul(
                    kernel_factor,
                    equivalence.input_exchange_factor,
                )
            factor = _complex_mul(factor, kernel_factor)
            contract_ids = (equivalence.class_id,)

        payload = {
            "kind": "amplitude-closure",
            "root_contract": _root_contract(
                root,
                color_sector_aliases=color_sector_aliases,
            ),
            "parent_recurrence_classes": list(parent_classes),
        }
        direct_signature = _canonical_json(payload)
        opposite_signature = _canonical_json(_negated_root_payload(payload))
        class_index = class_by_signature.get(direct_signature)
        member_factor = factor
        if class_index is None and direct_signature != opposite_signature:
            class_index = class_by_signature.get(opposite_signature)
            if class_index is not None:
                member_factor = _complex_mul(member_factor, (-1.0, 0.0))
        if class_index is None:
            proof_digest = hashlib.sha256(direct_signature.encode("ascii")).hexdigest()
            class_index = len(pending)
            class_by_signature[direct_signature] = class_index
            pending.append(
                _PendingAmplitudeClass(
                    class_id=f"helicity-amplitude-sha256:{proof_digest}",
                    representative_root_id=root.id,
                    proof_digest=proof_digest,
                    transition_contract_ids=contract_ids,
                    members=[],
                )
            )
        pending[class_index].members.append(
            (root.id, selector_states, member_factor)
        )
    return pending, residual, diagnostics


def _semantic_amplitude_root_alias_signatures(
    dag: GenericDAG,
    model: Model,
    *,
    color_sector_aliases: _ColorSectorAliases,
) -> tuple[_SemanticAmplitudeRootSignature | None, ...]:
    """Return exact root signatures under one traversal-sector projection.

    This is deliberately a construction-only view of the same proof machinery
    used by helicity replay.  The projection changes only sector identifiers;
    traversal order, source labels, fermion exchange factors and every model
    kernel contract remain exact.  A ``None`` entry means that no proof was
    available and callers must fail closed rather than remove that root.
    """

    (
        source_by_bit,
        _source_contracts_by_id,
        _pending_classes,
        current_proofs,
        _residual_current_ids,
        _diagnostics,
    ) = _derive_current_replay_proofs(
        dag,
        model,
        color_sector_aliases=color_sector_aliases,
    )
    pending, residual, _root_diagnostics = _amplitude_classes(
        dag,
        model,
        current_proofs,
        source_by_bit,
        color_sector_aliases=color_sector_aliases,
    )
    signatures: list[_SemanticAmplitudeRootSignature | None] = [
        None
    ] * len(dag.amplitude_roots)
    for recurrence in pending:
        for root_id, selector_states, factor in recurrence.members:
            signatures[root_id] = _SemanticAmplitudeRootSignature(
                proof_digest=recurrence.proof_digest,
                selector_states=selector_states,
                factor=factor,
            )
    for root_id in residual:
        signatures[root_id] = None
    return tuple(signatures)


def _root_selector_states(
    root: AmplitudeRoot,
    dag: GenericDAG,
    source_by_bit: Mapping[int, tuple[int, int]],
) -> tuple[tuple[_SelectorState, ...], str | None]:
    selector_state, selector_error = _root_selector_state(
        root,
        dag,
        source_by_bit,
    )
    if selector_error is not None:
        return (), selector_error
    selector_states = [selector_state]
    if math.isclose(root.helicity_weight, 2.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
        flipped = tuple((label, -helicity) for label, helicity in selector_state)
        if flipped != selector_state:
            selector_states.append(flipped)
    elif not math.isclose(
        root.helicity_weight,
        1.0,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        return (
            (),
            f"unsupported helicity multiplicity {root.helicity_weight}",
        )
    return tuple(selector_states), None


def _root_selector_state(
    root: AmplitudeRoot,
    dag: GenericDAG,
    source_by_bit: Mapping[int, tuple[int, int]],
) -> tuple[_SelectorState, str | None]:
    ancestry = int(
        dag.currents[root.left_id].index.helicity_ancestry
        | dag.currents[root.right_id].index.helicity_ancestry
    )
    return _selector_state_for_ancestry(ancestry, source_by_bit)


def _selector_state_for_ancestry(
    ancestry: int,
    source_by_bit: Mapping[int, tuple[int, int]],
) -> tuple[_SelectorState, str | None]:
    states: dict[int, int] = {}
    remaining = int(ancestry)
    while remaining:
        least = remaining & -remaining
        bit = least.bit_length() - 1
        source = source_by_bit.get(bit)
        if source is None:
            return tuple(sorted(states.items())), f"unknown source ancestry bit {bit}"
        label, helicity = source
        previous = states.setdefault(label, helicity)
        if previous != helicity:
            return (
                tuple(sorted(states.items())),
                f"ancestry selects multiple helicities for external leg {label}",
            )
        remaining ^= least
    return tuple(sorted(states.items())), None


def _possible_helicity_states(
    dag: GenericDAG,
    model: Model,
) -> tuple[_SelectorState, ...]:
    labels: list[int] = []
    per_leg: list[tuple[int, ...]] = []
    for leg in dag.process.legs:
        if leg.outgoing_pdg is None:
            continue
        source_ir = model._source_ir(int(leg.outgoing_pdg))
        values = {
            int(
                (source_ir.crossing.apply(state) if leg.is_initial else state).helicity
            )
            for state in source_ir.states
        }
        labels.append(int(leg.label))
        per_leg.append(tuple(sorted(values)))
    return tuple(
        tuple(zip(labels, values, strict=True)) for values in product(*per_leg)
    )


def _current_contract(
    current: CurrentNode,
    *,
    include_spin_state: bool = True,
    color_sector_aliases: _ColorSectorAliases | None = None,
) -> dict[str, object]:
    index = current.index
    color_state = index.color_state.to_json_dict()
    if color_sector_aliases is not None:
        sector_id = int(color_state["sector_id"])
        color_state["sector_id"] = int(
            color_sector_aliases.get(sector_id, sector_id)
        )
    payload: dict[str, object] = {
        "particle_id": int(index.particle_id),
        "external_mask": int(index.external_mask),
        "external_labels": list(index.external_labels),
        "ordered_external_labels": list(index.ordered_external_labels),
        "chirality": int(index.chirality),
        "flavour_flow": list(index.flavour_flow),
        "quantum_number_flow": [list(item) for item in index.quantum_number_flow],
        "color_state": color_state,
        "momentum_mask": int(index.momentum_mask),
        "coupling_orders": [list(item) for item in index.coupling_orders],
        "auxiliary_kind": index.auxiliary_kind,
        "dimension": int(current.dimension),
        "is_source": bool(current.is_source),
    }
    if include_spin_state:
        payload["spin_state"] = _json_spin_state(index.spin_state)
    return payload


def _root_contract(
    root: AmplitudeRoot,
    *,
    color_sector_aliases: _ColorSectorAliases | None = None,
) -> dict[str, object]:
    color_sector_id = root.color_sector_id
    if color_sector_id is not None and color_sector_aliases is not None:
        color_sector_id = int(
            color_sector_aliases.get(int(color_sector_id), int(color_sector_id))
        )
    return {
        "kind": root.kind,
        "color_weight": list(root.color_weight),
        "color_sector_id": color_sector_id,
        "vertex_kind": root.vertex_kind,
        "vertex_particles": (
            None if root.vertex_particles is None else list(root.vertex_particles)
        ),
        "coupling": list(root.coupling),
        "contraction_ir": root.contraction_ir.to_json_dict(),
    }


def _runtime_coupling_identity(
    model: Model,
    interaction: InteractionNode,
) -> dict[str, object]:
    names = runtime_coupling_parameter_names(
        interaction.vertex_kind,
        interaction.vertex_particles,
        interaction.coupling,
        model=model,
    )
    return {
        "default": list(interaction.coupling),
        "runtime_parameters": [name for name in names],
    }


def _negated_signature_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    result = json.loads(_canonical_json(payload))
    terms = result.get("terms")
    if not isinstance(terms, list):
        return result
    for term in terms:
        if not isinstance(term, dict):
            continue
        coefficient = term.get("coefficient")
        if isinstance(coefficient, list) and len(coefficient) == 2:
            term["coefficient"] = [-float(coefficient[0]), -float(coefficient[1])]
    return result


def _negated_root_payload(payload: Mapping[str, object]) -> dict[str, object]:
    result = json.loads(_canonical_json(payload))
    root = result.get("root_contract")
    if not isinstance(root, dict):
        return result
    weight = root.get("color_weight")
    if isinstance(weight, list) and len(weight) == 2:
        root["color_weight"] = [-float(weight[0]), -float(weight[1])]
    return result


def _spin_state_shape(value: int | tuple[int, ...]) -> dict[str, object]:
    if isinstance(value, tuple):
        return {"kind": "tuple", "length": len(value)}
    return {"kind": "scalar"}


def _json_spin_state(value: int | tuple[int, ...]) -> object:
    return list(value) if isinstance(value, tuple) else value


def _complex_mul(left: _ComplexWeight, right: _ComplexWeight) -> _ComplexWeight:
    return (
        _canonical_zero(left[0] * right[0] - left[1] * right[1]),
        _canonical_zero(left[0] * right[1] + left[1] * right[0]),
    )


def _sum_weights(values: Sequence[_ComplexWeight]) -> _ComplexWeight:
    return (
        _canonical_zero(fsum(value[0] for value in values)),
        _canonical_zero(fsum(value[1] for value in values)),
    )


def _canonical_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _validate_digest(value: str, *, context: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")


__all__ = [
    "HELICITY_RECURRENCE_CONTRACT_VERSION",
    "HELICITY_RECURRENCE_PROOF_ALGORITHM",
    "RUNTIME_SELECTOR_PROVENANCE",
    "HelicityRecurrencePlan",
    "build_helicity_recurrence_plan",
]
