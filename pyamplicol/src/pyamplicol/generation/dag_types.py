# SPDX-License-Identifier: 0BSD
"""Frozen graph records shared by generation phases."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ..color.plan import (
    ColorTopologyReplayCertificate,
    GenericColorPlan,
    LCColorSector,
    LCColorTopologyReplayPlan,
)
from ..models._physics_ir import ContractionIR
from ..models.base import QuantumNumberFlow
from ..processes.ir import CanonicalProcessIR

if TYPE_CHECKING:
    from .helicity_materialization import HelicityMaterialization
    from .helicity_replay import HelicityRecurrencePlan

ColorAccuracy = Literal["lc", "nlc", "full"]

_LC_FIERZ_SINGLET_BASIS = "lc-fierz-singlet"
_LC_COLOR_IDENTITY_CLOSURE_PREFIX = "lc-color-identity-closure:"


def _lc_color_identity_closure_key(labels: Iterable[int]) -> str:
    normalized = tuple(sorted(set(int(label) for label in labels)))
    return _LC_COLOR_IDENTITY_CLOSURE_PREFIX + ",".join(
        str(label) for label in normalized
    )


def _lc_color_identity_closures(
    basis_keys: Iterable[str],
) -> tuple[frozenset[int], ...]:
    closures: set[frozenset[int]] = set()
    for key in basis_keys:
        if not key.startswith(_LC_COLOR_IDENTITY_CLOSURE_PREFIX):
            continue
        payload = key.removeprefix(_LC_COLOR_IDENTITY_CLOSURE_PREFIX)
        try:
            labels = frozenset(int(label) for label in payload.split(",") if label)
        except ValueError:
            return (frozenset(),)
        if not labels:
            return (frozenset(),)
        closures.add(labels)
    return tuple(sorted(closures, key=lambda labels: tuple(sorted(labels))))


def _lc_color_identity_closures_compatible(
    closures: Iterable[frozenset[int]],
    sector: LCColorSector,
) -> bool:
    groups = tuple(frozenset(group) for group in sector.coloured_label_groups)
    return all(
        bool(closure) and any(closure.issubset(group) for group in groups)
        for closure in closures
    )


@dataclass(frozen=True)
class ColorState:
    """Colour identity carried by one current.

    The LC implementation stores the warmup sector and the open-line/trace
    groups touched by the current.  NLC/full colour will replace ``basis_key``
    with Idenso basis components without changing the current-index contract.
    """

    accuracy: str
    sector_id: int = 0
    line_groups: tuple[int, ...] = ()
    basis_key: tuple[str, ...] = ()
    _key: tuple[object, ...] = field(init=False, repr=False, compare=False)
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        key = (
            self.accuracy,
            self.sector_id,
            self.line_groups,
            self.basis_key,
        )
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_hash", hash(key))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColorState):
            return NotImplemented
        return self._key == other._key

    def __hash__(self) -> int:
        return self._hash

    def to_json_dict(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "sector_id": self.sector_id,
            "line_groups": list(self.line_groups),
            "basis_key": list(self.basis_key),
        }


@dataclass(frozen=True)
class ColorFlow:
    state: ColorState
    weight: tuple[float, float] = (1.0, 0.0)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "state": self.state.to_json_dict(),
            "weight": list(self.weight),
        }


@dataclass(frozen=True)
class CurrentIndex:
    """Complete model-driven current identity.

    Equality of this dataclass is the only current deduplication rule.  It
    deliberately carries physics state, not a process-family label.
    """

    particle_id: int
    external_mask: int
    external_labels: tuple[int, ...]
    helicity_ancestry: int
    chirality: int
    spin_state: int | tuple[int, ...]
    flavour_flow: tuple[int, ...]
    quantum_number_flow: QuantumNumberFlow
    color_state: ColorState
    momentum_mask: int
    coupling_orders: tuple[tuple[str, int], ...] = ()
    auxiliary_kind: str | None = None
    ordered_external_labels: tuple[int, ...] = ()
    _key: tuple[object, ...] = field(init=False, repr=False, compare=False)
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        labels = tuple(sorted(self.external_labels))
        if labels != self.external_labels:
            object.__setattr__(self, "external_labels", labels)
        if not self.ordered_external_labels:
            object.__setattr__(self, "ordered_external_labels", labels)
        elif tuple(sorted(self.ordered_external_labels)) != labels:
            raise ValueError(
                "ordered_external_labels must contain the same labels as "
                "external_labels"
            )
        expected_mask = _labels_mask(self.external_labels)
        if self.external_mask != expected_mask:
            raise ValueError(
                "external_mask does not match external_labels: "
                f"{self.external_mask} != {expected_mask}"
            )
        if self.momentum_mask == 0:
            raise ValueError("momentum_mask must be nonzero for a current")
        if self.coupling_orders:
            object.__setattr__(
                self,
                "coupling_orders",
                tuple(
                    sorted(
                        (str(name).upper(), int(value))
                        for name, value in self.coupling_orders
                        if int(value) != 0
                    )
                ),
            )
        key = (
            self.particle_id,
            self.external_mask,
            self.external_labels,
            self.helicity_ancestry,
            self.chirality,
            self.spin_state,
            self.flavour_flow,
            self.quantum_number_flow,
            self.color_state,
            self.momentum_mask,
            self.coupling_orders,
            self.auxiliary_kind,
            self.ordered_external_labels,
        )
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_hash", hash(key))

    @classmethod
    def _from_trusted_values(
        cls,
        *,
        particle_id: int,
        external_mask: int,
        external_labels: tuple[int, ...],
        helicity_ancestry: int,
        chirality: int,
        spin_state: int | tuple[int, ...],
        flavour_flow: tuple[int, ...],
        quantum_number_flow: QuantumNumberFlow,
        color_state: ColorState,
        momentum_mask: int,
        coupling_orders: tuple[tuple[str, int], ...] = (),
        auxiliary_kind: str | None = None,
        ordered_external_labels: tuple[int, ...] = (),
    ) -> CurrentIndex:
        """Build an already canonical compiler-owned index.

        Public construction and artifact decoding retain the normal validation
        path. The eager compiler uses this only after its mask, ordering, and
        coupling-order planners have produced canonical values.
        """

        row = object.__new__(cls)
        object.__setattr__(row, "particle_id", particle_id)
        object.__setattr__(row, "external_mask", external_mask)
        object.__setattr__(row, "external_labels", external_labels)
        object.__setattr__(row, "helicity_ancestry", helicity_ancestry)
        object.__setattr__(row, "chirality", chirality)
        object.__setattr__(row, "spin_state", spin_state)
        object.__setattr__(row, "flavour_flow", flavour_flow)
        object.__setattr__(row, "quantum_number_flow", quantum_number_flow)
        object.__setattr__(row, "color_state", color_state)
        object.__setattr__(row, "momentum_mask", momentum_mask)
        object.__setattr__(row, "coupling_orders", coupling_orders)
        object.__setattr__(row, "auxiliary_kind", auxiliary_kind)
        object.__setattr__(
            row,
            "ordered_external_labels",
            ordered_external_labels or external_labels,
        )
        key = (
            row.particle_id,
            row.external_mask,
            row.external_labels,
            row.helicity_ancestry,
            row.chirality,
            row.spin_state,
            row.flavour_flow,
            row.quantum_number_flow,
            row.color_state,
            row.momentum_mask,
            row.coupling_orders,
            row.auxiliary_kind,
            row.ordered_external_labels,
        )
        object.__setattr__(row, "_key", key)
        object.__setattr__(row, "_hash", hash(key))
        return row

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CurrentIndex):
            return NotImplemented
        return self._key == other._key

    def __hash__(self) -> int:
        return self._hash

    def overlaps(self, other: CurrentIndex) -> bool:
        return bool(self.external_mask & other.external_mask)

    def to_json_dict(self) -> dict[str, object]:
        spin_state: object
        if isinstance(self.spin_state, tuple):
            spin_state = list(self.spin_state)
        else:
            spin_state = self.spin_state
        return {
            "particle_id": self.particle_id,
            "external_mask": self.external_mask,
            "external_labels": list(self.external_labels),
            "ordered_external_labels": list(self.ordered_external_labels),
            "helicity_ancestry": self.helicity_ancestry,
            "chirality": self.chirality,
            "spin_state": spin_state,
            "flavour_flow": list(self.flavour_flow),
            "quantum_number_flow": [
                [name, expression] for name, expression in self.quantum_number_flow
            ],
            "color_state": self.color_state.to_json_dict(),
            "momentum_mask": self.momentum_mask,
            "coupling_orders": [[name, value] for name, value in self.coupling_orders],
            "auxiliary_kind": self.auxiliary_kind,
        }


@dataclass(frozen=True)
class CurrentNode:
    id: int
    index: CurrentIndex
    dimension: int
    is_source: bool
    source_leg_label: int | None = None
    source_helicity: int | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "index": self.index.to_json_dict(),
            "dimension": self.dimension,
            "is_source": self.is_source,
            "source_leg_label": self.source_leg_label,
            "source_helicity": self.source_helicity,
        }


@dataclass(frozen=True)
class InteractionNode:
    id: int
    vertex_kind: int
    vertex_particles: tuple[int, int, int]
    left_id: int
    right_id: int
    result_id: int
    coupling: tuple[float, float]
    color_weight: tuple[float, float]
    lowering_backend: str
    full_tensor_network_ready: bool
    evaluation_group_id: int | None = None
    evaluation_factor: tuple[float, float] = (1.0, 0.0)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "vertex_kind": self.vertex_kind,
            "vertex_particles": list(self.vertex_particles),
            "left_id": self.left_id,
            "right_id": self.right_id,
            "result_id": self.result_id,
            "coupling": list(self.coupling),
            "color_weight": list(self.color_weight),
            "lowering_backend": self.lowering_backend,
            "full_tensor_network_ready": self.full_tensor_network_ready,
            "evaluation_group_id": self.evaluation_group_id,
            "evaluation_factor": list(self.evaluation_factor),
        }


@dataclass(frozen=True)
class AmplitudeRoot:
    id: int
    kind: str
    left_id: int
    right_id: int
    color_weight: tuple[float, float]
    contraction_ir: ContractionIR
    color_sector_id: int | None = None
    vertex_kind: int | None = None
    vertex_particles: tuple[int, int, int] | None = None
    coupling: tuple[float, float] = (1.0, 0.0)
    helicity_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.contraction_ir, ContractionIR):
            raise TypeError("amplitude root requires a frozen ContractionIR")

    @property
    def contraction(self) -> str:
        """Display/wire projection retained for schema compatibility."""

        return self.contraction_ir.name

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "color_weight": list(self.color_weight),
            "color_sector_id": self.color_sector_id,
            "vertex_kind": self.vertex_kind,
            "vertex_particles": (
                list(self.vertex_particles)
                if self.vertex_particles is not None
                else None
            ),
            "coupling": list(self.coupling),
            "contraction": self.contraction,
            "contraction_ir": self.contraction_ir.to_json_dict(),
            "helicity_weight": self.helicity_weight,
        }


@dataclass(frozen=True)
class GenericDAG:
    process: CanonicalProcessIR
    color_plan: GenericColorPlan
    currents: tuple[CurrentNode, ...]
    sources: tuple[int, ...]
    interactions: tuple[InteractionNode, ...]
    amplitude_roots: tuple[AmplitudeRoot, ...]
    truncated: bool = False
    helicity_coverage: str = "complete"
    color_coverage: str = "complete"
    selected_source_helicities: tuple[tuple[int, int], ...] = ()
    selected_color_sector_ids: tuple[int, ...] = ()
    lc_topology_replay: LCColorTopologyReplayPlan | None = None
    color_topology_replay: ColorTopologyReplayCertificate | None = None
    helicity_recurrence: HelicityRecurrencePlan | None = None
    helicity_materialization: HelicityMaterialization | None = None

    def __post_init__(self) -> None:
        if self.helicity_coverage not in {"complete", "selected"}:
            raise ValueError("DAG helicity coverage must be complete or selected")
        if self.color_coverage not in {"complete", "selected"}:
            raise ValueError("DAG color coverage must be complete or selected")
        selected_color_sector_ids = tuple(
            sorted({int(sector_id) for sector_id in self.selected_color_sector_ids})
        )
        object.__setattr__(
            self,
            "selected_color_sector_ids",
            selected_color_sector_ids,
        )
        known_sector_ids = {int(sector.id) for sector in self.color_plan.sectors}
        if not set(selected_color_sector_ids) <= known_sector_ids:
            raise ValueError(
                "selected color-sector provenance references an absent sector"
            )
        if (
            self.lc_topology_replay is not None
            and self.color_topology_replay is not None
        ):
            raise ValueError("DAG cannot carry both LC and generic topology replay")
        if self.lc_topology_replay is not None:
            if self.process.color_accuracy != "lc":
                raise ValueError("LC topology replay requires LC color accuracy")
            if self.color_coverage != "complete":
                raise ValueError("LC topology replay requires complete color coverage")
            if selected_color_sector_ids:
                raise ValueError(
                    "LC topology replay cannot carry generation-specialized color "
                    "sectors"
                )
            if set(self.lc_topology_replay.physical_sector_ids) != {
                int(sector.id) for sector in self.color_plan.sectors
            }:
                raise ValueError(
                    "LC topology replay physical sectors do not match the color plan"
                )
        if self.color_topology_replay is not None:
            if self.process.color_accuracy not in {"nlc", "full"}:
                raise ValueError(
                    "generic topology replay requires NLC or full color accuracy"
                )
            if self.color_topology_replay.color_accuracy != self.process.color_accuracy:
                raise ValueError(
                    "generic topology replay color accuracy does not match the DAG"
                )
            if self.color_coverage != "complete":
                raise ValueError(
                    "generic topology replay requires complete color coverage"
                )
            if selected_color_sector_ids:
                raise ValueError(
                    "generic topology replay cannot carry generation-specialized "
                    "color sectors"
                )
            if set(self.color_topology_replay.physical_sector_ids) != {
                int(sector.id) for sector in self.color_plan.sectors
            }:
                raise ValueError(
                    "generic topology replay physical sectors do not match the "
                    "color plan"
                )
        if self.helicity_recurrence is not None and (
            self.helicity_coverage != "complete"
            or bool(self.selected_source_helicities)
        ):
            raise ValueError(
                "helicity recurrence metadata requires complete helicity coverage"
            )
        if self.helicity_materialization is not None:
            if self.helicity_recurrence is None:
                raise ValueError("helicity materialization requires a recurrence proof")
            if self.helicity_materialization.dag is not self:
                # The materialization object owns the same immutable payload but
                # cannot hold the final self-reference after dataclass creation.
                materialized = self.helicity_materialization.dag
                if (
                    materialized.currents != self.currents
                    or materialized.interactions != self.interactions
                    or materialized.amplitude_roots != self.amplitude_roots
                ):
                    raise ValueError(
                        "helicity materialization does not describe this DAG"
                    )

    @property
    def has_amplitudes(self) -> bool:
        return bool(self.amplitude_roots)

    @property
    def required_vertex_kinds(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {interaction.vertex_kind for interaction in self.interactions}
                | {
                    root.vertex_kind
                    for root in self.amplitude_roots
                    if root.vertex_kind is not None
                }
            )
        )

    @property
    def interaction_evaluation_count(self) -> int:
        return len(
            {
                (
                    "group",
                    interaction.evaluation_group_id,
                )
                if interaction.evaluation_group_id is not None
                else ("interaction", interaction.id)
                for interaction in self.interactions
                if interaction.evaluation_factor != (0.0, 0.0)
            }
        )

    @property
    def interaction_fanout_histogram(self) -> tuple[tuple[int, int], ...]:
        group_sizes: dict[tuple[str, int], int] = {}
        for interaction in self.interactions:
            if interaction.evaluation_factor == (0.0, 0.0):
                continue
            group = (
                ("group", interaction.evaluation_group_id)
                if interaction.evaluation_group_id is not None
                else ("interaction", interaction.id)
            )
            group_sizes[group] = group_sizes.get(group, 0) + 1
        histogram: dict[int, int] = {}
        for size in group_sizes.values():
            histogram[size] = histogram.get(size, 0) + 1
        return tuple(sorted(histogram.items()))

    def currents_by_external_labels(
        self,
        labels: Iterable[int],
    ) -> tuple[CurrentNode, ...]:
        wanted = tuple(sorted(labels))
        return tuple(
            current
            for current in self.currents
            if current.index.external_labels == wanted
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "process": self.process.to_json_dict(),
            "color_plan": self.color_plan.to_json_dict(),
            "current_count": len(self.currents),
            "source_count": len(self.sources),
            "helicity_coverage": self.helicity_coverage,
            "color_coverage": self.color_coverage,
            "selected_source_helicities": {
                str(label): helicity
                for label, helicity in self.selected_source_helicities
            },
            **(
                {}
                if not self.selected_color_sector_ids
                else {"selected_color_sector_ids": list(self.selected_color_sector_ids)}
            ),
            **(
                {}
                if self.lc_topology_replay is None
                else {"lc_topology_replay": self.lc_topology_replay.to_json_dict()}
            ),
            **(
                {}
                if self.color_topology_replay is None
                else {
                    "color_topology_replay": (self.color_topology_replay.to_json_dict())
                }
            ),
            **(
                {}
                if self.helicity_recurrence is None
                else {
                    "helicity_recurrence": (
                        {
                            **self.helicity_recurrence.to_json_dict(),
                            **(
                                {}
                                if self.helicity_materialization is None
                                else {
                                    "materialization": (
                                        self.helicity_materialization.to_runtime_manifest()
                                    )
                                }
                            ),
                        }
                    )
                }
            ),
            "interaction_count": len(self.interactions),
            "interaction_evaluation_count": self.interaction_evaluation_count,
            "interaction_fanout_histogram": [
                [fanout, count] for fanout, count in self.interaction_fanout_histogram
            ],
            "amplitude_root_count": len(self.amplitude_roots),
            "truncated": self.truncated,
            "required_vertex_kinds": list(self.required_vertex_kinds),
            "currents": [current.to_json_dict() for current in self.currents],
            "sources": list(self.sources),
            "interactions": [
                interaction.to_json_dict() for interaction in self.interactions
            ],
            "amplitude_roots": [root.to_json_dict() for root in self.amplitude_roots],
        }


def _labels_mask(labels: Iterable[int]) -> int:
    mask = 0
    for label in labels:
        mask |= 1 << (label - 1)
    return mask
