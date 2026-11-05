# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..color.plan import GenericColorPlan
from ..models.base import Model
from ..processes.ir import CanonicalProcessIR
from .dag_algorithms import contributing_color_sector_ids
from .dag_compiler import GenericDAGCompiler
from .dag_types import (
    AmplitudeRoot,
    CurrentNode,
    GenericDAG,
    InteractionNode,
)

GenericCurrentNode = CurrentNode
GenericInteractionNode = InteractionNode
GenericClosureNode = AmplitudeRoot


@dataclass(frozen=True)
class GenericColorSectorUseSummary:
    color_sector: int
    current_count: int
    interaction_count: int
    closure_count: int
    required_vertex_kind_counts: tuple[tuple[int, int], ...]
    ready_vertex_kinds: tuple[int, ...]
    pending_vertex_kinds: tuple[int, ...]
    unimplemented_vertex_kinds: tuple[int, ...]

    @property
    def full_tensor_network_ready(self) -> bool:
        return not self.pending_vertex_kinds and not self.unimplemented_vertex_kinds

    def to_json_dict(self) -> dict[str, object]:
        return {
            "color_sector": self.color_sector,
            "current_count": self.current_count,
            "interaction_count": self.interaction_count,
            "closure_count": self.closure_count,
            "full_tensor_network_ready": self.full_tensor_network_ready,
            "required_vertex_kind_counts": [
                [kind, count] for kind, count in self.required_vertex_kind_counts
            ],
            "ready_vertex_kinds": list(self.ready_vertex_kinds),
            "pending_vertex_kinds": list(self.pending_vertex_kinds),
            "unimplemented_vertex_kinds": list(self.unimplemented_vertex_kinds),
        }


@dataclass(frozen=True)
class GenericCurrentStage:
    stage_index: int
    subset_size: int
    current_ids: tuple[int, ...]
    interaction_ids: tuple[int, ...]
    required_vertex_kind_counts: tuple[tuple[int, int], ...]
    ready_vertex_kinds: tuple[int, ...]
    pending_vertex_kinds: tuple[int, ...]
    unimplemented_vertex_kinds: tuple[int, ...]
    color_sector_summaries: tuple[GenericColorSectorUseSummary, ...]

    @property
    def full_tensor_network_ready(self) -> bool:
        return not self.pending_vertex_kinds and not self.unimplemented_vertex_kinds

    def to_json_dict(self) -> dict[str, object]:
        return {
            "stage_index": self.stage_index,
            "subset_size": self.subset_size,
            "current_ids": list(self.current_ids),
            "interaction_ids": list(self.interaction_ids),
            "interaction_count": len(self.interaction_ids),
            "full_tensor_network_ready": self.full_tensor_network_ready,
            "required_vertex_kind_counts": [
                [kind, count] for kind, count in self.required_vertex_kind_counts
            ],
            "ready_vertex_kinds": list(self.ready_vertex_kinds),
            "pending_vertex_kinds": list(self.pending_vertex_kinds),
            "unimplemented_vertex_kinds": list(self.unimplemented_vertex_kinds),
            "color_sector_summaries": [
                summary.to_json_dict() for summary in self.color_sector_summaries
            ],
        }


@dataclass(frozen=True)
class GenericAmplitudeStage:
    closure_ids: tuple[int, ...]
    required_vertex_kind_counts: tuple[tuple[int, int], ...]
    ready_vertex_kinds: tuple[int, ...]
    pending_vertex_kinds: tuple[int, ...]
    unimplemented_vertex_kinds: tuple[int, ...]
    color_sector_summaries: tuple[GenericColorSectorUseSummary, ...]

    @property
    def full_tensor_network_ready(self) -> bool:
        return (
            bool(self.closure_ids)
            and not self.pending_vertex_kinds
            and not self.unimplemented_vertex_kinds
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "closure_ids": list(self.closure_ids),
            "closure_count": len(self.closure_ids),
            "full_tensor_network_ready": self.full_tensor_network_ready,
            "required_vertex_kind_counts": [
                [kind, count] for kind, count in self.required_vertex_kind_counts
            ],
            "ready_vertex_kinds": list(self.ready_vertex_kinds),
            "pending_vertex_kinds": list(self.pending_vertex_kinds),
            "unimplemented_vertex_kinds": list(self.unimplemented_vertex_kinds),
            "color_sector_summaries": [
                summary.to_json_dict() for summary in self.color_sector_summaries
            ],
        }


@dataclass(frozen=True)
class GenericStagePlan:
    process: CanonicalProcessIR
    current_stages: tuple[GenericCurrentStage, ...]
    amplitude_stage: GenericAmplitudeStage

    @property
    def stage_count(self) -> int:
        return len(self.current_stages) + 1

    @property
    def full_tensor_network_ready(self) -> bool:
        return self.amplitude_stage.full_tensor_network_ready and all(
            stage.full_tensor_network_ready for stage in self.current_stages
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "process": self.process.to_json_dict(),
            "stage_count": self.stage_count,
            "full_tensor_network_ready": self.full_tensor_network_ready,
            "current_stages": [stage.to_json_dict() for stage in self.current_stages],
            "amplitude_stage": self.amplitude_stage.to_json_dict(),
        }


@dataclass(frozen=True)
class GenericCurrentPlan:
    """Compatibility view over the production `GenericDAG`.

    This module intentionally no longer owns an independent recursion
    implementation.  All current identities come from `generic_dag.CurrentIndex`,
    so process-support diagnostics and tests inspect the same model-driven DAG
    compiler used by schema-v3 artifacts.
    """

    dag: GenericDAG
    model: Model

    @property
    def process(self) -> CanonicalProcessIR:
        return self.dag.process

    @property
    def color_plan(self) -> GenericColorPlan:
        return self.dag.color_plan

    @property
    def currents(self) -> tuple[CurrentNode, ...]:
        return self.dag.currents

    @property
    def sources(self) -> tuple[int, ...]:
        return self.dag.sources

    @property
    def interactions(self) -> tuple[InteractionNode, ...]:
        return self.dag.interactions

    @property
    def closures(self) -> tuple[AmplitudeRoot, ...]:
        return self.dag.amplitude_roots

    @property
    def truncated(self) -> bool:
        return self.dag.truncated

    @property
    def color_sectors(self) -> tuple[int, ...]:
        root_sectors = contributing_color_sector_ids(self.dag)
        if root_sectors:
            return root_sectors
        return tuple(
            sorted({current.index.color_state.sector_id for current in self.currents})
        )

    @property
    def has_closure(self) -> bool:
        return bool(self.closures)

    @property
    def required_vertex_kind_counts(self) -> tuple[tuple[int, int], ...]:
        return _vertex_kind_counts(self.interactions, self.closures)

    @property
    def ready_vertex_kinds(self) -> tuple[int, ...]:
        return _vertex_kind_status(
            (kind for kind, _ in self.required_vertex_kind_counts),
            self.model,
        )[0]

    @property
    def pending_vertex_kinds(self) -> tuple[int, ...]:
        return _vertex_kind_status(
            (kind for kind, _ in self.required_vertex_kind_counts),
            self.model,
        )[1]

    @property
    def unimplemented_vertex_kinds(self) -> tuple[int, ...]:
        return _vertex_kind_status(
            (kind for kind, _ in self.required_vertex_kind_counts),
            self.model,
        )[2]

    @property
    def full_tensor_network_ready(self) -> bool:
        return (
            self.has_closure
            and not self.truncated
            and not self.pending_vertex_kinds
            and not self.unimplemented_vertex_kinds
        )

    def build_stage_plan(self) -> GenericStagePlan:
        return build_generic_stage_plan(self)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "process": self.process.to_json_dict(),
            "color_plan": self.color_plan.to_json_dict(),
            "current_count": len(self.currents),
            "source_count": len(self.sources),
            "interaction_count": len(self.interactions),
            "closure_count": len(self.closures),
            "truncated": self.truncated,
            "has_closure": self.has_closure,
            "full_tensor_network_ready": self.full_tensor_network_ready,
            "required_vertex_kind_counts": [
                [kind, count] for kind, count in self.required_vertex_kind_counts
            ],
            "ready_vertex_kinds": list(self.ready_vertex_kinds),
            "pending_vertex_kinds": list(self.pending_vertex_kinds),
            "unimplemented_vertex_kinds": list(self.unimplemented_vertex_kinds),
            "currents": [current.to_json_dict() for current in self.currents],
            "sources": list(self.sources),
            "interactions": [
                interaction.to_json_dict() for interaction in self.interactions
            ],
            "closures": [closure.to_json_dict() for closure in self.closures],
            "color_sector_summaries": [
                summary.to_json_dict()
                for summary in _sector_summaries(
                    self.currents,
                    self.interactions,
                    self.closures,
                    self.model,
                )
            ],
        }


def build_generic_current_plan(
    process: CanonicalProcessIR | GenericDAG,
    *,
    model: Model,
    max_currents: int | None = None,
    max_color_sectors: int | None = None,
    reference_color_order: tuple[int, ...] | None = None,
    selected_color_sector_ids: Iterable[int] | None = None,
    max_coupling_orders: Mapping[str, int] | None = None,
    max_lc_current_line_groups: int | None = None,
    max_quark_pairs: int | None = None,
    closure_side_mask_pruning: bool = True,
    color_order_mask_pruning: bool = True,
    species_reachability_pruning: bool = True,
    ignored_particle_ids: Iterable[int] | None = None,
    ignored_vertex_kinds: Iterable[int] | None = None,
) -> GenericCurrentPlan:
    dag = (
        process
        if isinstance(process, GenericDAG)
        else GenericDAGCompiler(
            model=model,
            max_currents=max_currents,
            max_color_sectors=max_color_sectors,
            reference_color_order=reference_color_order,
            selected_color_sector_ids=selected_color_sector_ids,
            max_coupling_orders=max_coupling_orders,
            max_lc_current_line_groups=max_lc_current_line_groups,
            max_quark_pairs=max_quark_pairs,
            closure_side_mask_pruning=closure_side_mask_pruning,
            color_order_mask_pruning=color_order_mask_pruning,
            species_reachability_pruning=species_reachability_pruning,
            ignored_particle_ids=ignored_particle_ids,
            ignored_vertex_kinds=ignored_vertex_kinds,
        ).compile(process)
    )
    return GenericCurrentPlan(dag=dag, model=model)


def build_generic_stage_plan(
    plan: GenericCurrentPlan,
) -> GenericStagePlan:
    current_plan = plan
    stages: list[GenericCurrentStage] = []
    interactions_by_subset: dict[int, list[InteractionNode]] = {}
    for interaction in current_plan.interactions:
        result = current_plan.currents[interaction.result_id]
        subset_size = len(result.index.external_labels)
        interactions_by_subset.setdefault(subset_size, []).append(interaction)

    for stage_index, subset_size in enumerate(sorted(interactions_by_subset)):
        interactions = tuple(interactions_by_subset[subset_size])
        current_ids = tuple(
            sorted({interaction.result_id for interaction in interactions})
        )
        vertex_counts = _vertex_kind_counts(interactions, ())
        ready, pending, unimplemented = _vertex_kind_status(
            (kind for kind, _ in vertex_counts),
            current_plan.model,
        )
        stages.append(
            GenericCurrentStage(
                stage_index=stage_index,
                subset_size=subset_size,
                current_ids=current_ids,
                interaction_ids=tuple(interaction.id for interaction in interactions),
                required_vertex_kind_counts=vertex_counts,
                ready_vertex_kinds=ready,
                pending_vertex_kinds=pending,
                unimplemented_vertex_kinds=unimplemented,
                color_sector_summaries=_sector_summaries(
                    current_plan.currents,
                    interactions,
                    (),
                    current_plan.model,
                ),
            )
        )

    closure_counts = _vertex_kind_counts((), current_plan.closures)
    ready, pending, unimplemented = _vertex_kind_status(
        (kind for kind, _ in closure_counts),
        current_plan.model,
    )
    amplitude_stage = GenericAmplitudeStage(
        closure_ids=tuple(closure.id for closure in current_plan.closures),
        required_vertex_kind_counts=closure_counts,
        ready_vertex_kinds=ready,
        pending_vertex_kinds=pending,
        unimplemented_vertex_kinds=unimplemented,
        color_sector_summaries=_sector_summaries(
            current_plan.currents,
            (),
            current_plan.closures,
            current_plan.model,
        ),
    )
    return GenericStagePlan(
        process=current_plan.process,
        current_stages=tuple(stages),
        amplitude_stage=amplitude_stage,
    )


def _vertex_kind_counts(
    interactions: Iterable[InteractionNode],
    closures: Iterable[AmplitudeRoot],
) -> tuple[tuple[int, int], ...]:
    counts: Counter[int] = Counter(
        interaction.vertex_kind for interaction in interactions
    )
    counts.update(
        int(closure.vertex_kind)
        for closure in closures
        if closure.vertex_kind is not None
    )
    return tuple(sorted(counts.items()))


def _vertex_kind_status(
    kinds: Iterable[int],
    model: Model,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    ready: set[int] = set()
    pending: set[int] = set()
    unimplemented: set[int] = set()
    for kind in kinds:
        rule = model.vertex_lowering_rule(kind)
        if rule.backend == "unimplemented":
            unimplemented.add(kind)
        elif rule.full_tensor_network_ready:
            ready.add(kind)
        else:
            pending.add(kind)
    return tuple(sorted(ready)), tuple(sorted(pending)), tuple(sorted(unimplemented))


def _sector_summaries(
    currents: tuple[CurrentNode, ...],
    interactions: Iterable[InteractionNode],
    closures: Iterable[AmplitudeRoot],
    model: Model,
) -> tuple[GenericColorSectorUseSummary, ...]:
    current_ids_by_sector: dict[int, set[int]] = {}
    interactions_by_sector: dict[int, dict[int, InteractionNode]] = {}
    closures_by_sector: dict[int, list[AmplitudeRoot]] = {}
    interaction_tuple = tuple(interactions)
    closure_tuple = tuple(closures)
    interactions_by_result: dict[int, list[InteractionNode]] = {}
    for interaction in interaction_tuple:
        interactions_by_result.setdefault(interaction.result_id, []).append(interaction)
    if closure_tuple:
        for closure in closure_tuple:
            sector = _closure_color_sector(currents, closure)
            closures_by_sector.setdefault(sector, []).append(closure)
            seen_currents = current_ids_by_sector.setdefault(sector, set())
            sector_interactions = interactions_by_sector.setdefault(sector, {})
            stack = [closure.left_id, closure.right_id]
            while stack:
                current_id = stack.pop()
                if current_id in seen_currents:
                    continue
                seen_currents.add(current_id)
                for interaction in interactions_by_result.get(current_id, ()):
                    sector_interactions.setdefault(interaction.id, interaction)
                    stack.extend((interaction.left_id, interaction.right_id))
    else:
        for current in currents:
            current_ids_by_sector.setdefault(
                current.index.color_state.sector_id,
                set(),
            ).add(current.id)
        for interaction in interaction_tuple:
            sector = currents[interaction.result_id].index.color_state.sector_id
            interactions_by_sector.setdefault(sector, {})[interaction.id] = interaction
            current_ids_by_sector.setdefault(sector, set()).update(
                (interaction.left_id, interaction.right_id, interaction.result_id)
            )
    summaries = []
    for sector in sorted(
        set(interactions_by_sector)
        | set(closures_by_sector)
        | set(current_ids_by_sector)
    ):
        sector_interaction_nodes = tuple(
            interactions_by_sector.get(sector, {}).values()
        )
        sector_closures = tuple(closures_by_sector.get(sector, ()))
        vertex_counts = _vertex_kind_counts(
            sector_interaction_nodes,
            sector_closures,
        )
        ready, pending, unimplemented = _vertex_kind_status(
            (kind for kind, _ in vertex_counts),
            model,
        )
        summaries.append(
            GenericColorSectorUseSummary(
                color_sector=sector,
                current_count=len(current_ids_by_sector.get(sector, ())),
                interaction_count=len(sector_interaction_nodes),
                closure_count=len(sector_closures),
                required_vertex_kind_counts=vertex_counts,
                ready_vertex_kinds=ready,
                pending_vertex_kinds=pending,
                unimplemented_vertex_kinds=unimplemented,
            )
        )
    return tuple(summaries)


def _closure_color_sector(
    currents: tuple[CurrentNode, ...],
    closure: AmplitudeRoot,
) -> int:
    if closure.color_sector_id is not None:
        return int(closure.color_sector_id)
    return int(currents[closure.left_id].index.color_state.sector_id)


__all__ = [
    "GenericAmplitudeStage",
    "GenericClosureNode",
    "GenericColorSectorUseSummary",
    "GenericCurrentNode",
    "GenericCurrentPlan",
    "GenericCurrentStage",
    "GenericInteractionNode",
    "GenericStagePlan",
    "build_generic_current_plan",
    "build_generic_stage_plan",
]
