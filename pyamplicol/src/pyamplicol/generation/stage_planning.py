# SPDX-License-Identifier: 0BSD
"""Stage blueprint assembly and fanout-aware output scheduling."""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from ..models.base import Model
from .contracts import StageCompilationInput
from .dag_types import GenericDAG
from .helicity_materialization import _compiled_representative_dependencies
from .stage_expressions import (
    _compile_amplitude_stage_blueprint,
    _compile_current_stage_blueprint,
)
from .stage_parameters import (
    _current_slots_by_id,
    _dict,
    _expression_previews,
    _list,
    _logical_model_parameter_symbols,
    _manifest_model,
    _momentum_slots_by_id,
    _parameter_builder,
    _value_slots_by_id,
)
from .stage_settings import _stage_symbolica_settings
from .stage_types import (
    GenericCompiledStageBlueprint,
    GenericStageCompilerBlueprint,
    GenericStageOutputSlot,
    StageBlueprintConsumer,
    StageBlueprintProgress,
    _RuntimeParameterizedModel,
)


def build_generic_stage_compiler_blueprint(
    manifest: StageCompilationInput | GenericDAG,
    *,
    model: Model | None = None,
    selected_color_sector_ids: set[int] | None = None,
    enable_lc_sector_runtime_selector: bool | None = None,
    runtime_schema: Mapping[str, object] | None = None,
    stage_local_parameter_layout: bool = True,
    progress_callback: StageBlueprintProgress | None = None,
    stage_consumer: StageBlueprintConsumer | None = None,
    release_consumed_expressions: bool = False,
) -> GenericStageCompilerBlueprint:
    """Build evaluator-ready symbolic stages from a neutral runtime schema.

    This is intentionally separate from the legacy shared-current table path.
    It consumes the process-generic current DAG and runtime schema, asks the
    model for local vertex and propagator component expressions, and records
    stage output slots in terms of stable value-slot identifiers.
    """

    dag = manifest.dag if isinstance(manifest, StageCompilationInput) else manifest
    if (
        isinstance(manifest, StageCompilationInput)
        and model is not None
        and model is not manifest.model
    ):
        raise ValueError("stage model override conflicts with compilation input")
    if not stage_local_parameter_layout:
        raise ValueError("stage-local parameter layout is mandatory")
    if selected_color_sector_ids is not None:
        raise ValueError(
            "selected color sectors must already be encoded in the runtime schema"
        )
    if enable_lc_sector_runtime_selector is not None:
        raise ValueError(
            "the LC selector policy must already be encoded in the runtime schema"
        )
    if runtime_schema is not None:
        schema = _dict(runtime_schema)
    elif isinstance(manifest, StageCompilationInput):
        schema = _dict(manifest.runtime_schema.to_mapping())
    else:
        raise ValueError(
            "GenericDAG stage compilation requires an explicit runtime schema"
        )
    selected_model = model or _manifest_model(manifest)
    parameter_layout = _dict(schema["parameter_layout"])
    global_value_component_count = int(parameter_layout["value_component_count"])
    global_momentum_parameter_count = int(parameter_layout["momentum_parameter_count"])
    global_model_parameter_count = int(parameter_layout.get("model_parameter_count", 0))
    global_parameter_count = (
        global_value_component_count
        + global_momentum_parameter_count
        + global_model_parameter_count
    )
    if stage_local_parameter_layout:
        # Every compiled stage constructs its own compact symbols below. Avoid
        # materializing the large, otherwise-unused global Symbolica input set.
        parameter_symbols: tuple[Any, ...] = ()
        value_symbols: tuple[Any, ...] = ()
        momentum_symbols: tuple[Any, ...] = ()
        model_parameter_symbols: tuple[Any, ...] = ()
        global_real_valued_inputs = tuple(
            range(global_value_component_count, global_parameter_count)
        )
    else:
        builder = _parameter_builder(schema)
        parameter_symbols = tuple(builder.parameter_symbols())
        value_symbols = parameter_symbols[:global_value_component_count]
        momentum_start = global_value_component_count
        momentum_stop = momentum_start + global_momentum_parameter_count
        momentum_symbols = parameter_symbols[momentum_start:momentum_stop]
        model_parameter_symbols = parameter_symbols[momentum_stop:]
        global_real_valued_inputs = tuple(
            int(index) for index in builder.real_valued_inputs
        )
    model_parameter_records = tuple(
        _dict(item) for item in _list(schema.get("model_parameters", []))
    )
    model_parameter_symbols_by_name = (
        {}
        if stage_local_parameter_layout
        else _logical_model_parameter_symbols(
            model_parameter_records,
            {
                str(record["name"]): model_parameter_symbols[
                    int(record["parameter_index"])
                ]
                for record in model_parameter_records
            },
        )
    )
    expression_model = (
        selected_model.with_runtime_parameters(model_parameter_symbols_by_name)
        if hasattr(selected_model, "with_runtime_parameters")
        else _RuntimeParameterizedModel(
            selected_model,
            model_parameter_symbols_by_name,
        )
    )
    value_slots = _value_slots_by_id(schema)
    current_slots = _current_slots_by_id(schema)
    momentum_slots = _momentum_slots_by_id(schema)
    stage_records = tuple(_dict(stage) for stage in _list(schema["stages"]))
    stage_total = len(stage_records) + 1
    compiled_stages: list[GenericCompiledStageBlueprint] = []
    for stage_index, stage in enumerate(stage_records, start=1):
        if progress_callback is not None:
            progress_callback("current stage", stage_index, stage_total)
        compiled_stage = _compile_current_stage_blueprint(
            dag,
            expression_model,
            stage,
            value_slots=value_slots,
            current_slots=current_slots,
            momentum_slots=momentum_slots,
            global_value_component_count=global_value_component_count,
            global_momentum_parameter_count=global_momentum_parameter_count,
            model_parameter_records=model_parameter_records,
            global_parameter_symbols=parameter_symbols,
            global_value_symbols=value_symbols,
            global_momentum_symbols=momentum_symbols,
            global_model_parameter_symbols=model_parameter_symbols_by_name,
            global_real_valued_inputs=global_real_valued_inputs,
            stage_local_parameter_layout=stage_local_parameter_layout,
        )
        compiled_stage = _stage_with_selector_domain_memberships(
            compiled_stage,
            dag,
        )
        if stage_consumer is not None:
            stage_consumer(compiled_stage, stage_index - 1, len(stage_records))
        if release_consumed_expressions and stage_consumer is not None:
            compiled_stage = replace(
                compiled_stage,
                parameter_symbols=(),
                output_expressions=(),
                symbolica_functions=(),
            )
        compiled_stages.append(compiled_stage)
    stages = tuple(compiled_stages)
    if progress_callback is not None:
        progress_callback("amplitude stage", stage_total, stage_total)
    amplitude_stage = _compile_amplitude_stage_blueprint(
        expression_model,
        _dict(schema["amplitude_stage"]),
        value_slots=value_slots,
        global_value_component_count=global_value_component_count,
        global_momentum_parameter_count=global_momentum_parameter_count,
        model_parameter_records=model_parameter_records,
        global_parameter_symbols=parameter_symbols,
        global_value_symbols=value_symbols,
        global_model_parameter_symbols=model_parameter_symbols_by_name,
        global_real_valued_inputs=global_real_valued_inputs,
        stage_local_parameter_layout=stage_local_parameter_layout,
    )
    amplitude_stage = _stage_with_selector_domain_memberships(
        amplitude_stage,
        dag,
    )
    if stage_consumer is not None:
        stage_consumer(amplitude_stage, len(stage_records), len(stage_records))
    if release_consumed_expressions and stage_consumer is not None:
        amplitude_stage = replace(
            amplitude_stage,
            parameter_symbols=(),
            output_expressions=(),
            symbolica_functions=(),
        )
    blockers = tuple(
        blocker for stage in (*stages, amplitude_stage) for blocker in stage.blockers
    )
    return GenericStageCompilerBlueprint(
        kind="pyamplicol-generic-stage-compiler-blueprint",
        runtime_available=False,
        parameter_count=global_parameter_count,
        value_parameter_count=int(schema["parameter_layout"]["value_component_count"]),
        momentum_parameter_count=global_momentum_parameter_count,
        model_parameter_count=global_model_parameter_count,
        real_valued_inputs=global_real_valued_inputs,
        stage_count=len(stages) + 1,
        stages=stages,
        amplitude_stage=amplitude_stage,
        expression_ready=not blockers,
        blockers=blockers,
        parameter_symbols=parameter_symbols,
    )


def _chunk_evaluation_occurrence_count(
    current_order: Sequence[int],
    *,
    output_size_by_current: Mapping[int, int],
    evaluation_groups_by_current: Mapping[int, frozenset[int]],
    chunk_size: int,
) -> int:
    chunk_groups: list[set[int]] = []
    output_offset = 0
    for current_id in current_order:
        output_size = int(output_size_by_current[current_id])
        if output_size <= 0:
            continue
        first_chunk = output_offset // chunk_size
        last_chunk = (output_offset + output_size - 1) // chunk_size
        while len(chunk_groups) <= last_chunk:
            chunk_groups.append(set())
        groups = evaluation_groups_by_current.get(current_id, frozenset())
        for chunk_index in range(first_chunk, last_chunk + 1):
            chunk_groups[chunk_index].update(groups)
        output_offset += output_size
    return sum(len(groups) for groups in chunk_groups)


def _fanout_aware_current_order(
    current_ids: Sequence[int],
    *,
    output_size_by_current: Mapping[int, int],
    evaluation_groups_by_current: Mapping[int, frozenset[int]],
    chunk_size: int,
) -> tuple[tuple[int, ...], int, int]:
    """Cluster current outputs whose kernel evaluations have shared fan-out.

    Shared evaluation groups form a sparse hypergraph over result currents.
    Indexed heaps reproduce the overlap/benefit greedy choice without scanning
    every unplaced current.  Pathologically large fan-outs use a deterministic
    anchor-group sort, keeping the construction bounded for very large stages.
    """

    natural_order = tuple(int(current_id) for current_id in current_ids)
    before = _chunk_evaluation_occurrence_count(
        natural_order,
        output_size_by_current=output_size_by_current,
        evaluation_groups_by_current=evaluation_groups_by_current,
        chunk_size=chunk_size,
    )
    frequencies = Counter(
        group_id
        for current_id in natural_order
        for group_id in evaluation_groups_by_current.get(current_id, frozenset())
    )
    shared_groups = {
        group_id for group_id, frequency in frequencies.items() if frequency > 1
    }
    if not shared_groups or len(natural_order) < 2:
        return natural_order, before, before

    shared_by_current = {
        current_id: tuple(
            sorted(
                group_id
                for group_id in evaluation_groups_by_current.get(
                    current_id,
                    frozenset(),
                )
                if group_id in shared_groups
            )
        )
        for current_id in natural_order
    }
    members_by_group: dict[int, list[int]] = {
        group_id: [] for group_id in shared_groups
    }
    for current_id, group_ids in shared_by_current.items():
        for group_id in group_ids:
            members_by_group[group_id].append(current_id)

    benefit_by_current = {
        current_id: sum(
            frequencies[group_id] - 1 for group_id in shared_by_current[current_id]
        )
        for current_id in natural_order
    }
    large_fanout_limit = max(1024, 8 * chunk_size)
    if max(frequencies.values()) > large_fanout_limit:

        def anchor_key(current_id: int) -> tuple[int, int, int, int, int]:
            group_ids = shared_by_current[current_id]
            if not group_ids:
                return (1, 0, 0, 0, current_id)
            anchor = min(
                group_ids,
                key=lambda group_id: (-frequencies[group_id], group_id),
            )
            return (
                0,
                -frequencies[anchor],
                anchor,
                -benefit_by_current[current_id],
                current_id,
            )

        candidate_order = tuple(sorted(natural_order, key=anchor_key))
    else:
        remaining = set(natural_order)
        seed_heap = [
            (
                -benefit_by_current[current_id],
                -len(
                    evaluation_groups_by_current.get(
                        current_id,
                        frozenset(),
                    )
                ),
                current_id,
            )
            for current_id in natural_order
        ]
        heapq.heapify(seed_heap)
        fitting_heaps: dict[int, list[tuple[int, int]]] = {}
        for current_id in natural_order:
            size = int(output_size_by_current[current_id])
            fitting_heaps.setdefault(size, []).append(
                (-benefit_by_current[current_id], current_id)
            )
        for heap in fitting_heaps.values():
            heapq.heapify(heap)
        output_sizes = tuple(sorted(fitting_heaps))

        def pop_seed() -> int:
            while seed_heap:
                _benefit, _group_count, current_id = heapq.heappop(seed_heap)
                if current_id in remaining:
                    return current_id
            raise ValueError("fan-out ordering lost an unplaced current")

        def pop_fitting(capacity: int) -> int | None:
            selected_size: int | None = None
            selected_key: tuple[int, int, int] | None = None
            for size in output_sizes:
                if size > capacity:
                    break
                heap = fitting_heaps[size]
                while heap and heap[0][1] not in remaining:
                    heapq.heappop(heap)
                if not heap:
                    continue
                benefit, current_id = heap[0]
                key = (benefit, -size, current_id)
                if selected_key is None or key < selected_key:
                    selected_size = size
                    selected_key = key
            if selected_size is None:
                return None
            _benefit, current_id = heapq.heappop(fitting_heaps[selected_size])
            return current_id

        bins: list[list[int]] = []
        while remaining:
            seed = pop_seed()
            current_bin: list[int] = []
            groups_in_bin: set[int] = set()
            overlap_by_current: dict[int, int] = {}
            candidate_heap: list[tuple[int, int, int, int]] = []
            used = 0

            def add_current(
                current_id: int,
                current_bin: list[int],
                groups_in_bin: set[int],
                overlap_by_current: dict[int, int],
                candidate_heap: list[tuple[int, int, int, int]],
            ) -> None:
                nonlocal used
                remaining.remove(current_id)
                current_bin.append(current_id)
                used += int(output_size_by_current[current_id])
                for group_id in shared_by_current[current_id]:
                    if group_id in groups_in_bin:
                        continue
                    groups_in_bin.add(group_id)
                    for candidate_id in members_by_group[group_id]:
                        if candidate_id not in remaining:
                            continue
                        overlap = overlap_by_current.get(candidate_id, 0) + 1
                        overlap_by_current[candidate_id] = overlap
                        heapq.heappush(
                            candidate_heap,
                            (
                                -overlap,
                                -benefit_by_current[candidate_id],
                                -int(output_size_by_current[candidate_id]),
                                candidate_id,
                            ),
                        )

            def pop_overlapping(
                capacity: int,
                candidate_heap: list[tuple[int, int, int, int]],
                overlap_by_current: dict[int, int],
            ) -> int | None:
                postponed: list[tuple[int, int, int, int]] = []
                selected: int | None = None
                while candidate_heap:
                    item = heapq.heappop(candidate_heap)
                    overlap, _benefit, _size, current_id = item
                    if current_id not in remaining:
                        continue
                    if -overlap != overlap_by_current.get(current_id, 0):
                        continue
                    if int(output_size_by_current[current_id]) > capacity:
                        postponed.append(item)
                        continue
                    selected = current_id
                    break
                for item in postponed:
                    heapq.heappush(candidate_heap, item)
                return selected

            add_current(
                seed,
                current_bin,
                groups_in_bin,
                overlap_by_current,
                candidate_heap,
            )
            while remaining:
                capacity = chunk_size - used
                if capacity < 0:
                    break
                selected = pop_overlapping(
                    capacity,
                    candidate_heap,
                    overlap_by_current,
                )
                if selected is None:
                    selected = pop_fitting(capacity)
                if selected is None:
                    break
                add_current(
                    selected,
                    current_bin,
                    groups_in_bin,
                    overlap_by_current,
                    candidate_heap,
                )
            bins.append(current_bin)

        candidate_order = tuple(
            current_id for current_bin in bins for current_id in current_bin
        )
    after = _chunk_evaluation_occurrence_count(
        candidate_order,
        output_size_by_current=output_size_by_current,
        evaluation_groups_by_current=evaluation_groups_by_current,
        chunk_size=chunk_size,
    )
    if after >= before:
        return natural_order, before, before
    return candidate_order, before, after


def _stage_with_selector_domain_memberships(
    stage: GenericCompiledStageBlueprint,
    dag: GenericDAG,
) -> GenericCompiledStageBlueprint:
    materialization = dag.helicity_materialization
    replay = dag.lc_topology_replay
    if materialization is None and replay is None:
        return stage
    domains_by_current: dict[int, list[int]] = {}
    domains_by_root: dict[int, list[int]] = {}
    if materialization is not None:
        for schedule in materialization.selector_schedules:
            if schedule.structural_zero:
                continue
            for current_id in schedule.active_current_ids:
                domains_by_current.setdefault(int(current_id), []).append(
                    int(schedule.selector_domain_id)
                )
            for root_id in schedule.active_root_ids:
                domains_by_root.setdefault(int(root_id), []).append(
                    int(schedule.selector_domain_id)
                )
    color_domains_by_current, color_domains_by_root = (
        _lc_materialized_sector_memberships(dag)
    )
    output_slots = []
    for slot in stage.output_slots:
        owner_id = (
            int(slot.component_start)
            if str(stage.stage_kind).startswith("amplitude")
            else int(slot.current_id)
        )
        domains = (
            domains_by_root.get(owner_id, ())
            if str(stage.stage_kind).startswith("amplitude")
            else domains_by_current.get(owner_id, ())
        )
        output_slots.append(
            replace(
                slot,
                selector_domain_ids=tuple(sorted(set(domains))),
                color_selector_domain_ids=(
                    color_domains_by_root.get(owner_id, ())
                    if str(stage.stage_kind).startswith("amplitude")
                    else color_domains_by_current.get(owner_id, ())
                ),
            )
        )
    return replace(stage, output_slots=tuple(output_slots))


def _lc_materialized_sector_memberships(
    dag: GenericDAG,
) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    if dag.process.color_accuracy != "lc" or dag.color_coverage != "complete":
        return {}, {}
    replay = dag.lc_topology_replay
    materialized_sector_ids = (
        {int(sector.id) for sector in dag.color_plan.sectors}
        if replay is None
        else set(replay.materialized_sector_ids)
    )
    roots_by_sector: dict[int, list[int]] = {
        sector_id: [] for sector_id in materialized_sector_ids
    }
    root_by_id = {int(root.id): root for root in dag.amplitude_roots}
    for root in dag.amplitude_roots:
        sector_id = (
            int(root.color_sector_id)
            if root.color_sector_id is not None
            else int(dag.currents[root.left_id].index.color_state.sector_id)
        )
        if sector_id in roots_by_sector:
            roots_by_sector[sector_id].append(int(root.id))

    interactions_by_result: dict[int, list[tuple[int, int]]] = {}
    for interaction in dag.interactions:
        interactions_by_result.setdefault(int(interaction.result_id), []).append(
            (int(interaction.left_id), int(interaction.right_id))
        )
    compiled_dependencies = _compiled_representative_dependencies(dag)

    sectors_by_current: dict[int, set[int]] = {}
    sectors_by_root: dict[int, set[int]] = {}
    for sector_id, root_ids in roots_by_sector.items():
        if not root_ids:
            raise ValueError(
                "LC topology replay materialized sector "
                f"{sector_id} has no amplitude root"
            )
        active_currents: set[int] = set()
        pending: list[int] = []
        for root_id in root_ids:
            root = root_by_id[root_id]
            sectors_by_root.setdefault(root_id, set()).add(sector_id)
            pending.extend((int(root.left_id), int(root.right_id)))
        while pending:
            current_id = pending.pop()
            if current_id in active_currents:
                continue
            active_currents.add(current_id)
            for left_id, right_id in interactions_by_result.get(current_id, ()):
                pending.extend((left_id, right_id))
            pending.extend(compiled_dependencies.get(current_id, ()))
        for current_id in active_currents:
            sectors_by_current.setdefault(current_id, set()).add(sector_id)

    return (
        {
            current_id: tuple(sorted(sector_ids))
            for current_id, sector_ids in sectors_by_current.items()
        },
        {
            root_id: tuple(sorted(sector_ids))
            for root_id, sector_ids in sectors_by_root.items()
        },
    )


def _stage_with_fanout_aware_output_order(
    stage: GenericCompiledStageBlueprint,
    *,
    chunk_size: int | None,
) -> GenericCompiledStageBlueprint:
    if not stage.output_slots:
        return stage

    amplitude_stage = str(stage.stage_kind).startswith("amplitude")
    slots_by_owner: dict[int, list[GenericStageOutputSlot]] = {}
    for slot_index, slot in enumerate(stage.output_slots):
        owner_id = slot_index if amplitude_stage else int(slot.current_id)
        slots_by_owner.setdefault(owner_id, []).append(slot)
    natural_order = tuple(slots_by_owner)
    output_size_by_owner = {
        owner_id: sum(slot.output_stop - slot.output_start for slot in slots)
        for owner_id, slots in slots_by_owner.items()
    }
    groups_by_current = {
        int(current_id): frozenset(int(group_id) for group_id in group_ids)
        for current_id, group_ids in stage.evaluation_groups_by_current
        if current_id in slots_by_owner
    }
    signature_by_owner = {
        owner_id: (
            slots[0].selector_domain_ids,
            slots[0].color_selector_domain_ids,
        )
        for owner_id, slots in slots_by_owner.items()
    }
    selector_partitioning = any(
        helicity_ids or color_ids
        for helicity_ids, color_ids in signature_by_owner.values()
    )
    owners_by_signature: dict[
        tuple[tuple[int, ...], tuple[int, ...]], list[int]
    ] = {}
    for owner_id in natural_order:
        owners_by_signature.setdefault(
            signature_by_owner[owner_id], []
        ).append(owner_id)

    owner_order: list[int] = []
    before = 0
    after = 0
    for signature_owners in owners_by_signature.values():
        ordered = tuple(signature_owners)
        if (
            not amplitude_stage
            and chunk_size is not None
            and int(chunk_size) > 0
            and groups_by_current
        ):
            ordered, group_before, group_after = _fanout_aware_current_order(
                ordered,
                output_size_by_current=output_size_by_owner,
                evaluation_groups_by_current=groups_by_current,
                chunk_size=int(chunk_size),
            )
            before += group_before
            after += group_after
        owner_order.extend(ordered)

    if not selector_partitioning and tuple(owner_order) == natural_order:
        if chunk_size is None or int(chunk_size) < 1 or amplitude_stage:
            return stage
        return replace(
            stage,
            fanout_chunk_size=int(chunk_size),
            fanout_evaluation_occurrences_before=before,
            fanout_evaluation_occurrences_after=after,
        )

    outputs: list[Any] = []
    output_slots: list[GenericStageOutputSlot] = []
    selector_output_partitions: list[tuple[int, int]] = []
    active_signature: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    partition_start = 0
    for owner_id in owner_order:
        signature = signature_by_owner[owner_id]
        if active_signature is None:
            active_signature = signature
        elif signature != active_signature:
            selector_output_partitions.append((partition_start, len(outputs)))
            partition_start = len(outputs)
            active_signature = signature
        for slot in slots_by_owner[owner_id]:
            components = stage.output_expressions[slot.output_start : slot.output_stop]
            start = len(outputs)
            outputs.extend(components)
            output_slots.append(
                replace(
                    slot,
                    output_start=start,
                    output_stop=len(outputs),
                )
            )
    if selector_partitioning:
        selector_output_partitions.append((partition_start, len(outputs)))
    if len(outputs) != len(stage.output_expressions):
        raise ValueError("fan-out output ordering lost stage expressions")
    return replace(
        stage,
        output_slots=tuple(output_slots),
        first_output_previews=_expression_previews(outputs),
        output_expressions=tuple(outputs),
        fanout_chunk_size=(None if chunk_size is None else int(chunk_size)),
        fanout_evaluation_occurrences_before=(before or None),
        fanout_evaluation_occurrences_after=(after or None),
        selector_output_partitions=tuple(selector_output_partitions),
    )


def _prepare_stage_for_output_chunking(
    stage: GenericCompiledStageBlueprint,
    *,
    blueprint: GenericStageCompilerBlueprint | None,
    symbolica_settings: Any | None,
    current_stage_position: int | None = None,
    current_stage_count: int | None = None,
) -> GenericCompiledStageBlueprint:
    if symbolica_settings is None:
        return stage
    settings = _stage_symbolica_settings(
        stage,
        blueprint,
        symbolica_settings,
        current_stage_position=current_stage_position,
        current_stage_count=current_stage_count,
    )
    return _stage_with_fanout_aware_output_order(
        stage,
        chunk_size=getattr(settings, "compiled_output_chunk_size", None),
    )
