# SPDX-License-Identifier: 0BSD
"""Per-stage evaluator settings."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .stage_types import GenericCompiledStageBlueprint, GenericStageCompilerBlueprint

_LOW_RANK_CURRENT_COMPONENT_LIMIT = 8
_HIGH_RANK_CURRENT_CHUNK_LIMIT = 256


def _stage_symbolica_settings(
    stage: GenericCompiledStageBlueprint,
    blueprint: GenericStageCompilerBlueprint | None,
    settings: Any,
    *,
    current_stage_position: int | None = None,
    current_stage_count: int | None = None,
) -> Any:
    """Derive the stage-specific evaluator output-chunk policy."""

    settings = _bound_high_rank_current_chunk(stage, settings)

    strategy = getattr(settings, "output_chunk_strategy", "uniform")
    if strategy == "auto":
        base = getattr(settings, "compiled_output_chunk_size", None)
        output_count = int(
            getattr(
                stage,
                "output_length",
                0 if base is None else int(base) + 1,
            )
        )
        strategy = (
            "measured-stage"
            if getattr(settings, "backend", None) == "jit"
            and base is not None
            and int(base) <= 256
            and output_count > int(base)
            else "uniform"
        )
        settings = replace(settings, output_chunk_strategy=strategy)
    if strategy != "tapered-stage":
        return settings
    base = getattr(settings, "compiled_output_chunk_size", None)
    if base is None:
        return settings
    if str(stage.stage_kind).startswith("amplitude"):
        return replace(settings, compiled_output_chunk_size=None)

    if blueprint is not None:
        try:
            position = next(
                index
                for index, current_stage in enumerate(blueprint.stages)
                if current_stage.stage_index == stage.stage_index
            )
        except StopIteration:
            return settings
        stage_count = len(blueprint.stages)
    else:
        if current_stage_position is None or current_stage_count is None:
            return settings
        position = int(current_stage_position)
        stage_count = int(current_stage_count)
    remaining = stage_count - position - 1
    if position < 2:
        chunk_size = None
    elif remaining == 0:
        chunk_size = max(1, int(base) // 2)
    elif remaining <= 2:
        chunk_size = int(base)
    else:
        chunk_size = int(base) * 2
    return replace(settings, compiled_output_chunk_size=chunk_size)


def _bound_high_rank_current_chunk(
    stage: GenericCompiledStageBlueprint,
    settings: Any,
) -> Any:
    """Keep native evaluators for higher-rank current states at a safe size."""

    chunk_size = getattr(settings, "compiled_output_chunk_size", None)
    if (
        chunk_size is None
        or int(chunk_size) <= _HIGH_RANK_CURRENT_CHUNK_LIMIT
        or str(stage.stage_kind).startswith("amplitude")
    ):
        return settings
    max_slot_width = max(
        (
            int(slot.component_stop) - int(slot.component_start)
            for slot in stage.output_slots
        ),
        default=0,
    )
    if max_slot_width <= _LOW_RANK_CURRENT_COMPONENT_LIMIT:
        return settings
    return replace(
        settings,
        compiled_output_chunk_size=_HIGH_RANK_CURRENT_CHUNK_LIMIT,
    )
