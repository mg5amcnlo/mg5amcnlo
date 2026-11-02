# SPDX-License-Identifier: 0BSD
"""Compatibility facade for stage blueprint and evaluator compilation."""

from __future__ import annotations

from .stage_artifacts import (
    build_and_write_generic_stage_evaluator_artifacts,
    write_generic_stage_evaluator_artifacts,
    write_model_parameter_evaluator_artifact,
)
from .stage_planning import (
    _fanout_aware_current_order as _fanout_aware_current_order,
)
from .stage_planning import (
    build_generic_stage_compiler_blueprint,
)
from .stage_types import (
    GenericCompiledStageBlueprint,
    GenericStageCompilerBlueprint,
    GenericStageInputComponent,
    GenericStageOutputSlot,
)

__all__ = [
    "GenericCompiledStageBlueprint",
    "GenericStageCompilerBlueprint",
    "GenericStageInputComponent",
    "GenericStageOutputSlot",
    "build_and_write_generic_stage_evaluator_artifacts",
    "build_generic_stage_compiler_blueprint",
    "write_generic_stage_evaluator_artifacts",
    "write_model_parameter_evaluator_artifact",
]
