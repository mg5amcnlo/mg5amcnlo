# SPDX-License-Identifier: 0BSD
"""Compatibility facade for model IR compilation."""

from __future__ import annotations

from collections.abc import Mapping

from .compiler_contact_trees import eager_color_singlet_vertex_term_components
from .compiler_entry import compile_ufo_model_ir
from .compiler_kernels import (
    _as_expression as _as_expression,
)
from .compiler_kernels import (
    _spin_representations as _spin_representations,
)
from .compiler_kernels import (
    _spin_slots as _spin_slots,
)
from .compiler_records import (
    _replace_evaluator_constants as _replace_evaluator_constants,
)
from .contracts import (
    CompiledCouplingOrder,
    CompiledCouplingRecord,
    CompiledModelIR,
    CompiledParameterRecord,
    CompiledParticleRecord,
    CompiledPropagatorRecord,
    CompiledVertexTerm,
)
from .contracts import (
    CompiledOrientedKernel as CompiledOrientedKernel,
)


def compile_builtin_model_ir(model: Mapping[str, object]) -> CompiledModelIR:
    """Compile the built-in model without loading it on external-model imports."""
    from .builtin.compiler import compile_model_ir

    return compile_model_ir(model)

__all__ = [
    "CompiledCouplingOrder",
    "CompiledCouplingRecord",
    "CompiledModelIR",
    "CompiledParameterRecord",
    "CompiledParticleRecord",
    "CompiledPropagatorRecord",
    "CompiledVertexTerm",
    "compile_builtin_model_ir",
    "compile_ufo_model_ir",
    "eager_color_singlet_vertex_term_components",
]
