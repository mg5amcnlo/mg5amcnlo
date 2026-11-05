# SPDX-License-Identifier: 0BSD
"""Compatibility facade for Symbolica evaluator construction."""

from __future__ import annotations

from .symbolica_adapters import (
    _ChunkedSymbolicaEvaluator,
    _load_symbolica_evaluator_artifact,
)
from .symbolica_adapters import (
    _CompiledComplexEvaluatorAdapter as _CompiledComplexEvaluatorAdapter,
)
from .symbolica_adapters import (
    _JITSymbolicaEvaluatorAdapter as _JITSymbolicaEvaluatorAdapter,
)
from .symbolica_compile import _compile_symbolica_outputs
from .symbolica_helpers import (
    ComplexOutput,
    _artifact_path_for_manifest,
    _artifact_path_from_manifest,
    _evaluate_complex_outputs,
    _symbolica_evaluator_artifact_manifest,
)
from .symbolica_settings import ProgressCallback, SymbolicaEvaluatorSettings

__all__ = [
    "ComplexOutput",
    "ProgressCallback",
    "SymbolicaEvaluatorSettings",
    "_ChunkedSymbolicaEvaluator",
    "_artifact_path_for_manifest",
    "_artifact_path_from_manifest",
    "_compile_symbolica_outputs",
    "_evaluate_complex_outputs",
    "_load_symbolica_evaluator_artifact",
    "_symbolica_evaluator_artifact_manifest",
]
