# SPDX-License-Identifier: 0BSD
"""Array, artifact-path, and progress helpers for evaluator adapters."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .._internal.physics.types import NativeEvaluationError

ComplexOutput = np.ndarray | tuple[np.ndarray, ...]


def _symbolica_instruction_program(exported: Any) -> Any:
    """Extract the linear program from Symbolica's structured instruction export."""
    if isinstance(exported, tuple):
        return exported
    if exported.sub_evaluators:
        raise NativeEvaluationError(
            "native kernels require inlined Symbolica function bodies"
        )
    return exported.instructions, exported.temporary_count, exported.constants


def _symbolica_instruction_program_repr(exported: Any) -> str:
    """Spell out unit imaginary coefficients for SymJIT's numeric grammar.

    Symbolica omits coefficients of magnitude one before its imaginary-unit
    glyph, but SymJIT's instruction parser requires them. Keep the exact
    rational constants and the original evaluator state unchanged.
    """
    instructions, temporary_count, constants = _symbolica_instruction_program(exported)
    serialized_constants = ", ".join(
        re.sub(r"(?<![0-9])\U0001d456", "1\U0001d456", repr(value))
        for value in constants
    )
    return f"({instructions!r}, {temporary_count}, [{serialized_constants}])"


def _complex128_parameter_rows(parameter_rows: Any) -> np.ndarray:
    if (
        isinstance(parameter_rows, np.ndarray)
        and parameter_rows.dtype == np.complex128
        and parameter_rows.flags.c_contiguous
    ):
        return parameter_rows
    return np.asarray(parameter_rows, dtype=np.complex128)


def _evaluate_prepared_complex(evaluator: Any, parameter_rows: np.ndarray) -> Any:
    evaluate_prepared = getattr(evaluator, "_evaluate_complex_prepared", None)
    if callable(evaluate_prepared):
        return evaluate_prepared(parameter_rows)
    return evaluator.evaluate_complex(parameter_rows)


def _evaluate_prepared_complex_profiled(
    evaluator: Any,
    parameter_rows: np.ndarray,
) -> tuple[Any, tuple[float, float, float]]:
    evaluate_profiled = getattr(evaluator, "_evaluate_complex_profiled_prepared", None)
    if not callable(evaluate_profiled):
        raise NativeEvaluationError(
            "evaluator does not expose native complex profiling"
        )
    return evaluate_profiled(parameter_rows)


def _evaluate_complex_outputs(evaluator: Any, parameter_rows: Any) -> ComplexOutput:
    evaluate_chunks = getattr(evaluator, "evaluate_complex_chunks", None)
    if callable(evaluate_chunks):
        return tuple(
            np.asarray(chunk, dtype=np.complex128)
            for chunk in evaluate_chunks(parameter_rows)
        )
    return np.asarray(evaluator.evaluate_complex(parameter_rows), dtype=np.complex128)


def _symbolica_evaluator_artifact_manifest(
    evaluator: Any,
    artifact_dir: Path,
) -> dict[str, Any]:
    manifest = getattr(evaluator, "artifact_manifest", None)
    if not callable(manifest):
        raise NativeEvaluationError(
            "saving evaluator artifacts is currently supported only for "
            "Symbolica evaluator adapters"
        )
    return manifest(artifact_dir)


def _artifact_path_for_manifest(path: Path, artifact_dir: Path) -> str:
    artifact_root = artifact_dir.resolve()
    source_path = path.expanduser()
    source_resolved = source_path.resolve()
    try:
        source_resolved.relative_to(artifact_root)
    except ValueError:
        if source_path.exists():
            target_dir = artifact_dir / "compiled" / "repackaged"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / source_path.name
            if source_resolved != target.resolve():
                shutil.copy2(source_path, target)
            source_path = target
        else:
            source_path = source_resolved
    try:
        return os.path.relpath(source_path, artifact_dir)
    except ValueError:
        return str(source_path)


def _artifact_subdirectory(artifact_dir: Path, name: str) -> Path:
    directory = artifact_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _artifact_path_from_manifest(path: str, artifact_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return artifact_dir / candidate


def _safe_symbol_name(value: str) -> str:
    result = "".join(character if character.isalnum() else "_" for character in value)
    if not result:
        return "eval"
    if result[0].isdigit():
        return f"eval_{result}"
    return result


def _progress_outputs(
    outputs: tuple[Any, ...],
    *,
    enabled: bool,
) -> Iterable[Any]:
    if not enabled:
        return outputs
    try:
        from colorama import Fore, Style  # type: ignore[import-untyped]
        from progressbar import (  # type: ignore[import-not-found]
            ETA,
            Bar,
            Percentage,
            ProgressBar,
        )
    except ImportError:
        return outputs
    widgets = [
        Fore.CYAN,
        " merging evaluators ",
        Percentage(),
        " ",
        Bar(),
        " ",
        ETA(),
        Style.RESET_ALL,
    ]
    return ProgressBar(max_value=len(outputs), widgets=widgets)(outputs)
