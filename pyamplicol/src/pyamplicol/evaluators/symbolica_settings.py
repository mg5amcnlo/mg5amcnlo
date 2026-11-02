# SPDX-License-Identifier: 0BSD
"""Evaluator settings and progress callback contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .._internal.physics.types import NativeEvaluationError

ProgressCallback = Callable[[dict[str, object]], None]


def _report_progress(
    callback: ProgressCallback | None,
    *,
    stage: str,
    item: str,
    increment: int = 0,
    total: int | None = None,
) -> None:
    if callback is None:
        return
    payload: dict[str, object] = {
        "stage": stage,
        "item": item,
    }
    if increment:
        payload["increment"] = int(increment)
    if total is not None:
        payload["total"] = int(total)
    callback(payload)


@dataclass(frozen=True)
class SymbolicaEvaluatorSettings:
    backend: str = "jit"
    iterations: int = 10
    cpe_iterations: int | None = None
    n_cores: int = 4
    direct_translation: bool = True
    jit_direct_translation: bool = False
    jit_optimization_level: int = 2
    jit_compress: bool = False
    max_horner_scheme_variables: int = 1000
    max_common_pair_cache_entries: int = 5000000
    max_common_pair_distance: int = 1000
    collect_factors: bool = False
    compiled_inline_asm: str = "default"
    compiled_optimization_level: int = 3
    compiled_native: bool = True
    compiler_path: str | None = None
    compiler_flags: tuple[str, ...] = ()
    compiled_output_chunk_size: int | None = None
    output_chunk_strategy: str = "uniform"
    output_chunk_autotune_batch_size: int = 128
    compiled_chunk_compile_workers: int = 1
    compiled_output_dir: str | None = None
    raw_sum_final_stage: bool = False
    real_param_sqrt_real: bool = False
    real_param_log_real: bool = False
    real_param_powf_real: bool = False
    real_param_real_if_args_real: bool = False

    def __post_init__(self) -> None:
        if self.backend not in ("jit", "compiled-complex"):
            raise NativeEvaluationError(
                "symbolica evaluator backend must be 'jit' or 'compiled-complex'"
            )
        if self.iterations < 1:
            raise NativeEvaluationError("symbolica iterations must be positive")
        if self.cpe_iterations is not None and self.cpe_iterations < 0:
            raise NativeEvaluationError("symbolica cpe iterations must be non-negative")
        if self.n_cores < 1:
            raise NativeEvaluationError("symbolica n_cores must be positive")
        if self.jit_optimization_level not in (0, 1, 2, 3):
            raise NativeEvaluationError(
                "symbolica jit optimization level must be 0, 1, 2, or 3"
            )
        if not isinstance(self.jit_compress, bool):
            raise NativeEvaluationError("symbolica jit_compress must be a boolean")
        if self.compiled_optimization_level not in (0, 1, 2, 3):
            raise NativeEvaluationError(
                "symbolica compiled optimization level must be 0, 1, 2, or 3"
            )
        if (
            self.compiled_output_chunk_size is not None
            and self.compiled_output_chunk_size < 1
        ):
            raise NativeEvaluationError(
                "symbolica compiled output chunk size must be positive"
            )
        if self.output_chunk_strategy not in (
            "auto",
            "uniform",
            "tapered-stage",
            "measured-stage",
        ):
            raise NativeEvaluationError(
                "symbolica output chunk strategy must be 'auto', 'uniform', "
                "'tapered-stage', or 'measured-stage'"
            )
        if self.output_chunk_autotune_batch_size < 1:
            raise NativeEvaluationError(
                "symbolica output chunk autotune batch size must be positive"
            )
        if self.compiled_chunk_compile_workers < 1:
            raise NativeEvaluationError(
                "symbolica compiled chunk compile workers must be positive"
            )
        if self.max_horner_scheme_variables < 1:
            raise NativeEvaluationError(
                "symbolica max_horner_scheme_variables must be positive"
            )
        if self.max_common_pair_cache_entries < 1:
            raise NativeEvaluationError(
                "symbolica max_common_pair_cache_entries must be positive"
            )
        if self.max_common_pair_distance < 1:
            raise NativeEvaluationError(
                "symbolica max_common_pair_distance must be positive"
            )

    def to_json_dict(self) -> dict[str, object]:
        from .symbolica_adapters import _compiled_compiler_flags

        return {
            "backend": self.backend,
            "iterations": self.iterations,
            "cpe_iterations": self.cpe_iterations,
            "n_cores": self.n_cores,
            "direct_translation": self.direct_translation,
            "jit_direct_translation": self.jit_direct_translation,
            "jit_optimization_level": self.jit_optimization_level,
            "jit_compress": self.jit_compress,
            "max_horner_scheme_variables": self.max_horner_scheme_variables,
            "max_common_pair_cache_entries": self.max_common_pair_cache_entries,
            "max_common_pair_distance": self.max_common_pair_distance,
            "collect_factors": self.collect_factors,
            "compiled_inline_asm": self.compiled_inline_asm,
            "compiled_optimization_level": self.compiled_optimization_level,
            "compiled_native": self.compiled_native,
            "compiler_path": self.compiler_path,
            "compiler_flags": list(self.compiler_flags),
            "effective_compiler_flags": list(_compiled_compiler_flags(self)),
            "compiled_output_chunk_size": self.compiled_output_chunk_size,
            "output_chunk_strategy": self.output_chunk_strategy,
            "output_chunk_autotune_batch_size": (self.output_chunk_autotune_batch_size),
            "compiled_chunk_compile_workers": self.compiled_chunk_compile_workers,
            "compiled_output_dir": self.compiled_output_dir,
            "raw_sum_final_stage": self.raw_sum_final_stage,
            "real_param_sqrt_real": self.real_param_sqrt_real,
            "real_param_log_real": self.real_param_log_real,
            "real_param_powf_real": self.real_param_powf_real,
            "real_param_real_if_args_real": self.real_param_real_if_args_real,
        }
