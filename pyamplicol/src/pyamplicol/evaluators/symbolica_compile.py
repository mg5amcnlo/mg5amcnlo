# SPDX-License-Identifier: 0BSD
"""Compilation pipeline for Symbolica evaluator outputs."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from .._internal.physics.types import NativeEvaluationError
from .symbolica_adapters import (
    _ChunkedSymbolicaEvaluator,
    _CompiledComplexEvaluatorAdapter,
    _JITSymbolicaEvaluatorAdapter,
)
from .symbolica_helpers import _progress_outputs, _safe_symbol_name
from .symbolica_settings import (
    ProgressCallback,
    SymbolicaEvaluatorSettings,
    _report_progress,
)


def _compile_symbolica_outputs(
    outputs: tuple[Any, ...],
    params: list[Any],
    *,
    merge_evaluators_strategy: bool,
    verbose_evaluator_build: bool,
    functions: Mapping[tuple[Any, tuple[Any, ...]], Any] | None = None,
    real_params: Sequence[int] = (),
    symbolica_settings: SymbolicaEvaluatorSettings | None = None,
    jit_compile: bool = True,
    label: str = "symbolica",
    progress_callback: ProgressCallback | None = None,
    output_partitions: Sequence[tuple[int, int]] = (),
    native_direct_only: bool = False,
) -> Any:
    if not outputs:
        raise NativeEvaluationError("cannot build evaluator with zero outputs")
    settings = symbolica_settings or SymbolicaEvaluatorSettings()
    progress_stage = _symbolica_progress_stage(settings, jit_compile=jit_compile)
    _report_progress(
        progress_callback,
        stage=progress_stage,
        item=f"{label} prepare {len(outputs)}",
    )
    total_started = time.perf_counter()
    prepare_started = time.perf_counter()
    outputs = tuple(_prepare_symbolica_output(output, settings) for output in outputs)
    output_prepare_s = time.perf_counter() - prepare_started
    chunk_size = settings.compiled_output_chunk_size
    chunk_ranges = _partitioned_output_chunk_ranges(
        len(outputs),
        chunk_size=chunk_size,
        output_partitions=output_partitions,
    )
    if len(chunk_ranges) > 1:
        chunk_output_dir = (
            None
            if settings.compiled_output_dir is None
            else str(Path(settings.compiled_output_dir) / _safe_symbol_name(label))
        )
        unchunked_settings = replace(settings, compiled_output_chunk_size=None)
        if chunk_output_dir is not None:
            unchunked_settings = replace(
                unchunked_settings,
                compiled_output_dir=chunk_output_dir,
            )

        def compile_chunk(
            chunk_index: int,
            start: int,
            stop: int,
        ) -> tuple[Any, tuple[int, ...]]:
            chunk_outputs = outputs[start:stop]
            chunk_input_indices = _chunk_parameter_indices(
                chunk_outputs,
                params,
                functions=functions,
            )
            chunk_params = [params[index] for index in chunk_input_indices]
            parent_real_params = set(real_params)
            chunk_real_params = tuple(
                local_index
                for local_index, parent_index in enumerate(chunk_input_indices)
                if parent_index in parent_real_params
            )
            _report_progress(
                progress_callback,
                stage=progress_stage,
                item=(
                    f"{label} chunk {chunk_index + 1}/{len(chunk_ranges)} "
                    f"p={len(chunk_params)}/{len(params)}"
                ),
            )
            return (
                _compile_symbolica_outputs(
                    chunk_outputs,
                    chunk_params,
                    merge_evaluators_strategy=merge_evaluators_strategy,
                    verbose_evaluator_build=verbose_evaluator_build,
                    functions=functions,
                    real_params=chunk_real_params,
                    symbolica_settings=unchunked_settings,
                    jit_compile=jit_compile,
                    label=f"{label}_chunk_{chunk_index}",
                    progress_callback=progress_callback,
                    output_partitions=(),
                    native_direct_only=native_direct_only,
                ),
                chunk_input_indices,
            )

        workers = min(settings.compiled_chunk_compile_workers, len(chunk_ranges))
        if workers <= 1:
            compiled_chunks = [
                compile_chunk(chunk_index, start, stop)
                for chunk_index, (start, stop) in enumerate(chunk_ranges)
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(compile_chunk, chunk_index, start, stop)
                    for chunk_index, (start, stop) in enumerate(chunk_ranges)
                ]
                compiled_chunks = [future.result() for future in futures]
        chunks, chunk_input_indices = zip(*compiled_chunks, strict=True)
        chunked = _ChunkedSymbolicaEvaluator(
            tuple(chunks),
            input_len=len(params),
            chunk_input_indices=tuple(chunk_input_indices),
        )
        chunked.build_timing = {
            "output_prepare_s": output_prepare_s,
            "symbolica_evaluator_build_s": time.perf_counter() - total_started,
        }
        return chunked
    evaluator_kwargs = _symbolica_evaluator_kwargs(
        settings,
        verbose=verbose_evaluator_build,
        jit_compile=jit_compile,
    )
    function_kwargs: dict[str, Any] = {}
    if functions:
        from symbolica import FunctionDefinition

        # Process kernels retain one flat instruction program, including bodies
        # of tagged functions and functions that close over model parameters.
        function_kwargs["functions"] = [
            FunctionDefinition(function, arguments, body, inlining="always")
            for (function, arguments), body in functions.items()
        ]
    if merge_evaluators_strategy:
        _report_progress(
            progress_callback,
            stage=progress_stage,
            item=f"{label} evaluator 1/{len(outputs)}",
        )
        _report_jit_boundary(
            progress_callback,
            settings,
            jit_compile=jit_compile,
            phase="initialize",
            item=f"{label} eval 1/{len(outputs)} p={len(params)}",
        )
        with _JITBoundaryHeartbeat(
            progress_callback,
            settings,
            jit_compile=jit_compile,
            phase="initialize",
            item=f"{label} eval 1/{len(outputs)} p={len(params)}",
        ):
            evaluator_started = time.perf_counter()
            evaluator = outputs[0].evaluator(
                params,
                **function_kwargs,
                **evaluator_kwargs,
            )
            evaluator_construct_s = time.perf_counter() - evaluator_started
        _report_jit_boundary(
            progress_callback,
            settings,
            jit_compile=jit_compile,
            phase="returned",
            item=f"{label} eval 1/{len(outputs)}",
        )
        for expression in _progress_outputs(
            outputs[1:],
            enabled=verbose_evaluator_build,
        ):
            _report_progress(
                progress_callback,
                stage=progress_stage,
                item=f"{label} merge",
            )
            _report_jit_boundary(
                progress_callback,
                settings,
                jit_compile=jit_compile,
                phase="initialize",
                item=f"{label} merge p={len(params)}",
            )
            with _JITBoundaryHeartbeat(
                progress_callback,
                settings,
                jit_compile=jit_compile,
                phase="initialize",
                item=f"{label} merge p={len(params)}",
            ):
                merge_construct_started = time.perf_counter()
                other = expression.evaluator(
                    params,
                    **function_kwargs,
                    **evaluator_kwargs,
                )
                evaluator_construct_s += time.perf_counter() - merge_construct_started
            _report_jit_boundary(
                progress_callback,
                settings,
                jit_compile=jit_compile,
                phase="returned",
                item=f"{label} merge",
            )
            evaluator.merge(
                other,
                cpe_iterations=(
                    1 if settings.cpe_iterations is None else settings.cpe_iterations
                ),
            )
        if real_params:
            real_params_started = time.perf_counter()
            _report_jit_boundary(
                progress_callback,
                settings,
                jit_compile=jit_compile,
                phase="real params",
                item=f"{label} real={len(real_params)}",
            )
            evaluator.set_real_params(
                list(real_params),
                sqrt_real=settings.real_param_sqrt_real,
                log_real=settings.real_param_log_real,
                powf_real=settings.real_param_powf_real,
                real_if_args_real=settings.real_param_real_if_args_real,
                verbose=verbose_evaluator_build,
            )
            real_params_s = time.perf_counter() - real_params_started
        else:
            real_params_s = 0.0
        adapter = _finalize_symbolica_evaluator(
            evaluator,
            settings,
            label,
            input_len=len(params),
            output_len=len(outputs),
            progress_callback=progress_callback,
            native_direct_only=native_direct_only,
        )
        _set_evaluator_build_timing(
            adapter,
            {
                "output_prepare_s": output_prepare_s,
                "evaluator_construct_s": evaluator_construct_s,
                "real_params_s": real_params_s,
                "symbolica_evaluator_build_s": time.perf_counter() - total_started,
            },
        )
        return adapter

    from symbolica import Expression

    _report_progress(
        progress_callback,
        stage=progress_stage,
        item=f"{label} evaluator {len(outputs)}",
    )
    _report_jit_boundary(
        progress_callback,
        settings,
        jit_compile=jit_compile,
        phase="initialize",
        item=f"{label} out={len(outputs)} p={len(params)}",
    )
    with _JITBoundaryHeartbeat(
        progress_callback,
        settings,
        jit_compile=jit_compile,
        phase="initialize",
        item=f"{label} out={len(outputs)} p={len(params)}",
    ):
        evaluator_started = time.perf_counter()
        evaluator = Expression.evaluator_multiple(
            outputs,
            params,
            **function_kwargs,
            **evaluator_kwargs,
        )
        evaluator_construct_s = time.perf_counter() - evaluator_started
    _report_jit_boundary(
        progress_callback,
        settings,
        jit_compile=jit_compile,
        phase="returned",
        item=f"{label} out={len(outputs)}",
    )
    if real_params:
        real_params_started = time.perf_counter()
        _report_jit_boundary(
            progress_callback,
            settings,
            jit_compile=jit_compile,
            phase="real params",
            item=f"{label} real={len(real_params)}",
        )
        evaluator.set_real_params(
            list(real_params),
            sqrt_real=settings.real_param_sqrt_real,
            log_real=settings.real_param_log_real,
            powf_real=settings.real_param_powf_real,
            real_if_args_real=settings.real_param_real_if_args_real,
            verbose=verbose_evaluator_build,
        )
        real_params_s = time.perf_counter() - real_params_started
    else:
        real_params_s = 0.0
    adapter = _finalize_symbolica_evaluator(
        evaluator,
        settings,
        label,
        input_len=len(params),
        output_len=len(outputs),
        progress_callback=progress_callback,
        native_direct_only=native_direct_only,
    )
    _set_evaluator_build_timing(
        adapter,
        {
            "output_prepare_s": output_prepare_s,
            "evaluator_construct_s": evaluator_construct_s,
            "real_params_s": real_params_s,
            "symbolica_evaluator_build_s": time.perf_counter() - total_started,
        },
    )
    return adapter


def _partitioned_output_chunk_ranges(
    output_count: int,
    *,
    chunk_size: int | None,
    output_partitions: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    if output_count < 1:
        raise NativeEvaluationError("cannot partition zero evaluator outputs")
    partitions = tuple(output_partitions) or ((0, output_count),)
    expected_start = 0
    for start, stop in partitions:
        if start != expected_start or stop <= start or stop > output_count:
            raise NativeEvaluationError(
                "evaluator output partitions must be contiguous and exhaustive"
            )
        expected_start = stop
    if expected_start != output_count:
        raise NativeEvaluationError(
            "evaluator output partitions must cover every output"
        )

    ranges: list[tuple[int, int]] = []
    for start, stop in partitions:
        if chunk_size is None:
            ranges.append((start, stop))
            continue
        if chunk_size < 1:
            raise NativeEvaluationError("evaluator output chunk size must be positive")
        for chunk_start in range(start, stop, chunk_size):
            ranges.append((chunk_start, min(chunk_start + chunk_size, stop)))
    return tuple(ranges)


def _chunk_parameter_indices(
    outputs: Sequence[Any],
    params: Sequence[Any],
    *,
    functions: Mapping[tuple[Any, tuple[Any, ...]], Any] | None = None,
) -> tuple[int, ...]:
    """Return parent parameter indices needed by one output chunk.

    Symbolica already exposes structural symbol discovery, including symbols in
    function arguments.  Function bodies are included
    conservatively so a model kernel that closes over a runtime parameter can
    never be under-specified.  Parent ordering is retained to keep generated
    evaluator signatures and real-parameter metadata deterministic.
    """

    used_symbols: set[Any] = set()
    for expression in outputs:
        used_symbols.update(_expression_symbols(expression))
    if functions:
        for body in functions.values():
            used_symbols.update(_expression_symbols(body))
    return tuple(
        index for index, parameter in enumerate(params) if parameter in used_symbols
    )


def _expression_symbols(expression: Any) -> set[Any]:
    getter = getattr(expression, "get_all_symbols", None)
    if not callable(getter):
        raise NativeEvaluationError(
            "Symbolica expression does not expose structural symbol discovery"
        )
    return set(getter(False))


def _set_evaluator_build_timing(evaluator: Any, timing: Mapping[str, float]) -> None:
    try:
        current = getattr(evaluator, "build_timing", {})
        if not isinstance(current, dict):
            current = {}
        current.update({str(key): float(value) for key, value in timing.items()})
        evaluator.build_timing = current
    except Exception:
        return


def _report_jit_boundary(
    progress_callback: ProgressCallback | None,
    settings: SymbolicaEvaluatorSettings,
    *,
    jit_compile: bool,
    phase: str,
    item: str,
) -> None:
    if settings.backend != "jit" or not jit_compile:
        return
    _report_progress(
        progress_callback,
        stage=f"jit {phase}",
        item=item,
    )


class _JITBoundaryHeartbeat:
    """Emit progress while Symbolica is inside a blocking JIT build call."""

    def __init__(
        self,
        progress_callback: ProgressCallback | None,
        settings: SymbolicaEvaluatorSettings,
        *,
        jit_compile: bool,
        phase: str,
        item: str,
        interval_s: float = 5.0,
    ) -> None:
        self.progress_callback = progress_callback
        self.settings = settings
        self.jit_compile = jit_compile
        self.phase = phase
        self.item = item
        self.interval_s = max(float(interval_s), 0.01)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def __enter__(self) -> _JITBoundaryHeartbeat:
        if (
            self.progress_callback is None
            or self.settings.backend != "jit"
            or not self.jit_compile
        ):
            return self
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            elapsed_s = time.perf_counter() - self._started_at
            _report_progress(
                self.progress_callback,
                stage=f"jit {self.phase}",
                item=f"{self.item} waiting {elapsed_s:.0f}s",
            )


def _symbolica_progress_stage(
    settings: SymbolicaEvaluatorSettings,
    *,
    jit_compile: bool,
) -> str:
    if settings.backend == "jit":
        return "jit compile" if jit_compile else "jit build"
    if settings.backend == "compiled-complex":
        return "c++ build"
    return "eval build"


def _prepare_symbolica_output(
    output: Any,
    settings: SymbolicaEvaluatorSettings,
) -> Any:
    if not settings.collect_factors:
        return output
    collect_factors = getattr(output, "collect_factors", None)
    if callable(collect_factors):
        return collect_factors()
    return output


def _symbolica_evaluator_kwargs(
    settings: SymbolicaEvaluatorSettings,
    *,
    verbose: bool,
    jit_compile: bool = True,
) -> dict[str, Any]:
    return {
        "iterations": settings.iterations,
        "cpe_iterations": settings.cpe_iterations,
        "n_cores": settings.n_cores,
        "verbose": verbose,
        "jit_compile": jit_compile,
        "direct_translation": settings.direct_translation,
        "jit_direct_translation": settings.jit_direct_translation,
        "jit_optimization_level": settings.jit_optimization_level,
        "jit_options": {"compress": "true" if settings.jit_compress else "false"},
        "max_horner_scheme_variables": settings.max_horner_scheme_variables,
        "max_common_pair_cache_entries": settings.max_common_pair_cache_entries,
        "max_common_pair_distance": settings.max_common_pair_distance,
    }


def _finalize_symbolica_evaluator(
    evaluator: Any,
    settings: SymbolicaEvaluatorSettings,
    label: str,
    *,
    input_len: int,
    output_len: int,
    progress_callback: ProgressCallback | None = None,
    native_direct_only: bool = False,
) -> Any:
    if settings.backend == "jit":
        _report_progress(
            progress_callback,
            stage="jit ready",
            item=label,
        )
        return _JITSymbolicaEvaluatorAdapter(
            evaluator,
            settings,
            label,
            input_len=input_len,
            output_len=output_len,
            progress_callback=progress_callback,
        )
    if settings.backend == "compiled-complex":
        _report_progress(
            progress_callback,
            stage="c++ compile",
            item=label,
        )
        return _CompiledComplexEvaluatorAdapter(
            evaluator,
            settings,
            label,
            input_len=input_len,
            output_len=output_len,
            compile_dense=not native_direct_only,
        )
    raise NativeEvaluationError(
        f"unsupported symbolica evaluator backend: {settings.backend}"
    )
