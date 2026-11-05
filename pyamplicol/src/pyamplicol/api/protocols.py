# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Protocol, runtime_checkable

import pyamplicol as _pyamplicol
from pyamplicol.config import (
    BenchmarkConfig,
    ConfigResolution,
    GenerationConfig,
    RunConfig,
)
from pyamplicol.reporting import ProgressSink

from .requests import ModelSource, ProcessSet
from .results import (
    BenchmarkResult,
    GenerationPlan,
    GenerationResult,
    ProcessPhysics,
    ResolvedEvaluation,
)

ScalarInput = float | int | str | Decimal
ScalarValue = complex | Decimal
Momenta = Sequence[Sequence[Sequence[ScalarInput]]]
ModelParameters = Mapping[
    str,
    complex | float | int | list[float | int] | tuple[float | int, float | int],
]


@runtime_checkable
class GeneratorBackend(Protocol):
    def plan(
        self,
        processes: ProcessSet,
        *,
        model: ModelSource | _pyamplicol.CompiledModel | None = None,
    ) -> GenerationPlan: ...

    def generate(
        self,
        processes: ProcessSet,
        output: os.PathLike[str] | str,
        *,
        model: ModelSource | _pyamplicol.CompiledModel | None = None,
        mode: str = "error",
    ) -> GenerationResult: ...


@runtime_checkable
class RuntimeBackend(Protocol):
    """Minimum structural runtime contract.

    Authenticated ``artifact_id`` and ``execution_mode`` are facade capabilities
    rather than protocol requirements so existing injected third-party backends
    remain structurally compatible. Built-in backends expose both, and
    :class:`pyamplicol.Runtime` fails closed when either identity is requested
    but unavailable.
    """

    @property
    def physics(self) -> ProcessPhysics: ...

    def evaluate(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int = 16,
    ) -> Sequence[ScalarValue]: ...

    def evaluate_resolved(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int = 16,
    ) -> ResolvedEvaluation: ...

    def set_model_parameters(self, mapping: ModelParameters) -> None: ...

    def clear(self) -> None: ...

    def mute_warnings(self) -> None: ...

    def unmute_warnings(self) -> None: ...


@runtime_checkable
class BenchmarkBackend(Protocol):
    def run(
        self,
        target: RuntimeBackend | os.PathLike[str] | str,
        *,
        points: Momenta | None = None,
    ) -> BenchmarkResult: ...


class GeneratorFactory(Protocol):
    def __call__(
        self,
        config: GenerationConfig | RunConfig | ConfigResolution | None,
        progress: ProgressSink | None,
    ) -> GeneratorBackend: ...


class RuntimeLoader(Protocol):
    def __call__(
        self,
        artifact: os.PathLike[str] | str,
        *,
        process: str | None,
        model_parameters: ModelParameters | None,
        mute_warnings: bool,
    ) -> RuntimeBackend: ...


class BenchmarkFactory(Protocol):
    def __call__(
        self,
        config: BenchmarkConfig | RunConfig | None,
        progress: ProgressSink | None,
    ) -> BenchmarkBackend: ...


__all__ = [
    "BenchmarkBackend",
    "BenchmarkFactory",
    "GeneratorBackend",
    "GeneratorFactory",
    "ModelParameters",
    "Momenta",
    "RuntimeBackend",
    "RuntimeLoader",
    "ScalarInput",
    "ScalarValue",
]
