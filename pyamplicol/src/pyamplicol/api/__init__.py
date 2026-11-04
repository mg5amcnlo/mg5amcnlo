# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .errors import (
        ArtifactError,
        CompatibilityError,
        ConfigurationError,
        DependencyError,
        EvaluationError,
        GenerationError,
        ModelError,
        PyAmpliColError,
    )
    from .models import (
        CompiledModel,
        CompiledModelCapabilities,
        CompiledModelInfo,
        CompiledModelSource,
        ModelCompilationIssue,
        ModelCompilationPhase,
    )
    from .process_groups import group_processes
    from .protocols import (
        BenchmarkBackend,
        BenchmarkFactory,
        GeneratorBackend,
        GeneratorFactory,
        ModelParameters,
        Momenta,
        RuntimeBackend,
        RuntimeLoader,
    )
    from .requests import (
        ModelSource,
        ModelSourceKind,
        ProcessAlias,
        ProcessRequest,
        ProcessSet,
    )
    from .results import (
        BenchmarkComponentTiming,
        BenchmarkProfileCounters,
        BenchmarkResult,
        BenchmarkStageTiming,
        BenchmarkStatistics,
        BenchmarkTimingBreakdown,
        ColorComponent,
        ColorFlow,
        ContractedColorComponent,
        ExternalParticle,
        GenerationPlan,
        GenerationResult,
        HelicityConfiguration,
        ModelParameter,
        PhysicsReduction,
        ProcessPhysics,
        ReductionGroup,
        ResolvedEvaluation,
        WarmUpResult,
    )
    from .services import (
        BenchmarkRunner,
        Generator,
        Runtime,
        benchmark,
        generate,
        install_backend_factories,
        load,
    )

__all__ = [
    "ArtifactError",
    "BenchmarkBackend",
    "BenchmarkComponentTiming",
    "BenchmarkFactory",
    "BenchmarkProfileCounters",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkStageTiming",
    "BenchmarkStatistics",
    "BenchmarkTimingBreakdown",
    "ColorComponent",
    "ColorFlow",
    "CompatibilityError",
    "CompiledModel",
    "CompiledModelCapabilities",
    "CompiledModelInfo",
    "CompiledModelSource",
    "ConfigurationError",
    "ContractedColorComponent",
    "DependencyError",
    "EvaluationError",
    "ExternalParticle",
    "GenerationError",
    "GenerationPlan",
    "GenerationResult",
    "Generator",
    "GeneratorBackend",
    "GeneratorFactory",
    "HelicityConfiguration",
    "ModelCompilationIssue",
    "ModelCompilationPhase",
    "ModelError",
    "ModelParameter",
    "ModelParameters",
    "ModelSource",
    "ModelSourceKind",
    "Momenta",
    "PhysicsReduction",
    "ProcessAlias",
    "ProcessPhysics",
    "ProcessRequest",
    "ProcessSet",
    "PyAmpliColError",
    "ReductionGroup",
    "ResolvedEvaluation",
    "Runtime",
    "RuntimeBackend",
    "RuntimeLoader",
    "WarmUpResult",
    "benchmark",
    "generate",
    "group_processes",
    "install_backend_factories",
    "load",
]

_ERROR_EXPORTS = (
    "ArtifactError",
    "CompatibilityError",
    "ConfigurationError",
    "DependencyError",
    "EvaluationError",
    "GenerationError",
    "ModelError",
    "PyAmpliColError",
)
_PROTOCOL_EXPORTS = (
    "BenchmarkBackend",
    "BenchmarkFactory",
    "GeneratorBackend",
    "GeneratorFactory",
    "ModelParameters",
    "Momenta",
    "RuntimeBackend",
    "RuntimeLoader",
)
_REQUEST_EXPORTS = (
    "ModelSource",
    "ModelSourceKind",
    "ProcessAlias",
    "ProcessRequest",
    "ProcessSet",
)
_MODEL_EXPORTS = (
    "CompiledModel",
    "CompiledModelCapabilities",
    "CompiledModelInfo",
    "CompiledModelSource",
    "ModelCompilationIssue",
    "ModelCompilationPhase",
)
_RESULT_EXPORTS = (
    "BenchmarkComponentTiming",
    "BenchmarkProfileCounters",
    "BenchmarkResult",
    "BenchmarkStageTiming",
    "BenchmarkStatistics",
    "BenchmarkTimingBreakdown",
    "ColorComponent",
    "ColorFlow",
    "ContractedColorComponent",
    "ExternalParticle",
    "GenerationPlan",
    "GenerationResult",
    "HelicityConfiguration",
    "ModelParameter",
    "PhysicsReduction",
    "ProcessPhysics",
    "ReductionGroup",
    "ResolvedEvaluation",
    "WarmUpResult",
)
_SERVICE_EXPORTS = (
    "BenchmarkRunner",
    "Generator",
    "Runtime",
    "benchmark",
    "generate",
    "install_backend_factories",
    "load",
)
_PUBLIC_EXPORTS = {
    **{name: (".errors", name) for name in _ERROR_EXPORTS},
    **{name: (".protocols", name) for name in _PROTOCOL_EXPORTS},
    **{name: (".models", name) for name in _MODEL_EXPORTS},
    **{name: (".requests", name) for name in _REQUEST_EXPORTS},
    **{name: (".results", name) for name in _RESULT_EXPORTS},
    **{name: (".services", name) for name in _SERVICE_EXPORTS},
    "group_processes": (".process_groups", "group_processes"),
}


def __getattr__(name: str) -> Any:
    """Load a public object only when that object is requested."""

    export = _PUBLIC_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = export
    package = __name__ if module_name.startswith(".") else None
    value = getattr(import_module(module_name, package), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
