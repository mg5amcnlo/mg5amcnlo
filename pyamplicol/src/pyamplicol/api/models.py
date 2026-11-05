# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, cast

from .results import ModelParameter

CompiledModelSourceKind: TypeAlias = Literal[
    "built-in-sm",
    "built-in-sm-heft",
    "ufo",
    "json",
    "compiled",
]
ModelCompilationSeverity: TypeAlias = Literal["warning", "error"]
SupportedColorAccuracy: TypeAlias = Literal["lc", "nlc", "full"]


class _CompiledModelPayloadView(Protocol):
    name: str
    schema_version: int
    model_compiler_version: int
    source: Mapping[str, object]
    capabilities: Mapping[str, object]
    parameter_defaults: Mapping[str, object]
    issues: Iterable[object]
    phase_timings: Mapping[str, object]
    conversion_seconds: float
    prepared_backend: str | None


@dataclass(frozen=True, slots=True)
class CompiledModelSource:
    """Stable provenance for the source consumed by the model compiler."""

    kind: CompiledModelSourceKind
    name: str | None = None
    digest: str | None = None
    restriction: str | None = None
    restriction_digest: str | None = None
    simplify: bool = True

    def __post_init__(self) -> None:
        if self.kind not in (
            "built-in-sm",
            "built-in-sm-heft",
            "ufo",
            "json",
            "compiled",
        ):
            raise ValueError(f"unsupported compiled model source kind {self.kind!r}")
        for field_name in ("name", "digest", "restriction", "restriction_digest"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"compiled model source {field_name} must be non-empty or null"
                )
        if self.restriction is not None and self.restriction_digest is not None:
            raise ValueError(
                "compiled model source cannot have both a restriction name and digest"
            )
        if not isinstance(self.simplify, bool):
            raise TypeError("compiled model source simplify must be a boolean")


@dataclass(frozen=True, slots=True)
class CompiledModelCapabilities:
    """User-relevant capabilities of a compiled physics model."""

    particle_count: int = 0
    parameter_count: int = 0
    vertex_count: int = 0
    propagator_count: int = 0
    form_factor_count: int = 0
    maximum_valence: int = 0
    spins: tuple[int, ...] = ()
    color_representations: tuple[int, ...] = ()
    supported_color_accuracies: tuple[SupportedColorAccuracy, ...] = ()
    has_custom_propagators: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "particle_count",
            "parameter_count",
            "vertex_count",
            "propagator_count",
            "form_factor_count",
            "maximum_valence",
        ):
            _validate_non_negative_integer(getattr(self, field_name), field_name)

        spins = tuple(self.spins)
        colors = tuple(self.color_representations)
        accuracies = tuple(self.supported_color_accuracies)
        _validate_integer_tuple(spins, "spins")
        _validate_integer_tuple(colors, "color_representations")
        if len(set(spins)) != len(spins):
            raise ValueError("compiled model spins must be unique")
        if len(set(colors)) != len(colors):
            raise ValueError("compiled model color representations must be unique")
        if any(accuracy not in ("lc", "nlc", "full") for accuracy in accuracies):
            raise ValueError("compiled model has an unsupported color accuracy")
        if len(set(accuracies)) != len(accuracies):
            raise ValueError("compiled model color accuracies must be unique")
        if not isinstance(self.has_custom_propagators, bool):
            raise TypeError("has_custom_propagators must be a boolean")

        object.__setattr__(self, "spins", spins)
        object.__setattr__(self, "color_representations", colors)
        object.__setattr__(self, "supported_color_accuracies", accuracies)

    @property
    def max_vertex_valence(self) -> int:
        """Compatibility spelling for serialized model capability records."""

        return self.maximum_valence

    @property
    def color_accuracy_modes(self) -> tuple[SupportedColorAccuracy, ...]:
        """Compatibility spelling for serialized model capability records."""

        return self.supported_color_accuracies


@dataclass(frozen=True, slots=True)
class ModelCompilationIssue:
    """A stable diagnostic emitted while compiling a model."""

    severity: ModelCompilationSeverity
    code: str
    message: str
    context: str = ""

    def __post_init__(self) -> None:
        if self.severity not in ("warning", "error"):
            raise ValueError(f"unsupported model issue severity {self.severity!r}")
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("model compilation issue code must not be empty")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("model compilation issue message must not be empty")
        if not isinstance(self.context, str):
            raise TypeError("model compilation issue context must be text")


@dataclass(frozen=True, slots=True)
class ModelCompilationPhase:
    """Wall-clock duration of one model-compilation phase."""

    name: str
    seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("model compilation phase name must not be empty")
        object.__setattr__(
            self,
            "seconds",
            _non_negative_finite(self.seconds, "model compilation phase seconds"),
        )


@dataclass(frozen=True, slots=True)
class CompiledModelInfo:
    """Deeply immutable public metadata for one compiled model."""

    name: str
    schema_version: int
    model_compiler_version: int
    source: CompiledModelSource
    capabilities: CompiledModelCapabilities
    parameters: tuple[ModelParameter, ...]
    issues: tuple[ModelCompilationIssue, ...]
    compilation_phases: tuple[ModelCompilationPhase, ...]
    conversion_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("compiled model name must not be empty")
        _validate_non_negative_integer(self.schema_version, "schema_version")
        _validate_non_negative_integer(
            self.model_compiler_version,
            "model_compiler_version",
        )
        if not isinstance(self.source, CompiledModelSource):
            raise TypeError("compiled model source must be CompiledModelSource")
        if not isinstance(self.capabilities, CompiledModelCapabilities):
            raise TypeError(
                "compiled model capabilities must be CompiledModelCapabilities"
            )

        parameters = tuple(self.parameters)
        issues = tuple(self.issues)
        phases = tuple(self.compilation_phases)
        _validate_tuple_members(parameters, ModelParameter, "parameters")
        _validate_tuple_members(issues, ModelCompilationIssue, "issues")
        _validate_tuple_members(phases, ModelCompilationPhase, "compilation_phases")
        if len({parameter.name for parameter in parameters}) != len(parameters):
            raise ValueError("compiled model parameter names must be unique")
        if len({phase.name for phase in phases}) != len(phases):
            raise ValueError("model compilation phase names must be unique")
        object.__setattr__(
            self,
            "conversion_seconds",
            _non_negative_finite(
                self.conversion_seconds,
                "compiled model conversion_seconds",
            ),
        )

        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "compilation_phases", phases)

    @property
    def supported(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def phases(self) -> tuple[ModelCompilationPhase, ...]:
        """Short alias for :attr:`compilation_phases`."""

        return self.compilation_phases


class CompiledModel:
    """Opaque compiled-model handle accepted by :class:`~pyamplicol.Generator`.

    Construct handles with :meth:`pyamplicol.ModelSource.compile`. The compiler's
    mutable payload and expression IR intentionally remain private; stable model
    metadata is available through :attr:`info` and the typed convenience
    properties below.
    """

    __slots__ = ("_info", "_payload")

    _info: CompiledModelInfo
    _payload: object

    def __init__(self) -> None:
        raise TypeError("construct compiled models with ModelSource.compile()")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("CompiledModel handles are immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("CompiledModel handles are immutable")

    @property
    def info(self) -> CompiledModelInfo:
        """Stable, deeply immutable model metadata."""

        return self._info

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def schema_version(self) -> int:
        return self._info.schema_version

    @property
    def model_compiler_version(self) -> int:
        return self._info.model_compiler_version

    @property
    def source(self) -> CompiledModelSource:
        return self._info.source

    @property
    def capabilities(self) -> CompiledModelCapabilities:
        return self._info.capabilities

    @property
    def parameters(self) -> tuple[ModelParameter, ...]:
        return self._info.parameters

    @property
    def issues(self) -> tuple[ModelCompilationIssue, ...]:
        return self._info.issues

    @property
    def compilation_phases(self) -> tuple[ModelCompilationPhase, ...]:
        return self._info.compilation_phases

    @property
    def conversion_seconds(self) -> float:
        return self._info.conversion_seconds

    @property
    def supported(self) -> bool:
        return self._info.supported

    @property
    def is_prepared(self) -> bool:
        """Whether this handle owns a validated prepared-mode kernel pack.

        The pack is shared by eager, recurrence, and on-the-fly execution.
        """

        return self.prepared_backend is not None

    @property
    def prepared_backend(self) -> Literal["jit", "asm", "cpp"] | None:
        """Backend embedded in the prepared model, if any."""

        backend = getattr(self._payload, "prepared_backend", None)
        if backend is None:
            return None
        if backend not in ("jit", "asm", "cpp"):
            raise RuntimeError(f"invalid prepared-model backend {backend!r}")
        return cast(Literal["jit", "asm", "cpp"], backend)

    def write(self, path: os.PathLike[str] | str) -> Path:
        """Serialize the compiled model and return its absolute output path."""

        writer = getattr(self._payload, "write", None)
        if not callable(writer):
            raise RuntimeError("compiled model payload cannot be serialized")
        target = Path(os.fspath(path)).expanduser().resolve(strict=False)
        return Path(os.fspath(writer(target))).resolve(strict=False)

    def write_parameter_card(self, path: os.PathLike[str] | str) -> Path:
        """Write mutable external-parameter defaults as a JSON model card."""

        writer = getattr(self._payload, "write_parameter_card", None)
        if not callable(writer):
            raise RuntimeError("compiled model payload cannot write parameter cards")
        target = Path(os.fspath(path)).expanduser().resolve(strict=False)
        return Path(os.fspath(writer(target))).resolve(strict=False)

    def __repr__(self) -> str:
        return (
            f"CompiledModel(name={self.name!r}, supported={self.supported!r}, "
            f"schema_version={self.schema_version})"
        )


def _compiled_model_from_payload(payload: object) -> CompiledModel:
    """Create an opaque public handle from the private compiler payload."""

    view = cast(_CompiledModelPayloadView, payload)
    try:
        info = _compiled_model_info(
            name=str(view.name),
            schema_version=int(view.schema_version),
            model_compiler_version=int(view.model_compiler_version),
            source=view.source,
            capabilities=view.capabilities,
            parameter_defaults=view.parameter_defaults,
            issues=view.issues,
            phase_timings=view.phase_timings,
            conversion_seconds=float(view.conversion_seconds),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("invalid private compiled-model payload") from exc
    model = object.__new__(CompiledModel)
    object.__setattr__(model, "_payload", payload)
    object.__setattr__(model, "_info", info)
    return model


def _compiled_model_payload(model: CompiledModel) -> object:
    """Return the private payload at an internal generation boundary."""

    if not isinstance(model, CompiledModel):
        raise TypeError("expected a CompiledModel handle")
    return model._payload


def _compiled_model_info(
    *,
    name: str,
    schema_version: int,
    model_compiler_version: int,
    source: Mapping[str, object],
    capabilities: Mapping[str, object],
    parameter_defaults: Mapping[str, object],
    issues: Iterable[object],
    phase_timings: Mapping[str, object],
    conversion_seconds: float,
) -> CompiledModelInfo:
    """Convert compiler-owned records without importing the compiler module."""

    return CompiledModelInfo(
        name=name,
        schema_version=schema_version,
        model_compiler_version=model_compiler_version,
        source=_compiled_model_source(source),
        capabilities=_compiled_model_capabilities(capabilities),
        parameters=_compiled_model_parameters(parameter_defaults),
        issues=tuple(_model_compilation_issue(issue) for issue in issues),
        compilation_phases=_model_compilation_phases(phase_timings),
        conversion_seconds=conversion_seconds,
    )


def _compiled_model_source(source: Mapping[str, object]) -> CompiledModelSource:
    raw_kind = source.get("kind")
    if not isinstance(raw_kind, str) or raw_kind not in (
        "built-in-sm",
        "built-in-sm-heft",
        "ufo",
        "json",
        "compiled",
    ):
        raise ValueError(f"unsupported compiled model source kind {raw_kind!r}")

    raw_name = source.get("source_name", source.get("name"))
    name = _optional_non_empty_string(raw_name, "compiled model source name")
    digest = _optional_non_empty_string(
        source.get("digest"),
        "compiled model source digest",
    )

    options = _optional_mapping(source.get("options"), "compiled model source options")
    simplify = _boolean(
        source.get("simplify", options.get("simplify", True)),
        "compiled model source simplify",
    )
    restriction: str | None = None
    restriction_digest: str | None = None
    raw_restriction = source.get("restriction", options.get("restriction"))
    if isinstance(raw_restriction, str):
        restriction = _optional_non_empty_string(
            raw_restriction,
            "compiled model source restriction",
        )
    elif isinstance(raw_restriction, Mapping):
        restriction_kind = raw_restriction.get("kind")
        if restriction_kind == "name":
            restriction = _optional_non_empty_string(
                raw_restriction.get("value"),
                "compiled model source restriction",
            )
        elif restriction_kind == "file":
            restriction_digest = _optional_non_empty_string(
                raw_restriction.get("sha256"),
                "compiled model restriction digest",
            )

    return CompiledModelSource(
        kind=cast(CompiledModelSourceKind, raw_kind),
        name=name,
        digest=digest,
        restriction=restriction,
        restriction_digest=restriction_digest,
        simplify=simplify,
    )


def _compiled_model_capabilities(
    capabilities: Mapping[str, object],
) -> CompiledModelCapabilities:
    return CompiledModelCapabilities(
        particle_count=_mapping_non_negative_integer(
            capabilities,
            "particle_count",
        ),
        parameter_count=_mapping_non_negative_integer(
            capabilities,
            "parameter_count",
        ),
        vertex_count=_mapping_non_negative_integer(capabilities, "vertex_count"),
        propagator_count=_mapping_non_negative_integer(
            capabilities,
            "propagator_count",
            fallback="compiled_propagator_count",
        ),
        form_factor_count=_mapping_non_negative_integer(
            capabilities,
            "form_factor_count",
        ),
        maximum_valence=_mapping_non_negative_integer(
            capabilities,
            "maximum_valence",
            fallback="max_vertex_valence",
        ),
        spins=_integer_values(capabilities.get("spins", ()), "spins"),
        color_representations=_integer_values(
            capabilities.get("color_representations", ()),
            "color_representations",
        ),
        supported_color_accuracies=_color_accuracies(
            capabilities.get(
                "supported_color_accuracies",
                capabilities.get("color_accuracy_modes", ()),
            )
        ),
        has_custom_propagators=_boolean(
            capabilities.get("has_custom_propagators", False),
            "has_custom_propagators",
        ),
    )


def _compiled_model_parameters(
    parameter_defaults: Mapping[str, object],
) -> tuple[ModelParameter, ...]:
    parameters = []
    for raw_name, value in parameter_defaults.items():
        name = str(raw_name)
        real, imaginary = _complex_pair(value, f"model parameter {name!r}")
        parameters.append(
            ModelParameter(
                name=name,
                kind="external",
                default_real=real,
                default_imaginary=imaginary,
                mutable=True,
            )
        )
    return tuple(sorted(parameters, key=lambda parameter: parameter.name))


def _model_compilation_issue(issue: object) -> ModelCompilationIssue:
    if isinstance(issue, ModelCompilationIssue):
        return issue
    raw_severity = _record_value(issue, "severity")
    if not isinstance(raw_severity, str) or raw_severity not in ("warning", "error"):
        raise ValueError(f"unsupported model issue severity {raw_severity!r}")
    code = _record_text(issue, "code")
    message = _record_text(issue, "message")
    raw_context = _record_value(issue, "context", default="")
    context = "" if raw_context is None else raw_context
    if not isinstance(context, str):
        raise TypeError("model compilation issue context must be text")
    return ModelCompilationIssue(
        severity=cast(ModelCompilationSeverity, raw_severity),
        code=code,
        message=message,
        context=context,
    )


def _model_compilation_phases(
    phase_timings: Mapping[str, object],
) -> tuple[ModelCompilationPhase, ...]:
    return tuple(
        ModelCompilationPhase(
            name=str(name),
            seconds=_finite_float(value, f"model compilation phase {name!r}"),
        )
        for name, value in phase_timings.items()
    )


_MISSING = object()


def _record_value(
    record: object,
    field_name: str,
    *,
    default: object = _MISSING,
) -> object:
    if isinstance(record, Mapping):
        if field_name in record:
            return record[field_name]
    elif hasattr(record, field_name):
        return getattr(record, field_name)
    if default is not _MISSING:
        return default
    raise TypeError(f"model compilation issue has no {field_name!r} field")


def _record_text(record: object, field_name: str) -> str:
    value = _record_value(record, field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model compilation issue {field_name} must not be empty")
    return value


def _optional_mapping(value: object, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _optional_non_empty_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be non-empty or null")
    return value


def _mapping_non_negative_integer(
    mapping: Mapping[str, object],
    field_name: str,
    *,
    fallback: str | None = None,
) -> int:
    value = mapping.get(field_name, _MISSING)
    if value is _MISSING and fallback is not None:
        value = mapping.get(fallback, _MISSING)
    if value is _MISSING:
        value = 0
    _validate_non_negative_integer(value, field_name)
    return cast(int, value)


def _validate_non_negative_integer(value: object, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")


def _validate_integer_tuple(values: tuple[object, ...], context: str) -> None:
    if not all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    ):
        raise ValueError(f"{context} must contain integers")


def _integer_values(value: object, context: str) -> tuple[int, ...]:
    values = _sequence(value, context)
    _validate_integer_tuple(values, context)
    return cast(tuple[int, ...], values)


def _color_accuracies(value: object) -> tuple[SupportedColorAccuracy, ...]:
    values = _sequence(value, "supported color accuracies")
    if not all(
        isinstance(accuracy, str) and accuracy in ("lc", "nlc", "full")
        for accuracy in values
    ):
        raise ValueError("compiled model has an unsupported color accuracy")
    return cast(tuple[SupportedColorAccuracy, ...], values)


def _sequence(value: object, context: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be a sequence")
    return tuple(value)


def _complex_pair(value: object, context: str) -> tuple[float, float]:
    if isinstance(value, complex):
        return float(value.real), float(value.imag)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 2:
            raise ValueError(f"{context} must contain a real and imaginary value")
        return (
            _finite_float(value[0], f"{context} real component"),
            _finite_float(value[1], f"{context} imaginary component"),
        )
    return _finite_float(value, context), 0.0


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be a boolean")
    return value


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise TypeError(f"{context} must be a finite number")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{context} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _non_negative_finite(value: object, context: str) -> float:
    result = _finite_float(value, context)
    if result < 0.0:
        raise ValueError(f"{context} must be non-negative")
    return result


def _validate_tuple_members(
    values: tuple[object, ...],
    member_type: type[object],
    context: str,
) -> None:
    if not all(isinstance(value, member_type) for value in values):
        raise TypeError(f"compiled model {context} contain invalid records")


__all__ = [
    "CompiledModel",
    "CompiledModelCapabilities",
    "CompiledModelInfo",
    "CompiledModelSource",
    "ModelCompilationIssue",
    "ModelCompilationPhase",
]
