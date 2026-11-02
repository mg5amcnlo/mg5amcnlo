# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAlias, TypeVar

from .errors import ConfigurationError


class Action(StrEnum):
    GENERATE = "generate"
    EVALUATE = "evaluate"
    BENCHMARK = "benchmark"
    INSPECT = "inspect"
    MODEL_INSPECT = "model-inspect"
    MODEL_COMPILE = "model-compile"
    MODEL_PROCESSES = "model-processes"
    GROUP_PROCESSES = "group-processes"


class CouplingOrderPolicy(StrEnum):
    MINIMAL = "minimal"
    EXPLICIT = "explicit"


class ProcessGroupAnalysis(StrEnum):
    NONE = "none"
    NUMERICAL = "numerical"


class ColorAccuracy(StrEnum):
    LC = "lc"
    NLC = "nlc"
    FULL = "full"


class ColorContraction(StrEnum):
    AUTO = "auto"
    DIRECT = "direct"
    SYMMETRIC_GROUP_FFT = "symmetric-group-fft"


class ColorFFTBasis(StrEnum):
    TRACE = "trace"
    ADJOINT = "adjoint"


class LCFlowLayout(StrEnum):
    TOPOLOGY_REPLAY = "topology-replay"
    ALL_FLOW_UNION = "all-flow-union"


class GenerationMode(StrEnum):
    ERROR = "error"
    APPEND = "append"
    REPLACE = "replace"


class RelationDiscoveryMode(StrEnum):
    OFF = "off"
    DIAGNOSTIC = "diagnostic"
    CERTIFIED_REUSE = "certified-reuse"


class EvaluatorBackend(StrEnum):
    JIT = "jit"
    ASM = "asm"
    CPP = "cpp"


class EvaluatorExecutionMode(StrEnum):
    COMPILED = "compiled"
    EAGER = "eager"
    RECURRENCE = "recurrence"
    ON_THE_FLY = "on-the-fly"


_PORTABLE_CPP_EXTRA_FLAGS = frozenset({"-fno-math-errno"})
_PROCESS_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


class OutputFormat(StrEnum):
    HUMAN = "human"
    JSON = "json"


class ColorMode(StrEnum):
    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"


class ProgressMode(StrEnum):
    AUTO = "auto"
    TTY = "tty"
    LOG = "log"
    OFF = "off"


class LogLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


AutoInt: TypeAlias = Literal["auto"] | int
AutoBool: TypeAlias = Literal["auto"] | bool

ACTIONS: tuple[Action, ...] = tuple(Action)

_T = TypeVar("_T")
_EnumT = TypeVar("_EnumT", bound=StrEnum)


def _setting(
    kind: str,
    *,
    nullable: bool = False,
    choices: Sequence[object] = (),
    dynamic_kind: str | None = None,
) -> dict[str, object]:
    return {
        "config": {
            "kind": kind,
            "nullable": nullable,
            "choices": tuple(choices),
            "dynamic_kind": dynamic_kind,
        }
    }


def _section() -> dict[str, object]:
    return {"config_section": True}


def _path(value: os.PathLike[str] | str | None, name: str) -> Path | None:
    if value is None:
        return None
    try:
        return Path(os.fspath(value)).expanduser().resolve(strict=False)
    except TypeError as exc:
        raise ConfigurationError(f"{name} must be a path or null") from exc


def _tuple_of_strings(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ConfigurationError(f"{name} must be a list of strings")
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        raise ConfigurationError(f"{name} must contain non-empty strings")
    return result


def _tuple_of_ints(value: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ConfigurationError(f"{name} must be a list of integers")
    result = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise ConfigurationError(f"{name} must contain integers")
    return result


def _string_tuple_mapping(
    value: Mapping[str, Sequence[str]], name: str
) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a table")
    result: dict[str, tuple[str, ...]] = {}
    for key, entries in value.items():
        if not isinstance(key, str) or not key:
            raise ConfigurationError(f"{name} keys must be non-empty strings")
        result[key] = _tuple_of_strings(entries, f"{name}.{key}")
    return MappingProxyType(result)


def _integer_mapping(value: Mapping[str, int], name: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a table")
    result: dict[str, int] = {}
    for key, entry in value.items():
        if not isinstance(key, str) or not key:
            raise ConfigurationError(f"{name} keys must be non-empty strings")
        if isinstance(entry, bool) or not isinstance(entry, int) or entry < 0:
            raise ConfigurationError(f"{name}.{key} must be a non-negative integer")
        result[key] = entry
    return MappingProxyType(result)


def _signed_integer_mapping(value: Mapping[str, int], name: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a table")
    result: dict[str, int] = {}
    for key, entry in value.items():
        if not isinstance(key, str) or not key:
            raise ConfigurationError(f"{name} keys must be non-empty strings")
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ConfigurationError(f"{name}.{key} must be an integer")
        result[key] = entry
    return MappingProxyType(result)


def _choice(value: _T, choices: Sequence[_T], name: str) -> _T:
    if value not in choices:
        allowed = ", ".join(repr(item) for item in choices)
        raise ConfigurationError(f"{name} must be one of {allowed}; got {value!r}")
    return value


def _enum(value: object, enum_type: type[_EnumT], name: str) -> _EnumT:
    try:
        if not isinstance(value, str):
            raise TypeError
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(repr(item.value) for item in enum_type)
        raise ConfigurationError(
            f"{name} must be one of {allowed}; got {value!r}"
        ) from exc


def _integer(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ConfigurationError(f"{name} must be an integer {qualifier}")
    return value


def _optional_integer(value: int | None, name: str, *, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _integer(value, name, minimum=minimum)


def _finite_float(
    value: float, name: str, *, minimum: float, exclusive: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a number")
    result = float(value)
    invalid = result <= minimum if exclusive else result < minimum
    if not math.isfinite(result) or invalid:
        operator = ">" if exclusive else ">="
        raise ConfigurationError(f"{name} must be finite and {operator} {minimum}")
    return result


def _auto_integer(value: AutoInt, name: str) -> AutoInt:
    if value == "auto":
        return value
    return _integer(value, name)


def _auto_bool(value: AutoBool, name: str) -> AutoBool:
    if value == "auto" or isinstance(value, bool):
        return value
    raise ConfigurationError(f"{name} must be 'auto', true, or false")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    source: str = field(default="built-in-sm", metadata=_setting("str"))
    restriction: str | None = field(
        default=None, metadata=_setting("str", nullable=True)
    )
    simplify: bool = field(default=True, metadata=_setting("bool"))
    cache: bool = field(default=True, metadata=_setting("bool"))
    cache_dir: Path | None = field(
        default=None, metadata=_setting("path", nullable=True)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ConfigurationError("model.source must be a non-empty string")
        if self.restriction is not None and not isinstance(self.restriction, str):
            raise ConfigurationError("model.restriction must be a string or null")
        if not isinstance(self.simplify, bool):
            raise ConfigurationError("model.simplify must be a boolean")
        if not isinstance(self.cache, bool):
            raise ConfigurationError("model.cache must be a boolean")
        object.__setattr__(self, "cache_dir", _path(self.cache_dir, "model.cache_dir"))


@dataclass(frozen=True, slots=True)
class ProcessEntry:
    expression: str
    name: str | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.expression, str) or not self.expression.strip():
            errors.append("process entry expression must be a non-empty string")
        elif "\n" in self.expression or "\r" in self.expression:
            errors.append("process entry expression must contain exactly one line")
        if self.name is not None and (
            not isinstance(self.name, str) or not _PROCESS_NAME.fullmatch(self.name)
        ):
            errors.append(
                "process entry name must start with a letter and contain only "
                "letters, digits, '.', '_', or '-'"
            )
        if errors:
            raise ConfigurationError(errors)


@dataclass(frozen=True, slots=True)
class ProcessConfig:
    entries: tuple[ProcessEntry, ...] = field(
        default=(), metadata=_setting("process_entries")
    )
    multiparticles: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict,
        metadata=_setting("map_list_str", dynamic_kind="list_str"),
    )
    flavor_scheme: int = field(default=5, metadata=_setting("int"))
    max_quark_lines: int | None = field(
        default=None, metadata=_setting("int", nullable=True)
    )
    coupling_order_policy: CouplingOrderPolicy = field(
        default=CouplingOrderPolicy.MINIMAL,
        metadata=_setting("str", choices=tuple(CouplingOrderPolicy)),
    )
    max_coupling_orders: Mapping[str, int] = field(
        default_factory=dict,
        metadata=_setting("map_int", dynamic_kind="int"),
    )
    max_color_sectors: int | None = field(
        default=None, metadata=_setting("int", nullable=True)
    )
    reference_color_order: tuple[int, ...] = field(
        default=(), metadata=_setting("list_int")
    )
    selected_color_sector_ids: tuple[int, ...] = field(
        default=(), metadata=_setting("list_int")
    )
    selected_source_helicities: Mapping[str, int] = field(
        default_factory=dict,
        metadata=_setting("map_int", dynamic_kind="int"),
    )
    flavor_blind: bool = field(default=False, metadata=_setting("bool"))
    consider_charge_conjugation: bool = field(default=True, metadata=_setting("bool"))
    only_unit_multiplier: bool = field(default=False, metadata=_setting("bool"))
    analyze_process_groups: ProcessGroupAnalysis = field(
        default=ProcessGroupAnalysis.NONE,
        metadata=_setting("str", choices=tuple(ProcessGroupAnalysis)),
    )

    def __post_init__(self) -> None:
        if isinstance(self.entries, (str, bytes)):
            raise ConfigurationError(
                "process.entries must be a list of process entries"
            )
        entries = tuple(self.entries)
        if not all(isinstance(entry, ProcessEntry) for entry in entries):
            raise ConfigurationError(
                "process.entries must contain ProcessEntry objects"
            )
        names = tuple(entry.name for entry in entries if entry.name is not None)
        duplicate_names = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicate_names:
            raise ConfigurationError(
                "process entry names must be unique; duplicates: "
                + ", ".join(repr(name) for name in duplicate_names)
            )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self,
            "multiparticles",
            _string_tuple_mapping(self.multiparticles, "process.multiparticles"),
        )
        object.__setattr__(
            self,
            "flavor_scheme",
            _integer(self.flavor_scheme, "process.flavor_scheme", minimum=0),
        )
        object.__setattr__(
            self,
            "max_quark_lines",
            _optional_integer(
                self.max_quark_lines, "process.max_quark_lines", minimum=0
            ),
        )
        object.__setattr__(
            self,
            "coupling_order_policy",
            _enum(
                self.coupling_order_policy,
                CouplingOrderPolicy,
                "process.coupling_order_policy",
            ),
        )
        object.__setattr__(
            self,
            "max_coupling_orders",
            _integer_mapping(self.max_coupling_orders, "process.max_coupling_orders"),
        )
        object.__setattr__(
            self,
            "max_color_sectors",
            _optional_integer(
                self.max_color_sectors, "process.max_color_sectors", minimum=0
            ),
        )
        object.__setattr__(
            self,
            "reference_color_order",
            _tuple_of_ints(self.reference_color_order, "process.reference_color_order"),
        )
        object.__setattr__(
            self,
            "selected_color_sector_ids",
            _tuple_of_ints(
                self.selected_color_sector_ids,
                "process.selected_color_sector_ids",
            ),
        )
        object.__setattr__(
            self,
            "selected_source_helicities",
            _signed_integer_mapping(
                self.selected_source_helicities,
                "process.selected_source_helicities",
            ),
        )
        for name in (
            "flavor_blind",
            "consider_charge_conjugation",
            "only_unit_multiplier",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ConfigurationError(f"process.{name} must be a boolean")
        object.__setattr__(
            self,
            "analyze_process_groups",
            _enum(
                self.analyze_process_groups,
                ProcessGroupAnalysis,
                "process.analyze_process_groups",
            ),
        )


@dataclass(frozen=True, slots=True)
class ColorConfig:
    accuracy: ColorAccuracy = field(
        default=ColorAccuracy.LC,
        metadata=_setting("str", choices=tuple(ColorAccuracy)),
    )
    contraction: ColorContraction = field(
        default=ColorContraction.AUTO,
        metadata=_setting("str", choices=tuple(ColorContraction)),
    )
    lc_flow_layout: LCFlowLayout = field(
        default=LCFlowLayout.TOPOLOGY_REPLAY,
        metadata=_setting("str", choices=tuple(LCFlowLayout)),
    )
    fft_basis: ColorFFTBasis = field(
        default=ColorFFTBasis.ADJOINT,
        metadata=_setting("str", choices=tuple(ColorFFTBasis)),
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "accuracy", _enum(self.accuracy, ColorAccuracy, "color.accuracy")
        )
        object.__setattr__(
            self,
            "contraction",
            _enum(
                self.contraction,
                ColorContraction,
                "color.contraction",
            ),
        )
        object.__setattr__(
            self,
            "fft_basis",
            _enum(self.fft_basis, ColorFFTBasis, "color.fft_basis"),
        )
        object.__setattr__(
            self,
            "lc_flow_layout",
            _enum(self.lc_flow_layout, LCFlowLayout, "color.lc_flow_layout"),
        )
        if (
            self.lc_flow_layout is LCFlowLayout.ALL_FLOW_UNION
            and self.accuracy is not ColorAccuracy.LC
        ):
            raise ConfigurationError(
                "color.lc_flow_layout='all-flow-union' requires color.accuracy='lc'"
            )
        if (
            self.contraction is ColorContraction.SYMMETRIC_GROUP_FFT
            and self.accuracy is ColorAccuracy.LC
        ):
            raise ConfigurationError(
                "color.contraction='symmetric-group-fft' requires "
                "color.accuracy='nlc' or 'full'"
            )


@dataclass(frozen=True, slots=True)
class GenerationValidationConfig:
    enabled: bool = field(default=True, metadata=_setting("bool"))
    samples: int = field(default=2, metadata=_setting("int"))
    seed: int = field(default=12345, metadata=_setting("int"))
    relative_tolerance: float = field(default=1e-12, metadata=_setting("float"))
    absolute_tolerance: float = field(default=1e-300, metadata=_setting("float"))
    post_build_validation: bool = field(default=False, metadata=_setting("bool"))

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "post_build_validation",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ConfigurationError(
                    f"generation.validation.{name} must be a boolean"
                )
        object.__setattr__(
            self,
            "samples",
            _integer(self.samples, "generation.validation.samples"),
        )
        object.__setattr__(
            self,
            "seed",
            _integer(self.seed, "generation.validation.seed", minimum=0),
        )
        object.__setattr__(
            self,
            "relative_tolerance",
            _finite_float(
                self.relative_tolerance,
                "generation.validation.relative_tolerance",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "absolute_tolerance",
            _finite_float(
                self.absolute_tolerance,
                "generation.validation.absolute_tolerance",
                minimum=0.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class GenerationRelationDiscoveryConfig:
    """Opt-in, replayable current-relation discovery policy (off by default)."""

    mode: RelationDiscoveryMode = field(
        default=RelationDiscoveryMode.OFF,
        metadata=_setting("str", choices=tuple(RelationDiscoveryMode)),
    )
    precision_digits: int = field(default=96, metadata=_setting("int"))
    probe_count: int = field(default=4, metadata=_setting("int"))
    verification_probe_count: int = field(default=4, metadata=_setting("int"))
    relative_tolerance: float = field(default=1.0e-70, metadata=_setting("float"))
    absolute_tolerance: float = field(default=1.0e-80, metadata=_setting("float"))
    seed: int = field(default=0x5059414D, metadata=_setting("int"))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _enum(
                self.mode,
                RelationDiscoveryMode,
                "generation.relation_discovery.mode",
            ),
        )
        object.__setattr__(
            self,
            "precision_digits",
            _integer(
                self.precision_digits,
                "generation.relation_discovery.precision_digits",
                minimum=80,
            ),
        )
        object.__setattr__(
            self,
            "probe_count",
            _integer(
                self.probe_count,
                "generation.relation_discovery.probe_count",
                minimum=2,
            ),
        )
        object.__setattr__(
            self,
            "verification_probe_count",
            _integer(
                self.verification_probe_count,
                "generation.relation_discovery.verification_probe_count",
                minimum=2,
            ),
        )
        object.__setattr__(
            self,
            "relative_tolerance",
            _finite_float(
                self.relative_tolerance,
                "generation.relation_discovery.relative_tolerance",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "absolute_tolerance",
            _finite_float(
                self.absolute_tolerance,
                "generation.relation_discovery.absolute_tolerance",
                minimum=0.0,
            ),
        )
        if any(
            value == 0.0 and math.copysign(1.0, value) < 0.0
            for value in (self.relative_tolerance, self.absolute_tolerance)
        ):
            raise ConfigurationError(
                "generation.relation_discovery tolerances cannot use negative zero"
            )
        if self.relative_tolerance == 0.0 and self.absolute_tolerance == 0.0:
            raise ConfigurationError(
                "generation.relation_discovery tolerances cannot both be zero"
            )
        object.__setattr__(
            self,
            "seed",
            _integer(
                self.seed,
                "generation.relation_discovery.seed",
                minimum=0,
            ),
        )


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    output: Path | None = field(default=None, metadata=_setting("path", nullable=True))
    mode: GenerationMode = field(
        default=GenerationMode.ERROR,
        metadata=_setting("str", choices=tuple(GenerationMode)),
    )
    workers: AutoInt = field(default="auto", metadata=_setting("auto_int"))
    emit_api_bundle: bool = field(default=True, metadata=_setting("bool"))
    validation: GenerationValidationConfig = field(
        default_factory=GenerationValidationConfig, metadata=_section()
    )
    relation_discovery: GenerationRelationDiscoveryConfig = field(
        default_factory=GenerationRelationDiscoveryConfig,
        metadata=_section(),
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", _path(self.output, "generation.output"))
        object.__setattr__(
            self, "mode", _enum(self.mode, GenerationMode, "generation.mode")
        )
        object.__setattr__(
            self, "workers", _auto_integer(self.workers, "generation.workers")
        )
        if not isinstance(self.emit_api_bundle, bool):
            raise ConfigurationError("generation.emit_api_bundle must be a boolean")
        if not isinstance(self.validation, GenerationValidationConfig):
            raise ConfigurationError(
                "generation.validation must be a GenerationValidationConfig"
            )
        if not isinstance(
            self.relation_discovery,
            GenerationRelationDiscoveryConfig,
        ):
            raise ConfigurationError(
                "generation.relation_discovery must be a "
                "GenerationRelationDiscoveryConfig"
            )


@dataclass(frozen=True, slots=True)
class EvaluatorOptimizationConfig:
    horner_iterations: int = field(default=10, metadata=_setting("int"))
    cpe_iterations: int | None = field(
        default=None, metadata=_setting("int", nullable=True)
    )
    cores: AutoInt = field(default="auto", metadata=_setting("auto_int"))
    max_horner_variables: int = field(default=1000, metadata=_setting("int"))
    max_common_pair_cache_entries: int = field(
        default=5_000_000, metadata=_setting("int")
    )
    max_common_pair_distance: int = field(default=1000, metadata=_setting("int"))
    collect_factors: AutoBool = field(default="auto", metadata=_setting("auto_bool"))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "horner_iterations",
            _integer(
                self.horner_iterations,
                "evaluator.optimization.horner_iterations",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "cpe_iterations",
            _optional_integer(
                self.cpe_iterations,
                "evaluator.optimization.cpe_iterations",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "cores",
            _auto_integer(self.cores, "evaluator.optimization.cores"),
        )
        for name in (
            "max_horner_variables",
            "max_common_pair_cache_entries",
            "max_common_pair_distance",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    f"evaluator.optimization.{name}",
                    minimum=0,
                ),
            )
        object.__setattr__(
            self,
            "collect_factors",
            _auto_bool(self.collect_factors, "evaluator.optimization.collect_factors"),
        )


@dataclass(frozen=True, slots=True)
class JITConfig:
    optimization_level: Literal[0, 1, 2, 3] = field(
        default=2, metadata=_setting("int", choices=(0, 1, 2, 3))
    )
    compress: AutoBool = field(default="auto", metadata=_setting("auto_bool"))

    def __post_init__(self) -> None:
        if isinstance(self.optimization_level, bool) or not isinstance(
            self.optimization_level, int
        ):
            raise ConfigurationError(
                "evaluator.jit.optimization_level must be an integer"
            )
        _choice(
            self.optimization_level,
            (0, 1, 2, 3),
            "evaluator.jit.optimization_level",
        )
        object.__setattr__(
            self,
            "compress",
            _auto_bool(self.compress, "evaluator.jit.compress"),
        )


@dataclass(frozen=True, slots=True)
class CppConfig:
    optimization: str = field(default="O3", metadata=_setting("str"))
    compiler: str | None = field(default=None, metadata=_setting("str", nullable=True))
    native_arch: bool = field(default=False, metadata=_setting("bool"))
    extra_flags: tuple[str, ...] = field(default=(), metadata=_setting("list_str"))

    def __post_init__(self) -> None:
        if not isinstance(self.optimization, str) or not self.optimization:
            raise ConfigurationError(
                "evaluator.cpp.optimization must be a non-empty string"
            )
        if self.compiler is not None and not isinstance(self.compiler, str):
            raise ConfigurationError("evaluator.cpp.compiler must be a string or null")
        if not isinstance(self.native_arch, bool):
            raise ConfigurationError("evaluator.cpp.native_arch must be a boolean")
        extra_flags = _tuple_of_strings(self.extra_flags, "evaluator.cpp.extra_flags")
        unsupported = tuple(
            flag for flag in extra_flags if flag not in _PORTABLE_CPP_EXTRA_FLAGS
        )
        if unsupported:
            rendered = ", ".join(repr(flag) for flag in unsupported)
            allowed = ", ".join(sorted(_PORTABLE_CPP_EXTRA_FLAGS))
            raise ConfigurationError(
                "evaluator.cpp.extra_flags contains unsupported compiler arguments "
                f"({rendered}); arbitrary flags may introduce unrecorded target CPU "
                f"requirements. Allowed portable flags: {allowed}"
            )
        object.__setattr__(self, "extra_flags", extra_flags)


@dataclass(frozen=True, slots=True)
class EagerEvaluatorConfig:
    point_tile_size: int = field(default=1024, metadata=_setting("int"))
    # Soft batching target, not a cap on the workspace needed by one point.
    workspace_mib: int = field(default=256, metadata=_setting("int"))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "point_tile_size",
            _integer(
                self.point_tile_size,
                "evaluator.eager.point_tile_size",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "workspace_mib",
            _integer(
                self.workspace_mib,
                "evaluator.eager.workspace_mib",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class RecurrenceEvaluatorConfig:
    point_tile_size: int = field(default=1024, metadata=_setting("int"))
    # Soft batching target, not a cap on the workspace needed by one point.
    workspace_mib: int = field(default=256, metadata=_setting("int"))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "point_tile_size",
            _integer(
                self.point_tile_size,
                "evaluator.recurrence.point_tile_size",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "workspace_mib",
            _integer(
                self.workspace_mib,
                "evaluator.recurrence.workspace_mib",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class EvaluatorConfig:
    backend: EvaluatorBackend = field(
        default=EvaluatorBackend.JIT,
        metadata=_setting("str", choices=tuple(EvaluatorBackend)),
    )
    execution_mode: EvaluatorExecutionMode = field(
        default=EvaluatorExecutionMode.RECURRENCE,
        metadata=_setting("str", choices=tuple(EvaluatorExecutionMode)),
    )
    batch_size: int = field(default=128, metadata=_setting("int"))
    output_chunk_size: int | None = field(
        default=512, metadata=_setting("int", nullable=True)
    )
    optimization: EvaluatorOptimizationConfig = field(
        default_factory=EvaluatorOptimizationConfig, metadata=_section()
    )
    jit: JITConfig = field(default_factory=JITConfig, metadata=_section())
    cpp: CppConfig = field(default_factory=CppConfig, metadata=_section())
    eager: EagerEvaluatorConfig = field(
        default_factory=EagerEvaluatorConfig, metadata=_section()
    )
    recurrence: RecurrenceEvaluatorConfig = field(
        default_factory=RecurrenceEvaluatorConfig, metadata=_section()
    )

    @property
    def resolved_jit_compress(self) -> bool:
        """Use compressed code for compiled O2, preserving explicit choices."""

        if self.jit.compress != "auto":
            return self.jit.compress
        return (
            self.backend is EvaluatorBackend.JIT
            and self.execution_mode is EvaluatorExecutionMode.COMPILED
            and self.jit.optimization_level == 2
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend",
            _enum(self.backend, EvaluatorBackend, "evaluator.backend"),
        )
        object.__setattr__(
            self,
            "execution_mode",
            _enum(
                self.execution_mode,
                EvaluatorExecutionMode,
                "evaluator.execution_mode",
            ),
        )
        object.__setattr__(
            self, "batch_size", _integer(self.batch_size, "evaluator.batch_size")
        )
        object.__setattr__(
            self,
            "output_chunk_size",
            _optional_integer(
                self.output_chunk_size, "evaluator.output_chunk_size", minimum=1
            ),
        )
        if not isinstance(self.optimization, EvaluatorOptimizationConfig):
            raise ConfigurationError(
                "evaluator.optimization must be an EvaluatorOptimizationConfig"
            )
        if not isinstance(self.jit, JITConfig):
            raise ConfigurationError("evaluator.jit must be a JITConfig")
        if not isinstance(self.cpp, CppConfig):
            raise ConfigurationError("evaluator.cpp must be a CppConfig")
        if not isinstance(self.eager, EagerEvaluatorConfig):
            raise ConfigurationError("evaluator.eager must be an EagerEvaluatorConfig")
        if not isinstance(self.recurrence, RecurrenceEvaluatorConfig):
            raise ConfigurationError(
                "evaluator.recurrence must be a RecurrenceEvaluatorConfig"
            )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    artifact: Path | None = field(
        default=None, metadata=_setting("path", nullable=True)
    )
    process: str | None = field(default=None, metadata=_setting("str", nullable=True))
    precision: int = field(default=16, metadata=_setting("int"))
    resolved: bool = field(default=False, metadata=_setting("bool"))
    helicity_ids: tuple[str, ...] = field(default=(), metadata=_setting("list_str"))
    color_flow_ids: tuple[str, ...] = field(default=(), metadata=_setting("list_str"))
    model_parameters: Path | None = field(
        default=None, metadata=_setting("path", nullable=True)
    )
    momenta: Path | None = field(default=None, metadata=_setting("path", nullable=True))

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact", _path(self.artifact, "evaluation.artifact")
        )
        if self.process is not None and not isinstance(self.process, str):
            raise ConfigurationError("evaluation.process must be a string or null")
        object.__setattr__(
            self, "precision", _integer(self.precision, "evaluation.precision")
        )
        if not isinstance(self.resolved, bool):
            raise ConfigurationError("evaluation.resolved must be a boolean")
        object.__setattr__(
            self,
            "helicity_ids",
            _tuple_of_strings(self.helicity_ids, "evaluation.helicity_ids"),
        )
        object.__setattr__(
            self,
            "color_flow_ids",
            _tuple_of_strings(self.color_flow_ids, "evaluation.color_flow_ids"),
        )
        object.__setattr__(
            self,
            "model_parameters",
            _path(self.model_parameters, "evaluation.model_parameters"),
        )
        object.__setattr__(self, "momenta", _path(self.momenta, "evaluation.momenta"))


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    target_runtime: float = field(default=10.0, metadata=_setting("float"))
    batch_size: int = field(default=128, metadata=_setting("int"))
    precision: int = field(default=16, metadata=_setting("int"))
    warmup_runs: int = field(default=2, metadata=_setting("int"))
    minimum_samples: int = field(default=5, metadata=_setting("int"))
    helicity_ids: tuple[str, ...] = field(default=(), metadata=_setting("list_str"))
    color_flow_ids: tuple[str, ...] = field(default=(), metadata=_setting("list_str"))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_runtime",
            _finite_float(
                self.target_runtime,
                "benchmark.target_runtime",
                minimum=0.0,
                exclusive=True,
            ),
        )
        object.__setattr__(
            self, "batch_size", _integer(self.batch_size, "benchmark.batch_size")
        )
        object.__setattr__(
            self, "precision", _integer(self.precision, "benchmark.precision")
        )
        object.__setattr__(
            self,
            "warmup_runs",
            _integer(self.warmup_runs, "benchmark.warmup_runs", minimum=0),
        )
        object.__setattr__(
            self,
            "minimum_samples",
            _integer(self.minimum_samples, "benchmark.minimum_samples"),
        )
        object.__setattr__(
            self,
            "helicity_ids",
            _tuple_of_strings(self.helicity_ids, "benchmark.helicity_ids"),
        )
        object.__setattr__(
            self,
            "color_flow_ids",
            _tuple_of_strings(self.color_flow_ids, "benchmark.color_flow_ids"),
        )


@dataclass(frozen=True, slots=True)
class OutputConfig:
    format: OutputFormat = field(
        default=OutputFormat.HUMAN,
        metadata=_setting("str", choices=tuple(OutputFormat)),
    )
    color: ColorMode = field(
        default=ColorMode.AUTO,
        metadata=_setting("str", choices=tuple(ColorMode)),
    )
    progress: ProgressMode = field(
        default=ProgressMode.AUTO,
        metadata=_setting("str", choices=tuple(ProgressMode)),
    )
    log_level: LogLevel = field(
        default=LogLevel.INFO,
        metadata=_setting("str", choices=tuple(LogLevel)),
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "format", _enum(self.format, OutputFormat, "output.format")
        )
        object.__setattr__(self, "color", _enum(self.color, ColorMode, "output.color"))
        object.__setattr__(
            self, "progress", _enum(self.progress, ProgressMode, "output.progress")
        )
        object.__setattr__(
            self, "log_level", _enum(self.log_level, LogLevel, "output.log_level")
        )


@dataclass(frozen=True, slots=True)
class SymbolicaConfig:
    suggest_license: bool = field(default=True, metadata=_setting("bool"))

    def __post_init__(self) -> None:
        if not isinstance(self.suggest_license, bool):
            raise ConfigurationError("symbolica.suggest_license must be a boolean")


@dataclass(frozen=True, slots=True)
class RunConfig:
    action: Action = field(metadata=_setting("str", choices=ACTIONS))
    schema_version: int = field(default=1, metadata=_setting("int", choices=(1,)))
    model: ModelConfig = field(default_factory=ModelConfig, metadata=_section())
    process: ProcessConfig = field(default_factory=ProcessConfig, metadata=_section())
    color: ColorConfig = field(default_factory=ColorConfig, metadata=_section())
    generation: GenerationConfig = field(
        default_factory=GenerationConfig, metadata=_section()
    )
    evaluator: EvaluatorConfig = field(
        default_factory=EvaluatorConfig, metadata=_section()
    )
    evaluation: EvaluationConfig = field(
        default_factory=EvaluationConfig, metadata=_section()
    )
    benchmark: BenchmarkConfig = field(
        default_factory=BenchmarkConfig, metadata=_section()
    )
    output: OutputConfig = field(default_factory=OutputConfig, metadata=_section())
    symbolica: SymbolicaConfig = field(
        default_factory=SymbolicaConfig, metadata=_section()
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _enum(self.action, Action, "action"))
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise ConfigurationError("schema_version must be an integer")
        _choice(self.schema_version, (1,), "schema_version")
        expected = (
            ("model", ModelConfig),
            ("process", ProcessConfig),
            ("color", ColorConfig),
            ("generation", GenerationConfig),
            ("evaluator", EvaluatorConfig),
            ("evaluation", EvaluationConfig),
            ("benchmark", BenchmarkConfig),
            ("output", OutputConfig),
            ("symbolica", SymbolicaConfig),
        )
        for name, expected_type in expected:
            if not isinstance(getattr(self, name), expected_type):
                raise ConfigurationError(f"{name} must be a {expected_type.__name__}")
        if (
            self.color.contraction is ColorContraction.SYMMETRIC_GROUP_FFT
            and self.evaluator.execution_mode
            not in {
                EvaluatorExecutionMode.RECURRENCE,
                EvaluatorExecutionMode.ON_THE_FLY,
            }
        ):
            raise ConfigurationError(
                "color.contraction='symmetric-group-fft' requires "
                "evaluator.execution_mode='recurrence' or 'on-the-fly'"
            )

    @property
    def resolved_color(self) -> ColorConfig:
        """Keep automatic FFT selection only in its supported execution lanes.

        Eligible automatic requests stay automatic until the process-specific
        colour plan can certify FFT support. An explicit direct request ignores
        the FFT basis, including its adaptive default.
        """

        color = self.color
        if color.contraction is ColorContraction.DIRECT or (
            color.contraction is ColorContraction.AUTO
            and (
                color.accuracy is ColorAccuracy.LC
                or self.evaluator.execution_mode
                not in {
                    EvaluatorExecutionMode.RECURRENCE,
                    EvaluatorExecutionMode.ON_THE_FLY,
                }
            )
        ):
            return replace(
                color,
                contraction=ColorContraction.DIRECT,
                fft_basis=ColorFFTBasis.TRACE,
            )
        return color


__all__ = [
    "ACTIONS",
    "Action",
    "AutoBool",
    "AutoInt",
    "BenchmarkConfig",
    "ColorAccuracy",
    "ColorConfig",
    "ColorContraction",
    "ColorFFTBasis",
    "ColorMode",
    "CouplingOrderPolicy",
    "CppConfig",
    "EagerEvaluatorConfig",
    "EvaluationConfig",
    "EvaluatorBackend",
    "EvaluatorConfig",
    "EvaluatorExecutionMode",
    "EvaluatorOptimizationConfig",
    "GenerationConfig",
    "GenerationMode",
    "GenerationRelationDiscoveryConfig",
    "GenerationValidationConfig",
    "JITConfig",
    "LCFlowLayout",
    "LogLevel",
    "ModelConfig",
    "OutputConfig",
    "OutputFormat",
    "ProcessConfig",
    "ProcessEntry",
    "ProcessGroupAnalysis",
    "ProgressMode",
    "RecurrenceEvaluatorConfig",
    "RelationDiscoveryMode",
    "RunConfig",
    "SymbolicaConfig",
]
