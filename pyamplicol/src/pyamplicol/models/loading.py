# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .prepared import PreparedModelBundle

from .._internal.versions import (
    COMPILED_MODEL_SCHEMA_VERSION,
    SYMBOLICA_SERIALIZATION_ABI,
    package_version,
)
from .contact_decomposition import (
    CONTACT_DECOMPOSITION_ALGORITHM,
    CONTACT_DECOMPOSITION_ALGORITHM_VERSION,
)
from .contracts import (
    DEFAULT_FEYNMAN_PROPAGATOR_SOURCE,
    MODEL_SUPPLIED_PROPAGATOR_SOURCE,
    PROPAGATOR_SOURCE_FIELD,
    TENSOR_ORDERING_CONTRACT_VERSION,
    CompiledModelIR,
)

COMPILED_MODEL_KIND = "pyamplicol-compiled-model"
MODEL_COMPILER_VERSION = 16
BUILTIN_SM_ALIASES = frozenset(("builtin_sm", "built-in-sm"))
DEFAULT_MODEL_RESTRICTION = "default"
NO_MODEL_RESTRICTION = "none"
SANITIZED_MODEL_ENVIRONMENT_PREFIXES = (
    "UFO_SCALARS_MODEL_",
    "UFO_GRAVITY_MODEL_",
)
_MODEL_LOAD_LOCK = threading.RLock()

SUPPORTED_FUNCTION_ARITIES = {
    name: frozenset({1})
    for name in (
        "Theta",
        "abs",
        "acos",
        "acosh",
        "acsc",
        "asec",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "complexconjugate",
        "conj",
        "cos",
        "cosh",
        "csc",
        "exp",
        "im",
        "log",
        "log10",
        "re",
        "reglog",
        "reglogm",
        "reglogp",
        "sec",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    )
}
SUPPORTED_FUNCTION_ARITIES.update(
    {
        "complex": frozenset({2}),
        "cond": frozenset({3}),
        "if": frozenset({3}),
        "pow": frozenset({2}),
    }
)
SUPPORTED_FUNCTIONS = frozenset(SUPPORTED_FUNCTION_ARITIES)

UFO_TENSOR_HEADS = frozenset(
    {
        "C",
        "Epsilon",
        "EpsilonBar",
        "Gamma",
        "Gamma5",
        "Identity",
        "IdentityL",
        "K6",
        "K6Bar",
        "Metric",
        "P",
        "PSlash",
        "ProjM",
        "ProjP",
        "Sigma",
        "T",
        "T6",
        "d",
        "dummy",
        "f",
        "idx",
    }
)


def _require_ufo_model_loader() -> None:
    """Reject loaders predating the supported model serialization API."""

    installed = _distribution_version("ufo-model-loader", "not installed")
    # The loader publishes three-component release versions. Development and
    # prerelease builds are not substitutes for the required release interface.
    release = re.fullmatch(
        r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?(?:\+[\w.-]+)?", installed
    )
    if release is None or tuple(map(int, release.groups())) < (1, 0, 0):
        raise RuntimeError(
            "UFO model import requires ufo-model-loader release 1.0.0 or newer "
            f"for the supported model serialization API; found {installed}. "
            "Install the updated loader before importing a UFO model."
        )


@dataclass(frozen=True, slots=True)
class ModelCompileOptions:
    restriction: str = DEFAULT_MODEL_RESTRICTION
    simplify: bool = True

    def __post_init__(self) -> None:
        restriction_path = Path(self.restriction).expanduser()
        if not self.restriction or (
            any(character.isspace() for character in self.restriction)
            and not restriction_path.is_file()
        ):
            raise ValueError(
                "model restriction must be default, none, or a restriction name"
            )
        if not isinstance(self.simplify, bool):
            raise TypeError("model simplification must be a boolean")

    def canonical_payload(self) -> dict[str, object]:
        restriction_path = Path(self.restriction).expanduser()
        if restriction_path.is_file():
            restriction: object = {
                "kind": "file",
                "sha256": hashlib.sha256(restriction_path.read_bytes()).hexdigest(),
            }
        else:
            restriction = {"kind": "name", "value": self.restriction}
        return {
            "restriction": restriction,
            "simplify": self.simplify,
        }


@dataclass(frozen=True)
class ModelCompatibilityIssue:
    severity: str
    code: str
    message: str
    context: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, object]) -> ModelCompatibilityIssue:
        return ModelCompatibilityIssue(
            severity=str(payload["severity"]),
            code=str(payload["code"]),
            message=str(payload["message"]),
            context=str(payload.get("context", "")),
        )


@dataclass(frozen=True)
class CompiledModel:
    source: Mapping[str, object]
    producer: Mapping[str, object]
    model: Mapping[str, object]
    ir: CompiledModelIR
    parameter_defaults: Mapping[str, tuple[float, float]]
    capabilities: Mapping[str, object]
    issues: tuple[ModelCompatibilityIssue, ...]
    phase_timings: Mapping[str, float]
    conversion_seconds: float
    _serialized_path: Path | None = field(default=None, compare=False, repr=False)
    _prepared_bundle: PreparedModelBundle | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    @property
    def name(self) -> str:
        return str(self.model.get("name", "unnamed-model"))

    @property
    def schema_version(self) -> int:
        return COMPILED_MODEL_SCHEMA_VERSION

    @property
    def model_compiler_version(self) -> int:
        return MODEL_COMPILER_VERSION

    @property
    def supported(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def prepared_bundle(self) -> PreparedModelBundle | None:
        return self._prepared_bundle

    @property
    def prepared_backend(self) -> str | None:
        return None if self._prepared_bundle is None else self._prepared_bundle.backend

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": COMPILED_MODEL_KIND,
            "schema_version": COMPILED_MODEL_SCHEMA_VERSION,
            "model_compiler_version": MODEL_COMPILER_VERSION,
            "source": dict(self.source),
            "producer": dict(self.producer),
            "model": dict(self.model),
            "ir": self.ir.to_dict(),
            "parameter_defaults": {
                name: [value[0], value[1]]
                for name, value in sorted(self.parameter_defaults.items())
            },
            "capabilities": dict(self.capabilities),
            "issues": [issue.to_dict() for issue in self.issues],
            "phase_timings": dict(self.phase_timings),
            "conversion_seconds": self.conversion_seconds,
        }

    def write(self, path: Path) -> Path:
        output = _compiled_model_output_path(path)
        if self._serialized_path is None or not self._serialized_path.is_file():
            _atomic_write_json(output, self.to_dict(), compact=True)
        else:
            _atomic_copy(self._serialized_path, output)
        return output

    def write_parameter_card(self, path: Path) -> Path:
        _atomic_write_json(
            path,
            {
                name: [value[0], value[1]]
                for name, value in sorted(self.parameter_defaults.items())
            },
        )
        return path

    @staticmethod
    def from_dict(
        payload: Mapping[str, object],
        *,
        validate_fingerprint: bool = True,
        serialized_path: Path | None = None,
    ) -> CompiledModel:
        if payload.get("kind") != COMPILED_MODEL_KIND:
            raise ValueError("file is not a pyAmpliCol compiled model")
        if int(payload.get("schema_version", -1)) != COMPILED_MODEL_SCHEMA_VERSION:
            raise ValueError("compiled model schema mismatch; regenerate the model")
        if int(payload.get("model_compiler_version", -1)) != MODEL_COMPILER_VERSION:
            raise ValueError("compiled model compiler mismatch; regenerate the model")
        producer = _mapping(payload.get("producer"), "producer")
        if validate_fingerprint and producer != compiler_fingerprint():
            raise ValueError(
                "compiled model dependency fingerprint mismatch; regenerate the model"
            )
        parameter_payload = _mapping(
            payload.get("parameter_defaults"),
            "parameter_defaults",
        )
        parameters = {
            str(name): _complex_pair(value, context=f"parameter {name}")
            for name, value in parameter_payload.items()
        }
        raw_issues = payload.get("issues", ())
        if not isinstance(raw_issues, list):
            raise ValueError("compiled model issues must be a list")
        return CompiledModel(
            source=_mapping(payload.get("source"), "source"),
            producer=producer,
            model=_mapping(payload.get("model"), "model"),
            ir=CompiledModelIR.from_dict(_mapping(payload.get("ir"), "ir")),
            parameter_defaults=parameters,
            capabilities=_mapping(payload.get("capabilities"), "capabilities"),
            issues=tuple(
                ModelCompatibilityIssue.from_dict(_mapping(issue, "issue"))
                for issue in raw_issues
            ),
            phase_timings={
                str(name): float(value)
                for name, value in _mapping(
                    payload.get("phase_timings"),
                    "phase_timings",
                ).items()
            },
            conversion_seconds=float(payload.get("conversion_seconds", 0.0)),
            _serialized_path=(
                None if serialized_path is None else Path(serialized_path).resolve()
            ),
        )


def compiler_fingerprint() -> dict[str, object]:
    return {
        "pyamplicol": package_version(),
        "ufo_model_loader": _distribution_version("ufo-model-loader", "missing"),
        "symbolica": _distribution_version("symbolica", "missing"),
        "compiled_model_schema_version": COMPILED_MODEL_SCHEMA_VERSION,
        "model_compiler_version": MODEL_COMPILER_VERSION,
        "model_compiler_sha256": _model_compiler_digest(),
        "symbolica_serialization_abi": SYMBOLICA_SERIALIZATION_ABI,
        "contact_decomposition_policy": (
            f"{CONTACT_DECOMPOSITION_ALGORITHM}-v"
            f"{CONTACT_DECOMPOSITION_ALGORITHM_VERSION}"
        ),
        "tensor_ordering_contract": (
            f"explicit-canonical-component-order-v{TENSOR_ORDERING_CONTRACT_VERSION}"
        ),
        "model_environment_policy": "sanitize-historical-scalar-options-v1",
        "symbol_namespace_policy": "model-name-and-pyamplicol-registry-v1",
    }


def detect_model_source(source: str | Path) -> tuple[str, str | Path]:
    source_kind, resolved, _payload = _detect_model_source_payload(source)
    return source_kind, resolved


def _detect_model_source_payload(
    source: str | Path,
) -> tuple[str, str | Path, dict[str, object] | None]:
    text = str(source)
    if text.lower() in BUILTIN_SM_ALIASES:
        return "built-in-sm", "built-in-sm", None
    path = Path(text).expanduser().resolve()
    if path.is_dir():
        if not (path / "__init__.py").is_file():
            raise ValueError(f"UFO model directory has no __init__.py: {path}")
        return "ufo", path, None
    if not path.is_file():
        raise ValueError(f"model source does not exist: {path}")
    from .prepared import PREPARED_MODEL_BUNDLE_SUFFIX

    if path.name.lower().endswith(PREPARED_MODEL_BUNDLE_SUFFIX):
        return "prepared", path, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"model JSON could not be read: {path}: {exc}") from exc
    if isinstance(payload, dict) and payload.get("kind") == COMPILED_MODEL_KIND:
        return "pyamplicol", path, payload
    return "json", path, payload if isinstance(payload, dict) else None


def compile_model_source(
    source: str | Path = "BUILTIN_SM",
    *,
    restriction: str = "default",
    simplify: bool = True,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    require_supported: bool = True,
) -> CompiledModel:
    options = ModelCompileOptions(restriction=restriction, simplify=simplify)
    source_kind, resolved, detected_payload = _detect_model_source_payload(source)
    if source_kind in {"pyamplicol", "prepared"}:
        if restriction != DEFAULT_MODEL_RESTRICTION or not simplify:
            raise ValueError(
                "restriction and simplification options cannot be applied to an "
                "already compiled pyAmpliCol model"
            )
        if source_kind == "prepared":
            from .prepared import load_prepared_model_bundle

            bundle = load_prepared_model_bundle(Path(resolved))
            compiled = CompiledModel.from_dict(
                bundle.compiled_model_payload(),
                validate_fingerprint=False,
            )
            compiled = replace(compiled, _prepared_bundle=bundle)
            _raise_for_unsupported(compiled, require_supported=require_supported)
            return compiled
        if detected_payload is None:
            raise RuntimeError(
                "compiled model detection did not retain its JSON payload"
            )
        return CompiledModel.from_dict(
            detected_payload,
            serialized_path=Path(resolved),
        )
    source_digest, fingerprint, cache_path = _compilation_cache_identity(
        source_kind,
        resolved,
        options=options,
        cache_dir=cache_dir,
    )
    if use_cache and cache_path.is_file():
        compiled = load_compiled_model(cache_path)
        _raise_for_unsupported(compiled, require_supported=require_supported)
        return compiled

    started = time.perf_counter()
    phase_timings: dict[str, float] = {}
    phase_started = time.perf_counter()
    if source_kind == "built-in-sm":
        from .builtin.adapters import build_model_payload

        model_payload, parameter_defaults = build_model_payload()
    else:
        worker_payload = _load_external_model(
            Path(resolved),
            options=options,
        )
        model_payload = _mapping(worker_payload.get("model"), "worker model")
        card = _mapping(worker_payload.get("parameter_card"), "parameter card")
        parameter_defaults = {
            str(name): _complex_pair(value, context=f"parameter {name}")
            for name, value in card.items()
        }
    phase_timings["model_loading"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    issues, capabilities = preflight_model(model_payload)
    phase_timings["preflight"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    from .compiler import compile_builtin_model_ir, compile_ufo_model_ir

    model_ir = (
        compile_builtin_model_ir(model_payload)
        if source_kind == "built-in-sm"
        else compile_ufo_model_ir(model_payload)
    )
    phase_timings["tensor_lowering"] = time.perf_counter() - phase_started
    contact_term_ids = {term.id for term in model_ir.vertex_terms if term.valence > 3}
    lowered_contact_term_ids = {
        term_id
        for kernel in model_ir.oriented_kernels
        if "::contact-" in kernel.vertex and "final" in kernel.vertex
        for term_id in (kernel.term_ids or (kernel.term_id,))
    }
    unlowered_contact_term_ids = contact_term_ids - lowered_contact_term_ids
    if unlowered_contact_term_ids:
        issues = (
            *issues,
            ModelCompatibilityIssue(
                "error",
                "unsupported-contact-color-lowering",
                "one or more higher-point color tensors could not be lowered",
                ", ".join(
                    str(term_id) for term_id in sorted(unlowered_contact_term_ids)
                ),
            ),
        )
    evaluation_class_sizes: dict[str, int] = {}
    for kernel in model_ir.oriented_kernels:
        if not kernel.evaluation_equivalence_verified:
            continue
        evaluation_class_sizes[kernel.evaluation_class] = (
            evaluation_class_sizes.get(kernel.evaluation_class, 0) + 1
        )
    capabilities = {
        **capabilities,
        "compiled_vertex_term_count": len(model_ir.vertex_terms),
        "compiled_contact_term_count": len(lowered_contact_term_ids),
        "unlowered_contact_term_count": len(unlowered_contact_term_ids),
        "compiled_propagator_count": len(model_ir.propagators),
        "verified_kernel_evaluation_class_count": len(evaluation_class_sizes),
        "reusable_kernel_evaluation_class_count": sum(
            size > 1 for size in evaluation_class_sizes.values()
        ),
        "signed_kernel_evaluation_relation_count": sum(
            kernel.evaluation_factor != (1.0, 0.0)
            for kernel in model_ir.oriented_kernels
            if kernel.evaluation_equivalence_verified
        ),
        "swapped_kernel_evaluation_relation_count": sum(
            kernel.evaluation_input_order == (1, 0)
            for kernel in model_ir.oriented_kernels
            if kernel.evaluation_equivalence_verified
        ),
        "exchange_symmetric_kernel_evaluation_relation_count": sum(
            kernel.evaluation_input_exchange_factor is not None
            for kernel in model_ir.oriented_kernels
            if kernel.evaluation_equivalence_verified
        ),
    }
    conversion_seconds = time.perf_counter() - started
    phase_timings["total"] = conversion_seconds
    compiled = CompiledModel(
        source={
            "kind": source_kind,
            "source_name": (
                None if source_kind == "built-in-sm" else Path(resolved).name
            ),
            "digest": source_digest,
            "options": options.canonical_payload(),
        },
        producer=fingerprint,
        model=model_payload,
        ir=model_ir,
        parameter_defaults=parameter_defaults,
        capabilities=capabilities,
        issues=issues,
        phase_timings=phase_timings,
        conversion_seconds=conversion_seconds,
    )
    if use_cache:
        compiled.write(cache_path)
        compiled = replace(compiled, _serialized_path=cache_path.resolve())
    _raise_for_unsupported(compiled, require_supported=require_supported)
    return compiled


def load_cached_model_source(
    source: str | Path,
    *,
    restriction: str = DEFAULT_MODEL_RESTRICTION,
    simplify: bool = True,
    cache_dir: Path | None = None,
    require_supported: bool = True,
) -> CompiledModel | None:
    """Load an already compiled source without compiling or writing a cache."""

    options = ModelCompileOptions(restriction=restriction, simplify=simplify)
    source_kind, resolved, detected_payload = _detect_model_source_payload(source)
    if source_kind in {"pyamplicol", "prepared"}:
        if restriction != DEFAULT_MODEL_RESTRICTION or not simplify:
            raise ValueError(
                "restriction and simplification options cannot be applied to an "
                "already compiled pyAmpliCol model"
            )
        if source_kind == "prepared":
            from .prepared import load_prepared_model_bundle

            bundle = load_prepared_model_bundle(Path(resolved))
            compiled = CompiledModel.from_dict(
                bundle.compiled_model_payload(),
                validate_fingerprint=False,
            )
            compiled = replace(compiled, _prepared_bundle=bundle)
            _raise_for_unsupported(compiled, require_supported=require_supported)
            return compiled
        if detected_payload is None:
            raise RuntimeError(
                "compiled model detection did not retain its JSON payload"
            )
        compiled = CompiledModel.from_dict(
            detected_payload,
            serialized_path=Path(resolved),
        )
        _raise_for_unsupported(compiled, require_supported=require_supported)
        return compiled

    _, _, cache_path = _compilation_cache_identity(
        source_kind,
        resolved,
        options=options,
        cache_dir=cache_dir,
    )
    if not cache_path.is_file():
        return None
    compiled = load_compiled_model(cache_path)
    _raise_for_unsupported(compiled, require_supported=require_supported)
    return compiled


def load_compiled_model(path: Path) -> CompiledModel:
    from .prepared import PREPARED_MODEL_BUNDLE_SUFFIX, load_prepared_model_bundle

    if Path(path).name.lower().endswith(PREPARED_MODEL_BUNDLE_SUFFIX):
        bundle = load_prepared_model_bundle(Path(path))
        return replace(
            CompiledModel.from_dict(
                bundle.compiled_model_payload(),
                validate_fingerprint=False,
            ),
            _prepared_bundle=bundle,
        )
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load compiled model {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"compiled model root must be an object: {path}")
    return CompiledModel.from_dict(payload, serialized_path=Path(path))


def preflight_model(
    model: Mapping[str, object],
) -> tuple[tuple[ModelCompatibilityIssue, ...], dict[str, object]]:
    particles = _list_of_mappings(model.get("particles"), "particles")
    propagators = _list_of_mappings(model.get("propagators", []), "propagators")
    vertices = _list_of_mappings(model.get("vertex_rules"), "vertex_rules")
    functions = _list_of_mappings(model.get("functions", []), "functions")
    form_factors = _list_of_mappings(model.get("form_factors", []), "form_factors")
    issues: list[ModelCompatibilityIssue] = []
    spins = sorted({int(particle.get("spin", 0)) for particle in particles})
    colors = sorted({int(particle.get("color", 0)) for particle in particles})
    unsupported_spins = sorted(set(spins) - {-1, 1, 2, 3, 5})
    if unsupported_spins:
        issues.append(
            ModelCompatibilityIssue(
                "error",
                "unsupported-spin",
                f"unsupported UFO spin codes: {unsupported_spins}",
            )
        )
    unsupported_colors = sorted(set(colors) - {-3, 1, 3, 8})
    if unsupported_colors:
        issues.append(
            ModelCompatibilityIssue(
                "error",
                "unsupported-color-representation",
                f"unsupported UFO color representations: {unsupported_colors}",
            )
        )
    majorana = sorted(
        str(particle.get("name"))
        for particle in particles
        if int(particle.get("spin", 0)) == 2
        and particle.get("name") == particle.get("antiname")
    )
    if majorana:
        issues.append(
            ModelCompatibilityIssue(
                "error",
                "majorana-fermion",
                "Majorana/FNV fermion flow is not implemented",
                ", ".join(majorana),
            )
        )
    if form_factors:
        issues.append(
            ModelCompatibilityIssue(
                "error",
                "form-factors",
                "UFO form factors are not supported in this milestone",
                ", ".join(sorted(str(item.get("name")) for item in form_factors)),
            )
        )
    declared_functions = {str(function.get("name")) for function in functions}
    invalid_function_arities = []
    for function in functions:
        name = str(function.get("name"))
        if name not in SUPPORTED_FUNCTION_ARITIES:
            continue
        arity = len(_sequence(function.get("arguments")))
        if arity not in SUPPORTED_FUNCTION_ARITIES[name]:
            invalid_function_arities.append(f"{name}/{arity}")
    if invalid_function_arities:
        issues.append(
            ModelCompatibilityIssue(
                "error",
                "function-arity",
                "model function declarations do not match pyAmpliCol's registry",
                ", ".join(sorted(invalid_function_arities)),
            )
        )
    expression_functions = _model_expression_functions(model)
    unknown_functions = sorted(
        (declared_functions | expression_functions)
        - SUPPORTED_FUNCTIONS
        - UFO_TENSOR_HEADS
    )
    if unknown_functions:
        issues.append(
            ModelCompatibilityIssue(
                "error",
                "unknown-functions",
                "model uses functions that are not in pyAmpliCol's hard-coded registry",
                ", ".join(unknown_functions),
            )
        )
    massive_tensors = [
        particle
        for particle in particles
        if int(particle.get("spin", 0)) == 5
        and str(particle.get("mass", "ZERO")).upper() != "ZERO"
    ]
    if massive_tensors:
        issues.append(
            ModelCompatibilityIssue(
                "warning",
                "experimental-massive-spin-2",
                "massive spin-2 support is experimental",
                ", ".join(str(item.get("name")) for item in massive_tensors),
            )
        )
    max_valence = max(
        (len(_sequence(vertex.get("particles"))) for vertex in vertices), default=0
    )
    capabilities = {
        "supported": not any(issue.severity == "error" for issue in issues),
        "particle_count": len(particles),
        "parameter_count": len(_sequence(model.get("parameters"))),
        "vertex_count": len(vertices),
        "max_vertex_valence": max_valence,
        "spins": spins,
        "color_representations": colors,
        "declared_functions": sorted(declared_functions),
        "expression_functions": sorted(expression_functions),
        "form_factor_count": len(form_factors),
        "has_custom_propagators": any(
            propagator.get(PROPAGATOR_SOURCE_FIELD) == MODEL_SUPPLIED_PROPAGATOR_SOURCE
            for propagator in propagators
        ),
        "color_accuracy_modes": ["lc", "nlc", "full"],
    }
    return tuple(issues), capabilities


@contextmanager
def _sanitized_model_environment() -> Iterator[None]:
    """Hide historical untyped model switches while trusted UFO code executes."""

    with _MODEL_LOAD_LOCK:
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        removed = {
            name: value
            for name, value in os.environ.items()
            if name.startswith(SANITIZED_MODEL_ENVIRONMENT_PREFIXES)
        }
        for name in removed:
            os.environ.pop(name, None)
        try:
            yield
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode
            for name in tuple(os.environ):
                if name.startswith(SANITIZED_MODEL_ENVIRONMENT_PREFIXES):
                    os.environ.pop(name, None)
            os.environ.update(removed)


def _load_external_model(
    source: Path,
    *,
    options: ModelCompileOptions,
) -> dict[str, object]:
    # These imports intentionally live inside the execution boundary so package
    # import, metadata inspection, and built-in-SM planning stay dependency-light.
    _require_ufo_model_loader()
    with _sanitized_model_environment():
        from ufo_model_loader.commands import load_model
        from ufo_model_loader.common import JSONLook

        restriction = _loader_restriction_name(source, options.restriction)
        model, parameter_card = load_model(
            str(source),
            restriction,
            options.simplify,
            wrap_indices_in_lorentz_structures=False,
        )
        propagator_sources = _classify_external_propagators(model)
        model.wrap_indices_in_lorentz_structures()
        model_payload = json.loads(model.to_json(JSONLook.COMPACT))
        raw_propagators = model_payload.get("propagators", [])
        if not isinstance(raw_propagators, list):
            raise ValueError("serialized model propagators must be a list")
        for propagator in raw_propagators:
            if not isinstance(propagator, dict):
                raise ValueError("serialized model propagator must be an object")
            particle = str(propagator["particle"])
            propagator[PROPAGATOR_SOURCE_FIELD] = propagator_sources[particle]
        return {
            "loader_version": _distribution_version("ufo-model-loader", "missing"),
            "model": model_payload,
            "parameter_card": {
                name: [value.real, value.imag]
                for name, value in sorted(parameter_card.items())
            },
        }


def _classify_external_propagators(model: Any) -> dict[str, str]:
    """Classify propagators by expression, independently of UFO object names."""

    from ufo_model_loader.model import Propagator

    result: dict[str, str] = {}
    for propagator in model.propagators:
        actual = (
            propagator.numerator.to_canonical_string(),
            propagator.denominator.to_canonical_string(),
        )
        defaults = {
            (
                default.numerator.to_canonical_string(),
                default.denominator.to_canonical_string(),
            )
            for default in _default_feynman_propagator_candidates(
                Propagator,
                propagator.particle,
            )
        }
        result[str(propagator.particle.name)] = (
            DEFAULT_FEYNMAN_PROPAGATOR_SOURCE
            if actual in defaults
            else MODEL_SUPPLIED_PROPAGATOR_SOURCE
        )
    return result


def _default_feynman_propagator_candidates(
    propagator_type: Any,
    particle: Any,
) -> tuple[Any, ...]:
    """Cover loader defaults created before a restriction changes a mass value."""

    current = propagator_type.from_particle(particle, "Feynman")
    if particle.spin not in {1, 2, 3, -1}:
        return (current,)
    alternate_particle = copy(particle)
    alternate_particle.mass = copy(particle.mass)
    alternate_particle.mass.value = 0j if particle.is_massive() else 1.0 + 0j
    alternate = propagator_type.from_particle(alternate_particle, "Feynman")
    return current, alternate


def _loader_restriction_name(source: Path, restriction: str) -> str | None:
    """Translate the public restriction-file contract to the loader's name API."""

    if restriction == DEFAULT_MODEL_RESTRICTION:
        return None
    if restriction == NO_MODEL_RESTRICTION:
        return "full"

    path = Path(restriction).expanduser()
    if not path.is_absolute():
        return restriction
    path = path.resolve(strict=False)
    if not path.is_file():
        raise ValueError(f"model restriction file does not exist: {path}")

    model_root = source if source.is_dir() else source.parent
    if path.parent != model_root.resolve(strict=False):
        raise ValueError(
            "model restriction files must be stored next to the selected model"
        )

    suffix = ".dat" if source.is_dir() else ".json"
    prefix = "restrict_"
    if path.suffix.lower() != suffix or not path.name.startswith(prefix):
        legacy_default = (
            not source.is_dir() and path.name == f"{source.stem}_default.json"
        )
        if legacy_default:
            return DEFAULT_MODEL_RESTRICTION
        raise ValueError(
            f"model restriction file must be named {prefix}<name>{suffix}: {path}"
        )
    name = path.name[len(prefix) : -len(suffix)]
    if not name:
        raise ValueError(f"model restriction filename has no restriction name: {path}")
    return name


def _model_expression_functions(model: Mapping[str, object]) -> set[str]:
    expressions: list[str] = []
    for key in ("parameters", "couplings", "propagators", "lorentz_structures"):
        for item in _list_of_mappings(model.get(key, []), key):
            for value in item.values():
                if isinstance(value, str):
                    expressions.append(value)
    for vertex in _list_of_mappings(model.get("vertex_rules", []), "vertex_rules"):
        expressions.extend(
            str(value) for value in _sequence(vertex.get("color_structures"))
        )
    heads = set()
    for expression in expressions:
        for match in re.finditer(
            r"(?:(?:UFO|spenso)::(?:\{\}::)?)?([A-Za-z][A-Za-z0-9_]*)\(",
            expression,
        ):
            heads.add(match.group(1))
    return heads


def _source_digest(kind: str, source: str | Path) -> str:
    digest = hashlib.sha256()
    digest.update(kind.encode("utf-8") + b"\0")
    if kind == "built-in-sm":
        from .builtin.adapters import source_digest

        digest.update(source_digest().encode("ascii"))
        return digest.hexdigest()
    path = Path(source)
    root = path.parent if path.is_file() else path
    files = (
        [path]
        if path.is_file()
        else sorted(
            item
            for item in path.rglob("*")
            if item.is_file()
            and item.suffix != ".pyc"
            and "__pycache__" not in item.parts
        )
    )
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _compilation_cache_identity(
    source_kind: str,
    source: str | Path,
    *,
    options: ModelCompileOptions,
    cache_dir: Path | None,
) -> tuple[str, dict[str, object], Path]:
    source_digest = _source_digest(source_kind, source)
    fingerprint = compiler_fingerprint()
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "source_digest": source_digest,
                "options": options.canonical_payload(),
                "producer": fingerprint,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    active_cache = cache_dir or _default_model_cache_dir()
    return (
        source_digest,
        fingerprint,
        active_cache / f"{cache_key}.pyAmplicol-model.json",
    )


def _model_compiler_digest() -> str:
    digest = hashlib.sha256()
    package_root = Path(__file__).resolve().parents[1]
    for path in _model_compiler_source_paths():
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _model_compiler_source_paths() -> tuple[Path, ...]:
    """Return every first-party source file that can change compiled model IR."""
    package_root = Path(__file__).resolve().parents[1]
    model_sources = Path(__file__).resolve().parent.glob("*.py")
    physics_sources = (package_root / "_internal" / "physics").glob("*.py")
    process_sources = (package_root / "processes" / "core_syntax.py",)
    return tuple(sorted((*model_sources, *physics_sources, *process_sources)))


def _default_model_cache_dir() -> Path:
    configured = os.environ.get("PYAMPLICOL_CACHE_DIR")
    if configured:
        root = Path(configured).expanduser()
    else:
        from platformdirs import user_cache_path

        root = user_cache_path("pyamplicol")
    return root / "models"


def _compiled_model_output_path(path: Path) -> Path:
    text = str(path)
    suffix = ".pyAmplicol-model.json"
    return Path(text if text.endswith(suffix) else text + suffix)


def _atomic_write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    compact: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    serialized = (
        json.dumps(payload, separators=(",", ":"), sort_keys=True)
        if compact
        else json.dumps(payload, indent=2, sort_keys=True)
    )
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_copy(source: Path, path: Path) -> None:
    source = source.resolve()
    path = path.resolve()
    if source == path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(path)


def _distribution_version(name: str, default: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return default


def _raise_for_unsupported(
    model: CompiledModel,
    *,
    require_supported: bool,
) -> None:
    if require_supported and not model.supported:
        details = "; ".join(
            f"{issue.code}: {issue.message}"
            + (f" ({issue.context})" if issue.context else "")
            for issue in model.issues
            if issue.severity == "error"
        )
        raise ValueError(f"model {model.name!r} is not supported: {details}")


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _list_of_mappings(value: object, context: str) -> list[dict[str, object]]:
    return [_mapping(item, context) for item in _sequence(value)]


def _complex_pair(value: object, *, context: str) -> tuple[float, float]:
    pair = _sequence(value)
    if len(pair) != 2:
        raise ValueError(f"{context} must be [real, imaginary]")
    return float(pair[0]), float(pair[1])
