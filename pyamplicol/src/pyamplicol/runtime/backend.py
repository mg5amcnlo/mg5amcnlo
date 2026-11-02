# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
import inspect
import json
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pyamplicol._internal.versions import (
    COMPILED_RUNTIME_SELECTORS_CAPABILITY,
    EAGER_RUNTIME_LAYOUT_F64_CAPABILITY,
    ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
    ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
    ON_THE_FLY_RUNTIME_CAPABILITY,
    RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
    RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY,
    SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
    verify_native_module,
)
from pyamplicol.api.errors import (
    ArtifactError,
    CompatibilityError,
    DependencyError,
    EvaluationError,
)
from pyamplicol.api.protocols import ModelParameters, Momenta
from pyamplicol.api.results import (
    ColorFlow,
    ContractedColorComponent,
    ExternalParticle,
    HelicityConfiguration,
    ModelParameter,
    PhysicsReduction,
    ProcessPhysics,
    ReductionGroup,
    ResolvedEvaluation,
    WarmUpResult,
)
from pyamplicol.artifacts import MANIFEST_NAME, ArtifactManifest, load_manifest
from pyamplicol.reporting import (
    ProgressEnd,
    ProgressSink,
    ProgressStart,
    ProgressUpdate,
)
from pyamplicol.runtime._native_selection import (
    charge_conjugate_color_id,
    native_process_selection,
)

if TYPE_CHECKING:
    from .eager_exact import EagerExactExecutor
    from .recurrence_exact import RecurrenceExactExecutor
    from .symbolica_exact import SymbolicaExactExecutor

    _ExactExecutor = (
        EagerExactExecutor | RecurrenceExactExecutor | SymbolicaExactExecutor
    )

_Accuracy = Literal["lc", "nlc", "full"]
_ParticleState = Literal["incoming", "outgoing"]
_ExecutionMode = Literal["compiled", "eager", "recurrence", "on-the-fly"]

_ON_THE_FLY_WARM_UP_TASK_ID = "runtime-warm-up"
_ON_THE_FLY_WARM_UP_STAGES = (
    "process_preparation",
    "query_family",
    "family_finalization",
    "first_evaluation",
)
_ON_THE_FLY_WARM_UP_STAGE_LABELS = {
    "process_preparation": "Preparing the OTF process",
    "query_family": "Constructing the OTF query family",
    "family_finalization": "Finalizing the OTF family",
    "first_evaluation": "Evaluating the warm-up point",
}


def _manifest_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{description} must be an integer")
    return value


def _load_native_module() -> Any:
    try:
        module = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(module)
    except ImportError as exc:
        raise DependencyError(
            "the pyamplicol._rusticol native runtime extension is unavailable"
        ) from exc
    if not hasattr(module, "Runtime"):
        raise DependencyError(
            "the pyamplicol._rusticol extension does not provide Runtime"
        )
    return module


def _translated_error(module: Any, error: Exception) -> Exception | None:
    mappings = (
        ("CompatibilityError", CompatibilityError),
        ("ArtifactError", ArtifactError),
        ("SelectorError", EvaluationError),
        ("ModelParameterError", EvaluationError),
        ("EvaluationError", EvaluationError),
        ("RusticolError", EvaluationError),
    )
    for native_name, public_type in mappings:
        native_type = getattr(module, native_name, None)
        if isinstance(native_type, type) and isinstance(error, native_type):
            return public_type(str(error))
    return None


def _invoke(module: Any, operation: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return operation(*args, **kwargs)
    except Exception as exc:
        translated = _translated_error(module, exc)
        if translated is None:
            raise
        raise translated from exc


def _accepts_keyword_arguments(operation: object, *names: str) -> bool:
    if not callable(operation):
        return False
    try:
        parameters = inspect.signature(operation).parameters.values()
    except (TypeError, ValueError):
        return False
    declared = {parameter.name for parameter in parameters}
    accepts_arbitrary = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    return accepts_arbitrary or all(name in declared for name in names)


def _normalized_selectors(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    return tuple(values) or None


def _warm_up_integer(
    payload: Mapping[str, object],
    name: str,
    *,
    optional: bool = False,
) -> int | None:
    value = payload.get(name)
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        qualifier = "non-negative integer"
        if optional:
            qualifier += " or null"
        raise ArtifactError(f"native OTF warm-up {name} must be a {qualifier}")
    return value


def _warm_up_seconds(payload: Mapping[str, object], name: str) -> float:
    value = payload.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0.0
    ):
        raise ArtifactError(
            f"native OTF warm-up {name} must be finite and non-negative"
        )
    return float(value)


def _warm_up_payload(raw: object, description: str) -> Mapping[str, object]:
    try:
        payload = json.loads(str(raw))
    except (TypeError, ValueError) as exc:
        raise ArtifactError(
            f"native OTF warm-up {description} is invalid: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError(f"native OTF warm-up {description} must be an object")
    if payload.get("schema_version") != 1:
        raise ArtifactError(
            f"native OTF warm-up {description} has an invalid schema version"
        )
    return payload


def _warm_up_result(raw: object) -> WarmUpResult:
    payload = _warm_up_payload(raw, "result")
    if payload.get("first_evaluation_completed") is not True:
        raise ArtifactError(
            "native OTF warm-up first_evaluation_completed must be true"
        )
    already_warm = payload.get("already_warm")
    if not isinstance(already_warm, bool):
        raise ArtifactError("native OTF warm-up already_warm must be a boolean")
    query_count = _warm_up_integer(payload, "query_count")
    warmed_query_count = _warm_up_integer(payload, "warmed_query_count")
    current_rss_bytes = _warm_up_integer(payload, "current_rss_bytes", optional=True)
    peak_rss_bytes = _warm_up_integer(payload, "peak_rss_bytes", optional=True)
    assert query_count is not None
    assert warmed_query_count is not None
    try:
        return WarmUpResult(
            elapsed_seconds=_warm_up_seconds(payload, "elapsed_seconds"),
            query_count=query_count,
            warmed_query_count=warmed_query_count,
            current_rss_bytes=current_rss_bytes,
            peak_rss_bytes=peak_rss_bytes,
            already_warm=already_warm,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"native OTF warm-up result is invalid: {exc}") from exc


class _OnTheFlyWarmUpProgress:
    """Translate throttled native snapshots into the shared progress protocol."""

    def __init__(
        self,
        sink: ProgressSink,
        *,
        process: object,
        color_accuracy: object,
    ) -> None:
        self._sink = sink
        self._active_stage: str | None = None
        self._process = process if isinstance(process, str) else None
        self._color_accuracy = (
            color_accuracy if isinstance(color_accuracy, str) else None
        )
        details: dict[str, str] = {"execution_mode": "on-the-fly"}
        if self._process:
            details["process"] = self._process
        if self._color_accuracy:
            details["color_accuracy"] = self._color_accuracy
        self._sink.emit(
            ProgressStart(
                _ON_THE_FLY_WARM_UP_TASK_ID,
                "Warming the on-the-fly runtime",
                total=len(_ON_THE_FLY_WARM_UP_STAGES),
                unit="phases",
                details=details,
            )
        )

    def __call__(self, raw: object) -> bool:
        payload = _warm_up_payload(raw, "progress event")
        kind = payload.get("kind")
        if kind not in {"start", "update", "end"}:
            raise ArtifactError("native OTF warm-up progress kind is invalid")
        stage = payload.get("stage")
        if stage not in _ON_THE_FLY_WARM_UP_STAGES:
            raise ArtifactError("native OTF warm-up progress stage is invalid")
        assert isinstance(stage, str)
        completed = _warm_up_integer(payload, "completed")
        total = _warm_up_integer(payload, "total")
        workers = _warm_up_integer(payload, "workers")
        current_rss_bytes = _warm_up_integer(
            payload, "current_rss_bytes", optional=True
        )
        peak_rss_bytes = _warm_up_integer(payload, "peak_rss_bytes", optional=True)
        assert completed is not None
        assert total is not None
        assert workers is not None
        if completed > total:
            raise ArtifactError(
                "native OTF warm-up progress completed count exceeds its total"
            )
        if (
            current_rss_bytes is not None
            and peak_rss_bytes is not None
            and current_rss_bytes > peak_rss_bytes
        ):
            raise ArtifactError("native OTF warm-up current RSS exceeds its peak RSS")
        elapsed_seconds = _warm_up_seconds(payload, "elapsed_seconds")
        message = payload.get("message")
        if message is not None and not isinstance(message, str):
            raise ArtifactError(
                "native OTF warm-up progress message must be a string or null"
            )
        stage_index = _ON_THE_FLY_WARM_UP_STAGES.index(stage)
        if stage != self._active_stage:
            self._finish_active_stage(success=True)
            self._active_stage = stage
            self._sink.emit(
                ProgressUpdate(
                    _ON_THE_FLY_WARM_UP_TASK_ID,
                    completed=stage_index,
                    total=len(_ON_THE_FLY_WARM_UP_STAGES),
                    message=_ON_THE_FLY_WARM_UP_STAGE_LABELS[stage],
                )
            )
            self._sink.emit(
                ProgressStart(
                    self._stage_task_id(stage),
                    _ON_THE_FLY_WARM_UP_STAGE_LABELS[stage],
                    total=total,
                    parent_task_id=_ON_THE_FLY_WARM_UP_TASK_ID,
                    unit="queries" if stage == "query_family" else "steps",
                    details={"stage": stage},
                )
            )
        details: dict[str, str | int | float] = {
            "stage": stage,
            "workers": workers,
            "elapsed_seconds": elapsed_seconds,
        }
        if current_rss_bytes is not None:
            details["native_current_rss_bytes"] = current_rss_bytes
        if peak_rss_bytes is not None:
            details["native_peak_rss_bytes"] = peak_rss_bytes
        self._sink.emit(
            ProgressUpdate(
                self._stage_task_id(stage),
                completed=completed,
                total=total,
                message=message,
                details=details,
            )
        )
        return True

    def finish(
        self,
        *,
        success: bool,
        elapsed_seconds: float | None = None,
        message: str | None = None,
    ) -> None:
        self._finish_active_stage(success=success, message=message)
        if success:
            self._sink.emit(
                ProgressUpdate(
                    _ON_THE_FLY_WARM_UP_TASK_ID,
                    completed=len(_ON_THE_FLY_WARM_UP_STAGES),
                    total=len(_ON_THE_FLY_WARM_UP_STAGES),
                    message="warm-up complete",
                )
            )
        self._sink.emit(
            ProgressEnd(
                _ON_THE_FLY_WARM_UP_TASK_ID,
                success=success,
                message=message,
                elapsed_seconds=elapsed_seconds,
            )
        )

    @staticmethod
    def _stage_task_id(stage: str) -> str:
        return f"{_ON_THE_FLY_WARM_UP_TASK_ID}:{stage}"

    def _finish_active_stage(
        self,
        *,
        success: bool,
        message: str | None = None,
    ) -> None:
        if self._active_stage is None:
            return
        self._sink.emit(
            ProgressEnd(
                self._stage_task_id(self._active_stage),
                success=success,
                message=message,
            )
        )
        self._active_stage = None


def _selected_manifest_process(
    manifest: ArtifactManifest,
    selected_id: str,
) -> Mapping[str, object]:
    for process in manifest.processes:
        if process["id"] == selected_id:
            return process
        if any(
            alias["id"] == selected_id
            for alias in cast(Sequence[Mapping[str, object]], process["aliases"])
        ):
            return process
    raise ArtifactError(
        f"runtime selected process {selected_id!r} is absent from its artifact"
    )


def _has_reusable_runtime_selector_contract(
    manifest: ArtifactManifest,
    process: Mapping[str, object],
) -> bool:
    relative = str(process["physics_path"])
    try:
        payload = json.loads((manifest.root / relative).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArtifactError(
            f"could not read runtime selector metadata {relative}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(
            f"invalid runtime selector metadata {relative}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError(f"runtime physics metadata {relative} must be an object")
    extensions = payload.get("extensions")
    if extensions is None:
        return False
    if not isinstance(extensions, Mapping):
        raise ArtifactError(
            f"runtime physics metadata {relative}.extensions is invalid"
        )
    selectors = extensions.get("runtime_selectors")
    if selectors is None:
        return False
    if not isinstance(selectors, Mapping):
        raise ArtifactError(
            f"runtime physics metadata {relative}.extensions.runtime_selectors "
            "is invalid"
        )
    axes = selectors.get("axes")
    if not isinstance(axes, Mapping):
        raise ArtifactError(
            f"runtime physics metadata {relative}.extensions.runtime_selectors.axes "
            "is invalid"
        )
    return any(
        isinstance(axis, Mapping)
        and axis.get("runtime_contract") == "complete-reusable"
        for axis in axes.values()
    )


def _physics_from_native(value: Any) -> ProcessPhysics:
    return ProcessPhysics(
        process_id=str(value.process_id),
        process=str(value.process),
        color_accuracy=cast(_Accuracy, str(value.color_accuracy)),
        helicity_coverage=str(value.helicity_coverage),
        color_coverage=str(value.color_coverage),
        color_kind=str(value.color_kind),
        structural_zero_helicity_count=int(value.structural_zero_helicity_count),
        external_particles=tuple(
            ExternalParticle(
                index=int(particle.index),
                label=int(particle.label),
                name=str(particle.name),
                pdg_id=int(particle.pdg_id),
                state=cast(_ParticleState, str(particle.state)),
                momentum_slot=int(particle.momentum_slot),
            )
            for particle in value.external_particles
        ),
        helicities=tuple(
            HelicityConfiguration(
                id=str(helicity.id),
                index=int(helicity.index),
                values=tuple(int(entry) for entry in helicity.values),
                computed=bool(helicity.computed),
                structural_zero=bool(helicity.structural_zero),
                representative_id=str(helicity.representative_id),
                coefficient=float(helicity.coefficient),
            )
            for helicity in value.helicities
        ),
        color_flows=tuple(
            ColorFlow(
                id=str(flow.id),
                index=int(flow.index),
                word=tuple(int(entry) for entry in flow.word),
                computed=bool(flow.computed),
                representative_id=str(flow.representative_id),
                coefficient=float(flow.coefficient),
            )
            for flow in value.color_flows
        ),
        contracted_color_components=tuple(
            ContractedColorComponent(
                id=str(component.id),
                index=int(component.index),
                description=str(component.description),
            )
            for component in value.contracted_color_components
        ),
        reduction=PhysicsReduction(
            kind=cast(
                Literal["lc-diagonal", "contracted-color"],
                str(value.reduction.kind),
            ),
            groups=tuple(
                ReductionGroup(
                    id=str(group.id),
                    representative_helicity_id=str(group.representative_helicity_id),
                    representative_color_id=str(group.representative_color_id),
                    physical_helicity_ids=tuple(
                        str(identifier) for identifier in group.physical_helicity_ids
                    ),
                    physical_color_ids=tuple(
                        str(identifier) for identifier in group.physical_color_ids
                    ),
                )
                for group in value.reduction.groups
            ),
        ),
        model_parameters=tuple(
            ModelParameter(
                name=str(parameter.name),
                kind=str(parameter.kind),
                default_real=float(parameter.default_real),
                default_imaginary=float(parameter.default_imaginary),
                mutable=bool(parameter.mutable),
            )
            for parameter in value.model_parameters
        ),
        selector_capabilities=tuple(
            str(capability) for capability in value.selector_capabilities
        ),
    )


def _scalar_from_native(value: Any) -> complex | Decimal:
    return value if isinstance(value, Decimal) else complex(value)


def _native_runtime_metadata(runtime: Any) -> Mapping[str, object]:
    metadata_json = getattr(runtime, "metadata_json", None)
    if not callable(metadata_json):
        return {"execution_mode": "compiled"}
    try:
        payload = json.loads(str(metadata_json()))
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"native runtime metadata is invalid: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError("native runtime metadata must be an object")
    return dict(payload)


def _native_execution_mode(
    metadata: Mapping[str, object],
) -> _ExecutionMode:
    payload = metadata
    mode = payload.get("execution_mode", "compiled")
    if mode not in {"compiled", "eager", "recurrence", "on-the-fly"}:
        raise CompatibilityError(f"unsupported runtime execution mode {mode!r}")
    return cast(_ExecutionMode, mode)


class RusticolRuntimeBackend:
    """Public-protocol adapter around ``pyamplicol._rusticol.Runtime``."""

    def __init__(
        self,
        runtime: Any,
        native_module: Any,
        artifact_path: Path,
        manifest: ArtifactManifest | None = None,
    ) -> None:
        self._runtime = runtime
        self._native_module = native_module
        self._artifact_path = artifact_path
        self._native_metadata = _native_runtime_metadata(runtime)
        self._execution_mode = _native_execution_mode(self._native_metadata)
        self._physics: ProcessPhysics | None = None
        self._exact_executor: _ExactExecutor | None = None
        self._required_runtime_capabilities: tuple[str, ...] = ()
        self._supports_per_point_selectors = _accepts_keyword_arguments(
            runtime.evaluate,
            "helicity_by_point",
            "color_flow_by_point",
        )
        if manifest is not None:
            self._validate_runtime_selector_capability(manifest)

    @property
    def physics(self) -> ProcessPhysics:
        if self._physics is None:
            self._physics = _physics_from_native(self._runtime.physics)
        return self._physics

    @property
    def execution_mode(self) -> _ExecutionMode:
        """Return the native execution lane selected by the artifact."""

        return self._execution_mode

    def _require_supported_precision(self, precision: int) -> None:
        if self._execution_mode == "on-the-fly" and precision != 16:
            raise CompatibilityError(
                "on-the-fly execution supports only precision=16 (native f64); "
                f"received precision={precision}"
            )
        if (
            self._execution_mode == "compiled"
            and precision != 16
            and SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY
            in self._required_runtime_capabilities
        ):
            raise CompatibilityError(
                "compiled symmetric-group FFT diagnostic execution supports only "
                "precision=16 native f64; exact/high-precision execution is unavailable"
            )

    def _on_the_fly_runtime_state_census(self) -> Mapping[str, object] | None:
        owns_companion = (
            self._execution_mode == "recurrence"
            and RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY
            in self._required_runtime_capabilities
        )
        if self._execution_mode != "on-the-fly" and not owns_companion:
            return None
        loader = getattr(
            self._runtime,
            "_on_the_fly_runtime_state_census_json",
            None,
        )
        if not callable(loader):
            raise CompatibilityError(
                "installed native runtime has no compact on-the-fly state census"
            )
        raw = _invoke(self._native_module, loader)
        if raw is None:
            raise CompatibilityError(
                "installed native runtime returned no on-the-fly state census"
            )
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError) as exc:
            raise ArtifactError(
                f"native on-the-fly state census is invalid: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ArtifactError("native on-the-fly state census must be an object")
        if payload.get("kind") != "rusticol-on-the-fly-runtime-state-census-v1":
            raise ArtifactError("native on-the-fly state census has an invalid kind")
        if payload.get("family_cache_policy") != "last-family-only":
            raise ArtifactError(
                "native on-the-fly state census has an invalid family cache policy"
            )
        cache_limit = payload.get("family_cache_limit")
        if (
            isinstance(cache_limit, bool)
            or not isinstance(cache_limit, int)
            or cache_limit != 1
        ):
            raise ArtifactError(
                "native on-the-fly state census has an invalid family cache limit"
            )
        return dict(payload)

    def inspect(self) -> Mapping[str, object]:
        """Return compact authenticated metadata without opening dense physics."""

        result: dict[str, object] = {
            "kind": "pyamplicol-runtime-inspection",
            "schema_version": 1,
            "artifact_id": self.artifact_id,
            "runtime_metadata": deepcopy(dict(self._native_metadata)),
            "on_the_fly_state": self._on_the_fly_runtime_state_census(),
        }
        if self._execution_mode == "on-the-fly":
            result["supported_precisions"] = (16,)
        return result

    def save(self, path: os.PathLike[str] | str) -> None:
        """Save the retained native on-the-fly structural cache."""

        self._cache_operation("save", path)

    def load_cache(self, path: os.PathLike[str] | str) -> None:
        """Replace the retained native on-the-fly cache after validation."""

        self._cache_operation("load_cache", path)

    def _cache_operation(self, name: str, path: os.PathLike[str] | str) -> None:
        if self._execution_mode != "on-the-fly":
            raise CompatibilityError(
                "saving and restoring caches is available only for on-the-fly runtimes"
            )
        operation = getattr(self._runtime, name, None)
        if not callable(operation):
            raise CompatibilityError(
                f"installed native runtime has no on-the-fly {name} binding"
            )
        _invoke(self._native_module, operation, path)

    def _on_the_fly_benchmark_context(
        self,
        color_flow_ids: Sequence[str],
    ) -> Mapping[str, object] | None:
        """Resolve the existing benchmark selectors through compact OTF state."""

        if self._execution_mode != "on-the-fly":
            return None
        loader = getattr(self._runtime, "_on_the_fly_benchmark_context_json", None)
        if not callable(loader):
            raise CompatibilityError(
                "installed native runtime has no compact on-the-fly benchmark context"
            )
        try:
            payload = json.loads(str(loader(tuple(color_flow_ids) or None)))
        except (TypeError, ValueError) as exc:
            raise ArtifactError(
                f"native on-the-fly benchmark context is invalid: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ArtifactError("native on-the-fly benchmark context must be an object")
        return dict(payload)

    @property
    def representative_process_key(self) -> str:
        """Stable artifact process whose payload implements this selection."""

        value = self._native_metadata.get("representative_process_key")
        if not isinstance(value, str) or not value:
            raise CompatibilityError(
                "native runtime metadata has no representative process key"
            )
        return value

    @property
    def external_count(self) -> int:
        """Authenticated number of public external legs."""

        value = self._native_metadata.get("external_count")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CompatibilityError(
                "native runtime metadata has no valid external process count"
            )
        return value

    @property
    def external_permutation(self) -> tuple[int, ...]:
        """Representative-index to public-index external-leg permutation."""

        raw = self._native_metadata.get("external_permutation")
        if not isinstance(raw, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in raw
        ):
            raise CompatibilityError(
                "native runtime metadata has no external process permutation"
            )
        permutation = tuple(raw)
        count = self.external_count
        if len(permutation) != count or sorted(permutation) != list(range(count)):
            raise ArtifactError("native runtime external permutation is invalid")
        return permutation

    @property
    def process_group_factor(self) -> Fraction:
        """Exact factor relating the selected process to its representative."""

        raw = self._native_metadata.get("process_group_factor")
        if raw is None:
            return Fraction(1)
        if not isinstance(raw, Mapping):
            raise ArtifactError("native runtime process-group factor is invalid")
        numerator, denominator = raw.get("numerator"), raw.get("denominator")
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or numerator < 1
            or denominator < 1
        ):
            raise ArtifactError("native runtime process-group factor is invalid")
        return Fraction(numerator, denominator)

    @property
    def charge_conjugated(self) -> bool:
        """Whether the selected process is its representative's conjugate."""

        return self._native_metadata.get("charge_conjugated") is True

    @property
    def artifact_id(self) -> str:
        """Return the manifest identity authenticated by the native loader."""

        value = getattr(self._runtime, "artifact_id", None)
        if not isinstance(value, str) or len(value) != 64:
            raise CompatibilityError(
                "native runtime has no authenticated artifact identity; "
                "reinstall pyAmpliCol from the current source revision"
            )
        try:
            int(value, 16)
        except ValueError as exc:
            raise ArtifactError(
                "native runtime carries an invalid artifact identity"
            ) from exc
        return value

    @property
    def supports_profiling(self) -> bool:
        """Whether this installed native runtime exposes the optional profiler."""

        return callable(getattr(self._runtime, "profile", None))

    @property
    def supports_per_point_selectors(self) -> bool:
        """Whether this artifact/runtime pair supports per-point selectors."""

        return self._supports_per_point_selectors

    @property
    def required_runtime_capabilities(self) -> tuple[str, ...]:
        """Runtime capabilities declared by the selected artifact process."""

        return self._required_runtime_capabilities

    def _validate_runtime_selector_capability(
        self,
        manifest: ArtifactManifest,
    ) -> None:
        selected_id = self._native_metadata.get("process_group_member_of")
        if selected_id is None:
            selected_id = self._native_metadata.get("process_key")
        if not isinstance(selected_id, str) or not selected_id:
            selected_id = self.physics.process_id
        process = _selected_manifest_process(manifest, selected_id)
        capabilities = tuple(
            str(value)
            for value in cast(
                Sequence[object], process["required_runtime_capabilities"]
            )
        )
        self._required_runtime_capabilities = capabilities
        if self._execution_mode == "compiled":
            reusable_selectors = _has_reusable_runtime_selector_contract(
                manifest,
                process,
            )
            declares_selector_capability = (
                COMPILED_RUNTIME_SELECTORS_CAPABILITY in capabilities
            )
            if reusable_selectors and not declares_selector_capability:
                raise CompatibilityError(
                    "compiled artifact declares reusable runtime selectors but does "
                    f"not require {COMPILED_RUNTIME_SELECTORS_CAPABILITY!r}; "
                    "regenerate the artifact with the current pyAmpliCol"
                )
            self._supports_per_point_selectors = declares_selector_capability
        elif self._execution_mode == "eager":
            self._supports_per_point_selectors = (
                EAGER_RUNTIME_LAYOUT_F64_CAPABILITY in capabilities
            )
        elif self._execution_mode == "on-the-fly":
            color_capabilities = {
                ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
                ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
            }
            capability_values = set(capabilities)
            selected_color_capabilities = capability_values.intersection(
                color_capabilities
            )
            expected_capabilities: set[str] = set()
            if len(selected_color_capabilities) == 1:
                selected_color_capability = next(iter(selected_color_capabilities))
                expected_capabilities = {
                    ON_THE_FLY_RUNTIME_CAPABILITY,
                    selected_color_capability,
                }
                if (
                    selected_color_capability
                    == ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY
                    and SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY
                    in capability_values
                ):
                    expected_capabilities.add(
                        SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY
                    )
            # The native loader has already authenticated the OTF color payload.
            # Keep this adapter check exact: only contracted color may add FFT.
            if (
                len(selected_color_capabilities) != 1
                or capability_values != expected_capabilities
                or len(capabilities) != len(expected_capabilities)
            ):
                raise CompatibilityError(
                    "on-the-fly artifact has an invalid runtime capability contract"
                )
            self._supports_per_point_selectors = True
        else:
            self._supports_per_point_selectors = (
                RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY in capabilities
            )
        if self._supports_per_point_selectors and not _accepts_keyword_arguments(
            self._runtime.evaluate,
            "helicity_by_point",
            "color_flow_by_point",
        ):
            if self._execution_mode == "compiled":
                required = COMPILED_RUNTIME_SELECTORS_CAPABILITY
            elif self._execution_mode == "eager":
                required = EAGER_RUNTIME_LAYOUT_F64_CAPABILITY
            elif self._execution_mode == "on-the-fly":
                required = ON_THE_FLY_RUNTIME_CAPABILITY
            else:
                required = RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY
            raise CompatibilityError(
                f"artifact requires runtime capability {required!r}, but the "
                "installed native runtime does not accept per-point selectors"
            )

    def warm_up(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int = 16,
        progress: ProgressSink | None = None,
        n_cores: int | None = None,
    ) -> WarmUpResult:
        """Warm one compact OTF selector family from exactly one f64 point."""

        if self._execution_mode != "on-the-fly":
            raise CompatibilityError(
                "explicit warm-up is available only for on-the-fly runtimes"
            )
        self._require_supported_precision(precision)
        if len(momenta) != 1:
            raise EvaluationError(
                "on-the-fly warm-up requires exactly one phase-space point"
            )
        helicities = _normalized_selectors(helicities)
        color_flows = _normalized_selectors(color_flows)
        color_accuracy = self._native_metadata.get("color_accuracy")
        if not isinstance(color_accuracy, str):
            color_accuracy = getattr(self._runtime, "color_accuracy", None)
        if color_accuracy not in {"lc", "nlc", "full"}:
            raise ArtifactError(
                "native on-the-fly runtime metadata has invalid color accuracy"
            )
        if color_accuracy != "lc" and color_flows is not None:
            raise EvaluationError(
                "NLC/full on-the-fly warm-up does not expose color-flow selectors"
            )
        color_flows = self._resolve_color_flows(color_flows)
        operation = getattr(self._runtime, "_on_the_fly_warm_up_f64_json", None)
        if not callable(operation):
            raise CompatibilityError(
                "installed native runtime has no explicit on-the-fly warm-up binding"
            )
        reporter = (
            None
            if progress is None
            else _OnTheFlyWarmUpProgress(
                progress,
                process=self._native_metadata.get("process"),
                color_accuracy=color_accuracy,
            )
        )
        try:
            raw = _invoke(
                self._native_module,
                operation,
                momenta,
                helicity_ids=helicities,
                color_flow_ids=color_flows,
                progress_callback=reporter,
                n_cores=n_cores,
            )
            result = _warm_up_result(raw)
        except Exception as exc:
            if reporter is not None:
                # Cleanup reporting must not replace the native/callback error.
                with suppress(Exception):
                    reporter.finish(success=False, message=str(exc))
            raise
        if reporter is not None:
            # The native warm-up is committed; terminal reporting cannot undo it.
            with suppress(Exception):
                reporter.finish(success=True, elapsed_seconds=result.elapsed_seconds)
        return result

    def evaluate(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        helicity_by_point: Sequence[str] | None = None,
        color_flow_by_point: Sequence[str] | None = None,
        precision: int = 16,
    ) -> tuple[complex | Decimal, ...]:
        self._require_supported_precision(precision)
        helicities = _normalized_selectors(helicities)
        color_flows = _normalized_selectors(color_flows)
        helicity_by_point = _normalized_selectors(helicity_by_point)
        color_flow_by_point = _normalized_selectors(color_flow_by_point)
        if helicities is not None and helicity_by_point is not None:
            raise EvaluationError(
                "helicities and helicity_by_point are mutually exclusive"
            )
        if color_flows is not None and color_flow_by_point is not None:
            raise EvaluationError(
                "color_flows and color_flow_by_point are mutually exclusive"
            )
        color_flows = self._resolve_color_flows(color_flows)
        color_flow_by_point = self._resolve_color_flows(color_flow_by_point)
        if (
            helicity_by_point is not None or color_flow_by_point is not None
        ) and not self._supports_per_point_selectors:
            raise CompatibilityError(
                "the selected artifact/runtime does not support per-point "
                "helicity or color-flow selectors; regenerate a reusable-selector "
                "artifact with the current pyAmpliCol"
            )
        helicity_indices = (
            None
            if helicity_by_point is None
            else self._point_selector_indices(
                helicity_by_point,
                "helicity_by_point",
            )
        )
        color_indices = (
            None
            if color_flow_by_point is None
            else self._point_selector_indices(
                color_flow_by_point,
                "color_flow_by_point",
            )
        )
        if precision != 16:
            if helicity_by_point is not None or color_flow_by_point is not None:
                return self._evaluate_exact_by_point(
                    momenta,
                    helicities=helicities,
                    color_flows=color_flows,
                    helicity_by_point=helicity_by_point,
                    color_flow_by_point=color_flow_by_point,
                    precision=precision,
                )
            return (
                self._exact()
                .evaluate_resolved(
                    momenta,
                    helicities=helicities,
                    color_flows=color_flows,
                    precision=precision,
                )
                .total()
            )
        selector_arguments: dict[str, object] = {}
        if helicity_indices is not None:
            selector_arguments["helicity_by_point"] = helicity_indices
        if color_indices is not None:
            selector_arguments["color_flow_by_point"] = color_indices
        values = _invoke(
            self._native_module,
            self._runtime.evaluate,
            momenta,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
            **selector_arguments,
        )
        return tuple(_scalar_from_native(value) for value in values)

    def _point_selector_indices(
        self,
        values: Sequence[str] | None,
        name: str,
    ) -> tuple[int, ...] | None:
        if values is None:
            return None
        if self._execution_mode == "on-the-fly":
            resolver = getattr(
                self._runtime,
                "_on_the_fly_selector_ordinals_json",
                None,
            )
            if not callable(resolver):
                raise CompatibilityError(
                    "installed native runtime has no compact on-the-fly "
                    "selector resolver"
                )
            helicities = values if name == "helicity_by_point" else None
            colors = values if name == "color_flow_by_point" else None
            try:
                payload = json.loads(str(resolver(helicities, colors)))
            except (TypeError, ValueError) as exc:
                raise ArtifactError(
                    f"native on-the-fly point-selector response is invalid: {exc}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ArtifactError(
                    "native on-the-fly point-selector response must be an object"
                )
            key = (
                "helicity_ordinals" if name == "helicity_by_point" else "color_ordinals"
            )
            raw = payload.get(key)
            if (
                not isinstance(raw, list)
                or len(raw) != len(values)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in raw
                )
            ):
                raise ArtifactError(f"native on-the-fly {name} ordinals are invalid")
            return tuple(raw)
        available = (
            self.physics.helicity_ids
            if name == "helicity_by_point"
            else self.physics.color_ids
        )
        index_by_id = {identifier: index for index, identifier in enumerate(available)}
        indices: list[int] = []
        for point_index, identifier in enumerate(values):
            try:
                indices.append(index_by_id[identifier])
            except KeyError as exc:
                raise EvaluationError(
                    f"{name}[{point_index}] references unknown selector {identifier!r}"
                ) from exc
        return tuple(indices)

    def _evaluate_exact_by_point(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None,
        color_flows: Sequence[str] | None,
        helicity_by_point: Sequence[str] | None,
        color_flow_by_point: Sequence[str] | None,
        precision: int,
    ) -> tuple[complex | Decimal, ...]:
        point_count = len(momenta)
        for name, selectors in (
            ("helicity_by_point", helicity_by_point),
            ("color_flow_by_point", color_flow_by_point),
        ):
            if selectors is not None and len(selectors) != point_count:
                raise EvaluationError(
                    f"{name} contains {len(selectors)} entries, expected one "
                    f"selector for each of {point_count} points"
                )
        grouped: dict[tuple[str | None, str | None], list[int]] = {}
        for point_index in range(point_count):
            key = (
                None if helicity_by_point is None else helicity_by_point[point_index],
                None
                if color_flow_by_point is None
                else color_flow_by_point[point_index],
            )
            grouped.setdefault(key, []).append(point_index)
        output: list[complex | Decimal | None] = [None] * point_count
        exact = self._exact()
        for (point_helicity, point_color), point_indices in grouped.items():
            selected_momenta = tuple(momenta[index] for index in point_indices)
            resolved = exact.evaluate_resolved(
                selected_momenta,
                helicities=(
                    (point_helicity,) if point_helicity is not None else helicities
                ),
                color_flows=(point_color,) if point_color is not None else color_flows,
                precision=precision,
            )
            for point_index, value in zip(point_indices, resolved.total(), strict=True):
                output[point_index] = value
        if any(value is None for value in output):
            raise EvaluationError("per-point exact selector evaluation was incomplete")
        return cast(tuple[complex | Decimal, ...], tuple(output))

    def _benchmark_f64_wall_time(
        self,
        momenta: Momenta,
        repetitions: int,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        helicity_by_point: Sequence[str] | None = None,
        color_flow_by_point: Sequence[str] | None = None,
        precision: int = 16,
    ) -> float:
        self._require_supported_precision(precision)
        helicities = _normalized_selectors(helicities)
        color_flows = _normalized_selectors(color_flows)
        helicity_by_point = _normalized_selectors(helicity_by_point)
        color_flow_by_point = _normalized_selectors(color_flow_by_point)
        if helicities is not None and helicity_by_point is not None:
            raise EvaluationError(
                "helicities and helicity_by_point are mutually exclusive"
            )
        if color_flows is not None and color_flow_by_point is not None:
            raise EvaluationError(
                "color_flows and color_flow_by_point are mutually exclusive"
            )
        color_flows = self._resolve_color_flows(color_flows)
        color_flow_by_point = self._resolve_color_flows(color_flow_by_point)
        if (
            helicity_by_point is not None or color_flow_by_point is not None
        ) and not self._supports_per_point_selectors:
            raise CompatibilityError(
                "the selected artifact/runtime does not support per-point "
                "helicity or color-flow selectors"
            )
        helicity_indices = (
            None
            if helicity_by_point is None
            else self._point_selector_indices(
                helicity_by_point,
                "helicity_by_point",
            )
        )
        color_indices = (
            None
            if color_flow_by_point is None
            else self._point_selector_indices(
                color_flow_by_point,
                "color_flow_by_point",
            )
        )
        timer = getattr(self._runtime, "_benchmark_f64_wall_time", None)
        if not callable(timer):
            raise EvaluationError("native Rusticol wall timer is unavailable")
        selector_arguments: dict[str, object] = {}
        if helicity_indices is not None:
            selector_arguments["helicity_by_point"] = helicity_indices
        if color_indices is not None:
            selector_arguments["color_flow_by_point"] = color_indices
        return float(
            _invoke(
                self._native_module,
                timer,
                momenta,
                repetitions,
                helicities=helicities,
                color_flows=color_flows,
                precision=precision,
                **selector_arguments,
            )
        )

    def _profile_arena_repeated(
        self,
        momenta: Momenta,
        repetitions: int,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int = 16,
        include_values: bool = False,
    ) -> Mapping[str, object]:
        """Use the private warmed-Arena profiler without a public fallback."""

        self._require_supported_precision(precision)
        helicities = _normalized_selectors(helicities)
        color_flows = self._resolve_color_flows(_normalized_selectors(color_flows))
        if precision != 16:
            raise EvaluationError(
                "Arena profiling is available only for native f64 precision"
            )
        profiler = getattr(self._runtime, "_profile_arena_repeated", None)
        if not callable(profiler):
            raise EvaluationError(
                "native runtime does not expose warmed Arena profiling"
            )
        payload = _invoke(
            self._native_module,
            profiler,
            momenta,
            repetitions,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
            include_values=include_values,
        )
        if not isinstance(payload, Mapping):
            raise EvaluationError("native warmed Arena profile is not a mapping")
        return cast(Mapping[str, object], dict(payload))

    def evaluate_resolved(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int = 16,
    ) -> ResolvedEvaluation:
        self._require_supported_precision(precision)
        helicities = _normalized_selectors(helicities)
        color_flows = self._resolve_color_flows(_normalized_selectors(color_flows))
        if precision != 16:
            return self._exact().evaluate_resolved(
                momenta,
                helicities=helicities,
                color_flows=color_flows,
                precision=precision,
            )
        native = _invoke(
            self._native_module,
            self._runtime.evaluate_resolved,
            momenta,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
        )
        values = tuple(
            tuple(
                tuple(_scalar_from_native(entry) for entry in colors)
                for colors in helicities_at_point
            )
            for helicities_at_point in native.values
        )
        return ResolvedEvaluation(
            values=values,
            helicity_ids=tuple(str(value) for value in native.helicity_ids),
            color_ids=tuple(str(value) for value in native.color_ids),
            color_accuracy=cast(_Accuracy, str(native.color_accuracy)),
        )

    def profile(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        helicity_by_point: Sequence[str] | None = None,
        color_flow_by_point: Sequence[str] | None = None,
        precision: int = 16,
        include_values: bool = False,
    ) -> Mapping[str, object]:
        self._require_supported_precision(precision)
        helicities = _normalized_selectors(helicities)
        color_flows = self._resolve_color_flows(_normalized_selectors(color_flows))
        helicity_by_point = _normalized_selectors(helicity_by_point)
        color_flow_by_point = self._resolve_color_flows(
            _normalized_selectors(color_flow_by_point)
        )
        if helicities is not None and helicity_by_point is not None:
            raise EvaluationError(
                "helicities and helicity_by_point are mutually exclusive"
            )
        if color_flows is not None and color_flow_by_point is not None:
            raise EvaluationError(
                "color_flows and color_flow_by_point are mutually exclusive"
            )
        if precision != 16:
            raise EvaluationError(
                "runtime profiling is available only for native f64 precision"
            )
        profiler = getattr(self._runtime, "profile", None)
        if not callable(profiler):
            raise EvaluationError("native runtime does not expose profiling")
        selector_arguments: dict[str, object] = {}
        if helicity_by_point is not None or color_flow_by_point is not None:
            if not self._supports_per_point_selectors or not _accepts_keyword_arguments(
                profiler,
                "helicity_by_point",
                "color_flow_by_point",
            ):
                raise CompatibilityError(
                    "the selected artifact/runtime does not support profiling "
                    "with per-point helicity or color-flow selectors"
                )
            helicity_indices = self._point_selector_indices(
                helicity_by_point,
                "helicity_by_point",
            )
            color_indices = self._point_selector_indices(
                color_flow_by_point,
                "color_flow_by_point",
            )
            if helicity_indices is not None:
                selector_arguments["helicity_by_point"] = helicity_indices
            if color_indices is not None:
                selector_arguments["color_flow_by_point"] = color_indices
        payload = _invoke(
            self._native_module,
            profiler,
            momenta,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
            include_values=include_values,
            **selector_arguments,
        )
        if not isinstance(payload, Mapping):
            raise EvaluationError("native runtime profile is not a mapping")
        return cast(Mapping[str, object], dict(payload))

    def profile_repeated(
        self,
        momenta: Momenta,
        repetitions: int,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        helicity_by_point: Sequence[str] | None = None,
        color_flow_by_point: Sequence[str] | None = None,
        precision: int = 16,
        include_values: bool = False,
    ) -> Mapping[str, object]:
        self._require_supported_precision(precision)
        helicities = _normalized_selectors(helicities)
        color_flows = self._resolve_color_flows(_normalized_selectors(color_flows))
        helicity_by_point = _normalized_selectors(helicity_by_point)
        color_flow_by_point = self._resolve_color_flows(
            _normalized_selectors(color_flow_by_point)
        )
        if helicities is not None and helicity_by_point is not None:
            raise EvaluationError(
                "helicities and helicity_by_point are mutually exclusive"
            )
        if color_flows is not None and color_flow_by_point is not None:
            raise EvaluationError(
                "color_flows and color_flow_by_point are mutually exclusive"
            )
        if precision != 16:
            raise EvaluationError(
                "runtime profiling is available only for native f64 precision"
            )
        profiler = getattr(self._runtime, "profile_repeated", None)
        if not callable(profiler):
            raise EvaluationError("native runtime does not expose repeated profiling")
        selector_arguments: dict[str, object] = {}
        if helicity_by_point is not None or color_flow_by_point is not None:
            if not self._supports_per_point_selectors or not _accepts_keyword_arguments(
                profiler,
                "helicity_by_point",
                "color_flow_by_point",
            ):
                raise CompatibilityError(
                    "the selected artifact/runtime does not support profiling "
                    "with per-point helicity or color-flow selectors"
                )
            helicity_indices = self._point_selector_indices(
                helicity_by_point,
                "helicity_by_point",
            )
            color_indices = self._point_selector_indices(
                color_flow_by_point,
                "color_flow_by_point",
            )
            if helicity_indices is not None:
                selector_arguments["helicity_by_point"] = helicity_indices
            if color_indices is not None:
                selector_arguments["color_flow_by_point"] = color_indices
        payload = _invoke(
            self._native_module,
            profiler,
            momenta,
            repetitions,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
            include_values=include_values,
            **selector_arguments,
        )
        if not isinstance(payload, Mapping):
            raise EvaluationError("native repeated runtime profile is not a mapping")
        return cast(Mapping[str, object], dict(payload))

    def evaluate_profile(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        helicity_by_point: Sequence[str] | None = None,
        color_flow_by_point: Sequence[str] | None = None,
        precision: int = 16,
        include_values: bool = False,
    ) -> Mapping[str, object]:
        return self.profile(
            momenta,
            helicities=helicities,
            color_flows=color_flows,
            helicity_by_point=helicity_by_point,
            color_flow_by_point=color_flow_by_point,
            precision=precision,
            include_values=include_values,
        )

    def _resolve_color_flows(
        self,
        color_flows: Sequence[str] | None,
    ) -> tuple[str, ...] | None:
        if color_flows is None or not color_flows:
            return None
        if self._execution_mode == "on-the-fly":

            def is_ordinal(value: str) -> bool:
                try:
                    ordinal = int(value, 10)
                except ValueError:
                    return False
                return str(ordinal) == value.strip()

            if not any(is_ordinal(value) for value in color_flows):
                return tuple(color_flows)
            context = self._on_the_fly_benchmark_context(color_flows)
            assert context is not None
            selected = context.get("selected_color_ids")
            if not isinstance(selected, Sequence) or isinstance(
                selected,
                (str, bytes),
            ):
                raise ArtifactError("native on-the-fly selected color IDs are invalid")
            return tuple(str(value) for value in selected)
        available = self.physics.color_ids
        resolved: list[str] = []
        for requested in color_flows:
            if requested in available:
                resolved.append(requested)
                continue
            try:
                ordinal = int(requested, 10)
            except ValueError:
                resolved.append(requested)
                continue
            if (
                str(ordinal) != requested.strip()
                or ordinal < 1
                or ordinal > len(available)
            ):
                raise EvaluationError(
                    f"color-flow ordinal {requested!r} is out of range; choose "
                    f"1..{len(available)} or a stable color component ID"
                )
            resolved.append(available[ordinal - 1])
        return tuple(resolved)

    def _exact(self) -> _ExactExecutor:
        if self._exact_executor is None:
            if self._execution_mode == "eager":
                from .eager_exact import EagerExactExecutor

                self._exact_executor = EagerExactExecutor(
                    self._artifact_path, self.physics.process_id, self._runtime
                )
            elif self._execution_mode == "recurrence":
                from .recurrence_exact import RecurrenceExactExecutor

                self._exact_executor = RecurrenceExactExecutor(
                    self._artifact_path, self.physics.process_id, self._runtime
                )
            else:
                from .symbolica_exact import SymbolicaExactExecutor

                self._exact_executor = SymbolicaExactExecutor(
                    self._artifact_path, self.physics.process_id, self._runtime
                )
            factor, conjugated = self.process_group_factor, self.charge_conjugated
            if factor != 1 or conjugated:
                self._exact_executor = cast(
                    "_ExactExecutor",
                    _ProcessGroupExactExecutor(
                        self._exact_executor, factor, conjugated
                    ),
                )
        return self._exact_executor

    def _diagnostic_project_onshell(
        self,
        momenta: Momenta,
        *,
        precision: int,
    ) -> tuple[
        tuple[tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...], ...],
        dict[str, object],
    ]:
        """Project kinematics only for the bounded validation diagnostic."""

        self._require_supported_precision(precision)
        return self._exact()._diagnostic_project_onshell(
            momenta,
            precision=precision,
        )

    def set_model_parameters(self, mapping: ModelParameters) -> None:
        _invoke(
            self._native_module,
            self._runtime.set_model_parameters,
            dict(mapping),
        )

    def clear(self) -> None:
        _invoke(self._native_module, self._runtime.clear)

    def mute_warnings(self) -> None:
        _invoke(self._native_module, self._runtime.mute_warnings)

    def unmute_warnings(self) -> None:
        _invoke(self._native_module, self._runtime.unmute_warnings)

    def take_warnings(self) -> tuple[str, ...]:
        values = _invoke(self._native_module, self._runtime.take_warnings)
        return tuple(str(value) for value in values)

    def validation_momenta(self) -> Momenta | None:
        """Return the selected process's verified deterministic artifact point."""

        manifest = load_manifest(self._artifact_path)
        selection = native_process_selection(self._runtime, manifest.processes)
        representative = selection.process
        permutation = selection.external_permutation
        process_id = selection.representative_process_id
        payloads = tuple(
            payload
            for payload in manifest.payloads
            if payload.role == "validation-momenta" and payload.process_id == process_id
        )
        if len(payloads) != 1:
            raise ArtifactError(
                f"process {process_id!r} must declare exactly one validation point"
            )
        try:
            raw = json.loads((manifest.root / payloads[0].path).read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(
                f"could not read validation point for process {process_id!r}: {exc}"
            ) from exc
        if not isinstance(raw, dict) or raw.get("available") is not True:
            return None
        points = raw.get("points")
        if not isinstance(points, list) or len(points) != 1:
            raise ArtifactError(
                f"process {process_id!r} has invalid validation-point metadata"
            )
        particles = points[0]
        expected_pdgs = tuple(
            _manifest_integer(value, "representative external PDG")
            for value in cast(Sequence[object], representative["external_pdgs"])
        )
        if not isinstance(particles, list) or len(particles) != len(expected_pdgs):
            raise ArtifactError(
                f"process {process_id!r} validation point has the wrong particle count"
            )
        vectors: list[tuple[float, float, float, float]] = []
        pdgs: list[int] = []
        for index, particle in enumerate(particles):
            if not isinstance(particle, dict):
                raise ArtifactError(
                    f"process {process_id!r} validation particle {index} is invalid"
                )
            momentum = particle.get("momentum")
            if not isinstance(momentum, list) or len(momentum) != 4:
                raise ArtifactError(
                    f"process {process_id!r} validation momentum {index} is invalid"
                )
            try:
                pdgs.append(int(particle["pdg"]))
                vectors.append(
                    cast(
                        tuple[float, float, float, float],
                        tuple(float(component) for component in momentum),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ArtifactError(
                    f"process {process_id!r} validation particle {index} is invalid"
                ) from exc
        if tuple(pdgs) != expected_pdgs:
            raise ArtifactError(
                f"process {process_id!r} validation PDGs do not match its metadata"
            )
        reordered: list[tuple[float, float, float, float] | None] = [None] * len(
            vectors
        )
        for representative_index, public_index in enumerate(permutation):
            reordered[public_index] = vectors[representative_index]
        if any(vector is None for vector in reordered):
            raise ArtifactError(
                f"selected process {process_id!r} has an incomplete permutation"
            )
        vectors = cast(list[tuple[float, float, float, float]], reordered)
        return (tuple(vectors),)


class _ProcessGroupExactExecutor:
    """Present a representative's exact results as a process-group member.

    Values are multiplied by the member's exact factor; a charge-conjugated
    member's LC color-flow IDs are the reversed representative words, mapped
    on input and output exactly as in the native lane.
    """

    def __init__(self, inner: Any, factor: Fraction, conjugated: bool) -> None:
        self._inner = inner
        self._factor = factor
        self._conjugated = conjugated

    def _color_ids(self, values: Sequence[str] | None) -> tuple[str, ...] | None:
        if values is None or not self._conjugated:
            return None if values is None else tuple(values)
        return tuple(charge_conjugate_color_id(value) for value in values)

    def _scaled(self, value: complex | Decimal, precision: int) -> complex | Decimal:
        if self._factor == 1:
            return value
        if not isinstance(value, Decimal):
            return value * self._factor.numerator / self._factor.denominator
        digits = max(precision, len(value.as_tuple().digits))
        with localcontext() as context:
            context.rounding = ROUND_HALF_EVEN
            # Exact product, then one correctly rounded division.
            context.prec = digits + len(str(self._factor.numerator))
            product = value * self._factor.numerator
            context.prec = digits
            return product / self._factor.denominator

    def evaluate_resolved(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None,
        color_flows: Sequence[str] | None,
        precision: int,
    ) -> ResolvedEvaluation:
        resolved = self._inner.evaluate_resolved(
            momenta,
            helicities=helicities,
            color_flows=self._color_ids(color_flows),
            precision=precision,
        )
        return ResolvedEvaluation(
            values=tuple(
                tuple(
                    tuple(self._scaled(value, precision) for value in colors)
                    for colors in point
                )
                for point in resolved.values
            ),
            helicity_ids=resolved.helicity_ids,
            color_ids=cast(tuple[str, ...], self._color_ids(resolved.color_ids)),
            color_accuracy=resolved.color_accuracy,
        )

    def _diagnostic_project_onshell(self, momenta: Momenta, *, precision: int) -> Any:
        return self._inner._diagnostic_project_onshell(momenta, precision=precision)


def load_runtime_backend(
    artifact: os.PathLike[str] | str,
    *,
    process: str | None = None,
    model_parameters: ModelParameters | None = None,
    mute_warnings: bool = False,
) -> RusticolRuntimeBackend:
    """Load a schema-v3 artifact without importing the extension during discovery."""

    path = Path(os.fspath(artifact)).expanduser().resolve(strict=False)
    parameters = dict(model_parameters) if model_parameters is not None else None
    module = _load_native_module()
    # Rusticol is the authoritative runtime loader and verifies every declared
    # payload before returning.  Parse the manifest here only for the Python
    # adapter's selector-contract checks; hashing it again would add a complete
    # extra pass over large compact eager containers.
    manifest = (
        load_manifest(path, verify_payloads=False)
        if (path / MANIFEST_NAME).is_file()
        else None
    )
    runtime = _invoke(
        module,
        module.Runtime.load,
        path,
        process=process,
        model_parameters=parameters,
        mute_warnings=mute_warnings,
    )
    return RusticolRuntimeBackend(runtime, module, path, manifest)


__all__ = ["RusticolRuntimeBackend", "load_runtime_backend"]
