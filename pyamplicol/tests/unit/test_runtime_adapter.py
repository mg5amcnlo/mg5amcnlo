# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
import json
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from pyamplicol._internal.versions import (
    COMPILED_RUNTIME_SELECTORS_CAPABILITY,
    EAGER_DAG_F64_RUNTIME_CAPABILITY,
    EAGER_RUNTIME_LAYOUT_F64_CAPABILITY,
    ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
    ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
    ON_THE_FLY_RUNTIME_CAPABILITY,
    RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
    RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY,
    SYMJIT_F64_RUNTIME_CAPABILITY,
    SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
)
from pyamplicol.api import (
    ArtifactError,
    ColorFlow,
    ContractedColorComponent,
    ExternalParticle,
    HelicityConfiguration,
    ModelParameter,
    ProcessPhysics,
    ResolvedEvaluation,
    Runtime,
    RuntimeBackend,
    WarmUpResult,
)
from pyamplicol.api.errors import CompatibilityError, EvaluationError
from pyamplicol.artifacts import ArtifactManifest
from pyamplicol.reporting import (
    CallbackProgressSink,
    ProgressEnd,
    ProgressStart,
    ProgressUpdate,
)


class _FailingWarmUpProgressSink:
    def __init__(self, *, fail_during_update: bool = False) -> None:
        self.fail_during_update = fail_during_update
        self.events: list[ProgressStart | ProgressUpdate | ProgressEnd] = []

    def emit(self, event: ProgressStart | ProgressUpdate | ProgressEnd) -> None:
        self.events.append(event)
        if self.fail_during_update and isinstance(event, ProgressUpdate):
            raise LookupError("warm-up progress callback failed")
        if isinstance(event, ProgressEnd) and event.task_id == "runtime-warm-up":
            raise RuntimeError("terminal warm-up progress delivery failed")


def _native_physics(accuracy: str = "lc") -> SimpleNamespace:
    particles = [
        SimpleNamespace(
            index=index,
            label=index + 1,
            name=name,
            pdg_id=pdg,
            state="incoming" if index < 2 else "outgoing",
            momentum_slot=index,
        )
        for index, (name, pdg) in enumerate((("u", 2), ("u~", -2), ("g", 21)))
    ]
    helicities = [
        SimpleNamespace(
            id="h0",
            index=0,
            values=[-1, 1, 1],
            computed=True,
            structural_zero=False,
            representative_id="h0",
            coefficient=2.0,
        )
    ]
    flows = []
    contracted = []
    if accuracy == "lc":
        flows.append(
            SimpleNamespace(
                id="c0",
                index=0,
                word=[1, 2, 3],
                computed=True,
                representative_id="c0",
                coefficient=1.0,
            )
        )
    else:
        contracted.append(
            SimpleNamespace(id="contracted", index=0, description="summed color")
        )
    return SimpleNamespace(
        process_id="uux_g",
        process="u u~ > g",
        color_accuracy=accuracy,
        helicity_coverage="complete",
        color_coverage="complete" if accuracy == "lc" else "contracted",
        color_kind="physical-lc-flows" if accuracy == "lc" else "contracted-color",
        structural_zero_helicity_count=0,
        external_particles=particles,
        helicities=helicities,
        color_flows=flows,
        contracted_color_components=contracted,
        reduction=SimpleNamespace(
            kind="lc-diagonal" if accuracy == "lc" else "contracted-color",
            groups=[
                SimpleNamespace(
                    id="g0",
                    representative_helicity_id="h0",
                    representative_color_id=(
                        "c0" if accuracy == "lc" else "contracted"
                    ),
                    physical_helicity_ids=["h0"],
                    physical_color_ids=["c0" if accuracy == "lc" else "contracted"],
                )
            ],
        ),
        model_parameters=[
            SimpleNamespace(
                name="aS",
                kind="coupling",
                default_real=0.118,
                default_imaginary=0.0,
                mutable=True,
            )
        ],
        selector_capabilities=["helicity"]
        + (["color_flow"] if accuracy == "lc" else []),
    )


class _NativeArtifactError(Exception):
    pass


class _NativeRuntime:
    physics_value = _native_physics()
    load_arguments: tuple[object, ...] | None = None
    execution_mode = "compiled"
    artifact_id = "a" * 64
    last_evaluate_options: dict[str, object] | None = None
    last_benchmark_options: dict[str, object] | None = None
    last_profile_options: dict[str, object] | None = None
    last_arena_profile_options: dict[str, object] | None = None
    last_warm_up_options: dict[str, object] | None = None

    def __init__(self) -> None:
        self.parameter_updates: list[dict[str, complex | float | int]] = []
        self.muted = False
        self.clear_count = 0
        self.physics_access_count = 0
        self.selector_context_access_count = 0
        self.selector_ordinal_access_count = 0
        self.runtime_state_census_access_count = 0
        self.otf_process_preparation_count = 0
        self.otf_retained_family_count = 0
        self.otf_retained_selection_count = 0

    @classmethod
    def load(cls, artifact: Path, **kwargs: object) -> _NativeRuntime:
        cls.load_arguments = (artifact, kwargs)
        return cls()

    @property
    def physics(self) -> SimpleNamespace:
        self.physics_access_count += 1
        return self.physics_value

    def metadata_json(self) -> str:
        return json.dumps(
            {
                "execution_mode": self.execution_mode,
                "color_accuracy": self.physics_value.color_accuracy,
                "process_key": "uux_g",
                "representative_process_key": "uux_g",
                "external_count": 3,
                "external_permutation": [0, 1, 2],
                "on_the_fly_requested_query_construction_threads": (
                    8 if self.execution_mode == "on-the-fly" else None
                ),
                "on_the_fly_effective_query_construction_threads": (
                    4 if self.execution_mode == "on-the-fly" else None
                ),
            }
        )

    def _on_the_fly_benchmark_context_json(
        self,
        color_flow_ids: tuple[str, ...] | None = None,
    ) -> str:
        self.selector_context_access_count += 1
        return json.dumps(
            {
                "process_id": "uux_g",
                "process_expression": "u u~ > g",
                "color_accuracy": "lc",
                "helicity_count": 1,
                "color_count": 1,
                "selected_color_ids": list(color_flow_ids or ()),
            }
        )

    def _on_the_fly_selector_ordinals_json(
        self,
        helicity_ids: tuple[str, ...] | None = None,
        color_flow_ids: tuple[str, ...] | None = None,
    ) -> str:
        self.selector_ordinal_access_count += 1
        return json.dumps(
            {
                "helicity_ordinals": (
                    None if helicity_ids is None else [0] * len(helicity_ids)
                ),
                "color_ordinals": (
                    None if color_flow_ids is None else [0] * len(color_flow_ids)
                ),
            }
        )

    def _on_the_fly_runtime_state_census_json(self) -> str | None:
        self.runtime_state_census_access_count += 1
        if self.execution_mode not in {"on-the-fly", "recurrence"}:
            return None
        payload = {
            "kind": "rusticol-on-the-fly-runtime-state-census-v1",
            "process_id": "uux_g",
            "family_cache_policy": "last-family-only",
            "family_cache_limit": 1,
            "process_preparation_count": self.otf_process_preparation_count,
            "retained_family_count": self.otf_retained_family_count,
            "pending_family_count": 0,
            "retained_selection_count": self.otf_retained_selection_count,
            "retained_request_count": self.otf_retained_family_count,
            "retained_amplitude_destination_count": (self.otf_retained_family_count),
            "retained_executor_handle_count": self.otf_retained_family_count,
            "retained_query_local_trace_count": 0,
            "retained_embedded_lookup_key_count": 0,
            "semantic_executor_binding_count": self.otf_retained_family_count,
            "active_family_union_census": None,
        }
        if self.execution_mode == "recurrence":
            payload["owner"] = "recurrence-helicity-selector-companion"
        return json.dumps(payload)

    def _warm_on_the_fly(self) -> None:
        if self.execution_mode == "on-the-fly":
            self.otf_process_preparation_count = 1
            self.otf_retained_family_count = 1
            self.otf_retained_selection_count = 1

    def _on_the_fly_warm_up_f64_json(
        self,
        momenta: object,
        helicity_ids: tuple[str, ...] | None = None,
        color_flow_ids: tuple[str, ...] | None = None,
        progress_callback: object = None,
        n_cores: int | None = None,
    ) -> str:
        type(self).last_warm_up_options = {
            "momenta": momenta,
            "helicity_ids": helicity_ids,
            "color_flow_ids": color_flow_ids,
            "n_cores": n_cores,
        }
        if callable(progress_callback):
            for kind, stage, completed, total in (
                ("start", "process_preparation", 0, 1),
                ("update", "process_preparation", 1, 1),
                ("update", "query_family", 2, 3),
                ("update", "query_family", 3, 3),
                ("update", "family_finalization", 1, 1),
                ("end", "first_evaluation", 1, 1),
            ):
                keep_going = progress_callback(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "kind": kind,
                            "stage": stage,
                            "completed": completed,
                            "total": total,
                            "elapsed_seconds": 0.25,
                            "current_rss_bytes": 256 * 1024**2,
                            "peak_rss_bytes": 384 * 1024**2,
                            "workers": 4,
                            "message": None,
                        }
                    )
                )
                if keep_going is False:
                    raise RuntimeError("warm-up cancelled")
        self._warm_on_the_fly()
        return json.dumps(
            {
                "schema_version": 1,
                "elapsed_seconds": 0.5,
                "query_count": 3,
                "warmed_query_count": 3,
                "current_rss_bytes": 256 * 1024**2,
                "peak_rss_bytes": 384 * 1024**2,
                "already_warm": False,
                "first_evaluation_completed": True,
            }
        )

    def evaluate(self, _momenta: object, **kwargs: object) -> list[object]:
        self._warm_on_the_fly()
        type(self).last_evaluate_options = dict(kwargs)
        return [Decimal("1.25")] if kwargs["precision"] == 32 else [2.0]

    def evaluate_resolved(self, _momenta: object, **kwargs: object) -> SimpleNamespace:
        self._warm_on_the_fly()
        accuracy = self.physics_value.color_accuracy
        color_id = "c0" if accuracy == "lc" else "contracted"
        scalar: object = Decimal("1.25") if kwargs["precision"] == 32 else 2.0
        return SimpleNamespace(
            values=[[[scalar]]],
            helicity_ids=["h0"],
            color_ids=[color_id],
            color_accuracy=accuracy,
        )

    def _benchmark_f64_wall_time(
        self, _momenta: object, _repetitions: int, **kwargs: object
    ) -> float:
        self._warm_on_the_fly()
        type(self).last_benchmark_options = dict(kwargs)
        return 0.25

    def profile(self, _momenta: object, **kwargs: object) -> dict[str, object]:
        self._warm_on_the_fly()
        type(self).last_profile_options = dict(kwargs)
        return {"wall_time_s": 0.25}

    def profile_repeated(
        self, _momenta: object, _repetitions: int, **kwargs: object
    ) -> dict[str, object]:
        self._warm_on_the_fly()
        type(self).last_profile_options = dict(kwargs)
        return {"wall_time_s": 0.5}

    def _profile_arena_repeated(
        self, _momenta: object, _repetitions: int, **kwargs: object
    ) -> dict[str, object]:
        self._warm_on_the_fly()
        type(self).last_arena_profile_options = dict(kwargs)
        return {"wall_time_s": 0.4}

    def set_model_parameters(self, mapping: dict[str, complex | float | int]) -> None:
        self.parameter_updates.append(mapping)

    def clear(self) -> None:
        self.clear_count += 1
        self.otf_process_preparation_count = 0
        self.otf_retained_family_count = 0
        self.otf_retained_selection_count = 0

    def mute_warnings(self) -> None:
        self.muted = True

    def unmute_warnings(self) -> None:
        self.muted = False

    def take_warnings(self) -> list[str]:
        return ["native warning"]


class _PreSelectorNativeRuntime(_NativeRuntime):
    @classmethod
    def load(cls, artifact: Path, **kwargs: object) -> _PreSelectorNativeRuntime:
        cls.load_arguments = (artifact, kwargs)
        return cls()

    def evaluate(
        self,
        _momenta: object,
        *,
        helicities: object = None,
        color_flows: object = None,
        precision: int = 16,
    ) -> list[object]:
        type(self).last_evaluate_options = {
            "helicities": helicities,
            "color_flows": color_flows,
            "precision": precision,
        }
        return [2.0]


class _ExactExecutor:
    def __init__(self, _artifact: Path, _process_id: str, _runtime: object) -> None:
        pass

    def evaluate_resolved(
        self, _momenta: object, **_kwargs: object
    ) -> ResolvedEvaluation:
        accuracy = _NativeRuntime.physics_value.color_accuracy
        color_id = "c0" if accuracy == "lc" else "contracted"
        return ResolvedEvaluation(
            values=(((Decimal("1.25"),),),),
            helicity_ids=("h0",),
            color_ids=(color_id,),
            color_accuracy=accuracy,
        )


def _install_native(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("pyamplicol._rusticol")
    module.Runtime = _NativeRuntime  # type: ignore[attr-defined]
    module.ArtifactError = _NativeArtifactError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _selector_manifest(
    tmp_path: Path,
    *,
    capabilities: tuple[str, ...],
) -> ArtifactManifest:
    physics_path = Path("processes/uux_g/physics.json")
    destination = tmp_path / physics_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "extensions": {
                    "runtime_selectors": {
                        "axes": {
                            "helicity": {"runtime_contract": "complete-reusable"},
                            "color_flow": {"runtime_contract": "complete-reusable"},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifact.json").write_text("{}", encoding="utf-8")
    process = {
        "id": "uux_g",
        "physics_path": physics_path.as_posix(),
        "required_runtime_capabilities": capabilities,
        "aliases": (),
    }
    return ArtifactManifest(
        root=tmp_path,
        kind="pyamplicol-process",
        artifact_id="0" * 64,
        created_utc="2026-07-21T00:00:00Z",
        producer={},
        model={},
        configuration={},
        processes=(process,),
        default_process_id="uux_g",
        runtime={"required_runtime_capabilities": capabilities},
        payloads=(),
        dependencies=(),
        extensions={},
    )


def _load_on_the_fly_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Any:
    _install_native(monkeypatch)
    import pyamplicol.runtime.backend as backend_module

    manifest = _selector_manifest(
        tmp_path,
        capabilities=(
            ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
            ON_THE_FLY_RUNTIME_CAPABILITY,
        ),
    )
    monkeypatch.setattr(
        backend_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )
    _NativeRuntime.execution_mode = "on-the-fly"
    _NativeRuntime.physics_value = _native_physics("lc")
    _NativeRuntime.last_evaluate_options = None
    _NativeRuntime.last_benchmark_options = None
    _NativeRuntime.last_profile_options = None
    _NativeRuntime.last_arena_profile_options = None
    _NativeRuntime.last_warm_up_options = None
    return backend_module.load_runtime_backend(tmp_path, process="uux_g")


def _load_recurrence_companion_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Any:
    _install_native(monkeypatch)
    import pyamplicol.runtime.backend as backend_module

    manifest = _selector_manifest(
        tmp_path,
        capabilities=(
            RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
            RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY,
        ),
    )
    monkeypatch.setattr(
        backend_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )
    _NativeRuntime.execution_mode = "recurrence"
    _NativeRuntime.physics_value = _native_physics("full")
    return backend_module.load_runtime_backend(tmp_path, process="uux_g")


def test_runtime_discovery_does_not_import_native_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module
    imports: list[str] = []

    def tracked_import(name: str, package: str | None = None) -> Any:
        imports.append(name)
        return real_import(name, package)

    monkeypatch.delitem(sys.modules, "pyamplicol._rusticol", raising=False)
    monkeypatch.setattr(importlib, "import_module", tracked_import)
    runtime = real_import("pyamplicol.runtime")

    assert callable(runtime.load_runtime_backend)
    assert "pyamplicol._rusticol" not in imports


def test_adapter_maps_typed_metadata_totals_and_runtime_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    monkeypatch.setattr(
        "pyamplicol.runtime.symbolica_exact.SymbolicaExactExecutor",
        _ExactExecutor,
    )
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.physics_value = _native_physics("lc")
    backend = load_runtime_backend(
        tmp_path,
        process="uux_g",
        model_parameters={"aS": 0.12},
        mute_warnings=True,
    )

    assert isinstance(backend, RuntimeBackend)
    assert isinstance(backend.physics, ProcessPhysics)
    assert isinstance(backend.physics.external_particles[0], ExternalParticle)
    assert isinstance(backend.physics.helicities[0], HelicityConfiguration)
    assert isinstance(backend.physics.color_flows[0], ColorFlow)
    assert isinstance(backend.physics.model_parameters[0], ModelParameter)
    assert backend.artifact_id == "a" * 64
    assert backend.representative_process_key == "uux_g"
    assert backend.external_permutation == (0, 1, 2)
    assert backend.physics.external_particles[0].name == "u"
    assert backend.physics.external_particles[0].pdg_id == 2
    assert backend.physics.external_particles[0].state == "incoming"
    assert backend.physics.selector_capabilities == ("helicity", "color_flow")
    assert backend.evaluate([], precision=16) == (2.0 + 0.0j,)
    assert backend.evaluate([], precision=32) == (Decimal("1.25"),)

    resolved = backend.evaluate_resolved([], precision=32)
    assert isinstance(resolved, ResolvedEvaluation)
    assert resolved.shape == (1, 1, 1)
    assert resolved.total() == (Decimal("1.25"),)

    backend.set_model_parameters({"aS": 0.13})
    backend.clear()
    backend.mute_warnings()
    assert backend._runtime.parameter_updates == [{"aS": 0.13}]
    assert backend._runtime.clear_count == 1
    assert backend._runtime.muted is True
    backend.unmute_warnings()
    assert backend._runtime.muted is False
    assert backend.take_warnings() == ("native warning",)

    path, options = _NativeRuntime.load_arguments or (None, {})
    assert path == tmp_path.resolve()
    assert options == {
        "process": "uux_g",
        "model_parameters": {"aS": 0.12},
        "mute_warnings": True,
    }

    public = Runtime.load(tmp_path, process="uux_g")
    public.clear()
    assert public._backend._runtime.clear_count == 1
    assert public.artifact_id == "a" * 64
    assert public.execution_mode == "compiled"
    assert public.representative_process_key == "uux_g"
    assert public.external_permutation == (0, 1, 2)
    assert isinstance(public.physics, ProcessPhysics)
    assert public.evaluate([], precision=32) == (Decimal("1.25"),)
    for invalid_artifact_id in ("g" * 64, "A" * 64):
        _NativeRuntime.artifact_id = invalid_artifact_id
        with pytest.raises(EvaluationError, match="authenticated artifact identity"):
            _ = public.artifact_id
    _NativeRuntime.artifact_id = "a" * 64


def test_adapter_routes_eager_high_precision_to_eager_exact_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    monkeypatch.setattr(
        "pyamplicol.runtime.eager_exact.EagerExactExecutor",
        _ExactExecutor,
    )
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.execution_mode = "eager"
    try:
        backend = load_runtime_backend(tmp_path, process="uux_g")
        assert backend.evaluate([], precision=32) == (Decimal("1.25"),)
    finally:
        _NativeRuntime.execution_mode = "compiled"


def test_default_runtime_paths_do_not_force_process_physics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    from pyamplicol.runtime import load_runtime_backend

    backend = load_runtime_backend(tmp_path, process="uux_g")

    backend.evaluate([], precision=16)
    backend.evaluate_resolved([], precision=16)
    backend._benchmark_f64_wall_time([], 2)
    backend._profile_arena_repeated([], 2)
    backend.profile([])
    backend.profile_repeated([], 2)
    backend.clear()
    assert backend._runtime.physics_access_count == 0

    backend.evaluate([()], helicity_by_point=("h0",))
    assert backend._runtime.physics_access_count == 1


def test_external_permutation_does_not_materialize_process_physics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_native(monkeypatch)
    from pyamplicol.runtime import load_runtime_backend

    backend = load_runtime_backend(tmp_path, process="uux_g")
    runtime = Runtime(backend)

    assert backend._runtime.physics_access_count == 0
    assert backend.external_count == 3
    assert backend.external_permutation == (0, 1, 2)
    assert runtime.external_permutation == (0, 1, 2)
    assert backend._runtime.physics_access_count == 0


def test_on_the_fly_non_f64_paths_fail_before_selector_or_physics_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = _load_on_the_fly_backend(monkeypatch, tmp_path)
    runtime = Runtime(backend)
    point = ((),)
    try:
        operations = (
            lambda: runtime.evaluate(point, color_flows=("1",), precision=32),
            lambda: runtime.evaluate_with_prec(
                point,
                32,
                color_flows=("1",),
            ),
            lambda: backend.evaluate(
                point,
                helicity_by_point=("h0",),
                precision=32,
            ),
            lambda: runtime.evaluate_resolved(
                point,
                color_flows=("1",),
                precision=32,
            ),
            lambda: runtime.evaluate_resolved_with_prec(
                point,
                32,
                color_flows=("1",),
            ),
            lambda: backend._benchmark_f64_wall_time(
                point,
                2,
                color_flows=("1",),
                precision=32,
            ),
            lambda: backend._profile_arena_repeated(
                point,
                2,
                color_flows=("1",),
                precision=32,
            ),
            lambda: backend.profile(
                point,
                color_flows=("1",),
                precision=32,
            ),
            lambda: backend.profile_repeated(
                point,
                2,
                color_flows=("1",),
                precision=32,
            ),
            lambda: backend.evaluate_profile(
                point,
                color_flows=("1",),
                precision=32,
            ),
            lambda: runtime._diagnostic_project_onshell(point, precision=32),
        )
        for operation in operations:
            with pytest.raises(
                CompatibilityError,
                match=r"on-the-fly execution supports only precision=16",
            ):
                operation()

        native = backend._runtime
        assert native.physics_access_count == 0
        assert native.selector_context_access_count == 0
        assert native.selector_ordinal_access_count == 0
        assert native.runtime_state_census_access_count == 0
        assert native.otf_process_preparation_count == 0
        assert native.otf_retained_family_count == 0
        assert native.otf_retained_selection_count == 0
        assert backend._exact_executor is None
        assert _NativeRuntime.last_evaluate_options is None
        assert _NativeRuntime.last_benchmark_options is None
        assert _NativeRuntime.last_profile_options is None
        assert _NativeRuntime.last_arena_profile_options is None
    finally:
        _NativeRuntime.execution_mode = "compiled"


def test_runtime_inspect_observes_otf_state_without_opening_physics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = _load_on_the_fly_backend(monkeypatch, tmp_path)
    runtime = Runtime(backend)
    try:
        cold = runtime.inspect()
        assert cold["kind"] == "pyamplicol-runtime-inspection"
        assert cold["schema_version"] == 1
        assert cold["artifact_id"] == "a" * 64
        assert cold["supported_precisions"] == (16,)
        metadata = cold["runtime_metadata"]
        assert isinstance(metadata, dict)
        assert metadata["execution_mode"] == "on-the-fly"
        assert metadata["external_count"] == 3
        assert metadata["external_permutation"] == [0, 1, 2]
        assert metadata["on_the_fly_requested_query_construction_threads"] == 8
        assert metadata["on_the_fly_effective_query_construction_threads"] == 4
        cold_state = cold["on_the_fly_state"]
        assert isinstance(cold_state, dict)
        assert cold_state["family_cache_policy"] == "last-family-only"
        assert cold_state["family_cache_limit"] == 1
        assert cold_state["retained_family_count"] == 0
        assert cold_state["retained_selection_count"] == 0

        metadata["external_permutation"][0] = 2
        repeated = runtime.inspect()
        repeated_metadata = repeated["runtime_metadata"]
        assert isinstance(repeated_metadata, dict)
        assert repeated_metadata["external_permutation"] == [0, 1, 2]

        assert runtime.evaluate(((),), precision=16) == (2.0 + 0.0j,)
        warm = runtime.inspect()
        warm_state = warm["on_the_fly_state"]
        assert isinstance(warm_state, dict)
        assert warm_state["process_preparation_count"] == 1
        assert warm_state["retained_family_count"] == 1
        assert warm_state["retained_selection_count"] == 1

        runtime.clear()
        cleared = runtime.inspect()
        cleared_state = cleared["on_the_fly_state"]
        assert isinstance(cleared_state, dict)
        assert cleared_state["process_preparation_count"] == 0
        assert cleared_state["retained_family_count"] == 0
        assert cleared_state["retained_selection_count"] == 0
        assert backend._runtime.physics_access_count == 0
        assert backend._runtime.selector_context_access_count == 0
        assert backend._runtime.selector_ordinal_access_count == 0
        assert backend._runtime.runtime_state_census_access_count == 4
    finally:
        _NativeRuntime.execution_mode = "compiled"


def test_runtime_inspect_observes_recurrence_companion_without_relabeling_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = _load_recurrence_companion_backend(monkeypatch, tmp_path)
    runtime = Runtime(backend)
    try:
        inspected = runtime.inspect()

        assert runtime.execution_mode == "recurrence"
        assert inspected["runtime_metadata"]["execution_mode"] == "recurrence"
        assert "supported_precisions" not in inspected
        companion = inspected["on_the_fly_state"]
        assert isinstance(companion, dict)
        assert companion["owner"] == "recurrence-helicity-selector-companion"
        assert companion["retained_family_count"] == 0
        assert backend._runtime.physics_access_count == 0
        assert backend._runtime.runtime_state_census_access_count == 1
    finally:
        _NativeRuntime.execution_mode = "compiled"


def test_runtime_inspect_does_not_probe_ordinary_recurrence_as_otf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_native(monkeypatch)
    import pyamplicol.runtime.backend as backend_module

    manifest = _selector_manifest(
        tmp_path,
        capabilities=(RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,),
    )
    monkeypatch.setattr(
        backend_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )
    _NativeRuntime.execution_mode = "recurrence"
    try:
        backend = backend_module.load_runtime_backend(tmp_path, process="uux_g")

        assert backend.inspect()["on_the_fly_state"] is None
        assert backend._runtime.runtime_state_census_access_count == 0
    finally:
        _NativeRuntime.execution_mode = "compiled"


def test_adapter_accepts_one_based_color_flow_ordinals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.physics_value = _native_physics("lc")
    _NativeRuntime.last_evaluate_options = None
    backend = load_runtime_backend(tmp_path, process="uux_g")

    backend.evaluate([], color_flows=("1",))

    assert _NativeRuntime.last_evaluate_options is not None
    assert _NativeRuntime.last_evaluate_options["color_flows"] == ("c0",)
    with pytest.raises(EvaluationError, match=r"choose 1\.\.1"):
        backend.evaluate([], color_flows=("2",))


def test_adapter_resolves_per_point_selector_ids_to_native_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.physics_value = _native_physics("lc")
    _NativeRuntime.last_evaluate_options = None
    backend = load_runtime_backend(tmp_path, process="uux_g")

    backend.evaluate(
        [(), ()],
        helicity_by_point=("h0", "h0"),
        color_flow_by_point=("1", "c0"),
    )

    assert _NativeRuntime.last_evaluate_options is not None
    assert _NativeRuntime.last_evaluate_options["helicity_by_point"] == (0, 0)
    assert _NativeRuntime.last_evaluate_options["color_flow_by_point"] == (0, 0)
    with pytest.raises(EvaluationError, match=r"helicity_by_point\[1\]"):
        backend.evaluate(
            [(), ()],
            helicity_by_point=("h0", "missing"),
        )


def test_adapter_requires_selector_capability_for_reusable_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_native(monkeypatch)
    import pyamplicol.runtime.backend as backend_module

    manifest = _selector_manifest(
        tmp_path,
        capabilities=(SYMJIT_F64_RUNTIME_CAPABILITY,),
    )
    manifest_options: list[dict[str, object]] = []

    def load_adapter_manifest(
        _path: Path,
        **options: object,
    ) -> ArtifactManifest:
        manifest_options.append(options)
        return manifest

    monkeypatch.setattr(
        backend_module,
        "load_manifest",
        load_adapter_manifest,
    )

    with pytest.raises(
        CompatibilityError,
        match=r"declares reusable runtime selectors.*does not require",
    ):
        backend_module.load_runtime_backend(tmp_path, process="uux_g")
    assert manifest_options == [{"verify_payloads": False}]


def test_adapter_checks_native_selector_callable_against_capability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _install_native(monkeypatch)
    module.Runtime = _PreSelectorNativeRuntime  # type: ignore[attr-defined]
    import pyamplicol.runtime.backend as backend_module

    capabilities = (
        COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        SYMJIT_F64_RUNTIME_CAPABILITY,
    )
    manifest = _selector_manifest(tmp_path, capabilities=capabilities)
    monkeypatch.setattr(
        backend_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )

    with pytest.raises(
        CompatibilityError,
        match="installed native runtime does not accept per-point selectors",
    ):
        backend_module.load_runtime_backend(tmp_path, process="uux_g")


def test_adapter_accepts_compact_eager_runtime_selector_capability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_native(monkeypatch)
    import pyamplicol.runtime.backend as backend_module

    manifest = _selector_manifest(
        tmp_path,
        capabilities=(EAGER_RUNTIME_LAYOUT_F64_CAPABILITY,),
    )
    monkeypatch.setattr(
        backend_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )
    _NativeRuntime.execution_mode = "eager"
    try:
        backend = backend_module.load_runtime_backend(tmp_path, process="uux_g")
        assert backend.supports_per_point_selectors is True
        backend.evaluate(
            [(), ()],
            helicity_by_point=("h0", "h0"),
            color_flow_by_point=("c0", "c0"),
        )
        assert _NativeRuntime.last_evaluate_options is not None
        assert _NativeRuntime.last_evaluate_options["helicity_by_point"] == (0, 0)
        assert _NativeRuntime.last_evaluate_options["color_flow_by_point"] == (0, 0)
    finally:
        _NativeRuntime.execution_mode = "compiled"


def test_adapter_accepts_on_the_fly_without_reading_dense_selector_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_native(monkeypatch)
    import pyamplicol.runtime.backend as backend_module

    manifest = _selector_manifest(
        tmp_path,
        capabilities=(
            ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
            ON_THE_FLY_RUNTIME_CAPABILITY,
        ),
    )
    monkeypatch.setattr(
        backend_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        backend_module,
        "_has_reusable_runtime_selector_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("on-the-fly adapter read dense selector metadata")
        ),
    )
    _NativeRuntime.execution_mode = "on-the-fly"
    try:
        backend = backend_module.load_runtime_backend(tmp_path, process="uux_g")
        assert backend.execution_mode == "on-the-fly"
        assert backend.supports_per_point_selectors is True
        assert backend.evaluate([()], helicities=("h0",)) == (2.0 + 0.0j,)
        backend.evaluate(
            [(), ()],
            helicity_by_point=("h0", "h0"),
            color_flow_by_point=("c0", "c0"),
        )
        assert _NativeRuntime.last_evaluate_options is not None
        assert _NativeRuntime.last_evaluate_options["helicity_by_point"] == (0, 0)
        assert _NativeRuntime.last_evaluate_options["color_flow_by_point"] == (0, 0)
        backend._benchmark_f64_wall_time(
            [(), ()],
            2,
            helicity_by_point=("h0", "h0"),
            color_flow_by_point=("c0", "c0"),
        )
        backend.profile(
            [(), ()],
            helicity_by_point=("h0", "h0"),
            color_flow_by_point=("c0", "c0"),
        )
        assert backend._runtime.physics_access_count == 0
    finally:
        _NativeRuntime.execution_mode = "compiled"


def test_on_the_fly_warm_up_maps_native_progress_without_dense_physics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = _load_on_the_fly_backend(monkeypatch, tmp_path)
    events: list[object] = []
    try:
        result = Runtime(backend).warm_up(
            ((),),
            helicities=("h0",),
            color_flows=("c0",),
            progress=CallbackProgressSink(events.append),
            n_cores=4,
        )

        assert result == WarmUpResult(
            elapsed_seconds=0.5,
            query_count=3,
            warmed_query_count=3,
            current_rss_bytes=256 * 1024**2,
            peak_rss_bytes=384 * 1024**2,
            already_warm=False,
        )
        assert _NativeRuntime.last_warm_up_options == {
            "momenta": ((),),
            "helicity_ids": ("h0",),
            "color_flow_ids": ("c0",),
            "n_cores": 4,
        }
        assert backend._runtime.physics_access_count == 0
        assert (
            sum(
                isinstance(event, ProgressStart) and event.task_id == "runtime-warm-up"
                for event in events
            )
            == 1
        )
        assert (
            sum(
                isinstance(event, ProgressEnd)
                and event.task_id == "runtime-warm-up"
                and event.success
                for event in events
            )
            == 1
        )
        query_updates = [
            event
            for event in events
            if isinstance(event, ProgressUpdate)
            and event.task_id == "runtime-warm-up:query_family"
        ]
        assert query_updates[-1].completed == 3
        assert query_updates[-1].total == 3
        assert query_updates[-1].details["native_current_rss_bytes"] == (256 * 1024**2)
    finally:
        _NativeRuntime.execution_mode = "compiled"


def test_on_the_fly_warm_up_ignores_terminal_progress_error_after_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = _load_on_the_fly_backend(monkeypatch, tmp_path)
    sink = _FailingWarmUpProgressSink()
    try:
        result = Runtime(backend).warm_up(((),), progress=sink)

        assert result.warmed_query_count == 3
        assert backend._runtime.otf_retained_family_count == 1
        assert any(
            isinstance(event, ProgressEnd)
            and event.task_id == "runtime-warm-up"
            and event.success
            for event in sink.events
        )
    finally:
        _NativeRuntime.execution_mode = "compiled"


def test_on_the_fly_warm_up_cleanup_does_not_mask_callback_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = _load_on_the_fly_backend(monkeypatch, tmp_path)
    sink = _FailingWarmUpProgressSink(fail_during_update=True)
    try:
        with pytest.raises(LookupError, match="progress callback failed"):
            Runtime(backend).warm_up(((),), progress=sink)

        assert backend._runtime.otf_retained_family_count == 0
        assert any(
            isinstance(event, ProgressEnd)
            and event.task_id == "runtime-warm-up"
            and not event.success
            for event in sink.events
        )
    finally:
        _NativeRuntime.execution_mode = "compiled"


@pytest.mark.parametrize("first_evaluation_completed", (None, False))
def test_on_the_fly_warm_up_requires_completed_first_evaluation(
    first_evaluation_completed: bool | None,
) -> None:
    import pyamplicol.runtime.backend as backend_module

    payload: dict[str, object] = {
        "schema_version": 1,
        "elapsed_seconds": 0.5,
        "query_count": 3,
        "warmed_query_count": 3,
        "current_rss_bytes": None,
        "peak_rss_bytes": None,
        "already_warm": False,
    }
    if first_evaluation_completed is not None:
        payload["first_evaluation_completed"] = first_evaluation_completed

    with pytest.raises(ArtifactError, match="first_evaluation_completed"):
        backend_module._warm_up_result(json.dumps(payload))


def test_adapter_accepts_on_the_fly_contracted_color_capability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_native(monkeypatch)
    import pyamplicol.runtime.backend as backend_module

    manifest = _selector_manifest(
        tmp_path,
        capabilities=(
            ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
            ON_THE_FLY_RUNTIME_CAPABILITY,
        ),
    )
    monkeypatch.setattr(
        backend_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )
    _NativeRuntime.execution_mode = "on-the-fly"
    _NativeRuntime.physics_value = _native_physics("full")
    _NativeRuntime.last_warm_up_options = None
    try:
        backend = backend_module.load_runtime_backend(tmp_path, process="uux_g")
        assert backend.execution_mode == "on-the-fly"
        assert backend.physics.color_accuracy == "full"
        assert backend.required_runtime_capabilities == (
            ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
            ON_THE_FLY_RUNTIME_CAPABILITY,
        )
        with pytest.raises(EvaluationError, match="does not expose color-flow"):
            Runtime(backend).warm_up(((),), color_flows=("c0",))
        assert _NativeRuntime.last_warm_up_options is None
    finally:
        _NativeRuntime.execution_mode = "compiled"
        _NativeRuntime.physics_value = _native_physics("lc")


def test_adapter_accepts_on_the_fly_symmetric_group_fft_capability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_native(monkeypatch)
    import pyamplicol.runtime.backend as backend_module

    capabilities = (
        ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
        ON_THE_FLY_RUNTIME_CAPABILITY,
        SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
    )
    manifest = _selector_manifest(tmp_path, capabilities=capabilities)
    monkeypatch.setattr(
        backend_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )
    _NativeRuntime.execution_mode = "on-the-fly"
    _NativeRuntime.physics_value = _native_physics("full")
    try:
        runtime = Runtime.load(tmp_path, process="uux_g")
        assert runtime.execution_mode == "on-the-fly"
        assert runtime.physics.color_accuracy == "full"
        assert runtime._backend.required_runtime_capabilities == capabilities
        assert runtime._backend.supports_per_point_selectors
    finally:
        _NativeRuntime.execution_mode = "compiled"
        _NativeRuntime.physics_value = _native_physics("lc")


def test_compiled_symmetric_group_fft_rejects_exact_precision_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_native(monkeypatch)
    import pyamplicol.runtime.backend as backend_module

    capabilities = (
        COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
    )
    manifest = _selector_manifest(tmp_path, capabilities=capabilities)
    monkeypatch.setattr(
        backend_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )
    _NativeRuntime.execution_mode = "compiled"
    _NativeRuntime.physics_value = _native_physics("full")
    runtime = Runtime.load(tmp_path, process="uux_g")

    assert runtime.evaluate([], precision=16) == (2.0 + 0.0j,)
    with pytest.raises(CompatibilityError, match="only precision=16 native f64"):
        runtime.evaluate([], precision=32)


@pytest.mark.parametrize(
    "capabilities",
    (
        (
            ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
            ON_THE_FLY_RUNTIME_CAPABILITY,
            "unknown.runtime-capability.v1",
        ),
        (
            ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
            ON_THE_FLY_RUNTIME_CAPABILITY,
            SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
            "unknown.runtime-capability.v1",
        ),
        (
            ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
            ON_THE_FLY_RUNTIME_CAPABILITY,
            SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
        ),
        (
            ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
            ON_THE_FLY_RUNTIME_CAPABILITY,
            SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
            SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
        ),
        (
            ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
            SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
        ),
    ),
    ids=(
        "direct-extra",
        "fft-extra",
        "fft-with-lc",
        "duplicate-fft",
        "fft-without-otf",
    ),
)
def test_adapter_rejects_invalid_on_the_fly_capability_sets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capabilities: tuple[str, ...],
) -> None:
    _install_native(monkeypatch)
    import pyamplicol.runtime.backend as backend_module

    manifest = _selector_manifest(tmp_path, capabilities=capabilities)
    monkeypatch.setattr(
        backend_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )
    _NativeRuntime.execution_mode = "on-the-fly"
    _NativeRuntime.physics_value = _native_physics("full")
    try:
        with pytest.raises(
            CompatibilityError,
            match="invalid runtime capability contract",
        ):
            Runtime.load(tmp_path, process="uux_g")
    finally:
        _NativeRuntime.execution_mode = "compiled"
        _NativeRuntime.physics_value = _native_physics("lc")


def test_adapter_does_not_treat_retired_eager_v2_as_selector_support(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_native(monkeypatch)
    import pyamplicol.runtime.backend as backend_module

    manifest = _selector_manifest(
        tmp_path,
        capabilities=(EAGER_DAG_F64_RUNTIME_CAPABILITY,),
    )
    monkeypatch.setattr(
        backend_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )
    _NativeRuntime.execution_mode = "eager"
    try:
        backend = backend_module.load_runtime_backend(tmp_path, process="uux_g")
        assert backend.supports_per_point_selectors is False
    finally:
        _NativeRuntime.execution_mode = "compiled"


def test_adapter_does_not_pass_selector_keywords_when_axes_are_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _install_native(monkeypatch)
    module.Runtime = _PreSelectorNativeRuntime  # type: ignore[attr-defined]
    from pyamplicol.runtime import load_runtime_backend

    _PreSelectorNativeRuntime.last_evaluate_options = None
    backend = load_runtime_backend(tmp_path, process="uux_g")

    assert backend.evaluate(
        [],
        helicities=(),
        color_flows=(),
        helicity_by_point=(),
        color_flow_by_point=(),
    ) == (2.0 + 0.0j,)
    assert _PreSelectorNativeRuntime.last_evaluate_options == {
        "helicities": None,
        "color_flows": None,
        "precision": 16,
    }

    with pytest.raises(CompatibilityError, match="does not support per-point"):
        backend.evaluate([()], helicity_by_point=("h0",))


def test_native_wall_timer_resolves_per_point_selectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.physics_value = _native_physics("lc")
    backend = load_runtime_backend(tmp_path, process="uux_g")
    elapsed = backend._benchmark_f64_wall_time(
        ((), ()),
        3,
        helicity_by_point=("h0", "h0"),
        color_flow_by_point=("1", "c0"),
    )

    assert elapsed == 0.25
    assert _NativeRuntime.last_benchmark_options == {
        "helicities": None,
        "color_flows": None,
        "precision": 16,
        "helicity_by_point": (0, 0),
        "color_flow_by_point": (0, 0),
    }


def test_native_profile_resolves_per_point_selectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.physics_value = _native_physics("lc")
    backend = load_runtime_backend(tmp_path, process="uux_g")
    payload = backend.profile_repeated(
        ((), ()),
        3,
        helicity_by_point=("h0", "h0"),
        color_flow_by_point=("1", "c0"),
    )

    assert payload == {"wall_time_s": 0.5}
    assert _NativeRuntime.last_profile_options == {
        "helicities": None,
        "color_flows": None,
        "precision": 16,
        "include_values": False,
        "helicity_by_point": (0, 0),
        "color_flow_by_point": (0, 0),
    }


def test_private_arena_profiler_never_uses_public_profile_repeated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_native(monkeypatch)
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.physics_value = _native_physics("lc")
    _NativeRuntime.last_profile_options = None
    _NativeRuntime.last_arena_profile_options = None
    backend = load_runtime_backend(tmp_path, process="uux_g")
    payload = backend._profile_arena_repeated(
        ((), ()),
        3,
        helicities=("h0",),
        color_flows=("1",),
    )

    assert payload == {"wall_time_s": 0.4}
    assert _NativeRuntime.last_arena_profile_options == {
        "helicities": ("h0",),
        "color_flows": ("c0",),
        "precision": 16,
        "include_values": False,
    }
    assert _NativeRuntime.last_profile_options is None


def test_adapter_maps_contracted_color_and_native_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.physics_value = _native_physics("full")
    backend = load_runtime_backend(
        tmp_path,
        process=None,
        model_parameters=None,
        mute_warnings=False,
    )

    assert backend.physics.color_flows == ()
    assert isinstance(
        backend.physics.contracted_color_components[0],
        ContractedColorComponent,
    )
    assert backend.evaluate_resolved([], precision=16).color_ids == ("contracted",)
    assert backend.physics.selector_capabilities == ("helicity",)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise _NativeArtifactError("broken artifact")

    backend._runtime.evaluate = fail
    with pytest.raises(ArtifactError, match="broken artifact"):
        backend.evaluate([])
