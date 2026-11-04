# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace

import pytest

import pyamplicol.generation.service as service_module
import pyamplicol.licensing as licensing_module
from pyamplicol.api import ModelSource, ProcessSet
from pyamplicol.api.errors import GenerationError
from pyamplicol.config import (
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    ProcessEntry,
    RunConfig,
)
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.licensing import SymbolicaLicenseState
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


class _RuntimeSchemaStub:
    def to_mapping(self) -> dict[str, object]:
        return {"stages": []}


def _process_artifact_stub(expanded: object) -> SimpleNamespace:
    """The process fields that the artifact writer reads besides payloads."""

    return SimpleNamespace(
        process_id=expanded.request.name,  # type: ignore[attr-defined]
        expression=expanded.request.expression,  # type: ignore[attr-defined]
        external_pdgs=tuple(leg.pdg for leg in expanded.process_ir.legs),  # type: ignore[attr-defined]
    )


def test_parallel_process_phase_overlaps_and_preserves_input_order() -> None:
    rendezvous = Barrier(2)
    state_lock = Lock()
    active = 0
    peak_active = 0

    def operation(item: int) -> str:
        nonlocal active, peak_active
        with state_lock:
            active += 1
            peak_active = max(peak_active, active)
        try:
            if item < 2:
                rendezvous.wait(timeout=2.0)
            time.sleep(0.002 * (4 - item))
            return f"result-{item}"
        finally:
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = service_module._map_process_phase(
            (0, 1, 2, 3),
            operation,
            executor=executor,
            max_in_flight=2,
            phase_name="test phase",
            item_name=lambda item: f"process-{item}",
        )

    assert peak_active == 2
    assert results == ("result-0", "result-1", "result-2", "result-3")


def test_one_generation_worker_runs_process_phase_serially() -> None:
    backend = service_module.GenerationBackend(GenerationConfig(workers=1), None)
    visited: list[int] = []

    results = service_module._map_process_phase(
        (0, 1, 2),
        lambda item: visited.append(item) or item * 2,
        executor=None,
        max_in_flight=backend._process_worker_count(3),
        phase_name="serial test phase",
        item_name=lambda item: f"process-{item}",
    )

    assert backend._process_worker_count(3) == 1
    assert visited == [0, 1, 2]
    assert results == (0, 2, 4)


def test_process_phase_stops_unscheduled_work_and_adds_process_context() -> None:
    slow_started = Event()
    started: list[str] = []
    started_lock = Lock()

    def operation(item: str) -> str:
        with started_lock:
            started.append(item)
        if item == "slow":
            slow_started.set()
            time.sleep(0.05)
            return item
        if item == "bad":
            assert slow_started.wait(timeout=1.0)
            raise RuntimeError("deliberate failure")
        raise AssertionError("unscheduled process ran")

    with (
        ThreadPoolExecutor(max_workers=2) as executor,
        pytest.raises(
            GenerationError,
            match=("test compilation failed for process 'bad': deliberate failure"),
        ),
    ):
        service_module._map_process_phase(
            ("slow", "bad", "never"),
            operation,
            executor=executor,
            max_in_flight=2,
            phase_name="test compilation",
            item_name=str,
        )

    assert sorted(started) == ["bad", "slow"]


def test_symbolica_resources_do_not_nest_generation_worker_fanout() -> None:
    backend = service_module.GenerationBackend(
        RunConfig(
            action="generate",
            generation=GenerationConfig(workers=3),
            evaluator=EvaluatorConfig(
                optimization=EvaluatorOptimizationConfig(cores=4),
                jit=JITConfig(compress=False),
            ),
        ),
        None,
    )

    settings = backend._symbolica_settings()

    assert backend._process_worker_count(5) == 3
    assert settings.n_cores == 4
    assert settings.jit_compress is False
    assert settings.compiled_chunk_compile_workers == 1


@pytest.mark.parametrize("mode", ("compiled", "recurrence", "on-the-fly", "eager"))
def test_generation_symbolica_settings_resolve_python_compression_defaults(
    mode: str,
) -> None:
    backend = service_module.GenerationBackend(
        RunConfig(action="generate", evaluator=EvaluatorConfig(execution_mode=mode)),
        None,
    )
    assert backend._symbolica_settings().jit_compress is (mode == "compiled")


def test_symbolica_materialization_is_process_wide_serialized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = service_module.GenerationBackend(GenerationConfig(workers=2), None)
    state_lock = Lock()
    active = 0
    peak_active = 0

    def materialize(
        process: object,
        _model: object,
        _temporary_root: Path,
        _phase: object,
        _progress: object,
        *,
        backend: str,
    ) -> object:
        assert backend == "JIT"
        nonlocal active, peak_active
        with state_lock:
            active += 1
            peak_active = max(peak_active, active)
        try:
            time.sleep(0.02)
            return process
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(backend, "_materialize_evaluator_unlocked", materialize)
    processes = tuple(
        SimpleNamespace(
            compiled=SimpleNamespace(
                expanded=SimpleNamespace(request=SimpleNamespace(name=f"process-{index}"))
            )
        )
        for index in range(3)
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = tuple(
            executor.map(
                lambda process: backend._materialize_evaluator(
                    process,  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    tmp_path,
                    PhaseHandle("test", None, None),
                ),
                processes,
            )
        )

    assert results == processes
    assert peak_active == 1


def test_restricted_policy_is_applied_before_model_generation_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = service_module.GenerationBackend(
        RunConfig(
            action="generate",
            generation=GenerationConfig(workers=4),
            evaluator=EvaluatorConfig(
                optimization=EvaluatorOptimizationConfig(cores=8)
            ),
        ),
        None,
    )
    observed: list[tuple[object, object]] = []
    monkeypatch.setattr(
        backend,
        "_detect_symbolica_license",
        lambda: SymbolicaLicenseState(licensed=False, restricted=True),
    )

    def stop_at_model_loading(_source: object) -> object:
        observed.append(
            (
                backend._generation_config.workers,
                backend._symbolica_settings().n_cores,
            )
        )
        raise GenerationError("stop after restricted preflight")

    monkeypatch.setattr(backend, "_resolve_model", stop_at_model_loading)

    with pytest.raises(GenerationError, match="stop after restricted preflight"):
        backend.generate(
            ProcessSet.from_expressions(("p p > z",)),
            tmp_path / "artifact",
        )

    assert observed == [(1, 1)]
    assert backend._process_worker_count(50) == 1
    assert backend._configuration.requested.generation.workers == 4
    assert backend._configuration.effective.generation.workers == 1


def test_licensed_multiparticle_expansion_drives_workers_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cpu_budget = 12
    backend = service_module.GenerationBackend(
        RunConfig(
            action="generate",
            process=ProcessConfig(entries=(ProcessEntry("p p > z"),)),
            generation=GenerationConfig(
                workers="auto",
                validation=GenerationValidationConfig(
                    enabled=False,
                    post_build_validation=False,
                ),
            ),
            evaluator=EvaluatorConfig(
                execution_mode="compiled",
                optimization=EvaluatorOptimizationConfig(cores="auto")
            ),
        ),
        None,
    )
    monkeypatch.setattr(
        backend,
        "_detect_symbolica_license",
        lambda: SymbolicaLicenseState(licensed=True, restricted=False),
    )
    monkeypatch.setattr(licensing_module, "_cpu_budget", lambda: cpu_budget)
    monkeypatch.setattr(
        backend,
        "_artifact_model",
        lambda _resolved: SimpleNamespace(name="test-model"),
    )

    expansion_calls = 0
    expand_request = backend._expand_request

    def track_expansion(
        request: object,
        resolved_model: object,
    ) -> object:
        nonlocal expansion_calls
        expansion_calls += 1
        return expand_request(request, resolved_model)  # type: ignore[arg-type]

    monkeypatch.setattr(backend, "_expand_request", track_expansion)
    monkeypatch.setattr(
        backend,
        "_compile_for_generation",
        lambda entry, _model, _phase: SimpleNamespace(expanded=entry),
    )
    monkeypatch.setattr(
        backend,
        "_prepare_warmup_process",
        lambda process, _model, **_kwargs: SimpleNamespace(
            expanded=process.expanded,
            filters={},
            validation_points=(),
        ),
    )
    schema = _RuntimeSchemaStub()
    monkeypatch.setattr(
        backend,
        "_construct_evaluator",
        lambda process, _model, _phase: SimpleNamespace(
            compiled=process,
            runtime_schema=schema,
            helicity_sum_runtime_schema=None,
            helicity_selector_lanes=(),
            color_selector_lanes=(),
        ),
    )
    generation_thread = get_ident()
    materialization_threads: list[int] = []

    def materialize_on_generation_thread(
        process: object,
        _model: object,
        _root: Path,
        _phase: object,
    ) -> SimpleNamespace:
        materialization_threads.append(get_ident())
        return _process_artifact_stub(process.compiled.expanded)  # type: ignore[union-attr]

    monkeypatch.setattr(
        backend,
        "_materialize_evaluator",
        materialize_on_generation_thread,
    )

    worker_counts: list[int] = []
    real_executor = service_module.ThreadPoolExecutor

    def capture_executor(
        *,
        max_workers: int,
        thread_name_prefix: str,
    ) -> ThreadPoolExecutor:
        worker_counts.append(max_workers)
        return real_executor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )

    monkeypatch.setattr(service_module, "ThreadPoolExecutor", capture_executor)
    captured: dict[str, object] = {}

    def capture_artifact_configuration(
        destination: Path,
        **kwargs: object,
    ) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            output=Path(destination),
            files=(),
            api_bundle_path=None,
        )

    monkeypatch.setattr(
        service_module,
        "write_schema_v3_artifact",
        capture_artifact_configuration,
    )

    result = backend.generate(
        ProcessSet.from_expressions(("p p > z",)),
        tmp_path / "artifact",
    )

    assert expansion_calls == 1
    assert len(result.processes.requests) == 5
    assert worker_counts == [5]
    assert materialization_threads == [generation_thread] * 5
    settings = backend._symbolica_settings()
    assert settings.n_cores == 2
    assert worker_counts[0] * settings.n_cores <= cpu_budget

    provenance = captured["configuration"]
    requested = provenance.requested  # type: ignore[union-attr]
    effective = provenance.effective  # type: ignore[union-attr]
    adjustments = provenance.adjustments  # type: ignore[union-attr]
    assert requested.generation.workers == "auto"
    assert requested.evaluator.optimization.cores == "auto"
    assert effective.generation.workers == 5
    assert effective.evaluator.optimization.cores == 2
    assert [adjustment.reason for adjustment in adjustments] == [
        "shared affinity-aware CPU budget for concurrent process generation",
        "shared affinity-aware CPU budget for Symbolica evaluator work",
    ]


def test_generation_prunes_unsupported_multiparticle_tree_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = service_module.GenerationBackend(
        RunConfig(
            action="generate",
            generation=GenerationConfig(
                workers=1,
                validation=GenerationValidationConfig(
                    enabled=False,
                    post_build_validation=False,
                ),
            ),
            evaluator=EvaluatorConfig(execution_mode="compiled"),
        ),
        None,
    )
    model = BuiltinSMModel()
    resolved = service_module._ResolvedModel(
        source=ModelSource.built_in_sm(),
        model=model,
        use_compiled_process_catalog=False,
    )
    monkeypatch.setattr(
        backend,
        "_detect_symbolica_license",
        lambda: SymbolicaLicenseState(licensed=True, restricted=False),
    )
    monkeypatch.setattr(backend, "_resolve_model", lambda _source: resolved)
    monkeypatch.setattr(
        backend,
        "_artifact_model",
        lambda _resolved: SimpleNamespace(name="test-model"),
    )
    monkeypatch.setattr(
        backend,
        "_expand_request",
        lambda _request, _resolved: (
            build_process_ir("d d~ > z"),
            build_process_ir("g g > z g g"),
        ),
    )
    monkeypatch.setattr(
        backend,
        "_prepare_warmup_process",
        lambda process, _model, **_kwargs: SimpleNamespace(
            expanded=process.expanded,
            validation_points=(),
            filters={},
        ),
    )
    schema = _RuntimeSchemaStub()
    monkeypatch.setattr(
        backend,
        "_construct_evaluator",
        lambda process, _model, _phase: SimpleNamespace(
            compiled=process,
            runtime_schema=schema,
            helicity_sum_runtime_schema=None,
            helicity_selector_lanes=(),
            color_selector_lanes=(),
        ),
    )
    monkeypatch.setattr(
        backend,
        "_materialize_evaluator",
        lambda process, _model, _root, _phase: _process_artifact_stub(
            process.compiled.expanded
        ),
    )
    captured: dict[str, object] = {}

    def capture_artifact(destination: Path, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            output=Path(destination),
            files=(),
            api_bundle_path=None,
        )

    monkeypatch.setattr(
        service_module,
        "write_schema_v3_artifact",
        capture_artifact,
    )

    result = backend.generate(
        ProcessSet.from_expressions(("p p > z j j",)),
        tmp_path / "artifact",
    )

    assert [request.name for request in result.processes.requests] == [
        "p_p_to_z_j_j_1"
    ]
    assert [
        process.process_id for process in captured["processes"]  # type: ignore[union-attr]
    ] == ["p_p_to_z_j_j_1"]
    assert "g g > z g g" not in {
        request.expression for request in result.processes.requests
    }
    assert (
        "Skipped 1 concrete subprocess with no model-supported tree-level "
        "amplitudes: p_p_to_z_j_j_2 (g g > z g g)"
    ) in caplog.text


def test_eager_generation_never_enters_compiled_evaluator_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = service_module.GenerationBackend(
        RunConfig(
            action="generate",
            generation=GenerationConfig(
                workers=1,
                validation=GenerationValidationConfig(
                    enabled=False,
                    post_build_validation=False,
                ),
            ),
            evaluator=EvaluatorConfig(execution_mode="eager"),
        ),
        None,
    )
    monkeypatch.setattr(
        backend,
        "_detect_symbolica_license",
        lambda: SymbolicaLicenseState(licensed=True, restricted=False),
    )
    monkeypatch.setattr(backend, "_require_eager_kernel_pack", lambda _model: None)
    monkeypatch.setattr(
        backend, "_apply_prepared_kernel_pack_policy", lambda _model: None
    )
    monkeypatch.setattr(
        backend,
        "_artifact_model",
        lambda _resolved: SimpleNamespace(name="prepared-test-model"),
    )
    monkeypatch.setattr(
        backend,
        "_compile_for_generation",
        lambda entry, _model, _phase: SimpleNamespace(expanded=entry),
    )
    monkeypatch.setattr(
        backend,
        "_prepare_warmup_process",
        lambda process, _model, **_kwargs: SimpleNamespace(
            expanded=process.expanded,
            filters={},
            validation_points=(),
        ),
    )
    eager_processes: list[str] = []

    def construct_eager(process: object, *_args: object) -> SimpleNamespace:
        eager_processes.append(process.expanded.request.name)  # type: ignore[union-attr]
        return _process_artifact_stub(process.expanded)  # type: ignore[union-attr]

    monkeypatch.setattr(backend, "_construct_eager_artifact", construct_eager)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("eager generation entered the compiled evaluator lane")

    monkeypatch.setattr(backend, "_construct_evaluator", forbidden)
    monkeypatch.setattr(backend, "_materialize_evaluator", forbidden)
    monkeypatch.setattr(
        service_module,
        "build_and_write_generic_stage_evaluator_artifacts",
        forbidden,
    )
    captured: dict[str, object] = {}

    def capture_artifact(destination: Path, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            output=Path(destination),
            files=(),
            api_bundle_path=None,
        )

    monkeypatch.setattr(service_module, "write_schema_v3_artifact", capture_artifact)
    monkeypatch.setattr(backend, "_validate_generated_artifact", lambda *_a, **_k: None)

    result = backend.generate(
        ProcessSet.from_expressions(("d d~ > z",)),
        tmp_path / "artifact",
    )

    assert eager_processes == ["d_dbar_to_z"]
    assert [process.process_id for process in captured["processes"]] == [  # type: ignore[union-attr]
        "d_dbar_to_z"
    ]
    assert result.output == tmp_path / "artifact"
