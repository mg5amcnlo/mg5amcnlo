# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

import pyamplicol.api.services as service_module
from pyamplicol.api import (
    BenchmarkResult,
    BenchmarkStatistics,
    ColorFlow,
    CompatibilityError,
    EvaluationError,
    Generator,
    HelicityConfiguration,
    PhysicsReduction,
    ProcessPhysics,
    ProcessSet,
    ReductionGroup,
    ResolvedEvaluation,
    Runtime,
    WarmUpResult,
)
from pyamplicol.api.results import GenerationPlan
from pyamplicol.config import BenchmarkConfig, GenerationConfig


class _GeneratorBackend:
    def __init__(self) -> None:
        self.processes: ProcessSet | None = None

    def plan(self, processes: ProcessSet, *, model: object = None) -> GenerationPlan:
        del model
        self.processes = processes
        return GenerationPlan(
            concrete_processes=processes.requests,
            estimated_coverage={"fraction": 1.0},
            requested_settings=GenerationConfig(),
            effective_settings=GenerationConfig(),
        )

    def generate(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("not used")


class _RuntimeBackend:
    muted = False

    def __init__(self) -> None:
        self.selectors: dict[str, object] = {}
        self.clear_count = 0

    @property
    def physics(self) -> ProcessPhysics:
        return ProcessPhysics(
            "test",
            "d d~ > z",
            "lc",
            "all",
            "all",
            "flow",
            0,
            (),
            (HelicityConfiguration("h0", 0, (1, -1), True, False, "h0", 1.0),),
            (ColorFlow("c0", 0, (1,), True, "c0", 1.0),),
            (),
            PhysicsReduction(
                "lc-diagonal",
                (ReductionGroup("g0", "h0", "c0", ("h0",), ("c0",)),),
            ),
            (),
            ("helicity",),
        )

    def evaluate(self, momenta: object, **selectors: object) -> tuple[complex, ...]:
        del momenta
        self.selectors = selectors
        return (3.0 + 0j,)

    def evaluate_resolved(
        self, momenta: object, **selectors: object
    ) -> ResolvedEvaluation:
        del momenta, selectors
        return ResolvedEvaluation((((3.0 + 0j,),),), ("h0",), ("c0",))

    def set_model_parameters(self, mapping: object) -> None:
        self.parameters = mapping

    def clear(self) -> None:
        self.clear_count += 1

    def mute_warnings(self) -> None:
        self.muted = True

    def unmute_warnings(self) -> None:
        self.muted = False


class _PreSelectorRuntimeBackend(_RuntimeBackend):
    def evaluate(
        self,
        momenta: object,
        *,
        helicities: tuple[str, ...] | None = None,
        color_flows: tuple[str, ...] | None = None,
        precision: int = 16,
    ) -> tuple[complex, ...]:
        del momenta
        self.selectors = {
            "helicities": helicities,
            "color_flows": color_flows,
            "precision": precision,
        }
        return (3.0 + 0j,)


class _DiagnosticRuntimeBackend(_RuntimeBackend):
    def _diagnostic_project_onshell(
        self, momenta: object, *, precision: int
    ) -> tuple[object, dict[str, object]]:
        self.diagnostic_call = (momenta, precision)
        return momenta, {"unchanged": True, "projected_digest": "a" * 64}


class _WarmUpRuntimeBackend(_RuntimeBackend):
    execution_mode = "on-the-fly"

    def warm_up(self, momenta: object, **options: object) -> WarmUpResult:
        self.warm_up_call = (momenta, options)
        return WarmUpResult(0.5, 7, 7, None, None, False)


class _BenchmarkBackend:
    def run(self, target: object, *, points: object = None) -> BenchmarkResult:
        del target, points
        config = BenchmarkConfig(target_runtime=0.1)
        return BenchmarkResult(
            requested_config=config,
            effective_config=config,
            sample_count=5,
            wall_time_per_point=1e-6,
            evaluator_time_per_point=8e-7,
            uncertainty=BenchmarkStatistics(1e-7, 5e-8, 0.05),
            environment={"python": "test"},
        )


def test_generator_backend_is_created_only_on_first_operation(
    monkeypatch: object,
) -> None:
    calls: list[object] = []
    backend = _GeneratorBackend()

    def factory(config: object, progress: object) -> _GeneratorBackend:
        calls.append((config, progress))
        return backend

    monkeypatch.setattr(service_module, "_generator_factory", factory)  # type: ignore[attr-defined]
    generator = Generator(GenerationConfig())
    assert calls == []
    plan = generator.plan("d d~ > z g")
    assert len(calls) == 1
    assert plan.concrete_processes[0].expression == "d d~ > z g"


def test_generator_rejects_existing_output_before_backend_construction(
    monkeypatch: object, tmp_path: Path
) -> None:
    destination = tmp_path / "artifact"
    destination.mkdir()
    calls: list[object] = []

    def factory(config: object, progress: object) -> _GeneratorBackend:
        calls.append((config, progress))
        return _GeneratorBackend()

    monkeypatch.setattr(service_module, "_generator_factory", factory)  # type: ignore[attr-defined]
    generator = Generator(GenerationConfig())
    with pytest.raises(FileExistsError, match="artifact already exists"):
        generator.generate("d d~ > z g", destination)
    assert calls == []


def test_runtime_and_benchmark_facades_use_typed_backends(
    monkeypatch: object, tmp_path: Path
) -> None:
    backend = _RuntimeBackend()
    loaded: list[Path] = []

    def loader(path: Path, **kwargs: object) -> _RuntimeBackend:
        del kwargs
        loaded.append(path)
        return backend

    monkeypatch.setattr(service_module, "_runtime_loader", loader)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        service_module,
        "_benchmark_factory",
        lambda config, progress: _BenchmarkBackend(),
    )
    runtime = Runtime.load(tmp_path / "artifact")
    assert loaded[0].is_absolute()
    assert runtime.evaluate([[[[1.0, 0.0, 0.0, 1.0]]]]) == (3.0 + 0j,)
    assert runtime.evaluate_resolved([]).total() == (3.0 + 0j,)
    runtime.mute_warnings()
    assert backend.muted
    runtime.clear()
    assert backend.clear_count == 1
    result = service_module.BenchmarkRunner(BenchmarkConfig()).run(runtime)
    assert result.sample_count == 5


def test_runtime_forwards_ufo_style_complex_parameter_pairs(
    monkeypatch: object, tmp_path: Path
) -> None:
    backend = _RuntimeBackend()
    received: list[object] = []

    def loader(path: Path, **kwargs: object) -> _RuntimeBackend:
        del path
        received.append(kwargs["model_parameters"])
        return backend

    monkeypatch.setattr(service_module, "_runtime_loader", loader)  # type: ignore[attr-defined]

    Runtime.load(
        tmp_path / "artifact",
        model_parameters={"aS": [0.117, 0.0], "complex": [2.5, -0.25]},
    )

    assert received == [{"aS": [0.117, 0.0], "complex": [2.5, -0.25]}]


def test_runtime_accepts_typed_physics_selectors(
    monkeypatch: object, tmp_path: Path
) -> None:
    backend = _RuntimeBackend()
    monkeypatch.setattr(  # type: ignore[attr-defined]
        service_module,
        "_runtime_loader",
        lambda path, **kwargs: backend,
    )
    runtime = Runtime.load(tmp_path / "artifact")
    physics = runtime.physics
    runtime.evaluate(
        [[[[1.0, 0.0, 0.0, 1.0]]]],
        helicities=(physics.helicities[0],),
        color_flows=(physics.color_flows[0],),
    )
    assert backend.selectors["helicities"] == ("h0",)
    assert backend.selectors["color_flows"] == ("c0",)


def test_runtime_accepts_typed_per_point_selectors() -> None:
    backend = _RuntimeBackend()
    runtime = Runtime(backend)
    physics = runtime.physics

    runtime.evaluate(
        [[[[1.0, 0.0, 0.0, 1.0]]]],
        helicity_by_point=(physics.helicities[0],),
        color_flow_by_point=(physics.color_flows[0],),
    )

    assert backend.selectors["helicity_by_point"] == ("h0",)
    assert backend.selectors["color_flow_by_point"] == ("c0",)


def test_runtime_forwards_private_diagnostic_projection() -> None:
    backend = _DiagnosticRuntimeBackend()
    runtime = Runtime(backend)
    points = ((((1.0, 0.0, 0.0, 1.0),),),)

    projected, metadata = runtime._diagnostic_project_onshell(
        points,
        precision=200,
    )

    assert projected is points
    assert metadata == {"unchanged": True, "projected_digest": "a" * 64}
    assert backend.diagnostic_call == (points, 200)


def test_runtime_preserves_pre_selector_backend_compatibility() -> None:
    backend = _PreSelectorRuntimeBackend()
    runtime = Runtime(backend)

    assert runtime.evaluate([], helicities=("h0",)) == (3.0 + 0j,)
    assert backend.selectors == {
        "helicities": ("h0",),
        "color_flows": None,
        "precision": 16,
    }
    assert runtime.evaluate([], helicity_by_point=()) == (3.0 + 0j,)

    with pytest.raises(EvaluationError, match="does not support per-point"):
        runtime.evaluate([[]], helicity_by_point=("h0",))


def test_runtime_warm_up_is_one_point_native_f64_only() -> None:
    backend = _WarmUpRuntimeBackend()
    runtime = Runtime(backend)
    point = ((1.0, 0.0, 0.0, 1.0),)

    result = runtime.warm_up((point,), helicities=("h0",), color_flows=("c0",))

    assert result == WarmUpResult(0.5, 7, 7, None, None, False)
    assert backend.warm_up_call == (
        (point,),
        {
            "helicities": ("h0",),
            "color_flows": ("c0",),
            "precision": 16,
            "progress": None,
            "n_cores": None,
        },
    )
    with pytest.raises(ValueError, match="exactly one"):
        runtime.warm_up(())
    with pytest.raises(ValueError, match="exactly one"):
        runtime.warm_up((point, point))
    with pytest.raises(CompatibilityError, match="only precision=16"):
        runtime.warm_up((point,), precision=200)
    backend.execution_mode = "compiled"
    with pytest.raises(CompatibilityError, match="only for on-the-fly"):
        runtime.warm_up((point,))


def test_runtime_warm_up_forwards_per_call_core_override() -> None:
    backend = _WarmUpRuntimeBackend()
    runtime = Runtime(backend)
    point = ((1.0, 0.0, 0.0, 1.0),)

    runtime.warm_up((point,), n_cores=4)
    assert backend.warm_up_call[1]["n_cores"] == 4
    runtime.warm_up((point,))
    assert backend.warm_up_call[1]["n_cores"] is None


@pytest.mark.parametrize("n_cores", [True, False, 1.5, "4"])
def test_runtime_warm_up_rejects_noninteger_core_count(n_cores: object) -> None:
    backend = _WarmUpRuntimeBackend()
    with pytest.raises(TypeError, match="n_cores must be a positive integer"):
        Runtime(backend).warm_up(((),), n_cores=n_cores)  # type: ignore[arg-type]
    assert not hasattr(backend, "warm_up_call")


@pytest.mark.parametrize("n_cores", [0, -1])
def test_runtime_warm_up_rejects_nonpositive_core_count(n_cores: int) -> None:
    backend = _WarmUpRuntimeBackend()
    with pytest.raises(ValueError, match="n_cores must be a positive integer"):
        Runtime(backend).warm_up(((),), n_cores=n_cores)
    assert not hasattr(backend, "warm_up_call")


def test_runtime_normalizes_empty_selector_sequences_to_omitted_axes() -> None:
    backend = _RuntimeBackend()
    runtime = Runtime(backend)

    runtime.evaluate(
        [],
        helicities=(),
        color_flows=(),
        helicity_by_point=(),
        color_flow_by_point=(),
    )

    assert backend.selectors == {
        "helicities": None,
        "color_flows": None,
        "precision": 16,
    }

    runtime.evaluate([], helicities=(), helicity_by_point=("h0",))
    assert backend.selectors["helicity_by_point"] == ("h0",)


def test_runtime_rejects_global_and_per_point_selector_on_same_axis() -> None:
    runtime = Runtime(_RuntimeBackend())

    with pytest.raises(ValueError, match="helicities and helicity_by_point"):
        runtime.evaluate([], helicities=("h0",), helicity_by_point=("h0",))
    with pytest.raises(ValueError, match="color_flows and color_flow_by_point"):
        runtime.evaluate([], color_flows=("c0",), color_flow_by_point=("c0",))


def test_runtime_identity_is_optional_for_backend_conformance_but_fails_closed() -> (
    None
):
    runtime = Runtime(_RuntimeBackend())

    assert runtime.external_permutation == ()
    with pytest.raises(EvaluationError, match="authenticated artifact identity"):
        _ = runtime.artifact_id
    with pytest.raises(EvaluationError, match="valid execution mode"):
        _ = runtime.execution_mode
    with pytest.raises(CompatibilityError, match="compact runtime inspection"):
        runtime.inspect()


def test_runtime_validates_legacy_permutation_against_physics_count() -> None:
    backend = _RuntimeBackend()
    backend.external_permutation = ()
    runtime = Runtime(backend)

    assert runtime.external_permutation == ()
    backend.external_permutation = (0,)
    with pytest.raises(CompatibilityError, match="external permutation is invalid"):
        _ = runtime.external_permutation


def test_runtime_prefers_authenticated_compact_external_count() -> None:
    backend = _RuntimeBackend()
    backend.external_count = 1
    backend.external_permutation = (0,)

    assert Runtime(backend).external_permutation == (0,)


@pytest.mark.parametrize("precision", (True, False, 1.5, "32"))
def test_runtime_rejects_non_integer_precision(precision: object) -> None:
    runtime = Runtime(_RuntimeBackend())

    with pytest.raises(TypeError, match="positive integer"):
        runtime.evaluate([], precision=precision)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="positive integer"):
        runtime.evaluate_resolved([], precision=precision)  # type: ignore[arg-type]


@pytest.mark.parametrize("precision", (0, -1))
def test_runtime_rejects_non_positive_precision(precision: int) -> None:
    runtime = Runtime(_RuntimeBackend())

    with pytest.raises(ValueError, match="positive integer"):
        runtime.evaluate([], precision=precision)
    with pytest.raises(ValueError, match="positive integer"):
        runtime.evaluate_resolved([], precision=precision)


def test_resolved_evaluation_preserves_decimal_precision() -> None:
    resolved = ResolvedEvaluation(
        (((Decimal("1.25"), Decimal("2.75")),),),
        ("h0",),
        ("c0", "c1"),
    )
    assert resolved.total() == (Decimal("4.00"),)
    assert resolved.color_accuracy == resolved.accuracy == "lc"
