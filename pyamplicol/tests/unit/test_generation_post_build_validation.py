# SPDX-License-Identifier: 0BSD
from types import SimpleNamespace

import pytest

import pyamplicol.generation.service as generation_service
from pyamplicol.api import Runtime
from pyamplicol.api.errors import ArtifactError, GenerationError
from pyamplicol.config import (
    Action,
    EvaluatorConfig,
    GenerationConfig,
    GenerationValidationConfig,
    RunConfig,
)
from pyamplicol.generation.service import GenerationBackend
from pyamplicol.generation.validation import ValidationPointRecord


def _helicity(
    identifier: str,
    *,
    computed: bool = True,
    structural_zero: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=identifier,
        computed=computed,
        structural_zero=structural_zero,
    )


def _color_flow(identifier: str, *, computed: bool = True) -> SimpleNamespace:
    return SimpleNamespace(id=identifier, computed=computed)


def _on_the_fly_backend(*, validation_enabled: bool = True) -> GenerationBackend:
    return GenerationBackend(
        RunConfig(
            action=Action.GENERATE,
            generation=GenerationConfig(
                validation=GenerationValidationConfig(
                    enabled=validation_enabled,
                    post_build_validation=True,
                )
            ),
            evaluator=EvaluatorConfig(execution_mode="on-the-fly"),
        ),
        None,
    )


def _compact_manifest(*, artifact_id: str = "a" * 64) -> SimpleNamespace:
    return SimpleNamespace(
        artifact_id=artifact_id,
        processes=({"id": "process"},),
        payloads=tuple(
            SimpleNamespace(path=f"processes/process/{name}")
            for name in (
                "physics.json",
                "execution.json",
                "validation-momenta.json",
                "on-the-fly-runtime.pacbin",
            )
        ),
        runtime={"api_bundle_path": None},
    )


class _CompactRuntime:
    artifact_id = "a" * 64
    execution_mode = "on-the-fly"
    representative_process_key = "process"

    def __init__(
        self,
        *,
        total: complex = 1.0 + 0.0j,
        resolved: complex | None = None,
    ) -> None:
        self.total = total
        self.resolved = total if resolved is None else resolved
        self.calls: list[
            tuple[str, int, tuple[str, ...] | None, tuple[str, ...] | None]
        ] = []

    @property
    def physics(self) -> object:
        raise AssertionError("compact validation opened dense process physics")

    def evaluate(
        self,
        samples: tuple[object, ...],
        *,
        helicities: tuple[str, ...] | None = None,
        color_flows: tuple[str, ...] | None = None,
    ) -> tuple[complex, ...]:
        sample_count = len(samples)
        self.calls.append(("total", sample_count, helicities, color_flows))
        return (self.total,) * sample_count

    def evaluate_resolved(
        self,
        samples: tuple[object, ...],
        *,
        helicities: tuple[str, ...] | None = None,
        color_flows: tuple[str, ...] | None = None,
    ) -> object:
        sample_count = len(samples)
        self.calls.append(("resolved", sample_count, helicities, color_flows))
        values = (self.resolved,) * sample_count
        return SimpleNamespace(total=lambda: values)


_COMPACT_VALIDATION_SELECTORS = {
    "process": {
        "helicities": ("h:-1,+1,+0",),
        "color_flows": ("1",),
    }
}


def _validation_point() -> ValidationPointRecord:
    return ValidationPointRecord(
        process_id="process",
        process="d d~ > z",
        seed=1,
        particles=(
            (1, (500.0, 0.0, 0.0, 500.0)),
            (-1, (500.0, 0.0, 0.0, -500.0)),
            (23, (1000.0, 0.0, 0.0, 0.0)),
        ),
    )


def _install_compact_validation_spies(
    monkeypatch: pytest.MonkeyPatch,
    runtime: object,
    *,
    manifest: object | None = None,
) -> None:
    monkeypatch.setattr(
        Runtime,
        "load",
        classmethod(lambda _cls, _output, *, process: runtime),
    )
    monkeypatch.setattr(
        "pyamplicol.artifacts.load_manifest",
        lambda _output: manifest or _compact_manifest(),
    )
    monkeypatch.setattr(
        generation_service,
        "_post_build_validation_slices",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("compact validation entered dense validation slicing")
        ),
    )


def test_post_build_validation_keeps_small_resolved_axes_complete() -> None:
    physics = SimpleNamespace(
        helicities=(_helicity("h:0"), _helicity("h:1")),
        color_flows=(_color_flow("flow:0"), _color_flow("flow:1")),
        contracted_color_components=(),
    )

    assert generation_service._post_build_validation_slices(physics, 10) == (
        ("complete", (), ()),
    )


def test_post_build_validation_slices_large_helicity_color_product(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        generation_service,
        "_MAX_POST_BUILD_RESOLVED_COMPONENTS",
        100,
    )
    physics = SimpleNamespace(
        helicities=tuple(
            [
                _helicity("h:zero", structural_zero=True),
                _helicity("h:representative"),
            ]
            + [_helicity(f"h:{index}", computed=False) for index in range(10)]
        ),
        color_flows=tuple(
            [_color_flow("flow:representative")]
            + [_color_flow(f"flow:{index}", computed=False) for index in range(14)]
        ),
        contracted_color_components=(),
    )

    assert generation_service._post_build_validation_slices(physics, 1) == (
        ("selected-helicity", ("h:representative",), ()),
        ("selected-flow", (), ("flow:representative",)),
    )


def test_post_build_validation_always_bounds_each_slice(monkeypatch) -> None:
    monkeypatch.setattr(
        generation_service,
        "_MAX_POST_BUILD_RESOLVED_COMPONENTS",
        5,
    )
    physics = SimpleNamespace(
        helicities=tuple(_helicity(f"h:{index}") for index in range(3)),
        color_flows=tuple(_color_flow(f"flow:{index}") for index in range(3)),
        contracted_color_components=(),
    )

    assert generation_service._post_build_validation_slices(physics, 2) == (
        ("selected-helicity-and-flow", ("h:0",), ("flow:0",)),
    )


def test_post_build_validation_never_evaluates_large_axes_unselected(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        generation_service,
        "_MAX_POST_BUILD_RESOLVED_COMPONENTS",
        100,
    )
    physics = SimpleNamespace(
        process_id="process",
        helicities=tuple(_helicity(f"h:{index}") for index in range(12)),
        color_flows=tuple(_color_flow(f"flow:{index}") for index in range(15)),
        contracted_color_components=(),
    )
    calls: list[tuple[str, tuple[str, ...] | None, tuple[str, ...] | None]] = []

    class _Runtime:
        def __init__(self) -> None:
            self.physics = physics

        def evaluate(self, samples, *, helicities=None, color_flows=None):
            calls.append(("total", helicities, color_flows))
            return tuple(1.0 + 0.0j for _sample in samples)

        def evaluate_resolved(self, samples, *, helicities=None, color_flows=None):
            calls.append(("resolved", helicities, color_flows))
            values = tuple(1.0 + 0.0j for _sample in samples)
            return SimpleNamespace(total=lambda: values)

    runtime = _Runtime()
    monkeypatch.setattr(
        Runtime,
        "load",
        classmethod(lambda _cls, _output, *, process: runtime),
    )
    manifest = SimpleNamespace(
        processes=({"id": "process"},),
        payloads=tuple(
            SimpleNamespace(path=f"processes/process/{name}")
            for name in (
                "physics.json",
                "execution.json",
                "validation-momenta.json",
            )
        ),
        runtime={"api_bundle_path": None},
    )
    monkeypatch.setattr(
        "pyamplicol.artifacts.load_manifest",
        lambda _output: manifest,
    )
    point = ValidationPointRecord(
        process_id="process",
        process="d d~ > z",
        seed=1,
        particles=(
            (1, (500.0, 0.0, 0.0, 500.0)),
            (-1, (500.0, 0.0, 0.0, -500.0)),
            (23, (1000.0, 0.0, 0.0, 0.0)),
        ),
    )

    GenerationBackend(GenerationConfig(), None)._validate_generated_artifact(
        tmp_path,
        ("process",),
        validation_points={"process": (point,)},
        expected_api_bundle_path=None,
    )

    assert calls == [
        ("total", ("h:0",), None),
        ("resolved", ("h:0",), None),
        ("total", None, ("flow:0",)),
        ("resolved", None, ("flow:0",)),
    ]


def test_on_the_fly_post_build_validation_runs_one_compact_numerical_sample(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    runtime = _CompactRuntime()
    _install_compact_validation_spies(monkeypatch, runtime)
    point = _validation_point()

    _on_the_fly_backend()._validate_generated_artifact(
        tmp_path,
        ("process",),
        validation_points={"process": (point, point)},
        expected_api_bundle_path=None,
        compact_validation_selectors=_COMPACT_VALIDATION_SELECTORS,
    )

    assert runtime.calls == [
        ("total", 1, ("h:-1,+1,+0",), ("1",)),
        ("resolved", 1, ("h:-1,+1,+0",), ("1",)),
    ]


def test_on_the_fly_post_build_validation_rejects_total_resolved_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    runtime = _CompactRuntime(resolved=2.0 + 0.0j)
    _install_compact_validation_spies(monkeypatch, runtime)

    with pytest.raises(GenerationError, match="does not reduce to the total"):
        _on_the_fly_backend()._validate_generated_artifact(
            tmp_path,
            ("process",),
            validation_points={"process": (_validation_point(),)},
            expected_api_bundle_path=None,
            compact_validation_selectors=_COMPACT_VALIDATION_SELECTORS,
        )


def test_on_the_fly_post_build_validation_disabled_skips_numerical_sample(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    runtime = _CompactRuntime()
    _install_compact_validation_spies(monkeypatch, runtime)

    _on_the_fly_backend(validation_enabled=False)._validate_generated_artifact(
        tmp_path,
        ("process",),
        validation_points={"process": (_validation_point(),)},
        expected_api_bundle_path=None,
    )

    assert runtime.calls == []


def test_on_the_fly_validation_selector_uses_first_projected_source_states() -> None:
    projection = SimpleNamespace(
        external_sources=(
            SimpleNamespace(
                source_states=(
                    SimpleNamespace(public_helicity=-1),
                    SimpleNamespace(public_helicity=1),
                )
            ),
            SimpleNamespace(
                source_states=(
                    SimpleNamespace(public_helicity=1),
                    SimpleNamespace(public_helicity=-1),
                )
            ),
            SimpleNamespace(source_states=(SimpleNamespace(public_helicity=0),)),
        )
    )

    selectors = generation_service._on_the_fly_validation_selectors_v1(
        projection  # type: ignore[arg-type]
    )

    assert dict(selectors) == {
        "helicities": ("h:-1,+1,+0",),
        "color_flows": ("1",),
    }


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    (
        ("artifact_id", "b" * 64, "wrong artifact"),
        ("execution_mode", "recurrence", "wrong execution mode"),
        ("representative_process_key", "other", "wrong process"),
    ),
)
def test_on_the_fly_post_build_validation_rejects_compact_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    attribute: str,
    value: str,
    message: str,
) -> None:
    runtime = _CompactRuntime()
    setattr(runtime, attribute, value)
    _install_compact_validation_spies(monkeypatch, runtime)

    with pytest.raises(GenerationError, match=message):
        _on_the_fly_backend()._validate_generated_artifact(
            tmp_path,
            ("process",),
            validation_points={},
            expected_api_bundle_path=None,
        )


def test_on_the_fly_post_build_validation_propagates_compact_selector_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def reject_selector(_cls, _output, *, process: str) -> object:
        assert process == "process"
        raise ArtifactError("compact selector contract is inconsistent")

    monkeypatch.setattr(Runtime, "load", classmethod(reject_selector))
    monkeypatch.setattr(
        "pyamplicol.artifacts.load_manifest",
        lambda _output: _compact_manifest(),
    )

    with pytest.raises(ArtifactError, match="compact selector contract"):
        _on_the_fly_backend()._validate_generated_artifact(
            tmp_path,
            ("process",),
            validation_points={},
            expected_api_bundle_path=None,
        )


def test_on_the_fly_post_build_validation_requires_canonical_seed_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    manifest = _compact_manifest()
    manifest.payloads = manifest.payloads[:-1]
    _install_compact_validation_spies(
        monkeypatch,
        _CompactRuntime(),
        manifest=manifest,
    )

    with pytest.raises(GenerationError, match="payload set is incomplete"):
        _on_the_fly_backend()._validate_generated_artifact(
            tmp_path,
            ("process",),
            validation_points={},
            expected_api_bundle_path=None,
        )
