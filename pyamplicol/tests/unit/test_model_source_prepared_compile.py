# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol import ModelSource
from pyamplicol.api import ModelError
from pyamplicol.config import EvaluatorBackend, EvaluatorConfig


class _CompiledPayload:
    name = "toy"
    schema_version = 9
    model_compiler_version = 13
    conversion_seconds = 0.1

    def __init__(self, prepared_backend: str | None = None) -> None:
        self.source = {"kind": "built-in-sm"}
        self.capabilities = {"particle_count": 1}
        self.parameter_defaults: dict[str, tuple[float, float]] = {}
        self.issues: tuple[object, ...] = ()
        self.phase_timings = {"total": 0.1}
        self.prepared_backend = prepared_backend


def test_model_source_compile_can_prepare_and_return_the_bundle_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyamplicol.models import loading, prepared_compile

    base_payload = _CompiledPayload()
    prepared_payload = _CompiledPayload(prepared_backend="asm")
    compilation_calls: list[tuple[object, dict[str, object]]] = []
    preparation_calls: list[tuple[object, Path, EvaluatorConfig]] = []
    load_calls: list[Path] = []

    def compile_model_source(source: object, **options: object) -> object:
        compilation_calls.append((source, options))
        return base_payload

    def prepare_model_bundle(
        payload: object,
        output: Path,
        *,
        evaluator: EvaluatorConfig,
    ) -> object:
        preparation_calls.append((payload, output, evaluator))
        return SimpleNamespace(output=output)

    def load_compiled_model(path: Path) -> object:
        load_calls.append(path)
        return prepared_payload

    monkeypatch.setattr(loading, "compile_model_source", compile_model_source)
    monkeypatch.setattr(loading, "load_compiled_model", load_compiled_model)
    monkeypatch.setattr(
        prepared_compile,
        "prepare_model_bundle",
        prepare_model_bundle,
    )
    evaluator = EvaluatorConfig(backend=EvaluatorBackend.ASM)
    output = tmp_path / "toy.pyamplicol-model"

    compiled = ModelSource.built_in_sm().compile(
        cache_dir=tmp_path / "cache",
        use_cache=False,
        prepared_output=output,
        evaluator=evaluator,
    )

    assert compilation_calls == [
        (
            "built-in-sm",
            {
                "restriction": "default",
                "simplify": True,
                "cache_dir": (tmp_path / "cache").resolve(),
                "use_cache": False,
                "require_supported": True,
            },
        )
    ]
    assert preparation_calls == [(base_payload, output.resolve(), evaluator)]
    assert load_calls == [output.resolve()]
    assert compiled.is_prepared
    assert compiled.prepared_backend == "asm"


def test_model_source_compile_default_does_not_prepare_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyamplicol.models import loading, prepared_compile

    monkeypatch.setattr(
        loading,
        "compile_model_source",
        lambda *args, **kwargs: _CompiledPayload(),
    )
    monkeypatch.setattr(
        prepared_compile,
        "prepare_model_bundle",
        lambda *_args, **_kwargs: pytest.fail(
            "default ModelSource.compile must remain IR-only"
        ),
    )

    compiled = ModelSource.built_in_sm().compile()

    assert not compiled.is_prepared


@pytest.mark.parametrize(
    ("prepared_output", "evaluator", "message"),
    (
        (
            Path("toy.pyamplicol-model"),
            None,
            "requires both prepared_output",
        ),
        (
            None,
            EvaluatorConfig(),
            "requires both prepared_output",
        ),
        (
            Path("toy.json"),
            EvaluatorConfig(),
            "must end with '.pyamplicol-model'",
        ),
    ),
)
def test_model_source_compile_rejects_incomplete_preparation_requests(
    prepared_output: Path | None,
    evaluator: EvaluatorConfig | None,
    message: str,
) -> None:
    with pytest.raises(ModelError, match=message):
        ModelSource.built_in_sm().compile(
            prepared_output=prepared_output,
            evaluator=evaluator,
        )
