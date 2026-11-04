# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.cli import DefaultCliServices, parse_cli
from pyamplicol.config import ConfigurationError, EvaluatorBackend, EvaluatorConfig
from pyamplicol.models.contracts import CompiledModelIR, CompiledParticleRecord
from pyamplicol.reporting import NullProgressSink


class _CompiledModel:
    ir = CompiledModelIR(
        name="toy",
        orders=(),
        parameters=(),
        particles=(
            CompiledParticleRecord(
                "d",
                "d~",
                1,
                2,
                3,
                "ZERO",
                "ZERO",
                -1.0 / 3.0,
                (("electric_charge", "-1/3"),),
                0,
                True,
                False,
                None,
            ),
            CompiledParticleRecord(
                "d~",
                "d",
                -1,
                2,
                -3,
                "ZERO",
                "ZERO",
                1.0 / 3.0,
                (("electric_charge", "1/3"),),
                0,
                True,
                False,
                None,
            ),
            CompiledParticleRecord(
                "z",
                "z",
                23,
                3,
                1,
                "MZ",
                "WZ",
                0.0,
                (("electric_charge", "0"),),
                0,
                True,
                False,
                None,
            ),
        ),
        couplings=(),
        propagators=(),
        vertex_terms=(),
        oriented_kernels=(),
        direct_contractions=(),
        closure_contractions=(),
    )

    def __init__(self) -> None:
        self.name = "toy"
        self.supported = True
        self.source = {"kind": "built-in-sm"}
        self.producer = {"pyamplicol": "test"}
        self.capabilities = {"particle_count": 3}
        self.issues: tuple[object, ...] = ()
        self.conversion_seconds = 0.25
        self.phase_timings = {"total": 0.25}
        self.parameter_defaults = {"mass": (1.0, 0.0)}

    def write(self, path: Path) -> Path:
        return path.with_name(path.name + ".pyAmplicol-model.json")


def test_default_model_compile_calls_existing_compiler_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyamplicol.models import loading, prepared_compile

    calls: list[dict[str, object]] = []

    def compile_model_source(source: str, **options: object) -> _CompiledModel:
        calls.append({"source": source, **options})
        return _CompiledModel()

    monkeypatch.setattr(loading, "compile_model_source", compile_model_source)
    monkeypatch.setattr(
        prepared_compile,
        "prepare_model_bundle",
        lambda *_args, **_kwargs: pytest.fail(
            "IR-only model compilation must not prepare evaluator kernels"
        ),
    )
    output = tmp_path / "toy"
    invocation = parse_cli(("model", "compile", "built-in-sm", str(output)))
    result = DefaultCliServices().model_compile(
        invocation.resolve().effective,
        NullProgressSink(),
    )

    assert calls == [
        {
            "source": "built-in-sm",
            "restriction": "default",
            "simplify": True,
            "cache_dir": None,
            "use_cache": True,
            "require_supported": True,
        }
    ]
    assert result == {
        "model": "toy",
        "supported": True,
        "output": str(output) + ".pyAmplicol-model.json",
        "conversion_seconds": 0.25,
        "phase_timings": {"total": 0.25},
        "source": {"kind": "built-in-sm"},
        "capabilities": {"particle_count": 3},
    }


def test_model_inspect_routes_builtin_sm_heft_through_packaged_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyamplicol._internal import sm_heft

    calls: list[dict[str, object]] = []

    def compile_sm_heft_source(**options: object) -> _CompiledModel:
        calls.append(options)
        compiled = _CompiledModel()
        compiled.name = "built-in-sm-heft"
        compiled.source = {"kind": "built-in-sm-heft"}
        return compiled

    monkeypatch.setattr(sm_heft, "compile_sm_heft_source", compile_sm_heft_source)
    invocation = parse_cli(("model", "inspect", "built-in-sm-heft"))
    result = DefaultCliServices().model_inspect(
        invocation.resolve().effective,
        NullProgressSink(),
    )

    assert calls == [
        {
            "cache_dir": None,
            "use_cache": True,
            "require_supported": False,
        }
    ]
    assert result["name"] == "built-in-sm-heft"
    assert result["source"] == {"kind": "built-in-sm-heft"}


def test_default_model_compile_prepares_one_requested_backend_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyamplicol.models import loading, prepared_compile

    compiled = _CompiledModel()
    monkeypatch.setattr(
        loading,
        "compile_model_source",
        lambda *args, **kwargs: compiled,
    )
    captured: dict[str, object] = {}

    def prepare_model_bundle(
        model: object,
        output: Path,
        *,
        evaluator: object,
        progress: object,
    ) -> object:
        captured.update(
            model=model,
            output=output,
            evaluator=evaluator,
            progress=progress,
        )
        assert callable(progress)
        progress("prepare kernel 0", 0, 1)
        progress("prepared model complete", 1, 1)
        return SimpleNamespace(
            output=output,
            bundle=SimpleNamespace(backend="jit"),
            kernel_count=1,
            phase_timings_seconds={"catalog": 0.1, "total": 0.3},
        )

    monkeypatch.setattr(
        prepared_compile,
        "prepare_model_bundle",
        prepare_model_bundle,
    )
    output = tmp_path / "toy.pyamplicol-model"
    invocation = parse_cli(
        (
            "model",
            "compile",
            "built-in-sm",
            str(output),
            "--backend",
            "jit",
            "--horner-iterations",
            "7",
            "--cpe-iterations",
            "3",
            "--cores",
            "2",
            "--max-horner-variables",
            "321",
            "--max-common-pair-cache-entries",
            "456",
            "--max-common-pair-distance",
            "78",
            "--collect-factors",
            "--no-jit-compress",
        )
    )
    config = invocation.resolve().effective

    assert config.evaluator.backend is EvaluatorBackend.JIT
    assert config.evaluator.jit.optimization_level == 2
    assert config.evaluator.jit.compress is False
    assert config.evaluator.optimization.horner_iterations == 7
    assert config.evaluator.optimization.cpe_iterations == 3
    assert config.evaluator.optimization.cores == 2
    assert config.evaluator.optimization.max_horner_variables == 321
    assert config.evaluator.optimization.max_common_pair_cache_entries == 456
    assert config.evaluator.optimization.max_common_pair_distance == 78
    assert config.evaluator.optimization.collect_factors is True

    result = DefaultCliServices().model_compile(config, NullProgressSink())

    assert captured["model"] is compiled
    assert captured["output"] == output.resolve()
    assert captured["evaluator"] is config.evaluator
    assert result["output"] == str(output.resolve())
    assert result["prepared_backend"] == "jit"
    assert result["kernel_count"] == 1
    assert result["preparation_phase_timings"] == {
        "catalog": 0.1,
        "total": 0.3,
    }


@pytest.mark.parametrize(
    ("flags", "expected"),
    (((), False), (("--jit-compress",), True), (("--no-jit-compress",), False)),
)
def test_model_compile_jit_compression_selection(
    tmp_path: Path, flags: tuple[str, ...], expected: bool
) -> None:
    config = (
        parse_cli(
            (
                "model",
                "compile",
                "built-in-sm",
                str(tmp_path / "model.pyamplicol-model"),
                *flags,
            )
        )
        .resolve()
        .effective
    )
    assert config.evaluator.jit.compress is expected


def test_model_compile_rejects_evaluator_options_for_ir_output(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ConfigurationError,
        match=r"evaluator options require.*\.pyamplicol-model",
    ):
        parse_cli(
            (
                "model",
                "compile",
                "built-in-sm",
                str(tmp_path / "toy.json"),
                "--backend",
                "jit",
            )
        )


def test_model_compile_service_rejects_nondefault_card_evaluator_for_ir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyamplicol.models import loading

    monkeypatch.setattr(
        loading,
        "compile_model_source",
        lambda *args, **kwargs: _CompiledModel(),
    )
    invocation = parse_cli(
        ("model", "compile", "built-in-sm", str(tmp_path / "toy.json"))
    )
    config = replace(
        invocation.resolve().effective,
        evaluator=EvaluatorConfig(backend=EvaluatorBackend.ASM),
    )

    with pytest.raises(
        ConfigurationError,
        match=r"evaluator settings require.*\.pyamplicol-model",
    ):
        DefaultCliServices().model_compile(config, NullProgressSink())


def test_default_model_processes_uses_compiled_model_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyamplicol.models import loading

    monkeypatch.setattr(
        loading,
        "compile_model_source",
        lambda *args, **kwargs: _CompiledModel(),
    )
    invocation = parse_cli(("model", "processes", "d d~ > z"))
    result = DefaultCliServices().model_processes(
        invocation.resolve().effective,
        NullProgressSink(),
    )

    assert isinstance(result, dict)
    assert result["model"] == "toy"
    assert result["n_entries"] == 1
    entry = result["entries"][0]
    assert entry["process"] == "d d~ > z"
    assert entry["ir"]["outgoing_pdgs"] == [-1, 1, 23]
