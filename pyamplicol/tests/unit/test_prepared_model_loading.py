# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

import pytest

import pyamplicol.licensing as licensing
from pyamplicol import Generator, ModelSource
from pyamplicol.api.errors import GenerationError, ModelError
from pyamplicol.api.models import _compiled_model_payload
from pyamplicol.config import Action, EvaluatorConfig, JITConfig, ModelConfig, RunConfig
from pyamplicol.licensing import SymbolicaLicenseState
from pyamplicol.models.loading import compile_model_source, load_compiled_model
from pyamplicol.models.prepared import (
    PreparedKernelPack,
    PreparedKernelRecord,
    write_prepared_model_bundle,
)
from pyamplicol.models.prepared_target import (
    SYMJIT_STORAGE_V3_ABI,
    canonical_architecture,
    symjit_storage_v3_target,
)


def _prepared_builtin_sm(
    tmp_path: Path,
    *,
    machine: str | None = None,
    jit_compress: bool | None = True,
) -> Path:
    compiled = compile_model_source("built-in-sm", use_cache=False)
    kernel = PreparedKernelRecord(
        kernel_id=0,
        contract_kind="vertex",
        canonical_signature="test:prepared-loading",
        input_arity=1,
        output_arity=1,
        input_layout=("input",),
        input_contracts=(
            {
                "role": "current",
                "component": 0,
                "symbol": "pyamplicol::input",
                "model_parameter_name": None,
                "model_parameter_index": None,
            },
        ),
        output_layout=("output",),
        exact_expressions=("pyamplicol::input",),
        exact_evaluator_state_path="kernels/0/exact.evaluator.bin",
        f64_evaluator_manifest={
            "kind": "symjit-application-evaluator",
            "optimization_level": 2,
            "settings": {
                "iterations": 10,
                "cpe_iterations": None,
                "jit_optimization_level": 2,
                "max_horner_scheme_variables": 1000,
                "max_common_pair_cache_entries": 5_000_000,
                "max_common_pair_distance": 1000,
                "collect_factors": False,
            },
            "input_len": 1,
            "output_len": 1,
            "application_path": "kernels/0/application.symjit",
            "evaluator_state_path": "kernels/0/exact.evaluator.bin",
        },
    )
    optimization_settings: dict[str, object] = {
        "iterations": 10,
        "cpe_iterations": None,
        "jit_optimization_level": 2,
        "max_horner_scheme_variables": 1000,
        "max_common_pair_cache_entries": 5_000_000,
        "max_common_pair_distance": 1000,
        "collect_factors": False,
    }
    if jit_compress is not None:
        optimization_settings["jit_compress"] = jit_compress
    pack = PreparedKernelPack(
        backend="jit",
        optimization_settings=optimization_settings,
        producer={"distribution": "pyamplicol", "version": "test"},
        dependency_abis={
            "symjit_application": SYMJIT_STORAGE_V3_ABI,
            "symjit_plane_application": "pyamplicol-symjit-plane-application-v2",
        },
        provenance={"compiled_model": "test"},
        target=symjit_storage_v3_target(machine=machine),
        resolver_manifest={
            "abi": "pyamplicol-prepared-kernel-catalog-v1",
            "model_name": "built-in-sm",
        },
        kernels=(kernel,),
    )
    return write_prepared_model_bundle(
        tmp_path / "builtin",
        compiled_model=compiled.to_dict(),
        kernel_pack=pack,
        payloads={
            "kernels/0/exact.evaluator.bin": b"exact",
            "kernels/0/application.symjit": b"jit",
        },
    )


def test_model_source_loads_prepared_bundle_without_changing_provenance(
    tmp_path: Path,
) -> None:
    bundle_path = _prepared_builtin_sm(tmp_path)

    source = ModelSource.from_path(bundle_path)
    compiled = source.compile(use_cache=False)
    payload = _compiled_model_payload(compiled)

    assert source.kind == "prepared"
    assert compiled.name == "built-in-sm"
    assert compiled.source.kind == "built-in-sm"
    assert compiled.is_prepared
    assert compiled.prepared_backend == "jit"
    assert payload.prepared_bundle is not None
    assert payload.prepared_bundle.path == bundle_path


def test_internal_compiled_model_loader_accepts_prepared_bundle(tmp_path: Path) -> None:
    bundle_path = _prepared_builtin_sm(tmp_path)

    compiled = load_compiled_model(bundle_path)

    assert compiled.name == "built-in-sm"
    assert compiled.prepared_backend == "jit"
    assert compiled.prepared_bundle is not None


def test_prepared_model_rejects_compilation_options(tmp_path: Path) -> None:
    bundle_path = _prepared_builtin_sm(tmp_path)

    with pytest.raises(ModelError, match="already compiled"):
        ModelSource.from_path(bundle_path, simplify=False).compile(use_cache=False)


def _eager_config() -> RunConfig:
    return RunConfig(
        action=Action.GENERATE,
        evaluator=EvaluatorConfig(execution_mode="eager"),
    )


def _restricted_license(**_kwargs: object) -> SymbolicaLicenseState:
    return SymbolicaLicenseState(licensed=False, restricted=True)


def test_eager_plan_uses_packaged_builtin_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)
    bundle_path = _prepared_builtin_sm(tmp_path)
    monkeypatch.setattr(
        "pyamplicol.assets.prepared_models.materialize_packaged_prepared_model",
        lambda _identifier: bundle_path,
    )
    plan = Generator(_eager_config()).plan("d d~ > z")

    assert len(plan.concrete_processes) == 1
    assert plan.estimated_coverage["model_kind"] == "prepared"


def test_default_recurrence_plan_uses_packaged_builtin_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)
    bundle_path = _prepared_builtin_sm(tmp_path)
    monkeypatch.setattr(
        "pyamplicol.assets.prepared_models.materialize_packaged_prepared_model",
        lambda _identifier: bundle_path,
    )
    # The small loading fixture predates the recurrence catalog; this test
    # isolates default source selection, while recurrence-pack validation has
    # dedicated bundle/preflight coverage.
    monkeypatch.setattr(
        "pyamplicol.generation.service.GenerationBackend._require_eager_kernel_pack",
        lambda _self, _resolved: None,
    )

    plan = Generator(RunConfig(action=Action.GENERATE)).plan("d d~ > z")

    assert plan.effective_settings.evaluator.execution_mode == "recurrence"
    assert plan.estimated_coverage["model_kind"] == "prepared"


def test_default_heft_recurrence_selects_its_own_packaged_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyamplicol.assets.prepared_models import BUILTIN_SM_HEFT_JIT_O2

    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)
    bundle_path = _prepared_builtin_sm(tmp_path)
    selected: list[str] = []

    def materialize(identifier: str) -> Path:
        selected.append(identifier)
        return bundle_path

    monkeypatch.setattr(
        "pyamplicol.assets.prepared_models.materialize_packaged_prepared_model",
        materialize,
    )
    monkeypatch.setattr(
        "pyamplicol.generation.service.GenerationBackend._require_eager_kernel_pack",
        lambda _self, _resolved: None,
    )
    config = RunConfig(
        action=Action.GENERATE,
        model=ModelConfig(source="built-in-sm-heft"),
    )

    plan = Generator(config).plan("d d~ > z")

    assert selected == [BUILTIN_SM_HEFT_JIT_O2]
    assert plan.estimated_coverage["model_kind"] == "prepared"


def test_eager_compiled_handle_without_pack_fails_before_dag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)
    model = ModelSource.built_in_sm().compile(use_cache=False)

    with pytest.raises(
        GenerationError,
        match=r"pyamplicol model compile.*--backend jit",
    ):
        Generator(_eager_config()).plan("d d~ > z", model=model)


def test_eager_accepts_portable_prepared_model_from_other_architecture_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)
    other = "x86_64" if canonical_architecture() == "aarch64" else "aarch64"
    model = ModelSource.from_path(_prepared_builtin_sm(tmp_path, machine=other))

    plan = Generator(_eager_config()).plan("d d~ > z", model=model)

    assert len(plan.concrete_processes) == 1
    assert plan.effective_settings.evaluator.jit.optimization_level == 2


def test_compiled_plan_does_not_materialize_packaged_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)

    def fail(_identifier: str) -> Path:
        raise AssertionError("compiled mode touched eager packaged resources")

    monkeypatch.setattr(
        "pyamplicol.assets.prepared_models.materialize_packaged_prepared_model",
        fail,
    )

    plan = Generator(
        RunConfig(
            action=Action.GENERATE,
            evaluator=EvaluatorConfig(execution_mode="compiled"),
        )
    ).plan("d d~ > z")

    assert plan.estimated_coverage["model_kind"] == "built-in-sm"


def test_eager_plan_accepts_prepared_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)
    model = ModelSource.from_path(_prepared_builtin_sm(tmp_path))

    plan = Generator(_eager_config()).plan("d d~ > z", model=model)

    assert len(plan.concrete_processes) == 1
    assert plan.estimated_coverage["model_kind"] == "prepared"
    assert plan.effective_settings.evaluator.backend == "jit"
    assert plan.effective_settings.evaluator.jit.optimization_level == 2
    assert plan.effective_settings.evaluator.jit.compress is True
    assert {adjustment.path for adjustment in plan.adjustments} >= {
        "evaluator.optimization.collect_factors"
    }


def test_eager_prepared_pack_settings_are_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)
    model = ModelSource.from_path(_prepared_builtin_sm(tmp_path, jit_compress=False))
    config = RunConfig(
        action=Action.GENERATE,
        evaluator=EvaluatorConfig(
            backend="cpp",
            execution_mode="eager",
            jit=JITConfig(optimization_level=1, compress=True),
        ),
    )

    plan = Generator(config).plan("d d~ > z", model=model)

    assert plan.requested_settings.evaluator.backend == "cpp"
    assert plan.requested_settings.evaluator.jit.optimization_level == 1
    assert plan.requested_settings.evaluator.jit.compress is True
    assert plan.effective_settings.evaluator.backend == "jit"
    assert plan.effective_settings.evaluator.jit.optimization_level == 2
    assert plan.effective_settings.evaluator.jit.compress is False
    by_path = {adjustment.path: adjustment for adjustment in plan.adjustments}
    assert by_path["evaluator.backend"].requested == "cpp"
    assert by_path["evaluator.backend"].effective == "jit"
    assert by_path["evaluator.jit.optimization_level"].requested == 1
    assert by_path["evaluator.jit.optimization_level"].effective == 2
    assert by_path["evaluator.jit.compress"].requested is True
    assert by_path["evaluator.jit.compress"].effective is False
    assert "prepared model settings" in caplog.text


def test_prepared_jit_pack_requires_explicit_compression_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)
    model = ModelSource.from_path(_prepared_builtin_sm(tmp_path, jit_compress=None))

    with pytest.raises(
        GenerationError,
        match="omits optimization setting 'jit_compress'",
    ):
        Generator(_eager_config()).plan("d d~ > z", model=model)
