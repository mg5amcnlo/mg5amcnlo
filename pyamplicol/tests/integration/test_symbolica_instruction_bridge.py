# SPDX-License-Identifier: 0BSD
"""Numerically exercise released Symbolica instructions in native plane kernels."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyamplicol import Generator, ModelSource, Runtime
from pyamplicol.config import (
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)


def test_unit_imaginary_instructions_survive_native_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        if os.environ.get("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == "1":
            pytest.fail("the Rusticol extension has not been built")
        pytest.skip("the Rusticol extension has not been built")

    from symbolica import E

    import pyamplicol.evaluators.symbolica as evaluator_module
    from pyamplicol.evaluators.symbolica_adapters import _JITSymbolicaEvaluatorAdapter

    coefficients = (E("1i"), E("-1i"), E("1/3-1i/2"))
    original_compile = evaluator_module._compile_symbolica_outputs
    exported_constants: set[Any] = set()
    original_export = _JITSymbolicaEvaluatorAdapter._export_symjit_plane_application

    def export_plane(adapter: Any, **kwargs: Any) -> Any:
        exported_constants.update(
            adapter._source_evaluator.get_instructions().constants
        )
        return original_export(adapter, **kwargs)

    def compile_amplitudes(outputs: tuple[Any, ...], *args: Any, **kwargs: Any) -> Any:
        if "amplitude" in kwargs.get("label", ""):
            # The surrounding process supplies a valid native runtime schema;
            # only its final amplitudes are synthetic. In particular, -i must
            # not be silently parsed as zero by the SymJIT instruction bridge.
            outputs = tuple(
                coefficients[i % len(coefficients)] * (output + output**2)
                for i, output in enumerate(outputs)
            )
        return original_compile(outputs, *args, **kwargs)

    monkeypatch.setattr(
        evaluator_module, "_compile_symbolica_outputs", compile_amplitudes
    )
    monkeypatch.setattr(
        _JITSymbolicaEvaluatorAdapter, "_export_symjit_plane_application", export_plane
    )
    artifact = tmp_path / "imaginary-constants"
    Generator(
        RunConfig(
            action="generate",
            generation=GenerationConfig(
                workers=1,
                emit_api_bundle=False,
                relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
                validation=GenerationValidationConfig(
                    enabled=False, post_build_validation=False
                ),
            ),
            evaluator=EvaluatorConfig(
                execution_mode="compiled",
                optimization=EvaluatorOptimizationConfig(cores=1),
                jit=JITConfig(optimization_level=2),
            ),
        )
    ).generate("d d~ > z", artifact, model=ModelSource.built_in_sm())
    assert set(coefficients) <= exported_constants

    momenta = [
        [
            [45.594, 0.0, 0.0, 45.594],
            [45.594, 0.0, 0.0, -45.594],
            [91.188, 0.0, 0.0, 0.0],
        ]
    ]
    runtime = Runtime.load(artifact)
    native = np.asarray(runtime.evaluate_resolved(momenta).values, dtype=complex)
    exact = np.asarray(
        runtime.evaluate_resolved(momenta, precision=40).values, dtype=complex
    )
    assert np.any(np.abs(exact) > 0)
    np.testing.assert_allclose(native, exact, rtol=1e-13, atol=1e-15)
