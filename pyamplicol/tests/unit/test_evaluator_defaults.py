# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import subprocess
import sys

import pytest

from pyamplicol.evaluators import SymbolicaEvaluatorSettings
from pyamplicol.evaluators.symbolica_compile import _symbolica_evaluator_kwargs


def test_evaluator_defaults_preserve_production_optimization_policy() -> None:
    settings = SymbolicaEvaluatorSettings()
    assert settings.backend == "jit"
    assert settings.iterations == 10
    assert settings.cpe_iterations is None
    assert settings.jit_optimization_level == 2
    assert settings.jit_compress is False
    assert settings.jit_direct_translation is False
    assert settings.max_horner_scheme_variables == 1000
    assert settings.max_common_pair_cache_entries == 5_000_000
    assert settings.max_common_pair_distance == 1000


@pytest.mark.parametrize(
    ("compress", "jit_options"),
    ((True, {"compress": "true"}), (False, {"compress": "false"})),
)
def test_jit_compression_is_forwarded_to_symbolica(
    compress: bool,
    jit_options: dict[str, str],
) -> None:
    settings = SymbolicaEvaluatorSettings(jit_compress=compress)

    assert settings.to_json_dict()["jit_compress"] is compress
    assert (
        _symbolica_evaluator_kwargs(
            settings,
            verbose=False,
        )["jit_options"]
        == jit_options
    )


def test_jit_compression_setting_requires_a_boolean() -> None:
    with pytest.raises(ValueError, match="jit_compress"):
        SymbolicaEvaluatorSettings(jit_compress=1)  # type: ignore[arg-type]


def test_evaluator_module_import_registers_symbolica_without_model_tooling() -> None:
    code = """
import sys
assert 'symbolica' not in sys.modules
from pyamplicol.evaluators import SymbolicaEvaluatorSettings
SymbolicaEvaluatorSettings()
assert 'symbolica' in sys.modules
assert 'pyamplicol.models.compiler_symbolica' not in sys.modules
    """
    subprocess.run([sys.executable, "-c", code], check=True)


def test_removed_simd_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="compiled-complex"):
        SymbolicaEvaluatorSettings(backend="compiled-complex-4x")


def test_compiled_flags_do_not_probe_checkout_dependencies() -> None:
    settings = SymbolicaEvaluatorSettings(
        backend="compiled-complex",
        compiler_flags=("-DPYAMPLICOL_TEST_FLAG=1",),
    )

    payload = settings.to_json_dict()
    assert payload["compiler_flags"] == ["-DPYAMPLICOL_TEST_FLAG=1"]
    assert payload["effective_compiler_flags"] == [
        "-std=c++17",
        "-DPYAMPLICOL_TEST_FLAG=1",
    ]

    jit_payload = SymbolicaEvaluatorSettings(
        backend="jit",
        compiler_flags=("-DPYAMPLICOL_TEST_FLAG=1",),
    ).to_json_dict()
    assert jit_payload["effective_compiler_flags"] == ["-DPYAMPLICOL_TEST_FLAG=1"]
