# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import pyamplicol.evaluators.symbolica_adapters as symbolica_adapters
from pyamplicol._internal.physics.types import NativeEvaluationError
from pyamplicol._internal.versions import (
    SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    SYMBOLICA_CPP_RUNTIME_CAPABILITY,
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    SYMJIT_APPLICATION_ABI,
    SYMJIT_F64_RUNTIME_CAPABILITY,
    SYMJIT_PLANE_APPLICATION_ABI,
)
from pyamplicol.evaluators.execution_schema import evaluator_runtime_capabilities
from pyamplicol.evaluators.symbolica_adapters import (
    _ChunkedSymbolicaEvaluator,
    _compiled_complex_custom_header,
    _compiled_runtime_capability,
    _CompiledComplexEvaluatorAdapter,
    _JITSymbolicaEvaluatorAdapter,
)
from pyamplicol.evaluators.symbolica_compile import _chunk_parameter_indices
from pyamplicol.evaluators.symbolica_helpers import (
    _symbolica_instruction_program,
    _symbolica_instruction_program_repr,
)
from pyamplicol.evaluators.symbolica_settings import SymbolicaEvaluatorSettings
from pyamplicol.generation.artifact_writer import _evaluator, _stage_evaluator_set

ROOT = Path(__file__).resolve().parents[2]


class _FakeJITEvaluator:
    def __init__(self, *, export_error: Exception | None = None) -> None:
        self.export_error = export_error
        self.evaluation_count = 0

    def evaluate_complex(self, rows: Any) -> np.ndarray:
        self.evaluation_count += 1
        return np.zeros((len(rows), 2), dtype=np.complex128)

    def export_symjit(self, *, complex: bool = False) -> bytes:
        assert complex is True
        if self.export_error is not None:
            raise self.export_error
        return b"symjit-application-v3"

    def save(self) -> bytes:
        return b"symbolica-evaluator-state"

    def get_instructions(self) -> tuple[list[object], int, list[object]]:
        return ([("assign", ("out", 0), ("param", 0))], 1, [])


class _FakeRusticol:
    @staticmethod
    def _compile_symjit_plane_application_v2(
        program: str,
        input_len: int,
        output_len: int,
        optimization_level: int,
        compress: bool,
    ) -> bytes:
        assert program
        assert input_len >= 1
        assert output_len >= 1
        assert optimization_level in {0, 1, 2, 3}
        assert isinstance(compress, bool)
        return b"symjit-plane-application-v2"


@pytest.fixture(autouse=True)
def _fake_plane_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRusticol()
    monkeypatch.setitem(sys.modules, "pyamplicol._rusticol", fake)
    monkeypatch.setattr(
        symbolica_adapters,
        "verify_native_module",
        lambda _module: None,
    )


class _MappedEvaluator:
    def __init__(self, input_len: int) -> None:
        self.input_len = input_len

    def evaluate_complex(self, rows: Any) -> np.ndarray:
        values = np.asarray(rows, dtype=np.complex128)
        return values.sum(axis=1, keepdims=True)


class _FakeCompiledEvaluator:
    def __init__(self) -> None:
        self.compile_kwargs: dict[str, object] | None = None

    def save(self) -> bytes:
        return b"symbolica-evaluator-state"

    def compile(self, *_args: object, **kwargs: object) -> _MappedEvaluator:
        self.compile_kwargs = kwargs
        return _MappedEvaluator(1)


def _jit_adapter(
    evaluator: object | None = None,
    *,
    direct_translation: bool = False,
) -> _JITSymbolicaEvaluatorAdapter:
    return _JITSymbolicaEvaluatorAdapter(
        _FakeJITEvaluator() if evaluator is None else evaluator,
        SymbolicaEvaluatorSettings(
            backend="jit",
            jit_direct_translation=direct_translation,
            jit_optimization_level=3,
            n_cores=1,
        ),
        "test stage",
        input_len=3,
        output_len=2,
    )


@pytest.mark.parametrize("structured", [False, True])
def test_symjit_plane_export_preserves_shared_outputs_and_matching_digest(
    monkeypatch: pytest.MonkeyPatch,
    structured: bool,
) -> None:
    program = (
        [
            ("assign", ("out", 0), ("param", 0)),
            ("assign", ("out", 1), ("out", 0)),
        ],
        0,
        [],
    )
    expected = repr(program)
    exported = (
        SimpleNamespace(
            instructions=program[0],
            temporary_count=program[1],
            constants=program[2],
            sub_evaluators=[],
        )
        if structured
        else program
    )
    source = _FakeJITEvaluator()
    monkeypatch.setattr(source, "get_instructions", lambda: exported)
    received: list[tuple[object, ...]] = []

    def compile_plane(*args: object) -> bytes:
        received.append(args)
        return b"unmodified-plane"

    monkeypatch.setattr(
        _FakeRusticol,
        "_compile_symjit_plane_application_v2",
        staticmethod(compile_plane),
    )

    application, digest, _target = _jit_adapter(
        source
    )._export_symjit_plane_application(optimization_level=2)

    assert received == [(expected, 3, 2, 2, False)]  # JIT compression is opt-in.
    assert application == b"unmodified-plane"
    assert digest == hashlib.sha256(expected.encode()).hexdigest()
    assert source.get_instructions() is exported


def test_symbolica_instruction_export_rejects_uninlined_functions() -> None:
    exported = SimpleNamespace(
        instructions=[], temporary_count=0, constants=[], sub_evaluators=[object()]
    )
    with pytest.raises(
        NativeEvaluationError, match="inlined Symbolica function bodies"
    ):
        _symbolica_instruction_program(exported)


@pytest.mark.parametrize(
    ("constant", "serialized"),
    [
        ("1i", "1\U0001d456"),
        ("-1i", "-1\U0001d456"),
        ("1+1i", "1+1\U0001d456"),
        ("1/3-1i/2", "1/3-1\U0001d456/2"),
        ("2i/3", "2\U0001d456/3"),
    ],
)
def test_symbolica_instruction_constants_have_explicit_imaginary_coefficients(
    constant: str, serialized: str
) -> None:
    from symbolica import E, Expression, S

    x = S("instruction_constant_x")
    evaluator = Expression.evaluator_multiple(
        [E(constant) * x], [x], jit_compile=False, n_cores=1
    )
    exported = evaluator.get_instructions()
    original = repr(exported.constants)

    program = _symbolica_instruction_program_repr(exported)

    assert program == (
        f"({exported.instructions!r}, {exported.temporary_count}, [{serialized}])"
    )
    assert repr(exported.constants) == original


def test_jit_artifact_persists_direct_application_and_precision_fallback(
    tmp_path: Path,
) -> None:
    source = _FakeJITEvaluator()
    manifest = _jit_adapter(source).artifact_manifest(tmp_path)

    assert source.evaluation_count == 1
    assert manifest["kind"] == "symjit-application-evaluator"
    assert manifest["runtime_capability"] == SYMJIT_F64_RUNTIME_CAPABILITY
    assert manifest["application_abi"] == SYMJIT_APPLICATION_ABI
    assert manifest["batch_layout"] == "row-major"
    assert manifest["compiler_type"] == "native"
    assert manifest["translation_mode"] == "indirect"
    assert manifest["optimization_level"] == 3
    assert manifest["word_bits"] == 64
    assert manifest["endianness"] == "little"
    assert manifest["required_defuns"] == []
    assert manifest["evaluator_state_runtime_capability"] == (
        SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY
    )
    assert (tmp_path / str(manifest["application_path"])).read_bytes() == (
        b"symjit-application-v3"
    )
    plane = manifest["plane_application"]
    assert isinstance(plane, dict)
    assert plane["application_abi"] == SYMJIT_PLANE_APPLICATION_ABI
    assert (
        tmp_path / str(plane["application_path"])
    ).read_bytes() == b"symjit-plane-application-v2"
    assert (tmp_path / str(manifest["evaluator_state_path"])).read_bytes() == (
        b"symbolica-evaluator-state"
    )
    assert evaluator_runtime_capabilities(manifest) == (SYMJIT_F64_RUNTIME_CAPABILITY,)


@pytest.mark.parametrize(
    "compress, expected", (("auto", True), (False, False), (True, True))
)
def test_compiled_o2_compression_matches_symbolica_native_lowering_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    compress: bool | str,
    expected: bool,
) -> None:
    from pyamplicol.config import resolve_config
    from pyamplicol.evaluators.symbolica_compile import _symbolica_evaluator_kwargs
    from pyamplicol.generation.service import GenerationBackend

    resolution = resolve_config(
        {
            "action": "generate",
            "evaluator": {"execution_mode": "compiled", "jit": {"compress": compress}},
        }
    )
    settings = GenerationBackend(resolution, None)._symbolica_settings()
    calls: list[tuple[object, ...]] = []

    def compile_plane(*args: object) -> bytes:
        calls.append(args)
        return b"compressed-plane"

    monkeypatch.setattr(
        _FakeRusticol,
        "_compile_symjit_plane_application_v2",
        staticmethod(compile_plane),
    )
    manifest = _JITSymbolicaEvaluatorAdapter(
        _FakeJITEvaluator(), settings, "compiled-o2", input_len=3, output_len=2
    ).artifact_manifest(tmp_path)

    assert resolution.effective.evaluator.jit.compress is expected
    assert settings.jit_compress is expected
    assert _symbolica_evaluator_kwargs(settings, verbose=False)["jit_options"] == {
        "compress": str(expected).lower()
    }
    assert calls[0][3:] == (2, expected)
    assert manifest["settings"]["jit_compress"] is expected
    assert manifest["plane_application"]["compression"] is expected


def test_jit_artifact_rejects_non_self_contained_export(tmp_path: Path) -> None:
    adapter = _jit_adapter(
        _FakeJITEvaluator(export_error=ValueError("external defuns present"))
    )
    with pytest.raises(NativeEvaluationError, match="external evaluator functions"):
        adapter.artifact_manifest(tmp_path)


def test_jit_artifact_rejects_direct_translation(tmp_path: Path) -> None:
    adapter = _jit_adapter(direct_translation=True)
    with pytest.raises(NativeEvaluationError, match="cannot be persisted"):
        adapter.artifact_manifest(tmp_path)


def test_chunked_evaluator_aggregates_primary_capabilities(tmp_path: Path) -> None:
    manifest = _ChunkedSymbolicaEvaluator(
        (_jit_adapter(), _jit_adapter())
    ).artifact_manifest(tmp_path)

    assert manifest["required_runtime_capabilities"] == [SYMJIT_F64_RUNTIME_CAPABILITY]
    assert manifest["input_len"] == 3
    assert manifest["chunk_input_indices"] == [[0, 1, 2], [0, 1, 2]]
    assert evaluator_runtime_capabilities(manifest) == (SYMJIT_F64_RUNTIME_CAPABILITY,)
    serialized = _evaluator(manifest)
    assert serialized["input_len"] == 3
    assert serialized["chunk_input_indices"] == [[0, 1, 2], [0, 1, 2]]


def test_chunked_evaluator_selects_parent_inputs_per_chunk() -> None:
    evaluator = _ChunkedSymbolicaEvaluator(
        (_MappedEvaluator(2), _MappedEvaluator(1)),
        input_len=3,
        chunk_input_indices=((0, 2), (1,)),
    )

    values = evaluator.evaluate_complex(
        np.asarray([[1.0, 10.0, 3.0], [2.0, 20.0, 5.0]], dtype=np.complex128)
    )

    assert values.tolist() == [[4.0 + 0.0j, 10.0 + 0.0j], [7.0 + 0.0j, 20.0 + 0.0j]]


def test_symbolica_chunk_dependencies_preserve_parent_parameter_order() -> None:
    from symbolica import S

    x, y, z, closed, argument = S("chunk_x", "chunk_y", "chunk_z", "closed", "arg")
    function = S("chunk_function")

    assert _chunk_parameter_indices(
        (z + function(x),),
        (x, y, z, closed),
        functions={(function, (argument,)): argument + closed},
    ) == (0, 2, 3)


def test_compiled_capability_distinguishes_cpp_and_asm() -> None:
    assert _compiled_runtime_capability({"compiled_inline_asm": "none"}) == (
        SYMBOLICA_CPP_RUNTIME_CAPABILITY
    )
    assert _compiled_runtime_capability({"compiled_inline_asm": "default"}) == (
        SYMBOLICA_ASM_RUNTIME_CAPABILITY
    )


def test_cpp_adapter_supplies_nested_complex_literal_compatibility_header(
    tmp_path: Path,
) -> None:
    evaluator = _FakeCompiledEvaluator()
    settings = SymbolicaEvaluatorSettings(
        backend="compiled-complex",
        compiled_inline_asm="none",
        compiled_output_dir=str(tmp_path),
    )

    _CompiledComplexEvaluatorAdapter(
        evaluator,
        settings,
        "cpp literal",
        input_len=1,
        output_len=1,
    )

    assert evaluator.compile_kwargs is not None
    assert evaluator.compile_kwargs["compiler_flags"] == ("-std=c++17",)
    assert evaluator.compile_kwargs["custom_header"] == (
        _compiled_complex_custom_header(settings)
    )
    assert "pyamplicol_complex_literal" in str(
        evaluator.compile_kwargs["custom_header"]
    )


def test_asm_adapter_does_not_use_cpp_literal_compatibility_header(
    tmp_path: Path,
) -> None:
    evaluator = _FakeCompiledEvaluator()
    settings = SymbolicaEvaluatorSettings(
        backend="compiled-complex",
        compiled_inline_asm="default",
        compiled_output_dir=str(tmp_path),
    )

    _CompiledComplexEvaluatorAdapter(
        evaluator,
        settings,
        "asm literal",
        input_len=1,
        output_len=1,
    )

    assert evaluator.compile_kwargs is not None
    assert evaluator.compile_kwargs["compiler_flags"] == ("-std=c++17",)
    assert "custom_header" not in evaluator.compile_kwargs


def test_cpp_literal_compatibility_header_executes_nested_constructors(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the compatibility smoke")
    settings = SymbolicaEvaluatorSettings(
        backend="compiled-complex",
        compiled_inline_asm="none",
    )
    header = _compiled_complex_custom_header(settings)
    assert header is not None
    source = tmp_path / "nested-complex.cpp"
    executable = tmp_path / "nested-complex"
    source.write_text(
        f"""
#include <cmath>
#include <complex>
{header}

template<typename T>
T evaluate(T value) {{
    return value * T(T(5e-1), T(1e0));
}}

int main() {{
    const auto result = evaluate(std::complex<double>(2.0, 3.0));
    return std::abs(result.real() + 2.0) < 1e-15
        && std::abs(result.imag() - 3.5) < 1e-15 ? 0 : 1;
}}
""",
        encoding="utf-8",
    )

    subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True)


def test_artifact_writer_preserves_direct_symjit_contract(tmp_path: Path) -> None:
    generated = _jit_adapter().artifact_manifest(tmp_path)
    serialized = _evaluator(generated)

    assert set(serialized) == {
        "application_abi",
        "application_path",
        "batch_layout",
        "compiler_type",
        "element_layout",
        "endianness",
        "evaluator_state_path",
        "evaluator_state_runtime_capability",
        "input_len",
        "kind",
        "optimization_level",
        "output_len",
        "plane_application",
        "required_defuns",
        "runtime_capability",
        "translation_mode",
        "word_bits",
    }
    assert serialized["application_path"] == generated["application_path"]
    assert serialized["runtime_capability"] == SYMJIT_F64_RUNTIME_CAPABILITY


def test_stage_manifest_rejects_jit_payload_without_direct_arena(
    tmp_path: Path,
) -> None:
    generated = _jit_adapter().artifact_manifest(tmp_path)
    amplitude_stage = {
        "stage_index": 0,
        "stage_kind": "amplitude",
        "subset_size": None,
        "evaluator_label": "amplitude",
        "parameter_layout": "stage-local-value-momentum",
        "output_length": 2,
        "output_slots": [],
        "input_value_slot_ids": [],
        "output_value_slot_ids": [],
        "interaction_ids": [],
        "input_components": [],
        "parameter_count": 3,
        "value_parameter_count": 0,
        "momentum_parameter_count": 3,
        "model_parameter_count": 0,
        "real_valued_inputs": [0, 1, 2],
        "expression_ready": True,
        "blockers": [],
        "evaluator": generated,
    }
    stage_set = {
        "kind": "generic-dag-stage-evaluator-artifacts",
        "runtime_available": True,
        "runtime_unavailable_message": None,
        "parameter_count": 0,
        "value_parameter_count": 0,
        "momentum_parameter_count": 0,
        "model_parameter_count": 0,
        "real_valued_inputs": [],
        "parameter_layout": "stage-local-value-momentum",
        "stage_count": 1,
        "required_runtime_capabilities": [SYMJIT_F64_RUNTIME_CAPABILITY],
        "stages": [],
        "amplitude_stage": amplitude_stage,
    }

    with pytest.raises(ValueError, match="compiled f64 artifacts require"):
        _stage_evaluator_set(stage_set)


def test_symjit_application_abi_matches_contributor_contract() -> None:
    lock = tomllib.loads(
        (ROOT / "dependencies" / "contributor-lock.toml").read_text(encoding="utf-8")
    )
    assert lock["abis"]["symjit_application"] == SYMJIT_APPLICATION_ABI
