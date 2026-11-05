# SPDX-License-Identifier: 0BSD
"""Fail-closed contracts for the native compiled DirectApplication producer."""

from __future__ import annotations

import math
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pyamplicol._internal.physics.types import NativeEvaluationError
from pyamplicol._internal.versions import (
    NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
)
from pyamplicol.evaluators.native_direct_cpp import (
    NativeDirectCppArtifact,
    NativeDirectCppCompiler,
    NativeDirectCppSpec,
    compile_native_direct_cpp,
    compiler_from_symbolica_settings,
    render_native_direct_cpp,
)
from pyamplicol.evaluators.native_direct_cpp import (
    NativeDirectCppParameterKind as ParameterKind,
)
from pyamplicol.evaluators.symbolica_adapters import (
    _CompiledComplexEvaluatorAdapter,
    _native_direct_application_manifest,
)
from pyamplicol.evaluators.symbolica_settings import SymbolicaEvaluatorSettings
from pyamplicol.generation.artifact_writer import _evaluator


class _Evaluator:
    def __init__(self, instructions: list[tuple[object, ...]]) -> None:
        self.instructions = instructions

    def get_instructions(self) -> tuple[list[tuple[object, ...]], int, list[complex]]:
        return self.instructions, 2, [1j]

    def save(self) -> bytes:
        return b"retained-real-evaluator-state"


def _spec(**changes: object) -> NativeDirectCppSpec:
    values: dict[str, object] = {
        "function_name": "retained_leaf",
        "parameter_kinds": (
            ParameterKind.COMPLEX_PLANE,
            ParameterKind.REAL_SCALAR,
        ),
        "output_count": 1,
        "target_triple": "aarch64-apple-darwin",
        "cpu_features": ("neon",),
        "simd_lane_width": 2,
    }
    values.update(changes)
    return NativeDirectCppSpec(**values)  # type: ignore[arg-type]


def test_real_instruction_stream_renders_plane_native_simd_and_scalar_tail() -> None:
    evaluator = _Evaluator(
        [
            ("mul", ("temp", 0), [("param", 0), ("param", 0)], 0),
            ("mul", ("temp", 1), [("param", 1), ("param", 1)], 2),
            ("pow", ("temp", 1), ("temp", 1), -1, True),
            ("add", ("out", 0), [("temp", 0), ("temp", 1)], 0),
        ]
    )

    rendered = render_native_direct_cpp(evaluator, _spec())

    assert rendered.instruction_count == 4
    assert rendered.temporary_count == 2
    assert rendered.input_plane_count == 2
    assert rendered.scalar_input_count == 1
    assert rendered.output_plane_count == 2
    assert rendered.logical_stack_bytes == 96
    assert rendered.target_triple == "aarch64-apple-darwin"
    assert rendered.cpu_features == ("neon",)
    assert rendered.simd_lane_width == 2
    assert "_complexf64(" not in rendered.source
    assert "using DirectVector = double __attribute__((vector_size(16)))" in (
        rendered.source
    )
    assert "evaluate_direct_bundle<DirectVector>" in rendered.source
    assert "evaluate_direct_bundle<double>" in rendered.source
    assert "direct_real_scalar<Lane>(*scalars[0].value)" in rendered.source
    assert 'return "aarch64-apple-darwin";' in rendered.source
    assert 'return "neon";' in rendered.source


def test_unknown_symbolica_operation_fails_without_dense_fallback() -> None:
    evaluator = _Evaluator(
        [
            (
                "fun",
                ("out", 0),
                "sqrt",
                [],
                [("param", 0)],
                False,
            )
        ]
    )

    with pytest.raises(
        NativeEvaluationError,
        match=r"cannot lower Symbolica operation.*refusing a dense-row fallback",
    ):
        render_native_direct_cpp(evaluator, _spec())


def test_exact_wavefunction_radical_renders_as_an_f64_constant() -> None:
    symbolica = pytest.importorskip("symbolica")
    momentum = symbolica.S("direct_radical_input")
    evaluator = symbolica.E("1i/sqrt(2)*direct_radical_input").evaluator(
        [momentum], iterations=0, n_cores=1, jit_compile=False
    )
    exact_state = evaluator.save()

    rendered = render_native_direct_cpp(
        evaluator,
        _spec(parameter_kinds=(ParameterKind.COMPLEX_PLANE,)),
    )

    assert (
        f"direct_constant<Lane>({math.sqrt(0.5).hex()}, {0.0.hex()})"
        in rendered.source
    )
    assert "direct_multiply(" in rendered.source
    assert evaluator.save() == exact_state


def test_real_argument_and_stack_contracts_fail_closed() -> None:
    inconsistent = _Evaluator(
        [
            ("mul", ("out", 0), [("param", 0), ("param", 0)], 2),
        ]
    )
    with pytest.raises(NativeEvaluationError, match="metadata is inconsistent"):
        render_native_direct_cpp(inconsistent, _spec())

    bounded = _Evaluator(
        [
            ("mul", ("out", 0), [("param", 0), ("param", 0)], 0),
        ]
    )
    with pytest.raises(NativeEvaluationError, match="bounded stack contract"):
        render_native_direct_cpp(
            bounded,
            _spec(max_logical_stack_bytes=1),
        )


def test_compiler_settings_retain_cpp_and_asm_direct_flags() -> None:
    compiler = compiler_from_symbolica_settings(
        {
            "compiled_inline_asm": "none",
            "compiled_optimization_level": 3,
            "compiled_native": True,
            "compiler_path": "/usr/bin/c++",
            "effective_compiler_flags": [
                "-std=c++17",
                "-fno-math-errno",
            ],
        }
    )
    assert compiler.executable == "/usr/bin/c++"
    assert compiler.optimization_level == 3
    assert compiler.native_arch is True
    assert compiler.extra_flags == ("-fno-math-errno",)
    assert compiler.timeout_seconds == 1800.0

    asm_compiler = compiler_from_symbolica_settings(
        {
            "compiled_inline_asm": "default",
            "compiled_optimization_level": 3,
            "compiled_native": False,
            "compiler_flags": ["-fno-math-errno"],
        }
    )
    assert asm_compiler.optimization_level == 3
    assert asm_compiler.native_arch is False
    assert asm_compiler.extra_flags == ("-fno-math-errno",)
    assert asm_compiler.timeout_seconds == 1800.0


def test_native_direct_cpp_timeout_is_explicit_and_cleans_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout_run(
        command: tuple[str, ...],
        *,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        Path(command[command.index("-o") + 1]).write_bytes(b"partial-library")
        raise subprocess.TimeoutExpired(command, timeout_seconds)

    monkeypatch.setattr(
        "pyamplicol.evaluators.native_direct_cpp._run_native_compiler",
        timeout_run,
    )
    output_library = tmp_path / "libretained_leaf.direct"

    with pytest.raises(
        NativeEvaluationError,
        match=r"compilation exceeded its finite 7s timeout",
    ):
        compile_native_direct_cpp(
            _Evaluator(
                [
                    ("mul", ("out", 0), [("param", 0), ("param", 0)], 0),
                ]
            ),
            spec=_spec(),
            compiler=NativeDirectCppCompiler(timeout_seconds=7.0),
            output_source_path=tmp_path / "retained_leaf.direct.cpp",
            output_library_path=output_library,
        )

    assert not output_library.exists()
    assert not tuple(tmp_path.glob(".libretained_leaf.direct.tmp-*"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract")
def test_native_direct_cpp_failure_kills_descendants_before_releasing_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler_script = tmp_path / "fake-c++"
    driver_pid_path = tmp_path / "driver.pid"
    child_pid_path = tmp_path / "child.pid"
    compiler_script.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

output = Path(sys.argv[sys.argv.index("-o") + 1])
output.write_bytes(b"partial-library")
if os.environ.get("PYAMPLICOL_TEST_COMPILER_MODE") == "success":
    raise SystemExit(0)
Path(os.environ["PYAMPLICOL_TEST_DRIVER_PID"]).write_text(str(os.getpid()))
child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        (
            "import signal,time;"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "time.sleep(60)"
        ),
    ]
)
Path(os.environ["PYAMPLICOL_TEST_CHILD_PID"]).write_text(str(child.pid))
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(60)
""",
        encoding="utf-8",
    )
    compiler_script.chmod(0o755)
    monkeypatch.setenv(
        "PYAMPLICOL_NATIVE_COMPILER_GATE_DIR",
        str(tmp_path / "compiler-gate"),
    )
    monkeypatch.setenv("PYAMPLICOL_NATIVE_COMPILER_SLOT_COUNT", "1")
    monkeypatch.setenv("PYAMPLICOL_TEST_DRIVER_PID", str(driver_pid_path))
    monkeypatch.setenv("PYAMPLICOL_TEST_CHILD_PID", str(child_pid_path))
    output_library = tmp_path / "timed-out.so"
    evaluator = _Evaluator(
        [
            ("mul", ("out", 0), [("param", 0), ("param", 0)], 0),
        ]
    )

    with pytest.raises(
        NativeEvaluationError,
        match=r"compilation exceeded its finite 0.5s timeout",
    ):
        compile_native_direct_cpp(
            evaluator,
            spec=_spec(),
            compiler=NativeDirectCppCompiler(
                executable=str(compiler_script),
                timeout_seconds=0.5,
            ),
            output_source_path=tmp_path / "timed-out.cpp",
            output_library_path=output_library,
        )

    for pid_path in (driver_pid_path, child_pid_path):
        pid = int(pid_path.read_text(encoding="ascii"))
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert not output_library.exists()
    assert not tuple(tmp_path.glob(".timed-out.so.tmp-*"))

    monkeypatch.setenv("PYAMPLICOL_TEST_COMPILER_MODE", "success")
    succeeded = compile_native_direct_cpp(
        evaluator,
        spec=_spec(),
        compiler=NativeDirectCppCompiler(
            executable=str(compiler_script),
            timeout_seconds=0.5,
        ),
        output_source_path=tmp_path / "succeeded.cpp",
        output_library_path=tmp_path / "succeeded.so",
    )
    assert succeeded.library_path.read_bytes() == b"partial-library"
    assert succeeded.compiler_gate_slot_count == 1

    for pid_path in (driver_pid_path, child_pid_path):
        pid_path.unlink()
    monkeypatch.delenv("PYAMPLICOL_TEST_COMPILER_MODE")
    original_communicate = subprocess.Popen.communicate
    interrupt_pending = True

    def interrupt_after_descendant_starts(
        process: subprocess.Popen[str],
        *args: object,
        **kwargs: object,
    ) -> tuple[str, str]:
        nonlocal interrupt_pending
        if interrupt_pending:
            deadline = time.monotonic() + 5.0
            while not child_pid_path.exists():
                if time.monotonic() >= deadline:
                    pytest.fail("fake compiler descendant did not start")
                time.sleep(0.01)
            interrupt_pending = False
            raise KeyboardInterrupt
        return original_communicate(process, *args, **kwargs)

    monkeypatch.setattr(
        subprocess.Popen,
        "communicate",
        interrupt_after_descendant_starts,
    )
    interrupted_output = tmp_path / "interrupted.so"
    with pytest.raises(KeyboardInterrupt):
        compile_native_direct_cpp(
            evaluator,
            spec=_spec(),
            compiler=NativeDirectCppCompiler(
                executable=str(compiler_script),
                timeout_seconds=60.0,
            ),
            output_source_path=tmp_path / "interrupted.cpp",
            output_library_path=interrupted_output,
        )

    for pid_path in (driver_pid_path, child_pid_path):
        pid = int(pid_path.read_text(encoding="ascii"))
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert not interrupted_output.exists()
    assert not tuple(tmp_path.glob(".interrupted.so.tmp-*"))

    monkeypatch.setenv("PYAMPLICOL_TEST_COMPILER_MODE", "success")
    succeeded_after_interrupt = compile_native_direct_cpp(
        evaluator,
        spec=_spec(),
        compiler=NativeDirectCppCompiler(
            executable=str(compiler_script),
            timeout_seconds=0.5,
        ),
        output_source_path=tmp_path / "after-interrupt.cpp",
        output_library_path=tmp_path / "after-interrupt.so",
    )
    assert succeeded_after_interrupt.library_path.read_bytes() == b"partial-library"
    assert succeeded_after_interrupt.compiler_gate_slot_count == 1


def test_native_compiler_gate_limits_four_parallel_compilers_and_reports_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate_dir = tmp_path / "compiler-gate"
    monkeypatch.setenv("PYAMPLICOL_NATIVE_COMPILER_GATE_DIR", str(gate_dir))
    monkeypatch.setenv("PYAMPLICOL_NATIVE_COMPILER_SLOT_COUNT", "4")
    four_entered = threading.Event()
    release_first_four = threading.Event()
    fifth_entered = threading.Event()
    call_lock = threading.Lock()
    call_count = 0
    observed_timeouts: list[float] = []

    def gated_run(
        command: tuple[str, ...],
        *,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        with call_lock:
            call_index = call_count
            call_count += 1
            observed_timeouts.append(timeout_seconds)
            if call_count == 4:
                four_entered.set()
        if call_index < 4:
            assert release_first_four.wait(timeout=5.0)
        else:
            fifth_entered.set()
        Path(command[command.index("-o") + 1]).write_bytes(b"native-library")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "pyamplicol.evaluators.native_direct_cpp._run_native_compiler",
        gated_run,
    )
    evaluator = _Evaluator(
        [
            ("mul", ("out", 0), [("param", 0), ("param", 0)], 0),
        ]
    )

    def compile_named(name: str) -> NativeDirectCppArtifact:
        return compile_native_direct_cpp(
            evaluator,
            spec=_spec(),
            compiler=NativeDirectCppCompiler(timeout_seconds=7.0),
            output_source_path=tmp_path / f"{name}.cpp",
            output_library_path=tmp_path / f"{name}.so",
        )

    with ThreadPoolExecutor(max_workers=5) as executor:
        first_four = [
            executor.submit(compile_named, f"parallel-{index}")
            for index in range(4)
        ]
        assert four_entered.wait(timeout=5.0)
        fifth = executor.submit(compile_named, "waiting")
        assert not fifth_entered.wait(timeout=0.1)
        release_first_four.set()
        artifacts = [
            future.result(timeout=5.0)
            for future in (*first_four, fifth)
        ]

    assert call_count == 5
    assert all(artifact.compiler_gate_slot_count == 4 for artifact in artifacts)
    assert artifacts[-1].compiler_gate_wait_seconds > 0.0
    assert observed_timeouts == [7.0] * 5


def test_native_compiler_gate_releases_after_compiler_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PYAMPLICOL_NATIVE_COMPILER_GATE_DIR",
        str(tmp_path / "compiler-gate"),
    )
    monkeypatch.setenv("PYAMPLICOL_NATIVE_COMPILER_SLOT_COUNT", "1")
    call_count = 0

    def fail_then_succeed(
        command: tuple[str, ...],
        *,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        assert timeout_seconds == 7.0
        call_count += 1
        if call_count == 1:
            raise OSError("compiler unavailable")
        Path(command[command.index("-o") + 1]).write_bytes(b"native-library")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "pyamplicol.evaluators.native_direct_cpp._run_native_compiler",
        fail_then_succeed,
    )
    evaluator = _Evaluator(
        [
            ("mul", ("out", 0), [("param", 0), ("param", 0)], 0),
        ]
    )
    compiler = NativeDirectCppCompiler(timeout_seconds=7.0)
    with pytest.raises(
        NativeEvaluationError,
        match="could not execute the native DirectApplication C\\+\\+ compiler",
    ):
        compile_native_direct_cpp(
            evaluator,
            spec=_spec(),
            compiler=compiler,
            output_source_path=tmp_path / "failed.cpp",
            output_library_path=tmp_path / "failed.so",
        )

    artifact = compile_native_direct_cpp(
        evaluator,
        spec=_spec(),
        compiler=compiler,
        output_source_path=tmp_path / "succeeded.cpp",
        output_library_path=tmp_path / "succeeded.so",
    )

    assert call_count == 2
    assert artifact.library_path.read_bytes() == b"native-library"
    assert artifact.compiler_gate_slot_count == 1


def test_direct_only_companion_identity_survives_evaluator_manifest(
    tmp_path: Path,
) -> None:
    rendered = render_native_direct_cpp(
        _Evaluator(
            [
                ("mul", ("out", 0), [("param", 0), ("param", 0)], 0),
            ]
        ),
        _spec(),
    )
    source_path = tmp_path / "retained_leaf.direct.cpp"
    library_path = tmp_path / "libretained_leaf.direct"
    source_path.write_text(rendered.source, encoding="utf-8")
    library_path.write_bytes(b"direct-only-test-library")
    companion = NativeDirectCppArtifact(
        source_path=source_path,
        library_path=library_path,
        source=rendered,
        compiler_command=("c++",),
        compile_seconds=0.125,
        compiler_gate_wait_seconds=3.5,
        compiler_gate_slot_count=4,
    )

    identity = _native_direct_application_manifest(
        companion,
        tmp_path,
        expected_function_name="retained_leaf",
    )
    evaluator = _evaluator(
        {
            "kind": "compiled-complex-evaluator",
            "runtime_capability": "symbolica.compiled-cpp.complex-f64.v1",
            "function_name": "retained_leaf",
            "input_len": 2,
            "output_len": 1,
            "library_path": "libretained_leaf.direct",
            "evaluator_state_path": "retained_leaf.evaluator.bin",
            "number_type": "complex",
            "native_direct_application": identity,
        }
    )

    assert identity["application_abi"] == NATIVE_COMPILED_DIRECT_APPLICATION_ABI
    assert identity["target"] == {
        "triple": "aarch64-apple-darwin",
        "cpu_features": ["neon"],
    }
    assert identity["source_path"] == "retained_leaf.direct.cpp"
    assert identity["library_path"] == "libretained_leaf.direct"
    assert "compiler_gate_wait_seconds" not in identity
    assert "compiler_gate_slot_count" not in identity
    assert evaluator["native_direct_application"] == identity

    asm_evaluator = _evaluator(
        {
            "kind": "compiled-complex-evaluator",
            "runtime_capability": "symbolica.compiled-asm.complex-f64.v1",
            "function_name": "retained_leaf",
            "input_len": 2,
            "output_len": 1,
            "library_path": "libretained_leaf.direct",
            "evaluator_state_path": "retained_leaf.evaluator.bin",
            "number_type": "complex",
            "native_direct_application": identity,
        }
    )
    assert asm_evaluator["native_direct_application"] == identity


def test_process_stage_adapter_serializes_no_dense_library(tmp_path: Path) -> None:
    source = _Evaluator(
        [
            ("mul", ("out", 0), [("param", 0), ("param", 0)], 0),
        ]
    )
    adapter = _CompiledComplexEvaluatorAdapter(
        source,
        SymbolicaEvaluatorSettings(
            backend="compiled-complex",
            compiled_inline_asm="none",
            compiled_output_dir=str(tmp_path),
        ),
        "direct only stage",
        input_len=1,
        output_len=1,
        compile_dense=False,
    )
    rendered = render_native_direct_cpp(
        source,
        _spec(
            function_name=adapter.function_name,
            parameter_kinds=(ParameterKind.COMPLEX_PLANE,),
        ),
    )
    direct_source = tmp_path / f"{adapter.function_name}.direct.cpp"
    direct_library = tmp_path / f"lib{adapter.function_name}.direct"
    direct_source.write_text(rendered.source, encoding="utf-8")
    direct_library.write_bytes(b"direct-only-test-library")
    adapter.native_direct_application = NativeDirectCppArtifact(
        source_path=direct_source,
        library_path=direct_library,
        source=rendered,
        compiler_command=("c++",),
        compile_seconds=0.125,
    )

    manifest = adapter.artifact_manifest(tmp_path)

    assert manifest["library_path"] == direct_library.name
    assert manifest["source_path"] == direct_source.name
    assert not adapter.library_path.exists()
    assert not adapter.source_path.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [
            direct_source.name,
            direct_library.name,
            adapter.evaluator_state_path.name,
        ]
    )
