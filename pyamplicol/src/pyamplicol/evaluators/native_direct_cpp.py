# SPDX-License-Identifier: 0BSD
"""Generate a genuine plane-native C++ leaf from a Symbolica evaluator.

This module is deliberately a narrow producer prototype.  It consumes
Symbolica's documented ``Evaluator.get_instructions()`` representation and
emits that optimized instruction stream a second time against fixed
split-complex plane/scalar descriptors.  The generated DirectApplication
entry never calls the historical ``*_complexf64`` function and never builds a
dense parameter or output row.

Only the instruction subset proved by the retained real-process prototype is
accepted.  Unknown operations fail closed instead of falling back to the
dense ABI.
"""

from __future__ import annotations

import cmath
import hashlib
import math
import os
import re
import signal
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .._internal.physics.types import NativeEvaluationError
from .._internal.versions import NATIVE_COMPILED_DIRECT_APPLICATION_ABI
from .symbolica_helpers import _symbolica_instruction_program

_C_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DIRECT_MARKER = "// pyAmpliCol genuine native DirectApplication producer v1"
_DIRECT_FLAGS = 0x3F
_DEFAULT_STACK_LIMIT = 64 * 1024
_SUPPORTED_LANE_WIDTHS = frozenset({2, 4})
_COMPILER_GATE_DIR_ENV = "PYAMPLICOL_NATIVE_COMPILER_GATE_DIR"
_COMPILER_GATE_SLOT_COUNT_ENV = "PYAMPLICOL_NATIVE_COMPILER_SLOT_COUNT"
_COMPILER_GATE_POLL_SECONDS = 0.05
_MAX_COMPILER_GATE_SLOTS = 64
_COMPILER_TERMINATE_GRACE_SECONDS = 0.5
_COMPILER_KILL_GRACE_SECONDS = 5.0


class NativeDirectCppParameterKind(StrEnum):
    """Storage class for one complex Symbolica source parameter."""

    COMPLEX_PLANE = "complex-plane"
    REAL_PLANE = "real-plane"
    COMPLEX_SCALAR = "complex-scalar"
    REAL_SCALAR = "real-scalar"


@dataclass(frozen=True, slots=True)
class NativeDirectCppSpec:
    """Cold-path code-generation contract for one fused evaluator leaf."""

    function_name: str
    parameter_kinds: tuple[NativeDirectCppParameterKind, ...]
    output_count: int
    target_triple: str
    cpu_features: tuple[str, ...] = ()
    simd_lane_width: int = 2
    max_logical_stack_bytes: int = _DEFAULT_STACK_LIMIT

    def __post_init__(self) -> None:
        if _C_IDENTIFIER.fullmatch(self.function_name) is None:
            raise NativeEvaluationError(
                "native DirectApplication function name is not a portable C identifier"
            )
        if not self.parameter_kinds:
            raise NativeEvaluationError(
                "native DirectApplication needs at least one source parameter"
            )
        if self.output_count < 1:
            raise NativeEvaluationError(
                "native DirectApplication needs at least one output"
            )
        if not self.target_triple or "\x00" in self.target_triple:
            raise NativeEvaluationError(
                "native DirectApplication target triple is invalid"
            )
        canonical_features = tuple(sorted(set(self.cpu_features)))
        if self.cpu_features != canonical_features or any(
            not feature or "\x00" in feature for feature in self.cpu_features
        ):
            raise NativeEvaluationError(
                "native DirectApplication CPU features must be sorted and unique"
            )
        if self.simd_lane_width not in _SUPPORTED_LANE_WIDTHS:
            raise NativeEvaluationError(
                "native DirectApplication SIMD lane width must be 2 or 4"
            )
        if self.max_logical_stack_bytes < 1:
            raise NativeEvaluationError(
                "native DirectApplication logical stack limit must be positive"
            )


@dataclass(frozen=True, slots=True)
class NativeDirectCppSource:
    """Rendered source and its authenticated producer metadata."""

    source: str
    target_triple: str
    cpu_features: tuple[str, ...]
    simd_lane_width: int
    evaluator_state_sha256: str
    instruction_count: int
    temporary_count: int
    input_plane_count: int
    scalar_input_count: int
    output_plane_count: int
    logical_stack_bytes: int


@dataclass(frozen=True, slots=True)
class NativeDirectCppCompiler:
    """Compiler settings matching the ordinary compiled-C++ path."""

    executable: str = "c++"
    optimization_level: int = 3
    native_arch: bool = False
    extra_flags: tuple[str, ...] = ()
    timeout_seconds: float = 1800.0

    def __post_init__(self) -> None:
        if not self.executable or "\x00" in self.executable:
            raise NativeEvaluationError("native DirectApplication compiler is invalid")
        if self.optimization_level not in (0, 1, 2, 3):
            raise NativeEvaluationError(
                "native DirectApplication compiler optimization must be 0 through 3"
            )
        if self.timeout_seconds <= 0 or not math.isfinite(self.timeout_seconds):
            raise NativeEvaluationError(
                "native DirectApplication compiler timeout must be finite and positive"
            )
        if any(not flag or "\x00" in flag for flag in self.extra_flags):
            raise NativeEvaluationError(
                "native DirectApplication compiler flags are invalid"
            )


@dataclass(frozen=True, slots=True)
class NativeDirectCppArtifact:
    """One direct-only native library produced from retained instructions."""

    source_path: Path
    library_path: Path
    source: NativeDirectCppSource
    compiler_command: tuple[str, ...]
    compile_seconds: float
    compiler_gate_wait_seconds: float = 0.0
    compiler_gate_slot_count: int | None = None


@dataclass(frozen=True, slots=True)
class _NativeCompilerGateLease:
    wait_seconds: float
    slot_count: int | None


@dataclass(frozen=True, slots=True)
class _ParameterAccess:
    kind: NativeDirectCppParameterKind
    first_index: int
    second_index: int | None


@dataclass(frozen=True, slots=True)
class _InstructionProgram:
    statements: tuple[str, ...]
    instruction_count: int
    temporary_count: int
    constants: tuple[complex, ...]


def render_native_direct_cpp(
    evaluator: Any,
    spec: NativeDirectCppSpec,
) -> NativeDirectCppSource:
    """Render one factor-free plane-native entry from a Symbolica evaluator.

    ``Evaluator.get_instructions`` is the only accepted expression source.
    Generated dense C++ text is never parsed or wrapped.
    """

    get_instructions = getattr(evaluator, "get_instructions", None)
    save = getattr(evaluator, "save", None)
    if not callable(get_instructions) or not callable(save):
        raise NativeEvaluationError(
            "this Symbolica evaluator exposes no reusable instruction/state API"
        )
    try:
        raw_program = _symbolica_instruction_program(get_instructions())
        evaluator_state = save()
    except Exception as error:
        raise NativeEvaluationError(
            "Symbolica could not expose its optimized evaluator instruction stream"
        ) from error
    if not isinstance(evaluator_state, bytes) or not evaluator_state:
        raise NativeEvaluationError(
            "Symbolica returned an empty or non-bytes evaluator state"
        )

    accesses, input_plane_count, scalar_input_count = _parameter_accesses(
        spec.parameter_kinds
    )
    program = _lower_instruction_program(raw_program, spec, accesses)
    logical_stack_bytes = (
        (program.temporary_count + spec.output_count) * 2 * 8 * spec.simd_lane_width
    )
    if logical_stack_bytes > spec.max_logical_stack_bytes:
        raise NativeEvaluationError(
            "native DirectApplication logical scratch exceeds its bounded stack "
            f"contract: {logical_stack_bytes} > {spec.max_logical_stack_bytes}"
        )

    state_sha256 = hashlib.sha256(evaluator_state).hexdigest()
    source = _render_translation_unit(
        spec=spec,
        program=program,
        accesses=accesses,
        input_plane_count=input_plane_count,
        scalar_input_count=scalar_input_count,
        evaluator_state_sha256=state_sha256,
        logical_stack_bytes=logical_stack_bytes,
    )
    return NativeDirectCppSource(
        source=source,
        target_triple=spec.target_triple,
        cpu_features=spec.cpu_features,
        simd_lane_width=spec.simd_lane_width,
        evaluator_state_sha256=state_sha256,
        instruction_count=program.instruction_count,
        temporary_count=program.temporary_count,
        input_plane_count=input_plane_count,
        scalar_input_count=scalar_input_count,
        output_plane_count=2 * spec.output_count,
        logical_stack_bytes=logical_stack_bytes,
    )


def compile_native_direct_cpp(
    evaluator: Any,
    *,
    spec: NativeDirectCppSpec,
    compiler: NativeDirectCppCompiler,
    output_source_path: Path,
    output_library_path: Path,
) -> NativeDirectCppArtifact:
    """Compile one genuine direct-only native leaf.

    The translation unit is generated independently from retained evaluator
    instructions.  It contains neither the ordinary dense source nor a call to
    a dense symbol.  Validation may load the original dense library separately
    as an exact same-expression oracle.
    """

    rendered = render_native_direct_cpp(evaluator, spec)
    output_source_path.parent.mkdir(parents=True, exist_ok=True)
    output_library_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_source_path.write_text(rendered.source, encoding="utf-8")
    except OSError as error:
        raise NativeEvaluationError(
            f"could not write native DirectApplication source: {output_source_path}"
        ) from error

    temporary_library = output_library_path.with_name(
        f".{output_library_path.name}.tmp-{os.getpid()}"
    )
    command = [
        compiler.executable,
        "-std=c++17",
        "-shared",
        f"-O{compiler.optimization_level}",
        "-fPIC",
        "-ffast-math",
        "-funsafe-math-optimizations",
    ]
    if compiler.native_arch:
        command.append("-march=native")
    command.extend(compiler.extra_flags)
    command.extend(
        [
            str(output_source_path),
            "-o",
            str(temporary_library),
        ]
    )
    with _native_compiler_gate() as compiler_gate:
        # Queueing behind another compiler is operational scheduling, not
        # compilation time.  Start the finite subprocess timeout only after
        # this worker owns one shared slot.
        started = time.perf_counter()
        try:
            completed = _run_native_compiler(
                command,
                timeout_seconds=compiler.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            temporary_library.unlink(missing_ok=True)
            raise NativeEvaluationError(
                "native DirectApplication C++ compilation exceeded its finite "
                f"{compiler.timeout_seconds:g}s timeout"
            ) from error
        except NativeEvaluationError:
            temporary_library.unlink(missing_ok=True)
            raise
        except (OSError, subprocess.SubprocessError) as error:
            temporary_library.unlink(missing_ok=True)
            raise NativeEvaluationError(
                "could not execute the native DirectApplication C++ compiler"
            ) from error
        except BaseException:
            # Interrupts must not publish a partial library or release the
            # shared compiler slot before the detached compiler tree is gone.
            temporary_library.unlink(missing_ok=True)
            raise
    compile_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        temporary_library.unlink(missing_ok=True)
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise NativeEvaluationError(
            "native DirectApplication C++ compilation failed"
            + (f": {detail}" if detail else "")
        )
    try:
        os.replace(temporary_library, output_library_path)
    except OSError as error:
        temporary_library.unlink(missing_ok=True)
        raise NativeEvaluationError(
            f"could not publish native DirectApplication library: {output_library_path}"
        ) from error
    return NativeDirectCppArtifact(
        source_path=output_source_path,
        library_path=output_library_path,
        source=rendered,
        compiler_command=tuple(command),
        compile_seconds=compile_seconds,
        compiler_gate_wait_seconds=compiler_gate.wait_seconds,
        compiler_gate_slot_count=compiler_gate.slot_count,
    )


def _run_native_compiler(
    command: Sequence[str],
    *,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    """Run one compiler in a disposable process group with finite cleanup."""

    process = subprocess.Popen(
        tuple(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except BaseException:
        # The compiler owns a new session on POSIX, so caller interruption
        # does not reach it automatically.  Reap the complete tree before
        # propagating any timeout, cancellation, or unexpected failure.
        _terminate_native_compiler_tree(process)
        raise
    return subprocess.CompletedProcess(
        tuple(command),
        process.returncode,
        stdout,
        stderr,
    )


def _terminate_native_compiler_tree(process: subprocess.Popen[str]) -> None:
    """Terminate and reap a timed-out compiler and all of its descendants."""

    if os.name != "posix":
        process.kill()
        try:
            process.communicate(timeout=_COMPILER_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise NativeEvaluationError(
                "native DirectApplication compiler timeout cleanup failed"
            ) from error
        return

    process_group_id = process.pid
    _signal_process_group(process_group_id, signal.SIGTERM)
    try:
        process.communicate(timeout=_COMPILER_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass

    if _process_group_exists(process_group_id):
        _signal_process_group(process_group_id, signal.SIGKILL)
    try:
        process.communicate(timeout=_COMPILER_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired as error:
        _signal_process_group(process_group_id, signal.SIGKILL)
        raise NativeEvaluationError(
            "native DirectApplication compiler timeout cleanup failed"
        ) from error

    deadline = time.monotonic() + _COMPILER_KILL_GRACE_SECONDS
    while _process_group_exists(process_group_id):
        if time.monotonic() >= deadline:
            _signal_process_group(process_group_id, signal.SIGKILL)
            raise NativeEvaluationError(
                "native DirectApplication compiler timeout cleanup left a "
                "live process group"
            )
        time.sleep(0.01)


def _signal_process_group(process_group_id: int, signal_number: int) -> None:
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        return
    except OSError as error:
        raise NativeEvaluationError(
            "native DirectApplication compiler process-group cleanup failed"
        ) from error


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _native_compiler_gate() -> Iterator[_NativeCompilerGateLease]:
    """Lease one optional cross-controller native-compiler slot.

    The report scheduler injects the gate only for the EPYC campaign.  Other
    callers retain the historical ungated behavior.  Slot paths and wait time
    are deliberately operational metadata and never enter evaluator identity.
    """

    raw_directory = os.environ.get(_COMPILER_GATE_DIR_ENV)
    raw_slot_count = os.environ.get(_COMPILER_GATE_SLOT_COUNT_ENV)
    if raw_directory is None and raw_slot_count is None:
        yield _NativeCompilerGateLease(0.0, None)
        return
    if raw_directory is None or raw_slot_count is None:
        raise NativeEvaluationError(
            "native DirectApplication compiler gate environment is incomplete"
        )
    if not raw_directory or "\x00" in raw_directory:
        raise NativeEvaluationError(
            "native DirectApplication compiler gate directory is invalid"
        )
    try:
        slot_count = int(raw_slot_count)
    except ValueError as error:
        raise NativeEvaluationError(
            "native DirectApplication compiler gate slot count is invalid"
        ) from error
    if not 1 <= slot_count <= _MAX_COMPILER_GATE_SLOTS:
        raise NativeEvaluationError(
            "native DirectApplication compiler gate slot count is invalid"
        )
    directory = Path(raw_directory)
    if not directory.is_absolute():
        raise NativeEvaluationError(
            "native DirectApplication compiler gate directory must be absolute"
        )

    descriptors: list[int] = []
    try:
        import fcntl

        directory.mkdir(parents=True, exist_ok=True)
        for index in range(slot_count):
            descriptors.append(
                os.open(
                    directory / f"slot-{index:02d}.lock",
                    os.O_CREAT | os.O_RDWR,
                    0o600,
                )
            )
    except (ImportError, OSError) as error:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise NativeEvaluationError(
            "native DirectApplication compiler gate could not be initialized"
        ) from error

    acquired: int | None = None
    wait_started = time.perf_counter()
    try:
        while acquired is None:
            for descriptor in descriptors:
                try:
                    fcntl.flock(
                        descriptor,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError:
                    continue
                except OSError as error:
                    raise NativeEvaluationError(
                        "native DirectApplication compiler gate acquisition failed"
                    ) from error
                acquired = descriptor
                break
            if acquired is None:
                time.sleep(_COMPILER_GATE_POLL_SECONDS)
        yield _NativeCompilerGateLease(
            time.perf_counter() - wait_started,
            slot_count,
        )
    finally:
        if acquired is not None:
            try:
                fcntl.flock(acquired, fcntl.LOCK_UN)
            except OSError:
                pass
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


def compiler_from_symbolica_settings(
    settings: Mapping[str, object],
) -> NativeDirectCppCompiler:
    """Translate the adapter's recorded settings into the direct compiler."""

    # The direct translation is intentionally independent from Symbolica's
    # dense scalar-row emitter.  An ``asm`` public backend therefore reuses the
    # same optimized instruction stream and compiler policy here, without
    # parsing, wrapping, or calling its inline-ASM row function.
    executable_value = settings.get("compiler_path")
    executable = "c++" if executable_value is None else str(executable_value)
    optimization = settings.get("compiled_optimization_level", 3)
    if isinstance(optimization, bool) or not isinstance(optimization, int):
        raise NativeEvaluationError(
            "native DirectApplication compiler optimization metadata is invalid"
        )
    native_arch = settings.get("compiled_native", False)
    if not isinstance(native_arch, bool):
        raise NativeEvaluationError(
            "native DirectApplication native-architecture metadata is invalid"
        )
    raw_flags = settings.get(
        "effective_compiler_flags",
        settings.get("compiler_flags", ()),
    )
    if not isinstance(raw_flags, Sequence) or isinstance(raw_flags, (str, bytes)):
        raise NativeEvaluationError(
            "native DirectApplication compiler flag metadata is invalid"
        )
    extra_flags = tuple(str(flag) for flag in raw_flags if str(flag) != "-std=c++17")
    return NativeDirectCppCompiler(
        executable=executable,
        optimization_level=optimization,
        native_arch=native_arch,
        extra_flags=extra_flags,
    )


def _parameter_accesses(
    kinds: Sequence[NativeDirectCppParameterKind],
) -> tuple[tuple[_ParameterAccess, ...], int, int]:
    plane_index = 0
    scalar_index = 0
    accesses: list[_ParameterAccess] = []
    for raw_kind in kinds:
        try:
            kind = NativeDirectCppParameterKind(raw_kind)
        except ValueError as error:
            raise NativeEvaluationError(
                f"unsupported native DirectApplication parameter kind: {raw_kind!r}"
            ) from error
        if kind is NativeDirectCppParameterKind.COMPLEX_PLANE:
            accesses.append(_ParameterAccess(kind, plane_index, plane_index + 1))
            plane_index += 2
        elif kind is NativeDirectCppParameterKind.REAL_PLANE:
            accesses.append(_ParameterAccess(kind, plane_index, None))
            plane_index += 1
        elif kind is NativeDirectCppParameterKind.COMPLEX_SCALAR:
            accesses.append(_ParameterAccess(kind, scalar_index, scalar_index + 1))
            scalar_index += 2
        else:
            accesses.append(_ParameterAccess(kind, scalar_index, None))
            scalar_index += 1
    return tuple(accesses), plane_index, scalar_index


def _lower_instruction_program(
    raw_program: object,
    spec: NativeDirectCppSpec,
    accesses: Sequence[_ParameterAccess],
) -> _InstructionProgram:
    if not isinstance(raw_program, tuple) or len(raw_program) != 3:
        raise NativeEvaluationError(
            "Symbolica returned an invalid evaluator instruction program"
        )
    raw_instructions, raw_temporary_count, raw_constants = raw_program
    if not isinstance(raw_instructions, list):
        raise NativeEvaluationError("Symbolica evaluator instructions are not a list")
    if (
        isinstance(raw_temporary_count, bool)
        or not isinstance(raw_temporary_count, int)
        or raw_temporary_count < 0
    ):
        raise NativeEvaluationError("Symbolica evaluator temporary count is invalid")
    if not isinstance(raw_constants, list):
        raise NativeEvaluationError("Symbolica evaluator constants are not a list")
    constants = tuple(_finite_complex(value) for value in raw_constants)

    written: set[tuple[str, int]] = set()
    real_values: dict[tuple[str, int], bool] = {}
    for index, access in enumerate(accesses):
        real_values[("param", index)] = access.kind in {
            NativeDirectCppParameterKind.REAL_PLANE,
            NativeDirectCppParameterKind.REAL_SCALAR,
        }
    for index, value in enumerate(constants):
        real_values[("const", index)] = value.imag == 0.0

    def parse_reference(raw: object, *, destination: bool = False) -> tuple[str, int]:
        if (
            not isinstance(raw, tuple)
            or len(raw) != 2
            or not isinstance(raw[0], str)
            or isinstance(raw[1], bool)
            or not isinstance(raw[1], int)
            or raw[1] < 0
        ):
            raise NativeEvaluationError(
                "Symbolica evaluator instruction contains an invalid reference"
            )
        kind, index = raw
        allowed = {"temp", "out"} if destination else {"param", "temp", "const", "out"}
        if kind not in allowed:
            raise NativeEvaluationError(
                "Symbolica evaluator instruction reference kind "
                f"{kind!r} is unsupported"
            )
        if kind == "param" and index >= len(accesses):
            raise NativeEvaluationError(
                "Symbolica evaluator parameter reference is out of bounds"
            )
        if kind == "temp" and index >= raw_temporary_count:
            raise NativeEvaluationError(
                "Symbolica evaluator temporary reference is out of bounds"
            )
        if kind == "const" and index >= len(constants):
            raise NativeEvaluationError(
                "Symbolica evaluator constant reference is out of bounds"
            )
        if kind == "out" and index >= spec.output_count:
            raise NativeEvaluationError(
                "Symbolica evaluator output reference is out of bounds"
            )
        if not destination and kind in {"temp", "out"} and (kind, index) not in written:
            raise NativeEvaluationError(
                "Symbolica evaluator reads an uninitialized temporary/output"
            )
        return kind, index

    def reference_expression(reference: tuple[str, int]) -> str:
        kind, index = reference
        if kind == "temp":
            return f"temporary[{index}]"
        if kind == "out":
            return f"output[{index}]"
        if kind == "const":
            value = constants[index]
            return (
                f"direct_constant<Lane>({_cpp_f64(value.real)}, {_cpp_f64(value.imag)})"
            )
        access = accesses[index]
        if access.kind is NativeDirectCppParameterKind.COMPLEX_PLANE:
            assert access.second_index is not None
            return (
                "direct_complex_plane<Lane>("
                f"inputs[{access.first_index}].values, "
                f"inputs[{access.second_index}].values, point)"
            )
        if access.kind is NativeDirectCppParameterKind.REAL_PLANE:
            return (
                f"direct_real_plane<Lane>(inputs[{access.first_index}].values, point)"
            )
        if access.kind is NativeDirectCppParameterKind.COMPLEX_SCALAR:
            assert access.second_index is not None
            return (
                "direct_complex_scalar<Lane>("
                f"*scalars[{access.first_index}].value, "
                f"*scalars[{access.second_index}].value)"
            )
        return f"direct_real_scalar<Lane>(*scalars[{access.first_index}].value)"

    statements: list[str] = []
    for row_index, raw_row in enumerate(raw_instructions):
        if not isinstance(raw_row, tuple) or not raw_row:
            raise NativeEvaluationError(
                f"Symbolica evaluator instruction {row_index} is invalid"
            )
        operation = raw_row[0]
        if operation in {"add", "mul"}:
            if len(raw_row) != 4 or not isinstance(raw_row[2], list):
                raise NativeEvaluationError(
                    f"Symbolica {operation} instruction has an invalid shape"
                )
            destination = parse_reference(raw_row[1], destination=True)
            arguments = tuple(parse_reference(item) for item in raw_row[2])
            real_count = raw_row[3]
            if (
                not arguments
                or isinstance(real_count, bool)
                or not isinstance(real_count, int)
                or real_count < 0
                or real_count > len(arguments)
            ):
                raise NativeEvaluationError(
                    f"Symbolica {operation} instruction has invalid "
                    "real-argument metadata"
                )
            if any(
                not real_values.get(argument, False)
                for argument in arguments[:real_count]
            ):
                raise NativeEvaluationError(
                    f"Symbolica {operation} instruction real-argument "
                    "metadata is inconsistent"
                )
            expression = reference_expression(arguments[0])
            helper = "direct_add" if operation == "add" else "direct_multiply"
            for argument in arguments[1:]:
                expression = f"{helper}({expression}, {reference_expression(argument)})"
            statements.append(
                f"    {reference_expression(destination)} = {expression};"
            )
            written.add(destination)
            real_values[destination] = real_count == len(arguments)
            continue
        if operation == "pow":
            if len(raw_row) != 5:
                raise NativeEvaluationError(
                    "Symbolica pow instruction has an invalid shape"
                )
            destination = parse_reference(raw_row[1], destination=True)
            base = parse_reference(raw_row[2])
            exponent = raw_row[3]
            output_is_real = raw_row[4]
            if exponent != -1 or not isinstance(output_is_real, bool):
                raise NativeEvaluationError(
                    "native DirectApplication only supports Symbolica "
                    "reciprocal pow(-1)"
                )
            if output_is_real and not real_values.get(base, False):
                raise NativeEvaluationError(
                    "Symbolica real reciprocal metadata is inconsistent"
                )
            helper = (
                "direct_real_reciprocal"
                if output_is_real
                else "direct_complex_reciprocal"
            )
            statements.append(
                f"    {reference_expression(destination)} = "
                f"{helper}({reference_expression(base)});"
            )
            written.add(destination)
            real_values[destination] = output_is_real
            continue
        if operation == "assign":
            if len(raw_row) != 3:
                raise NativeEvaluationError(
                    "Symbolica assign instruction has an invalid shape"
                )
            destination = parse_reference(raw_row[1], destination=True)
            source = parse_reference(raw_row[2])
            statements.append(
                f"    {reference_expression(destination)} = "
                f"{reference_expression(source)};"
            )
            written.add(destination)
            real_values[destination] = real_values.get(source, False)
            continue
        if operation == "fun" and len(raw_row) == 6:
            function_name = getattr(raw_row[2], "get_name", None)
            arguments = raw_row[4]
            if (
                callable(function_name)
                and function_name() == "symbolica::sqrt"
                and raw_row[3] == []
                and isinstance(arguments, list)
                and len(arguments) == 1
                and isinstance(raw_row[5], bool)
            ):
                destination = parse_reference(raw_row[1], destination=True)
                source = parse_reference(arguments[0])
                if source[0] == "const":
                    # Keep the saved evaluator exact; round constant radicals
                    # only when emitting this f64 C++ evaluation path.
                    value = cmath.sqrt(constants[source[1]])
                    if raw_row[5] and value.imag != 0.0:
                        raise NativeEvaluationError(
                            "Symbolica real square-root metadata is inconsistent"
                        )
                    statements.append(
                        f"    {reference_expression(destination)} = "
                        f"direct_constant<Lane>({_cpp_f64(value.real)}, "
                        f"{_cpp_f64(value.imag)});"
                    )
                    written.add(destination)
                    real_values[destination] = value.imag == 0.0
                    continue
        raise NativeEvaluationError(
            "native DirectApplication cannot lower Symbolica operation "
            f"{operation!r}; refusing a dense-row fallback"
        )
    missing_outputs = [
        index for index in range(spec.output_count) if ("out", index) not in written
    ]
    if missing_outputs:
        raise NativeEvaluationError(
            "Symbolica evaluator instructions do not assign every output: "
            + ", ".join(str(index) for index in missing_outputs)
        )
    return _InstructionProgram(
        statements=tuple(statements),
        instruction_count=len(raw_instructions),
        temporary_count=raw_temporary_count,
        constants=constants,
    )


def _finite_complex(value: object) -> complex:
    try:
        converted = complex(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise NativeEvaluationError(
            "Symbolica evaluator constant cannot be represented as complex-f64"
        ) from error
    if not math.isfinite(converted.real) or not math.isfinite(converted.imag):
        raise NativeEvaluationError(
            "Symbolica evaluator constant is not finite complex-f64"
        )
    return converted


def _cpp_f64(value: float) -> str:
    if not math.isfinite(value):
        raise NativeEvaluationError(
            "native DirectApplication cannot emit non-finite constants"
        )
    return value.hex()


def _cpp_string(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _render_translation_unit(
    *,
    spec: NativeDirectCppSpec,
    program: _InstructionProgram,
    accesses: Sequence[_ParameterAccess],
    input_plane_count: int,
    scalar_input_count: int,
    evaluator_state_sha256: str,
    logical_stack_bytes: int,
) -> str:
    function_name = spec.function_name
    vector_bytes = spec.simd_lane_width * 8
    output_plane_count = 2 * spec.output_count
    temporary_extent = max(program.temporary_count, 1)
    output_extent = max(spec.output_count, 1)
    feature_string = ",".join(spec.cpu_features)
    statements = "\n".join(program.statements)
    stores = "\n".join(
        (
            f"    direct_store_lane(outputs[{2 * index}].values, point, "
            f"output[{index}].real);\n"
            f"    direct_store_lane(outputs[{2 * index + 1}].values, point, "
            f"output[{index}].imaginary);"
        )
        for index in range(spec.output_count)
    )
    del accesses  # accesses are fully compiled into the generated statements.
    return f"""{_DIRECT_MARKER}
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#if !defined(__clang__) && !defined(__GNUC__)
#error "pyAmpliCol native DirectApplication requires Clang or GCC vector extensions"
#endif

namespace pyamplicol_native_direct_{function_name} {{

constexpr std::uint32_t kInputPlaneCount = {input_plane_count}u;
constexpr std::uint32_t kScalarInputCount = {scalar_input_count}u;
constexpr std::uint32_t kOutputPlaneCount = {output_plane_count}u;
constexpr std::uint32_t kSimdLaneWidth = {spec.simd_lane_width}u;
constexpr std::uint32_t kInstructionCount = {program.instruction_count}u;
constexpr std::uint32_t kTemporaryCount = {program.temporary_count}u;
constexpr std::uint32_t kLogicalStackBytes = {logical_stack_bytes}u;
static_assert(kLogicalStackBytes <= {spec.max_logical_stack_bytes}u);

using DirectVector = double __attribute__((vector_size({vector_bytes})));

template <typename Lane>
struct DirectComplex {{
    Lane real;
    Lane imaginary;
}};

template <typename Lane>
inline Lane direct_broadcast(double value) noexcept {{
    if constexpr (std::is_same_v<Lane, double>) {{
        return value;
    }} else {{
        Lane result{{}};
        for (std::size_t lane = 0; lane < sizeof(Lane) / sizeof(double); ++lane) {{
            result[lane] = value;
        }}
        return result;
    }}
}}

template <typename Lane>
inline Lane direct_load_lane(const double* values, std::uint32_t point) noexcept {{
    Lane result;
    std::memcpy(&result, values + point, sizeof(Lane));
    return result;
}}

template <typename Lane>
inline void direct_store_lane(
    double* values,
    std::uint32_t point,
    Lane value
) noexcept {{
    std::memcpy(values + point, &value, sizeof(Lane));
}}

template <typename Lane>
inline DirectComplex<Lane> direct_constant(double real, double imaginary) noexcept {{
    return {{direct_broadcast<Lane>(real), direct_broadcast<Lane>(imaginary)}};
}}

template <typename Lane>
inline DirectComplex<Lane> direct_complex_plane(
    const double* real,
    const double* imaginary,
    std::uint32_t point
) noexcept {{
    return {{direct_load_lane<Lane>(real, point),
             direct_load_lane<Lane>(imaginary, point)}};
}}

template <typename Lane>
inline DirectComplex<Lane> direct_real_plane(
    const double* real,
    std::uint32_t point
) noexcept {{
    return {{direct_load_lane<Lane>(real, point), direct_broadcast<Lane>(0.0)}};
}}

template <typename Lane>
inline DirectComplex<Lane> direct_complex_scalar(
    double real,
    double imaginary
) noexcept {{
    return direct_constant<Lane>(real, imaginary);
}}

template <typename Lane>
inline DirectComplex<Lane> direct_real_scalar(double real) noexcept {{
    return direct_constant<Lane>(real, 0.0);
}}

template <typename Lane>
inline DirectComplex<Lane> direct_add(
    DirectComplex<Lane> left,
    DirectComplex<Lane> right
) noexcept {{
    return {{left.real + right.real, left.imaginary + right.imaginary}};
}}

template <typename Lane>
inline DirectComplex<Lane> direct_multiply(
    DirectComplex<Lane> left,
    DirectComplex<Lane> right
) noexcept {{
    return {{
        left.real * right.real - left.imaginary * right.imaginary,
        left.real * right.imaginary + left.imaginary * right.real,
    }};
}}

template <typename Lane>
inline DirectComplex<Lane> direct_real_reciprocal(
    DirectComplex<Lane> value
) noexcept {{
    return {{direct_broadcast<Lane>(1.0) / value.real,
             direct_broadcast<Lane>(0.0)}};
}}

template <typename Lane>
inline DirectComplex<Lane> direct_complex_reciprocal(
    DirectComplex<Lane> value
) noexcept {{
    const Lane denominator =
        value.real * value.real + value.imaginary * value.imaginary;
    return {{value.real / denominator, -value.imaginary / denominator}};
}}

struct NativeCompiledDirectInputPlaneV1 {{
    const double* values;
}};

struct NativeCompiledDirectOutputPlaneV1 {{
    double* values;
}};

struct NativeCompiledDirectScalarV1 {{
    const double* value;
}};

struct NativeCompiledDirectMetadataV1 {{
    std::uint32_t abi_version;
    std::uint32_t struct_size;
    std::uint32_t flags;
    std::uint32_t input_plane_count;
    std::uint32_t scalar_input_count;
    std::uint32_t output_plane_count;
    std::uint32_t simd_lane_width;
    std::uint32_t reserved;
}};

template <typename Lane>
inline void evaluate_direct_bundle(
    const NativeCompiledDirectInputPlaneV1* inputs,
    const NativeCompiledDirectScalarV1* scalars,
    const NativeCompiledDirectOutputPlaneV1* outputs,
    std::uint32_t point
) noexcept {{
    std::array<DirectComplex<Lane>, {temporary_extent}> temporary;
    std::array<DirectComplex<Lane>, {output_extent}> output;
{statements}
{stores}
}}

}}  // namespace pyamplicol_native_direct_{function_name}

extern "C" pyamplicol_native_direct_{function_name}::NativeCompiledDirectMetadataV1
{function_name}_direct_application_v1_metadata() noexcept {{
    using namespace pyamplicol_native_direct_{function_name};
    return {{
        1u,
        static_cast<std::uint32_t>(sizeof(NativeCompiledDirectMetadataV1)),
        {_DIRECT_FLAGS}u,
        kInputPlaneCount,
        kScalarInputCount,
        kOutputPlaneCount,
        kSimdLaneWidth,
        0u,
    }};
}}

extern "C" const char*
{function_name}_direct_application_v1_target_triple() noexcept {{
    return "{_cpp_string(spec.target_triple)}";
}}

extern "C" const char*
{function_name}_direct_application_v1_cpu_features() noexcept {{
    return "{_cpp_string(feature_string)}";
}}

extern "C" const char*
{function_name}_direct_application_v1_source_sha256() noexcept {{
    return "{evaluator_state_sha256}";
}}

extern "C" std::uint32_t
{function_name}_direct_application_v1_logical_stack_bytes() noexcept {{
    return pyamplicol_native_direct_{function_name}::kLogicalStackBytes;
}}

extern "C" int {function_name}_direct_application_v1(
    const pyamplicol_native_direct_{function_name}::
        NativeCompiledDirectInputPlaneV1* inputs,
    std::uint32_t input_count,
    const pyamplicol_native_direct_{function_name}::
        NativeCompiledDirectScalarV1* scalars,
    std::uint32_t scalar_count,
    const pyamplicol_native_direct_{function_name}::
        NativeCompiledDirectOutputPlaneV1* outputs,
    std::uint32_t output_count,
    std::uint32_t point_start,
    std::uint32_t point_count
) noexcept {{
    using namespace pyamplicol_native_direct_{function_name};
    try {{
        if (input_count != kInputPlaneCount ||
            scalar_count != kScalarInputCount ||
            output_count != kOutputPlaneCount) {{
            return 2;
        }}
        if ((input_count != 0u && inputs == nullptr) ||
            (scalar_count != 0u && scalars == nullptr) ||
            outputs == nullptr) {{
            return 1;
        }}
        for (std::uint32_t input = 0; input < input_count; ++input) {{
            if (inputs[input].values == nullptr) {{
                return 3;
            }}
        }}
        for (std::uint32_t scalar = 0; scalar < scalar_count; ++scalar) {{
            if (scalars[scalar].value == nullptr) {{
                return 3;
            }}
        }}
        for (std::uint32_t output = 0; output < output_count; ++output) {{
            if (outputs[output].values == nullptr) {{
                return 3;
            }}
            for (std::uint32_t input = 0; input < input_count; ++input) {{
                if (outputs[output].values == inputs[input].values) {{
                    return 4;
                }}
            }}
            for (std::uint32_t earlier = 0; earlier < output; ++earlier) {{
                if (outputs[output].values == outputs[earlier].values) {{
                    return 4;
                }}
            }}
        }}
        if (point_count == 0u) {{
            return 5;
        }}
        const std::uint64_t point_stop_wide =
            static_cast<std::uint64_t>(point_start) +
            static_cast<std::uint64_t>(point_count);
        if (point_stop_wide >
            static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {{
            return 5;
        }}
        const std::uint32_t point_stop =
            static_cast<std::uint32_t>(point_stop_wide);
        const std::uint32_t vector_points =
            (point_count / kSimdLaneWidth) * kSimdLaneWidth;
        const std::uint32_t vector_stop = point_start + vector_points;
        std::uint32_t point = point_start;
        for (; point < vector_stop; point += kSimdLaneWidth) {{
            evaluate_direct_bundle<DirectVector>(inputs, scalars, outputs, point);
        }}
        for (; point < point_stop; ++point) {{
            evaluate_direct_bundle<double>(inputs, scalars, outputs, point);
        }}
        return 0;
    }} catch (...) {{
        return 6;
    }}
}}
"""


__all__ = [
    "NATIVE_COMPILED_DIRECT_APPLICATION_ABI",
    "NativeDirectCppArtifact",
    "NativeDirectCppCompiler",
    "NativeDirectCppParameterKind",
    "NativeDirectCppSource",
    "NativeDirectCppSpec",
    "compile_native_direct_cpp",
    "compiler_from_symbolica_settings",
    "render_native_direct_cpp",
]
