# SPDX-License-Identifier: 0BSD
"""Runtime adapters for JIT, compiled, and chunked Symbolica evaluators."""

from __future__ import annotations

import hashlib
import importlib
import struct
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .._internal.physics.types import NativeEvaluationError
from .._internal.versions import (
    SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    SYMBOLICA_CPP_RUNTIME_CAPABILITY,
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    SYMJIT_APPLICATION_ABI,
    SYMJIT_F64_RUNTIME_CAPABILITY,
    SYMJIT_PLANE_APPLICATION_ABI,
    verify_native_module,
)
from .execution_schema import aggregate_runtime_capabilities
from .symbolica_helpers import (
    _artifact_path_for_manifest,
    _artifact_path_from_manifest,
    _artifact_subdirectory,
    _complex128_parameter_rows,
    _evaluate_prepared_complex,
    _evaluate_prepared_complex_profiled,
    _safe_symbol_name,
    _symbolica_evaluator_artifact_manifest,
    _symbolica_instruction_program_repr,
)
from .symbolica_settings import (
    ProgressCallback,
    SymbolicaEvaluatorSettings,
    _report_progress,
)


class _JITSymbolicaEvaluatorAdapter:
    def __init__(
        self,
        evaluator: Any,
        settings: SymbolicaEvaluatorSettings,
        label: str,
        *,
        input_len: int,
        output_len: int,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.backend = settings.backend
        self.settings = settings.to_json_dict()
        self.label = _safe_symbol_name(label)
        self._source_evaluator = evaluator
        self._progress_callback = progress_callback
        self.application_path: Path | None = None
        self.evaluator_state_path: Path | None = None
        self.build_timing: dict[str, float] = {}

    def evaluate_complex(self, parameter_rows: Any) -> Any:
        return self._source_evaluator.evaluate_complex(parameter_rows)

    def evaluate_complex_profiled(
        self, parameter_rows: Any
    ) -> tuple[Any, tuple[float, float, float]]:
        return self._evaluate_complex_profiled_prepared(
            _complex128_parameter_rows(parameter_rows)
        )

    def supports_complex_profiled(self) -> bool:
        return callable(
            getattr(self._source_evaluator, "evaluate_complex_profiled", None)
        )

    def evaluate(self, parameter_rows: Any) -> Any:
        return self._source_evaluator.evaluate(parameter_rows)

    def _evaluate_complex_prepared(self, parameter_rows: np.ndarray) -> Any:
        return self._source_evaluator.evaluate_complex(parameter_rows)

    def _evaluate_complex_profiled_prepared(
        self, parameter_rows: np.ndarray
    ) -> tuple[Any, tuple[float, float, float]]:
        profile = getattr(self._source_evaluator, "evaluate_complex_profiled", None)
        if not callable(profile):
            raise NativeEvaluationError(
                "this Symbolica build does not expose native complex profiling"
            )
        output, timing = profile(parameter_rows)
        if not isinstance(timing, tuple) or len(timing) != 3:
            raise NativeEvaluationError(
                "Symbolica returned an invalid native complex profile"
            )
        return output, tuple(float(value) for value in timing)

    def materialize(self) -> None:
        self._ensure_jit_compiled()

    def _ensure_jit_compiled(self) -> None:
        dummy = np.ones((1, self.input_len), dtype=np.complex128)
        self._source_evaluator.evaluate_complex(dummy)

    @classmethod
    def from_artifact(
        cls,
        manifest: dict[str, Any],
        artifact_dir: Path,
    ) -> _JITSymbolicaEvaluatorAdapter:
        from symbolica import Evaluator

        instance = cls.__new__(cls)
        instance.input_len = int(manifest["input_len"])
        instance.output_len = int(manifest["output_len"])
        instance.backend = str(manifest["backend"])
        instance.settings = dict(manifest.get("settings", {}))
        instance.label = str(manifest.get("label", "jit_symbolica_evaluator"))
        application_path = manifest.get("application_path")
        instance.application_path = (
            None
            if application_path is None
            else _artifact_path_from_manifest(str(application_path), artifact_dir)
        )
        instance.evaluator_state_path = _artifact_path_from_manifest(
            str(manifest["evaluator_state_path"]),
            artifact_dir,
        )
        instance._source_evaluator = Evaluator.load(
            instance.evaluator_state_path.read_bytes()
        )
        instance._progress_callback = None
        instance.build_timing = {}
        return instance

    def artifact_manifest(self, artifact_dir: Path) -> dict[str, Any]:
        if bool(self.settings.get("jit_direct_translation")):
            raise NativeEvaluationError(
                "direct SymJIT translation cannot be persisted as a portable "
                "process artifact; use indirect translation"
            )
        _report_progress(
            self._progress_callback,
            stage="jit materialize",
            item=self.label,
        )
        materialize_started = time.perf_counter()
        self._ensure_jit_compiled()
        jit_compile_s = time.perf_counter() - materialize_started
        _report_progress(
            self._progress_callback,
            stage="jit materialized",
            item=f"{self.label} {jit_compile_s:.3f}s",
            increment=1,
        )
        evaluator_dir = _artifact_subdirectory(artifact_dir, "evaluators")
        unique = uuid.uuid4().hex
        application_path = evaluator_dir / f"{self.label}_{unique}.symjit"
        plane_application_path = evaluator_dir / f"{self.label}_{unique}.plane.symjit"
        evaluator_state_path = evaluator_dir / f"{self.label}_{unique}.evaluator.bin"
        save_started = time.perf_counter()
        application, element_layout = self._export_symjit_application()
        application_path.write_bytes(application)
        application_export_s = time.perf_counter() - save_started
        plane_export_started = time.perf_counter()
        (
            plane_application,
            plane_source_digest,
            plane_target,
        ) = self._export_symjit_plane_application(
            optimization_level=self._optimization_level(),
        )
        plane_application_path.write_bytes(plane_application)
        plane_application_export_s = time.perf_counter() - plane_export_started
        state_save_started = time.perf_counter()
        evaluator_state_path.write_bytes(self._source_evaluator.save())
        evaluator_save_s = time.perf_counter() - save_started
        evaluator_state_save_s = time.perf_counter() - state_save_started
        self.application_path = application_path
        self.evaluator_state_path = evaluator_state_path
        build_timing = dict(self.build_timing)
        build_timing["jit_materialize_s"] = jit_compile_s
        build_timing["symjit_application_export_s"] = application_export_s
        build_timing["symjit_plane_application_export_s"] = plane_application_export_s
        build_timing["evaluator_save_s"] = evaluator_state_save_s
        build_timing["artifact_manifest_s"] = jit_compile_s + evaluator_save_s
        optimization_level = self._optimization_level()
        return {
            "kind": "symjit-application-evaluator",
            "runtime_capability": SYMJIT_F64_RUNTIME_CAPABILITY,
            "backend": self.backend,
            "label": self.label,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "application_path": _artifact_path_for_manifest(
                application_path, artifact_dir
            ),
            "application_abi": SYMJIT_APPLICATION_ABI,
            "plane_application": {
                "application_path": _artifact_path_for_manifest(
                    plane_application_path,
                    artifact_dir,
                ),
                "application_abi": SYMJIT_PLANE_APPLICATION_ABI,
                "storage_abi": SYMJIT_APPLICATION_ABI,
                "element_layout": "split-complex-plane-major",
                "descriptor_order": "inputs-re-im-then-outputs-re-im",
                "input_complex_count": self.input_len,
                "output_complex_count": self.output_len,
                "input_plane_count": 2 * self.input_len,
                "output_plane_count": 2 * self.output_len,
                "compiler_type": "native",
                "translation_mode": "symbolica-structured-instructions",
                "optimization_level": optimization_level,
                "simd": True,
                "complex": True,
                "fast_math": True,
                "fast_complex": False,
                "compression": bool(self.settings.get("jit_compress", False)),
                "threading": False,
                "direct_arena": True,
                "source_digest": plane_source_digest,
                "target": plane_target,
            },
            "element_layout": element_layout,
            "batch_layout": "row-major",
            "compiler_type": "native",
            "translation_mode": "indirect",
            "optimization_level": optimization_level,
            "word_bits": struct.calcsize("P") * 8,
            "endianness": sys.byteorder,
            "required_defuns": [],
            "evaluator_state_path": _artifact_path_for_manifest(
                evaluator_state_path, artifact_dir
            ),
            "evaluator_state_runtime_capability": (
                SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY
            ),
            "settings": self.settings,
            "build_timing": build_timing,
        }

    def _optimization_level(self) -> int:
        optimization_level = self.settings.get("jit_optimization_level", 2)
        if (
            isinstance(optimization_level, bool)
            or not isinstance(optimization_level, int)
            or optimization_level not in {0, 1, 2, 3}
        ):
            raise NativeEvaluationError("invalid SymJIT optimization-level metadata")
        return optimization_level

    def _export_symjit_plane_application(
        self,
        *,
        optimization_level: int,
    ) -> tuple[bytes, str, dict[str, object]]:
        """Compile a standard complex P-kernel from Symbolica instructions."""

        instructions = getattr(self._source_evaluator, "get_instructions", None)
        if not callable(instructions):
            raise NativeEvaluationError(
                "this Symbolica build cannot expose structured evaluator "
                "instructions for a direct-arena P-kernel; install the pinned "
                "pyAmpliCol candidate dependency"
            )
        try:
            program_repr = _symbolica_instruction_program_repr(instructions())
        except Exception as error:
            raise NativeEvaluationError(
                "Symbolica could not export structured evaluator instructions "
                "for a direct-arena P-kernel"
            ) from error
        if not program_repr:
            raise NativeEvaluationError(
                "Symbolica returned empty direct-arena P-kernel instructions"
            )

        try:
            rusticol = importlib.import_module("pyamplicol._rusticol")
            verify_native_module(rusticol)
        except Exception as error:
            raise NativeEvaluationError(
                "the Rusticol extension required to compile a direct-arena "
                "P-kernel is unavailable; reinstall pyAmpliCol"
            ) from error
        compile_plane = getattr(
            rusticol,
            "_compile_symjit_plane_application_v2",
            None,
        )
        if not callable(compile_plane):
            raise NativeEvaluationError(
                "the installed Rusticol extension does not expose the SymJIT "
                "2.22 plane compiler; reinstall pyAmpliCol"
            )
        try:
            application = compile_plane(
                program_repr,
                self.input_len,
                self.output_len,
                optimization_level,
                bool(self.settings.get("jit_compress", False)),
            )
        except Exception as error:
            raise NativeEvaluationError(
                "SymJIT could not compile the structured evaluator instructions "
                "as a complex direct-arena P-kernel"
            ) from error
        if not isinstance(application, bytes) or not application:
            raise NativeEvaluationError(
                "Rusticol returned an empty or non-bytes direct-arena P-kernel"
            )

        target_info = getattr(rusticol, "target_info", None)
        target: dict[str, object] = {
            "word_bits": struct.calcsize("P") * 8,
            "endianness": sys.byteorder,
        }
        if callable(target_info):
            info = target_info()
            triple = getattr(info, "triple", None)
            cpu_features = getattr(info, "cpu_features", None)
            if isinstance(triple, str) and triple:
                target["triple"] = triple
            if isinstance(cpu_features, list) and all(
                isinstance(value, str) for value in cpu_features
            ):
                target["cpu_features"] = list(cpu_features)
        return (
            application,
            hashlib.sha256(program_repr.encode("utf-8")).hexdigest(),
            target,
        )

    def _export_symjit_application(self) -> tuple[bytes, str]:
        if struct.calcsize("P") * 8 != 64 or sys.byteorder != "little":
            raise NativeEvaluationError(
                "direct SymJIT f64 artifacts require a 64-bit little-endian host"
            )
        exporter = getattr(self._source_evaluator, "export_symjit", None)
        if not callable(exporter):
            raise NativeEvaluationError(
                "this Symbolica build cannot export a direct SymJIT f64 application; "
                "install the pinned pyAmpliCol candidate dependency"
            )
        try:
            application = exporter(complex=True)
        except Exception as error:
            raise NativeEvaluationError(
                "Symbolica could not export a self-contained SymJIT f64 application; "
                "external evaluator functions are not supported in process artifacts"
            ) from error
        if not isinstance(application, bytes) or not application:
            raise NativeEvaluationError(
                "Symbolica returned an empty or non-bytes SymJIT application"
            )
        expected_layout = _symjit_element_layout()
        return application, expected_layout


class _CompiledComplexEvaluatorAdapter:
    def __init__(
        self,
        evaluator: Any,
        settings: SymbolicaEvaluatorSettings,
        label: str,
        *,
        input_len: int,
        output_len: int,
        compile_dense: bool = True,
    ) -> None:
        safe_label = _safe_symbol_name(label)
        unique = uuid.uuid4().hex
        function_name = f"pyamplicol_{safe_label}_{unique}"
        self.function_name = function_name
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.backend = settings.backend
        self.settings = settings.to_json_dict()
        self.runtime_capability = _compiled_runtime_capability(self.settings)
        self.number_type = "complex"
        self._source_evaluator = evaluator
        self.build_timing: dict[str, float] = {}
        if settings.compiled_output_dir is None:
            self._tmpdir: tempfile.TemporaryDirectory[str] | None = (
                tempfile.TemporaryDirectory(prefix="pyamplicol-symbolica-")
            )
            path = Path(self._tmpdir.name)
        else:
            self._tmpdir = None
            path = Path(settings.compiled_output_dir).expanduser()
            path.mkdir(parents=True, exist_ok=True)
        self.source_path = path / f"{function_name}.cpp"
        self.library_path = path / f"lib{function_name}"
        self.evaluator_state_path: Path | None = path / f"{function_name}.evaluator.bin"
        save = getattr(evaluator, "save", None)
        if callable(save):
            save_started = time.perf_counter()
            self.evaluator_state_path.write_bytes(save())
            self.build_timing["evaluator_save_s"] = time.perf_counter() - save_started
        else:
            self.evaluator_state_path = None
        self._compiled = None
        if compile_dense:
            compile_started = time.perf_counter()
            compile_kwargs: dict[str, object] = {
                "inline_asm": settings.compiled_inline_asm,
                "optimization_level": settings.compiled_optimization_level,
                "native": settings.compiled_native,
                "compiler_path": settings.compiler_path,
                "compiler_flags": _compiled_compiler_flags(settings),
            }
            custom_header = _compiled_complex_custom_header(settings)
            if custom_header is not None:
                compile_kwargs["custom_header"] = custom_header
            self._compiled = evaluator.compile(
                function_name,
                str(self.source_path),
                str(self.library_path),
                self.number_type,
                **compile_kwargs,
            )
            self.build_timing["cxx_compile_s"] = time.perf_counter() - compile_started
        self.native_direct_application = None

    def evaluate_complex(self, parameter_rows: Any) -> Any:
        return self._evaluate_complex_prepared(
            _complex128_parameter_rows(parameter_rows)
        )

    def _evaluate_complex_prepared(self, parameter_rows: np.ndarray) -> Any:
        if self._compiled is not None:
            return self._compiled.evaluate(parameter_rows)
        if self._source_evaluator is None:
            raise NativeEvaluationError(
                "direct-only compiled evaluator has no generation-time source state"
            )
        return self._source_evaluator.evaluate_complex(parameter_rows)

    @classmethod
    def from_artifact(
        cls,
        manifest: dict[str, Any],
        artifact_dir: Path,
    ) -> _CompiledComplexEvaluatorAdapter:
        if manifest.get("native_direct_application") is not None:
            raise NativeEvaluationError(
                "plane-native compiled process evaluators load only through "
                "Rusticol Direct-Arena; no dense callable fallback is retained"
            )
        from symbolica import CompiledComplexEvaluator

        instance = cls.__new__(cls)
        instance._tmpdir = None
        instance.function_name = str(manifest["function_name"])
        instance.input_len = int(manifest["input_len"])
        instance.output_len = int(manifest["output_len"])
        instance.backend = str(manifest["backend"])
        instance.settings = dict(manifest.get("settings", {}))
        instance.runtime_capability = str(
            manifest.get(
                "runtime_capability",
                _compiled_runtime_capability(instance.settings),
            )
        )
        instance.number_type = str(manifest["number_type"])
        instance._source_evaluator = None
        instance.source_path = _artifact_path_from_manifest(
            str(manifest["source_path"]),
            artifact_dir,
        )
        instance.library_path = _artifact_path_from_manifest(
            str(manifest["library_path"]),
            artifact_dir,
        )
        state_path = manifest.get("evaluator_state_path")
        instance.evaluator_state_path = (
            None
            if state_path is None
            else _artifact_path_from_manifest(str(state_path), artifact_dir)
        )
        instance.build_timing = {}
        instance.native_direct_application = None
        if instance.number_type != "complex":
            raise NativeEvaluationError(
                f"unsupported compiled evaluator number type: {instance.number_type!r}"
            )
        instance._compiled = CompiledComplexEvaluator.load(
            str(instance.library_path),
            instance.function_name,
            instance.input_len,
            instance.output_len,
        )
        return instance

    def compile_native_direct_application(
        self,
        parameter_kinds: Sequence[object],
        *,
        target_triple: str,
        cpu_features: Sequence[str] = (),
        simd_lane_width: int = 2,
        output_dir: Path | None = None,
    ) -> Any:
        """Emit this compiled leaf directly against persistent arena planes.

        Stage lowering supplies semantic storage kinds; guessing model scalars
        from evaluator arity is intentionally forbidden.  Both public native
        backends use this independent plane-native instruction emitter.  For an
        ``asm`` selection the historical dense scalar-row library may contain
        Symbolica inline assembly, while the production DirectApplication is
        deliberately regenerated as a plane loop from the same optimized
        instruction stream.  It never wraps or calls that dense entry point.
        """

        from .native_direct_cpp import (
            NativeDirectCppParameterKind,
            NativeDirectCppSpec,
            compile_native_direct_cpp,
            compiler_from_symbolica_settings,
        )

        if self._source_evaluator is None:
            raise NativeEvaluationError(
                "native DirectApplication production requires the retained "
                "Symbolica evaluator state; regenerate this compiled leaf"
            )
        if self.runtime_capability not in {
            SYMBOLICA_CPP_RUNTIME_CAPABILITY,
            SYMBOLICA_ASM_RUNTIME_CAPABILITY,
        }:
            raise NativeEvaluationError(
                "native DirectApplication production requires a C++ or ASM "
                "compiled evaluator"
            )
        optimization = self.settings.get("compiled_optimization_level", 3)
        if optimization != 3:
            raise NativeEvaluationError(
                "native DirectApplication production requires compiled "
                "optimization level 3"
            )
        try:
            kinds = tuple(
                NativeDirectCppParameterKind(kind) for kind in parameter_kinds
            )
        except (TypeError, ValueError) as error:
            raise NativeEvaluationError(
                "native DirectApplication stage parameter kinds are invalid"
            ) from error
        if len(kinds) != self.input_len:
            raise NativeEvaluationError(
                "native DirectApplication stage parameter kinds do not match "
                "the evaluator input width"
            )
        features = tuple(str(feature) for feature in cpu_features)
        destination = self.source_path.parent if output_dir is None else output_dir
        spec = NativeDirectCppSpec(
            function_name=self.function_name,
            parameter_kinds=kinds,
            output_count=self.output_len,
            target_triple=target_triple,
            cpu_features=features,
            simd_lane_width=simd_lane_width,
        )
        compiler = compiler_from_symbolica_settings(self.settings)
        result = compile_native_direct_cpp(
            self._source_evaluator,
            spec=spec,
            compiler=compiler,
            output_source_path=destination / f"{self.function_name}.direct.cpp",
            output_library_path=destination / f"lib{self.function_name}.direct",
        )
        self.native_direct_application = result
        self.build_timing["native_direct_cpp_compile_s"] = result.compile_seconds
        self.build_timing["native_direct_cpp_gate_wait_s"] = (
            result.compiler_gate_wait_seconds
        )
        if result.compiler_gate_slot_count is not None:
            self.build_timing["native_direct_cpp_gate_slots"] = float(
                result.compiler_gate_slot_count
            )
        return result

    # Kept as a narrow source-level alias for existing producer tests.  Artifact
    # production uses ``compile_native_direct_application`` exclusively.
    compile_native_direct_application_prototype = compile_native_direct_application

    def artifact_manifest(self, artifact_dir: Path) -> dict[str, Any]:
        direct = self.native_direct_application
        source_path = self.source_path if direct is None else direct.source_path
        library_path = self.library_path if direct is None else direct.library_path
        manifest = {
            "kind": "compiled-complex-evaluator",
            "runtime_capability": self.runtime_capability,
            "backend": self.backend,
            "number_type": self.number_type,
            "function_name": self.function_name,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "settings": self.settings,
            "source_path": _artifact_path_for_manifest(
                source_path,
                artifact_dir,
            ),
            "library_path": _artifact_path_for_manifest(
                library_path,
                artifact_dir,
            ),
            "evaluator_state_path": (
                None
                if self.evaluator_state_path is None
                else _artifact_path_for_manifest(
                    self.evaluator_state_path,
                    artifact_dir,
                )
            ),
            "build_timing": dict(self.build_timing),
        }
        if direct is not None:
            manifest["native_direct_application"] = _native_direct_application_manifest(
                direct,
                artifact_dir,
                expected_function_name=self.function_name,
            )
        return manifest


def _native_direct_application_manifest(
    artifact: Any,
    artifact_dir: Path,
    *,
    expected_function_name: str,
) -> dict[str, object]:
    """Serialize one direct-only companion without accepting a dense wrapper."""

    from .native_direct_cpp import (
        NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
        NativeDirectCppArtifact,
    )

    if not isinstance(artifact, NativeDirectCppArtifact):
        raise NativeEvaluationError(
            "native DirectApplication producer returned an invalid artifact"
        )
    source = artifact.source
    if (
        not artifact.source_path.is_file()
        or not artifact.library_path.is_file()
        or source.input_plane_count < 1
        or source.output_plane_count < 2
        or source.output_plane_count % 2 != 0
        or source.scalar_input_count < 0
        or source.simd_lane_width not in {2, 4}
        or source.logical_stack_bytes < 1
        or source.logical_stack_bytes > 64 * 1024
        or len(source.evaluator_state_sha256) != 64
    ):
        raise NativeEvaluationError(
            "native DirectApplication artifact metadata is incomplete"
        )
    function_name = expected_function_name
    if _safe_symbol_name(function_name) != function_name:
        raise NativeEvaluationError(
            "native DirectApplication function identity is invalid"
        )
    return {
        "application_abi": NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
        "function_name": function_name,
        "source_path": _artifact_path_for_manifest(
            artifact.source_path,
            artifact_dir,
        ),
        "library_path": _artifact_path_for_manifest(
            artifact.library_path,
            artifact_dir,
        ),
        "target": {
            "triple": source.target_triple,
            "cpu_features": list(source.cpu_features),
        },
        "evaluator_state_sha256": source.evaluator_state_sha256,
        "instruction_count": source.instruction_count,
        "temporary_count": source.temporary_count,
        "input_plane_count": source.input_plane_count,
        "scalar_input_count": source.scalar_input_count,
        "output_plane_count": source.output_plane_count,
        "simd_lane_width": source.simd_lane_width,
        "logical_stack_bytes": source.logical_stack_bytes,
        "output_semantics": "factor-free-overwrite",
    }


def compile_native_direct_applications(
    evaluator: Any,
    parameter_kinds: Sequence[object],
    *,
    target_triple: str,
    cpu_features: Sequence[str] = (),
    simd_lane_width: int = 2,
) -> bool:
    """Compile every native leaf in one evaluator tree as DirectApplication.

    Chunk input maps are composed here, before artifact serialization, so each
    leaf receives exactly the semantic storage classes represented by its
    retained Symbolica instruction stream.
    """

    kinds = tuple(parameter_kinds)
    if isinstance(evaluator, _CompiledComplexEvaluatorAdapter):
        evaluator.compile_native_direct_application(
            kinds,
            target_triple=target_triple,
            cpu_features=cpu_features,
            simd_lane_width=simd_lane_width,
        )
        return True
    if isinstance(evaluator, _ChunkedSymbolicaEvaluator):
        compiled = False
        for child, indices in zip(
            evaluator._evaluators,
            evaluator._chunk_input_indices,
            strict=True,
        ):
            child_kinds = tuple(kinds[index] for index in indices)
            child_compiled = compile_native_direct_applications(
                child,
                child_kinds,
                target_triple=target_triple,
                cpu_features=cpu_features,
                simd_lane_width=simd_lane_width,
            )
            if compiled and not child_compiled:
                raise NativeEvaluationError(
                    "native DirectApplication compilation cannot mix native "
                    "and non-native evaluator chunks"
                )
            compiled = compiled or child_compiled
        return compiled
    return False


def _compiled_compiler_flags(settings: SymbolicaEvaluatorSettings) -> tuple[str, ...]:
    flags = tuple(settings.compiler_flags)
    if settings.backend == "compiled-complex":
        # Symbolica's generated C++ and pyAmpliCol's injected direct-table
        # support use C++17 features.  Apple clang otherwise defaults to C++98.
        return ("-std=c++17", *flags)
    return flags


_CPP_COMPLEX_LITERAL_COMPAT_HEADER = r"""
template<typename Complex, typename Value>
inline Complex pyamplicol_complex_literal(Value value) {
    return Complex(value);
}

template<typename Complex, typename Real, typename Imag>
inline Complex pyamplicol_complex_literal(Real real, Imag imag) {
    return Complex(std::real(real), std::real(imag));
}

#define T(...) pyamplicol_complex_literal<T>(__VA_ARGS__)
""".strip()


def _compiled_complex_custom_header(
    settings: SymbolicaEvaluatorSettings,
) -> str | None:
    """Adapt Symbolica 2.2 nested complex literals in generated C++ sources."""

    if settings.compiled_inline_asm != "none":
        return None
    return _CPP_COMPLEX_LITERAL_COMPAT_HEADER


def _symjit_element_layout() -> str:
    return "complex-f64"


def _compiled_runtime_capability(settings: Mapping[str, object]) -> str:
    if str(settings.get("compiled_inline_asm", "default")) == "none":
        return SYMBOLICA_CPP_RUNTIME_CAPABILITY
    return SYMBOLICA_ASM_RUNTIME_CAPABILITY


class _ChunkedSymbolicaEvaluator:
    def __init__(
        self,
        evaluators: tuple[Any, ...],
        *,
        input_len: int | None = None,
        chunk_input_indices: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        if not evaluators:
            raise NativeEvaluationError("chunked evaluator needs at least one chunk")
        child_input_lengths = tuple(
            _evaluator_input_len(evaluator) for evaluator in evaluators
        )
        if input_len is None:
            if len(set(child_input_lengths)) != 1:
                raise NativeEvaluationError(
                    "legacy chunked evaluators need equal child input lengths"
                )
            input_len = child_input_lengths[0]
        if (
            isinstance(input_len, bool)
            or not isinstance(input_len, int)
            or input_len < 0
        ):
            raise NativeEvaluationError("chunked evaluator input length is invalid")
        if chunk_input_indices is None:
            chunk_input_indices = tuple(
                tuple(range(input_len)) for _evaluator in evaluators
            )
        if len(chunk_input_indices) != len(evaluators):
            raise NativeEvaluationError(
                "chunked evaluator input maps do not match evaluator chunks"
            )
        normalized_indices: list[tuple[int, ...]] = []
        for child_input_len, raw_indices in zip(
            child_input_lengths,
            chunk_input_indices,
            strict=True,
        ):
            indices = tuple(raw_indices)
            if len(indices) != child_input_len:
                raise NativeEvaluationError(
                    "chunked evaluator input map length does not match child "
                    "input length"
                )
            if any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= input_len
                for index in indices
            ):
                raise NativeEvaluationError(
                    "chunked evaluator input map contains an invalid parent index"
                )
            if any(left >= right for left, right in pairwise(indices)):
                raise NativeEvaluationError(
                    "chunked evaluator input maps must be strictly increasing"
                )
            normalized_indices.append(indices)
        self._evaluators = evaluators
        self.input_len = input_len
        self._chunk_input_indices = tuple(normalized_indices)
        self.build_timing: dict[str, float] = {}

    def evaluate_complex(self, parameter_rows: Any) -> Any:
        return np.concatenate(self.evaluate_complex_chunks(parameter_rows), axis=1)

    def evaluate_complex_chunks(self, parameter_rows: Any) -> tuple[Any, ...]:
        prepared_rows = self._prepare_parameter_rows(parameter_rows)
        return tuple(
            _evaluate_prepared_complex(
                evaluator,
                self._chunk_parameter_rows(prepared_rows, indices),
            )
            for evaluator, indices in zip(
                self._evaluators,
                self._chunk_input_indices,
                strict=True,
            )
        )

    def evaluate_complex_profiled(
        self, parameter_rows: Any
    ) -> tuple[tuple[Any, ...], tuple[float, float, float]]:
        prepared_rows = self._prepare_parameter_rows(parameter_rows)
        outputs: list[Any] = []
        profiles: list[tuple[float, float, float]] = []
        gather_s = 0.0
        for evaluator, indices in zip(
            self._evaluators,
            self._chunk_input_indices,
            strict=True,
        ):
            gather_started = time.perf_counter()
            chunk_rows = self._chunk_parameter_rows(prepared_rows, indices)
            gather_s += time.perf_counter() - gather_started
            output, profile = _evaluate_prepared_complex_profiled(evaluator, chunk_rows)
            outputs.append(output)
            profiles.append(profile)
        return tuple(outputs), (
            gather_s + sum(profile[0] for profile in profiles),
            sum(profile[1] for profile in profiles),
            sum(profile[2] for profile in profiles),
        )

    def _prepare_parameter_rows(self, parameter_rows: Any) -> np.ndarray:
        prepared_rows = _complex128_parameter_rows(parameter_rows)
        if prepared_rows.ndim != 2 or prepared_rows.shape[1] != self.input_len:
            raise NativeEvaluationError(
                "chunked evaluator parameter rows have an inconsistent width"
            )
        return prepared_rows

    def _chunk_parameter_rows(
        self,
        parameter_rows: np.ndarray,
        indices: tuple[int, ...],
    ) -> np.ndarray:
        if len(indices) == self.input_len and all(
            index == expected for expected, index in enumerate(indices)
        ):
            return parameter_rows
        return np.ascontiguousarray(parameter_rows[:, indices])

    def supports_complex_profiled(self) -> bool:
        return all(
            bool(getattr(evaluator, "supports_complex_profiled", lambda: False)())
            for evaluator in self._evaluators
        )

    @classmethod
    def from_artifact(
        cls,
        manifest: dict[str, Any],
        artifact_dir: Path,
    ) -> _ChunkedSymbolicaEvaluator:
        chunks = manifest.get("chunks")
        if not isinstance(chunks, list):
            raise NativeEvaluationError("chunked evaluator artifact is missing chunks")
        raw_input_indices = manifest.get("chunk_input_indices")
        chunk_input_indices = None
        if raw_input_indices is not None:
            if not isinstance(raw_input_indices, list):
                raise NativeEvaluationError(
                    "chunked evaluator artifact has invalid input maps"
                )
            parsed_indices: list[tuple[int, ...]] = []
            for indices in raw_input_indices:
                if not isinstance(indices, list):
                    raise NativeEvaluationError(
                        "chunked evaluator artifact has invalid input maps"
                    )
                parsed_indices.append(tuple(indices))
            chunk_input_indices = tuple(parsed_indices)
        raw_input_len = manifest.get("input_len")
        if raw_input_len is not None and (
            isinstance(raw_input_len, bool) or not isinstance(raw_input_len, int)
        ):
            raise NativeEvaluationError(
                "chunked evaluator artifact has an invalid input length"
            )
        return cls(
            tuple(
                _load_symbolica_evaluator_artifact(chunk, artifact_dir)
                for chunk in chunks
            ),
            input_len=raw_input_len,
            chunk_input_indices=chunk_input_indices,
        )

    def artifact_manifest(self, artifact_dir: Path) -> dict[str, Any]:
        worker_counts = []
        for evaluator in self._evaluators:
            settings = getattr(evaluator, "settings", None)
            if isinstance(settings, Mapping):
                workers = settings.get("compiled_chunk_compile_workers")
                if isinstance(workers, int):
                    worker_counts.append(workers)
        workers = min(max(worker_counts, default=1), len(self._evaluators))
        if workers <= 1:
            chunks = [
                _symbolica_evaluator_artifact_manifest(evaluator, artifact_dir)
                for evaluator in self._evaluators
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _symbolica_evaluator_artifact_manifest,
                        evaluator,
                        artifact_dir,
                    )
                    for evaluator in self._evaluators
                ]
                chunks = [future.result() for future in futures]
        timing_totals = dict(self.build_timing)
        timing_totals["chunk_count"] = float(len(chunks))
        for chunk in chunks:
            chunk_timing = (
                chunk.get("build_timing") if isinstance(chunk, dict) else None
            )
            if not isinstance(chunk_timing, Mapping):
                continue
            for key, value in chunk_timing.items():
                if isinstance(value, (float, int)):
                    timing_totals[str(key)] = timing_totals.get(str(key), 0.0) + float(
                        value
                    )
        return {
            "kind": "chunked-symbolica-evaluator",
            "input_len": self.input_len,
            "chunk_input_indices": [
                list(indices) for indices in self._chunk_input_indices
            ],
            "chunks": chunks,
            "required_runtime_capabilities": list(
                aggregate_runtime_capabilities(chunks)
            ),
            "build_timing": timing_totals,
        }


def _evaluator_input_len(evaluator: Any) -> int:
    value = getattr(evaluator, "input_len", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NativeEvaluationError("evaluator chunk has no valid input length")
    return value


def _load_symbolica_evaluator_artifact(
    manifest: Any,
    artifact_dir: Path,
) -> Any:
    if not isinstance(manifest, dict):
        raise NativeEvaluationError("compiled evaluator artifact entry is invalid")
    kind = manifest.get("kind")
    if kind in {"jit-symbolica-evaluator", "symjit-application-evaluator"}:
        return _JITSymbolicaEvaluatorAdapter.from_artifact(manifest, artifact_dir)
    if kind == "compiled-complex-evaluator":
        return _CompiledComplexEvaluatorAdapter.from_artifact(manifest, artifact_dir)
    if kind == "chunked-symbolica-evaluator":
        return _ChunkedSymbolicaEvaluator.from_artifact(manifest, artifact_dir)
    raise NativeEvaluationError(f"unsupported evaluator artifact kind: {kind!r}")
