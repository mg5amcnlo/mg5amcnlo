# SPDX-License-Identifier: 0BSD
"""Compile process-independent exact catalogs into prepared model bundles."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import re
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import cast

from .._internal.physics.symbols import symbols
from .._internal.versions import (
    EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
    EAGER_DIRECT_TABLE_BINDING_ABI,
    EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
    NATIVE_EAGER_DIRECT_TABLE_APPLICATION_ABI,
    SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    SYMBOLICA_CPP_RUNTIME_CAPABILITY,
    SYMBOLICA_SERIALIZATION_ABI,
    SYMJIT_APPLICATION_ABI,
    SYMJIT_PLANE_APPLICATION_ABI,
    package_version,
    verify_native_module,
)
from ..config import EvaluatorConfig
from ..evaluators.native_eager_direct_cpp import (
    render_native_eager_direct_table_cpp,
)
from ..evaluators.symbolica_adapters import _compiled_compiler_flags
from ..evaluators.symbolica_compile import (
    _compile_symbolica_outputs,
    _symbolica_evaluator_kwargs,
)
from ..evaluators.symbolica_helpers import _symbolica_evaluator_artifact_manifest
from ..evaluators.symbolica_settings import SymbolicaEvaluatorSettings
from .base import Model
from .loading import CompiledModel
from .prepared import (
    PREPARED_INDEPENDENT_BLOCK_SIZE,
    PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL,
    PREPARED_KERNEL_VARIANT_ABI,
    PreparedBackend,
    PreparedKernelPack,
    PreparedKernelRecord,
    PreparedKernelVariantRecord,
    PreparedModelBundle,
    PreparedModelBundleError,
    load_prepared_model_bundle,
    prepared_compiled_model_digest,
    prepared_expression_digest,
    prepared_input_contract_digest,
    prepared_kernel_pack_identity,
    prepared_optimization_settings_digest,
    prepared_output_contract_digest,
    prepared_payload_identity_records,
    write_prepared_model_bundle,
)
from .prepared_catalog import (
    PREPARED_INDEPENDENT_BLOCK_PROOF,
    PreparedKernelSpec,
    build_prepared_kernel_catalog,
)
from .prepared_target import (
    PreparedTargetError,
    canonical_architecture,
    native_prepared_target,
    symjit_storage_v3_target,
)
from .recurrence_catalog_builder import build_recurrence_template_catalog
from .recurrence_direct_template import (
    RECURRENCE_DIRECT_BACKEND_ABI,
    PreparedJitDirectSourceV1,
    PreparedNativeDirectCallableSpecV1,
    PreparedNativeDirectSourceV1,
    build_prepared_native_direct_callable_specs,
    build_recurrence_direct_template_catalog,
    prepared_kernel_payload_digest,
)
from .recurrence_template import RecurrenceTemplateCatalog

PreparedModelProgress = Callable[[str, int, int], None]
_PATH_FIELDS = frozenset(
    (
        "application_path",
        "evaluator_state_path",
        "library_path",
        "payload_path",
        "source_path",
    )
)
_PATH_LIST_FIELDS = frozenset(("payload_paths",))
_RECURRENCE_PREFLIGHT_PACK_DIGEST = "0" * 63 + "1"
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True, slots=True)
class PreparedModelBuildResult:
    output: Path
    bundle: PreparedModelBundle
    kernel_count: int
    phase_timings_seconds: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class _IndependentBlockContract:
    parameters: tuple[object, ...]
    outputs: tuple[object, ...]
    input_layout: tuple[str, ...]
    output_layout: tuple[str, ...]


def _validate_native_recurrence_template_input_v1(
    catalog: RecurrenceTemplateCatalog,
    authenticated_kernel_ids: Sequence[int],
) -> Mapping[str, object]:
    """Validate the fixed-width model projection with the installed Rust core."""

    from ..generation.recurrence_template_columnar import (
        RECURRENCE_TEMPLATE_INPUT_ABI,
        RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION,
        build_recurrence_template_input_v1,
    )

    kernel_ids = tuple(authenticated_kernel_ids)
    if kernel_ids != tuple(sorted(set(kernel_ids))) or any(
        type(kernel_id) is not int or kernel_id < 0 for kernel_id in kernel_ids
    ):
        raise PreparedModelBundleError(
            "authenticated prepared-kernel IDs must be sorted, unique, "
            "nonnegative integers"
        )
    template_input = build_recurrence_template_input_v1(catalog)
    try:
        module = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(module)
    except (ImportError, RuntimeError) as exc:
        raise PreparedModelBundleError(
            "recurrence template preparation requires the matching installed "
            "pyamplicol._rusticol extension"
        ) from exc
    candidate = getattr(module, "_validate_recurrence_template_input_v1", None)
    if not callable(candidate):
        raise PreparedModelBundleError(
            "the installed pyamplicol._rusticol extension does not provide "
            "_validate_recurrence_template_input_v1"
        )
    try:
        raw = candidate(
            template_input,
            list(kernel_ids),
        )
    except Exception as exc:
        raise PreparedModelBundleError(
            f"native recurrence template validation failed: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise PreparedModelBundleError(
            "native recurrence template validation returned a non-object result"
        )
    expected = {
        "kind": "pyamplicol-recurrence-template-validation-result",
        "schema_version": 1,
        "validation_status": "validated",
        "template_input_abi": RECURRENCE_TEMPLATE_INPUT_ABI,
        "template_input_schema_version": RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION,
        "template_input_sha256": template_input.canonical_digest,
        "catalog_digest": catalog.catalog_digest,
        "compiled_model_digest": catalog.header.compiled_model_digest,
        "prepared_kernel_pack_digest": (catalog.header.prepared_kernel_pack_digest),
        "prepared_kernel_inventory_verified": True,
        "prepared_kernel_inventory_count": len(kernel_ids),
    }
    for name, expected_value in expected.items():
        actual_value = raw.get(name)
        if type(actual_value) is not type(expected_value) or (
            actual_value != expected_value
        ):
            raise PreparedModelBundleError(
                "native recurrence template validation returned inconsistent "
                f"{name}: expected {expected_value!r}, found {actual_value!r}"
            )
    counts = raw.get("counts")
    if not isinstance(counts, Mapping):
        raise PreparedModelBundleError(
            "native recurrence template validation omitted its count summary"
        )
    expected_counts = {
        "parameters": len(catalog.parameters),
        "current_states": len(catalog.current_states),
        "sources": len(catalog.sources),
        "quantum_flows": len(catalog.quantum_flows),
        "transitions": len(catalog.transitions),
        "propagators": len(catalog.propagators),
        "closures": len(catalog.closures),
        "color_contractions": len(catalog.color_contractions),
        "symmetry_proofs": len(catalog.symmetry_proofs),
        "runtime_helicity_contracts": len(catalog.runtime_helicity_contracts),
        "evaluator_bindings": len(catalog.evaluator_bindings),
        "prepared_kernels": len(
            {
                binding.prepared_kernel_id
                for binding in catalog.evaluator_bindings
                if binding.prepared_kernel_id is not None
            }
        ),
        "referenced_prepared_kernels": len(
            {
                binding.prepared_kernel_id
                for binding in catalog.evaluator_bindings
                if binding.prepared_kernel_id is not None
            }
        ),
    }
    for name, expected_value in expected_counts.items():
        actual_value = counts.get(name)
        if type(actual_value) is not int or actual_value != expected_value:
            raise PreparedModelBundleError(
                "native recurrence template validation returned inconsistent "
                f"{name} count: expected {expected_value}, found {actual_value!r}"
            )
    return raw


def _native_build_inputs_sha256() -> str:
    """Return the authenticated native compiler source identity."""

    try:
        module = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(module)
    except (ImportError, RuntimeError) as exc:
        raise PreparedModelBundleError(
            "prepared-model compilation requires the matching installed "
            "pyamplicol._rusticol extension"
        ) from exc
    operation = getattr(module, "native_build_inputs_sha256", None)
    if not callable(operation):
        raise PreparedModelBundleError(
            "the installed pyamplicol._rusticol extension does not expose "
            "its native build identity"
        )
    digest = operation()
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise PreparedModelBundleError(
            "the installed pyamplicol._rusticol extension has an invalid "
            "native build identity"
        )
    return digest


def _rebind_recurrence_template_pack_digest(
    catalog: RecurrenceTemplateCatalog,
    prepared_kernel_pack_digest: str,
) -> RecurrenceTemplateCatalog:
    """Rebind an already-proven semantic catalog to final evaluator payloads."""

    return RecurrenceTemplateCatalog.create(
        compiled_model_digest=catalog.header.compiled_model_digest,
        prepared_kernel_pack_digest=prepared_kernel_pack_digest,
        parameters=catalog.parameters,
        current_states=catalog.current_states,
        sources=catalog.sources,
        quantum_flows=catalog.quantum_flows,
        contact_orbit_certificates=catalog.contact_orbit_certificates,
        contact_orbit_steps=catalog.contact_orbit_steps,
        transitions=catalog.transitions,
        propagators=catalog.propagators,
        closures=catalog.closures,
        color_contractions=catalog.color_contractions,
        symmetry_proofs=catalog.symmetry_proofs,
        runtime_helicity_contracts=catalog.runtime_helicity_contracts,
        evaluator_bindings=catalog.evaluator_bindings,
    )


def prepare_model_bundle(
    compiled_model: CompiledModel,
    output: Path,
    *,
    evaluator: EvaluatorConfig,
    progress: PreparedModelProgress | None = None,
) -> PreparedModelBuildResult:
    """Build a prepared evaluator-kernel bundle.

    The bundle is shared by eager, recurrence, and on-the-fly execution.
    """

    started = time.perf_counter()
    model = _runtime_model(compiled_model)
    catalog_started = time.perf_counter()
    catalog = build_prepared_kernel_catalog(model)
    catalog_seconds = time.perf_counter() - catalog_started
    compiled_model_payload = compiled_model.to_dict()
    compiled_model_digest = prepared_compiled_model_digest(compiled_model_payload)
    recurrence_catalog_started = time.perf_counter()
    provisional_recurrence_catalog = build_recurrence_template_catalog(
        model,
        catalog,
        compiled_model_digest=compiled_model_digest,
        prepared_kernel_pack_digest=_RECURRENCE_PREFLIGHT_PACK_DIGEST,
    )
    recurrence_catalog_seconds = time.perf_counter() - recurrence_catalog_started
    recurrence_preflight_started = time.perf_counter()
    _validate_native_recurrence_template_input_v1(
        provisional_recurrence_catalog,
        tuple(kernel.kernel_id for kernel in catalog.kernels),
    )
    native_build_inputs_sha256 = _native_build_inputs_sha256()
    recurrence_preflight_seconds = time.perf_counter() - recurrence_preflight_started
    settings = prepared_symbolica_settings(evaluator)
    backend = cast(PreparedBackend, str(evaluator.backend))
    optimization_metadata = _optimization_metadata(settings)
    optimization_digest = prepared_optimization_settings_digest(optimization_metadata)
    payloads: dict[str, bytes | Path] = {}
    records: list[PreparedKernelRecord] = []
    variants: list[PreparedKernelVariantRecord] = []
    native_direct_specs = (
        build_prepared_native_direct_callable_specs(
            provisional_recurrence_catalog,
            catalog.by_id,
        )
        if backend in {"cpp", "asm"}
        else {}
    )
    real_model_parameter_names = frozenset(
        parameter.name
        for parameter in provisional_recurrence_catalog.parameters
        if parameter.value_type == "real"
    )
    compile_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pyamplicol-prepared-model-") as raw:
        staging = Path(raw)
        for index, kernel in enumerate(catalog.kernels):
            if progress is not None:
                progress(
                    f"prepare {kernel.contract_kind} kernel {kernel.kernel_id}",
                    index,
                    len(catalog.kernels),
                )
            record, kernel_variants, kernel_payloads = _compile_kernel(
                kernel,
                settings=settings,
                staging=staging / f"kernel-{kernel.kernel_id:06d}",
                backend=backend,
                optimization_settings_digest=optimization_digest,
                native_direct_spec=native_direct_specs.get(kernel.kernel_id),
                real_model_parameter_names=real_model_parameter_names,
            )
            records.append(record)
            variants.extend(kernel_variants)
            overlap = payloads.keys() & kernel_payloads.keys()
            if overlap:
                raise PreparedModelBundleError(
                    "prepared kernels produced duplicate payload paths: "
                    + ", ".join(sorted(overlap))
                )
            payloads.update(kernel_payloads)

        compile_seconds = time.perf_counter() - compile_started
        base_pack = PreparedKernelPack(
            backend=backend,
            optimization_settings=optimization_metadata,
            producer={
                "distribution": "pyamplicol",
                "version": package_version(),
                "compiled_model_schema": compiled_model.schema_version,
                "model_compiler_version": compiled_model.model_compiler_version,
                "native_build_inputs_sha256": native_build_inputs_sha256,
            },
            dependency_abis={
                "symbolica_serialization": SYMBOLICA_SERIALIZATION_ABI,
                "symjit_application": SYMJIT_APPLICATION_ABI,
                "symjit_plane_application": SYMJIT_PLANE_APPLICATION_ABI,
                "symbolica_version": _distribution_version("symbolica"),
            },
            provenance={
                "model_name": compiled_model.name,
                "model_source": dict(compiled_model.source),
                "compiled_model_digest": str(
                    compiled_model.source.get("digest", "unavailable")
                ),
                "catalog_kernel_count": len(catalog.kernels),
                "unsupported_variant_count": len(catalog.unsupported_variants),
            },
            target=_prepared_target(backend, evaluator),
            resolver_manifest=catalog.resolver_manifest(),
            kernels=tuple(records),
            kernel_variants=tuple(variants),
        )
        pack_identity = prepared_kernel_pack_identity(
            base_pack,
            prepared_payload_identity_records(payloads),
        )
        payload_identity_records = prepared_payload_identity_records(payloads)
        recurrence_binding_started = time.perf_counter()
        records_by_id = {record.kernel_id: record for record in records}
        recurrence_catalog = _rebind_recurrence_template_pack_digest(
            provisional_recurrence_catalog,
            prepared_kernel_pack_digest=pack_identity.pack_digest,
        )
        prepared_jit_sources = (
            {
                record.kernel_id: _prepared_jit_direct_source(
                    record,
                    payload_identity_records=payload_identity_records,
                )
                for record in records
            }
            if backend == "jit"
            else None
        )
        prepared_native_sources = (
            {
                kernel_id: _prepared_native_direct_source(
                    records_by_id[kernel_id],
                    spec=spec,
                    payload_identity_records=payload_identity_records,
                )
                for kernel_id, spec in native_direct_specs.items()
            }
            if backend in {"cpp", "asm"}
            else None
        )
        direct_catalog = build_recurrence_direct_template_catalog(
            recurrence_catalog,
            backend=backend,
            target_triple=str(base_pack.target["target_triple"]),
            portable=bool(base_pack.target["portable"]),
            optimization_level=(
                PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL
                if backend == "jit"
                else int(settings.compiled_optimization_level)
            ),
            prepared_kernel_pack_digest=pack_identity.pack_digest,
            prepared_kernel_contract_digest=pack_identity.contract_digest,
            prepared_kernel_payload_digest=pack_identity.payload_digest,
            optimization_settings_digest=optimization_digest,
            prepared_kernel_payload_digests={
                record.kernel_id: prepared_kernel_payload_digest(
                    kernel_id=record.kernel_id,
                    payload_records=payload_identity_records,
                    referenced_paths=record.referenced_payload_paths,
                )
                for record in records
            },
            prepared_jit_sources=prepared_jit_sources,
            prepared_native_sources=prepared_native_sources,
        )
        recurrence_binding_seconds = time.perf_counter() - recurrence_binding_started
        authenticated_pack = replace(
            base_pack,
            provenance={
                **dict(base_pack.provenance),
                "prepared_kernel_contract_digest": pack_identity.contract_digest,
                "prepared_kernel_payload_digest": pack_identity.payload_digest,
                "prepared_kernel_pack_digest": pack_identity.pack_digest,
                "recurrence_template_abi": recurrence_catalog.header.abi,
                "recurrence_template_digest": recurrence_catalog.catalog_digest,
                "recurrence_direct_backend_abi": RECURRENCE_DIRECT_BACKEND_ABI,
                "recurrence_direct_template_abi": direct_catalog.abi,
                "recurrence_direct_template_digest": direct_catalog.catalog_digest,
                "direct_template_catalog_digest": direct_catalog.catalog_digest,
                "recurrence_direct_payload_status": (
                    "executable"
                    if direct_catalog.executable
                    else "pending-direct-call-abi"
                ),
            },
            recurrence_template=recurrence_catalog.to_dict(),
            recurrence_direct_template=direct_catalog.to_dict(),
        )
        recurrence_validation_started = time.perf_counter()
        recurrence_validation = _validate_native_recurrence_template_input_v1(
            authenticated_pack.recurrence_template_catalog
            or recurrence_catalog,  # constructor invariant, kept explicit for typing
            tuple(kernel.kernel_id for kernel in authenticated_pack.kernels),
        )
        recurrence_validation_seconds = (
            time.perf_counter() - recurrence_validation_started
        )
        pack = replace(
            authenticated_pack,
            provenance={
                **dict(authenticated_pack.provenance),
                "recurrence_template_input_digest": recurrence_validation[
                    "template_input_sha256"
                ],
                "recurrence_template_native_validation_kind": (
                    recurrence_validation["kind"]
                ),
            },
            recurrence_template=recurrence_catalog.to_dict(),
            recurrence_direct_template=direct_catalog.to_dict(),
        )
        bundle_path = write_prepared_model_bundle(
            output,
            compiled_model=compiled_model_payload,
            kernel_pack=pack,
            payloads=payloads,
        )

    bundle = load_prepared_model_bundle(bundle_path)
    timings = {
        "catalog": catalog_seconds,
        "recurrence_catalog": recurrence_catalog_seconds,
        "recurrence_template_preflight": recurrence_preflight_seconds,
        "recurrence_template_binding": recurrence_binding_seconds,
        "recurrence_template_validation": recurrence_validation_seconds,
        "kernel_compilation": compile_seconds,
        "total": time.perf_counter() - started,
    }
    if progress is not None:
        progress("prepared model complete", len(catalog.kernels), len(catalog.kernels))
    return PreparedModelBuildResult(
        output=bundle_path,
        bundle=bundle,
        kernel_count=len(catalog.kernels),
        phase_timings_seconds=timings,
    )


def prepared_symbolica_settings(
    evaluator: EvaluatorConfig,
) -> SymbolicaEvaluatorSettings:
    """Translate the public evaluator configuration into one-kernel settings."""

    optimization = evaluator.optimization
    backend = str(evaluator.backend)
    cores = (
        max(1, os.cpu_count() or 1)
        if optimization.cores == "auto"
        else int(optimization.cores)
    )
    collect_factors = (
        False
        if optimization.collect_factors == "auto"
        else bool(optimization.collect_factors)
    )
    if (
        backend == "jit"
        and evaluator.jit.optimization_level != PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL
    ):
        warnings.warn(
            "prepared JIT model bundles use SymJIT optimization level 2 for "
            "cross-architecture portability; the requested process-evaluator "
            f"level {evaluator.jit.optimization_level} is unchanged for compiled "
            "DAG generation",
            UserWarning,
            stacklevel=2,
        )
    return SymbolicaEvaluatorSettings(
        backend="jit" if backend == "jit" else "compiled-complex",
        iterations=optimization.horner_iterations,
        cpe_iterations=optimization.cpe_iterations,
        n_cores=cores,
        jit_direct_translation=False,
        # Saved SymJIT applications are process-independent model assets.  O2
        # is the only storage-v3 optimization level whose MIR is portable
        # across the supported architecture classes.  Process-local compiled
        # DAG evaluators continue to honor evaluator.jit.optimization_level.
        jit_optimization_level=(
            PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL
            if backend == "jit"
            else evaluator.jit.optimization_level
        ),
        jit_compress=evaluator.resolved_jit_compress,
        max_horner_scheme_variables=optimization.max_horner_variables,
        max_common_pair_cache_entries=optimization.max_common_pair_cache_entries,
        max_common_pair_distance=optimization.max_common_pair_distance,
        collect_factors=collect_factors,
        compiled_inline_asm="default" if backend == "asm" else "none",
        compiled_optimization_level=_cpp_optimization_level(evaluator.cpp.optimization),
        compiled_native=evaluator.cpp.native_arch,
        compiler_path=evaluator.cpp.compiler,
        compiler_flags=evaluator.cpp.extra_flags,
        compiled_output_chunk_size=None,
        output_chunk_strategy="uniform",
        output_chunk_autotune_batch_size=evaluator.batch_size,
        compiled_chunk_compile_workers=1,
        compiled_output_dir=None,
    )


def _runtime_model(compiled: CompiledModel) -> Model:
    from . import BuiltinSMModel, CompiledUFOModel

    if compiled.source.get("kind") == "built-in-sm":
        return BuiltinSMModel()
    return CompiledUFOModel(compiled)


def _prepared_jit_direct_source(
    record: PreparedKernelRecord,
    *,
    payload_identity_records: Mapping[str, tuple[int, str]],
) -> PreparedJitDirectSourceV1:
    manifest = record.f64_evaluator_manifest
    plane_application = manifest.get("plane_application")
    if not isinstance(plane_application, Mapping):
        raise PreparedModelBundleError(
            f"prepared JIT kernel {record.kernel_id} has no plane application"
        )
    application_path = plane_application.get("application_path")
    if not isinstance(application_path, str) or not application_path:
        raise PreparedModelBundleError(
            f"prepared JIT kernel {record.kernel_id} has no plane application payload"
        )
    try:
        _size, application_sha256 = payload_identity_records[application_path]
    except KeyError as exc:
        raise PreparedModelBundleError(
            f"prepared JIT kernel {record.kernel_id} application payload "
            f"{application_path!r} has no identity record"
        ) from exc
    application_abi = plane_application.get("application_abi")
    if application_abi != SYMJIT_PLANE_APPLICATION_ABI:
        raise PreparedModelBundleError(
            f"prepared JIT kernel {record.kernel_id} has incompatible "
            f"plane application ABI {application_abi!r}"
        )
    return PreparedJitDirectSourceV1(
        prepared_kernel_id=record.kernel_id,
        source_application_path=application_path,
        source_application_sha256=application_sha256,
        source_application_abi=application_abi,
        input_contracts=tuple(
            json.dumps(
                dict(contract),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            for contract in record.input_contracts
        ),
        exact_expressions=record.exact_expressions,
        output_arity=record.output_arity,
    )


def _prepared_native_direct_source(
    record: PreparedKernelRecord,
    *,
    spec: PreparedNativeDirectCallableSpecV1,
    payload_identity_records: Mapping[str, tuple[int, str]],
) -> PreparedNativeDirectSourceV1:
    manifest = record.f64_evaluator_manifest
    library_path = manifest.get("library_path")
    if not isinstance(library_path, str) or not library_path:
        raise PreparedModelBundleError(
            f"prepared native kernel {record.kernel_id} has no library payload"
        )
    try:
        _size, library_sha256 = payload_identity_records[library_path]
    except KeyError as exc:
        raise PreparedModelBundleError(
            f"prepared native kernel {record.kernel_id} library payload "
            f"{library_path!r} has no identity record"
        ) from exc
    source_application_abi = manifest.get("runtime_capability")
    if source_application_abi not in {
        SYMBOLICA_CPP_RUNTIME_CAPABILITY,
        SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    }:
        raise PreparedModelBundleError(
            f"prepared native kernel {record.kernel_id} has incompatible runtime "
            f"capability {source_application_abi!r}"
        )
    if spec.prepared_kernel_id != record.kernel_id:
        raise PreparedModelBundleError(
            f"prepared native direct spec for kernel {record.kernel_id} identifies "
            f"kernel {spec.prepared_kernel_id}"
        )
    return PreparedNativeDirectSourceV1(
        prepared_kernel_id=record.kernel_id,
        role=spec.role,
        native_entry_point=spec.native_entry_point,
        source_application_path=library_path,
        source_application_sha256=library_sha256,
        source_application_abi=source_application_abi,
        input_contracts=spec.input_contracts,
        exact_expressions=record.exact_expressions,
        output_arity=record.output_arity,
    )


def _compile_kernel(
    kernel: PreparedKernelSpec,
    *,
    settings: SymbolicaEvaluatorSettings,
    staging: Path,
    backend: PreparedBackend,
    optimization_settings_digest: str,
    native_direct_spec: PreparedNativeDirectCallableSpecV1 | None = None,
    real_model_parameter_names: frozenset[str] = frozenset(),
) -> tuple[
    PreparedKernelRecord,
    tuple[PreparedKernelVariantRecord, ...],
    dict[str, Path],
]:
    from symbolica import Expression

    staging.mkdir(parents=True, exist_ok=True)
    scalar_staging = staging / "scalar"
    scalar_staging.mkdir(parents=True, exist_ok=True)
    from .compiler_records import _replace_evaluator_constants

    outputs = tuple(
        _replace_evaluator_constants(Expression.parse(value))
        for value in kernel.exact_expressions
    )
    parameters = [Expression.parse(item.symbol) for item in kernel.inputs]
    real_parameters = _real_kernel_parameter_indices(
        kernel,
        real_model_parameter_names=real_model_parameter_names,
    )
    if backend == "jit":
        if native_direct_spec is not None:
            raise PreparedModelBundleError(
                f"prepared JIT kernel {kernel.kernel_id} cannot carry a native "
                "Direct-Arena compile specification"
            )
        adapter = _compile_symbolica_outputs(
            outputs,
            parameters,
            merge_evaluators_strategy=False,
            verbose_evaluator_build=False,
            real_params=real_parameters,
            symbolica_settings=replace(settings, compiled_output_chunk_size=None),
            jit_compile=True,
            label=f"prepared_{kernel.contract_kind}_{kernel.kernel_id:06d}",
        )
        raw_manifest = _symbolica_evaluator_artifact_manifest(adapter, scalar_staging)
    else:
        raw_manifest = _compile_native_split_real_kernel(
            kernel,
            outputs=outputs,
            parameters=parameters,
            real_parameters=real_parameters,
            settings=replace(settings, compiled_output_chunk_size=None),
            staging=scalar_staging,
            direct_spec=native_direct_spec,
        )
    manifest, payloads = _relocate_manifest_payloads(
        raw_manifest,
        staging=scalar_staging,
        kernel_id=kernel.kernel_id,
    )
    _validate_backend_manifest(manifest, settings=settings)
    exact_state = manifest.get("evaluator_state_path")
    if not isinstance(exact_state, str) or not exact_state:
        raise PreparedModelBundleError(
            f"prepared kernel {kernel.kernel_id} lacks retained exact evaluator state"
        )
    record = PreparedKernelRecord(
        kernel_id=kernel.kernel_id,
        contract_kind=kernel.contract_kind,
        canonical_signature=kernel.canonical_signature,
        input_arity=kernel.input_arity,
        output_arity=kernel.output_dimension,
        input_layout=tuple(f"{item.role}:{item.component}" for item in kernel.inputs),
        input_contracts=tuple(item.to_dict() for item in kernel.inputs),
        output_layout=kernel.output_layout,
        exact_expressions=kernel.exact_expressions,
        proof_classes=kernel.proof_classes,
        exact_evaluator_state_path=exact_state,
        f64_evaluator_manifest=manifest,
    )
    variants: tuple[PreparedKernelVariantRecord, ...] = ()
    if backend == "jit" and PREPARED_INDEPENDENT_BLOCK_PROOF in kernel.proof_classes:
        variant, variant_payloads = _compile_independent_block_variant(
            kernel,
            settings=settings,
            staging=staging,
            backend=backend,
            optimization_settings_digest=optimization_settings_digest,
        )
        overlap = payloads.keys() & variant_payloads.keys()
        if overlap:
            raise PreparedModelBundleError(
                "prepared scalar and block evaluator payload paths overlap: "
                + ", ".join(sorted(overlap))
            )
        payloads.update(variant_payloads)
        variants = (variant,)
    return record, variants, payloads


def _real_kernel_parameter_indices(
    kernel: PreparedKernelSpec,
    *,
    real_model_parameter_names: frozenset[str],
) -> tuple[int, ...]:
    """Return parameters certified real by their generic prepared contracts."""

    always_real_roles = {
        "left-momentum",
        "right-momentum",
        "momentum",
        "coupling-real",
        "coupling-imag",
    }
    return tuple(
        index
        for index, item in enumerate(kernel.inputs)
        if item.role in always_real_roles
        or (
            item.role == "model-parameter"
            and item.model_parameter_name in real_model_parameter_names
        )
    )


def _native_eager_simd_lane_width(target: Mapping[str, object]) -> int:
    target_triple = str(target["target_triple"])
    raw_features = target.get("cpu_features", ())
    cpu_features = frozenset(str(feature) for feature in raw_features)
    return 4 if target_triple.startswith("x86_64") and "avx2" in cpu_features else 2


def _compile_native_split_real_kernel(
    kernel: PreparedKernelSpec,
    *,
    outputs: tuple[object, ...],
    parameters: list[object],
    real_parameters: tuple[int, ...],
    settings: SymbolicaEvaluatorSettings,
    staging: Path,
    direct_spec: PreparedNativeDirectCallableSpecV1 | None,
) -> dict[str, object]:
    """Compile a native scalar/direct library without Symbolica's complex exporter."""

    from symbolica import Expression

    if settings.backend != "compiled-complex":
        raise PreparedModelBundleError(
            "split-real native preparation requires the compiled backend"
        )
    function_name = f"pyamplicol_prepared_native_k{kernel.kernel_id:08x}"
    raw_function_name = f"{function_name}_split"
    source_path = staging / f"{function_name}.cpp"
    library_path = staging / f"lib{function_name}"
    evaluator_state_path = staging / f"{function_name}.evaluator.bin"
    evaluator_kwargs = _symbolica_evaluator_kwargs(
        settings,
        verbose=False,
        jit_compile=False,
    )

    exact_evaluator = Expression.evaluator_multiple(
        outputs,
        parameters,
        **evaluator_kwargs,
    )
    if real_parameters:
        exact_evaluator.set_real_params(
            list(real_parameters),
            sqrt_real=settings.real_param_sqrt_real,
            log_real=settings.real_param_log_real,
            powf_real=settings.real_param_powf_real,
            real_if_args_real=settings.real_param_real_if_args_real,
            verbose=False,
        )
    save = getattr(exact_evaluator, "save", None)
    if not callable(save):
        raise PreparedModelBundleError(
            f"prepared native kernel {kernel.kernel_id} cannot retain exact state"
        )
    evaluator_state_bytes = save()
    if not isinstance(evaluator_state_bytes, bytes) or not evaluator_state_bytes:
        raise PreparedModelBundleError(
            f"prepared native kernel {kernel.kernel_id} returned invalid exact state"
        )
    evaluator_state_path.write_bytes(evaluator_state_bytes)
    try:
        native_target = native_prepared_target(
            include_cpu_features=settings.compiled_native
        )
    except PreparedTargetError:
        # Low-level compiler tests intentionally exercise the native producer
        # without a built Rust extension. Full prepared-pack publication still
        # calls `_prepared_target` and therefore fails closed without exact
        # Rusticol target introspection.
        native_target = {
            "target_triple": (
                f"{canonical_architecture()}-{sys.platform}-compile-only"
            ),
            "cpu_features": [],
        }
    target_triple = str(native_target["target_triple"])
    simd_lane_width = _native_eager_simd_lane_width(native_target)
    eager_direct = render_native_eager_direct_table_cpp(
        exact_evaluator,
        kernel_id=kernel.kernel_id,
        input_complex_count=kernel.input_arity,
        output_complex_count=kernel.output_dimension,
        target_triple=target_triple,
        simd_lane_width=simd_lane_width,
        real_parameter_indices=real_parameters,
        evaluator_state_bytes=evaluator_state_bytes,
    )

    split_parameters, split_outputs = _split_complex_kernel_contract(
        kernel.kernel_id,
        outputs=outputs,
        parameters=parameters,
        real_parameters=real_parameters,
    )
    split_evaluator = Expression.evaluator_multiple(
        split_outputs,
        list(split_parameters),
        **evaluator_kwargs,
    )
    split_evaluator.set_real_params(
        list(range(len(split_parameters))),
        sqrt_real=settings.real_param_sqrt_real,
        log_real=settings.real_param_log_real,
        powf_real=settings.real_param_powf_real,
        real_if_args_real=settings.real_param_real_if_args_real,
        verbose=False,
    )
    custom_header = _native_split_real_custom_header(
        kernel,
        raw_function_name=raw_function_name,
        direct_spec=direct_spec,
        eager_direct_source=eager_direct.source,
        raw_parameters_const=settings.compiled_inline_asm != "none",
    )
    compile_started = time.perf_counter()
    split_evaluator.compile(
        raw_function_name,
        str(source_path),
        str(library_path),
        "real",
        inline_asm=settings.compiled_inline_asm,
        optimization_level=settings.compiled_optimization_level,
        native=settings.compiled_native,
        compiler_path=settings.compiler_path,
        compiler_flags=_compiled_compiler_flags(settings),
        custom_header=custom_header,
    )
    compile_seconds = time.perf_counter() - compile_started
    runtime_capability = (
        SYMBOLICA_CPP_RUNTIME_CAPABILITY
        if settings.compiled_inline_asm == "none"
        else SYMBOLICA_ASM_RUNTIME_CAPABILITY
    )
    return {
        "kind": "compiled-complex-evaluator",
        "runtime_capability": runtime_capability,
        "backend": settings.backend,
        "number_type": "complex",
        # The sole eager f64 entry is the DirectTable callable. The split-real
        # symbol remains private implementation support for the separate
        # recurrence direct callable when that lane is present; no legacy
        # dense complex entry is published.
        "function_name": eager_direct.function_name,
        "input_len": kernel.input_arity,
        "output_len": kernel.output_dimension,
        "settings": settings.to_json_dict(),
        "source_path": str(source_path),
        "library_path": str(library_path),
        "evaluator_state_path": str(evaluator_state_path),
        "build_timing": {
            "cxx_compile_s": compile_seconds,
        },
        "direct_table": {
            "capability": EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
            "source_application_abi": (NATIVE_EAGER_DIRECT_TABLE_APPLICATION_ABI),
            "descriptor_abi": EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
            "binding_abi": EAGER_DIRECT_TABLE_BINDING_ABI,
            "library_path": str(library_path),
            "function_name": eager_direct.function_name,
            "evaluator_state_sha256": (eager_direct.evaluator_state_sha256),
            "input_complex_count": eager_direct.input_complex_count,
            "output_complex_count": eager_direct.output_complex_count,
            "invocation_stride": eager_direct.invocation_stride,
            "attachment_stride": eager_direct.attachment_stride,
            "simd_lane_width": eager_direct.simd_lane_width,
            "instruction_count": eager_direct.instruction_count,
            "temporary_count": eager_direct.temporary_count,
        },
    }


def _split_complex_kernel_contract(
    kernel_id: int,
    *,
    outputs: tuple[object, ...],
    parameters: list[object],
    real_parameters: tuple[int, ...],
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Lower the exact algebraic kernel into structurally real expressions."""

    from symbolica import AtomType, Expression

    split_parameters: list[object] = []
    parameter_pairs: dict[str, tuple[object, object]] = {}
    real_parameter_set = set(real_parameters)
    if any(index < 0 or index >= len(parameters) for index in real_parameter_set):
        raise PreparedModelBundleError(
            f"prepared native kernel {kernel_id} has an invalid real parameter index"
        )
    zero = Expression.num(0)
    for index, parameter in enumerate(parameters):
        tree = parameter.to_atom_tree()
        if tree.atom_type != AtomType.Var or not isinstance(tree.head, str):
            raise PreparedModelBundleError(
                f"prepared native kernel {kernel_id} parameter {index} is not "
                "a scalar symbol"
            )
        real = symbols.real_symbol(
            f"prepared_native::kernel_{kernel_id:08x}::input_{index}::re"
        )
        imag = symbols.real_symbol(
            f"prepared_native::kernel_{kernel_id:08x}::input_{index}::im"
        )
        split_parameters.extend((real, imag))
        parameter_pairs[tree.head] = (
            real,
            zero if index in real_parameter_set else imag,
        )

    split_outputs: list[object] = []
    admitted_symbols = {
        tree.head
        for symbol in split_parameters
        for tree in (symbol.to_atom_tree(),)
        if tree.atom_type == AtomType.Var and isinstance(tree.head, str)
    }
    for output_index, output in enumerate(outputs):
        real, imag = _split_complex_atom_tree(
            output.to_atom_tree(),
            kernel_id=kernel_id,
            output_index=output_index,
            parameter_pairs=parameter_pairs,
        )
        if not _is_structurally_real_expression(
            real,
            admitted_symbols=admitted_symbols,
        ) or not _is_structurally_real_expression(
            imag,
            admitted_symbols=admitted_symbols,
        ):
            raise PreparedModelBundleError(
                f"prepared native kernel {kernel_id} output {output_index} "
                "cannot be certified as split-real"
            )
        used_symbols = {
            tree.head
            for expression in (real, imag)
            for symbol in expression.get_all_symbols(False)
            if not symbol.is_constant()
            for tree in (symbol.to_atom_tree(),)
            if tree.atom_type == AtomType.Var and isinstance(tree.head, str)
        }
        if not used_symbols.issubset(admitted_symbols):
            raise PreparedModelBundleError(
                f"prepared native kernel {kernel_id} output {output_index} "
                "contains an unbound split-real symbol"
            )
        split_outputs.extend((real, imag))
    return tuple(split_parameters), tuple(split_outputs)


def _split_complex_atom_tree(
    tree: object,
    *,
    kernel_id: int,
    output_index: int,
    parameter_pairs: Mapping[str, tuple[object, object]],
) -> tuple[object, object]:
    """Recursively evaluate one algebraic expression over real/imaginary pairs."""

    from symbolica import AtomType, Expression

    atom_type = tree.atom_type
    if atom_type == AtomType.Var:
        if not isinstance(tree.head, str) or tree.head not in parameter_pairs:
            raise PreparedModelBundleError(
                f"prepared native kernel {kernel_id} output {output_index} "
                f"uses an unbound symbol {tree.head!r}"
            )
        return parameter_pairs[tree.head]
    if atom_type == AtomType.Num:
        if not isinstance(tree.head, str):
            raise PreparedModelBundleError(
                f"prepared native kernel {kernel_id} output {output_index} "
                "has a malformed numeric coefficient"
            )
        number = Expression.parse(_ANSI_ESCAPE_RE.sub("", tree.head))
        imaginary_unit = Expression.parse("sqrt(-1)")
        return (
            ((number + number.conj()) / 2).expand(),
            ((number - number.conj()) / (2 * imaginary_unit)).expand(),
        )
    if atom_type == AtomType.Add:
        result = (Expression.num(0), Expression.num(0))
        for child in tree.tail:
            value = _split_complex_atom_tree(
                child,
                kernel_id=kernel_id,
                output_index=output_index,
                parameter_pairs=parameter_pairs,
            )
            result = (result[0] + value[0], result[1] + value[1])
        return result
    if atom_type == AtomType.Mul:
        result = (Expression.num(1), Expression.num(0))
        for child in tree.tail:
            value = _split_complex_atom_tree(
                child,
                kernel_id=kernel_id,
                output_index=output_index,
                parameter_pairs=parameter_pairs,
            )
            result = _multiply_complex_pairs(result, value)
        return result
    if atom_type == AtomType.Pow:
        if (
            len(tree.tail) != 2
            or tree.tail[1].atom_type != AtomType.Num
            or not isinstance(tree.tail[1].head, str)
        ):
            raise PreparedModelBundleError(
                f"prepared native kernel {kernel_id} output {output_index} "
                "uses a non-numeric exponent"
            )
        exponent = Expression.parse(_ANSI_ESCAPE_RE.sub("", tree.tail[1].head))
        try:
            exact_exponent = Fraction(exponent.to_canonical_string())
        except ValueError as exc:
            raise PreparedModelBundleError(
                f"prepared native kernel {kernel_id} output {output_index} "
                f"uses unsupported complex power {exponent}"
            ) from exc
        base = _split_complex_atom_tree(
            tree.tail[0],
            kernel_id=kernel_id,
            output_index=output_index,
            parameter_pairs=parameter_pairs,
        )
        if exact_exponent.denominator != 1:
            if base[1].to_canonical_string() != "0":
                raise PreparedModelBundleError(
                    f"prepared native kernel {kernel_id} output {output_index} "
                    f"uses non-integer power {exponent} of a complex expression"
                )
            if exact_exponent.denominator != 2:
                raise PreparedModelBundleError(
                    f"prepared native kernel {kernel_id} output {output_index} "
                    f"uses unsupported real non-integer power {exponent}"
                )
            return base[0] ** exponent, Expression.num(0)
        return _integer_power_complex_pair(base, exact_exponent.numerator)
    raise PreparedModelBundleError(
        f"prepared native kernel {kernel_id} output {output_index} uses "
        f"unsupported expression atom {atom_type}"
    )


def _is_structurally_real_expression(
    expression: object,
    *,
    admitted_symbols: set[str],
) -> bool:
    """Certify the real-branch grammar emitted by split-complex lowering."""

    from symbolica import AtomType, Expression

    def visit(tree: object) -> bool:
        if tree.atom_type == AtomType.Var:
            return isinstance(tree.head, str) and tree.head in admitted_symbols
        if tree.atom_type == AtomType.Num:
            return (
                isinstance(tree.head, str)
                and Expression.parse(_ANSI_ESCAPE_RE.sub("", tree.head)).is_real()
            )
        if tree.atom_type == AtomType.Add or tree.atom_type == AtomType.Mul:
            return all(visit(child) for child in tree.tail)
        if tree.atom_type == AtomType.Pow:
            if len(tree.tail) != 2 or not visit(tree.tail[0]):
                return False
            exponent = tree.tail[1]
            return (
                exponent.atom_type == AtomType.Num
                and isinstance(exponent.head, str)
                and Expression.parse(_ANSI_ESCAPE_RE.sub("", exponent.head)).is_real()
            )
        return False

    return visit(expression.to_atom_tree())


def _multiply_complex_pairs(
    left: tuple[object, object],
    right: tuple[object, object],
) -> tuple[object, object]:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _integer_power_complex_pair(
    base: tuple[object, object],
    exponent: int,
) -> tuple[object, object]:
    from symbolica import Expression

    if exponent == 0:
        return Expression.num(1), Expression.num(0)
    if exponent < 0:
        positive = _integer_power_complex_pair(base, -exponent)
        norm = positive[0] * positive[0] + positive[1] * positive[1]
        return positive[0] / norm, -positive[1] / norm
    result = (Expression.num(1), Expression.num(0))
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _multiply_complex_pairs(result, factor)
        remaining >>= 1
        if remaining:
            factor = _multiply_complex_pairs(factor, factor)
    return result


def _native_split_real_custom_header(
    kernel: PreparedKernelSpec,
    *,
    raw_function_name: str,
    direct_spec: PreparedNativeDirectCallableSpecV1 | None,
    eager_direct_source: str = "",
    raw_parameters_const: bool = False,
) -> str:
    lines = [
        "#include <cstddef>",
        "#include <cstdint>",
    ]
    if direct_spec is not None:
        lines.extend(
            (
                "",
                (
                    'extern "C" unsigned long '
                    f"{raw_function_name}_realf64_get_buffer_len();"
                ),
                (
                    f'extern "C" void {raw_function_name}_realf64('
                    f"{'const ' if raw_parameters_const else ''}"
                    "double*, double*, double*);"
                ),
                "",
                _native_direct_arena_declarations(),
                "",
                _native_direct_export(
                    kernel,
                    raw_function_name=raw_function_name,
                    spec=direct_spec,
                ),
            )
        )
    if eager_direct_source:
        lines.extend(("", eager_direct_source))
    return "\n".join(lines)


def _native_direct_arena_declarations() -> str:
    return r"""
extern "C" {
struct DirectArenaView {
  double* current_re;
  double* current_im;
  std::uint64_t current_scalar_len;
  double* amplitude_re;
  double* amplitude_im;
  std::uint64_t amplitude_scalar_len;
  std::uint32_t point_stride;
};
struct DirectMomentumView {
  const double* values;
  std::uint64_t scalar_len;
  std::uint32_t form_count;
  std::uint16_t lorentz_component_count;
  std::uint32_t point_stride;
};
struct DirectParameterView {
  const double* values_re;
  const double* values_im;
  std::uint32_t value_count;
};
struct DirectFactorView {
  const double* values_re;
  const double* values_im;
  std::uint32_t value_count;
};
struct DirectNativeBindingContextV1 {
  double coupling_re;
  double coupling_im;
};
struct DirectContributionRow {
  std::uint32_t parent0_component_base;
  std::uint32_t parent1_component_base_or_sentinel;
  std::uint32_t parent0_momentum_form_id;
  std::uint32_t parent1_momentum_form_id_or_sentinel;
  std::uint32_t destination_component_base;
  std::uint32_t exact_factor_id;
  std::uint32_t selector_domain_id;
  std::uint32_t flags;
};
struct DirectFinalizationRow {
  std::uint32_t component_base;
  std::uint16_t component_count;
  std::uint32_t momentum_form_id;
  std::uint32_t exact_factor_id;
  std::uint32_t selector_domain_id;
  std::uint32_t flags;
};
struct DirectClosureRow {
  std::uint32_t parent0_component_base;
  std::uint32_t parent1_component_base_or_sentinel;
  std::uint32_t parent0_momentum_form_id;
  std::uint32_t parent1_momentum_form_id_or_sentinel;
  std::uint32_t amplitude_destination_id;
  std::uint32_t exact_factor_id;
  std::uint32_t component_factor_start;
  std::uint16_t component_count;
  std::uint32_t selector_domain_id;
  std::uint32_t flags;
};
}

static_assert(sizeof(void*) == 8u, "Direct-Arena requires a 64-bit target");
static_assert(sizeof(DirectArenaView) == 56u);
static_assert(offsetof(DirectArenaView, current_scalar_len) == 16u);
static_assert(offsetof(DirectArenaView, amplitude_re) == 24u);
static_assert(offsetof(DirectArenaView, point_stride) == 48u);
static_assert(sizeof(DirectMomentumView) == 32u);
static_assert(offsetof(DirectMomentumView, form_count) == 16u);
static_assert(offsetof(DirectMomentumView, point_stride) == 24u);
static_assert(sizeof(DirectParameterView) == 24u);
static_assert(offsetof(DirectParameterView, value_count) == 16u);
static_assert(sizeof(DirectFactorView) == 24u);
static_assert(sizeof(DirectNativeBindingContextV1) == 16u);
static_assert(sizeof(DirectContributionRow) == 32u);
static_assert(sizeof(DirectFinalizationRow) == 24u);
static_assert(offsetof(DirectFinalizationRow, momentum_form_id) == 8u);
static_assert(sizeof(DirectClosureRow) == 40u);
static_assert(offsetof(DirectClosureRow, component_count) == 28u);
static_assert(offsetof(DirectClosureRow, selector_domain_id) == 32u);

static constexpr std::uint32_t PAC_DIRECT_NONE = 0xffffffffu;
static constexpr std::uint32_t PAC_DIRECT_INITIALIZE = 1u;
static constexpr unsigned long PAC_DIRECT_MAX_SCRATCH_DOUBLES = 65536ul;

static inline bool pac_direct_plane_index(
    std::uint32_t base,
    std::uint32_t component,
    std::uint32_t point,
    std::uint32_t stride,
    std::uint64_t scalar_len,
    std::uint64_t* index) {
  if (stride == 0u || point >= stride) return false;
  const std::uint64_t plane =
      static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(component);
  if (plane > (UINT64_MAX - point) / stride) return false;
  *index = plane * stride + point;
  return *index < scalar_len;
}

static inline bool pac_direct_load_current(
    DirectArenaView arena,
    std::uint32_t base,
    std::uint32_t component,
    std::uint32_t point,
    double* re,
    double* im) {
  std::uint64_t index = 0;
  if (arena.current_re == nullptr || arena.current_im == nullptr ||
      !pac_direct_plane_index(
          base, component, point, arena.point_stride,
          arena.current_scalar_len, &index)) return false;
  *re = arena.current_re[index];
  *im = arena.current_im[index];
  return true;
}

static inline bool pac_direct_load_momentum(
    DirectMomentumView momenta,
    std::uint32_t form,
    std::uint32_t component,
    std::uint32_t point,
    double* value) {
  if (momenta.values == nullptr || momenta.point_stride == 0u ||
      point >= momenta.point_stride || form >= momenta.form_count ||
      component >= momenta.lorentz_component_count) return false;
  const std::uint64_t plane =
      static_cast<std::uint64_t>(form) * momenta.lorentz_component_count +
      component;
  if (plane > (UINT64_MAX - point) / momenta.point_stride) return false;
  const std::uint64_t index = plane * momenta.point_stride + point;
  if (index >= momenta.scalar_len) return false;
  *value = momenta.values[index];
  return true;
}

static inline bool pac_direct_load_parameter(
    DirectParameterView parameters,
    std::uint32_t index,
    double* re,
    double* im) {
  if (index >= parameters.value_count || parameters.values_re == nullptr ||
      parameters.values_im == nullptr) return false;
  *re = parameters.values_re[index];
  *im = parameters.values_im[index];
  return true;
}

static inline bool pac_direct_load_factor(
    DirectFactorView factors,
    std::uint32_t index,
    double* re,
    double* im) {
  if (index >= factors.value_count || factors.values_re == nullptr ||
      factors.values_im == nullptr) return false;
  *re = factors.values_re[index];
  *im = factors.values_im[index];
  return true;
}

static inline bool pac_direct_store_current(
    DirectArenaView arena,
    std::uint32_t base,
    std::uint32_t component,
    std::uint32_t point,
    double re,
    double im,
    bool add) {
  std::uint64_t index = 0;
  if (arena.current_re == nullptr || arena.current_im == nullptr ||
      !pac_direct_plane_index(
          base, component, point, arena.point_stride,
          arena.current_scalar_len, &index)) return false;
  if (add) {
    arena.current_re[index] += re;
    arena.current_im[index] += im;
  } else {
    arena.current_re[index] = re;
    arena.current_im[index] = im;
  }
  return true;
}

static inline bool pac_direct_store_amplitude(
    DirectArenaView arena,
    std::uint32_t base,
    std::uint32_t component,
    std::uint32_t point,
    double re,
    double im) {
  std::uint64_t index = 0;
  if (arena.amplitude_re == nullptr || arena.amplitude_im == nullptr ||
      !pac_direct_plane_index(
          base, component, point, arena.point_stride,
          arena.amplitude_scalar_len, &index)) return false;
  arena.amplitude_re[index] += re;
  arena.amplitude_im[index] += im;
  return true;
}
""".strip()


def _native_direct_export(
    kernel: PreparedKernelSpec,
    *,
    raw_function_name: str,
    spec: PreparedNativeDirectCallableSpecV1,
) -> str:
    if spec.prepared_kernel_id != kernel.kernel_id:
        raise PreparedModelBundleError(
            "native direct export kernel ID does not match its prepared kernel"
        )
    if spec.exact_expressions != kernel.exact_expressions:
        raise PreparedModelBundleError(
            f"native direct kernel {kernel.kernel_id} expressions do not match "
            "the recurrence callable specification"
        )
    if spec.output_arity > min(spec.destination_component_counts):
        raise PreparedModelBundleError(
            f"native direct kernel {kernel.kernel_id} output exceeds destination "
            "component count"
        )
    if spec.role == "closure" and spec.output_arity != 1:
        raise PreparedModelBundleError(
            "native Direct-Arena closure callables must produce one amplitude"
        )
    contracts = tuple(json.loads(item) for item in spec.input_contracts)
    if tuple(item.to_dict() for item in kernel.inputs) != contracts:
        raise PreparedModelBundleError(
            f"native direct kernel {kernel.kernel_id} input contracts do not match"
        )
    row_type = {
        "contribution": "DirectContributionRow",
        "finalization": "DirectFinalizationRow",
        "closure": "DirectClosureRow",
    }[spec.role]
    uses_binding_coupling = any(
        contract.get("role") in {"coupling-real", "coupling-imag"}
        for contract in contracts
    )
    lines = [
        f'extern "C" int {spec.native_entry_point}(',
        "    const void* context,",
        "    DirectArenaView arena,",
        "    DirectMomentumView momenta,",
        "    DirectParameterView parameters,",
        "    DirectFactorView factors,",
        f"    const {row_type}* rows,",
        "    std::uint32_t row_count,",
        "    std::uint32_t point_count) {",
        (
            "  if (context == nullptr) return 1;"
            if uses_binding_coupling
            else "  (void)context;"
        ),
        *(
            (
                "  const auto* binding_context = static_cast<const "
                "DirectNativeBindingContextV1*>(context);",
            )
            if uses_binding_coupling
            else ()
        ),
        "  if (rows == nullptr || row_count == 0u || point_count == 0u ||",
        "      arena.point_stride == 0u || point_count > arena.point_stride ||",
        "      arena.point_stride != momenta.point_stride) return 1;",
        (
            "  const unsigned long raw_buffer_len = "
            f"{raw_function_name}_realf64_get_buffer_len();"
        ),
        ("  if (raw_buffer_len > PAC_DIRECT_MAX_SCRATCH_DOUBLES) return 3;"),
        (
            "  double* storage = static_cast<double*>("
            "__builtin_alloca(sizeof(double) * "
            f"({2 * kernel.input_arity}u + {2 * kernel.output_dimension}u + "
            "raw_buffer_len)));"
        ),
        "  if (storage == nullptr) return 3;",
        "  double* split_params = storage;",
        f"  double* split_out = split_params + {2 * kernel.input_arity}u;",
        f"  double* split_buffer = split_out + {2 * kernel.output_dimension}u;",
        "  for (std::uint32_t row_index = 0; row_index < row_count; ++row_index) {",
        "    const auto& row = rows[row_index];",
    ]
    if spec.role == "contribution":
        lines.append("    if ((row.flags & ~PAC_DIRECT_INITIALIZE) != 0u) return 2;")
    else:
        lines.append("    if (row.flags != 0u) return 2;")
    if spec.role == "finalization":
        admitted_counts = " || ".join(
            f"row.component_count == {count}u"
            for count in spec.destination_component_counts
        )
        lines.append(f"    if (!({admitted_counts})) return 2;")
    lines.extend(
        (
            "    double factor_re = 0.0;",
            "    double factor_im = 0.0;",
            "    if (!pac_direct_load_factor(",
            "            factors, row.exact_factor_id, &factor_re, "
            "&factor_im)) return 2;",
            "    for (std::uint32_t point = 0; point < point_count; ++point) {",
        )
    )
    lines.extend(_native_direct_input_loads(spec, contracts))
    lines.append(
        f"      {raw_function_name}_realf64(split_params, split_buffer, split_out);"
    )
    for component in range(kernel.output_dimension):
        lines.extend(
            (
                f"      const double value_{component}_re = "
                f"split_out[{2 * component}u] * factor_re - "
                f"split_out[{2 * component + 1}u] * factor_im;",
                f"      const double value_{component}_im = "
                f"split_out[{2 * component}u] * factor_im + "
                f"split_out[{2 * component + 1}u] * factor_re;",
            )
        )
        if spec.role == "closure":
            lines.extend(
                (
                    "      if (!pac_direct_store_amplitude(",
                    "              arena, row.amplitude_destination_id, "
                    f"{component}u, point,",
                    f"              value_{component}_re, "
                    f"value_{component}_im)) return 2;",
                )
            )
        else:
            base = (
                "row.destination_component_base"
                if spec.role == "contribution"
                else "row.component_base"
            )
            add = (
                "(row.flags & PAC_DIRECT_INITIALIZE) == 0u"
                if spec.role == "contribution"
                else "false"
            )
            lines.extend(
                (
                    "      if (!pac_direct_store_current(",
                    f"              arena, {base}, {component}u, point,",
                    f"              value_{component}_re, "
                    f"value_{component}_im, {add})) return 2;",
                )
            )
    lines.extend(("    }", "  }", "  return 0;", "}"))
    return "\n".join(lines)


def _native_direct_input_loads(
    spec: PreparedNativeDirectCallableSpecV1,
    contracts: Sequence[Mapping[str, object]],
) -> list[str]:
    lines: list[str] = []
    for index, contract in enumerate(contracts):
        role = contract.get("role")
        component = contract.get("component")
        if type(component) is not int or component < 0:
            raise PreparedModelBundleError(
                "native direct input component must be a nonnegative integer"
            )
        real_slot = 2 * index
        imag_slot = real_slot + 1
        if role in {"left-current", "right-current", "current"}:
            parent = 1 if role == "right-current" else 0
            if parent >= len(spec.parent_component_shapes[0]):
                raise PreparedModelBundleError(
                    f"native direct {role} input has no recurrence parent"
                )
            if any(
                component >= shape[parent] for shape in spec.parent_component_shapes
            ):
                raise PreparedModelBundleError(
                    f"native direct {role} component exceeds an admitted parent state"
                )
            if spec.role == "finalization":
                base = "row.component_base"
            elif parent == 0:
                base = "row.parent0_component_base"
            else:
                base = "row.parent1_component_base_or_sentinel"
                lines.append(
                    "      if (row.parent1_component_base_or_sentinel == "
                    "PAC_DIRECT_NONE) return 2;"
                )
            lines.extend(
                (
                    "      if (!pac_direct_load_current(",
                    f"              arena, {base}, {component}u, point,",
                    f"              &split_params[{real_slot}u], "
                    f"&split_params[{imag_slot}u])) return 2;",
                )
            )
        elif role in {"left-momentum", "right-momentum", "momentum"}:
            operand = 1 if role == "right-momentum" else 0
            if spec.role == "finalization":
                form = "row.momentum_form_id"
            elif operand == 0:
                form = "row.parent0_momentum_form_id"
            else:
                form = "row.parent1_momentum_form_id_or_sentinel"
                lines.append(
                    "      if (row.parent1_momentum_form_id_or_sentinel == "
                    "PAC_DIRECT_NONE) return 2;"
                )
            lines.extend(
                (
                    "      if (!pac_direct_load_momentum(",
                    f"              momenta, {form}, {component}u, point,",
                    f"              &split_params[{real_slot}u])) return 2;",
                    f"      split_params[{imag_slot}u] = 0.0;",
                )
            )
        elif role in {"coupling-real", "coupling-imag"}:
            lines.extend(
                (
                    f"      split_params[{real_slot}u] = "
                    "binding_context->"
                    f"{'coupling_im' if role == 'coupling-imag' else 'coupling_re'};",
                    f"      split_params[{imag_slot}u] = 0.0;",
                )
            )
        elif role == "model-parameter":
            parameter_index = contract.get("model_parameter_index")
            if type(parameter_index) is not int or parameter_index < 0:
                raise PreparedModelBundleError(
                    "native direct model parameter has no stable index"
                )
            lines.extend(
                (
                    "      if (!pac_direct_load_parameter(",
                    f"              parameters, {parameter_index}u,",
                    f"              &split_params[{real_slot}u], "
                    f"&split_params[{imag_slot}u])) return 2;",
                )
            )
        else:
            raise PreparedModelBundleError(
                f"unsupported native direct input role {role!r}"
            )
    return lines


def _cpp_float_literal(value: float) -> str:
    if not value.is_integer():
        return value.hex()
    return f"{value:.1f}"


def _independent_block_contract(
    kernel: PreparedKernelSpec,
    *,
    block_size: int = PREPARED_INDEPENDENT_BLOCK_SIZE,
) -> _IndependentBlockContract:
    """Construct lane-major expressions for independent scalar calls."""

    from symbolica import Expression, Replacement

    if PREPARED_INDEPENDENT_BLOCK_PROOF not in kernel.proof_classes:
        raise PreparedModelBundleError(
            f"prepared kernel {kernel.kernel_id} lacks the independent-block proof"
        )
    if block_size != PREPARED_INDEPENDENT_BLOCK_SIZE:
        raise PreparedModelBundleError(
            f"unsupported prepared independent block size {block_size}"
        )
    scalar_inputs = tuple(Expression.parse(item.symbol) for item in kernel.inputs)
    scalar_outputs = tuple(
        Expression.parse(value) for value in kernel.exact_expressions
    )
    parameters: list[object] = []
    outputs: list[object] = []
    input_layout: list[str] = []
    output_layout: list[str] = []
    for lane in range(block_size):
        lane_inputs = tuple(
            symbols.symbol(
                "prepared_block::"
                f"kernel_{kernel.canonical_signature}::lane_{lane}::input_{index}"
            )
            for index in range(kernel.input_arity)
        )
        forward = tuple(
            Replacement(source, target)
            for source, target in zip(scalar_inputs, lane_inputs, strict=True)
        )
        reverse = tuple(
            Replacement(target, source)
            for source, target in zip(scalar_inputs, lane_inputs, strict=True)
        )
        lane_outputs = tuple(
            expression.replace_multiple(forward) for expression in scalar_outputs
        )
        reconstructed = tuple(
            expression.replace_multiple(reverse).to_canonical_string()
            for expression in lane_outputs
        )
        expected = tuple(
            expression.to_canonical_string() for expression in scalar_outputs
        )
        if reconstructed != expected:
            raise PreparedModelBundleError(
                f"prepared kernel {kernel.kernel_id} block lane {lane} "
                "does not reconstruct its scalar expressions"
            )
        lane_symbols = {value.to_canonical_string() for value in lane_inputs}
        used_symbols = {
            symbol.to_canonical_string()
            for expression in lane_outputs
            for symbol in expression.get_all_symbols(False)
            if not symbol.is_constant()
        }
        if not used_symbols.issubset(lane_symbols):
            raise PreparedModelBundleError(
                f"prepared kernel {kernel.kernel_id} block lane {lane} "
                "contains inputs from another lane"
            )
        parameters.extend(lane_inputs)
        outputs.extend(lane_outputs)
        input_layout.extend(
            f"lane:{lane}:{item}" for item in _kernel_input_layout(kernel)
        )
        output_layout.extend(f"lane:{lane}:{item}" for item in kernel.output_layout)
    return _IndependentBlockContract(
        parameters=tuple(parameters),
        outputs=tuple(outputs),
        input_layout=tuple(input_layout),
        output_layout=tuple(output_layout),
    )


def _compile_independent_block_variant(
    kernel: PreparedKernelSpec,
    *,
    settings: SymbolicaEvaluatorSettings,
    staging: Path,
    backend: PreparedBackend,
    optimization_settings_digest: str,
) -> tuple[PreparedKernelVariantRecord, dict[str, Path]]:
    contract = _independent_block_contract(kernel)
    variant_id = f"independent-block-{PREPARED_INDEPENDENT_BLOCK_SIZE}"
    variant_staging = staging / "variants" / variant_id
    variant_staging.mkdir(parents=True, exist_ok=True)
    adapter = _compile_symbolica_outputs(
        contract.outputs,
        list(contract.parameters),
        merge_evaluators_strategy=False,
        verbose_evaluator_build=False,
        real_params=(),
        symbolica_settings=replace(settings, compiled_output_chunk_size=None),
        jit_compile=True,
        label=(
            f"prepared_{kernel.contract_kind}_{kernel.kernel_id:06d}_"
            f"independent_block_{PREPARED_INDEPENDENT_BLOCK_SIZE}"
        ),
    )
    raw_manifest = _symbolica_evaluator_artifact_manifest(adapter, variant_staging)
    manifest, payloads = _relocate_manifest_payloads(
        raw_manifest,
        staging=variant_staging,
        kernel_id=kernel.kernel_id,
        variant_id=variant_id,
    )
    _validate_backend_manifest(manifest, settings=settings)
    return (
        PreparedKernelVariantRecord(
            variant_id=variant_id,
            variant_abi=PREPARED_KERNEL_VARIANT_ABI,
            kind="independent-block",
            block_size=PREPARED_INDEPENDENT_BLOCK_SIZE,
            lane_layout="lane-major",
            base_kernel_id=kernel.kernel_id,
            base_canonical_signature=kernel.canonical_signature,
            base_expression_digest=prepared_expression_digest(kernel.exact_expressions),
            base_input_contract_digest=prepared_input_contract_digest(
                _kernel_input_layout(kernel),
                tuple(item.to_dict() for item in kernel.inputs),
            ),
            base_output_contract_digest=prepared_output_contract_digest(
                kernel.output_layout
            ),
            backend=backend,
            optimization_settings_digest=optimization_settings_digest,
            input_arity=len(contract.parameters),
            output_arity=len(contract.outputs),
            input_lane_stride=kernel.input_arity,
            output_lane_stride=kernel.output_dimension,
            input_layout=contract.input_layout,
            output_layout=contract.output_layout,
            f64_evaluator_manifest=manifest,
        ),
        payloads,
    )


def _kernel_input_layout(kernel: PreparedKernelSpec) -> tuple[str, ...]:
    return tuple(f"{item.role}:{item.component}" for item in kernel.inputs)


def _relocate_manifest_payloads(
    manifest: Mapping[str, object],
    *,
    staging: Path,
    kernel_id: int,
    variant_id: str | None = None,
) -> tuple[dict[str, object], dict[str, Path]]:
    payloads: dict[str, Path] = {}
    relocated: dict[Path, str] = {}
    counters: dict[str, int] = {}

    def relocate(path: object, field: str) -> str:
        if not isinstance(path, str) or not path:
            raise PreparedModelBundleError(
                f"prepared evaluator {field} must be a nonempty path"
            )
        source = Path(path)
        if not source.is_absolute():
            source = staging / source
        source = source.resolve()
        if not source.is_file() or source.is_symlink():
            raise PreparedModelBundleError(
                f"prepared evaluator payload does not exist: {source}"
            )
        existing = relocated.get(source)
        if existing is not None:
            return existing
        count = counters.get(field, 0)
        counters[field] = count + 1
        stem = field.removesuffix("_path").replace("_", "-")
        suffix = "".join(source.suffixes)
        root = PurePosixPath("kernels") / f"{kernel_id:06d}"
        if variant_id is not None:
            root = root / "variants" / variant_id
        member = (root / f"{stem}-{count}{suffix}").as_posix()
        relocated[source] = member
        payloads[member] = source
        return member

    def visit(value: object) -> object:
        if isinstance(value, Mapping):
            result: dict[str, object] = {}
            for key, child in value.items():
                if key in _PATH_FIELDS:
                    result[str(key)] = (
                        None if child is None else relocate(child, str(key))
                    )
                elif key in _PATH_LIST_FIELDS:
                    if not isinstance(child, Sequence) or isinstance(
                        child, (str, bytes, bytearray)
                    ):
                        raise PreparedModelBundleError(
                            f"prepared evaluator {key} must be a path array"
                        )
                    result[str(key)] = [relocate(item, str(key)) for item in child]
                else:
                    result[str(key)] = visit(child)
            return result
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [visit(item) for item in value]
        return value

    result = visit(manifest)
    if not isinstance(result, dict):  # pragma: no cover - mapping root invariant
        raise PreparedModelBundleError("prepared evaluator manifest must be an object")
    return result, payloads


def _validate_backend_manifest(
    manifest: Mapping[str, object],
    *,
    settings: SymbolicaEvaluatorSettings,
) -> None:
    if settings.backend != "jit":
        if manifest.get("kind") != "compiled-complex-evaluator":
            raise PreparedModelBundleError(
                "prepared native evaluator has an unexpected manifest kind"
            )
        return
    expected = {
        "kind": "symjit-application-evaluator",
        "application_abi": SYMJIT_APPLICATION_ABI,
        "compiler_type": "native",
        "translation_mode": "indirect",
        "word_bits": 64,
        "endianness": "little",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise PreparedModelBundleError(
                f"prepared JIT evaluator has incompatible {key}: {manifest.get(key)!r}"
            )
    if manifest.get("required_defuns") != []:
        raise PreparedModelBundleError(
            "prepared JIT evaluators must not depend on external functions"
        )
    plane_application = manifest.get("plane_application")
    if not isinstance(plane_application, Mapping):
        raise PreparedModelBundleError(
            "prepared JIT evaluator has no direct-arena plane application"
        )
    expected_plane = {
        "application_abi": SYMJIT_PLANE_APPLICATION_ABI,
        "storage_abi": SYMJIT_APPLICATION_ABI,
        "element_layout": "split-complex-plane-major",
        "descriptor_order": "inputs-re-im-then-outputs-re-im",
        "input_complex_count": manifest.get("input_len"),
        "output_complex_count": manifest.get("output_len"),
        "input_plane_count": 2 * int(manifest.get("input_len", -1)),
        "output_plane_count": 2 * int(manifest.get("output_len", -1)),
        "compiler_type": "native",
        "translation_mode": "symbolica-structured-instructions",
        "simd": True,
        "complex": True,
        "fast_math": True,
        "fast_complex": False,
        "threading": False,
        "direct_arena": True,
    }
    for key, value in expected_plane.items():
        if plane_application.get(key) != value:
            raise PreparedModelBundleError(
                "prepared JIT plane application has incompatible "
                f"{key}: {plane_application.get(key)!r}"
            )
    expected_level = PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL
    if manifest.get("optimization_level") != expected_level:
        raise PreparedModelBundleError(
            "prepared JIT evaluator is not portable: expected SymJIT "
            f"optimization level {expected_level}, found "
            f"{manifest.get('optimization_level')!r}"
        )
    if plane_application.get("optimization_level") != expected_level:
        raise PreparedModelBundleError(
            "prepared JIT plane application is not portable: expected SymJIT "
            f"optimization level {expected_level}, found "
            f"{plane_application.get('optimization_level')!r}"
        )
    manifest_settings = manifest.get("settings")
    if not isinstance(manifest_settings, Mapping) or (
        manifest_settings.get("jit_optimization_level") != expected_level
    ):
        raise PreparedModelBundleError(
            "prepared JIT evaluator settings do not attest the portable "
            f"SymJIT optimization level {expected_level}"
        )


def _optimization_metadata(settings: SymbolicaEvaluatorSettings) -> dict[str, object]:
    result = settings.to_json_dict()
    result["compiled_output_dir"] = None
    result["compiled_output_chunk_size"] = None
    return result


def _prepared_target(
    backend: PreparedBackend,
    evaluator: EvaluatorConfig,
) -> dict[str, object]:
    try:
        if backend == "jit":
            return symjit_storage_v3_target()
        return native_prepared_target(include_cpu_features=evaluator.cpp.native_arch)
    except PreparedTargetError as error:
        raise PreparedModelBundleError(str(error)) from error


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _cpp_optimization_level(value: str) -> int:
    normalized = value.strip().lower()
    if normalized.startswith("-o"):
        normalized = normalized[2:]
    elif normalized.startswith("o"):
        normalized = normalized[1:]
    if normalized in {"0", "1", "2", "3"}:
        return int(normalized)
    raise PreparedModelBundleError(
        f"unsupported prepared C++ optimization level {value!r}"
    )


__all__ = [
    "PreparedModelBuildResult",
    "prepare_model_bundle",
    "prepared_symbolica_settings",
]
