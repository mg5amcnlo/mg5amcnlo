# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.models import BuiltinSMModel, CompiledUFOModel, compile_model_source
from pyamplicol.models.prepared_catalog import build_prepared_kernel_catalog
from pyamplicol.models.recurrence_catalog_builder import (
    build_recurrence_template_catalog,
)
from pyamplicol.models.recurrence_direct_intrinsics import (
    CertifiedRecurrenceIntrinsic,
)
from pyamplicol.models.recurrence_direct_template import (
    NATIVE_DIRECT_APPLICATION_ABI,
    RECURRENCE_DIRECT_BACKEND_ABI,
    RECURRENCE_DIRECT_IDENTITY_FINALIZER,
    RECURRENCE_DIRECT_TEMPLATE_ABI,
    SYMJIT_DIRECT_APPLICATION_ABI,
    PreparedJitDirectSourceV1,
    PreparedNativeDirectCallableSpecV1,
    PreparedNativeDirectSourceV1,
    RecurrenceDirectPayloadBindingV1,
    RecurrenceDirectTemplateCatalogV1,
    RecurrenceDirectTemplateError,
    RecurrenceDirectTemplateV1,
    _build_certified_intrinsic_binding,
    _build_prepared_jit_direct_binding,
    _uniform_binding_coupling,
    build_prepared_native_direct_callable_specs,
    build_recurrence_direct_template_catalog,
    native_direct_entry_point,
    prepared_kernel_payload_digest,
)
from pyamplicol.models.recurrence_template import ExactComplexRationalV1

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64
_DIGEST_D = "d" * 64
_DIGEST_E = "e" * 64
_DIGEST_F = "f" * 64
_UFO_SM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)


def _payload_binding(
    *,
    kind: str = "pending-direct-call-abi",
) -> RecurrenceDirectPayloadBindingV1:
    if kind == "rusticol-intrinsic":
        return RecurrenceDirectPayloadBindingV1(
            kind="rusticol-intrinsic",
            payload_digest=_DIGEST_B,
            runtime_template="rusticol.source-fill.vector.v1:test",
        )
    return RecurrenceDirectPayloadBindingV1(
        kind="pending-direct-call-abi",
        payload_digest=_DIGEST_B,
        prepared_kernel_id=0,
    )


def _template(
    *,
    executor_id: int = 0,
    evaluator_binding_id: int | None = None,
    role: str = "contribution",
    backend: str = "jit",
    payload_kind: str = "pending-direct-call-abi",
) -> RecurrenceDirectTemplateV1:
    operations = {
        "source": "initialize",
        "contribution": "add",
        "finalization": "finalize-in-place",
        "closure": "closure-add",
    }
    parent_counts = {
        "source": (),
        "contribution": (4, 4),
        "finalization": (4,),
        "closure": (4, 4),
    }
    return RecurrenceDirectTemplateV1(
        template_id=f"direct:{role}:{executor_id}",
        direct_executor_id=executor_id,
        evaluator_binding_id=(
            executor_id if evaluator_binding_id is None else evaluator_binding_id
        ),
        evaluator_resolver_key=f"evaluator:{role}:{executor_id}",
        role=role,  # type: ignore[arg-type]
        parent_arity=len(parent_counts[role]),
        parent_component_counts=parent_counts[role],
        destination_component_count=4 if role != "closure" else 1,
        momentum_operand_count=(
            0 if role == "closure" else len(parent_counts[role]) or 1
        ),
        destination_operation=operations[role],  # type: ignore[arg-type]
        coupling_slot_count=1 if role == "contribution" else 0,
        parameter_slot_count=1,
        semantic_template_ids=(f"semantic:{role}:{executor_id}",),
        exact_expression_digest=_DIGEST_A,
        payload_binding=_payload_binding(kind=payload_kind),
        backend=backend,  # type: ignore[arg-type]
        target_triple=(
            "symjit-storage-v3-portable" if backend == "jit" else "x86_64-linux"
        ),
        portable=backend == "jit",
        optimization_level=2 if backend == "jit" else 3,
        alignment_bytes=64,
        simd_axis="points-contiguous",
        destination_aliasing=role == "finalization",
    )


def _catalog(
    templates: tuple[RecurrenceDirectTemplateV1, ...],
    *,
    backend: str = "jit",
) -> RecurrenceDirectTemplateCatalogV1:
    return RecurrenceDirectTemplateCatalogV1(
        templates=templates,
        backend=backend,  # type: ignore[arg-type]
        target_triple=(
            "symjit-storage-v3-portable" if backend == "jit" else "x86_64-linux"
        ),
        portable=backend == "jit",
        optimization_level=2 if backend == "jit" else 3,
        compiled_model_digest=_DIGEST_A,
        recurrence_template_catalog_digest=_DIGEST_B,
        prepared_kernel_pack_digest=_DIGEST_C,
        prepared_kernel_contract_digest=_DIGEST_D,
        prepared_kernel_payload_digest=_DIGEST_E,
        optimization_settings_digest=_DIGEST_F,
    )


def test_direct_template_catalog_round_trips_canonical_payload() -> None:
    source = _template(
        executor_id=0,
        role="source",
        payload_kind="rusticol-intrinsic",
    )
    contribution = _template(executor_id=1)
    catalog = _catalog((source, contribution))

    restored = RecurrenceDirectTemplateCatalogV1.from_dict(
        json.loads(catalog.canonical_json)
    )

    assert restored == catalog
    assert restored.abi == RECURRENCE_DIRECT_TEMPLATE_ABI
    assert restored.backend_abi == RECURRENCE_DIRECT_BACKEND_ABI
    assert restored.catalog_digest == catalog.catalog_digest
    assert restored.direct_executor_id_for("source", 0) == 0
    assert restored.direct_executor_id_for("contribution", 1) == 1
    assert not restored.executable


def test_catalog_digest_authenticates_serialized_metadata() -> None:
    payload = _catalog((_template(),)).to_dict()
    payload["prepared_kernel_payload_digest"] = "0" * 64

    with pytest.raises(RecurrenceDirectTemplateError, match="catalog digest"):
        RecurrenceDirectTemplateCatalogV1.from_dict(payload)


def test_direct_jit_templates_require_portable_symjit_o2() -> None:
    template = _template()

    with pytest.raises(RecurrenceDirectTemplateError, match="portable SymJIT O2"):
        replace(template, optimization_level=3, semantic_digest="")
    with pytest.raises(RecurrenceDirectTemplateError, match="portable SymJIT O2"):
        replace(template, portable=False, semantic_digest="")


def test_direct_cpp_and_asm_templates_are_target_native() -> None:
    for backend in ("cpp", "asm"):
        template = _template(backend=backend)
        assert template.optimization_level == 3
        assert not template.portable
        with pytest.raises(RecurrenceDirectTemplateError, match="target-native"):
            replace(template, portable=True, semantic_digest="")


def test_role_fixes_destination_operation_parent_contract_and_aliasing() -> None:
    template = _template()

    with pytest.raises(RecurrenceDirectTemplateError, match="must use 'add'"):
        replace(template, destination_operation="initialize", semantic_digest="")
    with pytest.raises(
        RecurrenceDirectTemplateError, match="cover every nonempty parent"
    ):
        replace(template, parent_component_counts=(4,), semantic_digest="")
    with pytest.raises(RecurrenceDirectTemplateError, match="only direct finalization"):
        replace(template, destination_aliasing=True, semantic_digest="")


def test_catalog_requires_dense_ids_and_unique_semantic_mapping() -> None:
    first = _template(executor_id=0, evaluator_binding_id=4)
    duplicate_mapping = replace(
        _template(executor_id=1, evaluator_binding_id=5),
        evaluator_binding_id=4,
        semantic_digest="",
    )
    with pytest.raises(RecurrenceDirectTemplateError, match="mappings must be unique"):
        _catalog((first, duplicate_mapping))

    with pytest.raises(RecurrenceDirectTemplateError, match="dense zero-based"):
        _catalog((_template(executor_id=1),))


def test_pending_binding_cannot_claim_direct_payload_paths() -> None:
    with pytest.raises(RecurrenceDirectTemplateError, match="cannot claim"):
        RecurrenceDirectPayloadBindingV1(
            kind="pending-direct-call-abi",
            prepared_kernel_id=0,
            payload_digest=_DIGEST_A,
            payload_paths=("kernels/000000/direct.symjit",),
        )


def test_payload_binding_rejects_predecessor_abi_with_regeneration_message() -> None:
    payload = _payload_binding().to_dict()
    payload["abi"] = "pyamplicol-recurrence-direct-" + "payload-binding-v1"

    with pytest.raises(RecurrenceDirectTemplateError, match="regenerate"):
        RecurrenceDirectPayloadBindingV1.from_dict(payload)


def test_certified_intrinsic_parent_permutation_round_trips_and_is_authenticated() -> (
    None
):
    binding = _build_certified_intrinsic_binding(
        CertifiedRecurrenceIntrinsic(
            runtime_template=("rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1"),
            contract_digest=_DIGEST_A,
            constant_scale=-1.0j,
            model_parameter_index=17,
            parent_permutation=(1, 0),
        )
    )

    assert binding.contribution_parent_permutation == (1, 0)
    assert binding.to_dict()["contribution_parent_permutation"] == [1, 0]
    assert RecurrenceDirectPayloadBindingV1.from_dict(binding.to_dict()) == binding

    tampered = binding.to_dict()
    tampered["contribution_parent_permutation"] = [0, 1]
    with pytest.raises(
        RecurrenceDirectTemplateError,
        match="payload digest does not match",
    ):
        RecurrenceDirectPayloadBindingV1.from_dict(tampered)


@pytest.mark.parametrize(
    "parent_permutation",
    ((0, 0), (1, 1), (0,), (0, 1, 2)),
)
def test_payload_binding_rejects_malformed_parent_permutation(
    parent_permutation: tuple[int, ...],
) -> None:
    with pytest.raises(
        RecurrenceDirectTemplateError,
        match="parent permutation must be",
    ):
        replace(
            _payload_binding(),
            contribution_parent_permutation=parent_permutation,  # type: ignore[arg-type]
        )


def test_only_contribution_intrinsics_allow_reversed_parents() -> None:
    with pytest.raises(
        RecurrenceDirectTemplateError,
        match="non-contribution intrinsics require the identity",
    ):
        replace(
            _payload_binding(kind="rusticol-intrinsic"),
            contribution_parent_permutation=(1, 0),
        )
    with pytest.raises(
        RecurrenceDirectTemplateError,
        match="prepared direct payloads require the identity",
    ):
        replace(
            _payload_binding(),
            contribution_parent_permutation=(1, 0),
        )


def test_per_kernel_payload_digest_is_deterministic_and_complete() -> None:
    records = {
        "kernels/000000/application.symjit": (3, _DIGEST_A),
        "kernels/000000/exact.bin": (7, _DIGEST_B),
    }
    first = prepared_kernel_payload_digest(
        kernel_id=0,
        payload_records=records,
        referenced_paths=tuple(reversed(tuple(records))),
    )
    second = prepared_kernel_payload_digest(
        kernel_id=0,
        payload_records=records,
        referenced_paths=tuple(records),
    )
    assert first == second
    with pytest.raises(RecurrenceDirectTemplateError, match="is absent"):
        prepared_kernel_payload_digest(
            kernel_id=0,
            payload_records=records,
            referenced_paths=("missing",),
        )


def test_direct_jit_binding_complexifies_real_inputs_with_shared_zero() -> None:
    contracts = (
        {"component": 0, "role": "left-current"},
        {"component": 1, "role": "left-momentum"},
        {"component": 2, "role": "coupling-real"},
        {
            "component": 3,
            "model_parameter_index": 4,
            "role": "model-parameter",
        },
    )
    source = PreparedJitDirectSourceV1(
        prepared_kernel_id=7,
        source_application_path="kernels/000007/application.plane.symjit",
        source_application_sha256=_DIGEST_A,
        source_application_abi=SYMJIT_DIRECT_APPLICATION_ABI,
        input_contracts=tuple(
            json.dumps(
                contract,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            for contract in contracts
        ),
        exact_expressions=("pyamplicol::prepared_output",),
        output_arity=1,
    )

    binding = _build_prepared_jit_direct_binding(
        source=source,
        role="contribution",
        parent_component_counts=(4,),
        destination_component_count=4,
        binding_coupling=ExactComplexRationalV1.from_fractions(
            Fraction(-3, 2),
            Fraction(1, 4),
        ),
        prepared_template_semantic_digest=_DIGEST_B,
    )
    parameter_bindings = tuple(json.loads(item) for item in binding.parameter_bindings)
    scalar_projections = tuple(json.loads(item) for item in binding.scalar_projections)

    assert len(parameter_bindings) == 2 * len(contracts)
    assert [item["kind"] for item in parameter_bindings] == [
        "plane",
        "plane",
        "plane",
        "scalar",
        "scalar",
        "scalar",
        "scalar",
        "scalar",
    ]
    zero_indices = [
        item["index"] for item in (parameter_bindings[3], parameter_bindings[5])
    ]
    assert zero_indices[0] == zero_indices[1]
    assert scalar_projections[zero_indices[0]] == {
        "kind": "literal",
        "value": 0.0,
    }
    assert {"kind": "literal", "value": -1.5} in scalar_projections
    assert not any(
        item.get("kind") == "parameter" and item.get("index") == 2
        for item in scalar_projections
    )
    assert binding.contribution_parent_permutation == (0, 1)
    with pytest.raises(
        RecurrenceDirectTemplateError,
        match="prepared direct payloads require the identity",
    ):
        replace(binding, contribution_parent_permutation=(1, 0))


def test_direct_jit_binding_rejects_conflicting_semantic_couplings() -> None:
    first = ExactComplexRationalV1.one()
    second = ExactComplexRationalV1.from_fractions(2)
    records = {
        "first": SimpleNamespace(binding_coupling=first),
        "second": SimpleNamespace(binding_coupling=second),
    }

    with pytest.raises(RecurrenceDirectTemplateError, match="conflicting"):
        _uniform_binding_coupling(
            ("first", "second"),
            records,
            required=True,
        )


@pytest.mark.parametrize("model_source", ("built-in", "ufo-sm"))
def test_direct_catalog_is_model_generic_and_covers_identity_finalizers(
    model_source: str,
) -> None:
    if model_source == "built-in":
        model = BuiltinSMModel()
    else:
        compiled = compile_model_source(
            _UFO_SM_ROOT / "sm.json",
            restriction=str((_UFO_SM_ROOT / "restrict_default.json").resolve()),
            use_cache=True,
        )
        model = CompiledUFOModel(compiled)
    prepared = build_prepared_kernel_catalog(model)
    semantic = build_recurrence_template_catalog(
        model,
        prepared,
        compiled_model_digest=_DIGEST_A,
        prepared_kernel_pack_digest=_DIGEST_C,
    )
    native_specs = build_prepared_native_direct_callable_specs(
        semantic,
        prepared.by_id,
    )
    assert native_specs
    assert all(
        kernel_id == spec.prepared_kernel_id for kernel_id, spec in native_specs.items()
    )
    assert all(spec.role != "source" for spec in native_specs.values())
    assert all(
        spec.native_entry_point
        == native_direct_entry_point(spec.role, spec.prepared_kernel_id)
        for spec in native_specs.values()
    )
    if model_source == "built-in":
        # Kernel IDs are assigned in content-signature order and may therefore
        # move when Symbolica changes an equivalent canonical spelling.  The
        # callable shape inventory is the stable execution contract.
        merged_parent_shapes = sorted(
            spec.parent_component_shapes
            for spec in native_specs.values()
            if len(spec.parent_component_shapes) > 1
        )
        assert merged_parent_shapes == [
            ((2, 2), (2, 4)),
            ((2, 2), (2, 4)),
            ((2, 2), (2, 4)),
            ((2, 2), (4, 2)),
            ((2, 2), (4, 2)),
            ((2, 2), (4, 2)),
        ]
    direct = build_recurrence_direct_template_catalog(
        semantic,
        backend="jit",
        target_triple="symjit-storage-v3-portable",
        portable=True,
        optimization_level=2,
        prepared_kernel_pack_digest=_DIGEST_C,
        prepared_kernel_contract_digest=_DIGEST_D,
        prepared_kernel_payload_digest=_DIGEST_E,
        optimization_settings_digest=_DIGEST_F,
        prepared_kernel_payload_digests={
            kernel.kernel_id: hashlib.sha256(
                f"{kernel.kernel_id}:{kernel.canonical_signature}".encode()
            ).hexdigest()
            for kernel in prepared.kernels
        },
        prepared_jit_sources={
            kernel.kernel_id: PreparedJitDirectSourceV1(
                prepared_kernel_id=kernel.kernel_id,
                source_application_path=(
                    f"kernels/{kernel.kernel_id:06d}/application.plane.symjit"
                ),
                source_application_sha256=hashlib.sha256(
                    f"application:{kernel.kernel_id}".encode()
                ).hexdigest(),
                source_application_abi=SYMJIT_DIRECT_APPLICATION_ABI,
                input_contracts=tuple(
                    json.dumps(
                        item.to_dict(),
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    for item in kernel.inputs
                ),
                exact_expressions=kernel.exact_expressions,
                output_arity=kernel.output_dimension,
            )
            for kernel in prepared.kernels
        },
    )

    assert [item.direct_executor_id for item in direct.templates] == list(
        range(len(direct.templates))
    )
    contract_roles = {
        "source": "source",
        "vertex": "contribution",
        "propagator": "finalization",
        "closure": "closure",
    }
    for binding_id, binding in enumerate(semantic.evaluator_bindings):
        role = contract_roles.get(binding.contract_kind)
        if role is not None:
            direct.direct_executor_id_for(role, binding_id)  # type: ignore[arg-type]
    identity_templates = tuple(
        item
        for item in direct.templates
        if item.payload_binding.runtime_template
        and item.payload_binding.runtime_template.startswith(
            RECURRENCE_DIRECT_IDENTITY_FINALIZER
        )
    )
    identity_propagator_ids = tuple(
        sorted(
            propagator.template_id
            for propagator in semantic.propagators
            if not propagator.applies_propagator
        )
    )
    assert len(identity_templates) == (1 if identity_propagator_ids else 0)
    assert all(item.role == "finalization" for item in identity_templates)
    assert all(
        item.payload_binding.runtime_template == RECURRENCE_DIRECT_IDENTITY_FINALIZER
        for item in identity_templates
    )
    assert all(
        item.semantic_template_ids == identity_propagator_ids
        for item in identity_templates
    )
    assert all(
        item.payload_binding.kind == "rusticol-intrinsic" for item in identity_templates
    )
    contribution_intrinsics = tuple(
        item
        for item in direct.templates
        if item.role == "contribution"
        and item.payload_binding.kind == "rusticol-intrinsic"
    )
    intrinsic_families = {
        item.payload_binding.runtime_template for item in contribution_intrinsics
    }
    assert {
        "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1",
        "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-b.v1",
    }.issubset(intrinsic_families)
    if model_source == "ufo-sm":
        assert any(
            item.payload_binding.contribution_parent_permutation == (1, 0)
            for item in contribution_intrinsics
        )
    prepared_templates = tuple(
        item
        for item in direct.templates
        if item.payload_binding.prepared_kernel_id is not None
    )
    assert prepared_templates
    assert all(
        item.payload_binding.kind == "prepared-direct-call"
        for item in prepared_templates
    )
    assert all(item.payload_binding.executable for item in prepared_templates)
    assert all(
        item.payload_binding.source_application_abi == SYMJIT_DIRECT_APPLICATION_ABI
        for item in prepared_templates
    )
    assert all(
        item.payload_binding.direct_application_abi == SYMJIT_DIRECT_APPLICATION_ABI
        for item in prepared_templates
    )
    assert all(
        item.payload_binding.exact_factor_scalar_slots == (0, 1)
        for item in prepared_templates
    )
    assert all(
        item.payload_binding.payload_paths
        == (item.payload_binding.source_application_path,)
        for item in prepared_templates
    )
    assert all(
        item.payload_binding.contribution_parent_permutation == (0, 1)
        for item in prepared_templates
    )
    assert all(
        item.payload_binding.contribution_parent_permutation == (0, 1)
        for item in direct.templates
        if item.role != "contribution"
    )
    for item in prepared_templates:
        payload = item.payload_binding.to_dict()
        for projection in payload["input_plane_projections"]:
            kind = projection["kind"]
            if kind == "parent-current":
                parent = projection["parent"]
                assert parent < len(item.parent_component_counts)
                assert projection["component"] < item.parent_component_counts[parent]
            elif kind == "momentum":
                assert projection["operand"] < item.momentum_operand_count
                assert projection["lorentz_component"] < 4
            elif kind in {"destination-current", "destination-amplitude"}:
                assert projection["component"] < item.destination_component_count
            else:  # pragma: no cover - schema validation owns this branch
                raise AssertionError(f"unknown projection kind {kind!r}")
    assert (
        RecurrenceDirectTemplateCatalogV1.from_dict(json.loads(direct.canonical_json))
        == direct
    )
    native_direct = build_recurrence_direct_template_catalog(
        semantic,
        backend="cpp",
        target_triple="x86_64-linux",
        portable=False,
        optimization_level=3,
        prepared_kernel_pack_digest=_DIGEST_C,
        prepared_kernel_contract_digest=_DIGEST_D,
        prepared_kernel_payload_digest=_DIGEST_E,
        optimization_settings_digest=_DIGEST_F,
        prepared_kernel_payload_digests={
            kernel.kernel_id: hashlib.sha256(
                f"{kernel.kernel_id}:{kernel.canonical_signature}".encode()
            ).hexdigest()
            for kernel in prepared.kernels
        },
        prepared_native_sources={
            kernel_id: PreparedNativeDirectSourceV1(
                prepared_kernel_id=kernel_id,
                role=spec.role,
                native_entry_point=spec.native_entry_point,
                source_application_path=(
                    f"kernels/{kernel_id:06d}/libprepared-native.so"
                ),
                source_application_sha256=hashlib.sha256(
                    f"native-library:{kernel_id}".encode()
                ).hexdigest(),
                source_application_abi="symbolica.compiled-cpp.complex-f64.v1",
                input_contracts=spec.input_contracts,
                exact_expressions=spec.exact_expressions,
                output_arity=spec.output_arity,
            )
            for kernel_id, spec in native_specs.items()
        },
    )
    native_prepared = tuple(
        item
        for item in native_direct.templates
        if item.payload_binding.prepared_kernel_id is not None
    )
    assert native_prepared
    assert all(
        item.payload_binding.kind == "prepared-direct-call"
        and item.payload_binding.direct_application_abi == NATIVE_DIRECT_APPLICATION_ABI
        and item.payload_binding.native_entry_point
        == native_direct_entry_point(
            item.role,
            item.payload_binding.prepared_kernel_id,
        )
        for item in native_prepared
    )
    assert native_direct.executable
    assert model_source not in direct.canonical_json


def test_native_direct_specs_merge_shapes_but_reject_semantic_conflicts() -> None:
    import pyamplicol.models.recurrence_direct_template as direct_template

    common = {
        "prepared_kernel_id": 20,
        "role": "contribution",
        "native_entry_point": native_direct_entry_point(
            "contribution",
            20,
        ),
        "input_contracts": (
            json.dumps(
                {"component": 1, "role": "left-current"},
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
        "exact_expressions": ("test::out0", "test::out1"),
        "output_arity": 2,
        "destination_component_counts": (2,),
    }
    narrow = PreparedNativeDirectCallableSpecV1(
        **common,
        parent_component_shapes=((2, 2),),
    )
    wide = replace(
        narrow,
        parent_component_shapes=((4, 2),),
        destination_component_counts=(2, 4),
    )
    merged = direct_template._merge_prepared_native_direct_callable_specs(narrow, wide)

    assert merged.parent_component_shapes == ((2, 2), (4, 2))
    assert merged.destination_component_counts == (2, 4)

    incompatible = replace(wide, exact_expressions=("other::out0", "other::out1"))
    with pytest.raises(RecurrenceDirectTemplateError, match="exact_expressions"):
        direct_template._merge_prepared_native_direct_callable_specs(
            narrow, incompatible
        )

    out_of_bounds = replace(
        narrow,
        input_contracts=(
            json.dumps(
                {"component": 2, "role": "left-current"},
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
    )
    with pytest.raises(RecurrenceDirectTemplateError, match="outside"):
        direct_template._validate_prepared_native_direct_projection_bounds(
            out_of_bounds
        )
