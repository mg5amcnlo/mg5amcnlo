# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol.models import prepared as prepared_module
from pyamplicol.models.prepared import (
    EAGER_KERNEL_ABI,
    PREPARED_KERNEL_PACK_IDENTITY_ABI,
    PREPARED_KERNEL_VARIANT_ABI,
    PREPARED_MODEL_BUNDLE_KIND,
    PREPARED_MODEL_BUNDLE_SCHEMA_VERSION,
    PREPARED_MODEL_COMPILED_MODEL_PATH,
    PREPARED_MODEL_MANIFEST_PATH,
    PreparedKernelPack,
    PreparedKernelPackIdentity,
    PreparedKernelRecord,
    PreparedKernelVariantRecord,
    PreparedModelBundleError,
    load_prepared_model_bundle,
    prepared_compiled_model_digest,
    prepared_expression_digest,
    prepared_input_contract_digest,
    prepared_kernel_pack_identity,
    prepared_kernel_pack_manifest_identity_sha256,
    prepared_optimization_settings_digest,
    prepared_output_contract_digest,
    prepared_payload_identity_records,
    write_prepared_model_bundle,
)
from pyamplicol.models.recurrence_direct_template import (
    build_recurrence_direct_template_catalog,
    prepared_kernel_payload_digest,
)
from pyamplicol.models.recurrence_template import (
    CurrentStateTemplateV1,
    EvaluatorBindingV1,
    ParameterTemplateV1,
    PropagatorTemplateV1,
    RecurrenceTemplateCatalog,
)


def _kernel(
    kernel_id: int = 0,
    *,
    signature: str = "vertex:q-qbar-g:v1",
    contract_kind: str = "vertex",
) -> PreparedKernelRecord:
    root = f"kernels/{kernel_id}"
    return PreparedKernelRecord(
        kernel_id=kernel_id,
        contract_kind=contract_kind,  # type: ignore[arg-type]
        canonical_signature=signature,
        input_arity=2,
        output_arity=1,
        input_layout=("left-current", "right-current"),
        input_contracts=(
            {
                "role": "left-current",
                "component": 0,
                "symbol": "pyamplicol::left",
                "model_parameter_name": None,
                "model_parameter_index": None,
            },
            {
                "role": "right-current",
                "component": 0,
                "symbol": "pyamplicol::right",
                "model_parameter_name": None,
                "model_parameter_index": None,
            },
        ),
        output_layout=("current-contribution",),
        exact_expressions=("pyamplicol::left+pyamplicol::right",),
        exact_evaluator_state_path=f"{root}/exact.evaluator.bin",
        f64_evaluator_manifest={
            "kind": "symjit-application-evaluator",
            "optimization_level": 2,
            "settings": {"jit_optimization_level": 2},
            "input_len": 2,
            "output_len": 1,
            "application_path": f"{root}/application.symjit",
            "evaluator_state_path": f"{root}/exact.evaluator.bin",
        },
    )


def _pack(
    *kernels: PreparedKernelRecord,
    kernel_variants: tuple[PreparedKernelVariantRecord, ...] = (),
) -> PreparedKernelPack:
    return PreparedKernelPack(
        backend="jit",
        optimization_settings={
            "jit_optimization_level": 2,
            "cpe_rounds": "default",
        },
        producer={"distribution": "pyamplicol", "version": "0.1.0"},
        dependency_abis={
            "symbolica_serialization": "symbolica-community-v1",
            "symjit_application": "symjit-application-complex-f64-v1",
            "symjit_plane_application": "pyamplicol-symjit-plane-application-v2",
        },
        provenance={
            "model_content_sha256": "1" * 64,
            "compiler_revision": "2" * 40,
        },
        target={
            "portable": True,
            "word_bits": 64,
            "endianness": "little",
            "target_triple": "symjit-storage-v3-portable",
            "cpu_features": [],
        },
        resolver_manifest={
            "abi": "pyamplicol-prepared-kernel-catalog-v1",
            "model_name": "test-model",
        },
        kernels=kernels or (_kernel(),),
        kernel_variants=kernel_variants,
    )


def _block_variant(kernel: PreparedKernelRecord) -> PreparedKernelVariantRecord:
    settings = {"jit_optimization_level": 2, "cpe_rounds": "default"}
    root = f"kernels/{kernel.kernel_id}/variants/independent-block-4"
    return PreparedKernelVariantRecord(
        variant_id="independent-block-4",
        variant_abi=PREPARED_KERNEL_VARIANT_ABI,
        kind="independent-block",
        block_size=4,
        lane_layout="lane-major",
        base_kernel_id=kernel.kernel_id,
        base_canonical_signature=kernel.canonical_signature,
        base_expression_digest=prepared_expression_digest(kernel.exact_expressions),
        base_input_contract_digest=prepared_input_contract_digest(
            kernel.input_layout,
            kernel.input_contracts,
        ),
        base_output_contract_digest=prepared_output_contract_digest(
            kernel.output_layout
        ),
        backend="jit",
        optimization_settings_digest=prepared_optimization_settings_digest(settings),
        input_arity=4 * kernel.input_arity,
        output_arity=4 * kernel.output_arity,
        input_lane_stride=kernel.input_arity,
        output_lane_stride=kernel.output_arity,
        input_layout=tuple(
            f"lane:{lane}:{item}" for lane in range(4) for item in kernel.input_layout
        ),
        output_layout=tuple(
            f"lane:{lane}:{item}" for lane in range(4) for item in kernel.output_layout
        ),
        f64_evaluator_manifest={
            "kind": "symjit-application-evaluator",
            "optimization_level": 2,
            "settings": {"jit_optimization_level": 2},
            "input_len": 4 * kernel.input_arity,
            "output_len": 4 * kernel.output_arity,
            "application_path": f"{root}/application.symjit",
            "evaluator_state_path": f"{root}/evaluator-state.evaluator.bin",
        },
    )


def _compiled_model() -> dict[str, object]:
    return {
        "kind": "pyamplicol-compiled-model",
        "schema_version": 9,
        "model_compiler_version": 13,
        "model": {"name": "test-model"},
    }


def _payloads(*kernels: PreparedKernelRecord) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for kernel in kernels or (_kernel(),):
        result[kernel.exact_evaluator_state_path] = f"exact:{kernel.kernel_id}".encode()
        application_path = str(kernel.f64_evaluator_manifest["application_path"])
        result[application_path] = f"jit:{kernel.kernel_id}".encode()
    return result


def _recurrence_catalog(
    pack: PreparedKernelPack,
    payloads: Mapping[str, bytes],
    *,
    compiled_model_digest: str | None = None,
) -> RecurrenceTemplateCatalog:
    identity = prepared_kernel_pack_identity(
        pack,
        prepared_payload_identity_records(payloads),
    )
    return RecurrenceTemplateCatalog.create(
        compiled_model_digest=(
            prepared_compiled_model_digest(_compiled_model())
            if compiled_model_digest is None
            else compiled_model_digest
        ),
        prepared_kernel_pack_digest=identity.pack_digest,
    )


def _pack_with_direct_recurrence() -> tuple[
    PreparedKernelPack,
    dict[str, bytes],
    str,
]:
    kernel = _kernel()
    base_pack = _pack(kernel)
    payloads = _payloads(kernel)
    payload_records = prepared_payload_identity_records(payloads)
    identity = prepared_kernel_pack_identity(base_pack, payload_records)
    state = CurrentStateTemplateV1(
        template_id="state:test-scalar",
        particle_id=9000001,
        anti_particle_id=9000001,
        species_id="test-scalar",
        orientation="self-conjugate",
        statistics="boson",
        color_representation=1,
        basis="scalar",
        tensor_ordering=("scalar",),
        dimension=1,
        chirality=0,
        lc_color_shape_kind="singlet-forest",
        auxiliary_kind=None,
        mass_parameter_id=None,
        width_parameter_id=None,
    )
    identity_propagator = PropagatorTemplateV1(
        template_id="propagator:test-identity",
        state_template_id=state.template_id,
        applies_propagator=False,
        evaluator_resolver_key=None,
        numerator_expression_digest=None,
        denominator_expression_digest=None,
        mass_parameter_id=None,
        width_parameter_id=None,
        gauge=None,
        linearity_proof_template_id=None,
    )
    semantic_catalog = RecurrenceTemplateCatalog.create(
        compiled_model_digest=prepared_compiled_model_digest(_compiled_model()),
        prepared_kernel_pack_digest=identity.pack_digest,
        current_states=(state,),
        propagators=(identity_propagator,),
    )
    optimization_digest = prepared_optimization_settings_digest(
        base_pack.optimization_settings
    )
    direct_catalog = build_recurrence_direct_template_catalog(
        semantic_catalog,
        backend="jit",
        target_triple=str(base_pack.target["target_triple"]),
        portable=True,
        optimization_level=2,
        prepared_kernel_pack_digest=identity.pack_digest,
        prepared_kernel_contract_digest=identity.contract_digest,
        prepared_kernel_payload_digest=identity.payload_digest,
        optimization_settings_digest=optimization_digest,
        prepared_kernel_payload_digests={
            kernel.kernel_id: prepared_kernel_payload_digest(
                kernel_id=kernel.kernel_id,
                payload_records=payload_records,
                referenced_paths=kernel.referenced_payload_paths,
            )
        },
    )
    pack = replace(
        base_pack,
        provenance={
            **dict(base_pack.provenance),
            "direct_template_catalog_digest": direct_catalog.catalog_digest,
            "prepared_kernel_contract_digest": identity.contract_digest,
            "prepared_kernel_payload_digest": identity.payload_digest,
            "prepared_kernel_pack_digest": identity.pack_digest,
            "recurrence_direct_template_abi": direct_catalog.abi,
            "recurrence_direct_template_digest": direct_catalog.catalog_digest,
        },
        recurrence_template=semantic_catalog.to_dict(),
        recurrence_direct_template=direct_catalog.to_dict(),
    )
    return pack, payloads, direct_catalog.catalog_digest


def _pack_with_bound_recurrence_kernel(
    *,
    prepared_kernel_id: int = 0,
    callable_signature: str | None = None,
    input_layout: tuple[str, ...] | None = None,
    output_layout: tuple[str, ...] | None = None,
    exact_expression_digests: tuple[str, ...] | None = None,
) -> PreparedKernelPack:
    signature = "a" * 64
    kernel = _kernel(signature=signature, contract_kind="model-parameter")
    pack = _pack(kernel)
    payloads = _payloads(kernel)
    expression_digest = hashlib.sha256(
        kernel.exact_expressions[0].encode("utf-8")
    ).hexdigest()
    parameter = ParameterTemplateV1(
        template_id="parameter:derived",
        name="derived",
        parameter_kind="derived",
        value_type="real",
        mutable=False,
        default_value=None,
        exact_expression_digest=expression_digest,
        dependency_parameter_ids=(),
        prepared_parameter_id=0,
    )
    canonical_inputs = tuple(
        json.dumps(
            dict(contract),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for contract in kernel.input_contracts
    )
    binding = EvaluatorBindingV1(
        resolver_key="evaluator:model-parameter:derived",
        prepared_kernel_id=prepared_kernel_id,
        contract_kind="model-parameter",
        callable_signature=(
            signature if callable_signature is None else callable_signature
        ),
        input_state_template_ids=(),
        output_state_template_id=None,
        input_layout=canonical_inputs if input_layout is None else input_layout,
        output_layout=kernel.output_layout if output_layout is None else output_layout,
        exact_expression_digests=(
            (expression_digest,)
            if exact_expression_digests is None
            else exact_expression_digests
        ),
        semantic_template_ids=(parameter.template_id,),
    )
    identity = prepared_kernel_pack_identity(
        pack,
        prepared_payload_identity_records(payloads),
    )
    catalog = RecurrenceTemplateCatalog.create(
        compiled_model_digest=prepared_compiled_model_digest(_compiled_model()),
        prepared_kernel_pack_digest=identity.pack_digest,
        parameters=(parameter,),
        evaluator_bindings=(binding,),
    )
    return replace(pack, recurrence_template=catalog.to_dict())


def _variant_payloads(
    *variants: PreparedKernelVariantRecord,
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for variant in variants:
        for path in variant.referenced_payload_paths:
            result[path] = f"variant:{variant.base_kernel_id}:{path}".encode()
    return result


def _valid_bundle(tmp_path: Path) -> Path:
    kernel = _kernel()
    return write_prepared_model_bundle(
        tmp_path / "prepared",
        compiled_model=_compiled_model(),
        kernel_pack=_pack(kernel),
        payloads=_payloads(kernel),
    )


def _entries(path: Path) -> list[tuple[zipfile.ZipInfo, bytes]]:
    with zipfile.ZipFile(path, "r") as archive:
        return [(info, archive.read(info)) for info in archive.infolist()]


def _rewrite(
    path: Path,
    entries: list[tuple[zipfile.ZipInfo, bytes]],
) -> None:
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for original, data in entries:
            info = zipfile.ZipInfo(original.filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = original.external_attr
            archive.writestr(info, data)


def _mutate_manifest(
    path: Path,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    entries = _entries(path)
    updated: list[tuple[zipfile.ZipInfo, bytes]] = []
    for info, data in entries:
        if info.filename == PREPARED_MODEL_MANIFEST_PATH:
            manifest = json.loads(data)
            assert isinstance(manifest, dict)
            mutate(manifest)
            data = (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("ascii")
        updated.append((info, data))
    _rewrite(path, updated)


def test_prepared_model_bundle_round_trip_and_payload_copy(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)

    assert path == (tmp_path / "prepared.pyamplicol-model").resolve()
    bundle = load_prepared_model_bundle(path)

    assert bundle.backend == "jit"
    assert bundle.compiled_model == _compiled_model()
    assert bundle.kernel_pack.kernels[0].canonical_signature == "vertex:q-qbar-g:v1"
    assert bundle.read_payload("kernels/0/application.symjit") == b"jit:0"
    with pytest.raises(TypeError):
        bundle.compiled_model["kind"] = "changed"  # type: ignore[index]

    extracted = bundle.copy_referenced_payloads(tmp_path / "extracted")
    extracted_paths = tuple(
        path.relative_to(tmp_path / "extracted").as_posix() for path in extracted
    )
    assert extracted_paths == (
        "kernels/0/application.symjit",
        "kernels/0/exact.evaluator.bin",
    )
    assert extracted[0].read_bytes() == b"jit:0"
    assert extracted[1].read_bytes() == b"exact:0"
    assert not (tmp_path / "extracted" / "model").exists()

    with zipfile.ZipFile(path, "r") as archive:
        assert archive.namelist() == [
            PREPARED_MODEL_MANIFEST_PATH,
            "kernels/0/application.symjit",
            "kernels/0/exact.evaluator.bin",
            PREPARED_MODEL_COMPILED_MODEL_PATH,
        ]
        manifest = json.loads(archive.read(PREPARED_MODEL_MANIFEST_PATH))
    assert manifest["kind"] == PREPARED_MODEL_BUNDLE_KIND
    assert manifest["schema_version"] == PREPARED_MODEL_BUNDLE_SCHEMA_VERSION
    assert manifest["eager_kernel_abi"] == EAGER_KERNEL_ABI
    assert manifest["kernel_pack"]["backend"] == "jit"
    assert "recurrence_template" not in manifest["kernel_pack"]
    assert {record["path"] for record in manifest["members"]} == {
        PREPARED_MODEL_COMPILED_MODEL_PATH,
        "kernels/0/application.symjit",
        "kernels/0/exact.evaluator.bin",
    }


def test_prepared_jit_pack_requires_plane_application_dependency_abi() -> None:
    pack = _pack(_kernel())
    stale_dependencies = dict(pack.dependency_abis)
    stale_dependencies.pop("symjit_plane_application")

    with pytest.raises(
        PreparedModelBundleError,
        match="authenticate SymJIT plane-application ABI",
    ):
        replace(pack, dependency_abis=stale_dependencies)

    predecessor_dependencies = dict(pack.dependency_abis)
    predecessor_dependencies["symjit_plane_application"] = (
        "pyamplicol-symjit-plane-application-" + "v1"
    )

    with pytest.raises(PreparedModelBundleError, match="regenerate"):
        replace(pack, dependency_abis=predecessor_dependencies)


def test_prepared_model_bundle_round_trips_optional_recurrence_template(
    tmp_path: Path,
) -> None:
    kernel = _kernel()
    base_pack = _pack(kernel)
    payloads = _payloads(kernel)
    catalog = _recurrence_catalog(base_pack, payloads)
    pack = replace(base_pack, recurrence_template=catalog.to_dict())

    path = write_prepared_model_bundle(
        tmp_path / "recurrence-ready",
        compiled_model=_compiled_model(),
        kernel_pack=pack,
        payloads=payloads,
    )
    loaded = load_prepared_model_bundle(path)

    assert loaded.kernel_pack.recurrence_template is not None
    decoded = loaded.kernel_pack.recurrence_template_catalog
    assert decoded is not None
    assert decoded.catalog_digest == catalog.catalog_digest
    with zipfile.ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read(PREPARED_MODEL_MANIFEST_PATH))
    assert (
        manifest["kernel_pack"]["recurrence_template"]["header"]["catalog_digest"]
        == catalog.catalog_digest
    )


def test_direct_template_catalog_round_trip_is_independently_authenticated(
    tmp_path: Path,
) -> None:
    pack, payloads, expected_digest = _pack_with_direct_recurrence()
    path = write_prepared_model_bundle(
        tmp_path / "direct",
        compiled_model=_compiled_model(),
        kernel_pack=pack,
        payloads=payloads,
    )

    bundle = load_prepared_model_bundle(path)
    catalog = bundle.kernel_pack.recurrence_direct_template_catalog

    assert catalog is not None
    assert catalog is bundle.kernel_pack.recurrence_direct_template_catalog
    assert catalog.catalog_digest == expected_digest
    assert bundle.direct_template_catalog_digest == expected_digest
    assert bundle.manifest["direct_template_catalog_digest"] == expected_digest
    assert len(expected_digest) == 64
    assert expected_digest == expected_digest.lower()
    assert set(expected_digest) <= set("0123456789abcdef")
    assert catalog.direct_executor_id_for("finalization", 0) == 0
    assert (
        catalog.templates[0].payload_binding.runtime_template
        == "rusticol.identity-finalize-in-place.v1"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda manifest: manifest.__setitem__(
                "direct_template_catalog_digest",
                str(manifest["direct_template_catalog_digest"]).upper(),
            ),
            "lowercase SHA-256",
        ),
        (
            lambda manifest: manifest.__setitem__(
                "direct_template_catalog_digest",
                "0" * 64,
            ),
            "does not match",
        ),
        (
            lambda manifest: manifest.pop("direct_template_catalog_digest"),
            "does not match",
        ),
    ),
)
def test_direct_template_catalog_manifest_rejects_malformed_identity(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    pack, payloads, _digest = _pack_with_direct_recurrence()
    path = write_prepared_model_bundle(
        tmp_path / "direct-malformed",
        compiled_model=_compiled_model(),
        kernel_pack=pack,
        payloads=payloads,
    )
    _mutate_manifest(path, mutate)

    with pytest.raises(PreparedModelBundleError, match=message):
        load_prepared_model_bundle(path)


def test_bundle_without_direct_companion_rejects_spurious_digest(
    tmp_path: Path,
) -> None:
    path = _valid_bundle(tmp_path)
    _mutate_manifest(
        path,
        lambda manifest: manifest.__setitem__(
            "direct_template_catalog_digest",
            None,
        ),
    )

    with pytest.raises(
        PreparedModelBundleError,
        match="advertises a direct template catalog that is absent",
    ):
        load_prepared_model_bundle(path)


def test_prepared_kernel_pack_rejects_stale_recurrence_template() -> None:
    kernel = _kernel()
    base_pack = _pack(kernel)
    catalog = _recurrence_catalog(base_pack, _payloads(kernel))
    payload = catalog.to_dict()
    payload["header"]["catalog_digest"] = "b" * 64

    with pytest.raises(PreparedModelBundleError, match="stale recurrence"):
        replace(base_pack, recurrence_template=payload)


def test_prepared_kernel_pack_authenticates_recurrence_callable_contract() -> None:
    pack = _pack_with_bound_recurrence_kernel()

    catalog = pack.recurrence_template_catalog
    assert catalog is not None
    assert catalog.evaluator_bindings[0].prepared_kernel_id == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"prepared_kernel_id": 7}, "unknown prepared kernel ID 7"),
        ({"callable_signature": "b" * 64}, "callable signature"),
        ({"input_layout": ("stale", "layout")}, "input layout"),
        ({"output_layout": ("stale",)}, "output layout"),
        ({"exact_expression_digests": ("c" * 64,)}, "exact expression digests"),
    ),
)
def test_prepared_kernel_pack_rejects_stale_recurrence_callable_contract(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(PreparedModelBundleError, match=message):
        _pack_with_bound_recurrence_kernel(**overrides)  # type: ignore[arg-type]


def test_prepared_bundle_rejects_recurrence_template_for_another_model(
    tmp_path: Path,
) -> None:
    kernel = _kernel()
    base_pack = _pack(kernel)
    payloads = _payloads(kernel)
    catalog = _recurrence_catalog(
        base_pack,
        payloads,
        compiled_model_digest="f" * 64,
    )
    pack = replace(base_pack, recurrence_template=catalog.to_dict())

    with pytest.raises(PreparedModelBundleError, match="compiled-model digest"):
        write_prepared_model_bundle(
            tmp_path / "wrong-model",
            compiled_model=_compiled_model(),
            kernel_pack=pack,
            payloads=payloads,
        )


def test_prepared_bundle_rejects_recurrence_template_for_changed_payloads(
    tmp_path: Path,
) -> None:
    kernel = _kernel()
    base_pack = _pack(kernel)
    payloads = _payloads(kernel)
    catalog = _recurrence_catalog(base_pack, payloads)
    pack = replace(base_pack, recurrence_template=catalog.to_dict())
    changed_payloads = dict(payloads)
    changed_payloads["kernels/0/application.symjit"] = b"changed application"

    with pytest.raises(PreparedModelBundleError, match="prepared-pack digest"):
        write_prepared_model_bundle(
            tmp_path / "changed-payload",
            compiled_model=_compiled_model(),
            kernel_pack=pack,
            payloads=changed_payloads,
        )


def test_reader_rejects_rehashed_payload_outside_recurrence_identity(
    tmp_path: Path,
) -> None:
    kernel = _kernel()
    base_pack = _pack(kernel)
    payloads = _payloads(kernel)
    catalog = _recurrence_catalog(base_pack, payloads)
    pack = replace(base_pack, recurrence_template=catalog.to_dict())
    path = write_prepared_model_bundle(
        tmp_path / "rehashed-payload",
        compiled_model=_compiled_model(),
        kernel_pack=pack,
        payloads=payloads,
    )

    payload_path = "kernels/0/application.symjit"
    changed_payload = b"changed and rehashed application"
    entries = _entries(path)
    updated: list[tuple[zipfile.ZipInfo, bytes]] = []
    for info, data in entries:
        if info.filename == payload_path:
            data = changed_payload
        elif info.filename == PREPARED_MODEL_MANIFEST_PATH:
            manifest = json.loads(data)
            assert isinstance(manifest, dict)
            members = manifest["members"]
            assert isinstance(members, list)
            record = next(item for item in members if item["path"] == payload_path)
            record["size"] = len(changed_payload)
            record["sha256"] = hashlib.sha256(changed_payload).hexdigest()
            data = (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("ascii")
        updated.append((info, data))
    _rewrite(path, updated)

    with pytest.raises(PreparedModelBundleError, match="prepared-pack digest"):
        load_prepared_model_bundle(path)


def test_prepared_block_variant_round_trip_and_payload_index(tmp_path: Path) -> None:
    kernel = _kernel(signature="a" * 64)
    variant = _block_variant(kernel)
    pack = _pack(kernel, kernel_variants=(variant,))
    path = write_prepared_model_bundle(
        tmp_path / "block",
        compiled_model=_compiled_model(),
        kernel_pack=pack,
        payloads={**_payloads(kernel), **_variant_payloads(variant)},
    )

    loaded = load_prepared_model_bundle(path)

    assert loaded.kernel_pack.kernel_variants == (variant,)
    assert variant.input_lane_stride == kernel.input_arity
    assert variant.output_lane_stride == kernel.output_arity
    assert set(variant.referenced_payload_paths).issubset(
        loaded.kernel_pack.referenced_payload_paths
    )
    assert loaded.read_payload(
        "kernels/0/variants/independent-block-4/application.symjit"
    ).startswith(b"variant:0:")


def test_reader_rejects_private_fork_prepared_variant_with_regeneration_message(
    tmp_path: Path,
) -> None:
    kernel = _kernel(signature="a" * 64)
    variant = _block_variant(kernel)
    path = write_prepared_model_bundle(
        tmp_path / "old-variant",
        compiled_model=_compiled_model(),
        kernel_pack=_pack(kernel, kernel_variants=(variant,)),
        payloads={**_payloads(kernel), **_variant_payloads(variant)},
    )

    def restore_old_variant_abi(manifest: dict[str, object]) -> None:
        pack = manifest["kernel_pack"]
        assert isinstance(pack, dict)
        variants = pack["kernel_variants"]
        assert isinstance(variants, list)
        variant_manifest = variants[0]
        assert isinstance(variant_manifest, dict)
        variant_manifest["variant_abi"] = "pyamplicol-prepared-kernel-" + "variant-v1"

    _mutate_manifest(path, restore_old_variant_abi)

    with pytest.raises(PreparedModelBundleError, match="regenerate"):
        load_prepared_model_bundle(path)


def test_prepared_pack_identity_rejects_predecessor_abi() -> None:
    with pytest.raises(PreparedModelBundleError, match="regenerate"):
        PreparedKernelPackIdentity(
            contract_digest="a" * 64,
            payload_digest="b" * 64,
            pack_digest="c" * 64,
            abi="pyamplicol-prepared-kernel-pack-" + "identity-v1",
        )


def test_prepared_bundle_manifest_identity_uses_the_published_abi_domain() -> None:
    manifest = {
        "kind": "pyamplicol-prepared-model",
        "schema_version": 1,
        "kernel_pack": {"backend": "jit", "kernels": []},
    }
    canonical = json.dumps(
        manifest,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    expected = hashlib.sha256(
        PREPARED_KERNEL_PACK_IDENTITY_ABI.encode("ascii")
        + b"\x00"
        + canonical
    ).hexdigest()

    assert prepared_kernel_pack_manifest_identity_sha256(manifest) == expected


def test_reader_rejects_pack_without_kernel_variants(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)

    def remove_variants(manifest: dict[str, object]) -> None:
        pack = manifest["kernel_pack"]
        assert isinstance(pack, dict)
        del pack["kernel_variants"]

    _mutate_manifest(path, remove_variants)

    with pytest.raises(PreparedModelBundleError, match="missing fields"):
        load_prepared_model_bundle(path)


def test_reader_rejects_kernel_without_proof_classes(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)

    def remove_proof_classes(manifest: dict[str, object]) -> None:
        pack = manifest["kernel_pack"]
        assert isinstance(pack, dict)
        kernels = pack["kernels"]
        assert isinstance(kernels, list)
        kernel = kernels[0]
        assert isinstance(kernel, dict)
        del kernel["proof_classes"]

    _mutate_manifest(path, remove_proof_classes)

    with pytest.raises(PreparedModelBundleError, match="missing fields"):
        load_prepared_model_bundle(path)


@pytest.mark.parametrize(
    ("field", "replacement_value", "pattern"),
    (
        ("base_kernel_id", 7, "unknown base kernel"),
        ("base_expression_digest", "0" * 64, "contract digest"),
        ("base_input_contract_digest", "0" * 64, "contract digest"),
        ("base_output_contract_digest", "0" * 64, "contract digest"),
        ("backend", "cpp", "backend does not match"),
        ("optimization_settings_digest", "0" * 64, "optimization digest"),
        ("input_layout", tuple(f"wrong:{index}" for index in range(8)), "layout"),
    ),
)
def test_pack_rejects_mismatched_block_variant_binding(
    field: str,
    replacement_value: object,
    pattern: str,
) -> None:
    kernel = _kernel(signature="a" * 64)
    variant = replace(_block_variant(kernel), **{field: replacement_value})

    with pytest.raises(PreparedModelBundleError, match=pattern):
        _pack(kernel, kernel_variants=(variant,))


def test_block_variant_rejects_evaluator_arity_mismatch() -> None:
    kernel = _kernel(signature="a" * 64)
    variant = _block_variant(kernel)
    manifest = dict(variant.f64_evaluator_manifest)
    manifest["output_len"] = 3

    with pytest.raises(PreparedModelBundleError, match="output_len"):
        replace(variant, f64_evaluator_manifest=manifest)


def test_payload_reference_index_is_reused_after_bundle_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _valid_bundle(tmp_path)
    bundle = load_prepared_model_bundle(path)
    expected = (
        "kernels/0/application.symjit",
        "kernels/0/exact.evaluator.bin",
    )

    def fail_manifest_traversal(*_args: object, **_kwargs: object) -> tuple[str, ...]:
        raise AssertionError("manifest payload paths were recomputed")

    monkeypatch.setattr(
        prepared_module,
        "_collect_manifest_paths",
        fail_manifest_traversal,
    )
    assert bundle.kernel_pack.kernels[0].referenced_payload_paths == expected
    assert bundle.kernel_pack.kernels[0].referenced_payload_paths == expected

    def fail_kernel_scan(_kernel: PreparedKernelRecord) -> tuple[str, ...]:
        raise AssertionError("kernel payload paths were rescanned")

    monkeypatch.setattr(
        PreparedKernelRecord,
        "referenced_payload_paths",
        property(fail_kernel_scan),
    )
    assert bundle.kernel_pack.referenced_payload_paths == expected
    assert bundle.kernel_pack.referenced_payload_paths == expected
    assert bundle.read_payload("kernels/0/application.symjit") == b"jit:0"
    assert bundle.read_payload("kernels/0/application.symjit") == b"jit:0"


def test_cached_payload_index_keeps_path_and_checksum_validation(
    tmp_path: Path,
) -> None:
    path = _valid_bundle(tmp_path)
    bundle = load_prepared_model_bundle(path)

    with pytest.raises(PreparedModelBundleError, match="normalized relative POSIX"):
        bundle.read_payload("../kernels/0/application.symjit")

    entries = [
        (info, b"bad!!" if info.filename.endswith("application.symjit") else data)
        for info, data in _entries(path)
    ]
    _rewrite(path, entries)

    with pytest.raises(PreparedModelBundleError, match="SHA-256 mismatch"):
        bundle.read_payload("kernels/0/application.symjit")


def test_prepared_model_bundle_bytes_are_deterministic(tmp_path: Path) -> None:
    first_kernel = _kernel(5, signature="closure:q-qbar:v1")
    second_kernel = _kernel(2, signature="vertex:g-g-g:v1")
    pack = _pack(first_kernel, second_kernel)
    payloads = _payloads(first_kernel, second_kernel)

    first = write_prepared_model_bundle(
        tmp_path / "first.pyamplicol-model",
        compiled_model=_compiled_model(),
        kernel_pack=pack,
        payloads=payloads,
    )
    second = write_prepared_model_bundle(
        tmp_path / "second.pyamplicol-model",
        compiled_model=dict(reversed(tuple(_compiled_model().items()))),
        kernel_pack=pack,
        payloads=dict(reversed(tuple(payloads.items()))),
    )

    assert first.read_bytes() == second.read_bytes()
    assert tuple(kernel.kernel_id for kernel in pack.kernels) == (2, 5)


@pytest.mark.parametrize(
    ("payload_path", "pattern"),
    (
        ("../escape.bin", "normalized relative POSIX"),
        ("/absolute.bin", "normalized relative POSIX"),
        ("nested\\windows.bin", "normalized relative POSIX"),
    ),
)
def test_writer_rejects_unsafe_payload_references(
    payload_path: str,
    pattern: str,
) -> None:
    with pytest.raises(PreparedModelBundleError, match=pattern):
        PreparedKernelRecord(
            kernel_id=0,
            contract_kind="closure",
            canonical_signature="closure:test",
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
            exact_evaluator_state_path=payload_path,
            f64_evaluator_manifest={
                "kind": "test",
                "application_path": "kernels/application.bin",
            },
        )


def test_writer_requires_exact_referenced_payload_set(tmp_path: Path) -> None:
    kernel = _kernel()
    payloads = _payloads(kernel)
    del payloads["kernels/0/application.symjit"]
    payloads["kernels/0/unreferenced.bin"] = b"extra"

    with pytest.raises(
        PreparedModelBundleError,
        match=r"missing: kernels/0/application\.symjit.*unreferenced",
    ):
        write_prepared_model_bundle(
            tmp_path / "invalid",
            compiled_model=_compiled_model(),
            kernel_pack=_pack(kernel),
            payloads=payloads,
        )


def test_writer_rejects_symlink_payload(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    link = tmp_path / "payload-link.bin"
    link.symlink_to(source)
    kernel = _kernel()
    payloads: dict[str, bytes | Path] = _payloads(kernel)
    payloads[kernel.exact_evaluator_state_path] = link

    with pytest.raises(PreparedModelBundleError, match="must not be a symlink"):
        write_prepared_model_bundle(
            tmp_path / "invalid",
            compiled_model=_compiled_model(),
            kernel_pack=_pack(kernel),
            payloads=payloads,
        )


def test_reader_rejects_traversal_archive_member(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)
    entries = _entries(path)
    info = zipfile.ZipInfo("../escape.bin")
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    entries.append((info, b"escape"))
    _rewrite(path, entries)

    with pytest.raises(PreparedModelBundleError, match="normalized relative POSIX"):
        load_prepared_model_bundle(path)


def test_reader_rejects_duplicate_archive_member(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)
    entries = _entries(path)
    duplicate = next(
        entry
        for entry in entries
        if entry[0].filename == "kernels/0/application.symjit"
    )
    entries.append(duplicate)
    with pytest.warns(UserWarning, match="Duplicate name"):
        _rewrite(path, entries)

    with pytest.raises(PreparedModelBundleError, match="duplicate member"):
        load_prepared_model_bundle(path)


def test_reader_rejects_symlink_archive_member(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)
    entries = _entries(path)
    updated: list[tuple[zipfile.ZipInfo, bytes]] = []
    for info, data in entries:
        if info.filename == "kernels/0/application.symjit":
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        updated.append((info, data))
    _rewrite(path, updated)

    with pytest.raises(PreparedModelBundleError, match="regular file"):
        load_prepared_model_bundle(path)


def test_reader_rejects_missing_manifest_member(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)
    entries = [
        entry
        for entry in _entries(path)
        if entry[0].filename != "kernels/0/application.symjit"
    ]
    _rewrite(path, entries)

    with pytest.raises(
        PreparedModelBundleError,
        match=r"members do not match.*missing",
    ):
        load_prepared_model_bundle(path)


def test_reader_rejects_payload_hash_mismatch(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)
    entries = [
        (info, b"other" if info.filename.endswith("application.symjit") else data)
        for info, data in _entries(path)
    ]
    _rewrite(path, entries)

    with pytest.raises(PreparedModelBundleError, match="SHA-256 mismatch"):
        load_prepared_model_bundle(path)


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    (
        ("kind", "not-pyamplicol", "bundle kind"),
        ("schema_version", 99, "bundle schema"),
        ("eager_kernel_abi", "eager-v99", "kernel ABI"),
    ),
)
def test_reader_rejects_bad_bundle_contract(
    tmp_path: Path,
    field: str,
    value: object,
    pattern: str,
) -> None:
    path = _valid_bundle(tmp_path)
    _mutate_manifest(path, lambda manifest: manifest.__setitem__(field, value))

    with pytest.raises(PreparedModelBundleError, match=pattern):
        load_prepared_model_bundle(path)


def test_reader_rejects_bad_backend(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)

    def mutate(manifest: dict[str, object]) -> None:
        pack = manifest["kernel_pack"]
        assert isinstance(pack, dict)
        pack["backend"] = "cuda"

    _mutate_manifest(path, mutate)
    with pytest.raises(PreparedModelBundleError, match="unsupported prepared backend"):
        load_prepared_model_bundle(path)


def test_exactly_one_pack_and_unique_kernel_contracts(tmp_path: Path) -> None:
    first = _kernel(1, signature="same")
    with pytest.raises(PreparedModelBundleError, match="kernel IDs must be unique"):
        _pack(first, _kernel(1, signature="other"))
    with pytest.raises(PreparedModelBundleError, match="signatures must be unique"):
        _pack(first, _kernel(2, signature="same"))

    path = _valid_bundle(tmp_path)

    def replace_singular_pack(manifest: dict[str, object]) -> None:
        pack = manifest.pop("kernel_pack")
        manifest["kernel_packs"] = [pack, pack]

    _mutate_manifest(path, replace_singular_pack)
    with pytest.raises(PreparedModelBundleError, match="missing fields: kernel_pack"):
        load_prepared_model_bundle(path)


def test_reader_rejects_unreferenced_payload_member(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)
    entries = _entries(path)
    manifest_index = next(
        index
        for index, (info, _) in enumerate(entries)
        if info.filename == PREPARED_MODEL_MANIFEST_PATH
    )
    manifest = json.loads(entries[manifest_index][1])
    assert isinstance(manifest, dict)
    members = manifest["members"]
    assert isinstance(members, list)
    members.append(
        {
            "path": "kernels/unreferenced.bin",
            "size": 5,
            "sha256": hashlib.sha256(b"extra").hexdigest(),
        }
    )
    entries[manifest_index] = (
        entries[manifest_index][0],
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        ),
    )
    extra = zipfile.ZipInfo("kernels/unreferenced.bin")
    extra.create_system = 3
    extra.external_attr = (stat.S_IFREG | 0o644) << 16
    entries.append((extra, b"extra"))
    _rewrite(path, entries)

    with pytest.raises(PreparedModelBundleError, match="unreferenced"):
        load_prepared_model_bundle(path)


def test_manifest_payload_mapping_is_deeply_immutable() -> None:
    kernel = _kernel()
    manifest = kernel.f64_evaluator_manifest
    assert isinstance(manifest, Mapping)
    with pytest.raises(TypeError):
        manifest["kind"] = "changed"  # type: ignore[index]
