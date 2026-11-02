# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pyamplicol.config import EvaluatorConfig, JITConfig
from pyamplicol.generation.recurrence_template_columnar import (
    RECURRENCE_TEMPLATE_INPUT_ABI,
    RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION,
)
from pyamplicol.models.loading import compile_model_source
from pyamplicol.models.prepared import PreparedKernelRecord, PreparedModelBundleError
from pyamplicol.models.prepared_catalog import (
    PREPARED_INDEPENDENT_BLOCK_PROOF,
    PreparedKernelCatalog,
    PreparedKernelInput,
    PreparedKernelSpec,
)
from pyamplicol.models.prepared_compile import (
    prepare_model_bundle,
    prepared_symbolica_settings,
)
from pyamplicol.models.recurrence_direct_template import (
    NATIVE_DIRECT_APPLICATION_ABI,
    PreparedNativeDirectCallableSpecV1,
    native_direct_entry_point,
)
from pyamplicol.models.recurrence_template import (
    CurrentStateTemplateV1,
    EvaluatorBindingV1,
    PropagatorTemplateV1,
    RecurrenceTemplateCatalog,
)

_NATIVE_BUILD_INPUTS_SHA256 = "a" * 64


class _FakeJitAdapter:
    def __init__(self, input_len: int = 1, output_len: int = 1) -> None:
        self.input_len = input_len
        self.output_len = output_len

    def artifact_manifest(self, artifact_dir: Path) -> dict[str, object]:
        payload_dir = artifact_dir / "evaluators"
        payload_dir.mkdir(parents=True, exist_ok=True)
        application = payload_dir / "random-application.symjit"
        plane_application = payload_dir / "random-application.plane.symjit"
        state = payload_dir / "random-state.evaluator.bin"
        application.write_bytes(b"portable-symjit-application")
        plane_application.write_bytes(b"portable-symjit-plane-application")
        state.write_bytes(b"exact-symbolica-state")
        return {
            "kind": "symjit-application-evaluator",
            "runtime_capability": "symjit.application.complex-f64.v1",
            "backend": "jit",
            "label": "prepared_test",
            "input_len": self.input_len,
            "output_len": self.output_len,
            "application_path": str(application.relative_to(artifact_dir)),
            "application_abi": "symjit-application-storage-v3",
            "plane_application": {
                "application_path": str(
                    plane_application.relative_to(artifact_dir)
                ),
                "application_abi": "pyamplicol-symjit-plane-application-v2",
                "storage_abi": "symjit-application-storage-v3",
                "element_layout": "split-complex-plane-major",
                "descriptor_order": "inputs-re-im-then-outputs-re-im",
                "input_complex_count": self.input_len,
                "output_complex_count": self.output_len,
                "input_plane_count": 2 * self.input_len,
                "output_plane_count": 2 * self.output_len,
                "compiler_type": "native",
                "translation_mode": "symbolica-structured-instructions",
                "optimization_level": 2,
                "simd": True,
                "complex": True,
                "fast_math": True,
                "fast_complex": False,
                "compression": True,
                "threading": False,
                "direct_arena": True,
                "source_digest": hashlib.sha256(b"instructions").hexdigest(),
                "target": {
                    "word_bits": 64,
                    "endianness": "little",
                },
            },
            "element_layout": "complex-f64",
            "batch_layout": "row-major",
            "compiler_type": "native",
            "translation_mode": "indirect",
            "optimization_level": 2,
            "word_bits": 64,
            "endianness": "little",
            "required_defuns": [],
            "evaluator_state_path": str(state.relative_to(artifact_dir)),
            "evaluator_state_runtime_capability": (
                "symbolica.legacy-jit-container.complex-f64.v1"
            ),
            "settings": {"jit_optimization_level": 2},
            "build_timing": {},
        }


def _catalog() -> PreparedKernelCatalog:
    input_contract = PreparedKernelInput(
        role="current",
        component=0,
        symbol="pyamplicol::prepared_test_input",
    )
    return PreparedKernelCatalog(
        model_name="built-in-sm",
        kernels=(
            PreparedKernelSpec(
                kernel_id=0,
                contract_kind="propagator",
                canonical_signature="1" * 64,
                exact_expressions=("pyamplicol::prepared_test_input",),
                inputs=(input_contract,),
                output_layout=("scalar:c0",),
            ),
        ),
        vertex_bindings=(),
        propagator_bindings=(),
        closure_bindings=(),
        model_parameter_kernel_id=None,
    )


def test_prepared_jit_direct_source_reuses_authenticated_application() -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    application_path = "kernels/000000/application.symjit"
    application_digest = hashlib.sha256(b"portable-o2").hexdigest()
    record = PreparedKernelRecord(
        kernel_id=0,
        contract_kind="propagator",
        canonical_signature="1" * 64,
        input_arity=1,
        output_arity=1,
        input_layout=("current:0",),
        input_contracts=(
            {
                "role": "current",
                "component": 0,
                "symbol": "pyamplicol::prepared_test_input",
                "model_parameter_name": None,
                "model_parameter_index": None,
            },
        ),
        output_layout=("scalar:c0",),
        exact_expressions=("pyamplicol::prepared_test_input",),
        exact_evaluator_state_path="kernels/000000/exact.evaluator.bin",
        f64_evaluator_manifest={
            "plane_application": {
                "application_path": application_path,
                "application_abi": "pyamplicol-symjit-plane-application-v2",
            },
        },
    )

    source = prepared_compile._prepared_jit_direct_source(
        record,
        payload_identity_records={
            application_path: (11, application_digest),
            record.exact_evaluator_state_path: (
                7,
                hashlib.sha256(b"exact").hexdigest(),
            ),
        },
    )

    assert source.prepared_kernel_id == 0
    assert source.source_application_path == application_path
    assert source.source_application_sha256 == application_digest
    assert (
        source.source_application_abi
        == "pyamplicol-symjit-plane-application-v2"
    )
    assert source.output_arity == 1
    assert source.exact_expressions == record.exact_expressions
    assert json.loads(source.input_contracts[0])["role"] == "current"


def _native_direct_spec(
    kernel: PreparedKernelSpec,
) -> PreparedNativeDirectCallableSpecV1:
    return PreparedNativeDirectCallableSpecV1(
        prepared_kernel_id=kernel.kernel_id,
        role="finalization",
        native_entry_point=native_direct_entry_point(
            "finalization",
            kernel.kernel_id,
        ),
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
        parent_component_shapes=((1,), (4,)),
        destination_component_counts=(1, 4),
    )


def test_prepared_native_direct_source_reuses_authenticated_library() -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    library_path = "kernels/000000/libprepared-native"
    library_digest = hashlib.sha256(b"native-library").hexdigest()
    kernel = _catalog().kernels[0]
    spec = _native_direct_spec(kernel)
    record = PreparedKernelRecord(
        kernel_id=kernel.kernel_id,
        contract_kind=kernel.contract_kind,
        canonical_signature=kernel.canonical_signature,
        input_arity=kernel.input_arity,
        output_arity=kernel.output_dimension,
        input_layout=tuple(
            f"{item.role}:{item.component}" for item in kernel.inputs
        ),
        input_contracts=tuple(item.to_dict() for item in kernel.inputs),
        output_layout=kernel.output_layout,
        exact_expressions=kernel.exact_expressions,
        exact_evaluator_state_path="kernels/000000/exact.evaluator.bin",
        f64_evaluator_manifest={
            "library_path": library_path,
            "runtime_capability": "symbolica.compiled-cpp.complex-f64.v1",
        },
    )

    source = prepared_compile._prepared_native_direct_source(
        record,
        spec=spec,
        payload_identity_records={
            library_path: (17, library_digest),
            record.exact_evaluator_state_path: (
                7,
                hashlib.sha256(b"exact").hexdigest(),
            ),
        },
    )

    assert source.prepared_kernel_id == 0
    assert source.role == "finalization"
    assert source.native_entry_point == spec.native_entry_point
    assert source.source_application_path == library_path
    assert source.source_application_sha256 == library_digest
    assert (
        source.source_application_abi
        == "symbolica.compiled-cpp.complex-f64.v1"
    )


def test_native_split_real_header_exports_only_direct_abis() -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    kernel = _catalog().kernels[0]
    spec = _native_direct_spec(kernel)
    header = prepared_compile._native_split_real_custom_header(
        kernel,
        raw_function_name="prepared_split",
        direct_spec=spec,
        eager_direct_source=(
            'extern "C" int pyamplicol_eager_direct_table_k00000000_v1('
            "const void*) { return 0; }"
        ),
    )

    assert "prepared_complexf64_get_buffer_len" not in header
    assert "prepared_complexf64(" not in header
    assert "pyamplicol_eager_direct_table_k00000000_v1" in header
    assert spec.native_entry_point in header
    assert "DirectArenaView arena" in header
    assert "DirectMomentumView momenta" in header
    assert "DirectParameterView parameters" in header
    assert "DirectFactorView factors" in header
    assert "const DirectFinalizationRow* rows" in header
    assert "row.component_count == 1u" in header
    assert "row.component_count == 4u" in header
    assert "std::vector" not in header
    assert "malloc(" not in header
    assert "EagerKernelInput" not in header
    assert NATIVE_DIRECT_APPLICATION_ABI not in header
    assert "static_assert(sizeof(DirectArenaView) == 56u)" in header
    assert "static_assert(sizeof(DirectClosureRow) == 40u)" in header
    assert "PAC_DIRECT_MAX_SCRATCH_DOUBLES" in header
    assert "raw_buffer_len > PAC_DIRECT_MAX_SCRATCH_DOUBLES" in header
    asm_header = prepared_compile._native_split_real_custom_header(
        kernel,
        raw_function_name="prepared_split",
        direct_spec=spec,
        raw_parameters_const=True,
    )
    assert (
        'extern "C" void prepared_split_realf64('
        "const double*, double*, double*);"
    ) in asm_header


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            {
                "target_triple": "x86_64-unknown-linux-gnu",
                "cpu_features": ["avx2"],
            },
            4,
        ),
        (
            {
                "target_triple": "x86_64-unknown-linux-gnu",
                "cpu_features": [],
            },
            2,
        ),
        (
            {
                "target_triple": "aarch64-apple-darwin",
                "cpu_features": ["neon"],
            },
            2,
        ),
    ],
)
def test_native_eager_simd_width_follows_portable_target(
    target: dict[str, object],
    expected: int,
) -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    assert prepared_compile._native_eager_simd_lane_width(target) == expected


def test_native_direct_header_reads_binding_coupling_from_context() -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    inputs = (
        PreparedKernelInput(
            role="coupling-real",
            component=0,
            symbol="prepared::coupling_re",
        ),
        PreparedKernelInput(
            role="coupling-imag",
            component=0,
            symbol="prepared::coupling_im",
        ),
    )
    kernel = PreparedKernelSpec(
        kernel_id=17,
        contract_kind="vertex",
        canonical_signature="7" * 64,
        exact_expressions=("prepared::coupling_re+prepared::coupling_im",),
        inputs=inputs,
        output_layout=("scalar:c0",),
    )
    spec = PreparedNativeDirectCallableSpecV1(
        prepared_kernel_id=kernel.kernel_id,
        role="contribution",
        native_entry_point=native_direct_entry_point(
            "contribution", kernel.kernel_id
        ),
        input_contracts=tuple(
            json.dumps(
                item.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            for item in inputs
        ),
        exact_expressions=kernel.exact_expressions,
        output_arity=1,
        parent_component_shapes=((1, 1),),
        destination_component_counts=(1,),
    )
    header = prepared_compile._native_split_real_custom_header(
        kernel,
        raw_function_name="prepared_split",
        direct_spec=spec,
    )

    assert "struct DirectNativeBindingContextV1" in header
    assert "if (context == nullptr) return 1;" in header
    assert "binding_context->coupling_re" in header
    assert "binding_context->coupling_im" in header


@pytest.mark.parametrize("backend", ["cpp", "asm"])
def test_native_direct_complex_contract_matches_exact_arithmetic(
    tmp_path: Path,
    backend: str,
) -> None:
    """Exercise coupling context, complex parameters, and row factors together."""

    import ctypes

    from symbolica import Expression

    import pyamplicol.models.prepared_compile as prepared_compile

    inputs = (
        PreparedKernelInput(
            role="current",
            component=0,
            symbol="test::current",
        ),
        PreparedKernelInput(
            role="coupling-real",
            component=0,
            symbol="test::coupling_re",
        ),
        PreparedKernelInput(
            role="coupling-imag",
            component=0,
            symbol="test::coupling_im",
        ),
        PreparedKernelInput(
            role="model-parameter",
            component=0,
            symbol="test::model_parameter",
            model_parameter_name="model.complex_parameter",
            model_parameter_index=0,
        ),
    )
    expression = (
        "(test::coupling_re+sqrt(-1)*test::coupling_im)"
        "*test::model_parameter*test::current"
    )
    kernel = PreparedKernelSpec(
        kernel_id=73,
        contract_kind="vertex",
        canonical_signature="9" * 64,
        exact_expressions=(expression,),
        inputs=inputs,
        output_layout=("scalar:c0",),
    )
    spec = PreparedNativeDirectCallableSpecV1(
        prepared_kernel_id=kernel.kernel_id,
        role="contribution",
        native_entry_point=native_direct_entry_point(
            "contribution",
            kernel.kernel_id,
        ),
        input_contracts=tuple(
            json.dumps(
                item.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            for item in inputs
        ),
        exact_expressions=kernel.exact_expressions,
        output_arity=1,
        parent_component_shapes=((1, 1),),
        destination_component_counts=(1,),
    )
    settings = prepared_symbolica_settings(EvaluatorConfig(backend=backend))
    manifest = prepared_compile._compile_native_split_real_kernel(
        kernel,
        outputs=(Expression.parse(expression),),
        parameters=[Expression.parse(item.symbol) for item in inputs],
        real_parameters=(1, 2),
        settings=settings,
        staging=tmp_path,
        direct_spec=spec,
    )
    assert manifest["function_name"] == manifest["direct_table"]["function_name"]
    manifest_settings = manifest["settings"]
    assert isinstance(manifest_settings, dict)
    assert manifest_settings["compiler_flags"] == []
    assert manifest_settings["effective_compiler_flags"] == ["-std=c++17"]
    assert "_complexf64(" not in Path(manifest["source_path"]).read_text(
        encoding="utf-8"
    )

    class Arena(ctypes.Structure):
        _fields_ = [
            ("current_re", ctypes.POINTER(ctypes.c_double)),
            ("current_im", ctypes.POINTER(ctypes.c_double)),
            ("current_scalar_len", ctypes.c_uint64),
            ("amplitude_re", ctypes.POINTER(ctypes.c_double)),
            ("amplitude_im", ctypes.POINTER(ctypes.c_double)),
            ("amplitude_scalar_len", ctypes.c_uint64),
            ("point_stride", ctypes.c_uint32),
        ]

    class Momentum(ctypes.Structure):
        _fields_ = [
            ("values", ctypes.POINTER(ctypes.c_double)),
            ("scalar_len", ctypes.c_uint64),
            ("form_count", ctypes.c_uint32),
            ("lorentz_component_count", ctypes.c_uint16),
            ("point_stride", ctypes.c_uint32),
        ]

    class Values(ctypes.Structure):
        _fields_ = [
            ("values_re", ctypes.POINTER(ctypes.c_double)),
            ("values_im", ctypes.POINTER(ctypes.c_double)),
            ("value_count", ctypes.c_uint32),
        ]

    class Context(ctypes.Structure):
        _fields_ = [
            ("coupling_re", ctypes.c_double),
            ("coupling_im", ctypes.c_double),
        ]

    class ContributionRow(ctypes.Structure):
        _fields_ = [
            ("parent0_component_base", ctypes.c_uint32),
            ("parent1_component_base_or_sentinel", ctypes.c_uint32),
            ("parent0_momentum_form_id", ctypes.c_uint32),
            ("parent1_momentum_form_id_or_sentinel", ctypes.c_uint32),
            ("destination_component_base", ctypes.c_uint32),
            ("exact_factor_id", ctypes.c_uint32),
            ("selector_domain_id", ctypes.c_uint32),
            ("flags", ctypes.c_uint32),
        ]

    current_re = (ctypes.c_double * 2)(17.0, 0.0)
    current_im = (ctypes.c_double * 2)(19.0, 0.0)
    amplitudes_re = (ctypes.c_double * 1)(0.0)
    amplitudes_im = (ctypes.c_double * 1)(0.0)
    momentum_values = (ctypes.c_double * 1)(0.0)
    parameter_re = (ctypes.c_double * 1)(5.0)
    parameter_im = (ctypes.c_double * 1)(7.0)
    factor_re = (ctypes.c_double * 1)(11.0)
    factor_im = (ctypes.c_double * 1)(13.0)
    arena = Arena(
        current_re,
        current_im,
        2,
        amplitudes_re,
        amplitudes_im,
        1,
        1,
    )
    momenta = Momentum(momentum_values, 1, 1, 4, 1)
    parameters = Values(parameter_re, parameter_im, 1)
    factors = Values(factor_re, factor_im, 1)
    context = Context(2.0, 3.0)
    row = ContributionRow(0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 1, 0, 0, 1)

    library = ctypes.CDLL(str(manifest["library_path"]))
    function = getattr(library, spec.native_entry_point)
    function.argtypes = (
        ctypes.c_void_p,
        Arena,
        Momentum,
        Values,
        Values,
        ctypes.POINTER(ContributionRow),
        ctypes.c_uint32,
        ctypes.c_uint32,
    )
    function.restype = ctypes.c_int
    status = function(
        ctypes.byref(context),
        arena,
        momenta,
        parameters,
        factors,
        ctypes.byref(row),
        1,
        1,
    )

    expected = (17 + 19j) * (2 + 3j) * (5 + 7j) * (11 + 13j)
    assert status == 0
    assert complex(current_re[1], current_im[1]) == pytest.approx(
        expected,
        rel=1.0e-14,
        abs=1.0e-14,
    )


def test_native_split_real_contract_keeps_certified_real_inputs_real() -> None:
    from symbolica import Expression

    import pyamplicol.models.prepared_compile as prepared_compile

    momentum = Expression.parse("test::p0")
    current = Expression.parse("test::current0")
    split_parameters, split_outputs = (
        prepared_compile._split_complex_kernel_contract(
            17,
            outputs=(current / (momentum**2 + 1),),
            parameters=[momentum, current],
            real_parameters=(0,),
        )
    )

    assert len(split_parameters) == 4
    momentum_imaginary = split_parameters[1].to_canonical_string()
    assert all(
        momentum_imaginary
        not in expression.to_canonical_string()
        for expression in split_outputs
    )
    assert split_outputs[0].is_real()
    assert split_outputs[1].is_real()


def test_native_split_real_contract_supports_only_certified_real_roots() -> None:
    from symbolica import Expression

    import pyamplicol.models.prepared_compile as prepared_compile

    real_parameter = Expression.parse("test::real_parameter")
    complex_parameter = Expression.parse("test::complex_parameter")
    split_parameters, split_outputs = (
        prepared_compile._split_complex_kernel_contract(
            19,
            outputs=(
                Expression.parse(
                    "test::real_parameter^(1/2)"
                    "+test::real_parameter^(-1/2)"
                    "+test::real_parameter^(3/2)"
                ),
            ),
            parameters=[real_parameter],
            real_parameters=(0,),
        )
    )

    admitted_symbols = {
        symbol.to_atom_tree().head for symbol in split_parameters
    }
    assert prepared_compile._is_structurally_real_expression(
        split_outputs[0],
        admitted_symbols=admitted_symbols,
    )
    assert split_outputs[1].to_canonical_string() == "0"
    with pytest.raises(PreparedModelBundleError, match="complex expression"):
        prepared_compile._split_complex_kernel_contract(
            20,
            outputs=(Expression.parse("test::complex_parameter^(1/2)"),),
            parameters=[complex_parameter],
            real_parameters=(),
        )
    with pytest.raises(
        PreparedModelBundleError,
        match="unsupported real non-integer power",
    ):
        prepared_compile._split_complex_kernel_contract(
            21,
            outputs=(Expression.parse("test::real_parameter^(1/3)"),),
            parameters=[real_parameter],
            real_parameters=(0,),
        )


def test_native_real_parameter_indices_follow_catalog_domains() -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    kernel = PreparedKernelSpec(
        kernel_id=23,
        contract_kind="propagator",
        canonical_signature="8" * 64,
        exact_expressions=("test::p+test::mass+test::phase",),
        inputs=(
            PreparedKernelInput(
                role="momentum",
                component=0,
                symbol="test::p",
            ),
            PreparedKernelInput(
                role="model-parameter",
                component=0,
                symbol="test::mass",
                model_parameter_name="particle.23.mass",
                model_parameter_index=0,
            ),
            PreparedKernelInput(
                role="model-parameter",
                component=0,
                symbol="test::phase",
                model_parameter_name="model.phase",
                model_parameter_index=1,
            ),
        ),
        output_layout=("scalar:c0",),
    )

    assert prepared_compile._real_kernel_parameter_indices(
        kernel,
        real_model_parameter_names=frozenset({"particle.23.mass"}),
    ) == (0, 1)


def _block_catalog() -> PreparedKernelCatalog:
    return PreparedKernelCatalog(
        model_name="built-in-sm",
        kernels=(
            PreparedKernelSpec(
                kernel_id=0,
                contract_kind="vertex",
                canonical_signature="2" * 64,
                exact_expressions=(
                    "pyamplicol::prepared_block_left+pyamplicol::prepared_block_right",
                ),
                inputs=(
                    PreparedKernelInput(
                        role="left-current",
                        component=0,
                        symbol="pyamplicol::prepared_block_left",
                    ),
                    PreparedKernelInput(
                        role="right-current",
                        component=0,
                        symbol="pyamplicol::prepared_block_right",
                    ),
                ),
                output_layout=("scalar:c0",),
                proof_classes=(PREPARED_INDEPENDENT_BLOCK_PROOF,),
            ),
        ),
        vertex_bindings=(),
        propagator_bindings=(),
        closure_bindings=(),
        model_parameter_kernel_id=None,
    )


@pytest.mark.parametrize("compress", (True, False))
def test_prepared_symbolica_settings_forward_jit_compression(
    compress: bool,
) -> None:
    settings = prepared_symbolica_settings(
        EvaluatorConfig(jit=JITConfig(compress=compress))
    )

    assert settings.jit_compress is compress
    assert settings.to_json_dict()["jit_compress"] is compress


def _native_recurrence_validation(*_args, **_kwargs) -> dict[str, object]:
    return {
        "kind": "pyamplicol-recurrence-template-validation-result",
        "template_input_sha256": "d" * 64,
    }


def _recurrence_catalog(
    *,
    compiled_model_digest: str,
    prepared_kernel_pack_digest: str,
) -> RecurrenceTemplateCatalog:
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
    expression_digest = hashlib.sha256(b"pyamplicol::prepared_test_input").hexdigest()
    propagator = PropagatorTemplateV1(
        template_id="propagator:test-active",
        state_template_id=state.template_id,
        applies_propagator=True,
        evaluator_resolver_key="evaluator:propagator:test",
        numerator_expression_digest=expression_digest,
        denominator_expression_digest=expression_digest,
        mass_parameter_id=None,
        width_parameter_id=None,
        gauge=None,
        linearity_proof_template_id=None,
    )
    binding = EvaluatorBindingV1(
        resolver_key="evaluator:propagator:test",
        prepared_kernel_id=0,
        contract_kind="propagator",
        callable_signature="1" * 64,
        input_state_template_ids=(state.template_id,),
        output_state_template_id=state.template_id,
        input_layout=(
            json.dumps(
                {
                    "component": 0,
                    "model_parameter_index": None,
                    "model_parameter_name": None,
                    "role": "current",
                    "symbol": "pyamplicol::prepared_test_input",
                },
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
        output_layout=("scalar:c0",),
        exact_expression_digests=(expression_digest,),
        semantic_template_ids=(propagator.template_id,),
    )
    return RecurrenceTemplateCatalog.create(
        compiled_model_digest=compiled_model_digest,
        prepared_kernel_pack_digest=prepared_kernel_pack_digest,
        current_states=(state,),
        propagators=(propagator,),
        evaluator_bindings=(binding,),
    )


def _identity_recurrence_catalog(
    *,
    compiled_model_digest: str,
    prepared_kernel_pack_digest: str,
) -> RecurrenceTemplateCatalog:
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
    identity = PropagatorTemplateV1(
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
    return RecurrenceTemplateCatalog.create(
        compiled_model_digest=compiled_model_digest,
        prepared_kernel_pack_digest=prepared_kernel_pack_digest,
        current_states=(state,),
        propagators=(identity,),
    )


def _native_result(template_input, authenticated_ids) -> dict[str, object]:
    assert authenticated_ids == [0]
    return {
        "kind": "pyamplicol-recurrence-template-validation-result",
        "schema_version": 1,
        "validation_status": "validated",
        "template_input_abi": RECURRENCE_TEMPLATE_INPUT_ABI,
        "template_input_schema_version": RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION,
        "template_input_sha256": template_input.canonical_digest,
        "catalog_digest": template_input.catalog_digest,
        "compiled_model_digest": template_input.compiled_model_digest,
        "prepared_kernel_pack_digest": template_input.prepared_kernel_pack_digest,
        "prepared_kernel_inventory_verified": True,
        "prepared_kernel_inventory_count": 1,
        "counts": {
            "parameters": 0,
            "current_states": 0,
            "sources": 0,
            "quantum_flows": 0,
            "transitions": 0,
            "propagators": 0,
            "closures": 0,
            "color_contractions": 0,
            "symmetry_proofs": 0,
            "runtime_helicity_contracts": 0,
            "evaluator_bindings": 0,
            "prepared_kernels": 0,
            "referenced_prepared_kernels": 0,
        },
    }


def test_native_recurrence_validation_checks_complete_result_contract(
    monkeypatch,
) -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    catalog = RecurrenceTemplateCatalog.create(
        compiled_model_digest="a" * 64,
        prepared_kernel_pack_digest="b" * 64,
    )
    module = type("NativeModule", (), {})()
    module._validate_recurrence_template_input_v1 = _native_result
    monkeypatch.setattr(
        prepared_compile.importlib,
        "import_module",
        lambda _name: module,
    )

    result = prepared_compile._validate_native_recurrence_template_input_v1(
        catalog,
        (0,),
    )
    assert result["prepared_kernel_inventory_verified"] is True

    def wrong_result(template_input, authenticated_ids):
        result = _native_result(template_input, authenticated_ids)
        result["schema_version"] = True
        return result

    module._validate_recurrence_template_input_v1 = wrong_result
    with pytest.raises(PreparedModelBundleError, match="schema_version"):
        prepared_compile._validate_native_recurrence_template_input_v1(catalog, (0,))


def test_native_recurrence_validation_requires_matching_extension(
    monkeypatch,
) -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    catalog = RecurrenceTemplateCatalog.create(
        compiled_model_digest="a" * 64,
        prepared_kernel_pack_digest="b" * 64,
    )
    monkeypatch.setattr(
        prepared_compile.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )

    with pytest.raises(PreparedModelBundleError, match="matching installed"):
        prepared_compile._validate_native_recurrence_template_input_v1(catalog, ())


def test_recurrence_preflight_fails_before_backend_compilation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    compiled = compile_model_source("built-in-sm", use_cache=False)
    compiled_kernel = False

    def fail_recurrence(*_args, **_kwargs):
        raise ValueError("unsupported recurrence semantics")

    def observe_compile(*_args, **_kwargs):
        nonlocal compiled_kernel
        compiled_kernel = True
        raise AssertionError("backend compilation must not run")

    monkeypatch.setattr(
        prepared_compile,
        "build_prepared_kernel_catalog",
        lambda _: _catalog(),
    )
    monkeypatch.setattr(
        prepared_compile,
        "build_recurrence_template_catalog",
        fail_recurrence,
    )
    monkeypatch.setattr(prepared_compile, "_compile_kernel", observe_compile)

    with pytest.raises(ValueError, match="unsupported recurrence semantics"):
        prepare_model_bundle(
            compiled,
            tmp_path / "unsupported-recurrence",
            evaluator=EvaluatorConfig(),
        )
    assert compiled_kernel is False


def test_native_recurrence_preflight_fails_before_backend_compilation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    compiled = compile_model_source("built-in-sm", use_cache=False)
    compiled_kernel = False

    def observe_compile(*_args, **_kwargs):
        nonlocal compiled_kernel
        compiled_kernel = True
        raise AssertionError("backend compilation must not run")

    monkeypatch.setattr(
        prepared_compile,
        "build_prepared_kernel_catalog",
        lambda _: _catalog(),
    )
    monkeypatch.setattr(
        prepared_compile,
        "build_recurrence_template_catalog",
        lambda *_args, compiled_model_digest, prepared_kernel_pack_digest, **_kwargs: (
            _identity_recurrence_catalog(
                compiled_model_digest=compiled_model_digest,
                prepared_kernel_pack_digest=prepared_kernel_pack_digest,
            )
        ),
    )
    monkeypatch.setattr(
        prepared_compile,
        "_validate_native_recurrence_template_input_v1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("native recurrence unavailable")
        ),
    )
    monkeypatch.setattr(prepared_compile, "_compile_kernel", observe_compile)

    with pytest.raises(ValueError, match="native recurrence unavailable"):
        prepare_model_bundle(
            compiled,
            tmp_path / "missing-native-recurrence",
            evaluator=EvaluatorConfig(),
        )
    assert compiled_kernel is False


def test_prepared_compiler_writes_structured_architecture_kernel_pack(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    monkeypatch.setattr(
        prepared_compile,
        "build_prepared_kernel_catalog",
        lambda _: _catalog(),
    )
    monkeypatch.setattr(
        prepared_compile,
        "build_recurrence_template_catalog",
        lambda *_args, compiled_model_digest, prepared_kernel_pack_digest, **_kwargs: (
            _recurrence_catalog(
                compiled_model_digest=compiled_model_digest,
                prepared_kernel_pack_digest=prepared_kernel_pack_digest,
            )
        ),
    )
    monkeypatch.setattr(
        prepared_compile,
        "_compile_symbolica_outputs",
        lambda *_args, **_kwargs: _FakeJitAdapter(),
    )
    monkeypatch.setattr(
        prepared_compile,
        "_validate_native_recurrence_template_input_v1",
        _native_recurrence_validation,
    )
    monkeypatch.setattr(
        prepared_compile,
        "_native_build_inputs_sha256",
        lambda: _NATIVE_BUILD_INPUTS_SHA256,
    )
    compiled = compile_model_source("built-in-sm", use_cache=False)
    progress: list[tuple[str, int, int]] = []

    result = prepare_model_bundle(
        compiled,
        tmp_path / "prepared-built-in",
        evaluator=EvaluatorConfig(),
        progress=lambda label, completed, total: progress.append(
            (label, completed, total)
        ),
    )

    assert result.output.name.endswith(".pyamplicol-model")
    assert result.kernel_count == 1
    assert result.bundle.backend == "jit"
    assert (
        result.bundle.kernel_pack.producer["native_build_inputs_sha256"]
        == _NATIVE_BUILD_INPUTS_SHA256
    )
    assert result.bundle.kernel_pack.optimization_settings["jit_compress"] is False
    assert result.bundle.kernel_pack.target["portable"] is True
    assert (
        result.bundle.kernel_pack.target["target_triple"]
        == "symjit-storage-v3-portable"
    )
    assert result.bundle.kernel_pack.resolver_manifest["model_name"] == "built-in-sm"
    kernel = result.bundle.kernel_pack.kernels[0]
    assert kernel.input_contracts[0]["role"] == "current"
    assert kernel.exact_expressions == ("pyamplicol::prepared_test_input",)
    assert kernel.exact_evaluator_state_path.startswith("kernels/000000/")
    assert all(
        path.startswith("kernels/000000/") for path in kernel.referenced_payload_paths
    )
    direct_catalog = result.bundle.kernel_pack.recurrence_direct_template_catalog
    assert direct_catalog is not None
    assert direct_catalog.backend == "jit"
    assert direct_catalog.portable is True
    assert direct_catalog.optimization_level == 2
    assert direct_catalog.direct_executor_id_for("finalization", 0) == 0
    payload_binding = direct_catalog.templates[0].payload_binding
    assert payload_binding.kind == "prepared-direct-call"
    assert payload_binding.source_application_path in kernel.referenced_payload_paths
    assert (
        payload_binding.source_application_abi
        == "pyamplicol-symjit-plane-application-v2"
    )
    assert payload_binding.direct_application_abi == (
        "pyamplicol-symjit-plane-application-v2"
    )
    assert payload_binding.role == "finalization"
    assert payload_binding.destination_operation == "finalize-in-place"
    assert payload_binding.exact_factor_scalar_slots == (0, 1)
    assert payload_binding.prepared_template_semantic_digest is not None
    assert direct_catalog.executable
    assert result.bundle.direct_template_catalog_digest == direct_catalog.catalog_digest
    assert (
        result.bundle.manifest["direct_template_catalog_digest"]
        == direct_catalog.catalog_digest
    )
    assert progress[-1] == ("prepared model complete", 1, 1)
    assert "recurrence_template_validation" in result.phase_timings_seconds


def test_prepared_jit_compiler_forces_portable_o2_without_changing_public_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    observed_levels: list[int] = []

    def fake_compile(*_args, symbolica_settings, **_kwargs):
        observed_levels.append(symbolica_settings.jit_optimization_level)
        return _FakeJitAdapter()

    monkeypatch.setattr(
        prepared_compile,
        "build_prepared_kernel_catalog",
        lambda _: _catalog(),
    )
    monkeypatch.setattr(
        prepared_compile,
        "build_recurrence_template_catalog",
        lambda *_args, compiled_model_digest, prepared_kernel_pack_digest, **_kwargs: (
            _recurrence_catalog(
                compiled_model_digest=compiled_model_digest,
                prepared_kernel_pack_digest=prepared_kernel_pack_digest,
            )
        ),
    )
    monkeypatch.setattr(prepared_compile, "_compile_symbolica_outputs", fake_compile)
    monkeypatch.setattr(
        prepared_compile,
        "_validate_native_recurrence_template_input_v1",
        _native_recurrence_validation,
    )
    monkeypatch.setattr(
        prepared_compile,
        "_native_build_inputs_sha256",
        lambda: _NATIVE_BUILD_INPUTS_SHA256,
    )
    requested = EvaluatorConfig(jit=JITConfig(optimization_level=3))
    compiled = compile_model_source("built-in-sm", use_cache=False)

    with pytest.warns(UserWarning, match="optimization level 2"):
        result = prepare_model_bundle(
            compiled,
            tmp_path / "portable-o2",
            evaluator=requested,
        )

    assert requested.jit.optimization_level == 3
    assert observed_levels == [2]
    assert (
        result.bundle.kernel_pack.optimization_settings["jit_optimization_level"] == 2
    )
    assert (
        result.bundle.kernel_pack.kernels[0].f64_evaluator_manifest[
            "optimization_level"
        ]
        == 2
    )
    direct_catalog = result.bundle.kernel_pack.recurrence_direct_template_catalog
    assert direct_catalog is not None
    assert direct_catalog.optimization_level == 2
    assert direct_catalog.portable is True


def test_prepared_compiler_emits_independent_block4_jit_variant(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pyamplicol.models.prepared_compile as prepared_compile

    compiled_shapes: list[tuple[int, int]] = []

    def fake_compile(outputs, parameters, **_kwargs):
        compiled_shapes.append((len(parameters), len(outputs)))
        return _FakeJitAdapter(len(parameters), len(outputs))

    monkeypatch.setattr(
        prepared_compile,
        "build_prepared_kernel_catalog",
        lambda _: _block_catalog(),
    )
    monkeypatch.setattr(
        prepared_compile,
        "build_recurrence_template_catalog",
        lambda *_args, compiled_model_digest, prepared_kernel_pack_digest, **_kwargs: (
            _identity_recurrence_catalog(
                compiled_model_digest=compiled_model_digest,
                prepared_kernel_pack_digest=prepared_kernel_pack_digest,
            )
        ),
    )
    monkeypatch.setattr(
        prepared_compile,
        "_compile_symbolica_outputs",
        fake_compile,
    )
    monkeypatch.setattr(
        prepared_compile,
        "_validate_native_recurrence_template_input_v1",
        _native_recurrence_validation,
    )
    monkeypatch.setattr(
        prepared_compile,
        "_native_build_inputs_sha256",
        lambda: _NATIVE_BUILD_INPUTS_SHA256,
    )
    compiled = compile_model_source("built-in-sm", use_cache=False)

    result = prepare_model_bundle(
        compiled,
        tmp_path / "prepared-block",
        evaluator=EvaluatorConfig(),
    )

    assert compiled_shapes == [(2, 1), (8, 4)]
    (variant,) = result.bundle.kernel_pack.kernel_variants
    assert variant.variant_id == "independent-block-4"
    assert variant.input_lane_stride == 2
    assert variant.output_lane_stride == 1
    assert variant.input_layout == (
        "lane:0:left-current:0",
        "lane:0:right-current:0",
        "lane:1:left-current:0",
        "lane:1:right-current:0",
        "lane:2:left-current:0",
        "lane:2:right-current:0",
        "lane:3:left-current:0",
        "lane:3:right-current:0",
    )
    assert all(
        path.startswith("kernels/000000/variants/independent-block-4/")
        for path in variant.referenced_payload_paths
    )
