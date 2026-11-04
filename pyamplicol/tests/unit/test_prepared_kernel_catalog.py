# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections import Counter
from copy import copy
from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol.models import (
    BuiltinSMModel,
    CompiledUFOModel,
    compile_model_source,
)
from pyamplicol.models.prepared_catalog import (
    PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF,
    PREPARED_INDEPENDENT_BLOCK_PROOF,
    PreparedKernelCatalogError,
    PropagatorKernelKey,
    build_prepared_kernel_catalog,
)
from pyamplicol.models.prepared_catalog_helpers import (
    proves_homogeneous_complex_linearity,
)
from pyamplicol.models.prepared_compile import _independent_block_contract
from pyamplicol.models.recurrence_catalog_builder import (
    build_recurrence_template_catalog,
)
from pyamplicol.models.recurrence_template import ExactComplexRationalV1

MODEL_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)


def test_homogeneous_current_linearity_proof_fails_closed() -> None:
    from symbolica import E

    current = (E("proof_current_0"), E("proof_current_1"))
    momentum = E("proof_momentum")

    assert proves_homogeneous_complex_linearity(
        (current[0] * momentum + 2 * current[1],), current
    )
    assert not proves_homogeneous_complex_linearity(
        (current[0] * momentum + 2 * current[1] + 3,), current
    )
    assert not proves_homogeneous_complex_linearity((current[0] * current[1],), current)
    assert not proves_homogeneous_complex_linearity(
        (E("conj(proof_current_0)"),), current
    )


@pytest.fixture(scope="module")
def builtin_catalog():
    model = BuiltinSMModel()
    return model, build_prepared_kernel_catalog(model)


@pytest.fixture(scope="module")
def external_sm_catalog():
    compiled = compile_model_source(
        MODEL_ROOT / "sm.json",
        restriction=str((MODEL_ROOT / "restrict_default.json").resolve()),
        use_cache=True,
    )
    model = CompiledUFOModel(compiled)
    return compiled, model, build_prepared_kernel_catalog(model)


def test_builtin_catalog_is_deterministic_and_process_independent(
    builtin_catalog,
) -> None:
    model, expected = builtin_catalog
    reordered = copy(model)
    reordered.vertices = tuple(reversed(model.vertices))

    assert build_prepared_kernel_catalog(model) == expected
    assert build_prepared_kernel_catalog(reordered) == expected
    assert (
        build_prepared_kernel_catalog(reordered).resolver_manifest()
        == expected.resolver_manifest()
    )


def test_every_builtin_vertex_kind_has_a_constructive_binding(
    builtin_catalog,
) -> None:
    model, catalog = builtin_catalog

    assert (
        {binding.key.kind for binding in catalog.vertex_bindings}
        == {vertex.kind for vertex in model.vertices}
        == set(range(25))
    )
    assert not [
        gap for gap in catalog.unsupported_variants if gap.contract_kind == "vertex"
    ]


def test_builtin_chirality_dimensions_and_propagator_coverage(
    builtin_catalog,
) -> None:
    model, catalog = builtin_catalog
    by_id = catalog.by_id

    for binding in catalog.vertex_bindings:
        assert binding.left_state.dimension == model.current_dimension(
            binding.key.particles[0],
            binding.key.left_chirality,
        )
        assert binding.right_state.dimension == model.current_dimension(
            binding.key.particles[1],
            binding.key.right_chirality,
        )
        assert binding.result_state.dimension == model.current_dimension(
            binding.key.particles[2],
            binding.key.result_chirality,
        )
        assert by_id[binding.kernel_id].output_dimension == (
            binding.result_state.dimension
        )

    expected: set[PropagatorKernelKey] = set()
    for particle_id in sorted(
        {
            *(
                particle_id
                for particle_id in model.particles
                if model.source_wavefunction_kind(particle_id) != "ghost"
            ),
            *(
                model.anti_particle(particle_id)
                for particle_id in model.particles
                if model.source_wavefunction_kind(particle_id) != "ghost"
            ),
        }
    ):
        chiralities = (
            (0,)
            if model.auxiliary_kind(particle_id) is not None
            or model.particle(particle_id).spin < 0
            else tuple(
                sorted(
                    {state.chirality for state in model.source_spin_states(particle_id)}
                )
            )
        )
        expected.update(
            PropagatorKernelKey(particle_id, chirality) for chirality in chiralities
        )
    expected.update(
        PropagatorKernelKey(state.particle_id, state.chirality)
        for binding in catalog.vertex_bindings
        for state in (binding.left_state, binding.right_state, binding.result_state)
        if model.source_wavefunction_kind(state.particle_id) != "ghost"
    )
    assert {binding.key for binding in catalog.propagator_bindings} == expected
    assert all(
        (binding.kernel_id is not None) == binding.applies_propagator
        for binding in catalog.propagator_bindings
    )
    assert all(
        PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF
        in by_id[binding.kernel_id].proof_classes
        for binding in catalog.propagator_bindings
        if binding.kernel_id is not None
    )
    assert all(
        PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF not in kernel.proof_classes
        for kernel in catalog.kernels
        if kernel.contract_kind != "propagator"
    )
    full_bottom = next(
        binding
        for binding in catalog.propagator_bindings
        if binding.key == PropagatorKernelKey(5, 0)
    )
    assert full_bottom.applies_propagator
    assert full_bottom.propagator_kind == "dirac-fermion"
    assert full_bottom.mass_class == "massless"
    assert any(
        binding.key.particles == (5, 21, 5)
        and binding.key.left_chirality == 0
        and binding.key.right_chirality == 0
        and binding.key.result_chirality == 0
        for binding in catalog.vertex_bindings
    )


def test_direct_contractions_remain_native_while_vertex_closures_are_catalogued(
    builtin_catalog,
) -> None:
    _model, catalog = builtin_catalog
    kinds = Counter(kernel.contract_kind for kernel in catalog.kernels)

    assert "direct-contraction" not in kinds
    assert kinds["closure"] > 0
    assert catalog.closure_bindings
    assert all(binding.projection == "scalar" for binding in catalog.closure_bindings)


def test_builtin_catalog_factors_and_deduplicates_linear_coupling_outputs(
    builtin_catalog,
) -> None:
    _model, catalog = builtin_catalog
    bindings_by_kernel: dict[int, set[str]] = {}
    factored_bindings = 0
    for binding in catalog.vertex_bindings:
        bindings_by_kernel.setdefault(binding.kernel_id, set()).add(
            binding.output_factor_source
        )
        if binding.output_factor_source == "none":
            continue
        factored_bindings += 1
        assert not {
            descriptor.role for descriptor in catalog.by_id[binding.kernel_id].inputs
        } & {"coupling-real", "coupling-imag"}

    assert factored_bindings > 0
    assert any(
        "none" in sources and bool(sources & {"coupling-real", "coupling-imag"})
        for sources in bindings_by_kernel.values()
    )
    assert all(
        binding.output_factor_source == "coupling-real"
        for binding in catalog.closure_bindings
    )


def test_catalog_exact_contracts_are_finite_namespaced_and_nonempty(
    builtin_catalog,
) -> None:
    _model, catalog = builtin_catalog
    forbidden = ("indeterminate", "nan", "infinity", "complexinf")

    for kernel in catalog.kernels:
        assert kernel.exact_expressions
        assert kernel.output_dimension == len(kernel.output_layout)
        assert all(
            expression and not any(marker in expression.lower() for marker in forbidden)
            for expression in kernel.exact_expressions
        )
        assert all(
            descriptor.symbol.startswith("pyamplicol::") for descriptor in kernel.inputs
        )


def test_every_certified_builtin_block_reconstructs_four_scalar_calls(
    builtin_catalog,
) -> None:
    from symbolica import E, Replacement

    _model, catalog = builtin_catalog
    certified = tuple(
        kernel
        for kernel in catalog.kernels
        if PREPARED_INDEPENDENT_BLOCK_PROOF in kernel.proof_classes
    )
    assert certified
    for kernel in certified:
        assert kernel.contract_kind == "vertex"
        assert {item.role for item in kernel.inputs}.issubset(
            {"left-current", "right-current"}
        )
        contract = _independent_block_contract(kernel)
        assert len(contract.parameters) == 4 * kernel.input_arity
        assert len(contract.outputs) == 4 * kernel.output_dimension
        scalar_inputs = tuple(E(item.symbol) for item in kernel.inputs)
        expected = tuple(
            E(item).to_canonical_string() for item in kernel.exact_expressions
        )
        for lane in range(4):
            lane_inputs = contract.parameters[
                lane * kernel.input_arity : (lane + 1) * kernel.input_arity
            ]
            reverse = tuple(
                Replacement(block_input, scalar_input)
                for block_input, scalar_input in zip(
                    lane_inputs, scalar_inputs, strict=True
                )
            )
            lane_outputs = contract.outputs[
                lane * kernel.output_dimension : (lane + 1) * kernel.output_dimension
            ]
            assert (
                tuple(
                    output.replace_multiple(reverse).to_canonical_string()
                    for output in lane_outputs
                )
                == expected
            )


def test_independent_block_proof_excludes_non_current_vertex_contracts(
    builtin_catalog,
) -> None:
    _model, catalog = builtin_catalog
    for kernel in catalog.kernels:
        if PREPARED_INDEPENDENT_BLOCK_PROOF in kernel.proof_classes:
            continue
        if kernel.contract_kind != "vertex":
            continue
        assert any(
            item.role not in {"left-current", "right-current"} for item in kernel.inputs
        )


def test_external_sm_active_oriented_kernels_and_parameters_are_catalogued(
    external_sm_catalog,
) -> None:
    _compiled, model, catalog = external_sm_catalog
    active_kinds = {vertex.kind for vertex in model.vertices}

    assert {binding.key.kind for binding in catalog.vertex_bindings} == active_kinds
    assert catalog.model_parameter_kernel_id is not None
    parameter_kernel = catalog.by_id[catalog.model_parameter_kernel_id]
    assert parameter_kernel.contract_kind == "model-parameter"
    assert parameter_kernel.output_dimension == len(
        model.runtime_derived_parameter_definitions()
    )
    assert not catalog.closure_bindings
    assert not catalog.unsupported_variants
    assert all(
        PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF in kernel.proof_classes
        for kernel in catalog.kernels
        if kernel.contract_kind == "propagator"
    )


def test_builtin_and_external_runtime_helicity_contracts_cover_antifermions(
    builtin_catalog,
    external_sm_catalog,
) -> None:
    builtin_model, builtin_prepared = builtin_catalog
    _compiled, external_model, external_prepared = external_sm_catalog

    for model, prepared in (
        (builtin_model, builtin_prepared),
        (external_model, external_prepared),
    ):
        semantic = build_recurrence_template_catalog(
            model,
            prepared,
            compiled_model_digest="a" * 64,
            prepared_kernel_pack_digest="b" * 64,
        )
        states = {state.template_id: state for state in semantic.current_states}
        contracts = {
            states[contract.full_state_template_id].particle_id: contract
            for contract in semantic.runtime_helicity_contracts
        }
        for particle_id in (1, -1):
            contract = contracts[particle_id]
            assert states[contract.full_state_template_id].chirality == 0
            variants = {
                states[variant.source_state_template_id].chirality: variant
                for variant in contract.variants
            }
            assert variants[-1].embedding_source_components == (
                None,
                None,
                0,
                1,
            )
            assert variants[-1].projection_full_components == (2, 3)
            assert variants[1].embedding_source_components == (
                0,
                1,
                None,
                None,
            )
            assert variants[1].projection_full_components == (0, 1)


def test_builtin_and_external_reflection_proofs_cover_selected_transitions(
    builtin_catalog,
    external_sm_catalog,
) -> None:
    builtin_model, builtin_prepared = builtin_catalog
    _compiled, external_model, external_prepared = external_sm_catalog

    for model, prepared in (
        (builtin_model, builtin_prepared),
        (external_model, external_prepared),
    ):
        semantic = build_recurrence_template_catalog(
            model,
            prepared,
            compiled_model_digest="a" * 64,
            prepared_kernel_pack_digest="b" * 64,
        )
        evaluators = {
            semantic_template_id: evaluator
            for evaluator in semantic.evaluator_bindings
            for semantic_template_id in evaluator.semantic_template_ids
        }
        colors = {color.template_id: color for color in semantic.color_contractions}
        expected_subjects = {
            transition.template_id
            for transition in semantic.transitions
            if colors[transition.color_contraction_template_id].input_representations
            == (8, 8)
            and colors[transition.color_contraction_template_id].output_representation
            == 8
            and {
                witness.input_permutation
                for witness in colors[
                    transition.color_contraction_template_id
                ].transition_witnesses
            }
            == {(0, 1), (1, 0)}
        }
        proofs = tuple(
            proof
            for proof in semantic.symmetry_proofs
            if proof.proof_algorithm == "canonical-current-word-reversal-v1"
        )

        assert expected_subjects
        assert {proof.subject_template_ids[0] for proof in proofs} == expected_subjects
        assert all(proof.input_permutation == (1, 0) for proof in proofs)
        assert {proof.exact_phase for proof in proofs} == {
            ExactComplexRationalV1.from_binary64(-1.0)
        }
        for proof in proofs:
            evaluator = evaluators[proof.subject_template_ids[0]]
            assert proof.expression_digests == evaluator.exact_expression_digests


def test_external_sm_catalog_signatures_ignore_compiled_inventory_order(
    external_sm_catalog,
) -> None:
    compiled, _model, expected = external_sm_catalog
    reordered_ir = replace(
        compiled.ir,
        oriented_kernels=tuple(reversed(compiled.ir.oriented_kernels)),
    )
    reordered = CompiledUFOModel(replace(compiled, ir=reordered_ir))
    actual = build_prepared_kernel_catalog(reordered)

    assert tuple(kernel.canonical_signature for kernel in actual.kernels) == tuple(
        kernel.canonical_signature for kernel in expected.kernels
    )
    assert actual.resolver_mappings() == expected.resolver_mappings()


def test_external_and_builtin_orientation_metadata_share_generic_contracts(
    builtin_catalog,
    external_sm_catalog,
) -> None:
    _builtin_model, builtin = builtin_catalog
    _compiled, _external_model, external = external_sm_catalog
    builtin_states = {
        (
            binding.left_state.orientation,
            binding.left_state.basis,
            binding.left_state.dimension,
        )
        for binding in builtin.vertex_bindings
    }
    external_states = {
        (
            binding.left_state.orientation,
            binding.left_state.basis,
            binding.left_state.dimension,
        )
        for binding in external.vertex_bindings
    }

    assert ("self-conjugate", "lorentz-vector", 4) in (builtin_states & external_states)
    assert any(
        state[1:] == ("weyl-chiral", 2) for state in builtin_states & external_states
    )
    assert any(
        binding.left_state.chirality == 0
        and binding.result_state.chirality == 0
        and binding.left_state.dimension == 4
        and binding.result_state.dimension == 4
        for binding in external.vertex_bindings
    )


def test_catalog_fails_closed_when_a_vertex_kind_has_no_exact_lowering() -> None:
    class BrokenBuiltin(BuiltinSMModel):
        def vertex_component_expression(self, kind, *args, **kwargs):
            if kind == 0:
                raise ValueError("deliberately unavailable")
            return super().vertex_component_expression(kind, *args, **kwargs)

    with pytest.raises(
        PreparedKernelCatalogError,
        match=r"cannot lower admitted vertex orientation kind=0",
    ):
        build_prepared_kernel_catalog(BrokenBuiltin())


def test_catalog_fails_closed_when_one_admitted_propagator_is_missing() -> None:
    class BrokenBuiltin(BuiltinSMModel):
        def _propagator_ir(self, particle_id, chirality=0):
            if particle_id == 21:
                raise ValueError("deliberately unavailable")
            return super()._propagator_ir(particle_id, chirality)

    with pytest.raises(
        PreparedKernelCatalogError,
        match=r"cannot lower admitted propagator particle=21, chirality=0",
    ):
        build_prepared_kernel_catalog(BrokenBuiltin())


def _rational_scalar_candidate(factor):
    from pyamplicol.models import compiler_symbolica
    from pyamplicol.models.base import Vertex, VertexEvaluationEquivalence
    from pyamplicol.models.prepared_catalog_builder import _vertex_candidate
    from pyamplicol.models.prepared_catalog_helpers import formal_components

    class RationalScalarModel(BuiltinSMModel):
        def vertex_evaluation_equivalence(self, kind):
            return VertexEvaluationEquivalence(
                "rational-scalar", factor=(factor.real, factor.imag)
            )

        def vertex_component_expression(self, kind, left, right, **kwargs):
            return (left[0] * right[0] / 3,)

    compiler_symbolica._ensure_symbolica()
    model = RationalScalarModel()
    result = _vertex_candidate(
        model,
        Vertex(0, (25, 25, 25), (1.0, 0.0)),
        left_chirality=0,
        right_chirality=0,
        result_chirality=0,
        parameter_indices={},
        contract_kind="vertex",
    )
    original = (
        formal_components(model, "left-current", 1)[0]
        * formal_components(model, "right-current", 1)[0]
        / 3
    )
    return result, original


@pytest.mark.parametrize("factor", (1 + 0j, -1 + 0j, 1j, 0.1 + 0j, 0.2j))
def test_prepared_normalization_preserves_exact_rational_coefficients(factor):
    from fractions import Fraction

    from symbolica import E

    candidate, original = _rational_scalar_candidate(factor)
    # Preserve the stored binary64 value, not a nearby simple fraction.
    exact_factor = E(f"({Fraction(factor.real)})+1i*({Fraction(factor.imag)})")
    normalized = E(candidate.candidate.exact_expressions[0])
    assert (normalized * exact_factor - original).expand() == 0


@pytest.mark.parametrize("coefficient", (1 + 0j, -1 + 0j, 1j, 0.1 + 0j, 0.2j))
def test_prepared_closure_preserves_exact_rational_coefficients(coefficient):
    from fractions import Fraction

    from symbolica import E

    from pyamplicol.models._physics_ir import ContractionIR
    from pyamplicol.models.prepared_catalog_builder import _closure_candidate

    candidate, original = _rational_scalar_candidate(1 + 0j)
    closure = _closure_candidate(
        candidate,
        ContractionIR(
            "rational-scalar",
            "scalar",
            "scalar",
            ((coefficient.real, coefficient.imag),),
        ),
    )
    exact_coefficient = E(
        f"({Fraction(coefficient.real)})+1i*({Fraction(coefficient.imag)})"
    )
    projected = E(closure.candidate.exact_expressions[0])
    assert (projected - exact_coefficient * original).expand() == 0
