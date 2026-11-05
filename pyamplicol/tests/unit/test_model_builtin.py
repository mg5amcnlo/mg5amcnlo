# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math

import pytest

import pyamplicol.models as models
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.base import Model, Vertex
from pyamplicol.models.builtin.adapters import build_model_payload
from pyamplicol.models.builtin.model import BuiltinModel
from pyamplicol.models.loading import compile_model_source


def test_builtin_sm_normalization_uses_the_compiled_alpha_ew_default() -> None:
    model = BuiltinSMModel()
    _payload, parameter_defaults = build_model_payload()

    assert model.alpha_ew == 1.0 / 132.507
    assert parameter_defaults["alpha_ew"] == (model.alpha_ew, 0.0)
    assert (
        model.runtime_normalization_parameter_defaults()["normalization.alpha_ew"]
        == parameter_defaults["alpha_ew"][0]
    )


def test_model_builtin_sm_preserves_production_tables_and_couplings() -> None:
    model = BuiltinSMModel()

    assert len(model.particles) == 24
    assert len(model.vertices) == 211
    assert model.sin_weak == 0.47143025548407230
    assert model.mass(23) == 91.188
    assert model.width(23) == 2.441404
    assert model.mass(24) == 80.419002445756163
    left, right = model.z_fermion_coupling(1)
    prefactor = 1.0 / (model.sin_weak * model.cos_weak)
    assert math.isclose(left, prefactor * (-0.5 + model.sin_weak**2 / 3.0))
    assert math.isclose(right, prefactor * (model.sin_weak**2 / 3.0))
    assert model.leading_color_factor([1, -1, 23, 21, 21]) == 27
    assert model.runtime_normalization_parameter_defaults() == {
        "normalization.alpha_s_me_check": model.alpha_s_me_check,
        "normalization.alpha_ew": model.alpha_ew,
    }


def test_model_builtin_lowering_metadata_uses_owned_symbol_names() -> None:
    model = BuiltinSMModel()

    tensor = model.vertex_lowering_rule(1)
    assert tensor.tensor_names == ("pyamplicol::two_gluon_to_tensor",)
    assert tensor.expression_head == "pyamplicol::two_gluon_to_tensor"
    assert model.vertex_lowering_rule(6).expression_head == ("quark_gluon_weyl_current")
    assert model.vertex_lowering_rule(999).backend == "unimplemented"
    assert model.propagator_lowering_rule(-21).description.endswith(
        "adjacent built-in-SM vertex kernels"
    )
    assert model.propagator_lowering_rule(125).description.endswith(
        "the built-in-SM model"
    )
    full_bottom = model.propagator_lowering_rule(5, chirality=0)
    assert full_bottom.kind == "dirac-fermion"
    assert full_bottom.mass_class == "massless"
    assert full_bottom.kernel == "massless_dirac_fermion"


def test_builtin_yang_mills_kernel_evaluation_relations_are_exact() -> None:
    model = BuiltinSMModel()
    left = (1.0 + 2.0j, -0.5j, 3.0, -2.0 + 0.25j)
    right = (0.75, 1.5 - 0.5j, -1.0j, 2.25)
    left_momentum = (5.0, 1.0, -2.0, 0.5)
    right_momentum = (4.0, -0.5, 1.5, -1.0)

    three_vector = model.vertex_component_expression(
        0,
        left,
        right,
        result_particle_id=21,
        result_chirality=0,
        left_momentum=left_momentum,
        right_momentum=right_momentum,
    )
    three_vector_swapped = model.vertex_component_expression(
        0,
        right,
        left,
        result_particle_id=21,
        result_chirality=0,
        left_momentum=right_momentum,
        right_momentum=left_momentum,
    )
    assert three_vector_swapped == pytest.approx(
        tuple(-value for value in three_vector)
    )

    tensor = model.vertex_component_expression(
        1,
        left,
        right,
        result_particle_id=-21,
        result_chirality=0,
    )
    tensor_swapped = model.vertex_component_expression(
        1,
        right,
        left,
        result_particle_id=-21,
        result_chirality=0,
    )
    assert tensor_swapped == pytest.approx(tuple(-value for value in tensor))

    tensor_vector = model.vertex_component_expression(
        2,
        tensor,
        left,
        result_particle_id=21,
        result_chirality=0,
    )
    vector_tensor = model.vertex_component_expression(
        3,
        left,
        tensor,
        result_particle_id=21,
        result_chirality=0,
    )
    assert vector_tensor == pytest.approx(tuple(-value for value in tensor_vector))

    assert model.vertex_evaluation_equivalence(0).input_exchange_factor == (
        -1.0,
        0.0,
    )
    assert model.vertex_evaluation_equivalence(1).input_exchange_factor == (
        -1.0,
        0.0,
    )
    assert model.vertex_evaluation_equivalence(2).class_id == (
        model.vertex_evaluation_equivalence(3).class_id
    )
    assert model.vertex_evaluation_equivalence(3).input_order == (1, 0)
    assert model.vertex_evaluation_equivalence(3).factor == (-1.0, 0.0)
    assert model.vertex_evaluation_equivalence(4).class_id == (
        model.vertex_evaluation_equivalence(6).class_id
    )
    assert model.vertex_evaluation_equivalence(4).input_order == (1, 0)
    assert model.vertex_evaluation_equivalence(6).input_order == (0, 1)
    assert model.vertex_evaluation_equivalence(5).class_id == (
        model.vertex_evaluation_equivalence(7).class_id
    )
    assert model.vertex_evaluation_equivalence(5).input_order == (1, 0)
    assert model.vertex_evaluation_equivalence(7).input_order == (0, 1)
    assert model.vertex_evaluation_equivalence(21).class_id == (
        model.vertex_evaluation_equivalence(22).class_id
    )
    assert model.vertex_evaluation_equivalence(21).input_order == (0, 1)
    assert model.vertex_evaluation_equivalence(22).input_order == (1, 0)
    assert model.vertex_evaluation_equivalence(18).class_id == (
        model.vertex_evaluation_equivalence(19).class_id
    )
    assert model.vertex_evaluation_equivalence(18).input_order == (0, 1)
    assert model.vertex_evaluation_equivalence(19).input_order == (1, 0)


def test_builtin_higgs_vector_kernel_orientations_are_exact_aliases() -> None:
    model = BuiltinSMModel()
    scalar = (1.25 - 0.75j,)
    vector = (2.0 + 0.5j, -1.0j, 0.25 - 0.5j, -3.0)
    coupling = (model.mass(23) * model.weak_coupling_over_cosine(), 0.0)

    scalar_vector = model.vertex_component_expression(
        18,
        scalar,
        vector,
        result_particle_id=23,
        result_chirality=0,
        coupling=coupling,
    )
    vector_scalar = model.vertex_component_expression(
        19,
        vector,
        scalar,
        result_particle_id=23,
        result_chirality=0,
        coupling=coupling,
    )

    assert vector_scalar == pytest.approx(scalar_vector, rel=0.0, abs=0.0)


@pytest.mark.parametrize("kind,result_particle_id", ((10, 6), (11, -6)))
@pytest.mark.parametrize("chirality", (-1, 1))
def test_builtin_charged_current_embeds_weyl_input_in_massive_dirac_output(
    kind: int,
    result_particle_id: int,
    chirality: int,
) -> None:
    model = BuiltinSMModel()
    fermion = (1.25 - 0.5j, -0.75 + 2.0j)
    vector = (2.0, -1.0j, 0.5 + 0.25j, -3.0)
    coupling = (0.7 - 0.1j, -0.4 + 0.2j)
    padded = (
        (0j, 0j, *fermion)
        if chirality == -1
        else (*fermion, 0j, 0j)
    )

    mixed = model.vertex_component_expression(
        kind,
        fermion,
        vector,
        result_particle_id=result_particle_id,
        result_chirality=0,
        left_chirality=chirality,
        coupling=coupling,
    )
    dirac = model.vertex_component_expression(
        kind,
        padded,
        vector,
        result_particle_id=result_particle_id,
        result_chirality=0,
        coupling=coupling,
    )

    assert mixed == pytest.approx(dirac)


def test_model_builtin_compiles_to_canonical_records_without_replacing_path() -> None:
    compiled = compile_model_source("built-in-sm", use_cache=False)

    assert compiled.name == "built-in-sm"
    assert len(compiled.ir.particles) == 38
    assert len(compiled.ir.vertex_terms) == 211
    assert compiled.ir.oriented_kernels == ()
    assert compiled.source["kind"] == "built-in-sm"
    assert compiled.source["source_name"] is None


def test_model_builtin_public_name_has_no_legacy_alias() -> None:
    assert models.BuiltinSMModel is BuiltinSMModel
    assert not hasattr(models, "AmplicolSMLeadingColorModel")


def test_shared_model_contract_has_no_builtin_sm_pdg_fallbacks() -> None:
    generic = Model(name="generic-model-contract")

    assert issubclass(BuiltinSMModel, BuiltinModel)
    assert not issubclass(BuiltinModel, BuiltinSMModel)
    with pytest.raises(NotImplementedError, match="massless adjoint-vector role"):
        generic.is_massless_adjoint_vector(21)
    with pytest.raises(NotImplementedError, match="fundamental colored-fermion role"):
        generic.is_fundamental_colored_fermion(1)
    with pytest.raises(NotImplementedError, match="propagator lowering"):
        generic.propagator_lowering_rule(21)
    assert not generic.global_helicity_flip_equivalence_is_proven(
        (Vertex(0, (1, -1, 21)),)
    )
    assert not generic.pure_massless_adjoint_helicity_zero_rule_is_proven(
        object(),
        (Vertex(0, (21, 21, 21)),),
    )


def test_global_helicity_flip_proof_is_owned_by_the_builtin_model() -> None:
    model = BuiltinSMModel()

    assert model.global_helicity_flip_equivalence_is_proven(
        (Vertex(0, (1, -1, 21)), Vertex(6, (1, 21, 1)))
    )
    assert not model.global_helicity_flip_equivalence_is_proven(
        (Vertex(10, (1, -1, 22)),)
    )
