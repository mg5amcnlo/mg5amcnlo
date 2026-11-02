# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from pyamplicol.models._physics_ir import TensorAxisIR, TensorOrderingIR
from pyamplicol.models.compiler_contacts import _compress_contact_components
from pyamplicol.models.compiler_tensor_ordering import (
    compile_tensor_ordering_metadata,
    compile_vertex_index_bindings,
    identity_ordering_for_materialized_axes,
)
from pyamplicol.models.contracts import (
    CompiledModelIR,
    CompiledOrientedKernel,
    CompiledParticleRecord,
    CompiledVertexTerm,
)
from pyamplicol.models.loading import compiler_fingerprint


def _scalar(name: str = "phi") -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=name,
        pdg_code=8_100_001,
        spin=1,
        color=1,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )


def _fermion(name: str, antiname: str, pdg: int) -> CompiledParticleRecord:
    charge = 1.0 if pdg > 0 else -1.0
    return CompiledParticleRecord(
        name=name,
        antiname=antiname,
        pdg_code=pdg,
        spin=2,
        color=1,
        mass="ZERO",
        width="ZERO",
        charge=charge,
        quantum_numbers=(("electric_charge", "1" if pdg > 0 else "-1"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )


def _vector(name: str = "phi") -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=name,
        pdg_code=8_100_002,
        spin=3,
        color=8,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )


def _term(*, dummy: int | None = None) -> CompiledVertexTerm:
    color_expression = "1"
    lorentz_expression = "1"
    if dummy is not None:
        color_expression = (
            f"spenso::f(ufo_c_1,ufo_c_dummy_{dummy}_adjoint,ufo_c_2)"
            f"*spenso::f(ufo_c_dummy_{dummy}_adjoint,ufo_c_3,ufo_c_1)"
        )
        lorentz_expression = (
            f"spenso::g(ufo_l_1_1,ufo_l_dummy_{dummy})"
            f"*spenso::g(ufo_l_dummy_{dummy},ufo_l_1_2)"
        )
    return CompiledVertexTerm(
        id=0,
        vertex="V_1",
        particles=("phi", "phi", "phi"),
        color_index=0,
        lorentz_index=0,
        color_source="1",
        color_expression=color_expression,
        lorentz_name="L_1",
        lorentz_source="1",
        lorentz_expression=lorentz_expression,
        coupling="GC_1",
        coupling_expression="1",
        coupling_orders=(),
    )


def _compiled_scalar_model() -> CompiledModelIR:
    particle = _scalar()
    term = _term()
    kernel = CompiledOrientedKernel(
        kind=0,
        term_id=0,
        vertex="V_1",
        particles=("phi", "phi", "phi"),
        source_particle_legs=(0, 1, 2),
        component_expressions=("1",),
        coupling_expression="1",
        coupling_orders=(),
        runtime_parameters=(),
        color_source="1",
        color_expression="1",
        term_ids=(0,),
    )
    terms, kernels, orderings, current_orderings = compile_tensor_ordering_metadata(
        (term,),
        (particle,),
        (kernel,),
        (),
        (),
    )
    return CompiledModelIR(
        name="tensor-ordering-model",
        orders=(),
        parameters=(),
        particles=(particle,),
        couplings=(),
        propagators=(),
        vertex_terms=terms,
        oriented_kernels=kernels,
        direct_contractions=(),
        closure_contractions=(),
        tensor_orderings=orderings,
        current_orderings=current_orderings,
    )


def _compiled_fermion_model() -> CompiledModelIR:
    particles = (
        _fermion("psi", "psi_bar", 8_200_001),
        _fermion("psi_bar", "psi", -8_200_001),
        _scalar(),
    )
    term = CompiledVertexTerm(
        id=0,
        vertex="V_fermion",
        particles=("psi", "psi_bar", "phi"),
        color_index=0,
        lorentz_index=0,
        color_source="1",
        color_expression="1",
        lorentz_name="L_fermion",
        lorentz_source="1",
        lorentz_expression="1",
        coupling="GC_1",
        coupling_expression="1",
        coupling_orders=(),
    )
    kernel = CompiledOrientedKernel(
        kind=0,
        term_id=0,
        vertex="V_fermion",
        particles=("psi", "psi_bar", "phi"),
        source_particle_legs=(0, 1, 2),
        component_expressions=("1",),
        coupling_expression="1",
        coupling_orders=(),
        runtime_parameters=(),
        color_source="1",
        color_expression="1",
        term_ids=(0,),
    )
    terms, kernels, orderings, current_orderings = compile_tensor_ordering_metadata(
        (term,), particles, (kernel,), (), ()
    )
    return CompiledModelIR(
        name="tensor-ordering-fermion-model",
        orders=(),
        parameters=(),
        particles=particles,
        couplings=(),
        propagators=(),
        vertex_terms=terms,
        oriented_kernels=kernels,
        direct_contractions=(),
        closure_contractions=(),
        tensor_orderings=orderings,
        current_orderings=current_orderings,
    )


def _replace_ordering_id(payload: dict[str, object], old: str, new: str) -> None:
    for term in payload["vertex_terms"]:
        term["source_ordering_ids"] = [
            new if value == old else value for value in term["source_ordering_ids"]
        ]
    for kernel in payload["oriented_kernels"]:
        kernel["input_ordering_ids"] = [
            new if value == old else value for value in kernel["input_ordering_ids"]
        ]
        if kernel["output_ordering_id"] == old:
            kernel["output_ordering_id"] = new
    for current in payload["current_orderings"]:
        if current["ordering_id"] == old:
            current["ordering_id"] = new
        if current["kernel_ordering_id"] == old:
            current["kernel_ordering_id"] = new


def test_non_square_tensor_ordering_round_trips_with_permuted_storage() -> None:
    component_basis = (2, 0, 5, 1, 4, 3)
    component_expansion: list[tuple[int, int] | None] = [None] * 6
    for storage_slot, canonical_component in enumerate(component_basis):
        component_expansion[canonical_component] = (storage_slot, 1)
    ordering = TensorOrderingIR.create(
        basis="rectangular-test",
        axes=(
            TensorAxisIR("axis-0", "test-row", 2),
            TensorAxisIR("axis-1", "test-column", 3),
        ),
        component_basis=component_basis,
        component_expansion=component_expansion,
    )

    assert ordering.canonical_size == 6
    assert ordering.stored_size == 6
    assert TensorOrderingIR.from_json_dict(ordering.to_json_dict()) == ordering


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.__setitem__("ordering_id", "stale"),
            "ID does not match",
        ),
        (
            lambda payload: payload["axes"].append(deepcopy(payload["axes"][0])),
            "axis names must be unique",
        ),
        (
            lambda payload: payload["component_basis"].__setitem__(0, 99),
            "basis index is out of range",
        ),
        (
            lambda payload: payload["component_expansion"][0].__setitem__(1, 2),
            r"sign must be \+/-1",
        ),
    ],
)
def test_tensor_ordering_corruption_fails_closed(mutate, message: str) -> None:
    ordering = TensorOrderingIR.identity(
        basis="rectangular-test",
        axes=(
            TensorAxisIR("axis-0", "test-row", 2),
            TensorAxisIR("axis-1", "test-column", 3),
        ),
    )
    payload = ordering.to_json_dict()

    mutate(payload)

    with pytest.raises(ValueError, match=message):
        TensorOrderingIR.from_json_dict(payload)


def test_source_axis_and_dummy_renaming_do_not_change_ordering_identity() -> None:
    first = identity_ordering_for_materialized_axes(
        ("ufo_l_1_1", "ufo_l_2_1"),
        (2, 3),
    )
    relabelled = identity_ordering_for_materialized_axes(
        ("ufo_l_1_9", "ufo_l_2_9"),
        (2, 3),
    )
    particle = _vector()
    bindings_7 = compile_vertex_index_bindings(_term(dummy=7), {"phi": particle})
    bindings_91 = compile_vertex_index_bindings(_term(dummy=91), {"phi": particle})

    assert first.ordering_id == relabelled.ordering_id
    assert tuple(binding.normalized_name for binding in bindings_7) == tuple(
        binding.normalized_name for binding in bindings_91
    )
    assert {binding.source_dummy for binding in bindings_7 if binding.source_dummy} == {
        -7
    }
    assert {
        binding.source_dummy for binding in bindings_91 if binding.source_dummy
    } == {-91}


def test_compiled_model_ordering_graph_round_trips_and_rejects_stale_links() -> None:
    model = _compiled_scalar_model()
    payload = model.to_dict()

    assert CompiledModelIR.from_dict(payload) == model

    stale_kernel = deepcopy(payload)
    stale_kernel["oriented_kernels"][0]["output_ordering_id"] = "absent"
    with pytest.raises(ValueError, match="absent tensor ordering 'absent'"):
        CompiledModelIR.from_dict(stale_kernel)

    wrong_size = deepcopy(payload)
    wrong_size["oriented_kernels"][0]["component_expressions"].append("0")
    with pytest.raises(ValueError, match="2 components for output ordering size 1"):
        CompiledModelIR.from_dict(wrong_size)

    stale_source = deepcopy(payload)
    stale_source["vertex_terms"][0]["source_ordering_ids"][0] = "absent"
    with pytest.raises(ValueError, match="source leg 1 refers to absent"):
        CompiledModelIR.from_dict(stale_source)

    duplicate_ordering = deepcopy(payload)
    duplicate_ordering["tensor_orderings"].append(
        deepcopy(duplicate_ordering["tensor_orderings"][0])
    )
    with pytest.raises(ValueError, match="duplicate tensor ordering"):
        CompiledModelIR.from_dict(duplicate_ordering)

    invalid_embedding = deepcopy(payload)
    invalid_embedding["current_orderings"][0]["input_embedding"] = [1]
    with pytest.raises(ValueError, match="does not map every stored component"):
        CompiledModelIR.from_dict(invalid_embedding)

    unverified_exchange = deepcopy(payload)
    unverified_exchange["oriented_kernels"][0][
        "evaluation_input_exchange_factor"
    ] = [-1.0, 0.0]
    with pytest.raises(ValueError, match="unverified input-exchange relation"):
        CompiledModelIR.from_dict(unverified_exchange)

    invalid_exchange = deepcopy(unverified_exchange)
    invalid_exchange["oriented_kernels"][0][
        "evaluation_equivalence_verified"
    ] = True
    invalid_exchange["oriented_kernels"][0][
        "evaluation_input_exchange_factor"
    ] = [0.0, 1.0]
    with pytest.raises(ValueError, match="invalid input-exchange phase"):
        CompiledModelIR.from_dict(invalid_exchange)


def test_compiled_model_rejects_self_consistent_but_false_ordering() -> None:
    payload = _compiled_scalar_model().to_dict()
    ordering = payload["tensor_orderings"][0]
    old_id = ordering["ordering_id"]
    ordering["basis"] = "forged-scalar"
    replacement = TensorOrderingIR.create(
        basis=ordering["basis"],
        axes=(),
        component_basis=ordering["component_basis"],
        component_expansion=tuple(
            None if entry is None else tuple(entry)
            for entry in ordering["component_expansion"]
        ),
    )
    ordering["ordering_id"] = replacement.ordering_id
    _replace_ordering_id(payload, old_id, replacement.ordering_id)

    with pytest.raises(ValueError, match="stale source tensor orderings"):
        CompiledModelIR.from_dict(payload)


def test_compiled_model_rejects_fabricated_index_bindings() -> None:
    particle = _vector()
    term = _term(dummy=7)
    kernel = CompiledOrientedKernel(
        kind=0,
        term_id=0,
        vertex="V_1",
        particles=("phi", "phi", "phi"),
        source_particle_legs=(0, 1, 2),
        component_expressions=("1", "1", "1", "1"),
        coupling_expression="1",
        coupling_orders=(),
        runtime_parameters=(),
        color_source="1",
        color_expression="1",
        term_ids=(0,),
    )
    terms, kernels, orderings, current_orderings = compile_tensor_ordering_metadata(
        (term,), (particle,), (kernel,), (), ()
    )
    model = CompiledModelIR(
        name="tensor-index-binding-model",
        orders=(),
        parameters=(),
        particles=(particle,),
        couplings=(),
        propagators=(),
        vertex_terms=terms,
        oriented_kernels=kernels,
        direct_contractions=(),
        closure_contractions=(),
        tensor_orderings=orderings,
        current_orderings=current_orderings,
    )
    payload = model.to_dict()
    payload["vertex_terms"][0]["index_bindings"][0]["normalized_name"] = (
        "color:color-singlet:dummy-99"
    )

    with pytest.raises(ValueError, match="stale tensor index bindings"):
        CompiledModelIR.from_dict(payload)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("spenso::g(ufo_l_bad_1,ufo_l_1_1)", "malformed normalized"),
        ("spenso::g(ufo_s_1_1,ufo_l_1_2)", "incompatible with particle"),
    ],
)
def test_index_bindings_reject_malformed_or_particle_incompatible_indices(
    expression: str,
    message: str,
) -> None:
    term = replace(
        _term(),
        particles=("phi", "phi", "phi"),
        lorentz_expression=expression,
    )

    with pytest.raises(ValueError, match=message):
        compile_vertex_index_bindings(term, {"phi": _vector()})


def test_compiled_model_rejects_runtime_incompatible_current_mapping() -> None:
    payload = _compiled_fermion_model().to_dict()
    current = next(
        item
        for item in payload["current_orderings"]
        if item["particle"] == "psi" and item["chirality"] == 1
    )
    current["input_embedding"] = [1, 0, None, None]
    current["result_projection"] = [3, 2]

    with pytest.raises(ValueError, match="current tensor mappings are not canonical"):
        CompiledModelIR.from_dict(payload)


def test_all_zero_contact_compression_keeps_one_explicit_zero_slot() -> None:
    component_basis, component_expansion = _compress_contact_components(("0", "0"))

    assert component_basis == (0,)
    assert component_expansion == ((0, 1), None)
    ordering = TensorOrderingIR.create(
        basis="zero-contact",
        axes=(TensorAxisIR("axis-0", "test", 2),),
        component_basis=component_basis,
        component_expansion=component_expansion,
    )
    assert ordering.stored_size == 1


def test_compiled_model_ordering_fields_are_required_on_deserialization() -> None:
    payload = _compiled_scalar_model().to_dict()
    del payload["tensor_orderings"]

    with pytest.raises(
        ValueError,
        match="missing required field 'tensor_orderings'",
    ):
        CompiledModelIR.from_dict(payload)

    assert compiler_fingerprint()["tensor_ordering_contract"] == (
        "explicit-canonical-component-order-v1"
    )


def test_weyl_current_embeddings_and_result_projections_are_explicit() -> None:
    particles = (
        _fermion("psi", "psi_bar", 8_200_001),
        _fermion("psi_bar", "psi", -8_200_001),
    )
    _terms, _kernels, orderings, current_orderings = (
        compile_tensor_ordering_metadata((), particles, (), (), ())
    )
    by_selector = {record.selector: record for record in current_orderings}

    assert by_selector[("psi", 1)].input_embedding == (0, 1, None, None)
    assert by_selector[("psi", 1)].result_projection == (2, 3)
    assert by_selector[("psi", -1)].input_embedding == (None, None, 0, 1)
    assert by_selector[("psi", -1)].result_projection == (0, 1)
    assert {ordering.basis for ordering in orderings} == {
        "dirac",
        "weyl-chirality:+1",
        "weyl-chirality:-1",
    }
