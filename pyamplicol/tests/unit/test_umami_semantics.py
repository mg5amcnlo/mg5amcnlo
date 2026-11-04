# SPDX-License-Identifier: 0BSD
"""Symbolic grouping proofs, with no numerical equivalence discovery."""

from __future__ import annotations

import copy
from types import SimpleNamespace as NS

import pytest

from pyamplicol.generation.umami_semantics import (
    ParameterSemantics,
    recurrence_structural_keys,
)


def model_ir():
    return {
        "name": "unfamiliar_theory",
        "parameters": [
            {
                "name": "base",
                "nature": "external",
                "parameter_type": "real",
                "value": [2, 0],
            },
            {
                "name": "independent",
                "nature": "external",
                "parameter_type": "real",
                "value": [2, 0],
            },
            {
                "name": "first",
                "nature": "internal",
                "parameter_type": "real",
                "expression": "2*base",
                "resolved_expression": "2*base",
            },
            {
                "name": "second",
                "nature": "internal",
                "parameter_type": "real",
                "expression": "2*base",
                "resolved_expression": "2*base",
            },
        ],
        "couplings": [],
    }


def circuit(parameter="first", pdg=900001):
    kernel = NS(
        kernel_id=0,
        contract_kind="vertex",
        output_layout=("component0",),
        exact_expressions=("g0*left(0)*right(0)",),
        input_contracts=(
            {
                "symbol": "g0",
                "role": "model-parameter",
                "component": 0,
                "model_parameter_index": 0,
            },
            {"symbol": "left(0)", "role": "left-current", "component": 0},
            {"symbol": "right(0)", "role": "right-current", "component": 0},
        ),
    )
    direct = NS(
        direct_executor_id=0,
        role="contribution",
        evaluator_binding_id=0,
        evaluator_resolver_key="vertex",
        semantic_template_ids=(),
        payload_binding=NS(
            prepared_kernel_id=0, contribution_parent_permutation=(0, 1)
        ),
    )
    pack = NS(
        kernels=(kernel,),
        recurrence_template_catalog=NS(
            parameters=(NS(prepared_parameter_id=0, name=parameter),),
            evaluator_bindings=(
                NS(
                    callable_kind="prepared-kernel",
                    prepared_kernel_id=0,
                    resolver_key="vertex",
                ),
            ),
            transitions=(),
            closures=(),
        ),
        recurrence_direct_template_catalog=NS(templates=(direct,)),
    )
    source = {
        "source_template_id": 0,
        "dimension": 1,
        "helicity": 0,
        "chirality": 0,
        "spin_state": 0,
        "crossing": {"phase": [1, 0]},
        "source_ir": {
            "identity": {"pdg": pdg, "orientation": "self-conjugate"},
            "wavefunction_family": "scalar",
            "statistics": "boson",
            "component_dimension": 1,
            "basis": "scalar-unit",
            "mass_parameter": None,
            "width_parameter": None,
        },
    }
    sections = {
        "abi": "pyamplicol-recurrence-exact-sections-v1",
        "strategy": "topology-replay",
        "process_id": f"process_{pdg}",
        "semantic_digest": str(pdg),
        "runtime_layout_digest": str(pdg),
        "counts": [3, 1, 1, 2],
        "executors": [[0, "contribution", "accumulate", [1, 1], 1, 0, 0, None]],
        "currents": [[pdg, 0, pdg, 0, 1, 0, 0, 0, 0, 1, 0, 0]],
        "sources": [[0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]],
        "contributions": [[0, 1, 0, 1, 2, 0, 0, 0]],
        "closures": [[2, 0, 0, 1, 0, 0, 0, 1, 0, 0]],
        "row_groups": [[0, 1, 0, 0, 0, 1]],
        "source_dispatch_variants": [],
        "exact_factors": [[1, 1, 0, 1]],
        "amplitude_destinations": [[0, 0, 0, 0, 1, 0]],
    }
    metadata = {
        "source_templates": [source],
        "normalization": {"average_factor": 1, "identical_factor": 1},
        "external_legs": [
            {"source_slot": 0, "is_initial": True},
            {"source_slot": 1, "is_initial": False},
        ],
        "parameter_projection": [
            {"runtime_slot": 0, "runtime_name": parameter, "component": 0}
        ],
    }
    remap = NS(
        source_slots=(0, 1),
        source_momentum_signs=(1, 1),
        source_helicity_signs=(1, 1),
        source_state_offsets=(0, 1, 2),
        source_state_indices=(0, 0),
        state_template_changes=(),
        source_template_changes=(),
        direct_executor_changes=(),
        parameter_slot_changes=(),
        public_flow_ids=(0,),
        physical_sector_ids=(0,),
    )
    return dict(
        exact_sections=sections,
        runtime_metadata=metadata,
        kernel_pack=pack,
        ir=model_ir(),
        physics={
            "color_accuracy": "full",
            "helicities": [{"values": [0, 0]}],
            "color_components": [{"id": "color:contracted", "index": 0}],
        },
        remap=remap,
        color_contraction_payload=b"same exact color contraction",
    )


def key(**kwargs):
    result = recurrence_structural_keys(**kwargs)
    assert result, "supported proof fixture must not silently fall back"
    return result


def test_parameter_definitions_resolve_without_freezing_mutable_defaults():
    semantics = ParameterSemantics(model_ir())
    assert semantics.parameter("first") == semantics.parameter("second")
    assert semantics.parameter("base") != semantics.parameter("independent")
    changed = model_ir()
    changed["parameters"][0]["value"] = [37, 0]
    assert ParameterSemantics(changed).parameter("first") == semantics.parameter(
        "first"
    )


def test_non_sm_flavours_with_equal_symbolic_circuits_share_keys():
    assert key(**circuit("first", 900001)) == key(**circuit("second", 900002))


def test_independent_inputs_with_equal_defaults_do_not_share_keys():
    assert key(**circuit("base")) != key(**circuit("independent"))


def test_complete_function_atom_input_contracts_are_bound():
    semantics = ParameterSemantics(model_ir())
    left = semantics.expression("f(0)*left(0)", {"left(0)": ("left", 0)})
    renamed = semantics.expression("f(0)*right(0)", {"right(0)": ("left", 0)})
    different = semantics.expression("f(0)*left(0)", {"left(0)": ("left", 1)})
    assert left == renamed
    assert left != different


def test_derived_contact_couplings_use_catalog_expression_not_generated_names():
    first = circuit("contact_coupling_one")
    second = circuit("contact_coupling_two")
    for data, name in (
        (first, "contact_coupling_one"),
        (second, "contact_coupling_two"),
    ):
        catalog = data["kernel_pack"].recurrence_template_catalog
        catalog.parameters = (
            NS(
                name=name,
                template_id="derived",
                prepared_parameter_id=0,
                parameter_kind="derived",
                value_type="complex",
                exact_expression_digest="same-symbolic-expression",
                dependency_parameter_ids=("external",),
            ),
            NS(
                name="base",
                template_id="external",
                prepared_parameter_id=1,
                parameter_kind="external",
            ),
        )
        data["kernel_pack"].kernels[0].output_layout = (f"auxiliary:{name}:c0",)
    assert key(**first) == key(**second)
    second["kernel_pack"].recurrence_template_catalog.parameters[
        0
    ].exact_expression_digest = "different-expression"
    assert key(**first) != key(**second)


def test_binding_couplings_keep_exact_rationals_beyond_decimal_precision():
    first = circuit()
    pack = first["kernel_pack"]
    pack.kernels[0].input_contracts[0].update(role="coupling-real")
    pack.recurrence_direct_template_catalog.templates[0].semantic_template_ids = (5,)
    pack.recurrence_template_catalog.transitions = (
        NS(
            template_id=5,
            binding_coupling=NS(
                real_numerator=1,
                real_denominator=3,
                imag_numerator=0,
                imag_denominator=1,
            ),
        ),
    )
    second = copy.deepcopy(first)
    coupling = (
        second["kernel_pack"]
        .recurrence_template_catalog.transitions[0]
        .binding_coupling
    )
    coupling.real_numerator = 10**35 + 1
    coupling.real_denominator = 3 * 10**35
    assert key(**first) != key(**second)


@pytest.mark.parametrize(
    "change",
    [
        "propagator",
        "chirality",
        "orientation",
        "source_basis",
        "width",
        "fermion_sign",
        "coherent_root",
        "color_contraction",
        "normalization",
        "momentum_route",
        "input_component",
        "parent_permutation",
    ],
)
def test_distinct_physics_never_shares_keys(change):
    original = circuit()
    modified = copy.deepcopy(original)
    source = modified["runtime_metadata"]["source_templates"][0]
    sections = modified["exact_sections"]
    if change == "propagator":
        modified["kernel_pack"].kernels[0].exact_expressions = (
            "g0*left(0)*right(0)/(p2-m2)",
        )
    elif change == "chirality":
        source["chirality"] = 1
    elif change == "orientation":
        source["source_ir"]["identity"]["orientation"] = "antiparticle"
    elif change == "source_basis":
        source["source_ir"]["basis"] = "different-basis"
    elif change == "width":
        source["source_ir"]["width_parameter"] = "base"
    elif change == "fermion_sign":
        sections["exact_factors"][0][0] = -1
    elif change == "coherent_root":
        sections["closures"].append(copy.deepcopy(sections["closures"][0]))
    elif change == "color_contraction":
        modified["color_contraction_payload"] = b"another exact contraction"
    elif change == "normalization":
        modified["runtime_metadata"]["normalization"]["identical_factor"] = 2
    elif change == "momentum_route":
        sections["contributions"][0][2] = 1
    elif change == "input_component":
        modified["kernel_pack"].kernels[0].input_contracts[1]["component"] = 1
    else:
        modified["kernel_pack"].recurrence_direct_template_catalog.templates[
            0
        ].payload_binding.contribution_parent_permutation = (1, 0)
    assert key(**original) != key(**modified)


def test_unknown_intrinsic_or_missing_data_falls_back_without_proof():
    data = circuit()
    data["runtime_metadata"].pop("source_templates")
    assert recurrence_structural_keys(**data) == {}
    data = circuit()
    data["kernel_pack"].recurrence_direct_template_catalog.templates = ()
    data["exact_sections"]["executors"][0][7] = "unknown-physics-intrinsic"
    assert recurrence_structural_keys(**data) == {}


@pytest.mark.parametrize("field", ["basis", "statistics", "component_dimension"])
def test_missing_source_semantics_cannot_authorize_grouping(field):
    data = circuit()
    data["runtime_metadata"]["source_templates"][0]["source_ir"].pop(field)
    assert recurrence_structural_keys(**data) == {}


def test_missing_contracted_color_payload_cannot_authorize_grouping():
    data = circuit()
    data["color_contraction_payload"] = None
    assert recurrence_structural_keys(**data) == {}


def test_nonidentity_source_state_mapping_is_not_misinterpreted():
    data = circuit()
    data["remap"].source_state_offsets = (0, 2, 3)
    data["remap"].source_state_indices = (1, 0, 0)
    assert recurrence_structural_keys(**data) == {}


def test_unreferenced_model_executors_do_not_block_process_proof():
    data = circuit()
    data["exact_sections"]["executors"].append(
        [73, "finalization", "unused", [], 1, 0, None, "unknown-unused-intrinsic"]
    )
    assert key(**data) == key(**circuit())


def test_source_record_digests_are_not_flavour_semantics():
    data = circuit()
    data["exact_sections"]["executors"].append(
        [
            1,
            "source",
            "initialize",
            [],
            1,
            1,
            None,
            "rusticol.source-fill.scalar.v1:" + "a" * 24,
        ]
    )
    data["exact_sections"]["row_groups"].append([0, 1, 0, 1, 0, 1])
    changed = copy.deepcopy(data)
    changed["exact_sections"]["executors"][1][7] = (
        "rusticol.source-fill.scalar.v1:" + "b" * 24
    )
    assert key(**data) == key(**changed)
    changed["exact_sections"]["executors"][1][7] = (
        "rusticol.source-fill.scalar.v2:" + "b" * 24
    )
    assert recurrence_structural_keys(**changed) == {}


def test_diagonal_closure_intrinsic_keeps_exact_coefficients():
    data = circuit()
    data["exact_sections"]["executors"].append(
        [
            1,
            "closure",
            "closure-add",
            [1, 1],
            1,
            2,
            None,
            "rusticol.closure-reduce.v1:" + "a" * 24,
        ]
    )
    data["exact_sections"]["row_groups"].append([0, 1, 0, 1, 0, 1])
    changed = copy.deepcopy(data)
    changed["exact_sections"]["exact_factors"][0][0] = -1
    assert key(**data) != key(**changed)


def test_parameter_cycles_do_not_authorize_grouping():
    data = circuit()
    data["ir"]["parameters"][2].update(expression="first", resolved_expression="first")
    assert recurrence_structural_keys(**data) == {}
