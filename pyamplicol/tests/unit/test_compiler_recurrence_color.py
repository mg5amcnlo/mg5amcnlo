# SPDX-License-Identifier: 0BSD

from dataclasses import replace

import pytest

from pyamplicol.models.compiler_color_flow import (
    compile_lc_color_transition_terms,
)
from pyamplicol.models.contracts import CompiledOrientedKernel


def _kernel(structure: str) -> CompiledOrientedKernel:
    return CompiledOrientedKernel(
        kind=7,
        term_id=3,
        vertex="V_1",
        particles=("left", "right", "result"),
        source_particle_legs=(0, 1, 2),
        component_expressions=("1",),
        coupling_expression="1",
        coupling_orders=(),
        runtime_parameters=(),
        color_source="exact-color-source",
        color_expression="exact-color-expression",
        color_projection_structure=structure,
        color_projection_coefficient=(1.0, 0.0),
    )


@pytest.mark.parametrize(
    ("representations", "permutation", "component"),
    (
        ((3, 8, -3), (0, 1), "open-string"),
        ((8, 3, -3), (1, 0), "open-string"),
        ((8, -3, 3), (0, 1), "open-string"),
        ((-3, 8, 3), (1, 0), "open-string"),
        ((3, -3, 8), (0, 1), "adjoint-segment"),
        ((-3, 3, 8), (1, 0), "adjoint-segment"),
    ),
)
def test_fundamental_generator_terms_are_orientation_explicit(
    representations: tuple[int, int, int],
    permutation: tuple[int, int],
    component: str,
) -> None:
    terms = compile_lc_color_transition_terms(
        _kernel("fundamental-generator"),
        representations,
        proof_source="test-exact-projection",
    )
    assert len(terms) == 1
    assert terms[0].input_permutation == permutation
    assert terms[0].component_operation == "concatenate-join"
    assert terms[0].result_component_kind == component


@pytest.mark.parametrize(
    ("current_representations", "tensor_representations", "permutation"),
    (
        ((3, 8, 3), (3, 8, -3), (0, 1)),
        ((-3, 8, -3), (-3, 8, 3), (1, 0)),
    ),
)
def test_fundamental_generator_separates_current_shape_from_tensor_role(
    current_representations: tuple[int, int, int],
    tensor_representations: tuple[int, int, int],
    permutation: tuple[int, int],
) -> None:
    terms = compile_lc_color_transition_terms(
        _kernel("fundamental-generator"),
        current_representations,
        proof_source="test-exact-projection",
        tensor_role_representations=tensor_representations,
    )

    assert len(terms) == 1
    assert terms[0].input_permutation == permutation
    assert terms[0].component_operation == "concatenate-join"
    assert terms[0].result_shape_kind == (
        "fundamental-open-string"
        if current_representations[2] == 3
        else "antifundamental-open-string"
    )


def test_adjoint_structure_constant_is_an_exact_commutator() -> None:
    terms = compile_lc_color_transition_terms(
        _kernel("adjoint-structure-constant"),
        (8, 8, 8),
        proof_source="test-exact-projection",
    )
    assert tuple(term.input_permutation for term in terms) == ((0, 1), (1, 0))
    assert tuple(term.exact_factor_expression for term in terms) == ("1", "-1")
    assert len({term.proof_digest for term in terms}) == 2


@pytest.mark.parametrize(
    ("structure", "representations", "closure_kind"),
    (
        ("color-identity", (3, -3, 1), "open-string"),
        ("adjoint-structure-constant", (8, 8, 8), "trace"),
        ("singlet", (1, 1, 1), None),
    ),
)
def test_compiler_attaches_exact_lc_closure_companions(
    structure: str,
    representations: tuple[int, int, int],
    closure_kind: str | None,
) -> None:
    kernel = _kernel(structure)
    transitions = compile_lc_color_transition_terms(
        kernel,
        representations,
        proof_source="test-exact-projection",
    )

    compiled = replace(kernel, lc_color_transition_terms=transitions)

    assert len(compiled.lc_color_closure_terms) == 1
    closure = compiled.lc_color_closure_terms[0]
    assert closure.component_operation == "close"
    assert closure.result_component_kind == closure_kind
    assert closure.result_component_role == "none"
    assert closure.result_shape_kind is None
    assert closure.exact_factor_expression == "1"
    assert ("contract-kind", "closure") in closure.provenance


def test_compiler_omits_closure_for_noncontractible_input_pair() -> None:
    kernel = _kernel("fundamental-generator")
    transitions = compile_lc_color_transition_terms(
        kernel,
        (3, 8, -3),
        proof_source="test-exact-projection",
    )

    compiled = replace(kernel, lc_color_transition_terms=transitions)

    assert compiled.lc_color_closure_terms == ()


def test_colored_literal_singlet_fails_without_contact_provenance() -> None:
    with pytest.raises(ValueError, match="requires explicit contact provenance"):
        compile_lc_color_transition_terms(
            replace(_kernel("singlet"), particles=("a", "b", "c")),
            (3, -3, 1),
            proof_source="test-exact-projection",
        )
