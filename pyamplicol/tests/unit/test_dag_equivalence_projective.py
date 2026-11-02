# SPDX-License-Identifier: 0BSD
"""Algebraically exact projective-current proof contracts."""

from __future__ import annotations

from dataclasses import replace
from math import inf

import pytest

from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_equivalence import (
    _canonicalize_projective_term_vector,
    _classify_current_term_vector,
    _complex_weight_mul,
    _exact_representable_complex_product,
    _exact_representable_complex_ratio,
    _term_vector_scaled_exactly,
    _validate_numerical_current_application_baseline,
    assign_recursive_current_evaluation_reuse,
    project_rectangular_dynamic_color_classes,
)
from pyamplicol.generation.dag_types import ColorState, GenericDAG
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def _vector(
    *coefficients: tuple[float, float],
) -> tuple[tuple[tuple[object, ...], tuple[float, float]], ...]:
    return tuple(
        ((index,), coefficient) for index, coefficient in enumerate(coefficients)
    )


def _classify_sequence(
    vectors: tuple[
        tuple[tuple[tuple[object, ...], tuple[float, float]], ...],
        ...,
    ],
):
    equivalence_by_expression = {}
    projective_representatives_by_expression = {}
    return tuple(
        _classify_current_term_vector(
            current_id=current_id,
            contract=("same-current-contract",),
            term_vector=term_vector,
            equivalence_by_expression=equivalence_by_expression,
            projective_representatives_by_expression=(
                projective_representatives_by_expression
            ),
        )
        for current_id, term_vector in enumerate(vectors)
    )


def test_projective_current_reuse_retains_an_exact_complex_factor() -> None:
    representative = _vector((2.0, 0.0), (0.0, 4.0))
    scaled = _vector((0.0, 1.0), (-2.0, 0.0))

    equivalences = _classify_sequence((representative, scaled, scaled))

    assert equivalences[0].representative_id == 0
    assert equivalences[0].factor == (1.0, 0.0)
    assert equivalences[1].representative_id == 0
    assert equivalences[1].factor == (0.0, 0.5)
    assert equivalences[2] == equivalences[1]
    assert _term_vector_scaled_exactly(
        representative,
        equivalences[1].factor,
        scaled,
    )


def test_identity_and_sign_reuse_remain_deterministic_fast_paths() -> None:
    representative = _vector((2.0, 1.0), (-4.0, 3.0))
    opposite = _vector((-2.0, -1.0), (4.0, -3.0))
    vectors = (representative, representative, opposite)

    first = _classify_sequence(vectors)
    second = _classify_sequence(vectors)

    assert first == second
    assert tuple(item.representative_id for item in first) == (0, 0, 0)
    assert tuple(item.factor for item in first) == (
        (1.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
    )


def test_exact_zero_vectors_reuse_without_projective_normalization() -> None:
    equivalences = _classify_sequence(((), ()))

    assert tuple(item.representative_id for item in equivalences) == (0, 0)
    assert tuple(item.factor for item in equivalences) == (
        (1.0, 0.0),
        (1.0, 0.0),
    )
    assert _canonicalize_projective_term_vector(()) is None


def test_projective_reuse_rejects_a_nonproportional_coefficient() -> None:
    representative = _vector((2.0, 0.0), (0.0, 4.0))
    deformed = _vector((0.0, 1.0), (-2.5, 0.0))

    equivalences = _classify_sequence((representative, deformed))

    assert equivalences[1].representative_id == 1
    assert equivalences[1].factor == (1.0, 0.0)


def test_projective_reuse_rejects_a_rounded_unrepresentable_factor() -> None:
    representative = _vector((3.0, 0.0), (6.0, 0.0))
    rounded_third = _vector((1.0, 0.0), (2.0, 0.0))

    assert (
        _exact_representable_complex_ratio(
            rounded_third[0][1],
            representative[0][1],
        )
        is None
    )
    equivalences = _classify_sequence((representative, rounded_third))
    assert equivalences[1].representative_id == 1


def test_recursive_projective_factor_composition_fails_closed() -> None:
    assert _exact_representable_complex_product(
        (2.0, 0.0),
        (0.5, 0.0),
    ) == (1.0, 0.0)
    assert (
        _exact_representable_complex_product(
            (1.0e308, 0.0),
            (1.0e308, 0.0),
        )
        is None
    )
    assert (
        _exact_representable_complex_product(
            (0.1, 0.0),
            (0.2, 0.0),
        )
        is None
    )


def test_projective_reuse_rejects_an_ambiguous_exact_representative() -> None:
    first = _vector((3.0, 0.0), (6.0, 0.0))
    second = _vector((1.0, 0.0), (2.0, 0.0))
    exact_multiple_of_both = _vector((6.0, 0.0), (12.0, 0.0))

    equivalences = _classify_sequence((first, second, exact_multiple_of_both))

    assert equivalences[0].representative_id == 0
    assert equivalences[1].representative_id == 1
    assert equivalences[2].representative_id == 2


def test_projective_canonicalization_rejects_unsafe_vectors() -> None:
    assert _canonicalize_projective_term_vector(()) is None
    assert _canonicalize_projective_term_vector(_vector((0.0, 0.0))) is None
    assert _canonicalize_projective_term_vector(_vector((inf, 0.0))) is None

    duplicate_key = (
        ((0,), (1.0, 0.0)),
        ((0,), (2.0, 0.0)),
    )
    unordered = (
        ((1,), (1.0, 0.0)),
        ((0,), (2.0, 0.0)),
    )
    assert _canonicalize_projective_term_vector(duplicate_key) is None
    assert _canonicalize_projective_term_vector(unordered) is None


def test_projective_factor_propagates_to_a_child_evaluation_group() -> None:
    """A proportional parent must share its child's kernel, not the root value."""

    model = BuiltinSMModel()
    seed = compile_generic_dag(
        build_process_ir("d d~ > z g", color_accuracy="lc"),
        model=model,
    )
    template = seed.interactions[0]
    generated_template = seed.currents[template.result_id]
    source_templates = tuple(seed.currents[source_id] for source_id in seed.sources[:2])
    sources = tuple(
        replace(source, id=current_id)
        for current_id, source in enumerate(source_templates)
    )
    generated = tuple(
        replace(generated_template, id=current_id) for current_id in range(2, 6)
    )

    def interaction(
        interaction_id: int,
        left_id: int,
        right_id: int,
        result_id: int,
        color_weight: tuple[float, float],
    ):
        return replace(
            template,
            id=interaction_id,
            left_id=left_id,
            right_id=right_id,
            result_id=result_id,
            color_weight=color_weight,
            evaluation_group_id=None,
            evaluation_factor=(1.0, 0.0),
        )

    interactions = (
        interaction(0, 0, 1, 2, (1.0, 0.0)),
        interaction(1, 0, 1, 3, (2.0, 0.0)),
        interaction(2, 2, 1, 4, (1.0, 0.0)),
        interaction(3, 3, 1, 5, (1.0, 0.0)),
    )
    root = replace(
        seed.amplitude_roots[0],
        id=0,
        left_id=5,
        right_id=1,
    )
    dag = replace(
        seed,
        currents=(*sources, *generated),
        sources=(0, 1),
        interactions=interactions,
        amplitude_roots=(root,),
        selected_source_helicities=(),
        selected_color_sector_ids=(),
        lc_topology_replay=None,
        helicity_recurrence=None,
        helicity_materialization=None,
    )

    rewritten = assign_recursive_current_evaluation_reuse(dag, model)

    parent, proportional_parent, child, proportional_child = rewritten.interactions
    assert parent.evaluation_group_id == proportional_parent.evaluation_group_id
    assert child.evaluation_group_id == proportional_child.evaluation_group_id
    assert proportional_child.evaluation_factor == _complex_weight_mul(
        (2.0, 0.0),
        child.evaluation_factor,
    )
    # Roots consume separately materialized currents. Rewriting a shared child
    # kernel must therefore leave the physical amplitude contraction untouched.
    assert rewritten.amplitude_roots == dag.amplitude_roots


def test_numerical_baseline_authentication_canonicalizes_signed_zero_only() -> None:
    model = BuiltinSMModel()
    dag = assign_recursive_current_evaluation_reuse(
        compile_generic_dag(
            build_process_ir("d d~ > z g", color_accuracy="lc"),
            model=model,
        ),
        model,
    )
    interaction = dag.interactions[0]
    factor = interaction.evaluation_factor
    signed_zero = replace(
        interaction,
        evaluation_factor=(factor[0], -0.0),
    )
    signed_zero_dag = replace(
        dag,
        interactions=(signed_zero, *dag.interactions[1:]),
    )

    _validate_numerical_current_application_baseline(signed_zero_dag, model)

    tampered = replace(signed_zero, evaluation_factor=(factor[0] + 1.0, -0.0))
    with pytest.raises(ValueError, match="unauthenticated for interaction 0"):
        _validate_numerical_current_application_baseline(
            replace(dag, interactions=(tampered, *dag.interactions[1:])),
            model,
        )


def _dynamic_color_projection_fixture() -> tuple[
    GenericDAG,
    BuiltinSMModel,
    dict[int, complex],
    int,
]:
    model = BuiltinSMModel()
    seed = compile_generic_dag(
        build_process_ir("d d~ > z g", color_accuracy="lc"),
        model=model,
    )
    source0 = replace(seed.currents[seed.sources[0]], id=0)
    source1 = replace(seed.currents[seed.sources[1]], id=1)
    generated_template = seed.currents[seed.interactions[0].result_id]

    def projected_current(
        current_id: int,
        *,
        family: str,
        color_id: int,
    ):
        index = replace(
            generated_template.index,
            color_state=ColorState("lc", sector_id=color_id),
            auxiliary_kind=family,
        )
        return replace(generated_template, id=current_id, index=index)

    currents = (
        source0,
        source1,
        projected_current(2, family="a", color_id=11),
        projected_current(3, family="a", color_id=12),
        projected_current(4, family="b", color_id=21),
        projected_current(5, family="b", color_id=22),
        projected_current(6, family="c", color_id=132),
        projected_current(7, family="c", color_id=32),
    )
    template = seed.interactions[0]

    def interaction(
        interaction_id: int,
        left_id: int,
        right_id: int,
        result_id: int,
        weight: tuple[float, float],
    ):
        return replace(
            template,
            id=interaction_id,
            left_id=left_id,
            right_id=right_id,
            result_id=result_id,
            color_weight=weight,
            evaluation_group_id=None,
            evaluation_factor=(1.0, 0.0),
        )

    interactions = (
        interaction(0, 0, 1, 2, (1.0, 2.0)),
        interaction(1, 0, 1, 3, (2.0, -1.0)),
        interaction(2, 0, 1, 4, (-3.0, 1.0)),
        interaction(3, 0, 1, 5, (4.0, 0.5)),
        interaction(4, 2, 4, 6, (1.0, -0.5)),
        interaction(5, 2, 5, 6, (1.0, -0.5)),
        interaction(6, 3, 4, 6, (1.0, -0.5)),
        interaction(7, 3, 5, 6, (1.0, -0.5)),
        interaction(8, 2, 4, 7, (-0.25, 2.0)),
        interaction(9, 2, 5, 7, (-0.25, 2.0)),
        interaction(10, 3, 4, 7, (-0.25, 2.0)),
        interaction(11, 3, 5, 7, (-0.25, 2.0)),
    )
    selected_sector_id = int(seed.color_plan.sectors[0].id)
    currents = (
        *currents[:6],
        replace(
            currents[6],
            index=replace(
                currents[6].index,
                color_state=ColorState("lc", sector_id=selected_sector_id + 100),
            ),
        ),
        replace(
            currents[7],
            index=replace(
                currents[7].index,
                color_state=ColorState("lc", sector_id=selected_sector_id),
            ),
        ),
    )
    roots = (
        replace(
            seed.amplitude_roots[0],
            id=0,
            left_id=6,
            right_id=0,
            color_weight=(0.75, -0.25),
            color_sector_id=selected_sector_id,
        ),
        replace(
            seed.amplitude_roots[0],
            id=1,
            left_id=7,
            right_id=0,
            color_weight=(0.75, -0.25),
            color_sector_id=None,
        ),
    )
    dag = GenericDAG(
        process=seed.process,
        color_plan=seed.color_plan,
        currents=currents,
        sources=(0, 1),
        interactions=interactions,
        amplitude_roots=roots,
        truncated=False,
        helicity_coverage="selected",
        color_coverage="selected",
        selected_source_helicities=(),
        selected_color_sector_ids=(selected_sector_id,),
    )
    return dag, model, {0: 0.75 - 0.2j, 1: -1.25 + 0.5j}, selected_sector_id


def _evaluate_scalar_bilinear_dag(
    dag: GenericDAG,
    source_values: dict[int, complex],
) -> complex:
    """Numerically exercise the exact multilinear algebra of the projection."""

    values = [0j] * len(dag.currents)
    for current_id, value in source_values.items():
        values[current_id] = value
    for row in dag.interactions:
        values[row.result_id] += (
            complex(*row.color_weight) * values[row.left_id] * values[row.right_id]
        )
    return sum(
        complex(*root.color_weight) * values[root.left_id] * values[root.right_id]
        for root in dag.amplitude_roots
    )


def test_dynamic_color_projection_preserves_complex_cartesian_sums() -> None:
    dag, model, source_values, selected_sector_id = (
        _dynamic_color_projection_fixture()
    )

    projected, certificate = project_rectangular_dynamic_color_classes(
        dag,
        model,
        source_revision="a" * 40,
    )
    projected_source_values = {
        certificate.old_to_new_current_ids[current_id]: value
        for current_id, value in source_values.items()
    }

    assert certificate.applied is True
    assert certificate.before_current_count == 8
    assert certificate.after_current_count == 5
    assert certificate.before_interaction_count == 12
    assert certificate.after_interaction_count == 6
    assert certificate.projected_current_class_count == 3
    assert certificate.rectangular_interaction_group_count == 6
    assert certificate.rectangular_closure_group_count == 1
    assert certificate.source_revision == "a" * 40
    assert certificate.equality_check_status == (
        "passed-exact-cartesian-products-no-duplicates"
    )
    assert len(certificate.old_to_new_current_ids) == len(dag.currents)
    assert all(
        len(value) == 64
        for key, value in certificate.to_json_dict().items()
        if key.endswith("_sha256")
    )
    assert len(projected.amplitude_roots) == 1
    # Root 1 originally resolved sector ``selected_sector_id`` implicitly,
    # while the class representative (old current 6) carried another sector.
    # The emitted closure must retain the pre-projection physical sector.
    assert projected.amplitude_roots[0].color_sector_id == selected_sector_id
    assert _evaluate_scalar_bilinear_dag(
        projected,
        projected_source_values,
    ) == pytest.approx(_evaluate_scalar_bilinear_dag(dag, source_values))


def test_dynamic_color_projection_retains_basis_and_line_group_metadata() -> None:
    dag, model, _source_values, _selected_sector_id = (
        _dynamic_color_projection_fixture()
    )
    currents = list(dag.currents)
    current = currents[3]
    currents[3] = replace(
        current,
        index=replace(
            current.index,
            color_state=replace(
                current.index.color_state,
                line_groups=(7, 8),
                basis_key=("distinct-coherent-basis",),
            ),
        ),
    )
    separated = replace(dag, currents=tuple(currents))

    _projected, certificate = project_rectangular_dynamic_color_classes(
        separated,
        model,
    )

    assert certificate.old_to_new_current_ids[2] != (
        certificate.old_to_new_current_ids[3]
    )
