# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import replace

import pytest
from symbolica import Expression
from ufo_model_loader.symbolica_processing import wrap_indices

from pyamplicol._internal.physics.symbols import symbols
from pyamplicol.models import compiler_symbolica as _sym
from pyamplicol.models.compiler_auxiliary_components import (
    reduce_contact_auxiliary_components,
)
from pyamplicol.models.compiler_contact_trees import (
    _compile_heft_colored_contact_trees,
)
from pyamplicol.models.contracts import (
    CompiledParticleRecord,
    CompiledVertexTerm,
)
from pyamplicol.models.tensors import (
    normalize_color_expression,
    normalize_lorentz_expression,
)

_MODEL_SYMBOLS = symbols.model("synthetic-heft-contact")
_VVVS2 = (
    "P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) "
    "- P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) "
    "+ P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)"
)
_HGGGG_PAIRINGS = (
    (
        "f(-1,1,2)*f(3,4,-1)",
        "Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)",
        frozenset({(0, 1), (2, 3)}),
    ),
    (
        "f(-1,1,3)*f(2,4,-1)",
        "Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)",
        frozenset({(0, 2), (1, 3)}),
    ),
    (
        "f(-1,1,4)*f(2,3,-1)",
        "Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)",
        frozenset({(0, 3), (1, 2)}),
    ),
)


def _particle(
    name: str,
    pdg: int,
    *,
    spin: int,
    color: int,
) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=name,
        pdg_code=pdg,
        spin=spin,
        color=color,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )


_GLUON = _particle("G", 21, spin=3, color=8)
_HIGGS = _particle("H", 25, spin=1, color=1)


def _ufo_source(source: str) -> str:
    return Expression.parse(source, default_namespace="UFO").to_canonical_string()


def _wrapped_lorentz(source: str) -> str:
    return wrap_indices(
        Expression.parse(source, default_namespace="UFO")
    ).to_canonical_string()


def _term(
    term_id: int,
    *,
    color: str,
    lorentz: str,
    valence: int,
) -> CompiledVertexTerm:
    particles = ("G",) * (valence - 1) + ("H",)
    colors = (8,) * (valence - 1) + (1,)
    spins = (3,) * (valence - 1) + (1,)
    color_source = _ufo_source(color)
    lorentz_source = _wrapped_lorentz(lorentz)
    return CompiledVertexTerm(
        id=term_id,
        vertex=f"V_HEFT_{term_id}",
        particles=particles,
        color_index=0,
        lorentz_index=0,
        color_source=color_source,
        color_expression=normalize_color_expression(
            color_source,
            colors,
        ).expression,
        lorentz_name=f"L_HEFT_{term_id}",
        lorentz_source=lorentz_source,
        lorentz_expression=normalize_lorentz_expression(
            lorentz_source,
            spins,
            model_symbols=_MODEL_SYMBOLS,
        ).expression,
        coupling="GC_HEFT",
        coupling_expression="ufo_heft_coupling",
        coupling_orders=(("HIG", 1),),
    )


def _compile(*terms: CompiledVertexTerm):
    return _compile_heft_colored_contact_trees(
        terms,
        (_GLUON, _HIGGS),
        start_kind=0,
        model_symbols=_MODEL_SYMBOLS,
    )


def test_hggg_contact_uses_authenticated_adjoint_and_identity_nodes() -> None:
    term = _term(100, color="f(1,2,3)", lorentz=_VVVS2, valence=4)

    auxiliaries, kernels = _compile(term)

    assert len(auxiliaries) == 2
    assert {(particle.spin, particle.color) for particle in auxiliaries} == {(-1, 8)}
    assert len(kernels) == 7
    assert {kernel.color_projection_structure for kernel in kernels} == {None}
    assert {kernel.color_source.split("(")[0] for kernel in kernels} == {
        "UFO::f",
        "UFO::Identity",
    }
    assert all(kernel.coupling_expression == "1" for kernel in kernels)
    partials = tuple(kernel for kernel in kernels if kernel.vertex.endswith("partial"))
    finals = tuple(kernel for kernel in kernels if kernel.vertex.endswith("final"))
    assert all(
        not kernel.runtime_parameters and not kernel.coupling_orders
        for kernel in partials
    )
    assert all(
        kernel.runtime_parameters == ("derived_coupling_100",) for kernel in finals
    )
    assert all(kernel.coupling_orders == (("HIG", 1),) for kernel in finals)
    assert all(
        expression.count("derived_coupling_100") == 1
        for kernel in finals
        for expression in kernel.component_expressions
    )


def test_hggg_source_permutation_sign_is_carried_by_the_adjoint_node() -> None:
    canonical = _term(101, color="f(1,2,3)", lorentz=_VVVS2, valence=4)
    reversed_source = replace(
        canonical,
        color_source=_ufo_source("f(2,1,3)"),
        color_expression=normalize_color_expression(
            _ufo_source("f(2,1,3)"),
            (8, 8, 8, 1),
        ).expression,
    )

    _canonical_auxiliaries, canonical_kernels = _compile(canonical)
    _reversed_auxiliaries, reversed_kernels = _compile(reversed_source)
    assert len(canonical_kernels) == len(reversed_kernels) == 7
    for canonical_kernel, reversed_kernel in zip(
        canonical_kernels,
        reversed_kernels,
        strict=True,
    ):
        assert (
            canonical_kernel.source_particle_legs
            == reversed_kernel.source_particle_legs
        )
        carries_reversed_structure_constant = canonical_kernel.color_source.startswith(
            "UFO::f"
        )
        comparison = (
            (lambda left, right: _sym.E(left) + _sym.E(right))
            if carries_reversed_structure_constant
            else (lambda left, right: _sym.E(left) - _sym.E(right))
        )
        assert all(
            comparison(left, right).to_canonical_string() == "0"
            for left, right in zip(
                canonical_kernel.component_expressions,
                reversed_kernel.component_expressions,
                strict=True,
            )
        )


def test_hgggg_lowers_all_three_exact_color_lorentz_pairings() -> None:
    terms = tuple(
        _term(200 + index, color=color, lorentz=lorentz, valence=5)
        for index, (color, lorentz, _pairs) in enumerate(_HGGGG_PAIRINGS)
    )

    auxiliaries, kernels = _compile(*terms)

    assert auxiliaries
    assert {kernel.term_id for kernel in kernels} == {200, 201, 202}
    for term, (_color, _lorentz, expected_pairs) in zip(
        terms,
        _HGGGG_PAIRINGS,
        strict=True,
    ):
        higgs_result_partials = {
            tuple(sorted(kernel.source_particle_legs[:2]))
            for kernel in kernels
            if kernel.term_id == term.id
            and kernel.vertex.endswith("partial")
            and f"_{term.id}_r4_" in kernel.particles[2]
        }
        assert higgs_result_partials == expected_pairs
        finals = tuple(
            kernel
            for kernel in kernels
            if kernel.term_id == term.id and kernel.vertex.endswith("final")
        )
        assert finals
        assert all(
            expression.count(f"derived_coupling_{term.id}") == 1
            for kernel in finals
            for expression in kernel.component_expressions
        )


def test_heft_contact_identical_gluon_multiplicities_are_applied_once() -> None:
    hggg = _term(300, color="f(1,2,3)", lorentz=_VVVS2, valence=4)
    color, lorentz, _pairs = _HGGGG_PAIRINGS[0]
    hgggg = _term(301, color=color, lorentz=lorentz, valence=5)

    _auxiliaries, kernels = _compile(hggg, hgggg)

    hggg_higgs = tuple(
        kernel
        for kernel in kernels
        if kernel.term_id == 300
        and kernel.vertex.endswith("final")
        and kernel.source_particle_legs[-1] == 3
    )
    hgggg_higgs = tuple(
        kernel
        for kernel in kernels
        if kernel.term_id == 301
        and kernel.vertex.endswith("final")
        and kernel.source_particle_legs[-1] == 4
    )
    assert hggg_higgs and hgggg_higgs
    assert all("1/3" in kernel.component_expressions[0] for kernel in hggg_higgs)
    assert all("1/6" in kernel.component_expressions[0] for kernel in hgggg_higgs)


def test_all_hgggg_pairings_reduce_exactly_to_antisymmetric_current_components() -> (
    None
):
    terms = tuple(
        _term(500 + index, color=color, lorentz=lorentz, valence=5)
        for index, (color, lorentz, _pairs) in enumerate(_HGGGG_PAIRINGS)
    )
    auxiliaries, kernels = _compile(*terms)
    particles, reduced = reduce_contact_auxiliary_components(
        (_GLUON, _HIGGS, *auxiliaries),
        kernels,
        auxiliary_particles=auxiliaries,
        model_symbols=_MODEL_SYMBOLS,
    )
    projected = {
        particle.name for particle in auxiliaries if particle.component_dimension == 16
    }
    assert len(projected) == 9
    assert all(
        particle.component_dimension == 6
        for particle in particles
        if particle.name in projected
    )
    for old, new in zip(kernels, reduced, strict=True):
        assert replace(new, component_expressions=old.component_expressions) == old
        if old.particles[2] not in projected:
            continue
        expected = tuple(
            _sym.E(old.component_expressions[4 * left + right])
            - _sym.E(old.component_expressions[4 * right + left])
            for left in range(4)
            for right in range(left + 1, 4)
        )
        assert all(
            (value - _sym.E(actual)).expand().to_canonical_string() == "0"
            for value, actual in zip(expected, new.component_expressions, strict=True)
        )


@pytest.mark.parametrize(
    "term",
    (
        _term(400, color="f(1,2,3)", lorentz="Metric(1,2)", valence=5),
        _term(
            401,
            color="f(-1,1,2)*f(3,4,-2)",
            lorentz="Metric(1,2)*Metric(3,4)",
            valence=5,
        ),
    ),
)
def test_other_colored_contact_topologies_fail_closed(term: CompiledVertexTerm) -> None:
    auxiliaries, kernels = _compile(term)
    assert auxiliaries == ()
    assert kernels == ()
