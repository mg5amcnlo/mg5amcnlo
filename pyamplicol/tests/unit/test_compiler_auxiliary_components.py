# SPDX-License-Identifier: 0BSD
"""Exact auxiliary reduction without generating a native process output."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pyamplicol._internal.physics.symbols import symbols
from pyamplicol.models import compiler_symbolica as _sym
from pyamplicol.models.compiler_auxiliary_components import (
    reduce_contact_auxiliary_components,
)
from pyamplicol.models.compiler_tensor_ordering import compile_tensor_ordering_metadata
from pyamplicol.models.contracts import CompiledOrientedKernel, CompiledParticleRecord

_SYMBOLS = symbols.model("auxiliary-component-test")


def _particle(name: str, dimension: int | None = None) -> CompiledParticleRecord:
    auxiliary = dimension is not None
    return CompiledParticleRecord(
        name=name,
        antiname=name,
        pdg_code={"p": 1, "q": 2, "aux": 3, "aux2": 4}[name],
        spin=-1 if auxiliary else 3,
        color=1,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=not auxiliary,
        goldstoneboson=False,
        propagator=None,
        component_dimension=dimension,
        auxiliary_kind="test-contact-tree" if auxiliary else None,
    )


def _kernel(kind: int, names: tuple[str, str, str], *expressions: str):
    _sym._ensure_symbolica()
    replacements = [
        _sym.Replacement(
            _sym.E(f"{letter}{index}"),
            _SYMBOLS.kernel_component(kind, side, index),
        )
        for letter, side in (("L", "left"), ("R", "right"))
        for index in range(8)
    ]
    return CompiledOrientedKernel(
        kind=kind,
        term_id=42,
        vertex="test-contact-tree",
        particles=names,
        source_particle_legs=tuple(
            -1 if name.startswith("aux") else index for index, name in enumerate(names)
        ),
        component_expressions=tuple(
            _sym.E(source).replace_multiple(replacements).to_canonical_string()
            for source in expressions
        ),
        coupling_expression="1",
        coupling_orders=(("HIG", 1),),
        runtime_parameters=(),
        color_source="1",
        color_expression="1",
    )


def _reduce(auxiliaries, kernels):
    return reduce_contact_auxiliary_components(
        (_particle("p"), _particle("q"), *auxiliaries),
        kernels,
        auxiliary_particles=auxiliaries,
        model_symbols=_SYMBOLS,
    )


def _compose(consumer, producers):
    replacements = [
        _sym.Replacement(
            _SYMBOLS.kernel_component(consumer.kind, side, index),
            _sym.E(expression),
        )
        for side, producer in producers.items()
        for index, expression in enumerate(producer.component_expressions)
    ]
    return tuple(
        _sym.E(expression).replace_multiple(replacements)
        for expression in consumer.component_expressions
    )


def _assert_equal(before, after):
    assert len(before) == len(after)
    assert all(
        (left - right).expand().to_canonical_string() == "0"
        for left, right in zip(before, after, strict=True)
    )


def test_signed_columns_reduce_all_producers_and_both_input_orientations():
    aux = _particle("aux", 5)
    producer = _kernel(
        0, ("p", "q", "aux"), "L0*R0", "L0*R1", "L1*R0", "L1*R1", "L2*R2"
    )
    other_producer = _kernel(
        1, ("q", "p", "aux"), "-L1*R0", "L2*R0", "L3*R1", "L0*R2", "L1*R3"
    )
    left = _kernel(2, ("aux", "p", "p"), "(L1-L2)*R0+(L3+L4)*R1", "0", "0", "0")
    right = _kernel(3, ("p", "aux", "p"), "L0*(R1-R2)+L1*(R3+R4)", "0", "0", "0")
    original = (producer, other_producer, left, right)
    particles, reduced = _reduce((aux,), original)

    assert particles[-1].component_dimension == 2
    for old, new in zip(original, reduced, strict=True):
        assert replace(new, component_expressions=old.component_expressions) == old
    assert _sym.E(producer.component_expressions[0]) != _sym.E("0")
    _assert_equal(
        (
            _sym.E(producer.component_expressions[1])
            - _sym.E(producer.component_expressions[2]),
            _sym.E(producer.component_expressions[3])
            + _sym.E(producer.component_expressions[4]),
        ),
        tuple(_sym.E(value) for value in reduced[0].component_expressions),
    )
    for index in (0, 1):
        for consumer, side in ((2, "left"), (3, "right")):
            _assert_equal(
                _compose(original[consumer], {side: original[index]}),
                _compose(reduced[consumer], {side: reduced[index]}),
            )
    _, ordered, orderings, current_orderings = compile_tensor_ordering_metadata(
        (), particles, reduced, (), ()
    )
    current = next(record for record in current_orderings if record.particle == "aux")
    ordering = next(
        record for record in orderings if record.ordering_id == current.ordering_id
    )
    assert ordering.stored_size == 2
    assert current.input_embedding == current.result_projection == (0, 1)
    assert ordered[0].output_ordering_id == current.ordering_id


def test_every_consumer_constrains_the_shared_basis():
    aux = _particle("aux", 3)
    producer = _kernel(0, ("p", "q", "aux"), "L0*R0", "L1*R1", "L2*R2")
    left = _kernel(1, ("aux", "p", "p"), "(L0-L1)*R0", "0", "0", "0")
    right = _kernel(2, ("p", "aux", "p"), "L0*R0+L1*R2", "0", "0", "0")
    particles, reduced = _reduce((aux,), (producer, left, right))
    assert particles[-1] == aux
    assert reduced == (producer, left, right)


@pytest.mark.parametrize(
    ("expression", "dimension"),
    (("(a+b)*L0+(-a-b)*L1", 1), ("a*L0+b*L1", 2)),
)
def test_relations_are_exact_for_all_parameters(expression, dimension):
    aux = _particle("aux", 2)
    producer = _kernel(0, ("p", "q", "aux"), "L0*R0", "L1*R1")
    consumer = _kernel(1, ("aux", "p", "p"), expression, "0", "0", "0")
    particles, reduced = _reduce((aux,), (producer, consumer))
    assert particles[-1].component_dimension == dimension
    _assert_equal(
        _compose(consumer, {"left": producer}),
        _compose(reduced[1], {"left": reduced[0]}),
    )


@pytest.mark.parametrize("coefficient", ("1.0", "0.1"))
def test_binary64_producer_coefficients_keep_their_exact_value(coefficient):
    aux = _particle("aux", 2)
    producer = _kernel(0, ("p", "q", "aux"), "L0", "L1")
    left = _SYMBOLS.kernel_component(0, "left", 0)
    # Retain the original binary64 literal, before canonical string formatting
    # assigns a different precision to a subsequently reparsed Float atom.
    producer = replace(
        producer,
        component_expressions=(
            f"({left.to_canonical_string()}+{coefficient})/3",
            producer.component_expressions[1],
        ),
    )
    consumer = _kernel(1, ("aux", "p", "p"), "L0+L1", "0", "0", "0")
    particles, reduced = _reduce((aux,), (producer, consumer))
    assert particles[-1].component_dimension == 1
    numerator, denominator = float(coefficient).as_integer_ratio()
    expected = (
        left + _sym.E(f"{numerator}/{denominator}")
    ) / 3 + _SYMBOLS.kernel_component(0, "left", 1)
    _assert_equal((expected,), (_sym.E(reduced[0].component_expressions[0]),))


@pytest.mark.parametrize("location", ("producer", "consumer"))
def test_unsupported_float_coefficients_leave_the_whole_basis_unchanged(location):
    aux = _particle("aux", 2)
    producer = _kernel(0, ("p", "q", "aux"), "L0", "L1")
    consumer = _kernel(1, ("aux", "p", "p"), "L0+L1", "0", "0", "0")
    if location == "producer":
        producer = _kernel(
            0, ("p", "q", "aux"), "(L0+0.1000000000000000000000000000000000123)/3", "L1"
        )
    else:
        consumer = _kernel(
            1,
            ("aux", "p", "p"),
            "(0.1000000000000000000000000000000000123+R0/3)*(L0+L1)",
            "0",
            "0",
            "0",
        )
    particles, reduced = _reduce((aux,), (producer, consumer))
    assert particles[-1] == aux
    assert reduced == (producer, consumer)


@pytest.mark.parametrize("sign", (1, -1))
@pytest.mark.parametrize("nearby", (False, True))
def test_arbitrary_precision_units_are_exactified_but_nearby_values_are_not(
    sign, nearby
):
    aux = _particle("aux", 2)
    literal = ("-" if sign < 0 else "") + "1." + "0" * 45 + ("1" if nearby else "0")
    producer = _kernel(0, ("p", "q", "aux"), f"(L0+{literal})/3", "L1")
    consumer = _kernel(1, ("aux", "p", "p"), "L0+L1", "0", "0", "0")
    particles, reduced = _reduce((aux,), (producer, consumer))
    if nearby:
        assert particles[-1] == aux
        assert reduced == (producer, consumer)
    else:
        assert particles[-1].component_dimension == 1
        expected = (
            _SYMBOLS.kernel_component(0, "left", 0) + sign
        ) / 3 + _SYMBOLS.kernel_component(0, "left", 1)
        _assert_equal((expected,), (_sym.E(reduced[0].component_expressions[0]),))


def test_near_equal_consumer_columns_are_not_grouped_by_float_rounding():
    aux = _particle("aux", 2)
    producer = _kernel(0, ("p", "q", "aux"), "L0", "L1")
    left0 = _SYMBOLS.kernel_component(1, "left", 0).to_canonical_string()
    left1 = _SYMBOLS.kernel_component(1, "left", 1).to_canonical_string()
    right0 = _SYMBOLS.kernel_component(1, "right", 0).to_canonical_string()
    consumer = _kernel(1, ("aux", "p", "p"), "L0+L1", "0", "0", "0")
    consumer = replace(
        consumer,
        component_expressions=(
            f"0.1*({right0}*{left0}+({right0}+1/10^60)*{left1})",
            "0",
            "0",
            "0",
        ),
    )
    particles, reduced = _reduce((aux,), (producer, consumer))
    assert particles[-1] == aux
    assert reduced == (producer, consumer)


@pytest.mark.parametrize("sign", (1, -1))
def test_distinct_complex_columns_do_not_depend_on_printed_parentheses(sign):
    aux = _particle("aux", 2)
    producer = _kernel(0, ("p", "q", "aux"), "L0", "L1")
    consumer = _kernel(1, ("aux", "p", "p"), "L0+L1", "0", "0", "0")
    left0 = _SYMBOLS.kernel_component(1, "left", 0).to_canonical_string()
    left1 = _SYMBOLS.kernel_component(1, "left", 1).to_canonical_string()
    # These coefficients are neither equal nor opposite. Some Symbolica
    # canonical printers omit the parentheses around the mixed complex atom,
    # making (1+2i)*a and 1+2i*a indistinguishable as printed strings.
    consumer = replace(
        consumer,
        component_expressions=(
            f"(1+2i)*a*{left0}+({sign})*(1+2i*a)*{left1}",
            "0",
            "0",
            "0",
        ),
    )
    particles, reduced = _reduce((aux,), (producer, consumer))
    assert particles[-1] == aux
    assert reduced == (producer, consumer)


@pytest.mark.parametrize(
    "expression", ("L0^2", "1+L0", "1/L0", "f(L0)", "L0/L1", "f(L0)*L1")
)
def test_uncertified_nonlinear_or_inhomogeneous_consumer_is_unchanged(expression):
    aux = _particle("aux", 2)
    producer = _kernel(0, ("p", "q", "aux"), "L0*R0", "L1*R1")
    consumer = _kernel(1, ("aux", "p", "p"), expression, "0", "0", "0")
    particles, reduced = _reduce((aux,), (producer, consumer))
    assert particles[-1] == aux
    assert reduced == (producer, consumer)


@pytest.mark.parametrize(
    "contract", ("propagating", "propagator", "external", "ordering")
)
def test_nonopaque_or_externally_used_auxiliaries_are_unchanged(contract):
    aux = _particle("aux", 2)
    producer = _kernel(0, ("p", "q", "aux"), "L0*R0", "L1*R1")
    consumer = _kernel(1, ("aux", "p", "p"), "L0*R0", "0", "0", "0")
    if contract == "propagating":
        aux = replace(aux, propagating=True)
    elif contract == "propagator":
        aux = replace(aux, propagator="custom")
    elif contract == "external":
        consumer = replace(consumer, source_particle_legs=(0, 1, 2))
    else:
        producer = replace(producer, output_ordering_id="prescribed")
    particles, reduced = _reduce((aux,), (producer, consumer))
    assert particles[-1] == aux
    assert reduced == (producer, consumer)


def test_two_reduced_inputs_are_replaced_simultaneously():
    auxiliaries = (_particle("aux", 3), _particle("aux2", 3))
    first = _kernel(0, ("p", "q", "aux"), "L0*R0", "L0*R1", "L1*R0")
    second = _kernel(1, ("q", "p", "aux2"), "L2*R2", "L2*R3", "L3*R2")
    consumer = _kernel(2, ("aux", "aux2", "p"), "(L1-L2)*(R1-R2)", "0", "0", "0")
    particles, reduced = _reduce(auxiliaries, (first, second, consumer))
    assert [particle.component_dimension for particle in particles[-2:]] == [1, 1]
    _assert_equal(
        _compose(consumer, {"left": first, "right": second}),
        _compose(reduced[2], {"left": reduced[0], "right": reduced[1]}),
    )


def test_same_auxiliary_on_both_sides_uses_one_shared_reduced_basis():
    aux = _particle("aux", 3)
    first = _kernel(0, ("p", "q", "aux"), "L0*R0", "L0*R1", "L1*R0")
    second = _kernel(1, ("q", "p", "aux"), "L2*R2", "L2*R3", "L3*R2")
    consumer = _kernel(2, ("aux", "aux", "p"), "(L1-L2)*(R1-R2)", "0", "0", "0")
    particles, reduced = _reduce((aux,), (first, second, consumer))
    assert particles[-1].component_dimension == 1
    _assert_equal(
        _compose(consumer, {"left": first, "right": second}),
        _compose(reduced[2], {"left": reduced[0], "right": reduced[1]}),
    )


def test_downstream_reduction_exposes_additional_upstream_dead_components():
    auxiliaries = (_particle("aux", 3), _particle("aux2", 2))
    first = _kernel(0, ("p", "q", "aux"), "L0*R0", "L0*R1", "L1*R0")
    second = _kernel(1, ("aux", "p", "aux2"), "L0*R0", "(L1+L2)*R1")
    final = _kernel(2, ("aux2", "p", "p"), "L1*R2", "0", "0", "0")
    particles, reduced = _reduce(auxiliaries, (first, second, final))
    assert [particle.component_dimension for particle in particles[-2:]] == [1, 1]
    old_middle = replace(
        second,
        component_expressions=tuple(
            value.to_canonical_string() for value in _compose(second, {"left": first})
        ),
    )
    new_middle = replace(
        reduced[1],
        component_expressions=tuple(
            value.to_canonical_string()
            for value in _compose(reduced[1], {"left": reduced[0]})
        ),
    )
    _assert_equal(
        _compose(final, {"left": old_middle}),
        _compose(reduced[2], {"left": new_middle}),
    )


@pytest.mark.parametrize("with_consumer", (False, True))
def test_unobserved_family_does_not_create_a_zero_width_current(with_consumer):
    aux = _particle("aux", 2)
    kernels = (_kernel(0, ("p", "q", "aux"), "L0*R0", "L1*R1"),)
    if with_consumer:
        kernels += (_kernel(1, ("aux", "p", "p"), "0", "0", "0", "0"),)
    particles, reduced = _reduce((aux,), kernels)
    assert particles[-1] == aux
    assert reduced == kernels


def test_no_contact_trees_does_not_start_additional_symbolic_work(monkeypatch):
    particles = (_particle("p"),)

    def unexpected():
        raise AssertionError("no symbolic reduction is needed")

    monkeypatch.setattr(_sym, "_ensure_symbolica", unexpected)
    assert reduce_contact_auxiliary_components(
        particles, (), auxiliary_particles=(), model_symbols=_SYMBOLS
    ) == (particles, ())
