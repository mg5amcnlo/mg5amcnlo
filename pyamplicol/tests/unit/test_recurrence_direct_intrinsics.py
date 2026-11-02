# SPDX-License-Identifier: 0BSD
# ruff: noqa: RUF001

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from pyamplicol.models.recurrence_direct_intrinsics import (
    _FINALIZATION_WITNESSES,
    RECURRENCE_INTRINSIC_SCALE_KIND,
    _exact_binary64_coefficients,
    _normalized_expressions,
    certify_recurrence_contribution_intrinsic,
    certify_recurrence_finalization_intrinsic,
)
from pyamplicol.models.recurrence_template import ExactComplexRationalV1


@pytest.mark.parametrize("literal", ("-1.0", "1.0", "-2.0", "2.0", "1.0i", "-1.0i"))
def test_certification_normalizes_floating_units_exactly(literal: str) -> None:
    from symbolica import E

    expression = E(f"({literal})*sqrt(1/2)*x")
    normalized = _exact_binary64_coefficients(expression)
    integer_literal = literal.replace(".0", "")
    assert bool(normalized == E(f"({integer_literal})*sqrt(1/2)*x"))
    # The retained expression used by exact evaluation is never rewritten.
    assert bool(expression == E(f"({literal})*sqrt(1/2)*x"))


def test_certification_preserves_the_exact_binary64_value() -> None:
    from symbolica import E

    numerator, denominator = (0.1).as_integer_ratio()
    normalized = _exact_binary64_coefficients(E("0.1*x"))
    assert bool(normalized == E(f"({numerator}/{denominator})*x"))
    assert not bool(normalized == E("x/10"))
    assert bool(_exact_binary64_coefficients(E("x/3")) == E("x/3"))


def test_certification_preserves_exactly_representable_high_precision_literal() -> None:
    from symbolica import Expression, S

    coefficient = Expression.num(Decimal("0.5000000000000000000000000000000000"))
    expression = coefficient * S("literal_kind_x")
    normalized = _exact_binary64_coefficients(expression)

    # Numeric equality with binary64 0.5 is insufficient: its declared precision
    # must survive, and it must not become the exact rational 1/2 here.
    assert normalized.to_canonical_string() == expression.to_canonical_string()
    assert bool(normalized.matches(expression))


@pytest.mark.parametrize("coefficient", ("-1.0", "-1.000000000000001", "(-1+1/10^60)"))
def test_radical_intrinsic_matching_does_not_round_away_coefficient_changes(
    coefficient: str,
) -> None:
    expressions = (
        "sqrt(1/2)*(-1.0i*l0*r3-1i*l1*r1+l1*r2+1i*l0*r0)",
        f"sqrt(1/2)*(({coefficient})*l0*r2-1i*l0*r1+1i*l1*r0+1i*l1*r3)",
    )
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(_substitute(item) for item in expressions),
        input_contracts=_contracts(2, 4),
        parent_component_counts=(2, 4),
        destination_component_count=2,
        binding_coupling=None,
    )
    if coefficient != "-1.0":
        assert result is None
    else:
        assert result is not None
        assert result.runtime_template.endswith("weyl-vector-to-weyl-a.v1")
        assert result.constant_scale == pytest.approx(2**-0.5)


@pytest.mark.parametrize("precision", (32, 100))
def test_intrinsic_certification_retains_high_precision_radical(precision: int) -> None:
    from symbolica import E, Evaluator

    source = E("-1.0*sqrt(1/2)")
    exact_evaluator = source.evaluator([], jit_compile=False)
    serialized = exact_evaluator.save()
    _exact_binary64_coefficients(source)
    assert exact_evaluator.save() == serialized
    value, imaginary = (
        Evaluator.load(serialized)
        .evaluate_complex_with_prec((), precision)[0]
        .to_decimal_tuple()
    )
    with localcontext() as context:
        context.prec = precision
        expected = -(Decimal(1) / Decimal(2)).sqrt()
        assert abs(value - expected) < Decimal(10) ** (2 - precision)
    assert imaginary == 0


def _contracts(
    left_components: int,
    right_components: int,
    *,
    parameter_index: int | None = None,
    coupling: bool = False,
) -> tuple[str, ...]:
    values: list[dict[str, object]] = []
    for role, count, prefix in (
        ("left-current", left_components, "left"),
        ("right-current", right_components, "right"),
    ):
        values.extend(
            {
                "component": component,
                "role": role,
                "symbol": f"model::prepared::{prefix}_{component}",
            }
            for component in range(count)
        )
    if parameter_index is not None:
        values.append(
            {
                "component": 0,
                "model_parameter_index": parameter_index,
                "role": "model-parameter",
                "symbol": "model::prepared::parameter",
            }
        )
    if coupling:
        values.extend(
            (
                {
                    "component": 0,
                    "role": "coupling-real",
                    "symbol": "model::prepared::coupling_re",
                },
                {
                    "component": 0,
                    "role": "coupling-imag",
                    "symbol": "model::prepared::coupling_im",
                },
            )
        )
    return tuple(
        json.dumps(
            item,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for item in values
    )


def _substitute(expression: str) -> str:
    result = expression
    for side, count in (("l", 6), ("r", 4)):
        prefix = "left" if side == "l" else "right"
        for component in range(count):
            result = result.replace(
                f"{side}{component}",
                f"model::prepared::{prefix}_{component}",
            )
    return result


def _reversed_contracts(
    canonical_left_components: int,
    canonical_right_components: int,
    *,
    parameter_indexes: tuple[int, ...] = (),
) -> tuple[str, ...]:
    values: list[dict[str, object]] = []
    for role, count, prefix in (
        ("left-current", canonical_right_components, "vector"),
        ("right-current", canonical_left_components, "weyl"),
    ):
        values.extend(
            {
                "component": component,
                "role": role,
                "symbol": f"ufo::prepared::{prefix}_{component}",
            }
            for component in range(count)
        )
    values.extend(
        {
            "component": 0,
            "model_parameter_index": parameter_index,
            "role": "model-parameter",
            "symbol": f"ufo::prepared::parameter_{parameter_index}",
        }
        for parameter_index in parameter_indexes
    )
    return tuple(
        json.dumps(
            item,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for item in values
    )


def _substitute_reversed(expression: str) -> str:
    result = expression
    for component in range(2):
        result = result.replace(
            f"l{component}",
            f"ufo::prepared::weyl_{component}",
        )
    for component in range(4):
        result = result.replace(
            f"r{component}",
            f"ufo::prepared::vector_{component}",
        )
    return result


def _three_vector_contracts() -> tuple[str, ...]:
    values = [
        {
            "component": component,
            "role": role,
            "symbol": f"model::prepared::{prefix}_{component}",
        }
        for role, prefix in (
            ("left-current", "left"),
            ("right-current", "right"),
            ("left-momentum", "left_momentum"),
            ("right-momentum", "right_momentum"),
        )
        for component in range(4)
    ]
    return tuple(
        json.dumps(
            item,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for item in values
    )


def _substitute_three_vector(expression: str) -> str:
    result = expression
    for symbol, prefix in (
        ("l", "left"),
        ("r", "right"),
        ("p", "left_momentum"),
        ("q", "right_momentum"),
    ):
        for component in range(4):
            result = result.replace(
                f"{symbol}{component}",
                f"model::prepared::{prefix}_{component}",
            )
    return result


def _finalization_contracts(components: int) -> tuple[str, ...]:
    values = [
        {
            "component": component,
            "role": "current",
            "symbol": f"model::prepared::current_{component}",
        }
        for component in range(components)
    ]
    values.extend(
        {
            "component": component,
            "role": "momentum",
            "symbol": f"model::prepared::momentum_{component}",
        }
        for component in range(4)
    )
    return tuple(
        json.dumps(
            item,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for item in values
    )


def _substitute_finalization(expression: str, components: int) -> str:
    result = expression
    for component in range(components):
        result = result.replace(
            f"l{component}",
            f"model::prepared::current_{component}",
        )
    for component in range(4):
        result = result.replace(
            f"p{component}",
            f"model::prepared::momentum_{component}",
        )
    return result


@pytest.mark.parametrize(
    ("expressions", "shape", "destination", "expected_template", "expected_scale"),
    (
        (
            (
                "-1𝑖*l0*r3-1𝑖*l1*r1+l1*r2+1𝑖*l0*r0",
                "-l0*r2-1𝑖*l0*r1+1𝑖*l1*r0+1𝑖*l1*r3",
            ),
            (2, 4),
            2,
            "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1",
            1.0 + 0.0j,
        ),
        (
            (
                "-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1",
                "-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0",
            ),
            (2, 4),
            2,
            "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-b.v1",
            1.0 + 0.0j,
        ),
        (
            (
                "1𝑖/2*(l0*r1+l1*r2+l2*r3)",
                "1𝑖/2*(l0*r0+l3*r2+l4*r3)",
                "1𝑖/2*(-l3*r1+l1*r0+l5*r3)",
                "1𝑖/2*(-l4*r1-l5*r2+l2*r0)",
            ),
            (6, 4),
            4,
            "rusticol.recurrence-intrinsic.antisymmetric-tensor-vector.v1",
            0.0 + 0.5j,
        ),
        (
            (
                "-l1*r0+l0*r1",
                "-l2*r0+l0*r2",
                "-l3*r0+l0*r3",
                "-l2*r1+l1*r2",
                "-l3*r1+l1*r3",
                "-l3*r2+l2*r3",
            ),
            (4, 4),
            6,
            "rusticol.recurrence-intrinsic.vector-wedge-vector.v1",
            1.0 + 0.0j,
        ),
    ),
)
def test_certifies_model_namespace_independent_intrinsics(
    expressions: tuple[str, ...],
    shape: tuple[int, int],
    destination: int,
    expected_template: str,
    expected_scale: complex,
) -> None:
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(_substitute(item) for item in expressions),
        input_contracts=_contracts(*shape),
        parent_component_counts=shape,
        destination_component_count=destination,
        binding_coupling=None,
    )

    assert result is not None
    assert result.runtime_template == expected_template
    assert result.constant_scale == expected_scale
    assert result.model_parameter_index is None
    assert result.parent_permutation == (0, 1)
    assert result.scale_projection()["kind"] == RECURRENCE_INTRINSIC_SCALE_KIND


@pytest.mark.parametrize(
    ("base", "expected_template"),
    (
        (
            (
                "-1𝑖*l0*r3-1𝑖*l1*r1+l1*r2+1𝑖*l0*r0",
                "-l0*r2-1𝑖*l0*r1+1𝑖*l1*r0+1𝑖*l1*r3",
            ),
            "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1",
        ),
        (
            (
                "-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1",
                "-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0",
            ),
            "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-b.v1",
        ),
    ),
)
def test_built_in_and_ufo_parent_orders_certify_the_same_weyl_family(
    base: tuple[str, str],
    expected_template: str,
) -> None:
    built_in = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(
            f"-1𝑖*model::prepared::parameter*({_substitute(item)})" for item in base
        ),
        input_contracts=_contracts(2, 4, parameter_index=17),
        parent_component_counts=(2, 4),
        destination_component_count=2,
        binding_coupling=None,
    )
    ufo = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(
            f"-1𝑖*ufo::prepared::parameter_91*({_substitute_reversed(item)})"
            for item in base
        ),
        input_contracts=_reversed_contracts(
            2,
            4,
            parameter_indexes=(91,),
        ),
        parent_component_counts=(4, 2),
        destination_component_count=2,
        binding_coupling=None,
        allow_nontrivial_parent_permutation=True,
    )

    assert built_in is not None
    assert ufo is not None
    assert built_in.runtime_template == expected_template
    assert ufo.runtime_template == expected_template
    assert built_in.contract_digest == ufo.contract_digest
    assert built_in.constant_scale == ufo.constant_scale == -1.0j
    assert built_in.parent_permutation == (0, 1)
    assert ufo.parent_permutation == (1, 0)
    assert built_in.model_parameter_index == 17
    assert ufo.model_parameter_index == 91


def test_reversed_parent_intrinsic_is_fail_closed_without_permutation_support() -> None:
    base = (
        "-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1",
        "-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0",
    )

    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(_substitute_reversed(item) for item in base),
        input_contracts=_reversed_contracts(2, 4),
        parent_component_counts=(4, 2),
        destination_component_count=2,
        binding_coupling=None,
    )

    assert result is None


def test_rejects_reversed_weyl_family_with_two_independent_parameters() -> None:
    base = (
        "-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1",
        "-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0",
    )
    scale = "(ufo::prepared::parameter_91+ufo::prepared::parameter_92)"

    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(
            f"{scale}*({_substitute_reversed(item)})" for item in base
        ),
        input_contracts=_reversed_contracts(
            2,
            4,
            parameter_indexes=(91, 92),
        ),
        parent_component_counts=(4, 2),
        destination_component_count=2,
        binding_coupling=None,
        allow_nontrivial_parent_permutation=True,
    )

    assert result is None


def test_rejects_noncontiguous_reversed_parent_basis() -> None:
    contracts = [json.loads(item) for item in _reversed_contracts(2, 4)]
    contracts = [
        item
        for item in contracts
        if not (item["role"] == "left-current" and item["component"] == 2)
    ]

    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=(
            _substitute_reversed("-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1"),
            _substitute_reversed("-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0"),
        ),
        input_contracts=tuple(
            json.dumps(
                item,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            for item in contracts
        ),
        parent_component_counts=(4, 2),
        destination_component_count=2,
        binding_coupling=None,
        allow_nontrivial_parent_permutation=True,
    )

    assert result is None


def test_certifies_one_model_parameter_scale() -> None:
    base = (
        "-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1",
        "-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0",
    )
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(
            f"-2*model::prepared::parameter*({_substitute(item)})" for item in base
        ),
        input_contracts=_contracts(2, 4, parameter_index=17),
        parent_component_counts=(2, 4),
        destination_component_count=2,
        binding_coupling=None,
    )

    assert result is not None
    assert result.constant_scale == -2.0 + 0.0j
    assert result.model_parameter_index == 17


def test_certifies_color_ordered_three_vector_with_momentum_operands() -> None:
    dot_lr = "(l0*r0-l1*r1-l2*r2-l3*r3)"
    dot_lq = "(l0*q0-l1*q1-l2*q2-l3*q3)"
    dot_rp = "(r0*p0-r1*p1-r2*p2-r3*p3)"
    expressions = tuple(
        _substitute_three_vector(
            f"3𝑖/2*({dot_lr}*(p{component}-q{component})"
            f"+2*({dot_lq}*r{component}-{dot_rp}*l{component}))"
        )
        for component in range(4)
    )

    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=expressions,
        input_contracts=_three_vector_contracts(),
        parent_component_counts=(4, 4),
        destination_component_count=4,
        binding_coupling=None,
    )

    assert result is not None
    assert (
        result.runtime_template
        == "rusticol.recurrence-intrinsic.color-ordered-three-vector.v1"
    )
    assert result.constant_scale == 0.0 + 1.5j
    assert result.model_parameter_index is None


def test_substitutes_exact_binding_coupling_before_certification() -> None:
    base = (
        "-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1",
        "-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0",
    )
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(
            f"model::prepared::coupling_re*({_substitute(item)})" for item in base
        ),
        input_contracts=_contracts(2, 4, coupling=True),
        parent_component_counts=(2, 4),
        destination_component_count=2,
        binding_coupling=ExactComplexRationalV1.from_fractions(
            Fraction(3, 2), Fraction(0)
        ),
    )

    assert result is not None
    assert result.constant_scale == 1.5 + 0.0j


def test_rejects_near_match_with_extra_tensor_term() -> None:
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=(
            _substitute("-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1+l0*r2"),
            _substitute("-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0"),
        ),
        input_contracts=_contracts(2, 4),
        parent_component_counts=(2, 4),
        destination_component_count=2,
        binding_coupling=None,
    )

    assert result is None


@pytest.mark.parametrize(
    ("expressions", "components", "expected_template"),
    (
        (
            (
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)"
                "*(-1𝑖*l0*p3-1𝑖*l1*p1+l1*p2+1𝑖*l0*p0)",
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)"
                "*(-l0*p2-1𝑖*l0*p1+1𝑖*l1*p0+1𝑖*l1*p3)",
            ),
            2,
            "rusticol.recurrence-intrinsic.weyl-propagator-a.v1",
        ),
        (
            (
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)"
                "*(-l1*p2+1𝑖*l0*p0+1𝑖*l0*p3+1𝑖*l1*p1)",
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)"
                "*(-1𝑖*l1*p3+l0*p2+1𝑖*l0*p1+1𝑖*l1*p0)",
            ),
            2,
            "rusticol.recurrence-intrinsic.weyl-propagator-b.v1",
        ),
        (
            (
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)*-1𝑖*l0",
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)*-1𝑖*l1",
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)*-1𝑖*l2",
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)*-1𝑖*l3",
            ),
            4,
            "rusticol.recurrence-intrinsic.vector-propagator-feynman.v1",
        ),
    ),
)
@pytest.mark.parametrize("floating_units", (False, True))
def test_certifies_finalization_intrinsics(
    expressions: tuple[str, ...],
    components: int,
    expected_template: str,
    floating_units: bool,
) -> None:
    if floating_units:
        expressions = tuple(item.replace("-1", "-1.0") for item in expressions)
    result = certify_recurrence_finalization_intrinsic(
        exact_expressions=tuple(
            _substitute_finalization(item, components) for item in expressions
        ),
        input_contracts=_finalization_contracts(components),
        component_count=components,
    )

    assert result == expected_template
    if "weyl-propagator" in expected_template:
        from symbolica import E

        # Unlike contribution intrinsics, these native finalizers have no
        # extra scale input: the whole expression must equal the primitive.
        normalized, _ = _normalized_expressions(
            tuple(_substitute_finalization(item, components) for item in expressions),
            _finalization_contracts(components),
            binding_coupling=None,
        )
        witness = next(
            item
            for item in _FINALIZATION_WITNESSES
            if item.runtime_template == expected_template
        )
        assert all(
            bool(actual == E(reference).expand())
            for actual, reference in zip(normalized, witness.expressions, strict=True)
        )
