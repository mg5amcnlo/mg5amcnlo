# SPDX-License-Identifier: 0BSD
"""Binary64 metadata must not round exact symbolic kernel coefficients."""

from fractions import Fraction
from types import SimpleNamespace

import pytest

from pyamplicol.generation import stage_expressions
from pyamplicol.generation.stage_parameters import _apply_contraction_coefficient
from pyamplicol.models.expressions import _exact_complex_expression


def _expected(value):
    from symbolica import E

    return E(str(Fraction(value.real))) + E("1i") * E(str(Fraction(value.imag)))


@pytest.mark.parametrize("value", (0j, 1 + 0j, -1 + 0j, 1j, -1j, 0.1 + 0.2j))
def test_exact_complex_weight_preserves_stored_binary64_value(value):
    from symbolica import E

    exact = _exact_complex_expression(value)
    assert (exact - _expected(value)).expand() == E("0")
    assert _exact_complex_expression(value) is exact
    if value == 0.1 + 0.2j:
        assert exact != E("1/10+1i/5")


@pytest.mark.parametrize("weight", (-1 + 0j, 1j, -1j, 0.1 + 0.2j))
@pytest.mark.parametrize("source", ("x/3", "x/3+y/6", "(x/3+y/6)*z"))
def test_root_and_contraction_weights_keep_rational_coefficients(weight, source):
    from symbolica import E

    expression = E(source)
    expected = _expected(weight) * expression
    pair = (weight.real, weight.imag)
    for actual in (
        stage_expressions._apply_amplitude_color_weight(pair, expression),
        _apply_contraction_coefficient(pair, expression),
    ):
        assert (actual - expected).expand() == E("0")
    numeric = _apply_contraction_coefficient(pair, 2.0)
    assert isinstance(numeric, complex | float)
    assert numeric == weight * 2.0


def _fixture(monkeypatch, *, factor, weight):
    from symbolica import E

    currents = tuple(
        SimpleNamespace(
            index=SimpleNamespace(particle_id=1, chirality=0, momentum_mask=index),
            dimension=1,
        )
        for index in range(3)
    )
    interaction = SimpleNamespace(
        evaluation_group_id=7,
        evaluation_factor=(factor.real, factor.imag),
        color_weight=(weight.real, weight.imag),
        left_id=0,
        right_id=1,
        result_id=2,
        vertex_kind=1,
        vertex_particles=(1, 1, 1),
        coupling=(1.0, 0.0),
    )
    dag = SimpleNamespace(currents=currents, interactions=(interaction,))
    model = SimpleNamespace(
        vertex_component_expression=lambda *args, **kwargs: (E("x/3"),)
    )
    monkeypatch.setattr(
        stage_expressions, "_runtime_coupling_parameter_names", lambda *a, **kw: ()
    )
    return dag, model


@pytest.mark.parametrize("factor", (-1 + 0j, 1j, 0.1 + 0.2j))
def test_compact_reused_evaluation_keeps_exact_normalization(monkeypatch, factor):
    from symbolica import E

    weight = 0.3 + 0.4j
    dag, model = _fixture(monkeypatch, factor=factor, weight=weight)
    cache = {}
    arguments = dict(
        value_components_by_slot_id={0: (), 1: ()},
        input_value_slot_by_current_id={0: 0, 1: 1},
        momentum_components_by_slot_id={0: (), 1: ()},
        momentum_slot_by_mask={0: 0, 1: 1},
        model_parameter_symbols={},
        coupling_cache={},
        evaluation_cache=cache,
    )
    actual = stage_expressions._compact_interaction_contribution(
        dag, model, 0, **arguments
    )
    assert (actual[0] - _expected(weight) * E("x/3")).expand() == E("0")
    assert (cache[7][0] - E("x/3") / _expected(factor)).expand() == E("0")
    # A shared cached kernel must multiply the two exact factors separately,
    # without first rounding their product as a Python complex number.
    cache[7] = (E("x/3"),)
    actual = stage_expressions._compact_interaction_contribution(
        dag, model, 0, **arguments
    )
    expected = _expected(weight) * _expected(factor) * E("x/3")
    assert (actual[0] - expected).expand() == E("0")


def test_legacy_interaction_color_weight_keeps_rational_coefficients(monkeypatch):
    from symbolica import E

    dag, model = _fixture(monkeypatch, factor=1 + 0j, weight=-1 + 0j)
    interaction = {
        "left_value_slot": {"value_slot_id": 0},
        "right_value_slot": {"value_slot_id": 1},
        "momentum_slots": {"left": 0, "right": 1},
        "left_current_id": 0,
        "right_current_id": 1,
        "result_current_id": 2,
        "vertex_kind": 1,
        "coupling": (1.0, 0.0),
        "color_weight": (-1.0, 0.0),
    }
    result = stage_expressions._interaction_contribution(
        dag,
        model,
        interaction,
        value_components_by_slot_id={0: (), 1: ()},
        momentum_components_by_slot_id={0: (), 1: ()},
        model_parameter_symbols={},
    )
    assert (result[0] + E("x/3")).expand() == E("0")
