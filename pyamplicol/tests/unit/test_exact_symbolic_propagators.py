# SPDX-License-Identifier: 0BSD
"""Propagating a rational current must not freeze binary64 roundoff into it."""

from decimal import Decimal
from types import SimpleNamespace

import pytest
from symbolica import E

from pyamplicol.models.expressions import (
    _expr_antifermion_propagator_dirac,
    _expr_antifermion_propagator_weyl,
    _expr_fermion_propagator_dirac,
    _expr_fermion_propagator_weyl,
    _imaginary_unit_for,
)
from pyamplicol.models.external_evaluation import ExternalModelEvaluationMixin
from pyamplicol.models.external_helpers import _expr_spin2_propagator


class _PropagatorProbe(ExternalModelEvaluationMixin):
    """Exercise the ordinary external-model formula without compiling a model."""

    def __init__(self, *, kind, gauge, mass, width=0):
        self._mass = mass
        self._width = width
        self.identity = SimpleNamespace(canonical_id="probe")
        self.propagator = SimpleNamespace(
            identity=self.identity,
            chirality=0,
            applies_propagator=True,
            kind=kind,
            mass_class="massless" if mass == 0 else "massive",
            gauge=gauge,
        )

    def _propagator_ir(self, *_):
        return self.propagator

    def _particle_identity_ir(self, *_):
        return self.identity

    def mass(self, *_):
        return self._mass

    def width(self, *_):
        return self._width


@pytest.mark.parametrize(
    "kind,gauge,mass,slot,sign",
    (
        ("scalar", None, 0, 0, 1),
        ("scalar", None, E("m"), 0, 1),
        ("vector", "feynman", 0, 1, -1),
        ("vector", "unitary", E("m"), 1, -1),
    ),
)
def test_external_propagator_preserves_rational_current(kind, gauge, mass, slot, sign):
    model = _PropagatorProbe(kind=kind, gauge=gauge, mass=mass)
    current = [E("0")] * (1 if kind == "scalar" else 4)
    current[slot] = E("x/3")
    actual = model.propagator_component_expression(1, current, (E("p"), 0, 0, 0))
    expected = sign * E("1i*x/3") / (E("p^2") - mass * mass)
    assert (actual[slot] - expected).expand() == E("0")
    assert all(value == E("0") for index, value in enumerate(actual) if index != slot)


@pytest.mark.parametrize("antiparticle", (False, True))
@pytest.mark.parametrize("chirality", (-1, 1))
def test_weyl_propagator_preserves_rational_current(antiparticle, chirality):
    propagate = (
        _expr_antifermion_propagator_weyl
        if antiparticle
        else _expr_fermion_propagator_weyl
    )
    actual = propagate((E("x/3"), E("0")), (E("p"), 0, 0, 0), chirality)
    expected = (-1 if antiparticle else 1) * E("1i*x/(3*p)")
    assert (actual[0] - expected).expand() == E("0")
    assert actual[1] == E("0")


@pytest.mark.parametrize("antiparticle", (False, True))
def test_dirac_propagator_preserves_rational_current(antiparticle):
    propagate = (
        _expr_antifermion_propagator_dirac
        if antiparticle
        else _expr_fermion_propagator_dirac
    )
    actual = propagate((E("x/3"), E("0"), E("0"), E("0")), (E("p"), 0, 0, 0), 0, 0)
    expected = (-1 if antiparticle else 1) * E("1i*x/(3*p)")
    assert (actual[2] - expected).expand() == E("0")
    assert all(value == E("0") for index, value in enumerate(actual) if index != 2)


@pytest.mark.parametrize("massive", (False, True))
def test_spin2_propagator_preserves_rational_projector(massive):
    current = [E("0")] * 16
    current[5] = E("x/3")
    mass = E("m") if massive else 0
    actual = _expr_spin2_propagator(
        current, (E("p"), 0, 0, 0), mass, 0, dimension=4.0, massive=massive
    )
    # For a purely spatial diagonal source, the trace subtraction is 1/2
    # in de Donder gauge and 1/3 in the massive Fierz--Pauli projector.
    expected = E("1i*x") * (E("2/9") if massive else E("1/6"))
    expected /= E("p^2") - mass * mass
    assert (actual[5] - expected).expand() == E("0")


def test_propagator_phase_does_not_promote_numeric_types_to_symbolica():
    assert _imaginary_unit_for((Decimal("0.123"), 1.0, 2j)) == 1j
    # A numeric wrapper need not inherit numbers.Number.
    assert _imaginary_unit_for((SimpleNamespace(value=1),)) == 1j
    model = _PropagatorProbe(kind="scalar", gauge=None, mass=0)
    actual = model.propagator_component_expression(1, (2.0,), (2.0, 0.0, 0.0, 0.0))
    assert actual == (0.5j,)
    assert isinstance(actual[0], complex)
