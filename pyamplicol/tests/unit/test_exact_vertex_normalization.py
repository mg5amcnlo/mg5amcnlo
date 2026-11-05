# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from symbolica import E, Expression, S

from pyamplicol.models import BuiltinSMModel, compiler_symbolica
from pyamplicol.models.builtin.expressions import _quark_vector_weyl_data
from pyamplicol.models.external_catalog import ExternalModelCatalogMixin
from pyamplicol.models.external_kernels import ExternalModelKernelMixin
from pyamplicol.models.prepared_catalog_builder import _vertex_expressions


def test_builtin_cubic_normalization_matches_the_quartic_vertex_exactly() -> None:
    model = BuiltinSMModel()
    zero, one = E("0"), E("1")
    cubic = model.vertex_component_expression(
        0,
        (one, zero, zero, zero),
        (zero, one, zero, zero),
        result_particle_id=21,
        result_chirality=0,
        left_momentum=(zero, zero, zero, zero),
        right_momentum=(E("1/2"), zero, zero, zero),
    )[1]
    quartic = model.vertex_component_expression(
        2,
        (one, zero, zero, zero, zero, zero),
        (zero, one, zero, zero),
        result_particle_id=21,
        result_chirality=0,
    )[0]

    assert (cubic**2 + E("1/2")).expand() == zero
    assert (cubic**2 - E("1i") * quartic).expand().to_canonical_string() == "0"
    assert complex(cubic.evaluate({})) == pytest.approx(1j / math.sqrt(2))


@pytest.mark.parametrize("chirality", (-1, 1))
def test_builtin_quark_tensor_has_exact_colour_normalization(chirality: int) -> None:
    entries = _quark_vector_weyl_data(chirality=chirality)
    nonzero = [value for value in entries if value != 0]
    assert nonzero
    for value in nonzero:
        assert isinstance(value, Expression)
        # Nonzero entries are +/-1/sqrt(2) or +/-i/sqrt(2).
        assert (value**4 - E("1/4")).expand() == E("0")


def test_exact_radical_preserves_prepared_symbolic_coupling_inputs() -> None:
    compiler_symbolica._ensure_symbolica()
    model = BuiltinSMModel()
    vertex = next(
        vertex
        for vertex in model.vertices
        if vertex.kind == 21 and vertex.particles == (11, -11, 22)
    )
    expressions, coupling_inputs = _vertex_expressions(
        model,
        vertex,
        left=tuple(S(f"normalization_left_{index}") for index in range(4)),
        right=tuple(S(f"normalization_right_{index}") for index in range(2)),
        left_chirality=0,
        right_chirality=-1,
        result_chirality=0,
        left_momentum=(0, 0, 0, 0),
        right_momentum=(0, 0, 0, 0),
        coupling_symbols=S(
            "normalization_coupling_left", "normalization_coupling_right"
        ),
        parameter_symbols={},
    )
    assert len(expressions) == 4
    assert coupling_inputs == (0, 1)


class _ExternalNormalizationModel(ExternalModelKernelMixin, ExternalModelCatalogMixin):
    def __init__(self, power: int, structure: str) -> None:
        self.name = "normalization_test"
        self._kernel_coupling_expression_cache = {}
        self._color_projection_cache = {}
        self.kernel = SimpleNamespace(
            kind=0,
            coupling_expression="1",
            lc_color_normalization_power=power,
            color_projection_structure=structure,
            color_projection_coefficient=(0.5, 0.0),
        )

    def _kernel(self, kind: int):
        assert kind == 0
        return self.kernel


@pytest.mark.parametrize("power", (0, 1, 2, 3, 4))
@pytest.mark.parametrize(
    "structure",
    ("adjoint-structure-constant", "adjoint-structure-constant-product", "identity"),
)
def test_external_normalization_is_symbolic_and_counted_once(
    power: int, structure: str
) -> None:
    compiler_symbolica._ensure_symbolica()
    model = _ExternalNormalizationModel(power, structure)
    vertex = SimpleNamespace(kind=0)
    coupling = model._resolved_kernel_coupling_expression(model.kernel, {})
    weight = complex(*model.vertex_color_weight(vertex, color_accuracy="full"))
    phase = (-1j) ** power if structure.startswith("adjoint-") else 1

    # Empty parameter lists must not cause a symbolic constant to be evaluated
    # as binary64 before the high-precision kernel has been constructed.
    assert isinstance(coupling, Expression)
    assert (coupling**2 - E(f"1/{2**power}")).expand() == E("0")
    assert weight == 0.5 * phase
    for color_accuracy in ("lc", "nlc"):
        assert (
            complex(*model.vertex_color_weight(vertex, color_accuracy=color_accuracy))
            == weight
        )

    numeric_coupling = model._resolved_kernel_coupling_expression(
        model.kernel, {}, numeric=True
    )
    assert isinstance(numeric_coupling, complex)
    assert numeric_coupling * weight == pytest.approx(
        0.5 * phase * 2.0 ** (-0.5 * power), rel=2e-15, abs=0
    )
