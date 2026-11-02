# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pyamplicol.models import compiler_symbolica as _sym
from pyamplicol.models.compiler_kernels import _equivalent_component_scale


def test_compact_component_scale_accepts_only_an_exact_sign() -> None:
    _sym._ensure_symbolica()
    compact = (_sym.E("x+1"), _sym.E("2*x-3"))

    positive = _equivalent_component_scale(compact, compact)
    negative = _equivalent_component_scale(
        tuple(-component for component in compact),
        compact,
    )
    float_positive = _equivalent_component_scale(
        (_sym.E("-1.00000000000000*(x+1)"),),
        (_sym.E("-(x+1)"),),
    )
    rescaled = _equivalent_component_scale(
        tuple(2 * component for component in compact),
        compact,
    )
    disjoint = _equivalent_component_scale(
        (_sym.E("1.1*(x+1)"),),
        (_sym.E("y+1"),),
    )

    assert positive == _sym.E("1")
    assert negative == _sym.E("-1")
    assert float_positive == _sym.E("1")
    assert rescaled is None
    assert disjoint is None


def test_disjoint_symbol_support_fails_closed_before_expansion() -> None:
    class FakeSymbol:
        def __init__(self, name: str) -> None:
            self.name = name

        def to_canonical_string(self) -> str:
            return self.name

    class NoExpandExpression:
        def __init__(self, symbol: str) -> None:
            self.symbol = FakeSymbol(symbol)

        def get_all_symbols(self, _include_functions: bool) -> list[FakeSymbol]:
            return [self.symbol]

        def __sub__(self, _other: object) -> object:
            raise AssertionError("disjoint support must return before subtraction")

    assert _equivalent_component_scale(
        (NoExpandExpression("x"),),  # type: ignore[arg-type]
        (NoExpandExpression("y"),),  # type: ignore[arg-type]
    ) is None
