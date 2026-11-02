# SPDX-License-Identifier: 0BSD
"""Process-wide lazy Symbolica namespace for model compilation."""

from __future__ import annotations

import math
from threading import RLock
from typing import Any

E: Any = None
S: Any = None
Expression: Any = None
Replacement: Any = None
Tensor: Any = None
Representation: Any = None
TensorLibrary: Any = None
TensorName: Any = None
TensorNetwork: Any = None
as_tensor: Any = None
_SYMBOLICA_READY = False
_SYMBOLICA_LOCK = RLock()


def _ensure_symbolica() -> None:
    global E, S, Expression, Replacement, Tensor, Representation
    global TensorLibrary, TensorName, TensorNetwork, as_tensor, _SYMBOLICA_READY

    if _SYMBOLICA_READY:
        return
    with _SYMBOLICA_LOCK:
        if _SYMBOLICA_READY:
            return
        from symbolica import E as expression_parser
        from symbolica import Expression as expression_type
        from symbolica import Replacement as replacement_type
        from symbolica import S as symbol
        from symbolica.community.tensor import Representation as representation_type
        from symbolica.community.tensor import Tensor as tensor_type
        from symbolica.community.tensor import TensorLibrary as tensor_library_type
        from symbolica.community.tensor import TensorName as tensor_name_type
        from symbolica.community.tensor import TensorNetwork as tensor_network_type
        from symbolica.community.tensor import as_tensor as tensor_expression

        E = expression_parser
        S = symbol
        Expression = expression_type
        Replacement = replacement_type
        Tensor = tensor_type
        Representation = representation_type
        TensorLibrary = tensor_library_type
        TensorName = tensor_name_type
        TensorNetwork = tensor_network_type
        as_tensor = tensor_expression
        _SYMBOLICA_READY = True


def _exact_binary64_coefficients(expression: Any) -> Any:
    """Replace binary64 atoms by rationals with exactly their stored value.

    Literal matching preserves the numeric kind and precision: rational and
    higher-precision atoms must never be rounded through Python's ``complex``.
    """
    from symbolica import AtomType

    _ensure_symbolica()
    pending = [expression]
    replacements: dict[Any, Any] = {}
    while pending:
        atom = pending.pop()
        kind = atom.get_type()
        if kind == AtomType.Var:
            continue
        if kind != AtomType.Num:
            pending.extend(atom)
            continue
        try:
            value = complex(atom)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            continue
        if not bool(atom.matches(Expression.num(value))):
            continue
        real_n, real_d = value.real.as_integer_ratio()
        imag_n, imag_d = value.imag.as_integer_ratio()
        replacements[atom] = E(f"({real_n}/{real_d})+1i*({imag_n}/{imag_d})")
    if not replacements:
        return expression
    return expression.replace_multiple(
        [Replacement(source, target) for source, target in replacements.items()]
    )
