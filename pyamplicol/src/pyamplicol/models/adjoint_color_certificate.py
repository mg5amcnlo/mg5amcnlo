# SPDX-License-Identifier: 0BSD
"""Colour-only certificates for a connected DDM tree with singlet spectators.

This is not a Yang--Mills helicity theorem.  Arbitrary Lorentz interactions
are permitted when their *actual compiled colour tensors* are connected trees
of structure constants and adjoint identities.  A colourless propagator
between two such trees would instead produce a colour forest.  Until a
forest decomposition is selected, the proof therefore permits at most one
mixed coloured/singlet vertex in a complete tree amplitude.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING

from .tensors import normalize_color_expression

if TYPE_CHECKING:
    from ..processes.ir import CanonicalProcessIR
    from .base import Model
    from .contracts import CompiledModelIR


def adjoint_tree_color_basis_is_proven(
    model: Model,
    process: CanonicalProcessIR,
    *,
    max_coupling_orders: Mapping[str, int] | None = None,
) -> bool:
    """Dispatch the model-owned connected-adjoint-tree certificate."""

    return model.adjoint_tree_color_basis_is_proven(
        process, max_coupling_orders=max_coupling_orders
    )


def compiled_adjoint_tree_color_basis_is_proven(
    ir: CompiledModelIR,
    process: CanonicalProcessIR,
    *,
    max_coupling_orders: Mapping[str, int] | None = None,
) -> bool:
    """Conservatively prove connected f-tree colour for this process domain.

    Species closure deliberately overestimates reachable vertices; no colour
    tensor can be discarded by a numerical probe or by a UFO/model name.
    A common positive order with a one-insertion budget proves that two
    mixed vertices cannot occur.  For example HIG <= 1 certifies a single
    effective Higgs insertion, while unrestricted Higgs exchange is rejected.
    """

    if (
        len(process.adjoint_labels) < 2
        or process.fundamental_labels
        or process.antifundamental_labels
        or len(process.adjoint_labels) + len(process.singlet_labels)
        != len(process.legs)
    ):
        return False
    particles = {particle.name: particle for particle in ir.particles}
    by_pdg = {particle.pdg_code: particle for particle in ir.particles}
    reachable: set[str] = set()
    for leg in process.legs:
        if leg.outgoing_pdg is None or leg.outgoing_pdg not in by_pdg:
            return False
        reachable.add(by_pdg[leg.outgoing_pdg].name)
    limits = {
        str(name).upper(): int(value)
        for name, value in (max_coupling_orders or {}).items()
    }
    active = {}
    changed = True
    while changed:
        changed = False
        for term in ir.vertex_terms:
            orders = {name.upper(): value for name, value in term.coupling_orders}
            # Nonnegative coupling orders make per-vertex pruning sound.
            if any(value < 0 for value in orders.values()):
                return False
            if any(orders.get(name, 0) > limit for name, limit in limits.items()):
                continue
            for output_index, output_name in enumerate(term.particles):
                if not all(
                    name in reachable
                    for index, name in enumerate(term.particles)
                    if index != output_index
                ):
                    continue
                active[term.id] = term
                result = particles[output_name].antiname
                if result not in reachable:
                    reachable.add(result)
                    changed = True
    mixed_orders = []
    found_coloured_tree = False
    for term in active.values():
        representations = tuple(particles[name].color for name in term.particles)
        coloured_count = sum(color != 1 for color in representations)
        if not coloured_count:
            continue
        if any(color not in {1, 8} for color in representations):
            return False
        if not _is_connected_adjoint_tree_tensor(
            term.color_source, term.color_expression, representations
        ):
            return False
        found_coloured_tree = True
        if coloured_count != len(representations):
            mixed_orders.append(
                {name.upper(): value for name, value in term.coupling_orders}
            )
    if mixed_orders and not any(
        0 <= limit < 2 * min(orders.get(name, 0) for orders in mixed_orders)
        for name, limit in limits.items()
    ):
        return False
    return found_coloured_tree


@lru_cache(maxsize=512)
def _is_connected_adjoint_tree_tensor(
    source: str, expression: str, representations: tuple[int, ...]
) -> bool:
    """Prove each source monomial is a tree, then bind it to the compiled tensor."""

    coloured_legs = {
        index for index, color in enumerate(representations, start=1) if color != 1
    }
    if len(coloured_legs) < 2 or any(color not in {1, 8} for color in representations):
        return False
    source_python = re.sub(r"UFO::(?:\{\}::)?", "", source)
    try:
        monomials = _tensor_monomials(ast.parse(source_python, mode="eval").body)
    except (SyntaxError, TypeError, ValueError):
        return False
    if not monomials or not all(
        _monomial_is_tree(factors, coloured_legs) for factors in monomials
    ):
        return False
    from . import compiler_symbolica as _sym

    _sym._ensure_symbolica()
    try:
        normalized = normalize_color_expression(source, representations).expression
        return (_sym.E(expression) - _sym.E(normalized)).expand() == _sym.E("0")
    except (RuntimeError, TypeError, ValueError):
        return False


def _tensor_monomials(node: ast.AST) -> list[tuple[tuple[int, ...], ...]]:
    """Expand a small UFO colour polynomial, never evaluating Python code."""

    if isinstance(node, ast.Constant) and isinstance(node.value, int | float | complex):
        return [()]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        return _tensor_monomials(node.operand)
    if isinstance(node, ast.BinOp):
        left = _tensor_monomials(node.left)
        right = _tensor_monomials(node.right)
        if isinstance(node.op, ast.Add | ast.Sub):
            return left + right
        if isinstance(node.op, ast.Mult):
            return [(*a, *b) for a in left for b in right]
        if isinstance(node.op, ast.Div) and right == [()]:
            return left
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arity = {"f": 3, "Identity": 2}.get(node.func.id)
        if arity is None or len(node.args) != arity or node.keywords:
            raise ValueError("not a structure constant or adjoint identity")
        labels = tuple(ast.literal_eval(argument) for argument in node.args)
        if any(type(label) is not int or label == 0 for label in labels):
            raise ValueError("colour tensor indices must be nonzero integers")
        return [(labels,)]
    raise ValueError("not a polynomial of structure constants and identities")


def _monomial_is_tree(
    factors: tuple[tuple[int, ...], ...], coloured_legs: set[int]
) -> bool:
    counts = Counter(label for factor in factors for label in factor)
    if {label for label in counts if label > 0} != coloured_legs:
        return False
    if any(count != (1 if label > 0 else 2) for label, count in counts.items()):
        return False
    if sum(label < 0 for label in counts) != len(factors) - 1:
        return False
    reached = {0}
    changed = True
    while changed:
        changed = False
        indices = {label for index in reached for label in factors[index] if label < 0}
        for index, factor in enumerate(factors):
            if index not in reached and indices.intersection(factor):
                reached.add(index)
                changed = True
    return len(reached) == len(factors)
