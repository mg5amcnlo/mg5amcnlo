# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import product
from threading import RLock
from typing import Any

from .._internal.physics.symbols import ModelSymbolRegistry, symbols
from .contracts import validate_color_representation

E: Any = None
Expression: Any = None
Representation: Any = None
TensorLibrary: Any = None
TensorName: Any = None
TensorNetwork: Any = None
TensorExpression: Any = None
as_tensor: Any = None
_INDEX_PATTERN: Any = None
_DUMMY_PATTERN: Any = None
_WILDCARDS: dict[str, Any] = {}
_SYMBOLICA_READY = False
_SYMBOLICA_LOCK = RLock()


def _ensure_symbolica() -> None:
    global E, Expression, Representation, TensorLibrary, TensorName
    global TensorNetwork, TensorExpression, as_tensor
    global _INDEX_PATTERN, _DUMMY_PATTERN, _WILDCARDS, _SYMBOLICA_READY

    if _SYMBOLICA_READY:
        return
    with _SYMBOLICA_LOCK:
        if _SYMBOLICA_READY:
            return
        from symbolica import E as expression_parser
        from symbolica import Expression as expression_type
        from symbolica.community.tensor import Representation as representation_type
        from symbolica.community.tensor import (
            TensorExpression as tensor_expression_type,
        )
        from symbolica.community.tensor import TensorLibrary as tensor_library_type
        from symbolica.community.tensor import TensorName as tensor_name_type
        from symbolica.community.tensor import TensorNetwork as tensor_network_type
        from symbolica.community.tensor import as_tensor as tensor_converter

        E = expression_parser
        Expression = expression_type
        Representation = representation_type
        TensorLibrary = tensor_library_type
        TensorName = tensor_name_type
        TensorNetwork = tensor_network_type
        TensorExpression = tensor_expression_type
        as_tensor = tensor_converter
        _INDEX_PATTERN = E("UFO::idx(component_,leg_)")
        _DUMMY_PATTERN = E("UFO::dummy(label_)")
        _WILDCARDS = {name: E(name) for name in ("a_", "b_", "c_", "d_")}
        _SYMBOLICA_READY = True


@dataclass(frozen=True)
class NormalizedTensorExpression:
    source: str
    expression: str
    tensor_heads: tuple[str, ...]


def normalize_lorentz_expression(
    source: str,
    spins: Sequence[int],
    *,
    model_symbols: ModelSymbolRegistry | None = None,
    vertex: bool = True,
) -> NormalizedTensorExpression:
    """Convert one wrapped UFO Lorentz expression into typed spenso form.

    ``vertex`` selects the vertex spinor order; propagators keep their own.
    """

    _ensure_symbolica()
    context = _LorentzContext(
        tuple(int(spin) for spin in spins),
        model_symbols=model_symbols,
        vertex=vertex,
    )
    expression = E(source)
    expression = _replace(expression, "UFO::PSlash(a_,b_)", context.p_slash)
    expression = _replace(expression, "UFO::Gamma(a_,b_,c_)", context.gamma)
    expression = _replace(expression, "UFO::Gamma5(a_,b_)", context.gamma5)
    expression = _replace(expression, "UFO::ProjM(a_,b_)", context.projector_minus)
    expression = _replace(expression, "UFO::ProjP(a_,b_)", context.projector_plus)
    expression = _replace(expression, "UFO::Sigma(a_,b_,c_,d_)", context.sigma)
    expression = _replace(expression, "UFO::Identity(a_,b_)", context.identity)
    expression = _replace(expression, "UFO::Metric(a_,b_)", context.metric)
    expression = _replace(expression, "UFO::P(a_,b_)", context.momentum)
    expression = _replace(expression, "UFO::P(a_)^2", context.momentum_square)
    expression = _replace(expression, "UFO::P(a_)", context.propagator_momentum)
    _reject_residual_ufo_tensors(expression, context="Lorentz expression")
    _reject_indeterminate(expression, context="Lorentz expression")
    expression = as_tensor(expression).simplify_algebra(color=False).to_expression()
    return NormalizedTensorExpression(
        source=source,
        expression=expression.format_plain(),
        tensor_heads=tuple(sorted(_tensor_heads(expression))),
    )


def normalize_color_expression(
    source: str,
    color_representations: Sequence[int],
) -> NormalizedTensorExpression:
    """Convert one UFO color expression into typed spenso SU(3) tensors."""

    representations = tuple(int(color) for color in color_representations)
    _validate_color_representations(representations)
    _ensure_symbolica()
    context = _ColorContext(representations)
    expression = E(source)
    expression = _replace(expression, "UFO::Identity(a_,b_)", context.identity)
    expression = _replace(expression, "UFO::T(a_,b_,c_)", context.generator)
    expression = _replace(expression, "UFO::f(a_,b_,c_)", context.structure_constant)
    expression = _replace(expression, "UFO::d(a_,b_,c_)", context.symmetric_invariant)
    _reject_residual_ufo_tensors(expression, context="color expression")
    _reject_indeterminate(expression, context="color expression")
    simplified = (
        as_tensor(expression)
        .simplify_algebra(
            gamma=False,
            contract="none",
            color_substitute_cof_dimension_invariants=True,
        )
        .to_expression()
    )
    # Keep d^{abc} as its explicit generator trace for component projection,
    # rather than introducing a separate symbolic invariant tensor.
    expression = expression if context.expanded_symmetric_invariant else simplified
    return NormalizedTensorExpression(
        source=source,
        expression=expression.format_plain(),
        tensor_heads=tuple(sorted(_tensor_heads(expression))),
    )


def normalize_vertex_tensor_term(
    *,
    lorentz: str,
    color: str,
    spins: Sequence[int],
    color_representations: Sequence[int],
    model_symbols: ModelSymbolRegistry | None = None,
) -> tuple[NormalizedTensorExpression, NormalizedTensorExpression]:
    return (
        normalize_lorentz_expression(
            lorentz,
            spins,
            model_symbols=model_symbols,
        ),
        normalize_color_expression(color, color_representations),
    )


class _LorentzContext:
    def __init__(
        self,
        spins: tuple[int, ...],
        *,
        model_symbols: ModelSymbolRegistry | None = None,
        vertex: bool = True,
    ) -> None:
        _ensure_symbolica()
        self.spins = spins
        self.model_symbols = model_symbols
        self.minkowski = Representation.mink(4)
        self.bispinor = Representation.bis(4)
        self.gamma_tensor = TensorExpression.dirac_gamma(4)
        self.gamma5_tensor = TensorExpression.gamma5(4)
        self.projm_tensor = TensorExpression.projm(4)
        self.projp_tensor = TensorExpression.projp(4)
        self._fresh_index = 0
        # Standard UFO vertices index a fermion pair in the opposite order to
        # the internal spinor slots, e.g. Gamma(3,2,1) for the SM FFV1.
        dirac_legs = (
            [leg for leg, spin in enumerate(spins, 1) if spin == 2] if vertex else []
        )
        self._spinor_partner = {
            leg: partner
            for first, second in zip(dirac_legs[::2], dirac_legs[1::2], strict=False)
            for leg, partner in ((first, second), (second, first))
        }

    def metric(self, match: Mapping[Expression, Expression]) -> Expression:
        left = self._minkowski_slot(match[_WILDCARDS["a_"]])
        right = self._minkowski_slot(match[_WILDCARDS["b_"]])
        return TensorExpression.g(self.minkowski)(left, right).to_expression()

    def gamma(self, match: Mapping[Expression, Expression]) -> Expression:
        lorentz = self._minkowski_label(match[_WILDCARDS["a_"]])
        output = self._bispinor_label(match[_WILDCARDS["b_"]])
        input_ = self._bispinor_label(match[_WILDCARDS["c_"]])
        return _as_expression(self.gamma_tensor(input_, output, lorentz))

    def gamma5(self, match: Mapping[Expression, Expression]) -> Expression:
        output = self._bispinor_label(match[_WILDCARDS["a_"]])
        input_ = self._bispinor_label(match[_WILDCARDS["b_"]])
        return _as_expression(self.gamma5_tensor(input_, output))

    def projector_minus(
        self,
        match: Mapping[Expression, Expression],
    ) -> Expression:
        output = self._bispinor_label(match[_WILDCARDS["a_"]])
        input_ = self._bispinor_label(match[_WILDCARDS["b_"]])
        return _as_expression(self.projp_tensor(input_, output))

    def projector_plus(
        self,
        match: Mapping[Expression, Expression],
    ) -> Expression:
        output = self._bispinor_label(match[_WILDCARDS["a_"]])
        input_ = self._bispinor_label(match[_WILDCARDS["b_"]])
        return _as_expression(self.projm_tensor(input_, output))

    def identity(self, match: Mapping[Expression, Expression]) -> Expression:
        left = self.bispinor(self._bispinor_label(match[_WILDCARDS["a_"]]))
        right = self.bispinor(self._bispinor_label(match[_WILDCARDS["b_"]]))
        return TensorExpression.g(self.bispinor)(left, right).to_expression()

    def sigma(self, match: Mapping[Expression, Expression]) -> Expression:
        mu = self._minkowski_label(match[_WILDCARDS["a_"]])
        nu = self._minkowski_label(match[_WILDCARDS["b_"]])
        output = self._bispinor_label(match[_WILDCARDS["c_"]])
        input_ = self._bispinor_label(match[_WILDCARDS["d_"]])
        dummy = self._fresh_label("sigma")
        first = _as_expression(self.gamma_tensor(input_, dummy, mu))
        second = _as_expression(self.gamma_tensor(dummy, output, nu))
        reverse_first = _as_expression(self.gamma_tensor(input_, dummy, nu))
        reverse_second = _as_expression(self.gamma_tensor(dummy, output, mu))
        return E("1𝑖/2") * (  # noqa: RUF001
            first * second - reverse_first * reverse_second
        )

    def momentum(self, match: Mapping[Expression, Expression]) -> Expression:
        lorentz_index = match[_WILDCARDS["a_"]]
        momentum_index = match[_WILDCARDS["b_"]]
        _component, leg = _ufo_index(momentum_index)
        return self._momentum_tensor(leg, self._minkowski_slot(lorentz_index))

    def momentum_square(
        self,
        match: Mapping[Expression, Expression],
    ) -> Expression:
        index = match[_WILDCARDS["a_"]]
        _component, leg = _ufo_index(index)
        left = self.minkowski(self._fresh_label("momentum_square"))
        right = self.minkowski(self._fresh_label("momentum_square"))
        return (
            TensorExpression.g(self.minkowski)(left, right).to_expression()
            * self._momentum_tensor(leg, left)
            * self._momentum_tensor(leg, right)
        )

    def propagator_momentum(
        self,
        match: Mapping[Expression, Expression],
    ) -> Expression:
        index = match[_WILDCARDS["a_"]]
        _component, leg = _ufo_index(index)
        return self._momentum_tensor(leg, self._minkowski_slot(index))

    def p_slash(self, match: Mapping[Expression, Expression]) -> Expression:
        output = self._bispinor_label(match[_WILDCARDS["a_"]])
        input_ = self._bispinor_label(match[_WILDCARDS["b_"]])
        lorentz_label = self._fresh_label("pslash")
        lorentz = self.minkowski(lorentz_label)
        return _as_expression(
            self.gamma_tensor(input_, output, lorentz_label)
        ) * self._momentum_tensor(
            1,
            lorentz,
        )

    def _minkowski_slot(self, index: Expression):
        return self.minkowski(self._minkowski_label(index))

    def _minkowski_label(self, index: Expression) -> str:
        return _ufo_lorentz_label(index)

    def _bispinor_label(self, index: Expression) -> str:
        dummy = _ufo_dummy(index)
        if dummy is not None:
            return f"ufo_s_dummy_{dummy}"
        component, leg = _ufo_index(index)
        if leg < 1 or leg > len(self.spins):
            raise ValueError(f"UFO spin index refers to absent leg {leg}")
        if self.spins[leg - 1] != 2:
            raise ValueError(
                f"UFO spin index ({component}, {leg}) refers to spin "
                f"{self.spins[leg - 1]}, not a Dirac fermion"
            )
        if component != 1:
            raise ValueError(f"Dirac particle leg {leg} has no spin index {component}")
        return f"ufo_s_{component}_{self._spinor_partner.get(leg, leg)}"

    def _momentum_tensor(self, leg: int, slot) -> Expression:
        if leg < 1 or leg > len(self.spins):
            raise ValueError(f"UFO momentum refers to absent leg {leg}")
        name = (
            symbols.ufo_momentum_tensor_name(leg)
            if self.model_symbols is None
            else self.model_symbols.ufo_momentum_tensor_name(leg)
        )
        return TensorName(name)(slot).to_expression()

    def _fresh_label(self, prefix: str) -> str:
        label = f"ufo_{prefix}_internal_{self._fresh_index}"
        self._fresh_index += 1
        return label


class _ColorContext:
    def __init__(self, color_representations: tuple[int, ...]) -> None:
        _validate_color_representations(color_representations)
        _ensure_symbolica()
        self.color_representations = color_representations
        self.fundamental = Representation.cof(3)
        self.adjoint = Representation.coad(8)
        self.generator_tensor = TensorExpression.color_t(8, 3)
        self.f_tensor = TensorExpression.color_f(8)
        self.expanded_symmetric_invariant = False
        self._dummy_index = 0

    def identity(self, match: Mapping[Expression, Expression]) -> Expression:
        left = self._leg_slot(match[_WILDCARDS["a_"]])
        right = self._leg_slot(match[_WILDCARDS["b_"]])
        left_leg = int(match[_WILDCARDS["a_"]])
        right_leg = int(match[_WILDCARDS["b_"]])
        left_representation = self.color_representations[left_leg - 1]
        right_representation = self.color_representations[right_leg - 1]
        if left_representation == right_representation:
            # Oriented current kernels can have equal input/output colour roles.
            representation = {
                8: self.adjoint,
                3: self.fundamental,
                -3: self.fundamental.dual(),
            }[left_representation]
            return TensorExpression.g(representation)(left, right).to_expression()
        if left_representation != -right_representation:
            raise ValueError("UFO color identity requires dual representations")
        representation = (
            self.fundamental if right_representation == 3 else self.fundamental.dual()
        )
        return representation.id(
            f"ufo_c_{left_leg}", f"ufo_c_{right_leg}"
        ).to_expression()

    def generator(self, match: Mapping[Expression, Expression]) -> Expression:
        adjoint = self._slot(match[_WILDCARDS["a_"]], expected="adjoint")
        fundamental = self._slot(
            match[_WILDCARDS["b_"]],
            expected="fundamental",
        )
        antifundamental = self._slot(
            match[_WILDCARDS["c_"]],
            expected="antifundamental",
        )
        return self.generator_tensor(
            adjoint,
            fundamental,
            antifundamental,
        ).to_expression()

    def structure_constant(
        self,
        match: Mapping[Expression, Expression],
    ) -> Expression:
        slots = [
            self._slot(match[_WILDCARDS[name]], expected="adjoint")
            for name in ("a_", "b_", "c_")
        ]
        return self.f_tensor(*slots).to_expression()

    def symmetric_invariant(
        self,
        match: Mapping[Expression, Expression],
    ) -> Expression:
        self.expanded_symmetric_invariant = True
        adjoint = [
            self._slot(match[_WILDCARDS[name]], expected="adjoint")
            for name in ("a_", "b_", "c_")
        ]
        suffix = self._dummy_index
        self._dummy_index += 1
        fundamental = [
            self.fundamental(f"ufo_c_d_{suffix}_{name}") for name in ("i", "j", "k")
        ]
        antifundamental = [
            self.fundamental.dual()(f"ufo_c_d_{suffix}_{name}")
            for name in ("i", "j", "k")
        ]

        def generator(a: Any, row: int, column: int) -> Expression:
            return self.generator_tensor(
                a,
                fundamental[row],
                antifundamental[column],
            ).to_expression()

        first = (
            generator(adjoint[0], 0, 1)
            * generator(adjoint[1], 1, 2)
            * generator(adjoint[2], 2, 0)
        )
        second = (
            generator(adjoint[1], 0, 1)
            * generator(adjoint[0], 1, 2)
            * generator(adjoint[2], 2, 0)
        )
        return 2 * (first + second)

    def _leg_slot(self, index: Expression):
        label = int(index)
        if label < 1 or label > len(self.color_representations):
            raise ValueError(f"UFO color index refers to absent leg {label}")
        representation = self.color_representations[label - 1]
        if representation == 3:
            return self.fundamental(f"ufo_c_{label}")
        if representation == -3:
            return self.fundamental.dual()(f"ufo_c_{label}")
        if representation == 8:
            return self.adjoint(f"ufo_c_{label}")
        raise ValueError(
            f"UFO color index {label} refers to color representation {representation}"
        )

    def _slot(self, index: Expression, *, expected: str):
        label = int(index)
        if label > 0:
            slot = self._leg_slot(index)
            actual = self._slot_kind(label)
            if actual != expected:
                raise ValueError(
                    f"UFO color index {label} has type {actual}, expected {expected}"
                )
            return slot
        name = f"ufo_c_dummy_{abs(label)}_{expected}"
        if expected == "adjoint":
            return self.adjoint(name)
        if expected == "fundamental":
            return self.fundamental(name)
        if expected == "antifundamental":
            return self.fundamental.dual()(name)
        raise ValueError(f"unknown color index type {expected}")

    def _slot_kind(self, leg: int) -> str:
        representation = self.color_representations[leg - 1]
        kind = {
            3: "fundamental",
            -3: "antifundamental",
            8: "adjoint",
            1: "singlet",
        }.get(representation)
        if kind is None:
            raise ValueError(
                f"unsupported UFO color representation {representation} on leg {leg}"
            )
        return kind


def _replace(
    expression: Expression,
    pattern: str,
    callback: Callable[[Mapping[Expression, Expression]], Expression],
) -> Expression:
    errors: list[Exception] = []

    def guarded(match: Mapping[Expression, Expression]) -> Expression:
        try:
            return callback(match)
        except Exception as exc:  # Symbolica otherwise converts this to indeterminate.
            errors.append(exc)
            return E("indeterminate")

    result = expression.replace(E(pattern), guarded, bottom_up=True)
    if errors:
        raise errors[0]
    return result


def _ufo_index(expression: Expression) -> tuple[int, int]:
    match = next(
        iter(expression.match(_INDEX_PATTERN, min_level=0, max_level=0, partial=False)),
        None,
    )
    if match is None:
        raise ValueError(f"expected wrapped UFO index, got {expression}")
    values = dict(match)
    return int(values[E("component_")]), int(values[E("leg_")])


def _ufo_lorentz_label(expression: Expression) -> str:
    index_match = next(
        iter(expression.match(_INDEX_PATTERN, min_level=0, max_level=0, partial=False)),
        None,
    )
    if index_match is not None:
        values = dict(index_match)
        return f"ufo_l_{int(values[E('component_')])}_{int(values[E('leg_')])}"
    dummy = _ufo_dummy(expression)
    if dummy is not None:
        return f"ufo_l_dummy_{dummy}"
    raise ValueError(f"expected wrapped UFO Lorentz index, got {expression}")


def _ufo_dummy(expression: Expression) -> int | None:
    dummy_match = next(
        iter(expression.match(_DUMMY_PATTERN, min_level=0, max_level=0, partial=False)),
        None,
    )
    if dummy_match is None:
        return None
    return int(dict(dummy_match)[E("label_")])


def _reject_residual_ufo_tensors(expression: Expression, *, context: str) -> None:
    residual = sorted(
        set(
            re.findall(
                r"UFO::(?:\{[^}]*\}::)?([A-Za-z][A-Za-z0-9_]*)\(",
                expression.to_canonical_string(),
            )
        )
        - {"idx", "dummy"}
    )
    if residual:
        raise ValueError(
            f"{context} contains unsupported UFO tensors: {', '.join(residual)}"
        )


def _reject_indeterminate(expression: Expression, *, context: str) -> None:
    if (
        "¿" in str(expression)
        or "indeterminate" in expression.to_canonical_string().lower()
    ):
        raise ValueError(f"{context} became indeterminate during tensor normalization")


def _tensor_heads(expression: Expression) -> set[str]:
    return set(
        re.findall(
            r"(?:(?:[A-Za-z0-9_]+|\{[^}]*\})::)*([A-Za-z][A-Za-z0-9_]*)\(",
            expression.to_canonical_string(),
        )
    )


def project_trilinear_color_expression(
    expression: str,
    color_representations: Sequence[int],
    *,
    allow_numeric_fallback: bool = False,
) -> tuple[str, complex]:
    """Project a local color tensor onto the supported SU(3) basis.

    The default path accepts only exact Symbolica identities. Dense numerical
    projection is available for diagnostics, but must not certify executable
    model contracts.
    """

    representations = tuple(int(value) for value in color_representations)
    _validate_color_representations(representations)
    if len(representations) != 3:
        return "generic-tensor", 1.0 + 0.0j
    colored_legs = tuple(
        index + 1
        for index, representation in enumerate(representations)
        if abs(representation) != 1
    )
    candidates: list[tuple[str, str]] = []
    if not colored_legs:
        candidates.append(("singlet", "1"))
    elif len(colored_legs) == 2:
        left, right = colored_legs
        left_representation = representations[left - 1]
        right_representation = representations[right - 1]
        if (
            left_representation == right_representation == 8
            or left_representation == -right_representation
        ):
            candidates.append(("color-identity", f"UFO::Identity({left},{right})"))
    elif sorted(representations) == [-3, 3, 8]:
        adjoint = representations.index(8) + 1
        fundamental = representations.index(3) + 1
        antifundamental = representations.index(-3) + 1
        candidates.append(
            (
                "fundamental-generator",
                f"UFO::T({adjoint},{fundamental},{antifundamental})",
            )
        )
    elif all(representation == 8 for representation in representations):
        candidates.extend(
            (
                ("adjoint-structure-constant", "UFO::f(1,2,3)"),
                ("adjoint-symmetric-invariant", "UFO::d(1,2,3)"),
            )
        )
    if not candidates:
        return "generic-tensor", 1.0 + 0.0j

    try:
        normalized_candidates: list[tuple[str, NormalizedTensorExpression]] = []
        for structure, source in candidates:
            normalized = normalize_color_expression(source, representations)
            coefficient = _constant_expression_ratio(
                E(expression),
                E(normalized.expression),
            )
            if coefficient is not None:
                return structure, coefficient
            normalized_candidates.append((structure, normalized))
        if not allow_numeric_fallback:
            return "generic-tensor", 1.0 + 0.0j
        target = _materialized_color_components(expression, colored_legs)
        for structure, normalized in normalized_candidates:
            candidate = _materialized_color_components(
                normalized.expression,
                colored_legs,
            )
            norm = sum(abs(value) ** 2 for value in candidate.values())
            if norm == 0.0:
                continue
            coefficient = (
                sum(
                    candidate.get(key, 0.0j).conjugate() * value
                    for key, value in target.items()
                )
                / norm
            )
            scale = max(
                1.0,
                sum(abs(value) ** 2 for value in target.values()),
                abs(coefficient) ** 2 * norm,
            )
            residual = sum(
                abs(target.get(key, 0.0j) - coefficient * candidate.get(key, 0.0j)) ** 2
                for key in set(target) | set(candidate)
            )
            if residual <= 1.0e-24 * scale:
                return structure, coefficient
    except (RuntimeError, TypeError, ValueError):
        pass
    return "generic-tensor", 1.0 + 0.0j


def classify_trilinear_color_expression(
    expression: str,
    source: str,
    color_representations: Sequence[int],
    *,
    allow_source_fallback: bool = False,
) -> tuple[str, complex]:
    """Resolve a trilinear color tensor through an exact tensor projection.

    Source-text recognition is reserved for compiler-generated contact fragments
    whose color decomposition has already been proven independently.  Arbitrary
    UFO vertices must never acquire a color-flow contract from spelling alone.
    """

    representations = tuple(int(value) for value in color_representations)
    _validate_color_representations(representations)
    return _classify_trilinear_color_expression_cached(
        expression,
        source,
        representations,
        bool(allow_source_fallback),
    )


@cache
def _classify_trilinear_color_expression_cached(
    expression: str,
    source: str,
    color_representations: tuple[int, ...],
    allow_source_fallback: bool,
) -> tuple[str, complex]:
    projected = project_trilinear_color_expression(
        expression,
        color_representations,
    )
    if projected[0] != "generic-tensor":
        return projected
    if not allow_source_fallback:
        return projected

    compact_source = re.sub(r"\s+", "", source)
    tensor = r"(?:UFO::(?:\{\}::)?)?"
    if re.fullmatch(tensor + r"T\([^()]+\)", compact_source):
        return "fundamental-generator", 1.0 + 0.0j
    if re.fullmatch(tensor + r"f\([^()]+\)", compact_source):
        return "adjoint-structure-constant", 1.0 + 0.0j
    factors = compact_source.split("*")
    if len(factors) > 1 and all(
        re.fullmatch(tensor + r"f\([^()]+\)", factor) for factor in factors
    ):
        return "adjoint-structure-constant-product", 1.0 + 0.0j
    if compact_source in {"1", "UFO::{}::1"}:
        return "singlet", 1.0 + 0.0j
    return projected


def _constant_expression_ratio(
    target: Expression,
    candidate: Expression,
) -> complex | None:
    """Return the constant scale between algebraically identical tensors."""

    if candidate == E("0"):
        return None
    ratio = (target / candidate).cancel()
    if ratio.get_all_symbols(False):
        return None
    if (target - ratio * candidate).expand() != E("0"):
        return None
    try:
        return complex(ratio)
    except (RuntimeError, TypeError, ValueError):
        return None


def _materialized_color_components(
    expression: str,
    colored_legs: Sequence[int],
) -> dict[tuple[int, ...], complex]:
    library = TensorLibrary.hep_lib_atom()
    atom = E(expression)
    network = TensorNetwork(as_tensor(atom) if colored_legs else atom, library)
    network.execute(library=library)
    tensor = network.result_tensor(library).to_dense()
    axes = []
    for slot in tensor.axes:
        argument = slot.to_expression()
        match = re.search(r"ufo_c_([0-9]+)", argument.to_canonical_string())
        if match is None:
            raise ValueError("materialized color tensor has an unknown open index")
        axes.append(int(match.group(1)))
    if set(axes) != set(colored_legs):
        raise ValueError(
            "materialized color tensor open indices do not match its particles"
        )
    result: dict[tuple[int, ...], complex] = {}
    for coordinates in product(*(range(dimension) for dimension in tensor.shape)):
        by_leg = {leg: int(coordinates[axis]) for axis, leg in enumerate(axes)}
        key = tuple(by_leg[leg] for leg in colored_legs)
        result[key] = complex(tensor[coordinates])
    return result


def _as_expression(value: object) -> Expression:
    _ensure_symbolica()
    if isinstance(value, Expression):
        return value
    converter = getattr(value, "to_expression", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Expression):
            return converted
    raise TypeError(
        f"tensor value is not convertible to Expression: {type(value).__name__}"
    )


def _validate_color_representations(representations: Sequence[int]) -> None:
    for index, value in enumerate(representations, start=1):
        validate_color_representation(value, context=f"UFO color leg {index}")


__all__ = [
    "NormalizedTensorExpression",
    "classify_trilinear_color_expression",
    "normalize_color_expression",
    "normalize_lorentz_expression",
    "normalize_vertex_tensor_term",
    "project_trilinear_color_expression",
]
