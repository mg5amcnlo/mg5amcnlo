# SPDX-License-Identifier: 0BSD
"""Model-generic leading-color port wiring for certified local tensors.

The compiler in this module is deliberately bounded to trivalent recurrence
transitions and direct two-current closures in the representations currently
supported by the LC recurrence ABI.  It derives connectivity only from the
certified tensor family and oriented/tensor-role representations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

LCColorTensorFamily: TypeAlias = Literal[
    "singlet",
    "color-identity",
    "fundamental-generator",
    "adjoint-structure-constant",
    "direct-closure",
]
LCColorPortKind: TypeAlias = Literal["fundamental", "antifundamental"]

_SUPPORTED_FAMILIES = frozenset(
    {
        "singlet",
        "color-identity",
        "fundamental-generator",
        "adjoint-structure-constant",
        "direct-closure",
    }
)
_SUPPORTED_REPRESENTATIONS = frozenset({1, 3, -3, 8})


@dataclass(frozen=True, slots=True, order=True)
class ParentPortRef:
    """Reference to one ordered color-flow port of a parent current."""

    parent_index: int
    local_port_index: int

    def __post_init__(self) -> None:
        if isinstance(self.parent_index, bool) or not isinstance(
            self.parent_index, int
        ):
            raise TypeError("LC parent-port parent index must be an integer")
        if self.parent_index not in (0, 1):
            raise ValueError("LC parent-port parent index must be zero or one")
        if isinstance(self.local_port_index, bool) or not isinstance(
            self.local_port_index, int
        ):
            raise TypeError("LC parent-port local index must be an integer")
        if self.local_port_index < 0:
            raise ValueError("LC parent-port local index cannot be negative")


@dataclass(frozen=True, slots=True)
class ParentPortPairing:
    """An input-input contraction, ordered fundamental to antifundamental."""

    fundamental: ParentPortRef
    antifundamental: ParentPortRef

    def __post_init__(self) -> None:
        if not isinstance(self.fundamental, ParentPortRef) or not isinstance(
            self.antifundamental, ParentPortRef
        ):
            raise TypeError("LC parent-port pairings require immutable port refs")
        if self.fundamental.parent_index == self.antifundamental.parent_index:
            raise ValueError("LC parent-port pairings must join different parents")


@dataclass(frozen=True, slots=True)
class LCColorPortWiring:
    """One exact color-flow wiring term for a transition or closure.

    ``result_port_bindings[i]`` identifies the parent port exposed as ordered
    local port ``i`` of the result current.  Every parent port must occur once,
    either in one input-input pairing or in this result-binding tuple.
    """

    tensor_family: LCColorTensorFamily
    oriented_representations: tuple[int, ...]
    tensor_role_representations: tuple[int, ...]
    term_index: int
    exact_factor: int
    input_pairings: tuple[ParentPortPairing, ...]
    result_port_bindings: tuple[ParentPortRef, ...]

    def __post_init__(self) -> None:
        _validate_contract_representations(
            self.tensor_family,
            self.oriented_representations,
            self.tensor_role_representations,
        )
        if isinstance(self.term_index, bool) or not isinstance(self.term_index, int):
            raise TypeError("LC port-wiring term index must be an integer")
        if self.term_index < 0:
            raise ValueError("LC port-wiring term index cannot be negative")
        if self.exact_factor not in (-1, 1) or isinstance(self.exact_factor, bool):
            raise ValueError("LC port-wiring exact factor must be +1 or -1")
        if not isinstance(self.input_pairings, tuple) or any(
            not isinstance(pairing, ParentPortPairing)
            for pairing in self.input_pairings
        ):
            raise TypeError("LC input pairings must be an immutable pairing tuple")
        if not isinstance(self.result_port_bindings, tuple) or any(
            not isinstance(port, ParentPortRef) for port in self.result_port_bindings
        ):
            raise TypeError(
                "LC result-port bindings must be an immutable parent-port tuple"
            )

        parent_representations = self.oriented_representations[:2]
        expected_parent_ports = frozenset(
            ParentPortRef(parent_index, local_port_index)
            for parent_index, representation in enumerate(parent_representations)
            for local_port_index in range(len(_port_kinds(representation)))
        )
        consumed_ports: list[ParentPortRef] = []
        for pairing in self.input_pairings:
            fundamental_kind = _parent_port_kind(
                pairing.fundamental,
                parent_representations,
            )
            antifundamental_kind = _parent_port_kind(
                pairing.antifundamental,
                parent_representations,
            )
            if fundamental_kind != "fundamental":
                raise ValueError(
                    "LC pairing fundamental endpoint is not a fundamental port"
                )
            if antifundamental_kind != "antifundamental":
                raise ValueError(
                    "LC pairing antifundamental endpoint is not an antifundamental port"
                )
            consumed_ports.extend((pairing.fundamental, pairing.antifundamental))

        output_representation = self.output_representation
        output_port_kinds = (
            () if output_representation is None else _port_kinds(output_representation)
        )
        if len(self.result_port_bindings) != len(output_port_kinds):
            raise ValueError(
                "LC result-port binding count does not match the output representation"
            )
        for result_kind, parent_port in zip(
            output_port_kinds,
            self.result_port_bindings,
            strict=True,
        ):
            if _parent_port_kind(parent_port, parent_representations) != result_kind:
                raise ValueError(
                    "LC result port is bound to a parent port with the wrong "
                    "orientation"
                )
            consumed_ports.append(parent_port)

        if len(consumed_ports) != len(set(consumed_ports)):
            raise ValueError("LC port wiring consumes a parent port more than once")
        consumed_set = frozenset(consumed_ports)
        if consumed_set != expected_parent_ports:
            missing = tuple(sorted(expected_parent_ports - consumed_set))
            extra = tuple(sorted(consumed_set - expected_parent_ports))
            raise ValueError(
                "LC port wiring does not consume every parent port exactly once: "
                f"missing={missing!r}, extra={extra!r}"
            )

    @property
    def output_representation(self) -> int | None:
        """Return the current-shape output representation, if this is not a closure."""

        if self.tensor_family == "direct-closure":
            return None
        return self.oriented_representations[2]


def compile_lc_color_port_wirings(
    tensor_family: LCColorTensorFamily,
    oriented_representations: Sequence[int],
    tensor_role_representations: Sequence[int],
) -> tuple[LCColorPortWiring, ...]:
    """Compile exact LC parent/result port connectivity from a tensor contract.

    Trivalent tensor families return one term except for the adjoint structure
    constant, which returns its two ordered commutator terms with factors
    ``+1`` and ``-1``.  Direct closures consume two parent currents and expose
    no result ports.
    """

    oriented = _canonical_representations(
        oriented_representations,
        context="oriented",
    )
    tensor_roles = _canonical_representations(
        tensor_role_representations,
        context="tensor-role",
    )
    _validate_contract_representations(tensor_family, oriented, tensor_roles)

    def port(parent_index: int, local_port_index: int) -> ParentPortRef:
        return ParentPortRef(parent_index, local_port_index)

    def pair(
        fundamental: ParentPortRef,
        antifundamental: ParentPortRef,
    ) -> ParentPortPairing:
        return ParentPortPairing(fundamental, antifundamental)

    def wiring(
        *,
        term_index: int = 0,
        exact_factor: int = 1,
        input_pairings: tuple[ParentPortPairing, ...] = (),
        result_port_bindings: tuple[ParentPortRef, ...] = (),
    ) -> LCColorPortWiring:
        return LCColorPortWiring(
            tensor_family=tensor_family,
            oriented_representations=oriented,
            tensor_role_representations=tensor_roles,
            term_index=term_index,
            exact_factor=exact_factor,
            input_pairings=input_pairings,
            result_port_bindings=result_port_bindings,
        )

    if tensor_family == "singlet":
        return (wiring(),)

    if tensor_family == "color-identity":
        output = oriented[2]
        if output == 1:
            return (
                wiring(
                    input_pairings=_closure_pairings(oriented[:2]),
                ),
            )
        colored_parents = tuple(
            index
            for index, representation in enumerate(oriented[:2])
            if representation != 1
        )
        if len(colored_parents) != 1:
            raise ValueError(
                "color-identity transition with a colored output requires exactly "
                "one colored parent"
            )
        colored_parent = colored_parents[0]
        if oriented[colored_parent] != output:
            raise ValueError(
                "color-identity transition cannot change the colored current shape"
            )
        return (
            wiring(
                result_port_bindings=tuple(
                    port(colored_parent, local_port_index)
                    for local_port_index in range(len(_port_kinds(output)))
                )
            ),
        )

    if tensor_family == "fundamental-generator":
        output = oriented[2]
        parent_by_representation = {
            representation: index for index, representation in enumerate(oriented[:2])
        }
        if output == 8:
            fundamental_parent = parent_by_representation[3]
            antifundamental_parent = parent_by_representation[-3]
            return (
                wiring(
                    result_port_bindings=(
                        port(fundamental_parent, 0),
                        port(antifundamental_parent, 0),
                    )
                ),
            )
        adjoint_parent = parent_by_representation[8]
        if output == 3:
            fundamental_parent = parent_by_representation[3]
            return (
                wiring(
                    input_pairings=(
                        pair(port(fundamental_parent, 0), port(adjoint_parent, 1)),
                    ),
                    result_port_bindings=(port(adjoint_parent, 0),),
                ),
            )
        if output == -3:
            antifundamental_parent = parent_by_representation[-3]
            return (
                wiring(
                    input_pairings=(
                        pair(port(adjoint_parent, 0), port(antifundamental_parent, 0)),
                    ),
                    result_port_bindings=(port(adjoint_parent, 1),),
                ),
            )
        raise AssertionError("validated fundamental-generator output is unsupported")

    if tensor_family == "adjoint-structure-constant":
        return (
            wiring(
                term_index=0,
                input_pairings=(pair(port(0, 0), port(1, 1)),),
                result_port_bindings=(port(1, 0), port(0, 1)),
            ),
            wiring(
                term_index=1,
                exact_factor=-1,
                input_pairings=(pair(port(1, 0), port(0, 1)),),
                result_port_bindings=(port(0, 0), port(1, 1)),
            ),
        )

    if tensor_family == "direct-closure":
        return (wiring(input_pairings=_closure_pairings(oriented)),)

    raise AssertionError("validated LC tensor family is unsupported")


def _canonical_representations(
    values: Sequence[int],
    *,
    context: str,
) -> tuple[int, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(f"LC {context} representations must be a sequence")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"LC {context} representations must contain integers")
        if value not in _SUPPORTED_REPRESENTATIONS:
            raise ValueError(
                f"LC {context} representation {value} is unsupported; expected "
                "one of -3, 1, 3, or 8"
            )
        result.append(value)
    return tuple(result)


def _validate_contract_representations(
    tensor_family: str,
    oriented_representations: tuple[int, ...],
    tensor_role_representations: tuple[int, ...],
) -> None:
    if not isinstance(tensor_family, str):
        raise TypeError("LC tensor family must be a string")
    if tensor_family not in _SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported certified LC tensor family {tensor_family!r}")
    if not isinstance(oriented_representations, tuple) or not isinstance(
        tensor_role_representations, tuple
    ):
        raise TypeError("LC port-wiring representations must be immutable tuples")
    expected_arity = 2 if tensor_family == "direct-closure" else 3
    if (
        len(oriented_representations) != expected_arity
        or len(tensor_role_representations) != expected_arity
    ):
        raise ValueError(
            f"LC {tensor_family} contract requires {expected_arity} representations"
        )
    for context, representations in (
        ("oriented", oriented_representations),
        ("tensor-role", tensor_role_representations),
    ):
        for representation in representations:
            if isinstance(representation, bool) or not isinstance(representation, int):
                raise TypeError(f"LC {context} representations must contain integers")
            if representation not in _SUPPORTED_REPRESENTATIONS:
                raise ValueError(
                    f"LC {context} representation {representation} is unsupported; "
                    "expected one of -3, 1, 3, or 8"
                )
    if tuple(abs(value) for value in oriented_representations) != tuple(
        abs(value) for value in tensor_role_representations
    ):
        raise ValueError(
            "LC oriented and tensor-role representations describe different shapes"
        )

    if tensor_family == "direct-closure":
        if oriented_representations != tensor_role_representations:
            raise ValueError(
                "direct LC closure tensor roles must equal the parent current shapes"
            )
        _validate_closure_representations(oriented_representations)
        return

    if oriented_representations[:2] != tensor_role_representations[:2]:
        raise ValueError(
            "LC parent tensor roles must equal the oriented parent current shapes"
        )
    if tensor_role_representations[2] != _dual(oriented_representations[2]):
        raise ValueError(
            "LC output tensor role must be dual to the oriented result current shape"
        )

    tensor_roles = tensor_role_representations
    if tensor_family == "singlet":
        if tensor_roles != (1, 1, 1):
            raise ValueError("singlet LC tensor requires representations (1, 1, 1)")
        return
    if tensor_family == "color-identity":
        role_counts = {
            representation: tensor_roles.count(representation)
            for representation in _SUPPORTED_REPRESENTATIONS
        }
        fundamental_identity = (
            role_counts[3] == 1
            and role_counts[-3] == 1
            and role_counts[1] == 1
            and role_counts[8] == 0
        )
        adjoint_identity = role_counts[8] == 2 and role_counts[1] == 1
        singlet_identity = role_counts[1] == 3
        if not (fundamental_identity or adjoint_identity or singlet_identity):
            raise ValueError(
                "color-identity LC tensor requires one singlet plus a dual "
                "fundamental pair or two adjoints"
            )
        return
    if tensor_family == "fundamental-generator":
        if sorted(tensor_roles) != [-3, 3, 8]:
            raise ValueError(
                "fundamental-generator LC tensor requires roles (-3, 3, 8)"
            )
        return
    if tensor_family == "adjoint-structure-constant" and tensor_roles != (8, 8, 8):
        raise ValueError(
            "adjoint-structure-constant LC tensor requires representations (8, 8, 8)"
        )


def _validate_closure_representations(representations: tuple[int, ...]) -> None:
    if representations == (1, 1) or representations == (8, 8):
        return
    if set(representations) == {3, -3}:
        return
    raise ValueError(
        "direct LC closure requires two singlets, a fundamental/antifundamental "
        "pair, or two adjoints"
    )


def _closure_pairings(
    representations: tuple[int, ...],
) -> tuple[ParentPortPairing, ...]:
    _validate_closure_representations(representations)
    if representations == (1, 1):
        return ()
    if set(representations) == {3, -3}:
        fundamental_parent = representations.index(3)
        antifundamental_parent = representations.index(-3)
        return (
            ParentPortPairing(
                ParentPortRef(fundamental_parent, 0),
                ParentPortRef(antifundamental_parent, 0),
            ),
        )
    return (
        ParentPortPairing(ParentPortRef(0, 0), ParentPortRef(1, 1)),
        ParentPortPairing(ParentPortRef(1, 0), ParentPortRef(0, 1)),
    )


def _parent_port_kind(
    port: ParentPortRef,
    parent_representations: tuple[int, ...],
) -> LCColorPortKind:
    if port.parent_index >= len(parent_representations):
        raise ValueError("LC parent-port reference addresses a missing parent")
    port_kinds = _port_kinds(parent_representations[port.parent_index])
    if port.local_port_index >= len(port_kinds):
        raise ValueError("LC parent-port reference addresses a missing local port")
    return port_kinds[port.local_port_index]


def _port_kinds(representation: int) -> tuple[LCColorPortKind, ...]:
    if representation == 1:
        return ()
    if representation == 3:
        return ("fundamental",)
    if representation == -3:
        return ("antifundamental",)
    if representation == 8:
        return ("fundamental", "antifundamental")
    raise ValueError(f"unsupported LC color representation {representation}")


def _dual(representation: int) -> int:
    return -representation if abs(representation) == 3 else representation


__all__ = [
    "LCColorPortWiring",
    "LCColorTensorFamily",
    "ParentPortPairing",
    "ParentPortRef",
    "compile_lc_color_port_wirings",
]
