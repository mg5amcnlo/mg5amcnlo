# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import pytest

from pyamplicol.models.lc_color_port_wiring import (
    LCColorPortWiring,
    ParentPortPairing,
    ParentPortRef,
    compile_lc_color_port_wirings,
)


def _port(parent: int, local: int) -> ParentPortRef:
    return ParentPortRef(parent, local)


def _pair(
    fundamental: ParentPortRef,
    antifundamental: ParentPortRef,
) -> ParentPortPairing:
    return ParentPortPairing(fundamental, antifundamental)


def test_fundamental_pair_exposes_ordered_adjoint_result_ports() -> None:
    (wiring,) = compile_lc_color_port_wirings(
        "fundamental-generator",
        (3, -3, 8),
        (3, -3, 8),
    )

    assert wiring.input_pairings == ()
    assert wiring.result_port_bindings == (_port(0, 0), _port(1, 0))
    assert wiring.output_representation == 8


@pytest.mark.parametrize(
    ("oriented", "tensor_roles", "pairing", "result_binding"),
    (
        (
            (3, 8, 3),
            (3, 8, -3),
            _pair(_port(0, 0), _port(1, 1)),
            _port(1, 0),
        ),
        (
            (-3, 8, -3),
            (-3, 8, 3),
            _pair(_port(1, 0), _port(0, 0)),
            _port(1, 1),
        ),
    ),
)
def test_fundamental_generator_respects_result_current_shape_crossing(
    oriented: tuple[int, int, int],
    tensor_roles: tuple[int, int, int],
    pairing: ParentPortPairing,
    result_binding: ParentPortRef,
) -> None:
    (wiring,) = compile_lc_color_port_wirings(
        "fundamental-generator",
        oriented,
        tensor_roles,
    )

    assert wiring.input_pairings == (pairing,)
    assert wiring.result_port_bindings == (result_binding,)


def test_structure_constant_emits_both_exact_commutator_wirings() -> None:
    terms = compile_lc_color_port_wirings(
        "adjoint-structure-constant",
        (8, 8, 8),
        (8, 8, 8),
    )

    assert tuple(term.exact_factor for term in terms) == (1, -1)
    assert tuple(term.term_index for term in terms) == (0, 1)
    assert terms[0].input_pairings == (_pair(_port(0, 0), _port(1, 1)),)
    assert terms[0].result_port_bindings == (_port(1, 0), _port(0, 1))
    assert terms[1].input_pairings == (_pair(_port(1, 0), _port(0, 1)),)
    assert terms[1].result_port_bindings == (_port(0, 0), _port(1, 1))


@pytest.mark.parametrize(
    ("representations", "expected_pairings"),
    (
        (
            (3, -3),
            (_pair(_port(0, 0), _port(1, 0)),),
        ),
        (
            (-3, 3),
            (_pair(_port(1, 0), _port(0, 0)),),
        ),
        (
            (8, 8),
            (
                _pair(_port(0, 0), _port(1, 1)),
                _pair(_port(1, 0), _port(0, 1)),
            ),
        ),
    ),
)
def test_direct_closure_consumes_q_qbar_and_gluon_ports(
    representations: tuple[int, int],
    expected_pairings: tuple[ParentPortPairing, ...],
) -> None:
    (wiring,) = compile_lc_color_port_wirings(
        "direct-closure",
        representations,
        representations,
    )

    assert wiring.input_pairings == expected_pairings
    assert wiring.result_port_bindings == ()
    assert wiring.output_representation is None


def test_singlet_and_identity_families_have_bounded_exact_wiring() -> None:
    (singlet,) = compile_lc_color_port_wirings(
        "singlet",
        (1, 1, 1),
        (1, 1, 1),
    )
    (identity_copy,) = compile_lc_color_port_wirings(
        "color-identity",
        (1, 3, 3),
        (1, 3, -3),
    )
    (identity_close,) = compile_lc_color_port_wirings(
        "color-identity",
        (8, 8, 1),
        (8, 8, 1),
    )

    assert singlet.input_pairings == singlet.result_port_bindings == ()
    assert identity_copy.input_pairings == ()
    assert identity_copy.result_port_bindings == (_port(1, 0),)
    assert identity_close.input_pairings == (
        _pair(_port(0, 0), _port(1, 1)),
        _pair(_port(1, 0), _port(0, 1)),
    )
    assert identity_close.result_port_bindings == ()


@pytest.mark.parametrize(
    ("family", "oriented", "tensor_roles", "message"),
    (
        (
            "fundamental-generator",
            (3, -3, 6),
            (3, -3, 6),
            "representation 6 is unsupported",
        ),
        (
            "fundamental-generator",
            (3, 8, 3),
            (3, 8, 3),
            "output tensor role must be dual",
        ),
        (
            "color-identity",
            (3, 8, 3),
            (3, 8, -3),
            "color-identity LC tensor requires",
        ),
        (
            "direct-closure",
            (3, 3),
            (3, 3),
            "direct LC closure requires",
        ),
        (
            "singlet",
            (1, 1),
            (1, 1),
            "requires 3 representations",
        ),
    ),
)
def test_malformed_or_unsupported_tensor_contracts_fail_closed(
    family: str,
    oriented: tuple[int, ...],
    tensor_roles: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compile_lc_color_port_wirings(  # type: ignore[arg-type]
            family,
            oriented,
            tensor_roles,
        )


def test_wiring_record_rejects_duplicate_or_unconsumed_parent_ports() -> None:
    with pytest.raises(ValueError, match="more than once"):
        LCColorPortWiring(
            tensor_family="fundamental-generator",
            oriented_representations=(3, 8, 3),
            tensor_role_representations=(3, 8, -3),
            term_index=0,
            exact_factor=1,
            input_pairings=(
                ParentPortPairing(_port(0, 0), _port(1, 1)),
            ),
            result_port_bindings=(_port(0, 0),),
        )


def test_port_wiring_is_model_neutral_and_deterministic() -> None:
    # Built-in and UFO compilation provide the same certified tensor family and
    # representation roles.  No model identity participates in this compiler.
    built_in_contract = compile_lc_color_port_wirings(
        "fundamental-generator",
        (3, 8, 3),
        (3, 8, -3),
    )
    ufo_contract = compile_lc_color_port_wirings(
        "fundamental-generator",
        [3, 8, 3],
        [3, 8, -3],
    )

    assert built_in_contract == ufo_contract
    assert hash(built_in_contract) == hash(ufo_contract)
