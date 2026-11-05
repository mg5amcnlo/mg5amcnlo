# SPDX-License-Identifier: 0BSD
"""Generation-time exterior-algebra signs for external fermions."""

from __future__ import annotations

import pytest

from pyamplicol.generation.fermion_ordering import (
    _external_fermion_requires_exterior_sign,
    _external_fermion_support_sign,
    _fermion_input_orientation_sign,
    _fermion_rep_requires_exterior_sign,
)


@pytest.mark.parametrize(
    ("statistics", "color_role", "expected"),
    (
        ("fermion", "singlet", True),
        ("fermion", "fundamental", False),
        ("fermion", "antifundamental", False),
        ("boson", "adjoint", False),
    ),
)
def test_external_source_sign_ownership_uses_canonical_color_role(
    statistics: str,
    color_role: str,
    expected: bool,
) -> None:
    assert _external_fermion_requires_exterior_sign(statistics, color_role) is expected


@pytest.mark.parametrize("color_role", ("adjoint", "inclusive"))
def test_unsupported_external_fermion_color_role_fails_closed(
    color_role: str,
) -> None:
    with pytest.raises(ValueError, match="fermion color role"):
        _external_fermion_requires_exterior_sign("fermion", color_role)


@pytest.mark.parametrize(
    ("color_representation", "expected"),
    ((1, True), (3, False), (-3, False)),
)
def test_model_representation_sign_ownership_is_generic(
    color_representation: int,
    expected: bool,
) -> None:
    assert _fermion_rep_requires_exterior_sign(color_representation) is expected


def test_unsupported_fermion_representation_fails_closed() -> None:
    with pytest.raises(ValueError, match="fermion color representation"):
        _fermion_rep_requires_exterior_sign(8)


def test_support_parity_uses_canonical_source_rank_not_label_value() -> None:
    source_ranks = {40: 0, 7: 1, 99: 2, 3: 3}

    sign = _external_fermion_support_sign(
        (3, 7),
        (40, 99),
        source_ranks,
    )

    assert sign == -1


@pytest.mark.parametrize(
    (
        "left_is_fermion",
        "left_orientation",
        "right_is_fermion",
        "right_orientation",
        "expected",
    ),
    (
        (True, "antiparticle", True, "particle", 1),
        (True, "particle", True, "antiparticle", -1),
        (True, "particle", False, "self-conjugate", 1),
        (False, "self-conjugate", True, "antiparticle", 1),
    ),
)
def test_two_fermion_kernel_orientation_is_a_cached_sign(
    left_is_fermion: bool,
    left_orientation: str,
    right_is_fermion: bool,
    right_orientation: str,
    expected: int,
) -> None:
    assert (
        _fermion_input_orientation_sign(
            left_is_fermion,
            left_orientation,
            True,
            right_is_fermion,
            right_orientation,
            True,
        )
        == expected
    )


def test_colored_kernel_keeps_authenticated_open_line_sign() -> None:
    assert (
        _fermion_input_orientation_sign(
            True,
            "particle",
            False,
            True,
            "antiparticle",
            False,
        )
        == 1
    )


def test_mixed_kernel_sign_ownership_fails_closed() -> None:
    with pytest.raises(ValueError, match="mixed color-sign ownership"):
        _fermion_input_orientation_sign(
            True,
            "particle",
            True,
            True,
            "antiparticle",
            False,
        )


def test_unsupported_self_conjugate_fermion_fails_during_generation() -> None:
    with pytest.raises(ValueError, match="non-self-conjugate"):
        _fermion_input_orientation_sign(
            True,
            "self-conjugate",
            True,
            True,
            "particle",
            True,
        )


@pytest.mark.parametrize("orientation", ("particle", "antiparticle"))
def test_same_dirac_orientation_fails_during_generation(orientation: str) -> None:
    with pytest.raises(ValueError, match="opposite Dirac orientations"):
        _fermion_input_orientation_sign(
            True,
            orientation,
            True,
            True,
            orientation,
            True,
        )


def test_identical_dirac_pairings_have_opposite_tree_products() -> None:
    source_ranks = {10: 0, 20: 1, 30: 2, 40: 3}

    direct = _external_fermion_support_sign((10,), (20,), source_ranks)
    direct *= _fermion_input_orientation_sign(
        True,
        "antiparticle",
        True,
        True,
        "particle",
        True,
    )
    direct *= _external_fermion_support_sign((10, 20), (30,), source_ranks)

    exchange = _external_fermion_support_sign((20,), (30,), source_ranks)
    exchange *= _fermion_input_orientation_sign(
        True,
        "particle",
        True,
        True,
        "antiparticle",
        True,
    )
    exchange *= _external_fermion_support_sign((10,), (20, 30), source_ranks)

    assert direct == 1
    assert exchange == -1
