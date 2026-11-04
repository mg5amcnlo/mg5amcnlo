# SPDX-License-Identifier: 0BSD
import json

import pytest

from pyamplicol import (
    ColorCorrelator,
    CorrelatedRequest,
    CorrelatorConfig,
    EmitGluon,
    SplitGluon,
)
from pyamplicol.cli.parser import parse_cli


def test_correlated_request_defaults_and_physical_override():
    assert CorrelatedRequest().color_correlation == "born"
    assert CorrelatedRequest().spin_vectors is None
    assert CorrelatedRequest("T12", {}).spin_vectors == {}


@pytest.mark.parametrize("identifier", ("", None, 3))
def test_correlated_request_requires_colour_id(identifier):
    with pytest.raises(ValueError, match="correlation ID"):
        CorrelatedRequest(identifier)


def test_correlated_request_requires_vector_mapping():
    with pytest.raises(TypeError, match="must map"):
        CorrelatedRequest(spin_vectors=[(0, 1, 0, 0)])


def test_declarations_json_roundtrip_and_automatic_born():
    chain = (EmitGluon(1, -1), SplitGluon(-1, -2, -3), EmitGluon(-2, -4))
    declarations = CorrelatorConfig(
        (ColorCorrelator.dipole("T12", 1, 2), ColorCorrelator("N3LO", chain, chain)),
        ((3, 1), (1,)),
    )
    encoded = json.loads(json.dumps(declarations.to_json_dict()))
    assert CorrelatorConfig.from_json_dict(encoded) == declarations
    assert declarations.spin_legs == (1, 3)
    assert declarations.spin_correlations == ((1,), (1, 3))
    assert [item.id for item in declarations.color_requests] == ["born", "T12", "N3LO"]


@pytest.mark.parametrize("order", (1, 2, 3, 4, 7))
def test_automatic_catalogue_declaration_roundtrip(order):
    config = CorrelatorConfig.all_color(
        through_order=order, spin_correlations=((3,),)
    )
    assert config.all_color_through_order == order
    assert config.spin_correlations == ((3,),)
    assert CorrelatorConfig.from_json_dict(config.to_json_dict()) == config
    with pytest.raises(ValueError, match="resolved for a process"):
        _ = config.color_requests


@pytest.mark.parametrize("order", (0, -1, True, 1.0, "NLO", None))
def test_automatic_catalogue_rejects_unsupported_orders(order):
    with pytest.raises(ValueError, match="all_color_through_order"):
        CorrelatorConfig.all_color(through_order=order)


def test_automatic_catalogue_is_resolved_per_process_without_mutating_config():
    from pyamplicol.color.connections import ColorLeg

    config = CorrelatorConfig.all_color(through_order=1)
    first = config._resolve_color((ColorLeg(1, 1), ColorLeg(2, 3), ColorLeg(3, -3)))
    second = config._resolve_color((ColorLeg(1, 8), ColorLeg(2, 8), ColorLeg(3, 8)))
    assert len(first.color_requests) == 5  # born and four directed dipoles
    assert len(second.color_requests) == 10
    assert config.all_color_through_order == 1
    assert first.all_color_through_order is second.all_color_through_order is None
    assert {item.bra[0].emitter_label for item in first.color_correlations} == {2, 3}
    assert first._resolve_color(()) is first
    assert CorrelatorConfig.from_json_dict(first.to_json_dict()) == first


def test_explicit_and_automatic_requests_can_be_combined_but_not_id_collisions():
    from pyamplicol.color.connections import ColorLeg

    legs = (ColorLeg(1, 8),)
    config = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("my-Casimir", 1, 1),),
        all_color_through_order=1,
    )
    resolved = config._resolve_color(legs)
    assert len(resolved.color_requests) == 3
    with pytest.raises(ValueError, match="catalogue IDs are reserved"):
        CorrelatorConfig(
            color_correlations=(resolved.color_correlations[1],),
            all_color_through_order=1,
        )


@pytest.mark.parametrize(
    "groups", (((),), ((0,),), ((True,),), ((1, 1),), ((1, 2), (2, 1)))
)
def test_invalid_spin_classes(groups):
    with pytest.raises(ValueError):
        CorrelatorConfig(spin_correlations=groups)


def test_duplicate_or_reserved_ids_rejected():
    with pytest.raises(ValueError, match="reserved"):
        CorrelatorConfig((ColorCorrelator("born"),))
    with pytest.raises(ValueError, match="unique"):
        CorrelatorConfig((ColorCorrelator("a"), ColorCorrelator("a")))


def test_unknown_json_fields_and_step_kinds_rejected():
    with pytest.raises(ValueError, match="unknown"):
        CorrelatorConfig.from_json_dict({"colour_correlations": []})
    with pytest.raises(ValueError, match="unknown"):
        CorrelatorConfig.from_json_dict(
            {"color_correlations": [{"id": "a", "bra": [{"kind": "unknown"}]}]}
        )


def test_cli_correlator_path_is_separate_from_default_generation_identity():
    plain = parse_cli(["generate", "g g > g g", "output"])
    selected = parse_cli(
        ["generate", "g g > g g", "output", "--correlators", "correlators.json"]
    )
    assert plain.correlators is None
    assert str(selected.correlators) == "correlators.json"
    assert dict(plain.dedicated) == dict(selected.dedicated)
    assert plain.resolve().effective == selected.resolve().effective
